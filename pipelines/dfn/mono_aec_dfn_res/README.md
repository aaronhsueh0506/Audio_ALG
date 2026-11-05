# Mono AEC + RES + DeepFilterNet2

This example replaces the MMSE-LSA denoiser of the conventional mono
chain with DeepFilterNet2 while keeping the linear AEC, the AEC3 residual echo
suppressor and the comfort-noise recipe. There is no mono unit and no DFN wrapper
library: `main.c` initialises lib/aec's `Aec` (which carries the RES gain itself), the gain
and spectrum buffers of the RES path (no denoiser) and the DFN2
stage (`dfn/dfn_res_stage.h`, or `dfn/dfn_rate_bridge.h` below 48 kHz) separately, forms P in
`rig_process()`, and `dfn/dfn_hop.h` runs the network between the post-RES spectrum and the
synthesis. `dfn/dfn_hop_pool.h` sizes and carves the DFN tail of the pool (the stage or the bridge,
its synthesis and the output spectrum):

```text
mic/ref -> aec_process_context(aec, mic, ref)
           follow_post_reset()                     new aec_post_reset_serial(): rng = AEC_CNG_SEED (+ the host-synthesis OLA clear, see below)
           aec_get_res_context(aec, &ctx)          G_res = ctx.res_gain, comfort-noise PSD
           aec_get_output_base(aec, &base)         B = base.base_spec = the rendering base (E, or E + echo on a guard hop)
                                                   -> DFN2 feature input
           P = B * min(1, ctx.res_gain)            P = B * G_res (+ comfort noise) -> DFN2 applied here
                                                   (sk_min_f32 + sk_capply_gain_f32 + aec_cng_fill)
         -> dfn_hop_begin(&hop, B, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-L (L = M + D, 2 for the shipped pair),
                                                       sqrt-Hann iFFT / OLA
```

`dfn_hop_process(&hop, B, P, out)` runs the same sequence with the model callback; P must
be formed before `dfn_hop_begin()`. A refused hop is the integrator's to recover: reset the AEC,
the comfort-noise generator (`rng = AEC_CNG_SEED`, the library's constant) and the `DfnHop` together
(`rig_reset()` in `main.c`).

**Two spectra.** `ctx.error_spec` / `ctx.formed_hop` are the AEC's analysis seam: the selected linear residual, never
altered by the output guard. B (`aec_get_output_base()`) is what `aec_process()` itself would multiply by its gain:
E, or the capture spectrum E + echo on a hop where the linear estimate is unusable and E is louder than the
capture. B is E bit for bit on every hop the guard leaves alone. The stage reads the pair (B, P): its feature input
and the spectrum it filters stay one signal, so a guard hop never pairs a capture-based P with a residual-based
estimate. There is no denoiser here, so no further estimator reads either spectrum.

**Hard post-chain reset event.** The AEC moves `aec_post_reset_serial()` when its post chain is hard-reset inside a
processing call (a confirmed delay change that restarts the filter, a first acquisition without warm
transfer) or by `aec_reset()`; warm-transfer and soft realignments and the bad-relock soft revert do not move it. `follow_post_reset()` keeps the
last handled serial. On a new value the comfort noise restarts from `AEC_CNG_SEED` before that hop's first draw. The
stage never resets on it and the program never resets the AEC in response. At 48 kHz the program owns the synthesis
(`AcIstft`) and the stage delays every frame by `dfn_res_stage_lookahead_frames()` hops, so the overlap-add tail is
dropped when the frame built on the reset hop reaches the synthesis (a delay line of that many hops, not at the
reset hop: that would cut the previous filter's last frame out of the output). Below 48 kHz the host-grid synthesis
lives inside `dfn_rate_bridge` and runs at the top of its `begin()`, with nothing between the spectra and it, so the
call is `dfn_rate_bridge_reset_synthesis()` on the hop of the event, before the hop enters the bridge: no delay line.
It clears the overlap-add tail of the two native-grid synthesis objects (estimate and apply lane) and nothing else;
DFN2 (no reset on an AEC event), both resamplers, the 48 kHz FIFOs, analysis histories and synthesis, and the output
FIFO with its prefill keep running, so the bridge's delay does not change.

The network estimates from the pre-RES spectrum because it was trained on
unprocessed noisy speech and must not see the spectral holes RES cuts; it is
applied to the post-RES spectrum so far-end residue that looks like speech is
preserved by the network and left to RES, and non-speech is suppressed by
both. The host's comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins where it is largest, which breaks the estimate/apply phase
consistency the complex deep filter relies on. Turn it on only for a
subjective far-end-only / silence A/B; when on it travels through the network like every other component of P
and `atten_lim_db` bounds how far any bin can be attenuated. The dual-input
topology with the full deep filter is checked structurally by the C gates;
its quality against a mask-only compose was evaluated by an offline driver
that is now archived outside the repository (see the cleanup record).

With no model callback (or a failing one) the output is the conventional
chain with the denoiser off delayed: bit for bit by L = M + D hops at 48 kHz (two for the shipped
lookahead pair; the acceptance gate in `tests/test_mono_dfn.c`, compared with the hand-written chain of
`pipelines/mono_aec_nr_res/tests/conv_chain_mono.h`, asserted with
comfort noise off and on, across a real matched-delay re-lock, with the
suppressor demonstrably active, and on streams that fire the output guard and hard-reset the post chain, with
the reset event delayed by L hops), the post spectrum P itself equals `aec_process()` of a full-mode `Aec`
after the host's own synthesis with the AEC's window table, bit for bit, at 16 kHz/256, 16 kHz/512 and 48 kHz, and through the resampler round trip by
the bridge delay on a native grid (the same test pins the delay with a
near-end impulse, the frame schedule, and, where the delay is a whole number of native samples, the pass-band
error against the conventional 16/8 kHz output, observed at about 1e-5 against a gate of 1e-3 for the shipped pair,
on streams that carry hard-reset events with the bridge's synthesis cleared on the hop of the event). The bridge's own
native-grid synthesis (`dfn_rate_bridge_get_host_hops()`) equals the hand-written chain bit for bit at 16 kHz/256,
16 kHz/512 and 8 kHz/256 on streams that fire the guard and reset the post chain, and stays within the window rounding
of `aec_process()` (audiolib's default window against the AEC's table; observed below 1e-7); a bridge that ignores
the event or clears one hop late fails both gates.

The host runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 and 8 kHz/256 as the host offers them, or 48 kHz. As shipped, only
the DFN2 stage runs at 48 kHz (the model grid is chosen at run time with `--ai-sample-rate` / `--ai-fft-size`, see "DFN2 grid" below). At 48 kHz the host's spectra are DFN2's grid
and the stage is applied directly (algorithmic latency L + 1 hops: L model
lookahead hops plus one synthesis-WOLA hop; three hops, 32 ms, for the shipped pair). Below 48 kHz the rate
bridge (`pipelines/dfn/dfn_rate_bridge.h`) synthesises E and P on the native
grid, resamples them up with `audiolib`'s polyphase FIR, runs the stage
on the 48 kHz 1024/512 grid every 512 new samples, and resamples the result
down; the AEC, RES and comfort noise never leave the product rate. The
bridge adds a fixed delay to the host's own output, `prefill + 32 + (L + 1) x 512 x sr / 48000` native samples
(integer division): for the shipped pair 672 samples at 16 kHz
with hop 128 (42 ms), 629 at 16 kHz with hop 256 (39 ms), 330 at 8 kHz
(41 ms); the total algorithmic latency is that plus the host's WOLA hop
(16 kHz, hop 128: 800 samples, 50 ms). For a pair whose L + 1 is not a multiple of 3 the stage delay is a
fraction of a native sample and is floored (pair 0/1: 501 / 458 / 244 samples). Per native hop the stage runs a fixed
schedule (16 kHz/hop 128: 0,1,1,1 frames; 16 kHz/hop 256 and 8 kHz: 1,2), so
two inferences in one hop is a real per-hop compute fact. The Python reference
`pipelines/dfn/mono_aec_dfn_res/pipeline.py` mirrors both paths (16 and 48 kHz
hosts); its stage reads the pair from the model.

DFN2 grid. Neither the DFN2 library nor the pipeline units hold a grid (see [`AINR/DeepFilterNet2/README.md`](../../../AINR/DeepFilterNet2/README.md),
"Choosing the grid"): `main.c` builds the model grid at run time from `--ai-sample-rate <Hz>` / `--ai-fft-size <n>`
(default 48000 / 1024) with `dfn2_grid_make()`, using the three-row table of `dfn/dfn_example_args.h`
(16000/256 -> n_erb 24, df_bins 96, lookahead 0/0; 32000/512 -> 28, 96, 0/0, provisional; 48000/1024 -> 32, 96, 1/1; any other
pair is refused with the list, exit 2) and hands it to `DfnResStageConfig.grid`. The stage takes its rate, FFT size and its input and
output scales (1/sqrt(n_fft) and its inverse) from that grid. A host at the model's rate needs no bridge:
`--ai-sample-rate 16000 --ai-fft-size 256` with the 16 kHz host runs straight through the stage (hop 128, algorithmic latency 128
samples for the 0/0 row, pool 594,048 B: the library's state is sized for its capacity, not for the grid), against 800 samples and
718,096 B for the shipped 48 kHz model with the same 16 kHz host through the bridge. The bridge exists only for the shipped
48 kHz / 1024 model (`resolve()` returns -1 on any other stage grid) and its hosts stay 8 kHz and 16 kHz, so a 16 or 32 kHz model
refuses a host of another rate ("pool query rejected", exit 1), and the 48 kHz host needs the 48 kHz model. The banner prints the model
grid and whether the stage or the bridge runs. `tests/test_mono_dfn.c` and the Python reference cover the shipped grid; the stage test
(`dfn/tests/test_dfn_res_stage.c`) runs all three table grids.

Lookahead pair. The DFN2 mask and deep-filter lookaheads M (0..2) and D (0..4) are two fields of the model grid
(a row of the table above, not a build setting) and must equal the pair the model was trained for;
`dfn_res_stage_lookahead_frames(grid)` = M + D. The make variables `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` choose only the pair
the test programs run (and, for a ring deeper than 5, the library's ring capacity), never the pair `main.c` runs. The rules and
`make test-pairs` are in [`../README.md`](../README.md). Only the shipped 1/1 pair has a trained checkpoint, and the numbers in this
file are for it unless a pair is named.

Build and test from `pipelines/` (the Makefile that owns the rules):

```sh
make WERROR=1 libaudio_pipeline_dfn.a mono_aec_dfn_res
make WERROR=1 test        # includes test_dfn_res_stage, test_dfn_rate_bridge,
                          # test_dfn_hop, test_mono_dfn and the skeleton
```

The `Makefile` in this directory is a thin forwarder: `make -C pipelines/dfn/mono_aec_dfn_res test`
builds and runs `test_dfn_res_stage`, `test_dfn_rate_bridge` and `test_mono_dfn`, then the
skeleton (default, `--delay-num-filters 1 --cng --atten-lim 12`, `--sample-rate 48000`,
`--sample-rate 8000`, `--ai-sample-rate 16000 --ai-fft-size 256`) and checks that malformed numbers such as `16000Hz` and `12dB`, and a model grid outside the table, exit with 2, and that a model grid the bridge cannot serve is a rejected pool query (exit 1).

`libaudio_pipeline_dfn.a` holds the shared stage, the rate bridge and the DFN2
host pre/post objects. The example links it with `libaec` and
`libaudio_common` (no denoiser, so no `libanr`); the board supplies the stateless model callback
(`DFN2Model`) and the exported ERB matrices (`erb_fwd.bin` / `erb_inv.bin`
from `AINR/DeepFilterNet2/export_erb_matrix.py`). `main.c` is the board
skeleton: delay-profile flags in the `aec_wav` spelling, `--erb-fwd` /
`--erb-inv`, `--sample-rate` (default 16000), `--ai-sample-rate` / `--ai-fft-size` (the model grid, default 48000 / 1024), `--atten-lim`, `--cng`, and a fail-open `run_accelerator()`
stub to replace with the runtime call. Numbers are parsed by `common/example_cli.h`
(`example_cli_parse_int` / `example_cli_parse_float`): empty strings, leading spaces, trailing characters,
out-of-range values, inf and nan are rejected with exit code 2; plain decimals and a leading `+`
are accepted. It publishes the model I/O descriptor
of the geometry it was compiled against; a product fills that descriptor from
its model's exported metadata instead, which is what makes a mismatched graph
detectable.

## Calling sequence and parameters

```
App                 Aec           DfnHop + DfnResStage | DfnRateBridge      AcIstft (DFN2 grid rate; 48 kHz as shipped)
 |-- aec_resolve_signal_grid; aec_config_from_level; aec_get_mem_size / aec_init -->|
 |-- dfn_res_stage_get_mem_requirements / dfn_res_stage_init_ex            (host at the DFN2 rate: 48 kHz as shipped)
 |   or dfn_rate_bridge_get_mem_requirements / dfn_rate_bridge_init_ex     (8 / 16 kHz host)
 |-- dfn_hop_init_stage(...) or dfn_hop_init_bridge(...)
 |   per hop: aec_process_context --> follow_post_reset --> aec_get_res_context / aec_get_output_base
 |            --> P = B * min(1, res_gain) (+ comfort noise, off by default)
 |            --> dfn_hop_begin(B, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused hop: aec_reset + comfort-noise rng = AEC_CNG_SEED + dfn_hop_reset   (rig_reset())
 |   never reset the stage on an AEC delay change
 |-- aec_destroy; dfn_res_stage_destroy | dfn_rate_bridge_destroy; release the blocks
```

External parameters (board-skeleton command line, `dfn/dfn_example_args.h`): `--sample-rate` (8000 / 16000 default
/ 48000), `--ai-sample-rate` / `--ai-fft-size` (the DFN2 model grid, default 48000 / 1024), the delay profile `--delay-mode matched|fixed|external` with `--delay-num-filters` (1..5) and
`--fixed-delay`, `--erb-fwd` / `--erb-inv` (the exported ERB matrices; without them the skeleton runs a
fail-open smoke only), `--atten-lim` (dB, 0 = unlimited; `dfn_res_stage_set_atten_lim()` between hops),
`--cng` (comfort noise, off by default). An option followed by another `--` option, or by the end of the line, has no value and the
program exits with code 2; a file named `--x` is written `./--x`. Strength of the RES gain: `aec_set_level()`. Internal: the
stage grid (the model grid of the config; 48 kHz / 1024 / 512 as shipped), the grid's `stft_scale` / `stft_unscale` scales (2^-5 / 2^5 on the shipped grid), the rate
bridge's FIR resamplers and frame schedule, the AEC and DFN2 model constants. Public functions are the
`dfn_res_stage_*`, `dfn_rate_bridge_*` of `dfn_res_stage.h` / `dfn_rate_bridge.h` and the
`dfn_hop_*` of `dfn/dfn_hop.h`; the stage and the bridge carry 32-byte descriptors
(`DFN_RES_STAGE_LAYOUT_VERSION`, `DFN_RATE_BRIDGE_LAYOUT_VERSION`).

## Compute admission

`pipelines/tools/profile_dfn2_ops.py` counts 2,134,971 trainable parameters
and 3,307,136 major model MACs per 512-sample invocation, or 310.044 MMAC/s
at 48 kHz. This excludes elementwise operations, host pre/post DSP and memory
traffic; final target-platform admission therefore depends on the board's int8/fp
runtime. Weight storage is about 2.04 MiB int8, 4.07 MiB fp16 or 8.14 MiB
fp32.

At a 16-kHz product boundary, two input resamplers (mic and render) plus the
mono output resampler add about 4.720 MMAC/s (counted over the real taps: 33 per
output sample up, 97 down; the NEON build pads each phase to a multiple of 4 taps, so
the up phases run 36 MACs, +9%, and the down phase 100, +3%).
Whole-front-end conversion also
moves the AEC itself to 48 kHz; that extra AEC cost is separate from this
resampler count and must be included in the board measurement.

Pool printed by the skeleton (`mono_aec_dfn_res`, matched delay, five-filter bank, NE10; the sum
of `aec_get_mem_size()`, the three gain and spectrum buffers of the RES path and the stage or
bridge requirement, each 16-byte aligned): 718,096 bytes at the 16 kHz default (the AEC and the RES gain path plus the
rate bridge, whose own pool is 350,000 bytes including the stage), 1,450,016 bytes at 48 kHz (the
stage's 232,096 bytes, its synthesis and the AEC; the conventional MMSE-LSA chain on the 48 kHz
grid needs 1,356,352 bytes) and 622,720 bytes at 8 kHz. Model weights and accelerator activation/
workspace memory are external to this descriptor (the library's state is sized for its ring capacity: a build with the
capacity 7 that pair 2/4 needs, `make DFN2_MASK_LOOKAHEAD=2 DFN2_DF_LOOKAHEAD=4`, prints 742,208 bytes at the 16 kHz default
although the skeleton still runs the table's pair); adding int8 weights alone
brings the known minimum to about 2.7 MiB at 16 kHz.
