# Mono AEC + RES + DeepFilterNet2

This example replaces the MMSE-LSA denoiser of the conventional mono
chain with DeepFilterNet2 while keeping the linear AEC, the AEC3 residual echo
suppressor and the comfort-noise recipe. There is no mono unit and no DFN wrapper
library: `main.c` initialises lib/aec's `Aec` (which carries the RES gain itself), the post spectrum
buffer of the RES path (no denoiser) and the DFN2
model stream (`stream/ainr/dfn2/dfn2_stream.h`, memory mode, `freq_only`; below 48 kHz with the sample-rate bridge
`stream/ainr/bridge/ainr_bridge.h` around it) separately, forms P in
`rig_process()`, and `dfn/dfn_hop.h` runs the network between the post-RES spectrum and the
synthesis. `dfn/dfn_hop_pool.h` sizes and carves the DFN tail of the pool (the model stream, the
scale-conversion buffers, then the synthesis and the output spectrum, or the bridge):

```text
mic/ref -> aec_process_linear(aec, mic, ref); aec_process_residual(aec, NULL)
           follow_post_reset()                     new aec_post_reset_serial(): the host-synthesis OLA clear, see below (the library reseeds its comfort noise)
           aec_get_residual(aec, &v)               G_res = v.res_gain, comfort-noise PSD
                                                   B = v.base_spec = the rendering base (E, or E + echo on a guard hop)
                                                   -> DFN2 feature input
           P = B * G_res                           aec_apply_gain(aec, NULL, P) (+ aec_apply_cng(aec, NULL, P) when on)
                                                   -> DFN2 applied here
         -> dfn_hop_begin(&hop, B, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-L (L = M + D, 2 for the shipped pair),
                                                       sqrt-Hann iFFT / OLA
```

`dfn_hop_process(&hop, B, P, out)` runs the same sequence with the model callback; P must
be formed before `dfn_hop_begin()`. A refused hop is the integrator's to recover: reset the AEC (which also
reseeds its own comfort-noise generator) and the `DfnHop` together (`rig_reset()` in `main.c`).

**Two spectra.** `v.error_spec` / `v.formed_hop` are the AEC's analysis seam: the selected linear residual, never
altered by the output guard. B (`v.base_spec`) is what `aec_process()` itself would multiply by its gain:
E, or the capture spectrum E + echo on a hop where the linear estimate is unusable and E is louder than the
capture. B is E bit for bit on every hop the guard leaves alone. The DFN2 hop reads the pair (B, P): its feature input
and the spectrum it filters stay one signal, so a guard hop never pairs a capture-based P with a residual-based
estimate. There is no denoiser here, so no further estimator reads either spectrum.

**Hard post-chain reset event.** The AEC moves `aec_post_reset_serial()` when its post chain is hard-reset inside a
processing call (a confirmed delay change that restarts the filter, a first acquisition without warm
transfer) or by `aec_reset()`; warm-transfer and soft realignments and the bad-relock soft revert do not move it. `follow_post_reset()` keeps the
last handled serial; the library reseeds its comfort-noise generator to `AEC_CNG_SEED` on the same event, before that
hop's first draw. The model stream never resets on it and the program never resets the AEC in response. At 48 kHz the program owns the synthesis
(`AcIstft`, `hop.istft`) and the model delays every frame by `dfn_hop_lookahead_frames()` hops, so the overlap-add tail is
dropped when the frame built on the reset hop reaches the synthesis (a delay line of that many hops, not at the
reset hop: that would cut the previous filter's last frame out of the output). Below 48 kHz the host-grid synthesis
lives inside the `AinrBridge` (`hop.bridge`) and runs at the top of its `begin()`, with nothing between the spectra and it, so the
call is `ainr_bridge_reset_synthesis()` on the hop of the event, before the hop enters the bridge: no delay line.
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
its quality against a mask-only variant was evaluated by an offline driver
that is now archived outside the repository (see the cleanup record).

With no model callback (or a failing one) the output is the conventional
chain with the denoiser off delayed: bit for bit by L = M + D hops at 48 kHz (two for the shipped
lookahead pair; the acceptance gate in `tests/test_mono_dfn.c`, compared with the hand-written chain of
`pipelines/mono/aec_nr_res/tests/conv_chain_mono.h`, asserted with
comfort noise off and on, across a real matched-delay re-lock, with the
suppressor demonstrably active, and on streams that fire the output guard and hard-reset the post chain, with
the reset event delayed by L hops), the post spectrum P itself equals `aec_process()` of an `Aec` with `AEC_RESIDUAL_FULL`
after the host's own synthesis with the AEC's window table, bit for bit, at 16 kHz/256, 16 kHz/512 and 48 kHz, and through the resampler round trip by
the bridge delay on a native grid (the same test pins the delay with a
near-end impulse, the frame schedule, and, where the delay is a whole number of native samples, the pass-band
error against the conventional 16/8 kHz output, observed at about 1e-5 against a gate of 1e-3 for the shipped pair,
on streams that carry hard-reset events with the bridge's synthesis cleared on the hop of the event). The bridge's own
native-grid synthesis (`ainr_bridge_get_host_hops()`) equals the hand-written chain bit for bit at 16 kHz/256,
16 kHz/512 and 8 kHz/256 on streams that fire the guard and reset the post chain, and stays within the window rounding
of `aec_process()` (audiolib's default window against the AEC's table; observed below 1e-7); a bridge that ignores
the event or clears one hop late fails both gates.

The host runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 and 8 kHz/256 as the host offers them, or 48 kHz. As shipped, only
the DFN2 model runs at 48 kHz (the model grid is chosen at run time with `--ai-sample-rate` / `--ai-fft-size`, see "DFN2 grid" below). At 48 kHz the host's spectra are DFN2's grid
and the model stream is applied directly (algorithmic latency L + 1 hops: L model
lookahead hops plus one synthesis-WOLA hop; three hops, 32 ms, for the shipped pair). Below 48 kHz the rate
bridge (`stream/ainr/bridge/ainr_bridge.h`) synthesises E and P on the native
grid, resamples them up with `audiolib`'s polyphase FIR, hands the hop one 48 kHz 1024/512 frame
every 512 new samples for the model stream, and resamples the result
down; the AEC, RES and comfort noise never leave the product rate. The
bridge adds a fixed delay to the host's own output, `prefill + 32 + (L + 1) x 512 x sr / 48000` native samples
(integer division): for the shipped pair 672 samples at 16 kHz
with hop 128 (42 ms), 629 at 16 kHz with hop 256 (39 ms), 330 at 8 kHz
(41 ms); the total algorithmic latency is that plus the host's WOLA hop
(16 kHz, hop 128: 800 samples, 50 ms). For a pair whose L + 1 is not a multiple of 3 the model delay is a
fraction of a native sample and is floored (pair 0/1: 501 / 458 / 244 samples). Per native hop the model runs a fixed
schedule (16 kHz/hop 128: 0,1,1,1 frames; 16 kHz/hop 256 and 8 kHz: 1,2), so
two inferences in one hop is a real per-hop compute fact. The Python reference
`pipelines/mono/aec_dfn_res/pipeline.py` mirrors both paths (16 and 48 kHz
hosts); its stage reads the pair from the model.

DFN2 grid. Neither the DFN2 library nor the pipeline units hold a grid (see [`AINR/DeepFilterNet2/README.md`](../../../AINR/DeepFilterNet2/README.md),
"Choosing the grid"): `main.c` builds the model grid at run time from `--ai-sample-rate <Hz>` / `--ai-fft-size <n>`
(default 48000 / 1024) with `dfn2_grid_make()`, using the two-row table of `dfn/dfn_example_args.h`
(16000/256 -> n_erb 24, df_bins 96, lookahead 0/0; 48000/1024 -> 32, 96, 1/1; no 32 kHz row, the AEC has none; any other
pair is refused with the list, exit 2) and hands it to `DfnHopPoolPlan.stream_cfg.grid` (`dfn_hop_pool_plan_init()`). The model stream takes its rate and FFT size from that grid, and
`DfnHop` its input and output scales (1/sqrt(n_fft) and its inverse). A host at the model's rate needs no bridge:
`--ai-sample-rate 16000 --ai-fft-size 256` with the 16 kHz host runs straight through the model stream (hop 128, algorithmic latency 128
samples for the 0/0 row, pool 301,312 B), against 800 samples and
454,720 B for the shipped 48 kHz model with the same 16 kHz host through the bridge. The bridge exists only for the shipped
48 kHz / 1024 model (`dfn_hop_pool_plan()` returns -1 on any other model grid) and its hosts stay 8 kHz and 16 kHz, so a 16 kHz model
refuses a host of another rate ("pool query rejected", exit 1), and the 48 kHz host needs the 48 kHz model. The banner prints the model
grid and whether the model stream runs on the host grid or through the bridge. `tests/test_mono_dfn.c` and the Python reference cover the shipped grid; `dfn/tests/test_dfn_hop.c` holds the hop to a hand-written recipe on the model stream, and `make test-dfn-grids` runs it on the 16 and 32 kHz test grids.

Lookahead pair. The DFN2 mask and deep-filter lookaheads M (0..2) and D (0..4) are two fields of the model grid
(a row of the table above, not a build setting) and must equal the pair the model was trained for;
`dfn_hop_lookahead_frames(grid)` = M + D. The make variables `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` choose only the pair
the test programs run, never the pair `main.c` runs. The rules and
`make test-pairs` are in [`../../reference/build.md`](../../reference/build.md) ("DFN lookahead pair"). Only the shipped 1/1 pair has a trained checkpoint, and the numbers in this
file are for it unless a pair is named.

Build and test from `pipelines/` (the Makefile that owns the rules):

```sh
make WERROR=1 libaudio_pipeline_dfn.a mono_aec_dfn_res
make WERROR=1 test        # includes test_dfn_hop, test_mono_dfn and the skeleton
```

The `Makefile` in this directory is a thin forwarder: `make -C pipelines/mono/aec_dfn_res test`
builds and runs `test_dfn_hop` and `test_mono_dfn`, then the
skeleton (default, `--delay-num-filters 1 --cng --atten-lim 12`, `--sample-rate 48000`,
`--sample-rate 8000`, `--ai-sample-rate 16000 --ai-fft-size 256`) and checks that malformed numbers such as `16000Hz` and `12dB`, and a model grid outside the table, exit with 2, and that a model grid the bridge cannot serve is a rejected pool query (exit 1).

`libaudio_pipeline_dfn.a` holds the DFN2 model stream (`dfn2_stream.o`), the sample-rate bridge
(`ainr_bridge.o`), the DFN2 host pre/post objects. The example links it with `libaec` and
`libaudio_common` (no denoiser, so no `libanr`); the board supplies the stateless model callback
(`DFN2Model`) and the ERB matrices (`erb_fwd.bin` / `erb_inv.bin`
from `AINR/DeepFilterNet2/export_erb_matrix.py`; layout in [`../../dfn/README.md`](../../dfn/README.md)). `main.c` is the board
skeleton: delay-profile flags in the `aec_wav` spelling, `--erb-fwd` /
`--erb-inv`, `--sample-rate` (default 16000), `--ai-sample-rate` / `--ai-fft-size` (the model grid, default 48000 / 1024), `--atten-lim`, `--cng`, and a fail-open `run_accelerator()`
stub to replace with the runtime call (the recurrent state is the accelerator's: it reloads quantized zeros when
`job.state_generation` (`dfn_hop_state_generation()`) changes: a reset, a
refused commit or a run that started and failed; a callback that ran nothing returns `STREAM_RUN_NOT_STARTED` and moves
nothing). Numbers are parsed by `example_cli.h`
(`example_cli_parse_int` / `example_cli_parse_float`): empty strings, leading spaces, trailing characters,
out-of-range values, inf and nan are rejected with exit code 2; plain decimals and a leading `+`
are accepted. It publishes the model I/O descriptor
of the model grid (`dfn2_model_io_descriptor_default(&grid, &d)`); a product fills that descriptor from
its model's exported metadata instead, which is what makes a mismatched graph
detectable.

## Calling sequence and parameters

```
App                 Aec           DfnHop + DFN2Stream (+ AinrBridge)        AcIstft (DFN2 grid rate; 48 kHz as shipped)
 |-- dfn2_grid_make (row of --ai-sample-rate / --ai-fft-size); load erb_fwd.bin / erb_inv.bin (--erb-fwd / --erb-inv)
 |-- aec_resolve_signal_grid; aec_config_from_level; aec_get_mem_size / aec_init -->|
 |-- dfn_hop_pool_plan_init / dfn_hop_pool_plan / dfn_hop_pool_bytes / dfn_hop_pool_build:
 |     dfn2_stream_get_mem_size / dfn2_stream_init_ex (memory mode, freq_only), the six conversion buffers, then
 |     ac_istft_init + dfn_hop_init_direct                               (host at the DFN2 rate: 48 kHz as shipped)
 |     or ainr_bridge_get_mem_requirements / ainr_bridge_init_ex + dfn_hop_init_bridge   (8 / 16 kHz host)
 |   per hop: aec_process_linear --> aec_process_residual --> follow_post_reset --> aec_get_residual
 |            --> aec_apply_gain (+ aec_apply_cng, off by default) -> P = B * G_res
 |            --> dfn_hop_begin(B, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused hop: aec_reset + dfn_hop_reset   (rig_reset())
 |   never reset the hop on an AEC delay change
 |-- aec_destroy; release the pool (the DFN tail owns nothing else)
```

External parameters (board-skeleton command line, `dfn/dfn_example_args.h`): `--sample-rate` (8000 / 16000 default
/ 48000), `--ai-sample-rate` / `--ai-fft-size` (the DFN2 model grid, default 48000 / 1024), the delay profile `--delay-mode matched|fixed|external` with `--delay-num-filters` (1..5) and
`--fixed-delay`, `--erb-fwd` / `--erb-inv` (the exported ERB matrices of the model grid, both or neither; without them the skeleton runs a
fail-open smoke only), `--atten-lim` (dB, 0 = unlimited; `dfn_hop_set_atten_lim()` between hops),
`--cng` (comfort noise, off by default). An option followed by another `--` option, or by the end of the line, has no value and the
program exits with code 2; a file named `--x` is written `./--x`. Strength of the RES gain: `aec_set_level()`. Internal: the
model grid (the grid of the plan; 48 kHz / 1024 / 512 as shipped), `dfn2_stft_scale()` / `dfn2_stft_unscale()` (2^-5 / 2^5 on the shipped grid), the rate
bridge's FIR resamplers and frame schedule, the AEC and DFN2 model constants. Public functions are the
`dfn2_stream_*` of `stream/ainr/dfn2/dfn2_stream.h`, the `ainr_bridge_*` of `stream/ainr/bridge/ainr_bridge.h` and the
`dfn_hop_*` / `dfn_hop_pool_*` of `dfn/dfn_hop.h` / `dfn/dfn_hop_pool.h`; the stream and the bridge carry 32-byte descriptors
(`DFN2_STREAM_LAYOUT_VERSION`, `AINR_BRIDGE_LAYOUT_VERSION`), which `dfn_hop_pool_build()` hands to their `_init_ex`.

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
of `aec_get_mem_size()`, the post spectrum buffer of the RES path and the DFN tail of
`dfn_hop_pool_bytes()`, each part 16-byte aligned): 454,720 bytes at the 16 kHz default (the AEC and the RES post spectrum plus a
DFN tail of 214,448 bytes: the model stream 84,272, the six conversion buffers 12,312 and the bridge 117,856),
921,648 bytes at 48 kHz (a DFN tail of 143,536 bytes: the stream and the buffers as above, the synthesis 42,832 and the
output spectrum 4,104; the conventional MMSE-LSA chain on the 48 kHz
grid needs 956,240 bytes) and 427,984 bytes at 8 kHz (tail 216,080). The 16 kHz / 256 model on a 16 kHz host: 301,312 bytes
(tail 61,040: stream 45,552, buffers 3,096, synthesis 11,344, spectrum 1,032). The stream is a `freq_only` one
(`DFN2StreamConfig.freq_only`, set by `dfn_hop_pool_plan()`): its pool is the `DFN2Prepost` (79,984 of the 84,272 bytes), with no
analysis, synthesis or PCM buffer, since the hop uses the spectrum entries only. Model weights, accelerator activation/
workspace memory and the recurrent state are external to this descriptor (the state of the shipped graph is
1,280 + 30,720 float32 values, registered by a float host or held by the accelerator); adding int8 weights alone
brings the pool plus weights to about 2.5 MiB at 16 kHz.
