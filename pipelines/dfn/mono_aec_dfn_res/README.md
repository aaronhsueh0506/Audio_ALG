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
           aec_get_res_context(aec, &ctx)          E = ctx.error_spec = linear-AEC error spectrum -> DFN2 feature input
           P = E * min(1, ctx.res_gain)            P = E * G_res (+ comfort noise) -> DFN2 applied here
                                                   (sk_min_f32 + sk_capply_gain_f32 + aec_cng_fill)
         -> dfn_hop_begin(&hop, ctx.error_spec, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-L (L = M + D, 2 for the shipped pair),
                                                       sqrt-Hann iFFT / OLA
```

`dfn_hop_process(&hop, ctx.error_spec, P, out)` runs the same sequence with the model callback; P must
be formed before `dfn_hop_begin()`. A refused hop is the integrator's to recover: reset the AEC,
the comfort-noise generator (`rng = CNG_RNG_SEED`, a macro of `main.c`) and the `DfnHop` together
(`rig_reset()` in `main.c`).

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
suppressor demonstrably active), and through the resampler round trip by
the bridge delay on a native grid (the same test pins the delay with a
near-end impulse, the frame schedule, and, where the delay is a whole number of native samples, the pass-band
error against the conventional 16/8 kHz output, observed at about 1e-5 against a gate of 1e-3 for the shipped pair).

The host runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 and 8 kHz/256 as the host offers them, or 48 kHz. Only
the DFN2 stage runs at 48 kHz. At 48 kHz the host's spectra are DFN2's grid
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

Lookahead pair. The DFN2 mask and deep-filter lookaheads M (0..2) and D (0..4) are a build setting of this
example (`DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD`, default 1/1) and must equal the pair the model was
trained for; `DFN_RES_STAGE_LOOKAHEAD_FRAMES` = M + D. The rules, the make variables and `make test-pairs` are in
[`../README.md`](../README.md). Only the shipped 1/1 pair has a trained checkpoint, and the numbers in this
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
`--sample-rate 8000`) and checks that malformed numbers such as `16000Hz` and `12dB` exit with 2.

`libaudio_pipeline_dfn.a` holds the shared stage, the rate bridge and the DFN2
host pre/post objects. The example links it with `libaec` and
`libaudio_common` (no denoiser, so no `libanr`); the board supplies the stateless model callback
(`DFN2Model`) and the exported ERB matrices (`erb_fwd.bin` / `erb_inv.bin`
from `AINR/DeepFilterNet2/export_erb_matrix.py`). `main.c` is the board
skeleton: delay-profile flags in the `aec_wav` spelling, `--erb-fwd` /
`--erb-inv`, `--sample-rate` (default 16000), `--atten-lim`, `--cng`, and a fail-open `run_accelerator()`
stub to replace with the runtime call. Numbers are parsed by `common/example_cli.h`
(`example_cli_parse_int` / `example_cli_parse_float`): empty strings, leading spaces, trailing characters,
out-of-range values, inf and nan are rejected with exit code 2; plain decimals and a leading `+`
are accepted. It publishes the model I/O descriptor
of the geometry it was compiled against; a product fills that descriptor from
its model's exported metadata instead, which is what makes a mismatched graph
detectable.

## Calling sequence and parameters

```
App                 Aec           DfnHop + DfnResStage | DfnRateBridge      AcIstft (48 kHz only)
 |-- aec_resolve_signal_grid; aec_config_from_preset; aec_get_mem_size / aec_init -->|
 |-- dfn_res_stage_get_mem_requirements / dfn_res_stage_init_ex            (48 kHz host)
 |   or dfn_rate_bridge_get_mem_requirements / dfn_rate_bridge_init_ex     (8 / 16 kHz host)
 |-- dfn_hop_init_stage(...) or dfn_hop_init_bridge(...)
 |   per hop: aec_process_context --> aec_get_res_context --> P = E * min(1, res_gain) (+ comfort noise, off by default)
 |            --> dfn_hop_begin(E, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused hop: aec_reset + comfort-noise rng = CNG_RNG_SEED + dfn_hop_reset   (rig_reset())
 |   never reset the stage on an AEC delay change
 |-- aec_destroy; dfn_res_stage_destroy | dfn_rate_bridge_destroy; release the blocks
```

External parameters (board-skeleton command line, `dfn/dfn_example_args.h`): `--sample-rate` (8000 / 16000 default
/ 48000), the delay profile `--delay-mode matched|fixed|external` with `--delay-num-filters` (1..5) and
`--fixed-delay`, `--erb-fwd` / `--erb-inv` (the exported ERB matrices; without them the skeleton runs a
fail-open smoke only), `--atten-lim` (dB, 0 = unlimited; `dfn_res_stage_set_atten_lim()` between hops),
`--cng` (comfort noise, off by default). An option followed by another `--` option, or by the end of the line, has no value and the
program exits with code 2; a file named `--x` is written `./--x`. Strength of the RES gain: `aec_set_preset()`. Internal: the
stage grid (48 kHz / 1024 / 512 only), the `DFN_RES_STAGE_IN_SCALE` / `_OUT_SCALE` powers of two, the rate
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
bridge requirement, each 16-byte aligned): 716,400 bytes at the 16 kHz default (the AEC and the RES gain path plus the
rate bridge, whose own pool is 349,888 bytes including the stage), 1,443,712 bytes at 48 kHz (the
stage's 231,984 bytes, its synthesis and the AEC; the conventional MMSE-LSA chain on the 48 kHz
grid needs 1,354,256 bytes) and 621,024 bytes at 8 kHz. Model weights and accelerator activation/
workspace memory are external to this descriptor (the DFN2 ring grows beyond the shipped size when M + D
exceeds 4: the 16 kHz skeleton with pair 2/4 printed 740,496 bytes); adding int8 weights alone
brings the known minimum to about 2.7 MiB at 16 kHz.
