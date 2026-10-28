# 4-channel AEC + beamformer (bf) + RES + DeepFilterNet2

This example replaces the post-beam MMSE-LSA denoiser of the conventional four-channel
application (`4ch_aec_bf_nr_res/`) with DeepFilterNet2 while keeping the shared delay estimator,
the four linear AEC lanes, the SRP-PHAT and beamformer (bf) glue, a post-beam AEC3 residual
suppressor and the comfort-noise recipe. mcab (`4ch_aec_bf_nr_res/mcab.h`) is a flow layer, not a
library: the application initialises the four libaec `Aec` lanes and, in MATCHED mode, the one
shared `DelayAec`, and passes the pointers; mcab owns only its flow state. There is no DFN wrapper
library either. `main.c` builds the lanes, the estimator and mcab in one block with
`mcab_sample_flow.h`, adds the sample SRP-PHAT/BF helper (`mcab_sample_beam.h`, not part of the mcab
archive), initialises the post-beam `AecRes` (libaec: `aec_res_get_mem_size()` / `aec_res_init()`
on the flow's grid, balanced level), the gain and spectrum buffers of the RES path (no denoiser)
and the DFN2 stage (or the rate bridge, sized and carved by `dfn/dfn_hop_pool.h`) separately,
forms P in `dfn_form()`, and `dfn/dfn_hop.h` runs the network between the post-RES spectrum and
the synthesis:

```text
mics/ref -> mcab_sample_beam_process(unit, mics, far, &beam, NULL)
            mcab_process_front(): shared delay, four lanes; helper: SRP-PHAT, beamformer (bf)
         -> mcab_sample_beam_fuse(unit, &frame)     = mcab_fuse(flow, weights, E, &frame)
              E = beamformed error spectrum         -> DFN2 feature input
              frame.g_res_ctx = NULL (mcab has no RES)
         -> G_res = aec_res_process(res, &in)       the example's own AecRes; in filled by mcab_sample_res_input(&frame, &in)
         -> P = E * min(1, G_res)                   P = E * G_res (+ comfort noise) -> DFN2 applied here
                                                   (dfn_form(): sk_min_f32 + sk_capply_gain_f32 + aec_cng_fill)
         -> dfn_hop_begin(&hop, frame.e, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-L (L = M + D, 2 for the shipped pair),
                                                       the host's iFFT / OLA
```

The skeleton uses the helper's own SRP-PHAT and BF, so `mcab_sample_beam_fuse()` hands mcab the
BF's spectrum and weights. An application with its own beamformer calls `mcab_process_front()` and
`mcab_fuse(p, W, E, &frame)` instead. `dfn_hop_process(&hop, frame.e, P, out)` is the sequence of
the `dfn_hop_*` calls with the model callback (`dfn_res_stage_infer()` here) as the runner. The RES
strength is `aec_res_set_level(res, level, ramp_ms)` on the example's own `AecRes`. The example
owns its RES, gain scratch and comfort-noise state and the ISTFT in the `DfnHop`: mcab carries none
of them, and a refused call resets the helper (and with it the flow), the `AecRes`, the
comfort-noise generator and the `DfnHop` together (`dfn_reset()` in `main.c`). The
network estimates from the beamformed error and is applied to the post-RES
spectrum. The comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins, which breaks the estimate/apply phase consistency the
complex deep filter relies on (when on it travels through the network with the
rest of P and `atten_lim_db` bounds the attenuation of every bin). With no model callback
(or a failing one) the output is the
conventional chain with the denoiser off delayed: bit for bit by L = M + D hops at
48 kHz (two for the shipped lookahead pair; the acceptance gate in `tests/test_4ch_dfn.c` asserts it against the hand-written chain of
`4ch_aec_bf_nr_res/tests/conv_chain_mcab.h`, with comfort noise off and on, across a real
shared-estimator delay re-lock, on a stimulus where a run without the example's `AecRes` provably
differs, and checks that the stage is not reset on the shared aligner's delay change), and through the resampler round trip by the bridge delay at
16 kHz (the same test pins the delay with a near-end impulse, the frame
schedule, and, where the delay is a whole number of native samples, the pass-band error
against the conventional 16 kHz output, observed at about 1e-5 against a gate of 1e-3 for the shipped pair).

The flow runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 (covered by `test_4ch_dfn`; the skeleton has no fft flag), or 48 kHz (mcab's
rates, no 8 kHz). Only the DFN2 stage runs
at 48 kHz: directly on the flow's spectra at 48 kHz (algorithmic latency
L + 1 hops; three hops, 32 ms, for the shipped pair), or through the rate bridge (`pipelines/dfn/dfn_rate_bridge.h`)
at 16 kHz, which synthesises E and P on the native grid, resamples them up,
runs the stage on the 48 kHz 1024/512 grid and resamples the result down.
The four lanes, the beamformer (bf) seam, the RES and the comfort noise never leave
the product rate. The bridge adds `prefill + 32 + (L + 1) x 512 x sr / 48000` native samples (integer
division) to the host's own output: for the shipped pair 672 samples (42 ms) at 16 kHz with hop
128 and 629 (39 ms) with hop 256; total algorithmic
latency is that plus the host's WOLA hop (16 kHz, hop 128: 800 samples,
50 ms). A pair whose L + 1 is not a multiple of 3 has a fractional stage delay that is floored
(pair 0/1: 501 / 458 samples). The stage schedule per native hop is 0,1,1,1 frames at hop 128 and
1,2 at hop 256.

Lookahead pair. The DFN2 mask and deep-filter lookaheads M (0..2) and D (0..4) are a build setting
(`DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD`, default 1/1; this folder's Makefile forwards to
`4ch_aec_bf_nr_res/Makefile`, which takes both variables, folds them into its build signature and has no
`test-pairs` target). They must equal the pair the model was trained for. The rules, the pipelines make
variables and `make -C pipelines test-pairs` are in [`../README.md`](../README.md). Only the shipped 1/1 pair has a
trained checkpoint; the numbers in this file are for it unless a pair is named.

Build and test from `pipelines/4ch_aec_bf_nr_res/` (the Makefile that owns the rules):

```sh
make WERROR=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res
make WERROR=1 test        # includes test_4ch_dfn and the skeleton
```

The `Makefile` in this directory is a thin forwarder: `make -C pipelines/dfn/4ch_aec_bf_dfn_res test`
builds and runs `test_4ch_dfn`, then the skeleton (default, `--delay-num-filters 1 --cng --atten-lim 12`,
`--sample-rate 48000`) and checks that `--sample-rate 16000Hz`, `--sample-rate 8000` and
`--atten-lim 12dB` exit with 2.

`libdfn_res_stage.a` holds the shared stage, the rate bridge and the DFN2 host
pre/post objects; link it ahead of the flow's archives (`libaudio_pipeline_mcab.a`, which holds
`mcab.o` only, the SRP/BF archives of audiolib, `libaec`, `libanr` and `libaudio_common`). The
sample objects `mcab_sample_flow.o`, `mcab_sample_beam.o` and `mcab_sample_driver.o` are linked into
the program, not archived. The board supplies the stateless model
callback (`DFN2Model`) and the exported ERB matrices. `main.c` is the board
skeleton (delay-profile flags, `--sample-rate` (16000 default or 48000), `--erb-fwd`
/ `--erb-inv`, `--atten-lim`, `--cng`; the helper's default array geometry stands in for
the product's measured one; numbers are parsed strictly by `common/example_cli.h`, so values such as
`16000Hz` exit with 2; it
publishes the compiled-in model I/O descriptor, which a product replaces with
the one from its model's exported metadata).

The Python evaluator compares the original NR+RES tail, RES-only and the DFN
feature-source experiments on the same fused contexts. The C example uses
the beamformed-error source only; a raw-capture or single-lane estimate is
not phase-consistent with the post-beam spectrum, so those arms are
meaningful with the mask-only compose and stay experiments:

```sh
python pipelines/tools/eval_dfn2_4ch_inputs.py \
  --checkpoint /path/to/a-v6-checkpoint.pth \
  --output-dir dfn2-input-ab
```

`--identity-model` is a pipeline/resampler smoke gate only. It requires every
DFN arm to equal RES-only exactly and makes no speech-quality claim. The
contract gate refuses a checkpoint whose model/feature versions differ from the
current model-v6/feature-v5.

## Calling sequence and parameters

```
App          McabSampleFlow + Mcab   McabSampleBeam   libaec AecRes   DfnHop + DfnResStage | DfnRateBridge   AcIstft (48 kHz only)
 |-- mcab_default_config; mcab_get_mem_requirements / _breakdown; mcab_sample_flow_get_mem_size / _init -->|
 |-- mcab_sample_beam_default_params; mcab_sample_beam_get_mem_size / _init(flow) -->|
 |-- aec_config_from_level (fft_size = mcab_get_mem_breakdown().fft_size); aec_res_get_mem_size; aec_res_init -->|
 |-- dfn_hop_pool_plan / dfn_hop_pool_build: dfn_res_stage_* (48 kHz) or dfn_rate_bridge_* (16 kHz), dfn_hop_init_*
 |   per hop: mcab_sample_beam_process --> mcab_sample_beam_fuse --> aec_res_process(res, &in) --> P = E * min(1, G_res) (dfn_form())
 |            --> dfn_hop_begin(E, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused call: mcab_sample_beam_reset + aec_res_reset + comfort-noise rng = MCAB_SAMPLE_CNG_RNG_SEED + dfn_hop_reset   (dfn_reset())
 |   never reset the stage on the shared aligner's delay change
 |-- stage / bridge destroy; mcab_sample_beam_destroy; mcab_sample_flow_destroy; release the blocks
```

External parameters: the same board-skeleton flags as the mono variant (`--sample-rate` 16000 default or
48000, delay profile, `--erb-fwd` / `--erb-inv`, `--atten-lim`, `--cng`), plus the flow's `McabConfig`
(delay profile, `capture_proxy_channel`, `enable_post`) and the helper's `McabSampleBeamParams`
(array geometry, `num_angles`; the skeleton uses the default 35 mm UCA, which a
product replaces with the measured one) and `aec_res_set_level(res, ...)` on the example's own `AecRes`.
Internal: the helper's BF / DOA / VAD constants (section 0.3 of
[`../../../docs/integration_4ch_spatial_zh_TW.md`](../../../docs/integration_4ch_spatial_zh_TW.md)), the stage
grid and scale constants, the bridge FIR and schedule.

## Compute admission

`pipelines/tools/profile_dfn2_ops.py` counts the current streaming graph. The
v6 architecture has 2,134,971 trainable parameters and 3,307,136 major MACs
per 512-sample invocation, or 310.044 MMAC/s at 48 kHz. This excludes
elementwise operations, host pre/post DSP and memory traffic, so it is not a
CPU cycle claim. Weight storage alone is about 2.04 MiB int8, 4.07 MiB fp16,
or 8.14 MiB fp32.

For a 16-kHz source, the house FIR rate boundary adds approximately 1.584
MMAC/s per input channel and 1.552 MMAC/s on the mono output: 4.720 MMAC/s for
mono mic+render+output, or 9.472 MMAC/s for four mics+render+output. The DFN
model dominates this arithmetic budget; target-CPU-only admission still
requires the final int8/fp runtime and board cycle measurement.

Pool printed by the skeleton (`4ch_aec_bf_dfn_res`, matched delay, five-filter bank, NE10; the
flow objects (four lanes, shared estimator, mcab pool), the SRP/BF helper, the post-beam `AecRes` carved by the example,
the P and gain buffers, and the stage or bridge, each 16-byte
aligned; shipped pair): 2,058,672 bytes at the 16 kHz default (flow plus helper 1,698,784 bytes, plus the rate bridge) and
6,616,928 bytes at 48 kHz (flow plus helper 6,300,352 bytes). mcab's own pool is 73,504 bytes at
16 kHz/256 (68,288 bytes front-only); the lanes and the estimator have their own pools.
Model weights and accelerator activation/workspace memory are external;
adding int8 weights alone brings the known minimum to about 4.0 MiB at
16 kHz.
