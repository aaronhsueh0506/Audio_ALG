# 4-channel AEC + beamformer (bf) + RES + DeepFilterNet2

This example replaces the post-beam MMSE-LSA denoiser of the conventional four-channel
application (`4ch_aec_bf_nr_res/`) with DeepFilterNet2 while keeping the shared delay estimator,
the four linear AEC lanes, the SRP-PHAT and beamformer (bf) of the linear AEC + BF unit
(`4ch_aec_bf_nr_res/audio_pipeline_4ch.h`), a post-beam AEC3 residual suppressor and the
comfort-noise recipe. There is no DFN wrapper library: `main.c` initialises the unit
(`AudioPipeline4Ch`, which holds no RES), the post-beam `AecRes` (libaec: `aec_res_get_mem_size()` /
`aec_res_init()` on the unit's grid, balanced preset), the gain and spectrum buffers of the RES path
(no denoiser) and the DFN2 stage (or the rate bridge) separately, forms P in `dfn_form()`, and
`samples/dfn_hop.h` runs the network between the post-RES spectrum and the synthesis:

```text
mics/ref -> audio_pipeline_4ch_process(unit, mics, far, &beam, NULL)
            shared delay, four lanes, SRP-PHAT, beamformer (bf)
         -> audio_pipeline_4ch_fuse(unit, &frame)
              E = beamformed error spectrum         -> DFN2 feature input
              frame.g_res_ctx = NULL (the unit has no RES)
         -> G_res = se_res_stage_gain(res, &frame)  the example's own AecRes
         -> P = E * min(1, G_res)                   P = E * G_res (+ comfort noise) -> DFN2 applied here
                                                   (dfn_form(): sk_min_f32 + sk_capply_gain_f32 + aec_cng_fill)
         -> dfn_hop_begin(&hop, frame.e, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-2,
                                                       the host's iFFT / OLA
```

The skeleton uses the unit's own SRP-PHAT and BF, so `audio_pipeline_4ch_fuse()` builds E from the BF's
spectrum and weights. `dfn_hop_process(&hop, frame.e, P, out)` is the sequence of the `dfn_hop_*`
calls with the model callback (`dfn_res_stage_infer()` here) as the runner. The RES strength is
`aec_res_set_preset(res, preset, ramp_ms)` on the example's own `AecRes`. The example owns its RES, gain
scratch and comfort-noise state and the ISTFT in the `DfnHop`: the unit carries none of them, and a
refused call resets the unit, the `AecRes`, the comfort-noise generator and the `DfnHop` together
(`dfn_reset()` in `main.c`). The
network estimates from the beamformed error and is applied to the post-RES
spectrum. The comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins, which breaks the estimate/apply phase consistency the
complex deep filter relies on (when on it travels through the network with the
rest of P and `atten_lim_db` bounds the attenuation of every bin). With no model callback
(or a failing one) the output is the
conventional chain with the denoiser off delayed: bit for bit by two hops at
48 kHz (the acceptance gate in `tests/test_4ch_dfn.c` asserts it against the hand-written chain of
`4ch_aec_bf_nr_res/tests/conv_chain_4ch.h`, with comfort noise off and on, across a real
shared-estimator delay re-lock, on a stimulus where a run without the example's `AecRes` provably
differs, and checks that the stage is not reset on the shared aligner's delay change), and through the resampler round trip by the bridge delay at
16 kHz (the same test pins the delay with a near-end impulse, the frame
schedule, and the pass-band error against the conventional 16 kHz output,
measured at 1e-5).

The unit runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512, or 48 kHz (the unit's rates). Only the DFN2 stage runs
at 48 kHz: directly on the unit's spectra at 48 kHz (algorithmic latency
three hops, 32 ms), or through the rate bridge (`pipelines/dfn_rate_bridge.h`)
at 16 kHz, which synthesises E and P on the native grid, resamples them up,
runs the stage on the 48 kHz 1024/512 grid and resamples the result down.
The four lanes, the beamformer (bf) seam, the RES and the comfort noise never leave
the product rate. The bridge adds 672 samples (42 ms) at 16 kHz with hop
128 and 629 (39 ms) with hop 256 to the host's own output; total algorithmic
latency is that plus the host's WOLA hop (16 kHz, hop 128: 800 samples,
50 ms). The stage schedule per native hop is 0,1,1,1 frames at hop 128 and
1,2 at hop 256.

Build and test from `pipelines/4ch_aec_bf_nr_res/` (the directory `Makefile`
forwards):

```sh
make WERROR=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res
make WERROR=1 test        # includes test_4ch_dfn and the skeleton
```

`libdfn_res_stage.a` holds the shared stage, the rate bridge and the DFN2 host
pre/post objects; link it ahead of the unit's archives (`libaudio_pipeline_4ch.a`, the
SRP/BF archives of audiolib, `libaec`, `libanr` and `libaudio_common`). The board supplies the stateless model
callback (`DFN2Model`) and the exported ERB matrices. `main.c` is the board
skeleton (delay-profile flags, `--sample-rate` (default 16000), `--erb-fwd`
/ `--erb-inv`, `--atten-lim`, `--cng`; the unit's default array geometry stands in for
the product's measured one; it
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
App               AudioPipeline4Ch     libaec AecRes     DfnHop + DfnResStage | DfnRateBridge     AcIstft (48 kHz only)
 |-- audio_pipeline_4ch_default_config; _get_mem_requirements; _init_ex -->|
 |-- aec_config_from_preset (fft_size = audio_pipeline_4ch_fft_size); aec_res_get_mem_size; aec_res_init -->|
 |-- dfn_res_stage_* (48 kHz) or dfn_rate_bridge_* (16 kHz): size query + init_ex, dfn_hop_init_*
 |   per hop: audio_pipeline_4ch_process --> audio_pipeline_4ch_fuse --> se_res_stage_gain(res) --> P = E * min(1, G_res) (dfn_form())
 |            --> dfn_hop_begin(E, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused call: audio_pipeline_4ch_reset + aec_res_reset + comfort-noise rng = SE_CNG_RNG_SEED + dfn_hop_reset   (dfn_reset())
 |   never reset the stage on the shared aligner's delay change
 |-- audio_pipeline_4ch_destroy; stage / bridge destroy; release the blocks
```

External parameters: the same board-skeleton flags as the mono variant (`--sample-rate` 16000 default or
48000, delay profile, `--erb-fwd` / `--erb-inv`, `--atten-lim`, `--cng`), plus the unit's `AudioPipeline4ChConfig`
(array geometry, `num_angles`; the skeleton uses the default 35 mm UCA, which a
product replaces with the measured one) and `aec_res_set_preset(res, ...)` on the example's own `AecRes`.
Internal: the unit's BF / DOA / VAD constants (section 0.3 of
[`../../docs/integration_4ch_spatial_zh_TW.md`](../../docs/integration_4ch_spatial_zh_TW.md)), the stage
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
model dominates this arithmetic budget; A53/A73 CPU-only admission still
requires the final int8/fp runtime and board cycle measurement.

Pool printed by the skeleton (`4ch_aec_bf_dfn_res`, matched delay, five-filter bank, NE10; the
unit, the post-beam `AecRes` carved by the example, the P and gain buffers, and the stage or bridge, each 16-byte
aligned): 2,058,752 bytes at the 16 kHz default (the unit plus the rate bridge) and
6,617,008 bytes at 48 kHz (the conventional four-channel application on the same unit,
`audio_pipeline_4ch_static`, needs 6,525,488 bytes on the 48 kHz grid).
Model weights and accelerator activation/workspace memory are external;
adding int8 weights alone brings the known minimum to about 4.0 MiB at
16 kHz.
