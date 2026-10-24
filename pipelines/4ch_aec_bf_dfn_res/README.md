# 4-channel AEC + beamformer (bf) + RES + DeepFilterNet2

This example replaces the post-beam MMSE-LSA denoiser of the conventional four-channel
application (`4ch_aec_bf_nr_res/`) with DeepFilterNet2 while keeping the shared delay estimator,
the four linear AEC lanes, the SRP-PHAT and beamformer (bf) of the linear AEC + BF unit
(`4ch_aec_bf_nr_res/audio_pipeline_4ch.h`), the unit's post-beam AEC3 residual suppressor and the
comfort-noise recipe. There is no DFN wrapper library: `main.c` initialises the unit
(`AudioPipeline4Ch`, whose pool already holds the post-beam `AecRes`: `enable_res`, `res_preset`),
the gain and spectrum buffers of the RES path (no denoiser) and the DFN2 stage (or the rate bridge)
separately, forms P in `dfn_form()`, and `samples/dfn_hop.h` runs the network between the post-RES
spectrum and the synthesis:

```text
mics/ref -> audio_pipeline_4ch_process(unit, mics, far, &beam, NULL)
            shared delay, four lanes, SRP-PHAT, beamformer (bf)
         -> audio_pipeline_4ch_fuse(unit, &frame)
              E = beamformed error spectrum         -> DFN2 feature input
              frame.g_res_ctx = the unit's RES gain
         -> P = E * min(1, frame.g_res_ctx)         P = E * G_res (+ comfort noise) -> DFN2 applied here
                                                   (dfn_form(): sk_min_f32 + sk_capply_gain_f32 + aec_cng_fill)
         -> dfn_hop_begin(&hop, frame.e, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-2,
                                                       the host's iFFT / OLA
```

The skeleton uses the unit's own SRP-PHAT and BF, so `audio_pipeline_4ch_fuse()` builds E from the BF's
spectrum and weights. `dfn_hop_process(&hop, frame.e, P, out)` is the sequence of the `dfn_hop_*`
calls with the model callback (`dfn_res_stage_infer()` here) as the runner. The RES strength is
`aec_res_set_preset(audio_pipeline_4ch_res(unit), preset, ramp_ms)`. The example owns its gain
scratch and comfort-noise state and the ISTFT in the `DfnHop`: the unit carries none of them, and a
refused call resets the unit, the comfort-noise generator and the `DfnHop` together
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
shared-estimator delay re-lock, on a stimulus where a RES-off unit provably
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
  --output-dir /tmp/dfn2-input-ab
```

`--identity-model` is a pipeline/resampler smoke gate only. It requires every
DFN arm to equal RES-only exactly and makes no speech-quality claim. The
contract gate refuses a checkpoint whose model/feature versions differ from the
current model-v6/feature-v5.

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
unit with its post-beam RES, the P and gain buffers, and the stage or bridge, each 16-byte
aligned): 2,058,768 bytes at the 16 kHz default (the unit plus the rate bridge) and
6,617,024 bytes at 48 kHz (the conventional four-channel application on the same unit,
`audio_pipeline_4ch_static`, needs 6,527,568 bytes on the 48 kHz grid).
Model weights and accelerator activation/workspace memory are external;
adding int8 weights alone brings the known minimum to about 4.0 MiB at
16 kHz.
