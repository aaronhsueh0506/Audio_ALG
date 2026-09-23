# 4-channel AEC + BF/GSC + RES + DeepFilterNet2

This wrapper replaces the post-beam MMSE-LSA denoiser of the four-channel
core (`4ch_aec_bf_nr_res/4aec_nr_res.h`) with DeepFilterNet2 while keeping
the shared delay estimator, the four linear AEC lanes, the external BF/GSC
seam, the post-beam AEC3 residual suppressor, the comfort-noise recipe and
the synthesis WOLA. The core is hosted with `enable_post=1`, `enable_res=1`,
`enable_nr=0`, and the network is inserted between the core's post-RES
spectrum and the core's synthesis:

```text
mics/ref -> four_aec_dfn_res_process_pre()         four lanes, token
            (external BF/GSC -> effective weights, optional trusted spectrum)
         -> four_aec_nr_res_process_post_view()
              E = beamformed error spectrum         -> DFN2 feature input
              P = E * G_res + comfort noise         -> DFN2 applied here
            dfn_res_stage_process(E, P)             -> enhanced P, frame t-2
            four_aec_nr_res_synthesize_external()   -> the core's iFFT / OLA
```

The pre/post token contract and the external beamformer seam are the same as
`FourAecNrRes`: `process_post()` reconstructs the beamformed error from the
weights, `process_post_trusted_spectrum()` takes the GSC's own spectrum. The
network estimates from the beamformed error and is applied to the post-RES
spectrum. The core's comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins, which breaks the estimate/apply phase consistency the
complex deep filter relies on, and no true-model result yet supports enabling
it (when on it travels through the network with the rest of P and
`atten_lim_db` bounds the attenuation of every bin). With no model callback
(or a failing one) the output is the
conventional core with `enable_nr=0` delayed: bit for bit by two hops at
48 kHz (the acceptance gate in `tests/test_4aec_dfn_res.c` asserts it for
both post entries, with comfort noise off and on, across a real
shared-estimator delay re-lock, on a stimulus where a RES-off core provably
differs), and through the resampler round trip by the bridge delay at
16 kHz (the same test pins the delay with a near-end impulse, the frame
schedule, and the pass-band error against the conventional 16 kHz core,
measured at 1e-5).

The core runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512, or 48 kHz (the core's rates). Only the DFN2 stage runs
at 48 kHz: directly on the core's spectra at 48 kHz (algorithmic latency
three hops, 32 ms), or through the rate bridge (`pipelines/dfn_rate_bridge.h`)
at 16 kHz, which synthesises E and P on the native grid, resamples them up,
runs the stage on the 48 kHz 1024/512 grid and resamples the result down.
The four lanes, the BF/GSC seam, the RES and the comfort noise never leave
the product rate. The bridge adds 672 samples (42 ms) at 16 kHz with hop
128 and 629 (39 ms) with hop 256 to the core's own output; total algorithmic
latency is that plus the core's WOLA hop (16 kHz, hop 128: 800 samples,
50 ms). The stage schedule per native hop is 0,1,1,1 frames at hop 128 and
1,2 at hop 256.

Build and test from `pipelines/4ch_aec_bf_nr_res/` (the directory `Makefile`
forwards):

```sh
make BACKEND=kiss WERROR=1 lib4aec_dfn_res.a 4ch_aec_bf_dfn_res
make BACKEND=kiss WERROR=1 test        # includes test_4aec_dfn_res and the skeleton
```

`lib4aec_dfn_res.a` holds the wrapper, the shared stage and the DFN2 host
pre/post objects; link it with `libaudio_pipeline_4ch.a` (the core), `libaec`,
`libmmse_lsa` and `libaudio_common`. The board supplies the stateless model
callback (`DFN2Model`) and the exported ERB matrices. `main.c` is the board
skeleton (delay-profile flags, `--sample-rate` (default 16000), `--erb-fwd`
/ `--erb-inv`, `--atten-lim`, `--cng`, uniform weights standing in for the
product's beamformer; it
publishes the compiled-in model I/O descriptor, which a product replaces with
the one from its model's exported metadata).

The Python evaluator compares the original NR+RES tail, RES-only and the DFN
feature-source experiments on the same fused contexts. The C wrapper ships
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
checked-in `dfn2_best.pth` and `dfn2_last.pth` are v5/feature-v3
checkpoints, while current code requires model-v6/feature-v5. The contract
gate intentionally refuses them; a fresh compatible training run is required
before PESQ/STOI or echo-cohort conclusions are valid.

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

Host DSP pool (`four_aec_dfn_res_get_mem_requirements`, matched delay,
five-filter bank, NE10): 1,440,288 bytes at the 16 kHz default (core plus
the rate bridge) and 3,703,360 bytes at 48 kHz (the conventional
four-channel MMSE-LSA pipeline on the 48 kHz grid needs 3,740,224 bytes).
Model weights and accelerator activation/workspace memory are external;
adding int8 weights alone brings the known minimum to about 3.4 MiB at
16 kHz.
