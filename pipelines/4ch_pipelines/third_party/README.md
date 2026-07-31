# External spatial DSP integration

`doa/` and `GSC/` contain the supplied SRP-PHAT/DOA smoother and GSC
implementations vendored with this repository and used by
`pipelines/4ch_pipelines/audio_pipeline_4ch.c`.
`utility/complex.*` is their shared complex arithmetic.

`third_party/Makefile` builds three config-keyed reusable archives:

- `libspatial_common.a`: namespaced complex helpers and scalar/NEON kernels;
- `libdoa.a`: geometry, steering, SRP-PHAT, and DOA smoothing;
- `libgsc.a`: the GSC state and processing API.

Their public spectra use `audio_common`'s backend-neutral `Complex` type.
They do not expose KISS types, so the same headers and source-level API work
with `BACKEND=kiss` and `BACKEND=ne10`; `SIMD=0` selects the scalar kernels.

The integration adds two algorithm-preserving facilities:

- `utility/spatial_simd.*` computes each microphone-pair PHAT spectrum once per
  frame and provides an AArch64 NEON path (via audio_common's sanctioned
  `sk__cquad_load`/`sk__cquad_store` complex-quad primitives, not a raw
  pointer-cast load/store). Its test requires sample-for-sample bit identity
  with the scalar implementation.
- GSC routes both the distortionless beamformer and adaptive-canceller
  beamformer through the same scalar/NEON complex kernel. Its small 4-by-4
  per-bin RLS recursion remains scalar because those matrix elements are not
  contiguous across frequency; the two long contiguous frequency passes are
  the accelerated hot path.
- GSC can export the exact effective complex channel weights used to produce
  its mono spectrum. This lets the downstream post-beam RES project all AEC
  contexts with the same spatial state; the original `gsc_process()` API and
  output remain available and are checked against the exporting path.

`GSC/gsc.c`'s per-hop RLS update additionally carries two numerical-hardening
additions that are **not** algorithm-preserving/bit-exact against the
originally supplied recursion (both are plain constants, not exposed as
config): the covariance matrix `P` is re-Hermitianized after each per-bin
update (guards against float32 rounding asymmetry accumulating over long
continuous runtime), and the adaptive weight state `wa` carries a slow leak
(`GSC_WA_LEAK`, queried via `gsc_wa_leak_factor()`; time constant far slower
than the RLS forgetting factor's own adaptation timescale, and only applied
when this bin's RLS update actually runs -- i.e. counted in updates, not raw
hops, since VAD/mask gating means not every hop updates) so it cannot grow
unbounded under sustained non-target conditions with no forgetting mechanism
otherwise. Neither has been
perceptually validated against real recordings; the `lambda` forgetting
factor itself is separately bounded to `(0, 1]` at both `gsc_create()` and
the wrapper's config validation.

Steering tables are precomputed at construction. The 48 kHz pipeline's optional
16 kHz DOA mode builds a separate 48 kHz steering table for GSC without
creating a second SRP scorer. No allocation occurs in SRP/GSC per-frame
processing.

Run the spatial arithmetic and full wrapper gates through:

```bash
make -C Audio_ALG/pipelines/4ch_pipelines/third_party BACKEND=kiss SIMD=1
make -C Audio_ALG/pipelines/4ch_pipelines test_spatial_third_party test_audio_pipeline_4ch
make -C Audio_ALG/pipelines/4ch_pipelines test
```

The complete pipeline supports `16k/256/128`, `16k/512/256`, and
`48k/1024/512` (`sample-rate/frame/hop`, with frame equal to FFT). With the
explicit DOA-downsample flag, only 48 kHz SRP-PHAT changes to
`16k/512/256`; GSC stays on `48k/1024/512`.
