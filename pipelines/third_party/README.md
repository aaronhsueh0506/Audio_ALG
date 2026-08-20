# External spatial DSP and gain-staging integration

`doa/` and `GSC/` contain the supplied SRP-PHAT/DOA smoother and GSC
implementations vendored with this repository and used by
`pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c`.
`utility/complex.*` is their shared complex arithmetic. `gain/` holds the
input, directional and post-NR gain stages from the same supplier.

`third_party/Makefile` builds five config-keyed reusable archives:

- `libspatial_common.a`: namespaced complex helpers and scalar/NEON kernels;
- `libdoa.a`: geometry, steering, SRP-PHAT, and DOA smoothing;
- `libgsc.a`: the GSC state and processing API;
- `libvad.a`: single-channel spectral mask estimation and the mask VAD;
- `libgain.a`: the input fix-gain, directional post-gain and post-NR gain
  compensation stages in `gain/`.

Every archive offers the caller-pool constructor pair the rest of this stack
uses -- `X_get_mem_size()` sizes, `X_init()` carves a caller-owned block, and
`X_destroy()` frees nothing on that path.

Their public spectra use `audio_common`'s backend-neutral `Complex` type.
They do not expose KISS types, so the same headers and source-level API work
with `BACKEND=kiss` and `BACKEND=ne10`; `SIMD=0` selects the scalar kernels.

`gain/` arrived using KISS types and an application-wide runtime-config header
directly, neither of which exists on the `BACKEND=ne10` include path, so two
interfaces changed on the way in. Integrators updating to this copy need:

- `post_gain_apply()` now takes `Complex*` instead of `kiss_fft_cpx*`. The two
  are byte-compatible -- both are `{float r; float i;}` -- so a caller still
  holding a KISS spectrum spells the conversion as a cast at the call site.
- `nr_gain_create()`/`nr_gain_process()` now take an `NrGainConfig` this module
  owns instead of the application's own runtime-config struct, matching
  `FixGainConfig` and `PostGainConfig` beside them. `nr_gain.h` carries the
  field-by-field mapping. `nr_gain_create()` takes no argument: it never read
  the config it was passed.

`fix_gain`'s public contract is unchanged; it gained the pool constructors its
integrator was already calling.

Two things about `fix_gain` are worth knowing before picking it:

- audio_common's `AudioPreGain` is this stack's canonical input-gain stage and
  does the same job for a single gain, with validated dB conversion. `fix_gain`
  is the multi-channel variant -- one gain per microphone plus a clip -- and is
  the right choice only when that per-channel table is what you need. The two
  now shape the signal with the same kernel, so they cannot drift numerically.
- `fix_gain_db_to_linear()` does not validate, inherited from the original.
  Rather than change its signature, `fix_gain_get_mem_size()` refuses any
  config carrying a non-finite gain or clip value, so a bad dB conversion
  fails at construction instead of turning the signal into NaN.

Each stage answers "can I retune between hops?" differently, so it is stated
once here: `fix_gain` takes its config at construction only -- build a new
instance to change gains; `nr_gain` takes its config per call and holds none,
so retuning is free (and `nr_enable` is live state, not tuning, so a cached
config goes stale); `post_gain` takes its config per call but also reads `F`
and `gain_match` at construction to size the pool and seed the smoother, so
`F` is fixed for the instance's life while everything else is per-hop.

The integration adds two algorithm-preserving facilities:

- `utility/spatial_simd.*` computes each microphone-pair PHAT spectrum once per
  frame and provides an AArch64 NEON path (via audio_common's sanctioned
  `sk__cquad_load`/`sk__cquad_store` complex-quad primitives, not a raw
  pointer-cast load/store). Its test requires sample-for-sample bit identity
  with the scalar implementation.
- GSC routes both the distortionless beamformer and adaptive-canceller
  beamformer through the same scalar/NEON complex kernel. The covariance is
  stored internally as `P[M][M][F]`, so the four-channel RLS recursion can
  update four adjacent frequency bins without gather/scatter. The AArch64
  path preserves the scalar channel/matrix accumulation order and avoids FMA,
  reciprocal estimates and horizontal reductions; the test-only scalar
  oracle compares output, effective weights, `P` and `wa` byte-for-byte after
  every hop. Non-four-channel shapes and exceptional bins use the scalar path.
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

Steering tables are precomputed at construction for whichever single grid the
pipeline is running (`srp_cfg.sr`/FFT/hop are the pipeline's own native
values -- see `audio_pipeline_4ch.c`'s `..._doa_sample_rate()`/`..._doa_fft_size()`
getters, which are pure pass-throughs). No allocation occurs in SRP/GSC
per-frame processing.

There is currently no separate/optional 16 kHz DOA mode: this paragraph
previously described one, but no corresponding config field, flag, or
second steering table exists anywhere in this directory or
`audio_pipeline_4ch.c` (confirmed by grep, 2026-08-02) -- treat the two
paragraphs below as aspirational, not implemented. `4ch_aec_bf_nr_res/README.md`'s
"There is no legacy/downsample grid entry" is the accurate statement of the
current design. If a real 16 kHz-DOA-at-48-kHz-pipeline requirement ever
lands, prefer first restricting SRP-PHAT's per-bin PHAT/pair-steering/scratch
work to the frequency band it actually searches (`f_start..f_end`, `srp.c`'s
`pair_steering`/`prealign` loops currently walk the full bin range even
though the search band is much narrower) before adding a second STFT/
downsample domain -- reusing the AEC lanes' existing 48 kHz spectra and
narrowing the band SRP scores is cheaper than a second FFT pipeline, and a
naive stride-pick 48->16 kHz decimation would alias exactly like the
mono/4ch AEC delay-estimator gap documented elsewhere in this repo.

The gain stages are shaped by the same shared kernels as everything else here:
`fix_gain` scales through `sk_scale_f32` and clips through `sk_clip_f32`,
`nr_gain` scales through `sk_scale_f32`, and `post_gain` clamps and applies its
per-bin gain through `sk_clip_f32`/`sk_capply_gain_f32`. `sk_scale_f32` is new,
added for this integration: the broadcast multiply had three open-coded
implementations, one of them a hand-written NEON loop in `audio_common`'s own
`audio_pre_gain.c` guarded by a private copy of the `SK_HAVE_NEON` predicate.
That copy is gone and all three sites now share one kernel.

Two loops stay open-coded, each for a stated reason rather than by omission:
`post_gain`'s box-average frequency smoother (a running accumulator would
reassociate the sum, so it is split into edge/interior/edge instead, which
vectorises without touching the addition order), and its attack/release
smoother (`sk_asym_ema_f32` selects its falling coefficient on `x < s` where
this module selects attack on `target > prev`, so the two disagree at
`target == prev` -- exactly the steady state a constant-target gain sits in,
and the branches genuinely differ there for 17.0% of values, measured).

These modules own their own tests: `make -C pipelines/third_party test` builds
the GSC_TESTING object, runs the equivalence suite and runs
`audit-no-gsc-test-symbols`, which proves no test-only symbol reached a
deployable archive. The 4ch pipeline's `make test` delegates to it. The gain
tests carry a transcription of each module's pre-kernel formulation and require
byte-identical output against it, so the kernel routing above is a refactor and
not a retune; the suite runs under `SIMD=0` and `SIMD=1` alike, which makes the
same comparison the scalar-versus-NEON gate.

Run the spatial arithmetic and full wrapper gates through:

```bash
make -C Audio_ALG/pipelines/third_party BACKEND=kiss SIMD=1 test
make -C Audio_ALG/pipelines/4ch_aec_bf_nr_res test
```

The complete pipeline supports `16k/256/128`, `16k/512/256`, and
`48k/1024/512` (`sample-rate/frame/hop`, with frame equal to FFT), one grid
at a time -- DOA, GSC and AEC/NR/RES all run at the same selected tuple
(see `4ch_aec_bf_nr_res/README.md`). There is no DOA-downsample flag; the
sentence that used to describe one above is corrected.
