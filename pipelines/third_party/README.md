# Gain-staging integration

`gain/` holds the input, directional and post-NR gain stages from the
supplier that also provided the SRP-PHAT/DOA smoother, BF and VAD sources
this directory used to vendor. Those moved to the sibling `audiolib` checkout
(`../../../audiolib/{doa,bf,VAD,common/utility}`) in the audio_common ->
audiolib rename (2026-10-20) -- they build and test there now
(`make -C ../../../audiolib SIMD=1 test`), alongside `audiolib/common`'s own
FFT/fast_math/SIMD-kernel layer, whose backend-neutral `Complex` type
(`{float r; float i;}`) `post_gain_apply()` still takes. `gain/` itself
stayed behind: `make -C pipelines` does not link it into any shipped binary
(see `gain/` interfaces below), so it was out of scope for that merge.

`gain/`'s tests moved too, split out of the combined
`tests/test_spatial_third_party.c` (now `audiolib/tests/test_spatial_third_party.c`,
PHAT/BF/SRP/VAD only) into `gain/tests/test_gain.c` alongside a duplicate
copy of the shared `check_pool_contract` helper the two files used to share.
`gain/` currently has no Makefile of its own -- `gain/tests/test_gain.c`
builds and passes (verified manually against `gain/{fix_gain,nr_gain,
post_gain}.c` and `audiolib/common`'s headers/archive), but lost its
automated `make test` runner when `third_party/Makefile` was retired with
the rest of this directory's spatial DSP. Flagging this for a follow-up
decision rather than silently restoring a build target: whether `gain/`
needs a small Makefile of its own, or whether its test coverage is meant to
move somewhere else, was not decided as part of the rename.

`gain/` interfaces:

- `post_gain_apply()` takes `Complex*`; a caller holding an FFT library's own
  complex type of the same layout spells the conversion as a cast at the call
  site.
- `nr_gain_create()`/`nr_gain_process()` take an `NrGainConfig` this module
  owns instead of the application's own runtime-config struct, matching
  `FixGainConfig` and `PostGainConfig` beside them. `nr_gain.h` carries the
  field-by-field mapping. `nr_gain_create()` takes no argument: it never read
  the config it was passed.

`fix_gain`'s public contract is unchanged; it gained the pool constructors its
integrator was already calling.

Two things about `fix_gain` are worth knowing before picking it:

- audiolib's `AudioPreGain` is this stack's canonical input-gain stage and
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

The gain stages are shaped by the same shared kernels as the rest of the
stack (now in `audiolib`): `fix_gain` scales through `sk_scale_f32` and clips
through `sk_clip_f32`, `nr_gain` scales through `sk_scale_f32`, and
`post_gain` clamps and applies its per-bin gain through
`sk_clip_f32`/`sk_capply_gain_f32`. `sk_scale_f32` is the shared
broadcast-multiply kernel; the gain stages call it instead of carrying their
own loops.

Two loops stay open-coded, each for a stated reason rather than by omission:
`post_gain`'s box-average frequency smoother (a running accumulator would
reassociate the sum, so it is split into edge/interior/edge instead, which
vectorises without touching the addition order), and its attack/release
smoother (`sk_asym_ema_f32` selects its falling coefficient on `x < s` where
this module selects attack on `target > prev`, so the two disagree at
`target == prev` -- exactly the steady state a constant-target gain sits in,
and the branches genuinely differ there for 17.0% of values, measured).

The gain tests (`gain/tests/test_gain.c`) carry a transcription of each
module's pre-kernel formulation and require byte-identical output against
it, so the kernel routing above is a refactor and not a retune.

The doa/BF/VAD arithmetic and the 4ch pipeline's full wrapper gate run
through:

```bash
make -C audiolib SIMD=1 test
make -C Audio_ALG/pipelines/4ch_aec_bf_nr_res test
```

(the second delegates to the first for its spatial dependencies; see
`audiolib`'s own README and `4ch_aec_bf_nr_res/README.md`). The complete
pipeline supports `16k/256/128`, `16k/512/256`, and `48k/1024/512`
(`sample-rate/frame/hop`, with frame equal to FFT), one grid at a time --
DOA, BF and AEC/NR/RES all run at the same selected tuple (see
`4ch_aec_bf_nr_res/README.md`). There is no DOA-downsample flag.
