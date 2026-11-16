# 4-channel linear AEC + beamformer flow and its examples

The four-microphone linear-AEC flow is an instance of the AEC library (`AecConfig.n_mics = AEC_ARRAY_MICS`, `aec.h`,
`AEC/c_impl/src/aec_array.c`); this directory holds the sample-layer helpers built beside it (`array_sample_beam`,
`array_sample_driver`, header-only `array_sample_post.h` and `array_sample_unit.h`), the example programs that chain the post-beam
stages over it (`4ch_external_static`, `4ch_static`, `4ch_raw`), its Python twin (`pipeline.py`) and their tests.

The array is **one `Aec`, one pool**. A single `aec_get_mem_requirements()` / `aec_init_ex()` builds the shared delay
estimator (MATCHED mode), the alignment ring, the aligned far, four lanes (each an `EXTERNAL_ALIGNED` single-channel
instance), the fusion buffers and, with `AecConfig.residual = AEC_RESIDUAL_FULL`, the one residual gain stage. The
application supplies no lane and no estimator. There is no beamformer inside it: the caller reads the lanes' error
spectra from `aec_get_array()`, runs its own SRP-PHAT/BF and hands the weights to `aec_process_residual()`. NR,
synthesis and neural post-filters are separate libraries that the integrator initialises (each with its own `mem_size`
and `init`) and chains, as each example `main` does; the gain and the comfort noise come from `aec_apply_gain()` and
`aec_apply_cng()` on the array. `AecConfig.residual` selects how far the array goes: `STATE` (lanes only, for a network
that reads the beamformed spectrum itself), `ESTIMATE` (fusion, no gain) or `FULL`.
The sibling applications `aec_bf_dfn_res/` and `alignulcnet/` place the array with `aec_get_mem_requirements()` /
`aec_init_ex()` and build the beam helper (`array_sample_beam.h`) on it; they do not duplicate AEC, DOA, BF or NR
implementations.

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as the array instance of lib/aec.

The array's boundary is intentionally narrow: the beamformer and the post-beam stages are not part of it.
`array_sample_beam.h` (sample code, in no archive) is the optional glue that runs audiolib's
SRP-PHAT and BF between the two calls:

```text
raw/aligned far ──> one shared aligner (MATCHED, FIXED or EXTERNAL_ALIGNED)
                              |
                              v (one aligned reference, identical to every lane)
mic 0 ──> linear AEC 0 ─┐ (computes the far-end FFT)
mic 1 ──> linear AEC 1 ─┤ (normally borrows lane 0's far_spec;
mic 2 ──> linear AEC 2 ─┤  own FFT instead on a mixed warm/soft-realign hop -- see "Far-end FFT
mic 3 ──> linear AEC 3 ─┘  sharing" below)
                        │    aec_process_linear() + aec_get_array()   → AecArrayView
                        └─>[caller's SRP-PHAT/BF (or array_sample_beam.h): weights + beamformed error]
                              │    aec_process_residual(&AecBeam{weights, beamformed_error}) + aec_get_residual()   → AecResidualView
                              v
        the array's one gain stage (residual = FULL): G_res over the fused spectra   → AecResidualView.res_gain
        application: array_sample_anr_gain (NR on error_spec) ─ aec_apply_gain(aec, G_nr, P) ─ aec_apply_cng(aec, G_nr, P)
              ─ ac_istft_push → mono hop          (the application's own code)
```

Each `linear AEC` box still runs its own PBFDKF/shadow filter, residual-echo
(R2) estimate, and `DominantNearend` hold-state update every hop -- the echo
path genuinely differs per microphone, and the next hop's onset-gated R2
depends on that lane's own hold-state. What it does not do is compute its
own mono suppression gain: the lanes run at `AEC_RESIDUAL_ESTIMATE` (`STATE` when the array is `STATE`), which
produces R2 and the comfort-noise PSD but no `res_gain`, so the array's gain stage (fed the fused frame by
`aec_process_residual()`) is the only place a suppression gain is calculated.

**Far-end FFT sharing** (one far-end rfft per hop instead of four): all four
lanes see the byte-identical aligned reference and realign on the same hop by the
same delta (see the hard invariants below), but
the warm-vs-soft outcome of a lane's realign is per lane, and a soft
outcome zeros that lane's far_buffer OLA history. When every lane agrees (all
warm, all soft, or no realign), lane 0's far-end FFT is what any lane would
compute: lane 0 runs the far FFT and lanes 1-3 borrow its `far_spec`. On a mixed warm/soft hop
lanes 1-3 run their own FFT, so each far_spec reflects the lane's own outcome.
`aec_far_fft_real_compute_count()` exposes the cumulative far-FFT count over the four lanes
(+1 per hop, or +4 on a mixed hop). `ref` is passed to every lane either way: the OLA
far-buffer history and the non-FFT uses of the raw far signal (saturation
detection, delay estimation, mu_scale) are per-lane state.

The fusion's cross-lane far-spec agreement gate is bypassed on
a mixed hop; the fused `far_spec` takes lane 0's either way, as in `pipeline.py`'s
`_fuse_contexts()`.

Hard invariants:

- There is at most **one** shared matched-filter bank. MATCHED mode uses
  1..5 filters and a configurable capture proxy; FIXED uses only the exact
  configured-delay ring; EXTERNAL_ALIGNED consumes caller-aligned far and needs neither a matcher
  nor an alignment ring (neither is carved).
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and realign together
  if that shared alignment changes -- this is also the precondition the
  far-end-FFT-sharing scheme above depends on, not a separate guarantee.
- No NR, RES, neural model, or delay estimator is replicated per microphone --
  enforced structurally, not just left unused: each lane keeps only the
  R2/`DominantNearend` state its own echo path needs and never calculates its
  own suppression gain (see above).
- The lanes and the estimator are inside the array's pool and have no public handle: they cannot be driven or reset
  by the application, only through `aec_reset()` on the array.
- `delay_num_filters`, each PBFDKF lane's filter length (set by the sample rate, not by the application), and an
  Align-ULCNet checkpoint's TA depth `D` are independent budgets.
- The Python reference and the C flow leave SRP-PHAT/BF at an explicit
  ownership seam. The C sample helper `array_sample_beam.*` connects
  that seam to the reusable implementations under `../../../audiolib/{doa,bf}`
  (a sibling checkout of this repo, shared by every pipeline).
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. `AecArrayView.lane_error_spec` are reconstructing 50%-overlap
sqrt-Hann STFT frames and can be consumed directly without four duplicate FFTs.
`AecArrayView.lane_formed_hop[ch]` are the exact selected/crossfaded time-domain hops
underlying those spectra (borrowed from each lane, not copied) for integrations
whose beamformer owns its four analysis FFTs. The per-lane candidate set is
refined or coarse, so `lane_formed_hop[ch]` is always the selected linear residual; the
output guard only chooses the rendering base and never replaces it. The Python
reference also accepts the external mono hop.

`aec_process_residual()` takes the weights and, optionally, the mono spectrum the same
beamformer call produced (`AecBeam.beamformed_error`). With `beamformed_error == NULL` it rebuilds the error
from the weights, so independently supplied output samples and context cannot
come from different beamformer states. A beamformer that atomically returns both
its mono spectrum and effective weights passes the spectrum and skips that
duplicate weighted sum; both arrays are validated for finiteness, while their
coherence remains an explicit caller contract. The helper's
`array_sample_beam_fuse()` uses this form with `bf_process_with_weights()`. The far-end spectrum remains the one
shared digital render reference -- it is verified equal across lanes and is not
spatially weighted. The result passes through one mono NR+RES path without a
fifth AEC or replicated post-filters. When `front.delay.changed` is non-zero, an
external beamformer that owns STFT/OLA state must clear that overlap state
before consuming the frame; the application's own synthesis is not restarted
(the lanes realign their filters in place).

- [4-channel linear AEC + beamformer flow and its examples](README.md#4-channel-linear-aec--beamformer-flow-and-its-examples) <!-- split-toc -->
  - [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) <!-- split-toc -->
  - [C API](reference/c-api.md#c-api) <!-- split-toc -->
  - [Python reference seam](README.md#python-reference-seam) <!-- split-toc -->
## Call sequence and parameter surface

Per-hop lifelines of the two entry points (return values, refusal rules and the hop swimlane: section 0
of [四麥克風 array](../../docs/reference/array/README.md) and
[SRP／BF helper](../../docs/reference/array/beam.md)):

```
Array + your beamformer     App   Aec array (n_mics = 4)   your BF   AnrDenoiser  AcIstft
 init (once)                 |-- aec_config_from_level (or array_sample_default_config), n_mics = AEC_ARRAY_MICS, residual
                             |-- aec_get_mem_requirements / aec_init_ex(pool, bytes, &cfg, &req) -->|
                             |-- anr_*, ac_istft_* : each its own mem_size + init ----------------------->|
 per hop                     |-- aec_process_linear(mics, ref) ; aec_get_array(&front) -->|
                             |-- front.lane_error_spec --------------------->| weights [4][n_freqs]
                             |-- aec_process_residual(&{weights, NULL|beamformed_error}) ; aec_get_residual(&v)   (residual STATE: not called)
                             |-- array_sample_anr_gain --> G_nr  (on v.error_spec = E)
                             |-- v.post_reset_serial moved: ac_istft_reset (array_sample_post_reset_due)
                             |-- aec_apply_gain(aec, G_nr, P) ; aec_apply_cng(aec, G_nr, P) ; ac_istft_push --> out (1 hop behind the input)
 refused / non-finite        |-- aec_reset (resets the lanes, the estimator and the gain stage), anr_reset, ac_istft_reset
 teardown                    |-- aec_destroy, anr_destroy, then free the memory

Array + sample beam helper  App   ArraySampleBeam (SRP-PHAT + BF)   AnrDenoiser  AcIstft
 init (once)                 |-- the array as above, then array_sample_beam_get_mem_size / array_sample_beam_init(mem, bytes, &cfg, &bp, aec)
                             |-- anr_*, ac_istft_* : each its own mem_size + init ----------------------->|
 per hop                     |-- array_sample_beam_process_front --> (mask, vad_raw, vad_out) --> array_sample_beam_process_beam
                             |       or array_sample_beam_process_with_activity, or array_sample_beam_process (built-in energy VAD)
                             |-- array_sample_beam_fuse(&v) : aec_process_residual + aec_get_residual
                             |-- array_sample_anr_gain --> G_nr  (on v.error_spec)
                             |-- v.post_reset_serial moved: ac_istft_reset (array_sample_post_reset_due)
                             |-- aec_apply_gain ; aec_apply_cng ; ac_istft_push --> out (1 hop behind the input)
 cfg.residual = STATE        |-- no fuse (AEC_ERR_MODE), no gain; beam.spectrum (the analysis beam) --> ulcnet_stream_pre_process_freq
 refused / non-finite        |-- array_sample_beam_reset, anr_reset, ac_istft_reset
 teardown                    |-- array_sample_beam_destroy, aec_destroy, anr_destroy, then free the memory
```

### Analysis and rendering spectra, post-chain resets

Each lane's library evaluates its own output guard once per hop (lib/aec: when the
lane's linear estimate is not usable and its windowed residual carries more energy than its capture, the
lane's rendering base is the capture `E + echo`, else `E`). The array adds no guard of its own, before or after the
beam. Which spectrum feeds what:

| Consumer | Reads | Field |
|---|---|---|
| SRP-PHAT / BF (advanced once per hop) | the lanes' analysis spectra | `AecArrayView.lane_error_spec` |
| the array's gain stage, NR estimator (`array_sample_anr_gain`), the DFN2 estimate | the analysis beam `E` and the projected `near`, `R^2`, `N^2` | `AecResidualView.error_spec`, `.near_spec`, `.r2`, `.comfort_noise` |
| synthesis (`P = B * min(G_nr, G_res)` + comfort noise), the DFN2 `P` | the rendering base `B = sum_ch W_ch * B_ch` with the same hop's weights | `AecResidualView.base_spec` |
| Align-ULCNet on the beam (`residual = STATE`) | the analysis beam (as a network on one lane reads its error spectrum) | `beam.spectrum` |

`B` is `E` itself (the same pointer) unless a contributing lane's guard fired (`AecResidualView.guard_lanes`).
A lane's guard bounds that lane's residual by its capture; the bound is not carried through the beam (complex
weights mix the lanes), and `B` is not the transform of any formed time hop. On the blind-set guard subset
(lanes = one clip delayed 0..3 samples, the sample beam helper's weights) `|B|^2` sits within -0.7..+0.2 dB of
the projected capture `|Y|^2` over the guard hops, while `|E|^2` (what the chain rendered before) sits
+0.02..+2.4 dB above it; single hops with `|B|^2 > |Y|^2` do occur.

`AecArrayView/AecResidualView.post_reset_serial` is the four lanes' `aec_post_reset_serial()` summed modulo 2^32. The
one synthesizer of the fused output keeps the value it last followed; when the view's value differs it clears
its overlap-add tail before that hop's inverse transform (`array_sample_post_reset_due()`); the array reseeds its own
comfort-noise generator to `AEC_CNG_SEED` (42) before that hop's draw. The array never resets a lane in response; whether the
denoiser or a network follows is the application's policy (the DFN2 example clears only its own synthesis, on the hop the
event reaches it at: the model keeps running). The lanes take an aligned far (`EXTERNAL_ALIGNED`), so in this flow the serial moves only
with `aec_reset()`.

| | External (a third party may tune) | Internal (exists, not recommended to open) |
|---|---|---|
| Array (`AecConfig`, 12 public fields) | `sample_rate`, `fft_size`, `delay_mode` + `delay_num_filters` / `fixed_delay_samples`, `capture_proxy_channel`, `residual`, `level` (`aec_config_from_level()`, `aec_set_level()`), `enable_cng` | `delay_backward_quarantine_enabled` / `_s` (off by default, kept off in products: it only holds an earlier estimate for one window and does not cure a mis-lock), the lanes' derived fields (`EXTERNAL_ALIGNED`, lane `residual`); everything else is the library's own tuning (`AecTuning`: filter length, delay ring range, duty cycle, shadow, thresholds) and is not settable |
| Sample beam helper (`ArraySampleBeamParams`) | `geometry` with `uca_radius_m` / `ula_spacing_m` / `microphone_x_m` / `microphone_y_m`, `num_angles`; advanced: `bf_enable`, `bf_fixed_mode` + `bf_fixed_doa_rad`, `doa_low_freq_hz` / `doa_high_freq_hz` | `bf_lambda`, `bf_mu`, `bf_adapt_interval`, `bf_fixed_align_notebook` (the set value differs from the effective one), `doa_enable_smoothing`, `doa_switch_consecutive`, `doa_angle_tolerance_rad`, `doa_update_interval`, `speed_of_sound_m_s`, `auto_vad_*` (only the fallback VAD) |
| Runtime | `aec_set_level()` on the array (its one gain stage), `anr_set_level(nr, level)` | a level on a lane (a no-op by construction: the lanes have no handle) |

Public header: `aec.h` (the "Array instance" section: `AecBeam`, `AecArrayView`, `AecResidualView`, `AecDelayState`, `aec_array_*`).
Sample-layer headers (in no archive):
`array_sample_beam.h` (with `array_sample_default_config()`), `array_sample_driver.h`, and the header-only `array_sample_post.h`
and `array_sample_unit.h` (the array + helper unit and the stages behind the fuse, shared by `4ch_raw` and `4ch_static`)
(`array_sample_anr_gain()` runs the denoiser on a fused view, `array_sample_post_reset_due()` and `ARRAY_SAMPLE_PSD_SCALE_INV`; it needs `anr.h`).
Internal, do not include: `aec_array_internal.h`, `aec_view_internal.h`, `bf.h`, `srp.h`.

## Python reference seam

The Python twin has the same classes under `Mcab*` names: `McabConfig`, `McabPipeline`, `McabFront`, `McabBeam`,
`McabFrame`, `McabDelayState`, plus `mcab_apply_lane_contract()` and `mcab_lane_config()` in `pipeline.py`.
`McabPipeline(config=None, beamformer=None, lane_config=None)` takes an optional application-composed lane
`AecConfig`, stamps the contract onto it, and defaults to `mcab_lane_config()` (a PBFDKF lane with a 52 ms filter at
16 kHz and 64 ms at 48 kHz; the C default lane is the balanced level). The Python grid is chosen by
`McabConfig.frame_size` / `hop_size`, and the Python shared delay is always the MATCHED estimator.
The Python `McabConfig` carries the grid, `capture_proxy_channel`, `max_delay_ms` (1024 ms, Python only) and the backward-quarantine pair; it
has no `delay_mode`, `delay_num_filters`, `fixed_delay_samples`, `delay_est_period_s` or `enable_post` (C: `AecConfig.residual`). The matched-filter
duty cycle is C-only and internal to the library: its tuning leaves it off (every hop is analysed) and `AecConfig` cannot set it. `tests/test_delay_parity.py` drives the C flow (`dump_delay_parity`) and
`pipeline.py` from one synthetic scene and compares the applied delay, `solid`,
`changed` and the served aligned far hop by hop.

```python
from importlib import import_module

mcab = import_module("pipelines.4ch")
McabBeam = mcab.McabBeam
McabPipeline = mcab.McabPipeline

pipeline = McabPipeline()      # no beamformer and no RES suppressor are created
suppressor = mcab.PostBeamResidualSuppressor(    # the post-beam RES (C: the array's one gain stage, AecConfig.residual = FULL)
    pipeline.config.sample_rate, pipeline.hop_size, pipeline.n_freqs)

# 1. Our code: one shared aligner + four linear AEC lanes.
pre = pipeline.process_front(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + BF. The C sample helper supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_bf.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights.  Like the C
#    aec_process_residual() with AEC_RESIDUAL_ESTIMATE, this stops at the projection: post.context.res_gain is None.
post = pipeline.fuse(
    pre,
    McabBeam(samples=mono_hop, weights=effective_weights),
)

# 4. The one post-beam RES (the one stateful gain calculator; in C the array's gain stage
#    inside aec_process_residual() with AEC_RESIDUAL_FULL): res_gain is set on the returned context.
context = suppressor.process(post.context, post.aligned_render)
#    Runtime strength: suppressor.set_preset(...) (C: aec_set_level() on the array).
#    The existing mono NR+RES seam consumes post.beamformed and the returned context.
```

Python shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because the caller's
  `PostBeamResidualSuppressor` has temporal state;
- a queued Python `McabFront` owns snapshots of all four contexts, so
  later linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight and resets the lanes and the
  shared delay; the caller resets its own `PostBeamResidualSuppressor` (C: `aec_reset()` resets the array's gain stage with the rest).

If the external BF is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently projects error and near spectra, converts
each lane's R2 into an echo-phase-bearing residual vector and combines that
vector with the same spatial weights; `fuse()` stops there (`res_gain` is `None`; in C the gain stage that follows the projection inside
`aec_process_residual()` exists only with `AEC_RESIDUAL_FULL`, for the helper's `array_sample_beam_fuse()` as
for a direct call) and exactly one stateful post-beam suppressor (the array's gain stage in C,
`PostBeamResidualSuppressor` in Python) runs over the fused frame to calculate the mono gain used after NR. It does
not calculate unused beamformed echo/far spectra. Residual normalization uses
one square root and one divide per channel/bin; `aec_process_residual()` with `beamformed_error == NULL`
performs 12 complex MACs per bin (error + near + residual across four lanes),
while `array_sample_beam_fuse()` reuses the helper's atomic mono spectrum and performs 8
(near + residual). Comfort-noise power projection is a separate real SIMD
accumulation.

For the fixed four-channel BF, the internal RLS covariance is laid out as
`P[M][M][F]` (`audiolib/bf/bf.h`). This makes frequency contiguous and lets AArch64 NEON update four
bins at once without changing the public API or recurrence order. The audiolib test
`run_bf_rls_dispatch_matches_scalar_state` (`audiolib/tests/test_spatial_third_party.c`) compares the
production dispatch with the scalar RLS oracle after every hop, including the four-bin vectors, the scalar tail,
mixed mask groups, skipped adaptation, reset and exceptional per-bin recovery.

The lanes' residual views do not export the complete stationarity/AecState surface, so
the post-beam RES omits the stationary mask. R2 itself is **not** bounded. The phase-bearing residual
vector's intermediate is not clamped: a clamp would break the projection's
`|out|^2 == r2` contract — `r2` arrives int16^2-scaled while `echo_spec` is
audio-scaled, so the honest scale routinely exceeds any fixed bound and a
clamp would simply understate residual echo. Only the overflow
fallback guards the intermediates, and it preserves the same magnitude
contract for every finite float32 input.

The array instance does not expose an async multi-frame queue. The sample beam helper
does execute SRP-PHAT/BF and remains synchronous; the array supports the heap convenience
`Aec* aec_create(const AecConfig*)` (freed by `aec_destroy()`) and the caller-owned, zero-heap `aec_init_ex()` shown in
[the C API](reference/c-api.md#c-api). Automated tests prove caller-pool and heap lifecycle,
byte-identical construction paths, topology, supported grids, spatial
arithmetic equivalence,
sequencing, and finite DSP output; they do not replace objective speech
metrics or subjective quality sign-off on real array recordings.

DOA and BF follow the same selected no-padding grid as AEC/NR/RES:

| sample rate | frame | FFT | hop | bins | selection |
|---:|---:|---:|---:|---:|---|
| 16 kHz | 256 | 256 | 128 | 129 | default |
| 16 kHz | 512 | 512 | 256 | 257 | `cfg.fft_size = 512` |
| 48 kHz | 1024 | 1024 | 512 | 513 | default/only main grid |

Frame and hop are derived from the selected FFT rather than independently
configured: `frame == FFT`, `hop == frame/2`. SRP receives the matching
sample rate, FFT and bin count for steering; BF operates on the exact
`FFT/2+1` spectra exported by the four linear AEC lanes. The raw runner
selects the same tuple with `--fft-size 256|512|1024` and rejects cross-rate
combinations.

DOA, BF and AEC/NR/RES always use the same selected tuple, so changing the grid
changes the complete pipeline atomically; there is no second spatial timing
domain.

**Command lines of the examples.** `4ch_raw`, `4ch_static` and `4ch_external_static` read their numeric
options (`--sample-rate`, `--fft-size`, `--uca-radius-m`, `--fixed-doa-deg`, `--capture-proxy`) with the
strict parsers of `../example_cli.h`: the whole string must be one number, so an empty string, a leading
space, trailing characters (`16000Hz`), a value outside `int`, `inf` and `nan` are command-line errors, and so
are an option without its value and an unknown option. A token that starts with `--` where a value belongs is
the next option, so the value is missing: `--nr-level --no-res` is an error, not the level `--no-res` (this holds
for every option that takes a value: numbers, level names and the file options of `4ch_raw`). A lone `-`, `-1`
and `-x` are values: a negative number is one, and a file option takes the name as given (a file named `--x` is
written `./--x`). The three programs exit with code 2 and print a message or the
usage line on stderr. A number that parses but names no grid or channel (`--sample-rate 44100`,
`--capture-proxy 4`) is refused later, when the config is built, and exits 1 in `4ch_static` and
`4ch_external_static`; `4ch_raw` checks the grid while parsing and exits 2. `tests/test_cli_strict.sh <bin_dir>`
runs both static programs through every case, plus the file options of `4ch_raw` and `--erb-fwd`/`--erb-inv` of
`4ch_aec_bf_dfn_res`, which share their option scan (`make test` runs it), and `tests/test_array_sample_driver.c` covers the
shared parser.
`4ch_external_static` also refuses an AEC level (the positional argument or `--aec-level <level>`, the same setting) or `--nr-level` value that is not a whole decimal `0..3`
(AEC: mild, balanced, aggressive, no_floor; NR: mild, moderate, balanced, aggressive; a name is refused) and a second AEC level (positional or `--aec-level`), with exit 2; `--nr-preset` is the same option as
`--nr-level` under its earlier name; the mono programs (`aec_nr_pipeline`, `aec_nr_pipeline_static`) apply the same one-AEC-level rule. Still lenient: a repeated option keeps the last value. C level == Python preset.

The sample helper's default is a four-microphone UCA with 35 mm radius only so the API
has a runnable example. Product integration must set the measured UCA radius,
ULA spacing, or all four custom `(x, y)` coordinates and confirm channel order.
