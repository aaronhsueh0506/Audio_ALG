# 4-channel PBFDKF + beamforming + NR/RES application

This is the conventional four-channel product application directory. It owns
the reusable four-lane linear-AEC front end (`FourAecNrRes`), the one wrapped unit
`AudioPipeline4Ch` (shared delay aligner, four AEC lanes, SRP-PHAT, BF and the `fuse()` projection),
the example programs that chain the post-beam stages over them, host evaluation tools and their tests.
Neither the core nor the unit holds a residual suppressor: RES is managed by libaec. An application
that wants RES creates the `AecRes` itself in its own init (`aec_config_from_preset()`,
`aec_res_get_mem_size()`, `aec_res_init()`) and calls `se_res_stage_gain()` each hop; NR, comfort noise,
synthesis and neural post-filters are likewise separate libraries that the integrator initialises
(each with its own `mem_size` and `init`) and chains, as each example `main` does.
The sibling `../4ch_alignulcnet/` and
`../4ch_aec_bf_dfn_res/` applications reuse the unit and the core; they do not duplicate AEC,
DOA, BF or NR implementations.

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as a synchronous C library.

The core's boundary is intentionally narrow: the post-beam stages are not part of it, and the
unit adds only SRP-PHAT and BF in front of the same `fuse()`:

```text
raw/aligned far ──> one shared aligner (MATCHED, FIXED or EXTERNAL)
                              |
                              v (one aligned reference, identical to every lane)
mic 0 ──> linear AEC 0 ─┐ (computes the far-end FFT: aec_process_context())
mic 1 ──> linear AEC 1 ─┤ (normally borrows lane 0's far_spec: aec_process_context_shared_far_from();
mic 2 ──> linear AEC 2 ─┤  own FFT instead on a mixed warm/soft-realign hop -- see "Far-end FFT
mic 3 ──> linear AEC 3 ─┘  sharing" below)
                        │    four_aec_front_process()
                        └─>[external SRP-PHAT/BF: weights + beamformed error]
                              │    four_aec_fuse()   → SeSpecFrame
                              v
        unit and core: frame.g_res_ctx = NULL (no RES inside either)
        application: se_res_stage_gain(its own AecRes, &frame) → G_res
        G_res ─ se_anr_stage_gain ─ sk_min_f32 ─ sk_capply_gain_f32
              ─ aec_cng_fill ─ ac_istft_push → mono hop          (the application's own code)
```

Each `linear AEC` box still runs its own PBFDKF/shadow filter, residual-echo
(R2) estimate, and `DominantNearend` hold-state update every hop -- the echo
path genuinely differs per microphone, and the next hop's onset-gated R2
depends on that lane's own hold-state. What it does not do is compute its
own mono suppression gain: `AecConfig.spatial_linear_context` (set
internally by this pipeline, not caller-configurable: `AEC_CONTEXT_NO_GAIN` = 1 with
`enable_post = 1`, `AEC_CONTEXT_LINEAR_ONLY` = 2 with `enable_post = 0`) skips straight to the
`DominantNearend` update (or stops after the AEC state update) and leaves `res_gain` `NULL`, so the post-beam RES
gain (`se_res_stage_gain()` over the fused frame, called by the application on its own `AecRes`) is the
only place a suppression gain is calculated.

**Far-end FFT sharing** (one far-end rfft per hop instead of four): all four
lanes see the byte-identical `p->aligned_ref` and realign on the same hop by the
same delta (see the hard invariants below), but
`aec_apply_external_realign()`'s warm-vs-soft outcome is per lane, and a soft
outcome zeros that lane's far_buffer OLA history. When every lane agrees (all
warm, all soft, or no realign), lane 0's far-end FFT is what any lane would
compute: lane 0 runs `aec_process_context()` and lanes 1-3 borrow its `far_spec`
through `aec_process_context_shared_far_from()`. On a mixed warm/soft hop
(`FourAecNrRes.far_spec_mixed_hop`) lanes 1-3 run their own
`aec_process_context()`, so each far_spec reflects the lane's own outcome.
`four_aec_nr_res_far_fft_real_compute_count()` exposes the per-hop FFT count
(1, or 4 on a mixed hop). `ref` is passed to every lane either way: the OLA
far-buffer history and the non-FFT uses of the raw far signal (saturation
detection, delay estimation, mu_scale) are per-lane state.

`fuse_contexts()`'s cross-lane far-spec agreement gate
(`four_aec_nr_res_far_spec_agrees()`, in `4aec_nr_res_internal.h`) is bypassed on
a mixed hop; the fused `X` takes lane 0's far_spec either way
(`frame->x = p->snapshots[0].far_spec` in `four_aec_fuse()`), as in `pipeline.py`'s
`_fuse_contexts()`.

Hard invariants:

- There is at most **one** shared matched-filter bank. MATCHED mode uses
  1..5 filters and a configurable capture proxy; FIXED uses only the exact
  configured-delay ring; EXTERNAL consumes caller-aligned far and allocates
  neither a matcher nor an alignment ring.
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and realign together
  if that shared alignment changes -- this is also the precondition the
  far-end-FFT-sharing scheme above depends on, not a separate guarantee.
- No NR, RES, neural model, or delay estimator is replicated per microphone --
  enforced structurally, not just left unused: each lane keeps only the
  R2/`DominantNearend` state its own echo path needs and never calculates its
  own suppression gain (`spatial_linear_context`; see above).
- `delay_num_filters`, each PBFDKF lane's `filter_length`, and an
  Align-ULCNet checkpoint's TA depth `D` are independent budgets.
- The Python reference and the C front end leave SRP-PHAT/BF at an explicit
  ownership seam. The C unit in `audio_pipeline_4ch.*` connects
  that seam to the reusable implementations under `../../../audiolib/{doa,bf}`
  (a sibling checkout of this repo, shared by every pipeline).
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. `FourAecFront.linear_spectra` are reconstructing 50%-overlap
sqrt-Hann STFT frames and can be consumed directly without four duplicate FFTs.
`FourAecFront.formed_hop[ch]` are the exact selected/crossfaded time-domain hops
underlying those spectra (borrowed from each lane, not copied) for integrations
whose beamformer owns its four analysis FFTs. The per-lane candidate set is
refined, coarse, or -- in this seam's context-only mode -- the lane's own mic
capture, on a hop the filtering-quality analyzer has not cleared; a beamformer
computing its own residual-echo estimate from these hops inherits that
substitution. The Python reference also accepts the external mono hop.

`four_aec_fuse()` takes the weights and, optionally, the mono spectrum the same
beamformer call produced. With `beamformed_error == NULL` it rebuilds the error
from the weights, so independently supplied output samples and context cannot
come from different beamformer states. A beamformer that atomically returns both
its mono spectrum and effective weights passes the spectrum and skips that
duplicate weighted sum; both arrays are validated for finiteness, while their
coherence remains an explicit caller contract. The unit's
`audio_pipeline_4ch_fuse()` uses this form with `bf_process_with_weights()`. The far-end spectrum remains the one
shared digital render reference -- it is verified equal across lanes and is not
spatially weighted. The result passes through one mono NR+RES path without a
fifth AEC or replicated post-filters. When `front.delay.changed` is non-zero, an
external beamformer that owns STFT/OLA state must clear that overlap state
before consuming the frame; the application's own synthesis is not restarted
(the lanes realign their filters in place).

- [4-channel PBFDKF + beamforming + NR/RES application](README.md#4-channel-pbfdkf--beamforming--nrres-application) <!-- split-toc -->
  - [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) <!-- split-toc -->
  - [C API](README_01_c_api.md#c-api) <!-- split-toc -->
  - [Python reference seam](README.md#python-reference-seam) <!-- split-toc -->
## Call sequence and parameter surface

Per-hop lifelines of the two entry points (return values, refusal rules and the hop swimlane: section 0
of [`integration_4ch_core_zh_TW.md`](../../docs/integration_4ch_core_zh_TW.md) and
[`integration_4ch_spatial_zh_TW.md`](../../docs/integration_4ch_spatial_zh_TW.md)):

```
Core + your beamformer      App   FourAecNrRes   your BF     AecRes (yours)  AnrDenoiser  AcIstft
 init (once)                 |-- four_aec_nr_res_default_config / _get_mem_requirements / _init_ex -->|
                             |-- aec_res_*, anr_*, ac_istft_* : each its own mem_size + init ----------------------->|
 per hop                     |-- four_aec_front_process(mics, ref, &front) -->|
                             |-- front.linear_spectra --------------------->| weights [4][n_freqs]
                             |-- four_aec_fuse(weights, NULL|beamformed_error, &frame) -->|  frame.g_res_ctx = NULL
                             |-- se_res_stage_gain --> G_res ; se_anr_stage_gain --> G_nr ; sk_min_f32 ; sk_capply_gain_f32
                             |-- aec_cng_fill ; ac_istft_push --> out (1 hop behind the input)
 refused / non-finite        |-- four_aec_nr_res_reset, aec_res_reset, anr_reset, ac_istft_reset, rng = SE_CNG_RNG_SEED

Unit (SRP-PHAT + BF)        App   AudioPipeline4Ch    AecRes (yours)  AnrDenoiser  AcIstft
 init (once)                 |-- audio_pipeline_4ch_default_config / _get_mem_requirements / _init_ex -->|
                             |-- aec_config_from_preset (fft_size = audio_pipeline_4ch_fft_size(p)), aec_res_*, anr_*, ac_istft_* :
                             |   each its own mem_size + init ----------------------------------------->|
 per hop                     |-- _process_front --> (mask, vad_raw, vad_out) --> _process_beam
                             |       or _process_with_activity, or _process (built-in energy VAD)
                             |-- _fuse(&frame) : frame.g_res_ctx = NULL (the unit has no RES)
                             |-- se_res_stage_gain --> G_res ; se_anr_stage_gain --> G_nr ; sk_min_f32 ; sk_capply_gain_f32
                             |-- aec_cng_fill ; ac_istft_push --> out (1 hop behind the input)
 core.enable_post = 0        |-- no _fuse (INVALID_ARGUMENT), no RES; beam.spectrum --> ulcnet_nn_stage_begin_complex
 refused / non-finite        |-- audio_pipeline_4ch_reset, aec_res_reset, anr_reset, ac_istft_reset, rng = SE_CNG_RNG_SEED
```

| | External (a third party may tune) | Internal (exists, not recommended to open) |
|---|---|---|
| Core (`FourAecNrResConfig`) | `sample_rate`, `fft_size`, `filter_length`, `delay_mode` + `delay_num_filters` / `fixed_delay_samples`, `capture_proxy_channel`, `max_delay_ms`, `enable_post`; opt-in and off by default: `delay_est_period_s`, `delay_backward_quarantine_enabled` / `_s` | the lanes' `AecConfig` (BALANCED preset, `EXTERNAL_ALIGNED`, `enable_res = 0`, `return_res_context = 1`, `spatial_linear_context`), `delay_est_init_s`, `DA_*`, projection kernels |
| Unit (`AudioPipeline4ChConfig`) | `core.*` as above, `geometry` with `uca_radius_m` / `ula_spacing_m` / `microphone_x_m` / `microphone_y_m`, `num_angles`; advanced: `bf_enable`, `bf_fixed_mode` + `bf_fixed_doa_rad`, `doa_low_freq_hz` / `doa_high_freq_hz` | `bf_lambda`, `bf_mu`, `bf_adapt_interval`, `bf_fixed_align_notebook` (the set value differs from the effective one), `doa_enable_smoothing`, `doa_switch_consecutive`, `doa_angle_tolerance_rad`, `doa_update_interval`, `speed_of_sound_m_s`, `auto_vad_*` (only the fallback VAD) |
| Runtime | `aec_res_set_preset()` on your own `AecRes` (unit and core alike), `anr_set_mode(nr, mode)` | `aec_set_preset()` on a lane (a no-op by construction) |

Public headers: `4aec_nr_res.h`, `audio_pipeline_4ch.h`. Internal, do not include:
`4aec_nr_res_internal.h`, `4aec_projection_kernels.h`, `bf.h`, `srp.h`.

## Python reference seam

```python
from importlib import import_module

aec4 = import_module("pipelines.4ch_aec_bf_nr_res")
BeamformerFrame = aec4.BeamformerFrame
FourChannelAecPipeline = aec4.FourChannelAecPipeline

pipeline = FourChannelAecPipeline()      # no beamformer and no RES suppressor are created
suppressor = aec4.PostBeamResidualSuppressor(    # the application's own post-beam RES (the C AecRes)
    pipeline.config.sample_rate, pipeline.hop_size, pipeline.n_freqs)

# 1. Our code: one shared aligner + four linear AEC lanes.
pre = pipeline.front_process(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + BF. The C unit supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_bf.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights.  Like the C
#    audio_pipeline_4ch_fuse(), this stops at the projection: post.context.res_gain is None.
post = pipeline.fuse(
    pre,
    BeamformerFrame(samples=mono_hop, weights=effective_weights),
)
frame = post.spec_frame()               # g_res_ctx = None, like the C frame

# 4. The application's one post-beam RES (the one stateful gain calculator, the C AecRes
#    behind se_res_stage_gain()): res_gain is set on the returned context.
context = suppressor.process(post.context, post.aligned_render)
#    Runtime strength: suppressor.set_preset(...) (C: aec_res_set_preset() on your AecRes).
#    The existing mono NR+RES seam consumes post.beamformed and the returned context.
```

Python shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because the caller's
  `PostBeamResidualSuppressor` has temporal state;
- a queued Python `FrontFrame` owns snapshots of all four contexts, so
  later linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight and resets the lanes and the
  shared delay; the caller resets its own `PostBeamResidualSuppressor` (C: `aec_res_reset()`).

If the external BF is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently projects error and near spectra, converts
each lane's R2 into an echo-phase-bearing residual vector and combines that
vector with the same spatial weights; `fuse()` stops there (`res_gain` is `None` / `g_res_ctx` is `NULL`, for the
unit's `audio_pipeline_4ch_fuse()` as for the core's `four_aec_fuse()`) and the application then runs exactly one
stateful post-beam suppressor (its own `AecRes` in C, `PostBeamResidualSuppressor` in Python) over the
fused frame to calculate the mono gain used after NR. It does
not calculate unused beamformed echo/far spectra. Residual normalization uses
one square root and one divide per channel/bin; the public weights-only path
performs 12 complex MACs per bin (error + near + residual across four lanes),
while the unit's `audio_pipeline_4ch_fuse()` reuses its atomic mono spectrum and performs 8
(near + residual). Comfort-noise power projection is a separate real SIMD
accumulation.

For the fixed four-channel BF, the internal RLS covariance is laid out as
`P[M][M][F]`. This makes frequency contiguous and lets AArch64 NEON update four
bins at once without changing the public API or recurrence order. Automated
tests compare the SIMD and scalar output, effective weights, `P` and `wa`
byte-for-byte after every hop, including reset, masked, tail-bin and non-finite
recovery cases.

`AecResContext` does not export the complete stationarity/AecState surface, so
the post-beam RES omits the stationary mask. R2 itself is **not** bounded. The phase-bearing residual
vector's intermediate is not clamped: a clamp would break the projection's
`|out|^2 == r2` contract — `r2` arrives int16^2-scaled while `echo_spec` is
audio-scaled, so the honest scale routinely exceeds any fixed bound and a
clamp would simply understate residual echo. Only the overflow
fallback guards the intermediates, and it preserves the same magnitude
contract for every finite float32 input.

The core C API does not expose an async multi-frame queue. The C
unit does execute SRP-PHAT/BF and remains synchronous; it supports both
the heap convenience constructor and the caller-owned, zero-heap static-pool
constructor shown in [the C API](README_01_c_api.md#c-api). Automated tests prove caller-pool and heap lifecycle,
byte-identical core construction paths, topology, supported grids, spatial
arithmetic equivalence,
sequencing, and finite DSP output; they do not replace objective speech
metrics or subjective quality sign-off on real array recordings.

DOA and BF follow the same selected no-padding grid as AEC/NR/RES:

| sample rate | frame | FFT | hop | bins | selection |
|---:|---:|---:|---:|---:|---|
| 16 kHz | 256 | 256 | 128 | 129 | default |
| 16 kHz | 512 | 512 | 256 | 257 | `cfg.core.fft_size = 512` |
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

Reproduce the two checked-in recording tests from the `Audio_ALG` directory:

```bash
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_recordings

# Complete C SRP-PHAT/BF path:
make -C pipelines/4ch_aec_bf_nr_res audio_pipeline_4ch_raw
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings

# Exercise both checked-in 16 kHz grids:
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings \
  --fft-size 256
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings \
  --fft-size 512
```

The command is an acceptance test, not only a report generator. It exits
nonzero unless both cases remain finite, preserve the `1 matcher / 4 linear /
1 post-beam RES` resource boundary (the runner reports the `AecRes` of its own chain as `post_res`), acquire a solid nonzero shared delay, and
finish within the declared delay tolerance of the independent file-level
measurement (the C low-latency evaluator uses `max(half hop, 5 ms)`).
Use `--no-contract-check` only when inspecting a deliberately changed fixture.
`evaluate_external_recordings` runs the C unit and the example chain (`audio_pipeline_4ch_raw`: the unit's shared
matcher, four linear AEC lanes, SRP-PHAT and BF, then the example's own `AecRes` and one NR) and verifies
the reported DOA/BF grids. The separate Python `evaluate_recordings` command
uses `EqualWeightBeamformer` only to test the open Python pre/post seam.

The unit's default is a four-microphone UCA with 35 mm radius only so the API
has a runnable example. Product integration must set the measured UCA radius,
ULA spacing, or all four custom `(x, y)` coordinates and confirm channel order.
The checked-in recordings do not carry array-geometry metadata, so their DOA
angles and attenuation report are integration smoke evidence, not a beamformer
quality sign-off.

`woman(ref).wav` and `man.wav` are two stems on one source timeline, while that
timeline and `unprocessed_4ch.wav` have different recording starts. The
evaluator estimates the one fixture offset from the near stem and applies it to
both. It deliberately does **not** align the far stem directly to its echo in
the mic, because doing so would erase the acoustic/system delay that the one
live shared matched filter is supposed to estimate. The C evaluator uses the
same alignment rule, writes raw float hops for `audio_pipeline_4ch_raw`, and checks
the complete `1 matcher / 4 linear AEC / SRP-PHAT / BF / 1 NR / 1 post-RES`
resource and delay contract.
