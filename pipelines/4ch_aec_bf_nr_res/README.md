# 4-channel linear AEC + beamformer flow (mcab) and its examples

This directory holds `mcab`, the four-microphone linear-AEC flow (`mcab.h`, `mcab.c`, and the private
`mcab_internal.h`), the sample-layer helpers built beside it (`mcab_sample_flow`, `mcab_sample_beam`,
`mcab_sample_driver`), the example programs that chain the post-beam stages over it (`mcab_external_static`,
`mcab_static`, `mcab_raw`), its Python twin (`pipeline.py`), host evaluation tools and their tests.

`mcab` is a **flow layer, not a library**. The application initialises the four libaec `Aec` lanes and, in
MATCHED mode only, one shared `DelayAec` (each with its own `get_mem_size` / `init`), and passes the pointers in
`McabLibs`; mcab never allocates, resets-by-destroying or frees one of them. It owns only flow state, in its own
small pool: the shared-delay admission state machine, the alignment ring, the aligned far, the per-lane views and
the fuse buffers. There is no beamformer inside it and no residual suppressor: RES is managed by libaec. An
application that wants RES creates the `AecRes` itself in its own init (`aec_config_from_level()`,
`aec_res_get_mem_size()`, `aec_res_init()`) and calls `aec_res_process()` over a `mcab_sample_res_input()` each hop; NR, comfort noise,
synthesis and neural post-filters are likewise separate libraries that the integrator initialises (each with its
own `mem_size` and `init`) and chains, as each example `main` does.
The sibling applications `../dfn/4ch_aec_bf_dfn_res/` and `../ulcnet/4ch_alignulcnet/` build the flow and the beam
helper through `mcab_sample_flow.h` / `mcab_sample_beam.h`; they do not duplicate AEC, DOA, BF or NR
implementations.

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as a synchronous C module.

The flow's boundary is intentionally narrow: the beamformer and the post-beam stages are not part of it.
`mcab_sample_beam.h` (sample code, not in `libaudio_pipeline_mcab.a`) is the optional glue that runs audiolib's
SRP-PHAT and BF between the two calls:

```text
raw/aligned far ──> one shared aligner (MATCHED, FIXED or EXTERNAL_ALIGNED)
                              |
                              v (one aligned reference, identical to every lane)
mic 0 ──> linear AEC 0 ─┐ (computes the far-end FFT: aec_process_context())
mic 1 ──> linear AEC 1 ─┤ (normally borrows lane 0's far_spec: aec_process_context_shared_far_from();
mic 2 ──> linear AEC 2 ─┤  own FFT instead on a mixed warm/soft-realign hop -- see "Far-end FFT
mic 3 ──> linear AEC 3 ─┘  sharing" below)
                        │    mcab_process_front()
                        └─>[caller's SRP-PHAT/BF (or mcab_sample_beam.h): weights + beamformed error]
                              │    mcab_fuse()   → McabFrame
                              v
        frame.g_res_ctx = NULL (no RES inside mcab)
        application: mcab_sample_res_input(&frame, &in); aec_res_process(its own AecRes, &in) → G_res
        G_res ─ mcab_sample_anr_gain ─ sk_min_f32 ─ sk_capply_gain_f32
              ─ aec_cng_fill ─ ac_istft_push → mono hop          (the application's own code)
```

Each `linear AEC` box still runs its own PBFDKF/shadow filter, residual-echo
(R2) estimate, and `DominantNearend` hold-state update every hop -- the echo
path genuinely differs per microphone, and the next hop's onset-gated R2
depends on that lane's own hold-state. What it does not do is compute its
own mono suppression gain: `AecConfig.spatial_linear_context` (stamped by
`mcab_apply_lane_contract()`, `AEC_CONTEXT_NO_GAIN` = 1 with
`enable_post = 1`, `AEC_CONTEXT_LINEAR_ONLY` = 2 with `enable_post = 0`) skips straight to the
`DominantNearend` update (or stops after the AEC state update) and leaves `res_gain` `NULL`, so the post-beam RES
gain (`aec_res_process()` over the fused frame, called by the application on its own `AecRes`) is the
only place a suppression gain is calculated.

**Far-end FFT sharing** (one far-end rfft per hop instead of four): all four
lanes see the byte-identical `p->aligned_ref` and realign on the same hop by the
same delta (see the hard invariants below), but
`aec_apply_external_realign()`'s warm-vs-soft outcome is per lane, and a soft
outcome zeros that lane's far_buffer OLA history. When every lane agrees (all
warm, all soft, or no realign), lane 0's far-end FFT is what any lane would
compute: lane 0 runs `aec_process_context()` and lanes 1-3 borrow its `far_spec`
through `aec_process_context_shared_far_from()`. On a mixed warm/soft hop
(`Mcab.far_spec_mixed_hop`) lanes 1-3 run their own
`aec_process_context()`, so each far_spec reflects the lane's own outcome.
`mcab_far_fft_real_compute_count()` exposes the cumulative far-FFT count over the four lanes
(+1 per hop, or +4 on a mixed hop). `ref` is passed to every lane either way: the OLA
far-buffer history and the non-FFT uses of the raw far signal (saturation
detection, delay estimation, mu_scale) are per-lane state.

`fuse_contexts()`'s cross-lane far-spec agreement gate
(`mcab_far_spec_agrees()`, in `mcab_internal.h`) is bypassed on
a mixed hop; the fused `X` takes lane 0's far_spec either way
(`frame->x = p->snapshots[0].far_spec` in `mcab_fuse()`), as in `pipeline.py`'s
`_fuse_contexts()`.

Hard invariants:

- There is at most **one** shared matched-filter bank. MATCHED mode uses
  1..5 filters and a configurable capture proxy; FIXED uses only the exact
  configured-delay ring; EXTERNAL_ALIGNED consumes caller-aligned far and needs neither a matcher
  (`McabLibs.delay` must be `NULL`) nor an alignment ring.
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and realign together
  if that shared alignment changes -- this is also the precondition the
  far-end-FFT-sharing scheme above depends on, not a separate guarantee.
- No NR, RES, neural model, or delay estimator is replicated per microphone --
  enforced structurally, not just left unused: each lane keeps only the
  R2/`DominantNearend` state its own echo path needs and never calculates its
  own suppression gain (`spatial_linear_context`; see above).
- mcab borrows its lanes and estimator: they must outlive the `Mcab` handle, be freshly initialised or reset
  before `mcab_init()`, and must not be driven or reset by the application while the flow is attached.
- `delay_num_filters`, each PBFDKF lane's `filter_length`, and an
  Align-ULCNet checkpoint's TA depth `D` are independent budgets.
- The Python reference and the C flow leave SRP-PHAT/BF at an explicit
  ownership seam. The C sample helper `mcab_sample_beam.*` connects
  that seam to the reusable implementations under `../../../audiolib/{doa,bf}`
  (a sibling checkout of this repo, shared by every pipeline).
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. `McabFront.linear_spectra` are reconstructing 50%-overlap
sqrt-Hann STFT frames and can be consumed directly without four duplicate FFTs.
`McabFront.formed_hop[ch]` are the exact selected/crossfaded time-domain hops
underlying those spectra (borrowed from each lane, not copied) for integrations
whose beamformer owns its four analysis FFTs. The per-lane candidate set is
refined, coarse, or -- in this seam's context-only mode -- the lane's own mic
capture, on a hop the filtering-quality analyzer has not cleared; a beamformer
computing its own residual-echo estimate from these hops inherits that
substitution. The Python reference also accepts the external mono hop.

`mcab_fuse()` takes the weights and, optionally, the mono spectrum the same
beamformer call produced. With `beamformed_error == NULL` it rebuilds the error
from the weights, so independently supplied output samples and context cannot
come from different beamformer states. A beamformer that atomically returns both
its mono spectrum and effective weights passes the spectrum and skips that
duplicate weighted sum; both arrays are validated for finiteness, while their
coherence remains an explicit caller contract. The helper's
`mcab_sample_beam_fuse()` uses this form with `bf_process_with_weights()`. The far-end spectrum remains the one
shared digital render reference -- it is verified equal across lanes and is not
spatially weighted. The result passes through one mono NR+RES path without a
fifth AEC or replicated post-filters. When `front.delay.changed` is non-zero, an
external beamformer that owns STFT/OLA state must clear that overlap state
before consuming the frame; the application's own synthesis is not restarted
(the lanes realign their filters in place).

- [4-channel linear AEC + beamformer flow (mcab) and its examples](README.md#4-channel-linear-aec--beamformer-flow-mcab-and-its-examples) <!-- split-toc -->
  - [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) <!-- split-toc -->
  - [C API](README_01_c_api.md#c-api) <!-- split-toc -->
  - [Python reference seam](README.md#python-reference-seam) <!-- split-toc -->
## Call sequence and parameter surface

Per-hop lifelines of the two entry points (return values, refusal rules and the hop swimlane: section 0
of [`integration_4ch_core_zh_TW.md`](../../docs/integration_4ch_core_zh_TW.md) and
[`integration_4ch_spatial_zh_TW.md`](../../docs/integration_4ch_spatial_zh_TW.md)):

```
Flow + your beamformer      App   lanes/DelayAec (yours)   Mcab   your BF   AecRes (yours)  AnrDenoiser  AcIstft
 init (once)                 |-- mcab_default_config, mcab_lane_config (or aec_config_from_level + mcab_apply_lane_contract)
                             |-- aec_get_mem_size / aec_init x4 ; mcab_delay_estimator_config + delay_aec_get_mem_size / delay_aec_init (MATCHED)
                             |-- mcab_get_mem_requirements / mcab_init_ex(pool, bytes, &cfg, &libs, &req) -->|
                             |-- aec_res_*, anr_*, ac_istft_* : each its own mem_size + init ----------------------->|
 per hop                     |-- mcab_process_front(mics, ref, &front) -->|
                             |-- front.linear_spectra --------------------->| weights [4][n_freqs]
                             |-- mcab_fuse(weights, NULL|beamformed_error, &frame) -->|  frame.g_res_ctx = NULL
                             |-- mcab_sample_res_input ; aec_res_process --> G_res ; mcab_sample_anr_gain --> G_nr  (all on frame.e) ; sk_min_f32
                             |-- frame.post_reset_serial moved: ac_istft_reset, rng = MCAB_SAMPLE_CNG_RNG_SEED (mcab_sample_post_reset_due)
                             |-- sk_capply_gain_f32 over frame.b ; aec_cng_fill ; ac_istft_push --> out (1 hop behind the input)
 refused / non-finite        |-- mcab_reset (resets the lanes and the estimator through their own reset), aec_res_reset, anr_reset, ac_istft_reset, rng = MCAB_SAMPLE_CNG_RNG_SEED
 teardown                    |-- mcab_destroy, then aec_destroy x4, anr_destroy, then free the memory

Flow + sample beam helper   App   McabSampleBeam (SRP-PHAT + BF)   AecRes (yours)  AnrDenoiser  AcIstft
 init (once)                 |-- the flow as above, then mcab_sample_beam_get_mem_size / mcab_sample_beam_init(mem, bytes, &cfg, &bp, flow)
                             |-- aec_config_from_level (fft_size = mcab_fft_size(flow)), aec_res_*, anr_*, ac_istft_* :
                             |   each its own mem_size + init ----------------------------------------->|
 per hop                     |-- mcab_sample_beam_process_front --> (mask, vad_raw, vad_out) --> mcab_sample_beam_process_beam
                             |       or mcab_sample_beam_process_with_activity, or mcab_sample_beam_process (built-in energy VAD)
                             |-- mcab_sample_beam_fuse(&frame) : frame.g_res_ctx = NULL (no RES inside)
                             |-- mcab_sample_res_input ; aec_res_process --> G_res ; mcab_sample_anr_gain --> G_nr  (all on frame.e) ; sk_min_f32
                             |-- frame.post_reset_serial moved: ac_istft_reset, rng = MCAB_SAMPLE_CNG_RNG_SEED (mcab_sample_post_reset_due)
                             |-- sk_capply_gain_f32 over frame.b ; aec_cng_fill ; ac_istft_push --> out (1 hop behind the input)
 cfg.enable_post = 0         |-- no fuse (INVALID_ARGUMENT), no RES; beam.spectrum (the analysis beam) --> ulcnet_nn_stage_begin_complex
 refused / non-finite        |-- mcab_sample_beam_reset, aec_res_reset, anr_reset, ac_istft_reset, rng = MCAB_SAMPLE_CNG_RNG_SEED
 teardown                    |-- mcab_sample_beam_destroy, mcab_destroy, aec_destroy x4, anr_destroy, then free the memory
```

### Analysis and rendering spectra, post-chain resets

Each lane's library evaluates its own output guard once per hop (lib/aec `aec_get_output_base()`: when the
lane's linear estimate is not usable and its windowed residual carries more energy than its capture, the
lane's rendering base is the capture `E + echo`, else `E`). mcab adds no guard of its own, before or after the
beam. Which spectrum feeds what:

| Consumer | Reads | Field |
|---|---|---|
| SRP-PHAT / BF (advanced once per hop) | the lanes' analysis spectra | `McabFront.linear_spectra` |
| post-beam RES (`aec_res_process`), NR estimator (`mcab_sample_anr_gain`), the DFN2 estimate | the analysis beam `E` and the projected `Y`, `R^2`, `N^2` | `McabFrame.e`, `.y`, `.r2`, `.cn` |
| synthesis (`P = B * min(G_nr, G_res)` + comfort noise), the DFN2 `P` | the rendering base `B = sum_ch W_ch * B_ch` with the same hop's weights | `McabFrame.b` |
| Align-ULCNet on the beam (`enable_post = 0`) | the analysis beam (as a network on one lane reads its error spectrum) | `beam.spectrum` |

`B` is `E` itself (the same pointer) unless a contributing lane's guard fired (`McabFrame.guard_lanes`).
A lane's guard bounds that lane's residual by its capture; the bound is not carried through the beam (complex
weights mix the lanes), and `B` is not the transform of any formed time hop. On the blind-set guard subset
(lanes = one clip delayed 0..3 samples, the sample beam helper's weights) `|B|^2` sits within -0.7..+0.2 dB of
the projected capture `|Y|^2` over the guard hops, while `|E|^2` (what the chain rendered before) sits
+0.02..+2.4 dB above it; single hops with `|B|^2 > |Y|^2` do occur.

`McabFront/McabFrame.post_reset_serial` is the four lanes' `aec_post_reset_serial()` summed modulo 2^32. The
one synthesizer of the fused output keeps the value it last followed; when the frame's value differs it clears
its overlap-add tail and reseeds its comfort noise to `MCAB_SAMPLE_CNG_RNG_SEED` (= `AEC_CNG_SEED`, 42) before
that hop's comfort-noise draw and inverse transform. mcab never resets a lane in response; whether the denoiser
or a network follows is the application's policy (the DFN2 example resets its whole hop: the stage keeps its
synthesis inside). The lanes take an aligned far (`EXTERNAL_ALIGNED`), so in this flow the serial moves only
with `aec_reset()`, i.e. `mcab_reset()`.

| | External (a third party may tune) | Internal (exists, not recommended to open) |
|---|---|---|
| Flow (`McabConfig`) | `sample_rate`, `fft_size`, `delay_mode` + `delay_num_filters` / `fixed_delay_samples`, `capture_proxy_channel`, `max_delay_ms`, `enable_post`; opt-in and off by default: `delay_est_period_s` (a compute knob; a non-zero value can change the lag it picks) | `delay_backward_quarantine_enabled` / `_s` (off by default, kept off in products: it only holds an earlier estimate for one window and does not cure a mis-lock), the lanes' contract fields stamped by `mcab_apply_lane_contract()` (`EXTERNAL_ALIGNED`, `enable_delay_est = 0`, `enable_res = 0`, `return_res_context = 1`, `spatial_linear_context`), `DUTY_HOLD_S`, `DA_*`, projection kernels |
| Lanes (the application's `AecConfig`) | `mcab_lane_config()` = balanced level plus the contract; anything else (level values, `filter_length`) via `aec_config_from_level()` and `mcab_apply_lane_contract()` | the contract fields above |
| Sample beam helper (`McabSampleBeamParams`) | `geometry` with `uca_radius_m` / `ula_spacing_m` / `microphone_x_m` / `microphone_y_m`, `num_angles`; advanced: `bf_enable`, `bf_fixed_mode` + `bf_fixed_doa_rad`, `doa_low_freq_hz` / `doa_high_freq_hz` | `bf_lambda`, `bf_mu`, `bf_adapt_interval`, `bf_fixed_align_notebook` (the set value differs from the effective one), `doa_enable_smoothing`, `doa_switch_consecutive`, `doa_angle_tolerance_rad`, `doa_update_interval`, `speed_of_sound_m_s`, `auto_vad_*` (only the fallback VAD) |
| Runtime | `aec_res_set_level()` on your own `AecRes`, `anr_set_level(nr, level)` | `aec_set_level()` on a lane (a no-op by construction) |

Public header: `mcab.h` (it pulls `aec.h`; `McabFrame` is defined in it). Sample-layer headers (in no archive):
`mcab_sample_flow.h`, `mcab_sample_beam.h`, `mcab_sample_driver.h`, and the header-only `mcab_sample_post.h`
(`mcab_sample_res_input()` fills the `AecResInput` of a fused frame, `mcab_sample_anr_gain()` runs the denoiser on it,
and the `MCAB_SAMPLE_CNG_*` / `MCAB_SAMPLE_PSD_SCALE_INV` constants; it needs `aec_res.h` and `anr.h`). Internal, do not include:
`mcab_internal.h`, `bf.h`, `srp.h`.

## Python reference seam

The Python twin has the same classes under `Mcab*` names: `McabConfig`, `McabPipeline`, `McabFront`, `McabBeam`,
`McabFrame`, `McabDelayState`, plus `mcab_apply_lane_contract()` and `mcab_lane_config()` in `pipeline.py`.
`McabPipeline(config=None, beamformer=None, lane_config=None)` takes an optional application-composed lane
`AecConfig`, stamps the contract onto it, and defaults to `mcab_lane_config()` (a PBFDKF lane with a 52 ms filter at
16 kHz and 64 ms at 48 kHz; the C default lane is the balanced level). The Python grid is chosen by
`McabConfig.frame_size` / `hop_size`, and the Python shared delay is always the MATCHED estimator.
The Python `McabConfig` carries the grid, `capture_proxy_channel`, `max_delay_ms` and the backward-quarantine pair; it
has no `delay_mode`, `delay_num_filters`, `fixed_delay_samples`, `delay_est_period_s` or `enable_post`. The matched-filter
duty cycle (`delay_est_period_s`) is C-only. `tests/test_delay_parity.py` drives the C flow (`dump_delay_parity`) and
`pipeline.py` from one synthetic scene with the duty cycle pinned at full rate and compares the applied delay, `solid`,
`changed` and the served aligned far hop by hop.

```python
from importlib import import_module

mcab = import_module("pipelines.4ch_aec_bf_nr_res")
McabBeam = mcab.McabBeam
McabPipeline = mcab.McabPipeline

pipeline = McabPipeline()      # no beamformer and no RES suppressor are created
suppressor = mcab.PostBeamResidualSuppressor(    # the application's own post-beam RES (the C AecRes)
    pipeline.config.sample_rate, pipeline.hop_size, pipeline.n_freqs)

# 1. Our code: one shared aligner + four linear AEC lanes.
pre = pipeline.process_front(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + BF. The C sample helper supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_bf.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights.  Like the C
#    mcab_fuse(), this stops at the projection: post.context.res_gain is None.
post = pipeline.fuse(
    pre,
    McabBeam(samples=mono_hop, weights=effective_weights),
)

# 4. The application's one post-beam RES (the one stateful gain calculator, the C AecRes
#    behind aec_res_process()): res_gain is set on the returned context.
context = suppressor.process(post.context, post.aligned_render)
#    Runtime strength: suppressor.set_preset(...) (C: aec_res_set_level() on your AecRes).
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
  shared delay; the caller resets its own `PostBeamResidualSuppressor` (C: `aec_res_reset()`).

If the external BF is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently projects error and near spectra, converts
each lane's R2 into an echo-phase-bearing residual vector and combines that
vector with the same spatial weights; `fuse()` stops there (`res_gain` is `None` / `g_res_ctx` is `NULL`, for the
helper's `mcab_sample_beam_fuse()` as for `mcab_fuse()`) and the application then runs exactly one
stateful post-beam suppressor (its own `AecRes` in C, `PostBeamResidualSuppressor` in Python) over the
fused frame to calculate the mono gain used after NR. It does
not calculate unused beamformed echo/far spectra. Residual normalization uses
one square root and one divide per channel/bin; `mcab_fuse()` with `beamformed_error == NULL`
performs 12 complex MACs per bin (error + near + residual across four lanes),
while `mcab_sample_beam_fuse()` reuses the helper's atomic mono spectrum and performs 8
(near + residual). Comfort-noise power projection is a separate real SIMD
accumulation.

For the fixed four-channel BF, the internal RLS covariance is laid out as
`P[M][M][F]` (`audiolib/bf/bf.h`). This makes frequency contiguous and lets AArch64 NEON update four
bins at once without changing the public API or recurrence order. The audiolib test
`run_bf_rls_dispatch_matches_scalar_state` (`audiolib/tests/test_spatial_third_party.c`) compares the
production dispatch with the scalar RLS oracle after every hop, including the four-bin vectors, the scalar tail,
mixed mask groups, skipped adaptation, reset and exceptional per-bin recovery.

`AecResContext` does not export the complete stationarity/AecState surface, so
the post-beam RES omits the stationary mask. R2 itself is **not** bounded. The phase-bearing residual
vector's intermediate is not clamped: a clamp would break the projection's
`|out|^2 == r2` contract — `r2` arrives int16^2-scaled while `echo_spec` is
audio-scaled, so the honest scale routinely exceeds any fixed bound and a
clamp would simply understate residual echo. Only the overflow
fallback guards the intermediates, and it preserves the same magnitude
contract for every finite float32 input.

The C module does not expose an async multi-frame queue. The sample beam helper
does execute SRP-PHAT/BF and remains synchronous; the flow supports the heap convenience
`mcab_create()` (mcab's own pool only) and the caller-owned, zero-heap `mcab_init_ex()` shown in
[the C API](README_01_c_api.md#c-api). Automated tests prove caller-pool and heap lifecycle,
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

**Command lines of the examples.** `mcab_raw`, `mcab_static` and `mcab_external_static` read their numeric
options (`--sample-rate`, `--fft-size`, `--uca-radius-m`, `--fixed-doa-deg`, `--capture-proxy`) with the
strict parsers of `../common/example_cli.h`: the whole string must be one number, so an empty string, a leading
space, trailing characters (`16000Hz`), a value outside `int`, `inf` and `nan` are command-line errors, and so
are an option without its value and an unknown option. A token that starts with `--` where a value belongs is
the next option, so the value is missing: `--nr-level --no-res` is an error, not the level `--no-res` (this holds
for every option that takes a value: numbers, level names and the file options of `mcab_raw`). A lone `-`, `-1`
and `-x` are values: a negative number is one, and a file option takes the name as given (a file named `--x` is
written `./--x`). The three programs exit with code 2 and print a message or the
usage line on stderr. A number that parses but names no grid or channel (`--sample-rate 44100`,
`--capture-proxy 4`) is refused later, when the config is built, and exits 1 in `mcab_static` and
`mcab_external_static`; `mcab_raw` checks the grid while parsing and exits 2. `tests/test_cli_strict.sh <bin_dir>`
runs both static programs through every case, plus the file options of `mcab_raw` and `--erb-fwd`/`--erb-inv` of
`4ch_aec_bf_dfn_res`, which share their option scan (`make test` runs it), and `tests/test_mcab_sample_driver.c` covers the
shared parser.
`mcab_external_static` also refuses an AEC level (the positional argument or `--aec-level <level>`, the same setting) or `--nr-level` name it does not know (`mild|balanced|aggressive|no_floor`
or `0..3`; `mild|moderate|balanced|aggressive` or `0..3`) and a second AEC level (positional or `--aec-level`), with exit 2; `--nr-preset` is the same option as
`--nr-level` under its earlier name; the mono programs (`aec_nr_pipeline`, `aec_nr_pipeline_static`) apply the same one-AEC-level rule. Still lenient: a repeated option keeps the last value. C level == Python preset.

Reproduce the two recording tests from the `Audio_ALG` directory. The evaluators read the case directories
`aec_take_turn` and `aec_together` under `--datasets-root` (default: the `datasets` directory beside `Audio_ALG`,
which is not in git); each needs `unprocessed_4ch.wav`, `man.wav` and `woman(ref).wav` (or `women(ref).wav`):

```bash
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_recordings

# Complete C SRP-PHAT/BF path:
make -C pipelines/4ch_aec_bf_nr_res mcab_raw
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings

# Exercise both 16 kHz grids:
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
`evaluate_external_recordings` runs the C flow and the example chain (`mcab_raw`: the flow's shared
matcher and four linear AEC lanes, the sample helper's SRP-PHAT and BF, then the example's own `AecRes` and one NR) and verifies
the reported DOA/BF grids. The separate Python `evaluate_recordings` command
uses `EqualWeightBeamformer` only to test the open Python pre/post seam.

The sample helper's default is a four-microphone UCA with 35 mm radius only so the API
has a runnable example. Product integration must set the measured UCA radius,
ULA spacing, or all four custom `(x, y)` coordinates and confirm channel order.
The recordings do not carry array-geometry metadata, so their DOA
angles and attenuation report are integration smoke evidence, not a beamformer
quality sign-off.

`woman(ref).wav` and `man.wav` are two stems on one source timeline, while that
timeline and `unprocessed_4ch.wav` have different recording starts. The
evaluator estimates the one fixture offset from the near stem and applies it to
both. It deliberately does **not** align the far stem directly to its echo in
the mic, because doing so would erase the acoustic/system delay that the one
live shared matched filter is supposed to estimate. The C evaluator uses the
same alignment rule, writes raw float hops for `mcab_raw`, and checks
the complete `1 matcher / 4 linear AEC / SRP-PHAT / BF / 1 NR / 1 post-RES`
resource and delay contract.
