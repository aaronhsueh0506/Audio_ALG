# Audio DSP library family — internal reference

Scope: `audio_common`, `AEC`, `NR`, `Audio_ALG/pipelines` under `SE/`. Written from the code as it stands as of 2026-08-03 (dated comments below are the *files'* own dates, not claims about when this doc was written).

---

## 1. `audio_common` — shared DSP infrastructure

### Purpose

One shared, vendored-once copy of the low-level DSP primitives every other repo (AEC, NR, Audio_ALG) links against: an FFT wrapper with two interchangeable backends (KISS FFT / NE10, both NEON-capable), header-only `fast_math`/`simd_kernels` approximations, a biquad HPF, an amplitude-dB pre-gain, and a stateful rational polyphase audio resampler. Everything here builds with `-ffp-contract=off` under a repo-wide policy shared with AEC/NR/Audio_ALG.

### Current default configuration

Not applicable in the same sense as AEC/NR — this library has no single sample-rate/grid default of its own; `fft_get_mem_size`/`fft_create` accept any power-of-two `fft_size` in `[16, 8192]`. The resampler's own rate whitelist is fixed: `{8000, 16000, 24000, 32000, 48000}` Hz in any pair, 1–8 interleaved channels (`AUDIO_RESAMPLER_MAX_CHANNELS`).

### Public API

**FFT wrapper** (`include/fft_wrapper.h`):

```c
FftHandle* fft_create(int fft_size);                                   // heap
size_t     fft_get_mem_size(int fft_size);                             // static
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);         // static
void       fft_destroy(FftHandle* handle);
int        fft_get_n_freqs(const FftHandle* handle);
void fft_forward(FftHandle* h, const float* restrict time_in, Complex* restrict freq_out);
void fft_inverse(FftHandle* h, const Complex* restrict freq_in, float* restrict time_out);
void fft_forward_scratch(FftHandle* h, float* time_in_clobbered, Complex* complex_out);
void fft_inverse_scratch(FftHandle* h, Complex* freq_in_clobbered, float* real_out);
void fft_magnitude(const Complex* freq, float* magnitude, int n_freqs);
void fft_power(const Complex* freq, float* power, int n_freqs);
void fft_apply_gain(Complex* freq, const float* gain, int n_freqs);
```

Usage (heap path):

```c
FftHandle* fft = fft_create(512);
fft_forward(fft, time_in, spec);      /* time_in[512] -> spec[257] */
fft_inverse(fft, spec, time_out);
fft_destroy(fft);
```

`fft_phase()` / `fft_from_mag_phase()` are declared and implemented in both backends but have **no caller anywhere in AEC/NR/Audio_ALG/audio_common** today.

**Resampler** (`include/audio_resampler.h`):

```c
int audio_resampler_rate_supported(int sample_rate);
AudioResampler* audio_resampler_create(int input_rate, int output_rate, int channels);
size_t audio_resampler_get_mem_size(int input_rate, int output_rate, int channels);
AudioResampler* audio_resampler_init(void* mem, size_t mem_size,
                                     int input_rate, int output_rate, int channels);
void audio_resampler_destroy(AudioResampler* self);
void audio_resampler_reset(AudioResampler* self);
int audio_resampler_process(AudioResampler* self, const float* input, int input_frames,
                            float* output, int output_capacity_frames,
                            int* consumed_frames, int* produced_frames);
int audio_resampler_output_bound(const AudioResampler* self, int input_frames);
int audio_resampler_input_rate(const AudioResampler* self);
int audio_resampler_output_rate(const AudioResampler* self);
int audio_resampler_channels(const AudioResampler* self);
int audio_resampler_latency_input_frames(const AudioResampler* self);
```

Usage:

```c
AudioResampler *rs = audio_resampler_create(48000, 16000, 4);
int capacity = audio_resampler_output_bound(rs, input_frames);
int consumed, produced;
audio_resampler_process(rs, input_4ch, input_frames, output_4ch, capacity,
                        &consumed, &produced);
audio_resampler_destroy(rs);
```

Unequal-rate input/output buffers must not overlap; equal-rate conversion is an exact `memmove` pass-through; state persists across `process()` calls so arbitrary block-boundary chunking reproduces one contiguous call byte-for-byte. NEON inner loop is gated behind `__aarch64__ && __ARM_NEON` in `src/audio_resampler.c`.

### Known limitations / gaps

- **The resampler is built, tested, NEON-accelerated, and still unused in production.** No call site anywhere in `AEC`/`NR`/`Audio_ALG` links against it. AEC's new 48kHz→16kHz anti-alias sidechain (`DaResample48` in C, `_Resample48` in the Python port, both 2026-08-02/03) duplicated similar decimate-by-3 anti-alias filtering with a hand-written biquad cascade instead of reusing this canonical resampler. Any future consolidation should point that sidechain at `audio_resampler.h` instead.
- `fft_phase()`/`fft_from_mag_phase()` are dead code — same status, lower stakes.

---

## 2. `AEC` — single-channel acoustic echo cancellation

### Purpose

Single-channel (1 mic + 1 far-end reference) AEC. Two implementations of the same algorithm: `python/` is the fp64 algorithm spec (AEC3-aligned architecture — PBFDKF main filter + PBFDAF shadow filter + `PathChangeRegimeHandler` + AEC3-style post-filter chain), `c_impl/` is the float32 production port. Python↔C parity is tolerance-based, not bit-exact. Current production version is `__version__ = "3.24.1"` in `python/aec.py`; `AEC/CLAUDE.md` documents the fuller architecture.

### Current default configuration

| sample_rate | default `frame_size`/`fft_size` | other allowed grid | `filter_length` (auto) |
|---|---|---|---|
| 8000 | 256 (hop 128, 16 ms) | — | `sr*52/1000` = 416 |
| 16000 | **256** (hop 128, 8 ms) | 512 (hop 256, 16 ms) | `sr*52/1000` = 832 |
| 48000 | 1024 (hop 512) | — | `sr*64/1000` = 3072 |

The 16 kHz default flipped from 512→256 on 2026-08-01; 512 remains fully supported as an explicit choice. Sample-rate whitelist in C: `aec_is_valid_sample_rate()` — `{8000, 16000, 48000}`.

Three presets (`mild` / `balanced` / `aggressive`) differ **only** in `min_gain_floor_far_active_db`: mild −20 dB, balanced −28 dB (default), aggressive −38 dB.

### Public API

**Python** (`python/aec.py` re-exports `python/modules/orchestrator.py` / `config.py`):

```python
class AecConfig:                      # dataclass, modules/config.py
    sample_rate: int = 16000
    frame_size: int = -1              # -1 = auto from sample_rate
    hop_size: int = -1
    filter_length: int = -1
    mode: AecMode = AecMode.PBFDKF
    enable_res: bool = True
    enable_cng: bool = False

    @classmethod
    def from_preset(cls, preset: 'AecPreset', **kwargs) -> 'AecConfig': ...

class AEC:                            # modules/orchestrator.py
    def __init__(self, config: Optional[AecConfig] = None): ...
    def process(self, near_end: np.ndarray, far_end: np.ndarray) -> np.ndarray: ...
```

Usage:

```bash
python3 python/aec.py mic.wav ref.wav out.wav --mode pbfdkf --preset balanced --enable-res --cng
```

```python
from aec import AEC, AecConfig, AecPreset

cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000)
aec = AEC(cfg)
hop = aec.config.hop_size
for i in range(0, len(mic) - hop + 1, hop):
    out[i:i+hop] = aec.process(mic[i:i+hop], ref[i:i+hop])
```

**C** (`c_impl/include/aec.h`):

```c
void   aec_config_defaults(AecConfig* cfg, int sample_rate);
void   aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);
int    aec_is_valid_sample_rate(int sample_rate);

int    aec_create(Aec* a, const AecConfig* cfg);
void   aec_destroy(Aec* a);
void   aec_reset(Aec* a);
size_t aec_get_mem_size(const AecConfig* cfg);
Aec*   aec_init(void* mem, size_t mem_size, const AecConfig* cfg);
void   aec_process(Aec* a, const float* mic, const float* ref, float* out);
int    aec_hop_size(const Aec* a);
void   aec_get_res_context(const Aec* a, AecResContext* ctx);
void   aec_debug_status(const Aec* a, AecDebugStatus* out);
```

Usage (from `c_impl/example/aec_wav.c`):

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);

Aec aec;
if (aec_create(&aec, &cfg) != 0) { /* invalid grid */ }
int hop = aec_hop_size(&aec);

float mic[hop], ref[hop], out[hop];
aec_process(&aec, mic, ref, out);
aec_destroy(&aec);
```

CLI: `bin/aec_wav mic.wav ref.wav out.wav --preset {mild|balanced|aggressive} [--cng] [--no-delay-est] [--no-res] [--no-shadow] [--no-hpf] [--fft-size N]`. Build via `cd c_impl && make`.

### Known limitations / gaps

- `reverb_decay_estimator.c` is the sole remaining `double`-typed C file post-float32-campaign, and it is dead code with no production caller.
- The NE10 embedded backend and the KISS host/reference backend are byte-equal to *themselves* (static pool == heap path) but not bit-identical to *each other* — documented as pre-existing/expected.
- **Resolved 2026-08-03** (kept here for context, not an open item): the Python delay estimator (`modules/delay/echo_path_delay_estimator.py`) previously lacked a 48kHz anti-alias sidechain, causing Python and C to lock onto different delay estimates at 48kHz and fail `parity_aec_e2e` (max diff 0.46 vs 0.1 tolerance). Ported C's `DaResample48` filter to Python as `_Resample48` (same coefficients, verified sample-for-sample against C to float32 precision); `ClockdriftDetector`/`MatchedFilterLagAggregator` now use the effective post-sidechain rate rather than the raw native rate. `parity_aec_e2e` now passes at all three rates (48kHz max diff 3.9e-2, within the 1.0e-1 tolerance and back in the class of the originally-calibrated 6.4e-2 baseline).
- **Resolved 2026-08-03** (kept here for context, not an open item): the 4-channel pipeline's own `SharedMatchedDelayEstimator` (`Audio_ALG/pipelines/4ch_pipelines/pipeline.py`) used to do its own separate, naive stride-pick 48kHz decimation ahead of this estimator, hardcoding the inner estimator to `sample_rate=16000` and manually rescaling the returned delay — the fix above did not automatically extend to it. Refactored to match the same fix applied to `delay_aec3.c`'s `DaResample48` (2026-08-02) and this estimator's `_Resample48` (above): the inner `LegacyDelayShim` is now constructed at the wrapper's TRUE native `sample_rate` and fed raw, un-decimated hops; `EchoPathDelayEstimator` anti-alias-filters + decimates + rescales internally. See `pipelines/4ch_pipelines/test_pipeline.py`'s standalone-script verification for before/after accuracy.

---

## 3. `NR` — MMSE-LSA / OM-LSA noise reduction

### Purpose

Traditional (non-neural) single-channel noise suppression. The shipped production algorithm is **V3-2**, an MMSE log-spectral-amplitude (LSA) estimator (Ephraim–Malah 1985) with an MCRA/IMCRA noise tracker and an asymmetric-smoothed gain — internally one OM-LSA core (`MmseLsaDenoiser` is, despite the name, OM-LSA). Two independently-tunable axes compose on top: a suppression-**depth** axis (`strength`) and a content-**preservation** axis (`mode`). Current version: `4.5.0`.

### Current default configuration

| sample_rate | allowed `fft_size` | default (since 2026-08-02/03) |
|---|---|---|
| 8000 | 128, 256 | **128** (8 ms hop) |
| 16000 | 256, 512 | **256** (8 ms hop) |
| 48000 | 1024 | 1024 |

Strict no-padding grid: `frame_size == fft_size`, `hop_size == frame_size/2`. The 16 ms-hop grids (256@8k / 512@16k) remain supported, explicit alternates — several V3-2 preset constants (`alpha_xi=0.92`, `L=32`) are tuned directly against the 16 ms grid and retimed relative to that anchor via `core/signal_grid.py`'s `retime_ema_alpha()`/`retime_frame_count()` (C mirror: `mmse_lsa_retime_alpha_ref()`/`mmse_lsa_retime_frames_ref()`).

### Strength vs. mode

- **`strength`** (`mild`/`moderate`/`balanced`/`aggressive`) is the suppression-**depth** axis (`g_min_db` −20/−25/−30/−40 dB). `balanced` is an **empty overlay**.
- **`mode`** (`full`/`stationary`) is an **orthogonal content-preservation** axis: `full` removes noise and noise-like content (current/legacy default); `stationary` (ReSpeaker-like) adds a Wiener gain lower bound so only genuinely stationary noise floor is removed.
- Composition order is **strength-then-mode** on both sides — `mode='stationary'`'s own overlay always wins over whichever strength was chosen underneath.

### Public API

**Python** (`denoisers/v3_2_mmse_lsa.py`):

```python
class MmseLsaDenoiser(BaseDenoiser):
    def __init__(self, sample_rate: int = 16000,
                 frame_size=None, frame_shift=None, fft_size=None,
                 alpha_xi=0.98, q=0.5, xi_min_db=-25.0,
                 g_min_db=-40.0, alpha_g=0.7,
                 noise_method='recursive_average',
                 mode: str = 'full', strength: str = 'balanced', ...): ...

    def denoise(self, noisy_signal, return_spp=False, return_gain=False,
                return_noise_psd=False): ...
```

Construction factory (`process_audio.py`):

```python
def create_denoiser_from_config(
    version: str, config_dir: str, sample_rate: int,
    mode: str = None, strength: str = None, fft_size: int = None,
): ...
```

Usage:

```python
from process_audio import create_denoiser_from_config

denoiser = create_denoiser_from_config(
    'V3-2', config_dir='config', sample_rate=16000,
    mode='stationary', strength='aggressive')
enhanced = denoiser.denoise(noisy_signal)
```

**C** (`c_impl/include/mmse_lsa_denoiser.h` + `mmse_lsa_types.h`), frequency-domain, caller-owns-FFT/OLA:

```c
MmseLsaConfig mmse_lsa_default_config(int sample_rate);
MmseLsaConfig mmse_lsa_config_for_mode(int sample_rate, MmseLsaNrMode mode);
void          mmse_lsa_apply_stationary(MmseLsaConfig* config);
bool          mmse_lsa_validate_config(const MmseLsaConfig* config);

MmseLsaDenoiser* mmse_lsa_create(const MmseLsaConfig* config);
size_t           mmse_lsa_get_mem_size(const MmseLsaConfig* config);
MmseLsaDenoiser* mmse_lsa_init(void* mem, size_t mem_size, const MmseLsaConfig* config);
void             mmse_lsa_destroy(MmseLsaDenoiser* self);

int  mmse_lsa_process(MmseLsaDenoiser* self, const Complex* spectrum_in, Complex* spectrum_out);
int  mmse_lsa_process_gain(MmseLsaDenoiser* self, const Complex* spectrum_in,
                           const float* extra_noise_psd, float* gain_out);
void mmse_lsa_reset(MmseLsaDenoiser* self);
```

Usage (from `c_impl/example/main.c`):

```c
MmseLsaConfig config = mmse_lsa_config_for_mode_grid(
    sample_rate, mmse_lsa_default_fft_size(sample_rate), MMSE_LSA_NR_BALANCED);

MmseLsaDenoiser* denoiser = mmse_lsa_create(&config);
FftHandle*       fft      = fft_create(config.fft_size);

fft_forward(fft, frame_buf, spec_in);
mmse_lsa_process(denoiser, spec_in, spec_out);
fft_inverse(fft, spec_out, time_out);

mmse_lsa_destroy(denoiser);
fft_destroy(fft);
```

CLI: `denoise_wav input.wav output.wav [--bypass] [--nr-mode mild|moderate|balanced|aggressive] [--stationary] [--fft-size N]`.

`mmse_lsa_process_gain()` additionally supports the AEC-integrated case: it folds an external residual-echo PSD into the noise floor so a caller can drive an AEC(linear)→NR→RES chain — this is what `Audio_ALG/pipelines` uses.

### Known limitations / gaps

- ~~The checked-in VCTK benchmark tooling is non-functional as shipped.~~ **Fixed (2026-08-03, NR commit `3f28774`).** The dead `compute_improvement_vctk.py` / `regenerate_all_vctk.py` pair (which imported modules that didn't exist anywhere in the repo, and hardcoded a `noisy`/`clean` subdirectory layout that never matched the actual downloaded dataset) has been replaced and removed. `run_vctk_benchmark.py` + `compare_vctk_benchmark.py` are the current, checked-in, working VCTK+DEMAND runner: the former resamples/denoises/scores PESQ/STOI/SI-SDR/segSNR per file with explicit `--mode`/`--strength`/`--fft-size` selection against `noisy_testset_wav`/`clean_testset_wav`, the latter does a fail-closed baseline-vs-candidate comparison with regression gates (see its `GATES` dict). Note: the 824-case VCTK A/B cited in the same commit's NR tuning decision (CHANGELOG `[Unreleased] - 2026-08-03`) predates this tool and still used ad hoc scratchpad tooling; use `run_vctk_benchmark.py` for any future re-run of that comparison.
- The `L` retiming fix (2026-08-03, see CHANGELOG `[4.5.0]`) has **zero observable effect** in the standalone VCTK benchmark specifically, because `L`'s consumer (the MCRA `S_min`/minima-tracking indicator) is architecturally bypassed whenever `mcra_accept_external_spp=True` (the standalone-NR default). It should be observable in the AEC-integrated path (`accept_external_spp=False`), not yet separately benchmarked.

---

## 4. `Audio_ALG/pipelines` — integration layer (AEC + NR [+ DOA + GSC])

### Purpose

Wires the standalone `AEC` and `NR` libraries (git submodules `lib/aec`, `lib/nr`) into end-to-end chains: a mono AEC(linear)→echo-aware-NR→RES chain (`pipelines/audio_pipeline.h/.c`), and a 4-microphone front end (`4ch_pipelines/4aec_nr_res.h/.c`) that runs four independent linear AEC filters off one shared delay estimate, hands the four aligned spectra to an externally-supplied SRP-PHAT/GSC beamformer, and finishes with a shared mono NR+RES tail.

### Current default configuration

As of 2026-08-03, **all three pipeline entry points now agree with AEC/NR's own defaults**: 16 kHz → 256/128 (8 ms hop), 48 kHz → 1024/512. This required two separate fixes this session:
- `4ch_pipelines/4aec_nr_res.c`'s `derive_dims_and_configs()` had its own independent hardcoded 512 default at 16 kHz — fixed.
- The mono pipeline's `pipelines/pipeline_dims.h` (`compute_frame_dims()`, used by `audio_pipeline_default_config()`) and `aec_nr_pipeline.py`'s `_project_grid()` both also independently hardcoded 512 at 16 kHz — fixed in the same pass. 512 remains a supported, explicit alternate everywhere.

### Public API

**Mono pipeline, Python** (`pipelines/aec_nr_pipeline.py`) — CLI-shaped:

```bash
cd Audio_ALG
python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav \
    --aec-preset balanced --nr-preset balanced
```

**Mono pipeline, C** (`pipelines/audio_pipeline.h`):

```c
AudioPipelineConfig audio_pipeline_default_config(int sample_rate);
int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg, AudioPipelineMemReq* out);
AudioPipeline* audio_pipeline_init(void* mem, size_t bytes, const AudioPipelineConfig* cfg);
AudioPipeline* audio_pipeline_init_ex(void* mem, size_t bytes, const AudioPipelineConfig* cfg,
                                      const AudioPipelineMemReq* expected);
AudioPipeline* audio_pipeline_create(const AudioPipelineConfig* cfg);
int  audio_pipeline_process(AudioPipeline* p, const float* mic, const float* ref, float* out);
void audio_pipeline_reset(AudioPipeline* p);
void audio_pipeline_destroy(AudioPipeline* p);
int  audio_pipeline_hop_size(const AudioPipeline* p);
```

Usage — pool-first (board path):

```c
AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
AudioPipelineMemReq req;
audio_pipeline_get_mem_requirements(&cfg, &req);
void* pool = platform_alloc(req.bytes, req.alignment);
AudioPipeline* p = audio_pipeline_init_ex(pool, req.bytes, &cfg, &req);

int hop = audio_pipeline_hop_size(p);
audio_pipeline_process(p, mic, ref, out);

audio_pipeline_destroy(p);
platform_free(pool);
```

**4-channel pipeline, C** (`4ch_pipelines/4aec_nr_res.h`) — pre/post split so an external beamformer sits in between:

```c
FourAecNrResConfig four_aec_nr_res_default_config(int sample_rate);
int four_aec_nr_res_get_mem_requirements(const FourAecNrResConfig* cfg, FourAecNrResMemReq* out);
FourAecNrRes* four_aec_nr_res_init(void* mem, size_t bytes, const FourAecNrResConfig* cfg);
FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg);

int four_aec_nr_res_process_pre(FourAecNrRes* p, const float* microphones_interleaved,
                                const float* ref, FourAecNrResPreFrame* out);
int four_aec_nr_res_process_post(FourAecNrRes* p, const FourAecNrResFrameToken* token,
                                 const Complex* weights, float* out);
void four_aec_nr_res_destroy(FourAecNrRes* p);
```

Usage:

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
FourAecNrRes* p = four_aec_nr_res_create(&cfg);

FourAecNrResPreFrame pre;
four_aec_nr_res_process_pre(p, mics_interleaved, ref, &pre);
/* external SRP-PHAT/GSC consumes pre.linear_spectra[4][n_freqs], produces weights[4][n_freqs] */
four_aec_nr_res_process_post(p, &pre.token, weights, out);

four_aec_nr_res_destroy(p);
```

**DOA / SRP-PHAT** (`4ch_pipelines/third_party/doa/srp.c`): steered-response-power PHAT direction-of-arrival estimator. `srp_create()`/`srp_create_from_geometry()` build per-angle steering vectors; `srp(SRP*, Complex** X, const int* mask)` scores every candidate angle from the cross-spectrum PHAT weighting; `srp2doa()`/`doa_step()` reduce to a DOA estimate. Partially optimized this session to a band-limited PHAT computation (restricted to `[f_start, f_end]`, the configured search band — a real cut in the hot loop). **Known follow-up:** steering-vector precompute (`pair_steer`) and the `score_scratch`/`best_score` buffers are still allocated and cleared full-band every frame — only the runtime accumulate loop was narrowed.

**GSC beamformer** (`4ch_pipelines/third_party/GSC/gsc.h`): generalized sidelobe canceller — `gsc_create()`, `gsc_process()`/`gsc_process_with_weights()`, `gsc_reset()`, `gsc_destroy()`. Supports fixed-beam and adaptive (RLS) modes.

### Known limitations / gaps

- The DOA/SRP-PHAT band-limiting and GSC modules are vendored/adapted third-party code — treat their internals as a black box beyond the entry points documented here.
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter and is never selected by default — do not treat it as a real beamforming option.
- **Resolved 2026-08-03**: the 4-channel pipeline's shared delay estimator (`SharedMatchedDelayEstimator` in `pipeline.py`) used to do a naive, non-anti-aliased stride-pick decimation at 48 kHz against a hardcoded-16kHz inner estimator — the Python anti-alias fix in §2 was applied to the mono AEC's `EchoPathDelayEstimator` but not originally ported to this separate 4-channel wrapper. Now fixed: the wrapper constructs its inner estimator at the true native sample rate and feeds it raw hops, relying on `EchoPathDelayEstimator`'s own internal 48kHz anti-alias sidechain (see §2).
