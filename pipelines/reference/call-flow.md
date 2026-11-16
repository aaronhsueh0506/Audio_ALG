# Pipeline 時序與延遲

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **AEC linear path** | 0 samples | PBFDKF overlap-save 的當前 hop 線性誤差可立即取得 |
| **Synthesis WOLA delay** | 1 hop | NR/RES 頻譜由 sqrt-Hann WOLA 重建；這就是 full-chain 的演算法延遲 |
| **Pipeline total latency** | 1 hop — 16 ms @ 8 kHz, 8 ms @ 16 kHz default, 16 ms @ 16 kHz alt, ~10.7 ms @ 48 kHz | 不額外保存上一個 AEC context；AEC-only 為 0 samples |
| **Processing (per hop)** | 待量測 | 目標平台上 AEC + NR + RES 合計的實測；主機端用 `--timing` 取各階段 µs |
| **RTF** | 待量測 | 同上；由實測 per-hop 時間除以 hop 時長 |

上表是 conventional mono pipeline 的數字。兩條 Align-ULCNet application 的
加總額外延遲同樣是 **1 hop**（唯一來源是 ULCNet synthesis WOLA）：mono 與
4ch 變體都是——4ch 把 BF 的 beamformed error 頻譜直接當成模型的分析幀，
沒有中間的 beam WOLA→再 analysis 來回，也沒有 one-hop far 補償。細節見
[`4ch/alignulcnet/README.md`](../4ch/alignulcnet/README.md)。

The two DFN2 applications have an algorithmic latency of `M + D + 1` hops on a 48 kHz host (`M` = mask lookahead, `D` =
deep-filter lookahead, fixed by the model and carried by the `DFN2Grid` in the hop's pool plan (`DfnHopPoolPlan.stream_cfg.grid`); the shipped 1/1 pair gives 3 hops, 32 ms). Below 48 kHz the rate bridge (`stream/ainr/bridge`) is used and
the latency is the host's own WOLA hop plus `ainr_bridge_added_delay_samples()` native samples (shipped 1/1 pair: 672 at
16 kHz hop 128, 629 at 16 kHz hop 256, 330 at 8 kHz hop 128; the floor of the exact value, which `ainr_bridge_added_delay_exact()` returns as a multiple of 1/3 sample, whole only when `M + D + 1` is a multiple of 3), a figure that grows with `M + D`; the other `(M, D)` pairs
follow the same formula but have no trained checkpoint, so their audio quality is unmeasured. See [`dfn/README.md`](../dfn/README.md) and [Build](build.md#build).

### Memory Budget

Measured with `"$(make -s print-bin-dir)"/aec_nr_pipeline_static --print-mem-size
--sample-rate <sr> [--fft-size <alt>]` (balanced levels). The AEC column is the
composite `aec_get_mem_size()` pool — it already contains HPF, PBFDKF ×2
(main+shadow), delay estimator, the residual stage (RES gain, comfort noise) and the AEC-internal FFTs.
The remaining columns are the example's other stages, each sized by its own query
(`ac_istft_get_mem_size()` for the synthesis `AcIstft`, `anr_get_mem_size()` for the
denoiser) plus the gain and spectrum buffers the per-hop chain needs; the mono chain takes
the AEC's own gain, so there is no separate RES object. The NE10
twiddle configs are carved from these pools too, so the columns are the complete
memory requirement (strict init→destroy zero-heap). 16 kHz's default
grid is 256/128 (see [mono lifecycle](../../docs/reference/mono/lifecycle.md), "Grid"); the alternate 512/256 grid
stays explicitly selectable.

| Rate | AEC | Synthesis (`AcIstft`) | NR | Gain/spectrum buffers | **Total** |
|------|-----|-----------|-----|---------------|-----------|
| **8 kHz** | 210,864 B | 10,320 B | 26,384 B | 2,096 B | **249,664 B (243.8 KB)** |
| **16 kHz (default, 256/128)** | 239,232 B | 10,320 B | 42,944 B | 2,096 B | **294,592 B (287.7 KB)** |
| **16 kHz (alt, 512/256)** | 351,088 B | 19,792 B | 51,728 B | 4,144 B | **426,752 B (416.8 KB)** |
| **48 kHz** | 774,000 B | 38,736 B | 135,264 B | 8,240 B | **956,240 B (933.8 KB)** |

The AEC column is owned by `AEC/docs/reference/lifecycle.md` (§4, the pool-size section) — re-measure from
there rather than editing it here, and always prefer the values `aec_get_mem_size()`,
`anr_get_mem_size()` and `ac_istft_get_mem_size()` return at runtime. `--no-nr` drops the NR column;
`--aec-only` is the AEC column alone. The synthesis borrows the AEC's synthesis window (`aec_synthesis_window()`), so its column carries no window table.

> AEC 的濾波器長度由取樣率決定（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072
> taps、6 partitions at hop=512），不是 `AecConfig` 欄位；記憶體吃緊時調 `delay_num_filters`
> （MATCHED，每少一個單支省 11,264 B、48 kHz 為 23,552 B）或改 `delay_mode`。NR 的 `L` 是內部欄位且決定 pool 版面：要縮它就得自己組
> `AnrConfig`（先 `anr_config_for_level_grid()` 再改 `L`）、重新 `anr_get_mem_size()`，
> 該實例之後不再等於裸 level 組態，`anr_set_level()` 會被拒（見 [call sequence](#call-flow) 一節）。
> `n_partitions` 由濾波器長度與 hop 推導（函式庫內部）。三個 rate 各自依 grid 的 hop=fft_size/2
> 規則自動推導，不是統一的 10 ms 規則；16 kHz 另外還有一個可選的
> 512/256 grid（見上表）。靜態範例的 AEC、NR 與合成 config 都由同一個
> `AecSignalGrid`（`derive_lane()` 的輸出，即 `aec_resolve_signal_grid()` 的結果）導出，三方格點因此一致。

## Call flow

Every application analyses once per hop. The front end returns its context spectra as one borrowed
view: `AecResidualView` ([`aec.h`](../../../AEC/c_impl/include/aec.h); read with `aec_get_residual()` after
`aec_process_linear()` and `aec_process_residual()`) for the mono `Aec`, and the same `AecResidualView` of the 4-channel array (`AecConfig.n_mics = 4`; filled
by `aec_process_residual(&AecBeam)`) for the 4-channel flow. Each later stage is a plain function over that view, and the
application writes the combination itself; each example `main` shows it (`mono_form()` in
[`mono/aec_nr_res/main.c`](../mono/aec_nr_res/main.c), `array_sample_stages_chain()` in
[`array_sample_unit.h`](../4ch/array_sample_unit.h)):

```
mono:  mic/ref ─► Aec ─► AecResidualView { error_spec, near_spec, far_spec, r2, comfort_noise, res_gain, base_spec, usable_linear, ... }
                        └► base_spec = B = rendering base (E, or E + echo on a guard hop); aec_post_reset_serial ─► reset event
4-ch:  mics/ref ─► aec_process_linear + aec_get_array ─► beamformer ─► aec_process_residual ─► AecResidualView { error_spec, near_spec, far_spec, r2, comfort_noise, res_gain, base_spec, formed_hop = NULL, ... }
                      │
        gain stages   ├─ G_nr: anr_process_gain_with_residual_psd(nr, S, residual_psd_audio)    (residual_psd_audio = R² / 32768², while the linear estimate is usable)
                      │        S = B for mono; S = E = v.error_spec for 4-ch (array_sample_anr_gain()); the gain is applied on B in both
                      ├─ G_res: v.res_gain (mono: the AEC's own; 4-ch: the array's one gain stage over the fused spectra; applied by aec_apply_gain)
        combination   └─ aec_apply_gain → aec_apply_cng → ac_istft_push → out   (the same two calls for both)
```

The mono per-hop calls with their signatures (`aec.h`, `anr.h`, `audio_stft.h`,
`utility/simd_kernels.h`); every returned pointer is borrowed from its producer:

```c
int   aec_process_linear(Aec* a, const float* mic, const float* ref);      /* hop floats each; AEC_OK, or an AEC_ERR_* with the instance untouched */
int   aec_process_residual(Aec* a, const AecBeam* beam);                   /* directly after aec_process_linear(); beam NULL for one channel */
int   aec_get_residual(const Aec* a, AecResidualView* out);                /* out->struct_size = sizeof *out; error_spec, base_spec, r2, comfort_noise, res_gain, ... */
const float* anr_process_gain_with_residual_psd(AnrDenoiser* self, const Complex* spectrum_in,
                       const float* residual_echo_psd);                    /* n_freqs gains (borrowed), or NULL: NULL argument / refused hop */
const float* anr_process_gain(AnrDenoiser* self, const Complex* spectrum_in);   /* the same with residual_echo_psd = NULL */
void  sk_scale_f32(float* out, const float* x, float g, int n);            /* residual_psd_audio = R^2 * (1/32768^2) */
int   aec_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec);   /* out = base_spec * min(G_res, extra_gain); extra_gain may be NULL */
int   aec_apply_cng(Aec* a, const float* extra_gain, Complex* spec);      /* comfort noise on the cut bins from the Aec's own generator; the same extra_gain */
int   ac_istft_push(AcIstft* s, const Complex* spec, float* out);          /* writes hop samples (0 only for the first AC_OLA_ENV_NORM frame) */
```

`aec_apply_gain_spec()` / `aec_apply_cng_spec()` are the same two steps over explicit arrays (and a caller-held generator state), for a caller
that holds its own `G_res`, base and comfort noise.

There is no independent time-domain RES filter after NR and no function-pointer
table of stages: `G_res` comes with the linear residual (inside the AEC; for the 4-ch array, computed once over the fused spectra by `aec_process_residual()`), is fused with `G_nr` (by `aec_apply_gain`, which takes `G_nr` as its extra gain) and applied once to
the rendering base. Which stages exist, in which order, is read straight from each example's per-hop
function ([`main.c`](../mono/aec_nr_res/main.c) `process_hop()` / `mono_form()`,
[`4ch_raw.c`](../4ch/4ch_raw.c) `raw_post()` / `array_sample_stages_chain()`). The 4-channel
programs share two small header-only helpers, [`array_sample_post.h`](../4ch/array_sample_post.h)
(`array_sample_anr_gain()`, `array_sample_post_reset_due()`, `ARRAY_SAMPLE_PSD_SCALE_INV`; tested by `test_array_sample_post`) and [`array_sample_unit.h`](../4ch/array_sample_unit.h) (the array + helper unit and the stages behind the fuse, shared by `4ch_raw` and `4ch_static`); the mono examples keep their own
`#define` (`PSD_SCALE_INV`) in the file; the comfort noise of both is the `Aec`'s own generator
(`aec_apply_cng()`: seed `AEC_CNG_SEED`, floor `AEC_CNG_NR_GAIN_FLOOR`).

Only the multi-channel shared delay, the four lanes and the fusion with its gain stage are one object (the `Aec` array, `n_mics = 4`); every other stage (the beamformer and the denoiser
included) is a separate object that the integrator initialises (its own `mem_size` and `init`) and chains.

| Application | Analysis call | Stages written out | Synthesis |
|---|---|---|---|
| Mono AEC + NR/RES | `aec_process_linear()` → `aec_process_residual()` → `aec_get_residual()` | `anr_process_gain_with_residual_psd()` on `v.base_spec`, `aec_apply_gain()` (`v.res_gain` with the denoiser gain), `aec_apply_cng()` | `ac_istft_push()` |
| 4-ch AEC + BF + NR/RES | `aec_process_linear()` + `aec_get_array()` → the beamformer (the sample helper's `array_sample_beam_process_beam()` or your own) → `aec_process_residual()` + `aec_get_residual()` | `array_sample_anr_gain()`, `aec_apply_gain()` (the array's `G_res` with the denoiser gain), `aec_apply_cng()` | `ac_istft_push()` |
| Mono / 4-ch Align-ULCNet | mono: `aec_process_linear()` → `aec_process_residual()` (`AEC_RESIDUAL_STATE`) → `aec_get_residual()` / `aec_get_linear()`; 4-ch: the array with `residual = AEC_RESIDUAL_STATE` and the sample beam helper; the application's `ac_stft_push_planar()` of the raw far hop (mono: also of `formed_hop`), then `ulcnet_stream_pre_process_freq()` | the network (between `ulcnet_stream_pre_process_freq()` and `ulcnet_stream_post_process_freq()`) | `ulcnet_stream_post_process_freq()` + `ac_istft_push_planar()` |
| Mono / 4-ch DFN2 | mono: `aec_process_linear()` → `aec_process_residual()` → `aec_get_residual()`; 4-ch: `array_sample_beam_process()` → `array_sample_beam_fuse()` | the example forms P itself (`rig_process()` / `dfn_form()`: `aec_apply_gain(aec, NULL, P)` and the optional `aec_apply_cng()`, for mono and 4-ch alike; no NR, no synthesis of its own), then the DFN2 model stream (`dfn2_stream_*_freq` of `stream/ainr/dfn2`) through `dfn_hop_*` | `dfn_hop_end()` |

### How do I chain a new frequency-domain stage?

Write it as a gain-type function over the frame and fold its result into the
extra gain that `aec_apply_gain()` and `aec_apply_cng()` take. The example below is compiled and run by
[`tests/test_readme_flow.c`](../tests/test_readme_flow.c) (`make test_readme_flow`):
with a unity stage the chain is bit-identical to the hand-written chain of
[`mono/aec_nr_res/tests/conv_chain_mono.h`](../mono/aec_nr_res/tests/conv_chain_mono.h).

```c
typedef struct HfTilt { int bin_hi; float amount; float* g; } HfTilt;   /* g: n_freqs floats, caller memory */

static const float* hf_tilt_gain(HfTilt* s, int n_freqs) {
    for (int k = 0; k < n_freqs; ++k) s->g[k] = k > s->bin_hi ? s->amount : 1.0f;
    return s->g;                     /* valid until the next call, like every gain stage */
}

int chain_hop(Chain* c, const float* mic, const float* ref, float* out) {
    AecResidualView f = { sizeof(f) };
    const float* residual_psd_audio = NULL;
    if (aec_process_linear(c->aec, mic, ref) != AEC_OK) return -1;                 /* the one analysis */
    if (aec_process_residual(c->aec, NULL) != AEC_OK) return -1;
    if (aec_get_residual(c->aec, &f) != AEC_OK) return -1;                         /* f.base_spec: the spectrum to render */
    if (aec_post_reset_serial(c->aec) != c->reset_serial) {                       /* hard post-chain reset in this hop: */
        c->reset_serial = aec_post_reset_serial(c->aec);                          /* before the push (the Aec reseeds its own CNG) */
        ac_istft_reset(c->istft);
    }
    if (f.r2 && f.usable_linear) {                      /* residual-echo powers into the noise floor */
        sk_scale_f32(c->g_nr_scratch, f.r2, PSD_SCALE_INV, f.n_freqs);
        residual_psd_audio = c->g_nr_scratch;
    }
    const float* nr_gain = anr_process_gain_with_residual_psd(c->nr, f.base_spec, residual_psd_audio);
    if (!nr_gain) return -1;
    const float* tilt = hf_tilt_gain(c->tilt, f.n_freqs);                          /* the new stage */
    sk_min_f32(c->g_extra, nr_gain, tilt, f.n_freqs);                              /* every gain but the AEC's own */
    if (aec_apply_gain(c->aec, c->g_extra, c->spec) != AEC_OK) return -1;          /* S(f) = B(f) . min(g_extra, G_res) */
    if (aec_apply_cng(c->aec, c->g_extra, c->spec) != AEC_OK) return -1;           /* comfort noise on the cut bins */
    (void)ac_istft_push(c->istft, c->spec, out);                                   /* the one inverse transform */
    return 0;
}
```

Rules the chain relies on:

- A disabled gain source reads as the **unity vector** (all ones) or, for the extra gain of `aec_apply_gain()` /
  `aec_apply_cng()`, as NULL, so one chain serves every combination. `sk_min_f32` is a compare-and-select (`a < b ? a : b`), not
  `fminf`: a NaN in its second operand reaches the output, a NaN in its first operand is replaced by the second,
  so a finite check on the emitted hop (`skn_all_finite_f32()`) stays the guard against a non-finite gain.
- `aec_apply_cng()` fills the bins the AEC's own gain (`G_res`) cut, which alone decides *which* bins are
  filled, and scales the *level* by the extra gain (floored at `AEC_CNG_NR_GAIN_FLOOR`); pass the same extra gain
  as to `aec_apply_gain()`. The generator is the `Aec`'s own (seed `AEC_CNG_SEED`, reseeded by every hard post-chain
  reset and by `aec_reset()`); the caller only clears its overlap-add tail when `aec_post_reset_serial()` moves.
  The 4-channel array behaves the same way: its generator restarts at `AEC_CNG_SEED` on a lane's hard reset and on `aec_reset()`.
- `v.error_spec` / `v.formed_hop` are the analysis seam and are never rendered; the chain renders the
  rendering base `v.base_spec` (the same spectrum `aec_process()` multiplies by its gain, `E`
  bit for bit on every hop the output guard leaves alone). Mono: the denoiser estimates on that same spectrum. 4-channel:
  it estimates on `v.error_spec` (`array_sample_anr_gain()`; the two differ only on output-guard hops, see the 4ch README,
  "Analysis and rendering spectra") and its gain is applied on `v.base_spec`.
- A spectrum-type stage (the 4-channel beamformer) is a function that takes the
  lane spectra and returns a spectrum; it replaces the error spectrum (`error_spec` of the
  `AecResidualView`) in the chain. The sample helper hands the BF spectrum to `aec_process_residual()` inside
  `array_sample_beam_fuse()`.
- Every pointer in the frame is borrowed from the producer. It is valid until the
  producer runs again (the next `aec_process_linear()`,
  a reset) and is never written through.
- A refused call (a stage returns NULL or an error, or the emitted hop is not finite) is the
  integrator's to recover: reset the front end and every stage of the chain together
  (`aec_reset()` restarts the `Aec`'s comfort-noise generator too, for the 4-channel array as well).

### Where do I call the inverse transform?

Once per hop, last: `ac_istft_push()` (audiolib) on the spectrum that has all gains applied
and comfort noise added.
It performs the inverse FFT, the sqrt-Hann window and the overlap-add, and emits
exactly one hop; the half-window it holds back is the pipeline's one hop of
algorithmic latency. The spectrum must sit on the analysis frame of the same hop
(the AEC's `error_spec` is the exact 50 %-overlap sqrt-Hann STFT, and the rendering base is `error_spec` or the
capture spectrum of the same frame); calling it
twice in a hop, or skipping a hop, breaks the overlap-add. Applications that end in
a neural stage call it after `ulcnet_stream_post_process_freq()` (the stream's spectrum entry returns a spectrum) or inside `dfn_hop_end()`. The neural libraries
(`AINR`, `AIAEC`) are frequency domain only: they take and return spectra, and the framing, FFT and
window come from audiolib in the application (the examples frame around the streams' spectrum entries; the rate bridge carries its own framing for a host below the model rate).

### What memory do I provide?

Static memory only: each object is `get_mem_requirements()` (or
`<x>_mem_size()`), one caller block that is 16-byte aligned, then `init_ex()`.
Nothing allocates inside `process`. Every library is queried and initialised on its
own; the integrator sums the sizes and cuts the blocks:

```c
size_t aec_b   = aec_get_mem_size(&aec_cfg);                       /* aec_cfg: aec_config_from_level() + the grid and delay fields; 0 = invalid */
AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, grid.fft_size, ANR_LEVEL_BALANCED);   /* grid from aec_resolve_signal_grid() */
size_t nr_b    = anr_get_mem_size(&nr_cfg);
AcStftConfig sy_cfg;  ac_stft_config_default(&sy_cfg, grid.fft_size);
size_t sy_b    = ac_istft_get_mem_size(&sy_cfg);
uint8_t* pool  = board_alloc(ALIGN16(aec_b) + ALIGN16(nr_b) + ALIGN16(sy_b) + gain_buffers + my_stage_bytes, 16);
Aec* aec       = aec_init(pool, aec_b, &aec_cfg);                  /* mono: G_res is the AEC's own */
AnrDenoiser* nr = anr_init(pool + ALIGN16(aec_b), nr_b, &nr_cfg);
AcIstft* istft = ac_istft_init(pool + ALIGN16(aec_b) + ALIGN16(nr_b), sy_b, &sy_cfg);
/* gain vectors, the output spectrum and your stage (HfTilt, hf_tilt_init()) follow in the same block */
```

- A stage keeps no state of its own when its functions take the object the
  application carved (`anr_process_gain()` takes the `AnrDenoiser`, `aec_apply_gain()`
  the `Aec`). Gain vectors the chain needs (the extra gain, the NR prior scratch, the
  spectrum buffer) are `n_freqs` floats / bins each, also from the caller's block.
- The 4-channel array is one `Aec` pool (`aec_get_mem_requirements()`): the `Aec` structure, the four lanes, the shared `DelayAec` (MATCHED only),
  the alignment ring, the fusion buffers and the gain stage: 524,448 B at 16 kHz/256 (MATCHED, `AEC_RESIDUAL_FULL`; 420,736 B with
  `AEC_RESIDUAL_STATE`, `test_target_abi`), of which the four lanes are 383,232 B (4 x 95,808 B).
  `4ch_external_static --print-mem-size` prints lanes (`AEC x4`) 383,232 B and front end (the rest of the array pool) 141,216 B
  at 16 kHz/256 (48 kHz/1024: 1,371,456 B and 404,896 B). The sample beam helper
  (SRP-PHAT, BF, scratch) is one more block. The post-beam stages are separate blocks sized by their own queries
  (`anr_get_mem_size()`, `ac_istft_get_mem_size()`),
  and a neural stage (`ulcnet_stream_get_mem_size()` of a `freq_only` stream, 10,720 B at 16 kHz/512 and 20,960 B at 48 kHz/1024, plus the application's `AcStft` / `AcIstft`; for DFN2 the tail `dfn_hop_pool_bytes()`
  sizes: `dfn2_stream_get_mem_size()` (a `freq_only` stream), six `n_bins` float buffers, then `ainr_bridge_get_mem_requirements()` or the synthesis and the output
  spectrum: 214,448 B for a 16 kHz/256 host and 143,536 B for a 48 kHz host with the shipped 48 kHz / 1024 model, NE10)
  is one more block. The recurrent state of the network is in none of these pools: a float host registers
  its own buffers, a board's accelerator holds it.
- Re-query the descriptor after any library, backend, build-option or config
  change; `aec_init_ex()` refuses a stale one. `Aec` (`AecMemReq`, single-channel and 4-channel array alike), the
  DFN2 model stream, the rate bridge and the other model streams carry descriptors; the example headers do not.

### How do I run the NN stage on the NPU?

The network sits between two calls. The example code and the model stream do everything before (analyses,
delay policy) and after (accept or fail open, overlap-add); you run the
accelerator on the views they hand out. The loop is the same for the four neural
applications; the examples are compiled by `tests/test_readme_flow.c`.

```c
/* Align-ULCNet (mono or 4-ch): at most one frame per hop */
ac_stft_push_planar(far_stft, far, far_re, far_im);            /* the application's far analysis (mono: and err_stft on formed_hop) */
pre = ulcnet_stream_pre_process_freq(stream, err_re, err_im, far_re, far_im, delay, &io);   /* reprime / model restart */
if (pre == 1)                                                  /* io: the graph's tensors, NaN-prefilled mask */
    ai_rc = run_ulcnet_npu(&io);                               /* STREAM_RUN_OK = every tensor written; else the identity */
ulcnet_stream_post_process_freq(stream, ai_rc, out_re, out_im); /* accepted enhanced spectrum, else the error spectrum */
ac_istft_push_planar(istft, out_re, out_im, out);              /* the application's WOLA: one hop, real output at hop 0 */

/* DFN2 (mono or 4-ch): up to two frames per hop through the 8/16 kHz rate bridge */
dfn_host_begin(&host, mic, ref);                               /* AEC, RES gain, P = E.G_res; dfn_hop_begin(&hop, e, p) */
while (dfn_hop_nn_begin(&hop, &job) == 1)                      /* job: DFN2PrepostInputs / Outputs */
    dfn_hop_nn_end(&hop, run_npu(&job));
dfn_hop_end(&hop, out);
```

The 4-ch Align-ULCNet application also splits before the beamformer, for a
caller whose mask and VAD come from the lane spectra of the same hop (the masker
reads lane 0): `array_sample_beam_process_front()` returns the read-only lane
spectra (`AecArrayView`), the caller computes mask, `vad_raw` and `vad_out`,
`array_sample_beam_process_beam()` runs SRP and BF with them and fills the
`ArraySampleBeamInfo` (DOA, BF adaptation state and `info.delay`).
`array_sample_beam_process_with_activity()` is `process_front()` +
`process_beam(front, v, v, NULL)`; see
[`4ch/alignulcnet/README.md`](../4ch/alignulcnet/README.md). The stream is handed the array's
`front.delay` (`changed`, `solid`) and the spectrum of the caller's raw far hop; nothing else crosses.

Fail-open is part of the contract, not an option: a non-zero result, an output
with any non-finite value, a DFN2 frame left open or never fetched and an Align-ULCNet mask left unwritten all take
the identity for that frame (ULCNet: the error spectrum passes; DFN2: a unit band
mask), the framing and the clocks advance exactly once, and the latency never
changes. `dfn_hop_process()` is the DFN2 sequence driven by the model callback installed in the hop's pool plan, so a board with no
callback and a host with one produce identical bytes; its model's `reset` callback
(if set) is still called on an alignment boundary and by `dfn_hop_reset()`. The Align-ULCNet stream has no callback:
an alignment boundary and `ulcnet_stream_reset()` advance the state generation instead. A board that
registers no state keeps it in the accelerator and reloads quantized zeros at start and whenever the
state generation changes (`job.state_generation` / `dfn_hop_state_generation()`, the model stream's `dfn2_stream_state_generation()`,
for DFN2, `io.state_generation` / `ulcnet_stream_state_generation()` for ULCNet); a refused commit (a
non-finite output) advances the counter as a reset does. DFN2's first `mask_lookahead` hops carry no model frame: the model does not run and the counter stays; on a real frame a
nonzero model result does not advance it.

### How do the 4-channel applications split the hop?

```c
aec_process_linear(aec, mics_interleaved, ref);       /* shared delay + 4 lanes */
aec_get_array(aec, &front);                           /* AecArrayView: front.lane_error_spec[4], front.aligned_ref */
/* your beamformer on front.lane_error_spec -> weights W (channel-major Complex[4][n_freqs]) and, optionally, its spectrum E */
AecBeam beam = { W, E };                              /* E == NULL rebuilds it from W */
aec_process_residual(aec, &beam);                     /* project near, R^2, comfort noise with those weights; the one G_res */
aec_get_residual(aec, &v);                            /* AecResidualView; v.formed_hop = NULL */
/* then the application's chain: array_sample_anr_gain(), aec_apply_gain(aec, G_nr, P), aec_apply_cng(aec, G_nr, P), ac_istft_push() */
```

The examples run SRP-PHAT and the beamformer through the sample helper, which sits between the two
calls (the array is then driven only through the helper):

```c
array_sample_beam_process_front(b, mics_interleaved, ref, &front);
/* [optional] mask / VAD from the lane spectra */
array_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT + BF -> beam.spectrum, beam.weights */
array_sample_beam_fuse(b, &v);    /* aec_process_residual() with this hop's beam, then aec_get_residual() */
```

Neither the helper nor the application builds a RES: the array's gain stage (`AecConfig.residual = AEC_RESIDUAL_FULL`) computes the one `G_res`
over the fused spectra and `aec_apply_gain()` applies it (`4ch_static.c` and `4ch_external_static.c` do).

`aec_process_linear()` refuses a hop (`AEC_ERR_DSP`, array reset) when a lane's result is not finite: a microphone sample that is finite but huge is refused on the hop that carried it, a reference sample only when it reaches the lanes through the alignment ring, hops later. After a refused hop `aec_get_residual()` is `AEC_ERR_SEQUENCE` until the next hop.

`aec_process_residual()` needs a linear frame processed since the last reset (otherwise
`AEC_ERR_SEQUENCE`, nothing changed), refuses an all-zero or non-finite weight set
(`AEC_ERR_ARG`, nothing changed) and an `AEC_RESIDUAL_STATE` array (`AEC_ERR_MODE`), and resets the array (lanes and
estimator included) with `AEC_ERR_DSP` when the lanes do not share one far spectrum, the
beamformed spectrum is non-finite or a projection overflows (finite but huge weights). With `AEC_RESIDUAL_FULL` a second
call on the same hop is `AEC_ERR_SEQUENCE`. The helper's `array_sample_beam_fuse()` returns
`AEC_ERR_SEQUENCE` without a beam of the hop, `AEC_ERR_MODE` for a `STATE` array, and
resets the whole helper (`array_sample_beam_reset()`: array, SRP, BF, VAD) on any other refusal; the
caller resets its own post stages too, also after a refusal of its own
`array_sample_anr_gain()` or `aec_apply_*`. The linear view lives until the next `aec_process_linear()`, `aec_reset()` or
`aec_destroy()`; there is no token to return. The Align-ULCNet application never fuses: it takes the BF spectrum as
its error frame directly.

### What is in the frame?

| Meaning | Mono: `AecResidualView` (`aec_get_residual()`) | 4-ch: `AecResidualView` (after `aec_process_residual(&AecBeam)`) |
|---|---|---|
| linear-AEC error E(f) | `error_spec` | `error_spec` (beamformed error) |
| capture / far spectra | `near_spec`, `far_spec` | `near_spec` (projected near), `far_spec` (lane 0 far); `echo_spec` is NULL |
| residual-echo and comfort-noise PSD (int16²) | `r2`, `comfort_noise` | `r2`, `comfort_noise` (projected) |
| the residual gain | `res_gain` (the AEC's own `G_res`) | `res_gain` (the array's gain stage over the fused spectra; `aec_get_residual().res_gain`) |
| time-domain hop under E | `formed_hop` | `formed_hop`: NULL (one per lane in `AecArrayView.lane_formed_hop`) |
| aligned far hop | `far_hop` (also `AecLinearView.aligned_far_hop`, `aec_get_linear()`) | `far_hop` |
| the rendering base | `base_spec` (`error_spec`, or `error_spec + echo` on a guard hop) | `base_spec` |
| control scalars | `max_dt`, `max_saturation`, `all_converged`, `usable_linear`, `clock_drift` (`AecLinearView` adds `far_power`, `dt_indicator`, `saturation_level`, `filter_converged`) | `max_dt`, `max_saturation`, `all_converged`, `usable_linear`, `clock_drift` (worst contributing lane) |
| alignment generation flags | not in this view: `AecLinearView.delay_state` (`UNLOCKED` / `LOCKED` / `CHANGED`, `aec_get_linear()`) | not in this view: `AecArrayView.delay.changed` / `.solid` (shared estimator, `aec_get_array()`) |

## NR OLA Delay

The final sqrt-Hann WOLA introduces exactly one hop of full-chain latency
(e.g. 8 ms at the 16 kHz default 256/128 grid). NR gain and the matching AEC
context are computed and applied in the same processing call; the pipeline
does not retain the previous AEC context. The overlap state delays only the
time-domain emission. The AEC-only overlap-save path remains zero-sample
latency.

[返回本主題](../README.md)
