Navigation: [Index](README.md) | [Previous part](README.md) | [Next part](README_02_build_debug_board.md) <!-- split-nav -->
## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **AEC linear path** | 0 samples | PBFDKF overlap-save 的當前 hop 線性誤差可立即取得 |
| **Synthesis WOLA delay** | 1 hop | NR/RES 頻譜由 sqrt-Hann WOLA 重建；這就是 full-chain 的演算法延遲 |
| **Pipeline total latency** | 1 hop — 16 ms @ 8 kHz, 8 ms @ 16 kHz default, 16 ms @ 16 kHz alt, ~10.7 ms @ 48 kHz | 不額外保存上一個 AEC context；AEC-only 為 0 samples |
| **Processing (per hop)** | 待量測 | 目標平台上 AEC + NR + RES 合計的實測；主機端用 `--timing`（`make PROFILE=1`）取各階段 µs |
| **RTF** | 待量測 | 同上；由實測 per-hop 時間除以 hop 時長 |

上表是 conventional mono pipeline 的數字。兩條 Align-ULCNet application 的
加總額外延遲同樣是 **1 hop**（唯一來源是 ULCNet synthesis WOLA）：mono 與
4ch 變體都是——4ch 把 BF 的 beamformed error 頻譜直接當成模型的分析幀，
沒有中間的 beam WOLA→再 analysis 來回，也沒有 one-hop far 補償。細節見
[`4ch_alignulcnet/README.md`](4ch_alignulcnet/README.md)。

### Memory Budget

Measured with `"$(make -s print-bin-dir)"/aec_nr_pipeline_static --print-mem-size
--sample-rate <sr> [--fft-size <alt>]` (balanced presets). The AEC column is the
composite `aec_get_mem_size()` pool — it already contains HPF, PBFDKF ×2
(main+shadow), delay estimator, the RES/post context and the AEC-internal FFTs.
The remaining columns are the example's other stages, each sized by its own query
(`ac_istft_get_mem_size()` for the synthesis `AcIstft`, `anr_get_mem_size()` for the
denoiser) plus the gain and spectrum buffers the per-hop chain needs; the mono chain takes
the AEC's own gain, so there is no `AecRes`. The NE10
twiddle configs are carved from these pools too, so the columns are the complete
memory requirement (strict init→destroy zero-heap). 16 kHz's default
grid is 256/128 (see [Parameter Alignment](README.md#parameter-alignment)); the alternate 512/256 grid
stays explicitly selectable.

| Rate | AEC | Synthesis (`AcIstft`) | NR | Gain/spectrum buffers | **Total** |
|------|-----|-----------|-----|---------------|-----------|
| **8 kHz** | 267,408 B | 11,344 B | 28,784 B | 2,624 B | **310,160 B (302.9 KB)** |
| **16 kHz (default, 256/128)** | 364,416 B | 11,344 B | 45,344 B | 2,624 B | **423,728 B (413.8 KB)** |
| **16 kHz (alt, 512/256)** | 491,600 B | 21,840 B | 56,432 B | 5,184 B | **575,056 B (561.6 KB)** |
| **48 kHz** | 1,156,544 B | 42,832 B | 144,576 B | 10,304 B | **1,354,256 B (1,322.5 KB)** |

The AEC column is owned by `lib/aec/docs/c_user_manual_zh_TW_01_lifecycle.md` §4 — re-measure from
there rather than editing it here, and always prefer the values `aec_get_mem_size()`,
`anr_get_mem_size()` and `ac_istft_get_mem_size()` return at runtime. `--no-nr` drops the NR column;
`--aec-only` is the AEC column alone.

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072
> taps、6 partitions at hop=512），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮
> `filter_length`（外部參數）。NR 的 `L` 是內部欄位且決定 pool 版面：要縮它就得自己組
> `AnrConfig`（先 `anr_config_for_mode_grid()` 再改 `L`）、重新 `anr_get_mem_size()`，
> 該實例之後不再等於裸 preset，`anr_set_mode()` 會被拒（見 [call sequence](README.md#call-sequence-publicinternal-functions-parameters-at-a-glance) 一節）。
> `n_partitions` 由 `filter_length` 與 hop 推導，不建議手填。三個 rate 各自依 grid 的 hop=fft_size/2
> 規則自動推導 (`aec_derive_dims()`），不是統一的 10 ms 規則；16 kHz 另外還有一個可選的
> 512/256 grid（見上表）。靜態範例的 AEC、NR 與合成 config 都由同一個
> `SeGrid`（`se_derive()` 的輸出）導出，三方格點因此一致。

## Call flow

Every application analyses once per hop. The front end (the mono `Aec`, or the 4-channel
unit's `audio_pipeline_4ch_fuse()`) returns its context spectra as one borrowed view,
`SeSpecFrame` ([`common/se_stage.h`](common/se_stage.h); `se_frame_from_aec()` fills it from a mono
`Aec`). Each later stage is a plain function over that frame, and the application
writes the combination itself; each example `main` shows it (`mono_form()` in
[`mono_aec_nr_res/main.c`](mono_aec_nr_res/main.c), `raw_chain()` in
[`audio_pipeline_4ch_raw.c`](4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c)):

```
mic/ref ─► AEC ─► SeSpecFrame { e, y, x, r2, cn, g_res_ctx, far_hop, ... }
                      │
        gain stages   ├─ se_anr_stage_gain(nr, &f, scratch, use_r2)  → G_nr
                      ├─ G_res = f.g_res_ctx                         (mono: the AEC's own ctx.res_gain;
                      │                                               4-ch: g_res_ctx is NULL, G_res = se_res_stage_gain(res, &f)
                      │                                               from the application's AecRes)
        combination   └─ sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill → ac_istft_push → out
```

There is no independent time-domain RES filter after NR and no function-pointer
table of stages: `G_res` comes with the linear residual (mono, inside the AEC call) or from the application's own `AecRes`
(4-ch, through `se_res_stage_gain()` after `audio_pipeline_4ch_fuse()`), is fused with `G_nr` by `sk_min_f32` and applied once to
`E(f)`. Which stages exist, in which order, is read straight from each example's per-hop
function ([`main.c`](mono_aec_nr_res/main.c) `process_hop()` / `mono_form()`,
[`audio_pipeline_4ch_raw.c`](4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c) `raw_post()` / `raw_chain()`).

Only the multi-channel linear AEC + beamformer is wrapped as one unit; every other
stage (the 4-ch post-beam RES included) is a separate library that the integrator initialises (its own `mem_size` and `init`) and chains.

| Application | Analysis call | Stages written out | Synthesis |
|---|---|---|---|
| Mono AEC + NR/RES | `aec_process_context()` → `se_frame_from_aec()` | `frame.g_res_ctx`, `se_anr_stage_gain`, `sk_min_f32`, `sk_capply_gain_f32`, `aec_cng_fill` | `ac_istft_push()` |
| 4-ch AEC + BF + NR/RES | `audio_pipeline_4ch_process_front()` + `process_beam()` (or `process_with_activity()`) → `audio_pipeline_4ch_fuse()` (`frame.g_res_ctx` is NULL) | `se_res_stage_gain` (the application's `AecRes`), `se_anr_stage_gain`, `sk_min_f32`, `sk_capply_gain_f32`, `aec_cng_fill` | `ac_istft_push()` |
| Mono / 4-ch Align-ULCNet | mono: `aec_process_context()`; 4-ch: the unit with `enable_post = 0`; then `ulcnet_nn_stage_begin_time()` / `_begin_complex()` | the network (`ulcnet_nn_stage_nn_begin` / `nn_end`) | `ulcnet_nn_stage_end()` |
| Mono / 4-ch DFN2 | mono: `aec_process_context()` → `se_frame_from_aec()`; 4-ch: `audio_pipeline_4ch_process()` → `audio_pipeline_4ch_fuse()` | the example forms P itself (`rig_process()` / `dfn_form()`: `frame.g_res_ctx` for mono, `se_res_stage_gain()` for 4-ch, `sk_min_f32` with the unity gain, apply, optional comfort noise; no NR, no synthesis of its own), then the DFN2 stage through `dfn_hop_*` | `dfn_hop_end()` |

### How do I chain a new frequency-domain stage?

Write it as a gain-type function over the frame and put its result into the
`sk_min_f32` chain. The example below is compiled and run by
[`tests/test_readme_flow.c`](tests/test_readme_flow.c) (`make test_readme_flow`):
with a unity stage the chain is bit-identical to the hand-written chain of
[`mono_aec_nr_res/tests/conv_chain_mono.h`](mono_aec_nr_res/tests/conv_chain_mono.h).

```c
typedef struct HfTilt { int bin_hi; float amount; float* g; } HfTilt;   /* g: n_freqs floats, caller memory */

static const float* hf_tilt_gain(HfTilt* s, const SeSpecFrame* f) {
    for (int k = 0; k < f->n_freqs; ++k) s->g[k] = k > s->bin_hi ? s->amount : 1.0f;
    return s->g;                     /* valid until the next call, like every gain stage */
}

int chain_hop(Chain* c, const float* mic, const float* ref, float* out) {
    SeSpecFrame f;
    aec_process_context(c->aec, mic, ref);                                        /* the one analysis */
    if (se_frame_from_aec(c->aec, &f) != 0) return -1;
    const float* nr_gain = se_anr_stage_gain(c->nr, &f, c->g_nr_scratch, 1);
    const float* tilt    = hf_tilt_gain(c->tilt, &f);                              /* the new stage */
    sk_min_f32(c->g_min,   nr_gain,  f.g_res_ctx, f.n_freqs);                      /* min of the gains */
    sk_min_f32(c->g_total, c->g_min, tilt,        f.n_freqs);
    sk_capply_gain_f32(c->spec, f.e, c->g_total, f.n_freqs);                       /* S(f) = E(f) . g */
    aec_cng_fill(&c->rng, c->spec, f.cn, f.g_res_ctx, nr_gain,                     /* comfort noise on the cut bins */
                 SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
    (void)ac_istft_push(c->istft, c->spec, out);                                   /* the one inverse transform */
    return 0;
}
```

Rules the chain relies on:

- A disabled gain source reads as the **unity vector** (all ones), so one
  `sk_min_f32` serves every combination. `sk_min_f32` propagates a non-finite gain
  to the output gate instead of masking it the way `fminf` would.
- `aec_cng_fill` is told the AEC's own gain (`g_res`), which alone decides *which*
  bins are filled, and the denoiser gain (floored at `SE_CNG_NR_GAIN_FLOOR`),
  which scales the *level*; it advances the caller's `rng` state, so one RNG word
  per generator, reset to `SE_CNG_RNG_SEED`.
- A spectrum-type stage (the 4-channel beamformer) is a function that takes the
  lane spectra and returns a spectrum; it replaces `f.e` in the chain. The unit
  hands the BF spectrum to `four_aec_fuse()` inside `audio_pipeline_4ch_fuse()`.
- Every pointer in the frame is borrowed from the producer. It is valid until the
  producer runs again (the next `aec_process_context()` / `audio_pipeline_4ch_process_front()`,
  a reset) and is never written through.
- A refused call (a stage returns NULL or an error, or the emitted hop is not finite) is the
  integrator's to recover: reset the front end and every stage of the chain together
  (the comfort-noise generator is restarted too: `rng = SE_CNG_RNG_SEED`).

### Where do I call the inverse transform?

Once per hop, last: `ac_istft_push()` (audiolib) on the spectrum that has all gains applied
and comfort noise added.
It performs the inverse FFT, the sqrt-Hann window and the overlap-add, and emits
exactly one hop; the half-window it holds back is the pipeline's one hop of
algorithmic latency. The spectrum must sit on the analysis frame of the same hop
(the AEC's `error_spec` is the exact 50 %-overlap sqrt-Hann STFT); calling it
twice in a hop, or skipping a hop, breaks the overlap-add. Applications that end in
a neural stage call it inside `ulcnet_nn_stage_end()` or `dfn_hop_end()`.

### What memory do I provide?

Static memory only: each object is `get_mem_requirements()` (or
`<x>_mem_size()`), one caller block that is 16-byte aligned, then `init_ex()`.
Nothing allocates inside `process`. Every library is queried and initialised on its
own; the integrator sums the sizes and cuts the blocks:

```c
size_t aec_b   = aec_get_mem_size(&aec_cfg);                       /* aec_cfg from se_derive(); 0 = invalid */
AnrConfig nr_cfg = anr_config_for_mode_grid(sample_rate, grid.fft_size, ANR_MODE_BALANCED);
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
  application carved (`se_anr_stage_gain` takes the `AnrDenoiser`, `se_res_stage_gain`
  the `AecRes`). Gain vectors the chain needs (`g_total`, the NR prior scratch, the
  spectrum buffer) are `n_freqs` floats / bins each, also from the caller's block.
- The 4-channel core (`FourAecNrRes`) owns only the shared aligner, the four
  lanes and the fuse buffers; the unit (`AudioPipeline4Ch`) adds SRP-PHAT, the BF and their
  scratch (no RES). The post-beam stages, the RES included, are separate blocks sized by their own queries
  (`aec_res_get_mem_size()` on an `AecConfig` from `aec_config_from_preset()` with the unit's `fft_size`,
  `anr_get_mem_size()`, `ac_istft_get_mem_size()`),
  and a neural stage (`ulcnet_nn_stage_mem_size()`, `dfn_rate_bridge_get_mem_requirements()`)
  is one more block.
- Re-query the descriptor after any library, backend, build-option or config
  change; `init_ex()` refuses a stale one (8-point gate). Only the 4-channel core and unit
  and the DFN stage and bridge carry descriptors; the example headers do not.

### How do I run the NN stage on the NPU?

The network sits between two calls. The example stage does everything before (analyses,
delay policy) and after (accept or fail open, overlap-add); you run the
accelerator on the views it hands out. The loop is the same for the four neural
applications; the examples are compiled by `tests/test_readme_flow.c`.

```c
/* Align-ULCNet (mono or 4-ch): at most one frame per hop */
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay); /* far analysis, reprime / model reset; mono:
                                                                    ulcnet_nn_stage_begin_time(stage, formed_hop, ref, delay) */
if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)                /* job: err/far spectra, NaN-prefilled out_re/out_im */
    ulcnet_nn_stage_nn_end(stage, run_ulcnet_npu(&job));       /* 0 = all bins written; else the identity */
ulcnet_nn_stage_end(stage, out);                               /* WOLA: one hop, zeros at hop 0   */

/* DFN2 (mono or 4-ch): up to two frames per hop through the 8/16 kHz rate bridge */
dfn_host_begin(&host, mic, ref);                               /* AEC, RES gain, P = E.G_res; dfn_hop_begin(&hop, e, p) */
while (dfn_hop_nn_begin(&hop, &job) == 1)                      /* job: DFN2PrepostInputs / Outputs */
    dfn_hop_nn_end(&hop, run_npu(&job));
dfn_hop_end(&hop, out);
```

The 4-ch Align-ULCNet application also splits before the beamformer, for a
caller whose mask and VAD come from the lane spectra of the same hop (the masker
reads lane 0): `audio_pipeline_4ch_process_front()` returns the read-only lane
spectra, the caller computes mask, `vad_raw` and `vad_out`,
`audio_pipeline_4ch_process_beam()` runs SRP and BF with them and fills the
`AudioPipeline4ChFrameInfo` (DOA and BF adaptation state).
`audio_pipeline_4ch_process_with_activity()` is `process_front()` +
`process_beam(front, v, v, NULL)`; see
[`4ch_alignulcnet/README.md`](4ch_alignulcnet/README.md). The stage reads the unit's
`front.delay` (`changed`, `solid`) and the caller's raw far hop; nothing else crosses.

Fail-open is part of the contract, not an option: a non-zero result, an output
with any non-finite value, a frame left open and a frame never fetched all take
the identity for that frame (ULCNet: the error spectrum passes; DFN2: a unit band
mask), the framing and the clocks advance exactly once, and the latency never
changes. `ulcnet_nn_stage_process()` and `dfn_hop_process()` are this same
sequence driven by the model callback installed on the stage, so a board with no
callback and a host with one produce identical bytes. The model's `reset` callback
(if set) is still called on an alignment boundary and by the stage's `reset()`, so the
board keeps its recurrent state in step.

### How do the 4-channel applications split the hop?

```c
audio_pipeline_4ch_process_front(unit, mics_interleaved, ref, &front);   /* shared aligner + 4 lanes; front.linear_spectra[4] */
/* [optional] mask / VAD from the lane spectra */
audio_pipeline_4ch_process_beam(unit, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT + BF -> beam.spectrum, beam.weights */
SeSpecFrame f;
audio_pipeline_4ch_fuse(unit, &f);       /* project near, R^2, comfort noise with those weights (four_aec_fuse()); f.g_res_ctx = NULL */
/* then the application's chain: se_res_stage_gain(res, &f) (its own AecRes), se_anr_stage_gain(), sk_min_f32(),
   sk_capply_gain_f32(), aec_cng_fill(), ac_istft_push() */
```

A caller that runs its own beamformer uses the core instead:
`four_aec_front_process(core, ...)`, `bf_process_with_weights(...)`, `four_aec_fuse(core, bf_weights,
bf_spectrum, &f)`. Neither the core nor the unit has a RES: `f.g_res_ctx` is NULL, and the caller builds its own `AecRes`
and calls `se_res_stage_gain(res, &f)` (`audio_pipeline_4ch_static.c` and `4aec_nr_res_static.c` do).

`four_aec_fuse()` needs a front frame processed since the last reset (otherwise
`FOUR_AEC_NR_RES_SEQUENCE_ERROR`), refuses an all-zero or non-finite weight set
and resets the core on a lane disagreement. The unit's `audio_pipeline_4ch_fuse()` returns
`SEQUENCE_ERROR` without a beam of the hop, `INVALID_ARGUMENT` when `core.enable_post = 0`, and
resets the whole unit (core, SRP, BF) on any other refusal; the caller resets its own post stages (the `AecRes` included) too, also after a refusal of its own `se_res_stage_gain()`.
The front view lives until the next `process_front()` or `audio_pipeline_4ch_reset()`; there is
no token to return. The Align-ULCNet application never fuses: it takes the BF spectrum as
its error frame directly.

### What is in the frame?

| Field | Meaning | Mono | 4-ch |
|---|---|---|---|
| `e` | linear-AEC error E(f) | AEC error | beamformed error |
| `y`, `x` | matching capture / far spectra | AEC | projected near / lane 0 far |
| `r2`, `cn` | residual-echo and comfort-noise PSD (int16²) | AEC | projected |
| `g_res_ctx` | the residual gain | the AEC's own `G_res` | NULL (the unit's `fuse()` and the core's `four_aec_fuse()` do not run a RES; the gain comes from the application's `AecRes`) |
| `far_hop`, `formed_hop` | aligned far, time-domain hop under `e` | AEC | aligned far / NULL |
| `max_dt`, `max_sat`, `all_converged`, `clock_drift` | control scalars | AEC | worst contributing lane |
| `delay_changed`, `delay_locked` | alignment generation flags | AEC | shared aligner |

## NR OLA Delay

The final sqrt-Hann WOLA introduces exactly one hop of full-chain latency
(e.g. 8 ms at the 16 kHz default 256/128 grid). NR gain and the matching AEC
context are computed and applied in the same processing call; the pipeline
does not retain the previous AEC context. The overlap state delays only the
time-domain emission. The AEC-only overlap-save path remains zero-sample
latency.

Navigation: [Index](README.md) | [Previous part](README.md) | [Next part](README_02_build_debug_board.md) <!-- split-nav -->
