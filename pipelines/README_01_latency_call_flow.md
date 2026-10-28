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
[`ulcnet/4ch_alignulcnet/README.md`](ulcnet/4ch_alignulcnet/README.md)。

The two DFN2 applications have an algorithmic latency of `M + D + 1` hops on a 48 kHz host (`M` = mask lookahead, `D` =
deep-filter lookahead, fixed at build time; the shipped 1/1 pair gives 3 hops, 32 ms). Below 48 kHz the rate bridge is used and
the latency is the host's own WOLA hop plus `dfn_rate_bridge_added_delay_samples()` native samples (shipped 1/1 pair: 672 at
16 kHz hop 128, 629 at 16 kHz hop 256, 330 at 8 kHz hop 128; the floor of the exact value, which `dfn_rate_bridge_added_delay_exact()` returns as a multiple of 1/3 sample, whole only when `M + D + 1` is a multiple of 3), a figure that grows with `M + D`; the other `(M, D)` pairs
follow the same formula but have no trained checkpoint, so their audio quality is unmeasured. See [`dfn/README.md`](dfn/README.md) and [Build](README_02_build_debug_board.md#build).

### Memory Budget

Measured with `"$(make -s print-bin-dir)"/aec_nr_pipeline_static --print-mem-size
--sample-rate <sr> [--fft-size <alt>]` (balanced levels). The AEC column is the
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
| **8 kHz** | 268,992 B | 10,320 B | 28,784 B | 2,624 B | **310,720 B (303.4 KB)** |
| **16 kHz (default, 256/128)** | 366,000 B | 10,320 B | 45,344 B | 2,624 B | **424,288 B (414.3 KB)** |
| **16 kHz (alt, 512/256)** | 494,720 B | 19,792 B | 56,432 B | 5,184 B | **576,128 B (562.6 KB)** |
| **48 kHz** | 1,162,736 B | 38,736 B | 144,576 B | 10,304 B | **1,356,352 B (1,324.6 KB)** |

The AEC column is owned by `lib/aec/docs/c_user_manual_zh_TW_01_lifecycle.md` §4 — re-measure from
there rather than editing it here, and always prefer the values `aec_get_mem_size()`,
`anr_get_mem_size()` and `ac_istft_get_mem_size()` return at runtime. `--no-nr` drops the NR column;
`--aec-only` is the AEC column alone. The synthesis borrows the AEC's synthesis window (`aec_synthesis_window()`), so its column carries no window table.

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072
> taps、6 partitions at hop=512），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮
> `filter_length`（外部參數）。NR 的 `L` 是內部欄位且決定 pool 版面：要縮它就得自己組
> `AnrConfig`（先 `anr_config_for_level_grid()` 再改 `L`）、重新 `anr_get_mem_size()`，
> 該實例之後不再等於裸 level 組態，`anr_set_level()` 會被拒（見 [call sequence](README.md#call-sequence-publicinternal-functions-parameters-at-a-glance) 一節）。
> `n_partitions` 由 `filter_length` 與 hop 推導，不建議手填。三個 rate 各自依 grid 的 hop=fft_size/2
> 規則自動推導 (`aec_derive_dims()`），不是統一的 10 ms 規則；16 kHz 另外還有一個可選的
> 512/256 grid（見上表）。靜態範例的 AEC、NR 與合成 config 都由同一個
> `AecSignalGrid`（`derive_lane()` 的輸出，即 `aec_resolve_signal_grid()` 的結果）導出，三方格點因此一致。

## Call flow

Every application analyses once per hop. The front end returns its context spectra as one borrowed
view: `AecResContext` ([`aec.h`](../lib/aec/c_impl/include/aec.h); read with `aec_get_res_context()` after
`aec_process_context()`) for the mono `Aec`, and `McabFrame` ([`mcab.h`](4ch_aec_bf_nr_res/mcab.h); filled
by `mcab_fuse()`) for the 4-channel flow. Each later stage is a plain function over that view, and the
application writes the combination itself; each example `main` shows it (`mono_form()` in
[`mono_aec_nr_res/main.c`](mono_aec_nr_res/main.c), `raw_chain()` in
[`mcab_raw.c`](4ch_aec_bf_nr_res/mcab_raw.c)):

```
mono:  mic/ref ─► Aec ─► AecResContext { error_spec, near_spec, far_spec, r2, comfort_noise, res_gain, usable_linear, ... }
                        └► aec_get_output_base ─► B = rendering base (E, or E + echo on a guard hop); aec_post_reset_serial ─► reset event
4-ch:  mics/ref ─► mcab_process_front ─► beamformer ─► mcab_fuse ─► McabFrame { e, y, x, r2, cn, g_res_ctx = NULL, far_hop, ... }
                      │
        gain stages   ├─ G_nr: anr_process_gain(nr, E, extra_noise) + anr_get_gain(nr)    (extra_noise = R² / 32768², while the linear estimate is usable)
                      ├─ G_res: mono: ctx.res_gain (the AEC's own)
                      │         4-ch: aec_res_process(res, &in) from the application's AecRes, in from mcab_sample_res_input(&frame, &in)
        combination   └─ sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill → ac_istft_push → out
```

The mono per-hop calls with their signatures (`aec.h`, `aec_res.h`, `anr.h`, `audio_stft.h`,
`utility/simd_kernels.h`); every returned pointer is borrowed from its producer:

```c
void  aec_process_context(Aec* a, const float* mic, const float* ref);     /* hop floats each; no time-domain output */
void  aec_get_res_context(const Aec* a, AecResContext* ctx);              /* error_spec, r2, comfort_noise, res_gain, ...; NULL members = no context */
int   anr_process_gain(AnrDenoiser* self, const Complex* spectrum_in,
                       const float* extra_noise_psd, float* gain_out);     /* 0, or -1; gain_out = NULL: read it with anr_get_gain() */
const float* anr_get_gain(const AnrDenoiser* self, int* n_freqs);          /* n_freqs may be NULL */
void  sk_scale_f32(float* out, const float* x, float g, int n);            /* extra_noise = R^2 * (1/32768^2) */
void  sk_min_f32(float* out, const float* a, const float* b, int n);       /* g_total = min(g_nr, g_res) */
void  sk_capply_gain_f32(Complex* out, const Complex* z, const float* g, int n);   /* out may equal z */
void  aec_cng_fill(uint32_t* rng, Complex* spec, const float* cn, const float* g_res,
                   const float* g_nr, float nr_floor, int n);              /* adds comfort noise to spec; g_nr may be NULL */
int   ac_istft_push(AcIstft* s, const Complex* spec, float* out);          /* writes hop samples (0 only for the first AC_OLA_ENV_NORM frame) */
```

There is no independent time-domain RES filter after NR and no function-pointer
table of stages: `G_res` comes with the linear residual (mono, inside the AEC call) or from the application's own `AecRes`
(4-ch, through `aec_res_process()` after `mcab_fuse()`), is fused with `G_nr` by `sk_min_f32` and applied once to
`E(f)`. Which stages exist, in which order, is read straight from each example's per-hop
function ([`main.c`](mono_aec_nr_res/main.c) `process_hop()` / `mono_form()`,
[`mcab_raw.c`](4ch_aec_bf_nr_res/mcab_raw.c) `raw_post()` / `raw_chain()`). The 4-channel
programs share two small header-only helpers, [`mcab_sample_post.h`](4ch_aec_bf_nr_res/mcab_sample_post.h)
(`mcab_sample_res_input()`, `mcab_sample_anr_gain()`, `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`,
`MCAB_SAMPLE_CNG_RNG_SEED`; tested by `test_mcab_sample_post`); the mono examples keep their own
`#define`s (`CNG_NR_GAIN_FLOOR`, `PSD_SCALE_INV`) in the file and start the comfort noise from the library's
`AEC_CNG_SEED`.

Only the multi-channel shared delay and four-lane flow is one module (`Mcab`, over lanes and an
estimator the application initialises); every other stage (the beamformer and the 4-ch post-beam RES
included) is a separate object that the integrator initialises (its own `mem_size` and `init`) and chains.

| Application | Analysis call | Stages written out | Synthesis |
|---|---|---|---|
| Mono AEC + NR/RES | `aec_process_context()` → `aec_get_res_context()` | `ctx.res_gain`, `anr_process_gain()` + `anr_get_gain()`, `sk_min_f32`, `sk_capply_gain_f32`, `aec_cng_fill` | `ac_istft_push()` |
| 4-ch AEC + BF + NR/RES | `mcab_process_front()` → the beamformer (the sample helper's `mcab_sample_beam_process_beam()` or your own) → `mcab_fuse()` (`frame.g_res_ctx` is NULL) | `mcab_sample_res_input()` + `aec_res_process()` (the application's `AecRes`), `mcab_sample_anr_gain()`, `sk_min_f32`, `sk_capply_gain_f32`, `aec_cng_fill` | `ac_istft_push()` |
| Mono / 4-ch Align-ULCNet | mono: `aec_process_context()`; 4-ch: the flow with `enable_post = 0` and the sample beam helper; then `ulcnet_nn_stage_begin_time()` / `_begin_complex()` | the network (`ulcnet_nn_stage_nn_begin` / `nn_end`) | `ulcnet_nn_stage_end()` |
| Mono / 4-ch DFN2 | mono: `aec_process_context()` → `aec_get_res_context()`; 4-ch: `mcab_sample_beam_process()` → `mcab_sample_beam_fuse()` | the example forms P itself (`rig_process()` / `dfn_form()`: `ctx.res_gain` for mono, `mcab_sample_res_input()` + `aec_res_process()` for 4-ch, `sk_min_f32` with the unity gain, apply, optional comfort noise; no NR, no synthesis of its own), then the DFN2 stage through `dfn_hop_*` | `dfn_hop_end()` |

### How do I chain a new frequency-domain stage?

Write it as a gain-type function over the frame and put its result into the
`sk_min_f32` chain. The example below is compiled and run by
[`tests/test_readme_flow.c`](tests/test_readme_flow.c) (`make test_readme_flow`):
with a unity stage the chain is bit-identical to the hand-written chain of
[`mono_aec_nr_res/tests/conv_chain_mono.h`](mono_aec_nr_res/tests/conv_chain_mono.h).

```c
typedef struct HfTilt { int bin_hi; float amount; float* g; } HfTilt;   /* g: n_freqs floats, caller memory */

static const float* hf_tilt_gain(HfTilt* s, int n_freqs) {
    for (int k = 0; k < n_freqs; ++k) s->g[k] = k > s->bin_hi ? s->amount : 1.0f;
    return s->g;                     /* valid until the next call, like every gain stage */
}

int chain_hop(Chain* c, const float* mic, const float* ref, float* out) {
    AecResContext f;
    AecOutputBase b = { sizeof(b) };
    const float* extra = NULL;
    aec_process_context(c->aec, mic, ref);                                        /* the one analysis */
    aec_get_res_context(c->aec, &f);
    if (!f.error_spec || aec_get_output_base(c->aec, &b) != 0) return -1;         /* b.base_spec: the spectrum to render */
    if (aec_post_reset_serial(c->aec) != c->reset_serial) {                       /* hard post-chain reset in this hop: */
        c->reset_serial = aec_post_reset_serial(c->aec);                          /* before the CNG draw and the push */
        ac_istft_reset(c->istft);  c->rng = AEC_CNG_SEED;
    }
    if (f.r2 && f.usable_linear) {                      /* residual-echo powers into the noise floor */
        sk_scale_f32(c->g_nr_scratch, f.r2, PSD_SCALE_INV, f.n_freqs);
        extra = c->g_nr_scratch;
    }
    if (anr_process_gain(c->nr, b.base_spec, extra, NULL) != 0) return -1;
    const float* nr_gain = anr_get_gain(c->nr, NULL);
    if (!nr_gain) return -1;
    const float* tilt = hf_tilt_gain(c->tilt, f.n_freqs);                          /* the new stage */
    sk_min_f32(c->g_min,   nr_gain,  f.res_gain, f.n_freqs);                       /* min of the gains */
    sk_min_f32(c->g_total, c->g_min, tilt,       f.n_freqs);
    sk_capply_gain_f32(c->spec, b.base_spec, c->g_total, f.n_freqs);               /* S(f) = B(f) . g */
    aec_cng_fill(&c->rng, c->spec, f.comfort_noise, f.res_gain, nr_gain,           /* comfort noise on the cut bins */
                 CNG_NR_GAIN_FLOOR, f.n_freqs);
    (void)ac_istft_push(c->istft, c->spec, out);                                   /* the one inverse transform */
    return 0;
}
```

Rules the chain relies on:

- A disabled gain source reads as the **unity vector** (all ones), so one
  `sk_min_f32` serves every combination. `sk_min_f32` is a compare-and-select (`a < b ? a : b`), not
  `fminf`: a NaN in its second operand reaches the output, a NaN in its first operand is replaced by the second,
  so a finite check on the emitted hop (`skn_all_finite_f32()`) stays the guard against a non-finite gain.
- `aec_cng_fill` is told the AEC's own gain (`g_res`), which alone decides *which*
  bins are filled, and the denoiser gain (floored at `CNG_NR_GAIN_FLOOR`, or
  `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR` in the 4-channel programs), which scales the *level*; it advances the
  caller's `rng` state, so one RNG word per generator, reset to `AEC_CNG_SEED`
  (`MCAB_SAMPLE_CNG_RNG_SEED`) at construction and whenever `aec_post_reset_serial()` moves.
- `ctx.error_spec` / `ctx.formed_hop` are the analysis seam and are never rendered; the chain renders the
  rendering base `aec_get_output_base()` hands out (the same spectrum `aec_process()` multiplies by its gain, `E`
  bit for bit on every hop the output guard leaves alone) and the denoiser estimates on that same spectrum.
- A spectrum-type stage (the 4-channel beamformer) is a function that takes the
  lane spectra and returns a spectrum; it replaces the error spectrum (`f.e` of the
  `McabFrame`) in the chain. The sample helper hands the BF spectrum to `mcab_fuse()` inside
  `mcab_sample_beam_fuse()`.
- Every pointer in the frame is borrowed from the producer. It is valid until the
  producer runs again (the next `aec_process_context()` / `mcab_process_front()`,
  a reset) and is never written through.
- A refused call (a stage returns NULL or an error, or the emitted hop is not finite) is the
  integrator's to recover: reset the front end and every stage of the chain together
  (the comfort-noise generator is restarted too: `rng = AEC_CNG_SEED`).

### Where do I call the inverse transform?

Once per hop, last: `ac_istft_push()` (audiolib) on the spectrum that has all gains applied
and comfort noise added.
It performs the inverse FFT, the sqrt-Hann window and the overlap-add, and emits
exactly one hop; the half-window it holds back is the pipeline's one hop of
algorithmic latency. The spectrum must sit on the analysis frame of the same hop
(the AEC's `error_spec` is the exact 50 %-overlap sqrt-Hann STFT, and the rendering base is `error_spec` or the
capture spectrum of the same frame); calling it
twice in a hop, or skipping a hop, breaks the overlap-add. Applications that end in
a neural stage call it inside `ulcnet_nn_stage_end()` or `dfn_hop_end()`.

### What memory do I provide?

Static memory only: each object is `get_mem_requirements()` (or
`<x>_mem_size()`), one caller block that is 16-byte aligned, then `init_ex()`.
Nothing allocates inside `process`. Every library is queried and initialised on its
own; the integrator sums the sizes and cuts the blocks:

```c
size_t aec_b   = aec_get_mem_size(&aec_cfg);                       /* aec_cfg: aec_config_from_level() + the lane's seam fields; 0 = invalid */
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
  application carved (`anr_process_gain()` takes the `AnrDenoiser`, `aec_res_process()`
  the `AecRes`). Gain vectors the chain needs (`g_total`, the NR prior scratch, the
  spectrum buffer) are `n_freqs` floats / bins each, also from the caller's block.
- `Mcab` owns only the flow state in its own pool (control block, mic lane, aligned far, the fuse
  buffers with `enable_post`, the alignment ring): 74,624 B at 16 kHz/256 (MATCHED,
  `enable_post = 1`; 68,368 B with `enable_post = 0`, `test_target_abi`). The four `Aec` lanes
  (4 x 200,992 B at 16 kHz/256) and, MATCHED only, the shared `DelayAec` are separate objects the
  application sizes and initialises (`aec_get_mem_size()`, `delay_aec_get_mem_size()`);
  `mcab_external_static --print-mem-size` prints lanes 803,968 B and front end (estimator plus the
  mcab pool) 110,048 B at 16 kHz/256 (48 kHz/1024: 2,936,832 B and 267,072 B). The sample beam helper
  (SRP-PHAT, BF, scratch) is one more block. The post-beam stages, the RES included, are separate blocks sized by their own queries
  (`aec_res_get_mem_size()` on an `AecConfig` from `aec_config_from_level()` with the flow's `fft_size`,
  `anr_get_mem_size()`, `ac_istft_get_mem_size()`),
  and a neural stage (`ulcnet_nn_stage_mem_size()`, `dfn_rate_bridge_get_mem_requirements()`)
  is one more block.
- Re-query the descriptor after any library, backend, build-option or config
  change; `mcab_init_ex()` refuses a stale one. Only `Mcab` (`McabMemReq`, its own pool) and the
  DFN stage and bridge carry descriptors; the example headers do not.

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
reads lane 0): `mcab_sample_beam_process_front()` returns the read-only lane
spectra, the caller computes mask, `vad_raw` and `vad_out`,
`mcab_sample_beam_process_beam()` runs SRP and BF with them and fills the
`McabSampleBeamInfo` (DOA, BF adaptation state and `info.delay`).
`mcab_sample_beam_process_with_activity()` is `process_front()` +
`process_beam(front, v, v, NULL)`; see
[`ulcnet/4ch_alignulcnet/README.md`](ulcnet/4ch_alignulcnet/README.md). The stage reads the flow's
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
mcab_process_front(flow, mics_interleaved, ref, &front);  /* shared delay + 4 lanes; front.linear_spectra[4], front.aligned_ref */
/* your beamformer on front.linear_spectra -> weights W (channel-major Complex[4][n_freqs]) and, optionally, its spectrum E */
McabFrame f;
mcab_fuse(flow, W, E, &f);       /* project near, R^2, comfort noise with those weights; E == NULL rebuilds it from W; f.g_res_ctx = NULL */
/* then the application's chain: mcab_sample_res_input(&f, &in) + aec_res_process(res, &in) (its own AecRes),
   mcab_sample_anr_gain(), sk_min_f32(), sk_capply_gain_f32(), aec_cng_fill(), ac_istft_push() */
```

The examples run SRP-PHAT and the beamformer through the sample helper, which sits between the two
`mcab` calls (the flow is then driven only through the helper):

```c
mcab_sample_beam_process_front(b, mics_interleaved, ref, &front);
/* [optional] mask / VAD from the lane spectra */
mcab_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT + BF -> beam.spectrum, beam.weights */
mcab_sample_beam_fuse(b, &f);    /* mcab_fuse() with this hop's beam; f.g_res_ctx = NULL */
```

Neither `Mcab` nor the helper has a RES: `f.g_res_ctx` is NULL, and the caller builds its own `AecRes`
and calls `aec_res_process(res, &in)` on the input `mcab_sample_res_input(&f, &in)` makes (`mcab_static.c`
and `mcab_external_static.c` do).

`mcab_fuse()` needs a front frame processed since the last reset (otherwise
`MCAB_SEQUENCE_ERROR`, nothing changed), refuses an all-zero or non-finite weight set and
`enable_post = 0` (`MCAB_INVALID_ARGUMENT`, nothing changed), and resets the flow (lanes and
estimator included) with `MCAB_DSP_ERROR` when the lanes do not share one far spectrum, the
beamformed spectrum is non-finite or a projection overflows (finite but huge weights). The helper's `mcab_sample_beam_fuse()` returns
`MCAB_SEQUENCE_ERROR` without a beam of the hop, `MCAB_INVALID_ARGUMENT` when `enable_post = 0`, and
resets the whole helper (`mcab_sample_beam_reset()`: flow, SRP, BF, VAD) on any other refusal; the
caller resets its own post stages (the `AecRes` included) too, also after a refusal of its own
`aec_res_process()` (a NULL return). The front view lives until the next `mcab_process_front()`, `mcab_reset()` or
`mcab_destroy()`; there is no token to return. The Align-ULCNet application never fuses: it takes the BF spectrum as
its error frame directly.

### What is in the frame?

| Meaning | Mono: `AecResContext` (`aec_get_res_context()`) | 4-ch: `McabFrame` (`mcab_fuse()`) |
|---|---|---|
| linear-AEC error E(f) | `error_spec` | `e` (beamformed error) |
| capture / far spectra | `near_spec`, `far_spec` | `y` (projected near), `x` (lane 0 far) |
| residual-echo and comfort-noise PSD (int16²) | `r2`, `comfort_noise` | `r2`, `cn` (projected) |
| the residual gain | `res_gain` (the AEC's own `G_res`) | `g_res_ctx`: NULL (`mcab_fuse()` runs no RES; the gain comes from the application's `AecRes`) |
| time-domain hop under E | `formed_hop` | `formed_hop`: NULL (one per lane in `McabFront.formed_hop`) |
| aligned far hop | `AecLinearContext.aligned_far_hop` (`aec_get_linear_context()`) | `far_hop` |
| control scalars | `far_power`, `dt_indicator`, `saturation_level`, `filter_converged`, `usable_linear` | `far_power`, `max_dt`, `max_sat`, `all_converged`, `usable_linear`, `clock_drift` (worst contributing lane) |
| alignment generation flags | `AecLinearContext.delay_state` (`UNLOCKED` / `LOCKED` / `CHANGED`) | `delay_changed`, `delay_locked` (shared estimator) |

## NR OLA Delay

The final sqrt-Hann WOLA introduces exactly one hop of full-chain latency
(e.g. 8 ms at the 16 kHz default 256/128 grid). NR gain and the matching AEC
context are computed and applied in the same processing call; the pipeline
does not retain the previous AEC context. The overlap state delays only the
time-domain emission. The AEC-only overlap-save path remains zero-sample
latency.

Navigation: [Index](README.md) | [Previous part](README.md) | [Next part](README_02_build_debug_board.md) <!-- split-nav -->
