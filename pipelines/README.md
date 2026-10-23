# C Pipeline: linear AEC + echo-aware NR/RES gain fusion

## ARM target selection

C deployment supports both ARM EABI32 and AArch64 builds. The target compiler's
predefined macros select the path: AArch64 uses the guarded NEON model kernels;
ARM32 uses their existing scalar implementations. `SIMD=0` still forces scalar.
The shared FFT independently selects its available target implementation; select
the board compiler and ABI flags rather than editing source guards. Signal grids,
model state tensors and latency do not change with pointer width.

For ARM32 NEON use `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` on
AArch32 ARMv8 cores), not plain `-mfpu=neon`: without hardware FMA each explicit
`fmaf()` in the AEC, NR and shared kernels becomes a libm call (same bits, much
slower). ARMv7 Advanced SIMD flushes subnormals to zero and returns the default
NaN, so `SIMD=1` and `SIMD=0` can differ on ARM32 for subnormal or NaN-payload
values; only the NE10 NEON FFT and, on cores with fused FMA
(`-mfpu=neon-vfpv4`), the AEC delay matched-filter kernels are affected.

Public memory descriptors remain 32 bytes. Their build hashes add an ILP32 token
on 32-bit targets and preserve the existing hashes on 64-bit targets. Re-query
requirements for the target build; `_init_ex` rejects a descriptor from the other
ABI before touching the supplied pool.

`make target-abi-test` builds and runs the 8-interface descriptor test with the
same target compiler/producer settings; `make test_target_abi` builds it without
running. The keyed `test_target_abi` binary accepts `--write descriptor.bin` and
`--reject descriptor.bin`. Exchange files between 32/64-bit builds with the same
model/grid settings: rejection must leave caller memory unchanged even after the
foreign byte budget is normalized to the current requirement.


## Architecture

```
mic/ref → linear AEC (aec_process_context(): context-only, no emit)
              ├─ E(f) ────────────────→ echo-aware NR ─→ G_nr ─┐
              └─ AecResContext {R², G_res, CNG, far power} ─────┤
                                                                ↓
                                  G_total = min(G_nr, G_res) + CNG
                                                                ↓
                                                       one iFFT/OLA

aec_only mode (no NR/RES stage at all):
mic/ref → linear AEC (aec_process(): emits into out) → out
```

There is no pipeline library behind this diagram: the AEC, the denoiser, the RES object and the
synthesis are separate libraries that the application initialises (each with its own
`mem_size` and `init`) and chains; `samples/se_sample_chain.h` is the example chain. There is no
independent time-domain RES filter after NR. In the mono flow `G_res` is calculated
inside the AEC3 suppression path, exported with the linear residual, fused with
`G_nr`, and applied once to `E(f)`.

The non-`aec_only` path never reads `aec_process()`'s own returned audio —
only `AecResContext`'s linear residual matters downstream — so it drives AEC
via `aec_process_context()` instead: same linear filter and AEC3
post/RES-context computation, minus the copy into an `out` buffer neither the
example nor the caller ever reads on that path. The mono examples pick one entry
point at start-up from `--aec-only`; the two entry points advance identical state and
may be mixed freely on one instance.

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libanr.a | anr.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a | aec_res.h (`AecRes`) / aec.h (`AecResContext`) | Independent post-beam frequency gain API; mono also exposes the AEC-owned gain through its context |
| Mono | no unit: `Aec` itself | aec.h, common/se_stage.h (`se_frame_from_aec()`) | One-mic AEC; the application calls `aec_process_context()` and chains RES gain / NR / comfort noise / synthesis (example: samples/se_sample_chain.h) |
| 4-ch front end | libaudio_pipeline_4ch.a | 4ch_aec_bf_nr_res/4aec_nr_res.h | One shared aligner + four linear AECs + the fuse (`four_aec_front_process()`, `four_aec_fuse()`); the post-beam stages are the application's |
| 4-ch linear AEC + BF unit | libaudio_pipeline_4ch.a | 4ch_aec_bf_nr_res/audio_pipeline_4ch.h | The one wrapped unit: front end + reusable SRP-PHAT/BF libraries; per hop `process_front()` / `process_beam()` (or `process_with_activity()`) give the BF spectrum and weights, `audio_pipeline_4ch_fuse()` is the optional step that fills a `SeSpecFrame`. RES, NR, comfort noise, synthesis and neural post-filters are not in it |
| DFN2 stage / rate bridge | libaudio_pipeline_dfn.a (`pipelines/` build), libdfn_res_stage.a (4-ch build) | dfn_res_stage.h, dfn_rate_bridge.h | The post-RES dual-input DFN2 stage (48 kHz directly, below 48 kHz through the bridge); the hop driver is `samples/dfn_hop.h` |
| Example code | not a library: headers and objects of the examples | samples/ | `se_sample_chain.h`, `se_anr_stage.h`, `dfn_hop.h`, `ulcnet_nn_stage.{h,c}`: no ABI, no layout version, in no archive |

RES is available as an independent frequency-domain `AecRes` object in
`aec_res.h`, shipped in `libaec.a`. The four-channel application owns one such
object after the fuse; `se_res_stage_gain()` calls `aec_res_process()`
without another transform. NR independently calls `anr_process_gain()` on the
same fused `frame.e`; the application merges the two gains, applies them once,
and performs one final inverse transform/OLA. This is gain fusion, not a
RES-applied spectrum feeding NR.

The conventional mono application instead takes the AEC-owned `res_gain` from
`AecResContext` (`g_res_ctx` in `SeSpecFrame`). With `return_res_context=1` and
`enable_res=0`, `aec_process_context()` advances the linear filter and context
calculation without copying returned time audio. The `aec_process()`
convenience API also emits time audio; with RES enabled it applies its own
suppression/synthesis internally. That full convenience path is distinct from
the four-channel linear front → SRP/BF → fuse → gain stages flow.
See `lib/aec/c_impl/include/aec.h` and `aec_res.h` for ownership and inputs.

**`error_spec` does not always mean "error."** That configuration —
`enable_res=0` with `return_res_context=1` — is the one mode where `lib/aec`'s
formed selector opens a third candidate: on a hop the filtering-quality
analyzer has not yet cleared, whose selected residual carries more energy than
the mic capture, `formed_hop` and `error_spec` both become the capture itself,
faded in over the existing 30-sample transition. NR/RES still run on it as
usual — this is the steady-state fallback for a filter that is adding signal
rather than cancelling. `aec_process()`'s own emitted audio is unaffected.

The four-channel API is a separate zero-padding-free grid. See [the 4-channel contract](4ch_aec_bf_nr_res/README.md) and
[`4ch_aec_bf_nr_res/4aec_nr_res.h`](4ch_aec_bf_nr_res/4aec_nr_res.h) for its two
front calls (`four_aec_front_process()`, `four_aec_fuse()`) around the beamformer, and
[`4ch_aec_bf_nr_res/audio_pipeline_4ch.h`](4ch_aec_bf_nr_res/audio_pipeline_4ch.h) for the unit that owns the SRP-PHAT and BF.

The directory root contains only the aggregate build, documentation,
requirements and the public Python reference module. C sources live with
their applications. Shared ULCNet model-I/O support is owned by
`../AIAEC/Align_ULCNet/`; offline scoring and benchmark drivers remain under
`tools/`.

## Application layout

Only the multi-channel linear AEC + beamformer is wrapped as one unit (`AudioPipeline4Ch`).
The product flows are compositions: each directory's `main.c` (and the static, raw and mcab
drivers) initialises the libraries one by one and chains them as an example; the chaining headers
live in [`samples/`](samples/README.md):

| Application | Entry directory | Composition |
|---|---|---|
| Mono AEC + NR/RES | `mono_aec_nr_res/` | `Aec` + `se_frame_from_aec()` + `se_sample_chain` (the AEC's own RES gain, NR, comfort noise, synthesis) + audiolib |
| 4-ch AEC + BF + NR/RES | `4ch_aec_bf_nr_res/` | the unit (4-lane front end + DOA/BF) + `audio_pipeline_4ch_fuse()` + `se_sample_chain` (post-beam `AecRes`, NR, comfort noise, synthesis) + audiolib |
| Mono Align-ULCNet | `mono_alignulcnet/` | `Aec` + `ulcnet_nn_stage` + AIAEC pre/post |
| 4-ch Align-ULCNet | `4ch_alignulcnet/` | the unit with `enable_post = 0` + `ulcnet_nn_stage` + AIAEC pre/post |
| Mono DFN2 | `mono_aec_dfn_res/` | `Aec` + `se_sample_chain` (RES gain, no NR, no synthesis) + `dfn_hop` + the dual-input DFN2 stage |
| 4-ch DFN2 | `4ch_aec_bf_dfn_res/` | 4-lane core + `four_aec_fuse()` (uniform weights in the skeleton) + `se_sample_chain` + `dfn_hop` + the dual-input DFN2 stage |

The mono, 4-channel and mono ULCNet examples take their signal grid, AEC
configuration and delay-request rules from `common/se_config.{h,c}`
(`se_derive()`; each application passes its own `SeRules`: rate set, strict
delay check, lane alignment, seam context). The NR examples use the NR
library's own strength preset on the AEC grid (`anr_config_for_mode_grid()`,
no overrides), so `anr_set_mode()` changes the strength of a running denoiser;
`samples/se_anr_stage.h` holds the per-hop gain call that folds the AEC
residual-echo powers into the noise floor.

The ULCNet applications use `../AIAEC/Align_ULCNet/ulcnet_accelerator_adapter.*`: CPU memory owns
K/V, logit and GRU state, while the board implements one stateless tensor
invocation callback. An absent/failing accelerator is fail-open. The adapter
drives an `ULCNET_IO_FREQ` instance of the `ulcnet_prepost` pre/post class
(spectrum in, spectrum out, no transform of its own), so both application
Makefiles link `ulcnet_prepost.o` wherever the adapter is linked — see
`ULCNET_ADAPTER_OBJS` in [`Makefile`](Makefile) and
[`4ch_aec_bf_nr_res/Makefile`](4ch_aec_bf_nr_res/Makefile).

## Parameter Alignment

All modules use an explicit, zero-padding-free 50%-overlap grid (frame ==
fft_size, hop == fft_size/2), auto-configured by sample rate. 16 kHz has two
selectable grids; the others have exactly one:

| Parameter | 8 kHz | 16 kHz (default) | 16 kHz (alt) | 48 kHz | Formula |
|-----------|-------|-------------------|--------------|--------|---------|
| frame_size / fft_size | 256 | 256 | 512 | 1024 | frame == fft_size, no padding |
| hop_size | 128 | 128 | 256 | 512 | frame / 2 |
| n_freqs | 129 | 129 | 257 | 513 | fft/2 + 1 |
| filter_length | 416 | 832 | 832 | 3072 | ms-derived: sr × 52ms (64ms ≥44.1 kHz) |
| n_partitions | 4 | 7 | 4 | 6 | ceil(filter_length / hop) |

## Four-channel sample driver

The four 4-channel sample mains use `4ch_aec_bf_nr_res/mcab_sample_driver.{c,h}`
for option scanning and the setup → read → process → finite check → consume →
finish → cleanup sequence. Each main supplies typed pipeline/model adapters;
the driver is linked into these executables and is excluded from delivered
algorithm archives.

| Executable | Input and lifecycle |
|---|---|
| `audio_pipeline_4ch_raw` | Complete interleaved float32 raw hops, optional per-hop VAD, heap pipeline; trailing incomplete hops are ignored. |
| `audio_pipeline_4ch_static` | 48 generated tone hops and an explicit VAD schedule; query/allocate/init_ex with caller-owned pool. |
| `4ch_aec_bf_dfn_res` | Eight silent fail-open model hops; DFN2 matrix, delay-profile and descriptor setup stays local. |
| `4ch_alignulcnet` | Four silent fail-open model hops; descriptor/accelerator pool and model installation stay local. |

Model samples report success after resource destruction. The shared hop loop
performs no allocation. Run `make -C 4ch_aec_bf_nr_res test-mcab-sample-driver`
for CLI boundary, short-circuit, finite-output and cleanup-order regressions;
this gate also runs through that directory's `make test`.

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **AEC linear path** | 0 samples | PBFDKF overlap-save 的當前 hop 線性誤差可立即取得 |
| **Synthesis WOLA delay** | 1 hop | NR/RES 頻譜由 sqrt-Hann WOLA 重建；這就是 full-chain 的演算法延遲 |
| **Pipeline total latency** | 1 hop — 16 ms @ 8 kHz, 8 ms @ 16 kHz default, 16 ms @ 16 kHz alt, ~10.7 ms @ 48 kHz | 不額外保存上一個 AEC context；AEC-only 為 0 samples |
| **Processing (per hop)** | < 0.5 ms | AEC + NR + RES 合計(ARM Cortex-A53 @ 1GHz 估計) |
| **RTF** | < 0.05 | 遠低於即時要求 |

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
The remaining columns are the example chain (`se_sample_chain_mem_size()`: synthesis
`AcIstft`, the denoiser, and the chain's control block with its gain and spectrum
buffers; the mono chain takes the AEC's own gain, so there is no `AecRes`). The NE10
twiddle configs are carved from these pools too, so the columns are the complete
memory requirement (strict init→destroy zero-heap). 16 kHz's default
grid is 256/128 (see "Parameter Alignment" above); the alternate 512/256 grid
stays explicitly selectable.

| Rate | AEC | Synthesis (`AcIstft`) | NR | Chain control + bufs | **Total** |
|------|-----|-----------|-----|---------------|-----------|
| **8 kHz** | 284,352 B | 11,344 B | 28,784 B | 2,752 B | **327,232 B (319.6 KB)** |
| **16 kHz (default, 256/128)** | 397,744 B | 11,344 B | 45,344 B | 2,752 B | **457,184 B (446.5 KB)** |
| **16 kHz (alt, 512/256)** | 525,440 B | 21,840 B | 56,432 B | 5,312 B | **609,024 B (594.8 KB)** |
| **48 kHz** | 1,224,176 B | 42,832 B | 144,576 B | 10,432 B | **1,422,016 B (1,388.7 KB)** |

The AEC column is owned by `lib/aec/docs/c_user_manual_zh_TW.md` §4 — re-measure from
there rather than editing it here, and always prefer the values `aec_get_mem_size()` and
`se_sample_chain_mem_size()` return at runtime. `--no-nr` drops the NR column;
`--aec-only` is the AEC column alone.

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072
> taps、6 partitions at hop=512），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮
> `filter_length` 與 NR 的 `L`（48 kHz 也可用 `n_partitions` override 換較短
> 尾巴）。三個 rate 各自依 grid 的 hop=fft_size/2 規則自動推導
> (`aec_derive_dims()`），不是統一的 10 ms 規則；16 kHz 另外還有一個可選的
> 512/256 grid（見上表）。靜態範例在 init 以 grid assert 驗證 AEC/FFT/NR
> 三方一致。

## Call flow

Every application analyses once per hop. The front end (the mono `Aec`, or the 4-channel
unit's `audio_pipeline_4ch_fuse()`) returns its context spectra as one borrowed view,
`SeSpecFrame` ([`common/se_stage.h`](common/se_stage.h); `se_frame_from_aec()` fills it from a mono
`Aec`). Each later stage is a plain function over that frame, and the application
writes the combination itself; [`samples/se_sample_chain.h`](samples/se_sample_chain.h) is the
example that does it:

```
mic/ref ─► AEC ─► SeSpecFrame { e, y, x, r2, cn, g_res_ctx, far_hop, ... }
                      │
        gain stages   ├─ se_anr_stage_gain(nr, &f, scratch, use_r2)  → G_nr
                      ├─ se_res_stage_gain(res, &f)                  → G_res   (4-ch: post-beam AecRes)
                      │                                                 (mono: the AEC's own, f.g_res_ctx)
        combination   └─ sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill → ac_istft_push → out
```

There is no independent time-domain RES filter after NR and no function-pointer
table of stages: `G_res` comes with the linear residual (mono) or from the post-beam `AecRes`
(4-ch), is fused with `G_nr` by `sk_min_f32` and applied once to `E(f)`. Which stages exist, in
which order, is read straight from `se_sample_chain_form()` and from each example's per-hop
function ([`main.c`](mono_aec_nr_res/main.c) `process_hop()`,
[`audio_pipeline_4ch_raw.c`](4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c) `raw_post()`).

Only the multi-channel linear AEC + beamformer is wrapped as one unit; every other stage is a
separate library that the integrator initialises (its own `mem_size` and `init`) and chains.

| Application | Analysis call | Stages written out | Synthesis |
|---|---|---|---|
| Mono AEC + NR/RES | `aec_process_context()` → `se_frame_from_aec()` | `se_sample_chain_form()`: `g_res_ctx`, `se_anr_stage_gain`, min, apply, `aec_cng_fill` | `se_sample_chain_synthesize()` |
| 4-ch AEC + BF + NR/RES | `audio_pipeline_4ch_process_front()` + `process_beam()` (or `process_with_activity()`) → `audio_pipeline_4ch_fuse()` | `se_sample_chain_form()`: `se_res_stage_gain`, `se_anr_stage_gain`, min, apply, `aec_cng_fill` | `se_sample_chain_synthesize()` |
| Mono / 4-ch Align-ULCNet | mono: `aec_process_context()`; 4-ch: the unit with `enable_post = 0`; then `ulcnet_nn_stage_begin_time()` / `_begin_complex()` | the network (`ulcnet_nn_stage_nn_begin` / `nn_end`) | `ulcnet_nn_stage_end()` |
| Mono / 4-ch DFN2 | mono: `aec_process_context()` → `se_frame_from_aec()`; 4-ch: `four_aec_front_process()` → `four_aec_fuse()` | `se_sample_chain_form()` (RES gain, min with the unity gain, apply; no NR, no synthesis), then the DFN2 stage through `dfn_hop_*` | `dfn_hop_end()` |

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
  (`se_sample_chain_reset()` also restarts the comfort-noise generator).

### Where do I call the inverse transform?

Once per hop, last: `se_sample_chain_synthesize()` (or `ac_istft_push()` from audiolib
directly) on the spectrum that has all gains applied and comfort noise added.
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
size_t aec_b   = aec_get_mem_size(&aec_cfg);                       /* aec_cfg from se_derive() */
SeSampleChainConfig sc = se_sample_chain_default_config(sample_rate, grid.fft_size);
sc.res = SE_SAMPLE_RES_FRAME_GAIN;                                 /* mono: the AEC's own G_res */
size_t chain_b = se_sample_chain_mem_size(&sc);                    /* RES/NR/CNG/synthesis, 0 = invalid */
uint8_t* pool  = board_alloc(ALIGN16(aec_b) + ALIGN16(chain_b) + my_stage_bytes, 16);
Aec* aec       = aec_init(pool, aec_b, &aec_cfg);
SeSampleChain* chain = se_sample_chain_init(pool + ALIGN16(aec_b), chain_b, &sc);
HfTilt* tilt   = hf_tilt_init(pool + ALIGN16(aec_b) + ALIGN16(chain_b), n_freqs, 8, 0.0f);  /* your stage, after it */
```

- A stage keeps no state of its own when its functions take the object the
  application carved (`se_anr_stage_gain` takes the `AnrDenoiser`, `se_res_stage_gain`
  the `AecRes`). Gain vectors the chain needs (`g_total`, the NR prior scratch, the
  spectrum buffer) are `n_freqs` floats / bins each, also from the caller's block.
- The 4-channel core (`FourAecNrRes`) owns only the shared aligner, the four
  lanes and the fuse buffers; the unit (`AudioPipeline4Ch`) adds SRP-PHAT, the BF and
  their scratch. The post-beam stages are separate blocks: `se_sample_chain_mem_size()` sums
  the `AecRes`, the denoiser and the synthesis for one configuration, and a
  neural stage (`ulcnet_nn_stage_mem_size()`, `dfn_rate_bridge_get_mem_requirements()`)
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
audio_pipeline_4ch_fuse(unit, &f);       /* project near, R^2, comfort noise with those weights (four_aec_fuse()) */
/* then the same chain: se_sample_chain_form(chain, &f), se_sample_chain_synthesize(...) */
```

A caller that runs its own beamformer uses the core instead:
`four_aec_front_process(core, ...)`, `bf_process_with_weights(...)`, `four_aec_fuse(core, bf_weights,
bf_spectrum, &f)`.

`four_aec_fuse()` needs a front frame processed since the last reset (otherwise
`FOUR_AEC_NR_RES_SEQUENCE_ERROR`), refuses an all-zero or non-finite weight set
and resets the core on a lane disagreement. The unit's `audio_pipeline_4ch_fuse()` returns
`SEQUENCE_ERROR` without a beam of the hop, `INVALID_ARGUMENT` when `core.enable_post = 0`, and
resets the whole unit (core, SRP, BF) on any other refusal; the caller resets its own chain too.
The front view lives until the next `process_front()` or `audio_pipeline_4ch_reset()`; there is
no token to return. The Align-ULCNet application never fuses: it takes the BF spectrum as
its error frame directly.

### What is in the frame?

| Field | Meaning | Mono | 4-ch |
|---|---|---|---|
| `e` | linear-AEC error E(f) | AEC error | beamformed error |
| `y`, `x` | matching capture / far spectra | AEC | projected near / lane 0 far |
| `r2`, `cn` | residual-echo and comfort-noise PSD (int16²) | AEC | projected |
| `g_res_ctx` | the AEC's own residual gain | `G_res` | NULL (the lanes compute none) |
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

## Build

```bash
# From Audio_ALG/pipelines/ — builds component libs and mono applications
make                # aec_nr_pipeline, static variant, mono_alignulcnet, mono_aec_dfn_res
make SIMD=0         # one switch: examples + AEC + NR + audiolib all scalar

# SIMD=0 and SIMD=1 outputs are not bit-identical end to end (see the note below).

# Binaries land in a config-keyed directory:
#   bin/<backend>-<config-hash>/  — resolve it with `make print-bin-dir`
# (same flags as your build), or use `make publish` for the stable
# dist/<backend>/current/ handoff path.
BIN="$(make -s print-bin-dir)"

# Run Version A (malloc)
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-preset aggressive

# Run Version B (static memory) — same CLI, plus a mem-size query mode
"$BIN"/aec_nr_pipeline_static mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline_static --print-mem-size --sample-rate 16000

# Mono chain test (pool carve vs heap, reset, switches, comfort noise, chain vs the
# hand-written oracle tests/conv_chain_mono.h; mono_aec_nr_res/tests/test_mono_chain.c)
# plus the example_board_adapter smoke test (see "Board Integration" below).
# Four-channel tests live in 4ch_aec_bf_nr_res/Makefile.
make test

# Build/test the four application packages.
make -C mono_aec_nr_res SIMD=0 test
make -C mono_alignulcnet SIMD=0 test
make -C 4ch_aec_bf_nr_res SIMD=0 test
make -C 4ch_alignulcnet SIMD=0 test
make -C 4ch_aec_bf_nr_res 4aec_nr_res_static
make -C 4ch_aec_bf_nr_res audio_pipeline_4ch_raw

# Build + run JUST the REFERENCE ONLY board-adapter example standalone
# (also runs as part of `make test` above):
make example-adapter

# Build the DFN2 stage archive with no stdio linked in at all (board images that
# forbid the stdio symbol set), and audit that it holds:
make NO_STDIO=1 libaudio_pipeline_dfn.a
make audit-no-stdio
```

> **SIMD=0 and SIMD=1 are not bit-identical end to end, by design.** Each shared
> kernel is byte-exact against its scalar twin (`audiolib/common/test/simd_selftest.c`),
> but the AEC matched filter (`delay_aec.c`) uses four accumulators with
> `vfmaq_f32` and a fused NLMS update. The contract is per configuration: a
> change must be byte-exact against the same configuration before it. Across
> configurations only finite output, the same delay winner/state and a quality
> tolerance hold; do not gate a release on cross-SIMD WAV equality.

`make` builds the mono executables and the DFN2 stage archive. There is no mono
library archive: the mono examples link `common/se_config.o` and the producer
libraries (`libaec.a`, `libanr.a`, `libaudio_common.a`) directly and include the
header-only chain. `4ch_aec_bf_nr_res/libaudio_pipeline_4ch.a` holds the shared
linear core (`4aec_nr_res.o`), the unit (`audio_pipeline_4ch.o`) and
`common/`'s `se_config.o`; `audit-no-stdio` covers it for the 4-channel build. The example
programs (`audio_pipeline_4ch_raw`, `audio_pipeline_4ch_static`, `4aec_nr_res_static`,
`4ch_aec_bf_dfn_res`, `4ch_alignulcnet`) link the archive and the example headers or objects
of `samples/`. `four_aec_nr_res_create()` is the heap convenience
path, while `four_aec_nr_res_get_mem_requirements()` +
`four_aec_nr_res_init_ex()` place the complete four-AEC/FFT/shared-aligner
state in one caller-owned 16-byte-aligned pool; the unit does the same with
`audio_pipeline_4ch_get_mem_requirements()` + `audio_pipeline_4ch_init_ex()`. Both construction
paths share the same allocation-free processing core and are byte-parity tested
at 16 and 48 kHz. See [`4ch_aec_bf_nr_res/README.md`](4ch_aec_bf_nr_res/README.md) and
`4aec_nr_res_static` for the directly comparable board sequence.

## Debugging & Performance Flags

`aec_nr_pipeline` and `aec_nr_pipeline_static` support the same debug CLI
options (mirrored, byte-for-byte identical wiring in both binaries): `--debug`
for the periodic AEC+NR status line, and `--timing` for a per-stage cost
summary printed once at exit.

`--timing` is a DISPLAY flag: it decides whether a breakdown is printed. The
gain-chain and synthesis rows are the program's own clock around those calls; the
AEC rows come from `aec_get_last_timing()`, whose measurement is a build-time
choice, off by default so a release binary takes none of the AEC's clock reads:

```sh
make PROFILE=1                 # lib/aec's stage timing (-DAEC_STAGE_TIMING=1)
"$(make -s print-bin-dir PROFILE=1)"/aec_nr_pipeline mic.wav ref.wav out.wav --timing
```

Run `--timing` against a build that did not measure and the AEC rows read 0;
the report says so in words rather than printing a table of zeros.

`CLOCK_MONOTONIC` is POSIX rather than C99. A target whose libc lacks it names
its own microsecond timer for lib/aec instead — `make PROFILE=1
EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'`, a plain
identifier because these Makefiles reject parentheses in `EXTRA_CFLAGS`. The 4-channel
core has its own override (`-DFOUR_AEC_NR_RES_NOW_US=...`), so a build names one timer per
component it actually builds. A substitute that returns a constant keeps the flag on
and reads 0, which `--timing` cannot distinguish from a build without it. The example
programs read their own clock in `timing_now_us()` (`mono_aec_nr_res/main.c`).

There are no other optional performance compile flags.

### `--debug`

Once per second of processed audio, prints one compact status line to stderr
combining `aec_debug_status()` (lib/aec) and `anr_debug_status()` (lib/nr) —
both are read-only snapshots of state the engines already maintain, so this adds
no per-frame cost when the flag is off and doesn't perturb the DSP output when on
(stdout/the output WAV are unaffected either way).

```
./aec_nr_pipeline mic.wav ref.wav out.wav --debug
[dbg   1.0s] aec: delay=288 conf=1.0 upd=68 erle=6.0dB lin=1 conv=0 near=1.48e-03 out=1.63e-05 | nr: init=1 gain=-13.3/-15.0dB spp=0.50 noise=-22.1dB
[dbg   2.0s] aec: delay=288 conf=1.0 upd=169 erle=9.4dB lin=1 conv=1 near=1.86e-03 out=2.98e-05 | nr: init=1 gain=-14.9/-17.2dB spp=0.51 noise=-27.5dB
...
```

`aec:` fields are `AecDebugStatus` (delay in samples, `-1` = not yet acquired;
`conf`/`upd` = delay-estimator confidence/update count; `erle` = windowed ERLE dB;
`lin`/`conv` = usable-linear-estimate / filter-converged gates; `near`/`out` = EMA
power). `nr:` fields are `AnrDebugStatus` (`init` = noise-floor initialized;
`gain` = mean/min linear gain dB; `spp` = mean speech-presence probability;
`noise` = mean noise-floor dB). With `--aec-only` the `nr:` half prints `n/a`
(no denoiser exists in that mode).

### Delay-estimator duty-cycling (opt-in)

`AecConfig.delay_est_period_s` (default `0` = analyse every hop) duty-cycles the
AEC3 matched-filter delay estimator: once the estimate is solid (confidence 1.0)
and unchanged for `delay_est_init_s` (default 0.3 s), analysis drops to 1 hop in
every K = round(`delay_est_period_s`/hop)/5 (minimum 1). Full-rate analysis
resumes when the estimate changes, loses solidity, or ERLE drops more than 6 dB
below its running peak. A decimated hop publishes no new estimate, so a delay
movement is seen at the next scheduled analysis, up to one duty period later.
The pipelines in this directory leave it at `0`.

## Two Versions

### Version A: malloc (`mono_aec_nr_res/main.c`)
Each module uses `_create()` / `_destroy()` and manages its own memory internally.
Suitable for desktop testing and Linux servers.

### Version B: static memory (`mono_aec_nr_res/static_main.c`)

Built by default alongside Version A. One caller-owned pool, no malloc after
init, byte-identical output to Version A (see Verification below). With NE10
vendored patch P0001 (`audiolib/common/lib/ne10/VENDORED.md`), the NE10 backend's three R2C/C2R
twiddle configs are carved FROM the caller pool too
(`ne10_fft_init_r2c_float32_ext`: carve + twiddle-generate directly over
caller-supplied memory, no `malloc()` involved) and are already counted in
the `*_get_mem_size` figures — so "no malloc after init" is zero heap
allocation ever, end to end, not just on the per-hop audio path.

## Board Integration

沒有 mono 函式庫 TU：板端自己用 `se_derive()` 導出 grid 與 `AecConfig`，查
`aec_get_mem_size()` 與 `se_sample_chain_mem_size()`，把兩塊切在自己的 pool 裡，
再逐個 `init`（`mono_aec_nr_res/static_main.c` 的 `pool_bytes()` / `rig_build()` 就是這段
的最小寫法；範例鏈是 header-only，可整份複製改寫）。唯一帶 descriptor 的單元是 4ch
的 `AudioPipeline4Ch`（`audio_pipeline_4ch_get_mem_requirements()` →
`audio_pipeline_4ch_init_ex()`），其餘函式庫（`Aec`、`AnrDenoiser`、`AecRes`、
`AcIstft`、`DfnResStage`、`DfnRateBridge`）各有自己的 `*_mem_size` 與 `init`，整合者自己
加總。

[`example_board_adapter.c`](mono_aec_nr_res/example_board_adapter.c) 是這個順序的可編譯、
可執行的 HOST 模擬（`make example-adapter`，也接進 `make test`）：pool 計畫（AEC 加
鏈的大小加總）、過小或未對齊的 pool 與重複 destroy 被拒絕、被拒的 init 不弄髒 pool、reset、destroy、pool 重用。
**它不取代板端整合與簽核**——裡面每個 `board_mem_*` 都只是主機陣列的替身，
真正的平台程式碼在標了 `// BOARD:` 的位置。

完整的韌體契約——pool 大小、query → init → process → destroy 的順序、descriptor
語意、`NO_STDIO=1`、交叉編譯／publish 流程與板上驗證清單——在
[`../docs/pipeline_board_integration.md`](../docs/pipeline_board_integration.md)。

## Tunable Parameters

### AEC (`AecConfig`, see `aec.h`)

**Presets**: `AEC_PRESET_MILD` / `AEC_PRESET_BALANCED`（default）/ `AEC_PRESET_AGGRESSIVE` / `AEC_PRESET_NO_FLOOR`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×52ms（sr≥44100 時 sr×64ms） | 自適應濾波器長度（416@8k, 832@16k／52ms, 3072@48k／64ms；`aec_config_defaults()` 的公式，不是固定 32ms） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES / preset strength axis**：RES has no standalone `ResConfig` — it lives inside `AecConfig`
and the AEC3 post-filter chain (`SuppressionGain`, `ResidualEchoEstimator`, etc.), surfaced
externally through the `AecResContext` seam (see `## Modules` above). The four AEC presets
differ in exactly three fields:

| Parameter | Mild | Balanced | Aggressive | No-floor | Description |
|-----------|------|----------|------------|----------|-------------|
| `min_gain_floor_far_active_db` | -20 | -28 | -38 | -28（inert） | AEC3 `SuppressionGain` 遠端活躍時的最低增益下限 dB（最大抑制量） |
| `min_gain_floor_dt_db` | -20 | -20 | -30 | -20（inert） | double-talk 期間套用的地板 dB；mild/balanced 停在 dataclass 共用預設 -20，只有 aggressive 額外下探到 -30（其餘欄位：filter length、Kalman Q、delay buffer、CNG…四個 preset 皆相同） |
| `min_gain_split_floor_enabled` | true | true | true | **false** | 地板機制本身的開關；`AEC_PRESET_NO_FLOOR` 把它關掉，此時上面兩個 dB 欄位不生效（AEC3 原生 min_gain 不被任何東西往上拉），抑制可以深到 AEC3 自己算出的極限 |

### Runtime strength control（不重建、不動 pool）

兩條強度軸都可以在**運轉中**改。呼叫時機一律是**兩個 hop 之間**、與 process 序列化；
**非 thread-safe**。全部回 `0` 或 `-1`，`-1` 時什麼都不寫。強度 setter 屬於各函式庫，
不屬於任何 pipeline：

| Entry point | 目標 |
|---|---|
| `aec_set_preset(aec, preset, ramp_ms)` | mono：`Aec` 的 far-active 地板（mono 的 RES 增益 `g_res_ctx` 由 lane 自己算） |
| `aec_res_set_preset(se_sample_chain_res(chain), preset, ramp_ms)` | 4ch：整合者自己的後波束 `AecRes`（`res = SE_SAMPLE_RES_AEC_RES`） |
| `aec_res_split_floor(res, &live, &target)` | 4ch：唯讀，線性功率。`live == target` 代表 ramp 已走完 |
| `anr_set_mode(se_sample_chain_nr(chain), mode)` | 範例鏈那**一個**降噪器；沒有降噪器的鏈（`se_sample_chain_nr()` 回 `NULL`）回 `-1` |

單元 `AudioPipeline4Ch` 與 4ch 核心 `FourAecNrRes` 沒有強度 setter：它們的 lane 只算線性
殘留，RES 與 NR 不在裡面。

`ramp_ms == 0` 代表下一個 hop 套用（**不是錯誤**），落點與「用該 preset 從頭建一個新
實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒（mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，互動式旋鈕應該給一個 ramp）。ramp 途中再呼叫會從當前
live 值重新起走。兩者都**不是重啟**：濾波器、延遲鎖定、噪聲底與增益平滑歷史全部繼續
跑；要重啟請用對應的 `_reset()`。

`AEC_PRESET_NO_FLOOR` 也可以在**已經在跑的** instance 上用同一個 setter 切換，不必重建：
`min_gain_split_floor_enabled` 這個 gate 會跟兩個 dB 欄位同一次呼叫一起 retarget（沒有
ramp，立即生效，跟 DT 地板一樣是硬切），所以不會有一個 hop 是「preset 標籤已經是
NO_FLOOR，但 gate 還沒跟上」的中間狀態。Python 對應：mono 的 `AEC.set_preset()`；4ch 的
`PostBeamResidualSuppressor.set_preset()`（抑制器由呼叫端持有，`FourChannelAecPipeline`
不再有 `set_aec_preset()`）。

> **4ch：對四條 lane 重新指定 preset 是無效操作。** 四條 lane 都以
> `spatial_linear_context` 建立，從不走到自己的殘留抑制器，它們的地板
> 什麼都不塑形。真正乘上輸出的 gain 來自**唯一一個**後波束 `AecRes`，所以強度要設在它身上
> （由 `4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` 與 `tests/test_se_common.c`
> 的 sample chain 區段釘住）。

> **NR：範例鏈的降噪器就是函式庫的 preset。** 它以 `anr_config_for_mode_grid()` 建立、
> 沒有任何覆寫，所以 `anr_set_mode()` 直接換強度。要改 preset 以外的欄位時，自己組好
> 完整 target 再呼叫 `anr_reconfigure()`（grid、`L`、`num_init_frames` 不可變）。

> **A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
> hop 生效：double-talk 期間套的是 DT 地板（mild/balanced 共用 -20 dB，aggressive 是
> -30 dB，見上表 `min_gain_floor_dt_db`）；far-active latch 觸發前套的是 far-silent 地板。同一個 `G_res` 還決定注入的 comfort
> noise 量（振幅正比於 `sqrt(1 - G_res^2)`，再乘該 bin 的 `G_nr`（下限 −10 dB），見兩處實作的 CNG
> 步驟——地板越深、CNG 反而越多）。所以**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且只量
> echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制。請在 echo 對齊或
> degradation 對齊的條件下比較，並實際試聽。

兩條 Align-ULCNet 範例（`mono_alignulcnet/`、`4ch_alignulcnet/`）**刻意不提供**
這些 setter：mono 變體沒有 NR 實例、且算出的 `G_res` 沒有任何消費者（輸出由 ULCNet
mask 塑形）；4ch 變體用的是 `core.enable_post = 0` 的單元，抑制器與降噪器
根本不存在。強度是模型的性質，不是 runtime 旋鈕。

### NR (`AnrConfig`, see `anr_types.h`)

**Modes**: `ANR_MODE_MILD` / `ANR_MODE_MODERATE` / `ANR_MODE_BALANCED`（default）/ `ANR_MODE_AGGRESSIVE`

> Both C pipeline CLIs' `parse_nr_mode()` — `mono_aec_nr_res/main.c` and
> `4ch_aec_bf_nr_res/4aec_nr_res_static.c` — accept
> `--nr-preset mild|moderate|balanced|aggressive`; anything else falls back to
> `ANR_MODE_BALANCED` with no error. The Python `aec_nr_pipeline.py` CLI accepts
> `mild|balanced|aggressive` only.

`g_min_db` is amplitude dB (`g_min = 10^(g_min_db/20)`).

> **Retime basis.** `anr_config_for_mode_grid()` retimes every `alpha_*`, `L`,
> `num_init_frames` and `scene_change_min_frames` from its authored reference
> duration to the actual hop of the grid (`anr_retime_alpha()`,
> `anr_retime_alpha_ref()`, `anr_retime_frames()`, `anr_retime_frames_ms()`);
> `q`, `g_min_db` and `xi_min_db` are plain constants. The reference is 16 ms/hop
> for `alpha_xi`, `alpha_attack`, `alpha_d_speech` and `L`, and 10 ms/hop for the
> other `alpha_*` and the frame counts. The tables list the authored constants,
> not the realized value at a grid; read that back from `anr_default_config()` /
> `anr_config_for_mode()` (or Python `core/nr_strength.py`) instead of scaling by hand.

The strength presets change only these four fields:

| Parameter | Mild | Moderate | Balanced | Aggressive | Description |
|-----------|------|----------|----------|------------|-------------|
| `g_min_db` | -20 | -23 | -25 | -28 | 最小增益 dB（最大抑制量，amplitude dB, /20） |
| `q` | 0.58 | 0.54 | 0.52 | 0.45 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -10 | -10 | -10 | -12 | 先驗 SNR 下限 dB |
| `noise_over_subtraction` | 1.2 | 1.3 | 1.4 | 1.3 | 噪聲過減係數 |

All modes share these tracker and gain-smoothing constants:

| Parameter | Authored value | Basis | Description |
|-----------|----------------|-------|-------------|
| `alpha_xi` | 0.92 | 16 ms | Decision Directed 先驗 SNR 平滑（SPP） |
| `alpha_attack` | 0.15 | 16 ms | Attack 平滑（語音起始追蹤） |
| `alpha_g` | 0.88 | 10 ms | 增益時間平滑（高→平滑） |
| `alpha_decay` | 0.88 | 10 ms | Decay 平滑（噪聲釋放） |
| `alpha_s` | 0.95 | 10 ms | 功率譜時間平滑（MCRA） |
| `alpha_d` | 0.903414 | 10 ms | 噪聲更新速率（= 0.85 @ 16 ms） |
| `L` | 32 frames | 16 ms | 最小值追蹤視窗；16 kHz 預設 256/128 grid（hop 8 ms）的實際值為 64 |
| `num_init_frames` | 20 frames | 10 ms | 初始化靜默幀數；同一 grid 的實際值為 25（約 200 ms） |
| `scene_change_threshold_db` | 10.0 | 純 dB 常數 | 場景轉換偵測閾值，不 retime |

> 範例鏈（`samples/se_sample_chain.h`）直接用上表的函式庫預設（`anr_config_for_mode_grid()`，
> `broadband_threshold = 1.0`），不另外覆寫。

---

## Troubleshooting & Tuning Guide

### AEC 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **殘留回聲明顯** | RES 抑制不足 | 改用更 aggressive preset，或直接覆寫 `min_gain_floor_far_active_db`（如 -28→-38） |
| **殘留回聲 + 遠端持續講話** | Filter 未完全收斂 | 增加 `filter_length`（如 832→1536），確認 mic-ref delay < filter_length |
| **近端語音被壓制（DT degradation）** | RES 過度抑制 | 改用 mild preset，或直接覆寫 `min_gain_floor_far_active_db`（如 -28→-20） |
| **輸出底噪不自然（突然靜音）** | CNG 未開啟 | 確認 `enable_cng=1`（preset 預設已開啟） |
| **收斂太慢** | Kalman Q 太保守 | 提高 `kalman_q_high`（如 1e-3→2e-3），減少 `warmup_frames`（如 100→50） |
| **Filter 發散（輸出爆音）** | Kalman Q 太激進或 echo path 劇變 | 降低 `kalman_q_high`（如 1e-3→5e-4） |
| **Echo path 變化後適應慢** | Shadow filter 太保守 | 提高 `shadow_mu_nlms`（如 0.5→0.7），降低 `shadow_err_alpha`（如 0.8→0.6） |

> `min_gain_floor_far_active_db` 與 `min_gain_floor_dt_db` 是在 mild/balanced/aggressive
> 三個 preset 間變動的兩個欄位（aggressive 的 DT 地板是 -30 dB，其餘兩個 preset 共用 -20 dB）；
> 沒有獨立的 `res_*` tunable struct（見上方 `AEC (AecConfig, see aec.h)`）。

### NR 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **噪聲殘留太多** | 抑制量不夠 | 降低 `g_min_db` 與 `q`（朝 aggressive preset 的 -28 dB／0.45） |
| **語音被吃掉** | 抑制太激進 | 提高 `g_min_db` 與 `q`（朝 mild preset 的 -20 dB／0.58），提高 `alpha_g`（如 0.88→0.92） |
| **Musical noise（隨機顆粒噪聲）** | 增益抖動 | 提高 `alpha_g`（增益更平滑），提高 `alpha_decay`（釋放更慢） |
| **語音起始被截斷** | Attack 太慢 | 降低 `alpha_attack`（預設 0.15；如 0.15→0.10），讓增益快速回升 |
| **噪聲環境切換後適應慢** | MCRA 追蹤窗太長 | 減小 `L`（如 32→16），但會增加噪聲底噪估計抖動 |
| **初始化期語音被壓** | 噪聲底噪估計偏高 | 減少 `num_init_frames`（如 20→10），但需確保前段有足夠噪聲 |
| **穩態噪聲殘留（風扇聲）** | 噪聲更新太慢 | 降低 `alpha_d`（預設 0.903；如 0.90→0.80），讓噪聲估計更快跟上 |
| **語音段噪聲估計上升** | SPP 平滑不足 | 提高 `alpha_xi`（預設 0.92；如 0.92→0.95），讓 SPP 更穩定判別語音 |

### Pipeline 整體

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **回聲消了但底噪變大** | NR 沒開或太保守 | 確認 NR mode 非 MILD，或降低 `g_min_db` |
| **NR 把回聲當噪聲學進去** | AEC 殘留回聲被 MCRA 當底噪 | 先確保 AEC 收斂良好，再調 NR。提高 `num_init_frames` 讓 MCRA 避開 AEC 收斂期 |
| **整體語音品質差（悶、失真）** | 多階段過度處理 | 改用 MILD preset（AEC + NR 都放鬆），只在必要時加強個別模組 |
| **處理 48kHz 音訊記憶體不足** | 模組記憶體隨 fft_size 增長 | 縮短 `filter_length`、減小 NR `L`（主要記憶體佔用） |

### Verification

Both versions build from the default `make` and produce byte-identical output
(`cmp` on the full rendered WAV) for the same arguments, and the static build's
init asserts the FFT grids agree across AEC/NR/OLA (`n_freqs` cross-check at
init).
