# C Pipeline: Linear AEC → NR → RES

## Architecture

```
mic ─┐                       ┌─ aec_out ──┐              ┌─ nr_out ──┐                  ┌─ output
     ├→ AEC (linear) ────────┤            ├→ NR (MMSE) ──┤           ├→ RES (post) ─────┤
ref ─┘   PBFDKF+Shadow      └─ context   ┘  LSA+MCRA    └─ gain[]   ┘  echo×nr_gain    └─ final
```

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libmmse_lsa.a | mmse_lsa_denoiser.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a (included) | aec.h (`AecResContext`) | Residual echo suppression, folded into AEC's freq-domain seam |

RES is not a standalone module/library — it is exposed as the `AecResContext` seam on
the AEC object. With `AecConfig.return_res_context=1` and `enable_res=0`, `aec_process()`
(or the streaming `aec_analyze_render()` / `aec_process_capture()` pair) still computes the
AEC3 post-filter's residual-echo suppression internals but does not apply them to the time
output; `aec_get_res_context(a, &ctx)` then exposes `AecResContext` — `echo_spec`, `error_spec`,
`res_gain` (G_res(f)), `r2` (residual-echo PSD), `comfort_noise`, etc. — so an external caller
can run AEC(linear) → NR → RES itself. See `lib/aec/c_impl/include/aec.h` (`AecResContext`,
`aec_get_res_context()`) for the full field list.

## Parameter Alignment

All modules use unified 20ms frame / 10ms hop, auto-configured by sample rate:

| Parameter | 8 kHz | 16 kHz | 48 kHz | Formula |
|-----------|-------|--------|--------|---------|
| frame_size | 160 | 320 | 960 | sr × 20ms |
| hop_size | 80 | 160 | 480 | frame / 2 |
| fft_size | 256 | 512 | 1024 | next pow2 ≥ frame |
| n_freqs | 129 | 257 | 513 | fft/2 + 1 |
| filter_length | 416 | 832 | 3072 | ms-derived: sr × 52ms (64ms ≥44.1 kHz) |
| n_partitions | 6 | 6 | 7 | ceil(filter_length / hop) |

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **Algorithmic latency** | 10 ms | 1 hop（所有 sample rate 一致） |
| **NR OLA delay** | +10 ms | NR frame 處理引入額外 1 hop 延遲 |
| **Pipeline total latency** | **20 ms** | AEC hop + NR OLA delay |
| **Processing (per hop)** | < 0.5 ms | AEC + NR + RES 合計（ARM Cortex-A53 @ 1GHz 估計） |
| **RTF** | < 0.05 | 遠低於即時要求 |

### Memory Budget

Measured figures from `./aec_nr_pipeline_static --print-mem-size --sample-rate 16000`
(balanced presets). The AEC row is the composite `aec_get_mem_size()` pool — it
already contains HPF, PBFDKF ×2 (main+shadow), delay estimator, the RES/post
context and the AEC-internal FFTs. Since NE10 vendored patch P0001 the NE10
twiddle configs are carved from these pools too, so both columns are the
complete memory requirement (strict init→destroy zero-heap on both backends):

| Rate / Backend | AEC | FFT (OLA) | NR | Pipeline bufs | **Total** |
|--------|-----|-----------|-----|---------------|-----------|
| **8 kHz KISS** | 290,352 B | 8,784 B | 97,792 B | 7,264 B | **404,192 B (394.7 KB)** |
| **8 kHz NE10** | 288,528 B | 8,176 B | 97,792 B | 7,264 B | **401,760 B (392.3 KB)** |
| **16 kHz KISS** | 537,680 B | 16,976 B | 194,048 B | 14,432 B | **763,136 B (745.2 KB)** |
| **16 kHz NE10** | 533,552 B | 15,600 B | 194,048 B | 14,432 B | **757,632 B (739.9 KB)** |
| **48 kHz KISS** | 1,251,760 B | 33,360 B | 386,560 B | 33,888 B | **1,705,568 B (1,665.6 KB)** |
| **48 kHz NE10** | 1,243,024 B | 30,448 B | 386,560 B | 33,888 B | **1,693,920 B (1,654.2 KB)** |

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072 taps、
> 7 partitions），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮 `filter_length`
> 與 NR 的 `L`（48 kHz 也可用 `n_partitions` override 換較短尾巴）。
> 三個 rate 都由同一 hop=10 ms 規則自動推導（`pipeline_dims.h`），並在 init 以
> grid assert 驗證 pipeline/AEC/FFT/NR 四方一致。

## Integration Flow

1. **AEC (linear)**: Set `enable_res=0` and `return_res_context=1`, call `aec_process()` (or the
   streaming `aec_analyze_render()` / `aec_process_capture()` pair), then `aec_get_res_context(a, &ctx)`
   to read the `AecResContext` seam
2. **NR**: `mmse_lsa_process()` for denoising, `mmse_lsa_get_gain()` for per-bin gain
3. **RES**: Correct echo PSD with `echo_spec *= nr_gain`, then apply `ctx.res_gain` (AEC3
   `SuppressionGain` G_res(f)) to the NR output — there is no separate `res_process()` call

### Echo PSD Correction

```c
const float* gain = mmse_lsa_get_gain(nr, NULL);
for (int k = 0; k < n_freqs; k++) {
    corrected_echo[k].r = ctx.echo_spec[k].r * gain[k];
    corrected_echo[k].i = ctx.echo_spec[k].i * gain[k];
}
/* apply ctx.res_gain[k] (G_res(f)) to nr_out[k] to get the final RES-suppressed output */
```

NR already attenuated certain frequency bins. The echo PSD estimate must
reflect this, otherwise RES will over-suppress (seeing echo that NR already
removed). Multiplying by the NR gain corrects for this.

## NR OLA Delay

NR uses OLA (frame_size=320, hop=160), introducing 1-frame (10ms) delay.
The pipeline saves the previous AEC context and uses it when the
corresponding NR output becomes available.

## Build

```bash
# From Audio_ALG/pipelines/ — builds the submodule libs + BOTH binaries
make                # libs (BACKEND=kiss) + aec_nr_pipeline + aec_nr_pipeline_static

# Run Version A (malloc)
./aec_nr_pipeline mic.wav ref.wav output.wav balanced
./aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
./aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-preset aggressive

# Run Version B (static memory) — same CLI, plus a mem-size query mode
./aec_nr_pipeline_static mic.wav ref.wav output.wav balanced
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000

# Run the audio_pipeline.h library's own acceptance tests (F20/R08/§7.3) —
# create-vs-init byte equality (incl. a poisoned pool), destroy idempotence,
# misaligned/undersized pool rejection, sample-rate whitelist rejection,
# AudioPipelineConfig reject-first validation (bad enum/bool fields) — each
# run once per supported rate (8000/16000/48000; 48 kHz uses a reduced hop
# count, see test_audio_pipeline.c)
make test
```

`make` also builds `libaudio_pipeline.a` (the linkable pool-sizing/carving/
processing library both CLIs above are now thin shells over) as a side
effect of building either binary. See "Board Integration" below for the API
this exposes to a firmware/board consumer.

## Debugging & Performance Flags

Both `aec_nr_pipeline` and `aec_nr_pipeline_static` support the same debug CLI
option (mirrored, byte-for-byte identical wiring in both binaries). There are
no optional performance compile flags — the fast matched-filter arithmetic and
delay-estimator duty-cycling are built into `lib/aec` unconditionally.

### `--debug`

Once per second of processed audio, prints one compact status line to stderr
combining `aec_debug_status()` (lib/aec) and `mmse_lsa_debug_status()` (lib/nr) —
both are read-only snapshots of state the engines already maintain, so this adds
no per-frame cost when the flag is off and doesn't perturb the DSP output when on
(stdout/the output WAV are unaffected either way).

```
./aec_nr_pipeline mic.wav ref.wav out.wav --debug
[dbg   1.0s] aec: delay=-1 conf=0.5 upd=6 erle=0.0dB lin=0 conv=0 near=8.74e-04 out=8.08e-04 | nr: init=1 gain=-18.2/-23.4dB spp=0.50 noise=-1.2dB
[dbg   2.0s] aec: delay=320 conf=1.0 upd=18 erle=0.0dB lin=1 conv=0 near=4.79e-03 out=3.47e-03 | nr: init=1 gain=-18.8/-23.9dB spp=0.51 noise=-4.8dB
...
```

`aec:` fields are `AecDebugStatus` (delay in samples, `-1` = not yet acquired;
`conf`/`upd` = delay-estimator confidence/update count; `erle` = windowed ERLE dB;
`lin`/`conv` = usable-linear-estimate / filter-converged gates; `near`/`out` = EMA
power). `nr:` fields are `MmseLsaDebugStatus` (`init` = noise-floor initialized;
`gain` = mean/min linear gain dB; `spp` = mean speech-presence probability;
`noise` = mean noise-floor dB). With `--aec-only` the `nr:` half prints `n/a`
(no denoiser exists in that mode).

> **Caveat**: this pipeline always runs AEC in **linear mode**
> (`enable_res=0`, `return_res_context=1` — the external NR/RES seam), and
> `lib/aec/c_impl/src/aec.c` only caches `last_erle_windowed` when
> `cfg.enable_res` is true. So `erle=` in this pipeline's `--debug` output
> always reads `0.0dB` — that's expected here, not a broken query. (The field
> does move if you drive `aec_debug_status()` from a caller running with
> `enable_res=1`, e.g. `lib/aec/c_impl/example/aec_wav.c --debug`.)

### Delay-estimator duty-cycling (built in, always on)

The AEC3 matched-filter delay estimator duty-cycles itself — no flag or
config field: once the delay estimate is solid (confidence 1.0) and unchanged
for `delay_est_period_s` (default 0.5s), analysis drops to 1 hop in every K
(K=10 by default) instead of every hop — full-rate analysis resumes
immediately if the estimate changes, loses solidity, or ERLE drops >6dB off
its running peak. **Sampled-quality-verified ~zero cost** (60-case AECMOS:
≤+0.014 / worst −0.006). On a stable-delay clip the decimated schedule never
actually skips a *different* outcome; verified here on
`wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_*`.

Note: in THIS pipeline the ERLE-watchdog resume leg is inert — the AEC runs
linear-mode (`enable_res=0`) so `last_erle_windowed` is never updated (same
root cause as the `--debug` `erle=0.0` caveat above). Full-rate analysis
still resumes on estimate change or lost solidity, which are the primary
resume paths.

### Compile flags (`EXTRA_CFLAGS`)

`pipelines/Makefile` passes `EXTRA_CFLAGS` through to the `lib/aec` and
`lib/nr` sub-builds *and* this pipeline's own compile, so one invocation
reaches every `.o`. The fast matched-filter arithmetic and delay-estimator
duty-cycling are built in unconditionally — there are no optional
performance flags at present.

`make clean-libs` first is required when switching `EXTRA_CFLAGS` — object
files aren't flag-tagged, so a stale non-flagged (or stale flagged) `.o` will
otherwise silently persist across the flag change.


## Two Versions

### Version A: malloc (`aec_nr_pipeline.c`)
Each module uses `_create()` / `_destroy()` and manages its own memory internally.
Suitable for desktop testing and Linux servers.

### Version B: static memory (`aec_nr_pipeline_static.c`)

Built by default alongside Version A (both `lib/aec` and `lib/nr` track
`main` — each library ships the heap and static APIs side by side,
selected at runtime). One
caller-owned pool, no malloc after init, byte-identical output to Version A
(see Verification below). "No malloc after init" describes the per-hop audio
path, not zero heap allocation ever: on the NE10 backend, `aec_init`/`fft_init`
each trigger a one-time backend-internal twiddle-config allocation *during*
init itself (outside the caller pool, not counted in the `*_get_mem_size`
figures) — see the `destroy()` note in the code block below for how that
memory is reclaimed. The KISS backend has no such exception; it is zero-heap
end to end.

The pipeline uses exactly THREE composite static APIs — there are no
per-submodule `_get_mem_size()` entry points; each library slices its own
internals (HPF, PBFDKF ×2, delay estimator, RES/post context, internal FFTs
for AEC; MCRA + SPP for NR) inside its single pool segment:

```c
size_t aec_sz = aec_get_mem_size(&aec_cfg);          /* lib/aec           */
size_t nr_sz  = mmse_lsa_get_mem_size(&nr_cfg);      /* lib/nr            */
size_t fft_sz = fft_get_mem_size(fft_size);          /* audio_common (OLA) */
/* + pipeline buffers; every segment 16-byte aligned (ALIGN16)            */

Aec*             aec = aec_init(mem_aec, aec_sz, &aec_cfg);
MmseLsaDenoiser* nr  = mmse_lsa_init(mem_nr, nr_sz, &nr_cfg);
FftHandle*       fft = fft_init(mem_fft, fft_sz, fft_size);
/* destroy() on static instances frees no pool memory (runtime is_static); it
 * still releases backend-owned handles (e.g. NE10 twiddle configs) — on
 * NE10 each of aec_destroy/mmse_lsa_destroy/fft_destroy above must be
 * called exactly once, before its pool segment is freed or reused: skipping
 * it leaks the twiddle config, calling it twice double-frees it. KISS is a
 * genuine no-op here (nothing to release), safe to call any number of times. */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the reported total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. The pool base MUST be 16-byte
aligned (both libraries assert this).

## Board Integration

Review F20: the pool-sizing/carving/per-hop-processing logic both CLIs above
embed is also available as a standalone, linkable library —
[`audio_pipeline.h`](audio_pipeline.h) / [`audio_pipeline.c`](audio_pipeline.c),
built into `libaudio_pipeline.a`. A board's own memory manager consumes this
directly instead of copying `aec_nr_pipeline_static.c`'s file-local carve
code into firmware. Both CLIs are now thin shells over it (arg parsing + WAV
I/O + the `--print-mem-size`/`--debug`/`DUMP_CTX` diagnostics) — see
`aec_nr_pipeline_static.c` for the caller-pool flavor of the sequence below,
or `aec_nr_pipeline.c` for the heap-convenience flavor
(`audio_pipeline_create()`).

### Sequence

```
1. query    AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
            cfg.aec_preset/nr_mode/aec_only/enable_cng/legacy_amin = ...;
            AudioPipelineMemReq req;
            audio_pipeline_get_mem_requirements(&cfg, &req);   // -> req.bytes/alignment/...

2. allocate void* pool = platform_alloc(req.bytes, req.alignment);
            // req.alignment is always 16 today; pool need NOT be zeroed —
            // see "Dirty-pool contract" below. Pool must stay STABLE and
            // EXCLUSIVE (nothing else reads/writes it, not shared with any
            // other instance) for the entire lifetime of the handle below.

3. init     AudioPipeline* p = audio_pipeline_init(pool, req.bytes, &cfg);
            // NULL on misaligned/undersized pool, invalid cfg, or a
            // sub-module init/grid-agreement failure (stderr has detail).

4. process  float mic[hop], ref[hop], out[hop];   // hop = audio_pipeline_hop_size(p)
            while (have_audio()) {
                read_hop(mic, ref, hop);
                audio_pipeline_process(p, mic, ref, out);
                write_hop(out, hop);
            }

5. reset?   audio_pipeline_reset(p);   // optional: echo-path change, stream switch
            // re-zeros pipeline/AEC/NR state in place; no re-validation, no
            // pool re-touch beyond that.

6. destroy  audio_pipeline_destroy(p);
            // NR -> pipeline FFT -> AEC, reverse of the init carve order.
            // NULL-safe; idempotent for THIS pool-resident instance (every
            // sub-destroy is already a genuine no-op on the pool path — see
            // "Two Versions" above). Call it exactly once if `p` came from
            // audio_pipeline_create() instead (ordinary free() semantics).

7. release  platform_free(pool);   // only after step 6 — the pool is dead once
            //                        audio_pipeline_init/destroy have run on it.
```

### Descriptor semantics (`AudioPipelineMemReq`)

| Field | Meaning |
|-------|---------|
| `bytes` | Total pool size to allocate (includes the opaque `AudioPipeline` control block itself, carved at the front — a few hundred bytes — plus AEC + FFT(OLA) + NR + the 13 pipeline scratch buffers, same carve `aec_nr_pipeline_static.c`'s old file-local `pipeline_pool_size()` produced). |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, both FFT backends, `mem_align.h`'s `ALIGN16` — carves to). |
| `layout_version` | Bumped whenever `audio_pipeline.c`'s OWN carve order/buffer set/sizing formula changes — i.e. whenever a `bytes` figure computed by an older build would misdescribe a newer build's actual carve, or vice versa. Starts at 1. Does **not** need bumping for a change purely inside AEC's/NR's/an FFT backend's own internal `_get_mem_size` layout (each is consumed as one opaque composite blob here, same as the pre-F20 static CLI already treated them — a stale cached `bytes` from an old submodule build is still caught by the undersized-pool rejection at init). |
| `backend` | Compile-time FFT backend identity this `audio_pipeline.o` was built with — `"kiss"` or `"ne10"` (matches this Makefile's `BACKEND=`). The two backends are not byte-identical to each other (pre-existing, expected — see `lib/aec/CLAUDE.md`); a descriptor from one is never valid for the other even at matching `bytes`. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the pipeline's own carve STRUCTURE: the backend identity above, a literal token list naming the 13 scratch buffers in carve order, and the alignment granularity — see `audio_pipeline_build_flags_hash()` in `audio_pipeline.c`. **Covers:** a change to this file's own carve order/buffer set/alignment. **Does NOT cover:** `AudioPipelineConfig` preset/tunable VALUES (`aec_preset`, `nr_mode`, `sample_rate`, `aec_only`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; AEC's/NR's/an FFT backend's internal struct layouts (opaque blobs, as above); the compiler/ABI/toolchain. |

A board integrator who caches a descriptor across a library upgrade should
compare `layout_version` + `build_flags_hash` (and `backend`) before trusting
an old `bytes` figure; if either changed, re-query
`audio_pipeline_get_mem_requirements()` rather than reusing the stale value.

### Dirty-pool contract

`audio_pipeline_init()` does **not** require a zero-filled pool. Every
pipeline-owned scratch buffer (the OLA accumulator, per-bin gain/spectrum
scratch, the mic/ref/output hop copies) is explicitly zeroed at carve time,
and AEC/NR/the FFT backend each zero their own sub-region during their own
`_init()` — so a pool filled with poison bytes inits and processes
identically to a freshly-zeroed one. `test_audio_pipeline.c`'s
create-vs-init parity case exercises exactly this: a `memset(pool, 0xA5,
bytes)`-poisoned pool run through `audio_pipeline_init()` produces
byte-for-byte the same 1000-hop output as `audio_pipeline_create()`'s
(unpoisoned) heap path. There is no need for a caller-side blanket
`memset(pool, 0, bytes)` before `audio_pipeline_init()` — it was only ever a
defensive habit carried over from the pre-F20 static CLI, not a requirement.

### `USE_EXT_MEM` — not a thing here

Both the heap path (`audio_pipeline_create`/`audio_pipeline_destroy`) and the
pool path (`audio_pipeline_get_mem_requirements`/`audio_pipeline_init`/
`audio_pipeline_destroy`) are always compiled into `libaudio_pipeline.a` —
which one you use is selected purely by which entry point you call, at
runtime, same as `lib/aec`'s and `lib/nr`'s own `_create` vs. `_get_mem_size`/
`_init` pairs. There is no `-DUSE_EXT_MEM`-style compile-time switch (that
pattern existed historically in `lib/nr` and was removed — see
`lib/nr/c_impl/CHANGELOG.md` `[v1.11.0]`/later entries); do not look for one,
and do not add one.

### Teardown order

`audio_pipeline_destroy()` tears down NR → pipeline FFT (the OLA irfft
instance) → AEC — the reverse of `audio_pipeline_init()`'s carve order (AEC →
FFT → NR → scratch). Every one of those three calls is a genuine no-op for a
pool-resident instance today (matches the "Two Versions" section above); the
order is kept anyway as forward-compat insurance — a future backend/module
MAY hold something outside the pool that a destroy call needs to release
(see the NE10-twiddle-config caveat earlier in this file), and it is exactly
what the heap-convenience path needs for real (`free()` on the pool
`audio_pipeline_create()` allocated).

## Tunable Parameters

### AEC (`AecConfig`, see `aec.h`)

**Presets**: `AEC_PRESET_GENTLE` / `AEC_PRESET_BALANCED`（default）/ `AEC_PRESET_AGGRESSIVE`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×32ms | 自適應濾波器長度（256@8k, 512@16k, 1536@48k） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES / preset strength axis**：RES has no standalone `ResConfig` — it lives inside `AecConfig`
and the AEC3 post-filter chain (`SuppressionGain`, `ResidualEchoEstimator`, etc.), surfaced
externally through the `AecResContext` seam (see `## Modules` above). The three AEC presets
differ in exactly one field:

| Parameter | Gentle | Balanced | Aggressive | Description |
|-----------|--------|----------|------------|-------------|
| `min_gain_floor_far_active_db` | -20 | -28 | -38 | AEC3 `SuppressionGain` 遠端活躍時的最低增益下限 dB（最大抑制量）；其餘欄位（filter length、Kalman Q、delay buffer、CNG…）三個 preset 皆相同 |

### NR (`MmseLsaConfig`, see `mmse_lsa_types.h`)

**Modes**: `MMSE_LSA_NR_MILD` / `MMSE_LSA_NR_MODERATE` / `MMSE_LSA_NR_BALANCED`（default）/ `MMSE_LSA_NR_AGGRESSIVE`

> These are the library's mode enum, not this pipeline's CLI surface. `aec_nr_pipeline.c`'s
> `parse_nr_mode()` only recognizes `"mild"` / `"aggressive"` (anything else, including
> `"moderate"`, silently falls back to `MMSE_LSA_NR_BALANCED` — no error); the Python
> `aec_nr_pipeline.py` CLI likewise restricts `--nr-preset` to `choices=['mild', 'balanced',
> 'aggressive']`. `MODERATE` is only reachable by calling `mmse_lsa_config_for_mode()` directly.

`g_min_db` is in the amplitude-dB convention (`/20`, i.e. `g_min = 10^(g_min_db/20)`), not the
older power-dB (`/10`) convention:

| Parameter | Mild | Moderate | Balanced | Aggressive | Description |
|-----------|------|----------|----------|------------|-------------|
| `g_min_db` | -20 | -25 | -30 | -40 | 最小增益 dB（最大抑制量，amplitude dB, /20） |
| `q` | 0.60 | 0.55 | 0.50 | 0.35 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -15 | -18 | -20 | -25 | 先驗 SNR 下限 dB |
| `alpha_d` | 0.85 | 0.85 | 0.70 | 0.50 | 噪聲追蹤 IIR 係數 |
| `alpha_g` | 0.92 | 0.92 | 0.88 | 0.85 | 增益時間平滑（高→平滑） |
| `alpha_attack` | 0.40 | 0.40 | 0.30 | 0.15 | Attack 平滑（語音起始追蹤） |
| `alpha_decay` | 0.92 | 0.92 | 0.88 | 0.88 | Decay 平滑（噪聲釋放） |

**MCRA 噪聲估計**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_s` | 0.95 | 功率譜時間平滑 |
| `alpha_d` | 0.70 | 噪聲更新速率 |
| `L` | 32 | 最小值追蹤視窗（幀數，×10ms = 320ms） |
| `num_init_frames` | 20 | 初始化靜默幀數（200ms） |
| `scene_change_threshold_db` | 10.0 | 場景轉換偵測閾值 |

**SPP**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_xi` | 0.88 | Decision Directed 先驗 SNR 平滑 |

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

> `min_gain_floor_far_active_db` 是唯一在 mild/balanced/aggressive 三個 preset 間變動的欄位；
> 沒有獨立的 `res_*` tunable struct（見上方 `AEC (AecConfig, see aec.h)`）。

### NR 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **噪聲殘留太多** | 抑制量不夠 | 降低 `g_min_db`（如 -30→-40），降低 `q`（如 0.5→0.35） |
| **語音被吃掉** | 抑制太激進 | 提高 `g_min_db`（如 -30→-20），提高 `q`（如 0.5→0.6），提高 `alpha_g`（如 0.88→0.92） |
| **Musical noise（隨機顆粒噪聲）** | 增益抖動 | 提高 `alpha_g`（增益更平滑），提高 `alpha_decay`（釋放更慢） |
| **語音起始被截斷** | Attack 太慢 | 降低 `alpha_attack`（如 0.3→0.15），讓增益快速回升 |
| **噪聲環境切換後適應慢** | MCRA 追蹤窗太長 | 減小 `L`（如 32→16），但會增加噪聲底噪估計抖動 |
| **初始化期語音被壓** | 噪聲底噪估計偏高 | 減少 `num_init_frames`（如 20→10），但需確保前段有足夠噪聲 |
| **穩態噪聲殘留（風扇聲）** | 噪聲更新太慢 | 降低 `alpha_d`（如 0.7→0.5），讓噪聲估計更快跟上 |
| **語音段噪聲估計上升** | SPP 平滑不足 | 提高 `alpha_xi`（如 0.88→0.95），讓 SPP 更穩定判別語音 |

### Pipeline 整體

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **回聲消了但底噪變大** | NR 沒開或太保守 | 確認 NR mode 非 MILD，或降低 `g_min_db` |
| **NR 把回聲當噪聲學進去** | AEC 殘留回聲被 MCRA 當底噪 | 先確保 AEC 收斂良好，再調 NR。提高 `num_init_frames` 讓 MCRA 避開 AEC 收斂期 |
| **整體語音品質差（悶、失真）** | 多階段過度處理 | 改用 MILD preset（AEC + NR 都放鬆），只在必要時加強個別模組 |
| **處理 48kHz 音訊記憶體不足** | 模組記憶體隨 fft_size 增長 | 縮短 `filter_length`、減小 NR `L`（主要記憶體佔用） |

### Verification

Both versions build from the default `make` and have been verified
**byte-identical** to each other (`cmp` on the full rendered WAV) at 16 kHz on
real doubletalk material (`aec_challenge_blind` case `0I0XMl3M`, balanced
presets, CNG on), and the static build's init asserts the 8 kHz / 16 kHz FFT
grids agree across AEC/NR/OLA (`n_freqs` cross-check at init).
