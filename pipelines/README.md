# C Pipeline: linear AEC + echo-aware NR/RES gain fusion

## Architecture

```
mic/ref → linear AEC (aec_process_context(): context-only, no emit)
              ├─ E(f) ────────────────→ echo-aware NR ─→ G_nr ─┐
              └─ AecResContext {R², G_res, CNG, far power} ─────┤
                                                                ↓
                                  G_total = min(G_nr, G_res)
                                  + near-end floor + CNG
                                                                ↓
                                                       one iFFT/OLA

aec_only mode (no NR/RES stage at all):
mic/ref → linear AEC (aec_process(): emits into out) → out
```

There is no independent time-domain RES filter after NR. `G_res` is calculated
inside the AEC3 suppression path, exported with the linear residual, fused with
`G_nr`, and applied once to `E(f)`.

The non-`aec_only` path never reads `aec_process()`'s own returned audio —
only `AecResContext`'s linear residual matters downstream — so it drives AEC
via `aec_process_context()` instead: same linear filter and AEC3
post/RES-context computation, minus the copy into an `out` buffer neither this
pipeline nor the caller ever reads on that path. This pipeline still picks one
entry point at init from `aec_only`, but that is now only a cost choice: the
two entry points advance identical state and may be mixed freely on one
instance.

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libmmse_lsa.a | mmse_lsa_denoiser.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a (included) | aec.h (`AecResContext`) | Residual echo suppression, folded into AEC's freq-domain seam |
| Mono integration | libaudio_pipeline.a | mono_aec_nr_res/audio_pipeline.h | One-mic AEC + NR/RES, heap or caller-owned pool |
| 4-ch linear core | libaudio_pipeline_4ch.a | 4ch_aec_bf_nr_res/4aec_nr_res.h | One shared aligner + four linear AECs; the archive contains only this core |
| Complete 4-ch spatial | application object | 4ch_aec_bf_nr_res/audio_pipeline_4ch.h | Core + reusable SRP-PHAT/GSC/NR libraries; linked into the application, not another archive |
| Mono/4-ch Align-ULCNet | application objects | mono_alignulcnet/audio_pipeline_ulcnet.h and 4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h | Existing component libraries + C pre/post + external accelerator callback |

RES is not a standalone module/library — it is exposed as the `AecResContext` seam on
the AEC object. With `AecConfig.return_res_context=1` and `enable_res=0`, `aec_process_context()`
(or the streaming `aec_analyze_render()` / `aec_process_capture()` pair, if the offline/streaming
split matters more than shedding the emit copy) still computes the
AEC3 post-filter's residual-echo suppression internals but does not apply them to the time
output; `aec_get_res_context(a, &ctx)` then exposes `AecResContext` — the
reconstructing 50%-overlap WOLA `error_spec`, its matching `echo_spec` /
`near_spec`, `formed_hop`, `res_gain` (G_res(f)), `r2` (residual-echo PSD),
`comfort_noise`, etc. — so an external caller
can run AEC(linear) → NR → RES itself. `aec_process_context()` is this mono
pipeline's own choice for exactly this pattern (this file's Architecture
diagram above) — it never emits an `out` buffer, pure waste when nothing
downstream reads it. See `lib/aec/c_impl/include/aec.h` (`AecResContext`,
`aec_get_res_context()`, `aec_process_context()`) for the full field list.

**`error_spec` does not always mean "error."** That configuration —
`enable_res=0` with `return_res_context=1` — is the one mode where `lib/aec`'s
formed selector opens a third candidate: on a hop the filtering-quality
analyzer has not yet cleared, whose selected residual carries more energy than
the mic capture, `formed_hop` and `error_spec` both become the capture itself,
faded in over the existing 30-sample transition. NR/RES still run on it as
usual — this is the steady-state fallback for a filter that is adding signal
rather than cancelling. `aec_process()`'s own emitted audio is unaffected.

The four-channel API is a separate zero-padding-free grid and does not use
`AudioPipeline`. See [the 4-channel contract](4ch_aec_bf_nr_res/README.md) and
[`4ch_aec_bf_nr_res/4aec_nr_res.h`](4ch_aec_bf_nr_res/4aec_nr_res.h) for its synchronous
pre/post boundary.

The directory root contains only the aggregate build, documentation,
requirements and the public Python reference module. C sources live with
their applications. Shared ULCNet model-I/O support is owned by
`../AIAEC/Align_ULCNet/`; offline scoring and benchmark drivers remain under
`tools/`.

## Application layout

The four product flows are applications that assemble existing components;
they are not four additional libraries:

| Application | Entry directory | Composition |
|---|---|---|
| Mono AEC + NR/RES | `mono_aec_nr_res/` | `audio_pipeline` + AEC + NR + audio_common |
| 4-ch AEC + BF + NR/RES | `4ch_aec_bf_nr_res/` | 4-lane core + DOA/GSC + NR + audio_common |
| Mono Align-ULCNet | `mono_alignulcnet/` | mono ULCNet wrapper + AEC + AIAEC pre/post |
| 4-ch Align-ULCNet | `4ch_alignulcnet/` | pre-only 4-lane core + DOA/GSC + AIAEC pre/post |

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

(Verified against a live `aec_create()` at each grid, not hand-derived.)

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **AEC linear path** | 0 samples | PBFDKF overlap-save 的當前 hop 線性誤差可立即取得 |
| **Synthesis WOLA delay** | 1 hop | NR/RES 頻譜由 sqrt-Hann WOLA 重建；這就是 full-chain 的演算法延遲 |
| **Pipeline total latency** | 1 hop — 16 ms @ 8 kHz, 8 ms @ 16 kHz default, 16 ms @ 16 kHz alt, ~10.7 ms @ 48 kHz | 不再額外保存上一個 AEC context；AEC-only 為 0 samples |
| **Processing (per hop)** | < 0.5 ms | AEC + NR + RES 合計(ARM Cortex-A53 @ 1GHz 估計) |
| **RTF** | < 0.05 | 遠低於即時要求 |

上表是 conventional mono pipeline 的數字。兩條 Align-ULCNet application 的
加總額外延遲同樣是 **1 hop**（唯一來源是 ULCNet synthesis WOLA）：mono 一直
如此，4ch 變體自 2026-09-03 起由 2 hop 降為 1 hop——GSC 的 beamformed error
頻譜直接當成模型的分析幀，中間的 beam WOLA→再 analysis 來回與配套的 one-hop
far 補償都已移除。細節見
[`4ch_alignulcnet/README.md`](4ch_alignulcnet/README.md)。

### Memory Budget

Measured figures from `"$(make -s print-bin-dir)"/aec_nr_pipeline_static --print-mem-size --sample-rate 16000`
(balanced presets). The AEC row is the composite `aec_get_mem_size()` pool — it
already contains HPF, PBFDKF ×2 (main+shadow), delay estimator, the RES/post
context and the AEC-internal FFTs. Since NE10 vendored patch P0001 the NE10
twiddle configs are carved from these pools too, so both columns are the
complete memory requirement (strict init→destroy zero-heap on both backends):

Re-measured against the current `lib/aec` pin (descriptor now reports
layout_version=10). Both the AEC column and the control block moved this
round: `sizeof(Aec)` grew 5832 -> 5848 B and the AEC pool grew by a per-grid
constant (+2,560 B @8 kHz, +5,664 B @16 kHz/256, +5,120 B @16 kHz/512,
+18,464 B @48 kHz), while `AudioPipelineLastTiming` embeds `AecStageTiming`
verbatim, which went 16 -> 20 B and pushed the control block 176 -> 192 B --
see the note under the table. Measured via
`"$(make -s print-bin-dir)"/aec_nr_pipeline_static --print-mem-size balanced
--sample-rate <sr> [--fft-size <alt>]` on both backends, against the current
grid (16 kHz default is 256/128 — see "Parameter Alignment" above). 16 kHz's
alternate 512/256 grid is included since it remains explicitly selectable.

| Rate / Backend | AEC | FFT (OLA) | NR | Pipeline bufs | **Total** |
|--------|-----|-----------|-----|---------------|-----------|
| **8 kHz KISS** | 278,256 B | 8,784 B | 67,424 B | 5,696 B | **360,352 B (351.9 KB)** |
| **8 kHz NE10** | 277,648 B | 8,176 B | 67,424 B | 5,696 B | **359,136 B (350.7 KB)** |
| **16 kHz KISS (default, 256/128)** | 385,440 B | 8,784 B | 122,160 B | 5,696 B | **522,272 B (510.0 KB)** |
| **16 kHz NE10 (default, 256/128)** | 384,832 B | 8,176 B | 122,160 B | 5,696 B | **521,056 B (508.8 KB)** |
| **16 kHz KISS (alt, 512/256)** | 513,968 B | 16,976 B | 133,472 B | 11,328 B | **675,936 B (660.1 KB)** |
| **16 kHz NE10 (alt, 512/256)** | 512,592 B | 15,600 B | 133,472 B | 11,328 B | **673,184 B (657.4 KB)** |
| **48 kHz KISS** | 1,185,536 B | 33,360 B | 374,336 B | 22,592 B | **1,616,016 B (1,578.1 KB)** |
| **48 kHz NE10** | 1,182,624 B | 30,448 B | 374,336 B | 22,592 B | **1,610,192 B (1,572.5 KB)** |

The AEC column is owned by `lib/aec/docs/c_user_manual_zh_TW.md` §4 — re-measure from
there rather than editing it here, and always prefer the value
`audio_pipeline_get_mem_requirements()` returns at runtime.

(Totals include the 192 B `AudioPipeline` control block, not broken out as
its own column above. It grew 176 -> 192 B with the `AecStageTiming` growth
described above: the record itself gained 4 B, but `ALIGN16(sizeof(
AudioPipeline))` had no slack left to absorb it, so the whole 16 B lands in
every Total. That is measured from `--print-mem-size`, which reports the
control block directly, not derived from the struct.)

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072
> taps、6 partitions at hop=512），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮
> `filter_length` 與 NR 的 `L`（48 kHz 也可用 `n_partitions` override 換較短
> 尾巴）。三個 rate 現在是各自 grid 的 hop=fft_size/2 規則自動推導
> (`aec_derive_dims()`），不再是統一的 10 ms 規則；16 kHz 另外還有一個可選的
> 512/256 grid（見上表）。並在 init 以 grid assert 驗證 pipeline/AEC/FFT/NR
> 四方一致。

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

The final sqrt-Hann WOLA introduces exactly one hop of full-chain latency
(e.g. 8 ms at the 16 kHz default 256/128 grid). NR gain and the matching AEC
context are computed and applied in the same processing call; the pipeline
does not retain the previous AEC context. The overlap state delays only the
time-domain emission. The AEC-only overlap-save path remains zero-sample
latency.

## Build

```bash
# From Audio_ALG/pipelines/ — builds component libs and mono applications
make                # aec_nr_pipeline, static variant and mono_alignulcnet
make SIMD=0         # one switch: mono pipeline + AEC + NR + audio_common all scalar

> **SIMD=0 and SIMD=1 are not bit-identical end to end, by design.** Each
> shared kernel is byte-exact against its scalar twin (proved per kernel by
> `audio_common/test/simd_selftest.c`), but the AEC matched filter diverges
> deliberately: `delay_aec3.c`'s dot product uses four accumulators plus
> `vfmaq_f32` — reordered summation *and* fused rounding — and its NLMS update
> fuses where the scalar spells a separate multiply and add. Measured on the
> mono pipeline, the two builds differ in 75 bytes at fft=256 and 67 at
> fft=512 over a 6 s stimulus. The contract is **per configuration**: a change
> must be byte-exact against the same configuration before it. Across
> configurations only finite output, the same delay winner/state and a quality
> tolerance are guaranteed — do not gate a release on cross-SIMD WAV equality.

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

# Run the audio_pipeline.h library's own acceptance tests —
# create-vs-init byte equality (incl. a poisoned pool), destroy idempotence,
# misaligned/undersized pool rejection, sample-rate whitelist rejection,
# AudioPipelineConfig reject-first validation (bad enum/bool fields),
# audio_pipeline_init_ex()'s `expected` descriptor gate (tampered
# descriptor_version/layout_version/backend_id/build_flags_hash/alignment/
# bytes each rejected) — each per-rate case runs once per supported rate
# (8000/16000/48000; 48 kHz uses a reduced hop count, see
# mono_aec_nr_res/tests/test_audio_pipeline.c)
# — AND builds + runs the example_board_adapter smoke test (see "Board
# Integration" below). Four-channel tests are intentionally isolated in
# 4ch_aec_bf_nr_res/Makefile.
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

# Build libaudio_pipeline.a with no stdio linked in at all (board images that
# forbid the stdio symbol set), and audit that it holds:
make NO_STDIO=1 libaudio_pipeline.a
make audit-no-stdio
```

`make` builds the mono executables and the existing `libaudio_pipeline.a`.
`libaudio_pipeline.a` is the pool-sizing/carving/processing library that both
mono CLIs wrap; see "Board Integration" below for its firmware API,
`NO_STDIO=1` knob, and `audit-no-stdio` target. The independent
`4ch_aec_bf_nr_res/libaudio_pipeline_4ch.a` contains only the shared linear core.
The standard and ULCNet 4-ch applications link their wrapper objects directly
with that core and the existing algorithm libraries. `four_aec_nr_res_create()` is the heap convenience
path, while `four_aec_nr_res_get_mem_requirements()` +
`four_aec_nr_res_init_ex()` place the complete four-AEC/NR/FFT/shared-wrapper
state in one caller-owned 16-byte-aligned pool. Both construction paths share
the same allocation-free pre/post processing core and are byte-parity tested
at 16 and 48 kHz. See [`4ch_aec_bf_nr_res/README.md`](4ch_aec_bf_nr_res/README.md) and
`4aec_nr_res_static` for the directly comparable board sequence.

## Debugging & Performance Flags

Both `aec_nr_pipeline` and `aec_nr_pipeline_static` support the same debug CLI
options (mirrored, byte-for-byte identical wiring in both binaries): `--debug`
for the periodic AEC+NR status line, and `--timing` for a per-stage cost
summary printed once at exit.

`--timing` is a DISPLAY flag: it decides whether a breakdown is printed, not
whether anything is measured. The measurement is a build-time choice, off by
default so a release binary takes none of the clock reads:

```sh
make PROFILE=1                 # both halves: the pipeline's own stages AND lib/aec's
"$(make -s print-bin-dir PROFILE=1)"/aec_nr_pipeline mic.wav ref.wav out.wav --timing
```

Run `--timing` against a build that did not measure and every stage reads 0;
the report says so in words rather than printing a table of zeros. Setting
only one of the two `-D` flags by hand is legible rather than broken — the
other half simply reads 0.

`CLOCK_MONOTONIC` is POSIX rather than C99. A target whose libc lacks it names
its own microsecond timer instead — `make PROFILE=1
EXTRA_CFLAGS='-DAUDIO_PIPELINE_NOW_US=board_timer_us -DAEC_NOW_US=board_timer_us
-include my_timer.h'`, a plain identifier because these Makefiles reject
parentheses in `EXTRA_CFLAGS`. Each component carries its own override, so a
chain names one timer per component it actually builds. A substitute that
returns a constant keeps the flags on and reads 0, which `--timing` cannot
distinguish from a build without them. See `AudioPipelineLastTiming` in
`mono_aec_nr_res/audio_pipeline.h` for what each stage covers and for why the
stages deliberately do not sum to the call.

There are no other optional performance compile flags — the fast
matched-filter arithmetic and delay-estimator duty-cycling are built into
`lib/aec` unconditionally.

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
for `delay_est_init_s` (default 0.3s), analysis drops to 1 hop in every
K = round(`delay_est_period_s`/hop)/5 instead of every hop — K = 12 at
8 ms hops, 6 at 16 ms, 9 at 10.67 ms, i.e. a duty period of ~96 ms on all
three of this pipeline's grids. Full-rate analysis resumes when the estimate
changes, loses solidity, or ERLE drops >6dB off its running peak.
**Sampled-quality-verified ~zero cost** (60-case AECMOS: ≤+0.014 / worst
−0.006). On a stable-delay clip the decimated schedule never actually skips a
*different* outcome; verified here on
`wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_*`.

Two things that read as "immediately" but are not. First, a decimated hop
publishes no new estimate, so "the estimate changed" can only be observed on
an ANALYSED hop: a delay movement mid-stream is seen at the next scheduled
analysis, and full rate resumes from the hop that sees it — up to one duty
period (~96 ms) later than the movement itself. Second, in THIS pipeline the
ERLE-watchdog resume leg is inert — the AEC runs linear-mode
(`enable_res=0`) so `last_erle_windowed` is never updated (same root cause as
the `--debug` `erle=0.0` caveat above). Together that leaves the
1-in-K-sampled estimate change as the only resume path here, which is exactly
the bound above and not a shorter one.

## Two Versions

### Version A: malloc (`mono_aec_nr_res/main.c`)
Each module uses `_create()` / `_destroy()` and manages its own memory internally.
Suitable for desktop testing and Linux servers.

### Version B: static memory (`mono_aec_nr_res/static_main.c`)

Built by default alongside Version A (both `lib/aec` and `lib/nr` track
`main` — each library ships the heap and static APIs side by side,
selected at runtime). One
caller-owned pool, no malloc after init, byte-identical output to Version A
(see Verification below). Since NE10 vendored patch P0001
(`audio_common/lib/ne10/VENDORED.md`), the NE10 backend's three R2C/C2R
twiddle configs are carved FROM the caller pool too
(`ne10_fft_init_r2c_float32_ext`: carve + twiddle-generate directly over
caller-supplied memory, no `malloc()` involved) and are already counted in
the `*_get_mem_size` figures — so "no malloc after init" is zero heap
allocation ever, end to end, on **both** backends, not just on the per-hop
audio path.

## Board Integration

The pool-sizing/carving/per-hop-processing logic both CLIs above embed is also
available as a standalone, linkable library —
[`audio_pipeline.h`](mono_aec_nr_res/audio_pipeline.h) /
[`audio_pipeline.c`](mono_aec_nr_res/audio_pipeline.c),
built into `libaudio_pipeline.a`. A board's own memory manager consumes this
directly instead of copying `mono_aec_nr_res/static_main.c`'s file-local carve code
into firmware; both CLIs are thin shells over it (arg parsing + WAV I/O + the
`--print-mem-size`/`--debug`/`DUMP_CTX` diagnostics).

[`example_board_adapter.c`](mono_aec_nr_res/example_board_adapter.c) is a compilable, runnable
HOST SIMULATION of that sequence (`make example-adapter`, also wired into
`make test`). **It does NOT replace production board integration and sign-off**
— every `board_mem_*` function in it is a plain host-array stand-in, marked
`// BOARD:` where real platform code belongs.

The full firmware contract — pool sizing, the query -> init -> process ->
destroy sequence, descriptor semantics, `NO_STDIO=1`, the cross-compile /
publish flow, and the on-target verification checklist — is in
[`../docs/pipeline_board_integration.md`](../docs/pipeline_board_integration.md).

## Tunable Parameters

### AEC (`AecConfig`, see `aec.h`)

**Presets**: `AEC_PRESET_MILD` / `AEC_PRESET_BALANCED`（default）/ `AEC_PRESET_AGGRESSIVE`（`MILD` 在
2026-07-15 前叫 `GENTLE`，同一組 −20dB 參數，只是改名，數值未變）

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×52ms（sr≥44100 時 sr×64ms） | 自適應濾波器長度（416@8k, 832@16k／52ms, 3072@48k／64ms；`aec_config_defaults()` 的公式，不是固定 32ms） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES / preset strength axis**：RES has no standalone `ResConfig` — it lives inside `AecConfig`
and the AEC3 post-filter chain (`SuppressionGain`, `ResidualEchoEstimator`, etc.), surfaced
externally through the `AecResContext` seam (see `## Modules` above). The three AEC presets
differ in exactly one field:

| Parameter | Mild | Balanced | Aggressive | Description |
|-----------|------|----------|------------|-------------|
| `min_gain_floor_far_active_db` | -20 | -28 | -38 | AEC3 `SuppressionGain` 遠端活躍時的最低增益下限 dB（最大抑制量）；其餘欄位（filter length、Kalman Q、delay buffer、CNG…）三個 preset 皆相同 |

### Runtime strength control（不重建、不動 pool）

兩條強度軸都可以在**運轉中的 pipeline** 上改。呼叫時機一律是**兩個 hop 之間**、
與 process 序列化；**非 thread-safe**。全部回 `0` 或 `-1`，`-1` 時什麼都不寫。

| Entry point | 目標 |
|---|---|
| `audio_pipeline_set_aec_preset(p, preset, ramp_ms)` | mono：底層 `Aec` 的 far-active 地板 |
| `audio_pipeline_set_nr_mode(p, mode)` | mono：共用降噪器（`aec_only` 建置無降噪器，回 `-1`） |
| `four_aec_nr_res_set_aec_preset(p, preset, ramp_ms)` | 4ch 核心：**共用 post 級抑制器** |
| `four_aec_nr_res_set_nr_mode(p, mode)` | 4ch 核心：那**一個**共用降噪器 |
| `four_aec_nr_res_post_split_floor(p, &live, &target)` | 4ch 核心：唯讀，線性功率。`live == target` 代表 ramp 已走完 |
| `audio_pipeline_4ch_set_aec_preset()` / `_set_nr_mode()` | 4ch 完整 wrapper：轉呼叫核心的薄殼 |

`ramp_ms == 0` 代表下一個 hop 套用（**不是錯誤**），落點與「用該 preset 從頭建一個新
實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒（mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，互動式旋鈕應該給一個 ramp）。ramp 途中再呼叫會從當前
live 值重新起走。兩者都**不是重啟**：濾波器、延遲鎖定、噪聲底與增益平滑歷史全部繼續
跑；要重啟請用對應的 `_reset()`。

> **4ch：對四條 lane 重新指定 preset 是無效操作。** 四條 lane 都以
> `spatial_linear_context` 建立，從不走到 `suppression_gain_get_gain()`，它們的地板
> 什麼都不塑形。真正乘上輸出的 gain 來自**唯一一個**共用的 post 級抑制器，這就是
> `four_aec_nr_res_set_aec_preset()` 存在、而不是要你迴圈呼叫 `aec_set_preset()` 的
> 原因（由 `4ch_aec_bf_nr_res/tests/test_4aec_nr_res.c` 的 `test_runtime_strength()` 釘住）。

> **NR：不要繞過 pipeline 的 setter 去呼叫 `mmse_lsa_set_mode()`。** 兩條 pipeline 的
> NR 組態都是「canonical preset **加上**自己的覆寫」（`broadband_threshold`、`L`、
> `alpha_decay`，見兩處的 `compose_nr_config()`）。`mmse_lsa_set_mode()` 組的是裸的
> canonical preset，在這種實例上會被**拒絕**（它的 `L` 不同）——所以 pipeline 的
> setter 做的事是重組完整組態再交給 `mmse_lsa_reconfigure()`。

> **A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
> hop 生效：double-talk 期間套的是 DT 地板，而 DT 地板三個 preset **完全相同**；
> far-active latch 觸發前套的是 far-silent 地板。同一個 `G_res` 還決定注入的 comfort
> noise 量（振幅正比於 `sqrt(1 - G_res^2)`，見兩處實作的 CNG 步驟——地板越深、CNG
> 反而越多）。所以**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且只量
> echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制。請在 echo 對齊或
> degradation 對齊的條件下比較，並實際試聽。

兩條 Align-ULCNet pipeline（`mono_alignulcnet/`、`4ch_alignulcnet/`）**刻意不提供**
這些 setter：mono 變體沒有 NR 實例、且算出的 `G_res` 沒有任何消費者（輸出由 ULCNet
mask 塑形）；4ch 變體用的是 `core.enable_post = 0` 的 pre-only 核心，抑制器與降噪器
根本不存在。強度是模型的性質，不是 runtime 旋鈕。

Python 對應：`AEC.set_preset(preset, ramp_ms=0.0)` 與
`FourChannelAecPipeline.set_aec_preset(preset, ramp_ms=0.0)`（後者同樣只動共用的
post-beam 抑制器）。

### NR (`MmseLsaConfig`, see `mmse_lsa_types.h`)

**Modes**: `MMSE_LSA_NR_MILD` / `MMSE_LSA_NR_MODERATE` / `MMSE_LSA_NR_BALANCED`（default）/ `MMSE_LSA_NR_AGGRESSIVE`

> These are the library's mode enum. Both C pipeline CLIs' `parse_nr_mode()` —
> `mono_aec_nr_res/main.c` and `4ch_aec_bf_nr_res/4aec_nr_res_static.c` — now also recognize
> `"moderate"` (`--nr-preset mild|moderate|balanced|aggressive`, since 2026-09-03); anything else
> still silently falls back to `MMSE_LSA_NR_BALANCED` — no error. The Python `aec_nr_pipeline.py`
> CLI has not been updated and still restricts `--nr-preset` to `choices=['mild', 'balanced',
> 'aggressive']`; `MODERATE` there is only reachable by calling `mmse_lsa_config_for_mode()`
> directly.

`g_min_db` is in the amplitude-dB convention (`/20`, i.e. `g_min = 10^(g_min_db/20)`), not the
older power-dB (`/10`) convention.

> **Retime basis — read before hand-editing any `alpha_*`/`L`/`num_init_frames` below.**
> Every `alpha_*` (except `q`/`g_min_db`/`xi_min_db`, which are plain dB/probability
> constants) and `L`/`num_init_frames`/`scene_change_min_frames` is **not** used as-authored —
> `mmse_lsa_types.h`'s `mmse_lsa_config_for_mode_grid()` retimes each one from its authored
> reference duration to the actual `hop_size` of the grid you construct it for
> (`mmse_lsa_retime_alpha()`/`mmse_lsa_retime_alpha_ref()`/`mmse_lsa_retime_frames[_ref]()`).
> The tables below list the **authored constants exactly as they appear as literals in
> source** (what you'd override via `MmseLsaConfig` before construction, or diff against in
> code) — **not** the realized runtime value at any particular grid; `L`/`num_init_frames`
> are the one exception, called out per-row below, since a plain frame count without a
> stated grid is meaningless.
>
> The reference duration is **not the same for every field** — mixing them up when
> hand-deriving a value for a different grid silently reintroduces the exact regression the
> 2026-07-10 musical-noise fix and the 2026-08-02/03 strength-mode fix both corrected:
> - **16 ms-authored** (retimed off a 16 ms/hop reference): `alpha_xi`, `L`, `alpha_attack`
>   (all modes), and `alpha_d`/`alpha_g`/`alpha_decay` **in MILD/MODERATE/AGGRESSIVE only**.
> - **10 ms-authored** (retimed off the legacy 10 ms/hop reference): `alpha_s`, `alpha_p`,
>   `num_init_frames`, `scene_change_min_frames`, and `alpha_d`/`alpha_g`/`alpha_decay`
>   **in BALANCED only** (BALANCED inherits these three straight from
>   `mmse_lsa_default_config_for_grid()`, which predates the 16 ms-hop grid switch).
>
> Don't hand-scale one of these numbers by a grid's sample-rate/hop ratio and expect it to
> match the library's own value — call `mmse_lsa_default_config()`/`mmse_lsa_config_for_mode()`
> (or the Python `core/nr_strength.py` equivalent) and read the field back instead.

| Parameter | Mild | Moderate | Balanced | Aggressive | Description |
|-----------|------|----------|----------|------------|-------------|
| `g_min_db` | -20 | -25 | -30 | -40 | 最小增益 dB（最大抑制量，amplitude dB, /20） |
| `q` | 0.60 | 0.55 | 0.50 | 0.35 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -15 | -18 | -20 | -25 | 先驗 SNR 下限 dB |
| `alpha_d` | 0.85 | 0.85 | 0.70 | 0.50 | 噪聲追蹤 IIR 係數（authored 常數，見上方 retime 說明） |
| `alpha_g` | 0.92 | 0.92 | 0.88 | 0.85 | 增益時間平滑（高→平滑；authored 常數） |
| `alpha_attack` | 0.40 | 0.40 | 0.30 | 0.15 | Attack 平滑（語音起始追蹤；authored 常數，四個 preset 皆 16ms 基準） |
| `alpha_decay` | 0.92 | 0.92 | 0.88 | 0.88 | Decay 平滑（噪聲釋放；authored 常數） |

**MCRA 噪聲估計**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_s` | 0.95 | 功率譜時間平滑（authored 常數，10ms 基準） |
| `alpha_d` | 0.70 | 噪聲更新速率（authored 常數，10ms 基準；BALANCED 值，見上方 retime 說明） |
| `L` | 64 | 最小值追蹤視窗（幀數；**已 retime**；authored 值為 32 hops @ 16 ms，16 kHz 預設 256/128 grid 的實際 hop 為 8 ms，因此 effective 值為 64；其他 grid 請由設定 API 讀回） |
| `num_init_frames` | 25 | 初始化靜默幀數（**已 retime**；常數以 10 ms 為 authored basis，16 kHz 預設 256/128 的實際 hop 是 8 ms，因此 25 hops ≈200 ms） |
| `scene_change_threshold_db` | 10.0 | 場景轉換偵測閾值（純 dB 常數，不 retime） |

**SPP**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_xi` | 0.92 | Decision Directed 先驗 SNR 平滑（authored 常數，16ms 基準；2026-07-10 musical-noise fix 由 0.88 調高，所有 preset 共用） |

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
