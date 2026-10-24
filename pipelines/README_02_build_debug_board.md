Navigation: [Index](README.md) | [Previous part](README_01_latency_call_flow.md) | [Next part](README_03_tuning_troubleshooting.md) <!-- split-nav -->
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

# Mono chain test (the examples' pool carve vs heap, reset, switches, comfort noise, and both
# mono programs run as black boxes against the hand-written oracle
# mono_aec_nr_res/tests/conv_chain_mono.h; mono_aec_nr_res/tests/test_mono_chain.c)
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
libraries (`libaec.a`, `libanr.a`, `libaudio_common.a`) directly and carry their
per-hop chain in their own sources. `4ch_aec_bf_nr_res/libaudio_pipeline_4ch.a` holds the shared
linear core (`4aec_nr_res.o`), the unit (`audio_pipeline_4ch.o`) and
`common/`'s `se_config.o`; `audit-no-stdio` covers it for the 4-channel build. The example
programs (`audio_pipeline_4ch_raw`, `audio_pipeline_4ch_static`, `4aec_nr_res_static`,
`4ch_aec_bf_dfn_res`, `4ch_alignulcnet`) link the archive and the example headers or objects
of `samples/`. The unit carries no RES: every 4-channel example that has a RES (all but `4ch_alignulcnet`, which runs
with `core.enable_post = 0` and has none) builds its own `AecRes` after it
(`aec_res_init()`), `4aec_nr_res_static` on the core. `four_aec_nr_res_create()` is the heap convenience
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
Each library block (the `Aec`, the denoiser with its synthesis and gain buffers) is allocated on the
heap with `posix_memalign()` at the size its own `*_get_mem_size()` query returned, then initialised
with the same `aec_init()` / `anr_init()` / `ac_istft_init()` calls as Version B; `rig_free()` releases
the blocks after `anr_destroy()` / `aec_destroy()`. Suitable for desktop testing and Linux servers.

### Version B: static memory (`mono_aec_nr_res/static_main.c`)

Built by default alongside Version A. One caller-owned pool, no malloc after
init; its output equals Version A's (see [Verification](README_03_tuning_troubleshooting.md#verification) below). With NE10
vendored patch P0001 (`audiolib/common/lib/ne10/VENDORED.md`), the NE10 backend's three R2C/C2R
twiddle configs are carved FROM the caller pool too
(`ne10_fft_init_r2c_float32_ext`: carve + twiddle-generate directly over
caller-supplied memory, no `malloc()` involved) and are already counted in
the `*_get_mem_size` figures — so "no malloc after init" is zero heap
allocation ever, end to end, not just on the per-hop audio path.

## Board Integration

沒有 mono 函式庫 TU：板端自己用 `se_derive()` 導出 grid 與 `AecConfig`，查
`aec_get_mem_size()`、`anr_get_mem_size()` 與 `ac_istft_get_mem_size()`，加上每 hop 的
增益／頻譜緩衝，把各塊切在自己的 pool 裡，再逐個 `init`
（`mono_aec_nr_res/static_main.c` 的 `post_plan()` / `pool_bytes()` / `rig_build()` 就是這段
的最小寫法；每 hop 的串接寫在該檔的 `mono_form()`，可整段複製改寫）。唯一帶 descriptor 的單元是 4ch
的 `AudioPipeline4Ch`（`audio_pipeline_4ch_get_mem_requirements()` →
`audio_pipeline_4ch_init_ex()`；後波束 `AecRes` 不在它的 pool 內），其餘函式庫（`Aec`、
`AnrDenoiser`、`AcIstft`、`DfnResStage`、`DfnRateBridge`，以及 4ch 應用在單元之後自備的
`AecRes`）各有自己的 `*_mem_size` 與 `init`，整合者自己加總。

[`example_board_adapter.c`](mono_aec_nr_res/example_board_adapter.c) 是這個順序的可編譯、
可執行的 HOST 模擬（`make example-adapter`，也接進 `make test`）：pool 計畫（AEC、
降噪器、合成與緩衝的大小加總）、過小或未對齊的 pool 與重複 destroy 被拒絕、被拒的 init 不弄髒 pool、reset、destroy、pool 重用。
**它不取代板端整合與簽核**——裡面每個 `board_mem_*` 都只是主機陣列的替身，
真正的平台程式碼在標了 `// BOARD:` 的位置。

完整的韌體契約——pool 大小、query → init → process → destroy 的順序、descriptor
語意、`NO_STDIO=1`、交叉編譯／publish 流程與板上驗證清單——在
[`../docs/pipeline_board_integration.md`](../docs/pipeline_board_integration.md)。

Navigation: [Index](README.md) | [Previous part](README_01_latency_call_flow.md) | [Next part](README_03_tuning_troubleshooting.md) <!-- split-nav -->
