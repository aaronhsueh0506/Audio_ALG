Navigation: [Index](README.md) | [Previous part](README_01_latency_call_flow.md) | [Next part](README_03_tuning_troubleshooting.md) <!-- split-nav -->
## Build

```bash
# From Audio_ALG/pipelines/ — builds component libs and mono applications
make                # aec_nr_pipeline, aec_nr_pipeline_static, mono_alignulcnet, mono_aec_dfn_res, libaudio_pipeline_dfn.a
make SIMD=0         # one switch: examples + AEC + NR + audiolib all scalar

# SIMD=0 and SIMD=1 outputs are not bit-identical end to end (see the note below).

# Binaries land in a config-keyed directory:
#   bin/<backend>-<config-hash>/  — resolve it with `make print-bin-dir`
# (same flags as your build), or use `make publish` for the stable
# dist/<backend>/current/ handoff path.
BIN="$(make -s print-bin-dir)"

# Run Version A (split pool: the AEC block and a post-stage block)
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-preset aggressive

# Run Version B (one caller pool) — same CLI, plus a mem-size query mode
"$BIN"/aec_nr_pipeline_static mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline_static --print-mem-size --sample-rate 16000

# Mono chain test (the examples' pool carve vs heap, reset, switches, comfort noise, and both
# mono programs run as black boxes against the hand-written oracle
# mono_aec_nr_res/tests/conv_chain_mono.h; mono_aec_nr_res/tests/test_mono_chain.c)
# plus the example_board_adapter smoke test (see "Board Integration" below), the DFN2 and
# ULCNet stage tests and the mono DFN2 / ULCNet programs.
# Four-channel tests live in 4ch_aec_bf_nr_res/Makefile.
make test

# Build/test the application packages.
make -C mono_aec_nr_res SIMD=0 test
make -C ulcnet/mono_alignulcnet SIMD=0 test
make -C dfn/mono_aec_dfn_res SIMD=0 test
make -C 4ch_aec_bf_nr_res SIMD=0 test
make -C ulcnet/4ch_alignulcnet SIMD=0 test
make -C dfn/4ch_aec_bf_dfn_res SIMD=0 test
make -C 4ch_aec_bf_nr_res mcab_external_static
make -C 4ch_aec_bf_nr_res mcab_raw
make -C 4ch_aec_bf_nr_res mcab_static

# Build + run JUST the REFERENCE ONLY board-adapter example standalone
# (also runs as part of `make test` above):
make example-adapter

# Build the DFN2 stage archive with no stdio linked in at all (board images that
# forbid the stdio symbol set), and audit that it holds:
make NO_STDIO=1 libaudio_pipeline_dfn.a
make audit-no-stdio

# A DeepFilterNet2 lookahead pair other than the default 1/1 (see "DFN lookahead pair" below),
# and the DFN C tests rebuilt and run per pair:
make DFN2_MASK_LOOKAHEAD=2 DFN2_DF_LOOKAHEAD=4 libaudio_pipeline_dfn.a mono_aec_dfn_res
make test-pairs DFN_TEST_PAIRS="0/0 2/4"

# Python twin (module paths are relative to the Audio_ALG root):
(cd .. && python3 -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav --nr-preset balanced)
```

> **SIMD=0 and SIMD=1 are not bit-identical end to end, by design.** Each shared
> kernel is byte-exact against its scalar twin (`audiolib/common/test/simd_selftest.c`),
> but the AEC matched filter (`delay_aec.c`) uses four accumulators with
> `vfmaq_f32` and a fused NLMS update. The contract is per configuration: a
> change must be byte-exact against the same configuration before it. Across
> configurations only finite output, the same delay winner/state and a quality
> tolerance hold; do not gate a release on cross-SIMD WAV equality.

The Python twins sit beside their applications: `mono_aec_nr_res/pipeline.py`, `4ch_aec_bf_nr_res/pipeline.py`
(`McabPipeline`, `McabFrame`, `mcab_apply_lane_contract()`, `mcab_lane_config()`), `dfn/dfn2_stage.py`,
`dfn/dfn2_rate_adapter.py`, `dfn/mono_aec_dfn_res/pipeline.py`, `dfn/4ch_aec_bf_dfn_res/pipeline.py` and
`common/cng_aec3.py` (module paths `pipelines.<folder>.<module>`). Python and C agree for the default settings; the documented
differences S-02 (AEC live observer guard) and S-13 (NR make-up prior) exist only when those default-off features are enabled,
so do not read the twins as identical for every setting (`docs/c_python_function_map.md`, `make pyc-parity`; the two differences are defined in `lib/aec/c_impl/test/pyc_parity/pyc_parity.py` and `lib/nr/docs/c_python_function_map.md`).

**DFN lookahead pair.** The DeepFilterNet2/3 model lookahead is a build-time pair: mask `M` in 0..2 and deep-filter
`D` in 0..4 (15 legal pairs, no ordering rule between them; `AINR/README.md`, "Lookahead pairs"). The exported graph does
not depend on the pair, but a checkpoint is tied to the pair it was trained for, and only the shipped 1/1 has one: the
audio quality of every other pair is unmeasured. `Makefile` takes `DFN2_MASK_LOOKAHEAD` and `DFN2_DF_LOOKAHEAD` (and
`DFN3_MASK_LOOKAHEAD`, `DFN3_DF_LOOKAHEAD`), default 1/1, each a single decimal in range. A pair other than 1/1 is
forwarded as `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` to every unit and is part of the build signature: it gets
its own `bin/` and `obj/` directories, and units built for different pairs fail to link (pair symbol
`dfn2_lookahead_mask<M>_df<D>`). `4ch_aec_bf_nr_res/Makefile` takes the same two DFN2 variables. The DFN2 stage delays its
output by `M + D` frames of 48 kHz; the rate bridge adds `dfn_rate_bridge_added_delay_samples()` native samples (the floor; `dfn_rate_bridge_added_delay_exact()` returns the exact value as a multiple of 1/3 sample), which grow
with `M + D` (shipped pair: 672, 629 and 330 samples at 16 kHz hop 128, 16 kHz hop 256 and 8 kHz hop 128). `make test-pairs`
rebuilds and runs the DFN C tests and the mono DFN2 program for each pair of `DFN_TEST_PAIRS` (default `0/0 0/1 2/0 2/4`); the
other pairs are not exercised by it.

`make` builds the mono executables and the DFN2 stage archive. There is no mono
library archive: the mono examples link the producer libraries (`libaec.a`, `libanr.a`,
`libaudio_common.a`) directly and carry their grid derivation and per-hop chain in their own
sources. `4ch_aec_bf_nr_res/libaudio_pipeline_mcab.a` holds
`mcab.o` only (the flow), and its build directory is keyed on audiolib common and lib/aec alone: `make
libaudio_pipeline_mcab.a` succeeds with lib/nr and audiolib's spatial kernels absent (`make -s print-lib-path` in
`4ch_aec_bf_nr_res` prints it; the same goal built by `make all` lands in the full-scope directory, which
`PRINT_FOR=all` selects). The sample objects `mcab_sample_flow.o`
and `mcab_sample_beam.o` (the application-side init and the SRP-PHAT/BF helper) are not in it
either; `audit-no-stdio` (run there with `NO_STDIO=1`) covers the archive, the DFN archive, the
producers and those two objects. The example programs (`mcab_raw`, `mcab_static`,
`mcab_external_static`, `4ch_aec_bf_dfn_res`, `4ch_alignulcnet`) link the archive and the
example headers or objects. `Mcab` carries no RES: every 4-channel example that has a RES (all
but `4ch_alignulcnet`, which runs with `enable_post = 0` and has none) builds its own `AecRes`
(`aec_res_init()`). The four lanes and the shared estimator are likewise built by the
application with `aec_init()` and `delay_aec_init()` and passed to `mcab_init_ex()` by pointer;
`mcab_create()` is only the heap convenience for mcab's own pool, while
`mcab_get_mem_requirements()` + `mcab_init_ex()` place the flow state in one caller-owned
16-byte-aligned pool. Both paths share the same allocation-free processing code;
`4ch_aec_bf_nr_res/tests/test_mcab.c` checks the pooled flow at 16 kHz/256, 16 kHz/512 and
48 kHz/1024 over four pool placements and four fill patterns. See
[`4ch_aec_bf_nr_res/README.md`](4ch_aec_bf_nr_res/README.md) and `mcab_external_static.c` for the
directly comparable board sequence.

## Debugging & Performance Flags

`aec_nr_pipeline` and `aec_nr_pipeline_static` support the same debug CLI
options (one source file, `mono_aec_nr_res/main.c`, builds both binaries): `--debug`
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
identifier because these Makefiles reject parentheses in `EXTRA_CFLAGS`. `mcab`
has its own override (`-DMCAB_NOW_US=...`), so a build names one timer per
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

One source file, `mono_aec_nr_res/main.c`, builds both binaries; they differ only in how the
stages are placed in memory. The `Aec`, the denoiser and the synthesis are initialised with the same
`aec_init()` / `anr_init()` / `ac_istft_init()` calls in both.

### Version A: split pool (`aec_nr_pipeline`)
Built with `-DAEC_NR_POOL_SPLIT=1`. Two blocks, each allocated with `posix_memalign()` at the size
the libraries' `*_get_mem_size()` queries returned: the `Aec`, and one block for the post stages
(denoiser, synthesis, gain and spectrum buffers); `rig_free()` releases them after
`anr_destroy()` / `aec_destroy()`. Suitable for desktop testing and Linux servers. It has no
`--print-mem-size`.

### Version B: one caller pool (`aec_nr_pipeline_static`)

Built by default alongside Version A. One caller-owned pool, no malloc after
init; its output equals Version A's (see [Verification](README_03_tuning_troubleshooting.md#verification) below). With NE10
vendored patch P0001 (`audiolib/common/lib/ne10/VENDORED.md`), the NE10 backend's three R2C/C2R
twiddle configs are carved FROM the caller pool too
(`ne10_fft_init_r2c_float32_ext`: carve + twiddle-generate directly over
caller-supplied memory, no `malloc()` involved) and are already counted in
the `*_get_mem_size` figures — so "no malloc after init" is zero heap
allocation ever, end to end, not just on the per-hop audio path.

## Board Integration

沒有 mono 函式庫 TU：板端自己用 `aec_resolve_signal_grid()` 與 `aec_config_from_preset()` 導出 grid 與 `AecConfig`
（再設接縫欄位 `enable_res = 0`、`return_res_context = 1` 等），查
`aec_get_mem_size()`、`anr_get_mem_size()` 與 `ac_istft_get_mem_size()`，加上每 hop 的
增益／頻譜緩衝，把各塊切在自己的 pool 裡，再逐個 `init`
（`mono_aec_nr_res/main.c` 的 `derive_lane()` / `post_plan()` / `pool_bytes()` / `rig_build()` 就是這段
的最小寫法；每 hop 的串接寫在該檔的 `mono_form()`，可整段複製改寫）。帶 descriptor 的物件是 4ch
的 flow 層 `Mcab`（`mcab_get_mem_requirements()` → `mcab_init_ex()`；它只涵蓋 mcab 自己的 pool，四條
`Aec` lane、共用 `DelayAec`、後波束 `AecRes` 都不在內）以及 DFN2 的 stage／bridge，其餘函式庫
（`Aec`、`AnrDenoiser`、`AcIstft`、`DfnResStage`、`DfnRateBridge`，以及 4ch 應用自備的 lane、
estimator 與 `AecRes`）各有自己的 `*_mem_size` 與 `init`，整合者自己加總。

[`example_board_adapter.c`](mono_aec_nr_res/example_board_adapter.c) 是這個順序的可編譯、
可執行的 HOST 模擬（`make example-adapter`，也接進 `make test`）：pool 計畫（AEC、
降噪器、合成與緩衝的大小加總）、過小或未對齊的 pool 與重複 destroy 被拒絕、被拒的 init 不弄髒 pool、reset、destroy、pool 重用。
**它不取代板端整合與簽核**——裡面每個 `board_mem_*` 都只是主機陣列的替身，
真正的平台程式碼在標了 `// BOARD:` 的位置。

完整的韌體契約——pool 大小、query → init → process → destroy 的順序、descriptor
語意、`NO_STDIO=1`、交叉編譯／publish 流程與板上驗證清單——在
[`../docs/pipeline_board_integration.md`](../docs/pipeline_board_integration.md)。

Navigation: [Index](README.md) | [Previous part](README_01_latency_call_flow.md) | [Next part](README_03_tuning_troubleshooting.md) <!-- split-nav -->
