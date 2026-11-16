# Changelog

`Audio_ALG/pipelines`（含它建置的 `AINR`、`AIAEC` C 程式）的重要改動記錄在此，格式依
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。README 只寫現況，改動經過寫在這裡。

## [Unreleased] — 2026-11-15 — `stream/common/stream_run.h`: the run contract of the AI model streams

- **New header** `stream/common/stream_run.h` (header-only, SDK-free): `StreamTensorIn` / `StreamTensorOut`, `StreamWorkAction` (`STREAM_WORK_BYPASS`, `STREAM_WORK_SUBMIT`), `StreamWork` (up to `STREAM_WORK_MAX_INPUTS` = 5 inputs and `STREAM_WORK_MAX_OUTPUTS` = 3 outputs, `state_generation`) and the run statuses `STREAM_RUN_OK` (0), `STREAM_RUN_NOT_STARTED` (0x4E53), `STREAM_RUN_FAILED` (-1). Nothing includes it yet; `ainr.h` and `aiaec.h` take it over in their own changes.
- **Tests**: `stream/common/tests/test_stream_run.c` (values, member order, constness, an all-zero work being an empty BYPASS block, `GRAPH_IO_CHECK_WORK` over a `StreamWork`) and `make -C stream/common check-stream-run` (the header alone as C99 `-Wpedantic`, included twice, and as C++11, compiled and run). Both are part of `make -C stream/common test`.

## [Unreleased] — 2026-11-16 — DFN2: the library closes a run that did not succeed (`dfn2_prepost_frame_fail`, three-way run status)

- **New** `dfn2_prepost_frame_fail(p, run_started)` (`lib/ainr/dfn2`): closes the open frame with the `frame_bypass` output. `run_started != 0` (the accelerator may have moved its state) zeroes the registered state and advances the state generation; `0` (nothing ran) leaves both. A warm-up frame steps the generation once either way, as before. -1 for NULL or no open frame.
- **`dfn2_stream_post_process{,_freq}`** read the status of `stream/common/stream_run.h`: `STREAM_RUN_OK` commits; `STREAM_RUN_NOT_STARTED` closes with `frame_fail(p, 0)` (state and generation untouched); any other value closes with `frame_fail(p, 1)` (generation +1). Before, any non-zero status was a plain bypass and the generation did not move, so the caller had to zero the device state itself.
- **Behaviour change**: `dfn2_model_run_frame` with a failing `infer()` is `frame_fail(p, 1)`: registered state zeroed, generation +1 (before: state frozen, generation unchanged). A caller that zeroes the accelerator state only when `io.state_generation` changed now covers a failed run by itself; a caller that still zeroes after every non-zero status stays correct (one extra zeroing after a run that never started). Accepted frames are bit-identical (PCM, all catalog grids, N, `DFN2_MASK_ONLY` 0/1, SIMD 0/1).
- `dfn2_stream.h` usage text follows the new protocol (no "refill zero after a failed run"). No layout version changes.
- **Tests**: `stream/ainr/dfn2/tests/fail_table.c` (nine statuses x warm-up / steady / N > 1 against a stand-in accelerator that keeps its own state, part of `make -C stream/ainr/dfn2 test`); `verify_vs_sample.c` oracle closes a failed run with `frame_fail` and asserts the generation step (+1 started, 0 not started); `AINR/tests/test_dfn2_prepost_c.py` gains the `frame_fail` case (all lookahead pairs) and its failing-`infer()` pins change (state zeroed, generation +1); `test_dfn2_runtime_grid.py` generation line moves by one per failing model.
- `docs/c_python_function_map.md`: AINR-DFN2-16 lists `dfn2_prepost_frame_fail`; statistics updated.

## [Unreleased] — 2026-11-16 — DeepVQE-S: failure recovery in the library, the CCM ring restarts with the state generation

Accepted frames are unchanged. What moves is the generation after a failed run and the frames that follow a state restart.

- **`deepvqe_prepost_frame_fail(p, run_started)`** (`lib/aiaec/deepvqe_s`). Closes the open batch after a run that did not succeed, with the `frame_bypass` output. `run_started != 0` (the device state may be dirty) also zeroes the registered state, empties the CCM ring and advances the generation; `0` changes nothing. -1 with nothing changed when no batch is open. Function map row AIAEC-DV-07.
- **`deepvqe_stream_post_process{,_freq}`** closes a job by the run status of `stream/common/stream_run.h`: `STREAM_RUN_OK` commits (a refused commit restarts the state inside `frame_finish`, then bypasses), `STREAM_RUN_NOT_STARTED` leaves the device state alone, any other value counts as a dirty state. Before, any status other than 0 only bypassed and left the generation where it was, so the sample had to zero its own state after a failure; now the generation says so. The `ai_rc` of a no-job hop is still ignored.
- **CCM ring restart.** `zero_state()`, the only place the generation advances, also empties the CCM ring: a refused commit, `frame_fail` with a started run and `reset` all give a cold start (empty ring, zero state), as in the Python model. `frame_bypass` and `frame_fail(p, 0)` keep the ring. Measured on the PCM stream (`tools/ident.sh`, 120 hops, no registered state): a clean run is bit-identical to the previous commit; in a run with five restarts, every hop outside the three after each restart (two while the ring refills, one for the overlap-add of the previous hop) is bit-identical.
- **No layout change.** No pool carve or public struct size moves (`DEEPVQE_STREAM_LAYOUT_VERSION` 4, `DEEPVQE_PREPOST_LAYOUT_VERSION` 4, `DEEPVQE_PREPOST_CARVE_VERSION` 8).
- **Tests and mutation evidence.** `verify_vs_sample` drives the statuses `STREAM_RUN_FAILED`, `STREAM_RUN_NOT_STARTED`, 7 and 1 and a refused commit with and without a failure status, and requires the generation to move by exactly one on every hop but `NOT_STARTED` (before: "must not move on a bypass"). `test_deepvqe_prepost_c.py` has the `fail` and `failboard` cases. The new `AIAEC/tests/test_deepvqe_ccm_restart_c.py` compares the frames after a restart with a fresh instance (class level at N = 1 and 4, 16 kHz and 48 kHz, native and scalar kernels; board form with nothing registered; the `_freq` entries of `deepvqe_stream`), each paired with the same frames after no restart. The three named pins of `test_deepvqe_prepost_c.py` keep their values; two claim texts are corrected. Thirteen mutations of the library and the stream (frame_fail ignoring `run_started`, no generation advance, state not zeroed, always dirty, no open-batch gate, NOT_STARTED counted dirty, a refused commit counted twice, status ignored; ring not emptied, emptied by a bypass, emptied by `frame_fail(p, 0)`, kept by the `frame_fail` path, kept by a refused commit) are all killed.
- **Docs.** `docs/html/models/deepvqe-s/{index,reference/core}.html`, the DeepVQE-S sentences of `docs/html/models/reference/onnx.html`, `deepvqe_prepost.h`, `deepvqe_stream.h`, `c_python_function_map.md` AIAEC-DV-07.

## [Unreleased] — 2026-11-16 — Align-ULCNet: failed-run recovery in the library, an alignment event restarts only the model

- **Align-ULCNet: a failed run is closed by the library (`ulcnet_prepost_frame_fail`, `stream_run.h` statuses).** `ulcnet_prepost_frame_fail(p, run_started)` closes the open frame after a run that did not succeed with the `frame_bypass` output. `run_started != 0` zeroes the registered state and advances the state generation; `run_started == 0` leaves both, except that a `frame_open()`ed packet with an event frame restarts the model (generation +1, once), the restart its graph would have applied. That event loop used to sit in `ulcnet_prepost_frame_bypass()`, which is a pure identity now (error spectrum through; state and generation untouched). `ulcnet_stream_post_process{,_freq}` closes every hop and packet three ways: `STREAM_RUN_OK` commits (a refused commit restarts the model, then bypasses), `STREAM_RUN_NOT_STARTED` and any other status call `frame_fail` with `run_started = (status != STREAM_RUN_NOT_STARTED)`. Behaviour change: a status other than 0 used to leave the generation alone; it advances it now unless it is `STREAM_RUN_NOT_STARTED`. `ulcnet_stream.h` includes `stream_run.h`, so every TU that includes it needs `stream/common` on its include path (`pipelines/Makefile`, `pipelines/4ch/Makefile`: `ULCNET_STREAM_INC`). The desktop mains' stub accelerators return `STREAM_RUN_NOT_STARTED`. Tests: `AIAEC/tests/test_ulcnet_prepost_c.py` case `fail`, `verify_vs_sample` (failed hop +1, not-started hop +0), `test_ulcnet_stream_freq` (fail-open rows by status), `test_aiaec_packet` `section_failure` (status 7, `NOT_STARTED` with and without an event frame), the pipeline fixture counts a generation step per dirty status. Docs and examples no longer clear the device state by hand after a failed run.
- **Align-ULCNet: an alignment event restarts only the model, on every entry.** The PCM entries (`ulcnet_stream_pre_process{,_f32}`, `_packet_f32`) no longer reset the stream's own `AcStft` pair or the `AcIstft` at an event hop, so the event hop is no longer 16 ms of silence and the previous hop's audio is no longer lost. The PCM of every hop, event hops included, equals the spectrum entries around the caller's own `AcStft` / `AcIstft`; N = 8 still equals N = 1. Removed: `ulcnet_stream_restart_mask()`, `UlcnetStream.restart_mask`, `aiaec_set_restart` and its two calls (the `AiaecWork` restart fields stay and are always 0). `ULCNET_STREAM_LAYOUT_VERSION` 4 -> 5 (pooled struct lost a field; pool bytes unchanged at the tested grids). `ulcnet_stream_reset()` still resets everything. Measured on a 48-hop stream with events at 5, 21, 22, 33 and 40 (16 kHz / 256): only those five hops differ from the previous build (all zero before, 1.1e9 to 2.5e9 energy after); every other hop is bit-identical. New test `test_ulcnet_pcm_events` (two grids; f32, int16 and N = 8 entries against `_freq` around `AcStft` / `AcIstft`). `verify_vs_sample`'s oracle restarts the model only at an edge; `test_front_block`, `test_front_oneshot`, `test_aiaec_spec_freq` and `test_aiaec_packet` expect no mute and restart fields of 0. Function map: exception D-44 deleted (row PL-ULC-01 and `alignment.md` reworded). The board sample `alg_ai3_alignulcnet.c` needs no code change for this (it calls `aiaec_front_preprocess` / `aiaec_postprocess`; the framing is inside the library): rebuild it, and its A/B reference recording changes at the event hops.

## [Unreleased] — 2026-11-16 — AINR product: `StreamWork`, run status in the DFN2 product and the desktop DFN pipelines, 4ch direct-path reset

- **`StreamWork`.** `ainr_preprocess{,_freq}` take `StreamWork *` (`stream/common/stream_run.h`). `AinrWork`, `AinrTensorIn`, `AinrTensorOut`, `AINR_MAX_INPUT_TENSORS`, `AINR_MAX_OUTPUT_TENSORS` are removed with no alias; `ainr_fill_work` sets `action = STREAM_WORK_SUBMIT` on every block. A sample passes `STREAM_WORK_MAX_INPUTS` / `STREAM_WORK_MAX_OUTPUTS` to `graph_io_fits()`.
- **Run status (product).** `ainr_postprocess{,_freq}(ctx, run_status, ...)` (was `infer_status`) hands the status to `dfn2_stream_post_process{,_freq}`: 0 commits, `STREAM_RUN_NOT_STARTED` leaves state and generation, any other value zeroes the registered state and advances the generation. A sample no longer needs a "recovery pending" flag: it zeroes the device state when `work.state_generation` differs from its value at the last zeroing. `ainr.h` carries the rule sentence of the stream headers.
- **Run status (desktop DFN).** `dfn_hop_infer()` / `dfn_res_stage_infer()` return `STREAM_RUN_NOT_STARTED` without a model; `dfn_hop_end()` and `dfn_rate_bridge_finish()` close a frame left open with `STREAM_RUN_FAILED`, a frame never fetched with `STREAM_RUN_NOT_STARTED`; `dfn_res_stage_end()` closes through `dfn2_prepost_frame_fail`. The two `run_accelerator()` stubs return `STREAM_RUN_NOT_STARTED`.
- **4ch direct-path reset.** `4ch_aec_bf_dfn_res` (direct path) delays an upstream hard reset by `dfn_hop_lookahead_frames()` hops with a per-hop shift register and clears only the hop's `AcIstft`; it no longer calls `dfn_hop_reset()` on the event.
- Behaviour change on the 48 kHz 4ch path: the event no longer restarts DFN2 (the old output lost `queue_frames` hops and the model state; measured 2 hops differ, generation 2 -> 4 with the old rule).
- Not changed in this entry (the later entries on the DFN2 and DFN3 warm-up, the DeepVQE-S packet entries and the product Info fields change them): warm-up job, product stashes, trim/queue/tail fields, latency formula.

## [Unreleased] — 2026-11-16 — AIAEC product: `aiaec_*_spec` retired, shared `StreamWork`, run status and generation rule in the library, `AIAEC_ERR_NONFINITE_INPUT`

Finite input and accepted frames are unchanged. Pool sizes of the PCM and `_freq` families are unchanged (DeepVQE-S 16k/256 N=1 64192 / 18368, N=8 167024 / 126640, 16k/512 124096 / 35776; Align-ULCNet 16k/256 N=1 53744 / 7920, 16k/512 103408 / 15088; PCM / `_freq`, bytes).

- **`aiaec_*_spec` family retired.** Removed: `aiaec_get_mem_size_spec`, `aiaec_init_spec`, `aiaec_reset_spec`, `aiaec_state_generation_spec`, `aiaec_preprocess_spec`, `aiaec_postprocess_spec`, `AiaecSpecCtx`, `AIAEC_FAMILY_SPEC` and `AiaecWork.synthesis_restart_mask` / `synthesis_restarted` (always 0 since the PCM entries restart only the model). Migration: `AcStft` on the input, `aiaec_*_freq`, `AcIstft` on the output; the PCM family equals that chain sample for sample, event hops included. `test_aiaec_spec_freq.c` is `test_aiaec_freq.c` (`make test-aiaec-freq`) and now compares the PCM family with `_freq` behind an external `AcStft` pair and `AcIstft`; `test_aiaec_packet.c` does the same at N = 8; the bind-info dumper takes family 0 (PCM) or 1 (`_freq`); `link-only-aiaec` fails if an `aiaec_*_spec` symbol is exported.
- **`StreamWork`.** `aiaec_preprocess`, `aiaec_preprocess_freq` and `aiaec_front_preprocess` fill the shared `StreamWork` of `stream/common/stream_run.h`; `AiaecWork`, `AiaecTensorIn`, `AiaecTensorOut`, `AiaecAction`, `AIAEC_BYPASS`, `AIAEC_SUBMIT`, `AIAEC_MAX_INPUT_TENSORS` and `AIAEC_MAX_OUTPUT_TENSORS` are removed (no aliases): `STREAM_WORK_SUBMIT`, `STREAM_WORK_BYPASS`, `STREAM_WORK_MAX_INPUTS` (5), `STREAM_WORK_MAX_OUTPUTS` (3). Fields are the same; the output array has two more (zeroed) slots.
- **Run status.** `aiaec_postprocess{,_freq}(ctx, run_status, ...)` (parameter renamed from `infer_status`) passes the `stream_run.h` status to the streams, which close a failed run in the library: a dirty status or a refused result advances the state generation by one (both together count once), `STREAM_RUN_NOT_STARTED` leaves it (Align-ULCNet packet with an event frame: +1). `aiaec.h` states the rule in the words of `ainr.h`. A sample no longer tracks failures itself. New checks: `test_aiaec_block.c` / `test_aiaec_freq.c` add a run that never started and assert one generation step per dirty status, refused commit and both together; the Align-ULCNet block test follows `ulcnet_stream_*` driven directly, generation included; a bypass hop ignores any status, generation included.
- **Non-finite input.** New `AIAEC_ERR_NONFINITE_INPUT` (-10), appended to the status list. One scan after the kind / size / NULL gates and before any frame enters the stream: the four planes of the `_freq` family (both models, `N * n_bins` floats each) must be finite; Align-ULCNet's float `residual_in` and `far_in` must be finite and at most `2^112` (`AIAEC_PCM_LIMIT_BITS`, built like `DFN2_STREAM_BIN_LIMIT_BITS`). DeepVQE-S int16 input needs no scan. A refusal opens no block and moves nothing; a run offered the bad block between two good ones equals, bit for bit, a twin that never saw it (72 injections for DeepVQE-S, 144 for Align-ULCNet, first and last element of every plane). **Why a PCM bound:** `+-FLT_MAX` in every `_freq` plane keeps every tensor handed to the accelerator finite for both models (33 and 63 cases at the largest accepted magnitude, the block closed with a stand-in head also finite), so spectrum bins need nothing beyond finite; the Align-ULCNet PCM entry analyses its own input, and a constant `|sample|` of `2^121` (fft 256) / `2^120` (fft 512) already hands non-finite tensors to the accelerator, so it gets a bound with 8 bits of margin. `aiaec_front_preprocess()` finds a non-finite AEC residual after the AEC ran: reset the front and the ctx. New `test_aiaec_nonfinite.c` (`make test-aiaec-nonfinite`).
- **Docs.** `docs/html/models/{align-ulcnet,deepvqe-s}/index.html` (examples report `STREAM_RUN_NOT_STARTED` before `net_proc` and `STREAM_RUN_FAILED` after; input-check paragraph; pool tables without the `_spec` column), `docs/html/models/index.html` (AIAEC rows), `AIAEC/{Align_ULCNet,DeepVQE_S}/README.md`, `AIAEC/Align_ULCNet/reference/{alignment,deployment}.md`, `docs/maintenance/graph-wiring.md` and `docs/html/models/reference/onnx.html` (the `StreamWork` and two-family sentences only).

## [Unreleased] — 2026-11-16 — Board samples: `StreamWork`, run status and library-driven state sync, model-directory check, post-chain reset of `uvc_audio_live2a.c`; cross-model board docs

- **Eight AI3 samples** (`alg_ai3_dfn.c`, `alg_ai3_dfn_lite.c`, `alg_ai3_aec_dfn.c`, `alg_ai3_deepvqes.c`, `alg_ai3_alignulcnet.c`, `uvc_ai_nr_stream.c`, `uvc_audio_live2a.c`, `uvc_audio_live3.c`). `AinrWork` / `AiaecWork` become `StreamWork`, `AIAEC_SUBMIT` becomes `STREAM_WORK_SUBMIT`, `graph_io_fits()` takes `STREAM_WORK_MAX_INPUTS` / `STREAM_WORK_MAX_OUTPUTS`. A block whose `work.action` is not `STREAM_WORK_SUBMIT` is not run (every block is SUBMIT today).
- **Run status.** Every return before `vendor_ai3_net_proc` (block check, feature copy, state zeroing, `set_buf_*`) is `STREAM_RUN_NOT_STARTED`; a `net_proc` failure and a head readback failure are `STREAM_RUN_FAILED`; 0 only when every head was read back. `g_state_pending_recovery` and every write to it are gone. The one counter `g_state_gen_seen` (generation at the last zeroing) decides: the state is zeroed before a run when `work.state_generation` differs. `state_reset()` / `state_generation_sync()` return the flush result; a failed zeroing makes the block `STREAM_RUN_NOT_STARTED` and leaves the counter, so the next block zeroes again (six samples ignored a failed zeroing and ran on).
- **`alg_ai3_aec_dfn.c`.** Direct path: the host synthesis is cleared `queue_frames` hops after the AEC hard reset, with a per-hop shift register; the sample also takes the `--dfn-atten-lim` help text. Model-directory check: `--quant8` / `--quant16_8` and the file name stay; at start-up the sample refuses, before the model is opened, when the flag directory is not the directory of `g_dfn_info.graph_path` (exit 1; a lite build always refuses, a build without a graph path would refuse (not reachable: init refuses it first)). No SD-card layout change. `--quant8` can no longer run on any build: the default (16x16) and ALT2 (16x8) builds refuse it, and a build bound to the 8x8 model id has no catalog row, so init refuses it first.
- **`uvc_audio_live2a.c`.** A post-chain reset calls `ainr_bridge_reset_synthesis()` instead of `uvc_dfn_reset()`; DFN2, the resamplers and the FIFOs keep running. `UvcResPost.istft` (allocated and reset, never pushed) and `uvc_res_post_reset()` are removed. The failure paths still reset the whole chain.
- **Behaviour change.** After a failure before `net_proc` the device state is not zeroed again (the state was not touched); outputs of an unfailed run are unchanged. `alg_ai3_aec_dfn.c` differs at the AEC hard-reset hop e and e + 2 only (48 kHz direct path). `uvc_audio_live2a.c` keeps DFN2's state through a post-chain reset.
- **Docs.** `docs/reference/board/sequence.md` (state and generation paragraph), `docs/maintenance/graph-wiring.md` (run contract row, the one sample rule), `docs/maintenance/models.md` and `docs/html/models/index.html` (`stream_run.h` in the customer header set), `docs/reference/board/build.md`, `docs/maintenance/float-io-board-ab.md` (re-record the A reference); `check-public-headers` includes `stream_run.h`.

## [Unreleased] — 2026-11-16 — AEC 82ee6b6: the 4ch array's reference ring follows the estimator reach; an overflowing lane is refused; `max_delay_ms` gone from C

**Smaller 4ch pools, same output.** The AEC array instance sizes its `MATCHED` aligned-far ring like the single channel (the estimator's largest reportable delay plus one hop) instead of 1024 ms + 2 hops + 1. Every 4ch pool that holds a `MATCHED` array shrinks by 33,472 B at 16 kHz/256, 33,984 B at 16 kHz/512 and 100,896 B at 48 kHz/1024 (`delay_num_filters = 5`). Measured on the live build (`--print-mem-size` and the pool lines of `make -C pipelines/4ch test`): `4ch_static` Unit / Total 1,459,184 / 1,515,280 B (16k/256), 2,626,992 / 2,704,160 B (16k/512), 5,435,184 / 5,620,464 B (48k); `4ch_external_static` Front end / Total 96,112 / 723,616, 119,408 / 1,135,280, 235,632 / 2,497,152 B; `4ch_aec_bf_dfn_res` pool 1,683,088 B (16 kHz), 5,591,248 B (48 kHz), 1,523,536 B (`--ai-sample-rate 16000 --ai-fft-size 256`); `4ch_alignulcnet` unit pool 2,419,056 B (16 kHz) and 5,019,376 B (48 kHz). The `num_angles = 360` row of `beam-settings.md` is the measured 16k/256 Unit minus the same 33,472 B. Descriptor unchanged (`layout_version=4`, `build_flags_hash=0xc8cb8b58`). `make -C pipelines/4ch test` (`SIMD=1` / `0`) prints the same lines except these sizes.

**Refused lane.** The array now checks each lane like a single channel: an overflowing finite sample on any microphone (from about 1.9e19) or the reference makes `aec_process_linear()` return `AEC_ERR_DSP` with the array reset, in every residual mode including `STATE` (before: `STATE` never refused, and with `ESTIMATE` / `FULL` the refusal came from `aec_process_residual()` or not at all). After a refusal `aec_get_array()` is `AEC_ERR_SEQUENCE` and `aec_get_linear()` reads zeros. The 4ch code already handles `AEC_ERR_DSP` from `aec_process_linear()`; no code change here.

- Python unchanged: `pipelines/4ch/pipeline.py` still sizes its ring from `max_delay_ms` and agrees at its defaults. Map: PL-MCAB-03 note and gap D-46 (the AEC map marks `array_dims` partial, D-17); the map grammar cannot spell a `partial` row whose C function lives in the AEC library, so the row stays Python-only.
- Pins: `lib/aec.pin` and the `lib/aec` gitlink to AEC `82ee6b6`; `lib/nr.pin` and the `lib/nr` gitlink to NR `53b3834` (NR's Python parity tool drops the obsolete S-10 finding; no C change).
- Docs: the pool figures in `docs/reference/array/{beam-settings,integration}.md`, `docs/reference/board/memory.md`, `pipelines/reference/call-flow.md`, `pipelines/4ch/{aec_bf_dfn_res,alignulcnet}/README.md` and `docs/html/pipelines/4ch/{aec-nr-res,align-ulcnet,dfn2}/`; the ring formula in `docs/reference/array/{config,errors}.md`, `docs/html/pipelines/4ch/aec-nr-res/reference/configuration.html`, `docs/html/pipelines/reference/dfn2/configuration.html` and `pipelines/4ch/README.md`.

## [Unreleased] — 2026-11-16 — DFN2: the first `mask_lookahead` hops run no accelerator job (`StreamWork.action = STREAM_WORK_BYPASS`)

Output (PCM and spectra) is bit-identical; what moves is the number of accelerator runs and the state generation. This replaces the rule of the DFN2 failure-recovery entry above that a warm-up frame steps the generation once.

- **`lib/ainr/dfn2/dfn2_prepost`.** With `frames_per_call` 1 and `mask_lookahead` L >= 1, the first L hops after init or reset advance the enhance clock with `heads_valid = 0`, open no frame and return 0 from `pre_process_freq{,_dual}` (the N > 1 flow already did this for the hops between two calls). `PP_IN_WARMUP` and both warm-up branches are gone: a warm-up frame no longer zeroes the registered state and no longer steps the generation by one (the old rule cost L accelerator runs and L generation steps per init or reset). `frame_fail(p, run_started)` is now `if (run_started) zero; bypass`. No carve, layout or pool change (`DFN2_PREPOST_CARVE_VERSION`, `DFN2_STREAM_LAYOUT_VERSION`, every pool size and `build_flags_hash` are as before).
- **`stream/ainr/dfn2`.** A hop with no job (`DFN2HopIO.job_ready` 0) is accepted for N == 1 too; its views are not set. `dfn2_stream_set_atten_lim` is refused while a hop awaits its post_process (the old guard was "a prepost frame is open", which a hop without a job does not have).
- **`stream/ainr` product.** `ainr_preprocess{,_freq}` return an all-zero `StreamWork` (`action = STREAM_WORK_BYPASS`, `state_generation` 0) for such a block, with `AINR_OK`; `ainr_postprocess*` is still called once and ignores the status. Board samples skip the accelerator on BYPASS and post `STREAM_RUN_NOT_STARTED` (the five live board samples already do).
- **`pipelines/dfn` (desktop API change).** `dfn_res_stage_begin()` returns 1 (model frame open), 0 (accepted, no model frame: the first L hops; call `dfn_res_stage_end()` all the same, its status is ignored) or -1. `dfn_res_stage_end()` runs finish/bypass/fail and the commit/skip counters only for a hop that had a frame; `frames_in` still counts every hop. `dfn_res_stage_set_atten_lim()` is refused between begin and end. `dfn_hop_nn_begin()` returns 0 for such a hop on the direct path (the hop is closed by `dfn_hop_end()`); the bridge paths (`dfn_hop`, `dfn_rate_bridge_nn_begin`) close the frame inside `nn_begin` and fetch the next one. `dfn_hop_open_frame()` no longer fails on a hop without a job. `dfn_res_stage_process()` skips the model for it. No struct size, pool size or layout version changes.
- **Behaviour change.** The state generation no longer moves for the first L hops after init/reset (before: +1 per hop). A sample that seeds `g_state_gen_seen` with `ainr_state_generation*()` right after init must have zeroed the accelerator state first, because the first SUBMIT block now carries the init generation and nothing bumps it before the first real run (the old warm-up job did). The five live samples do.
- **Tests.** New and rewritten for the no-job flow: `stream/ainr/dfn2/tests` (`smoke`, `check_erb_file`, `verify_vs_sample`, `fail_table`, `variant_consumer`), `stream/ainr/tests` (`test_ainr_state_generation`, `test_ainr_graph_io`, `link_only_ainr`), `pipelines/dfn/tests` (`test_dfn_hop`, `test_dfn_res_stage`, `test_dfn_rate_bridge`, `test_dfn_stream_equivalence`), `pipelines/{mono,4ch}` generation pins, `stream/ainr/tests/dump_work.c` (the work-dumper of `test_dfn2_port_order.py` prints the first block with a job), `AINR/tests` (`test_nn_chains.c`, `dfn2_runtime_grid_chain.c`, `test_dfn2_prepost_c.py`, `test_dfn2_frames_per_call_c.py`, `test_dfn2_runtime_grid.py`, `test_dfn2_mask_only_c.py`, `test_python_c_prepost_parity.py`). The recorded `test_nn_chains` digest `a7632b4049c9235c:25de8df4df69d614` and the `dfn2_runtime_grid_chain` digests are unchanged.
- **Docs.** `lib/ainr/dfn2/dfn2_prepost.h`, `stream/ainr/dfn2/dfn2_stream.h`, `stream/ainr/ainr.h`, `pipelines/dfn/{dfn_hop.h,dfn_res_stage.h,dfn_rate_bridge.h}`, the DFN2 pages under `docs/html/{models,pipelines,start}`, `docs/reference/board/*`, `docs/maintenance/{models,pipeline}.md`, `AINR/DeepFilterNet2/{README.md,reference}`, `pipelines/{dfn,mono/aec_dfn_res,4ch/aec_bf_dfn_res}/README.md`, `pipelines/reference/call-flow.md`, `docs/maintenance/models.md` (include roots), `docs/c_python_function_map.md` (AINR-DFN2-16, D-42: C and the Python twin now agree on the first L hops).
- Python twins (`dfn2_stage.py`, `dfn2_rate_adapter.py`) are untouched: the twin already evaluates no heads for the first M frames and has no state generation.

## [Unreleased] — 2026-11-16 — `dfn2_model_run_frame`: `infer()` returning `STREAM_RUN_NOT_STARTED` is a run that did not start

- `dfn2_model_run_frame` closes the frame with `frame_fail(p, run_status != STREAM_RUN_NOT_STARTED)` for a nonzero `infer()` result; before, every nonzero result was `frame_fail(p, 1)`. `STREAM_RUN_NOT_STARTED` leaves the registered state and the generation alone; every other nonzero value still zeroes both and advances the generation. The same status `dfn_hop`, `dfn_res_stage` and `dfn_rate_bridge` already hand to the stream.
- `dfn2_prepost.c` includes `stream_run.h`: `AINR/Makefile`, `pipelines/Makefile` and `pipelines/4ch/Makefile` add `stream/common` to the include path of the objects that compile it. Test: `AINR/tests/test_dfn2_prepost_c.py` (stand-in mode returning `STREAM_RUN_NOT_STARTED`).

## [Unreleased] — 2026-11-16 — build and test fixes

- `stream/ainr/dfn2/Makefile` `test-variant-guard` asks `make` for the archive path and the compile flags in separate calls; under `make -j` the two goals of one call could print in either order and the target took an empty archive path (3 failures in 25 `make -j3` runs before, 0 in 50 after).
- `stream/ainr/tests/test_ainr_freq_nonfinite.c` reads the tensor views of a `DFN2HopIO` only when `job_ready` is set (it read an uninitialised struct on a hop without a job).

## [Unreleased] — 2026-11-16 — AINR: no tail, no trim; `AinrInfo` is `graph_path` and `queue_frames`

- **`stream/ainr` (API change).** `AinrInfo.trim_samples` and `AinrInfo.tail_zero_blocks` are removed. The stream has no tail: the output trails the input by `queue_frames + 1` model hops (`_freq`: after the caller's own ISTFT), and nothing is trimmed, padded or flushed. `AinrInfo` is 16 bytes on LP64 (was 24), so the library and every caller that includes `ainr.h` are rebuilt together; pool sizes, layout versions and the pool goldens do not change. Per-hop values do not change: an exact-shift scan of both families on every catalog row (16 kHz/256 N=1 and N=8, 48 kHz/1024 full, 48 kHz/1024 lite; SIMD 0 and 1) finds the one shift that the removed `trim_samples` used to report (128, 1024, 1536, 1024), with identical output hashes before and after.
- **Board samples.** A sample that read the two fields (`alg_ai3_dfn.c`, `alg_ai3_dfn_lite.c`) must drop its trim and drain windows; its output is then `floor(S / hop_in)` full blocks starting with the latency in samples, `(queue_frames + 1) * (fft / 2)`.
- **Texts.** The Tail paragraph and usage lines of `ainr.h`, the flush texts of `dfn2_stream.h`, `dfn2_prepost.h`, `dfn2_model_io.h` and `dfn2_process.h`, the DFN2 pages and the board and maintenance docs state the latency instead of a drain. `pipelines/dfn/dfn2_stage.py` no longer calls its flush "the same tail discipline the C class documents" (docstring only).
- **Tests.** `test_ainr_tail.c` is a latency test: a pulse at a fixed input index comes back at index + the literal latency of the catalog row through both families, no drain. `dump_work` prints `info queue <int>`; `test_dfn2_port_order.py` reads it.

## [Unreleased] — 2026-11-16 — `4ch_aec_bf_dfn_res`: the post-reset synthesis clear is one function, and its test runs `main.c`

- `pipelines/dfn/dfn_hop.h` gains `DfnResetFollow` with `dfn_reset_follow_init()`, `dfn_reset_follow_clear()` and `dfn_reset_follow()`: the per-hop shift register that clears the direct path's `AcIstft` `dfn_hop_lookahead_frames()` hops after an upstream hard-reset event, and the bridge's host-grid clear on the hop of the event. `4ch_aec_bf_dfn_res/main.c` calls it from `dfn_build`, `dfn_reset` and `dfn_form` (behaviour unchanged).
- `tests/test_4ch_dfn.c` calls the same functions in its composition and compiles `main.c` into itself (its `main` renamed) with a counter around `ac_istft_reset`: two rows drive `main.c`'s own sample (event on hop 81: one clear, on hop 83; event on hop 40 then a reset of every stage on hop 41: only the reset's own clear). `pipelines/4ch/Makefile`: `test_4ch_dfn` links `array_sample_driver.o` and depends on `main.c`.

## [Unreleased] — 2026-11-16 — `AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED` says the rule

- The status text, the enum comment and the DFN2 page state `frames_per_call > 1 requires mask_lookahead = 0`; the refusal and the error number (-5) are unchanged.

## [Unreleased] — 2026-11-16 — DFN3: the first `mask_lookahead` hops run no accelerator job

`dfn3_prepost_pre_process_freq()` (and `_dual`) now returns 0 for the first `mask_lookahead` hops after init or reset, the hops whose graph window is not complete: the enhance clock advances without heads, no frame is opened, and `frame_open()` / `dfn3_model_run_frame()` / `frame_finish()` / `frame_bypass()` return -1 for them. From hop `mask_lookahead` on it returns 1 as before. Callers open and run a frame only when the return value is 1; `post_process_freq()` is called on every hop and gives a zero spectrum with `valid` 0 until the first output.

- Output spectra, `valid` flags and the recurrent state every real job starts from are bit-identical to the previous behaviour (the previous warm-up job's heads were discarded and its state zeroed).
- The accelerator runs `mask_lookahead` fewer times per init or reset (one fewer for the shipped 1/1 pair).
- `dfn3_prepost_state_generation()` no longer advances on those hops: it is 1 after init, plus 1 per reset and per refused `frame_finish()`. A runtime that reloads quantized zeros when the generation changes does that `mask_lookahead` times fewer per start.
- No pool size, carve, layout or descriptor change (`DFN3_PREPOST_CARVE_VERSION` and `DFN3_MODEL_IO_LAYOUT_VERSION` stay 9; `dfn3_prepost` 90,384 B, hash `d496d34e`).
- `dfn3_model_io.h`, `dfn3_prepost.h` and `dfn3_process.h` state when each result is ready (heads[t] with the push of frame t + `mask_lookahead`; enhanced frame t from hop t + `output_delay`; audio `output_delay` + 1 hops behind the input through an AcStft / AcIstft pair) instead of a flush procedure.
- `dfn3_prepost_set_erb_matrices()` and `dfn3_prepost_set_atten_lim()` are accepted on those hops (they were refused while the hop's frame was open). The registered state is no longer zeroed at the end of those hops, so a write by the caller between init/reset and the first job persists into it.
- `AINR/tests/test_python_c_prepost_parity.py` expects no frame for those hops on DFN3 as on DFN2.

## [Unreleased] — 2026-11-16 — Align-ULCNet stream: the alignment-edge memory is `prev_solid` alone

- `UlcnetStream` no longer keeps a first-hop flag. `prev_solid` is 1 after `ulcnet_stream_init` and after `ulcnet_stream_reset`, and every entry (one hop, packet, PCM and spectrum) applies `changed || (!prev_solid && solid)`. The first hop after init or reset restarts only on `changed`, as before; every later hop behaves as before. Outputs, generations, accelerator jobs, pool bytes and `build_flags_hash` are unchanged, and `ULCNET_STREAM_LAYOUT_VERSION` stays 5.
- `ULCNET_STREAM_LAYOUT_VERSION` = 5 is now listed on the integration page (layout-version table) and checked against the header by `pipelines/tests/test_doc_layout_versions.py`.
- `export_onnx.py`: the `c_frontend_postprocess` metadata value is now `lib/aiaec/ulcnet/ulcnet_model_io.c` (was `lib/aiaec/ulcnet/ulcnet_process.c`, deleted). Existing exports keep the old string and need not be re-exported; nothing in this repository reads it.
- Text only: `ulcnet_prepost.h` and `ulcnet_stream.h` comments shortened (preprocessed output identical), the manual no longer lists a first-hop flag, `test_ulcnet_prepost_c.py` / `test_deepvqe_prepost_c.py` state that the hop-0 checks depend on the synthesis mode.

## [Unreleased] — 2026-11-16 — DeepVQE-S takes N frames per call: packet entries, one path for every N

- **DeepVQE-S takes N frames per call.** `deepvqe_stream_pre_process_packet` / `post_process_packet` and the `_freq` pair: pre analyses the N hops, hands each frame to the prepost at once and opens the batch; post closes it (commit on a clean run, otherwise the library's fail-closed close) and writes the batch's N frames, oldest first, as PCM or spectra. A call's output is its own N frames: no queue across calls, no start-up case. The one-hop entries (`deepvqe_stream_pre_process`, `post_process`, `_freq`) are the N = 1 form of the same sequence and return -1, touching nothing, on a stream with N > 1.
- **`aiaec` has one path for every N.** The per-hop stash and drain loop (and the stash in the pool) is gone; `aiaec_preprocess{,_freq}` / `aiaec_postprocess{,_freq}` call the packet entries. Output at N > 1 is the old output moved N - 1 frames earlier (the old first N - 1 frames of each reset segment were zero); N = 1 is bit-identical, the shipped 16 kHz / 512 N = 1 row included.
- **A failed batch silences its own block.** The failed job's N frames are zero spectra and they are the failed block's own N frames: its first hop keeps the half window of the previous frame, the other N - 1 hops are exactly zero, the next block's first hop is half again. (Before, the same frames came out one block later, N - 1 hops into the next block.)
- **Latency follows the behaviour:** the DeepVQE-S output trails the input by one model hop at every N (was `N * hop`, with `N - 1` hops of queue). See the `AiaecInfo` entry below for the fields.
- **Pools change at every N.** `DEEPVQE_STREAM_LAYOUT_VERSION` 4 -> 5: the pooled stream lost two fields and the product pool lost its stash. Host pool bytes, PCM / `_freq`: 16k/256 N=1 64192 -> 64160 / 18368 -> 18336, N=8 167024 -> 165200 / 126640 -> 119376; 16k/512 N=1 124096 -> 124064 / 35776 -> 35744. A build that compiles both models also gives the Align-ULCNet pools 16 bytes less (`_freq` 7920 -> 7904, 15088 -> 15072; PCM 53744 -> 53728, 125904 -> 125888, 103408 -> 103392).
- **`aiaec.h` non-finite refusal text:** `AIAEC_ERR_NONFINITE_INPUT` also covers the DeepVQE-S `_freq` planes (NaN or Inf in any plane of a `_freq` input, either model) and names `AIAEC_PCM_LIMIT_BITS` for the Align-ULCNet float PCM input; DeepVQE-S int16 PCM needs no scan. Comment only.
- **`result_ready` and the `open` flag are gone** from the stream; `deepvqe_prepost.h` documents the order pre x N, frame_open / run / finish, post x N.

## [Unreleased] — 2026-11-16 — DeepVQE-S: a non-finite enhanced spectrum is refused

- `deepvqe_prepost_frame_finish` also scans the N enhanced spectra after the convolution. A finite input near FLT_MAX with finite taps (0.5 on every tap) used to hand an Inf spectrum to `post_process_freq`; it is now refused like a non-finite tap: -1, the batch stays open for `frame_bypass`, registered state zeroed, CCM ring emptied, generation +1 once. The streams close the batch with the bypass, so the block is silent and the next block is a cold start. Same rule Align-ULCNet already has.

## [Unreleased] — 2026-11-16 — AIAEC: no trim, no tail, `AiaecInfo` is the graph path

- **`AiaecInfo` carries `graph_path` only.** `trim_samples`, `queue_frames` and `tail_zero_blocks` are removed (`sizeof(AiaecInfo)` 24 -> 8 bytes on LP64): rebuild the library and every sample that includes `aiaec.h` together. The pool, the layout versions and the stream outputs do not change.
- **The AIAEC stream is a stream.** Nothing is trimmed, flushed or zero-padded by the library: the output trails the input by one model hop (`fft / 2` samples) for both models at every N. That is the value the three fields reported on every catalog row (trim = fft / 2, queue 0, tail 1), and an identity accelerator shows exactly that shift. An application that processes a file pushes its whole blocks and keeps everything; the output starts with the one-hop latency and the last model hop of input is still in the synthesis when the input ends.
- `aiaec.h` usage text, the Align-ULCNet and DeepVQE-S manual pages (latency section, init text, struct comment) and the DeepVQE-S core page follow; the `_freq` family's latency is the same one model hop after the caller's own ISTFT.
- The board samples (outside this repository) must drop their trim and tail windows in the same drop as the library.

## [Unreleased] — 2026-11-16 — AEC d38248e and NR cbbdc76: the refused-hop thresholds are scene-dependent, `aec_wav` handles the refusal, the NR `xi/(1+xi)` change is sized

No library behaviour changes. The AEC `aec_wav` example changes (below).

- **Refused hop, thresholds.** The overflow onsets quoted for a refused hop (microphone about 1.9e19, reference about 1.3e19, or 5.6e14 with `ESTIMATE` / `FULL`; array reference 3.8e19 with `STATE`, 1.1e15 with `ESTIMATE` / `FULL`) came from single scenes. They are now stated as scene-dependent: measured with the balanced level, a 0.12 s delay, at 16k/256, 16k/512 and 48k/1024, a microphone sample is refused from about 1.9e19 on every lane, a reference sample from about 1.3e19 with `OFF` / `STATE` and from about 3e13 to 6e14 with `ESTIMATE` / `FULL` (single channel and array). PCM16 input cannot reach any of them. Older entries that quote the other numbers are history of those scenes.
- **Refused hop, timing.** A microphone sample is refused on the hop that carried it; a reference sample when it reaches the filters through the alignment ring, a number of hops later that depends on the delay in use (measured: 7 to 22 hops after the call that carried it at a 0.12 s delay, over the three grids and the sample values tried).
- **Refused hop, calls afterwards.** `aec_get_residual()` is `AEC_ERR_SEQUENCE` until the next hop; `aec_apply_gain()` / `aec_apply_cng()` are `AEC_ERR_SEQUENCE` with `FULL` and `AEC_ERR_MODE` on any hop with `STATE` / `ESTIMATE`.
- **`max_delay_ms`.** The AEC user documents no longer carry notes about the deleted `max_delay_ms` field.
- **`aec_wav`.** `aec_process()`'s return is handled: `AEC_ERR_DSP` writes the hop as silence and continues (with a count at the end); any other error prints a message and stops with exit code 5.
- **NR.** The `xi/(1+xi)` entry stated "at most a few ulp per bin". Measured on 24 VCTK+DEMAND files: 2.5-12.7 % of the gain bins differ, the 99th percentile of the differing bins is 6 ulp, the tail reaches 147 ulp, float waveform max |d| 1.19e-7, difference energy -143.1 dB or lower. `pyc_parity_nr.py` carries no leftover of the removed finding.

## [Unreleased] — 2026-11-15 — NR without `gain_prev`, without its two removed build flags and with `xi/(1+xi)` from the SPP stage; NR pool numbers re-measured

Pins move to the NR and audiolib commits that remove `AnrDenoiser.gain_prev` (output unchanged), remove the build flags `USE_FAST_GAIN_SMOOTHING` / `USE_SHARED_XI_RATIO` (shipping output unchanged; `DEBUG=1` now runs the shipping gain path) and hand the gain stage `xi/(1+xi)` from the SPP stage (NR output changes by a few ulp in about 5-6 % of the gain bins; see `NR/CHANGELOG.md`). No Audio_ALG source changes apart from the test driver texts below.

- **NR pool numbers in the docs.** `anr_get_mem_size()` 8k/256 27,696 to 26,624 B, 16k/256 44,256 to 43,184, 16k/512 54,320 to 52,224, 48k/1024 140,416 to 136,272. Re-measured with `aec_nr_pipeline_static --print-mem-size`, `4ch_static --print-mem-size` and `4ch_external_static --print-mem-size` (`4ch_external_static` Total 758,160 to 757,088, 1,171,360 to 1,169,264, 2,602,192 to 2,598,048): mono total 8k 251,504 to 250,432 B, 16k/256 296,432 to 295,360, 16k/512 430,384 to 428,288, 48k/1024 963,456 to 959,312; `4ch_static` Total 16k/256 1,549,824 to 1,548,752, 16k/512 2,740,240 to 2,738,144, 48k/1024 5,725,504 to 5,721,360 (these start from the AEC layout-4 figures, 16 B below the AEC lane base; post stage 57,168 to 56,096, 79,264 to 77,168, 189,424 to 185,280, from the printed NR, synthesis and buffer lines). Derived, not re-measured (each holds one NR instance, so it moves by the NR delta of its grid): the delay-mode table (16k/256) MATCHED n=5 295,360, n=1 250,304, FIXED 239,920, EXTERNAL_ALIGNED 233,008 (the minus column is unchanged) and the `num_angles = 360` row 4,551,936 to 4,550,864. `--no-nr` (252,176 B) and `--aec-only` (239,760 B) are unchanged. No layout or descriptor version covers the NR pool, so none moves.
- **`pyc_parity_pipelines.sh` / `pyc_parity.py`.** The build-flag assertion lists only `USE_OPTIMIZED_E1` (the variable is `nr_flags`, the failure token `FAIL_nr_flag_`), and the layer texts say so. Measured before and after each NR step (91 PASS, 1 SKIP throughout; frozen tolerances unchanged): the `gain_prev` removal leaves every row's numbers identical; the flag removal (Layer 1 now runs the shipping gain clamp) moves the 16k/512 4-channel NR rows from 2.2e-6 to 3.5e-6 and the mono 16k/512 / 48k NR glue rows from 3.5e-6 / 3.3e-6 to 3.6e-6 / 3.5e-6; the `xi/(1+xi)` change brings the mono glue rows back to 3.5e-6 / 3.3e-6. The Layer-1 worst stays 8.9e-6 (`a.nr4.16k256`, tolerance 2e-5) and `d.mono.16k256.nr_glue` stays 8.3e-6; the Layer-2 worst stays 2.8e-2 (`l2.res4.48k1024.mild.replay`, tolerance 5e-2).
- **`docs/c_python_function_map.md`.** `AC-SIMD-08` loses `skr_gain_prepare_elem/_scalar/_neon` and its text follows the NR change (two `ANR_VECTOR` wrappers, `#if ANR_VECTOR`); `simd_nr.h` counts 16 functions, AC 295 to 292, TOTAL 874 to 871 (C-only 698 to 695).

## [Unreleased] — 2026-11-15 — AEC 9f7d97d: a refused single-channel hop (`AEC_ERR_DSP`) is handled by every mono caller; `AecResidualView.delay_changed` / `delay_locked` gone; AEC layout 4

- **AEC pin.** `lib/aec.pin` and the `lib/aec` gitlink move from `13e804a` to `921937e` (three AEC commits: the two `AecResidualView` delay fields removed with `AEC_LAYOUT_VERSION` 4, the single-channel refused hop `9f7d97d`, and a function-map path). NR and audiolib are unchanged.
- **Refused hop in the mono pipelines.** A single-channel AEC call now returns `AEC_ERR_DSP` when a finite input sample overflows the hop (measured: microphone from about 1.9e19, reference from about 1.3e19, or 5.6e14 with `ESTIMATE` / `FULL`); the AEC has already reset itself. Every caller already treated any non-`AEC_OK` as a refused hop: `mono/aec_nr_res/main.c` (zero hop, `rig_reset()`: AEC, denoiser, synthesis, serial) is unchanged; `mono/aec_dfn_res/main.c`, `mono/alignulcnet/main.c` and `mono/aec_nr_res/example_board_adapter.c` now write a zero hop, restart every stage and go on to the next hop, and still fail their smoke run on any refusal (their synthetic input is benign); `stream/aiaec/front` already resets the AEC and returns `AIAEC_FRONT_ERR_AEC` (its header comment now says a refused hop is one of the causes). The test rigs (`conv_chain_mono.h`, `test_readme_flow.c`, `pyc_drv.c`, ...) feed normal input and keep returning -1.
- **Tests.** `test_mono_chain` gains `test_refused_hop_restarts_chain` on the four grids: one microphone sample of 3e38 on hop 120, handled as `main.c` does, then 200 hops bit-identical to a fresh chain started on hop 121 (mutations: without the caller's `rig_reset()` the NR state carries over and the rows fail; built against the AEC of `13e804a` the poisoned hop is not refused and all four rows fail). `test_front_refusal` adds a 3e38 sample in hop 0 and hop 1 of a block next to the NaN cases (the next block equals a fresh front's).
- **Removed fields.** `pipelines/4ch/tests/test_array_sample_beam.c` `same_frame()` drops the two removed `AecResidualView` terms; `pipelines/reference/call-flow.md` and the DFN2 integration page point at `AecLinearView.delay_state` / `AecArrayView.delay`. No pipeline code read them.
- **Pool numbers (AEC layout 4: every array pool 16 B smaller).** Re-measured with `4ch_static --print-mem-size`, `4ch_external_static --print-mem-size` and the `4ch_aec_bf_dfn_res` banner: array FULL 16k/256 700,480 to 700,464 B (STATE 594,944 to 594,928), unit 1,492,672 to 1,492,656, `4ch_static` total 1,549,840 to 1,549,824, `4ch_external_static` total 758,176 to 758,160 and front end 129,600 to 129,584, `4ch_aec_bf_dfn_res` pool 1,716,576 to 1,716,560 (48 kHz 5,692,160 to 5,692,144; its 16 kHz core with the 16000/256 model 1,557,024 to 1,557,008); every other figure that contains an array pool moves by the same 16 B (`docs/reference/array/{beam-settings,integration}.md`, `docs/reference/board/memory.md`, `pipelines/reference/call-flow.md`, `pipelines/4ch/{aec_bf_dfn_res,alignulcnet}/README.md`, the 4ch HTML pages). `AEC_LAYOUT_VERSION` 3 to 4 in the docs that name it. Mono pools are unchanged.
- Docs: `docs/reference/mono/lifecycle.md` (the return-code row and the recover step name `AEC_ERR_DSP`).

## [Unreleased] — 2026-11-15 — `dfn_hop` refuses more than one frame per call; `check-public-headers`

No output change: every DFN2 example and test runs a one-frame-per-call grid, where both helpers return the same value as before. No library ABI, layout or pool change.

- **`pipelines/dfn/dfn_hop.h` (example header, no ABI).** `dfn_hop_lookahead_frames()` returns `dfn2_prepost_latency_frames()` (L + D + N - 1; L + D on the grids the hop accepts), so the two formulas are one. `dfn_hop_init_direct()` and `dfn_hop_init_bridge()` return `int`: 0, or -1 with the hop untouched for a NULL hop, grid or model or a grid of more than one frame per call (the direct path's one-frame-per-hop rule and the delay above assume N == 1). `dfn_hop_pool_build()` passes the refusal on; its plan already refused such a grid, so that path is defensive. A caller of the two inits that ignored their (void) result needs no change.
- **`make -C stream/common check-public-headers`** (in `make -C stream/common test`; `tests/check_public_headers.sh`). The customer header set (`ainr.h`, `ainr_bridge.h`, `aiaec.h`, `aiaec_front.h`, `stream_mem.h`, `stream_bind.h`, `graph_io.h` and the three `graph_io_*.h`) compiles as one C99 `-Wpedantic -Werror` unit and as C++11, with only the documented include roots, for DFN2 full and `-DDFN2_MASK_ONLY`. Every `ainr_` / `aiaec_` / `stream_` / `graph_io_` function the unit declares (90) is referenced; the C++ object's external references must equal the C object's and none may be mangled. A positive control (a declaration outside `extern "C"`) proves the scan fires. `check-graph-io-cxx` stays and runs first.
- **`deepvqe_prepost.c`.** The local byte counts named `bins` / `n_bins` (next to the bin count `p->n_bins`) are `frame_bytes` / `call_bytes`; the object files are byte-identical (`-O0` and `-O2`).
- **Tests and mutation evidence.** `test_dfn_hop` has `test_frames_per_call_guard` (the helper at N = 1 and N = 3, NULL, both inits refusing N = 3 without writing the hop and NULL arguments, accepting N = 1). Six mutations of `dfn_hop.h` / `dfn_hop_pool.h`: the helper back to L + D, either init without the N check, the direct init clearing the hop before the check, the direct init without its NULL-grid check are killed; `dfn_hop_pool_build()` ignoring the direct init's result survives (unreachable: the plan refuses N != 1 first). Seven mutations of the customer headers (the `extern "C"` close moved above the last prototype of `ainr.h`, `aiaec.h`, `ainr_bridge.h`; an include outside the documented roots; a C99 pedantic error; a C++-only error in `aiaec_front.h`; an error only under `-DDFN2_MASK_ONLY`) are all killed, the first three by the mangling scan.
- **Docs.** `pipelines/dfn/README.md`, `c_python_function_map.md` PL-DFN-17, `docs/html/pipelines/reference/dfn2/configuration.html`, `docs/maintenance/dependencies.md`, `docs/maintenance/models.md` §13 and `docs/html/models/index.html`.

## [Unreleased] — 2026-11-15 — Header contracts and manual pages that were missing or stale

Text only: every changed C file preprocesses identically (`cc -E -P`, plain and `-DDFN2_MASK_ONLY`); no layout, ABI or output change.

- **`ainr.h` / `aiaec.h`.** Preprocess NaN-fills the head outputs, so a head left unwritten counts as non-finite; all-zero bytes are the state's quantized zero because the port check refuses a fixed-point state port with `zero_point` != 0; `graph_path` is a static string; a context is used from one thread at a time. `ainr.h` also states that the PCM family is int16 in and out (a float caller pays about 1 LSB per side), and the restart rule of a direct (non-bridge) `_freq` caller: an upstream restart at block t belongs to the frame delivered at block t + (`queue_frames` - (N - 1)); clear the caller's synthesis just before pushing that frame.
- **`dfn2_prepost.h`.** Both `pre_process_freq` entries leave the bins unchecked: each must be finite with |bin| <= 2^63 at model scale (the `stream/ainr` entries check this).
- **DeepVQE-S.** `DeepVqePrepostMemReq.build_flags_hash` and the pool hash comment name N; the member comments of the taps and the planar mic/far buffers give their N-frame sizes; `c_python_function_map.md` AIAEC-DV-08 has the `frames_per_call` factor. `core.html`: hop 0 emits a full hop (`AC_OLA_ACCUMULATE`; it covers the half window before the stream start and `trim_samples` drops it), the dead `== 0` branch of the code sample is gone, and the D = 32 pools are 35,520 B (16 kHz / 512) and 70,336 B (48 kHz / 1024).
- **Align-ULCNet.** `deployment.md` synthesises with `AC_OLA_ACCUMULATE` (every push emits a hop, hop 0 included), points at `aiaec.h` for the framing, and says that with in-place state a board zeroes the state before the next run after any failed run or readback. `UlcnetPrepostMemReq.layout_version` reads 18, or 19 for a packet.
- **DFN2.** The model page has a build-and-link section (`libainr.a`, `libdfn2_stream.a`, `libaudio_common.a`; `-Istream/ainr -Istream/common -I$AUDIOLIB/common`; lite with `MASK_ONLY=1`; the bridge archive and its include roots), compiled and linked for full, lite and the bridge. `graph-wiring.md` says the library clamps `df_alpha` at the mix.
- **`mono_aec_dfn_res` descriptor comment.** It no longer says a product fills the descriptor from the model metadata: a host that reads that metadata does; the board samples check ports only, which do not depend on the lookahead pair.

## [Unreleased] — 2026-11-15 — The Align-ULCNet examples name the refused delay depth

`mono_alignulcnet` and `4ch_alignulcnet` built with an out-of-range `-DULCNET_EXAMPLE_DELAY_FRAMES` used to exit 1 with no message. They now print one stderr line, e.g. `mono_alignulcnet: grid refused (-4): delay depth 1, accepted range [2, 64] (-DULCNET_EXAMPLE_DELAY_FRAMES)`; the exit code stays 1 and nothing else changes. `make -C pipelines test` and `make -C pipelines/4ch test` build each example a second time at D = 1 (`*_alignulcnet_d1`, not in `all` or the published artifacts) and `scripts/test_delay_depth_refusal.sh` checks exit 1, empty stdout and the exact stderr line.

## [Unreleased] — 2026-11-15 — Docs and header comments that contradicted the code

Text only: every changed C file preprocesses identically (`cc -E -P`, plain and `-DDFN2_MASK_ONLY`); no layout, ABI or output change.

- **Build lines.** The DeepVQE-S and Align-ULCNet manual `cc` lines (three) now add `-Istream/common -I$AUDIOLIB/common`; without them `aiaec.h` -> `stream_mem.h` -> `utility/mem_align.h` does not resolve. `ainr.h`, `aiaec.h` and `ainr_bridge.h` name their include roots. The customer header set is listed in `docs/maintenance/models.md` §13 and `docs/html/models/index.html`.
- **Board checks.** The DeepVQE-S and Align-ULCNet pages and `onnx.html` no longer say the board validates the layout version or `c_descriptor`: board samples check port counts, shapes, formats and pairing by position; the version is pinned at export and by host tests; the descriptor comment of both Align-ULCNet `main.c` says "a host that reads the exported model metadata". The Python docstrings with the same claim (`AIAEC/_streaming_export.py` `_deepvqe_c_descriptor`, `AIAEC/tests/test_deepvqe_prepost_c.py` ~2283) are unchanged.
- **DFN2.** The scale is `dfn2_stft_scale(n_fft)` / `dfn2_stft_unscale(n_fft)`, not a `DFN2Grid` field (headers, `core.html`, `grid.html`, `dfn_rate_bridge.h`); the attenuation limit is `|atten_lim_db|` dB, sign ignored, as `inference.py`; `df_alpha` is clamped to [0, 1] before use; `dfn_hop_lookahead_frames()` is L + D, valid for N == 1 only.
- **Recovery rule.** `ainr.h`, `aiaec.h`, `dfn2_stream.h`, `deepvqe_stream.h`, `ulcnet_stream.h` and `docs/reference/board/sequence.md` state one rule, in the same words: a failed run or readback (status != 0) leaves the generation unchanged, so zero the accelerator's state before its next run.
- **`DeepVqeStreamConfig`.** The state-registration mode is in the stream's hash (it was documented as not).
- **Align-ULCNet enhanced-spectrum check.** The Board-impact note of the "enhanced spectrum that is not finite" entry, `configuration.html` and `sequence.md` now say that the generation also advances on a finite head whose enhanced spectrum is not finite, and that with a NaN error bin the bypassed output still carries that bin.

## [Unreleased] — 2026-11-15 — Align-ULCNet enhanced-spectrum check: rows for the last bin and for packet frame 0

`stream/aiaec/tests/test_aiaec_packet.c` `section_model_io_enhanced_check` now also puts the overflowing mask bin alone in frame 0 of a packet, in the last bin of every packet frame (the event frame's is accepted: it is neither written nor scanned) and in the last bin of a one-frame call, the bin the NEON scan handles in its scalar tail. Three mutants of `ulcnet_model_io.c` that every test survived are now caught at SIMD=1 and 0: the one-frame scan skipping the last bin, the packet scan skipping frame 0, and the packet scan skipping each frame's last bin. No library change.

- `docs/c_python_function_map.md` AIAEC-UL-02 is `partial` with the new gap D-45 (the check is C only; `export_onnx.host_back_end` has none); the statistics follow (AIAEC mirror 14, partial 21).
- `aiaec_postprocess()`'s comment is back to three lines.

## [Unreleased] — 2026-11-15 — The int16 PCM entries round to nearest (`stream_saturate_s16()`), so an identity pass is exact

`stream/common/stream_mem.h` `stream_saturate_s16()` truncated toward zero after scaling by 32768 and clipping, so every value in (-1, 1) LSB became 0 and every other one lost up to 1 LSB of magnitude: an int16 -> STFT -> ISTFT -> int16 identity pass changed 12-52 % of the samples by 1 LSB. It now rounds half away from zero, the rule of audiolib `wav_io.h`'s writer; NaN is still 0 and +-Inf / out-of-range values still clip.

- **What changes.** The int16 output of the PCM entries `dfn2_stream_post_process()` (and `ainr_postprocess()` over it), `deepvqe_stream_post_process()`, `ulcnet_stream_post_process()` (and `aiaec_postprocess()` over the two): by at most 1 LSB per sample. The `_freq` / `_spec` / f32 entries and every float output are unchanged; Python parity is in float and is not affected. No pool, struct or layout change.
- **Proof.** Every one of the 2^32 float inputs: identical to the `wav_io.h` rule on all finite inputs, at most 1 LSB from the old truncation. Real recordings through the three PCM streams on 16 kHz (DFN2 full 256, Align-ULCNet 256, DeepVQE-S 512; three pairs of speech and DNS 2020 noisy clips) and 48 kHz / 1024 grids (DFN2 full and mask-only, Align-ULCNet, DeepVQE-S; two DNS fullband pairs): the identity pass (DFN2 and Align-ULCNet bypassed, DeepVQE-S with identity CCM taps) went from 12.3-51.7 % of samples 1 LSB toward zero to 0 differing samples; with stand-in models, 13-50 % of the samples change, none by more than 1 LSB.
- **Tests.** `stream/common/tests/test_stream_mem.c`: rows at +-0.4 / +-0.5 / +-0.6 / +-1.5 / +-2.5 LSB and at the clip edges, and the int16 -> float -> int16 round trip of all 65,536 values, also one float step either side; truncation, ties to even, `floor(v + 0.5)`, a truncating negative side, a missing clip and NaN not 0 each fail it. The hand-written oracles of `stream/ainr/dfn2/tests/verify_vs_sample.c`, `stream/aiaec/{deepvqe_s,ulcnet}/tests/verify_vs_sample.c` and `stream/aiaec/tests/test_aiaec_block.c` round the same way, and the four PCM hashes of `stream/ainr/dfn2/tests/pool_size.c` are re-pinned (the old values come back with the old rule).
- **Board.** The samples that take int16 PCM out of these entries change by at most 1 LSB: `alg_ai3_dfn`, `alg_ai3_dfn_lite`, `uvc_ai_nr_stream`, `alg_ai3_deepvqes`, `alg_ai3_alignulcnet`. `uvc_ai_nr_stream` also converts its 48 kHz float hop to int16 with this helper before `ainr_preprocess()`, so its model input changes by at most 1 LSB and its output by more than that where the model is not linear. An A/B of these samples against an older build is no longer bit-identical. `uvc_audio_live3`, `uvc_audio_live2a` and `alg_ai3_aec_dfn` use the `_freq` entries and are unchanged. No sample source changes.

## [Unreleased] — 2026-11-15 — `dfn2_process.h`: the whole header has C linkage under C++

The `extern "C"` block of `lib/ainr/dfn2/dfn2_process.h` closed right after the variant-link symbol, so a C++ consumer got C++ linkage for `DFN2State`'s functions (`dfn2_state_*`, `dfn2_set_erb_matrices`, `dfn2_compute_features`, `dfn2_apply_atten_lim`, `dfn2_enhance_spectrum[_stream]`) and failed to link. The block now ends before the include guard's `#endif`. C is unaffected (`cc -E -P` output identical, full and mask-only).

- **Test.** `make -C stream/ainr/dfn2 test` runs `check-header-cxx`: `tests/cxx_consumer.cpp` calls every function of `dfn2_process.h` plus a few of the other DFN2 headers from C++, and its object must reference no mangled symbol; a generated probe that calls one C++-linkage function must show one (positive control). With the old header the check fails and lists all eight `dfn2_process.h` functions mangled.

## [Unreleased] — 2026-11-15 — Align-ULCNet packet (N > 1): a failed job that carries an alignment event advances the generation

At N > 1 the graph restarts the model at an event hop itself (`frame_reset`), so the generation stays. A job that fails before the device run never applied that restart: the library bypassed the packet with the generation unchanged, and the next packet ran from the pre-event recurrent state. `ulcnet_prepost_frame_bypass()` now zeroes the registered state and advances the generation for a packet opened by `frame_open()` that has an event frame, the same step an N = 1 event takes; the board sample's generation check then zeroes the device state.

- **Unchanged.** Every committed packet, every failed packet without an event, every one-frame hop and every refused finish (which already restarted the state and clears `prepared`, so it is not restarted twice). No pool, struct or layout change. No Python twin.
- **Tests.** `stream/aiaec/tests/test_aiaec_packet.c` `section_failure`: a packet with the hop-10 event fails before the run (generation +1), a packet without an event fails (generation unchanged), and the next packet equals a run from a zero state. Mutations killed: the `prepared` gate removed (the refused-mask rows advance twice), no restart, a restart without an event, and a refused finish that keeps `prepared`.
- **Board.** `alg_ai3_alignulcnet.c` and `uvc_audio_live3.c` only need a rebuild; no production Align-ULCNet graph is registered yet.
- The general recovery sentence ("a failed run or readback leaves the generation unchanged") in `ulcnet_stream.h`, `aiaec.h` and `docs/reference/board/sequence.md` names this exception.

## [Unreleased] — 2026-11-15 — The DFN2 ERB file loader refuses bytes after the two tables and a forward table that is not finite (`AINR_ERR_ERB_FILE`)

`dfn2_stream_init_from_erb_file[_ex]` (both kinds of context; so `ainr_init*` with `AINR_ERR_ERB_FILE`) now also returns NULL when the file goes on after the forward and inverse tables (one more byte is enough) or when a forward value is NaN or infinite. Before, such a file loaded: a NaN forward value passed init and put non-finite values into the accelerator's input tensors on every hop. Checked at init only; no hop code, numbers, pool size, descriptor or layout version changed. Memory mode (caller tables) is unchanged.

- **Real files.** The 15 `fb_*.bin` copies found on this machine and `export_erb_matrix.py --format bin` outputs for six grids (48000 / 1024 / 32, 32000 / 512 / 28, 16000 / 512 / 32 and 24, 16000 / 256 / 24, 8000 / 256 / 24) are all exactly 8 + 2 tables and load under both variants (`erb_partition SR FFT N_ERB FILE`); the 48 kHz exporter file with one byte appended, or with its first forward value set to NaN, is refused.
- **Tests.** `stream/ainr/dfn2/tests/check_erb_file.c`: one byte and one float after the tables, NaN first, +Inf last, -Inf middle and NaN last forward values refused, a finite 3e38 and the clean file accepted, for a full and a `freq_only` context, in the same pool after the refusals. `stream/ainr/tests/test_ainr_pool_gates.c`: a byte appended, a NaN first and an Inf last forward value are `AINR_ERR_ERB_FILE` for both families, ctx NULL and info untouched. Eight mutations of the loader (no EOF probe, a float-sized probe, the probe's result ignored, no scan, a scan skipping the first or the last value, NaN-only, a range limit) are each killed.
- Docs: `stream/ainr/ainr.h` (`AINR_ERR_ERB_FILE`), `ainr_status_text()`, `docs/html/models/dfn2/index.html`, `docs/html/models/dfn2/reference/grid.html`, `docs/reference/board/build.md`, `AINR/DeepFilterNet2/reference/grid.md`.

## [Unreleased] — 2026-11-15 — DFN2 file mode reads the ERB file in chunks for both entry families: the PCM pools lose the table-sized load scratch

`dfn2_stream_init_from_erb_file[_ex]` on a context without `freq_only` (and so `ainr_init()` / `ainr_init_from_erb_file()`) now loads the inverse ERB table the way a `freq_only` context does: 64 floats at a time straight into band-major order. The table-sized load scratch (`n_bins * n_erb * 4` bytes: 65,664 B at 48 kHz / 1024 / 32 bands, 12,384 B at 16 kHz / 256 / 24) is no longer carved; one loader serves both kinds of context. Header, short-file, foreign-grid and partition-of-unity refusals are unchanged.

- **Versions.** `DFN2_STREAM_LAYOUT_VERSION` 4 -> 5 (the file-mode carve of a full context changed; `build_flags_hash` folds the version, so every layout-4 descriptor is refused). `DFN2_PREPOST_CARVE_VERSION` 12 and `AINR_BRIDGE_LAYOUT_VERSION` 1 stay.
- **Pool bytes (host arm64, NE10, from the size queries; the board ABI differs).** Memory mode and every `_freq` pool are unchanged.

  | PCM pool | 16 kHz / 256 N=1 | 16 kHz / 256 N=8 | 48 kHz / 1024 full | 48 kHz / 1024 mask-only |
  |---|---|---|---|---|
  | `dfn2_stream`, file mode | 103,600 -> 91,216 | 158,976 -> 146,592 | 361,296 -> 295,632 | 300,928 -> 235,264 |
  | `ainr_get_mem_size()` | 103,696 -> 91,312 | 160,864 -> 148,480 | 361,392 -> 295,728 | 301,024 -> 235,360 |

- **Same output.** The digest harness of the `freq_only` change (return codes, job flag, state generation, model input tensors and outputs of every hop; three catalog grids and four more, full and mask-only, `SIMD=1` and `0`, seven scenarios) gives the digests of the previous commit on every line (273 + 141 per `SIMD` setting), PCM entries of the stream and of `ainr` included; the size lines move only for `stream_pcm_file` and `ainr_pcm`, each by exactly one table.
- **Board impact.** No sample source changes: `alg_ai3_dfn`, `alg_ai3_dfn_lite` and `uvc_ai_nr_stream` size their pool with `ainr_get_mem_size()` at run time. They must be rebuilt with the library (the pool layout changed).
- **Tests.** `stream/ainr/dfn2/tests/pool_size.c`: file-mode literals, file mode equal to memory mode plus exactly two ERB tables for a full and a `freq_only` context, the `freq_only` saving the same in both modes, and a descriptor of the previous layout version refused. `stream/ainr/tests/test_ainr_pool_gates.c`: the four PCM literals. `check_erb_file` (PCM file mode against memory mode, hop for hop) and `freq_only` (chunk loader against caller tables) cover the loader.
- Docs: `docs/html/models/dfn2/reference/board.html`, `docs/html/pipelines/{mono,4ch}/dfn2/index.html`, `docs/html/pipelines/reference/dfn2/configuration.html`, `docs/html/start/reference/grids.html`.

## [Unreleased] — 2026-11-15 — AIAEC spectrum-only contexts carry no analysis, synthesis or PCM buffer (`UlcnetStreamConfig.freq_only`, `DeepVqeStreamConfig.freq_only`, `aiaec_*_freq`)

An `UlcnetStream` or `DeepVqeStream` served both entry families from one pool, so a caller of the `_freq` entries alone still paid for the FFT table, two `AcStft`, one `AcIstft` and the PCM scratch slab of the PCM entries. Both config structs have a new field `freq_only` (`UlcnetStreamConfig`: after `frames_per_call`; `DeepVqeStreamConfig`: after `grid`): non-zero carves the stream header and the `UlcnetPrepost` / `DeepVqePrepost`, nothing else. Such a stream runs the `_freq` entries exactly as before; its PCM entries (`ulcnet_stream_pre_process_f32()`, `_pre_process()`, `_pre_process_packet_f32()`, `_post_process()`; `deepvqe_stream_pre_process()`, `_post_process()`) return -1 and change nothing (an open `_freq` hop stays open), and reset leaves out the framing it does not have. `ulcnet_stream_pre_process_packet_freq()` hands the caller's planes to `UlcnetPrepost` directly instead of copying them into the stream's scratch first (`UlcnetPrepost` copies them itself); outputs are unchanged on both kinds of stream.

- **Callers moved to it.** `stream/aiaec/aiaec.c`: the inner stream of the `_freq` family (`aiaec_get_mem_size_freq()`, `aiaec_init_freq()`), through one config helper per model that sizing and init share; the PCM and `_spec` families are untouched. The desktop Align-ULCNet pipelines, which call only the `_freq` entries: `pipelines/mono/alignulcnet/main.c`, `pipelines/4ch/alignulcnet/main.c`, `pipelines/tests/ulcnet_test_fixture.h`, `pipelines/tests/test_readme_flow.c`.
- **Versions.** `ULCNET_STREAM_LAYOUT_VERSION` 3 -> 4 and `DEEPVQE_STREAM_LAYOUT_VERSION` 3 -> 4: the config structs grow (host arm64 `UlcnetStreamConfig` 24 -> 28, `DeepVqeStreamConfig` 280 -> 288 bytes) and `build_flags_hash` folds `freq_only` (any non-zero value is one build), so a descriptor of one kind of stream is refused by the init of the other, and a descriptor kept from layout 3 is refused. Full-stream pools keep their bytes; their hashes move with the version (ULCNet 16 kHz / 256 `1000c299` -> `01ecac8e`, 16 kHz / 512 `50231f40` -> `0bd40233`, 48 kHz / 1024 `e57b352a` -> `7d87e325`; DeepVQE-S 16 kHz / 256 `2e331961` -> `3109bf1e`, 16 kHz / 512 `d1afb068` -> `67287f4b`, N=1). `ULCNET_PREPOST_CARVE_VERSION` 9, `ULCNET_MODEL_IO_LAYOUT_VERSION` 18 / 19, `DEEPVQE_PREPOST_LAYOUT_VERSION` 4 and both `*_STREAM_DESCRIPTOR_VERSION` 1 stay; `aiaec` has no descriptor (`StreamMemReq` is bytes and alignment).
- **Pool bytes (host arm64, NE10, from the size queries; the board ABI differs).** A `freq_only` stream leaves out the FFT, 2 x `AcStft`, `AcIstft` and `(3 hop + 6 planes) x 4` bytes of scratch (ULCNet planes are N x n_bins, DeepVQE-S n_bins): 45,824 B at fft 256, 88,320 B at fft 512, 173,312 B at fft 1024, plus four more ULCNet planes per packet hop.

  | `_freq` pool | 16 kHz / 256 N=1 | 16 kHz / 256 N=8 | 16 kHz / 512 N=1 | 16 kHz / 512 N=8 | 48 kHz / 1024 N=1 | 48 kHz / 1024 N=8 |
  |---|---|---|---|---|---|---|
  | `ulcnet_stream` | 53,648 -> 7,824 | 125,808 -> 58,320 | 103,312 -> 14,992 | 247,152 -> 115,664 | 202,640 -> 29,328 | 489,840 -> 230,352 |
  | `deepvqe_stream` | 64,096 -> 18,272 | 165,136 -> 119,312 | 124,000 -> 35,680 | 325,392 -> 237,072 | — | — |
  | `aiaec_get_mem_size_freq()`, Align-ULCNet (test catalog) | 53,744 -> 7,920 | 125,904 -> 58,416 | 103,408 -> 15,088 | — | — | — |
  | `aiaec_get_mem_size_freq()`, DeepVQE-S | 64,192 -> 18,368 | 172,464 -> 126,640 | 124,096 -> 35,776 | — | — | — |

  `ulcnet_stream` 16 kHz / 512 N=64: 1,398,960 -> 922,064. A build of `aiaec.c` for one model has a 16-byte smaller `AiaecCore`. Every PCM and `_spec` pool is unchanged. Skeleton banners: `mono_alignulcnet` 16 kHz n=5 527,200 -> 438,880, n=1 482,144 -> 393,824, 48 kHz n=5 1,119,584 -> 946,272, n=1 1,025,376 -> 852,064; `4ch_alignulcnet` stream pool 103,312 -> 14,992 (16 kHz) and 202,640 -> 29,328 (48 kHz), unit and framing pools unchanged.
- **Same output.** A scratch digest harness of the public API (every return code, every tensor handed to the accelerator with its element count, the NaN prefill, generation, restart mask, output spectra or PCM, `AiaecWork`, 64-byte guards on both sides of each exactly sized pool) on 16 kHz / 256, 16 kHz / 512 and 48 kHz / 1024 at N=1 and 8 and 16 kHz / 512 N=64, through alignment events and their reprime hops, model failure, NaN / Inf / 3e30 / unwritten heads, a NaN input bin, resets between hops and with a hop open, calls out of order, all-zero and full-scale hops: the 63 digests (`_freq` entries on `freq_only` streams, `ulcnet_stream` / `deepvqe_stream` PCM, `_spec` and `_freq` entries on full streams, the three `aiaec` families) equal those of `c4faa0d` at `SIMD=1` and `0`; the 46 pools that are not spectrum-only keep their bytes and the 17 spectrum-only pools have the sizes above. On `4badc39` the same `_freq` runs with the left-out regions filled with NaN after init gave the same digests (the regions are written only by reset, with zeros, and never read).
- **Board impact.** No sample source changes. `uvc_audio_live3.c` is the one sample on `aiaec_*_freq` (Align-ULCNet 16 kHz / 512 N=1, sized at run time from its arena): its `aiaec_bytes` drops by 88,320 B once the production catalog has that row. `alg_ai3_alignulcnet.c` and `alg_ai3_deepvqes.c` use the PCM family (unchanged). Every unit that includes `ulcnet_stream.h` / `deepvqe_stream.h` or links `aiaec.c` must be rebuilt together.
- **Tests.** New `stream/aiaec/ulcnet/tests/freq_only.c` and `stream/aiaec/deepvqe_s/tests/freq_only.c` (`make test-freq-only` in each directory, part of `test`): a pool of exactly the queried size between two canary zones (16 bytes less refused), the pinned byte counts and the difference to a full stream derived from audiolib's size queries, the `freq_only` stream against a full one call for call (codes, tensors, generation, restart mask, spectra) through the scenarios above at N=1, 8 and (ULCNet) 64, every PCM entry refused with the pool bytes, generation and output unchanged, both descriptors refused by the other init, and `freq_only = 7` the build of 1. `stream/aiaec/tests/test_aiaec_spec_freq.c`: every family runs in an exactly sized pool with a canary behind it, the `_freq` pool is pinned and equals `_spec` less the framing. `pipelines/tests/test_target_abi.c`: a `freq_only` `ulcnet_stream` row (12 descriptors). 33 mutations of `ulcnet_stream.c`, `deepvqe_stream.c` and `aiaec.c` (scratch or framing still carved, hash without `freq_only` or with its raw value, each PCM gate removed, framing built on a `freq_only` stream, a query 16 bytes short, the packet entry reading the stream's scratch or swapping planes, `aiaec` sizing and init disagreeing, `_spec` built `freq_only`, `_freq` built full): every one fails a test except four that cannot change behaviour (reset without its guard, twice: audiolib's reset accepts NULL; dropping the explicit NULL framing pointers: init zeroes the pool; swapping the far planes of a packet: the model sees far magnitude only); error and far planes swapped in the packet entry are caught by `test_aiaec_packet`. Layout versions left at 3 fail the harness (hash moved without the version).
- **Not verified here:** the board build, board RAM and an ILP32 target's byte counts.
- Docs: `pipelines/{mono,4ch}/alignulcnet/README.md`, `pipelines/reference/call-flow.md`, `docs/html/pipelines/{mono,4ch}/align-ulcnet/index.html`, `docs/html/pipelines/reference/align-ulcnet/{configuration,integration}.html`, `docs/html/models/{align-ulcnet,deepvqe-s}/index.html`, `docs/reference/board/sequence.md`, `docs/c_python_function_map.md` (row PL-ULC-01). The Python twins are untouched.

## [Unreleased] — 2026-11-15 — Align-ULCNet accelerator adapter and `ulcnet_process` removed

`lib/aiaec/ulcnet/ulcnet_accelerator_adapter.{c,h}`, `lib/aiaec/ulcnet/ulcnet_process.{c,h}` and `AIAEC/tests/test_ulcnet_accelerator_adapter.c` are deleted. Since the desktop pipelines moved to `ulcnet_stream_*_freq`, nothing called the adapter, `UlcnetModel`, `UlcnetNnJob` or `ulcnet_frame_is_finite()`; the board samples, the model stream and `aiaec.c` never did.

- **Framing contract.** The audiolib framing the model is trained against (sqrt-Hann, center 0, `AC_OLA_ACCUMULATE`, scales 1) moved verbatim into the header comment of `ulcnet_prepost.h`, which now includes `ulcnet_model_io.h` directly. The docs that cited `ulcnet_process.h` point there.
- **Build.** `libaiaec_prepost.a` is `ulcnet_model_io`, `ulcnet_prepost`, `deepvqe_process`, `deepvqe_prepost`. `pipelines/Makefile` drops the adapter objects, `test_ulcnet_accelerator_adapter` and `ulcnet_process.c` from `TARGET_ABI_SRCS`; the header lists of `pipelines/4ch/Makefile`, `stream/aiaec/Makefile` and `stream/aiaec/ulcnet/Makefile` drop `ulcnet_process.h`.
- **No behaviour, pool or layout change.** The 49 remaining objects per SIMD setting (pipelines, 4ch, the AIAEC archive, `stream/aiaec`, `stream/aiaec/ulcnet`; no `-g`) are byte-identical at SIMD 1 and 0, the `ulcnet_stream_*_freq` digest is unchanged, and `ULCNET_MODEL_IO_LAYOUT_VERSION` 18 / 19, `ULCNET_PREPOST_CARVE_VERSION` 9 and `ULCNET_STREAM_LAYOUT_VERSION` 3 stay.
- **Python tests.** Compile lists and strings only: `test_prepost_chains.py` no longer compiles `ulcnet_process.c`, `test_export_streaming_ulcnet.py` no longer pins the deleted test's state table, and two docstrings name `ulcnet_prepost.h`.
- Docs: `AIAEC/Align_ULCNet/reference/deployment.md`, `docs/maintenance/models.md`, `docs/reference/board/sequence.md`, `docs/html/models/align-ulcnet/reference/core.html` (the callback example is gone), `docs/html/models/reference/onnx.html`, `docs/html/start/reference/grids.html`, `docs/html/pipelines/reference/align-ulcnet/configuration.html`, `pipelines/{mono,4ch}/alignulcnet/README.md`, `docs/c_python_function_map.md` (874 C functions).

## [Unreleased] — 2026-11-15 — The desktop Align-ULCNet pipelines run the model stream (`ulcnet_stream_*_freq`); `pipelines/ulcnet/` is retired

The two example mains (`mono_alignulcnet`, `4ch_alignulcnet`) now drive the spectrum entries of the model-stream layer `stream/aiaec/ulcnet` (`ulcnet_stream_init_ex()`, `ulcnet_stream_pre_process_freq()` / `ulcnet_stream_post_process_freq()`) between their own `AcStft` / `AcIstft`. `pipelines/ulcnet/` (`ulcnet_nn_stage.{h,c}`, `tests/test_ulcnet_nn_stage.c`, `README.md`) is deleted; the examples no longer link `ulcnet_accelerator_adapter.o` or `ulcnet_process.o` (both stay in `lib/aiaec/ulcnet` and `libaiaec_prepost.a`, with `test_ulcnet_accelerator_adapter`). No library source changed apart from two comment lines of `ulcnet_stream.h` (`cc -E -P` identical); `ULCNET_STREAM_LAYOUT_VERSION` 3 stays. The Python twins are untouched.

- **What a hop is now.** There is no model callback: `pre_process_freq` returns 1 with the graph's tensors (`UlcnetHopIO`: the four feature tensors, the NaN-prefilled complex mask, `state_generation`) or 0 on the identity hop after an alignment boundary, the caller runs the accelerator, and `post_process_freq(ai_rc)` hands back the enhanced spectrum or the error spectrum. `run_accelerator()` in both mains keeps its signature and its `return -1` body. Each main still checks the graph's descriptor against the grid before any hop (`ulcnet_model_io_descriptor_validate()` and one frame per call), which the stage's `set_model` did before. A restart of the model state (alignment boundary, reset) is the stream's state generation advancing, where the stage called `model->reset`.
- **Bit-identical output.** (1) Stage + adapter against `ulcnet_stream_*_freq` on the tree before the deletion: 40 cases (16 kHz / 256, 16 kHz / 512 at D 8 and 32, 48 kHz / 1024; ten scenarios of 200 hops: no event, alignment events, failed run, unwritten / NaN / Inf / 3.0e30 mask, NaN error bin, reset), 0 divergent in output, model calls and state generation, 8 of 8 injected faults caught, at `SIMD=1` and `SIMD=0`. (2) Each `main.c` before and after, with one stand-in accelerator substituted in `run_accelerator()` and two recordings fed through the mains' own hop functions: 120 runs (mono and 4ch x 16 kHz and 48 kHz x `--delay-num-filters` 5 / 1 and `--delay-mode fixed` x a model that always fails, one that always writes its mask, one that also fails, writes NaN, leaves the mask unwritten and writes 3.0e30 on a schedule x `SIMD=1` / `0`), every output file identical, with the same model call count and final state generation; every run has one alignment boundary. Planted defects in the new mains (error and far swapped, imaginary plane dropped, output planes swapped, a failed run reported as success, no alignment events) turn the comparison red in 14 of 18 runs; the four that stay green are a model that never fails under "failure reported as success" (2) and `changed` not forwarded on recordings whose only boundary also raises `solid` (2). Rerun with two more recordings, the first two joined end to end in both orders: 240 runs identical. In one order the delay moves while the alignment stays solid (a second alignment boundary in all 48 of its runs that estimate the delay), and there `changed` not forwarded turns red in both mains. The 4ch microphones are four staggered copies of the mono recording with independent noise.
- **Pool sizes (NE10, from the skeleton banners).** The stream's pool also holds the analysis and synthesis objects of its PCM entries, which the examples do not call. Model part (stage + adapter before, stream after): 16 kHz / 512 6,320 + 14,864 = 21,184 -> 103,312 B; 48 kHz / 1024 12,464 + 29,200 = 41,664 -> 202,640 B. Mono banner total (the adapter was a separate block before): 16 kHz n=5 430,160 + 14,864 -> 527,200 B, n=1 385,104 + 14,864 -> 482,144 B; 48 kHz n=5 929,360 + 29,200 -> 1,119,584 B, n=1 835,152 + 29,200 -> 1,025,376 B. 4ch: unit pool unchanged (2,453,056 / 5,120,288 B); framing 47,808 -> 49,920 B and 93,888 -> 98,048 B (six float planes where there were two complex spectra: the beam spectrum is copied into planes).
- **Behaviour that changed.** The banners: `model I/O D=8 -> model stream ... bytes` replaces the `accelerator adapter pool` line; the 4ch line says `stream pool` and `six spectrum planes`, and the `--sample-rate` usage line names the stream. Nothing else: exit codes, the delay-profile line (apart from its pool figure) and the audio are the same.
- **Tests.** `test_mono_ulcnet` (20 rows) and `test_4ch_ulcnet` (19 rows, one new) drive the stream through `pipelines/tests/ulcnet_test_fixture.h` (moved from `pipelines/ulcnet/tests/`, rewritten): the stand-in models write the graph's mask instead of a spectrum, a model "reset" is a generation step seen before the frame, and the fixture checks on every frame that the tensors are the front end of the spectra handed in and that the mask staging is NaN-filled. Expectations that were exact for a spectrum-level model are now exact against the fixture's own copy of the mask arithmetic (accumulate references, the 4ch identity chain, the first applied frame) and a tolerance where the claim is "the model changed nothing" (unit mask within 2e-6 of the peak, measured 2e-7). The raw-far timestamp rows synthesise the far frames the model was handed. The band-mask models are complex. Rows of the stage's own API are gone: the descriptor rule "an inferring model must publish a descriptor", the frame left open / never fetched, `nn_begin` / `nn_end` ordering; in their place a mask left unwritten (NaN prefill) and the state-generation step of every refused frame. New 4ch row: a delay shift while the alignment stays solid restarts the model (it kills "`changed` not forwarded", which no 4ch row did). `test_readme_flow` runs the documented sequence and checks the mask is applied on every frame it ran and the error spectrum handed on otherwise. `AIAEC/tests/test_deployed_frontend_contract.py`: compile list (`ulcnet_stream.c`, no stage, no adapter) and the `process_hop()` patterns (`ac_stft_push_planar` of `rctx.formed_hop` and of `ref`, then one `ulcnet_stream_pre_process_freq` on those planes).
- **The stage's test moved.** `stream/aiaec/ulcnet/tests/test_ulcnet_stream_freq.c` (in `make -C stream/aiaec/ulcnet test`, four grids) holds what `test_ulcnet_nn_stage.c` asserted about the spectrum path, against expectations built in the test: the six alignment schedules and a seventh, every fail-open kind with its generation step, the tensors and NaN staging of every frame, an accepted hop bit for bit against a second copy of the mask arithmetic, the call order and the refusals that must not change the pool, reset and replay against a new stream, the grid refusals, refused inits that leave the block bytes alone, a stale pool, and the packet / one-hop entry split.
- **Mutation evidence.** 48 mutations of `ulcnet_stream.c`, `ulcnet_model_io.c`, `ulcnet_prepost.c`, the two compositions and the fixture against the two pipeline tests: every mono row and every 4ch row that involves the stream is killed by at least one; survivors are init without its `memset` (every field is set before it is read) and, in mono only, reset keeping the edge memory and an init writing 16 bytes past the query (4ch kills both). 40 mutations against the stream test: 35 killed; survivors are init without `memset`, reset keeping a pending reprime (never pending between hops while `ULCNET_STREAM_REPRIME_FRAMES` is 1), and three checks that `UlcnetPrepost` repeats (delay depth in the hash, a negative `frames_per_call`, the grid check in the query). 11 of 11 text mutations of the lane are caught by the Python gate.
- **Descriptor gate.** `test_4ch_ulcnet`'s descriptor gate also refuses a descriptor one field off the grid's own (D below and above the range, sample rate, FFT size, hop, bin count) and packet descriptors of N = 2 and the maximum (built with layout 19 and their frame count, refused by the one-hop composition), the stage test's field-by-field rows: without them, dropping the D, FFT-size or hop check of `ulcnet_model_io_descriptor_validate()` passed every test.
- **Build.** `pipelines/Makefile` and `pipelines/4ch/Makefile` compile `ulcnet_stream.o` (`ULCNET_STREAM_INC` = `-I stream/aiaec/ulcnet -I lib/aiaec/ulcnet`) and link it with `ulcnet_prepost.o` and `ulcnet_model_io.o` into the example and its tests; `test_ulcnet_nn_stage` is gone from `make -C pipelines test`. `audit_fp_contract.sh` audits `ulcnet_model_io.o` in the two example builds in place of `ulcnet_nn_stage.o` (11 objects; built with contraction allowed it shows 24 fma-class instructions, so the row can fail). `test_target_abi` has a `ulcnet_stream` row (11 descriptors).
- Docs: `pipelines/{mono,4ch}/alignulcnet/README.md`, `pipelines/README.md`, `pipelines/4ch/README.md`, `pipelines/reference/{call-flow,build}.md`, `pipelines/4ch/reference/c-api.md`, `docs/html/pipelines/{mono,4ch}/align-ulcnet/index.html`, `docs/html/pipelines/reference/align-ulcnet/{configuration,integration}.html`, `docs/html/pipelines/4ch/aec-nr-res/index.html`, `docs/html/start/reference/{grids,integration,timing}.html`, `docs/html/models/align-ulcnet/{index,reference/core}.html`, `docs/html/models/reference/onnx.html`, `docs/reference/{array/beam,board/memory,board/sequence,mono/frequency-domain}.md`, `docs/maintenance/publish.md`, `docs/c_python_function_map.md` (row PL-ULC-01, 884 C functions), `AIAEC/Align_ULCNet/reference/{alignment,deployment}.md`, `AIAEC/dataset_gen/README.md`.

## [Unreleased] — 2026-11-15 — DFN2 spectrum-only contexts carry no analysis, synthesis or PCM buffer (`DFN2StreamConfig.freq_only`, `ainr_*_freq`)

A `DFN2Stream` served both entry families from one pool, so a caller that only used `dfn2_stream_pre_process_freq_dual()` / `dfn2_stream_post_process_freq()` still paid for the FFT table, the `AcStft`, the `AcIstft` and the PCM scratch slab of the PCM entries, and in file mode for a third ERB-table-sized load scratch. `DFN2StreamConfig` has a new field `freq_only` (after `hpf_cutoff_hz`): non-zero carves the header, the `DFN2Prepost` and, in file mode, the two ERB tables, nothing else. Such a context runs the `_freq` entries exactly as before; `dfn2_stream_pre_process()` / `dfn2_stream_post_process()` return -1 on it and write nothing, `dfn2_stream_reset()` resets the `DFN2Prepost` alone, and `hpf_enable` with `freq_only` is a configuration error (the HPF belongs before the caller's analysis). Its file loader reads the inverse table 64 floats at a time straight into band-major order; header, short-file and partition-of-unity refusals are unchanged. `freq_only == 0` is the context as it was: same carve, same bytes, both families.

- **Callers moved to it.** `stream/ainr/ainr.c`: the inner stream of the `_freq` family (`ainr_get_mem_size_freq()`, `ainr_init_freq()`); the PCM family is untouched. `pipelines/dfn/dfn_hop_pool.h`: `dfn_hop_pool_plan()` sets `stream_cfg.freq_only`, which shrinks the DFN tail of `mono_aec_dfn_res` and `4ch_aec_bf_dfn_res`.
- **Versions.** `DFN2_STREAM_LAYOUT_VERSION` 3 -> 4: `sizeof(DFN2StreamConfig)` grows (host arm64 88 -> 96 bytes) and `build_flags_hash` folds `freq_only`, so a descriptor of one kind of context is refused by the init of the other, and a descriptor kept from layout 3 is refused. `DFN2_PREPOST_CARVE_VERSION` 12 and `AINR_BRIDGE_LAYOUT_VERSION` 1 stay; `ainr` has no descriptor (`StreamMemReq` is bytes and alignment).
- **Pool bytes (host arm64, NE10, from the size queries; the board ABI differs).** What a `freq_only` context leaves out: 71,616 B at 48 kHz / 1024 (FFT 26,352, STFT 16,480, ISTFT 16,480, scratch 12,304) and 18,624 B at 16 kHz / 256; in file mode one more table, 65,664 B and 12,384 B.

  | `_freq` pool | 16 kHz / 256 N=1 | 16 kHz / 256 N=8 | 48 kHz / 1024 full | 48 kHz / 1024 mask-only |
  |---|---|---|---|---|
  | `dfn2_stream`, memory mode | 66,448 -> 47,824 | 121,824 -> 103,200 | 164,304 -> 92,688 | 103,936 -> 32,320 |
  | `dfn2_stream`, file mode | 103,600 -> 72,592 | 158,976 -> 127,968 | 361,296 -> 224,016 | 300,928 -> 163,648 |
  | `ainr_get_mem_size_freq()` | 105,760 -> 74,752 | 168,368 -> 137,360 | 369,600 -> 232,320 | 309,232 -> 171,952 |

  Every PCM pool is unchanged (`dfn2_stream` memory / file mode, `ainr_get_mem_size()`). Skeleton banners: `mono_aec_dfn_res` 16 kHz 535,280 -> 463,664, 48 kHz 1,003,744 -> 932,128, 8 kHz 508,544 -> 436,928, 16 kHz host on the 16 kHz model 322,736 -> 304,112; `4ch_aec_bf_dfn_res` 16 kHz 1,788,192 -> 1,716,576, 48 kHz 5,763,776 -> 5,692,160, 16 kHz core with the 16 kHz model 1,575,648 -> 1,557,024. DFN2 part of `alg_ai3_aec_dfn` (ainr pool, bridge pool, conversion scratch): 48 kHz host 381,912 -> 244,632, 16 kHz host on the 16 kHz model 108,856 -> 77,848, 16 kHz / 256 bridged 499,768 -> 362,488, 8 kHz / 256 bridged 501,400 -> 364,120; the ainr pool of `uvc_audio_live2a` 369,600 -> 232,320.
- **Same output.** A scratch digest harness (return codes, job flag, state generation, both model input tensors, spectra and validity flag of every hop) on the three catalog grids and four more (32 kHz / 512 lookahead 2/4, 16 kHz / 512, 48 kHz lookahead 2/3, 16 kHz N=4), full and mask-only, `SIMD=1` and `0`, seven scenarios each (normal, attenuation limit, failed run, non-finite heads, refused bins 1e21 / NaN / Inf, resets between hops and inside one, all mixed over 2,400 hops): `dfn2_stream` memory mode, file mode and `ainr_*_freq` give the digests of `4badc39` on all 140 + 70 `_freq` lines, and the PCM entries of the stream and of `ainr` give theirs with unchanged pool sizes. On `4badc39` the same `_freq` runs with the left-out regions filled with 0xFF after init give the same digests and leave every filled byte alone. The whole `alg_ai3_aec_dfn.c` on the host (stand-in SDK, real AEC) gives the 84 + 84 rows of the stage-4 comparison unchanged, and PCM and model-call trace equal to the same sample linked against `4badc39` on 40 + 40 runs; the block harness of `uvc_audio_live2a.c` gives its 148 + 148 rows unchanged apart from the byte counts.
- **Board impact.** No sample source changes: both samples size their pool with `ainr_get_mem_size_freq()` at run time. Every unit that includes `dfn2_stream.h` or links `ainr.c` / `dfn2_stream.c` must be rebuilt together (the config struct and the pool layout of the `_freq` family changed).
- **Tests.** New `stream/ainr/dfn2/tests/freq_only.c` (`make -C stream/ainr/dfn2 freq-only`, part of `test`; memory mode also under `NO_STDIO=1`): a pool of exactly the queried size between two canary zones, the `freq_only` context against a full one hop for hop through failures, refused bins, attenuation limit and resets, both descriptors refused by the other init, the PCM entries and `hpf_enable` refused, and the file loader's refusals. `pool_size.c` pins the `freq_only` byte counts and derives the difference from audiolib's size queries; `stream/ainr/tests/test_ainr_pool_gates.c` pins the byte counts of both `ainr` families and checks the bytes after an exactly sized pool; `pipelines/dfn/tests/test_dfn_hop.c` checks that the plan's stream is the `freq_only` one. `pipelines/dfn/tests/test_dfn_stream_equivalence.c` is unchanged and runs on a full context.
- **Not verified here:** the board build, board RAM, and the real ERB files through the chunked loader (the loader's result is compared with the staged one on synthetic tables).
- Docs: `pipelines/mono/aec_dfn_res/README.md`, `pipelines/4ch/aec_bf_dfn_res/README.md`, `pipelines/dfn/README.md`, `pipelines/reference/call-flow.md`, `docs/html/pipelines/{mono,4ch}/dfn2/index.html`, `docs/html/pipelines/reference/dfn2/configuration.html`, `docs/html/models/dfn2/reference/{board,bridge,core}.html`, `docs/html/start/reference/grids.html`, `docs/reference/board/sequence.md`, `docs/maintenance/pipeline.md`, `docs/c_python_function_map.md` (row PL-DFN-18).

## [Unreleased] — 2026-11-15 — Align-ULCNet: an enhanced spectrum that is not finite is refused (`ulcnet_model_io_enhance_spectrum()` / `_enhance_packet()`)

A finite mask could still give a non-finite frame: the inverse signed power (exponent 1 / 0.3) of a mask bin such as 3.0e30 times the compressed error overflows to Inf, and a non-finite error bin passes through a finite mask. `ulcnet_model_io_enhance_spectrum()` and `ulcnet_model_io_enhance_packet()` (`lib/aiaec/ulcnet/ulcnet_model_io.c`) now scan the enhanced real and imaginary planes right after the back end (`skn_all_finite_f32`, one pass over `n_bins` per plane; a packet scans every frame it wrote, never an event frame) and refuse the frame exactly as they refuse a non-finite mask or state: the registered recurrent state is zeroed, the state generation advances by one, the transaction is dropped and -1 is returned. `ulcnet_prepost_frame_finish()` returns -1 with the frame still open, so `ulcnet_stream_post_process*()` and `aiaec_postprocess*()` bypass it: the hop (a packet: all N hops) is the linear-error spectrum bit for bit. Before, `ulcnet_stream` and `aiaec` put the Inf bin out and kept the state; the desktop `UlcnetNnStage` already fell back to the error spectrum but kept the state.

- **No pool, struct or layout change**: `ULCNET_MODEL_IO_LAYOUT_VERSION` 18 / 19, `ULCNET_PREPOST_CARVE_VERSION` 9 and `ULCNET_STREAM_LAYOUT_VERSION` 3 stay.
- **Contract change of the two model-I/O calls.** After a refused mask or state the caller's enhanced arrays are untouched, as before; after a refused spectrum they hold the refused frame (no valid frame). Through `UlcnetPrepost` that staging cannot be read: `post_process_freq()` refuses while the frame is open, and `frame_bypass()` or a fresh `frame_open()` + `frame_finish()` overwrites it.
- **Every accepted frame is bit-identical.** A scratch harness (stage + adapter against `ulcnet_stream_*_freq`; 16 kHz / 256, 16 kHz / 512 at D 8 and 32, 48 kHz / 1024; ten scenarios of 200 hops) gives the same stream digest before and after for the normal, alignment-event, failed-run, unwritten-mask, NaN-mask, Inf-mask and reset scenarios at `SIMD=1` and `SIMD=0` (28 of 40 cases); the 12 that change are the two 3.0e30-mask scenarios (15 hops each) and the NaN-error-bin scenario (1 hop), which are also the only cases where the stage and the stream disagreed before; they agree now (0 of 40 divergent).
- **Board impact.** `alg_ai3_alignulcnet.c` and `uvc_audio_live3.c` reach this through `aiaec_postprocess()` / `aiaec_postprocess_freq()` and already reload their state buffers when `work.state_generation` changes; no sample source changes. The behaviour differs only on a hop whose enhanced spectrum is not finite: the state restarts, and the bypass sends the error spectrum out instead of the enhanced frame. When the enhanced spectrum overflowed from a finite error spectrum, that output is finite; when the error spectrum itself had a NaN bin, the bypassed output is that error spectrum and still carries the bin.
- **Cost (host arm64, `-O2`, `ulcnet_prepost_frame_finish()` per hop, median of five runs).** The denominator is the whole call (mask scan, back end, the new scan). NEON: 129 bins 638 -> 638 ns, 257 bins 1256 -> 1270 ns, 513 bins 2503 -> 2532 ns (about 1 %, at most 2 % across runs; the difference is near the timer's noise). Forced scalar: 726 -> 794, 1429 -> 1571, 2867 -> 3140 ns (about 9.5 %).
- **Python.** `export_onnx.host_back_end` has no such check; the guard is C only (function map row AIAEC-UL-02).
- **Tests.** `AIAEC/tests/test_prepost_chains.c`: the chain's huge-mask hops (mask and state scaled by 1e38) must be refused with the state zeroed, the generation +1 and the error spectrum out, and the one over a zero spectrum accepted; the NEON and scalar digests still agree. `stream/aiaec/ulcnet/tests/verify_vs_sample.c` (`run_refused_spectrum_row`): on the `_freq` entries a 3.0e30 mask bin that overflows only the real plane, one that overflows only the imaginary plane, a NaN error bin, and the mask over a zero spectrum (accepted); on the bare class a retried `frame_finish()` does not advance the generation again and a fresh `frame_open()` recovers the frame. `stream/aiaec/tests/test_aiaec_packet.c`: two overflowing-mask modes in `section_failure` (frame 5 real plane, last frame imaginary plane) and `section_model_io_enhanced_check` (an event frame is neither written nor scanned; one refusal per bad frame with the registered state zeroed, a retry adds none, at both boundaries; a NaN mask bin in an event frame and a NaN in the registered state of a packet are still refused by the mask and state gate). Sixteen mutations of the new check (removed, one plane only, first frame only, no restart, state not zeroed, transaction left armed, event frames scanned or written, scan before the back end) and the removal of the mask and state gate at either call site each fail at least one test.
- Docs: `AIAEC/Align_ULCNet/reference/deployment.md`, `docs/html/models/align-ulcnet/{index,reference/core}.html`, `docs/html/pipelines/reference/align-ulcnet/configuration.html`, `docs/maintenance/graph-wiring.md`, `docs/c_python_function_map.md`.

## [Unreleased] — 2026-11-15 — `pipelines/common/` dissolved: `example_cli.h` sits in the `pipelines/` root, `cng_aec3.py` and its tests under `pipelines/mono/` (no behaviour change)

Four files moved, none changed in content; the folder is gone.

| Before | Now |
|---|---|
| `pipelines/common/example_cli.h` | `pipelines/example_cli.h` |
| `pipelines/common/cng_aec3.py` (`pipelines.common.cng_aec3`) | `pipelines/mono/cng_aec3.py` (`pipelines.mono.cng_aec3`) |
| `pipelines/common/tests/test_cng_aec3.py`, `test_cng_floor_constant.py` | `pipelines/mono/tests/` |

- **C.** The seven includes read `#include "example_cli.h"` (`mono/aec_nr_res/main.c`, `mono/alignulcnet/main.c`, `dfn/dfn_example_args.h`, `4ch/4ch_external_static.c`, `4ch/array_sample_driver.c`, `4ch/alignulcnet/main.c`, `4ch/tests/test_array_sample_driver.c`). No `-I` changed: the header resolves through the existing `-I.` (`pipelines/Makefile`) and `-I$(PIPELINES_DIR)` (`pipelines/4ch/Makefile`); only the prerequisite lists of the two Makefiles name the new path. Preprocessed text (`cc -E -P`) and `-g0` objects of the nine translation units are identical before and after at `SIMD=1` and `0`.
- **Python.** Import sites: `mono/aec_dfn_res/pipeline.py`, `tests/pyc_parity/pyc_parity.py` (three imports and the hashed `SOURCES` path) and `mono/tests/test_cng_aec3.py`. The module path is the only edit.
- **AEC `cli_level.h` was not reused**: it holds `aec_cli_parse_level()` / `aec_cli_parse_int()` only, not the delay options or the `example_cli_*` parsers.
- **Outside this repo.** No board sample includes a file of the old folder. `AEC/docs/c_python_function_map.md` names `Audio_ALG/pipelines/common/cng_aec3.py` in one sentence.

## [Unreleased] — 2026-11-15 — Board samples `alg_ai3_aec_dfn.c` and `uvc_audio_live2a.c` run DFN2 through `ainr_*_freq` and the library `AinrBridge`

No library source changed. The two samples (`audio_sample/`, outside this repo) no longer use `DfnResStage` / `dfn_rate_bridge` (`alg_ai3_aec_dfn.c`) or an embedded copy of the bridge (`uvc_audio_live2a.c`): each 48 kHz or model-grid frame goes `ainr_preprocess_freq()` -> the sample's AI3 run -> `ainr_postprocess_freq()`, and a host that is not on the model rate goes around the `AinrBridge` of `stream/ainr/bridge`. `dfn_res_stage.{h,c}`, `dfn_rate_bridge.{h,c}` and their archive members stay.

- **Sample build.** `alg_ai3_aec_dfn`: add `stream/ainr/ainr.c`, `stream/ainr/dfn2/dfn2_stream.c`, `stream/ainr/bridge/ainr_bridge.c` and the include directories `stream/ainr/bridge` and audiolib's `common` (`ainr.h` reaches `utility/mem_align.h`; `ainr.c` and `dfn2_stream.c` also need `stream/ainr/dfn2`); `pipelines/dfn/dfn_res_stage.c`, `dfn_rate_bridge.c` and `-I pipelines` are no longer needed. `uvc_audio_live2a`: add `stream/ainr/bridge/ainr_bridge.c` and `-I stream/ainr/bridge`. The ERB directory of `alg_ai3_aec_dfn` is now the library's (`-DAINR_ERB_MATRIX_DIR` on `ainr.c`, same default and file naming); its model file is still chosen by the sample (`--quant8` / `--quant16_8`), and `ainr.c` must be bound to a product model with a catalog row (`AINR_PRODUCT_MODEL_ID` 1, the default, or 11: with 10 every grid is refused).
- **Same output on every path that delivers a hop** (host, stand-in accelerator, `SIMD=1` and `0`): the whole `alg_ai3_aec_dfn.c` old against new on two recordings, hosts 48 kHz, 16 kHz direct, 16 kHz and 8 kHz bridged, with and without attenuation limit, comfort noise, a failed run and non-finite heads; the embedded bridge of `uvc_audio_live2a.c` against `AinrBridge` on 16 kHz / 512 and 16 kHz / 256, full and mask-only. PCM, model inputs and state reloads are identical.
- **Behaviour that changed, `alg_ai3_aec_dfn.c`.** (1) A hop DFN2 cannot deliver (a non-finite or out-of-range bin refused by `ainr_preprocess_freq()`, a bridge or sequence error) is one zero hop plus a restart of the AEC, the model stream, the bridge and the sample's synthesis in that same hop; the next hop equals a fresh run. Before, the bridged path returned an error on every later hop (silence until the end of the run) and the direct path zeroed the hop without a restart. (2) After a failed head readback the device state is reloaded with quantized zero before the next run. (3) Start-up and refusal messages name ainr and the bridge; the grid list no longer prints the lookahead pair; a missing or invalid ERB file is reported as `AINR_ERR_ERB_FILE` before any SDK call.
- **Behaviour that changed, `uvc_audio_live2a.c`:** none in the audio; a failed DFN2 hop now resets the DFN side inside `uvc_dfn_process()` as well as at the call site.
- **Memory (host arm64, from the size queries; the board ABI differs).** `alg_ai3_aec_dfn`, DFN2 part: 48 kHz host 301,968 -> 381,912 B (ainr pool 369,600 + conversion scratch 12,312; before: stage 104,976 + three ERB arrays 196,992); 16 kHz host on the 16 kHz model 88,048 -> 108,856 B; 16 kHz / 256 host bridged 419,872 -> 499,768 B; 8 kHz / 256 host bridged 421,504 -> 501,400 B. `uvc_audio_live2a`, bridge part of the arena at 16 kHz / 512: 132,260 -> 144,760 B (`AinrBridge` pool 132,448 + scratch 12,312); its ainr pool is unchanged.
- **Not verified here:** the board build, the real graph and quantized heads, the ERB files on the board, board RAM and timing.
- Docs: `docs/html/models/dfn2/{index,reference/grid,reference/board,reference/bridge}.html`, `docs/html/pipelines/reference/dfn2/configuration.html`, `AINR/DeepFilterNet2/reference/grid.md`.

## [Unreleased] — 2026-11-15 — The desktop DFN2 pipelines run the model stream (`dfn2_stream_*_freq`), not `DfnResStage`

`dfn/dfn_hop.h`, `dfn/dfn_hop_pool.h` and the two example mains (`mono_aec_dfn_res`, `4ch_aec_bf_dfn_res`) now drive the model-stream layer `stream/ainr/dfn2` (`dfn2_stream_init_ex()` in memory mode, `dfn2_stream_pre_process_freq_dual()` / `dfn2_stream_post_process_freq()`) and, on a host rate other than the model's, `stream/ainr/bridge`'s `AinrBridge` directly. `dfn_res_stage.o` and the `dfn_rate_bridge.o` adapter stay in `libaudio_pipeline_dfn.a` / `libdfn_res_stage.a` with their headers and API unchanged (spectrum-level callers written against them keep building); nothing in the examples calls them any more. The Python twins are untouched.

- **What the hop adds to the stream.** `DfnHop` holds the `DFN2Stream`, the `DFN2Model` (by value), six `n_bins` float buffers and the two scales: `dfn_hop_open_frame()` multiplies E and P by `dfn2_stft_scale(n_fft)` (the stream's spectrum entries are model scale), `dfn_hop_close_frame()` multiplies the output by `dfn2_stft_unscale(n_fft)` or writes zeros while `valid` is 0. On the bridge path the hop runs the same two calls on every frame `ainr_bridge_model_begin()` hands out, and `dfn_hop_end()` closes frames left open or never fetched as the identity. `DfnHopJob` is the stream's `DFN2HopIO` (`inputs`, `outputs`, `state_generation`).
- **API of the example headers** (header-only, no ABI). `DfnHopPoolPlan` carries `stream_cfg` (`DFN2StreamConfig`: grid, `atten_lim_db`, `erb_fwd`, `erb_inv`) and `model` instead of `stage_cfg`; `dfn_hop_pool_plan_init(&plan, &grid)` zeroes it. `dfn_hop_pool_build()` lost its three out-pointers (the bridge and the synthesis are `hop.bridge` / `hop.istft`; nothing in the tail owns heap, so there is no destroy). `dfn_hop_init_stage()` -> `dfn_hop_init_direct()`; `dfn_hop_stage()` is gone, with `dfn_hop_infer()`, `dfn_hop_set_atten_lim()`, `dfn_hop_state_generation()`, `dfn_hop_lookahead_frames()` and `dfn_hop_model_valid()` in its place. The bridge is addressed as `ainr_bridge_reset_synthesis(hop.bridge)`, `ainr_bridge_added_delay_samples(hop.bridge)`.
- **Bit-identical output.** `test_dfn_hop.c` (rewritten) holds `dfn_hop_process()` to two twins bit for bit with a model whose heads are a hash of the windows it is shown: a hand-written recipe on a second `dfn2_stream` (plain scalar scaling; for the bridge an explicit `ainr_bridge_*` frame loop), and the shipped `DfnResStage` / `DfnRateBridge`; host grids 48 kHz, 16 kHz/256, 16 kHz/512, 8 kHz/256 x six scenarios (plain, attenuation limit from the start, model failures, heads left unwritten, reset, attenuation change). A digest program built once against the previous headers and once against these (11 host/model grid rows incl. lookahead 2/4 and 0/0, the 16 kHz model and refused combinations x 6 models x 5 scenarios incl. refused blocks and a synthesis reset x one-call / phased / `check_finite`; output bits, return codes, state generation, model call and reset counts; 990 configurations, 433 distinct digests) prints identical values at SIMD=1, SIMD=0, `-DDFN2_MASK_ONLY` and both together. The stdout of `test_mono_dfn`, `test_4ch_dfn`, `test_dfn_res_stage`, `test_dfn_rate_bridge` and the skeleton runs is unchanged at SIMD=1 and 0 except the pool sizes and the banner below. The mains have no WAV input or output; they run silent fail-open hops.
- **Pool sizes (NE10; from `dfn_hop_pool_bytes()` and the skeleton banners).** The DFN tail is the stream, the six conversion buffers, then the bridge or the synthesis and the output spectrum: 48 kHz host 151,920 -> 223,568 B (164,304 + 12,312 + 42,832 + 4,104); 16 kHz/256 222,880 -> 294,480 B (164,304 + 12,312 + 117,856); 16 kHz/512 237,472 -> 309,072 B; 8 kHz/256 224,512 -> 296,112 B; 16 kHz/256 model on a 16 kHz host 63,280 -> 81,936 B. Skeleton totals: mono 16 kHz 463,680 -> 535,280, 48 kHz 932,096 -> 1,003,744, 8 kHz 436,944 -> 508,544, 16 kHz model 304,080 -> 322,736; 4ch 16 kHz 1,716,592 -> 1,788,192, 48 kHz 5,692,128 -> 5,763,776, 16 kHz model 1,556,992 -> 1,575,648. The growth is 71,600 B through the bridge and 71,648 B direct (18,656 B for the 16 kHz model): the stream's pool is 59,328 B larger than the stage's because it also holds the analysis and synthesis of its PCM entries, which the spectrum entries do not use, and the stage's internal 12,312 B conversion slab is now the hop's. `test_target_abi` has two more rows (`dfn2_stream` memory mode 164,304 B, `ainr_bridge` 117,856 B; 10 cases).
- **Behaviour that changed.** (1) The banner of the two mains says `the model stream on the host grid` where it said `the stage on the host grid` (the Makefile greps and `test_cli_exit_codes.sh` comments follow). (2) An inverse ERB matrix that is not a partition of unity is refused when the pool is built (`dfn2_stream_init_ex()` inside `dfn_hop_pool_build()`; the mains print `pipeline init failed`, exit 1), where the stage refused it in the pool query (`pool query rejected`, exit 1). (3) The stage's frame counters (`frames_in` / `commits` / `skips`) and `output_frame_index` have no counterpart: the mono and 4ch tests assert "no mid-stream reset" through the state generation (1 + one per warm-up frame at the end of the identity run) and the frame count through the bridge's `ainr_bridge_get_counters()`; "only the run behaviour commits" is shown with a half-mask model instead of the commit counter.
- **Build.** `pipelines/Makefile` and `pipelines/4ch/Makefile` compile every unit that includes `dfn_hop.h` / `dfn_hop_pool.h` with `DFN_HOP_INC` (`-I stream/ainr/dfn2 -I stream/ainr/bridge -I stream/common`); `dfn2_stream.o` is a member of `libaudio_pipeline_dfn.a` and `libdfn_res_stage.a`, built with `-DDFN2_STREAM_NO_STDIO` under `NO_STDIO=1`. `audit-no-stdio` (both Makefiles) lists the member and also forbids `fopen` / `fread` / `fclose` in the archive (the stream's ERB file loader). `audit_fp_contract.sh` has three more rows (`dfn2_stream.o` and the two example mains, which now carry the boundary scaling inline), 11 objects, fma-free at SIMD=1 and 0. `test_build_isolation.sh` needed no change (it pins no member list): 212 passed.
- **Tests.** `test_dfn_hop.c` also covers the four phased behaviours on the unit partition (and, with a half-mask model and one that writes and then fails, that only a successful run commits), the sequence guards, a refused block (NaN, 1e21 in E or P) leaving the generation untouched, reset equal to a fresh instance, `check_finite` (through the pool build, with its control), `model.reset`, the generation at start, after a warm-up frame, in the job and after a reset, and every refusal of the plan and the build (frames per call, descriptor of another grid in five fields, missing matrices, both missing, non-partition inverse, attenuation limit, host fft, front-end grid on both paths, a model grid the bridge cannot serve). Mutation: 31 mutants of `dfn_hop.h` / `dfn_hop_pool.h` (scale in and out, E and P swapped on either path, a failed result committed, frames left open or unfetched not closed, reset skipping the stream, the model callback, the bridge or the synthesis, a no-op attenuation setter, a constant or stale generation, the model never run, each plan and build refusal removed, `check_finite` dropped) are killed by the four rewritten tests, one mutant of `dfn2_stream.c` (partition check removed) by the new build row, and five test-level mutants (a hop reset before the generation check in the mono and 4ch tests, a whole-hop reset on the 4ch bridge event, a stream reset mid-run, the readme flow's phased path no longer running the model) by the test they target. One mutant survives as equivalent: not zeroing the output when `valid` is 0 (the direct path's spectrum is zeroed at init and reset, the bridge hands out a zeroed spectrum); emitting anything else there is killed. `test_target_abi`: `--write` / `--reject` round trip with a file of flipped hashes passes, and dropping the stream's `expected` descriptor fails it.
- **Docs.** `dfn/README.md`, the mono and 4ch DFN READMEs, `reference/call-flow.md`, `reference/build.md`, `4ch/reference/c-api.md`, `docs/maintenance/pipeline.md`, `docs/reference/board/sequence.md`, `docs/reference/array/beam.md`, the function map (PL-DFN-17 / 18) and the HTML pages `pipelines/mono/dfn2`, `pipelines/4ch/dfn2`, `pipelines/reference/dfn2/{configuration,integration}` (the three C examples there were recompiled with `-Wall -Wextra -Werror` and run with an identity model), `start/reference/{grids,integration}`, `models/dfn2/reference/{bridge,grid,core}`, `models/reference/onnx`.

## [Unreleased] — 2026-11-15 — The sample-rate bridge is a model-free library piece, `stream/ainr/bridge` (`AinrBridge`); `dfn_rate_bridge` is a thin adapter over it

The 8 / 16 kHz host to 48 kHz model bridge stays a product feature. Its signal path (host-grid WOLA, resampling up and down, 48 kHz analysis and synthesis, the FIFOs, the prefill, the added delay) moved out of `pipelines/dfn/dfn_rate_bridge.c` into `stream/ainr/bridge/ainr_bridge.{h,c}`, and the model went out of it: the caller runs the model between `ainr_bridge_model_begin()` (the two 48 kHz spectra and a zeroed output spectrum) and `ainr_bridge_model_end()`; the configuration carries only `lookahead_frames` (the model's delay L, which the added-delay formula needs). It is `stream/`, not `lib/`: it frames with `AcStft` / `AcIstft` and the resamplers, which `audit_no_transform_in_models.sh` forbids in the model cores (it does not scan `stream/`). The 48000 / 1024 / 512 / 513 model grid is still a set of constants (`AINR_BRIDGE_MODEL_*`); taking it from the caller is a later step.

- **`pipelines/dfn/dfn_rate_bridge.{h,c}` keep their path and public API** and are now `AinrBridge` + `DfnResStage`: `nn_begin` hands the bridge's spectra to the stage, `nn_end` puts the stage output back, and `finish` takes frames left open or never fetched as the identity as before. Proof (bit-identical on every non-refused path; see the refused-frame note below): `test_dfn_rate_bridge.c` (unchanged, 909 lines) and the 27 rows of `test_dfn2_rate_bridge_c_parity.py` (unchanged bodies; its compile list gained `ainr_bridge.c`) pass, and a digest program over the public API (3 host grids x no model / identity / half-mask x 4 stimuli x one-call and phased hops, with resets, a `reset_synthesis`, models that fail and models that write a NaN, frames left open; 72 configurations: output bits, host-hop tap, counters, state generation, delay, return codes) prints identical values before and after. Ten mutations of the adapter (model result dropped, lookahead 0, a reset skipped, left-open frames not taken as identity, unfetched frames not drained, E and P swapped at the stage, `reset_synthesis` a no-op, generation not forwarded, ...) are caught by the C test and the digest; the one not caught is the adapter's own descriptor hash omitting the bridge's hash (the bridge's rows pin that its hash differs per grid and lookahead).
- **One behaviour difference, on a refused frame, phased calls only.** When `dfn_res_stage_begin()` refuses a frame inside `dfn_rate_bridge_nn_begin()` (a host bin that reaches 48 kHz beyond 2^63 at model scale, or non-finite), `nn_begin` returns -1 as before. HEAD then left the bridge in the begun hop with the frame consumed (later `nn_begin` / `finish` calls went on); now the `AinrBridge` frame stays open, so `begin` / `finish` / `nn_begin` return -1 until `dfn_rate_bridge_reset()`. Measured by the digest program with two extra stimuli (a 1e21 apply lane for three hops, and a 1e21 estimate and apply bin for one hop; 90 of 108 configurations identical, the 18 that differ are exactly those two stimuli in the phased mode on all grids and models; `process()` is identical). `dfn_hop.h` documents a negative return as "reset the AEC front end, the comfort-noise generator and DfnHop together", so no caller continues past it.
- **`DFN_RATE_BRIDGE_LAYOUT_VERSION` 6 -> 7**: the carve is now the adapter struct, the `AinrBridge` pool, the stage pool (16 kHz / fft 256: 222,832 -> 222,880 B; 16 kHz / fft 512: 237,424 -> 237,472 B; 8 kHz / fft 256: 224,464 -> 224,512 B; the stage's 104,976 B is unchanged). Board samples size the pool with `dfn_rate_bridge_get_mem_requirements()` and need no change; `libaudio_pipeline_dfn.a` (and `libdfn_res_stage.a` of the 4ch build) gain the member `ainr_bridge.o`. A board Makefile that compiles `dfn_rate_bridge.c` from source needs `ainr_bridge.c` and `-I stream/ainr/bridge` as well.
- **`AINR_BRIDGE_LAYOUT_VERSION` 1**, descriptor version 1, 32-byte `AinrBridgeMemReq`; a config, size or descriptor rejection leaves the pool untouched.
- **`stream/ainr/bridge/tests/test_ainr_bridge.c`** (`make -C stream/ainr/bridge test`, `SIMD=0/1`; only audiolib is linked, the model is a delay of L frames on one lane): config and descriptor rows (also: the hash differs per grid and per lookahead), the frame schedule, the exact added delay for L = 0..3 against the independent sum and the impulse centroid, silence, the pass-band tolerance, lane isolation and bit-for-bit lane symmetry, reset equality, the phased calls on every hop of all three grids (every out-of-order use refused, the output spectrum opens zero), `reset_synthesis` against an audiolib oracle (bit-exact tap, counters unchanged, the output difference carries 0.99..1.01 of the cleared tail's energy). Of 19 mutations of `ainr_bridge.c` 17 are caught; the other two are equivalent: `finish` not refusing with a frame due or open is unobservable, because with the calibrated prefill the output FIFO cannot serve a hop at those points (probed on all three grids), so the FIFO check refuses the same calls.
- `audit_fp_contract.sh` audits `ainr_bridge.o` (FMA-free at `SIMD=1`) next to `dfn_rate_bridge.o`; `audit-no-stdio` checks the new member.
- Docs: `docs/html/models/dfn2/reference/bridge.html` (new), the DFN2 pipeline pages, `pipelines/dfn/README.md`, `docs/maintenance/pipeline.md`, `docs/reference/board/build.md`, `docs/c_python_function_map.md` (PL-DFN-07..13 point at `ainr_bridge.c`, new PL-DFN-19; 887 functions).

## [Unreleased] — 2026-11-15 — `DfnResStage` and `dfn2_stream_*_freq` are shown equivalent in CI; `dfn2_stream` init checks the inverse ERB table

No change to `DfnResStage` or to any pool size (`DFN2_STREAM_LAYOUT_VERSION` stays 3, the `dfn2_stream` descriptor and pool bytes are unchanged; the table is not part of either). The stage and the stream layer are two thin wrappers over the same `dfn2_prepost_*` calls; this makes the claim a test.

- **`pipelines/dfn/tests/test_dfn_stream_equivalence.c`** (in `make test`, so `SIMD=0` and `SIMD=1`): six grids (the catalog's 16k/256 and 48k/1024, 32k/512, 16k/512 and the lookahead pairs 2/4, 2/3 and 0/1 that no catalog row has) x ten scenarios x 400 hops (plain; attenuation limit from the start; model failure; unwritten heads; a NaN bin; E and P the same pointers; a 1e21 bin; the largest bin below 2^63 at model scale and the next float; a reset; an attenuation change). Per hop the model-visible windows, the output spectra (bit for bit), the state generation, the return codes and the valid flag of `DfnResStage` are compared with `dfn2_stream_pre_process_freq_dual` / `post_process_freq` in memory-mode init with the caller applying `dfn2_stft_scale` / `unscale`; a refused block must leave both generations untouched, refusals are counted per scenario, and a run that compares fewer than half of its hops with a frame fails. Two injected faults (a perturbed bin on one side, a reset on one side) must be reported. Shown to fail when the stage's limit is finite-only or 2^64, when the stream's range check is removed, when the apply spectrum is scaled differently, and when the stage ignores a failing model result. `pipelines/Makefile` now compiles `stream/ainr/dfn2/dfn2_stream.c` for this one test (`pipelines` may include `stream`, not the reverse); no library or archive member changes.
- **`dfn2_stream_init[_ex]` and `dfn2_stream_init_from_erb_file[_ex]` (hence `ainr_init*`) refuse an inverse ERB table that is not a partition of unity** (each bin's band weights, summed band-ascending in float32, exactly 1.0f: the rule of `dfn_res_stage.c` and `export_erb_matrix.py`) and return NULL (`AINR_ERR_ERB_FILE` from `ainr`). Before the check was added, every ERB matrix reachable here was scanned with the stage's rule: the exporter's `erb_matrix.bin` for 48k/1024/32, 16k/256/24, 32k/512/28, 16k/512/24, 16k/512/32 and 8k/256/24 (also exported from `AINR/DeepFilterNet2/output/dfn2_best.pth` and `dfn2_last.pth`: byte-identical to the fresh export), and 15 `fb_*.bin` files of 3 grids found under a review working directory: all pass, largest deviation of the float32 sum from 1 is 0. No shipped `fb_*.bin` of the board's `/mnt/sd/dfn2_blobs` is on this machine; the board files are what `export_erb_matrix.py` writes (it applies the same check before writing).
- **Tests.** New `stream/ainr/dfn2/tests/erb_partition.c` (`make erb-partition`, in `make test`, also built under `NO_STDIO=1`): the unit and a 0.75 / 0.25 partition are accepted; eight broken tables (every band owning every bin, all zero, a bin summing to 0 / 1 + ulp / 1 - ulp / 1.5, NaN, Inf) are refused by `init`, `init_ex`, `init_from_erb_file` and `_ex`; the descriptor is the same for a good and a broken table; the pool a refusal leaves takes a good table; `erb_partition SR FFT N_ERB FILE` loads a real file. Shown to fail with the check removed, with the last bin or the last band skipped, with a 1e-6 tolerance, with a transposed index and with NaN passing. `test_ainr_pool_gates` gains a non-partition ERB file row (`AINR_ERR_ERB_FILE`, ctx NULL, info untouched).
- **Stand-in ERB tables of the stream and ainr tests are partitions now** (they were all-ones or `((i + 3) % 11) * 0.02` inverses that init no longer takes; forward tables unchanged): `tests/erb_fixture.h` (band 0 owns every bin / a 0.75 / 0.25 split) for the dfn2 tests, `ERB_PATTERN` of `stream/ainr/tests/ainr_test_fixture.h`, the copies in `link_only_ainr.c` and `test_ainr_product.c`, and the zero-filled files of `test_ainr_graph_io.c` and `dump_work.c`. `stream/ainr/dfn2/tests/pool_size.c`'s four pcm-hash goldens (three full rows, one mask-only row) were re-recorded for its new table (0.75 / 0.25 split): they are identical with the pre-change `dfn2_stream.c`, differ under a 0.5 / 0.5 table, and the mask-only hash is now the same for NEON and scalar kernels (its separate NEON golden is gone). `test_ainr_dfn2.c`'s bypass test runs two hops: with a partition table a bypassed first hop is exactly the overlap-add's zero history half, so the pass-through shows from the second hop (a bypass forced to silence still fails it). `check_erb_file.c` uses an asymmetric partition for the file-vs-memory comparison.

## [Unreleased] — 2026-11-15 — `DfnResStage` refuses a block with a bin beyond 2^63 (model scale), like `dfn2_stream_pre_process_freq_dual`

`dfn_res_stage_begin()` only checked that the two input spectra were finite. A finite bin whose product with `dfn2_stft_scale(n_fft)` exceeds 2^63 (re^2 + im^2 no longer fits a float) was taken, put Inf into the feature normaliser, and every later hop then ran on poisoned features until reset (a model fed Inf/NaN produces non-finite heads, so each frame was a refused commit). The input check is now the one the stream layer applies: every bin times `dfn2_stft_scale(n_fft)` must be finite and at most 2^63 in magnitude, or the whole block is refused (`begin` returns -1) before a frame is consumed, with the pool, the counters and the state generation untouched. A normal signal is far below the limit, so nothing else changes. `pipelines/dfn/tests/test_dfn_res_stage.c` gains a row: one 1e21 bin among 80 normal hops took 81 frames as identity and moved the state generation 2 -> 83 before the change, and is refused alone (0 identity frames, generation 2 -> 2) after it; the four spectrum components in both signs are refused with every pool byte unchanged, and the limit is pinned on the scaled value (2x the limit refused, 0.75x taken). The check is local to `dfn_res_stage.c` (no new include path for the board build).

## [Unreleased] — 2026-11-15 — The stream graph port tables renamed `ai3_model_io*` -> `graph_io*` (no behaviour change)

Rename only. The lib `<model>_model_io` (CPU-side graph-boundary adapter) and the stream per-(model, grid, N) graph port table shared a word; the stream side is now `graph_io`, and the `ai3_` prefix (a target-SDK mark, not a library prefix) is gone from `stream/`. The `cc -E -P` output of every `stream/` and `pipelines/` source and test is identical to the old tree after the identifiers are mapped back, the non-test objects are identical, and both `dump_graph_io` dumpers print what the old ones printed.

- **Sample code and SDK Makefiles that include the old header names must switch** (no transitional headers are kept): `ai3_model_io.h` -> `graph_io.h`, `ai3_model_io_dfn2.h` -> `graph_io_dfn2.h`, `ai3_model_io_deepvqes.h` -> `graph_io_deepvqes.h`, `ai3_model_io_alignulcnet.h` -> `graph_io_alignulcnet.h` (in `#include` lines and in any dependency list). Symbols: `ai3_model_io_*` -> `graph_io_*`, `AI3_MODEL_IO_*` -> `GRAPH_IO_*`, `Ai3ModelIo*` -> `GraphIo*` (`GraphIoRow`, `GraphIoConfig`, `GraphIoAlignulcnetGrid`); the model keys are spelled like their `_ROWS` macros: `AI3_MODEL_IO_DEEPVQE_S` -> `GRAPH_IO_DEEPVQES`, `AI3_MODEL_IO_ALIGN_ULCNET` -> `GRAPH_IO_ALIGNULCNET`.
- **Files and targets.** Tests `test_model_io.c`, `test_ainr_model_io.c`, `test_aiaec_model_io.c` -> `test_graph_io.c`, `test_ainr_graph_io.c`, `test_aiaec_graph_io.c`; both `dump_model_io.c` -> `dump_graph_io.c`. Make targets `test-model-io`, `check-model-io-cxx`, `test-ainr-model-io`, `test-aiaec-model-io`, `model-io-dumper` -> `test-graph-io`, `check-graph-io-cxx`, `test-ainr-graph-io`, `test-aiaec-graph-io`, `graph-io-dumper`; environment variables `AINR_MODEL_IO_DUMPER`, `AIAEC_MODEL_IO_DUMPER` -> `AINR_GRAPH_IO_DUMPER`, `AIAEC_GRAPH_IO_DUMPER`. The lib `*_model_io` files and symbols (`dfn2_model_io_*`, `ulcnet_model_io_*`, `*_MODEL_IO_LAYOUT_VERSION*`) are unchanged.
- **Docs.** `CLAUDE.md` gains the vocabulary (`model_io` / `graph_io` / `wiring` / `stage` / `adapter` / `bridge` / `front`, and the two meanings of `LAYOUT_VERSION`); the three prose lines "the sample's AI3 run" read "accelerator run".

## [Unreleased] — 2026-11-15 — No `_Static_assert` left: every pin is a run-time check of a test; the ainr PCM-to-spectrum `_spec` family retired (use `AcStft` + `_freq`)

No numerics changed. A static assert emits no code: 74 of the 77 objects of every non-test source of `lib/`, `stream/` and `pipelines/` (compiled at `SIMD=1`, `SIMD=0` and, for dfn2 and ainr, mask-only) are byte-identical (`cmp`) before and after; the other three are `ainr.o` in those three builds, which lose the seven `_spec` entry points, and where 18 of the 22 remaining functions have identical address-normalised disassembly, three differ only in the immediate `2 -> 1` of the internal family enumerator (`ainr_get_mem_size_freq`, `ainr_init_freq`, `ainr_init_freq_from_erb_file`) and `ainr_postprocess_freq` now contains the former static `ainr_close_spectrum`. The PCM family and the `_freq` family produce the same bytes as before on every registered grid; all AINR/AIAEC test digests are unchanged at `SIMD=1` and `SIMD=0`.

- **`_Static_assert` removed everywhere (47 sites in 34 headers/sources, 4 in the embedded C drivers of three `AINR/tests/*.py`, and the `typedef char ...[... ? 1 : -1]` idiom in `test_aiaec_model_io.c`).** Together with the `#ifdef __STDC_VERSION__ >= 201112L` wrappers that existed only for them, and the macros that only fed them (`DFN2/DFN3_MODEL_IO_DESCRIPTOR_INTERIOR_FIELDS`, the `*_COUNT_*` macros and `*_FIELD_COUNT` enum of `dfn2_model_io.c` / `dfn3_model_io.c`). The facts are now checked by the tests that already include the headers (a failing check names the pinned fact and prints both numbers):
  - sizes: `DFN2PrepostMemReq`, `GTCRNProcessMemReq` (32), `DFN2ModelIoDescriptor` (72 bytes = 18 ints; its 12 external fields are refused by `validate()` when edited and its 6 interior ones are zero from `default()` and carried) in `AINR/tests/test_c_prepost.c`; the DFN3 counterparts in `AINR/tests/test_dfn3_stream.c`; `RNNoiseMemReq` (24) in `AINR/RNNoise-ERB/test/test_rnnoise_features.c`; `DeepVqePrepostMemReq` (32), `UlcnetModelIoDescriptor` (52) in `AIAEC/tests/test_prepost_kernels.c`, `UlcnetPrepostMemReq` in `test_prepost_chains.c`; `DFN2StreamMemReq` in `stream/ainr/dfn2/tests/pool_size.c`, `DeepVqeStreamMemReq` / `UlcnetStreamMemReq` in the two stream smokes; `DfnRateBridgeMemReq` / `DfnResStageMemReq` in `test_dfn_rate_bridge.c` / `test_dfn_res_stage.c`;
  - the 16-byte carve alignment of each module (`*_ALIGNMENT == 16` and equal to audiolib's `ck_align16_size(1)`), next to its size pin;
  - `dv_state_names` / `dv_state_names_out` cover `DEEPVQE_STATE_COUNT` (`test_prepost_kernels.c`, which includes the `.c`);
  - cross-layer constants: the wiring tables' layout numbers against the libraries' (`test_ainr_model_io.c`, `test_aiaec_model_io.c`, and a non-zero exit of the two `dump_model_io` dumpers, which the Python pins run without the C tests), `AiaecInfo` stays graph path + four ints, the front's delay-mode values and `AEC_DELAY_NUM_FILTERS_MAX == 5` against `aec.h` (`test_front_events.c`);
  - the embedded drivers keep theirs as `CHECK`s / an exit-3 check in their `main`;
  - the `-D` range guards of the Align-ULCNet and DFN examples (`ULCNET_EXAMPLE_DELAY_FRAMES`, `*_EXAMPLE_DELAY_NUM_FILTERS`) are gone: an out-of-range value now builds and is refused at start-up (delay depth by `ulcnet_grid_make`, exit 1, range pinned by `test_ulcnet_prepost_c.py`; bank size by the example's argument check, exit 2 with the message).
  Each distinct pin was shown to fail when its fact is changed in a scratch copy (grown struct, 32-byte alignment, a table row dropped, a layout number or enum value changed, a descriptor field left out of the list).
- **ainr `_spec` family retired.** `ainr_get_mem_size_spec`, `ainr_init_spec_from_erb_file`, `ainr_reset_spec`, `ainr_set_atten_lim_spec`, `ainr_state_generation_spec`, `ainr_preprocess_spec`, `ainr_postprocess_spec` and `AinrSpecCtx` are gone from `ainr.h` / `ainr_test.h` / `ainr.c` (the PCM family and `dfn2_stream`, which owns the STFT for the PCM family, are untouched). The STFT is audiolib's: analyse with `AcStft` (`ac_stft_config_default()`, `spec_scale` 1, PCM as `int16 / 32768`), feed `ainr_preprocess_freq()` with E == P, synthesise with `AcIstft`; after `ainr_reset_freq()` also `ac_stft_reset()` the analyser. Before the removal, a harness fed the same PCM to the HEAD `_spec` path and to `AcStft` + `_freq` (full 16k/256 N=1 and N=8, 48k/1024 N=1, lite 48k/1024 N=1, both ERB tables, `SIMD=1` and `0`; 40 / 12 blocks and the tail zeros per grid with commit, caller bypass, a refused commit from a NaN head, `atten_lim` 12 dB / NaN / 0, resets, a second preprocess and a set-while-open, a post without a pre): outputs, return codes, state-generation sequences and the `AinrWork` views are bit-identical, and the post-removal `_freq` dump equals the HEAD `_spec` dump. `ainr_get_mem_size_freq()` (105760 / 168368 / 369600 B, lite 309232) and the PCM pool (103696 / 160864 / 361392 B, lite 301024) are unchanged. `ainr_get_mem_size_spec()` had been 103696 / 166304 / 361392 / lite 301024. Tests: `test_ainr_spec_freq.c` (name kept) now compares the PCM family with `_freq` + the caller's `AcStft` / `AcIstft` bit-exactly, at N=1 on both grids and at N=8 over a committed and a bypassed block (this N=8 PCM comparison is new); the `_spec` vs `_freq` equality check, which only existed for the removed API, is gone; `test_ainr_tail.c` runs its lost-tail reproduction through `_freq`; `pool_gates`, `state_generation`, `model_io`, `atten_lim`, `dump_work` (`family` 1 is retired, `2` stays `_freq`) and `link_only_ainr` lose their `_spec` legs. `AINR_SYMBOLS` and the make comments follow.
- **Docs.** `docs/html/models/dfn2/index.html` and `reference/board.html` (two families, the migration recipe above, the pool table without the `_spec` column, the verification table), `docs/reference/board/memory.md` (`AecMemReq` is pinned by the AEC library's `test_memory_layout`), `docs/html/pipelines/reference/align-ulcnet/configuration.html` (the `-D` ranges are refused at run time). `docs/c_python_function_map.md` has no row for the retired functions.

## [Unreleased] — 2026-11-14 — Write-only getters, backend names and never-written fields of the AINR/AIAEC C layer removed; helpers made static; NR pool numbers re-measured in the docs

No numerics changed: the AINR digests (`test_c_prepost` `5871f9cf42a59a3b` NEON / `a802805e2fcc0e30` scalar, `test_nn_chains` `a7632b4049c9235c:25de8df4df69d614`, `test_dfn3_stream` `5b831894`, RNNoise SIMD digests `40ae2085af0a65aa` / `612b4e2935056190`, the `dfn2_runtime_grid_chain` digests `fc60a75b9318ac73` (48k/1024) and `2992f585b824da3e` (16k/256)) are identical to a build of the tree before this entry (SIMD=1 and 0), and the `test_target_abi` descriptor lines (`dfn2_prepost` 92,528 B, `dfn3_prepost` 90,384 B hash `d496d34e`, `ulcnet_prepost` 14,832 B, `deepvqe_prepost` 35,520 B, `aec` 700,480 / 594,944 B) are byte-identical. Pool sizes of every class are unchanged except through AEC and NR (below).

- **Removed accessors (no caller outside tests that only compared them with the value the test had supplied).** `deepvqe_prepost_hop_size/_num_bins/_skip_policy_name`, `ulcnet_prepost_hop_size/_num_bins/_frames_per_call`, `ulcnet_far_input_mode_name`, `dfn2_prepost_hop_size/_num_bins/_model_lookahead_frames/_layout_version`, `dfn3_prepost_hop_size/_num_bins/_grid/_model_lookahead_frames/_layout_version`, the macro `DEEPVQE_PREPOST_OUTPUT_NAME`, and the four log-line helpers `dfn2_simd_backend`, `dfn3_simd_backend`, `gtcrn_simd_backend`, `rnnoise_simd_backend` (the test banners print `SK_NEON_BACKEND_NAME`, text unchanged; the `backend=` token read by `test_dfn2_runtime_grid.py` stays). `dfn3_grid_status_text`, `rnnoise_grid_status_text`, `rnnoise_erb_window_elements`, `rnnoise_spec_window_elements` (the README example writes `RNNOISE_MODEL_INPUT_FRAMES * n_bands` and `* 2 * spec_bins`) and `ainr_init_spec` (`ainr_get_mem_size_spec` and `ainr_init_spec_from_erb_file` stay; `AINR_SYMBOLS` and `lib_api_call_reference.html` follow). The DFN3 grid probe in `AINR/tests/dfn_pairs.py` printed `dfn3_grid_status_text`; its text column is `-` for DFN3 now (assertion messages only).
- **Write-only struct members.** `DFN3PrepostOutputs.gru_hidden_elements` / `.df_convp_window_elements` (written in `frame_open`, read by nobody) and `DFN3Model.reset` (never called): `sizeof(DFN3PrepostOutputs)` 64 to 48 and `sizeof(DFN3Model)` 32 to 24 on arm64 (armv7a 32 to 24 and 16 to 12, from the review's clang record-layout dump, not re-measured here); both live on the caller's stack, so no pool changes and `DFN3_PREPOST_CARVE_VERSION` / `DFN3_MODEL_IO_LAYOUT_VERSION` stay 9. A positional `DFN3Model` initialiser `{user, infer, reset, descriptor}` must become `{user, infer, descriptor}` (`test_dfn3_stream.c` did); the `_Outputs` ctypes mirror of `test_dfn3_prepost_c.py` lost the same two fields. `DeepVqePrepost.hop` and `UlcnetPrepost.hop` (read only by the removed accessors): `sizeof(UlcnetPrepost)` 96 to 88 (arm64), `DeepVqePrepost` 456 to 456 (armv7a 268 to 264, same source as above); `get_mem_size` identical before and after at 16k/256, 16k/512 and 48k/1024 (UlcnetPrepost 7,664 / 14,832 / 29,168 B, DeepVqePrepost 18,112 / 35,520 / 70,336 B), so `ULCNET_PREPOST_CARVE_VERSION` (9) and `DEEPVQE_PREPOST_CARVE_VERSION` (8) stay.
- **`UlcnetStreamConfig` loses the eight state-registration fields** (`key_history`, `value_history`, `logit_history`, `h_gru` and their element counts; no caller ever set them, the stream layer only forwarded them to `UlcnetPrepostConfig`, whose registration stays): `sizeof` 88 to 24 (arm64). `build_flags_hash` loses the on/off term, so **`ULCNET_STREAM_LAYOUT_VERSION` 2 to 3**; a persisted `UlcnetStreamMemReq` is refused by `ulcnet_stream_init_ex()` after the bump (hash at 16k/256 `fa75db45` to `8ca507a8`, 16k/512 `cb99761d` to `50231f40`, 48k/1024 `05f11928` to `d3b49ee9`). Pool bytes unchanged: 53,648 / 103,312 / 202,640 B at 16k/256, 16k/512, 48k/1024.
- **Made static** (no caller outside their own file; header declarations and comments naming them go): `dfn2_model_io_push_erb_window/_push_spec_window`, `dfn3_model_io_push_erb_window/_push_spec_window/_validate_arrays`, `dfn3_norm_alpha`, `dfn3_apply_atten_lim`, `rnnoise_norm_alpha`, `rnnoise_grid_validate`. `dfn2_model_io_validate_arrays` and `dfn2_model_io_push_features` stay public.
- **Test fake accelerators** (`smoke_deepvqe_stream`, `smoke_ulcnet_stream`, `check_erb_file`, `smoke_dfn2_stream`, `variant_consumer`, `test_ainr_dfn2`) lose their unused `in` parameter; the `fake_accelerator`s are called directly, never through a pointer.
- **NR pool numbers (NR 4dda04a) and AEC layout 3 in the docs.** `anr_get_mem_size()` 8k/256 28,784 to 27,696 B, 16k/256 45,344 to 44,256, 16k/512 56,432 to 54,320, 48k/1024 144,576 to 140,416. Re-measured with `aec_nr_pipeline_static --print-mem-size`, `4ch_static --print-mem-size` and `4ch_external_static --print-mem-size`: mono total 8k 252,752 to 251,504 B, 16k/256 297,680 to 296,432, 16k/512 432,656 to 430,384, 48k/1024 967,776 to 963,456; delay-mode table (16k/256) MATCHED n=5 296,432, n=1 251,376, FIXED 240,992, EXTERNAL_ALIGNED 234,080 (computed with `aec_get_mem_size()` plus the same post-stage sizes); `4ch_static` Total 16k/256 1,551,728 to 1,549,840, 16k/512 2,743,168 to 2,740,256, 48k/1024 5,730,480 to 5,725,520 (post stage 58,256 to 57,168, 81,376 to 79,264, 193,584 to 189,424); `4ch_external_static` Total 760,064 to 758,176, 1,174,288 to 1,171,376, 2,607,168 to 2,602,208. `--no-nr` (252,176 B) and `--aec-only` (239,760 B) do not contain NR and are unchanged. The "from" totals are the builds before this round (AEC layout 2 and the old NR pool), so each delta also contains the AEC layout-3 change of -160 B mono / -800 B per array. The AEC part (`AEC_LAYOUT_VERSION` 3: mono 239,760 B, array 700,480 B FULL, lanes 4 x 142,720 B) was already in the tree and agrees with the binaries.
- **Docs.** `docs/c_python_function_map.md` loses rows `AINR-DFN2-21/-22`, `AINR-RN-15` and the symbols above from their rows (AINR 155 to 145 functions, 51 to 48 mapped; AIAEC 113 to 106, 36 to 35; TOTAL 881 to 864); the model manuals (DeepVQE-S, Align-ULCNet, DFN2, DFN3, GTCRN, RNNoise-ERB core pages, the DFN3/RNNoise index pages, the ONNX boundary page), `AINR/DeepFilterNet2/reference/export.md` (the window-slide helpers are internal now), `AINR/DeepFilterNet3/{README,UPSTREAM_ALIGNMENT}.md` and `AINR/RNNoise-ERB/README.md` follow. SE-root `aec_res_nr_flow.html` and `param_surface_overview.html`: every `aec.c:` / `aec_array.c:` line cite is re-pointed to the current line of the function it names (the cites were made against AEC acf02c5; the `aec_array.c` cng-seed cite now names the serial-change reseed in `aec_array_impl_process_residual`, the `max_saturation` cite the `out->formed_hop = NULL` line of `aec_array_impl_get_residual` it was written for); `lib_api_call_reference.html` drops `ainr_init_spec` and re-numbers the `ainr.h` rows.

## [Unreleased] — 2026-11-14 — Mask-apply C remnants of Align-ULCNet, Align-CRUSE and CAGCRN retired; `stream_msg.h` folded into `stream_bind.c`; model-config tables guarded against the library

No numerics changed: `stream_bind.o` is byte-identical before and after the fold (SIMD 1 and 0), and the shipped Align-ULCNet back end (`ulcnet_model_io_enhance_spectrum`) is untouched.

- **Retired C.** `lib/aiaec/ulcnet/ulcnet_mask_process.{c,h}`, `lib/aiaec/align_cruse/align_cruse_process.{c,h}` and `lib/aiaec/cagcrn/cagcrn_process.{c,h}` are gone (no caller in a pipeline, a stream or a sample; the Align-ULCNet math is `back_end` in `ulcnet_model_io.c`). `libaiaec_prepost.a` loses three objects (six remain: `ulcnet_process`, `ulcnet_model_io`, `ulcnet_prepost`, `ulcnet_accelerator_adapter`, `deepvqe_process`, `deepvqe_prepost`); `AIAEC/Makefile` (objects, `VPATH`, `-I`), `pipelines/Makefile` (`TARGET_ABI_SRCS`) and the `stream/aiaec/ulcnet/Makefile` comment follow. Align-CRUSE and CAGCRN have no C code any more; their Python models, exporters and ONNX contracts stay, and their manual pages now say the application does the mask multiply.
- **Test.** `AIAEC/tests/test_ulcnet_process_c.py` loses `test_shared_real_and_complex_masks_match_python` (it compiled only the retired files). `test_ulcnet_compressed_mask_helper_matches_python` becomes `test_ulcnet_back_end_matches_python`: the C driver now runs the shipped path (grid, descriptor, `ulcnet_model_io_init` with no state registered, `prepare`, the mask written into `outputs.output`, `enhance_spectrum`) and the same torch statement judges it (< 2e-5), built once as shipped and once with `-DSIMD_KERNELS_FORCE_SCALAR`. Flipping the sign of the real part in `back_end_neon` fails the shipped case, in `back_end_scalar` the scalar-twin case.
- **Python names.** `AIAEC/_export_common.py` `COMPOSITION` (read by nobody): Align-ULCNet now names `ulcnet_model_io_enhance_spectrum`, Align-CRUSE and CAGCRN `none`. `AIAEC/_streaming_export.py` `_C_BOUNDARIES` (written into the exported ONNX metadata key `c_prepost`): Align-CRUSE and CAGCRN carry `none` instead of the two deleted file paths; an export made before this change still names the deleted files.
- **Docs.** `docs/c_python_function_map.md` loses rows `AIAEC-MASK-01/02/03`, gap `D-36` and three file lines (AIAEC 117 to 113 functions, 40 to 36 mapped), `AIAEC-UL-02` lists the new test; the model manuals (index, Align-CRUSE, CAGCRN, ONNX boundary page, grids page), `docs/maintenance/models.md` and the three `AIAEC` READMEs drop the deleted headers and `aiaec_apply_*` names.
- **`stream/common/stream_msg.h`** is folded into `stream_bind.c` (the `StreamMsg` type and the five static inline helpers, unchanged, return types included); the header and its three Makefile dependencies are gone.
- **Model-config tables.** `test_aiaec_model_io.c` compares the DeepVQE-S row's state count with `DEEPVQE_STATE_COUNT`, `test_ainr_model_io.c` compares every DFN2 row's with `dfn2_model_io_descriptor_default().state_tensor_count` (full and mask-only builds); a wrong count in a row passed before and fails now. `ai3_model_io_alignulcnet.h` derives `bins`, `ta_bins` and the delay depth from `fft` and `sample_rate` instead of carrying them per row (`dump_model_io` output unchanged, 151 lines). `docs/maintenance/graph-wiring.md` says the tables list every registered configuration, graph delivered or not, and that the AIAEC production catalog has a graph only for DeepVQE-S 16 kHz / fft 512 / N=1.
- **Hashed sources.** `pipelines/tests/pyc_parity/pyc_parity.py` `SOURCES` adds `AEC/c_impl/src/aec_config_build.c` (the configuration builders moved out of `aec.c`).

## [Unreleased] — 2026-11-14 — dead parameters, members and locals of samples and tests removed; follows lib/aec's `include/internal/` headers and the removed per-hop trace

No numerics changed; the library archives are untouched. Every item is in a sample, a test or an inline helper of a private header.

- **Build paths.** `pipelines/Makefile`, `pyc_parity_pipelines.sh`, `pyc_parity.py` (hashed `SOURCES`) and `AIAEC/tests/test_deployed_frontend_contract.py` take AEC's private headers from `$(AEC_DIR)/include/internal` (they moved out of `src/`). `test_build_isolation.sh` SP-S10 and SP17 name `aec_post.o` / `aec_post.c` where they named the removed `aec_debug.o` / `aec_debug.c` (the member and the dirtied file only have to exist in both trees).
- **`ArraySampleHopView` / `ArraySampleOps`.** `microphones` and `reference` (written by the four samples' `*_setup`, read by nobody) are gone from the view, and `read`, `process`, `consume` and `finish` no longer take the view (all ten sample implementations and the four test probes ignored it): `ops->read(user, frame)` and so on. `setup` keeps the view.
- **Tests and helpers.** `ul4_in_pools()` loses the unused `req`; `dfn_example_rates_text()` returns `void` (both callers dropped the pointer); `Rig.bridge_mode` (test_dfn_hop), `SceneRun.hops` and `BridgeSceneRun.reset_hops_off` (test_mono_dfn), `GuardScene.name` with the `name` argument of `guard_scene_make()`, `AccStats.hops` / `.state[]` and the argument of `frames_at_hop()` (test_mono_ulcnet), the `bad_rig` / `bad_cm` tallies of `test_full_mode_identity`, and the `n_freqs` argument of the README-flow `hf_tilt_init()` are removed.
- **`test_ulcnet_nn_stage.c`** uses `ulcnet/tests/ulcnet_test_fixture.h` (grid table, `g_grid`, `HOP`/`BINS`, `test_io_descriptor()`) like the mono and 4ch tests instead of its own identical copy; the Makefile object rule lists the fixture.
- **Stream tests and header.** `AiaecFrontDelayMode` (an enum name nobody used as a type) is an anonymous enum, `Accel.have_gen`, two never-read arrays and `nb` in the AIAEC packet / spec-freq tests, `bad_grid` in `check_erb_file.c`, `did_reset` in the DFN2 `verify_vs_sample.c` and `INJ_NONE` in `test_ainr_freq_nonfinite.c` are removed.

## [Unreleased] — 2026-11-14 — Review round: the AIAEC front resets the AEC on a refused block; AEC test hooks never reach a pipeline build; `print-deps-sh` is not eval'd; NR flag set is four

No numerics changed (the 4ch, mono and stream outputs are the ones of the previous entry).

- **`aiaec_front_process_*()` / `aiaec_front_preprocess()`.** A refused block (a non-finite sample in hop h of the block, `AIAEC_FRONT_ERR_AEC`) used to leave the AEC advanced by the hops before h, so a retried or skipped block no longer matched the stream. The front now calls `aec_reset()` on that path, like the mono sample does; the next block equals a fresh front's output (`stream/aiaec/front/tests/test_front_refusal.c`, hop 0 and hop 1 of a two-hop block). `aiaec_front.h` says so.
- **`AEC_TESTING=0` is forwarded** to every AEC sub-make of `pipelines/Makefile`, `pipelines/4ch/Makefile` and `stream/aiaec/front/Makefile`. An exported `AEC_TESTING=1` (left over from an AEC test session) used to select the archive that carries the `aec_testing_*` hooks for a pipeline build and a `make publish`; the command-line value now wins. The archive path is unchanged for the default `AEC_TESTING=0`.
- **`lib/deps.mk print-deps-sh`** output is read one key at a time (`pipelines/scripts/test_build_isolation.sh`, `pipelines/tests/pyc_parity_pipelines.sh`) instead of `eval`; a checkout path with a space or a shell metacharacter is data. `pipelines/scripts/test_deps_env_safe.sh` (run by `make -C pipelines test`) checks it.
- **`docs/maintenance/float_io_compare.py`** fails a `--fail-max-abs` / `--fail-audio-lsb` limit when the two runs cannot be compared (different length or channel count, empty WAV, a head dump shorter than one hop); a B run that stopped halfway used to compare its common prefix and exit 0. `pipelines/tests/test_float_io_compare.py`.
- **NR flags.** `USE_OPTIMIZED_MIN_BUFFER` and `USE_INCREMENTAL_MIN` no longer exist in NR (one implementation); `pyc_parity_pipelines.sh` asserts the four remaining `USE_*` flags on the L2 build and none on L1.
- `4ch_external_static` and `mono/aec_nr_res/main.c` use `aec_level_name()` and `example_cli_int_option()`; `4ch_raw` and `4ch_static` drop includes nothing in them uses.

## [Unreleased] — 2026-11-14 — `aec_nr_pipeline` refuses unknown options and malformed positionals; the denoiser-input docs agree

- **CLI.** `aec_nr_pipeline` and `aec_nr_pipeline_static` ignored an unknown option and a positional `-1` and exited 0; they now print `unknown option or malformed argument '<token>'` and the usage to stderr and exit 2 (the policy of `4ch_external_static` and of the level parser), also on the `--print-mem-size` path. Every valid command line is byte-identical to before (stdout, stderr, exit code and output WAV, 10 lines on both binaries; the usage on argc < 4 and the `--print-mem-size` report); `--legacy-amin`, `--debug` and `--timing` stay accepted, as no-ops, after `--print-mem-size`. `test_mono_chain.c` has 13 unknown-argument rows per binary, the stderr text and `--print-mem-size` rows; `cli-and-example.md` and the README no longer say that unknown options are skipped.
- **Docs.** One statement everywhere: the shipped mono chain hands `base_spec` to the denoiser (it estimates on the spectrum its gain is multiplied on), the shipped 4ch chain hands `error_spec` (the beam E), the two differ only on output-guard hops; board samples outside the repository (`alg_audio_aec_nr.c`) feed `error_spec` in mono, a known difference. No code behaviour changed.

## [Unreleased] — 2026-11-14 — follows lib/aec's `Aec* aec_create()`, the 12-field `AecConfig`, opaque `Aec` and `AEC_LAYOUT_VERSION` 2; pool figures re-measured

No numerics changed in this repository; the AEC pool figures and the layout version moved with lib/aec.

- **API.** Every caller of the old `aec_create(Aec*, cfg)` (`test_mono_chain.c`, `conv_chain_mono.h`, `test_mono_dfn.c`, `test_mono_ulcnet.c`) takes the returned `Aec*`, and each `aec_destroy(x); free(x);` pair is a single `aec_destroy(x)`
  (the pointer is dead afterwards). `DA_NUM_FILTERS` -> `AEC_DELAY_NUM_FILTERS_MAX` and the three `DA_*` geometry macros of `test_mono_ulcnet.c` -> `AEC_DELAY_*` in 14 files; `MonoUlcnetConfig.filter_length` and the
  write to `AecConfig.filter_length` are gone (the length follows the rate). `AIAEC/tests/deployed_lane_dump.c` reads `n_partitions` and `delay_period_hops` from `aec_debug_status()` and the level's floors from
  `aec_level_*()`; the printed keys are unchanged. Six keys (`filter_length`, `enable_shadow`, `enable_highpass`, `enable_saturation`, `saturation_softclip_ref`, `live_observer_enabled`) have no reader in the shipped library any more
  and print the lane's fixed values until `AecDebugStatus` is widened.
- `test_mono_dfn.c` `run_native()`: the instance and the baseline chain are destroyed on every return path (the planted-defect rows returned early and leaked 6 `MonoDfn` and 6 `ConvMono`, 4.8 MB at exit); `leaks --atExit` now reports 0.
- **Guards.** `test_cng_aec3.py` reads `AEC_CNG_SEED` from `aec.h`; `pyc_parity.py` hashes `src/aec_impl.h`; `test_build_isolation.sh` SP-S17 gives `check-pins` the scratch clones' own HEADs and audiolib repo root (a dirty
  producer makes the adoption commit differ from the pin).
- **Docs.** `AEC_LAYOUT_VERSION` 2 in the nine policed lines and the prose and table lines that quote it; pool figures from `aec_get_mem_size()`, `aec_get_mem_breakdown()`, `aec_nr_pipeline_static`, `4ch_static`, `4ch_external_static`, `4ch_alignulcnet`,
  `4ch_aec_bf_dfn_res` and `mono_aec_dfn_res` (mono 16k/256 AEC 343,760 -> 239,920 B, total 401,520 -> 297,680 B; array FULL 720,352 -> 701,280 B; `4ch_static` Unit 1,512,576 -> 1,493,472 B; the per-filter step of
  `delay_num_filters` 5,728 -> 11,264 B single channel, 5,120 B array; SRP/BF helper 792,224 -> 792,192 B after audiolib's 16-byte `BF`/`SRP` shrink); the `AecConfig` tables of the mono and 4ch pages, `tuning.md`, `build.md`,
  `call-flow.md` and the array references describe the 12 public fields and the internal `AecTuning`; `aec_create()` snippets and destroy rules follow the new contract.

## [Unreleased] — 2026-11-14 — tidy of stream/, the model cores' test wiring and the pipeline tests

No numerics changed: every lib/ainr and lib/aiaec object file is byte-identical (SIMD 1 and 0, full and mask-only) and the AINR chain digests are unchanged.

- **Removed.** The 16 forwarding headers under `AINR/` and `AIAEC/` (`AINR/DeepFilterNet{2,3}/dfn*_*.h`, `AINR/GTCRN/gtcrn_process.h`, `AINR/RNNoise-ERB/process.h`, `AIAEC/Align_ULCNet/*.h`,
  `AIAEC/DeepVQE_S/*.h`, `AIAEC/aiaec_process.h`): the tests, the test-grid headers, the RNNoise-ERB Makefile and the docs include the core headers from `lib/ainr/<model>/` and `lib/aiaec/<model>/`.
  Seven accessors with no caller (`dfn2_stream_grid`, `dfn2_stream_layout_version`, `deepvqe_stream_grid`, `deepvqe_stream_layout_version`, `ulcnet_stream_grid`, `ulcnet_stream_frames_per_call`,
  `ulcnet_stream_layout_version`) and the two `dfn2_stream_*_mask_only` rename macros for them; the write-only `DFN2Stream.atten_lim_db` and `AinrCore.cfg` (`dfn2_stream_get_mem_size()` is 16 bytes smaller and `ainr_get_mem_size*()` 32 bytes smaller; no layout version moved; `pool_size.c` and `models/dfn2/reference/board.html` follow);
  `df_common_expand_mask{,_scalar,_neon}` in `dfn2_process.c` (the three sites call `skn_erb_expand_mask_f32`; `test_nn_kernels.c` tests the `skn_` names); `ULCNET_COMPRESSION_EXP`;
  `deepvqe_stream_internal.h` and `ulcnet_stream_internal.h` (folded into the `.c` files; the layout helpers are static); `ArraySampleHopView.program`; `dfn_fixture_bridge_added_delay`;
  `Audio_ALG/shared/`, `scripts/update_submodules.sh`, the unreachable `example_board_adapter` make alias.
  Tests: `test_ainr_work_shapes.c` (its refusal and second-block checks moved into `test_ainr_model_io.c`) and `test_aiaec_ports.c` (its disjoint-buffer, empty-slot, slim-`AiaecInfo` and three-family-info checks moved into
  `test_aiaec_model_io.c`).
- **CLI.** `4ch_external_static` parses its AEC and NR levels with `example_cli_level_option()`: one digit 0..3 only, and the error text is the shared "is not a level (accepted: 0..3)" (exit code 2 as before).
- **Shared.** `scripts/test_cli_exit_codes.sh` (run as `sh scripts/test_cli_exit_codes.sh PROGRAM BINARY`) holds the command-line exit-code cases that six Makefile recipes spelled out (the union of the lists); `ulcnet/tests/ulcnet_test_fixture.h` the grid table, descriptor and stand-in
  models of the mono and 4ch ULCNet tests; `stream/ainr/tests/ainr_test_fixture.h` and `stream/aiaec/tests/aiaec_test_fixture.h` the ERB writer and pool allocator of the stream tests;
  `ai3_model_io_shape_ok()` the axis check of `ai3_model_io_expand()`. The commented 8x8 catalogue rows of `stream/ainr/ainr.c` are in `docs/maintenance/ainr-release-data.md`.
- **Build and gates.** `test_build_isolation.sh` resolves AEC, NR and audiolib through `lib/deps.mk` (it used the `lib/aec` and `lib/nr` gitlink worktrees); `audit_dead_defines.sh` also covers `stream/`, takes AEC's and NR's
  sources from `lib/deps.mk` and reads `*_TEST_*` flags as test flags; `lib/deps.mk` lists its twelve includers.
- **Docs.** `models/dfn2/reference/grid.html` (the 8-field `DFN2GridSpec`, fixed normaliser constants, `DFN2_GRID_ERR_NORM_ALPHA`), stale file and test names in six pages, `c_python_function_map.md` (AINR-DFN2-05, counts), `dependencies.md`,
  `publish.md` Appendix A (`test-fp-contract`, `test-publish-manifest`, `target-abi-test`, the two audits).

## [Unreleased] — 2026-11-13 — the 4-channel sample layer is named `array_sample_*`; the flow wrapper is gone

Nothing was released, so the old names are gone without aliases.

- **Renames.** `mcab_sample_{beam,driver,post}` -> `array_sample_{beam,driver,post}` (types `ArraySample*`, macros `ARRAY_SAMPLE_*`); programs `mcab_raw`,
  `mcab_static`, `mcab_external_static` -> `4ch_raw`, `4ch_static`, `4ch_external_static` (the banners follow: `4ch_static: smoke PASS`); tests
  `test_mcab` -> `test_4ch_chain` (`All 4ch chain tests passed`), `test_mcab_sample_{beam,driver,post}` -> `test_array_sample_*`, `conv_chain_mcab.h` ->
  `conv_chain_4ch.h` (`CONV_4CH_CNG_NR_GAIN_FLOOR`), `mcab_test_rig.h` -> `array_test_rig.h`; make goal `test-mcab-sample-driver` -> `test-array-sample-driver`;
  `docs/reference/mcab/` -> `docs/reference/array/`. The Python twin keeps its `Mcab*` names.
- **Removed.** `mcab_sample_flow.{h,c}` (`McabSampleFlow`): the programs call `aec_get_mem_requirements()` + `aec_init_ex()` and `aec_destroy()` themselves;
  `array_sample_default_config()` moved into `array_sample_beam.h`. `mcab_sample_resolved_fft()` (the driver prints `cfg->fft_size`, which
  `aec_config_from_level()` always fills). The `Unit` / `PostStages` code that `4ch_raw.c` and `4ch_static.c` each carried is `array_sample_unit.h`.
  `ArraySampleBeamInfo.doa_analysis_frames` (always 1); the raw runner's JSON key `doa_analysis_frames` stays and counts the processed hops.
- **Tests.** `test_array_sample_beam.c` drops the array gain reset/level, the ramped level and the view/timing checks that the AEC repo's
  `test_aec_array*.c` cover; the ramp-vs-oracle chain test stays. The rig reads the sample rate from its own slot instead of `Aec` internals.
- **Build.** The 4ch Makefile no longer adds `-DAEC_STAGE_TIMING` (no 4ch TU reads it; `PROFILE` still reaches lib/aec). `test_build_isolation.sh` SP-S19..S25
  follow the new goal names; SP-S24 compares `test_4ch_chain.o` across the two audiolib variants.
- **Docs.** `docs/reference/array/config.md` §4.4, `errors.md` §5.4 and §8 and `integration.md` §1.1 link to the AEC docs instead of repeating them.

## [Unreleased] — 2026-11-13 — mono CLI levels are digits, one-off golden retired

- **Levels.** `aec_nr_pipeline` / `aec_nr_pipeline_static` take the AEC level (positional or `--aec-level`) and the NR level (`--nr-level`, alias
  `--nr-preset`) as one digit only (AEC 0 mild, 1 balanced, 2 aggressive, 3 no_floor; NR 0 mild, 1 moderate, 2 balanced, 3 aggressive). Any other
  text (a name, an out-of-range, signed, padded or empty value) exits 2 instead of falling back to balanced; `--sample-rate` and `--fft-size` are whole
  decimal integers (`example_cli_int_option()`, `example_cli_level_option()` in `common/example_cli.h`; `atoi()` is gone) and exit 2 otherwise.
  The banner's first line reads `(one pool)` / `(split pools)`; with `--legacy-amin` it adds `no residual-echo prior`. The failed-build message is
  the same in both binaries.
- **`--aec-only`** is unchanged: the lane is built with `AEC_RESIDUAL_STATE` and each hop runs `aec_process_linear()` then `aec_process_residual()`; the
  output is `aec_get_linear().linear_hop`. The residual stage's state update steers the linear filter's step size, so this hop equals the full chain's lane
  bit for bit. `test_mono_chain` compares it with a second `STATE` lane driven by the two stages and with the `FULL` lane of the chain (400 hops on its
  moving-echo scene at 16 kHz/256, 16 kHz/512 and 48 kHz/1024); the pool is the `Aec` block of the chain.
- **`enable_delay_est`** is no longer written by any program, test or the AIAEC front (`delay_mode` is the only delay input); `deployed_lane_dump`
  derives its `enable_delay_est` key from `delay_mode`.
- **Golden.** `ulcnet/tests/golden_ulcnet_stage_oldapi.txt` and `test_golden_oldapi()` are removed: the replay compared equal (1,920 hops per build, SIMD=1
  `apple-aarch64-neon` and SIMD=0 `apple-aarch64-scalar`) before the file was deleted.
- **`AEC_NO_ARRAY`** is no longer appended to the pipeline TUs' `CFLAGS` (nothing read it); it stays in `CFG_SIG` and in `AEC_FWD`.
  `test_target_abi` learns it through `TARGET_ABI_TEST_CFLAGS` (`-DTARGET_ABI_TEST_NO_ARRAY`): an array `libaec.a` must accept `n_mics = 4`, a
  single-channel one must refuse it, and the two `aec` descriptor carves are then the one-channel lane (8 cases in both modes).
  `make audit-dead-defines` passes.
- `stream/aiaec/front` builds its AEC with `AEC_RESIDUAL_STATE` (it reads `formed_hop` and `delay_state` only); the front tests stay bit-exact.

## [Unreleased] — 2026-11-13 — Callers follow the AEC public-header trims

- `aec_get_residual_gain()` is gone: `conv_chain_4ch.h` reads `c->frame.res_gain`, `test_4ch_dfn.c` / `test_array_sample_beam.c` drop the equality with the getter, `test_4ch_chain.c` checks the array's `residual` mode. `aec_cng_fill` (`pyc_drv.c`, `conv_chain_mono.h`) is included from `aec_internal.h`.
- `enable_delay_est` is no longer written anywhere (mono mains, `example_board_adapter.c`, `aiaec_front.c`, `pyc_drv.c`, the tests and the sample `alg_audio_aec_nr.c`); `AIAEC/tests/deployed_lane_dump.c` still prints `enable_delay_est=` for the Python contract, derived from `delay_mode`, and reads `duty.period_hops`.
- Pool figures re-measured after the AEC single-channel allocation trims: mono 16k/256 `Aec` 343,760 B (total 401,520 B), array 720,352 B `FULL` / 616,912 B `STATE` (Unit 1,512,576 B; ULCNet unit 2,490,416 B at 16k/512); the `docs/` and pipeline READMEs quote the new values. Docs no longer say that `AecPreset` / `ANR_MODE_*` stay in the headers. The pool figures in the earlier 2026-11-13 entries below (the `mcab` removal, the mono single-init entry, the review follow-ups) describe the state before this entry and are superseded by the values above and in the docs.

## [Unreleased] — 2026-11-13 — Level arguments are numbers; `aec_set_level()` has no `ramp_ms`; the 4-channel `replay` recomputes G_res again

- `aec_set_level(aec, level)`: the library's far-active ramp is a fixed 100 ms. `test_4ch_dfn.c`, `test_mono_dfn.c` and `test_array_sample_beam.c` follow the signature. `test_array_gain_ramped_level` is removed with the argument it tested (a 3000 ms ramp against an immediate twin); the AEC library's `test_runtime_preset.c`, `test_aec_res.c` and `test_aec_array.c` check the ramp and its landing, and `test_array_level_ramp_vs_oracle` (the same call as 100 ms before) still checks the chain.
- `mono/aec_nr_res/main.c` and `4ch/4ch_external_static.c` take levels as numbers: the AEC level (positional or `--aec-level`) and `--nr-level` / `--nr-preset` are a whole decimal string 0..3, anything else (a name included) exits 2. They used `anr_nr_level_from_name()` (removed from NR) and name tables; the mono program turned an unknown word into balanced. `test_cli_strict.sh` (742 command lines) and `test_mono_chain.c` use digits and gain refusals of names, `-x`, `1x`, `4`, `-1`; README, `docs/reference/mono/*`, `pipelines/reference/{tuning,build}.md` and the HTML pages follow.
- `tests/pyc_parity`: `pyc_drv` takes `aec=` / `nr=` as numbers (`pyc_parity.py` translates its names); the nine `l2.res4.*.replay` rows recompute G_res through a standalone gain stage again. The array's stage is internal, so `pyc_parity_pipelines.sh` builds the AEC archive with `AEC_TESTING=1` per layer and links it into a third driver, `pyc_drv_4ch_replay` (the `replay` mode of the 4-channel build; `pyc_drv_4ch` keeps only `chain`, `pyc_drv_mono` is unchanged); the production archive and the other two drivers carry no `aec_testing_*` symbol. The rows measure the shipped build against the std-math build over identical frames again (max |dG| up to 0.028 at 48k/1024, tolerance 0.05), and `a.nr4.*.replay_identity` shows the standalone stage reproduces the array's own gain bit for bit (`out_max_abs <= 0.0`).

## [Unreleased] — 2026-11-13 — Float I/O of the AI3 samples: documentation and board A/B tools

- The model is always fixed point; `FLOAT_IN` / `FLOAT_OUT` only make the hardware convert at the boundary, so the requested mode is not compared with the port types the SDK reports. `stream/common` keeps the SDK-free `stream_check_ports()` (`stream_bind.{h,c}`, port codes -101..-110) and the header-only `ai3_model_io.h`; the wiring fixtures of `test_stream_bind` live in `tests/stream_test_wirings.h`. The samples print `[AI3] float_in=.. float_out=..` and one line per port (`float`, `bits`, `scale_ratio`, `zero_point`) at start-up for the board A/B.
- Non-finite float heads (tests only): `test_ainr_state_generation` poisons the first / last element of each DFN2 head with NaN, +Inf and -Inf; `test_aiaec_block` (DeepVQE-S taps) and `test_aiaec_packet` (Align-ULCNet mask, both planes) add Inf at the first and last element. Each is a refused commit with the generation advanced. Mutating the DFN2 alpha scan, the coefficient or tap tail and the ULCNet mask tail fails them. `lib/ainr`, `lib/aiaec`, `stream/ainr` and `stream/aiaec` outside `tests/` are byte-identical.
- `docs/maintenance/graph-wiring.md` has the Float I/O section (these models inherit the state port types so both settings should be alike, mixed only warns, state ports follow the SDK with a type-agnostic zero fill, the start-up log); `docs/maintenance/float-io-board-ab.md` and `float_io_compare.py` are the board A/B checklist and comparison script. The sample-side edits are a separate change. Open on the board: whether the control bits also apply to state ports, numeric equality of the NPU conversion with the host call, and whether the existing `model.bin` files accept float I/O.

## [Unreleased] — 2026-11-13 — Review follow-ups: fma audit of the array instance, flow reset between beam and fuse

- `scripts/audit_fp_contract.sh` audits `aec_array.o` of the AEC checkout `lib/deps.mk` selects (lane-spectrum projection and fused PSD views, the arithmetic the deleted `mcab.o` row covered): 7 objects, 0 fma-class instructions.
- `test_mcab_sample_beam.c`: an `aec_reset()` of the flow between the beam half and `mcab_sample_beam_fuse()` makes the fuse answer `AEC_ERR_SEQUENCE` without resetting the helper (the frame counter goes on). The test fails when the `AEC_ERR_SEQUENCE` exemption in `mcab_sample_beam_fuse()` is removed.
- Pool figures of the `mcab` removal entry re-measured with `mcab_static --print-mem-size`, `4ch_alignulcnet`, `4ch_aec_bf_dfn_res` and `aec_get_mem_requirements()`: each was 256 B too high (array FULL 724,736 / 1,132,736 / 2,487,200).

## [Unreleased] — 2026-11-13 — `make AEC_NO_ARRAY=1`: the mono programs on the single-channel lib/aec

`AEC_NO_ARRAY=1` is forwarded to the lib/aec sub-makes (`AEC_FWD`, also in `audit-no-stdio-stack`), folded into `CFG_SIG` (no pipeline TU reads it; `test_target_abi` gets its own test flag). `make test AEC_NO_ARRAY=1` runs every mono program and
test against the array-less `libaec.a` (the default build is unchanged); `pipelines/4ch/Makefile` builds nothing under it (every build and test goal prints a SKIPPED line and succeeds; `clean` / `clean-all` still remove the trees, and `print-bin-dir` / `print-obj-dir` fail instead of printing a sentence as if it were a path; the two example Makefiles under `4ch/` skip the same way) and `make pyc-parity` is skipped.
The mono board samples (`alg_audio_aec.c`, `alg_audio_aec_nr.c`, `alg_ai3_aec_dfn.c`) need no code change; their headers note it.

## [Unreleased] — 2026-11-13 — the 4-channel flow is an `Aec` array instance; `mcab` is removed

The four-microphone flow moved into lib/aec: `AecConfig.n_mics = AEC_ARRAY_MICS` builds the shared delay estimator, the alignment ring, the
four lanes, the fusion buffers and, with `AecConfig.residual = AEC_RESIDUAL_FULL`, the one residual gain stage in a single pool
(`aec_get_mem_requirements()` + `aec_init_ex()`). `4ch/mcab.c`, `mcab.h`, `mcab_internal.h`, `libaudio_pipeline_mcab.a` (and its `print-lib-path`
and `lib` goals) are deleted; nothing was released, so the old names are gone without aliases. The array's own tests, including the golden that
reproduces the former flow bit for bit, live in the AEC repo.

- **API mapping.** `mcab_process_front()` -> `aec_process_linear()` + `aec_get_array()` (`AecArrayView`: `lane_error_spec`, `lane_base_spec`,
  `lane_formed_hop`, `lane_guard_active`, `aligned_ref`, `delay`, `post_reset_serial`); `mcab_fuse(W, E)` -> `aec_process_residual(&AecBeam{W, E})` +
  `aec_get_residual()` (`AecResidualView`: `error_spec`, `base_spec`, `near_spec`, `far_spec`, `r2`, `comfort_noise`, `res_gain`); `McabLibs`,
  `mcab_lane_config()`, `mcab_apply_lane_contract()`, `mcab_delay_estimator_config()` and caller-owned lanes/`DelayAec` are gone (the array derives
  its lanes); `McabMemReq` -> `AecMemReq` (`AEC_LAYOUT_VERSION` 1 replaces `MCAB_LAYOUT_VERSION` 26); `mcab_*` diagnostics -> `aec_array_*`;
  `MCAB_OK/INVALID_ARGUMENT/SEQUENCE_ERROR/DSP_ERROR` (0/-1/-2/-3) -> `AEC_OK/ERR_ARG/ERR_SEQUENCE/ERR_DSP` (0/-1/-3/-4) plus `AEC_ERR_MODE` (-2).
  `enable_post = 0` is `residual = AEC_RESIDUAL_STATE` (lanes only; `aec_process_residual()` and `aec_get_residual()` return `AEC_ERR_MODE`), `enable_post = 1`
  is `ESTIMATE` (fusion without a gain) or `FULL`.
- **Behaviour differences (decided).** A second `aec_process_residual()` on one hop is `AEC_ERR_SEQUENCE` with `FULL` (allowed with `ESTIMATE`);
  `formed_hop` and `echo_spec` are NULL in the fused view; `aec_get_linear()` on an array is lane 0's; `aec_apply_external_realign()` and
  `aec_process_linear_shared_far_from()` are refused on an array.
- **Gain and comfort noise.** The standalone `AecRes` of the 4-channel programs (`mcab_sample_res_input()`, `aec_res_process()`, `aec_res_set_level()`,
  `aec_res_reset()`) is the array's gain stage: `aec_get_residual_gain()`, `aec_apply_gain(aec, nr, spec)` and `aec_apply_cng(aec, nr, spec)` with the NR
  gain as the extra gain of both, and `aec_set_level()` for strength; the comfort-noise generator is the array's and restarts at `AEC_CNG_SEED` on a lane
  hard reset and on `aec_reset()`. `array_sample_post.h` keeps `array_sample_anr_gain()` (NR estimated on `error_spec`), `array_sample_post_reset_due()` and
  `ARRAY_SAMPLE_PSD_SCALE_INV`; the old `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR` is `AEC_CNG_NR_GAIN_FLOOR` and `MCAB_SAMPLE_CNG_RNG_SEED` is gone.
- **Sample layer.** `array_sample_default_config(&AecConfig, sr)` (balanced, MATCHED, bank 5, FULL, 4 mics) is in `array_sample_beam.h`; the array is placed by
  `aec_get_mem_requirements()` + `aec_init_ex()`. `array_sample_beam_*` take the `Aec*`;
  `array_sample_beam_fuse()` returns `AEC_ERR_MODE` for a `STATE` array and resets the helper on any refusal except SEQUENCE and MODE.
- **Bit-exactness.** The five production programs (`4ch_external_static`, `4ch_static`, `4ch_raw`, `4ch_aec_bf_dfn_res`, `4ch_alignulcnet`) match the
  previous tree's outputs byte for byte at SIMD=1 and SIMD=0. The 4ch tests, `pyc_drv.c` `chain`/`replay`, `test_target_abi`, `test_build_isolation.sh`
  and `pyc_parity_pipelines.sh` follow the array API; the groups of the old `test_mcab.c` that test the flow itself moved to the AEC repo (the list was in a one-off document retired on 2026-11-13), the chain-level ones stay.
- **Pools** (NE10, SIMD=1, MATCHED n=5; bytes; measured against the sibling AEC checkout including its 2026-11-13 lane-pool trims, which are bit-exact).
  Array (`aec_get_mem_requirements()`, `FULL`): 16 kHz/256 724,736, 16 kHz/512 1,132,736, 48 kHz/1024 2,487,200
  (lanes 593,152 / 980,416 / 2,157,888; before: lanes 803,968 / 1,318,848 / 2,936,832, plus estimator, mcab pool and a separate `AecRes` of 7,904 /
  14,048 / 29,408). `mcab_static` Unit (array + SRP/BF helper): 1,706,240 -> 1,516,960, 3,007,040 -> 2,702,688, 6,329,312 -> 5,612,608; Total
  1,772,928 -> 1,575,216, 3,103,504 -> 2,784,064, 6,554,368 -> 5,806,192. `4ch_alignulcnet` unit pool (`STATE`): 16 kHz (fft 512) 2,994,640 -> 2,676,240,
  48 kHz 6,304,624 -> 5,558,512. `4ch_aec_bf_dfn_res`: 16 kHz 1,939,200 -> 1,740,880, 48 kHz 6,518,960 -> 5,768,640.
- **Docs.** `docs/reference/array/*`, `docs/reference/board/*`, the 4ch and mono HTML pages, `4ch/README.md`, `4ch/reference/c-api.md`, the two 4ch example
  READMEs, `reference/{tuning,call-flow,build}.md` and the function map follow the array API; the examples of the two quick starts compile with
  `-Wall -Wextra -Werror` and run.

## [Unreleased] — 2026-11-13 — mono examples and tests follow lib/aec's single-init, three-entry API

The single-channel programs, tests and docs use `AecConfig.residual` and the new entry points; nothing was released, so the removed names
(`aec_process_context`, `aec_get_res_context`, `aec_get_output_base`, `aec_get_linear_context`, `AecResContext`, `AecOutputBase`,
`AecLinearContext`, `enable_res`, `return_res_context`, `spatial_linear_context`, `AEC_CONTEXT_*`) are gone without aliases. The 4-channel
programs, `mcab` and `AecRes` were migrated in the entry above.

- **Lane.** `derive_lane()` sets `residual = AEC_RESIDUAL_FULL` (`mono/aec_nr_res`, `mono/aec_dfn_res`) or `AEC_RESIDUAL_STATE`
  (`mono/alignulcnet`, formerly `LINEAR_ONLY`). Per hop: `aec_process_linear()` then `aec_process_residual(aec, NULL)`, then
  `aec_get_residual()` (rendering base `base_spec`, `res_gain`, `r2`, `comfort_noise`, `formed_hop`, `usable_linear`, `guard_active`) and
  `aec_get_linear()` (`linear_hop`, `far_power`, `delay_state`, `aligned_far_hop`); a non-`AEC_OK` return is a refused hop.
- **Gain and comfort noise.** The hand-written `sk_min_f32` / `sk_capply_gain_f32` / `aec_cng_fill` step and the caller's `rng` are replaced by
  `aec_apply_gain(aec, nr, spec)` and `aec_apply_cng(aec, nr, spec)` with the NR gain as the extra gain of both (NULL without a denoiser; the
  DFN2 chain passes NULL). `--no-res` uses `aec_apply_gain_spec()` with the unity vector and no comfort noise. The comfort-noise
  generator is the `Aec`'s own and is reseeded by every hard post-chain reset and by `aec_reset()`; the application still clears its
  overlap-add tail when `aec_post_reset_serial()` moves. The `total gain` vector (and the DFN2 chain's unity vector) are no longer carved.
- **Bit-exactness.** Output WAVs and `DUMP_CTX` seam dumps of `aec_nr_pipeline` / `aec_nr_pipeline_static` (42 runs over 16 kHz / 8 kHz / 48 kHz
  inputs and the `--no-nr`, `--no-res`, `--no-cng`, `--aec-only`, `--legacy-amin`, level and `--fft-size` options, SIMD=1 and SIMD=0) equal
  the runs of the previous tree byte for byte; the `pyc-parity` mono rows (`d.mono.*`) and their seam dumps are unchanged.
- **Tests.** The hand-written oracle `conv_chain_mono.h` keeps its own min / apply / `aec_cng_fill` and generator, so it stays independent of
  `aec_apply_gain` / `aec_apply_cng`; the planted defects that need the caller to own the generator (`SAB_NO_RESEED`, `SAB_OLD_SEED`,
  `MDFN_SAB_NO_RESEED`) draw through `aec_apply_cng_spec()` with the test's own state, `SAB_ERROR_SPEC` through `aec_apply_gain_spec()` on
  `error_spec`. The `--aec-only` oracle is `aec_process()` on a second `Aec` with `AEC_RESIDUAL_STATE` (below `FULL` it returns the linear hop).
  `test_readme_flow` folds its extra stage (the HF tilt) into the extra gain given to both calls.
- **Pools** (NE10, SIMD=1, MATCHED n=5; bytes). `aec_nr_pipeline_static --print-mem-size`: 8 kHz 310,720 -> 310,240, 16 kHz/256 424,288 -> 423,808,
  16 kHz/512 576,128 -> 575,136, 48 kHz 1,356,352 -> 1,354,336 (the AEC block is 48 B larger in lib/aec, the gain buffers one vector smaller).
  `mono_aec_dfn_res` banner: 16 kHz 590,976 -> 589,968, 48 kHz 1,322,896 -> 1,318,816, 8 kHz 495,600 -> 494,592.

## [Unreleased] — 2026-11-12 — NR gain calls use `anr_process_gain_with_residual_psd()`

The pipeline code, tests and docs follow lib/nr's gain API (`anr_get_gain()` and the `gain_out` argument are gone): the denoiser
gain of a hop is now the return value of `anr_process_gain_with_residual_psd(nr, spectrum, residual_psd_audio)`
(`residual_psd_audio` = the AEC `r2` divided by 32768^2, NULL when the linear estimate is not usable), NULL for a refused hop.
Touched: `4ch/mcab_sample_post.h` (`mcab_sample_anr_gain()`), `mono/aec_nr_res/main.c` (`mono_nr_gain()`) and
`example_board_adapter.c`, the chain tests and the pyc-parity driver. Gains are unchanged; the locals that held the residual PSD are
named `residual_psd_audio`.

## [Unreleased] — 2026-11-12 — `ulcnet_nn_stage` takes and returns spectra

The stage no longer owns an STFT or an ISTFT. The application analyses the raw far hop (and the error hop when its error is a time-domain
signal) with its own `AcStft` (audiolib defaults on the grid) and synthesises the output spectrum with its own `AcIstft`
(`AC_OLA_ACCUMULATE`); nothing was added to a library. Nothing was released, so the old names are gone without aliases.

- **API.** `ulcnet_nn_stage_begin_spec(stage, err_spec, far_spec, delay)` (`Complex[n_bins]` each, audio scale) replaces `_begin`, `_begin_complex`
  and `_begin_time`; `ulcnet_nn_stage_end_spec(stage, out_spec)` replaces `_end` and writes the accepted model spectrum, else the error spectrum
  (identity), without synthesising; `ulcnet_nn_stage_process(stage, out_spec)` is the begin-less hop after `begin_spec`.
  `ulcnet_nn_stage_mem_size(grid)` / `_init(mem, bytes, grid)` lose the `err_analysis` argument. `ULCNET_NN_STAGE_DSP_ERROR` is gone (the stage
  has no synthesis to fail).
- **Reset.** `ulcnet_nn_stage_reset()` clears the model, the alignment memory, the frame index and a pending reprime; it no longer clears
  transform history. The application resets its `AcStft` / `AcIstft` together with it.
- **Pool.** The stage block is the state plus six planar spectra (16 kHz / fft 512: 6,320 B, was 34,416 B, or 40,656 B with the error
  analysis; 48 kHz / fft 1024: 12,464 B, was 67,696 B / 80,080 B). The application adds one `AcStft` per analysed signal and one `AcIstft`.
- **Bit-exactness.** The stage plus the caller's `AcStft` / `AcIstft` reproduce the hops of the previous stage bit for bit:
  `ulcnet/tests/golden_ulcnet_stage_oldapi.txt` holds the FNV-1a-64 hash of every output hop (6 model/error-source variants, 320 hops, alignment
  events changed / solid false-to-true / relock, a reset mid-stream, both grids, macOS arm64 with NEON and scalar), dumped from the previous sources;
  `test_ulcnet_nn_stage` replays it. Any other host (the window table comes from libm) reports SKIP for the golden rows and runs the other checks.
- **Callers.** `mono/alignulcnet/main.c` and `test_mono_ulcnet.c`, `4ch/alignulcnet/main.c` and `test_4ch_ulcnet.c`, `tests/test_readme_flow.c`
  own the transforms (mono: error and far analyses; 4ch: far analysis only, the beam spectrum is the error); `AIAEC/tests/test_deployed_frontend_contract.py`
  checks the new seam calls in `main.c`.

## [Unreleased] — 2026-11-12 — `stream/ainr` and `stream/aiaec` customer API: slim Info, graph wiring in the sample's model configuration

Nothing was released, so the old names are gone without aliases.

- **Slim Info.** `AinrInfo` / `AiaecInfo` carry `graph_path`, `trim_samples`, `queue_frames` and `tail_zero_blocks` only; they replace
  `*_get_bind_info*`, `pcm_delay_samples` and the queue-depth accessors. N is not reported: the sample derives `N = cfg.hop_in / (cfg.fft / 2)`
  from the config (the client still sets only `sr`, `fft`, `hop_in`). `trim_samples` = `(queue_frames + 1) * model_hop` for DFN2 (full
  16 kHz/256 N=1 128, N=8 1024; 48 kHz/1024 full 1536, lite 1024), DeepVQE-S `N * model_hop`, Align-ULCNet `model_hop`. The initial state
  generation is read with `*_state_generation()`.
- **Init signature.** `int ainr_init(cfg, pool, pool_bytes, &ctx, &info)` and `aiaec_init(...)` (and `_spec` / `_freq`) return 0 or a negative
  `AINR_ERR_*` / `AIAEC_ERR_*`; there is no `err_out`, no `expected` descriptor and no `*_ERR_STALE_POOL`. Init recomputes the size and refuses a
  too-small or misaligned pool (`*_ERR_POOL`); a larger pool is accepted. Codes renumbered after the stale-pool gate was removed: `AINR_ERR_ERB_FILE`
  -9 -> -8, `AINR_ERR_SEQUENCE` -10 -> -9, `AINR_ERR_NONFINITE_INPUT` -11 -> -10 (ends at -10); `AIAEC_ERR_SEQUENCE` -8 -> -7,
  `AIAEC_ERR_INPUT_KIND` -9 -> -8, `AIAEC_ERR_INPUT_SIZE` -10 -> -9 (ends at -9). The model-io helper's own codes, `AI3_MODEL_IO_ERR_*`, are
  -301..-306. `ainr_preprocess*` returns `AINR_OK` (0) on success instead of 1, like `aiaec_preprocess*`.
- **No graph wiring in the libraries.** `AinrInfo.wiring`, `ainr_fill_wiring()`, the `aiaec_core_wiring()` / `aiaec_describe_*` generators, the
  test wiring accessors and every state-dimension constant that existed only for graph wiring are removed. The libraries keep what the
  work views need: `AinrWork` / `AiaecWork` carry the block's signal and head pointers, element counts and `[b,c,h,w]`, the generation, the
  action and, for Align-ULCNet at N > 1, the `frame_reset` control values the stream computes from the hop events as a fifth signal
  (`[1,1,1,N]`; hop 3 of a packet alone gives `00100000`). `ainr.h` / `aiaec.h` include `stream_mem.h` only; `libainr.a` / `libaiaec.a` no
  longer compile or link `stream_bind.c`.
- **Wiring in the sample.** Every converted graph is inputs = signals then states, outputs = heads then states, state k = input
  `signal_in_count + k` <-> output `head_out_count + k` (counts: DFN2 full 2/3, lite 2/1, DeepVQE-S 2/1, Align-ULCNet 4/1 at N=1 and 5/1 at
  N>1). The shapes of every registered `(model, sr, fft, N)` row, state shapes included, are fixed tables in `stream/ainr/ai3_model_io_dfn2.h`,
  `stream/aiaec/ai3_model_io_deepvqes.h` and `ai3_model_io_alignulcnet.h` (header-only, not included by `ainr.h` / `aiaec.h`); the shared
  `stream/common/ai3_model_io.h` (tested by `make -C stream/common test-model-io`) expands a row, pairs states by count, calls the
  checker, sizes buffers and checks each block's work against the row before any AI3 call (a mismatch is no run, a failed post). A config
  without a row, or one that does not match the selected graph, is refused at start-up before any SDK call.
- **Shared checker and memory type.** `stream/common/stream_bind.{h,c}` (`StreamWiring`, `StreamDeviceDesc`, `stream_check_ports()`, codes
  `STREAM_PORT_ERR_*`, -101..-110) is the SDK-free sample-side checker; the `AINR_PORT_*` / `AIAEC_PORT_*` codes are gone. `StreamMemReq
  {bytes, alignment}` moved to `stream/common/stream_mem.h` and no longer depends on the wiring types.
- **Renames.** The `aiaec_*_block` entries are `aiaec_*` (`AiaecCtx`, `AiaecInput` / `AIAEC_INPUT_*`, `AiaecWork`, `AIAEC_SUBMIT` /
  `AIAEC_BYPASS`, `aiaec_front_preprocess`); `AinrHopIO` is `AinrWork`. The `ainr_*_from_erb_file` entries are in the test-only
  `stream/ainr/ainr_test.h`; `aiaec_internal.h` is likewise not a customer header. The three DFN2 families share one `AinrCore`.
- **Tests.** `test-ainr-work-shapes` (replaces `test-ainr-ports`) checks the `AinrWork` views of real pre calls against tables spelled out from the
  exporter; `AIAEC/tests/test_aiaec_bind_info_pin.py` and `AINR/tests/test_dfn2_port_order.py` compare the work views of real pre calls (dumped by
  `make -C stream/aiaec bind-dumpers`, `make -C stream/ainr [MASK_ONLY=1] work-dumper`) with fresh exports and assert the exporters' name order and
  state pairing. The same two tests compare every row of the model tables (`ai3_model_io_<model>.h`, dumped by `make model-io-dumper`) with a fresh export.
- **zero_point rule for every model.** `STREAM_PORT_ERR_STATE_ZERO_POINT` (-110) also covers DFN2: a fixed-point state port with
  `zero_point` != 0 at either end is refused; float state ports ignore it.
- **DFN2 pool.** The inner STFT and ISTFT borrow the stream's one FFT instance (`DFN2_STREAM_LAYOUT_VERSION` 2 -> 3).
  `ainr_get_mem_size()` bytes (PCM / `_spec` / `_freq`), before -> after: full 16 kHz/256 N=1 118032 / 118032 / 120112 -> 103728 /
  103728 / 105792; N=8 175200 / 180640 / 182720 -> 160896 / 166336 / 168400; full 48 kHz/1024 414128 / 414128 / 422352 -> 361424 /
  361424 / 369632; lite 353760 / 353760 / 361984 -> 301056 / 301056 / 309264 (arm64 host, SIMD=1, ne10). The public
  `AINR_DFN2_*_LAYOUT_VERSION` and `AIAEC_*_LAYOUT_VERSION` macros are gone with the descriptor; the AIAEC pools are unchanged.

## [Unreleased] — 2026-11-11 — Align-ULCNet packet graph (layout 19) in the C model I/O, prepost and stream

`lib/aiaec/ulcnet` and `stream/aiaec/ulcnet` run the exporter's N > 1 packet graph (`PACKET_LAYOUT_VERSION` = 19):
`ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION`, `UlcnetModelIoDescriptor.frames_per_call` (the struct is 52 bytes now; 0 or 1
means layout 18, 2..64 layout 19, any other pairing is refused), `ulcnet_model_io_prepare_packet` /
`_enhance_packet`, `UlcnetPrepostConfig.frames_per_call` with `ulcnet_prepost_pre_process_packet`, and
`ulcnet_stream_pre_process_packet_{f32,freq}`. One-frame callers (layout 18, N = 1) keep their call sequence and tensor layout (PCM
comparison below); `ulcnet_nn_stage` refuses a packet descriptor. Pool sizes grew by the extra state fields, so
`ULCNET_PREPOST_CARVE_VERSION` 8 -> 9, `ULCNET_STREAM_LAYOUT_VERSION` 1 -> 2 and the three `AIAEC_*_LAYOUT_VERSION`
bumped; a `MemReq` queried before this change is refused by `init_ex()`. Align calib keeps the real tail hops when
`--events` is given (padding slots carry `frame_reset` 0); without `--events` only whole packets are captured.

`stream/ainr/ainr.h` adds `ainr_set_atten_lim` (`_spec`, `_freq`): runtime attenuation limit, default the catalog row's 0 dB, callable between blocks only,
kept across `ainr_reset`; pool size and layout are unchanged.

N = 1 audio was compared only against the `5c61d2b` sources (identity mask,
16 kHz FFT 256 and 512, 24 hops, no event and one event hop): at most 1 PCM16 step apart; nothing wider is claimed.

## [Unreleased] — 2026-11-09 — Pipeline layout: top-level split into `mono/` and `4ch/`, one directory per application chain

The flat layout mixed two things per NN family (`dfn/`, `ulcnet/`: shared stage infra plus a
mono *and* a 4-channel example subdirectory each) with two already-separate conventional chains
(`mono_aec_nr_res/`, `4ch_aec_bf_nr_res/` — the latter was also the mcab flow's actual home, reached
into by the other two 4-channel examples via `#include "4ch_aec_bf_nr_res/mcab*.h"`). Renamed with
`git mv` (history preserved) to one topology-first split:

```
pipelines/mono/{aec_nr_res,aec_dfn_res,alignulcnet}/   # was mono_aec_nr_res/, dfn/mono_aec_dfn_res/, ulcnet/mono_alignulcnet/
pipelines/4ch/{aec_bf_dfn_res,alignulcnet}/            # was dfn/4ch_aec_bf_dfn_res/, ulcnet/4ch_alignulcnet/
pipelines/4ch/                                          # was 4ch_aec_bf_nr_res/ itself: the mcab flow (mcab*.{c,h},
                                                          # pipeline.py, evaluate_recordings.py, Makefile) now sits one
                                                          # level up from the two example chains that reach into it,
                                                          # instead of being reached into through a sibling-example's own
                                                          # directory name
```

`pipelines/dfn/` and `pipelines/ulcnet/` keep only their shared, model-family infra (the stage/
bridge code, Python twins, shared tests) now that both chains per family moved out. No source
code changed — this is a path move plus the follow-on fixes it forced:

### Fixed
- `pipelines/mono/aec_nr_res/pipeline.py`'s `_ROOT` was `dirname()` x3 off `__file__`, correct
  when the file was one level under `pipelines/` (`mono_aec_nr_res/pipeline.py`); moving it one
  level deeper (`mono/aec_nr_res/pipeline.py`) needed x4, or it silently resolved to
  `pipelines/lib/nr/...` instead of `Audio_ALG/lib/nr/...` and `load_config` raised
  `FileNotFoundError` (caught by `pipelines/tests/pyc_parity_pipelines.sh`'s `a.nr4.*`/`d.mono.*`
  rows going `NOT_RUN`, not by any Python import error). The other moved `pipeline.py`/test files
  kept the same nesting depth under `pipelines/` (e.g. `dfn/mono_aec_dfn_res/` -> `mono/aec_dfn_res/`
  is still two levels down), so none of them needed this fix.
- Six doc pages still quoted `DFN_RES_STAGE_LAYOUT_VERSION`/`DFN_RATE_BRIDGE_LAYOUT_VERSION = 5`
  and `DFN2_PREPOST_CARVE_VERSION = 11` (the prior entry below bumped both to 6 and 12 in code but
  missed these pages); `docs/html/models/dfn2/reference/grid.html`'s `build_flags_hash` field list
  also still named the seven removed feature-normaliser fields instead of the two norm alphas and
  two STFT-scale functions that replaced them. All now match
  (`pipelines/tests/test_doc_layout_versions.py`).

### Notes
- Every `make -C <dir>` directory argument, `#include "<old>/..."` path, Python dotted-module
  import string (`pipelines.<old>.pipeline`), and `pathlib`/`os.path` path-join literal referencing
  one of the six moved directories was updated across `pipelines/`, `AINR/`, `AIAEC/`, `docs/`,
  `CLAUDE.md` and `.gitignore` (already correct). Make *goal*/*binary* names (`mono_alignulcnet`,
  `4ch_aec_bf_dfn_res`, …) are unchanged — only the directories moved, not what `make` calls them.
  `lib/aec`'s and `lib/nr`'s own docs mention a couple of these paths too; left alone this round
  (separate repos, already carrying unrelated uncommitted changes).
- Verified: `make -C pipelines test` and `make -C pipelines/4ch test` both exit 0;
  `pipelines/tests/pyc_parity_pipelines.sh` now reports `RESULT: OK` (91 PASS, 1 SKIP, was 6
  NOT_RUN before the `_ROOT` fix); `test_doc_layout_versions.py`/`test_c_python_map.py` pass;
  a filename-cross-reference sweep of every `.py`/`.c`/`.h`/`.sh` under `pipelines/` found no
  dead files.

## [Unreleased] — 2026-11-08 — DFN2 grid: the feature-normaliser settings and FFT scale become fixed, not carried fields

### Changed
- `lib/ainr/dfn2/dfn2_grid.h`'s `DFN2GridSpec` and `DFN2Grid` lose the seven feature-normaliser
  tau/init/scale fields (`erb_norm_tau_sec`, `erb_norm_init_lo_db`, `erb_norm_init_hi_db`,
  `erb_norm_scale_db`, `spec_norm_tau_sec`, `spec_norm_init_lo`, `spec_norm_init_hi`) and
  `DFN2Grid` loses `stft_scale`/`stft_unscale`: they never varied across any shipped grid, so
  they are now fixed constants (`DFN2_ERB_NORM_TAU_SEC` etc.) and two pure functions
  (`dfn2_stft_scale(n_fft)`/`dfn2_stft_unscale(n_fft)`) in the header instead of run-time fields.
  `erb_norm_alpha`/`spec_norm_alpha` are unchanged in meaning and are still derived from the
  grid's own sample rate and hop.
- `DFN2_GRID_ERR_ERB_NORM`/`DFN2_GRID_ERR_SPEC_NORM` merge into one `DFN2_GRID_ERR_NORM_ALPHA`
  (-12): with the tau/init/scale values no longer caller-supplied, the only way either alpha can
  still be degenerate is an extreme sample rate, which is one failure mode, not two. -13 is now
  permanently unassigned, like -8.
- `DFN2_PREPOST_CARVE_VERSION` 11 -> 12 (`lib/ainr/dfn2/dfn2_prepost.h`),
  `DFN2_STREAM_LAYOUT_VERSION` 1 -> 2 (`stream/ainr/dfn2/dfn2_stream.h`),
  `DFN_RES_STAGE_LAYOUT_VERSION`/`DFN_RATE_BRIDGE_LAYOUT_VERSION` 5 -> 6 (`pipelines/dfn/`): the
  carved pool byte counts shrank with the removed fields; a pool allocated by a caller before this
  change is refused, not silently misread.

## [Unreleased] — 2026-11-08 — Single-model release stage 2-4: model cores move to lib/, a PCM stream SDK for DFN2/DeepVQE-S/Align-ULCNet

Stages 2-4 of the 6-stage split-into-`lib/{ainr,aiaec}`+`stream/` plan (stage 0-1, the dependency-path
convergence, is the entry above). Five commits: `70c4be4` (move), `6ec6892`/`00b7a72` (DFN2 stream),
`6e6adf7` (DeepVQE-S stream), `9fe3acc` (Align-ULCNet stream).

### Added
- `lib/ainr/{dfn2,dfn3,gtcrn,rnnoise_erb}/`, `lib/aiaec/{ulcnet,deepvqe_s,cagcrn,align_cruse}/`: all
  8 models' C cores, moved out of `AINR/<Model>/`/`AIAEC/<Model>/` so a single model can be built and
  shipped without the rest of its family (confirmed: each of the 8 compiles standalone against only
  its own directory + audiolib's `common/include`, zero cross-model symbol references). `AIAEC/`'s
  shared `aiaec_process.{c,h}` split back into the model it serves
  (`lib/aiaec/{cagcrn,align_cruse}/*_process.{c,h}`, `lib/aiaec/ulcnet/ulcnet_mask_process.{c,h}`);
  `AIAEC/aiaec_process.h` is now a 3-line forwarding shim. One-line forwarding headers left at every
  old public-header path for out-of-tree consumers (board samples) whose own build still points `-I`
  there.
- `stream/ainr/dfn2/dfn2_stream.h` (+ `.c`, the ERB-matrix file loader in its own stdio-only TU,
  Makefile, tests): `dfn2_stream_init[_from_erb_file][_ex]`/`_reset`/`_pre_process`/`_post_process`
  wrap `lib/ainr/dfn2`'s `frame_open`/`_finish`/`_bypass` per-hop lifecycle, the analysis/synthesis
  STFT (`AC_OLA_ACCUMULATE`) and an optional input HPF around int16 PCM, one hop per call,
  `frames_per_call == 1` only. No stdio/heap outside the file loader (links into `NO_STDIO`
  targets). Builds full and `-DDFN2_MASK_ONLY`, reusing the existing `DFN2_VARIANT_SYMBOL` link-time
  check.
- `stream/aiaec/deepvqe_s/deepvqe_stream.h`: same shape, two PCM inputs (`mic_in`/`far_in`), no HPF,
  no file-loader, `AC_OLA_ENV_NORM` (the first push after init/reset emits nothing, surfaced as
  `post_process`'s own distinct `0` return). Bypass fails **closed** — silence, never a passthrough
  of the raw mic (there is no canceller in front of this model).
- `stream/aiaec/ulcnet/ulcnet_stream.h`: two PCM inputs, the linear-AEC's own `err_in` and raw
  `far_in` — never raw mic — plus the alignment-driven reset+one-hop-reprime bookkeeping the board
  sample does itself, re-implemented at the `lib/aiaec/ulcnet/ulcnet_prepost.h` level (not by
  wrapping `pipelines/ulcnet/ulcnet_nn_stage.{c,h}`, a different, higher-level component for a
  different caller — see the header's own comment for why). Bypass fails **open** — the linear-error
  spectrum passes through unguarded — the opposite of DeepVQE-S's.
- Every stream module: verified against an independent re-implementation of its board sample's own
  per-hop loop under a shared deterministic fake accelerator (feature-array and full per-hop trace
  comparison, zero mismatches), with the comparison itself proven capable of failing
  (`--inject-shift`, and model-specific injections: DeepVQE-S's `--ola-matters`, ULCNet's
  fail-open/reprime/solid-transition/explicit-reset rows). Host-only, fake accelerator — not
  board-verified, not real-graph-verified.

### Fixed
- `dfn2_stream`/`deepvqe_stream`/`ulcnet_stream`'s `descriptor_matches()` compared every
  `*StreamMemReq` field except the second `reserved` one (latent, since every writer zeroes it).
- `ulcnet_stream_reset()` didn't clear `had_hop`/`prev_solid`, so an explicit caller-driven reset
  left stale alignment-tracking state that could spuriously re-trigger a second reset+reprime on the
  very next hop.

### Not done this pass (deliberate, per the plan / standing scope)
- Board multi-frame (`frames_per_call > 1`, stage 3b): deferred, DFN2's own C already supports N>1 at
  the `lib/ainr/dfn2` level, only the board SDK wrapper is out of scope here.
- CAGCRN, Align-CRUSE: `lib/`-only (stage 2's move, confirmed standalone-buildable); no `stream/` or
  graph-binding work for these two.
- `pipelines/ulcnet/ulcnet_nn_stage.{c,h}` and `pipelines/dfn/{dfn_res_stage,dfn_rate_bridge}.{c,h}`:
  untouched, intentionally — different, higher-level components for different existing callers
  (the 4-channel pipeline's synchronous model-callback consumer; the AEC+DFN2 dual-spectrum stage),
  not superseded by the new `stream/` SDKs.
- `docs/html/*`, `docs/audio_libraries_c_api_zh_TW*.md`: still carry some pre-move header paths;
  routed to the `algdoc` pass, not hand-edited here.

## [Unreleased] — 2026-11-07 — Dependency path convergence, stage 0-1: one lib/deps.mk, three lib/*.pin

Stages 0-1 of a 6-stage split-into-lib/{ainr,aiaec}+stream/ plan (stage 2, the actual file move, is
a later, separate pass). This pass only converges where AEC/NR/audiolib are read from and how each
is pinned; no model source moved.

### Added
- `lib/deps.mk`: the one place `AEC_DIR`/`NR_DIR`/`AUDIOLIB_DIR`/`COMMON_DIR` are resolved, included
  by `pipelines/Makefile`, `pipelines/4ch_aec_bf_nr_res/Makefile`, `AINR/Makefile`,
  `AINR/RNNoise-ERB/Makefile` and `AIAEC/Makefile`. Default changed from the `lib/aec`/`lib/nr`
  submodule checkouts to the sibling dev checkouts (`../AEC/c_impl`, `../NR/c_impl`; audiolib was
  already sibling-resolved) — the "recommended sole dev working directory" for each dependency.
  Every value stays a caller-overridable `?=`; `AEC_DIR=lib/aec/c_impl NR_DIR=lib/nr/c_impl` still
  builds against the submodules. Two extra targets: `bootstrap-deps` (clone whichever sibling is
  missing, at its pinned commit, from an overridable URL) and `check-deps-pins` (report, not
  enforce, each pin against its resolved checkout and against the two gitlinks).
- `lib/aec.pin`, `lib/nr.pin`: pinned commit + bootstrap URL for AEC and NR, the same shape
  `pipelines/audiolib.pin` already had (now `lib/audiolib.pin`, moved, same value — see Removed).
  Each dependency has exactly one authoritative pinned version, read from one place, instead of
  `lib/aec`'s/`lib/nr`'s gitlink being the only record.

### Changed
- `pipelines/Makefile`'s `check-pins`: now checks `AEC_DIR`/`NR_DIR`/`AUDIOLIB_DIR` — the checkout
  that actually resolved for this build — against `lib/aec.pin`/`lib/nr.pin`/`lib/audiolib.pin`,
  and fails on a mismatch there. The `lib/aec`/`lib/nr` gitlinks are reported as a NOTE only (the
  step-5 "gitlink migration" readiness gate), never failed on, since they are no longer what a
  default build reads.
- `AIAEC/Makefile`'s hardcoded `../../audiolib/common` (`CPPFLAGS`, the `target_abi_signature.sh`
  call) now reads `$(COMMON_DIR)` from `lib/deps.mk`.
- `pipelines/tests/pyc_parity_pipelines.sh`: `aec`/`nr` archive-path queries and the driver's `-I`
  flags, previously hardcoded to `$ALG/lib/aec/c_impl` / `$ALG/lib/nr/c_impl`, now use
  `$AEC_DIR`/`$NR_DIR` from `lib/deps.mk print-deps-sh` — the same resolution the `aec_nr_pipeline`
  build two lines above it already uses, so the driver links against the archive the build actually
  produced instead of a second, independently-hardcoded path that would silently diverge from it.
- `pipelines/scripts/test_build_isolation.sh`: this script's own `$AEC_DIR`/`$NR_DIR`/`$COMMON_DIR`
  (set once, near the top, to the submodule/sibling-audiolib paths, used by its `resolve_producers`
  helper) were never the problem by themselves, but 13 real-tree `make ... COMMON_LIB=/AEC_LIB=/
  NR_LIB=...` invocations (SP1's `all`/`print-bin-dir` pair ×3, and S7's/SP-S11's/SP-S15's `publish`
  ×7) resolved those archive paths via this script's own vars and then built/published *without*
  re-passing `AEC_DIR=`/`NR_DIR=`/`COMMON_DIR=`, relying on `pipelines/Makefile`'s bare default to
  agree with the script's own choice. That default is exactly what this pass changed (submodule ->
  sibling), so left alone these 13 calls would have started resolving the `$(AEC_LIB)`/`$(NR_LIB)`
  FORCE rules (and, for `publish`, the git-dirty/identity checks) against the sibling while the
  archive path and the script's own invariants stayed keyed to the submodule -- the exact
  divergence this pass already found and fixed once in `pyc_parity_pipelines.sh`. Fixed by adding
  the explicit `AEC_DIR="$AEC_DIR" NR_DIR="$NR_DIR" COMMON_DIR="$COMMON_DIR"` to each of the 13
  calls, restoring self-consistency with `resolve_producers`. The suite was read in full for this
  class of issue and this fix was applied surgically (no other line changed); it was **not run**
  end-to-end in this pass -- it is a long-running suite with its own heavy, carefully-documented
  safety contract (scratch roots, sentinel snapshots of four to six real trees, a cross-process
  lock) that a partial or rushed run risks tripping or desyncing. Running it to green is a
  follow-up, not a claim made here.

### Removed
- `pipelines/audiolib.pin` → moved to `lib/audiolib.pin` (same `AUDIOLIB_COMMIT`, not a new pin;
  sits beside `lib/aec.pin`/`lib/nr.pin` now instead of alone next to two unpinned gitlinks).

### Not done this pass (deliberate)
- `.gitmodules` and the `lib/aec`/`lib/nr` gitlink commits are untouched (step 5 of the proposal):
  owner review of steps 0-1's result comes first. `check-deps-pins`/`check-pins`'s gitlink NOTE is
  the readiness signal for that review, not an instruction to act on it here.
- `pipelines/scripts/audit_fp_contract.sh` / `audit_dead_defines.sh` were read but left alone: both
  only ever needed `COMMON_DIR`, already sibling-resolved before this pass. `test_build_isolation.sh`
  needed and got a scoped fix -- see Changed above -- but was not run end-to-end.

## [Unreleased] — 2026-11-06 — The AI model libraries are frequency domain only, take run-time grids and caller memory; the pipelines follow

The model libraries (`AINR`, `AIAEC`) lose everything that is not the model's own pre/post arithmetic, and the DFN and ULCNet pipeline
units, example mains and tests follow. Library-side detail: `AINR/CHANGELOG.md`.

### Changed
- Frequency domain only. `*_IO_TIME`, `*_prepost_pre_process` / `_post_process`, `io_mode`, `*_framing_config`, `*_analysis`,
  `*_synthesis` and every FFT, window, `AcStft` and `AcIstft` member of the classes are gone. The application frames with audiolib
  (`ac_stft_push_planar`, `ac_istft_push_planar`) and calls `*_pre_process_freq` / `*_post_process_freq`. The scales belong to the
  application's `AcStftConfig`: DFN2 driven directly uses `spec_scale = grid.stft_scale` and `ispec_scale = grid.stft_unscale`, while
  the stage and the bridge keep 1 and apply the grid's scales themselves.
  `scripts/audit_no_transform_in_models.sh` now forbids the analysis side too (`AcStft`, `ac_stft_*`, `fft_forward*`, `spec_scale`,
  `ispec_scale`); only `ac_stft_config_default`, `ac_stft_get_mem_size` and `AcStftConfig` (the `n_fft` check) may appear in a
  model directory (33 sources checked).
- Run-time grid specs. `DFN2GridSpec` (15 fields, with `df_order`, `frames_per_call` and the seven normaliser values),
  `DFN3GridSpec`, `UlcnetGridSpec` and `DeepVqeGridSpec` turn into a grid value with `<lib>_grid_make()` that goes into every
  config. Gone with the compile-time geometry: the capacity macros (`DFN2_MAX_*`,
  `DFN2_GRID_ERR_CAPACITY`), `DFN2_DF_ORDER`, `DFN2_FRAMES_PER_CALL`, the `DFN3_*` geometry and lookahead macros and their link
  symbol, `ULCNET_*` / `AIAEC_*` grid macros, the `-D` normaliser starts, and the ring-suffixed `dfn2_variant_full_*_r<ring>`
  symbol (`DFN2_VARIANT_SYMBOL` stays, so units built with and without `-DDFN2_MASK_ONLY` still do not link together).
- One caller block per class: `<lib>_get_mem_size(cfg)` -> caller memory (16-byte aligned) -> `<lib>_init`. Carve versions:
  `DFN2_PREPOST_CARVE_VERSION` 9 -> 11, `DFN3_PREPOST_CARVE_VERSION` 8 -> 9, `ULCNET_PREPOST_CARVE_VERSION` 7 -> 8,
  `DEEPVQE_PREPOST_CARVE_VERSION` 6 -> 8; the model I/O layout versions are unchanged. A `MemReq` queried before this change is
  refused by `init_ex()`; query again.
- Recurrent state belongs to the application. A float host registers (pointer, element count) pairs in the prepost config; a board
  registers nothing and its accelerator holds the state; a state generation counter (`*_state_generation()`) tells the board when
  to reload quantized zeros. The registered state is in no pool size and no hash. `dfn_res_stage_state_generation()` and
  `dfn_rate_bridge_state_generation()` are new (the stage registers no state; the bridge forwards the stage's value; 0 for NULL):
  the value is 1 after init, plus one per reset, per warm-up frame and per refused commit of a real frame, unchanged when the model callback returns nonzero.
- One flow on every hop. `dfn_res_stage_begin()` returns 1 on every hop (it returned 0 on the first `grid.mask_lookahead` hops): the first
  M hops are warm-up frames, the model callback runs on them, `dfn_res_stage_end()` discards the heads and the library restarts the
  model state (generation + 1 per warm-up frame, so a callback that holds state reloads quantized zeros after each). Audio is bit-identical
  to the old flow. `end()` still returns 0 with an all-zero `out_spec` for the first L hops. `dfn_res_stage_get_counters()`: the warm-up
  frames count in `frames_in` only, never in `commits` or `skips`. Through `dfn_rate_bridge`, `nn_begin()` returns 1 on those frames too
  (the schedule, the prefill, the FIFO sizes and the added delay of 672 / 629 / 330 samples are unchanged: they count 48 kHz frames, not
  model calls), so the first 48 kHz-frame host hop (hop 1 at 16 kHz / 128, hop 0 at 16 kHz / 256 and 8 kHz / 128) makes one more model
  call, M calls once per start or reset; a host hop still carries at most two stage frames. `dfn_hop.h` and the example mains change in
  comments only, apart from three dead conditions removed: `job_pending` is set unconditionally in `dfn_hop_begin()`, the identity result
  is passed unconditionally in `dfn_hop_end()` (an unclosed stage frame is always open), and `dfn_res_stage_process()` always calls
  `infer`. One frame per call only: a grid with `frames_per_call` > 1 is refused as before; with M = 0 there is
  no warm-up frame. The Python twin is unchanged (it never mirrored the return value or the generation; the C stage evaluates the heads M
  more times per stream and Python's `heads_calls` / `skips` count real frames only).
- Renames: `<lib>_prepost_frame_inputs` / `_commit` / `_skip` -> `_frame_open` / `_finish` / `_bypass`, `dfn2_compose[_stream]` and
  `dfn3_compose[_stream]` -> `dfn2_enhance_spectrum[_stream]` and `dfn3_enhance_spectrum[_stream]`, `ulcnet_model_io_commit` ->
  `ulcnet_model_io_enhance_spectrum`.
- `dfn/dfn_example_args.h`: `DFN_EXAMPLE_MODEL_ROWS` are `DFN2GridSpec` rows (each carries `df_order`, `frames_per_call` and the normaliser values).
- `DFN_RES_STAGE_LAYOUT_VERSION` and `DFN_RATE_BRIDGE_LAYOUT_VERSION` 4 -> 5. The pools shrink because the classes no longer carry the
  transform objects or the recurrent state; measured with the `test_target_abi` rows (48 kHz test grid, pair 1/1, ULCNet and DeepVQE-S
  16 kHz / 512, D 8, bridge host 16 kHz): `dfn2_prepost` 219,648 -> 92,608 B, `dfn3_prepost` 219,472 -> 90,384, `ulcnet_prepost`
  75,248 -> 14,784, `deepvqe_prepost` 438,640 -> 35,504, `dfn_res_stage` 232,096 -> 105,056, `dfn_rate_bridge` 350,000 -> 222,960.
  Unit banners: `mono_aec_dfn_res` 718,096 -> 591,056 B (host 16 kHz), 1,450,016 -> 1,322,976 (48 kHz), 622,720 -> 495,680 (8 kHz),
  594,048 -> 431,456 (16 kHz host and 16000 / 256 model); `4ch_aec_bf_dfn_res` 2,066,240 -> 1,939,200 (core 16 kHz), 6,646,000 ->
  6,518,960 (48 kHz), 1,942,192 -> 1,779,600 (16 kHz and 16000 / 256 model). The 48 kHz-model rows all shrink by 127,040 B; the shipped
  graph's state is 32,000 float32 values (128,000 B), and the 960 B difference is not decomposed. `mcab` is unchanged.
- `ulcnet/ulcnet_nn_stage.{h,c}`: `ulcnet_nn_stage_mem_size(const UlcnetGrid *, err_analysis)`,
  `ulcnet_nn_stage_init(mem, bytes, const UlcnetGrid *, err_analysis)` and new `ulcnet_nn_stage_grid()`; spectra have `grid.n_bins`
  bins, hops `grid.hop` samples. `ulcnet_accelerator_adapter_init()` takes the state registration (NULL in the examples) and has
  `ulcnet_accelerator_adapter_state_generation()`.
- `mono_alignulcnet` and `4ch_alignulcnet`: new `--sample-rate 16000|48000` (default 16000; anything else exits 2), so one binary serves both
  grids. The `ULCNET_GRID` make variable (`ULCNET_GRID_CFLAGS`, `-DULCNET_MODEL_IO_SR` / `-DULCNET_MODEL_IO_N_FFT`) and the `test-grids`
  targets of `pipelines/Makefile` and `4ch_aec_bf_nr_res/Makefile` are removed; both `test` recipes gain a `--sample-rate 48000` run.
  Pools at 16 kHz / 512: `mono_alignulcnet` 537,360 -> 537,424 B; `4ch_alignulcnet` unit pool 2,994,640 B unchanged, stage pool
  36,400 -> 36,464 B, adapter pool 75,280 -> 14,816 B (the +64 B are not decomposed). At 48 kHz / 1024 (the old build could not hold both
  grids in one binary): `mono_alignulcnet` 1,246,912 B; `4ch_alignulcnet` unit pool 6,304,624 B, stage pool 71,792 B, adapter pool
  29,152 B. The adapter pool follows the grid, not `D`.
- Makefiles: besides the variables above, `DFN3_MASK_LOOKAHEAD` / `DFN3_DF_LOOKAHEAD`, `DFN_PAIR_CFLAGS` and `DFN2_PAIR_RING` (so
  `-DDFN2_MAX_DF_RING` is never forwarded) are removed; a retired knob on make's command
  line is silently ignored, not an error. `DFN2_MASK_LOOKAHEAD`, `DFN2_DF_LOOKAHEAD` and `DFN2_TEST_GRID_KHZ` stay for the test
  programs. `make test-pairs` keeps the `check_pair_report.sh` gate; its archive-symbol check went with the ring-suffixed symbol. New
  `make audit-dead-defines` (`scripts/audit_dead_defines.sh`: every `-D` flag of a Makefile under `AINR/`, `AIAEC/` and `pipelines/`
  must be read by a C source) and `make print-untracked`, with the variables `ALG_ROOT`, `UNTRACKED_FILES_CMD` and `DEAD_DEFINES_ARGS`.
  `scripts/audit_fp_contract.sh` exempts `dfn2_norm_alpha` (four explicit `fma()` calls, by function block).
- `dfn/tests/dfn_test_grid.h` loses `DFN2_WIN_LEN`, `DFN2_OVL_LEN`, `DFN2_DF_ORDER`, `DFN2_DF_HISTORY`, `DFN2_DF_RING`,
  `DFN2_DF_RING_NOISY`; it gains the `DFN_TEST_*` model numbers (`DFN_TEST_DF_ORDER` 5, `DFN_TEST_FRAMES_PER_CALL` 1, the normaliser
  values), `dfn_test_spec()`, `dfn_test_spec_of()` and `dfn_test_table_fits()`, and stops at an `#error` under `-DDFN2_MASK_ONLY`.
- Python twin `dfn2_stage.py`: names in comments and one error message only; the arithmetic is unchanged.

### Tests
- Re-run for this change on a copy of the working tree (host, default NEON build): `make -C pipelines test` and
  `make -C pipelines/4ch_aec_bf_nr_res test` exit 0; `test_cli_strict.sh` reports 756 command lines PASS, five of them new
  (`--erb-fwd` alone and `--erb-inv` alone exit 2; an unopenable file, a file with fewer floats and a file with more floats than
  513 bins x 32 bands exit 1, each with its message);
  `make -C pipelines target-abi-test` passes its 8 descriptor rows; `make -C pipelines test-pairs` (default 4 pairs) and
  `test-dfn-grids` exit 0; `audit-dead-defines` passes (22 `-D` flags, each read by a source), `audit-no-transform` passes.
- pytest: `test_dfn_test_grid_pin.py` (new, compares the test header's model numbers with `AINR/DeepFilterNet2/config.ini`) 2,
  `test_dfn2_stage_c_parity.py` 10 passed and 2 skipped, `test_dfn2_rate_bridge_c_parity.py` 27 (allclose, rtol 2e-4 / atol 2e-6: the Python
  and C FFT backends differ), `test_dfn2_stage.py` 37, `test_dfn2_rate_adapter.py` 21, the mono contract 19, the 4ch contract 3,
  `test_no_transform_in_models.py` 68.
- Warm-up frames: `test_dfn_res_stage.c` expects begin == 1 on the first hop, generation 1 + M after the test grid's hops (every pair),
  the capture of the model windows on all eight hops with hop 0 showing the zero-initialised window, a counting model (one call per hop,
  counters untouched during warm-up) and a stateful mock model run three ways (reload on every generation change equals a model that
  never saw a warm-up frame bit for bit, also with NaN and 1e30 heads on the warm-up calls and across a reset; reload only at start
  differs when M > 0). `test_dfn_rate_bridge.c` separates the warm-up +1 from the refused-commit +1 and checks model calls per host hop
  against the schedule, generation 1 + min(calls, M) and the added delays; `test_dfn_hop.c` checks one stage frame per hop from the first
  hop and model calls == stage `frames_in`.
- `test_dfn_res_stage.c` drops the capacity branch and gains `frames_per_call` 2 refused, state-generation rows and byte snapshots of
  refused calls; `test_target_abi.c` and `test_readme_flow.c` are ported with the same case count. `test_ulcnet_nn_stage.c` runs every
  grid of its table and checks the grid and pool plan (refusals leave the block bytes unchanged); `test_mono_ulcnet.c` and
  `test_4ch_ulcnet.c` run both grids from one binary.
- Not run: any board; an ARM32 or target-toolchain build; ASan (its runtime does not start on the host, UBSan was used); the audio quality of any lookahead pair other than 1/1. The 32-bit pool sizes of the new classes are predicted
  by a layout mirror only, and none exists for the stage, the bridge or the units.

## [Unreleased] — 2026-11-05 — The DFN2 pipelines take their grid in the config; the rate bridge refuses a non-48 kHz model

The DFN2 library (`AINR/DeepFilterNet2`) holds no grid any more (`DFN2Grid`, `dfn2_grid_make()`; see `AINR/CHANGELOG.md`), and neither
do the pipeline units: the application builds a `DFN2Grid` and passes it in the stage config. No pipeline object, example main or
board sample chooses a grid at build time; only the test programs do.

### Changed
- `dfn/dfn_res_stage.{h,c}`: `DfnResStageConfig` drops `sample_rate` / `fft_size` and carries `DFN2Grid grid`;
  `dfn_res_stage_default_config(const DFN2Grid *)` copies the grid in. `get_mem_requirements()` refuses a grid the library refuses
  (`dfn2_grid_validate()`: NULL, zeroed, a hand-edited field) and a model whose I/O descriptor is for another grid, lookahead pair or
  frame count, at run time. `DFN_RES_STAGE_IN_SCALE` / `_OUT_SCALE` / `_LOOKAHEAD_FRAMES` are gone: the stage scales with the grid's
  `stft_scale` / `stft_unscale` (exact 2^-5 and 2^5 on the shipped 48 kHz / 1024 grid, so its output does not change) and the latency
  is the inline `dfn_res_stage_lookahead_frames(grid)` = mask + df lookahead; `dfn_res_stage_grid(stage)` returns the grid it runs.
  `DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK()` is replaced by `DFN2_REQUIRE_VARIANT_LINK()` (the DFN2 capacities of this object must be the
  library objects'); there is no link-time pair symbol any more. The same stage code runs the 16, 32 and 48 kHz grids of the
  example table.
- `dfn/dfn_rate_bridge.{h,c}`: `dfn_rate_bridge_default_config(sample_rate, const DFN2Grid *model_grid)`. The bridge keeps its own
  constants (48000 Hz, FFT 1024, hop 512, 513 bins) as its contract and refuses (-1) a stage grid that is not 48000 / 1024: the
  resampling ratios, the per-hop inference schedule, the prefill calibration and the FIFO sizes are written for it. The delay formulas
  use the stage grid's lookahead. The host side is unchanged (8 kHz / fft 256, 16 kHz / fft 256, 16 kHz / fft 512).
- `dfn/dfn_hop.h`, `dfn_hop_pool.h`, `dfn_unit_partition.h`: read the grid from the stage config or the arguments;
  `dfn_hop_pool_plan()` sets `use_bridge` when the host rate differs from the grid's rate; `dfn_unit_partition(fwd, inv, grid)`.
- `dfn/dfn_example_args.h` and the two example mains (`mono_aec_dfn_res`, `4ch_aec_bf_dfn_res`): new `--ai-sample-rate` and
  `--ai-fft-size` (default 48000 and 1024; `--sample-rate` stays the host rate), a two-row table (16000/256 -> n_erb 24, df_bins 96,
  lookahead 0/0; 48000/1024 -> 32, 96, 1/1; no 32 kHz row, the AEC has no 32 kHz grid) shared with the board sample `alg_ai3_aec_dfn.c`, and
  a `model grid ...` banner line. A pair outside the table prints the accepted ones and exits 2; a host rate the bridge cannot serve
  with that model (anything but the 48 kHz / 1024 model with an 8 or 16 kHz host) is a rejected pool query, exit 1. A host on the
  model's own rate runs the stage directly.
- No `dfn/dfn_grid.h`: the build-time selector (`-DDFN2_GRID_KHZ=16|32|48`, or `-DDFN2_SR` / `-DDFN2_N_FFT` / `-DDFN2_N_ERB` /
  `-DDFN2_DF_BINS`, with `-DDFN2_MASK_LOOKAHEAD` / `-DDFN2_DF_LOOKAHEAD`) lives only in the new test header
  `dfn/tests/dfn_test_grid.h`, which gives test programs the `DFN2_*` size macros for their arrays and `dfn_test_grid()`; product code
  never includes it.
- `pipelines/Makefile` and `4ch_aec_bf_nr_res/Makefile`: `DFN_TEST_CFLAGS` (the grid preset and the DFN2 pair) reaches the test units
  alone and is part of the build signature; new `DFN2_TEST_GRID_KHZ` (16, 32, 48) and `make test-dfn-grids` (stage, bridge and hop tests
  at the 16 and 32 kHz test grids, with the test header's pair 1/1). A pair whose state ring `max(5, mask + df + 1)` is deeper than
  the library's default capacity of 5 (1/4 and 2/3: 6, 2/4: 7) still forwards `-DDFN2_MAX_DF_RING=<ring>` to every unit.
- `make test-pairs`: the old archive check for `dfn2_lookahead_mask<M>_df<D>` is gone with the symbol, and ring 5 is the same for most
  pairs, so the gate now has two parts. Each pair test prints the pair it was built with and (stage test, mono composition) the
  stage latency it measured; `dfn/tests/check_pair_report.sh` compares them with the pair asked for and with `(M + D + 1) * hop`,
  so a pair flag that never reached the test build prints 1/1 and fails. The archive must also carry the
  `dfn2_variant_full_n*_e*_d*_r<ring>` symbol of the ring the pair needs.
- `DFN_RES_STAGE_LAYOUT_VERSION` and `DFN_RATE_BRIDGE_LAYOUT_VERSION` 3 -> 4: every DFN pool grew by 112 bytes, because the grid is
  now a value in the DFN2 state and in the pre/post class (`sizeof(DFN2Grid)` is 56 bytes) that the stage pool carves, and the
  pre/post pool hash folds the whole grid. Measured with `get_mem_requirements()` on the shipped 48 kHz / 1/1 grid (ne10): stage
  231,984 -> 232,096 B, 16 kHz rate bridge 349,888 -> 350,000 B, the DFN2 pre/post FREQ pool inside them 219,536 -> 219,648 B
  (`DFN2_PREPOST_CARVE_VERSION` 8 -> 9, see `AINR/CHANGELOG.md`). A `MemReq` queried before this change is refused by `init_ex()`;
  query again. At the shipped grid every digest and checksum the stage, bridge, hop, mono and 4ch tests printed before is unchanged
  (the stage and bridge outputs of a probe over identity, random heads and an attenuation limit are bit-identical).
- The Python twin (`dfn2_stage.py`) and the pipelines' C-parity tests still verify the shipped grid; the other grids are exercised
  by the stage, bridge and hop tests (`make test-dfn-grids`) and are not verified inside a full pipeline.

### Tests
- `dfn/tests/dfn_test_fixture.h` gains `dfn_fixture_foreign_descriptor()`: the I/O descriptor of a model for another mask lookahead,
  another deep-filter lookahead or another grid. `test_dfn_res_stage.c` and `test_dfn_rate_bridge.c` check that the stage and the
  bridge refuse it (and accept the test grid's own descriptor), the run-time replacement for the link-time pair check, and refuse a
  zeroed, NULL or hand-edited grid. `test_dfn_res_stage.c` also runs a table of all three grids (16, 32 and 48 kHz) at run time with a
  stand-in model (finite output, latency equal to the grid's lookahead) and the cross-grid refusals; `test_dfn_rate_bridge.c` refuses
  every non-48 kHz / 1024 stage grid in every build.
- The example mains' new flags are checked by `make test` in both Makefiles (the stage on the host grid at 16000/256, exit 1 for a
  model the bridge cannot serve, exit 2 for a pair outside the table or a malformed value).
- The grid itself is covered by `AINR/tests/test_dfn2_grid.py`.

## [Unreleased] — 2026-11-02 — NEON model kernels also run on 32-bit ARM

### Changed
- The NEON model pre/post kernels of `AINR`, `AIAEC` and the pipelines are keyed on the shared `SK_HAVE_NEON`, which is now true on
  32-bit ARM with NEON plus fused FMA (`-mcpu=cortex-a32/a53/a55/a73`, `-mfpu=neon-vfpv4`); other ARM32 targets and `SIMD=0` stay
  scalar. The backend labels come from `SK_NEON_BACKEND_NAME` (`aarch32-neon` on 32-bit ARM).
- `AIAEC/Makefile` appends `-ffp-contract=off -fno-math-errno` with `override`, as `AINR/Makefile` does, so a `CFLAGS=` on the
  command line no longer drops the floating-point contract.

- `AINR/RNNoise-ERB/process.c`: `hidden_finite` takes the flat `const float *` block instead of a `const float [3][128]` parameter. Passing
  `state->hidden` (a `float [3][128]`) to it was a `-Wpedantic` error under gcc before C23 ("pointers to arrays with different qualifiers");
  the check itself is unchanged.
- gcc on 32-bit ARM warned `-Waggressive-loop-optimizations` on the scalar tail after the NEON loop of `df_common_features`
  (`dfn2_process.c`, `dfn3_process.c`) and of `sk_ema_f32`, an error under `WERROR=1`; the warning is silenced around those
  functions only (gcc; the pragma is a diagnostic, it changes no code). The tails are unchanged and correct for every size.

### Tests
- `dfn/tests/test_dfn_res_stage.c` allocates the synthesis block with `posix_memalign` (as the same file already does elsewhere) instead of
  `aligned_alloc`, which glibc does not declare under the build's `-std=gnu99`: the call was an implicit declaration (a
  `-Wimplicit-function-declaration` warning, and a truncated pointer on a 64-bit target).
- `AINR/tests/test_nn_kernels.c` and `AIAEC/tests/test_prepost_kernels.c` returned `EXIT_SUCCESS` from `main` through their `CHECK`
  macro when a sub-test failed; they now return `EXIT_FAILURE`. `AIAEC/tests/test_prepost_kernels.c` defines `_POSIX_C_SOURCE` for
  `posix_memalign` under `-std=c11`. Tests that compare NEON with scalar call `SK_MATCH_SCALAR_TO_VECTOR_MODE()` first.

## [Unreleased] — 2026-10-29 — Consumers borrow the AEC synthesis window through `aec_synthesis_window()`

### Changed
- Every consumer that synthesises with the AEC's window table takes it from the public `aec_synthesis_window()` of its
  initialised `Aec` instead of the AEC-internal `aec_balanced_config.h` (`aecb_rate_cfg()->synth_window`): `mono_aec_nr_res`
  (`main.c`, `example_board_adapter.c`, `tests/test_mono_chain.c`, `tests/conv_chain_mono.h`), `tests/test_readme_flow.c`,
  `dfn/mono_aec_dfn_res/tests/test_mono_dfn.c`, `4ch_aec_bf_nr_res/tests/test_mcab.c`, `tests/pyc_parity/pyc_drv.c`, and the
  board samples `alg_audio_aec.c` / `alg_audio_aec_nr.c`. The accessor needs the `Aec`, which a pool-sized program creates after
  sizing; a borrowed window carves nothing, so the pool is sized with a stand-in pointer and the AEC's table is installed at init.
  Same table, so the outputs are bit-identical; no consumer includes `aec_balanced_config.h` any more.

## [Unreleased] — 2026-10-28 — The examples render the AEC's rendering base and follow its post-chain resets (mono, mono-DFN, mono-ULCNet, four-channel flow, rate bridge)

Follows lib/aec's split of the analysis seam (`error_spec` / `formed_hop`, never guarded) from the rendering base
(`aec_get_output_base()`), its reset counter (`aec_post_reset_serial()`), its comfort-noise seed (`AEC_CNG_SEED`, 42) and its
synthesis window (`aec_synthesis_window()`). The ownership rule is the one of option C below: the inverse FFT, the
overlap-add and the comfort-noise draw stay in the application (audiolib `ac_istft_*`, `aec_cng_fill()`); the library only
states the facts the application needs (the base, the serial, the seed, the window).

### Reset-serial protocol
- Every chain that synthesises or draws comfort noise itself keeps its own last-handled serial. When the value read after a
  processing call differs, then, before the hop's first comfort-noise draw and before its `ac_istft_push()`, it calls
  `ac_istft_reset()` (the overlap-add tail goes), sets its generator to `AEC_CNG_SEED` and stores the new value. It never
  resets the AEC in response, and the denoiser keeps running. The counter moves on the hard post-chain resets only
  (`aec_reset()`, the hard branch of a first acquisition or of a confirmed shift); warm and soft realignments, the bad-relock
  soft revert and `aec_apply_external_realign()` do not move it.
- `mono_aec_nr_res` (`follow_post_reset()`), `mono_aec_dfn_res`, the 4ch chains (`mcab_sample_post_reset_due()`), the board
  adapter `example_board_adapter.c` and the Python reference (`ctx.post_reset_serial`) follow it. `mono_alignulcnet` has no
  comfort noise and no synthesis of its own, so there is nothing to clear or reseed there, and `4ch_alignulcnet` moves the
  serial only with a flow reset, which its recovery already pairs with `ulcnet_nn_stage_reset()`.

### Changed
- `mono_aec_nr_res` (`main.c`, `example_board_adapter.c`, the test chains): the denoiser estimates on the rendering base
  B and the gain is applied to B (`aec_get_output_base()`; B is `error_spec` bit for bit on every hop the output guard
  leaves alone, `error_spec + echo_spec` on a guard hop). The synthesis borrows the AEC's window table
  (`aec_synthesis_window()`) instead of audiolib's default window (they differ in the last float bit), so `--no-nr` with the
  RES gain on equals `aec_process()` in full mode bit for bit. `DUMP_CTX` writes B in its first field.
- `mono_aec_dfn_res`: P = B * G_res and the stage's estimate is B. The comfort noise reseeds on the event; at 48 kHz the
  host synthesis drops its tail when the frame built on the reset hop reaches it (a delay line of
  `DFN_RES_STAGE_LOOKAHEAD_FRAMES` hops). Below 48 kHz the host-grid synthesis is inside `dfn_rate_bridge`; the bridge call
  that clears it is described below.
- `mono_alignulcnet`, `4ch_alignulcnet`: unchanged code; the stage reads the analysis seam (`formed_hop`; in the 4ch flow the
  beam of the lanes' `error_spec`), documented in `main.c` and the READMEs, together with the fact that these chains have no
  rendering guard: on a guard hop the network sees the real linear residual, and the identity and fail-open hops emit the
  unguarded residual.
- `mcab`: each lane's rendering base and guard flag are bound per hop (`McabFront.render_spectra`, `.guard_active`);
  `mcab_fuse()` adds `McabFrame.b` (the same weights over the lanes' rendering bases; `e` itself unless a contributing
  lane's guard fired, then a new fuse buffer), `.guard_lanes` and `.post_reset_serial` (the lanes' counters summed
  modulo 2^32, also in `McabFront`). No guard of mcab's own, before or after the beam; SRP/BF and every estimator
  still read the analysis spectra. `MCAB_LAYOUT_VERSION` 25 -> 26 (one more fuse buffer, larger control block;
  25 is refused like 21..24).
- 4ch example chains (`mcab_raw`, `mcab_static`, `mcab_external_static`, `4ch_aec_bf_dfn_res`, the test chains): render
  `P = B * gain`, estimate on `E`; clear the synthesis tail and reseed the comfort noise when `post_reset_serial` moves
  (`mcab_sample_post_reset_due()`; the DFN2 example resets its whole hop at 48 kHz, see below for the bridge path).
- Comfort-noise seed: the library's `AEC_CNG_SEED` (42) in every file instead of `0x9e3779b9`
  (`MCAB_SAMPLE_CNG_RNG_SEED` is its alias). The generator is the library's recipe, `aec_cng_fill()`; in the Python reference
  `comfort_noise_fill()` calls the AEC library's `cng_fill` (the Gaussian look-alike is gone) and `common/cng_aec3.py` is a
  stateful wrapper of it (`Aec3ComfortNoise.fill()` replaces `.add()`).
- Python reference: `run_nr_spectrum()` and `run_res()` read `ctx.output_base_spec` and follow
  `ctx.post_reset_serial`; `run_dfn2_res()` forms P from B and delays the OLA clear with the stage;
  `4ch_aec_bf_nr_res/pipeline.py`: the fused context carries `output_base_spec`, `guard_active` and `post_reset_serial` by
  the same rule.
- `pyc_parity`: the mono seam records the rendering base flag and the reset serial (`sc[12]`, `sc[13]`); the mono driver
  renders the base and follows the event; new row `d.mono.<grid>.reset_glue`; `d.mono.cng_lookalike` became
  `d.mono.cng_pipeline` (bit exact against C). For the 4ch chain the seam record carries B (`McabFrame.b`) after Y and
  `sc[12]`/`sc[13]` hold `guard_lanes`/`post_reset_serial`; the 4ch replay renders the recorded B (it left `McabFrame.b`
  NULL and crashed, so `a.nr4.*` and `l2.*` were NOT_RUN). The Python NR of a 4ch seam is estimated on E
  (`mcab_sample_anr_gain()`), of a mono seam on B. New row `a.nr4.<grid>.replay_identity`: the replay over the chain's
  own seam equals the chain bit for bit, with guard hops present (a replay rendering E fails it).
- Board samples (the board-sample directory outside the repository): `alg_audio_aec.c`, `alg_audio_aec_nr.c` and `alg_ai3_aec_dfn.c`
  add comfort noise only with `--cng` (off by default; `alg_audio_aec_nr.c` also accepts `--no-cng`, and the last of `--cng` / `--no-cng` wins). This
  is a sample setting, not a library change: `aec_config_from_level()` still sets `enable_cng = 1`. `--cng` on and off are each
  compared against the same configuration. All three render the rendering base
  (`aec_get_output_base()`); `alg_audio_aec.c` and `alg_audio_aec_nr.c` also borrow the AEC's window; `alg_ai3_aec_dfn.c` calls
  the bridge reset below.

### Added
- `dfn_rate_bridge_reset_synthesis()` (C) and `Dfn2RateBridge.reset_synthesis()` (Python). Below 48 kHz the host-grid
  synthesis (`AcIstft` of the estimate and of the apply lane) lives inside `dfn_rate_bridge`, and the DFN stage doc forbids a
  DFN reset on an AEC delay change, so a hard post-chain reset could not be followed there: the pre-reset overlap-add tail
  leaked into the first post-reset hop, and at 16 kHz three blind-set clips differed from the full-mode `aec_process()` chain
  exactly at their hard-reset hops. The call clears only the overlap-add tail and frame counter of the two native-grid
  synthesis objects. Kept: the stage (model state, DF history, lookahead delay lines, counters), both resamplers, both
  48 kHz FIFOs and analysis histories, the 48 kHz synthesis, the output FIFO with its prefill, the diagnostics counters; the
  added delay is unchanged. The synthesis runs at the top of `begin()` with no lookahead in front of it, so the call goes in
  on the hop of the event, before `begin()`/`process()`, with no delay line (the 48 kHz path delays its clear by
  `DFN_RES_STAGE_LOOKAHEAD_FRAMES` hops because its synthesis sits behind the stage). Returns -1 and clears nothing while a
  hop is in progress. Callers: `mono_aec_dfn_res/main.c` (`follow_post_reset()`), `4ch_aec_bf_dfn_res/main.c` (`dfn_form()`,
  bridge path only: the 48 kHz path still resets the whole hop), `mono_aec_dfn_res/pipeline.py` (`run_dfn2_res()`, shared by
  the 4ch Python twin), and the board sample `alg_ai3_aec_dfn.c` (8/16 kHz path). Not changed, and why: `uvc_audio_live2a.c`
  pairs every serial move (`mcab_reset()`) with `uvc_dfn_reset()`, which resets the whole bridge; `alg_ai3_dfn.c` and
  `uvc_ai_nr_stream.c` have no AEC and no bridge.
- `dfn_rate_bridge_get_host_hops()` / `Dfn2RateBridge.host_hops`: the native-grid hops the bridge synthesised from the last
  hop's spectra (diagnostics; the test tap).
- `DFN_RATE_BRIDGE_LAYOUT_VERSION` stays 3: two new functions over existing objects, no struct, pool or descriptor change.

### Output and memory
- CNG-on rows change with the seed (42 instead of `0x9e3779b9`): the noise realisation moves, the amplitude rule and the
  draw schedule do not. Mono: every CNG-on row of the baseline matrix differs. 4ch: of the 216 rows of the baseline matrix,
  188 are byte-identical to the previous tree; the 28 that differ are the `mcab_raw` rows (comfort noise on, no switch), and the
  seed alone (previous tree with seed 42) already changes all 28. CNG-off rows are not touched by the seed.
- Hops where the guard fires or the filter restarts move to the full-mode family; the analysis seam of the lane no longer
  takes the library's time-domain capture fallback, so the first differences can start earlier than the first guard hop.
  In the 4ch flow only hops where a contributing lane's guard fired (and the next hop, through the overlap-add) change.
  `--aec-only` is unchanged byte for byte.
- Pool sizes (ne10, `SIMD=1`, MATCHED n=5; `--print-mem-size` or the banner of the program), previous tree -> now. Mono,
  16k/256 / 16k/512 / 48k/1024: 423,728 -> 424,288, 575,056 -> 576,128, 1,354,256 -> 1,356,352 B (the AEC pool grows by
  lib/aec, 364,416 -> 366,000 / 491,600 -> 494,720 / 1,156,544 -> 1,162,736 B, and the synthesis shrinks by the borrowed
  window, 11,344 -> 10,320 / 21,840 -> 19,792 / 42,832 -> 38,736 B). 4ch unit (flow plus helper): 1,698,784 -> 1,706,240,
  2,992,416 -> 3,007,040, 6,300,352 -> 6,329,312 B; mcab's own pool 73,504 -> 74,624 B and one lane 199,408 -> 200,992 B at
  16k/256. `mono_aec_dfn_res` at 16 kHz / 48 kHz / 8 kHz: 716,400 -> 717,984, 1,443,712 -> 1,449,904, 621,024 -> 622,608 B;
  `4ch_aec_bf_dfn_res` at 16 kHz / 48 kHz: 2,058,672 -> 2,066,128, 6,616,928 -> 6,645,888 B; `mono_alignulcnet` at 16 kHz:
  534,240 -> 537,360 B; `4ch_alignulcnet` unit pool at 16 kHz: 2,982,080 -> 2,994,640 B. The tables in the READMEs and `docs/`
  carry the new values.

### Added (tests)
- `test_mono_chain`: `test_full_mode_identity` (rig of `main.c`, hand-written chain and `aec_process()` bit for bit at
  16k/256, 16k/512 and 48k/1024 on moving-echo, comfort-noise-off, never-usable and `aec_reset` streams; four planted
  defects detected), `test_nr_keeps_running` (and a planted NR reset), black boxes of both programs (`--no-nr` against
  `aec_process()`, the full chain against the hand-written chain, across guard hops and reset events).
  `tests/guard_scene.h` is the shared stream generator.
- `test_mono_dfn`: P chain against `aec_process()` at 16k/256, 16k/512, 48k/1024; 48 kHz stage identity with the delayed
  OLA clear; planted defects (no delay line, no clear, error_spec, no reseed). The `keep_ola_on_reset` workaround of the
  bridge rows is gone (the field of `conv_chain_mono.h` too); the pass-band rows run on streams that carry hard-reset events
  and reject a missing or late clear; new scene rows at 16k/256, 16k/512 and 8k/256 compare the bridge's apply lane with the
  hand-written chain bit for bit (guard hops, delay changes, `aec_reset`, never-usable stream; max deviation from
  `aec_process()` below 1e-7, the window's last bit) and its estimate lane with an independent synthesis of the rendering base.
- `test_dfn_rate_bridge`: the tap equals an independent audiolib synthesis bit for bit with the clear on the named hop (one
  hop differs from the uncleared run, no other), a clear one hop late does not, the stage and bridge counters are unchanged,
  the output difference carries exactly the cleared tail's energy (0.998..1.000; clearing the 48 kHz synthesis or the
  resamplers as well gives 1.2..2.1) and resumes bit-identical once it has drained; NULL and a hop in progress are refused.
- `test_4ch_dfn`: flow-reset rows at 16k/256 and 16k/512 (unit and RES restarted, the hop and DFN2 not): both lanes bit for
  bit, DFN2 frame count without a gap; a missing and a late clear are detected.
- `test_mcab`: `test_render_base_contract` (three grids, MATCHED and EXTERNAL_ALIGNED; all / some / no lanes firing,
  five weight shapes, trusted and rebuilt beams, a double reference of the projection, unit weight = the lane's own
  base, the reset serial across realigns and a reset) and `test_unit_weight_matches_full_mode` (unit weights rendered
  from `frame.b` with a full-mode twin's gain equal that twin's `aec_process()` bit for bit, guard hops and a reset
  included; rendering `e`, ignoring the serial and the old seed are each detected).
- `mono_aec_nr_res/tests/test_full_mode_identity.py`; `test_dfn2_rate_bridge_c_parity.py` (C against Python, with resets);
  `test_dfn2_pipeline_contract.py` (comfort noise on, reset event; 16 kHz through the bridge, bit-level against the
  conventional chain; no call and a late call fail); `4ch_aec_bf_nr_res/tests/test_pipeline.py::test_rendering_base_is_the_weighted_lane_bases`
  (the Python rule).
- `docs/c_python_function_map.md`: PL-DFN-13.

## [Unreleased] — 2026-10-28 — The example programs do the inverse FFT and overlap-add through `ac_istft`; `--aec-only` no longer calls `aec_process()`

Rule: the inverse FFT and the overlap-add are one independent function the application calls (audiolib `ac_istft_*`),
not code inside an algorithm library, and the example programs follow it. `aec_process()` stays a public library API
(unchanged); the example hop and the harness hops of `mono_aec_nr_res` no longer call it, and the tests call it only as
the independent oracle.

### Changed
- `mono_aec_nr_res/main.c` (`aec_nr_pipeline`, `aec_nr_pipeline_static`): with `--aec-only` the hop is `aec_process_context()`,
  `aec_get_res_context()` and a copy of `ctx.linear_hop` into `out`, instead of `aec_process()`. The lane configuration
  (`enable_res = 0`, `return_res_context = 1`) makes `aec_process()` return exactly that buffer (the PBFDKF overlap-save
  residual, `raw_output`; no synthesis runs on that path), so the output WAV, the exit code and every printed line are
  unchanged and there is still no synthesis lag. The default path was already `aec_process_context()` ... `ac_istft_push()`.
- Tests: `rig_hop()` in `test_mono_chain.c` and `conv_mono_hop()` in `conv_chain_mono.h` take the same path for an
  `aec_only` plan (the latter's `aec_process()` branch was never exercised before). READMEs, `docs/` and the HTML pages that
  said `--aec-only` is `aec_process()` now describe `ctx.linear_hop`.

- `DFN2Prepost`, `DFN3Prepost`, `UlcnetPrepost`, `DeepVqePrepost` in `*_IO_TIME` borrow the application's `AcIstft` (new `istft` field of the config, validated against the model's `*_framing_config()`, reset together with the class; `deepvqe_framing_config()` is new) instead of carving one in their pool, so no model directory sizes or constructs a synthesis object. The pools shrink by one `AcIstft` and `*_PREPOST_CARVE_VERSION` is 8 / 8 / 7 / 6 (stale pools are refused by `_init_ex`); the `*_IO_TIME` output is bit-identical to before (SIMD 1 and 0), the `*_IO_FREQ` pools and outputs do not change. The four board samples that use `*_IO_TIME` build the `AcIstft` themselves. New gates: `make audit-no-transform` (no inverse-FFT / overlap-add code in `AINR/` and `AIAEC/`) and the DFN3 `IO_TIME` vs `IO_FREQ` bitwise test in `test_dfn3_stream.c`.

### Added (tests)
- `test_aec_only_output`: the `--aec-only` hop of the rig and of the harness equals `aec_process()` on a second instance of
  the same lane configuration, bit for bit, over 400 hops at 16 kHz/256, 16 kHz/512 and 48 kHz/1024, and the same run
  requires the hops where `ctx.formed_hop` differs from `ctx.linear_hop` to be non-zero (242 / 125 / 299 in the SIMD=1
  build), so the comparison can tell the selected signal from the linear one.
- `test_examples_aec_only`: both programs run with `--aec-only` on a 400-hop WAV pair (16 kHz/256, 16 kHz/512 aggressive,
  8 kHz level 3); their output WAV must equal the oracle's WAV (written through the same WAV writer) sample for sample,
  with no extra hop at the end.
- Both were mutated from a saved copy (`ctx.formed_hop` copied instead of `ctx.linear_hop`): all 6 program checks fail with
  the `main.c` mutation, and the 3 in-process checks fail with the mutation of `rig_hop()` and `conv_mono_hop()`.

### Not changed
- Stage-owned synthesis: `dfn/dfn_rate_bridge` (any rate other than 48 kHz) and `ulcnet/ulcnet_nn_stage` keep their own
  `AcIstft`; making the caller own it is an interface change to those example stages and is not part of this entry. At 48 kHz
  the DFN mains already synthesise with the caller's `AcIstft` through `dfn_hop`.
- `test_mono_chain.c` (bypass check) and `4ch_aec_bf_nr_res/tests/test_mcab.c` keep a hand-written `fft_inverse` plus
  overlap-add as an independent reference for `ac_istft`; they are oracles, not example code.
- Python is not changed. `--aec-only` in the Python pipeline already uses the standalone time output, which is the buffer the
  C program now copies.

## [Unreleased] — 2026-10-28 — C strength setting named level in the examples, tests and READMEs (follows lib/aec and lib/nr)

Pure rename; no DSP, default output or exit code of a valid command line changed. Baseline matrix (725 rows, s1/s0/g48):
701 rows identical (every run row of the mono, `mcab_*`, DFN2 and ULCNet programs, with produced files); the 22 that differ
are the 18 known ones from the earlier rounds plus the usage text listed under "Printed text".
**C level == Python preset.** Python is unchanged and keeps `preset`.

### Changed
- Callers follow the library rename (`lib/aec`, `lib/nr`); the old function names are gone, so the examples, tests and
  sample code use only the new ones:

  | Old C name | New C name |
  |---|---|
  | `AecPreset`, `AEC_PRESET_{MILD,BALANCED,AGGRESSIVE,NO_FLOOR}` | `AecLevel`, `AEC_LEVEL_*` |
  | `aec_config_from_preset` / `aec_set_preset` / `aec_res_set_preset` | `aec_config_from_level` / `aec_set_level` / `aec_res_set_level` |
  | `AnrMode`, `ANR_MODE_{MILD,MODERATE,BALANCED,AGGRESSIVE}` | `AnrLevel`, `ANR_LEVEL_*` |
  | `anr_set_mode` / `anr_config_for_mode` / `anr_config_for_mode_grid` | `anr_set_level` / `anr_config_for_level` / `anr_config_for_level_grid` |
  | `anr_nr_mode_is_valid` / `_from_name` / `_name` | `anr_nr_level_is_valid` / `_from_name` / `_name` |
  | `mono_aec_nr_res/main.c`: `parse_preset`, `preset_name`, `parse_nr_mode`, `nr_mode_name` | `parse_level`, `level_name`, `parse_nr_level`, `nr_level_name`, plus `is_nr_level_option` |
  | `mcab_external_static.c`: `parse_preset`, `preset_name`, `nr_mode_name` | `parse_level`, `level_name`, `nr_level_name` |
  | test harness fields `aec_preset` / `nr_mode` (`conv_chain_mono.h`, `conv_chain_mcab.h`, `test_mono_ulcnet.c`) | `aec_level` / `nr_level` |
  | tests `test_caller_res_reset_and_preset`, `test_caller_res_ramped_preset`, `test_caller_res_preset_ramp_vs_oracle`, `test_nr_set_mode` | `..._level`, `test_nr_set_level` |

  The libraries keep `AecPreset`, `AEC_PRESET_*`, `AnrMode` and `ANR_MODE_*` as legacy names with literal values; nothing
  in this tree uses them.
- **Command lines.** `aec_nr_pipeline`, `aec_nr_pipeline_static` and `mcab_external_static` document `--nr-level
  mild|moderate|balanced|aggressive` (or `0`..`3`); `--nr-preset` is the same option under its earlier name, with the same
  value parsing, the same fallback (mono) or exit 2 (`mcab_external_static`) for an unknown name, and the same
  flag-as-value rule (a token starting with `--` is the next option, so the value is missing: exit 2). The same three
  programs take the AEC level as the positional argument (`mild|balanced|aggressive|no_floor` or `0`..`3`, unchanged, now
  called `[aec-level]`) or as the new option `--aec-level <level>`; both set the same value and use the same parser. In the
  mono programs an unknown name falls back to balanced; `mcab_external_static` exits 2 for an unknown name. A second AEC
  level in any mix of the two forms is a command-line error (exit 2, message `a second AEC level '<x>'; give one`) in
  all three programs, so the mono programs no longer keep the last one given; their usage text says so. The board samples
  that take these options (`--aec-level`/`--level` and `--nr-level`, aliases `--aec-preset`/`--preset`/`--nr-mode`) now
  match a level name or one of the digits `0`..`3` exactly and fall back to balanced for anything else, where
  `--nr-level` read the value with `atoi()` and turned `moderate` and `aggressive` into mild. The DFN2 and ULCNet programs have no strength
  option and gain none.
- READMEs under `pipelines/` say level for the C names and keep Python names (`set_preset()`, `from_preset`, the Python
  `--nr-preset`, `PostBeamResidualSuppressor(..., preset)`) where they describe Python.

### Printed text
- Usage text of the mono programs (stdout, no arguments) and of `mcab_external_static` (stderr, unknown option) names
  `--nr-level` and says `--nr-preset` is the same option. These are the only new matrix differences: `mono_err_noargs_*`
  (stdout, 4 rows) and `ext_err_unknown_flag` (stderr, 2 rows, which already differed in the exit code).
- Diagnostics on stderr: `mcab_external_static` now prints `<flag as typed> '<x>' is not an NR level`, `AEC level '<x>' is not
  a level` and `a second AEC level '<x>'` (they said preset); `pyc_drv` prints `unknown aec level` / `unknown nr level`.
- Unchanged on purpose: the run banner `AEC preset: <name>   NR preset: <name> ...` of the mono programs and
  `mcab_external_static` (logs of earlier runs and the baseline matrix compare it), the `pyc_drv` key=value grammar
  (`aec=`, `nr=`) and the pyc-parity scenario ids.

### Tests
- `4ch_aec_bf_nr_res/tests/test_cli_strict.sh` 486 -> 751 command lines (the 486 are unchanged and pass): `--aec-level` and
  `--nr-level` in every missing-value, flag-as-value, bad-name and valid case next to `--nr-preset` and the positional level;
  second-level cases in every mix of positional and option; the banner names the AEC and NR level in force for each of the
  4 names and 4 digits through every spelling (so a flag that is parsed but ignored fails); the spellings print the same
  output; the usage names them and states the one-AEC-level rule.
- `mono_aec_nr_res/tests/test_mono_chain.c`: the black-box cases run the programs with `--nr-level`, `--nr-preset` and
  `--aec-level` (names, digits, several on one line, positional and option mixed, the `no_floor` and digit levels) and
  compare every hop with the oracle; missing-value and `-x` cases for the two new options; the usage names them; ten second-AEC-level command lines per program
  (and six with `--print-mem-size`) exit 2 and the message names the problem (the earlier last-one-wins black-box case
  `aggressive --aec-level mild` is gone). Each new check was shown to fail on a
  mutated program (`--nr-level`, `--nr-preset` or `--aec-level` unknown in the mono programs, `--aec-level` value ignored there, flag
  guard removed there, `--nr-level` or `--aec-level` unknown or its value ignored in `mcab_external_static`, the
  second-level check skipped, the second-level latch removed in the mono programs, usage text changed in either).

### Compatibility
- Link-incompatible for prebuilt objects that call the renamed library functions: recompile and relink.
- Python unchanged. Stale `preset` wording about C names remains in Python comments at
  `4ch_aec_bf_nr_res/pipeline.py:599,686`, `4ch_aec_bf_nr_res/tests/test_pipeline.py:637,662` and
  `tests/pyc_parity/pyc_parity.py:35`.

## [Unreleased] — 2026-10-28 — Offline research drivers archived out of `pipelines/tools/`

### Removed
- `pipelines/tools/` keeps only `eval_dfn2_4ch_inputs.py` and `profile_dfn2_ops.py`. `__init__.py`, `rebench_aec_only.py`,
  `rebench_joint.py`, `rebench_sep_vs_classic.py`, `bench_dnsmos.py`, `compare_res_vs_nr.py`, `eval_dfn2_ood.py`,
  `eval_dfn2_variants.py` and `eval_pipeline_blind.py` had no caller in the build, the tests or the gates. They are archived
  outside the repository and stay in git history through `d4e1230`; the archive is not runnable as is. The two kept
  drivers run without `__init__.py`.

### Changed
- `README.md` (the `tools/` folder), `dfn/mono_aec_dfn_res/README.md`, the `CResampler` docstring of
  `dfn/dfn2_rate_adapter.py`, `docs/architecture_detail.html`, `docs/html/pipeline_dfn_mono.html` and
  `docs/release_checklist_03_docs_publish_appendices.md` no longer name the archived drivers or `AEC/eval/ab_evidence/`.
  No code or output change.

## [Unreleased] — 2026-10-27 — Follow-up to the mcab round: strict CLI numbers in the static examples, per-goal producer scopes of the 4ch Makefile, exact rate-bridge delay, lib/aec rate guard, DFN2 frames per call (AINR)

No DSP changed; for valid input the example programs' outputs are those of the previous entry (the 136 baseline-matrix
rows of `mcab_raw`, `mcab_static` and `mcab_external_static` were compared before and after: only the three exit codes
listed under "Behaviour changes" differ). The AEC behaviour hash `aec_python_behavior_hash()` is unchanged.

### Changed
- **`mcab_static` and `mcab_external_static` parse their numeric options strictly** (`--sample-rate`, `--fft-size`,
  `--uca-radius-m`, `--fixed-doa-deg`, `--capture-proxy`) with `example_cli_parse_int()` / `_float()`, like `mcab_raw`, the
  DFN2 and the ULCNet examples: a bad number, an option without its value and an unknown option exit 2. The shared
  `mcab_sample_parse_spatial_cli()` no longer has an `atoi` / `atof` path; its `raw` argument now only selects the option
  set (9 versus 5 options) and the parse-time checks of the grid, the radius and the file options. A number that parses but
  names no grid or channel (`--sample-rate 44100`, `--capture-proxy 4`) is still refused when the config is built and exits
  1. Still lenient: a repeated option keeps the last value (the preset names are strict in `mcab_external_static`, see the
  next entry). Only `mono_aec_nr_res/main.c` still reads its numbers with `atoi()` (and the test helpers `pyc_drv.c` and
  `dump_delay_parity.c`). New `4ch_aec_bf_nr_res/tests/test_cli_strict.sh <bin_dir>` runs the real binaries over 486 command
  lines and is part of `make -C 4ch_aec_bf_nr_res test`; `tests/test_mcab_sample_driver.c` covers the shared parser in both
  `raw` modes.
- **A token that starts with `--` where an option's value belongs is a missing value** (exit 2, message on stderr), not the
  value. This holds for every value option of `mcab_static`, `mcab_external_static` and `mcab_raw` (the shared
  `mcab_sample_scan_cli`, which `4ch_aec_bf_dfn_res` and `4ch_alignulcnet` use as well), for `--nr-preset`, `--sample-rate` and
  `--fft-size` of `aec_nr_pipeline` / `aec_nr_pipeline_static` (including their `--print-mem-size` path) and for every value
  option of `mono_aec_dfn_res` (`dfn/dfn_example_args.h`: `--sample-rate`, the delay options, `--erb-fwd`, `--erb-inv`,
  `--atten-lim`). `--nr-preset --no-res` used to take `--no-res` as the preset name and leave RES on; `--erb-fwd --cng` used
  to take `--cng` as a file name. A lone `-` and negative numbers (`-1`, `-x`) stay values, so a file named `--x` must be
  written `./--x`. New helpers `example_cli_is_option()` and `example_cli_option_value()` in `common/example_cli.h`; the
  private copy in `mono_aec_nr_res/main.c` is gone. `mono_alignulcnet` already refused such a token (it is neither a delay mode
  nor a number) and is unchanged. Valid command lines give the same output. `test_cli_strict.sh` now runs 486 command lines
  (was 198); the mono rejections are checked by `test_mono_chain` (`test_examples_cli_values`) and the `mono_aec_dfn_res`
  ones by the `make test` recipe of `pipelines/Makefile` and `dfn/mono_aec_dfn_res/Makefile`.
- **`mcab_external_static` refuses names it does not know**: an unknown `--nr-preset` name or AEC preset and a second AEC
  preset exit 2 (they fell back to balanced, and the last preset won). Valid presets and all valid output are unchanged.
  `mono_aec_nr_res/main.c` (`aec_nr_pipeline`, `aec_nr_pipeline_static`) and the NR library's `example/main.c` still fall back to
  balanced for an unknown name.
- **`AINR/DeepFilterNet2`: `DFN2ModelIoDescriptor` is back to 18 fields / 72 bytes** (N = `input_frames` - 2, see
  `AINR/CHANGELOG.md`), so a board adapter built against the previous header no longer has its descriptor overrun by
  `dfn2_model_io_descriptor_default()` / `_validate()`. The pipelines DFN stage and rate bridge (N = 1) see no change: pool
  sizes, hashes and outputs are identical.
- **The 4ch Makefile resolves, and keys its build directories on, only the producers a goal needs.**
  `libaudio_pipeline_mcab.a` (and `lib`) needs audiolib common and lib/aec only (lib/nr, audiolib's spatial kernels may be
  absent; `mcab.o` is byte-identical in every scope); `test_mcab` adds lib/nr; every other goal, `all` and `test` included,
  keeps all four producers and its previous build signature. A goal that lacks a producer it needs stops and names it.
  `make -s print-lib-path` now prints the flow archive of its own (smaller) scope; `print-bin-dir` / `print-obj-dir` still
  print the directory of `make all`, and `PRINT_FOR=<goal>` selects the scope a query reports (`PRINT_FOR=all` locates the
  copy of the archive that `make all` builds). `scripts/test_build_isolation.sh` gains SP-S21..SP-S25 for these rules.
- **Rate bridge: exact added delay.** `dfn_rate_bridge_added_delay_exact(b, &numerator, &denominator)` returns the delay
  without the floor as `numerator / DFN_RATE_BRIDGE_DELAY_DENOMINATOR` native samples (the denominator is 6, not reduced;
  Python: `Dfn2RateBridge.added_delay_exact`). `dfn_rate_bridge_added_delay_samples()` is its floor, so its values (672 / 629
  / 330 for the shipped pair) and every output are unchanged; the private struct field `added_delay` became
  `added_delay_sixths` with the same size, so `DFN_RATE_BRIDGE_LAYOUT_VERSION` stays 3. The bridge test, the mono example
  test and the 4ch example test now compare against the exact delay for every lookahead pair (impulse centroid within 0.01
  sample of `hop + exact delay`) and keep their pass-band error gates (5e-3 and 1e-3) for the non-integer pairs too, with
  the references evaluated at the fractional time. Pair 0/1 gives 501 1/3 / 458 1/3 / 244 2/3 at 16 kHz hop 128 / 16 kHz
  hop 256 / 8 kHz hop 128. Python `pipeline.py` of the mono DFN example still aligns by the integer delay, and the two
  `main.c` print the floored one.
- **lib/aec** (submodule bump): `aec_config_defaults()` / `aec_config_from_preset()` give `filter_length = 0` (with the
  `fft_size = 0` they already gave) for a sample rate other than 8000, 16000 or 48000 instead of forming `sr * 64` in `int`
  (signed overflow for `sr >= 33554432`); `aec_get_mem_size()` returns 0, `aec_init()` NULL and `aec_create()` nonzero for
  such a config. The 8 / 16 / 48 kHz configs are byte-identical. The pipeline rejection paths for 44100 now see
  `filter_length = 0` instead of 2822.
- **`AINR/DeepFilterNet2`: N frames per invocation** (`export_onnx.py --frames-per-call N`, C build `DFN2_FRAMES_PER_CALL`,
  details in `AINR/CHANGELOG.md`). The pipelines DFN stage, rate bridge and `dfn2_stage.py` are one-frame units and stay at
  N = 1: building them with `-DDFN2_FRAMES_PER_CALL=<N>`, N > 1, stops at the `DFN2_MULTI_FRAME_HOST` `#error` of
  `dfn2_model_io.h`; at N = 1 the pool sizes and hashes are unchanged and `make test` / `make test-pairs` pass.
- `docs/c_python_function_map.md`: rows for `dfn_rate_bridge_added_delay_exact`, `int_option` and the DFN2 frame-slot
  helpers; the removed `mcab_sample_raw_int` / `mcab_sample_raw_float` are gone from the rows and the coverage table.

### Behaviour changes
- `mcab_static` exits 2 (was 1) for every command-line error: an unknown option or a missing value (both were 1) and a bad
  number (it was read by `atoi` / `atof`, so `--sample-rate 16000Hz` ran as 16000 and an empty `--fixed-doa-deg` as 0, and
  what was not a grid failed later at config, exit 1). `mcab_external_static` exits 2 for an unknown option (was 1) and,
  like `mcab_static`, for a bad number or a missing value (`--nr-preset` included), which the old parser accepted or
  refused with exit 1. Config refusals (a number that names no grid or channel) still exit 1. The baseline matrix has three
  rows that change, in the exit code only: `static_err_unknown_flag`, `static_err_missing_value` and
  `ext_err_unknown_flag` (136 rows compared, per the strict-parse run's `rowdiff.log`); it has no bad-number rows.
- Every example that takes a value option now exits 2 when the next token starts with `--` (see the entry above); a script
  that passed a flag-looking file name must write `./--x`.
- SDK note: the live2, live2a and live3 board samples now require `gsc_adapt_interval` / `bf_adapt_interval` in
  1..1048576 (`MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL`); a former 0 must be written as 1.
- `make -s print-lib-path` in `4ch_aec_bf_nr_res` prints the narrow-scope archive. A script that wants the archive next to
  the binaries of `make all` passes `PRINT_FOR=all`.

### Removed
- `McabSampleOps.cli_error_status` (every example passed 2): `mcab_sample_main()` returns 2 for a parse error.
- `tests/pyc_parity/mutation_proof.sh` and the `--mutation-proof` / `--make-test` options of `pyc_parity_pipelines.sh`; this was the only checked-in proof that the parity gate can fail.

## [Unreleased] — 2026-10-27 — mcab: the 4-channel flow layer replaces `4aec_nr_res` and `audio_pipeline_4ch`; `dfn/` and `ulcnet/` folders; the `se_*` glue, `samples/` and `third_party/` are gone; DFN2 lookahead pairs

Supersedes the 2026-10-24 entries that describe `FourAecNrRes`, `AudioPipeline4Ch`, their layouts 24 and 21 and
their pool sizes; those entries stay as history. No DSP was changed: for valid input the example programs' outputs
equal those of the previous tree (compared run by run with `SIMD=1`, `SIMD=0` and the 48 kHz grid). The changes a
user can notice are listed under "Behaviour changes".

### Changed
- **`mcab` is a flow layer, not a library.** `4ch_aec_bf_nr_res/mcab.{h,c}` (private `mcab_internal.h`) replaces
  the 4-lane core (`4aec_nr_res`) and the 4-channel unit (`audio_pipeline_4ch`). The application creates the four
  libaec `Aec` lanes and, in MATCHED mode only, one shared `DelayAec` with their own `get_mem_size` / `init`, and
  passes the pointers in `McabLibs` to `mcab_init_ex()`. `mcab` owns only flow state in its own pool: the shared-delay
  admission state machine, the alignment ring, the aligned-reference buffer, the per-lane views and the fuse buffers.
  It never allocates, resets by destroying or frees an injected object; `mcab_destroy()` ends the flow and touches
  no library object. Injected lanes and the estimator must be freshly initialised, or reset (`aec_reset()`,
  `delay_aec_reset()`, or `mcab_reset()` of the flow that ran them), before `mcab_init()`; `mcab` cannot see that an
  object has already run (a used estimator would hand the new flow a stale delay).
- **There is no beamformer inside `mcab`.** `mcab_fuse(p, weights, beamformed_error, frame)` takes the caller's
  weights `W[4][n_freqs]` and optionally the beamformed spectrum `E`, fills a `McabFrame` (declared in `mcab.h`) and
  leaves `frame->g_res_ctx` NULL (RES stays outside: the application creates a libaec `AecRes`). `McabConfig` holds
  flow parameters only (`sample_rate`, `fft_size`, `delay_mode`, `delay_num_filters`, `fixed_delay_samples`,
  `capture_proxy_channel`, the backward-quarantine pair, `delay_est_period_s`, `max_delay_ms`, `enable_post`); it has
  no `filter_length` and no beamformer field.
- **Lane contract split.** The lane parameters (preset, filter length, ...) belong to the application's `AecConfig`.
  `mcab_apply_lane_contract(&cfg, &lane)` stamps only the contract fields (`fft_size`, EXTERNAL_ALIGNED with
  `delay_num_filters = DA_NUM_FILTERS`, `fixed_delay_samples = -1`, `enable_delay_est = 0`, `enable_res = 0`,
  `return_res_context = 1`, `spatial_linear_context` = `AEC_CONTEXT_NO_GAIN` with `enable_post` else
  `AEC_CONTEXT_LINEAR_ONLY`, `delay_est_period_s`), is idempotent and refuses a lane built for another sample rate.
  `mcab_lane_config(&cfg, &lane)` is the convenience: balanced preset plus the contract.
  `mcab_delay_estimator_config()` gives the estimator's `(sample_rate, hop_size, num_filters)`; it is required only
  in MATCHED. `mcab_init()` checks what the public getters can show (each lane's hop and bin count, a context
  without a gain of its own, the post half against `enable_post`, an already aligned far, four distinct lanes; the
  estimator's bank size, rate and 48 kHz scratch). `Aec` has no config getter, so the lane parameters themselves are
  not checked.
- **Descriptor.** `MCAB_DESCRIPTOR_VERSION` 1 and **`MCAB_LAYOUT_VERSION` 25**, which describes mcab's own pool only
  (control block, mic lane, aligned far, fuse buffers when `enable_post`, alignment ring in MATCHED/FIXED);
  library objects are not in it. It replaces `FOUR_AEC_NR_RES_LAYOUT_VERSION` 24 and
  `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 21; `mcab_init_ex()` refuses descriptors carrying layouts 17 to 24
  (`test_mcab`), so a persisted descriptor must be queried again with `mcab_get_mem_requirements()`.
- **Archive.** `libaudio_pipeline_mcab.a` holds `mcab.o` only and references no SRP-PHAT/BF symbol;
  `libaudio_pipeline_4ch.a` is gone. `libdfn_res_stage.a` (built by `4ch_aec_bf_nr_res/Makefile`) is unchanged in role.
- **Sample-level helpers (header and source beside `mcab`, outside the archive).** `mcab_sample_flow.{h,c}` builds
  lanes, estimator and `mcab` in one caller pool (`mcab_sample_flow_get_mem_size/_init/_create/_destroy`).
  `mcab_sample_beam.{h,c}` owns audiolib's SRP-PHAT and BF and a fallback energy VAD and runs the per-hop glue
  (`doa_step()` + `bf_process_with_weights()`) between `mcab_process_front()` and `mcab_fuse()`
  (`mcab_sample_beam_process_front/_process_beam/_process_with_activity/_process/_fuse`); a front `DSP_ERROR` resets
  only the flow (SRP, BF, the VAD and `frame_index` stay), a refusal of `mcab_sample_beam_fuse()` other than a sequence error
  resets everything the helper owns. `mcab_sample_driver.{h,c}` is the CLI/driver. `mcab_sample_post.h` (header-only) carries the post-beam
  wiring of the 4-channel programs: `mcab_sample_res_input()` fills the `AecResInput` of the application's `AecRes`
  from a `McabFrame`, `mcab_sample_anr_gain()` returns the denoiser gain, and `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`,
  `MCAB_SAMPLE_CNG_RNG_SEED` and `MCAB_SAMPLE_PSD_SCALE_INV` are the comfort-noise and scale constants
  (`test_mcab_sample_post`). An application with its own beamformer calls `mcab_process_front()` and `mcab_fuse()`
  directly.
- **Programs.** `mcab_raw`, `mcab_static` and `mcab_external_static` write their application-side init out in the
  file; the DFN2 and Align-ULCNet 4-channel examples and the tests build through `mcab_sample_flow` plus the beam
  helper. `make PROFILE=1` sets `-DMCAB_STAGE_TIMING=1 -DAEC_STAGE_TIMING=1`.
- **Folders.** `pipelines/` holds `common/`, `mono_aec_nr_res/`, `4ch_aec_bf_nr_res/`, `dfn/`, `ulcnet/`, `tests/`,
  `tools/` and `scripts/`. `dfn/` holds `dfn_res_stage`, `dfn_rate_bridge`, `dfn_hop.h`, `dfn_hop_pool.h`,
  `dfn_unit_partition.h`, `dfn_example_args.h`, the Python twins `dfn2_stage.py` and `dfn2_rate_adapter.py`, `tests/`,
  `mono_aec_dfn_res/` and `4ch_aec_bf_dfn_res/` (each with its `pipeline.py` and `tests/`). `ulcnet/` holds
  `ulcnet_nn_stage.{c,h}`, `tests/`, `mono_alignulcnet/` and `4ch_alignulcnet/`. `common/` holds `example_cli.h`,
  the Python `cng_aec3.py` and `tests/`. `pipelines/tests/` keeps the cross-cutting tests (`test_c_python_map.py`,
  `test_doc_layout_versions.py`, `test_readme_flow.c`, `test_target_abi.c`, `pyc_parity*`, `c_shim.py`). Tests moved
  with their code. Cross-directory `#include`s are written relative to the `pipelines/` root (`"dfn/dfn_hop.h"`,
  `"common/example_cli.h"`), includes inside one family directory use the plain file name. `pipelines/Makefile` and
  `4ch_aec_bf_nr_res/Makefile` build, each folder's `Makefile` is a thin forwarder.
- **Python twins moved.** `dfn2_stage.py` and `dfn2_rate_adapter.py` are `pipelines/dfn/dfn2_stage.py` and
  `pipelines/dfn/dfn2_rate_adapter.py` (modules `pipelines.dfn.dfn2_stage`, `pipelines.dfn.dfn2_rate_adapter`);
  `aec_dfn2_res_pipeline.py` is `dfn/mono_aec_dfn_res/pipeline.py` (`python -m pipelines.dfn.mono_aec_dfn_res.pipeline`);
  the 4-channel DFN2 twin is `dfn/4ch_aec_bf_dfn_res/pipeline.py`; `aec_nr_pipeline.py` is
  `mono_aec_nr_res/pipeline.py` (`python -m pipelines.mono_aec_nr_res.pipeline`); `cng_aec3.py` is
  `common/cng_aec3.py`. The Python tests moved to `dfn/tests/`, `dfn/mono_aec_dfn_res/tests/`,
  `dfn/4ch_aec_bf_dfn_res/tests/`, `mono_aec_nr_res/tests/` (`test_signal_grids.py`) and `common/tests/`.
- **Example glue.** The `se_*` layer is gone: `common/se_config.{h,c}`, `se_stage.h`, `se_anr_stage.h`,
  `se_delay_cli.h`, `pipelines/se_stage.py` and `pipelines/third_party/gain` (`fix_gain`, `nr_gain`, `post_gain`).
  Each mono program calls libaec and libanr directly: `derive_lane()` resolves the grid with `aec_resolve_signal_grid()`
  and builds the lane `AecConfig` from `aec_config_from_preset()` plus explicit fields, the AEC's own RES gain is read
  with `aec_get_res_context()`, the NR gain comes from `anr_process_gain()` (`mono_nr_gain()` in `mono_aec_nr_res/main.c`),
  and the program chains gain minimum, `aec_cng_fill()` comfort noise (local `CNG_*` constants) and synthesis itself
  (`mono_form()`). The 4-channel programs use `mcab_sample_post.h` (above). The finite check of an emitted hop is
  `skn_all_finite_f32()` from audiolib's `utility/simd_kernel_nn.h`.
- **Strict number parsing.** `common/example_cli.h` (header-only; it also holds the `example_cli_delay_*` helpers
  that check the `--delay-mode` / `--delay-num-filters` / `--fixed-delay` combination) is the one parser for the
  numeric options of the DFN2 and Align-ULCNet examples (`dfn_example_args.h` builds on it) and of `mcab_raw`:
  `example_cli_parse_int()` / `example_cli_parse_float()` reject the empty string, a leading space, trailing
  characters, `ERANGE`, values outside `int` and (floats) `inf`/`nan`, and accept plain decimals with an optional
  leading `+`.
- **Mono.** `mono_aec_nr_res/main.c` and `static_main.c` are one `main.c`; it builds both binaries
  (`-DAEC_NR_POOL_SPLIT=1` gives `aec_nr_pipeline` its two-block pool; `aec_nr_pipeline_static` keeps
  `--print-mem-size`).
- **DFN2 lookahead pairs.** The DFN2 stage and rate bridge follow AINR's mask/deep-filter lookahead pair
  (`AINR/README.md`, "Lookahead pairs": 15 legal pairs, mask 0..2, deep filter 0..4, no ordering rule between them;
  the exported graph does not depend on the pair, a checkpoint is tied to it). The pair is set with the make
  variables `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` (default 1/1, one decimal in range, otherwise a Makefile
  error; `pipelines/Makefile` also takes `DFN3_MASK_LOOKAHEAD` / `DFN3_DF_LOOKAHEAD` for the DFN3 sources it compiles
  for the target-ABI test, `4ch_aec_bf_nr_res/Makefile` only the DFN2 pair). A pair other than 1/1 is forwarded as
  `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` to every unit and is part of the build signature, so it gets
  its own `obj`/`bin` directories; an out-of-range pair is a compile error. (Superseded by the 2026-11-05 entry above: the library
  has no pair symbol, a model or pool of another pair is refused at run time, and a pair with a ring deeper than 5 also needs
  `-DDFN2_MAX_DF_RING`.) `DFN_RES_STAGE_LOOKAHEAD_FRAMES` =
  `DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD` = L (the DFN2 cascade adds the two): the spectrum emitted on hop t belongs
  to source frame t - L and the first L hops emit zeros. The rate bridge adds `prefill + up and down resampler group delays (16 + 16 native samples) +
  (L + 1) * 512 * sr / 48000` native samples (`dfn_rate_bridge_added_delay_samples()`; integer arithmetic, floored, so a
  fraction of a sample is lost for some pairs):
  672 / 629 / 330 samples at 16 kHz hop 128, 16 kHz hop 256 and 8 kHz hop 128 for the shipped 1/1 pair; 501 / 458 / 244
  for 0/1 and 1354 / 1311 / 671 for 2/4. `make -C pipelines test-pairs` rebuilds and runs the DFN C tests and
  `mono_aec_dfn_res` for each pair in `DFN_TEST_PAIRS` (default `0/0 0/1 2/0 2/4`, e.g.
  `make -C pipelines test-pairs DFN_TEST_PAIRS="0/1 2/4"`); the 4-channel DFN2 example takes the same variables and has no `test-pairs` target.
  The Python `Dfn2Stage` takes the pair from the loaded model. Only the shipped 1/1 pair has a trained checkpoint, so
  the audio quality of any other pair is not measured, and the C builds were exercised for a subset of the pairs only.
- **Python twin of `mcab`.** Classes `Mcab*` (`McabConfig`, `McabPipeline`, `McabFront`, `McabBeam`, `McabFrame`,
  `McabDelayState`), `McabPipeline.process_front()`, `mcab_apply_lane_contract()`, `mcab_lane_config()`;
  `McabPipeline(config=None, beamformer=None, lane_config=None)` stamps the contract onto an application-supplied lane config.
  Python and C agree for the default settings; the documented differences S-02 (AEC live observer guard) and S-13
  (NR make-up prior) exist only when those default-off features are enabled.

### Behaviour changes
- Malformed numbers (`12x`, `3x`, an empty string, a leading space) given to `mono_alignulcnet`, `4ch_alignulcnet`,
  the DFN2 examples or `mcab_raw` exit with status 2 (the ULCNet examples used `atoi()` before and accepted them);
  `mcab_raw` with an empty `--fixed-doa-deg` exits 2. `aec_nr_pipeline`, `aec_nr_pipeline_static`, `mcab_static` and
  `mcab_external_static` still accept them through `atoi()` / `atof()`.
- `mcab_raw` reports a failed write of its output file.
- `mcab_fuse()` returns `MCAB_DSP_ERROR` and resets the flow when the lanes do not share one far spectrum, when the
  beamformed spectrum is non-finite or when the projection overflows (finite but huge weights).
- Injected lanes and the estimator must be fresh or reset before `mcab_init()` (see above).

### Removed
- `4aec_nr_res.{c,h}`, `4aec_nr_res_internal.h`, `4aec_projection_kernels.h`, `audio_pipeline_4ch.{c,h}`,
  `audio_pipeline_4ch_raw.c`, `audio_pipeline_4ch_static.c`, `4aec_nr_res_static.c`; the binaries
  `audio_pipeline_4ch_raw`, `audio_pipeline_4ch_static`, `4aec_nr_res_static`, `test_4aec_nr_res`,
  `test_audio_pipeline_4ch`; the `libaudio_pipeline_4ch.a` alias target; `mono_aec_nr_res/static_main.c`;
  `samples/mcab_demo.c`; `test_se_common` (replaced by `test_mcab_sample_post`).
- `pipelines/common/se_config.{h,c}`, `se_stage.h`, `se_anr_stage.h`, `se_delay_cli.h`, `pipelines/se_stage.py`.
- `pipelines/third_party/` (`gain`: `nr_gain`, `post_gain`, `fix_gain` and `test_gain`) and `pipelines/samples/`.
- Python: the `FourChannelAec*` names, with no alias.

### Pool sizes (NE10, `SIMD=1`, MATCHED n=5, 16 kHz/256 unless noted; bytes, as printed by `test_target_abi` and the example binaries)
| Item | Bytes |
|---|---|
| `mcab` pool alone (`enable_post = 1`, default config; `test_target_abi`) | 73,504 |
| `mcab` pool alone (`enable_post = 0`; `test_target_abi`) | 68,288 |
| `mcab_external_static` "Front end" (shared estimator + `mcab` pool), 16 kHz/256 | 108,928 |
| `mcab_external_static` "Front end", 48 kHz/1024 | 262,880 |
| `mcab_static` "Unit" (flow objects + beam helper), 16 kHz/256 | 1,698,784 |
| `mcab_static` "Unit", 48 kHz/1024 | 6,300,352 |

The four lanes (printed as "AEC x4" by `mcab_external_static`: 797,632 at 16 kHz/256) and the shared estimator are
sized by their own libraries and are part of the application's total: "Front end" is the estimator plus the `mcab`
pool, "Unit" adds the four lanes and the beam helper. Re-query every number with `mcab_get_mem_requirements()`,
`aec_get_mem_size()`, `delay_aec_get_mem_size()` and `mcab_sample_beam_get_mem_size()` on the target build; do not copy
these.

### Migration notes
| Before | After |
|---|---|
| `FourAecNrRes`, `FourAecNrResConfig`, `FourAecNrResStatus`, `FOUR_AEC_NR_RES_*` | `Mcab`, `McabConfig`, `McabStatus`, `MCAB_*` (`MCAB_OK`, `MCAB_INVALID_ARGUMENT`, `MCAB_SEQUENCE_ERROR`, `MCAB_DSP_ERROR`) |
| `four_aec_nr_res_default_config()`, `_get_mem_requirements()`, `_init()`, `_init_ex()`, `_create()`, `_destroy()`, `_reset()` | `mcab_default_config()`, `mcab_get_mem_requirements()`, `mcab_init()`, `mcab_init_ex()`, `mcab_create()`, `mcab_destroy()`, `mcab_reset()`; init now also takes `const McabLibs*` (lanes and, in MATCHED, the estimator, created by the application) |
| `four_aec_front_process()`, `four_aec_fuse()` | `mcab_process_front()`, `mcab_fuse()` |
| `four_aec_nr_res_hop_size()`, `_fft_size()`, `_n_freqs()`, `_sample_rate()`, `_matched_filter_count()`, `_linear_aec_count()`, `_far_fft_real_compute_count()`, `_realign_warm_lane_count()`, `_realign_soft_lane_count()`, `_pending_delay_candidate()`, `_duty_hops_total()`, `_duty_hops_run()`, `_get_last_timing()`, `_get_mem_breakdown()` | the same accessors with the prefix `mcab_` (`mcab_hop_size()`, ...); new: `mcab_enable_post()` |
| `FourAecFront`, `FourAecNrResDelayState`, `FourAecNrResLastTiming`, `FourAecNrResMemBreakdown` | `McabFront`, `McabDelayState`, `McabLastTiming`, `McabMemBreakdown` (`control_bytes` / `buffer_bytes` / `total_bytes`; no lane bytes) |
| `FourAecNrResConfig.filter_length` | the application's lane `AecConfig`; apply the contract with `mcab_apply_lane_contract()` or take `mcab_lane_config()` |
| `AudioPipeline4Ch`, `audio_pipeline_4ch_*` (SRP-PHAT/BF inside the unit) | `mcab` plus `McabSampleBeam` / `mcab_sample_beam_*`; `AudioPipeline4ChBeam`, `AudioPipeline4ChFrameInfo`, `AudioPipeline4ChGeometry` become `McabSampleBeamResult`, `McabSampleBeamInfo`, `McabSampleGeometry`; the beam-related `AudioPipeline4ChConfig` fields become `McabSampleBeamParams` |
| `audio_pipeline_4ch_get_core()`, `_last_core_timing()` | the flow handle you passed to `mcab_sample_beam_init()`; `mcab_get_last_timing()`. The `audio_pipeline_4ch_doa_*` / `_bf_*` grid getters are removed (they equalled the main grid); `_bf_effective_adapt_interval()`, `_bf_lambda()` and `_spatial_backend()` became `mcab_sample_beam_bf_effective_adapt_interval()`, `mcab_sample_beam_bf_lambda()` and `mcab_sample_beam_spatial_backend()` |
| `FOUR_AEC_NR_RES_LAYOUT_VERSION` 24, `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 21 | `MCAB_LAYOUT_VERSION` 25 |
| `FOUR_AEC_NR_RES_STAGE_TIMING` | `MCAB_STAGE_TIMING` |
| `libaudio_pipeline_4ch.a` | `libaudio_pipeline_mcab.a` (`mcab.o` only); add `mcab_sample_flow.o` / `mcab_sample_beam.o` if you use the helpers |
| binaries `audio_pipeline_4ch_raw`, `audio_pipeline_4ch_static`, `4aec_nr_res_static` | `mcab_raw`, `mcab_static`, `mcab_external_static` |
| tests `test_4aec_nr_res`, `test_audio_pipeline_4ch`, `tests/conv_chain_4ch.h` | `test_mcab`, `test_mcab_sample_beam`, `tests/conv_chain_mcab.h` (new `tests/mcab_test_rig.h`) |
| `FourChannelAecConfig`, `FourChannelAecPipeline` (`front_process`), `FourChannelFrame`, `FrontFrame`, `BeamformerFrame`, `SharedDelayState` | `McabConfig`, `McabPipeline` (`process_front`), `McabFrame`, `McabFront`, `McabBeam`, `McabDelayState` |
| `pipelines/dfn_res_stage.*`, `dfn_rate_bridge.*`, `dfn_unit_partition.h`, `dfn_example_args.h` | `pipelines/dfn/...` |
| `pipelines/samples/dfn_hop.h`, `samples/ulcnet_nn_stage.*` | `pipelines/dfn/dfn_hop.h`, `pipelines/ulcnet/ulcnet_nn_stage.*` |
| `common/se_stage.h` (`SeSpecFrame`, `se_frame_from_aec()`, `se_res_stage_gain()`), `samples/se_anr_stage.h` (`se_anr_stage_gain()`) | `aec_get_res_context()` (`AecResContext.res_gain`), `aec_res_process()` over `mcab_sample_res_input()`, `mono_nr_gain()` / `mcab_sample_anr_gain()` (`4ch_aec_bf_nr_res/mcab_sample_post.h`) |
| `common/se_config.{h,c}` (grid and lane derivation), `se_hop_is_finite()` | `derive_lane()` in each program over `aec_resolve_signal_grid()` and `aec_config_from_preset()`; `skn_all_finite_f32()` |
| `common/se_delay_cli.h` (`se_cli_parse_int()`, `se_cli_parse_float()`) | `common/example_cli.h` (`example_cli_parse_int()`, `example_cli_parse_float()`) |
| `pipelines/third_party/gain` (`fix_gain`, `nr_gain`, `post_gain`) | `mono_nr_gain()` / `mcab_sample_anr_gain()` for the NR gain; an input gain is `sk_scale_f32()` plus `sk_clip_f32()` from audiolib's `utility/simd_kernels.h` |
| `pipelines/mono_aec_dfn_res/`, `4ch_aec_bf_dfn_res/`, `mono_alignulcnet/`, `4ch_alignulcnet/` | `pipelines/dfn/mono_aec_dfn_res/`, `pipelines/dfn/4ch_aec_bf_dfn_res/`, `pipelines/ulcnet/mono_alignulcnet/`, `pipelines/ulcnet/4ch_alignulcnet/` |
| `pipelines/dfn2_stage.py`, `dfn2_rate_adapter.py`, `aec_dfn2_res_pipeline.py`, `aec_nr_pipeline.py`, `cng_aec3.py` | `pipelines/dfn/dfn2_stage.py`, `pipelines/dfn/dfn2_rate_adapter.py`, `pipelines/dfn/mono_aec_dfn_res/pipeline.py`, `pipelines/mono_aec_nr_res/pipeline.py`, `pipelines/common/cng_aec3.py` |
| `pipelines/tests/test_dfn_res_stage.c`, `test_dfn_rate_bridge.c`, `dfn_test_fixture.h`, `test_dfn2_*.py`, `test_signal_grids.py`, `test_cng_aec3.py` | `pipelines/dfn/tests/...`, `pipelines/dfn/mono_aec_dfn_res/tests/`, `pipelines/dfn/4ch_aec_bf_dfn_res/tests/`, `pipelines/mono_aec_nr_res/tests/test_signal_grids.py`, `pipelines/common/tests/` |

Board code: create the four lanes with `aec_init()` over `mcab_lane_config()` (or your own `AecConfig` plus
`mcab_apply_lane_contract()`), the `DelayAec` when `mcab_delay_estimator_config()` reports `required`, then
`mcab_get_mem_requirements()` and `mcab_init_ex()`. Tear down in the opposite order: sample helper, `mcab_destroy()`,
the lanes. Board build notes: drop `fix_gain.c` and the `third_party/gain` include path (an input gain is
`sk_scale_f32()` plus `sk_clip_f32()`); when the DFN2 lookahead pair is not 1/1, pass the same
`-DDFN2_MASK_LOOKAHEAD` / `-DDFN2_DF_LOOKAHEAD` flags to every DFN2 unit and install a model exported for that pair.
`alg_ai3_dfn.c` and `uvc_ai_nr_stream.c` hard-code mask and DF lookahead 1 with a static assert. (Superseded by the 2026-11-05
entry above: the pair is a field of the `DFN2Grid` the application passes, so no pair flag is needed on a product unit and only a
ring deeper than 5 needs `-DDFN2_MAX_DF_RING` on every DFN2 unit; `alg_ai3_dfn.c` takes its grid and pair from its own table, and
`uvc_ai_nr_stream.c` builds its shipped 48 kHz 1/1 grid with `dfn2_grid_make()`.)

## [Unreleased] — 2026-10-24 — The 4-channel unit no longer carries the post-beam RES: applications create the AecRes with libaec

Supersedes the statements in the same-day "Pipeline usage fix" entry that the 4-channel unit owns the RES
(`enable_res` / `res_preset` fields, `audio_pipeline_4ch_res()`, `fuse()` filling `g_res_ctx`, the Python
pipeline owning the suppressor, layout 19 and hash 0x9a3abcc7). The removal of `samples/se_sample_chain.h`
recorded there stands. Mono is unchanged.

### Changed
- `AudioPipeline4Ch` is again only the shared delay aligner, the four AEC lanes, SRP-PHAT, BF and the `fuse()`
  projection. `AudioPipeline4ChConfig` has no `enable_res` / `res_preset`, there is no `audio_pipeline_4ch_res()`, and
  the unit pool carves no `AecRes`. `audio_pipeline_4ch_fuse()` leaves `frame->g_res_ctx = NULL`, like the core's
  `four_aec_fuse()`. `audio_pipeline_4ch_reset()` resets only the core, SRP, BF and the fallback VAD; the application
  calls `aec_res_reset()` itself.
- RES is managed by libaec. A 4-channel application that wants it builds it in its own init:
  `aec_config_from_preset(preset, sample_rate)` (with `fft_size` taken from the unit's grid), `aec_res_get_mem_size()`,
  `aec_res_init()`; per hop it calls `se_res_stage_gain(res, &frame)` (`common/se_stage.h`, a wrapper over
  `aec_res_process()`); strength is `aec_res_set_preset(res, preset, ramp_ms)`; reset is `aec_res_reset()` (there is
  no `aec_res_destroy`: the header states there is nothing to release). Examples: `4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c`,
  `audio_pipeline_4ch_static.c`, `4aec_nr_res_static.c` (core) and `4ch_aec_bf_dfn_res/main.c`. `4ch_alignulcnet` runs the
  unit with `core.enable_post = 0` and has no RES.
- Per-hop order of a 4-channel application: `process` (delay, lanes, SRP-PHAT, BF) -> `fuse` (E, R2, CN; `g_res_ctx`
  NULL) -> `se_res_stage_gain()` -> NR gain -> `sk_min_f32()` -> CNG -> synthesis. A refused call (including a NULL
  RES gain) resets the unit and every post stage together; a refusal of the unit's own `process` also resets the post
  stages in `audio_pipeline_4ch_raw` and `audio_pipeline_4ch_static`.
- Mono is unchanged: the RES gain is computed inside `aec_process_context()` and handed over as `frame.g_res_ctx` by
  `se_frame_from_aec()`, strength through `aec_set_preset()`; a pure-AEC example may call the full `aec_process()`
  (RES and the final synthesis in one call).
- `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 20 -> **21**; the unit's `build_flags_hash` (the carve token no longer contains
  `res`) 0x9a3abcc7 -> 0x8aabdea5. `FOUR_AEC_NR_RES_LAYOUT_VERSION` stays **24**.
- The NR examples (`audio_pipeline_4ch_raw`, `audio_pipeline_4ch_static`, `4aec_nr_res_static`) produce output
  bit-identical to the previous commit.
- Python: `FourChannelAecPipeline` no longer owns the post-beam suppressor; `fuse()` only projects
  (`context.res_gain` is None). `PostBeamResidualSuppressor.from_aec_config()`, `FourChannelFrame.aligned_render` and the
  package export of `PostBeamResidualSuppressor` are back (as at 2a01f0e); the application builds the suppressor and calls
  `process(post.context, post.aligned_render)`, strength through its `set_preset()`.

### Removed
- `AudioPipeline4ChConfig.enable_res`, `AudioPipeline4ChConfig.res_preset`, `audio_pipeline_4ch_res()`.
- Python: `FourChannelAecConfig.enable_res` / `aec_preset`, `pipeline.post_beam_res`.

### Pool sizes (NE10, `SIMD=1`, matched n=5; bytes; as printed by `--print-mem-size` and the example binaries)
| Item | 16 kHz 256/128 | 16 kHz 512/256 | 48 kHz 1024/512 |
|---|---|---|---|
| 4ch unit alone (`audio_pipeline_4ch_get_mem_requirements()`) | 1,698,864 | 2,992,496 | 6,300,432 |
| `AecRes` carved by the example (`aec_res_get_mem_size()`) | 7,904 | 14,048 | 29,408 |
| Unit + post stages (`audio_pipeline_4ch_static`) | 1,765,552 | 3,088,960 | 6,525,488 |
| 4ch front end + own `AecRes` + post stages (`4aec_nr_res_static`, unchanged) | 973,776 | 1,519,968 | 3,402,064 |
| 4ch DFN2 (`4ch_aec_bf_dfn_res`, host side) | 2,058,752 | | 6,617,008 |
| 4ch Align-ULCNet unit (no RES) + stage (`4ch_alignulcnet`) | 2,982,160 + 36,400 | | 6,279,856 + 71,728 |

The unit alone is smaller than in the previous entry by the `AecRes` plus a 16 B control block (16 kHz/256:
1,706,784 -> 1,698,864); with `core.enable_post = 0` the unit is 1,693,648 at 16 kHz/256, 2,982,160 at 16 kHz/512 and
6,279,856 at 48 kHz/1024.

### Migration notes for board code
- Rebuild. `init_ex()` rejects unit descriptors of layout 18, 19 and 20; query them again.
- Code that set `cfg.enable_res` / `cfg.res_preset` or called `audio_pipeline_4ch_res(p)`: create the `AecRes` yourself
  (sequence above), use the return value of `se_res_stage_gain()` instead of `frame.g_res_ctx`, and add
  `aec_res_reset(res)` to your reset path.

## [Unreleased] — 2026-10-24 — AEC streaming FIFO removed: smaller lane pools

### Changed
- Every `Aec` instance is smaller (the render FIFO and the shrunken struct): 16 kHz/256 33,328 B, 16 kHz/512
  33,840 B, 48 kHz/1024 67,632 B, 8 kHz/256 16,944 B. The 4ch core and unit carry four lanes, so their pools
  shrink by 133,312 B (16 kHz/256), 135,360 B (16 kHz/512) and 270,528 B (48 kHz); the mono chains by one `Aec`.
  The pool tables in the docs and READMEs carry the values the example binaries print for this tree (rows
  without a binary, such as `num_angles = 360`, are the previous value minus the lane reduction).
- `FOUR_AEC_NR_RES_LAYOUT_VERSION` 23 -> **24** and `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 19 -> **20**: the pool size
  changed while `build_flags_hash` (a hash of the carve tokens) did not, so the version is the only signal for a
  persisted descriptor. The literal pins in `test_4aec_nr_res.c` and `test_audio_pipeline_4ch.c` and the
  documentation claims read by `tests/test_doc_layout_versions.py` follow.
- The 4ch core calls `aec_process_context_shared_far_from()` only; the removed `aec_process_context_shared_far()` is
  no longer named in the comments and READMEs.

## [Unreleased] — 2026-10-24 — Pipeline usage fix: the 4-channel unit carries the post-beam RES; no shared sample chain

Supersedes the statements of the pipeline split entry below that RES is a separate library for the
4-channel unit and that the gain sequence lives in `samples/se_sample_chain.h`.

### Changed
- `AudioPipeline4Ch` carries the post-beam RES. New config fields `enable_res` (default 1) and
  `res_preset` (default `AEC_PRESET_BALANCED`); the `AecRes` is carved after `bf_weights` only when
  `enable_res && core.enable_post` (the front-only Align-ULCNet profile carves none), from
  `aec_config_from_preset(res_preset, sample_rate)` on the unit's resolved FFT. `audio_pipeline_4ch_fuse()`
  runs it over the fused frame (`se_res_stage_gain()`) and sets `frame->g_res_ctx` (NULL with RES off); a
  RES refusal is `FOUR_AEC_NR_RES_DSP_ERROR` and resets the whole unit. `audio_pipeline_4ch_reset()` resets
  the suppressor and keeps a preset set at run time. New accessor `audio_pipeline_4ch_res(p)`; strength is
  `aec_res_set_preset(audio_pipeline_4ch_res(p), preset, ramp_ms)` (`NULL` without RES). The core
  (`FourAecNrRes`, `four_aec_fuse()`) is unchanged and has no RES.
- `samples/se_sample_chain.h` is deleted. Every example `main` initialises the libraries it uses
  (`Aec` or the unit, `AnrDenoiser` through `anr_config_for_mode_grid()` with the library-default `L`,
  `AcIstft`) and writes the per-hop chain itself: RES gain (`frame.g_res_ctx` or the unity vector),
  `se_anr_stage_gain()`, `sk_min_f32()`, `sk_capply_gain_f32()`, `aec_cng_fill()`, `ac_istft_push()`.
  `se_hop_is_finite()` moved to `common/se_stage.h`; `samples/se_anr_stage.h` is unchanged. A refused
  hop is reset by the integrator stage by stage, including `rng = SE_CNG_RNG_SEED`.
- Two examples still build their own `AecRes` because they drive the core directly with fixed weights:
  `4ch_aec_bf_nr_res/4aec_nr_res_static.c` (`--no-res`, `--legacy-amin`, `--no-cng` kept) and the parity
  driver `tests/pyc_parity/pyc_drv.c`. `AecRes` stays a stand-alone lib API for that use.
- `4ch_aec_bf_dfn_res/main.c` runs on the unit instead of the core with uniform weights: real SRP-PHAT and
  BF, RES inside the unit. Its pool grows by 791,664 B at 16 kHz and 3,123,312 B at 48 kHz (against the
  earlier core example on the same AEC sizes); its output was silent fail-open hops before and after.
- Python: `FourChannelAecPipeline` owns the one `PostBeamResidualSuppressor` again
  (`FourChannelAecConfig.enable_res` default True, `aec_preset` default balanced). `fuse()` applies it,
  `reset()` resets it, `pipeline.post_beam_res` mirrors the C accessor (strength:
  `post_beam_res.set_preset()`), and `FourChannelFrame.spec_frame()` carries `g_res_ctx = context.res_gain`.
- `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 18 -> **19**; the unit's `build_flags_hash` (carve token now
  `...,bf_weights,res`) 0x8aabdea5 -> 0x9a3abcc7. The FIFO entry above takes the version to 20 (the value in
  this tree). `FOUR_AEC_NR_RES_LAYOUT_VERSION` and the core's hash are not touched by this change.
- Output is bit-identical to the previous revision at `SIMD=1` and `SIMD=0` (golden subset 24 and 12
  keys; 672 chain-level runs of the unfolded sequences against the removed object; 82 Python digest rows).

### Removed
- `samples/se_sample_chain.h`: `SeSampleRes`, `SE_SAMPLE_RES_{OFF,FRAME_GAIN,AEC_RES}`, `SeSampleChainConfig`,
  `SeSampleChain`, `SeSampleChainSizes` and `se_sample_chain_{default_config,res_config,sizes,mem_size,init,res,nr,
  last_res_gain,last_nr_gain,form,synthesize,reset,destroy}`. Board samples that include it must carry the
  per-hop sequence themselves, as the examples now do.
- Python: `PostBeamResidualSuppressor.from_aec_config()`, `FourChannelFrame.aligned_render` (only fed the
  external suppressor; `FrontFrame.aligned_render` stays), the package export of
  `PostBeamResidualSuppressor` (the class stays in `pipeline.py` for the parity driver), and the
  `aec_preset` argument of `4ch_aec_bf_dfn_res/pipeline.py:run_front_end` (strength is `config.aec_preset`).

### Pool sizes (NE10, `SIMD=1`, matched n=5; bytes, as printed by the example binaries of this tree)
| Item | 16 kHz 256/128 | 16 kHz 512/256 | 48 kHz 1024/512 |
|---|---|---|---|
| Mono AEC + NR chain (`aec_nr_pipeline_static`) | 423,728 | 575,056 | 1,354,256 |
| 4-channel front end + own `AecRes` + chain (`4aec_nr_res_static`; `AecRes` alone) | 973,776 (7,904) | 1,519,968 (14,048) | 3,402,064 (29,408) |
| 4-channel unit + chain (`audio_pipeline_4ch_static`; the unit alone, RES included) | 1,766,096 (1,706,784) | 3,090,016 (3,006,560) | 6,527,568 (6,329,856) |
| Mono DFN2 (`mono_aec_dfn_res`; 8 kHz 621,024) | 716,400 | | 1,443,712 |
| 4-channel DFN2 (`4ch_aec_bf_dfn_res`) | 2,058,768 | | 6,617,024 |
| 4-channel Align-ULCNet unit (no RES) + stage | 2,982,176 + 36,400 | | 6,279,872 + 71,728 |

The unit grows by the `AecRes` plus 16 B of control block and the chain block shrinks with the removed
control block (128 B in the mono and core chains); every unit pool, the front-only profile included, grows
by 16 B.

### Migration notes for board code
- Recompile. A persisted unit descriptor from layout 18 (or 19) is refused by `init_ex()`; query a fresh one.
- 4-channel code that kept its own post-beam `AecRes` next to the unit can drop it: set `enable_res` /
  `res_preset` in the unit config, read `frame.g_res_ctx` after `audio_pipeline_4ch_fuse()` and call
  `aec_res_set_preset(audio_pipeline_4ch_res(p), ...)`. Code that drives the core directly keeps its `AecRes`.
- Code that used `se_sample_chain_*`: size and initialise the stages (`aec_get_mem_size()`,
  `anr_get_mem_size()`, `ac_istft_get_mem_size()` plus the gain and spectrum buffers) and chain them as
  `mono_aec_nr_res/main.c` (`mono_form()`) and `4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` (`raw_chain()`) do;
  `anr_set_mode(nr, mode)` takes the denoiser you initialised.


## [Unreleased] — 2026-10-24 — NR minimum window: library default

### Changed
- The pipelines build their denoiser from the NR library preset alone,
  `anr_config_for_mode_grid(sample_rate, fft_size, nr_mode)`. The override of the minimum-tracking
  window `L` (`anr_retime_frames(150, sample_rate, hop)`: 188 / 94 / 141 hops at 16 kHz 256/128,
  16 kHz 512/256, 48 kHz 1024/512, about 1.5 s) is gone; `L` is the library default, 32 frames of
  16 ms (64 / 32 / 48 hops, about 0.512 s; 32 hops at 8 kHz). The `alpha_decay = alpha_g` line went with
  it: the preset already sets both from the same expression. Python mirrors it: `_build_denoiser()`
  no longer sets `L = 94` (`_NR_L_MINIMA_WINDOW`), so its `L` comes from `v3_2_config.yaml`.
- Output is bit-identical. With the scene-reset gate off (`broadband_threshold = 1.0`) the window
  length does not reach the gain: the golden matrix (audio, seam and driver rows) and the
  `pyc_parity` rows are unchanged at `SIMD=1` and `SIMD=0`; a control with the gate on
  (`broadband_threshold = 0.8`) does tell the two windows apart.
- `anr_set_mode()` now accepts the chain's denoiser: `anr_set_mode(se_sample_chain_nr(chain), mode)`
  changes its strength while it runs.
- The tests' oracles (`conv_chain_mono.h`, `conv_chain_4ch.h`, `test_se_common.c`) write the 512 ms
  window out in whole hops; `test_se_common` checks the window per grid as literals, that the chain
  sizes its denoiser like the oracle, that `anr_set_mode()` before the first hop equals a denoiser
  built in that mode on every grid, and that a target one hop longer is refused.

### Removed (example code, no ABI)
- `se_anr_compose_config()`, `se_anr_stage_set_mode()` and the pass-throughs
  `se_anr_stage_{mem_size,init,reset,destroy}()` (`samples/se_anr_stage.h`; call
  `anr_config_for_mode_grid()`, `anr_get_mem_size()`, `anr_init()`, `anr_set_mode()`, `anr_reset()`,
  `anr_destroy()`). `samples/se_anr_stage.h` keeps `se_anr_stage_gain()`, `se_psd_scale()` and
  `SE_PSD_SCALE` / `SE_PSD_SCALE_INV`.
- `se_sample_chain_set_nr_mode()` (`samples/se_sample_chain.h`): use `anr_set_mode()` on
  `se_sample_chain_nr(chain)`; a chain without NR returns `NULL` there and `anr_set_mode()` returns -1.

### Pool sizes (NE10, `SIMD=1`; bytes, after the split entry below -> now)
| Item | 8 kHz 256/128 | 16 kHz 256/128 | 16 kHz 512/256 | 48 kHz 1024/512 |
|---|---|---|---|---|
| `AnrDenoiser` (`anr_get_mem_size()`; `L` in hops) | 60,784 -> 28,784 (94 -> 32) | 109,328 -> 45,344 (188 -> 64) | 120,176 -> 56,432 (94 -> 32) | 335,424 -> 144,576 (141 -> 48) |
| Mono AEC + chain (`aec_nr_pipeline_static --print-mem-size`) | 359,232 -> 327,232 | 521,168 -> 457,184 | 672,768 -> 609,024 | 1,612,864 -> 1,422,016 |
| 4-channel front end + chain (`4aec_nr_res_static`) | | 1,171,200 -> 1,107,216 | 1,719,200 -> 1,655,456 | 3,863,568 -> 3,672,720 |
| 4-channel unit + chain (`audio_pipeline_4ch_static`; chain alone) | | 1,963,504 -> 1,899,520 (131,328 -> 67,344) | 3,289,232 -> 3,225,488 (161,376 -> 97,632) | 6,989,056 -> 6,798,208 (418,096 -> 227,248) |

Layout versions and `build_flags_hash` are unchanged: the denoiser lives in the header-only sample
chain, not in the unit or the core.


## [Unreleased] — 2026-10-24 — Pipeline split: one wrapped unit, separate libraries

### Changed
- Only the multi-channel linear AEC + beamformer is wrapped as one unit: `AudioPipeline4Ch`
  (`4ch_aec_bf_nr_res/audio_pipeline_4ch.{h,c}`), now `process_front()` / `process_beam()` /
  `process_with_activity()` / `process()` returning the beamformed spectrum and weights
  (`AudioPipeline4ChBeam`), plus an independent `audio_pipeline_4ch_fuse()` that fills a `SeSpecFrame`.
  RES, NR, comfort noise, synthesis and the neural post-filters are separate libraries that the
  integrator initialises (each with its own `mem_size` and `init`) and chains. Outputs of the
  example programs are bit-identical to the previous revision at `SIMD=1` and `SIMD=0` (a golden
  matrix of audio, seam and driver outputs, plus the `pyc_parity` seam rows); only pool sizes, titles and
  memory lines changed.
- The gain sequence (RES gain, NR gain, `min`, apply, comfort noise, synthesis) is example code:
  `pipelines/samples/se_sample_chain.h`. The NR recipe moved to `samples/se_anr_stage.h`, the DFN2
  hop driver to `samples/dfn_hop.h`, the Align-ULCNet stage to `samples/ulcnet_nn_stage.{h,c}`.
  The samples have no ABI, no layout version and no descriptor, and are in no archive.
- `se_frame_from_aec()` (`common/se_stage.h`) fills the `SeSpecFrame` of a mono `Aec`; mono has no
  unit at all (`Aec` + `se_frame_from_aec` + the chain).
- `pipelines/mono_*`, `4ch_alignulcnet` and `4ch_aec_bf_dfn_res` mains, the static mains, the raw
  runner, the mcab driver and `pyc_drv.c` are compositions of the above and stay as examples; CLI
  flags, positional arguments, output formats and the raw runner JSON keys are unchanged.
  `pyc_drv` now runs the shipped sample chain instead of the test oracle.
- Python: `FourChannelAecPipeline.fuse()` no longer runs the post-beam suppressor; its
  `context.res_gain` is `None`, and the caller applies `PostBeamResidualSuppressor` (exported from
  `4ch_aec_bf_nr_res/__init__.py`) to `FourChannelFrame.aligned_render`.
  `PostBeamResidualSuppressor.from_aec_config()` builds one from an `AecConfig`.
  `run_front_end` (DFN2 4-channel), `evaluate_recordings.py` and `tools/eval_dfn2_4ch_inputs.py`
  own their suppressor. Output is unchanged. `FourChannelAecPipeline.reset()` no longer resets the
  suppressor: the caller calls `suppressor.reset()` together with `pipeline.reset()`.
- Tests: `test_audio_pipeline*.c` and `test_4aec_dfn_res.c` are replaced by composition-level tests
  against hand-written oracles (`conv_chain_mono.h`, `conv_chain_4ch.h`): `test_mono_chain`,
  `test_mono_dfn`, `test_mono_ulcnet`, `test_4ch_dfn`, `test_4ch_ulcnet`, and
  `samples/tests/test_ulcnet_nn_stage.c`, `test_dfn_hop.c`. The cng-floor witness line lives in
  `mono_aec_nr_res/tests/test_mono_chain.c`.
- Fixed in the mono examples: `--no-nr` no longer dereferences a NULL denoiser under `DUMP_CTX`
  (the NR gain reads as the unity vector), and `static_main.c --print-mem-size` honours
  `--no-nr`, `--no-res` and `--no-cng`.
- A refused hop is recovered by the integrator, not by a library: reset the unit (or `Aec`), the
  chain and any neural stage together (`audio_pipeline_4ch_reset()` / `aec_reset()`,
  `se_sample_chain_reset()` — which also restarts the comfort-noise generator —
  `ulcnet_nn_stage_reset()` / `dfn_hop_reset()`). The DFN stage is still not reset by an AEC delay change.
- `audio_pipeline_4ch_process_front()` (and the calls built on it) leaves no front or beam pending
  after any refusal, so a following `process_beam()` / `audio_pipeline_4ch_fuse()` returns
  `SEQUENCE_ERROR`; `audio_pipeline_4ch_fuse()` no longer resets the unit on `SEQUENCE_ERROR`.
  A caller that skips a hop refused for a non-finite sample keeps all adaptive state.

### Removed (hard break, no shims)
- Mono: `audio_pipeline_{default_config,get_mem_requirements,init,init_ex,create,process,process_spec,synthesize,set_aec_preset,set_nr_mode,reset,destroy,hop_size,n_freqs,sample_rate,get_aec,get_nr,get_mem_breakdown,get_last_timing}`,
  `AudioPipelineConfig` / `AudioPipelineMemReq` / `AudioPipelineMemBreakdown` / `AudioPipelineLastTiming`,
  `mono_aec_nr_res/audio_pipeline.{h,c}`, `libaudio_pipeline.a`.
- `audio_pipeline_ulcnet_*`, `audio_pipeline_4ch_ulcnet_*`, `mono_aec_dfn_res_*`, `four_aec_dfn_res_*`
  and their config / memory-requirement types, headers and sources.
- `FourAecNrResConfig.{aec_preset,nr_mode,enable_res,enable_nr,enable_cng,legacy_amin}` and
  `four_aec_nr_res_stage_configs()`: the lanes are always derived with `AEC_PRESET_BALANCED` and
  compute only the linear residual; the settings belong to the chain (`SeSampleChainConfig`).
- `AudioPipeline4Ch`: the time-domain `output` of `process()` / `process_with_activity()`,
  `audio_pipeline_4ch_{set_aec_preset,post_split_floor,set_nr_mode,nr_count,post_res_count}`.
  The unit's config validation no longer requires `core.enable_post == 1`; with `enable_post = 0`
  `audio_pipeline_4ch_fuse()` returns `INVALID_ARGUMENT`.
- Build: `-DAUDIO_PIPELINE_STAGE_TIMING`, `-DAUDIO_PIPELINE_NO_STDIO` and `-DAUDIO_PIPELINE_NOW_US`
  (`PROFILE=1` now sets `-DAEC_STAGE_TIMING=1` only; the example programs read their own clock).
  `libaudio_pipeline_4ch.a` now holds `4aec_nr_res.o`, `audio_pipeline_4ch.o` and `se_config.o`.
- `common/se_anr_stage.{c,h}` (the recipe lives in `samples/se_anr_stage.h`), `SeRules.with_nr` and
  `SeRequest.nr_mode` (`common/se_config.h`; `se_derive()` no longer looks at the NR mode, and
  `se_config.h` no longer includes `anr_types.h`).
- Python: `FourChannelAecConfig.aec_preset`, `FourChannelAecPipeline.set_aec_preset()`,
  `FourChannelAecPipeline.residual_suppressor_instance_count`.

### Layout versions
| Macro | Old | New |
|---|---|---|
| `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` | 17 | **18** (pool no longer carries RES / NR / synthesis; `build_flags_hash` carve token changed) |
| `FOUR_AEC_NR_RES_LAYOUT_VERSION` | 22 | **23** (the config embedded in the control block shrank) |
| `AUDIO_PIPELINE_LAYOUT_VERSION` (+ descriptor version 2) | 15 | removed |
| `AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION` | 15 | removed |
| `AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION` | 23 | removed |
| `MONO_AEC_DFN_RES_LAYOUT_VERSION` | 5 | removed |
| `FOUR_AEC_DFN_RES_LAYOUT_VERSION` | 6 | removed |

`DFN_RES_STAGE_LAYOUT_VERSION` (3), `DFN_RATE_BRIDGE_LAYOUT_VERSION` (3) and the AINR / AIAEC layouts are
unchanged. The unit's `build_flags_hash` is `0x0ac1063e` -> `0x8aabdea5`; the core's is `0x5812ead9`.
`tests/test_doc_layout_versions.py` no longer lists the removed macros.

### Pool sizes (NE10, `SIMD=1`, matched n=5; bytes, old -> new)
| Item | 16 kHz | 48 kHz |
|---|---|---|
| Mono AEC + NR chain, `--print-mem-size` (16 kHz default 256/128; 48 kHz 1024/512) | 520,672 -> 521,168 | 1,610,832 -> 1,612,864 |
| Mono chain at 16 kHz 512/256; 8 kHz | 671,760 -> 672,768; 358,736 -> 359,232 | |
| 4-channel front end (`FourAecNrRes`) | 1,039,904 -> 1,039,872 | 3,445,504 -> 3,445,472 |
| 4-channel front end + chain (`4aec_nr_res_static`, 16 kHz 512/256 in the second row) | 1,170,576 -> 1,171,200; 1,718,064 -> 1,719,200 | 3,861,408 -> 3,863,568 |
| 4-channel unit + chain (`audio_pipeline_4ch_static`; the unit alone is 1,832,176 / 6,570,960) | 1,962,960 -> 1,963,504 | 6,986,976 -> 6,989,056 |
| Mono DFN2 | 764,096 -> 750,384 | 1,522,048 -> 1,513,536 |
| 4-channel DFN2 | 1,400,048 -> 1,400,416 | 3,762,336 -> 3,764,240 |
| Mono Align-ULCNet (n=5) | 568,128 -> 568,080 | 1,308,336 -> 1,308,288 |
| 4-channel Align-ULCNet (unit + stage) | 3,153,936 -> 3,117,520 + 36,400 | 6,622,128 -> 6,550,384 + 71,728 |

The READMEs, the integration manuals and the HTML pages carry the re-measured numbers; the
8 kHz mono DFN2 host is 652,336 -> 638,624.

### Migration notes for board code
- Recompile. A persisted unit or core descriptor from the previous layout is refused by `init_ex()`
  (layout 17 -> 18, 22 -> 23): query a fresh one.
- Mono: replace the `audio_pipeline_*` calls by `se_derive()` -> `aec_get_mem_size()` /
  `se_sample_chain_mem_size()` -> `aec_init()` / `se_sample_chain_init()`; per hop
  `aec_process_context()` -> `se_frame_from_aec()` -> `se_sample_chain_form()` ->
  `se_sample_chain_synthesize()`; `--aec-only` is `aec_process()`. Runtime strength moves to
  `aec_set_preset()` and `anr_set_mode(se_sample_chain_nr(chain), mode)` (see the NR entry above).
- 4-channel: `audio_pipeline_4ch_process*()` now returns the beam; call `audio_pipeline_4ch_fuse()`
  and the chain for the frequency-domain stages (`aec_res_set_preset(se_sample_chain_res(chain), ...)`
  replaces `audio_pipeline_4ch_set_aec_preset()`). Align-ULCNet code builds the unit with
  `core.enable_post = 0` and feeds `beam.spectrum`, the raw far hop and the unit's
  `delay.changed` / `delay.solid` to `ulcnet_nn_stage_begin_complex()`.
- DFN2: `dfn_hop_*` over the stage or the rate bridge replaces `*_process_begin` / `nn_begin` /
  `nn_end` / `process_end`; P must be formed before `dfn_hop_begin()`.
- The pool of each library is sized and carved separately; the 4-channel unit and the chain no longer
  share one block.
- Everything an application did with a failed hop (`reset` of the whole wrapper) is now the integrator's
  paired reset (see Changed).

### NR configuration
This entry changes no NR numbers. The NR window override it carried over into `samples/se_anr_stage.h`
is removed by the entry above.


## [Unreleased] — 2026-10-23 — Sample driver tidy-up; sources and docs condensed

### Changed
- `mcab_sample_driver`: a NULL `finish` hook counts as success; the raw and static drivers and the 4-channel
  DFN / ULCNet `main.c` use it, with a test for the NULL case.
- `dfn3_process.c` drops duplicate includes. Docs: the quick-start clone URL, archive paths (`print-lib-path`),
  layout-version numbers and the Align-ULCNet far-input description (raw far) match the code; the long design and
  upstream-alignment notes are cut to present-state tables, with history left in git.


## [Unreleased] — 2026-10-23 — Smaller AEC pool; DFN3 pre/post accessor and ERB-swap tests

### Changed
- The AEC pool shrank (`PBFDAF.scr_e2` and the filters' `is_static` fields were removed; the FFT handle
  stages its preserving transforms in one block), and the denoiser's start-up rows share the minimum
  ring, so every pipeline that composes them carries a new carve layout version, as the version
  comments require: `AUDIO_PIPELINE_LAYOUT_VERSION` 15, `AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION` 15,
  `MONO_AEC_DFN_RES_LAYOUT_VERSION` 5, `FOUR_AEC_NR_RES_LAYOUT_VERSION` 22,
  `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 17, `AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION` 23,
  `FOUR_AEC_DFN_RES_LAYOUT_VERSION` 6 and `DFN_RATE_BRIDGE_LAYOUT_VERSION` 3. A persisted descriptor
  from the previous layout is refused by `init_ex()`; query a fresh one. The DFN stage and the
  AINR/AIAEC pre/post and model layouts are unchanged (their pools borrow the FFT handle).
- Pool totals in the memory tables and CLI examples are re-measured (for example mono 16 kHz/256
  520,672 B, 4-channel front end 16 kHz/256 1,039,904 B).

### Added
- `AINR/tests/test_dfn3_prepost_c.py`: ctypes tests of `dfn3_prepost_hop_size/num_bins/io_mode` and of
  the between-hops `dfn3_prepost_set_erb_matrices()` contract (a swap before the first hop equals
  constructing with the new pair, a mid-stream swap keeps the rows already in the window, NULL
  arguments and an open frame are refused and change nothing).

### Removed
- `dfn_rate_bridge_set_atten_lim()`, a pure forwarder; call `dfn_res_stage_set_atten_lim()` on
  `dfn_rate_bridge_stage()`, or set `atten_lim_db` at creation.

### Source comments condensed
- Release comments in the pipeline sources are cut to contracts and one-line rationale; the design
  decisions that lived only in comments are recorded here.
- Shared-delay duty cycle (`4aec_nr_res.c`): the period floor is 1 hop (a floor of 2 capped the
  `delay_est_period_s = 0.0` default at 50 % engagement); the watchdog leak is lib/aec's 0.1 dB per
  second and scales linearly with the hop period (0.001 dB per 10 ms hop), not as a retention
  power law. The opt-in throttle saved about 4x of the shared delay stage (17.0 -> 4.3 us/hop
  measured) at the price of a possibly different settled lag; with no lane clearing the ERLE arming
  threshold, re-acquisition costs up to about 2.5x on 16 kHz/256 (bounded at K).
- Shared-delay change admission uses lib/aec's Path-B numbers (movement above 32 native samples,
  offered again within 16, candidate life 3 hops). DelayAec publishes on a 16-downsampled grid
  (64 native samples at 16 kHz, 192 at 48 kHz), so the 32-sample term is a floor under a finer
  source. The TTL is not reachable from a stream (DelayAec re-offers every hop), hence the
  non-static state machine.
- An alignment change realigns the four lanes (`aec_apply_external_realign()`: warm tap transfer or
  a filter-only reset) instead of resetting them; a full reset costs one near-zero hop plus dozens
  of hops of re-exposed echo. A mixed warm/soft sweep makes the lanes' far spectra diverge (seen on a
  real 48 kHz 4-microphone fixture), so that hop skips lane-0 far-FFT sharing and the agreement
  predicate; on all other hops four far-FFT counter reads replace 4*n_freqs compares.
- Identity reprime after an alignment boundary is 1 frame in both ULCNet pipelines: both branches are
  framed from the current hop and each 50 %-overlap frame spans two hops, so only the boundary hop's
  frame straddles. The count was derived from a marker run containing no boundary.
- `delay_backward_quarantine_enabled` stays off in the 4-channel ULCNet defaults: the guard only
  delays a mis-lock by its window, so enabling it is a policy decision pending a spot check with the
  deployed checkpoint.
- DFN stage: the comfort-noise fill stays off by default because additive noise perturbs P's phase in
  the RES-cut bins, where the deep filter's estimate/apply consistency (fitted on E, applied on P)
  no longer holds; quality of the dual-input topology against a mask-only compose is to be settled
  with the Python evaluators once a checkpoint of the current contract exists.
- `post_gain`: the angle-to-bin mapping rounds in radians (SRP's rounds through degrees and differs
  on 208 of 2.4M sampled angles); the frequency smoother splits edge/interior loops (247.2 -> 34.6
  ns/frame at F=257; a running accumulator is rejected because it reassociates the sum, hand NEON
  38.0 ns); the target loop uses `&` instead of `&&` (103.0 -> 18.7 ns/frame); hoisting the time
  smoother's coefficients took 96.1 -> 26.7 ns/frame; a two-pass dilation was slower (132.5 -> 201.1);
  `sk_asym_ema_f32` is not used because it breaks ties on the other side (17.0 % of values differ).
- `fix_gain`/`nr_gain`: a `global_gain` of exactly 0 means unset (1.0) so existing tuning files keep
  their meaning; `fix_gain` clips in a separate pass through the shared kernels.
- Stage timing stays off by default: nine clock reads per 4-channel hop plus five per lane (29 in
  all), four per mono hop; the profile build sets both halves with `PROFILE=1`.

## [Unreleased] — 2026-10-23 — Frozen-digest regression checks are no longer shipped

### Removed
- The recorded `kExpect*` digests and libm probes of the AINR (DFN2/GTCRN) and AIAEC
  (Align-ULCNet/DeepVQE-S) pre/post chain tests, and the unreferenced reference-trace manifests
  `AINR/tests/state_layout_traces.sha256` and `AIAEC/tests/state_window_reference.sha256`.
  `AINR/tests/test_nn_chains.c` and `AIAEC/tests/test_prepost_chains.{c,py}` keep the whole-chain
  drive (non-finite heads and state, skipped and unwritten frames); `make -C AINR test-simd` and the
  pytest module require the NEON and forced-scalar builds to print identical digests, and each chain
  must reproduce its own digest within a process.

## [Unreleased] — 2026-10-22 — ARM32 memory descriptor compatibility

### Changed
- All 13 pipeline and neural pre/post memory-descriptor producers include the
  shared ILP32 ABI token on 32-bit targets; AArch64 hashes and the fixed 32-byte
  descriptor ABI stay unchanged. Target selection follows compiler macros.
- Initialize the DFN2/DFN3 pool-walk output pointer explicitly for GCC's strict
  ARM32 build. C11 test calls use explicit const-array views and portable
  declarations; algorithms, presets, state geometry and latency are unchanged.
- Add `tests/test_target_abi.c` to exchange real descriptors between 32/64-bit
  builds and verify rejection without modifying caller memory, independently
  of differing pool byte budgets.
- Static mono/4-channel host examples request their descriptor's alignment
  explicitly; ARM32's ordinary malloc alignment is insufficient for a 16-byte
  caller pool. RNNoise C test framing fixtures follow the same rule.
- Four-lane diagnostic sums and realign totals saturate at `LONG_MAX`.
  Saturated/rewound per-lane counts cannot establish far-spectrum provenance;
  the existing full comparison is used. Borrow/history calls and normal-range
  values are unchanged. Boundary regressions cover both long widths.

## [Unreleased] — 2026-10-21 — Build isolation with uncommitted new sources

### Fixed
- Build-isolation assertions compare the pipeline attestation's untracked flag with independently enumerated current files. New files can remain untracked during review; publish refusal policies and explicit escape hatches are unchanged.
- Added scratch-only fixtures for whitelist-only (`0`) and ordinary new-source (`1`) states. No staging or commits occur in the real checkout.

## [Unreleased] — 2026-10-21 — Parity driver include roots

### Fixed
- The standalone pipeline parity driver includes the common root for `utility/` headers, matching the production build after header relocation. Numeric drivers, frozen tolerances and expected-red registries are unchanged.

## [Unreleased] — 2026-10-21 — Common lifecycle for all four 4-channel samples

### Changed
- All four 4-channel mains now share actual CLI scanning, delay-profile
  validation and the setup/read/process/finite-check/consume/finish/cleanup
  driver. Typed adapters retain each pipeline's config, model and pool APIs.
- Preserve per-target CLI conversion/default/error policies, complete-hop
  behavior, VAD handling, finite checks and destruction-before-success order.
  The driver links into sample executables only; algorithm archives, pool
  layouts and hot-path allocation behavior are unchanged.
- Add a permanent control-flow/CLI regression gate to the 4-channel Makefile.

### Validation
- Scalar/SIMD builds for both ULCNet deployment grids pass. Across 328 paired
  before/after cases, exits, stdout/stderr, raw output files and 36 observed
  real-API sample output traces are byte-identical, including all three raw
  signal grids, explicit/automatic VAD and incomplete/empty/error inputs.
- The shared runner tests setup/read/process/consume failures, early help,
  finite-output policies and cleanup before model-sample success reporting.
  No dataset evaluation or performance conclusion is part of this change.

## [Unreleased] — 2026-10-21 — Numeric AEC presets in conventional samples

### Changed
- The mono heap/static WAV CLIs and `4aec_nr_res_static` accept AEC preset
  digits `0/1/2/3` as aliases for `mild/balanced/aggressive/no_floor`.
  Successful output labels include `no_floor`; usage text documents the mapping.
- Existing named mild/balanced/aggressive behavior, default balanced preset and
  the historical unknown-string fallback to balanced remain unchanged. This is
  CLI exposure only: no DSP equations, defaults, configs, pool or ABI changed.

## [Unreleased] — 2026-10-21 — Shared periodic sqrt-Hann window

### Changed
- Pipeline test fixtures and the RNNoise host table generator use audiolib's
  standalone `audio_window.h`. All generated RNNoise table values stay byte-identical;
  the header comment now gives the complete standalone regeneration command;
  its independent table recomputation and audiolib's frozen STFT references are
  retained. RNNoise's runtime FFT and framing are unchanged by this extraction.
- Updated the C/Python function map to locate `ac_window_build` in
  `audiolib/common/src/audio_window.c`.

## [Unreleased] · 2026-10-20/21 · Part A：`audio_common` → `audiolib` 改名＋third_party 合併

純識別字/路徑改名＋目錄合併（跨 repo C/Python 整理計畫的一部分），不含任何行為改動；
binary-golden／pyc_parity 驗證全過（細節見 commit 說明／執行報告）。

### 變更 (Changed)

- 同層 `audio_common` 檢出改名為 `audiolib`：`git mv` 原內容（`include/`、`src/`、`lib/`、`test/`、
  `scripts/`、`docs/`、`Makefile`、`README.md`、`CHANGELOG.md`、`.gitignore`）進新的 `common/` 子目錄
  （保留 git 歷史），`audio_common`/`AC_DIR` 的 Wave-S kiss 移除等未提交改動隨 `git mv` 一併帶過去。
- `pipelines/third_party/{doa,bf,VAD}` 與 `third_party/utility/{complex,spatial_simd}.*` 合併進
  `audiolib`（`doa/`、`bf/`、`VAD/`、新的 `common/utility/`）；`complex.{c,h}` 改名
  `spatial_complex.{c,h}`（guard 改名、符號已是 `spatial_complex_*` 不變）。`third_party/gain/` 留在原地
  （不進 `pipelines/` 的連結範圍，不在此次合併內）；`third_party/tests/test_spatial_third_party.c`
  依此拆成兩半：PHAT/BF/SRP/VAD 移到 `audiolib/tests/`，gain 五項測試（連同各自一份
  `check_pool_contract`）移到新的 `pipelines/third_party/gain/tests/test_gain.c`（目前手動建置驗證過，
  尚無自己的 Makefile target — 待裁定）。`third_party/Makefile` 隨之退場；`third_party/README.md` 改寫為
  只涵蓋留下的 `gain/`。
- 本檔（`pipelines/Makefile`）、`4ch_aec_bf_nr_res/Makefile`、`AEC/c_impl/Makefile`、
  `NR/c_impl/Makefile`（`lib/aec`、`lib/nr` 兩個 submodule 工作樹各自同步鏡像，不經由 submodule
  commit/push）、`AINR/Makefile`、`AIAEC/Makefile` 的 `AC_DIR`/`AC_LIB`/`AC_FWD` 改名為
  `COMMON_DIR`/`COMMON_LIB`/`COMMON_FWD`；`4ch_aec_bf_nr_res/Makefile` 的 `THIRD_PARTY_DIR` 改名
  `AUDIOLIB_DIR`。所有硬寫的 `../../audio_common` 類相對路徑改成多一層的 `../../audiolib/common`。
- `audio_common.pin` 改名 `audiolib.pin`，`AUDIO_COMMON_COMMIT` 改名 `AUDIOLIB_COMMIT`；`check-pins`
  改查整個 `audiolib` 檢出的 HEAD（新 `AUDIOLIB_DIR` 變數），不只 `common/` 子目錄；pin 的 commit 值
  不變（改名是未提交的工作樹搬移，不是新 commit）。
- `docs/c_python_function_map.md`：`pipelines/third_party/{VAD,doa,utility}` 的路徑列改成
  `audiolib/{VAD,doa,common/utility}`；`tests/test_c_python_map.py` 的路徑解析一併擴充（原本只認
  `audiolib/common/` 前綴，現在整個 `audiolib/` 檢出都能解析）。**留著不動、本輪範圍外**：表內仍有
  Part E（GSC→BF 改名）遺留的 `GSC`/`gsc_*`/`audio_pipeline_4ch_gsc_*` 等舊識別字與
  `pipelines/third_party/GSC/gsc.c` 路徑尚未同步（`test_c_python_map.py` 這兩項測試在本輪之前就已經會
  失敗，與本次 audiolib 改名無關，留給之後專門的 Part E 收尾處理）。
- 歸檔檔名不變（`libaudio_common.a` 仍是這個名字）——只有產生它的目錄路徑搬了家。

## [Unreleased] · 2026-10-20 · Part E：`third_party/GSC` 與相關 4ch 識別字全面改名為 `bf`/`BF`

純識別字/路徑改名（跨 repo C/Python 整理計畫的一部分），不含任何行為改動；`pipelines/` only。

### 變更 (Changed)

- 目錄/檔案（以 `git mv` 追蹤為搬移）：`pipelines/third_party/GSC/` → `pipelines/third_party/bf/`；
  `gsc.c`→`bf.c`、`gsc.h`→`bf.h`、`gsc_test_hooks.h`→`bf_test_hooks.h`。
- C 識別字（函式/型別/巨集，大小寫精確）：`gsc_create`/`gsc_init`/`gsc_destroy`/`gsc_get_mem_size`/
  `gsc_reset`/`gsc_reset_bin`/`gsc_process`/`gsc_process_with_weights`(`_impl`)/`gsc_get_bin_resets`/
  `gsc_get_doa_used`/`gsc_get_adaptive`/`gsc_effective_adapt_interval`/`gsc_wa_leak_factor`/
  `gsc_p_diag_floor`/`gsc_p_diag_ceil`/`gsc_rls_update_bin_scalar`/`gsc_rls_update_bins`/
  `gsc_rls_update_four_neon`/`gsc_conj` → 同動詞 `bf_` 前綴；型別 `GSC`→`BF`、`GSC_Config`→`BF_Config`；
  巨集 `GSC_H`/`GSC_WA_LEAK`/`GSC_P_DIAG_FLOOR`/`GSC_P_DIAG_CEIL`/`GSC_USE_PROJECTION_BLOCKING`/
  `GSC_TESTING`/`GSC_TEST_HOOKS_H` → `BF_` 對應；`spatial_simd` 的 `spatial_gsc_projection*`/
  `spatial_gsc_effective_weights*` → `spatial_bf_*`。
- 4ch pipeline 本地識別字（非公開 API，但仍是 C 程式碼，依既有命名清理慣例一併改）：Config 欄位
  `gsc_fixed_mode`/`gsc_fixed_doa_rad`/`gsc_fixed_align_notebook`/`gsc_adaptive`/`gsc_mu`/`gsc_lambda`/
  `gsc_adapt_interval`/`gsc_enable` → `bf_` 前綴；本地變數/欄位 `gsc_cfg`/`gsc_spectrum`/`gsc_weights`/
  `gsc_out`/`gsc_region`/`gsc_steering`/`gsc_bytes`/`saw_gsc_adaptive`/`fixed_gsc` → `bf_` 前綴；
  `audio_pipeline_4ch` 的 getter `audio_pipeline_4ch_gsc_*` → `audio_pipeline_4ch_bf_*`；測試識別字
  `test_gsc_*`/`run_gsc_*`/`_c_srp_gsc_nr_res` → `test_bf_*`/`run_bf_*`/`_c_srp_bf_nr_res`。
- Makefile/build（`third_party/Makefile`、`4ch_aec_bf_nr_res/Makefile`、`pipelines/Makefile` 三處）：
  `GSC_LIB`→`BF_LIB`、`libgsc.a`→`libbf.a`、`print-gsc-lib-path`→`print-bf-lib-path`、
  `audit-no-gsc-test-symbols`→`audit-no-bf-test-symbols`、`-DGSC_TESTING`→`-DBF_TESTING`、
  `-IGSC`→`-Ibf`。
- Python 端跟隨改名（功能性，非僅風格）：`evaluate_external_recordings.py`／`test_external_evaluator.py`
  的 JSON 契約鍵 `gsc_sample_rate`/`gsc_frame_size`/`gsc_hop_size`/`gsc_fft_size`/`gsc_adaptive_frames`
  → `bf_` 對應（這些鍵直接鏡射 C 端 `printf` 出的 JSON；C 端改名後若 Python 端不跟著改，
  `validate()` 契約檢查會全部失敗，這不是風格選擇而是維持契約可用的必要改動）；`pipeline.py`
  （`4ch_aec_bf_nr_res`、`4ch_aec_bf_dfn_res` 兩份）、`evaluate_recordings.py`、
  `tools/eval_dfn2_4ch_inputs.py` 的描述性 docstring／註解同步改字，避免指向已不存在的符號。
- 註解/說明文字：C 檔案中「GSC beamformer」一類描述性文字改寫為「beamformer (bf)」或視語境改寫
  整句，不做盲目字串替換（例如避免產生「the bf GSC」這類無意義文字；原本就是 `BF/GSC` 複合占位用語
  的句子，改寫為單一的「beamformer (bf)」）；`.md` 檔案視情況同步（非強制，但修掉會指向已不存在
  檔案/符號的失效引用，例如 `third_party/README.md` 的 `` `GSC/gsc.h` `` → `` `bf/bf.h` ``、
  `` `libgsc.a` `` → `` `libbf.a` ``）。

### 範圍確認 (Scope)

- 只動 `pipelines/` 底下的檔案；不動 `docs/*.md`、`docs/html/*.html`（這些檔案仍含歷史性 `GSC`
  字串引用，不在這次「`pipelines/` only」範圍內，已列在複核報告供之後的文件同步批次處理，故意
  延後、不是遺漏）。

### 證據 (Evidence)

- `grep -rli "gsc" pipelines/ --include=*.c --include=*.h`：改動後回零筆（C-code-only 政策要求的
  確認指令）。
- `third_party` 測試套件（SIMD=0／SIMD=1）：改動前後各 **10/10 PASS**，逐行訊息相同
  （`All third-party spatial tests passed`）。
- 4ch 測試套件（`4ch_aec_bf_nr_res/` 的 `make test`，涵蓋 third-party／`test_4aec_nr_res`／
  `test_4aec_dfn_res`／`test_audio_pipeline_4ch`／`test_audio_pipeline_4ch_ulcnet`／
  `4ch_alignulcnet`／`4ch_aec_bf_dfn_res`）：改動前後各 SIMD=0、SIMD=1 皆 **582/582 PASS**、無
  `FAIL`。
- 行為不變（binary-golden）：改動前後各自重新建置的 `audio_pipeline_4ch_raw`，對兩筆真實錄音
  （`datasets/aec_take_turn`、`datasets/aec_together`）跑 `evaluate_external_recordings.py`：
  輸出 wav 的 **sha256 逐位元相同**；JSON 量化指標（`finite`／`rms`／delay／cohort
  `attenuation_db` 等）逐值相同；raw C `summary` dict 除鍵名 `gsc_*`→`bf_*` 外其餘完全相同。
  改動後的 `validate()` 契約檢查（使用新鍵名）對改動後的二進位輸出 **PASS**。
- `audit-no-bf-test-symbols` 閂門驗證「會失敗」（非恆真閂）：對真正的 `libbf.a` 跑 **PASS**（0 個
  `bf_test_` 符號洩漏到可部署 archive）；人工把 `bf_test.o` 注入同一份 archive 副本後重新
  `nm`/`grep`，閂門正確偵測到並回報 **FAIL**。
- `pytest pipelines/4ch_aec_bf_nr_res/tests/test_external_evaluator.py`：**4 passed**。

### 代價與取捨 (Cost / tradeoffs)

- 無已知代價：純識別字/路徑改名，未觸及任何演算法邏輯、常數或控制流；第三方 GSC/VAD/doa/gain 的
  實際數學完全未動（僅改函式/型別/巨集名稱與其內部字面字串）。
- `pipelines/third_party/GSC/` 目錄已不存在；`docs/` 底下仍有歷史引用（見下）留待之後處理。

### 已知限制 (Known limitation)

- `docs/*.md`、`docs/html/*.html` 底下仍有對 `GSC`／`libgsc.a`／`third_party/GSC` 的歷史引用
  （不在這次「`pipelines/` only」範圍內），已在複核報告中列出，留給之後的文件同步批次處理。
- `third_party/Makefile` 是這次唯一完全手動改字（未經腳本）的檔案，因為它在這次工作開始前就已經
  被另一條並行工作串改動過（`BACKEND=kiss` 移除）；人工改字時已逐行核對避免衝突，驗證方式見上述
  「行為不變」證據（third_party 測試套件 10/10 PASS）。

## [Unreleased] · 2026-10-20 · S-30：4ch NO_FLOOR 的 construction-time 缺口補齊（`se_derive()`/`preset_valid()`，opus review C3）

### 變更 (Changed)

- `common/se_config.c`：`preset_valid()` 白名單加入 `AEC_PRESET_NO_FLOOR`（原本只列
  mild/balanced/aggressive，是 S-29 落地 runtime 半邊時就已標記、刻意延後的缺口——`git blame`
  確認這個函式自建檔以來沒再改過，純粹是 NO_FLOOR 加入時沒回頭更新，不是有意的限制）。檔內只有
  `se_derive()` 這一處呼叫 `preset_valid()`，沒有其他 keyed-by-preset 的 switch/array 要同步改。
- `tests/pyc_parity/pyc_drv.c`：拿掉 `run_chain()` 的「BALANCED 建構 + `aec_res_set_preset()`
  runtime retarget」workaround，`aec=no_floor` 現在直接走 `cfg.aec_preset = a->aec` 建構時路徑；
  同步更新檔頭與行內註解、移除不再使用的 `aec_res_set_preset()` include 註解。
- `tests/pyc_parity/pyc_parity.py`：`PRESETS_B` 上方與 `sec_b()` 內 no_floor 分支的註解改成
  「construction-time 直接支援」，不再描述已不存在的 retarget workaround。
- `tests/test_se_common.c`：`test_derive_rejects()` 原本的 `r.aec_preset = (AecPreset)3` 斷言
  `se_derive() == -1`（值 3 就是 `AEC_PRESET_NO_FLOOR`，修完白名單後這個斷言方向是錯的）改成
  斷言 NO_FLOOR 在建構時被接受、且 `min_gain_split_floor_enabled == 0`；真正的「超出範圍」案例
  換成 `(AecPreset)4`。`test_derive_fields()` 加一段：NO_FLOOR 建構出的 `AecConfig` 三個欄位
  （`min_gain_floor_far_active_db`/`min_gain_floor_dt_db`/`min_gain_split_floor_enabled`）與
  `aec_config_from_preset(NO_FLOOR)` 直接呼叫逐欄位相同。

### 證據 (Evidence)

- **can-fail，單元層（`preset_valid()` 修法本身）**：在 `SE_verify_20261019/scratch/
  C3_nofloor_construct/` 下的整樹拷貝裡把 `preset_valid()` 還原成修法前的三項白名單、保留新測試，
  `make SIMD=0 test_se_common` 建置後執行：4 個新斷言全部 `FAIL`（`mono derive no_floor`、欄位
  相符、`preset NO_FLOOR accepted at construction`、`NO_FLOOR construction disables the split
  floor gate`）；還原回真樹後 `./bin/.../test_se_common` 重跑 → `ALL PASS`。
- **can-fail，construction path 本身（`pyc_drv_4ch` → `conv_create()` → `se_derive()`）**：另一份
  整樹拷貝（含 `../../audio_common` 的 symlink，`PYC_WORK` 指到 `.../pyc_work_red/`），同樣還原
  `preset_valid()`、但用**修完的** `pyc_drv.c`（已拿掉 workaround、走 `cfg.aec_preset = a->aec`
  直接建構）執行 `pyc_parity_pipelines.sh --layers 1 --only b.res4.16k256.no_floor`：
  `pyc_drv_4ch` 印出 `conv_create failed`、exit code 1，Python 側 `cx.run()` 拋
  `RuntimeError('driver failed (...): 1\nconv_create failed\n')`，該 scenario 判定
  `NOT_RUN`，腳本整體 `RESULT: INCOMPLETE`、`exit=3`。還原回真樹（`preset_valid()` 含
  NO_FLOOR）後同一個 scenario 重跑：`PASS`、`max_abs=6.5e-06`、`RESULT: OK`、`exit=0`。兩份
  scratch 樹驗證完都已刪除。
- **pipelines 自測**：真樹上 `make SIMD=0 test_se_common` 與 `make SIMD=1 test_se_common` 皆
  `ALL PASS`（`exit=0`）。
- **construction-time 與舊 retarget workaround 行為等價（byte-identical）**：用
  `tests/pyc_parity_pipelines.sh --layers 1 --only b.res4.16k256.no_floor`（`PYC_WORK` 指到
  `SE_verify_20261019/scratch/C3_nofloor_construct/pyc_work_before/`）在改 `pyc_drv.c` **之前**
  先跑一次，存一份 `chain_16k256_aec_no_floor.bin`（當時仍是 BALANCED 建構 + runtime retarget
  的舊路徑，`se_config.c` 的白名單修法對這條路徑沒有影響）；改完 `pyc_drv.c`（拿掉 workaround、
  直接用 NO_FLOOR 建構）後在同一個 `PYC_WORK` 重跑同一個 scenario，兩份 `.bin` 用 `cmp` 比對
  **逐位元相同**（sha256 兩邊都是 `158dbfe0...4cbd17`）：新的 construction-time 路徑與舊的
  runtime-retarget workaround 在同一筆真實錄音輸入上輸出完全一致。
- **三個 grid 的 `b.res4.*.no_floor` 數字與 S-29 落地時記錄的完全一致**（6.50e-06 / 4.47e-06 /
  3.93e-06，皆 ≤ 2e-5 容差），`--layers 1 --only b.res4` 全 PASS（13 列）。vacuity guard
  （`c_gres_vs_balanced_max_abs`：直接建構的 C no_floor 輸出與 C balanced 輸出差多少）在新的
  construction-time 路徑下量到 0.9747 / 0.9622 / 0.9799——跟 S-29 記錄的數字一致，證明 gate
  在「直接建構」下一樣真的生效，不是退化成跟 balanced 沒差的建構。
- **`pyc_parity_pipelines.sh --layers 1,2`（全量）**：84 PASS、1 SKIP、1 INFO，與改動前的基準
  （S-29 記錄的 84/1/1）完全一致，逐列無退步，`RESULT: OK`。
- **4ch C 測試套件**：`4ch_aec_bf_nr_res/` 下 `make SIMD=0 test` 與 `make SIMD=1 test` 均
  `exit=0`，`third_party`/`4aec_nr_res`/`audio_pipeline_4ch`/`audio_pipeline_4ch_ulcnet` 四個
  套件都印出「All ... tests passed」，無 `FAIL`；`pipelines/` 上層 `make SIMD=0/1 test`
  （mono 系列）亦均 `exit=0`、無 `FAIL`。
- **4ch Python 測試**（未改動，確認不受影響）：`pytest
  pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py` 49 passed。

### 代價與取捨 (Cost / tradeoffs)

- 無已知代價：MILD/BALANCED/AGGRESSIVE 既有三個 preset 的 construction-time 行為完全不變
  （`preset_valid()` 只新增一個 `||` 分支）；S-29 遺留的 `pyc_drv.c` 技術債已清償，
  `tests/pyc_parity/pyc_drv.c` 不再需要為 NO_FLOOR 維護一條平行的 retarget 路徑。
- `se_derive()` 是 mono（`mono_aec_nr_res/audio_pipeline.c`）、ulcnet
  （`mono_alignulcnet/audio_pipeline_ulcnet.c`）、4ch 三條應用共用的同一個函式，這次修法連帶把
  construction-time NO_FLOOR 對 mono／ulcnet 也打開了，不只 4ch——頂層 `pipelines/` 的
  `make SIMD=0/1 test`（含 mono／ulcnet 套件）已跑過確認無迴歸，但本輪沒有另外替 mono／ulcnet
  加 NO_FLOOR 專屬斷言，範圍就 opus review C3 點名的 4ch 缺口。
- 觀察到但不在本輪範圍內的後續缺口：`4ch_aec_bf_nr_res/4aec_nr_res_static.c`、
  `mono_aec_nr_res/main.c`、`mono_aec_nr_res/static_main.c` 的命令列字串解析（`"mild"`/
  `"balanced"`/`"aggressive"`）都沒有 `"no_floor"` 分支，所以這幾個執行檔的 CLI 仍然碰不到
  NO_FLOOR（會落到預設的 BALANCED）；留給後續輪次，這次沒動。

## [Unreleased] · 2026-10-20 · S-29：4ch NO_FLOOR 的 runtime 半邊補齊（`PostBeamResidualSuppressor.set_preset()`）

### 變更 (Changed)

- `4ch_aec_bf_nr_res/pipeline.py`：`PostBeamResidualSuppressor.set_preset()` 新增第三個欄位的
  retarget——`min_gain_split_floor_enabled`（經 `SuppressionGain.set_split_floor_enabled()`，不帶
  ramp，跟 DT 地板一樣立即生效）。之前這個 setter 只動兩個 dB 欄位（far-active／DT 地板），把一條
  已經在跑的 4ch suppressor 切到 `AEC_PRESET_NO_FLOOR` 不會真的關掉地板機制——gate 停在建構時的值
  （預設 `True`，等於 BALANCED），只有建構時傳 `aec_preset=NO_FLOOR` 才有效（這一半是先前 S-26+
  NO_FLOOR construction 那輪已經落地的）。C 側（`aec_res_set_preset()`）已經是三欄位一起 retarget；
  這次把 Python 的 runtime 半邊補齊，兩邊重新一致。
- `audio_pipeline_4ch.h`：`audio_pipeline_4ch_set_aec_preset()` 的文件註解改成三欄位（原文只提兩個
  gain 地板），加一句 NO_FLOOR 可在 runtime 切換（gate 跟兩個 dB 欄位同一次呼叫一起 retarget，沒有
  中間態）。
- `README.md`：**Presets** 清單加 `AEC_PRESET_NO_FLOOR`；preset 強度表從「兩欄位」改成「三欄位」
  （新增 `min_gain_split_floor_enabled` 一列）；Runtime strength control 段落加一段說明 NO_FLOOR
  可在運轉中的 instance 上切換，mono／4ch、Python／C 四邊都支援。
- `4ch_aec_bf_nr_res/tests/test_pipeline.py`：新增 4 個測試——
  `test_preset_retarget_moves_split_floor_enabled_too`、
  `test_preset_retarget_split_floor_enabled_survives_reset`、
  `test_construction_time_no_floor_matches_runtime_retarget`、
  `test_runtime_no_floor_switch_changes_suppressor_gain_output`（最後一個直接比較
  `PostBeamResidualSuppressor.process()` 在一個地板真的會生效的場景——converged、far-active latch
  一打就中、r2 遠大於 near_spec——上，BALANCED 跟「runtime 切到 NO_FLOOR」兩個 instance 算出來的
  `res_gain` 確實不同，不只是內部旗標翻轉）。順便修正兩個失效的舊註解：`__init__`／
  `FourChannelAecConfig.aec_preset` 原本寫「NO_FLOOR 沒有 runtime setter」，`test_pipeline.py` 一條
  舊測試的 docstring 也這樣寫，三處都已經過期（construction-time 半邊落地時就已經過期，不是這次新
  造成的，只是一直沒人回頭改）。
- `tests/pyc_parity/pyc_drv.c` / `pyc_parity.py`：新增 `b.res4.<grid>.no_floor` parity scenario
  （3 個，一個 grid 一個）。C 側 `se_derive()` 的 `preset_valid()`（`pipelines/common/se_config.c`）
  還沒把 `AEC_PRESET_NO_FLOOR` 列入白名單（獨立、已另外標記的缺口，這次不修），所以 `pyc_drv.c`
  的 `run_chain()` 用 BALANCED 建構、再呼叫 `aec_res_set_preset(c->res, AEC_PRESET_NO_FLOOR, 0)`
  retarget——跟 Python 側 `py_res()` 本來就有的「balanced 建構 + `set_preset()`」配對完全對應。這
  三列**沒有**動 `PRESETS`（影響 `l2.res4.*.replay` 等其他 scenario 的共用清單），只加進 `sec_b`
  自己用的 `PRESETS_B = PRESETS + ['no_floor']`。

### 證據 (Evidence)

- **pytest**：`PYTHONPATH=Audio_ALG python3 -m pytest
  Audio_ALG/pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py -q` → **49 passed**（45 條既有 + 4
  條新增）。can-fail：把 `set_preset()` 裡新加的
  `self._gain.set_split_floor_enabled(enabled)` / `self._split_floor_enabled = enabled` 兩行直接在
  真實樹上註解掉、跑完測試馬上還原，4 個新測試全部 RED（
  `test_construction_time_no_floor_matches_runtime_retarget` 斷言 `False == True`；
  `test_runtime_no_floor_switch_changes_suppressor_gain_output` 斷言 `_gate(pipeline2) is False` 失敗）；
  還原後重跑 49 passed。
- **pyc_parity**（`pyc_parity_pipelines.sh --layers 1,2`，`PYC_WORK` 指到
  `SE_verify_20261019/scratch/nofloor-4ch-runtime-python/`）：**84 PASS、1 SKIP、1 INFO**（改動前
  跑一次基準是 81 PASS／1 SKIP／1 INFO；新增的 3 列剛好補上差額，沒有任何既有列的狀態改變）。三個
  新列的實測數字：

  | scenario | Python vs C G_res max_abs | tolerance | C no_floor vs C balanced max_abs（gate 有沒有真的動） | double-talk hops |
  |---|---|---|---|---|
  | `b.res4.16k256.no_floor` | 6.50e-06 | ≤2e-05 | 0.9747 | 883/1250 |
  | `b.res4.16k512.no_floor` | 4.47e-06 | ≤2e-05 | 0.9622 | 436/625 |
  | `b.res4.48k1024.no_floor` | 3.93e-06 | ≤2e-05 | 0.9799 | 655/937 |

  「C no_floor vs C balanced」那欄是這次新加的 vacuity guard：三列都遠大於 0，證明 NO_FLOOR 的
  runtime retarget真的把 C 的輸出推離了 balanced（不是建構出一個跟 balanced 沒差的 instance），
  guard 本身若量到 0 會把該列判成 `NOT_RUN` 而不是虛假的 PASS。
- **既有 83 列（balanced／mild／aggressive／NR／mono／realign／Layer 2 全部）完全未受影響**：改動
  前後兩次 `--layers 1,2` 全跑的 `results.json` 逐列比對（`status`／`gate`／`metrics`／
  `tolerance`），83 列 diff=0；9 個 `chain_<grid>_aec_<preset>.bin`（balanced/mild/aggressive ×
  3 grid）seam 檔 `cmp` 逐位元相同。
- **can-fail，pyc_parity 側**：同一個 Python mutation（註解掉 gate retarget 兩行）下用
  `--layers 1 --only b.res4 --reuse-build` 重跑：3 個 `no_floor` 列全部 FAIL（max_abs 0.975／
  0.962／0.98，等於「Python 停在 balanced，C 已經是 no_floor」兩邊的真實差距），`balanced`／
  `mild`／`aggressive` 9 列不受影響仍 PASS，整個呼叫 `exit=1`。還原後重跑：12 列全 PASS，`exit=0`。
- **C 側 4ch 測試套件（`4ch_aec_bf_nr_res/`，這次只動了 `.h` 裡的文件註解，純驗證無迴歸）**：
  `make SIMD=0 test` 與 `make SIMD=1 test` 皆 `exit=0`；兩次各自的
  `test_4aec_nr_res`／`test_audio_pipeline_4ch`／`test_audio_pipeline_4ch_ulcnet`／third-party
  spatial 測試都印出「All ... tests passed」且沒有任何 `FAIL`；兩次各自 grep 到的 `PASS`/`PASS:`
  行數都是 572（SIMD=0 backend=scalar／SIMD=1 backend=aarch64-neon，兩次行數相同本身就是一個
  一致性訊號）。

### 代價與取捨 (Cost / tradeoffs)

- `set_preset()` 新增一個布林欄位的寫入路徑；因為沒有驗證需求（任何真值都合法，跟
  `set_initial_state()` 同一個契約），放在兩個會拋例外的 dB setter **之後**寫，順序本身就是防呆
  （拒絕的呼叫不會留下「dB 改了但 gate 沒改」的半套狀態），不是額外成本，但是後續有人要再加第四個
  retarget 欄位時要記得遵守同一個順序規則。
- `pyc_drv.c` 的「BALANCED 建構 + runtime retarget」是**測試 harness 的 workaround**，不是產品路徑
  ——真正的缺口是 `se_derive()`／`preset_valid()` 還沒把 `AEC_PRESET_NO_FLOOR` 納入白名單（C 側
  construction-time 的 NO_FLOOR 仍然是一個 Python→C 不對稱：Python 的
  `FourChannelAecConfig(aec_preset=NO_FLOOR)` 建構沒問題，C 的 `se_derive()` 會拒絕）。等那個缺口
  真正修掉之後，這個 workaround 要跟著拿掉，`pyc_drv.c` 可以直接讓 `aec=no_floor` 走construction-time
  路徑。
- 本益：好處是「運轉中切 NO_FLOOR」現在 Python／C、mono／4ch 四邊語義一致，不會有切換後 gate 沒跟上
  的隱性退化；量到的既有 preset 數值成本是零（83 列既有 parity scenario 與 49 條既有 pytest 全部
  逐位元/逐值不變），但不是沒有成本：上一條提到的 `pyc_drv.c` workaround 本身就是一筆要還的技術債，
  `se_derive()` 補上 NO_FLOOR 白名單之前都要留著。

## [Unreleased] · 2026-10-20 · S-28：凍結 `pyc_parity_pipelines.sh` 容差（原卡在 S-27）

### 變更 (Changed)

- `tests/pyc_parity/pyc_parity.py` 的 `TOL1`／`TOL2` 從「第一次量測的暫定值」改為凍結值。凍結前先確認
  `EXPECTED_RED = {}`（S-25／S-26／S-27 都已真正落地，沒有殘留的已知缺口標記），再用
  `pyc_parity_pipelines.sh --layers 1,2` 對現在的真實樹重跑一次全部 scenario：81 PASS、1 SKIP
  （`d.mono.8k`，Python mono pipeline 本就不支援 8 kHz）、1 INFO（`d.mono.cng_lookalike`，本就不判
  gate），沒有 FAIL／UNEXPECTED_GREEN／NOT_RUN，跟 S-27 條目自己重跑的數字一致。
- 凍結值 = 對每個常數「所有引用它的 scenario、兩個 layer」量到的實際最大誤差，取約 1.8–2.7×
  margin（dB 常數取線性域約 1.9× 的 margin）後取一個乾淨數字：

  | 常數 | 舊（暫定） | 新（凍結） | 這次量到的最大值（scenario） | margin |
  |---|---|---|---|---|
  | `TOL1.gain_abs` | 1e-4 | **2e-5** | 8.3e-6（`d.mono.16k256.nr_glue`） | 2.4× |
  | `TOL1.out_abs` | 1e-6 | **2e-7** | 7.5e-8（`d.mono.48k1024.min_apply_ola`） | 2.7× |
  | `TOL1.cng_abs` | 0.0 | 0.0（不變） | 0.0（18/18 case bit-exact） | — |
  | `TOL2.replay_gain_abs` | 5e-2 | 5e-2（不變） | 0.0280（`l2.res4.48k1024.mild.replay`） | 1.8× |
  | `TOL2.replay_out_abs` | 2e-3 | **1e-3** | 4.25e-4（`l2.mono.16k256.replay.out`） | 2.4× |
  | `TOL2.chain_gain_p99_9` | 5e-2 | 5e-2（不變） | 0.0286（`l2.chain4.48k1024.gres`） | 1.7× |
  | `TOL2.chain_out_p99_9` | 1e-3 | **4e-4** | 1.69e-4（`l2.chain4.48k1024.out`） | 2.4× |
  | `TOL2.chain_rms_db` | -40.0 dB | **-45.0 dB** | -50.6 dB（`l2.chain4.48k1024.gres`，最差／最接近門檻） | 5.6 dB（~1.9×） |

  三個 Layer 1 常數（std-math C vs Python）都明顯收緊（13× margin 收到 2.4–2.7×）；Layer 2
  （fast-math/NEON shipped vs std-math C）的 `replay_out_abs`／`chain_out_p99_9` 也收緊到同一量級。
  `replay_gain_abs` 與 `chain_gain_p99_9` 維持 5e-2 不變——**不是因為這兩個本來就夠緊**：這次重跑量到
  的最大值（0.0280／0.0286）比 S-25/26/27 落地前的量測（同一台機器上的 `det1.out` 對照：
  `l2.res4.*.replay` 最大 0.0155、`l2.chain4.48k1024.gres` p99.9 0.0173）大了接近 2×，因為
  S-25/26/27 新接上／新啟用的路徑（R² fold、DT 地板、mixed realign 的 lane-0 far_spec 共用）讓
  Layer 2（fast-math/NEON）跟 Layer 1 之間新多出可觀的分歧量。維持 5e-2 是「這次量到的最大值
  （0.0286）乘以約 1.75× margin」算出來的結果，剛好跟舊的暫定值一樣，是巧合，不是「這兩個不用緊」。
- 套用凍結值後重跑：verdict 與 before 完全一致（81 PASS／1 SKIP／1 INFO，沒有任何列從 PASS 變
  FAIL），`--layers 1,2` 連跑兩次的 `table.out` 與 `results.json` 逐位元相同
  （sha256 都是 `973361a2...`／`3bcc4818...`）。這次凍結所量測的 8 個 `SOURCES` 檔案的 sha256
  （`results.json` 的 `sources_sha256`，`run1`／`run2_frozen`／`run3_frozen` 三次重跑逐檔相同）：
  `4aec_nr_res.c`=`0c8d9ec3f7db...`、`pipeline.py`=`bb0f3e09e223...`、`se_anr_stage.c`=`b06484c9c809...`、
  `aec_res.c`=`c494f9b3f054...`、`audio_pipeline.c`（mono）=`afae24e47412...`、
  `aec_nr_pipeline.py`=`9e701d972d80...`、`cng_aec3.py`=`f8040b813f49...`、
  `conv_chain_4ch.h`=`89cdfae37710...`（皆取前 12 hex 字）——凍結值可以追溯到跑出這些數字的確切程式碼。
- 連帶把輸出介面上宣告「還沒 freeze」的文字改掉：每列的 `tolerance_provisional` 欄位改名
  `tolerance`（repo 內沒有別的程式讀這個 JSON 欄位，純粹是這個檔案自己的輸出，改名不影響任何消費
  端）；JSON 的 `tolerances` 欄位值從 `"provisional"` 改成 `"frozen_2026_10_20_S28"`；終端輸出的
  `tolerances are PROVISIONAL (not frozen)` 改成 `tolerances are FROZEN (S-28, 2026-10-20)`；
  `pyc_parity_pipelines.sh` 開頭的說明段落同步改掉。

### 證據 (Evidence) — mutation-proof 證明凍結後的容差仍然「會失敗」

- `pyc_parity_pipelines.sh --mutation-proof`（內建機制：在一份 scratch 副本裡一次注入一個真實的小
  偏差，確認對應列變 FAIL，再還原確認變回 PASS；詳見 `pyc_parity/mutation_proof.sh` 開頭的機制說明）
  在凍結後的樹上重跑，所有既有、跟容差數值相關的 mutation 仍被偵測到，而且因為容差變緊，偵測門檻比
  凍結前更靈敏：
  - `fold_coef_ulp`（`se_anr_stage.c` 的 R² fold 係數偏移 K 個 float32 ULP）：凍結前（`gain_abs`=1e-4）
    最小偵測到的 K 是 256（見本輪量測前、同一台機器上跑的對照：`SE_verify_20261019/scratch/PARITY_PIPE/mutation_full.log`）；
    凍結後（`gain_abs`=2e-5）K=16 就被 `d.mono.16k256.nr_glue` 抓到、K=256 開始連
    `a.nr4.16k256.glue_cflag`／`glue_fold` 一起 FAIL——偵測門檻變靈敏了約 16×，K=1 兩邊都還是「no」
    （1 ULP ≈ 1.2e-7 相對值，遠低於 Python/C 本來就有的 ~1e-5 量級一致度，兩個容差都量不到，這是
    文件裡本來就說明過的已知限制，不是這次改動造成的）。
  - `xi_min_db_div10_5`（NR `xi_min_db / 10.0f` 改成 `/ 10.5f`）、`cng_seed_1bit`（CNG RNG seed 改 1
    bit）、`aec_floor_balanced_ulp`（RES balanced 地板偏移 K ULP，K=4096 起偵測到，跟凍結前一樣，
    因為這條走的是 RES 路徑、數值鏈路跟 NR 路徑不同，門檻沒有變）、`usable_gate_dropped`（R² fold 的
    `usable_linear` 閂拿掉，跟這次改的 `gain_abs` 直接相關）：全部仍被偵測到（RED，該次子跑
    `exit=1`），對照組（不注入任何 mutation）本身是 `exit=0`，證明「沒有 FAIL 不是因為 harness 本身
    壞了」。還原 mutation 後（腳本自動 `cp` 回原檔）下一個 mutation 的對照同樣 `exit=0`，證明每次都
    確實變回 GREEN。整個 `--mutation-proof` 呼叫本身最終是 `exit=2`（見下面「已知限制」的 flip_s27
    段落），不是 0——這裡回報的是個別 mutation 子跑的 exit code，不是整個指令的 exit code。
  - `mutation_proof.sh` 用 `--layers 1`（只跑 Layer 1），不會碰到 `TOL2` 的
    `replay_out_abs`／`chain_out_p99_9`／`chain_rms_db`。這三個改過的 Layer 2 常數另外手動證明
    「會失敗」：在真實樹的 `pyc_parity.py` 把三個值暫時改到比這次量到的最大值更緊（`replay_out_abs`
    1e-3→2e-4、`chain_out_p99_9` 4e-4→1e-4、`chain_rms_db` -45.0→-51.0dB），用
    `--reuse-build`（沿用同一個 build，不用重編譯）跑
    `--only l2.mono.16k256.replay.out,l2.chain4.48k1024.out,l2.chain4.48k1024.gres`：三列全部
    FAIL（`exit=1`，`RESULT: FAIL`）。改回真正的凍結值（1e-3／4e-4／-45.0）用同樣的 `--only` 重跑：
    三列全部回到 PASS（`exit=0`，`RESULT: OK`），數字跟之前的量測（`max_abs=0.000425`／
    `p99_9=0.000169`／`rms_db=-50.6`）完全一致。
  - 完整表格見 `scratch/S28_freeze_pyc_tol/mutproof.log`
    （Layer 1 的 5 個 mutation）；TOL2 的 RED/GREEN 紀錄見同目錄下
    `run3_frozen/canfail.out`（RED）與 `run3_frozen/canfail_restored.out`（GREEN）。

### 已知限制 (Known limitation)

- 這次凍結的量測只在這台機器、這個編譯器（Apple clang 17、arm64 macOS）上做，沒有跨機器／跨
  toolchain（例如目標平台的交叉編譯器）重跑過；凍結值的 margin（1.8–2.7×／~1.9× in amplitude）是
  根據「多一次 libm／NEON lowering 版本差異，不應該把 PASS 翻成 FAIL，但誤差加倍應該被抓到」這個
  經驗法則訂的，不是從多台機器的實測分佈反推出來的統計門檻；如果之後真的在另一個 toolchain 上出現
  誤報，margin 需要重新量測調整。
- `TOL2.chain_rms_db` 用 dB 表示、margin 只有 5.6 dB（~1.9× in amplitude）：如果真實回歸把誤差放大
  不到 1.9 倍（例如放大 1.5 倍，從 -50.6 dB 變成約 -46.9 dB），仍然會落在新門檻 -45.0 dB 之內、不會
  被這個常數單獨抓到——跟其他線性容差「加倍一定抓到」的保證不完全一樣，這是 log 尺度本身的取捨，
  已誠實記錄不是「無成本」。
- **附帶發現、跟這次凍結容差無關、不在這次範圍內修，但重要（澄清：是測試選錯列，不是 harness 量
  不到這個回歸）**：`mutation_proof.sh` 的 `--mutation-proof` 這次整個呼叫是 `exit=2`（不是全綠），
  因為跑到 `flip_s27_python_borrow` 時直接中止：它要比對的
  `raise ValueError("linear AEC lanes do not share one far-end spectrum")` 字串在現在的
  `pipeline.py` 裡已經找不到（S-27 真正的修法就是刪掉這一行），`perl` 替換沒有任何效果，
  `mutation_proof.sh` 自己偵測到「檔案沒被改到」就回報 `pattern drifted?` 並 `return 2`。
  在它之前的 `flip_s25_c_flag_forced`（讓 `four_aec_fuse()` 強制 `frame->usable_linear = 1`）、
  `flip_s26_python_dt_floor`（讓 `PostBeamResidualSuppressor.set_preset()` 再多做一次 DT 地板
  retarget）兩個子跑都回報「沒偵測到」（`a.nr4.16k256.legacy_amin`／`b.res4.16k256.aggressive`
  都還是 PASS）。**查證後確認：這不是 harness 量不到這個回歸，是這兩個 flip 測試自己檢查的列選錯
  了**。手動把 `flip_s25` 的 mutation（強制 `usable_linear=1`）重新套到 scratch 副本、把 `--only`
  擴大到也包含 `a.nr4.16k256.verdict_flag`（原本的 `M_ONLY` 沒有這一列）：這一列立刻變 FAIL
  （`mismatch 288 of 1250 hops, Python verdict true 962, C flag true 1250`，`exit=1`）；還原
  mutation 後重跑同一組 `--only`：三列都回到 PASS（`exit=0`）。也就是說，`usable_linear` 被強制設
  1 這種回歸，`a.nr4.*.verdict_flag`（逐 hop 比對 Python 判定 vs C 旗標）本來就抓得到，`flip_s25`
  只是檢查了 `legacy_amin`／`glue_cflag` 這兩個已經不會因為這個特定 mutation 而改變的列
  （`legacy_amin` 測的是另一個獨立開關；`glue_cflag` 在 16k256 這個 grid 碰巧量到的差異本來就低於
  容差，forcing usable_linear=1 沒有把它推過門檻）。`flip_s26` 大概同理（疑似把「S-26 真正的修法」
  又套了一次、變成冗餘動作），但沒有像 `flip_s25` 一樣逐列重新驗證過，這點保留、不下定論。
  這三個 flip case 本質上是「S-25/26/27 還沒真正落地、`EXPECTED_RED` 還非空」那個時期寫的，現在
  `EXPECTED_RED = {}` 之後，它們檢查的列組合已經過期，需要重新設計（例如 `flip_s25` 應該把
  `a.nr4.*.verdict_flag` 也納入 `M_MUST`），不是這次凍結容差的範圍，留給之後整理 `mutation_proof.sh`
  的人。
- **附帶發現、已查明是良性、不影響這次的量測**：凍結容差的過程中，`4ch_aec_bf_nr_res/4aec_nr_res.c`
  在工作樹上被改動過一次（本次工作期間；`git status` 顯示它跟 `pipeline.py` 一樣是既有的
  uncommitted 修改）——sha256 從量測時的 `0c8d9ec3...`（`run1`／`run2_frozen`／`run3_frozen` 三次都
  是這個值）變成之後的 `97bbba58...`。逐行 diff 確認只有一段註解被精簡（S-27 那段「fused X 還是取
  lane 0 的 far_spec」的說明文字去重），沒有任何數字、邏輯或空白敏感的程式碼改動；`pipeline.py` 全程
  sha256 不變（`bb0f3e09...`）。拿這之後的狀態重跑一次完整 `pyc_parity_pipelines.sh --layers 1,2`
  （`run4_final`）：81 PASS／1 SKIP／1 INFO，跟 `run2_frozen` 的 `table.out` 逐位元相同、83 列的
  status 逐列相同——確認這次凍結的容差值對這個註解級別的改動不受影響，不需要重新量測。

## [Unreleased] · 2026-10-20 · S-27：4ch 混合 warm/soft-realign hop 的 far-spectrum 處理

### 修復 (Fixed)

- **4ch 前端在混合 warm/soft realign 的那一個 hop 上，原本仍無條件把 lane 0 的 far FFT 共享給
  其他三個 lane**（`four_aec_front_process()` 的 per-lane loop），即使 soft lane 在同一 hop 已被
  `aec_apply_external_realign()` 清空 far_buffer OLA history、warm lane 沒有——兩者的真實 far FFT
  本該不同，卻被當成相同處理；`fuse_contexts()` 的 provenance 快路徑（`far_spec_shared_hop`）因為
  計數器證據（lane 0 跑 1 次 transform、lane 1-3 跑 0 次）仍然成立而直接 accept，從未察覺這個差異，
  soft lane 靜默吃到 lane 0 未重置的 far_spec（錯誤但不報錯）。

  修法是「路由」層級的改動，沒有新增任何 DSP/FFT 邏輯：`aec_process_context()`（每個 lane 原本就有、
  非共享情境下每個 lane 本來就會用的真正 far FFT）早就存在，只是這一個 hop 從沒被路由過去。新增
  `FourAecNrRes.far_spec_mixed_hop`（跟既有 `far_spec_shared_hop` 同 precedent），在
  `delay.changed` 的 realign 迴圈裡用「這一 hop」的 warm/soft outcome 計數（不是累計總數）判定
  `hop_warm>0 && hop_soft>0`；per-lane loop 的分支條件從 `ch==0` 改成
  `ch==0 || p->far_spec_mixed_hop`，讓每個 lane 在混合 hop 上各自跑自己的真實 far FFT（這一 hop
  最多多付 3 次 FFT，符合先前估算）。既有的 provenance 計數器邏輯本身沒動——lane 1-3 這時
  `ran==1`（不再是 0），自然讓 `lanes_borrowed=0`、`far_spec_shared_hop=0`，如實反映「這 hop 沒有
  共享」。

  只做上面這一半、不放寬 `fuse_contexts()` 的 far_spec 一致性檢查，會讓每個混合 hop 變成全部
  reset（`four_aec_nr_res_far_spec_agrees()` 的全比較分支會如實偵測到四個 lane 的 far_spec 現在真的
  不同而 reject，`four_aec_fuse()` 接著呼叫 `four_aec_nr_res_reset()`）——比現有的靜默錯誤還糟，等於
  把每次混合 realign 都變成一次全重置。因此兩半一起做：`fuse_contexts()` 在呼叫
  `four_aec_nr_res_far_spec_agrees()` 之前先檢查 `p->far_spec_mixed_hop`，是的話整段比較都跳過——
  不是放寬斷言本身，`four_aec_nr_res_far_spec_agrees()` 這個 predicate 維持原語意不變，
  `test_far_spec_provenance_guard` 仍直接驅動它；融合後交給下游的 `X`（`four_aec_fuse()` 裡的
  `frame->x`）不管哪種情況都還是取 lane 0 的 far_spec，跟 `pipeline.py` 的語意一致。

  Python 側對應：`pipeline.py` 的 `_fuse_contexts()` 本來就沒有任何 FFT 共享優化（每個 lane 本來就
  各自算自己的 context），差異只在它原本對跨 lane far_spec 不一致會 `raise ValueError`。移除這個
  raise，改成接受並直接取 `far_specs[0]`（跟 C 核心融合後的 `X` 同一語意），其他三個 lane 保留自己
  未融合的 far_spec。

  連帶更新了三處描述這個機制的註解（`4aec_nr_res.c` 的 `four_aec_nr_res_far_spec_agrees()` 上方、
  同檔案 per-lane loop 裡）、`4aec_nr_res_internal.h` 裡宣稱「lane 永遠共享同一個 far spectrum，
  所以這個 predicate 的拒絕分支測試不到」的過期說法、以及 `README.md`「Far-end FFT sharing」整段
  與開頭的流程圖，全部改成反映修法後的實際行為（而不是像草稿版本那樣停在「這裡還沒修」）。

- `tests/pyc_parity/pyc_parity.py`：確認 `pyc_parity_pipelines.sh --only c.realign4` 重跑後
  `c.realign4.48k1024` 變成 `PASS`，才移除 `EXPECTED_RED['c.realign4.48k1024'] = 'S-27'` 標記；
  同步修正兩處描述舊「raise」語意的過期文字（該 scenario 的標題、`verdict_flag` 行的 producer
  字串），並把 `py4_run()` 裡原本靠 monkeypatch `_fuse_contexts()`、`except ValueError` 來模擬
  「lane 0 far_spec 融合」的輔助邏輯整個拿掉——那段邏輯現在是死碼（真正的 `pipeline.py` 已經直接
  做這件事，不會再 raise），改成直接用 pipeline 自己的 `realign_warm_lane_count`／
  `realign_soft_lane_count` 計數混合 hop 數，供既有的 `s27_borrowed_hops` 指標使用。

### 證據 (Evidence)

- `pyc_parity_pipelines.sh --only c.realign4 --layers 1`：修法前 `c.realign4.48k1024` =
  `EXPECTED_RED(S-27)`（`python_raised=True` at hop 190、
  `python_message="linear AEC lanes do not share one far-end spectrum"`；C 側當時沒被影響，
  `c_fail_hop=-1`，因為舊碼是強制共享而非報錯——這條本身是「靜默吃錯值」而非「崩潰」）。修法後
  重跑同一 scenario：`PASS`（`python_raised=False`，C 側 `c_fail_hop=-1` 不變、
  `c_mixed_warm_soft_hops=[190]` 確認那一 hop 真的是混合 hop、且沒有觸發
  `four_aec_nr_res_reset()`）。`pyc_parity_pipelines.sh --layers 1,2`（全部 scenario）重跑：
  81 PASS、1 SKIP（8k，既有、跟這次修法無關）、1 INFO，`RESULT: OK`，沒有任何 `FAIL` 或
  `UNEXPECTED_GREEN`；Layer 2（SIMD=1 shipped build）的 `l2.realign4.48k1024` 也 PASS（events
  `[(190, 2, 2)]`）。
- `measure_mixed_realign.py`（唯讀，驅動 Python 參考 pipeline 的 `front_process()`，不呼叫
  `fuse()`，所以不直接測到這次修法動到的程式碼，但確認 realign 計數本身沒被動到）：修法前後對
  `datasets/aec_together`、`datasets/aec_take_turn` 兩個真實 4-mic 錄音跑出**完全相同**的結果——
  共 12620 個 hop，3 次 realign，其中 1 次是混合 hop（`aec_together` hop 976：3 個 lane warm、
  1 個 lane soft，`delta=60`），0 次混合且 `delta<0`（retard）。
- 4ch 測試套件（`4ch_aec_bf_nr_res/` 的 `make SIMD=0 test`、`make SIMD=1 test`）：572/572 全部
  PASS，兩個 SIMD 設定都是。pipelines 頂層（mono）測試套件（`make SIMD=0 test`、
  `make SIMD=1 test`）：全部 PASS，exit code 0，沒有 `FAIL`——mono pipeline 完全不含 multi-lane
  realign 的概念，架構上就沒有路徑會碰到這次改動的任何檔案（`4aec_nr_res.c`、
  `4aec_nr_res_internal.h`、4ch 的 `pipeline.py`），這次重跑是確認性的，不是「可能受影響」。

### 已知限制 (Known limitation)

- 目前唯一能取得的真實錄音證據只有 **3 次 realign、其中 1 次混合 hop、0 次混合且 `delta<0`**
  （來自 `aec_together`／`aec_take_turn` 兩個既有 4-mic 錄音）。這是很小的樣本，尤其
  「混合 hop 又正好是 delay 後退（retard）」的組合完全沒有實例可驗證。使用者已核准在這個樣本量
  下先行落地此修法，重新驗證（在更大樣本或新的多 realign 語料上）延後到有這樣的語料時再做。

## [Unreleased] · 2026-10-20 · S-25：4ch NR 重新接上 `usable_linear`（遺失更新回歸）

### 修復 (Fixed)

- **`4ch_aec_bf_nr_res/4aec_nr_res.c` 的 `four_aec_fuse()` 從未寫入 `frame->usable_linear`**，
  一律是 `memset` 留下的 0。這是兩個 commit 之間的遺失更新：`49f73b2` 加進的那段邏輯，被
  `af914d4` 意外刪掉的 9 行帶走，之後沒人補回來。後果：NR glue 永遠把已收斂的線性 AEC 估計
  當成「不可信」，R² fold 對 4ch 一直是死碼，double-talk／far-only 時少做了該有的回音／噪聲
  抑制；near-end-only 的行為不受影響（fold 只在 `usable_linear` 為真時啟動，一直是假，所以
  一直不啟動，近端這段時間本就沒有 fold 的差異可言）。
  修法：補回 8 個插入點——`FourAecLaneSnapshot` 加 `usable_linear` 欄位、`bind_lane_view()` 從
  `AecResContext` 複製該欄位（`AecResContext.usable_linear` 本身一直都有，只是沒傳到這個快照
  裡）、`fuse_contexts()` 的簽名／seed／AND-fold（只 AND 過 `lane_contributes` 為真的 lane，
  和 `all_converged` 的 fold 同一個 gate，不是對全部四個 lane 做 AND）、`four_aec_fuse()` 裡的
  區域變數與呼叫處、最後寫回 `frame->usable_linear`。mono pipeline（`mono_aec_nr_res/audio_pipeline.c`
  第 655 行附近的 `f->usable_linear = ctx.usable_linear;`）本來就是對的，沒受這個回歸影響，這次
  也完全沒改動。
- `tests/test_4aec_nr_res.c` 的 `test_fuse_frame_contract()` 加兩條斷言：`usable_linear` 每個 hop
  都是 0/1、且在四個等權重、同步收斂的 lane 下於 60-hop 視窗內確實變成過 1（這條在修法前的程式碼
  上必定 FAIL：驗證過，暫時把 `frame->usable_linear = usable_linear;` 那行拿掉重跑會讓它 FAIL，
  補回來後 PASS）。另加一個新測試
  `test_fuse_usable_linear_and_over_contributors()`：一個 lane 權重設 0（不貢獻）且刻意用跟
  far-end 不相關的雜訊餵它，讓它自己的 `usable_linear` 全程為假，其餘三個 lane 正常收斂——同一個
  front frame 分別用「該 lane 權重 0」與「該 lane 權重 0.25（變成貢獻者）」兩組權重各 fuse 一次：
  前者必須讀到 `usable_linear=true`（否則是誤把 AND 做在全部四個 lane 而非只做在貢獻的 lane
  上），後者必須讀到 `false`（否則是「永遠強制 1」或「漏掉 AND-fold」這類修法，第一個既有測試
  等權重的設計看不出這種差異，這個新測試可以分辨）。兩個 mutation 都已手動注入驗證過會讓對應
  斷言 FAIL。
- `tests/pyc_parity/pyc_parity.py` 移除 `a.nr4.*.verdict_flag`／`a.nr4.*.gnr_e2e`／`a.nr4.*.legacy_amin`
  三族的 `EXPECTED_RED(S-25)` 標記（16k/256、16k/512、48k/1024 三個 grid）；`pyc_parity_pipelines.sh`
  重跑後這三族在三個 grid 上全部變成 PASS（`mismatch_hops=0`；`legacy_amin` 的 R² fold on/off
  差異在 C 內部也量到了）。**附帶發現、與本次修法本身無關、但已查清並一併收尾**：這次重跑時
  `b.res4.*.aggressive`（另一個已知缺口 S-26，RES 的 double-talk floor）也是
  `UNEXPECTED_GREEN(S-26)` 而非預期的 `EXPECTED_RED(S-26)`。最初以為是跟這次 S-25 修法無關的
  既有未解之謎（已查過 `se_res_stage_gain()` 根本沒把 `usable_linear` 傳進 `AecResInput`，把
  S-25 的修法暫時還原重跑 `--only b.res4` 數值完全不變，確認跟 S-25 無關）。後來核對同一時間
  平行進行的另一批修法才查明：S-26 本身（`pipeline.py` 的 `PostBeamResidualSuppressor` 補上
  DT 地板傳播、`FourChannelAecConfig` 加 `aec_preset`）剛好在同一個工作樹上落地，這個
  `UNEXPECTED_GREEN(S-26)` 正是 S-26 真的修好了的結果，不是謎——已重跑 `--only b.res4` 確認三個
  grid 的 `aggressive` 全部 PASS（`max_abs` 7.69e-6～9.78e-6，容差 1e-4），並把
  `EXPECTED_RED(S-26)` 標記移除。
- 用真實重建的 `audio_pipeline_4ch_raw` 跑 `evaluate_external_recordings.py`（`aec_take_turn`、
  `aec_together`，16 kHz/256 與 16 kHz/512）：near-end-only attenuation 基本不變（例如
  `aec_take_turn` @256 12.4164→12.4165 dB），double-talk／far-only attenuation 上升——
  `aec_together` double_talk @256 15.51→16.49 dB、@512 13.83→15.24 dB；far_only @256
  18.69→18.85 dB、@512 19.63→20.07 dB；`aec_take_turn` far_only @256 20.04→20.13 dB、@512
  21.06→21.15 dB。silence 與 delay 相關欄位（`final_delay_error_samples` 等）不變。

### 證據 (Evidence) — AECMOS 近端品質量測

- 用官方未修改的 `FastAECMOS`（`AEC/python/bench_aecmos.py`）對修前（單點還原
  `frame->usable_linear = usable_linear;` 這一行寫回，其餘 7 個插入點不動）／修後兩個重建的
  `audio_pipeline_4ch_raw` 跑分：`datasets/aec_take_turn`、`datasets/aec_together`（各 16 kHz
  @256／@512）＋ 4 組 `~/Desktop/exp` 真實錄音（`ref.wav` 已核對是原始遠端回授訊號，非 AEC
  內部估計值），10 秒窗分話務型態（dt／nst／st），排除收斂期第 0 窗。雙講（dt）19 個窗：
  `echo_mos` 平均 +0.071（回音消除變好，符合預期）、`deg_mos` 平均 **−0.073**（74% 的窗變差，
  6 個窗掉超過 0.1，最差窗 `0827_doub_tog` w4 掉 0.333）；近端單講（nst）2 個窗持平；遠端單講
  （st）8 個窗因該情境本來就無近端語音，`deg_mos` 兩邊都頂在 ≈4.998～4.999，量不出東西。
- **根因核對（opus 二次確認）**：`frame->usable_linear` 在整條 4ch pipeline 裡只有一處讀取——
  `pipelines/common/se_anr_stage.c:51` 的 `if (use_r2 && frame->r2 && scratch &&
  frame->usable_linear)`，是 RES→NR 的 R² 注入閘。`four_aec_fuse()` 進來時先把 `frame`
  `memset` 成 0，所以 `af914d4` 刪掉這行寫回之後，`usable_linear` 永遠是 0——不是只關掉
  cold-start 保護，是把整條 R² 注入路徑整個斷開。因此這次 AECMOS 的「修前」其實量的是「R² 注入
  全關」，「修後」量的是「R² 注入（有 cold-start gate）重新接上」，兩者的差異正是**R1 設計本來就
  接受的雙講代價**（`49f73b2` 之前的長期上線行為是無條件注入，由 `legacy_amin` 這個 static flag
  控制；`49f73b2`→`af914d4` 中間那段「有 gate」的版本只上線過約 1.5 小時，是意外被刪除的那個片段，
  不是新增邏輯），不是這次修法引入的新過度抑制 bug。
- **裁定：維持現狀出貨，不需要為此延後或另修**。雙講 deg_mos 的下降是重新接上既有、曾經上線過的
  RES→NR 耦合之後的既有代價，衡量基準應該是「R² 注入開」的 R1 行為，不是「注入全關」這個意外狀態。
  若未來認為這個雙講代價太高，屬於既有的 DT over-suppression／RES 操作點調校 arc 的範疇（調多重的
  R² 權重），不應該用「讓這條路徑維持斷開」的方式處理。

### 已知限制 (Known limitation)

- 這批 AECMOS 證據只有 16 kHz（這個 checkout 沒有 48 kHz 的 AECMOS 模型）、只建置
  SIMD=1/NE10/arm64（未測 SIMD=0 或目標平台工具鏈）；近端單講只有 2 個窗，證據薄；
  `~/Desktop/exp` 的話務型態是依檔名指派、不是量測到的近端活動量；是 36 個窗的配對證據，
  不是有統計檢定力的語料級結論。

## [Unreleased] · 2026-10-19 · 跟上 AEC／NR 的 C 符號改名、移除 `BACKEND=kiss`、`four_aec_nr_res_stage_configs()` 在 `enable_nr = 0` 時不組 NR 設定

所有 pipeline 的輸出與改動前逐位元相同：以 `harness/pipelines_suite.sh` 對改動前的樹與改動後的樹各跑一次
（SIMD=1 與 SIMD=0，含 mono／4ch／ULCNet／DFN 的 CLI 輸出檔、`--print-mem-size` 表、`make test` 變體、
AINR／AIAEC 的 C 測試與 C/Python parity），manifest 逐行相同。唯一的文字差異是 README、註解、
`make` 的 banner 與 `make help` 類說明；`build_flags_hash`、`layout_version`、`backend_id` 的數值都沒變。

### 變更 (Changed)

- **公開 API 的行為改變：`four_aec_nr_res_stage_configs()`。** `cfg->enable_nr == 0` 時不再呼叫
  `se_anr_compose_config()`，`*nr_cfg` 回傳全零的 `AnrConfig`（`se_anr_stage_mem_size()` 對它回 0，不可拿去
  初始化 NR）；`enable_nr != 0` 時與先前逐位元相同。函式簽章、回傳值規則（NULL 引數或核心會拒絕的設定
  回 -1）、`res_cfg` 都不變。五個呼叫端：
  - `4ch_aec_bf_nr_res/audio_pipeline_4ch.c`：本來就只在 `enable_nr` 時才計算 `nr_bytes`，不需改。
  - `4ch_aec_bf_nr_res/4aec_nr_res_static.c`：`plan_stages()` 只在 `enable_nr` 時才要求 `nr_bytes != 0`
    （先前無條件要求，`nr_bytes == 0` 會讓整個規劃失敗）。
  - `4ch_aec_bf_nr_res/tests/conv_chain_4ch.h`：NR 區塊大小只在 `enable_nr` 時計算，`enable_nr = 0`
    的測試鏈不再多配一塊用不到的 NR 記憶體。
  - `4ch_aec_bf_dfn_res/4aec_dfn_res.c`：要求 `enable_nr == 0`，丟棄本地 `nr_cfg`，加註解說明。
  - `4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c`：新增檢查——`enable_nr = 0` 時 `nr_cfg` 為全零、
    `res_cfg` 與 NR 開啟時推導的一致。已做變異測試：把閘門拿掉，這項檢查與整個
    `test_stage_sizing_and_switches` 會失敗。
  - 這個修法**不會**讓任何 Makefile 的連結依賴變小：`4aec_nr_res.o` 仍參照 `se_anr_compose_config`，
    `audio_pipeline_4ch.o` 仍直接參照 `se_anr_stage_*`（`4ch_alignulcnet` 照舊連結 `se_anr_stage.o`）。
- **跟上 AEC 的 C 符號改名**（AEC 的改名表見 `lib/aec` 的 CHANGELOG）：`DelayAec3` → `DelayAec`、
  `delay_aec3.h` → `delay_aec.h`、`delay_aec3_*` → `delay_aec_*`、`AEC3B_*` → `AECB_*`、
  `aec3_balanced_config.h` → `aec_balanced_config.h`、`pbfdkf_update_weights_aec3` → `pbfdkf_update_weights`、
  `AEC3_FILTER_NOISE_GATE_POWER_FLOAT` → `AEC_FILTER_NOISE_GATE_POWER_FLOAT`。
  消費端：`4aec_nr_res.{c,h}`、`4aec_nr_res_internal.h`、`tests/test_4aec_nr_res.c`、
  `4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c`、`mono_alignulcnet/tests/test_audio_pipeline_ulcnet.c`，
  以及未追蹤的診斷程式 `experimental/linear_aec_probe.c`（`DelayAec`、`delay_aec_estimated_delay`）、
  `experimental/algorithm_completeness/check_aec_contract.c`（`pbfdkf_update_weights`、
  `AEC_FILTER_NOISE_GATE_POWER_FLOAT`）、`tools/joint_postfilter_cli.c`（`aec_balanced_config.h`、`AECB_SQRT2_SIN_LUT`）；
  Python 側只跟著改讀 C 標頭的地方（`tests/test_cng_aec3.py`、`cng_aec3.py` 的說明文字），
  另有只改註解裡舊名的 `4ch_aec_bf_nr_res/pipeline.py`、`tests/_scripted_delay.py`、`tests/test_pipeline.py`。
- **跟上 NR 的 C 符號改名**：`McraNoiseEstimator` → `NoiseEstimator`、`mcra_noise_estimator.{h,c}` →
  `noise_estimator.{h,c}`、`mcra_{get_noise_psd,update_ex,accumulate_init_power,init_noise}` →
  `noise_estimator_*`。使用者是診斷探測程式：`tools/res_nr_{coupling,dd,near,external_noise,tracker}_probe.c`、
  `experimental/nr_soft_scene_probe.c`、`experimental/algorithm_completeness/nr_probe.c`；
  指到 `mcra_noise_estimator.c` 路徑的建置腳本與 Python 一併改（`tools/build_*.sh`、
  `tools/evaluate_res_nr_coupling.py`、`tools/finalize_res_nr_repair.py`、`experimental/algorithm_completeness/build.sh`）。
- `4aec_nr_res.c` 的 `build_flags_hash` 輸入字串原本含 `aec3`（第 4 條 lane 的標籤）。改用巨集
  `LANE_TAG(i)` 組出同一個字串，hash 值不變（已用 `--print-mem-size` 的 `build_flags_hash` 比對）。
- 精簡經手檔案裡帶日期或 campaign 代號的註解（`4aec_nr_res.{c,h}`、`tests/test_4aec_nr_res.c`、`Makefile`），
  保留技術理由。
- `README.md` 的 Memory Budget 表改為只列 NE10 的現行實測值（8 kHz 368,608 B；16 kHz 536,736 B（256/128）／
  691,360 B（512/256）；48 kHz 1,662,192 B），控制區塊 160 B 計入總量；`4ch_alignulcnet/README.md` 的 pool
  大小改為 3,172,656 B（16 kHz）／6,659,280 B（48 kHz）。`mono_aec_nr_res/example_board_adapter.c` 的 arena
  大小說明同步（2 MiB 對 48 kHz 的 1,662,192 B 約有 425 KiB 餘量）。

### 移除 (Removed)

- `BACKEND=kiss`：`pipelines/Makefile`、`4ch_aec_bf_nr_res/Makefile`、`third_party/Makefile`、`AINR/Makefile`
  只接受 `ne10`，其他值在解析階段以 `BACKEND must be ne10` 報錯。`BACKEND` 仍是變數，因為它是 `obj/`、
  `bin/`、`dist/` 目錄名的一部分，也轉送給 audio_common／lib/aec／lib/nr。連帶移除：
  `-I$(AC_DIR)/lib/kiss_fft`、`LINK` 依 backend 切換（改為恆用 `$(CXX)`）、`_cfg_guard` 的 ne10-only 條件、
  banner 與用法說明裡的 `make BACKEND=kiss` 範例、README 與測試檔頭的 kiss 建置指令。
  `AIAEC/Makefile` 本來就沒有 BACKEND 分支，不需改。
- **公開巨集移除（API 變更）：四個 kiss 後端 ID，值都是 1。** 各在一個公開標頭裡：
  `FOUR_AEC_NR_RES_BACKEND_KISS`（`4ch_aec_bf_nr_res/4aec_nr_res.h`）、
  `AUDIO_PIPELINE_BACKEND_KISS`（`mono_aec_nr_res/audio_pipeline.h`）、
  `AUDIO_PIPELINE_ULCNET_BACKEND_KISS`（`mono_alignulcnet/audio_pipeline_ulcnet.h`）、
  `DFN_RES_STAGE_BACKEND_KISS`（`dfn_res_stage.h`）。程式若拿它們跟 descriptor 的 `backend_id` 比對，
  升級後會編譯失敗。對應的 `"kiss"` 字串比對與 `backend_id_name()`／`static_main.c` 的 `case` 一併移除
  （`4aec_nr_res.c`、`4aec_nr_res_static.c`、`audio_pipeline_4ch_raw.c`、`audio_pipeline_4ch_static.c`、
  `mono_aec_nr_res/audio_pipeline.c`、`mono_aec_nr_res/static_main.c`、`mono_alignulcnet/audio_pipeline_ulcnet.c`、
  `dfn_res_stage.c`）。四個 `*_BACKEND_NE10`（值 2）不變，descriptor 裡的 `backend_id` 仍是 2；值 1 保留不再回傳。
- `audit-no-stdio-stack`（pipelines）與 `audit-no-stdio`（4ch）原本對 kiss 只查較小的 stdio 符號集合，
  現在只剩完整集合（`stderr`／`__stderrp`／`fputc`／`fwrite` 也算），與先前 ne10 的檢查相同。
- `scripts/audit_fp_contract.sh` 的預設 backend 清單由 `ne10 kiss` 改為 `ne10`。
- `scripts/test_build_isolation.sh`：所有 `BACKEND=kiss` 改為 `ne10`（dist 路徑、MANIFEST 的 producer
  簽章、`[ne10]` 日誌比對、`aec_post.o`／`noise_estimator.o` 檔名一併跟上）。其中兩組測試**語意改寫**，
  不只是改字：
  - SP1 原本是「kiss → ne10 → kiss」的 A→B→A（以後端當作 A／B 兩個建置），後端只剩一種後，改為
    「SIMD=1 → SIMD=0 → SIMD=1」的 A→B→A：斷言變成「SIMD=1 與 SIMD=0 落在不同 bin 目錄」「SIMD=1 的 bin 目錄在
    A→B→A 後不變」「第三次不重連 `aec_nr_pipeline`」「SIMD=1 連得到 ne10 FFT 符號」（原本的 ne10 斷言移到
    SIMD=1 的建置上）「SIMD=0 只連標量 ne10 FFT、不含 NEON FFT 符號」（新增），取代原本的 kiss 對應斷言
    （kiss 的 bin 目錄穩定、kiss 第三次不重連、kiss 不連 ne10 FFT 符號）。
  - SP-S12 原本驗證「`BACKEND=kiss` 時，CXX 與 CC 目標不一致的守衛不生效、建置仍成功」，改為驗證
    `TOOLCHAIN_CHECK=0` 同樣略過該守衛、建置成功（同一個目標不一致的 CXX shim）。
  - `resolve_producers` 新增可選的 SIMD 引數。
  這兩組測試的斷言名稱與條數和改動前不同，因此這個腳本的 PASS 清單不能與改動前逐行比對。

### AINR／AIAEC

- `AINR/Makefile`：加上與其他 Makefile 相同的 `BACKEND must be ne10` 檢查（它把 `BACKEND` 轉送給
  audio_common 以取得封存檔路徑，audio_common 現在拒絕其他值）。`CFG_SIG` 不變。
- `AIAEC/Align_ULCNet/ulcnet_process.h` 與 `AIAEC/tests/test_ulcnet_process_c.py`：移除說明文字中的
  kiss／backend 切換敘述。AINR 與 AIAEC 的 C 原始碼沒有其他改動。

### 已知、未動

- `Audio_ALG/docs/`（Markdown 與 `docs/html`）仍有引用舊 C 名稱與 `BACKEND=kiss` 的敘述
  （`docs/c_user_manual_zh_TW.md`、`docs/audio_libraries_c_api_zh_TW.md`、`docs/integration_*`、
  `docs/pipeline_board_integration.md`、`docs/release_checklist.md`、`docs/html/*` 等），不在這一輪範圍；
  只有 `docs/integration_4ch_core_zh_TW.md` 裡三處 `four_aec_nr_res_stage_configs()` 的敘述同步了
  `enable_nr = 0` 的新行為。
- `pipelines/tools/` 與 `pipelines/experimental/` 曾是未追蹤的研究用程式，其中多個檔案（`nr_overlay.h`／
  `anr_echo_internal.h`／`run_frame_gain_stage` 參數個數／`NoiseEstimator::scene_change_tonal_veto`／
  `AudioPipelinePostView` 等）在改名之前就已經編不過（逐一驗證過，錯誤相同）。結構性整理這一輪
  （2026-10-20）確認這些檔案在真正的建置系統（所有 Makefile）裡零引用，使用者核准整批刪除：
  `pipelines/tools/` 下 12 個孤兒 `.c`、4 個獨立 `build_*.sh`、135 個未追蹤的 `.py` 診斷腳本（保留 11 個
  確認被 Makefile 使用的 `.py` 工具不動），以及整個 `pipelines/experimental/`（17 個檔案，同樣未追蹤、
  同樣零建置引用，且其中 5 個檔案 import 了已刪除的 `pipelines/tools/` 模組，刪除前就已經壞掉）。
