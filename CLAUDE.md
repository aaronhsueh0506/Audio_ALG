# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Three independent product surfaces that share build conventions but must never share signal
grids or checkpoint contracts:

- **`pipelines/`** — the deployable conventional AEC + echo-aware NR/RES path (mono and
  4-channel), plus the DeepFilterNet2 (`pipelines/dfn/` hop glue, run by the `mono/aec_dfn_res` and `4ch/aec_bf_dfn_res` mains) and Align-ULCNet
  (`pipelines/{mono,4ch}/alignulcnet/`, on `stream/aiaec/ulcnet`) application examples that sit on top of it. This is the reference
  integration path; `AINR/` and `AIAEC/` models are not silently wired into it.
- **`AINR/`** — standalone AI speech-enhancement models (RNNoise-ERB, GTCRN, DeepFilterNet2,
  DeepFilterNet3): Python training/export side; the C runtime itself lives in `lib/ainr/<model>/`.
- **`AIAEC/`** — four neural AEC candidate architectures (Align_ULCNet, Align_CRUSE,
  DeepVQE_S, CAGCRN), every one a joint AEC+RES+NR route (no AEC-only candidate); Python
  training/export side, C runtime in `lib/aiaec/<model>/`.
- **`lib/ainr/`, `lib/aiaec/`** — the 6 models' C cores (single-model-release restructuring: each
  model builds and ships independently of the rest of its family); **`stream/`** — for DFN2,
  DeepVQE-S and Align-ULCNet the per-call lifecycle over the lib's per-frame functions (catalog and graph
  selection, `StreamWork`, input range checks, STFT/ISTFT, N > 1 loops; see below).

## External dependencies: mid-transition, read this before touching build paths

AEC and NR used to be pure git submodules (`lib/aec`, `lib/nr`); audiolib (`common/` = shared
FFT/fast-math/SIMD) used to be resolved by a single hardcoded pin file. As of a recent
"dependency-path convergence" pass, the **default** build source for all three is sibling dev
checkouts next to this repo (`../AEC`, `../NR`, `../audiolib`), resolved by `lib/deps.mk` against
`lib/aec.pin` / `lib/nr.pin` / `lib/audiolib.pin`. The `lib/aec` and `lib/nr` git submodules still
exist, still carry a gitlink, and still build standalone — they are a second, not-yet-retired
record of the same two dependencies (`AEC_DIR=lib/aec/c_impl NR_DIR=lib/nr/c_impl` to force the
submodule checkout). Migrating the two gitlinks onto the `.pin` files and retiring the submodule
is a deliberately separate, not-yet-done step. Any Makefile or script that needs AEC/NR/audiolib
paths should `include`/call `lib/deps.mk` rather than hardcoding a relative path a second time —
that duplication is exactly what silently drifted out of sync before.

```bash
make -f lib/deps.mk bootstrap-deps   # clone whichever of ../AEC, ../NR, ../audiolib is missing, pinned
make -C pipelines check-pins         # verify AEC_DIR/NR_DIR/AUDIOLIB_DIR match their .pin (+ gitlink drift note)
```

## Common commands

```bash
# Build conventional mono examples + the DFN2 stage archive
make -C pipelines
# Build the 4-channel (AEC array instance) examples
make -C pipelines/4ch

# Tests (every C library/pipeline target takes SIMD=1 (default) / SIMD=0 (forced scalar, for
# NEON-vs-scalar A/B parity) and WERROR=1 (promote warnings to errors, the closest thing to lint)):
make -C pipelines test
make -C pipelines SIMD=0 test
make -C pipelines/4ch test
make -C AINR test               # also: test-simd (NEON+scalar in one run), test-pairs (all 15 DFN lookahead pairs)
make -C AINR/RNNoise-ERB test
make -C AIAEC test

# A single test binary, once built, runs standalone (binaries land in a config-keyed directory):
BIN="$(make -s -C pipelines print-bin-dir)"; ls "$BIN"

# Python side (training/export models, and the C<->Python parity suites) — run from repo root:
python3 -m pytest -q AINR/tests/test_dfn2_prepost_c.py           # a single file
python3 -m pytest -q AINR/tests/test_dfn2_prepost_c.py -k warmup # a single test by name
python3 -m pytest -q pipelines/tests/test_doc_layout_versions.py pipelines/tests/test_c_python_map.py
```

`make -C pipelines publish` re-checks pins before building any release artifact;
`check-pins` is the quick standalone preflight. `print-lib-path` (`AINR/`, `AIAEC/`, `pipelines/` and the `stream/` SDK Makefiles except `stream/common`; not `AINR/RNNoise-ERB`) and, on
`pipelines/` and `pipelines/4ch`, `print-bin-dir` / `print-obj-dir` print the config-keyed build
output; `AINR/` and `AIAEC/` tests build into `build/simd<SIMD>-<hash>/`.

## Architecture that spans multiple files

**The model file pattern.** The roles are `*_process.{c,h}` (feature normalization,
mask/deep-filter math and the per-stream model state: normaliser EMAs, rings, frame clock), `*_model_io.{c,h}` (tensor
windowing/batching and recurrent-state registration) and `*_prepost.{c,h}` (the frequency-domain
pre/post processing). For DFN2, Align-ULCNet and DeepVQE-S the lib is per-frame functions plus the
model state, and the per-call lifecycle lives in the model's stream; DFN3, GTCRN and RNNoise-ERB have
no stream layer and keep their per-hop lifecycle classes (see the per-hop contract below). Each model
has the files it needs:

| Model | Files under `lib/ainr/` / `lib/aiaec/` |
|---|---|
| DFN2 | `dfn2_grid.h`, `dfn2_process`, `dfn2_model_io`, `dfn2_prepost` |
| DFN3 | `dfn3_process`, `dfn3_model_io`, `dfn3_prepost` |
| Align-ULCNet | `ulcnet_model_io`, `ulcnet_prepost` |
| DeepVQE-S | `deepvqe_process`, `deepvqe_prepost` (no `_model_io`: the boundary contract, `DEEPVQE_PREPOST_LAYOUT_VERSION` and the descriptor are in `deepvqe_prepost.h`) |
| GTCRN | `gtcrn_process` only |
| RNNoise-ERB | `rnnoise_erb/process` only |

GTCRN/RNNoise-ERB use their own `*_process_mask_validate`/`*_process_state_validate` /
`*_frame_finish` names instead of `frame_open`/`frame_finish`/`frame_bypass`, see below. Reading
just the model's `*_prepost.h` (or `*_process.h` for GTCRN/RNNoise-ERB) is usually enough to
integrate it; `*_process.c` is only needed to change the model's math itself. All 6 models' cores
now live under `lib/ainr/<model>/` / `lib/aiaec/<model>/` (the single-model-release restructuring);
`AINR/`/`AIAEC/` keep the Python training/export side only.

**Vocabulary.** A lib `<model>_model_io` is the CPU-side graph-boundary adapter (descriptor, feature windows, state registration); a stream `graph_io` is the per-(model, grid, N) graph port table of the exported graph (`stream/*/graph_io_*.h`). `wiring` stays only for the `stream_bind` `StreamWiring` struct; `stage` only for units inside pipelines and AEC's `aec_stage_*`; `adapter` = callback adapter, `bridge` = sample-rate bridge, `front` = AEC front end.
`LAYOUT_VERSION` has two meanings: a lib's tensor/carve contract (`DFN2_MODEL_IO_LAYOUT_VERSION_*`, `ULCNET_MODEL_IO_LAYOUT_VERSION`, `DEEPVQE_PREPOST_LAYOUT_VERSION`, `AEC_LAYOUT_VERSION`) and the `layout_version` of a `graph_io` row, which must equal the lib's.

**PCM stream SDKs.** Each wraps a frequency-domain core with analysis STFT and synthesis ISTFT (a `freq_only` context, which only the `_freq` entries use, holds neither). The caller owns the accelerator. The run status of `stream/common/stream_run.h` goes back to postprocess: `STREAM_RUN_OK` (0) = every head written, `STREAM_RUN_NOT_STARTED` = the device state was not touched, any other value = it may be dirty. The library advances the state generation whenever the device state may no longer be the one to continue from; before a run the caller zeroes the accelerator's state (quantized zeros) when the generation differs from its value at the last zeroing.

| SDK | Integration contract |
|---|---|
| `stream/ainr/dfn2` | One int16 input, optional HPF, `AC_OLA_ACCUMULATE`, `frames_per_call` 1..64 (`DFN2HopIO.action`, a `StreamWorkAction`, says which hop runs the accelerator: `STREAM_WORK_SUBMIT` or `STREAM_WORK_BYPASS`). Full and mask-only use variant-specific public symbols. Post reports spectral validity. At `frames_per_call` 1 the first `mask_lookahead` hops after init or reset carry no job (`STREAM_WORK_BYPASS`; the product returns an all-zero `StreamWork` with `STREAM_WORK_BYPASS`); at N > 1 every block is a job at any `mask_lookahead`. The customer entry is `stream/ainr/ainr.h` (`ainr_init(cfg, pool, bytes, &ctx, &info)`, `AinrInfo`, `StreamWork`). |
| `stream/aiaec/deepvqe_s` | Mic and far int16 inputs, `AC_OLA_ACCUMULATE`, `frames_per_call` N 1..64, used by every registered grid alike (the 512 grid's first-hop PCM differs from its pre-batching history as a result). One call carries N frames through `deepvqe_stream_{pre,post}_process_packet` (and `_freq`); the one-hop entries are the N == 1 form and return -1 at N > 1. The PCM delay is one hop at every N, with no pre-roll. A job that did not succeed contributes a zero spectrum (fail-closed); every state-generation advance also empties the CCM ring. The library (`lib/aiaec/deepvqe_s/deepvqe_prepost.{c,h}`) has per-frame functions only (`pre_frame`, `views`, `validate_run`, `post_frame`, `zero_state`) and no per-call latch; `deepvqe_stream.{c,h}` owns the sequence (guards, the raw mic spectra kept from pre to post, the all-N-frames-then-synthesis order, `zero_state` on a dirty run) and refuses a non-finite `_freq` bin with `DEEPVQE_STREAM_ERR_RANGE` (-2) before it moves. |
| `stream/aiaec` (`aiaec.h`) | The customer-facing entry over DeepVQE-S and Align-ULCNet (PCM and `_freq` families; the `_spec` family is retired, use `AcStft` + `_freq`; registered grids for both models: 16kHz/256 N = 1 and 8, 16kHz/512 N = 1, 48kHz/1024 N = 1 and 2): `aiaec_init(cfg, pool, bytes, &ctx, &info)` returns 0 or a negative `AIAEC_ERR_*` and fills the slim `AiaecInfo` (`graph_path` only; N is `cfg.hop_in / (cfg.fft / 2)`, computed by the caller). For both models at every N the output trails the input by one model hop (`fft / 2` samples); the library trims and flushes nothing. Preprocess fills a `StreamWork`, postprocess takes the run status. NaN/Inf input (and, on the Align-ULCNet float PCM entry, a sample beyond 2^112) is refused before any state change: the model stream returns its `*_STREAM_ERR_RANGE` (-2) and the product maps it to `AIAEC_ERR_NONFINITE_INPUT`; the product keeps no scan of its own. |
| `stream/aiaec/ulcnet` | Feed the linear AEC's float residual and raw far to `ulcnet_stream_pre_process_f32()`. The stream owns the call lifecycle and the input range check: a NaN/Inf spectrum bin (PCM entries: a NaN/Inf sample or one beyond 2^112) returns `ULCNET_STREAM_ERR_RANGE` (-2) before any state moves. Reprime after an alignment change, and pass the error spectrum through on bypass. An alignment event restarts only the model (in a packet stream, `frames_per_call` N >= 2, the graph does it before the event hop); no entry restarts the stream's own analysis or synthesis, so the PCM entries give the samples of the `_freq` entries around an `AcStft` / `AcIstft`, event hops included, and N = 8 PCM equals the N = 1 stream. |

**Customer headers.** A board sample includes only `stream/ainr/ainr.h`, `stream/ainr/bridge/ainr_bridge.h`, `stream/aiaec/aiaec.h`, `stream/aiaec/front/aiaec_front.h`, `stream/common/{stream_mem,stream_bind,stream_run,graph_io}.h` and the three `graph_io_*.h` (`docs/maintenance/models.md` §13). Include roots: each header's directory, `stream/common` (the lib directories include no stream header; `stream/common` is on the path for the stream `.c` files and the tests), audiolib `common/` (for `utility/mem_align.h`) and, with `ainr_bridge.h`, audiolib `common/include`. `make -C stream/common check-public-headers` compiles the set as C99 `-Wpedantic` and C++11 with only those roots and checks C linkage.

`ainr.h` / `aiaec.h` report no graph ports: `AinrInfo` carries the selected `graph_path` and `queue_frames` (the output trails the input by `queue_frames + 1` model hops), `AiaecInfo` only `graph_path`, and a `StreamWork` (`stream/common/stream_run.h`, shared by both) holds the block's signal and head tensors. The graph wiring (signal, head and state shapes per registered `(model, sr, fft, N)`) is the board sample's model configuration: the per-graph tables `graph_io_<model>.h` next to each library (`stream/ainr/graph_io_dfn2.h`, `stream/aiaec/graph_io_{deepvqes,alignulcnet}.h`; header-only, not included by `ainr.h` / `aiaec.h`) and the shared helper `stream/common/graph_io.h` (tested by `make -C stream/common test-graph-io`); samples add `-I stream/common -I stream/ainr|stream/aiaec`. The SDK-free checker `StreamWiring` / `StreamDeviceDesc` / `stream_check_ports()` (`stream/common/stream_bind.{h,c}`) is compiled by samples only, and `StreamMemReq` lives in `stream/common/stream_mem.h`. A board sample binds by index, never by name: inputs = signal inputs, then state inputs; outputs = head outputs, then state outputs; state k is input `signal_in_count + k` and output `head_out_count + k`. Align-ULCNet at N > 1 has `frame_reset` as the last signal input; the stream fills it as the last input of the `StreamWork` and the sample moves it like any other signal. The sample checks every block's work against its row before the accelerator call (a mismatch means no run) and the opened graph's ports once after opening; fixed-point state ports need `zero_point` 0 at both ends. Float I/O is a compile-time choice of the AI3 samples (`FLOAT_IN` / `FLOAT_OUT`, default 0; the model stays fixed point and the hardware converts at the boundary). These models inherit the state port types, so set both alike (a mixed setting only warns); the samples print the reported port types at start-up for the board A/B (`docs/maintenance/float-io-board-ab.md`). `docs/maintenance/graph-wiring.md` has the details and the pins.

DFN3, GTCRN and RNNoise-ERB expose frequency-domain cores. CAGCRN and Align-CRUSE have no C code (Python model and ONNX export only; the application does the mask multiply). At the spectrum level, desktop pipelines use `stream/ainr/dfn2/dfn2_stream.h` (`_freq`) and `stream/aiaec/ulcnet/ulcnet_stream.h` (`_freq`); board samples use `stream/ainr/ainr.h` (`ainr_*_freq`), with `stream/ainr/bridge/ainr_bridge.h` for an 8/16 kHz host. `pipelines/dfn/dfn_hop.h` is the hop glue over `stream/ainr/dfn2` and `stream/ainr/bridge` (example code, no ABI). The [model manual](docs/html/models/index.html) contains the per-hop examples.

**Layering and per-hop contract.** DFN2, Align-ULCNet and DeepVQE-S: the lib holds per-frame functions and the
model state (features and their normalisation, the model I/O window, lookahead rings, per-frame
post-processing, output and state validation, zero state with the generation, reset); it has no
per-call latch and includes no stream header. The stream holds the per-call lifecycle (submit or
bypass, the tensor views handed out, finish / fail / bypass sequencing), the STFT / ISTFT, N > 1 as N
per-frame lib calls around one accelerator call, and the input range checks (`<M>_STREAM_ERR_RANGE`,
-2, through `stream/common/stream_range.h`; the product maps it to `*_ERR_NONFINITE_INPUT`). The lib
verbs are the same in all three: `pre_frame` (N calls, one per frame), `views` (once per accelerator
run, after the last `pre_frame`: NaN-fills the lib-owned heads and publishes the tensor views),
`validate_run` (0 accepted; -1 refused: the registered state is zeroed and the generation advances once,
inside the lib), `post_frame` per frame (DFN2 also `post_frame_identity`), `zero_state`. The stream
calls `zero_state` once for a run status other than `STREAM_RUN_OK` and `STREAM_RUN_NOT_STARTED` (and
for a -2 from `post_frame`), never after a refused `validate_run`. A failed or refused call gives the
identity frames (DFN2: `post_frame_identity` per frame; DeepVQE-S: zero spectra; Align-ULCNet: the
error spectrum), N of them. Missing a stream post before the next pre is a caller bug: the next call returns -1.
DFN2: with `frames_per_call` 1 the first `mask_lookahead` hops after init/reset have no job
(`DFN2HopIO.action` = `STREAM_WORK_BYPASS`, the generation does not move, the output is zero spectrum with
valid 0); with N > 1 every block is a job at any `mask_lookahead` (the heads are frames
T - N + 1 - M .. T - M, the first block's first M heads belong to frames before the start and are
discarded, silence), latency L + D, and the per-hop entries return -1 at N > 1. Align-ULCNet and DeepVQE-S
have no such hops (DeepVQE-S no pre-roll at any N); Align-ULCNet has identity hops after an alignment
boundary (`ulcnet_stream.h`: no `pre_frame`, no `views`, the hop is the error spectrum and `ai_rc` is not read).

DFN3, GTCRN and RNNoise-ERB have no stream layer and keep the earlier class (the documented exception, unchanged):
DFN3 `pre_process_freq()` → `frame_open()` → accelerator fills the tensors → `frame_finish()` (`frame_bypass()`
after a commit `frame_finish()` refused or on any failure) → `post_process_freq()` → push to `AcIstft`;
missing a `frame_finish`/`frame_bypass` before the next `pre_process`/`post_process` call is a caller bug, the
next call returns -1. DFN3's first `mask_lookahead` hops after init/reset have no complete graph window, so
`pre_process_freq` returns 0, no frame is opened, no accelerator job runs and the registered state and the
generation do not move; the post-processed spectrum is zero with valid 0. GTCRN/RNNoise-ERB have no
`mask_lookahead`/warm-up concept at all: a callback failure is instead caught by their
`*_mask_validate`/`*_frame_finish` finite-check, which (if the NaN-filled output sentinel the
caller's own code must set before the call stays NaN) zeroes the registered state and advances the
generation on that same hop, regardless of the callback's own return code.

**State ownership and the generation counter.** Recurrent state is either registered by the host
(pointer+count pairs, all-or-none) and owned in caller memory, or left to a hardware accelerator
that tracks its own quantized copy. A `*_state_generation()` counter advances on every reset, every refused `validate_run` (or DFN3 / GTCRN / RNNoise-ERB commit) and every `zero_state` after a dirty run (the stream does this for a run
status other than `STREAM_RUN_OK` and `STREAM_RUN_NOT_STARTED`); an Align-ULCNet alignment restart
(`frames_per_call` N = 1) advances it too; in a packet the graph restarts the model before the event
hop and the generation stays, unless the run was `STREAM_RUN_NOT_STARTED`, which advances it once. Any accelerator-side state must be reloaded with quantized zero whenever the generation
differs from its value at the last zeroing (not just at init/reset) or it will carry stale state
into the next real frame; a board sample keeps no failure flag of its own. An accelerator call that fails is a `zero_state` by the stream (state zeroed, generation advances once); one that returns `STREAM_RUN_NOT_STARTED` leaves state and generation alone (the Align-ULCNet event packet above excepted).

**Caller-owned memory.** `*_grid_make(spec)` → `*_get_mem_size(cfg)` → caller allocates one block
→ `*_init`/`*_init_ex`. No internal `malloc`, except the optional `*_create` / `*_destroy` helpers documented in each header (`*_prepost_create`, `ainr_bridge_create`, `aec_create`, ...), which allocate the pool with `aligned_alloc` or `posix_memalign` and free only what they allocated. A `CARVE_VERSION`/layout-version constant and a
descriptor hash must bump together whenever a struct's carved layout changes size — both are
checked by `pipelines/tests/test_doc_layout_versions.py` and guard against a stale caller pool
being handed to a newer library build.

**Frequency domain only inside models.** Framing, windowing, FFT/IFFT belong to the application
via audiolib's `AcStft`/`AcIstft`; model libraries only ever see/produce spectra. The model streams'
PCM entries run that STFT/ISTFT for the caller, and `ainr_bridge` (an 8/16 kHz host) resamples; a
caller that owns its analysis uses the `_freq` entries. The `_freq` entries (`dfn2_stream`, `ulcnet_stream`, `deepvqe_stream`, `ainr_*_freq`,
`aiaec_*_freq`) take and return spectra.

**The 4-channel flow is an `Aec` array instance (`AecConfig.n_mics == AEC_ARRAY_MICS`).** One
`aec_init` builds the four lanes, the shared delay estimator, the aligned far ring and the fusion /
gain stage (`src/aec_array.c` in the AEC repo). The per-hop
shape is `aec_process_linear` → `aec_get_array` (lane error spectra for the caller's SRP/BF) →
`aec_process_residual(&AecBeam)` → `aec_get_residual` → `aec_apply_gain` / `aec_apply_cng`. The
instance exposes no beamformer: the caller supplies the BF weights and spectrum. The `array_sample_*`
files under `pipelines/4ch/` are sample-side helpers (init in one place, SRP/BF), not a library.

**Python <-> C parity is enforced, not aspirational.** Changes to model math are expected
Python-first; the C mirror is then checked bit-exact (or against a frozen, previously-measured
tolerance) by dedicated parity suites (`pipelines/tests/pyc_parity/`,
`AINR/tests/test_python_c_prepost_parity.py`, and friends) that hash the sources they judged into
their result JSON. `docs/c_python_function_map.md` plus
`pipelines/tests/test_c_python_map.py` keep the C/Python symbol mapping and function counts in
sync — update the map in the same change as adding/removing a C function with a Python twin.

**Documentation.** The public entry point is `docs/html/index.html`. Group integration procedures under the corresponding model or pipeline, with detailed contracts in its reference directory. `docs/README.md` links the Markdown integration references. `docs/maintenance/README.md` covers the
submodule/dependency workflow in more depth than the section above.

## Working across this repo's tree

Because AEC/NR/audiolib are developed in sibling directories (`../AEC`, `../NR`, `../audiolib`)
rather than inside this repo, a change that spans a model library and its conventional-algorithm
counterpart touches two separate git repositories. Commit and push inside the dependency's own
repo first, then bump its pin (`lib/aec.pin`'s `AEC_COMMIT`, etc.) in the same Audio_ALG commit —
never land a pin bump without the commit it points to already existing upstream.
