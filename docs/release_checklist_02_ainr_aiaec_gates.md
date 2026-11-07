[Index](release_checklist.md) | [Previous](release_checklist_01_pipeline_gates.md) | [Next](release_checklist_03_docs_publish_appendices.md) <!-- split-nav -->
## 11. AINR framework gate

Public AINR models are RNNoise-ERB, GTCRN, DeepFilterNet2, and DeepFilterNet3. Full training
and trained-model quality are out of scope.

### 11.1 Dataset generator

Run small real end-to-end jobs separately at 16 kHz and 48 kHz:

- [ ] generate;
- [ ] interrupted-run resume;
- [ ] refusal of a plain re-run into a non-empty output;
- [ ] index-gap rejection and temporary-WAV cleanup;
- [ ] pack;
- [ ] mmap load;
- [ ] WAV-only layout (`NNNNNN.wav`, leftover JSON ignored);
- [ ] post-resample `effective_rms_dbfs` validation;
- [ ] train/validation clipping contract;
- [ ] rejection of mixed sample-rate directories;
- [ ] packed sample-index validation.

The generator intentionally does not persist config, seed, or per-sample JSON.
Resume therefore cannot detect a changed generation distribution; that is a
documented WAV-only trade-off, not a contract-mismatch gate. Use a fresh output
directory when config or sample rate changes.

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AINR/dataset_gen/tests AINR/tests
```

### 11.2 Training-framework smoke

For every public model, use a tiny packed dataset and run at least 10--20
optimizer steps. Validate:

- [ ] CPU smoke;
- [ ] `--packed-dir`;
- [ ] `--gpu` argument and device selection;
- [ ] `--mmap`;
- [ ] progress reporting;
- [ ] checkpoint save/resume;
- [ ] dataset/grid/model-contract mismatch rejection;
- [ ] EMA and scheduler resume;
- [ ] NaN/Inf halt;
- [ ] validation does not update model state;
- [ ] deterministic seed and split behavior.

A short CUDA smoke may be executed on the training machine. A long training
run is not required.

### 11.3 Reference inference

A legal checkpoint created from a randomly initialized model is sufficient.
For each model verify:

- [ ] checkpoint load;
- [ ] WAV input/output;
- [ ] finite output;
- [ ] sample-rate/grid mismatch rejection;
- [ ] streaming state and reset;
- [ ] lookahead/warm-up frame indexing;
- [ ] every advertised export path;
- [ ] Python/C feature parity;
- [ ] C pre/post SIMD/scalar parity;
- [ ] accelerator-boundary tensor shape, layout, normalization, and latency
      documentation.

Do not claim trained-model quality without a trained checkpoint.

### 11.4 C pre/post archive

`AINR/Makefile` builds `libainr_prepost.a` — `dfn2_process`, `dfn2_model_io`,
`dfn2_prepost`, `dfn3_process`, `dfn3_model_io`, `dfn3_prepost`, `gtcrn_process` and
RNNoise-ERB's `process.c` (member `rnnoise_process.o`; RNNoise-ERB's own Makefile builds its
tests). The members call audiolib, so a link of the archive takes audiolib's archive too.

```bash
make -C AINR SIMD=1 WERROR=1 lib
make -C AINR SIMD=1 WERROR=1 print-lib-path   # absolute path of THIS config
make -C AINR SIMD=0 WERROR=1 test
make -C AINR test-pairs   # DeepFilterNet2/3 lookahead pairs other than the shipped 1/1
make -C AINR/RNNoise-ERB test   # ERB table, feature contract, features, framing contract, SIMD/scalar
```

- [ ] Objects, test binaries and the archive are configuration-keyed
      (`build/simd<N>-<sig>/`), so a `SIMD=0` build never hands back the
      `SIMD=1` archive; resolve the archive with `print-lib-path`, never by
      spelling the directory.
- [ ] DeepFilterNet2 and DeepFilterNet3 lookahead pairs (`mask_lookahead` 0..2,
      `df_lookahead` 0..4, 15 pairs, shipped 1/1; the pair is a field of the
      run-time spec): the Python parts pass for all pairs,
      `tests/test_dfn3_prepost_c.py` runs all 15 against one library build, and
      `make -C AINR test-pairs` rebuilds the DFN2 C hosts and the DFN3 stream host
      only for the pairs the tests list (the six pairs with `D <= M`, and (0,1),
      (1,3), (2,4)); a pair outside that list is not claimed as tested in those
      hosts, and the audio quality of any pair other than 1/1 is unmeasured (no
      checkpoint).
- [ ] The recurrent state of the AINR classes (DeepFilterNet2, DeepFilterNet3, GTCRN, RNNoise-ERB) belongs to the application:
      registration is all-or-none (a pointer with a count of 0, or a count with a NULL pointer, is refused), the state
      generation is 1 after init and advances on a reset, on every DeepFilterNet2 / DeepFilterNet3 warm-up frame (the first
      `mask_lookahead` hops, one flow on every hop, heads discarded, audio unchanged) and on a refused finish or validation, and a
      refusal zeroes any registered state (`tests/test_c_prepost.c` for DeepFilterNet2 and GTCRN, `tests/test_dfn3_stream.c` for DeepFilterNet3,
      `RNNoise-ERB/test/test_rnnoise_features.c` for RNNoise-ERB, and `tests/test_dfn2_prepost_c.py::
      test_state_generation_counts_every_zeroing_and_nothing_else`) pass.
- [ ] The DeepFilterNet2 grid is a run-time value (`DFN2GridSpec` -> `dfn2_grid_make()`; the library holds no grid, preset or
      default, and the application owns the (sample rate, FFT size) table): `tests/test_dfn2_grid.py` (every refusal in check
      order, `make` agreeing with `validate`, the `n_erb` overflow cases under UBSan, `dfn2_norm_alpha()` equal to
      `make_norm_alpha` of `train.py`) and `tests/test_dfn2_runtime_grid.py` (the whole chain on every grid and pair, pool bytes
      and hash equal to an independent Python layout mirror, a foreign pool refused, the variant link symbol) pass;
      `tests/test_dfn2_lookahead_pairs.py` builds the C suites on the 48, 32 and 16 kHz test grids for the shipped pair and 2/4;
      a pool or `MemReq` recorded under another carve version (`DFN2_PREPOST_CARVE_VERSION` = 11) is refused. DeepFilterNet3
      is the same (`DFN3GridSpec` -> `dfn3_grid_make()`; its pair refusals are in `tests/test_dfn3_stream_c.py`).
- [ ] DeepFilterNet2 frames per call (`export_onnx.py --frames-per-call N`, N in 1..64; the C side takes it as
      `DFN2GridSpec.frames_per_call`): N = 1 is the one-frame graph byte for byte; for N > 1 the chunked stream equals
      the one-frame stream (`tests/test_dfn2_frames_per_call.py`, and `tests/test_dfn2_frames_per_call_c.py` for the C host,
      both in `make -C AINR test-pairs`); `make test` and the pipelines stay at N = 1 (the pipeline stage and rate bridge
      refuse any other value). The accelerator toolchain's support for a GRU with a sequence longer than one and the
      quantized accuracy of an N-frame graph are not measured.
- [ ] No model library (AINR, AIAEC) holds framing, FFT or window code and no Makefile passes a dead `-D` flag:
      `make -C pipelines audit-no-transform` and `make -C pipelines audit-dead-defines` pass.
- [ ] `WERROR=1` actually reaches the compiler.
- [ ] The SIMD knob changes every member that includes `utility/simd_kernel_nn.h`
      (dfn2/dfn3 `process`, `model_io`, `prepost`, `gtcrn_process` and `rnnoise_process`:
      all eight members); never mix a `SIMD=0` archive with a `SIMD=1`
      application, and resolve the archive with `print-lib-path`.

## 12. AIAEC framework gate

The current public candidates are Align-CRUSE, Align-ULCNet, DeepVQE-S, and
CAGCRN. A candidate that is not intended to
be runnable must be clearly marked experimental or removed from the public
surface.

### 12.1 Dataset generator

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AIAEC/dataset_gen/tests
```

Pass conditions:

- [ ] Five-stem contract passes.
- [ ] `linear_error` is not output-limiter processed.
- [ ] Complete stateful PBFDKF scenarios are rendered before chunking.
- [ ] Behavior hash is stable across supported Python versions.
- [ ] Source hash and behavior hash retain their separate meanings.
- [ ] Final WAV writes are atomic; leftover JSON is neither required nor read.
- [ ] Plain re-run, index gaps, and conflicting resume start indices fail
      closed.
- [ ] Temporary-WAV cleanup, pack, and mmap load pass.
- [ ] Every public model view maps stems consistently.
- [ ] 16/48 kHz model-grid validation passes.
- [ ] Packed four-stem version and PBFDKF behavior-hash compatibility are
      rejected without silent migration.

### 12.2 Training-architecture smoke

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AIAEC/tests
```

For every public candidate, run a tiny packed dataset for 10--20 optimizer
steps and verify:

- [ ] `train.py --packed-dir ... --gpu ... --mmap`;
- [ ] finite loss;
- [ ] checkpoint save/resume;
- [ ] dataset fingerprint and split indices are stored;
- [ ] PBFDKF behavior contract is stored;
- [ ] contract mismatch fails closed;
- [ ] EMA, scheduler, and non-finite halt behavior;
- [ ] validation does not update training state.

### 12.3 Reference inference

For every public candidate verify:

- [ ] `inference.py` loads a legal random checkpoint.
- [ ] Mic/far or linear-error/far input mapping is correct.
- [ ] Feature and waveform shapes are correct.
- [ ] Supported 16/48 kHz grids match the model README.
- [ ] Offline/streaming behavior, state, reset, cache, and lookahead are
      explicit.
- [ ] Hardware inference tensor layout, normalization, output meaning, and
      pre/post-processing are documented.

No production quality claim is required or permitted without trained models.

### 12.4 C pre/post archive

`AIAEC/Makefile` builds `libaiaec_prepost.a` — `ulcnet_process`,
`ulcnet_model_io`, `ulcnet_prepost`, `ulcnet_accelerator_adapter`,
`aiaec_process`, `deepvqe_process`, `deepvqe_prepost`.

```bash
make -C AIAEC SIMD=1 WERROR=1 lib
make -C AIAEC SIMD=1 WERROR=1 print-lib-path   # absolute path of THIS config
make -C AIAEC SIMD=0 WERROR=1 test             # lib + pytest
```

- [ ] The archive is configuration-keyed (`build/simd<N>-<sig>/`, keyed on
      `CC`/`AR`/`CFLAGS`/`CPPFLAGS`/`SIMD`/`WERROR`) and a signature collision
      is refused rather than reinterpreted; resolve it with `print-lib-path`.
- [ ] The SIMD knob changes every member that includes `utility/simd_kernel_nn.h`
      (`ulcnet_process`, `ulcnet_model_io`, `deepvqe_process`,
      `deepvqe_prepost`); never mix a `SIMD=0` archive with a `SIMD=1`
      application, and resolve the archive with `print-lib-path`.
- [ ] Each pre/post class (`ulcnet_prepost`, `deepvqe_prepost`) is a translation
      unit of its own that uses `ulcnet_model_io` / `deepvqe_process`, which stay
      independently linkable for their own parity tests, and its byte-identity
      tests against the hand-composed path pass.
- [ ] The grid is a run-time value (`UlcnetGridSpec` -> `ulcnet_grid_make()`,
      `DeepVqeGridSpec` -> `deepvqe_grid_make()`): both classes run 16000/512 and
      48000/1024 from one build (`test_class_matches_the_hand_composed_path` with `grid=16000,512` and `grid=48000,1024`),
      a pool recorded for another grid or delay depth is refused by `init_ex`
      (`test_stale_pool_gate_holds_and_the_pool_follows_the_grid`), and a refused
      `frame_finish` advances the state generation and zeroes any registered state
      (DeepVQE-S: `test_state_generation_counts_restarts`; Align-ULCNet: `test_reset_clears_state` and
      `test_frame_state_machine_guards`; 0 after init for both classes).

[Index](release_checklist.md) | [Previous](release_checklist_01_pipeline_gates.md) | [Next](release_checklist_03_docs_publish_appendices.md) <!-- split-nav -->
