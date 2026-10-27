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
`dfn2_prepost`, `dfn3_process`, `dfn3_model_io`, `dfn3_prepost` and `gtcrn_process`.
RNNoise-ERB is deliberately NOT a member: it has its own Makefile and no archive.

```bash
make -C AINR SIMD=1 WERROR=1 lib
make -C AINR SIMD=1 WERROR=1 print-lib-path   # absolute path of THIS config
make -C AINR SIMD=0 WERROR=1 test
make -C AINR test-pairs   # DeepFilterNet2/3 lookahead pairs other than the shipped 1/1
```

- [ ] Objects, test binaries and the archive are configuration-keyed
      (`build/simd<N>-<sig>/`), so a `SIMD=0` build never hands back the
      `SIMD=1` archive; resolve the archive with `print-lib-path`, never by
      spelling the directory.
- [ ] DeepFilterNet2 and DeepFilterNet3 lookahead pairs (`mask_lookahead` 0..2,
      `df_lookahead` 0..4, 15 pairs, shipped 1/1): the Python parts pass for all
      pairs, and `make -C AINR test-pairs` rebuilds the C hosts only for the
      pairs the tests list (the six pairs with `D <= M`, and (0,1), (1,3),
      (2,4)); a pair outside that list is not claimed as tested in C, and the
      audio quality of any pair other than 1/1 is unmeasured (no checkpoint).
- [ ] DeepFilterNet2 frames per call (`export_onnx.py --frames-per-call N`, `make -C AINR DFN2_FRAMES_PER_CALL=<N>`, N in
      1..64): N = 1 writes the one-frame graph byte for byte and builds with the same compile lines; for N > 1 the
      chunked stream equals the one-frame stream (`tests/test_dfn2_frames_per_call.py`,
      `tests/test_dfn2_frames_per_call_c.py`, both in `make -C AINR test-pairs`); `make test` and the pipelines stay at
      N = 1 and an N > 1 build of a one-frame unit stops at the `DFN2_MULTI_FRAME_HOST` `#error`. The accelerator
      toolchain's support for a GRU with a sequence longer than one and the quantized accuracy of an N-frame graph are
      not measured.
- [ ] `WERROR=1` actually reaches the compiler.
- [ ] The SIMD knob changes every member that includes `utility/simd_kernel_nn.h`
      (dfn2/dfn3 `process`, `model_io`, `prepost` and `gtcrn_process`: all seven
      members); never mix a `SIMD=0` archive with a `SIMD=1` application, and
      resolve the archive with `print-lib-path`.

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
- [ ] No AIAEC translation unit includes `simd_kernels.h`, so the SIMD knob
      must leave every AIAEC object byte-identical — a difference here means
      something started depending on the kernels and the claim needs redoing.
- [ ] Each pre/post class (`ulcnet_prepost`, `deepvqe_prepost`) is a THIRD
      translation unit composing two files that stay independently linkable
      for their own parity tests, and its byte-identity tests against the
      hand-composed path pass.

[Index](release_checklist.md) | [Previous](release_checklist_01_pipeline_gates.md) | [Next](release_checklist_03_docs_publish_appendices.md) <!-- split-nav -->
