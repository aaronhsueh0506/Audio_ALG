# Changelog

## Unreleased

### DFN2 / DFN3 lookahead pairs: 15 legal pairs, exported and run

- Legal pairs are `mask_lookahead` in 0..2 and `df_lookahead` in
  0..`df_order`-1, independent of each other (15 pairs); the former rule
  `df_lookahead <= mask_lookahead` is gone from the models, the C range checks
  and the tests. A checkpoint trained for one pair is still refused under
  another, with one shared instruction (`onnx_streaming_contract`).
- The graph, tensor names, shapes, state layouts (DFN2 8, DFN3 9) and the shipped
  1/1 graph are unchanged; `mask_lookahead` only selects the host window
  `[t-(2-M) .. t+M]`. `feature_windows()` takes the lookahead (no default) and
  validates it when called; the export metadata gains `mask_lookahead`,
  `df_lookahead` and a derived `input_window_alignment` (unchanged string for 1);
  the calibration reports (`inference.py calib`, `inference_batch.py calib`) name
  the pair.
- C: `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` and the DFN3 twins are build
  constants (defaults 1/1) range-checked by preprocessor `#error` in every
  language mode; objects built for different pairs fail to link (a one-byte
  symbol named after the pair, no code or allocation); model-io readiness is
  `MASK + 1` frames. DFN2 (cascade) delays `M + D` frames with a ring of
  `max(DF_ORDER, M + D + 1)`; DFN3 (parallel) delays `max(M, D)` frames with a
  ring of `DF_ORDER + max(M, D) - D`, and keeps the heads in the ring when
  `D > M` (`DFN3_HEAD_QUEUE`). For the shipped pair the code, state sizes, pool
  hash and test digests are unchanged. `dfn3_compose()` documents its
  same-frame head pairing.
- Tests: `tests/dfn_pairs.py` is the one source of the pair set (probed from the
  model constructors), the shipped pair (`config.ini`) and the compile flags
  (`make print-cflags`, `make print-sources`); `make test-pairs` runs the
  per-pair pytest files. New: `test_dfn_lookahead_link.py` (link and compile
  guards), `test_dfn_lookahead_calibration.py`, `test_dfn_lookahead_contract.py`,
  `test_dfn2_lookahead_pairs.py`; the DFN3 stream driver gained a pool-identity
  check across pairs. C builds run for the six pairs with `D <= M` plus (0,1),
  (1,3), (2,4).

### Known gap (AIAEC)
- `AIAEC/dataset_gen/linear_aec.py` registers the AEC Python behaviour-hash step 5fe64e46 -> 428aa9b7 as a
  compatible migration without a measurement that the frozen front end is unchanged over that step; a mismatch only
  warns. Re-materialised corpora use the current hash.

### Source comments condensed

- Comments and docstrings in the AINR/AIAEC model C/H files (dfn2, dfn3, gtcrn,
  RNNoise-ERB `process.*`, ulcnet, deepvqe, aiaec) and their Python model and
  streaming modules were reduced to contract text (C/H comment lines 2,998 ->
  1,700; Python 1,283 -> 963). Code, identifiers, literals and generated object
  code are unchanged. The records below were removed from the sources.
- DFN2/DFN3 `model.py` upstream alignment: `df_dec.df_gru` takes the module
  default `linear_groups=8` (checkpoint weight `(8, 64, 32)`; 16 breaks loading);
  the `lsnr` head is omitted (513 parameters, loss term not trained), so a
  streaming port cannot skip the ERB/DF stages on clean frames; DFN3's 257-parameter
  `df_fc_a` is kept for the 2,135,484 reconciliation; the released DFN3
  `convt_depthwise` key is never read upstream; the DF decoder guard
  (`emb_in_dim == df_hidden`) intentionally differs from upstream's.
- DFN2/DFN3 `process.h`: the grid is 1024/512 (DF cutoff 4500 Hz instead of
  4800 Hz at 960/480); `ERB_NORM_ALPHA` is rounded to 3 decimals (effective tau
  0.9644 s); `ERB_NORM_SCALE_DB = |log floor - init_lo|`; `ANALYSIS_SCALE`
  is the libDF `wnorm` residual 1/32 and acts only on the feature branch;
  `SPEC_NORM_EPS` stays in C although the trainer dropped it (denormal flush);
  `torch.istft` rejects `center=False` at 1024/512, and the causal EMA carries a
  first-frame difference through a whole 3 s segment (3 tau = 273 of 281 frames).
- Latency figures: DFN2 identity path 3 hops (1536 samples, 32 ms at 48 kHz);
  DFN3 2 hops (1024 samples, 21.33 ms).
- RNNoise-ERB normaliser init values come from this project's 16 kHz corpus
  (libDF's -60/-90 dB would start the EMA 35..48 dB off for the triangular bank;
  0.001 against the measured 0.008161 would put the top bin 2.86x out).
- ULCNet/DeepVQE-S `prepost`: analysis is the rolling `center=False` schedule;
  centered frame k and rolling frame k-1 cover the same samples and window (bit
  identical when measured), and algorithmic latency stays one hop (synthesis
  half-window trim). Each `prepost.c` is a separate translation unit so that the
  process and model-io files stay independently linkable.
- DeepVQE-S `frame_skip()` fails closed (silence): its stream 0 is the raw
  microphone, so pass-through would emit the full echo; the cost is a one-frame
  notch. The ULCNet adapter cannot verify that the descriptor `delay_depth`
  equals the exported depth; there is no exporter-generated descriptor header yet.
- ULCNet model-io layout versions 3-17 and DeepVQE-S 1-3 are retired; numbers
  are never reused.

### Removed (2026-10-23)

- Dead C code with no caller in any supported configuration: `dfn2_post_filter`
  and `dfn3_post_filter` (with their `df_common_post_filter` helpers, the
  `DFN*_MASK_PF` / `DFN*_PF_BETA` macros and the guarded call sites; the macro
  was hard-defined to 0, so the post-filter was never compiled), the empty
  `rnnoise_tables_init`, and the unused `dfn3_model_io_init`. The Python
  `mask_pf` option and the shipped output are unchanged.

### Changed (2026-10-22)

- DFN2/DFN3 memory descriptor hashes distinguish the 32-bit target ABI while
  preserving 64-bit hashes and layouts. Pool-walk locals initialize explicitly
  for GCC ARM32; strict C11 test declarations and array qualifiers are portable.
- ARM32 model code selects scalar twins automatically; shared FFT dispatch is
  independent. Model geometry, state, presets and latency are unchanged.

### Added

- Restored the independent DeepFilterNet3 band-split comparison branch from
  commit `c0ddbf84aac9d7dbd66d5627690a6c03a68a293a`, adapted to current shared
  training, calibration, ONNX and audiolib C interfaces. DFN2 remains available.

### Changed (2026-10-21)

- RNNoise-ERB 的 C 前後處理改用呼叫端持有的共用 FFT／STFT／ISTFT；完整
  512-sample frame、512/512/256 格點、縮放與 gain 乘法順序、首 hop 和
  overlap-add 行為維持不變。前後處理 layout 為 2，GRU／ONNX state layout 仍為 1。
- 新增 framing 尺寸、窗、縮放與模式的拒絕檢查、獨立建立的測試物件、頻譜修改／
  延遲／重置對照，以及測試 header 依賴。文件明確說明模型窗逐位元要求與 NULL 行為。
- 本機獨立 static memory 由 15,328 增至 31,136 bytes；固定 frame 的前後處理
  （不含網路推論）在 SIMD=0／1 下分別加速約 2.12／3.02 倍。FFT 捨入差異已對照
  舊實作與 Python 量測；相同頻譜的特徵處理保持逐位元相同。尚無本地已訓練權重，
  感知品質未量測，固定係數網路測試僅作數值整合驗證。

### Restoration validation (2026-10-20)

- Integrated Python suite: **220 passed**, running
  `python -m pytest AINR/DeepFilterNet2/tests AINR/DeepFilterNet3/tests AINR/tests -q`
  from the repository root.
- `make -C AINR WERROR=1 lib test-simd` and
  `make -C AINR SIMD=0 WERROR=1 test` passed. The DFN2/GTCRN output
  digests of that date were unchanged; DFN3 scalar/NEON stream digest was `5b831894`. The frozen-digest
  checks were removed on 2026-10-23: the current tests check reproducibility and NEON/scalar agreement only.
- A tiny synthetic 48 kHz corpus exercised train, resume, checkpoint load,
  WAV inference, normalizer calibration, graph/calibration export and ERB
  export. Batch-1 / batch-3 ONNXRuntime replay errors were at most
  `2.98e-7` / `2.38e-7` absolute in that smoke run.
- All 17 pre-existing DFN2 source/header/config/documentation files retained
  their recorded SHA256 values. The restored DFN3 has independent names and
  C symbols; the build includes both models.

These checks validate integration and arithmetic, not trained audio quality.
Use accepted production checkpoints or train both models before measuring
noise suppression or speech preservation.
