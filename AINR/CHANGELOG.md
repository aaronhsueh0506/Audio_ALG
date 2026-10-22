# Changelog

## Unreleased

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
- 本機獨立 static memory 由 15,328 增至 33,184 bytes；固定 frame 的前後處理
  （不含網路推論）在 SIMD=0／1 下分別加速約 2.12／3.02 倍。FFT 捨入差異已對照
  舊實作與 Python 量測；相同頻譜的特徵處理保持逐位元相同。尚無本地已訓練權重，
  感知品質未量測，固定係數網路測試僅作數值整合驗證。

### Restoration validation (2026-10-20)

- Integrated Python suite: **220 passed**, running
  `python -m pytest AINR/DeepFilterNet2/tests AINR/DeepFilterNet3/tests AINR/tests -q`
  from the repository root.
- `make -C AINR WERROR=1 lib test-simd` and
  `make -C AINR SIMD=0 WERROR=1 test` passed. Existing DFN2/GTCRN golden
  digests stayed unchanged; DFN3 scalar/NEON stream digest was `5b831894`.
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
