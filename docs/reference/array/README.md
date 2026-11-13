# 四麥克風 array

四路同步 mic 共用一路 far。`AecConfig.n_mics = AEC_ARRAY_MICS`（4）時，一次 `aec_init_ex()` 在 `libaec.a` 內建出共用 delay estimator、對齊後的 far、四條 linear AEC lane、融合緩衝與唯一一個 residual 增益級。
每 hop 呼叫 `aec_process_linear()` 取得四路 error 頻譜（`aec_get_array()`），由應用選擇 beam，再以同一組權重呼叫 `aec_process_residual()`，最後用 `aec_get_residual()`、`aec_apply_gain()`、`aec_apply_cng()` 渲染。下游可接傳統 NR、DFN2 或 Align-ULCNet。

| 要接的介面 | 閱讀順序 |
|---|---|
| 自備 beamformer、只取前端或串自有後級 | [初始化與 per-hop 範例](integration.md) → [格點、記憶體與設定](config.md) → [錯誤與 descriptor](errors.md) |
| 使用內附 SRP／BF sample helper | [beam helper 的 C 範例與生命週期](beam.md) → [參數](beam-settings.md) → [錯誤與相容](beam-errors.md) |

array 屬於 `libaec.a`，不需要額外的 archive；`make -C pipelines/4ch libs` 編出 sample 層 helper（`array_sample_beam.o`）。模型的 PCM stream SDK 可以各自使用。[四聲道路由](../../html/pipelines/index.html)
