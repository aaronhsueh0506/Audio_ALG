# 四麥克風 MCAB

四路同步 mic 共用一路 far。應用端提供四條線性 AEC lane，MATCHED 模式再提供一個共享 estimator。每 hop 呼叫 `mcab_process_front()` 取得四路 error 頻譜，由應用選擇 beam，再以同一組權重呼叫 `mcab_fuse()`。下游可接傳統 RES／NR、DFN2 或 Align-ULCNet。

| 要接的介面 | 閱讀順序 |
|---|---|
| 自備 beamformer、只取前端或串自有後級 | [初始化與 per-hop 範例](integration.md) → [格點、記憶體與設定](config.md) → [錯誤與 descriptor](errors.md) |
| 使用內附 SRP／BF sample helper | [beam helper 的 C 範例與生命週期](beam.md) → [參數](beam-settings.md) → [錯誤與相容](beam-errors.md) |

MCAB 由 `make -C pipelines/4ch` 建置，提供 `libaudio_pipeline_mcab.a`。模型的 PCM stream SDK 可以各自使用。[四聲道路由](../../html/pipelines/index.html)
