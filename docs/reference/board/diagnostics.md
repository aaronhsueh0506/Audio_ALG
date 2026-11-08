# 板端診斷

| 問題 | 讀取位置 |
|---|---|
| 沒有抵消，或延遲跳動 | 用 `aec_debug_status()` 查看目前狀態，以 `aec_get_linear_context()` 比對 linear error／far。狀態切換前後用同一個 hop 編號記錄。 |
| 頻域輸入正常但輸出被壓低 | 在同一 hop 讀 `aec_get_res_context()` 的 RES gain 及 `anr_debug_status()`。保留 AEC 的 rendering base 與分析用 residual 的來源區別。 |
| 不確定板上有沒有跑模型 | 在實際 AI3 呼叫處計數，同時記 pre 回傳、AI3 結果、generation 與 post 回傳。DFN2 的 N=1 暖機也會呼叫推論，stage 的 frames_in 包含暖機。 |
| 量測單一 stage 的時間 | AEC 編譯時啟用 `AEC_STAGE_TIMING` 後讀 `aec_get_last_timing()`。關閉時數值為零。 |
| 評估記憶體 | 用各庫的 size／breakdown 查詢帶入目標 grid 和 build，確認對齊與 descriptor。 |

`NO_STDIO=1` 仍可讀結構化狀態。逐 hop 的檔案 trace 需要具備 stdio 的建置。[板端整合](README.md)

[返回本主題](README.md)
