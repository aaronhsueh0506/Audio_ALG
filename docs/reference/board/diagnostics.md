# 板端診斷

| 問題 | 讀取位置 |
|---|---|
| 沒有抵消，或延遲跳動 | 用 `aec_debug_status()` 查看目前狀態（四麥克風 array 同樣可讀，延遲取共用對齊），以 `aec_get_linear()` 比對 linear error／far。四麥克風另讀 `aec_get_array()` 的 `delay` 與 `aec_array_*` 計數（realign 次數、pending 候選、duty census）。狀態切換前後用同一個 hop 編號記錄。 |
| 頻域輸入正常但輸出被壓低 | 在同一 hop 讀 `aec_get_residual()` 的 `res_gain`（array 為融合後的唯一增益）及 `anr_debug_status()`。保留 AEC 的 rendering base（`base_spec`）與分析用 residual（`error_spec`）的來源區別。 |
| 不確定板上有沒有跑模型 | 在實際 AI3 呼叫處計數，同時記 pre 回傳、AI3 結果、generation 與 post 回傳。DFN2 的 N=1 在 init 與 reset 後的前 M 拍沒有 job（work.action 為 BYPASS、不呼叫推論），stage 的 frames_in 仍包含這幾拍，commits／skips 不含。 |
| 量測單一 stage 的時間 | 讀 `aec_get_last_timing()`（四麥克風讀 `aec_array_get_timing()`，多一欄 `fuse_us`）；AEC 預設建置即量測，`make PROFILE=0`（`-DAEC_STAGE_TIMING=0`）關閉後數值為零。 |
| 評估記憶體 | 用各庫的 size／breakdown 查詢帶入目標 grid 和 build（array：`aec_get_mem_requirements()`、`aec_get_mem_breakdown()`），確認對齊與 descriptor。 |

`NO_STDIO=1` 仍可讀結構化狀態。[板端整合](README.md)

[返回本主題](README.md)
