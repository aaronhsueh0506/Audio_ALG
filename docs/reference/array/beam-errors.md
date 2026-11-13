# SRP／BF 錯誤與相容性

## 5. 錯誤處理

### 5.1 錯誤語意總表

回傳碼是 `AecStatus`（`aec.h`），helper 沿用同一組：
`AEC_OK` = 0、`AEC_ERR_ARG` = −1、`AEC_ERR_MODE` = −2、`AEC_ERR_SEQUENCE` = −3、`AEC_ERR_DSP` = −4。

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `array_sample_default_config()`／`array_sample_beam_default_params()` | 填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `aec_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | `cfg` 為 NULL 或無效 |
| `aec_init()`／`aec_init_ex()` | 非 NULL handle | `NULL` | descriptor 檢查任一不符即 NULL，pool 不被寫入（[errors.md §8](errors.md#8-版本與相容性)） |
| `array_sample_beam_get_mem_size()` | `0`，`*bytes` 填妥 | `-1` | `cfg` 或 `bp` 無效 |
| `array_sample_beam_init()` | 非 NULL handle | `NULL` | 參數、pool、array 的 hop 與 `cfg` 不符、不是 array，或 SRP／BF init 失敗（§5.7） |
| `array_sample_beam_process_front()` | `AEC_OK`，`*out` 為借用的前端視圖 | `-1` / `-4` | 見 §5.2 |
| `array_sample_beam_process_beam()` | `AEC_OK`，填 `*beam` 與（非 NULL 時）`*info` | `-1` / `-3` | 見 §5.2 |
| `array_sample_beam_process_with_activity()` | `AEC_OK` | `-1` / `-3` / `-4` | 見 §5.2 |
| `array_sample_beam_process()` | `AEC_OK` | `-1` / `-3` / `-4` | 見 §5.2 |
| `array_sample_beam_fuse()` | `AEC_OK`，填 `*frame` | `-1` / `-2` / `-3` / `-4` | 見 §5.2；除 `-1`、`-2`、`-3` 的 helper 自己的拒絕外，失敗已 reset 整個 helper（含 array） |
| `aec_array_get_timing()` | `void`，`*out` 填妥 | 不會失敗 | `a` 為 NULL 或已 destroy 時不動 `*out`；`out` 為 NULL 時什麼都不做（header 規定 `out` 不得為 NULL） |
| `aec_reset()`／`array_sample_beam_reset()` | `void` | 不會失敗 | helper 的 reset：NULL 或已 destroy → 靜默 no-op |
| `aec_destroy()`／`array_sample_beam_destroy()` | `void` | 不會失敗 | 見 §5.5 |
| `array_sample_beam_bf_effective_adapt_interval()` | `>= 1` | `-1` | `b == NULL` 或已 destroy |
| **`array_sample_beam_bf_lambda()`** | 有限的 `float` | **`NaN`** | `b == NULL` 或已 destroy。**這是本 API 面上唯一用 NaN 表示錯誤的 function** —— 要用 `isnan()` 判斷，不能用 `== -1` 之類 |
| `aec_array_matched_filter_count()`／`aec_array_linear_aec_count()` | `1`（僅 `MATCHED`，其他模式 `0`）／ `4` | `0` | `a == NULL` 或已 destroy |
| `array_sample_beam_spatial_backend()` | `const char*`，永不為 NULL | 不會失敗 | 不吃 handle |

> **注意慣例不一致**：`aec_get_mem_requirements()`、`array_sample_beam_get_mem_size()` 用 **0 = 成功 / −1 = 失敗**；
> 下層模組的 `srp_get_mem_size()` / `bf_get_mem_size()` /
> `aec_get_mem_size()` / `anr_get_mem_size()` / `fft_get_mem_size()`
> 是**回傳 0 代表失敗**。形狀 accessor（`aec_hop_size()`）對 NULL 回 `0`、`aec_get_mem_breakdown()`／`aec_resolve_signal_grid()` 與處理呼叫一樣回 `AecStatus`（0 = 成功）、結構稽核 accessor 用 `0`、
> 而 `bf_lambda()` 用 `NaN`。三種都不一樣，寫錯誤處理時請逐一對照本表。

### 5.2 process 各入口與 fuse 的完整拒絕清單

`array_sample_beam_process_front()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、`b` 已 destroy、`microphones_interleaved` / `far_reference` / `out` 任一為 NULL | `AEC_ERR_ARG`(-1) | 無 |
| 2 | `aec_process_linear()` 或 `aec_get_array()` 拒絕：mic/ref 含非有限值 → `AEC_ERR_ARG`；某條 lane 拒絕該 hop、參考環無法服務目前 delay 或 lane view 綁定失敗 → `AEC_ERR_DSP` | 原樣傳回 array 的碼 | 不留待處理的 front／beam（之後的 `process_beam()`／`fuse()` 回 `AEC_ERR_SEQUENCE`、不動狀態）；非有限值時自適應狀態不變，`AEC_ERR_DSP` 時 array 已自動 reset，**SRP、BF、fallback VAD 與 `frame_index` 不動** |

隨時都可以作為一個 hop 的第一個呼叫；它會取代尚未完成的前一個 front。

`array_sample_beam_process_beam()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、已 destroy、`front` 或 `beam` 為 NULL、`vad_raw` 或 `vad_out` 不是 0 或 1、`front->n_freqs` 與 helper 不同、`front->lane_error_spec[ch]` 有 NULL | `AEC_ERR_ARG`(-1) | 無 |
| 2 | 沒有待處理的 front，或這一 hop 的 beam 已經跑過 | `AEC_ERR_SEQUENCE`(-3) | 無 |

`array_sample_beam_process_with_activity()` = `process_front()` 加 `process_beam()`：先檢查 `b`／兩個 buffer／`beam` 非 NULL、
`vad_raw`／`vad_out` 為 0 或 1（`-1`，無副作用），再依序得到上面兩張表的結果。
`array_sample_beam_process()` 在這之前額外先做兩件事：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 0a | `b` 為 NULL、`b` 已 destroy、`microphones_interleaved` / `far_reference` / `beam` 任一為 NULL | `AEC_ERR_ARG`(-1) | 無，**內建 VAD 狀態不受影響** |
| 0b | `far_reference`（`hop` 個）含非有限值 | `AEC_ERR_ARG`(-1) | 無，**內建 VAD 狀態不受影響** |

（0b 之所以存在：內建 VAD 只看麥克風資料，不看 reference。若不在這裡先擋，一筆壞掉的 reference 會先讓 VAD 的噪聲估計與 hangover 前進，之後才在更深層被拒。）

內建 VAD 遇到非有限的**麥克風**資料時也不會動任何 VAD 狀態；那一幀會在 `aec_process_linear()` 被 `AEC_ERR_ARG` 拒絕。

`info` 參數可以是 `NULL`（不影響回傳值）。**所有入口都不會檢查 buffer 長度** —— `hop * 4` 與 `hop` 是你的責任。

`array_sample_beam_fuse()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、已 destroy、`frame` 為 NULL | `AEC_ERR_ARG`(-1) | 無 |
| 2 | 這一 hop 沒有 beam（沒跑過 `process_beam()`/`process_with_activity()`/`process()`） | `AEC_ERR_SEQUENCE`(-3) | 無 |
| 3 | array 是 `AEC_RESIDUAL_STATE`（`aec_process_residual()` 回 `AEC_ERR_MODE`） | `AEC_ERR_MODE`(-2) | 無（helper 與 array 都不動；beam 仍有效） |
| 4 | `aec_process_residual()` 或 `aec_get_residual()` 其他失敗：helper 自己算的權重非有限或全零、或 beamformed 頻譜含非有限 bin → `AEC_ERR_ARG`（array 自己不動）；lane 的 far 頻譜不一致、lane 頻譜非有限、投影溢位 → `AEC_ERR_DSP` | 原樣傳回 array 的碼 | **整個 helper 已被自動 reset**（array、SRP、BF、VAD、`frame_index`；`AEC_ERR_DSP` 時 lane 因此被 reset 兩次：array 內一次、helper 整體一次）；你的後級（含 NR）也要跟著 reset |

`frame` 的指標借自 array，到下一次 `process_front()`／`reset()` 為止有效；`frame->res_gain` 是陣列增益級的輸出（FULL）。
你的 `array_sample_anr_gain()` 或 `aec_apply_*` 拒絕該 frame 時 helper 不會自動 reset，helper 與你的後級要由你一起 reset（[§3.1](beam.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼)）。

只用 array、自己算權重的應用（不經 helper）直接呼叫 `aec_process_linear()`／`aec_process_residual()`，拒絕規則見 [errors.md §5.2、§5.3](errors.md#52-aec_process_linear-的完整拒絕清單依實際檢查順序)：
權重全零或非有限、或 `beamformed_error` 含非有限值時回 `AEC_ERR_ARG`（不動任何狀態，linear 仍可用，可修正後重試同一 hop），沒有 linear 時回 `AEC_ERR_SEQUENCE`，
lane 的 far 頻譜不一致、lane 頻譜非有限、投影溢位時回 `AEC_ERR_DSP` 並自動 reset array。

### 5.3 `process_with_activity()` 的三個活動參數

| 參數 | 型別 | 合法值 | 意義 |
|---|---|---|---|
| `vad_raw` | `int` | 只接受 `0` 或 `1` | 控制 SRP-PHAT 這一幀要不要吃新的觀測 |
| `vad_out` | `int` | 只接受 `0` 或 `1` | 「目標語音正在發生」的保持狀態。餵給 DOA 平滑器；非 0 時**凍結** BF 的自適應更新 |
| `frequency_mask` | `const int*` | `NULL`，或剛好 `n_freqs` 個 `int` | 每個 bin 一格。值為 `1` 表示該 bin 被選入 SRP，且該 bin 的 BF 自適應被凍結。`NULL` = 不套遮罩 |

`vad_raw` / `vad_out` 傳 `0`/`1` 以外的值會直接被 `-1` 拒絕（`vad_raw = 2` → `-1`）。
`frequency_mask` 的**內容值不做驗證**，長度也不檢查 —— 你要保證它至少 `n_freqs` 個 `int`。

### 5.4 診斷輸出去向

**array 與 helper 在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號**
（`audit-no-stdio` 把 `array_sample_beam.o` 與各 archive 都列入檢查）。

也就是說：`aec_init_ex()` 或 `array_sample_beam_init()` 回 `NULL` 時你**不會**知道是哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，把差異印出來，並對照 [§5.7](#57-helper-的-init-驗證什麼) 的清單檢查 config 與 helper 參數。

`NO_STDIO=1` 對這兩層沒有任何影響（本來就沒有 stdio）。

### 5.5 `destroy()` 的語意

| 呼叫 | 行為 |
|---|---|
| `aec_destroy(aec)`（`aec_init()`／`init_ex()` 的 caller-owned pool） | **不釋放**你的 pool。NULL 安全、冪等 |
| `aec_destroy(aec)`（`aec_create()`，heap） | 釋放 `aec_create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次** |
| `array_sample_beam_destroy(b)` | 先 `bf_destroy()` 再 `srp_destroy()`（BF 借用 SRP 的 steering 表）；不釋放 helper 的 pool，也不動 array。NULL 安全、冪等 |

拆除順序：helper → `aec_destroy()`。destroy 之後不要再呼叫任何 accessor。

### 5.6 `ArraySampleBeamInfo`（唯讀 telemetry）

`process_beam()` / `process_with_activity()` / `process()` 成功時填入。`info` 傳 `NULL` 就不填。

| 欄位 | 型別 | 內容 |
|---|---|---|
| `frame_index` | `uint64_t` | 自 init 或上次 `reset()` 起已做過的 beam 數（本次呼叫填入的是呼叫前的計數） |
| `delay` | `AecDelayState` | 共用 delay matcher 狀態：`delay_samples` / `confidence` / `solid` / `changed` / `estimator_calls` / `estimator_updates` |
| `doa_raw_rad` | `float` | 這一幀的原始 DOA 估計（弧度） |
| `doa_smooth_rad` | `float` | 平滑後的 DOA（弧度） |
| `doa_used_rad` | `float` | BF 實際採用的方向（弧度） |
| `vad_raw` | `int` | 這一幀實際使用的 `vad_raw` |
| `vad_out` | `int` | 這一幀實際使用的 `vad_out` |
| `bf_adaptive` | `int` | BF 這一幀是否處於自適應狀態 |
| `doa_analysis_frames` | `int` | 恆為 1（helper 固定填入） |

走 `array_sample_beam_process()` 時，`info.vad_raw` 與 `info.vad_out` 就是內建 VAD 這一幀的判定結果 —— 這是你唯一能觀察內建 VAD 行為的管道。

### 5.7 helper 的 init 驗證什麼

array 的 config 與 descriptor 檢查見 [errors.md §8](errors.md#8-版本與相容性) 與 [config.md §4.1](config.md#41-逐欄位)。`array_sample_beam_init()` 另外驗證：

| 檢查 | 怎麼查 |
|---|---|
| `bp` 的全部規則（含 auto-VAD） | [beam-settings.md §4.2～§4.5](beam-settings.md#42-陣列幾何4--2-欄) |
| pool 16-byte 對齊、大小足夠 | 與 `array_sample_beam_get_mem_size()` 同一套規則 |
| `aec` 的 hop 與 `cfg` 導出的相同 | `aec_hop_size()` |
| `aec` 是四麥克風 array | `aec_array_linear_aec_count(aec) == AEC_ARRAY_MICS` |
| SRP 與 BF 初始化成功 | 回傳值 |

**不能驗證**：helper 不檢查 `cfg.residual`（STATE 的 array 只是讓 `fuse` 回 `AEC_ERR_MODE`），也無從核對 `cfg` 與建 array 的是同一份（只比 hop）；
`sample_rate` 與 `fft_size` 請傳同一份 `cfg`。

## 8. 版本與相容性

array 有 descriptor（`AecMemReq`，`libaec.a` 的整個 array pool），格式、檢查項與相容性規則見 [errors.md §8](errors.md#8-版本與相容性)。
helper（`array_sample_beam_*`）是範例碼，pool 以 `array_sample_beam_get_mem_size()` 查大小，沒有 descriptor、沒有 layout version。

升級之後也要重新確認 [§4.6](beam-settings.md#46-你設定的值不一定是實際生效的值) 的三個生效值：改 grid、改 backend、
或函式庫改版都可能讓生效值變動，而設定值看起來完全沒變。

[返回本主題](README.md)
