> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_03_config.md) ｜ [下一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) <!-- split-nav -->
## 5. 錯誤處理

### 5.1 錯誤語意總表

回傳碼沿用核心層的 `FourAecNrResStatus`：
`FOUR_AEC_NR_RES_OK` = 0、`INVALID_ARGUMENT` = −1、`SEQUENCE_ERROR` = −2、`DSP_ERROR` = −3。

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `audio_pipeline_4ch_default_config()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `audio_pipeline_4ch_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | — |
| `audio_pipeline_4ch_init()` | 非 NULL handle | `NULL` | 等同 `init_ex(..., NULL)` |
| `audio_pipeline_4ch_init_ex()` | 非 NULL handle | `NULL` | 8 項 descriptor 檢查任一不符即 NULL |
| `audio_pipeline_4ch_create()` | 非 NULL handle | `NULL` | config 無效或配置失敗 |
| `audio_pipeline_4ch_process_front()` | `0`，`*out` 為借用的前端視圖 | `-1` / `-3` | 見 §5.2 |
| `audio_pipeline_4ch_process_beam()` | `0`，填 `*beam` 與（非 NULL 時）`*info` | `-1` / `-2` | 見 §5.2 |
| `audio_pipeline_4ch_process_with_activity()` | `0` | `-1` / `-2` / `-3` | 見 §5.2 |
| `audio_pipeline_4ch_process()` | `0` | `-1` / `-2` / `-3` | 見 §5.2 |
| `audio_pipeline_4ch_fuse()` | `0`，填 `*frame` | `-1` / `-2` / `-3` | 見 §5.2；失敗時整個單元已 reset |
| `audio_pipeline_4ch_get_core()` | 非 NULL（唯讀） | `NULL`（`p == NULL` 或已 destroy） | 診斷用 |
| `audio_pipeline_4ch_last_core_timing()` | `void`，`*out` 填妥 | 不會失敗 | `p` 為 NULL 或已 destroy 時把 `*out` 清零 |
| `audio_pipeline_4ch_reset()` | `void` | 不會失敗 | `p == NULL` 或已 destroy → 靜默 no-op |
| `audio_pipeline_4ch_destroy()` | `void` | 不會失敗 | 見 §5.5 |
| `*_hop_size` / `*_frame_size` / `*_fft_size` / `*_n_freqs` / `*_sample_rate` | 正值 | `-1` | `p == NULL` 或已 destroy |
| `*_doa_*` / `*_bf_sample_rate` / `*_bf_frame_size` / `*_bf_hop_size` / `*_bf_fft_size` | 正值 | `-1` | 同上 |
| `audio_pipeline_4ch_bf_effective_adapt_interval()` | `>= 1` | `-1` | `p == NULL`、已 destroy，或 BF 不存在 |
| **`audio_pipeline_4ch_bf_lambda()`** | 有限的 `float` | **`NaN`** | `p == NULL`、已 destroy，或 BF 不存在。**這是本 API 面上唯一用 NaN 表示錯誤的 function** —— 要用 `isnan()` 判斷，不能用 `== -1` 之類 |
| `*_matched_filter_count` / `*_linear_aec_count` | `1`（僅 `MATCHED`，其他模式 `0`）／ `4` | `0` | `p == NULL` 或已 destroy |
| `audio_pipeline_4ch_spatial_backend()` | `const char*`，永不為 NULL | 不會失敗 | 不吃 handle |

> **注意慣例不一致**：`get_mem_requirements()` 用 **0 = 成功 / −1 = 失敗**；
> 下層模組的 `srp_get_mem_size()` / `bf_get_mem_size()` /
> `aec_get_mem_size()` / `anr_get_mem_size()` / `fft_get_mem_size()`
> 是**回傳 0 代表失敗**。形狀 accessor 用 `-1`、結構稽核 accessor 用 `0`、
> 而 `bf_lambda()` 用 `NaN`。三種都不一樣，寫錯誤處理時請逐一對照本表。

### 5.2 process 各入口與 fuse 的完整拒絕清單

`audio_pipeline_4ch_process_front()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `far_reference` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 前端 `four_aec_front_process()` 失敗（含 mic/ref 含非有限值 → `-1`） | 原樣傳回核心層的碼 | 不留待處理的 front／beam（之後的 `process_beam()`／`fuse()` 回 `SEQUENCE_ERROR`、不動狀態）；`-1` 時自適應狀態不變，`-3` 時核心層已自動 reset |
| 3 | 核心層回報的形狀與本層記錄的不一致 | `DSP_ERROR`(-3) | **整個單元已被自動 reset** |

隨時都可以作為一個 hop 的第一個呼叫；它會取代尚未完成的前一個 front。

`audio_pipeline_4ch_process_beam()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、已 destroy、`front` 或 `beam` 為 NULL、`vad_raw` 或 `vad_out` 不是 0 或 1、`front->n_freqs` 與單元不同、`front->linear_spectra[ch]` 有 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 沒有待處理的 front，或這一 hop 的 beam 已經跑過 | `SEQUENCE_ERROR`(-2) | 無 |

`audio_pipeline_4ch_process_with_activity()` = `process_front()` 加 `process_beam()`：先檢查 `p`／三個 buffer／`beam` 非 NULL、
`vad_raw`／`vad_out` 為 0 或 1（`-1`，無副作用），再依序得到上面兩張表的結果。
`audio_pipeline_4ch_process()` 在這之前額外先做兩件事：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 0a | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `far_reference` / `beam` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無，**內建 VAD 狀態不受影響** |
| 0b | `far_reference`（`hop` 個）含非有限值 | `INVALID_ARGUMENT`(-1) | 無，**內建 VAD 狀態不受影響** |

（0b 之所以存在：內建 VAD 只看麥克風資料，不看 reference。若不在這裡先擋，
一筆壞掉的 reference 會先讓 VAD 的噪聲估計與 hangover 前進，之後才在更深層被拒。）

內建 VAD 遇到非有限的**麥克風**資料時也不會動任何 VAD 狀態；
那一幀會在核心層被 `-1` 拒絕。

`info` 參數可以是 `NULL`（不影響回傳值）。**所有入口都不會檢查 buffer 長度** —— `hop * 4` 與 `hop` 是你的責任。

`audio_pipeline_4ch_fuse()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、已 destroy、`frame` 為 NULL，或 `core.enable_post == 0` | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 這一 hop 沒有 beam（沒跑過 `process_beam()`/`process_with_activity()`/`process()`） | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | 核心層 `four_aec_fuse()` 失敗（權重／頻譜非有限、lane 的 far 頻譜不一致等；`SEQUENCE_ERROR` 除外，它照 #2 處理） | 原樣傳回核心層的碼 | **整個單元（核心、SRP、BF）已被自動 reset**；你的後級（含你的 `AecRes`）也要跟著 reset |

`frame` 的指標借自核心，到下一次 `process_front()`／`reset()` 為止有效；`frame->g_res_ctx` 恆為 NULL，
你的 `se_res_stage_gain()` 拒絕該 frame（回 `NULL`）時單元不會自動 reset，單元與你的後級要由你一起 reset（[§3.1](integration_4ch_spatial_zh_TW_02_lifecycle.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼)）。

### 5.3 `process_with_activity()` 的三個活動參數

| 參數 | 型別 | 合法值 | 意義 |
|---|---|---|---|
| `vad_raw` | `int` | 只接受 `0` 或 `1` | 控制 SRP-PHAT 這一幀要不要吃新的觀測 |
| `vad_out` | `int` | 只接受 `0` 或 `1` | 「目標語音正在發生」的保持狀態。餵給 DOA 平滑器；非 0 時**凍結** BF 的自適應更新 |
| `frequency_mask` | `const int*` | `NULL`，或剛好 `n_freqs` 個 `int` | 每個 bin 一格。值為 `1` 表示該 bin 被選入 SRP，且該 bin 的 BF 自適應被凍結。`NULL` = 不套遮罩 |

`vad_raw` / `vad_out` 傳 `0`/`1` 以外的值會直接被 `-1` 拒絕（實測 `vad_raw = 2` → `-1`）。
`frequency_mask` 的**內容值不做驗證**，長度也不檢查 —— 你要保證它至少 `n_freqs` 個 `int`。

### 5.4 診斷輸出去向

**這一層與核心層在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號。**

也就是說：`init_ex()` 回 `NULL` 時你**不會**知道是哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，把差異印出來。

`NO_STDIO=1` 對這兩層沒有任何影響（本來就沒有 stdio）。

### 5.5 `destroy()` 的兩種語意

| Instance 來源 | `destroy()` 行為 |
|---|---|
| `audio_pipeline_4ch_init()` / `init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、冪等。內部的 core / SRP / BF 也都不會釋放任何東西（它們都是從你的 pool 切出來的） |
| `audio_pipeline_4ch_create()`（heap） | 釋放 `create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次** |

destroy 之後不要再呼叫任何 accessor。

### 5.6 `AudioPipeline4ChFrameInfo`（唯讀 telemetry）

`process()` / `process_with_activity()` 成功時填入。`info` 傳 `NULL` 就不填。

| 欄位 | 型別 | 內容 |
|---|---|---|
| `frame_index` | `uint64_t` | 自 init 或上次 `reset()` 起的幀序號 |
| `delay` | `FourAecNrResDelayState` | 共用 delay matcher 狀態：`delay_samples` / `confidence` / `solid` / `changed` / `estimator_calls` / `estimator_updates` |
| `doa_raw_rad` | `float` | 這一幀的原始 DOA 估計（弧度） |
| `doa_smooth_rad` | `float` | 平滑後的 DOA（弧度） |
| `doa_used_rad` | `float` | BF 實際採用的方向（弧度） |
| `vad_raw` | `int` | 這一幀實際使用的 `vad_raw` |
| `vad_out` | `int` | 這一幀實際使用的 `vad_out` |
| `bf_adaptive` | `int` | BF 這一幀是否處於自適應狀態 |
| `doa_analysis_frames` | `int` | 這一幀做了幾次 DOA 分析 |

走 `audio_pipeline_4ch_process()` 時，`info.vad_raw` 與 `info.vad_out` 就是內建 VAD
這一幀的判定結果 —— 這是你唯一能觀察內建 VAD 行為的管道。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_03_config.md) ｜ [下一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) <!-- split-nav -->
