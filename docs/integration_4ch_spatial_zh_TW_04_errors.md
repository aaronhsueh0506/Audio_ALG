> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_03_config.md) ｜ [下一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) <!-- split-nav -->
## 5. 錯誤處理

### 5.1 錯誤語意總表

回傳碼是 `McabStatus`（`mcab.h`），helper 沿用同一組：
`MCAB_OK` = 0、`MCAB_INVALID_ARGUMENT` = −1、`MCAB_SEQUENCE_ERROR` = −2、`MCAB_DSP_ERROR` = −3。

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `mcab_default_config()`／`mcab_sample_beam_default_params()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `mcab_lane_config()`／`mcab_apply_lane_contract()`／`mcab_delay_estimator_config()` | `0` | `-1` | `cfg` 無效（`mcab_apply_lane_contract()` 另含 lane 的 `sample_rate` 與 flow 不同）；lane 保持原樣 |
| `mcab_get_mem_requirements()`／`mcab_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | — |
| `mcab_init()`／`mcab_init_ex()`／`mcab_create()` | 非 NULL handle | `NULL` | `init` 等同 `init_ex(..., NULL)`；8 項 descriptor 檢查與 [§5.7](integration_4ch_spatial_zh_TW_04_errors.md#57-mcab_init_ex-驗證什麼不能驗證什麼) 的物件驗證任一不符即 NULL，pool 不被寫入 |
| `mcab_sample_flow_get_mem_size()`／`_init()`／`_create()` | `0` | `-1` | `_init()` 失敗時 `f` 清零 |
| `mcab_sample_beam_get_mem_size()` | `0`，`*bytes` 填妥 | `-1` | `cfg` 或 `bp` 無效 |
| `mcab_sample_beam_init()` | 非 NULL handle | `NULL` | 參數、pool、flow 與 cfg 的 hop／bin／rate／`enable_post` 不符，或 SRP／BF init 失敗 |
| `mcab_sample_beam_process_front()` | `0`，`*out` 為借用的前端視圖 | `-1` / `-3` | 見 §5.2 |
| `mcab_sample_beam_process_beam()` | `0`，填 `*beam` 與（非 NULL 時）`*info` | `-1` / `-2` | 見 §5.2 |
| `mcab_sample_beam_process_with_activity()` | `0` | `-1` / `-2` / `-3` | 見 §5.2 |
| `mcab_sample_beam_process()` | `0` | `-1` / `-2` / `-3` | 見 §5.2 |
| `mcab_sample_beam_fuse()` | `0`，填 `*frame` | `-1` / `-2` / `-3` | 見 §5.2；除 `-2` 與 helper 自己的引數拒絕（NULL 引數、`enable_post = 0`）外，失敗已 reset 整個 helper（含 flow） |
| `mcab_get_last_timing()` | `void`，`*out` 填妥 | 不會失敗 | `p` 為 NULL 或已 destroy 時把 `*out` 清零；`out` 為 NULL 時什麼都不做（header 規定 `out` 不得為 NULL） |
| `mcab_reset()`／`mcab_sample_beam_reset()` | `void` | 不會失敗 | NULL 或已 destroy → 靜默 no-op |
| `mcab_destroy()`／`mcab_sample_beam_destroy()`／`mcab_sample_flow_destroy()` | `void` | 不會失敗 | 見 §5.5 |
| `mcab_hop_size()`／`_fft_size()`／`_n_freqs()`／`_sample_rate()` | 正值 | `-1` | `p == NULL` 或已 destroy |
| `mcab_enable_post()` | `0` 或 `1` | `-1` | 同上 |
| `mcab_sample_beam_bf_effective_adapt_interval()` | `>= 1` | `-1` | `b == NULL` 或已 destroy |
| **`mcab_sample_beam_bf_lambda()`** | 有限的 `float` | **`NaN`** | `b == NULL` 或已 destroy。**這是本 API 面上唯一用 NaN 表示錯誤的 function** —— 要用 `isnan()` 判斷，不能用 `== -1` 之類 |
| `mcab_matched_filter_count()`／`mcab_linear_aec_count()` | `1`（僅 `MATCHED`，其他模式 `0`）／ `4` | `0` | `p == NULL` 或已 destroy |
| `mcab_sample_beam_spatial_backend()` | `const char*`，永不為 NULL | 不會失敗 | 不吃 handle |

> **注意慣例不一致**：`mcab_get_mem_requirements()`、`mcab_sample_beam_get_mem_size()` 用 **0 = 成功 / −1 = 失敗**；
> 下層模組的 `srp_get_mem_size()` / `bf_get_mem_size()` /
> `aec_get_mem_size()` / `delay_aec_get_mem_size()` / `anr_get_mem_size()` / `fft_get_mem_size()`
> 是**回傳 0 代表失敗**（`delay_aec_init()` 則是 0 = 成功）。形狀 accessor 用 `-1`、結構稽核 accessor 用 `0`、
> 而 `bf_lambda()` 用 `NaN`。三種都不一樣，寫錯誤處理時請逐一對照本表。

### 5.2 process 各入口與 fuse 的完整拒絕清單

`mcab_sample_beam_process_front()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、`b` 已 destroy、`microphones_interleaved` / `far_reference` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | `mcab_process_front()` 拒絕：mic/ref 含非有限值 → `INVALID_ARGUMENT`；某條 lane 拒絕該 hop、參考環無法服務目前 delay 或 lane view 綁定失敗 → `DSP_ERROR` | 原樣傳回 mcab 的碼 | 不留待處理的 front／beam（之後的 `process_beam()`／`fuse()` 回 `SEQUENCE_ERROR`、不動狀態）；非有限值時自適應狀態不變，`DSP_ERROR` 時 flow（lane＋estimator）已自動 reset，**SRP、BF、fallback VAD 與 `frame_index` 不動** |

隨時都可以作為一個 hop 的第一個呼叫；它會取代尚未完成的前一個 front。

`mcab_sample_beam_process_beam()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、已 destroy、`front` 或 `beam` 為 NULL、`vad_raw` 或 `vad_out` 不是 0 或 1、`front->n_freqs` 與 helper 不同、`front->linear_spectra[ch]` 有 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 沒有待處理的 front，或這一 hop 的 beam 已經跑過 | `SEQUENCE_ERROR`(-2) | 無 |

`mcab_sample_beam_process_with_activity()` = `process_front()` 加 `process_beam()`：先檢查 `b`／兩個 buffer／`beam` 非 NULL、
`vad_raw`／`vad_out` 為 0 或 1（`-1`，無副作用），再依序得到上面兩張表的結果。
`mcab_sample_beam_process()` 在這之前額外先做兩件事：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 0a | `b` 為 NULL、`b` 已 destroy、`microphones_interleaved` / `far_reference` / `beam` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無，**內建 VAD 狀態不受影響** |
| 0b | `far_reference`（`hop` 個）含非有限值 | `INVALID_ARGUMENT`(-1) | 無，**內建 VAD 狀態不受影響** |

（0b 之所以存在：內建 VAD 只看麥克風資料，不看 reference。若不在這裡先擋，
一筆壞掉的 reference 會先讓 VAD 的噪聲估計與 hangover 前進，之後才在更深層被拒。）

內建 VAD 遇到非有限的**麥克風**資料時也不會動任何 VAD 狀態；
那一幀會在 `mcab_process_front()` 被 `-1` 拒絕。

`info` 參數可以是 `NULL`（不影響回傳值）。**所有入口都不會檢查 buffer 長度** —— `hop * 4` 與 `hop` 是你的責任。

`mcab_sample_beam_fuse()`（依實際檢查順序）：

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `b` 為 NULL、已 destroy、`frame` 為 NULL，或 flow 的 `enable_post == 0`（以 `mcab_enable_post(flow)` 判斷，不看 helper 收到的 cfg） | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 這一 hop 沒有 beam（沒跑過 `process_beam()`/`process_with_activity()`/`process()`） | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | `mcab_fuse()` 失敗：helper 自己算的權重非有限或全零 → `INVALID_ARGUMENT`；lane 的 far 頻譜不一致、beamformed 頻譜或 lane 頻譜非有限、投影溢位 → `DSP_ERROR`（`SEQUENCE_ERROR` 除外，它照 #2 處理） | 原樣傳回 mcab 的碼 | **整個 helper 已被自動 reset**（flow、SRP、BF、VAD、`frame_index`；`DSP_ERROR` 時 lane 因此被 reset 兩次：`mcab_fuse()` 內一次、helper 整體一次）；你的後級（含你的 `AecRes`）也要跟著 reset |

`frame` 的指標借自 lane／mcab，到下一次 `process_front()`／`reset()` 為止有效；`frame->g_res_ctx` 恆為 NULL，
你的 `aec_res_process()` 拒絕該 frame（回 `NULL`）時 helper 不會自動 reset，flow／helper 與你的後級要由你一起 reset（[§3.1](integration_4ch_spatial_zh_TW_02_lifecycle.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼)）。

只用 mcab、自己算權重的應用（不經 helper）直接呼叫 `mcab_process_front()`／`mcab_fuse()`，拒絕規則相同：
`mcab_fuse(p, weights, beamformed_error, frame)` 在 NULL 引數、`enable_post = 0`、權重全零或非有限時回 `INVALID_ARGUMENT`（不動任何狀態，front 仍可用，可修正權重後重試同一 hop），
沒有 front 時回 `SEQUENCE_ERROR`，lane 的 far 頻譜不一致、`beamformed_error` 或 lane 頻譜非有限、投影溢位時回 `DSP_ERROR` 並 `mcab_reset()`。

### 5.3 `process_with_activity()` 的三個活動參數

| 參數 | 型別 | 合法值 | 意義 |
|---|---|---|---|
| `vad_raw` | `int` | 只接受 `0` 或 `1` | 控制 SRP-PHAT 這一幀要不要吃新的觀測 |
| `vad_out` | `int` | 只接受 `0` 或 `1` | 「目標語音正在發生」的保持狀態。餵給 DOA 平滑器；非 0 時**凍結** BF 的自適應更新 |
| `frequency_mask` | `const int*` | `NULL`，或剛好 `n_freqs` 個 `int` | 每個 bin 一格。值為 `1` 表示該 bin 被選入 SRP，且該 bin 的 BF 自適應被凍結。`NULL` = 不套遮罩 |

`vad_raw` / `vad_out` 傳 `0`/`1` 以外的值會直接被 `-1` 拒絕（`vad_raw = 2` → `-1`）。
`frequency_mask` 的**內容值不做驗證**，長度也不檢查 —— 你要保證它至少 `n_freqs` 個 `int`。

### 5.4 診斷輸出去向

**mcab 與 helper 在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號**
（`audit-no-stdio` 把 `mcab.o`、`mcab_sample_flow.o`、`mcab_sample_beam.o` 都列入檢查）。

也就是說：`mcab_init_ex()` 或 `mcab_sample_beam_init()` 回 `NULL` 時你**不會**知道是哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，把差異印出來，並對照 [§5.7](integration_4ch_spatial_zh_TW_04_errors.md#57-mcab_init_ex-驗證什麼不能驗證什麼) 的清單檢查 lane 與 estimator。

`NO_STDIO=1` 對這兩層沒有任何影響（本來就沒有 stdio）。

### 5.5 `destroy()` 的語意

| 呼叫 | 行為 |
|---|---|
| `mcab_destroy(flow)`（`mcab_init()`／`init_ex()` 的 caller-owned pool） | **不釋放**你的 pool，**也不碰** lane 與 estimator（不 `aec_destroy`、不 free）。NULL 安全、冪等 |
| `mcab_destroy(flow)`（`mcab_create()`，heap） | 釋放 `mcab_create()` 配置的那一塊（只有 mcab 自己的 pool）。遵循一般 `free()` 語意：**只能呼叫一次** |
| `mcab_sample_beam_destroy(b)` | 先 `bf_destroy()` 再 `srp_destroy()`（BF 借用 SRP 的 steering 表）；不釋放 helper 的 pool，也不動 flow。NULL 安全、冪等 |
| `mcab_sample_flow_destroy(&f)` | `mcab_destroy()`，再 `aec_destroy()` 四條 lane，最後釋放 `mcab_sample_flow_create()` 的 heap 區塊；冪等 |

拆除順序：helper → `mcab_destroy()` → lane（`mcab_sample_flow_destroy()` 已含後兩步）。destroy 之後不要再呼叫任何 accessor。

### 5.6 `McabSampleBeamInfo`（唯讀 telemetry）

`process_beam()` / `process_with_activity()` / `process()` 成功時填入。`info` 傳 `NULL` 就不填。

| 欄位 | 型別 | 內容 |
|---|---|---|
| `frame_index` | `uint64_t` | 自 init 或上次 `reset()` 起已做過的 beam 數（本次呼叫填入的是呼叫前的計數） |
| `delay` | `McabDelayState` | 共用 delay matcher 狀態：`delay_samples` / `confidence` / `solid` / `changed` / `estimator_calls` / `estimator_updates` |
| `doa_raw_rad` | `float` | 這一幀的原始 DOA 估計（弧度） |
| `doa_smooth_rad` | `float` | 平滑後的 DOA（弧度） |
| `doa_used_rad` | `float` | BF 實際採用的方向（弧度） |
| `vad_raw` | `int` | 這一幀實際使用的 `vad_raw` |
| `vad_out` | `int` | 這一幀實際使用的 `vad_out` |
| `bf_adaptive` | `int` | BF 這一幀是否處於自適應狀態 |
| `doa_analysis_frames` | `int` | 恆為 1（helper 固定填入） |

走 `mcab_sample_beam_process()` 時，`info.vad_raw` 與 `info.vad_out` 就是內建 VAD
這一幀的判定結果 —— 這是你唯一能觀察內建 VAD 行為的管道。

### 5.7 `mcab_init_ex()` 驗證什麼、不能驗證什麼

mcab 借用你建立的物件，所以 init 時在寫入 pool 之前先檢查它們，任何一項不符就回 `NULL`（pool 不動）：

| 檢查 | 怎麼查 |
|---|---|
| config 合法（grid、delay 規則、quarantine／duty／對齊環範圍、MATCHED 的 estimator 能定大小） | 與 `mcab_get_mem_requirements()` 同一套規則 |
| descriptor 8 點 gate、16-byte 對齊、bytes 足夠 | [§8.2](integration_4ch_spatial_zh_TW_07_versioning.md#82-mcab_init_ex-的-8-項檢查) |
| 四條 lane 都存在且互不相同 | 指標比較 |
| 每條 lane 的 hop 與 bin 數 | `aec_hop_size()`、`AecResContext.hop_size/n_freqs` |
| lane 有回傳 linear context、沒有自己的 gain（⇒ `return_res_context = 1`、`enable_res = 0`） | context 指標非 NULL、`res_gain == NULL` |
| lane 的 post half 與 `cfg.enable_post` 一致 | `r2`／`comfort_noise` 有無 |
| lane 吃已對齊的 far（`EXTERNAL_ALIGNED`） | `aec_get_linear_context()` 回 `LOCKED` 且 `delay_samples == 0` |
| MATCHED 必須給 estimator、其他模式必須是 NULL | 指標 |
| estimator 的 bank 大小、rate、48 kHz scratch 夠這個 hop | `DelayAec` 的 `est.matched_filter.num_filters`、`rate_factor`、`resample_cap`、scratch 指標 |

**不能驗證**：`Aec` 沒有 config getter，所以 lane 的參數（level、`filter_length` 等）是你的，mcab 不檢查，
契約欄位也只和你那次 `mcab_apply_lane_contract()` 一樣正確。`EXTERNAL_ALIGNED` 的檢查是必要不充分：
一條已經跑過音訊、鎖在 0 延遲的 lane 也會回 `LOCKED, 0`——所以 lane 與 estimator 必須是**新建的**或已 reset
（[§3](integration_4ch_spatial_zh_TW_02_lifecycle.md#3-lifecycle)）。estimator 的 hop 只在 48 kHz 才看得出（16 kHz 沒有依 hop 定大小的 scratch）。
`tests/test_mcab.c` 的 `test_init_validates_objects` 實際驗過：重複／缺 lane、MATCHED 缺 estimator、非 MATCHED 卻給 estimator、bank 大小不符、
別的格點的 lane、front-only lane 配 `enable_post = 1`、自行對齊的 lane、48 kHz 用小 hop 建的 estimator 全部被拒且 pool 不被寫入；合法組合被接受。

helper 的 `mcab_sample_beam_init()` 另外驗證：`bp` 的全部規則（含 auto-VAD）、pool 大小與對齊、flow 的 hop／bin／sample rate／`enable_post` 與 `cfg` 一致。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_03_config.md) ｜ [下一份](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md) <!-- split-nav -->
