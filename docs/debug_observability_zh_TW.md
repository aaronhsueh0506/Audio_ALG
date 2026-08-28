# DEBUG 可觀測性：現況盤點與新增提案

本文件回答一個問題：**release 出貨之後，如果現場出現回音殘留、近端被削、
或是聽起來「NN 沒在跑」，我們手上有什麼可以判斷？**

前半是現況盤點（已驗證的事實），後半是提案（**尚未實作**）。
文件站 `html/` 只描述已存在的程式碼，所以提案放在這裡。

盤點的對應說明頁：[`html/integration_example.html`](html/integration_example.html) §7。

---

## 1. 結論先講：一條硬性設計約束

以 `NO_STDIO=1` 建出 archive 後用 `nm` 檢查，得到一個非常乾淨的分界：

| | 結果 |
|---|---|
| **pull 型結構化快照** | **全部存活**（`aec_debug_status`、`aec_get_res_context`、`aec_get_linear_context`、`aec_get_mem_breakdown`、`aec_linear_is_cancelling`、`aec_far_fft_real_compute_count`、`mmse_lsa_debug_status`、4ch 的全部 counter accessor） |
| **push 型 log / trace** | **全部消失**（沒有 `aec_debug_logf`、沒有 `aec_debug_set_trace`、archive 內連 `fprintf`/`vfprintf` 的引用都沒有） |
| **stage timing** | 函式存活但**數值全為 0**（`AEC_STAGE_TIMING` 預設 0） |
| **finite guards / 錯誤碼** | 全部存活，不受任何開關影響 |

> **約束：出貨板上唯一還活著的診斷面，是「主動 poll 的結構化狀態快照」。**
> 因此所有 DEBUG 設計都必須落在 pull 型快照與計數器上。
> **不要投資在 printf 型 logging** —— 它在出貨組態下必然是零。

---

## 2. 現況盤點

### 2.1 已經可用而且夠好的部分

| 面向 | 現況 |
|---|---|
| AEC 狀態快照 | `AecDebugStatus` 10 欄，零 per-hop 成本（只讀引擎本來就每 hop 維護的欄位） |
| NR 狀態快照 | `MmseLsaDebugStatus` 5 欄 |
| 4ch 計數器 | duty census、realign warm/soft、pending delay candidate、far-FFT 實跑次數、post split floor（live vs target） |
| 記憶體 | `aec_get_mem_breakdown()` / `audio_pipeline_get_mem_breakdown()` |
| 錯誤分級 | 4ch core 的 `FourAecNrResStatus` 四值（唯一有語意分級的） |
| Finite guards | 4ch 的輸入與輸出各一道；兩條 ULCNet 的 model 輸出 guard（含先填 `NAN` 的 FULL-WRITE CONTRACT） |

### 2.2 十個已驗證的盲點

| # | 盲點 | 證據 |
|---|---|---|
| **G1** | **`assert()` 在出貨板上是活的**。四個 Makefile 沒有任何一個設 `-DNDEBUG`；`suppression_gain.c` 的 table-length assert **沒有 release-path fallback**，失敗即 `abort()`——沒有回傳碼、沒有 log、沒有 hook 可攔 | 全樹 grep `-DNDEBUG` = 0 命中 |
| **G2** | **層級式 log 是死碼**。`AEC_DEBUG_LOG` 在整個 `src/` 有 **0 個呼叫點**（唯一命中是一句註解） | grep `AEC_DEBUG_LOG src/` |
| **G3** | **per-hop CSV trace 在四條 pipeline 全部不可達**。唯一填寫點要求 `cfg.enable_res == 1` 且走 `aec_process()`；四條 pipeline **全部**無條件設 `enable_res = 0` | `audio_pipeline.c:358`、`audio_pipeline_ulcnet.c:229`、`4aec_nr_res.c:471` |
| **G4** | **self-reset 不計數，而且會抹掉證據**。`DSP_ERROR` 路徑內部呼叫 `reset()`，而 `reset()` 把 realign counters / duty census 全部歸零 → 事後完全看不出發生過 | `4aec_nr_res.c` 的 reset 實作 |
| **G5** | **delay lock 遺失不可見**。`delay_state` 與 `generation` 每 hop 可讀，但 `AecDebugStatus` **兩者都沒有** → 只 poll 快照的整合者看不到 lock 的丟失與重獲 | `aec.h` 的 `AecDebugStatus` 欄位表 |
| **G6** | **mis-lock 連代理指標都沒有**。`delay_updates` 是更新總數，**不區分**「同一值重複確認」與「跳到新值」 | — |
| **G7** | **realign 第三種落點不計數**。只累加 `outcome==1`（warm）與 `==0`（soft）；`-1`（wiring fault）不計 | `4aec_nr_res.c` realign sweep |
| **G8** | **沒有任何 frame counter 的 public accessor**。四條 pipeline 都答不出「處理了幾個 hop」，遑論 xrun | grep 四個 header |
| **G9** | **init 失敗一個字都不說**。4ch core `init_ex()` 的 8 項檢查任一失敗都靜默回 NULL；mono 有逐欄位 stderr，但 `NO_STDIO=1`（正是板端組態）一開就沒了 | `4aec_nr_res.h` 自承「not implemented here yet」 |
| **G10** | **NN fail-open 靜默且不可區分**。三種 fallback（`infer==NULL` / 回非 0 / 輸出非有限）走同一條 else 分支、無計數 → 「NPU 掛了」與「偶發 NaN」外部無法分辨 | 兩條 ULCNet 的 infer 分支 |

另有三個次要盲點：backward quarantine 開火不可見；`far_spec_shared_hop` 退化到昂貴路徑不可見（provenance 函式只在 internal header）；mono pipeline 完全沒有 finite 守衛（輸入輸出都不檢）。

---

## 3. 提案

以下皆為**提案，尚未實作**。分層排序，愈前面 CP 值愈高。

### T0 — 先修現有的洞（幾乎不新增 API）

| 項目 | 動作 | 成本 |
|---|---|---|
| **T0.1（最高優先，屬安全性不只 debug）** | 為 release build 設 `-DNDEBUG`，**並且**替 `suppression_gain.c` 那個沒有 fallback 的 assert 補上 release-path guard（其餘兩處已有）。或者反過來，把該 assert 改成建構期拒絕（回 NULL）——它檢查的是 init 期的一致性，本來就不該用 abort 表達 | 零 per-hop |
| **T0.2** | `AEC_DEBUG_LOG` 整套死碼：**刪掉**。留著只會讓人以為 library 有 log 能力 | 負值（減碼） |
| **T0.3（回收最大）** | 把 trace 的 gate 從 `cfg.enable_res` 改成 `cfg.enable_res \|\| cfg.return_res_context`（與 `last_erle_windowed` 的更新條件一致），並讓 `aec_process_context()` 也能觸發 | 一個 NULL 測試 |

T0.3 之所以關鍵：`dominant_nearend`、`saturated_echo`、`usable_linear`、
`fullband_erle`、`gain_mean` 這五個量**沒有第二個出口**，改完 gate 之後
它們才在產品組態下拿得到。

### T1 — 擴充結構化快照（欄位都已在引擎內，零 per-hop 成本）

`AecDebugStatus` 建議新增：

| 欄位 | 來源 | 為什麼要 |
|---|---|---|
| `delay_state` | 已存在於 `AecLinearContext` | **解 G5**。lock 遺失的直接訊號 |
| `delay_generation` | 同上（saturating counter） | **解 G5**。只比較 `delay_samples` 會漏掉 A→B→A 的暫態 |
| `saturation_level` | 已存在於 `AecResContext` | 大音量下 ERLE 掉的第一嫌疑犯 |
| `dt_indicator` | 同上 | 判斷「近端被削」是不是 DT 誤判 |
| `erl_estimate` | 同上 | 回聲路徑增益的合理性檢查 |
| `frames_processed` | **新增一個 `++`** | **解 G8**。所有比率型判讀的分母 |

`MmseLsaDebugStatus` 建議新增 `frames_processed`（header 自承目前被拿掉了）
與一個 `scene_change_count`（MCRA 已經在內部維護）。

### T2 — 事件計數器，並區分兩級生命週期

**這是本提案唯一的新概念，也是 G4 的正解。**

現有的 counter 全是 **session 級**（`reset()` 會清）。但內部 self-reset 正是
我們最想知道的事件——它一發生就把自己的證據抹掉了。因此需要第二級：

| 級別 | 何時清零 | 用途 |
|---|---|---|
| **session** | `reset()` 與 init | 現有的 duty census、realign counters 屬此級，維持不變 |
| **lifetime** | **只有 init 清，`reset()` 不清** | 跨越 self-reset 的事件史 |

建議的 lifetime counters（全部是非熱路徑上的 `++`，`unsigned long long`）：

| 計數器 | 解決 |
|---|---|
| `self_reset_count` | **G4** |
| `delay_lock_lost_count` / `delay_relock_count` | **G5** |
| `delay_jump_count`、`delay_max_jump_samples` | **G6**（mis-lock 的代理指標：`delay_updates` 做不到，因為它不分「重複確認」與「跳值」） |
| `realign_reject_count` | **G7** |
| `quarantine_block_count` | backward quarantine 開火 |
| `nonfinite_input_frames` / `nonfinite_output_frames` | 現有 guard 只擋不數 |
| `saturation_frames` | 飽和發生率 |
| `nn_identity_frames_no_model` / `_infer_failed` / `_nonfinite` | **G10**——三種原因**必須分開數**，否則「NPU 掛了」與「偶發 NaN」仍然無法區分 |
| `nn_reprime_frames` | 邊界重啟頻率 |
| `far_spec_shared_degraded_hops` | 掉回昂貴路徑 |

### T3 — init 失敗要能說話（不需 stdio）

**解 G9。** 加 `*_last_init_error()` 回傳一個 enum（哪一項檢查失敗、哪個欄位），
純 enum、無字串、無 stdio → **release 存活**。

這比「加 stderr」正確：板端組態本來就是 `NO_STDIO=1`，stderr 方案在出貨時必然歸零。

### T4 — 把 trace 從 `FILE*` 改成 caller 提供的 ring buffer

現在 `AecDebugTraceRow` 只能寫進 `FILE*`，所以 `NO_STDIO=1` 直接消失。
改成：caller 提供一塊 `AecDebugTraceRow[N]` 的 ring，library 只負責寫入與推進索引；
要不要落盤、怎麼落盤完全是 caller 的事。

好處：出貨板可以常時開著一個 256-entry 的 ring（約 14 KB），**故障當下 dump 最近 N 個 hop** ——
這是唯一能回答「出事那一刻演算法在想什麼」的機制。
成本：一個 NULL 測試 + 一次 struct 寫入，且不引入任何 stdio 相依。

### T5 — 讓「沒開 timing」與「很快」可區分

`aec_get_last_timing()` 回全 0 目前有兩種意思，而函式不會告訴你是哪一種。
提案：在 `AecStageTiming` 加一個 `enabled` 欄位（或導出
`*_stage_timing_available()`），wrapper 層的兩個宏（`AUDIO_PIPELINE_STAGE_TIMING`、
`FOUR_AEC_NR_RES_STAGE_TIMING`）目前是 `.c` 內部宏、consumer 測不到，也一併導出。

### T6 — 兩條 ULCNet pipeline 補上觀測面

**解 G3 的 wrapper 版本。** 兩者目前幾乎沒有暴露任何東西：

- mono ULCNet：只有 `_get_aec()`。補 `_get_last_timing()`、`_get_mem_breakdown()`。
- 4ch ULCNet：**沒有暴露 core handle** → core 的 duty census、realign counters、
  timing、post split floor 對整合者全部不可達。補一個 `_get_core()` 就全部打通，
  成本是一行 getter。

---

## 4. 成本總表

| 提案 | per-hop 成本 | release 存活 | 解決 |
|---|---|---|---|
| T0.1 assert | 零 | — | 安全性 |
| T0.2 刪死碼 | 負 | — | G2 |
| T0.3 trace gate | 一個 NULL 測試 | 需搭配 T4 | G3 |
| T1 快照擴充 | **零**（欄位已存在）+ 一個 `++` | ✅ | G5 G8 |
| T2 lifetime counters | 數個非熱路徑 `++` | ✅ | G4 G6 G7 G10 |
| T3 init error enum | 零 | ✅ | G9 |
| T4 ring trace | 一個 NULL 測試 + 一次 struct 寫入 | ✅ | G3 |
| T5 timing 可偵測 | 零 | ✅ | 誤判 |
| T6 ULCNet accessor | 零 | ✅ | G3(wrapper) |

**建議順序**：T0.1（安全性）→ T1 + T3（零成本、直接可用）→ T2 → T0.3 + T4（配套）→ T5 → T6。

---

## 5. 明確不建議做的事

- **不要加 printf 型 logging**：出貨組態下必然是零，投資報酬為負。
- **不要為診斷新增 per-hop 浮點運算**：現有快照之所以零成本，是因為它只讀引擎
  本來就維護的欄位。任何「為了 debug 才算」的量都會進熱路徑。
- **不要把診斷欄位塞進既有 struct 的中間**：`AecStageTiming` 的 `steering_us`
  是後加在尾端的，前四個 offset 才得以不變。新欄位一律加在尾端，並依既有慣例
  bump 對應的 layout version。
- **不要用字串**：enum + 計數器就夠，字串會把 stdio 依賴帶回來。
