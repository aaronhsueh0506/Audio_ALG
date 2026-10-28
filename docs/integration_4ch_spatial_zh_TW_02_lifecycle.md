> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_01_quick_start.md) ｜ [下一份](integration_4ch_spatial_zh_TW_03_config.md) <!-- split-nav -->
## 3. Lifecycle

三塊記憶體各管各的：**應用端 flow 區塊**（四條 lane、`DelayAec` 結構與它的 pool、mcab 自己的 pool，各自 16-byte 對齊，
`mcab_sample_flow_*` 就是把它們放在一塊）、**helper pool**（SRP、BF、beam 頻譜與權重）、**你的後級**（RES、NR、合成與暫存）。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `mcab_get_mem_requirements(&cfg, &req)`；每條 lane `aec_get_mem_size(&lane)`；MATCHED 時 `delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters)`；helper `mcab_sample_beam_get_mem_size(&cfg, &bp, &bytes)`（`mcab_sample_flow_get_mem_size()` 一次查前三項的合計） | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | `mcab_get_mem_requirements()` 回 `-1`：`out` 為 NULL；`cfg` 未通過 flow 驗證（[§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-mcabconfigflow-參數與-lane-的-aecconfig)）；編譯 `mcab.c` 時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"`。`aec_get_mem_size()`／`delay_aec_get_mem_size()` 回 0 = 設定無效。`mcab_sample_beam_get_mem_size()` 回 `-1`：`cfg` 無效、`bp` 未通過 [§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄)～§4.5 的驗證、SRP 或 BF 回報 size 0，或 `bf_lambda` 依生效 interval retime 後下溢為 0 |
| 2 | allocate | 你自己的配置器：mcab pool 取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體；lane、estimator、helper 各自 16-byte 對齊 | 拿到大小之後 | pool **不需要**預先清零（poison pattern 可接受）；`mcab_sample_beam_init()` 自己會把 helper 的整塊 memset 成 0 |
| 3a | init：lane | `aec_init(mem, bytes, &lane)` ×4，`lane` 來自 `mcab_lane_config(&cfg, &lane)`，或 `aec_config_from_level()` 加 `mcab_apply_lane_contract(&cfg, &lane)` | 四條必須是不同物件 | `NULL`：config 無效、pool 過小或未對齊 |
| 3b | init：estimator | 僅 `mcab_delay_estimator_config(&cfg, &dc)` 的 `dc.required`（MATCHED）：`delay_aec_init(&delay, mem, bytes, dc.sample_rate, dc.hop_size, dc.num_filters)` | 非 MATCHED 不建；`McabLibs.delay` 必須是 NULL | 回非 0 |
| 3c | init：mcab | `mcab_init_ex(pool, req.bytes, &cfg, &libs, &req)` | lane 與 estimator 建好之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符、pool 未對齊或太小、config 無效、[§5.7](integration_4ch_spatial_zh_TW_04_errors.md#57-mcab_init_ex-驗證什麼不能驗證什麼) 的物件驗證任一不符。**沒有任何診斷輸出**（見 [§5.4](integration_4ch_spatial_zh_TW_04_errors.md#54-診斷輸出去向)） |
| 3d | init：helper | `mcab_sample_beam_init(mem, bytes, &cfg, &bp, flow)` | `flow` 建好之後；`cfg` 必須與建 flow 的是同一份（至少 `sample_rate`、`fft_size`、`enable_post` 一致） | 回傳 `NULL`：`bp` 無效、pool 未對齊或太小、flow 的 hop／bin／sample rate／`enable_post` 與 `cfg` 不符、SRP 或 BF init 失敗 |
| 4a | process | `mcab_sample_beam_process_front()` → `mcab_sample_beam_process_beam()`；或 `_process_with_activity()`；或 `_process()` | 每收到一個 hop 呼叫一次 | `-1` / `-2` / `-3`，見 [§5.2](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4b | fuse（可選） | `mcab_sample_beam_fuse(b, &frame)` | 同一 hop 的 beam 之後、下一次 `process_front()` 之前 | `-1` / `-2` / `-3`，見 [§5.2](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4c | 你的後級 | `mcab_sample_res_input()` + `aec_res_process()` → `mcab_sample_anr_gain()` → `sk_min_f32()` → `sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`，或 NN stage | fuse（或 beam）之後 | 由你負責；`frame.g_res_ctx` 為 NULL，RES 增益來自你自己的 `AecRes`。任何一步被拒，你必須把 flow／helper 與你的後級（含 `aec_res_reset()`）**一起** reset（§3.1） |
| 5 | reset | `mcab_sample_beam_reset(b)`，加上你的後級的 reset | 換回聲路徑、換場景、切換到不相關的另一條串流時 | 不會失敗。`b == NULL` 或已 destroy 時是 no-op |
| 6 | destroy | `mcab_sample_beam_destroy(b)` → `mcab_destroy(flow)` → 四次 `aec_destroy(lane)` | 不再處理任何 hop 之後、釋放記憶體之前；**順序不可顛倒**（mcab 借用 lane 與 estimator，helper 借用 flow） | 不會失敗。NULL 安全，helper 與 caller-pool 的 mcab 冪等；`mcab_create()` 的 handle 只能 destroy 一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`mcab_sample_flow_create(&flow, &cfg)` 把第 1～3c 步合成一步（一塊 heap，`mcab_sample_flow_destroy()` 依序 `mcab_destroy()`、lane 後釋放）；
`mcab_create(&cfg, &libs)` 只替 mcab 自己的 pool 配置 heap（`libs` 裡的物件仍歸你）。helper 沒有 heap 版，pool 自己配。

`mcab_init(pool, bytes, &cfg, &libs)` 等同 `mcab_init_ex(pool, bytes, &cfg, &libs, NULL)` —— 不做 descriptor 檢查。

lane 與 estimator 在 `mcab_init_ex()` 之前必須是**新建的**，或已經 reset（`aec_reset()`／`delay_aec_reset()`，或前一個 flow 的 `mcab_reset()`）：
mcab 看不出一個物件已經跑過音訊，用過的 estimator 會把舊的 delay 交給新的 flow。flow 掛著的期間，應用也不可以自己對這些 lane／estimator
呼叫處理或 reset 函式，flow 的 admission 與參考環會和它們脫節，mcab 偵測不到。

### 3.1 `reset()` 到底重置了什麼，以及失敗時整合者要做什麼

`mcab_sample_beam_reset()` 重置：

* flow（`mcab_reset()`）：共用 estimator（`delay_aec_reset()`）、四條 lane（各自 `aec_reset()`）、參考環與 aligned far 清零、
  delay admission／quarantine／duty 狀態與計數器歸零、front 有效旗標
* SRP-PHAT 狀態
* BF 狀態
* helper 的內建 VAD 狀態（`noise_power` 回到門檻值、hangover 歸零）
* `frame_index` 與 per-hop phase 歸零

`mcab_reset()` 不清 lane 的 buffer：reset 之前取得的 `McabFront`／frame 指標會讀到舊內容（不是 fault），但 `mcab_fuse()` 在新的 front 之前一律拒絕。
reset **不會**重置你的後級（`AecRes`、NR、CNG rng、合成、NN stage），也**不會**重新驗證 config，
更不會重新計算 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) 那三個「生效值」——那三個是在 init 時算好的，`reset()` 不動它們。

**失敗語意是整合者的責任。** `MCAB_DSP_ERROR`（flow 已自己 reset，你的後級沒有）、`mcab_sample_beam_fuse()` 因 flow 的拒絕而回非 `MCAB_OK`、非 `MCAB_SEQUENCE_ERROR`（helper 已整個 reset；NULL 引數與 `enable_post = 0` 的 `-1` 是 helper 自己的引數拒絕，什麼都不改）、
你的 `aec_res_process()`／`mcab_sample_anr_gain()` 回 `NULL`，或合成輸出不是有限值，你要**同時**呼叫 `mcab_sample_beam_reset()`、`aec_res_reset()`、`anr_reset()`、`ac_istft_reset()`，
並把 comfort-noise RNG 設回 `MCAB_SAMPLE_CNG_RNG_SEED`，以及用到 NN stage 時的 `ulcnet_nn_stage_reset()`。
漏掉其中一個，系統就不等於新建的實例。front／beam 階段的 `MCAB_INVALID_ARGUMENT` 與 `MCAB_SEQUENCE_ERROR` 不改動 flow 與 helper 的自適應狀態（`process_front()` 因非有限輸入被拒時另外不再留待處理的 front，[§5.2](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單)），
修正輸入或呼叫順序後重送同一個 hop 即可；本手冊的範例為求簡單，對任何非 `MCAB_OK` 一律一起 reset。DFN2 的 stage 不因為 AEC 的 delay change 而 reset（見
`pipelines/dfn/dfn_res_stage.h`）。

### 3.2 執行期唯讀存取

形狀 accessor（`flow == NULL` 或已 destroy 時全部回 `-1`）：

| 呼叫 | 回傳 |
|---|---|
| `mcab_hop_size(flow)` | 每次 process 要餵幾個 sample（per channel） |
| `mcab_fft_size(flow)` | FFT 長度（= frame 長度） |
| `mcab_n_freqs(flow)` | 頻域 bin 數 |
| `mcab_sample_rate(flow)` | 實際生效的取樣率 |
| `mcab_enable_post(flow)` | `cfg.enable_post`：1／0 |

AEC、DOA、BF，以及你串的 RES 與 NR，全部**共用同一組格點**，`frame == fft_size`、`hop == fft_size / 2`。
你不能分別指定互相矛盾的 frame / hop / FFT：helper 由 flow 的格點導出交給 SRP 與 BF 的設定
（`mcab_sample_spatial_config()`），`tests/test_mcab_sample_beam.c` 的 `run_grid` 檢查 SRP／BF 實際收到的 `sr`、`NFFT`、`F`、`M`、
`n_freqs` 與 BF 的 adapt_interval 都在 flow 的格點上（16 kHz 預設下 hop 128 / fft 256 / n_freqs 129）。

結構稽核 accessor（`flow == NULL` 或已 destroy 時回 `0`）：

| 呼叫 | 有效 handle 的回傳 |
|---|---|
| `mcab_matched_filter_count(flow)` | `1`（`MATCHED`）；`FIXED`／`EXTERNAL_ALIGNED` 時 `0` |
| `mcab_linear_aec_count(flow)` | `4` |

結構稽核計數只有上面兩個；RES、NR 與合成由你自己決定，mcab 不持有它們。另有唯讀計數器
（`mcab_far_fft_real_compute_count()`、`mcab_realign_warm_lane_count()`／`_soft_lane_count()`、`mcab_pending_delay_candidate()`、
`mcab_duty_hops_total()`／`_run()`），語意見 `mcab.h`。

生效值 accessor（見 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值)）：
`mcab_sample_beam_bf_effective_adapt_interval(b)`、`mcab_sample_beam_bf_lambda(b)`（`b == NULL` 或已 destroy 時回 `-1`／NaN）。

診斷：`mcab_get_last_timing(flow, &t)`（最近一個 hop 的各階段耗時，`McabLastTiming`；`flow` 為 NULL 或已 destroy 時把 `t` 清零，`t` 不得為 NULL；
預設全 0，要 `-DMCAB_STAGE_TIMING=1` 與 `-DAEC_STAGE_TIMING=1`，`make PROFILE=1` 兩者都開）。

建置資訊：`mcab_sample_beam_spatial_backend()` —— **不吃 handle**，回傳一個
`const char*`（永不為 NULL），表示 spatial kernel 實際編進去的是哪一種實作
（NEON 或 scalar）。純資訊用途，可寫進你的開機 log。

### 3.3 執行期強度控制

mcab 與 helper **沒有**強度 setter 函式，也不持有 `AecRes`：四條 lane 只算線性殘留，RES 與 NR 都是你串的後級。
你對自己 init 的物件直接呼叫函式庫的 setter（完整契約見 mcab 手冊 [`integration_4ch_core_zh_TW.md`](integration_4ch_core_zh_TW.md) 的執行期強度控制一節）：

```c
aec_res_set_level(res, AEC_LEVEL_AGGRESSIVE, 100.0f);    /* res 是你自己的 AecRes；回 0 成功 */
anr_set_level(nr, ANR_LEVEL_MILD);                       /* 函式庫的 setter；nr 是你自己的降噪器 */
float live, target;                               /* 線性功率 */
if (aec_res_split_floor(res, &live, &target) == 0 && live == target) {
    /* ramp 已經走完 */
}
```

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名與整數相同。舊名 `AecPreset`／`AEC_PRESET_*`、`AnrMode`／`ANR_MODE_*` 仍保留在公開 header（同值），舊函式名已移除。

這裡只重述整合時最容易踩到的一點：

> **不要自己對四條 lane 迴圈呼叫 `aec_set_level()`。** 四條 lane 都經 `mcab_apply_lane_contract()` 蓋上
> `enable_res = 0`、`return_res_context = 1` 與 `spatial_linear_context`，永遠不會走到自己的 suppression-gain 路徑，
> 它們的地板什麼都不塑形。真正乘上輸出的 gain 來自你串的**唯一一個**
> post-beam `AecRes`；同樣地，NR 也只有一個共用實例，不是每條 lane 一個。
> 上面兩個 setter 針對的就是那兩個共用實例。`mcab_sample_beam_reset()` 不碰它們，
> 你自己呼叫的 `aec_res_reset()` 不會丟掉用 `aec_res_set_level()` 設的 level。

`ramp_ms == 0` 代表下一個 hop 就套用（**不是錯誤**），`> 0` 則以 dB 為單位線性走
過去，上限 60 秒。兩個 setter 都不是重啟：AEC、SRP、BF、NR 的狀態全部繼續跑；
level 的改動撐得過該 stage 的 reset，ramp 進度不會。**在兩個 hop 之間
呼叫、與 process 序列化；非 thread-safe。**

A/B 量測時：far-active 地板只在 **far-active 且非 double-talk** 的 hop 上生效
（DT 地板：mild/balanced 共用 -20 dB，aggressive 為 -30 dB；latch 觸發前用的是 far-silent 地板），而同一個 gain
還決定 comfort noise 量，所以整段錄音的平均值移動幅度會小於 dB 落差所暗示的量。
細節與建議的比較方式見 mcab 手冊的執行期強度控制一節。

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_01_quick_start.md) ｜ [下一份](integration_4ch_spatial_zh_TW_03_config.md) <!-- split-nav -->
