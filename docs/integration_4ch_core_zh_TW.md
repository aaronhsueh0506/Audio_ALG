# 四麥克風前端整合手冊（`libaudio_pipeline_mcab.a` / `mcab.h`）

適用對象：**已經擁有自己的 beamformer（SRP-PHAT / GSC 或任何等效方案）**，
想要四路 linear AEC 前端，並自己串接後級頻域 stage（RES / NR / CNG / ISTFT，或自己的 NN）的函式庫使用者。

`mcab` 是 **flow 層**，不是函式庫：四條 linear AEC lane（`Aec`）和 MATCHED 模式的共用 delay estimator（`DelayAec`）
都由**應用程式**用 libaec 自己的 `get_mem_size`／`init` 建立，再以指標交給 mcab；mcab 只擁有自己小 pool 裡的 flow 狀態
（共用 delay 的 admission 狀態機、reference ring、對齊後的 far、fuse 緩衝），從不配置、釋放或以 destroy 取代 reset 這些物件。
模組裡**沒有 beamformer**，RES 也在模組之外。

如果你**沒有**自己的 beamformer、想要含 direction finding 與 beamformer 的線性 AEC + BF，
請看 `docs/integration_4ch_spatial_zh_TW.md`：那是 mcab 前端加上 sample 層 helper（`mcab_sample_beam.h`）；
helper 是範例碼，不在 `libaudio_pipeline_mcab.a` 內。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。
頻域 stage 串接的總覽（任何 pipeline 都適用）見 `pipelines/README_01_latency_call_flow.md` 的 Call flow。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`；
> 從 `pipelines/4ch_aec_bf_nr_res/` 出發的相對路徑是 `../../../audiolib/common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 [§4.6](integration_4ch_core_zh_TW_03_config.md#46-實測記憶體僅供量級參考務必自己重查)）。

---
## 目錄 <!-- split-toc -->
- [四麥克風前端整合手冊（`libaudio_pipeline_mcab.a` / `mcab.h`）](#四麥克風前端整合手冊libaudio_pipeline_mcaba--mcabh) <!-- split-toc -->
  - [0. 速覽：流程時序圖、公開／內部函式、外部／內部參數](#0-速覽流程時序圖公開內部函式外部內部參數) <!-- split-toc -->
    - [0.1 流程時序圖](#01-流程時序圖) <!-- split-toc -->
    - [0.2 公開函式與內部函式](#02-公開函式與內部函式) <!-- split-toc -->
    - [0.3 參數面：外部與內部](#03-參數面外部與內部) <!-- split-toc -->
  - [1. 這個 flow 層是什麼、你連結什麼、你 `#include` 什麼](integration_4ch_core_zh_TW_01_library_quickstart.md#1-這個-flow-層是什麼你連結什麼你-include-什麼) <!-- split-toc -->
    - [1.1 這是什麼](integration_4ch_core_zh_TW_01_library_quickstart.md#11-這是什麼) <!-- split-toc -->
    - [1.2 Header closure](integration_4ch_core_zh_TW_01_library_quickstart.md#12-header-closure) <!-- split-toc -->
    - [1.3 `Complex` type 從哪來](integration_4ch_core_zh_TW_01_library_quickstart.md#13-complex-type-從哪來) <!-- split-toc -->
    - [1.4 編譯需要的 `-I` 清單（consumer 端最小集合）](integration_4ch_core_zh_TW_01_library_quickstart.md#14-編譯需要的--i-清單consumer-端最小集合) <!-- split-toc -->
    - [1.5 Archive link order](integration_4ch_core_zh_TW_01_library_quickstart.md#15-archive-link-order) <!-- split-toc -->
  - [2. Quick start](integration_4ch_core_zh_TW_01_library_quickstart.md#2-quick-start) <!-- split-toc -->
  - [3. Lifecycle](integration_4ch_core_zh_TW_02_lifecycle.md#3-lifecycle) <!-- split-toc -->
    - [3.1 執行期唯讀存取](integration_4ch_core_zh_TW_02_lifecycle.md#31-執行期唯讀存取) <!-- split-toc -->
    - [3.1b 逐階段耗時（診斷用）](integration_4ch_core_zh_TW_02_lifecycle.md#31b-逐階段耗時診斷用) <!-- split-toc -->
    - [3.2 執行期強度控制](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制) <!-- split-toc -->
  - [4. Config 完整參考](integration_4ch_core_zh_TW_03_config.md#4-config-完整參考) <!-- split-toc -->
    - [4.1 逐欄位](integration_4ch_core_zh_TW_03_config.md#41-逐欄位) <!-- split-toc -->
    - [4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）](integration_4ch_core_zh_TW_03_config.md#42-grid由-sample_rate--fft_size-唯一決定) <!-- split-toc -->
    - [4.3 Buffer 形狀（每次 `mcab_process_front` / `mcab_fuse`）](integration_4ch_core_zh_TW_03_config.md#43-buffer-形狀每次-mcab_process_front--mcab_fuse) <!-- split-toc -->
    - [4.4 `McabFront` 與 `McabFrame` 各欄位](integration_4ch_core_zh_TW_03_config.md#44-mcabfront-與-mcabframe-各欄位) <!-- split-toc -->
    - [4.5 哪些欄位會改變記憶體用量](integration_4ch_core_zh_TW_03_config.md#45-哪些欄位會改變記憶體用量) <!-- split-toc -->
    - [4.6 實測記憶體（僅供量級參考，務必自己重查）](integration_4ch_core_zh_TW_03_config.md#46-實測記憶體僅供量級參考務必自己重查) <!-- split-toc -->
  - [5. 錯誤處理](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#5-錯誤處理) <!-- split-toc -->
    - [5.1 front → fuse 的次序規則](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#51-front--fuse-的次序規則) <!-- split-toc -->
    - [5.2 `mcab_process_front()` 的完整拒絕清單（依實際檢查順序）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#52-mcab_process_front-的完整拒絕清單依實際檢查順序) <!-- split-toc -->
    - [5.3 `mcab_fuse()` 的完整拒絕清單（依實際檢查順序）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#53-mcab_fuse-的完整拒絕清單依實際檢查順序) <!-- split-toc -->
    - [5.4 錯誤語意總表](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#54-錯誤語意總表) <!-- split-toc -->
    - [5.5 診斷輸出去向](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#55-診斷輸出去向) <!-- split-toc -->
    - [5.6 `mcab_destroy()` 的兩種語意](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#56-mcab_destroy-的兩種語意) <!-- split-toc -->
  - [6. 出貨內容 vs 範例](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#6-出貨內容-vs-範例) <!-- split-toc -->
    - [6.1 函式庫（會進產品）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#61-函式庫會進產品) <!-- split-toc -->
    - [6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#62-參考執行檔全部都有-main沒有任何一個會進產品) <!-- split-toc -->
    - [6.3 其他（不出貨）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#63-其他不出貨) <!-- split-toc -->
  - [7. 整合檢查清單](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#7-整合檢查清單) <!-- split-toc -->
  - [8. 版本與相容性](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#8-版本與相容性) <!-- split-toc -->
    - [8.1 `McabMemReq`](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#81-mcabmemreq) <!-- split-toc -->
    - [8.2 `mcab_init_ex()` 的 8 項檢查](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#82-mcab_init_ex-的-8-項檢查) <!-- split-toc -->
    - [8.3 序列化](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#83-序列化) <!-- split-toc -->
    - [8.4 相容性規則（給整合者的三條）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#84-相容性規則給整合者的三條) <!-- split-toc -->

## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖

生命線：應用、你建立並擁有的 lane（`Aec` ×4）與 estimator（`DelayAec`）、`Mcab`（本手冊的 flow）、你的 beamformer、
`AecRes`（你自己建）、`AnrDenoiser`、`AcIstft`。
flow 沒有 RES：`frame.g_res_ctx` 恆為 NULL，RES 增益由你的 `AecRes` 經 `mcab_sample_res_input()` 填 `AecResInput`、再呼叫 `aec_res_process()` 算出。

```
應用           lane Aec×4 / DelayAec        Mcab（flow）       你的 beamformer   AecRes(你建)  AnrDenoiser  AcIstft
 │                 │（你建立並擁有）            │                    │               │             │            │
 │══ 初始化（一次；每次 init 前重查，不快取）════════════════════════════════════════
 │ mcab_default_config(sr)  → cfg（不驗證，只填值；覆寫你要改的欄位）
 │ mcab_lane_config(&cfg,&lane)  或  aec_config_from_level(&lane,…) ＋ mcab_apply_lane_contract(&cfg,&lane)
 │ aec_get_mem_size(&lane) → 配置 → aec_init() ×4 ───►│（四個不同的 Aec；新建或已 reset）
 │ mcab_delay_estimator_config(&cfg,&dc)；dc.required（只在 MATCHED）時：
 │   delay_aec_get_mem_size → 配置 → delay_aec_init(&delay,…) ──►│（DelayAec 結構本身也是你的儲存空間）
 │ mcab_get_mem_requirements(&cfg,&req) ───────────────────►│    → req.bytes、req.alignment（0 成功／-1 失敗）
 │ 配置 req.bytes（16-byte 對齊）
 │ mcab_init_ex(pool,req.bytes,&cfg,&libs,&req) ───────────►│    → handle（NULL = descriptor 8 項、config、lane／estimator 檢查之一不符；無診斷輸出）
 │ 後級各自 mem_size + init：aec_config_from_level(&res_cfg,…) ＋ res_cfg.fft_size = mcab_fft_size(p)
 │   aec_res_get_mem_size/aec_res_init ──────────────────────────────────────────►│
 │   anr_config_for_level_grid/anr_get_mem_size/anr_init ────────────────────────────────────►│
 │   ac_stft_config_default/ac_istft_get_mem_size/ac_istft_init ────────────────────────────────────────►│
 │══ 每個 hop（mics[hop*4] interleaved、ref[hop]）══════════════════════════
 │ mcab_process_front(p,mics,ref,&front) ───────────────────►│    → MCAB_OK / -1 / -3；front 為借用視圖
 │   front.delay.changed ≠ 0 ⇒ 先清掉你的 STFT/OLA 歷史（flow 只 realign 四條 lane）
 │ front.linear_spectra[ch] ───────────────────────────────────────────►│ 算 effective weights Complex[4][n_freqs]（channel-major）
 │ mcab_fuse(p,weights,beamformed_error|NULL,&frame) ───────►│    → 0 / -1 / -2 / -3；frame.g_res_ctx = NULL
 │ mcab_sample_res_input(&frame,&in); aec_res_process(res,&in) ───────────────────►│ G_res（NULL = 輸入指標為 NULL）
 │ mcab_sample_anr_gain(nr,&frame,extra_noise,1) ──────────────────────────────────────────►│ G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,G_res,n); sk_capply_gain_f32(spec,frame.e,total,n)
 │ aec_cng_fill(&rng,spec,frame.cn,G_res,G_nr,MCAB_SAMPLE_CNG_NR_GAIN_FLOOR,n)
 │ ac_istft_push(istft,spec,out) ─────────────────────────────────────────────────────────────►│ hop 個樣本
 │ skn_all_finite_f32(out,hop)
 │══ 任一步被拒（-3 時 flow 已自己 reset），或輸出非有限 ══════════════════════════
 │ mcab_reset(p)（含四條 lane 與 estimator 自己的 reset）; aec_res_reset(res); anr_reset(nr); ac_istft_reset(istft); rng = MCAB_SAMPLE_CNG_RNG_SEED
 │══ 強度（hop 之間）：aec_res_set_level(res,level,ramp_ms)；anr_set_level(nr,level) ═══
 │══ 結束：mcab_destroy(p) → aec_destroy(lane) ×4；anr_destroy(nr)；之後才釋放記憶體 ═══
```

延遲泳道（橫軸 = hop）：

```
hop 序號                 t         t+1        t+2
四路 mic + ref           in[t]     in[t+1]    in[t+2]
front → BF → fuse → 後級  處理 in[t] 處理 in[t+1] 處理 in[t+2]   （flow 與後級同一個 hop 內完成）
輸出 out                 y[t-1]    y[t]       y[t+1]            （你的 ac_istft_push 多 1 hop）
共用 aligner             delay.delay_samples 隨 delay.changed 在 hop 邊界切換（該 hop 的對齊已用新值）；
                         changed ≠ 0 的那個 hop：處理該 frame 之前先清掉你的 STFT/OLA 歷史
```

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開，`mcab.h`）**

| 階段 | 函式 | 備註 |
|---|---|---|
| 設定 | `mcab_default_config()`、`mcab_apply_lane_contract()`、`mcab_lane_config()`、`mcab_delay_estimator_config()` | 不配置記憶體。`mcab_default_config()` 只填值；其餘三個回 `0`／`-1`（無效 config） |
| 你建立的 lib 物件 | `aec_get_mem_size()`／`aec_init()`（×4）、`delay_aec_get_mem_size()`／`delay_aec_init()`（只在 MATCHED）、結束時 `aec_destroy()` | libaec 的 API；mcab 從不呼叫它們的 init／destroy |
| 建立 | `mcab_get_mem_requirements()`、`mcab_init_ex()`（或 `mcab_init()`）；`mcab_create()`（heap 便利路徑） | 描述子 32 bytes，8 項檢查；`mcab_create()` 只配置 mcab 自己的那一塊 pool |
| 每 hop | `mcab_process_front()`、`mcab_fuse()` | 之間是你的 beamformer |
| 重置／收尾 | `mcab_reset()`、`mcab_destroy()` | reset 會作廢 front 視圖與 `McabFrame`，並呼叫 lane 與 estimator 自己的 reset |
| 形狀 | `mcab_hop_size()`、`_fft_size()`、`_n_freqs()`、`_sample_rate()`、`_enable_post()` | 無效 handle 回 `-1` |
| 記憶體預算 | `mcab_get_mem_breakdown()` | `control_bytes`／`buffer_bytes`／`total_bytes`，只含 mcab 自己的 pool |
| 診斷（唯讀） | `mcab_get_last_timing()`、`_duty_hops_total()`、`_duty_hops_run()`、`_realign_warm_lane_count()`、`_realign_soft_lane_count()`、`_pending_delay_candidate()` | 時序欄位預設全 0（`-DMCAB_STAGE_TIMING=1` 與 `-DAEC_STAGE_TIMING=1`） |
| 結構稽核（唯讀） | `mcab_matched_filter_count()`、`_linear_aec_count()`、`_far_fft_real_compute_count()` | 整合測試證明拓撲，不影響處理 |
| 你串的後級 | `mcab_sample_res_input()`＋`aec_res_process()`、`mcab_sample_anr_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`、`ac_istft_push()`、`skn_all_finite_f32()` | 後級函式庫是 libaec／libanr／audio_common 的；RES 用 `aec_res_*`（`aec_res.h`）；`mcab_sample_*` 是 `mcab_sample_post.h` 的範例碼（不在 archive 內） |

**內部（不要 include、不要依賴）**：`mcab_internal.h`（delay admission 狀態機 `mcab_admission_offer()`／`mcab_admission_age()` 與
`MCAB_DELAY_CHANGE_*` 常數、`mcab_far_spec_agrees()`、duty 診斷 hook、投影與 residual 向量內核）、
`mcab.c` 內的 static 函式與 `struct Mcab` 欄位；四條 lane 用到的 `aec_process_context_shared_far_from()` 也只該由 mcab 使用
（`aec.h` 說明它有嚴格的同 hop／同 reference 前提）。

### 0.3 參數面：外部與內部

**外部（`McabConfig`，`mcab.h`；以 `mcab_default_config()` 為起點）**

| 欄位 | 範圍／預設 | 何時設定 | 作用 |
|---|---|---|---|
| `sample_rate` | 16000／48000（**沒有** 8000） | init 前，不可變 | 決定格點 |
| `fft_size` | 0 = rate 預設；16 kHz 另可 256／512；48 kHz 只 1024 | init 前，不可變 | 格點（[§4.2](integration_4ch_core_zh_TW_03_config.md#42-grid由-sample_rate--fft_size-唯一決定)） |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `MATCHED`（預設，bank 1..5）／`FIXED`（`fixed_delay_samples` ∈ `[0, 120 × sample_rate]`）／`EXTERNAL_ALIGNED` | init 前，不可變 | 四條 lane 共用的對齊方式（[§4.1](integration_4ch_core_zh_TW_03_config.md#41-逐欄位)） |
| `capture_proxy_channel` | `[0, 3]`，預設 0 | init 前 | 共用 delay matcher 吃哪一支 mic |
| `max_delay_ms` | `[0, 4096]`，預設 1024 | init 前 | `MATCHED` 的 reference ring 容量；略大於真實最大延遲即可 |
| `enable_post` | 0／1，預設 1 | init 前 | 1 = 可呼叫 `mcab_fuse()`；0 = 只有 front（神經後濾波直接吃 BF 頻譜） |
| 每條 lane 的 `AecConfig`（level、`filter_length`…） | `mcab_lane_config()` = balanced level；或應用自己 `aec_config_from_level()` 後再 `mcab_apply_lane_contract()` | `aec_init()` 前，不可變 | **應用的決定，不在 `McabConfig` 內**：四路 AEC 記憶體最主要的旋鈕是 lane 的 `filter_length` |
| 後級強度（你串的 stage） | `aec_res_set_level(res, level, ramp_ms)`、`anr_set_level(nr, level)` | hop 之間 | RES、NR 強度（[§3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)）；flow 本身沒有 level 欄位 |

進階外部（預設關閉；打開前請閱讀標頭註解）：`delay_est_period_s`（預設 0.0 = 全速；非 0 可能改變選到的 lag，是 opt-in 的算量旋鈕）。

**內部（存在、可調，但不建議開放）**

| 項目 | 位置 | 為什麼不開放 |
|---|---|---|
| lane 的契約欄位：`delay_mode = AEC_DELAY_EXTERNAL_ALIGNED`（`delay_num_filters = DA_NUM_FILTERS`、`fixed_delay_samples = -1`、`enable_delay_est = 0`）、`enable_res = 0`、`return_res_context = 1`、`spatial_linear_context`（`enable_post = 1` 為 `AEC_CONTEXT_NO_GAIN`，`0` 為 `AEC_CONTEXT_LINEAR_ONLY`）、`delay_est_period_s` | `mcab.c` 的 `mcab_apply_lane_contract()` | 只能由 `mcab_apply_lane_contract()` 蓋上；`mcab_init_ex()` 只驗得到其中一部分（hop／bin 數、不帶自己的 gain、post 半邊、已對齊的 far），之後自行改欄位等於繞過檢查。四條 lane 的地板什麼都不塑形；對 lane 呼叫 `aec_set_level()` 是無效操作（[§3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)） |
| backward quarantine：`delay_backward_quarantine_enabled`／`delay_backward_quarantine_s`（預設 0／1.0 s） | `McabConfig`（`mcab.h`）；行為在 `mcab.c` 的 `update_shared_delay()` | 研究開關，產品組態維持 0：只把往前的候選延後一個窗，不修正誤鎖，也不是 pre-echo 誤鎖的解藥；欄位在 `enable` 為 0 時仍被驗證（[§4.1](integration_4ch_core_zh_TW_03_config.md#41-逐欄位)） |
| duty-cycle 的啟動時間 `DUTY_HOLD_S`（0.3 s，固定）與 matched filter 的 `DA_*` 常數 | `mcab.c`、`delay_aec.h` | 估計器聯合調過；duty-cycle 的啟動時間不是旋鈕 |
| delay admission 的門檻 `MCAB_DELAY_CHANGE_MIN_SAMPLES`（32）、`_CONFIRM_SAMPLES`（16）、`_CANDIDATE_TTL`（3 hops） | `mcab_internal.h` | 與 lib/aec 同一條規則，不是介面 |
| 其餘 `AecConfig` 欄位家族（時間常數、步進、偵測門檻、延遲獲取策略、實驗開關） | `aec.h` | 理由與逐項清單見 [mono 手冊 §0.3](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) |
| 投影與 residual 向量的數值保護（overflow fallback、單一 sqrt 與 divide） | `mcab_internal.h` | 屬於 `|out|² == r2` 契約的實作細節 |
| 後級的 `AnrConfig` 家族與 `L`／`num_init_frames` | `anr_types.h` | 同 [mono 手冊 §0.3](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) |

---

