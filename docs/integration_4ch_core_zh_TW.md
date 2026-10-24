# 四麥克風前端整合手冊（`libaudio_pipeline_4ch.a` / `4aec_nr_res.h`）

適用對象：**已經擁有自己的 beamformer（SRP-PHAT / GSC 或任何等效方案）**，
想要四路 linear AEC 前端，並自己串接後級頻域 stage（RES / NR / CNG / ISTFT，或自己的 NN）的函式庫使用者。

如果你**沒有**自己的 beamformer、想要含 direction finding 與 beamformer 的線性 AEC + BF
單元（`AudioPipeline4Ch`），請看 `docs/integration_4ch_spatial_zh_TW.md`。

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
- [四麥克風前端整合手冊（`libaudio_pipeline_4ch.a` / `4aec_nr_res.h`）](#四麥克風前端整合手冊libaudio_pipeline_4cha--4aec_nr_resh) <!-- split-toc -->
  - [0. 速覽：流程時序圖、公開／內部函式、外部／內部參數](#0-速覽流程時序圖公開內部函式外部內部參數) <!-- split-toc -->
    - [0.1 流程時序圖（核心 `FourAecNrRes` + 你的 beamformer + 你串的後級）](#01-流程時序圖核心-fouraecnrres--你的-beamformer--你串的後級) <!-- split-toc -->
    - [0.2 公開函式與內部函式](#02-公開函式與內部函式) <!-- split-toc -->
    - [0.3 參數面：外部與內部](#03-參數面外部與內部) <!-- split-toc -->
  - [1. 這個函式庫是什麼、你連結什麼、你 `#include` 什麼](integration_4ch_core_zh_TW_01_library_quickstart.md#1-這個函式庫是什麼你連結什麼你-include-什麼) <!-- split-toc -->
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
    - [4.3 Buffer 形狀（每次 `front_process` / `fuse`）](integration_4ch_core_zh_TW_03_config.md#43-buffer-形狀每次-front_process--fuse) <!-- split-toc -->
    - [4.4 `FourAecFront` 與 `SeSpecFrame` 各欄位](integration_4ch_core_zh_TW_03_config.md#44-fouraecfront-與-sespecframe-各欄位) <!-- split-toc -->
    - [4.5 哪些欄位會改變記憶體用量](integration_4ch_core_zh_TW_03_config.md#45-哪些欄位會改變記憶體用量) <!-- split-toc -->
    - [4.6 實測記憶體（僅供量級參考，務必自己重查）](integration_4ch_core_zh_TW_03_config.md#46-實測記憶體僅供量級參考務必自己重查) <!-- split-toc -->
  - [5. 錯誤處理](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#5-錯誤處理) <!-- split-toc -->
    - [5.1 front → fuse 的次序規則](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#51-front--fuse-的次序規則) <!-- split-toc -->
    - [5.2 `four_aec_front_process()` 的完整拒絕清單（依實際檢查順序）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#52-four_aec_front_process-的完整拒絕清單依實際檢查順序) <!-- split-toc -->
    - [5.3 `four_aec_fuse()` 的完整拒絕清單（依實際檢查順序）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#53-four_aec_fuse-的完整拒絕清單依實際檢查順序) <!-- split-toc -->
    - [5.4 錯誤語意總表](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#54-錯誤語意總表) <!-- split-toc -->
    - [5.5 診斷輸出去向](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#55-診斷輸出去向) <!-- split-toc -->
    - [5.6 `destroy()` 的兩種語意](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#56-destroy-的兩種語意) <!-- split-toc -->
  - [6. 出貨內容 vs 範例](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#6-出貨內容-vs-範例) <!-- split-toc -->
    - [6.1 函式庫（會進產品）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#61-函式庫會進產品) <!-- split-toc -->
    - [6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#62-參考執行檔全部都有-main沒有任何一個會進產品) <!-- split-toc -->
    - [6.3 其他（不出貨）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#63-其他不出貨) <!-- split-toc -->
  - [7. 整合檢查清單](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#7-整合檢查清單) <!-- split-toc -->
  - [8. 版本與相容性](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#8-版本與相容性) <!-- split-toc -->
    - [8.1 `FourAecNrResMemReq`](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#81-fouraecnrresmemreq) <!-- split-toc -->
    - [8.2 `init_ex()` 的 8 項檢查](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#82-init_ex-的-8-項檢查) <!-- split-toc -->
    - [8.3 序列化](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#83-序列化) <!-- split-toc -->
    - [8.4 相容性規則（給整合者的三條）](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#84-相容性規則給整合者的三條) <!-- split-toc -->

## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖（核心 `FourAecNrRes` + 你的 beamformer + 你串的後級）

生命線：應用、`FourAecNrRes`（本手冊的核心）、你的 beamformer、`AecRes`（你自己建）、`AnrDenoiser`、`AcIstft`。
核心沒有 RES：`frame.g_res_ctx` 恆為 NULL，RES 增益由你的 `AecRes` 經 `se_res_stage_gain()` 算出。

```
應用            FourAecNrRes          你的 beamformer      AecRes(你建)   AnrDenoiser    AcIstft
 │                  │                      │                  │              │             │
 │══ 初始化（一次；每次 init 前重查，不快取）══════════════════════════════════════════
 │ four_aec_nr_res_default_config(sr)   → cfg（不驗證，只填值；覆寫你要改的欄位）
 │ four_aec_nr_res_get_mem_requirements(&cfg,&req) ►│    → req.bytes、req.alignment（0 成功／-1 失敗）
 │ 配置 req.bytes（16-byte 對齊）
 │ four_aec_nr_res_init_ex(pool,req.bytes,&cfg,&req) ►│    → handle（NULL = 8 項檢查之一不符；無診斷輸出）
 │ 後級各自 mem_size + init：aec_config_from_preset(&res_cfg,…) ＋ res_cfg.fft_size = four_aec_nr_res_fft_size(p)
 │   aec_res_get_mem_size/aec_res_init ──────────────────────►│
 │   anr_config_for_mode_grid/anr_get_mem_size/anr_init ───────────────────►│
 │   ac_stft_config_default/ac_istft_get_mem_size/ac_istft_init ───────────────────────────►│
 │══ 每個 hop（mics[hop*4] interleaved、ref[hop]）══════════════════════════
 │ four_aec_front_process(p,mics,ref,&front) ►│            → 0 / -1 / -3；front 為借用視圖
 │   front.delay.changed ≠ 0 ⇒ 先清掉你的 STFT/OLA 歷史（核心只 realign 四條 lane）
 │ front.linear_spectra[ch] ─────────────────►│ 算 effective weights Complex[4][n_freqs]（channel-major）
 │ four_aec_fuse(p,weights,beamformed_error|NULL,&frame) ►│   → 0 / -1 / -2 / -3；frame.g_res_ctx = NULL
 │ se_res_stage_gain(res,&frame) ───────────────────────►│ G_res（NULL = 被拒）
 │ se_anr_stage_gain(nr,&frame,extra_noise,1) ─────────────────────────►│ G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,G_res,n); sk_capply_gain_f32(spec,frame.e,total,n)
 │ aec_cng_fill(&rng,spec,frame.cn,G_res,G_nr,SE_CNG_NR_GAIN_FLOOR,n)
 │ ac_istft_push(istft,spec,out) ──────────────────────────────────────────────────────►│ hop 個樣本
 │ se_hop_is_finite(out,hop)
 │══ 任一步被拒（-3 時核心已自己 reset），或輸出非有限 ══════════════════════════
 │ four_aec_nr_res_reset(p); aec_res_reset(res); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED
 │══ 強度（hop 之間）：aec_res_set_preset(res,preset,ramp_ms)；anr_set_mode(nr,mode) ═══
 │══ 結束：four_aec_nr_res_destroy(p); anr_destroy(nr)；之後才釋放 pool ═══════════════
```

延遲泳道（橫軸 = hop）：

```
hop 序號                 t         t+1        t+2
四路 mic + ref           in[t]     in[t+1]    in[t+2]
front → BF → fuse → 後級  處理 in[t] 處理 in[t+1] 處理 in[t+2]   （核心與後級同一個 hop 內完成）
輸出 out                 y[t-1]    y[t]       y[t+1]            （你的 ac_istft_push 多 1 hop）
共用 aligner             delay.delay_samples 隨 delay.changed 在 hop 邊界切換（該 hop 的對齊已用新值）；
                         changed ≠ 0 的那個 hop：處理該 frame 之前先清掉你的 STFT/OLA 歷史
```

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開，`4aec_nr_res.h`）**

| 階段 | 函式 | 備註 |
|---|---|---|
| 建立 | `four_aec_nr_res_default_config()`、`four_aec_nr_res_get_mem_requirements()`、`four_aec_nr_res_init_ex()`（或 `init()`）；桌面 `four_aec_nr_res_create()` | 描述子 32 bytes，8 項檢查 |
| 每 hop | `four_aec_front_process()`、`four_aec_fuse()` | 之間是你的 beamformer |
| 重置／收尾 | `four_aec_nr_res_reset()`、`four_aec_nr_res_destroy()` | reset 會作廢 front 視圖與 `SeSpecFrame` |
| 形狀 | `four_aec_nr_res_hop_size()`、`_fft_size()`、`_n_freqs()`、`_sample_rate()` | 無效 handle 回 `-1` |
| 記憶體預算 | `four_aec_nr_res_get_mem_breakdown()` | 板端預算顯示 |
| 診斷（唯讀） | `four_aec_nr_res_get_last_timing()`、`_duty_hops_total()`、`_duty_hops_run()`、`_realign_warm_lane_count()`、`_realign_soft_lane_count()`、`_pending_delay_candidate()` | 時序欄位預設全 0（`-DFOUR_AEC_NR_RES_STAGE_TIMING=1`） |
| 結構稽核（唯讀） | `four_aec_nr_res_matched_filter_count()`、`_linear_aec_count()`、`_far_fft_real_compute_count()` | 整合測試證明拓撲，不影響處理 |
| 你串的後級 | `se_res_stage_gain()`、`se_anr_stage_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`、`ac_istft_push()`、`se_hop_is_finite()` | 與 mono 相同的後級函式庫；RES 用 `aec_res_*`（`aec_res.h`） |

**內部（不要 include、不要依賴）**：`4aec_nr_res_internal.h`（delay admission、`four_aec_nr_res_far_spec_agrees()` 等）、
`4aec_projection_kernels.h`（投影與 residual 向量內核）、`4aec_nr_res.c` 內的 static 函式與 `FourAecNrRes` 欄位；
四條 lane 用到的 `aec_process_context_shared_far_from()` 也只該由核心使用（`aec.h` 說明它有嚴格的同 hop／同 reference 前提）。

### 0.3 參數面：外部與內部

**外部（`FourAecNrResConfig`，`4aec_nr_res.h:96-111`；以 `four_aec_nr_res_default_config()` 為起點）**

| 欄位 | 範圍／預設 | 何時設定 | 作用 |
|---|---|---|---|
| `sample_rate` | 16000／48000（**沒有** 8000） | init 前，不可變 | 決定格點 |
| `fft_size` | 0 = rate 預設；16 kHz 另可 256／512；48 kHz 只 1024 | init 前，不可變 | 格點（[§4.2](integration_4ch_core_zh_TW_03_config.md#42-grid由-sample_rate--fft_size-唯一決定)） |
| `filter_length` | `[0, 4096]`，0 = 依取樣率預設 | init 前，不可變 | 四路 AEC 記憶體最主要的旋鈕 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `MATCHED`（預設，bank 1..5）／`FIXED`／`EXTERNAL_ALIGNED` | init 前，不可變 | 四條 lane 共用的對齊方式（[§4.1](integration_4ch_core_zh_TW_03_config.md#41-逐欄位)） |
| `capture_proxy_channel` | `[0, 3]`，預設 0 | init 前 | 共用 delay matcher 吃哪一支 mic |
| `max_delay_ms` | `[0, 4096]`，預設 1024 | init 前 | `MATCHED` 的 reference ring 容量；略大於真實最大延遲即可 |
| `enable_post` | 0／1，預設 1 | init 前 | 1 = 可呼叫 `four_aec_fuse()`；0 = 只有 front（神經後濾波直接吃 BF 頻譜） |
| 後級強度（你串的 stage） | `aec_res_set_preset(res, preset, ramp_ms)`、`anr_set_mode(nr, mode)` | hop 之間 | RES、NR 強度（[§3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)）；前端本身沒有 preset 欄位 |

進階外部（預設關閉；打開前請閱讀標頭註解）：`delay_est_period_s`（預設 0.0 = 全速；非 0 可能改變選到的 lag）、
`delay_backward_quarantine_enabled`／`delay_backward_quarantine_s`（預設 0／1.0 s；只延後一個窗，不修正誤鎖，產品組態維持 0）。

**內部（存在、可調，但不建議開放）**

| 項目 | 位置 | 為什麼不開放 |
|---|---|---|
| 四條 lane 的 `AecConfig`：preset 固定 `AEC_PRESET_BALANCED`、`delay_mode = AEC_DELAY_EXTERNAL_ALIGNED`、`enable_res = 0`、`return_res_context = 1`、`spatial_linear_context`（`enable_post = 1` 為 `AEC_CONTEXT_NO_GAIN`，`0` 為 `AEC_CONTEXT_LINEAR_ONLY`） | `4aec_nr_res.c` 的 `derive_dims_and_configs()` | 四條 lane 的地板什麼都不塑形；對 lane 呼叫 `aec_set_preset()` 是無效操作（[§3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)） |
| `delay_est_init_s`（0.3 s，固定）與 matched filter 的 `DA_*` 常數 | `aec.h`、`delay_aec.h` | 估計器聯合調過；duty-cycle 的啟動時間不是旋鈕 |
| 其餘 `AecConfig` 欄位家族（時間常數、步進、偵測門檻、延遲獲取策略、實驗開關） | `aec.h:59-156` | 理由與逐項清單見 [mono 手冊 §0.3](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) |
| 投影與 residual 向量的數值保護（overflow fallback、單一 sqrt 與 divide） | `4aec_projection_kernels.h` | 屬於 `|out|² == r2` 契約的實作細節 |
| 後級的 `AnrConfig` 家族與 `L`／`num_init_frames` | `anr_types.h` | 同 [mono 手冊 §0.3](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) |

---

