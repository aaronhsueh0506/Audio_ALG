# 四麥克風線性 AEC + BF 單元整合手冊（`audio_pipeline_4ch.h`）

適用對象：想要四麥克風的**共用 delay 對齊 + 四條線性 AEC + SRP-PHAT direction finding + beamformer**
這一個單元（`AudioPipeline4Ch`）的函式庫使用者。你餵四路麥克風與一路 reference，拿回 BF 後的誤差頻譜、
權重，以及（`fuse()` 之後）給頻域 stage 用的 `SeSpecFrame`；post-beam RES（libaec 的 `AecRes`）、NR、comfort noise、
合成與神經網路後濾波**都不在單元內**，是各自獨立的函式庫，由你初始化並逐 hop 串接
（範例見 `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` 與 `audio_pipeline_4ch_raw.c`）。

如果你**已經有**自己的 beamformer，只想要 AEC 前端，
請改看 `docs/integration_4ch_core_zh_TW.md`。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 [§4.7](integration_4ch_spatial_zh_TW_03_config.md#47-哪些欄位會改變記憶體用量)）。

---

## 目錄 <!-- split-toc -->
- [四麥克風線性 AEC + BF 單元整合手冊（`audio_pipeline_4ch.h`）](#四麥克風線性-aec--bf-單元整合手冊audio_pipeline_4chh) <!-- split-toc -->
  - [0. 速覽：流程時序圖、公開／內部函式、外部／內部參數](#0-速覽流程時序圖公開內部函式外部內部參數) <!-- split-toc -->
    - [0.1 流程時序圖（單元 `AudioPipeline4Ch` + 你串的後級）](#01-流程時序圖單元-audiopipeline4ch--你串的後級) <!-- split-toc -->
    - [0.2 公開函式與內部函式](#02-公開函式與內部函式) <!-- split-toc -->
    - [0.3 參數面：外部與內部](#03-參數面外部與內部) <!-- split-toc -->
  - [1. 這個單元是什麼、你連結什麼、你 `#include` 什麼](#1-這個單元是什麼你連結什麼你-include-什麼) <!-- split-toc -->
    - [1.1 這是什麼](#11-這是什麼) <!-- split-toc -->
    - [1.2 Header closure](#12-header-closure) <!-- split-toc -->
    - [1.3 `Complex` type 從哪來](#13-complex-type-從哪來) <!-- split-toc -->
    - [1.4 編譯需要的 `-I` 清單（consumer 端最小集合）](#14-編譯需要的--i-清單consumer-端最小集合) <!-- split-toc -->
    - [1.5 Archive link order](#15-archive-link-order) <!-- split-toc -->
  - [2. Quick start](integration_4ch_spatial_zh_TW_01_quick_start.md#2-quick-start) <!-- split-toc -->
  - [3. Lifecycle](integration_4ch_spatial_zh_TW_02_lifecycle.md#3-lifecycle) <!-- split-toc -->
    - [3.1 `reset()` 到底重置了什麼，以及失敗時整合者要做什麼](integration_4ch_spatial_zh_TW_02_lifecycle.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼) <!-- split-toc -->
    - [3.2 執行期唯讀存取](integration_4ch_spatial_zh_TW_02_lifecycle.md#32-執行期唯讀存取) <!-- split-toc -->
    - [3.3 執行期強度控制](integration_4ch_spatial_zh_TW_02_lifecycle.md#33-執行期強度控制) <!-- split-toc -->
  - [4. Config 完整參考](integration_4ch_spatial_zh_TW_03_config.md#4-config-完整參考) <!-- split-toc -->
    - [4.1 `cfg.core`（核心層欄位）](integration_4ch_spatial_zh_TW_03_config.md#41-cfgcore核心層欄位) <!-- split-toc -->
    - [4.2 陣列幾何（4 + 2 欄）](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄) <!-- split-toc -->
    - [4.3 Direction finding（DOA，7 欄）](integration_4ch_spatial_zh_TW_03_config.md#43-direction-findingdoa7-欄) <!-- split-toc -->
    - [4.4 Beamformer（BF，7 欄）](integration_4ch_spatial_zh_TW_03_config.md#44-beamformerbf7-欄) <!-- split-toc -->
    - [4.5 內建 fallback VAD（3 欄）](integration_4ch_spatial_zh_TW_03_config.md#45-內建-fallback-vad3-欄) <!-- split-toc -->
    - [4.6 你設定的值不一定是實際生效的值](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) <!-- split-toc -->
    - [4.7 哪些欄位會改變記憶體用量](integration_4ch_spatial_zh_TW_03_config.md#47-哪些欄位會改變記憶體用量) <!-- split-toc -->
  - [5. 錯誤處理](integration_4ch_spatial_zh_TW_04_errors.md#5-錯誤處理) <!-- split-toc -->
    - [5.1 錯誤語意總表](integration_4ch_spatial_zh_TW_04_errors.md#51-錯誤語意總表) <!-- split-toc -->
    - [5.2 process 各入口與 fuse 的完整拒絕清單](integration_4ch_spatial_zh_TW_04_errors.md#52-process-各入口與-fuse-的完整拒絕清單) <!-- split-toc -->
    - [5.3 `process_with_activity()` 的三個活動參數](integration_4ch_spatial_zh_TW_04_errors.md#53-process_with_activity-的三個活動參數) <!-- split-toc -->
    - [5.4 診斷輸出去向](integration_4ch_spatial_zh_TW_04_errors.md#54-診斷輸出去向) <!-- split-toc -->
    - [5.5 `destroy()` 的兩種語意](integration_4ch_spatial_zh_TW_04_errors.md#55-destroy-的兩種語意) <!-- split-toc -->
    - [5.6 `AudioPipeline4ChFrameInfo`（唯讀 telemetry）](integration_4ch_spatial_zh_TW_04_errors.md#56-audiopipeline4chframeinfo唯讀-telemetry) <!-- split-toc -->
  - [6. 出貨內容 vs 範例](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#6-出貨內容-vs-範例) <!-- split-toc -->
    - [6.1 函式庫（會進產品）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#61-函式庫會進產品) <!-- split-toc -->
    - [6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#62-參考執行檔全部都有-main沒有任何一個會進產品) <!-- split-toc -->
    - [6.3 其他（不出貨）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#63-其他不出貨) <!-- split-toc -->
  - [7. 整合檢查清單](integration_4ch_spatial_zh_TW_06_checklist.md#7-整合檢查清單) <!-- split-toc -->
  - [8. 版本與相容性](integration_4ch_spatial_zh_TW_07_versioning.md#8-版本與相容性) <!-- split-toc -->
    - [8.1 `AudioPipeline4ChMemReq`](integration_4ch_spatial_zh_TW_07_versioning.md#81-audiopipeline4chmemreq) <!-- split-toc -->
    - [8.2 `init_ex()` 的 8 項檢查](integration_4ch_spatial_zh_TW_07_versioning.md#82-init_ex-的-8-項檢查) <!-- split-toc -->
    - [8.3 序列化](integration_4ch_spatial_zh_TW_07_versioning.md#83-序列化) <!-- split-toc -->
    - [8.4 相容性規則（給整合者的四條）](integration_4ch_spatial_zh_TW_07_versioning.md#84-相容性規則給整合者的四條) <!-- split-toc -->
## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖（單元 `AudioPipeline4Ch` + 你串的後級）

生命線：應用、`AudioPipeline4Ch`（內含共用 aligner、四條 lane、SRP-PHAT、BF；**沒有 RES**）、libaec 的 `AecRes`（你自己建）、
`AnrDenoiser`、`AcIstft`。NN 後濾波的分支（`core.enable_post = 0`）列在圖後。

```
應用                 AudioPipeline4Ch            libaec AecRes(你建)     AnrDenoiser          AcIstft
 │                         │                           │                    │                  │
 │══ 初始化（一次；每次 init 前重查，不快取）════════════════════════════════════
 │ audio_pipeline_4ch_default_config(sr)   → cfg（不驗證；覆寫 geometry、uca_radius_m、num_angles、core.* …）
 │ audio_pipeline_4ch_get_mem_requirements(&cfg,&req) ►│   → req.bytes、req.alignment（0 成功／-1 失敗）
 │ 配置 req.bytes（16-byte 對齊）
 │ audio_pipeline_4ch_init_ex(pool,req.bytes,&cfg,&req) ►│  → handle（NULL = 8 項檢查之一不符，或 core／SRP／BF init 失敗）
 │ 後級各自 mem_size + init（位址不在單元的 pool 內）：
 │   aec_config_from_preset(&res_cfg,preset,sr) ＋ res_cfg.fft_size = audio_pipeline_4ch_fft_size(p)
 │     → aec_res_get_mem_size → aec_res_init ────────────────────────►│
 │   anr_config_for_mode_grid(sr, audio_pipeline_4ch_fft_size(p), mode) → anr_get_mem_size → anr_init ─────►│
 │   ac_stft_config_default → ac_istft_get_mem_size → ac_istft_init ──────────────────────────────►│
 │══ 每個 hop（mics[hop*4] interleaved、far[hop]）══════════════════════════
 │ 形式 A：audio_pipeline_4ch_process_front(p,mics,far,&front) ►│  共用 delay + 四條 lane
 │          （可選）由 front.linear_spectra 算 mask、vad_raw、vad_out
 │          audio_pipeline_4ch_process_beam(p,&front,vad_raw,vad_out,mask,&beam,&info) ►│  SRP-PHAT + BF
 │ 形式 B：audio_pipeline_4ch_process_with_activity(p,mics,far,vad_raw,vad_out,mask,&beam,&info)  （= A 兩步）
 │ 形式 C：audio_pipeline_4ch_process(p,mics,far,&beam,&info)   （內建能量 VAD 當 fallback）
 │   beam.spectrum（BF 後誤差頻譜）、beam.weights（channel-major [4][n_freqs]）、info（DOA、delay …）
 │ audio_pipeline_4ch_fuse(p,&frame) ►│  投影 near／R²／comfort noise → SeSpecFrame；frame.g_res_ctx = NULL
 │ se_res_stage_gain(res,&frame) ────────────────────────────►│ G_res（NULL = 被拒）
 │ se_anr_stage_gain(nr,&frame,extra_noise,1) ─────────────────────────────►│ G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,G_res,n); sk_capply_gain_f32(spec,frame.e,total,n)
 │ aec_cng_fill(&rng,spec,frame.cn,G_res,G_nr,SE_CNG_NR_GAIN_FLOOR,n)
 │ ac_istft_push(istft,spec,out) ───────────────────────────────────────────────────►│ hop 個樣本
 │ se_hop_is_finite(out,hop)
 │══ 任一步回非 OK／NULL，或輸出非有限（fuse 失敗時單元已自己 reset）═══════════
 │ audio_pipeline_4ch_reset(p); aec_res_reset(res); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED
 │══ 強度（hop 之間）：aec_res_set_preset(res,preset,ramp_ms)；anr_set_mode(nr,mode) ═══
 │══ 結束：audio_pipeline_4ch_destroy(p); anr_destroy(nr)；之後才釋放 pool（`AecRes` 沒有 destroy，header 註明無物可釋放）═══
```

`core.enable_post = 0`（Align-ULCNet 這類直接吃 BF 頻譜的後濾波）：同一組 `process_*` 呼叫，**不呼叫** `fuse()`
（會回 `INVALID_ARGUMENT`）；把 `beam.spectrum`、同一次呼叫的原始 far 與 `info.delay.changed`／`info.delay.solid`
交給 `ulcnet_nn_stage_begin_complex()`，其後 `nn_begin` → 加速器 → `nn_end` → `ulcnet_nn_stage_end()`（見 `pipelines/4ch_alignulcnet/README.md`）。

延遲泳道（橫軸 = hop）：

```
hop 序號                       t          t+1         t+2
四路 mic + far                 in[t]      in[t+1]     in[t+2]
lanes → SRP/BF → fuse → 後級    處理 in[t]  處理 in[t+1]  處理 in[t+2]   （同一個 hop 內完成）
輸出 out（ac_istft_push）       y[t-1]     y[t]        y[t+1]         （合成 OLA 多 1 hop）
```

SRP-PHAT 與 BF 的 DOA 更新間隔由 `doa_update_interval`（預設每 2 個 hop）與 `bf_adapt_interval`（預設 1）決定，不影響輸出時序；
共用 aligner 從 `delay.changed` 的那個 hop 起（含該 hop）使用新的 `delay_samples`；你的 STFT/OLA 歷史要在處理該 hop 的 frame 之前清掉。

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開，`audio_pipeline_4ch.h`）**

| 階段 | 函式 | 備註 |
|---|---|---|
| 建立 | `audio_pipeline_4ch_default_config()`、`audio_pipeline_4ch_get_mem_requirements()`、`audio_pipeline_4ch_init_ex()`（或 `init()`）；桌面 `audio_pipeline_4ch_create()` | 描述子 32 bytes；`layout_version` 見 [§8.1](integration_4ch_spatial_zh_TW_07_versioning.md#81-audiopipeline4chmemreq) |
| 每 hop | `process_front()` → `process_beam()`；或 `process_with_activity()`；或 `process()` | 三種形式等價於前兩者的組合（§1.1） |
| 頻域串接 | `audio_pipeline_4ch_fuse()` | 可選；需要 `core.enable_post = 1` |
| 重置／收尾 | `audio_pipeline_4ch_reset()`、`audio_pipeline_4ch_destroy()` | reset 不重置你的後級（[§3.1](integration_4ch_spatial_zh_TW_02_lifecycle.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼)） |
| 強度（你的後級） | `aec_res_set_preset()`、`aec_res_split_floor()`（作用在你自己的 `AecRes`）；`anr_set_mode()`（作用在你的降噪器） | 單元沒有強度 setter，也不持有 `AecRes` |
| 形狀與生效值 | `audio_pipeline_4ch_hop_size()`、`_frame_size()`、`_fft_size()`、`_n_freqs()`、`_sample_rate()`、`_doa_*`、`_bf_*`、`_bf_effective_adapt_interval()`、`_bf_lambda()` | 無效 handle 回 `-1`（`_bf_lambda()` 回 NaN） |
| 診斷（唯讀） | `audio_pipeline_4ch_get_core()`、`audio_pipeline_4ch_last_core_timing()`、`audio_pipeline_4ch_spatial_backend()`、`_matched_filter_count()`、`_linear_aec_count()` | 時序欄位預設全 0 |
| 你串的後級 | `aec_config_from_preset()`、`aec_res_get_mem_size()`／`aec_res_init()`、`se_res_stage_gain()`、`se_anr_stage_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`、`ac_istft_push()`、`se_hop_is_finite()` | `fuse()` 之後 `frame.g_res_ctx` 恆為 NULL，RES 增益由你的 `AecRes` 經 `se_res_stage_gain()` 算出 |

**內部（不要 include、不要依賴）**：`bf.h`、`srp.h` 與 `<audiolib>/{doa,bf,common/utility}` 的所有函式（單元擁有它們的生命週期，
BF 借用 SRP 的 steering 表，所以 destroy 次序是 BF → SRP → 核心）；`4aec_nr_res_internal.h`、`4aec_projection_kernels.h`；
單元 `.c` 內的 static 函式（含 `build_flags_hash` 的計算）。

### 0.3 參數面：外部與內部

欄位的逐項範圍、預設與驗證規則在 [§4](integration_4ch_spatial_zh_TW_03_config.md#4-config-完整參考)；下表是分級與理由。結構在 `audio_pipeline_4ch.h:45-75`（`AudioPipeline4ChConfig`），核心部分在 `4aec_nr_res.h:96-111`。

**外部（部署資訊或單一取捨軸，建議讓第三方整合者設）**

| 欄位 | 位置 | 何時設定 | 作用 |
|---|---|---|---|
| `core.sample_rate`、`core.fft_size`、`core.filter_length`、`core.delay_mode`、`core.delay_num_filters`、`core.fixed_delay_samples`、`core.capture_proxy_channel`、`core.max_delay_ms`、`core.enable_post` | `AudioPipeline4ChConfig.core`（[§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-cfgcore核心層欄位)；分級同 [核心手冊 §0.3](integration_4ch_core_zh_TW.md#03-參數面外部與內部)） | init 前，不可變 | 格點、AEC 長度、對齊方式、是否可 `fuse()` |
| `geometry`、`uca_radius_m` ／ `ula_spacing_m` ／ `microphone_x_m[4]`+`microphone_y_m[4]` | [§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄) | init 前 | 麥克風陣列的實體資訊；預設 UCA 半徑 0.035 m 只是讓範例能跑，產品必須設成實測值 |
| `num_angles` | [§4.3](integration_4ch_spatial_zh_TW_03_config.md#43-direction-findingdoa7-欄) | init 前 | 角度解析度；**最大的記憶體旋鈕**（72 → 360 約 1.70 MB → 4.70 MB，[§4.7](integration_4ch_spatial_zh_TW_03_config.md#47-哪些欄位會改變記憶體用量)） |
| 後級強度 | `aec_res_set_preset()`（你的 `AecRes`）、`anr_set_mode()`（你的降噪器） | hop 之間；單元本身沒有 RES 欄位，初始強度由你傳給 `aec_config_from_preset()` 的 preset 決定 |

進階外部（有明確使用情境才動，改了要自己重跑驗證）：`bf_enable`（0 = 不 beamform，直接輸出 channel 0）、
`bf_fixed_mode`＋`bf_fixed_doa_rad`（說話方向已知的部署）、`doa_low_freq_hz`／`doa_high_freq_hz`（定位頻帶，須 `< Nyquist`）。

**內部（存在、可調，但不建議開放）**

| 欄位／項目 | 位置 | 為什麼不開放 |
|---|---|---|
| `bf_lambda`、`bf_adapt_interval`、`bf_fixed_align_notebook` | [§4.4](integration_4ch_spatial_zh_TW_03_config.md#44-beamformerbf7-欄)、[§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) | 設定值與生效值不同：`bf_lambda` 依 hop 與生效 interval retime，fixed-notebook 模式強制 interval = 1；要讀回生效值才不會誤判 |
| `bf_mu` | [§4.4](integration_4ch_spatial_zh_TW_03_config.md#44-beamformerbf7-欄) | RLS 步長增益，與 `bf_lambda` 聯合調過，不 retime |
| `doa_enable_smoothing`、`doa_switch_consecutive`、`doa_angle_tolerance_rad`、`doa_update_interval` | [§4.3](integration_4ch_spatial_zh_TW_03_config.md#43-direction-findingdoa7-欄) | 追不上方向變化與誤切換是同一條軸上的取捨，放給外部等於把我們的取捨丟出去 |
| `speed_of_sound_m_s` | [§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄) | 一般不需要改（343 m/s） |
| `auto_vad_threshold_dbfs`、`auto_vad_snr_ratio`、`auto_vad_hangover_frames` | [§4.5](integration_4ch_spatial_zh_TW_03_config.md#45-內建-fallback-vad3-欄) | 只在 `audio_pipeline_4ch_process()` 的 fallback VAD 用到；正式整合用自己的 VAD 走 `process_with_activity()`；`auto_vad_hangover_frames` 沒有讀回函式 |
| 四條 lane 與核心內部設定 | [核心手冊 §0.3](integration_4ch_core_zh_TW.md#03-參數面外部與內部) | 同核心 |
| `core.delay_est_period_s`、`core.delay_backward_quarantine_*` | [§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-cfgcore核心層欄位) | 預設關閉；打開會改變選到的 lag 或只延後一個窗 |
| SRP-PHAT／BF 內部常數與 RLS 共變異數佈局 | `<audiolib>/doa`、`<audiolib>/bf` | audiolib 的內部實作，單元不開放 |

---

## 1. 這個單元是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► audio_pipeline_4ch_process_front()   1 個共用 delay matcher → 4 條獨立 linear AEC
   │      （可選：由 front.linear_spectra 算 mask / VAD）
   ├─► audio_pipeline_4ch_process_beam()    SRP-PHAT direction finding + BF beamformer
   │      → AudioPipeline4ChBeam { spectrum（BF 後誤差頻譜）, weights（有效權重） } + frame telemetry
   │
   ├─► audio_pipeline_4ch_fuse()            （可選，要 core.enable_post = 1）
   │      以那組權重做 post-beam 同調投影 → SeSpecFrame（e, y, x, r2, cn, far_hop, …）
   │      frame.g_res_ctx = NULL（單元不算 RES 增益，和核心一樣）
   ▼
 【你的後級】  G_res = se_res_stage_gain(你的 AecRes, &frame) → se_anr_stage_gain → sk_min_f32 → sk_capply_gain_f32
              → aec_cng_fill → ac_istft_push（或 ULCNet／DFN2 的 NN stage）→ out hop [hop]（mono）
```

單元**擁有**共用 delay 對齊、四條線性 AEC、SRP-PHAT 與 BF 的生命週期。
它**不擁有**也不初始化 post-beam 的殘餘回音抑制器（libaec 的 `AecRes`）、降噪器（`AnrDenoiser`）、comfort noise、
合成（`AcIstft`）或任何 NN stage：每一個都是獨立的函式庫物件，有自己的 `mem_size` 與 `init`，
初始化與每 hop 串接是整合者的事（範例程式各自把 RES、NR、合成與增益暫存放進自己的 pool，沒有 ABI）。
要 RES 就自己用 libaec 建一個 `AecRes`：`aec_config_from_preset(preset, sample_rate)`，`fft_size` 設成
`audio_pipeline_4ch_fft_size(p)`，再 `aec_res_get_mem_size()`／`aec_res_init()`；每 hop 用 `se_res_stage_gain()` 算增益，
強度用 `aec_res_set_preset()`（[§3.3](integration_4ch_spatial_zh_TW_02_lifecycle.md#33-執行期強度控制)）。

每 hop 的呼叫有三種形式：

* `audio_pipeline_4ch_process_front()` → （可選：mask／VAD）→ `audio_pipeline_4ch_process_beam()`：
  你的 VAD／mask 要由 lane 頻譜算出時用這一組。
* `audio_pipeline_4ch_process_with_activity()` —— 兩段合一，由**你的產品 VAD** 提供活動偵測。
  **有自己 VAD 的產品整合請用這一個。**
* `audio_pipeline_4ch_process()` —— 用內建的保守能量 VAD 當 fallback。

`audio_pipeline_4ch_fuse()` 是獨立步驟，不在上述 per-hop 呼叫裡；只吃頻域 stage 的流程才需要，
`core.enable_post = 0`（ULCNet 等直接吃 BF 頻譜的後濾波）時回 `INVALID_ARGUMENT`。

### 1.2 Header closure

```c
#include "audio_pipeline_4ch.h"
```

一行就夠單元本身用。它會拉進 `4aec_nr_res.h`，後者再拉進 `aec.h` /
`fft_wrapper.h` / `common/se_stage.h`（不含 NR 的 header）。你**不需要** include `bf.h`、`srp.h` 或任何 `audiolib` 的 header。
你的 RES 需要 `aec_res.h`（`aec.h` 之外的另一個 libaec header）。要串 RES、NR 與合成時另外 include `aec_res.h`、`anr.h`、`audio_stft.h`、
`samples/se_anr_stage.h` 與 `utility/simd_kernels.h`；NN stage 另 include `samples/ulcnet_nn_stage.h` 或 `samples/dfn_hop.h`。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audiolib>/common/include/fft_wrapper.h`，
透過 `audio_pipeline_4ch.h` → `4aec_nr_res.h` → `aec.h` → `fft_wrapper.h` 傳遞進來。

你會直接用到它：`AudioPipeline4ChBeam.spectrum` 與 `.weights` 是 `const Complex*`，
`FourAecFront.linear_spectra[ch]` 也是。這些指標都借自單元，到下一次
`process_front()`／`process_with_activity()`／`process()`／`reset()`／`destroy()` 之前有效。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines/4ch_aec_bf_nr_res
-I pipelines
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
```

五個 include roots 涵蓋 `audio_pipeline_4ch.h` 與其 `common/se_stage.h` 相依。
串 NR／合成（`samples/se_anr_stage.h`）時另加 `-I <audiolib>/common`，讓 `utility/mem_align.h` 與
`utility/simd_kernels.h` 找得到。
`<audiolib>/doa`、`<audiolib>/bf`、`<audiolib>/common/utility` 這三個 `-I`
是**建置這個 library 自己**才需要的，consumer 端不需要。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o
libaudio_pipeline_4ch.a
libdoa.a
libbf.a
libspatial_common.a
libaec.a
libanr.a
libaudio_common.a
-lm
```

* `libaudio_pipeline_4ch.a` 的 `PIPELINE_OBJS` 含 `4aec_nr_res.o`（核心）、`audio_pipeline_4ch.o`（本單元）與
  `se_config.o`（見核心版手冊 §1.5）——本手冊這層的公開 API（`audio_pipeline_4ch_*`）就在 archive 裡，
  不需要另外編 object。
* 比核心版（另一份手冊）多了 `libdoa.a` / `libbf.a` / `libspatial_common.a` 三個
  archive。這一層真的會參照它們，少連就是 undefined symbol。
* 順序有意義：`libaudio_pipeline_4ch.a` 在前，三個 spatial archive 在中，AEC / NR / audiolib 在後。
* 所有 archive 路徑都是 config-keyed 的（`bin/<backend>-<hash>/`）。
  用各 Makefile 的 `make -s print-lib-path` 取得，不要硬寫。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* **七個 archive 必須用同一組 backend / `SIMD` / 編譯選項建置。**
  這一層的 `build_flags_hash` 有把核心層的 hash 摺進去，所以核心層版本一變動，
  這一層所有已保存的 descriptor 也會一併失效 —— 不會發生「用舊核心佈局的 pool
  還剛好塞得下」這種事。
* `se_anr_stage.h`、`ulcnet_nn_stage.c` 等範例碼不在任何 archive 裡：header-only 的直接 include，
  `ulcnet_nn_stage.c` 由你的專案自己編（`4ch_aec_bf_nr_res/Makefile` 的 `ulcnet_nn_stage.o` 規則是範例）。

---

