# 四麥克風 mcab flow + SRP/BF sample helper 整合手冊（`mcab_sample_beam.h`）

適用對象：想要四麥克風的**共用 delay 對齊 + 四條線性 AEC**（`mcab` flow）再加上
**SRP-PHAT direction finding + beamformer** 現成接線（sample helper `McabSampleBeam`）的整合者。
你餵四路麥克風與一路 reference，拿回 BF 後的誤差頻譜、權重，以及（`mcab_sample_beam_fuse()` 之後）給頻域 stage 的
`McabFrame`；post-beam RES（libaec 的 `AecRes`）、NR、comfort noise、合成與神經網路後濾波
**都不在 mcab 與 helper 內**，是各自獨立的函式庫，由你初始化並逐 hop 串接
（範例見 `pipelines/4ch_aec_bf_nr_res/mcab_static.c` 與 `mcab_raw.c`）。

三個東西要分清楚：

* **`mcab`（`mcab.h`／`mcab.c`，出貨）**：flow 層。應用先用各 lib 自己的 `get_mem_size`／`init`
  建立四條 libaec `Aec` lane（MATCHED 模式另加一個共用 `DelayAec`），以指標交給 `mcab_init_ex()`；
  mcab 只持有自己小 pool 裡的 flow 狀態（delay admission、參考環、aligned far、fuse 緩衝），內部**沒有 beamformer**：`mcab_fuse()` 吃呼叫端的權重。
* **`mcab_sample_beam.{h,c}`（sample 層 helper，不在 archive）**：擁有 audiolib 的 SRP-PHAT、BF 與備援 energy VAD，
  把 `doa_step()` + `bf_process_with_weights()` 接在 `mcab_process_front()` 與 `mcab_fuse()` 之間。範例碼：沒有 ABI、沒有 layout version，可整份複製改寫。
* **`mcab_sample_flow.{h,c}`（sample 層 builder，不在 archive）**：把應用端 init 順序（四條 lane、`DelayAec`、mcab pool）寫在一處。

如果你**已經有**自己的 beamformer，不需要 helper，直接用 `mcab_process_front()` + `mcab_fuse()`，
請改看 `docs/integration_4ch_core_zh_TW.md`。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」，不含演算法推導。

> **路徑慣例**：本文路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`。
>
> **數值來源**：本文數字取自當前 source 或實測輸出。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 [§4.7](integration_4ch_spatial_zh_TW_03_config.md#47-哪些欄位會改變記憶體用量)）。

---

## 目錄 <!-- split-toc -->
- [四麥克風 mcab flow + SRP/BF sample helper 整合手冊（`mcab_sample_beam.h`）](#四麥克風-mcab-flow--srpbf-sample-helper-整合手冊mcab_sample_beamh) <!-- split-toc -->
  - [0. 速覽：流程時序圖、公開／內部函式、外部／內部參數](#0-速覽流程時序圖公開內部函式外部內部參數) <!-- split-toc -->
    - [0.1 流程時序圖（mcab flow + helper + 你串的後級）](#01-流程時序圖mcab-flow--helper--你串的後級) <!-- split-toc -->
    - [0.2 公開函式與內部函式](#02-公開函式與內部函式) <!-- split-toc -->
    - [0.3 參數面：外部與內部](#03-參數面外部與內部) <!-- split-toc -->
  - [1. 這個 flow 是什麼、你連結什麼、你 `#include` 什麼](#1-這個-flow-是什麼你連結什麼你-include-什麼) <!-- split-toc -->
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
    - [4.1 `McabConfig`（flow 參數）與 lane 的 `AecConfig`](integration_4ch_spatial_zh_TW_03_config.md#41-mcabconfigflow-參數與-lane-的-aecconfig) <!-- split-toc -->
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
    - [5.5 `destroy()` 的語意](integration_4ch_spatial_zh_TW_04_errors.md#55-destroy-的語意) <!-- split-toc -->
    - [5.6 `McabSampleBeamInfo`（唯讀 telemetry）](integration_4ch_spatial_zh_TW_04_errors.md#56-mcabsamplebeaminfo唯讀-telemetry) <!-- split-toc -->
    - [5.7 `mcab_init_ex()` 驗證什麼、不能驗證什麼](integration_4ch_spatial_zh_TW_04_errors.md#57-mcab_init_ex-驗證什麼不能驗證什麼) <!-- split-toc -->
  - [6. 出貨內容 vs 範例](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#6-出貨內容-vs-範例) <!-- split-toc -->
    - [6.1 函式庫（會進產品）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#61-函式庫會進產品) <!-- split-toc -->
    - [6.2 範例碼與參考執行檔（**沒有任何一個會進產品**）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#62-範例碼與參考執行檔沒有任何一個會進產品) <!-- split-toc -->
    - [6.3 其他（不出貨）](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#63-其他不出貨) <!-- split-toc -->
  - [7. 整合檢查清單](integration_4ch_spatial_zh_TW_06_checklist.md#7-整合檢查清單) <!-- split-toc -->
  - [8. 版本與相容性](integration_4ch_spatial_zh_TW_07_versioning.md#8-版本與相容性) <!-- split-toc -->
    - [8.1 `McabMemReq`](integration_4ch_spatial_zh_TW_07_versioning.md#81-mcabmemreq) <!-- split-toc -->
    - [8.2 `mcab_init_ex()` 的 8 項檢查](integration_4ch_spatial_zh_TW_07_versioning.md#82-mcab_init_ex-的-8-項檢查) <!-- split-toc -->
    - [8.3 序列化](integration_4ch_spatial_zh_TW_07_versioning.md#83-序列化) <!-- split-toc -->
    - [8.4 相容性規則（給整合者的四條）](integration_4ch_spatial_zh_TW_07_versioning.md#84-相容性規則給整合者的四條) <!-- split-toc -->
## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與位置都列出）。

### 0.1 流程時序圖（mcab flow + helper + 你串的後級）

生命線：應用、`Mcab`（共用 aligner 與四條注入 lane 的 flow 狀態；**沒有 RES、沒有 beamformer**）、`McabSampleBeam`
（helper：SRP-PHAT、BF）、libaec 的 `AecRes`（你自己建）、`AnrDenoiser`、`AcIstft`。
NN 後濾波的分支（`enable_post = 0`）列在圖後。

```
應用                 Mcab(flow)         McabSampleBeam(helper)    libaec AecRes(你建)   AnrDenoiser    AcIstft
 │                       │                       │                       │                  │             │
 │══ 初始化（一次；每次 init 前重查，不快取）═══════════════════════════════════════════════════
 │ mcab_default_config(sr) → cfg（flow 參數；不驗證）；mcab_sample_beam_default_params() → bp（覆寫 geometry、uca_radius_m、num_angles …）
 │ 四條 lane：mcab_lane_config(&cfg,&lane) → aec_get_mem_size → aec_init ×4（lane 的 preset／filter_length 由你決定，見 §4.1）
 │ MATCHED：mcab_delay_estimator_config(&cfg,&dc) → delay_aec_get_mem_size → delay_aec_init（DelayAec 結構也由你保管）
 │ mcab_get_mem_requirements(&cfg,&req) → 配置 req.bytes（16-byte 對齊）→ mcab_init_ex(pool,req.bytes,&cfg,&libs,&req) ►│ → Mcab*（NULL = 拒絕）
 │   （以上 = mcab_sample_flow_init／_create 的內容）
 │ mcab_sample_beam_get_mem_size(&cfg,&bp,&bytes) → 配置 → mcab_sample_beam_init(mem,bytes,&cfg,&bp,flow) ►│ → helper（NULL = 拒絕）
 │ 後級各自 mem_size + init（位址不在 mcab／helper 的 pool 內）：
 │   aec_config_from_preset(&res_cfg,preset,sr) ＋ res_cfg.fft_size = mcab_fft_size(flow)
 │     → aec_res_get_mem_size → aec_res_init ───────────────────────────────►│
 │   anr_config_for_mode_grid(sr, mcab_fft_size(flow), mode) → anr_get_mem_size → anr_init ──────────────►│
 │   ac_stft_config_default → ac_istft_get_mem_size → ac_istft_init ───────────────────────────────────►│
 │══ 每個 hop（mics[hop*4] interleaved、far[hop]）══════════════════════════════════════════════
 │ 形式 A：mcab_sample_beam_process_front(b,mics,far,&front) ►│ mcab_process_front：共用 delay + 四條 lane
 │          （可選）由 front.linear_spectra 算 mask、vad_raw、vad_out
 │          mcab_sample_beam_process_beam(b,&front,vad_raw,vad_out,mask,&beam,&info) ►│ SRP-PHAT + BF
 │ 形式 B：mcab_sample_beam_process_with_activity(b,mics,far,vad_raw,vad_out,mask,&beam,&info)  （= A 兩步）
 │ 形式 C：mcab_sample_beam_process(b,mics,far,&beam,&info)   （內建能量 VAD 當 fallback）
 │   beam.spectrum（BF 後誤差頻譜）、beam.weights（channel-major [4][n_freqs]）、info（DOA、delay …）
 │ mcab_sample_beam_fuse(b,&frame) ►│ mcab_fuse(flow, beam.weights, beam.spectrum)：投影 near／R²／comfort noise → McabFrame；frame.g_res_ctx = NULL
 │ mcab_sample_res_input(&frame,&in); aec_res_process(res,&in) ──────────────►│ G_res（NULL = 被拒）
 │ mcab_sample_anr_gain(nr,&frame,extra_noise,1) ───────────────────────────►│ G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,G_res,n); sk_capply_gain_f32(spec,frame.e,total,n)
 │ aec_cng_fill(&rng,spec,frame.cn,G_res,G_nr,MCAB_SAMPLE_CNG_NR_GAIN_FLOOR,n)
 │ ac_istft_push(istft,spec,out) ─────────────────────────────────────────────────────────►│ hop 個樣本
 │ skn_all_finite_f32(out,hop)
 │══ 任一步回非 OK／NULL，或輸出非有限（DSP_ERROR 時 flow 已自己 reset）══════════════════════════
 │ mcab_sample_beam_reset(b); aec_res_reset(res); anr_reset(nr); ac_istft_reset(istft); rng = MCAB_SAMPLE_CNG_RNG_SEED
 │══ 強度（hop 之間）：aec_res_set_preset(res,preset,ramp_ms)；anr_set_mode(nr,mode) ═══
 │══ 結束：mcab_sample_beam_destroy(b)；mcab_destroy(flow)；aec_destroy ×4；anr_destroy(nr)；之後才釋放 pool（`AecRes` 沒有 destroy）═══
```

`enable_post = 0`（Align-ULCNet 這類直接吃 BF 頻譜的後濾波）：同一組 `process_*` 呼叫，**不呼叫** `mcab_sample_beam_fuse()`
（會回 `MCAB_INVALID_ARGUMENT`）；把 `beam.spectrum`、同一次呼叫的原始 far 與 `info.delay.changed`／`info.delay.solid`
交給 `ulcnet_nn_stage_begin_complex()`，其後 `nn_begin` → 加速器 → `nn_end` → `ulcnet_nn_stage_end()`（見 `pipelines/ulcnet/4ch_alignulcnet/README.md`）。

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

**整合者實際會呼叫的**

| 階段 | 函式 | 備註 |
|---|---|---|
| 建立（mcab，出貨） | `mcab_default_config()`、`mcab_lane_config()`／`mcab_apply_lane_contract()`、`mcab_delay_estimator_config()`、`mcab_get_mem_requirements()`、`mcab_init_ex()`（或 `mcab_init()`）；桌面 `mcab_create()` | 描述子 32 bytes；`layout_version` 見 [§8.1](integration_4ch_spatial_zh_TW_07_versioning.md#81-mcabmemreq) |
| 建立（libaec） | `aec_get_mem_size()`／`aec_init()` ×4；MATCHED 時 `delay_aec_get_mem_size()`／`delay_aec_init()` | 物件由你保管，壽命要涵蓋 `Mcab` |
| 建立（sample 層） | `mcab_sample_flow_get_mem_size()`／`_init()`／`_create()`；`mcab_sample_beam_default_params()`、`mcab_sample_beam_get_mem_size()`、`mcab_sample_beam_init()` | 前者把上面兩列寫在一處；helper 借用 `Mcab*` |
| 每 hop | `mcab_sample_beam_process_front()` → `_process_beam()`；或 `_process_with_activity()`；或 `_process()` | 三種形式等價於前兩者的組合（§1.1） |
| 頻域串接 | `mcab_sample_beam_fuse()` | 可選；需要 `enable_post = 1` |
| 重置／收尾 | `mcab_sample_beam_reset()`、`mcab_sample_beam_destroy()`、`mcab_destroy()`、`mcab_sample_flow_destroy()` | reset 不重置你的後級（[§3.1](integration_4ch_spatial_zh_TW_02_lifecycle.md#31-reset-到底重置了什麼以及失敗時整合者要做什麼)） |
| 強度（你的後級） | `aec_res_set_preset()`、`aec_res_split_floor()`（作用在你自己的 `AecRes`）；`anr_set_mode()`（作用在你的降噪器） | mcab 與 helper 沒有強度 setter，也不持有 `AecRes` |
| 形狀與生效值 | `mcab_hop_size()`、`mcab_fft_size()`、`mcab_n_freqs()`、`mcab_sample_rate()`、`mcab_enable_post()`、`mcab_sample_beam_bf_effective_adapt_interval()`、`mcab_sample_beam_bf_lambda()` | 無效 handle 回 `-1`（`_bf_lambda()` 回 NaN） |
| 診斷（唯讀） | `mcab_get_last_timing()`、`mcab_matched_filter_count()`、`mcab_linear_aec_count()`、`mcab_sample_beam_spatial_backend()` | 時序預設全 0 |
| 你串的後級 | `aec_config_from_preset()`、`aec_res_get_mem_size()`／`aec_res_init()`、`mcab_sample_res_input()` + `aec_res_process()`、`mcab_sample_anr_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`、`ac_istft_push()`、`skn_all_finite_f32()` | `fuse` 之後 `frame.g_res_ctx` 恆為 NULL，RES 增益由你的 `AecRes` 經 `aec_res_process()` 算出（輸入由 `mcab_sample_res_input()` 從 frame 填；`mcab_sample_post.h` 是範例碼） |

**內部（不要依賴）**：`mcab_internal.h`（admission 狀態機、duty 診斷 hook、投影 kernel；不交付）與 `mcab.c` 內的 static 函式
（含 `build_flags_hash` 的計算）；`<audiolib>/{doa,bf,common/utility}` 的所有函式——helper 擁有它們的生命週期，
BF 借用 SRP 的 steering 表，所以 helper 內 destroy 次序是 BF → SRP（見 [§6.2](integration_4ch_spatial_zh_TW_05_shipped_vs_examples.md#62-範例碼與參考執行檔沒有任何一個會進產品)）。

### 0.3 參數面：外部與內部

欄位的逐項範圍、預設與驗證規則在 [§4](integration_4ch_spatial_zh_TW_03_config.md#4-config-完整參考)；下表是分級與理由。
flow 參數在 `mcab.h` 的 `McabConfig`（11 欄），陣列與 SRP／BF 參數在 `mcab_sample_beam.h` 的 `McabSampleBeamParams`（23 欄）。

**外部（部署資訊或單一取捨軸，建議讓第三方整合者設）**

| 欄位 | 位置 | 何時設定 | 作用 |
|---|---|---|---|
| `sample_rate`、`fft_size`、`delay_mode`、`delay_num_filters`、`fixed_delay_samples`、`capture_proxy_channel`、`max_delay_ms`、`enable_post` | `McabConfig`（[§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-mcabconfigflow-參數與-lane-的-aecconfig)；分級同 [mcab 手冊](integration_4ch_core_zh_TW.md)） | init 前，不可變 | 格點、對齊方式、是否可 `fuse()` |
| lane 的 `AecConfig`（`filter_length`、preset 等） | 應用自己組：`aec_config_from_preset()` ＋ `mcab_apply_lane_contract()`，或 `mcab_lane_config()`（balanced 預設） | init 前，不可變 | AEC 長度與記憶體；mcab 只蓋上 flow 的契約欄位，其餘不檢查 |
| `geometry`、`uca_radius_m` ／ `ula_spacing_m` ／ `microphone_x_m[4]`+`microphone_y_m[4]` | `McabSampleBeamParams`（[§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄)） | init 前 | 麥克風陣列的實體資訊；預設 UCA 半徑 0.035 m 只是讓範例能跑，產品必須設成實測值 |
| `num_angles` | [§4.3](integration_4ch_spatial_zh_TW_03_config.md#43-direction-findingdoa7-欄) | init 前 | 角度解析度；**helper 最大的記憶體旋鈕**（steering table，[§4.7](integration_4ch_spatial_zh_TW_03_config.md#47-哪些欄位會改變記憶體用量)） |
| 後級強度 | `aec_res_set_preset()`（你的 `AecRes`）、`anr_set_mode()`（你的降噪器） | hop 之間 | mcab 與 helper 本身沒有 RES 欄位，初始強度由你傳給 `aec_config_from_preset()` 的 preset 決定 |

進階外部（有明確使用情境才動，改了要自己重跑驗證）：`bf_enable`（0 = 不 beamform，直接輸出 channel 0）、
`bf_fixed_mode`＋`bf_fixed_doa_rad`（說話方向已知的部署）、`doa_low_freq_hz`／`doa_high_freq_hz`（定位頻帶，low < 解析後 high < Nyquist）、
`McabConfig.delay_est_period_s`（預設 0.0 = 全速；opt-in 的算量旋鈕，非 0 可能改變選到的 lag）。

**內部（存在、可調，但不建議開放）**

| 欄位／項目 | 位置 | 為什麼不開放 |
|---|---|---|
| `bf_lambda`、`bf_adapt_interval`、`bf_fixed_align_notebook` | [§4.4](integration_4ch_spatial_zh_TW_03_config.md#44-beamformerbf7-欄)、[§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) | 設定值與生效值不同：`bf_lambda` 依 hop 與生效 interval retime，fixed-notebook 模式強制 interval = 1；要讀回生效值才不會誤判 |
| `bf_mu` | [§4.4](integration_4ch_spatial_zh_TW_03_config.md#44-beamformerbf7-欄) | RLS 步長增益，與 `bf_lambda` 聯合調過，不 retime |
| `doa_enable_smoothing`、`doa_switch_consecutive`、`doa_angle_tolerance_rad`、`doa_update_interval` | [§4.3](integration_4ch_spatial_zh_TW_03_config.md#43-direction-findingdoa7-欄) | 追不上方向變化與誤切換是同一條軸上的取捨，放給外部等於把我們的取捨丟出去 |
| `speed_of_sound_m_s` | [§4.2](integration_4ch_spatial_zh_TW_03_config.md#42-陣列幾何4--2-欄) | 一般不需要改（343 m/s） |
| `auto_vad_threshold_dbfs`、`auto_vad_snr_ratio`、`auto_vad_hangover_frames` | [§4.5](integration_4ch_spatial_zh_TW_03_config.md#45-內建-fallback-vad3-欄) | 只在 `mcab_sample_beam_process()` 的 fallback VAD 用到；正式整合用自己的 VAD 走 `process_with_activity()`；`auto_vad_hangover_frames` 沒有讀回函式 |
| `delay_backward_quarantine_enabled`／`_s` | [§4.1](integration_4ch_spatial_zh_TW_03_config.md#41-mcabconfigflow-參數與-lane-的-aecconfig) | 預設關閉的研究開關，產品維持 0：只把往前的候選延後一個窗，不修正誤鎖 |
| SRP-PHAT／BF 內部常數與 RLS 共變異數佈局 | `<audiolib>/doa`、`<audiolib>/bf` | audiolib 的內部實作，helper 不開放 |

---

## 1. 這個 flow 是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► mcab_sample_beam_process_front()     mcab_process_front()：1 個共用 delay matcher → 4 條獨立 linear AEC
   │      （可選：由 front.linear_spectra 算 mask / VAD）
   ├─► mcab_sample_beam_process_beam()      SRP-PHAT direction finding + BF beamformer
   │      → McabSampleBeamResult { spectrum（BF 後誤差頻譜）, weights（有效權重）} + McabSampleBeamInfo
   │
   ├─► mcab_sample_beam_fuse()              （可選，要 enable_post = 1）
   │      mcab_fuse()：以那組權重做 post-beam 同調投影 → McabFrame（e, y, x, r2, cn, far_hop, …）
   │      frame.g_res_ctx = NULL（mcab 不算 RES 增益）
   ▼
 【你的後級】  G_res = aec_res_process(你的 AecRes, &in)（in = mcab_sample_res_input(&frame)）→ mcab_sample_anr_gain → sk_min_f32 → sk_capply_gain_f32
              → aec_cng_fill → ac_istft_push（或 ULCNet／DFN2 的 NN stage）→ out hop [hop]（mono）
```

mcab **借用**你建立的共用 delay 估計器與四條線性 AEC lane，並擁有它們之間的 flow 狀態；helper 擁有 SRP-PHAT 與 BF 的生命週期。
兩者都**不擁有**也不初始化 post-beam 的殘餘回音抑制器（libaec 的 `AecRes`）、降噪器（`AnrDenoiser`）、comfort noise、
合成（`AcIstft`）或任何 NN stage：每一個都是獨立的函式庫物件，有自己的 `mem_size` 與 `init`，初始化與每 hop 串接是整合者的事。
要 RES 就自己用 libaec 建一個 `AecRes`：`aec_config_from_preset(preset, sample_rate)`，`fft_size` 設成
`mcab_fft_size(flow)`，再 `aec_res_get_mem_size()`／`aec_res_init()`；每 hop 用 `aec_res_process()` 算增益（輸入由 `mcab_sample_res_input()` 填），
強度用 `aec_res_set_preset()`（[§3.3](integration_4ch_spatial_zh_TW_02_lifecycle.md#33-執行期強度控制)）。

每 hop 的呼叫有三種形式：

* `mcab_sample_beam_process_front()` → （可選：mask／VAD）→ `mcab_sample_beam_process_beam()`：
  你的 VAD／mask 要由 lane 頻譜算出時用這一組。
* `mcab_sample_beam_process_with_activity()` —— 兩段合一，由**你的產品 VAD** 提供活動偵測。
  **有自己 VAD 的產品整合請用這一個。**
* `mcab_sample_beam_process()` —— 用內建的保守能量 VAD 當 fallback。

`mcab_sample_beam_fuse()` 是獨立步驟，不在上述 per-hop 呼叫裡；只有頻域 stage 流程需要，
`enable_post = 0`（如 ULCNet）時回 `MCAB_INVALID_ARGUMENT`。

### 1.2 Header closure

```c
#include "mcab.h"
#include "mcab_sample_beam.h"
```

`mcab.h` 只拉進 `aec.h`（含 `DelayAec`、`Complex`）；`McabFrame` 就定義在 `mcab.h`，不依賴 `pipelines/common`。
`mcab_sample_beam.h` 另外**公開 include `srp.h` 與 `bf.h`**（`McabSampleSpatialConfig` 內含 `SRP_Config`、`BF_Config`），
所以編譯 helper 的 consumer 需要 `<audiolib>/doa` 與 `<audiolib>/bf` 兩個 include root（見 §1.4），但你**不需要**自己呼叫
`srp_*`／`bf_*`。想少寫 init 樣板可再 include `mcab_sample_flow.h`。
你的 RES 需要 `aec_res.h`（`aec.h` 之外的另一個 libaec header）。要串 RES、NR 與合成時另外 include `aec_res.h`、`anr.h`、`audio_stft.h`、
`mcab_sample_post.h`（範例碼：RES 輸入、NR 增益與 comfort-noise 常數）、`utility/simd_kernels.h` 與 `utility/simd_kernel_nn.h`；NN stage 另 include `ulcnet/ulcnet_nn_stage.h` 或 `dfn/dfn_hop.h`。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audiolib>/common/include/ac_complex.h`，
透過 `mcab.h` → `aec.h` → `fft_wrapper.h` → `ac_complex.h` 傳遞進來。

你會直接用到它：`McabSampleBeamResult.spectrum` 與 `.weights` 是 `const Complex*`，
`McabFront.linear_spectra[ch]` 也是。這些指標都借自 lane／helper，到下一次
`process_front()`／`process_with_activity()`／`process()`／`reset()`／`destroy()` 之前有效。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines/4ch_aec_bf_nr_res
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
-I <audiolib>/doa
-I <audiolib>/bf
```

這七個 include roots 涵蓋 `mcab.h`、`mcab_sample_beam.h`、`mcab_sample_flow.h` 與 `mcab_sample_post.h` 的相依
（`<audiolib>/common` 讓 `utility/mem_align.h`、`utility/simd_kernels.h` 與 `utility/simd_kernel_nn.h` 找得到）；
`mcab`、helper、builder 與 `mcab_sample_post.h` 都不依賴 `pipelines/common`（只有 `mcab_sample_driver.c` 引用 `common/example_cli.h`）。
**編譯 `mcab_sample_beam.c` 本身**（它不在 archive 內，由你的專案編）另需 `-I <audiolib>/common/utility`（`spatial_simd.h`）。
`mcab.c` 以 `-DAUDIO_PIPELINE_BACKEND_STR=\"ne10\"` 編譯；編譯 `mcab.c` 時沒帶這個巨集，`mcab_get_mem_requirements()` 回 `-1`。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o（含 mcab_sample_beam.o、mcab_sample_flow.o）
libaudio_pipeline_mcab.a
libdoa.a
libbf.a
libspatial_common.a
libaec.a
libanr.a
libaudio_common.a
-lm
```

* `libaudio_pipeline_mcab.a` 只含 `mcab.o`（`PIPELINE_OBJS`）；本手冊 helper 與 builder 的兩個 object
  （`mcab_sample_beam.o`、`mcab_sample_flow.o`）**不在 archive 內**，由你的專案編或複製原始碼（Makefile 的 `UNIT_LIBS` 把它們排在 archive 前面）。
* 比只用 mcab（另一份手冊）多了 `libdoa.a` / `libbf.a` / `libspatial_common.a` 三個
  archive。helper 真的會參照它們，少連就是 undefined symbol；`libaudio_pipeline_mcab.a` 本身不參照 `srp_`／`bf_`。
* 順序有意義：你的 object 與 `libaudio_pipeline_mcab.a` 在前，三個 spatial archive 在中，AEC / NR / audiolib 在後。
* 所有 archive 路徑都是 config-keyed 的（`bin/<backend>-<hash>/`）。
  用各 Makefile 的 `make -s print-lib-path` 取得，不要硬寫。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* **所有 archive 與 sample object 必須用同一組 backend / `SIMD` / 編譯選項建置。**
  `mcab` 的 `build_flags_hash` 只描述 mcab 自己 pool 的 carve 結構（lane 與 estimator 不在其中），
  helper 的 pool 沒有 descriptor（[§8](integration_4ch_spatial_zh_TW_07_versioning.md#8-版本與相容性)）。
* `mcab_sample_post.h`（header-only，直接 include）與 `ulcnet_nn_stage.c`（由你的專案自己編）不在任何 archive 裡。

---

