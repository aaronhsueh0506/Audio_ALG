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
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.7）。

---

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
| 建立 | `audio_pipeline_4ch_default_config()`、`audio_pipeline_4ch_get_mem_requirements()`、`audio_pipeline_4ch_init_ex()`（或 `init()`）；桌面 `audio_pipeline_4ch_create()` | 描述子 32 bytes；`layout_version` 見 §8.1 |
| 每 hop | `process_front()` → `process_beam()`；或 `process_with_activity()`；或 `process()` | 三種形式等價於前兩者的組合（§1.1） |
| 頻域串接 | `audio_pipeline_4ch_fuse()` | 可選；需要 `core.enable_post = 1` |
| 重置／收尾 | `audio_pipeline_4ch_reset()`、`audio_pipeline_4ch_destroy()` | reset 不重置你的後級（§3.1） |
| 強度（你的後級） | `aec_res_set_preset()`、`aec_res_split_floor()`（作用在你自己的 `AecRes`）；`anr_set_mode()`（作用在你的降噪器） | 單元沒有強度 setter，也不持有 `AecRes` |
| 形狀與生效值 | `audio_pipeline_4ch_hop_size()`、`_frame_size()`、`_fft_size()`、`_n_freqs()`、`_sample_rate()`、`_doa_*`、`_bf_*`、`_bf_effective_adapt_interval()`、`_bf_lambda()` | 無效 handle 回 `-1`（`_bf_lambda()` 回 NaN） |
| 診斷（唯讀） | `audio_pipeline_4ch_get_core()`、`audio_pipeline_4ch_last_core_timing()`、`audio_pipeline_4ch_spatial_backend()`、`_matched_filter_count()`、`_linear_aec_count()` | 時序欄位預設全 0 |
| 你串的後級 | `aec_config_from_preset()`、`aec_res_get_mem_size()`／`aec_res_init()`、`se_res_stage_gain()`、`se_anr_stage_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`、`ac_istft_push()`、`se_hop_is_finite()` | `fuse()` 之後 `frame.g_res_ctx` 恆為 NULL，RES 增益由你的 `AecRes` 經 `se_res_stage_gain()` 算出 |

**內部（不要 include、不要依賴）**：`bf.h`、`srp.h` 與 `<audiolib>/{doa,bf,common/utility}` 的所有函式（單元擁有它們的生命週期，
BF 借用 SRP 的 steering 表，所以 destroy 次序是 BF → SRP → 核心）；`4aec_nr_res_internal.h`、`4aec_projection_kernels.h`；
單元 `.c` 內的 static 函式（含 `build_flags_hash` 的計算）。

### 0.3 參數面：外部與內部

欄位的逐項範圍、預設與驗證規則在 §4；下表是分級與理由。結構在 `audio_pipeline_4ch.h:45-75`（`AudioPipeline4ChConfig`），核心部分在 `4aec_nr_res.h:96-111`。

**外部（部署資訊或單一取捨軸，建議讓第三方整合者設）**

| 欄位 | 位置 | 何時設定 | 作用 |
|---|---|---|---|
| `core.sample_rate`、`core.fft_size`、`core.filter_length`、`core.delay_mode`、`core.delay_num_filters`、`core.fixed_delay_samples`、`core.capture_proxy_channel`、`core.max_delay_ms`、`core.enable_post` | `AudioPipeline4ChConfig.core`（§4.1；分級同 [核心手冊 §0.3](integration_4ch_core_zh_TW.md)） | init 前，不可變 | 格點、AEC 長度、對齊方式、是否可 `fuse()` |
| `geometry`、`uca_radius_m` ／ `ula_spacing_m` ／ `microphone_x_m[4]`+`microphone_y_m[4]` | §4.2 | init 前 | 麥克風陣列的實體資訊；預設 UCA 半徑 0.035 m 只是讓範例能跑，產品必須設成實測值 |
| `num_angles` | §4.3 | init 前 | 角度解析度；**最大的記憶體旋鈕**（72 → 360 約 1.70 MB → 4.70 MB，§4.7） |
| 後級強度 | `aec_res_set_preset()`（你的 `AecRes`）、`anr_set_mode()`（你的降噪器） | hop 之間；單元本身沒有 RES 欄位，初始強度由你傳給 `aec_config_from_preset()` 的 preset 決定 |

進階外部（有明確使用情境才動，改了要自己重跑驗證）：`bf_enable`（0 = 不 beamform，直接輸出 channel 0）、
`bf_fixed_mode`＋`bf_fixed_doa_rad`（說話方向已知的部署）、`doa_low_freq_hz`／`doa_high_freq_hz`（定位頻帶，須 `< Nyquist`）。

**內部（存在、可調，但不建議開放）**

| 欄位／項目 | 位置 | 為什麼不開放 |
|---|---|---|
| `bf_lambda`、`bf_adapt_interval`、`bf_fixed_align_notebook` | §4.4、§4.6 | 設定值與生效值不同：`bf_lambda` 依 hop 與生效 interval retime，fixed-notebook 模式強制 interval = 1；要讀回生效值才不會誤判 |
| `bf_mu` | §4.4 | RLS 步長增益，與 `bf_lambda` 聯合調過，不 retime |
| `doa_enable_smoothing`、`doa_switch_consecutive`、`doa_angle_tolerance_rad`、`doa_update_interval` | §4.3 | 追不上方向變化與誤切換是同一條軸上的取捨，放給外部等於把我們的取捨丟出去 |
| `speed_of_sound_m_s` | §4.2 | 一般不需要改（343 m/s） |
| `auto_vad_threshold_dbfs`、`auto_vad_snr_ratio`、`auto_vad_hangover_frames` | §4.5 | 只在 `audio_pipeline_4ch_process()` 的 fallback VAD 用到；正式整合用自己的 VAD 走 `process_with_activity()`；`auto_vad_hangover_frames` 沒有讀回函式 |
| 四條 lane 與核心內部設定 | [核心手冊 §0.3](integration_4ch_core_zh_TW.md) | 同核心 |
| `core.delay_est_period_s`、`core.delay_backward_quarantine_*` | §4.1 | 預設關閉；打開會改變選到的 lag 或只延後一個窗 |
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
強度用 `aec_res_set_preset()`（§3.3）。

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

## 2. Quick start

完整、可編譯、已實際跑過的最小整合（單元 heap 路徑；RES、NR 與合成由應用自己初始化）：

```c
#include <stdio.h>
#include <stdlib.h>
#include "audio_pipeline_4ch.h"
#include "anr.h"
#include "audio_stft.h"
#include "aec_res.h"
#include "samples/se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

int main(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    cfg.geometry     = AUDIO_PIPELINE_4CH_GEOMETRY_UCA;
    cfg.uca_radius_m = 0.035f;                     /* 換成你陣列的真實半徑 */

    AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);   /* heap path */
    if (!p) { fprintf(stderr, "create failed\n"); return 1; }

    /* RES (libaec AecRes), NR and synthesis are separate libraries the application sizes and
     * initialises itself; the unit owns none of them. */
    const int hop = audio_pipeline_4ch_hop_size(p);
    const int n = audio_pipeline_4ch_n_freqs(p);
    const int fft = audio_pipeline_4ch_fft_size(p);
    AecConfig res_cfg;
    aec_config_from_preset(&res_cfg, AEC_PRESET_BALANCED, 16000);
    res_cfg.fft_size = fft;                                  /* the unit's grid */
    AnrConfig nr_cfg = anr_config_for_mode_grid(16000, fft, ANR_MODE_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, fft);
    size_t res_bytes = aec_res_get_mem_size(&res_cfg);       /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t vec = ALIGN16((size_t)n * sizeof(float));
    size_t total = ALIGN16(res_bytes) + ALIGN16(nr_bytes) + ALIGN16(synth_bytes) + 2 * vec +
                   ALIGN16((size_t)n * sizeof(Complex));
    void* post_pool;
    if (res_bytes == 0 || nr_bytes == 0 || synth_bytes == 0 ||
        posix_memalign(&post_pool, 16, total) != 0) return 1;
    uint8_t* cur = (uint8_t*)post_pool;
    AecRes* res = aec_res_init(cur, res_bytes, &res_cfg);    cur += ALIGN16(res_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg); cur += ALIGN16(synth_bytes);
    if (!res || !nr || !istft) return 1;
    float* total_gain = (float*)cur;  cur += vec;
    float* extra_noise = (float*)cur; cur += vec;
    Complex* spec = (Complex*)cur;
    uint32_t rng = SE_CNG_RNG_SEED;

    float mics[4 * 1024], ref[1024], out[1024];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                    /* 換成你自己的擷取來源 */
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
                mics[i * FOUR_AEC_NR_RES_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }

        AudioPipeline4ChBeam beam;
        AudioPipeline4ChFrameInfo info;
        SeSpecFrame f;
        const float* g_res = NULL;
        const float* g_nr = NULL;
        int status = audio_pipeline_4ch_process(p, mics, ref, &beam, &info);   /* lanes + DOA + BF */
        if (status == FOUR_AEC_NR_RES_OK)
            status = audio_pipeline_4ch_fuse(p, &f);          /* SeSpecFrame; f.g_res_ctx is NULL */
        if (status == FOUR_AEC_NR_RES_OK)
            g_res = se_res_stage_gain(res, &f);               /* RES gain from your AecRes */
        if (g_res)
            g_nr = se_anr_stage_gain(nr, &f, extra_noise, 1);
        if (g_nr) {
            sk_min_f32(total_gain, g_nr, g_res, n);
            sk_capply_gain_f32(spec, f.e, total_gain, n);
            if (f.cn) aec_cng_fill(&rng, spec, f.cn, g_res, g_nr, SE_CNG_NR_GAIN_FLOOR, n);
            (void)ac_istft_push(istft, spec, out);
        }
        if (!g_nr || !se_hop_is_finite(out, hop)) {
            fprintf(stderr, "status=%d\n", status);
            audio_pipeline_4ch_reset(p);                   /* a refused call: reset every stage */
            aec_res_reset(res);
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = SE_CNG_RNG_SEED;
            continue;
        }
        /* 消費 out：hop 個 float，已增強的 mono 音訊；
         * info.doa_used_rad / info.vad_out / info.delay 與 beam.spectrum 是唯讀 telemetry。 */
    }

    printf("ok hop=%d effective_adapt_interval=%d lambda=%.9f\n", hop,
           audio_pipeline_4ch_bf_effective_adapt_interval(p),
           (double)audio_pipeline_4ch_bf_lambda(p));
    anr_destroy(nr);                                       /* AecRes has no destroy */
    free(post_pool);
    audio_pipeline_4ch_destroy(p);
    return 0;
}
```

編譯：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines/4ch_aec_bf_nr_res \
  -I pipelines \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audiolib>/common/include \
  -I <audiolib>/common
c++ -o app app.o \
  <4ch-bin>/libaudio_pipeline_4ch.a \
  <spatial-bin>/libdoa.a <spatial-bin>/libbf.a <spatial-bin>/libspatial_common.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

要接 ULCNet 後濾波時，不呼叫 `fuse()`、`core.enable_post = 0`，把 `beam.spectrum`、同一次呼叫的**原始** far hop 與
`info.delay.changed`／`info.delay.solid` 交給 `ulcnet_nn_stage_begin_complex()`（見 `pipelines/4ch_alignulcnet/README.md`）。

---

## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `audio_pipeline_4ch_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`out` 為 NULL；`cfg` 未通過本層驗證（§4.2）或核心層驗證（§4.1）；SRP 或 BF 回報 size 0；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體 | 拿到 `req` 之後 | pool **不需要**預先清零（poison pattern 可接受）。`init_ex()` 自己會把整塊 memset 成 0 |
| 3 | init | `audio_pipeline_4ch_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；core / SRP / BF 任一 init 失敗。**沒有任何診斷輸出**（見 §5.4） |
| 4a | process | `audio_pipeline_4ch_process_front()` → `audio_pipeline_4ch_process_beam()`；或 `process_with_activity()`；或 `process()` | 每收到一個 hop 呼叫一次 | `-1` / `-2` / `-3`，見 §5.2 |
| 4b | fuse（可選） | `audio_pipeline_4ch_fuse(p, &frame)` | 同一 hop 的 beam 之後、下一次 `process_front()` 之前 | `-1` / `-2` / `-3`，見 §5.2 |
| 4c | 你的後級 | `se_res_stage_gain()` → `se_anr_stage_gain()` → `sk_min_f32()` → `sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`，或 NN stage | fuse（或 beam）之後 | 由你負責；`frame.g_res_ctx` 為 NULL，RES 增益來自你自己的 `AecRes`。任何一步被拒，你必須把單元與你的後級（含 `aec_res_reset()`）**一起** reset（§3.1） |
| 5 | reset | `audio_pipeline_4ch_reset(p)`，加上你的後級的 reset | 換回聲路徑、換場景、切換到不相關的另一條串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op |
| 6 | destroy | `audio_pipeline_4ch_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。NULL 安全。pool instance 冪等；`create()` instance 只能呼叫一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`audio_pipeline_4ch_create(&cfg)` 把第 1~3 步合成一步。

`audio_pipeline_4ch_init(pool, bytes, &cfg)` 等同
`audio_pipeline_4ch_init_ex(pool, bytes, &cfg, NULL)` —— 不做 descriptor 檢查。

### 3.1 `reset()` 到底重置了什麼，以及失敗時整合者要做什麼

單元的 `reset()` 重置：

* 前端（delay matcher、四路 AEC、front 有效旗標）
* SRP-PHAT 狀態
* BF 狀態
* 本層的內建 VAD 狀態（`noise_power` 回到門檻值、hangover 歸零）
* `frame_index` 與 per-hop phase 歸零

它**不會**重置你的後級（`AecRes`、NR、CNG rng、合成、NN stage），也**不會**重新驗證 config，
更不會重新計算 §4.6 那三個「生效值」——那三個是在 init 時算好的，`reset()` 不動它們。

**失敗語意是整合者的責任。** 單元的任何一個呼叫回非 `OK`，或你的 `se_res_stage_gain()`／`se_anr_stage_gain()` 回 `NULL`，
或合成輸出不是有限值，你要**同時**呼叫 `audio_pipeline_4ch_reset()`、`aec_res_reset()`、`anr_reset()`、`ac_istft_reset()`，
並把 comfort-noise RNG 設回 `SE_CNG_RNG_SEED`，以及用到 NN stage 時的 `ulcnet_nn_stage_reset()`。
漏掉其中一個，系統就不等於新建的實例。DFN2 的 stage 不因為 AEC 的 delay change 而 reset（見
`pipelines/dfn_res_stage.h`）。

### 3.2 執行期唯讀存取

形狀 accessor（`p == NULL` 或已 destroy 時全部回 `-1`）：

| 呼叫 | 回傳 |
|---|---|
| `audio_pipeline_4ch_hop_size(p)` | 每次 process 要餵幾個 sample（per channel） |
| `audio_pipeline_4ch_frame_size(p)` | frame 長度（等於 FFT 長度） |
| `audio_pipeline_4ch_fft_size(p)` | FFT 長度 |
| `audio_pipeline_4ch_n_freqs(p)` | 頻域 bin 數 |
| `audio_pipeline_4ch_sample_rate(p)` | 實際生效的取樣率 |
| `audio_pipeline_4ch_doa_sample_rate/frame_size/hop_size/fft_size(p)` | DOA 階段的格點 |
| `audio_pipeline_4ch_bf_sample_rate/frame_size/hop_size/fft_size(p)` | BF 階段的格點 |

AEC、DOA、BF，以及你串的 RES 與 NR，全部**共用同一組格點**，`frame == fft_size`、`hop == fft_size / 2`。
你不能分別指定互相矛盾的 frame / hop / FFT。上面三組 accessor 之所以分開，
是讓整合測試能證明三者確實一致（實測 16 kHz 預設下三組皆為 hop 128 / fft 256 / n_freqs 129）。

結構稽核 accessor（`p == NULL` 或已 destroy 時回 `0`）：

| 呼叫 | 有效 handle 的回傳 |
|---|---|
| `audio_pipeline_4ch_matched_filter_count(p)` | `1`（`MATCHED`）；`FIXED`／`EXTERNAL_ALIGNED` 時 `0` |
| `audio_pipeline_4ch_linear_aec_count(p)` | `4` |

單元的結構稽核計數只有上面兩個；RES、NR 與合成由你自己決定，單元不持有它們。

生效值 accessor（見 §4.6）：`audio_pipeline_4ch_bf_effective_adapt_interval(p)`、
`audio_pipeline_4ch_bf_lambda(p)`。

診斷：`audio_pipeline_4ch_get_core(p)`（唯讀的 `const FourAecNrRes*`，給核心層的稽核 accessor 與
`four_aec_nr_res_get_last_timing()` 用；不要拿它去呼叫會改狀態的函式）與
`audio_pipeline_4ch_last_core_timing(p, &t)`（核心最近一個 hop 的各階段耗時，`FourAecNrResLastTiming`；
`p` 為 NULL 時把 `t` 清零，`t` 不得為 NULL）。

建置資訊：`audio_pipeline_4ch_spatial_backend()` —— **不吃 handle**，回傳一個
`const char*`（永不為 NULL），表示 spatial kernel 實際編進去的是哪一種實作
（NEON 或 scalar）。純資訊用途，可寫進你的開機 log。

### 3.3 執行期強度控制

單元**沒有**強度 setter 函式，也不持有 `AecRes`：四條 lane 只算線性殘留，RES 與 NR 都是你串的後級。
你對自己 init 的物件直接呼叫函式庫的 setter（完整契約見核心手冊 [`integration_4ch_core_zh_TW.md` §3.2](integration_4ch_core_zh_TW.md)）：

```c
aec_res_set_preset(res, AEC_PRESET_AGGRESSIVE, 100.0f);   /* res 是你自己的 AecRes；回 0 成功 */
anr_set_mode(nr, ANR_MODE_MILD);                  /* 函式庫的 setter；nr 是你自己的降噪器 */
float live, target;                               /* 線性功率 */
if (aec_res_split_floor(res, &live, &target) == 0 && live == target) {
    /* ramp 已經走完 */
}
```

這裡只重述整合時最容易踩到的一點：

> **不要自己對四條 lane 迴圈呼叫 `aec_set_preset()`。** 四條 lane 都以
> `spatial_linear_context` 建立，永遠不會走到自己的 suppression-gain 路徑，
> 它們的地板什麼都不塑形。真正乘上輸出的 gain 來自你串的**唯一一個**
> post-beam `AecRes`；同樣地，NR 也只有一個共用實例，不是每條 lane 一個。
> 上面兩個 setter 針對的就是那兩個共用實例。單元的 `reset()` 不碰它們，
> 你自己呼叫的 `aec_res_reset()` 不會丟掉用 `aec_res_set_preset()` 設的 preset。

`ramp_ms == 0` 代表下一個 hop 就套用（**不是錯誤**），`> 0` 則以 dB 為單位線性走
過去，上限 60 秒。兩個 setter 都不是重啟：AEC、SRP、BF、NR 的狀態全部繼續跑；
preset 的改動撐得過該 stage 的 reset，ramp 進度不會。**在兩個 hop 之間
呼叫、與 process 序列化；非 thread-safe。**

A/B 量測時：far-active 地板只在 **far-active 且非 double-talk** 的 hop 上生效
（DT 地板：mild/balanced 共用 -20 dB，aggressive 為 -30 dB；latch 觸發前用的是 far-silent 地板），而同一個 gain
還決定 comfort noise 量，所以整段錄音的平均值移動幅度會小於 dB 落差所暗示的量。
細節與建議的比較方式見核心手冊 §3.2。

---

## 4. Config 完整參考

`AudioPipeline4ChConfig` = 巢狀的 `FourAecNrResConfig core`（12 個欄位）
+ 本層自己的 23 個欄位（幾何 6、DOA 7、BF 7、內建 VAD 3）。RES、NR、comfort noise 的設定**不在**這裡（它們屬於你的後級串接）。取得預設值的唯一正確方式：

```c
AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`audio_pipeline_4ch_default_config()` **不做驗證**，只是填值。

### 4.1 `cfg.core`（核心層欄位）

本層會把 `cfg.core` 原封不動交給核心層驗證與建構，規則完全一樣：

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `core.sample_rate` | `int` | 你傳給 `default_config()` 的值 | **只有 `16000` 或 `48000`** | 你音訊的取樣率 |
| `core.fft_size` | `int` | `0` | `0` = rate 預設（16 kHz → 256，48 kHz → 1024）；16 kHz 另可指定 `512`；48 kHz 只接受 `1024` | 一般留 `0` |
| `core.filter_length` | `int` | `0` | `[0, 4096]` | `0` = AEC 依取樣率算的預設（16 kHz → 832，48 kHz → 3072）。記憶體吃緊時最有效的旋鈕 |
| `core.capture_proxy_channel` | `int` | `0` | `[0, 3]` | 共用 delay matcher 用哪一支麥克風 |
| `core.delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1` | 預設關閉。開啟後只隔離**一種**共用估計：比現行 accepted delay **更早**（backward／pre-echo 方向）、且 **`capture_proxy_channel` 那一條 lane**（不是任一條——那正是餵給共用 estimator 的那支 mic，也是唯一能對這個估計提供證據的 lane）在目前對齊上仍明顯在消除回音（`aec_linear_is_cancelling()`）。**往後跳（delay 變大）永不隔離**，**首次取得對齊不受影響**。隔離**有上限**（見下一列），到期就採用；消除能力崩掉時則立即採用，但這是 hard replacement 常見而非所有真實路徑變化必然具備的現象。**它不是 pre-echo 誤鎖的解藥**：那個誤鎖已在估計器端根治（aggregator 改為回報 dominant peak，見 `AEC/docs/aec_methods.md`），這個機制擋的是真實路徑往前移動時的候選變動，仍然只是延後一個窗。`FIXED` / `EXTERNAL_ALIGNED` 不重新決定對齊，此旗標對它們無作用 |
| `core.delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 上一列的隔離窗長（秒），在 `init` 時按解析後的 hop／取樣率換算成 hop 數（最小 1；要關閉請用上一列的 enable）。enable 為 0 時無作用，但仍會被驗證——設定檔今天就要是「翻一個旗標也不會壞」的狀態。實測（multipath 場景，舊路徑不消失所以消除能力不崩、只有到期能釋放）：窗 1.0 s = 62 hops，採用點精確等於「未隔離 hop ＋ 窗長」。 |
| `core.max_delay_ms` | `float` | `1024.0f` | 有限值，`[0.0, 4096.0]` | 設成略大於系統真實最大 mic↔ref 延遲。設太大只是白吃記憶體 |
| `core.enable_post` | `int`（bool） | `1` | `0` 或 `1` | `1` = lane 保留 post context，`audio_pipeline_4ch_fuse()` 可用；`0` = 只留線性 context（`AEC_CONTEXT_LINEAR_ONLY`），`fuse()` 回 `INVALID_ARGUMENT`，給直接吃 BF 頻譜的 ULCNet。單元的驗證**不再**要求它為 1 |

另外還有 `core.delay_mode`、`core.delay_num_filters`、`core.fixed_delay_samples`、`core.delay_est_period_s`
（語意見核心手冊 §4.1）。核心沒有 `aec_preset`、`nr_mode`、`enable_nr`、`enable_cng`、`legacy_amin` 欄位：
lane 一律以 `AEC_PRESET_BALANCED` 導出並只算線性殘留。RES 不是本層欄位，由你自己的 `AecRes` 持有（§1.1、§2）。

> Quarantine 窗只對「連續符合條件的 backward episode」有界。候選轉為
> forward／無效、confidence 中斷，或 proxy lane 的 cancellation 證據消失時
> 會解除本次隔離；之後的新 backward episode 重新計時。Cancellation collapse
> 常見於路徑完全替換，但不是 multipath／新增路徑的必然特徵。
>
> **產品預設維持 `0`。** 打開之後只是把誤鎖往後推一個窗，不會修正誤鎖本身。

Grid 只有三組（`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`）：

| `core.sample_rate` | `core.fft_size` | `hop` | `n_freqs` |
|---:|---:|---:|---:|
| 16000 | 256（rate 預設） | 128 | 129 |
| 16000 | 512 | 256 | 257 |
| 48000 | 1024（rate 預設） | 512 | 513 |

### 4.2 陣列幾何（4 + 2 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `geometry` | `AudioPipeline4ChGeometry` | `AUDIO_PIPELINE_4CH_GEOMETRY_UCA`（= 0） | `UCA`(0) / `ULA`(1) / `CUSTOM`(2)。其他值拒絕 | 依你的實體陣列選 |
| `uca_radius_m` | `float` | `0.035f` | **僅在 `geometry == UCA` 時檢查**：必須有限且 `> 0` | 圓陣半徑（公尺）。量你的板子 |
| `ula_spacing_m` | `float` | `0.035f` | **僅在 `geometry == ULA` 時檢查**：必須有限且 `> 0` | 線陣相鄰間距（公尺） |
| `microphone_x_m[4]` | `float[4]` | 依預設半徑 0.035 排出的 UCA 座標 | **僅在 `geometry == CUSTOM` 時檢查**：4 個都必須是有限值 | 每支麥克風的 x 座標（公尺） |
| `microphone_y_m[4]` | `float[4]` | 同上 | 同上 | 每支麥克風的 y 座標（公尺） |
| `speed_of_sound_m_s` | `float` | `343.0f` | 有限且 `> 0` | 一般不需要改 |

**重要陷阱**：`geometry == UCA` 時，實際幾何是**從 `uca_radius_m` 現算的**，
`microphone_x_m` / `microphone_y_m` 的內容**會被忽略**。
同理 `ULA` 只看 `ula_spacing_m`。只有 `CUSTOM` 才會讀那兩個陣列。
`default_config()` 之所以順手把兩個陣列填成預設 UCA 座標，是為了讓你直接切成
`CUSTOM` 時不會拿到未初始化的值 —— 但那組座標對應的是**預設半徑**，
不是你之後改過的 `uca_radius_m`。要用 `CUSTOM` 就自己把兩個陣列填滿。

### 4.3 Direction finding（DOA，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `num_angles` | `int` | `72` | `[4, 3600]` | 角度掃描解析度。**這是本層最大的記憶體旋鈕**（見 §4.7）。72 相當於每 5 度一格 |
| `doa_low_freq_hz` | `float` | `300.0f` | 有限、`>= 0`、且 **`<` Nyquist**（`sample_rate / 2`） | 定位用的頻帶下限 |
| `doa_high_freq_hz` | `float` | `7000.0f` | 有限、`>= 0`、`<=` Nyquist；若 `> 0` 則必須 `>= doa_low_freq_hz` | 定位用的頻帶上限。**`0` 代表自動**，實際取 `min(7000, Nyquist)` |
| `doa_enable_smoothing` | `int`（bool） | `1` | `0` 或 `1` | `1` = 啟用角度平滑 |
| `doa_switch_consecutive` | `int` | `3` | **必須 `> 0`** | 要連續幾幀指向新方向才真的切換。調大 = 更遲鈍但更穩 |
| `doa_angle_tolerance_rad` | `float` | `10° ≈ 0.174533` rad | 有限且 `>= 0` | 多大的角度差才算「換方向」 |
| `doa_update_interval` | `int` | `2` | **必須 `> 0`** | 每幾個 hop 更新一次 DOA |

### 4.4 Beamformer（BF，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `bf_enable` | `int`（bool） | `1` | `0` 或 `1` | `0` = 關閉 BF：直接輸出 channel 0，並在每個 bin 送出單位權重 `[1,0,0,0]`，因此 lane 1–3 權重全零。全零權重的 lane 不參與 post 級的三個縮減，見 [`integration_4ch_core_zh_TW.md` §4.3](integration_4ch_core_zh_TW.md) |
| `bf_lambda` | `float` | `0.995f` | 有限，且 `> 0.0` 並 `<= 1.0` | 自適應的遺忘因子。**你設的值不是實際生效的值，見 §4.6** |
| `bf_mu` | `float` | `0.1f` | 有限且 `>= 0` | 步長增益。**不會**被 retime，設多少就是多少 |
| `bf_fixed_mode` | `int`（bool） | `0` | `0` 或 `1` | `1` = 用固定方向，不吃 DOA 結果 |
| `bf_fixed_doa_rad` | `float` | `0.0f` | 有限值（不限範圍） | `bf_fixed_mode == 1` 時採用的固定方向（弧度） |
| `bf_fixed_align_notebook` | `int`（bool） | `0` | `0` 或 `1` | 固定對齊模式。**開啟時會強制改變 `bf_adapt_interval` 的生效值，見 §4.6** |
| `bf_adapt_interval` | `int` | `1` | **必須 `> 0`** | 每幾個 hop 做一次自適應更新。**你設的值不一定是實際生效的值，見 §4.6** |

### 4.5 內建 fallback VAD（3 欄）

**這三個欄位只在你呼叫 `audio_pipeline_4ch_process()` 時才會被用到。**
如果你走 `audio_pipeline_4ch_process_with_activity()`（產品整合建議用這條），
活動偵測完全由你提供，這三個欄位不影響任何行為 —— 但**仍然會被驗證**，
所以還是要保持合法值（照 `default_config()` 留著即可）。

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `auto_vad_threshold_dbfs` | `float` | `-55.0f` | 必須是有限值（不限範圍） | 判定為語音的最低能量門檻（dBFS）。調高 = 更不容易誤判成語音 |
| `auto_vad_snr_ratio` | `float` | `3.0f` | 有限且 **`>= 1.0`** | 相對背景噪聲要高出幾倍才算語音 |
| `auto_vad_hangover_frames` | `int` | `8` | 必須 `>= 0` | 語音結束後再保持多少幀。**你設的值不是實際生效的值，見 §4.6** |

### 4.6 你設定的值不一定是實際生效的值

**這一節是本手冊最重要的一節。** 有三個欄位在 init 時會被靜默轉換。
你在 `AudioPipeline4ChConfig` 裡讀到的值，和 pipeline 內部真正使用的值**不同**。
不知道這件事，你會在調參時得到完全無法解釋的結果。

#### (1) `bf_adapt_interval` —— 在 fixed-notebook 模式下被強制成 1

規則（來自 `bf_effective_adapt_interval()`）：

```
effective_interval = (bf_adapt_interval > 0) ? bf_adapt_interval : 1
if (bf_fixed_mode == 1 && bf_fixed_align_notebook == 1)
    effective_interval = 1        /* 不管你設了什麼 */
```

也就是說：只要 `bf_fixed_mode` 與 `bf_fixed_align_notebook` **同時**為 1，
你設的 `bf_adapt_interval = 4` 會被完全忽略，實際就是每一個 hop 更新一次。

**讀回生效值**：

```c
int effective = audio_pipeline_4ch_bf_effective_adapt_interval(p);
```

#### (2) `bf_lambda` —— 依 hop × 生效 interval 重新定時

`bf_lambda` 是以「每 10 ms 更新一次」為基準調出來的係數。
真正的遺忘時間常數會隨 `hop_size` / `sample_rate` 改變，所以 init 時會做換算：

```
effective_lambda = configured_lambda ^ ( (hop_size * effective_interval / sample_rate) / 0.010 )
```

注意指數裡用的是 **(1) 算出來的 `effective_interval`**，不是你設的原始值。
（`bf_mu` 是步長增益、不是時間常數，所以**不做**這個換算。）

**讀回生效值**：

```c
float effective = audio_pipeline_4ch_bf_lambda(p);
```

實測對照（`bf_lambda = 0.995`）：

| `sample_rate` | `hop` | 你設的 `bf_adapt_interval` | `bf_fixed_mode` / `bf_fixed_align_notebook` | 生效 interval | 生效 lambda |
|---:|---:|---:|---|---:|---:|
| 16000 | 128 | 1 | 0 / 0 | 1 | 0.995998025 |
| 16000 | 128 | 4 | 0 / 0 | 4 | 0.984087825 |
| 16000 | 128 | 4 | **1 / 1** | **1** | **0.995998025** |
| 16000 | 128 | 4 | 1 / 0 | 4 | 0.984087825 |
| 48000 | 512 | 1 | 0 / 0 | 1 | 0.994667590 |

看第 2 列與第 3 列：`bf_adapt_interval` 明明都是 4，
只因為多開了 `bf_fixed_align_notebook`，interval 與 lambda **兩個都變了**。

#### (3) `auto_vad_hangover_frames` —— 依 hop 重新定時

這個欄位的單位是「以 10 ms 為一幀」的幀數，init 時會換算成這個 grid 的實際幀數：

```
effective_frames = ceil( auto_vad_hangover_frames * sample_rate / (100 * hop_size) )
```

（`auto_vad_hangover_frames == 0` 時生效值也是 0，即關閉 hangover。）

實測對照（預設 `auto_vad_hangover_frames = 8`）：

| `sample_rate` | `hop` | 生效幀數 | 實際時間長度 |
|---:|---:|---:|---|
| 16000 | 128 | **10** | 80 ms |
| 16000 | 256 | **5** | 80 ms |
| 48000 | 512 | **8** | ≈ 85.3 ms |

**這個欄位沒有 read-back accessor。**
本層沒有提供 `audio_pipeline_4ch_..._vad_hangover_frames()` 之類的函式，
所以你只能用上面的公式自己算。（`bf_adapt_interval` 與 `bf_lambda` 兩個有，
`auto_vad_hangover_frames` 沒有 —— 這是目前 API 的不對稱之處。）

如果你走 `process_with_activity()`，這一項對你完全沒有影響。

#### 三個生效值的共同規則

* 都在 **init 時**算好並存起來。
* `reset()` **不會**重算它們（reset 只清 runtime 狀態）。
* 想改它們，唯一的方法是用新的 config 重新 init 一個 instance。

### 4.7 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `core.sample_rate` / `core.fft_size` | 會（改 grid） |
| `core.filter_length` | 會（四路 AEC 等比縮放） |
| `core.delay_mode` / `core.delay_num_filters` / `core.fixed_delay_samples` | 會（見核心層手冊 §4.5/4.6a——僅省一份共用 estimator/ring，不是四份） |
| `core.max_delay_ms` | 會（僅 `MATCHED` 用於 delay ring 大小） |
| **`num_angles`** | **會，而且影響很大**（見下表） |
| 其他 DOA / BF / VAD 欄位 | 不變 |
| `geometry` 與座標 | 不變 |

ne10、`SIMD=1`、`delay_mode=MATCHED` 預設的單元 `req.bytes`
（`layout_version=21`；請以 `--print-mem-size` 重查；最後一欄標明每一列的來源）。單元**不含** RES、NR 與合成；
RES（`AecRes`）、NR、`AcIstft` 與增益／頻譜暫存（範例程式自己各自查大小後加總）另計，加起來是
`audio_pipeline_4ch_static --print-mem-size` 的 `Total:`：

| Config | 單元 `req.bytes` | RES + NR + 合成 + 暫存 | 合計 | 來源 |
|---|---:|---:|---:|---|
| 16000，全預設（256/128，`num_angles=72`） | 1,698,864 | 66,688 | 1,765,552 | 實測 |
| 16000，`core.fft_size = 512` | 2,992,496 | 96,464 | 3,088,960 | 實測 |
| 16000，`num_angles = 360` | 4,700,976 | 66,688 | 4,767,664 | 單元實測（直接呼叫 `get_mem_requirements()`）；合計為相加 |
| 48000，全預設（1024/512，`num_angles=72`） | 6,300,432 | 225,056 | 6,525,488 | 實測 |

範例程式在單元之外另切的 `AecRes`：16 kHz/256 為 7,904 B、16 kHz/512 為 14,048 B、48 kHz/1024 為 29,408 B
（`aec_res_get_mem_size()`，已含在上表第三欄）。`core.enable_post = 0` 時單元的 `req.bytes`：16 kHz/256 為 1,693,648、
16 kHz/512 為 2,982,160、48 kHz/1024 為 6,279,856。

⚠ 覆蓋差異：只有 `num_angles = 72` 的 16 kHz/256 與 48 kHz/1024 兩組會被
C 關卡自動驗證（static smoke 各印一次 `Total:` bytes）。`core.fft_size = 512`、
`num_angles = 360` 與 `core.enable_post = 0` 的各列是**手動查詢**得到的，沒有任何自動測試會在這些
config 上呼叫 `audio_pipeline_4ch_get_mem_requirements()`，所以它們不會隨程式
改動自動失效——引用前請自己現查一次。

`num_angles` 從 72 調到 360，單元的記憶體約從 1.70 MB 變成約 4.70 MB（量級可信，
精確值見上）。**先確認你的角度解析度真的需要那麼細，再調這個值。**

換 backend、換編譯選項、更新 submodule 都會讓上表失效。
**實際配置一律以 `audio_pipeline_4ch_get_mem_requirements()` 現查的 `req.bytes`，加上你各後級 stage（RES 的 `AecRes`、NR、合成）自己查的大小為準。**

---

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
你的 `se_res_stage_gain()` 拒絕該 frame（回 `NULL`）時單元不會自動 reset，單元與你的後級要由你一起 reset（§3.1）。

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

## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | `libaudio_pipeline_4ch.a`（object 之一） | **本手冊的主體**（共用 delay 對齊 + 四條線性 AEC + SRP-PHAT + BF 單元；不含 RES／NR／合成） |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | `libaudio_pipeline_4ch.a`（object 之二） | 核心層，本層一定會用到 |
| `pipelines/common/se_config.c` | `libaudio_pipeline_4ch.a`（object 之三） | 共用組態 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` | — | 核心層公開 API（本 header 會 include 它） |
| `<audiolib>/doa/` | `libdoa.a` | SRP-PHAT / steering / DOA 平滑。可重用函式庫 |
| `<audiolib>/bf/` | `libbf.a` | BF beamformer。可重用函式庫 |
| `<audiolib>/common/utility/` | `libspatial_common.a` | spatial kernel 與 complex 輔助 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_internal.h`、`4aec_projection_kernels.h` | — | 內部 header，**不是**公開 API，不要 include |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/samples/se_anr_stage.h`、`ulcnet_nn_stage.{h,c}`、`dfn_hop.h` | header-only／單檔範例碼 | NR 級的 gain 呼叫與 NN stage：沒有 ABI、沒有 layout version、不進任何 archive，可整份複製改寫 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | caller-owned pool 路徑的主機參考程式 | 示範單元加 RES（libaec `AecRes`）／NR／合成的 query → allocate → init_ex → process → fuse → 後級 → destroy → release（RES、NR、合成與暫存另有一塊 pool）。真正的 SRP-PHAT + BF 在這個 process 裡跑，不是固定權重的替身 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | **核心層**（外部 beamformer seam）的 pool 路徑參考程式 | 屬於另一份手冊。它在 front/fuse 之間塞的是固定等權重的 smoke adapter，不是 production beamformer |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | raw-float 錄音驗證 runner（單元加 RES／NR／合成） | 主機一次性工具，刻意走 heap `create()` 路徑（不是板端佈署示範） |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、
`static_main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試。
* `.py` 檔（`pipeline.py`、`evaluate_recordings.py`、`evaluate_external_recordings.py`）
  —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 七個 archive（`libaudio_pipeline_4ch.a` / `libdoa.a` / `libbf.a` /
      `libspatial_common.a` / `libaec.a` / `libanr.a` / `libaudio_common.a`）
      全部是用**同一組** backend、`SIMD` 與編譯選項建出來的
- [ ] link 順序照 §1.5
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `bf.h` / `srp.h` / 任何 `audiolib` header
- [ ] 沒有 include `4aec_nr_res_internal.h` 或 `4aec_projection_kernels.h`
- [ ] 開機 log 有記錄 `audio_pipeline_4ch_spatial_backend()`

Config：

- [ ] `core.sample_rate` 是 16000 或 48000（**不是** 8000）
- [ ] `geometry` 與對應的幾何欄位配對正確
      （UCA → `uca_radius_m`；ULA → `ula_spacing_m`；CUSTOM → 兩個座標陣列）
- [ ] 用 UCA/ULA 時，你知道 `microphone_x_m` / `microphone_y_m` **會被忽略**
- [ ] `num_angles` 是依實際需要的角度解析度設的，不是隨手加大的（記憶體影響很大）
- [ ] `doa_low_freq_hz < Nyquist`、`doa_high_freq_hz <= Nyquist`
- [ ] `doa_switch_consecutive`、`doa_update_interval`、`bf_adapt_interval` 都 `> 0`
- [ ] `bf_lambda` 落在 `(0, 1]`
- [ ] `auto_vad_snr_ratio >= 1.0`

生效值（§4.6）：

- [ ] 你知道 `bf_lambda` / `bf_adapt_interval` / `auto_vad_hangover_frames`
      三者的**設定值 ≠ 生效值**
- [ ] init 之後有讀 `audio_pipeline_4ch_bf_effective_adapt_interval()` 與
      `audio_pipeline_4ch_bf_lambda()`，並確認是你預期的值
- [ ] 換 grid（改 `sample_rate` 或 `fft_size`）之後，有重新確認上面兩個生效值
- [ ] 若使用內建 VAD，`auto_vad_hangover_frames` 的生效幀數是自己用公式算過的
      （**沒有 read-back accessor**）

初始化：

- [ ] 每次 init 之前都重新呼叫 `audio_pipeline_4ch_get_mem_requirements()`
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）

執行期：

- [ ] 有自己 VAD 的話用的是 `process_with_activity()`，不是 `process()`
- [ ] `vad_raw` / `vad_out` 只傳 `0` 或 `1`
- [ ] `frequency_mask` 若非 NULL，長度至少 `n_freqs` 個 `int`
- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 有處理 `-3`（整個單元已被自動 reset，你的後級——RES、NR／合成的 reset 與 CNG rng、NN stage——與外部狀態也要跟著清）
- [ ] 任何呼叫被拒、`se_res_stage_gain()` 或 `se_anr_stage_gain()` 回 `NULL` 或輸出非有限時，單元與後級（含 `aec_res_reset()`）是**一起** reset 的
- [ ] 每一個後級 stage 都各自查大小（`aec_res_get_mem_size()`、`anr_get_mem_size()`、`ac_istft_get_mem_size()` 等），與單元 pool 分開配置；`AecRes` 的 `fft_size` 取自 `audio_pipeline_4ch_fft_size(p)`
- [ ] 用 `isnan()` 判斷 `audio_pipeline_4ch_bf_lambda()` 的失敗，不是拿它跟數值比較
- [ ] 同一個 instance 沒有被多 thread 同時處理

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

### 8.1 `AudioPipeline4ChMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。形狀與核心層 `FourAecNrResMemReq`
相同，但**版本號各自獨立**；本次 checkout，`layout_version=21`（`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`）。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `21` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（直接沿用核心層的值；目前唯一的 FFT 後端；永遠不會是 0） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，**已把核心層的 `build_flags_hash` 摺進去** |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求（含 core + SRP + BF + 本層 scratch；不含 RES／NR／合成等後級 stage） |

`build_flags_hash` 摺入核心層 hash 的意義：**核心層佈局一改版，
你所有已保存的複合 descriptor 就一起失效**，不會出現「用舊核心佈局算出來的 pool
剛好還塞得下」這種安靜的錯誤。

### 8.2 `init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `audio_pipeline_4ch_init()`。
`expected != NULL` 時，下列 8 項全部成立才會開始 carve；任一不符即回 `NULL`
（**不輸出任何訊息**）：

1. `descriptor_version` 相符
2. `layout_version` 相符
3. `backend_id` 相符
4. `build_flags_hash` 相符
5. `alignment` 相符
6. `reserved == 0`
7. `expected->bytes >= 目前 build 的需求`
8. `bytes`（實際交進來的 pool 大小）`>= 目前 build 的需求`

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的四條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，並確認 pool 預算仍足夠。
4. **升級之後也要重新確認 §4.6 的三個生效值**：改 grid、改 backend、
   或核心層改版都可能讓生效值變動，而設定值看起來完全沒變。
