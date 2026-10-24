# 四麥克風前端整合手冊（`libaudio_pipeline_4ch.a` / `4aec_nr_res.h`）

適用對象：**已經擁有自己的 beamformer（SRP-PHAT / GSC 或任何等效方案）**，
想要四路 linear AEC 前端，並自己串接後級頻域 stage（RES / NR / CNG / ISTFT，或自己的 NN）的函式庫使用者。

如果你**沒有**自己的 beamformer、想要含 direction finding 與 beamformer 的線性 AEC + BF
單元（`AudioPipeline4Ch`），請看 `docs/integration_4ch_spatial_zh_TW.md`。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。
頻域 stage 串接的總覽（任何 pipeline 都適用）見 `pipelines/README.md` 的 Call flow。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`；
> 從 `pipelines/4ch_aec_bf_nr_res/` 出發的相對路徑是 `../../../audiolib/common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.6）。

---

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
| `fft_size` | 0 = rate 預設；16 kHz 另可 256／512；48 kHz 只 1024 | init 前，不可變 | 格點（§4.2） |
| `filter_length` | `[0, 4096]`，0 = 依取樣率預設 | init 前，不可變 | 四路 AEC 記憶體最主要的旋鈕 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `MATCHED`（預設，bank 1..5）／`FIXED`／`EXTERNAL_ALIGNED` | init 前，不可變 | 四條 lane 共用的對齊方式（§4.1） |
| `capture_proxy_channel` | `[0, 3]`，預設 0 | init 前 | 共用 delay matcher 吃哪一支 mic |
| `max_delay_ms` | `[0, 4096]`，預設 1024 | init 前 | `MATCHED` 的 reference ring 容量；略大於真實最大延遲即可 |
| `enable_post` | 0／1，預設 1 | init 前 | 1 = 可呼叫 `four_aec_fuse()`；0 = 只有 front（神經後濾波直接吃 BF 頻譜） |
| 後級強度（你串的 stage） | `aec_res_set_preset(res, preset, ramp_ms)`、`anr_set_mode(nr, mode)` | hop 之間 | RES、NR 強度（§3.2）；前端本身沒有 preset 欄位 |

進階外部（預設關閉；打開前請閱讀標頭註解）：`delay_est_period_s`（預設 0.0 = 全速；非 0 可能改變選到的 lag）、
`delay_backward_quarantine_enabled`／`delay_backward_quarantine_s`（預設 0／1.0 s；只延後一個窗，不修正誤鎖，產品組態維持 0）。

**內部（存在、可調，但不建議開放）**

| 項目 | 位置 | 為什麼不開放 |
|---|---|---|
| 四條 lane 的 `AecConfig`：preset 固定 `AEC_PRESET_BALANCED`、`delay_mode = AEC_DELAY_EXTERNAL_ALIGNED`、`enable_res = 0`、`return_res_context = 1`、`spatial_linear_context`（`enable_post = 1` 為 `AEC_CONTEXT_NO_GAIN`，`0` 為 `AEC_CONTEXT_LINEAR_ONLY`） | `4aec_nr_res.c` 的 `derive_dims_and_configs()` | 四條 lane 的地板什麼都不塑形；對 lane 呼叫 `aec_set_preset()` 是無效操作（§3.2） |
| `delay_est_init_s`（0.3 s，固定）與 matched filter 的 `DA_*` 常數 | `aec.h`、`delay_aec.h` | 估計器聯合調過；duty-cycle 的啟動時間不是旋鈕 |
| 其餘 `AecConfig` 欄位家族（時間常數、步進、偵測門檻、延遲獲取策略、實驗開關） | `aec.h:59-156` | 理由與逐項清單見 [mono 手冊 §0.3](integration_mono_zh_TW.md) |
| 投影與 residual 向量的數值保護（overflow fallback、單一 sqrt 與 divide） | `4aec_projection_kernels.h` | 屬於 `|out|² == r2` 契約的實作細節 |
| 後級的 `AnrConfig` 家族與 `L`／`num_init_frames` | `anr_types.h` | 同 [mono 手冊 §0.3](integration_mono_zh_TW.md) |

---

## 1. 這個函式庫是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

`4aec_nr_res.h` 提供的是四路 linear AEC **前端**，每 hop 兩個呼叫，中間是你的 beamformer：

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► four_aec_front_process()   1 個共用 delay matcher → 1 條共用對齊後 reference → 4 條獨立 linear AEC
   │                               回傳：借用的 FourAecFront（4 路 linear 頻譜、4 路時域 hop、aligned_ref、delay 狀態）
   ▼
 【你的 beamformer】    讀那 4 路頻譜，算出這一幀的「有效權重」Complex[4][n_freqs]
   │
   ├─► four_aec_fuse()            以那組權重同調投影 near / 殘餘回音 / comfort noise
   │                               回傳：SeSpecFrame（e, y, x, r2, cn, far_hop, max_dt, …）
   ▼
 【你的後級 stage】     se_res_stage_gain → se_anr_stage_gain → sk_min_f32 → sk_capply_gain_f32
                        → aec_cng_fill → ac_istft_push → 1 個 hop 的 mono 輸出
```

這個模組**明確擁有**：1 個共用 delay matcher、4 個獨立 linear AEC adaptive filter、
beamformer 權重的同調投影。post-beam residual suppressor（`AecRes`）、mono MMSE-LSA
denoiser（`AnrDenoiser`）、comfort noise 與最終 iFFT/OLA（`AcIstft`）**不在**前端內：
每一個都是獨立的函式庫物件，有自己的 `mem_size` 與 `init`，由應用各自初始化並明寫每 hop 的串接
（完整範例見 §2 與 `4aec_nr_res_static.c` 的 `post_plan()`／`post_init()`／`post_hop()`、`tests/conv_chain_4ch.h`）。
線性 AEC + BF 單元 `audio_pipeline_4ch.c`（`AudioPipeline4Ch`）在這個前端之上只加 SRP-PHAT 與 BF；它和核心一樣沒有 RES，
`fuse()` 之後 `frame->g_res_ctx` 同樣是 NULL，RES、NR、comfort noise 與合成都由應用串接。
本文的核心前端**沒有**自己的 RES：`frame.g_res_ctx` 恆為 NULL，要 RES 就自己用 libaec 初始化一個 `AecRes`（見 §2）。

它**明確不做**：beamformer、DOA、VAD。它也不會挑某一支麥克風的 RES gain，
更不會每聲道各跑一份 NR/RES。

### 1.2 Header closure

```c
#include "4aec_nr_res.h"
```

一行就夠前端本身用。它自己會拉進 `aec.h` 與 `common/se_stage.h`（`SeSpecFrame`）；NR 的 header 不在其中。
要串後級 stage 時另外 include 你用到的函式庫 header：`aec_res.h`（`AecRes`、`aec_cng_fill()`）、`anr.h`、`audio_stft.h`（`AcIstft`）、
`samples/se_anr_stage.h`（`se_anr_stage_gain()`，範例碼）、`utility/simd_kernels.h`（`sk_min_f32()`、`sk_capply_gain_f32()`）。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audiolib>/common/include/fft_wrapper.h`，
透過 `4aec_nr_res.h` → `aec.h` → `fft_wrapper.h` 傳遞進來。
**你會直接用到它**：

* `FourAecFront.linear_spectra[ch]` 是 `const Complex*`
* `four_aec_fuse()` 的 `weights` 是 `const Complex*`

不需要另外 include `fft_wrapper.h`。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines/4ch_aec_bf_nr_res
-I pipelines
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
```

這五個 include roots 足以使用前端；下方串接範例另加 `-I <audiolib>/common`，
讓 `utility/mem_align.h` 與 `utility/simd_kernels.h` 找得到。`-I pipelines` 讓 `common/se_stage.h`、`samples/se_anr_stage.h` 找得到。
**注意：不需要 `audiolib/doa`、`audiolib/bf`、`audiolib/common/utility`**
—— 那三個是線性 AEC + BF 單元才需要的（見另一份手冊）。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o
libaudio_pipeline_4ch.a
libaec.a
libanr.a
libaudio_common.a
-lm
```

* `libaudio_pipeline_4ch.a` 含三個 object：`4aec_nr_res.o`、`audio_pipeline_4ch.o`（線性 AEC + BF 單元）、
  `se_config.o`（Makefile `PIPELINE_OBJS`；後者是 `pipelines/common/` 的共用組態）。只用本手冊這套 API
  時，你的程式不參照 `audio_pipeline_4ch.o` 裡的符號，linker 不會把它從 archive 帶出來，所以你**不需要**連
  `libdoa.a` / `libbf.a` / `libspatial_common.a`。
  （已實測：上面五項就能把只用 `4aec_nr_res.h` 與後級函式庫的程式連成執行檔。）
* archive 路徑是 config-keyed 的（`bin/<backend>-<hash>/`）。用各 Makefile 的
  `make -s print-lib-path` 取得，不要硬寫。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* 四個 archive 必須用同一組 backend / 編譯選項建置，否則會在
  `four_aec_nr_res_init_ex()` 的 descriptor 檢查被擋下。

---

## 2. Quick start

完整、可編譯、已實際跑過的最小整合（heap 路徑，權重先用固定值佔位；後級 stage 由應用自己初始化：RES、NR、
合成與增益暫存放在應用配置的一塊 16-byte 對齊記憶體，每個函式庫各自查大小、各自 init）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "4aec_nr_res.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "samples/se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

int main(void) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* p = four_aec_nr_res_create(&cfg);          /* heap path */
    if (!p) return 1;

    const int hop = four_aec_nr_res_hop_size(p);
    const int n = four_aec_nr_res_n_freqs(p);
    const int fft = four_aec_nr_res_fft_size(p);

    /* The application owns the post-beam stages: each library sizes and initialises itself.
     * RES (the one shared post-beam AecRes), NR, synthesis and the gain/spectrum buffers
     * share one 16-byte-aligned block here. */
    AecConfig res_cfg;
    aec_config_from_preset(&res_cfg, AEC_PRESET_BALANCED, 16000);
    res_cfg.fft_size = fft;
    AnrConfig nr_cfg = anr_config_for_mode_grid(16000, fft, ANR_MODE_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, fft);
    size_t res_bytes = aec_res_get_mem_size(&res_cfg);       /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t vec = ALIGN16((size_t)n * sizeof(float));
    size_t total = ALIGN16(res_bytes) + ALIGN16(nr_bytes) + ALIGN16(synth_bytes) +
                   3 * vec + ALIGN16((size_t)n * sizeof(Complex));
    void* mem;
    if (res_bytes == 0 || nr_bytes == 0 || synth_bytes == 0 || posix_memalign(&mem, 16, total) != 0)
        return 1;
    uint8_t* cur = (uint8_t*)mem;
    AecRes* res = aec_res_init(cur, res_bytes, &res_cfg);    cur += ALIGN16(res_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg); cur += ALIGN16(synth_bytes);
    if (!res || !nr || !istft) return 1;
    float* total_gain = (float*)cur;  cur += vec;
    float* extra_noise = (float*)cur; cur += vec;
    Complex* spec = (Complex*)cur;
    uint32_t rng = SE_CNG_RNG_SEED;

    float mics[4 * 1024], ref[1024], out[1024];
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * 1024];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                  /* replace with your capture */
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
                mics[i * FOUR_AEC_NR_RES_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        FourAecFront front;
        if (four_aec_front_process(p, mics, ref, &front) != FOUR_AEC_NR_RES_OK) break;

        /* Your beamformer reads front.linear_spectra[ch][bin] and writes this
         * frame's effective weights, channel-major: [ch * n + bin]. */
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ch++)
            for (int k = 0; k < n; k++) {
                weights[ch * n + k].r = 0.25f;
                weights[ch * n + k].i = 0.0f;
            }

        SeSpecFrame f;
        if (four_aec_fuse(p, weights, NULL, &f) != FOUR_AEC_NR_RES_OK) break;
        const float* g_res = se_res_stage_gain(res, &f);                 /* RES gain */
        const float* g_nr = g_res ? se_anr_stage_gain(nr, &f, extra_noise, 1) : NULL;
        if (g_nr) {
            sk_min_f32(total_gain, g_nr, g_res, n);
            sk_capply_gain_f32(spec, f.e, total_gain, n);
            if (f.cn) aec_cng_fill(&rng, spec, f.cn, g_res, g_nr, SE_CNG_NR_GAIN_FLOOR, n);
            (void)ac_istft_push(istft, spec, out);
        }
        if (!g_nr || !se_hop_is_finite(out, hop)) {
            four_aec_nr_res_reset(p);                            /* a refused call: reset every stage */
            aec_res_reset(res);
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = SE_CNG_RNG_SEED;
            continue;
        }
        /* consume out: hop floats of enhanced mono audio */
    }
    anr_destroy(nr);
    free(mem);
    four_aec_nr_res_destroy(p);
    printf("ok hop=%d n_freqs=%d\n", hop, n);
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
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

只要回音抑制、不要降噪的產品：不建 `AnrDenoiser`，NR 增益讀 unity 全 1 向量（`sk_min_f32` 涵蓋所有組合）。
不要 RES 時不建 `AecRes`，RES 增益讀 unity 向量，且不呼叫 `aec_cng_fill()`（沒有被 RES 挖掉的 bin）。
每個 stage 各自 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` 查大小、各自 init；
每 hop 的順序見 `4aec_nr_res_static.c` 的 `post_hop()`。

---

## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `four_aec_nr_res_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `-1`：`out` 為 NULL；`cfg` 未通過驗證（§4）；此 TU 編譯時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| 2 | allocate | 你自己的配置器，取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體 | 拿到 `req` 之後 | pool **不需要**預先清零（poison pattern 可接受，已實測）。`init_ex()` 自己會把整塊 memset 成 0 |
| 3 | init | `four_aec_nr_res_init_ex(pool, req.bytes, &cfg, &req)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；子模組 init 失敗。**沒有任何診斷輸出**（見 §5.5） |
| 3b | 後級 stage | 各 stage 自己的 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` → `aec_res_init()`／`anr_init()`／`ac_istft_init()` | 你的後級 stage 要用時 | `mem_size` 回 `0`、`init` 回 `NULL`：config 無效或 pool 太小／未對齊。stage 的記憶體由你自己查大小、自己 carve，與前端的 pool 分開（§2 的範例） |
| 4a | front_process | `four_aec_front_process(p, mics, ref, &front)` | 每收到一個 hop 呼叫一次 | `-1` / `-3`，見 §5.2 |
| 4b | 你的 beamformer | — | 在 front 與 fuse 之間 | 由你負責 |
| 4c | fuse | `four_aec_fuse(p, weights, beamformed_error, &frame)` | 緊接著同一幀的 `front_process()` | `-1` / `-2` / `-3`，見 §5.3 |
| 4d | 你的後級 stage | `se_res_stage_gain` … `ac_istft_push` | fuse 之後、下一次 `front_process()` 之前 | 由你負責 |
| 5 | reset | `four_aec_nr_res_reset(p)` | 換回聲路徑、切換串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op。**會作廢目前的 front 視圖與 `SeSpecFrame`**；你的 stage 要各自 reset（`aec_res_reset`、`anr_reset`、`ac_istft_reset`，並把 CNG 的 rng 設回 `SE_CNG_RNG_SEED`） |
| 6 | destroy | `four_aec_nr_res_destroy(p)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。NULL 安全。pool instance 冪等；`create()` instance 只能呼叫一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

heap 便利路徑：`four_aec_nr_res_create(&cfg)` 把第 1~3 步合成一步，
`four_aec_nr_res_destroy()` 會釋放它自己配置的那一塊。其餘完全相同。

### 3.1 執行期唯讀存取

| 呼叫 | 回傳 | NULL / 已 destroy 時 |
|---|---|---|
| `four_aec_nr_res_hop_size(p)` | 每次 front_process 要餵幾個 sample（per channel） | `-1` |
| `four_aec_nr_res_fft_size(p)` | FFT 長度 | `-1` |
| `four_aec_nr_res_n_freqs(p)` | 頻域 bin 數 | `-1` |
| `four_aec_nr_res_sample_rate(p)` | 實際生效的取樣率 | `-1` |
| `four_aec_nr_res_matched_filter_count(p)` | `MATCHED` 時為 `1`，`FIXED`／`EXTERNAL_ALIGNED` 時為 `0` | `0` |
| `four_aec_nr_res_duty_hops_run(p)` | 共用 estimator 實跑 matched-filter 分析的 hop 數（duty-cycle census 分子；分母用 `estimator_calls`，`duty_hops_total()` 是其別名） | `0` |
| `four_aec_nr_res_linear_aec_count(p)` | 恆為 `4` | `0` |
| `four_aec_nr_res_far_fft_real_compute_count(p)` | 累計計數（`long`），每次 `front_process()` **+1**（不是 +4）；`reset()` 會歸零 | `0` |

後面的計數是**結構稽核用**的：前兩個對有效 handle 回傳固定常數（`linear_aec_count` 恆 4），
用來在整合測試裡證明拓撲沒有被改掉；`far_fft_real_compute_count` 用來證明跨聲道的 far-end 共用真的生效。
它們對處理結果沒有任何影響。後級 stage 是否存在由你自己的串接決定，前端沒有 `nr`／`post_res` 的稽核計數。

### 3.1b 逐階段耗時（診斷用）

```c
typedef struct FourAecNrResLastTiming {
    uint32_t delay_us, frontend_us, linear_us, lane_res_us;   /* four_aec_front_process */
    uint32_t fuse_us;                                         /* four_aec_fuse          */
} FourAecNrResLastTiming;

void four_aec_nr_res_get_last_timing(const FourAecNrRes* p,
                                     FourAecNrResLastTiming* out);
```

最近一個 hop 前端每個階段的 wall-clock 成本（微秒，`CLOCK_MONOTONIC`）。純診斷：
pipeline 自己不回讀，也不影響處理。`p` 為 NULL 或已 destroy 時把 `out` 清零而
不是失敗；`out` 不得為 NULL。RES／NR／CNG／逆轉換由你的應用串接，耗時由你自己量。

本層自己量的是 `fuse_us` 與 `delay_us` 的共用延遲估計部分（不含 `align_render()` 的 ring-buffer 複製與
realign 掃描）。`frontend_us`、`linear_us`、`lane_res_us` 來自 `aec_get_last_timing()`
（`aec.h`）：四條 lane 各讀一次再相加，所以得到的是該階段在整個 lane 迴圈上的
wall-clock 成本。lane 內部的 `delay_us` 也一併加進上面的 `delay_us`——它是同一
個量（對齊遠端所花的時間），且**結構上恆為 0**，因為四條 lane 都是
`AEC_DELAY_EXTERNAL_ALIGNED`，沒有自己的 ring 也沒有估計器；相加而不是丟棄，是
為了讓日後改變 lane 模式時不會靜默漏掉那筆成本。

這三欄由**另一半旗標** `-DAEC_STAGE_TIMING=1` 管轄。只建本層那半，這三欄讀 0
而其餘照常量測，是可讀的狀態而非壞掉；`make PROFILE=1` 兩半一起開。

各階段**加起來不等於**整個呼叫的時間，呼叫端要自己用減法補餘額：front 側餘額涵蓋
輸入去交錯／有限值檢查、`align_render()` 與 realign 掃描。

前半在 `four_aec_front_process()` 接受引數**之後**清零，`fuse_us` 在 `four_aec_fuse()` 接受之後清零，所以中途 bail-out 的 hop 會回報 0 而不是
上一個 hop 的數字；被引數檢查直接拒絕的呼叫則整份記錄不動。

解析度是微秒，所以低於 1 µs 的階段會讀到 0。

**目標平台沒有 `CLOCK_MONOTONIC` 時**：`clock_gettime` 是 POSIX 而非 C99。用
`make PROFILE=1 EXTRA_CFLAGS='-DFOUR_AEC_NR_RES_NOW_US=board_timer_us
-DAEC_NOW_US=board_timer_us -include my_timer.h'` 換掉時鐘；巨集要是**純識別字**
（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），指向不收參數、回傳
`uint32_t` 微秒的函式。替代時鐘必須**單調**，否則無號減法會產生接近 32-bit 全距
的荒謬值。本層與 `lib/aec` 各有自己的覆寫點，刻意不共用——一個元件的時鐘屬於該
元件自己的建置契約。開著旗標但指向常數函式是合法用法，所有欄位讀 0。

### 3.2 執行期強度控制

強度面屬於擁有 RES 與 NR 的那一層。前端 `FourAecNrRes` **沒有** preset／mode setter；
你自己串後級時，直接對你 carve 的 stage 呼叫：

```c
aec_res_set_preset(res, AEC_PRESET_AGGRESSIVE, 100.0f);          /* 回 0 成功 */
anr_set_mode(nr, ANR_MODE_MILD);
float live, target;
aec_res_split_floor(res, &live, &target);                        /* 唯讀，線性功率 */
```

上式的 `res`、`nr` 就是你自己 `aec_res_init()`／`anr_init()` 回來的那兩個 handle
（`4aec_nr_res_static.c` 的 `Post` 結構）。線性 AEC + BF 單元同樣不持有 RES，強度也是對你自己建的 `AecRes`
呼叫 `aec_res_set_preset()`（見 `docs/integration_4ch_spatial_zh_TW.md`）。

> **本節最重要的一件事：對四條 lane 重新指定 preset 是無效操作。**
>
> 四條 AEC lane 都以 `spatial_linear_context` 建立，因此**沒有任何一條**會走到
> 自己的 suppression-gain 路徑；它們各自的地板什麼都不塑形。真正乘上輸出、
> 也真正決定 comfort noise 量的那個 gain，來自**唯一一個**共用的 post 級抑制器
> （`AecRes`；核心前端與單元都由你的應用擁有）。這也是為什麼 preset setter 要作用在它上面，而不是要你自己
> 對四條 lane 迴圈呼叫 `aec_set_preset()`。這一點由測試釘住
> （`pipelines/4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` 的 `test_caller_res_reset_and_preset`），不是推測。

| 呼叫 | 改／讀什麼 | 失敗（回 `-1`，**什麼都不寫**） |
|---|---|---|
| `aec_res_set_preset(res, preset, ramp_ms)` | **共用 post 級抑制器**的 far-active split floor | `res` 為 NULL；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_mode(nr, mode)` | 你擁有的**那一個**降噪器（NR 是對 beamform 後的訊號跑，不是每條 lane 一個） | `nr` 為 NULL；`mode` 超出 enum |
| `aec_res_split_floor(res, &live, &target)` | 唯讀。`live` = 抑制器**當下**套用的值，`target` = 撐得過 reset 的已設定值；兩者只有在 ramp 走到一半時才不同。任一指標可為 NULL | `res` 為 NULL |

`live` / `target` 這一對是**板端唯一值得記錄的強度量**：它是真正塑形輸出的東西
（lane 自己的地板什麼都不塑形），而兩個值合在一起就回答了「我要求的改動落地了沒有」。

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，所以互動式旋鈕應該給一個 ramp。ramp 進行中再呼叫
一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：四條 lane 的濾波器、共用延遲鎖定、NR 的噪聲底與增益
平滑歷史全部繼續跑。preset 的改動**撐得過**該 stage 的 reset（目標值另存一份，reset 會重新套用），但 ramp 進度會被丟掉。
**在兩個 hop 之間呼叫、與 `front_process()`／`fuse()` 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_mode_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_mode()`
> 直接換強度。若你在 preset 之上改過欄位，`anr_set_mode()` 會把它們蓋回 preset 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間強制套用 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 gain 還
決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB）——地板壓得越深、CNG
反而越多）。因此**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊或
degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

`FourAecNrResConfig` 的欄位如下表。取得預設值的唯一正確方式：

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`four_aec_nr_res_default_config()` **不做驗證**，只是填值。
驗證發生在 `get_mem_requirements()` / `init()` / `init_ex()` / `get_mem_breakdown()`
共用的同一個 reject-first 閘。

### 4.1 逐欄位

| 欄位 | 型別 | `default_config()` 給的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你傳給 `default_config()` 的值 | **只有 `16000` 或 `48000`**。`8000` 會被拒絕（這點和單聲道 pipeline 不同） | 設成你音訊實際的取樣率 |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設（16 kHz → 256，48 kHz → 1024）。明確指定時：16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024` | 除非有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `[0, 4096]`。負數或 > 4096 拒絕 | `0` = 沿用 AEC 依取樣率算出的預設（16 kHz → 832 taps，48 kHz → 3072 taps）。回聲尾巴短、記憶體吃緊時可調小；這是四路 AEC 記憶體最主要的旋鈕 |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | 四條 lane 共用同一個決策。`MATCHED` 建共用 estimator 與對齊 ring；`FIXED` 只用固定延遲 ring；`EXTERNAL_ALIGNED` 由呼叫端給已對齊的 far，不建 matcher 也不建 ring（記憶體差異見 4.6a） |
| `delay_num_filters` | `int` | `5` | 僅 `MATCHED`：`[1, 5]`；其餘模式須為預設 | matched-filter bank 大小，每少一個省 5,728 B |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED`：原生取樣點數；其餘模式須為 `-1` | 決定共用 delay ring 大小 |
| `capture_proxy_channel` | `int` | `0` | `[0, 3]`。其他值拒絕 | 共用 delay matcher 要用哪一支麥克風當輸入。挑訊噪比最穩定、離喇叭關係最典型的那一支 |
| `max_delay_ms` | `float` | `1024.0f` | 必須是有限值，且落在 `[0.0, 4096.0]` | 共用 reference delay line 的容量上限（毫秒）。實際的 ring 大小 = `ceil(max_delay_ms * sample_rate / 1000) + 2 * hop + 1` 個 sample。設得比你系統真實最大 mic↔ref 延遲**略大**即可；設太大只是白白吃記憶體 |
| `delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1` | 僅 `MATCHED`。開啟後，比現行 accepted delay **更早**的共用估計會先被隔離（`capture_proxy_channel` 那條 lane 在目前對齊上仍明顯在消除回音時），到期才採用；往後跳與首次取得對齊不受影響。產品組態維持 `0`；機制說明見 `integration_4ch_spatial_zh_TW.md` 的 `core.delay_backward_quarantine_enabled` 列 |
| `delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 隔離窗長（秒），`init` 時換算成 hop 數（最小 1）；enable 為 0 時無作用但仍被驗證 |
| `delay_est_period_s` | `float` | `0.0` | 有限值，`[0.0, 3600.0]` | 共用 matched-filter 的 duty-cycle 週期：`0.0` = 每 hop 全速分析；正值時估計 solid 且不變 `delay_est_init_s`（0.3 s，固定）後，改為每 round(`delay_est_period_s`/hop)÷5 個 hop（最小 1）分析一次，省算力但可能鎖到不同的 lag |
| `enable_post` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = 前端 carve fuse 緩衝，可呼叫 `four_aec_fuse()`；`0` = front-only 前端（只有 `four_aec_front_process()`，`four_aec_fuse()` 回 `INVALID_ARGUMENT`），給從不 fuse 的應用（例如 ULCNet）。見 4.6b |

前端**沒有** RES、NR、comfort noise 或 preset 的設定欄位：`aec_preset`、`nr_mode`、`enable_res`、`enable_nr`、
`enable_cng`、`legacy_amin` 都不在 `FourAecNrResConfig` 裡（lane 一律以 `AEC_PRESET_BALANCED` 導出，且只算線性殘留）。
這些屬於你的後級串接（例如 `4aec_nr_res_static.c` 的 `--no-res`、`--nr-preset`、`--no-cng`、`--legacy-amin` 旗標對應的選項）：
RES 用 `aec_config_from_preset(&res_cfg, preset, sample_rate)` 配上核心解析出的 `fft_size` 建 `AecRes`，NR 用 `anr_config_for_mode_grid()`。

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`。合法組合只有三組：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---|
| 16000 | 256 | 128 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop / n_freqs 寫死。** 用 `four_aec_nr_res_hop_size()` / `four_aec_nr_res_n_freqs()` 查。


### 4.3 Buffer 形狀（每次 `front_process` / `fuse`）

| Buffer | 誰擁有 | 形狀 | 說明 |
|---|---|---|---|
| `microphones_interleaved` | 你 | `float[hop * 4]`，`[sample * 4 + channel]` | 唯讀，只在該次呼叫期間被讀取。所有元素必須是有限值 |
| `ref` | 你 | `float[hop]` | 唯讀。所有元素必須是有限值 |
| `weights` | 你 | `Complex[4 * n_freqs]`，**channel-major**：`[channel * n_freqs + bin]` | 唯讀 |
| `beamformed_error` | 你（可為 NULL） | `Complex[n_freqs]` | 與 `weights` 同一次 beamformer 呼叫產生的 mono 誤差頻譜；NULL 則由權重重建。供應時 `frame.e` 別名這份陣列，它必須活過 frame |
| `out` | 你 | `float[hop]` | 由你的 `ac_istft_push()` 寫入 |
| `front.formed_hop[ch]` | 函式庫 | `const float[hop]` | 見 §4.4 |
| `front.linear_spectra[ch]` | 函式庫 | `const Complex[n_freqs]` | 見 §4.4 |

**權重約定**：`output[bin] = Σ_channel weights[channel][bin] * input[channel][bin]`，
**不做共軛**。多幀 / 時域型 GSC 請把它「這一幀的有效頻率響應」填進來。

**一條在每個 bin 權重都是 0 的 lane 不會投票。** `four_aec_fuse()` 除了投影頻譜，還會把四條
lane 的 `filter_converged` / `dt_indicator` / `saturation_level` 縮成三個純量（`frame.all_converged`／`max_dt`／`max_sat`）
去操縱那個共用的 post 級抑制器（其中 `dt_indicator` 決定它套哪一個地板）。一條在每個 bin 權重都是 0 的 lane 對
融合後的頻譜貢獻恰好為零，所以也不參與這三個縮減——縮減從中性值起算，lane 0 一樣受檢。整組
全零權重會先被 `-1` 擋掉（見 §5.3），所以至少有一條 lane 會通過這個檢查；真正在 beamform 的
權重每條 lane 都有非零 bin，這條規則不會跳過任何東西。

外部 beamformer**不需要**再回傳一份 mono hop —— 合成是在你自己的 `ac_istft_push()` 內
以那組權重加權後的頻譜完成的。

### 4.4 `FourAecFront` 與 `SeSpecFrame` 各欄位

`FourAecFront`（`four_aec_front_process()` 填）：

| 欄位 | 型別 | 內容 |
|---|---|---|
| `delay` | `FourAecNrResDelayState` | 共用 matcher 這一幀的狀態，見下 |
| `hop_size` | `int` | 這一幀的 hop |
| `n_channels` | `int` | 恆為 `FOUR_AEC_NR_RES_CHANNELS`（4） |
| `n_freqs` | `int` | 這一幀的 bin 數 |
| `linear_spectra[ch]` | `const Complex*` | 四路 linear AEC 的**頻譜**，就是 `four_aec_fuse()` 稍後會用的同一份。讓外部 beamformer 直接沿用既有的分析轉換，不必重做一次 STFT。**這是別名**，直接指向各 lane 自己的 buffer |
| `formed_hop[ch]` | `const float*` | 各 lane 與 `linear_spectra[ch]` 同一次選擇的**時域** hop（`hop_size` 個 sample）。做自己的 sqrt-Hann 分析的時域 beamformer 要吃這個。**這也是借用的**，沒有 interleaved 拷貝 |
| `aligned_ref` | `const float*` | 這一 hop 實際餵給四條 lane 的對齊後 far（`hop_size` 個 sample）；給需要「與線性濾波器吃到同一份 far」的外部 NN 後級用 |

`SeSpecFrame`（`four_aec_fuse()` 填，定義在 `pipelines/common/se_stage.h`）：

| 欄位 | 內容 |
|---|---|
| `n_freqs`, `hop` | 這一幀的 bin 數與 hop |
| `e` | beamformed 誤差頻譜（pre-RES；RES 與 NR 據以估計，也是增益套用的對象） |
| `y` | 投影後的近端頻譜 |
| `x` | lane 0 的 far 頻譜（四條 lane 已驗證吃同一份） |
| `r2` | 投影後的殘餘回音 PSD（audio scale，餵 NR 前由 stage 除以 `32768²`） |
| `cn` | 投影後的 comfort-noise PSD |
| `far_hop` | 對齊後的 far（同 `aligned_ref`） |
| `max_dt`, `max_sat`, `all_converged` | 對權重有非零響應的 lane 取的 double-talk／飽和／收斂縮減 |
| `clock_drift`, `delay_changed`, `delay_locked` | 共用 aligner 的旗標 |
| `g_res_ctx` | 恆為 NULL（核心的四條 lane 都不算自己的增益；單元的 `audio_pipeline_4ch_fuse()` 也一樣，RES 增益由你的 `AecRes` 經 `se_res_stage_gain()` 算出） |

`FourAecNrResDelayState` 欄位：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `delay_samples` | `int` | 目前採用的延遲（sample） |
| `confidence` | `float` | 估計信心值 |
| `solid` | `int` | 估計是否已穩定 |
| `changed` | `int` | 非 0 表示這一幀延遲值改變了 —— **你外部的 STFT / OLA 歷史必須跟著清掉**（前端自己對四條 lane 做 realign，不重置也不清合成 OLA） |
| `estimator_calls` | `uint64_t` | 累計呼叫次數 |
| `estimator_updates` | `int` | 累計更新次數 |

**front 視圖與 frame 的生命週期（重要）**：
`FourAecFront` 的所有指標與 `SeSpecFrame` 的所有指標，**只在下列任一事件發生之前有效**：
下一次 `four_aec_front_process()`、`four_aec_nr_res_reset()`、`four_aec_nr_res_destroy()`。
所有 stage 對它們都是**唯讀**。

這是 **API 約定，不是記憶體安全保證**。
`reset()` 不保證那些記憶體被清掉或覆寫 —— lane 的舊頻譜可能就原地留著。
也就是說：你違反這條規則時，程式**不會**壞給你看，只會安靜地讀到未定義內容。
不要用「讀到垃圾」當作偵測手段。

### 4.5 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `sample_rate` | 會（改 grid、改 AEC 預設 filter 長度） |
| `fft_size` | 會（改 grid） |
| `filter_length` | 會（四路 AEC 等比縮放） |
| `delay_mode` | 會（僅 `MATCHED` 建共用 estimator；三態差異見 4.6a） |
| `delay_num_filters` | 僅 `MATCHED` 有效，每少一個固定省 5,728 B |
| `fixed_delay_samples` | 僅 `FIXED` 有效，決定共用 delay ring 大小 |
| `max_delay_ms` | 會（僅 `MATCHED` 用於 sizing delay ring） |
| `enable_post` | 會（`0` 時不 carve fuse 緩衝，見 4.6b；ULCNet 範例用這個） |
| `capture_proxy_channel` | 不變 |

後級 stage 的記憶體（`AecRes`、`AnrDenoiser`、`AcIstft` 與暫存向量）不在前端的 `req.bytes` 裡：
由應用用各自的 `mem_size` 查、自己加總。
線性 AEC + BF 單元的 `audio_pipeline_4ch_get_mem_requirements()` 同樣不含 `AecRes`、NR 與 `AcIstft`。

### 4.6 實測記憶體（僅供量級參考，務必自己重查）

以下是本次 checkout（`layout_version=24`）、`BACKEND=ne10`、`SIMD=1`、
`delay_mode=MATCHED`（預設,n=5）、`enable_post=1`（預設）下直接呼叫 API
量到的前端值。換 backend、換編譯選項、更新 submodule 都會變。前端不含 RES／NR／iFFT，
所以 `FourAecNrResMemBreakdown` 只有 `aec_bytes`、`wrapper_bytes`（控制區塊、共用 aligner 與 fuse 緩衝）與 `total_bytes`。

| Config | `req.bytes` | `aec_bytes`（四路合計） | `wrapper_bytes` |
|---|---:|---:|---:|
| 16000，預設（256/128，`max_delay_ms=1024`） | 906,560 | 797,632 | 108,928 |
| 16000，`fft_size=512` | 1,422,464 | 1,306,368 | 116,096 |
| 16000，`max_delay_ms=100` | 847,424 | 797,632 | 49,792 |
| 16000，`filter_length=512` | 782,720 | 673,792 | 108,928 |
| 48000，預設（1024/512） | 3,174,944 | 2,912,064 | 262,880 |

`four_aec_nr_res_get_mem_breakdown()` 的 `total_bytes` 與 `get_mem_requirements()` 的
`req.bytes` 在上述每一組都相等；`wrapper_bytes` 已包含控制區塊。四路合計的
`aec_bytes` 除以 4 得單路 199,408 B（@16k/256）——與 `lib/aec` 的
`AEC_DELAY_EXTERNAL_ALIGNED` 單體大小完全相同,因為每路內部本來就是
`EXTERNAL_ALIGNED`（delay 由本層共用估計器提供,不建自己的 estimator/ring）。

#### 4.6a delay_mode 對 16000/預設 grid 的影響（`enable_post=1`，`MATCHED n=5` 為 baseline）

| `delay_mode` | `req.bytes` | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 906,560 | — |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 812,960 | −93,600 |
| `EXTERNAL_ALIGNED` | 806,048 | −100,512 |

省下的量比單聲道版本小，因為這裡只省**一份共用**的 estimator/ring（四路
共用一個 aligner），不是四份各自的——與本頁「單一共用 aligner」的結構
一致。

#### 4.6b `enable_post=0`（ULCNet 範例用的 front-only 前端）

| Config | `req.bytes` |
|---|---:|
| 16000，`fft_size=512`，`enable_post=0` | 1,412,128 |
| 48000，`enable_post=0` | 3,154,368 |

這是 [`pipeline_ulcnet_4ch.html`](html/pipeline_ulcnet_4ch.html) 的 ULCNet 4ch 範例所用單元
（`AudioPipeline4Ch`，`core.enable_post = 0`）內的前端大小；比同格點 `enable_post=1` 少的是 fuse 緩衝（16 kHz/512：10,336 B，48 kHz：20,576 B）。
**實際配置一律以 `req.bytes` 為準。**

---

## 5. 錯誤處理

### 5.1 front → fuse 的次序規則

前端沒有 token，只有一個「front frame 有效」旗標。次序規則很短：

| 情境 | 回傳 | 之後的狀態 |
|---|---|---|
| 正常 `front_process()` → `fuse()` | 兩者皆 `OK`(0) | front 視圖與 frame 有效至下一次 `front_process()`／`reset()`／`destroy()` |
| 連續兩次 `front_process()`（不 fuse） | 兩次皆 `OK` | 前一幀的視圖作廢，這是正常用法（從不 fuse 的 front-only 應用每 hop 都這樣） |
| 建構或 `reset()` 之後、尚未 `front_process()` 就 `fuse()` | `SEQUENCE_ERROR`(-2) | 不變 |
| `front_process()` 在接受引數後失敗（非有限輸入、`DSP_ERROR`），之後 `fuse()` | `SEQUENCE_ERROR`(-2) | 該次失敗清掉 front 有效旗標；沒有可 fuse 的幀 |
| `fuse()` 傳入全零 / 非有限的 weights | `INVALID_ARGUMENT`(-1) | **front 幀保留**，你可以修正 weights 後重試 |
| `enable_post = 0` 的前端呼叫 `fuse()` | `INVALID_ARGUMENT`(-1) | 不變（沒有 fuse 緩衝） |
| `front_process()` 或 `fuse()` 內部一致性檢查失敗 | `DSP_ERROR`(-3) | **instance 已被自動 `reset()`**，front 幀消失 |

**同一個 front frame 可以 fuse 多次**（例如以不同權重比較），每次都重新填 frame；但每次 fuse 都會覆寫上一次的 frame 內容。
想要 pipeline 化（beamformer 在另一顆核心上跑、排隊多幀）在這個 API 上是做不到的：視圖借用 lane 的 buffer，下一次 `front_process()` 就被覆寫。

### 5.2 `four_aec_front_process()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `ref` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | `microphones_interleaved`（`hop*4` 個）或 `ref`（`hop` 個）含非有限值 | `INVALID_ARGUMENT`(-1) | 逐階段耗時的前半已清零；front 有效旗標已清 |
| 3 | 內部 render 對齊失敗 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |
| 4 | 某條 lane 的輸出 context 不完整 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

`front_process()` **不會**檢查 buffer 長度 —— `hop*4` 與 `hop` 是你的責任。

### 5.3 `four_aec_fuse()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`weights` / `frame` 任一為 NULL，或 `cfg.enable_post == 0` | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 之前沒有有效的 front frame | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | `weights` 含非有限值，或 `Σ(|re| + |im|)` ≤ `1e-12`（等同全零） | `INVALID_ARGUMENT`(-1) | **front 幀保留**，可修正後重試 |
| 4 | 內部融合/一致性檢查失敗（含 `beamformed_error` 非有限、四條 lane 的 far 頻譜不一致） | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

**序列檢查排在 weights 檢查之前**：沒有 front frame + weights 也壞掉 → 你拿到的是 `-2`。

### 5.4 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `four_aec_nr_res_default_config()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `four_aec_nr_res_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_init()` | 非 NULL handle | `NULL` | 等同 `init_ex(..., NULL)` |
| `four_aec_nr_res_init_ex()` | 非 NULL handle | `NULL` | 8 項檢查任一不符即 NULL |
| `four_aec_nr_res_create()` | 非 NULL handle | `NULL` | config 無效或配置失敗 |
| `four_aec_front_process()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-3` | 見 §5.2 |
| `four_aec_fuse()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-2` / `-3` | 見 §5.3 |
| `four_aec_nr_res_reset()` | `void` | 不會失敗 | `p == NULL` 或已 destroy → 靜默 no-op |
| `four_aec_nr_res_destroy()` | `void` | 不會失敗 | 見 §5.6 |
| `*_hop_size` / `*_fft_size` / `*_n_freqs` / `*_sample_rate` | 正值 | `-1` | `p == NULL` 或已 destroy |
| `*_matched_filter_count` / `*_linear_aec_count` | `0`／`1` / `4` | `0` | `p == NULL` 或已 destroy |
| `four_aec_nr_res_far_fft_real_compute_count()` | 累計值（`long`） | `0` | `p == NULL` 或已 destroy |

> **注意慣例不一致**：本層的 `get_mem_requirements()` / `get_mem_breakdown()` 用
> **0 = 成功 / −1 = 失敗**；但下層模組的 `aec_get_mem_size()` /
> `anr_get_mem_size()` / `fft_get_mem_size()` 是**回傳 0 代表失敗**。
> 另外，形狀 accessor 用 `-1` 表示無效 handle，而結構稽核 accessor 用 `0`。

### 5.5 診斷輸出去向

**這個模組在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號。**
（`4aec_nr_res.c` 完全沒有 `printf` 家族的呼叫；已用 `nm` 確認 archive 內無相關符號。）

也就是說：`init_ex()` 回 `NULL` 時你**不會**知道是 8 項裡的哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，
把差異印出來。

`NO_STDIO=1` 對這個模組沒有任何影響（本來就沒有 stdio）。

### 5.6 `destroy()` 的兩種語意

| Instance 來源 | `destroy()` 行為 |
|---|---|
| `four_aec_nr_res_init()` / `init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、**冪等**，重複呼叫安全（已實測）。之後由你自己交還或重用 pool |
| `four_aec_nr_res_create()`（heap） | 釋放 `create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次** |

destroy 之後不要再呼叫任何 accessor。

---

## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | `libaudio_pipeline_4ch.a`（object 之一） | **本手冊的主體** |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | `libaudio_pipeline_4ch.a`（object 之二） | 線性 AEC + BF 單元（SRP-PHAT + BF；不含 RES／NR／ISTFT）。只用本手冊 API 時不會被拉進來 |
| `pipelines/common/se_config.c` | `libaudio_pipeline_4ch.a`（object 之三） | 共用組態；`se_stage.h` / `se_res_stage_gain()` 為 header-only |
| `pipelines/samples/se_anr_stage.h` | header-only 範例碼 | NR 級的 gain 呼叫（`se_anr_stage_gain()`）：沒有 ABI、沒有 layout version、不進任何 archive，可整份複製改寫 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` | — | 另一份手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_internal.h`、`4aec_projection_kernels.h` | — | 內部 header，**不是**公開 API，不要 include |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | caller-owned pool 路徑的主機參考程式 | 示範 query → allocate → init → front_process → fuse → 明寫的 stage 串接（`post_hop()`，它自己建一個 `AecRes`） → destroy → release。它在 front/fuse 之間塞的是**固定等權重**，那是決定性的 smoke adapter，**不是** production beamformer |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | 線性 AEC + BF 單元加 RES（libaec `AecRes`）／NR／合成的 pool 路徑主機參考程式 | 屬於另一份手冊；列在這裡是為了讓目錄內容一目了然 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | raw-float 錄音驗證 runner（單元加 RES／NR／合成） | 主機一次性工具，走 heap `create()` 路徑 |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、
`static_main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試；`tests/conv_chain_4ch.h` 是把後級串接寫在前端之上的參考鏈。
* `audiolib/{doa,bf,common/utility}` —— DOA / BF / spatial kernel 函式庫。
  **只用本手冊這套 API 的話完全不需要它們。**
* `.py` 檔 —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 四個 archive（`libaudio_pipeline_4ch.a` / `libaec.a` / `libanr.a` /
      `libaudio_common.a`）是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 §1.5；沒有多連 `libdoa.a` / `libbf.a` / `libspatial_common.a`
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `4aec_nr_res_internal.h` 或 `4aec_projection_kernels.h`

初始化：

- [ ] 每次 init 之前都重新呼叫 `four_aec_nr_res_get_mem_requirements()`
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）
- [ ] `sample_rate` 沒有設成 8000
- [ ] 後級 stage 的記憶體以各自的 `mem_size` 查大小，16-byte 對齊，與前端 pool 分開

front / fuse 協定：

- [ ] 每 hop 先 `front_process()` 再 `fuse()`；`fuse()` 之前沒有 front frame 會得到 `-2`
- [ ] `weights` 是 channel-major `[ch * n_freqs + bin]`，不是 bin-major
- [ ] `weights` 不會全零（會被 `-1` 拒絕；front 幀保留，可重試）
- [ ] 有處理 `-3`（instance 已被自動 reset，你的外部狀態與後級 stage 也要跟著 reset：`aec_res_reset()`／`anr_reset()`／`ac_istft_reset()` 與 CNG rng）
- [ ] `front.delay.changed` 非 0 時，你外部 beamformer 的 STFT / OLA 歷史有跟著清掉
- [ ] `FourAecFront` 與 `SeSpecFrame` 的指標只在下一次 `front_process()`／`reset()`／`destroy()` 之前讀取，沒有跨幀保存，也沒有寫入
- [ ] `reset()` 之後你的 beamformer 立刻停止讀那些指標

執行期：

- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值（NaN/Inf 會被 `-1` 拒絕）
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 同一個 instance 沒有被多 thread 同時處理
- [ ] 鏈的順序：`se_res_stage_gain` → `se_anr_stage_gain` → `sk_min_f32` → `sk_capply_gain_f32` → `aec_cng_fill` → `ac_istft_push`；缺席的增益來源用 unity 全 1 向量

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

### 8.1 `FourAecNrResMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。與 `AudioPipeline4ChMemReq` 同形狀、同語意，但**版本號各自獨立**（單元內嵌這個核心，其 `build_flags_hash` 折入核心的 hash）。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `24` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（目前唯一的 FFT 後端）；永遠不會是 0 |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，隨 build 變動 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求 |

### 8.2 `init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `four_aec_nr_res_init()`。
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

### 8.4 相容性規則（給整合者的三條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，並確認 pool 預算仍足夠。
