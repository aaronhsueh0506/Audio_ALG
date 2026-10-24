導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW.md) ｜ [下一部分](integration_4ch_core_zh_TW_02_lifecycle.md) <!-- split-nav -->
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

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW.md) ｜ [下一部分](integration_4ch_core_zh_TW_02_lifecycle.md) <!-- split-nav -->
