導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW.md) ｜ [下一部分](integration_4ch_core_zh_TW_02_lifecycle.md) <!-- split-nav -->
## 1. 這個 flow 層是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

`mcab.h` 提供的是四路 linear AEC **前端的 flow**，每 hop 兩個呼叫，中間是你的 beamformer：

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► mcab_process_front()      1 個共用 delay matcher → 1 條共用對齊後 reference → 4 條獨立 linear AEC
   │                               回傳：借用的 McabFront（4 路 linear 頻譜、4 路時域 hop、aligned_ref、delay 狀態）
   ▼
 【你的 beamformer】    讀那 4 路頻譜，算出這一幀的「有效權重」Complex[4][n_freqs]
   │
   ├─► mcab_fuse()               以那組權重同調投影 near / 殘餘回音 / comfort noise
   │                               回傳：McabFrame（e, y, x, r2, cn, far_hop, max_dt, …）
   ▼
 【你的後級 stage】     mcab_sample_res_input → aec_res_process → mcab_sample_anr_gain → sk_min_f32
                        → sk_capply_gain_f32 → aec_cng_fill → ac_istft_push → 1 個 hop 的 mono 輸出
```

mcab **明確擁有**的只有 flow 狀態：共用 delay 的 admission 狀態機（含 backward-jump quarantine 與 duty-cycle）、
reference ring 與對齊後的 far、每條 lane 的視圖、beamformer 權重的同調投影緩衝。**它借用、但不擁有**的是
你建立的 4 個獨立 linear AEC adaptive filter（`Aec`）與 MATCHED 模式的 1 個共用 delay matcher（`DelayAec`）：
它們由應用以 libaec 的 `aec_get_mem_size()`／`aec_init()`、`delay_aec_get_mem_size()`／`delay_aec_init()` 建立，
壽命必須涵蓋 `Mcab` handle；mcab 只會呼叫它們**自己的**處理與 reset 函式（`aec_process_context*()`、
`aec_apply_external_realign()`、`delay_aec_accumulate_ex()`、`aec_reset()`、`delay_aec_reset()`）與唯讀 getter。
post-beam residual suppressor（`AecRes`）、mono MMSE-LSA denoiser（`AnrDenoiser`）、comfort noise 與最終 iFFT/OLA（`AcIstft`）
**不在**模組內：每一個都是獨立的函式庫物件，有自己的 `mem_size` 與 `init`，由應用各自初始化並明寫每 hop 的串接
（完整範例見 §2 與 `mcab_external_static.c` 的 `post_plan()`／`post_init()`／`post_hop()`、`pipelines/4ch_aec_bf_nr_res/tests/conv_chain_mcab.h`；
範例用的銜接 helper 是 `mcab_sample_post.h`，見 §1.2）。
flow 之後 `frame->g_res_ctx` 恆為 NULL：要 RES 就自己用 libaec 初始化一個 `AecRes`（見 §2）。

如果還要 SRP-PHAT 與 BF，不必自己從頭寫：sample 層 helper `mcab_sample_beam.h` 把 audiolib 的 SRP-PHAT 與 BF 接在
`mcab_process_front()` 與 `mcab_fuse()` 之間（範例碼，不在模組 archive 內；手冊見 `docs/integration_4ch_spatial_zh_TW.md`）。
有自己 beamformer 的應用直接呼叫 `mcab_process_front()` 與 `mcab_fuse()` 這兩個函式即可。

它**明確不做**：beamformer、DOA、VAD。它也不會挑某一支麥克風的 RES gain，
更不會每聲道各跑一份 NR/RES。

### 1.2 Header closure

```c
#include "mcab.h"
```

一行就夠 flow 本身用。它自己會拉進 `aec.h`（`Aec`、`AecConfig`、`DelayAec`、`AecDelayMode`、`Complex`），並自己定義 `McabFrame`（`mcab_fuse()` 回傳的 frame）；NR 的 header 不在其中。
要串後級 stage 時另外 include 你用到的函式庫 header：`aec_res.h`（`AecRes`、`AecResInput`、`aec_res_process()`、`aec_cng_fill()`）、`anr.h`、`audio_stft.h`（`AcIstft`）、
`utility/simd_kernels.h`（`sk_min_f32()`、`sk_capply_gain_f32()`）、`utility/simd_kernel_nn.h`（`skn_all_finite_f32()`）。
`mcab_sample_post.h`（與 `mcab.h` 同目錄的 header-only 範例碼，沒有 ABI、不進 archive，可整份複製改寫）把 frame 銜接到這些函式庫：
`mcab_sample_res_input()` 填 `AecResInput`、`mcab_sample_anr_gain()` 呼叫降噪器（可選把 `r2` 折進噪聲底），以及常數
`MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`（−10 dB 的 comfort noise 下限）與 `MCAB_SAMPLE_CNG_RNG_SEED`。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audiolib>/common/include/ac_complex.h`，
透過 `mcab.h` → `aec.h` → `fft_wrapper.h` → `ac_complex.h` 傳遞進來。
**你會直接用到它**：

* `McabFront.linear_spectra[ch]` 是 `const Complex*`
* `mcab_fuse()` 的 `weights` 是 `const Complex*`

不需要另外 include `fft_wrapper.h` 或 `ac_complex.h`。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines/4ch_aec_bf_nr_res
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
```

這四個 include roots 足以使用 flow 與後級的 header（只 `#include "mcab.h"` 時用不到 `lib/nr`）；下方串接範例另加 `-I <audiolib>/common`，
讓 `utility/mem_align.h`、`utility/simd_kernels.h` 與 `utility/simd_kernel_nn.h` 找得到；`mcab_sample_post.h` 與 `mcab.h` 同目錄，由第一個 `-I` 涵蓋。
**注意：不需要 `<audiolib>/doa`、`<audiolib>/bf`、`<audiolib>/common/utility` 這三個 include root**
—— 那是 sample 層 beamformer helper 才需要的（見另一份手冊）。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o
libaudio_pipeline_mcab.a
libaec.a
libanr.a
libaudio_common.a
-lm
```

* `libaudio_pipeline_mcab.a` 只含一個 object：`mcab.o`。`mcab_sample_flow.o` 與 `mcab_sample_beam.o` 是 sample 層 helper，
  由同一個 Makefile 編出（`make libs`），**不在** archive 內。
  `libanr.a` 只在你串 NR 後級時才需要（上面的範例有串 NR，所以連了它）：`mcab.o` 只參照 lib/aec 與 audiolib common 的
  utility，不參照 NR、SRP／BF／spatial 符號，所以只要流程本身時你**不需要**連 `libanr.a` / `libdoa.a` / `libbf.a` /
  `libspatial_common.a`，`make libaudio_pipeline_mcab.a` 也只要求 common 與 lib/aec 兩個 producer（`NR_DIR` 與 spatial
  kernel 不在也能建）。
  （此連結集合就是本目錄 Makefile 的 `CORE_LIBS`，`mcab_external_static` 即以它連成執行檔：mcab、AEC、NR、common。）
* archive 路徑是 config-keyed 的（`bin/<backend>-<hash>/`）。用各 Makefile 的
  `make -s print-lib-path` 取得，不要硬寫。本目錄的 `print-lib-path` 印的是只含 common 與 lib/aec 的窄範圍
  archive；要 `make all` 建出、與各執行檔同一目錄的那一份，用 `make -s print-lib-path PRINT_FOR=all`。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* 四個 archive 必須用同一組 backend / 編譯選項建置。`mcab_init_ex()` 的 descriptor 只涵蓋 mcab 自己的 pool 版面與
  `backend_id`（[§8](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#8-版本與相容性)），不會替 `libaec.a`、`libanr.a`、`libaudio_common.a` 把關。

---

## 2. Quick start

完整、可編譯的最小整合（lane 與 estimator 由應用建立；權重先用固定值佔位；後級 stage 由應用自己初始化：RES、NR、
合成與增益暫存放在應用配置的一塊 16-byte 對齊記憶體，每個函式庫各自查大小、各自 init）。
mcab 自己的 pool 只放 flow 狀態，這裡用 `posix_memalign()` 配置；也可以改用靜態陣列（對齊 16 bytes）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mcab.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "mcab_sample_post.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

int main(void) {
    /* 1. The objects mcab drives are the application's: four lanes and (MATCHED) the shared
     *    delay estimator, each sized and initialised by its own libaec calls. */
    McabConfig cfg = mcab_default_config(16000);
    AecConfig lane_cfg;
    McabDelayEstimatorConfig dc;
    McabMemReq req;
    McabLibs libs;
    DelayAec delay;                                /* storage of the estimator object itself */
    void *lane_mem[MCAB_CHANNELS] = {0}, *delay_mem = NULL, *flow_mem = NULL;
    size_t lane_bytes;
    Mcab* p;
    int ch;

    memset(&libs, 0, sizeof(libs));
    if (mcab_lane_config(&cfg, &lane_cfg) != 0 ||            /* balanced level + contract */
        mcab_delay_estimator_config(&cfg, &dc) != 0 ||
        mcab_get_mem_requirements(&cfg, &req) != 0) return 1;
    lane_bytes = aec_get_mem_size(&lane_cfg);                /* 0 = invalid */
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        if (posix_memalign(&lane_mem[ch], 16, lane_bytes) != 0) return 1;
        libs.lanes[ch] = aec_init(lane_mem[ch], lane_bytes, &lane_cfg);
        if (!libs.lanes[ch]) return 1;
    }
    if (dc.required) {
        size_t bytes = delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters);
        if (bytes == 0 || posix_memalign(&delay_mem, 16, bytes) != 0 ||
            delay_aec_init(&delay, delay_mem, bytes, dc.sample_rate, dc.hop_size,
                           dc.num_filters) != 0) return 1;
        libs.delay = &delay;
    }
    /* 2. mcab's own pool holds flow state only. */
    if (posix_memalign(&flow_mem, 16, (size_t)req.bytes) != 0) return 1;
    p = mcab_init_ex(flow_mem, (size_t)req.bytes, &cfg, &libs, &req);
    if (!p) return 1;

    const int hop = mcab_hop_size(p);
    const int n = mcab_n_freqs(p);
    const int fft = mcab_fft_size(p);

    /* 3. The post-beam stages are the application's too: each library sizes and initialises
     *    itself; RES, NR, synthesis and the gain/spectrum buffers share one aligned block. */
    AecConfig res_cfg;
    aec_config_from_level(&res_cfg, AEC_LEVEL_BALANCED, 16000);
    res_cfg.fft_size = fft;
    AnrConfig nr_cfg = anr_config_for_level_grid(16000, fft, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, fft);
    size_t res_bytes = aec_res_get_mem_size(&res_cfg);       /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t vec = ALIGN16((size_t)n * sizeof(float));
    size_t total = ALIGN16(res_bytes) + ALIGN16(nr_bytes) + ALIGN16(synth_bytes) +
                   2 * vec + ALIGN16((size_t)n * sizeof(Complex));
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
    uint32_t rng = MCAB_SAMPLE_CNG_RNG_SEED;

    float mics[4 * 512], ref[512], out[512];             /* 512 = the largest hop (48 kHz) */
    Complex weights[MCAB_CHANNELS * 513];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                      /* replace with your capture */
            for (ch = 0; ch < MCAB_CHANNELS; ch++)
                mics[i * MCAB_CHANNELS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        McabFront front;
        if (mcab_process_front(p, mics, ref, &front) != MCAB_OK) break;

        /* Your beamformer reads front.linear_spectra[ch][bin] and writes this
         * frame's effective weights, channel-major: [ch * n + bin]. */
        for (ch = 0; ch < MCAB_CHANNELS; ch++)
            for (int k = 0; k < n; k++) {
                weights[ch * n + k].r = 0.25f;
                weights[ch * n + k].i = 0.0f;
            }

        McabFrame f;
        if (mcab_fuse(p, weights, NULL, &f) != MCAB_OK) break;
        AecResInput rin;
        mcab_sample_res_input(&f, &rin);
        const float* g_res = aec_res_process(res, &rin);                 /* RES gain */
        const float* g_nr = g_res ? mcab_sample_anr_gain(nr, &f, extra_noise, 1) : NULL;
        if (g_nr) {
            sk_min_f32(total_gain, g_nr, g_res, n);
            sk_capply_gain_f32(spec, f.b, total_gain, n);   /* 估計用 f.e，合成用 f.b */
            if (f.cn) aec_cng_fill(&rng, spec, f.cn, g_res, g_nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, n);
            (void)ac_istft_push(istft, spec, out);
        }
        if (!g_nr || !skn_all_finite_f32(out, (size_t)hop)) {
            mcab_reset(p);                       /* resets the four lanes and the estimator too */
            aec_res_reset(res);                  /* a refused call: reset every post stage */
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = MCAB_SAMPLE_CNG_RNG_SEED;
            continue;
        }
        /* consume out: hop floats of enhanced mono audio */
    }
    /* 4. Teardown in the reverse order: the flow first (it borrows the lanes), then the lanes. */
    mcab_destroy(p);
    for (ch = 0; ch < MCAB_CHANNELS; ch++) { aec_destroy(libs.lanes[ch]); free(lane_mem[ch]); }
    free(delay_mem);
    free(flow_mem);
    anr_destroy(nr);
    free(mem);
    printf("ok hop=%d n_freqs=%d\n", hop, n);
    return 0;
}
```

編譯：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines/4ch_aec_bf_nr_res \
  -I lib/aec/c_impl/include \
  -I lib/nr/c_impl/include \
  -I <audiolib>/common/include \
  -I <audiolib>/common
c++ -o app app.o \
  <4ch-bin>/libaudio_pipeline_mcab.a \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

四條 lane 的參數（level、`filter_length`…）是應用的決定：範例用 `mcab_lane_config()`（balanced level 加契約）；
要自己的 lane 就改成 `aec_config_from_level(&lane_cfg, level, 16000)`、設好 `filter_length` 等欄位，
再 `mcab_apply_lane_contract(&cfg, &lane_cfg)` 把契約欄位蓋上去（細節見 [§4.1](integration_4ch_core_zh_TW_03_config.md#41-逐欄位)）。

只要回音抑制、不要降噪的產品：不建 `AnrDenoiser`，NR 增益讀 unity 全 1 向量（`sk_min_f32` 涵蓋所有組合）。
不要 RES 時不建 `AecRes`，RES 增益讀 unity 向量，且不呼叫 `aec_cng_fill()`（沒有被 RES 挖掉的 bin）。
每個 stage 各自 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` 查大小、各自 init；
每 hop 的順序見 `mcab_external_static.c` 的 `post_hop()`。
同一份 init 順序也寫成可重用的 sample 函式 `mcab_sample_flow_get_mem_size()`／`_init()`（`mcab_sample_flow.h`，一塊 pool 放四條 lane、estimator 與 mcab 的 pool），
給測試與範例用；產品端照上面的步驟自己做即可。

---

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW.md) ｜ [下一部分](integration_4ch_core_zh_TW_02_lifecycle.md) <!-- split-nav -->
