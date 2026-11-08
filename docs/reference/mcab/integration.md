# C 呼叫與生命週期

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
`mcab_process_front()` 與 `mcab_fuse()` 之間（範例碼，不在模組 archive 內；手冊見 `docs/reference/mcab/beam.md`）。
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
-I ../AEC/c_impl/include
-I ../NR/c_impl/include
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
  `backend_id`（[§8](errors.md#8-版本與相容性)），不會替 `libaec.a`、`libanr.a`、`libaudio_common.a` 把關。

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


將範例存成 Audio_ALG 根目錄的 `app.c`。以下使用預設的 sibling 依賴。

```sh
make -C pipelines/4ch_aec_bf_nr_res libs
make -C pipelines/4ch_aec_bf_nr_res libaudio_pipeline_mcab.a
common_dir="$(cd ../audiolib/common && pwd)"
common="$(make -s -C "$common_dir" print-lib-path)"
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I ../AEC/c_impl/include -I ../NR/c_impl/include \
  -I ../audiolib/common/include -I ../audiolib/common \
  -I pipelines/4ch_aec_bf_nr_res
c++ -o app app.o \
  "$(make -s -C pipelines/4ch_aec_bf_nr_res print-lib-path)" \
  "$(make -s -C ../AEC/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$(make -s -C ../NR/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$common" -lm
./app
```

四條 lane 的參數（level、`filter_length`…）是應用的決定：範例用 `mcab_lane_config()`（balanced level 加契約）；
要自己的 lane 就改成 `aec_config_from_level(&lane_cfg, level, 16000)`、設好 `filter_length` 等欄位，
再 `mcab_apply_lane_contract(&cfg, &lane_cfg)` 把契約欄位蓋上去（細節見 [§4.1](config.md#41-逐欄位)）。

只要回音抑制、不要降噪的產品：不建 `AnrDenoiser`，NR 增益讀 unity 全 1 向量（`sk_min_f32` 涵蓋所有組合）。
不要 RES 時不建 `AecRes`，RES 增益讀 unity 向量，且不呼叫 `aec_cng_fill()`（沒有被 RES 挖掉的 bin）。
每個 stage 各自 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` 查大小、各自 init；
每 hop 的順序見 `mcab_external_static.c` 的 `post_hop()`。
同一份 init 順序也寫成可重用的 sample 函式 `mcab_sample_flow_get_mem_size()`／`_init()`（`mcab_sample_flow.h`，一塊 pool 放四條 lane、estimator 與 mcab 的 pool），
給測試與範例用；產品端照上面的步驟自己做即可。

## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | config | `mcab_default_config(sr)`，再 `mcab_lane_config(&cfg, &lane)`（或 `aec_config_from_level()` ＋ `mcab_apply_lane_contract(&cfg, &lane)`）與 `mcab_delay_estimator_config(&cfg, &dc)` | 配置任何記憶體之前 | `mcab_default_config()` 不驗證。其餘回 `-1`：`cfg` 未通過驗證（[§4](config.md#4-config-完整參考)）；`mcab_apply_lane_contract()` 另外在 lane 是為別的取樣率建的時候回 `-1`，lane 不變 |
| 2 | query | `aec_get_mem_size(&lane)`、`delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters)`（只在 `dc.required`）、`mcab_get_mem_requirements(&cfg, &req)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | `aec_get_mem_size()`／`delay_aec_get_mem_size()` 回 `0` = 無效。`mcab_get_mem_requirements()` 回 `-1`：`out` 為 NULL；`cfg` 未通過驗證；編譯 `mcab.c` 時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| 3 | allocate | 你自己的配置器：四塊 lane 記憶體、estimator 的 pool 與 `DelayAec` 結構本身、`req.bytes` 且對齊 `req.alignment`（16）的 mcab pool | 拿到各個大小之後 | mcab 的 pool **不需要**預先清零（poison pattern 可接受，已實測）。`mcab_init_ex()` 驗證全部通過後自己把整塊 memset 成 0；被拒時 pool 一個 byte 都不動 |
| 4 | init lib 物件 | `aec_init(mem, bytes, &lane)` ×4、`delay_aec_init(&delay, mem, bytes, sr, hop, nf)`（只在 MATCHED） | 配置之後、`mcab_init_ex()` 之前 | `aec_init()` 回 `NULL`、`delay_aec_init()` 回非 0：config 無效或 pool 太小／未對齊。四條 lane 必須是**四個不同的物件**，且必須是剛建好的，或已經 `aec_reset()`／`delay_aec_reset()`（或由先前那個 flow 的 `mcab_reset()`）重置過：mcab 看不出物件已經跑過，用過的 estimator 會把舊的 delay 交給新的 flow |
| 5 | init flow | `mcab_init_ex(pool, req.bytes, &cfg, &libs, &req)` | 所有 lib 物件都就緒之後，處理第一個 hop 之前 | 回傳 `NULL`：8 項 descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效；lane／estimator 檢查不符（lane 的 hop 與 bin 數、不帶自己 gain 的 linear context、post 半邊與 `enable_post` 一致、吃已對齊的 far、四條互不相同；estimator 的 bank 大小、取樣率、48 kHz scratch；非 MATCHED 卻給了 estimator）。**沒有任何診斷輸出**（見 [§5.5](errors.md#55-診斷輸出去向)） |
| 5b | 後級 stage | 各 stage 自己的 `aec_res_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` → `aec_res_init()`／`anr_init()`／`ac_istft_init()` | 你的後級 stage 要用時 | `mem_size` 回 `0`、`init` 回 `NULL`：config 無效或 pool 太小／未對齊。stage 的記憶體由你自己查大小、自己 carve，與 mcab 的 pool 分開（[§2](#2-quick-start) 的範例） |
| 6a | front | `mcab_process_front(p, mics, ref, &front)` | 每收到一個 hop 呼叫一次 | `-1` / `-3`，見 [§5.2](errors.md#52-mcab_process_front-的完整拒絕清單依實際檢查順序) |
| 6b | 你的 beamformer | — | 在 front 與 fuse 之間 | 由你負責 |
| 6c | fuse | `mcab_fuse(p, weights, beamformed_error, &frame)` | 緊接著同一幀的 `mcab_process_front()` | `-1` / `-2` / `-3`，見 [§5.3](errors.md#53-mcab_fuse-的完整拒絕清單依實際檢查順序) |
| 6d | 你的後級 stage | `mcab_sample_res_input`／`aec_res_process` … `ac_istft_push` | fuse 之後、下一次 `mcab_process_front()` 之前 | 由你負責 |
| 7 | reset | `mcab_reset(p)` | 換回聲路徑、切換串流時 | 不會失敗。`p == NULL` 或已 destroy 時是 no-op。它會透過各物件**自己的** reset 函式重置四條 lane 與 estimator（不需要、也不應該再自己對它們呼叫 `aec_reset()`），flow 的 admission、ring、計數也歸零。**會作廢目前的 front 視圖與 `McabFrame`**，且之後 `mcab_fuse()` 回 `MCAB_SEQUENCE_ERROR` 直到下一個 front；你的 stage 要各自 reset（`aec_res_reset`、`anr_reset`、`ac_istft_reset`，並把 CNG 的 rng 設回 `MCAB_SAMPLE_CNG_RNG_SEED`） |
| 8 | destroy | `mcab_destroy(p)`，之後 `aec_destroy(lane)` ×4 | 不再處理任何 hop 之後、釋放記憶體之前 | 不會失敗。NULL 安全。mcab 的 destroy **不碰任何 lib 物件**：lane 的拆除是你的事，順序是 flow 先、lane 後。pool instance 冪等；`mcab_create()` instance 只能呼叫一次 |
| 9 | release | 你自己的釋放器 | **一定要在第 8 步之後** | 順序顛倒 = use-after-free |

`mcab_create(&cfg, &libs)` 是 heap 便利路徑，**只**把第 5 步 mcab 自己的那一塊 pool 換成 heap 配置；lane 與 estimator 仍然是你的，
`mcab_destroy()` 只釋放它自己配置的那一塊。其餘完全相同。

**flow 掛著的期間，不要自己對這些 lane 或 estimator 呼叫處理或 reset 函式**：它們的狀態會和 flow 的 admission、ring 脫節，
而 mcab 沒有辦法偵測。要重置請呼叫 `mcab_reset()`。

### 3.1 執行期唯讀存取

| 呼叫 | 回傳 | NULL / 已 destroy 時 |
|---|---|---|
| `mcab_hop_size(p)` | 每次 `mcab_process_front()` 要餵幾個 sample（per channel） | `-1` |
| `mcab_fft_size(p)` | FFT 長度 | `-1` |
| `mcab_n_freqs(p)` | 頻域 bin 數 | `-1` |
| `mcab_sample_rate(p)` | 實際生效的取樣率 | `-1` |
| `mcab_enable_post(p)` | `cfg.enable_post`（`1`／`0`） | `-1` |
| `mcab_matched_filter_count(p)` | `MATCHED` 時為 `1`，`FIXED`／`EXTERNAL_ALIGNED` 時為 `0` | `0` |
| `mcab_duty_hops_run(p)` | 共用 estimator 實跑 matched-filter 分析的 hop 數（duty-cycle census 分子；分母用 `estimator_calls`，`mcab_duty_hops_total()` 是其別名） | `0` |
| `mcab_linear_aec_count(p)` | 恆為 `4` | `0` |
| `mcab_far_fft_real_compute_count(p)` | 四條 lane 累計的 far-FFT 次數（`long`），每次 `mcab_process_front()` **+1**（不是 +4；混合 warm/soft realign 的那個 hop 是 +4）；`mcab_reset()` 會歸零 | `0` |
| `mcab_realign_warm_lane_count(p)`、`mcab_realign_soft_lane_count(p)` | 累計的 lane realign 結果（`aec_apply_external_realign()` 的 warm／soft 兩種）；`delay.changed` 的 hop 兩者合計 +4（有 lane 拒絕時較少） | `0` |
| `mcab_pending_delay_candidate(p)` | 等待第二次確認的 delay 候選值（samples） | `-1`（也用於「沒有候選」） |

此表「NULL／已 destroy」欄的行為只適用於 caller-owned pool 的 instance（destroy 後 handle 仍在你的 pool 裡，旗標讓 accessor 回報無效）；
`mcab_create()` 的 instance 在 `mcab_destroy()` 時已被釋放，之後不可再呼叫任何 accessor。

後面的計數是**結構稽核用**的：`mcab_linear_aec_count()` 對有效 handle 回傳固定常數 4，
用來在整合測試裡證明拓撲沒有被改掉；`far_fft_real_compute_count` 用來證明跨聲道的 far-end 共用真的生效。
它們對處理結果沒有任何影響。後級 stage 是否存在由你自己的串接決定，flow 沒有 `nr`／`post_res` 的稽核計數。

### 3.1b 逐階段耗時（診斷用）

```c
typedef struct McabLastTiming {
    uint32_t delay_us, frontend_us, linear_us, lane_res_us;   /* mcab_process_front */
    uint32_t fuse_us;                                         /* mcab_fuse          */
} McabLastTiming;

void mcab_get_last_timing(const Mcab* p, McabLastTiming* out);
```

最近一個 hop flow 每個階段的 wall-clock 成本（微秒，`CLOCK_MONOTONIC`）。純診斷：
flow 自己不回讀，也不影響處理。`p` 為 NULL 或已 destroy 時把 `out` 清零而
不是失敗；`out` 不得為 NULL。RES／NR／CNG／逆轉換由你的應用串接，耗時由你自己量。

本層自己量的是 `fuse_us` 與 `delay_us` 的共用延遲估計部分（不含 `align_render()` 的 ring-buffer 複製與
realign 掃描）。`frontend_us`、`linear_us`、`lane_res_us` 來自 `aec_get_last_timing()`
（`aec.h`）：四條 lane 各讀一次再相加，所以得到的是該階段在整個 lane 迴圈上的
wall-clock 成本。lane 內部的 `delay_us` 也一併加進上面的 `delay_us`——它是同一
個量（對齊遠端所花的時間），且**結構上恆為 0**，因為四條 lane 都是
`AEC_DELAY_EXTERNAL_ALIGNED`，沒有自己的 ring 也沒有估計器；相加而不是丟棄，是
為了讓日後改變 lane 模式時不會靜默漏掉那筆成本。

這三欄由**另一半旗標** `-DAEC_STAGE_TIMING=1` 管轄；`delay_us` 的 flow 部分與 `fuse_us` 由 `-DMCAB_STAGE_TIMING=1` 管轄。
只開本層那半，lane 的三欄讀 0
而其餘照常量測，是可讀的狀態而非壞掉；`make PROFILE=1` 兩半一起開。兩個旗標都沒開時所有欄位恆為 0（release build 不讀時鐘）。

各階段**加起來不等於**整個呼叫的時間，呼叫端要自己用減法補餘額：front 側餘額涵蓋
輸入去交錯／有限值檢查、`align_render()` 與 realign 掃描。

前半四欄在 `mcab_process_front()` 接受引數（非 NULL）**之後**清零，`fuse_us` 在 `mcab_fuse()` 通過引數、sequence 與 weights 檢查之後清零，
所以 front 中途 bail-out 的 hop（含非有限輸入被拒的那一個），前四欄回報 0（或只含失敗前已量到的階段）而不是上一個 hop 的數字；
`fuse_us` 則只在 `mcab_fuse()` 清零，front 失敗不會動它（仍是上一次 fuse 的值）。被 NULL 引數直接拒絕的呼叫整份記錄不動。

解析度是微秒，所以低於 1 µs 的階段會讀到 0。

**目標平台沒有 `CLOCK_MONOTONIC` 時**：`clock_gettime` 是 POSIX 而非 C99。用
`make PROFILE=1 EXTRA_CFLAGS='-DMCAB_NOW_US=board_timer_us -DAEC_NOW_US=board_timer_us -include my_timer.h'`
換掉時鐘；巨集要寫成**純函式名**（程式碼以 `MCAB_NOW_US()` 呼叫它），指向不收參數、回傳
`uint32_t` 微秒的函式。替代時鐘必須**單調**，否則無號減法會產生接近 32-bit 全距
的荒謬值。本層與 `lib/aec` 各有自己的覆寫點，刻意不共用——一個元件的時鐘屬於該
元件自己的建置契約。開著旗標但指向常數函式是合法用法，所有欄位讀 0。

### 3.2 執行期強度控制

強度面屬於擁有 RES 與 NR 的那一層。flow `Mcab` **沒有** level setter；
你自己串後級時，直接對你 carve 的 stage 呼叫：

```c
aec_res_set_level(res, AEC_LEVEL_AGGRESSIVE, 100.0f);          /* 回 0 成功 */
anr_set_level(nr, ANR_LEVEL_MILD);
float live, target;
aec_res_split_floor(res, &live, &target);                        /* 唯讀，線性功率 */
```

上式的 `res`、`nr` 就是你自己 `aec_res_init()`／`anr_init()` 回來的那兩個 handle
（`mcab_external_static.c` 的 `Post` 結構）。sample 層 beamformer helper 同樣不持有 RES，強度也是對你自己建的 `AecRes`
呼叫 `aec_res_set_level()`（見 `docs/reference/mcab/beam.md`）。

> **本節最重要的一件事：對四條 lane 重新指定 level 是無效操作。**
>
> 四條 AEC lane 都以 `spatial_linear_context` 建立（契約欄位，`mcab_apply_lane_contract()` 蓋上），因此**沒有任何一條**會走到
> 自己的 suppression-gain 路徑；它們各自的地板什麼都不塑形。真正乘上輸出、
> 也真正決定 comfort noise 量的那個 gain，來自**唯一一個**共用的 post 級抑制器
> （`AecRes`；由你的應用擁有）。這也是為什麼 level setter 要作用在它上面，而不是要你自己
> 對四條 lane 迴圈呼叫 `aec_set_level()`。共用 post 級 `AecRes` 的 reset 與 level 行為由測試釘住
> （`pipelines/4ch_aec_bf_nr_res/tests/test_mcab_sample_beam.c` 的 `test_caller_res_reset_and_level`），不是推測。

| 呼叫 | 改／讀什麼 | 失敗（回 `-1`，**什麼都不寫**） |
|---|---|---|
| `aec_res_set_level(res, level, ramp_ms)` | **共用 post 級抑制器**的 far-active split floor | `res` 為 NULL；`level` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_level(nr, level)` | 你擁有的**那一個**降噪器（NR 是對 beamform 後的訊號跑，不是每條 lane 一個） | `nr` 為 NULL；`level` 超出 enum；該 level 的 config 被 `anr_reconfigure()` 拒絕 |
| `aec_res_split_floor(res, &live, &target)` | 唯讀。`live` = 抑制器**當下**套用的值，`target` = 撐得過 reset 的已設定值；兩者只有在 ramp 走到一半時才不同。任一指標可為 NULL | `res` 為 NULL |

`live` / `target` 這一對是**板端唯一值得記錄的強度量**：它是真正塑形輸出的東西
（lane 自己的地板什麼都不塑形），而兩個值合在一起就回答了「我要求的改動落地了沒有」。

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名與整數相同。舊名 `AecPreset`／`AEC_PRESET_*`、`AnrMode`／`ANR_MODE_*` 仍保留在公開 header（同值），舊函式名已移除。

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 level 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、地板又是硬性 clamp，所以互動式旋鈕應該給一個 ramp。ramp 進行中再呼叫
一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：四條 lane 的濾波器、共用延遲鎖定、NR 的噪聲底與增益
平滑歷史全部繼續跑。level 的改動**撐得過**該 stage 的 reset（目標值另存一份，reset 會重新套用），但 ramp 進度會被丟掉。
**在兩個 hop 之間呼叫、與 `mcab_process_front()`／`mcab_fuse()` 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_level_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_level()`
> 直接換強度。若你在 level 組態之上改過欄位，`anr_set_level()` 會把它們蓋回 level 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間強制套用 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 gain 還
決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB）——地板壓得越深、CNG
反而越多）。因此**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量
echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。請在 echo 對齊或
degradation 對齊的條件下比較，並實際試聽。

[返回本主題](README.md)
