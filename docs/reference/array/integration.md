# C 呼叫與生命週期

## 1. 這個 array instance 是什麼、你連結什麼、你 `#include` 什麼

### 1.1 這是什麼

四路 linear AEC 前端是 `libaec.a` 的一個物件：`AecConfig.n_mics = AEC_ARRAY_MICS`（4）時，一次 `aec_init_ex()` 在同一塊 caller pool 裡建出
共用 delay estimator、對齊後的 far、四條 lane（內部一律 `EXTERNAL_ALIGNED`）、融合緩衝與唯一一個 residual 增益級。
介面是 `aec.h` 的「Array instance」一節；每 hop 兩個呼叫，中間是你的 beamformer：

```
mic hop [hop][4] + ref hop [hop]
   │
   ├─► aec_process_linear()      1 個共用 delay matcher → 1 條共用對齊後 reference → 4 條獨立 linear AEC（含各 lane 的殘餘階段）
   ├─► aec_get_array()           借用的 AecArrayView：4 路誤差頻譜 lane_error_spec、4 路 rendering base、4 路時域 hop、aligned_ref、delay 狀態
   ▼
 【你的 beamformer】    讀那 4 路頻譜，算出這一幀的「有效權重」Complex[4][n_freqs]（可附同一次呼叫的 beamformed 頻譜）
   │
   ├─► aec_process_residual()    以那組權重同調融合 near / 殘餘回音 / comfort noise；FULL 時再算唯一一個 G_res
   ├─► aec_get_residual()        AecResidualView（error_spec = E、base_spec = B、near_spec、r2、comfort_noise、旗標…）
   ▼
 【你的後級 stage】     anr_process_gain_with_residual_psd(E) → aec_apply_gain(G_nr) → aec_apply_cng(G_nr) → ac_istft_push → 1 個 hop 的 mono 輸出
```

陣列物件的內容（共用 delay 的 admission、對齊環、四條 lane、權重的同調融合、residual 增益級、comfort-noise 產生器）、
三種 `AecConfig.residual` 模式的行為與「它不做什麼」（beamformer、DOA、VAD）是函式庫的行為，以 AEC 的
[integration.md §8.8](../../../../AEC/docs/reference/integration.md#88-麥克風陣列實例n_mics--4) 與 `aec.h` 為準，這裡不重複。
Consumer 端的事：

* mono MMSE-LSA denoiser（`AnrDenoiser`）與最終 iFFT/OLA（`AcIstft`）**不在**物件內：各是獨立的函式庫物件，有自己的 `mem_size` 與 `init`，
  由應用各自初始化並明寫每 hop 的串接（完整範例見 §2 與 `4ch_external_static.c` 的 `post_plan()`／`post_init()`／`post_hop()`，
  以及 `pipelines/4ch/tests/conv_chain_4ch.h`；銜接 helper 是 `array_sample_post.h`，見 §1.2）。
* 要 SRP-PHAT 與 BF 時，sample 層 helper `array_sample_beam.h` 把 audiolib 的 SRP-PHAT 與 BF 接在
  `aec_process_linear()`／`aec_get_array()` 與 `aec_process_residual()` 之間（範例碼，不在函式庫 archive 內；手冊見 [beam.md](beam.md)）。
  有自己 beamformer 的應用直接呼叫 `aec.h` 的這幾個函式即可。
* `AEC_RESIDUAL_STATE` 的陣列（例如接 ULCNet）只有四條 lane，`aec_process_residual()` 與 `aec_get_residual()` 回 `AEC_ERR_MODE`；
  `AEC_RESIDUAL_ESTIMATE` 沒有增益；`AEC_RESIDUAL_FULL`（預設）才有唯一一個 G_res。`aec_process()` 需要單支麥克風，對陣列回 `AEC_ERR_MODE`。

### 1.2 Header closure

```c
#include "aec.h"
```

一行就夠 array 本身用。它自己會拉進 `Complex`（`fft_wrapper.h` → `ac_complex.h`）、`AecConfig`、`AecDelayMode`、`AecArrayView`、`AecResidualView`、`AecBeam`。
要串後級 stage 時另外 include 你用到的函式庫 header：`anr.h`、`audio_stft.h`（`AcIstft`）、`utility/mem_align.h`（`ALIGN16`）、`utility/simd_kernel_nn.h`（`skn_all_finite_f32()`）。
`array_sample_post.h`（與 `array_sample_beam.h` 同目錄的 header-only 範例碼，沒有 ABI、不進 archive，可整份複製改寫）把 `AecResidualView` 銜接到降噪器：
`array_sample_anr_gain()` 呼叫降噪器（可選把 `r2` 折進噪聲底）、`array_sample_post_reset_due()` 判斷合成尾巴是否要清，以及 `ARRAY_SAMPLE_PSD_SCALE_INV`。
−10 dB 的 comfort noise 下限是 `aec.h` 的 `AEC_CNG_NR_GAIN_FLOOR`；亂數產生器由陣列持有。

### 1.3 `Complex` type 從哪來

`Complex` 定義在 `<audiolib>/common/include/ac_complex.h`，透過 `aec.h` → `fft_wrapper.h` → `ac_complex.h` 傳遞進來。**你會直接用到它**：

* `AecArrayView.lane_error_spec[ch]` 是 `const Complex*`
* `AecBeam.weights` 與 `AecBeam.beamformed_error` 是 `const Complex*`

不需要另外 include `fft_wrapper.h` 或 `ac_complex.h`。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I ../AEC/c_impl/include
-I ../NR/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
```

前三個足以使用 array 與後級的 header（只用 array、不串 NR 時用不到 `lib/nr`）；`-I <audiolib>/common` 讓 `utility/mem_align.h`、
`utility/simd_kernel_nn.h` 找得到。`array_sample_*.h` 在 `pipelines/4ch`（再加 `-I pipelines/4ch`）。
**注意：不需要 `<audiolib>/doa`、`<audiolib>/bf`、`<audiolib>/common/utility` 這三個 include root**
—— 那是 sample 層 beamformer helper 才需要的（見 [beam.md](beam.md)）。

語言標準：`-std=gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

```
你的 .o
libaec.a
libanr.a
libaudio_common.a
-lm
```

* 沒有 array 專用的 archive：array 就在 `libaec.a`。`array_sample_beam.o` 是 sample 層 helper，由 `make -C pipelines/4ch libs` 編出，
  **不在**任何 archive 內；只有用到 helper 時才連它，並加 `libdoa.a`、`libbf.a`、`libspatial_common.a`（[beam.md](beam.md) 的建置步驟）。
  `libanr.a` 只在你串 NR 後級時才需要；只做 array、不串 NR 時連 `libaec.a` 與 `libaudio_common.a`。
* archive 路徑是 config-keyed 的（`bin/<backend>-<hash>/`）。用各 repo Makefile 的 `make -s print-lib-path` 取得，不要硬寫。
* `BACKEND=ne10` 時最終 link 要用 C++ driver（`c++`）。
* 三個 archive 必須用同一組 backend / 編譯選項建置。`aec_init_ex()` 的 descriptor 只涵蓋 `libaec.a` 自己的 pool 版面與 `backend_id`（[errors.md §8](errors.md#8-版本與相容性)），
  不會替 `libanr.a`、`libaudio_common.a` 把關。

---

## 2. Quick start

完整、可編譯的最小整合（權重先用固定值佔位；NR 與合成由應用自己初始化，放在應用配置的另一塊 16-byte 對齊記憶體，每個函式庫各自查大小、各自 init）。
array 的 pool 用 `posix_memalign()` 配置；也可以改用靜態陣列（對齊 16 bytes）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

int main(void) {
    /* 1. The array: one config, one query, one pool. */
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);   /* sets the grid: fft_size 256 */
    cfg.n_mics = AEC_ARRAY_MICS;
    cfg.residual = AEC_RESIDUAL_FULL;
    AecMemReq req;
    if (aec_get_mem_requirements(&cfg, &req) != 0) return 1;   /* -1 = invalid config */
    void* aec_mem;
    if (posix_memalign(&aec_mem, 16, (size_t)req.bytes) != 0) return 1;
    Aec* a = aec_init_ex(aec_mem, (size_t)req.bytes, &cfg, &req);   /* NULL = refused */
    if (!a) return 1;

    const int hop = aec_hop_size(a);
    const int n = hop + 1;

    /* 2. The stages behind it are separate libraries with their own sizes. */
    AnrConfig nr_cfg = anr_config_for_level_grid(cfg.sample_rate, cfg.fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, cfg.fft_size);
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);               /* 0 = invalid */
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t spec_bytes = ALIGN16((size_t)n * sizeof(Complex));
    void* post_mem;
    if (nr_bytes == 0 || synth_bytes == 0 ||
        posix_memalign(&post_mem, 16, ALIGN16(nr_bytes) + ALIGN16(synth_bytes) + spec_bytes) != 0)
        return 1;
    uint8_t* cur = (uint8_t*)post_mem;
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);               cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);     cur += ALIGN16(synth_bytes);
    Complex* spec = (Complex*)cur;
    if (!nr || !istft) return 1;

    uint32_t handled = aec_post_reset_serial(a);
    float mics[4 * 512], ref[512], out[512];                   /* 512 = the largest hop (48 kHz) */
    Complex weights[AEC_ARRAY_MICS * 513];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                        /* replace with your capture */
            for (int ch = 0; ch < AEC_ARRAY_MICS; ch++)
                mics[i * AEC_ARRAY_MICS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        /* Stage 1: shared delay, aligned far, four lanes. */
        AecArrayView front;
        memset(&front, 0, sizeof(front));
        front.struct_size = sizeof(front);
        int rc = aec_process_linear(a, mics, ref);
        if (rc == AEC_OK) rc = aec_get_array(a, &front);

        /* Your beamformer reads front.lane_error_spec[ch][bin] and writes this hop's
         * weights, channel-major [ch * n + bin]. */
        for (int ch = 0; ch < AEC_ARRAY_MICS; ch++)
            for (int k = 0; k < n; k++) {
                weights[ch * n + k].r = 0.25f;
                weights[ch * n + k].i = 0.0f;
            }

        /* Stage 2: fuse the lanes with those weights, one residual gain. */
        AecResidualView v;
        memset(&v, 0, sizeof(v));
        v.struct_size = sizeof(v);
        if (rc == AEC_OK) {
            AecBeam beam = { weights, NULL };                  /* NULL: E rebuilt from the weights */
            rc = aec_process_residual(a, &beam);
            if (rc == AEC_OK) rc = aec_get_residual(a, &v);
        }
        const float* g_nr = rc == AEC_OK ? anr_process_gain(nr, v.error_spec) : NULL;  /* estimated on the beam E; B = E unless a lane guard fired */
        if (g_nr) {
            if (v.post_reset_serial != handled) {              /* a lane hard-reset: clear the OLA tail */
                ac_istft_reset(istft);
                handled = v.post_reset_serial;
            }
            if (aec_apply_gain(a, g_nr, spec) == AEC_OK &&     /* P = base * min(G_nr, G_res) */
                aec_apply_cng(a, g_nr, spec) == AEC_OK)        /* comfort noise on the bins the gain cut */
                (void)ac_istft_push(istft, spec, out);
            else
                g_nr = NULL;
        }
        if (!g_nr || !skn_all_finite_f32(out, (size_t)hop)) {
            aec_reset(a);                                      /* every stage back to a fresh start */
            anr_reset(nr);
            ac_istft_reset(istft);
            handled = aec_post_reset_serial(a);
            continue;
        }
        /* consume out: hop floats of enhanced mono audio */
    }
    /* 3. Teardown: the stages, then the array, then the memory. */
    anr_destroy(nr);
    aec_destroy(a);
    free(post_mem);
    free(aec_mem);
    printf("ok hop=%d n_freqs=%d pool=%llu\n", hop, n, (unsigned long long)req.bytes);
    return 0;
}
```

將範例存成 Audio_ALG 根目錄的 `app.c`。以下使用預設的 sibling 依賴；執行會印 `ok hop=128 n_freqs=129 pool=701280`
（`pool` 隨函式庫版本與編譯選項而變，見 [config.md §4.4](config.md#44-檢視欄位記憶體用量與實測-pool)）。

```sh
make -C pipelines/4ch libs
common_dir="$(cd ../audiolib/common && pwd)"
common="$(make -s -C "$common_dir" print-lib-path)"
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I ../AEC/c_impl/include -I ../NR/c_impl/include \
  -I ../audiolib/common/include -I ../audiolib/common
c++ -o app app.o \
  "$(make -s -C ../AEC/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$(make -s -C ../NR/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$common" -lm
./app
```

四條 lane 的參數只有 `AecConfig` 的公開欄位（`level`、`delay_*`、`enable_cng`…）：`aec_config_from_level()` 填預設，之後直接覆寫欄位；濾波器長度等調校值由函式庫依取樣率與 level 決定；
陣列從這份 config 衍生 lane 的 config（EXTERNAL_ALIGNED、單支、`residual` 為 STATE 或 ESTIMATE），應用不必也不能自己建 lane。
**level 只作用在融合後的增益級**；lane 自己的 level 沒有效果（細節見 §3.2）。

只要回音抑制、不要降噪的產品：不建 `AnrDenoiser`，`aec_apply_gain(a, NULL, spec)` 與 `aec_apply_cng(a, NULL, spec)` 的 `extra_gain` 傳 NULL。
不要 RES 增益時把 `residual` 設成 `AEC_RESIDUAL_ESTIMATE`，用 `aec_apply_gain_spec(out, v.base_spec, unity, g_nr, n)` 套自己的增益（`unity` 為全 1 向量）且不呼叫 CNG。
每 hop 的順序見 `4ch_external_static.c` 的 `post_hop()`。
`array_sample_default_config()`（`array_sample_beam.h`）給範例用的預設 config；init 順序就是上面的 `aec_get_mem_requirements()` 加 `aec_init_ex()`。

## 3. Lifecycle

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | config | `aec_config_from_level(&cfg, level, sr)`，再設 `n_mics = AEC_ARRAY_MICS`、`residual`、delay 欄位 | 配置任何記憶體之前 | 兩個 factory 不驗證；對 8000／16000／48000 以外的取樣率，`fft_size` 為 0，之後被拒。驗證發生在第 2 步與第 4 步（[config.md §4](config.md#4-config-完整參考)） |
| 2 | query | `aec_get_mem_requirements(&cfg, &req)`（或 `aec_get_mem_size()`） | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | `aec_get_mem_requirements()` 回 `-1`（`req` 清零）：`cfg` 為 NULL 或未通過驗證。`aec_get_mem_size()` 回 `0` = 無效 |
| 3 | allocate | 你自己的配置器：`req.bytes` 且對齊 `req.alignment`（16）的一塊 | 拿到大小之後 | pool **不需要**預先清零；`aec_init_ex()` 在驗證全部通過後才寫入，被拒時 pool 一個 byte 都不動 |
| 4 | init | `aec_init_ex(pool, bytes, &cfg, &req)`（`aec_init(pool, bytes, &cfg)` 等同 `expected = NULL`） | 配置之後、處理第一個 hop 之前 | 回傳 `NULL`：descriptor 檢查任一不符；pool 未 16-byte 對齊；pool 太小；config 無效。**沒有任何診斷輸出**（見 [§5.5](errors.md#55-診斷輸出去向)） |
| 4b | 後級 stage | 各 stage 自己的 `anr_get_mem_size()`／`ac_istft_get_mem_size()` → `anr_init()`／`ac_istft_init()` | 你的後級 stage 要用時 | `mem_size` 回 `0`、`init` 回 `NULL`：config 無效或 pool 太小／未對齊。stage 的記憶體由你自己查大小、自己 carve，與 array 的 pool 分開（[§2](#2-quick-start) 的範例） |
| 5a | linear | `aec_process_linear(a, mics, ref)`；`aec_get_array(a, &front)` | 每收到一個 hop 呼叫一次 | `AEC_ERR_ARG` / `AEC_ERR_DSP`，見 [§5.2](errors.md#52-aec_process_linear-的完整拒絕清單依實際檢查順序) |
| 5b | 你的 beamformer | — | 在 linear 與 residual 之間 | 由你負責 |
| 5c | residual | `aec_process_residual(a, &beam)`；`aec_get_residual(a, &v)` | 緊接著同一幀的 `aec_process_linear()` | `AEC_ERR_ARG` / `AEC_ERR_MODE` / `AEC_ERR_SEQUENCE` / `AEC_ERR_DSP`，見 [§5.3](errors.md#53-aec_process_residual-的完整拒絕清單依實際檢查順序) |
| 5d | 你的後級 stage | `anr_process_gain()` → `aec_apply_gain()` → `aec_apply_cng()` → `ac_istft_push()` | residual 之後、下一次 `aec_process_linear()` 之前 | 由你負責 |
| 6 | reset | `aec_reset(a)` | 換回聲路徑、切換串流時 | 不會失敗。`a` 為 NULL 或已 destroy 的陣列是 no-op。四條 lane、共用 estimator、對齊環、增益級（level 設定保留）、CNG 種子與計數全部回到初始狀態。**會作廢目前的視圖**，且之後 `aec_process_residual()` 回 `AEC_ERR_SEQUENCE` 直到下一次 `aec_process_linear()`；你的 stage 要各自 reset（`anr_reset`、`ac_istft_reset`，並把你記的 `post_reset_serial` 更新成 `aec_post_reset_serial(a)`） |
| 7 | destroy | `aec_destroy(a)` | 不再處理任何 hop 之後、釋放記憶體之前 | 不會失敗。NULL 安全。caller-owned pool 不被釋放，可重複呼叫（已 destroy 的 handle 之後被拒）；`aec_create()` 建的 instance 釋放自己的 heap 區塊，**只能 destroy 一次**，指標之後作廢 |
| 8 | release | 你自己的釋放器 | **一定要在第 7 步之後** | 順序顛倒 = use-after-free |

`Aec* a = aec_create(&cfg)` 是 heap 便利路徑：函式庫配置一塊 16-byte 對齊的記憶體（大小 `aec_get_mem_size(&cfg)`），`Aec` 與 pool 都在裡面，回傳 `NULL` 表示 `cfg` 無效或配置失敗；`aec_destroy(a)` 釋放那一塊，之後指標作廢，呼叫端不再 `free()`。`Aec` 是不透明型別（不能 `sizeof(Aec)`、不能宣告變數）。其餘完全相同。

**物件存活期間，不要對它的內部物件做任何事**：四條 lane、共用 estimator 都在 pool 裡，沒有公開 handle；要重置請呼叫 `aec_reset()`。

### 3.1 執行期唯讀存取

| 呼叫 | 回傳 | NULL / 已 destroy 時 |
|---|---|---|
| `aec_hop_size(a)` | 每次 `aec_process_linear()` 要餵幾個 sample（per channel）；n_freqs = hop + 1 | NULL 或已 destroy 的 handle 回 `0` |
| `aec_resolve_signal_grid(sr, fft, &g)` | 格點（`hop_size`、`n_freqs`…）；`AEC_OK` = 成功，`AEC_ERR_ARG` = 不支援 | — |
| `aec_array_matched_filter_count(a)` | `MATCHED` 時為 `1`，`FIXED`／`EXTERNAL_ALIGNED` 時為 `0` | `0` |
| `aec_array_linear_aec_count(a)` | 陣列恆為 `4`（單支為 `1`） | `0` |
| `aec_far_fft_real_compute_count(a)` | 四條 lane 累計的 far-FFT 次數（`long`），每次 `aec_process_linear()` **+1**（不是 +4；混合 warm/soft realign 的那個 hop 是 +4）；`aec_reset()` 會歸零 | NULL 或已 destroy 的 handle 回 `0` |
| `aec_array_realign_warm_lane_count(a)`、`aec_array_realign_soft_lane_count(a)` | 累計的 lane realign 結果（`aec_apply_external_realign()` 的 warm／soft 兩種）；`delay.changed` 的 hop 兩者合計 +4（有 lane 拒絕時較少） | `0` |
| `aec_debug_status(a, &st)` | 陣列的延遲、ERLE、`usable_linear` 等快照（lane 0 的狀態配共用 delay） | 不可為 NULL |

此表「NULL／已 destroy」欄的行為只適用於 caller-owned pool 的 instance（destroy 後 handle 仍在你的 pool 裡，旗標讓 accessor 回報無效）；
`aec_create()` 的 instance 在 `aec_destroy()` 時已被釋放，之後不可再呼叫任何 accessor。

後面的計數是**結構稽核用**的：`aec_array_linear_aec_count()` 對有效 handle 回傳固定常數 4，用來在整合測試裡證明拓撲沒有被改掉；
`aec_far_fft_real_compute_count()` 用來證明跨聲道的 far-end 共用真的生效。它們對處理結果沒有任何影響。
後級 stage 是否存在由你自己的串接決定，沒有 `nr`／`post_res` 的稽核計數。

### 3.1b 逐階段耗時（診斷用）

```c
typedef struct AecArrayTiming {
    uint32_t delay_us;       /* shared delay estimator */
    uint32_t frontend_us;    /* four lanes: entry to the main filter */
    uint32_t linear_us;      /* four lanes: main adaptive filter incl. the far FFT */
    uint32_t lane_res_us;    /* four lanes: the lanes' residual stage */
    uint32_t fuse_us;        /* aec_process_residual(): the fusion and the gain */
} AecArrayTiming;

void aec_array_get_timing(const Aec* a, AecArrayTiming* out);
```

最近一個 hop 每個階段的 wall-clock 成本（微秒，`CLOCK_MONOTONIC`）。純診斷：陣列自己不回讀，也不影響處理。
`a` 為 NULL 或已 destroy 時整份記錄不動；`out` 不得為 NULL。RES／NR／CNG／逆轉換由你的應用串接，耗時由你自己量。
`aec_get_last_timing()` 回報同一份記錄但沒有 `fuse_us`（`res_us` = `lane_res_us`）。

`delay_us` 是共用 estimator 的時間，加上四條 lane 內部的 `delay_us`——後者**結構上恆為 0**，因為四條 lane 都是 `AEC_DELAY_EXTERNAL_ALIGNED`，
沒有自己的 ring 也沒有估計器；相加而不是丟棄，是為了讓日後改變 lane 模式時不會靜默漏掉那筆成本。`frontend_us`、`linear_us`、`lane_res_us` 是四條 lane 各讀一次再相加，
所以得到的是該階段在整個 lane 迴圈上的 wall-clock 成本。

全部五欄由**同一個旗標** `-DAEC_STAGE_TIMING=1` 管轄（`make PROFILE=1` 打開）；沒開時所有欄位恆為 0（release build 不讀時鐘）。
各階段**加起來不等於**整個呼叫的時間，呼叫端要自己用減法補餘額：linear 側的餘額涵蓋輸入去交錯／有限值檢查、對齊環的複製與 realign 掃描。

前四欄在 `aec_process_linear()` 接受引數（非 NULL、未 destroy、樣本皆有限）之後清零，`fuse_us` 在 `aec_process_residual()` 通過引數、sequence 與權重檢查之後清零；
所以 linear 中途 bail-out 的 hop（`AEC_ERR_DSP`），前四欄回報 0（或只含失敗前已量到的階段）而不是上一個 hop 的數字；`fuse_us` 則只在 residual 清零，
linear 失敗不會動它（仍是上一次 fuse 的值）。被 NULL 引數、已 destroy 的 handle 或非有限樣本直接拒絕的呼叫整份記錄不動。解析度是微秒，所以低於 1 µs 的階段會讀到 0。

**目標平台沒有 `CLOCK_MONOTONIC` 時**：`clock_gettime` 是 POSIX 而非 C99。用
`make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'` 換掉時鐘；巨集要寫成**純函式名**（程式碼以 `AEC_NOW_US()` 呼叫它），
指向不收參數、回傳 `uint32_t` 微秒的函式。替代時鐘必須**單調**，否則無號減法會產生接近 32-bit 全距的荒謬值。

### 3.2 執行期強度控制

強度面分兩處：融合後的增益級（在陣列裡）與降噪器（你擁有）。

```c
aec_set_level(a, AEC_LEVEL_AGGRESSIVE);                 /* 回 0 成功 */
anr_set_level(nr, ANR_LEVEL_MILD);
```

`a` 是你 `aec_init_ex()` 回來的陣列，`nr` 是你 `anr_init()` 回來的降噪器（`4ch_external_static.c` 的 `Post` 結構）。
sample 層 beamformer helper 同樣不持有這兩者，強度也是對這兩個 handle 呼叫（見 [beam.md](beam.md)）。

> **本節最重要的一件事：level 只作用在融合後的那一個增益級。**
>
> 四條 lane 的殘餘階段只算 `r2`、comfort noise 與 linear 狀態，沒有任何一條會走到自己的 suppression-gain 路徑；它們各自的地板什麼都不塑形。
> 真正乘上輸出、也真正決定 comfort noise 量的那個 gain，來自陣列內**唯一一個**增益級（`residual = AEC_RESIDUAL_FULL`）。
> `aec_set_level()` 作用在它上面；`residual` 低於 FULL 的陣列沒有增益級，呼叫只更新 config 欄位，對輸出無效。
> 這也是為什麼不要、也不需要對 lane 迴圈呼叫任何 level setter。reset 與 level 行為由測試釘住
> （`pipelines/4ch/tests/test_array_sample_beam.c` 的 `test_array_gain_*`），不是推測。

| 呼叫 | 改什麼 | 失敗（回 `-1`，**什麼都不寫**） |
|---|---|---|
| `aec_set_level(a, level)` | 陣列增益級的 far-active split floor（以及 aggressive 的 DT floor、split-floor 旗標） | `a` 為 NULL；`level` 超出 enum |
| `anr_set_level(nr, level)` | 你擁有的**那一個**降噪器（NR 是對 beamform 後的訊號跑，不是每條 lane 一個） | `nr` 為 NULL；`level` 超出 enum；該 level 的 config 被 `anr_reconfigure()` 拒絕 |

目前沒有公開的「live／target」讀回函式可查增益級的 ramp 進度；要確認改動落地，用 `aec_get_residual().res_gain` 的輸出比較。

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名與整數相同。

far-active 地板固定在 100 ms 內以 dB 為單位線性走過去（AEC 內部常數，呼叫端不能調），落點與「用該 level 從頭建一個新實例」完全相同。
mild ↔ aggressive 是 18 dB 落差、地板又是硬性 clamp，所以互動式旋鈕應該給一個 ramp。ramp 進行中再呼叫一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：四條 lane 的濾波器、共用延遲鎖定、NR 的噪聲底與增益平滑歷史全部繼續跑。level 的改動**撐得過** `aec_reset()`（目標值存在增益級的設定裡，reset 重建時沿用），
但 ramp 進度會被丟掉。**在兩個 hop 之間呼叫、與 `aec_process_linear()`／`aec_process_residual()` 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_level_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_level()`
> 直接換強度。若你在 level 組態之上改過欄位，`anr_set_level()` 會把它們蓋回 level 值：那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的 hop 上生效：double-talk 期間強制套用 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；
far-active latch 觸發之前套用的是 far-silent 地板。同一個 gain 還決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB）——地板壓得越深、CNG 反而越多）。
因此**整段錄音的平均值移動幅度會小於 dB 落差所暗示的量**，而且一個只量 echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制頭上。
請在 echo 對齊或 degradation 對齊的條件下比較，並實際試聽。

[返回本主題](README.md)
