# SRP／BF helper

## 2. Quick start

最小整合（四條 lane、共用 estimator、融合緩衝與增益級由 `aec_init_ex()` 在同一塊 heap 記憶體建立成一個 array instance，
SRP/BF helper 另一塊，NR 與合成由應用自己初始化）。下列程式照下方編譯步驟建置並執行過，
輸出 `ok hop=128 effective_adapt_interval=1 lambda=0.995998025`（與 [§4.6](beam-settings.md#46-你設定的值不一定是實際生效的值) 的第一列相同）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "array_sample_beam.h"
#include "array_sample_post.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"   /* skn_all_finite_f32 */

int main(void) {
    AecConfig cfg;
    array_sample_default_config(&cfg, 16000);              /* balanced, MATCHED, FULL, n_mics 4 */
    ArraySampleBeamParams bp = array_sample_beam_default_params();
    bp.geometry     = ARRAY_SAMPLE_GEOMETRY_UCA;
    bp.uca_radius_m = 0.035f;                             /* replace with your array's radius */

    /* The array (lanes, shared delay, aligned far, fusion, gain stage) in one heap block. */
    AecMemReq req;
    void* aec_mem = NULL;
    if (aec_get_mem_requirements(&cfg, &req) != 0 ||
        posix_memalign(&aec_mem, req.alignment, ALIGN16((size_t)req.bytes)) != 0) return 1;
    Aec* aec = aec_init_ex(aec_mem, ALIGN16((size_t)req.bytes), &cfg, &req);
    if (!aec) { fprintf(stderr, "array init failed\n"); return 1; }
    /* The SRP-PHAT/BF helper, its own pool. */
    size_t beam_bytes;
    void* beam_mem = NULL;
    if (array_sample_beam_get_mem_size(&cfg, &bp, &beam_bytes) != 0 ||
        posix_memalign(&beam_mem, 16, beam_bytes) != 0) return 1;
    ArraySampleBeam* beam = array_sample_beam_init(beam_mem, beam_bytes, &cfg, &bp, aec);
    if (!beam) { fprintf(stderr, "helper init failed\n"); return 1; }

    /* NR and synthesis are separate libraries the application sizes and initialises itself. */
    const int hop = aec_hop_size(aec);
    const int n = hop + 1;
    AnrConfig nr_cfg = anr_config_for_level_grid(cfg.sample_rate, cfg.fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, cfg.fft_size);
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    size_t vec = ALIGN16((size_t)n * sizeof(float));
    size_t total = ALIGN16(nr_bytes) + ALIGN16(synth_bytes) + vec + ALIGN16((size_t)n * sizeof(Complex));
    void* post_pool;
    if (nr_bytes == 0 || synth_bytes == 0 || posix_memalign(&post_pool, 16, total) != 0) return 1;
    uint8_t* cur = (uint8_t*)post_pool;
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);               cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);     cur += ALIGN16(synth_bytes);
    if (!nr || !istft) return 1;
    float* scratch = (float*)cur;  cur += vec;                        /* r2 folded into the NR floor */
    Complex* spec = (Complex*)cur;
    uint32_t handled = 0;

    float mics[4 * 512], ref[512], out[512];                          /* 512 = the largest hop (48 kHz) */
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < hop; i++) {                               /* replace with your capture */
            for (int ch = 0; ch < AEC_ARRAY_MICS; ch++)
                mics[i * AEC_ARRAY_MICS + ch] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        ArraySampleBeamResult bres;
        ArraySampleBeamInfo info;
        AecResidualView v;
        const float* g_nr = NULL;
        int status = array_sample_beam_process(beam, mics, ref, &bres, &info);  /* lanes + DOA + BF */
        if (status == AEC_OK)
            status = array_sample_beam_fuse(beam, &v);                          /* fuse with that BF's weights */
        if (status == AEC_OK)
            g_nr = array_sample_anr_gain(nr, &v, scratch, 1);                   /* estimated on v.error_spec */
        if (g_nr) {
            if (array_sample_post_reset_due(&handled, &v)) ac_istft_reset(istft);
            if (aec_apply_gain(aec, g_nr, spec) == AEC_OK &&                   /* P = base * min(G_nr, G_res) */
                aec_apply_cng(aec, g_nr, spec) == AEC_OK)
                (void)ac_istft_push(istft, spec, out);
            else
                g_nr = NULL;
        }
        if (!g_nr || !skn_all_finite_f32(out, (size_t)hop)) {
            fprintf(stderr, "status=%d\n", status);
            array_sample_beam_reset(beam);                    /* helper + array + SRP + BF */
            anr_reset(nr);
            ac_istft_reset(istft);
            handled = aec_post_reset_serial(aec);
            continue;
        }
        /* consume out: hop floats of enhanced mono audio;
         * info.doa_used_rad / info.vad_out / info.delay and bres.spectrum are read-only telemetry. */
    }

    printf("ok hop=%d effective_adapt_interval=%d lambda=%.9f\n", hop,
           array_sample_beam_bf_effective_adapt_interval(beam),
           (double)array_sample_beam_bf_lambda(beam));
    anr_destroy(nr);
    free(post_pool);
    array_sample_beam_destroy(beam);                          /* before the array */
    free(beam_mem);
    aec_destroy(aec);
    free(aec_mem);
    return 0;
}
```

將範例存成 Audio_ALG 根目錄的 `app.c`。以下使用預設的 sibling 依賴。

```sh
make -C pipelines/4ch libs
common_dir="$(cd ../audiolib/common && pwd)"
common="$(make -s -C "$common_dir" print-lib-path)"
obj="$(make -s -C pipelines/4ch print-obj-dir)"
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I ../AEC/c_impl/include -I ../NR/c_impl/include \
  -I ../audiolib/common/include -I ../audiolib/common \
  -I pipelines/4ch \
  -I ../audiolib/doa -I ../audiolib/bf
c++ -o app app.o \
  "$obj/array_sample_beam.o" \
  "$(make -s -C ../audiolib print-doa-lib-path)" \
  "$(make -s -C ../audiolib print-bf-lib-path)" \
  "$(make -s -C ../audiolib print-spatial-common-lib-path)" \
  "$(make -s -C ../AEC/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$(make -s -C ../NR/c_impl COMMON_DIR="$common_dir" COMMON_LIB="$common" print-lib-path)" \
  "$common" -lm
./app
```

要接 ULCNet 後濾波時，把 `cfg.residual` 設成 `AEC_RESIDUAL_STATE`、不呼叫 `array_sample_beam_fuse()`，把 `beam.spectrum`、同一次呼叫的**原始** far hop 與
`info.delay.changed`／`info.delay.solid` 交給 `ulcnet_stream_pre_process_freq()`（`beam.spectrum` 先拆成 re／im 平面，原始 far 由應用端自己的 `AcStft` 做成頻譜；見 `pipelines/4ch/alignulcnet/README.md`）。

init 順序就是 `aec_get_mem_requirements()` 加 `aec_init_ex()`，見 [integration.md §2](integration.md#2-quick-start) 與 `pipelines/4ch/array_sample_unit.h` 的
`array_sample_unit_plan()`／`_init()`／`_destroy()`（4ch_raw、4ch_static 共用；[§3](#3-lifecycle) 表列其步驟）。

## 3. Lifecycle

三塊記憶體各管各的：**array pool**（`aec_get_mem_requirements()` 的 `req.bytes`：四條 lane、estimator、對齊環、融合緩衝與增益級，16-byte 對齊）、**helper pool**（SRP、BF、beam 頻譜與權重）、**你的後級**（NR、合成與暫存）。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | query | `aec_get_mem_requirements(&cfg, &req)`；helper `array_sample_beam_get_mem_size(&cfg, &bp, &bytes)` | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | `aec_get_mem_requirements()` 回 `-1`：`out` 為 NULL；`cfg` 未通過驗證（[config.md §4.1](config.md#41-逐欄位)）。`array_sample_beam_get_mem_size()` 回 `-1`：`cfg` 無效、`bp` 未通過 [§4.2](beam-settings.md#42-陣列幾何4--2-欄)～§4.5 的驗證、SRP 或 BF 回報 size 0，或 `bf_lambda` 依生效 interval retime 後下溢為 0 |
| 2 | allocate | 你自己的配置器：array pool 取得 `req.bytes` 且對齊 `req.alignment`（16）的記憶體；helper 另一塊 16-byte 對齊 | 拿到大小之後 | pool **不需要**預先清零（poison pattern 可接受）；`array_sample_beam_init()` 自己會把 helper 的整塊 memset 成 0 |
| 3a | init：array | `aec_init_ex(mem, bytes, &cfg, &req)` | 配置之後、處理第一個 hop 之前 | 回傳 `NULL`：descriptor 檢查任一不符、pool 未對齊或太小、config 無效。**沒有任何診斷輸出**（見 [§5.4](beam-errors.md#54-診斷輸出去向)） |
| 3b | init：helper | `array_sample_beam_init(mem, bytes, &cfg, &bp, aec)` | array 建好之後；`cfg` 必須與建 array 的是同一份（至少 `sample_rate`、`fft_size` 一致） | 回傳 `NULL`：`bp` 無效、pool 未對齊或太小、array 的 hop 與 `cfg` 導出的不符、`aec` 不是四麥克風 array（`aec_array_linear_aec_count() != 4`）、SRP 或 BF init 失敗。helper 不檢查 `residual` |
| 4a | process | `array_sample_beam_process_front()` → `array_sample_beam_process_beam()`；或 `_process_with_activity()`；或 `_process()` | 每收到一個 hop 呼叫一次 | `AEC_ERR_ARG` / `AEC_ERR_SEQUENCE` / `AEC_ERR_DSP`，見 [§5.2](beam-errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4b | fuse（可選） | `array_sample_beam_fuse(b, &frame)` | 同一 hop 的 beam 之後、下一次 `process_front()` 之前 | 同上，見 [§5.2](beam-errors.md#52-process-各入口與-fuse-的完整拒絕清單) |
| 4c | 你的後級 | `array_sample_anr_gain()` → `aec_apply_gain()` → `aec_apply_cng()` → `ac_istft_push()`，或 NN stage | fuse（或 beam）之後 | 由你負責；增益來自陣列內的增益級。任何一步被拒，你必須把 helper（含 array）與你的後級**一起** reset（§3.1） |
| 5 | reset | `array_sample_beam_reset(b)`，加上你的後級的 reset | 換回聲路徑、換場景、切換到不相關的另一條串流時 | 不會失敗。`b == NULL` 或已 destroy 時是 no-op |
| 6 | destroy | `array_sample_beam_destroy(b)` → `aec_destroy(aec)` | 不再處理任何 hop 之後、釋放記憶體之前；**順序不可顛倒**（helper 借用 array） | 不會失敗。NULL 安全，helper 與 caller-pool 的 array 冪等；`aec_create()` 的 handle 只能 destroy 一次 |
| 7 | release | 你自己的釋放器 | **一定要在第 6 步之後** | 順序顛倒 = use-after-free |

helper 沒有 heap 版，pool 自己配。
`aec_init(pool, bytes, &cfg)` 等同 `aec_init_ex(pool, bytes, &cfg, NULL)` —— 不做 descriptor 檢查。

array 掛著的期間，應用不可以同時用 helper 之外的路徑驅動它（例如自己再呼叫 `aec_process_linear()`）：helper 的 SRP／BF 與 frame 計數會和 array 脫節。

### 3.1 `reset()` 到底重置了什麼，以及失敗時整合者要做什麼

`array_sample_beam_reset()` 重置：

* array（`aec_reset()`）：共用 estimator、四條 lane、增益級（level 設定保留）、參考環與 aligned far 清零、delay admission／quarantine／duty 狀態與計數器歸零、linear 有效旗標、CNG 種子
* SRP-PHAT 狀態
* BF 狀態
* helper 的內建 VAD 狀態（`noise_power` 回到門檻值、hangover 歸零）
* `frame_index` 與 per-hop phase 歸零

`aec_reset()` 不清 lane 的 buffer：reset 之前取得的 `AecArrayView`／`AecResidualView` 指標會讀到舊內容（不是 fault），但 `aec_process_residual()` 在新的 linear 之前一律拒絕。
reset **不會**重置你的後級（NR、合成、NN stage），也**不會**重新驗證 config，更不會重新計算 [§4.6](beam-settings.md#46-你設定的值不一定是實際生效的值) 那三個「生效值」——那三個是在 init 時算好的，`reset()` 不動它們。

**失敗語意是整合者的責任。** `AEC_ERR_DSP`（array 已自己 reset，你的後級沒有）、`array_sample_beam_fuse()` 因 array 的拒絕而回非 `AEC_OK`、非 `AEC_ERR_SEQUENCE`、非 `AEC_ERR_MODE`
（helper 已整個 reset；NULL 引數、沒有 beam 的 SEQUENCE 與 STATE 陣列的 MODE 是 helper 自己的拒絕，什麼都不改）、你的 `array_sample_anr_gain()` 或 `aec_apply_*` 回 `NULL`／非 `AEC_OK`，或合成輸出不是有限值，
你要**同時**呼叫 `array_sample_beam_reset()`、`anr_reset()`、`ac_istft_reset()`，並把你記的 `post_reset_serial` 設成 `aec_post_reset_serial(aec)`，以及用到 ULCNet 模型串流時的 `ulcnet_stream_reset()`。
漏掉其中一個，系統就不等於新建的實例。front／beam 階段的 `AEC_ERR_ARG` 與 `AEC_ERR_SEQUENCE` 不改動 array 與 helper 的自適應狀態（`process_front()` 因非有限輸入被拒時另外不再留待處理的 front，[§5.2](beam-errors.md#52-process-各入口與-fuse-的完整拒絕清單)），
修正輸入或呼叫順序後重送同一個 hop 即可；本手冊的範例為求簡單，對任何非 `AEC_OK` 一律一起 reset。DFN2 的 hop（`DfnHop` 與它帶的模型串流）不因為 AEC 的 delay change 而 reset（見 `pipelines/dfn/dfn_hop.h`）。

### 3.2 執行期唯讀存取

形狀：`aec_hop_size(aec)`（每次 process 要餵幾個 sample，per channel；FFT 長度 = `2 × hop`，n_freqs = `hop + 1`）、
`aec_resolve_signal_grid(cfg.sample_rate, cfg.fft_size, &g)`。

AEC、DOA、BF，以及你串的 NR，全部**共用同一組格點**，`frame == fft_size`、`hop == fft_size / 2`。
你不能分別指定互相矛盾的 frame / hop / FFT：helper 由 array 的格點導出交給 SRP 與 BF 的設定
（`array_sample_spatial_config()`），`tests/test_array_sample_beam.c` 的 `run_grid` 檢查 SRP／BF 實際收到的 `sr`、`NFFT`、`F`、`M`、
`n_freqs` 與 BF 的 adapt_interval 都在 array 的格點上（16 kHz 預設下 hop 128 / fft 256 / n_freqs 129）。

結構稽核 accessor（`aec == NULL` 或已 destroy 時回 `0`）：

| 呼叫 | 有效 handle 的回傳 |
|---|---|
| `aec_array_matched_filter_count(aec)` | `1`（`MATCHED`）；`FIXED`／`EXTERNAL_ALIGNED` 時 `0` |
| `aec_array_linear_aec_count(aec)` | `4` |

結構稽核計數只有上面兩個；NR 與合成由你自己決定。另有唯讀計數器
（`aec_far_fft_real_compute_count()`、`aec_array_realign_warm_lane_count()`／`_soft_lane_count()`），語意見 `aec.h`；duty-cycle census 讀 `aec_debug_status()`。

生效值 accessor（見 [§4.6](beam-settings.md#46-你設定的值不一定是實際生效的值)）：
`array_sample_beam_bf_effective_adapt_interval(b)`、`array_sample_beam_bf_lambda(b)`（`b == NULL` 或已 destroy 時回 `-1`／NaN）。

診斷：`aec_array_get_timing(aec, &t)`（最近一個 hop 的各階段耗時，`AecArrayTiming`；`t` 不得為 NULL；預設全 0，要 `-DAEC_STAGE_TIMING=1`，`make PROFILE=1` 打開）。

建置資訊：`array_sample_beam_spatial_backend()` —— **不吃 handle**，回傳一個 `const char*`（永不為 NULL），表示 spatial kernel 實際編進去的是哪一種實作（NEON 或 scalar）。純資訊用途，可寫進你的開機 log。

### 3.3 執行期強度控制

helper 沒有強度 setter 函式，也不持有增益級或降噪器：增益級在 array 內（`residual = FULL`），NR 是你串的後級。
你對這兩個 handle 直接呼叫函式庫的 setter（完整契約見 [integration.md §3.2](integration.md#32-執行期強度控制)）：

```c
aec_set_level(aec, AEC_LEVEL_AGGRESSIVE);    /* aec 是 array；回 0 成功 */
anr_set_level(nr, ANR_LEVEL_MILD);                   /* nr 是你自己的降噪器 */
```

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名與整數相同。

這裡只重述整合時最容易踩到的一點：

> **不要對四條 lane 做任何 level 設定。** 四條 lane 的殘餘階段只算 `r2`、comfort noise 與 linear 狀態，永遠不會走到自己的 suppression-gain 路徑，
> 它們的地板什麼都不塑形。真正乘上輸出的 gain 來自 array 內**唯一一個**增益級；同樣地，NR 也只有一個共用實例，不是每條 lane 一個。
> `aec_set_level()` 針對的就是那個增益級。`array_sample_beam_reset()` 不碰你的 NR；array 的 `aec_reset()` 不會丟掉用 `aec_set_level()` 設的 level。

far-active 地板固定在 100 ms 內以 dB 為單位線性走過去（AEC 內部常數，呼叫端不能調）。兩個 setter 都不是重啟：AEC、SRP、BF、NR 的狀態全部繼續跑；
level 的改動撐得過 reset，ramp 進度不會。**在兩個 hop 之間呼叫、與 process 序列化；非 thread-safe。**

A/B 量測時：far-active 地板只在 **far-active 且非 double-talk** 的 hop 上生效（DT 地板：mild/balanced 共用 -20 dB，aggressive 為 -30 dB；latch 觸發前用的是 far-silent 地板），而同一個 gain
還決定 comfort noise 量，所以整段錄音的平均值移動幅度會小於 dB 落差所暗示的量。細節與建議的比較方式見 [integration.md §3.2](integration.md#32-執行期強度控制)。

[返回本主題](README.md)
