# 單聲道整合手冊（`Aec` + 逐個初始化的函式庫，沒有 mono 函式庫）

適用對象：要把單聲道 AEC → NR → RES 連進自己產品的**函式庫使用者**。

本文只講「什麼時候呼叫哪個 function」「每個參數怎麼設」。演算法內部推導不在本文範圍。

> **路徑慣例**：本文所有路徑都相對於 Audio_ALG repository 根目錄。
> `audiolib` 是與 Audio_ALG 平行的另一個 checkout，本文寫成 `<audiolib>/`；
> 從 `pipelines/` 出發的相對路徑是 `../../audiolib/common`。
>
> **數值來源**：本文所有數字都是從當前 source 讀出，或以當前 checkout 實際建置後量測。
> 記憶體數字**必須**用 API 重新查詢，不要抄本文（見 §4.5）。

---

## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖（mono：`Aec` 加逐個初始化的函式庫）

生命線：應用（你的 `main`）、`Aec`（lib/aec）、`AnrDenoiser`（lib/nr）、`AcIstft`（audiolib）。
箭頭上的名稱都是真實函式；`se_derive()`（`common/se_config.c`）與 `se_frame_from_aec()`（`common/se_stage.h`）是 `pipelines/common/` 的共用程式碼，
`se_anr_stage_gain()` 在 `samples/`（範例碼，不進任何 archive，可整份複製改寫）。

```
應用                       Aec                 AnrDenoiser            AcIstft
 │                          │                       │                     │
 │══ 初始化（一次；每次 init 前都重查大小，不快取）═══════════════════════════
 │ se_derive(&req,&rules,&grid,&aec_cfg)      → grid、AecConfig（-1 = 規則被違反）
 │ aec_get_mem_size(&aec_cfg) ─────►│            → bytes（0 = 無效）
 │ anr_config_for_mode_grid(sr,fft,mode); anr_get_mem_size(&nr_cfg) ──►│   → bytes（0 = 無效）
 │ ac_stft_config_default(&sc,fft); ac_istft_get_mem_size(&sc) ───────────────►│   → bytes（0 = 無效）
 │ 配置一塊 16-byte 對齊的 pool = Σ ALIGN16(各塊) + 增益/頻譜暫存
 │ aec_init(mem,bytes,&aec_cfg) ────►│  anr_init(mem,bytes,&nr_cfg) ─►│  ac_istft_init(mem,bytes,&sc) ─►│
 │                                  （任一個回 NULL：停止，不要繼續跑）
 │══ 每個 hop（mic[hop]、ref[hop]；hop = grid.hop_size）════════════════════
 │ aec_process_context(aec,mic,ref) ►│            （void；線性 AEC + AEC 自己的 RES 增益）
 │ se_frame_from_aec(aec,&frame) ◄──│            frame{e,y,x,r2,cn,g_res_ctx,...}；0 / -1
 │ g_res = frame.g_res_ctx            （NULL = 該 hop 被拒；關 RES 時用 unity 向量）
 │ se_anr_stage_gain(nr,&frame,extra_noise,use_r2) ──────►│ anr_process_gain → G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,g_res,n); sk_capply_gain_f32(spec,frame.e,total,n)
 │ aec_cng_fill(&rng,spec,frame.cn,g_res,G_nr,SE_CNG_NR_GAIN_FLOOR,n)    （CNG 開、有 RES 增益時）
 │ ac_istft_push(istft,spec,out) ────────────────────────────────────────►│ 寫出 hop 個樣本
 │ se_hop_is_finite(out,hop)          （0 = 輸出不可用）
 │══ 任一步被拒，或輸出非有限 ══════════════════════════════════════════════
 │ aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED     （四件一起做）
 │══ 強度（兩個 hop 之間呼叫，非 thread-safe）═══════════════════════════════
 │ aec_set_preset(aec,preset,ramp_ms) ►│          anr_set_mode(nr,mode) ──►│
 │══ 結束 ═══════════════════════════════════════════════════════════════
 │ anr_destroy(nr); aec_destroy(aec); 釋放 pool    （AcIstft 沒有 destroy；先 destroy 再釋放 pool）
 │
 │ --aec-only：只建 Aec，每 hop 改呼叫 aec_process(aec,mic,ref,out)（不經 NR、合成與增益暫存）
```

延遲泳道（橫軸 = hop；`ac_istft_push()` 的 identity 鏈延遲 = 1 hop，見 `audio_stft.h` 檔頭）：

```
hop 序號          t        t+1       t+2
輸入 mic/ref      m[t]     m[t+1]    m[t+2]
Aec + RES + NR    處理 m[t] 處理 m[t+1] 處理 m[t+2]     （同一次呼叫內完成，線性路徑本身延遲 0）
輸出 out          y[t-1]   y[t]      y[t+1]            （合成 OLA 多 1 hop；--aec-only 沒有這 1 hop）
```

1 hop = `grid.hop_size` 個樣本：16 kHz/256 為 128 個樣本（8 ms）、16 kHz/512 為 256（16 ms）、48 kHz 為 512（約 10.7 ms）；8 kHz/256 為 128（16 ms）。數字由 §4.3 的格點推得。

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開）**

| 階段 | 函式（標頭） | 備註 |
|---|---|---|
| 規則與格點 | `se_derive()`（`common/se_config.h`） | 純函式，`SeRequest` + `SeRules` → `SeGrid` + `AecConfig`；回 `-1` 即拒絕 |
| 大小查詢 | `aec_get_mem_size()`（`aec.h`）、`anr_get_mem_size()`（`anr.h`）、`ac_istft_get_mem_size()`（`audio_stft.h`） | 回 `0` = config 無效 |
| 建立 | `aec_init()`、`anr_init()`、`ac_istft_init()`；桌面也可 `aec_create()`、`anr_create()` | pool 路徑回 `NULL` = 失敗 |
| 每 hop | `aec_process_context()`、`se_frame_from_aec()`（`common/se_stage.h`）、`se_anr_stage_gain()`（`samples/se_anr_stage.h`）、`sk_min_f32()`、`sk_capply_gain_f32()`（`utility/simd_kernels.h`）、`aec_cng_fill()`（`aec_res.h`）、`ac_istft_push()`、`se_hop_is_finite()` | 順序見 §0.1 |
| 只要線性 AEC | `aec_process()` | 與 `aec_process_context()` 推進相同狀態 |
| 強度 | `aec_set_preset()`、`anr_set_mode()`（需要整組覆寫時 `anr_reconfigure()`） | §3.2 |
| 重置／收尾 | `aec_reset()`、`anr_reset()`、`ac_istft_reset()`、`aec_destroy()`、`anr_destroy()` | §3、§5 |
| 診斷（唯讀） | `aec_debug_status()`、`anr_debug_status()`、`aec_get_last_timing()` | §3.1、§3.1b |

**內部函式（不要依賴，隨版本改變）**：`aec.h` 以外的 AEC 內部標頭（`pbfdkf.h`、`delay_aec.h`、`aec_post.h`、
`suppression_gain.h`、`aec_config_build.h`、`aec_balanced_config.h`）；NR 的 `noise_estimator.h`、`spp_estimator.h`、
`anr_internal.h`，以及 `anr_apply_*_experiment()`；audiolib 的 `utility/` 內核（除上表列出的 `sk_*`、`ALIGN16` 外）。
`struct Aec` 目前整個攤在 `aec.h` 裡，只是因為 C 需要完整型別；欄位是內部，只能經由函式存取。

### 0.3 參數面：外部與內部

**外部（建議讓第三方整合者調）**

| 名稱 | 位置 | 範圍／預設 | 何時設定 | 作用 |
|---|---|---|---|---|
| `sample_rate`、`fft_size` | `SeRequest`（`se_config.h:29-37`） | 8000（需 `rules.allow_8k`）／16000／48000；`fft_size` 0 = rate 預設，16 kHz 另可 256／512 | init 前，之後不可變 | 決定格點（§4.3）；改了要重查所有大小 |
| `filter_length` | `SeRequest` | 0 = preset／rate 預設，否則 `[1, 4096]` | init 前 | 回聲路徑長度（taps）；越長記憶體與算量越大 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `SeRequest` | `MATCHED`（預設，bank 1..5，預設 5）／`FIXED`（已知延遲）／`EXTERNAL_ALIGNED`（上游已對齊） | init 前，之後不可變 | 對齊方式；決定 pool 版面（§4.4、§4.5a） |
| `aec_preset` | `SeRequest` | `MILD`／`BALANCED`（預設）／`AGGRESSIVE`／`NO_FLOOR`；enum 外被拒 | init 前；執行期用下一列 | 回聲抑制 vs 近端保留的單一取捨軸 |
| `aec_set_preset(aec, preset, ramp_ms)` | `aec.h` | `ramp_ms` 屬於 `[0, 60000]`，`0` = 下一 hop 生效 | 兩個 hop 之間 | 執行期換 AEC 強度；回 `-1` 時什麼都不寫 |
| NR 強度 | `AnrMode`（`anr_types.h`）：`anr_config_for_mode_grid()`、`anr_set_mode()` | `MILD`／`MODERATE`／`BALANCED`（預設）／`AGGRESSIVE` | init 前或兩個 hop 之間 | 降噪深度的單一入口（一次調 `g_min_db`、`q`、`xi_min_db`、`noise_over_subtraction`） |
| 後級開關（範例程式的欄位，不是函式庫 config） | `PostOptions`／`PostPlan`（`main.c`、`static_main.c`）；CLI `--no-res`、`--no-nr`、`--no-cng`、`--legacy-amin` | 各 0／1 | init 前 | RES 增益用不用、要不要建降噪器、CNG、NR 是否折入 R²（§4.2） |

進階外部（選擇性；改了以後自己重跑三個 grid 的驗證）：`AecConfig` 的 `enable_cng`（`aec.h:67`，下游已有舒適噪音時關）、
`enable_highpass`／`highpass_cutoff_hz`（`aec.h:70,105`，上游已有高通時關；開啟時截止頻率須 `> 0` 且低於 `0.45 × sample_rate`）、
`enable_shadow`（`aec.h:72`，關掉省記憶體但路徑突變後恢復變慢）、`delay_est_period_s`（`aec.h:99`，預設 0 = 每 hop 全速分析；非 0 會省算力，但可能改變選到的 lag）。這些在 `se_derive()` 之後、`aec_get_mem_size()` 之前直接改 `aec_cfg`。

**內部（存在、可調，但不建議開放）**

| 家族 | 位置 | 為什麼不開放 |
|---|---|---|
| `se_derive()` 固定的接縫：`enable_res = 0`、`return_res_context = 1`、`spatial_linear_context`（`SE_CONTEXT_PRESET` = 沿用 preset） | `aec.h:73,78,82`；`se_config.c` 的 `se_derive()` | 改了就沒有 `frame.g_res_ctx`（或時域輸出改道），串接整個失效 |
| 以 10 ms 格點撰寫、建立時 retime 的時間欄位：`warmup_frames`、`epc_hangover`、`ne_recent_hold`、`filter_misadjustment_*_frames`、`shadow_err_alpha` | `aec.h:84-136` | 單位是「N × 10 ms」，不是 hop 數；填錯單位會靜默失效 |
| 適應步進與模型不確定度：`mu`、`delta`、`kalman_q_high`、`kalman_q_low`、`shadow_mu_*`、`epc_mu_floor` | `aec.h:64-65,86-93,107-108` | 決定收斂與發散邊界；驗證器範圍很寬，範圍內多數值不收斂 |
| 偵測門檻：`epc_*`、`ne_recent_*`、`saturation_threshold`、`filter_misadjustment_scale_*` | `aec.h:91-92,106,123-136` | 與其他偵測器聯合調過；`scale_min > scale_max` 不會被驗證器擋 |
| 延遲獲取策略：`delay_acquire_*`、`delay_est_init_s`、`max_delay_ms`、`delay_buffer_ms`（`delay_est_period_s` 是上面的進階外部） | `aec.h:94-114` | 影響冷啟動；`max_delay_ms`／`delay_buffer_ms` 只決定搜尋環大小，不是可擷取上限（上限由 `delay_num_filters` 決定） |
| Double-talk 政策與 floor：`dt_aware_*`、`min_gain_floor_*_db`、`min_gain_split_floor_enabled` | `aec.h:119-129,155` | 三個 floor 欄位已由 `aec_preset`／`aec_set_preset()` 成組設定；單獨改 dB 值會得到沒有物理意義的 floor |
| 預設關閉的實驗：`live_observer_*`、`cng_n2_windowed_psd` | `aec.h:68,147-152` | 未通過 default-on 門檻；observer 開啟會增加 pool 與算量 |
| 衍生值：`n_partitions`、`enable_delay_est`（deprecated） | `aec.h:63,69` | 由 `filter_length` 與 hop 推導，手填只會造成不一致 |
| `SuppressionGainConfig`、`AecPostConfig`、`AecStateConfig`、`aec_balanced_config.h` 的常數 | AEC 內部標頭 | 編譯期依 `(sample_rate, fft_size)` 查表，runtime 改不了，也不該提供接口 |
| NR 的追蹤器與平滑欄位：`alpha_xi`、`alpha_s`、`alpha_d`、`alpha_p`、`alpha_attack`、`alpha_decay`、`delta_db`、`scene_change_*`、`speech_protect_*`、`speech_aware_*`、`noise_gate_*`、`makeup_*`、`cross_band_*` | `anr_types.h:56-110` | 聯合調出的操作點，四個 `AnrMode` 共用；`alpha_g` 在增益路徑上沒有作用（`anr_types.h:107`） |
| NR 的 `L`、`num_init_frames` | `anr_types.h:94,96` | 決定 pool 版面，`anr_reconfigure()` 與 `anr_set_mode()` 不能改它們；範例用函式庫預設 |

---

## 1. 你連結什麼、你 `#include` 什麼

### 1.1 構成：每個函式庫各自初始化，每個 hop 的串接由你寫

mono 沒有 unit、也沒有 pipeline 函式庫。處理鏈由下列獨立的函式庫物件組成，
**每一個都有自己的 `mem_size` 與 `init`**，初始化與每個 hop 的串接都是整合者的事：

```
mic hop + ref hop ──► Aec (aec_process_context) ──► se_frame_from_aec ──► SeSpecFrame
                                                                              │
        ┌─────────────────────────────────────────────────────────────────────┘
        ▼   G_res = frame.g_res_ctx（AEC 自己的殘留 gain；不要 RES 時用 unity）
   nr = se_anr_stage_gain(...)  ─►  sk_min_f32(nr, G_res)  ─►  sk_capply_gain_f32(E)
        ─►  aec_cng_fill（CNG 開著、且有 RES gain 時）  ─►  ac_istft_push  ──► out hop
```

mono 的 RES 是明確的例外：它在 `Aec` 的呼叫裡（`aec_process_context()` 算出殘留 gain，`se_frame_from_aec()` 以
`frame.g_res_ctx` 交出，強度用 `aec_set_preset()`），沒有獨立的 `AecRes` 物件；範例只決定要不要套用這個 gain。
純 AEC 的範例（`--aec-only`）可改呼叫完整的 `aec_process()`，RES 與最後的合成在同一個呼叫內完成。

| 物件 | 來源 | 大小查詢 | 初始化 |
|---|---|---|---|
| `Aec`（線性 AEC，含 AEC3 殘留 gain 的 context；RES gain 就是它自己的 `ctx.res_gain`） | `lib/aec`（`libaec.a`） | `aec_get_mem_size(&aec_cfg)` | `aec_init(mem, bytes, &aec_cfg)` |
| `AnrDenoiser`（可選） | `lib/nr`（`libanr.a`） | `anr_get_mem_size(&nr_cfg)`（`nr_cfg = anr_config_for_mode_grid(sr, fft_size, mode)`，最小值追蹤視窗 `L` 用函式庫預設） | `anr_init()` |
| `AcIstft`（合成） | audiolib（`libaudio_common.a`） | `ac_istft_get_mem_size(&synth_cfg)`（`ac_stft_config_default(&synth_cfg, fft_size)`） | `ac_istft_init()` |
| 增益／頻譜暫存（unity、總增益、NR prior、輸出頻譜） | 你自己的記憶體 | `3 × ALIGN16(n_freqs × sizeof(float)) + ALIGN16(n_freqs × sizeof(Complex))` | 全部設為 unity 的向量要自己填 1 |
| 格點與 `AecConfig` 推導 | `pipelines/common/se_config.{c,h}`（`se_derive()`） | – | – |

上面幾項怎麼放進 pool（heap 路徑各自配置，caller-pool 路徑依序切同一塊）與每個 hop 怎麼串，是範例程式
（`mono_aec_nr_res/main.c`、`static_main.c`、`example_board_adapter.c`）各自寫出來的，沒有 ABI、沒有 layout version、沒有 descriptor、
不進任何 archive，可整份複製改寫。唯一可共用的範例 header 是 `samples/se_anr_stage.h`
（`se_anr_stage_gain()`：把 `SeSpecFrame` 餵給降噪器並回傳 NR 增益）。

每個 hop 的串接（`main.c` 與 `static_main.c` 的 `mono_form()`）：

1. `res = frame.g_res_ctx`（不要 RES 時用全 1 的 unity 向量；`g_res_ctx` 是 NULL 表示 lane 沒有 seam，視為拒絕）。
2. `nr = se_anr_stage_gain(nr_handle, &frame, extra_noise, use_r2)`；沒有降噪器時 `nr` 用 unity 向量，NULL 回傳表示降噪器拒絕該 hop。
3. `sk_min_f32(total, nr, res, n_freqs)`，再 `sk_capply_gain_f32(spec, frame.e, total, n_freqs)`。
4. CNG 開著（`aec_cfg.enable_cng` 且沒有 `--no-cng`）、有 RES gain、且 `frame.cn` 非 NULL 時 `aec_cng_fill(&rng, spec, frame.cn, res, nr, SE_CNG_NR_GAIN_FLOOR, n_freqs)`。
5. `ac_istft_push(istft, spec, out)` 產生這個 hop 的時域輸出。

### 1.2 `--aec-only`

只要線性 AEC 的輸出，就不建後級：`aec_process(aec, mic, ref, out)` 直接輸出時域 hop，
不需要降噪器、合成與增益暫存（見 `static_main.c --print-mem-size --aec-only`：pool 只有 `Aec` 那一塊）。
`aec_process()` 與 `aec_process_context()` 推進相同的狀態，同一個實例上可以混用。

### 1.3 Header closure

```c
#include "aec.h"                       /* Aec, AecConfig, aec_process_context()               */
#include "common/se_config.h"          /* se_derive(): 格點 + AecConfig                         */
#include "common/se_stage.h"           /* SeSpecFrame, se_frame_from_aec(), se_res_stage_gain() */
#include "aec_res.h"                   /* aec_cng_fill()                                        */
#include "anr.h"                       /* AnrDenoiser, anr_config_for_mode_grid(), anr_init()   */
#include "audio_stft.h"                /* AcIstft, ac_istft_init(), ac_istft_push()           */
#include "samples/se_anr_stage.h"      /* se_anr_stage_gain()                                 */
#include "utility/simd_kernels.h"      /* sk_min_f32(), sk_capply_gain_f32()                  */
#include "utility/mem_align.h"         /* ALIGN16                                             */
```

`Complex` 定義在 `<audiolib>/common/include/fft_wrapper.h`，已透過 `se_stage.h` 帶進來。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I pipelines
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
```

`pipelines/Makefile` 自己建置範例時的 `-I` 清單比這個長（多了 `lib/aec/c_impl/example`），
那是給 CLI 的 WAV I/O 用的，consumer 不需要。語言標準：以 `-std=gnu99` 建置；
NE10 後端的 header 需要 GNU 擴充，所以你的 TU 也請用 `gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

link 順序有意義（單向相依，由上而下）：

```
你的 .o
se_config.o                 （pipelines/common/se_config.c 編出來的 object；你自己編或取 print-obj-dir 下的）
libaec.a
libanr.a
libaudio_common.a
-lm
```

* mono 沒有函式庫 TU。`se_config.o` 是唯一需要 pipelines 提供的 object，
  `se_stage.h`、`se_anr_stage.h` 是 header-only。
* 三個 archive 的實際路徑是 config-keyed 的（`bin/<backend>-<hash>/`），不是固定名稱。
  用各自 Makefile 的 `make -s print-lib-path` 取得，不要硬寫路徑。
* `BACKEND=ne10` 時 audiolib 內含一個 C++ TU，最終 link 要用 C++ driver（`c++`），不是 `cc`。
* 各 archive 必須用**同一組** backend / 編譯選項建置。

---

## 2. Quick start

以下是完整、可編譯、已實際跑過的最小整合（用 `posix_memalign` 當 pool 的主機版）：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

int main(void) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    SeGrid grid;
    AecConfig aec_cfg;
    memset(&req, 0, sizeof(req));
    req.sample_rate = 16000;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = 5;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &grid, &aec_cfg) != 0) return 1;      /* grid + AecConfig */
    const int n = grid.n_freqs;

    /* every library sizes and initialises itself */
    AnrConfig nr_cfg = anr_config_for_mode_grid(16000, grid.fft_size, ANR_MODE_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, grid.fft_size);
    size_t aec_bytes = aec_get_mem_size(&aec_cfg);                    /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);                      /* 0 = invalid */
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);           /* 0 = invalid */
    size_t vec = ALIGN16((size_t)n * sizeof(float));                  /* unity, total, NR extra_noise */
    size_t buf_bytes = 3 * vec + ALIGN16((size_t)n * sizeof(Complex));/* + the output spectrum  */
    if (aec_bytes == 0 || nr_bytes == 0 || synth_bytes == 0) return 1;
    void* pool = NULL;
    if (posix_memalign(&pool, 16, ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                  ALIGN16(synth_bytes) + buf_bytes) != 0) return 1;

    uint8_t* cur = (uint8_t*)pool;
    Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);                    cur += ALIGN16(aec_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);               cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);     cur += ALIGN16(synth_bytes);
    if (!aec || !nr || !istft) return 1;
    float* unity = (float*)cur;       cur += vec;
    float* total = (float*)cur;       cur += vec;
    float* extra_noise = (float*)cur;       cur += vec;
    Complex* spec = (Complex*)cur;
    for (int k = 0; k < n; k++) unity[k] = 1.0f;
    uint32_t rng = SE_CNG_RNG_SEED;                                   /* comfort-noise generator */

    float mic[512], ref[512], out[512];
    for (int frame = 0; frame < 100; frame++) {
        SeSpecFrame f;
        for (int i = 0; i < grid.hop_size; i++) {            /* your capture source */
            mic[i] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        aec_process_context(aec, mic, ref);                   /* linear AEC, no time-domain emit */
        const float* res = NULL;
        const float* gnr = NULL;
        if (se_frame_from_aec(aec, &f) == 0 && (res = f.g_res_ctx) != NULL)   /* G_res: the AEC's own */
            gnr = se_anr_stage_gain(nr, &f, extra_noise, 1);        /* NR gain, with the echo powers */
        if (!gnr) {
            aec_reset(aec);                                   /* a refused hop: reset every stage */
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = SE_CNG_RNG_SEED;
            continue;
        }
        sk_min_f32(total, gnr, res, n);                       /* keeps a non-finite gain visible */
        sk_capply_gain_f32(spec, f.e, total, n);
        if (aec_cfg.enable_cng && f.cn)                       /* comfort noise on the bins RES cut */
            aec_cng_fill(&rng, spec, f.cn, res, gnr, SE_CNG_NR_GAIN_FLOOR, n);
        (void)ac_istft_push(istft, spec, out);
        if (!se_hop_is_finite(out, grid.hop_size)) {
            aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED;
            continue;
        }
        /* consume out: grid.hop_size floats */
    }

    anr_destroy(nr);                                          /* releases the denoiser, not the pool */
    aec_destroy(aec);
    free(pool);
    printf("ok hop=%d\n", grid.hop_size);
    return 0;
}
```

編譯（`<...>` 換成你 `print-lib-path` 查到的路徑；`se_config.c` 用同一組 `-I`）：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I pipelines -I lib/aec/c_impl/include -I lib/nr/c_impl/include \
  -I <audiolib>/common/include -I <audiolib>/common
cc -std=gnu99 -O2 -c pipelines/common/se_config.c -o se_config.o \
  -I pipelines -I lib/aec/c_impl/include -I lib/nr/c_impl/include \
  -I <audiolib>/common/include -I <audiolib>/common
c++ -o app app.o se_config.o \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

> `float mic[512]` 是因為本例已知 hop 最大為 512（48 kHz）。
> 產品程式請依 `grid.hop_size` 配置，不要寫死。

---

## 3. Lifecycle

板端（caller-owned pool）的完整順序如下。每個函式庫各自查大小、各自 init，整合者負責加總與切塊。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | derive | `se_derive(&req, &rules, &grid, &aec_cfg)` | 每次 init 之前 | 回傳 `-1`：任何一條規則被違反（§4.1）；輸出內容未定義 |
| 2 | query | `aec_get_mem_size(&aec_cfg)`、`anr_get_mem_size(&nr_cfg)`、`ac_istft_get_mem_size(&synth_cfg)`，加上你自己的增益／頻譜暫存 | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `0`：config 無效（**注意：這幾個是 0 = 失敗，與 4ch 單元的 `-1` 慣例不同**） |
| 3 | allocate | 你自己的配置器，取得各塊 `ALIGN16(...)` 相加且 16-byte 對齊的記憶體 | 拿到大小之後 | 配置失敗由你自己處理。pool **不需要**預先清零（poison pattern 也可以）；但必須在 handle 生命週期內獨佔、不被其他人寫入 |
| 4 | init | `aec_init(pool, aec_bytes, &aec_cfg)`、`anr_init(pool + ALIGN16(aec_bytes), nr_bytes, &nr_cfg)`、`ac_istft_init(...)`；每一塊的起點都是前一塊加 `ALIGN16(前一塊大小)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：pool 未 16-byte 對齊、太小、config 無效、子模組 init 失敗。已建好的那幾個要由你 destroy |
| 5 | process | `aec_process_context(aec, mic, ref)` → `se_frame_from_aec(aec, &f)` → 每 hop 串接（§1.1：`g_res_ctx`、`se_anr_stage_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`）→ `ac_istft_push(istft, spec, out)` | 每收到一個 hop 呼叫一次。`mic`/`ref`/`out` 都必須剛好 `grid.hop_size` 個 float | `aec_process_context()` 回傳 `void`；`se_frame_from_aec()` 回 `-1`（NULL 引數或 lane 沒有 seam）；`g_res_ctx` 為 NULL、`se_anr_stage_gain()` 回 `NULL`（NULL 引數、或降噪器拒絕該 hop）；`ac_istft_push()` 的回傳值與輸出有限性（`se_hop_is_finite()`）要自己檢查。**不會**檢查 buffer 長度 —— 長度錯是你的責任 |
| 6 | recover | `aec_reset(aec)`、`anr_reset(nr)`、`ac_istft_reset(istft)`，並把你的 comfort-noise RNG 設回 `SE_CNG_RNG_SEED` | 任何一步被拒、輸出非有限、換回聲路徑（換喇叭、重新 seat）或切換到不相關的另一條串流時 | 不會失敗。必須全部一起 reset：漏掉其中一個就不等於新建的實例（`main.c`／`static_main.c` 的 `rig_reset()` 就是這四件事） |
| 7 | destroy | `anr_destroy(nr)`、`aec_destroy(aec)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。`aec_destroy` NULL 安全、可重複，只放開實例內部的參照，**不**釋放你的 pool；`anr_destroy` 對 pool 實例立即返回，對 `anr_create()` 的 heap 實例才釋放記憶體 |
| 8 | release | 你自己的釋放器 | **一定要在第 7 步之後** | 順序顛倒 = use-after-free |

補充規則：

* **大小不可快取**。firmware image 若拿舊 build 算出的大小去餵新 library，就會在錯誤大小/佈局的 pool 上切記憶體。
  mono 沒有 descriptor 閘（那是 4ch 單元的 `AudioPipeline4Ch` 才有的契約，見
  [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md)），所以每次 build 都重查大小是你的責任；
  `aec_init()`、`anr_init()` 與 `ac_istft_init()` 只會擋下過小或未對齊的 pool。
* pool 生命週期內 sub-module 都是指進 pool 的裸指標，不是拷貝。pool 不能被搬動、
  不能被別人寫、不能和另一個 `Aec`/NR/FFT instance 共用。
* 單一 instance 不可被多個 thread 同時 `process`。多條串流請各自建立 instance。
* `SeSpecFrame` 的指標都借自 `Aec`，到下一次 `aec_process_context()` 或 reset 為止有效；`P`（輸出頻譜）必須在同一個 hop 內形成並合成。

### 3.1 執行期唯讀存取

| 呼叫 | 用途 |
|---|---|
| `grid.hop_size` / `grid.n_freqs` / `grid.fft_size`（`se_derive()` 輸出） | 每次 process 要餵幾個 float、頻域 bin 數 |
| `anr_debug_status(nr, &st)` | 診斷：降噪器的 init 旗標、增益、噪聲底（唯讀，不改狀態） |
| `aec_debug_status(aec, &st)` | 診斷：delay、ERLE、收斂旗標（mono CLI 的 `--debug` 就是這個加 `anr_debug_status()`） |

### 3.1b 逐階段耗時（診斷用）

`aec_get_last_timing(aec, &t)` 回報 **AEC 上一個 hop** 的各階段牆鐘耗時（微秒），純診斷：

```c
AecStageTiming t;
aec_process_context(aec, mic, ref);
aec_get_last_timing(aec, &t);
/* t.delay_us t.frontend_us t.linear_us t.res_us ← lib/aec 量 */
```

增益串接與合成的耗時是**範例程式自己**在呼叫前後讀時鐘量的（`mono_aec_nr_res/main.c` 的 `process_hop()`
與 `timing_now_us()`），不是任何函式庫的 API。

| 欄位 | 涵蓋範圍 |
|---|---|
| `delay_us` | **延遲估計與 ring 對齊**：matched-filter bank、其 duty-cycle 記帳、reference ring 寫入與延遲補償讀取。`EXTERNAL_ALIGNED` 下為 0，`FIXED` 下很小 |
| `frontend_us` | AEC 到主濾波器之前的**其餘**部分：mic HPF、飽和、render activity、`mu_scale`、mic-clip、RSA、shadow filter。**不含 `delay_us`**，兩者不重複計算 |
| `linear_us` | AEC 主自適應濾波器（`pbfdkf_process_with_far_cmag2`），含 far-end FFT |
| `res_us` | AEC 自己的 AEC3 post/RES 區塊 |

**本機實測：`delay_us` 是整個 hop 最大的一塊。** 主機（macOS）上以 `make PROFILE=1` 建置，
`aec_nr_pipeline --timing`、16 kHz/256、balanced、一段 20 s 的合成雜訊輸入、跑兩次：`delay_us` 約佔呼叫牆鐘的
54.6–54.8 %，`frontend_us`（mic HPF/RSA/shadow filter）約 10.1 %。比例隨輸入與平台改變，目標平台上的比例待量測。
這條線直接對應 `delay_num_filters`（1..5）這個可調旋鈕。

**各階段不會加總成整通呼叫。** 餘數包含引數檢查、`aec_get_res_context()`，以及
AEC 內部（因此從這層看不見）的 stationarity 更新、`e2_coarse`/ERL 發布、DT
analyzer、EPV、shadow_rise、失調估計、功率 EMA 與收斂偵測。要呈現完整分解的
呼叫端必須把餘數明確列出來，不能讓部分看起來像全部。

**預設關閉。** AEC 內部 7 次 clock 讀取，release build 一次都不做，欄位讀 0。
`make PROFILE=1` 會開啟（`-DAEC_STAGE_TIMING=1`）。呼叫端自己的顯示旗標（`--timing`）**不會**打開 AEC 的量測。

**記錄本身不是條件編譯的，只有取樣是。** 欄位無論旗標開關都在 struct 裡，所以
profile build 與 release build 的 pool 大小與每個 offset 都相同，開旗標不會移動任何 offset。

**目標平台沒有 `CLOCK_MONOTONIC` 時。** `clock_gettime` 是 POSIX 而非 C99。這種
平台用 `make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'`
換掉 AEC 的時鐘：巨集要是**純識別字**（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），
指向不收參數、回傳 `uint32_t` 微秒的函式。替代時鐘必須單調，否則無號減法會產生接近 32-bit 全距的
荒謬值。每個元件各有自己的覆寫點（`lib/aec`、四路核心），刻意不共用——一個元件的時鐘屬於該元件自己的建置契約。
開著旗標但指向常數函式是合法用法，所有欄位讀 0，`--timing` 的報表無法分辨這種情況與「根本沒開旗標」。

兩支 CLI 都有 `--timing`，在結束時印出整段 run 的每 hop 平均與佔比（含未歸屬餘數）。

### 3.2 執行期強度控制

兩條強度軸都可以在**運轉中**改，不需要重建、不動 pool：

```c
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);
int anr_set_mode(AnrDenoiser* nr, AnrMode mode);   /* nr = 你自己 anr_init() 回來的那一個 */
```

```c
/* 使用者把回音抑制轉到 aggressive、降噪轉到 mild */
if (aec_set_preset(aec, AEC_PRESET_AGGRESSIVE, 100.0f) != 0)   { /* 引數不合法 */ }
if (anr_set_mode(nr, ANR_MODE_MILD) != 0)                      { /* 同上 */ }

aec_process_context(aec, mic, ref);   /* 下一個 hop 起生效 */
```

| 呼叫 | 改什麼 | 回 `-1` 的情況（`-1` 時**什麼都不寫**） |
|---|---|---|
| `aec_set_preset()` | `Aec` 的殘留回音地板（preset 改變 far-active 地板 -20/-28/-38 dB；aggressive 另把 DT 地板設成 -30 dB；NO_FLOOR 關閉 `min_gain_split_floor_enabled`；mono 的 RES 增益 `g_res_ctx` 就是這個 lane 自己算的，所以 mono 的 RES 強度就是這個 setter） | `a` 為 NULL；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_mode()` | 降噪器的強度 preset（`q`、`xi_min_db`、`g_min_db`、`noise_over_subtraction`） | `nr` 為 NULL（含你沒有建降噪器）；`mode` 超出 enum |

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、而地板是硬性 clamp，所以互動式旋鈕應該給一個 ramp（100 ms 是合理起點）。
ramp 進行中再呼叫一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：AEC 的濾波器與延遲鎖定、NR 追蹤中的噪聲底與增益平滑歷史
全部繼續跑。要重啟請用 §3 第 6 步的整組 reset。**在兩個 hop 之間呼叫、與
process 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_mode_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_mode()`
> 直接換強度。若你在 preset 之上改過欄位，`anr_set_mode()` 會把它們蓋回 preset 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間套用的是 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 `ctx.res_gain`
還決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB），見
`aec_cng_fill()`，範例的 `mono_form()` 在 CNG 步驟呼叫它——地板壓得越深、CNG 反而越多）。
因此**整段錄音的平均值移動
幅度會小於 dB 落差所暗示的量**，而且一個只量 echo／degradation 的 A/B 會把 CNG 的
變化錯記到別的機制頭上。請在 echo 對齊或 degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

mono 的設定分成兩份：`SeRequest`（交給 `se_derive()`，決定格點與 `AecConfig`）與
範例程式自己的後級選項（RES gain 用不用、NR 開不開與強度、CNG、`use_r2`、合成），後者**不是**任何函式庫的
config struct，是 `main.c`／`static_main.c` 的 `PostOptions`／`PostPlan` 這類程式內部欄位，決定要不要呼叫、
要不要建對應的函式庫物件。驗證發生在 `se_derive()` 本身（回 `-1`）與
`aec_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()`（回 `0`）。

### 4.1 `SeRequest` → `se_derive()`

| 欄位 | 型別 | 範例的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你的取樣率 | `8000`（需 `rules.allow_8k`）/ `16000` / `48000`。其他一律拒絕（例如 `44100`） | 設成你音訊實際的取樣率。這個值同時決定預設 grid |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設。明確指定時：8 kHz 只接受 `256`；16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024`。其他組合拒絕 | 除非你有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `0` = 依 preset／rate 取預設；否則 `[1, SE_FILTER_LENGTH_MAX(4096)]` | PBFDKF taps，init 後不可變；沒有明確需求就留 `0` |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `AEC_DELAY_MATCHED` / `AEC_DELAY_FIXED` / `AEC_DELAY_EXTERNAL_ALIGNED` | 語意同 `lib/aec`；記憶體差異見 §4.5a |
| `delay_num_filters` | `int` | `5` | 僅 `MATCHED` 有效：`[1, 5]`；其他模式要求維持 `5`（`rules.check_delay = 1` 時檢查） | matched-filter bank 大小，決定 bulk delay 搜尋範圍；init 參數，不是 runtime setter |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED` 有效（原生取樣點數，上限為該取樣率下的兩分鐘）；其他模式須為 `-1`（`rules.check_delay = 1` 時檢查） | 已知固定 delay 時使用 |
| `aec_preset` | `AecPreset` | `AEC_PRESET_BALANCED`（= 1） | `AEC_PRESET_MILD`(0) / `AEC_PRESET_BALANCED`(1) / `AEC_PRESET_AGGRESSIVE`(2) / `AEC_PRESET_NO_FLOOR`(3)。列舉以外的整數會被拒絕（**不會**默默 fallback 成 balanced） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE`；不確定就留預設 |

`SeRules` 是你的應用自己的規則：`allow_8k`、`check_delay`（是否拒絕與 `delay_mode` 矛盾的 delay 設定；
mono 範例沒開）、`spatial_context`（mono 用 `SE_CONTEXT_PRESET`：lane 保留 preset 自己的 context 層級，
所以 `g_res_ctx` 有值）。`se_derive()` 輸出的 `AecConfig` 已設好 `enable_res = 0`、`return_res_context = 1`：
lane 不自己輸出時域音訊，增益由你的每 hop 串接套用。

### 4.2 後級選項（範例程式的欄位與 CLI 旗標）

| 選項 | 範例的預設 | 它決定什麼 |
|---|---|---|
| RES gain | 用（`frame.g_res_ctx`，即 AEC 自己的殘留 gain） | `--no-res`：`G_res` 視為全 1 的 unity 向量，comfort noise 因為沒有被 RES 挖掉的 bin 而不填。強度用 `aec_set_preset()`（§3.2），沒有獨立的 `AecRes`、也沒有 `res_preset` |
| NR | 用 | `--no-nr`：不建 `AnrDenoiser`（state 不進 pool），總增益只剩 `G_res`，`anr_set_mode()` 沒有對象 |
| NR 強度 | `ANR_MODE_BALANCED`（= 2） | `--nr-preset mild\|moderate\|balanced\|aggressive`：交給 `anr_config_for_mode_grid(sr, fft_size, mode)`，四級都可用；最小值追蹤視窗 `L` 為函式庫預設，不覆寫 |
| `use_r2` | 用 | `--legacy-amin`：`se_anr_stage_gain(..., use_r2 = 0)`，noise-only NR（noise floor 不摺入 R²）。只用於比對，新整合請保持用 R² |
| CNG | 開，與 `aec_cfg.enable_cng`（AEC preset 自己的值）取 AND | `--no-cng`：不呼叫 `aec_cng_fill()`。位準再乘該 bin 的 NR gain、下限 −10 dB（`SE_CNG_NR_GAIN_FLOOR`；NR 看不到 comfort noise，在此補套）；沒有 RES gain 時也不填 |
| 合成 | 用 | 範例是 `ac_istft_push()` 之後直接輸出；DFN2 之類後接的 stage 自己合成時就不建 `AcIstft`（見 `mono_aec_dfn_res`） |

`--aec-only` 不建任何後級、用 `aec_process()`（§1.2）。

### 4.3 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`frame = fft_size`、`n_freqs = fft_size / 2 + 1`。
合法組合只有這四組：

| `sample_rate` | `fft_size` | `hop` | `frame` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---:|---|
| 8000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 512 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 1024 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop 寫死。** 16 kHz 有兩組合法 grid，寫死任一組對另一組就是錯的。
一律用 `se_derive()` 輸出的 `grid.hop_size` / `grid.n_freqs` 查。

### 4.4 哪些欄位會改變記憶體用量

| 欄位 | 影響 pool 大小嗎 |
|---|---|
| `sample_rate`、`fft_size` | 會（改 grid、改 AEC filter 長度） |
| `filter_length` | 會（非 0 時覆寫 AEC filter 長度／n_partitions） |
| `delay_mode` | 會（`MATCHED` 建 matched-filter estimator + 大 ring；`FIXED` 只建剛好夠用的小 ring；`EXTERNAL_ALIGNED` 兩者都不建，見 §4.5a） |
| `delay_num_filters` | 僅 `MATCHED` 有效：每少一個固定省 5,728 B（四格點皆同，與 `lib/aec` 一致） |
| `fixed_delay_samples` | 僅 `FIXED` 有效：ring 大小 = `ALIGN16((fixed_delay_samples+hop)×4)` B |
| `aec_preset`、`nr_mode` | 實測不變 |
| 要不要建降噪器（`--no-nr`） | 會（不建就省下整份 MMSE-LSA state） |
| 要不要建合成 | 會（不建就省下 `AcIstft`） |
| RES gain、CNG、`use_r2` | 不變（RES 是 `Aec` 內的 gain，不另配記憶體） |

### 4.4b 診斷用的分項

沒有「後級總大小」的單一查詢：你的 pool 大小就是各個 `mem_size` 的回傳值各自 `ALIGN16` 後相加，再加上你自己的
增益／頻譜暫存。`static_main.c --print-mem-size` 印的表格（`AEC`、`Synthesis`、`NR (MMSE-LSA)`、
`Gain/spectrum buffers`、`Total`）就是這個加總，並且真的在那塊 pool 上跑一次每個物件的 init。
這只是給人看的表格；**真正要配置多少，永遠以各 `mem_size` 的回傳值為準**，加上各塊的對齊。

### 4.5 實測記憶體（僅供量級參考，務必自己重查）

以下是**本次 checkout、`BACKEND=ne10`、`pipelines/Makefile` 預設選項**下，以
`aec_nr_pipeline_static --print-mem-size` 量到的值（delay 列用 `aec_get_mem_size()` 加上同一組後級大小直接呼叫）。
換編譯選項、更新 submodule 都會變，請自己重查。

增益／頻譜暫存合計於「暫存」欄（三個 `n_freqs` float 向量與一個 `n_freqs` 複數向量，各自 `ALIGN16`）。

| Config | 總計 | AEC（`aec_get_mem_size`） | 合成 `AcIstft` | `nr_bytes` | 暫存 |
|---|---:|---:|---:|---:|---:|
| 8000，預設 | 310,160 | 267,408 | 11,344 | 28,784 | 2,624 |
| 16000，預設（256/128） | 423,728 | 364,416 | 11,344 | 45,344 | 2,624 |
| 16000，`fft_size=512` | 575,056 | 491,600 | 21,840 | 56,432 | 5,184 |
| 48000，預設（1024/512） | 1,354,256 | 1,156,544 | 42,832 | 144,576 | 10,304 |
| 16000，`--no-nr`（不建降噪器） | 378,384 | 364,416 | 11,344 | 0 | 2,624 |
| 16000，`--aec-only`（不建後級） | 364,416 | 364,416 | – | – | – |

總計 = `ALIGN16(AEC)` + `ALIGN16(nr_bytes)` + `ALIGN16(合成)` + 暫存（各塊對齊後相加）。

#### 4.5a delay_mode 對 16000/預設 grid 的影響（`MATCHED` n=5 為 baseline）

| `delay_mode` | AEC 加後級合計 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 423,728 | — |
| `MATCHED` n=1 | 400,816 | −22,912 |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 265,632 | −158,096 |
| `EXTERNAL_ALIGNED` | 258,720 | −165,008 |

這四列的差額全部落在 AEC 那一塊（`delay_mode`/`delay_num_filters` 不影響
後級），數字與 `lib/aec` 自己的 `aec_get_mem_size()` 一致
（見 `lib/aec/docs/c_user_manual_zh_TW.md` §4）。**沒有 CLI 旗標**能測試
非預設 delay mode——上表用直接呼叫
`aec_get_mem_size()` 加上同一組後級大小（59,312 B）量得。

---

## 5. 錯誤處理

這個 codebase 混用了數種回傳慣例。下表是**每個 function 實際的**慣例。

### 5.1 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `se_derive()` | `0`，輸出填妥 | `-1` | 輸出未定義；無任何 log |
| `aec_get_mem_size()` | pool bytes | `0`（config 無效） | 與 4ch 的 `get_mem_requirements()`（0 = 成功）相反 |
| `anr_get_mem_size()` / `ac_istft_get_mem_size()` | pool bytes | `0`（config 無效） | 同上 |
| `aec_init()` / `anr_init()` / `ac_istft_init()` | 非 NULL handle | `NULL` | pool 未對齊、太小、config 無效；失敗後 pool 仍可重新用於正確的 init |
| `aec_process_context()` / `aec_process()` | `void` | – | 不回傳錯誤；異常由後面的 `se_frame_from_aec()`／`se_anr_stage_gain()` 回傳值看出 |
| `se_frame_from_aec()` | `0` | `-1` | NULL 引數，或 lane 沒有 seam（`enable_res=0`、`return_res_context=1`） |
| `se_anr_stage_gain()` | 降噪器擁有的 `n_freqs` 個增益 | `NULL` | NULL 引數、`frame->e` 為 NULL，或降噪器拒絕該 hop |
| `ac_istft_push()` | 寫出的樣本數（`hop`；`AC_OLA_ENV_NORM` 的第一個 frame 為 `0`） | – | 不檢查輸出是否有限，要另外查 |
| `se_hop_is_finite()` | `1` | `0` | 你在採用輸出之前要做的檢查 |
| `aec_reset()` / `anr_reset()` / `ac_istft_reset()` | `void` | 不會失敗 | 要一起呼叫，並把 comfort-noise RNG 設回 `SE_CNG_RNG_SEED` |
| `aec_destroy()` / `anr_destroy()` | `void` | 不會失敗 | NULL 安全 |
| `aec_set_preset()` / `anr_set_mode()` | `0` | `-1` | `-1` 時什麼都不寫（§3.2） |

> **注意慣例不一致**：`aec_get_mem_size()` / `anr_get_mem_size()` / `ac_istft_get_mem_size()` /
> `fft_get_mem_size()` 是 **回傳 0 代表失敗**；4ch 單元的 `audio_pipeline_4ch_get_mem_requirements()` 是
> **0 = 成功 / −1 = 失敗**。判斷式別混用。

### 5.2 init 的拒絕清單

`aec_init()`：`mem == NULL`、`mem` 未 16-byte 對齊、`mem_size` 小於 `aec_get_mem_size()`、config 無效、子模組 init 失敗 → `NULL`。
`anr_init()`、`ac_istft_init()`：同樣的 NULL／未對齊／過小／config 無效規則（出界 enum、不合法格點）→ `NULL`。
這些 init 都**不會**動到 pool 以外的記憶體，被拒之後同一塊 pool 仍可用在正確的 init
（`example_board_adapter.c` 的 `run_negative_demonstrations()` 展示並檢查這一點）。

### 5.3 `destroy()` 的語意

`aec_destroy()` 與 `anr_destroy()` 都**不釋放**你的 pool；`aec_destroy()` NULL 安全、可重複呼叫（只放開實例內部的參照）；`anr_destroy()` 對 pool 實例立即返回，只有 `anr_create()` 的 heap 實例才釋放記憶體。`AcIstft` 沒有 destroy。
`posix_memalign` 之類由你配置的 pool，由你在 destroy 之後釋放。

### 5.4 診斷輸出去向

* mono 範例的每 hop 串接（`se_anr_stage.h` 與 `mono_form()`）不輸出任何東西。lib/aec／lib/nr／audiolib 的 init 期診斷寫到 **stderr**，
  以 `NO_STDIO=1` 建置時編成 no-op，**回傳值語意完全不變**。
* 每 hop 的 `aec_process_context()`、`se_anr_stage_gain()`、`ac_istft_push()` **在任何 build 下都不會輸出任何東西**。

---

## 6. 出貨內容 vs 範例

`pipelines/` 目錄不是自解釋的。下表區分「函式庫」與「參考程式」。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `lib/aec`、`lib/nr`、audiolib | `libaec.a`、`libanr.a`、`libaudio_common.a` | 線性 AEC（含 AEC3 殘留 gain context）、降噪、FFT／STFT／合成 |
| `pipelines/common/se_config.c` / `.h` | `se_config.o`（4ch 單元的 archive 也含它） | 格點解析與 `AecConfig` 推導（`se_derive()`）；各應用的規則（rate 集合、delay 檢查、lane 對齊、seam context）以 `SeRules` 傳入 |
| `pipelines/common/se_stage.h` | header-only | `SeSpecFrame`、`se_res_stage_gain()`、`se_frame_from_aec()`、CNG 常數 |

### 6.2 範例碼與參考執行檔（**沒有任何一個會進產品 archive**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/samples/se_anr_stage.h` | header-only 的 NR 級 gain 呼叫（`se_anr_stage_gain()`） | 把 AEC 的殘留回音功率摺進降噪器的 noise floor 並回傳 NR 增益；可整份複製改寫，沒有 ABI |
| `pipelines/mono_aec_nr_res/main.c` | heap 路徑 CLI（WAV in / WAV out） | 示範 `Aec` + `AnrDenoiser` + `AcIstft` 各自初始化（`aec`、`nr` 與合成/暫存各一塊 heap）與每 hop 串接（`mono_form()`）；離線比對用 |
| `pipelines/mono_aec_nr_res/static_main.c` | caller-owned pool 路徑 CLI | 示範 query → allocate（AEC、降噪器、合成、暫存在**一塊**對齊 pool 內依序切）→ init → process → destroy → release；另提供 `--print-mem-size` |
| `pipelines/mono_aec_nr_res/example_board_adapter.c` | 板端 adapter 的**主機模擬** | 示範完整呼叫序列（AEC、降噪器、合成、暫存在一塊 arena 內）、pool 大小加總與拒絕路徑。裡面每個 `board_mem_*` 都是主機上的假實作，不是平台程式碼。標了 `// BOARD:` 的地方就是你要換成真實平台程式碼的位置 |

四麥克風的 library 與參考程式是**另外兩份**手冊的主題：
`docs/integration_4ch_core_zh_TW.md`、`docs/integration_4ch_spatial_zh_TW.md`。

### 6.3 測試（不出貨）

`pipelines/mono_aec_nr_res/tests/test_mono_chain.c`（照 `static_main.c` 的展開順序建 Rig，並把兩支 mono 程式當黑箱跑，與手寫 oracle `conv_chain_mono.h` 逐位元比對）、
`pipelines/mono_aec_nr_res/tests/test_no_stdio_stack.c`。

---

## 7. 整合檢查清單

建置與連結：

- [ ] `libaec.a` / `libanr.a` / `libaudio_common.a` 與你的 `se_config.o` 是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 §1.5
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用各 Makefile 的 `print-lib-path` 查的，不是硬寫的
- [ ] 你的 TU 用 `gnu99`（或以上的 GNU 方言）

初始化：

- [ ] 每次 init 之前都重新呼叫 `aec_get_mem_size()`、`anr_get_mem_size()`、`ac_istft_get_mem_size()`，沒有快取大小
- [ ] 每一塊的大小都檢查過 `!= 0`
- [ ] pool 16-byte 對齊，大小 ≥ 各塊 `ALIGN16` 後相加（AEC、降噪器、合成、暫存，加上你自己的 stage）
- [ ] `aec_init()`、`anr_init()`、`ac_istft_init()` 回 `NULL` 時你的程式會停下來，不會繼續跑

執行期：

- [ ] `mic` / `ref` / `out` 三個 buffer 都剛好 `grid.hop_size` 個 float
- [ ] hop 大小是查來的，不是寫死的
- [ ] `se_frame_from_aec()`、`g_res_ctx`、`se_anr_stage_gain()` 的回傳值，以及合成輸出的有限性（`se_hop_is_finite()`）都有檢查
- [ ] 任何一步被拒，或換回聲路徑／換串流時，`aec_reset()`、`anr_reset()`、`ac_istft_reset()` 與 comfort-noise RNG 的重設**一起**做
- [ ] 同一個 instance 沒有被多 thread 同時 process
- [ ] 降噪器的 reset／destroy 與 `Aec` 同進退，沒有只重設其中一個

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] pool 在 handle 存活期間沒有被搬動、覆寫或共用

板端額外：

- [ ] 若 image 不能連 stdio，函式庫是以 `NO_STDIO=1` 建的
- [ ] 已確認 pool 不需要預先清零（dirty pool 可接受）
- [ ] bring-up log 有保留 init 的拒絕訊息（NO_STDIO build 就沒有了，此時只能靠回傳值）

---

## 8. 版本與相容性

mono 沒有 descriptor，也沒有 layout version：`Aec`、`AnrDenoiser`、`AecRes`、`AcIstft` 只提供大小查詢，
`se_anr_stage.h` 與各範例程式是範例碼。需要 descriptor 閘的出貨單元只有 4ch 的 `AudioPipeline4Ch`
（`AudioPipeline4ChMemReq`，`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`）與 DFN2 的 stage／bridge，
見 [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md) 與
[`pipeline_board_integration.md`](pipeline_board_integration.md)。

給整合者的兩條規則：

1. **大小永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **升級 library 之後一定要重跑一次大小查詢**，並確認你的 pool 預算仍然足夠。
