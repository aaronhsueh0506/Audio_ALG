導覽：[索引](integration_mono_zh_TW.md) ｜ [上一部分](integration_mono_zh_TW.md) ｜ [下一部分](integration_mono_zh_TW_02_lifecycle_config_errors.md) <!-- split-nav -->
## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖（mono：`Aec` 加逐個初始化的函式庫）

生命線：應用（你的 `main`）、`Aec`（lib/aec）、`AnrDenoiser`（lib/nr）、`AcIstft`（audiolib）。
箭頭上的名稱都是真實函式；沒有共用的串接 header，每 hop 的串接寫在各範例自己的 `main`
（`mono_aec_nr_res/main.c` 的 `derive_lane()`、`mono_nr_gain()`、`mono_form()`），可整份複製改寫。

```
應用                       Aec                 AnrDenoiser            AcIstft
 │                          │                       │                     │
 │══ 初始化（一次；每次 init 前都重查大小，不快取）═══════════════════════════
 │ aec_resolve_signal_grid(sr,fft,&grid)    → 1 = 有這組格點；0 = 沒有（fft 取 aec_default_fft_size(sr) 或你指定的值）
 │ aec_config_from_level(&aec_cfg,level,sr); 再設 fft_size、delay_*、enable_res = 0、return_res_context = 1（§4.1）
 │ aec_get_mem_size(&aec_cfg) ─────►│            → bytes（0 = 無效）
 │ anr_config_for_level_grid(sr,fft,level); anr_get_mem_size(&nr_cfg) ──►│   → bytes（0 = 無效）
 │ ac_stft_config_default(&sc,fft); ac_istft_get_mem_size(&sc) ───────────────►│   → bytes（0 = 無效）
 │ 配置一塊 16-byte 對齊的 pool = Σ ALIGN16(各塊) + 增益/頻譜暫存
 │ aec_init(mem,bytes,&aec_cfg) ────►│  anr_init(mem,bytes,&nr_cfg) ─►│  ac_istft_init(mem,bytes,&sc) ─►│
 │                                  （任一個回 NULL：停止，不要繼續跑）
 │══ 每個 hop（mic[hop]、ref[hop]；hop = grid.hop_size）════════════════════
 │ aec_process_context(aec,mic,ref) ►│            （void；線性 AEC + AEC 自己的 RES 增益）
 │ aec_get_res_context(aec,&ctx) ◄──│            ctx{error_spec,near_spec,far_spec,r2,comfort_noise,res_gain,...}（void）
 │ g_res = ctx.res_gain               （ctx.error_spec 或 res_gain 為 NULL = 該 hop 被拒；關 RES 時用 unity 向量）
 │ sk_scale_f32(extra,ctx.r2,PSD_SCALE_INV,n)   （use_r2 且 ctx.usable_linear 時：回聲功率摺進 noise floor）
 │ anr_process_gain(nr,ctx.error_spec,extra,NULL) ──────►│ 0 / -1；anr_get_gain(nr,NULL) → G_nr（NULL = 被拒）
 │ sk_min_f32(total,G_nr,g_res,n); sk_capply_gain_f32(spec,ctx.error_spec,total,n)
 │ aec_cng_fill(&rng,spec,ctx.comfort_noise,g_res,G_nr,CNG_NR_GAIN_FLOOR,n)    （CNG 開、有 RES 增益時）
 │ ac_istft_push(istft,spec,out) ────────────────────────────────────────►│ 寫出 hop 個樣本
 │ skn_all_finite_f32(out,hop)        （選用；板端 adapter 與 §2 範例做，main.c 不做）
 │══ 任一步被拒（或你的輸出檢查失敗）══════════════════════════════════════════
 │ aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = CNG_RNG_SEED     （四件一起做）
 │══ 強度（兩個 hop 之間呼叫，非 thread-safe）═══════════════════════════════
 │ aec_set_level(aec,level,ramp_ms) ►│          anr_set_level(nr,level) ──►│
 │══ 結束 ═══════════════════════════════════════════════════════════════
 │ anr_destroy(nr); aec_destroy(aec); 釋放 pool    （AcIstft 沒有 destroy；先 destroy 再釋放 pool）
 │
 │ --aec-only：只建 Aec，每 hop 呼叫 aec_process_context() 後把 ctx.linear_hop 複製到 out（不經 NR、合成與增益暫存）
```

延遲泳道（橫軸 = hop；`ac_istft_push()` 的 identity 鏈延遲 = 1 hop，見 `audio_stft.h` 檔頭）：

```
hop 序號          t        t+1       t+2
輸入 mic/ref      m[t]     m[t+1]    m[t+2]
Aec + RES + NR    處理 m[t] 處理 m[t+1] 處理 m[t+2]     （同一次呼叫內完成，線性路徑本身延遲 0）
輸出 out          y[t-1]   y[t]      y[t+1]            （合成 OLA 多 1 hop；--aec-only 沒有這 1 hop）
```

1 hop = `grid.hop_size` 個樣本：16 kHz/256 為 128 個樣本（8 ms）、16 kHz/512 為 256（16 ms）、48 kHz 為 512（約 10.7 ms）；8 kHz/256 為 128（16 ms）。數字由 [§4.3](integration_mono_zh_TW_02_lifecycle_config_errors.md#43-grid由-sample_rate--fft_size-唯一決定) 的格點推得。

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開）**

| 階段 | 函式（標頭） | 備註 |
|---|---|---|
| 格點與 `AecConfig` | `aec_resolve_signal_grid()`、`aec_default_fft_size()`、`aec_config_from_level()`（`aec.h`） | `aec_resolve_signal_grid()` 回 `1` = 有這組格點、`0` = 沒有；其餘接縫欄位由你設（[§4.1](integration_mono_zh_TW_02_lifecycle_config_errors.md#41-aecconfig-的接縫欄位範例的-derive_lane)） |
| 大小查詢 | `aec_get_mem_size()`（`aec.h`）、`anr_get_mem_size()`（`anr.h`）、`ac_istft_get_mem_size()`（`audio_stft.h`） | 回 `0` = config 無效 |
| 建立 | `aec_init()`、`anr_init()`、`ac_istft_init()`；桌面也可 `aec_create()`、`anr_create()` | pool 路徑回 `NULL` = 失敗 |
| 每 hop | `aec_process_context()`、`aec_get_res_context()`（`aec.h`）、`anr_process_gain()`、`anr_get_gain()`（`anr.h`）、`sk_scale_f32()`、`sk_min_f32()`、`sk_capply_gain_f32()`（`utility/simd_kernels.h`）、`aec_cng_fill()`（`aec_res.h`）、`ac_istft_push()`；選用 `skn_all_finite_f32()`（`utility/simd_kernel_nn.h`） | 順序見 [§0.1](#01-流程時序圖monoaec-加逐個初始化的函式庫) |
| 只要線性 AEC | `aec_process_context()`、`aec_get_res_context()`，輸出取 `ctx.linear_hop` | 範例的 `--aec-only` 這樣做；`aec_process()` 是函式庫的時域便利入口，lane 組態下回傳同一個 hop |
| 強度 | `aec_set_level()`、`anr_set_level()`（需要整組覆寫時 `anr_reconfigure()`） | [§3.2](integration_mono_zh_TW_02_lifecycle_config_errors.md#32-執行期強度控制) |
| 重置／收尾 | `aec_reset()`、`anr_reset()`、`ac_istft_reset()`、`aec_destroy()`、`anr_destroy()` | [§3](integration_mono_zh_TW_02_lifecycle_config_errors.md#3-lifecycle)、[§5](integration_mono_zh_TW_02_lifecycle_config_errors.md#5-錯誤處理) |
| 診斷（唯讀） | `aec_debug_status()`、`anr_debug_status()`、`aec_get_last_timing()` | [§3.1](integration_mono_zh_TW_02_lifecycle_config_errors.md#31-執行期唯讀存取)、[§3.1b](integration_mono_zh_TW_02_lifecycle_config_errors.md#31b-逐階段耗時診斷用) |

**內部函式（不要依賴，隨版本改變）**：`aec.h` 以外的 AEC 內部標頭（`pbfdkf.h`、`delay_aec.h`、`aec_post.h`、
`suppression_gain.h`、`aec_config_build.h`、`aec_balanced_config.h`）；NR 的 `noise_estimator.h`、`spp_estimator.h`、
`anr_internal.h`，以及 `anr_apply_*_experiment()`；audiolib 的 `utility/` 內核（除上表列出的 `sk_*`、`ALIGN16` 外）。
`struct Aec` 目前整個攤在 `aec.h` 裡，只是因為 C 需要完整型別；欄位是內部，只能經由函式存取。

### 0.3 參數面：外部與內部

**外部（建議讓第三方整合者調）**

| 名稱 | 位置 | 範圍／預設 | 何時設定 | 作用 |
|---|---|---|---|---|
| `sample_rate`、`fft_size` | `aec_resolve_signal_grid()`、`AecConfig.fft_size`（`aec.h`） | 8000／16000／48000；`fft_size` 取 `aec_default_fft_size(sr)`（0 = 不支援的 rate），16 kHz 另可 256／512 | init 前，之後不可變 | 決定格點（[§4.3](integration_mono_zh_TW_02_lifecycle_config_errors.md#43-grid由-sample_rate--fft_size-唯一決定)）；改了要重查所有大小 |
| `filter_length` | `AecConfig`（`aec.h:72`） | `aec_config_from_level()` 依 rate 填預設；有效範圍 `[1, 4096]`，範圍外 `aec_get_mem_size()` 回 `0` | init 前 | 回聲路徑長度（taps）；越長記憶體與算量越大 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `AecConfig` | `MATCHED`（預設，bank 1..5，預設 5）／`FIXED`（已知延遲）／`EXTERNAL_ALIGNED`（上游已對齊） | init 前，之後不可變 | 對齊方式；決定 pool 版面（[§4.4](integration_mono_zh_TW_02_lifecycle_config_errors.md#44-哪些欄位會改變記憶體用量)、[§4.5a](integration_mono_zh_TW_02_lifecycle_config_errors.md#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline)） |
| `aec_config_from_level()` 的 `level` 引數 | `aec.h` | `MILD`／`BALANCED`（預設）／`AGGRESSIVE`／`NO_FLOOR`；enum 外時退回 balanced | init 前；執行期用下一列 | 回聲抑制 vs 近端保留的單一取捨軸 |
| `aec_set_level(aec, level, ramp_ms)` | `aec.h` | `ramp_ms` 屬於 `[0, 60000]`，`0` = 下一 hop 生效 | 兩個 hop 之間 | 執行期換 AEC 強度；回 `-1` 時什麼都不寫 |
| NR 強度 | `AnrLevel`（`anr_types.h`）：`anr_config_for_level_grid()`、`anr_set_level()` | `MILD`／`MODERATE`／`BALANCED`（預設）／`AGGRESSIVE` | init 前或兩個 hop 之間 | 降噪深度的單一入口（一次調 `g_min_db`、`q`、`xi_min_db`、`noise_over_subtraction`） |
| 後級開關（範例程式的欄位，不是函式庫 config） | `PostPlan`（`mono_aec_nr_res/main.c`）；CLI `--no-res`、`--no-nr`、`--no-cng`、`--legacy-amin` | 各 0／1 | init 前 | RES 增益用不用、要不要建降噪器、CNG、NR 是否折入 R²（[§4.2](integration_mono_zh_TW_02_lifecycle_config_errors.md#42-後級選項範例程式的欄位與-cli-旗標)） |

進階外部（選擇性；改了以後自己重跑三個 grid 的驗證）：`AecConfig` 的 `enable_cng`（`aec.h:77`，下游已有舒適噪音時關）、
`enable_highpass`／`highpass_cutoff_hz`（`aec.h:80,115`，上游已有高通時關；開啟時截止頻率須 `> 0` 且低於 `0.45 × sample_rate`）、
`enable_shadow`（`aec.h:82`，關掉省記憶體但路徑突變後恢復變慢）、`delay_est_period_s`（`aec.h:109`，預設 0 = 每 hop 全速分析；非 0 會省算力，但可能改變選到的 lag）。這些在 `aec_config_from_level()` 之後、`aec_get_mem_size()` 之前直接改 `aec_cfg`。

**內部（存在、可調，但不建議開放）**

| 家族 | 位置 | 為什麼不開放 |
|---|---|---|
| 範例 `derive_lane()` 固定的接縫：`enable_res = 0`、`return_res_context = 1`、`spatial_linear_context`（保持預設 0 = 完整 context，`ctx.res_gain` 才有值） | `aec.h:83,88,92`；`mono_aec_nr_res/main.c` 的 `derive_lane()` | 改了就沒有 `ctx.res_gain`（或時域輸出改道），串接整個失效 |
| 以 10 ms 格點撰寫、建立時 retime 的時間欄位：`warmup_frames`、`epc_hangover`、`ne_recent_hold`、`filter_misadjustment_*_frames`、`shadow_err_alpha` | `aec.h:94-146` | 單位是「N × 10 ms」，不是 hop 數；填錯單位會靜默失效 |
| 適應步進與模型不確定度：`mu`、`delta`、`kalman_q_high`、`kalman_q_low`、`shadow_mu_*`、`epc_mu_floor` | `aec.h:74-75,96-103,117-118` | 決定收斂與發散邊界；驗證器範圍很寬，範圍內多數值不收斂 |
| 偵測門檻：`epc_*`、`ne_recent_*`、`saturation_threshold`、`filter_misadjustment_scale_*` | `aec.h:101-102,116,133-146` | 與其他偵測器聯合調過；`scale_min > scale_max` 不會被驗證器擋 |
| 延遲獲取策略：`delay_acquire_*`、`delay_est_init_s`、`max_delay_ms`、`delay_buffer_ms`（`delay_est_period_s` 是上面的進階外部） | `aec.h:104-124` | 影響冷啟動；`max_delay_ms`／`delay_buffer_ms` 只決定搜尋環大小，不是可擷取上限（上限由 `delay_num_filters` 決定） |
| Double-talk 政策與 floor：`dt_aware_*`、`min_gain_floor_*_db`、`min_gain_split_floor_enabled` | `aec.h:129-139,165` | 三個 floor 欄位已由 `aec_level`／`aec_set_level()` 成組設定；單獨改 dB 值會得到沒有物理意義的 floor |
| 預設關閉的實驗：`live_observer_*`、`cng_n2_windowed_psd` | `aec.h:78,157-162` | 未通過 default-on 門檻；observer 開啟會增加 pool 與算量 |
| 衍生值：`n_partitions`、`enable_delay_est`（deprecated） | `aec.h:73,79` | 由 `filter_length` 與 hop 推導，手填只會造成不一致 |
| `SuppressionGainConfig`、`AecPostConfig`、`AecStateConfig`、`aec_balanced_config.h` 的常數 | AEC 內部標頭 | 編譯期依 `(sample_rate, fft_size)` 查表，runtime 改不了，也不該提供接口 |
| NR 的追蹤器與平滑欄位：`alpha_xi`、`alpha_s`、`alpha_d`、`alpha_p`、`alpha_attack`、`alpha_decay`、`delta_db`、`scene_change_*`、`speech_protect_*`、`speech_aware_*`、`noise_gate_*`、`makeup_*`、`cross_band_*` | `anr_types.h:66-120` | 聯合調出的操作點，四個 `AnrLevel` 共用；`alpha_g` 在增益路徑上沒有作用（`anr_types.h:117`） |
| NR 的 `L`、`num_init_frames` | `anr_types.h:104,106` | 決定 pool 版面，`anr_reconfigure()` 與 `anr_set_level()` 不能改它們；範例用函式庫預設 |

---

## 1. 你連結什麼、你 `#include` 什麼

### 1.1 構成：每個函式庫各自初始化，每個 hop 的串接由你寫

mono 沒有包起來的模組、沒有 pipeline 函式庫，也沒有共用的串接 header。處理鏈由下列獨立的函式庫物件組成，
**每一個都有自己的 `mem_size` 與 `init`**，初始化與每個 hop 的串接都是整合者的事：

```
mic hop + ref hop ──► Aec (aec_process_context) ──► aec_get_res_context ──► AecResContext
                                                                              │
        ┌─────────────────────────────────────────────────────────────────────┘
        ▼   G_res = ctx.res_gain（AEC 自己的殘留 gain；不要 RES 時用 unity）
   G_nr = anr_process_gain(...) + anr_get_gain(...)  ─►  sk_min_f32(G_nr, G_res)  ─►  sk_capply_gain_f32(E)
        ─►  aec_cng_fill（CNG 開著、且有 RES gain 時）  ─►  ac_istft_push  ──► out hop
```

mono 的 RES 是明確的例外：它在 `Aec` 的呼叫裡（`aec_process_context()` 算出殘留 gain，`aec_get_res_context()` 以
`ctx.res_gain` 交出，強度用 `aec_set_level()`），沒有獨立的 `AecRes` 物件；範例只決定要不要套用這個 gain。
純 AEC 的範例（`--aec-only`）不經過合成：每 hop 一樣呼叫 `aec_process_context()`，輸出取 `ctx.linear_hop`。反向 FFT 與 overlap-add 一律由應用呼叫獨立的 `ac_istft_push()`，不放在演算法函式庫裡。

| 物件 | 來源 | 大小查詢 | 初始化 |
|---|---|---|---|
| `Aec`（線性 AEC，含 AEC3 殘留 gain 的 context；RES gain 就是它自己的 `ctx.res_gain`） | `lib/aec`（`libaec.a`） | `aec_get_mem_size(&aec_cfg)` | `aec_init(mem, bytes, &aec_cfg)` |
| `AnrDenoiser`（可選） | `lib/nr`（`libanr.a`） | `anr_get_mem_size(&nr_cfg)`（`nr_cfg = anr_config_for_level_grid(sr, fft_size, level)`，最小值追蹤視窗 `L` 用函式庫預設） | `anr_init()` |
| `AcIstft`（合成） | audiolib（`libaudio_common.a`） | `ac_istft_get_mem_size(&synth_cfg)`（`ac_stft_config_default(&synth_cfg, fft_size)`） | `ac_istft_init()` |
| 增益／頻譜暫存（unity、總增益、NR 額外噪聲、輸出頻譜） | 你自己的記憶體 | `3 × ALIGN16(n_freqs × sizeof(float)) + ALIGN16(n_freqs × sizeof(Complex))`（`main.c` 的 `post_plan()`；不要 RES／NR 時的 unity 向量用得到，§2 的最小範例只要 2 個 float 向量） | unity 向量要自己填 1 |
| 格點與 `AecConfig` | `lib/aec`（`libaec.a`） | – | `aec_resolve_signal_grid()` + `aec_config_from_level()`，接縫欄位自己設（[§4.1](integration_mono_zh_TW_02_lifecycle_config_errors.md#41-aecconfig-的接縫欄位範例的-derive_lane)） |

上面幾項怎麼放進 pool（heap 路徑各自配置，caller-pool 路徑依序切同一塊）與每個 hop 怎麼串，是範例程式
（`mono_aec_nr_res/main.c`、`example_board_adapter.c`）各自寫出來的，沒有 ABI、沒有 layout version、沒有 descriptor、
不進任何 archive，也沒有共用的範例 header，可整份複製改寫。

每個 hop 的串接（`main.c` 的 `mono_form()` 與 `mono_nr_gain()`；兩支執行檔共用）：

1. `res = ctx.res_gain`（不要 RES 時用全 1 的 unity 向量；`ctx.error_spec` 為 NULL 表示沒有 context、`res_gain` 為 NULL 表示 lane 沒有 gain，都視為拒絕）。
2. NR 增益：`use_r2` 開著、`ctx.r2` 非 NULL 且 `ctx.usable_linear` 時，先 `sk_scale_f32(extra_noise, ctx.r2, PSD_SCALE_INV, n_freqs)`；再 `anr_process_gain(nr, ctx.error_spec, extra, NULL)`（非 0 = 降噪器拒絕該 hop）與 `anr_get_gain(nr, NULL)`。沒有降噪器時 `nr` 用 unity 向量。
3. `sk_min_f32(total, nr, res, n_freqs)`，再 `sk_capply_gain_f32(spec, ctx.error_spec, total, n_freqs)`。
4. CNG 開著（`aec_cfg.enable_cng` 且沒有 `--no-cng`）、有 RES gain、且 `ctx.comfort_noise` 非 NULL 時 `aec_cng_fill(&rng, spec, ctx.comfort_noise, res, nr, CNG_NR_GAIN_FLOOR, n_freqs)`。
5. `ac_istft_push(istft, spec, out)` 產生這個 hop 的時域輸出。

`CNG_NR_GAIN_FLOOR`（0.31622777f，即 −10 dB）、`CNG_RNG_SEED`（0x9e3779b9u）與 `PSD_SCALE_INV`（1/32768²，AEC 的功率在 int16² 尺度、降噪器在音訊尺度）是範例檔頭自己的 `#define`，不是函式庫常數。

### 1.2 `--aec-only`

只要線性 AEC 的輸出，就不建後級：`aec_process_context(aec, mic, ref)` 之後 `aec_get_res_context(aec, &ctx)`，
把 `ctx.linear_hop`（`ctx.hop_size` 個 float，AEC 自己的時域 hop）複製到 `out`。
不需要降噪器、合成與增益暫存（見 `aec_nr_pipeline_static --print-mem-size --aec-only`：pool 只有 `Aec` 那一塊）。
`aec_process()` 與 `aec_process_context()` 推進相同的狀態；在範例的 lane 組態（`enable_res=0`、`return_res_context=1`）下
`aec_process()` 回傳的就是這個 hop，測試用它當獨立的對照。

### 1.3 Header closure

```c
#include "aec.h"                       /* Aec, AecConfig, AecResContext, aec_resolve_signal_grid(), aec_process_context() */
#include "aec_res.h"                   /* aec_cng_fill()                                        */
#include "anr.h"                       /* AnrDenoiser, anr_config_for_level_grid(), anr_init()  */
#include "audio_stft.h"                /* AcIstft, ac_istft_init(), ac_istft_push()           */
#include "utility/simd_kernels.h"      /* sk_scale_f32(), sk_min_f32(), sk_capply_gain_f32()  */
#include "utility/mem_align.h"         /* ALIGN16                                             */
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32()，選用的輸出檢查                    */
```

`Complex` 定義在 `<audiolib>/common/include/fft_wrapper.h`，已透過 `aec.h` 帶進來。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I lib/aec/c_impl/include
-I lib/nr/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
```

`pipelines/Makefile` 自己建置範例時的 `-I` 清單比這個長（多了 `lib/aec/c_impl/example`，給 CLI 的 WAV I/O 用，以及 AIAEC／AINR 的目錄），
consumer 不需要。語言標準：以 `-std=gnu99` 建置；
NE10 後端的 header 需要 GNU 擴充，所以你的 TU 也請用 `gnu99`（或以上的 GNU 方言）。

### 1.5 Archive link order

link 順序有意義（單向相依，由上而下）：

```
你的 .o
libaec.a
libanr.a
libaudio_common.a
-lm
```

* mono 沒有 pipelines 提供的 TU 或 object，也沒有 pipelines 的 header：你的 `.o` 只連三個 archive。
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
#include "aec_res.h"                   /* aec_cng_fill() */
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32() */

/* Comfort noise follows the denoiser gain of its bin but never below this amplitude (-10 dB). */
#define CNG_NR_GAIN_FLOOR 0.31622777f
#define CNG_RNG_SEED 0x9e3779b9u                           /* start state of aec_cng_fill()'s generator */
#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))       /* AEC powers are int16^2, the denoiser's audio scale */

int main(void) {
    const int sample_rate = 16000;
    AecSignalGrid grid;
    AecConfig aec_cfg;
    if (!aec_resolve_signal_grid(sample_rate, aec_default_fft_size(sample_rate), &grid))
        return 1;                                                  /* 0 = no grid for this pair */
    aec_config_from_level(&aec_cfg, AEC_LEVEL_BALANCED, sample_rate);
    aec_cfg.fft_size = grid.fft_size;
    aec_cfg.delay_mode = AEC_DELAY_MATCHED;
    aec_cfg.delay_num_filters = DA_NUM_FILTERS;
    aec_cfg.fixed_delay_samples = -1;
    aec_cfg.enable_delay_est = 1;
    aec_cfg.enable_res = 0;                                        /* the lane keeps its gain in the context ... */
    aec_cfg.return_res_context = 1;                                /* ... and aec_get_res_context() exposes it */
    const int n = grid.n_freqs;

    /* every library sizes and initialises itself */
    AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, grid.fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, grid.fft_size);
    size_t aec_bytes = aec_get_mem_size(&aec_cfg);                    /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);                      /* 0 = invalid */
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);           /* 0 = invalid */
    size_t vec = ALIGN16((size_t)n * sizeof(float));                  /* total gain, NR extra noise */
    size_t buf_bytes = 2 * vec + ALIGN16((size_t)n * sizeof(Complex));/* + the output spectrum      */
    if (aec_bytes == 0 || nr_bytes == 0 || synth_bytes == 0) return 1;
    void* pool = NULL;
    if (posix_memalign(&pool, 16, ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                  ALIGN16(synth_bytes) + buf_bytes) != 0) return 1;

    uint8_t* cur = (uint8_t*)pool;
    Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);                    cur += ALIGN16(aec_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);               cur += ALIGN16(nr_bytes);
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);     cur += ALIGN16(synth_bytes);
    if (!aec || !nr || !istft) return 1;
    float* total = (float*)cur;       cur += vec;
    float* extra_noise = (float*)cur; cur += vec;
    Complex* spec = (Complex*)cur;
    uint32_t rng = CNG_RNG_SEED;                                      /* comfort-noise generator */

    float mic[512], ref[512], out[512];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < grid.hop_size; i++) {            /* your capture source */
            mic[i] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        AecResContext c;
        const float* gnr = NULL;
        aec_process_context(aec, mic, ref);                   /* linear AEC, no time-domain emit */
        aec_get_res_context(aec, &c);                         /* E, R2, comfort noise, G_res (c.res_gain) */
        if (c.error_spec && c.res_gain) {
            const float* extra = NULL;
            if (c.r2 && c.usable_linear) {                    /* echo powers into the noise floor */
                sk_scale_f32(extra_noise, c.r2, PSD_SCALE_INV, n);
                extra = extra_noise;
            }
            if (anr_process_gain(nr, c.error_spec, extra, NULL) == 0)
                gnr = anr_get_gain(nr, NULL);                 /* G_nr, owned by the denoiser */
        }
        if (!gnr) {
            aec_reset(aec);                                   /* a refused hop: reset every stage */
            anr_reset(nr);
            ac_istft_reset(istft);
            rng = CNG_RNG_SEED;
            continue;
        }
        sk_min_f32(total, gnr, c.res_gain, n);                /* a < b ? a : b, not fminf */
        sk_capply_gain_f32(spec, c.error_spec, total, n);
        if (aec_cfg.enable_cng && c.comfort_noise)            /* comfort noise on the bins RES cut */
            aec_cng_fill(&rng, spec, c.comfort_noise, c.res_gain, gnr, CNG_NR_GAIN_FLOOR, n);
        (void)ac_istft_push(istft, spec, out);
        if (!skn_all_finite_f32(out, (size_t)grid.hop_size)) {
            aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = CNG_RNG_SEED;
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

編譯（`<...>` 換成你 `print-lib-path` 查到的路徑）：

```sh
cc -std=gnu99 -O2 -Wall -Wextra -c app.c -o app.o \
  -I lib/aec/c_impl/include -I lib/nr/c_impl/include \
  -I <audiolib>/common/include -I <audiolib>/common
c++ -o app app.o \
  <aec-bin>/libaec.a \
  <nr-bin>/libanr.a \
  <audiolib-bin>/libaudio_common.a \
  -lm
```

> `float mic[512]` 是因為本例已知 hop 最大為 512（48 kHz）。
> 產品程式請依 `grid.hop_size` 配置，不要寫死。

---

導覽：[索引](integration_mono_zh_TW.md) ｜ [上一部分](integration_mono_zh_TW.md) ｜ [下一部分](integration_mono_zh_TW_02_lifecycle_config_errors.md) <!-- split-nav -->
