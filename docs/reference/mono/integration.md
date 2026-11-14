# C 呼叫與生命週期

## 0. 速覽：流程時序圖、公開／內部函式、外部／內部參數

本節是整份手冊的地圖；每一列都對得到後面章節或原始碼。名詞：**外部** = 第三方整合者可以調的參數；
**內部** = 程式碼裡存在、技術上也調得動，但**不建議開放**（理由與所在位置都列出）。

### 0.1 流程時序圖（mono：`Aec` 加逐個初始化的函式庫）

生命線：應用（你的 `main`）、`Aec`（lib/aec）、`AnrDenoiser`（lib/nr）、`AcIstft`（audiolib）。
箭頭上的名稱都是真實函式；沒有共用的串接 header，每 hop 的串接寫在各範例自己的 `main`
（`mono/aec_nr_res/main.c` 的 `derive_lane()`、`mono_nr_gain()`、`mono_form()`），可整份複製改寫。

```
應用                       Aec                 AnrDenoiser            AcIstft
 │                          │                       │                     │
 │══ 初始化（一次；每次 init 前都重查大小，不快取）═══════════════════════════
 │ aec_resolve_signal_grid(sr,fft,&grid)    → AEC_OK = 有這組格點；AEC_ERR_ARG = 沒有（fft 取 aec_default_fft_size(sr) 或你指定的值）
 │ aec_config_from_level(&aec_cfg,level,sr); 再設 fft_size、delay_*、residual = AEC_RESIDUAL_FULL（§4.1）
 │ aec_get_mem_size(&aec_cfg) ─────►│            → bytes（0 = 無效）
 │ anr_config_for_level_grid(sr,fft,level); anr_get_mem_size(&nr_cfg) ──►│   → bytes（0 = 無效）
 │ ac_stft_config_default(&sc,fft); sc.window = 任一非 NULL 替身表; ac_istft_get_mem_size(&sc) ──►│   → bytes（0 = 無效）
 │ 配置一塊 16-byte 對齊的 pool = Σ ALIGN16(各塊) + 增益/頻譜暫存
 │ aec_init(mem,bytes,&aec_cfg) ────►│  anr_init(mem,bytes,&nr_cfg) ─►│  aec_synthesis_window(aec,&sc.window,&len) → ac_istft_init(mem,bytes,&sc) ─►│
 │                                  （任一個回 NULL：停止，不要繼續跑）
 │══ 每個 hop（mic[hop]、ref[hop]；hop = grid.hop_size）════════════════════
 │ aec_process_linear(aec,mic,ref) ►│            （回 AEC_OK 或錯誤碼；線性 AEC）
 │ aec_process_residual(aec,NULL) ──►│            （殘留階段：AEC 自己的 RES 增益、r2、comfort noise 在這裡算好）
 │ aec_get_residual(aec,&v) ◄───────│            v{error_spec,near_spec,r2,comfort_noise,res_gain,base_spec,guard_active,...}
 │                                  │            v.base_spec = 渲染基底 B（E；守衛觸發的 hop 為 E+echo）
 │ aec_post_reset_serial(aec) 變了？  → ac_istft_reset(istft)   （在 push 之前；Aec 自己重設 comfort noise 的種子；降噪器不動）
 │ sk_scale_f32(residual_psd_audio,v.r2,PSD_SCALE_INV,n)   （use_r2 且 v.usable_linear 時：回聲功率摺進 noise floor）
 │ anr_process_gain_with_residual_psd(nr,v.base_spec,residual_psd_audio) ─►│ G_nr（NULL = 被拒）
 │ aec_apply_gain(aec,G_nr,spec)      （spec = B × min(G_nr, G_res)；沒有降噪器時 extra 傳 NULL；--no-res 改用 aec_apply_gain_spec 配 unity 向量）
 │ aec_apply_cng(aec,G_nr,spec)       （CNG 開、有 RES 增益時；與上一行同一個 extra 增益）
 │ ac_istft_push(istft,spec,out) ────────────────────────────────────────►│ 寫出 hop 個樣本
 │ skn_all_finite_f32(out,hop)        （選用；板端 adapter 與 §2 範例做，main.c 不做）
 │══ 任一步被拒（或你的輸出檢查失敗）══════════════════════════════════════════
 │ aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); handled = aec_post_reset_serial(aec)   （一起做）
 │══ 強度（兩個 hop 之間呼叫，非 thread-safe）═══════════════════════════════
 │ aec_set_level(aec,level) ─────────►│          anr_set_level(nr,level) ──►│
 │══ 結束 ═══════════════════════════════════════════════════════════════
 │ anr_destroy(nr); aec_destroy(aec); 釋放 pool    （AcIstft 沒有 destroy；先 destroy 再釋放 pool）
 │
 │ --aec-only：只建 Aec（AEC_RESIDUAL_STATE），每 hop 呼叫 aec_process_linear()、aec_process_residual() 後把 aec_get_linear() 的 linear_hop 複製到 out（不經 NR、合成與增益暫存）
```

延遲泳道（橫軸 = hop；`ac_istft_push()` 的 identity 鏈延遲 = 1 hop，見 `audio_stft.h` 檔頭）：

```
hop 序號          t        t+1       t+2
輸入 mic/ref      m[t]     m[t+1]    m[t+2]
Aec + RES + NR    處理 m[t] 處理 m[t+1] 處理 m[t+2]     （同一次呼叫內完成，線性路徑本身延遲 0）
輸出 out          y[t-1]   y[t]      y[t+1]            （合成 OLA 多 1 hop；--aec-only 沒有這 1 hop）
```

1 hop = `grid.hop_size` 個樣本：16 kHz/256 為 128 個樣本（8 ms）、16 kHz/512 為 256（16 ms）、48 kHz 為 512（約 10.7 ms）；8 kHz/256 為 128（16 ms）。數字由 [§4.3](lifecycle.md#43-grid由-sample_rate--fft_size-唯一決定) 的格點推得。

### 0.2 公開函式與內部函式

**整合者實際會呼叫的（公開）**

| 階段 | 函式（標頭） | 備註 |
|---|---|---|
| 格點與 `AecConfig` | `aec_resolve_signal_grid()`、`aec_default_fft_size()`、`aec_config_from_level()`（`aec.h`） | `aec_resolve_signal_grid()` 回 `AEC_OK` = 有這組格點、`AEC_ERR_ARG` = 沒有；其餘接縫欄位由你設（[§4.1](lifecycle.md#41-aecconfig-的接縫欄位範例的-derive_lane)） |
| 大小查詢 | `aec_get_mem_size()`（`aec.h`）、`anr_get_mem_size()`（`anr.h`）、`ac_istft_get_mem_size()`（`audio_stft.h`） | 回 `0` = config 無效 |
| 建立 | `aec_init()`、`anr_init()`、`ac_istft_init()`；桌面也可 `Aec* aec_create(const AecConfig*)`、`anr_create()` | 失敗回 `NULL`。`aec_create()` 是一塊 heap 配置，`aec_destroy()` 釋放它，之後指標作廢（不要再 `free()`）；`aec_init()` 的 pool 實例由呼叫端持有 |
| 每 hop | `aec_process_linear()`、`aec_process_residual()`、`aec_get_residual()`、`aec_apply_gain()`、`aec_apply_cng()`（`aec.h`）、`anr_process_gain_with_residual_psd()`（沒有殘留回音 PSD 時傳 NULL；`anr.h`）、`sk_scale_f32()`（`utility/simd_kernels.h`）、`ac_istft_push()`；選用 `skn_all_finite_f32()`（`utility/simd_kernel_nn.h`） | 順序見 [§0.1](#01-流程時序圖monoaec-加逐個初始化的函式庫) |
| 只要線性 AEC | lane 設 `residual = AEC_RESIDUAL_STATE`，`aec_process_linear()`、`aec_process_residual()` 之後 `aec_get_linear()`，輸出取 `AecLinearView.linear_hop` | 範例的 `--aec-only` 這樣做；`aec_process()` 是函式庫的時域便利入口，`residual` 低於 `AEC_RESIDUAL_FULL` 時回傳同一個 hop |
| 強度 | `aec_set_level()`、`anr_set_level()`（需要整組覆寫時 `anr_reconfigure()`） | [§3.2](lifecycle.md#32-執行期強度控制) |
| 重置／收尾 | `aec_reset()`、`anr_reset()`、`ac_istft_reset()`、`aec_destroy()`、`anr_destroy()` | [§3](lifecycle.md#3-lifecycle)、[§5](lifecycle.md#5-錯誤處理) |
| 診斷（唯讀） | `aec_debug_status()`、`anr_debug_status()`、`aec_get_last_timing()` | [§3.1](lifecycle.md#31-執行期唯讀存取)、[§3.1b](lifecycle.md#31b-逐階段耗時診斷用) |

**內部函式（不要依賴，隨版本改變）**：AEC 的內部標頭在函式庫的 `include/internal/`（`pbfdkf.h`、`delay_aec.h`、`aec_post.h`、
`suppression_gain.h`、`aec_config_build.h`、`aec_tuning.h` 等），不在公開的 include 路徑（`-I include`）上；`include/` 只有 `aec.h`；NR 的 `noise_estimator.h`、`spp_estimator.h`、
`anr_internal.h`，以及 `anr_apply_*_experiment()`；audiolib 的 `utility/` 內核（除上表列出的 `sk_*`、`ALIGN16` 外）。
`Aec` 是不透明型別（`aec.h` 只有 `typedef struct Aec Aec;`，本體在函式庫的 `include/internal/aec_impl.h`）：不能 `sizeof(Aec)`、不能宣告 `Aec` 變數、沒有欄位可讀，只能經由函式存取。

### 0.3 參數面：外部與內部

**外部（建議讓第三方整合者調）**

| 名稱 | 位置 | 範圍／預設 | 何時設定 | 作用 |
|---|---|---|---|---|
| `sample_rate`、`fft_size` | `aec_resolve_signal_grid()`、`AecConfig.fft_size`（`aec.h`） | 8000／16000／48000；`fft_size` 取 `aec_default_fft_size(sr)`（0 = 不支援的 rate），16 kHz 另可 256／512 | init 前，之後不可變 | 決定格點（[§4.3](lifecycle.md#43-grid由-sample_rate--fft_size-唯一決定)）；改了要重查所有大小 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `AecConfig` | `MATCHED`（預設，bank 1..5，預設 5）／`FIXED`（已知延遲）／`EXTERNAL_ALIGNED`（上游已對齊） | init 前，之後不可變 | 對齊方式；決定 pool 版面（[§4.4](lifecycle.md#44-哪些欄位會改變記憶體用量)、[§4.5a](lifecycle.md#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline)） |
| `AecConfig.level`（`aec_config_from_level(&cfg, level, sr)` 寫入） | `aec.h` | `MILD`(0)／`BALANCED`(1，預設)／`AGGRESSIVE`(2)／`NO_FLOOR`(3)；`aec_config_from_level()` 對 enum 外存成 balanced，手填的 `level` 不在 0..3 會被拒絕 | init 前；執行期用下一列 | 回聲抑制 vs 近端保留的單一取捨軸 |
| `aec_set_level(aec, level)` | `aec.h` | level 在 enum 內（ramp 固定 100 ms） | 兩個 hop 之間 | 執行期換 AEC 強度；回 `-1` 時什麼都不寫 |
| NR 強度 | `AnrLevel`（`anr_types.h`）：`anr_config_for_level_grid()`、`anr_set_level()` | `MILD`／`MODERATE`／`BALANCED`（預設）／`AGGRESSIVE` | init 前或兩個 hop 之間 | 降噪深度的單一入口（一次調 `g_min_db`、`q`、`xi_min_db`、`noise_over_subtraction`） |
| 後級開關（範例程式的欄位，不是函式庫 config） | `PostPlan`（`mono/aec_nr_res/main.c`）；CLI `--no-res`、`--no-nr`、`--no-cng`、`--legacy-amin` | 各 0／1 | init 前 | RES 增益用不用、要不要建降噪器、CNG、NR 是否折入 R²（[§4.2](lifecycle.md#42-後級選項範例程式的欄位與-cli-旗標)） |

進階外部（選擇性；改了以後自己重跑三個 grid 的驗證）：`AecConfig.enable_cng`（`aec.h`，下游已有舒適噪音時關）。`AecConfig` 只有 12 個公開欄位；濾波器長度、HPF、shadow、飽和處理與所有門檻都不是欄位，由函式庫依 `(sample_rate, fft_size, n_mics, level)` 決定。

**內部（存在、可調，但不建議開放）**

| 家族 | 位置 | 為什麼不開放 |
|---|---|---|
| 範例 `derive_lane()` 固定的接縫：`residual = AEC_RESIDUAL_FULL`（預設值；`OFF`／`STATE`／`ESTIMATE` 依序少給 `res_gain`、`r2`／`comfort_noise` 等） | `aec.h` 的 `AecConfig.residual`；`mono/aec_nr_res/main.c` 的 `derive_lane()` | 低於 `FULL` 就沒有 `v.res_gain`，`aec_apply_gain()` 回 `AEC_ERR_MODE`，串接整個失效 |
| AEC 的全部調校值（濾波器長度與步進、shadow、HPF、飽和處理、EPC／近端偵測門檻、延遲獲取策略、double-talk 與 floor 常數、預設關閉的實驗、以 10 ms 格點撰寫的時間欄位） | 函式庫內部的 `AecTuning`（`include/internal/aec_tuning.h`），不在 `aec.h` | 由 `aec_resolve()` 依 `(sample_rate, fft_size, n_mics, level)` 建立並在建立時 retime，呼叫端沒有設定入口；level 的三個 floor 值隨 `AecConfig.level`／`aec_set_level()` 成組改變 |
| `SuppressionGainConfig`、`AecPostConfig`、`AecStateConfig`、`aec_balanced_config.h` 的常數 | AEC 內部標頭 | 編譯期依 `(sample_rate, fft_size)` 查表，runtime 改不了，也不該提供接口 |
| NR 的追蹤器與平滑欄位：`alpha_xi`、`alpha_s`、`alpha_d`、`alpha_p`、`alpha_attack`、`alpha_decay`、`delta_db`、`scene_change_*`、`speech_protect_*`、`speech_aware_*`、`noise_gate_*`、`makeup_*`、`cross_band_*` | `AnrConfig`（`anr_types.h`）的同名欄位 | 聯合調出的操作點，四個 `AnrLevel` 共用；`AnrConfig.alpha_g` 在增益路徑上沒有作用 |
| NR 的 `L`、`num_init_frames` | `AnrConfig.L`、`AnrConfig.num_init_frames`（`anr_types.h`） | 決定 pool 版面，`anr_reconfigure()` 與 `anr_set_level()` 不能改它們；範例用函式庫預設 |

---

## 1. 你連結什麼、你 `#include` 什麼

### 1.1 構成：每個函式庫各自初始化，每個 hop 的串接由你寫

mono 沒有包起來的模組、沒有 pipeline 函式庫，也沒有共用的串接 header。處理鏈由下列獨立的函式庫物件組成，
**每一個都有自己的 `mem_size` 與 `init`**，初始化與每個 hop 的串接都是整合者的事：

```
mic hop + ref hop ──► Aec (aec_process_linear + aec_process_residual) ──► aec_get_residual ──► AecResidualView
                                                                                                    │
        ┌───────────────────────────────────────────────────────────────────────────────────────────┘
        ▼   G_res = v.res_gain（AEC 自己的殘留 gain，留在 Aec 裡，由 aec_apply_gain 取用）
   G_nr = anr_process_gain_with_residual_psd(...)  ─►  aec_apply_gain(aec, G_nr, spec)（B × min(G_nr, G_res)）
        ─►  aec_apply_cng(aec, G_nr, spec)（CNG 開著、且有 RES gain 時）  ─►  ac_istft_push  ──► out hop
```

圖中 B 是 `aec_get_residual()` 交出的渲染基底 `v.base_spec`（`v.error_spec` 只是分析 seam，不拿來渲染）；`aec_post_reset_serial()` 變了時的處理見下方第 0 步。

mono 的 RES 是明確的例外：它在 `Aec` 的呼叫裡（`aec_process_residual()` 算出殘留 gain，`aec_get_residual()` 以
`v.res_gain` 交出，`aec_apply_gain()` 取用，強度用 `aec_set_level()`），沒有獨立的 `AecRes` 物件；範例只決定要不要套用這個 gain。
純 AEC 的範例（`--aec-only`）不經過合成：lane 是 `AEC_RESIDUAL_STATE`，每 hop 呼叫 `aec_process_linear()` 與 `aec_process_residual()`，輸出取 `aec_get_linear()` 的 `linear_hop`。反向 FFT 與 overlap-add 一律由應用呼叫獨立的 `ac_istft_push()`，不放在演算法函式庫裡。

| 物件 | 來源 | 大小查詢 | 初始化 |
|---|---|---|---|
| `Aec`（線性 AEC，含 AEC3 殘留 gain 與 comfort noise 產生器；RES gain 就是它自己的 `v.res_gain`） | `lib/aec`（`libaec.a`） | `aec_get_mem_size(&aec_cfg)` | `aec_init(mem, bytes, &aec_cfg)` |
| `AnrDenoiser`（可選） | `lib/nr`（`libanr.a`） | `anr_get_mem_size(&nr_cfg)`（`nr_cfg = anr_config_for_level_grid(sr, fft_size, level)`，最小值追蹤視窗 `L` 用函式庫預設） | `anr_init()` |
| `AcIstft`（合成） | audiolib（`libaudio_common.a`） | `ac_istft_get_mem_size(&synth_cfg)`（`ac_stft_config_default(&synth_cfg, fft_size)`；`synth_cfg.window` 先設任一非 NULL 的替身表來定尺寸，`aec_init()` 之後以 `aec_synthesis_window()` 換成 AEC 自己的合成窗，借用的窗不佔 pool） | `ac_istft_init()` |
| 增益／頻譜暫存（unity、NR 額外噪聲、輸出頻譜） | 你自己的記憶體 | `2 × ALIGN16(n_freqs × sizeof(float)) + ALIGN16(n_freqs × sizeof(Complex))`（`main.c` 的 `post_plan()`；unity 向量只有 `--no-res` 的路徑用得到，§2 的最小範例只要 1 個 float 向量） | unity 向量要自己填 1 |
| 格點與 `AecConfig` | `lib/aec`（`libaec.a`） | – | `aec_resolve_signal_grid()` + `aec_config_from_level()`，接縫欄位自己設（[§4.1](lifecycle.md#41-aecconfig-的接縫欄位範例的-derive_lane)） |

上面幾項怎麼放進 pool（heap 路徑各自配置，caller-pool 路徑依序切同一塊）與每個 hop 怎麼串，是範例程式
（`mono/aec_nr_res/main.c`、`example_board_adapter.c`）各自寫出來的，沒有 ABI、沒有 layout version、沒有 descriptor、
不進任何 archive，也沒有共用的範例 header，可整份複製改寫。

每個 hop 的串接（`main.c` 的 `mono_form()` 與 `mono_nr_gain()`；兩支執行檔共用）：

0. `aec_post_reset_serial(aec)` 與已處理的值不同（AEC 在這個 hop 內做了硬重置）時，在 `ac_istft_push()` 之前 `ac_istft_reset(istft)`（Aec 自己的 comfort noise 產生器已在同一個事件重設種子）；不要因此重置 AEC，降噪器也不動。
1. `aec_process_linear()` 與 `aec_process_residual()` 都回 `AEC_OK`，`aec_get_residual()` 回 `AEC_OK` 且 `v.res_gain` 非 NULL；任一個不是就視為拒絕該 hop。
2. NR 增益：`use_r2` 開著、`v.r2` 非 NULL 且 `v.usable_linear` 時，先 `sk_scale_f32(extra_noise, v.r2, PSD_SCALE_INV, n_freqs)` 得到殘留回音 PSD（audio 功率尺度）；再 `anr_process_gain_with_residual_psd(nr, v.base_spec, residual_psd_audio)`（`v.base_spec` 是渲染基底 B，與要套增益的是同一張頻譜，所以降噪器判斷的就是它要濾的訊號；未觸發守衛的 hop 上 B 與 `v.error_spec` 逐位元相同；回傳降噪器擁有的 `n_freqs` 個增益，NULL = 降噪器拒絕該 hop；不用殘留回音 PSD 時傳 NULL）。沒有降噪器時 `nr` 用 NULL。
3. `aec_apply_gain(aec, nr, spec)`：`spec = B × min(nr, G_res)`（`nr` 為 NULL 時是 `B × G_res`）。`v.error_spec`／`v.formed_hop` 是分析 seam（永不被輸出守衛改動；AI 模型的分析輸入讀它），不在這裡渲染。git 之外的板端 sample（例如 `alg_audio_aec_nr.c`）目前把 `v.error_spec` 交給降噪器，這是已知差異，不是建議。`--no-res` 時 G_res 視為全 1：`aec_apply_gain_spec(spec, v.base_spec, unity, nr, n_freqs)`。
4. CNG 開著（`aec_cfg.enable_cng` 且沒有 `--no-cng`）且有 RES gain 時 `aec_apply_cng(aec, nr, spec)`：與上一步同一個 `nr`，由 Aec 自己的產生器（種子 `AEC_CNG_SEED`）抽樣。
5. `ac_istft_push(istft, spec, out)` 產生這個 hop 的時域輸出。

`PSD_SCALE_INV`（1/32768²，AEC 的功率在 int16² 尺度、降噪器在音訊尺度）是範例檔頭自己的 `#define`，不是函式庫常數；comfort noise 的種子是函式庫的 `AEC_CNG_SEED`（42），降噪器增益的下限 `AEC_CNG_NR_GAIN_FLOOR`（0.31622777f，即 −10 dB）也在 `aec.h`，由 `aec_apply_cng()` 內部套用。

### 1.2 `--aec-only`

只要線性 AEC 的輸出，就不建後級：lane 以 `residual = AEC_RESIDUAL_STATE` 建立，
每 hop `aec_process_linear(aec, mic, ref)`、`aec_process_residual(aec, NULL)`，再 `aec_get_linear(aec, &lv)`，
把 `lv.linear_hop`（`lv.hop_size` 個 float，AEC 自己的時域 hop）複製到 `out`。
不需要降噪器、合成與增益暫存（見 `aec_nr_pipeline_static --print-mem-size --aec-only`：pool 只有 `Aec` 那一塊，即 STATE lane 的 `aec_get_mem_size`，與完整 chain 的 `Aec` 區塊同大）。
殘留階段照跑，因為它的狀態更新會回饋線性濾波器的步進控制：`linear_hop` 與完整 chain（`FULL` lane）逐位元相同，`test_mono_chain` 在動態回音路徑上逐 hop 驗證。
`aec_process()` 與兩階段呼叫推進相同的狀態；`residual` 低於 `AEC_RESIDUAL_FULL` 的 lane 上
`aec_process()` 回傳的就是這個 hop，測試用同一份 STATE lane 的兩階段呼叫與完整 chain 的 lane 當獨立的對照。

### 1.3 Header closure

```c
#include "aec.h"                       /* Aec, AecConfig, AecResidualView, aec_resolve_signal_grid(), aec_process_linear(), aec_apply_gain() */
#include "anr.h"                       /* AnrDenoiser, anr_config_for_level_grid(), anr_init()  */
#include "audio_stft.h"                /* AcIstft, ac_istft_init(), ac_istft_push()           */
#include "utility/simd_kernels.h"      /* sk_scale_f32()                                      */
#include "utility/mem_align.h"         /* ALIGN16                                             */
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32()，選用的輸出檢查                    */
```

`Complex` 定義在 `<audiolib>/common/include/ac_complex.h`，經 `aec.h` → `fft_wrapper.h` → `ac_complex.h` 帶進來。

### 1.4 編譯需要的 `-I` 清單（consumer 端最小集合）

```
-I ../AEC/c_impl/include
-I ../NR/c_impl/include
-I <audiolib>/common/include
-I <audiolib>/common
```

`pipelines/Makefile` 自己建置範例時的 `-I` 清單比這個長（多了 `../AEC/c_impl/example`，給 CLI 的 WAV I/O 用，以及 AIAEC／AINR 的目錄），
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
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32() */

/* aec_apply_cng()'s generator belongs to the Aec: it starts from AEC_CNG_SEED (42) and restarts from it on every hard post-chain reset */
#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))       /* AEC powers are int16^2, the denoiser's audio scale */
/* The synthesis borrows the AEC's window (aec_synthesis_window()), which exists only after aec_init();
 * a borrowed table carves nothing, so any non-NULL pointer sizes it the same. */
static const float k_sizing_window[1] = {0.0f};

int main(void) {
    const int sample_rate = 16000;
    AecSignalGrid grid;
    AecConfig aec_cfg;
    if (aec_resolve_signal_grid(sample_rate, aec_default_fft_size(sample_rate), &grid) != AEC_OK)
        return 1;                                                  /* AEC_ERR_ARG = no grid for this pair */
    aec_config_from_level(&aec_cfg, AEC_LEVEL_BALANCED, sample_rate);
    aec_cfg.fft_size = grid.fft_size;
    aec_cfg.delay_mode = AEC_DELAY_MATCHED;
    aec_cfg.delay_num_filters = AEC_DELAY_NUM_FILTERS_MAX;
    aec_cfg.fixed_delay_samples = -1;
    aec_cfg.residual = AEC_RESIDUAL_FULL;                          /* the lane keeps its own RES gain (the default) */
    const int n = grid.n_freqs;

    /* every library sizes and initialises itself */
    AnrConfig nr_cfg = anr_config_for_level_grid(sample_rate, grid.fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig synth_cfg;
    ac_stft_config_default(&synth_cfg, grid.fft_size);
    synth_cfg.window = k_sizing_window;                               /* stand-in until aec_init() */
    size_t aec_bytes = aec_get_mem_size(&aec_cfg);                    /* 0 = invalid */
    size_t nr_bytes = anr_get_mem_size(&nr_cfg);                      /* 0 = invalid */
    size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);           /* 0 = invalid */
    size_t vec = ALIGN16((size_t)n * sizeof(float));                  /* NR extra noise            */
    size_t buf_bytes = vec + ALIGN16((size_t)n * sizeof(Complex));    /* + the output spectrum     */
    if (aec_bytes == 0 || nr_bytes == 0 || synth_bytes == 0) return 1;
    void* pool = NULL;
    if (posix_memalign(&pool, 16, ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                  ALIGN16(synth_bytes) + buf_bytes) != 0) return 1;

    uint8_t* cur = (uint8_t*)pool;
    Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);                    cur += ALIGN16(aec_bytes);
    AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);               cur += ALIGN16(nr_bytes);
    int win_n = 0;
    if (!aec || aec_synthesis_window(aec, &synth_cfg.window, &win_n) != 0 || win_n != synth_cfg.fft_size)
        return 1;                                                     /* the AEC's own synthesis window */
    AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);     cur += ALIGN16(synth_bytes);
    if (!nr || !istft) return 1;
    float* extra_noise = (float*)cur; cur += vec;
    Complex* spec = (Complex*)cur;
    uint32_t handled = aec_post_reset_serial(aec);                    /* hard post-chain resets already followed */

    float mic[512], ref[512], out[512];
    for (int frame = 0; frame < 100; frame++) {
        for (int i = 0; i < grid.hop_size; i++) {            /* your capture source */
            mic[i] = 0.01f * (float)((i % 61) - 30);
            ref[i] = 0.02f * (float)((i % 37) - 18);
        }
        AecResidualView v = { .struct_size = sizeof(v) };         /* v.base_spec: the rendering base B */
        const float* gnr = NULL;
        if (aec_process_linear(aec, mic, ref) == AEC_OK &&        /* linear AEC, no time-domain emit */
            aec_process_residual(aec, NULL) == AEC_OK &&          /* G_res, R2, comfort noise inside the Aec */
            aec_get_residual(aec, &v) == AEC_OK && v.res_gain) {
            if (aec_post_reset_serial(aec) != handled) {          /* hard reset inside this hop: before the push */
                handled = aec_post_reset_serial(aec);
                ac_istft_reset(istft);                            /* the Aec reseeded its comfort noise itself */
            }
            const float* residual_psd_audio = NULL;
            if (v.r2 && v.usable_linear) {                        /* echo powers into the noise floor */
                sk_scale_f32(extra_noise, v.r2, PSD_SCALE_INV, n);
                residual_psd_audio = extra_noise;
            }
            gnr = anr_process_gain_with_residual_psd(nr, v.base_spec, residual_psd_audio);   /* G_nr, owned by the denoiser; NULL = refused */
        }
        if (!gnr) {
            aec_reset(aec);                                       /* a refused hop: reset every stage */
            anr_reset(nr);
            ac_istft_reset(istft);
            handled = aec_post_reset_serial(aec);
            continue;
        }
        aec_apply_gain(aec, gnr, spec);                           /* spec = B * min(G_nr, G_res) */
        if (aec_cfg.enable_cng)                                   /* comfort noise on the bins RES cut, same gnr */
            aec_apply_cng(aec, gnr, spec);
        (void)ac_istft_push(istft, spec, out);
        if (!skn_all_finite_f32(out, (size_t)grid.hop_size)) {
            aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); handled = aec_post_reset_serial(aec);
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



將範例存成 Audio_ALG 根目錄的 `app.c`。以下使用預設的 sibling 依賴。

```sh
make -C pipelines libs
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

> `float mic[512]` 是因為本例已知 hop 最大為 512（48 kHz）。
> 產品程式請依 `grid.hop_size` 配置，不要寫死。

[返回本主題](README.md)
