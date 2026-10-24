# AEC、NR、audiolib C API 整合速查

本文件是給 application／board integration 使用者的**短版入口**，涵蓋 AEC、NR、audiolib
三個 library，以及三個 AI 模型的 pre/post 類別（§6），回答三件事：

1. 應該 include 哪些 public header；
2. init、process、reset、destroy 的正確順序；
3. 哪些工作由 library 負責，哪些必須由呼叫端負責。

完整欄位、錯誤語意與演算法說明仍以各 library 的 canonical manual 為準：

- [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md)
- [NR C 使用手冊](../lib/nr/docs/c_user_manual_zh_TW.md)
- [audiolib C 使用手冊](../../audiolib/common/docs/c_user_manual_zh_TW.md)
- [Audio_ALG mono／4ch pipeline 使用手冊](c_user_manual_zh_TW.md)
- 模型側的圖邊界、I/O 表與延遲：`docs/html/` 的各模型頁（`ainr_dfn2.html` 等）

若本文件與 header 不一致，以目前 pin 住的 public header 與測試為準。

---

## 1. 先決定使用層級

| 需求 | 建議入口 |
|---|---|
| mono `AEC + NR + RES` | 沒有 mono 單元：`aec.h` 的 `Aec`（`aec_process_context()`）加 `common/se_stage.h` 的 `se_frame_from_aec()`，再串 `anr.h`、`audio_stft.h`（RES gain 就是 `Aec` 自己的 `ctx.res_gain`，經 `se_frame_from_aec()` 變成 `frame.g_res_ctx`）；範例 `pipelines/mono_aec_nr_res/main.c`、`static_main.c` |
| 4ch linear AEC + BF（唯一包起來的單元；不含 RES／NR／合成） | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` |
| 4ch linear AEC 與外部 beamformer seam | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` |
| 只做單路 AEC，或自行組合 NN／其他 post-filter | `aec.h` |
| 已有 STFT/WOLA，只需要頻域 NR | `anr.h` |
| 只需要 FFT、resample、HPF | `audiolib/common/include/*.h` |
| Align-ULCNet 加速器 handoff（hop 或頻譜） | `AIAEC/Align_ULCNet/ulcnet_prepost.h` |
| DeepVQE-S 加速器 handoff | `AIAEC/DeepVQE_S/deepvqe_prepost.h` |
| DeepFilterNet2 加速器 handoff | `AINR/DeepFilterNet2/dfn2_prepost.h` |
| DeepFilterNet3 加速器 handoff | `AINR/DeepFilterNet3/dfn3_prepost.h` |

只有「多聲道線性 AEC + BF」被包成一個單元（`AudioPipeline4Ch`；`fuse()` 之後 `frame->g_res_ctx` 為 NULL，單元不持有 `AecRes`）。
mono 是例外：它的 RES gain 就在 `Aec` 的呼叫裡（`aec_process_context()` 算出，`se_frame_from_aec()` 交成 `frame.g_res_ctx`）。
4ch 的 post-beam RES、NR、comfort noise、合成與 NN 後濾波都是獨立的 library，各自查大小、各自 init，由整合者串接。
各範例 `main` 自己初始化用到的函式庫並明寫每 hop 的串接（4ch 為 `se_res_stage_gain()` 的 RES gain → `se_anr_stage_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`）；`pipelines/samples/` 的共用範例碼
（`se_anr_stage.h`、`dfn_hop.h`、`ulcnet_nn_stage.{h,c}`）示範 NR 級與 NN stage 怎麼串；沒有 ABI，
可整份複製改寫，不要在 application 重新發明 AEC/NR gain fusion、window、OLA 或 CNG 的順序。

各條流程的**呼叫時序圖**（含回傳值、被拒時的處理與 hop 泳道）與**公開／內部函式、外部／內部參數**分級見各整合手冊的第 0 節：
[mono](integration_mono_zh_TW.md)、[4ch 核心](integration_4ch_core_zh_TW.md)、[4ch 單元](integration_4ch_spatial_zh_TW.md)；
板端的 query → allocate → init → teardown 順序見 [pipeline_board_integration.md](pipeline_board_integration.md)。本文 §3.6、§4.6 是 AEC 與 NR 兩個函式庫自己的參數分級。

---

## 2. 共通信號契約

### 2.1 產品 grid

| sample rate | frame = FFT | hop | bins | 狀態 |
|---:|---:|---:|---:|---|
| 16 kHz | 256 | 128 | 129 | 預設、低延遲 |
| 16 kHz | 512 | 256 | 257 | 可選 |
| 48 kHz | 1024 | 512 | 513 | 48 kHz 產品 grid |

AEC 與 NR 另支援 8 kHz grid（非產品主 grid），4ch product pipeline 不接受它。不要只傳
`sample_rate` 後讓底層猜 FFT；16 kHz 有兩個合法值，建立實例前必須 resolve 成明確的
`fft_size`。

### 2.2 PCM、window 與 block

- PCM 使用 mono/interleaved `float32`，標稱範圍 `[-1, 1]`。
- AEC 每次固定吃一個 hop；函式沒有長度參數，傳錯長度不會替你報錯。
- NR C library 只吃 `Complex[n_freqs]`，不做 framing、window、FFT、IFFT 或 OLA。
- 時域 NR wrapper 使用 periodic sqrt-Hann analysis/synthesis window、50% overlap、
  `frame_size == fft_size`，不補零。
- AEC 不做最終 peak limiter。若產品需要 limiter／AGC／DRC，放在整條
  AEC/BF/NR/RES chain 最後。
- 每個 stateful instance 只服務一條 stream；換檔、換 session 或 discontinuity 時 reset。

---

## 3. AEC library

### 3.1 Public header

一般整合只需要：

```c
#include "aec.h"
```

只有自行在 AEC 外建立 post-beam RES（4ch 應用）時，才額外使用：

```c
#include "aec_res.h"   /* AecRes、aec_cng_fill */
```

`pbfdkf.h`、`delay_aec.h`、`aec_post.h`、`suppression_gain.h` 等是內部 header，不是 application API。

### 3.2 Config 與建立

最常用的 config 流程：

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
cfg.fft_size = 256;  /* 16k/256/128；改成 512 即為 16k/512/256 */
```

延遲模式是 init-time 設定：

| 模式 | 用途 | 主要欄位 |
|---|---|---|
| `AEC_DELAY_MATCHED` | 一般產品；延遲未知或會變 | `delay_num_filters = 1..5` |
| `AEC_DELAY_FIXED` | route delay 已由 bring-up 量得 | `fixed_delay_samples >= 0` |
| `AEC_DELAY_EXTERNAL_ALIGNED` | HAL／上游保證 ref 已對齊 | 不建立 estimator/ring |

`sample_rate`、`fft_size`、`delay_mode`、`delay_num_filters`、filter length 都是
init-time immutable；改動後必須重新查詢 pool 大小並重新 init。

Heap 路徑：

```c
Aec aec;
if (aec_create(&aec, &cfg) != 0) {
    /* config 或 allocation 失敗 */
}

/* ... process ... */
aec_destroy(&aec);
```

Caller-owned static pool 路徑：

```c
size_t bytes = aec_get_mem_size(&cfg);
void *pool = board_alloc_aligned(16, bytes); /* 平台提供；至少 16-byte aligned */
Aec *aec = pool ? aec_init(pool, bytes, &cfg) : NULL;
if (!aec) {
    /* config、alignment 或 pool size 錯誤 */
}

/* ... process ... */
aec_destroy(aec);       /* static instance 不會釋放 caller pool */
board_free(pool);
```

不要把手冊中的歷史 byte 數寫死在產品；永遠以目前 config 呼叫
`aec_get_mem_size()`。若要拆解 estimator/ring 成本，可呼叫
`aec_get_mem_breakdown()`。

### 3.3 每-hop 處理入口

一般 lockstep AEC：

```c
int hop = aec_hop_size(aec);
aec_process(aec, mic_hop, ref_hop, out_hop);
```

三個 buffer 都至少有 `hop` 個 `float`。`ref_hop` 必須是實際送往 speaker 的 render
reference，且與 mic 同 sample clock。

若下游只取頻域／linear context，不需要 AEC 自己的時域輸出：

```c
AecResContext ctx;
AecLinearContext linear;

aec_process_context(aec, mic_hop, ref_hop);
aec_get_res_context(aec, &ctx);
aec_get_linear_context(aec, &linear);
```

這些 context 指標借用 AEC 內部記憶體，只能讀到下一次 process/reset。多 lane 共用同一
far 時，`aec_process_context_shared_far_from()` 可讓其他 lane 借用同 hop 已算好的 far FFT 與 magnitude²；它有嚴格
same-hop/same-reference 前提，通常只應由 4 通道核心（`pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c`）使用。

### 3.4 Reset、context 與診斷

| API | 用途 |
|---|---|
| `aec_reset()` | stream/session boundary；保留 config，清除 DSP state |
| `aec_get_linear_context()` | formed linear output、aligned far、delay state/generation |
| `aec_get_res_context()` | error/near/echo spectrum、R2、RES gain、CNG context |
| `aec_debug_status()` | delay、ERLE、convergence、power、duty-cycle snapshot |
| `aec_apply_external_realign()` | 僅 `EXTERNAL_ALIGNED` mode；通知外部 alignment 已改變 |

`AecLinearContext`／`AecResContext` 是 read-only borrowed views，不可保存其 pointer 跨 hop。
`enable_res=0` 搭 `return_res_context=1` 時 formed seam 多一個 capture 候選，**不保證每個 hop
都是誤差**，條件與後果見 [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md) §8.1。

### 3.5 執行期改抑制強度

```c
int aec_set_preset(Aec *aec, AecPreset preset, float ramp_ms);
```

preset 改變 far-active 地板 `min_gain_floor_far_active_db`（-20/-28/-38 dB）；aggressive 另把 DT 地板
`min_gain_floor_dt_db` 設成 -30 dB；NO_FLOOR 關閉 `min_gain_split_floor_enabled`。地板是抑制器每 hop 只讀一次的
單一純量下限，所以換 preset 是重新指定目標值、不是重建：濾波器、延遲鎖定與每一條
平滑歷史都繼續跑。

```c
/* 兩個 hop 之間呼叫；ramp_ms=0 下一個 hop 生效，>0 以 dB 線性走過去（上限 60 s） */
if (aec_set_preset(aec, AEC_PRESET_AGGRESSIVE, 100.0f) != 0) {
    /* NULL / preset 超出 enum / ramp_ms 超出 [0, 60000]；此時什麼都沒寫 */
}
```

`ramp_ms == 0` 不是錯誤。與 `aec_config_from_preset()` 遇到不認得的值退回 balanced
相反，這個 setter **直接拒絕**。ramp 途中再呼叫會從當前 live 值重新起走；
`aec_reset()` 保留目標值、丟棄 ramp 進度。**在 hop 之間呼叫、與 process 序列化；
非 thread-safe。**

自己在 beamformer 之後跑 RES 的整合（4ch 就是這樣）改用 `AecRes`（`aec_res.h`），
因為那個共用抑制器不屬於任何 `Aec`：

```c
#include "aec_res.h"

int aec_res_set_preset(AecRes *res, AecPreset preset, float ramp_ms);
int aec_res_split_floor(const AecRes *res, float *live, float *target);
```

語意與 `aec_set_preset()` 相同：只改 far-active 地板、`ramp_ms` 範圍 `[0, 60000]`、
失敗時什麼都不寫；`aec_res_reset()` 保留目標值。地板是與
`AecConfig.min_gain_floor_far_active_db` 相同的 dB-power 量換成的功率比；`live` 是現行
值、`target` 是正走向的值。

> **量測前務必知道**：far-active 地板只在 **far-active 且非 double-talk** 的 hop 生效
> （double-talk 期間套用 DT 地板：mild/balanced 共用 -20 dB，aggressive 為 -30 dB；far-active latch 觸發前用的是 far-silent 地板），
> 而同一個 gain 還決定 comfort noise 量。所以整段錄音的平均值移動幅度會小於 dB 落差
> 所暗示的量，只量 echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制。請在
> echo 對齊或 degradation 對齊的條件下比較，並實際試聽。
>
> 4ch 產品另有一條：四條 lane 以 `spatial_linear_context` 建立，從不走到自己的
> suppression-gain 路徑，對 lane 呼叫 `aec_set_preset()` 依建構方式即為無效操作。
> 請改用共用的 post-beam `AecRes` 的 `aec_res_set_preset()`：無論用 `AudioPipeline4Ch` 還是直接驅動核心 `FourAecNrRes`，
> 都是對應用自己用 libaec 建的那個 `AecRes` 呼叫。它針對的是那個共用的 post 級抑制器；單元沒有任何強度 setter。
> pipeline 層的對應入口見 [4ch 核心整合手冊 §3.2](integration_4ch_core_zh_TW.md)
> 與 [mono 整合手冊 §3.2](integration_mono_zh_TW.md)。

### 3.6 參數分級：外部與內部

欄位在 `aec.h:59-156`。建議一律以 `aec_config_from_preset(&cfg, preset, sample_rate)` 取得完整設定，再只覆寫外部欄位；
`aec_create()`／`aec_init()` 會複製設定，之後改你手上那份無效。

| 分級 | 欄位／函式 | 範圍與預設 | 何時設定 |
|---|---|---|---|
| 外部 | `sample_rate`、`fft_size`（`fft_size = 0` 被拒，要明確值）、`filter_length` | 8000／16000／48000；16 kHz 256 或 512、48 kHz 1024、8 kHz 256；`filter_length` `[1, 4096]`（52 ms，≥44.1 kHz 為 64 ms） | 建立時，不可變 |
| 外部 | `delay_mode`、`delay_num_filters`、`fixed_delay_samples` | `MATCHED`（預設，bank 1..5）／`FIXED`（≥ 0）／`EXTERNAL_ALIGNED`；其他模式的對應欄位須維持預設 | 建立時，不可變 |
| 外部 | `enable_res`、`enable_cng` | 0／1，預設 1 | 建立時 |
| 外部 | `AecPreset` | `MILD`／`BALANCED`（預設）／`AGGRESSIVE`／`NO_FLOOR`；`aec_config_from_preset()` 對 enum 外的值不報錯、靜默退回 balanced，`aec_set_preset()` 則拒絕 | 建立時，或執行期 `aec_set_preset(a, preset, ramp_ms)` |
| 進階外部 | `enable_highpass`、`highpass_cutoff_hz`、`enable_shadow`、`delay_est_period_s` | 開啟高通時截止頻率須 `> 0` 且 `< 0.45 × sample_rate`；`delay_est_period_s` 預設 0（全速），非 0 可能改變選到的 lag | 建立時 |
| 整合者專用 | `return_res_context`、`spatial_linear_context` | `spatial_linear_context` 非 0 時必須 `enable_res = 0` 且 `return_res_context = 1` | 串鏈拓撲，不是調參 |
| 內部 | 步進與不確定度（`mu`、`delta`、`kalman_q_*`、`shadow_mu_*`、`epc_mu_floor`）、偵測門檻（`epc_*`、`ne_recent_*`、`saturation_threshold`、`filter_misadjustment_scale_*`）、10 ms 格點撰寫的時間欄位（`warmup_frames` 等）、延遲獲取策略（`delay_acquire_*`、`max_delay_ms`、`delay_buffer_ms`）、DT 政策與 floor（`dt_aware_*`、`min_gain_*`）、預設關閉的實驗（`live_observer_*`、`cng_n2_windowed_psd`）、衍生值（`n_partitions`、`enable_delay_est`） | 驗證器只擋崩壞，不擋音質 | 不建議開放；理由見 [mono 手冊 §0.3](integration_mono_zh_TW.md) |

編譯期固定、runtime 改不了的常數（`SuppressionGainConfig`、`AecPostConfig`、`AecStateConfig`、`aec_balanced_config.h`、
`DA_*`）也屬內部。函式面：公開 = `aec.h` 與 `aec_res.h` 的函式；內部 = 其他 AEC 標頭（§3.1）。

---

## 4. NR library

### 4.1 Public header 與責任邊界

```c
#include "anr.h" /* 已 include anr_types.h/fft_wrapper.h */
```

NR 是 mono frequency-domain MMSE-LSA。呼叫端必須準備正確 WOLA/STFT；若 application
手上只有 PCM，參考 `pipelines/mono_aec_nr_res/main.c` 的 `AcIstft` 串接，或照 NR 手冊的 `NrStream`
範例實作。

### 4.2 Config 與建立

```c
AnrConfig cfg = anr_config_for_mode_grid(
    16000, 256, ANR_MODE_BALANCED);

if (!anr_validate_config(&cfg)) {
    /* 不支援的 grid 或參數 */
}
```

只傳 sample rate 的 `anr_config_for_mode()` 會選該 rate 的預設 grid；16 kHz 要選
512/256 時，應明確使用 `_for_mode_grid()`。

Heap 路徑：

```c
AnrDenoiser *nr = anr_create(&cfg);
/* ... process ... */
anr_destroy(nr);
```

Static pool 路徑：

```c
size_t bytes = anr_get_mem_size(&cfg);
void *pool = board_alloc_aligned(16, bytes);
AnrDenoiser *nr = pool ? anr_init(pool, bytes, &cfg) : NULL;

/* ... process ... */
anr_destroy(nr);  /* 不釋放 caller pool */
board_free(pool);
```

### 4.3 每-frame 頻域入口

直接套 NR gain，可 in-place：

```c
if (anr_process(nr, spectrum, spectrum) != 0) {
    /* invalid handle/buffer */
}
```

AEC + NR + RES gain fusion 應只算 gain、不先改 spectrum：

```c
float gain[n_freqs];
const float *extra_r2 = aec_r2_on_audio_power_scale;

if (anr_process_gain(nr, error_spec, extra_r2, gain) != 0) {
    /* error */
}
```

AEC 的 `AecResContext.r2` 是 int16-square scale，直接交給 NR 前必須除以
`32768^2`；`pipelines/samples/se_anr_stage.h` 的 `se_anr_stage_gain()` 已正確處理，直接呼叫它，不要在 application 再複製一份。

### 4.4 Reset 與查詢

| API | 用途 |
|---|---|
| `anr_reset()` | 清除 noise/SPP/gain state；WOLA history 仍由 caller 自己清 |
| `anr_get_gain()` | 最近一 frame 的 gain；borrowed pointer |
| `anr_get_noise_psd()` | noise PSD；borrowed pointer |
| `anr_debug_status()` | initialized、mean/min gain、SPP、noise floor |

切 stream 時必須同時 reset NR 與 caller 的 analysis/OLA buffers，不能只 reset 其中一邊。

### 4.5 執行期改降噪強度

```c
int anr_reconfigure(AnrDenoiser *nr, const AnrConfig *target);
int anr_set_mode(AnrDenoiser *nr, AnrMode mode);
```

不重新配置任何記憶體，所以 grid（`sample_rate`／`frame_size`／`hop_size`／`fft_size`）
與兩個決定 pool 大小的欄位（`L`、`num_init_frames`）**必須等於實例當下的值**，
`target` 另外要完整通過 `anr_validate_config()`。回 `0` 或 `-1`；`-1` 時什麼都
不寫。**在 frame 之間呼叫、與 process 序列化；非 thread-safe。**

狀態是刻意保留的：追蹤中的 noise floor、MCRA 最小值追蹤環、SPP 歷史與 gain 平滑歷史
全部繼續跑。換強度不是重啟——要重啟請用 `anr_reset()`。

```c
/* A. 以 preset 建立的實例（含範例程式的降噪器）：直接換 mode */
if (anr_set_mode(nr, ANR_MODE_AGGRESSIVE) != 0) { /* 引數不合法或 target 被拒 */ }

/* B. 自己在 preset 之上改了欄位：重組整份組態，逐字交出去 */
AnrConfig target = anr_config_for_mode_grid(sr, fft_size, ANR_MODE_AGGRESSIVE);
target.g_min_db = -30.0f;                                     /* 你的覆寫 */
if (anr_reconfigure(nr, &target) != 0) { /* grid 不符或 target 無效 */ }
```

> **兩者不可互換。** `anr_reconfigure()` 逐字採用 `target`，不做 preset 查表、
> 不做 overlay 疊加。`anr_set_mode()` 組的是**裸的 canonical preset**：對在 preset
> 之上改過欄位的實例，它會把那些覆寫蓋回 preset 值（改過 `L` 或 `num_init_frames`
> 的實例則直接拒絕），這種實例要用 B。範例程式的降噪器
> （`mono_aec_nr_res/main.c` 等）就是 `anr_config_for_mode_grid()` 的 preset、沒有覆寫（最小值追蹤視窗 `L` 為函式庫預設），
> 用 A：`anr_set_mode(nr, mode)`。

### 4.6 參數分級：外部與內部

欄位在 `anr_types.h:56-110`。以 `anr_config_for_mode_grid(sample_rate, fft_size, mode)` 取得完整設定。

| 分級 | 欄位／函式 | 範圍與預設 | 何時設定 |
|---|---|---|---|
| 外部 | `sample_rate`、`fft_size`（`frame_size`／`hop_size` 是衍生值，不要獨立設） | 8000／16000／48000；合法格點 8k/128、8k/256、16k/256、16k/512、48k/1024 | 建立時，不可變 |
| 外部 | `AnrMode` | `MILD`／`MODERATE`／`BALANCED`（預設）／`AGGRESSIVE`；一次調 `g_min_db`、`q`、`xi_min_db`、`noise_over_subtraction` | 建立時，或執行期 `anr_set_mode()` |
| 進階外部 | 上列四個旋鈕單獨調 | 它們是 preset 聯合調出的操作點；`g_min_db` 在 `speech_protect_floor` 開啟（預設）時須 `<= speech_protect_floor_db`（-15）；改 `q` 會連動幀級語音閘 | 建立時；執行期用 `anr_reconfigure()`（整組逐字取代，不查 preset） |
| 內部 | 噪聲追蹤器（`alpha_s`、`alpha_d`、`alpha_p`、`delta_db`）、`alpha_xi`、增益平滑（`alpha_attack`、`alpha_decay`）、場景切換（`scene_change_*`、`broadband_threshold`）、低頻語音保護（`speech_protect_*`）、speech-aware 追蹤（`speech_aware_*`、`alpha_d_speech`、`noise_gate_*`）、預設關閉的實驗（`cross_band_*`、`dd_from_gmmse`、`makeup_*`） | 與 preset 聯合調過，四級共用 | 不建議開放 |
| 內部（決定 pool 版面） | `L`、`num_init_frames` | `anr_reconfigure()` 與 `anr_set_mode()` 不能改，要改就重新查大小並重建 | 建立時 |
| 內部（無作用） | `alpha_g` | 增益路徑上沒有作用（`anr_types.h:107`），調它不會有任何效果 | — |

函式面：公開 = `anr.h` 的函式與 `anr_types.h` 的 `anr_config_for_mode*()`、`anr_nr_mode_*()`；內部 = `noise_estimator.h`、
`spp_estimator.h`、`anr_internal.h`、`anr_apply_*_experiment()`。

---

## 5. audiolib library

### 5.1 Header 與主要 API

| Header | 功能 | 主要 API |
|---|---|---|
| `fft_wrapper.h` | RFFT/IRFFT 與 spectrum utilities | `fft_create/init/destroy`、`fft_forward/inverse`、`fft_power`、`fft_apply_gain` |
| `audio_resampler.h` | stateful rational resampler，1–8 ch | `audio_resampler_create/init/process/reset/destroy` |
| `hpf.h` | stateful 2nd-order HPF | `hpf_create/init/process/reset/destroy` |
| `wav_io.h` | host WAV I/O helper | `wav_open_*`、`wav_read_float`、`wav_write_float` |

以上皆在 `include/`（對外 API）。`utility/mem_align.h`（pool alignment/size helper，
`ALIGN16` 等）、`utility/simd_kernels.h` 與 `utility/fast_math.h` 是內部 header-only
DSP kernels，不是一般 application control API。若 application TU 直接 include，
必須和 library 使用相同 SIMD/FP flags。

### 5.2 FFT

```c
FftHandle *fft = fft_create(256);
Complex bins[129];
float time[256];

fft_forward(fft, time, bins);   /* time 與 bins 不可 alias */
fft_inverse(fft, bins, time);   /* bins 與 time 不可 alias */
fft_destroy(fft);
```

若 input 是可破壞的 scratch，可用 `fft_forward_scratch()`／
`fft_inverse_scratch()` 省掉 backend defensive copy；呼叫後 input 內容未定義。

Static pool 形式與其他 library 相同：

```c
size_t bytes = fft_get_mem_size(256);
FftHandle *fft = fft_init(aligned_pool, bytes, 256);
```

### 5.3 Resampler

```c
AudioResampler *r = audio_resampler_create(48000, 16000, 1);
int out_cap = audio_resampler_output_bound(r, in_frames);
int consumed = 0, produced = 0;

if (audio_resampler_process(r, in, in_frames, out, out_cap,
                            &consumed, &produced) != 0) {
    /* error */
}
```

Resampler 保留跨 block state。不能假設一次一定吃完 input；以 `consumed/produced` 推進
buffer。不同 rate 時 input/output 不可 overlap；相同 rate 是精確 memmove passthrough。

### 5.4 HPF

```c
Hpf *hpf = hpf_create(80.0f, sample_rate);

hpf_process(hpf, buf, n);                  /* in-place */
```

HPF 與 resampler 都有跨呼叫 state。stream boundary
要 reset，不要每個 hop 重新 create。

---

## 6. AI 模型 pre/post 類別

四個模型各有一個 pre/post 類別，把「呼叫端音訊 ↔ 加速器張量」之間的所有東西
（framing、window、FFT/IFFT/OLA、特徵前端、遞迴 ring 與狀態交易）收成一個物件：

| 模型 | Public header | 訊號輸入 |
|---|---|---|
| Align-ULCNet | `AIAEC/Align_ULCNet/ulcnet_prepost.h` | linear-AEC error + aligned far |
| DeepVQE-S | `AIAEC/DeepVQE_S/deepvqe_prepost.h` | **raw microphone** + far reference |
| DeepFilterNet2 | `AINR/DeepFilterNet2/dfn2_prepost.h` | 單路 noisy |
| DeepFilterNet3 | `AINR/DeepFilterNet3/dfn3_prepost.h` | 單路 noisy；raw 低頻 DF / 高頻 ERB |

每個類別都是**第三個 TU**：它組合的兩個檔案（`dfn3_process.c` + `dfn3_model_io.c`；`ulcnet_process.c` + `ulcnet_model_io.c`；
`aiaec_process.c` + `deepvqe_process.c`；`dfn2_process.c` + `dfn2_model_io.c`）必須維持各自
可獨立連結給自己的 parity 測試用，所以類別不能放進其中任何一個。

### 6.1 共通生命週期

四個類別的生命週期形狀相同，前綴為 `ulcnet_` / `deepvqe_` / `dfn2_` / `dfn3_`。
DFN3 的兩個 heads、layout 9、state 綁定、單 hop lookahead 與 API 範例見
[DFN3 模型頁](html/ainr_dfn3.html)；下面原有的 DFN2 專節仍描述其 cascade/alpha。

```text
_config_defaults(&cfg, io_mode[, delay_depth])
  -> _get_mem_size(&cfg, &req)
  -> _init(pool, req.bytes, &cfg)  或  _init_ex(pool, bytes, &cfg, &expected)
  -> 每-hop 序列（6.2）
  -> _reset(p)（stream boundary）
  -> _destroy(p)
```

- `io_mode` **init-time 固定**，因為它決定 pool 大小：`*_IO_TIME` 是 hop 進 hop 出
  （類別做 framing/window/OLA，transform 呼叫**呼叫端的** `FftHandle`），
  `*_IO_FREQ` 是頻譜進頻譜出、完全不做 transform，用來接在 AEC/BF seam 後面串接。
- `_get_mem_size()` 是 reject-first：NULL 引數、未知 `io_mode`、`delay_depth` 超出範圍、
  TIME 模式沒有可用的 `fft`/`window`（DFN2 另加：缺 ERB 矩陣、非有限的 `atten_lim_db`），
  一律回 `-1` 且 `*req` 一個 byte 都不寫。
- `_init_ex()` 是 **stale-pool gate**：固定 32-byte 的 `*PrepostMemReq`
  （`descriptor_version`、`layout_version`、`io_mode`、`build_flags_hash`（FNV-1a-32）、
  `alignment`、`reserved`、`bytes`）與本 build 不符就拒絕，而不是重新解讀一塊為別的組態
  配出來的 pool。`expected` 傳 NULL 時等同 `_init()`。
- `_create()` 是 get_mem_size + 對齊配置 + init，給 host 工具與測試；板端走 `_init()`。
  `_destroy()` 只釋放 `_create()` 配置的東西：對 `_init()` 實例它是**可重複呼叫的 no-op**
  （pool 永遠是 caller 的），對 `_create()` 實例呼叫第二次就是 double free。
- 借用而不擁有：`FftHandle`、window 表、DFN2 的 ERB 矩陣都是 caller-owned，類別從不建立或銷毀。
  組幀物件（`AcStft`／`AcIstft`）是 `*_IO_TIME` 的 pool 內物件，借用這些表；DFN2 的 `cfg.window` 為 NULL 時，
  由分析物件在 pool 內自帶 sqrt-Hann 表、合成物件借用同一張。
- `_reset()` 清空所有遞迴／ring／狀態、丟掉未結束的 frame、framing 與 clock 回 init 值；
  config、borrowed 指標與 pool 都不動。

### 6.2 每-hop 序列與回傳值合約

```c
int n = ulcnet_prepost_pre_process(p, err_hop, far_hop);  /* 需要幾次推論 */
if (n == 1) {
    UlcnetModelIoInputs  in;
    UlcnetModelIoOutputs out;
    ulcnet_prepost_frame_inputs(p, &in, &out);   /* 指向本實例 pool 的 view */
    /* 每個 out.*_out 綁到 in 的同名位址；先讀完所有 state 輸入、才寫輸出。
     * （DFN2、GTCRN 同樣就地綁定。） */
    if (accelerator_run(&in, &out) != 0 ||
        ulcnet_prepost_frame_commit(p) != 0)
        ulcnet_prepost_frame_skip(p);            /* 保住 framing 節奏 */
}
int written = 0;
ulcnet_prepost_post_process(p, out_hop, &written);
```

- **`_pre_process()` 回的是「這一拍需要的加速器呼叫次數」，不是布林。**
  Align-ULCNet 與 DeepVQE-S **從第 0 拍起恆為 1**（一 hop 進、一次推論、一 hop 出，兩個 io_mode 皆然）；
  DeepFilterNet2 **第 0 拍回 0**、之後恆 1，因為它的圖吃 `[t-1, t, t+1]`，第 0 拍沒有右鄰居，
  那一拍**不可以**呼叫加速器。
- `_frame_inputs()` 只把**非 state 的輸出**（heads：`output`／`taps`／`erb_mask`／`coefs`／`alpha`，
  Align-ULCNet 的 `output`）**預填 NaN**，所以加速器只寫一半會在 commit
  被抓到，而不是把上一框的值當成本框結果。**state 不預填**（它同時是下一框的輸入）。
  回 `0`，沒有開啟中的 frame 回 `-1`。
- `_frame_commit()` 是**純驗證**：heads 與 state 都是有限值就成功（CCM ring／compose／synthesis 照常前進，
  commit 本身不複製 state）。就地路徑遇到任何一處非有限值時 state 已被加速器覆寫、**無法回滾**，
  所以**所有遞迴 state 歸零**（冷啟動）、state generation 加一，framing／CCM ring 不前進、frame 維持開啟、回 `-1`（Align-ULCNet：key／value／logit 三個
  window 與 h_gru 共四個 state 全歸零；DeepFilterNet2：`h_gru` 與 pathway window 兩個遞迴張量；DeepVQE-S：16 個 state）。
  接著只有兩條路：`_frame_skip()`，或重新 `_frame_inputs()` 再從冷 state 跑一次加速器。
- **commit 前面一定要有一次 `_frame_inputs()`**（`prepared` 閂鎖，四個類別皆然）：
  沒跑過的加速器不能把沒被碰過的 buffer 當成結果送出。
- **遞迴 state 的綁定。** 每個 state 張量在 pool 裡只有一個 buffer，位址在實例壽命內固定（沒有交替的 bank、沒有指標交換）。
  - **Align-ULCNet、DeepVQE-S、DFN2、GTCRN：就地綁定，沒有複製路徑。** runtime 把每個 `*_out`（C 端欄位名 `*_next`）綁到
    `_frame_inputs()` 回傳的同名輸入位址，**先讀完**所有 state 輸入、**才寫**輸出；graph 回完整的下一個 window
    （位移在 graph 內），CPU 不搬、不複製。commit／validate 被拒時無法回滾：遞迴 state 從零重來、state generation 加一；
    回報失敗的 runtime 視為沒有寫入。
  - **RNNoise-ERB（複製路徑）**：runtime 把 `h1_out`／`h2_out`／`h3_out` 寫進自己擁有的 tensor，host 呼叫
    `rnnoise_model_state_inherit()` 複製進 state。inherit 是**先檢查再複製**：任何一處非有限值就一個都不複製、回 `-1`、
    state 與原來逐位元相同，這一幀視為失敗。指標等於目的位址的 tensor 視為已就地寫入，不複製也不由 inherit 檢查，
    所以同一個呼叫兩種綁定都適用。
  非有限值以外的「寫一半／混寫」都偵測不到。
- **state generation（Align-ULCNet、DeepVQE-S、DFN2、GTCRN）**：`ulcnet_model_io_state_generation()`、
  `ulcnet_prepost_state_generation()`、`ulcnet_accelerator_adapter_state_generation()`、
  `deepvqe_prepost_state_generation()`、`dfn2_prepost_state_generation()`、`dfn2_model_io_state_generation()`、
  `gtcrn_model_state_generation()` 回傳 C 端把 state 清零的次數：`_reset()` 與被拒絕的 `_frame_commit()` 各加一，
  新建實例的起始值：Align-ULCNet、DeepVQE-S 為 0，DFN2、GTCRN 為 1；成功的 commit、`_frame_skip()` 與再次呼叫被拒的 commit 都不動它；`NULL` 回 0，數值回繞於 2^32。
  定點 accelerator 的 state 緩衝在 C 清零時不會跟著變，板端 sample 比較這個值，變了就用量化後的 0 重填 state 輸入。
- **Align-ULCNet 的四個 state**（`key_history`／`value_history`／`logit_history`／`h_gru`
  ↔ 對應的 `*_out`）：輸入是完整 window、graph 回每個 state 的**完整下一個 window**，shape 與輸入完全相同：
  K/V `[1,32,D,TA]`（newest-first）取輸入 slot `0..D-2` 再把新一幀接在前面，logit `[1,32,5,D]`（oldest-first）
  取輸入 frame `1..4` 再把新的一列接在後面（輸入端 Slice 加 Concat，沒有 state 輸出由 Slice 產生），`h_gru`
  `[1,4,1,128]` 是兩顆 subband GRU 的 hidden 疊在一起；沒有 CPU 端的 ring helper。`output` 是學到的複數 mask
  （planar `[1,2,1,BINS]`），commit 驗證後以壓縮域 error 乘上它並逆冪。
- **Align-ULCNet 的 commit 有限性檢查只看「這一幀 graph 新寫的那一格」**：`output`、每個 channel 的 K/V slot 0
  與 logit 最後一幀，以及 `h_gru` 全部；較舊的格子與被丟掉的那一格不重查，留在那裡的非有限值會在下一幀進到 `output` 被擋下。
  DeepVQE-S 同理：每個 tap、`h_gru` 全部，以及每個 window 的最新一幀（attention ring 的 slot 0、conv／score window 的最後一幀）。
  `_frame_skip()` 不動任何 state（維持它當下的樣子；就地路徑視同回報失敗的 runtime 沒寫）。

拒絕情境整理：

| 呼叫 | 情境 | 回傳 |
|---|---|---|
| `_pre_process` / `_pre_process_freq` | 上一個 frame 還開著（既未 commit 也未 skip） | `-1`（一拍被拒絕，不會默默疊上去） |
| `_pre_process` / `_pre_process_freq` | 與實例的 `io_mode` 不符 | `-1` |
| `_frame_inputs` / `_frame_commit` / `_frame_skip` | 沒有開啟中的 frame | `-1` |
| `_frame_commit` | 沒有 `_frame_inputs()` 在前面 | `-1`，狀態不動 |
| `rnnoise_model_state_inherit` | 任何將被複製的 tensor 非有限、NULL、缺 tensor 或元素數不符 | `-1`，不複製任何東西，state 逐位元不變 |
| `dfn2_prepost_set_erb_matrices` / `_set_atten_lim` | frame 開著 | `-1` |

### 6.3 skip 政策：三個模型不一樣

`_frame_skip()` 是加速器跑失敗（或對齊邊界 reprime）時唯一正確的出口——它讓 framing
節奏繼續走，ring／CCM 不步進，遞迴 state **維持原樣**（複製路徑是 inherit 沒有動它；就地路徑則是加速器寫完的樣子，回報失敗的 runtime 視為沒有寫入，
所以 state 停在原處；寫了一半的 state 只有在非有限值時才偵測得到）。但「這一框輸出什麼」每個模型不同：

| 模型 | skip 輸出 | 理由 |
|---|---|---|
| Align-ULCNet | error 頻譜原樣穿過（identity） | 它的 stream 0 已經是 linear AEC 之後的殘差 |
| DeepFilterNet2 | unit ERB mask + alpha 0（identity） | `erb_inv` 是 partition of unity（每個 bin 的 band 權重和為 1），unit band mask 展開就是 unit bin gain——**精確**而非近似 |
| DeepVQE-S | **靜音（fail closed）** | 它的 stream 0 是**原始麥克風**，pass-through 會送出**完全未消除的回音**；`deepvqe_prepost_skip_policy_name()` 回 `"mute_fail_closed"` |

DeepVQE-S 的代價是 near-end 一框的缺口，上限就是一框；另一個選項的代價則是無上限的回音進遠端。
要 graceful degradation 的呼叫端得自己跑一個 canceller 並在 pipeline 層 cross-fade——
那不是這個類別的決定，它看不到有沒有那個 canceller。

### 6.4 Align-ULCNet：`ulcnet_prepost.h`

```c
int  ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, int io_mode, int delay_depth);
int  ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg, UlcnetPrepostMemReq *req);
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg);
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);
UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);
void ulcnet_prepost_destroy(UlcnetPrepost *p);
void ulcnet_prepost_reset(UlcnetPrepost *p);
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);
int  ulcnet_prepost_hop_size(const UlcnetPrepost *p);
int  ulcnet_prepost_num_bins(const UlcnetPrepost *p);
int  ulcnet_prepost_io_mode(const UlcnetPrepost *p);
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

int  ulcnet_prepost_pre_process(UlcnetPrepost *p, const float err_hop[ULCNET_HOP],
                                const float far_hop[ULCNET_HOP]);
int  ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                     const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                                     const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS]);
int  ulcnet_prepost_frame_inputs(UlcnetPrepost *p, UlcnetModelIoInputs *inputs,
                                 UlcnetModelIoOutputs *outputs);
int  ulcnet_prepost_frame_commit(UlcnetPrepost *p);
int  ulcnet_prepost_frame_skip(UlcnetPrepost *p);
int  ulcnet_prepost_post_process(UlcnetPrepost *p, float out_hop[ULCNET_HOP], int *written);
int  ulcnet_prepost_post_process_freq(UlcnetPrepost *p, float re[ULCNET_BINS], float im[ULCNET_BINS]);
```

- `cfg` 欄位：`io_mode`、`delay_depth`（D，範圍 `ULCNET_MODEL_IO_MIN_D`=2 到
  `ULCNET_MODEL_IO_MAX_D`=64）、`fft`、`window`（後兩者 `ULCNET_IO_TIME` 必填、borrowed，
  window 為 `ac_window_build()` 產出的 `ULCNET_N_FFT` 項；`ULCNET_IO_FREQ` 不用）。
- **error 與 far 必須是同一個輸入 hop**：類別內部不做任何 skew。上游 WOLA 造成的落差由呼叫端先對齊。
- `ULCNET_IO_FREQ` 收的是模型自己 framing 的未正規化 rfft（center=False，一 hop 一框）——
  AEC 的 `AecResContext.error_spec` 已經正好是這個 framing 慣例（但它是交錯的 `Complex`，
  要拆成 re/im 兩個陣列再傳）。
- 加速器邊界的 `UlcnetModelIoInputs`／`UlcnetModelIoOutputs` 定義在 `ulcnet_model_io.h`。
- 一個 build 只服務一個 grid：預設 16000/512，用 `-DULCNET_MODEL_IO_SR=48000
  -DULCNET_MODEL_IO_N_FFT=1024` 建另一個產品 grid；其餘組合由 `#error` 擋掉。

### 6.5 DeepVQE-S：`deepvqe_prepost.h`

生命週期與每-hop 函式和 6.4 逐一對應（前綴 `deepvqe_prepost_`，訊號引數為
`mic_hop`/`far_hop`、`mic_re/mic_im/far_re/far_im`，長度 `AIAEC_HOP`／`AIAEC_N_BINS`），
`frame_inputs()` 用的是本檔自己的 `DeepVqePrepostInputs`／`DeepVqePrepostOutputs`。額外 API：

```c
const char *deepvqe_prepost_state_name(int state_id);       /* graph input 名 */
const char *deepvqe_prepost_state_name_out(int state_id);   /* graph output 名 */
int    deepvqe_prepost_state_shape(int state_id, int delay_depth,
                                   int dims[DEEPVQE_STATE_MAX_RANK]);
size_t deepvqe_prepost_state_elements(int state_id, int delay_depth);
const char *deepvqe_prepost_skip_policy_name(void);
int  deepvqe_prepost_descriptor_default(int delay_depth, DeepVqePrepostDescriptor *descriptor);
int  deepvqe_prepost_descriptor_validate(const DeepVqePrepostDescriptor *descriptor);
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(const DeepVqePrepost *p);
```

- 圖邊界：兩個 power-law 壓縮後的 planar 訊號 input（`mic`/`far`，`[1,2,1,AIAEC_N_BINS]`，
  先 real 平面、再 imaginary 平面；壓縮 `x·|x|^(DEEPVQE_COMPRESSION_EXP−1)` 由
  `frame_inputs()` 在 host 端做，圖自第一個 conv 起；`DeepVqePrepostInputs.mic`/`.far` 即這兩個 tensor，
  CCM 用的 raw mic 頻譜另存於類別內）＋
  **16 個顯式 state**，順序就是 `DeepVqeStateId` 列舉（= exporter 的 `input_names[2:]` 逐字），
  輸出是 packed CCM taps
  `[1,1,BINS,DEEPVQE_TIME_ORDER*DEEPVQE_FREQ_TAPS*2]`（16 kHz grid 為
  `[1,1,257,18]`，末軸仍按 `[time][frequency][RI]` 排列）＋ 16 個 next state。
  按 index 綁的 adapter 必須用這個列舉，按名字綁的用 `_state_name()`。
- DeepVQE-S 回的是**每個 state 的完整下一個 window**（不是差量；shape 與輸入相同，全部 rank 4，`h_gru` 為
  `[1,1,1,192]`，conv window 4 幀、score window 5 幀），pool 裡每個 state 只有一份 buffer，runtime 把 `*_out` 就地綁定到輸入位址；
  commit 的有限性檢查只走每個 tap、`h_gru` 與各 window 的最新一幀，遇非有限值時整組 state 歸零、
  `deepvqe_prepost_state_generation()` 加一（`uint32_t deepvqe_prepost_state_generation(const DeepVqePrepost *p)`）。
- `descriptor_validate()` 拿 ONNX/JSON metadata 裡的 13 欄 `c_descriptor` 對本 build 的 ABI 逐欄比對，
  只有 `delay_depth` 是 export-time 部署參數、僅做範圍檢查；`DEEPVQE_PREPOST_LAYOUT_VERSION` = 4。
- D 範圍 `DEEPVQE_PREPOST_MIN_D`=1 到 `DEEPVQE_PREPOST_MAX_D`=256，出貨值
  `DEEPVQE_PREPOST_DEFAULT_D`=32（16 kHz 上的半秒搜尋範圍；D 是 export 期參數，圖 shape 與 C 傳入的 `delay_depth` 必須相同，descriptor 只檢查範圍，抓不到不一致）。
- grid 只有 16 kHz/512/256：`aiaec_process.h` 帶 `#error` 守衛，grid 換掉是編譯失敗而不是默默重新解讀張量。

### 6.6 DeepFilterNet2：`dfn2_prepost.h`

生命週期與每-hop 函式同樣對應（前綴 `dfn2_prepost_`，單路訊號 `in_hop`／`spec_re,spec_im`，
長度 `DFN2_HOP_LEN`／`DFN2_N_BINS`，`_config_defaults(cfg, io_mode)` **沒有** `delay_depth`）。
額外 API：

```c
int  dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);  /* = 2 */
int  dfn2_prepost_layout_version(const DFN2Prepost *p);
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);  /* init 後 1；每次 _reset 與被拒的 commit 加 1 */
int  dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd, const float *erb_inv);
int  dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);
int  dfn2_prepost_post_process(DFN2Prepost *p, float out_hop[DFN2_HOP_LEN], int *written);
int  dfn2_prepost_post_process_freq(DFN2Prepost *p, float re[DFN2_N_BINS], float im[DFN2_N_BINS],
                                    int *valid);
int  dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);
```

- `cfg` 欄位：`io_mode`、`fft`（`DFN2_IO_TIME` 必填）、`window`（可選，**會被複製**，
  NULL = 內建 sqrt-Hann，出貨路徑都用 NULL）、`erb_fwd`／`erb_inv`（**兩個模式都必填**，borrowed）、
  `atten_lim_db`（0 = 關閉）。
- 兩個 setter **只能在 hop 之間**呼叫，frame 開著時回 `-1`。要精確理解它買到什麼：
  這是 **per-hop atomicity，不是 per-source-frame consistency**——`DFN2_MASK_LOOKAHEAD` = 1，
  hop *t* 展開的 mask 屬於 source frame *t*−1，而圖的特徵視窗 `[t-1, t, t+1]` 橫跨三拍，
  所以一次被接受的 between-hop 換手仍會有幾拍同時踩到新舊矩陣。要 per-source-frame 一致，
  那就是一個 stream boundary：先 `_reset()`，再換。
- `_output_frame_index()` 給出最近一次輸出屬於哪一個 source frame（自 reset 起 0 起算），
  用來把輸出串流與其他同框時鐘的東西配對；還沒有輸出時回 `-1`。
- 輸出前兩拍是暖機：`_post_process()` 的 `*written` = 0（`out_hop` 仍然寫滿零），
  `_post_process_freq()` 的 `*valid` = 0。收尾 flush 要用全零 hop 再跑 2 拍，**加速器那兩拍也要照跑**。
- **`DFN2_IO_FREQ` 的頻譜是 `torch.stft(normalized=True)` 尺度**（乘了 1/√`DFN2_N_FFT` = 1/32），
  AIAEC 兩個模型交出的則是**未正規化**的 rfft。直接串接會剛好差 32 倍（過驅動或欠驅動），
  而且兩邊都偵測不到——都是量綱合法的 float 頻譜，ERB 特徵又是 EMA 正規化的。
  兩者也不同 grid（48 kHz/1024/512 對 16 kHz/512），中間本來就有 resampler，換算就放在那一層。
  `DFN2_ANALYSIS_SCALE` **不是**這個因子（它只作用在特徵分支的拷貝上）。

### 6.7 Pool 大小

以現行 build 量得（NEON 開關不影響數字），供規劃用；產品一律以 resolved config 呼叫 `_get_mem_size()`，
**不要把這裡的數字寫死**（§7 規則同樣適用）。單位 bytes。

| 類別 / grid | D | `*_IO_TIME` | `*_IO_FREQ` |
|---|---:|---:|---:|
| Align-ULCNet 16 kHz | 4 | 67,856 | 46,064 |
| Align-ULCNet 16 kHz | 8 | 97,040 | 75,248 |
| Align-ULCNet 16 kHz | 16 | 155,408 | 133,616 |
| Align-ULCNet 16 kHz | 32 | 272,144 | 250,352 |
| Align-ULCNet 16 kHz | 64 | 505,616 | 483,824 |
| Align-ULCNet 48 kHz | 4 | 130,320 | 87,024 |
| Align-ULCNet 48 kHz | 8 | 186,128 | 142,832 |
| Align-ULCNet 48 kHz | 16 | 297,744 | 254,448 |
| Align-ULCNet 48 kHz | 32 | 520,976 | 477,680 |
| Align-ULCNet 48 kHz | 64 | 967,440 | 924,144 |
| DeepVQE-S 16 kHz | 32 | 637,072 | 615,280 |
| DeepFilterNet2 48 kHz | — | 246,320 | 219,536 |

DeepVQE-S 的量體由 16 個 state 張量（每個一份 buffer）主宰。DeepFilterNet2 的 FREQ 比 TIME 小 26,784 bytes：
`DFN2State` 不內嵌 analysis/window/synthesis 緩衝，TIME 另 carve 分析與合成兩個 audiolib 物件
（16,480 B 與 12,384 B），FREQ 只多 carve 雙輸入入口的第二組 staging（`apply_re`/`apply_im`）。

### 6.8 Framing：一 hop 一框的 analysis

兩個 AIAEC 類別在 TIME 模式下的 analysis 是 audiolib `audio_stft.h` 的 `AcStft`，`center` = 0：

```c
void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft, const float *window, int center);
int  ac_stft_push_planar(AcStft *s, const float *hop_in, float *re, float *im);
```

center=False，**從第一次 push 起就是一 push 一框**（恆回 1），與 AEC/BF seam 的頻譜同一個慣例，
所以只有一路頻譜在手的呼叫端可以用它補另一路而兩路保持 hop 對齊。它與 `center` = 1（center=True，
第一拍 0 框、第二拍 2 框）對同一個 hop 產出的**最後一框 bit-identical**，差別只在 centered 版本多吐一個 reflect-prefix 開頭框。
DeepVQE-S 在 AIAEC 16 kHz／512 邊界上自建同組態的物件。

> **同一個 `AcStft` 不可混用兩種 `center`**：`center` 在建立時固定，兩種排程不共用物件。


### 6.9 Pipeline 層：頻域 stage 串接與 NN 的 begin / nn_begin / nn_end / end

四個 pipeline 範例把三個 library 與這些類別串成一條鏈，並在 `pipelines/samples/` 示範；接縫是
`pipelines/common/se_stage.h` 的 `SeSpecFrame`（借用視圖，有效至其產生者下一次執行；任何 stage 都不得寫入）。
增益類 stage 回傳 `const float*`（`se_res_stage_gain()`、`se_anr_stage_gain()`；RES gain 也可直接讀
`frame.g_res_ctx`），各範例 `main` 在每個 hop 明寫 `sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill`，再 `ac_istft_push`；
沒有函式指標表，也沒有把這幾步包起來的共用物件。

初始化也是各自為政：每個函式庫（`Aec`、`AecRes`（4ch 的範例自己建，單元與核心都不含）、`AnrDenoiser`、`AcIstft`、`UlcnetNnStage`、`DfnResStage`／`DfnRateBridge`、
4ch 單元）有自己的大小查詢與 `init`，整合者加總大小並切塊；記憶體和（每個函式庫一行）見
[`pipelines/README.md`](../pipelines/README.md)「Memory Budget」與 [`integration_example.html`](html/integration_example.html)。

跑 NN 的應用把一個 hop 拆成四個呼叫，加速器夾在中間，模型 config 的 `infer` 回呼因此可以不設：

```c
/* DFN2：samples/dfn_hop.h；ULCNet：samples/ulcnet_nn_stage.h（begin 的引數不同） */
dfn_hop_begin(&hop, frame.e, P);                         /* P = E·G_res 已由範例 main 先形成 */
while (dfn_hop_nn_begin(&hop, &job) > 0) {               /* 1 = 有一幀；0 = 沒有（warm-up／reprime） */
    dfn_hop_nn_end(&hop, run_npu(&job));                 /* rc != 0 或輸出非有限 ⇒ identity（fail-open） */
}
dfn_hop_end(&hop, out);                                  /* 合成剛好一個 hop */
```

`job` 是 `UlcnetNnJob`（ULCNet）或 `DfnResStageJob`（DFN2，內含本章 `DFN2PrepostInputs`／`Outputs`）；
輸出預填 NaN，必須寫滿。DFN 應用經 rate bridge 時一個 native hop 可能開 0..2 個 48 kHz 幀，所以 `nn_begin` 放在迴圈裡。
4 聲道 Align-ULCNet 若要用同一 hop 的 lane 頻譜算出的 mask 與 VAD（例如 masker 讀 lane 0），把單元的 hop 拆成
`audio_pipeline_4ch_process_front()`（回傳唯讀 lane 頻譜）與 `audio_pipeline_4ch_process_beam(p, &front, vad_raw, vad_out, mask, &beam, &info)`
（`mask` 可為 NULL）；`info` 回 DOA（rad，NaN = 無）、BF 適應旗標與 `info.delay`。`audio_pipeline_4ch_process_with_activity(v)` 等於
`process_front` 加 `process_beam(front, v, v, NULL)`；`ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {delay.changed, delay.solid})` 接在後面。任何一步被拒，整合者把單元與 stage 一起 reset。
沒被取走或沒被關閉的幀走 identity，輸出時序不隨模型健康度改變。範例與記憶體規則見
`pipelines/README.md` 的 Call flow 與 [`integration_example.html`](html/integration_example.html)。

### 6.10 各函式庫各自初始化：記憶體加總與單元的公開面

沒有「整條 pipeline 的 `get_mem_requirements()`」：每個函式庫查自己的大小，整合者加總（每塊 16-byte 對齊）。
下表是本 checkout、`BACKEND=ne10`、`SIMD=1`、16 kHz／256（預設格點）的實測值；換格點、backend、編譯選項都會變，
一律用各自的查詢重量。

| 函式庫物件 | 大小查詢 | 16 kHz／256 | 備註 |
|---|---|---:|---|
| `Aec`（mono，MATCHED n=5） | `aec_get_mem_size()` | 364,416 | 48 kHz／1024：1,156,544 |
| `AnrDenoiser`（函式庫 preset） | `anr_get_mem_size()` | 45,344 | 範例程式有 NR 時 |
| `AcIstft`（合成） | `ac_istft_get_mem_size()` | 11,344 | 範例程式有合成時 |
| `AecRes`（4ch 後波束 RES） | `aec_res_get_mem_size()` | 7,904 | 單元與核心的 pool 都不含；4ch 的範例自己建一個（48 kHz／1024：29,408） |
| 增益／頻譜暫存 | 範例程式自己算 | 2,624 | mono 為三個 `n_freqs` float 向量 + 一個複數向量，各自 `ALIGN16`；4ch 範例為兩個向量 + 一個複數向量（2,096） |
| 4ch 前端 `FourAecNrRes` | `four_aec_nr_res_get_mem_requirements()` | 906,560 | 共用 aligner + 四條線性 AEC + fuse 緩衝 |
| 4ch 單元 `AudioPipeline4Ch` | `audio_pipeline_4ch_get_mem_requirements()` | 1,698,864 | 前端 + SRP-PHAT + BF（不含 RES）；`core.enable_post = 0` 時 1,693,648 |
| ULCNet stage | `ulcnet_nn_stage_mem_size(0)` | 36,400 | 一個建置一組格點（48 kHz：71,728） |
| DFN2 stage／rate bridge | `dfn_res_stage_get_mem_requirements()`／`dfn_rate_bridge_get_mem_requirements()` | 231,984／349,888 | bridge 內含 stage；16 kHz 走 bridge，48 kHz 直接用 stage 加呼叫端的 `AcIstft` |

mono 的 16 kHz／256 合計 = `ALIGN16(364,416) + 45,344 + 11,344 + 2,624 = 423,728`（AEC + NR + 合成 + 暫存）；4ch 單元加 RES／NR／合成／暫存 = `1,698,864 + (7,904 + 45,344 + 11,344 + 2,096) = 1,765,552`（`audio_pipeline_4ch_static --print-mem-size` 的 `Total:`）。

**單元的公開面**（`audio_pipeline_4ch.h`）：`default_config`、`get_mem_requirements`、`init`／`init_ex`／`create`、
`process_front`／`process_beam`／`process_with_activity`／`process`、`fuse`（可選）、`reset`／`destroy`、
形狀與生效值 accessor、`get_core`、`last_core_timing`。輸出是借用的 `AudioPipeline4ChBeam`
（BF 後頻譜與權重），`fuse()` 之後另有 `SeSpecFrame`（`g_res_ctx` 為 NULL）；RES、NR、comfort noise、合成是你的後級。

**串接順序**（4ch，一 hop）：單元 `process_*` → `audio_pipeline_4ch_fuse()`（`frame.g_res_ctx` 為 NULL）→ `se_res_stage_gain()` → `se_anr_stage_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`。**mono**：`aec_process_context()` → `se_frame_from_aec()`（`g_res_ctx` 是 `Aec` 自己的 RES gain）→ 同上。
失敗時重置單元（或 `Aec`）、4ch 的 `AecRes`、降噪器、合成與 comfort-noise RNG（以及 NN stage）。

---

## 7. Static-pool 共通規則

三個 library 都遵守相同模式：

```text
config
  -> get_mem_size()/get_mem_requirements()
  -> caller 配置至少 16-byte aligned 的 pool
  -> init(pool, bytes, config)
  -> process...（不可 malloc）
  -> reset（需要時）
  -> destroy（不釋放 caller pool）
  -> caller 釋放 pool
```

規則：

- pool 大小必須由**同一份 resolved config**查詢；查完後不可再改 grid/preset/delay 欄位。
- 不要共用同一塊可寫 pool 給兩個 live instance。
- `_init()` 回傳 NULL 就停止，不可退回較小尺寸或默默改 config。
- static path 的 destroy 是 no-op 或只做狀態清理；pool ownership 永遠在 caller。
- 沒有「整條 pipeline」的 aggregate sizing API：mono 的每個函式庫各自查大小、整合者加總（§6.10）；4ch 單元的
  `audio_pipeline_4ch_get_mem_requirements()` 只含單元自己，RES（`AecRes`）、NR、合成與 NN stage 另計。

---

## 8. 建置、SIMD 與連結

從 `SE/` 根目錄做參考建置（`libaudio_common.a` 在 `audiolib/common`；`audiolib` 根目錄的 `lib` 另建 `libspatial_common.a`、
`libdoa.a`、`libbf.a`、`libvad.a`，只有 4ch 單元需要）：

```bash
make -C audiolib/common lib BACKEND=ne10 SIMD=0
make -C AEC/c_impl      lib BACKEND=ne10 SIMD=0
make -C NR/c_impl       lib BACKEND=ne10 SIMD=0
```

ARM NEON 產品建置使用相同旗標：

```bash
make -C audiolib/common lib BACKEND=ne10 SIMD=1
make -C AEC/c_impl      lib BACKEND=ne10 SIMD=1
make -C NR/c_impl       lib BACKEND=ne10 SIMD=1
```

不要混用不同 `BACKEND`／`SIMD` 產物。archive 路徑由 build flags hash 決定，使用各 repo
的 `make -s ... print-lib-path` 取得，不要自行拼 `bin/` 路徑。

模型 pre/post 類別（§6）另有兩個 archive，同樣是 configuration-keyed：

```bash
make -C AIAEC lib ; make -C AIAEC -s print-lib-path   # libaiaec_prepost.a
make -C AINR  lib ; make -C AINR  -s print-lib-path   # libainr_prepost.a
```

兩者都落在自己 repo 的 `build/simd<N>-<sig>/` 下（`<sig>` 是各自 Makefile 的組態簽章 cksum：
AIAEC 蓋 `CC`/`AR`/`CFLAGS`/`CPPFLAGS`/`SIMD`/`WERROR`，AINR 蓋 `CC`/`AR`/`CFLAGS`/`LDFLAGS`/
`SIMD`/`SIMD_CPPFLAGS`/`BACKEND`/`WERROR`/`COMMON_LIB`；每個目錄另存 `config.manifest` 擋簽章碰撞），
所以路徑一律用 `print-lib-path` 取得。成員：`libaiaec_prepost.a` = `ulcnet_process`、
`ulcnet_model_io`、`ulcnet_prepost`、`ulcnet_accelerator_adapter`、`aiaec_process`、
`deepvqe_process`、`deepvqe_prepost`；`libainr_prepost.a` = `dfn2_process`、`dfn2_model_io`、
`dfn2_prepost`、`dfn3_process`、`dfn3_model_io`、`dfn3_prepost`、`gtcrn_process`。RNNoise-ERB **不在**任何 archive 裡（自己的 Makefile，
要用的板子把它的來源檔直接加進自己的 build）。`WERROR=1` 是 opt-in，且已納入組態簽章。

消費端至少要帶：

```text
-std=gnu99
-ffp-contract=off
-fno-math-errno
-I<AEC>/c_impl/include
-I<NR>/c_impl/include
-I<audiolib>/common/include
<libaec.a> <libanr.a> <libaudio_common.a> -lm
```

`SIMD_KERNELS_FORCE_SCALAR` 必須在 library 與 include header-only kernels 的 application TU
保持一致。不要使用 `-Ofast`／`-ffast-math`；它們會改變遞迴 DSP 數值與 parity。

---

## 9. 最小整合檢查清單

- [ ] sample rate、frame/FFT、hop 是同一個合法 grid。
- [ ] mic/ref 同 sample rate、同 clock；AEC ref 是實際 speaker render reference。
- [ ] 每次 AEC 恰好傳一個 hop。
- [ ] NR 前後使用 periodic sqrt-Hann 50% WOLA，且 frame 等於 FFT。
- [ ] 所有 pool 以 resolved config 查 size，base 至少 16-byte aligned。
- [ ] process hot path 沒有 malloc/free/file I/O/logging。
- [ ] stream boundary 同時 reset DSP instance 與 caller-owned ring/WOLA state。
- [ ] borrowed context/gain pointer 沒有跨下一個 process 呼叫保存。
- [ ] application 與 static libraries 使用相同 backend、SIMD、FP flags。
- [ ] 最終 float-to-PCM 前做 saturating conversion；需要 limiter 時只放在整條 chain 最後。
- [ ] 完整 AEC+NR/4ch 應用優先跑 pipeline tests，而不只跑各 library unit test。
