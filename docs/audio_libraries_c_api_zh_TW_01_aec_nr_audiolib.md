[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW.md) ｜ [下一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) <!-- split-nav -->
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
> pipeline 層的對應入口見 [4ch 核心整合手冊 §3.2](integration_4ch_core_zh_TW_02_lifecycle.md#32-執行期強度控制)
> 與 [mono 整合手冊 §3.2](integration_mono_zh_TW_02_lifecycle_config_errors.md#32-執行期強度控制)。

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
| 內部 | 步進與不確定度（`mu`、`delta`、`kalman_q_*`、`shadow_mu_*`、`epc_mu_floor`）、偵測門檻（`epc_*`、`ne_recent_*`、`saturation_threshold`、`filter_misadjustment_scale_*`）、10 ms 格點撰寫的時間欄位（`warmup_frames` 等）、延遲獲取策略（`delay_acquire_*`、`max_delay_ms`、`delay_buffer_ms`）、DT 政策與 floor（`dt_aware_*`、`min_gain_*`）、預設關閉的實驗（`live_observer_*`、`cng_n2_windowed_psd`）、衍生值（`n_partitions`、`enable_delay_est`） | 驗證器只擋崩壞，不擋音質 | 不建議開放；理由見 [mono 手冊 §0.3](integration_mono_zh_TW_01_overview_link_quickstart.md#03-參數面外部與內部) |

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

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW.md) ｜ [下一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) <!-- split-nav -->
