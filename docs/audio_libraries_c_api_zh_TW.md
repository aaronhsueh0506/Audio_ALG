# AEC、NR、audio_common C API 整合速查

本文件是給 application／board integration 使用者的**短版入口**，回答三件事：

1. 應該 include 哪些 public header；
2. init、process、reset、destroy 的正確順序；
3. 哪些工作由 library 負責，哪些必須由呼叫端負責。

完整欄位、錯誤語意與演算法說明仍以各 library 的 canonical manual 為準：

- [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md)
- [NR C 使用手冊](../lib/nr/docs/c_user_manual_zh_TW.md)
- [audio_common C 使用手冊](../../audio_common/docs/c_user_manual_zh_TW.md)
- [Audio_ALG mono／4ch pipeline 使用手冊](c_user_manual_zh_TW.md)

若本文件與 header 不一致，以目前 pin 住的 public header 與測試為準。

---

## 1. 先決定使用層級

| 需求 | 建議入口 |
|---|---|
| 完整 mono `AEC + NR + RES` | `pipelines/mono_aec_nr_res/audio_pipeline.h` |
| 完整 4ch `4×linear AEC -> BF -> mono NR+RES` | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` |
| 4ch linear AEC 與外部 beamformer seam | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` |
| 只做單路 AEC，或自行組合 NN／其他 post-filter | `aec.h` |
| 已有 STFT/WOLA，只需要頻域 NR | `mmse_lsa_denoiser.h` |
| 只需要 FFT、resample、HPF、pre-gain | `audio_common/include/*.h` |

產品 application 若要的是既有完整 flow，優先呼叫 pipeline API；不要在 application
重新複製 AEC/NR gain fusion、window、OLA 或 CNG 邏輯。只有要做不同演算法組合時，才直接組
底層 library。

---

## 2. 共通信號契約

### 2.1 產品 grid

| sample rate | frame = FFT | hop | bins | 狀態 |
|---:|---:|---:|---:|---|
| 16 kHz | 256 | 128 | 129 | 預設、低延遲 |
| 16 kHz | 512 | 256 | 257 | 可選 |
| 48 kHz | 1024 | 512 | 513 | 48 kHz 產品 grid |

AEC 與 NR 另保留 8 kHz legacy grid，但 4ch product pipeline 不接受它。不要只傳
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

只有自行在 AEC 外建立 post-beam RES 時，才額外使用：

```c
#include "suppression_gain.h"
#include "aec3_balanced_config.h"
```

`pbfdkf.h`、`delay_aec3.h`、`aec3_post.h` 等是內部 header，不是 application API。

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
far 時，`aec_process_context_shared_far()` 可讓其他 lane 借用已算好的 far FFT；它有嚴格
same-hop/same-reference 前提，通常只應由 4ch wrapper 使用。

render/capture 分執行緒時使用 SPSC streaming API：

```c
aec_analyze_render(aec, ref_hop);       /* 唯一 producer thread */
aec_process_capture(aec, mic_hop, out); /* 唯一 consumer thread */
```

單一 instance 只允許一個 render producer 與一個 capture consumer。reset/destroy 前必須
先讓兩邊停止；不要把 offline `aec_process()` 與 streaming API 同時呼叫。

### 3.4 Reset、context 與診斷

| API | 用途 |
|---|---|
| `aec_reset()` | stream/session boundary；保留 config，清除 DSP state |
| `aec_get_linear_context()` | formed linear output、aligned far、delay state/generation |
| `aec_get_res_context()` | error/near/echo spectrum、R2、RES gain、CNG context |
| `aec_debug_status()` | delay、ERLE、convergence、power、duty-cycle snapshot |
| `aec_apply_external_realign()` | 僅 `EXTERNAL_ALIGNED` mode；通知外部 alignment 已改變 |

`AecLinearContext`／`AecResContext` 是 read-only borrowed views，不可保存其 pointer 跨 hop。

### 3.5 執行期改抑制強度

```c
int aec_set_preset(Aec *aec, AecPreset preset, float ramp_ms);
```

三個 preset 只差 `min_gain_floor_far_active_db`，而該欄位是抑制器每 hop 只讀一次的
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

自己在 beamformer 之後跑 RES 的整合（4ch 就是這樣）改用底層 primitive，因為那個
共用抑制器不屬於任何 `Aec`：

```c
#include "suppression_gain.h"

int suppression_gain_set_split_floor_far_active_db(SuppressionGain *sg,
                                                   float db, float ramp_ms);
```

`db` 是與 `AecConfig.min_gain_floor_far_active_db` 相同的 dB-power 量，以**驗證器
自己的**範圍 `[-300, 50]` 檢查——執行期不可能裝進 init 當初會拒絕的值。

> **量測前務必知道**：far-active 地板只在 **far-active 且非 double-talk** 的 hop 生效
> （DT 地板三個 preset 完全相同，far-active latch 觸發前用的是 far-silent 地板），
> 而同一個 gain 還決定 comfort noise 量。所以整段錄音的平均值移動幅度會小於 dB 落差
> 所暗示的量，只量 echo／degradation 的 A/B 會把 CNG 的變化錯記到別的機制。請在
> echo 對齊或 degradation 對齊的條件下比較，並實際試聽。
>
> 4ch 產品另有一條：四條 lane 以 `spatial_linear_context` 建立，從不走到自己的
> suppression-gain 路徑，對 lane 呼叫 `aec_set_preset()` 依建構方式即為無效操作。
> 請改用 `four_aec_nr_res_set_aec_preset()`，它針對的是那個共用的 post 級抑制器。
> pipeline 層的對應入口見 [4ch 核心整合手冊 §3.2](integration_4ch_core_zh_TW.md)
> 與 [mono 整合手冊 §3.2](integration_mono_zh_TW.md)。

---

## 4. NR library

### 4.1 Public header 與責任邊界

```c
#include "mmse_lsa_denoiser.h" /* 已 include mmse_lsa_types.h/fft_wrapper.h */
```

NR 是 mono frequency-domain MMSE-LSA。呼叫端必須準備正確 WOLA/STFT；若 application
手上只有 PCM，直接使用 Audio_ALG pipeline，或照 NR canonical manual 的 `NrStream`
範例實作。

### 4.2 Config 與建立

```c
MmseLsaConfig cfg = mmse_lsa_config_for_mode_grid(
    16000, 256, MMSE_LSA_NR_BALANCED);

if (!mmse_lsa_validate_config(&cfg)) {
    /* 不支援的 grid 或參數 */
}
```

只傳 sample rate 的 `mmse_lsa_config_for_mode()` 會選該 rate 的預設 grid；16 kHz 要選
512/256 時，應明確使用 `_for_mode_grid()`。

Heap 路徑：

```c
MmseLsaDenoiser *nr = mmse_lsa_create(&cfg);
/* ... process ... */
mmse_lsa_destroy(nr);
```

Static pool 路徑：

```c
size_t bytes = mmse_lsa_get_mem_size(&cfg);
void *pool = board_alloc_aligned(16, bytes);
MmseLsaDenoiser *nr = pool ? mmse_lsa_init(pool, bytes, &cfg) : NULL;

/* ... process ... */
mmse_lsa_destroy(nr);  /* 不釋放 caller pool */
board_free(pool);
```

### 4.3 每-frame 頻域入口

直接套 NR gain，可 in-place：

```c
if (mmse_lsa_process(nr, spectrum, spectrum) != 0) {
    /* invalid handle/buffer */
}
```

AEC + NR + RES gain fusion 應只算 gain、不先改 spectrum：

```c
float gain[n_freqs];
const float *extra_r2 = aec_r2_on_audio_power_scale;

if (mmse_lsa_process_gain(nr, error_spec, extra_r2, gain) != 0) {
    /* error */
}
```

AEC 的 `AecResContext.r2` 是 int16-square scale，直接交給 NR 前必須除以
`32768^2`；現有 Audio_ALG pipeline 已正確處理，不要在 application 再複製一份。

### 4.4 Reset 與查詢

| API | 用途 |
|---|---|
| `mmse_lsa_reset()` | 清除 noise/SPP/gain state；WOLA history 仍由 caller 自己清 |
| `mmse_lsa_get_gain()` | 最近一 frame 的 gain；borrowed pointer |
| `mmse_lsa_get_spp()` | 最近一 frame 的 SPP；borrowed pointer |
| `mmse_lsa_get_noise_psd()` | noise PSD；borrowed pointer |
| `mmse_lsa_debug_status()` | initialized、mean/min gain、SPP、noise floor |

切 stream 時必須同時 reset NR 與 caller 的 analysis/OLA buffers，不能只 reset 其中一邊。

### 4.5 執行期改降噪強度

```c
int mmse_lsa_reconfigure(MmseLsaDenoiser *nr, const MmseLsaConfig *target);
int mmse_lsa_set_mode(MmseLsaDenoiser *nr, MmseLsaNrMode mode);
```

不重新配置任何記憶體，所以 grid（`sample_rate`／`frame_size`／`hop_size`／`fft_size`）
與兩個決定 pool 大小的欄位（`L`、`num_init_frames`）**必須等於實例當下的值**，
`target` 另外要完整通過 `mmse_lsa_validate_config()`。回 `0` 或 `-1`；`-1` 時什麼都
不寫。**在 frame 之間呼叫、與 process 序列化；非 thread-safe。**

狀態是刻意保留的：追蹤中的 noise floor、MCRA 最小值追蹤環、SPP 歷史與 gain 平滑歷史
全部繼續跑。換強度不是重啟——要重啟請用 `mmse_lsa_reset()`。

```c
/* A. standalone：直接換 mode。stationary overlay 會被重新套上 */
if (mmse_lsa_set_mode(nr, MMSE_LSA_NR_AGGRESSIVE) != 0) { /* 引數不合法或 target 被拒 */ }

/* B. 自己在 preset 之上疊了覆寫：重組整份組態，逐字交出去 */
MmseLsaConfig target = mmse_lsa_config_for_mode_grid(sr, fft_size, MMSE_LSA_NR_AGGRESSIVE);
target.broadband_threshold = 0.8f;                            /* 我方覆寫 */
target.L = mmse_lsa_retime_frames(150, sr, target.hop_size);  /* 我方覆寫 */
target.alpha_decay = target.alpha_g;                          /* 我方覆寫 */
if (mmse_lsa_reconfigure(nr, &target) != 0) { /* grid 不符或 target 無效 */ }
```

> **兩者不可互換。** `mmse_lsa_reconfigure()` 逐字採用 `target`，不做 preset 查表、
> 不做 overlay 疊加。`mmse_lsa_set_mode()` 組的是**裸的 canonical preset**，只適用於
> standalone。兩條出貨 pipeline 的 NR 組態都是「canonical preset **加上**自己的覆寫」
> （`broadband_threshold`、`L`、`alpha_decay`），把 canonical preset 交給
> `mmse_lsa_set_mode()` 在這種實例上會被**拒絕**（它的 `L` 不同）——這正是
> `audio_pipeline_set_nr_mode()` / `four_aec_nr_res_set_nr_mode()` 各自重組完整組態
> 再呼叫 `mmse_lsa_reconfigure()` 的原因。application 不要繞過 pipeline 的 setter 去
> 呼叫便利包裝。

---

## 5. audio_common library

### 5.1 Header 與主要 API

| Header | 功能 | 主要 API |
|---|---|---|
| `fft_wrapper.h` | RFFT/IRFFT 與 spectrum utilities | `fft_create/init/destroy`、`fft_forward/inverse`、`fft_power`、`fft_apply_gain` |
| `audio_resampler.h` | stateful rational resampler，1–8 ch | `audio_resampler_create/init/process/reset/destroy` |
| `hpf.h` | stateful 2nd-order HPF | `hpf_create/init/process/reset/destroy` |
| `audio_pre_gain.h` | dB pre-gain，不 clipping | `audio_pre_gain_create/init/set_db/process/destroy` |
| `wav_io.h` | host WAV I/O helper | `wav_open_*`、`wav_read_float`、`wav_write_float` |
| `mem_align.h` | pool alignment/size helper | `ALIGN16` 等 |

`simd_kernels.h` 與 `fast_math.h` 是 header-only DSP kernels，不是一般 application
control API。若 application TU 直接 include，必須和 library 使用相同 SIMD/FP flags。

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

### 5.4 HPF 與 pre-gain

```c
Hpf *hpf = hpf_create(80.0f, sample_rate);
AudioPreGain *gain = audio_pre_gain_create(-6.0f);

audio_pre_gain_process(gain, buf, buf, n); /* in-place 可用 */
hpf_process(hpf, buf, n);                  /* in-place */
```

`audio_pre_gain_process()` 不 clipping；HPF 與 resampler 都有跨呼叫 state。stream boundary
要 reset，不要每個 hop 重新 create。

---

## 6. Static-pool 共通規則

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
- 完整 mono/4ch pipeline 另有自己的 aggregate sizing API，優先使用它，不要手算各 module
  byte 數再相加。

---

## 7. 建置、SIMD 與連結

從 `SE/` 根目錄做參考建置：

```bash
make -C audio_common lib BACKEND=kiss SIMD=0
make -C AEC/c_impl  lib BACKEND=kiss SIMD=0
make -C NR/c_impl   lib BACKEND=kiss SIMD=0
```

ARM NEON 產品建置使用相同旗標：

```bash
make -C audio_common lib BACKEND=ne10 SIMD=1
make -C AEC/c_impl  lib BACKEND=ne10 SIMD=1
make -C NR/c_impl   lib BACKEND=ne10 SIMD=1
```

不要混用不同 `BACKEND`／`SIMD` 產物。archive 路徑由 build flags hash 決定，使用各 repo
的 `make -s ... print-lib-path` 取得，不要自行拼 `bin/` 路徑。

消費端至少要帶：

```text
-std=gnu99
-ffp-contract=off
-I<AEC>/c_impl/include
-I<NR>/c_impl/include
-I<audio_common>/include
<libaec.a> <libmmse_lsa.a> <libaudio_common.a> -lm
```

`SIMD_KERNELS_FORCE_SCALAR` 必須在 library 與 include header-only kernels 的 application TU
保持一致。不要使用 `-Ofast`／`-ffast-math`；它們會改變遞迴 DSP 數值與 parity。

---

## 8. 最小整合檢查清單

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

