# AEC、NR、audio_common C API 整合速查

本文件是給 application／board integration 使用者的**短版入口**，涵蓋 AEC、NR、audio_common
三個 library，以及三個 AI 模型的 pre/post 類別（§6），回答三件事：

1. 應該 include 哪些 public header；
2. init、process、reset、destroy 的正確順序；
3. 哪些工作由 library 負責，哪些必須由呼叫端負責。

完整欄位、錯誤語意與演算法說明仍以各 library 的 canonical manual 為準：

- [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md)
- [NR C 使用手冊](../lib/nr/docs/c_user_manual_zh_TW.md)
- [audio_common C 使用手冊](../../audio_common/docs/c_user_manual_zh_TW.md)
- [Audio_ALG mono／4ch pipeline 使用手冊](c_user_manual_zh_TW.md)
- 模型側的圖邊界、I/O 表與延遲：`docs/html/` 的各模型頁（`ainr_dfn2.html` 等）

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
| Align-ULCNet 加速器 handoff（hop 或頻譜） | `AIAEC/Align_ULCNet/ulcnet_prepost.h` |
| DeepVQE-S 加速器 handoff | `AIAEC/DeepVQE_S/deepvqe_prepost.h` |
| DeepFilterNet2 加速器 handoff | `AINR/DeepFilterNet2/dfn2_prepost.h` |

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
`enable_res=0` 搭 `return_res_context=1` 時 formed seam 多一個 capture 候選，**不保證每個 hop
都是誤差**，條件與後果見 [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md) §8.1。

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

## 6. AI 模型 pre/post 類別

三個模型各有一個 pre/post 類別，把「呼叫端音訊 ↔ 加速器張量」之間的所有東西
（framing、window、FFT/IFFT/OLA、特徵前端、遞迴 ring 與狀態交易）收成一個物件：

| 模型 | Public header | 訊號輸入 |
|---|---|---|
| Align-ULCNet | `AIAEC/Align_ULCNet/ulcnet_prepost.h` | linear-AEC error + aligned far |
| DeepVQE-S | `AIAEC/DeepVQE_S/deepvqe_prepost.h` | **raw microphone** + far reference |
| DeepFilterNet2 | `AINR/DeepFilterNet2/dfn2_prepost.h` | 單路 noisy |

每個類別都是**第三個 TU**：它組合的兩個檔案（`ulcnet_process.c` + `ulcnet_model_io.c`；
`aiaec_process.c` + `deepvqe_process.c`；`dfn2_process.c` + `dfn2_model_io.c`）必須維持各自
可獨立連結給自己的 parity 測試用，所以類別不能放進其中任何一個，而且**兩個被組合的檔案都沒有改動**。

### 6.1 共通生命週期

三個類別的形狀相同，只有前綴不同（`ulcnet_` / `deepvqe_` / `dfn2_`）：

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
  `*_IO_FREQ` 是頻譜進頻譜出、完全不做 transform，用來接在 AEC/GSC seam 後面串接。
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
  **例外：DFN2 的 window 是複製的**——`DFN2State` 以 `float window[DFN2_WIN_LEN]` 內嵌窗表，
  所以 `cfg.window` 非 NULL 時會在 init 與每次 reset 覆蓋它（類別仍然不配置也不釋放任何東西）。
- `_reset()` 清空所有遞迴／ring／狀態、丟掉未結束的 frame、framing 與 clock 回 init 值；
  config、borrowed 指標與 pool 都不動。

### 6.2 每-hop 序列與回傳值合約

```c
int n = ulcnet_prepost_pre_process(p, err_hop, far_hop);  /* 需要幾次推論 */
if (n == 1) {
    UlcnetModelIoInputs  in;
    UlcnetModelIoOutputs out;
    ulcnet_prepost_frame_inputs(p, &in, &out);   /* 指向本實例 pool 的 view */
    if (accelerator_run(&in, &out) != 0 || ulcnet_prepost_frame_commit(p) != 0)
        ulcnet_prepost_frame_skip(p);            /* 保住 framing 節奏 */
}
int written = 0;
ulcnet_prepost_post_process(p, out_hop, &written);
```

- **`_pre_process()` 回的是「這一拍需要的加速器呼叫次數」，不是布林。**
  Align-ULCNet 與 DeepVQE-S **從第 0 拍起恆為 1**（一 hop 進、一次推論、一 hop 出，兩個 io_mode 皆然）；
  DeepFilterNet2 **第 0 拍回 0**、之後恆 1，因為它的圖吃 `[t-1, t, t+1]`，第 0 拍沒有右鄰居，
  那一拍**不可以**呼叫加速器。
- `_frame_inputs()` 把每個可寫輸出**預填 NaN**，所以加速器只寫一半會在 commit 被抓到，
  而不是把上一框的值當成本框結果。回 `0`，沒有開啟中的 frame 回 `-1`。
- `_frame_commit()` **先驗證再搬動**：確認每個 head／state 都是有限值之後才 commit。
  失敗時**什麼都不動**——持久狀態 byte-identical、frame 維持開啟、回 `-1`。
  接著只有兩條路：`_frame_skip()`，或重新 `_frame_inputs()` 再跑一次加速器。
- **commit 後面一定要有一次 `_frame_inputs()`**（`prepared` 閂鎖，三個類別皆然）：
  沒跑過的加速器不能把沒被碰過的 buffer 當成結果送出。

拒絕情境整理：

| 呼叫 | 情境 | 回傳 |
|---|---|---|
| `_pre_process` / `_pre_process_freq` | 上一個 frame 還開著（既未 commit 也未 skip） | `-1`（一拍被拒絕，不會默默疊上去） |
| `_pre_process` / `_pre_process_freq` | 與實例的 `io_mode` 不符 | `-1` |
| `_frame_inputs` / `_frame_commit` / `_frame_skip` | 沒有開啟中的 frame | `-1` |
| `_frame_commit` | 沒有 `_frame_inputs()` 在前面 | `-1`，狀態不動 |
| `dfn2_prepost_set_erb_matrices` / `_set_atten_lim` | frame 開著 | `-1` |

### 6.3 skip 政策：三個模型不一樣

`_frame_skip()` 是加速器跑失敗（或對齊邊界 reprime）時唯一正確的出口——它讓 framing
節奏繼續走，而模型的遞迴狀態**不**前進。但「這一框輸出什麼」每個模型不同：

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
  window 為 `ulcnet_make_window()` 產出的 `ULCNET_N_FFT` 項；`ULCNET_IO_FREQ` 不用）。
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

- 圖邊界：兩個 RI 交錯的訊號 input（`mic`/`far`，`[1,1,AIAEC_N_BINS,2]`）＋
  **16 個顯式 state**，順序就是 `DeepVqeStateId` 列舉（= exporter 的 `input_names[2:]` 逐字），
  輸出是 CCM taps `[1,1,BINS,DEEPVQE_TIME_ORDER,DEEPVQE_FREQ_TAPS,2]` ＋ 16 個 next state。
  按 index 綁的 adapter 必須用這個列舉，按名字綁的用 `_state_name()`。
- DeepVQE-S 回的是**每個 state 的完整下一個值**（不是差量），所以 pool 裡有兩套 bank、
  commit 時交換；NaN 預填與有限性檢查會走過整個狀態，這正是「失敗時什麼都不動」能被強制而不只是聲稱的原因。
- `descriptor_validate()` 拿 ONNX/JSON metadata 裡的 13 欄 `c_descriptor` 對本 build 的 ABI 逐欄比對，
  只有 `delay_depth` 是 export-time 部署參數、僅做範圍檢查；`DEEPVQE_PREPOST_LAYOUT_VERSION` = 1。
- D 範圍 `DEEPVQE_PREPOST_MIN_D`=1 到 `DEEPVQE_PREPOST_MAX_D`=256，出貨值
  `DEEPVQE_PREPOST_DEFAULT_D`=63（16 kHz 上的一秒搜尋範圍）。
- grid 只有 16 kHz/512/256：`aiaec_process.h` 帶 `#error` 守衛，grid 換掉是編譯失敗而不是默默重新解讀張量。

### 6.6 DeepFilterNet2：`dfn2_prepost.h`

生命週期與每-hop 函式同樣對應（前綴 `dfn2_prepost_`，單路訊號 `in_hop`／`spec_re,spec_im`，
長度 `DFN2_HOP_LEN`／`DFN2_N_BINS`，`_config_defaults(cfg, io_mode)` **沒有** `delay_depth`）。
額外 API：

```c
int  dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);  /* = 2 */
int  dfn2_prepost_layout_version(const DFN2Prepost *p);
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

以現行 build 量得，供規劃用；產品一律以 resolved config 呼叫 `_get_mem_size()`，
**不要把這裡的數字寫死**（§7 規則同樣適用）。單位 bytes。

| 類別 / grid | D | `*_IO_TIME` | `*_IO_FREQ` |
|---|---:|---:|---:|
| Align-ULCNet 16 kHz | 4 | 69,808 | 48,208 |
| Align-ULCNet 16 kHz | 8 | 98,992 | 77,392 |
| Align-ULCNet 16 kHz | 16 | 157,360 | 135,760 |
| Align-ULCNet 16 kHz | 32 | 274,096 | 252,496 |
| Align-ULCNet 16 kHz | 64 | 507,568 | 485,968 |
| Align-ULCNet 48 kHz | 4 | 132,272 | 89,168 |
| Align-ULCNet 48 kHz | 8 | 188,080 | 144,976 |
| Align-ULCNet 48 kHz | 16 | 299,696 | 256,592 |
| Align-ULCNet 48 kHz | 32 | 522,928 | 479,824 |
| Align-ULCNet 48 kHz | 64 | 969,392 | 926,288 |
| DeepVQE-S 16 kHz | 63 | 1,499,408 | 1,477,808 |
| DeepFilterNet2 48 kHz | — | 314,464 | 312,416 |

DeepVQE-S 的量體由兩套 16 個 state 的 bank 主宰。DeepFilterNet2 兩個模式只差 2,048 bytes
（輸出 hop staging）：`DFN2State` 把 analysis/window/synthesis 緩衝以值內嵌，FREQ 實例甩不掉它們，
**不要期待 AIAEC 那種等比例節省**。

### 6.8 Framing helper：一 hop 一框的 analysis

兩個 AIAEC 類別在 TIME 模式下的 analysis 走的是新增的：

```c
int ulcnet_analysis_push_frame(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP],
                               float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]);
int aiaec_analysis_push_frame(AiaecAnalysis *state, const float input[AIAEC_HOP],
                              float real[AIAEC_N_BINS], float imag[AIAEC_N_BINS]);
```

center=False，**從第一次 push 起就是一 push 一框**（恆回 1），與 AEC/GSC seam 的頻譜同一個慣例，
所以只有一路頻譜在手的呼叫端可以用它補另一路而兩路保持 hop 對齊。它與既有的
`ulcnet_analysis_push()`／`aiaec_analysis_push()`（center=True，第一拍 0 框、第二拍 2 框）
對同一個 hop 產出的**最後一框 bit-identical**，差別只在 centered 版本多吐一個 reflect-prefix 開頭框。

> **同一個 state 上不可混用兩個 push 函式**：它們共用 history，不共用 schedule。

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
- 完整 mono/4ch pipeline 另有自己的 aggregate sizing API，優先使用它，不要手算各 module
  byte 數再相加。

---

## 8. 建置、SIMD 與連結

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

模型 pre/post 類別（§6）另有兩個 archive，同樣是 configuration-keyed：

```bash
make -C AIAEC lib ; make -C AIAEC -s print-lib-path   # libaiaec_prepost.a
make -C AINR  lib ; make -C AINR  -s print-lib-path   # libainr_prepost.a
```

兩者都落在自己 repo 的 `build/simd<N>-<sig>/` 下（`<sig>` 是各自 Makefile 的組態簽章 cksum：
AIAEC 蓋 `CC`/`AR`/`CFLAGS`/`CPPFLAGS`/`SIMD`/`WERROR`，AINR 蓋 `CC`/`AR`/`CFLAGS`/`LDFLAGS`/
`SIMD`/`SIMD_CPPFLAGS`/`BACKEND`/`WERROR`/`AC_LIB`；每個目錄另存 `config.manifest` 擋簽章碰撞），
所以路徑一律用 `print-lib-path` 取得。成員：`libaiaec_prepost.a` = `ulcnet_process`、
`ulcnet_model_io`、`ulcnet_prepost`、`ulcnet_accelerator_adapter`、`aiaec_process`、
`deepvqe_process`、`deepvqe_prepost`；`libainr_prepost.a` = `dfn2_process`、`dfn2_model_io`、
`dfn2_prepost`、`gtcrn_process`。RNNoise-ERB **不在**任何 archive 裡（自己的 Makefile，
要用的板子把它的來源檔直接加進自己的 build）。`WERROR=1` 是 opt-in，且已納入組態簽章。

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

