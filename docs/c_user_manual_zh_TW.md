# Audio_ALG C User Manual（繁體中文）

本手冊說明 conventional mono pipeline 的建置、命令列使用方式，以及
以 lib/aec 的 `Aec` 加逐個初始化的 NR／合成函式庫（範例見 `pipelines/mono_aec_nr_res/main.c`、`static_main.c`）組成 mono 處理的整合方式
（沒有 mono pipeline 函式庫；每個函式庫各自 `mem_size`／`init`，heap 與 caller-owned-pool 兩種）；
四麥克風 C seam 另列於第 1.1 節。

> 現行 API 與 sizing contract 以 `lib/aec/c_impl/include/aec.h`、`pipelines/common/se_stage.h`、
> `pipelines/samples/se_anr_stage.h`、`pipelines/mono_aec_nr_res/static_main.c` 和 `pipelines/README.md` 為準。更新 submodule
> 或 tuning 後，必須重新建置並執行 pipeline tests；不要只依賴本手冊中的
> 範例常數。

## 1. 目前可用範圍

目前 C 端可用產物是：

- AEC static library：`libaec.a`，路徑 `make -s -C lib/aec/c_impl print-lib-path`
- NR static library：`libanr.a`，路徑 `make -s -C lib/nr/c_impl print-lib-path`
- 共用層 library（FFT/fast_math/hpf）：`libaudio_common.a`，路徑 `make -s -C ../audiolib/common print-lib-path`
- 三個 archive 都放在 `bin/<backend>-<config-hash>/` 下，路徑隨 BACKEND／SIMD 等建置設定改變，一律用 `print-lib-path` 取得
  （只印路徑、不建置）；link 前須先建好共用層（`make -C ../audiolib/common`），再建 `lib/aec/c_impl`、`lib/nr/c_impl`
- malloc reference executable：config-keyed `pipelines/bin/.../aec_nr_pipeline`
  （source：`pipelines/mono_aec_nr_res/main.c`）
- static-memory reference executable：config-keyed `pipelines/bin/.../aec_nr_pipeline_static`
  （source：`pipelines/mono_aec_nr_res/static_main.c`；單一 caller pool、init 後零 malloc、與 malloc 版的輸出相等由 `tests/test_mono_chain.c` 驗證；`--print-mem-size` 可直接查任一取樣率的 pool 需求）
- mono 的 reusable 零件：`lib/aec` 的 `Aec`（`aec_process_context()`）、`pipelines/common/se_stage.h`
  的 `se_frame_from_aec()`、`lib/nr` 的 `AnrDenoiser`、audiolib 的 `AcIstft`，以及 header-only 的 NR 級範例 `pipelines/samples/se_anr_stage.h`（沒有 mono 函式庫 archive；每 hop 的串接寫在各範例 `main`）
- 四麥克風前端 API：`pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` / `4aec_nr_res.c`
- 四麥克風線性 AEC + BF 單元（唯一包起來的單元）：`pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` /
  `audio_pipeline_4ch.c`；單元只含共用 delay 對齊、四條線性 AEC、SRP-PHAT 與 BF，post-beam RES（libaec 的 `AecRes`）／NR／CNG／合成不在單元內
- 四麥克風 static 對照範例：`pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c`
- 單元加 RES／NR／合成的實錄 runner：`pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c`
- 四麥克風 archive：`pipelines/4ch_aec_bf_nr_res` 的 config-keyed build directory
  內的 `libaudio_pipeline_4ch.a`（前端、單元、`se_config.o`）

function、descriptor version、ownership 與錯誤行為，mono 以 lib/aec 與 `pipelines/common/se_stage.h`、
四麥克風以 `4ch_aec_bf_nr_res/4aec_nr_res.h` 與 `audio_pipeline_4ch.h` 為準。

嵌入產品時，各函式庫各自查大小、各自 init，再由整合者串接：mono 是 `se_derive()` →
`aec_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()` → 對齊配置 → `aec_init()`／`anr_init()`／`ac_istft_init()`
（桌面也可用 `aec_create()`）；4ch 的單元是 `audio_pipeline_4ch_get_mem_requirements()` → 對齊配置 →
`audio_pipeline_4ch_init_ex()`（桌面可用 `audio_pipeline_4ch_create()`）。見
[`integration_mono_zh_TW.md`](integration_mono_zh_TW.md)。第 5 節是直接
組合 AEC／NR 的 application-layer wrapper 範例。

### 1.1 四麥克風 C seam

`FourAecNrRes*` 是 zero-padding-free 介面（單元 `AudioPipeline4Ch*` 建立在它之上），只支援：

| 取樣率 | frame | FFT | hop | bins | 資源拓撲 |
|---|---:|---:|---:|---:|---|
| 16 kHz | 256 | 256 | 128 | 129 | 1 shared matcher + 4 linear AEC（後接 1 post-beam RES + 1 NR：核心與單元都由應用自己建）（預設） |
| 16 kHz | 512 | 512 | 256 | 257 | 同上（可選） |
| 48 kHz | 1024 | 1024 | 512 | 513 | 同上（唯一合法 48 kHz 格點） |

呼叫端先用 `four_aec_front_process()` 取得借用的 `FourAecFront`
（四路 linear 頻譜與時域 hop），交由外部 SRP-PHAT／beamformer 更新 channel-major
`Complex[4][n_freqs]` 有效權重，再呼叫 `four_aec_fuse()` 取得 `SeSpecFrame`。
模組不實作 beamformer，但會以該組權重一致地投影 error／near／echo／R2 context；
之後由應用明寫串接一次 RES 增益、一次 NR 增益、`sk_min_f32`、`sk_capply_gain_f32`、
CNG 與 iFFT/OLA（`4aec_nr_res_static.c` 的 `post_hop()` 是範例；單元的 `fuse()` 也不算 RES 增益，`frame.g_res_ctx` 為 NULL，要用 `se_res_stage_gain()` 呼叫你自己的 `AecRes`）。

`FourAecNrRes*` core 本身不實作 beamformer；若要接入現有的第三方
SRP-PHAT/GSC，就用這個核心加自己的 beamformer（`four_aec_front_process()` → 你的 BF → `four_aec_fuse()`）；
不想自備 beamformer，改用線性 AEC + BF 單元 `AudioPipeline4Ch*`（內建 SRP-PHAT 與 BF；不含 RES／NR／合成）。單元的 AEC、DOA、BF，
以及你串的 RES 與 NR 都繼承同一個格點，`frame == FFT`、`hop == frame/2`，不能分別
指定互相矛盾的 frame/hop/FFT。`audio_pipeline_4ch_hop_size()`／`_doa_hop_size()`／`_bf_hop_size()`
等 accessors 可分別查詢 main、DOA 與 BF 實際格點；預設三組設定中三者必須一致。

目前只允許一個 in-flight front frame（下一次 `four_aec_front_process()` 會使上一幀的視圖失效）。建立方式與 mono 的 `Aec` 一致（heap 便利路徑與 caller-owned pool 兩種）：

- `four_aec_nr_res_create()`：desktop／測試用 heap convenience；
- `four_aec_nr_res_get_mem_requirements()` →
  caller 配置 16-byte aligned pool →
  `four_aec_nr_res_init_ex()`：board/static 路徑；
- 兩種建立方式共用完全相同的 `four_aec_front_process()`／`four_aec_fuse()` 核心，
  process 不配置記憶體，16／48 kHz 皆有 byte-identical parity test。

`four_aec_nr_res_destroy()` 對 static handle 不釋放 caller 的 pool；caller
應在 destroy 後自行交還平台 memory manager。可直接對照
`pipelines/mono_aec_nr_res/static_main.c` 與
`pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` 的
query → allocate → init → process → destroy → release 順序。完整 contract、
權重 convention 與 parity 限制見
[`../pipelines/4ch_aec_bf_nr_res/README.md`](../pipelines/4ch_aec_bf_nr_res/README.md)。

## 2. Production path

這不是把三個時域 filter 簡單串成 `AEC -> NR -> RES`。目前 production path 只在最後做一次 gain application 與 IFFT/OLA：

```text
mic/ref
  -> Linear AEC（PBFDKF + shadow + delay/EPC）
       -> E(f) + AecResContext { G_res, R², CNG N², far_power }
  -> echo-aware MMSE-LSA：以 E(f) 與 R² 計算 G_nr
  -> G_total = min(G_nr, G_res)
  -> S(f) = E(f) * G_total + optional CNG
  -> iFFT + sqrt-Hann OLA
  -> output hop
```

主要設計點：

1. AEC 設 `enable_res=0`，讓 time output 保持 linear residual；同時設 `return_res_context=1`，仍由 AEC3 post block 計算 frequency seam（此組態下 formed seam 另有 capture 候選，`ctx.error_spec` 不保證每個 hop 都是誤差，見 [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW_04_seam_troubleshooting.md#81-context-only-處理路徑) §8.1）。
2. `R²` 除以 `32768²` 後，作為 NR 的 `extra_noise_psd`，得到 echo-aware `G_nr`。
3. `G_nr` 與 AEC3 `G_res` 逐 bin 取較小值，不重複跑另一個時域 RES。
4. `g_total` 直接取 `min(G_nr, G_res)`。post 段沒有 near-end floor，也沒有任何依賴廣帶能量的 lift；一個 bin 是否被保留，完全由 NR 自己的 speech-presence 模型與 AEC3 的 `G_res` 決定，所以環境噪聲大也不會把 NR 壓住或讓噪聲底跟著遠端起伏。
5. CNG 只依 `G_res` 決定填哪些 bin（AEC 抑制留下的頻譜空洞），注入位準再乘該 bin 的 `G_nr`（下限 −10 dB，`SE_CNG_NR_GAIN_FLOOR`）：NR 看不到 comfort noise，所以它的 gain 在這裡補套，讓補進去的噪聲落在降噪後的底噪，而不是未降噪的環境噪聲。

## 3. 取得 submodule 與建置

首次 clone：

```bash
git clone --recursive <Audio_ALG-repository-url>
cd Audio_ALG
```

既有 checkout：

```bash
git submodule update --init --recursive
```

建置目前的 C pipeline：

```bash
# 從 Audio_ALG 根目錄執行；mono 與 4ch 的 producer 完全分離
make -C pipelines            # mono heap/static 範例 + DFN2 stage archive
make -C pipelines BACKEND=ne10   # NE10 FFT 後端（obj/ 依 backend+參數雜湊分開目錄，免手動 clean-libs）
make -C pipelines/4ch_aec_bf_nr_res BACKEND=ne10 SIMD=1
make -C pipelines/4ch_aec_bf_nr_res 4aec_nr_res_static
make -C pipelines/4ch_aec_bf_nr_res audio_pipeline_4ch_raw
make -C pipelines/4ch_aec_bf_nr_res test
```

若自行編譯 wrapper，沿用目前 Makefile 的 include／link layout（注意：兩個 library 都依賴
共用層 `libaudio_common.a`，缺它會 link error；NE10 標頭需要 GNU 擴充，用 `gnu99`）：

```bash
cc -std=gnu99 -O2 -Wall -Wextra \
  -Ilib/aec/c_impl/include -Ilib/aec/c_impl/example \
  -Ilib/nr/c_impl/include \
  -I../audiolib/common/include \
  -c app.c -o app.o
# libaudio_common.a 內含一個 C++ TU，最終 link 用 c++ driver；路徑由 audiolib 的 Makefile 回報
c++ app.o \
  "$(make -s -C lib/aec/c_impl print-lib-path)" \
  "$(make -s -C lib/nr/c_impl print-lib-path)" \
  "$(make -s -C ../audiolib/common print-lib-path)" -lm -o app
```

AEC library 本身使用 `-ffp-contract=off -fno-math-errno` 建置；若 application 內重做同類遞迴 DSP 運算，也應保留這兩個選項，以維持數值一致性並確保 AArch64 `sqrtf` 直接降成 `FSQRT`。

## 4. 命令列使用

```bash
# 執行檔在 config-keyed 目錄（由 make 依 backend／旗標雜湊決定）
BIN="$(make -s -C pipelines print-bin-dir)"

# balanced AEC + balanced NR
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced

# 各自指定 preset
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav mild \
  --nr-preset mild
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav aggressive \
  --nr-preset aggressive

# 只輸出 linear AEC path，不跑 NR／frequency gain combine／final OLA
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced --aec-only

# 只關其中一個 gain 來源：--no-nr 只剩 RES；--no-res 只剩 NR（CNG 隨之略過）；
# 兩個都關 = 線性殘差經 iFFT/OLA 直通（與 --aec-only 差一個 hop 的延遲）
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced --no-nr
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced --no-res

# NR 的 noise prior 不折入 R²（noise-only NR）；只作比對用
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced --legacy-amin

# 不加入 comfort noise
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced --no-cng
```

### 4.1 CLI 參數

| 參數 | 值 | 預設 | 說明 |
|---|---|---|---|
| 第 4 個 positional argument | `mild`／`balanced`／`aggressive`／`no_floor`（或 `0`..`3`） | `balanced` | AEC preset；未知字串退回 `balanced` |
| `--nr-preset` | `mild`／`moderate`／`balanced`／`aggressive` | `balanced` | NR strength |
| `--aec-only` | flag | off | 跳過 NR 與最終 gain combine |
| `--no-nr` | flag | off | 不建降噪器，總增益只剩 RES |
| `--no-res` | flag | off | RES 增益視為全 1（comfort noise 隨之略過） |
| `--legacy-amin` | flag | off | NR 的 noise prior 不折入 R²（noise-only NR），只作比對用 |
| `--no-cng` | flag | off | 關閉最終 CNG 注入 |
| `--fft-size` | `256`／`512`／`1024` | 0（rate default） | 覆寫訊號格點的 FFT/frame 長度（16 kHz 唯一可切換的一項，見 pipeline_mono.html「訊號格點」表） |
| `--debug` | flag | off | 每秒一行 AEC+NR 狀態（read-only，不影響輸出） |
| `--timing` | flag | off | 結束時印每個階段的平均耗時；AEC 的列需要 `make PROFILE=1` |

`delay_mode`／`delay_num_filters`／`fixed_delay_samples`／`filter_length`
（見 `common/se_config.h` 的 `SeRequest`）**沒有對應 CLI 旗標**；
兩支 CLI（`aec_nr_pipeline`/`aec_nr_pipeline_static`）目前都只能吃
預設值（`MATCHED`、n=5、−1）。要測試
非預設 delay mode 得直接寫 C 呼叫 `se_derive()` 與 `aec_get_mem_size()`／`aec_init()`
（或改 CLI 原始碼），見 `pipeline_mono.html` API 節。

目前 parser 對未知 positional preset 會退回 balanced，且不會為所有未知 dash option 報錯。產品或自動化腳本應只傳上表列出的值，不要依賴 silent fallback。

### 4.2 Debug context dump

設定 `DUMP_CTX` 可將每 hop 的中間資料寫成 binary dump，供 port parity 或離線分析：

```bash
DUMP_CTX=pipeline_ctx.bin \
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced
```

檔案開頭是兩個 `int`：`n_freqs`、`hop`；後續每幀依序包含 `error_spec`、`res_gain`（lane 自己的增益）、`r2`、`comfort_noise`、`far_power`、`g_nr`（沒有降噪器時是 unity 全 1 向量）與 output hop。這是診斷格式，沒有穩定 ABI 保證。

### 4.3 WAV contract

- mic/ref 必須能被 submodule 的 `wav_io.h` 讀取，並具有相同 sample rate。
- 支援 PCM16、PCM32、IEEE float32；多聲道只讀第一聲道。
- 處理長度取 mic/ref 中較短者。
- 每次處理所選 signal grid 的一個 hop，最後不足一個 hop 的尾端捨棄。
- pipeline 沒有主動改寫 `AEC_OUT_FLOAT`，因此預設輸出單聲道 PCM16。
- 若要 float32 WAV：`AEC_OUT_FLOAT=1 "$BIN"/aec_nr_pipeline ...`。

## 5. 可編譯的 C integration wrapper

下列範例只使用目前 submodule 已存在的 API，以 malloc 手寫一份與 `pipelines/mono_aec_nr_res` 範例等價的串接（RES 增益取自 AEC 自己的 context）。上層每次傳入 `audio_alg_hop_size()` 個 float sample。

```c
/* AUDIO_ALG_PIPELINE_EXAMPLE_BEGIN */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"   /* aec_cng_fill */
#include "fft_wrapper.h"
#include "anr.h"
#include "anr_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PSD_SCALE                 (32768.0f * 32768.0f)

typedef struct {
    Aec *aec;
    int aec_ready;
    AnrDenoiser *nr;
    FftHandle *fft;
    int hop;
    int frame_size;
    int fft_size;
    int n_freqs;
    int enable_cng;
    uint32_t rng;
    float *linear;
    float *window;
    float *ola;
    float *ifft_buf;
    float *g_nr;
    float *g_total;
    float *g_aec;
    float *extra;
    Complex *spec;
} AudioAlgPipeline;

void audio_alg_destroy(AudioAlgPipeline *p)
{
    if (!p) return;
    if (p->aec_ready) aec_destroy(p->aec);
    free(p->aec);
    anr_destroy(p->nr);
    fft_destroy(p->fft);
    free(p->linear);
    free(p->window);
    free(p->ola);
    free(p->ifft_buf);
    free(p->g_nr);
    free(p->g_total);
    free(p->g_aec);
    free(p->extra);
    free(p->spec);
    memset(p, 0, sizeof(*p));
}

int audio_alg_init(AudioAlgPipeline *p, int sample_rate,
                   AecPreset aec_preset, AnrMode nr_mode,
                   int enable_cng)
{
    AecConfig aec_cfg;
    AnrConfig nr_cfg;
    int k;

    if (!p) return -1;
    memset(p, 0, sizeof(*p));
    p->rng = 0x9e3779b9u;
    p->enable_cng = enable_cng != 0;

    aec_config_from_preset(&aec_cfg, aec_preset, sample_rate);
    aec_cfg.enable_res = 0;
    aec_cfg.return_res_context = 1;

    p->aec = (Aec *)calloc(1, sizeof(Aec));
    if (!p->aec || aec_create(p->aec, &aec_cfg) != 0)
        goto fail;
    p->aec_ready = 1;
    p->hop = aec_hop_size(p->aec);
    p->frame_size = 2 * p->hop;
    p->fft_size = p->frame_size;   /* no padding: AEC's grid is frame == FFT */
    p->n_freqs = p->fft_size / 2 + 1;

    /* Use the grid-explicit entry point (as the pipelines/ examples do)
     * so NR resolves the SAME fft_size the AEC instance above just picked --
     * the 2-arg anr_config_for_mode() instead resolves NR's OWN
     * independent per-rate default grid, which is not the AEC default at
     * every rate (e.g. NR defaults to 128 at 8 kHz, AEC to 256) and would
     * silently create an NR instance sized for the wrong n_freqs.
     * L/alpha_d/alpha_attack are already correctly wall-clock-retimed for
     * this grid by anr_config_for_mode_grid() -- do not override them
     * with fixed literals authored for a different (10 ms/16 ms) grid. */
    nr_cfg = anr_config_for_mode_grid(sample_rate, p->fft_size, nr_mode);

    p->nr = anr_create(&nr_cfg);
    p->fft = fft_create(p->fft_size);
    p->linear = (float *)malloc((size_t)p->hop * sizeof(float));
    p->window = (float *)malloc((size_t)p->frame_size * sizeof(float));
    p->ola = (float *)calloc((size_t)p->frame_size, sizeof(float));
    p->ifft_buf = (float *)malloc((size_t)p->fft_size * sizeof(float));
    p->g_nr = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->g_total = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->g_aec = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->extra = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->spec = (Complex *)malloc((size_t)p->n_freqs * sizeof(Complex));

    if (!p->nr || !p->fft || !p->linear || !p->window || !p->ola ||
        !p->ifft_buf || !p->g_nr || !p->g_total || !p->g_aec ||
        !p->extra || !p->spec)
        goto fail;

    for (k = 0; k < p->frame_size; ++k) {
        float hann = 0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)k / (float)p->frame_size);
        p->window[k] = sqrtf(hann);
    }
    return 0;

fail:
    audio_alg_destroy(p);
    return -1;
}

int audio_alg_hop_size(const AudioAlgPipeline *p)
{
    return p ? p->hop : 0;
}

int audio_alg_process(AudioAlgPipeline *p,
                      const float *mic, const float *ref, float *out)
{
    AecResContext ctx;
    const float *extra = NULL;
    int k;

    if (!p || !p->aec_ready || !p->nr || !mic || !ref || !out)
        return -1;

    aec_process(p->aec, mic, ref, p->linear);
    aec_get_res_context(p->aec, &ctx);
    if (!ctx.error_spec || !ctx.res_gain || ctx.n_freqs != p->n_freqs)
        return -1;

    /* 與 production（se_anr_stage_gain）相同：R^2 只在 usable_linear != 0 時摺進降噪器的 noise floor */
    if (ctx.r2 && ctx.usable_linear) {
        for (k = 0; k < p->n_freqs; ++k)
            p->extra[k] = ctx.r2[k] / PSD_SCALE;
        extra = p->extra;
    }

    if (anr_process_gain(p->nr, ctx.error_spec, extra, p->g_nr) < 0)
        return -1;

    for (k = 0; k < p->n_freqs; ++k) {
        p->g_aec[k] = ctx.res_gain[k];
        p->g_total[k] = p->g_nr[k] < ctx.res_gain[k]
                      ? p->g_nr[k] : ctx.res_gain[k];
    }

    for (k = 0; k < p->n_freqs; ++k) {
        p->spec[k].r = ctx.error_spec[k].r * p->g_total[k];
        p->spec[k].i = ctx.error_spec[k].i * p->g_total[k];
    }

    if (p->enable_cng && ctx.comfort_noise)   /* 只填 RES 挖掉的 bin，位準乘該 bin 的 NR 增益（下限 -10 dB） */
        aec_cng_fill(&p->rng, p->spec, ctx.comfort_noise, p->g_aec, p->g_nr,
                     0.31622777f, p->n_freqs);

    fft_inverse(p->fft, p->spec, p->ifft_buf);
    for (k = 0; k < p->frame_size; ++k)
        p->ola[k] += p->ifft_buf[k] * p->window[k];

    memcpy(out, p->ola, (size_t)p->hop * sizeof(float));
    memmove(p->ola, p->ola + p->hop,
            (size_t)(p->frame_size - p->hop) * sizeof(float));
    memset(p->ola + p->frame_size - p->hop, 0,
           (size_t)p->hop * sizeof(float));
    return 0;
}

void audio_alg_reset(AudioAlgPipeline *p)
{
    if (!p || !p->aec_ready || !p->nr) return;
    aec_reset(p->aec);
    anr_reset(p->nr);
    memset(p->ola, 0, (size_t)p->frame_size * sizeof(float));
    p->rng = 0x9e3779b9u;
}
/* AUDIO_ALG_PIPELINE_EXAMPLE_END */
```

典型使用：

```c
AudioAlgPipeline pipeline;
if (audio_alg_init(&pipeline, 16000,
                   AEC_PRESET_BALANCED,
                   ANR_MODE_BALANCED, 1) != 0)
    return -1;

int hop = audio_alg_hop_size(&pipeline);
/* 每次準備 mic[hop]、ref[hop]，並接收 out[hop]。 */
if (audio_alg_process(&pipeline, mic, ref, out) != 0) {
    audio_alg_destroy(&pipeline);
    return -1;
}

/* 處理另一條獨立串流前： */
audio_alg_reset(&pipeline);
audio_alg_destroy(&pipeline);
```

這段 wrapper 故意保持在 application 層，沒有宣稱新的 public ABI。若要修改 production constant、CNG RNG 或 failure policy，應在自己的 wrapper 版本化並重新跑品質評測。

## 6. Buffer、尺度與 lifetime

| 資料 | 尺度／尺寸 | 規則 |
|---|---|---|
| mic/ref/out | float PCM，`hop` | 一次正好一個 grid hop，建議幅度 `[-1, 1]` |
| `error_spec` | complex audio amplitude，`n_freqs` | AEC instance 內部 pointer |
| `res_gain` | amplitude gain `[0,1]` | 與 `G_nr` 逐 bin 比較 |
| `r2` | int16² PSD | 除以 `32768²` 才能和 `|E|²` 合併 |
| `comfort_noise` | int16² PSD | 除以 `32768²` 後取平方根得到 amplitude |
| `G_nr`／`G_total` | amplitude gain | 直接乘 complex spectrum |

`AecResContext` pointer 只保證在下一次 AEC process/reset/destroy 前有效。不要修改、free 或跨 hop 保存 pointer；如需非同步分析，複製內容。

Conventional mono pipeline 跟 4-channel/AIAEC 一樣是 zero-padding-free
50%-overlap grid（frame == FFT、hop == FFT/2）：8 kHz 為 frame(=FFT)/hop `256/128`，16 kHz 預設為
`256/128`（512/256 也是明確支援的可選 grid），48 kHz 為 `1024/512`；
`n_freqs = FFT/2 + 1`。呼叫端應從 API query 實際 hop 與 bins，而不是寫死
（16 kHz 有兩種合法 grid，寫死任一組數字對另一組就是錯的）。

完整 pipeline 的 final IFFT/OLA 增加約一個 grid hop 延遲；`--aec-only` 走 linear time output，不經這段 final OLA。另需把裝置 I/O buffer、resampler 與 OS scheduling latency 加入產品總預算。

## 7. Preset 與 tuning 原則

AEC preset：

| Preset | 方向 |
|---|---|
| mild | 近端保留優先 |
| balanced | 預設 production operating point |
| aggressive | 回聲抑制優先 |

NR preset（`--nr-preset`，見 §4.1）：

| Preset | 方向 |
|---|---|
| mild | 最保守的背景降噪 |
| balanced | 預設平衡 |
| aggressive | 最強背景降噪 |

> `anr_config_for_mode()` 有四級：`mild`／`moderate`／`balanced`／`aggressive`（`moderate` 介於 mild 與 balanced 之間）。`aec_nr_pipeline` 的 `parse_nr_mode()`（`pipelines/mono_aec_nr_res/main.c`）經 `anr_nr_mode_from_name()` 接受這四個名稱（或 `0`..`3`），其餘字串一律 silently 落回 `balanced`，不會報錯。

Audio_ALG 的範例程式（`pipelines/mono_aec_nr_res/main.c` 等）直接用 NR 函式庫的 preset：`anr_config_for_mode_grid(sample_rate, fft_size, nr_mode)`，不另外覆寫欄位（最小值追蹤視窗 `L` 為函式庫預設，約 512 ms；16 kHz 256/128 為 64 hops），所以 `anr_set_mode()` 可以直接在執行中換強度。

若要改 CNG 或 gain combine（例如在 `min(G_nr, G_res)` 之外再加任何 floor），這已不是單純 API integration，而是演算法 operating point 變更，應重新跑 far-end-only、near-end-only、double-talk、movement 與 noisy cases。

## 8. Lifecycle 與 thread safety

- 每條串流各有一組 AEC、NR、FFT、OLA 與 RNG state。
- instance 不可由多個 thread 無同步同時 process。
- 換獨立串流前 reset AEC、NR、合成 OLA 與 comfort-noise RNG（四件一起，見 [mono 手冊 §3](integration_mono_zh_TW_02_lifecycle_config_errors.md#3-lifecycle)）；或直接 destroy/create。
- create 失敗時必須釋放已成功建立的子模組與 buffer。
- malloc reference path（`aec_nr_pipeline`）在初始化時用 `posix_memalign()` 配置各函式庫的區塊；process loop 本身不配置記憶體。
- 單一 caller-owned pool 的 static-memory 版本已完整支援：`aec_nr_pipeline_static` 各自查大小、各自 init
  （`aec_get_mem_size`/`aec_init`、`anr_get_mem_size`/`anr_init`、`ac_istft_get_mem_size`/`ac_istft_init`），
  從一塊 16-byte 對齊的 pool 依序切出全部記憶體，init 後零 malloc；`pipelines/mono_aec_nr_res/tests/test_mono_chain.c` 以逐位元比對兩支 CLI 與手寫 oracle
  （pooled 與 heap 的輸出因此相等，範圍為該測試涵蓋的選項組合）。
  NR 與 AEC 的 heap 與 static API 並存於同一份 `.a`（runtime 選擇）。

## 9. Troubleshooting

| 現象 | 先檢查 | 建議 |
|---|---|---|
| build 找不到 library | 未初始化 submodule 或未跑 `make ... libs` | `git submodule update --init --recursive` 後重建 |
| mic/ref 無法處理 | sample rate 不一致或 WAV format 不支援 | 先轉成同步、同 SR 的 PCM16/float32 mono 測試 |
| 回聲未消除 | ref routing／delay／AEC 尚未 convergence | 先用 `--aec-only` 隔離 AEC 問題 |
| AEC-only 正常，完整 pipeline 傷近端 | NR preset 或 RES gain | 先改 mild；再用 `--no-nr` / `--no-res` 隔離是哪個 gain 來源 |
| 背景噪聲殘留 | NR 還在 init 或 preset 太保守 | 確認開頭噪聲段，逐級試 balanced/aggressive |
| 輸出有洞或不自然靜音 | CNG 關閉或 gain 過深 | A/B 比較有無 `--no-cng`；CNG 只填 `G_res` 切掉的 bin，位準乘該 bin 的 NR gain（下限 −10 dB） |
| 不同檔案結果互相影響 | state 未 reset | 每個獨立 stream reset 或重建全部 instance |
| 尾端樣本變短 | CLI 只處理完整 hop | 上層先 padding，並在輸出後裁回原長度 |
| 想用現成的 mono 整合寫法 | — | 見 `pipelines/mono_aec_nr_res/main.c`／`static_main.c` + `pipelines/README_02_build_debug_board.md`「Board Integration」；本手冊第 5 節的 wrapper 是直接組合 AEC／NR 的範例 |

定位問題時建議依序比較：`--aec-only`、完整 production、`--no-cng`、`--legacy-amin`，再用 `DUMP_CTX` 檢查 `E(f)`、`G_res`、`R²` 與 `G_nr`。
