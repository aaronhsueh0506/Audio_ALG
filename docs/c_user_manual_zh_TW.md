# Audio_ALG C User Manual（繁體中文）

本手冊說明 conventional mono pipeline 的建置、命令列使用方式，以及
`pipelines/audio_pipeline.h` 的 heap／caller-owned-pool 兩種整合方式；
四麥克風 C seam 另列於第 1.1 節。

> 現行 API 與 sizing contract 以 `pipelines/audio_pipeline.h`、
> `pipelines/audio_pipeline.c` 和 `pipelines/README.md` 為準。更新 submodule
> 或 tuning 後，必須重新建置並執行 pipeline tests；不要只依賴本手冊中的
> 範例常數。

## 1. 目前可用範圍

目前 C 端可用產物是：

- AEC static library：`lib/aec/c_impl/bin/libaec.a`
- NR static library：`lib/nr/c_impl/bin/libmmse_lsa.a`
- 共用層 library（FFT/fast_math/hpf）：`../audio_common/bin/<backend>/libaudio_common.a`
- malloc reference executable：`pipelines/aec_nr_pipeline`（source：`aec_nr_pipeline.c`）
- static-memory reference executable：`pipelines/aec_nr_pipeline_static`（source：
  `aec_nr_pipeline_static.c`；單一 caller pool、init 後零 malloc、輸出與 malloc 版
  byte-identical；`--print-mem-size` 可直接查任一取樣率的 pool 需求）
- reusable API：`pipelines/audio_pipeline.h` / `audio_pipeline.c`
- linkable archive：config-keyed build directory 內的 `libaudio_pipeline.a`
- 四麥克風 API：`pipelines/aec_4ch/4aec_nr_res.h` / `4aec_nr_res.c`
- 四麥克風 static 對照範例：`pipelines/aec_4ch/4aec_nr_res_static.c`
- 四麥克風 archive：config-keyed build directory 內的 `lib4aec_nr_res.a`

`pipelines/PLAN_audio_pipeline_api.md` 是 API 實作前的歷史設計草案，
不是可依賴的介面。已實作的 function、descriptor version、ownership 與
錯誤行為，mono 以 `audio_pipeline.h`、四麥克風以
`aec_4ch/4aec_nr_res.h` 為準。

嵌入產品時優先使用 `AudioPipeline*`：桌面／服務端可用
`audio_pipeline_create()`，firmware 則以
`audio_pipeline_get_mem_requirements()` → 對齊配置 →
`audio_pipeline_init_ex()` 建立 caller-owned-pool instance。第 5 節的
AEC／NR 直接 wrapper 僅保留給尚未過渡的既有呼叫端。

### 1.1 四麥克風 C seam

`FourAecNrRes*` 是獨立於 `AudioPipeline*` 的 zero-padding-free 介面，只支援：

| 取樣率 | FFT / hop | 資源拓撲 |
|---|---|---|
| 16 kHz | 512 / 256 | 1 shared matcher + 4 linear AEC + 1 post-beam RES + 1 NR |
| 48 kHz | 1024 / 512 | 1 shared matcher + 4 linear AEC + 1 post-beam RES + 1 NR |

呼叫端先用 `four_aec_nr_res_process_pre()` 取得 interleaved `[hop][4]`
linear output 與 token，交由外部 SRP-PHAT/GSC 更新 channel-major
`Complex[4][n_freqs]` 有效權重，再用同一 token 呼叫
`four_aec_nr_res_process_post()` 取得 mono hop。模組不實作 beamformer，
但會以該組權重一致地投影 error／near／echo／R2 context，再只執行一次
NR、RES gain fusion 與 iFFT/OLA。

目前只允許一個 in-flight frame。建立方式比照 mono `AudioPipeline`：

- `four_aec_nr_res_create()`：desktop／測試用 heap convenience；
- `four_aec_nr_res_get_mem_requirements()` →
  caller 配置 16-byte aligned pool →
  `four_aec_nr_res_init_ex()`：board/static 路徑；
- 兩種建立方式共用完全相同的 `process_pre()`／`process_post()` 核心，
  process 不配置記憶體，16／48 kHz 皆有 byte-identical parity test。

`four_aec_nr_res_destroy()` 對 static handle 不釋放 caller 的 pool；caller
應在 destroy 後自行交還平台 memory manager。可直接對照
`pipelines/aec_nr_pipeline_static.c` 與
`pipelines/aec_4ch/4aec_nr_res_static.c` 的
query → allocate → init → process → destroy → release 順序。完整 contract、
權重 convention 與 parity 限制見
[`../pipelines/aec_4ch/README.md`](../pipelines/aec_4ch/README.md)。

## 2. Production path

這不是把三個時域 filter 簡單串成 `AEC -> NR -> RES`。目前 production path 只在最後做一次 gain application 與 IFFT/OLA：

```text
mic/ref
  -> Linear AEC（PBFDKF + shadow + delay/EPC）
       -> E(f) + AecResContext { G_res, R², CNG N², far_power }
  -> echo-aware MMSE-LSA：以 E(f) 與 R² 計算 G_nr
  -> G_total = min(G_nr, G_res)
  -> far/near gate 決定 near-end floor lift
  -> S(f) = E(f) * G_total + optional CNG
  -> iFFT + sqrt-Hann OLA
  -> output hop
```

主要設計點：

1. AEC 設 `enable_res=0`，讓 time output 保持 linear residual；同時設 `return_res_context=1`，仍由 AEC3 post block 計算 frequency seam。
2. `R²` 除以 `32768²` 後，作為 NR 的 `extra_noise_psd`，得到 echo-aware `G_nr`。
3. `G_nr` 與 AEC3 `G_res` 逐 bin 取較小值，不重複跑另一個時域 RES。
4. near-end floor 只在低 echo bin 拉高 gain，並以 far/near activity gate 控制保護強度。
5. CNG 只依 `G_res` 填回 AEC 抑制留下的頻譜空洞，不把 NR 剛去除的背景噪聲重新灌回。

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
# 從 Audio_ALG 根目錄執行 — 預設 target 也會建 4ch archive/static example
make -C pipelines            # mono heap/static + lib4aec_nr_res.a + 4ch static
make -C pipelines BACKEND=ne10   # NE10 FFT 後端（obj/ 依 backend+參數雜湊分開目錄，免手動 clean-libs）
make -C pipelines lib4aec_nr_res.a
make -C pipelines 4aec_nr_res_static
make -C pipelines test_4aec_nr_res
```

若自行編譯 wrapper，沿用目前 Makefile 的 include／link layout（注意：兩個 library 都依賴
共用層 `libaudio_common.a`，缺它會 link error；NE10 標頭需要 GNU 擴充，用 `gnu99`）：

```bash
cc -std=gnu99 -O2 -Wall -Wextra \
  -Ilib/aec/c_impl/include -Ilib/aec/c_impl/example \
  -Ilib/nr/c_impl/include \
  -I../audio_common/include \
  app.c \
  -Llib/aec/c_impl/bin -laec \
  -Llib/nr/c_impl/bin -lmmse_lsa \
  ../audio_common/bin/kiss/libaudio_common.a -lm -o app
```

AEC library 本身使用 `-ffp-contract=off` 建置；若 application 內重做同類遞迴 DSP 運算，也建議保留此選項以維持數值一致性。

## 4. 命令列使用

```bash
# balanced AEC + balanced NR
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced

# 各自指定 preset
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav mild \
  --nr-preset mild
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav aggressive \
  --nr-preset aggressive

# 只輸出 linear AEC path，不跑 NR／frequency gain combine／final OLA
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --aec-only

# 使用舊版 min-only 行為：NR 不注入 R²，near-end floor 改回 scalar
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --legacy-amin

# 不加入 comfort noise
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --no-cng
```

### 4.1 CLI 參數

| 參數 | 值 | 預設 | 說明 |
|---|---|---|---|
| 第 4 個 positional argument | `mild`／`balanced`／`aggressive` | `balanced` | AEC preset |
| `--nr-preset` | `mild`／`balanced`／`aggressive` | `balanced` | NR strength |
| `--aec-only` | flag | off | 跳過 NR 與最終 gain combine |
| `--legacy-amin` | flag | off | 還原舊版 min-only path |
| `--no-cng` | flag | off | 關閉最終 CNG 注入 |
| `--debug` | flag | off | 每秒一行 AEC+NR 狀態（read-only，不影響輸出） |

目前 parser 對未知 positional preset 會退回 balanced，且不會為所有未知 dash option 報錯。產品或自動化腳本應只傳上表列出的值，不要依賴 silent fallback。

### 4.2 Debug context dump

設定 `DUMP_CTX` 可將每 hop 的中間資料寫成 binary dump，供 port parity 或離線分析：

```bash
DUMP_CTX=/tmp/pipeline_ctx.bin \
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced
```

檔案開頭是兩個 `int`：`n_freqs`、`hop`；後續每幀依序包含 `error_spec`、`res_gain`、`r2`、`comfort_noise`、`far_power`、`g_nr` 與 output hop。這是診斷格式，沒有穩定 ABI 保證。

### 4.3 WAV contract

- mic/ref 必須能被 submodule 的 `wav_io.h` 讀取，並具有相同 sample rate。
- 支援 PCM16、PCM32、IEEE float32；多聲道只讀第一聲道。
- 處理長度取 mic/ref 中較短者。
- 每次處理所選 signal grid 的一個 hop，最後不足一個 hop 的尾端捨棄。
- pipeline 沒有主動改寫 `AEC_OUT_FLOAT`，因此預設輸出單聲道 PCM16。
- 若要 float32 WAV：`AEC_OUT_FLOAT=1 ./pipelines/aec_nr_pipeline ...`。

## 5. 可編譯的 C integration wrapper

下列範例只使用目前 submodule 已存在的 API，重現 production 的 malloc path。上層每次傳入 `audio_alg_hop_size()` 個 float sample。

```c
/* AUDIO_ALG_PIPELINE_EXAMPLE_BEGIN */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "fft_wrapper.h"
#include "mmse_lsa_denoiser.h"
#include "mmse_lsa_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PSD_SCALE                 (32768.0f * 32768.0f)
#define PROD_NE_FLOOR             0.4f
#define PROD_NE_FLOOR_FAR_ACTIVE  0.2f
#define PROD_FAR_GATE_THRESH      1e-4f
#define PROD_NEAR_GATE_THRESH     1e-3f
#define PROD_NEAR_HANGOVER        8

typedef struct {
    Aec *aec;
    int aec_ready;
    MmseLsaDenoiser *nr;
    FftHandle *fft;
    int hop;
    int frame_size;
    int fft_size;
    int n_freqs;
    int enable_cng;
    int near_hang;
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

static float pipeline_uniform(AudioAlgPipeline *p)
{
    p->rng ^= p->rng << 13;
    p->rng ^= p->rng >> 17;
    p->rng ^= p->rng << 5;
    return ((p->rng >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static float pipeline_gauss(AudioAlgPipeline *p)
{
    float u1 = pipeline_uniform(p);
    float u2 = pipeline_uniform(p);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) *
           cosf(2.0f * (float)M_PI * u2);
}

void audio_alg_destroy(AudioAlgPipeline *p)
{
    if (!p) return;
    if (p->aec_ready) aec_destroy(p->aec);
    free(p->aec);
    mmse_lsa_destroy(p->nr);
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
                   AecPreset aec_preset, MmseLsaNrMode nr_mode,
                   int enable_cng)
{
    AecConfig aec_cfg;
    MmseLsaConfig nr_cfg;
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
    p->fft_size = 512;
    while (p->fft_size < p->frame_size) p->fft_size *= 2;
    p->n_freqs = p->fft_size / 2 + 1;

    nr_cfg = mmse_lsa_config_for_mode(sample_rate, nr_mode);
    nr_cfg.L = 150;
    nr_cfg.alpha_d = 0.95f;
    nr_cfg.alpha_attack = 0.3f;
    nr_cfg.alpha_decay = nr_cfg.alpha_g;

    p->nr = mmse_lsa_create(&nr_cfg);
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
    int k;
    float nf_eff;

    if (!p || !p->aec_ready || !p->nr || !mic || !ref || !out)
        return -1;

    aec_process(p->aec, mic, ref, p->linear);
    aec_get_res_context(p->aec, &ctx);
    if (!ctx.error_spec || !ctx.res_gain || ctx.n_freqs != p->n_freqs)
        return -1;

    for (k = 0; k < p->n_freqs; ++k)
        p->extra[k] = ctx.r2 ? ctx.r2[k] / PSD_SCALE : 0.0f;

    if (mmse_lsa_process_gain(p->nr, ctx.error_spec,
                              ctx.r2 ? p->extra : NULL, p->g_nr) < 0)
        return -1;

    for (k = 0; k < p->n_freqs; ++k) {
        p->g_aec[k] = ctx.res_gain[k];
        p->g_total[k] = p->g_nr[k] < ctx.res_gain[k]
                      ? p->g_nr[k] : ctx.res_gain[k];
    }

    {
        double near_energy = 0.0;
        int far_active = ctx.far_power > PROD_FAR_GATE_THRESH;
        int near_active;

        for (k = 0; k < p->n_freqs; ++k) {
            float re = ctx.error_spec[k].r;
            float im = ctx.error_spec[k].i;
            near_energy += (double)(re * re + im * im);
        }
        near_energy /= (double)p->n_freqs;
        if (near_energy > PROD_NEAR_GATE_THRESH)
            p->near_hang = PROD_NEAR_HANGOVER;
        near_active = p->near_hang > 0;
        if (p->near_hang > 0) --p->near_hang;
        nf_eff = (!far_active && near_active)
               ? PROD_NE_FLOOR : PROD_NE_FLOOR_FAR_ACTIVE;
    }

    if (ctx.r2) {
        for (k = 0; k < p->n_freqs; ++k) {
            float re = ctx.error_spec[k].r;
            float im = ctx.error_spec[k].i;
            float e2 = re * re + im * im;
            float r2 = ctx.r2[k] / PSD_SCALE;
            float echo_frac = r2 / (e2 + 1e-12f);
            float no_echo;
            float lift;

            if (echo_frac < 0.0f) echo_frac = 0.0f;
            if (echo_frac > 1.0f) echo_frac = 1.0f;
            no_echo = ctx.res_gain[k] * (1.0f - echo_frac);
            lift = nf_eff * no_echo;
            p->g_total[k] = (1.0f - lift) * p->g_total[k] + lift;
        }
    }

    for (k = 0; k < p->n_freqs; ++k) {
        p->spec[k].r = ctx.error_spec[k].r * p->g_total[k];
        p->spec[k].i = ctx.error_spec[k].i * p->g_total[k];
    }

    if (p->enable_cng && ctx.comfort_noise) {
        for (k = 1; k < p->n_freqs - 1; ++k) {
            float n2 = ctx.comfort_noise[k] / PSD_SCALE;
            float ng2 = 1.0f - p->g_aec[k] * p->g_aec[k];
            float amp = sqrtf(n2 > 0.0f ? n2 : 0.0f) *
                        sqrtf(ng2 > 0.0f ? ng2 : 0.0f);
            p->spec[k].r += amp * pipeline_gauss(p);
            p->spec[k].i += amp * pipeline_gauss(p);
        }
    }

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
    mmse_lsa_reset(p->nr);
    memset(p->ola, 0, (size_t)p->frame_size * sizeof(float));
    p->near_hang = 0;
    p->rng = 0x9e3779b9u;
}
/* AUDIO_ALG_PIPELINE_EXAMPLE_END */
```

典型使用：

```c
AudioAlgPipeline pipeline;
if (audio_alg_init(&pipeline, 16000,
                   AEC_PRESET_BALANCED,
                   MMSE_LSA_NR_BALANCED, 1) != 0)
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

Conventional mono pipeline 由 20 ms frame／10 ms hop 推導尺寸：
8 kHz 為 frame/hop/FFT `160/80/256`，16 kHz 為
`320/160/512`，48 kHz 為 `960/480/1024`；`n_freqs = FFT/2 + 1`。
不要把 frame 誤寫成 FFT size，也不要套用 AIAEC 或 4-channel 的
zero-padding-free 512/256、1024/512 grid。呼叫端應從 API query 實際 hop
與 bins，而不是寫死。

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

> `mmse_lsa_config_for_mode()` 底層其實有第四級 `moderate`（介於 mild 與 balanced 之間，`g_min_db -25`，standalone NR 的 `denoise_wav --nr-mode` 可直接選用），但 `aec_nr_pipeline` 的 `parse_nr_mode()`（`pipelines/aec_nr_pipeline.c`）只認得 `"mild"`／`"aggressive"` 字串，其餘（含 `"moderate"`）一律 silently 落回 `balanced`，不會報錯。也就是說目前這條 pipeline 的 `--nr-preset` 實際只有三個可達 preset；要用 moderate 必須自己呼叫 library API。

Audio_ALG pipeline 會在 preset 之上固定套用 `L=150`、`alpha_d=0.95`、`alpha_attack=0.3`、`alpha_decay=alpha_g`，這是針對 AEC residual signal 的結構性 tuning；不要直接用 standalone NR 的所有預設取代。

若要改 `PROD_NE_FLOOR`、far/near threshold、CNG 或 gain combine，這已不是單純 API integration，而是演算法 operating point 變更，應重新跑 far-end-only、near-end-only、double-talk、movement 與 noisy cases。

## 8. Lifecycle 與 thread safety

- 每條串流各有一組 AEC、NR、FFT、OLA 與 RNG state。
- instance 不可由多個 thread 無同步同時 process。
- 換獨立串流前 reset AEC、NR、OLA、near hangover 與 RNG；或直接 destroy/create。
- create 失敗時必須釋放已成功建立的子模組與 buffer。
- malloc reference path（`aec_nr_pipeline`）在初始化時使用 malloc/calloc；process loop 本身不應配置記憶體。
- 單一 caller-owned pool 的 static-memory 版本已完整支援：`aec_nr_pipeline_static` 用三個複合 API
  （`aec_get_mem_size`/`aec_init`、`mmse_lsa_get_mem_size`/`mmse_lsa_init`、`fft_get_mem_size`/`fft_init`）
  從一塊 16-byte 對齊的 pool 切出全部記憶體，init 後零 malloc，輸出與 malloc 版 byte-identical。
  NR 與 AEC 的 static API 都在各自的 `main` branch（runtime 選擇，同一份 `.a` 兩套
  API 並存；`.gitmodules` 兩個 submodule 均釘 `main`）。

## 9. Troubleshooting

| 現象 | 先檢查 | 建議 |
|---|---|---|
| build 找不到 library | 未初始化 submodule 或未跑 `make ... libs` | `git submodule update --init --recursive` 後重建 |
| mic/ref 無法處理 | sample rate 不一致或 WAV format 不支援 | 先轉成同步、同 SR 的 PCM16/float32 mono 測試 |
| 回聲未消除 | ref routing／delay／AEC 尚未 convergence | 先用 `--aec-only` 隔離 AEC 問題 |
| AEC-only 正常，完整 pipeline 傷近端 | NR preset 或 near floor path | 先改 mild，確認不是 `--legacy-amin` |
| 背景噪聲殘留 | NR 還在 init 或 preset 太保守 | 確認開頭噪聲段，逐級試 balanced/aggressive |
| 輸出有洞或不自然靜音 | CNG 關閉或 gain 過深 | A/B 比較有無 `--no-cng`，不要以 NR gain 驅動 CNG |
| 不同檔案結果互相影響 | state 未 reset | 每個獨立 stream reset 或重建全部 instance |
| 尾端樣本變短 | CLI 只處理完整 hop | 上層先 padding，並在輸出後裁回原長度 |
| 想使用 `AudioPipeline*` | 已實作（review F20）| 見 `pipelines/audio_pipeline.h` + `pipelines/README.md`「Board Integration」；本手冊 wrapper 仍可用於未過渡的呼叫端 |

定位問題時建議依序比較：`--aec-only`、完整 production、`--no-cng`、`--legacy-amin`，再用 `DUMP_CTX` 檢查 `E(f)`、`G_res`、`R²` 與 `G_nr`。
