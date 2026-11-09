# Mono CLI 與 C 整合範例

## 4. 命令列使用

```bash
# 執行檔在 config-keyed 目錄（由 make 依 backend／旗標雜湊決定）
BIN="$(make -s -C pipelines print-bin-dir)"

# balanced AEC + balanced NR
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced

# 各自指定 level
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav mild \
  --nr-level mild
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav aggressive \
  --nr-level aggressive

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
| 第 4 個 positional argument | `mild`／`balanced`／`aggressive`／`no_floor`（或 `0`..`3`） | `balanced` | AEC level；未知字串退回 `balanced` |
| `--nr-level`（別名 `--nr-preset`） | `mild`／`moderate`／`balanced`／`aggressive` | `balanced` | NR level |
| `--aec-only` | flag | off | 跳過 NR 與最終 gain combine |
| `--no-nr` | flag | off | 不建降噪器，總增益只剩 RES |
| `--no-res` | flag | off | RES 增益視為全 1（comfort noise 隨之略過） |
| `--legacy-amin` | flag | off | NR 的 noise prior 不折入 R²（noise-only NR），只作比對用 |
| `--no-cng` | flag | off | 關閉最終 CNG 注入 |
| `--fft-size` | `256`／`512`／`1024` | 0（rate default） | 覆寫訊號格點的 FFT/frame 長度（16 kHz 唯一可切換的一項，見 pipeline_mono.html「訊號格點」表） |
| `--debug` | flag | off | 每秒一行 AEC+NR 狀態（read-only，不影響輸出） |
| `--timing` | flag | off | 結束時印每個階段的平均耗時；AEC 的列需要 `make PROFILE=1` |

`aec_nr_pipeline_static` 另有 `--print-mem-size [aec-level] [--nr-level ...] [--aec-only] [--no-nr] [--no-res]
[--sample-rate <hz>] [--fft-size <n>]`：印出 pool 預算並真的在那塊 pool 上 init 每個 stage 後結束，不讀 WAV；`aec_nr_pipeline` 沒有這個模式。

`delay_mode`／`delay_num_filters`／`fixed_delay_samples`／`filter_length`
（`AecConfig` 的欄位，見 `aec.h`）**沒有對應 CLI 旗標**；
兩支 CLI（`aec_nr_pipeline`/`aec_nr_pipeline_static`）目前都只能吃
預設值（`MATCHED`、n=5、−1）。要測試
非預設 delay mode 得直接寫 C 改 `AecConfig` 後呼叫 `aec_get_mem_size()`／`aec_init()`
（或改 CLI 原始碼），見 `pipeline_mono.html` API 節。

目前 `mono_aec_nr_res` parser 對未知 positional level 與未知 `--nr-level`（或別名 `--nr-preset`）名稱會退回 balanced，且不會為所有未知 dash option 報錯；`--nr-level`／`--fft-size`（`aec_nr_pipeline_static` 的 `--print-mem-size` 路徑另含 `--sample-rate`）後面直接接另一個 `--` 選項或行尾時視為缺值，exit 2。`mcab_external_static` 則對未知 level 名稱 exit 2。產品或自動化腳本應只傳上表列出的值，不要依賴 silent fallback。
只有 mono 的 smoke CLI（`mono/aec_nr_res/main.c`）的 `--fft-size`、`--sample-rate` 以 `atoi()` 轉換、不嚴格檢查（`12x` 當成 12）；DFN2／ULCNet 範例、`mcab_raw`、`mcab_static`、`mcab_external_static` 走 `common/example_cli.h` 的嚴格解析，非法數字 exit 2（見 [`pipelines/README.md`](../../../pipelines/README.md)）。

### 4.2 Debug context dump

設定 `DUMP_CTX` 可將每 hop 的中間資料寫成 binary dump，供 port parity 或離線分析：

```bash
DUMP_CTX=pipeline_ctx.bin \
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced
```

檔案開頭是兩個 `int`：`n_freqs`、`hop`；後續每幀依序包含渲染基底 B（`aec_get_output_base()`；未觸發守衛的 hop 與 `error_spec` 相同）、`res_gain`（lane 自己的增益）、`r2`、`comfort_noise`、`far_power`、`g_nr`（沒有降噪器時是 unity 全 1 向量）與 output hop。這是診斷格式，沒有穩定 ABI 保證。

### 4.3 WAV contract

- mic/ref 必須能被 audiolib 的 WAV reader 讀取，並具有相同 sample rate。
- 只接受 PCM16 與 IEEE float32（1 到 8 聲道；多聲道只讀第一聲道）；其他格式（例如 PCM32 整數）無法開啟，CLI 以 `Error: cannot open inputs` 結束。
- 處理長度取 mic/ref 中較短者。
- 每次處理所選 signal grid 的一個 hop，最後不足一個 hop 的尾端捨棄。
- pipeline 沒有主動改寫 `AEC_OUT_FLOAT`，因此預設輸出單聲道 PCM16。
- 若要 float32 WAV：`AEC_OUT_FLOAT=1 "$BIN"/aec_nr_pipeline ...`。

## 5. 可編譯的 C integration wrapper

下列範例只使用AEC、NR 與 audiolib 的公開 API，以 malloc 手寫一份與 `pipelines/mono/aec_nr_res` 範例等價的串接（RES 增益取自 AEC 自己的 context）。上層每次傳入 `audio_alg_hop_size()` 個 float sample。

```c
/* AUDIO_ALG_PIPELINE_EXAMPLE_BEGIN */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"   /* aec_cng_fill */
#include "audio_stft.h"   /* AcIstft: 唯一的 inverse FFT + overlap-add */
#include "utility/simd_kernels.h"   /* sk_capply_gain_f32 */
#include "anr.h"
#include "anr_types.h"

#if !defined(AEC_OUTPUT_GUARD_API_VERSION) || AEC_OUTPUT_GUARD_API_VERSION < 1
#error "aec.h is too old: this wrapper needs the output-guard API (aec_get_output_base, aec_post_reset_serial, aec_synthesis_window)"
#endif

#define PSD_SCALE                 (32768.0f * 32768.0f)

typedef struct {
    Aec *aec;
    int aec_ready;
    AnrDenoiser *nr;
    AcIstft *istft;
    void *istft_mem;
    int hop;
    int frame_size;
    int fft_size;
    int n_freqs;
    int enable_cng;
    uint32_t rng;
    uint32_t reset_serial;   /* 已處理的 aec_post_reset_serial() */
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
    free(p->istft_mem);
    free(p->g_nr);
    free(p->g_total);
    free(p->g_aec);
    free(p->extra);
    free(p->spec);
    memset(p, 0, sizeof(*p));
}

int audio_alg_init(AudioAlgPipeline *p, int sample_rate,
                   AecLevel aec_level, AnrLevel nr_level,
                   int enable_cng)
{
    AecConfig aec_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t synth_bytes;
    const float *synth_window = NULL;
    int synth_window_len = 0;

    if (!p) return -1;
    memset(p, 0, sizeof(*p));
    p->rng = AEC_CNG_SEED;
    p->reset_serial = 0u;
    p->enable_cng = enable_cng != 0;

    aec_config_from_level(&aec_cfg, aec_level, sample_rate);
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

    /* Grid-explicit entry point: NR must resolve the SAME fft_size the AEC just picked
     * (the 2-arg anr_config_for_level() uses NR's own per-rate default, e.g. 128 at 8 kHz
     * where the AEC uses 256). L/alpha_* are already retimed for this grid; do not override. */
    nr_cfg = anr_config_for_level_grid(sample_rate, p->fft_size, nr_level);

    p->nr = anr_create(&nr_cfg);

    /* 合成（inverse FFT + 窗 + overlap-add）是 audiolib 的 AcIstft，由應用端持有；fft 由它自己配置。
     * 窗表向 AEC 借用（aec_synthesis_window()，唯讀、壽命與 Aec 相同，AcIstft 不另配置窗表）：
     * 要與 aec_process() 逐位元相同就不能用 audiolib 預設的 sqrt-Hann（兩者差最後一個 float bit）。 */
    if (aec_synthesis_window(p->aec, &synth_window, &synth_window_len) != 0 ||
        synth_window_len != p->fft_size)
        goto fail;
    ac_stft_config_default(&synth_cfg, p->fft_size);
    synth_cfg.window = synth_window;
    synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    if (synth_bytes == 0) goto fail;
    p->istft_mem = aligned_alloc(16, (synth_bytes + 15u) / 16u * 16u);
    if (!p->istft_mem) goto fail;
    p->istft = ac_istft_init(p->istft_mem, synth_bytes, &synth_cfg);

    p->g_nr = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->g_total = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->g_aec = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->extra = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->spec = (Complex *)malloc((size_t)p->n_freqs * sizeof(Complex));

    if (!p->nr || !p->istft || !p->g_nr || !p->g_total || !p->g_aec ||
        !p->extra || !p->spec)
        goto fail;

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
    AecOutputBase base;
    const float *extra = NULL;
    uint32_t serial;
    int k;

    if (!p || !p->aec_ready || !p->nr || !mic || !ref || !out)
        return -1;

    aec_process_context(p->aec, mic, ref);
    aec_get_res_context(p->aec, &ctx);
    if (!ctx.error_spec || !ctx.res_gain || ctx.n_freqs != p->n_freqs)
        return -1;

    /* 渲染基底 B：未觸發輸出守衛的 hop 與 ctx.error_spec 逐位元相同，守衛觸發時是 error_spec + echo_spec；
       降噪器的估計與增益都用 B（ctx.error_spec / formed_hop 是分析 seam，不渲染）。 */
    memset(&base, 0, sizeof(base));
    base.struct_size = sizeof(base);
    if (aec_get_output_base(p->aec, &base) != 0)
        return -1;

    /* AEC 在這個 hop 內做了硬重置：在第一次 CNG 抽樣與 ac_istft_push 之前清 OLA 尾巴、重設 CNG 種子；
       不要因此重置 AEC，降噪器不動。 */
    serial = aec_post_reset_serial(p->aec);
    if (serial != p->reset_serial) {
        p->reset_serial = serial;
        ac_istft_reset(p->istft);
        p->rng = AEC_CNG_SEED;
    }

    /* 與範例 main.c 的 mono_nr_gain() 相同：R^2 只在 usable_linear != 0 時摺進降噪器的 noise floor */
    if (ctx.r2 && ctx.usable_linear) {
        for (k = 0; k < p->n_freqs; ++k)
            p->extra[k] = ctx.r2[k] / PSD_SCALE;
        extra = p->extra;
    }

    if (anr_process_gain(p->nr, base.base_spec, extra, p->g_nr) < 0)
        return -1;

    for (k = 0; k < p->n_freqs; ++k) {
        p->g_aec[k] = ctx.res_gain[k];
        p->g_total[k] = p->g_nr[k] < ctx.res_gain[k]
                      ? p->g_nr[k] : ctx.res_gain[k];
    }

    sk_capply_gain_f32(p->spec, base.base_spec, p->g_total, p->n_freqs);

    if (p->enable_cng && ctx.comfort_noise)   /* 只填 RES 挖掉的 bin，位準乘該 bin 的 NR 增益（下限 -10 dB） */
        aec_cng_fill(&p->rng, p->spec, ctx.comfort_noise, p->g_aec, p->g_nr,
                     0.31622777f, p->n_freqs);

    /* 一個 hop 的輸出：inverse FFT、窗、overlap-add 都在 AcIstft 內。 */
    return ac_istft_push(p->istft, p->spec, out) == p->hop ? 0 : -1;
}

void audio_alg_reset(AudioAlgPipeline *p)
{
    if (!p || !p->aec_ready || !p->nr) return;
    aec_reset(p->aec);
    anr_reset(p->nr);
    ac_istft_reset(p->istft);
    p->rng = AEC_CNG_SEED;
    p->reset_serial = aec_post_reset_serial(p->aec);
}
/* AUDIO_ALG_PIPELINE_EXAMPLE_END */
```

典型使用：

```c
AudioAlgPipeline pipeline;
if (audio_alg_init(&pipeline, 16000,
                   AEC_LEVEL_BALANCED,
                   ANR_LEVEL_BALANCED, 1) != 0)
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

這段 wrapper 故意保持在 application 層，沒有宣稱新的 public ABI。合成由應用端持有的 `AcIstft`（audiolib）負責：RES／降噪增益乘上渲染基底 `aec_get_output_base()` 之後交給 `ac_istft_push()`，AEC 本身只送出 `aec_process_context()` 的頻域 context。若要修改 production constant、CNG RNG 或 failure policy，應在自己的 wrapper 版本化並重新跑品質評測。

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

## 7. Level 與 tuning 原則

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名（mild／moderate／balanced／aggressive／no_floor）與整數（AEC 0..3、預設 1；NR 0..3、預設 2）兩邊相同。舊名 `AecPreset`／`AEC_PRESET_*`、`AnrMode`／`ANR_MODE_*` 仍保留在公開 header（同值），舊函式名已移除；CLI 的舊旗標 `--nr-preset` 保留為 `--nr-level` 的別名。

AEC level：

| Level | 方向 |
|---|---|
| mild | 近端保留優先 |
| balanced | 預設 production operating point |
| aggressive | 回聲抑制優先 |

NR level（`--nr-level`，見 §4.1）：

| Level | 方向 |
|---|---|
| mild | 最保守的背景降噪 |
| balanced | 預設平衡 |
| aggressive | 最強背景降噪 |

> `anr_config_for_level()` 有四級：`mild`／`moderate`／`balanced`／`aggressive`（`moderate` 介於 mild 與 balanced 之間）。`aec_nr_pipeline` 的 `parse_nr_level()`（`pipelines/mono/aec_nr_res/main.c`）經 `anr_nr_level_from_name()` 接受這四個名稱（或 `0`..`3`），其餘字串一律 silently 落回 `balanced`，不會報錯。

Audio_ALG 的範例程式（`pipelines/mono/aec_nr_res/main.c` 等）直接用 NR 函式庫的 level：`anr_config_for_level_grid(sample_rate, fft_size, nr_level)`，不另外覆寫欄位（最小值追蹤視窗 `L` 為函式庫預設，約 512 ms；16 kHz 256/128 為 64 hops），所以 `anr_set_level()` 可以直接在執行中換強度。

若要改 CNG 或 gain combine（例如在 `min(G_nr, G_res)` 之外再加任何 floor），這已不是單純 API integration，而是演算法 operating point 變更，應重新跑 far-end-only、near-end-only、double-talk、movement 與 noisy cases。

## 8. Lifecycle 與 thread safety

- 每條串流各有一組 AEC、NR、FFT、OLA 與 RNG state。
- instance 不可由多個 thread 無同步同時 process。
- 換獨立串流前 reset AEC、NR、合成 OLA 與 comfort-noise RNG（四件一起，見 [mono 手冊 §3](lifecycle.md#3-lifecycle)）；或直接 destroy/create。
- create 失敗時必須釋放已成功建立的子模組與 buffer。
- split-pool reference path（`aec_nr_pipeline`）在初始化時用 `posix_memalign()` 配置兩塊（AEC 一塊、降噪器／合成／暫存一塊）；process loop 本身不配置記憶體。
- 單一 caller-owned pool 的 static-memory 版本已完整支援：`aec_nr_pipeline_static` 各自查大小、各自 init
  （`aec_get_mem_size`/`aec_init`、`anr_get_mem_size`/`anr_init`、`ac_istft_get_mem_size`/`ac_istft_init`），
  從一塊 16-byte 對齊的 pool 依序切出全部記憶體，init 後零 malloc；`pipelines/mono/aec_nr_res/tests/test_mono_chain.c` 以逐位元比對兩支 CLI 與手寫 oracle
  （單一 pool 與 split 版的輸出因此相等，範圍為該測試涵蓋的選項組合）。
  NR 與 AEC 的 heap 與 static API 並存於同一份 `.a`（runtime 選擇）。

## 9. Troubleshooting

| 現象 | 先檢查 | 建議 |
|---|---|---|
| build 找不到 library | 缺少 pinned sibling checkout 或 library 未建置 | 在 repo 根執行 `make -f lib/deps.mk bootstrap-deps`、`make -C pipelines check-pins`，再建置 pipeline |
| mic/ref 無法處理 | sample rate 不一致或 WAV format 不支援 | 先轉成同步、同 SR 的 PCM16/float32 mono 測試 |
| 回聲未消除 | ref routing／delay／AEC 尚未 convergence | 先用 `--aec-only` 隔離 AEC 問題 |
| AEC-only 正常，完整 pipeline 傷近端 | NR level 或 RES gain | 先改 mild；再用 `--no-nr` / `--no-res` 隔離是哪個 gain 來源 |
| 背景噪聲殘留 | NR 還在 init 或 level 太保守 | 確認開頭噪聲段，逐級試 balanced/aggressive |
| 輸出有洞或不自然靜音 | CNG 關閉或 gain 過深 | A/B 比較有無 `--no-cng`；CNG 只填 `G_res` 切掉的 bin，位準乘該 bin 的 NR gain（下限 −10 dB） |
| 不同檔案結果互相影響 | state 未 reset | 每個獨立 stream reset 或重建全部 instance |
| 尾端樣本變短 | CLI 只處理完整 hop | 上層先 padding，並在輸出後裁回原長度 |
| 想用現成的 mono 整合寫法 | — | 見 `pipelines/mono/aec_nr_res/main.c` + `pipelines/reference/build.md`「Board Integration」；本手冊第 5 節的 wrapper 是直接組合 AEC／NR 的範例 |

定位問題時建議依序比較：`--aec-only`、完整 production、`--no-cng`、`--legacy-amin`，再用 `DUMP_CTX` 檢查渲染基底 `B(f)`、`G_res`、`R²` 與 `G_nr`。

[返回本主題](README.md)
