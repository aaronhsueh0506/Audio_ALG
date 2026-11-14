# Mono CLI 與 C 整合範例

## 4. 命令列使用

```bash
# 執行檔在 config-keyed 目錄（由 make 依 backend／旗標雜湊決定）
BIN="$(make -s -C pipelines print-bin-dir)"

# balanced AEC（1）+ balanced NR（2，預設）
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1

# 各自指定 level：AEC 0 mild／1 balanced／2 aggressive／3 no_floor，NR 0 mild／1 moderate／2 balanced／3 aggressive
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 0 \
  --nr-level 0
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 2 \
  --nr-level 3

# 只輸出 linear AEC path（AEC_RESIDUAL_STATE：兩個 AEC 階段照跑），不跑 NR／frequency gain combine／final OLA
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1 --aec-only

# 只關其中一個 gain 來源：--no-nr 只剩 RES；--no-res 只剩 NR（CNG 隨之略過）；
# 兩個都關 = 線性殘差經 iFFT/OLA 直通（與 --aec-only 差一個 hop 的延遲）
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1 --no-nr
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1 --no-res

# NR 的 noise prior 不折入 R²（noise-only NR）；只作比對用
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1 --legacy-amin

# 不加入 comfort noise
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1 --no-cng
```

### 4.1 CLI 參數

| 參數 | 值 | 預設 | 說明 |
|---|---|---|---|
| 第 4 個 positional argument，或 `--aec-level` | 一位數字：`0` mild／`1` balanced／`2` aggressive／`3` no_floor | `1` | AEC level；兩種寫法是同一個設定、只能給一次；其他文字 exit 2 |
| `--nr-level`（別名 `--nr-preset`） | 一位數字：`0` mild／`1` moderate／`2` balanced／`3` aggressive | `2` | NR level；其他文字 exit 2 |
| `--aec-only` | flag | off | 只建 `Aec`（`AEC_RESIDUAL_STATE`，殘留階段照跑），輸出 `linear_hop`；跳過 NR、RES 增益、CNG 與合成 |
| `--no-nr` | flag | off | 不建降噪器，總增益只剩 RES |
| `--no-res` | flag | off | RES 增益視為全 1（comfort noise 隨之略過） |
| `--legacy-amin` | flag | off | NR 的 noise prior 不折入 R²（noise-only NR），只作比對用 |
| `--no-cng` | flag | off | 關閉最終 CNG 注入 |
| `--fft-size` | `256`／`512`／`1024` | 0（rate default） | 覆寫訊號格點的 FFT/frame 長度（16 kHz 唯一可切換的一項，見 [設定參考](../../html/pipelines/mono/aec-nr-res/reference/configuration.html)「訊號格點」表）；不是十進位整數 exit 2 |
| `--debug` | flag | off | 每秒一行 AEC+NR 狀態（read-only，不影響輸出） |
| `--timing` | flag | off | 結束時印每個階段的平均耗時；AEC 的列需要 `make PROFILE=1` |

level 的值不是 `0`..`3` 的一位數字（名稱、`4`、`03`、空字串）一律 exit 2。以 `-` 開頭的 token 會被當成選項，未知的選項被忽略（exit 0、不報錯），所以位置參數寫成 `-1` 不會被拒絕，而是被略過、AEC level 維持預設。

`aec_nr_pipeline_static` 另有 `--print-mem-size [aec-level] [--nr-level <level>] [--aec-only] [--no-nr] [--no-res] [--no-cng]
[--sample-rate <hz>] [--fft-size <n>]`（`--sample-rate`／`--fft-size` 不是十進位整數 exit 2）：印出 pool 預算並真的在那塊 pool 上 init 每個 stage 後結束，不讀 WAV；`aec_nr_pipeline` 沒有這個模式。

`delay_mode`／`delay_num_filters`／`fixed_delay_samples`
（`AecConfig` 的欄位，見 `aec.h`；濾波器長度由取樣率決定，不是欄位）**沒有對應 CLI 旗標**；
兩支 CLI（`aec_nr_pipeline`/`aec_nr_pipeline_static`）目前都只能吃
預設值（`MATCHED`、n=5、−1）。要測試
非預設 delay mode 得直接寫 C 改 `AecConfig` 後呼叫 `aec_get_mem_size()`／`aec_init()`
（或改 CLI 原始碼），見 [設定參考](../../html/pipelines/mono/aec-nr-res/reference/configuration.html)。

`aec_nr_pipeline` 的 level 與數字選項一律嚴格解析（`common/example_cli.h`）：level 只收上表的一位數字，`--fft-size`／`--sample-rate` 只收完整的十進位整數，其他一律 exit 2；`--nr-level`／`--aec-level`／`--fft-size`（`--print-mem-size` 路徑另含 `--sample-rate`）後面直接接另一個 `--` 選項或行尾時視為缺值，exit 2。未知的 dash option 不報錯（被忽略）。DFN2／ULCNet 範例與 `4ch_raw`、`4ch_static`、`4ch_external_static` 走同一份嚴格解析（見 [`pipelines/README.md`](../../../pipelines/README.md)）。

### 4.2 Debug context dump

設定 `DUMP_CTX` 可將每 hop 的中間資料寫成 binary dump，供 port parity 或離線分析：

```bash
DUMP_CTX=pipeline_ctx.bin \
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav 1
```

檔案開頭是兩個 `int`：`n_freqs`、`hop`；後續每幀依序包含渲染基底 B（`aec_get_residual().base_spec`；未觸發守衛的 hop 與 `error_spec` 相同）、`res_gain`（lane 自己的增益）、`r2`、`comfort_noise`、`far_power`、`g_nr`（沒有降噪器時是 unity 全 1 向量）與 output hop。這是診斷格式，沒有穩定 ABI 保證。

### 4.3 WAV contract

- mic/ref 必須能被 audiolib 的 WAV reader 讀取，並具有相同 sample rate。
- 只接受 PCM16 與 IEEE float32（1 到 8 聲道；多聲道只讀第一聲道）；其他格式（例如 PCM32 整數）無法開啟，CLI 以 `Error: cannot open inputs` 結束。
- 處理長度取 mic/ref 中較短者。
- 每次處理所選 signal grid 的一個 hop，最後不足一個 hop 的尾端捨棄。
- pipeline 沒有主動改寫 `AEC_OUT_FLOAT`，因此預設輸出單聲道 PCM16。
- 若要 float32 WAV：`AEC_OUT_FLOAT=1 "$BIN"/aec_nr_pipeline ...`。

## 5. 可編譯的 C integration wrapper

下列範例只使用AEC、NR 與 audiolib 的公開 API，以 malloc 手寫一份與 `pipelines/mono/aec_nr_res` 範例等價的串接（RES 增益取自 AEC 自己的殘留視圖 `v.res_gain`）。上層每次傳入 `audio_alg_hop_size()` 個 float sample。

```c
/* AUDIO_ALG_PIPELINE_EXAMPLE_BEGIN */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "audio_stft.h"   /* AcIstft: 唯一的 inverse FFT + overlap-add */
#include "anr.h"
#include "anr_types.h"

#if !defined(AEC_OUTPUT_GUARD_API_VERSION) || AEC_OUTPUT_GUARD_API_VERSION < 2
#error "aec.h is too old: this wrapper needs the output-guard API (aec_get_residual().base_spec, aec_apply_gain, aec_apply_cng, aec_post_reset_serial, aec_synthesis_window)"
#endif

#define PSD_SCALE                 (32768.0f * 32768.0f)

typedef struct {
    Aec *aec;
    AnrDenoiser *nr;
    AcIstft *istft;
    void *istft_mem;
    int hop;
    int frame_size;
    int fft_size;
    int n_freqs;
    int enable_cng;
    uint32_t reset_serial;   /* 已處理的 aec_post_reset_serial() */
    float *g_nr;
    float *extra;
    Complex *spec;
} AudioAlgPipeline;

void audio_alg_destroy(AudioAlgPipeline *p)
{
    if (!p) return;
    if (p->aec) aec_destroy(p->aec);   /* 釋放 aec_create() 的那一塊；指標之後作廢 */
    anr_destroy(p->nr);
    free(p->istft_mem);
    free(p->g_nr);
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
    p->reset_serial = 0u;
    p->enable_cng = enable_cng != 0;

    aec_config_from_level(&aec_cfg, aec_level, sample_rate);
    aec_cfg.residual = AEC_RESIDUAL_FULL;   /* 預設值：殘留 gain 與 comfort noise 都由 Aec 提供 */

    p->aec = aec_create(&aec_cfg);   /* 一塊 16-byte 對齊的 heap 配置，Aec 結構與 pool 都在裡面 */
    if (!p->aec)
        goto fail;
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
    p->extra = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    p->spec = (Complex *)malloc((size_t)p->n_freqs * sizeof(Complex));

    if (!p->nr || !p->istft || !p->g_nr || !p->extra || !p->spec)
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
    AecResidualView v;
    const float *residual_psd_audio = NULL;
    const float *nr_gain;
    uint32_t serial;
    int k;

    if (!p || !p->aec || !p->nr || !mic || !ref || !out)
        return -1;

    if (aec_process_linear(p->aec, mic, ref) != AEC_OK ||
        aec_process_residual(p->aec, NULL) != AEC_OK)
        return -1;
    memset(&v, 0, sizeof(v));
    v.struct_size = sizeof(v);
    if (aec_get_residual(p->aec, &v) != AEC_OK || !v.res_gain || v.n_freqs != p->n_freqs)
        return -1;

    /* 渲染基底 B（v.base_spec）：未觸發輸出守衛的 hop 與 v.error_spec 逐位元相同，守衛觸發時是 error_spec + echo_spec；
       降噪器的估計與增益都用 B（v.error_spec / v.formed_hop 是分析 seam，不渲染）。 */

    /* AEC 在這個 hop 內做了硬重置：在 ac_istft_push 之前清 OLA 尾巴（Aec 自己的 comfort noise 產生器已重設種子）；
       不要因此重置 AEC，降噪器不動。 */
    serial = aec_post_reset_serial(p->aec);
    if (serial != p->reset_serial) {
        p->reset_serial = serial;
        ac_istft_reset(p->istft);
    }

    /* 與範例 main.c 的 mono_nr_gain() 相同：R^2 只在 usable_linear != 0 時摺進降噪器的 noise floor */
    if (v.r2 && v.usable_linear) {
        for (k = 0; k < p->n_freqs; ++k)
            p->extra[k] = v.r2[k] / PSD_SCALE;
        residual_psd_audio = p->extra;
    }

    nr_gain = anr_process_gain_with_residual_psd(p->nr, v.base_spec, residual_psd_audio);
    if (!nr_gain)
        return -1;
    memcpy(p->g_nr, nr_gain, (size_t)p->n_freqs * sizeof(float));   /* g_nr is also written to the diagnostic dump */

    /* spec = B * min(g_nr, G_res)；同一個 g_nr 也交給 aec_apply_cng：只填 RES 挖掉的 bin，位準乘該 bin 的 NR 增益（下限 -10 dB） */
    if (aec_apply_gain(p->aec, p->g_nr, p->spec) != AEC_OK)
        return -1;
    if (p->enable_cng && aec_apply_cng(p->aec, p->g_nr, p->spec) != AEC_OK)
        return -1;

    /* 一個 hop 的輸出：inverse FFT、窗、overlap-add 都在 AcIstft 內。 */
    return ac_istft_push(p->istft, p->spec, out) == p->hop ? 0 : -1;
}

void audio_alg_reset(AudioAlgPipeline *p)
{
    if (!p || !p->aec || !p->nr) return;
    aec_reset(p->aec);
    anr_reset(p->nr);
    ac_istft_reset(p->istft);
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

這段 wrapper 故意保持在 application 層，沒有宣稱新的 public ABI。合成由應用端持有的 `AcIstft`（audiolib）負責：RES／降噪增益乘上渲染基底 `aec_get_residual().base_spec`（由 `aec_apply_gain()` / `aec_apply_cng()` 完成）之後交給 `ac_istft_push()`，AEC 本身只送出 `aec_process_linear()` / `aec_process_residual()` 的頻域結果。若要修改 production constant、CNG 或 failure policy，應在自己的 wrapper 版本化並重新跑品質評測。

## 6. Buffer、尺度與 lifetime

| 資料 | 尺度／尺寸 | 規則 |
|---|---|---|
| mic/ref/out | float PCM，`hop` | 一次正好一個 grid hop，建議幅度 `[-1, 1]` |
| `error_spec` | complex audio amplitude，`n_freqs` | AEC instance 內部 pointer |
| `res_gain` | amplitude gain `[0,1]` | 與 `G_nr` 逐 bin 比較 |
| `r2` | int16² PSD | 除以 `32768²` 才能和 `|E|²` 合併 |
| `comfort_noise` | int16² PSD | 除以 `32768²` 後取平方根得到 amplitude |
| `G_nr`／`G_total` | amplitude gain | 直接乘 complex spectrum |

`AecResidualView` pointer 只保證在下一次 AEC process/reset/destroy 前有效。不要修改、free 或跨 hop 保存 pointer；如需非同步分析，複製內容。

Conventional mono pipeline 跟 4-channel/AIAEC 一樣是 zero-padding-free
50%-overlap grid（frame == FFT、hop == FFT/2）：8 kHz 為 frame(=FFT)/hop `256/128`，16 kHz 預設為
`256/128`（512/256 也是明確支援的可選 grid），48 kHz 為 `1024/512`；
`n_freqs = FFT/2 + 1`。呼叫端應從 API query 實際 hop 與 bins，而不是寫死
（16 kHz 有兩種合法 grid，寫死任一組數字對另一組就是錯的）。

完整 pipeline 的 final IFFT/OLA 增加約一個 grid hop 延遲；`--aec-only` 走 linear time output，不經這段 final OLA。另需把裝置 I/O buffer、resampler 與 OS scheduling latency 加入產品總預算。

## 7. Level 與 tuning 原則

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名（mild／moderate／balanced／aggressive／no_floor）與整數（AEC 0..3、預設 1；NR 0..3、預設 2）兩邊相同；C 的 CLI 與函式庫只收整數（`AecLevel`／`AnrLevel`），不解析名稱。CLI 的舊旗標 `--nr-preset` 保留為 `--nr-level` 的別名。

AEC level：

| Level | 值 | 方向 |
|---|---|---|
| mild | 0 | 近端保留優先 |
| balanced | 1 | 預設 production operating point |
| aggressive | 2 | 回聲抑制優先 |
| no_floor | 3 | 關閉分裂地板（`min_gain_split_floor_enabled = 0`） |

NR level（`--nr-level`，見 §4.1）：

| Level | 值 | 方向 |
|---|---|---|
| mild | 0 | 最保守的背景降噪 |
| moderate | 1 | 介於 mild 與 balanced 之間 |
| balanced | 2 | 預設平衡 |
| aggressive | 3 | 最強背景降噪 |

> `anr_config_for_level()` 有四級：0 `mild`／1 `moderate`／2 `balanced`／3 `aggressive`。`aec_nr_pipeline`（`pipelines/mono/aec_nr_res/main.c`）只收這四個數字，其餘文字 exit 2。

Audio_ALG 的範例程式（`pipelines/mono/aec_nr_res/main.c` 等）直接用 NR 函式庫的 level：`anr_config_for_level_grid(sample_rate, fft_size, nr_level)`，不另外覆寫欄位（最小值追蹤視窗 `L` 為函式庫預設，約 512 ms；16 kHz 256/128 為 64 hops），所以 `anr_set_level()` 可以直接在執行中換強度。

若要改 CNG 或 gain combine（例如在 `min(G_nr, G_res)` 之外再加任何 floor），這已不是單純 API integration，而是演算法 operating point 變更，應重新跑 far-end-only、near-end-only、double-talk、movement 與 noisy cases。

## 8. Lifecycle 與 thread safety

- 每條串流各有一組 AEC、NR、FFT、OLA 與 RNG state。
- instance 不可由多個 thread 無同步同時 process。
- 換獨立串流前 reset AEC（`aec_reset()` 連同它自己的 comfort-noise 產生器）、NR 與合成 OLA（三件一起，見 [mono 手冊 §3](lifecycle.md#3-lifecycle)）；或直接 destroy/create。
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

> `--aec-only` 的 lane 跑同一條線性路徑：它以 `AEC_RESIDUAL_STATE` 建立、殘留階段照跑，輸出的 `linear_hop` 與完整 chain 的 lane（`FULL`）逐位元相同（`test_mono_chain` 在動態回音路徑上逐 hop 驗證，16 kHz/256、16 kHz/512、48 kHz/1024 各 400 hop）。拿它隔離問題時，它就是完整 chain 的前半段。

[返回本主題](README.md)
