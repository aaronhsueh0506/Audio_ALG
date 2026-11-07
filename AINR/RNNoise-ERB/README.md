# RNNoise-ERB

以 RNNoise v0.2 的 Conv+GRU 方向為基礎的本地噪音抑制模型。出貨設定只讓網路
使用 mean-normalized log-ERB；程式仍保留可選的低頻 complex spectrum 分支，
但 `config.ini` 預設 `use_complex_input = false`。本模型並非官方 RNNoise v0.2
或 DeepFilterNet 的逐層相容移植。

## 架構

- **ERB 輸入**: 22 個 log-energy bands，做每-band causal EMA mean norm
- **Complex 輸入（可選、預設關閉）**: 0–4 kHz 的 129 個 real/imag bins，做每-bin magnitude EMA unit norm
- **模型**: ERB temporal conv → 3 層 GRU(128)；開啟 complex 分支時才加入 frequency encoder/temporal conv 與 fusion
- **輸出**: 22 個 gain masks [0, 1]
- **參數量**: 321,110（預設純 ERB、無 VAD head）；開啟 complex 分支為 376,254
- **額外 lookahead**: 16ms（`lookahead_frames=1`，gain 對應 3-frame window 中間幀）

### Feature preprocessing

目前 feature contract 為 `log_erb_dfn_mean_cplx_unit_0_4k_v8`：

```text
normalized complex STFT
├─ power → triangular ERB → 10*log10(E+1e-10)
│          → per-band EMA mean → (dB - mean) / 40 → model input
└─ bins 0..4 kHz → per-bin |X[k]| EMA
                    → real/imag / sqrt(EMA+eps) → optional branch (off by default)
```

ERB 路徑對穩態訊號會逐步趨近 0，這是 DeepFilterNet 公式的預期行為；但
DeepFilterNet 同時使用 complex feature，而本模型的出貨設定已關閉該分支。
因此純 ERB 模式不保留穩態頻譜的 absolute level、帶內細節或相位；這是刻意的
低算量 bake-off 設定，也是與 complex/hybrid 模型比較時必須納入解讀的限制。
`X[k] / sqrt(EMA(|X[k]|))` 不是完全 gain-invariant；穩態下輸入振幅放大 `a` 倍，
complex feature 約放大 `sqrt(a)` 倍。只有啟用 complex 分支時，模型才會使用這項
資訊。同一 stream 分 chunk 處理時必須傳遞 ERB EMA 與 GRU state；若 complex
分支啟用，還要傳遞 complex EMA。不同 WAV 之間必須重置。
兩路公式、state 初始值與先更新再輸出的順序精確對齊原作 `libDF`——
兩路都沒有 ERB 或 complex clip：upstream `libDF::band_mean_norm_erb`/`band_unit_norm`
與本 repo 自己的 `AINR/DeepFilterNet2` port 皆不 clip，兩路是 byte-for-byte
忠實移植，沒有額外的安全界線。
公式來源：[Rikorose/DeepFilterNet `libDF/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs)。

Feature 常數在 `config.ini [feature]`；C 端沒有編譯期常數，由應用端把同名值填入
`RNNoiseGridSpec`，再以 `rnnoise_grid_make()` 建成 `RNNoiseGrid`。修改後需同步 spec 並重訓。

## C 串流整合

`process.c` 只處理頻譜、維護模型特徵與 gain 處理；FFT、IFFT、窗與 overlap-add 由
`audiolib/common` 的 `AcStft`／`AcIstft` 執行，由應用端建立、持有與重置。函式庫不配置記憶體
（`rnnoise_get_mem_size()` → 呼叫端記憶體 → `rnnoise_init_ex()`）、不讀檔、也不持有 GRU state：
ERB 矩陣是應用端讀入後以指標交入的資料，float host 在 config 註冊三個 GRU 張量，板端不註冊。

Framing 契約（`process.h` 的標頭註解；函式庫不包裝它）：`ac_stft_config_default(&sc, grid.n_fft)`，
`sc.spec_scale = grid.stft_scale`（`1/sqrt(n_fft)`），每個 hop 一個完整 `n_fft` frame（hop = `n_fft / 2`），
其餘維持預設：center 0、兩側 periodic sqrt-Hann、`AC_OLA_ACCUMULATE`（不除窗包絡，與 Python `train.istft` 相同）、
`ispec_scale = 1.0f`。

```c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_stft.h"
#include "process.h"

enum { GRU_HIDDEN = 128 };    /* the exported graph's h1, h2, h3 are [1,1,128] each: the application's shapes */

static void *alloc16(size_t bytes) { return aligned_alloc(16, (bytes + 15) & ~(size_t)15); }

/* The accelerator: erb[3][n_bands], spec[3][2][spec_bins], h1..h3 -> band_gains[n_bands]; the
 * hidden outputs are written to the same buffers as the inputs. */
typedef int (*RunModel)(const float *erb, const float *spec, float *h1, float *h2, float *h3,
                        float *band_gains);

/* erb_fwd: bin-major [n_bins][n_bands]; erb_inv: band-major [n_bands][n_bins]. The application loads
 * both and keeps them alive; the library borrows the pointers. */
static int run_rnnoise(const float *erb_fwd, const float *erb_inv, RunModel run_model,
                       const float *in, float *out, long hops)
{
    const RNNoiseGridSpec spec = {            /* the shipped model: config.ini [signal] [feature] */
        .sample_rate = 16000, .n_fft = 512, .n_bands = 22, .min_bins_per_band = 2,
        .lookahead_frames = 1, .spec_max_hz = 4000,
        .erb_norm_tau_sec = 1.0, .erb_norm_init_lo_db = -20.0f, .erb_norm_init_hi_db = -45.0f,
        .erb_norm_scale_db = 40.0f, .spec_norm_tau_sec = 1.0,
        .spec_norm_init_lo = 0.04f, .spec_norm_init_hi = 0.008f, .spec_norm_eps = 1e-12f,
    };
    RNNoiseGrid grid;
    int status = rnnoise_grid_make(&grid, &spec);
    if (status != RNNOISE_GRID_OK) {
        fprintf(stderr, "grid refused: %s\n", rnnoise_grid_status_text(status));
        return -1;
    }

    int rc = -1;
    const size_t nb = (size_t)grid.n_bins, erb_n = rnnoise_erb_window_elements(&grid);
    float *h = calloc(3 * GRU_HIDDEN, sizeof *h);                          /* h1, h2, h3 */
    float *buf = malloc(sizeof *buf * (9 * nb + grid.n_bands + erb_n + rnnoise_spec_window_elements(&grid)));
    float *hist_re = buf, *hist_im = hist_re + 3 * nb;                     /* the last three spectra */
    float *cur_re = hist_im + 3 * nb, *cur_im = cur_re + nb, *bin_gains = cur_im + nb;
    float *band_gains = bin_gains + nb, *erb = band_gains + grid.n_bands, *spec_win = erb + erb_n;
    void *stft_mem = NULL, *istft_mem = NULL, *pool = NULL;
    AcStft *stft = NULL;
    AcIstft *istft = NULL;
    RNNoiseState *st = NULL;
    if (!h || !buf) goto done;

    AcStftConfig sc;                          /* the framing: audiolib */
    ac_stft_config_default(&sc, grid.n_fft);
    sc.spec_scale = grid.stft_scale;          /* ispec_scale stays 1.0f: rnnoise_apply_gains() scales */
    size_t stft_bytes = ac_stft_get_mem_size(&sc), istft_bytes = ac_istft_get_mem_size(&sc);
    stft_mem = alloc16(stft_bytes);
    istft_mem = alloc16(istft_bytes);
    if (!stft_mem || !istft_mem) goto done;
    stft = ac_stft_init(stft_mem, stft_bytes, &sc);
    istft = ac_istft_init(istft_mem, istft_bytes, &sc);
    if (!stft || !istft) goto done;

    RNNoiseConfig cfg;
    RNNoiseMemReq req;
    if (rnnoise_config_defaults(&cfg, &grid) != 0) goto done;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.h1 = h;                  cfg.h1_elements = GRU_HIDDEN;    /* float host: register the state, */
    cfg.h2 = h + GRU_HIDDEN;     cfg.h2_elements = GRU_HIDDEN;    /* all three or none; a board registers nothing */
    cfg.h3 = h + 2 * GRU_HIDDEN; cfg.h3_elements = GRU_HIDDEN;
    if (rnnoise_get_mem_size(&cfg, &req) != 0) goto done;
    pool = alloc16(req.bytes);
    st = pool ? rnnoise_init_ex(pool, (size_t)req.bytes, &cfg, &req) : NULL;
    if (!st) goto done;

    rc = 0;
    for (long t = 0; t < hops; ++t) {
        float *slot_re = hist_re + (size_t)(t % 3) * nb, *slot_im = hist_im + (size_t)(t % 3) * nb;
        float *hop_out = out + t * grid.hop;
        ac_stft_push_planar(stft, in + t * grid.hop, slot_re, slot_im);
        int ready = rnnoise_compute_features(st, slot_re, slot_im, erb, spec_win);
        if (ready < 0) { rc = -1; break; }
        for (int b = 0; b < grid.n_bands; ++b) band_gains[b] = 1.0f;          /* no 3-frame window yet: unit gains */
        if (ready) {
            for (int b = 0; b < grid.n_bands; ++b) band_gains[b] = NAN;       /* an unwritten output is refused */
            int run_rc = run_model(erb, spec_win, h, h + GRU_HIDDEN, h + 2 * GRU_HIDDEN, band_gains);
            /* rnnoise_frame_finish() always runs, even when run_model() itself reports failure:
             * the NaN-filled band_gains then still trips its own finite check, which zeroes the
             * registered state and advances the generation (feature ring and OLA timing
             * untouched) -- skipping it here would let a failed call's partial, finite state
             * write survive into the next hop undetected. */
            int finish_rc = rnnoise_frame_finish(st, band_gains);
            if (run_rc != 0 || finish_rc != 0)
                for (int b = 0; b < grid.n_bands; ++b) band_gains[b] = 1.0f;   /* refused: unit gains */
            rnnoise_apply_atten_lim(st, band_gains, 0.0f);                     /* optional; <= 0 disables */
        }
        rnnoise_expand_gains(st, band_gains, bin_gains);
        /* the gains of the window [t-2, t-1, t] belong to spectrum t - lookahead_frames; the spectra before the
         * first window are synthesised too (unit gains), or their overlap would be missing from the first output */
        long s = t - grid.lookahead_frames;
        if (s < 0) { memset(hop_out, 0, sizeof *hop_out * grid.hop); continue; }   /* no spectrum to emit yet */
        memcpy(cur_re, hist_re + (size_t)(s % 3) * nb, sizeof *cur_re * nb);
        memcpy(cur_im, hist_im + (size_t)(s % 3) * nb, sizeof *cur_im * nb);
        rnnoise_apply_gains(st, cur_re, cur_im, bin_gains);                    /* x stft_unscale inside */
        ac_istft_push_planar(istft, cur_re, cur_im, hop_out);
    }
done:
    free(pool); free(istft_mem); free(stft_mem); free(buf); free(h);
    return rc;
}
```

- 出貨模型是 16 kHz、FFT/window 512、hop 256、22 bands、lookahead 1（上面的 `RNNoiseGridSpec`）。grid 不是編譯期常數：
  `rnnoise_grid_make()` 依 `RNNoiseGridSpec` 檢查並推導 `hop`、`n_bins`、`spec_bins`、`conv_delay`、兩個 EMA 係數與兩個縮放；
  `test/test_rnnoise_framing.c` 另以 16000/256、48000/1024、8000/128 三組 grid 驗證 framing 契約（泛化驗證用，只有 16000/512 有出貨模型）。
  Python 設定允許 `win_len < n_fft`，C 端不支援該模式。
- `rnnoise_compute_features()` 每次收一個 spectrum（`n_bins` 個 bin；`ac_stft_push_planar()` 內部保留前一個 hop，零歷史起算），
  累積 3 個 frame 後回傳 1，輸出 `out_erb[3][n_bands]` 與 `out_spec[3][2][spec_bins]`。
- `rnnoise_apply_gains()` 原地執行 `spec[k] *= bin_gains[k] * grid.stft_unscale`（gain 與 `sqrt(n_fft)` 合為一次乘法，沒有反 FFT）；
  共用 ISTFT 的 `ispec_scale` 必須維持 1，不可再縮放一次。
- 最新特徵 window `[t-2,t-1,t]` 的 gain 屬於 spectrum `t - grid.lookahead_frames`（window 槽位 `grid.conv_delay`，出貨值 1）；
  呼叫端須保存並延遲對應頻譜。單位增益時，輸出為輸入延遲 `1 + lookahead_frames` 個 hop（1 hop 的 OLA 加 lookahead）。
- ERB 矩陣由應用端提供：`RNNoiseConfig.erb_fwd` 與 `erb_inv` 是兩個 `const float *`，函式庫只借用、不讀檔、不複製，
  應用端須在使用期間保持有效。`erb_fwd` 是 `train.py` 的 mode 0、bin-major `[n_bins][n_bands]`；`erb_inv` 是 mode 1、
  band-major `[n_bands][n_bins]`，為 partition of unity（單位 band gain 展開後 bin gain 為 1）；兩者皆 `float32`，各
  `n_bins * n_bands * 4` bytes（16000/512/22 為 22,616）。`export_erb_matrix.py` 沒有輸出這個 band-major inverse，見「ERB 矩陣匯出」。
- `RNNOISE_PREPOST_LAYOUT_VERSION` 為 2、`RNNOISE_CARVE_VERSION` 為 1（皆進 pool hash）；GRU 張量的
  `RNNOISE_MODEL_IO_LAYOUT_VERSION` 與 ONNX state layout 為 **1**。

建置時加入 `audiolib/common/include`、`audiolib/common` include paths，連結同一組
`SIMD=0/1` 的 common archive；此目錄的 Makefile 已處理這些依賴。

更換 FFT 實作保留數學定義，但 butterfly 的浮點捨入會改變結果，不能宣稱逐位元
相同；低能量 band 的 log 會放大微小頻譜誤差。測試涵蓋 Python/C 容差、
頻譜修改、framing 契約（單位增益下輸出為輸入延遲一個 hop）、首 hop、重置與固定係數網路輸出；本機沒有已訓練 RNNoise 權重，
**尚未量測真實模型的感知品質**，固定係數測試不能替代這項評估。

在目前 AArch64 host，`rnnoise_get_mem_size()` 為 **5,600 bytes**（16000/512、22 bands）；pool 不含 `AcStft`／`AcIstft`、
ERB 矩陣與註冊的 GRU state。應用端另持有：framing 物件（預設各自帶 FFT plan 與窗時，`ac_stft_get_mem_size()` 與
`ac_istft_get_mem_size()` 在 512 點各為 21,840 bytes；共用 FFT handle 與窗表時的大小待量測）、兩個 ERB 矩陣共 45,232 bytes（各 22,616），
以及 float host 的 GRU state（3 × 128 floats = 1,536 bytes）。這些是目前實作的量測，整合端仍應依 size-query 配置。

## 環境安裝

```bash
# 建議使用 Python 3.9+
pip install -r requirements.txt
```

依賴套件：
- `torch` >= 1.13, < 2.9
- `torchaudio` >= 0.13, < 2.9（2.9 會忽略 `PCM_F`/32-bit WAV 參數）
- `soundfile` >= 0.12
- `numpy`
- `tqdm`

匯出 ONNX 額外需要（訓練不需要）：
```bash
pip install onnx onnxoptimizer onnxruntime
```

## 資料集準備 (DNS4)

本專案使用 [DNS Challenge 4](https://github.com/microsoft/DNS-Challenge) 資料集。下載後目錄結構如下：

```
datasets_fullband/
├── clean_fullband/              ← config.ini 的 speech_dir
│   ├── emotional_speech/
│   ├── french_speech/
│   ├── german_speech/
│   ├── italian_speech/
│   ├── read_speech/
│   ├── russian_speech/
│   ├── spanish_speech/
│   ├── vctk_wav48_silence_trimmed/
│   └── ...
├── noise_fullband/              ← config.ini 的 noise_dir
│   ├── audioset/
│   ├── freesound/
│   └── ...
└── impulse_responses/           ← config.ini 的 rir_dir (optional)
    ├── SLR26/
    ├── SLR28/
    └── ...
```

### 注意事項

- 所有音檔必須是 **`.wav` 格式**（程式只掃描 `*.wav`）
- 音檔可以是任意 sample rate，程式會自動 resample 到 16kHz
- **RIR 為 optional**：若不使用，可將 `rir_dir` 設為空字串或刪除該行。建議使用以提升 dereverberation 效果
- RIR 會自動用 Schroeder 積分法估算 RT60，只保留 `rt60_min` ~ `rt60_max` (`dataset_gen/config.example.ini` 預設 0.2s ~ 1.0s) 範圍內的 RIR
- **RIR RT60 快取**：首次掃描後會自動存成 `.rir_cache_*.json`，後續相同設定直接讀取，大幅加速初始化
- 不一定要用 DNS4，**任何符合上述結構的語音/噪音資料集都可以**（只要是 wav 檔放在對應目錄下）

## 設定 config.ini

訓練前需要修改 `config.ini` 中的資料路徑：

```ini
[paths]
speech_dir = /your/path/to/datasets_fullband/clean_fullband
noise_dir = /your/path/to/datasets_fullband/noise_fullband
rir_dir = /your/path/to/datasets_fullband/impulse_responses
output_dir = ./output
```

其他常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `[training] epochs` | 100 | 訓練 epoch 數 |
| `[training] batch_size` | 128 | Batch size |
| `[training] lr` | 3e-4 | Learning rate |
| `[training] device` | cuda | 訓練裝置 (`cuda` 或 `cpu`) |
| `[training] epoch_size` | 0 | 每 epoch sample 上限；0 表示使用全部資料 |
| `[feature] erb_norm_tau_sec` | 1.0 | ERB per-band mean EMA 時間常數 |
| `[feature] erb_norm_init_lo_db` | -20 | 最低 ERB band EMA 初值 |
| `[feature] erb_norm_init_hi_db` | -45 | 最高 ERB band EMA 初值 |
| `[feature] erb_norm_scale_db` | 40.0 | mean-subtracted ERB 縮放尺度 |
| `[feature] spec_max_hz` | 4000 | complex branch 頻率上限 |
| `[feature] spec_norm_tau_sec` | 1.0 | complex per-bin magnitude EMA 時間常數 |

改動任何 feature 常數都需要同步修改應用端的 `RNNoiseGridSpec` 並重新訓練。Checkpoint 會保存
`feature_version` 與所有 normalization 常數；train resume、denoise 和 ONNX export
都會拒絕缺少版本的舊 checkpoint，或拒絕與 runtime config 不一致的 checkpoint。
匯出的 ONNX model metadata 也會帶上相同 feature contract，供部署端檢查。

> 只有 `feature_version` 等於上述 `..._v8` 的 checkpoint/ONNX 可用；其他版本
> 即使 input shape 相同，normalization semantics 或 erb_fwd/erb_inv 矩陣也不同，
> 必須重訓後重新匯出。

## 訓練

目前訓練只接受共用 `../dataset_gen/pack_dataset.py` 產生的 raw
noisy/clean WAV tensor；STFT、log-ERB 與 runtime normalization 會在訓練時
即時計算。資料增強、指定 sample-rate 的 WAV generation、resample 與
packing 全部由 `../dataset_gen/` 維護，RNNoise-ERB 不保存重複版本。

訓練 objective 是直接 ERB IRM：對 22 個 band gain 與 ideal ratio
mask 在 `gamma=0.5` 的壓縮域做 MSE，對齊原始 RNNoise 論文採用的 perceptual
compromise。MRSL 的兩個 factor 為 0，沒有 VAD/activity 加權。低於
`energy_floor` 的 mixture band 無法定義 ratio，會被 mask 掉而不產生梯度。

```bash
# Step 0: 建立 dataset_gen 專用設定並填入 speech/noise/RIR paths
cp ../dataset_gen/config.example.ini ../dataset_gen/config.ini

# Step 1: 產生 RNNoise-ERB 專用的 16 kHz dataset
python3 ../dataset_gen/gen_dataset.py \
    --config ../dataset_gen/config.ini --output data_16k --hours 25 \
    --sample-rate 16000 --workers 4

# Step 2: 打包
python3 ../dataset_gen/pack_dataset.py \
    --input data_16k/pairs --output data_16k/packed \
    --shard-clips 512 --dtype float16

# Step 3: 訓練
python3 train.py --config config.ini --packed-dir data_16k/packed

# 可選：指定 GPU 並用 mmap 降低 RAM
python3 train.py --config config.ini \
    --packed-dir data_16k/packed --gpu 0 --mmap
```

`--mmap` 模式不會在 `__getitem__` 逐筆展開成 float32；FP16 batch 送到 GPU
後才轉 float32。訓練會隨機化資料區塊與區塊內順序，避免大型 shard 的全域
random access page fault。`config.ini` 的 `mmap_block_size`、
`mmap_num_workers`、`prefetch_factor` 可依磁碟速度與共用 RAM 調整；預設
256 / 2 / 2 是偏向低 RAM 的起點。

`../dataset_gen/gen_dataset.py` 常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hours` | 8.3 | 目標音檔總時數；最多只向上補一個完整 segment |
| `--output` | `data` | 輸出目錄 |
| `--sample-rate` | config | 此次 generation rate；RNNoise 使用 16000 |
| `--workers` | 4 | DataLoader workers；0 表示單程序 |
| `--resume` | false | 從 `pairs/` 最大編號後繼續寫入 |
| `--seed` | config | 負數為每次 OS-random；非負數可重現 |

### 從斷點續訓

訓練中途中斷後，可以從最後的 checkpoint 繼續：

```bash
python3 train.py --config config.ini --packed-data data_16k/packed.pt \
    --resume output/rnnoise_epoch5.pth
```

續訓會恢復：model weights、optimizer 狀態、global_step、epoch 計數、best_val_loss。
⚠ **scheduler 狀態刻意不恢復**：checkpoint 裡沒有存它，因為 restore 會把寫入當時
那一輪的 `T_max` 帶回來蓋掉為現在這輪重建的值，續訓到較長的 epochs 時 lr 無法降到
min_lr。做法是照新的 epochs 重建，再用 `global_step` 快轉，這也是
`training_common.fast_forward_scheduler()` 做的事。Checkpoint 同時檢查 loss
contract，不一致的 checkpoint 不能 resume。

### 訓練輸出

訓練結果儲存在 `output/` 目錄：

```
output/
├── rnnoise_epoch1.pth      # 每 epoch 的 checkpoint
├── rnnoise_epoch2.pth
├── ...
└── rnnoise_best.pth        # 最佳 validation loss 的模型
```

## 推論 (Denoise)

對單一音檔進行降噪：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
                   --input noisy.wav --output clean.wav
```

`--pf-beta` 預設為 0（關閉），避免在還沒驗證 raw gain 前再加劇抑制。

`--atten-lim <dB>` 限制最大抑制量，做法對齊 Rikorose/DeepFilterNet
`enhance.py` 的 `--atten-lim`/`-a`：把 gain 往 1.0（不抑制）方向線性混合，
`lim = 10^(-|atten_lim_db|/20)`，`gain' = lim + gain*(1-lim)`——例如
`--atten-lim 12` 保證輸出最多只壓 12dB，其餘 noise floor 會保留。這在 ERB
band 層級套用（`rnnoise_apply_atten_lim`，C 端同名函式對應),
數學上等價於直接對 spectrum 做同樣的線性混合(因為 mode=1 反向矩陣是
partition of unity),但成本只需 22 次而非 257 次。預設不啟用(`None`,
維持目前最大抑制行為);純推論後處理,不影響 checkpoint/feature contract。
若要定位「全壓成 0」是 feature、model 或 post-filter 造成：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
  --input failing.wav --output debug.wav --dump-debug debug/failing.npz
```

dump 含 `erb_db`、兩路 features、`raw_gains`、`post_gains`
與輸入/輸出 waveform。這是 debug 用；checkpoint version gate 則是部署防呆。

## ONNX 匯出

將訓練好的模型匯出為 ONNX 格式（用於部署）：

```bash
# 匯出
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx

# 匯出 + 驗證 (比較 PyTorch 與 ONNX 輸出)
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx --verify
```

Streaming ONNX 輸入為 `erb_input[1,3,22]`、`spec_input[1,3,2,129]`
與三組 GRU hidden state（`h1_in`／`h2_in`／`h3_in`，各 `[1,1,128]`）；輸出中心 frame 的 gains 與更新後 hidden state。
加速器本身不保存 state，三個 hidden 張量屬於應用端：板端把圖的 `h*_in` 輸入與 `h*_out` 輸出綁到同一個 buffer，CPU 不複製；
runtime 須先讀完所有 state 輸入、才可寫同位址的 state 輸出。三個 feature frames
是輸入 temporal kernel 的完整可視範圍，不代表一次輸出三個 frames。

float host 在 `RNNoiseConfig` 註冊 `h1`／`h2`／`h3`（指標加元素數，各 128；三組全給或全不給，指標配元素數 0 或
元素數配 NULL 指標都會被拒絕）：`rnnoise_init()` 與 `rnnoise_reset()` 歸零，每次 `rnnoise_frame_finish(st, band_gains)`
檢查 `band_gains` 與整塊註冊的 state 是否 finite；任一為 NaN/Inf 時，註冊的張量歸零、state generation 加一、回傳 `-1`
（呼叫端改用單位增益）。仍為 finite 的部分寫入或混合寫入無法偵測。板端不註冊任何 state：加速器自己的定點副本不會被
reset 或歸零動到，所以 `rnnoise_state_generation()` 在 init 後為 1，每次 `rnnoise_reset()` 與每次被拒的
`rnnoise_frame_finish()` 加一（modulo 2^32，註冊與否都計），runtime 看到與上一 frame 不同的值就把自己的 state 張量重填
quantized zero。匯出的 metadata 帶 `state_layout_version`，數值與 `process.h` 的
`RNNOISE_MODEL_IO_LAYOUT_VERSION` 相同，整合端可據此拒絕 state layout
已經對不上自己配置的圖。

PTQ calibration 可直接使用：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
  --input noisy.wav --output enhanced.wav \
  --dump-calib calib/rnnoise_erb --format bin --max-frames 8192
```

BIN 會依 ONNX input 分資料夾，每個 streaming frame 各寫一個檔案；例如
`h1_in/h1_in_0000.bin`。使用 `--format npz --dump-calib
calib/rnnoise_erb.npz` 可改輸出 NumPy archive。兩種格式都保存實際串流中
的三-frame feature window 與非零 GRU state，而不是重複 zero-state 樣本。

## ERB 矩陣匯出

匯出 ERB 轉換矩陣（C 部署用）：

```bash
# 匯出所有格式 (bin + npy + C header)
python3 export_erb_matrix.py --config config.ini --format all
```

`--format bin` 的 `erb_matrix.bin`（little-endian）依序是 `int32 n_bins`、`int32 n_bands`、`float32` forward `[n_bins][n_bands]`、
`float32` inverse `[n_bins][n_bands]`（forward 與 inverse 都是 bin-major），尾端接 `int32 nfftborder[n_bands]`；`--format npy`
的 `erb_fwd.npy`／`erb_inv.npy` 同為 `(n_bins, n_bands)`。函式庫的 `erb_inv` 是 band-major `[n_bands][n_bins]`（見上面
「C 串流整合」）：樹內沒有 exporter 輸出這個 layout，應用端須把 inverse 轉置後再交入。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | 模型定義與 packed raw-WAV tensor 訓練迴圈 |
| `inference.py` | 推論與 calibration 入口 |
| `export_onnx.py` | ONNX 匯出（streaming 推論格式） |
| `export_erb_matrix.py` | ERB 矩陣匯出（bin / npy / C header） |
| `config.ini` | 所有超參數設定 |
| `process.c` / `process.h` | C 前後處理實作（頻譜域，嵌入式部署用） |
| `tests/` | Python feature、loss、checkpoint 與 Python/C parity tests |
| `../dataset_gen/` | 共用 augmentation、WAV generation、resample、packing 與 packed loader |
| `requirements.txt` | Python 依賴套件 |

## ERB 矩陣 drift-guard

C 端沒有編譯期 ERB 表；矩陣是應用端提供的資料。`make test-tables` 把測試讀入的矩陣與獨立重算的三角 ERB filterbank 比對，
並檢查矩陣契約（例如 partition of unity）。測試讀的是兩個無標頭 float32 檔 `$(BUILD_DIR)/erb_fwd.bin`（bin-major）與
`$(BUILD_DIR)/erb_inv.bin`（band-major），由 `tests/erb_fixture.py`（只用標準函式庫）依 `train.py` 的公式寫出；以
`ERB_FWD=FILE ERB_INV=FILE` 可改讀其他檔。`make test-erb-python`（訓練環境）再把這個 fixture 釘到 `train.py` 與
`export_erb_matrix.py`。

```bash
make test-tables
```

重算用的常數（SR、N_FFT、N_BANDS、MIN_BINS_PER_BAND）刻意不讀 `process.h` 或 `config.ini`，兩者之一改動時這裡會失敗。

## Feature tests

```bash
make test                 # tables + feature contract + feature state + framing + SIMD/scalar digest
make test-feature-python  # 需 torch training environment
```

C test 會以獨立 recurrence 對照 `rnnoise_compute_features()`，並讓固定非平坦
spectrum 運行 4096 frames，確認 ERB mean norm 收斂為 0，complex preprocessing
仍非零。Python test 另驗證兩路 state/chunk equivalence、ERB 穩態收斂、complex
穩態可觀測性，以及純 ERB／可選 complex 兩種 model forward contract；
`tests/test_python_c_features.py` 以 golden vectors 對照 Python/C 的 STFT、
兩路特徵與最終 normalization state。即使出貨模型未使用 complex 分支，保留的
可選路徑仍須通過這些測試，避免日後重新啟用時出現 Python/C 漂移。
