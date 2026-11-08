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
            if (run_rc != 0) {
                /* run_model() itself reported failure, but may already have overwritten
                 * band_gains and/or h1/h2/h3 with finite-looking garbage before discovering that:
                 * rnnoise_frame_finish() only catches NON-FINITE values, so a finite garbage
                 * write would otherwise be silently ACCEPTED (finish_rc 0, generation not
                 * advanced) and carry that garbage into the next hop undetected -- the model's
                 * own error code is the only signal that caught it. Poison all four with NaN
                 * here so the already-correct finite-check mechanism rejects this hop's state
                 * exactly as if run_model() had written nothing. */
                for (int b = 0; b < grid.n_bands; ++b) band_gains[b] = NAN;
                for (int k = 0; k < 3 * GRU_HIDDEN; ++k) h[k] = NAN;
            }
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

## 訓練與部署

- [環境、資料集、訓練與 WAV 推論](reference/training.md)
- [ONNX 匯出、ERB 矩陣與校驗](reference/export.md)
