/* ============================================================
 * RNNoise 前後處理 C 實現
 *
 * 包含:
 *   - Radix-2 FFT/IFFT (N=N_FFT)
 *   - Root Hann window (sqrt(hann), analysis+synthesis 各乘一次 → COLA)
 *   - ERB band energy + log + running normalization
 *   - Band gain → bin gain 展開
 *   - Overlap-add synthesis
 * ============================================================ */

#include "process.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG_FLOOR 1e-10f
#define NORM_ALPHA 0.99f   /* running mean/var 平滑係數 */

/* ============================================================
 * ERB band bin edges (預算好，對應 Python compute_erb_bands)
 *
 * N_BANDS bands, N_BANDS+1 edges
 * ERB_rate(f) = 21.4 * log10(0.00437*f + 1)
 * 均勻分佈在 ERB-rate scale
 * ============================================================ */
/* 動態計算的 bin edges (init 時填入) */
static int g_bin_edges[RNNOISE_N_BANDS + 1];
static int g_edges_ready = 0;

static void ensure_edges(void) {
    if (g_edges_ready) return;
    const int n_bins = RNNOISE_N_FFT / 2 + 1;  /* N_BINS */
    const float sr = (float)RNNOISE_SR;
    const float n_fft = (float)RNNOISE_N_FFT;

    float e_low  = 21.4f * log10f(0.00437f * 0.0f + 1.0f);
    float e_high = 21.4f * log10f(0.00437f * (sr / 2.0f) + 1.0f);

    for (int i = 0; i <= RNNOISE_N_BANDS; i++) {
        float e = e_low + (e_high - e_low) * i / RNNOISE_N_BANDS;
        float freq = (powf(10.0f, e / 21.4f) - 1.0f) / 0.00437f;
        int bin = (int)(freq / (sr / n_fft) + 0.5f);
        if (bin < 0) bin = 0;
        if (bin >= n_bins) bin = n_bins;
        g_bin_edges[i] = bin;
    }
    /* 確保單調遞增 */
    for (int i = 1; i <= RNNOISE_N_BANDS; i++) {
        if (g_bin_edges[i] <= g_bin_edges[i - 1])
            g_bin_edges[i] = g_bin_edges[i - 1] + 1;
    }
    if (g_bin_edges[RNNOISE_N_BANDS] > n_bins)
        g_bin_edges[RNNOISE_N_BANDS] = n_bins;

    g_edges_ready = 1;
}

/* ============================================================
 * Root Hann window (長度 WIN_LEN, 前算)
 * sqrt(hann) — analysis 與 synthesis 各乘一次，合計 = hann → COLA
 * 當 WIN_LEN < N_FFT 時，analysis 會 zero-pad 到 N_FFT
 * ============================================================ */
static float g_hann_win[RNNOISE_WIN_LEN];
static int   g_win_ready = 0;

static void ensure_window(void) {
    if (g_win_ready) return;
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        g_hann_win[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / RNNOISE_WIN_LEN)));
    }
    g_win_ready = 1;
}

/* ============================================================
 * Radix-2 FFT/IFFT (in-place, N=N_FFT)
 * re[], im[] 長度皆為 N
 * ============================================================ */
static void fft_radix2(float *re, float *im, int n, int inverse) {
    /* bit-reversal */
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int m = n >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }
    /* butterfly */
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.0f * (float)M_PI / len;
        float wre = cosf(ang);
        float wim = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int u = i + k;
                int v = i + k + len / 2;
                float tre = re[v] * cur_re - im[v] * cur_im;
                float tim = re[v] * cur_im + im[v] * cur_re;
                re[v] = re[u] - tre;
                im[v] = im[u] - tim;
                re[u] += tre;
                im[u] += tim;
                float new_re = cur_re * wre - cur_im * wim;
                float new_im = cur_re * wim + cur_im * wre;
                cur_re = new_re;
                cur_im = new_im;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

/* ============================================================
 * 公開 API
 * ============================================================ */

void rnnoise_state_init(RNNoiseState *st) {
    memset(st, 0, sizeof(RNNoiseState));
    /* running var 初始化為 1 避免除零 */
    for (int i = 0; i < RNNOISE_N_BANDS; i++) {
        st->run_var[i] = 1.0f;
    }
    ensure_edges();
    ensure_window();
}

/* --- 前處理: analysis --- */

void rnnoise_analysis(const float *frame, float *out_re, float *out_im) {
    ensure_window();

    /* Root Hann windowed frame + zero-pad to N_FFT */
    float buf_re[RNNOISE_N_FFT];
    float buf_im[RNNOISE_N_FFT];
    memset(buf_im, 0, sizeof(buf_im));

    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        buf_re[i] = frame[i] * g_hann_win[i];
    }
    /* zero-pad (當 WIN_LEN == N_FFT 時此 loop 不執行) */
    for (int i = RNNOISE_WIN_LEN; i < RNNOISE_N_FFT; i++) {
        buf_re[i] = 0.0f;
    }

    fft_radix2(buf_re, buf_im, RNNOISE_N_FFT, 0);

    /* 只取正頻率 (0 ~ N/2), 共 N_BINS 個 */
    memcpy(out_re, buf_re, sizeof(float) * RNNOISE_N_BINS);
    memcpy(out_im, buf_im, sizeof(float) * RNNOISE_N_BINS);
}

/* --- 前處理: compute features --- */

int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_features[3][RNNOISE_N_BANDS]) {
    ensure_edges();

    /* 計算 power spectrum */
    float power[RNNOISE_N_BINS];
    for (int i = 0; i < RNNOISE_N_BINS; i++) {
        power[i] = spec_re[i] * spec_re[i] + spec_im[i] * spec_im[i];
    }

    /* ERB band energy → log */
    float log_energy[RNNOISE_N_BANDS];
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        int lo = g_bin_edges[b];
        int hi = g_bin_edges[b + 1];
        float sum = 0.0f;
        for (int k = lo; k < hi; k++) sum += power[k];
        log_energy[b] = logf(sum + LOG_FLOOR);
    }

    /* 更新 running statistics */
    st->norm_count++;
    if (st->norm_count == 1) {
        /* 第一個 frame: 直接設定 */
        memcpy(st->run_mean, log_energy, sizeof(float) * RNNOISE_N_BANDS);
        for (int b = 0; b < RNNOISE_N_BANDS; b++) st->run_var[b] = 1.0f;
    } else {
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            st->run_mean[b] = NORM_ALPHA * st->run_mean[b] + (1.0f - NORM_ALPHA) * log_energy[b];
            float diff = log_energy[b] - st->run_mean[b];
            st->run_var[b]  = NORM_ALPHA * st->run_var[b]  + (1.0f - NORM_ALPHA) * diff * diff;
        }
    }

    /* 正規化並存入 ring buffer */
    int idx = st->feat_idx;
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        float std_val = sqrtf(st->run_var[b]) + 1e-8f;
        st->feat_buf[idx][b] = (log_energy[b] - st->run_mean[b]) / std_val;
    }
    st->feat_idx = (idx + 1) % 3;
    st->feat_count++;

    /* 需要至少 3 frame 才能送入 conv1 */
    if (st->feat_count < 3) return 0;

    /* 按時序排列 3 frame: oldest, middle, newest */
    int oldest = st->feat_idx;  /* feat_idx 指向下一個寫入位置 = 最舊的那格 */
    for (int f = 0; f < 3; f++) {
        int src = (oldest + f) % 3;
        memcpy(out_features[f], st->feat_buf[src], sizeof(float) * RNNOISE_N_BANDS);
    }
    return 1;
}

/* --- 後處理: expand gains --- */

void rnnoise_expand_gains(const float *band_gains, float *bin_gains) {
    ensure_edges();
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        int lo = g_bin_edges[b];
        int hi = g_bin_edges[b + 1];
        for (int k = lo; k < hi; k++) {
            bin_gains[k] = band_gains[b];
        }
    }
}

/* --- 後處理: synthesis (apply gain + IFFT + overlap-add) --- */

void rnnoise_synthesis(RNNoiseState *st,
                       float *spec_re, float *spec_im,
                       const float *bin_gains,
                       float *out_samples) {
    /* 套用 gain */
    for (int i = 0; i < RNNOISE_N_BINS; i++) {
        spec_re[i] *= bin_gains[i];
        spec_im[i] *= bin_gains[i];
    }

    /* 還原負頻率 (共軛對稱) */
    float full_re[RNNOISE_N_FFT];
    float full_im[RNNOISE_N_FFT];
    memcpy(full_re, spec_re, sizeof(float) * RNNOISE_N_BINS);
    memcpy(full_im, spec_im, sizeof(float) * RNNOISE_N_BINS);
    for (int i = 1; i < RNNOISE_N_FFT / 2; i++) {
        full_re[RNNOISE_N_FFT - i] =  spec_re[i];
        full_im[RNNOISE_N_FFT - i] = -spec_im[i];
    }

    /* IFFT */
    fft_radix2(full_re, full_im, RNNOISE_N_FFT, 1);

    /* Root Hann window (synthesis side) — 只取 WIN_LEN 點，丟棄 zero-pad 部分 */
    ensure_window();
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        full_re[i] *= g_hann_win[i];
    }
    /* full_re[WIN_LEN..N_FFT-1] 為 zero-pad 產生的殘留，不使用 */

    /* Overlap-add: 輸出 HOP_LEN 個 sample */
    for (int i = 0; i < RNNOISE_HOP_LEN; i++) {
        out_samples[i] = st->synthesis_buf[i] + full_re[i];
    }

    /* 更新 synthesis_buf: 存 overlap 部分 (OVL_LEN = WIN_LEN - HOP_LEN) */
    for (int i = 0; i < RNNOISE_OVL_LEN; i++) {
        st->synthesis_buf[i] = full_re[i + RNNOISE_HOP_LEN];
    }
    /* 清除剩餘 */
    for (int i = RNNOISE_OVL_LEN; i < RNNOISE_WIN_LEN; i++) {
        st->synthesis_buf[i] = 0.0f;
    }
}
