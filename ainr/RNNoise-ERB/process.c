/* ============================================================
 * RNNoise-ERB 前後處理 C 實現 — 對齊 train.py / denoise.py
 *
 * 包含:
 *   - Radix-2 FFT/IFFT (N=N_FFT)
 *   - Root Hann window (sqrt(hann), analysis+synthesis 各乘一次 → COLA)
 *   - normalized STFT/ISTFT (× N^∓0.5, = torch.stft/istft normalized=True)
 *   - Glasberg-Moore ERB band borders + 三角 filterbank (forward/inverse)
 *   - erb_db = 10*log10(energy+1e-10) + band_mean_norm ((x-state)/40)
 *   - Band gain → bin gain (mode=1 inverse matrix, partition of unity)
 *   - Overlap-add synthesis
 * ============================================================ */

#include "process.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LOG_FLOOR 1e-10f

/* ============================================================
 * Glasberg-Moore ERB band borders + 三角 filterbank
 * 忠實移植 train.py erb_bandborder() / compute_erb_matrix()
 * (DeepFilterNet(-Keras) 常數: 24.7 * 9.265)
 *
 * nfftborder: N_BANDS 個邊界 → N_BANDS-1 個 block, N_BANDS 個矩陣欄
 *   border[0]=0 (DC), border[N-1]=N_BINS (Nyquist+1)
 * forward (mode=0, 特徵用): 兩端單邊欄 ×2
 * inverse (mode=1, gain 展開用): 無 ×2, partition of unity
 * ============================================================ */
static int   g_nfftborder[RNNOISE_N_BANDS];
static float g_erb_fwd[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static float g_erb_inv[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static int   g_erb_ready = 0;

static double freq2erb(double f) {
    return 9.265 * log(1.0 + f / (24.7 * 9.265));
}
static double erb2freq(double e) {
    return 24.7 * 9.265 * (exp(e / 9.265) - 1.0);
}

/* F09: once-guard 改為 __atomic acquire/release (GCC/Clang 內建)。
 * fast-path 用 acquire load 讀 ready flag: 若已就緒, acquire 語意保證
 * 「看得到 flag==1」的執行緒也一定看得到下面對 g_nfftborder/g_erb_fwd/
 * g_erb_inv 的完整寫入 (無 torn read)。多個執行緒可能「同時」通過
 * fast-path 檢查、各自重算一次表格 — 這是良性的 (冪等、寫入相同常數),
 * 只有最後的 release store 需要正確排序。建議呼叫端在啟用多執行緒之前
 * 先呼叫一次 rnnoise_tables_init() (見 process.h), 避免這種重複運算。 */
static void ensure_erb(void) {
    if (__atomic_load_n(&g_erb_ready, __ATOMIC_ACQUIRE)) return;

    const int    N  = RNNOISE_N_BANDS;
    const double sr = (double)RNNOISE_SR;
    const double high_lim = sr / 2.0;
    const double bw = high_lim / ((double)RNNOISE_N_FFT / 2.0);  /* = sr/n_fft */

    /* erb_bandborder: cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N))
     * border = round((cutoff + bw/2) / bw), 再套 Keras 的
     * 「每隔一個 band 至少跨 2 bin」修正, 端點釘在 DC / Nyquist+1。 */
    {
        double e_lo = freq2erb(0.0), e_hi = freq2erb(high_lim);
        double nb[RNNOISE_N_BANDS];
        for (int i = 0; i < N; i++) {
            double e = e_lo + (e_hi - e_lo) * i / (N - 1);   /* linspace 含端點 */
            double cutoff = erb2freq(e);
            nb[i] = floor((cutoff + bw / 2.0) / bw + 0.5);   /* np.round (半數進位) */
        }
        for (int i = 0; i < N - 2; i++) {
            if (nb[i + 2] - nb[i] < 2.0)
                nb[i + 2] += 2.0 - (nb[i + 2] - nb[i]);
        }
        nb[0] = 0.0;
        nb[N - 1] = (double)(RNNOISE_N_FFT / 2 + 1);
        for (int i = 0; i < N; i++) g_nfftborder[i] = (int)nb[i];
    }

    /* compute_erb_matrix: block i 介於 border[i]..border[i+1], 欄 i 放下降斜坡、
     * 欄 i+1 放上升斜坡; forward 版兩端單邊欄 ×2。 (double 計算, float 儲存 —
     * 對應 numpy 以 double 算、float32 存的行為) */
    memset(g_erb_fwd, 0, sizeof(g_erb_fwd));
    memset(g_erb_inv, 0, sizeof(g_erb_inv));
    for (int i = 0; i < N - 1; i++) {
        int lo = g_nfftborder[i];
        int hi = g_nfftborder[i + 1];
        int bs = hi - lo;
        for (int j = 0; j < bs; j++) {
            float down = (float)(1.0 - (double)j / (double)bs);
            float up   = (float)((double)j / (double)bs);
            g_erb_inv[lo + j][i]     = down;
            g_erb_inv[lo + j][i + 1] = up;
            g_erb_fwd[lo + j][i]     = down;
            g_erb_fwd[lo + j][i + 1] = up;
        }
    }
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        g_erb_fwd[k][0]     *= 2.0f;
        g_erb_fwd[k][N - 1] *= 2.0f;
    }

    /* release store: 保證上面所有寫入在其他執行緒看到 g_erb_ready==1
     * 之後才可見 (搭配 ensure_erb 開頭的 acquire load)。 */
    __atomic_store_n(&g_erb_ready, 1, __ATOMIC_RELEASE);
}

/* ============================================================
 * Root Hann window (長度 WIN_LEN, 前算)
 * sqrt(hann) — analysis 與 synthesis 各乘一次，合計 = hann → COLA
 * (torch.hann_window 預設 periodic=True, 與此處分母 WIN_LEN 一致)
 * ============================================================ */
static float g_hann_win[RNNOISE_WIN_LEN];
static int   g_win_ready = 0;

/* F09: 同 ensure_erb() 的 __atomic acquire/release once-guard。 */
static void ensure_window(void) {
    if (__atomic_load_n(&g_win_ready, __ATOMIC_ACQUIRE)) return;
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        g_hann_win[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / RNNOISE_WIN_LEN)));
    }
    __atomic_store_n(&g_win_ready, 1, __ATOMIC_RELEASE);
}

/* normalized=True 的縮放常數 */
#define STFT_NORM_FWD (1.0f / 22.62741699796952f)   /* N_FFT^-0.5, N=512 */
#define STFT_NORM_INV (22.62741699796952f)          /* N_FFT^+0.5 */

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

/* 全域查表 (ERB filterbank + root-Hann window) 的公開一次性初始化入口。
 * 見 process.h 對 F09 thread-safety 語意的說明。 */
void rnnoise_tables_init(void) {
    ensure_erb();
    ensure_window();
}

void rnnoise_state_init(RNNoiseState *st) {
    memset(st, 0, sizeof(RNNoiseState));
    /* band_mean_norm 初值: linspace(-60, -90, N_BANDS) dB
     * (train.py/denoise.py mean_norm_init=(-60.0, -90.0)) */
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        st->ema_state[b] = RNNOISE_MEAN_NORM_LO +
            (RNNOISE_MEAN_NORM_HI - RNNOISE_MEAN_NORM_LO) * (float)b / (RNNOISE_N_BANDS - 1);
    }
    rnnoise_tables_init();
}

/* --- 前處理: analysis --- */

void rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im) {
    ensure_window();

    /* Root Hann windowed frame + zero-pad to N_FFT
     * (F13: 暫存區搬到 st->scratch_buf_re/im, 不佔用呼叫端 stack) */
    float *buf_re = st->scratch_buf_re;
    float *buf_im = st->scratch_buf_im;
    memset(buf_im, 0, sizeof(float) * RNNOISE_N_FFT);

    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        buf_re[i] = frame[i] * g_hann_win[i];
    }
    /* zero-pad (當 WIN_LEN == N_FFT 時此 loop 不執行) */
    for (int i = RNNOISE_WIN_LEN; i < RNNOISE_N_FFT; i++) {
        buf_re[i] = 0.0f;
    }

    fft_radix2(buf_re, buf_im, RNNOISE_N_FFT, 0);

    /* 只取正頻率 (0 ~ N/2) 共 N_BINS 個, 並乘 N^-0.5
     * (= torch.stft normalized=True; 特徵的絕對 dB 尺度依賴此縮放) */
    for (int i = 0; i < RNNOISE_N_BINS; i++) {
        out_re[i] = buf_re[i] * STFT_NORM_FWD;
        out_im[i] = buf_im[i] * STFT_NORM_FWD;
    }
}

/* --- 前處理: compute features --- */

int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_features[3][RNNOISE_N_BANDS]) {
    ensure_erb();

    /* power spectrum (normalized 域)
     * (F13: 暫存區搬到 st->scratch_power, 不佔用呼叫端 stack) */
    float *power = st->scratch_power;
    for (int i = 0; i < RNNOISE_N_BINS; i++) {
        power[i] = spec_re[i] * spec_re[i] + spec_im[i] * spec_im[i];
    }

    /* 三角 ERB band energy → dB
     * energy = power @ erb_fwd; erb_db = 10*log10(energy + 1e-10)
     * (F13: 暫存區搬到 st->scratch_erb_db) */
    float *erb_db = st->scratch_erb_db;
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        float sum = 0.0f;
        for (int k = 0; k < RNNOISE_N_BINS; k++) {
            sum += power[k] * g_erb_fwd[k][b];
        }
        erb_db[b] = 10.0f * log10f(sum + LOG_FLOOR);
    }

    /* band_mean_norm (train.py extract_erb_features / denoise.py):
     *   state = x*(1-a) + state*a;  feat = (x - state) / 40
     *   a = exp(-(hop/sr)/tau), tau = 1s (make_ema_alpha) */
    const float ema_a = (float)exp(-((double)RNNOISE_HOP_LEN / (double)RNNOISE_SR) / 1.0);
    int idx = st->feat_idx;
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        st->ema_state[b] = erb_db[b] * (1.0f - ema_a) + st->ema_state[b] * ema_a;
        st->feat_buf[idx][b] = (erb_db[b] - st->ema_state[b]) / RNNOISE_MEAN_NORM_DIV;
    }
    st->feat_idx = (idx + 1) % 3;
    /* 飽和計數，不無限累加: feat_count 唯一用途是與 3 比較 (< 3 vs >= 3)，
     * 無限遞增在長時間連續運行後會 overflow (signed int UB)，飽和在 3
     * 對這個用途完全等價且消除該風險 */
    if (st->feat_count < 3) ++st->feat_count;

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
    ensure_erb();
    /* bin_gains = gains @ erb_inv^T (denoise.py apply gains; mode=1 為
     * partition of unity → gains=1 對應 bin_gains=1, 無需列正規化) */
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        float g = 0.0f;
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            g += band_gains[b] * g_erb_inv[k][b];
        }
        bin_gains[k] = g;
    }
}

/* --- 後處理: synthesis (apply gain + IFFT + overlap-add) --- */

void rnnoise_synthesis(RNNoiseState *st,
                       float *spec_re, float *spec_im,
                       const float *bin_gains,
                       float *out_samples) {
    /* 套用 gain, 並乘 N^+0.5 反正規化 (= torch.istft normalized=True) */
    for (int i = 0; i < RNNOISE_N_BINS; i++) {
        spec_re[i] *= bin_gains[i] * STFT_NORM_INV;
        spec_im[i] *= bin_gains[i] * STFT_NORM_INV;
    }

    /* 還原負頻率 (共軛對稱)
     * (F13: 暫存區搬到 st->scratch_full_re/im, 不佔用呼叫端 stack) */
    float *full_re = st->scratch_full_re;
    float *full_im = st->scratch_full_im;
    memcpy(full_re, spec_re, sizeof(float) * RNNOISE_N_BINS);
    memcpy(full_im, spec_im, sizeof(float) * RNNOISE_N_BINS);
    for (int i = 1; i < RNNOISE_N_FFT / 2; i++) {
        full_re[RNNOISE_N_FFT - i] =  spec_re[i];
        full_im[RNNOISE_N_FFT - i] = -spec_im[i];
    }

    /* IFFT */
    fft_radix2(full_re, full_im, RNNOISE_N_FFT, 1);

    /* Root Hann window (synthesis side) — 只取 WIN_LEN 點，丟棄 zero-pad 部分
     * (sqrt-hann × sqrt-hann = hann, 50% overlap COLA 和 = 1 → 免除 torch.istft
     * 的 window-envelope 除法, 穩態等價) */
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
