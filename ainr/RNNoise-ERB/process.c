/* ============================================================
 * RNNoise-ERB 前後處理 C 實現 — 對齊 train.py / denoise.py
 *
 * 包含:
 *   - Radix-2 FFT/IFFT (N=N_FFT)
 *   - Root Hann window (sqrt(hann), analysis+synthesis 各乘一次 → COLA)
 *   - normalized STFT/ISTFT (× N^∓0.5, = torch.stft/istft normalized=True)
 *   - Glasberg-Moore ERB band borders + 三角 filterbank (forward/inverse)
 *   - erb_db = 10*log10(energy+1e-10) + shared broadband online mean/variance
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
 * Glasberg-Moore ERB band borders + 三角 filterbank + root-Hann window
 * 忠實移植 train.py erb_bandborder() / compute_erb_matrix()
 * (DeepFilterNet(-Keras) 常數: 24.7 * 9.265)
 *
 * F09 修正: 這四張表格 (rnn_nfftborder/rnn_erb_fwd/rnn_erb_inv/
 * rnn_hann_win) 原本是執行期 lazy-init 的全域可變狀態 (ensure_erb()/
 * ensure_window(), 見 git history)。外部審查指出即使 ready flag 用
 * __atomic acquire/release, 多個執行緒仍可能同時看到 ready==0、
 * 並行寫入同一組 non-atomic 陣列 — 依 C memory model 仍是 data race
 * (即使寫入的是相同常數值, 也是未定義行為)。
 *
 * 這些表格的輸入 (RNNOISE_SR/N_BANDS/N_BINS/N_FFT/WIN_LEN) 全部是編譯期
 * 常數, 因此改為編譯期 `static const` 表格: 由 gen_rnnoise_tables.c
 * (host-only 工具, 逐字複製原本 ensure_erb()/ensure_window() 的算式)
 * 產生 rnnoise_tables_gen.h 並直接 #include。從此表格在編譯期就固定,
 * 沒有共享可變狀態, 執行期不再有任何 race 可言, 也不需要 once-guard。
 * test_rnnoise_tables.c 用獨立複製的原始演算法在執行期重算並逐 byte
 * 比對這份表格, 當作 drift-guard。
 *
 * nfftborder: N_BANDS 個邊界 → N_BANDS-1 個 block, N_BANDS 個矩陣欄
 *   border[0]=0 (DC), border[N-1]=N_BINS (Nyquist+1)
 * forward (mode=0, 特徵用): 兩端單邊欄 ×2
 * inverse (mode=1, gain 展開用): 無 ×2, partition of unity
 *
 * root-Hann window (長度 WIN_LEN): sqrt(hann) — analysis 與 synthesis
 * 各乘一次，合計 = hann → COLA (torch.hann_window 預設 periodic=True,
 * 與此處分母 WIN_LEN 一致)
 * ============================================================ */
#include "rnnoise_tables_gen.h"

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

/* F09 修正後: ERB filterbank + root-Hann window 已是編譯期 `static const`
 * 表格 (rnnoise_tables_gen.h), 沒有執行期初始化可做。這個函式保留下來
 * 純粹是為了 API 相容 (呼叫端不必刪掉既有的 rnnoise_tables_init() 呼叫),
 * 本體是刻意留空的 no-op。見 process.h 對這個變更的說明。 */
void rnnoise_tables_init(void) {
    /* no-op: 表格在編譯期已就緒, 見上方 #include "rnnoise_tables_gen.h" */
}

void rnnoise_state_init(RNNoiseState *st) {
    memset(st, 0, sizeof(RNNoiseState));
    st->norm_mean = RNNOISE_NORM_MEAN_INIT_DB;
    st->norm_var = RNNOISE_NORM_VAR_INIT_DB2;
    rnnoise_tables_init();  /* no-op (表格編譯期已就緒); 保留呼叫只為相容 */
}

/* --- 前處理: analysis --- */

void rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im) {
    /* Root Hann windowed frame + zero-pad to N_FFT
     * (F13: 暫存區搬到 st->scratch_buf_re/im, 不佔用呼叫端 stack) */
    float *buf_re = st->scratch_buf_re;
    float *buf_im = st->scratch_buf_im;
    memset(buf_im, 0, sizeof(float) * RNNOISE_N_FFT);

    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        buf_re[i] = frame[i] * rnn_hann_win[i];
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
            sum += power[k] * rnn_erb_fwd[k][b];
        }
        erb_db[b] = 10.0f * log10f(sum + LOG_FLOOR);
    }

    /* log_erb_shared_online_cmvn_v1:
     *   - ONE scalar mean/variance is shared by all bands.
     *   - Features use the previous state (strictly causal).
     *   - State then observes the current frame's band-average log level.
     * Per-band temporal centering is intentionally forbidden because it erases
     * the stationary spectral envelope when ERB is the model's only input. */
    const float norm_a = (float)exp(
        -((double)RNNOISE_HOP_LEN / (double)RNNOISE_SR) /
        (double)RNNOISE_NORM_TAU_SEC);
    int idx = st->feat_idx;
    float denom = sqrtf(st->norm_var + RNNOISE_NORM_VAR_FLOOR_DB2);
    float level = 0.0f;
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        float feat = (erb_db[b] - st->norm_mean) / denom;
        if (feat > RNNOISE_NORM_CLIP) feat = RNNOISE_NORM_CLIP;
        if (feat < -RNNOISE_NORM_CLIP) feat = -RNNOISE_NORM_CLIP;
        st->feat_buf[idx][b] = feat;
        level += erb_db[b];
    }
    level /= RNNOISE_N_BANDS;

    {
        float delta = level - st->norm_mean;
        float new_mean = st->norm_mean + (1.0f - norm_a) * delta;
        float new_var = norm_a * st->norm_var +
            (1.0f - norm_a) * delta * (level - new_mean);
        st->norm_mean = new_mean;
        st->norm_var = new_var > 0.0f ? new_var : 0.0f;
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
    /* bin_gains = gains @ erb_inv^T (denoise.py apply gains; mode=1 為
     * partition of unity → gains=1 對應 bin_gains=1, 無需列正規化) */
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        float g = 0.0f;
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            g += band_gains[b] * rnn_erb_inv[k][b];
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
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        full_re[i] *= rnn_hann_win[i];
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
