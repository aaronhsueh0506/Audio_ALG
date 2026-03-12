#ifndef RNNOISE_PROCESS_H
#define RNNOISE_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * RNNoise 前後處理 (C 實現)
 * - FFT/IFFT (radix-2, N=512)
 * - Root Hann window + overlap-add (50% overlap, COLA)
 * - ERB band energy 計算
 * - 特徵正規化 (running statistics)
 * - Band gain → FFT bin gain 展開
 * ============================================================ */

#define RNNOISE_SR          16000
#define RNNOISE_N_FFT       512
#define RNNOISE_N_BINS      257   /* N_FFT/2 + 1 */
#define RNNOISE_WIN_LEN     512   /* = N_FFT (32ms) */
#define RNNOISE_HOP_LEN     256   /* = N_FFT/2 (16ms) */
#define RNNOISE_N_BANDS     22
#define RNNOISE_CONV_DELAY  2     /* conv1 kernel=3 valid → 需要緩衝 2 frame 歷史 */

/* 處理狀態 (呼叫端分配，跨 frame 保持) */
typedef struct {
    /* overlap-add 緩衝 */
    float synthesis_buf[RNNOISE_N_FFT];  /* ISTFT overlap buffer */

    /* 特徵歷史 (conv1 需要 3 frame) */
    float feat_buf[3][RNNOISE_N_BANDS];  /* ring buffer for 3 frames */
    int   feat_idx;                      /* 下一個寫入位置 (0,1,2) */
    int   feat_count;                    /* 已累積的 frame 數 */

    /* running normalization */
    float run_mean[RNNOISE_N_BANDS];
    float run_var[RNNOISE_N_BANDS];
    int   norm_count;
} RNNoiseState;

/* 初始化狀態 (全部歸零) */
void rnnoise_state_init(RNNoiseState *st);

/* --- 前處理 (每 frame 呼叫) --- */

/* 對 HOP_LEN (256) 個 sample 做 analysis:
 *   1. 加 root Hann window (WIN_LEN = N_FFT, 無需 zero-pad)
 *   2. FFT → 得到 N_BINS 個 complex bin
 *   out_re, out_im: 長度 N_BINS */
void rnnoise_analysis(const float *frame, float *out_re, float *out_im);

/* 從 FFT power spectrum 計算 ERB band features:
 *   1. 加總每個 band 的 power → log
 *   2. running normalization
 *   spec_re, spec_im: 長度 N_BINS
 *   out_features: 長度 N_BANDS
 *   回傳: 1 = 特徵可用 (已累積 3 frame), 0 = 尚需累積 */
int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_features[3][RNNOISE_N_BANDS]);

/* --- 後處理 --- */

/* 將 N_BANDS 個 band gain 展開到 N_BINS 個 FFT bin gain */
void rnnoise_expand_gains(const float *band_gains, float *bin_gains);

/* 套用 gain 並 ISTFT + overlap-add:
 *   spec_re, spec_im: 長度 N_BINS (會被修改)
 *   bin_gains: 長度 N_BINS
 *   out_samples: 長度 HOP_LEN (256), 輸出的時域 sample */
void rnnoise_synthesis(RNNoiseState *st,
                       float *spec_re, float *spec_im,
                       const float *bin_gains,
                       float *out_samples);

#ifdef __cplusplus
}
#endif

#endif /* RNNOISE_PROCESS_H */
