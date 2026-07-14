#ifndef RNNOISE_PROCESS_H
#define RNNOISE_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * RNNoise-ERB 前後處理 (C 實現) — 對齊 train.py / denoise.py
 *
 * 與 Python 參考的對應 (denoise.py enhance() 為準):
 *   - STFT: root-Hann window, normalized=True (× N_FFT^-0.5)
 *   - ERB: Glasberg-Moore band borders (erb_bandborder) +
 *     三角 filterbank (compute_erb_matrix mode=0 forward / mode=1 inverse)
 *   - 特徵: erb_db = 10*log10(power @ ERB + 1e-10);
 *     band_mean_norm: state = x*(1-a) + state*a; feat = (x-state)/40,
 *     state 初值 = linspace(-60, -90) dB, a = exp(-(hop/sr)/1.0)
 *   - Band gain → bin gain: gains @ ERB_inv^T (mode=1, partition of unity)
 *   - ISTFT: normalized=True (× N_FFT^+0.5) + root-Hann + 50% OLA (COLA)
 *
 * 注意: torch.stft(center=True) 的 reflect padding 是離線批次行為;
 * 本 streaming 實作等價於 center=False 的框對齊 (差 N_FFT/2 的時間偏移,
 * 僅影響首尾邊界 frame)。特徵/增益的數學與 Python 逐式對齊, 但 matmul
 * 的累加順序依實作而異 → 與 torch 為 float32 ULP 級近似, 非 bit-exact。
 * ============================================================ */

#define RNNOISE_SR          16000
#define RNNOISE_N_FFT       512
#define RNNOISE_N_BINS      257   /* N_FFT/2 + 1 */
#define RNNOISE_WIN_LEN     512   /* 分析窗長度 (≤ N_FFT, 預設 = N_FFT) */
#define RNNOISE_HOP_LEN     256   /* 幀移長度 (≤ WIN_LEN/2 for COLA) */
#define RNNOISE_OVL_LEN     (RNNOISE_WIN_LEN - RNNOISE_HOP_LEN)  /* overlap 長度 */
#define RNNOISE_N_BANDS     22    /* = config.ini [signal] n_bands (純 ERB 模式) */
#define RNNOISE_CONV_DELAY  2     /* conv1 kernel=3 causal → 需要緩衝 2 frame 歷史, 0 lookahead */

/* band_mean_norm 常數 (train.py extract_erb_features / denoise.py extract_features) */
#define RNNOISE_MEAN_NORM_LO   (-60.0f)  /* linspace 初值: 低頻端 */
#define RNNOISE_MEAN_NORM_HI   (-90.0f)  /* linspace 初值: 高頻端 */
#define RNNOISE_MEAN_NORM_DIV  40.0f     /* feat = (x - state) / 40 */

/* 處理狀態 (呼叫端分配，跨 frame 保持) */
typedef struct {
    /* overlap-add 緩衝 (長度 = WIN_LEN, 只用前 OVL_LEN) */
    float synthesis_buf[RNNOISE_WIN_LEN];

    /* 特徵歷史 (conv1 需要 3 frame) */
    float feat_buf[3][RNNOISE_N_BANDS];  /* ring buffer for 3 frames */
    int   feat_idx;                      /* 下一個寫入位置 (0,1,2) */
    int   feat_count;                    /* 已累積的 frame 數 */

    /* band_mean_norm running mean (dB 域); 初值 linspace(-60,-90) */
    float ema_state[RNNOISE_N_BANDS];

    /* --- 以下為每次呼叫用的暫存區 (scratch), 非跨 frame 狀態 ---
     * 原本配置在各函式的 stack 上 (F13: embedded hot-path stack 偏高);
     * 搬到 state 內以降低單次呼叫的 stack 佔用。呼叫間不需要清零,
     * 每次使用前都會被完整覆寫。 */
    float scratch_buf_re[RNNOISE_N_FFT];   /* rnnoise_analysis 用 */
    float scratch_buf_im[RNNOISE_N_FFT];
    float scratch_power[RNNOISE_N_BINS];   /* rnnoise_compute_features 用 */
    float scratch_erb_db[RNNOISE_N_BANDS];
    float scratch_full_re[RNNOISE_N_FFT];  /* rnnoise_synthesis 用 */
    float scratch_full_im[RNNOISE_N_FFT];
} RNNoiseState;

/* 初始化狀態 (歸零 + ema_state 設 linspace 初值; 內部亦會呼叫
 * rnnoise_tables_init() 確保 ERB/window 表已就緒) */
void rnnoise_state_init(RNNoiseState *st);

/* 全域查表 (ERB filterbank + root-Hann window) 的一次性初始化。
 *
 * 建議在啟用多執行緒之前、程式啟動階段就呼叫一次 (單執行緒環境下呼叫即可
 * 保證後續所有 rnnoise_* API 不會再觸發運算)。若省略呼叫, 各熱路徑 API
 * 內部仍有 fast-path once-guard 會 lazy 觸發計算 —
 * 該 guard 使用 __atomic acquire/release: 多個執行緒「同時」第一次呼叫時,
 * 可能各自重複計算一次表格 (冪等、常數相同, 無害), 但 flag 的
 * release-store / acquire-load 保證任何看到 flag=1 的執行緒都能看到
 * 完整寫入的表格內容 (無 torn read)。 */
void rnnoise_tables_init(void);

/* --- 前處理 (每 frame 呼叫) --- */

/* 對 WIN_LEN 個 sample 做 analysis:
 *   1. 加 root Hann window (WIN_LEN samples)
 *   2. Zero-pad 到 N_FFT (當 WIN_LEN < N_FFT)
 *   3. FFT → N_BINS 個 complex bin, 乘 N_FFT^-0.5 (= torch normalized=True)
 *   st: 提供 scratch 空間 (scratch_buf_re/im), 不讀寫任何跨 frame 狀態
 *   frame: 長度 WIN_LEN, out_re/out_im: 長度 N_BINS */
void rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im);

/* 從 normalized FFT spectrum 計算 ERB band features:
 *   1. power = |X|^2 → 三角 ERB filterbank (mode=0) → 10*log10(·+1e-10)
 *   2. band_mean_norm: state=x*(1-a)+state*a; feat=(x-state)/40
 *   spec_re, spec_im: 長度 N_BINS
 *   out_features: [3][N_BANDS] (oldest→newest, 供 conv1 k=3)
 *   回傳: 1 = 特徵可用 (已累積 3 frame), 0 = 尚需累積 */
int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_features[3][RNNOISE_N_BANDS]);

/* --- 後處理 --- */

/* 將 N_BANDS 個 band gain 經 mode=1 反向三角矩陣展開到 N_BINS 個 bin gain
 * (partition of unity: gains=1 → bin_gains=1, 無需列正規化) */
void rnnoise_expand_gains(const float *band_gains, float *bin_gains);

/* 套用 gain 並 ISTFT + overlap-add:
 *   spec_re, spec_im: 長度 N_BINS (會被修改; 內含 × N_FFT^+0.5 反正規化)
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
