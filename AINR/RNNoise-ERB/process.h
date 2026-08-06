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
 *   - ERB 特徵: log energy 的每-band causal EMA mean norm。
 *   - Complex 特徵: 0..4 kHz real/imag，用每-bin magnitude EMA unit norm。
 *   - Band gain → bin gain: gains @ ERB_inv^T (mode=1, partition of unity)
 *   - ISTFT: normalized=True (× N_FFT^+0.5) + root-Hann + 50% OLA (COLA)
 *
 * 框對齊: train.py 已改用 center=False, 與這個 streaming 實作一致。
 * (先前訓練用 center=True, 兩邊差 N_FFT/2 = 256 sample 的框偏移。舊註解稱
 *  該偏移「僅影響首尾邊界 frame」—— 對頻譜成立, 但對 causal EMA 正規化器
 *  不成立: 它的狀態從 frame 0 起累積, 而 3 秒 segment 全程都在暖機, 所以
 *  frame 0 的差異會一路傳播到整段。)
 * 合成端同樣不做 window-envelope 除法: root-Hann 分析 × root-Hann 合成
 * = Hann, 50% overlap 下 COLA 自動成立。train.py 的 istft() 亦然。
 * 特徵/增益的數學與 Python 逐式對齊, 但 matmul 的累加順序依實作而異
 * → 與 torch 為 float32 ULP 級近似, 非 bit-exact。
 * ============================================================ */

#define RNNOISE_SR          16000
#define RNNOISE_N_FFT       512
#define RNNOISE_N_BINS      257   /* N_FFT/2 + 1 */
#define RNNOISE_WIN_LEN     512   /* 分析窗長度 (≤ N_FFT, 預設 = N_FFT) */
#define RNNOISE_HOP_LEN     256   /* 幀移長度 (≤ WIN_LEN/2 for COLA) */
#define RNNOISE_OVL_LEN     (RNNOISE_WIN_LEN - RNNOISE_HOP_LEN)  /* overlap 長度 */
#define RNNOISE_N_BANDS     22    /* = config.ini [signal] n_bands (純 ERB 模式) */
#define RNNOISE_MIN_BINS_PER_BAND 2  /* = config.ini [signal] min_bins_per_band */
#define RNNOISE_LOOKAHEAD     1     /* = config.ini lookahead_frames */
#define RNNOISE_CONV_DELAY    (2 - RNNOISE_LOOKAHEAD)

/* log_erb_dfn_mean_cplx_unit_0_4k_v8 constants.  Keep byte-for-byte aligned with
 * config.ini [feature] and checkpoint validation in train.py.
 * v4 removes the erb_norm_clip/spec_clip deployment safety clamp v3 kept on
 * top of the DeepFilterNet formula -- verified against upstream
 * Rikorose/DeepFilterNet libDF/src/lib.rs (band_mean_norm_erb/band_unit_norm)
 * and this repo's own AINR/DeepFilterNet2 port that neither clips.
 * v5 fixes the ERB band-border minimum-width enforcement in
 * gen_rnnoise_tables.c (see that file for the algorithm; train.py's
 * erb_bandborder() is the Python side of the same fix) -- changes the
 * erb_fwd/erb_inv tables below.
 * v8 replaces libDF's imported normaliser init with values measured on this
 *     project's own 16 kHz corpus.  -60/-90 dB and 0.001/0.0001 are calibrated
 *     for libDF's rectangular, energy-MEAN filterbank; this port sums over a
 *     triangular overlapping bank, so those constants started the EMA
 *     +35..+48 dB away from where the features actually sit.  The ERB pair is
 *     rounded (fit gave -24.5/-41.9, costing 0.67 dB RMS) because it is in dB;
 *     the spec pair is NOT rounded past one significant figure because it is a
 *     linear magnitude the model divides by as x/sqrt(state), where 0.001 vs
 *     the measured 0.008161 would put features 2.86x out at the top bin.
 * v7 reverts to deriving the decay from tau (1 s), matching upstream libDF's
 *     _calculate_norm_alpha(sr, hop, tau): 16k/hop256 -> 0.984.  Pinning alpha
 *     at 0.99 gave a 1.59 s memory, so a 3 s training segment ended with ~15%
 *     of the init value still present -- every frame the model saw was in the
 *     init-dominated transient.
 * v6 pinned the normaliser decay directly (RNNOISE_*_NORM_ALPHA = 0.99) instead
 * of deriving it from tau.  alpha is the per-FRAME decay, so pinning it keeps
 * the normaliser's memory at 1/(1-alpha) = 100 frames under any sr/hop, which
 * is the invariant the GRU's learned time constants actually depend on;
 * deriving from tau pins it in seconds and lets the frame count drift.
 * ⚠ The longer decay stays harmless only while the INIT values match the
 * steady-state distribution -- see calibrate_norm_init.py.  Changing alpha
 * without recalibrating the init reintroduces a warm-up transient that now
 * spans 1.6x the training segment. */
#define RNNOISE_FEATURE_VERSION       "log_erb_dfn_mean_cplx_unit_0_4k_v8"
#define RNNOISE_ERB_NORM_TAU_SEC          1.0f
#define RNNOISE_ERB_NORM_ALPHA            0.984f
#define RNNOISE_ERB_NORM_INIT_LO_DB     (-20.0f)
#define RNNOISE_ERB_NORM_INIT_HI_DB     (-45.0f)
#define RNNOISE_ERB_NORM_SCALE_DB        40.0f
#define RNNOISE_SPEC_MAX_HZ            4000
#define RNNOISE_SPEC_BINS               129
#define RNNOISE_SPEC_NORM_TAU_SEC         1.0f
#define RNNOISE_SPEC_NORM_ALPHA           0.984f
#define RNNOISE_SPEC_NORM_INIT_LO          0.04f
#define RNNOISE_SPEC_NORM_INIT_HI          0.008f
#define RNNOISE_SPEC_NORM_EPS              1e-12f

/* 處理狀態 (呼叫端分配，跨 frame 保持) */
typedef struct {
    /* overlap-add 緩衝 (長度 = WIN_LEN, 只用前 OVL_LEN) */
    float synthesis_buf[RNNOISE_WIN_LEN];

    /* 特徵歷史 (conv1 需要 3 frame) */
    float erb_feat_buf[3][RNNOISE_N_BANDS];  /* ring buffer for 3 frames */
    float spec_feat_buf[3][2][RNNOISE_SPEC_BINS];
    int   feat_idx;                      /* 下一個寫入位置 (0,1,2) */
    int   feat_count;                    /* 已累積的 frame 數 */

    /* Per-band causal log-ERB mean EMA. */
    float erb_norm_state[RNNOISE_N_BANDS];

    /* Original DeepFilterNet per-bin complex-magnitude EMA. */
    float spec_norm_state[RNNOISE_SPEC_BINS];

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

/* 初始化狀態 (歸零 + ERB/complex norm 初值; 內部亦會呼叫
 * rnnoise_tables_init() — 見下方說明, 現為 no-op) */
void rnnoise_state_init(RNNoiseState *st);

/* F09 修正 (2026-07): ERB filterbank + root-Hann window 表格已改為編譯期
 * `static const` (由 gen_rnnoise_tables.c 產生 rnnoise_tables_gen.h,
 * process.c 直接 #include), 不再有任何執行期查表初始化。舊版用「ready
 * flag + __atomic acquire/release」的 lazy once-guard, 但外部審查指出:
 * 即使 flag 是 atomic, 兩個執行緒仍可能同時看到 flag==0、並行寫入同一組
 * non-atomic 表格陣列 — 依 C memory model 仍是 data race (即使寫入相同
 * 常數值, 也是未定義行為)。編譯期常數表格從根本上移除了這個共享可變
 * 狀態, 因此執行緒安全是「設計上就沒有可競爭的寫入」, 不是靠同步保證。
 *
 * 這個函式本體現在是刻意留空的 no-op, 純粹保留 API 相容 (舊呼叫端不需要
 * 修改), 呼叫與否都不影響任何 rnnoise_* API 的行為或正確性。 */
void rnnoise_tables_init(void);

/* --- 前處理 (每 frame 呼叫) --- */

/* 對 WIN_LEN 個 sample 做 analysis:
 *   1. 加 root Hann window (WIN_LEN samples)
 *   2. Zero-pad 到 N_FFT (當 WIN_LEN < N_FFT)
 *   3. FFT → N_BINS 個 complex bin, 乘 N_FFT^-0.5 (= torch normalized=True)
 *   st: 提供 scratch 空間 (scratch_buf_re/im), 不讀寫任何跨 frame 狀態
 *   frame: 長度 WIN_LEN, out_re/out_im: 長度 N_BINS */
void rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im);

/* 從 normalized FFT spectrum 計算雙路 features:
 *   1. power = |X|^2 → 三角 ERB filterbank (mode=0) → 10*log10(·+1e-10)
 *   2. ERB 先更新每-band dB EMA，再輸出 (erb_db - EMA) / 40
 *   3. 0..4 kHz complex bins 各自用 magnitude 更新 EMA，再輸出
 *      real/imag / sqrt(EMA)
 *   spec_re, spec_im: 長度 N_BINS
 *   out_erb: [3][N_BANDS]，out_spec: [3][2][SPEC_BINS]
 *   兩者皆 oldest→newest，供 k=3 temporal convolution
 *   lookahead=1 時，最新 window [t-2,t-1,t] 的 gain 對應 spectrum t-1；
 *   呼叫端必須保存/延遲正確的 spectrum，不能直接套到 t。
 *   回傳: 1 = 特徵可用 (已累積 3 frame), 0 = 尚需累積 */
int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_erb[3][RNNOISE_N_BANDS],
                             float out_spec[3][2][RNNOISE_SPEC_BINS]);

/* --- 後處理 --- */

/* Attenuation limit: 對齊 Rikorose/DeepFilterNet enhance.py 的 atten_lim_db
 * 機制 (df/enhance.py `enhanced = noisy*lim + enhanced*(1-lim)`,
 * lim = 10^(-|atten_lim_db|/20))。不是 mask clamp/floor (df/modules.py 的
 * Mask.forward 另一種、數學上不同的機制) ——這裡是線性內插:
 *   band_gains[b] = lim + band_gains[b] * (1 - lim)
 * 在 band 層級套用即可 (不需另外在 bin 層級重做一次)：因為 erb_inv 是
 * partition of unity (每一列元素和為 1)，這個仿射變換跟 rnnoise_expand_gains
 * 的矩陣乘法可交換 —— band 層級套用後再展開, 等價於展開後在 bin 層級套用,
 * 也等價於直接對 spectrum 做同樣的線性混合, 但成本只需 N_BANDS(22) 次而非
 * N_BINS(257) 次。
 * atten_lim_db <= 0 (或呼叫端選擇不呼叫這個函式) 為 no-op/停用, 行為與
 * 拿掉這個功能前完全相同。
 * band_gains: 長度 RNNOISE_N_BANDS, 就地修改。 */
/* band_gains 必須有 RNNOISE_N_BANDS 個元素：本函式無條件寫滿整段。 */
void rnnoise_apply_atten_lim(float band_gains[RNNOISE_N_BANDS],
                             float atten_lim_db);

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

/* Compile-time dispatch selected by SIMD=1/0 in the Makefile. */
const char *rnnoise_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* RNNOISE_PROCESS_H */
