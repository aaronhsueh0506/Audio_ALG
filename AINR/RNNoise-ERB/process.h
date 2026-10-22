#ifndef RNNOISE_PROCESS_H
#define RNNOISE_PROCESS_H

#include "audio_stft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * RNNoise-ERB 前後處理 (C 實現) — 對齊 train.py / inference.py
 *
 * 與 Python 參考的對應 (inference.py enhance() 為準):
 *   - STFT: root-Hann window, normalized=True (× N_FFT^-0.5)
 *   - ERB: Glasberg-Moore band borders (erb_bandborder) +
 *     三角 filterbank (compute_erb_matrix mode=0 forward / mode=1 inverse)
 *   - ERB 特徵: log energy 的每-band causal EMA mean norm。
 *   - Complex 特徵: 0..4 kHz real/imag，用每-bin magnitude EMA unit norm。
 *   - Band gain → bin gain: gains @ ERB_inv^T (mode=1, partition of unity)
 *   - ISTFT: normalized=True (× N_FFT^+0.5) + root-Hann + 50% OLA (COLA)
 *
 * 框對齊: train.py 用 center=False, 與這個 streaming 實作一致。
 * (center=True 會讓兩邊差 N_FFT/2 = 256 sample 的框偏移。這個偏移對頻譜
 *  只影響首尾邊界 frame, 但 causal EMA 正規化器的狀態從 frame 0 起累積,
 *  而 3 秒 segment 全程都在暖機, 所以 frame 0 的差異會一路傳播到整段。)
 * 合成端同樣不做 window-envelope 除法: root-Hann 分析 × root-Hann 合成
 * = Hann, 50% overlap 下 COLA 自動成立。train.py 的 istft() 亦然。
 * 特徵/增益的數學與 Python 逐式對齊, 但 matmul 的累加順序依實作而異
 * → 與 torch 為 float32 ULP 級近似, 非 bit-exact。
 * ============================================================ */

#define RNNOISE_SR          16000
#define RNNOISE_N_FFT       512
#define RNNOISE_N_BINS      257   /* N_FFT/2 + 1 */
#define RNNOISE_WIN_LEN     512   /* fixed deployed analysis window */
#define RNNOISE_HOP_LEN     256   /* fixed 50% overlap */
#define RNNOISE_OVL_LEN     (RNNOISE_WIN_LEN - RNNOISE_HOP_LEN)  /* overlap 長度 */
#define RNNOISE_N_BANDS     22    /* = config.ini [signal] n_bands (純 ERB 模式) */
#define RNNOISE_MIN_BINS_PER_BAND 2  /* = config.ini [signal] min_bins_per_band */
#define RNNOISE_LOOKAHEAD     1     /* = config.ini lookahead_frames */
#define RNNOISE_CONV_DELAY    (2 - RNNOISE_LOOKAHEAD)

/* Model-state shape from the shipped config (three GRU layers, width 128).
 * export_onnx.py exposes each hidden tensor as an input and *_out output; the
 * accelerator does not retain this state internally. RNNoiseModelState below
 * IS the state buffer. Normal path: the runtime writes h1_out/h2_out/h3_out
 * into its own output tensor and rnnoise_model_state_inherit() copies it into
 * the struct. Optimized path: bind each hidden input AND its *_out output to
 * the same field, so the updated state is the next call's input with no copy,
 * and call only rnnoise_model_state_validate() (optional). Binding table
 * (graph name -> field):
 *   h1_in / h1_out -> &state->hidden[0]
 *   h2_in / h2_out -> &state->hidden[1]
 *   h3_in / h3_out -> &state->hidden[2]
 * Aliasing precondition (optimized path): the runtime must read every state
 * input before it writes any state output at the same address. */
#define RNNOISE_PREPOST_LAYOUT_VERSION 2
#define RNNOISE_MODEL_IO_LAYOUT_VERSION 1
#define RNNOISE_MODEL_GRU_COUNT          3
#define RNNOISE_MODEL_GRU_SIZE         128

typedef struct {
    /* h1/h2/h3 each map to ONNX shape [1,1,RNNOISE_MODEL_GRU_SIZE]. */
    float hidden[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE];
} RNNoiseModelState;

/* log_erb_dfn_mean_cplx_unit_0_4k_v8 constants.  Keep byte-for-byte aligned with
 * config.ini [feature] and checkpoint validation in train.py.
 * Feature formulas follow upstream Rikorose/DeepFilterNet libDF/src/lib.rs
 * (band_mean_norm_erb/band_unit_norm) and this repo's AINR/DeepFilterNet2
 * port; neither applies an erb_norm_clip/spec_clip clamp, and neither does this.
 * ERB band borders: the minimum-width enforcement is in gen_rnnoise_tables.c
 * (algorithm described there); train.py's erb_bandborder() is its Python
 * counterpart. Both define the erb_fwd/erb_inv tables.
 * Normaliser init values are measured on this project's 16 kHz corpus.
 *     libDF's -60/-90 dB and 0.001/0.0001 are calibrated for its rectangular,
 *     energy-MEAN filterbank; this port sums over a triangular overlapping
 *     bank, which would start the EMA +35..+48 dB away from where the
 *     features sit. The ERB pair is rounded because it is in dB; the spec
 *     pair is NOT rounded past one significant figure because it is a linear
 *     magnitude the model divides by as x/sqrt(state), where 0.001 vs the
 *     measured 0.008161 would put features 2.86x out at the top bin.
 * Normaliser decay is derived from tau (1 s) as in upstream libDF
 *     _calculate_norm_alpha(sr, hop, tau): 16k/hop256 -> 0.984. alpha is the
 *     per-FRAME decay, so the memory is 1/(1-alpha) frames.
 * ⚠ The decay stays harmless only while the INIT values match the
 * steady-state distribution -- see calibrate_norm_init.py. Changing alpha
 * requires recalibrating the init, otherwise the warm-up transient grows
 * with the memory length. */
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
    /* Caller-owned framing objects; reset together with feature state. */
    AcStft *stft;
    AcIstft *istft;

    /* 特徵歷史 (conv1 需要 3 frame) */
    float erb_feat_buf[3][RNNOISE_N_BANDS];  /* ring buffer for 3 frames */
    float spec_feat_buf[3][2][RNNOISE_SPEC_BINS];
    int   feat_idx;                      /* 下一個寫入位置 (0,1,2) */
    int   feat_count;                    /* 已累積的 frame 數 */

    /* Per-band causal log-ERB mean EMA. */
    float erb_norm_state[RNNOISE_N_BANDS];

    /* Original DeepFilterNet per-bin complex-magnitude EMA. */
    float spec_norm_state[RNNOISE_SPEC_BINS];

    /* Per-call feature scratch, fully overwritten before use. */
    float scratch_power[RNNOISE_N_BINS];   /* rnnoise_compute_features 用 */
    float scratch_erb_db[RNNOISE_N_BANDS];
} RNNoiseState;

/* Fixed deployed grid: 512-point periodic root-Hann, hop 256, center=0,
 * forward N^-0.5, plain overlap-add. Synthesis itself mutates the spectra by
 * gain*N^0.5, so ispec_scale MUST stay 1. Borrowing one FFT handle is safe
 * when analysis and synthesis run sequentially; objects and their buffers
 * must outlive RNNoiseState. fft=NULL lets common carve its own FFT plan;
 * cfg=NULL is a no-op. Keep the helper's borrowed immutable model window:
 * all 512 window values must be byte-identical to the model table. Replacing
 * cfg.window with NULL makes common build a window; binding accepts it only
 * if its bytes match, which is not a portable substitute for the model table. */
void rnnoise_framing_config(AcStftConfig *cfg, FftHandle *fft);

/* Bind objects constructed from rnnoise_framing_config, clear feature state
 * and reset both framing objects. Returns 0, or -1 for a NULL state or
 * mismatched framing config.
 * The pre/post layout changed in version 2; GRU tensor layout stays version 1. */
int rnnoise_state_init(RNNoiseState *st, AcStft *stft, AcIstft *istft);

void rnnoise_model_state_init(RNNoiseModelState *state);

/* Optimized path: finite-check the state in place after an invocation.
 * h1_out/h2_out/h3_out are bound to hidden[] (see above), so nothing is
 * copied and calling this is optional for a trusted runtime. Returns 0 when
 * all 3 x 128 elements are finite. NULL returns -1. Any NaN or Inf zeroes
 * the whole state (there is no previous value to restore) and
 * returns -1, so one bad invocation cannot poison every later GRU step. The
 * check is finite-only: a partial or mixed write that is still finite cannot
 * be detected. */
int rnnoise_model_state_validate(RNNoiseModelState *state);

/* Normal path: take the runtime's own hidden-state output block (h1_out,
 * h2_out, h3_out stacked as hidden[3][128]) into the state. If hidden_out
 * already points at state->hidden the runtime wrote it in place and this is
 * a no-op returning 0 (not checked; use rnnoise_model_state_validate()).
 * Otherwise the block is finite-checked and only then copied. Returns 0 on
 * success. NULL arguments or a single NaN or Inf anywhere in the block
 * returns -1 and copies nothing: a refused inherit leaves the state intact,
 * so the caller treats the frame as failed. (The in-place path differs:
 * rnnoise_model_state_validate() restarts the state from zero.) */
int rnnoise_model_state_inherit(
    RNNoiseModelState *state,
    const float hidden_out[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE]);

/* --- 前處理 (每 frame 呼叫) --- */

/* Windowed, N^-0.5-normalized rFFT of one full 512-sample caller frame.
 * Does not advance the borrowed object's streaming history. Outputs have
 * RNNOISE_N_BINS floats each. Returns 0, or -1 for a NULL argument/binding
 * without writing outputs. */
int rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im);

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

/* band_gains[b] = lim + gain[b]*(1-lim), lim=10^(-atten_lim_db/20).
 * Matches inference.py's attenuation-limit mixture. The inverse ERB matrix
 * sums to one per bin, so applying this affine mix to the RNNOISE_N_BANDS
 * gains before expansion gives the same mixture. <=0 disables it. */
void rnnoise_apply_atten_lim(float band_gains[RNNOISE_N_BANDS],
                             float atten_lim_db);

/* 將 N_BANDS 個 band gain 經 mode=1 反向三角矩陣展開到 N_BINS 個 bin gain
 * (partition of unity: gains=1 → bin_gains=1, 無需列正規化) */
void rnnoise_expand_gains(const float *band_gains, float *bin_gains);

/* 套用 gain 並 ISTFT + overlap-add:
 *   spec_re, spec_im: 長度 N_BINS (會被修改; 內含 × N_FFT^+0.5 反正規化)
 *   bin_gains: 長度 N_BINS
 *   out_samples: 長度 HOP_LEN (256), 輸出的時域 sample
 * Returns 0, or -1 for a NULL argument/binding without modifying spectra or output.
 */
int rnnoise_synthesis(RNNoiseState *st,
                       float *spec_re, float *spec_im,
                       const float *bin_gains,
                       float *out_samples);

/* Compile-time dispatch selected by SIMD=1/0 in the Makefile. */
const char *rnnoise_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* RNNOISE_PROCESS_H */
