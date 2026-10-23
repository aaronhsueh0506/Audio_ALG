#ifndef RNNOISE_PROCESS_H
#define RNNOISE_PROCESS_H

#include "audio_stft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RNNoise-ERB signal path (C twin; Python reference: inference.py).
 * - STFT: root-Hann, normalized=True (x N_FFT^-0.5), center=False as in train.py.
 * - ERB: Glasberg-Moore band borders (erb_bandborder) with a triangular
 *   filterbank (compute_erb_matrix mode=0 forward, mode=1 inverse).
 * - ERB feature: per-band causal EMA mean norm of log energy.
 * - Complex feature: real/imag of 0..4 kHz with per-bin magnitude EMA unit norm.
 * - Band gain -> bin gain: gains @ ERB_inv^T (mode=1, partition of unity).
 * - ISTFT: normalized=True (x N_FFT^+0.5), root-Hann, 50% OLA; no window-envelope
 *   division (root-Hann x root-Hann = Hann is COLA at 50% overlap).
 * Results agree with torch to float32 ULP level (matmul accumulation order differs). */

#define RNNOISE_SR          16000
#define RNNOISE_N_FFT       512
#define RNNOISE_N_BINS      257   /* N_FFT/2 + 1 */
#define RNNOISE_WIN_LEN     512   /* fixed deployed analysis window */
#define RNNOISE_HOP_LEN     256   /* fixed 50% overlap */
#define RNNOISE_OVL_LEN     (RNNOISE_WIN_LEN - RNNOISE_HOP_LEN)  /* overlap length */
#define RNNOISE_N_BANDS     22    /* = config.ini [signal] n_bands */
#define RNNOISE_MIN_BINS_PER_BAND 2  /* = config.ini [signal] min_bins_per_band */
#define RNNOISE_LOOKAHEAD     1     /* = config.ini lookahead_frames */
#define RNNOISE_CONV_DELAY    (2 - RNNOISE_LOOKAHEAD)

/* Model state: three GRU layers of width 128. export_onnx.py exposes each hidden
 * tensor as an input and *_out output; the accelerator retains no state and
 * RNNoiseModelState is the state buffer. Normal path: the runtime writes
 * h1_out/h2_out/h3_out to its own output tensor and rnnoise_model_state_inherit()
 * copies it into the struct. Optimized path: bind each hidden input and its *_out
 * output to the same field (read every state input before writing any state
 * output) and optionally call rnnoise_model_state_validate(). Binding:
 *   h1_in / h1_out -> &state->hidden[0]
 *   h2_in / h2_out -> &state->hidden[1]
 *   h3_in / h3_out -> &state->hidden[2] */
#define RNNOISE_PREPOST_LAYOUT_VERSION 2
#define RNNOISE_MODEL_IO_LAYOUT_VERSION 1
#define RNNOISE_MODEL_GRU_COUNT          3
#define RNNOISE_MODEL_GRU_SIZE         128

typedef struct {
    /* h1/h2/h3 each map to ONNX shape [1,1,RNNOISE_MODEL_GRU_SIZE]. */
    float hidden[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE];
} RNNoiseModelState;

/* Feature contract tag; keep identical to config.ini [feature] and the checkpoint
 * validation in train.py. Features follow libDF (band_mean_norm_erb /
 * band_unit_norm) without erb_norm_clip/spec_clip clamps. ERB band borders come
 * from gen_rnnoise_tables.c (train.py erb_bandborder() is the Python
 * counterpart). The normaliser init values are calibrated for this triangular
 * filterbank on the 16 kHz corpus (calibrate_norm_init.py), not libDF's. The
 * decay alpha is per frame from tau = 1 s (16k/hop256 -> 0.984); recalibrate
 * the init if alpha changes. */
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

/* Processing state (caller-allocated, persists across frames). */
typedef struct {
    /* Caller-owned framing objects; reset together with the feature state. */
    AcStft *stft;
    AcIstft *istft;

    /* feature history (conv1 needs 3 frames) */
    float erb_feat_buf[3][RNNOISE_N_BANDS];  /* ring buffer of 3 frames */
    float spec_feat_buf[3][2][RNNOISE_SPEC_BINS];
    int   feat_idx;                      /* next write position (0..2) */
    int   feat_count;                    /* frames accumulated */

    /* Per-band causal log-ERB mean EMA. */
    float erb_norm_state[RNNOISE_N_BANDS];

    /* Original DeepFilterNet per-bin complex-magnitude EMA. */
    float spec_norm_state[RNNOISE_SPEC_BINS];

    /* Per-call feature scratch, fully overwritten before use. */
    float scratch_power[RNNOISE_N_BINS];   /* scratch for rnnoise_compute_features */
    float scratch_erb_db[RNNOISE_N_BANDS];
} RNNoiseState;

/* Fixed deployed grid: 512-point periodic root-Hann, hop 256, center=0, forward
 * N^-0.5, plain overlap-add. Synthesis scales the spectra by gain*N^0.5 itself,
 * so ispec_scale must stay 1. One FFT handle may be shared because analysis and
 * synthesis run sequentially; the objects and their buffers must outlive
 * RNNoiseState. fft=NULL lets audiolib carve its own plan; cfg=NULL is a no-op.
 * All 512 window values must be byte-identical to the model table, so keep the
 * helper's borrowed immutable window. */
void rnnoise_framing_config(AcStftConfig *cfg, FftHandle *fft);

/* Bind objects built from rnnoise_framing_config, clear the feature state and
 * reset both framing objects. Returns 0, or -1 for a NULL state or mismatched
 * framing config. */
int rnnoise_state_init(RNNoiseState *st, AcStft *stft, AcIstft *istft);

void rnnoise_model_state_init(RNNoiseModelState *state);

/* Optimized path: finite-check the state in place after an invocation
 * (h1_out/h2_out/h3_out are bound to hidden[], so nothing is copied). Returns 0
 * when all 3 x 128 elements are finite, -1 on NULL. Any NaN or Inf zeroes the
 * whole state and returns -1. A finite partial write is not detectable. */
int rnnoise_model_state_validate(RNNoiseModelState *state);

/* Normal path: take the runtime's hidden-state output block (h1_out, h2_out,
 * h3_out stacked as hidden[3][128]). If hidden_out already points at
 * state->hidden this is a no-op returning 0. Otherwise the block is
 * finite-checked, then copied. Returns -1 on NULL arguments or any NaN/Inf,
 * copying nothing (the state stays intact and the caller treats the frame as
 * failed). */
int rnnoise_model_state_inherit(
    RNNoiseModelState *state,
    const float hidden_out[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE]);

/* --- Pre-processing (per frame) --- */

/* Windowed, N^-0.5-normalized rFFT of one full 512-sample caller frame.
 * Does not advance the borrowed object's streaming history. Outputs have
 * RNNOISE_N_BINS floats each. Returns 0, or -1 for a NULL argument/binding
 * without writing outputs. */
int rnnoise_analysis(RNNoiseState *st, const float *frame, float *out_re, float *out_im);

/* Dual-branch features from a normalized FFT spectrum (spec_re/spec_im: N_BINS):
 *   1. power = |X|^2 -> triangular ERB filterbank (mode=0) -> 10*log10(. + 1e-10)
 *   2. the per-band dB EMA is updated first, then (erb_db - EMA) / 40 is output
 *   3. 0..4 kHz complex bins update their magnitude EMA, then real/imag /
 *      sqrt(EMA)
 * out_erb [3][N_BANDS] and out_spec [3][2][SPEC_BINS] are oldest to newest, for
 * the k=3 temporal convolution. With lookahead=1 the gain of the newest window
 * [t-2,t-1,t] belongs to spectrum t-1, so the caller must keep and delay the
 * right spectrum.
 * Returns 1 when features are available (3 frames accumulated), 0 otherwise. */
int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_erb[3][RNNOISE_N_BANDS],
                             float out_spec[3][2][RNNOISE_SPEC_BINS]);

/* --- Post-processing --- */

/* band_gains[b] = lim + gain[b]*(1-lim), lim=10^(-atten_lim_db/20)
 * (inference.py's attenuation-limit mixture). The inverse ERB matrix sums to one
 * per bin, so mixing the band gains before expansion is equivalent. <=0 disables. */
void rnnoise_apply_atten_lim(float band_gains[RNNOISE_N_BANDS],
                             float atten_lim_db);

/* Expand N_BANDS band gains to N_BINS bin gains through the mode=1 inverse
 * triangular matrix (partition of unity: gains=1 gives bin_gains=1). */
void rnnoise_expand_gains(const float *band_gains, float *bin_gains);

/* Apply gains, inverse STFT and overlap-add.
 *   spec_re, spec_im: N_BINS, modified in place (includes the x N_FFT^+0.5
 *                     denormalisation)
 *   bin_gains: N_BINS;  out_samples: HOP_LEN (256) time-domain samples
 * Returns 0, or -1 for a NULL argument/binding without modifying spectra or
 * output. */
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
