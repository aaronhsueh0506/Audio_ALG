/* RNNoise-ERB feature/gain kernels; caller-owned audiolib framing. */

#include "process.h"
#include <math.h>
#include <string.h>

#include "utility/simd_kernels.h"

#if SK_HAVE_NEON
#include <arm_neon.h>
#endif

_Static_assert(RNNOISE_N_FFT == 512 && RNNOISE_WIN_LEN == 512 &&
               RNNOISE_HOP_LEN == 256 && RNNOISE_N_BINS == 257,
               "RNNoise C framing is fixed at 512/512/256");

#define LOG_FLOOR 1e-10f

/* Immutable model tables, checked independently by test_rnnoise_tables.c.
 * Forward ERB doubles the two edge bands; inverse ERB sums to one per bin.
 * The periodic root-Hann is shared by analysis and synthesis. */
#include "rnnoise_tables_gen.h"

/* normalized=True 的縮放常數 */
#define STFT_NORM_FWD (1.0f / 22.62741699796952f)   /* N_FFT^-0.5, N=512 */
#define STFT_NORM_INV (22.62741699796952f)          /* N_FFT^+0.5 */

void rnnoise_framing_config(AcStftConfig *cfg, FftHandle *fft) {
    if (!cfg) return;
    ac_stft_config_default(cfg, RNNOISE_N_FFT);
    cfg->spec_scale = STFT_NORM_FWD;
    cfg->ispec_scale = 1.0f;
    cfg->fft = fft;
    cfg->window = rnn_hann_win;
}

int rnnoise_state_init(RNNoiseState *st, AcStft *stft, AcIstft *istft) {
    if (!st) return -1;
    memset(st, 0, sizeof(*st));
    AcStftConfig analysis, synthesis;
    if (ac_stft_get_config(stft, &analysis) != 0 ||
        ac_istft_get_config(istft, &synthesis) != 0 ||
        analysis.fft_size != RNNOISE_N_FFT ||
        synthesis.fft_size != RNNOISE_N_FFT || analysis.center != 0 ||
        synthesis.ola != AC_OLA_ACCUMULATE ||
        analysis.spec_scale != STFT_NORM_FWD || synthesis.ispec_scale != 1.0f ||
        memcmp(analysis.window, rnn_hann_win, sizeof(rnn_hann_win)) != 0 ||
        memcmp(synthesis.window, rnn_hann_win, sizeof(rnn_hann_win)) != 0)
        return -1;
    st->stft = stft;
    st->istft = istft;
    ac_stft_reset(stft);
    ac_istft_reset(istft);
    for (int b = 0; b < RNNOISE_N_BANDS; b++) {
        float pos = RNNOISE_N_BANDS > 1
            ? (float)b / (float)(RNNOISE_N_BANDS - 1) : 0.0f;
        st->erb_norm_state[b] = RNNOISE_ERB_NORM_INIT_LO_DB +
            pos * (RNNOISE_ERB_NORM_INIT_HI_DB - RNNOISE_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < RNNOISE_SPEC_BINS; k++) {
        float pos = RNNOISE_SPEC_BINS > 1
            ? (float)k / (float)(RNNOISE_SPEC_BINS - 1) : 0.0f;
        st->spec_norm_state[k] = RNNOISE_SPEC_NORM_INIT_LO +
            pos * (RNNOISE_SPEC_NORM_INIT_HI - RNNOISE_SPEC_NORM_INIT_LO);
    }
    return 0;
}

void rnnoise_model_state_init(RNNoiseModelState *state) {
    if (state != NULL) memset(state, 0, sizeof(*state));
}

static int hidden_finite(
    const float hidden[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE]) {
    int layer;
    int index;
    for (layer = 0; layer < RNNOISE_MODEL_GRU_COUNT; ++layer) {
        for (index = 0; index < RNNOISE_MODEL_GRU_SIZE; ++index) {
            if (!isfinite(hidden[layer][index])) return 0;
        }
    }
    return 1;
}

int rnnoise_model_state_validate(RNNoiseModelState *state) {
    if (state == NULL) return -1;
    if (!hidden_finite(state->hidden)) {
        rnnoise_model_state_init(state);
        return -1;
    }
    return 0;
}

int rnnoise_model_state_inherit(
    RNNoiseModelState *state,
    const float hidden_out[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE]) {
    if (state == NULL || hidden_out == NULL) return -1;
    /* The runtime wrote the whole block in place: nothing to check or copy
     * (copying a range onto itself is undefined). */
    if (&hidden_out[0][0] == &state->hidden[0][0]) return 0;
    if (!hidden_finite(hidden_out)) return -1;
    memcpy(state->hidden, hidden_out, sizeof(state->hidden));
    return 0;
}

/* --- 前處理: analysis --- */

int rnnoise_analysis(RNNoiseState *st, const float *frame,
                     float *out_re, float *out_im) {
    if (!st || !st->stft || !frame || !out_re || !out_im) return -1;
    ac_stft_frame_planar(st->stft, frame, out_re, out_im);
    return 0;
}

/* --- 前處理: compute features --- */

int rnnoise_compute_features(RNNoiseState *st,
                             const float *spec_re, const float *spec_im,
                             float out_erb[3][RNNOISE_N_BANDS],
                             float out_spec[3][2][RNNOISE_SPEC_BINS]) {
    /* power spectrum (normalized 域)
     * (暫存區在 st->scratch_power, 不佔用呼叫端 stack) */
    float *power = st->scratch_power;
    int i = 0;
#if SK_HAVE_NEON
    for (; i + 4 <= RNNOISE_N_BINS; i += 4) {
        float32x4_t re = vld1q_f32(spec_re + i);
        float32x4_t im = vld1q_f32(spec_im + i);
        vst1q_f32(power + i,
                  vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im)));
    }
#endif
    for (; i < RNNOISE_N_BINS; i++) {
        power[i] = spec_re[i] * spec_re[i] + spec_im[i] * spec_im[i];
    }

    /* 三角 ERB band energy → dB
     * energy = power @ erb_fwd; erb_db = 10*log10(energy + 1e-10)
     * (暫存區在 st->scratch_erb_db) */
    float *erb_db = st->scratch_erb_db;
    int b = 0;
#if SK_HAVE_NEON
    for (; b + 4 <= RNNOISE_N_BANDS; b += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int k = 0; k < RNNOISE_N_BINS; ++k) {
            sum = vaddq_f32(
                sum, vmulq_n_f32(vld1q_f32(&rnn_erb_fwd[k][b]), power[k]));
        }
        {
            float values[4];
            vst1q_f32(values, sum);
            for (int lane = 0; lane < 4; ++lane) {
                erb_db[b + lane] =
                    10.0f * log10f(values[lane] + LOG_FLOOR);
            }
        }
    }
#endif
    for (; b < RNNOISE_N_BANDS; b++) {
        float sum = 0.0f;
        for (int k = 0; k < RNNOISE_N_BINS; k++) {
            sum += power[k] * rnn_erb_fwd[k][b];
        }
        erb_db[b] = 10.0f * log10f(sum + LOG_FLOOR);
    }

    /* ERB branch: update the causal per-band dB mean with this frame first,
     * then subtract and scale, matching original DeepFilterNet band_mean_norm_erb. */
    int idx = st->feat_idx;
    const float erb_a = RNNOISE_ERB_NORM_ALPHA;
    b = 0;
#if SK_HAVE_NEON
    {
        const float32x4_t va = vdupq_n_f32(erb_a);
        const float32x4_t vb = vdupq_n_f32(1.0f - erb_a);
        const float32x4_t scale = vdupq_n_f32(RNNOISE_ERB_NORM_SCALE_DB);
        for (; b + 4 <= RNNOISE_N_BANDS; b += 4) {
            float32x4_t db = vld1q_f32(erb_db + b);
            float32x4_t mean = vaddq_f32(
                vmulq_f32(va, vld1q_f32(st->erb_norm_state + b)),
                vmulq_f32(vb, db));
            vst1q_f32(st->erb_norm_state + b, mean);
            vst1q_f32(st->erb_feat_buf[idx] + b,
                      vdivq_f32(vsubq_f32(db, mean), scale));
        }
    }
#endif
    for (; b < RNNOISE_N_BANDS; b++) {
        float mean = erb_a * st->erb_norm_state[b] + (1.0f - erb_a) * erb_db[b];
        float feat = (erb_db[b] - mean) / RNNOISE_ERB_NORM_SCALE_DB;
        st->erb_norm_state[b] = mean;
        st->erb_feat_buf[idx][b] = feat;
    }

    /* Complex branch: original DeepFilterNet libDF band_unit_norm.  Each bin
     * updates its own magnitude EMA before real/imag are normalised. */
    const float norm_a = RNNOISE_SPEC_NORM_ALPHA;
    int k = 0;
#if SK_HAVE_NEON
    {
        const float32x4_t va = vdupq_n_f32(norm_a);
        const float32x4_t vb = vdupq_n_f32(1.0f - norm_a);
        const float32x4_t eps = vdupq_n_f32(RNNOISE_SPEC_NORM_EPS);
        for (; k + 4 <= RNNOISE_SPEC_BINS; k += 4) {
            float32x4_t state = vaddq_f32(
                vmulq_f32(va, vld1q_f32(st->spec_norm_state + k)),
                vmulq_f32(vb, vsqrtq_f32(vld1q_f32(power + k))));
            float32x4_t denom = vsqrtq_f32(vaddq_f32(state, eps));
            vst1q_f32(st->spec_norm_state + k, state);
            vst1q_f32(st->spec_feat_buf[idx][0] + k,
                      vdivq_f32(vld1q_f32(spec_re + k), denom));
            vst1q_f32(st->spec_feat_buf[idx][1] + k,
                      vdivq_f32(vld1q_f32(spec_im + k), denom));
        }
    }
#endif
    for (; k < RNNOISE_SPEC_BINS; k++) {
        float magnitude = sqrtf(power[k]);
        float state = norm_a * st->spec_norm_state[k] +
            (1.0f - norm_a) * magnitude;
        float denom = sqrtf(state + RNNOISE_SPEC_NORM_EPS);
        float re = spec_re[k] / denom;
        float im = spec_im[k] / denom;
        st->spec_norm_state[k] = state;
        st->spec_feat_buf[idx][0][k] = re;
        st->spec_feat_buf[idx][1][k] = im;
    }
    st->feat_idx = (idx + 1) % 3;
    /* Only readiness matters; saturation avoids signed overflow on long streams. */
    if (st->feat_count < 3) ++st->feat_count;

    /* 需要至少 3 frame 才能送入 conv1 */
    if (st->feat_count < 3) return 0;

    /* 按時序排列 3 frame: oldest, middle, newest */
    int oldest = st->feat_idx;  /* feat_idx 指向下一個寫入位置 = 最舊的那格 */
    for (int f = 0; f < 3; f++) {
        int src = (oldest + f) % 3;
        memcpy(out_erb[f], st->erb_feat_buf[src],
               sizeof(float) * RNNOISE_N_BANDS);
        memcpy(out_spec[f], st->spec_feat_buf[src],
               sizeof(float) * 2 * RNNOISE_SPEC_BINS);
    }
    return 1;
}

/* --- 後處理: attenuation limit --- */

void rnnoise_apply_atten_lim(float band_gains[RNNOISE_N_BANDS],
                             float atten_lim_db) {
    if (atten_lim_db <= 0.0f) return;
    float lim = powf(10.0f, -atten_lim_db / 20.0f);
    int b = 0;
#if SK_HAVE_NEON
    {
        float32x4_t vlim = vdupq_n_f32(lim);
        float32x4_t mix = vdupq_n_f32(1.0f - lim);
        for (; b + 4 <= RNNOISE_N_BANDS; b += 4) {
            vst1q_f32(band_gains + b,
                      vaddq_f32(vlim,
                                vmulq_f32(vld1q_f32(band_gains + b), mix)));
        }
    }
#endif
    for (; b < RNNOISE_N_BANDS; b++) {
        band_gains[b] = lim + band_gains[b] * (1.0f - lim);
    }
}

/* --- 後處理: expand gains --- */

void rnnoise_expand_gains(const float *band_gains, float *bin_gains) {
    /* bin_gains = gains @ erb_inv^T (inference.py apply gains; mode=1 為
     * partition of unity → gains=1 對應 bin_gains=1, 無需列正規化) */
    int k = 0;
#if SK_HAVE_NEON
    for (; k + 4 <= RNNOISE_N_BINS; k += 4) {
        float32x4_t g = vdupq_n_f32(0.0f);
        for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
            float32x4_t w = {
                rnn_erb_inv[k][b], rnn_erb_inv[k + 1][b],
                rnn_erb_inv[k + 2][b], rnn_erb_inv[k + 3][b]
            };
            g = vaddq_f32(g, vmulq_n_f32(w, band_gains[b]));
        }
        vst1q_f32(bin_gains + k, g);
    }
#endif
    for (; k < RNNOISE_N_BINS; k++) {
        float g = 0.0f;
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            g += band_gains[b] * rnn_erb_inv[k][b];
        }
        bin_gains[k] = g;
    }
}

/* --- 後處理: synthesis (apply gain + IFFT + overlap-add) --- */

int rnnoise_synthesis(RNNoiseState *st,
                       float *spec_re, float *spec_im,
                       const float *bin_gains,
                       float *out_samples) {
    if (!st || !st->istft || !spec_re || !spec_im || !bin_gains ||
        !out_samples) return -1;
    /* 套用 gain, 並乘 N^+0.5 反正規化 (= torch.istft normalized=True) */
    int i = 0;
#if SK_HAVE_NEON
    for (; i + 4 <= RNNOISE_N_BINS; i += 4) {
        float32x4_t gain = vmulq_n_f32(
            vld1q_f32(bin_gains + i), STFT_NORM_INV);
        vst1q_f32(spec_re + i,
                  vmulq_f32(vld1q_f32(spec_re + i), gain));
        vst1q_f32(spec_im + i,
                  vmulq_f32(vld1q_f32(spec_im + i), gain));
    }
#endif
    for (; i < RNNOISE_N_BINS; i++) {
        spec_re[i] *= bin_gains[i] * STFT_NORM_INV;
        spec_im[i] *= bin_gains[i] * STFT_NORM_INV;
    }

    return ac_istft_push_planar(st->istft, spec_re, spec_im, out_samples)
        == RNNOISE_HOP_LEN ? 0 : -1;
}

const char *rnnoise_simd_backend(void) {
#if SK_HAVE_NEON
    return "aarch64-neon";
#else
    return "scalar";
#endif
}
