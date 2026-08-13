/* Align-ULCNet C pre/post-processing. Contract and parity notes: see the
 * header. Framing mirrors AIAEC/aiaec_streaming.py StreamSTFT/StreamISTFT,
 * which are the bit-exact twins of torch.stft/istft(center=True). */

#include "ulcnet_process.h"

#include <math.h>
#include <string.h>

/* Shared scalar/NEON radix-2 FFT + sqrt-Hann builder (header-only). */
#include "../../AINR/dfn_process_common.h"

/* ---- forward rFFT of one windowed 512-sample segment ---- */
static void ulcnet_rfft(const float *segment, const float *window,
                        float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    float re[ULCNET_N_FFT], im[ULCNET_N_FFT];
    for (int i = 0; i < ULCNET_N_FFT; ++i) {
        re[i] = segment[i] * window[i];
        im[i] = 0.0f;
    }
    df_common_fft(re, im, ULCNET_N_FFT, 0);
    memcpy(out_re, re, ULCNET_BINS * sizeof(float));
    memcpy(out_im, im, ULCNET_BINS * sizeof(float));
}

/* ============================== analysis ============================== */

void ulcnet_analysis_init(UlcnetAnalysis *st) {
    memset(st, 0, sizeof(*st));
    df_common_make_root_hann(st->window, ULCNET_N_FFT);
}

int ulcnet_analysis_push(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP],
                         float out_re[2][ULCNET_BINS],
                         float out_im[2][ULCNET_BINS]) {
    memmove(st->history, st->history + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memcpy(st->history + ULCNET_N_FFT - ULCNET_HOP, hop_in,
           (size_t)ULCNET_HOP * sizeof(float));
    st->hops_seen++;

    if (st->hops_seen == 1) return 0;   /* frame 0 needs sample index 256 */

    if (st->hops_seen == 2) {
        /* history == x[0..511]. Frame 0 covers the reflect prefix
         * (x[256..1], i.e. x[1..256] reversed) followed by x[0..255]. */
        float seg[ULCNET_N_FFT];
        for (int i = 0; i < ULCNET_HOP; ++i)
            seg[i] = st->history[ULCNET_HOP - i];      /* x[256-i], i=0..255 */
        memcpy(seg + ULCNET_HOP, st->history,
               (size_t)ULCNET_HOP * sizeof(float));    /* x[0..255] */
        ulcnet_rfft(seg, st->window, out_re[0], out_im[0]);
        ulcnet_rfft(st->history, st->window, out_re[1], out_im[1]);
        return 2;
    }

    /* Steady state: frame k = window over the last 512 raw samples,
     * centred on hop grid point k*HOP. */
    ulcnet_rfft(st->history, st->window, out_re[0], out_im[0]);
    return 1;
}

int ulcnet_analysis_flush(UlcnetAnalysis *st,
                          float out_re[2][ULCNET_BINS],
                          float out_im[2][ULCNET_BINS]) {
    if (st->hops_seen < 2) return 0;  /* centered contract needs > half a window */
    /* Trailing reflect pad mirrors the HOP samples before the last one:
     * suffix = x[L-2], x[L-3], ..., x[L-HOP-1]. One extra frame results
     * (total frames = L/HOP + 1 for hop-aligned L): it covers
     * x[L-HOP .. L-1] followed by the first HOP suffix samples. */
    float seg[ULCNET_N_FFT];
    memcpy(seg, st->history + ULCNET_N_FFT - ULCNET_HOP,
           (size_t)ULCNET_HOP * sizeof(float));        /* x[L-HOP..L-1] */
    for (int i = 0; i < ULCNET_HOP; ++i)
        seg[ULCNET_HOP + i] = st->history[ULCNET_N_FFT - 2 - i]; /* x[L-2-i] */
    ulcnet_rfft(seg, st->window, out_re[0], out_im[0]);
    return 1;
}

/* ============================== synthesis ============================= */

void ulcnet_synthesis_init(UlcnetSynthesis *st) {
    memset(st, 0, sizeof(*st));
    df_common_make_root_hann(st->window, ULCNET_N_FFT);
}

int ulcnet_synthesis_push(UlcnetSynthesis *st,
                          const float re[ULCNET_BINS],
                          const float im[ULCNET_BINS],
                          float out[ULCNET_HOP]) {
    float sre[ULCNET_N_FFT], sim[ULCNET_N_FFT];
    memcpy(sre, re, ULCNET_BINS * sizeof(float));
    memcpy(sim, im, ULCNET_BINS * sizeof(float));
    for (int k = 1; k < ULCNET_N_FFT / 2; ++k) {       /* Hermitian mirror */
        sre[ULCNET_N_FFT - k] = sre[k];
        sim[ULCNET_N_FFT - k] = -sim[k];
    }
    df_common_fft(sre, sim, ULCNET_N_FFT, 1);          /* includes the 1/N */

    /* Every frame lands at local offset 0: after each push the leading HOP
     * samples are finalized and shifted out, so the accumulator origin
     * always advances exactly one hop per frame. */
    for (int i = 0; i < ULCNET_N_FFT; ++i) {
        float w = st->window[i];
        st->acc[i] += sre[i] * w;
        st->env[i] += w * w;
    }
    st->frames_seen++;

    int emitted = 0;
    if (st->frames_seen > 1) {
        /* frame 0's finalized block lies inside the trimmed half window */
        for (int i = 0; i < ULCNET_HOP; ++i) {
            float e = st->env[i];
            out[i] = st->acc[i] / (e > 1e-11f ? e : 1e-11f);
        }
        emitted = ULCNET_HOP;
    }
    memmove(st->acc, st->acc + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memset(st->acc + ULCNET_N_FFT - ULCNET_HOP, 0,
           (size_t)ULCNET_HOP * sizeof(float));
    memmove(st->env, st->env + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memset(st->env + ULCNET_N_FFT - ULCNET_HOP, 0,
           (size_t)ULCNET_HOP * sizeof(float));
    return emitted;
}

int ulcnet_synthesis_flush(UlcnetSynthesis *st, float out[ULCNET_N_FFT]) {
    if (st->frames_seen == 0) return 0;
    int n = ULCNET_N_FFT - ULCNET_HOP;
    for (int i = 0; i < n; ++i) {
        float e = st->env[i];
        out[i] = st->acc[i] / (e > 1e-11f ? e : 1e-11f);
    }
    return n;
}

/* ========================= optional compression ======================= */

static float ulcnet_signed_pow(float x, float e) {
    float m = powf(fabsf(x), e);
    return x < 0.0f ? -m : m;
}

void ulcnet_compress_frame(const float re[ULCNET_BINS],
                           const float im[ULCNET_BINS],
                           float zr[ULCNET_BINS], float zi[ULCNET_BINS]) {
    for (int k = 0; k < ULCNET_BINS; ++k) {
        zr[k] = ulcnet_signed_pow(re[k], ULCNET_COMPRESSION_EXP);
        zi[k] = ulcnet_signed_pow(im[k], ULCNET_COMPRESSION_EXP);
    }
}

void ulcnet_expand_frame(const float re[ULCNET_BINS],
                         const float im[ULCNET_BINS],
                         float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    const float inv = 1.0f / ULCNET_COMPRESSION_EXP;
    for (int k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = ulcnet_signed_pow(re[k], inv);
        out_im[k] = ulcnet_signed_pow(im[k], inv);
    }
}
