/* Align-ULCNet C pre/post-processing. Contract and parity notes: see the
 * header. Framing mirrors AIAEC/aiaec_streaming.py StreamSTFT/StreamISTFT,
 * which are the bit-exact twins of torch.stft/istft(center=True).
 *
 * FFT: audio_common's fft_wrapper (caller-owned FftHandle) -- BACKEND=
 * kiss/ne10 selects the real backend, real signals pay an RFFT/IRFFT (not
 * a mirrored full-complex FFT), twiddles live in the handle (built once at
 * fft_init/fft_create, never per call). All per-call scratch is embedded
 * in the caller-owned structs; no multi-KB stack frames, no heap, no
 * globals. Constraint: -ffp-contract=off (all owning Makefiles append it). */

#include "ulcnet_process.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Same f32 expression order as the previous shared builder
 * (AINR/dfn_process_common.h df_common_make_root_hann) so the table is
 * bit-identical to every prior run of this chain. */
void ulcnet_make_window(float window[ULCNET_N_FFT]) {
    int i;
    for (i = 0; i < ULCNET_N_FFT; ++i) {
        window[i] = sqrtf(0.5f - 0.5f * cosf(
            2.0f * (float)M_PI * (float)i / (float)ULCNET_N_FFT));
    }
}

/* ---- forward rFFT of one windowed 512-sample segment ----
 * Stages the windowed copy in st->seg (struct-owned scratch; clobber
 * permitted, so fft_forward_scratch may skip any backend defensive copy).
 * `segment` may alias st->seg itself: the windowing loop is a same-index
 * read-then-write. Output ordering contract: out_re/out_im[k] = Re/Im of
 * rfft bin k, k = 0..ULCNET_BINS-1 (DC..Nyquist) -- exactly the ordering
 * the Python parity gate compares bin by bin. */
static void ulcnet_rfft(UlcnetAnalysis *st, const float *segment,
                        float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]) {
    int k;
    for (k = 0; k < ULCNET_N_FFT; ++k)
        st->seg[k] = segment[k] * st->window[k];
    fft_forward_scratch(st->fft, st->seg, st->spec);
    for (k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = st->spec[k].r;
        out_im[k] = st->spec[k].i;
    }
}

/* ============================== analysis ============================== */

int ulcnet_analysis_init(UlcnetAnalysis *st, FftHandle *fft,
                         const float *window) {
    if (!st || !fft || !window) return -1;
    /* Reject-first: the handle must be the compiled 512 grid -- a wrong
     * size would silently break the checkpoint's feature-time contract. */
    if (fft_get_n_freqs(fft) != ULCNET_BINS) return -1;
    memset(st, 0, sizeof(*st));
    st->fft = fft;
    st->window = window;
    return 0;
}

int ulcnet_analysis_push(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP],
                         float out_re[2][ULCNET_BINS],
                         float out_im[2][ULCNET_BINS]) {
    memmove(st->history, st->history + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memcpy(st->history + ULCNET_N_FFT - ULCNET_HOP, hop_in,
           (size_t)ULCNET_HOP * sizeof(float));
    /* Saturate at 3, guarded BEFORE the increment: every consumer only
     * distinguishes 1 / 2 / >= 3 (and flush: < 2 vs >= 2), so the clamp is
     * semantics-preserving while preventing signed overflow (UB) on
     * unbounded streams. */
    if (st->hops_seen < 3) st->hops_seen++;

    if (st->hops_seen == 1) return 0;   /* frame 0 needs sample index 256 */

    if (st->hops_seen == 2) {
        /* history == x[0..511]. Frame 0 covers the reflect prefix
         * (x[256..1], i.e. x[1..256] reversed) followed by x[0..255].
         * Built directly in st->seg (allowed to alias ulcnet_rfft's
         * windowing input -- see that function's comment). */
        int i;
        for (i = 0; i < ULCNET_HOP; ++i)
            st->seg[i] = st->history[ULCNET_HOP - i];  /* x[256-i], i=0..255 */
        memcpy(st->seg + ULCNET_HOP, st->history,
               (size_t)ULCNET_HOP * sizeof(float));    /* x[0..255] */
        ulcnet_rfft(st, st->seg, out_re[0], out_im[0]);
        ulcnet_rfft(st, st->history, out_re[1], out_im[1]);
        return 2;
    }

    /* Steady state: frame k = window over the last 512 raw samples,
     * centred on hop grid point k*HOP. */
    ulcnet_rfft(st, st->history, out_re[0], out_im[0]);
    return 1;
}

int ulcnet_analysis_flush(UlcnetAnalysis *st,
                          float out_re[2][ULCNET_BINS],
                          float out_im[2][ULCNET_BINS]) {
    int i;
    if (st->hops_seen < 2) return 0;  /* centered contract needs > half a window */
    /* Trailing reflect pad mirrors the HOP samples before the last one:
     * suffix = x[L-2], x[L-3], ..., x[L-HOP-1]. One extra frame results
     * (total frames = L/HOP + 1 for hop-aligned L): it covers
     * x[L-HOP .. L-1] followed by the first HOP suffix samples. Built in
     * st->seg (aliasing permitted -- see ulcnet_rfft). */
    memcpy(st->seg, st->history + ULCNET_N_FFT - ULCNET_HOP,
           (size_t)ULCNET_HOP * sizeof(float));        /* x[L-HOP..L-1] */
    for (i = 0; i < ULCNET_HOP; ++i)
        st->seg[ULCNET_HOP + i] = st->history[ULCNET_N_FFT - 2 - i]; /* x[L-2-i] */
    ulcnet_rfft(st, st->seg, out_re[0], out_im[0]);
    return 1;
}

/* ============================== synthesis ============================= */

int ulcnet_synthesis_init(UlcnetSynthesis *st, FftHandle *fft,
                          const float *window) {
    if (!st || !fft || !window) return -1;
    if (fft_get_n_freqs(fft) != ULCNET_BINS) return -1;
    memset(st, 0, sizeof(*st));
    st->fft = fft;
    st->window = window;
    return 0;
}

int ulcnet_synthesis_push(UlcnetSynthesis *st,
                          const float re[ULCNET_BINS],
                          const float im[ULCNET_BINS],
                          float out[ULCNET_HOP]) {
    int i, k, emitted;
    for (k = 0; k < ULCNET_BINS; ++k) {
        st->spec[k].r = re[k];
        st->spec[k].i = im[k];
    }
    /* IRFFT: the Hermitian upper half is implied by the real transform --
     * no explicit mirror, no full complex FFT. Both wrapper backends
     * normalise by 1/N (fft_wrapper.h contract), matching torch.istft's
     * irfft. spec is struct-owned scratch, so input clobber is fine. */
    fft_inverse_scratch(st->fft, st->spec, st->time);

    /* Every frame lands at local offset 0: after each push the leading HOP
     * samples are finalized and shifted out, so the accumulator origin
     * always advances exactly one hop per frame. */
    for (i = 0; i < ULCNET_N_FFT; ++i) {
        float w = st->window[i];
        st->acc[i] += st->time[i] * w;
        st->env[i] += w * w;
    }
    /* Saturate at 2, guarded BEFORE the increment: consumers only
     * distinguish 0 / 1 / >= 2, so the clamp is semantics-preserving while
     * preventing signed overflow (UB) on unbounded streams. */
    if (st->frames_seen < 2) st->frames_seen++;

    emitted = 0;
    if (st->frames_seen > 1) {
        /* frame 0's finalized block lies inside the trimmed half window */
        for (i = 0; i < ULCNET_HOP; ++i) {
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
    int i, n;
    if (st->frames_seen == 0) return 0;
    n = ULCNET_N_FFT - ULCNET_HOP;
    for (i = 0; i < n; ++i) {
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
