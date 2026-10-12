/* Golden digests of the Align-ULCNet and DeepVQE-S pre/post chains.
 *
 * Each chain is driven through its public frequency-domain class for HOPS
 * hops with a seeded generator: the analysed spectra going in, and stand-in
 * model outputs coming back (plausible mask or tap values, and on a fixed
 * schedule tiny values, huge values, non-finite heads, non-finite recurrent
 * state in the newest and in an older slot, a skipped frame and an unwritten
 * frame). Everything handed to the model, the recurrent state after each
 * frame and every post-processed spectrum is hashed bit by bit; a NaN counts
 * as one value whatever its payload. Align-ULCNet runs at two delay depths,
 * the grid is the one the translation unit is built for (16 kHz / 512 by
 * default, 48 kHz / 1024 with -DULCNET_MODEL_IO_SR=48000
 * -DULCNET_MODEL_IO_N_FFT=1024).
 *
 * The digests below were recorded from the reference implementation of the
 * chains. They hold for NEON and scalar builds alike, so a change that moves
 * one bit of any tensor, in either build, fails here. They also depend on the
 * host libm (powf, hypotf): a probe over those functions is compared first,
 * and a host whose libm differs reports SKIP instead of a verdict. Update
 * the constants only for an intended change of the arithmetic.
 *
 * GOLDEN_ULCNET selects Align-ULCNet, GOLDEN_DEEPVQE selects DeepVQE-S. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GOLDEN_ULCNET)
#include "ulcnet_prepost.h"
#elif defined(GOLDEN_DEEPVQE)
#include "deepvqe_prepost.h"
#else
#error "define GOLDEN_ULCNET or GOLDEN_DEEPVQE"
#endif

#define HOPS 640

static const uint64_t kExpectProbe = UINT64_C(0x5e4f76046d8cf74a);
#if defined(GOLDEN_ULCNET) && ULCNET_MODEL_IO_SR == 48000
#define NAME "ulcnet 48k"
static const int kDepths[2] = { 8, 5 };
static const uint64_t kExpect[2] = { UINT64_C(0xd815e57ec0aeb75a),
                                     UINT64_C(0xdef98a4565713e37) };
#elif defined(GOLDEN_ULCNET)
#define NAME "ulcnet 16k"
static const int kDepths[2] = { 8, 5 };
static const uint64_t kExpect[2] = { UINT64_C(0xb2fb1d1a53ca4885),
                                     UINT64_C(0xcd949c97e10b7425) };
#else
#define NAME "deepvqe"
static const int kDepths[2] = { DEEPVQE_PREPOST_DEFAULT_D, 7 };
static const uint64_t kExpect[2] = { UINT64_C(0x3a0a500c9824fea5),
                                     UINT64_C(0xd882f322566d6bba) };
#endif

/* ---- generator and digest -------------------------------------------- */

static uint32_t rng_state = 0x6b1d5eedu;
static uint32_t rng(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static float uniform(void) { return (float)(rng() & 0xffff) / 32768.0f - 1.0f; }

static void fill(float* x, size_t n, float scale)
{
    for (size_t i = 0; i < n; ++i) x[i] = uniform() * scale;
}

typedef struct {
    uint64_t in, state, post, status;
} Digest;

static void digest_init(Digest* d)
{
    d->in = d->state = d->post = d->status = UINT64_C(14695981039346656037);
}

static void hash_u32(uint64_t* h, uint32_t v)
{
    *h = (*h ^ v) * UINT64_C(1099511628211);
}

static void hash_f32(uint64_t* h, const float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        if (isnan(x[i])) bits = 0x7fc00000u;
        else memcpy(&bits, &x[i], sizeof(bits));
        hash_u32(h, bits);
    }
}

static uint64_t digest_fold(const Digest* d)
{
    uint64_t h = UINT64_C(14695981039346656037);
    const uint64_t parts[4] = { d->in, d->state, d->post, d->status };
    for (int i = 0; i < 4; ++i) {
        hash_u32(&h, (uint32_t)parts[i]);
        hash_u32(&h, (uint32_t)(parts[i] >> 32));
    }
    return h;
}

/* Speech-like analysed frame: magnitude falls with the bin, random phase.
 * Every few frames it is quiet, loud, exactly zero or subnormal. */
static void make_spectrum(int hop, int bins, float* re, float* im)
{
    float level = 0.3f;
    if (hop % 9 == 0) level = 1e-4f;
    if (hop % 31 == 7) level = 4.0f;
    if (hop % 53 == 11) level = 0.0f;
    if (hop % 71 == 13) level = 1e-20f;
    for (int k = 0; k < bins; ++k) {
        float m = level / (1.0f + 0.05f * (float)k) *
                  (0.5f + 0.5f * fabsf(uniform()));
        re[k] = m * uniform();
        im[k] = m * uniform();
    }
}

/* What the stand-in model does on a hop. */
enum { K_NORMAL, K_TINY, K_HUGE, K_BAD_HEAD, K_BAD_HEAD_INF, K_BAD_STATE_NEWEST,
       K_BAD_STATE_LAST, K_BAD_STATE_OLD, K_SKIP, K_UNWRITTEN };

static int hop_kind(int hop)
{
    switch (hop % 40) {
    case 5:  return K_TINY;
    case 9:  return K_HUGE;
    case 13: return K_BAD_HEAD;
    case 17: return K_BAD_HEAD_INF;
    case 21: return K_BAD_STATE_NEWEST;
    case 25: return K_BAD_STATE_LAST;
    case 29: return K_BAD_STATE_OLD;
    case 33: return K_SKIP;
    case 37: return K_UNWRITTEN;
    default: return K_NORMAL;
    }
}

static float kind_scale(int kind)
{
    return kind == K_TINY ? 1e-30f : kind == K_HUGE ? 1e38f : 1.0f;
}

static float bad_value(int hop)
{
    static const uint32_t bad[3] = { 0x7fc00000u, 0x7f800000u, 0xff800000u };
    float f;
    memcpy(&f, &bad[(hop / 40) % 3], sizeof(f));
    return f;
}

/* Poison one element of a [channels][frames][width] state window. NEWEST
 * hits the first element of the newest row of a channel that varies with the
 * cycle, LAST the last element of the newest row of the last channel, OLD an
 * element of the oldest row (which the commit does not check). */
static void poison_window(float* t, size_t channels, size_t frames,
                          size_t width, int newest_first, int kind, int hop)
{
    size_t newest = newest_first ? 0 : frames - 1;
    size_t oldest = newest_first ? frames - 1 : 0;
    size_t c = (size_t)(hop / 40) % channels;
    size_t at;
    if (kind == K_BAD_STATE_NEWEST) at = (c * frames + newest) * width;
    else if (kind == K_BAD_STATE_LAST)
        at = ((channels - 1) * frames + newest) * width + width - 1;
    else at = (c * frames + oldest) * width + width / 2;
    t[at] = bad_value(hop);
}

static int is_bad_state(int kind)
{
    return kind == K_BAD_STATE_NEWEST || kind == K_BAD_STATE_LAST ||
           kind == K_BAD_STATE_OLD;
}

static uint64_t libm_probe(void)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 1; i <= 4000; ++i) {
        float x = (float)i / 64.0f, y = 1.0f / (float)(i + 2);
        float v[4];
        v[0] = powf(x * y, 0.3f);
        v[1] = powf(y, 1.0f / 0.3f);
        v[2] = powf(x, 0.3f - 1.0f);
        v[3] = hypotf(x, -y * 3.0f);
        hash_f32(&h, v, 4);
    }
    return h;
}

static void print_digest(const char* what, int depth, const Digest* d)
{
    printf("  %s D=%d in=%016llx state=%016llx post=%016llx status=%016llx\n",
           what, depth, (unsigned long long)d->in, (unsigned long long)d->state,
           (unsigned long long)d->post, (unsigned long long)d->status);
}

#if defined(GOLDEN_ULCNET)
/* ---- Align-ULCNet ------------------------------------------------------ */

#define BINS ULCNET_BINS
#define TA_BINS ULCNET_MODEL_IO_TA_BINS
#define CHANNELS ULCNET_MODEL_IO_TA_CHANNELS
#define SCORE ULCNET_MODEL_IO_SCORE_WINDOW

static uint64_t run_chain(int depth)
{
    static float e_re[BINS], e_im[BINS], f_re[BINS], f_im[BINS], o_re[BINS],
        o_im[BINS];
    UlcnetPrepostConfig cfg;
    UlcnetPrepost* p;
    Digest d;
    digest_init(&d);
    if (ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_FREQ, depth) != 0) exit(2);
    p = ulcnet_prepost_create(&cfg);
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        UlcnetModelIoInputs in;
        UlcnetModelIoOutputs out;
        int kind = hop_kind(h), rc, skipped = 0;
        make_spectrum(h, BINS, e_re, e_im);
        make_spectrum(h + 500, BINS, f_re, f_im);
        hash_u32(&d.status,
                 (uint32_t)ulcnet_prepost_pre_process_freq(p, e_re, e_im, f_re, f_im));
        if (ulcnet_prepost_frame_inputs(p, &in, &out) != 0) exit(2);
        hash_f32(&d.in, in.error_mag, in.spectrum_bins_elements);
        hash_f32(&d.in, in.far_mag, in.spectrum_bins_elements);
        hash_f32(&d.in, in.error_cos, in.spectrum_bins_elements);
        hash_f32(&d.in, in.error_sin, in.spectrum_bins_elements);
        hash_f32(&d.in, in.key_history, in.key_history_elements);
        hash_f32(&d.in, in.value_history, in.value_history_elements);
        hash_f32(&d.in, in.logit_history, in.logit_history_elements);
        hash_f32(&d.in, in.h_gru, in.gru_hidden_elements);
        if (kind != K_SKIP && kind != K_UNWRITTEN) {
            float s = kind_scale(kind);
            for (size_t i = 0; i < BINS; ++i) {
                out.output[i] = (0.5f + 0.5f * uniform()) * s;
                out.output[BINS + i] = 0.2f * uniform() * s;
            }
            fill(out.key_history_out, out.key_history_elements, s);
            fill(out.value_history_out, out.value_history_elements, s);
            fill(out.logit_history_out, out.logit_history_elements, s);
            fill(out.h_gru_out, out.gru_hidden_elements, s);
            if (kind == K_BAD_HEAD)
                out.output[(size_t)(h * 37) % (2 * BINS)] = bad_value(h);
            if (kind == K_BAD_HEAD_INF)
                out.output[(size_t)(h * 41) % (2 * BINS)] = INFINITY;
            if (is_bad_state(kind)) {
                int tensor = (h / 40 + (kind - K_BAD_STATE_NEWEST)) % 4;
                if (tensor == 0)
                    poison_window(out.key_history_out, CHANNELS, (size_t)depth,
                                  TA_BINS, 1, kind, h);
                else if (tensor == 1)
                    poison_window(out.value_history_out, CHANNELS, (size_t)depth,
                                  TA_BINS, 1, kind, h);
                else if (tensor == 2)
                    poison_window(out.logit_history_out, CHANNELS, SCORE,
                                  (size_t)depth, 0, kind, h);
                else
                    out.h_gru_out[kind == K_BAD_STATE_LAST
                                      ? out.gru_hidden_elements - 1
                                      : (size_t)(h / 40) * 31u] = bad_value(h);
            }
        }
        rc = kind == K_SKIP ? ulcnet_prepost_frame_skip(p)
                            : ulcnet_prepost_frame_commit(p);
        if (rc != 0 && kind != K_SKIP) skipped = ulcnet_prepost_frame_skip(p);
        hash_u32(&d.status, (uint32_t)rc);
        hash_u32(&d.status, (uint32_t)skipped);
        hash_u32(&d.status, ulcnet_prepost_state_generation(p));
        hash_f32(&d.state, out.key_history_out, out.key_history_elements);
        hash_f32(&d.state, out.value_history_out, out.value_history_elements);
        hash_f32(&d.state, out.logit_history_out, out.logit_history_elements);
        hash_f32(&d.state, out.h_gru_out, out.gru_hidden_elements);
        if (ulcnet_prepost_post_process_freq(p, o_re, o_im) != 0) exit(2);
        hash_f32(&d.post, o_re, BINS);
        hash_f32(&d.post, o_im, BINS);
    }
    ulcnet_prepost_destroy(p);
    print_digest(NAME, depth, &d);
    return digest_fold(&d);
}

#else
/* ---- DeepVQE-S --------------------------------------------------------- */

#define BINS AIAEC_N_BINS

static uint64_t run_chain(int depth)
{
    static float m_re[BINS], m_im[BINS], f_re[BINS], f_im[BINS], o_re[BINS],
        o_im[BINS];
    DeepVqePrepostConfig cfg;
    DeepVqePrepost* p;
    Digest d;
    digest_init(&d);
    if (deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, depth) != 0) exit(2);
    p = deepvqe_prepost_create(&cfg);
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        DeepVqePrepostInputs in;
        DeepVqePrepostOutputs out;
        int kind = hop_kind(h), rc, skipped = 0;
        make_spectrum(h, BINS, m_re, m_im);
        make_spectrum(h + 500, BINS, f_re, f_im);
        hash_u32(&d.status,
                 (uint32_t)deepvqe_prepost_pre_process_freq(p, m_re, m_im, f_re, f_im));
        if (deepvqe_prepost_frame_inputs(p, &in, &out) != 0) exit(2);
        hash_f32(&d.in, in.mic, in.spectrum_ri_elements);
        hash_f32(&d.in, in.far, in.spectrum_ri_elements);
        for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            hash_f32(&d.in, in.state[id], in.state_elements[id]);
        if (kind != K_SKIP && kind != K_UNWRITTEN) {
            float s = kind_scale(kind);
            fill(out.taps, out.taps_elements, 0.05f * s);
            for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id)
                fill(out.state_out[id], out.state_elements[id], s);
            if (kind == K_BAD_HEAD)
                out.taps[(size_t)(h * 37) % out.taps_elements] = bad_value(h);
            if (kind == K_BAD_HEAD_INF)
                out.taps[(size_t)(h * 41) % out.taps_elements] = INFINITY;
            if (is_bad_state(kind)) {
                int cycle = h / 40;
                int id = kind == K_BAD_STATE_NEWEST ? cycle % DEEPVQE_STATE_COUNT
                       : kind == K_BAD_STATE_LAST
                           ? (cycle * 5 + 3) % DEEPVQE_STATE_COUNT
                           : (cycle * 3 + 1) % DEEPVQE_STATE_COUNT;
                int dims[DEEPVQE_STATE_MAX_RANK];
                if (id == DEEPVQE_STATE_H_GRU ||
                    deepvqe_prepost_state_shape(id, depth, dims) < 0)
                    out.state_out[id][kind == K_BAD_STATE_LAST
                                          ? out.state_elements[id] - 1
                                          : (size_t)cycle * 11u] = bad_value(h);
                else
                    poison_window(out.state_out[id], (size_t)dims[1],
                                  (size_t)dims[2], (size_t)dims[3],
                                  id == DEEPVQE_STATE_ALIGN_KEY_RING ||
                                      id == DEEPVQE_STATE_ALIGN_VALUE_RING,
                                  kind, h);
            }
        }
        rc = kind == K_SKIP ? deepvqe_prepost_frame_skip(p)
                            : deepvqe_prepost_frame_commit(p);
        if (rc != 0 && kind != K_SKIP) skipped = deepvqe_prepost_frame_skip(p);
        hash_u32(&d.status, (uint32_t)rc);
        hash_u32(&d.status, (uint32_t)skipped);
        hash_u32(&d.status, deepvqe_prepost_state_generation(p));
        for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            hash_f32(&d.state, out.state_out[id], out.state_elements[id]);
        if (deepvqe_prepost_post_process_freq(p, o_re, o_im) != 0) exit(2);
        hash_f32(&d.post, o_re, BINS);
        hash_f32(&d.post, o_im, BINS);
    }
    deepvqe_prepost_destroy(p);
    print_digest(NAME, depth, &d);
    return digest_fold(&d);
}
#endif

int main(void)
{
    uint64_t probe = libm_probe();
    uint64_t got[2];
    int bad = 0;
    for (int i = 0; i < 2; ++i) got[i] = run_chain(kDepths[i]);
    printf("prepost golden (%s): probe=%016llx D=%d:%016llx D=%d:%016llx\n", NAME,
           (unsigned long long)probe, kDepths[0], (unsigned long long)got[0],
           kDepths[1], (unsigned long long)got[1]);
    if (probe != kExpectProbe) {
        printf("prepost golden (%s): SKIP (host libm differs from the recorded one)\n",
               NAME);
        return EXIT_SUCCESS;
    }
    for (int i = 0; i < 2; ++i) {
        if (got[i] != kExpect[i]) {
            fprintf(stderr, "FAIL: %s D=%d digest differs from the recorded one\n",
                    NAME, kDepths[i]);
            bad = 1;
        }
    }
    if (bad) return EXIT_FAILURE;
    printf("prepost golden (%s): ok\n", NAME);
    return EXIT_SUCCESS;
}
