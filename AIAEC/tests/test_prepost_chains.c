/* Whole-chain digests of the Align-ULCNet and DeepVQE-S pre/post chains.
 *
 * Each chain runs HOPS hops through its public frequency-domain class with a
 * seeded generator: spectra in, stand-in model outputs back (plausible, tiny,
 * huge and non-finite values, bad state in the newest and an older slot, a
 * skipped and an unwritten frame). The driver is a float host: it owns the
 * recurrent state buffers (NaN-filled, so an instance that did not zero them
 * at init changes the digest) and registers them. Model inputs, the recurrent
 * state after each frame and every output spectrum are hashed bit by bit (a
 * NaN is one value). A non-finite value in the OLDEST slot of a window must be
 * refused like one in the newest: the hop that plants it asserts the refusal,
 * the zeroed state and the generation step.
 *
 * Align-ULCNet runs at two delay depths on the grid given as grid=<sample
 * rate>,<n_fft> (default 16000,512); DeepVQE-S on its shipped grid.
 *
 * One line per delay depth is printed. The NEON and forced-scalar builds must
 * print the same digests (test_prepost_chains.py compares them), and a second
 * run in the process must reproduce its own, which catches state read before it
 * is written. CHAIN_ULCNET or CHAIN_DEEPVQE selects the chain; RNG_SEED the
 * seed. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/simd_test_mode.h"

#if defined(CHAIN_ULCNET)
#include "ulcnet_prepost.h"
#elif defined(CHAIN_DEEPVQE)
#include "deepvqe_prepost.h"
#else
#error "define CHAIN_ULCNET or CHAIN_DEEPVQE"
#endif

#define HOPS 640
#define BINS_MAX 513

/* The grid under test (run-time): argument grid=<sample rate>,<n_fft>. */
static int g_sample_rate = 16000;
static int g_n_fft = 512;

#if defined(CHAIN_ULCNET)
static const int kDepths[2] = { 8, 5 };
static UlcnetGrid g_grid;
#else
static const int kDepths[2] = { 32, 7 };   /* 32: the shipped half-second search range */
static DeepVqeGrid g_grid;
#endif
#define BINS (g_grid.n_bins)

static const char* chain_name(void)
{
    static char name[48];
#if defined(CHAIN_ULCNET)
    if (g_sample_rate == 16000 && g_n_fft == 512) return "ulcnet 16k";
    if (g_sample_rate == 48000 && g_n_fft == 1024) return "ulcnet 48k";
    snprintf(name, sizeof(name), "ulcnet %d/%d", g_sample_rate, g_n_fft);
#else
    snprintf(name, sizeof(name), "deepvqe %d/%d", g_sample_rate, g_n_fft);
#endif
    return name;
}

/* ---- generator and digest -------------------------------------------- */

#ifndef RNG_SEED
#define RNG_SEED 0x6b1d5eedu
#endif
static uint32_t rng_state = RNG_SEED;
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
 * element of the oldest row. */
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

static void print_digest(const char* what, int depth, const Digest* d)
{
    printf("  %s D=%d in=%016llx state=%016llx post=%016llx status=%016llx\n",
           what, depth, (unsigned long long)d->in, (unsigned long long)d->state,
           (unsigned long long)d->post, (unsigned long long)d->status);
}

/* A buffer of the application's, NaN-filled: the instance must zero it. */
static float* host_block(size_t elements)
{
    float* x = (float*)malloc(elements * sizeof(float));
    if (!x) exit(2);
    for (size_t i = 0; i < elements; ++i) x[i] = NAN;
    return x;
}

static int all_zero(const float* x, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (x[i] != 0.0f) return 0;
    return 1;
}

/* The hop that planted a non-finite value in the oldest slot of a window: the
 * frame must have been refused, the state restarted and the generation advanced
 * once. */
static void require_old_slot_refusal(int rc, uint32_t generation_before,
                                     uint32_t generation_after, int zeroed)
{
    if (rc == 0 || generation_after != generation_before + 1u || !zeroed) {
        fprintf(stderr, "FAIL: a non-finite value in an older slot was not refused "
                        "(rc=%d, generation %u -> %u, zeroed=%d)\n",
                rc, (unsigned)generation_before, (unsigned)generation_after, zeroed);
        exit(EXIT_FAILURE);
    }
}

#if defined(CHAIN_ULCNET)
/* ---- Align-ULCNet ------------------------------------------------------ */

/* State table of the shipped graph (pinned to the exporter by
 * test_export_streaming_ulcnet.py). */
#define STATE_CHANNELS 32
#define STATE_SCORE_FRAMES 5
#define STATE_GRU_ROWS 4
#define STATE_GRU_HIDDEN 128
#define STATE_TA_BINS(n_bins) (((n_bins) + 9) / 10)

typedef struct {
    float *key, *value, *logit, *gru;
    size_t nk, nl, ng;
} Host;

static void host_make(Host* h, int depth)
{
    h->nk = (size_t)STATE_CHANNELS * (size_t)depth * (size_t)STATE_TA_BINS(BINS);
    h->nl = (size_t)STATE_CHANNELS * STATE_SCORE_FRAMES * (size_t)depth;
    h->ng = (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN;
    h->key = host_block(h->nk);
    h->value = host_block(h->nk);
    h->logit = host_block(h->nl);
    h->gru = host_block(h->ng);
}

static void host_free(Host* h)
{
    free(h->key);
    free(h->value);
    free(h->logit);
    free(h->gru);
}

static int host_zeroed(const Host* h)
{
    return all_zero(h->key, h->nk) && all_zero(h->value, h->nk) &&
           all_zero(h->logit, h->nl) && all_zero(h->gru, h->ng);
}

static uint64_t run_chain(int depth)
{
    static float e_re[BINS_MAX], e_im[BINS_MAX], f_re[BINS_MAX], f_im[BINS_MAX],
        o_re[BINS_MAX], o_im[BINS_MAX];
    const UlcnetGridSpec spec = {
        .sample_rate = g_sample_rate, .n_fft = g_n_fft, .delay_depth = depth };
    UlcnetPrepostConfig cfg;
    UlcnetPrepost* p;
    Host host;
    Digest d;
    rng_state = RNG_SEED;
    digest_init(&d);
    if (ulcnet_grid_make(&g_grid, &spec) != ULCNET_GRID_OK || BINS > BINS_MAX) exit(2);
    if (ulcnet_prepost_config_defaults(&cfg, &g_grid) != 0) exit(2);
    host_make(&host, depth);
    cfg.key_history = host.key;     cfg.key_history_elements = host.nk;
    cfg.value_history = host.value; cfg.value_history_elements = host.nk;
    cfg.logit_history = host.logit; cfg.logit_history_elements = host.nl;
    cfg.h_gru = host.gru;           cfg.gru_hidden_elements = host.ng;
    p = ulcnet_prepost_create(&cfg);
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        UlcnetModelIoInputs in;
        UlcnetModelIoOutputs out;
        int kind = hop_kind(h), rc, skipped = 0;
        uint32_t generation_before;
        make_spectrum(h, BINS, e_re, e_im);
        make_spectrum(h + 500, BINS, f_re, f_im);
        hash_u32(&d.status,
                 (uint32_t)ulcnet_prepost_pre_process_freq(p, e_re, e_im, f_re, f_im));
        if (ulcnet_prepost_frame_open(p, &in, &out) != 0) exit(2);
        /* The views are the registered buffers: a forgotten registration must not
         * hash zero elements and pass. */
        if (in.key_history_elements != host.nk || in.value_history_elements != host.nk ||
            in.logit_history_elements != host.nl || in.gru_hidden_elements != host.ng ||
            in.key_history != host.key || in.h_gru != host.gru) exit(2);
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
            for (size_t i = 0; i < (size_t)BINS; ++i) {
                out.output[i] = (0.5f + 0.5f * uniform()) * s;
                out.output[(size_t)BINS + i] = 0.2f * uniform() * s;
            }
            fill(out.key_history_out, out.key_history_elements, s);
            fill(out.value_history_out, out.value_history_elements, s);
            fill(out.logit_history_out, out.logit_history_elements, s);
            fill(out.h_gru_out, out.gru_hidden_elements, s);
            if (kind == K_BAD_HEAD)
                out.output[(size_t)(h * 37) % (2 * (size_t)BINS)] = bad_value(h);
            if (kind == K_BAD_HEAD_INF)
                out.output[(size_t)(h * 41) % (2 * (size_t)BINS)] = INFINITY;
            if (is_bad_state(kind)) {
                int tensor = (h / 40 + (kind - K_BAD_STATE_NEWEST)) % 4;
                if (tensor == 0)
                    poison_window(out.key_history_out, STATE_CHANNELS, (size_t)depth,
                                  (size_t)STATE_TA_BINS(BINS), 1, kind, h);
                else if (tensor == 1)
                    poison_window(out.value_history_out, STATE_CHANNELS, (size_t)depth,
                                  (size_t)STATE_TA_BINS(BINS), 1, kind, h);
                else if (tensor == 2)
                    poison_window(out.logit_history_out, STATE_CHANNELS,
                                  STATE_SCORE_FRAMES, (size_t)depth, 0, kind, h);
                else
                    out.h_gru_out[kind == K_BAD_STATE_LAST
                                      ? out.gru_hidden_elements - 1
                                      : (size_t)(h / 40) * 31u] = bad_value(h);
            }
        }
        generation_before = ulcnet_prepost_state_generation(p);
        rc = kind == K_SKIP ? ulcnet_prepost_frame_bypass(p)
                            : ulcnet_prepost_frame_finish(p);
        if (kind == K_BAD_STATE_OLD)
            require_old_slot_refusal(rc, generation_before,
                                     ulcnet_prepost_state_generation(p),
                                     host_zeroed(&host));
        if (rc != 0 && kind != K_SKIP) skipped = ulcnet_prepost_frame_bypass(p);
        hash_u32(&d.status, (uint32_t)rc);
        hash_u32(&d.status, (uint32_t)skipped);
        hash_u32(&d.status, ulcnet_prepost_state_generation(p));
        hash_f32(&d.state, out.key_history_out, out.key_history_elements);
        hash_f32(&d.state, out.value_history_out, out.value_history_elements);
        hash_f32(&d.state, out.logit_history_out, out.logit_history_elements);
        hash_f32(&d.state, out.h_gru_out, out.gru_hidden_elements);
        if (ulcnet_prepost_post_process_freq(p, o_re, o_im) != 0) exit(2);
        hash_f32(&d.post, o_re, (size_t)BINS);
        hash_f32(&d.post, o_im, (size_t)BINS);
    }
    ulcnet_prepost_destroy(p);
    host_free(&host);
    print_digest(chain_name(), depth, &d);
    return digest_fold(&d);
}

#else
/* ---- DeepVQE-S --------------------------------------------------------- */

/* BEGIN state table: the sixteen state tensors of the shipped graph at the grid under test, as
 * [1][channels][frames][width] (pinned to the exporter by test_deepvqe_prepost_c.py). */
#define F0 (g_grid.n_bins)
#define F1 ((F0 + 1) / 2)
#define F2 ((F1 + 1) / 2)
#define F3 ((F2 + 1) / 2)
#define F4 ((F3 + 1) / 2)
#define CONV_WINDOW 4
#define SCORE_WINDOW 5

static int state_dims(int id, int depth, int dims[4])
{
    int c = 0, t = 0, f = 0;
    switch (id) {
    case DEEPVQE_STATE_ALIGN_KEY_RING:      c = 4;  t = depth; f = F2; break;
    case DEEPVQE_STATE_ALIGN_VALUE_RING:    c = 24; t = depth; f = F2; break;
    case DEEPVQE_STATE_ALIGN_SCORE_HISTORY: c = 4;  t = SCORE_WINDOW; f = depth; break;
    case DEEPVQE_STATE_CCM_UP_HISTORY:      c = 32; t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_FAR1_HISTORY:        c = 2;  t = CONV_WINDOW; f = F0; break;
    case DEEPVQE_STATE_FAR2_HISTORY:        c = 8;  t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_H_GRU:
        dims[0] = 1; dims[1] = 1; dims[2] = 1; dims[3] = 192;
        return 4;
    case DEEPVQE_STATE_MIC1_HISTORY:        c = 2;  t = CONV_WINDOW; f = F0; break;
    case DEEPVQE_STATE_MIC2_HISTORY:        c = 16; t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_MIC3_HISTORY:        c = 64; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_MIC4_HISTORY:        c = 56; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_RES2_HISTORY:        c = 32; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_RES3_HISTORY:        c = 40; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_UP1_HISTORY:         c = 32; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_UP2_HISTORY:         c = 40; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_UP3_HISTORY:         c = 24; t = CONV_WINDOW; f = F4; break;
    default: return -1;
    }
    dims[0] = 1; dims[1] = c; dims[2] = t; dims[3] = f;
    return 4;
}
/* END state table */

static size_t state_count(int id, int depth)
{
    int dims[4];
    size_t n = 1;
    if (state_dims(id, depth, dims) != 4) return 0;
    for (int k = 0; k < 4; ++k) n *= (size_t)dims[k];
    return n;
}

static uint64_t run_chain(int depth)
{
    static float m_re[BINS_MAX], m_im[BINS_MAX], f_re[BINS_MAX], f_im[BINS_MAX],
        o_re[BINS_MAX], o_im[BINS_MAX];
    static DeepVqePrepostInputs in;
    static DeepVqePrepostOutputs out;
    const DeepVqeGridSpec spec = {
        .sample_rate = g_sample_rate, .n_fft = g_n_fft, .delay_depth = depth };
    float* buf[DEEPVQE_STATE_COUNT];
    DeepVqePrepostConfig cfg;
    DeepVqePrepost* p;
    Digest d;
    rng_state = RNG_SEED;
    digest_init(&d);
    if (deepvqe_grid_make(&g_grid, &spec) != DEEPVQE_GRID_OK || BINS > BINS_MAX) exit(2);
    if (deepvqe_prepost_config_defaults(&cfg, &g_grid) != 0) exit(2);
    for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        size_t n = state_count(id, depth);
        if (n == 0) exit(2);
        buf[id] = host_block(n);
        cfg.state[id] = buf[id];
        cfg.state_elements[id] = n;
    }
    p = deepvqe_prepost_create(&cfg);
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        int kind = hop_kind(h), rc, skipped = 0;
        uint32_t generation_before;
        make_spectrum(h, BINS, m_re, m_im);
        make_spectrum(h + 500, BINS, f_re, f_im);
        hash_u32(&d.status,
                 (uint32_t)deepvqe_prepost_pre_process_freq(p, m_re, m_im, f_re, f_im));
        if (deepvqe_prepost_frame_open(p, &in, &out) != 0) exit(2);
        hash_f32(&d.in, in.mic, in.spectrum_ri_elements);
        hash_f32(&d.in, in.far, in.spectrum_ri_elements);
        for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            /* The views are the registered buffers: a forgotten registration must not
             * hash zero elements and pass. */
            if (in.state[id] != buf[id] || in.state_elements[id] != state_count(id, depth))
                exit(2);
            hash_f32(&d.in, in.state[id], in.state_elements[id]);
        }
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
                int dims[4];
                if (id == DEEPVQE_STATE_H_GRU || state_dims(id, depth, dims) < 0)
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
        generation_before = deepvqe_prepost_state_generation(p);
        rc = kind == K_SKIP ? deepvqe_prepost_frame_bypass(p)
                            : deepvqe_prepost_frame_finish(p);
        if (kind == K_BAD_STATE_OLD) {
            int zeroed = 1;
            for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id)
                zeroed = zeroed && all_zero(buf[id], state_count(id, depth));
            require_old_slot_refusal(rc, generation_before,
                                     deepvqe_prepost_state_generation(p), zeroed);
        }
        if (rc != 0 && kind != K_SKIP) skipped = deepvqe_prepost_frame_bypass(p);
        hash_u32(&d.status, (uint32_t)rc);
        hash_u32(&d.status, (uint32_t)skipped);
        hash_u32(&d.status, deepvqe_prepost_state_generation(p));
        for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            hash_f32(&d.state, out.state_out[id], out.state_elements[id]);
        if (deepvqe_prepost_post_process_freq(p, o_re, o_im) != 0) exit(2);
        hash_f32(&d.post, o_re, (size_t)BINS);
        hash_f32(&d.post, o_im, (size_t)BINS);
    }
    deepvqe_prepost_destroy(p);
    for (int id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(buf[id]);
    print_digest(chain_name(), depth, &d);
    return digest_fold(&d);
}
#endif

int main(int argc, char** argv)
{
    uint64_t got[2];
    SK_MATCH_SCALAR_TO_VECTOR_MODE();
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "grid=", 5) != 0 ||
            sscanf(argv[i] + 5, "%d,%d", &g_sample_rate, &g_n_fft) != 2) {
            fprintf(stderr, "usage: %s [grid=<sample rate>,<n_fft>]\n", argv[0]);
            return 2;
        }
    }
    for (int i = 0; i < 2; ++i) {
        got[i] = run_chain(kDepths[i]);
        if (run_chain(kDepths[i]) != got[i]) {
            fprintf(stderr, "FAIL: %s D=%d does not reproduce its own digest\n",
                    chain_name(), kDepths[i]);
            return EXIT_FAILURE;
        }
    }
    printf("prepost chains (%s): digest=D=%d:%016llx D=%d:%016llx\n", chain_name(),
           kDepths[0], (unsigned long long)got[0],
           kDepths[1], (unsigned long long)got[1]);
    return EXIT_SUCCESS;
}
