/* Golden digests of the DeepFilterNet2 and GTCRN pre/post chains.
 *
 * Each chain is driven through its public frequency-domain API for HOPS hops
 * with a seeded generator: the analysed spectra going in, and stand-in model
 * outputs coming back (plausible head values, and on a fixed schedule tiny
 * values, huge values, non-finite heads, non-finite recurrent state, a
 * skipped frame and an unwritten frame). Everything handed to the model, the
 * recurrent state after each frame and every post-processed spectrum is
 * hashed bit by bit; a NaN counts as one value whatever its payload.
 *
 * The digests below were recorded from the reference implementation of the
 * chains. They hold for NEON and scalar builds alike, so a change that moves
 * one bit of any tensor, in either build, fails here. They also depend on the
 * host libm (log10f, sinf, powf): a probe over those functions is compared
 * first, and a host whose libm differs reports SKIP instead of a verdict.
 * Update the constants only for an intended change of the arithmetic. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"
#include "GTCRN/gtcrn_process.h"

#define HOPS 640

static const uint64_t kExpectProbe = UINT64_C(0x004f421b4e141555);
static const uint64_t kExpectDfn2  = UINT64_C(0xa7632b4049c9235c);
static const uint64_t kExpectGtcrn = UINT64_C(0x95f5eafc40027a07);

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
enum { K_NORMAL, K_TINY, K_HUGE, K_BAD_HEAD, K_BAD_HEAD_INF, K_BAD_STATE_FIRST,
       K_BAD_STATE_LAST, K_BAD_STATE_MID, K_SKIP, K_UNWRITTEN };

static int hop_kind(int hop)
{
    switch (hop % 40) {
    case 5:  return K_TINY;
    case 9:  return K_HUGE;
    case 13: return K_BAD_HEAD;
    case 17: return K_BAD_HEAD_INF;
    case 21: return K_BAD_STATE_FIRST;
    case 25: return K_BAD_STATE_LAST;
    case 29: return K_BAD_STATE_MID;
    case 33: return K_SKIP;
    case 37: return K_UNWRITTEN;
    default: return K_NORMAL;
    }
}

static float kind_scale(int kind)
{
    return kind == K_TINY ? 1e-30f : kind == K_HUGE ? 1e38f : 1.0f;
}

static void poison(float* x, size_t n, int where, int hop)
{
    static const uint32_t bad[3] = { 0x7fc00000u, 0x7f800000u, 0xff800000u };
    size_t at = where == 0 ? 0 : where == 1 ? n - 1 : n / 2;
    memcpy(&x[at], &bad[(hop / 40) % 3], sizeof(float));
}

static int poison_where(int kind)
{
    return kind == K_BAD_STATE_FIRST ? 0 : kind == K_BAD_STATE_LAST ? 1 : 2;
}

static int is_bad_state(int kind)
{
    return kind == K_BAD_STATE_FIRST || kind == K_BAD_STATE_LAST ||
           kind == K_BAD_STATE_MID;
}

static uint64_t libm_probe(void)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 1; i <= 4000; ++i) {
        float x = (float)i / 64.0f, y = 1.0f / (float)(i + 2);
        float v[3];
        v[0] = log10f(x * 1e-3f + 1e-9f);
        v[1] = sinf(3.14159274f * y * 0.5f * x);
        v[2] = powf(10.0f, -x / 20.0f);
        hash_f32(&h, v, 3);
    }
    return h;
}

/* ---- DeepFilterNet2 ---------------------------------------------------- */

static float dfn_fwd_a[DFN2_N_BINS * DFN2_N_ERB];
static float dfn_inv_a[DFN2_N_ERB * DFN2_N_BINS];
static float dfn_fwd_b[DFN2_N_BINS * DFN2_N_ERB];
static float dfn_inv_b[DFN2_N_ERB * DFN2_N_BINS];

/* Triangular interpolation over the shipped border table. */
static void dfn_tables_triangular(float* fwd, float* inv)
{
    static const int borders[32] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
        50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
        325, 378, 440, 513 };
    int segment = 0;
    for (int k = 0; k < DFN2_N_BINS; ++k) {
        int lo, width;
        float right, left;
        while (segment + 1 < 31 && k >= borders[segment + 1]) ++segment;
        lo = borders[segment];
        width = borders[segment + 1] - lo;
        right = (float)(k - lo) / (float)width;
        left = 1.0f - right;
        fwd[k * DFN2_N_ERB + segment] = left * (segment == 0 ? 2.0f : 1.0f);
        fwd[k * DFN2_N_ERB + segment + 1] =
            right * (segment + 1 == 31 ? 2.0f : 1.0f);
        inv[segment * DFN2_N_BINS + k] = left;
        inv[(segment + 1) * DFN2_N_BINS + k] = right;
    }
}

/* Irregular spans of 1 to 40 bins at any offset, some weights exactly zero. */
static void dfn_tables_irregular(float* fwd, float* inv)
{
    for (int b = 0; b < DFN2_N_ERB; ++b) {
        int width = 1 + (int)(rng() % 40);
        int lo = (int)(rng() % (unsigned)(DFN2_N_BINS - width));
        for (int k = lo; k < lo + width; ++k)
            inv[b * DFN2_N_BINS + k] = (k % 7 == 3) ? 0.0f : 0.5f + 0.5f * uniform();
    }
    for (int k = 0; k < DFN2_N_BINS; ++k) {
        int b = k * (DFN2_N_ERB - 1) / DFN2_N_BINS;
        fwd[k * DFN2_N_ERB + b] = 0.5f + 0.5f * uniform();
        fwd[k * DFN2_N_ERB + b + 1] = 0.5f + 0.5f * uniform();
    }
}

static void dfn_write_heads(const DFN2PrepostOutputs* out, int kind)
{
    float s = kind_scale(kind);
    fill(out->erb_mask, out->erb_mask_elements, 0.5f * s);
    for (size_t i = 0; i < out->erb_mask_elements; ++i)
        out->erb_mask[i] += 0.5f * s;
    fill(out->coefs, out->coefs_elements, 0.1f * s);
    out->alpha[0] = 0.5f * s * (1.0f + uniform());
}

static uint64_t run_dfn2(void)
{
    static float re[DFN2_N_BINS], im[DFN2_N_BINS], app_re[DFN2_N_BINS],
        app_im[DFN2_N_BINS], out_re[DFN2_N_BINS], out_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2Prepost* p;
    Digest d;
    digest_init(&d);
    dfn_tables_triangular(dfn_fwd_a, dfn_inv_a);
    dfn_tables_irregular(dfn_fwd_b, dfn_inv_b);
    if (dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) != 0) exit(2);
    cfg.erb_fwd = dfn_fwd_a;
    cfg.erb_inv = dfn_inv_a;
    p = dfn2_prepost_create(&cfg);
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        int v, valid = 0, kind = hop_kind(h);
        if (h == HOPS / 3)
            hash_u32(&d.status, (uint32_t)dfn2_prepost_set_atten_lim(p, 12.0f));
        if (h == HOPS / 2)
            hash_u32(&d.status,
                     (uint32_t)dfn2_prepost_set_erb_matrices(p, dfn_fwd_b,
                                                             dfn_inv_b));
        make_spectrum(h, DFN2_N_BINS, re, im);
        if (h % 4 == 3) {
            make_spectrum(h + 1, DFN2_N_BINS, app_re, app_im);
            v = dfn2_prepost_pre_process_freq_dual(p, re, im, app_re, app_im);
        } else {
            v = dfn2_prepost_pre_process_freq(p, re, im);
        }
        hash_u32(&d.status, (uint32_t)v);
        if (v == 1) {
            DFN2PrepostInputs in;
            DFN2PrepostOutputs out;
            float* gru;
            float* convp;
            int rc, skipped = 0;
            if (dfn2_prepost_frame_inputs(p, &in, &out) != 0) exit(2);
            hash_f32(&d.in, &in.erb_window[0][0], in.erb_window_elements);
            hash_f32(&d.in, &in.spec_window[0][0][0], in.spec_window_elements);
            hash_f32(&d.in, &in.gru_hidden[0][0], in.gru_hidden_elements);
            hash_f32(&d.in, &in.df_convp_window[0][0][0],
                     in.df_convp_window_elements);
            gru = &out.gru_hidden_next[0][0];
            convp = &out.df_convp_window_next[0][0][0];
            if (kind != K_SKIP && kind != K_UNWRITTEN) {
                dfn_write_heads(&out, kind);
                fill(gru, out.gru_hidden_elements, kind_scale(kind));
                fill(convp, out.df_convp_window_elements, kind_scale(kind));
                if (kind == K_BAD_HEAD)
                    out.coefs[(size_t)h % out.coefs_elements] = NAN;
                if (kind == K_BAD_HEAD_INF) out.erb_mask[h % DFN2_N_ERB] = INFINITY;
                if (is_bad_state(kind)) {
                    if (h % 2) poison(gru, out.gru_hidden_elements,
                                      poison_where(kind), h);
                    else poison(convp, out.df_convp_window_elements,
                                poison_where(kind), h);
                }
            }
            rc = kind == K_SKIP ? dfn2_prepost_frame_skip(p)
                                : dfn2_prepost_frame_commit(p);
            if (rc != 0 && kind != K_SKIP) skipped = dfn2_prepost_frame_skip(p);
            hash_u32(&d.status, (uint32_t)rc);
            hash_u32(&d.status, (uint32_t)skipped);
            hash_u32(&d.status, dfn2_prepost_state_generation(p));
            hash_f32(&d.state, gru, in.gru_hidden_elements);
            hash_f32(&d.state, convp, in.df_convp_window_elements);
        }
        if (dfn2_prepost_post_process_freq(p, out_re, out_im, &valid) != 0) exit(2);
        hash_u32(&d.status, (uint32_t)valid);
        hash_f32(&d.post, out_re, DFN2_N_BINS);
        hash_f32(&d.post, out_im, DFN2_N_BINS);
    }
    dfn2_prepost_destroy(p);
    printf("  dfn2   in=%016llx state=%016llx post=%016llx status=%016llx\n",
           (unsigned long long)d.in, (unsigned long long)d.state,
           (unsigned long long)d.post, (unsigned long long)d.status);
    return digest_fold(&d);
}

/* ---- GTCRN ------------------------------------------------------------- */

#define GT_HIGH_BINS GTCRN_MODEL_ERB_HIGH_BINS
#define GT_HIGH_BANDS GTCRN_MODEL_ERB_HIGH_BANDS
#define GT_STATE_FLOATS \
    (offsetof(GTCRNModelState, state_generation) / sizeof(float))

static float gt_fwd_a[GT_HIGH_BINS * GT_HIGH_BANDS];
static float gt_inv_a[GT_HIGH_BANDS * GT_HIGH_BINS];
static float gt_fwd_b[GT_HIGH_BINS * GT_HIGH_BANDS];
static float gt_inv_b[GT_HIGH_BANDS * GT_HIGH_BINS];

/* Two-tap interpolation, three bins per band. */
static void gt_tables_sparse(float* fwd, float* inv)
{
    for (int k = 0; k < GT_HIGH_BINS; ++k) {
        int b = k / 3;
        float w = (float)(k % 3) / 3.0f;
        fwd[k * GT_HIGH_BANDS + b] = 1.0f - w;
        inv[b * GT_HIGH_BINS + k] = 1.0f - w;
        if (b + 1 < GT_HIGH_BANDS) {
            fwd[k * GT_HIGH_BANDS + b + 1] = w;
            inv[(b + 1) * GT_HIGH_BINS + k] = w;
        }
    }
}

/* Dense random weights with some exact zeros. */
static void gt_tables_dense(float* fwd, float* inv)
{
    for (size_t i = 0; i < (size_t)GT_HIGH_BINS * GT_HIGH_BANDS; ++i) {
        fwd[i] = (i % 5 == 2) ? 0.0f : (0.5f + 0.5f * uniform()) / 8.0f;
        inv[i] = (i % 7 == 4) ? 0.0f : (0.5f + 0.5f * uniform()) / 8.0f;
    }
}

static uint64_t run_gtcrn(void)
{
    static GTCRNModelState state;
    static float spec[GTCRN_N_BINS][2], enh[GTCRN_N_BINS][2];
    static float re[GTCRN_N_BINS], im[GTCRN_N_BINS];
    float mag[GTCRN_MODEL_ERB_BANDS], rp[GTCRN_MODEL_ERB_BANDS],
        ip[GTCRN_MODEL_ERB_BANDS], mask[GTCRN_MODEL_ERB_BANDS][2];
    float* state_f = (float*)&state;
    Digest d;
    digest_init(&d);
    gt_tables_sparse(gt_fwd_a, gt_inv_a);
    gt_tables_dense(gt_fwd_b, gt_inv_b);
    gtcrn_model_state_init(&state);
    for (int h = 0; h < HOPS; ++h) {
        int kind = hop_kind(h);
        const float* fwd = h < HOPS / 2 ? gt_fwd_a : gt_fwd_b;
        const float* inv = h < HOPS / 2 ? gt_inv_a : gt_inv_b;
        make_spectrum(h, GTCRN_N_BINS, re, im);
        for (int k = 0; k < GTCRN_N_BINS; ++k) {
            spec[k][0] = re[k];
            spec[k][1] = im[k];
        }
        gtcrn_model_input(spec, fwd, mag, rp, ip);
        hash_f32(&d.in, mag, GTCRN_MODEL_ERB_BANDS);
        hash_f32(&d.in, rp, GTCRN_MODEL_ERB_BANDS);
        hash_f32(&d.in, ip, GTCRN_MODEL_ERB_BANDS);
        hash_f32(&d.in, state_f, GT_STATE_FLOATS);
        if (kind == K_SKIP || kind == K_UNWRITTEN) {
            for (int i = 0; i < GTCRN_MODEL_ERB_BANDS; ++i) {
                mask[i][0] = 1.0f;
                mask[i][1] = 0.0f;
            }
        } else {
            fill(&mask[0][0], 2 * GTCRN_MODEL_ERB_BANDS, 0.5f * kind_scale(kind));
            fill(state_f, GT_STATE_FLOATS, kind_scale(kind));
            if (kind == K_BAD_HEAD) mask[(h * 37) % GTCRN_MODEL_ERB_BANDS][h & 1] = NAN;
            if (kind == K_BAD_HEAD_INF)
                mask[(h * 37) % GTCRN_MODEL_ERB_BANDS][h & 1] = INFINITY;
            if (is_bad_state(kind))
                poison(state_f, GT_STATE_FLOATS, poison_where(kind), h);
        }
        hash_u32(&d.status, (uint32_t)gtcrn_model_state_validate(&state));
        hash_u32(&d.status, gtcrn_model_state_generation(&state));
        hash_f32(&d.state, state_f, GT_STATE_FLOATS);
        gtcrn_model_output(mask, inv, spec, enh);
        hash_f32(&d.post, &enh[0][0], 2 * GTCRN_N_BINS);
    }
    printf("  gtcrn  in=%016llx state=%016llx post=%016llx status=%016llx\n",
           (unsigned long long)d.in, (unsigned long long)d.state,
           (unsigned long long)d.post, (unsigned long long)d.status);
    return digest_fold(&d);
}

int main(void)
{
    uint64_t probe = libm_probe();
    uint64_t dfn2 = run_dfn2();
    uint64_t gtcrn = run_gtcrn();
    printf("nn golden: backend=%s/%s probe=%016llx dfn2=%016llx gtcrn=%016llx\n",
           dfn2_simd_backend(), gtcrn_simd_backend(),
           (unsigned long long)probe, (unsigned long long)dfn2,
           (unsigned long long)gtcrn);
    if (probe != kExpectProbe) {
        printf("nn golden: SKIP (host libm differs from the recorded one)\n");
        return EXIT_SUCCESS;
    }
    if (dfn2 != kExpectDfn2 || gtcrn != kExpectGtcrn) {
        fprintf(stderr, "FAIL: %s digest differs from the recorded one\n",
                dfn2 != kExpectDfn2 ? "DFN2" : "GTCRN");
        return EXIT_FAILURE;
    }
    printf("nn golden: ok\n");
    return EXIT_SUCCESS;
}
