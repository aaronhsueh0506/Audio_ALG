/* Whole-chain digests of the DeepFilterNet2 and GTCRN pre/post chains.
 *
 * DeepFilterNet2 returns 1 from pre_process_freq on every hop; its first mask_lookahead hops are warm-up
 * frames whose heads are discarded. The DFN2 chain below drives each as a discarded frame (garbage over the
 * state, heads left NaN, finish must return 0, state zero, generation +1 exactly), draws no random number
 * for it and hashes none of it (pre_process result 0, generation less the warm-up steps), so its digests
 * are those of a stream without warm-up frames.
 *
 * Each chain runs HOPS hops through its public frequency-domain API with a
 * seeded generator: spectra in, stand-in model outputs back (plausible values
 * plus tiny, huge and non-finite heads, non-finite state, a skipped and an
 * unwritten frame). Model inputs, the recurrent state after each frame and
 * every output spectrum are hashed bit by bit (a NaN is one value). The
 * program prints one digest per chain; the NEON and forced-scalar builds must
 * match (`make test-simd`), and a second run in the process must reproduce its
 * own digest, which catches state read before it is written.
 *
 * The recurrent state is the host's: float buffers registered in the DFN2 config
 * and the eight GTCRN tensors as slices of one array. Each chain checks the
 * registered element counts first, so a forgotten registration cannot hash zero
 * elements and still agree between builds.
 *
 * The DeepFilterNet2 grid and lookahead pair are chosen when the program is
 * built (dfn2_test_grid.h: -DDFN2_GRID_KHZ=16|32|48, -DDFN2_MASK_LOOKAHEAD,
 * -DDFN2_DF_LOOKAHEAD) and the library is handed the same grid at run time.
 * The digest printed for the shipped 48 kHz 1/1 grid is a recorded value; the
 * other grids print their own. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../lib/ainr/dfn2/dfn2_prepost.h"
#include "../lib/ainr/gtcrn/gtcrn_process.h"
#include "dfn2_test_grid.h"
#include "gtcrn_test_grid.h"

#define HOPS 640

#include "chain_standin.h"

/* ---- DeepFilterNet2 ---------------------------------------------------- */

static float dfn_gru[DFN2_TEST_GRU_ELEMENTS];
static float dfn_convp[DFN2_TEST_CONVP_ELEMENTS];
static float dfn_fwd_a[DFN2_N_BINS * DFN2_N_ERB];
static float dfn_inv_a[DFN2_N_ERB * DFN2_N_BINS];
static float dfn_fwd_b[DFN2_N_BINS * DFN2_N_ERB];
static float dfn_inv_b[DFN2_N_ERB * DFN2_N_BINS];

/* Triangular interpolation over the band border table (dfn2_test_band_borders(),
 * dfn2_test_grid.h). */
static void dfn_tables_triangular(float* fwd, float* inv)
{
    int borders[DFN2_N_ERB];
    const int last = DFN2_N_ERB - 1;
    int segment = 0;
    dfn2_test_band_borders(borders);
    if (borders[0] != 0 || borders[last] != DFN2_N_BINS) exit(2);
    for (int b = 0; b < last; ++b)
        if (borders[b + 1] - borders[b] < 2) exit(2);
    for (int k = 0; k < DFN2_N_BINS; ++k) {
        int lo, width;
        float right, left;
        while (segment + 1 < last && k >= borders[segment + 1]) ++segment;
        lo = borders[segment];
        width = borders[segment + 1] - lo;
        right = (float)(k - lo) / (float)width;
        left = 1.0f - right;
        fwd[k * DFN2_N_ERB + segment] = left * (segment == 0 ? 2.0f : 1.0f);
        fwd[k * DFN2_N_ERB + segment + 1] =
            right * (segment + 1 == last ? 2.0f : 1.0f);
        inv[segment * DFN2_N_BINS + k] = left;
        inv[(segment + 1) * DFN2_N_BINS + k] = right;
    }
}

/* Irregular spans of 1 to 40 bins at any offset, some weights exactly zero. */
static void dfn_tables_irregular(float* fwd, float* inv)
{
    const unsigned max_width = DFN2_N_BINS / 2 < 40 ? DFN2_N_BINS / 2 : 40;
    for (int b = 0; b < DFN2_N_ERB; ++b) {
        int width = 1 + (int)(rng() % max_width);
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
    const DFN2Grid* grid = dfn2_test_grid();
    DFN2PrepostConfig cfg;
    DFN2Prepost* p;
    Digest d;
    rng_state = RNG_SEED;
    digest_init(&d);
    dfn_tables_triangular(dfn_fwd_a, dfn_inv_a);
    dfn_tables_irregular(dfn_fwd_b, dfn_inv_b);
    if (dfn2_prepost_config_defaults(&cfg, grid) != 0) exit(2);
    cfg.erb_fwd = dfn_fwd_a;
    cfg.erb_inv = dfn_inv_a;
    cfg.gru_hidden = dfn_gru;
    cfg.gru_hidden_elements = DFN2_TEST_GRU_ELEMENTS;
    cfg.df_convp_history = dfn_convp;
    cfg.df_convp_history_elements = DFN2_TEST_CONVP_ELEMENTS;
    p = dfn2_prepost_create(&cfg);
    if (!p) exit(2);
    if (dfn2_prepost_hop_size(p) != DFN2_HOP_LEN ||
        dfn2_prepost_num_bins(p) != DFN2_N_BINS) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        int v, valid = 0, kind = hop_kind(h);
        const int warm = h < grid->mask_lookahead;
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
        if (warm && v != 1) exit(2);
        hash_u32(&d.status, warm ? 0u : (uint32_t)v);
        if (warm) {
            DFN2PrepostInputs in;
            DFN2PrepostOutputs out;
            const unsigned generation = dfn2_prepost_state_generation(p);
            if (dfn2_prepost_frame_open(p, &in, &out) != 0) exit(2);
            for (size_t i = 0; i < DFN2_TEST_GRU_ELEMENTS; ++i) dfn_gru[i] = 99.0f;
            for (size_t i = 0; i < DFN2_TEST_CONVP_ELEMENTS; ++i) dfn_convp[i] = -99.0f;
            if (dfn2_prepost_frame_finish(p) != 0 ||
                dfn2_prepost_state_generation(p) != generation + 1u) exit(2);
            for (size_t i = 0; i < DFN2_TEST_GRU_ELEMENTS; ++i)
                if (dfn_gru[i] != 0.0f) exit(2);
            for (size_t i = 0; i < DFN2_TEST_CONVP_ELEMENTS; ++i)
                if (dfn_convp[i] != 0.0f) exit(2);
        } else if (v == 1) {
            DFN2PrepostInputs in;
            DFN2PrepostOutputs out;
            float* gru;
            float* convp;
            int rc, skipped = 0;
            if (dfn2_prepost_frame_open(p, &in, &out) != 0) exit(2);
            if (in.erb_window_elements != dfn2_prepost_erb_window_elements(grid) ||
                in.spec_window_elements != dfn2_prepost_spec_window_elements(grid) ||
                in.gru_hidden != dfn_gru ||
                in.gru_hidden_elements != DFN2_TEST_GRU_ELEMENTS ||
                in.df_convp_window != dfn_convp ||
                in.df_convp_window_elements != DFN2_TEST_CONVP_ELEMENTS ||
                out.gru_hidden_next != dfn_gru ||
                out.df_convp_window_next != dfn_convp ||
                out.erb_mask_elements != dfn2_prepost_erb_mask_elements(grid) ||
                out.coefs_elements != dfn2_prepost_coefs_elements(grid)) exit(2);
            hash_f32(&d.in, in.erb_window, in.erb_window_elements);
            hash_f32(&d.in, in.spec_window, in.spec_window_elements);
            hash_f32(&d.in, in.gru_hidden, in.gru_hidden_elements);
            hash_f32(&d.in, in.df_convp_window, in.df_convp_window_elements);
            gru = out.gru_hidden_next;
            convp = out.df_convp_window_next;
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
            rc = kind == K_SKIP ? dfn2_prepost_frame_bypass(p)
                                : dfn2_prepost_frame_finish(p);
            if (rc != 0 && kind != K_SKIP) skipped = dfn2_prepost_frame_bypass(p);
            hash_u32(&d.status, (uint32_t)rc);
            hash_u32(&d.status, (uint32_t)skipped);
            hash_u32(&d.status, dfn2_prepost_state_generation(p) -
                                    (unsigned)grid->mask_lookahead);
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

#define GT_N_BINS GTCRN_TEST_N_BINS
#define GT_BANDS GTCRN_TEST_ERB_BANDS
#define GT_HIGH_BINS GTCRN_TEST_ERB_HIGH_BINS
#define GT_HIGH_BANDS GTCRN_TEST_ERB_HIGH_BANDS
#define GT_STATE_FLOATS GTCRN_TEST_STATE_FLOATS

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
    static float state_f[GT_STATE_FLOATS];
    static float spec[GT_N_BINS * 2], enh[GT_N_BINS * 2];
    static float re[GT_N_BINS], im[GT_N_BINS];
    float mag[GT_BANDS], rp[GT_BANDS], ip[GT_BANDS], mask[GT_BANDS * 2];
    GTCRNProcessConfig cfg;
    GTCRNProcessMemReq req;
    GTCRNProcess* p;
    void* pool;
    Digest d;
    rng_state = RNG_SEED;
    memset(state_f, 0, sizeof(state_f));
    digest_init(&d);
    gt_tables_sparse(gt_fwd_a, gt_inv_a);
    gt_tables_dense(gt_fwd_b, gt_inv_b);
    if (gtcrn_process_config_defaults(&cfg, gtcrn_test_grid()) != 0) exit(2);
    cfg.erb_fwd = gt_fwd_a;
    cfg.erb_inv = gt_inv_a;
    gtcrn_test_register(&cfg, state_f);
    {
        size_t total = 0;
        for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
            if (cfg.state[i] == NULL ||
                cfg.state_elements[i] != gtcrn_test_state_elements[i]) exit(2);
            total += cfg.state_elements[i];
        }
        if (total != GT_STATE_FLOATS) exit(2);
    }
    if (gtcrn_process_get_mem_size(&cfg, &req) != 0) exit(2);
    pool = aligned_alloc(16, ((size_t)req.bytes + 15u) & ~(size_t)15u);
    p = pool ? gtcrn_process_init(pool, (size_t)req.bytes, &cfg) : NULL;
    if (!p) exit(2);
    for (int h = 0; h < HOPS; ++h) {
        int kind = hop_kind(h);
        if (h == HOPS / 2 &&
            gtcrn_process_set_erb_matrices(p, gt_fwd_b, gt_inv_b) != 0) exit(2);
        make_spectrum(h, GT_N_BINS, re, im);
        for (int k = 0; k < GT_N_BINS; ++k) {
            spec[2 * k] = re[k];
            spec[2 * k + 1] = im[k];
        }
        gtcrn_model_input(p, spec, mag, rp, ip);
        hash_f32(&d.in, mag, GT_BANDS);
        hash_f32(&d.in, rp, GT_BANDS);
        hash_f32(&d.in, ip, GT_BANDS);
        hash_f32(&d.in, state_f, GT_STATE_FLOATS);
        if (kind == K_SKIP || kind == K_UNWRITTEN) {
            for (int i = 0; i < GT_BANDS; ++i) {
                mask[2 * i] = 1.0f;
                mask[2 * i + 1] = 0.0f;
            }
        } else {
            fill(mask, 2 * GT_BANDS, 0.5f * kind_scale(kind));
            fill(state_f, GT_STATE_FLOATS, kind_scale(kind));
            if (kind == K_BAD_HEAD) mask[2 * ((h * 37) % GT_BANDS) + (h & 1)] = NAN;
            if (kind == K_BAD_HEAD_INF)
                mask[2 * ((h * 37) % GT_BANDS) + (h & 1)] = INFINITY;
            if (is_bad_state(kind))
                poison(state_f, GT_STATE_FLOATS, poison_where(kind), h);
        }
        hash_u32(&d.status, (uint32_t)gtcrn_process_state_validate(p));
        hash_u32(&d.status, gtcrn_process_state_generation(p));
        hash_f32(&d.state, state_f, GT_STATE_FLOATS);
        gtcrn_model_output(p, mask, spec, enh);
        hash_f32(&d.post, enh, 2 * GT_N_BINS);
    }
    free(pool);
    printf("  gtcrn  in=%016llx state=%016llx post=%016llx status=%016llx\n",
           (unsigned long long)d.in, (unsigned long long)d.state,
           (unsigned long long)d.post, (unsigned long long)d.status);
    return digest_fold(&d);
}

int main(void)
{
    uint64_t dfn2 = run_dfn2();
    uint64_t gtcrn = run_gtcrn();
    if (run_dfn2() != dfn2 || run_gtcrn() != gtcrn) {
        fprintf(stderr, "FAIL: a chain does not reproduce its own digest\n");
        return EXIT_FAILURE;
    }
    printf("nn chains: backend=%s/%s digest=%016llx:%016llx\n",
           dfn2_simd_backend(), gtcrn_simd_backend(),
           (unsigned long long)dfn2, (unsigned long long)gtcrn);
    return EXIT_SUCCESS;
}
