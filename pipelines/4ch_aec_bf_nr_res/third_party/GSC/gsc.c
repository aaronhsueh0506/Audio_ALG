#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gsc.h"
#include "mem_align.h"
#include "../utility/complex.h"
#include "../utility/spatial_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * GSC blocking output mode switch
 *
 * 0: Original full blocking-matrix version
 *    - explicitly builds B[f][M][M] = I - a a^H / (a^H a)
 *    - computes u = B^H x
 *    - slower, but kept as baseline for A/B comparison
 *
 * 1: Projection-form optimized version
 *    - does not explicitly build B
 *    - computes u = x - a * (a^H x / a^H a)
 *    - mathematically equivalent to the original blocking matrix form
 */
/*
 * RLS numerical-hardening constants (not part of the original supplied
 * algorithm -- see third_party/README.md).
 *
 * GSC_WA_LEAK: leak on the adaptive weight state g->wa, applied as
 * `wa = GSC_WA_LEAK * wa + update` instead of the unleaked `wa += update`.
 * Chosen far slower than the RLS covariance's own forgetting time constant
 * (1/(1-lambda), a few hundred UPDATES at the shipped lambda=0.995) so it is
 * negligible across any normal utterance/adaptation timescale and only
 * bounds wa's growth over many continuous-operation UPDATES with no
 * forgetting mechanism otherwise (e.g. sustained DOA mismatch or
 * non-stationary noise).
 *
 * This does NOT apply every hop: it is inside the `do_adapt_this_frame`
 * block below, itself gated on the caller's `allow_adapt_in` (typically
 * `vad_out == 0` -- RLS adapts only on non-target-speech frames),
 * `adapt_interval` (optionally skips further frames), and this bin's own
 * mask-freeze. Time constant 1/(1-GSC_WA_LEAK) = 100000 UPDATES, not hops --
 * the wall-clock equivalent is 100000 / (fraction of hops where this bin
 * actually updates), so it is slower than a naive "every hop" reading would
 * suggest whenever speech/masking reduces the update rate (which is the
 * safe direction: a slower safety net, not a faster one).
 */
#define GSC_WA_LEAK 0.99999f

/*
 * RLS per-bin divergence guard + diagonal loading (2026-08-04, real recording
 * repro: pipeline_failed at frame ~2800-3300, ~20-27s in, at every checked-in
 * grid and both scalar/SIMD -- see AEC/NR/Audio_ALG review notes).
 *
 * The per-bin covariance downdate below, P = (P - gain*q) / lambda, has no
 * compensating floor/ceiling: dividing by lambda < 1 every adapted frame
 * amplifies P's diagonal unless the gain*q correction exactly cancels it.
 * Under sustained low excitation at a bin (the blocking-output u carries
 * little energy there -- e.g. no interferer in that direction/frequency for
 * an extended stretch of real audio, which short synthetic tests rarely
 * exercise), the correction underflows in float32 while the /lambda growth
 * does not, so P's diagonal drifts upward without bound over enough frames
 * and eventually overflows to inf -- corrupting gain/wa/gsc_spectrum and
 * tripping the caller's isfinite() gate (FOUR_AEC_NR_RES_DSP_ERROR).
 *
 * Two independent guards, both per-bin (not a whole-GSC reset, which would
 * discard every OTHER bin's adaptation over one bin's fault):
 *
 *   1. GSC_P_DIAG_FLOOR/CEIL: clamp every bin's P diagonal after each
 *      update, every frame -- proactive, keeps P inside a numerically safe
 *      range well before it could reach inf. CEIL is chosen generously
 *      large relative to P's identity initial condition (1.0) so it never
 *      engages anywhere near a real converged operating point.
 *   2. The isfinite(upu_real) && upu_real > 0 check before this bin's
 *      update uses last frame's (possibly already-corrupted) P: reactive,
 *      catches whatever the floor/ceil didn't (e.g. corruption from a
 *      source other than diagonal growth) and recovers by resetting only
 *      this bin's P to identity and wa to zero, skipping this bin's update
 *      for the current frame instead of propagating a poisoned gain.
 */
#define GSC_P_DIAG_FLOOR 1e-6f
#define GSC_P_DIAG_CEIL  1e6f

/* ===================== adapt-interval derivation ===================== */

int gsc_effective_adapt_interval(
    int enable_fix_mode, int fixed_align_notebook, int adapt_interval)
{
    int effective = adapt_interval > 0 ? adapt_interval : 1;

    /* For fixed notebook-alignment experiments, keep the original
     * frame-by-frame adaptive update behavior for easier baseline matching. */
    if (enable_fix_mode && fixed_align_notebook) {
        effective = 1;
    }
    return effective;
}

/* ===================== pool-first memory layout ===================== */

/*
 * Bump allocator carving GSC's flat pool into typed sub-blocks. This is a
 * private, file-local reimplementation of 4aec_nr_res.c's PoolCursor/
 * pool_carve() helper (that file keeps its own copy file-private too, per
 * its own header comment) -- not shared/exported -- so gsc_get_mem_size()
 * and gsc_init() below carve in exact lockstep, the same relationship
 * aec_get_mem_size()/aec_init() maintain in AEC/c_impl/src/aec.c.
 */
typedef struct PoolCursor {
    uint8_t* ptr;
    size_t remaining;
} PoolCursor;

static void* pool_carve(PoolCursor* cursor, size_t count,
                        size_t element_size) {
    size_t raw;
    size_t aligned;
    void* out;
    if (!cursor || count == 0 || element_size == 0) return NULL;
    raw = ck_mul_size(count, element_size);
    aligned = ck_align16_size(raw);
    if (MEM_SIZE_INVALID(raw) || MEM_SIZE_INVALID(aligned) ||
        aligned > cursor->remaining) return NULL;
    out = cursor->ptr;
    cursor->ptr += aligned;
    cursor->remaining -= aligned;
    return out;
}

/*
 * Total byte requirement for a GSC instance of shape (M, F). Deliberately
 * independent of num_angles/cfg: neither affects GSC's own storage --
 * num_angles only indexes into the caller-owned a_array (never carved from
 * this pool), and cfg's fields are plain scalars copied straight into the
 * struct. Carve order below MUST stay in lockstep with gsc_init()'s carve
 * order.
 */
size_t gsc_get_mem_size(int M, int F) {
    size_t total;
    size_t fM;

    if (M <= 0 || F <= 0) return 0;

    total = ck_align16_size(sizeof(GSC));

    /* P (F, M, M): Complex**[F] -> Complex*[F*M] -> Complex[F*M*M] */
    total = ck_field_size(total, (size_t)F, sizeof(Complex**));
    fM = ck_mul_size((size_t)F, (size_t)M);
    total = ck_field_size(total, fM, sizeof(Complex*));
    total = ck_field_size(total, ck_mul_size(fM, (size_t)M), sizeof(Complex));

    /* wa (M, F): Complex*[M] -> Complex[M*F] */
    total = ck_field_size(total, (size_t)M, sizeof(Complex*));
    total = ck_field_size(
        total, ck_mul_size((size_t)M, (size_t)F), sizeof(Complex));

    /* scratch: flat Complex[(M+3)*F] (+ F*M*M only when the original
     * full-blocking-matrix baseline is compiled in instead of the default
     * projection-form path -- see GSC_USE_PROJECTION_BLOCKING above). */
    {
        /* (size_t)M + 3, not (size_t)(M + 3): the latter computes M + 3 in
         * `int` arithmetic before the cast, which is signed-overflow UB if a
         * caller ever passes M within 3 of INT_MAX (M is always the fixed
         * channel count in practice, but this is a public entry point).
         * (size_t)M + 3 promotes M to size_t (safe: M > 0 is already
         * checked above) before the add, so it can never overflow. */
        size_t scratch_count = ck_mul_size((size_t)M + 3, (size_t)F);
#if !GSC_USE_PROJECTION_BLOCKING
        scratch_count =
            ck_add_size(scratch_count, ck_mul_size(fM, (size_t)M));
#endif
        total = ck_field_size(total, scratch_count, sizeof(Complex));
    }

    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/*
 * Caller-pool construction: places the GSC struct at mem[0] and carves every
 * backing array (P/wa/scratch) out of the same block -- no malloc called.
 * Applies the exact same validation gsc_create() below does. `a_array`
 * remains a BORROWED pointer, exactly like gsc_create(): owned by the
 * caller (typically SRP), never carved from this pool or freed here.
 */
GSC* gsc_init(void* mem, size_t mem_size, int M, int F,
              int num_angles, Complex*** a_array,
              const GSC_Config* cfg) {
    GSC* g;
    PoolCursor cursor;
    size_t need;
    size_t fM;

    if (!mem || !cfg || !a_array || M <= 0 || F <= 0 || num_angles <= 0 ||
        !isfinite(cfg->lambda) || cfg->lambda <= 0.0f ||
        cfg->lambda > 1.0f ||
        !isfinite(cfg->mu)) {
        return NULL;
    }

    need = gsc_get_mem_size(M, F);
    if (need == 0 || !MEM_IS_ALIGNED16(mem) || mem_size < need) return NULL;

    /* Only the region gsc_get_mem_size() actually budgeted is touched --
     * a caller-supplied pool larger than `need` keeps any trailing bytes
     * beyond it untouched (see test_gsc_init_pool_poison_and_bounds). */
    memset(mem, 0, need);
    g = (GSC*)mem;

    cursor.ptr = (uint8_t*)mem + ck_align16_size(sizeof(GSC));
    cursor.remaining = need - ck_align16_size(sizeof(GSC));

    g->enable = cfg->enable;
    g->M = M;
    g->F = F;
    g->num_angles = num_angles;
    g->a_array = a_array;
    g->lambda = cfg->lambda;
    g->mu = cfg->mu;
    g->enable_fix_mode = cfg->enable_fix_mode;
    g->fixed_doa_rad = cfg->fixed_doa_rad;
    g->fixed_align_notebook = cfg->fixed_align_notebook;

    g->adapt_interval = gsc_effective_adapt_interval(
        g->enable_fix_mode, g->fixed_align_notebook, cfg->adapt_interval);

    fM = (size_t)F * (size_t)M;

    /* P (F, M, M): pointer table -> mid pointer table -> flat data,
     * identity-initialized -- same values the old nested-loop calloc/malloc
     * version filled. */
    {
        Complex*** p_top = (Complex***)pool_carve(
            &cursor, (size_t)F, sizeof(Complex**));
        Complex** p_mid = (Complex**)pool_carve(
            &cursor, fM, sizeof(Complex*));
        Complex* p_data = (Complex*)pool_carve(
            &cursor, fM * (size_t)M, sizeof(Complex));
        if (!p_top || !p_mid || !p_data) return NULL;

        g->P = p_top;
        for (int f = 0; f < F; f++) {
            g->P[f] = p_mid + (size_t)f * (size_t)M;
            for (int i = 0; i < M; i++) {
                g->P[f][i] =
                    p_data + ((size_t)f * (size_t)M + (size_t)i) * (size_t)M;
                for (int j = 0; j < M; j++) {
                    g->P[f][i][j].r = (i == j);
                    g->P[f][i][j].i = 0;
                }
            }
        }
    }

    /* wa (M, F): pointer table -> flat data. Already zeroed by the memset
     * above, matching the old calloc's zero-fill. */
    {
        Complex** wa_top = (Complex**)pool_carve(
            &cursor, (size_t)M, sizeof(Complex*));
        Complex* wa_data = (Complex*)pool_carve(
            &cursor, fM, sizeof(Complex));
        if (!wa_top || !wa_data) return NULL;

        g->wa = wa_top;
        for (int m = 0; m < M; m++) {
            g->wa[m] = wa_data + (size_t)m * (size_t)F;
        }
    }

    /* scratch: one flat block, already zeroed by the memset above --
     * pointer arithmetic copied verbatim from the previous calloc'd
     * version. */
    {
        size_t count = ((size_t)M + 3) * (size_t)F;   /* see gsc_get_mem_size() */
#if !GSC_USE_PROJECTION_BLOCKING
        count += fM * (size_t)M;
#endif
        g->scratch = (Complex*)pool_carve(&cursor, count, sizeof(Complex));
        if (!g->scratch) return NULL;
        g->scratch_das = g->scratch;
        g->scratch_wu = g->scratch_das + F;
        g->scratch_spec = g->scratch_wu + F;
        g->scratch_u = g->scratch_spec + F;
#if !GSC_USE_PROJECTION_BLOCKING
        g->scratch_b = g->scratch_u + (size_t)M * F;
#endif
    }

    /* Lockstep proof: gsc_get_mem_size()'s walk and the carves above must
     * consume exactly `need` bytes, no more, no less. */
    if (cursor.remaining != 0) return NULL;

    g->initialized = 0;
    g->first_doa_found = 0;
    g->first_doa_frame = -1;
    g->current_doa = 0;
    g->frame_idx = 0;
    g->doa_used = NAN;
    g->adaptive = 0;
    g->bin_resets = 0;
    g->owned_heap = NULL;

    return g;
}

/* ===================== create ===================== */

GSC* gsc_create(int M, int F, int num_angles,
                Complex*** a_array,
                const GSC_Config* cfg)
{
    size_t need;
    void* pool = NULL;
    GSC* g;

    need = gsc_get_mem_size(M, F);
    if (need == 0) return NULL;

    if (posix_memalign(&pool, 16, need) != 0 || !pool) return NULL;

    g = gsc_init(pool, need, M, F, num_angles, a_array, cfg);
    if (!g) {
        free(pool);
        return NULL;
    }
    g->owned_heap = pool;
    return g;
}

/* Reset one bin's RLS state to the same fresh-start condition gsc_create()
 * gives every bin (P = identity, wa = 0). Used by the per-bin divergence
 * guards in the RLS update below; see GSC_P_DIAG_FLOOR/CEIL's comment. */
static void gsc_reset_bin(GSC* g, int f)
{
    for (int i = 0; i < g->M; i++) {
        for (int j = 0; j < g->M; j++) {
            g->P[f][i][j].r = (i == j) ? 1.0f : 0.0f;
            g->P[f][i][j].i = 0.0f;
        }
        g->wa[i][f].r = 0.0f;
        g->wa[i][f].i = 0.0f;
    }
    g->bin_resets += 1;
}

/* ===================== DOA index ===================== */

static int doa2index(float doa_rad, int num_angles)
{
    if (isnan(doa_rad)) return 0;

    float deg = doa_rad * 180.0f / M_PI;

    while (deg < 0.0f)   deg += 360.0f;
    while (deg >= 360.0f) deg -= 360.0f;

    float step = 360.0f / num_angles;

    int idx = (int)roundf(deg / step);

    if (idx == num_angles) idx = 0;
    if (idx < 0) idx = 0;
    if (idx >= num_angles) idx = num_angles - 1;

    return idx;
}

/* ===================== process ===================== */
void gsc_process_with_weights(GSC* g,
                              const Complex* const* X,
                              float doa_s,
                              int allow_adapt_in,
                              const int* mask,
                              Complex* gsc_out,
                              Complex* effective_weights)
{
    float doa_use;
    int doa_idx;
    Complex** a;   // (M,F)
    int allow_adapt = 0;
    int use_notebook_update = 0;
    int use_mask_freeze = 1;
    int use_mu_scaling = 1;

    /* ---------- GSC bypass ---------- */
    if (!g->enable) {
        for (int f = 0; f < g->F; f++) {
            gsc_out[f] = X[0][f];
            if (effective_weights) {
                for (int m = 0; m < g->M; ++m) {
                    effective_weights[m * g->F + f].r =
                        m == 0 ? 1.0f : 0.0f;
                    effective_weights[m * g->F + f].i = 0.0f;
                }
            }
        }

        g->doa_used = NAN;
        g->adaptive = 0;
        g->frame_idx += 1;
        return;
    }

    /* fixed_align_notebook=1 is used only for fixed-mode baseline matching. */
    if (g->enable_fix_mode && g->fixed_align_notebook) {
        use_notebook_update = 1;
        use_mask_freeze = 0;
        use_mu_scaling = 0;
    }

    /* ---------- DOA logic ---------- */
    /*
     * Policy after SRP refactor:
     *
     * - In auto mode, doa_s is already the final smoothed + held DOA
     *   prepared by SRP/DOA logic.
     * - GSC no longer performs its own DOA hold policy.
     * - If doa_s is still NAN, it means no usable DOA exists yet, so
     *   bypass to mic-0 and do not adapt.
     * - Fixed mode remains inside GSC for baseline experiments.
     * - allow_adapt_in is provided by caller, usually (vad_out == 0),
     *   so RLS adaptation updates only on non-target-speech frames.
     */
    if (g->enable_fix_mode) {
        doa_use = g->fixed_doa_rad;
        g->current_doa = doa_use;
        g->first_doa_found = 1;
        allow_adapt = allow_adapt_in ? 1 : 0;
    } else {
        if (isnan(doa_s)) {
            for (int f = 0; f < g->F; f++) {
                gsc_out[f] = X[0][f];
                if (effective_weights) {
                    for (int m = 0; m < g->M; ++m) {
                        effective_weights[m * g->F + f].r =
                            m == 0 ? 1.0f : 0.0f;
                        effective_weights[m * g->F + f].i = 0.0f;
                    }
                }
            }

            g->doa_used = NAN;
            g->adaptive = 0;
            g->frame_idx += 1;
            return;
        }

        doa_use = doa_s;
        g->current_doa = doa_use;
        allow_adapt = allow_adapt_in ? 1 : 0;

        if (!g->first_doa_found) {
            g->first_doa_found = 1;
            g->first_doa_frame = g->frame_idx;
        }
    }

    g->doa_used = doa_use;
    doa_idx = doa2index(doa_use, g->num_angles);
    a = g->a_array[doa_idx];   // (M,F)

    /* ---------- temp buffers ---------- */
    Complex* das = g->scratch_das;
    Complex* wu = g->scratch_wu;
    Complex* gsc_spec = g->scratch_spec;
    Complex (*u)[g->F] =
        (Complex (*)[g->F])g->scratch_u;

#if !GSC_USE_PROJECTION_BLOCKING
    Complex (*B)[g->M][g->M] =
        (Complex (*)[g->M][g->M])g->scratch_b;
#endif

    /* ---------- DAS beamforming ---------- */
    spatial_conj_beamform(
        (const Complex* const*)a,
        (const Complex* const*)X,
        g->M, g->F, 1.0f / (float)g->M, das);

#if GSC_USE_PROJECTION_BLOCKING

    /* ---------- Blocking output: projection form ---------- */
    /*
     * Original blocking matrix:
     *   B = I - a a^H / (a^H a)
     *   u = B^H x
     *
     * Equivalent projection form:
     *   u = x - a * (a^H x / a^H a)
     *
     * In the DAS block above:
     *   das[f] = a^H x / M
     *
     * Therefore:
     *   a^H x / a^H a = das[f] * M / denom
     *
     * This avoids explicitly building B[f][M][M] and reduces the
     * blocking-output calculation from O(F*M*M) to O(F*M).
     */
    for (int f = 0; f < g->F; f++) {
        float denom = 0.0f;

        for (int m = 0; m < g->M; m++) {
            denom += a[m][f].r * a[m][f].r + a[m][f].i * a[m][f].i;
        }

        if (denom < 1e-12f) {
            denom = 1e-12f;
        }

        Complex proj;
        proj.r = das[f].r * ((float)g->M / denom);
        proj.i = das[f].i * ((float)g->M / denom);

        for (int m = 0; m < g->M; m++) {
            Complex aproj = spatial_complex_mul(a[m][f], proj);
            u[m][f] = spatial_complex_sub(X[m][f], aproj);
        }
    }

#else

    /* ---------- Blocking matrix B = I - aa ---------- */
    /*
     * Original full blocking-matrix baseline.
     *
     * Keep this path for A/B testing against the optimized projection form.
     */
    for (int f = 0; f < g->F; f++) {
        float denom = 0.0f;

        for (int m = 0; m < g->M; m++) {
            denom += a[m][f].r * a[m][f].r + a[m][f].i * a[m][f].i;
        }

        if (denom < 1e-12f) {
            denom = 1e-12f;
        }

        for (int i = 0; i < g->M; i++) {
            for (int j = 0; j < g->M; j++) {
                Complex aa_ij = spatial_complex_mul(a[i][f], spatial_complex_conj(a[j][f]));
                aa_ij = spatial_complex_div_real(aa_ij, denom);

                B[f][i][j].r = (i == j ? 1.0f : 0.0f) - aa_ij.r;
                B[f][i][j].i = -aa_ij.i;
            }
        }
    }

    /* ---------- u = conj(B)^T x ---------- */
    for (int m = 0; m < g->M; m++) {
        for (int f = 0; f < g->F; f++) {
            u[m][f].r = 0.0f;
            u[m][f].i = 0.0f;

            for (int n = 0; n < g->M; n++) {
                Complex term = spatial_complex_mul(spatial_complex_conj(B[f][n][m]), X[n][f]);
                u[m][f] = spatial_complex_add(u[m][f], term);
            }
        }
    }

#endif

    /* ---------- wu = conj(wa)^T u ---------- */
    {
        const Complex* u_channels[g->M];
        for (int m = 0; m < g->M; ++m) u_channels[m] = u[m];
        spatial_conj_beamform(
            (const Complex* const*)g->wa,
            u_channels, g->M, g->F, 1.0f, wu);
    }

    /* ---------- gsc = das - wu ---------- */
    for (int f = 0; f < g->F; f++) {
        gsc_spec[f] = spatial_complex_sub(das[f], wu[f]);
        gsc_out[f] = gsc_spec[f];
    }

    /*
     * Export the response represented by the CURRENT wa, before the RLS
     * update below mutates it:
     *
     * y = a^H x/M - wa^H (x - a(a^H x)/denom)
     */
    if (effective_weights) {
        for (int f = 0; f < g->F; ++f) {
            float denom = 0.0f;
            Complex beta = {0.0f, 0.0f};
            for (int m = 0; m < g->M; ++m) {
                Complex wa_h_a =
                    spatial_complex_mul(spatial_complex_conj(g->wa[m][f]), a[m][f]);
                beta = spatial_complex_add(beta, wa_h_a);
                denom += a[m][f].r * a[m][f].r +
                         a[m][f].i * a[m][f].i;
            }
            if (denom < 1e-12f) denom = 1e-12f;
            beta = spatial_complex_div_real(beta, denom);
            for (int m = 0; m < g->M; ++m) {
                Complex a_h = spatial_complex_conj(a[m][f]);
                Complex weight =
                    spatial_complex_add(spatial_complex_div_real(a_h, (float)g->M),
                          spatial_complex_mul(beta, a_h));
                weight = spatial_complex_sub(weight, spatial_complex_conj(g->wa[m][f]));
                effective_weights[m * g->F + f] = weight;
            }
        }
    }

    /*
     * Adaptive update interval:
     * GSC output is still computed every frame using the current weights.
     * Only the RLS weight update is skipped on non-update frames.
     *
     * Example:
     *   adapt_interval = 2
     *   frame 0: update RLS
     *   frame 1: skip RLS
     *   frame 2: update RLS
     */
    int do_adapt_this_frame = allow_adapt;

    if (do_adapt_this_frame && g->adapt_interval > 1) {
        if ((g->frame_idx % g->adapt_interval) != 0) {
            do_adapt_this_frame = 0;
        }
    }

    if (do_adapt_this_frame) {

        /*
         * Per-bin in-place RLS update.
         *
         * Previous implementation used full-frame temporary buffers:
         *   pu[F][M]
         *   gain[F][M]
         *   P_new[F][M][M]
         *   wa_new[M][F]
         *
         * This version updates one frequency bin at a time.
         * It keeps the same RLS math, but removes full-frame P_new/wa_new
         * initialization and copy-back overhead.
         */
        for (int f = 0; f < g->F; f++) {

            /*
             * Early mask skip:
             * mask[f] == 1 means speech bin. For speech bins, adaptive
             * update should be frozen, so skip RLS computation directly.
             */
            if (use_mask_freeze && mask && mask[f]) {
                continue;
            }

            Complex pu[g->M];
            Complex gain[g->M];
            Complex q[g->M];

            /* ---------- pu = P * u ---------- */
            for (int i = 0; i < g->M; i++) {
                pu[i].r = 0.0f;
                pu[i].i = 0.0f;

                for (int k = 0; k < g->M; k++) {
                    Complex term = spatial_complex_mul(g->P[f][i][k], u[k][f]);
                    pu[i] = spatial_complex_add(pu[i], term);
                }
            }

            /* ---------- denominator = lambda + u^H P u ---------- */
            Complex upu_c;
            upu_c.r = g->lambda;
            upu_c.i = 0.0f;

            for (int i = 0; i < g->M; i++) {
                Complex term = spatial_complex_mul(spatial_complex_conj(u[i][f]), pu[i]);
                upu_c = spatial_complex_add(upu_c, term);
            }

            if (use_notebook_update) {
                upu_c.r += g->lambda;
            }

            float upu_real = upu_c.r;

            /* upu_real = lambda + u^H P u is only ever valid (finite,
             * positive) when P is still a well-conditioned positive-
             * semidefinite matrix; lambda > 0 and u^H P u >= 0 for a true
             * PSD P, so a non-finite or non-positive result here means P
             * (computed from LAST frame, before this frame's update) has
             * already drifted -- see GSC_P_DIAG_FLOOR/CEIL's comment above
             * for why. Recover by resetting only this bin (not the whole
             * GSC, which would discard every other bin's adaptation) and
             * skipping its update this frame; the floor/ceiling clamp below
             * is the proactive guard that should make this branch rare. */
            if (!isfinite(upu_real) || upu_real <= 0.0f) {
                gsc_reset_bin(g, f);
                continue;
            }

            if (fabsf(upu_real) < 1e-12f) {
                upu_real = (upu_real >= 0.0f) ? 1e-12f : -1e-12f;
            }

            float inv_upu = 1.0f / upu_real;
            float inv_lambda = 1.0f / g->lambda;

            /* ---------- gain = pu / denominator ---------- */
            for (int i = 0; i < g->M; i++) {
                gain[i].r = pu[i].r * inv_upu;
                gain[i].i = pu[i].i * inv_upu;
            }

            /* ---------- q = u^H * P ---------- */
            for (int j = 0; j < g->M; j++) {
                q[j].r = 0.0f;
                q[j].i = 0.0f;

                for (int k = 0; k < g->M; k++) {
                    Complex term = spatial_complex_mul(spatial_complex_conj(u[k][f]), g->P[f][k][j]);
                    q[j] = spatial_complex_add(q[j], term);
                }
            }

            /* ---------- P = (P - gain * q) / lambda ---------- */
            for (int i = 0; i < g->M; i++) {
                for (int j = 0; j < g->M; j++) {
                    Complex term = spatial_complex_mul(gain[i], q[j]);

                    g->P[f][i][j].r =
                        (g->P[f][i][j].r - term.r) * inv_lambda;

                    g->P[f][i][j].i =
                        (g->P[f][i][j].i - term.i) * inv_lambda;
                }
            }

            /* ---------- restore Hermitian symmetry ----------
             * P is supposed to stay Hermitian (P[j][i] == conj(P[i][j])) by
             * construction, but gain (= P*u) and q (= u^H*P) accumulate
             * independent float32 rounding on each side of the rank-1
             * downdate above, so the two off-diagonal entries of a pair can
             * drift apart over many hops. Average them back onto the
             * Hermitian manifold and force the (real-valued) diagonal's
             * imaginary part to exactly zero. This is a numerical-hardening
             * addition, not part of the original supplied algorithm -- see
             * third_party/README.md. */
            for (int i = 0; i < g->M; i++) {
                for (int j = i + 1; j < g->M; j++) {
                    Complex avg;
                    avg.r = 0.5f * (g->P[f][i][j].r + g->P[f][j][i].r);
                    avg.i = 0.5f * (g->P[f][i][j].i - g->P[f][j][i].i);
                    g->P[f][i][j] = avg;
                    g->P[f][j][i] = spatial_complex_conj(avg);
                }
                g->P[f][i][i].i = 0.0f;
            }

            /* ---------- diagonal loading: clamp P's diagonal ----------
             * Proactive counterpart to the isfinite(upu_real) guard above:
             * runs every adapted frame, on every bin, so P's diagonal never
             * gets close enough to inf for that reactive check to be the
             * only thing standing between a quiet stretch of audio and a
             * corrupted spectrum. See GSC_P_DIAG_FLOOR/CEIL's comment. */
            for (int i = 0; i < g->M; i++) {
                if (g->P[f][i][i].r < GSC_P_DIAG_FLOOR) {
                    g->P[f][i][i].r = GSC_P_DIAG_FLOOR;
                } else if (g->P[f][i][i].r > GSC_P_DIAG_CEIL) {
                    g->P[f][i][i].r = GSC_P_DIAG_CEIL;
                }
            }

            /* ---------- wa update: wa = leak*wa + mu * gain * conj(gsc) ---------- */
            Complex gsc_conj = spatial_complex_conj(gsc_spec[f]);

            for (int m = 0; m < g->M; m++) {
                Complex update = spatial_complex_mul(gain[m], gsc_conj);

                if (use_mu_scaling) {
                    update.r *= g->mu;
                    update.i *= g->mu;
                }

                g->wa[m][f] = spatial_complex_add(spatial_complex_scale(g->wa[m][f], GSC_WA_LEAK), update);
            }

            /* ---------- final per-bin finite check ----------
             * Defense in depth: catches corruption from any source other
             * than P's own diagonal (e.g. a non-finite gsc_spec[f] feeding
             * gsc_conj above) that the upu_real guard and diagonal clamp
             * would not see, since both only look at P. Same per-bin
             * reset+skip recovery -- this frame's wa update for this bin
             * is discarded in favour of a clean slate. */
            for (int m = 0; m < g->M; m++) {
                if (!isfinite(g->wa[m][f].r) || !isfinite(g->wa[m][f].i)) {
                    gsc_reset_bin(g, f);
                    break;
                }
            }
        }

        g->adaptive = 1;
    } else {
        g->adaptive = 0;
    }

    g->frame_idx += 1;
}

void gsc_process(GSC* g,
                 const Complex* const* X,
                 float doa_s,
                 int allow_adapt_in,
                 const int* mask,
                 Complex* gsc_out)
{
    gsc_process_with_weights(
        g, X, doa_s, allow_adapt_in, mask, gsc_out, NULL);
}

void gsc_reset(GSC* g)
{
    if (!g) return;
    for (int f = 0; f < g->F; ++f) {
        for (int i = 0; i < g->M; ++i) {
            for (int j = 0; j < g->M; ++j) {
                g->P[f][i][j].r = i == j ? 1.0f : 0.0f;
                g->P[f][i][j].i = 0.0f;
            }
        }
    }
    for (int m = 0; m < g->M; ++m) {
        memset(g->wa[m], 0, (size_t)g->F * sizeof(Complex));
    }
    g->initialized = 0;
    g->first_doa_found = 0;
    g->first_doa_frame = -1;
    g->current_doa = 0.0f;
    g->frame_idx = 0;
    g->doa_used = NAN;
    g->adaptive = 0;
}

void gsc_destroy(GSC* g)
{
    if (!g) return;

    /* Pool path (owned_heap == NULL, gsc_init()): the caller owns the
     * memory (struct + every array carved from it) -- nothing to free
     * here. Heap path (gsc_create()): a single posix_memalign()'d block
     * backs everything, so one free() tears it all down. */
    if (g->owned_heap) free(g->owned_heap);
}

float gsc_get_doa_used(const GSC* g)
{
    return g ? g->doa_used : NAN;
}

int gsc_get_adaptive(const GSC* g)
{
    return g ? g->adaptive : 0;
}

long gsc_get_bin_resets(const GSC* g)
{
    return g ? g->bin_resets : 0;
}

float gsc_p_diag_floor(void)
{
    return GSC_P_DIAG_FLOOR;
}

float gsc_p_diag_ceil(void)
{
    return GSC_P_DIAG_CEIL;
}

float gsc_wa_leak_factor(void)
{
    return GSC_WA_LEAK;
}
