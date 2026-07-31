#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gsc.h"
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

/* ===================== create ===================== */

GSC* gsc_create(int M, int F, int num_angles,
                Complex*** a_array,
                const GSC_Config* cfg)
{
    GSC* g;
    if (!cfg || !a_array || M <= 0 || F <= 0 || num_angles <= 0 ||
        !isfinite(cfg->lambda) || cfg->lambda <= 0.0f ||
        cfg->lambda > 1.0f ||
        !isfinite(cfg->mu)) {
        return NULL;
    }
    g = (GSC*)calloc(1, sizeof(GSC));
    if (!g) return NULL;

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

    g->adapt_interval = cfg->adapt_interval;
    if (g->adapt_interval <= 0) {
        g->adapt_interval = 1;
    }

    /* For fixed notebook-alignment experiments, keep the original
     * frame-by-frame adaptive update behavior for easier baseline matching. */
    if (g->enable_fix_mode && g->fixed_align_notebook) {
        g->adapt_interval = 1;
    }

    // allocate P (F,M,M)
    g->P = (Complex***)calloc(F, sizeof(Complex**));
    if (!g->P) {
        gsc_destroy(g);
        return NULL;
    }
    for (int f = 0; f < F; f++) {
        g->P[f] = (Complex**)calloc(M, sizeof(Complex*));
        if (!g->P[f]) {
            gsc_destroy(g);
            return NULL;
        }
        for (int i = 0; i < M; i++) {
            g->P[f][i] =
                (Complex*)malloc(M * sizeof(Complex));
            if (!g->P[f][i]) {
                gsc_destroy(g);
                return NULL;
            }
            for (int j = 0; j < M; j++) {
                g->P[f][i][j].r = (i == j);
                g->P[f][i][j].i = 0;
            }
        }
    }

    // wa (M,F)
    g->wa = (Complex**)calloc(M, sizeof(Complex*));
    if (!g->wa) {
        gsc_destroy(g);
        return NULL;
    }
    for (int m = 0; m < M; m++) {
        g->wa[m] = (Complex*)calloc(F, sizeof(Complex));
        if (!g->wa[m]) {
            gsc_destroy(g);
            return NULL;
        }
    }

    {
        size_t count = (size_t)(M + 3) * (size_t)F;
#if !GSC_USE_PROJECTION_BLOCKING
        count += (size_t)F * (size_t)M * (size_t)M;
#endif
        g->scratch = (Complex*)calloc(count, sizeof(Complex));
        if (!g->scratch) {
            gsc_destroy(g);
            return NULL;
        }
        g->scratch_das = g->scratch;
        g->scratch_wu = g->scratch_das + F;
        g->scratch_spec = g->scratch_wu + F;
        g->scratch_u = g->scratch_spec + F;
#if !GSC_USE_PROJECTION_BLOCKING
        g->scratch_b = g->scratch_u + (size_t)M * F;
#endif
    }

    g->initialized = 0;
    g->first_doa_found = 0;
    g->first_doa_frame = -1;
    g->current_doa = 0;
    g->frame_idx = 0;
    g->doa_used = NAN;
    g->adaptive = 0;

    return g;
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
                              Complex** X,
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
        }

        g->adaptive = 1;
    } else {
        g->adaptive = 0;
    }

    g->frame_idx += 1;
}

void gsc_process(GSC* g,
                 Complex** X,
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

    if (g->P) {
        for (int f = 0; f < g->F; f++) {
            if (g->P[f]) {
                for (int i = 0; i < g->M; i++) {
                    free(g->P[f][i]);
                }
                free(g->P[f]);
            }
        }
        free(g->P);
    }

    if (g->wa) {
        for (int m = 0; m < g->M; m++) {
            free(g->wa[m]);
        }
        free(g->wa);
    }

    free(g->scratch);

    free(g);
}

float gsc_get_doa_used(const GSC* g)
{
    return g ? g->doa_used : NAN;
}

int gsc_get_adaptive(const GSC* g)
{
    return g ? g->adaptive : 0;
}

float gsc_wa_leak_factor(void)
{
    return GSC_WA_LEAK;
}
