#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "srp.h"
#include "steering.h"
#include "../utility/complex.h"
#include "../utility/spatial_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * SRP-PHAT pair mode switch
 * 0: original full-pair baseline
 * 1: optimized unique-pair version with precomputed pair steering
 */
#ifndef SRP_USE_UNIQUE_PAIRS
#define SRP_USE_UNIQUE_PAIRS 1
#endif

#if SRP_USE_UNIQUE_PAIRS
static void srp_free_pair_precompute(SRP* s)
{
    if (!s) return;

    if (s->pair_steer) {
        for (int a = 0; a < s->num_angles; a++) {
            if (s->pair_steer[a]) {
                for (int p = 0; p < s->num_pairs; p++) {
                    free(s->pair_steer[a][p]);
                }
                free(s->pair_steer[a]);
            }
        }
        free(s->pair_steer);
    }

    free(s->pair_i);
    free(s->pair_j);
    if (s->pair_phat) {
        for (int p = 0; p < s->num_pairs; ++p) {
            free(s->pair_phat[p]);
        }
        free(s->pair_phat);
    }

    s->pair_steer = NULL;
    s->pair_i = NULL;
    s->pair_j = NULL;
    s->pair_phat = NULL;
    s->num_pairs = 0;
}

static int srp_init_pair_precompute(SRP* s)
{
    if (!s || !s->a_array || s->M <= 1 || s->F <= 0 || s->num_angles <= 0) {
        return -1;
    }

    s->num_pairs = s->M * (s->M - 1) / 2;

    s->pair_i = (int*)malloc(s->num_pairs * sizeof(int));
    s->pair_j = (int*)malloc(s->num_pairs * sizeof(int));
    if (!s->pair_i || !s->pair_j) {
        srp_free_pair_precompute(s);
        return -1;
    }

    int p = 0;
    for (int i = 0; i < s->M; i++) {
        for (int j = i + 1; j < s->M; j++) {
            s->pair_i[p] = i;
            s->pair_j[p] = j;
            p++;
        }
    }

    s->pair_steer = (Complex***)calloc(s->num_angles, sizeof(Complex**));
    if (!s->pair_steer) {
        srp_free_pair_precompute(s);
        return -1;
    }

    for (int a = 0; a < s->num_angles; a++) {
        s->pair_steer[a] = (Complex**)calloc(s->num_pairs, sizeof(Complex*));
        if (!s->pair_steer[a]) {
            srp_free_pair_precompute(s);
            return -1;
        }

        for (int pp = 0; pp < s->num_pairs; pp++) {
            s->pair_steer[a][pp] = (Complex*)malloc(s->F * sizeof(Complex));
            if (!s->pair_steer[a][pp]) {
                srp_free_pair_precompute(s);
                return -1;
            }

            int i = s->pair_i[pp];
            int j = s->pair_j[pp];

            for (int f = 0; f < s->F; f++) {
                /* pair_steer[a][p][f] = conj(w_i) * w_j */
                s->pair_steer[a][pp][f] = spatial_complex_mul(spatial_complex_conj(s->a_array[a][i][f]),
                                                s->a_array[a][j][f]);
            }
        }
    }

    s->pair_phat =
        (Complex**)calloc(s->num_pairs, sizeof(Complex*));
    if (!s->pair_phat) {
        srp_free_pair_precompute(s);
        return -1;
    }
    for (int pp = 0; pp < s->num_pairs; ++pp) {
        s->pair_phat[pp] =
            (Complex*)malloc(s->F * sizeof(Complex));
        if (!s->pair_phat[pp]) {
            srp_free_pair_precompute(s);
            return -1;
        }
    }

    return 0;
}
#endif

SRP* srp_create(
    const SRP_Config* cfg,
    float* angles,
    Complex*** a_array
)
{
    SRP* s;
    if (!cfg || !angles || !a_array || cfg->M <= 1 || cfg->F <= 1 ||
        cfg->num_angles <= 0 || !isfinite(cfg->sr) || cfg->sr <= 0.0f ||
        !isfinite(cfg->NFFT) || cfg->NFFT <= 0.0f ||
        !isfinite(cfg->low_freq) || !isfinite(cfg->high_freq)) {
        return NULL;
    }
    s = (SRP*)calloc(1, sizeof(SRP));
    if (!s) return NULL;

    s->M = cfg->M;
    s->F = cfg->F;
    s->num_angles = cfg->num_angles;
    s->angles = angles;
    s->a_array = a_array;
    s->S_theta = (float*)malloc(cfg->num_angles * sizeof(float));
    s->bin_best_idx = (int*)malloc(cfg->F * sizeof(int));
    s->score_scratch = (float*)malloc(cfg->F * sizeof(float));
    s->best_score = (float*)malloc(cfg->F * sizeof(float));

    s->update_interval = cfg->update_interval;
    if (s->update_interval <= 0) {
        s->update_interval = 1;
    }

    s->frame_counter = 0;
    s->last_doa_raw = NAN;
    s->last_doa_s = NAN;

    s->num_pairs = 0;
    s->pair_i = NULL;
    s->pair_j = NULL;
    s->pair_steer = NULL;
    s->pair_phat = NULL;

    s->doa_raw = NAN;
    s->doa_s   = NAN;

    if (!s->S_theta || !s->bin_best_idx ||
        !s->score_scratch || !s->best_score) {
        if (s->S_theta) free(s->S_theta);
        if (s->bin_best_idx) free(s->bin_best_idx);
        free(s->score_scratch);
        free(s->best_score);
        free(s);
        return NULL;
    }

    for (int f = 0; f < cfg->F; f++) {
        s->bin_best_idx[f] = -1;
    }

    /*smoothing related state*/
    s->enable_smoothing = cfg->enable_smoothing;

    s->smoother.switch_consec = cfg->switch_consec;
    s->smoother.angle_tol = cfg->angle_tol;
    s->smoother.null_value = NAN;
    s->smoother.last = NAN;
    s->smoother.pending = NAN;
    s->smoother.cnt = 0;
    s->smoother.initialized = 0;

    int f_start = (int)roundf(cfg->low_freq * cfg->NFFT / cfg->sr);
    int f_end   = (int)roundf(cfg->high_freq * cfg->NFFT / cfg->sr);

    if (f_start < 1) f_start = 1;
    if (f_end > cfg->F - 1) f_end = cfg->F - 1;
    if (f_end < f_start) f_end = f_start;

    s->f_start = f_start;
    s->f_end   = f_end;

#if SRP_USE_UNIQUE_PAIRS
    if (srp_init_pair_precompute(s) != 0) {
        srp_free_pair_precompute(s);
        free(s->S_theta);
        free(s->bin_best_idx);
        free(s->score_scratch);
        free(s->best_score);
        free(s);
        return NULL;
    }
#endif

    return s;
}

int srp_angle_to_index(SRP* s, float doa_rad)
{
    float deg, step;
    int idx;

    if (!s || isnan(doa_rad)) return -1;

    deg = doa_rad * 180.0f / (float)M_PI;
    while (deg < 0.0f) deg += 360.0f;
    while (deg >= 360.0f) deg -= 360.0f;

    step = 360.0f / (float)s->num_angles;
    idx = (int)lroundf(deg / step) % s->num_angles;

    return idx;
}
SRP* srp_create_from_geometry(
    const SRP_Config* cfg,
    const ArrayGeometry* geom
)
{
    float* angles = srp_create_uniform_angles(cfg->num_angles);
    Complex*** a_array;
    SRP* result;

    if (!angles) return NULL;
    a_array = srp_build_steering(cfg, geom, angles);
    if (!a_array) {
        free(angles);
        return NULL;
    }
    result = srp_create(cfg, angles, a_array);
    if (!result) {
        srp_destroy_steering(a_array, cfg->num_angles, cfg->M);
        free(angles);
    }
    return result;
}

void srp(SRP* s, Complex** X, const int* mask)
{
    int M = s->M;
    int F = s->F;
    int A = s->num_angles;

#if SRP_USE_UNIQUE_PAIRS
    (void)M;  /* M is only used directly in the full-pair fallback path. */
#endif

    /* init frame-level SRP score */
    for (int a = 0; a < A; a++) {
        s->S_theta[a] = 0.0f;
    }

    /* init per-bin best angle */
    for (int f = 0; f < F; f++) {
        s->bin_best_idx[f] = -1;
        s->best_score[f] = -1e30f;
    }

#if SRP_USE_UNIQUE_PAIRS
    /*
     * The PHAT cross spectrum is independent of candidate angle.  The
     * imported implementation recomputed it A times.  Materializing it once
     * keeps the score accumulation order and steering math unchanged while
     * removing that redundant work; spatial_phat_cross() supplies the
     * AArch64 NEON hot path.
     */
    for (int p = 0; p < s->num_pairs; ++p) {
        spatial_phat_cross(
            X[s->pair_i[p]], X[s->pair_j[p]], s->pair_phat[p], F);
    }
#endif

#if SRP_USE_UNIQUE_PAIRS
    /* Candidate-major traversal exposes contiguous frequency bins to NEON.
     * Per-angle accumulation still visits f in ascending order, and each
     * bin still visits p/a in ascending order, preserving scalar results. */
    for (int a = 0; a < A; ++a) {
        memset(s->score_scratch, 0, (size_t)F * sizeof(float));
        for (int p = 0; p < s->num_pairs; ++p) {
            spatial_pair_score_accumulate(
                s->pair_phat[p] + s->f_start,
                s->pair_steer[a][p] + s->f_start,
                s->score_scratch + s->f_start,
                s->f_end - s->f_start + 1);
        }
        for (int f = s->f_start; f <= s->f_end; ++f) {
            float score;
            if (mask && !mask[f]) continue;
            score = s->score_scratch[f];
            s->S_theta[a] += score;
            if (score > s->best_score[f]) {
                s->best_score[f] = score;
                s->bin_best_idx[f] = a;
            }
        }
    }
#else
    for (int f = s->f_start; f <= s->f_end; f++) {
        if (mask && !mask[f]) continue;
        for (int a = 0; a < A; a++) {
            float score = 0.0f;
            /*
             * Original full-pair SRP-PHAT baseline.
             * Keep this path for correctness comparison and debugging.
             */
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    Complex Rij = spatial_complex_mul(X[i][f], spatial_complex_conj(X[j][f]));

                    float mag = spatial_complex_abs(Rij) + 1e-8f;
                    Rij = spatial_complex_div_real(Rij, mag);

                    Complex w_i = s->a_array[a][i][f];
                    Complex w_j = s->a_array[a][j][f];

                    Complex tmp = spatial_complex_mul(spatial_complex_conj(w_i), Rij);
                    Complex val = spatial_complex_mul(tmp, w_j);

                    score += val.r;
                }
            }
            s->S_theta[a] += score;
            if (score > s->best_score[f]) {
                s->best_score[f] = score;
                s->bin_best_idx[f] = a;
            }
        }
    }
#endif
}

float srp2doa(SRP* s)
{
    int best = 0;
    float max_val = s->S_theta[0];

    for (int i = 1; i < s->num_angles; i++) {
        if (s->S_theta[i] > max_val) {
            max_val = s->S_theta[i];
            best = i;
        }
    }

    return s->angles[best];
}

void doa_step(SRP* s,
              Complex** X,
              const int* mask,
              int vad_raw,
              int vad_out)
{
    if (!s || !X) return;

    int run_srp = 0;

    /*
     * Only update SRP when:
     *   1. raw VAD says speech exists
     *   2. current frame matches update interval
     *
     * Example:
     *   update_interval = 3
     *   frame 0: run SRP
     *   frame 1: reuse last DOA
     *   frame 2: reuse last DOA
     *   frame 3: run SRP
     */
    if (vad_raw) {
        if ((s->frame_counter % s->update_interval) == 0) {
            run_srp = 1;
        }
    }

    if (run_srp) {

        srp(s, X, mask);

        /*
         * doa_raw: fresh SRP-PHAT result only.
         * It is not smoothed and not held.
         */
        s->doa_raw = srp2doa(s);
        s->last_doa_raw = s->doa_raw;

        /*
         * doa_s: pending-smoothed DOA with hold.
         * This is the stable DOA intended for downstream modules
         * such as GSC and post-gain.
         */
        float doa_s_new;

        if (s->enable_smoothing) {
            doa_s_new = doa_smoother_update(&s->smoother,
                                            s->doa_raw,
                                            vad_out);
        } else {
            doa_s_new = s->doa_raw;
        }

        if (!isnan(doa_s_new)) {
            s->doa_s = doa_s_new;
            s->last_doa_s = doa_s_new;
        } else {
            /*
             * If smoother has no valid output this frame, keep the
             * last valid smooth DOA. This prevents doa_s from being
             * overwritten by NAN while preserving doa_raw as fresh-only.
             */
            s->doa_s = s->last_doa_s;
        }

    } else {

        /*
         * No SRP update in this frame.
         *
         * Important:
         * Do NOT call doa_smoother_update() with NAN here,
         * otherwise the smoother pending state may be reset.
         */
        s->doa_raw = NAN;
        s->doa_s   = s->last_doa_s;
    }

    s->frame_counter++;
}

float doa_get_raw(const SRP* s)
{
    return s ? s->doa_raw : NAN;
}

float doa_get_smooth(const SRP* s)
{
    return s ? s->doa_s : NAN;
}

void srp_hold(SRP* s)
{
    if (!s) return;
    s->doa_raw = NAN;
    s->doa_s = s->last_doa_s;
}

void srp_reset(SRP* s)
{
    if (!s) return;
    s->frame_counter = 0;
    s->last_doa_raw = NAN;
    s->last_doa_s = NAN;
    s->doa_raw = NAN;
    s->doa_s = NAN;
    s->smoother.last = NAN;
    s->smoother.pending = NAN;
    s->smoother.cnt = 0;
    s->smoother.initialized = 0;
    if (s->S_theta) {
        memset(s->S_theta, 0, (size_t)s->num_angles * sizeof(float));
    }
    if (s->bin_best_idx) {
        for (int f = 0; f < s->F; ++f) s->bin_best_idx[f] = -1;
    }
}

void srp_destroy(SRP* s)
{
    if (!s) return;

    srp_destroy_steering(s->a_array, s->num_angles, s->M);

#if SRP_USE_UNIQUE_PAIRS
    srp_free_pair_precompute(s);
#endif

    free(s->angles);
    free(s->S_theta);
    free(s->bin_best_idx);
    free(s->score_scratch);
    free(s->best_score);
    free(s);
}
