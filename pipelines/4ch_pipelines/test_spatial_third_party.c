/**
 * test_spatial_third_party.c — spatial dependency equivalence tests.
 *
 * Keeps the third-party SRP/GSC arithmetic outside the 4AEC wrapper tests:
 * dispatch must match scalar PHAT, cached SRP must select the scalar golden
 * angle, and exported GSC weights must reconstruct the mono spectrum.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gsc.h"
#include "spatial_simd.h"
#include "srp.h"
#include "steering.h"

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 0;                                                  \
        }                                                              \
    } while (0)

static unsigned int test_rng = 0x12345678u;

static float random_signed(void) {
    test_rng = test_rng * 1664525u + 1013904223u;
    return ((float)((test_rng >> 8) & 0x00ffffffu) /
            8388608.0f) - 1.0f;
}

static float complex_error(Complex a, Complex b) {
    return fmaxf(fabsf(a.r - b.r), fabsf(a.i - b.i));
}

/* Shared 4-mic/65-bin UCA SRP setup used by every GSC-facing test below. */
static SRP* make_test_srp(SRP_Config* srp_cfg) {
    ArrayGeometry* geometry;
    SRP* srp_handle;
    memset(srp_cfg, 0, sizeof(*srp_cfg));
    srp_cfg->M = 4;
    srp_cfg->F = 65;
    srp_cfg->num_angles = 72;
    srp_cfg->sr = 16000.0f;
    srp_cfg->NFFT = 128.0f;
    srp_cfg->c = 343.0f;
    srp_cfg->low_freq = 300.0f;
    srp_cfg->high_freq = 7000.0f;
    srp_cfg->enable_smoothing = 0;
    srp_cfg->switch_consec = 1;
    srp_cfg->update_interval = 1;
    geometry = array_geometry_create_uca(4, 0.035f);
    if (!geometry) return NULL;
    srp_handle = srp_create_from_geometry(srp_cfg, geometry);
    array_geometry_destroy(geometry);
    return srp_handle;
}

static int test_phat_scalar_vs_dispatch(void) {
    enum { N = 259 };
    Complex x[N];
    Complex y[N];
    Complex scalar[N];
    Complex dispatched[N];
    float worst = 0.0f;
    for (int i = 0; i < N; ++i) {
        x[i].r = random_signed();
        x[i].i = random_signed();
        y[i].r = random_signed();
        y[i].i = random_signed();
    }
    spatial_phat_cross_scalar(x, y, scalar, N);
    spatial_phat_cross(x, y, dispatched, N);
    for (int i = 0; i < N; ++i) {
        float error = complex_error(scalar[i], dispatched[i]);
        if (error > worst) worst = error;
        CHECK(isfinite(dispatched[i].r) && isfinite(dispatched[i].i),
              "SIMD PHAT output must remain finite");
    }
    CHECK(worst == 0.0f &&
          memcmp(scalar, dispatched, sizeof(scalar)) == 0,
          "NEON PHAT must be bit-identical to the scalar golden");
    return 1;
}

static int test_beamform_and_score_scalar_vs_dispatch(void) {
    enum { M = 4, F = 259 };
    Complex w_storage[M][F];
    Complex x_storage[M][F];
    const Complex* w[M];
    const Complex* x[M];
    Complex beam_scalar[F];
    Complex beam_dispatch[F];
    float score_scalar[F];
    float score_dispatch[F];

    for (int m = 0; m < M; ++m) {
        w[m] = w_storage[m];
        x[m] = x_storage[m];
        for (int f = 0; f < F; ++f) {
            w_storage[m][f].r = random_signed();
            w_storage[m][f].i = random_signed();
            x_storage[m][f].r = random_signed();
            x_storage[m][f].i = random_signed();
        }
    }
    spatial_conj_beamform_scalar(w, x, M, F, 0.25f, beam_scalar);
    spatial_conj_beamform(w, x, M, F, 0.25f, beam_dispatch);
    CHECK(memcmp(beam_scalar, beam_dispatch, sizeof(beam_scalar)) == 0,
          "NEON conjugate beamformer must be bit-identical to scalar");

    memset(score_scalar, 0, sizeof(score_scalar));
    memset(score_dispatch, 0, sizeof(score_dispatch));
    for (int m = 0; m < M; ++m) {
        spatial_pair_score_accumulate_scalar(
            w[m], x[m], score_scalar, F);
        spatial_pair_score_accumulate(
            w[m], x[m], score_dispatch, F);
    }
    CHECK(memcmp(score_scalar, score_dispatch, sizeof(score_scalar)) == 0,
          "NEON SRP score kernel must be bit-identical to scalar");
    return 1;
}

static int test_srp_precompute_equivalence(void) {
    SRP_Config cfg;
    ArrayGeometry* geometry;
    SRP* srp_handle;
    Complex* storage;
    Complex* channels[4];
    float* golden;
    int golden_best = 0;

    memset(&cfg, 0, sizeof(cfg));
    cfg.M = 4;
    cfg.F = 65;
    cfg.num_angles = 72;
    cfg.sr = 16000.0f;
    cfg.NFFT = 128.0f;
    cfg.c = 343.0f;
    cfg.low_freq = 300.0f;
    cfg.high_freq = 7000.0f;
    cfg.enable_smoothing = 1;
    cfg.switch_consec = 2;
    cfg.angle_tol = 0.2f;
    cfg.update_interval = 1;

    geometry = array_geometry_create_uca(4, 0.035f);
    CHECK(geometry != NULL, "create UCA geometry");
    srp_handle = srp_create_from_geometry(&cfg, geometry);
    array_geometry_destroy(geometry);
    CHECK(srp_handle != NULL, "create SRP");
    storage = (Complex*)malloc(
        (size_t)cfg.M * cfg.F * sizeof(Complex));
    golden = (float*)calloc((size_t)cfg.num_angles, sizeof(float));
    CHECK(storage && golden, "allocate SRP test buffers");
    for (int m = 0; m < cfg.M; ++m) {
        channels[m] = storage + (size_t)m * cfg.F;
        for (int f = 0; f < cfg.F; ++f) {
            channels[m][f].r = random_signed();
            channels[m][f].i = random_signed();
        }
    }

    srp(srp_handle, channels, NULL);
    for (int p = 0; p < srp_handle->num_pairs; ++p) {
        Complex phat[65];
        spatial_phat_cross_scalar(
            channels[srp_handle->pair_i[p]],
            channels[srp_handle->pair_j[p]], phat, cfg.F);
        for (int f = srp_handle->f_start;
             f <= srp_handle->f_end; ++f) {
            CHECK(complex_error(
                      phat[f], srp_handle->pair_phat[p][f]) == 0.0f,
                  "SRP cached PHAT differs from scalar golden");
            for (int a = 0; a < cfg.num_angles; ++a) {
                Complex steer =
                    srp_handle->pair_steer[a][p][f];
                float real =
                    phat[f].r * steer.r - phat[f].i * steer.i;
                golden[a] += 2.0f * real;
            }
        }
    }
    for (int a = 1; a < cfg.num_angles; ++a) {
        if (golden[a] > golden[golden_best]) golden_best = a;
    }
    CHECK(srp_angle_to_index(
              srp_handle, srp2doa(srp_handle)) == golden_best,
          "optimized SRP must choose scalar golden angle");

    free(golden);
    free(storage);
    srp_destroy(srp_handle);
    return 1;
}

static int reconstruct_matches(const GSC* g,
                               Complex** x,
                               const Complex* weights,
                               const Complex* output) {
    for (int f = 0; f < g->F; ++f) {
        Complex value = {0.0f, 0.0f};
        for (int m = 0; m < g->M; ++m) {
            Complex w = weights[m * g->F + f];
            Complex term;
            term.r = w.r * x[m][f].r - w.i * x[m][f].i;
            term.i = w.r * x[m][f].i + w.i * x[m][f].r;
            value.r += term.r;
            value.i += term.i;
        }
        {
            float scale = 1.0f + fabsf(output[f].r) + fabsf(output[f].i);
            if (complex_error(value, output[f]) > 3e-5f * scale) return 0;
        }
    }
    return 1;
}

static int test_gsc_weight_export(void) {
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* original;
    GSC* exported;
    Complex* storage;
    Complex* channels[4];
    Complex output_original[65];
    Complex output_exported[65];
    Complex weights[4 * 65];

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;
    original = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    exported = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(original && exported, "create paired GSC instances");

    storage = (Complex*)malloc(4u * 65u * sizeof(Complex));
    CHECK(storage != NULL, "allocate GSC inputs");
    for (int m = 0; m < 4; ++m) channels[m] = storage + m * 65;
    for (int frame = 0; frame < 8; ++frame) {
        for (int m = 0; m < 4; ++m) {
            for (int f = 0; f < 65; ++f) {
                channels[m][f].r = random_signed();
                channels[m][f].i = random_signed();
            }
        }
        gsc_process(
            original, channels, 0.7f, frame < 5, NULL, output_original);
        gsc_process_with_weights(
            exported, channels, 0.7f, frame < 5, NULL,
            output_exported, weights);
        for (int f = 0; f < 65; ++f) {
            CHECK(complex_error(output_original[f], output_exported[f]) == 0.0f,
                  "weight export must not change original GSC output");
        }
        CHECK(reconstruct_matches(
                  exported, channels, weights, output_exported),
              "exported GSC weights must reconstruct its mono spectrum");
    }

    free(storage);
    gsc_destroy(exported);
    gsc_destroy(original);
    srp_destroy(srp_handle);
    return 1;
}

/* gsc_create() must reject an invalid forgetting factor on its own, not only
 * via the 4ch_pipelines wrapper's validate_config() -- a caller using GSC/gsc.h
 * directly (bypassing the wrapper) must get the same protection. */
static int test_gsc_create_rejects_invalid_lambda(void) {
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* g;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "lambda test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.adapt_interval = 1;

    gsc_cfg.lambda = 0.0f;
    CHECK(gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg) == NULL,
          "gsc_create rejects lambda == 0");
    gsc_cfg.lambda = -0.5f;
    CHECK(gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg) == NULL,
          "gsc_create rejects a negative lambda");
    gsc_cfg.lambda = NAN;
    CHECK(gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg) == NULL,
          "gsc_create rejects a NaN lambda");
    gsc_cfg.lambda = 1.0f + 1e-6f;
    CHECK(gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg) == NULL,
          "gsc_create rejects lambda above 1.0");

    gsc_cfg.lambda = 1.0f;
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "gsc_create accepts the lambda == 1.0 boundary");
    gsc_destroy(g);

    gsc_cfg.lambda = 0.995f;
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "gsc_create accepts a normal lambda");
    gsc_destroy(g);

    srp_destroy(srp_handle);
    return 1;
}

/* Long-run integration check: over many hops of an active (allow_adapt=1),
 * unmasked RLS update, g->P must stay exactly Hermitian (each off-diagonal
 * pair equal to its own conjugate, each diagonal's imaginary part exactly
 * zero) and every P/wa/output value must stay finite. A fixed (not
 * re-randomized per hop) input is used deliberately so the RLS state
 * settles into a steady operating regime instead of just reacting to fresh
 * noise every hop. */
static int test_gsc_long_run_hermitian_and_finite(void) {
    enum { HOPS = 4000 };
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* g;
    Complex* storage;
    Complex* channels[4];
    Complex output[65];
    int hop;
    int f;
    int i;
    int j;
    int m;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "long-run test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "long-run test: create GSC");

    storage = (Complex*)malloc(4u * 65u * sizeof(Complex));
    CHECK(storage != NULL, "long-run test: allocate inputs");
    for (m = 0; m < 4; ++m) channels[m] = storage + m * 65;
    for (m = 0; m < 4; ++m) {
        for (f = 0; f < 65; ++f) {
            channels[m][f].r = random_signed();
            channels[m][f].i = random_signed();
        }
    }

    for (hop = 0; hop < HOPS; ++hop) {
        gsc_process_with_weights(
            g, channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);

        for (f = 0; f < 65; ++f) {
            CHECK(isfinite(output[f].r) && isfinite(output[f].i),
                  "long-run test: GSC output stays finite");
            for (i = 0; i < 4; ++i) {
                CHECK(isfinite(g->P[f][i][i].r) && g->P[f][i][i].i == 0.0f,
                      "long-run test: P diagonal is real and finite");
                for (j = i + 1; j < 4; ++j) {
                    CHECK(isfinite(g->P[f][i][j].r) &&
                              isfinite(g->P[f][i][j].i),
                          "long-run test: P off-diagonal stays finite");
                    CHECK(g->P[f][i][j].r == g->P[f][j][i].r &&
                              g->P[f][i][j].i == -g->P[f][j][i].i,
                          "long-run test: P stays exactly Hermitian");
                }
            }
            for (m = 0; m < 4; ++m) {
                CHECK(isfinite(g->wa[m][f].r) && isfinite(g->wa[m][f].i),
                      "long-run test: wa stays finite");
            }
        }
    }

    free(storage);
    gsc_destroy(g);
    srp_destroy(srp_handle);
    return 1;
}

/* Regression test (Codex review): gsc_create() forces the actual RLS update
 * cadence (g->adapt_interval) to 1 whenever enable_fix_mode &&
 * fixed_align_notebook, regardless of the caller's requested adapt_interval
 * -- kept for baseline-matching against the reference notebook. Any external
 * computation that assumes a particular cadence (e.g. audio_pipeline_4ch.c
 * retiming lambda for a slower wall-clock update period) MUST derive it via
 * the same gsc_effective_adapt_interval() gsc_create() uses internally
 * instead of re-deriving the forcing rule, so the two can never silently
 * diverge again. This test creates real GSC instances across representative
 * (enable_fix_mode, fixed_align_notebook, adapt_interval) combinations and
 * confirms gsc_create()'s actual resulting cadence always matches what
 * gsc_effective_adapt_interval() predicts for those same inputs. */
static int test_gsc_effective_adapt_interval_matches_created_cadence(void) {
    static const struct {
        int enable_fix_mode;
        int fixed_align_notebook;
        int requested_adapt_interval;
    } cases[] = {
        {0, 0, 1}, /* auto mode, default cadence */
        {0, 0, 4}, /* auto mode, slower cadence: no forcing */
        {1, 0, 4}, /* fixed mode WITHOUT notebook alignment: no forcing */
        {1, 1, 1}, /* fixed-notebook mode, already 1: forcing is a no-op */
        {1, 1, 4}, /* fixed-notebook mode: MUST be forced down to 1 */
        {1, 1, 8},
    };
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    size_t i;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL,
          "effective-interval test: create steering owner");

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int expected;
        GSC* g;

        memset(&gsc_cfg, 0, sizeof(gsc_cfg));
        gsc_cfg.enable = 1;
        gsc_cfg.lambda = 0.995f;
        gsc_cfg.mu = 0.05f;
        gsc_cfg.enable_fix_mode = cases[i].enable_fix_mode;
        gsc_cfg.fixed_doa_rad = 0.7f;
        gsc_cfg.fixed_align_notebook = cases[i].fixed_align_notebook;
        gsc_cfg.adapt_interval = cases[i].requested_adapt_interval;

        expected = gsc_effective_adapt_interval(
            cases[i].enable_fix_mode, cases[i].fixed_align_notebook,
            cases[i].requested_adapt_interval);

        g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
        CHECK(g != NULL, "effective-interval test: create GSC");
        CHECK(g->adapt_interval == expected,
              "gsc_create's actual cadence matches "
              "gsc_effective_adapt_interval()");
        gsc_destroy(g);
    }

    srp_destroy(srp_handle);
    return 1;
}

/* GSC_WA_LEAK (queried via gsc_wa_leak_factor(), not duplicated as a
 * literal) must be a genuine forgetting factor in (0,1): a leaky
 * accumulator `wa = leak*wa + update` under constant forcing converges to
 * the closed-form steady state `update/(1-leak)`, whereas a plain
 * accumulator (leak == 1) would instead grow without bound. This proves the
 * constant actually has the bounding property the P/wa hardening relies on,
 * independent of how large a real deployment's own update count ever gets
 * (see gsc.c's GSC_WA_LEAK comment for why that count is hops-gated by VAD/
 * masking, not one-per-hop). */
static int test_gsc_wa_leak_is_bounded(void) {
    float leak = gsc_wa_leak_factor();
    double wa = 0.0;
    const double update = 0.37;
    /* Run enough updates to be ~20 leak time constants (1/(1-leak)) deep,
     * regardless of the actual leak value, so leak^n is astronomically
     * small (tight convergence to steady state) and the naive/leaky ratio
     * (== n*(1-leak)) is always ~20 -- this makes the test self-scaling if
     * GSC_WA_LEAK is ever retuned, instead of depending on today's specific
     * 0.99999 to happen to produce a good ratio at a fixed n. */
    const long n = (long)(20.0 / (1.0 - (double)leak)) + 1000L;
    long k;
    double expected_steady_state;
    double naive_unleaked;

    CHECK(leak > 0.0f && leak < 1.0f,
          "gsc_wa_leak_factor must be a valid forgetting factor in (0,1)");

    for (k = 0; k < n; ++k) {
        wa = (double)leak * wa + update;
    }
    expected_steady_state = update / (1.0 - (double)leak);
    naive_unleaked = update * (double)n;

    CHECK(fabs(wa - expected_steady_state) <
              1e-6 * expected_steady_state,
          "leaky accumulator converges to update/(1-leak)");
    CHECK(wa < naive_unleaked / 10.0,
          "leaky accumulator stays far below unleaked accumulation over "
          "the same update count");
    return 1;
}

int main(void) {
    CHECK(test_phat_scalar_vs_dispatch(), "PHAT SIMD test");
    CHECK(test_beamform_and_score_scalar_vs_dispatch(),
          "beamform/SRP-score SIMD test");
    CHECK(test_srp_precompute_equivalence(), "SRP optimization test");
    CHECK(test_gsc_weight_export(), "GSC effective-weight test");
    CHECK(test_gsc_create_rejects_invalid_lambda(),
          "GSC create lambda-bound test");
    CHECK(test_gsc_long_run_hermitian_and_finite(),
          "GSC long-run Hermitian/finite test");
    CHECK(test_gsc_effective_adapt_interval_matches_created_cadence(),
          "GSC effective-adapt-interval/created-cadence match test");
    CHECK(test_gsc_wa_leak_is_bounded(), "GSC wa-leak boundedness test");
    printf("All third-party spatial tests passed (backend=%s)\n",
           spatial_simd_backend());
    return 0;
}
