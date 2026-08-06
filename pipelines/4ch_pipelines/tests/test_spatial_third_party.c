/**
 * tests/test_spatial_third_party.c — spatial dependency equivalence tests.
 *
 * Keeps the third-party SRP/GSC arithmetic outside the 4AEC wrapper tests:
 * dispatch must match scalar PHAT, cached SRP must select the scalar golden
 * angle, and exported GSC weights must reconstruct the mono spectrum.
 */

#include <math.h>
#include <stdint.h>
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

    srp(srp_handle, (const Complex* const*)channels, NULL);
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

/* srp_get_mem_size()/srp_init() pool-first pair (Phase A.1): a caller-owned
 * pool larger than the queried size must have every byte beyond
 * srp_get_mem_size(cfg) left untouched, an undersized or misaligned pool
 * must be rejected outright, and the same pool must be reusable for a
 * second srp_init() after srp_destroy() releases the first instance (a
 * no-op on the pool path -- srp_destroy() only frees owned_heap). Also
 * exercises srp_get_mem_size()'s config validation directly (M<=1, F<=1,
 * num_angles<=0 -- the union of every check the old srp_create() used to
 * make). */
static int test_srp_init_pool_poison_and_bounds(void) {
    enum { EXTRA = 32 };
    SRP_Config cfg;
    ArrayGeometry* geometry;
    size_t need;
    unsigned char* pool;
    size_t pool_bytes;
    SRP* s;
    size_t i;

    memset(&cfg, 0, sizeof(cfg));
    cfg.M = 4;
    cfg.F = 65;
    cfg.num_angles = 72;
    cfg.sr = 16000.0f;
    cfg.NFFT = 128.0f;
    cfg.c = 343.0f;
    cfg.low_freq = 300.0f;
    cfg.high_freq = 7000.0f;
    cfg.enable_smoothing = 0;
    cfg.switch_consec = 1;
    cfg.update_interval = 1;

    geometry = array_geometry_create_uca(4, 0.035f);
    CHECK(geometry != NULL, "pool bounds test: create UCA geometry");

    need = srp_get_mem_size(&cfg);
    CHECK(need > 0,
          "srp_get_mem_size reports a positive size for a valid config");
    {
        SRP_Config bad = cfg;
        bad.M = 1;
        CHECK(srp_get_mem_size(&bad) == 0, "srp_get_mem_size rejects M <= 1");
        bad = cfg;
        bad.F = 1;
        CHECK(srp_get_mem_size(&bad) == 0, "srp_get_mem_size rejects F <= 1");
        bad = cfg;
        bad.num_angles = 0;
        CHECK(srp_get_mem_size(&bad) == 0,
              "srp_get_mem_size rejects num_angles <= 0");
        bad = cfg;
        bad.c = 0.0f;
        CHECK(srp_get_mem_size(&bad) == 0, "srp_get_mem_size rejects c <= 0");
    }
    pool_bytes = need + EXTRA;
    CHECK(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
          "pool bounds test: allocate aligned pool");
    memset(pool, 0xa5, pool_bytes);

    /* srp_init() must also reject a geometry/cfg mismatch -- a check
     * srp_get_mem_size() cannot make on its own since it never sees an
     * ArrayGeometry -- and must not touch the pool while rejecting it. */
    {
        ArrayGeometry mismatched = *geometry;
        mismatched.M = cfg.M + 1;
        CHECK(srp_init(pool, pool_bytes, &cfg, &mismatched) == NULL,
              "srp_init rejects a geometry whose M does not match cfg->M");
    }
    for (i = 0; i < pool_bytes; ++i) {
        CHECK(pool[i] == 0xa5,
              "rejected srp_init (geometry mismatch) must not touch the pool");
    }

    /* too-small pool must be rejected, not silently truncated. */
    CHECK(srp_init(pool, need - 1, &cfg, geometry) == NULL,
          "srp_init rejects a pool smaller than srp_get_mem_size()");
    /* misaligned base pointer must be rejected. */
    CHECK(srp_init(pool + 1, pool_bytes - 1, &cfg, geometry) == NULL,
          "srp_init rejects a base pointer that is not 16-byte aligned");
    /* re-poison check: the rejected calls above must not have written
     * anything. */
    for (i = 0; i < pool_bytes; ++i) {
        CHECK(pool[i] == 0xa5, "rejected srp_init calls must not touch the pool");
    }

    s = srp_init(pool, pool_bytes, &cfg, geometry);
    CHECK(s != NULL, "srp_init accepts a correctly sized/aligned pool");
    CHECK((void*)s == (void*)pool, "SRP struct is placed at mem[0]");
    CHECK(s->owned_heap == NULL,
          "pool-path SRP has a NULL owned_heap (nothing for srp_destroy to free)");

    for (i = 0; i < (size_t)EXTRA; ++i) {
        CHECK(pool[need + i] == 0xa5,
              "bytes beyond srp_get_mem_size() are left untouched by srp_init");
    }

    /* sanity: the returned instance is actually usable. */
    {
        Complex* storage = (Complex*)calloc((size_t)cfg.M * (size_t)cfg.F,
                                            sizeof(Complex));
        Complex* channels[4];
        int m;
        CHECK(storage != NULL, "pool bounds test: allocate process inputs");
        for (m = 0; m < cfg.M; ++m) channels[m] = storage + (size_t)m * cfg.F;
        doa_step(s, (const Complex* const*)channels, NULL, /*vad_raw=*/1, /*vad_out=*/1);
        CHECK(isfinite(doa_get_raw(s)) || isnan(doa_get_raw(s)),
              "pool-path SRP doa_step runs without crashing");
        free(storage);
    }

    srp_destroy(s); /* pool path: must be a no-op (owned_heap == NULL) */

    /* pool reuse: the same block must work again for a second instance. */
    memset(pool, 0xa5, pool_bytes);
    s = srp_init(pool, pool_bytes, &cfg, geometry);
    CHECK(s != NULL, "pool is reusable for a second srp_init after srp_destroy");
    srp_destroy(s);

    free(pool);
    array_geometry_destroy(geometry);
    return 1;
}

/* Heap-vs-pool byte-equal test: srp_create_from_geometry() (heap) and
 * srp_init() (caller pool) must produce byte-identical a_array steering
 * vectors and evolve identically across frames with varying VAD, including
 * doa_step()'s raw/smoothed DOA output and the per-bin S_theta/bin_best_idx
 * state srp() populates. Also spot-checks a handful of a_array[a][m]
 * entries directly against srp_build_steering()'s own output --
 * srp_build_steering() (steering.c) is still a standalone heap utility, no
 * longer called by either constructor under test here, so this is an
 * independent cross-check of the pool-carve inline fill formula in
 * srp_init() against the original formula, not the carve step checking
 * itself. This is the single highest-risk place for a silent bug now that
 * the steering formula exists in two separately-maintained places. */
static int test_srp_heap_vs_pool_byte_equal(void) {
    enum { M = 4, F = 65, NUM_ANGLES = 72, FRAMES = 40 };
    SRP_Config cfg;
    ArrayGeometry* geometry;
    SRP* heap_s;
    SRP* pool_s;
    size_t need;
    void* pool = NULL;
    float* angles;
    Complex*** golden_a_array;
    Complex* storage;
    Complex* channels[M];
    int frame;
    int a, m, f;

    memset(&cfg, 0, sizeof(cfg));
    cfg.M = M;
    cfg.F = F;
    cfg.num_angles = NUM_ANGLES;
    cfg.sr = 16000.0f;
    cfg.NFFT = 128.0f;
    cfg.c = 343.0f;
    cfg.low_freq = 300.0f;
    cfg.high_freq = 7000.0f;
    cfg.enable_smoothing = 1;
    cfg.switch_consec = 2;
    cfg.angle_tol = 0.2f;
    cfg.update_interval = 2;

    geometry = array_geometry_create_uca(M, 0.035f);
    CHECK(geometry != NULL, "heap-vs-pool test: create UCA geometry");

    heap_s = srp_create_from_geometry(&cfg, geometry);
    CHECK(heap_s != NULL, "heap-vs-pool test: srp_create_from_geometry");
    CHECK(heap_s->owned_heap != NULL,
          "heap-path SRP records a non-NULL owned_heap");

    need = srp_get_mem_size(&cfg);
    CHECK(need > 0, "heap-vs-pool test: srp_get_mem_size");
    CHECK(posix_memalign(&pool, 16, need) == 0 && pool,
          "heap-vs-pool test: allocate pool");
    pool_s = srp_init(pool, need, &cfg, geometry);
    CHECK(pool_s != NULL, "heap-vs-pool test: srp_init");
    CHECK(pool_s->owned_heap == NULL, "pool-path SRP has a NULL owned_heap");

    angles = srp_create_uniform_angles(NUM_ANGLES);
    CHECK(angles != NULL, "heap-vs-pool test: build golden angles");
    golden_a_array = srp_build_steering(&cfg, geometry, angles);
    CHECK(golden_a_array != NULL, "heap-vs-pool test: build golden steering");

    for (a = 0; a < NUM_ANGLES; a += 7) {
        for (m = 0; m < M; ++m) {
            for (f = 0; f < F; f += 5) {
                CHECK(heap_s->a_array[a][m][f].r ==
                              golden_a_array[a][m][f].r &&
                          heap_s->a_array[a][m][f].i ==
                              golden_a_array[a][m][f].i,
                      "heap SRP a_array matches srp_build_steering golden");
                CHECK(pool_s->a_array[a][m][f].r ==
                              golden_a_array[a][m][f].r &&
                          pool_s->a_array[a][m][f].i ==
                              golden_a_array[a][m][f].i,
                      "pool SRP a_array matches srp_build_steering golden");
            }
        }
    }
    srp_destroy_steering(golden_a_array, NUM_ANGLES, M);
    free(angles);

    storage = (Complex*)malloc((size_t)M * F * sizeof(Complex));
    CHECK(storage != NULL, "heap-vs-pool test: allocate shared inputs");
    for (m = 0; m < M; ++m) channels[m] = storage + (size_t)m * F;

    for (frame = 0; frame < FRAMES; ++frame) {
        int vad_raw = (frame % 3) != 0;
        int vad_out = (frame % 5) != 0;
        float heap_raw, pool_raw;
        float heap_smooth, pool_smooth;

        for (m = 0; m < M; ++m) {
            for (f = 0; f < F; ++f) {
                channels[m][f].r = random_signed();
                channels[m][f].i = random_signed();
            }
        }

        doa_step(heap_s, (const Complex* const*)channels, NULL, vad_raw, vad_out);
        doa_step(pool_s, (const Complex* const*)channels, NULL, vad_raw, vad_out);

        heap_raw = doa_get_raw(heap_s);
        pool_raw = doa_get_raw(pool_s);
        CHECK((isnan(heap_raw) && isnan(pool_raw)) || heap_raw == pool_raw,
              "heap vs pool: doa_get_raw is byte-equal every frame");

        heap_smooth = doa_get_smooth(heap_s);
        pool_smooth = doa_get_smooth(pool_s);
        CHECK((isnan(heap_smooth) && isnan(pool_smooth)) ||
                  heap_smooth == pool_smooth,
              "heap vs pool: doa_get_smooth is byte-equal every frame");

        for (a = 0; a < NUM_ANGLES; ++a) {
            CHECK(heap_s->S_theta[a] == pool_s->S_theta[a],
                  "heap vs pool: S_theta is byte-equal every frame");
        }
        for (f = 0; f < F; ++f) {
            CHECK(heap_s->bin_best_idx[f] == pool_s->bin_best_idx[f],
                  "heap vs pool: bin_best_idx is byte-equal every frame");
        }
    }

    free(storage);
    srp_destroy(pool_s);
    free(pool);
    srp_destroy(heap_s);
    array_geometry_destroy(geometry);
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
            original, (const Complex* const*)channels, 0.7f, frame < 5, NULL, output_original);
        gsc_process_with_weights(
            exported, (const Complex* const*)channels, 0.7f, frame < 5, NULL,
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
            g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);

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

/* Regression test: gsc_create() forces the actual RLS update
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

/* GSC's per-bin P-diagonal clamp (GSC_P_DIAG_FLOOR/CEIL) must actually bound
 * a runaway diagonal, not just exist as dead code. An all-zero spectrum at
 * one bin makes the blocking-output u == 0 there (u = x - a*(a^H x/a^H a)
 * is linear in x), which makes upu_real == lambda (finite, positive)
 * regardless of steering vectors -- so the reactive isfinite/positivity
 * guard does NOT fire, and P_new[i][i] = P[i][i]/lambda (dividing by
 * lambda<1 grows the magnitude further), isolating the proactive clamp as
 * the only thing standing between a poked-in extreme value and an
 * unbounded diagonal. */
static int test_gsc_p_diag_clamp_bounds_runaway_values(void) {
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* g;
    Complex* storage;
    Complex* channels[4];
    Complex output[65];
    const int target_bin = 10;
    const int other_bin = 11;
    int m;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "diag clamp test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "diag clamp test: create GSC");

    storage = (Complex*)calloc(4u * 65u, sizeof(Complex));
    CHECK(storage != NULL, "diag clamp test: allocate inputs");
    for (m = 0; m < 4; ++m) channels[m] = storage + m * 65;
    /* every channel's spectrum at every bin is exactly (0,0) (calloc) ->
     * u[m][f] == 0 for all m, f, regardless of the steering vector. */

    /* --- ceiling: poke every diagonal entry of target_bin far above the
     * ceiling, confirm one adapted hop clamps it back down. --- */
    for (m = 0; m < 4; ++m) g->P[target_bin][m][m].r = 1e12f;
    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
    for (m = 0; m < 4; ++m) {
        CHECK(g->P[target_bin][m][m].r == gsc_p_diag_ceil(),
              "P diagonal above the ceiling is clamped to exactly the ceiling");
    }
    CHECK(gsc_get_bin_resets(g) == 0,
          "the clamp path does not also trigger a bin reset");

    /* --- floor: poke every diagonal entry of target_bin far below the
     * floor (including negative), confirm one adapted hop clamps it back
     * up. --- */
    for (m = 0; m < 4; ++m) g->P[target_bin][m][m].r = -5.0f;
    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
    for (m = 0; m < 4; ++m) {
        CHECK(g->P[target_bin][m][m].r == gsc_p_diag_floor(),
              "P diagonal below the floor is clamped to exactly the floor");
    }
    CHECK(gsc_get_bin_resets(g) == 0,
          "the clamp path does not also trigger a bin reset");

    /* an unrelated, never-poked bin must be untouched by either clamp
     * call -- the guard operates strictly per-bin. */
    CHECK(g->P[other_bin][0][0].r != gsc_p_diag_ceil() &&
              g->P[other_bin][0][0].r != gsc_p_diag_floor(),
          "an unrelated bin's diagonal is not perturbed by another bin's clamp");

    free(storage);
    gsc_destroy(g);
    srp_destroy(srp_handle);
    return 1;
}

/* A NaN planted directly in P (as opposed to a runaway-but-finite diagonal,
 * which the proactive clamp handles) must not silently spread into gain/wa
 * and out through gsc_out -- some reset guard must catch it and recover
 * just that one bin. NaN propagates through multiplication unconditionally
 * (NaN * x == NaN for any x, including 0), so poking a single NaN into P is
 * enough to corrupt this bin's pipeline regardless of the input spectrum's
 * content.
 *
 * Mutation-tested: with the reactive isfinite(upu_real) guard alone
 * disabled, this specific corruption (NaN in P feeding a zero input
 * spectrum) still gets caught -- NaN propagates on into gain and then wa,
 * and the separate "final per-bin finite check" on wa catches it instead.
 * Only disabling BOTH guards makes this test fail. That overlap is
 * consistent with the guards' own comments (the wa check is documented as
 * defense-in-depth for sources other than P), so this test is verifying
 * "the guard system recovers from a non-finite P entry" rather than
 * isolating one specific guard. */
static int test_gsc_bin_reset_on_nonfinite_p_propagation(void) {
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* g;
    Complex* storage;
    Complex* channels[4];
    Complex output[65];
    const int target_bin = 20;
    const int other_bin = 21;
    int m, n;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "bin reset test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "bin reset test: create GSC");

    storage = (Complex*)calloc(4u * 65u, sizeof(Complex));
    CHECK(storage != NULL, "bin reset test: allocate inputs");
    for (m = 0; m < 4; ++m) channels[m] = storage + m * 65;

    /* Corrupt one bin's P (as could happen from accumulated float error)
     * and confirm gsc_reset_bin() recovers it cleanly, without disturbing
     * any other bin. */
    g->P[target_bin][0][0].r = NAN;
    CHECK(gsc_get_bin_resets(g) == 0, "bin reset test: starts at 0 resets");

    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);

    CHECK(gsc_get_bin_resets(g) == 1,
          "a non-finite P entry triggers exactly one bin reset");
    for (m = 0; m < 4; ++m) {
        for (n = 0; n < 4; ++n) {
            float expected = (m == n) ? 1.0f : 0.0f;
            CHECK(g->P[target_bin][m][n].r == expected &&
                      g->P[target_bin][m][n].i == 0.0f,
                  "reset bin's P is restored to the identity");
        }
        CHECK(g->wa[m][target_bin].r == 0.0f && g->wa[m][target_bin].i == 0.0f,
              "reset bin's wa is cleared to zero");
    }
    CHECK(isfinite(g->P[other_bin][0][0].r),
          "an unrelated bin's P is untouched by another bin's reset");

    free(storage);
    gsc_destroy(g);
    srp_destroy(srp_handle);
    return 1;
}

/* gsc_get_mem_size()/gsc_init() pool-first pair (Phase A.2): a caller-owned
 * pool larger than the queried size must have every byte beyond
 * gsc_get_mem_size(M, F) left untouched, an undersized or misaligned pool
 * must be rejected outright, and the same pool must be reusable for a
 * second gsc_init() after gsc_destroy() releases the first instance (a
 * no-op on the pool path -- gsc_destroy() only frees owned_heap). */
static int test_gsc_init_pool_poison_and_bounds(void) {
    enum { M = 4, F = 65, NUM_ANGLES = 72, EXTRA = 32 };
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    size_t need;
    unsigned char* pool;
    size_t pool_bytes;
    GSC* g;
    size_t i;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "pool bounds test: create steering owner");

    need = gsc_get_mem_size(M, F);
    CHECK(need > 0, "gsc_get_mem_size reports a positive size for a valid shape");
    CHECK(gsc_get_mem_size(0, F) == 0, "gsc_get_mem_size rejects M <= 0");
    CHECK(gsc_get_mem_size(M, 0) == 0, "gsc_get_mem_size rejects F <= 0");
    CHECK(gsc_get_mem_size(-1, F) == 0, "gsc_get_mem_size rejects negative M");

    pool_bytes = need + EXTRA;
    CHECK(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
          "pool bounds test: allocate aligned pool");
    memset(pool, 0xa5, pool_bytes);

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;

    /* too-small pool must be rejected, not silently truncated. */
    CHECK(gsc_init(pool, need - 1, M, F, NUM_ANGLES, srp_handle->a_array,
                    &gsc_cfg) == NULL,
          "gsc_init rejects a pool smaller than gsc_get_mem_size()");
    /* misaligned base pointer must be rejected. */
    CHECK(gsc_init(pool + 1, pool_bytes - 1, M, F, NUM_ANGLES,
                    srp_handle->a_array, &gsc_cfg) == NULL,
          "gsc_init rejects a base pointer that is not 16-byte aligned");
    /* re-poison: the rejected calls above must not have written anything. */
    for (i = 0; i < pool_bytes; ++i) {
        CHECK(pool[i] == 0xa5, "rejected gsc_init calls must not touch the pool");
    }

    g = gsc_init(pool, pool_bytes, M, F, NUM_ANGLES,
                 srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "gsc_init accepts a correctly sized/aligned pool");
    CHECK((void*)g == (void*)pool, "GSC struct is placed at mem[0]");
    CHECK(g->owned_heap == NULL,
          "pool-path GSC has a NULL owned_heap (nothing for gsc_destroy to free)");

    for (i = 0; i < (size_t)EXTRA; ++i) {
        CHECK(pool[need + i] == 0xa5,
              "bytes beyond gsc_get_mem_size() are left untouched by gsc_init");
    }

    /* sanity: the returned instance is actually usable. */
    {
        Complex* storage = (Complex*)calloc((size_t)M * (size_t)F, sizeof(Complex));
        Complex* channels[M];
        Complex output[F];
        int m;
        int f;
        CHECK(storage != NULL, "pool bounds test: allocate process inputs");
        for (m = 0; m < M; ++m) channels[m] = storage + (size_t)m * F;
        gsc_process_with_weights(
            g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
        for (f = 0; f < F; ++f) {
            CHECK(isfinite(output[f].r) && isfinite(output[f].i),
                  "pool-path GSC produces finite output");
        }
        free(storage);
    }

    gsc_destroy(g); /* pool path: must be a no-op (owned_heap == NULL) */

    /* pool reuse: the same block must work again for a second instance. */
    memset(pool, 0xa5, pool_bytes);
    g = gsc_init(pool, pool_bytes, M, F, NUM_ANGLES,
                 srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "pool is reusable for a second gsc_init after gsc_destroy");
    gsc_destroy(g);

    free(pool);
    srp_destroy(srp_handle);
    return 1;
}

/* Heap-vs-pool byte-equal test: gsc_create() (heap) and gsc_init() (caller
 * pool) must evolve identically, hop for hop, across frames that both skip
 * AND trigger RLS adaptation -- not just matching gsc_out/effective_weights,
 * but the internal P[f][i][j]/wa[m][f] RLS state too. This is the strongest
 * check available that flattening GSC's nested Complex triple- and
 * double-pointer arrays into one carved pool block does not perturb the RLS
 * recursion's numeric evolution at all versus the original nested
 * calloc/malloc layout. */
static int test_gsc_heap_vs_pool_byte_equal(void) {
    enum { M = 4, F = 65, NUM_ANGLES = 72, HOPS = 40 };
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    size_t need;
    void* pool = NULL;
    GSC* heap_g;
    GSC* pool_g;
    Complex* storage;
    Complex* channels[M];
    Complex heap_out[F];
    Complex pool_out[F];
    Complex heap_weights[M * F];
    Complex pool_weights[M * F];
    int hop;
    int f, i, j, m;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "heap-vs-pool test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 1;

    heap_g = gsc_create(M, F, NUM_ANGLES, srp_handle->a_array, &gsc_cfg);
    CHECK(heap_g != NULL, "heap-vs-pool test: gsc_create");
    CHECK(heap_g->owned_heap != NULL, "heap-path GSC records a non-NULL owned_heap");

    need = gsc_get_mem_size(M, F);
    CHECK(need > 0, "heap-vs-pool test: gsc_get_mem_size");
    CHECK(posix_memalign(&pool, 16, need) == 0 && pool,
          "heap-vs-pool test: allocate pool");
    pool_g = gsc_init(pool, need, M, F, NUM_ANGLES, srp_handle->a_array, &gsc_cfg);
    CHECK(pool_g != NULL, "heap-vs-pool test: gsc_init");
    CHECK(pool_g->owned_heap == NULL, "pool-path GSC has a NULL owned_heap");

    storage = (Complex*)malloc((size_t)M * F * sizeof(Complex));
    CHECK(storage != NULL, "heap-vs-pool test: allocate shared inputs");
    for (m = 0; m < M; ++m) channels[m] = storage + (size_t)m * F;

    for (hop = 0; hop < HOPS; ++hop) {
        /* Alternate allow_adapt so both instances exercise adapted AND
         * skipped-adaptation frames identically. */
        int allow_adapt = (hop % 3) != 0;
        for (m = 0; m < M; ++m) {
            for (f = 0; f < F; ++f) {
                channels[m][f].r = random_signed();
                channels[m][f].i = random_signed();
            }
        }

        gsc_process_with_weights(
            heap_g, (const Complex* const*)channels, 0.7f, allow_adapt, NULL, heap_out, heap_weights);
        gsc_process_with_weights(
            pool_g, (const Complex* const*)channels, 0.7f, allow_adapt, NULL, pool_out, pool_weights);

        CHECK(memcmp(heap_out, pool_out, sizeof(heap_out)) == 0,
              "heap vs pool: gsc_out is byte-equal every hop");
        CHECK(memcmp(heap_weights, pool_weights, sizeof(heap_weights)) == 0,
              "heap vs pool: effective_weights is byte-equal every hop");

        for (f = 0; f < F; ++f) {
            for (i = 0; i < M; ++i) {
                CHECK(heap_g->wa[i][f].r == pool_g->wa[i][f].r &&
                          heap_g->wa[i][f].i == pool_g->wa[i][f].i,
                      "heap vs pool: wa[m][f] state is byte-equal every hop");
                for (j = 0; j < M; ++j) {
                    CHECK(heap_g->P[f][i][j].r == pool_g->P[f][i][j].r &&
                              heap_g->P[f][i][j].i == pool_g->P[f][i][j].i,
                          "heap vs pool: P[f][i][j] state is byte-equal every hop");
                }
            }
        }
    }

    free(storage);
    gsc_destroy(pool_g);
    free(pool);
    gsc_destroy(heap_g);
    srp_destroy(srp_handle);
    return 1;
}

/* frame_idx must not lose precision or wrap once it exceeds what a 32-bit
 * signed int could hold (~2^31, ~199-265 days at typical hop rates) --
 * poke it to just past that old boundary and confirm one hop increments it
 * exactly by one, not silently truncating/wrapping to something smaller or
 * negative. */
static int test_gsc_frame_idx_survives_32bit_boundary(void) {
    SRP_Config srp_cfg;
    GSC_Config gsc_cfg;
    SRP* srp_handle;
    GSC* g;
    Complex* storage;
    Complex* channels[4];
    Complex output[65];
    uint64_t poked;
    int m;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "frame_idx overflow test: create steering owner");

    memset(&gsc_cfg, 0, sizeof(gsc_cfg));
    gsc_cfg.enable = 1;
    gsc_cfg.lambda = 0.995f;
    gsc_cfg.mu = 0.05f;
    gsc_cfg.enable_fix_mode = 1;
    gsc_cfg.fixed_doa_rad = 0.7f;
    gsc_cfg.adapt_interval = 4; /* exercises the modulo-gated cadence path too */
    g = gsc_create(4, 65, 72, srp_handle->a_array, &gsc_cfg);
    CHECK(g != NULL, "frame_idx overflow test: create GSC");

    storage = (Complex*)calloc(4u * 65u, sizeof(Complex));
    CHECK(storage != NULL, "frame_idx overflow test: allocate inputs");
    for (m = 0; m < 4; ++m) channels[m] = storage + m * 65;

    poked = (uint64_t)2147483647u + 5u; /* INT32_MAX + 5: not representable
                                          * as a positive 32-bit signed int */
    g->frame_idx = poked;
    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
    CHECK(g->frame_idx == poked + 1u,
          "frame_idx increments exactly by one past the old 32-bit boundary");

    free(storage);
    gsc_destroy(g);
    srp_destroy(srp_handle);
    return 1;
}

/* SRP's frame_counter has the same overflow class of issue GSC's frame_idx
 * had (see test_gsc_frame_idx_survives_32bit_boundary) -- verify the same
 * fix here. */
static int test_srp_frame_counter_survives_32bit_boundary(void) {
    SRP_Config srp_cfg;
    SRP* srp_handle;
    Complex* storage;
    Complex* channels[4];
    uint64_t poked;
    int m;

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL, "frame_counter overflow test: create SRP");

    storage = (Complex*)calloc(4u * 65u, sizeof(Complex));
    CHECK(storage != NULL, "frame_counter overflow test: allocate inputs");
    for (m = 0; m < 4; ++m) channels[m] = storage + m * 65;

    poked = (uint64_t)2147483647u + 5u;
    srp_handle->frame_counter = poked;
    doa_step(srp_handle, (const Complex* const*)channels, NULL, /*vad_raw=*/1, /*vad_out=*/1);
    CHECK(srp_handle->frame_counter == poked + 1u,
          "frame_counter increments exactly by one past the old 32-bit boundary");

    free(storage);
    srp_destroy(srp_handle);
    return 1;
}

int main(void) {
    CHECK(test_phat_scalar_vs_dispatch(), "PHAT SIMD test");
    CHECK(test_beamform_and_score_scalar_vs_dispatch(),
          "beamform/SRP-score SIMD test");
    CHECK(test_srp_precompute_equivalence(), "SRP optimization test");
    CHECK(test_srp_init_pool_poison_and_bounds(),
          "SRP pool-first poison/bounds test");
    CHECK(test_srp_heap_vs_pool_byte_equal(),
          "SRP heap-vs-pool byte-equal test");
    CHECK(test_gsc_weight_export(), "GSC effective-weight test");
    CHECK(test_gsc_create_rejects_invalid_lambda(),
          "GSC create lambda-bound test");
    CHECK(test_gsc_long_run_hermitian_and_finite(),
          "GSC long-run Hermitian/finite test");
    CHECK(test_gsc_effective_adapt_interval_matches_created_cadence(),
          "GSC effective-adapt-interval/created-cadence match test");
    CHECK(test_gsc_wa_leak_is_bounded(), "GSC wa-leak boundedness test");
    CHECK(test_gsc_p_diag_clamp_bounds_runaway_values(),
          "GSC P-diagonal clamp boundary test");
    CHECK(test_gsc_bin_reset_on_nonfinite_p_propagation(),
          "GSC non-finite-P bin-reset test");
    CHECK(test_gsc_init_pool_poison_and_bounds(),
          "GSC pool-first poison/bounds test");
    CHECK(test_gsc_heap_vs_pool_byte_equal(),
          "GSC heap-vs-pool byte-equal test");
    CHECK(test_gsc_frame_idx_survives_32bit_boundary(),
          "GSC frame_idx 32-bit-boundary test");
    CHECK(test_srp_frame_counter_survives_32bit_boundary(),
          "SRP frame_counter 32-bit-boundary test");
    printf("All third-party spatial tests passed (backend=%s)\n",
           spatial_simd_backend());
    return 0;
}
