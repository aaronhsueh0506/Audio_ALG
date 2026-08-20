/**
 * tests/test_spatial_third_party.c — equivalence tests for the modules in
 * this directory.
 *
 * Keeps this arithmetic outside the 4AEC wrapper tests: dispatch must match
 * scalar PHAT, cached SRP must select the scalar golden angle, exported GSC
 * weights must reconstruct the mono spectrum, the VAD's caller-pool path must
 * agree with its heap path bin for bin, and the gain modules must still
 * produce the bytes their pre-kernel formulations did.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gsc.h"
#include "gsc_test_hooks.h"
#include "spatial_simd.h"
#include "srp.h"
#include "steering.h"
#include "vad_api.h"
#include "fix_gain.h"
#include "nr_gain.h"
#include "post_gain.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

static int test_gsc_vector_kernels_scalar_vs_dispatch(void) {
    enum { M = 4, F = 259 };
    Complex a_storage[M][F];
    Complex x_storage[M][F];
    Complex wa_storage[M][F];
    const Complex* a[M];
    const Complex* x[M];
    const Complex* wa[M];
    Complex das[F];
    Complex rhs[F];
    Complex projection_scalar[M][F];
    Complex projection_dispatch[M][F];
    Complex weights_scalar[M][F];
    Complex weights_dispatch[M][F];
    Complex sub_scalar[F];
    Complex sub_dispatch[F];

    for (int m = 0; m < M; ++m) {
        a[m] = a_storage[m];
        x[m] = x_storage[m];
        wa[m] = wa_storage[m];
        for (int f = 0; f < F; ++f) {
            a_storage[m][f].r = random_signed();
            a_storage[m][f].i = random_signed();
            x_storage[m][f].r = random_signed();
            x_storage[m][f].i = random_signed();
            wa_storage[m][f].r = random_signed();
            wa_storage[m][f].i = random_signed();
        }
        /* Exercise the denominator floor in both the vector body and the
         * scalar tail, not only the ordinary non-zero steering path. */
        a_storage[m][0].r = 0.0f;
        a_storage[m][0].i = 0.0f;
        a_storage[m][F - 1].r = 0.0f;
        a_storage[m][F - 1].i = 0.0f;
    }
    spatial_conj_beamform(a, x, M, F, 1.0f / (float)M, das);
    for (int f = 0; f < F; ++f) {
        rhs[f].r = random_signed();
        rhs[f].i = random_signed();
    }

    spatial_gsc_projection_scalar(
        a, x, das, M, F, &projection_scalar[0][0]);
    spatial_gsc_projection(
        a, x, das, M, F, &projection_dispatch[0][0]);
    CHECK(memcmp(projection_scalar, projection_dispatch,
                 sizeof(projection_scalar)) == 0,
          "GSC projection dispatch must be bit-identical to scalar");

    spatial_gsc_effective_weights_scalar(
        a, wa, M, F, &weights_scalar[0][0]);
    spatial_gsc_effective_weights(
        a, wa, M, F, &weights_dispatch[0][0]);
    CHECK(memcmp(weights_scalar, weights_dispatch,
                 sizeof(weights_scalar)) == 0,
          "GSC effective-weight dispatch must be bit-identical to scalar");

    spatial_complex_sub_array_scalar(das, rhs, sub_scalar, F);
    spatial_complex_sub_array(das, rhs, sub_dispatch, F);
    CHECK(memcmp(sub_scalar, sub_dispatch, sizeof(sub_scalar)) == 0,
          "complex subtract dispatch must be bit-identical to scalar");
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
 * via the 4ch_aec_bf_nr_res wrapper's validate_config() -- a caller using GSC/gsc.h
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
                CHECK(isfinite(g->P[i][i][f].r) && g->P[i][i][f].i == 0.0f,
                      "long-run test: P diagonal is real and finite");
                for (j = i + 1; j < 4; ++j) {
                    CHECK(isfinite(g->P[i][j][f].r) &&
                              isfinite(g->P[i][j][f].i),
                          "long-run test: P off-diagonal stays finite");
                    CHECK(g->P[i][j][f].r == g->P[j][i][f].r &&
                              g->P[i][j][f].i == -g->P[j][i][f].i,
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
    for (m = 0; m < 4; ++m) g->P[m][m][target_bin].r = 1e12f;
    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
    for (m = 0; m < 4; ++m) {
        CHECK(g->P[m][m][target_bin].r == gsc_p_diag_ceil(),
              "P diagonal above the ceiling is clamped to exactly the ceiling");
    }
    CHECK(gsc_get_bin_resets(g) == 0,
          "the clamp path does not also trigger a bin reset");

    /* --- floor: poke every diagonal entry of target_bin far below the
     * floor (including negative), confirm one adapted hop clamps it back
     * up. --- */
    for (m = 0; m < 4; ++m) g->P[m][m][target_bin].r = -5.0f;
    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);
    for (m = 0; m < 4; ++m) {
        CHECK(g->P[m][m][target_bin].r == gsc_p_diag_floor(),
              "P diagonal below the floor is clamped to exactly the floor");
    }
    CHECK(gsc_get_bin_resets(g) == 0,
          "the clamp path does not also trigger a bin reset");

    /* an unrelated, never-poked bin must be untouched by either clamp
     * call -- the guard operates strictly per-bin. */
    CHECK(g->P[0][0][other_bin].r != gsc_p_diag_ceil() &&
              g->P[0][0][other_bin].r != gsc_p_diag_floor(),
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
    g->P[0][0][target_bin].r = NAN;
    CHECK(gsc_get_bin_resets(g) == 0, "bin reset test: starts at 0 resets");

    gsc_process_with_weights(
        g, (const Complex* const*)channels, 0.7f, /*allow_adapt_in=*/1, NULL, output, NULL);

    CHECK(gsc_get_bin_resets(g) == 1,
          "a non-finite P entry triggers exactly one bin reset");
    for (m = 0; m < 4; ++m) {
        for (n = 0; n < 4; ++n) {
            float expected = (m == n) ? 1.0f : 0.0f;
            CHECK(g->P[m][n][target_bin].r == expected &&
                      g->P[m][n][target_bin].i == 0.0f,
                  "reset bin's P is restored to the identity");
        }
        CHECK(g->wa[m][target_bin].r == 0.0f && g->wa[m][target_bin].i == 0.0f,
              "reset bin's wa is cleared to zero");
    }
    CHECK(isfinite(g->P[0][0][other_bin].r),
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
 * but the internal P[i][j][f]/wa[m][f] RLS state too. This is the strongest
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
                    CHECK(heap_g->P[i][j][f].r == pool_g->P[i][j][f].r &&
                              heap_g->P[i][j][f].i == pool_g->P[i][j][f].i,
                          "heap vs pool: P[i][j][f] state is byte-equal every hop");
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

/* Production dispatch vs the scalar RLS oracle, compared after every hop.
 * This catches a one-bit difference before recursive P/wa state can hide its
 * origin. It covers four-bin vectors, the scalar tail, mixed mask groups,
 * skipped adaptation, reset, and exceptional per-bin recovery. */
static int run_gsc_rls_dispatch_matches_scalar_state(int notebook_mode) {
    enum { M = 4, F = 65, NUM_ANGLES = 72, HOPS = 500 };
    SRP_Config srp_cfg;
    GSC_Config cfg;
    SRP* srp_handle;
    GSC* dispatched;
    GSC* scalar;
    Complex* storage;
    Complex* channels[M];
    Complex dispatched_out[F];
    Complex scalar_out[F];
    Complex dispatched_weights[M * F];
    Complex scalar_weights[M * F];
    int mask[F];

    srp_handle = make_test_srp(&srp_cfg);
    CHECK(srp_handle != NULL,
          "RLS dispatch parity: create steering owner");
    memset(&cfg, 0, sizeof(cfg));
    cfg.enable = 1;
    cfg.lambda = 0.995f;
    cfg.mu = 0.05f;
    cfg.enable_fix_mode = 1;
    cfg.fixed_doa_rad = 0.7f;
    cfg.fixed_align_notebook = notebook_mode;
    cfg.adapt_interval = notebook_mode ? 5 : 2;
    dispatched = gsc_create(M, F, NUM_ANGLES, srp_handle->a_array, &cfg);
    scalar = gsc_create(M, F, NUM_ANGLES, srp_handle->a_array, &cfg);
    CHECK(dispatched && scalar,
          "RLS dispatch parity: create both GSC instances");
    storage = (Complex*)malloc((size_t)M * F * sizeof(Complex));
    CHECK(storage != NULL, "RLS dispatch parity: allocate inputs");
    for (int m = 0; m < M; ++m) channels[m] = storage + (size_t)m * F;

    for (int hop = 0; hop < HOPS; ++hop) {
        int allow_adapt = (hop % 5) != 0;
        for (int m = 0; m < M; ++m) {
            for (int f = 0; f < F; ++f) {
                channels[m][f].r = random_signed();
                channels[m][f].i = random_signed();
            }
        }
        for (int f = 0; f < F; ++f) {
            mask[f] = (hop % 7 == 0) && ((hop + f) % 11 == 0);
        }
        if (hop == 173) {
            /* Force the vector group's pre-store exceptional fallback. */
            dispatched->P[0][0][4].r = NAN;
            scalar->P[0][0][4].r = NAN;
        }
        if (hop == 251) {
            gsc_reset(dispatched);
            gsc_reset(scalar);
        }

        gsc_process_with_weights(
            dispatched, (const Complex* const*)channels, 0.7f,
            allow_adapt, mask, dispatched_out, dispatched_weights);
        gsc_test_process_with_weights_scalar_rls(
            scalar, (const Complex* const*)channels, 0.7f,
            allow_adapt, mask, scalar_out, scalar_weights);

        CHECK(memcmp(dispatched_out, scalar_out,
                     sizeof(dispatched_out)) == 0,
              "RLS dispatch parity: output is byte-identical every hop");
        CHECK(memcmp(dispatched_weights, scalar_weights,
                     sizeof(dispatched_weights)) == 0,
              "RLS dispatch parity: effective weights are byte-identical");
        CHECK(memcmp(dispatched->P[0][0], scalar->P[0][0],
                     (size_t)M * M * F * sizeof(Complex)) == 0,
              "RLS dispatch parity: recursive P state is byte-identical");
        CHECK(memcmp(dispatched->wa[0], scalar->wa[0],
                     (size_t)M * F * sizeof(Complex)) == 0,
              "RLS dispatch parity: recursive wa state is byte-identical");
        CHECK(dispatched->bin_resets == scalar->bin_resets &&
                  dispatched->frame_idx == scalar->frame_idx &&
                  dispatched->adaptive == scalar->adaptive,
              "RLS dispatch parity: control state is identical");
    }

    free(storage);
    gsc_destroy(scalar);
    gsc_destroy(dispatched);
    srp_destroy(srp_handle);
    return 1;
}

static int test_gsc_rls_dispatch_matches_scalar_state(void) {
    CHECK(run_gsc_rls_dispatch_matches_scalar_state(0),
          "RLS dispatch parity: production update policy");
    CHECK(run_gsc_rls_dispatch_matches_scalar_state(1),
          "RLS dispatch parity: fixed-notebook update policy");
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


/* ===================== VAD (masker + mask VAD) ===================== */

static VADApiConfig vad_test_config(int nfft) {
    VADApiConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.backend = VAD_BACKEND_MASKER;
    cfg.masker_cfg.NFFT = nfft;
    cfg.masker_cfg.sr = 16000;
    cfg.masker_cfg.E_alpha_up = 0.999f;
    cfg.masker_cfg.E_alpha_down = 0.97f;
    cfg.masker_cfg.margin_dB = 6.0f;
    cfg.masker_cfg.low_freq = 200.0f;
    cfg.masker_cfg.high_freq = 8000.0f;
    cfg.masker_cfg.M_alpha = 0.995f;
    cfg.masker_cfg.spp_thr = 0.5f;
    cfg.masker_cfg.spp_upd_thr = 0.3f;
    cfg.masker_cfg.enable_freq_smooth = 1;
    cfg.masker_cfg.smooth_size = 5;
    cfg.masker_cfg.enable_time_smooth = 1;
    cfg.masker_cfg.T_alpha = 0.6f;
    cfg.masker_cfg.enable_energy = 1;
    cfg.masker_cfg.enable_spp = 1;
    cfg.masker_cfg.enable_band = 1;
    cfg.vad_cfg.F = nfft / 2 + 1;
    cfg.vad_cfg.mode = 1;
    cfg.vad_cfg.mask_thr = 0.2f;
    cfg.vad_cfg.min_bins = 3;
    cfg.vad_cfg.enable_median = 1;
    cfg.vad_cfg.median_k = 5;
    cfg.vad_cfg.enable_smooth = 1;
    cfg.vad_cfg.hangover = 4;
    return cfg;
}

/* ---------- the caller-pool contract, checked once ----------
 *
 * Every module in this directory implements the same one: an undersized or
 * misaligned block is refused without a single byte written, a legal block
 * puts the instance at mem[0] and is never written past `need`, and destroy()
 * leaves the block reusable. Four copies of this script had accumulated (VAD
 * plus the three gain stages); one copy means a new probe added here covers
 * all of them instead of three of four.
 *
 * The SRP and GSC copies further up are deliberately left alone: they check a
 * strictly larger set (they re-poison and re-check between the rejected
 * calls), so folding them in would either weaken them or complicate this. */
typedef void* (*PoolInitFn)(void* mem, size_t bytes, void* ctx);
typedef void (*PoolDestroyFn)(void* inst);

/* CHECK_L rather than CHECK: one shared body serves four modules, so a bare
 * "init on a legal pool" would not say which one failed. */
#define CHECK_L(condition, message)                                        \
    do {                                                                   \
        if (!(condition)) {                                                \
            fprintf(stderr, "FAIL: %s: %s (line %d)\n",                    \
                    label, message, __LINE__);                             \
            return 0;                                                      \
        }                                                                  \
    } while (0)

static int check_pool_contract(const char* label, size_t need,
                               PoolInitFn init, PoolDestroyFn destroy,
                               void* ctx) {
    size_t slack = 256;
    size_t pool_bytes = need + slack;
    uint8_t* pool = NULL;
    void* inst;
    size_t i;

    CHECK_L(need > 0, "get_mem_size must size a legal config");
    CHECK_L(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
            "pool allocation");
    memset(pool, 0xa5, pool_bytes);

    CHECK_L(init(pool, need - 1, ctx) == NULL,
            "undersized pool must be refused");
    CHECK_L(init(pool + 1, pool_bytes - 1, ctx) == NULL,
            "misaligned pool must be refused");
    /* A refusal must write NOTHING -- the whole pool is still poison. */
    for (i = 0; i < pool_bytes; ++i) {
        CHECK_L(pool[i] == 0xa5u, "a refused init wrote into the pool");
    }

    inst = init(pool, pool_bytes, ctx);
    CHECK_L(inst != NULL, "init on a legal pool");
    CHECK_L(inst == (void*)pool, "the instance must sit at mem[0]");

    /* Only the budgeted region may be touched: everything past `need` must
     * still hold the poison. */
    for (i = need; i < pool_bytes; ++i) {
        CHECK_L(pool[i] == 0xA5u, "init wrote past its own budget");
    }

    destroy(inst);   /* must NOT free caller memory */
    memset(pool, 0x5a, 1);   /* still ours to write */
    /* The same block must take a second init -- destroy left nothing behind. */
    CHECK_L(init(pool, pool_bytes, ctx) != NULL,
            "the pool must be reusable after destroy");
    free(pool);
    printf("PASS %s pool-first poison/bounds\n", label);
    return 1;
}

static void* pool_init_vad(void* mem, size_t bytes, void* ctx) {
    return vad_api_init((const VADApiConfig*)ctx, mem, bytes);
}
static void pool_destroy_vad(void* inst) { vad_api_destroy((VADApi*)inst); }

static void* pool_init_fix_gain(void* mem, size_t bytes, void* ctx) {
    return fix_gain_init((const FixGainConfig*)ctx, mem, bytes);
}
static void pool_destroy_fix_gain(void* inst) {
    fix_gain_destroy((FixGain*)inst);
}

static void* pool_init_nr_gain(void* mem, size_t bytes, void* ctx) {
    (void)ctx;   /* the instance is one float; no config affects its size */
    return nr_gain_init(mem, bytes);
}
static void pool_destroy_nr_gain(void* inst) { nr_gain_destroy((NrGain*)inst); }

static void* pool_init_post_gain(void* mem, size_t bytes, void* ctx) {
    return post_gain_init((const PostGainConfig*)ctx, mem, bytes);
}
static void pool_destroy_post_gain(void* inst) {
    post_gain_destroy((PostGainState*)inst);
}

static int test_vad_api_init_pool_poison_and_bounds(void) {
    VADApiConfig cfg = vad_test_config(512);
    return check_pool_contract("VAD", vad_api_get_mem_size(&cfg),
                               pool_init_vad, pool_destroy_vad, &cfg);
}

static int test_vad_api_heap_vs_pool_identical(void) {
    const int nfft = 512;
    const int F = nfft / 2 + 1;
    const int frames = 40;
    VADApiConfig cfg = vad_test_config(nfft);
    size_t need = vad_api_get_mem_size(&cfg);
    uint8_t* raw = (uint8_t*)malloc(need + 16);
    uint8_t* pool;
    VADApi* heap;
    VADApi* fromPool;
    Complex* frame = (Complex*)malloc((size_t)F * sizeof(Complex));
    int t, f;

    CHECK(raw && frame, "allocation");
    pool = (uint8_t*)(((uintptr_t)raw + 15u) & ~(uintptr_t)15u);

    heap = vad_api_create(&cfg);
    fromPool = vad_api_init(&cfg, pool, need);
    CHECK(heap && fromPool, "both constructors must succeed");

    test_rng = 0x2468acef;
    for (t = 0; t < frames; ++t) {
        const int* mask_h;
        const int* mask_p;
        for (f = 0; f < F; ++f) {
            frame[f].r = random_signed() * (f < F / 4 ? 4.0f : 0.25f);
            frame[f].i = random_signed() * (f < F / 4 ? 4.0f : 0.25f);
        }
        vad_api_process(heap, frame);
        vad_api_process(fromPool, frame);

        CHECK(vad_api_get(heap) == vad_api_get(fromPool),
              "heap and pool VAD decisions must agree");
        CHECK(vad_api_get_raw(heap) == vad_api_get_raw(fromPool),
              "heap and pool raw VAD must agree");
        mask_h = vad_api_get_mask(heap);
        mask_p = vad_api_get_mask(fromPool);
        CHECK(mask_h && mask_p, "both masks must exist");
        CHECK(memcmp(mask_h, mask_p, (size_t)F * sizeof(int)) == 0,
              "heap and pool TF masks must be byte-identical");
    }

    /* ⚠ Written so it can FAIL: if the driver never made the VAD fire, the
     * comparison above would be trivially satisfied by two silent instances. */
    {
        int ones = 0;
        const int* mask = vad_api_get_mask(heap);
        for (f = 0; f < F; ++f) ones += mask[f];
        CHECK(ones > 0, "the driver never activated any mask bin -- "
                        "the equivalence above would prove nothing");
    }

    vad_api_destroy(heap);
    vad_api_destroy(fromPool);
    free(frame);
    free(raw);
    printf("PASS VAD heap-vs-pool identical over %d frames\n", frames);
    return 1;
}

/* ===================== gain library ===================== */

/* Verbatim transcription of fix_gain_process()'s fused scale-and-clip loop as
 * it stood before the clip moved to sk_clip_f32. The module must still produce
 * these exact bytes. */
static void fix_gain_fused_reference(float* x, int n, float gain,
                                     int do_clip, float lim) {
    int i;
    for (i = 0; i < n; i++) {
        float y = x[i] * gain;
        if (do_clip) {
            if (y > lim) y = lim;
            else if (y < -lim) y = -lim;
        }
        x[i] = y;
    }
}

static FixGainConfig fix_gain_test_config(const float* per_channel) {
    FixGainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.channels = 4;
    cfg.enable = 1;
    cfg.global_gain = 2.5f;
    cfg.channel_gain = per_channel;
    cfg.enable_clip = 1;
    cfg.clip_value = 0.75f;
    return cfg;
}

static int test_fix_gain_pool_poison_and_bounds(void) {
    const float per_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };
    FixGainConfig cfg = fix_gain_test_config(per_channel);
    float bad_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };

    CHECK(check_pool_contract("fix_gain", fix_gain_get_mem_size(&cfg),
                              pool_init_fix_gain, pool_destroy_fix_gain, &cfg),
          "fix_gain pool contract");

    /* get_mem_size doubles as the config gate: a config it will not construct
     * must size to 0 rather than produce an instance that turns the signal
     * into NaN. */
    {
        FixGainConfig bad = cfg;
        bad.channels = 0;
        CHECK(fix_gain_get_mem_size(&bad) == 0, "channels == 0 must not size");
        CHECK(fix_gain_create(&bad) == NULL, "channels == 0 must not construct");

        bad = cfg;
        bad.global_gain = (float)NAN;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite global gain must not size");
        bad.global_gain = (float)INFINITY;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "an infinite global gain must not size");

        bad = cfg;
        bad.clip_value = (float)NAN;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite clip value must not size");

        bad = cfg;
        bad_channel[2] = (float)INFINITY;
        bad.channel_gain = bad_channel;
        CHECK(fix_gain_get_mem_size(&bad) == 0,
              "a non-finite per-channel gain must not size");
        CHECK(fix_gain_create(&bad) == NULL,
              "a non-finite per-channel gain must not construct");
    }

    /* fix_gain_db_to_linear does not validate; the config gate above is what
     * catches what it can produce. */
    CHECK(fix_gain_db_to_linear(0.0f) == 1.0f, "0 dB is unity");
    CHECK(!isfinite(fix_gain_db_to_linear((float)INFINITY)),
          "an infinite dB value converts to a non-finite gain");

    printf("PASS fix_gain config gate rejects non-finite tuning\n");
    return 1;
}

static int test_fix_gain_block_and_per_sample_match_reference(void) {
    const int n = 257;                  /* not a multiple of 4: kernel tail */
    const float per_channel[4] = { 1.0f, 0.5f, 2.0f, 0.125f };
    FixGainConfig cfg = fix_gain_test_config(per_channel);
    size_t need = fix_gain_get_mem_size(&cfg);
    uint8_t* raw = (uint8_t*)malloc(need + 16);
    uint8_t* pool;
    FixGain* heap;
    FixGain* fromPool;
    float* src = (float*)malloc((size_t)n * sizeof(float));
    float* got_block = (float*)malloc((size_t)n * sizeof(float));
    float* got_single = (float*)malloc((size_t)n * sizeof(float));
    float* got_pool = (float*)malloc((size_t)n * sizeof(float));
    float* want = (float*)malloc((size_t)n * sizeof(float));
    int ch, i, clipped = 0;

    CHECK(raw && src && got_block && got_single && got_pool && want,
          "allocation");
    pool = (uint8_t*)(((uintptr_t)raw + 15u) & ~(uintptr_t)15u);

    heap = fix_gain_create(&cfg);
    fromPool = fix_gain_init(&cfg, pool, need);
    CHECK(heap && fromPool, "both constructors must succeed");

    test_rng = 0x0f1e2d3cu;
    for (i = 0; i < n; ++i) src[i] = random_signed() * 1.5f;

    for (ch = 0; ch < cfg.channels; ++ch) {
        const float gain = cfg.global_gain * per_channel[ch];

        memcpy(want, src, (size_t)n * sizeof(float));
        fix_gain_fused_reference(want, n, gain, cfg.enable_clip,
                                 cfg.clip_value);

        memcpy(got_block, src, (size_t)n * sizeof(float));
        fix_gain_process(heap, got_block, n, ch);

        memcpy(got_pool, src, (size_t)n * sizeof(float));
        fix_gain_process(fromPool, got_pool, n, ch);

        /* The integrator drives this module one sample at a time. */
        memcpy(got_single, src, (size_t)n * sizeof(float));
        for (i = 0; i < n; ++i) fix_gain_process(heap, got_single + i, 1, ch);

        CHECK(memcmp(got_block, want, (size_t)n * sizeof(float)) == 0,
              "block call must be byte-identical to the fused reference");
        CHECK(memcmp(got_single, want, (size_t)n * sizeof(float)) == 0,
              "per-sample calls must be byte-identical to the block call");
        CHECK(memcmp(got_pool, want, (size_t)n * sizeof(float)) == 0,
              "pool instance must be byte-identical to the heap instance");

        for (i = 0; i < n; ++i) {
            if (fabsf(want[i]) >= cfg.clip_value) clipped++;
        }
    }

    /* ⚠ Written so it can FAIL: with no sample driven past clip_value the
     * comparisons above would never exercise the clip path at all. */
    CHECK(clipped > 0, "the driver never reached the clip limit -- "
                       "the equivalence above would prove nothing");

    /* Out-of-range channel, n <= 0 and a disabled instance are no-ops. */
    memcpy(got_block, src, (size_t)n * sizeof(float));
    fix_gain_process(heap, got_block, n, cfg.channels);
    fix_gain_process(heap, got_block, 0, 0);
    CHECK(memcmp(got_block, src, (size_t)n * sizeof(float)) == 0,
          "an out-of-range channel or empty block must not touch the signal");

    fix_gain_destroy(heap);
    fix_gain_destroy(fromPool);
    free(want); free(got_pool); free(got_single); free(got_block);
    free(src); free(raw);
    printf("PASS fix_gain block/per-sample/pool match the fused reference\n");
    return 1;
}

/* Verbatim transcription of nr_gain_process()'s smoother. */
static float nr_gain_reference(float prev, const NrGainConfig* cfg,
                               float doa_gain) {
    float min_gain = cfg->min_gain;
    float max_gain = cfg->max_gain;
    float target_gain, alpha, out;

    if (!cfg->enable) return 1.0f;
    if (cfg->only_when_nr_enable && !cfg->nr_enable) return 1.0f;

    if (min_gain < 0.0f) min_gain = 0.0f;
    if (max_gain < min_gain) max_gain = min_gain;

    target_gain = isnan(doa_gain) ? cfg->noise_gain : cfg->target_gain;
    if (target_gain < min_gain) target_gain = min_gain;
    else if (target_gain > max_gain) target_gain = max_gain;

    alpha = (target_gain > prev) ? cfg->attack_alpha : cfg->release_alpha;
    if (alpha < 0.0f) alpha = 0.0f;
    else if (alpha > 0.9999f) alpha = 0.9999f;

    out = alpha * prev + (1.0f - alpha) * target_gain;
    if (out < min_gain) out = min_gain;
    else if (out > max_gain) out = max_gain;
    return out;
}

static int test_nr_gain_pool_and_smoother_reference(void) {
    const int n = 130;
    const int hops = 64;
    size_t need = nr_gain_get_mem_size();
    size_t pool_bytes = need + 128;
    uint8_t* pool = NULL;
    NrGain* ng;
    NrGain* heap;
    NrGainConfig cfg;
    float* x = (float*)malloc((size_t)n * sizeof(float));
    float* y = (float*)malloc((size_t)n * sizeof(float));
    float ref = 1.0f;
    int t, i, rose = 0, fell = 0;

    CHECK(need > 0 && x && y, "sizing/allocation");
    CHECK(check_pool_contract("nr_gain", need,
                              pool_init_nr_gain, pool_destroy_nr_gain, NULL),
          "nr_gain pool contract");

    CHECK(posix_memalign((void**)&pool, 16, pool_bytes) == 0 && pool,
          "pool allocation");
    ng = nr_gain_init(pool, pool_bytes);
    heap = nr_gain_create();
    CHECK(ng && heap, "both constructors must succeed");
    CHECK(nr_gain_get_gain(ng) == 1.0f, "a fresh instance starts at unity");

    memset(&cfg, 0, sizeof(cfg));
    cfg.enable = 1;
    cfg.only_when_nr_enable = 1;
    cfg.nr_enable = 1;
    cfg.target_gain = 2.0f;
    cfg.noise_gain = 0.25f;
    cfg.min_gain = 0.1f;
    cfg.max_gain = 4.0f;
    cfg.attack_alpha = 0.4f;
    cfg.release_alpha = 0.85f;

    test_rng = 0x5150c0deu;
    for (t = 0; t < hops; ++t) {
        /* Alternate between a located source and no estimate so both the
         * attack and the release branch run. */
        float doa_gain = (t % 8 < 4) ? 1.0f : (float)NAN;
        float got_pool, got_heap, prev = ref;

        for (i = 0; i < n; ++i) x[i] = random_signed();
        memcpy(y, x, (size_t)n * sizeof(float));

        got_pool = nr_gain_process(ng, &cfg, x, n, doa_gain);
        got_heap = nr_gain_process(heap, &cfg, y, n, doa_gain);
        ref = nr_gain_reference(prev, &cfg, doa_gain);

        CHECK(got_pool == ref, "smoothed gain must match the reference");
        CHECK(got_heap == ref, "heap instance must track the pool instance");
        CHECK(nr_gain_get_gain(ng) == ref, "the getter must report it too");
        CHECK(memcmp(x, y, (size_t)n * sizeof(float)) == 0,
              "heap and pool must scale the hop identically");

        if (ref > prev) rose++;
        if (ref < prev) fell++;
    }

    /* ⚠ Written so it can FAIL: a driver that only ever rose (or only fell)
     * would leave one of the two coefficients untested. */
    CHECK(rose > 0 && fell > 0,
          "the driver never exercised both attack and release");

    /* A disabled config pins the gain back to unity. */
    cfg.enable = 0;
    CHECK(nr_gain_process(ng, &cfg, NULL, 0, 1.0f) == 1.0f,
          "a disabled config must pin the gain to unity");
    cfg.enable = 1;
    cfg.nr_enable = 0;
    CHECK(nr_gain_process(ng, &cfg, NULL, 0, 1.0f) == 1.0f,
          "only_when_nr_enable must pin the gain while NR is off");

    nr_gain_destroy(ng);
    nr_gain_destroy(heap);
    free(pool); free(y); free(x);
    printf("PASS nr_gain smoother matches the reference\n");
    return 1;
}

/* ---------- post_gain: the pre-kernel formulation, transcribed ----------
 *
 * An independent re-implementation of post_gain_apply() as it stood before
 * the per-bin clamps were hoisted and the stage-5 clip/apply moved to
 * sk_clip_f32/sk_capply_gain_f32: per-bin clamp of a frame-constant gain, a
 * copy of the caller's mask even when relaxation is off, and one fused
 * smooth-clamp-apply loop. Byte-equality against this is what makes the
 * restructure a refactor rather than a retune. */
typedef struct {
    int F;
    float* prev_gain;
    float* target_gain;
    float* freq_gain;
    int* mask_work;
    int* raw_mask_frame;
    int* class_frame;
    int cnt_match;
    int cnt_suppress;
    /* How many times the final clamp actually moved a value. The bounds are
     * already applied to the frame-constant gains upstream, so in steady
     * state this stage is a no-op -- it only bites after the caller tightens
     * min_gain/max_gain under a smoother still holding older, looser values.
     * Counted so the equivalence below cannot quietly stop covering it. */
    int clip_hits;
} PgRef;

/* pg_ref_angle_near and pg_ref_angle_to_index are transcribed UNCHANGED from
 * the module: the directional decision was not part of the restructure, so
 * these two are a mirror, not a gate -- a bug in the module's angle maths
 * would be copied here and compare equal. What this reference does gate is
 * every stage that did change: the mask-relaxation branch, the hoisted
 * clamps, the split box filter and the fused smooth-clamp-apply. */
static int pg_ref_angle_near(int a, int b, int num_angles, int tol) {
    int d;
    if (a < 0 || b < 0) return 0;
    d = abs(a - b);
    if (d > num_angles / 2) d = num_angles - d;
    return (d <= tol);
}

static int pg_ref_angle_to_index(float doa_rad, int num_angles) {
    float two_pi, idx_f;
    if (num_angles <= 0 || !isfinite(doa_rad)) return -1;
    two_pi = 2.0f * (float)M_PI;
    while (doa_rad < 0.0f) doa_rad += two_pi;
    while (doa_rad >= two_pi) doa_rad -= two_pi;
    idx_f = doa_rad / two_pi * (float)num_angles;
    return (int)roundf(idx_f) % num_angles;
}

static float pg_ref_clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static int pg_ref_init(PgRef* r, const PostGainConfig* cfg) {
    int f;
    r->F = cfg->F;
    r->prev_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->target_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->freq_gain = (float*)malloc((size_t)r->F * sizeof(float));
    r->mask_work = (int*)malloc((size_t)r->F * sizeof(int));
    r->raw_mask_frame = (int*)malloc((size_t)r->F * sizeof(int));
    r->class_frame = (int*)malloc((size_t)r->F * sizeof(int));
    if (!r->prev_gain || !r->target_gain || !r->freq_gain || !r->mask_work ||
        !r->raw_mask_frame || !r->class_frame) return 0;
    for (f = 0; f < r->F; f++) {
        r->prev_gain[f] = cfg->gain_match;
        r->target_gain[f] = cfg->gain_match;
        r->freq_gain[f] = cfg->gain_match;
        r->mask_work[f] = 0;
        r->raw_mask_frame[f] = 0;
        r->class_frame[f] = 0;
    }
    r->cnt_match = 0;
    r->cnt_suppress = 0;
    r->clip_hits = 0;
    return 1;
}

static void pg_ref_free(PgRef* r) {
    free(r->prev_gain); free(r->target_gain); free(r->freq_gain);
    free(r->mask_work); free(r->raw_mask_frame); free(r->class_frame);
}

static void pg_ref_apply(PgRef* st, const PostGainConfig* cfg, Complex* Y,
                         const int* mask, const int* bin_best_idx,
                         float doa_used) {
    const int* use_mask;
    int F = st->F;
    int num_angles, doa_used_idx, cnt_match = 0, cnt_suppress = 0;
    int angle_match_cnt = 0, angle_vad = 0, f;
    float angle_ratio;

    if (!cfg->enable) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f] = 0;
            st->target_gain[f] = 1.0f;
            st->freq_gain[f] = 1.0f;
            st->prev_gain[f] = 1.0f;
        }
        st->cnt_match = 0;
        st->cnt_suppress = 0;
        return;
    }

    num_angles = cfg->num_angles;
    doa_used_idx = pg_ref_angle_to_index(doa_used, num_angles);
    if (doa_used_idx < 0) {
        for (f = 0; f < F; f++) {
            st->raw_mask_frame[f] = 0;
            st->class_frame[f] = 0;
        }
        st->cnt_match = 0;
        st->cnt_suppress = F;
        return;
    }

    if (cfg->enable_mask_relax && cfg->mask_relax_bins > 0) {
        int r = cfg->mask_relax_bins;
        for (f = 0; f < F; f++) {
            int keep = 0, k;
            for (k = -r; k <= r; k++) {
                int ff = f + k;
                if (ff >= 0 && ff < F && mask[ff]) { keep = 1; break; }
            }
            st->mask_work[f] = keep;
        }
        use_mask = st->mask_work;
    } else {
        for (f = 0; f < F; f++) st->mask_work[f] = mask[f];
        use_mask = st->mask_work;
    }

    for (f = 0; f < F; f++) {
        int angle_match = 0;
        if (use_mask[f] && pg_ref_angle_near(bin_best_idx[f], doa_used_idx,
                                             num_angles, cfg->angle_tol)) {
            angle_match = 1;
            angle_match_cnt++;
        }
        st->raw_mask_frame[f] = angle_match;
    }

    angle_ratio = (float)angle_match_cnt / (float)F;
    if (angle_ratio > cfg->angle_vad_thr) angle_vad = 1;

    for (f = 0; f < F; f++) {
        int angle_match = st->raw_mask_frame[f];
        int cls = 0;
        float gain;
        if (angle_vad && angle_match) {
            gain = cfg->gain_match; cls = 2; cnt_match++;
        } else {
            gain = cfg->gain_suppress; cls = 0; cnt_suppress++;
        }
        gain = pg_ref_clamp(gain, cfg->min_gain, cfg->max_gain);
        st->target_gain[f] = gain;
        st->class_frame[f] = cls;
    }

    if (cfg->enable_freq_smooth && cfg->freq_smooth_radius > 0) {
        int r = cfg->freq_smooth_radius;
        for (f = 0; f < F; f++) {
            float sum = 0.0f;
            int count = 0, k;
            for (k = -r; k <= r; k++) {
                int ff = f + k;
                if (ff >= 0 && ff < F) { sum += st->target_gain[ff]; count++; }
            }
            st->freq_gain[f] = sum / (float)count;
        }
    } else {
        for (f = 0; f < F; f++) st->freq_gain[f] = st->target_gain[f];
    }

    for (f = 0; f < F; f++) {
        float target = st->freq_gain[f];
        float prev = st->prev_gain[f];
        float smooth_gain;
        if (cfg->enable_time_smooth) {
            float alpha = (target > prev) ? cfg->attack_alpha
                                          : cfg->release_alpha;
            smooth_gain = alpha * prev + (1.0f - alpha) * target;
        } else {
            smooth_gain = target;
        }
        {
            float clamped = pg_ref_clamp(smooth_gain, cfg->min_gain,
                                         cfg->max_gain);
            if (clamped != smooth_gain) st->clip_hits++;
            smooth_gain = clamped;
        }
        Y[f].r *= smooth_gain;
        Y[f].i *= smooth_gain;
        st->prev_gain[f] = smooth_gain;
    }

    st->cnt_match = cnt_match;
    st->cnt_suppress = cnt_suppress;
}

static PostGainConfig post_gain_test_config(int F, int relax, int freq_smooth) {
    PostGainConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.F = F;
    cfg.num_angles = 72;
    cfg.enable = 1;
    cfg.gain_match = 1.0f;
    cfg.gain_suppress = 0.25f;
    cfg.angle_tol = 3;
    cfg.angle_vad_thr = 0.1f;
    cfg.enable_freq_smooth = freq_smooth;
    cfg.freq_smooth_radius = 2;
    cfg.enable_time_smooth = 1;
    cfg.attack_alpha = 0.3f;
    cfg.release_alpha = 0.8f;
    cfg.min_gain = 0.3f;      /* below gain_suppress: the clamp must bite */
    cfg.max_gain = 0.9f;      /* below gain_match: ... at both ends */
    cfg.enable_mask_relax = relax;
    cfg.mask_relax_bins = 1;
    return cfg;
}

static int test_post_gain_pool_poison_and_bounds(void) {
    PostGainConfig cfg = post_gain_test_config(65, 1, 1);
    PostGainConfig bad = cfg;

    CHECK(check_pool_contract("post_gain", post_gain_get_mem_size(&cfg),
                              pool_init_post_gain, pool_destroy_post_gain,
                              &cfg),
          "post_gain pool contract");

    bad.F = 0;
    CHECK(post_gain_get_mem_size(&bad) == 0, "F == 0 must not size");
    CHECK(post_gain_create(&bad) == NULL, "F == 0 must not construct");
    printf("PASS post_gain config gate\n");
    return 1;
}

static int post_gain_drive_case(int relax, int freq_smooth,
                                int* out_vad_hits, int* out_clip_hits) {
    const int F = 65;                   /* not a multiple of 4: kernel tail */
    const int frames = 48;
    PostGainConfig cfg = post_gain_test_config(F, relax, freq_smooth);
    size_t need = post_gain_get_mem_size(&cfg);
    uint8_t* raw = (uint8_t*)malloc(need + 16);
    uint8_t* pool;
    PostGainState* heap;
    PostGainState* fromPool;
    const float* heap_gain;
    PgRef ref;
    Complex* Y_heap = (Complex*)malloc((size_t)F * sizeof(Complex));
    Complex* Y_pool = (Complex*)malloc((size_t)F * sizeof(Complex));
    Complex* Y_ref = (Complex*)malloc((size_t)F * sizeof(Complex));
    int* mask = (int*)malloc((size_t)F * sizeof(int));
    int* bin_best = (int*)malloc((size_t)F * sizeof(int));
    int t, f;

    CHECK(raw && Y_heap && Y_pool && Y_ref && mask && bin_best, "allocation");
    pool = (uint8_t*)(((uintptr_t)raw + 15u) & ~(uintptr_t)15u);

    heap = post_gain_create(&cfg);
    fromPool = post_gain_init(&cfg, pool, need);
    CHECK(heap && fromPool, "both constructors must succeed");
    CHECK(pg_ref_init(&ref, &cfg), "reference allocation");

    test_rng = 0x13579bdfu;
    for (t = 0; t < frames; ++t) {
        /* Sweep the steered angle so some frames clear the angle-VAD ratio
         * and some do not. */
        float doa = (float)t * 0.13f;
        const PostGainStats* stats;

        /* Half way through, tighten the gain window under a smoother still
         * carrying values from the looser one, so the final clamp has
         * something to do. */
        if (t == frames / 2) {
            cfg.min_gain = 0.7f;
            cfg.max_gain = 0.75f;
        }

        for (f = 0; f < F; ++f) {
            Y_heap[f].r = random_signed() * 2.0f;
            Y_heap[f].i = random_signed() * 2.0f;
            mask[f] = (random_signed() > -0.2f) ? 1 : 0;
            bin_best[f] = (int)((random_signed() * 0.5f + 0.5f) * 72.0f) % 72;
        }
        memcpy(Y_pool, Y_heap, (size_t)F * sizeof(Complex));
        memcpy(Y_ref, Y_heap, (size_t)F * sizeof(Complex));

        post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, doa);
        post_gain_apply(fromPool, &cfg, Y_pool, mask, bin_best, doa);
        pg_ref_apply(&ref, &cfg, Y_ref, mask, bin_best, doa);

        CHECK(memcmp(Y_heap, Y_ref, (size_t)F * sizeof(Complex)) == 0,
              "shaped spectrum must be byte-identical to the reference");
        CHECK(memcmp(Y_pool, Y_ref, (size_t)F * sizeof(Complex)) == 0,
              "pool instance must be byte-identical to the reference");
        /* target_gain and freq_gain are internal to the module now; they
         * both feed prev_gain, which is public and compared here, and the
         * shaped spectrum above. */
        heap_gain = post_gain_get_gain(heap);
        CHECK(heap_gain && memcmp(heap_gain, ref.prev_gain,
                                  (size_t)F * sizeof(float)) == 0,
              "applied per-bin gain must be byte-identical");
        CHECK(memcmp(post_gain_get_raw_mask(heap), ref.raw_mask_frame,
                     (size_t)F * sizeof(int)) == 0,
              "raw directional mask must be byte-identical");
        CHECK(memcmp(post_gain_get_class(heap), ref.class_frame,
                     (size_t)F * sizeof(int)) == 0,
              "per-bin class must be byte-identical");
        stats = post_gain_get_stats(heap);
        CHECK(stats->cnt_match == ref.cnt_match &&
              stats->cnt_suppress == ref.cnt_suppress,
              "frame stats must match the reference");

        if (ref.cnt_match > 0) (*out_vad_hits)++;
    }

    /* The bypass path resets state rather than shaping. */
    cfg.enable = 0;
    post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, 0.25f);
    pg_ref_apply(&ref, &cfg, Y_ref, mask, bin_best, 0.25f);
    CHECK(memcmp(post_gain_get_gain(heap), ref.prev_gain,
                 (size_t)F * sizeof(float)) == 0,
          "bypass must reset the gain state exactly like the reference");
    cfg.enable = 1;

    /* A steering angle that is not finite short-circuits before any gain is
     * applied. +/-inf matters as much as NaN: the module reduces the angle by
     * repeated subtraction, so an unguarded infinity spins forever -- this
     * assertion only returns at all because the guard tests isfinite(). */
    {
        const float bad_angles[3] = { (float)NAN, (float)INFINITY,
                                      -(float)INFINITY };
        int b;
        for (b = 0; b < 3; ++b) {
            memcpy(Y_pool, Y_heap, (size_t)F * sizeof(Complex));
            post_gain_apply(heap, &cfg, Y_heap, mask, bin_best, bad_angles[b]);
            CHECK(memcmp(Y_heap, Y_pool, (size_t)F * sizeof(Complex)) == 0,
                  "an unusable steering angle must leave the spectrum "
                  "untouched");
            CHECK(post_gain_get_stats(heap)->cnt_suppress == F,
                  "an unusable steering angle must classify the frame "
                  "all-suppress");
        }
    }

    *out_clip_hits += ref.clip_hits;

    pg_ref_free(&ref);
    post_gain_destroy(heap);
    post_gain_destroy(fromPool);
    free(bin_best); free(mask); free(Y_ref); free(Y_pool); free(Y_heap);
    free(raw);
    return 1;
}

static int test_post_gain_matches_prekernel_reference(void) {
    int vad_hits = 0;
    int clip_hits = 0;

    CHECK(post_gain_drive_case(1, 1, &vad_hits, &clip_hits),
          "mask relaxation + frequency smoothing");
    CHECK(post_gain_drive_case(0, 0, &vad_hits, &clip_hits),
          "no relaxation, no frequency smoothing");

    /* ⚠ Written so it can FAIL: if no frame ever produced a matched bin the
     * comparisons above would only ever have compared the suppress branch. */
    CHECK(vad_hits > 0, "the driver never produced a directional match -- "
                        "the equivalence above would prove nothing");
    /* ⚠ Likewise for the output clamp: without a frame whose smoothed gain
     * lands outside the window, dropping that stage entirely would still
     * compare equal. */
    CHECK(clip_hits > 0, "the driver never drove a gain outside the window -- "
                         "the output clamp would be untested");
    printf("PASS post_gain matches the pre-kernel reference "
           "over both smoothing configurations\n");
    return 1;
}

static int run_all_tests(void) {
    CHECK(test_phat_scalar_vs_dispatch(), "PHAT SIMD test");
    CHECK(test_beamform_and_score_scalar_vs_dispatch(),
          "beamform/SRP-score SIMD test");
    CHECK(test_gsc_vector_kernels_scalar_vs_dispatch(),
          "GSC vector-kernel SIMD test");
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
    CHECK(test_vad_api_init_pool_poison_and_bounds(),
          "VAD pool-first poison/bounds test");
    CHECK(test_vad_api_heap_vs_pool_identical(),
          "VAD heap-vs-pool equivalence test");
    CHECK(test_gsc_heap_vs_pool_byte_equal(),
          "GSC heap-vs-pool byte-equal test");
    CHECK(test_gsc_rls_dispatch_matches_scalar_state(),
          "GSC RLS scalar-vs-dispatch recursive-state test");
    CHECK(test_gsc_frame_idx_survives_32bit_boundary(),
          "GSC frame_idx 32-bit-boundary test");
    CHECK(test_srp_frame_counter_survives_32bit_boundary(),
          "SRP frame_counter 32-bit-boundary test");
    CHECK(test_fix_gain_pool_poison_and_bounds(),
          "fix_gain pool-first poison/bounds test");
    CHECK(test_fix_gain_block_and_per_sample_match_reference(),
          "fix_gain fused-reference equivalence test");
    CHECK(test_nr_gain_pool_and_smoother_reference(),
          "nr_gain pool-first bounds and smoother-reference test");
    CHECK(test_post_gain_pool_poison_and_bounds(),
          "post_gain pool-first poison/bounds test");
    CHECK(test_post_gain_matches_prekernel_reference(),
          "post_gain pre-kernel-reference equivalence test");
    printf("All third-party spatial tests passed (backend=%s)\n",
           spatial_simd_backend());
    return 1;
}

int main(void) {
    return run_all_tests() ? 0 : 1;
}
