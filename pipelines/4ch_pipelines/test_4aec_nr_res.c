/**
 * Structural and lifecycle tests for 4ch_pipelines/4aec_nr_res.h.
 *
 * The test intentionally uses equal weights as an external-beamformer stand
 * in. The library itself must never choose those weights in production.
 */

#include "4aec_nr_res.h"
#include "4aec_nr_res_internal.h"
#include "4aec_projection_kernels.h"
#include "fft_wrapper.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, label)                                            \
    do {                                                                   \
        if (condition) {                                                   \
            printf("PASS: %s\n", label);                                   \
        } else {                                                           \
            printf("FAIL: %s\n", label);                                   \
            failures += 1;                                                 \
        }                                                                  \
    } while (0)

static int all_finite(const float* values, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static void fill_inputs(float* microphones, float* ref, int hop,
                        int sample_rate, int frame_index);

static void test_projection_kernels(void) {
    enum { N = 17 };
    Complex weights[N];
    Complex input[N];
    Complex echo[N];
    Complex acc_scalar[N];
    Complex acc_dispatch[N];
    Complex residual_scalar[N];
    Complex residual_dispatch[N];
    float r2[N];
    float comfort[N];
    float comfort_scalar[N];
    float comfort_dispatch[N];
    float mag2_scalar[N];
    float mag2_dispatch[N];
    int i;

    for (i = 0; i < N; ++i) {
        float x = (float)(i + 1);
        weights[i].r = 0.03125f * x;
        weights[i].i = -0.015625f * (float)(i % 5);
        input[i].r = 0.0078125f * (float)(i - 8);
        input[i].i = 0.00390625f * (float)(7 - i);
        echo[i].r = 0.002f * (float)(i - 6);
        echo[i].i = -0.001f * (float)(i + 2);
        r2[i] = 1.0e-5f * x;
        comfort[i] = (i % 4 == 0) ? -0.25f : 0.01f * x;
        acc_scalar[i].r = acc_dispatch[i].r = 0.125f * x;
        acc_scalar[i].i = acc_dispatch[i].i = -0.0625f * x;
        comfort_scalar[i] = comfort_dispatch[i] = 0.5f * x;
    }
    /* Explicitly exercise both no-phase fallback shapes and a zero R2 bin. */
    echo[0].r = echo[0].i = 0.0f;
    echo[1].r = 1.0e-21f;
    echo[1].i = -1.0e-21f;
    r2[2] = 0.0f;

    four_aec_projection_cmac_scalar(
        acc_scalar, weights, input, N);
    four_aec_projection_cmac(
        acc_dispatch, weights, input, N);
    CHECK(memcmp(acc_scalar, acc_dispatch, sizeof(acc_scalar)) == 0,
          "context complex-MAC SIMD matches scalar bytes");

    four_aec_complex_mag2_scalar(mag2_scalar, input, N);
    four_aec_complex_mag2(mag2_dispatch, input, N);
    CHECK(memcmp(mag2_scalar, mag2_dispatch, sizeof(mag2_scalar)) == 0,
          "context magnitude-squared SIMD matches scalar bytes");

    four_aec_residual_vector_scalar(
        residual_scalar, echo, r2, N);
    four_aec_residual_vector(
        residual_dispatch, echo, r2, N);
    CHECK(memcmp(
              residual_scalar, residual_dispatch,
              sizeof(residual_scalar)) == 0,
          "residual projection SIMD matches scalar bytes");

    four_aec_comfort_accumulate_scalar(
        comfort_scalar, weights, comfort, N);
    four_aec_comfort_accumulate(
        comfort_dispatch, weights, comfort, N);
    CHECK(memcmp(
              comfort_scalar, comfort_dispatch,
              sizeof(comfort_scalar)) == 0,
          "comfort projection SIMD matches scalar bytes");

    /* n=0 is part of every header kernel's boundary contract. */
    four_aec_projection_cmac(acc_dispatch, weights, input, 0);
    four_aec_complex_mag2(mag2_dispatch, input, 0);
    four_aec_residual_vector(residual_dispatch, echo, r2, 0);
    four_aec_comfort_accumulate(
        comfort_dispatch, weights, comfort, 0);
    CHECK(1, "projection kernels accept an empty range");
}

static void test_trusted_spectrum_path(void) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* reconstructed = NULL;
    FourAecNrRes* trusted = NULL;
    FourAecNrResPreFrame reconstructed_pre;
    FourAecNrResPreFrame trusted_pre;
    float* microphones = NULL;
    float* ref = NULL;
    float* reconstructed_out = NULL;
    float* trusted_out = NULL;
    Complex* weights = NULL;
    Complex* trusted_spectrum = NULL;
    int hop;
    int n_freqs;
    int frame;
    int ch;
    int k;

    cfg.fft_size = 256;
    cfg.enable_cng = 0;
    reconstructed = four_aec_nr_res_create(&cfg);
    trusted = four_aec_nr_res_create(&cfg);
    CHECK(reconstructed && trusted,
          "trusted-spectrum parity instances create");
    if (!reconstructed || !trusted) goto cleanup;

    hop = four_aec_nr_res_hop_size(reconstructed);
    n_freqs = four_aec_nr_res_n_freqs(reconstructed);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    reconstructed_out = (float*)calloc((size_t)hop, sizeof(float));
    trusted_out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    trusted_spectrum = (Complex*)calloc(
        (size_t)n_freqs, sizeof(Complex));
    CHECK(microphones && ref && reconstructed_out && trusted_out &&
              weights && trusted_spectrum,
          "trusted-spectrum parity buffers allocate");
    if (!microphones || !ref || !reconstructed_out || !trusted_out ||
        !weights || !trusted_spectrum) goto cleanup;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            Complex* w = weights + (size_t)ch * n_freqs + k;
            w->r = 0.20f + 0.025f * (float)ch;
            w->i = 0.0001f * (float)((k % 7) - 3) * (float)ch;
        }
    }

    for (frame = 0; frame < 4; ++frame) {
        fill_inputs(microphones, ref, hop, 16000, frame);
        CHECK(four_aec_nr_res_process_pre(
                  reconstructed, microphones, ref, &reconstructed_pre) ==
                  FOUR_AEC_NR_RES_OK &&
              four_aec_nr_res_process_pre(
                  trusted, microphones, ref, &trusted_pre) ==
                  FOUR_AEC_NR_RES_OK,
              "trusted-spectrum parity pre stages succeed");
        memset(
            trusted_spectrum, 0,
            (size_t)n_freqs * sizeof(Complex));
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            four_aec_projection_cmac_scalar(
                trusted_spectrum,
                weights + (size_t)ch * n_freqs,
                trusted_pre.linear_spectra[ch], n_freqs);
        }
        CHECK(four_aec_nr_res_process_post(
                  reconstructed, &reconstructed_pre.token,
                  weights, reconstructed_out) == FOUR_AEC_NR_RES_OK &&
              four_aec_nr_res_process_post_trusted_spectrum(
                  trusted, &trusted_pre.token, weights,
                  trusted_spectrum, trusted_out) == FOUR_AEC_NR_RES_OK,
              "trusted and reconstructed post stages succeed");
        CHECK(memcmp(
                  reconstructed_out, trusted_out,
                  (size_t)hop * sizeof(float)) == 0,
              "trusted spectrum skips reconstruction without output drift");
    }

cleanup:
    free(trusted_spectrum);
    free(weights);
    free(trusted_out);
    free(reconstructed_out);
    free(ref);
    free(microphones);
    four_aec_nr_res_destroy(trusted);
    four_aec_nr_res_destroy(reconstructed);
}

static void fill_inputs(float* microphones, float* ref, int hop,
                        int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int64_t absolute = (int64_t)frame_index * hop + i;
        float phase = 2.0f * 3.14159265358979323846f *
                      440.0f * (float)absolute / (float)sample_rate;
        ref[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            float near = 0.01f * sinf(
                phase * (1.0f + 0.05f * (float)ch));
            microphones[
                i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                ref[i] * (0.4f + 0.05f * (float)ch) + near;
        }
    }
}

static int run_grid(int sample_rate, int fft_size) {
    FourAecNrResConfig cfg;
    FourAecNrRes* p;
    FourAecNrResPreFrame pre;
    FourAecNrResFrameToken stale;
    float* microphones;
    float* ref;
    float* out;
    Complex* weights;
    int hop;
    int n_freqs;
    int frame;
    int ch;
    int k;
    int rc;

    cfg = four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.enable_cng = 0;
    p = four_aec_nr_res_create(&cfg);
    CHECK(p != NULL, "create supported rate");
    if (!p) return 0;

    hop = four_aec_nr_res_hop_size(p);
    n_freqs = four_aec_nr_res_n_freqs(p);
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(four_aec_nr_res_fft_size(p) == 2 * hop,
          "FFT is two hops");
    CHECK(four_aec_nr_res_sample_rate(p) == sample_rate,
          "sample-rate accessor");
    CHECK(four_aec_nr_res_matched_filter_count(p) == 1,
          "one shared matcher");
    CHECK(four_aec_nr_res_linear_aec_count(p) ==
              FOUR_AEC_NR_RES_CHANNELS,
          "four linear AEC filters");
    CHECK(four_aec_nr_res_nr_count(p) == 1,
          "one mono NR");
    CHECK(four_aec_nr_res_post_res_count(p) == 1,
          "one post-beam RES");

    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    CHECK(microphones && ref && out && weights, "test buffers allocate");
    if (!microphones || !ref || !out || !weights) {
        free(microphones);
        free(ref);
        free(out);
        free(weights);
        four_aec_nr_res_destroy(p);
        return 0;
    }

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }

    for (frame = 0; frame < 3; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        rc = four_aec_nr_res_process_pre(
            p, microphones, ref, &pre);
        CHECK(rc == FOUR_AEC_NR_RES_OK, "pre stage succeeds");
        if (rc != FOUR_AEC_NR_RES_OK) break;
        CHECK(pre.linear_interleaved != NULL &&
              pre.hop_size == hop &&
              pre.n_channels == FOUR_AEC_NR_RES_CHANNELS,
              "pre frame shape contract");
        CHECK(all_finite(
                  pre.linear_interleaved,
                  hop * FOUR_AEC_NR_RES_CHANNELS),
              "four linear outputs finite");

        if (frame == 0) {
            FourAecNrResPreFrame duplicate;
            CHECK(four_aec_nr_res_process_pre(
                      p, microphones, ref, &duplicate) ==
                      FOUR_AEC_NR_RES_SEQUENCE_ERROR,
                  "second pre frame is rejected while one is pending");
        }

        stale = pre.token;
        rc = four_aec_nr_res_process_post(
            p, &pre.token, weights, out);
        CHECK(rc == FOUR_AEC_NR_RES_OK, "post NR/RES stage succeeds");
        CHECK(all_finite(out, hop), "mono output is finite");
        CHECK(four_aec_nr_res_process_post(
                  p, &stale, weights, out) ==
                  FOUR_AEC_NR_RES_SEQUENCE_ERROR,
              "post token cannot be replayed");
    }

    fill_inputs(microphones, ref, hop, sample_rate, 4);
    CHECK(four_aec_nr_res_process_pre(
              p, microphones, ref, &pre) == FOUR_AEC_NR_RES_OK,
          "pre before reset succeeds");
    stale = pre.token;
    four_aec_nr_res_reset(p);
    CHECK(four_aec_nr_res_process_post(
              p, &stale, weights, out) ==
              FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "reset invalidates in-flight token");

    fill_inputs(microphones, ref, hop, sample_rate, 5);
    CHECK(four_aec_nr_res_process_pre(
              p, microphones, ref, &pre) == FOUR_AEC_NR_RES_OK,
          "pre after reset succeeds");
    memset(weights, 0,
           (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs *
           sizeof(Complex));
    CHECK(four_aec_nr_res_process_post(
              p, &pre.token, weights, out) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "all-zero beamformer weights are rejected");
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }
    CHECK(four_aec_nr_res_process_post(
              p, &pre.token, weights, out) ==
              FOUR_AEC_NR_RES_OK,
          "pending frame may retry after invalid weights");

    free(microphones);
    free(ref);
    free(out);
    free(weights);
    four_aec_nr_res_destroy(p);
    return 1;
}

static void test_invalid_configs(void) {
    FourAecNrResConfig cfg;
    FourAecNrResMemReq req;

    cfg = four_aec_nr_res_default_config(8000);
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "8 kHz is outside the 4-channel contract");
    CHECK(four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0,
          "static sizing rejects 8 kHz");

    cfg = four_aec_nr_res_default_config(16000);
    cfg.fft_size = 1024;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "cross-rate FFT grid is rejected");

    cfg = four_aec_nr_res_default_config(16000);
    cfg.capture_proxy_channel = 4;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "invalid capture proxy is rejected");
}

/* A frame token is stamped with the owning instance's pointer
 * (owner_cookie); process_post() must reject a token minted by a different
 * live instance, and doing so must not disturb either instance's own
 * pending frame. */
static void test_cross_instance_token(void) {
    FourAecNrResConfig cfg;
    FourAecNrRes* a = NULL;
    FourAecNrRes* b = NULL;
    FourAecNrResPreFrame pre_a;
    FourAecNrResPreFrame pre_b;
    float* microphones = NULL;
    float* ref = NULL;
    float* out = NULL;
    Complex* weights = NULL;
    int hop;
    int n_freqs;
    int ch;
    int k;

    cfg = four_aec_nr_res_default_config(16000);
    cfg.enable_cng = 0;
    a = four_aec_nr_res_create(&cfg);
    b = four_aec_nr_res_create(&cfg);
    CHECK(a != NULL && b != NULL, "cross-instance test: two instances create");
    if (!a || !b) goto cleanup;

    hop = four_aec_nr_res_hop_size(a);
    n_freqs = four_aec_nr_res_n_freqs(a);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs, sizeof(Complex));
    CHECK(microphones && ref && out && weights,
          "cross-instance test: buffers allocate");
    if (!microphones || !ref || !out || !weights) goto cleanup;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }

    fill_inputs(microphones, ref, hop, 16000, 0);
    CHECK(four_aec_nr_res_process_pre(a, microphones, ref, &pre_a) ==
              FOUR_AEC_NR_RES_OK,
          "cross-instance test: instance A pre succeeds");
    fill_inputs(microphones, ref, hop, 16000, 1);
    CHECK(four_aec_nr_res_process_pre(b, microphones, ref, &pre_b) ==
              FOUR_AEC_NR_RES_OK,
          "cross-instance test: instance B pre succeeds");

    CHECK(four_aec_nr_res_process_post(
              b, &pre_a.token, weights, out) ==
              FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "post rejects a token minted by a different instance");

    CHECK(four_aec_nr_res_process_post(
              b, &pre_b.token, weights, out) == FOUR_AEC_NR_RES_OK,
          "instance B's own pending frame survives the rejected cross-use");
    CHECK(four_aec_nr_res_process_post(
              a, &pre_a.token, weights, out) == FOUR_AEC_NR_RES_OK,
          "instance A's own pending frame survives the rejected cross-use");

cleanup:
    four_aec_nr_res_destroy(a);
    four_aec_nr_res_destroy(b);
    free(weights);
    free(out);
    free(ref);
    free(microphones);
}

/* destroy() never releases caller-owned pool memory (documented contract),
 * so a caller may init() a brand-new instance into the exact same pool
 * bytes right after destroying the old one. init() memsets that pool to
 * zero, so the new instance's frame_index/generation restart at 0 and its
 * owner_cookie (the instance pointer) is identical to the destroyed
 * instance's, since it is literally the same memory -- a pre-destroy token
 * would be bit-identical to the new instance's own first-frame token if
 * instance_epoch did not also distinguish them. This reproduces that ABA
 * scenario directly and confirms process_post() still rejects the stale
 * pre-destroy token against the new instance. */
static void test_pool_reinit_token_rejected(void) {
    FourAecNrResConfig cfg;
    FourAecNrResMemReq req;
    FourAecNrRes* old_instance = NULL;
    FourAecNrRes* new_instance = NULL;
    FourAecNrResPreFrame pre_old;
    FourAecNrResPreFrame pre_new;
    unsigned char* pool = NULL;
    float* microphones = NULL;
    float* ref = NULL;
    float* out = NULL;
    Complex* weights = NULL;
    int hop;
    int n_freqs;
    int ch;
    int k;

    cfg = four_aec_nr_res_default_config(16000);
    cfg.enable_cng = 0;
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0 ||
        req.bytes > (uint64_t)SIZE_MAX) {
        CHECK(0, "pool-reinit test: memory requirement query succeeds");
        return;
    }
    if (posix_memalign(
            (void**)&pool, (size_t)req.alignment, (size_t)req.bytes) != 0)
        pool = NULL;
    CHECK(pool != NULL, "pool-reinit test: pool allocates");
    if (!pool) return;

    old_instance = four_aec_nr_res_init(pool, (size_t)req.bytes, &cfg);
    CHECK(old_instance != NULL, "pool-reinit test: first instance inits");
    if (!old_instance) { free(pool); return; }

    hop = four_aec_nr_res_hop_size(old_instance);
    n_freqs = four_aec_nr_res_n_freqs(old_instance);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs, sizeof(Complex));
    CHECK(microphones && ref && out && weights,
          "pool-reinit test: buffers allocate");
    if (!microphones || !ref || !out || !weights) {
        four_aec_nr_res_destroy(old_instance);
        free(pool);
        goto cleanup;
    }
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
        }
    }

    fill_inputs(microphones, ref, hop, 16000, 0);
    CHECK(four_aec_nr_res_process_pre(
              old_instance, microphones, ref, &pre_old) ==
              FOUR_AEC_NR_RES_OK,
          "pool-reinit test: first instance pre succeeds");

    /* destroy() does not release or clear caller-owned pool memory. */
    four_aec_nr_res_destroy(old_instance);

    new_instance = four_aec_nr_res_init(pool, (size_t)req.bytes, &cfg);
    CHECK(new_instance != NULL,
          "pool-reinit test: second instance inits into the same pool");
    if (!new_instance) { free(pool); goto cleanup; }

    CHECK((void*)new_instance == (void*)old_instance,
          "pool-reinit test: second instance reuses the exact same address");

    fill_inputs(microphones, ref, hop, 16000, 0);
    CHECK(four_aec_nr_res_process_pre(
              new_instance, microphones, ref, &pre_new) ==
              FOUR_AEC_NR_RES_OK,
          "pool-reinit test: second instance pre succeeds");
    CHECK(pre_old.token.frame_index == pre_new.token.frame_index &&
              pre_old.token.generation == pre_new.token.generation &&
              pre_old.token.owner_cookie == pre_new.token.owner_cookie &&
              pre_old.token.instance_epoch != pre_new.token.instance_epoch,
          "pool-reinit test: only instance_epoch differs between the two "
          "tokens");

    CHECK(four_aec_nr_res_process_post(
              new_instance, &pre_old.token, weights, out) ==
              FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "post rejects a pre-destroy token reused after same-pool re-init");
    CHECK(four_aec_nr_res_process_post(
              new_instance, &pre_new.token, weights, out) ==
              FOUR_AEC_NR_RES_OK,
          "the new instance's own token is still accepted");

    four_aec_nr_res_destroy(new_instance);
    free(pool);

cleanup:
    free(weights);
    free(out);
    free(ref);
    free(microphones);
}

static void run_static_parity(int sample_rate, int fft_size) {
    FourAecNrResConfig cfg;
    FourAecNrResMemReq req;
    FourAecNrResMemReq stale;
    FourAecNrResMemBreakdown breakdown;
    FourAecNrRes* heap = NULL;
    FourAecNrRes* stat = NULL;
    FourAecNrResPreFrame heap_pre;
    FourAecNrResPreFrame stat_pre;
    unsigned char* pool = NULL;
    float* microphones = NULL;
    float* ref = NULL;
    float* heap_out = NULL;
    float* stat_out = NULL;
    Complex* weights = NULL;
    int hop;
    int n_freqs;
    int frame;
    int ch;
    int k;
    int rc;

    cfg = four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    rc = four_aec_nr_res_get_mem_requirements(&cfg, &req);
    CHECK(rc == 0, "static memory requirement query succeeds");
    CHECK(
        four_aec_nr_res_get_mem_breakdown(&cfg, &breakdown) == 0 &&
        req.bytes == (uint64_t)breakdown.total_bytes &&
        breakdown.aec_bytes > breakdown.nr_bytes &&
        breakdown.wrapper_bytes > 0,
        "static memory breakdown reconciles with descriptor");
    if (rc != 0 || req.bytes > (uint64_t)SIZE_MAX) return;

    if (posix_memalign(
            (void**)&pool, (size_t)req.alignment,
            (size_t)req.bytes + 32u) != 0)
        pool = NULL;
    CHECK(pool != NULL, "aligned caller pool allocates");
    if (!pool) return;
    memset(pool, 0xa5, (size_t)req.bytes + 32u);

    CHECK(four_aec_nr_res_init(
              pool + 1, (size_t)req.bytes, &cfg) == NULL,
          "static init rejects a misaligned pool");
    CHECK(four_aec_nr_res_init(
              pool, (size_t)req.bytes - 1u, &cfg) == NULL,
          "static init rejects an undersized pool");

    stale = req;
    stale.descriptor_version += 1u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a stale descriptor ABI");
    stale = req;
    stale.layout_version += 1u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a stale carve layout");
    stale = req;
    stale.backend_id = 99u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a backend mismatch");
    stale = req;
    stale.build_flags_hash ^= 0xffffffffu;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a build-layout mismatch");
    stale = req;
    stale.alignment *= 2u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects an alignment mismatch");
    stale = req;
    stale.reserved = 1u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a corrupt reserved field");
    stale = req;
    stale.bytes -= 1u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects stale descriptor bytes");
    stale = req;
    stale.layout_version -= 1u;
    stale.bytes += 4096u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a stale layout even when its cached "
          "bytes are larger than current (byte count fitting must never "
          "substitute for layout/hash agreement)");

    stat = four_aec_nr_res_init_ex(
        pool, (size_t)req.bytes, &cfg, &req);
    heap = four_aec_nr_res_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");
    if (!stat || !heap) goto cleanup;

    for (k = 0; k < 32; ++k) {
        if (pool[(size_t)req.bytes + (size_t)k] != 0xa5) break;
    }
    CHECK(k == 32, "static init stays inside the queried pool");

    hop = four_aec_nr_res_hop_size(stat);
    n_freqs = four_aec_nr_res_n_freqs(stat);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    heap_out = (float*)calloc((size_t)hop, sizeof(float));
    stat_out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs,
        sizeof(Complex));
    CHECK(microphones && ref && heap_out && stat_out && weights,
          "static parity buffers allocate");
    if (!microphones || !ref || !heap_out || !stat_out || !weights)
        goto cleanup;

    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    for (frame = 0; frame < 4; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        if (four_aec_nr_res_process_pre(
                heap, microphones, ref, &heap_pre) !=
                FOUR_AEC_NR_RES_OK ||
            four_aec_nr_res_process_pre(
                stat, microphones, ref, &stat_pre) !=
                FOUR_AEC_NR_RES_OK) {
            CHECK(0, "heap/static pre stages both succeed");
            break;
        }
        CHECK(memcmp(
                  heap_pre.linear_interleaved,
                  stat_pre.linear_interleaved,
                  (size_t)hop * FOUR_AEC_NR_RES_CHANNELS *
                  sizeof(float)) == 0,
              "heap/static linear outputs are byte-identical");

        if (four_aec_nr_res_process_post(
                heap, &heap_pre.token, weights, heap_out) !=
                FOUR_AEC_NR_RES_OK ||
            four_aec_nr_res_process_post(
                stat, &stat_pre.token, weights, stat_out) !=
                FOUR_AEC_NR_RES_OK) {
            CHECK(0, "heap/static post stages both succeed");
            break;
        }
        CHECK(memcmp(
                  heap_out, stat_out,
                  (size_t)hop * sizeof(float)) == 0,
              "heap/static mono outputs are byte-identical");
    }

    four_aec_nr_res_destroy(stat);
    CHECK(four_aec_nr_res_hop_size(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    four_aec_nr_res_destroy(stat);
    CHECK(four_aec_nr_res_hop_size(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = four_aec_nr_res_init_ex(
        pool, (size_t)req.bytes, &cfg, &req);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

cleanup:
    four_aec_nr_res_destroy(stat);
    four_aec_nr_res_destroy(heap);
    free(weights);
    free(stat_out);
    free(heap_out);
    free(ref);
    free(microphones);
    free(pool);
}

static void test_pre_frame_wola_identity(int sample_rate, int fft_size) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(sample_rate);
    FourAecNrRes* pipeline = NULL;
    FourAecNrResPreFrame pre;
    FftHandle* fft = NULL;
    float* microphones = NULL;
    float* ref = NULL;
    float* out = NULL;
    float* previous = NULL;
    float* ola = NULL;
    float* time_frame = NULL;
    Complex* weights = NULL;
    float* window = NULL;
    float max_error = 0.0f;
    int valid = 1;
    int hop;
    int n_freqs;

    cfg.fft_size = fft_size;
    cfg.enable_cng = 0;
    pipeline = four_aec_nr_res_create(&cfg);
    if (pipeline) fft = fft_create(fft_size);
    CHECK(pipeline != NULL && fft != NULL,
          "4ch WOLA identity instances create");
    if (!pipeline || !fft) goto cleanup;

    hop = four_aec_nr_res_hop_size(pipeline);
    n_freqs = four_aec_nr_res_n_freqs(pipeline);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    previous = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ola = (float*)calloc(
        (size_t)fft_size * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    time_frame = (float*)calloc((size_t)fft_size, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)n_freqs * FOUR_AEC_NR_RES_CHANNELS, sizeof(Complex));
    window = (float*)calloc((size_t)fft_size, sizeof(float));
    if (!microphones || !ref || !out || !previous || !ola ||
        !time_frame || !weights || !window) {
        valid = 0;
        goto check_result;
    }
    for (int i = 0; i < fft_size; ++i) {
        window[i] = sqrtf(0.5f * (1.0f - cosf(
            2.0f * 3.14159265358979323846f * (float)i /
            (float)fft_size)));
    }
    for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (int k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    for (int frame = 0; frame < 40; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        if (four_aec_nr_res_process_pre(
                pipeline, microphones, ref, &pre) != FOUR_AEC_NR_RES_OK) {
            valid = 0;
            break;
        }
        if (pre.delay.changed) {
            /* A delay realignment resets each lane's WOLA history. An
             * external time-domain beamformer must reset its matching OLA
             * state at the same boundary. */
            memset(previous, 0,
                   (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
            memset(ola, 0,
                   (size_t)fft_size * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
        }
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            float* channel_ola = ola + (size_t)ch * fft_size;
            float* channel_previous = previous + (size_t)ch * hop;
            fft_inverse(fft, pre.linear_spectra[ch], time_frame);
            for (int i = 0; i < fft_size; ++i)
                channel_ola[i] += time_frame[i] * window[i];
            for (int i = 0; i < hop; ++i) {
                float d = fabsf(channel_ola[i] - channel_previous[i]);
                if (d > max_error) max_error = d;
                channel_previous[i] = pre.linear_interleaved[
                    i * FOUR_AEC_NR_RES_CHANNELS + ch];
            }
            memmove(channel_ola, channel_ola + hop,
                    (size_t)(fft_size - hop) * sizeof(float));
            memset(channel_ola + (fft_size - hop), 0,
                   (size_t)hop * sizeof(float));
        }
        if (four_aec_nr_res_process_post(
                pipeline, &pre.token, weights, out) != FOUR_AEC_NR_RES_OK) {
            valid = 0;
            break;
        }
    }

check_result:
    CHECK(valid && max_error <= 1e-4f,
          "4ch pre time/spectrum seams share one reconstructing WOLA grid");

cleanup:
    free(window);
    free(weights);
    free(time_frame);
    free(ola);
    free(previous);
    free(out);
    free(ref);
    free(microphones);
    fft_destroy(fft);
    four_aec_nr_res_destroy(pipeline);
}

/* Group 6: lane 0 runs its own far-end FFT every hop; lanes 1-3 borrow it
 * via aec_process_context_shared_far() instead of each recomputing an
 * identical transform. Proves the total (four_aec_nr_res_far_fft_real_
 * compute_count(), summed across all four lanes) increases by AT MOST 1
 * per four_aec_nr_res_process_pre() call, never 4 -- the actual, measured
 * consequence of the sharing, not just "the code compiles and still
 * produces plausible output" (already covered by run_grid()'s finite-
 * output checks and the byte-equal WAV/raw-f32 gates run outside this
 * suite).
 *
 * "At most 1", not "exactly 1": process_pre() resets all four lanes
 * (aec_reset(), which also zeroes this counter) whenever the shared delay
 * estimator's estimate changes -- expected, unrelated pre-existing
 * behavior during the synthetic signal's initial delay-acquisition hops,
 * confirmed empirically while writing this test (the total visibly drops
 * back down before climbing again). A drop is fine; the one invariant
 * that must never break is the total never jumping by MORE than 1 in a
 * single call, which is exactly what "lanes 1-3 silently stopped sharing
 * and went back to computing their own FFT" would look like. */
static void test_far_fft_sharing_reduces_four_to_one(
        int sample_rate, int fft_size) {
    FourAecNrResConfig cfg;
    FourAecNrRes* p;
    FourAecNrResPreFrame pre;
    float* microphones;
    float* ref;
    float* out;
    Complex* weights;
    int hop;
    int n_freqs;
    int frame;
    int ch, k;
    long before, after;
    int saw_a_plus_one_increment = 0;
    int max_single_hop_increment = 0;
    char label[160];

    cfg = four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    p = four_aec_nr_res_create(&cfg);
    CHECK(p != NULL, "far-FFT sharing test: create");
    if (!p) return;

    hop = four_aec_nr_res_hop_size(p);
    n_freqs = four_aec_nr_res_n_freqs(p);
    microphones = (float*)calloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * n_freqs, sizeof(Complex));
    CHECK(microphones && ref && out && weights,
          "far-FFT sharing test: buffers allocate");
    if (!microphones || !ref || !out || !weights) {
        free(microphones); free(ref); free(out); free(weights);
        four_aec_nr_res_destroy(p);
        return;
    }
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;

    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: counter starts at 0",
             sample_rate, fft_size);
    CHECK(four_aec_nr_res_far_fft_real_compute_count(p) == 0, label);

    for (frame = 0; frame < 40; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        before = four_aec_nr_res_far_fft_real_compute_count(p);
        if (four_aec_nr_res_process_pre(p, microphones, ref, &pre) !=
                FOUR_AEC_NR_RES_OK) {
            CHECK(0, "far-FFT sharing test: pre stage succeeds");
            break;
        }
        after = four_aec_nr_res_far_fft_real_compute_count(p);
        if (after - before > max_single_hop_increment)
            max_single_hop_increment = (int)(after - before);
        if (after == before + 1) saw_a_plus_one_increment = 1;
        if (four_aec_nr_res_process_post(p, &pre.token, weights, out) !=
                FOUR_AEC_NR_RES_OK) {
            CHECK(0, "far-FFT sharing test: post stage succeeds");
            break;
        }
    }

    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: total never jumps by more than "
             "1 in a single process_pre() call across 40 hops (max seen: %d)",
             sample_rate, fft_size, max_single_hop_increment);
    CHECK(max_single_hop_increment <= 1, label);
    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: at least one hop actually shows "
             "the +1 increment (not a vacuous pass from e.g. every hop "
             "being a reset)",
             sample_rate, fft_size);
    CHECK(saw_a_plus_one_increment, label);

    four_aec_nr_res_reset(p);
    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: four_aec_nr_res_reset() zeroes "
             "the counter",
             sample_rate, fft_size);
    CHECK(four_aec_nr_res_far_fft_real_compute_count(p) == 0, label);

    free(microphones);
    free(ref);
    free(out);
    free(weights);
    four_aec_nr_res_destroy(p);
}

int main(void) {
    test_projection_kernels();
    test_trusted_spectrum_path();
    test_invalid_configs();
    test_cross_instance_token();
    test_pool_reinit_token_rejected();
    run_grid(16000, 256);
    run_grid(16000, 512);
    run_grid(48000, 1024);
    run_static_parity(16000, 256);
    run_static_parity(16000, 512);
    run_static_parity(48000, 1024);
    test_pre_frame_wola_identity(16000, 256);
    test_pre_frame_wola_identity(16000, 512);
    test_pre_frame_wola_identity(48000, 1024);
    test_far_fft_sharing_reduces_four_to_one(16000, 256);
    test_far_fft_sharing_reduces_four_to_one(16000, 512);
    test_far_fft_sharing_reduces_four_to_one(48000, 1024);

    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All 4aec_nr_res tests passed\n");
    return 0;
}
