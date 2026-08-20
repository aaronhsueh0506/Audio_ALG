/**
 * Structural and lifecycle tests for 4ch_aec_bf_nr_res/4aec_nr_res.h.
 *
 * The test intentionally uses equal weights as an external-beamformer stand
 * in. The library itself must never choose those weights in production.
 */

#include "4aec_nr_res.h"
#include "4aec_nr_res_internal.h"
#include "4aec_projection_kernels.h"
#include "fft_wrapper.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

    /* Backward-jump quarantine. The WINDOW is checked even though the enable
     * defaults to 0: a config that would misbehave the moment someone flips
     * one field must not pass validation today. The accept row at the end is
     * what stops all of this from being a comparison that cannot fail. */
    cfg = four_aec_nr_res_default_config(16000);
    CHECK(cfg.delay_backward_quarantine_enabled == 0 &&
          cfg.delay_backward_quarantine_s == 1.0f,
          "the quarantine defaults to OFF with a 1.0 s window");
    cfg.delay_backward_quarantine_enabled = 2;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "a non-boolean quarantine enable is rejected");

    cfg = four_aec_nr_res_default_config(16000);
    cfg.delay_backward_quarantine_s = -1.0f;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "a negative quarantine window is rejected");
    cfg.delay_backward_quarantine_s = 3601.0f;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "an out-of-range quarantine window is rejected");
    cfg.delay_backward_quarantine_s = (float)NAN;
    CHECK(four_aec_nr_res_create(&cfg) == NULL,
          "a NaN quarantine window is rejected");

    cfg = four_aec_nr_res_default_config(16000);
    cfg.delay_backward_quarantine_enabled = 1;
    cfg.delay_backward_quarantine_s = 0.25f;
    CHECK(four_aec_nr_res_get_mem_requirements(&cfg, &req) == 0,
          "a valid enabled quarantine config is ACCEPTED (so the rows above "
          "are a real check, not an identity)");
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
    /* The superseded version spelled out, not `current - 1`: a descriptor
     * persisted by a version-10 build carries exactly this number, and its
     * byte count is left at the CURRENT figure so the only thing wrong with
     * it is the layout. A control-block-only growth moves no carve token, so
     * build_flags_hash still matches and this counter is the whole signal. */
    stale = req;
    stale.layout_version = 11u;
    CHECK(four_aec_nr_res_init_ex(
              pool, (size_t)req.bytes, &cfg, &stale) == NULL,
          "static init_ex rejects a descriptor from the superseded layout 11 "
          "even when its byte count exactly covers the current pool");
    CHECK(req.layout_version == FOUR_AEC_NR_RES_LAYOUT_VERSION &&
          FOUR_AEC_NR_RES_LAYOUT_VERSION == 12u,
          "the queried descriptor publishes the current carve layout (12)");

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

/* The WOLA identity below has to be measured ACROSS a realign boundary, so
 * its scene is a delayed echo that MOVES rather than the zero-delay tone the
 * rest of the file uses: `changed` on a zero-delay scene realigns by a delta
 * of 0, which aec_apply_external_realign() answers as a no-op, and nothing
 * about seam continuity would have been exercised. `history` holds
 * WOLA_SCENE_PAD samples of pre-roll so the echo is valid from the first
 * streamed sample. */
#define WOLA_SCENE_PAD   16384
#define WOLA_SHIFT_HOP   150
#define WOLA_SCENE_HOPS  600

static void fill_delayed_echo(float* microphones, float* ref, int hop,
                              const float* history, int base, int delay) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        float echo = 0.6f * history[base + i - delay];
        ref[i] = history[base + i];
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                echo * (1.0f + 0.05f * (float)ch);
        }
    }
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
    float* history = NULL;
    uint32_t rng = 0x1234567u;
    float max_error = 0.0f;
    long sweeps_before = 0;
    int applied_before = -1;
    int moving_realigns = 0;
    int lanes_swept_on_moves = 0;
    int rate_scale = sample_rate / 16000;
    int base_delay = 512 * rate_scale;
    int moved_delay = 1024 * rate_scale;
    int valid = 1;
    int hop;
    int n_freqs;
    char label[224];

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
    history = (float*)calloc(
        (size_t)(WOLA_SCENE_HOPS * hop + WOLA_SCENE_PAD), sizeof(float));
    if (!microphones || !ref || !out || !previous || !ola ||
        !time_frame || !weights || !window || !history) {
        valid = 0;
        goto check_result;
    }
    for (int i = 0; i < WOLA_SCENE_HOPS * hop + WOLA_SCENE_PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        history[i] = 0.1f * (((float)(rng >> 8) *
                              (1.0f / 16777216.0f)) - 0.5f);
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

    for (int frame = 0; frame < WOLA_SCENE_HOPS; ++frame) {
        long sweeps_after;
        fill_delayed_echo(
            microphones, ref, hop, history, frame * hop + WOLA_SCENE_PAD,
            frame >= WOLA_SHIFT_HOP ? moved_delay : base_delay);
        if (four_aec_nr_res_process_pre(
                pipeline, microphones, ref, &pre) != FOUR_AEC_NR_RES_OK) {
            valid = 0;
            break;
        }
        /* A delay realignment no longer restarts any WOLA sequence: the
         * lanes realign their filters in place, so the external mirror keeps
         * its OLA running too and this identity proves the seams stay
         * continuous ACROSS the realign boundary, not merely after both
         * sides were wiped. Counted here so the claim rests on realigns this
         * scene actually performed: a generation that MOVES the alignment is
         * the only one that shifts an IR, and each must sweep four lanes. */
        sweeps_after = four_aec_nr_res_realign_warm_lane_count(pipeline) +
                       four_aec_nr_res_realign_soft_lane_count(pipeline);
        if (pre.delay.changed && applied_before >= 0 &&
            pre.delay.delay_samples != applied_before) {
            moving_realigns += 1;
            lanes_swept_on_moves += (int)(sweeps_after - sweeps_before);
        }
        applied_before = pre.delay.delay_samples;
        sweeps_before = sweeps_after;
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
    snprintf(label, sizeof(label),
             "4ch pre time/spectrum seams share one reconstructing WOLA grid "
             "sr=%d/fft=%d (max error %.3g across %d alignment moves)",
             sample_rate, fft_size, (double)max_error, moving_realigns);
    CHECK(valid && max_error <= 1e-4f, label);
    /* Two moves at least: the acquisition off raw far, and the mid-stream
     * shift. Without them the identity above would only have been measured on
     * a stream whose alignment never moved. */
    snprintf(label, sizeof(label),
             "sr=%d/fft=%d: the scene really crosses realign boundaries (%d "
             "alignment moves, %d lane realigns)",
             sample_rate, fft_size, moving_realigns, lanes_swept_on_moves);
    CHECK(moving_realigns >= 2 &&
          lanes_swept_on_moves ==
              moving_realigns * FOUR_AEC_NR_RES_CHANNELS, label);

cleanup:
    free(history);
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

static void test_delay_modes_and_bank_sizing(void) {
    FourAecNrResConfig matched5 = four_aec_nr_res_default_config(16000);
    FourAecNrResConfig matched2 = matched5;
    FourAecNrResConfig fixed = matched5;
    FourAecNrResConfig external = matched5;
    FourAecNrResConfig pre_only;
    FourAecNrResMemReq r5, r2, rf, re, rp;
    FourAecNrRes *pf = NULL, *pe = NULL;
    FourAecNrResPreFrame pre;
    float *mic = NULL, *far = NULL;
    int hop;

    matched5.fft_size = 256;
    matched2.fft_size = 256;
    matched2.delay_num_filters = 2;
    fixed.fft_size = 256;
    fixed.delay_mode = AEC_DELAY_FIXED;
    fixed.fixed_delay_samples = 320;
    external.fft_size = 256;
    external.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    pre_only = external;
    pre_only.enable_post = 0;

    CHECK(four_aec_nr_res_get_mem_requirements(&matched5, &r5) == 0 &&
          four_aec_nr_res_get_mem_requirements(&matched2, &r2) == 0 &&
          four_aec_nr_res_get_mem_requirements(&fixed, &rf) == 0 &&
          four_aec_nr_res_get_mem_requirements(&external, &re) == 0,
          "all three delay modes and a short matched bank size cleanly");
    CHECK(r5.bytes > r2.bytes && r2.bytes > re.bytes,
          "matched-filter bank size changes the queried 4ch pool");
    CHECK(rf.bytes > re.bytes && rf.bytes < r2.bytes,
          "FIXED keeps only its exact reference ring; EXTERNAL has no matcher/ring");
    CHECK(four_aec_nr_res_get_mem_requirements(&pre_only, &rp) == 0 &&
          rp.bytes < re.bytes,
          "pre-only core omits NR, post-RES, FFT/WOLA and post scratch RAM");

    external.delay_num_filters = 2;
    CHECK(four_aec_nr_res_get_mem_requirements(&external, &re) != 0,
          "EXTERNAL rejects an inapplicable matched-filter bank size");
    external.delay_num_filters = DA_NUM_FILTERS;
    fixed.fixed_delay_samples = -1;
    CHECK(four_aec_nr_res_get_mem_requirements(&fixed, &rf) != 0,
          "FIXED rejects an unset fixed delay");
    fixed.fixed_delay_samples = 320;

    pf = four_aec_nr_res_create(&fixed);
    pe = four_aec_nr_res_create(&pre_only);
    CHECK(pf && pe, "FIXED and EXTERNAL instances initialize");
    CHECK(four_aec_nr_res_matched_filter_count(pf) == 0 &&
          four_aec_nr_res_matched_filter_count(pe) == 0,
          "FIXED and EXTERNAL allocate no matched-filter instance");
    CHECK(four_aec_nr_res_nr_count(pe) == 0 &&
          four_aec_nr_res_post_res_count(pe) == 0,
          "pre-only core exposes no post NR/RES instances");
    if (!pf || !pe) goto cleanup;
    hop = four_aec_nr_res_hop_size(pe);
    mic = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS,
                         sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    CHECK(mic && far, "delay-mode smoke buffers allocate");
    if (!mic || !far) goto cleanup;

    fill_inputs(mic, far, hop, 16000, 0);
    CHECK(four_aec_nr_res_process_pre(pe, mic, far, &pre) ==
              FOUR_AEC_NR_RES_OK && pre.delay.solid &&
              pre.delay.delay_samples == 0,
          "EXTERNAL consumes caller-aligned far without acquisition");
    CHECK(four_aec_nr_res_abandon_pre(pe, &pre.token) ==
              FOUR_AEC_NR_RES_OK,
          "EXTERNAL pre frame releases");
    CHECK(four_aec_nr_res_process_pre(pf, mic, far, &pre) ==
              FOUR_AEC_NR_RES_OK && !pre.delay.solid &&
              pre.delay.delay_samples == 320,
          "FIXED reports the configured delay and ring warm-up state");
    CHECK(four_aec_nr_res_abandon_pre(pf, &pre.token) ==
              FOUR_AEC_NR_RES_OK,
          "FIXED pre frame releases");

cleanup:
    free(mic);
    free(far);
    four_aec_nr_res_destroy(pf);
    four_aec_nr_res_destroy(pe);
}

/* ============================================================================
 * Known-delay profile verification (product delay gate, NOT an audio-quality
 * run -- see docs/align_ulcnet_delay_profile_plan_zh_TW.md §5.1/§6).
 *
 * Everything here is driven by a SYNTHESISED echo whose bulk delay is known
 * exactly, so every claim is checked against ground truth instead of against
 * a score. Four questions, in the order the plan asks them:
 *
 *   acquisition  Does each bank size acquire the delays it is supposed to
 *                cover, how fast, and how accurately?
 *   coverage     Does the reliable-range contract per n actually hold at the
 *                boundary -- locks just inside, does NOT lock just outside?
 *   mislock      When the true bulk delay is beyond the bank's reach, is the
 *                failure DETECTABLE? (It is not detectable from the seam.)
 *   cost         Pool bytes vs n, and a rough per-hop CPU figure.
 * ========================================================================== */

/* Reliable bulk-delay search ceiling for a bank of n matched filters, in
 * native 16 kHz samples, spelled with lib/aec's own constants: the ring
 * holds (n-1)*DA_FILTER_INTRA_SHIFT + (DA_FILTER_SIZE - 11) DOWNSAMPLED
 * samples of reach (the -11 is the `lag < filter_size - 10` reliability
 * cut), and one downsampled sample is DA_DOWN_SAMPLING_FACTOR native
 * samples (0.25 ms at 16 kHz). That is the 125/221/317/413/509 ms table in
 * the plan, derived rather than copied as five literals. */
#define KD_RELIABLE_SAMPLES(n) \
    (((n) - 1) * DA_FILTER_INTRA_SHIFT * DA_DOWN_SAMPLING_FACTOR + \
     (DA_FILTER_SIZE - 11) * DA_DOWN_SAMPLING_FACTOR)

/* The applied alignment must never land LATER than the true echo: the AEC3
 * design deliberately reports early so PBFDKF sees a POSITIVE residual it can
 * model (a negative residual is non-causal for the filter and cannot be).
 * Measured across n=1..5 and true delays 4..509 ms, the shortfall is 64 or 80
 * samples -- one estimator headroom (32) plus pre-echo block quantisation.
 * 128 samples (8 ms) leaves margin without admitting a whole extra hop. */
#define KD_MAX_UNDERSHOOT 128

typedef struct KnownDelayRun {
    int locked;
    int lock_hop;
    int applied_delay;
    float confidence;
    int changed_events;
    int first_changed_hop;
    /* Hops on which the published `solid` disagreed with the acceptance
     * predicate recomputed from the seam's OWN estimator fields, and hops on
     * which `solid` was already 1 before any alignment generation had been
     * accepted. Both must stay 0: see the invariant note above the recompute
     * in known_delay_run(). */
    int solid_disagreements;
    int solid_before_accept;
    double us_per_hop;
} KnownDelayRun;

/* Drives `hops` hops of a two-path synthetic echo (a dominant path at
 * dominant_delay plus an optional weaker early path) through a MATCHED core
 * with the given bank size, and reports what the shared estimator did. */
static void known_delay_run(int num_filters, int hops,
                            int dominant_delay, float dominant_gain,
                            int early_delay, float early_gain,
                            KnownDelayRun* out) {
    enum { KD_HOP = 256, KD_PAD = 16384 };
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* p;
    float* far_hist;
    float mic[KD_HOP * FOUR_AEC_NR_RES_CHANNELS];
    float far[KD_HOP];
    uint32_t rng = 0x1234567u;
    clock_t t0, t1;
    int hop, i, ch;
    int expect_solid = 0;

    memset(out, 0, sizeof(*out));
    out->lock_hop = -1;
    out->applied_delay = -1;
    out->first_changed_hop = -1;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    cfg.enable_cng = 0;
    cfg.delay_num_filters = num_filters;
    p = four_aec_nr_res_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * KD_HOP + KD_PAD) * sizeof(float));
    if (!far_hist) { four_aec_nr_res_destroy(p); return; }
    for (i = 0; i < hops * KD_HOP + KD_PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        far_hist[i] = 0.25f * (((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
    }

    t0 = clock();
    for (hop = 0; hop < hops; ++hop) {
        FourAecNrResPreFrame pre;
        int base = hop * KD_HOP + KD_PAD;
        for (i = 0; i < KD_HOP; ++i) {
            int t = base + i;
            float echo = dominant_gain * far_hist[t - dominant_delay] +
                         early_gain * far_hist[t - early_delay];
            far[i] = far_hist[t];
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = echo;
        }
        if (four_aec_nr_res_process_pre(p, mic, far, &pre) !=
            FOUR_AEC_NR_RES_OK) break;
        /* The published `solid` must be exactly "a usable accepted alignment
         * generation exists", recomputed here from the OTHER published
         * estimator fields rather than read back from `solid` itself:
         * confidence >= 1.0 IS delay_aec3_is_solid(), estimator_updates IS
         * delay_aec3_n_updates(), and the remaining acceptance term
         * (estimated >= 0) is implied by confidence >= 1.0 -- a confidence
         * above 0 requires a produced estimate, and a produced estimate is
         * never negative. Sticky, because a confidence dip must not retract
         * an alignment the audio path is still applying.
         *
         * MUTATIONS: publishing `solid` on COARSE confidence (`state.solid =
         * now_usable || delay_aec3_confidence(...) >= 0.5f`) makes it lead
         * the accepted delay and BOTH counters below go non-zero; widening
         * the wrapper's own `n_updates >= 3` acceptance term (e.g. to 30)
         * makes it lag instead, and solid_disagreements alone goes
         * non-zero. */
        if (pre.delay.changed) {
            out->changed_events += 1;
            if (out->first_changed_hop < 0) out->first_changed_hop = hop;
        }
        {
            int accepted_now = pre.delay.confidence >= 1.0f &&
                               pre.delay.estimator_updates >= 3;
            expect_solid = expect_solid || accepted_now;
            if (pre.delay.solid != expect_solid) out->solid_disagreements += 1;
            if (pre.delay.solid && out->first_changed_hop < 0)
                out->solid_before_accept += 1;
        }
        if (!out->locked && pre.delay.solid) {
            out->locked = 1;
            out->lock_hop = hop;
            out->applied_delay = pre.delay.delay_samples;
            out->confidence = pre.delay.confidence;
        }
        four_aec_nr_res_abandon_pre(p, &pre.token);
    }
    t1 = clock();
    out->us_per_hop = hops > 0
        ? (double)(t1 - t0) * 1e6 / (double)CLOCKS_PER_SEC / (double)hops
        : 0.0;

    free(far_hist);
    four_aec_nr_res_destroy(p);
}

static void test_known_delay_acquisition_and_coverage(void) {
    /* One point just INSIDE each bank's reliable ceiling and one just
     * OUTSIDE it (the next bank's inside point, which is by construction
     * beyond this bank's reach). */
    static const int inside_ms[6] = { 0, 125, 221, 317, 413, 509 };
    int n;

    printf("known-delay acquisition (16 kHz, hop 256, synthetic echo):\n");
    for (n = 1; n <= 5; ++n) {
        KnownDelayRun in_range, out_of_range;
        int inside = inside_ms[n] * 16;             /* ms -> samples @16k */
        /* n=5 has no next bank; 9271 samples = 579.44 ms is the real
         * recording that first exposed its ceiling. */
        int outside = (n == 5) ? 9271 : inside_ms[n + 1] * 16;
        char label[192];

        known_delay_run(n, 200, inside, 0.6f, inside, 0.0f, &in_range);
        known_delay_run(n, 200, outside, 0.6f, outside, 0.0f, &out_of_range);

        printf("  n=%d ceiling %d samples (%.2f ms): "
               "in-range %d ms -> lock hop %d applied %d (short by %d); "
               "out-of-range %d ms -> %s\n",
               n, KD_RELIABLE_SAMPLES(n), KD_RELIABLE_SAMPLES(n) / 16.0,
               inside_ms[n], in_range.lock_hop, in_range.applied_delay,
               in_range.locked ? inside - in_range.applied_delay : -1,
               outside / 16,
               out_of_range.locked ? "LOCKED" : "no lock");

        snprintf(label, sizeof(label),
                 "n=%d acquires a %d ms bulk delay within 60 hops (lock hop "
                 "%d, inside its %.2f ms ceiling)", n, inside_ms[n],
                 in_range.lock_hop, KD_RELIABLE_SAMPLES(n) / 16.0);
        CHECK(in_range.locked && in_range.lock_hop >= 0 &&
              in_range.lock_hop < 60, label);

        /* The alignment contract, not a tolerance pulled from the air: the
         * applied delay may sit EARLY of the true echo (PBFDKF then models a
         * positive residual) but must never sit LATE. */
        snprintf(label, sizeof(label),
                 "n=%d applied delay is early-or-exact and short by at most "
                 "%d samples (true %d, applied %d)",
                 n, KD_MAX_UNDERSHOOT, inside, in_range.applied_delay);
        CHECK(in_range.locked &&
              inside - in_range.applied_delay >= 0 &&
              inside - in_range.applied_delay <= KD_MAX_UNDERSHOOT, label);

        snprintf(label, sizeof(label),
                 "n=%d exactly one alignment generation for a static delay "
                 "(%d)", n, in_range.changed_events);
        CHECK(in_range.changed_events == 1, label);

        /* `solid` publishes "a usable accepted alignment exists", so it can
         * never lead the accepted delay: the hop it first goes 1 is exactly
         * the hop the generation is accepted on, and it agrees with the
         * acceptance predicate on EVERY hop of the run (in range and out of
         * range alike -- the out-of-range run must stay 0/0 too, which is
         * what stops these from being vacuous). */
        snprintf(label, sizeof(label),
                 "n=%d `solid` never leads the accepted delay (first solid "
                 "hop %d, first alignment generation hop %d, %d early-solid "
                 "hops)", n, in_range.lock_hop, in_range.first_changed_hop,
                 in_range.solid_before_accept);
        CHECK(in_range.solid_before_accept == 0 &&
              in_range.first_changed_hop == in_range.lock_hop, label);
        snprintf(label, sizeof(label),
                 "n=%d published `solid` equals the sticky acceptance "
                 "predicate on every hop (%d in-range and %d out-of-range "
                 "disagreements)", n, in_range.solid_disagreements,
                 out_of_range.solid_disagreements);
        CHECK(in_range.solid_disagreements == 0 &&
              out_of_range.solid_disagreements == 0, label);

        snprintf(label, sizeof(label),
                 "n=%d does NOT acquire a %d ms bulk delay (beyond its "
                 "%.2f ms ceiling)", n, outside / 16,
                 KD_RELIABLE_SAMPLES(n) / 16.0);
        CHECK(outside > KD_RELIABLE_SAMPLES(n) && !out_of_range.locked, label);
    }
}

/* An out-of-range bulk delay is NOT always a clean "never locks". Add any
 * in-range echo component -- an early reflection, a second speaker path, a
 * codec artefact -- and the estimator locks onto THAT with full confidence
 * while the dominant path stays unmodelled far outside the filter's reach.
 * This reproduces, synthetically, the recording that exposed the n=5 ceiling:
 * a dominant path at 579.44 ms and a confident lock at ~32 ms.
 *
 * This test does NOT bless that behaviour. It pins the property a product
 * delay gate has to be built around: the seam cannot see the error (solid=1,
 * confidence=1.0, exactly as for a correct lock), so the ONLY thing that
 * catches it is comparing the applied delay against an independently known
 * ground-truth delay. That comparison is the check being asserted here. */
static void test_known_delay_mislock_is_detectable(void) {
    KnownDelayRun mislock, control;
    const int dominant = 9271;      /* 579.44 ms -- beyond every bank */
    const int early = 512;          /* 32 ms -- inside every bank */
    int error_samples;
    char label[224];

    known_delay_run(5, 200, dominant, 0.6f, early, 0.5f, &mislock);
    /* Control: same two paths, but the dominant one now inside the ceiling.
     * The estimator must prefer the dominant path, and the ground-truth
     * check must stay quiet -- otherwise the check below is just "always
     * flags", which would prove nothing. */
    known_delay_run(5, 200, 3536, 0.6f, early, 0.5f, &control);

    error_samples = mislock.locked ? dominant - mislock.applied_delay : -1;
    printf("known-delay mislock: dominant %d samples (%.2f ms) + early %d "
           "(%.2f ms) -> lock hop %d applied %d (%.2f ms) confidence %.2f, "
           "wrong by %d samples (%.2f ms)\n",
           dominant, dominant / 16.0, early, early / 16.0,
           mislock.lock_hop, mislock.applied_delay,
           mislock.applied_delay / 16.0, (double)mislock.confidence,
           error_samples, error_samples / 16.0);

    CHECK(mislock.locked,
          "an in-range early path makes an out-of-range bulk delay lock "
          "anyway");
    CHECK(mislock.confidence >= 1.0f,
          "the mislock is reported at FULL confidence -- the seam's own "
          "fields cannot distinguish it from a correct lock");
    snprintf(label, sizeof(label),
             "ground-truth comparison FLAGS the mislock: applied is short by "
             "%d samples (%.2f ms), far past the %d-sample alignment "
             "contract", error_samples, error_samples / 16.0,
             KD_MAX_UNDERSHOOT);
    CHECK(mislock.locked && error_samples > KD_MAX_UNDERSHOOT, label);

    snprintf(label, sizeof(label),
             "control: with the dominant path in range the SAME check stays "
             "quiet (applied %d, short by %d)",
             control.applied_delay,
             control.locked ? 3536 - control.applied_delay : -1);
    CHECK(control.locked &&
          3536 - control.applied_delay >= 0 &&
          3536 - control.applied_delay <= KD_MAX_UNDERSHOOT, label);
}

/* Plan §8.6: report the 4ch numbers by CALLING the 4ch queries. n drives the
 * ONE shared estimator in the wrapper; the four lanes are EXTERNAL_ALIGNED
 * and must not move with it, so the mono "5,728 B per filter per AEC
 * instance" contract must NOT be multiplied by four here. */
static void test_known_delay_memory_and_cost(void) {
    FourAecNrResMemBreakdown b[6];
    FourAecNrResMemReq req[6];
    KnownDelayRun cost;
    int n;
    int ok = 1;
    char label[192];

    for (n = 1; n <= 5; ++n) {
        FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
        cfg.fft_size = 512;
        cfg.enable_cng = 0;
        cfg.delay_num_filters = n;
        if (four_aec_nr_res_get_mem_breakdown(&cfg, &b[n]) != 0 ||
            four_aec_nr_res_get_mem_requirements(&cfg, &req[n]) != 0) {
            ok = 0;
            break;
        }
    }
    CHECK(ok, "4ch memory breakdown and pool query answer for every n=1..5");
    if (!ok) return;

    printf("4ch pool vs matched-filter bank size (16 kHz, fft 512):\n");
    for (n = 1; n <= 5; ++n) {
        printf("  n=%d  lanes(4x AEC) %9zu  wrapper(shared est + ring + "
               "bufs) %9zu  total %9llu",
               n, b[n].aec_bytes, b[n].wrapper_bytes,
               (unsigned long long)req[n].bytes);
        if (n > 1)
            printf("  (+%lld vs n=%d)",
                   (long long)req[n].bytes - (long long)req[n - 1].bytes, n - 1);
        printf("\n");
    }

    for (n = 2; n <= 5; ++n) {
        snprintf(label, sizeof(label),
                 "n=%d leaves the four lane AEC pools untouched (%zu bytes, "
                 "same as n=1) -- n lives in the ONE shared estimator",
                 n, b[n].aec_bytes);
        CHECK(b[n].aec_bytes == b[1].aec_bytes, label);

        snprintf(label, sizeof(label),
                 "n=%d costs strictly more wrapper RAM than n=%d "
                 "(%zu > %zu)", n, n - 1, b[n].wrapper_bytes,
                 b[n - 1].wrapper_bytes);
        CHECK(b[n].wrapper_bytes > b[n - 1].wrapper_bytes, label);

        snprintf(label, sizeof(label),
                 "n=%d total pool grows by exactly the per-filter cost of "
                 "one shared estimator bank (%lld bytes, same as every other "
                 "step)", n,
                 (long long)req[n].bytes - (long long)req[n - 1].bytes);
        CHECK((long long)req[n].bytes - (long long)req[n - 1].bytes ==
              (long long)req[2].bytes - (long long)req[1].bytes, label);

        snprintf(label, sizeof(label),
                 "n=%d breakdown total agrees with the pool query (%zu == "
                 "%llu)", n, b[n].total_bytes,
                 (unsigned long long)req[n].bytes);
        CHECK((unsigned long long)b[n].total_bytes ==
              (unsigned long long)req[n].bytes, label);
    }

    /* Rough per-hop CPU, recorded rather than tuned: one hop is 256 samples
     * = 16 ms of audio at 16 kHz, so the real-time factor is
     * us_per_hop / 16000. The bound is a liveness guard (a catastrophic
     * regression), not a performance target -- this is a host measurement on
     * whatever machine ran the suite, not a board number. */
    known_delay_run(5, 400, 3536, 0.6f, 3536, 0.0f, &cost);
    printf("known-delay cost: 4 lanes + shared estimator (n=5) = "
           "%.1f us/hop, %.4f x real time (hop = 16.00 ms of audio)\n",
           cost.us_per_hop, cost.us_per_hop / 16000.0);
    snprintf(label, sizeof(label),
             "4ch core runs faster than real time on the host (%.1f us/hop "
             "vs a 16000 us budget)", cost.us_per_hop);
    CHECK(cost.us_per_hop > 0.0 && cost.us_per_hop < 16000.0, label);
}

/* ============================================================================
 * Shared-delay change admission and the four-lane realign sweep
 *
 * A published `changed` sweeps aec_apply_external_realign() over all four
 * lanes: an IR shift, plus (when the alignment retards) a far-history clear,
 * on every one of them. What the rows below pin is that the sweep happens on
 * exactly the hops it should, sized exactly four, and that the applied
 * alignment and the sweep are one event -- the delay may never move on a hop
 * that publishes no `changed`, or the lanes would be handed a reference
 * shifted out from under filters nothing had realigned.
 *
 * Two things absorb movement before the lanes ever see it, and they sit at
 * different scales:
 *   - DelayAec3 publishes on a 16-downsampled-sample grid (64 native samples
 *     at 16 kHz), so a bulk-delay wander finer than that never reaches the
 *     wrapper at all -- the WANDER row measures that end of the seam;
 *   - the wrapper's own admission (> 32 samples AND repeated within 16 on the
 *     next eligible hop) then holds every movement for one hop before it can
 *     spend a sweep -- the MOVING and SHIFT rows measure that end.
 * ========================================================================== */

typedef struct RealignRun {
    int acquisition_hop;        /* first `changed` (the acquisition)         */
    int later_changes;          /* `changed` hops after the acquisition      */
    int last_change_hop;
    int applied_at_end;
    int hold_hops;              /* hops holding an unconfirmed candidate     */
    int unconfirmed_changes;    /* accepted with no candidate held first     */
    int silent_delay_moves;     /* applied delay moved without a `changed`   */
    int wrong_sized_sweeps;     /* a `changed` hop that did not realign 4    */
    int sweeps_without_change;  /* lanes realigned on a hop with no `changed`*/
    long warm;
    long soft;
} RealignRun;

/* Shared synthesis for the realign/admission scenes: xorshift far history
 * and one hop of 4-lane echo at the given dominant delay (optional
 * simultaneous second path at delay_b when second_gain > 0). */
static void realign_scene_history(float* far_hist, int count) {
    uint32_t rng = 0x1234567u;
    int i;
    for (i = 0; i < count; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        far_hist[i] = 0.25f *
            (((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
    }
}

static void realign_scene_hop(const float* far_hist, int base, int hop_len,
                              int dominant, float second_gain, int delay_b,
                              float* mic, float* far) {
    int i, ch;
    for (i = 0; i < hop_len; ++i) {
        int t = base + i;
        float echo = 0.6f * far_hist[t - dominant];
        if (second_gain > 0.0f) echo += second_gain * far_hist[t - delay_b];
        far[i] = far_hist[t];
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = echo;
    }
}

/* Drives one 16 kHz/hop-256 MATCHED scene and reports what the shared
 * alignment did. The echo is a single path at `delay_a`, except:
 *   second_gain > 0  adds a SIMULTANEOUS second path at delay_b (an
 *                    ambiguous scene: the estimator's peak moves between the
 *                    two on its own, which is what real jitter looks like at
 *                    this seam);
 *   shift_at >= 0    moves the single path to delay_b from that hop on;
 *   wander_hops > 0  alternates the single path between delay_a and delay_b
 *                    every wander_hops hops. */
static void realign_run(int hops, int delay_a, int delay_b, float second_gain,
                        int shift_at, int wander_hops, RealignRun* out) {
    enum { RA_HOP = 256, RA_PAD = 16384 };
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* p;
    float* far_hist;
    float mic[RA_HOP * FOUR_AEC_NR_RES_CHANNELS];
    float far[RA_HOP];
    int hop;
    int previous_delay = 0;
    int previous_pending = -1;
    long previous_sweeps = 0;

    memset(out, 0, sizeof(*out));
    out->acquisition_hop = -1;
    out->last_change_hop = -1;
    out->applied_at_end = -1;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    cfg.enable_cng = 0;
    p = four_aec_nr_res_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * RA_HOP + RA_PAD) * sizeof(float));
    if (!far_hist) { four_aec_nr_res_destroy(p); return; }
    realign_scene_history(far_hist, hops * RA_HOP + RA_PAD);

    for (hop = 0; hop < hops; ++hop) {
        FourAecNrResPreFrame pre;
        int base = hop * RA_HOP + RA_PAD;
        int dominant = delay_a;
        int pending;
        long sweeps;
        if (shift_at >= 0 && hop >= shift_at) dominant = delay_b;
        if (wander_hops > 0 && ((hop / wander_hops) % 2)) dominant = delay_b;
        realign_scene_hop(far_hist, base, RA_HOP, dominant, second_gain,
                          delay_b, mic, far);
        if (four_aec_nr_res_process_pre(p, mic, far, &pre) !=
            FOUR_AEC_NR_RES_OK) break;

        sweeps = four_aec_nr_res_realign_warm_lane_count(p) +
                 four_aec_nr_res_realign_soft_lane_count(p);
        pending = four_aec_nr_res_pending_delay_candidate(p);
        if (pending >= 0) out->hold_hops += 1;

        if (pre.delay.changed) {
            if (out->acquisition_hop < 0) {
                out->acquisition_hop = hop;
            } else {
                out->later_changes += 1;
                /* The candidate must have been HELD on the previous hop and
                 * must be the value now applied: that pair is the whole
                 * two-step rule, and it is what disappears if acceptance
                 * moves back to the first sighting. */
                if (previous_pending < 0 ||
                    abs(previous_pending - pre.delay.delay_samples) >= 16)
                    out->unconfirmed_changes += 1;
            }
            out->last_change_hop = hop;
            if (sweeps - previous_sweeps != FOUR_AEC_NR_RES_CHANNELS)
                out->wrong_sized_sweeps += 1;
        } else {
            if (hop > 0 && pre.delay.delay_samples != previous_delay)
                out->silent_delay_moves += 1;
            if (sweeps != previous_sweeps) out->sweeps_without_change += 1;
        }

        previous_delay = pre.delay.delay_samples;
        previous_pending = pending;
        previous_sweeps = sweeps;
        out->applied_at_end = pre.delay.delay_samples;
        four_aec_nr_res_abandon_pre(p, &pre.token);
    }

    out->warm = four_aec_nr_res_realign_warm_lane_count(p);
    out->soft = four_aec_nr_res_realign_soft_lane_count(p);
    free(far_hist);
    four_aec_nr_res_destroy(p);
}

static void test_shared_delay_change_admission(void) {
    RealignRun wander, moving, shift;
    char label[224];

    /* WANDER: the bulk delay itself moves by 24 samples (1.5 ms) back and
     * forth, below both the wrapper's 32-sample admission and DelayAec3's own
     * 64-sample output grid. Nothing may reach the lanes: one acquisition
     * sweep for the whole run and an alignment that never moves again. */
    realign_run(240, 512, 536, 0.0f, -1, 30, &wander);
    printf("shared-delay admission: wander +/-24 -> acquired hop %d applied "
           "%d, %d later generations, %ld warm + %ld soft lane realigns\n",
           wander.acquisition_hop, wander.applied_at_end, wander.later_changes,
           wander.warm, wander.soft);
    snprintf(label, sizeof(label),
             "a 24-sample bulk-delay wander starts no alignment generation "
             "after the acquisition (%d later generations)",
             wander.later_changes);
    CHECK(wander.acquisition_hop >= 0 && wander.later_changes == 0, label);
    snprintf(label, sizeof(label),
             "and realigns the four lanes exactly once, for the acquisition "
             "itself (%ld warm + %ld soft = %ld lane calls)",
             wander.warm, wander.soft, wander.warm + wander.soft);
    CHECK(wander.warm + wander.soft == FOUR_AEC_NR_RES_CHANNELS, label);

    /* MOVING: the echo path really does travel, 512 <-> 1024 samples in
     * 100-hop blocks, so the shared estimate moves several times over the run
     * -- and every move is far larger than the admission band, which is what
     * a movement the lanes SHOULD follow looks like. This is the row that
     * makes the two-step rule falsifiable: a single-generation scene would
     * pass the hold assertions for free. */
    realign_run(600, 512, 1024, 0.0f, -1, 100, &moving);
    printf("shared-delay admission: 512<->1024 in 100-hop blocks -> acquired "
           "hop %d, %d later generations, %d holds, applied %d, %ld warm + "
           "%ld soft\n", moving.acquisition_hop, moving.later_changes,
           moving.hold_hops, moving.applied_at_end, moving.warm, moving.soft);
    snprintf(label, sizeof(label),
             "control: the moving-path scene really does re-lock several "
             "times (%d later generations)", moving.later_changes);
    CHECK(moving.later_changes >= 4, label);
    snprintf(label, sizeof(label),
             "every alignment change is admitted only after the same value is "
             "seen on a second eligible hop (%d accepted on a first sighting, "
             "%d hold hops for %d changes)", moving.unconfirmed_changes,
             moving.hold_hops, moving.later_changes);
    CHECK(moving.unconfirmed_changes == 0 &&
          moving.hold_hops == moving.later_changes, label);
    snprintf(label, sizeof(label),
             "the applied alignment never moves on a hop that publishes no "
             "`changed` (%d silent moves)", moving.silent_delay_moves);
    CHECK(moving.silent_delay_moves == 0, label);
    snprintf(label, sizeof(label),
             "every generation realigns exactly four lanes and no other hop "
             "realigns any (%d wrong-sized sweeps, %d sweeps without a "
             "generation)", moving.wrong_sized_sweeps,
             moving.sweeps_without_change);
    CHECK(moving.wrong_sized_sweeps == 0 &&
          moving.sweeps_without_change == 0, label);
    snprintf(label, sizeof(label),
             "the warm/soft split accounts for every lane call (%ld + %ld == "
             "4 x %d generations)", moving.warm, moving.soft,
             moving.later_changes + 1);
    CHECK(moving.warm + moving.soft ==
          (long)FOUR_AEC_NR_RES_CHANNELS * (moving.later_changes + 1),
          label);

    /* SHIFT: the echo really moves, 512 -> 1024 samples, and stays there. One
     * generation, held for one hop first, then all four lanes realigned. */
    realign_run(260, 512, 1024, 0.0f, 100, 0, &shift);
    printf("shared-delay admission: shift 512 -> 1024 at hop 100 -> acquired "
           "hop %d, adopted hop %d applied %d, %ld warm + %ld soft\n",
           shift.acquisition_hop, shift.last_change_hop, shift.applied_at_end,
           shift.warm, shift.soft);
    snprintf(label, sizeof(label),
             "a sustained 512-sample shift starts exactly one new alignment "
             "generation (%d), on hop %d", shift.later_changes,
             shift.last_change_hop);
    CHECK(shift.later_changes == 1 && shift.last_change_hop > 100, label);
    snprintf(label, sizeof(label),
             "it is held for confirmation for exactly one hop before it is "
             "applied (%d holds, %d accepted on a first sighting)",
             shift.hold_hops, shift.unconfirmed_changes);
    CHECK(shift.hold_hops == 1 && shift.unconfirmed_changes == 0, label);
    snprintf(label, sizeof(label),
             "and then realigns all four lanes on that one hop (%ld warm + "
             "%ld soft = %ld, two generations x 4 lanes)",
             shift.warm, shift.soft, shift.warm + shift.soft);
    CHECK(shift.wrong_sized_sweeps == 0 &&
          shift.sweeps_without_change == 0 &&
          shift.warm + shift.soft == 2 * FOUR_AEC_NR_RES_CHANNELS, label);
    snprintf(label, sizeof(label),
             "the alignment ends on the moved path, early-or-exact within the "
             "%d-sample contract (applied %d for a true 1024)",
             KD_MAX_UNDERSHOOT, shift.applied_at_end);
    CHECK(1024 - shift.applied_at_end >= 0 &&
          1024 - shift.applied_at_end <= KD_MAX_UNDERSHOOT, label);
}

/* ============================================================================
 * Held-candidate lifetime
 *
 * The admission state machine is driven directly here. Through a stream it
 * cannot be: DelayAec3 re-offers a movement on every hop once it has one, so
 * the candidate is always resolved on the very next eligible hop and its TTL
 * never runs out -- the same reason the expiry rule exists at all is the
 * reason a synthetic scene cannot exercise it. What a scene DOES cover (a
 * candidate is really held, and really spends a realign when confirmed) is
 * asserted in test_shared_delay_change_admission() above; the reset row below
 * closes the loop by clearing a candidate that a real stream produced.
 * ========================================================================== */

/* One hop, spent the way update_shared_delay() spends it: the held candidate
 * always ages, and the estimate is only offered on a hop the estimator was
 * eligible on. */
static int admission_hop(FourAecDelayAdmission* admission, int accepted,
                         int estimated, int eligible) {
    four_aec_nr_res_admission_age(admission);
    if (!eligible) return 0;
    return four_aec_nr_res_admission_offer(admission, accepted, estimated);
}

static void test_delay_change_candidate_ttl(void) {
    const int applied = 512;        /* the alignment in force */
    const int moved = 1024;         /* a movement far outside the band */
    FourAecDelayAdmission a;
    FourAecNrResConfig cfg;
    FourAecNrRes* p;
    char label[224];
    int i;
    int held_hop = -1;
    int candidate_before_reset = -1;
    int candidate_after_reset = 0;

    /* Offered once, repeated on the very next hop: admitted, and nothing is
     * left held afterwards. */
    memset(&a, 0, sizeof(a));
    CHECK(admission_hop(&a, applied, moved, 1) == 0 &&
          a.ttl == FOUR_DELAY_CHANGE_CANDIDATE_TTL && a.candidate == moved,
          "a first sighting is held, not applied, with a full life");
    CHECK(admission_hop(&a, applied, moved, 1) == 1 &&
          a.ttl == 0,
          "the same movement on the next hop is admitted and releases the "
          "candidate");

    /* One hop without a usable estimate does not end it: lib/aec's rule is a
     * bounded life, not a strictly consecutive pair. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    CHECK(admission_hop(&a, applied, 0, 0) == 0 && a.ttl > 0,
          "a hop with no usable estimate spends one hop of the candidate's "
          "life and keeps it");
    CHECK(admission_hop(&a, applied, moved, 1) == 1,
          "the movement is still admitted when it returns inside that life");

    /* Aged out: the candidate is gone, and a single reappearance can only
     * start a new one. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    for (i = 0; i < FOUR_DELAY_CHANGE_CANDIDATE_TTL - 1; ++i)
        admission_hop(&a, applied, 0, 0);
    snprintf(label, sizeof(label),
             "the candidate survives exactly %d hops without a usable "
             "estimate (life left %d)",
             FOUR_DELAY_CHANGE_CANDIDATE_TTL - 1, a.ttl);
    CHECK(a.ttl > 0 && a.candidate == moved, label);
    admission_hop(&a, applied, 0, 0);
    CHECK(a.ttl == 0 && a.candidate == 0,
          "one hop later it has expired, holding nothing");
    CHECK(admission_hop(&a, applied, moved, 1) == 0 &&
          a.ttl == FOUR_DELAY_CHANGE_CANDIDATE_TTL,
          "a single reappearance after expiry is NOT admitted -- it is only a "
          "new first sighting");
    CHECK(admission_hop(&a, applied, moved, 1) == 1,
          "and it takes a repeat inside the new life to admit it");

    /* An estimate back at the alignment in force is absorbed without ending
     * the candidate (lib/aec's Path B has no such clear); a movement to a
     * different place replaces it with a full life. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    CHECK(admission_hop(&a, applied,
                        applied + FOUR_DELAY_CHANGE_MIN_SAMPLES, 1) == 0 &&
          a.candidate == moved && a.ttl > 0,
          "an estimate inside the admission band is absorbed and leaves the "
          "held candidate to age");
    CHECK(admission_hop(&a, applied, 2048, 1) == 0 &&
          a.candidate == 2048 &&
          a.ttl == FOUR_DELAY_CHANGE_CANDIDATE_TTL,
          "a movement somewhere else entirely replaces the candidate and "
          "restarts its life");

    /* End to end: a candidate a real stream produced is cleared by reset(),
     * together with the life left on it. */
    cfg = four_aec_nr_res_default_config(16000);
    cfg.fft_size = 512;
    cfg.enable_cng = 0;
    p = four_aec_nr_res_create(&cfg);
    CHECK(p != NULL, "candidate-reset scene creates");
    if (p) {
        enum { TR_HOP = 256, TR_PAD = 16384, TR_HOPS = 600 };
        float* far_hist = (float*)malloc(
            (size_t)(TR_HOPS * TR_HOP + TR_PAD) * sizeof(float));
        float mic[TR_HOP * FOUR_AEC_NR_RES_CHANNELS];
        float far[TR_HOP];
        int hop;
        if (far_hist) {
            realign_scene_history(far_hist, TR_HOPS * TR_HOP + TR_PAD);
            for (hop = 0; hop < TR_HOPS && held_hop < 0; ++hop) {
                FourAecNrResPreFrame pre;
                int base = hop * TR_HOP + TR_PAD;
                int dominant = ((hop / 100) % 2) ? 1024 : 512;
                realign_scene_hop(far_hist, base, TR_HOP, dominant, 0.0f, 0,
                                  mic, far);
                if (four_aec_nr_res_process_pre(p, mic, far, &pre) !=
                    FOUR_AEC_NR_RES_OK) break;
                four_aec_nr_res_abandon_pre(p, &pre.token);
                candidate_before_reset =
                    four_aec_nr_res_pending_delay_candidate(p);
                if (candidate_before_reset >= 0) {
                    held_hop = hop;
                    four_aec_nr_res_reset(p);
                    candidate_after_reset =
                        four_aec_nr_res_pending_delay_candidate(p);
                }
            }
        }
        free(far_hist);
        four_aec_nr_res_destroy(p);
    }
    snprintf(label, sizeof(label),
             "reset() clears a candidate a real stream was holding (hop %d, "
             "held %d, after reset %d)",
             held_hop, candidate_before_reset, candidate_after_reset);
    CHECK(held_hop >= 0 && candidate_before_reset >= 0 &&
          candidate_after_reset == -1, label);
}


/* ============================================================================
 * Runtime strength control
 *
 * The load-bearing property: the four lanes run with spatial_linear_context
 * and therefore never reach suppression_gain_get_gain(). A preset change
 * pushed at them is a provable no-op, so this core's setter has to move the
 * SHARED post-stage suppressor -- and both of its config copies, because
 * post_sg_cfg is a separate by-value snapshot that reset_post_sg() re-applies.
 * A setter that updated only post_sg.cfg would work until the first reset and
 * then silently revert, which is exactly what test 3 below pins.
 * ========================================================================== */

static void feed_strength_hops(FourAecNrRes* p, int hops) {
    int hop = four_aec_nr_res_hop_size(p);
    int n = four_aec_nr_res_n_freqs(p);
    float* mics = (float*)calloc((size_t)hop * 4u, sizeof(float));
    float* far = (float*)calloc((size_t)hop, sizeof(float));
    float* out = (float*)calloc((size_t)hop, sizeof(float));
    /* Channel-major Complex[4][n_freqs] -- the weights contract in
     * 4aec_nr_res.h. Sizing this as [n_freqs] would have process_post() read
     * three channels past the end. */
    Complex* w = (Complex*)calloc(
        (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n, sizeof(Complex));
    FourAecNrResPreFrame pre;
    int h, i, ch, k;
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
        for (k = 0; k < n; ++k) {
            w[(size_t)ch * (size_t)n + (size_t)k].r = 0.25f;
            w[(size_t)ch * (size_t)n + (size_t)k].i = 0.0f;
        }
    for (h = 0; h < hops; ++h) {
        for (i = 0; i < hop; ++i) {
            float t = (float)(h * hop + i);
            float r = 0.6f * sinf(0.06f * t);
            far[i] = r;
            for (ch = 0; ch < 4; ++ch)
                mics[(size_t)i * 4u + (size_t)ch] = 0.5f * r + 0.02f * sinf(0.017f * t);
        }
        if (four_aec_nr_res_process_pre(p, mics, far, &pre) != FOUR_AEC_NR_RES_OK)
            break;
        (void)four_aec_nr_res_process_post(p, &pre.token, w, out);
    }
    free(mics); free(far); free(out); free(w);
}

static void test_runtime_strength(void) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
    FourAecNrRes* p = four_aec_nr_res_create(&cfg);
    float live0 = 0.0f, target0 = 0.0f, live1 = 0.0f, target1 = 0.0f;

    CHECK(p != NULL, "strength: core created");
    if (!p) return;
    feed_strength_hops(p, 30);
    CHECK(four_aec_nr_res_post_split_floor(p, &live0, &target0) == 0,
          "strength: the shared post floor is readable");

    /* 1. The setter moves the SHARED post suppressor -- the gain that
     *    actually multiplies this core's output. */
    CHECK(four_aec_nr_res_set_aec_preset(p, AEC_PRESET_AGGRESSIVE, 0.0f) == 0,
          "strength: aggressive accepted");
    CHECK(four_aec_nr_res_post_split_floor(p, &live1, &target1) == 0 &&
              live1 != live0 && live1 == target1,
          "strength: the shared post floor moved and landed immediately");

    /* 2. It SURVIVES a reset. reset_post_sg() rebuilds the suppressor from
     *    the separate post_sg_cfg snapshot, so a setter that wrote only
     *    post_sg.cfg would be silently reverted right here. */
    four_aec_nr_res_reset(p);
    {
        float live2 = 0.0f, target2 = 0.0f;
        CHECK(four_aec_nr_res_post_split_floor(p, &live2, &target2) == 0 &&
                  live2 == live1 && target2 == target1,
              "strength: the new floor survives a reset (a one-copy setter "
              "would have reverted it here)");
    }

    /* 3. A ramp is genuinely in flight before it lands. */
    feed_strength_hops(p, 20);
    CHECK(four_aec_nr_res_set_aec_preset(p, AEC_PRESET_MILD, 100.0f) == 0,
          "strength: ramped retarget accepted");
    {
        float lv = 0.0f, tg = 0.0f;
        feed_strength_hops(p, 2);
        four_aec_nr_res_post_split_floor(p, &lv, &tg);
        CHECK(lv != tg, "strength: the ramp is mid-flight after 2 hops");
        feed_strength_hops(p, 60);
        four_aec_nr_res_post_split_floor(p, &lv, &tg);
        CHECK(lv == tg, "strength: the ramp lands");
    }

    /* 4. Refusal. */
    CHECK(four_aec_nr_res_set_aec_preset(NULL, AEC_PRESET_MILD, 0.0f) == -1,
          "strength: NULL core refused");
    CHECK(four_aec_nr_res_set_aec_preset(p, (AecPreset)77, 0.0f) == -1,
          "strength: out-of-enum preset refused");
    CHECK(four_aec_nr_res_set_aec_preset(p, AEC_PRESET_MILD, 60001.0f) == -1,
          "strength: out-of-range ramp_ms refused");
    CHECK(four_aec_nr_res_set_nr_mode(p, (MmseLsaNrMode)42) == -1,
          "strength: out-of-enum NR mode refused");

    /* 5. The NR setter recomposes THIS pipeline's configuration. The
     *    canonical convenience wrapper must be REFUSED on this instance --
     *    its L differs -- which is precisely why recomposing is required and
     *    not merely tidier. */
    CHECK(four_aec_nr_res_set_nr_mode(p, MMSE_LSA_NR_AGGRESSIVE) == 0,
          "strength: NR mode change accepted through the recomposed target");

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
    test_delay_modes_and_bank_sizing();
    test_known_delay_acquisition_and_coverage();
    test_known_delay_mislock_is_detectable();
    test_known_delay_memory_and_cost();
    test_shared_delay_change_admission();
    test_delay_change_candidate_ttl();
    test_runtime_strength();

    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All 4aec_nr_res tests passed\n");
    return 0;
}
