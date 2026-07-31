/**
 * Structural and lifecycle tests for aec_4ch/4aec_nr_res.h.
 *
 * The test intentionally uses equal weights as an external-beamformer stand
 * in. The library itself must never choose those weights in production.
 */

#include "4aec_nr_res.h"

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

int main(void) {
    test_invalid_configs();
    test_cross_instance_token();
    test_pool_reinit_token_rejected();
    run_grid(16000, 256);
    run_grid(16000, 512);
    run_grid(48000, 1024);
    run_static_parity(16000, 256);
    run_static_parity(16000, 512);
    run_static_parity(48000, 1024);

    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All 4aec_nr_res tests passed\n");
    return 0;
}
