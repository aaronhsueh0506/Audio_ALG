/**
 * test_audio_pipeline.c — F20 acceptance tests for pipelines/audio_pipeline.h.
 *
 * Not a DSP-quality test (no AECMOS, no reference WAVs) — a contract test
 * for the library API surface itself: does the pool-first path behave
 * exactly like the heap-convenience path, does it reject what it must
 * reject, and is teardown safe to call more than once. Style mirrors
 * lib/aec/c_impl/test/test_zero_heap_aec.c (LCG synthetic input, a
 * 0xA5-poisoned pool to prove the "no blanket memset needed" claim,
 * PASS/FAIL prints, nonzero exit on any failure).
 *
 * Build (from pipelines/, after `make libs` for the selected BACKEND):
 *   make test                     # kiss backend (default)
 *   make BACKEND=ne10 test        # ne10 backend
 *
 * Standalone (no Makefile):
 *   cc -O2 -std=gnu99 -I../lib/aec/c_impl/include -I../lib/aec/c_impl/example \
 *      -I../lib/nr/c_impl/include -I../../audio_common/include \
 *      -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\" \
 *      test_audio_pipeline.c audio_pipeline.c \
 *      ../lib/aec/c_impl/bin/libaec.a ../lib/nr/c_impl/bin/libmmse_lsa.a \
 *      ../../audio_common/bin/kiss/libaudio_common.a -lm -o /tmp/tap && /tmp/tap
 *
 * Cases:
 *   1. audio_pipeline_get_mem_requirements: NULL cfg/out rejected,
 *      sample_rate=44100 rejected, sample_rate=16000 accepted.
 *   2. audio_pipeline_init: a misaligned pool rejected, an undersized pool
 *      rejected.
 *   3. audio_pipeline_create() (heap) vs audio_pipeline_init() (caller pool,
 *      deliberately poisoned with 0xA5 before init) produce BYTE-IDENTICAL
 *      output over 1000 hops of LCG synthetic mic/ref input — the direct
 *      proof of audio_pipeline_init's "a dirty pool is safe without the
 *      caller's blanket memset" claim.
 *   4. audio_pipeline_destroy() idempotence + NULL-safety on a pool-resident
 *      instance, and that the pool itself is untouched/reusable afterward.
 */
#include "audio_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define HOP_COUNT 1000

static int g_failures = 0;
#define CHECK(cond, msg) do { \
        if (cond) { printf("PASS: %s\n", (msg)); } \
        else      { fprintf(stderr, "FAIL: %s\n", (msg)); g_failures++; } \
    } while (0)

/* ---- LCG synthetic mic/ref generator (mirrors test_zero_heap_aec.c) ---- */
static uint32_t lcg_state;
static float lcg_sample(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

static void test_validation(void) {
    AudioPipelineMemReq req;

    CHECK(audio_pipeline_get_mem_requirements(NULL, &req) == -1,
          "get_mem_requirements rejects a NULL config");

    AudioPipelineConfig cfg16k = audio_pipeline_default_config(16000);
    CHECK(audio_pipeline_get_mem_requirements(&cfg16k, NULL) == -1,
          "get_mem_requirements rejects a NULL out-param");

    AudioPipelineConfig bad_rate = audio_pipeline_default_config(44100);
    CHECK(audio_pipeline_get_mem_requirements(&bad_rate, &req) == -1,
          "get_mem_requirements rejects sample_rate=44100 (rate whitelist)");

    CHECK(audio_pipeline_get_mem_requirements(&cfg16k, &req) == 0 && req.bytes > 0,
          "get_mem_requirements accepts sample_rate=16000");
    printf("       (16 kHz descriptor: bytes=%zu alignment=%zu layout_version=%u "
           "backend=%s build_flags_hash=0x%08x)\n",
           req.bytes, req.alignment, req.layout_version, req.backend, req.build_flags_hash);
}

static void test_pool_rejection(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for pool-rejection test\n");
        g_failures++;
        return;
    }

    /* Extra 16 bytes of headroom so a +1-byte-offset "misaligned" pointer
     * still has req.bytes of addressable space behind it (the alignment
     * check must reject it before any of that space is touched, but the
     * allocation itself must not be an OOB setup). */
    void* pool = NULL;
    if (posix_memalign(&pool, 16, req.bytes + 16) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for pool-rejection test\n");
        g_failures++;
        return;
    }

    AudioPipeline* p_misaligned = audio_pipeline_init((uint8_t*)pool + 1, req.bytes, &cfg);
    CHECK(p_misaligned == NULL, "audio_pipeline_init rejects a misaligned (base+1) pool");

    AudioPipeline* p_short = audio_pipeline_init(pool, req.bytes - 1, &cfg);
    CHECK(p_short == NULL, "audio_pipeline_init rejects an undersized (bytes-1) pool");

    AudioPipeline* p_ok = audio_pipeline_init(pool, req.bytes, &cfg);
    CHECK(p_ok != NULL, "audio_pipeline_init accepts a correctly aligned/sized pool");
    if (p_ok) audio_pipeline_destroy(p_ok);

    free(pool);
}

/* Feed `hops` hops of a deterministic LCG mic/ref sequence into `p`,
 * collecting each hop's output into out_all[hops*hop]. The LCG is reseeded
 * to the SAME constant at the top of every call, so two calls (heap vs.
 * pool instance) see byte-identical input. */
static void run_hops(AudioPipeline* p, int hop, int hops, float* out_all) {
    lcg_state = 0xC0FFEEu;
    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    for (int h = 0; h < hops; h++) {
        for (int i = 0; i < hop; i++) {
            ref[i] = lcg_sample();
            mic[i] = 0.3f * ref[i] + 0.05f * lcg_sample();   /* echo + a bit of near-end */
        }
        audio_pipeline_process(p, mic, ref, out_all + (size_t)h * hop);
    }
    free(mic);
    free(ref);
}

static void test_create_vs_init_parity(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);

    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for create-vs-init parity test\n");
        g_failures++;
        return;
    }

    AudioPipeline* p_heap = audio_pipeline_create(&cfg);
    if (!p_heap) {
        fprintf(stderr, "FAIL: audio_pipeline_create\n");
        g_failures++;
        return;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for create-vs-init parity test\n");
        g_failures++;
        audio_pipeline_destroy(p_heap);
        return;
    }
    memset(pool, 0xA5, req.bytes);   /* dirty pool: init must not rely on zeros */

    AudioPipeline* p_pool = audio_pipeline_init(pool, req.bytes, &cfg);
    if (!p_pool) {
        fprintf(stderr, "FAIL: audio_pipeline_init on a poisoned (0xA5) pool\n");
        g_failures++;
        audio_pipeline_destroy(p_heap);
        free(pool);
        return;
    }

    int hop = audio_pipeline_hop_size(p_heap);
    CHECK(hop > 0 && hop == audio_pipeline_hop_size(p_pool),
          "heap and pool instances agree on hop_size");

    float* out_heap = (float*)malloc((size_t)HOP_COUNT * (size_t)hop * sizeof(float));
    float* out_pool = (float*)malloc((size_t)HOP_COUNT * (size_t)hop * sizeof(float));

    run_hops(p_heap, hop, HOP_COUNT, out_heap);
    run_hops(p_pool, hop, HOP_COUNT, out_pool);

    int byte_equal = memcmp(out_heap, out_pool, (size_t)HOP_COUNT * (size_t)hop * sizeof(float)) == 0;
    CHECK(byte_equal,
          "audio_pipeline_create (heap) == audio_pipeline_init (0xA5-poisoned pool), "
          "1000 hops, byte-for-byte");

    int finite = 1;
    for (int i = 0; i < HOP_COUNT * hop; i++) {
        if (out_heap[i] != out_heap[i]) { finite = 0; break; }   /* NaN check */
    }
    CHECK(finite, "1000-hop synthetic run produces no NaN in the output");

    free(out_heap);
    free(out_pool);
    audio_pipeline_destroy(p_heap);   /* frees its own owned pool            */
    audio_pipeline_destroy(p_pool);   /* pool path: no-op, caller keeps pool */
    free(pool);
}

static void test_destroy_idempotence(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for destroy-idempotence test\n");
        g_failures++;
        return;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for destroy-idempotence test\n");
        g_failures++;
        return;
    }

    AudioPipeline* p = audio_pipeline_init(pool, req.bytes, &cfg);
    if (!p) {
        fprintf(stderr, "FAIL: audio_pipeline_init for destroy-idempotence test\n");
        g_failures++;
        free(pool);
        return;
    }

    audio_pipeline_destroy(p);
    audio_pipeline_destroy(p);      /* second call on the same pool-resident instance */
    audio_pipeline_destroy(NULL);   /* NULL-safe */
    printf("PASS: audio_pipeline_destroy is idempotent (2x) and NULL-safe on a "
           "pool-resident instance\n");

    /* The pool itself must be untouched/reusable: destroy() on a
     * pool-resident instance never frees `pool` (the caller owns it). */
    AudioPipeline* p2 = audio_pipeline_init(pool, req.bytes, &cfg);
    CHECK(p2 != NULL, "pool is reusable via a fresh audio_pipeline_init after destroy");
    if (p2) audio_pipeline_destroy(p2);

    free(pool);
}

int main(void) {
    test_validation();
    test_pool_rejection();
    test_create_vs_init_parity();
    test_destroy_idempotence();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
