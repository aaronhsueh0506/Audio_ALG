/** Reference board-adapter integration example for audio_pipeline.h (REFERENCE
 * ONLY): a compilable host simulation of a board-side common-memory-manager
 * integration. It shows the calling sequence, error handling and memory-descriptor
 * contract of pipelines/README.md "Board Integration". It does not replace the
 * platform adapter, memory manager, target build or link map. Every `board_mem_*`
 * function is a stand-in over a static host array; points where platform code must
 * be substituted are marked `BOARD:`.
 * Build/run (from pipelines/):  make example-adapter
 * Flow: 1 query audio_pipeline_get_mem_requirements() (every time, never cached
 * across a build/backend/config change); 2 allocate (board_mem_alloc); 3
 * audio_pipeline_init_ex(pool, req.bytes, &cfg, &req) with the queried `req` as
 * `expected`; 4 process N hops; 5 audio_pipeline_reset(); 6 process N more hops; 7
 * audio_pipeline_destroy(); 8 board_mem_free() (after 7). The cycle then repeats on
 * the same arena (pool reuse). run_negative_demonstrations() checks the rejection
 * paths: undersized pool, misaligned pool, stale descriptor (backend_id /
 * descriptor_version / build_flags_hash), double destroy, and that a rejected
 * init_ex() leaves the pool usable.
 * Exit code 0 iff every step succeeded and every negative demonstration rejected
 * what it should. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "audio_pipeline.h"

/* board_mem: simulated platform common-memory manager (REFERENCE ONLY).
 * BOARD: replace this whole module with the platform's memory manager; it only needs
 * a capacity check, an alloc and a free. The arena covers the largest supported
 * configuration (48 kHz, balanced presets) plus headroom; BOARD: size the real pool
 * from the configs the product ships, using
 * audio_pipeline_get_mem_requirements() / audio_pipeline_get_mem_breakdown(). */
#define BOARD_ARENA_BYTES ((size_t)2u * 1024u * 1024u)  /* 2 MiB */

static uint8_t g_board_arena[BOARD_ARENA_BYTES] __attribute__((aligned(16)));
static int     g_board_mem_in_use = 0;   /* exclusive-lifetime simulation */

/** BOARD: replace with the platform allocator. Must return a pointer aligned to at
 * least `alignment` with at least `bytes` usable bytes, or NULL; init_ex() re-checks
 * both. The capacity check below stands in for the pool manager's own accounting. */
static void* board_mem_alloc(uint64_t bytes, uint32_t alignment) {
    if (g_board_mem_in_use) {
        fprintf(stderr, "board_mem_alloc: arena already in use (exclusive-lifetime violation -- "
                        "a real board must never hand out an overlapping region)\n");
        return NULL;
    }
    if (bytes > BOARD_ARENA_BYTES) {
        fprintf(stderr, "board_mem_alloc: requested %llu bytes > arena capacity %llu bytes -- "
                        "BOARD: bump the platform pool size, or shrink the requested config\n",
                (unsigned long long)bytes, (unsigned long long)BOARD_ARENA_BYTES);
        return NULL;
    }
    if (((uintptr_t)g_board_arena) % alignment != 0) {
        /* Fails loudly here instead of as a confusing rejection in init_ex(). */
        fprintf(stderr, "board_mem_alloc: arena base is not %u-byte aligned\n", alignment);
        return NULL;
    }
    g_board_mem_in_use = 1;
    return g_board_arena;
}

/** BOARD: replace with the platform's free/release call. */
static void board_mem_free(void* p) {
    if (!p) return;
    if (p != (void*)g_board_arena) {
        fprintf(stderr, "board_mem_free: pointer does not belong to this arena\n");
        return;
    }
    if (!g_board_mem_in_use) {
        fprintf(stderr, "board_mem_free: double-free detected -- ignoring\n");
        return;
    }
    g_board_mem_in_use = 0;

    /* BOARD: cache/DMA coherence. If the pool is shared with a DMA engine or sits in
     * non-coherent memory, clean/invalidate its address range here; the library does no
     * cache maintenance of its own. */
}

/* Synthetic signal (smoke test, not a quality test) */

#define MAX_HOP 512   /* covers every supported rate's hop (80 @ 8k / 160 @ 16k / 480 @ 48k) */

static uint32_t g_lcg_state = 0xC0FFEEu;
static float lcg_sample(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(g_lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/** Drives `n_hops` hops of a synthetic mic/ref stream, checking that
 * audio_pipeline_process() succeeds and the output has no NaN.
 * BOARD: on a target, mic/ref come from the codec DMA ring and the far-end tap;
 * invalidate the cache lines they occupy before calling process(), and flush `out`
 * before any DMA-out. */
static int run_hops(AudioPipeline* p, int hop, int n_hops) {
    float mic[MAX_HOP], ref[MAX_HOP], out[MAX_HOP];
    if (hop <= 0 || hop > MAX_HOP) {
        fprintf(stderr, "run_hops: hop=%d out of expected range (1..%d)\n", hop, MAX_HOP);
        return 0;
    }
    for (int h = 0; h < n_hops; h++) {
        for (int i = 0; i < hop; i++) {
            ref[i] = lcg_sample();
            mic[i] = 0.3f * ref[i] + 0.05f * lcg_sample();   /* echo + a bit of near-end */
        }
        if (audio_pipeline_process(p, mic, ref, out) != 0) {
            fprintf(stderr, "run_hops: audio_pipeline_process failed at hop %d\n", h);
            return 0;
        }
        for (int i = 0; i < hop; i++) {
            if (out[i] != out[i]) {   /* NaN check */
                fprintf(stderr, "run_hops: NaN in output at hop %d, sample %d\n", h, i);
                return 0;
            }
        }
    }
    return 1;
}

/* Positive flow: query -> alloc -> init_ex -> process -> reset -> process -> destroy
 * -> free, run twice on the same arena to show pool reuse. */

#define HOPS_BEFORE_RESET 200
#define HOPS_AFTER_RESET  200

static int run_one_cycle(int sample_rate, int cycle_no) {
    printf("--- cycle %d (%d Hz) ---\n", cycle_no, sample_rate);

    /* Step 1: query, always fresh right before init_ex; never cache `req`. */
    AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "cycle %d: audio_pipeline_get_mem_requirements failed\n", cycle_no);
        return 0;
    }
    printf("  queried: bytes=%llu alignment=%u descriptor_version=%u backend_id=%u\n",
           (unsigned long long)req.bytes, req.alignment, req.descriptor_version, req.backend_id);

    /* Step 2: board alloc. BOARD: replace board_mem_alloc()/board_mem_free(). */
    void* pool = board_mem_alloc(req.bytes, req.alignment);
    if (!pool) {
        fprintf(stderr, "cycle %d: board_mem_alloc failed\n", cycle_no);
        return 0;
    }

    /* Step 3: init_ex with `req` as `expected`, so a stale descriptor is rejected. */
    AudioPipeline* p = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    if (!p) {
        fprintf(stderr, "cycle %d: audio_pipeline_init_ex failed\n", cycle_no);
        board_mem_free(pool);
        return 0;
    }
    int hop = audio_pipeline_hop_size(p);
    printf("  init OK: hop=%d n_freqs=%d\n", hop, audio_pipeline_n_freqs(p));

    /* Step 4: N hops of synthetic audio. */
    if (!run_hops(p, hop, HOPS_BEFORE_RESET)) {
        fprintf(stderr, "cycle %d: run_hops (pre-reset) failed\n", cycle_no);
        audio_pipeline_destroy(p);
        board_mem_free(pool);
        return 0;
    }

    /* Step 5: reset (echo-path change or a new stream on the same instance).
     * BOARD: on suspend/resume either confirm the platform preserved the pool contents
     * and keep using `p`, or destroy and init_ex() again on the same pool; resuming on
     * unpreserved contents corrupts state the library cannot detect. */
    audio_pipeline_reset(p);

    /* Step 6: more hops on the reset instance. */
    if (!run_hops(p, hop, HOPS_AFTER_RESET)) {
        fprintf(stderr, "cycle %d: run_hops (post-reset) failed\n", cycle_no);
        audio_pipeline_destroy(p);
        board_mem_free(pool);
        return 0;
    }

    /* Step 7: destroy (reverse carve order; a no-op per module on a pool-resident
     * instance, but still required). */
    audio_pipeline_destroy(p);

    /* Step 8: release, only after step 7. */
    board_mem_free(pool);

    printf("  cycle %d OK (%d + %d hops, reset in between)\n",
           cycle_no, HOPS_BEFORE_RESET, HOPS_AFTER_RESET);
    return 1;
}

/* Negative demonstrations: each attempts something init_ex() must reject, confirms
 * the NULL, and finally confirms the pool is still usable. */

static int g_failures = 0;
#define CHECK(cond, msg) do { \
        if (cond) { printf("PASS: %s\n", (msg)); } \
        else      { fprintf(stderr, "FAIL: %s\n", (msg)); g_failures++; } \
    } while (0)

static int run_negative_demonstrations(void) {
    printf("--- negative demonstrations ---\n");

    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "negative demo: get_mem_requirements setup failed\n");
        return 0;
    }
    void* pool = board_mem_alloc(req.bytes, req.alignment);
    if (!pool) {
        fprintf(stderr, "negative demo: board_mem_alloc setup failed\n");
        return 0;
    }

    /* 1. Undersized pool (bytes-1), correct descriptor. */
    AudioPipeline* p_short = audio_pipeline_init_ex(pool, (size_t)req.bytes - 1, &cfg, &req);
    CHECK(p_short == NULL, "init_ex rejects an undersized (bytes-1) pool");
    if (p_short) audio_pipeline_destroy(p_short);

    /* 2. Misaligned base (+8), correct descriptor and bytes; the arena has margin, so no
     * out-of-bounds access (init_ex() rejects before touching memory). */
    AudioPipeline* p_misaligned =
        audio_pipeline_init_ex((uint8_t*)pool + 8, (size_t)req.bytes, &cfg, &req);
    CHECK(p_misaligned == NULL, "init_ex rejects a misaligned (base+8) pool");
    if (p_misaligned) audio_pipeline_destroy(p_misaligned);

    /* 3. Stale/tampered descriptor: wrong descriptor_version. */
    AudioPipelineMemReq bad_dv = req;
    bad_dv.descriptor_version = req.descriptor_version + 1;
    AudioPipeline* p_dv = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_dv);
    CHECK(p_dv == NULL, "init_ex rejects a tampered descriptor_version");
    if (p_dv) audio_pipeline_destroy(p_dv);

    /* 4. Stale descriptor: wrong backend_id. */
    AudioPipelineMemReq bad_backend = req;
    bad_backend.backend_id = 99;
    AudioPipeline* p_be = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_backend);
    CHECK(p_be == NULL, "init_ex rejects a tampered backend_id");
    if (p_be) audio_pipeline_destroy(p_be);

    /* 5. Stale/tampered descriptor: wrong build_flags_hash. */
    AudioPipelineMemReq bad_hash = req;
    bad_hash.build_flags_hash = req.build_flags_hash ^ 0xFFFFFFFFu;
    AudioPipeline* p_hash = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_hash);
    CHECK(p_hash == NULL, "init_ex rejects a tampered build_flags_hash");
    if (p_hash) audio_pipeline_destroy(p_hash);

    /* 6. Init after rejected attempts still works; run real hops to prove it. */
    AudioPipeline* p_ok = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p_ok != NULL, "pool is still usable via a correct init_ex after rejected attempts");
    if (p_ok) {
        int hop = audio_pipeline_hop_size(p_ok);
        CHECK(run_hops(p_ok, hop, 10), "the still-usable instance actually processes audio");

        /* 7. Double-destroy safety. */
        audio_pipeline_destroy(p_ok);
        audio_pipeline_destroy(p_ok);   /* second call: must be a safe no-op */
        printf("PASS: audio_pipeline_destroy is safe to call twice\n");
    }

    board_mem_free(pool);
    return g_failures == 0;
}

int main(void) {
    printf("example_board_adapter -- REFERENCE ONLY host simulation.\n");
    printf("Does NOT replace production board integration: the actual\n");
    printf("adapter source, memory-manager implementation, build command, and\n");
    printf("final link map for the real target must still be submitted for sign-off.\n\n");

    int ok = 1;
    ok &= run_one_cycle(16000, 1);
    /* Second full cycle on the same arena (pool reuse). */
    ok &= run_one_cycle(16000, 2);
    ok &= run_negative_demonstrations();

    printf("\n");
    if (!ok || g_failures) {
        fprintf(stderr, "example_board_adapter: FAIL (%d negative-demo check failure(s), "
                        "positive-flow ok=%d)\n", g_failures, ok);
        return 1;
    }
    printf("example_board_adapter: ALL PASS (2 positive cycles + %d negative-demo checks)\n",
           /* number of CHECK() calls above, kept in sync by hand */
           7);
    return 0;
}
