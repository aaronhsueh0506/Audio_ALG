/**
 * example_board_adapter.c — reference board-adapter integration example for
 * mono_aec_nr_res/audio_pipeline.h.
 *
 * ============================================================================
 * REFERENCE ONLY
 * ============================================================================
 * This file is a compilable, runnable HOST SIMULATION of a board-side
 * common-memory-manager integration. It demonstrates the calling sequence,
 * error handling, and the memory-descriptor contract described in
 * pipelines/README.md's "Board Integration" section — nothing more.
 *
 * It does NOT replace production board integration and sign-off. The actual
 * platform adapter source, the real memory-manager
 * implementation (allocator, cache/DMA coherence handling, power-state
 * behaviour), the build command used to compile it for-target, and the
 * final link map still have to be authored for the real target and
 * submitted for sign-off. Every `board_mem_*` function below is a stand-in
 * over a plain static host array — it exists only so this file links and
 * runs on a desktop/CI machine; none of it is platform code. Every point
 * where real platform code must be substituted is marked with a
 * `// BOARD:` comment.
 *
 * Build/run (from pipelines/):
 *   make example-adapter                 # ne10 backend (default everywhere)
 *   make BACKEND=kiss example-adapter    # portable/bit-reproducible reference backend
 *
 * Flow demonstrated (mirrors README.md "Board Integration" -> "Sequence"):
 *   1. query    audio_pipeline_get_mem_requirements() — EVERY time, never
 *               cached across a build/backend/config change (see README's
 *               "Warnings").
 *   2. allocate board_mem_alloc() — the platform stand-in.
 *   3. init     audio_pipeline_init_ex(pool, req.bytes, &cfg, &req) —
 *               passing the just-queried `req` back in as `expected`.
 *   4. process  N hops of a synthetic mic/ref signal.
 *   5. reset    audio_pipeline_reset() (echo-path-change / stream-switch
 *               simulation).
 *   6. process  N more hops.
 *   7. destroy  audio_pipeline_destroy().
 *   8. release  board_mem_free() — only after step 7.
 *   Then the WHOLE cycle repeats on the SAME arena, proving pool
 *   reusability (a real board reuses its one fixed common pool across
 *   many stream lifetimes, not just one).
 *
 * After the two positive cycles, run_negative_demonstrations() exercises
 * the rejection paths a board bring-up log needs to see fire correctly:
 * an undersized pool, a misaligned pool, a stale/tampered descriptor
 * (wrong backend_id / wrong descriptor_version / wrong build_flags_hash),
 * double-destroy safety, and that a rejected init_ex() call never corrupts
 * the pool (a subsequent correct init_ex() on the SAME pool still works).
 *
 * Exit code: 0 iff every step above succeeded and every negative
 * demonstration rejected exactly what it was supposed to reject — this
 * file is wired into `make example-adapter` (and, standalone, `make test`)
 * as a smoke test, not just a demo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "audio_pipeline.h"

/* ============================================================================
 * board_mem — simulated platform common-memory-manager (REFERENCE ONLY)
 * ============================================================================
 *
 * BOARD: replace this ENTIRE module with the platform's real memory
 * manager. A real board integration does not carry a static host array —
 * it has a fixed-partition table, a dedicated audio heap, or a pool
 * allocator with its own bookkeeping. This stand-in only needs to expose
 * the same three operations any of those would: a capacity check, an
 * alloc, and a free — that surface is what the rest of this file (and
 * audio_pipeline.h) actually depends on.
 *
 * The arena is sized from the LARGEST configuration this repo currently
 * supports — 48 kHz, either FFT backend — plus headroom, per
 * pipelines/README.md's "Memory Budget" table (measured: 1,597,536 B KISS
 * / 1,591,712 B NE10 @ 48 kHz, balanced presets). 2 MiB leaves a comfortable
 * ~480+ KB margin over the larger (KISS) figure. This is a REFERENCE
 * sizing choice for the host simulation, not a platform requirement —
 * BOARD: size the platform's real common pool from whatever set of
 * configs (sample rates, presets, aec_only, backend) the product actually
 * ships, using audio_pipeline_get_mem_requirements() (or
 * audio_pipeline_get_mem_breakdown() for a per-module view) for each one,
 * not this file's guess.
 */
#define BOARD_ARENA_BYTES ((size_t)2u * 1024u * 1024u)  /* 2 MiB */

static uint8_t g_board_arena[BOARD_ARENA_BYTES] __attribute__((aligned(16)));
static int     g_board_mem_in_use = 0;   /* exclusive-lifetime simulation */

/**
 * BOARD: replace with the platform's real allocator call. Must return a
 * pointer aligned to (at least) `alignment` bytes with at least `bytes`
 * USABLE bytes, or NULL on failure — audio_pipeline_init_ex() itself
 * re-checks both properties and rejects cleanly if they don't hold, but a
 * real allocator should never hand back something that fails either check.
 *
 * The `req.bytes <= arena capacity` check below is exactly the "does the
 * platform's fixed common pool actually have room for THIS config" gate a
 * real board memory manager needs somewhere in its own alloc path — here
 * it's a static array bound; on a real target it's whatever your pool
 * manager's own capacity accounting looks like.
 */
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
        /* Would only fire if the arena's own alignment attribute above were
         * ever weakened below what the library asks for -- kept as an
         * explicit, named check anyway (fail loudly here rather than let a
         * misaligned pointer surface as a confusing rejection two calls
         * later inside audio_pipeline_init_ex()). */
        fprintf(stderr, "board_mem_alloc: arena base is not %u-byte aligned\n", alignment);
        return NULL;
    }
    g_board_mem_in_use = 1;
    return g_board_arena;
}

/**
 * BOARD: replace with the platform's real free/release call.
 */
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

    /* BOARD: cache/DMA coherence note. If this pool is shared with a DMA
     * engine or lives in non-cache-coherent memory on the real target,
     * this is one of the points where the platform's own cache-maintenance
     * API (clean/invalidate over the pool's address range) belongs --
     * before the region is handed to anything else. This host simulation
     * runs in ordinary cache-coherent process memory, so no action is
     * needed here; audio_pipeline.c itself has no notion of cache lines or
     * DMA and performs none of its own (see README.md's "Board-side
     * verification checklist"). */
}

/* ============================================================================
 * Synthetic signal (no WAV I/O -- this is a smoke test, not a quality test)
 * ============================================================================ */

#define MAX_HOP 512   /* covers every supported rate's hop (80 @ 8k / 160 @ 16k / 480 @ 48k) */

static uint32_t g_lcg_state = 0xC0FFEEu;
static float lcg_sample(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(g_lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/**
 * Drive `n_hops` hops of a synthetic mic/ref stream through `p`, checking
 * only that audio_pipeline_process() reports success and that the output
 * contains no NaN -- this is a functional smoke test, not a golden-output
 * comparison (see tests/test_audio_pipeline.c for the byte-exact parity proof).
 *
 * BOARD: `mic`/`ref` here are plain stack arrays filled by a host RNG. On a
 * real target these would instead be the current hop read out of a codec
 * DMA ring buffer (mic) and the far-end reference tap (ref); if that
 * transfer crosses a DMA/cache boundary, invalidate the cache lines it
 * wrote BEFORE calling audio_pipeline_process() below (mic/ref must be
 * CPU-visible on entry — audio_pipeline_process() assumes this and does no
 * cache maintenance of its own). Symmetrically, `out` must be fully
 * written (it is, unconditionally, by the time this call returns) before
 * any DMA-out on it begins; flush/clean it first if the platform requires
 * that for a DMA source buffer.
 */
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

/* ============================================================================
 * Positive flow: query -> alloc -> init_ex -> process -> reset -> process ->
 * destroy -> free. Called TWICE on the same arena (see main()) to prove pool
 * reusability across two full stream lifetimes, not just one.
 * ============================================================================ */

#define HOPS_BEFORE_RESET 200
#define HOPS_AFTER_RESET  200

static int run_one_cycle(int sample_rate, int cycle_no) {
    printf("--- cycle %d (%d Hz) ---\n", cycle_no, sample_rate);

    /* Step 1: query. ALWAYS fresh, right before the init_ex call below --
     * never cache `req` (or just its `bytes` field) across a build/backend/
     * config change and replay it later (see README.md's "Warnings"). */
    AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "cycle %d: audio_pipeline_get_mem_requirements failed\n", cycle_no);
        return 0;
    }
    printf("  queried: bytes=%llu alignment=%u descriptor_version=%u backend_id=%u\n",
           (unsigned long long)req.bytes, req.alignment, req.descriptor_version, req.backend_id);

    /* Step 2: board alloc. BOARD: board_mem_alloc()/board_mem_free() above
     * are the two calls to replace with the platform's real fixed-common-
     * pool allocator. */
    void* pool = board_mem_alloc(req.bytes, req.alignment);
    if (!pool) {
        fprintf(stderr, "cycle %d: board_mem_alloc failed\n", cycle_no);
        return 0;
    }

    /* Step 3: init_ex, passing `req` straight back in as `expected` -- this
     * is what makes a stale/mismatched descriptor rejected instead of
     * silently carving a pool sized/shaped for a different build. */
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

    /* Step 5: reset -- simulates an echo-path change (speaker swap, AEC
     * re-seat) or handing this SAME instance to a new, unrelated stream.
     *
     * BOARD: power/resume note. This example always reset()s in place on a
     * live pool. A real target may instead SUSPEND with the pool untouched
     * and RESUME later; this library has no notion of power state, so on
     * resume either (a) confirm the platform preserves the pool's contents
     * verbatim across suspend and just keep using the same `p`, or (b) if
     * it doesn't, tear down (destroy) and re-init_ex() from scratch on the
     * SAME pool after resume -- both are safe, but they are NOT
     * interchangeable with a silent no-op: reading stale/garbage pool
     * contents after a resume that did NOT preserve them corrupts state
     * this library has no way to detect (see README.md's "Board-side
     * verification checklist"). */
    audio_pipeline_reset(p);

    /* Step 6: more hops, on the reset instance. */
    if (!run_hops(p, hop, HOPS_AFTER_RESET)) {
        fprintf(stderr, "cycle %d: run_hops (post-reset) failed\n", cycle_no);
        audio_pipeline_destroy(p);
        board_mem_free(pool);
        return 0;
    }

    /* Step 7: destroy -- reverse carve order (NR -> pipeline FFT -> AEC),
     * a no-op on this pool-resident instance per module, but still the
     * required call (forward-compat insurance -- see audio_pipeline.h). */
    audio_pipeline_destroy(p);

    /* Step 8: release -- only after step 7; the pool is dead once
     * audio_pipeline_init_ex()/destroy() have run on it. */
    board_mem_free(pool);

    printf("  cycle %d OK (%d + %d hops, reset in between)\n",
           cycle_no, HOPS_BEFORE_RESET, HOPS_AFTER_RESET);
    return 1;
}

/* ============================================================================
 * Negative demonstrations -- the rejection paths a board bring-up log needs
 * to see fire correctly. Each one attempts something audio_pipeline_init_ex()
 * MUST reject, confirms the rejection (NULL), and -- at the end -- confirms
 * the pool is STILL usable via one final, correct init_ex() call: a
 * rejected init_ex() must never have partially carved/corrupted the pool.
 * ============================================================================ */

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

    /* 2. Misaligned base (+8), correct descriptor and bytes. The arena has
     * hundreds of KB of margin over req.bytes (see BOARD_ARENA_BYTES doc
     * above), so `pool + 8` sized `req.bytes` stays within the arena --
     * this demonstrates the alignment check firing, not an OOB access
     * (audio_pipeline_init_ex() rejects before touching any memory). */
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

    /* 4. Stale/tampered descriptor: wrong backend_id (plain wrong integer,
     * never a string -- see AudioPipelineMemReq.backend_id's doc). */
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

    /* 6. Init after a rejected init still works -- the pool must not have
     * been partially carved/corrupted by any of the five rejected attempts
     * above. Run a couple of real hops through it, not just a NULL check,
     * to prove it is genuinely functional. */
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

/* ============================================================================
 * main
 * ============================================================================ */

int main(void) {
    printf("example_board_adapter -- REFERENCE ONLY host simulation.\n");
    printf("Does NOT replace production board integration: the actual\n");
    printf("adapter source, memory-manager implementation, build command, and\n");
    printf("final link map for the real target must still be submitted for sign-off.\n\n");

    int ok = 1;
    ok &= run_one_cycle(16000, 1);
    /* Second full cycle on the SAME static arena -- proves pool reusability
     * across two independent stream lifetimes, not just one. */
    ok &= run_one_cycle(16000, 2);
    ok &= run_negative_demonstrations();

    printf("\n");
    if (!ok || g_failures) {
        fprintf(stderr, "example_board_adapter: FAIL (%d negative-demo check failure(s), "
                        "positive-flow ok=%d)\n", g_failures, ok);
        return 1;
    }
    printf("example_board_adapter: ALL PASS (2 positive cycles + %d negative-demo checks)\n",
           /* count of CHECK() calls above, kept in sync by hand: undersized,
            * misaligned, descriptor_version, backend_id, build_flags_hash,
            * reusable-after-reject, actually-processes, (destroy-safe is a
            * plain printf, not a CHECK) */
           7);
    return 0;
}
