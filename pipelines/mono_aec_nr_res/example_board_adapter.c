/** Reference board-adapter integration example for the mono AEC -> NR -> RES chain (REFERENCE
 * ONLY): a compilable host simulation of a board-side common-memory-manager integration. It
 * shows the calling sequence, error handling and pool-size accounting of pipelines/README.md
 * "Board Integration". It does not replace the platform adapter, memory manager, target build
 * or link map. Every `board_mem_*` function is a stand-in over a static host array; points
 * where platform code must be substituted are marked `BOARD:`.
 * Build/run (from pipelines/):  make example-adapter
 *
 * The integrator sums the pool of each stage it uses. Here that is lib/aec's Aec
 * (aec_get_mem_size) followed by the gain chain of samples/se_sample_chain.h
 * (se_sample_chain_mem_size), both 16-byte aligned, in one block.
 * Flow: 1 size the pool from the two stage queries (every time, never cached across a build or
 * config change); 2 allocate (board_mem_alloc); 3 aec_init() and se_sample_chain_init() on the
 * block; 4 process N hops; 5 reset both stages; 6 process N more hops; 7 destroy both stages;
 * 8 board_mem_free() (after 7). The cycle then repeats on the same arena (pool reuse).
 * run_negative_demonstrations() checks the rejection paths: undersized pool, misaligned pool,
 * double destroy, and that a rejected init leaves the pool usable.
 * Exit code 0 iff every step succeeded and every negative demonstration rejected what it
 * should. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "aec.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_sample_chain.h"
#include "utility/mem_align.h"

/* board_mem: simulated platform common-memory manager (REFERENCE ONLY).
 * BOARD: replace this whole module with the platform's memory manager; it only needs
 * a capacity check, an alloc and a free. The arena covers the largest supported
 * configuration (48 kHz, balanced presets) plus headroom; BOARD: size the real pool
 * from the configs the product ships, using the size queries of the stages. */
#define BOARD_ARENA_BYTES ((size_t)2u * 1024u * 1024u)  /* 2 MiB */

static uint8_t g_board_arena[BOARD_ARENA_BYTES] __attribute__((aligned(16)));
static int     g_board_mem_in_use = 0;   /* exclusive-lifetime simulation */

/** BOARD: replace with the platform allocator. Must return a pointer aligned to at
 * least `alignment` with at least `bytes` usable bytes, or NULL; the stage inits re-check
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
        /* Fails loudly here instead of as a confusing rejection in the init. */
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

/* The mono stages of one product configuration and the pool they share. */

#define POOL_ALIGNMENT 16u

typedef struct MonoPlan {
    SeGrid grid;
    AecConfig aec_cfg;
    SeSampleChainConfig chain_cfg;
    size_t aec_bytes, chain_bytes, total;
} MonoPlan;

typedef struct MonoStages {
    Aec* aec;
    SeSampleChain* chain;
} MonoStages;

/** Step 1: the lane's grid and AecConfig, the chain config, and the pool as the sum of the two
 * stage sizes. Returns 0, or -1 for a configuration a stage refuses. */
static int mono_plan(MonoPlan* plan, int sample_rate) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    memset(plan, 0, sizeof(*plan));
    memset(&req, 0, sizeof(req));
    req.sample_rate = sample_rate;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = DA_NUM_FILTERS;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &plan->grid, &plan->aec_cfg) != 0) return -1;
    plan->chain_cfg = se_sample_chain_default_config(sample_rate, plan->grid.fft_size);
    plan->chain_cfg.res = SE_SAMPLE_RES_FRAME_GAIN;
    plan->aec_bytes = aec_get_mem_size(&plan->aec_cfg);
    plan->chain_bytes = se_sample_chain_mem_size(&plan->chain_cfg);
    if (plan->aec_bytes == 0 || plan->chain_bytes == 0) return -1;
    plan->total = ALIGN16(plan->aec_bytes) + plan->chain_bytes;
    return 0;
}

/** Step 3: both stages on one block of `bytes`. Each stage gets the share of the block it
 * would own and checks it itself: the AEC refuses a misaligned base or a short block, the
 * chain a short remainder. A refused chain rolls the AEC back. NULL stages on refusal. */
static int mono_init(MonoStages* s, void* pool, size_t bytes, const MonoPlan* plan) {
    uint8_t* cur = (uint8_t*)pool;
    size_t chain_offset = ALIGN16(plan->aec_bytes);
    memset(s, 0, sizeof(*s));
    if (!pool) return -1;
    s->aec = aec_init(cur, bytes < plan->aec_bytes ? bytes : plan->aec_bytes, &plan->aec_cfg);
    if (!s->aec) return -1;
    s->chain = se_sample_chain_init(cur + chain_offset,
                                    bytes > chain_offset ? bytes - chain_offset : 0,
                                    &plan->chain_cfg);
    if (!s->chain) { aec_destroy(s->aec); s->aec = NULL; return -1; }
    return 0;
}

static int mono_hop(MonoStages* s, const float* mic, const float* ref, float* out) {
    SeSpecFrame frame;
    const Complex* spec;
    aec_process_context(s->aec, mic, ref);
    if (se_frame_from_aec(s->aec, &frame) != 0) return -1;
    spec = se_sample_chain_form(s->chain, &frame);
    if (!spec) return -1;
    return se_sample_chain_synthesize(s->chain, spec, out);
}

static void mono_reset(MonoStages* s) {
    aec_reset(s->aec);
    se_sample_chain_reset(s->chain);
}

/** Step 7: reverse carve order (NR inside the chain, then the AEC); a no-op per module on a
 * pool-resident instance, but still required. Safe to call twice. */
static void mono_destroy(MonoStages* s) {
    se_sample_chain_destroy(s->chain);
    aec_destroy(s->aec);
}

/* Synthetic signal (smoke test, not a quality test) */

#define MAX_HOP 512   /* covers every supported grid's hop (128 @ 8k / 128, 256 @ 16k / 512 @ 48k) */

static uint32_t g_lcg_state = 0xC0FFEEu;
static float lcg_sample(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(g_lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/** Drives `n_hops` hops of a synthetic mic/ref stream, checking that every call succeeds and
 * the output is finite.
 * BOARD: on a target, mic/ref come from the codec DMA ring and the far-end tap;
 * invalidate the cache lines they occupy before calling the hop, and flush `out`
 * before any DMA-out. */
static int run_hops(MonoStages* s, int hop, int n_hops) {
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
        if (mono_hop(s, mic, ref, out) != 0) {
            fprintf(stderr, "run_hops: the hop failed at hop %d\n", h);
            return 0;
        }
        if (!se_hop_is_finite(out, hop)) {
            fprintf(stderr, "run_hops: non-finite output at hop %d\n", h);
            return 0;
        }
    }
    return 1;
}

/* Positive flow: size -> alloc -> init -> process -> reset -> process -> destroy
 * -> free, run twice on the same arena to show pool reuse. */

#define HOPS_BEFORE_RESET 200
#define HOPS_AFTER_RESET  200

static int run_one_cycle(int sample_rate, int cycle_no) {
    MonoPlan plan;
    MonoStages stages;
    void* pool;
    printf("--- cycle %d (%d Hz) ---\n", cycle_no, sample_rate);

    /* Step 1: size, always fresh right before the init; never cache the plan. */
    if (mono_plan(&plan, sample_rate) != 0) {
        fprintf(stderr, "cycle %d: the stage size queries failed\n", cycle_no);
        return 0;
    }
    printf("  sized: aec=%zu chain=%zu total=%zu bytes alignment=%u\n",
           plan.aec_bytes, plan.chain_bytes, plan.total, POOL_ALIGNMENT);

    /* Step 2: board alloc. BOARD: replace board_mem_alloc()/board_mem_free(). */
    pool = board_mem_alloc(plan.total, POOL_ALIGNMENT);
    if (!pool) {
        fprintf(stderr, "cycle %d: board_mem_alloc failed\n", cycle_no);
        return 0;
    }

    /* Step 3: init both stages on the block. */
    if (mono_init(&stages, pool, (size_t)plan.total, &plan) != 0) {
        fprintf(stderr, "cycle %d: stage init failed\n", cycle_no);
        board_mem_free(pool);
        return 0;
    }
    int hop = plan.grid.hop_size;
    printf("  init OK: hop=%d n_freqs=%d\n", hop, plan.grid.n_freqs);

    /* Step 4: N hops of synthetic audio. */
    if (!run_hops(&stages, hop, HOPS_BEFORE_RESET)) {
        fprintf(stderr, "cycle %d: run_hops (pre-reset) failed\n", cycle_no);
        mono_destroy(&stages);
        board_mem_free(pool);
        return 0;
    }

    /* Step 5: reset (echo-path change or a new stream on the same instance); the AEC and
     * the chain are reset together.
     * BOARD: on suspend/resume either confirm the platform preserved the pool contents
     * and keep using the stages, or destroy and init again on the same pool; resuming on
     * unpreserved contents corrupts state the library cannot detect. */
    mono_reset(&stages);

    /* Step 6: more hops on the reset instance. */
    if (!run_hops(&stages, hop, HOPS_AFTER_RESET)) {
        fprintf(stderr, "cycle %d: run_hops (post-reset) failed\n", cycle_no);
        mono_destroy(&stages);
        board_mem_free(pool);
        return 0;
    }

    /* Step 7: destroy. */
    mono_destroy(&stages);

    /* Step 8: release, only after step 7. */
    board_mem_free(pool);

    printf("  cycle %d OK (%d + %d hops, reset in between)\n",
           cycle_no, HOPS_BEFORE_RESET, HOPS_AFTER_RESET);
    return 1;
}

/* Negative demonstrations: each attempts something the init must reject, confirms the NULL,
 * and finally confirms the pool is still usable. */

static int g_failures = 0;
#define CHECK(cond, msg) do { \
        if (cond) { printf("PASS: %s\n", (msg)); } \
        else      { fprintf(stderr, "FAIL: %s\n", (msg)); g_failures++; } \
    } while (0)

static int run_negative_demonstrations(void) {
    MonoPlan plan;
    MonoStages s;
    void* pool;
    printf("--- negative demonstrations ---\n");

    if (mono_plan(&plan, 16000) != 0) {
        fprintf(stderr, "negative demo: the stage size queries failed\n");
        return 0;
    }
    pool = board_mem_alloc(plan.total, POOL_ALIGNMENT);
    if (!pool) {
        fprintf(stderr, "negative demo: board_mem_alloc setup failed\n");
        return 0;
    }

    /* 1. A block one byte short of the AEC's own size: aec_init() refuses it. */
    CHECK(mono_init(&s, pool, plan.aec_bytes - 1, &plan) != 0 && !s.aec && !s.chain,
          "init rejects a pool smaller than the AEC (aec_bytes-1)");

    /* 2. Misaligned base (+8); the arena has margin, so no out-of-bounds access (the init
     * rejects before touching memory). */
    CHECK(mono_init(&s, (uint8_t*)pool + 8, (size_t)plan.total, &plan) != 0 && !s.aec && !s.chain,
          "init rejects a misaligned (base+8) pool");

    /* 3. The AEC fits, the chain does not: se_sample_chain_init() refuses the short remainder
     * and the AEC is rolled back. */
    CHECK(mono_init(&s, pool, (size_t)ALIGN16(plan.aec_bytes) + plan.chain_bytes / 2, &plan) != 0 &&
              !s.aec && !s.chain,
          "init rejects a pool that holds the AEC but not the chain");

    /* 4. Init after rejected attempts still works; run real hops to prove it. */
    if (mono_init(&s, pool, (size_t)plan.total, &plan) == 0) {
        CHECK(1, "pool is still usable via a correct init after rejected attempts");
        CHECK(run_hops(&s, plan.grid.hop_size, 10), "the still-usable instance actually processes audio");

        /* 5. Double-destroy safety. */
        mono_destroy(&s);
        mono_destroy(&s);   /* second call: must be a safe no-op */
        printf("PASS: destroy is safe to call twice\n");
    } else {
        CHECK(0, "pool is still usable via a correct init after rejected attempts");
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
           5);
    return 0;
}
