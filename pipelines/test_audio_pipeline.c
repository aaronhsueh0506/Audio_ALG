/**
 * test_audio_pipeline.c — acceptance tests for pipelines/audio_pipeline.h.
 *
 * Not a DSP-quality test (no AECMOS, no reference WAVs) — a contract test
 * for the library API surface itself: does the pool-first path behave
 * exactly like the heap-convenience path, does it reject what it must
 * reject, and is teardown safe to call more than once. Style mirrors
 * lib/aec/c_impl/test/test_zero_heap_aec.c (LCG synthetic input, a
 * 0xA5-poisoned pool to prove the "no blanket memset needed" claim,
 * PASS/FAIL prints, nonzero exit on any failure).
 *
 * Per-rate coverage: every case below (validation, pool rejection,
 * create-vs-init byte-equal parity, destroy idempotence) runs once per
 * supported sample rate — 8000 / 16000 / 48000 — not just 16000. The
 * create-vs-init byte-equality (heap vs. an 0xA5-poisoned pool) at every
 * rate, on whichever BACKEND this binary was built with, is the load-bearing
 * closure: it is the only place that proves the pool-carve arithmetic
 * (pipeline_pool_size/pipeline_build in audio_pipeline.c) agrees with itself
 * across the full 8k/16k/48k FFT-grid range.
 *
 * 48 kHz runs a REDUCED hop count (HOP_COUNT_48K, see below) — AEC's
 * filter_length (and therefore n_partitions/convolution cost) and the
 * pipeline's own hop/n_freqs all scale up substantially above 16 kHz
 * (pipeline_dims.h), so 1000 hops there is materially slower for no added
 * coverage value versus 8k/16k's 1000; the byte-equal parity check is
 * exactly as conclusive at 300 hops (same LCG-seeded stream, same
 * poisoned-pool-vs-heap comparison, just fewer of them).
 *
 * The sample_rate=44100 rate-whitelist rejection is checked exactly ONCE
 * (independent of the per-rate loop below) — it is a property of the
 * validator, not of any one supported rate.
 *
 * Also covers reject-first AudioPipelineConfig validation in
 * derive_dims_and_configs() (audio_pipeline.c): an out-of-enum aec_preset/
 * nr_mode and an out-of-{0,1} bool-typed field must all be rejected by both
 * audio_pipeline_get_mem_requirements() and audio_pipeline_init() — see
 * test_config_validation_rejects().
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
 * Cases (each run once per rate in RATES[] below, unless noted "once"):
 *   1. audio_pipeline_get_mem_requirements: NULL cfg/out rejected,
 *      sample_rate=44100 rejected (once), sample_rate=<rate> accepted.
 *   2. audio_pipeline_init: a misaligned pool rejected, an undersized pool
 *      rejected.
 *   3. audio_pipeline_create() (heap) vs audio_pipeline_init() (caller pool,
 *      deliberately poisoned with 0xA5 before init) produce BYTE-IDENTICAL
 *      output over HOP_COUNT (or HOP_COUNT_48K) hops of LCG synthetic
 *      mic/ref input — the direct proof of audio_pipeline_init's "a dirty
 *      pool is safe without the caller's blanket memset" claim.
 *   4. audio_pipeline_destroy() idempotence + NULL-safety on a pool-resident
 *      instance, and that the pool itself is untouched/reusable afterward.
 *   5. (once) reject-first AudioPipelineConfig validation: an
 *      out-of-enum aec_preset/nr_mode and a bool field holding 2 (or -1)
 *      are all rejected by get_mem_requirements() AND init().
 *   6. (once) audio_pipeline_init_ex()'s `expected` descriptor gate: a
 *      correct/current descriptor is accepted;
 *      a NULL descriptor behaves exactly like audio_pipeline_init(); a
 *      descriptor with a tampered descriptor_version, layout_version,
 *      backend_id, build_flags_hash, alignment, or bytes (or an undersized
 *      `bytes` pool argument alongside an otherwise-correct descriptor) is
 *      each independently rejected; the pool remains usable afterward.
 */
#include "audio_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

#define HOP_COUNT      1000   /* 8 kHz / 16 kHz */
#define HOP_COUNT_48K   300   /* 48 kHz: filter_length/hop/n_freqs all scale up
                                * substantially (pipeline_dims.h) -- 300 hops
                                * keeps runtime sane without weakening the
                                * byte-equal parity proof (see file header). */

typedef struct { int sample_rate; int fft_size; } GridCase;
static const GridCase GRIDS[] = {
    {8000, 256}, {16000, 256}, {16000, 512}, {48000, 1024},
};
#define N_GRIDS ((int)(sizeof(GRIDS) / sizeof(GRIDS[0])))

static int hop_count_for_rate(int sr) { return (sr >= 48000) ? HOP_COUNT_48K : HOP_COUNT; }

static int g_failures = 0;
#define CHECK(cond, msg) do { \
        if (cond) { printf("PASS: %s\n", (msg)); } \
        else      { fprintf(stderr, "FAIL: %s\n", (msg)); g_failures++; } \
    } while (0)

/* snprintf-into-scratch helper so per-rate CHECK messages can embed the rate
 * without every call site hand-rolling its own buffer. Single static buffer
 * is safe here: CHECK's cond/msg are evaluated synchronously, one CHECK at a
 * time, never nested. */
static char g_msgbuf[256];
static const char* fmt_msg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_msgbuf, sizeof(g_msgbuf), fmt, ap);
    va_end(ap);
    return g_msgbuf;
}

/* ---- LCG synthetic mic/ref generator (mirrors test_zero_heap_aec.c) ---- */
static uint32_t lcg_state;
static float lcg_sample(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

static AudioPipelineConfig grid_config(int sr, int fft_size) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(sr);
    cfg.fft_size = fft_size;
    return cfg;
}

/* sr < 0 disables the (once-only) 44100 rate-whitelist check. */
static void test_validation(int sr, int fft_size, int check_rate_whitelist) {
    AudioPipelineMemReq req;

    CHECK(audio_pipeline_get_mem_requirements(NULL, &req) == -1,
          "get_mem_requirements rejects a NULL config");

    AudioPipelineConfig cfg = grid_config(sr, fft_size);
    CHECK(audio_pipeline_get_mem_requirements(&cfg, NULL) == -1,
          "get_mem_requirements rejects a NULL out-param");

    if (check_rate_whitelist) {
        AudioPipelineConfig bad_rate = audio_pipeline_default_config(44100);
        CHECK(audio_pipeline_get_mem_requirements(&bad_rate, &req) == -1,
              "get_mem_requirements rejects sample_rate=44100 (rate whitelist)");
    }

    CHECK(audio_pipeline_get_mem_requirements(&cfg, &req) == 0 && req.bytes > 0,
          fmt_msg("get_mem_requirements accepts %d Hz / FFT %d", sr, fft_size));
    printf("       (%d Hz / FFT %d descriptor: descriptor_version=%u bytes=%llu alignment=%u "
           "layout_version=%u backend_id=%u build_flags_hash=0x%08x)\n",
           sr, fft_size, req.descriptor_version, (unsigned long long)req.bytes, req.alignment,
           req.layout_version, req.backend_id, req.build_flags_hash);
}

static void test_pool_rejection(int sr, int fft_size) {
    AudioPipelineConfig cfg = grid_config(sr, fft_size);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for pool-rejection test @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    /* Extra 16 bytes of headroom so a +1-byte-offset "misaligned" pointer
     * still has req.bytes of addressable space behind it (the alignment
     * check must reject it before any of that space is touched, but the
     * allocation itself must not be an OOB setup). */
    void* pool = NULL;
    if (posix_memalign(&pool, 16, req.bytes + 16) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for pool-rejection test @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    AudioPipeline* p_misaligned = audio_pipeline_init((uint8_t*)pool + 1, req.bytes, &cfg);
    CHECK(p_misaligned == NULL,
          fmt_msg("audio_pipeline_init rejects a misaligned (base+1) pool @ %d Hz", sr));

    AudioPipeline* p_short = audio_pipeline_init(pool, req.bytes - 1, &cfg);
    CHECK(p_short == NULL,
          fmt_msg("audio_pipeline_init rejects an undersized (bytes-1) pool @ %d Hz", sr));

    AudioPipeline* p_ok = audio_pipeline_init(pool, req.bytes, &cfg);
    CHECK(p_ok != NULL,
          fmt_msg("audio_pipeline_init accepts a correctly aligned/sized pool @ %d Hz", sr));
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

static void test_create_vs_init_parity(int sr, int fft_size, int hop_count) {
    AudioPipelineConfig cfg = grid_config(sr, fft_size);

    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for create-vs-init parity test @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    AudioPipeline* p_heap = audio_pipeline_create(&cfg);
    if (!p_heap) {
        fprintf(stderr, "FAIL: audio_pipeline_create @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for create-vs-init parity test @ %d Hz\n", sr);
        g_failures++;
        audio_pipeline_destroy(p_heap);
        return;
    }
    memset(pool, 0xA5, req.bytes);   /* dirty pool: init must not rely on zeros */

    AudioPipeline* p_pool = audio_pipeline_init(pool, req.bytes, &cfg);
    if (!p_pool) {
        fprintf(stderr, "FAIL: audio_pipeline_init on a poisoned (0xA5) pool @ %d Hz\n", sr);
        g_failures++;
        audio_pipeline_destroy(p_heap);
        free(pool);
        return;
    }

    int hop = audio_pipeline_hop_size(p_heap);
    CHECK(hop > 0 && hop == audio_pipeline_hop_size(p_pool),
          fmt_msg("heap and pool instances agree on hop_size @ %d Hz", sr));

    float* out_heap = (float*)malloc((size_t)hop_count * (size_t)hop * sizeof(float));
    float* out_pool = (float*)malloc((size_t)hop_count * (size_t)hop * sizeof(float));

    run_hops(p_heap, hop, hop_count, out_heap);
    run_hops(p_pool, hop, hop_count, out_pool);

    int byte_equal = memcmp(out_heap, out_pool, (size_t)hop_count * (size_t)hop * sizeof(float)) == 0;
    CHECK(byte_equal,
          fmt_msg("audio_pipeline_create (heap) == audio_pipeline_init (0xA5-poisoned pool) @ %d Hz, "
                  "%d hops, byte-for-byte", sr, hop_count));

    int finite = 1;
    for (int i = 0; i < hop_count * hop; i++) {
        if (out_heap[i] != out_heap[i]) { finite = 0; break; }   /* NaN check */
    }
    CHECK(finite, fmt_msg("%d-hop synthetic run @ %d Hz produces no NaN in the output", hop_count, sr));

    free(out_heap);
    free(out_pool);
    audio_pipeline_destroy(p_heap);   /* frees its own owned pool            */
    audio_pipeline_destroy(p_pool);   /* pool path: no-op, caller keeps pool */
    free(pool);
}

static void test_destroy_idempotence(int sr, int fft_size) {
    AudioPipelineConfig cfg = grid_config(sr, fft_size);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for destroy-idempotence test @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, req.alignment, req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for destroy-idempotence test @ %d Hz\n", sr);
        g_failures++;
        return;
    }

    AudioPipeline* p = audio_pipeline_init(pool, req.bytes, &cfg);
    if (!p) {
        fprintf(stderr, "FAIL: audio_pipeline_init for destroy-idempotence test @ %d Hz\n", sr);
        g_failures++;
        free(pool);
        return;
    }

    audio_pipeline_destroy(p);
    audio_pipeline_destroy(p);      /* second call on the same pool-resident instance */
    audio_pipeline_destroy(NULL);   /* NULL-safe */
    printf("PASS: audio_pipeline_destroy is idempotent (2x) and NULL-safe on a "
           "pool-resident instance @ %d Hz\n", sr);

    /* The pool itself must be untouched/reusable: destroy() on a
     * pool-resident instance never frees `pool` (the caller owns it). */
    AudioPipeline* p2 = audio_pipeline_init(pool, req.bytes, &cfg);
    CHECK(p2 != NULL, fmt_msg("pool is reusable via a fresh audio_pipeline_init after destroy @ %d Hz", sr));
    if (p2) audio_pipeline_destroy(p2);

    free(pool);
}

/* derive_dims_and_configs()'s reject-first AudioPipelineConfig validation
 * (audio_pipeline.c) -- an out-of-enum aec_preset/nr_mode and a
 * bool-typed field outside {0,1} must be rejected by BOTH
 * audio_pipeline_get_mem_requirements() and audio_pipeline_init(), not just
 * silently fall through to a module's own internal enum-default fallback
 * or be treated as truthy. Run once (config-validation is not a per-rate
 * property); 16000 is an arbitrary representative rate. */
static void test_config_validation_rejects(void) {
    AudioPipelineMemReq req;

    AudioPipelineConfig bad_preset = audio_pipeline_default_config(16000);
    bad_preset.aec_preset = (AecPreset)99;
    CHECK(audio_pipeline_get_mem_requirements(&bad_preset, &req) == -1,
          "get_mem_requirements rejects an out-of-enum aec_preset");

    AudioPipelineConfig bad_nr_mode = audio_pipeline_default_config(16000);
    bad_nr_mode.nr_mode = (MmseLsaNrMode)99;
    CHECK(audio_pipeline_get_mem_requirements(&bad_nr_mode, &req) == -1,
          "get_mem_requirements rejects an out-of-enum nr_mode");

    AudioPipelineConfig bad_aec_only = audio_pipeline_default_config(16000);
    bad_aec_only.aec_only = 2;
    CHECK(audio_pipeline_get_mem_requirements(&bad_aec_only, &req) == -1,
          "get_mem_requirements rejects aec_only=2 (bool must be 0/1)");

    AudioPipelineConfig bad_cng = audio_pipeline_default_config(16000);
    bad_cng.enable_cng = 2;
    CHECK(audio_pipeline_get_mem_requirements(&bad_cng, &req) == -1,
          "get_mem_requirements rejects enable_cng=2 (bool must be 0/1)");

    AudioPipelineConfig bad_legacy = audio_pipeline_default_config(16000);
    bad_legacy.legacy_amin = -1;
    CHECK(audio_pipeline_get_mem_requirements(&bad_legacy, &req) == -1,
          "get_mem_requirements rejects legacy_amin=-1 (bool must be 0/1)");

    AudioPipelineConfig bad_grid = audio_pipeline_default_config(48000);
    bad_grid.fft_size = 512;
    CHECK(audio_pipeline_get_mem_requirements(&bad_grid, &req) == -1,
          "get_mem_requirements rejects FFT 512 at 48 kHz");

    /* Same rejections must hold on the audio_pipeline_init() entry point too
     * (derive_dims_and_configs is the ONE gate both funnel through) -- build
     * a correctly-sized/aligned pool off a KNOWN-GOOD config, then hand
     * init() a bad config against that same pool. */
    AudioPipelineConfig good = audio_pipeline_default_config(16000);
    AudioPipelineMemReq good_req;
    if (audio_pipeline_get_mem_requirements(&good, &good_req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for config-validation init test\n");
        g_failures++;
        return;
    }
    void* pool = NULL;
    if (posix_memalign(&pool, good_req.alignment, good_req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for config-validation init test\n");
        g_failures++;
        return;
    }

    AudioPipelineConfig bad_init_enum = good;
    bad_init_enum.nr_mode = (MmseLsaNrMode)99;
    AudioPipeline* p1 = audio_pipeline_init(pool, good_req.bytes, &bad_init_enum);
    CHECK(p1 == NULL, "audio_pipeline_init rejects an out-of-enum nr_mode");
    if (p1) audio_pipeline_destroy(p1);

    AudioPipelineConfig bad_init_bool = good;
    bad_init_bool.enable_cng = 2;
    AudioPipeline* p2 = audio_pipeline_init(pool, good_req.bytes, &bad_init_bool);
    CHECK(p2 == NULL, "audio_pipeline_init rejects enable_cng=2 (bool must be 0/1)");
    if (p2) audio_pipeline_destroy(p2);

    /* The pool must still be usable afterward -- a rejected init() must not
     * have partially carved/corrupted it. */
    AudioPipeline* p_ok = audio_pipeline_init(pool, good_req.bytes, &good);
    CHECK(p_ok != NULL, "pool is still usable via a valid config after rejected init() attempts");
    if (p_ok) audio_pipeline_destroy(p_ok);

    free(pool);
}

/* audio_pipeline_init_ex()'s `expected` descriptor gate. Run once (16000 Hz,
 * an arbitrary representative rate) -- like the config validation above,
 * this exercises a comparison the
 * function does against a freshly-recomputed AudioPipelineMemReq, not a
 * per-rate carve property; see audio_pipeline.h's audio_pipeline_init_ex()
 * doc for the exact seven-condition contract this drills. */
static void test_init_ex_descriptor(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for init_ex descriptor test\n");
        g_failures++;
        return;
    }

    void* pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc for init_ex descriptor test\n");
        g_failures++;
        return;
    }

    AudioPipeline* p_ok = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p_ok != NULL, "audio_pipeline_init_ex accepts a correct/current descriptor");
    if (p_ok) audio_pipeline_destroy(p_ok);

    AudioPipeline* p_null = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, NULL);
    CHECK(p_null != NULL,
          "audio_pipeline_init_ex(expected=NULL) accepts, same as audio_pipeline_init");
    if (p_null) audio_pipeline_destroy(p_null);

    AudioPipelineMemReq bad_dv = req;
    bad_dv.descriptor_version = req.descriptor_version + 1;
    AudioPipeline* p_dv = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_dv);
    CHECK(p_dv == NULL, "audio_pipeline_init_ex rejects a tampered descriptor_version");
    if (p_dv) audio_pipeline_destroy(p_dv);

    AudioPipelineMemReq bad_lv = req;
    bad_lv.layout_version = req.layout_version + 1;
    AudioPipeline* p_lv = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_lv);
    CHECK(p_lv == NULL, "audio_pipeline_init_ex rejects a tampered layout_version");
    if (p_lv) audio_pipeline_destroy(p_lv);

    /* Tampered backend_id: a plain wrong integer (99), never a string --
     * V2 dropped the F20 `const char* backend` field entirely, so there is
     * no string to tamper any more (see AudioPipelineMemReq.backend_id's
     * doc for why that hazard is gone). */
    AudioPipelineMemReq bad_backend_id = req;
    bad_backend_id.backend_id = 99;
    AudioPipeline* p_be = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_backend_id);
    CHECK(p_be == NULL, "audio_pipeline_init_ex rejects a tampered backend_id");
    if (p_be) audio_pipeline_destroy(p_be);

    AudioPipelineMemReq bad_hash = req;
    bad_hash.build_flags_hash = req.build_flags_hash ^ 0xFFFFFFFFu;
    AudioPipeline* p_hash = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_hash);
    CHECK(p_hash == NULL, "audio_pipeline_init_ex rejects a tampered build_flags_hash");
    if (p_hash) audio_pipeline_destroy(p_hash);

    AudioPipelineMemReq bad_align = req;
    bad_align.alignment = req.alignment * 2;
    AudioPipeline* p_align = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_align);
    CHECK(p_align == NULL, "audio_pipeline_init_ex rejects a tampered alignment");
    if (p_align) audio_pipeline_destroy(p_align);

    /* `reserved` is documented as always zero in any descriptor
     * this library produced, and `expected` may arrive from persisted bytes
     * -- init_ex must VALIDATE the claim, not assume it. */
    AudioPipelineMemReq bad_reserved = req;
    bad_reserved.reserved = 1u;
    AudioPipeline* p_res = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_reserved);
    CHECK(p_res == NULL, "audio_pipeline_init_ex rejects a nonzero reserved field");
    if (p_res) audio_pipeline_destroy(p_res);

    AudioPipelineMemReq bad_bytes = req;
    bad_bytes.bytes = req.bytes - 1;
    AudioPipeline* p_bytes = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &bad_bytes);
    CHECK(p_bytes == NULL,
          "audio_pipeline_init_ex rejects expected->bytes smaller than the current requirement");
    if (p_bytes) audio_pipeline_destroy(p_bytes);

    AudioPipeline* p_short = audio_pipeline_init_ex(pool, (size_t)(req.bytes - 1), &cfg, &req);
    CHECK(p_short == NULL,
          "audio_pipeline_init_ex rejects an undersized pool even with a correct descriptor");
    if (p_short) audio_pipeline_destroy(p_short);

    AudioPipeline* p_final = audio_pipeline_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p_final != NULL, "pool is still usable via audio_pipeline_init_ex after rejected attempts");
    if (p_final) audio_pipeline_destroy(p_final);

    free(pool);
}

int main(void) {
    for (int r = 0; r < N_GRIDS; r++) {
        int sr = GRIDS[r].sample_rate;
        int fft_size = GRIDS[r].fft_size;
        int hop_count = hop_count_for_rate(sr);
        printf("\n=== sample_rate = %d Hz / FFT %d (hop_count=%d) ===\n",
               sr, fft_size, hop_count);
        test_validation(sr, fft_size, r == 0); /* 44100 rejection checked once */
        test_pool_rejection(sr, fft_size);
        test_create_vs_init_parity(sr, fft_size, hop_count);
        test_destroy_idempotence(sr, fft_size);
    }

    printf("\n=== AudioPipelineConfig reject-first validation ===\n");
    test_config_validation_rejects();

    printf("\n=== audio_pipeline_init_ex descriptor gate ===\n");
    test_init_ex_descriptor();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
