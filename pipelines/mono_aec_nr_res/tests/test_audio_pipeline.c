/**
 * tests/test_audio_pipeline.c — acceptance tests for
 * mono_aec_nr_res/audio_pipeline.h.
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
 *   make test                     # ne10 backend (default everywhere)
 *   make BACKEND=kiss test        # portable/bit-reproducible reference backend
 *
 * Standalone (no Makefile; kiss reference variant shown -- swap kiss->ne10
 * in the define and the audio_common archive path for the default backend):
 *   cc -O2 -std=gnu99 -I../lib/aec/c_impl/include -I../lib/aec/c_impl/example \
 *      -I../lib/nr/c_impl/include -I../../audio_common/include \
 *      -DAUDIO_PIPELINE_BACKEND_STR=\"kiss\" \
 *      tests/test_audio_pipeline.c audio_pipeline.c \
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
#include "aec3_balanced_config.h"   /* AEC3B_SQRT2_SIN_LUT -- the comfort-noise table */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   /* clock_gettime -- the timing test's own wall clock */
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

/* An echo path a warm run can converge on and the delay estimator can lock
 * onto, kept separate from run_hops(): that one deliberately re-seeds the LCG
 * per call so two instances see identical bytes, which is the wrong shape for
 * a run that has to CONTINUE past a reset. `hist` carries the far tail across
 * hops so the delay is real rather than a per-hop wrap. */
static void reset_scene_hop(unsigned int* rng, float* mic, float* ref, int hop,
                            int h, float* hist, int hist_len,
                            int delay, float erl) {
    float gain = ((h % 40) < 30) ? 0.4f : 0.0f;
    for (int i = 0; i < hop; i++) {
        *rng = *rng * 1103515245u + 12345u;
        ref[i] = gain * ((float)((*rng >> 9) & 0x7fffff) / 8388608.0f * 2.0f - 1.0f);
    }
    for (int i = 0; i < hop; i++) {
        int back = delay + (hop - 1 - i);
        float echo = (back < hist_len) ? hist[back] : 0.0f;
        *rng = *rng * 1103515245u + 12345u;
        float n1 = (float)((*rng >> 9) & 0x7fffff) / 8388608.0f * 2.0f - 1.0f;
        *rng = *rng * 1103515245u + 12345u;
        float n2 = (float)((*rng >> 9) & 0x7fffff) / 8388608.0f * 2.0f - 1.0f;
        mic[i] = erl * echo + (((h % 130) < 35) ? 0.25f * n1 : 0.0f) + 0.01f * n2;
    }
    memmove(hist + hop, hist, (size_t)(hist_len - hop) * sizeof(float));
    for (int i = 0; i < hop; i++) hist[hop - 1 - i] = ref[i];
}

/* audio_pipeline_reset()'s documented contract (audio_pipeline.h): equivalent
 * to a fresh audio_pipeline_init() on the same pool/cfg. That was prose until
 * the AEC's own reset was made to deliver it -- state survived aec_reset()
 * that a fresh instance never has, so the first post-reset hop of this
 * pipeline already differed from a never-warmed twin. Run with CNG both ON
 * and OFF: reset re-seeds the comfort-noise RNG to the same construction-time
 * seed, so the CNG path is compared for real rather than excused. */
static void test_reset_equals_fresh_instance(int sr, int fft_size,
                                             int enable_cng) {
    enum { HIST = 8192, WARM = 600, COMPARE = 300 };
    AudioPipelineConfig cfg = grid_config(sr, fft_size);
    cfg.enable_cng = enable_cng;

    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "FAIL: setup (get_mem_requirements) for reset parity @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    void* pool_a = NULL;
    void* pool_b = NULL;
    if (posix_memalign(&pool_a, req.alignment, req.bytes) != 0 ||
        posix_memalign(&pool_b, req.alignment, req.bytes) != 0) {
        fprintf(stderr, "FAIL: pool alloc for reset parity @ %d Hz\n", sr);
        g_failures++;
        free(pool_a); free(pool_b);
        return;
    }
    AudioPipeline* fresh  = audio_pipeline_init(pool_a, req.bytes, &cfg);
    AudioPipeline* warmed = audio_pipeline_init(pool_b, req.bytes, &cfg);
    if (!fresh || !warmed) {
        fprintf(stderr, "FAIL: audio_pipeline_init for reset parity @ %d Hz\n", sr);
        g_failures++;
        free(pool_a); free(pool_b);
        return;
    }

    int hop = audio_pipeline_hop_size(fresh);
    float* mic  = (float*)malloc((size_t)hop * sizeof(float));
    float* ref  = (float*)malloc((size_t)hop * sizeof(float));
    float* out_a = (float*)malloc((size_t)hop * sizeof(float));
    float* out_b = (float*)malloc((size_t)hop * sizeof(float));
    float* hist  = (float*)calloc(HIST, sizeof(float));

    /* Warm on an echo path the compare phase does NOT reuse, so anything the
     * subject remembers is wrong for what follows. */
    unsigned int rng = 1234567u;
    for (int h = 0; h < WARM; h++) {
        reset_scene_hop(&rng, mic, ref, hop, h, hist, HIST, 611, 0.65f);
        audio_pipeline_process(warmed, mic, ref, out_b);
    }
    audio_pipeline_reset(warmed);

    unsigned int rng_a = 0x89abcdefu;
    unsigned int rng_b = 0x89abcdefu;
    float* hist_a = (float*)calloc(HIST, sizeof(float));
    float* hist_b = (float*)calloc(HIST, sizeof(float));
    long differing = 0;
    int first_bad = -1;
    for (int h = 0; h < COMPARE; h++) {
        reset_scene_hop(&rng_a, mic, ref, hop, h, hist_a, HIST, 293, 0.5f);
        audio_pipeline_process(fresh, mic, ref, out_a);
        reset_scene_hop(&rng_b, mic, ref, hop, h, hist_b, HIST, 293, 0.5f);
        audio_pipeline_process(warmed, mic, ref, out_b);
        if (memcmp(out_a, out_b, (size_t)hop * sizeof(float)) != 0) {
            differing++;
            if (first_bad < 0) first_bad = h;
        }
    }
    CHECK(differing == 0,
          fmt_msg("audio_pipeline_reset == a fresh instance @ %d Hz, cng=%d: "
                  "%ld of %d post-reset hops differ (first at %d)",
                  sr, enable_cng, differing, COMPARE, first_bad));

    free(mic); free(ref); free(out_a); free(out_b);
    free(hist); free(hist_a); free(hist_b);
    audio_pipeline_destroy(fresh);
    audio_pipeline_destroy(warmed);
    free(pool_a); free(pool_b);
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

    AudioPipelineConfig bad_delay = audio_pipeline_default_config(16000);
    bad_delay.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    bad_delay.delay_num_filters = 2;
    CHECK(audio_pipeline_get_mem_requirements(&bad_delay, &req) == -1,
          "EXTERNAL delay mode rejects an inapplicable matched-filter count");

    AudioPipelineConfig matched5 = audio_pipeline_default_config(16000);
    AudioPipelineConfig matched2 = matched5;
    AudioPipelineConfig external = matched5;
    AudioPipelineMemReq r5, r2, re;
    matched2.delay_num_filters = 2;
    external.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    CHECK(audio_pipeline_get_mem_requirements(&matched5, &r5) == 0 &&
          audio_pipeline_get_mem_requirements(&matched2, &r2) == 0 &&
          audio_pipeline_get_mem_requirements(&external, &re) == 0 &&
          r5.bytes > r2.bytes && r2.bytes > re.bytes,
          "mono config passes delay mode/bank size into AEC pool sizing");

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

/* ---------------------------------------------------------------------------
 * Per-stage timing
 *
 * The numbers must be REAL measurements, not merely present. Two properties
 * carry that: every stage is bounded by the wall time of the call it sits
 * inside (a stray or uninitialised value would almost certainly exceed it),
 * and each half is asserted against its OWN build flag, so a compiled-out
 * build is required to report zeros rather than stale values.
 *
 * This pipeline has no accepted-then-bail path -- audio_pipeline_process()
 * validates its arguments and then runs to completion -- so the 4ch suite's
 * non-finite-hop case has no counterpart here and is deliberately absent
 * rather than written as an assertion that cannot fail. What replaces it is
 * the aec_only case on a POISONED pool: that instance never writes nr/post/
 * synth, so those three read whatever the caller's memory held unless the
 * control block was zeroed at init.
 *
 * These are a real gate on the control block's init-time
 * memset(p, 0, sizeof(*p)): remove it and they go red. That is the only
 * thing keeping an aec_only instance's nr/post/synth at zero, since no
 * path ever writes them.
 * ------------------------------------------------------------------------- */
static uint32_t wall_us_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}

static void test_stage_timing(void) {
    AudioPipelineConfig cfg = grid_config(16000, 256);
    AudioPipeline* p = audio_pipeline_create(&cfg);
    AudioPipelineLastTiming t, zero;
    uint32_t wall_us = 0, sum;
    int hop, i;

    CHECK(p != NULL, "timing: pipeline created");
    if (!p) return;
    hop = audio_pipeline_hop_size(p);

    /* A NULL pipeline zeroes the whole record rather than failing. Compared
     * as a struct, which is where a field-by-field copy that forgets one
     * would hide. */
    memset(&zero, 0, sizeof(zero));
    memset(&t, 0xa5, sizeof(t));
    audio_pipeline_get_last_timing(NULL, &t);
    CHECK(memcmp(&t, &zero, sizeof(t)) == 0,
          "timing: a NULL pipeline zeroes the record");

    /* Warm up so the filter is doing real work, then time the FINAL hop
     * around the same call the library stamps inside. */
    {
        float* mic = (float*)malloc((size_t)hop * sizeof(float));
        float* ref = (float*)malloc((size_t)hop * sizeof(float));
        float* out = (float*)malloc((size_t)hop * sizeof(float));
        int h;
        CHECK(mic && ref && out, "timing: allocation");
        if (mic && ref && out) {
            lcg_state = 0xC0FFEEu;
            for (h = 0; h < 200; h++) {
                for (i = 0; i < hop; i++) {
                    ref[i] = lcg_sample();
                    mic[i] = 0.3f * ref[i] + 0.05f * lcg_sample();
                }
                if (h == 199) {
                    uint32_t t_in = wall_us_now();
                    audio_pipeline_process(p, mic, ref, out);
                    wall_us = wall_us_now() - t_in;
                } else {
                    audio_pipeline_process(p, mic, ref, out);
                }
            }
        }
        free(mic); free(ref); free(out);
    }
    audio_pipeline_get_last_timing(p, &t);

    /* ⚠ Written so it can FAIL: over a full hop the AEC runs a main filter,
     * a pre-filter stage and a post/RES block, and this layer runs an NR
     * gain, the gain arithmetic and a synthesis -- on this grid the delay
     * bank, the main filter and the post/RES block cannot cost under a
     * microsecond, so a zero there means that window never ran. post_us and
     * synth_us are NOT asserted nonzero: they
     * sit at the 1 us clock floor on the development host, and an assertion
     * that depends on the host being slow is a flake, not a test.
     *
     * Each half is asserted against its own flag because the two are
     * separate builds: AUDIO_PIPELINE_STAGE_TIMING governs the stages
     * measured here, AEC_STAGE_TIMING governs the three copied out of
     * aec_get_last_timing(). The zero branches are not filler -- they are
     * what proves a flag removes the measurement rather than merely hiding
     * it. */
#if AUDIO_PIPELINE_STAGE_TIMING
    CHECK(t.nr_us > 0, "timing: the NR gain reports a real cost");
#else
    CHECK(t.nr_us == 0 && t.post_us == 0 && t.synth_us == 0,
          "timing: this layer's stages read zero when compiled out");
#endif
#if AUDIO_PIPELINE_STAGE_TIMING && !AEC_STAGE_TIMING
    /* Not an error: the halves are independent by design. But `make
     * PROFILE=1` sets both, so this combination means the flags were
     * hand-written -- or that the build lost one of them. Say so, because
     * the zero branch below agrees with a library that DID measure while
     * this layer's copy was compiled out, and would pass without proving
     * anything. The four-lane test catches that case by reading the lanes
     * unconditionally; this one cannot. */
    printf("NOTE: this layer measures but lib/aec's half is off -- the AEC\n"
           "      zero check below cannot distinguish 'not measured' from\n"
           "      'measured and not copied out'.\n");
#endif
#if AEC_STAGE_TIMING
    CHECK(t.aec.delay_us > 0,  "timing: the delay estimator reports a real cost");
    CHECK(t.aec.linear_us > 0, "timing: the AEC main filter reports a real cost");
    CHECK(t.aec.res_us > 0,    "timing: the AEC post/RES block reports a real cost");
    /* frontend_us is deliberately NOT asserted nonzero. Once delay_us was
     * split out of it, what remains -- mic HPF, saturation, render activity,
     * mu_scale, mic-clip, RSA, shadow filter -- measures ~1 us on this grid,
     * i.e. at the clock floor. Asserting it would be asserting that the host
     * is slow. That it can now read 0 while delay_us is large IS the
     * measurement this split exists to make. */
    CHECK(t.aec.delay_us > t.aec.frontend_us,
          "timing: the delay estimator dominates what is left of the frontend");
#else
    CHECK(t.aec.delay_us == 0 && t.aec.frontend_us == 0 && t.aec.linear_us == 0 &&
          t.aec.res_us == 0,
          "timing: the AEC stages read zero when compiled out");
#endif

    /* Every stage is bounded by the call that contains it. This is what
     * separates a measurement from a number. */
    sum = t.aec.delay_us + t.aec.frontend_us + t.aec.linear_us + t.aec.res_us
        + t.nr_us + t.post_us + t.synth_us;
    CHECK(sum <= wall_us,
          "timing: the stages must fit inside audio_pipeline_process's wall time");

    /* A REJECTED call must leave the record alone -- it describes the last
     * hop that ran, and no hop ran here. */
    {
        float* ref = (float*)calloc((size_t)hop, sizeof(float));
        float* out = (float*)calloc((size_t)hop, sizeof(float));
        AudioPipelineLastTiming after;
        CHECK(audio_pipeline_process(p, NULL, ref, out) == -1,
              "timing: a NULL mic pointer is rejected");
        audio_pipeline_get_last_timing(p, &after);
        CHECK(memcmp(&after, &t, sizeof(t)) == 0,
              "timing: a rejected call leaves the record untouched");
        free(ref); free(out);
    }
    audio_pipeline_destroy(p);

    /* aec_only on a POISONED pool: nr/post/synth are stages that do not
     * exist in that mode, so they must read zero even though nothing ever
     * writes them and the memory underneath arrived full of 0xA5. */
    {
        AudioPipelineConfig ao = grid_config(16000, 256);
        AudioPipelineMemReq req;
        void* pool = NULL;
        ao.aec_only = 1;
        if (audio_pipeline_get_mem_requirements(&ao, &req) == 0 &&
            posix_memalign(&pool, req.alignment, req.bytes) == 0) {
            AudioPipeline* q;
            memset(pool, 0xA5, req.bytes);
            q = audio_pipeline_init(pool, req.bytes, &ao);
            CHECK(q != NULL, "timing: aec_only init on a poisoned pool");
            if (q) {
                AudioPipelineLastTiming ao_t;
                int qhop = audio_pipeline_hop_size(q);
                float* m = (float*)calloc((size_t)qhop, sizeof(float));
                float* r = (float*)calloc((size_t)qhop, sizeof(float));
                float* o = (float*)calloc((size_t)qhop, sizeof(float));

                /* Read BEFORE any hop: the record must already be defined. */
                audio_pipeline_get_last_timing(q, &ao_t);
                CHECK(memcmp(&ao_t, &zero, sizeof(ao_t)) == 0,
                      "timing: a fresh instance on a poisoned pool reads all "
                      "zeros before the first hop");

                lcg_state = 0xC0FFEEu;
                for (i = 0; i < qhop; i++) { r[i] = lcg_sample(); m[i] = 0.3f * r[i]; }
                audio_pipeline_process(q, m, r, o);
                audio_pipeline_get_last_timing(q, &ao_t);
                CHECK(ao_t.nr_us == 0 && ao_t.post_us == 0 && ao_t.synth_us == 0,
                      "timing: aec_only reports zero for the stages it does "
                      "not have, on a poisoned pool");
                free(m); free(r); free(o);
                audio_pipeline_destroy(q);
            }
            free(pool);
        }
    }

    printf("timing: delay=%u frontend=%u linear=%u res=%u | nr=%u post=%u "
           "synth=%u (call wall %uus)\n",
           t.aec.delay_us, t.aec.frontend_us, t.aec.linear_us, t.aec.res_us,
           t.nr_us, t.post_us, t.synth_us, wall_us);
}

/* ---- comfort-noise contract ----------------------------------------------
 *
 * The pipeline fills AEC-suppressed bins from lib/aec's own comfort-noise
 * recipe: an LCG step per bin whose top five bits index AEC3B_SQRT2_SIN_LUT,
 * the imaginary part reading a quarter turn (+8 of 32) ahead of the real one.
 * The three properties the injected noise is required to have -- bounded
 * samples, the same expected power as the unit-variance Gaussian pair the
 * recipe replaced, and a sequence that actually advances -- are what this
 * pins. The exact output samples deliberately are NOT pinned: the recipe is
 * specified statistically, so a byte assertion here would only re-state the
 * arithmetic.
 *
 * Part 1 checks the shared table itself (a table edit moves every
 * CNG-enabled render's noise power). Part 2 checks the pipeline actually
 * draws from it and advances its state, by differencing a CNG-on run against
 * a byte-identical CNG-off one -- comfort noise is added after every gain,
 * so that difference IS the injected noise and nothing else. */
static void test_comfort_noise_contract(void) {
    const int sr = 16000, fft_size = 256, hops = 400;
    double sum = 0.0, sq = 0.0, worst_pair = 0.0;
    float bound = 0.0f;
    int i;

    for (i = 0; i < 32; i++) {
        float v = AEC3B_SQRT2_SIN_LUT[i];
        float q = AEC3B_SQRT2_SIN_LUT[(i + 8) & 31];
        double power = (double)v * v + (double)q * q;
        if (fabsf(v) > bound) bound = fabsf(v);
        sum += v;
        sq += (double)v * v;
        if (fabs(power - 2.0) > worst_pair) worst_pair = fabs(power - 2.0);
    }
    CHECK(bound <= 1.41421366f,
          "cng: every table entry is bounded by sqrt(2) (the injected noise "
          "has no unbounded tail)");
    CHECK(fabs(sum / 32.0) < 1e-6,
          "cng: the table is zero-mean over a uniform index (no DC injected)");
    CHECK(fabs(sq / 32.0 - 1.0) < 1e-6,
          "cng: the table has unit mean square -- same expected per-bin power "
          "as the unit-variance Gaussian pair it replaced");
    CHECK(worst_pair < 1e-6,
          "cng: real and imaginary entries are a quarter turn apart, so each "
          "bin's injected power is exactly 2*amplitude^2");

    {
        AudioPipelineConfig on = grid_config(sr, fft_size);
        AudioPipelineConfig off = grid_config(sr, fft_size);
        AudioPipeline* p_on;
        AudioPipeline* p_off;
        float *mic, *ref, *o_on, *o_off, *prev_noise, *noise_stream;
        int hop, h, k, injected = 0, all_finite = 1;
        int prev_active = 0, corr_n = 0;
        double noise_sq = 0.0, corr_sum = 0.0;

        off.enable_cng = 0;
        p_on = audio_pipeline_create(&on);
        p_off = audio_pipeline_create(&off);
        CHECK(p_on != NULL && p_off != NULL, "cng: both instances create");
        if (!p_on || !p_off) { audio_pipeline_destroy(p_on); audio_pipeline_destroy(p_off); return; }

        hop = audio_pipeline_hop_size(p_on);
        mic = (float*)calloc((size_t)hop, sizeof(float));
        ref = (float*)calloc((size_t)hop, sizeof(float));
        o_on = (float*)calloc((size_t)hop, sizeof(float));
        o_off = (float*)calloc((size_t)hop, sizeof(float));
        prev_noise = (float*)calloc((size_t)hop, sizeof(float));
        noise_stream = (float*)calloc((size_t)hops * hop, sizeof(float));

        lcg_state = 0x5EEDu;
        for (h = 0; h < hops; h++) {
            for (k = 0; k < hop; k++) { ref[k] = lcg_sample(); mic[k] = 0.5f * ref[k]; }
            audio_pipeline_process(p_on, mic, ref, o_on);
            audio_pipeline_process(p_off, mic, ref, o_off);
            {
                double hop_sq = 0.0, cross = 0.0, prev_sq = 0.0;
                for (k = 0; k < hop; k++) {
                    float noise = o_on[k] - o_off[k];
                    noise_stream[(size_t)h * hop + k] = noise;
                    if (!isfinite(o_on[k]) || !isfinite(noise)) all_finite = 0;
                    noise_sq += (double)noise * noise;
                    hop_sq += (double)noise * noise;
                    cross += (double)noise * prev_noise[k];
                    prev_sq += (double)prev_noise[k] * prev_noise[k];
                    if (noise != 0.0f) injected = 1;
                }
                /* A generator whose state never advanced would hand every bin
                 * of every hop the same table entry, so each hop's injected
                 * spectrum would be the amplitude profile times one fixed
                 * complex constant -- successive hops would then be nearly
                 * collinear. Advancing per bin decorrelates them. */
                if (hop_sq > 0.0 && prev_active && prev_sq > 0.0) {
                    corr_sum += fabs(cross / sqrt(hop_sq * prev_sq));
                    corr_n++;
                }
                prev_active = (hop_sq > 0.0);
                if (prev_active)
                    for (k = 0; k < hop; k++) prev_noise[k] = o_on[k] - o_off[k];
            }
        }
        CHECK(all_finite, "cng: every CNG-on sample and every injected sample "
                          "is finite");
        CHECK(injected && noise_sq > 0.0,
              "cng: comfort noise is actually injected (CNG-on and CNG-off "
              "renders carry different energy)");
        CHECK(corr_n > 20 && corr_sum / corr_n < 0.5,
              "cng: successive hops' injected noise is decorrelated -- the "
              "generator advances instead of re-injecting one fixed spectrum");

        /* The seed is per-INSTANCE, not per-process: a second, independently
         * created instance fed the same input must inject the byte-identical
         * noise stream. A generator that shared one process-wide state across
         * instances would fail here, and so would one seeded from anything
         * outside the instance. */
        {
            AudioPipeline* q_on = audio_pipeline_create(&on);
            AudioPipeline* q_off = audio_pipeline_create(&off);
            int same = 1;
            if (q_on && q_off) {
                lcg_state = 0x5EEDu;
                for (h = 0; h < hops && same; h++) {
                    for (k = 0; k < hop; k++) { ref[k] = lcg_sample(); mic[k] = 0.5f * ref[k]; }
                    audio_pipeline_process(q_on, mic, ref, o_on);
                    audio_pipeline_process(q_off, mic, ref, o_off);
                    for (k = 0; k < hop; k++)
                        if (o_on[k] - o_off[k] != noise_stream[(size_t)h * hop + k]) same = 0;
                }
            }
            CHECK(q_on && q_off && same,
                  "cng: a second, independently created instance injects the "
                  "identical noise stream (the seed is per-instance)");
            audio_pipeline_destroy(q_on);
            audio_pipeline_destroy(q_off);
        }

        printf("cng: %d hops, injected RMS %.3e, hop-to-hop |corr| %.4f over "
               "%d pairs (bound %.8f, table mean %.2e)\n",
               hops, sqrt(noise_sq / ((double)hops * hop)),
               corr_n ? corr_sum / corr_n : -1.0, corr_n, bound, sum / 32.0);
        free(mic); free(ref); free(o_on); free(o_off); free(prev_noise);
        free(noise_stream);
        audio_pipeline_destroy(p_on);
        audio_pipeline_destroy(p_off);
    }
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
        test_reset_equals_fresh_instance(sr, fft_size, /*enable_cng=*/1);
        test_reset_equals_fresh_instance(sr, fft_size, /*enable_cng=*/0);
        test_destroy_idempotence(sr, fft_size);
    }

    printf("\n=== AudioPipelineConfig reject-first validation ===\n");
    test_config_validation_rejects();

    printf("\n=== audio_pipeline_init_ex descriptor gate ===\n");
    test_init_ex_descriptor();

    printf("\n=== per-stage timing ===\n");
    test_stage_timing();

    printf("\n=== comfort-noise contract ===\n");
    test_comfort_noise_contract();

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
