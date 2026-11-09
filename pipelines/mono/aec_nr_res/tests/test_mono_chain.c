/**
 * tests/test_mono_chain.c -- acceptance tests of the mono AEC -> NR -> RES chain as the example
 * programs compose it: lib/aec's Aec (its own RES gain) read through aec_get_res_context(),
 * lib/nr's denoiser through anr_process_gain(), min / apply / comfort noise / synthesis written
 * out per hop.
 * The pool plan and hop of main.c (AEC, denoiser, synthesis, buffers in one aligned
 * block) are repeated here and tested: the pooled instance behaves exactly like the heap one,
 * reset restores a fresh instance, the switches route what they claim to, comfort noise obeys
 * its contract, and the chain equals the hand-written oracle of conv_chain_mono.h bit for bit.
 * The two example programs themselves (aec_nr_pipeline, aec_nr_pipeline_static) run as black
 * boxes on WAV input: the DUMP_CTX output hops equal the oracle bit for bit.
 * LCG synthetic input, a 0xA5-poisoned pool, PASS/FAIL prints, nonzero exit on any failure.
 * Per-rate cases run at 8000 / 16000 / 48000 Hz; 48 kHz runs HOP_COUNT_48K hops because its
 * filter length and grid are much larger.
 * Build and run: `make test` in pipelines/ (arguments: the directory of the example binaries and
 * a scratch directory for the black-box files).
 */
#include "aec.h"
#include "aec_res.h"               /* aec_cng_fill */
#include "anr.h"
#include "audio_stft.h"
#include "utility/simd_kernels.h"
#include "wav_io.h"
#include "conv_chain_mono.h"
#include "guard_scene.h"
#include "fft_wrapper.h"           /* test-side WOLA reference for the RES/NR bypass */
#include "audio_window.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

/* sqrt(2)*sin(2*pi*i/32): the phase table aec_cng_fill indexes (aec_res.h). */
static const float kCngTable[32] = {
    0.0f, 0.27589938f, 0.541196108f, 0.785694957f, 1.0f, 1.17587554f,
    1.30656302f, 1.3870399f, 1.41421354f, 1.3870399f, 1.30656302f,
    1.17587554f, 1.0f, 0.785694957f, 0.541196108f, 0.27589938f,
    1.73191212e-16f, -0.27589938f, -0.541196108f, -0.785694957f, -1.0f,
    -1.17587554f, -1.30656302f, -1.3870399f, -1.41421354f, -1.3870399f,
    -1.30656302f, -1.17587554f, -1.0f, -0.785694957f, -0.541196108f,
    -0.27589938f};

#define HOP_COUNT      1000   /* 8 kHz / 16 kHz */
#define HOP_COUNT_48K   300   /* 48 kHz: larger filter_length/hop/n_freqs, so fewer hops */

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

/* snprintf-into-scratch helper so per-rate CHECK messages can embed the rate. Single static
 * buffer is safe here: CHECK's cond/msg are evaluated synchronously, never nested. */
static char g_msgbuf[256];
static const char* fmt_msg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_msgbuf, sizeof(g_msgbuf), fmt, ap);
    va_end(ap);
    return g_msgbuf;
}

/* ---- LCG synthetic mic/ref generator ---- */
static uint32_t lcg_state;
static float lcg_sample(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

/* ---------------------------------------------------------------------------------------
 * The pool plan and hop of the example programs (main.c): the lane composed from the level,
 * one block holding the AEC, the denoiser, the synthesis and the gain/spectrum buffers.
 * --------------------------------------------------------------------------------------- */

#define CNG_NR_GAIN_FLOOR 0.31622777f
#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))   /* int16^2 -> audio scale, exact */

/* The synthesis borrows the AEC's own window table (aec_synthesis_window()), which exists once the AEC is
 * initialised, after the pool is sized. A borrowed table carves nothing, so any non-NULL pointer sizes the
 * synthesis the same; this one stands in until rig_carve_post() installs the AEC's. */
static const float k_sizing_window[1] = {0.0f};

typedef struct Plan {
    ConvMonoConfig opt;
    AecSignalGrid grid;
    AecConfig aec_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    int enable_cng;
    size_t aec_bytes, nr_bytes, synth_bytes, buf_bytes, total;
} Plan;

static ConvMonoConfig grid_opts(int sr, int fft_size) {
    return conv_mono_default_config(sr, fft_size);
}

/* 0, or -1 when the lane or a stage refuses the request. */
static int plan_make(Plan* p, const ConvMonoConfig* o) {
    memset(p, 0, sizeof(*p));
    p->opt = *o;
    if (!aec_resolve_signal_grid(o->sample_rate,
                                 o->fft_size ? o->fft_size : aec_default_fft_size(o->sample_rate),
                                 &p->grid))
        return -1;
    /* The CLI maps unknown names to balanced; an out-of-enum value is this harness's own refusal
     * (aec_config_from_level would fall back to balanced). */
    if (o->aec_level != AEC_LEVEL_MILD && o->aec_level != AEC_LEVEL_BALANCED &&
        o->aec_level != AEC_LEVEL_AGGRESSIVE && o->aec_level != AEC_LEVEL_NO_FLOOR)
        return -1;
    aec_config_from_level(&p->aec_cfg, o->aec_level, o->sample_rate);
    p->aec_cfg.fft_size = p->grid.fft_size;
    p->aec_cfg.delay_mode = o->delay_mode;
    p->aec_cfg.delay_num_filters = o->delay_num_filters;
    p->aec_cfg.fixed_delay_samples = o->fixed_delay_samples;
    p->aec_cfg.enable_delay_est = o->delay_mode == AEC_DELAY_MATCHED;
    p->aec_cfg.enable_res = 0;
    p->aec_cfg.return_res_context = 1;
    p->aec_bytes = aec_get_mem_size(&p->aec_cfg);
    if (p->aec_bytes == 0) return -1;
    p->total = ALIGN16(p->aec_bytes);
    if (o->aec_only) return 0;
    if (o->enable_nr) {
        if (!anr_nr_level_is_valid(o->nr_level)) return -1;
        p->nr_cfg = anr_config_for_level_grid(o->sample_rate, p->grid.fft_size, o->nr_level);
        p->nr_bytes = anr_get_mem_size(&p->nr_cfg);
        if (p->nr_bytes == 0) return -1;
    }
    ac_stft_config_default(&p->synth_cfg, p->grid.fft_size);
    p->synth_cfg.window = k_sizing_window;
    p->synth_bytes = ac_istft_get_mem_size(&p->synth_cfg);
    if (p->synth_bytes == 0) return -1;
    p->buf_bytes = 3 * ALIGN16((size_t)p->grid.n_freqs * sizeof(float)) +
                   ALIGN16((size_t)p->grid.n_freqs * sizeof(Complex));
    p->enable_cng = p->aec_cfg.enable_cng && o->enable_cng;
    p->total += ALIGN16(p->nr_bytes) + ALIGN16(p->synth_bytes) + p->buf_bytes;
    return 0;
}

/* The comfort-noise seed the examples used before the library's AEC_CNG_SEED; planted as a defect only. */
#define RETIRED_CNG_SEED 0x9e3779b9u

/* Deliberate defects the full-mode identity test plants in a copy of the hop to show it can fail. */
enum { SAB_NO_OLA_CLEAR = 1, SAB_NO_RESEED = 2, SAB_ERROR_SPEC = 4, SAB_OLD_SEED = 8, SAB_RESET_NR = 16 };

typedef struct Rig {
    Aec* aec;
    int with_post;            /* 0 with aec_only */
    int with_res, use_r2, enable_cng, n_freqs;
    AnrDenoiser* nr;          /* NULL without NR */
    AcIstft* istft;
    float *unity, *total_gain, *extra_noise;
    Complex* spec;
    uint32_t rng;
    uint32_t serial;          /* the aec_post_reset_serial() value already handled */
    long guard_hops, reset_hops;
    int sabotage;             /* negative controls of the full-mode identity test (SAB_*), 0 in the chain */
    int hop;
    Aec* heap_aec;            /* heap construction only */
    void* heap_post_mem;
} Rig;

/* The post stages carved from `cur` (at least the plan's post bytes). */
static int rig_carve_post(Rig* r, uint8_t* cur, const Plan* p) {
    size_t vec = ALIGN16((size_t)p->grid.n_freqs * sizeof(float));
    AcStftConfig synth;
    int k, win_n = 0;
    r->with_post = 1;
    r->with_res = p->opt.enable_res;
    r->use_r2 = !p->opt.legacy_amin;
    r->enable_cng = p->enable_cng;
    r->n_freqs = p->grid.n_freqs;
    if (p->opt.enable_nr) {
        r->nr = anr_init(cur, p->nr_bytes, &p->nr_cfg);
        if (!r->nr) return -1;
        cur += ALIGN16(p->nr_bytes);
    }
    synth = p->synth_cfg;      /* the AEC's own synthesis window table, as main.c borrows it */
    if (aec_synthesis_window(r->aec, &synth.window, &win_n) != 0 || win_n != synth.fft_size) return -1;
    r->istft = ac_istft_init(cur, p->synth_bytes, &synth);
    if (!r->istft) return -1;
    cur += ALIGN16(p->synth_bytes);
    r->unity = (float*)cur;       cur += vec;
    r->total_gain = (float*)cur;  cur += vec;
    r->extra_noise = (float*)cur; cur += vec;
    r->spec = (Complex*)cur;
    for (k = 0; k < r->n_freqs; ++k) r->unity[k] = 1.0f;
    r->rng = AEC_CNG_SEED;
    r->serial = aec_post_reset_serial(r->aec);
    return 0;
}

/* The example's carve in `pool` (`bytes` at least plan->total, 16-byte aligned). NULL fields
 * in *r on refusal. */
static int rig_init_pool(Rig* r, void* pool, size_t bytes, const Plan* p) {
    uint8_t* cur = (uint8_t*)pool;
    memset(r, 0, sizeof(*r));
    r->hop = p->grid.hop_size;
    if (!pool || bytes < p->total) return -1;
    r->aec = aec_init(cur, p->aec_bytes, &p->aec_cfg);
    if (!r->aec) return -1;
    if (p->opt.aec_only) return 0;
    if (rig_carve_post(r, cur + ALIGN16(p->aec_bytes), p) != 0) {
        memset(r, 0, sizeof(*r));
        return -1;
    }
    return 0;
}

/* Heap construction: aec_create for the AEC, a separately allocated block for the post stages. */
static int rig_init_heap(Rig* r, const Plan* p) {
    size_t post_bytes = p->total - ALIGN16(p->aec_bytes);
    memset(r, 0, sizeof(*r));
    r->hop = p->grid.hop_size;
    r->heap_aec = (Aec*)calloc(1, sizeof(Aec));
    if (!r->heap_aec || aec_create(r->heap_aec, &p->aec_cfg) != 0) {
        free(r->heap_aec);
        r->heap_aec = NULL;
        return -1;
    }
    r->aec = r->heap_aec;
    if (p->opt.aec_only) return 0;
    if (posix_memalign(&r->heap_post_mem, 16, post_bytes) != 0) return -1;
    memset(r->heap_post_mem, 0x5A, post_bytes);
    return rig_carve_post(r, (uint8_t*)r->heap_post_mem, p);
}

static void rig_free(Rig* r) {
    if (r->nr) anr_destroy(r->nr);
    if (r->aec) aec_destroy(r->aec);
    free(r->heap_aec);
    free(r->heap_post_mem);
    memset(r, 0, sizeof(*r));
}

/* A pool-resident instance's teardown, without forgetting its handles (for the idempotence
 * test). */
static void rig_destroy_only(Rig* r) {
    if (r->nr) anr_destroy(r->nr);
    if (r->aec) aec_destroy(r->aec);
}

/* The hop of main.c: G_res is the lane's own (ctx.res_gain) or unity, G_nr the
 * denoiser's or unity; P = E * min(G_nr, G_res) plus comfort noise on the bins the RES cut. */
static int rig_hop(Rig* r, const float* mic, const float* ref, float* out) {
    AecResContext c;
    AecOutputBase ob;
    const float* res;
    const float* nr;
    uint32_t serial;
    aec_process_context(r->aec, mic, ref);
    aec_get_res_context(r->aec, &c);
    if (!r->with_post) {
        memcpy(out, c.linear_hop, (size_t)c.hop_size * sizeof(float));
        return 0;
    }
    if (!c.error_spec) return -1;
    memset(&ob, 0, sizeof(ob));
    ob.struct_size = sizeof(ob);
    if (aec_get_output_base(r->aec, &ob) != 0) return -1;
    if (ob.guard_active) ++r->guard_hops;
    serial = aec_post_reset_serial(r->aec);
    if (serial != r->serial) {      /* before the first CNG draw and the ac_istft_push of this hop */
        r->serial = serial;
        ++r->reset_hops;
        if (!(r->sabotage & SAB_NO_OLA_CLEAR)) ac_istft_reset(r->istft);
        if (!(r->sabotage & SAB_NO_RESEED)) r->rng = (r->sabotage & SAB_OLD_SEED) ? RETIRED_CNG_SEED : AEC_CNG_SEED;
        if ((r->sabotage & SAB_RESET_NR) && r->nr) anr_reset(r->nr);
    }
    res = r->with_res ? c.res_gain : r->unity;
    nr = r->unity;
    if (!res) return -1;
    if (r->nr) {
        const float* extra = NULL;
        if (r->use_r2 && c.r2 && c.usable_linear) {
            sk_scale_f32(r->extra_noise, c.r2, PSD_SCALE_INV, r->n_freqs);
            extra = r->extra_noise;
        }
        if (anr_process_gain(r->nr, ob.base_spec, extra, NULL) != 0) return -1;
        if (!(nr = anr_get_gain(r->nr, NULL))) return -1;
    }
    sk_min_f32(r->total_gain, nr, res, r->n_freqs);
    sk_capply_gain_f32(r->spec, (r->sabotage & SAB_ERROR_SPEC) ? c.error_spec : ob.base_spec, r->total_gain, r->n_freqs);
    if (r->enable_cng && r->with_res && c.comfort_noise)
        aec_cng_fill(&r->rng, r->spec, c.comfort_noise, res, nr, CNG_NR_GAIN_FLOOR, r->n_freqs);
    (void)ac_istft_push(r->istft, r->spec, out);
    return 0;
}

static void rig_reset(Rig* r) {
    aec_reset(r->aec);
    if (r->nr) anr_reset(r->nr);
    if (r->istft) ac_istft_reset(r->istft);
    r->rng = AEC_CNG_SEED;
    r->serial = aec_post_reset_serial(r->aec);
}

static void* aligned_block(size_t bytes, int fill) {
    void* p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) return NULL;
    memset(p, fill, bytes);
    return p;
}

/* ---- validation: what the lane derivation and the chain sizing refuse or pass through ---- */

static void test_validation(int sr, int fft_size, int check_rate_whitelist) {
    Plan p;
    ConvMonoConfig o = grid_opts(sr, fft_size);

    if (check_rate_whitelist) {
        ConvMonoConfig bad_rate = grid_opts(44100, 0);
        CHECK(plan_make(&p, &bad_rate) == -1, "plan rejects sample_rate=44100 (rate whitelist)");
    }
    CHECK(plan_make(&p, &o) == 0 && p.total > 0,
          fmt_msg("plan accepts %d Hz / FFT %d", sr, fft_size));
    printf("       (%d Hz / FFT %d: aec=%zu nr=%zu synthesis=%zu total=%zu bytes, hop=%d n_freqs=%d)\n",
           sr, fft_size, p.aec_bytes, p.nr_bytes, p.synth_bytes, p.total, p.grid.hop_size,
           p.grid.n_freqs);
}

/* Once: out-of-enum level and NR level, the 48 kHz grid, and the delay mode and bank size
 * passing into the AEC pool size. */
static void test_config_validation_rejects(void) {
    Plan p;
    ConvMonoConfig o = grid_opts(16000, 0);
    ConvMonoConfig bad;

    bad = o; bad.aec_level = (AecLevel)99;
    CHECK(plan_make(&p, &bad) == -1, "plan rejects an out-of-enum aec_level");
    bad = o; bad.nr_level = (AnrLevel)99;
    CHECK(plan_make(&p, &bad) == -1, "plan rejects an out-of-enum nr_level");
    bad = o; bad.enable_nr = 0; bad.nr_level = (AnrLevel)99;
    CHECK(plan_make(&p, &bad) == 0, "an NR level is not read without a denoiser");
    bad = grid_opts(48000, 512);
    CHECK(plan_make(&p, &bad) == -1, "plan rejects FFT 512 at 48 kHz");
    bad = o; bad.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED; bad.delay_num_filters = 2;
    CHECK(plan_make(&p, &bad) == -1, "plan rejects EXTERNAL delay mode with an inapplicable matched-filter count");

    {
        ConvMonoConfig m5 = o, m2 = o, ex = o;
        Plan p5, p2, pe;
        m2.delay_num_filters = 2;
        ex.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        CHECK(plan_make(&p5, &m5) == 0 && plan_make(&p2, &m2) == 0 && plan_make(&pe, &ex) == 0 &&
              p5.aec_bytes > p2.aec_bytes && p2.aec_bytes > pe.aec_bytes,
              "delay mode/bank size reach the AEC pool size (5 filters > 2 filters > external)");
    }
}

static void test_pool_plan(int sr, int fft_size) {
    Plan p;
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Rig r;
    void* pool;

    if (plan_make(&p, &o) != 0) {
        fprintf(stderr, "FAIL: setup (plan) for the pool plan test @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    /* Extra 16 bytes of headroom so a +1-byte-offset "misaligned" pointer still has p.total
     * addressable bytes behind it. */
    pool = aligned_block(p.total + 16, 0xA5);
    if (!pool) {
        fprintf(stderr, "FAIL: pool alloc for the pool plan test @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    CHECK(rig_init_pool(&r, (uint8_t*)pool + 1, p.total, &p) != 0,
          fmt_msg("a misaligned (base+1) pool is refused @ %d Hz", sr));
    CHECK(rig_init_pool(&r, pool, p.total - 1, &p) != 0,
          fmt_msg("an undersized (bytes-1) pool is refused @ %d Hz", sr));
    CHECK(rig_init_pool(&r, pool, p.total, &p) == 0,
          fmt_msg("an aligned pool of exactly the planned size is accepted @ %d Hz", sr));
    rig_free(&r);
    free(pool);
}

/* Feed `hops` hops of a deterministic LCG mic/ref sequence into `r`, collecting each hop's
 * output into out_all[hops*hop]. The LCG is reseeded to the SAME constant at the top of every
 * call, so two calls (heap vs. pool instance) see byte-identical input. */
static void run_hops(Rig* r, int hop, int hops, float* out_all) {
    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    lcg_state = 0xC0FFEEu;
    for (int h = 0; h < hops; h++) {
        for (int i = 0; i < hop; i++) {
            ref[i] = lcg_sample();
            mic[i] = 0.3f * ref[i] + 0.05f * lcg_sample();   /* echo + a bit of near-end */
        }
        (void)rig_hop(r, mic, ref, out_all + (size_t)h * hop);
    }
    free(mic);
    free(ref);
}

static void test_create_vs_init_parity(int sr, int fft_size, int hop_count) {
    Plan p;
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Rig heap, pooled;
    void* pool;
    float *out_heap, *out_pool;
    int hop, finite = 1;

    if (plan_make(&p, &o) != 0) {
        fprintf(stderr, "FAIL: setup (plan) for create-vs-init parity @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    if (rig_init_heap(&heap, &p) != 0) {
        fprintf(stderr, "FAIL: heap construction @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    pool = aligned_block(p.total, 0xA5);   /* dirty pool: init must not rely on zeros */
    if (!pool || rig_init_pool(&pooled, pool, p.total, &p) != 0) {
        fprintf(stderr, "FAIL: init on a poisoned (0xA5) pool @ %d Hz\n", sr);
        g_failures++;
        rig_free(&heap);
        free(pool);
        return;
    }
    hop = p.grid.hop_size;
    out_heap = (float*)malloc((size_t)hop_count * (size_t)hop * sizeof(float));
    out_pool = (float*)malloc((size_t)hop_count * (size_t)hop * sizeof(float));
    run_hops(&heap, hop, hop_count, out_heap);
    run_hops(&pooled, hop, hop_count, out_pool);

    CHECK(memcmp(out_heap, out_pool, (size_t)hop_count * (size_t)hop * sizeof(float)) == 0,
          fmt_msg("heap construction == 0xA5-poisoned pool @ %d Hz, %d hops, byte-for-byte",
                  sr, hop_count));
    for (int i = 0; i < hop_count * hop; i++)
        if (out_heap[i] != out_heap[i]) { finite = 0; break; }   /* NaN check */
    CHECK(finite, fmt_msg("%d-hop synthetic run @ %d Hz produces no NaN in the output", hop_count, sr));

    free(out_heap);
    free(out_pool);
    rig_free(&heap);
    rig_free(&pooled);
    free(pool);
}

/* An echo path a warm run can converge on and the delay estimator can lock onto, kept
 * separate from run_hops(): that one deliberately re-seeds the LCG per call so two instances
 * see identical bytes, which is the wrong shape for a run that has to CONTINUE past a reset.
 * `hist` carries the far tail across hops so the delay is real rather than a per-hop wrap. */
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

/* The reset of the example programs (aec_reset + anr_reset + ac_istft_reset + the comfort-noise
 * seed) is equivalent to a
 * fresh instance on the same pool and config. Run with CNG both ON and OFF: the reset
 * re-seeds the comfort-noise RNG to its construction-time seed, so the CNG path is compared
 * for real. */
static void test_reset_equals_fresh_instance(int sr, int fft_size, int enable_cng, int enable_nr) {
    enum { HIST = 8192, WARM = 600, COMPARE = 300 };
    Plan p;
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Rig fresh, warmed;
    void *pool_a, *pool_b;
    float *mic, *ref, *out_a, *out_b, *hist, *hist_a, *hist_b;
    unsigned int rng, rng_a, rng_b;
    long differing = 0;
    int first_bad = -1, hop;

    o.enable_cng = enable_cng;
    o.enable_nr = enable_nr;
    if (plan_make(&p, &o) != 0) {
        fprintf(stderr, "FAIL: setup (plan) for reset parity @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    pool_a = aligned_block(p.total, 0);
    pool_b = aligned_block(p.total, 0);
    if (!pool_a || !pool_b || rig_init_pool(&fresh, pool_a, p.total, &p) != 0 ||
        rig_init_pool(&warmed, pool_b, p.total, &p) != 0) {
        fprintf(stderr, "FAIL: init for reset parity @ %d Hz\n", sr);
        g_failures++;
        free(pool_a); free(pool_b);
        return;
    }
    hop = p.grid.hop_size;
    mic  = (float*)malloc((size_t)hop * sizeof(float));
    ref  = (float*)malloc((size_t)hop * sizeof(float));
    out_a = (float*)malloc((size_t)hop * sizeof(float));
    out_b = (float*)malloc((size_t)hop * sizeof(float));
    hist  = (float*)calloc(HIST, sizeof(float));

    /* Warm on an echo path the compare phase does NOT reuse, so anything the subject
     * remembers is wrong for what follows. */
    rng = 1234567u;
    for (int h = 0; h < WARM; h++) {
        reset_scene_hop(&rng, mic, ref, hop, h, hist, HIST, 611, 0.65f);
        (void)rig_hop(&warmed, mic, ref, out_b);
    }
    rig_reset(&warmed);

    rng_a = 0x89abcdefu;
    rng_b = 0x89abcdefu;
    hist_a = (float*)calloc(HIST, sizeof(float));
    hist_b = (float*)calloc(HIST, sizeof(float));
    for (int h = 0; h < COMPARE; h++) {
        reset_scene_hop(&rng_a, mic, ref, hop, h, hist_a, HIST, 293, 0.5f);
        (void)rig_hop(&fresh, mic, ref, out_a);
        reset_scene_hop(&rng_b, mic, ref, hop, h, hist_b, HIST, 293, 0.5f);
        (void)rig_hop(&warmed, mic, ref, out_b);
        if (memcmp(out_a, out_b, (size_t)hop * sizeof(float)) != 0) {
            differing++;
            if (first_bad < 0) first_bad = h;
        }
    }
    CHECK(differing == 0,
          fmt_msg("reset == a fresh instance @ %d Hz, cng=%d, nr=%d: "
                  "%ld of %d post-reset hops differ (first at %d)",
                  sr, enable_cng, enable_nr, differing, COMPARE, first_bad));

    free(mic); free(ref); free(out_a); free(out_b);
    free(hist); free(hist_a); free(hist_b);
    rig_free(&fresh);
    rig_free(&warmed);
    free(pool_a); free(pool_b);
}

/* Teardown of a pool-resident instance is repeatable and NULL-safe, and the pool stays the
 * caller's and reusable. */
static void test_destroy_idempotence(int sr, int fft_size) {
    Plan p;
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Rig r;
    void* pool;

    if (plan_make(&p, &o) != 0 || !(pool = aligned_block(p.total, 0xA5))) {
        fprintf(stderr, "FAIL: setup for destroy-idempotence test @ %d Hz\n", sr);
        g_failures++;
        return;
    }
    if (rig_init_pool(&r, pool, p.total, &p) != 0) {
        fprintf(stderr, "FAIL: init for destroy-idempotence test @ %d Hz\n", sr);
        g_failures++;
        free(pool);
        return;
    }
    rig_destroy_only(&r);
    rig_destroy_only(&r);          /* second call on the same pool-resident instance */
    anr_destroy(NULL);
    aec_destroy(NULL);             /* NULL-safe */
    printf("PASS: denoiser and AEC teardown is idempotent (2x) and NULL-safe on a pool-resident "
           "instance @ %d Hz\n", sr);
    CHECK(rig_init_pool(&r, pool, p.total, &p) == 0,
          fmt_msg("pool is reusable via a fresh init after destroy @ %d Hz", sr));
    rig_free(&r);
    free(pool);
}

/* NR and RES are the post path's two gain sources. Two witnesses hold from outside:
 * (1) MMSE-LSA passes through (gain 1.0) for its first num_init_frames hops, so a chain
 * without NR must match one with NR bit for bit over that window and diverge afterwards;
 * (2) with NR, RES and CNG all off the post path applies gain 1.0 to the seam spectrum, so
 * resynthesising that spectrum here (read back through the AEC seam, one sqrt-Hann WOLA) must
 * reproduce the output to float precision, while the default instance must not. The
 * NR-disabled chain also drops the denoiser from the pool. */
static void test_nr_res_switches(void) {
    ConvMonoConfig on_o = grid_opts(16000, 0), nr_off_o, thru_o;
    Plan on_p, nr_off_p, thru_p;
    Rig on, nr_off, thru;
    void *pool_on, *pool_nr_off, *pool_thru;
    FftHandle* fft;
    void* fft_mem;
    int hop, frame, init_hops, total_hops, h, i;
    int prefix_equal = 1, prefix_nonzero = 0, diverged = 0;
    double max_abs = 0.0, err_thru = 0.0, err_on = 0.0;

    memset(&on, 0, sizeof(on)); memset(&nr_off, 0, sizeof(nr_off)); memset(&thru, 0, sizeof(thru));
    lcg_state = 12345u;
    on_o.enable_cng = 0;
    nr_off_o = on_o; nr_off_o.enable_nr = 0;
    thru_o = nr_off_o; thru_o.enable_res = 0;
    CHECK(plan_make(&on_p, &on_o) == 0 && plan_make(&nr_off_p, &nr_off_o) == 0 &&
          plan_make(&thru_p, &thru_o) == 0 &&
          on_p.nr_bytes > 0 && nr_off_p.nr_bytes == 0 &&
          on_p.total - nr_off_p.total == ALIGN16(on_p.nr_bytes),
          "NR-disabled sizing omits exactly the denoiser state");
    pool_on = aligned_block(on_p.total, 0xA5);
    pool_nr_off = aligned_block(nr_off_p.total, 0xA5);
    pool_thru = aligned_block(thru_p.total, 0xA5);
    CHECK(pool_on && pool_nr_off && pool_thru &&
          rig_init_pool(&on, pool_on, on_p.total, &on_p) == 0 &&
          rig_init_pool(&nr_off, pool_nr_off, nr_off_p.total, &nr_off_p) == 0 &&
          rig_init_pool(&thru, pool_thru, thru_p.total, &thru_p) == 0,
          "all three switch instances build");
    if (!on.with_post || !nr_off.with_post || !thru.with_post) return;
    CHECK(nr_off.nr == NULL && anr_set_level(nr_off.nr, ANR_LEVEL_MILD) == -1,
          "NR-disabled instance has no denoiser and refuses a level change");
    hop = on_p.grid.hop_size;
    frame = 2 * hop;                     /* 50% overlap: FFT == frame == 2 hops */
    init_hops = anr_retime_frames(20, 16000, hop);
    total_hops = init_hops + 120;
    /* fft_init requires a 16-byte aligned pool; a 32-bit malloc only guarantees 8. */
    if (posix_memalign(&fft_mem, 16, fft_get_mem_size(frame)) != 0) fft_mem = NULL;
    fft = fft_mem ? fft_init(fft_mem, fft_get_mem_size(frame), frame) : NULL;
    CHECK(fft != NULL, "test-side synthesis FFT builds");
    if (fft) {
        float* mic = (float*)malloc((size_t)hop * sizeof(float));
        float* ref = (float*)malloc((size_t)hop * sizeof(float));
        float* o_on = (float*)malloc((size_t)hop * sizeof(float));
        float* o_nr_off = (float*)malloc((size_t)hop * sizeof(float));
        float* o_thru = (float*)malloc((size_t)hop * sizeof(float));
        float* win = (float*)malloc((size_t)frame * sizeof(float));
        float* ola = (float*)calloc((size_t)frame, sizeof(float));
        float* ibuf = (float*)malloc((size_t)frame * sizeof(float));
        ac_window_build(win, frame);
        for (h = 0; h < total_hops; h++) {
            AecResContext ctx;
            for (i = 0; i < hop; i++) {
                float t = (float)(h * hop + i);
                float r = 0.6f * sinf(0.06f * t);
                ref[i] = r;
                /* lcg_sample() spans +-0.25: 0.08 x that is a +-0.02 broadband term, so
                 * the denoiser has something to shape after init. */
                mic[i] = 0.5f * r + 0.02f * sinf(0.017f * t) + 0.08f * lcg_sample();
            }
            (void)rig_hop(&on, mic, ref, o_on);
            (void)rig_hop(&nr_off, mic, ref, o_nr_off);
            (void)rig_hop(&thru, mic, ref, o_thru);
            /* Reference: the seam spectrum the thru instance just applied gain 1.0 to,
             * resynthesised with the same window and overlap. */
            aec_get_res_context(thru.aec, &ctx);
            {   /* the spectrum the instance rendered: the AEC's rendering base */
                AecOutputBase ob;
                memset(&ob, 0, sizeof(ob));
                ob.struct_size = sizeof(ob);
                if (aec_get_output_base(thru.aec, &ob) != 0) { g_failures++; continue; }
                fft_inverse(fft, ob.base_spec, ibuf);
            }
            for (i = 0; i < frame; i++) ola[i] += ibuf[i] * win[i];
            if (h >= 4)
                for (i = 0; i < hop; i++) {
                    double e_thru = fabs((double)o_thru[i] - ola[i]);
                    double e_on = fabs((double)o_on[i] - ola[i]);
                    if (fabs(ola[i]) > max_abs) max_abs = fabs(ola[i]);
                    if (e_thru > err_thru) err_thru = e_thru;
                    if (e_on > err_on) err_on = e_on;
                }
            memmove(ola, ola + hop, (size_t)(frame - hop) * sizeof(float));
            memset(ola + (frame - hop), 0, (size_t)hop * sizeof(float));
            if (h < init_hops) {
                if (memcmp(o_on, o_nr_off, (size_t)hop * sizeof(float)) != 0)
                    prefix_equal = 0;
                for (i = 0; i < hop; i++) if (o_nr_off[i] != 0.0f) prefix_nonzero = 1;
            } else if (memcmp(o_on, o_nr_off, (size_t)hop * sizeof(float)) != 0) {
                diverged = 1;
            }
        }
        free(mic); free(ref); free(o_on); free(o_nr_off); free(o_thru);
        free(win); free(ola); free(ibuf);
        fft_destroy(fft);
    }
    free(fft_mem);
    CHECK(prefix_equal,
          "NR-disabled output is bit-identical to NR-enabled while the denoiser "
          "passes through -- the RES-only branch applies G_res with nothing of its own");
    CHECK(prefix_nonzero, "that identity window carries a non-zero signal");
    CHECK(diverged, "the two paths diverge once the denoiser starts shaping");
    printf("  NR/RES switches: max|seam|=%.4f  err thru=%.2e  default=%.2e\n",
           max_abs, err_thru, err_on);
    CHECK(max_abs > 1e-3, "the seam reference carries signal");
    CHECK(err_thru <= 1e-4 * max_abs,
          "with NR, RES and CNG off the post path is a transparent synthesis of "
          "the seam spectrum");
    CHECK(err_on > 1e-2 * max_abs,
          "the default instance shapes the same stimulus, so the identity above "
          "constrains the bypass and not the stimulus");
    rig_free(&on); rig_free(&nr_off); rig_free(&thru);
    free(pool_on); free(pool_nr_off); free(pool_thru);
}

/* ---- comfort-noise contract ----------------------------------------------
 * The chain fills AEC-suppressed bins from lib/aec's comfort-noise recipe: an LCG step per bin
 * whose top five bits index kCngTable, the imaginary part a quarter turn (+8 of 32) ahead of
 * the real one. The injected noise must have bounded samples, the same expected power as the
 * unit-variance Gaussian pair and an advancing sequence; exact samples are not pinned (the
 * recipe is statistical). Part 1 checks the shared table; part 2 checks the chain draws from it
 * and advances its state, by differencing a CNG-on run against a byte-identical CNG-off one
 * (comfort noise is added after every gain). */
static void test_comfort_noise_contract(void) {
    const int sr = 16000, fft_size = 256, hops = 400;
    double sum = 0.0, sq = 0.0, worst_pair = 0.0;
    float bound = 0.0f;
    int i;

    for (i = 0; i < 32; i++) {
        float v = kCngTable[i];
        float q = kCngTable[(i + 8) & 31];
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
        ConvMonoConfig on_o = grid_opts(sr, fft_size), off_o = grid_opts(sr, fft_size);
        Plan on_p, off_p;
        Rig p_on, p_off;
        void *pool_on = NULL, *pool_off = NULL;
        float *mic, *ref, *o_on, *o_off, *prev_noise, *noise_stream;
        int hop, h, k, injected = 0, all_finite = 1;
        int prev_active = 0, corr_n = 0;
        double noise_sq = 0.0, corr_sum = 0.0;

        memset(&p_on, 0, sizeof(p_on)); memset(&p_off, 0, sizeof(p_off));
        off_o.enable_cng = 0;
        CHECK(plan_make(&on_p, &on_o) == 0 && plan_make(&off_p, &off_o) == 0 &&
              (pool_on = aligned_block(on_p.total, 0)) != NULL &&
              (pool_off = aligned_block(off_p.total, 0)) != NULL &&
              rig_init_pool(&p_on, pool_on, on_p.total, &on_p) == 0 &&
              rig_init_pool(&p_off, pool_off, off_p.total, &off_p) == 0,
              "cng: both instances build");
        if (!p_on.with_post || !p_off.with_post) return;

        hop = on_p.grid.hop_size;
        mic = (float*)calloc((size_t)hop, sizeof(float));
        ref = (float*)calloc((size_t)hop, sizeof(float));
        o_on = (float*)calloc((size_t)hop, sizeof(float));
        o_off = (float*)calloc((size_t)hop, sizeof(float));
        prev_noise = (float*)calloc((size_t)hop, sizeof(float));
        noise_stream = (float*)calloc((size_t)hops * hop, sizeof(float));

        lcg_state = 0x5EEDu;
        for (h = 0; h < hops; h++) {
            for (k = 0; k < hop; k++) { ref[k] = lcg_sample(); mic[k] = 0.5f * ref[k]; }
            (void)rig_hop(&p_on, mic, ref, o_on);
            (void)rig_hop(&p_off, mic, ref, o_off);
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
                /* A generator whose state never advanced would hand every bin of every hop
                 * the same table entry, so each hop's injected spectrum would be the
                 * amplitude profile times one fixed complex constant -- successive hops
                 * would then be nearly collinear. Advancing per bin decorrelates them. */
                if (hop_sq > 0.0 && prev_active && prev_sq > 0.0) {
                    corr_sum += fabs(cross / sqrt(hop_sq * prev_sq));
                    corr_n++;
                }
                prev_active = (hop_sq > 0.0);
                if (prev_active)
                    for (k = 0; k < hop; k++) prev_noise[k] = o_on[k] - o_off[k];
            }
        }
        CHECK(all_finite, "cng: every CNG-on sample and every injected sample is finite");
        CHECK(injected && noise_sq > 0.0,
              "cng: comfort noise is actually injected (CNG-on and CNG-off "
              "renders carry different energy)");
        CHECK(corr_n > 20 && corr_sum / corr_n < 0.5,
              "cng: successive hops' injected noise is decorrelated -- the "
              "generator advances instead of re-injecting one fixed spectrum");

        /* The seed is per-INSTANCE, not per-process: a second, independently built instance
         * fed the same input must inject the byte-identical noise stream. */
        {
            Rig q_on, q_off;
            void *pool_q_on = aligned_block(on_p.total, 0x33);
            void *pool_q_off = aligned_block(off_p.total, 0x33);
            int same = 1;
            if (pool_q_on && pool_q_off && rig_init_pool(&q_on, pool_q_on, on_p.total, &on_p) == 0 &&
                rig_init_pool(&q_off, pool_q_off, off_p.total, &off_p) == 0) {
                lcg_state = 0x5EEDu;
                for (h = 0; h < hops && same; h++) {
                    for (k = 0; k < hop; k++) { ref[k] = lcg_sample(); mic[k] = 0.5f * ref[k]; }
                    (void)rig_hop(&q_on, mic, ref, o_on);
                    (void)rig_hop(&q_off, mic, ref, o_off);
                    for (k = 0; k < hop; k++)
                        if (o_on[k] - o_off[k] != noise_stream[(size_t)h * hop + k]) same = 0;
                }
                CHECK(same,
                      "cng: a second, independently built instance injects the "
                      "identical noise stream (the seed is per-instance)");
                rig_free(&q_on);
                rig_free(&q_off);
            } else {
                CHECK(0, "cng: the second pair of instances builds");
            }
            free(pool_q_on);
            free(pool_q_off);
        }

        printf("cng: %d hops, injected RMS %.3e, hop-to-hop |corr| %.4f over "
               "%d pairs (bound %.8f, table mean %.2e)\n",
               hops, sqrt(noise_sq / ((double)hops * hop)),
               corr_n ? corr_sum / corr_n : -1.0, corr_n, bound, sum / 32.0);
        free(mic); free(ref); free(o_on); free(o_off); free(prev_noise);
        free(noise_stream);
        rig_free(&p_on);
        rig_free(&p_off);
        free(pool_on);
        free(pool_off);
    }
}

/* The comfort noise is scaled by the denoiser gain of its bin, bounded below by
 * CNG_NR_GAIN_FLOOR (the denoiser never sees the fill). Witness on an echo-plus-stationary-
 * noise stimulus where the denoiser settles far below the floor: the injected noise is the
 * exact difference between a CNG-on and a CNG-off render (same draw sequence). With the
 * denoiser off that difference is the unscaled reference d_ref; with it on, d_nr must be
 * CNG_NR_GAIN_FLOOR * d_ref sample for sample. The residual d_nr - floor * d_ref must carry
 * almost no energy, and the energy ratio must sit at floor^2 (no scaling would leave 1). */
static void test_comfort_noise_follows_nr_gain(void) {
    const int sr = 16000, fft_size = 256;
    const double floor_amp = 0.31622777, floor_sq = floor_amp * floor_amp;
    Plan plan[4];
    Rig p[4];
    void* pool[4] = { NULL, NULL, NULL, NULL };
    float* out[4] = { NULL, NULL, NULL, NULL };
    float *mic, *ref;
    int hop, settle_hops, hops, h, k, i, created = 1, finite = 1;
    double e_nr = 0.0, e_ref = 0.0, e_res = 0.0;

    /* [0] CNG off / NR on, [1] CNG on / NR on, [2] CNG off / NR off, [3] CNG on / NR off. */
    memset(p, 0, sizeof(p));
    for (i = 0; i < 4; i++) {
        ConvMonoConfig o = grid_opts(sr, fft_size);
        o.enable_cng = (i & 1);
        o.enable_nr = (i < 2);
        if (plan_make(&plan[i], &o) != 0 || !(pool[i] = aligned_block(plan[i].total, 0)) ||
            rig_init_pool(&p[i], pool[i], plan[i].total, &plan[i]) != 0)
            created = 0;
    }
    CHECK(created, "cng/nr: all four instances build");
    if (!created) {
        for (i = 0; i < 4; i++) { rig_free(&p[i]); free(pool[i]); }
        return;
    }
    hop = plan[0].grid.hop_size;
    /* The denoiser passes through for its init window and then needs a few hops to pull the
     * noise bins down to its gain floor. */
    settle_hops = anr_retime_frames(20, sr, hop) + 40;
    hops = settle_hops + 200;
    mic = (float*)calloc((size_t)hop, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    for (i = 0; i < 4; i++) out[i] = (float*)calloc((size_t)hop, sizeof(float));

    lcg_state = 0xC0FFEEu;
    for (h = 0; h < hops; h++) {
        for (k = 0; k < hop; k++) {
            ref[k] = lcg_sample();
            mic[k] = 0.5f * ref[k] + 0.3f * lcg_sample();
        }
        for (i = 0; i < 4; i++) (void)rig_hop(&p[i], mic, ref, out[i]);
        if (h < settle_hops) continue;
        for (k = 0; k < hop; k++) {
            double d_nr = out[1][k] - out[0][k];
            double d_ref = out[3][k] - out[2][k];
            double res = d_nr - floor_amp * d_ref;
            if (!isfinite(d_nr) || !isfinite(d_ref)) finite = 0;
            e_nr += d_nr * d_nr;
            e_ref += d_ref * d_ref;
            e_res += res * res;
        }
    }
    CHECK(finite && e_ref > 0.0,
          "cng/nr: the NR-off reference render injects comfort noise");
    CHECK(e_nr > 0.0 && e_nr > 0.8 * floor_sq * e_ref && e_nr < 1.5 * floor_sq * e_ref,
          "cng/nr: with the denoiser on, the injected noise carries floor^2 of "
          "the NR-off energy -- the fill is scaled by G_nr down to the floor");
    CHECK(e_res < 0.02 * e_ref,
          "cng/nr: the injected noise is floor * reference sample for sample "
          "-- the scaling is bounded by exactly CNG_NR_GAIN_FLOOR");
    for (i = 0; i < 4; i++) { rig_free(&p[i]); free(pool[i]); free(out[i]); }
    free(mic); free(ref);
}

/* ---- the chain against the hand-written oracle ---------------------------------------
 * The example composition (Aec + aec_get_res_context + the per-hop chain) must be conv_chain_mono.h
 * with its stages written out: the oracle's output is reproduced bit for bit, comfort noise ON
 * so the fill and its RNG are part of the claim. Every combination of the switches runs on
 * 16 kHz / FFT 256; the other grids run a sample of them. */

typedef struct SwitchCase { int nr, res, cng, legacy; } SwitchCase;

static int compare_to_oracle(int sr, int fft_size, SwitchCase s, AecLevel level, int hops,
                             int sabotage_legacy, int* nonzero_out) {
    ConvMonoConfig o = grid_opts(sr, fft_size);
    ConvMonoConfig oracle_o;
    Plan p;
    Rig r;
    void* pool;
    ConvMono* c;
    enum { HIST = 8192 };
    float *mic, *ref, *a, *b, *hist;
    unsigned int rng = 0x13572468u;
    int h, i, hop, ok = 1, identical = 1, nonzero = 0;

    o.enable_nr = s.nr; o.enable_res = s.res; o.enable_cng = s.cng; o.legacy_amin = s.legacy;
    o.aec_level = level;
    oracle_o = o;
    if (sabotage_legacy) oracle_o.legacy_amin = !o.legacy_amin;
    if (plan_make(&p, &o) != 0) return -1;
    pool = aligned_block(p.total, 0xA5);
    c = conv_mono_create(&oracle_o);
    if (!pool || !c || rig_init_pool(&r, pool, p.total, &p) != 0) {
        conv_mono_destroy(c);
        free(pool);
        return -1;
    }
    hop = p.grid.hop_size;
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    a = (float*)malloc((size_t)hop * sizeof(float));
    b = (float*)malloc((size_t)hop * sizeof(float));
    hist = (float*)calloc(HIST, sizeof(float));
    /* a delayed echo path with bursts of near-end speech: the delay locks, the RES cuts, the
     * double-talk guard and the comfort noise all run */
    for (h = 0; h < hops; ++h) {
        reset_scene_hop(&rng, mic, ref, hop, h, hist, HIST, 293, 0.5f);
        ok = ok && rig_hop(&r, mic, ref, a) == 0 && conv_mono_hop(c, mic, ref, b) == 0;
        if (!ok) break;
        if (memcmp(a, b, (size_t)hop * sizeof(float)) != 0) identical = 0;
        for (i = 0; i < hop; ++i) if (b[i] != 0.0f) nonzero = 1;
    }
    *nonzero_out = nonzero;
    free(mic); free(ref); free(a); free(b); free(hist);
    rig_free(&r);
    conv_mono_destroy(c);
    free(pool);
    return ok && identical ? 1 : 0;
}

static void test_chain_equals_oracle(void) {
    int combo, g, nonzero, rc, all_ok = 1, all_nonzero = 1, n = 0;
    for (combo = 0; combo < 16; ++combo) {
        SwitchCase s = { 1 - (combo & 1), 1 - ((combo >> 1) & 1), 1 - ((combo >> 2) & 1), (combo >> 3) & 1 };
        rc = compare_to_oracle(16000, 256, s, AEC_LEVEL_BALANCED, 800, 0, &nonzero);
        n++;
        if (rc != 1) { all_ok = 0; printf("  mismatch: nr=%d res=%d cng=%d legacy=%d\n", s.nr, s.res, s.cng, s.legacy); }
        if (!nonzero) all_nonzero = 0;
    }
    CHECK(all_ok && n == 16,
          "chain == oracle bit for bit, all 16 switch combinations @ 16000 Hz / FFT 256");
    CHECK(all_nonzero, "every compared render is non-silent");

    all_ok = 1;
    for (g = 0; g < N_GRIDS; ++g) {
        static const SwitchCase picks[] = { {1, 1, 1, 0}, {0, 1, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 1} };
        int k;
        int hops = hop_count_for_rate(GRIDS[g].sample_rate) / 2;
        for (k = 0; k < 4; ++k) {
            rc = compare_to_oracle(GRIDS[g].sample_rate, GRIDS[g].fft_size, picks[k],
                                   k == 3 ? AEC_LEVEL_AGGRESSIVE : AEC_LEVEL_BALANCED, hops, 0, &nonzero);
            if (rc != 1 || !nonzero) {
                all_ok = 0;
                printf("  mismatch @ %d Hz / %d, pick %d\n", GRIDS[g].sample_rate, GRIDS[g].fft_size, k);
            }
        }
    }
    CHECK(all_ok, "chain == oracle bit for bit on every grid (8 k, 16 k/256, 16 k/512, 48 k), four switch sets");

    /* The comparison can fail: an oracle with the NR prior flipped must differ. */
    {
        SwitchCase s = { 1, 1, 1, 0 };
        rc = compare_to_oracle(16000, 256, s, AEC_LEVEL_BALANCED, 800, 1, &nonzero);
        CHECK(rc == 0, "the oracle comparison detects a flipped residual-echo prior");
    }
}

/* ---- seam argument handling and the AEC-only plan ---- */
static void test_frame_and_chain_args(void) {
    Plan p;
    ConvMonoConfig o = grid_opts(16000, 256);
    Rig r;
    void* pool;
    AecResContext c;
    AecResContext untouched;
    static float mic[128], ref[128], out[128];

    if (plan_make(&p, &o) != 0 || !(pool = aligned_block(p.total, 0xA5)) ||
        rig_init_pool(&r, pool, p.total, &p) != 0) {
        CHECK(0, "args: setup");
        return;
    }
    aec_process_context(r.aec, mic, ref);
    aec_get_res_context(r.aec, &c);
    CHECK(c.error_spec != NULL && c.res_gain != NULL && c.n_freqs == 129,
          "args: the res context of the lane carries the error spectrum and the lane's own gain");
    memset(&untouched, 0x5A, sizeof untouched);
    aec_get_res_context(NULL, &untouched);
    aec_get_res_context(r.aec, NULL);
    CHECK(untouched.n_freqs == 0x5A5A5A5A, "args: aec_get_res_context ignores NULL arguments");
    rig_free(&r);
    free(pool);

    /* --aec-only: the AEC's own time-domain hop, no post stage and no synthesis */
    o.aec_only = 1;
    if (plan_make(&p, &o) != 0 || !(pool = aligned_block(p.total, 0xA5)) ||
        rig_init_pool(&r, pool, p.total, &p) != 0) {
        CHECK(0, "args: aec_only setup");
        return;
    }
    CHECK(!r.with_post && p.total == ALIGN16(p.aec_bytes) && rig_hop(&r, mic, ref, out) == 0,
          "args: an aec_only plan holds the AEC alone and copies the context's linear hop");
    rig_free(&r);
    free(pool);
}

/* --aec-only copies ctx.linear_hop out of aec_process_context(). The oracle is the library's
 * own time-domain entry, aec_process(), on a second instance of the same lane config: it must
 * return the same hop (the rig, the harness hop and the oracle advance identical state). The
 * count of hops where the formed hop differs from the linear hop shows the comparison can tell
 * the two apart. */
static int aec_only_equals_aec_process(int sr, int fft_size, int hops, int* formed_differs) {
    enum { HIST = 8192 };
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Plan p;
    Rig r;
    ConvMono* cm;
    Aec* oracle = (Aec*)calloc(1, sizeof(Aec));
    AecResContext c;
    float *mic, *ref, *want, *got_rig, *got_cm, *hist;
    unsigned int rng = 0x2468ACEu;
    int hop, h, ok = 1;
    o.aec_only = 1;
    *formed_differs = 0;
    if (!oracle || plan_make(&p, &o) != 0 || rig_init_heap(&r, &p) != 0 ||
        aec_create(oracle, &p.aec_cfg) != 0 || !(cm = conv_mono_create(&o))) {
        free(oracle);
        return -1;
    }
    hop = p.grid.hop_size;
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    want = (float*)malloc((size_t)hop * sizeof(float));
    got_rig = (float*)malloc((size_t)hop * sizeof(float));
    got_cm = (float*)malloc((size_t)hop * sizeof(float));
    hist = (float*)calloc(HIST, sizeof(float));
    for (h = 0; h < hops; ++h) {
        reset_scene_hop(&rng, mic, ref, hop, h, hist, HIST, 293, 0.5f);
        aec_process(oracle, mic, ref, want);
        if (rig_hop(&r, mic, ref, got_rig) != 0 || conv_mono_hop(cm, mic, ref, got_cm) != 0) {
            ok = 0;
            break;
        }
        aec_get_res_context(r.aec, &c);
        if (memcmp(c.formed_hop, c.linear_hop, (size_t)hop * sizeof(float)) != 0) ++*formed_differs;
        if (memcmp(want, got_rig, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(want, got_cm, (size_t)hop * sizeof(float)) != 0)
            ok = 0;
    }
    free(mic); free(ref); free(want); free(got_rig); free(got_cm); free(hist);
    conv_mono_destroy(cm);
    aec_destroy(oracle);
    free(oracle);
    rig_free(&r);
    return ok;
}

static void test_aec_only_output(void) {
    static const struct { int sr, fft; } g[] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    size_t i;
    for (i = 0; i < sizeof g / sizeof g[0]; ++i) {
        int formed = 0, rc = aec_only_equals_aec_process(g[i].sr, g[i].fft, 400, &formed);
        CHECK(rc == 1,
              fmt_msg("--aec-only hop (main.c rig, harness) == aec_process() oracle @ %d Hz / %d, 400 hops",
                      g[i].sr, g[i].fft));
        CHECK(formed > 0,
              fmt_msg("--aec-only @ %d Hz / %d: the formed hop differs from the linear hop on %d hops, so the comparison separates them",
                      g[i].sr, g[i].fft, formed));
    }
}


/* ---- the full-mode identity ---------------------------------------------------------------
 * With the denoiser off the chain is the AEC's own output: RES gain on the rendering base, comfort
 * noise, synthesis. aec_process() on a second Aec in full mode (enable_res = 1) is the independent
 * oracle: the rig of main.c, the hand-written chain and the oracle must agree bit for bit on streams
 * that fire the output guard, move the hard-reset serial (a delay change restarting the filter, an
 * aec_reset() the application issues), start with silence and never become usable. Deliberate
 * defects (a missing OLA clear, a missing reseed, rendering from the analysis seam, the old seed)
 * must each make the comparison fail. */


typedef struct FullRun {
    long hops, guard_hops, reset_hops;
    long rig_diff_hops, conv_diff_hops, first_rig_diff;
    int setup_ok;
} FullRun;

/* NR off, RES on, comfort noise as the scene says. `sabotage` is planted in the rig only. */
static void run_full_identity(const GuardScene* s, int sabotage, FullRun* out) {
    ConvMonoConfig o = grid_opts(s->sr, s->fft);
    Plan p;
    Rig r;
    ConvMono* cm;
    AecConfig oracle_cfg;
    Aec* oracle = (Aec*)calloc(1, sizeof(Aec));
    void* pool;
    float *want, *got_rig, *got_cm;
    int h, k, hop;
    memset(out, 0, sizeof(*out));
    out->first_rig_diff = -1;
    o.enable_nr = 0;
    o.enable_res = 1;
    o.enable_cng = s->enable_cng;
    if (!oracle || plan_make(&p, &o) != 0) { free(oracle); return; }
    oracle_cfg = p.aec_cfg;                    /* the lane, with the library's own full output */
    oracle_cfg.enable_res = 1;
    oracle_cfg.return_res_context = 0;
    oracle_cfg.enable_cng = p.enable_cng;
    pool = aligned_block(p.total, 0xA5);
    cm = conv_mono_create(&o);
    if (!pool || !cm || aec_create(oracle, &oracle_cfg) != 0 || rig_init_pool(&r, pool, p.total, &p) != 0) {
        conv_mono_destroy(cm);
        free(pool);
        free(oracle);
        return;
    }
    r.sabotage = sabotage;
    if (sabotage & SAB_OLD_SEED) r.rng = RETIRED_CNG_SEED;
    hop = p.grid.hop_size;
    want = (float*)malloc((size_t)hop * sizeof(float));
    got_rig = (float*)malloc((size_t)hop * sizeof(float));
    got_cm = (float*)malloc((size_t)hop * sizeof(float));
    out->setup_ok = want && got_rig && got_cm;
    for (h = 0; out->setup_ok && h < s->nh; ++h) {
        const float* mic = s->mic + (size_t)h * hop;
        const float* ref = s->far + (size_t)h * hop;
        for (k = 0; k < 4; ++k)
            if (s->reset_at[k] == h) {         /* the application restarts the AEC; every instance sees it */
                aec_reset(oracle);
                aec_reset(r.aec);
                aec_reset(cm->aec);
            }
        aec_process(oracle, mic, ref, want);
        if (rig_hop(&r, mic, ref, got_rig) != 0 || conv_mono_hop(cm, mic, ref, got_cm) != 0) {
            out->setup_ok = 0;
            break;
        }
        if (memcmp(want, got_rig, (size_t)hop * sizeof(float)) != 0) {
            ++out->rig_diff_hops;
            if (out->first_rig_diff < 0) out->first_rig_diff = h;
        }
        if (memcmp(want, got_cm, (size_t)hop * sizeof(float)) != 0) ++out->conv_diff_hops;
        ++out->hops;
    }
    out->guard_hops = r.guard_hops;
    out->reset_hops = r.reset_hops;
    free(want); free(got_rig); free(got_cm);
    rig_free(&r);
    conv_mono_destroy(cm);
    aec_destroy(oracle);
    free(oracle);
    free(pool);
}

static void test_full_mode_identity(void) {
    static const struct { int sr, fft, seconds; } g[] = { {16000, 256, 14}, {16000, 512, 14}, {48000, 1024, 6} };
    size_t gi;
    for (gi = 0; gi < sizeof g / sizeof g[0]; ++gi) {
        GuardScene sc[4];
        const char* nm[4] = { "moving echo path with double talk", "moving echo path, comfort noise off",
                              "silent head, no echo path (never usable)", "aec_reset events and a moving path" };
        long guard = 0, resets = 0, bad_rig = 0, bad_cm = 0, hops = 0;
        int i, built = 1;
        built &= guard_scene_make(&sc[0], nm[0], g[gi].sr, g[gi].fft, g[gi].seconds, 0x2545F491u, 1, 1, 1, 0) == 0;
        built &= guard_scene_make(&sc[1], nm[1], g[gi].sr, g[gi].fft, g[gi].seconds, 0x2545F491u, 1, 1, 1, 0) == 0;
        sc[1].enable_cng = 0;
        built &= guard_scene_make(&sc[2], nm[2], g[gi].sr, g[gi].fft, g[gi].seconds, 0x1234567u, 0, 0, 0, 30) == 0;
        if (built) {
            int n = sc[2].nh * sc[2].hop;
            for (i = 0; i < n; ++i) sc[2].mic[i] += 0.03f * gs_rnd();
        }
        built &= guard_scene_make(&sc[3], nm[3], g[gi].sr, g[gi].fft, g[gi].seconds, 0x2545F491u, 1, 1, 1, 0) == 0;
        if (built) { sc[3].reset_at[0] = sc[3].nh / 4; sc[3].reset_at[1] = sc[3].nh / 2; sc[3].reset_at[2] = 3 * sc[3].nh / 5; }
        CHECK(built, fmt_msg("full mode @ %d Hz / %d: the four guard scenes build", g[gi].sr, g[gi].fft));
        for (i = 0; built && i < 4; ++i) {
            FullRun fr;
            run_full_identity(&sc[i], 0, &fr);
            CHECK(fr.setup_ok && fr.hops == sc[i].nh,
                  fmt_msg("full mode @ %d Hz / %d, %s: all %d hops ran", g[gi].sr, g[gi].fft, nm[i], sc[i].nh));
            CHECK(fr.rig_diff_hops == 0 && fr.conv_diff_hops == 0,
                  fmt_msg("full mode @ %d Hz / %d, %s: main.c rig (%ld hops differ) and hand-written chain (%ld) equal aec_process(), "
                          "NR off, bit for bit", g[gi].sr, g[gi].fft, nm[i], fr.rig_diff_hops, fr.conv_diff_hops));
            guard += fr.guard_hops; resets += fr.reset_hops; hops += fr.hops;
            bad_rig += fr.rig_diff_hops; bad_cm += fr.conv_diff_hops;
        }
        printf("  full mode @ %d Hz / %d: %ld hops, guard fired on %ld, hard-reset serial moved on %ld\n",
               g[gi].sr, g[gi].fft, hops, guard, resets);
        CHECK(guard > 0, fmt_msg("full mode @ %d Hz / %d: the streams fire the output guard", g[gi].sr, g[gi].fft));
        CHECK(resets > 0, fmt_msg("full mode @ %d Hz / %d: the streams move the hard-reset serial", g[gi].sr, g[gi].fft));

        /* Negative controls on the stream that has both events and comfort noise on. */
        if (built) {
            static const struct { int sab; const char* what; } ctl[] = {
                { SAB_NO_OLA_CLEAR, "no OLA clear on the reset event" },
                { SAB_NO_RESEED, "no comfort-noise reseed on the reset event" },
                { SAB_ERROR_SPEC, "rendering from error_spec instead of the rendering base" },
                { SAB_OLD_SEED, "the old comfort-noise seed" },
            };
            size_t c;
            for (c = 0; c < sizeof ctl / sizeof ctl[0]; ++c) {
                FullRun fr;
                run_full_identity(&sc[3], ctl[c].sab, &fr);
                CHECK(fr.setup_ok && fr.rig_diff_hops > 0,
                      fmt_msg("full mode @ %d Hz / %d: a planted defect (%s) is detected (%ld hops differ)",
                              g[gi].sr, g[gi].fft, ctl[c].what, fr.rig_diff_hops));
            }
        }
        (void)bad_rig; (void)bad_cm;
        for (i = 0; i < 4; ++i) guard_scene_free(&sc[i]);
    }
}


/* The denoiser keeps running through the AEC's hard post-chain reset: the library does not touch NR state
 * on that event and the example does not either (following it there is application policy, and this
 * program's policy is to keep the estimator and its noise floor). Observable: the noise-floor
 * initialisation flag stays set across every reset hop that happens after the denoiser has
 * initialised. Returns the number of such hops; `cleared` counts those after which the flag dropped. */
static long run_nr_policy(const GuardScene* s, int sabotage, long* cleared) {
    ConvMonoConfig o = grid_opts(s->sr, s->fft);
    Plan p;
    Rig r;
    void* pool;
    float* out;
    long events = 0;
    int h;
    *cleared = 0;
    o.enable_cng = 1;
    if (plan_make(&p, &o) != 0 || !(pool = aligned_block(p.total, 0xA5)) ||
        rig_init_pool(&r, pool, p.total, &p) != 0)
        return -1;
    r.sabotage = sabotage;
    out = (float*)malloc((size_t)p.grid.hop_size * sizeof(float));
    for (h = 0; out && h < s->nh; ++h) {
        AnrDebugStatus before, after;
        long resets = r.reset_hops;
        anr_debug_status(r.nr, &before);
        (void)rig_hop(&r, s->mic + (size_t)h * p.grid.hop_size, s->far + (size_t)h * p.grid.hop_size, out);
        if (r.reset_hops == resets || !before.initialized) continue;
        anr_debug_status(r.nr, &after);
        ++events;
        if (!after.initialized) ++*cleared;
    }
    free(out);
    rig_free(&r);
    free(pool);
    return events;
}

static void test_nr_keeps_running(void) {
    GuardScene sc;
    long cleared = 0, events;
    if (guard_scene_make(&sc, "nr policy", 16000, 256, 14, 0x2545F491u, 1, 1, 1, 0) != 0) {
        CHECK(0, "nr policy: the scene builds");
        return;
    }
    events = run_nr_policy(&sc, 0, &cleared);
    CHECK(events >= 1 && cleared == 0,
          fmt_msg("nr policy: the denoiser keeps its state across hard post-chain resets "
                  "(%ld reset hops after its initialisation, flag dropped on %ld)", events, cleared));
    events = run_nr_policy(&sc, SAB_RESET_NR, &cleared);
    CHECK(events >= 1 && cleared >= 1,
          fmt_msg("nr policy: a planted NR reset on the event is detected (flag dropped on %ld of %ld)", cleared, events));
    guard_scene_free(&sc);
}

/* ---- the example programs as black boxes ----------------------------------------------
 * aec_nr_pipeline and aec_nr_pipeline_static read a WAV pair and, with DUMP_CTX, write every
 * hop's output samples. The oracle reads the same WAV pair (so it sees the same quantised
 * input) and must reproduce those hops bit for bit, per option set. */

typedef struct BlackBoxCase {
    const char* args;     /* the program's options after the three paths */
    int sr, fft_size;     /* fft_size 0: the rate's default grid */
    AecLevel level;
    AnrLevel nr_level;
    int nr, res, cng, legacy;
} BlackBoxCase;

static int write_scene_wav(const char* mic_path, const char* ref_path, int sr, int hops, int hop) {
    enum { HIST = 8192 };
    WavWriter* wm = wav_open_write(mic_path, sr, 1);
    WavWriter* wr = wav_open_write(ref_path, sr, 1);
    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    float* hist = (float*)calloc(HIST, sizeof(float));
    unsigned int rng = 0x2468ACEu;
    int h, ok = wm && wr && mic && ref && hist;
    for (h = 0; ok && h < hops; ++h) {
        reset_scene_hop(&rng, mic, ref, hop, h, hist, HIST, 293, 0.5f);
        wav_write_float(wm, mic, hop);
        wav_write_float(wr, ref, hop);
    }
    if (wm) wav_close_write(wm);
    if (wr) wav_close_write(wr);
    free(mic); free(ref); free(hist);
    return ok ? 0 : -1;
}

/* 1 when every dumped hop equals the oracle, 0 on a mismatch, -1 on a setup failure. */
static int black_box_case(const char* bin, const char* tmp, const BlackBoxCase* bc, int* hops_out) {
    char mic_path[512], ref_path[512], out_path[512], dump_path[512], cmd[2048];
    ConvMonoConfig o = grid_opts(bc->sr, bc->fft_size);
    ConvMono* c;
    WavReader *rm, *rr;
    FILE* fd;
    int hdr[2], hop, nf, ok = 1, n = 0, nonzero = 0;
    float *mic, *ref, *want, *rec;
    size_t rec_floats;
    snprintf(mic_path, sizeof mic_path, "%s/bb_mic_%d.wav", tmp, bc->sr);
    snprintf(ref_path, sizeof ref_path, "%s/bb_ref_%d.wav", tmp, bc->sr);
    snprintf(out_path, sizeof out_path, "%s/bb_out.wav", tmp);
    snprintf(dump_path, sizeof dump_path, "%s/bb_ctx.bin", tmp);
    o.aec_level = bc->level;
    o.nr_level = bc->nr_level;
    o.enable_nr = bc->nr; o.enable_res = bc->res; o.enable_cng = bc->cng; o.legacy_amin = bc->legacy;
    c = conv_mono_create(&o);
    if (!c) return -1;
    if (write_scene_wav(mic_path, ref_path, bc->sr, 400, c->hop) != 0) { conv_mono_destroy(c); return -1; }
    remove(dump_path);
    snprintf(cmd, sizeof cmd, "DUMP_CTX='%s' '%s' '%s' '%s' '%s' %s > '%s/bb_stdout.txt' 2>&1",
             dump_path, bin, mic_path, ref_path, out_path, bc->args, tmp);
    if (system(cmd) != 0) { conv_mono_destroy(c); return -1; }
    fd = fopen(dump_path, "rb");
    rm = wav_open_read(mic_path);
    rr = wav_open_read(ref_path);
    if (!fd || !rm || !rr || fread(hdr, sizeof(int), 2, fd) != 2) {
        if (fd) fclose(fd);
        if (rm) wav_close_read(rm);
        if (rr) wav_close_read(rr);
        conv_mono_destroy(c);
        return -1;
    }
    nf = hdr[0];
    hop = hdr[1];
    /* base(2 nf) res_gain(nf) r2(nf) comfort_noise(nf) far_power(1) g_nr(nf) out(hop) */
    rec_floats = (size_t)(6 * nf + 1 + hop);
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    want = (float*)malloc((size_t)hop * sizeof(float));
    rec = (float*)malloc(rec_floats * sizeof(float));
    if (nf != c->n_freqs || hop != c->hop) ok = 0;
    while (ok && fread(rec, sizeof(float), rec_floats, fd) == rec_floats) {
        int i;
        if (wav_read_float(rm, mic, hop) != hop || wav_read_float(rr, ref, hop) != hop ||
            conv_mono_hop(c, mic, ref, want) != 0) { ok = 0; break; }
        if (memcmp(want, rec + 6 * nf + 1, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (i = 0; i < hop; ++i) if (want[i] != 0.0f) nonzero = 1;
        n++;
    }
    fclose(fd);
    wav_close_read(rm);
    wav_close_read(rr);
    free(mic); free(ref); free(want); free(rec);
    conv_mono_destroy(c);
    *hops_out = n;
    return ok && nonzero ? 1 : 0;
}

/* --aec-only writes no context dump, so the output WAV is the observable: the oracle runs
 * aec_process() on a second instance over the same quantised WAV pair, writes its hops through
 * the same WAV writer and the two files' samples must be equal. 1 on equal, 0 on a mismatch,
 * -1 on a setup failure. */
static int black_box_aec_only(const char* bin, const char* tmp, int sr, int fft_size,
                              const char* args) {
    enum { HOPS = 400 };
    char mic_path[512], ref_path[512], out_path[512], ora_path[512], cmd[2048];
    ConvMonoConfig o = grid_opts(sr, fft_size);
    Plan p;
    Aec* oracle = (Aec*)calloc(1, sizeof(Aec));
    WavReader *rm, *rr, *ro, *rp;
    WavWriter* wo;
    float *mic, *ref, *want, *got, *back;
    int hop, h, ok = 1, nonzero = 0;
    o.aec_only = 1;
    snprintf(mic_path, sizeof mic_path, "%s/bba_mic.wav", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/bba_ref.wav", tmp);
    snprintf(out_path, sizeof out_path, "%s/bba_out.wav", tmp);
    snprintf(ora_path, sizeof ora_path, "%s/bba_oracle.wav", tmp);
    if (!oracle || plan_make(&p, &o) != 0 || aec_create(oracle, &p.aec_cfg) != 0) {
        free(oracle);
        return -1;
    }
    hop = p.grid.hop_size;
    if (write_scene_wav(mic_path, ref_path, sr, HOPS, hop) != 0) { aec_destroy(oracle); free(oracle); return -1; }
    snprintf(cmd, sizeof cmd, "'%s' '%s' '%s' '%s' %s > '%s/bba_stdout.txt' 2>&1",
             bin, mic_path, ref_path, out_path, args, tmp);
    if (system(cmd) != 0) { aec_destroy(oracle); free(oracle); return -1; }
    rm = wav_open_read(mic_path);
    rr = wav_open_read(ref_path);
    wo = wav_open_write(ora_path, sr, 1);
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    want = (float*)malloc((size_t)hop * sizeof(float));
    got = (float*)malloc((size_t)hop * sizeof(float));
    back = (float*)malloc((size_t)hop * sizeof(float));
    if (!rm || !rr || !wo || !mic || !ref || !want || !got || !back) ok = -1;
    for (h = 0; ok == 1 && h < HOPS; ++h) {
        if (wav_read_float(rm, mic, hop) != hop || wav_read_float(rr, ref, hop) != hop) { ok = -1; break; }
        aec_process(oracle, mic, ref, want);
        wav_write_float(wo, want, hop);
    }
    if (wo) wav_close_write(wo);
    ro = ok == 1 ? wav_open_read(ora_path) : NULL;
    rp = ok == 1 ? wav_open_read(out_path) : NULL;
    if (ok == 1 && (!ro || !rp)) ok = -1;
    for (h = 0; ok == 1 && h < HOPS; ++h) {
        int i;
        if (wav_read_float(ro, back, hop) != hop || wav_read_float(rp, got, hop) != hop) { ok = 0; break; }
        if (memcmp(back, got, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (i = 0; i < hop; ++i) if (got[i] != 0.0f) nonzero = 1;
    }
    if (ok == 1 && wav_read_float(rp, got, hop) != 0) ok = 0;   /* no extra hop: no synthesis lag */
    if (rm) wav_close_read(rm);
    if (rr) wav_close_read(rr);
    if (ro) wav_close_read(ro);
    if (rp) wav_close_read(rp);
    free(mic); free(ref); free(want); free(got); free(back);
    aec_destroy(oracle);
    free(oracle);
    return ok == 1 ? (nonzero ? 1 : 0) : ok;
}

static void test_examples_aec_only(const char* bin_dir, const char* tmp) {
    static const struct { const char* args; int sr, fft; } cases[] = {
        {"--fft-size 256 --aec-only", 16000, 256},
        {"--fft-size 512 aggressive --aec-only", 16000, 512},
        {"--aec-level 3 --aec-only", 8000, 0},
    };
    static const char* const programs[] = {"aec_nr_pipeline", "aec_nr_pipeline_static"};
    size_t pi, ci;
    char bin[512];
    for (pi = 0; pi < 2; ++pi) {
        snprintf(bin, sizeof bin, "%s/%s", bin_dir, programs[pi]);
        for (ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci)
            CHECK(black_box_aec_only(bin, tmp, cases[ci].sr, cases[ci].fft, cases[ci].args) == 1,
                  fmt_msg("black box: %s %s @ %d Hz equals the aec_process() oracle sample for sample",
                          programs[pi], cases[ci].args, cases[ci].sr));
    }
}


/* --no-nr: the example programs' chain is the AEC's own full output. The scene fires the output guard
 * and moves the hard-reset serial; the oracle is aec_process() in full mode on a second instance over
 * the same quantised WAV pair, and the DUMP_CTX output hops (float, no WAV quantisation) must equal it
 * bit for bit. The guard and reset counts come from the oracle's own public getters. 1 on equal with
 * both events seen, 0 on a mismatch or when the scene did not exercise them, -1 on a setup failure. */
static int black_box_full_mode(const char* bin, const char* tmp, const GuardScene* sc, const char* args,
                               int cng, long* guard_hops, long* reset_hops) {
    char mic_path[512], ref_path[512], out_path[512], dump_path[512], cmd[2048];
    ConvMonoConfig o = grid_opts(sc->sr, sc->fft);
    AecConfig cfg;
    Plan p;
    Aec* oracle = (Aec*)calloc(1, sizeof(Aec));
    WavWriter *wm, *wr;
    WavReader *rm, *rr;
    FILE* fd;
    int hdr[2], hop, nf, h, ok = 1, serial_seen;
    float *mic, *ref, *want, *rec;
    size_t rec_floats;
    *guard_hops = *reset_hops = 0;
    snprintf(mic_path, sizeof mic_path, "%s/bbf_mic.wav", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/bbf_ref.wav", tmp);
    snprintf(out_path, sizeof out_path, "%s/bbf_out.wav", tmp);
    snprintf(dump_path, sizeof dump_path, "%s/bbf_ctx.bin", tmp);
    o.enable_nr = 0; o.enable_res = 1; o.enable_cng = cng;
    if (!oracle || plan_make(&p, &o) != 0) { free(oracle); return -1; }
    cfg = p.aec_cfg;
    cfg.enable_res = 1;
    cfg.return_res_context = 0;
    cfg.enable_cng = p.enable_cng;
    if (aec_create(oracle, &cfg) != 0) { free(oracle); return -1; }
    hop = p.grid.hop_size;
    wm = wav_open_write(mic_path, sc->sr, 1);
    wr = wav_open_write(ref_path, sc->sr, 1);
    if (!wm || !wr) { ok = -1; }
    for (h = 0; ok == 1 && h < sc->nh; ++h) {
        wav_write_float(wm, sc->mic + (size_t)h * hop, hop);
        wav_write_float(wr, sc->far + (size_t)h * hop, hop);
    }
    if (wm) wav_close_write(wm);
    if (wr) wav_close_write(wr);
    if (ok != 1) { aec_destroy(oracle); free(oracle); return -1; }
    remove(dump_path);
    snprintf(cmd, sizeof cmd, "DUMP_CTX='%s' '%s' '%s' '%s' '%s' %s > '%s/bbf_stdout.txt' 2>&1",
             dump_path, bin, mic_path, ref_path, out_path, args, tmp);
    if (system(cmd) != 0) { aec_destroy(oracle); free(oracle); return -1; }
    fd = fopen(dump_path, "rb");
    rm = wav_open_read(mic_path);
    rr = wav_open_read(ref_path);
    if (!fd || !rm || !rr || fread(hdr, sizeof(int), 2, fd) != 2 || hdr[1] != hop) {
        if (fd) fclose(fd);
        if (rm) wav_close_read(rm);
        if (rr) wav_close_read(rr);
        aec_destroy(oracle); free(oracle);
        return -1;
    }
    nf = hdr[0];
    rec_floats = (size_t)(6 * nf + 1 + hop);
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    want = (float*)malloc((size_t)hop * sizeof(float));
    rec = (float*)malloc(rec_floats * sizeof(float));
    serial_seen = (int)aec_post_reset_serial(oracle);
    for (h = 0; ok == 1 && h < sc->nh; ++h) {
        AecOutputBase ob;
        if (fread(rec, sizeof(float), rec_floats, fd) != rec_floats) { ok = 0; break; }
        if (wav_read_float(rm, mic, hop) != hop || wav_read_float(rr, ref, hop) != hop) { ok = -1; break; }
        aec_process(oracle, mic, ref, want);
        memset(&ob, 0, sizeof ob);
        ob.struct_size = sizeof ob;
        if (aec_get_output_base(oracle, &ob) == 0 && ob.guard_active) ++*guard_hops;
        if ((int)aec_post_reset_serial(oracle) != serial_seen) {
            serial_seen = (int)aec_post_reset_serial(oracle);
            ++*reset_hops;
        }
        if (memcmp(want, rec + 6 * nf + 1, (size_t)hop * sizeof(float)) != 0) ok = 0;
    }
    if (ok == 1 && fread(rec, sizeof(float), 1, fd) != 0) ok = 0;   /* one dumped hop per input hop */
    fclose(fd);
    wav_close_read(rm);
    wav_close_read(rr);
    free(mic); free(ref); free(want); free(rec);
    aec_destroy(oracle);
    free(oracle);
    if (ok != 1) return ok;
    return (*guard_hops > 0 && *reset_hops > 0) ? 1 : 0;
}

static void test_examples_full_mode(const char* bin_dir, const char* tmp) {
    static const struct { int sr, fft, seconds, cng; const char* args; } cases[] = {
        {16000, 256, 14, 1, "--fft-size 256 --no-nr"},
        {16000, 512, 14, 0, "--fft-size 512 --no-nr --no-cng"},
        {48000, 1024, 6, 1, "--no-nr"},
    };
    static const char* const programs[] = {"aec_nr_pipeline", "aec_nr_pipeline_static"};
    size_t pi, ci;
    char bin[512];
    for (ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        GuardScene sc;
        if (guard_scene_make(&sc, "black box", cases[ci].sr, cases[ci].fft, cases[ci].seconds, 0x2545F491u, 1, 1, 1, 0) != 0) {
            CHECK(0, "black box full mode: the scene builds");
            continue;
        }
        for (pi = 0; pi < 2; ++pi) {
            long gh = 0, rh = 0;
            int rc;
            snprintf(bin, sizeof bin, "%s/%s", bin_dir, programs[pi]);
            rc = black_box_full_mode(bin, tmp, &sc, cases[ci].args, cases[ci].cng, &gh, &rh);
            CHECK(rc == 1,
                  fmt_msg("black box: %s %s @ %d Hz equals aec_process() in full mode bit for bit "
                          "(guard fired on %ld hops, hard resets %ld)",
                          programs[pi], cases[ci].args, cases[ci].sr, gh, rh));
        }
        guard_scene_free(&sc);
    }
}


/* The full chain (denoiser on) as a black box on a scene that fires the guard and moves the reset serial: the
 * DUMP_CTX output hops equal the hand-written chain bit for bit. The hand-written chain follows the event for
 * the synthesis and the comfort noise only; a program that also reset its denoiser on the event (or that fed it
 * the analysis seam on a guard hop) would differ. 1 on equal with both events seen, 0 otherwise, -1 on setup. */
static int black_box_scene_vs_chain(const char* bin, const char* tmp, const GuardScene* sc, const char* args,
                                    long* guard_hops, long* reset_hops) {
    char mic_path[512], ref_path[512], out_path[512], dump_path[512], cmd[2048];
    ConvMonoConfig o = grid_opts(sc->sr, sc->fft);
    ConvMono* c = conv_mono_create(&o);
    WavWriter *wm, *wr;
    WavReader *rm, *rr;
    FILE* fd;
    int hdr[2], hop, nf, h, ok = 1;
    float *mic, *ref, *want, *rec;
    size_t rec_floats;
    *guard_hops = *reset_hops = 0;
    snprintf(mic_path, sizeof mic_path, "%s/bbc_mic.wav", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/bbc_ref.wav", tmp);
    snprintf(out_path, sizeof out_path, "%s/bbc_out.wav", tmp);
    snprintf(dump_path, sizeof dump_path, "%s/bbc_ctx.bin", tmp);
    if (!c) return -1;
    hop = c->hop;
    wm = wav_open_write(mic_path, sc->sr, 1);
    wr = wav_open_write(ref_path, sc->sr, 1);
    if (!wm || !wr) ok = -1;
    for (h = 0; ok == 1 && h < sc->nh; ++h) {
        wav_write_float(wm, sc->mic + (size_t)h * hop, hop);
        wav_write_float(wr, sc->far + (size_t)h * hop, hop);
    }
    if (wm) wav_close_write(wm);
    if (wr) wav_close_write(wr);
    if (ok != 1) { conv_mono_destroy(c); return -1; }
    remove(dump_path);
    snprintf(cmd, sizeof cmd, "DUMP_CTX='%s' '%s' '%s' '%s' '%s' %s > '%s/bbc_stdout.txt' 2>&1",
             dump_path, bin, mic_path, ref_path, out_path, args, tmp);
    if (system(cmd) != 0) { conv_mono_destroy(c); return -1; }
    fd = fopen(dump_path, "rb");
    rm = wav_open_read(mic_path);
    rr = wav_open_read(ref_path);
    if (!fd || !rm || !rr || fread(hdr, sizeof(int), 2, fd) != 2 || hdr[1] != hop) {
        if (fd) fclose(fd);
        if (rm) wav_close_read(rm);
        if (rr) wav_close_read(rr);
        conv_mono_destroy(c);
        return -1;
    }
    nf = hdr[0];
    rec_floats = (size_t)(6 * nf + 1 + hop);
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    want = (float*)malloc((size_t)hop * sizeof(float));
    rec = (float*)malloc(rec_floats * sizeof(float));
    for (h = 0; ok == 1 && h < sc->nh; ++h) {
        if (fread(rec, sizeof(float), rec_floats, fd) != rec_floats) { ok = 0; break; }
        if (wav_read_float(rm, mic, hop) != hop || wav_read_float(rr, ref, hop) != hop ||
            conv_mono_hop(c, mic, ref, want) != 0) { ok = -1; break; }
        if (memcmp(want, rec + 6 * nf + 1, (size_t)hop * sizeof(float)) != 0) ok = 0;
    }
    *guard_hops = c->guard_hops;
    *reset_hops = c->reset_hops;
    fclose(fd);
    wav_close_read(rm);
    wav_close_read(rr);
    free(mic); free(ref); free(want); free(rec);
    conv_mono_destroy(c);
    if (ok != 1) return ok;
    return (*guard_hops > 0 && *reset_hops > 0) ? 1 : 0;
}

static void test_examples_full_chain(const char* bin_dir, const char* tmp) {
    static const struct { int sr, fft, seconds; const char* args; } cases[] = {
        {16000, 256, 14, "--fft-size 256"},
        {48000, 1024, 6, ""},
    };
    static const char* const programs[] = {"aec_nr_pipeline", "aec_nr_pipeline_static"};
    size_t pi, ci;
    char bin[512];
    for (ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        GuardScene sc;
        if (guard_scene_make(&sc, "black box chain", cases[ci].sr, cases[ci].fft, cases[ci].seconds, 0x2545F491u, 1, 1, 1, 0) != 0) {
            CHECK(0, "black box full chain: the scene builds");
            continue;
        }
        for (pi = 0; pi < 2; ++pi) {
            long gh = 0, rh = 0;
            snprintf(bin, sizeof bin, "%s/%s", bin_dir, programs[pi]);
            CHECK(black_box_scene_vs_chain(bin, tmp, &sc, cases[ci].args, &gh, &rh) == 1,
                  fmt_msg("black box: %s %s @ %d Hz, denoiser on, equals the hand-written chain bit for bit across "
                          "the guard (%ld hops) and the reset event (%ld)", programs[pi], cases[ci].args, cases[ci].sr, gh, rh));
        }
        guard_scene_free(&sc);
    }
}

static void test_examples_black_box(const char* bin_dir, const char* tmp) {
    static const BlackBoxCase cases[] = {
        {"--fft-size 256", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 1, 1, 1, 0},
        {"--fft-size 256 --no-res", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 1, 0, 1, 0},
        {"--fft-size 256 --no-nr", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 0, 1, 1, 0},
        {"--fft-size 256 --no-cng", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 1, 1, 0, 0},
        {"--fft-size 256 --legacy-amin", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 1, 1, 1, 1},
        {"--fft-size 256 --no-nr --no-res", 16000, 256, AEC_LEVEL_BALANCED, ANR_LEVEL_BALANCED, 0, 0, 1, 0},
        /* The NR level is --nr-level; --nr-preset is the same option under its earlier name. */
        {"--fft-size 256 aggressive --nr-level aggressive", 16000, 256, AEC_LEVEL_AGGRESSIVE,
         ANR_LEVEL_AGGRESSIVE, 1, 1, 1, 0},
        {"--fft-size 512 mild --nr-level mild", 16000, 512, AEC_LEVEL_MILD, ANR_LEVEL_MILD, 1, 1, 1, 0},
        {"--fft-size 256 2 --nr-level 3", 16000, 256, AEC_LEVEL_AGGRESSIVE, ANR_LEVEL_AGGRESSIVE, 1, 1, 1, 0},
        {"--fft-size 256 --nr-preset 0 3", 16000, 256, AEC_LEVEL_NO_FLOOR, ANR_LEVEL_MILD, 1, 1, 1, 0},
        {"--nr-level moderate", 8000, 0, AEC_LEVEL_BALANCED, ANR_LEVEL_MODERATE, 1, 1, 1, 0},
        {"--nr-preset aggressive --nr-level mild", 8000, 0, AEC_LEVEL_BALANCED, ANR_LEVEL_MILD, 1, 1, 1, 0},
        /* The AEC level is the positional argument or --aec-level. */
        {"--fft-size 256 --aec-level aggressive --nr-level aggressive", 16000, 256, AEC_LEVEL_AGGRESSIVE,
         ANR_LEVEL_AGGRESSIVE, 1, 1, 1, 0},
        {"--aec-level 3 --fft-size 256", 16000, 256, AEC_LEVEL_NO_FLOOR, ANR_LEVEL_BALANCED, 1, 1, 1, 0},
        {"--aec-level mild --fft-size 512 --nr-level mild", 16000, 512, AEC_LEVEL_MILD, ANR_LEVEL_MILD, 1, 1, 1, 0},
        {"mild --fft-size 256", 16000, 256, AEC_LEVEL_MILD, ANR_LEVEL_BALANCED, 1, 1, 1, 0},
        {"--aec-level 2", 8000, 0, AEC_LEVEL_AGGRESSIVE, ANR_LEVEL_BALANCED, 1, 1, 1, 0},
        {"--fft-size 256 aggressive --nr-preset aggressive", 16000, 256, AEC_LEVEL_AGGRESSIVE,
         ANR_LEVEL_AGGRESSIVE, 1, 1, 1, 0},
        {"--fft-size 512 mild --nr-preset mild", 16000, 512, AEC_LEVEL_MILD, ANR_LEVEL_MILD, 1, 1, 1, 0},
        {"--nr-preset moderate", 8000, 0, AEC_LEVEL_BALANCED, ANR_LEVEL_MODERATE, 1, 1, 1, 0},
    };
    static const char* const programs[] = {"aec_nr_pipeline", "aec_nr_pipeline_static"};
    size_t pi, ci;
    char bin[512];
    for (pi = 0; pi < 2; ++pi) {
        snprintf(bin, sizeof bin, "%s/%s", bin_dir, programs[pi]);
        for (ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
            BlackBoxCase bc = cases[ci];
            int hops = 0, rc;
            rc = black_box_case(bin, tmp, &bc, &hops);
            CHECK(rc == 1 && hops >= 390,
                  fmt_msg("black box: %s %s @ %d Hz equals the oracle bit for bit (%d hops)",
                          programs[pi], bc.args, bc.sr, hops));
        }
    }
}

/* An option's value is the next token unless that starts with "--": then the value is missing
 * and the program exits 2 before it opens a file (the three paths below do not exist). A value
 * such as "-x" is a value; the run then stops at the missing input, exit 1. */
static int example_exit_is(const char* bin, const char* args, int want) {
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "'%s' %s >/dev/null 2>&1; test $? -eq %d", bin, args, want);
    return system(cmd) == 0;
}

/* 1 when the program's output (stdout and stderr) for `args` contains `text`. */
static int example_output_has(const char* bin, const char* args, const char* text) {
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "'%s' %s 2>&1 | grep -q -F -e '%s'", bin, args, text);
    return system(cmd) == 0;
}

static void test_examples_cli_values(const char* bin_dir) {
    static const char* const programs[] = {"aec_nr_pipeline", "aec_nr_pipeline_static"};
    static const char* const missing[] = {
        "--nr-preset", "--fft-size", "--nr-preset --no-res", "--fft-size --no-cng",
        "--nr-preset --", "--fft-size --nr-preset mild", "mild --nr-preset --no-nr",
        "--nr-level", "--nr-level --no-res", "--nr-level --", "--fft-size --nr-level mild",
        "mild --nr-level --no-nr", "--nr-level --nr-preset", "--nr-preset --nr-level",
        "--nr-level mild --nr-level", "--nr-preset mild --nr-level --no-cng",
        "--aec-level", "--aec-level --no-res", "--aec-level --", "--fft-size --aec-level 1",
        "mild --aec-level --no-nr", "--aec-level --nr-level", "--aec-level mild --aec-level",
    };
    /* One AEC level per command line: a second one, in any mix of the positional and
     * --aec-level, is a command-line error (exit 2) even when it is an unknown word. */
    static const char* const second_level[] = {
        "1 2", "mild mild", "--aec-level 1 2", "1 --aec-level 2", "--aec-level 1 --aec-level 2",
        "--aec-level mild --aec-level mild", "1 --no-res 2", "--fft-size 256 1 bogus", "bogus 1",
        "2 --nr-level mild --aec-level 2",
    };
    size_t pi, ci;
    char bin[512], args[256];
    for (pi = 0; pi < 2; ++pi) {
        snprintf(bin, sizeof bin, "%s/%s", bin_dir, programs[pi]);
        for (ci = 0; ci < sizeof missing / sizeof missing[0]; ++ci) {
            snprintf(args, sizeof args, "no_mic.wav no_ref.wav no_out.wav %s", missing[ci]);
            CHECK(example_exit_is(bin, args, 2),
                  fmt_msg("%s %s: the value is missing, exit 2", programs[pi], missing[ci]));
        }
        for (ci = 0; ci < sizeof second_level / sizeof second_level[0]; ++ci) {
            snprintf(args, sizeof args, "no_mic.wav no_ref.wav no_out.wav %s", second_level[ci]);
            CHECK(example_exit_is(bin, args, 2),
                  fmt_msg("%s %s: a second AEC level, exit 2", programs[pi], second_level[ci]));
        }
        CHECK(example_output_has(bin, "no_mic.wav no_ref.wav no_out.wav 1 --aec-level 2", "second AEC level"),
              fmt_msg("%s a second AEC level names the problem on the output", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --nr-preset -x", 1),
              fmt_msg("%s --nr-preset -x: \"-x\" is a value, the run stops at the input", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --fft-size 256 --nr-preset mild", 1),
              fmt_msg("%s valid options reach the input check", programs[pi]));
        CHECK(example_output_has(bin, "", "--nr-level") && example_output_has(bin, "", "--nr-preset") &&
                  example_output_has(bin, "", "--aec-level"),
              fmt_msg("%s usage names --aec-level, --nr-level and its earlier spelling --nr-preset", programs[pi]));
        CHECK(example_output_has(bin, "", "give one of them once"),
              fmt_msg("%s usage says to give one AEC level", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --aec-level -x", 1),
              fmt_msg("%s --aec-level -x: \"-x\" is a value, the run stops at the input", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --aec-level 3 --fft-size 256", 1),
              fmt_msg("%s --aec-level with other options reaches the input check", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --nr-level -x", 1),
              fmt_msg("%s --nr-level -x: \"-x\" is a value, the run stops at the input", programs[pi]));
        CHECK(example_exit_is(bin, "no_mic.wav no_ref.wav no_out.wav --fft-size 256 --nr-level mild 2", 1),
              fmt_msg("%s --nr-level with an AEC level reaches the input check", programs[pi]));
    }
    snprintf(bin, sizeof bin, "%s/aec_nr_pipeline_static", bin_dir);
    CHECK(example_exit_is(bin, "--print-mem-size --nr-preset --no-res", 2),
          "aec_nr_pipeline_static --print-mem-size --nr-preset --no-res: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --nr-preset", 2),
          "aec_nr_pipeline_static --print-mem-size --nr-preset: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --nr-preset mild --fft-size 256", 0),
          "aec_nr_pipeline_static --print-mem-size with valid options: exit 0");
    CHECK(example_exit_is(bin, "--print-mem-size --nr-level --no-res", 2),
          "aec_nr_pipeline_static --print-mem-size --nr-level --no-res: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --nr-level", 2),
          "aec_nr_pipeline_static --print-mem-size --nr-level: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --aec-level --no-res", 2),
          "aec_nr_pipeline_static --print-mem-size --aec-level --no-res: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --aec-level", 2),
          "aec_nr_pipeline_static --print-mem-size --aec-level: exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --aec-level 3 --nr-level aggressive --fft-size 256", 0),
          "aec_nr_pipeline_static --print-mem-size --aec-level with --nr-level: exit 0");
    CHECK(example_exit_is(bin, "--print-mem-size 3 --nr-level aggressive --fft-size 256", 0),
          "aec_nr_pipeline_static --print-mem-size --nr-level with an AEC level: exit 0");
    CHECK(example_exit_is(bin, "--print-mem-size 1 2", 2),
          "aec_nr_pipeline_static --print-mem-size 1 2: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size mild mild", 2),
          "aec_nr_pipeline_static --print-mem-size mild mild: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --aec-level 1 2", 2),
          "aec_nr_pipeline_static --print-mem-size --aec-level 1 2: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size 1 --aec-level 2", 2),
          "aec_nr_pipeline_static --print-mem-size 1 --aec-level 2: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size --aec-level 1 --aec-level 2", 2),
          "aec_nr_pipeline_static --print-mem-size --aec-level 1 --aec-level 2: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size 1 --no-res bogus", 2),
          "aec_nr_pipeline_static --print-mem-size 1 --no-res bogus: a second AEC level, exit 2");
    CHECK(example_exit_is(bin, "--print-mem-size bogus", 0),
          "aec_nr_pipeline_static --print-mem-size bogus: one unknown level falls back, exit 0");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <example-bin-dir> <scratch-dir>\n", argv[0]);
        return 2;
    }
    for (int r = 0; r < N_GRIDS; r++) {
        int sr = GRIDS[r].sample_rate;
        int fft_size = GRIDS[r].fft_size;
        int hop_count = hop_count_for_rate(sr);
        printf("\n=== sample_rate = %d Hz / FFT %d (hop_count=%d) ===\n",
               sr, fft_size, hop_count);
        test_validation(sr, fft_size, r == 0); /* 44100 rejection checked once */
        test_pool_plan(sr, fft_size);
        test_create_vs_init_parity(sr, fft_size, hop_count);
        test_reset_equals_fresh_instance(sr, fft_size, /*enable_cng=*/1, /*enable_nr=*/1);
        test_reset_equals_fresh_instance(sr, fft_size, /*enable_cng=*/0, /*enable_nr=*/1);
        test_reset_equals_fresh_instance(sr, fft_size, /*enable_cng=*/0, /*enable_nr=*/0);
        test_destroy_idempotence(sr, fft_size);
    }

    printf("\n=== lane and chain configuration validation ===\n");
    test_config_validation_rejects();

    printf("\n=== NR / RES switches ===\n");
    test_nr_res_switches();

    printf("\n=== comfort-noise contract ===\n");
    test_comfort_noise_contract();
    test_comfort_noise_follows_nr_gain();

    printf("\n=== chain against the hand-written oracle ===\n");
    test_chain_equals_oracle();
    test_frame_and_chain_args();
    test_aec_only_output();

    printf("\n=== the full-mode identity (NR off) ===\n");
    test_full_mode_identity();
    test_nr_keeps_running();

    printf("\n=== the example programs against the oracle ===\n");
    test_examples_black_box(argv[1], argv[2]);
    test_examples_aec_only(argv[1], argv[2]);
    test_examples_full_mode(argv[1], argv[2]);
    test_examples_full_chain(argv[1], argv[2]);
    test_examples_cli_values(argv[1]);

    if (g_failures) {
        fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
