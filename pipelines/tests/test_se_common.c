/* Acceptance test for pipelines/common and pipelines/samples: the shared grid and AecConfig
 * derivation (se_config), the AEC-context frame view (se_stage.h), the noise-reduction stage
 * call (samples/se_anr_stage.h) and the gain sequence (samples/se_sample_chain.h). Expected values
 * are written out by hand, not recomputed with the code under test; the sample chain is
 * compared bit for bit against a chain written out in this file from the library calls. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_sample_chain.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

static SeRequest base_request(int sr) {
    SeRequest r;
    memset(&r, 0, sizeof r);
    r.sample_rate = sr;
    r.fft_size = 0;
    r.filter_length = 0;
    r.delay_mode = AEC_DELAY_MATCHED;
    r.delay_num_filters = 5;
    r.fixed_delay_samples = -1;
    r.aec_preset = AEC_PRESET_BALANCED;
    return r;
}

static SeRules mono_rules(void) {
    SeRules u = {1, 0, 0, 0, 0, SE_CONTEXT_PRESET};
    return u;
}

static SeRules four_channel_rules(int post) {
    SeRules u = {0, 0, 0, 1, 1, post ? AEC_CONTEXT_NO_GAIN : AEC_CONTEXT_LINEAR_ONLY};
    return u;
}

static void test_grid(void) {
    static const struct { int sr, req, allow8, ok, fft; } t[] = {
        {16000, 0, 0, 1, 256}, {16000, 256, 0, 1, 256}, {16000, 512, 0, 1, 512},
        {16000, 1024, 0, 0, 0}, {16000, 128, 0, 0, 0},
        {48000, 0, 0, 1, 1024}, {48000, 1024, 0, 1, 1024}, {48000, 512, 0, 0, 0},
        {8000, 0, 1, 1, 256}, {8000, 256, 1, 1, 256}, {8000, 512, 1, 0, 0},
        {8000, 0, 0, 0, 0}, {8000, 256, 0, 0, 0},
        {44100, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {-16000, 256, 1, 0, 0}, {32000, 256, 1, 0, 0},
        {16000, -1, 1, 0, 0},
    };
    size_t i;
    SeGrid g;
    for (i = 0; i < sizeof t / sizeof t[0]; ++i) {
        int rc = se_grid_resolve(t[i].sr, t[i].req, t[i].allow8, &g);
        CHECK((rc == 0) == t[i].ok, "grid accept/reject");
        if (rc == 0 && t[i].ok) {
            CHECK(g.fft_size == t[i].fft, "grid fft");
            CHECK(g.hop_size == t[i].fft / 2, "grid hop");
            CHECK(g.n_freqs == t[i].fft / 2 + 1, "grid bins");
        }
    }
    CHECK(se_grid_resolve(16000, 0, 0, NULL) == -1, "grid NULL out");
    CHECK(se_grid_fixed(16000, 0, 16000, 512, &g) == 0 && g.fft_size == 512 && g.hop_size == 256 &&
          g.n_freqs == 257, "fixed grid default");
    CHECK(se_grid_fixed(16000, 512, 16000, 512, &g) == 0, "fixed grid explicit");
    CHECK(se_grid_fixed(16000, 256, 16000, 512, &g) == -1, "fixed grid wrong fft");
    CHECK(se_grid_fixed(48000, 0, 16000, 512, &g) == -1, "fixed grid wrong rate");
}

static void test_derive_fields(void) {
    SeRequest r = base_request(16000);
    SeRules mono = mono_rules();
    SeGrid g;
    AecConfig a, preset;

    CHECK(se_derive(&r, &mono, &g, &a) == 0, "mono derive");
    aec_config_from_preset(&preset, AEC_PRESET_BALANCED, 16000);
    CHECK(g.fft_size == 256 && g.hop_size == 128 && g.n_freqs == 129, "mono grid");
    CHECK(a.fft_size == 256, "fft_size");
    CHECK(a.filter_length == preset.filter_length, "preset filter length kept for 0");
    CHECK(a.delay_mode == AEC_DELAY_MATCHED && a.delay_num_filters == 5 &&
          a.fixed_delay_samples == -1 && a.enable_delay_est == 1, "matched passthrough");
    CHECK(a.enable_res == 0 && a.return_res_context == 1, "seam");
    CHECK(a.spatial_linear_context == preset.spatial_linear_context, "context kept for the preset level");
    CHECK(a.min_gain_floor_far_active_db == preset.min_gain_floor_far_active_db, "preset applied");

    r.filter_length = 1234;
    r.delay_mode = AEC_DELAY_FIXED;
    r.delay_num_filters = 3;
    r.fixed_delay_samples = 77;
    r.aec_preset = AEC_PRESET_AGGRESSIVE;
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "mono derive fixed");
    aec_config_from_preset(&preset, AEC_PRESET_AGGRESSIVE, 16000);
    CHECK(a.filter_length == 1234, "filter length override");
    CHECK(a.delay_mode == AEC_DELAY_FIXED && a.delay_num_filters == 3 && a.fixed_delay_samples == 77 &&
          a.enable_delay_est == 0, "fixed passthrough");
    CHECK(a.min_gain_floor_far_active_db == preset.min_gain_floor_far_active_db, "aggressive preset applied");

    r.aec_preset = AEC_PRESET_NO_FLOOR;
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "mono derive no_floor");
    aec_config_from_preset(&preset, AEC_PRESET_NO_FLOOR, 16000);
    CHECK(a.min_gain_floor_far_active_db == preset.min_gain_floor_far_active_db &&
          a.min_gain_floor_dt_db == preset.min_gain_floor_dt_db &&
          a.min_gain_split_floor_enabled == preset.min_gain_split_floor_enabled &&
          a.min_gain_split_floor_enabled == 0,
          "no_floor preset applied, gate disabled, matches aec_config_from_preset() byte-for-byte "
          "(the same values the old BALANCED-construct + aec_res_set_preset(NO_FLOOR) runtime "
          "retarget workaround converged to)");

    {   /* lane aligned: the request's delay settings do not reach the AEC */
        SeRules four = four_channel_rules(1);
        SeRequest q = base_request(48000);
        q.delay_mode = AEC_DELAY_FIXED;
        q.delay_num_filters = DA_NUM_FILTERS;
        q.fixed_delay_samples = 480;
        CHECK(se_derive(&q, &four, &g, &a) == 0, "four-channel derive");
        CHECK(g.fft_size == 1024, "four-channel grid");
        CHECK(a.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED && a.delay_num_filters == DA_NUM_FILTERS &&
              a.fixed_delay_samples == -1 && a.enable_delay_est == 0, "lane aligned");
        CHECK(a.spatial_linear_context == AEC_CONTEXT_NO_GAIN, "post context level");
        four = four_channel_rules(0);
        CHECK(se_derive(&q, &four, &g, &a) == 0 && a.spatial_linear_context == AEC_CONTEXT_LINEAR_ONLY,
              "front-only context level");
    }
    {   /* compiled grid, linear-only seam */
        SeRules u = {0, 16000, 512, 0, 0, AEC_CONTEXT_LINEAR_ONLY};
        SeRequest q = base_request(16000);
        q.delay_mode = (AecDelayMode)9;   /* passed through, the AEC validates it */
        CHECK(se_derive(&q, &u, &g, &a) == 0, "compiled-grid derive");
        CHECK(g.fft_size == 512 && g.hop_size == 256 && g.n_freqs == 257, "compiled grid");
        CHECK(a.spatial_linear_context == AEC_CONTEXT_LINEAR_ONLY, "linear-only context");
        q.sample_rate = 48000;
        CHECK(se_derive(&q, &u, &g, &a) == -1, "compiled grid rate");
    }
}

static void test_derive_rejects(void) {
    SeRules mono = mono_rules();
    SeRules four = four_channel_rules(1);
    SeGrid g;
    AecConfig a;
    SeRequest r;
    int v;

    CHECK(se_derive(NULL, &mono, &g, &a) == -1 && se_derive(&(SeRequest){0}, NULL, &g, &a) == -1 &&
          se_derive(&(SeRequest){0}, &mono, NULL, &a) == -1 && se_derive(&(SeRequest){0}, &mono, &g, NULL) == -1,
          "NULL arguments");

    r = base_request(16000);
    r.aec_preset = AEC_PRESET_NO_FLOOR;   /* value 3: in range now, must be accepted */
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "preset NO_FLOOR accepted at construction");
    CHECK(a.min_gain_split_floor_enabled == 0, "NO_FLOOR construction disables the split floor gate");
    r.aec_preset = (AecPreset)4;
    CHECK(se_derive(&r, &mono, &g, &a) == -1, "preset 4 (past NO_FLOOR, still out of range)");
    r.aec_preset = (AecPreset)-1;
    CHECK(se_derive(&r, &mono, &g, &a) == -1, "preset -1");

    for (v = -2; v <= 4098; ++v) {
        r = base_request(16000);
        r.filter_length = v;
        CHECK((se_derive(&r, &mono, &g, &a) == 0) == (v >= 0 && v <= 4096), "filter length bound");
    }

    /* the strict delay request: MATCHED bank, FIXED bound, EXTERNAL_ALIGNED neither */
    r = base_request(16000);
    CHECK(se_derive(&r, &four, &g, &a) == 0, "matched ok");
    r.delay_num_filters = 0;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "matched 0 filters");
    r.delay_num_filters = DA_NUM_FILTERS + 1;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "matched too many filters");
    r = base_request(16000);
    r.fixed_delay_samples = 0;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "matched with a fixed delay");
    r = base_request(16000);
    r.delay_mode = AEC_DELAY_FIXED;
    r.delay_num_filters = DA_NUM_FILTERS;
    r.fixed_delay_samples = 120 * 16000;
    CHECK(se_derive(&r, &four, &g, &a) == 0, "fixed at the bound");
    r.fixed_delay_samples = 120 * 16000 + 1;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "fixed past the bound");
    r.fixed_delay_samples = -1;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "fixed negative");
    r.fixed_delay_samples = 5;
    r.delay_num_filters = 1;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "fixed with another bank size");
    r = base_request(16000);
    r.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    r.delay_num_filters = DA_NUM_FILTERS;
    CHECK(se_derive(&r, &four, &g, &a) == 0, "external ok");
    r.fixed_delay_samples = 3;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "external with a fixed delay");
    r = base_request(16000);
    r.delay_mode = (AecDelayMode)3;
    CHECK(se_derive(&r, &four, &g, &a) == -1, "delay mode 3");
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "delay mode passed through without the strict check");
    r = base_request(8000);
    CHECK(se_derive(&r, &four, &g, &a) == -1, "four-channel rules refuse 8 kHz");
    CHECK(se_derive(&r, &mono, &g, &a) == 0 && g.fft_size == 256, "mono rules accept 8 kHz");
}

/* The denoiser configuration the oracles use: the library preset for `mode` with the 512 ms
 * minimum window written out in whole hops rather than taken from the preset. */
static AnrConfig oracle_nr_config(int sr, int fft, AnrMode mode) {
    AnrConfig c = anr_config_for_mode_grid(sr, fft, mode);
    int hop = fft / 2;
    c.L = (512 * sr + 1000 * hop - 1) / (1000 * hop);
    return c;
}

/* The chain's denoiser is the library preset; the scene-reset gate stays off, which keeps the
 * gain independent of the window length. */
static void test_nr_config(void) {
    static const struct { int sr, fft, l; } grid[] = {
        {8000, 256, 32}, {16000, 256, 64}, {16000, 512, 32}, {48000, 1024, 48}};
    size_t i;
    int m;
    for (i = 0; i < sizeof grid / sizeof grid[0]; ++i) {
        for (m = 0; m < 4; ++m) {
            AnrConfig canon = anr_config_for_mode_grid(grid[i].sr, grid[i].fft, (AnrMode)m);
            AnrConfig o = oracle_nr_config(grid[i].sr, grid[i].fft, (AnrMode)m);
            CHECK(canon.L == grid[i].l, "the preset's minimum window is 512 ms in whole hops");
            CHECK(canon.alpha_decay == canon.alpha_g, "the preset's alpha_decay equals alpha_g");
            CHECK(canon.broadband_threshold == 1.0f, "scene-reset gate stays off");
            CHECK(memcmp(&o, &canon, sizeof o) == 0, "the oracle configuration equals the preset");
        }
    }
}

static void *aligned_pool(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) return NULL;
    memset(p, 0xCD, bytes);
    return p;
}

static void test_stage(void) {
    const int sr = 16000, fft = 256, n = 129;
    AnrConfig cfg = anr_config_for_mode_grid(sr, fft, ANR_MODE_BALANCED);
    size_t bytes = anr_get_mem_size(&cfg);
    void *pool = aligned_pool(bytes), *pool2 = aligned_pool(bytes), *pool3 = aligned_pool(bytes);
    AnrDenoiser *a, *b, *c;
    Complex spec[129];
    float r2[129], scratch[129], extra[129];
    const float *ga, *gb;
    SeSpecFrame f;
    int k, m, frame;

    memset(&f, 0, sizeof(f));
    f.n_freqs = n;
    f.e = spec;
    f.r2 = r2;
    f.usable_linear = 1;

    CHECK(bytes > 0 && pool && pool2 && pool3, "stage size");
    a = anr_init(pool, bytes, &cfg);
    b = anr_init(pool2, bytes, &cfg);
    c = anr_init(pool3, bytes, &cfg);
    CHECK(a && b && c, "stage init");

    for (m = 0; m < 4; ++m)
        CHECK(anr_set_mode(a, (AnrMode)m) == 0, "anr_set_mode accepts every mode on the preset instance");
    CHECK(anr_set_mode(a, ANR_MODE_BALANCED) == 0, "back to balanced");

    /* the residual-echo prior reaches the denoiser as r2 / SE_PSD_SCALE */
    for (frame = 0; frame < 40; ++frame) {
        for (k = 0; k < n; ++k) {
            spec[k].r = (float)((frame * 7 + k * 13) % 101) * 31.0f - 1500.0f;
            spec[k].i = (float)((frame * 5 + k * 3) % 89) * 17.0f - 700.0f;
            r2[k] = (float)((k + 1) * (frame + 3)) * 4.0e5f;
            extra[k] = r2[k] / (32768.0f * 32768.0f);
        }
        ga = se_anr_stage_gain(a, &f, scratch, 1);
        CHECK(ga != NULL, "stage process");
        CHECK(anr_process_gain(b, spec, extra, NULL) == 0, "direct process");
        gb = anr_get_gain(b, NULL);
        CHECK(ga && gb && memcmp(ga, gb, (size_t)n * sizeof(float)) == 0, "stage gain equals the direct call");
        CHECK(memcmp(scratch, extra, (size_t)n * sizeof(float)) == 0, "scratch holds r2 / SE_PSD_SCALE");
    }
    /* usable_linear = 0 fails closed: with r2 and scratch both non-NULL, a cold-start
     * unusable estimate must not fold into the noise floor, so the gain equals the plain
     * (use_r2 = 0) call. r2 is sized like the cold-start fault this gate exists for
     * (order 1e12). The mutation guard below (denoiser c) shows the comparison is not
     * vacuous. */
    anr_reset(a);
    anr_reset(b);
    anr_reset(c);
    f.usable_linear = 0;
    /* The denoiser bypasses NR for its first num_init_frames, so the mutation guard below
     * (folding this r2 must visibly differ from not folding it) is checked only after that
     * warm-up; the ga == gb gate check runs every hop. */
    for (frame = 0; frame < 35; ++frame) {
        for (k = 0; k < n; ++k) {
            spec[k].r = (float)((frame * 7 + k * 13) % 101) * 31.0f - 1500.0f;
            spec[k].i = (float)((frame * 5 + k * 3) % 89) * 17.0f - 700.0f;
            r2[k] = 3.0e12f + (float)k * 1.0e9f;
            extra[k] = r2[k] / (32768.0f * 32768.0f);
        }
        ga = se_anr_stage_gain(a, &f, scratch, 1);
        CHECK(ga != NULL, "usable_linear=0 stage process");
        CHECK(anr_process_gain(b, spec, NULL, NULL) == 0, "usable_linear=0 direct process (no prior)");
        gb = anr_get_gain(b, NULL);
        CHECK(ga && gb && memcmp(ga, gb, (size_t)n * sizeof(float)) == 0,
              "usable_linear=0 ignores r2 even though r2 and scratch are both present");
        /* Mutation guard: on these same inputs, actually folding r2 (as the
         * gate must prevent) measurably suppresses the gain below the plain
         * value -- so the CHECK above is not a vacuous no-op comparison. */
        CHECK(anr_process_gain(c, spec, extra, NULL) == 0, "counterfactual folded process");
        if (frame >= 25) {
            const float* gc = anr_get_gain(c, NULL);
            int differs = 0;
            for (k = 0; k < n; ++k)
                if (gc[k] < gb[k] - 1e-6f) { differs = 1; break; }
            CHECK(differs, "folding this r2 would have suppressed the gain below the gated value");
        }
    }
    f.usable_linear = 1;
    /* without a prior: NULL r2 and NULL scratch both select plain NR */
    anr_reset(a);
    anr_reset(b);
    for (frame = 0; frame < 10; ++frame) {
        for (k = 0; k < n; ++k) { spec[k].r = (float)(k + frame); spec[k].i = (float)(frame - k); }
        ga = se_anr_stage_gain(a, &f, scratch, 0);
        CHECK(ga != NULL, "no prior");
        CHECK(anr_process_gain(b, spec, NULL, NULL) == 0, "direct no prior");
        CHECK(memcmp(ga, anr_get_gain(b, NULL), (size_t)n * sizeof(float)) == 0, "no-prior gain");
        CHECK(se_anr_stage_gain(a, &f, NULL, 1) != NULL, "no scratch selects plain NR");
    }
    CHECK(se_anr_stage_gain(NULL, &f, NULL, 0) == NULL, "NULL denoiser");
    CHECK(se_anr_stage_gain(a, NULL, NULL, 0) == NULL, "NULL frame");
    f.e = NULL;
    CHECK(se_anr_stage_gain(a, &f, NULL, 0) == NULL, "NULL spectrum");

    anr_destroy(a);
    anr_destroy(b);
    anr_destroy(c);
    free(pool);
    free(pool2);
    free(pool3);
}

/* The prior scale is the exact reciprocal multiply of the division, for every
 * kind of float: random bit patterns, tiny values whose scaled result is
 * subnormal, zeros, infinities and NaN. The divisor goes through a volatile
 * so the reference stays a real division. */
static void test_psd_scale(void) {
    const int sr = 16000, fft = 256, n = 129;
    AnrConfig cfg = anr_config_for_mode_grid(sr, fft, ANR_MODE_BALANCED);
    size_t bytes = anr_get_mem_size(&cfg);
    void *pool = aligned_pool(bytes);
    AnrDenoiser *a = pool ? anr_init(pool, bytes, &cfg) : NULL;
    volatile float divisor = SE_PSD_SCALE;
    Complex spec[129];
    float r2[129], scratch[129];
    SeSpecFrame f;
    uint32_t seed = 987654321u;
    int frame, k, ok = 1, ok_len = 1;
    const uint32_t fixed[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
                              0x00800000u, 0x3f800000u, 0x4e800000u, 0x7f7fffffu,
                              0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7fc00001u,
                              0xffa00000u, 0x30800000u, 0x30800001u, 0x0f800000u};

    memset(&f, 0, sizeof f);
    f.n_freqs = n;
    f.e = spec;
    f.r2 = r2;
    f.usable_linear = 1;
    CHECK(a != NULL, "psd scale: stage init");
    if (!a) { free(pool); return; }
    for (k = 0; k < n; ++k) { spec[k].r = 100.0f + (float)k; spec[k].i = -50.0f; }
    for (frame = 0; frame < 4000 && ok; ++frame) {
        for (k = 0; k < n; ++k) {
            uint32_t bits;
            seed = seed * 1664525u + 1013904223u;
            bits = seed;
            seed = seed * 1664525u + 1013904223u;
            if ((seed >> 28) < 4u) bits = (bits & 0x807fffffu) | (((seed >> 8) % 42u) << 23);
            else if ((seed >> 28) < 6u) bits = fixed[(seed >> 8) % (sizeof fixed / sizeof fixed[0])];
            memcpy(&r2[k], &bits, sizeof bits);
        }
        f.n_freqs = n;
        (void)se_anr_stage_gain(a, &f, scratch, 1);
        for (k = 0; k < n; ++k) {
            float want = r2[k] / divisor;
            uint32_t gb, wb;
            memcpy(&gb, &scratch[k], sizeof gb);
            memcpy(&wb, &want, sizeof wb);
            if (gb != wb && !(isnan(scratch[k]) && isnan(want))) ok = 0;
        }
        anr_reset(a);
    }
    CHECK(ok, "psd scale: scratch equals r2 / SE_PSD_SCALE for every float class");

    /* every length around the vector widths */
    for (k = 0; k < n; ++k) r2[k] = (float)(k + 1) * 3.0e5f;
    for (frame = 1; frame <= n && ok_len; ++frame) {
        float guard = 1234.5f;
        scratch[frame < n ? frame : n - 1] = guard;
        f.n_freqs = frame;
        (void)se_anr_stage_gain(a, &f, scratch, 1);
        for (k = 0; k < frame; ++k)
            if (scratch[k] != r2[k] / divisor) ok_len = 0;
        if (frame < n && scratch[frame] != guard) ok_len = 0;
        anr_reset(a);
    }
    CHECK(ok_len, "psd scale: every length 1..129 converts exactly and stops at n_freqs");
    anr_destroy(a);
    free(pool);
}

/* ------------------------------------------------------------------------------------------
 * se_frame_from_aec: every SeSpecFrame field against the AEC's own context getters
 * ------------------------------------------------------------------------------------------ */

static uint32_t g_lcg;
static float lcg_unit(void) {   /* [0, 1) */
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)(g_lcg >> 8) / 16777216.0f;
}

#define ECHO_DELAY 2000   /* beyond the 832-tap reach: the matched filter must lock */

static void test_frame_from_aec(void) {
    enum { HOPS = 220, HOP = 128, NF = 129 };
    SeRequest r = base_request(16000);
    SeRules mono = mono_rules();
    SeGrid g;
    AecConfig a, bare;
    size_t bytes;
    void *pool, *pool2;
    Aec *aec, *aec_bare;
    static float dline[ECHO_DELAY];
    float mic[HOP], ref[HOP];
    int dpos = 0, h, i, k;
    int n_unlocked = 0, n_locked = 0, n_changed = 0, mismatch = 0, sep = 1;
    SeSpecFrame f;
    AecResContext c;
    AecLinearContext l;

    r.fft_size = 256;
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "frame view: mono derive");
    bytes = aec_get_mem_size(&a);
    pool = aligned_pool(bytes);
    aec = pool ? aec_init(pool, bytes, &a) : NULL;
    CHECK(aec != NULL, "frame view: aec init");
    if (!aec) { free(pool); return; }

    g_lcg = 0xC0FFEEu;
    for (h = 0; h < HOPS; ++h) {
        for (i = 0; i < HOP; ++i) {
            float x = (lcg_unit() - 0.5f) * 0.5f;
            float delayed = dline[dpos];
            dline[dpos] = x;
            dpos = (dpos + 1) % ECHO_DELAY;
            ref[i] = x;
            mic[i] = 0.3f * delayed + 0.02f * (lcg_unit() - 0.5f);
        }
        aec_process_context(aec, mic, ref);
        memset(&f, 0x5A, sizeof f);   /* a field the call forgets keeps this pattern */
        if (se_frame_from_aec(aec, &f) != 0) { mismatch++; continue; }
        aec_get_res_context(aec, &c);
        aec_get_linear_context(aec, &l);
        if (f.n_freqs != NF || f.n_freqs != c.n_freqs || f.hop != HOP || f.hop != c.hop_size) mismatch++;
        if (f.e != c.error_spec || f.y != c.near_spec || f.x != c.far_spec) mismatch++;
        if (f.r2 != c.r2 || f.cn != c.comfort_noise || f.g_res_ctx != c.res_gain) mismatch++;
        if (f.far_hop != l.aligned_far_hop || f.formed_hop != c.formed_hop) mismatch++;
        if (!f.e || !f.y || !f.x || !f.r2 || !f.cn || !f.g_res_ctx || !f.far_hop || !f.formed_hop) mismatch++;
        if (f.far_power != c.far_power || f.max_dt != c.dt_indicator || f.max_sat != c.saturation_level)
            mismatch++;
        if (f.all_converged != c.filter_converged || f.usable_linear != c.usable_linear ||
            f.clock_drift != 0) mismatch++;
        if (f.delay_changed != (l.delay_state == AEC_LINEAR_DELAY_CHANGED ? 1 : 0)) mismatch++;
        if (f.delay_locked != (l.delay_state == AEC_LINEAR_DELAY_UNLOCKED ? 0 : 1)) mismatch++;
        if (l.delay_state == AEC_LINEAR_DELAY_UNLOCKED) n_unlocked++;
        else if (l.delay_state == AEC_LINEAR_DELAY_LOCKED) n_locked++;
        else n_changed++;
        /* y is the capture spectrum (error plus echo estimate), x the far spectrum: a swap of
         * the two views cannot satisfy this. */
        if (h == HOPS - 1) {
            for (k = 0; k < NF; ++k) {
                float er = c.error_spec[k].r + c.echo_spec[k].r;
                float ei = c.error_spec[k].i + c.echo_spec[k].i;
                float tol = 1e-4f * (fabsf(er) + fabsf(ei)) + 1e-9f;
                if (fabsf(f.y[k].r - er) > tol || fabsf(f.y[k].i - ei) > tol) sep = 0;
            }
        }
    }
    CHECK(mismatch == 0, "frame view: every field equals the AEC context getters on every hop");
    CHECK(n_unlocked > 0 && n_locked > 0 && n_changed > 0,
          "frame view: the run visits UNLOCKED, LOCKED and CHANGED (delay flags are exercised)");
    CHECK(sep, "frame view: y is the capture spectrum (error + echo), not the far spectrum");

    CHECK(se_frame_from_aec(NULL, &f) == -1 && se_frame_from_aec(aec, NULL) == -1,
          "frame view: NULL arguments refused");

    /* a lane without the seam exposes no error spectrum */
    bare = a;
    bare.enable_res = 0;
    bare.return_res_context = 0;
    bytes = aec_get_mem_size(&bare);
    pool2 = aligned_pool(bytes);
    aec_bare = pool2 ? aec_init(pool2, bytes, &bare) : NULL;
    CHECK(aec_bare != NULL, "frame view: seam-less lane init");
    if (aec_bare) {
        aec_process_context(aec_bare, mic, ref);
        CHECK(se_frame_from_aec(aec_bare, &f) == -1, "frame view: lane without the seam refused");
        aec_destroy(aec_bare);
    }
    aec_destroy(aec);
    free(pool);
    free(pool2);
}

/* ------------------------------------------------------------------------------------------
 * the sample chain against a chain written out from the library calls
 * ------------------------------------------------------------------------------------------ */

#define MAXN 513

/* One scripted hop of the frame the chain consumes. */
typedef struct ScriptFrame {
    Complex e[MAXN], y[MAXN];
    float r2[MAXN], cn[MAXN], g_res[MAXN], far_hop[MAXN];
    SeSpecFrame f;
} ScriptFrame;

static void script_hop(ScriptFrame* s, int n, int hop, int hop_index) {
    int k;
    for (k = 0; k < n; ++k) {
        float amp = 0.25f / (1.0f + 0.04f * (float)k);
        s->e[k].r = (lcg_unit() - 0.5f) * amp;
        s->e[k].i = (lcg_unit() - 0.5f) * amp;
        s->y[k].r = s->e[k].r + (lcg_unit() - 0.5f) * amp;
        s->y[k].i = s->e[k].i + (lcg_unit() - 0.5f) * amp;
        s->r2[k] = lcg_unit() * 8.0e6f;
        s->cn[k] = lcg_unit() * 4.0e3f;
        s->g_res[k] = (k % 5 == 0) ? 1.0f : 0.05f + 0.95f * lcg_unit();
    }
    for (k = 0; k < hop; ++k) s->far_hop[k] = (lcg_unit() - 0.5f) * 0.6f;
    memset(&s->f, 0, sizeof s->f);
    s->f.n_freqs = n;
    s->f.hop = hop;
    s->f.e = s->e;
    s->f.y = s->y;
    s->f.r2 = s->r2;
    s->f.cn = s->cn;
    s->f.g_res_ctx = s->g_res;
    s->f.far_hop = s->far_hop;
    s->f.far_power = 0.1f;
    s->f.max_dt = (hop_index % 9 == 0) ? 0.6f : 0.05f;
    s->f.max_sat = (hop_index % 13 == 0) ? 0.7f : 0.1f;
    s->f.all_converged = (hop_index % 4) != 1;
    s->f.usable_linear = (hop_index % 5) != 3;
}

enum { SABOTAGE_NONE = 0, SABOTAGE_NO_CNG, SABOTAGE_NO_USABLE_GATE, SABOTAGE_OTHER_PRESET,
       SABOTAGE_PRIOR_FLIPPED, SABOTAGE_CNG_LEVEL_AFTER_MIN };

/* The chain spelled out over the library calls, with the NR recipe and the RES configuration
 * derived here and the min/apply written as scalar loops. `sabotage` plants one defect so a
 * test can show the comparison is able to fail. */
typedef struct Oracle {
    SeSampleChainConfig cfg;
    int n, hop, sabotage;
    AecRes* res;
    AnrDenoiser* nr;
    AcIstft* istft;
    void *res_mem, *nr_mem, *istft_mem;
    uint32_t rng;
    float unity[MAXN], total[MAXN], extra[MAXN];
    Complex spec[MAXN];
} Oracle;

static int oracle_init(Oracle* o, const SeSampleChainConfig* cfg, int sabotage) {
    SeGrid g;
    int k;
    memset(o, 0, sizeof *o);
    o->cfg = *cfg;
    o->sabotage = sabotage;
    o->n = cfg->fft_size / 2 + 1;
    o->hop = cfg->fft_size / 2;
    for (k = 0; k < o->n; ++k) o->unity[k] = 1.0f;
    o->rng = SE_CNG_RNG_SEED;
    if (cfg->res == SE_SAMPLE_RES_AEC_RES) {
        /* the lane derivation of a 4-channel front end (8 kHz is mono-only) */
        SeRequest q = base_request(cfg->sample_rate);
        SeRules rules = cfg->sample_rate == 8000 ? mono_rules() : four_channel_rules(1);
        AecConfig rc;
        size_t b;
        q.fft_size = cfg->fft_size;
        q.aec_preset = sabotage == SABOTAGE_OTHER_PRESET ? AEC_PRESET_MILD : cfg->res_preset;
        if (se_derive(&q, &rules, &g, &rc) != 0) return -1;
        b = aec_res_get_mem_size(&rc);
        o->res_mem = aligned_pool(b);
        o->res = o->res_mem ? aec_res_init(o->res_mem, b, &rc) : NULL;
        if (!o->res) return -1;
    }
    if (cfg->enable_nr) {
        AnrConfig nc = oracle_nr_config(cfg->sample_rate, cfg->fft_size, cfg->nr_mode);
        size_t b = anr_get_mem_size(&nc);
        o->nr_mem = aligned_pool(b);
        o->nr = o->nr_mem ? anr_init(o->nr_mem, b, &nc) : NULL;
        if (!o->nr) return -1;
    }
    if (cfg->with_synthesis) {
        AcStftConfig sc;
        size_t b;
        ac_stft_config_default(&sc, cfg->fft_size);
        b = ac_istft_get_mem_size(&sc);
        o->istft_mem = aligned_pool(b);
        o->istft = o->istft_mem ? ac_istft_init(o->istft_mem, b, &sc) : NULL;
        if (!o->istft) return -1;
    }
    return 0;
}

static void oracle_free(Oracle* o) {
    anr_destroy(o->nr);
    free(o->res_mem);
    free(o->nr_mem);
    free(o->istft_mem);
}

/* One hop; out (hop samples) is written when the oracle synthesises. Returns 0, or -1 when a
 * stage refuses. */
static int oracle_hop(Oracle* o, const SeSpecFrame* f, float* out) {
    const float *gres = o->unity, *gnr = o->unity;
    int k;
    switch (o->cfg.res) {
        case SE_SAMPLE_RES_FRAME_GAIN: gres = f->g_res_ctx; break;
        case SE_SAMPLE_RES_AEC_RES: {
            AecResInput in;
            in.error_spec = f->e;
            in.near_spec = f->y;
            in.r2 = f->r2;
            in.comfort_noise = f->cn;
            in.render_hop = f->far_hop;
            in.all_converged = f->all_converged;
            in.max_dt = f->max_dt;
            in.max_saturation = f->max_sat;
            in.clock_drift = f->clock_drift;
            gres = aec_res_process(o->res, &in);
            break;
        }
        default: break;
    }
    if (!gres) return -1;
    if (o->nr) {
        const float* extra = NULL;
        int use = o->sabotage == SABOTAGE_PRIOR_FLIPPED ? o->cfg.legacy_amin : !o->cfg.legacy_amin;
        int gate = o->sabotage == SABOTAGE_NO_USABLE_GATE ? 1 : f->usable_linear;
        if (use && f->r2 && gate) {
            for (k = 0; k < f->n_freqs; ++k) o->extra[k] = f->r2[k] * 0x1p-30f;
            extra = o->extra;
        }
        if (anr_process_gain(o->nr, f->e, extra, NULL) != 0) return -1;
        gnr = anr_get_gain(o->nr, NULL);
        if (!gnr) return -1;
    }
    for (k = 0; k < f->n_freqs; ++k) {
        float a = gnr[k], b = gres[k];
        o->total[k] = a < b ? a : b;
        o->spec[k].r = f->e[k].r * o->total[k];
        o->spec[k].i = f->e[k].i * o->total[k];
    }
    if (o->cfg.enable_cng && o->cfg.res != SE_SAMPLE_RES_OFF && f->cn && o->sabotage != SABOTAGE_NO_CNG)
        aec_cng_fill(&o->rng, o->spec, f->cn, gres, gnr, SE_CNG_NR_GAIN_FLOOR, f->n_freqs);
    if (o->istft && out) (void)ac_istft_push(o->istft, o->spec, out);
    return 0;
}

typedef struct ChainRun {
    int hops, mismatch_p, mismatch_out, mismatch_gain, null_hops, cut_hops;
} ChainRun;

/* Runs `hops` scripted hops through the sample chain and an oracle built for the same
 * configuration (with `sabotage`), comparing P, the hop and both gains bit for bit. */
static ChainRun compare_chain(const SeSampleChainConfig* cfg, int sabotage, int hops, uint32_t seed) {
    ChainRun r;
    static ScriptFrame sf;
    static float out_c[MAXN], out_o[MAXN];
    size_t bytes = se_sample_chain_mem_size(cfg);
    void* pool = aligned_pool(bytes);
    SeSampleChain* c = pool ? se_sample_chain_init(pool, bytes, cfg) : NULL;
    Oracle o;
    int h;
    memset(&r, 0, sizeof r);
    if (!c || oracle_init(&o, cfg, sabotage) != 0) { r.null_hops = -1; free(pool); return r; }
    g_lcg = seed;
    for (h = 0; h < hops; ++h) {
        const Complex* p;
        int orc;
        script_hop(&sf, c->n_freqs, c->hop, h);
        memset(out_c, 0, sizeof out_c);
        memset(out_o, 0, sizeof out_o);
        p = se_sample_chain_form(c, &sf.f);
        orc = oracle_hop(&o, &sf.f, out_o);
        if (!p || orc != 0) { r.null_hops++; continue; }
        if (memcmp(p, o.spec, (size_t)c->n_freqs * sizeof(Complex)) != 0) r.mismatch_p++;
        if (memcmp(p, sf.e, (size_t)c->n_freqs * sizeof(Complex)) != 0) r.cut_hops++;
        if (cfg->with_synthesis) {
            if (se_sample_chain_synthesize(c, p, out_c) != 0 ||
                memcmp(out_c, out_o, (size_t)c->hop * sizeof(float)) != 0) r.mismatch_out++;
        }
        {
            const float* gr = se_sample_chain_last_res_gain(c);
            const float* gn = se_sample_chain_last_nr_gain(c);
            const float* er = cfg->res == SE_SAMPLE_RES_OFF ? o.unity
                            : cfg->res == SE_SAMPLE_RES_FRAME_GAIN ? sf.f.g_res_ctx
                            : (const float*)NULL;
            if (!gr || !gn) r.mismatch_gain++;
            else {
                if (er && memcmp(gr, er, (size_t)c->n_freqs * sizeof(float)) != 0) r.mismatch_gain++;
                if (!o.nr && memcmp(gn, o.unity, (size_t)c->n_freqs * sizeof(float)) != 0) r.mismatch_gain++;
                if (o.nr && memcmp(gn, anr_get_gain(o.nr, NULL), (size_t)c->n_freqs * sizeof(float)) != 0)
                    r.mismatch_gain++;
            }
        }
        r.hops++;
    }
    se_sample_chain_destroy(c);
    oracle_free(&o);
    free(pool);
    return r;
}

static void test_chain_equals_oracle(void) {
    static const struct { int sr, fft; } grid[] = {{16000, 256}, {16000, 512}, {48000, 1024}, {8000, 256}};
    size_t gi;
    int res, nr, cng, legacy, syn, combos = 0, bad = 0, nulls = 0, ran = 0;
    for (gi = 0; gi < sizeof grid / sizeof grid[0]; ++gi) {
        int idx = 0;
        for (res = 0; res < 3; ++res) for (nr = 0; nr < 2; ++nr) for (cng = 0; cng < 2; ++cng)
        for (legacy = 0; legacy < 2; ++legacy) for (syn = 0; syn < 2; ++syn) {
            SeSampleChainConfig cfg = se_sample_chain_default_config(grid[gi].sr, grid[gi].fft);
            ChainRun r;
            cfg.res = (SeSampleRes)res;
            cfg.res_preset = (AecPreset)(idx % 4);
            cfg.enable_nr = nr;
            cfg.nr_mode = (AnrMode)((idx / 4) % 4);
            cfg.legacy_amin = legacy;
            cfg.enable_cng = cng;
            cfg.with_synthesis = syn;
            idx++;
            r = compare_chain(&cfg, SABOTAGE_NONE, 70, 1000u + (uint32_t)idx);
            combos++;
            ran += r.hops;
            if (r.mismatch_p || r.mismatch_out || r.mismatch_gain) bad++;
            if (r.null_hops) nulls++;
        }
    }
    CHECK(combos == 4 * 48, "chain: every grid x res x nr x cng x prior x synthesis combination ran");
    CHECK(ran == combos * 70, "chain: every hop of every combination produced a spectrum");
    CHECK(nulls == 0, "chain: no stage refused a scripted hop");
    CHECK(bad == 0, "chain: P, the synthesised hop and both gains equal the written-out chain bit for bit");

    /* all stages on: every RES preset x NR mode */
    {
        int preset, mode, bad_sweep = 0;
        for (preset = 0; preset < 4; ++preset) for (mode = 0; mode < 4; ++mode) {
            SeSampleChainConfig cfg = se_sample_chain_default_config(16000, 256);
            ChainRun r;
            cfg.res_preset = (AecPreset)preset;
            cfg.nr_mode = (AnrMode)mode;
            r = compare_chain(&cfg, SABOTAGE_NONE, 70, 5000u + (uint32_t)(preset * 4 + mode));
            if (r.mismatch_p || r.mismatch_out || r.mismatch_gain || r.null_hops) bad_sweep++;
            if (r.cut_hops < 60) bad_sweep++;   /* the stages really act on the frame */
        }
        CHECK(bad_sweep == 0, "chain: all stages on, every RES preset x NR mode matches and shapes the frame");
    }

    /* no stage: the frame passes through untouched */
    {
        SeSampleChainConfig cfg = se_sample_chain_default_config(16000, 256);
        ChainRun r;
        cfg.res = SE_SAMPLE_RES_OFF;
        cfg.enable_nr = 0;
        r = compare_chain(&cfg, SABOTAGE_NONE, 40, 777u);
        CHECK(r.hops == 40 && r.cut_hops == 0 && r.mismatch_p == 0 && r.mismatch_out == 0,
              "chain: RES, NR and comfort noise all off is transparent");
    }
}

/* The comparison can fail: each planted defect makes it report a mismatch. */
static void test_chain_oracle_can_fail(void) {
    static const int kinds[] = {SABOTAGE_NO_CNG, SABOTAGE_NO_USABLE_GATE, SABOTAGE_OTHER_PRESET,
                                SABOTAGE_PRIOR_FLIPPED};
    static const char* const names[] = {"comfort noise skipped", "usable_linear gate dropped",
                                        "other RES preset", "prior flag flipped"};
    size_t i;
    for (i = 0; i < sizeof kinds / sizeof kinds[0]; ++i) {
        SeSampleChainConfig cfg = se_sample_chain_default_config(16000, 256);
        ChainRun r;
        char msg[96];
        cfg.res_preset = AEC_PRESET_AGGRESSIVE;
        r = compare_chain(&cfg, kinds[i], 70, 9000u + (uint32_t)i);
        snprintf(msg, sizeof msg, "oracle control: %s is caught", names[i]);
        CHECK(r.hops > 0 && (r.mismatch_p > 0 || r.mismatch_out > 0), msg);
    }
}

/* The chain's denoiser takes the library's strength setter: switched before the first hop it
 * equals a denoiser built in the target mode, on every grid; its window is the oracle's. */
static void test_chain_nr_mode(void) {
    static const struct { int sr, fft; } grid[] = {{16000, 256}, {16000, 512}, {48000, 1024}, {8000, 256}};
    static ScriptFrame sf;
    size_t gi;
    int m, h, bad = 0, refused = 0, size_off = 0, window_off = 0, cut = 0;
    for (gi = 0; gi < sizeof grid / sizeof grid[0]; ++gi) {
        for (m = 0; m < 4; ++m) {
            SeSampleChainConfig cfg = se_sample_chain_default_config(grid[gi].sr, grid[gi].fft);
            SeSampleChainConfig direct;
            SeSampleChainSizes sz;
            AnrConfig oc = oracle_nr_config(grid[gi].sr, grid[gi].fft, (AnrMode)m);
            size_t bytes;
            void* pool;
            SeSampleChain* c;
            Oracle o;
            cfg.res = SE_SAMPLE_RES_FRAME_GAIN;
            cfg.nr_mode = m == ANR_MODE_BALANCED ? ANR_MODE_MILD : ANR_MODE_BALANCED;
            direct = cfg;
            direct.nr_mode = (AnrMode)m;
            bytes = se_sample_chain_mem_size(&cfg);
            pool = aligned_pool(bytes);
            c = pool ? se_sample_chain_init(pool, bytes, &cfg) : NULL;
            if (!c || oracle_init(&o, &direct, SABOTAGE_NONE) != 0) { refused++; free(pool); continue; }
            if (se_sample_chain_sizes(&cfg, &sz) != 0 || sz.nr_bytes != anr_get_mem_size(&oc)) size_off++;
            if (anr_set_mode(se_sample_chain_nr(c), (AnrMode)m) != 0) refused++;
            g_lcg = 600u + (uint32_t)(gi * 4 + (size_t)m);
            for (h = 0; h < 70; ++h) {
                const Complex* p;
                script_hop(&sf, c->n_freqs, c->hop, h);
                p = se_sample_chain_form(c, &sf.f);
                if (!p || oracle_hop(&o, &sf.f, NULL) != 0 ||
                    memcmp(p, o.spec, (size_t)c->n_freqs * sizeof(Complex)) != 0 ||
                    memcmp(se_sample_chain_last_nr_gain(c), anr_get_gain(o.nr, NULL),
                           (size_t)c->n_freqs * sizeof(float)) != 0) bad++;
                else if (memcmp(p, sf.e, (size_t)c->n_freqs * sizeof(Complex)) != 0) cut++;
            }
            /* the oracle's written-out window is accepted as a target, so the windows agree */
            if (anr_reconfigure(se_sample_chain_nr(c), &oc) != 0) window_off++;
            oc.L += 1;
            if (anr_reconfigure(se_sample_chain_nr(c), &oc) != -1) window_off++;
            se_sample_chain_destroy(c);
            oracle_free(&o);
            free(pool);
        }
    }
    CHECK(refused == 0, "nr mode: anr_set_mode accepts every mode on the chain's denoiser");
    CHECK(size_off == 0, "nr mode: the chain sizes its denoiser like the oracle's window");
    CHECK(window_off == 0, "nr mode: the oracle's window is accepted, one hop more is refused");
    CHECK(bad == 0, "nr mode: switched before the first hop equals a denoiser built in that mode");
    CHECK(cut > 0, "nr mode: the denoiser shapes the frame");
}

/* Reset restores the state of a fresh chain; a stale pool and a zeroed pool initialise alike;
 * setters on a running chain act like the same library calls on the written-out chain. */
static void test_chain_reset_pool_setters(void) {
    SeSampleChainConfig cfg = se_sample_chain_default_config(16000, 256);
    size_t bytes = se_sample_chain_mem_size(&cfg);
    void *pa = aligned_pool(bytes), *pb = aligned_pool(bytes), *pc = aligned_pool(bytes);
    SeSampleChain *a, *b, *c;
    static ScriptFrame sf;
    static float oa[MAXN], ob[MAXN];
    int h, bad_reset = 0, bad_pool = 0;

    if (pc) memset(pc, 0, bytes);
    a = pa ? se_sample_chain_init(pa, bytes, &cfg) : NULL;
    b = pb ? se_sample_chain_init(pb, bytes, &cfg) : NULL;   /* stale bytes (0xCD) */
    c = pc ? se_sample_chain_init(pc, bytes, &cfg) : NULL;   /* zeroed */
    CHECK(a && b && c, "chain: init on stale and zeroed pools");
    if (!a || !b || !c) { free(pa); free(pb); free(pc); return; }

    /* stale vs zeroed pool: identical from the first hop */
    g_lcg = 321u;
    for (h = 0; h < 40; ++h) {
        const Complex *pbv, *pcv;
        script_hop(&sf, b->n_freqs, b->hop, h);
        pbv = se_sample_chain_form(b, &sf.f);
        pcv = se_sample_chain_form(c, &sf.f);
        if (!pbv || !pcv || memcmp(pbv, pcv, (size_t)b->n_freqs * sizeof(Complex)) != 0) bad_pool++;
        if (pbv && pcv) {
            (void)se_sample_chain_synthesize(b, pbv, oa);
            (void)se_sample_chain_synthesize(c, pcv, ob);
            if (memcmp(oa, ob, (size_t)b->hop * sizeof(float)) != 0) bad_pool++;
        }
    }
    CHECK(bad_pool == 0, "chain: a stale pool initialises like a zeroed one");

    /* reset == fresh: run a, reset it (after a preset change that must survive), replay
     * against a chain that never ran */
    se_sample_chain_destroy(b);
    se_sample_chain_destroy(c);
    memset(pb, 0xCD, bytes);
    b = se_sample_chain_init(pb, bytes, &cfg);
    g_lcg = 99u;
    for (h = 0; h < 50; ++h) {
        script_hop(&sf, a->n_freqs, a->hop, h);
        { const Complex* p = se_sample_chain_form(a, &sf.f); if (p) (void)se_sample_chain_synthesize(a, p, oa); }
    }
    CHECK(aec_res_set_preset(se_sample_chain_res(a), AEC_PRESET_AGGRESSIVE, 0.0f) == 0 &&
          aec_res_set_preset(se_sample_chain_res(b), AEC_PRESET_AGGRESSIVE, 0.0f) == 0,
          "chain: RES preset change accepted");
    se_sample_chain_reset(a);
    CHECK(se_sample_chain_last_res_gain(a) == NULL && se_sample_chain_last_nr_gain(a) == NULL,
          "chain: a reset clears the last-gain views");
    g_lcg = 4242u;
    for (h = 0; h < 60; ++h) {
        const Complex *pa2, *pb2;
        script_hop(&sf, a->n_freqs, a->hop, h);
        pa2 = se_sample_chain_form(a, &sf.f);
        pb2 = se_sample_chain_form(b, &sf.f);
        if (!pa2 || !pb2 || memcmp(pa2, pb2, (size_t)a->n_freqs * sizeof(Complex)) != 0) bad_reset++;
        if (pa2 && pb2) {
            (void)se_sample_chain_synthesize(a, pa2, oa);
            (void)se_sample_chain_synthesize(b, pb2, ob);
            if (memcmp(oa, ob, (size_t)a->hop * sizeof(float)) != 0) bad_reset++;
        }
    }
    CHECK(bad_reset == 0, "chain: reset equals a fresh instance (RES/NR/synthesis history and comfort-noise seed)");
    se_sample_chain_destroy(a);
    se_sample_chain_destroy(b);
    free(pa);
    free(pb);
    free(pc);

    /* mid-run setters against the same calls on the written-out chain */
    {
        Oracle o;
        void* pool = aligned_pool(bytes);
        SeSampleChain* s = pool ? se_sample_chain_init(pool, bytes, &cfg) : NULL;
        int bad_set = 0, rc_set = 0, rc_bad_a, rc_bad_b, rc_bad_c;
        float lo_s, lo_o, hi_s, hi_o;
        CHECK(s && oracle_init(&o, &cfg, SABOTAGE_NONE) == 0, "chain: setter test setup");
        if (s) {
            g_lcg = 8080u;
            for (h = 0; h < 120; ++h) {
                const Complex* p;
                script_hop(&sf, s->n_freqs, s->hop, h);
                if (h == 30) {
                    AnrConfig target = oracle_nr_config(16000, 256, ANR_MODE_AGGRESSIVE);
                    rc_set |= anr_set_mode(se_sample_chain_nr(s), ANR_MODE_AGGRESSIVE);
                    rc_set |= anr_reconfigure(o.nr, &target);
                }
                if (h == 60) {
                    rc_set |= aec_res_set_preset(se_sample_chain_res(s), AEC_PRESET_MILD, 100.0f);
                    rc_set |= aec_res_set_preset(o.res, AEC_PRESET_MILD, 100.0f);
                }
                if (h == 90) {   /* refused changes leave everything as it was */
                    rc_bad_a = anr_set_mode(se_sample_chain_nr(s), (AnrMode)7);
                    rc_bad_b = aec_res_set_preset(se_sample_chain_res(s), (AecPreset)9, 0.0f);
                    rc_bad_c = anr_set_mode(se_sample_chain_nr(NULL), ANR_MODE_MILD);
                    if (rc_bad_a != -1 || rc_bad_b != -1 || rc_bad_c != -1) bad_set++;
                }
                p = se_sample_chain_form(s, &sf.f);
                if (!p || oracle_hop(&o, &sf.f, NULL) != 0 ||
                    memcmp(p, o.spec, (size_t)s->n_freqs * sizeof(Complex)) != 0) bad_set++;
            }
            CHECK(rc_set == 0, "chain: strength setters accepted");
            CHECK(bad_set == 0, "chain: setters mid-run match the written-out chain, refusals change nothing");
            CHECK(aec_res_split_floor(se_sample_chain_res(s), &lo_s, &hi_s) == 0 &&
                  aec_res_split_floor(o.res, &lo_o, &hi_o) == 0 && lo_s == lo_o && hi_s == hi_o,
                  "chain: the RES floor reads back as on the written-out chain");
            {
                SeSampleChainConfig nonr = cfg;
                nonr.enable_nr = 0;
                bytes = se_sample_chain_mem_size(&nonr);
                {
                    void* p2 = aligned_pool(bytes);
                    SeSampleChain* s2 = p2 ? se_sample_chain_init(p2, bytes, &nonr) : NULL;
                    CHECK(s2 && se_sample_chain_nr(s2) == NULL &&
                          anr_set_mode(se_sample_chain_nr(s2), ANR_MODE_MILD) == -1,
                          "chain: no NR stage, no strength change");
                    free(p2);
                }
            }
            oracle_free(&o);
            se_sample_chain_destroy(s);
        }
        free(pool);
    }
}

static void test_chain_sizes_and_rejects(void) {
    const int sr = 16000, fft = 256;
    SeSampleChainConfig on = se_sample_chain_default_config(sr, fft);
    SeSampleChainConfig c;
    size_t full = se_sample_chain_mem_size(&on);
    AnrConfig nc = oracle_nr_config(sr, fft, ANR_MODE_BALANCED);
    AecConfig rc;
    AcStftConfig sc;
    SeSpecFrame f;
    static ScriptFrame sf;
    void* pool;
    SeSampleChain* ch;

    CHECK(full > 0, "sizes: default configuration has a size");
    c = on;
    c.enable_nr = 0;
    CHECK(full - se_sample_chain_mem_size(&c) == ALIGN16(anr_get_mem_size(&nc)),
          "sizes: NR off saves exactly one aligned denoiser");
    c = on;
    c.res = SE_SAMPLE_RES_FRAME_GAIN;
    aec_config_from_preset(&rc, AEC_PRESET_BALANCED, sr);
    rc.fft_size = fft;
    CHECK(full - se_sample_chain_mem_size(&c) == ALIGN16(aec_res_get_mem_size(&rc)),
          "sizes: no RES stage saves exactly one aligned AecRes");
    c = on;
    c.with_synthesis = 0;
    ac_stft_config_default(&sc, fft);
    CHECK(full - se_sample_chain_mem_size(&c) == ALIGN16(ac_istft_get_mem_size(&sc)),
          "sizes: no synthesis saves exactly one aligned AcIstft");

    c = on; c.sample_rate = 44100;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: 44.1 kHz");
    c = on; c.fft_size = 512; c.sample_rate = 48000;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: 48 kHz with fft 512");
    c = on; c.res = (SeSampleRes)3;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: RES mode out of range");
    c = on; c.res_preset = (AecPreset)4;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: RES preset out of range");
    c = on; c.res = SE_SAMPLE_RES_FRAME_GAIN; c.res_preset = (AecPreset)4;
    CHECK(se_sample_chain_mem_size(&c) > 0, "accepts: RES preset is not read without AecRes");
    c = on; c.nr_mode = (AnrMode)4;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: NR mode out of range");
    c = on; c.enable_nr = 0; c.nr_mode = (AnrMode)4;
    CHECK(se_sample_chain_mem_size(&c) > 0, "accepts: NR mode is not read without NR");
    c = on; c.enable_cng = 2;
    CHECK(se_sample_chain_mem_size(&c) == 0, "rejects: flag outside {0,1}");
    CHECK(se_sample_chain_mem_size(NULL) == 0, "rejects: NULL configuration");

    pool = aligned_pool(full + 16);
    CHECK(pool && se_sample_chain_init(pool, full - 16, &on) == NULL, "init: short pool refused");
    CHECK(pool && se_sample_chain_init((uint8_t*)pool + 8, full, &on) == NULL, "init: misaligned pool refused");
    CHECK(se_sample_chain_init(NULL, full, &on) == NULL, "init: NULL pool refused");
    ch = pool ? se_sample_chain_init(pool, full, &on) : NULL;
    CHECK(ch != NULL, "init: exact-size pool");
    if (ch) {
        static float out[MAXN];
        script_hop(&sf, ch->n_freqs, ch->hop, 0);
        CHECK(se_sample_chain_form(NULL, &sf.f) == NULL && se_sample_chain_form(ch, NULL) == NULL,
              "form: NULL arguments");
        f = sf.f;
        f.n_freqs = 257;
        CHECK(se_sample_chain_form(ch, &f) == NULL, "form: a frame of another grid");
        f = sf.f;
        f.e = NULL;
        CHECK(se_sample_chain_form(ch, &f) == NULL, "form: a frame without the error spectrum");
        f = sf.f;
        f.cn = NULL;
        CHECK(se_sample_chain_form(ch, &f) == NULL, "form: AecRes needs the comfort-noise spectrum");
        CHECK(se_sample_chain_synthesize(ch, NULL, out) == -1 && se_sample_chain_synthesize(NULL, ch->spec, out) == -1 &&
              se_sample_chain_synthesize(ch, ch->spec, NULL) == -1, "synthesize: NULL arguments");
        CHECK(se_hop_is_finite(sf.far_hop, ch->hop), "finite helper accepts finite samples");
        sf.far_hop[3] = NAN;
        CHECK(!se_hop_is_finite(sf.far_hop, ch->hop), "finite helper flags a NaN");
        se_sample_chain_destroy(ch);
    }
    free(pool);

    c = on;
    c.res = SE_SAMPLE_RES_FRAME_GAIN;
    c.with_synthesis = 0;
    pool = aligned_pool(se_sample_chain_mem_size(&c));
    ch = pool ? se_sample_chain_init(pool, se_sample_chain_mem_size(&c), &c) : NULL;
    if (ch) {
        static float out[MAXN];
        script_hop(&sf, ch->n_freqs, ch->hop, 0);
        f = sf.f;
        f.g_res_ctx = NULL;
        CHECK(se_sample_chain_form(ch, &f) == NULL, "form: FRAME_GAIN needs the front end's gain");
        CHECK(se_sample_chain_form(ch, &sf.f) != NULL && se_sample_chain_synthesize(ch, ch->spec, out) == -1,
              "synthesize: refused without a synthesis stage");
        CHECK(se_sample_chain_res(ch) == NULL, "no AecRes in FRAME_GAIN mode");
        se_sample_chain_destroy(ch);
    }
    free(pool);
}


int main(void) {
    test_grid();
    test_derive_fields();
    test_derive_rejects();
    test_nr_config();
    test_stage();
    test_psd_scale();
    test_frame_from_aec();
    test_chain_equals_oracle();
    test_chain_oracle_can_fail();
    test_chain_nr_mode();
    test_chain_reset_pool_setters();
    test_chain_sizes_and_rejects();
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
