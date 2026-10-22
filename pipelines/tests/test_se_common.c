/* Acceptance test for pipelines/common: the shared grid and AecConfig
 * derivation (se_config) and the noise-reduction stage (se_anr_stage). The
 * expected values are written out by hand, not recomputed with the code under
 * test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/se_anr_stage.h"
#include "common/se_config.h"

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
    r.nr_mode = ANR_MODE_BALANCED;
    return r;
}

static SeRules mono_rules(void) {
    SeRules u = {1, 0, 0, 0, 0, SE_CONTEXT_PRESET, 1};
    return u;
}

static SeRules four_channel_rules(int post) {
    SeRules u = {0, 0, 0, 1, 1, post ? AEC_CONTEXT_NO_GAIN : AEC_CONTEXT_LINEAR_ONLY, 1};
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
    {   /* compiled grid, linear-only seam, no NR */
        SeRules u = {0, 16000, 512, 0, 0, AEC_CONTEXT_LINEAR_ONLY, 0};
        SeRequest q = base_request(16000);
        q.nr_mode = (AnrMode)99;   /* ignored without NR */
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
    r = base_request(16000);
    r.nr_mode = (AnrMode)4;
    CHECK(se_derive(&r, &mono, &g, &a) == -1, "nr mode 4 with NR");
    mono.with_nr = 0;
    CHECK(se_derive(&r, &mono, &g, &a) == 0, "nr mode ignored without NR");
    mono = mono_rules();

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

static void test_compose(void) {
    static const struct { int sr, fft; } grid[] = {{8000, 256}, {16000, 256}, {16000, 512}, {48000, 1024}};
    size_t i;
    int m;
    for (i = 0; i < sizeof grid / sizeof grid[0]; ++i) {
        for (m = 0; m < 4; ++m) {
            int hop = grid[i].fft / 2;
            AnrConfig canon = anr_config_for_mode_grid(grid[i].sr, grid[i].fft, (AnrMode)m);
            AnrConfig c = se_anr_compose_config(grid[i].sr, grid[i].fft, hop, (AnrMode)m);
            AnrConfig expect = canon;
            /* 150 frames of 10 ms, rounded up to whole hops */
            int want_l = (int)((150LL * 10 * grid[i].sr / 1000 + hop - 1) / hop);
            CHECK(c.L == want_l, "L is 150 frames of 10 ms retimed");
            CHECK(c.L != canon.L, "the overlay changes L (anr_set_mode cannot be used)");
            CHECK(c.alpha_decay == canon.alpha_g, "alpha_decay follows alpha_g");
            CHECK(c.broadband_threshold == canon.broadband_threshold && c.broadband_threshold == 1.0f,
                  "scene-reset gate stays off");
            expect.L = c.L;
            expect.alpha_decay = canon.alpha_g;
            CHECK(memcmp(&expect, &c, sizeof c) == 0, "only L and alpha_decay differ from the canonical preset");
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
    const int sr = 16000, fft = 256, hop = 128, n = 129;
    AnrConfig cfg = se_anr_compose_config(sr, fft, hop, ANR_MODE_BALANCED);
    size_t bytes = se_anr_stage_mem_size(&cfg);
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
    CHECK(se_anr_stage_mem_size(&(AnrConfig){0}) == 0, "invalid config has no size");
    a = se_anr_stage_init(pool, bytes, &cfg);
    b = se_anr_stage_init(pool2, bytes, &cfg);
    c = se_anr_stage_init(pool3, bytes, &cfg);
    CHECK(a && b && c, "stage init");
    CHECK(se_anr_stage_init(pool, bytes - 16, &cfg) == NULL, "short pool refused");

    /* the canonical setter refuses this instance, the stage's setter does not */
    for (m = 0; m < 4; ++m) {
        CHECK(anr_set_mode(a, (AnrMode)m) == -1, "anr_set_mode refuses the overlay instance");
        CHECK(se_anr_stage_set_mode(a, sr, fft, hop, (AnrMode)m) == 0, "stage set_mode accepts every mode");
    }
    CHECK(se_anr_stage_set_mode(a, sr, fft, hop, (AnrMode)-1) == -1, "mode -1 refused");
    CHECK(se_anr_stage_set_mode(a, sr, fft, hop, (AnrMode)4) == -1, "mode 4 refused");
    CHECK(se_anr_stage_set_mode(NULL, sr, fft, hop, ANR_MODE_MILD) == -1, "NULL denoiser refused");
    CHECK(se_anr_stage_set_mode(a, sr, 512, 256, ANR_MODE_MILD) == -1, "another grid refused");
    CHECK(se_anr_stage_set_mode(a, 48000, 1024, 512, ANR_MODE_MILD) == -1, "another rate refused");
    CHECK(se_anr_stage_set_mode(a, sr, fft, hop, ANR_MODE_BALANCED) == 0, "back to balanced");

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
    /* usable_linear = 0 fails closed: r2 and scratch are both non-NULL, but the
     * cold-start-unusable estimate must not fold into the noise floor -- the
     * gain must equal the plain (use_r2 = 0) call, never the r2-folded one.
     * r2 here is sized like the real cold-start fault this gate exists for
     * (on the order of 1e12, matching a field trace of the AEC's conservative
     * nonlinear estimate during cold start). The mutation guard below (denoiser
     * c) proves that comparison isn't vacuous. */
    se_anr_stage_reset(a);
    anr_reset(b);
    anr_reset(c);
    f.usable_linear = 0;
    /* The denoiser bypasses NR for its first num_init_frames (here it takes
     * ~25 hops for gb's own gain to move off unity), so the mutation guard
     * below -- folding this r2 must visibly differ from not folding it --
     * is only checked once that warm-up has passed; the primary ga == gb
     * gate check runs every hop regardless. */
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
    se_anr_stage_reset(a);
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

    se_anr_stage_destroy(a);
    se_anr_stage_destroy(b);
    se_anr_stage_destroy(c);
    free(pool);
    free(pool2);
    free(pool3);
}

/* The prior scale is the exact reciprocal multiply of the division, for every
 * kind of float: random bit patterns, tiny values whose scaled result is
 * subnormal, zeros, infinities and NaN. The divisor goes through a volatile
 * so the reference stays a real division. */
static void test_psd_scale(void) {
    const int sr = 16000, fft = 256, hop = 128, n = 129;
    AnrConfig cfg = se_anr_compose_config(sr, fft, hop, ANR_MODE_BALANCED);
    size_t bytes = se_anr_stage_mem_size(&cfg);
    void *pool = aligned_pool(bytes);
    AnrDenoiser *a = pool ? se_anr_stage_init(pool, bytes, &cfg) : NULL;
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
        se_anr_stage_reset(a);
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
        se_anr_stage_reset(a);
    }
    CHECK(ok_len, "psd scale: every length 1..129 converts exactly and stops at n_freqs");
    se_anr_stage_destroy(a);
    free(pool);
}

int main(void) {
    test_grid();
    test_derive_fields();
    test_derive_rejects();
    test_compose();
    test_stage();
    test_psd_scale();
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
