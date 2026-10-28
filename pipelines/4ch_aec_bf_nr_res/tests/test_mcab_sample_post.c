/* Acceptance test of 4ch_aec_bf_nr_res/mcab_sample_post.h, the glue between mcab_fuse() and the
 * libraries behind it: the libaec RES input of a fused frame, the denoiser call with the
 * residual-echo fold, the finite-hop check the programs run on an emitted hop, and the NR
 * configuration the oracles use. Expected values are written out by hand, not recomputed with
 * the code under test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anr.h"
#include "aec_res.h"
#include "4ch_aec_bf_nr_res/mcab_sample_post.h"
#include "utility/simd_kernel_nn.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

/* The denoiser configuration the oracles use: the library level for `level` with the 512 ms
 * minimum window written out in whole hops rather than taken from the level. */
static AnrConfig oracle_nr_config(int sr, int fft, AnrLevel level) {
    AnrConfig c = anr_config_for_level_grid(sr, fft, level);
    int hop = fft / 2;
    c.L = (512 * sr + 1000 * hop - 1) / (1000 * hop);
    return c;
}

/* The examples' denoiser is the library level; the scene-reset gate stays off, which keeps the
 * gain independent of the window length. */
static void test_nr_config(void) {
    static const struct { int sr, fft, l; } grid[] = {
        {8000, 256, 32}, {16000, 256, 64}, {16000, 512, 32}, {48000, 1024, 48}};
    size_t i;
    int m;
    for (i = 0; i < sizeof grid / sizeof grid[0]; ++i) {
        for (m = 0; m < 4; ++m) {
            AnrConfig canon = anr_config_for_level_grid(grid[i].sr, grid[i].fft, (AnrLevel)m);
            AnrConfig o = oracle_nr_config(grid[i].sr, grid[i].fft, (AnrLevel)m);
            CHECK(canon.L == grid[i].l, "the level's minimum window is 512 ms in whole hops");
            CHECK(canon.alpha_decay == canon.alpha_g, "the level's alpha_decay equals alpha_g");
            CHECK(canon.broadband_threshold == 1.0f, "scene-reset gate stays off");
            CHECK(memcmp(&o, &canon, sizeof o) == 0, "the oracle configuration equals the level");
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
    AnrConfig cfg = anr_config_for_level_grid(sr, fft, ANR_LEVEL_BALANCED);
    size_t bytes = anr_get_mem_size(&cfg);
    void *pool = aligned_pool(bytes), *pool2 = aligned_pool(bytes), *pool3 = aligned_pool(bytes);
    AnrDenoiser *a, *b, *c;
    Complex spec[129];
    float r2[129], scratch[129], extra[129];
    const float *ga, *gb;
    McabFrame f;
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
        CHECK(anr_set_level(a, (AnrLevel)m) == 0, "anr_set_level accepts every level on the instance");
    CHECK(anr_set_level(a, ANR_LEVEL_BALANCED) == 0, "back to balanced");

    /* the residual-echo prior reaches the denoiser as r2 / 32768^2 */
    for (frame = 0; frame < 40; ++frame) {
        for (k = 0; k < n; ++k) {
            spec[k].r = (float)((frame * 7 + k * 13) % 101) * 31.0f - 1500.0f;
            spec[k].i = (float)((frame * 5 + k * 3) % 89) * 17.0f - 700.0f;
            r2[k] = (float)((k + 1) * (frame + 3)) * 4.0e5f;
            extra[k] = r2[k] / (32768.0f * 32768.0f);
        }
        ga = mcab_sample_anr_gain(a, &f, scratch, 1);
        CHECK(ga != NULL, "stage process");
        CHECK(anr_process_gain(b, spec, extra, NULL) == 0, "direct process");
        gb = anr_get_gain(b, NULL);
        CHECK(ga && gb && memcmp(ga, gb, (size_t)n * sizeof(float)) == 0, "stage gain equals the direct call");
        CHECK(memcmp(scratch, extra, (size_t)n * sizeof(float)) == 0, "scratch holds r2 / 32768^2");
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
        ga = mcab_sample_anr_gain(a, &f, scratch, 1);
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
        ga = mcab_sample_anr_gain(a, &f, scratch, 0);
        CHECK(ga != NULL, "no prior");
        CHECK(anr_process_gain(b, spec, NULL, NULL) == 0, "direct no prior");
        CHECK(memcmp(ga, anr_get_gain(b, NULL), (size_t)n * sizeof(float)) == 0, "no-prior gain");
        CHECK(mcab_sample_anr_gain(a, &f, NULL, 1) != NULL, "no scratch selects plain NR");
    }
    CHECK(mcab_sample_anr_gain(NULL, &f, NULL, 0) == NULL, "NULL denoiser");
    CHECK(mcab_sample_anr_gain(a, NULL, NULL, 0) == NULL, "NULL frame");
    f.e = NULL;
    CHECK(mcab_sample_anr_gain(a, &f, NULL, 0) == NULL, "NULL spectrum");

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
    AnrConfig cfg = anr_config_for_level_grid(sr, fft, ANR_LEVEL_BALANCED);
    size_t bytes = anr_get_mem_size(&cfg);
    void *pool = aligned_pool(bytes);
    AnrDenoiser *a = pool ? anr_init(pool, bytes, &cfg) : NULL;
    volatile float divisor = 32768.0f * 32768.0f;
    Complex spec[129];
    float r2[129], scratch[129];
    McabFrame f;
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
        (void)mcab_sample_anr_gain(a, &f, scratch, 1);
        for (k = 0; k < n; ++k) {
            float want = r2[k] / divisor;
            uint32_t gb, wb;
            memcpy(&gb, &scratch[k], sizeof gb);
            memcpy(&wb, &want, sizeof wb);
            if (gb != wb && !(isnan(scratch[k]) && isnan(want))) ok = 0;
        }
        anr_reset(a);
    }
    CHECK(ok, "psd scale: scratch equals r2 / 32768^2 for every float class");

    /* every length around the vector widths */
    for (k = 0; k < n; ++k) r2[k] = (float)(k + 1) * 3.0e5f;
    for (frame = 1; frame <= n && ok_len; ++frame) {
        float guard = 1234.5f;
        scratch[frame < n ? frame : n - 1] = guard;
        f.n_freqs = frame;
        (void)mcab_sample_anr_gain(a, &f, scratch, 1);
        for (k = 0; k < frame; ++k)
            if (scratch[k] != r2[k] / divisor) ok_len = 0;
        if (frame < n && scratch[frame] != guard) ok_len = 0;
        anr_reset(a);
    }
    CHECK(ok_len, "psd scale: every length 1..129 converts exactly and stops at n_freqs");
    anr_destroy(a);
    free(pool);
}

/* mcab_sample_res_input: every AecResInput field from its McabFrame field; a NULL frame gives the
 * all-zero input aec_res_process() refuses. */
static void test_res_input(void) {
    static Complex e[129], y[129];
    static float r2[129], cn[129], far[128];
    McabFrame f;
    AecResInput in;
    AecConfig cfg;
    size_t bytes;
    void* pool;
    AecRes* res;

    memset(&f, 0, sizeof f);
    f.n_freqs = 129;
    f.hop = 128;
    f.e = e;
    f.y = y;
    f.r2 = r2;
    f.cn = cn;
    f.far_hop = far;
    f.all_converged = 1;
    f.max_dt = 0.25f;
    f.max_sat = 0.75f;
    f.clock_drift = 1;
    memset(&in, 0x5A, sizeof in);
    mcab_sample_res_input(&f, &in);
    CHECK(in.error_spec == e && in.near_spec == y && in.r2 == r2 && in.comfort_noise == cn &&
          in.render_hop == far, "res input: the spectra and the render hop alias the frame");
    CHECK(in.all_converged == 1 && in.max_dt == 0.25f && in.max_saturation == 0.75f &&
          in.clock_drift == 1, "res input: the scalars are the frame's");
    memset(&in, 0x5A, sizeof in);
    mcab_sample_res_input(NULL, &in);
    CHECK(!in.error_spec && !in.near_spec && !in.r2 && !in.comfort_noise && !in.render_hop &&
          in.all_converged == 0 && in.max_dt == 0.0f && in.max_saturation == 0.0f &&
          in.clock_drift == 0, "res input: a NULL frame zeroes the input");
    mcab_sample_res_input(&f, NULL);   /* a NULL input is a no-op */

    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.fft_size = 256;
    bytes = aec_res_get_mem_size(&cfg);
    pool = bytes ? aligned_pool(bytes) : NULL;
    res = pool ? aec_res_init(pool, bytes, &cfg) : NULL;
    CHECK(res != NULL, "res input: suppressor init");
    if (res) {
        CHECK(aec_res_process(res, &in) == NULL, "res input: the zeroed input is refused");
        mcab_sample_res_input(&f, &in);
        CHECK(aec_res_process(res, &in) != NULL, "res input: the frame's input is accepted");
    }
    free(pool);
}

/* NR strength at runtime: anr_set_level() before the first hop makes a denoiser equal, through
 * mcab_sample_anr_gain, to one built at the target level, on every grid the examples use. */
static void test_nr_set_level(void) {
    static const struct { int sr, fft; } grid[] = {{16000, 256}, {16000, 512}, {48000, 1024}, {8000, 256}};
    static Complex spec[513];
    static float r2[513], s_a[513], s_b[513];
    size_t gi;
    int m, h, k, bad = 0, refused = 0, shaped = 0;
    for (gi = 0; gi < sizeof grid / sizeof grid[0]; ++gi) {
        int n = grid[gi].fft / 2 + 1;
        for (m = 0; m < 4; ++m) {
            AnrLevel other = m == ANR_LEVEL_BALANCED ? ANR_LEVEL_MILD : ANR_LEVEL_BALANCED;
            AnrConfig ca = anr_config_for_level_grid(grid[gi].sr, grid[gi].fft, other);
            AnrConfig cb = anr_config_for_level_grid(grid[gi].sr, grid[gi].fft, (AnrLevel)m);
            size_t ba = anr_get_mem_size(&ca), bb = anr_get_mem_size(&cb);
            void *pa = aligned_pool(ba), *pb = aligned_pool(bb);
            AnrDenoiser* a = pa ? anr_init(pa, ba, &ca) : NULL;
            AnrDenoiser* b = pb ? anr_init(pb, bb, &cb) : NULL;
            McabFrame f;
            memset(&f, 0, sizeof f);
            f.n_freqs = n;
            f.e = spec;
            f.r2 = r2;
            f.usable_linear = 1;
            if (!a || !b || anr_set_level(a, (AnrLevel)m) != 0) { refused++; free(pa); free(pb); continue; }
            for (h = 0; h < 70; ++h) {
                const float *ga, *gb;
                for (k = 0; k < n; ++k) {
                    spec[k].r = (float)((h * 7 + k * 13) % 101) * 3.1e-4f - 0.015f;
                    spec[k].i = (float)((h * 5 + k * 3) % 89) * 1.7e-4f - 0.007f;
                    r2[k] = (float)((k + 1) * (h % 11 + 3)) * 4.0e3f;
                }
                ga = mcab_sample_anr_gain(a, &f, s_a, 1);
                gb = mcab_sample_anr_gain(b, &f, s_b, 1);
                if (!ga || !gb || memcmp(ga, gb, (size_t)n * sizeof(float)) != 0) bad++;
                else for (k = 0; k < n; ++k) if (ga[k] < 0.999f) { shaped = 1; break; }
            }
            anr_destroy(a);
            anr_destroy(b);
            free(pa);
            free(pb);
        }
    }
    CHECK(refused == 0, "nr level: anr_set_level accepts every level on a denoiser");
    CHECK(bad == 0, "nr level: switched before the first hop equals a denoiser built at that level");
    CHECK(shaped, "nr level: the denoiser shapes the scripted frames");
}

/* skn_all_finite_f32 (the check the programs run on an emitted hop): every non-finite class is
 * caught wherever it sits, n bounds the scan. */
static void test_hop_is_finite(void) {
    static const uint32_t bad[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u};
    float x[16];
    size_t b;
    int k, pos, ok = 1;
    for (k = 0; k < 16; ++k) x[k] = (float)k * 0.25f - 2.0f;
    x[3] = 3.4028235e38f;
    x[4] = 1.0e-45f;   /* denormal: finite */
    CHECK(skn_all_finite_f32(x, (size_t)16),
          "finite helper accepts finite samples, extremes included");
    CHECK(skn_all_finite_f32(x, (size_t)0), "finite helper: an empty hop is finite");
    for (b = 0; b < sizeof bad / sizeof bad[0]; ++b) {
        for (pos = 0; pos < 16; ++pos) {
            float keep = x[pos];
            memcpy(&x[pos], &bad[b], sizeof bad[b]);
            if (skn_all_finite_f32(x, (size_t)16) || !skn_all_finite_f32(x, (size_t)pos)) ok = 0;
            x[pos] = keep;
        }
    }
    CHECK(ok, "finite helper flags Inf and NaN at every position and stops at n");
}

int main(void) {
    test_nr_config();
    test_stage();
    test_psd_scale();
    test_res_input();
    test_nr_set_level();
    test_hop_is_finite();
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
