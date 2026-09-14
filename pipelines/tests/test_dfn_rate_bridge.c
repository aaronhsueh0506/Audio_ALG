/* dfn_rate_bridge acceptance test (no AEC): the count schedule and its
 * prefill, the added-delay constant pinned by an impulse, the tolerance
 * identity through the resampler round trip, the estimate/apply split, and
 * reset parity, on every supported native grid. Spectra are produced by the
 * same sqrt-Hann 50%-overlap analysis the hosting pipelines perform, so the
 * bridge sees exactly the frames it will see in product. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_rate_bridge.h"
#include "fft_wrapper.h"
#include "tests/dfn_test_fixture.h"

#define MAX_FFT 512
#define MAX_HOPS 200

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static float g_signal[MAX_HOPS * (MAX_FFT / 2)];
static float g_out[MAX_HOPS * (MAX_FFT / 2)];
static float g_out_b[MAX_HOPS * (MAX_FFT / 2)];

typedef struct Grid { int sr; int fft; } Grid;
static const Grid GRIDS[] = { {16000, 256}, {16000, 512}, {8000, 256} };
#define N_GRIDS 3

/* Host-side analysis: frame = [previous hop, this hop] * sqrt-Hann. */
typedef struct Analyzer {
    FftHandle *fft;
    int fft_n, hop;
    float win[MAX_FFT];
    float frame[MAX_FFT];
    float prev[MAX_FFT / 2];
} Analyzer;

static int analyzer_init(Analyzer *a, int fft_n) {
    a->fft = fft_create(fft_n);
    if (!a->fft) return -1;
    a->fft_n = fft_n;
    a->hop = fft_n / 2;
    dfn_fixture_root_hann(a->win, fft_n);
    memset(a->prev, 0, sizeof(a->prev));
    return 0;
}

static void analyzer_reset(Analyzer *a) {
    memset(a->prev, 0, sizeof(a->prev));
}

static void analyzer_push(Analyzer *a, const float *hop, Complex *spec) {
    int k;
    for (k = 0; k < a->hop; ++k) {
        a->frame[k] = a->prev[k] * a->win[k];
        a->frame[a->hop + k] = hop[k] * a->win[a->hop + k];
    }
    fft_forward(a->fft, a->frame, spec);
    memcpy(a->prev, hop, (size_t)a->hop * sizeof(float));
}

static DfnRateBridgeConfig bridge_config(int sr, int fft) {
    DfnRateBridgeConfig cfg = dfn_rate_bridge_default_config(sr);
    cfg.fft_size = fft;
    cfg.stage.erb_fwd = erb_fwd;
    cfg.stage.erb_inv = erb_inv;
    return cfg;
}

/* prefill (the schedule's largest deficit) + 16 + 1536*sr/48000 + 16. */
static int expected_added_delay(int sr, int hop) {
    int prefill = (sr == 16000 && hop == 128) ? 128
                : (sr == 16000 && hop == 256) ? 85 : 42;
    return prefill + 16 + 1536 * sr / 48000 + 16;
}

/* Run `hops` hops of estimate=est_signal, apply=app_signal (either may be
 * NULL for silence) through a fresh bridge; output into out. */
static int run_bridge(DfnRateBridge *b, Analyzer *ae, Analyzer *ap,
                      const float *est, const float *app, int hops,
                      float *out, int *schedule, int schedule_len) {
    static Complex se[MAX_FFT / 2 + 1], sa[MAX_FFT / 2 + 1];
    static float silence[MAX_FFT / 2];
    int hop = ae->hop, t, frames_last;
    long long frames_total = 0;
    memset(silence, 0, sizeof(silence));
    for (t = 0; t < hops; ++t) {
        analyzer_push(ae, est ? est + t * hop : silence, se);
        analyzer_push(ap, app ? app + t * hop : silence, sa);
        if (dfn_rate_bridge_process(b, se, sa, out + t * hop) != 0) return -1;
        dfn_rate_bridge_get_counters(b, &frames_total, &frames_last);
        if (schedule && t < schedule_len) schedule[t] = frames_last;
    }
    return (int)frames_total;
}

static int run_grid(const Grid *g) {
    DfnRateBridgeConfig cfg = bridge_config(g->sr, g->fft);
    DfnRateBridge *b = dfn_rate_bridge_create(&cfg);
    Analyzer ae, ap;
    int hop = g->fft / 2, hops = 200, n = hops * hop;
    int schedule[16];
    int t, k, delay, frames, impulse_at = 40 * hop + 7, peak = -1;
    float peak_abs = 0.0f, max_err = 0.0f, max_sig = 0.0f;
    double near_energy = 0.0, total_energy = 0.0;

    if (!b) return 1;
    if (dfn_rate_bridge_hop_size(b) != hop ||
        dfn_rate_bridge_n_freqs(b) != g->fft / 2 + 1)
        return 2;
    delay = dfn_rate_bridge_added_delay_samples(b);
    if (delay != expected_added_delay(g->sr, hop)) {
        fprintf(stderr, "  %d/%d: added delay %d, expected %d\n", g->sr, g->fft,
                delay, expected_added_delay(g->sr, hop));
        return 3;
    }

    /* Schedule and cumulative frame count (the count contract). */
    if (analyzer_init(&ae, g->fft) != 0 || analyzer_init(&ap, g->fft) != 0) return 4;
    frames = run_bridge(b, &ae, &ap, NULL, NULL, hops, g_out, schedule, 16);
    if (frames != (int)(((long long)hops * hop * 48000 / g->sr) / 512)) return 5;
    if (g->sr == 16000 && g->fft == 256) {
        for (t = 0; t < 16; ++t) if (schedule[t] != (t % 4 == 0 ? 0 : 1)) return 6;
    } else {
        for (t = 0; t < 16; ++t) if (schedule[t] != (t % 2 == 0 ? 1 : 2)) return 6;
    }
    for (k = 0; k < n; ++k) if (g_out[k] != 0.0f) return 7;   /* silence in, silence out */

    /* Impulse in P: the output peak sits exactly hop + added delay later
     * (the hop is the host-style WOLA the bridge performs on P). */
    dfn_rate_bridge_reset(b);
    analyzer_reset(&ae);
    analyzer_reset(&ap);
    memset(g_signal, 0, sizeof(g_signal));
    g_signal[impulse_at] = 1.0f;
    if (run_bridge(b, &ae, &ap, NULL, g_signal, hops, g_out, NULL, 0) < 0) return 8;
    for (k = 0; k < n; ++k) {
        float a = fabsf(g_out[k]);
        total_energy += (double)a * a;
        if (a > peak_abs) { peak_abs = a; peak = k; }
    }
    if (peak != impulse_at + hop + delay) {
        fprintf(stderr, "  %d/%d: impulse at %d -> peak %d, expected %d\n",
                g->sr, g->fft, impulse_at, peak, impulse_at + hop + delay);
        return 9;
    }
    for (k = peak - 8; k <= peak + 8; ++k) near_energy += (double)g_out[k] * g_out[k];
    if (near_energy < 0.9 * total_energy) return 10;

    /* Tolerance identity: a band-limited signal in P comes back as itself,
     * delayed, within the resamplers' pass-band error. E carries a
     * different signal, which the identity heads must ignore. */
    dfn_rate_bridge_reset(b);
    analyzer_reset(&ae);
    analyzer_reset(&ap);
    for (k = 0; k < n; ++k) {
        float t_s = (float)k / (float)g->sr;
        g_signal[k] = 0.3f * sinf(2.0f * (float)M_PI * 300.0f * t_s) +
                      0.2f * sinf(2.0f * (float)M_PI * 1000.0f * t_s) +
                      0.1f * sinf(2.0f * (float)M_PI * 2500.0f * t_s);
        g_out_b[k] = 0.5f * sinf(2.0f * (float)M_PI * 700.0f * t_s);   /* E */
    }
    if (run_bridge(b, &ae, &ap, g_out_b, g_signal, hops, g_out, NULL, 0) < 0) return 11;
    for (k = 4 * hop + delay + 2 * g->fft; k < n; ++k) {
        float ref = g_signal[k - hop - delay];
        float err = fabsf(g_out[k] - ref);
        if (err > max_err) max_err = err;
        if (fabsf(ref) > max_sig) max_sig = fabsf(ref);
    }
    printf("dfn_rate_bridge: %d Hz / fft %d: added delay %d samples, "
           "round-trip max error %.2e (signal peak %.2f)\n", g->sr, g->fft,
           delay, (double)max_err, (double)max_sig);
    if (!(max_err < 5e-3f)) return 12;

    /* E alone must not reach the output (identity heads apply to P). */
    dfn_rate_bridge_reset(b);
    analyzer_reset(&ae);
    analyzer_reset(&ap);
    if (run_bridge(b, &ae, &ap, g_out_b, NULL, hops, g_out, NULL, 0) < 0) return 13;
    for (k = 0; k < n; ++k) if (g_out[k] != 0.0f) return 14;

    /* Reset parity: N hops, reset, N hops == a fresh instance. */
    dfn_rate_bridge_reset(b);
    analyzer_reset(&ae);
    analyzer_reset(&ap);
    if (run_bridge(b, &ae, &ap, g_out_b, g_signal, 40, g_out, NULL, 0) < 0) return 15;
    dfn_rate_bridge_reset(b);
    analyzer_reset(&ae);
    analyzer_reset(&ap);
    if (run_bridge(b, &ae, &ap, g_out_b, g_signal, 40, g_out + 40 * hop, NULL, 0) < 0) return 16;
    if (memcmp(g_out, g_out + 40 * hop, (size_t)40 * hop * sizeof(float)) != 0) return 17;

    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    dfn_rate_bridge_destroy(b);
    return 0;
}

static int run_rejections(void) {
    DfnRateBridgeConfig cfg;
    DfnRateBridgeMemReq req, bad;
    void *mem = NULL;
    DfnRateBridge *b;
    cfg = bridge_config(48000, 1024);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 1;
    cfg = bridge_config(44100, 0);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 2;
    cfg = bridge_config(8000, 512);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 3;
    cfg = bridge_config(16000, 1024);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 4;
    cfg = bridge_config(16000, 0); cfg.stage.erb_inv = NULL;
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 5;
    cfg = bridge_config(16000, 0); cfg.stage.sample_rate = 16000;
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 6;

    cfg = bridge_config(16000, 0);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) != 0) return 7;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return 8;
    b = dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!b) { free(mem); return 9; }
    dfn_rate_bridge_destroy(b);
    bad = req; bad.build_flags_hash ^= 1u;
    if (dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 10; }
    bad = req; bad.layout_version += 1u;
    if (dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 11; }
    if (dfn_rate_bridge_init(mem, (size_t)req.bytes - 16u, &cfg)) { free(mem); return 12; }
    printf("dfn_rate_bridge: 16 kHz / fft 256 pool %llu bytes (model weights external)\n",
           (unsigned long long)req.bytes);
    free(mem);
    return 0;
}

int main(void) {
    int rc, i;
    dfn_unit_partition(erb_fwd, erb_inv);
    rc = run_rejections();
    if (rc) { fprintf(stderr, "rate bridge rejection row failed: %d\n", rc); return 1; }
    for (i = 0; i < N_GRIDS; ++i) {
        rc = run_grid(&GRIDS[i]);
        if (rc) {
            fprintf(stderr, "rate bridge grid %d/%d failed: %d\n",
                    GRIDS[i].sr, GRIDS[i].fft, rc);
            return 1;
        }
    }
    puts("dfn_rate_bridge: PASS");
    return 0;
}
