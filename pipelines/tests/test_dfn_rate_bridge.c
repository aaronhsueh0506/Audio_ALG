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
#include "audio_stft.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"   /* ALIGN16 */
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

/* Run `hops` hops of estimate=est, apply=app (either may be
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
    /* hop = fft / 2 is the grid contract, not a derived value. */
    if (dfn_rate_bridge_hop_size(b) != hop ||
        dfn_rate_bridge_n_freqs(b) != g->fft / 2 + 1)
        return 2;
    delay = dfn_rate_bridge_added_delay_samples(b);
    if (delay != expected_added_delay(g->sr, hop)) {
        fprintf(stderr, "  %d/%d: added delay %d, expected %d\n", g->sr, g->fft,
                delay, expected_added_delay(g->sr, hop));
        return 3;
    }

    /* Schedule and cumulative frame count (the count contract). The bridge
     * calibrated its prefill over eight hops; a grid whose deficit outgrew
     * that window would surface here as process() refusing a hop. */
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
    /* Below-set fft sizes, even where a host might one day offer them: the
     * gate is the bridge's own (rows 13-14). Row 15 passes that gate and is
     * refused by the stage's (1024 only), reached through the bridge. */
    cfg = bridge_config(16000, 128);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 13;
    cfg = bridge_config(8000, 128);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 14;
    cfg = bridge_config(16000, 0); cfg.stage.fft_size = 512;
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 15;

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

/* The bridge and both ULCNet pipelines size their STFT/ISTFT objects before
 * any FFT handle exists, as the object that carves its own handle and window
 * less those two regions. Pin that accounting against the size audiolib
 * reports for an object that really borrows them, for AcStft and AcIstft in
 * both synthesis modes, and show that a block one alignment unit short is
 * refused. */
static int run_borrowed_sizes(void) {
    static const int sizes[] = { 256, 512, 1024 };
    size_t i;
    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        const int n = sizes[i];
        AcStftConfig own, borrowed;
        FftHandle *fft = fft_create(n);
        float *window = (float *)malloc((size_t)n * sizeof(float));
        size_t shared, a_own, a_b, s_own, s_b;
        void *mem = NULL;
        int mode, rc = 0;
        if (!fft || !window) return 1;
        ac_window_build(window, n);
        shared = ALIGN16(fft_get_mem_size(n)) + ALIGN16((size_t)n * sizeof(float));
        for (mode = 0; mode < 2 && !rc; ++mode) {
            ac_stft_config_default(&own, n);
            own.ola = mode ? AC_OLA_ENV_NORM : AC_OLA_ACCUMULATE;
            borrowed = own;
            borrowed.fft = fft;
            borrowed.window = window;
            a_own = ac_stft_get_mem_size(&own);
            a_b = ac_stft_get_mem_size(&borrowed);
            s_own = ac_istft_get_mem_size(&own);
            s_b = ac_istft_get_mem_size(&borrowed);
            if (!a_own || !a_b || !s_own || !s_b) rc = 2;
            else if (a_own - shared != a_b) rc = 3;
            else if (s_own - shared != s_b) rc = 4;
            else {
                if (posix_memalign(&mem, 16, s_b + 16) != 0) rc = 5;
                else {
                    if (!ac_istft_init(mem, s_b, &borrowed)) rc = 6;
                    else if (ac_istft_init(mem, s_b - 16, &borrowed)) rc = 7;
                    else if (!ac_stft_init(mem, a_b, &borrowed)) rc = 8;
                    else if (ac_stft_init(mem, a_b - 16, &borrowed)) rc = 9;
                    free(mem);
                }
            }
        }
        free(window);
        fft_destroy(fft);
        if (rc) return 100 * (int)(i + 1) + rc;
    }
    return 0;
}

/* The phased hop (begin / nn_begin ... nn_end / finish) against the one-call
 * process(): with a non-identity model (a uniform 0.5 band mask) the loop the
 * caller writes reproduces the configured callback byte for byte on every
 * native grid, including the hops that carry two stage frames. The model-free
 * rows: a frame the caller reports as failed, and a frame it leaves open or
 * never fetches (finish takes it as the identity), equal a bridge with no
 * model. Sequence errors are refused. */
static int bridge_half_mask(void *user, const DFN2PrepostInputs *in,
                            DFN2PrepostOutputs *out) {
    size_t i;
    (void)user; (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.5f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    return 0;
}

/* behaviour of the caller in the phased twin */
enum { BP_RUN, BP_FAIL, BP_OPEN, BP_UNFETCHED };

static int run_phased_grid(const Grid *g, int behaviour) {
    DFN2ModelIoDescriptor descriptor;
    DfnRateBridgeConfig ph_cfg = bridge_config(g->sr, g->fft);
    DfnRateBridgeConfig cb_cfg = bridge_config(g->sr, g->fft);
    DfnRateBridge *ph, *cb;
    Analyzer ae, ap;
    static Complex se[MAX_FFT / 2 + 1], sa[MAX_FFT / 2 + 1];
    static float est[MAX_HOPS * (MAX_FFT / 2)], app[MAX_HOPS * (MAX_FFT / 2)];
    static float oa[MAX_FFT / 2], ob[MAX_FFT / 2];
    int hop = g->fft / 2, hops = 120, t, k, max_frames = 0, nonzero = 0;
    DfnResStageJob job;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    if (behaviour == BP_RUN) {
        cb_cfg.stage.model.infer = bridge_half_mask;
        cb_cfg.stage.model.io_descriptor = &descriptor;
    }
    ph = dfn_rate_bridge_create(&ph_cfg);
    cb = dfn_rate_bridge_create(&cb_cfg);
    if (!ph || !cb || analyzer_init(&ae, g->fft) != 0 || analyzer_init(&ap, g->fft) != 0)
        return 2;
    dfn_fixture_multitone(est, (size_t)hops * hop, g->sr);
    for (k = 0; k < hops * hop; ++k) app[k] = 0.7f * est[k];
    for (t = 0; t < hops; ++t) {
        int r, frames_this_hop = 0;
        analyzer_push(&ae, est + t * hop, se);
        analyzer_push(&ap, app + t * hop, sa);
        if (dfn_rate_bridge_process(cb, se, sa, ob) != 0) return 3;
        if (dfn_rate_bridge_begin(ph, se, sa) != 0) return 4;
        if (behaviour != BP_UNFETCHED) {
            while ((r = dfn_rate_bridge_nn_begin(ph, &job)) == 1) {
                int rc = 0;
                ++frames_this_hop;
                if (behaviour == BP_OPEN) break;       /* left open: finish takes the identity */
                if (behaviour == BP_RUN) rc = bridge_half_mask(NULL, &job.inputs, &job.outputs);
                else rc = -1;
                if (dfn_rate_bridge_nn_end(ph, rc) != 0) return 5;
            }
            if (r < 0) return 6;
        }
        if (dfn_rate_bridge_finish(ph, oa) != 0) return 7;
        if (frames_this_hop > max_frames) max_frames = frames_this_hop;
        if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) return 10 + t;
        for (k = 0; k < hop; ++k) if (oa[k] != 0.0f) nonzero = 1;
    }
    if (!nonzero) return 8;
    /* the 1,2 schedules (and 0,1,1,1) are walked by the run above */
    if (behaviour == BP_RUN && max_frames < 1) return 9;
    dfn_rate_bridge_destroy(ph);
    dfn_rate_bridge_destroy(cb);
    return 0;
}

static int run_phased_sequence(void) {
    DfnRateBridgeConfig cfg = bridge_config(16000, 256);
    DfnRateBridge *b = dfn_rate_bridge_create(&cfg);
    Analyzer ae, ap;
    static Complex se[129], sa[129];
    static float hopbuf[128], out[128];
    DfnResStageJob job;
    if (!b || analyzer_init(&ae, 256) != 0 || analyzer_init(&ap, 256) != 0) return 1;
    memset(hopbuf, 0, sizeof(hopbuf));
    analyzer_push(&ae, hopbuf, se);
    analyzer_push(&ap, hopbuf, sa);
    if (dfn_rate_bridge_nn_begin(b, &job) != -1 || dfn_rate_bridge_nn_end(b, 0) != -1 ||
        dfn_rate_bridge_finish(b, out) != -1)
        return 2;                                    /* nothing begun */
    if (dfn_rate_bridge_begin(b, se, sa) != 0 || dfn_rate_bridge_begin(b, se, sa) != -1)
        return 3;                                    /* a second begin */
    if (dfn_rate_bridge_nn_end(b, 0) != -1) return 4;  /* no frame open */
    if (dfn_rate_bridge_finish(b, NULL) != -1) return 5;
    if (dfn_rate_bridge_finish(b, out) != 0 || dfn_rate_bridge_finish(b, out) != -1)
        return 6;
    dfn_rate_bridge_destroy(b);
    return 0;
}

int main(void) {
    int rc, i;
    dfn_unit_partition(erb_fwd, erb_inv);
    rc = run_borrowed_sizes();
    if (rc) { fprintf(stderr, "borrowed STFT/ISTFT sizing row failed: %d\n", rc); return 1; }
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
    for (i = 0; i < N_GRIDS; ++i) {
        int b;
        for (b = 0; b < 4; ++b) {
            rc = run_phased_grid(&GRIDS[i], b);
            if (rc) {
                fprintf(stderr, "rate bridge phased grid %d/%d behaviour %d failed: %d\n",
                        GRIDS[i].sr, GRIDS[i].fft, b, rc);
                return 1;
            }
        }
    }
    rc = run_phased_sequence();
    if (rc) { fprintf(stderr, "rate bridge phased sequence row failed: %d\n", rc); return 1; }
    puts("dfn_rate_bridge: PASS");
    return 0;
}
