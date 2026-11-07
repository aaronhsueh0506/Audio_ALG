/* dfn_rate_bridge acceptance test (no AEC): the count schedule and its
 * prefill, the added-delay constant pinned by an impulse, the tolerance
 * identity through the resampler round trip, the estimate/apply split, reset
 * parity and the state generation, on every supported native grid, and the
 * reject-first configuration rows (a model grid other than 48 kHz / 1024 with
 * one frame per call, a foreign model). Spectra are produced by the same
 * sqrt-Hann 50%-overlap analysis the hosting pipelines perform, so the bridge
 * sees exactly the frames it will see in product. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn/dfn_rate_bridge.h"
#include "audio_stft.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"   /* ALIGN16 */
#include "dfn/tests/dfn_test_fixture.h"

/* The tolerance-identity input (300, 1000, 2500 Hz) in the precision of T, and the same tones as
 * the reference at the exact delayed time. */
#define THREE_TONES(T, SIN, PI, t) \
    ((T)0.3 * SIN((T)2 * PI * (T)300 * t) + (T)0.2 * SIN((T)2 * PI * (T)1000 * t) + \
     (T)0.1 * SIN((T)2 * PI * (T)2500 * t))

#define MAX_FFT 512
#define MAX_HOPS 200
#define M_FRAMES DFN2_MASK_LOOKAHEAD   /* warm-up stage frames after a start or reset */

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
    DfnRateBridgeConfig cfg = dfn_rate_bridge_default_config(sr, dfn_test_grid());
    cfg.fft_size = fft;
    cfg.stage.erb_fwd = erb_fwd;
    cfg.stage.erb_inv = erb_inv;
    return cfg;
}

/* prefill (the schedule's largest deficit) + 16 + (L + 1)*512*sr/48000 + 16, in sixths. */
static int expected_added_delay_sixths(int sr, int hop) {
    return dfn_fixture_bridge_added_delay_sixths(sr, hop);
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
    int num = 0, den = 0;
    double exact, centroid = 0.0, centroid_mass = 0.0;
    float peak_abs = 0.0f, max_err = 0.0f, max_sig = 0.0f;
    double near_energy = 0.0, total_energy = 0.0;

    if (!b) return 1;
    /* hop = fft / 2 is the grid contract, not a derived value. */
    if (dfn_rate_bridge_hop_size(b) != hop ||
        dfn_rate_bridge_n_freqs(b) != g->fft / 2 + 1)
        return 2;
    delay = dfn_rate_bridge_added_delay_samples(b);
    if (dfn_rate_bridge_added_delay_exact(b, &num, &den) != 0 ||
        den != DFN_RATE_BRIDGE_DELAY_DENOMINATOR ||
        num != expected_added_delay_sixths(g->sr, hop) ||
        delay != num / den) {
        fprintf(stderr, "  %d/%d: added delay %d = %d/%d, expected %d/%d\n", g->sr, g->fft,
                delay, num, den, expected_added_delay_sixths(g->sr, hop),
                DFN_RATE_BRIDGE_DELAY_DENOMINATOR);
        return 3;
    }
    exact = (double)num / (double)den;

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

    /* Impulse in P: the output sits hop + the exact added delay later (the hop is the
     * host-style WOLA the bridge performs on P). The round trip is a symmetric band-limited
     * kernel, so its centroid is that position to a fraction of a sample, and the peak sample
     * is the nearest one. */
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
    for (k = peak - 24; k <= peak + 24; ++k) {
        centroid += (double)k * g_out[k];
        centroid_mass += g_out[k];
    }
    centroid /= centroid_mass;
    if (fabs(centroid - (double)(impulse_at + hop) - exact) > 0.01 ||
        fabs((double)peak - (double)(impulse_at + hop) - exact) >= 0.5) {
        fprintf(stderr, "  %d/%d: impulse at %d -> peak %d centroid %.4f, expected %.4f\n",
                g->sr, g->fft, impulse_at, peak, centroid, (double)(impulse_at + hop) + exact);
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
        g_signal[k] = THREE_TONES(float, sinf, (float)M_PI, t_s);
        g_out_b[k] = 0.5f * sinf(2.0f * (float)M_PI * 700.0f * t_s);   /* E */
    }
    if (run_bridge(b, &ae, &ap, g_out_b, g_signal, hops, g_out, NULL, 0) < 0) return 11;
    for (k = 4 * hop + delay + 2 * g->fft; k < n; ++k) {
        /* The same three tones evaluated at the exact (possibly fractional) delayed time. */
        double t_s = ((double)k - (double)hop - exact) / (double)g->sr;
        float ref = (float)THREE_TONES(double, sin, M_PI, t_s);
        float err = fabsf(g_out[k] - ref);
        if (err > max_err) max_err = err;
        if (fabsf(ref) > max_sig) max_sig = fabsf(ref);
    }
    printf("dfn_rate_bridge: %d Hz / fft %d: added delay %d + %d/%d samples, "
           "round-trip max error %.2e (signal peak %.2f)\n", g->sr, g->fft,
           delay, num - delay * den, den, (double)max_err, (double)max_sig);
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
    unsigned char *snapshot;
    void *mem = NULL;
    DfnRateBridge *b;
    int k;
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
    /* The model must be the 48 kHz / 1024 grid with one frame per call: a stage on another grid is
     * refused whatever matrices it carries (rows 6 and 15 give it matrices of its own size, so the
     * grid is the only reason; run_other_grid_refusal() has the full set). */
    {
        static float other_fwd[DFN_TEST_TABLE_MAX_BINS * DFN_TEST_TABLE_MAX_ERB];
        static float other_inv[DFN_TEST_TABLE_MAX_ERB * DFN_TEST_TABLE_MAX_BINS];
        DFN2Grid g16, g48_512, g48_two;
        DFN2GridSpec spec16 = dfn_test_spec(16000, 256, 24, 96, 0, 0);
        DFN2GridSpec spec512 = dfn_test_spec(48000, 512, 28, 96, 1, 1);
        DFN2GridSpec spec_two = dfn_test_spec(48000, 1024, 32, 96, 1, 1);
        if (dfn2_grid_make(&g16, &spec16) != DFN2_GRID_OK) return 42;
        dfn_unit_partition(other_fwd, other_inv, &g16);
        cfg = bridge_config(16000, 0);
        cfg.stage.grid = g16;
        cfg.stage.erb_fwd = other_fwd;
        cfg.stage.erb_inv = other_inv;
        if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 6;
        /* the same stage config is a valid stage on its own grid */
        if (dfn_res_stage_get_mem_requirements(&cfg.stage, &(DfnResStageMemReq){0}) != 0) return 43;
        if (dfn2_grid_make(&g48_512, &spec512) != DFN2_GRID_OK) return 44;
        dfn_unit_partition(other_fwd, other_inv, &g48_512);
        cfg = bridge_config(16000, 0);
        cfg.stage.grid = g48_512;
        cfg.stage.erb_fwd = other_fwd;
        cfg.stage.erb_inv = other_inv;
        if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 15;
        if (dfn_res_stage_get_mem_requirements(&cfg.stage, &(DfnResStageMemReq){0}) != 0) return 45;
        /* 48 kHz / 1024 with two frames per call: the library makes the grid, the stage refuses it,
         * and so do the bridge's size query and create (the rate-default config of each native grid). */
        spec_two.frames_per_call = 2;
        if (dfn2_grid_make(&g48_two, &spec_two) != DFN2_GRID_OK) return 46;
        dfn_unit_partition(other_fwd, other_inv, &g48_two);
        {
            static const Grid native[] = { {16000, 0}, {16000, 512}, {8000, 0} };
            size_t n;
            for (n = 0; n < sizeof(native) / sizeof(native[0]); ++n) {
                cfg = bridge_config(native[n].sr, native[n].fft);
                cfg.stage.grid = g48_two;
                cfg.stage.erb_fwd = other_fwd;
                cfg.stage.erb_inv = other_inv;
                if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 47;
                if (dfn_rate_bridge_create(&cfg) != NULL) return 48;
                if (dfn_res_stage_get_mem_requirements(&cfg.stage, &(DfnResStageMemReq){0}) == 0) return 49;
                cfg.stage.grid = *dfn_test_grid();   /* the control: one frame per call, accepted */
                if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) != 0) return 50;
            }
        }
    }
    /* Below-set fft sizes, even where a host might one day offer them: the
     * gate is the bridge's own (rows 13-14). Row 15 passes that gate and is
     * refused by the stage's (1024 only), reached through the bridge. */
    cfg = bridge_config(16000, 128);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 13;
    cfg = bridge_config(8000, 128);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 14;
    cfg = bridge_config(16000, 0); memset(&cfg.stage.grid, 0, sizeof(cfg.stage.grid));
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 16;   /* a zeroed model grid */
    cfg = bridge_config(16000, 0); cfg.stage.grid.hop += 1;
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 17;   /* a hand-edited one */
    cfg = dfn_rate_bridge_default_config(16000, NULL);
    cfg.stage.erb_fwd = erb_fwd; cfg.stage.erb_inv = erb_inv;
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 18;   /* no model grid at all */
    /* A model built for another lookahead pair, grid, frame count or deep-filter order is refused
     * through the bridge's stage (run time, from its descriptor); the test grid's own descriptor is
     * the accepted control. */
    for (k = 0; k < 5; ++k) {
        DFN2ModelIoDescriptor foreign;
        if (dfn_fixture_foreign_descriptor(dfn_test_grid(), &foreign, k) != 0) return 20 + k;
        cfg = bridge_config(16000, 0);
        cfg.stage.model.infer = dfn_fixture_identity_infer;
        cfg.stage.model.io_descriptor = &foreign;
        if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 30 + k;
    }
    {
        DFN2ModelIoDescriptor own;
        cfg = bridge_config(16000, 0);
        if (dfn2_model_io_descriptor_default(dfn_test_grid(), &own) != 0) return 40;
        cfg.stage.model.infer = dfn_fixture_identity_infer;
        cfg.stage.model.io_descriptor = &own;
        if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) != 0) return 41;
    }

    cfg = bridge_config(16000, 0);
    if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) != 0) return 7;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return 8;
    snapshot = (unsigned char *)malloc((size_t)req.bytes);
    if (!snapshot) { free(mem); return 8; }
    b = dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!b) { free(snapshot); free(mem); return 9; }
    dfn_rate_bridge_destroy(b);
    /* A refused init leaves the caller's pool bytes as they were. */
    memset(mem, 0xA5, (size_t)req.bytes);
    memcpy(snapshot, mem, (size_t)req.bytes);
    bad = req; bad.build_flags_hash ^= 1u;
    if (dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(snapshot); free(mem); return 10; }
    bad = req; bad.layout_version += 1u;
    if (dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(snapshot); free(mem); return 11; }
    if (dfn_rate_bridge_init(mem, (size_t)req.bytes - 16u, &cfg)) { free(snapshot); free(mem); return 12; }
    if (memcmp(snapshot, mem, (size_t)req.bytes) != 0) { free(snapshot); free(mem); return 51; }
    free(snapshot);
    printf("dfn_rate_bridge: 16 kHz / fft 256 pool %llu bytes (model weights external)\n",
           (unsigned long long)req.bytes);
    printf("dfn_rate_bridge: 16 kHz / fft 256 pool descriptor hash %08x\n", req.build_flags_hash);
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
/* behaviour of the caller in the phased twin */

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

    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
    if (behaviour == BP_RUN) {
        cb_cfg.stage.model.infer = dfn_fixture_half_mask_infer;
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
                if (behaviour == BP_RUN)
                    rc = dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs);
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
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    return 0;
}

/* dfn_rate_bridge_reset_synthesis(): the host-grid synthesis of the bridge cleared on the hop a
 * post-chain reset names, and nothing else.
 *   tap       the native-grid hops the bridge synthesised (dfn_rate_bridge_get_host_hops) equal, bit
 *             for bit, an independent audiolib synthesis of the same spectra that clears its tail on
 *             that hop (both lanes), and a bridge never told differs from it on that hop only;
 *   timing    the same call one hop late leaves the reset hop uncleared (the tap row fails);
 *   kept      the stage and bridge counters of the cleared run equal the uncleared run's, the output
 *             starts to differ no earlier than the reset hop, and resumes bit-identical to the
 *             uncleared run once the one-hop perturbation has drained (a state reset anywhere else
 *             would shift the 48 kHz frame alignment or the counters for good);
 *   contract  NULL and a hop in progress are refused and change nothing.
 * Mutations the row must see: no call, a late call, and dfn_rate_bridge_reset() in its place. */
#define RS_HOPS 100
#define RS_AT 41   /* odd: the 48 kHz FIFO holds leftover samples at the start of this hop on every grid that has them */
enum { RS_CLEARED, RS_NEVER, RS_LATE, RS_FULL_RESET };

typedef struct RsRun {
    float out[RS_HOPS * (MAX_FFT / 2)];
    float tap_e[RS_HOPS * (MAX_FFT / 2)];
    float tap_p[RS_HOPS * (MAX_FFT / 2)];
    long long stage_frames, stage_commits, stage_skips, bridge_frames;
} RsRun;

static Complex rs_spec_e[RS_HOPS][MAX_FFT / 2 + 1];
static Complex rs_spec_p[RS_HOPS][MAX_FFT / 2 + 1];
static RsRun rs_run[4];
static float rs_oracle_cleared_e[RS_HOPS * (MAX_FFT / 2)], rs_oracle_cleared_p[RS_HOPS * (MAX_FFT / 2)];
static float rs_oracle_never_e[RS_HOPS * (MAX_FFT / 2)], rs_oracle_never_p[RS_HOPS * (MAX_FFT / 2)];

static int rs_drive(const Grid *g, int mode, RsRun *run) {
    DfnRateBridgeConfig cfg = bridge_config(g->sr, g->fft);
    DfnRateBridge *b = dfn_rate_bridge_create(&cfg);
    int hop = g->fft / 2, t;
    if (!b) return 1;
    for (t = 0; t < RS_HOPS; ++t) {
        const float *te, *tp;
        if (t == RS_AT && mode == RS_CLEARED && dfn_rate_bridge_reset_synthesis(b) != 0) return 2;
        if (t == RS_AT && mode == RS_FULL_RESET) dfn_rate_bridge_reset(b);
        if (dfn_rate_bridge_begin(b, rs_spec_e[t], rs_spec_p[t]) != 0) return 3;
        if (dfn_rate_bridge_get_host_hops(b, &te, &tp) != 0 || !te || !tp) return 4;
        memcpy(run->tap_e + t * hop, te, (size_t)hop * sizeof(float));
        memcpy(run->tap_p + t * hop, tp, (size_t)hop * sizeof(float));
        if (t == RS_AT && mode == RS_CLEARED) {
            /* a hop in progress: refused, nothing cleared (the tap keeps the hop just synthesised) */
            if (dfn_rate_bridge_reset_synthesis(b) != -1) return 5;
            if (dfn_rate_bridge_get_host_hops(b, &te, &tp) != 0 ||
                memcmp(te, run->tap_e + t * hop, (size_t)hop * sizeof(float)) != 0)
                return 6;
        }
        {
            DfnResStageJob job;
            int r;
            while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1)
                if (dfn_rate_bridge_nn_end(b, dfn_res_stage_infer(dfn_rate_bridge_stage(b), &job)) != 0)
                    return 7;
            if (r < 0) return 8;
        }
        if (dfn_rate_bridge_finish(b, run->out + t * hop) != 0) return 9;
        if (t == RS_AT && mode == RS_LATE && dfn_rate_bridge_reset_synthesis(b) != 0) return 10;
    }
    dfn_res_stage_get_counters(dfn_rate_bridge_stage(b), &run->stage_frames, &run->stage_commits,
                               &run->stage_skips);
    {
        int last;
        dfn_rate_bridge_get_counters(b, &run->bridge_frames, &last);
    }
    dfn_rate_bridge_destroy(b);
    return 0;
}

/* The oracle: audiolib's default-window synthesis of the same spectra, cleared before `clear_at`. */
static int rs_oracle(int fft, const Complex (*spec)[MAX_FFT / 2 + 1], int clear_at, float *out) {
    AcStftConfig sc;
    AcIstft *s;
    void *mem = NULL;
    int t, hop = fft / 2;
    size_t bytes;
    ac_stft_config_default(&sc, fft);
    bytes = ac_istft_get_mem_size(&sc);
    if (!bytes || posix_memalign(&mem, 16, bytes) != 0) return 1;
    s = ac_istft_init(mem, bytes, &sc);
    if (!s) { free(mem); return 2; }
    for (t = 0; t < RS_HOPS; ++t) {
        if (t == clear_at) ac_istft_reset(s);
        (void)ac_istft_push(s, spec[t], out + t * hop);
    }
    free(mem);
    return 0;
}

static int run_reset_synthesis(const Grid *g) {
    int hop = g->fft / 2, t, k, rc;
    Analyzer ae, ap;
    DfnRateBridgeConfig cfg = bridge_config(g->sr, g->fft);
    DfnRateBridge *probe = dfn_rate_bridge_create(&cfg);
    static float est[RS_HOPS * (MAX_FFT / 2)], app[RS_HOPS * (MAX_FFT / 2)];
    int added, horizon, first_diff = -1, last_diff = -1, tap_diff_hops = 0;
    float tap_delta = 0.0f;
    double out_energy = 0.0, tap_energy = 0.0, energy_ratio;
    const float *none = NULL;

    if (!probe || analyzer_init(&ae, g->fft) != 0 || analyzer_init(&ap, g->fft) != 0) return 1;
    added = dfn_rate_bridge_added_delay_samples(probe);
    if (dfn_rate_bridge_reset_synthesis(NULL) != -1 ||
        dfn_rate_bridge_get_host_hops(NULL, &none, &none) != -1 ||
        dfn_rate_bridge_get_host_hops(probe, NULL, NULL) != 0)
        return 2;
    dfn_rate_bridge_destroy(probe);

    dfn_fixture_multitone(app, (size_t)RS_HOPS * hop, g->sr);
    for (k = 0; k < RS_HOPS * hop; ++k)
        est[k] = 0.5f * sinf(2.0f * (float)M_PI * 700.0f * (float)k / (float)g->sr);
    for (t = 0; t < RS_HOPS; ++t) {
        analyzer_push(&ae, est + t * hop, rs_spec_e[t]);
        analyzer_push(&ap, app + t * hop, rs_spec_p[t]);
    }
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    if (rs_oracle(g->fft, rs_spec_e, RS_AT, rs_oracle_cleared_e) != 0 ||
        rs_oracle(g->fft, rs_spec_p, RS_AT, rs_oracle_cleared_p) != 0 ||
        rs_oracle(g->fft, rs_spec_e, -1, rs_oracle_never_e) != 0 ||
        rs_oracle(g->fft, rs_spec_p, -1, rs_oracle_never_p) != 0)
        return 3;
    for (k = 0; k < 4; ++k) {
        rc = rs_drive(g, k, &rs_run[k]);
        if (rc) { fprintf(stderr, "  %d/%d: drive %d failed: %d\n", g->sr, g->fft, k, rc); return 4; }
    }

    /* tap: cleared == oracle that clears on the hop; never == oracle that never clears */
    if (memcmp(rs_run[RS_CLEARED].tap_p, rs_oracle_cleared_p, (size_t)RS_HOPS * hop * sizeof(float)) != 0 ||
        memcmp(rs_run[RS_CLEARED].tap_e, rs_oracle_cleared_e, (size_t)RS_HOPS * hop * sizeof(float)) != 0)
        return 5;
    if (memcmp(rs_run[RS_NEVER].tap_p, rs_oracle_never_p, (size_t)RS_HOPS * hop * sizeof(float)) != 0 ||
        memcmp(rs_run[RS_NEVER].tap_e, rs_oracle_never_e, (size_t)RS_HOPS * hop * sizeof(float)) != 0)
        return 6;
    /* the two oracles, hence the two runs, differ on the reset hop and on no other hop */
    for (t = 0; t < RS_HOPS; ++t) {
        int same = memcmp(rs_run[RS_CLEARED].tap_p + t * hop, rs_run[RS_NEVER].tap_p + t * hop,
                          (size_t)hop * sizeof(float)) == 0 &&
                   memcmp(rs_run[RS_CLEARED].tap_e + t * hop, rs_run[RS_NEVER].tap_e + t * hop,
                          (size_t)hop * sizeof(float)) == 0;
        if (!same) {
            ++tap_diff_hops;
            if (t != RS_AT) return 7;
            for (k = 0; k < hop; ++k) {
                float d = fabsf(rs_run[RS_CLEARED].tap_p[t * hop + k] - rs_run[RS_NEVER].tap_p[t * hop + k]);
                if (d > tap_delta) tap_delta = d;
            }
        }
    }
    if (tap_diff_hops != 1 || !(tap_delta > 0.01f)) return 8;   /* the stimulus must carry a real tail */
    /* timing: one hop late the reset hop is not cleared, and the following hop is wrong too */
    if (memcmp(rs_run[RS_LATE].tap_p + RS_AT * hop, rs_oracle_cleared_p + RS_AT * hop,
               (size_t)hop * sizeof(float)) == 0)
        return 9;
    if (memcmp(rs_run[RS_LATE].tap_p + (RS_AT + 1) * hop, rs_oracle_cleared_p + (RS_AT + 1) * hop,
               (size_t)hop * sizeof(float)) == 0)
        return 10;

    /* kept: counters, and the output of the cleared run against the uncleared one */
    if (rs_run[RS_CLEARED].stage_frames != rs_run[RS_NEVER].stage_frames ||
        rs_run[RS_CLEARED].stage_commits != rs_run[RS_NEVER].stage_commits ||
        rs_run[RS_CLEARED].stage_skips != rs_run[RS_NEVER].stage_skips ||
        rs_run[RS_CLEARED].bridge_frames != rs_run[RS_NEVER].bridge_frames)
        return 11;
    if (rs_run[RS_FULL_RESET].stage_frames == rs_run[RS_NEVER].stage_frames &&
        rs_run[RS_FULL_RESET].stage_skips == rs_run[RS_NEVER].stage_skips)
        return 12;   /* the counters would not see a stage reset */
    for (k = 0; k < RS_HOPS * hop; ++k)
        if (rs_run[RS_CLEARED].out[k] != rs_run[RS_NEVER].out[k]) {
            if (first_diff < 0) first_diff = k;
            last_diff = k;
        }
    /* The identity bridge is a unit-gain linear path for this band-limited stimulus, so the output
     * difference carries the energy of the cleared tail and no more: a clear that also drops the
     * 48 kHz synthesis tail or the resamplers' history adds a second perturbation (ratios 1.2 to 2.1). */
    for (k = 0; k < RS_HOPS * hop; ++k) {
        double d = (double)rs_run[RS_CLEARED].out[k] - (double)rs_run[RS_NEVER].out[k];
        double dt = (double)rs_run[RS_CLEARED].tap_p[k] - (double)rs_run[RS_NEVER].tap_p[k];
        out_energy += d * d;
        tap_energy += dt * dt;
    }
    energy_ratio = out_energy / tap_energy;
    if (!(energy_ratio > 0.99 && energy_ratio < 1.01)) {
        fprintf(stderr, "  %d/%d: output difference carries %.4f of the cleared tail's energy\n", g->sr, g->fft, energy_ratio);
        return 15;
    }
    horizon = RS_AT * hop + added + 4 * g->fft;
    if (first_diff < RS_AT * hop || last_diff < 0 || last_diff >= horizon) {
        fprintf(stderr, "  %d/%d: output differs over [%d, %d], expected inside [%d, %d)\n",
                g->sr, g->fft, first_diff, last_diff, RS_AT * hop, horizon);
        return 13;
    }
    if (horizon + 4 * hop >= RS_HOPS * hop) return 14;   /* a drained tail must be observed */
    printf("dfn_rate_bridge: %d Hz / fft %d: reset_synthesis on hop %d: host hops bit-exact to the audiolib "
           "oracle (one hop differs from the uncleared run, max %.3f), output differs over samples [%d, %d] "
           "(delay %d) with %.4f of the cleared tail's energy, identical after, stage counters unchanged\n",
           g->sr, g->fft, RS_AT, (double)tap_delta, first_diff, last_diff, added, energy_ratio);
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
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    return 0;
}

/* The bridge reports its stage's state generation: 1 after create, plus one per reset, one per
 * warm-up frame (the first mask_lookahead stage frames, whatever the model wrote) and one per refused
 * commit of a real frame (a head the commit finds non-finite, closed through nn_end); a model that
 * reports failure leaves it alone. A caller that holds the model's state refills it when the value
 * changes. */
static int run_state_generation(void) {
    DfnRateBridgeConfig cfg = bridge_config(16000, 256);
    DfnRateBridge *b = dfn_rate_bridge_create(&cfg);
    Analyzer ae, ap;
    static Complex se[129], sa[129];
    static float hopbuf[128], out[128];
    DfnResStageJob job;
    unsigned int expect = 1u;
    int pass, t, refused = 0, warm = 0, seen = 0, r;

    if (dfn_rate_bridge_state_generation(NULL) != 0u) return 1;
    if (!b || analyzer_init(&ae, 256) != 0 || analyzer_init(&ap, 256) != 0) return 2;
    memset(hopbuf, 0, sizeof(hopbuf));
    if (dfn_rate_bridge_state_generation(b) != expect) return 3;
    for (pass = 0; pass < 2; ++pass) {   /* 0: heads with a NaN (refused commit), 1: failed model runs */
        for (t = 0; t < 12; ++t) {
            analyzer_push(&ae, hopbuf, se);
            analyzer_push(&ap, hopbuf, sa);
            if (dfn_rate_bridge_begin(b, se, sa) != 0) return 4;
            while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1) {
                int rc = -1;
                if (pass == 0) {
                    dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs);
                    job.outputs.erb_mask[3] = (float)NAN;
                    rc = 0;
                    ++expect;                          /* a warm-up frame discards its heads: +1 */
                    if (seen < M_FRAMES) ++warm;       /* a real frame refuses them: +1 */
                    else ++refused;
                }
                ++seen;
                if (dfn_rate_bridge_nn_end(b, rc) != 0) return 5;
                if (dfn_rate_bridge_state_generation(b) != expect) return 6 + pass;
            }
            if (r < 0 || dfn_rate_bridge_finish(b, out) != 0) return 8;
            if (dfn_rate_bridge_state_generation(b) !=
                dfn_res_stage_state_generation(dfn_rate_bridge_stage(b))) return 9;
        }
    }
    if (refused < 1) return 10;   /* the NaN pass must have refused commits of real frames */
    if (warm != M_FRAMES || expect != 1u + (unsigned int)M_FRAMES + (unsigned int)refused) return 12;
    dfn_rate_bridge_reset(b);
    if (dfn_rate_bridge_state_generation(b) != expect + 1u) return 11;
    dfn_rate_bridge_destroy(b);
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    return 0;
}

/* One model call per stage frame, warm-up frames included: the calls of each host hop equal the
 * hop's stage frames (the schedule 0,1,1,1 at 16 kHz / 128 and 1,2 elsewhere), their total the
 * 48 kHz frame count, and the generation is 1 + calls while the first mask_lookahead calls go by,
 * 1 + mask_lookahead after. The added delay does not depend on any of it. */
static int counting_calls(void *user, const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    ++*(int *)user;
    return dfn_fixture_identity_infer(NULL, in, out);
}

static int run_model_calls(const Grid *g, int pinned_delay) {
    DFN2ModelIoDescriptor descriptor;
    DfnRateBridgeConfig cfg = bridge_config(g->sr, g->fft);
    DfnRateBridge *b;
    Analyzer ae, ap;
    static Complex se[MAX_FFT / 2 + 1], sa[MAX_FFT / 2 + 1];
    static float hopbuf[MAX_FFT / 2], out[MAX_FFT / 2];
    long long frames = 0;
    int calls = 0, t, last = 0, first_call_hop = -1, rc = 0;

    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
    cfg.stage.model.infer = counting_calls;
    cfg.stage.model.user = &calls;
    cfg.stage.model.io_descriptor = &descriptor;
    b = dfn_rate_bridge_create(&cfg);
    if (!b || analyzer_init(&ae, g->fft) != 0 || analyzer_init(&ap, g->fft) != 0) return 2;
    memset(hopbuf, 0, sizeof(hopbuf));
    if (pinned_delay >= 0 && dfn_rate_bridge_added_delay_samples(b) != pinned_delay) rc = 3;
    for (t = 0; !rc && t < 24; ++t) {
        const int before = calls;
        unsigned int want_generation;
        analyzer_push(&ae, hopbuf, se);
        analyzer_push(&ap, hopbuf, sa);
        if (dfn_rate_bridge_process(b, se, sa, out) != 0) { rc = 4; break; }
        dfn_rate_bridge_get_counters(b, &frames, &last);
        if (calls - before != last) rc = 5;
        else if (last != (g->sr == 16000 && g->fft == 256 ? (t % 4 == 0 ? 0 : 1) : (t % 2 == 0 ? 1 : 2)))
            rc = 6;
        if (calls > 0 && first_call_hop < 0) first_call_hop = t;
        want_generation = 1u + (unsigned int)(calls < M_FRAMES ? calls : M_FRAMES);
        if (dfn_rate_bridge_state_generation(b) != want_generation) rc = 7;
        if (calls != frames) rc = 8;
    }
    if (!rc && first_call_hop != (g->sr == 16000 && g->fft == 256 ? 1 : 0)) rc = 9;
    dfn_rate_bridge_destroy(b);
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    return rc;
}

/* A model grid that is not 48 kHz / 1024 -- the test grid's own when this program was built for
 * -DDFN2_GRID_KHZ=16|32 (dfn_test_grid.h), and the 16 and 32 kHz rows of the board samples'
 * table whatever the test grid: every bridge configuration is refused at run time, the same stage
 * config is accepted on its own, and no bridge is created. */
static int run_other_grid_refusal(const DFN2Grid *model) {
    static float other_fwd[DFN_TEST_TABLE_MAX_BINS * DFN_TEST_TABLE_MAX_ERB];
    static float other_inv[DFN_TEST_TABLE_MAX_ERB * DFN_TEST_TABLE_MAX_BINS];
    DfnRateBridgeConfig cfg;
    DfnRateBridgeMemReq req;
    DfnResStageConfig stage;
    DfnResStageMemReq stage_req;
    static const int rates[] = { 8000, 16000 };
    int i;
    if (!dfn_test_table_fits(model)) return 4;
    dfn_unit_partition(other_fwd, other_inv, model);
    for (i = 0; i < 2; ++i) {
        cfg = dfn_rate_bridge_default_config(rates[i], model);
        cfg.stage.erb_fwd = other_fwd;
        cfg.stage.erb_inv = other_inv;
        if (dfn_rate_bridge_get_mem_requirements(&cfg, &req) == 0) return 1;
        if (dfn_rate_bridge_create(&cfg) != NULL) return 2;
    }
    stage = dfn_res_stage_default_config(model);
    stage.erb_fwd = other_fwd;
    stage.erb_inv = other_inv;
    if (dfn_res_stage_get_mem_requirements(&stage, &stage_req) != 0) return 3;
    return 0;
}

int main(void) {
    int rc, i;
    DFN2_REQUIRE_VARIANT_LINK();   /* does not link against an archive of the other DFN2 variant */
    dfn_unit_partition(erb_fwd, erb_inv, dfn_test_grid());
    dfn_test_report_pair("dfn_rate_bridge", -1);
    {   /* the table's non-48 kHz model grids: refused by the bridge, run at run time by the stage */
        static const DFN2GridSpec table[] = {
            { .sample_rate = 16000, .n_fft = 256, .n_erb = 24, .df_bins = 96,
              .df_order = DFN_TEST_DF_ORDER, .mask_lookahead = 0, .df_lookahead = 0,
              .frames_per_call = DFN_TEST_FRAMES_PER_CALL,
              .erb_norm_tau_sec = DFN_TEST_ERB_NORM_TAU_SEC,
              .erb_norm_init_lo_db = DFN_TEST_ERB_NORM_INIT_LO_DB,
              .erb_norm_init_hi_db = DFN_TEST_ERB_NORM_INIT_HI_DB,
              .erb_norm_scale_db = DFN_TEST_ERB_NORM_SCALE_DB,
              .spec_norm_tau_sec = DFN_TEST_SPEC_NORM_TAU_SEC,
              .spec_norm_init_lo = DFN_TEST_SPEC_NORM_INIT_LO,
              .spec_norm_init_hi = DFN_TEST_SPEC_NORM_INIT_HI },
            { .sample_rate = 32000, .n_fft = 512, .n_erb = 28, .df_bins = 96,
              .df_order = DFN_TEST_DF_ORDER, .mask_lookahead = 0, .df_lookahead = 0,
              .frames_per_call = DFN_TEST_FRAMES_PER_CALL,
              .erb_norm_tau_sec = DFN_TEST_ERB_NORM_TAU_SEC,
              .erb_norm_init_lo_db = DFN_TEST_ERB_NORM_INIT_LO_DB,
              .erb_norm_init_hi_db = DFN_TEST_ERB_NORM_INIT_HI_DB,
              .erb_norm_scale_db = DFN_TEST_ERB_NORM_SCALE_DB,
              .spec_norm_tau_sec = DFN_TEST_SPEC_NORM_TAU_SEC,
              .spec_norm_init_lo = DFN_TEST_SPEC_NORM_INIT_LO,
              .spec_norm_init_hi = DFN_TEST_SPEC_NORM_INIT_HI },
        };
        for (i = 0; i < 2; ++i) {
            DFN2Grid g;
            if (dfn2_grid_make(&g, &table[i]) != DFN2_GRID_OK)
                return 1;
            rc = run_other_grid_refusal(&g);
            if (rc) {
                fprintf(stderr, "other-grid refusal row (%d Hz / %d) failed: %d\n",
                        table[i].sample_rate, table[i].n_fft, rc);
                return 1;
            }
        }
        puts("dfn_rate_bridge: a 16 kHz or 32 kHz model grid is refused (the bridge needs 48 kHz / 1024)");
    }
    if (DFN2_SR != 48000 || DFN2_N_FFT != 1024) {
        /* built for another test grid (-DDFN2_GRID_KHZ=16|32 or an external grid): its own
         * grid is refused too, and the rows below (written for the 48 kHz model) do not apply */
        rc = run_other_grid_refusal(dfn_test_grid());
        if (rc) { fprintf(stderr, "other-grid refusal row failed: %d\n", rc); return 1; }
        puts("dfn_rate_bridge: PASS (this program's test grid is not 48 kHz / 1024: the bridge refuses it)");
        return 0;
    }
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
    rc = run_state_generation();
    if (rc) { fprintf(stderr, "rate bridge state generation row failed: %d\n", rc); return 1; }
    for (i = 0; i < N_GRIDS; ++i) {
        static const int shipped_delay[N_GRIDS] = { 672, 629, 330 };   /* the 1/1 pair, in native samples */
        rc = run_model_calls(&GRIDS[i], DFN2_MASK_LOOKAHEAD == 1 && DFN2_DF_LOOKAHEAD == 1 ? shipped_delay[i] : -1);
        if (rc) {
            fprintf(stderr, "rate bridge model-call row grid %d/%d failed: %d\n", GRIDS[i].sr, GRIDS[i].fft, rc);
            return 1;
        }
    }
    for (i = 0; i < N_GRIDS; ++i) {
        rc = run_reset_synthesis(&GRIDS[i]);
        if (rc) {
            fprintf(stderr, "rate bridge reset_synthesis grid %d/%d failed: %d\n",
                    GRIDS[i].sr, GRIDS[i].fft, rc);
            return 1;
        }
    }
    puts("dfn_rate_bridge: PASS");
    return 0;
}
