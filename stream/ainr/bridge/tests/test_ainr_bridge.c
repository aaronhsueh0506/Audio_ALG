/* ainr_bridge acceptance test. The bridge embeds no model, so the model here is a delay of L frames
 * that hands back one lane's spectrum (the first L frames are zero, as a stage of lookahead L emits).
 * Rows:
 *   config      the grids and lookahead range accepted, everything else refused; descriptor gate and
 *               untouched pool on refusal;
 *   schedule    frames per hop (0,1,1,1 / 1,2) and the cumulative count; silence in, silence out;
 *   delay       the exact added delay equals the independent sum (prefill + group delays + (L + 1)
 *               frames) and an impulse in the apply lane lands there, for L = 0..3;
 *   lanes       the estimate lane reaches the output only through a model that reads it, and the two
 *               lanes are bit-for-bit symmetric;
 *   tolerance   a band-limited signal comes back as itself, delayed, within the resamplers' error;
 *   reset       N hops, reset, N hops equals a fresh instance;
 *   contract    the phased calls refuse every out-of-order use and change nothing when they do; an
 *               opened frame's out_spec starts zero;
 *   synthesis   ainr_bridge_reset_synthesis(): the host hops equal an independent audiolib synthesis
 *               that clears on that hop, the counters and the later output are untouched.
 * Standalone (audiolib only): no DFN2, no ERB tables. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr_bridge.h"
#include "audio_stft.h"
#include "fft_wrapper.h"

#define MAX_FFT 512
#define MAX_BINS (MAX_FFT / 2 + 1)
#define HOPS 200
#define MAX_L 3
#define PI_D 3.14159265358979323846

typedef struct Grid { int sr; int fft; } Grid;
static const Grid GRIDS[] = { {16000, 256}, {16000, 512}, {8000, 256} };
#define N_GRIDS 3

#define CHECK(cond, code) do { if (!(cond)) { fprintf(stderr, "  line %d: %s (row %d)\n", __LINE__, #cond, (code)); return (code); } } while (0)

/* ---- host-side analysis: frame = [previous hop, this hop] * sqrt-Hann ---------------------- */
typedef struct Analyzer {
    FftHandle *fft;
    int hop;
    float win[MAX_FFT], frame[MAX_FFT], prev[MAX_FFT / 2];
} Analyzer;

static int analyzer_init(Analyzer *a, int fft_n) {
    int i;
    a->fft = fft_create(fft_n);
    if (!a->fft) return -1;
    a->hop = fft_n / 2;
    ac_window_build(a->win, fft_n);
    for (i = 0; i < fft_n; ++i) a->frame[i] = 0.0f;
    memset(a->prev, 0, sizeof(a->prev));
    return 0;
}
static void analyzer_reset(Analyzer *a) { memset(a->prev, 0, sizeof(a->prev)); }
static void analyzer_push(Analyzer *a, const float *hop, Complex *spec) {
    int k;
    for (k = 0; k < a->hop; ++k) {
        a->frame[k] = a->prev[k] * a->win[k];
        a->frame[a->hop + k] = hop[k] * a->win[a->hop + k];
    }
    fft_forward(a->fft, a->frame, spec);
    memcpy(a->prev, hop, (size_t)a->hop * sizeof(float));
}

/* ---- the model: a delay of L frames on one lane ------------------------------------------- */
enum { LANE_APPLY, LANE_ESTIMATE, LANE_NONE };
typedef struct Model {
    int lane, lookahead, slot;
    Complex ring[MAX_L + 1][AINR_BRIDGE_MODEL_BINS];
} Model;

static void model_reset(Model *m, int lane, int lookahead) {
    memset(m, 0, sizeof(*m));
    m->lane = lane;
    m->lookahead = lookahead;
}

/* Answers one open frame: out = the lane's spectrum of L frames ago (zero before that). */
static void model_run(Model *m, const AinrBridgeFrame *f) {
    const Complex *in = m->lane == LANE_APPLY ? f->apply_spec : f->estimate_spec;
    Complex *slot = m->ring[m->slot];
    if (m->lane == LANE_NONE) return;
    if (m->lookahead == 0) {
        memcpy(f->out_spec, in, sizeof(Complex) * AINR_BRIDGE_MODEL_BINS);
        return;
    }
    memcpy(f->out_spec, slot, sizeof(Complex) * AINR_BRIDGE_MODEL_BINS);   /* the oldest */
    memcpy(slot, in, sizeof(Complex) * AINR_BRIDGE_MODEL_BINS);
    m->slot = (m->slot + 1) % m->lookahead;
}

static int run_hop(AinrBridge *b, Model *m, const Complex *se, const Complex *sa, float *out) {
    AinrBridgeFrame f;
    int r;
    if (ainr_bridge_begin(b, se, sa) != 0) return -1;
    while ((r = ainr_bridge_model_begin(b, &f)) == 1) {
        model_run(m, &f);
        if (ainr_bridge_model_end(b) != 0) return -1;
    }
    if (r < 0) return -1;
    return ainr_bridge_finish(b, out);
}

static float g_signal[HOPS * (MAX_FFT / 2)];
static float g_other[HOPS * (MAX_FFT / 2)];
static float g_out[HOPS * (MAX_FFT / 2)];
static float g_out_b[HOPS * (MAX_FFT / 2)];

/* Runs `hops` hops of est/app (NULL = silence) through b with model m; returns the total frames. */
static int run_stream(AinrBridge *b, Model *m, Analyzer *ae, Analyzer *ap, const float *est,
                      const float *app, int hops, float *out, int *schedule, int schedule_len) {
    static Complex se[MAX_BINS], sa[MAX_BINS];
    static float silence[MAX_FFT / 2];
    int hop = ae->hop, t, last;
    long long total = 0;
    memset(silence, 0, sizeof(silence));
    for (t = 0; t < hops; ++t) {
        analyzer_push(ae, est ? est + t * hop : silence, se);
        analyzer_push(ap, app ? app + t * hop : silence, sa);
        if (run_hop(b, m, se, sa, out + t * hop) != 0) return -1;
        ainr_bridge_get_counters(b, &total, &last);
        if (schedule && t < schedule_len) schedule[t] = last;
    }
    return (int)total;
}

/* prefill (the schedule's largest deficit, per grid) + 16 + 16 group delays, plus (L + 1) 48 kHz
 * hops of model delay, in sixths of a host sample. */
static int expected_delay_sixths(int sr, int hop, int lookahead) {
    int prefill = (sr == 16000 && hop == 128) ? 128 : (sr == 16000 && hop == 256) ? 85 : 42;
    return (prefill + 16 + 16) * 6 + (int)((long long)(lookahead + 1) * 512 * sr / 8000);
}

static AinrBridge *make(int sr, int fft, int lookahead) {
    AinrBridgeConfig cfg = ainr_bridge_default_config(sr, lookahead);
    cfg.fft_size = fft;
    return ainr_bridge_create(&cfg);
}

/* ---- config and descriptor rows ----------------------------------------------------------- */
static int run_config(void) {
    AinrBridgeConfig cfg;
    AinrBridgeMemReq req, bad, other;
    AinrBridge *b;
    unsigned char *snapshot;
    void *mem = NULL;
    int g, l;

    CHECK(sizeof(AinrBridgeMemReq) == 32u, 1);
    CHECK(ainr_bridge_get_mem_requirements(NULL, &req) != 0, 2);
    cfg = ainr_bridge_default_config(16000, 1);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, NULL) != 0, 3);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) == 0, 4);
    CHECK(req.descriptor_version == AINR_BRIDGE_DESCRIPTOR_VERSION &&
          req.layout_version == AINR_BRIDGE_LAYOUT_VERSION &&
          req.backend_id == AINR_BRIDGE_BACKEND_NE10 && req.alignment == 16u && req.reserved == 0u &&
          req.bytes > 0u && req.bytes % 16u == 0u, 5);
    for (g = 0; g < N_GRIDS; ++g)
        for (l = 0; l <= AINR_BRIDGE_MAX_LOOKAHEAD; ++l) {
            cfg = ainr_bridge_default_config(GRIDS[g].sr, l);
            cfg.fft_size = GRIDS[g].fft;
            CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) == 0, 6);
        }
    /* refused: rates, fft sizes the rate does not offer, lookahead outside the range */
    cfg = ainr_bridge_default_config(48000, 1);  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 10);
    cfg = ainr_bridge_default_config(44100, 1);  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 11);
    cfg = ainr_bridge_default_config(32000, 1);  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 12);
    cfg = ainr_bridge_default_config(8000, 1);   cfg.fft_size = 512;  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 13);
    cfg = ainr_bridge_default_config(16000, 1);  cfg.fft_size = 1024; CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 14);
    cfg = ainr_bridge_default_config(16000, 1);  cfg.fft_size = 128;  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 15);
    cfg = ainr_bridge_default_config(8000, 1);   cfg.fft_size = 128;  CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 16);
    cfg = ainr_bridge_default_config(16000, -1); CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 17);
    cfg = ainr_bridge_default_config(16000, AINR_BRIDGE_MAX_LOOKAHEAD + 1);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) != 0, 18);
    CHECK(ainr_bridge_create(&cfg) == NULL, 19);
    /* the descriptor tells the grids and the lookahead apart (a pool recorded for another is refused) */
    cfg = ainr_bridge_default_config(16000, 1);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &req) == 0, 20);
    cfg = ainr_bridge_default_config(16000, 2);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &other) == 0 && other.build_flags_hash != req.build_flags_hash, 21);
    cfg = ainr_bridge_default_config(8000, 1);
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &other) == 0 && other.build_flags_hash != req.build_flags_hash, 22);
    cfg = ainr_bridge_default_config(16000, 1); cfg.fft_size = 512;
    CHECK(ainr_bridge_get_mem_requirements(&cfg, &other) == 0 && other.build_flags_hash != req.build_flags_hash, 23);

    cfg = ainr_bridge_default_config(16000, 1);
    CHECK(posix_memalign(&mem, req.alignment, (size_t)req.bytes + 16u) == 0, 30);
    snapshot = (unsigned char *)malloc((size_t)req.bytes);
    CHECK(snapshot != NULL, 31);
    b = ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    CHECK(b != NULL, 32);
    ainr_bridge_destroy(b);
    memset(mem, 0xA5, (size_t)req.bytes);
    memcpy(snapshot, mem, (size_t)req.bytes);
    bad = req; bad.build_flags_hash ^= 1u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 33);
    bad = req; bad.layout_version += 1u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 34);
    bad = req; bad.backend_id += 1u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 35);
    bad = req; bad.bytes -= 16u;   /* a recorded size below what this build needs */
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 36);
    bad = req; bad.descriptor_version += 1u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 45);
    bad = req; bad.reserved = 1u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 46);
    bad = req; bad.alignment = 0u;
    CHECK(ainr_bridge_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, 47);
    CHECK(ainr_bridge_init(mem, (size_t)req.bytes - 16u, &cfg) == NULL, 37);
    CHECK(ainr_bridge_init((unsigned char *)mem + 8, (size_t)req.bytes, &cfg) == NULL, 38);   /* misaligned */
    CHECK(ainr_bridge_init(NULL, (size_t)req.bytes, &cfg) == NULL, 39);
    CHECK(memcmp(snapshot, mem, (size_t)req.bytes) == 0, 40);   /* a refusal leaves the pool as it was */
    b = ainr_bridge_init_ex(mem, (size_t)req.bytes + 16u, &cfg, &req);   /* a larger block is fine */
    CHECK(b != NULL, 41);
    free(snapshot);
    free(mem);
    printf("ainr_bridge: 16 kHz / fft 256 pool %llu bytes, descriptor hash %08x\n",
           (unsigned long long)req.bytes, req.build_flags_hash);
    /* NULL is safe everywhere */
    ainr_bridge_destroy(NULL);
    ainr_bridge_reset(NULL);
    CHECK(ainr_bridge_reset_synthesis(NULL) == -1 && ainr_bridge_hop_size(NULL) == -1 &&
          ainr_bridge_n_freqs(NULL) == -1 && ainr_bridge_added_delay_samples(NULL) == -1 &&
          ainr_bridge_get_host_hops(NULL, NULL, NULL) == -1, 42);
    {
        long long f = 5; int h = 5, n = 0, d = 0;
        ainr_bridge_get_counters(NULL, &f, &h);
        CHECK(f == 0 && h == 0, 43);
        CHECK(ainr_bridge_added_delay_exact(NULL, &n, &d) == -1, 44);
    }
    return 0;
}

/* ---- schedule, delay, lanes, tolerance, reset parity, per grid and lookahead -------------- */
static int run_grid(const Grid *g, int lookahead, int verbose) {
    AinrBridge *b = make(g->sr, g->fft, lookahead);
    Model m;
    Analyzer ae, ap;
    const int hop = g->fft / 2, n = HOPS * hop, impulse_at = 40 * hop + 7;
    int schedule[16], t, k, delay, num = 0, den = 0, frames, peak = -1;
    double exact, centroid = 0.0, mass = 0.0, near_e = 0.0, total_e = 0.0;
    float peak_abs = 0.0f, max_err = 0.0f;

    CHECK(b != NULL, 1);
    CHECK(ainr_bridge_hop_size(b) == hop && ainr_bridge_n_freqs(b) == g->fft / 2 + 1, 2);
    delay = ainr_bridge_added_delay_samples(b);
    CHECK(ainr_bridge_added_delay_exact(b, &num, &den) == 0 && den == AINR_BRIDGE_DELAY_DENOMINATOR &&
          num == expected_delay_sixths(g->sr, hop, lookahead) && delay == num / den, 3);
    exact = (double)num / (double)den;
    CHECK(analyzer_init(&ae, g->fft) == 0 && analyzer_init(&ap, g->fft) == 0, 4);

    /* schedule and cumulative count; silence in, silence out */
    model_reset(&m, LANE_APPLY, lookahead);
    frames = run_stream(b, &m, &ae, &ap, NULL, NULL, HOPS, g_out, schedule, 16);
    CHECK(frames == (int)(((long long)HOPS * hop * 48000 / g->sr) / 512), 5);
    for (t = 0; t < 16; ++t) {
        int want = (g->sr == 16000 && g->fft == 256) ? (t % 4 == 0 ? 0 : 1) : (t % 2 == 0 ? 1 : 2);
        CHECK(schedule[t] == want, 6);
    }
    for (k = 0; k < n; ++k) CHECK(g_out[k] == 0.0f, 7);

    /* impulse in the apply lane: lands hop + the exact added delay later (the hop is the host-style
     * WOLA the bridge performs on the lane); the round trip is a symmetric band-limited kernel, so its
     * centroid is that position to a fraction of a sample */
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_APPLY, lookahead);
    memset(g_signal, 0, sizeof(g_signal));
    g_signal[impulse_at] = 1.0f;
    CHECK(run_stream(b, &m, &ae, &ap, NULL, g_signal, HOPS, g_out, NULL, 0) >= 0, 8);
    for (k = 0; k < n; ++k) {
        float a = fabsf(g_out[k]);
        total_e += (double)a * a;
        if (a > peak_abs) { peak_abs = a; peak = k; }
    }
    CHECK(peak >= 24 && peak + 24 < n, 9);
    for (k = peak - 24; k <= peak + 24; ++k) { centroid += (double)k * g_out[k]; mass += g_out[k]; }
    centroid /= mass;
    if (fabs(centroid - (double)(impulse_at + hop) - exact) > 0.01 ||
        fabs((double)peak - (double)(impulse_at + hop) - exact) >= 0.5) {
        fprintf(stderr, "  %d/%d L=%d: impulse at %d -> peak %d centroid %.4f, expected %.4f\n", g->sr, g->fft,
                lookahead, impulse_at, peak, centroid, (double)(impulse_at + hop) + exact);
        return 10;
    }
    for (k = peak - 8; k <= peak + 8; ++k) near_e += (double)g_out[k] * g_out[k];
    CHECK(near_e >= 0.9 * total_e, 11);

    /* tolerance identity: three tones in the apply lane come back as themselves, delayed; the estimate
     * lane carries a different signal that an apply-lane model must ignore */
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_APPLY, lookahead);
    for (k = 0; k < n; ++k) {
        float ts = (float)k / (float)g->sr;
        g_signal[k] = 0.3f * sinf(2.0f * (float)PI_D * 300.0f * ts) + 0.2f * sinf(2.0f * (float)PI_D * 1000.0f * ts) +
                      0.1f * sinf(2.0f * (float)PI_D * 2500.0f * ts);
        g_other[k] = 0.5f * sinf(2.0f * (float)PI_D * 700.0f * ts);
    }
    CHECK(run_stream(b, &m, &ae, &ap, g_other, g_signal, HOPS, g_out, NULL, 0) >= 0, 12);
    for (k = 4 * hop + delay + 2 * g->fft; k < n; ++k) {
        double ts = ((double)k - (double)hop - exact) / (double)g->sr;
        double ref = 0.3 * sin(2.0 * PI_D * 300.0 * ts) + 0.2 * sin(2.0 * PI_D * 1000.0 * ts) +
                     0.1 * sin(2.0 * PI_D * 2500.0 * ts);
        float err = fabsf(g_out[k] - (float)ref);
        if (err > max_err) max_err = err;
    }
    if (verbose)
        printf("ainr_bridge: %d Hz / fft %d / L %d: added delay %d + %d/%d samples, round-trip max error %.2e\n",
               g->sr, g->fft, lookahead, delay, num - delay * den, den, (double)max_err);
    CHECK(max_err < 5e-3f, 13);
    /* the shipped added delays at lookahead 2, in samples of the host rate: 16 kHz / 256 and 512, 8 kHz / 256 */
    if (lookahead == 2) CHECK(delay == (g->sr == 8000 ? 330 : g->fft == 256 ? 672 : 629), 26);

    /* lanes: an apply-lane model never passes the estimate lane; an estimate-lane model passes it (and
     * not the apply lane); a model that answers nothing leaves silence */
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_APPLY, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, g_other, NULL, HOPS, g_out_b, NULL, 0) >= 0, 14);
    for (k = 0; k < n; ++k) CHECK(g_out_b[k] == 0.0f, 15);
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_ESTIMATE, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, NULL, g_signal, HOPS, g_out_b, NULL, 0) >= 0, 16);
    for (k = 0; k < n; ++k) CHECK(g_out_b[k] == 0.0f, 17);
    /* the lanes are symmetric: the tones in the estimate lane through an estimate-lane model come back
     * bit for bit as the tolerance run above (tones in the apply lane, apply-lane model) */
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_ESTIMATE, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, g_signal, g_other, HOPS, g_out_b, NULL, 0) >= 0, 18);
    CHECK(memcmp(g_out, g_out_b, (size_t)n * sizeof(float)) == 0, 19);
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_NONE, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, g_other, g_signal, HOPS, g_out_b, NULL, 0) >= 0, 20);
    for (k = 0; k < n; ++k) CHECK(g_out_b[k] == 0.0f, 21);

    /* reset parity: 40 hops, reset, 40 hops equals a fresh instance */
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_APPLY, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, g_other, g_signal, 40, g_out, NULL, 0) >= 0, 22);
    ainr_bridge_reset(b); analyzer_reset(&ae); analyzer_reset(&ap); model_reset(&m, LANE_APPLY, lookahead);
    CHECK(run_stream(b, &m, &ae, &ap, g_other, g_signal, 40, g_out + 40 * hop, NULL, 0) >= 0, 23);
    CHECK(memcmp(g_out, g_out + 40 * hop, (size_t)40 * hop * sizeof(float)) == 0, 24);
    {
        long long f; int lh;
        ainr_bridge_get_counters(b, &f, &lh);
        CHECK(f == (long long)((40LL * hop * 48000 / g->sr) / 512), 25);   /* the counters restart too */
    }

    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    ainr_bridge_destroy(b);
    return 0;
}

/* ---- the phased contract -------------------------------------------------------------------- */
/* Every hop of every grid: finish refuses, and changes nothing, while a due frame is unfetched
 * (before the hop's first frame) and while a frame is open (between a model_begin and its model_end);
 * the same hop then completes normally. `frames_by_hop` is the schedule's count for the hop (pinned in
 * run_grid), so the refusal is checked on exactly the hops where a frame is due. With the calibrated
 * prefill the output FIFO cannot serve a hop while a frame is due or open (probed: out_fill < hop at
 * every one of these points on all three grids), so finish's phase and fill guards and its FIFO guard
 * refuse the same calls; the guards are the contract, the FIFO guard is what a test can tell apart. */
static int frames_by_hop(const Grid *g, int t) {
    return (g->sr == 16000 && g->fft == 256) ? (t % 4 == 0 ? 0 : 1) : (t % 2 == 0 ? 1 : 2);
}

static int run_contract_grid(const Grid *g) {
    AinrBridge *b = make(g->sr, g->fft, 1);
    Analyzer ae, ap;
    Model m;
    static Complex se[MAX_BINS], sa[MAX_BINS];
    static float est[40 * (MAX_FFT / 2)], app[40 * (MAX_FFT / 2)], out[MAX_FFT / 2], first[MAX_FFT / 2];
    const int hop = g->fft / 2;
    int t, k, r;

    CHECK(b != NULL && analyzer_init(&ae, g->fft) == 0 && analyzer_init(&ap, g->fft) == 0, 1);
    model_reset(&m, LANE_APPLY, 1);
    for (k = 0; k < 40 * hop; ++k) {
        float ts = (float)k / (float)g->sr;
        app[k] = 0.3f * sinf(2.0f * (float)PI_D * 440.0f * ts);
        est[k] = 0.2f * sinf(2.0f * (float)PI_D * 900.0f * ts);
    }
    for (t = 0; t < 40; ++t) {
        AinrBridgeFrame f, f2;
        int n = frames_by_hop(g, t), done = 0;
        analyzer_push(&ae, est + t * hop, se);
        analyzer_push(&ap, app + t * hop, sa);
        CHECK(ainr_bridge_begin(b, se, sa) == 0, 2);
        CHECK(ainr_bridge_begin(b, se, sa) == -1, 3);                       /* a second begin */
        CHECK(ainr_bridge_model_end(b) == -1, 4);                           /* no frame open */
        CHECK(ainr_bridge_finish(b, NULL) == -1, 5);
        CHECK(ainr_bridge_reset_synthesis(b) == -1, 6);                     /* a hop in progress */
        CHECK(ainr_bridge_model_begin(b, NULL) == -1, 7);
        while (1) {
            int more = done < n;
            if (more) CHECK(ainr_bridge_finish(b, first) == -1, 17);        /* a due frame not yet fetched */
            r = ainr_bridge_model_begin(b, &f);
            CHECK(r == (more ? 1 : 0), 8);
            if (!more) break;
            CHECK(f.estimate_spec && f.apply_spec && f.out_spec, 9);
            CHECK(ainr_bridge_model_begin(b, &f2) == -1, 10);               /* a frame is open */
            CHECK(ainr_bridge_finish(b, first) == -1, 11);                  /* an open frame */
            CHECK(ainr_bridge_reset_synthesis(b) == -1, 12);
            for (k = 0; k < AINR_BRIDGE_MODEL_BINS; ++k)
                CHECK(f.out_spec[k].r == 0.0f && f.out_spec[k].i == 0.0f, 13);   /* out_spec opens zero */
            model_run(&m, &f);
            CHECK(ainr_bridge_model_end(b) == 0 && ainr_bridge_model_end(b) == -1, 14);
            ++done;
        }
        CHECK(ainr_bridge_finish(b, out) == 0 && ainr_bridge_finish(b, out) == -1, 15);
        CHECK(ainr_bridge_reset_synthesis(b) == 0, 16);                     /* between hops */
    }
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    ainr_bridge_destroy(b);
    return 0;
}

static int run_contract(void) {
    AinrBridge *b = make(16000, 256, 1);
    Analyzer ae, ap;
    Model m;
    static Complex se[MAX_BINS], sa[MAX_BINS];
    static float hopbuf[128], out[128];
    AinrBridgeFrame f;
    int g, k, r;
    const float *e0, *a0, *e1, *a1;

    /* nothing begun */
    CHECK(b != NULL && analyzer_init(&ae, 256) == 0 && analyzer_init(&ap, 256) == 0, 1);
    memset(hopbuf, 0, sizeof(hopbuf));
    analyzer_push(&ae, hopbuf, se);
    analyzer_push(&ap, hopbuf, sa);
    CHECK(ainr_bridge_model_begin(b, &f) == -1 && ainr_bridge_model_end(b) == -1 &&
          ainr_bridge_finish(b, out) == -1, 2);
    CHECK(ainr_bridge_begin(NULL, se, sa) == -1 && ainr_bridge_begin(b, NULL, sa) == -1 &&
          ainr_bridge_begin(b, se, NULL) == -1, 3);
    CHECK(ainr_bridge_model_begin(NULL, &f) == -1, 4);
    for (g = 0; g < N_GRIDS; ++g) {
        int rc = run_contract_grid(&GRIDS[g]);
        if (rc) { fprintf(stderr, "  %d/%d: contract row %d\n", GRIDS[g].sr, GRIDS[g].fft, rc); return 100 + rc; }
    }

    /* the next frame opens with a zero out_spec again, though the last one was written */
    CHECK(ainr_bridge_begin(b, se, sa) == 0, 24);
    r = ainr_bridge_model_begin(b, &f);
    CHECK(r >= 0, 25);
    if (r == 1) {
        for (k = 0; k < AINR_BRIDGE_MODEL_BINS; ++k) { f.out_spec[k].r = 3.0f; f.out_spec[k].i = -2.0f; }
        CHECK(ainr_bridge_model_end(b) == 0, 26);
    }
    CHECK(ainr_bridge_finish(b, out) == 0, 27);
    CHECK(ainr_bridge_begin(b, se, sa) == 0, 28);
    r = ainr_bridge_model_begin(b, &f);
    CHECK(r >= 0, 29);
    if (r == 1) {
        for (k = 0; k < AINR_BRIDGE_MODEL_BINS; ++k)
            CHECK(f.out_spec[k].r == 0.0f && f.out_spec[k].i == 0.0f, 30);
        CHECK(ainr_bridge_model_end(b) == 0, 31);
    }
    CHECK(ainr_bridge_finish(b, out) == 0, 32);

    /* the diagnostics tap: stable pointers, holding the hops of the last begin() (zero for silence,
     * the synthesised hop for a nonzero spectrum) */
    ainr_bridge_reset(b);
    model_reset(&m, LANE_APPLY, 1);
    CHECK(run_hop(b, &m, se, sa, out) == 0, 33);
    CHECK(ainr_bridge_get_host_hops(b, &e0, &a0) == 0 && e0 && a0, 34);
    CHECK(ainr_bridge_get_host_hops(b, NULL, NULL) == 0, 35);
    for (k = 0; k < 128; ++k) CHECK(e0[k] == 0.0f && a0[k] == 0.0f, 36);
    hopbuf[3] = 1.0f;
    analyzer_push(&ae, hopbuf, se);
    analyzer_push(&ap, hopbuf, sa);
    CHECK(run_hop(b, &m, se, sa, out) == 0, 37);
    CHECK(ainr_bridge_get_host_hops(b, &e1, &a1) == 0 && e1 == e0 && a1 == a0, 38);
    r = 0;
    for (k = 0; k < 128; ++k) if (e1[k] != 0.0f && a1[k] != 0.0f) r = 1;
    CHECK(r == 1, 39);

    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    ainr_bridge_destroy(b);
    return 0;
}

/* ---- bins beyond the limit ------------------------------------------------------------------ */
/* Every bin the model could be handed is finite and at most 2^68; a host block beyond that is refused with the
 * bridge idle; in-phase bins just under it, which the resamplers and windows push over the limit on the model
 * grid, cost the frames that carry them (zero out_spec) and nothing after. */
static int run_range_grid(const Grid *g) {
    AinrBridge *b = make(g->sr, g->fft, 1);
    Analyzer ae, ap;
    Model m;
    static Complex se[MAX_BINS], sa[MAX_BINS], big[MAX_BINS];
    static float sig[(MAX_FFT / 2) * 60], out[MAX_FFT / 2];
    const float limit = 2.9e20f;                 /* below 2^68 = 2.95e20 */
    const int hop = g->fft / 2, n_freqs = g->fft / 2 + 1;
    int t, k, r, handed_over_limit = 0, skipped_hop = 0, failed_after = 0, nonfinite_after = 0;

    CHECK(b != NULL && analyzer_init(&ae, g->fft) == 0 && analyzer_init(&ap, g->fft) == 0, 1);
    model_reset(&m, LANE_APPLY, 1);
    for (k = 0; k < 60 * hop; ++k) sig[k] = 0.3f * sinf(2.0f * (float)PI_D * 440.0f * (float)k / (float)g->sr);
    analyzer_push(&ae, sig, se);
    analyzer_push(&ap, sig, sa);
    /* refused blocks leave the bridge idle: the good block is taken at once */
    for (r = 0; r < 4; ++r) {
        memcpy(big, sa, sizeof(Complex) * (size_t)n_freqs);
        if (r == 0) big[3].r = (float)NAN;
        else if (r == 1) big[5].i = -(float)INFINITY;
        else if (r == 2) big[7].r = 1.0e21f;
        else big[n_freqs - 1].i = -3.0e20f;
        CHECK(ainr_bridge_begin(b, se, big) == -1 && ainr_bridge_begin(b, big, sa) == -1, 2);
    }
    CHECK(run_hop(b, &m, se, sa, out) == 0, 3);
    ainr_bridge_reset(b);
    model_reset(&m, LANE_APPLY, 1);
    for (t = 0; t < 60; ++t) {
        AinrBridgeFrame f;
        analyzer_push(&ae, sig + t * hop, big);
        if (t == 20)
            for (k = 0; k < n_freqs; ++k) { big[k].r = (k & 1) ? -limit : limit; big[k].i = (k & 1) ? limit : -limit; }
        analyzer_push(&ap, sig + t * hop, sa);
        if (ainr_bridge_begin(b, big, sa) != 0) { failed_after += t >= 20; continue; }
        while ((r = ainr_bridge_model_begin(b, &f)) == 1) {
            for (k = 0; k < AINR_BRIDGE_MODEL_BINS; ++k) {
                float v[4] = { f.estimate_spec[k].r, f.estimate_spec[k].i, f.apply_spec[k].r, f.apply_spec[k].i };
                int j;
                for (j = 0; j < 4; ++j)
                    if (!(fabsf(v[j]) <= 2.95147905e20f)) handed_over_limit = 1;
            }
            model_run(&m, &f);
            CHECK(ainr_bridge_model_end(b) == 0, 4);
        }
        if (r < 0 || ainr_bridge_finish(b, out) != 0) { failed_after += t >= 20; continue; }
        if (t == 20) skipped_hop = 1;
        for (k = 0; k < hop; ++k) if (t > 26 && !isfinite(out[k])) nonfinite_after = 1;
    }
    CHECK(handed_over_limit == 0, 5);               /* the model is never handed a frame beyond the limit */
    CHECK(skipped_hop == 1 && failed_after == 0, 6);   /* the hop completes and so does every hop after it */
    CHECK(nonfinite_after == 0, 7);
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    ainr_bridge_destroy(b);
    return 0;
}

/* ---- reset_synthesis ------------------------------------------------------------------------ */
#define RS_HOPS 100
#define RS_AT 41   /* odd: the 48 kHz FIFO holds leftover samples at the start of this hop on every grid that has them */
enum { RS_CLEARED, RS_NEVER, RS_LATE, RS_FULL_RESET };
typedef struct RsRun {
    float out[RS_HOPS * (MAX_FFT / 2)], tap_e[RS_HOPS * (MAX_FFT / 2)], tap_p[RS_HOPS * (MAX_FFT / 2)];
    long long frames;
} RsRun;
static Complex rs_spec_e[RS_HOPS][MAX_BINS], rs_spec_p[RS_HOPS][MAX_BINS];
static RsRun rs_run[4];
static float rs_orc_cleared_e[RS_HOPS * (MAX_FFT / 2)], rs_orc_cleared_p[RS_HOPS * (MAX_FFT / 2)];
static float rs_orc_never_e[RS_HOPS * (MAX_FFT / 2)], rs_orc_never_p[RS_HOPS * (MAX_FFT / 2)];

static int rs_drive(const Grid *g, int mode, RsRun *run) {
    AinrBridge *b = make(g->sr, g->fft, 1);
    Model m;
    int hop = g->fft / 2, t, last;
    CHECK(b != NULL, 1);
    model_reset(&m, LANE_APPLY, 1);
    for (t = 0; t < RS_HOPS; ++t) {
        const float *te, *tp;
        AinrBridgeFrame f;
        int r;
        if (t == RS_AT && mode == RS_CLEARED) CHECK(ainr_bridge_reset_synthesis(b) == 0, 2);
        if (t == RS_AT && mode == RS_FULL_RESET) { ainr_bridge_reset(b); model_reset(&m, LANE_APPLY, 1); }
        CHECK(ainr_bridge_begin(b, rs_spec_e[t], rs_spec_p[t]) == 0, 3);
        CHECK(ainr_bridge_get_host_hops(b, &te, &tp) == 0 && te && tp, 4);
        memcpy(run->tap_e + t * hop, te, (size_t)hop * sizeof(float));
        memcpy(run->tap_p + t * hop, tp, (size_t)hop * sizeof(float));
        if (t == RS_AT && mode == RS_CLEARED) {
            CHECK(ainr_bridge_reset_synthesis(b) == -1, 5);   /* a hop in progress: nothing cleared */
            CHECK(memcmp(te, run->tap_e + t * hop, (size_t)hop * sizeof(float)) == 0, 6);
        }
        while ((r = ainr_bridge_model_begin(b, &f)) == 1) {
            model_run(&m, &f);
            CHECK(ainr_bridge_model_end(b) == 0, 7);
        }
        CHECK(r == 0, 8);
        CHECK(ainr_bridge_finish(b, run->out + t * hop) == 0, 9);
        if (t == RS_AT && mode == RS_LATE) CHECK(ainr_bridge_reset_synthesis(b) == 0, 10);
    }
    ainr_bridge_get_counters(b, &run->frames, &last);
    ainr_bridge_destroy(b);
    return 0;
}

/* The oracle: audiolib's default-window synthesis of the same spectra, cleared before `clear_at`. */
static int rs_oracle(int fft, const Complex (*spec)[MAX_BINS], int clear_at, float *out) {
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

static int run_reset_synthesis(const Grid *g, int verbose) {
    const int hop = g->fft / 2;
    int t, k, added, horizon, first_diff = -1, last_diff = -1, tap_diff_hops = 0;
    Analyzer ae, ap;
    static float est[RS_HOPS * (MAX_FFT / 2)], app[RS_HOPS * (MAX_FFT / 2)];
    float tap_delta = 0.0f;
    double out_energy = 0.0, tap_energy = 0.0, ratio;
    AinrBridge *probe = make(g->sr, g->fft, 1);

    CHECK(probe && analyzer_init(&ae, g->fft) == 0 && analyzer_init(&ap, g->fft) == 0, 1);
    added = ainr_bridge_added_delay_samples(probe);
    ainr_bridge_destroy(probe);
    for (k = 0; k < RS_HOPS * hop; ++k) {
        float ts = (float)k / (float)g->sr;
        app[k] = 0.3f * sinf(2.0f * (float)PI_D * 300.0f * ts) + 0.2f * sinf(2.0f * (float)PI_D * 1000.0f * ts) +
                 0.1f * sinf(2.0f * (float)PI_D * 2500.0f * ts);
        est[k] = 0.5f * sinf(2.0f * (float)PI_D * 700.0f * ts);
    }
    for (t = 0; t < RS_HOPS; ++t) {
        analyzer_push(&ae, est + t * hop, rs_spec_e[t]);
        analyzer_push(&ap, app + t * hop, rs_spec_p[t]);
    }
    fft_destroy(ae.fft);
    fft_destroy(ap.fft);
    CHECK(rs_oracle(g->fft, rs_spec_e, RS_AT, rs_orc_cleared_e) == 0 &&
          rs_oracle(g->fft, rs_spec_p, RS_AT, rs_orc_cleared_p) == 0 &&
          rs_oracle(g->fft, rs_spec_e, -1, rs_orc_never_e) == 0 &&
          rs_oracle(g->fft, rs_spec_p, -1, rs_orc_never_p) == 0, 2);
    for (k = 0; k < 4; ++k) {
        int rc = rs_drive(g, k, &rs_run[k]);
        if (rc) { fprintf(stderr, "  %d/%d: drive %d failed: %d\n", g->sr, g->fft, k, rc); return 3; }
    }
    /* tap: cleared == oracle that clears on the hop; never == oracle that never clears */
    CHECK(memcmp(rs_run[RS_CLEARED].tap_p, rs_orc_cleared_p, (size_t)RS_HOPS * hop * sizeof(float)) == 0 &&
          memcmp(rs_run[RS_CLEARED].tap_e, rs_orc_cleared_e, (size_t)RS_HOPS * hop * sizeof(float)) == 0, 4);
    CHECK(memcmp(rs_run[RS_NEVER].tap_p, rs_orc_never_p, (size_t)RS_HOPS * hop * sizeof(float)) == 0 &&
          memcmp(rs_run[RS_NEVER].tap_e, rs_orc_never_e, (size_t)RS_HOPS * hop * sizeof(float)) == 0, 5);
    /* the two runs differ on the reset hop and on no other hop */
    for (t = 0; t < RS_HOPS; ++t) {
        int same = memcmp(rs_run[RS_CLEARED].tap_p + t * hop, rs_run[RS_NEVER].tap_p + t * hop,
                          (size_t)hop * sizeof(float)) == 0 &&
                   memcmp(rs_run[RS_CLEARED].tap_e + t * hop, rs_run[RS_NEVER].tap_e + t * hop,
                          (size_t)hop * sizeof(float)) == 0;
        if (!same) {
            ++tap_diff_hops;
            CHECK(t == RS_AT, 6);
            for (k = 0; k < hop; ++k) {
                float d = fabsf(rs_run[RS_CLEARED].tap_p[t * hop + k] - rs_run[RS_NEVER].tap_p[t * hop + k]);
                if (d > tap_delta) tap_delta = d;
            }
        }
    }
    CHECK(tap_diff_hops == 1 && tap_delta > 0.01f, 7);   /* the stimulus must carry a real tail */
    /* timing: one hop late the reset hop is not cleared, and the following hop is wrong too */
    CHECK(memcmp(rs_run[RS_LATE].tap_p + RS_AT * hop, rs_orc_cleared_p + RS_AT * hop, (size_t)hop * sizeof(float)) != 0, 8);
    CHECK(memcmp(rs_run[RS_LATE].tap_p + (RS_AT + 1) * hop, rs_orc_cleared_p + (RS_AT + 1) * hop,
                 (size_t)hop * sizeof(float)) != 0, 9);
    /* kept: the counters, and the output of the cleared run against the uncleared one */
    CHECK(rs_run[RS_CLEARED].frames == rs_run[RS_NEVER].frames, 10);
    CHECK(rs_run[RS_FULL_RESET].frames != rs_run[RS_NEVER].frames, 11);   /* the counter would see a full reset */
    for (k = 0; k < RS_HOPS * hop; ++k)
        if (rs_run[RS_CLEARED].out[k] != rs_run[RS_NEVER].out[k]) {
            if (first_diff < 0) first_diff = k;
            last_diff = k;
        }
    /* The bridge is a unit-gain linear path for this band-limited stimulus, so the output difference
     * carries the energy of the cleared tail and no more: a clear that also drops the 48 kHz synthesis
     * tail or the resamplers' history adds a second perturbation. */
    for (k = 0; k < RS_HOPS * hop; ++k) {
        double d = (double)rs_run[RS_CLEARED].out[k] - (double)rs_run[RS_NEVER].out[k];
        double dt = (double)rs_run[RS_CLEARED].tap_p[k] - (double)rs_run[RS_NEVER].tap_p[k];
        out_energy += d * d;
        tap_energy += dt * dt;
    }
    ratio = out_energy / tap_energy;
    if (!(ratio > 0.99 && ratio < 1.01)) {
        fprintf(stderr, "  %d/%d: output difference carries %.4f of the cleared tail's energy\n", g->sr, g->fft, ratio);
        return 12;
    }
    horizon = RS_AT * hop + added + 4 * g->fft;
    if (first_diff < RS_AT * hop || last_diff < 0 || last_diff >= horizon) {
        fprintf(stderr, "  %d/%d: output differs over [%d, %d], expected inside [%d, %d)\n",
                g->sr, g->fft, first_diff, last_diff, RS_AT * hop, horizon);
        return 13;
    }
    CHECK(horizon + 4 * hop < RS_HOPS * hop, 14);   /* a drained tail must be observed */
    if (verbose)
        printf("ainr_bridge: %d Hz / fft %d: reset_synthesis on hop %d: host hops bit-exact to the audiolib oracle, "
               "output differs over samples [%d, %d] (delay %d) with %.4f of the cleared tail's energy, counters unchanged\n",
               g->sr, g->fft, RS_AT, first_diff, last_diff, added, ratio);
    return 0;
}

int main(void) {
    int g, l, rc;
    rc = run_config();
    if (rc) { fprintf(stderr, "ainr_bridge: config rows FAIL (%d)\n", rc); return 1; }
    for (g = 0; g < N_GRIDS; ++g)
        for (l = 0; l <= MAX_L; ++l) {
            rc = run_grid(&GRIDS[g], l, 1);
            if (rc) { fprintf(stderr, "ainr_bridge: %d/%d L=%d FAIL (%d)\n", GRIDS[g].sr, GRIDS[g].fft, l, rc); return 1; }
        }
    rc = run_contract();
    if (rc) { fprintf(stderr, "ainr_bridge: contract rows FAIL (%d)\n", rc); return 1; }
    for (g = 0; g < N_GRIDS; ++g) {
        rc = run_range_grid(&GRIDS[g]);
        if (rc) { fprintf(stderr, "ainr_bridge: %d/%d range rows FAIL (%d)\n", GRIDS[g].sr, GRIDS[g].fft, rc); return 1; }
    }
    for (g = 0; g < N_GRIDS; ++g) {
        rc = run_reset_synthesis(&GRIDS[g], 1);
        if (rc) { fprintf(stderr, "ainr_bridge: %d/%d reset_synthesis FAIL (%d)\n", GRIDS[g].sr, GRIDS[g].fft, rc); return 1; }
    }
    printf("ainr_bridge: PASS\n");
    return 0;
}
