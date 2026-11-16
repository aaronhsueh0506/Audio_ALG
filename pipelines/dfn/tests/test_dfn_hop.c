/* Tests of dfn/dfn_hop.h and dfn/dfn_hop_pool.h: the direct path (host on the model grid) and the
 * bridge path (16 kHz host, 48 kHz model) against twins built from the layers underneath, the four
 * phased behaviours (run, fail, left open, never fetched), the sequence guards, the refusals of the
 * pool plan and reset.
 *
 * Two twins drive the same spectra with a model whose heads depend on the windows it is shown, bit
 * for bit: a hand-written recipe on a second dfn2_stream (plain scalar scaling, dfn2_stream_*_freq,
 * and for the bridge path ainr_bridge_* with an explicit frame loop), and the stage and the rate
 * bridge that the archive still ships (dfn_res_stage.h, dfn_rate_bridge.h), which the hop replaced.
 * The unit ERB partition makes every fail-open path an exact identity, so the four behaviours must
 * agree bit for bit; a half-mask model then shows that only the run behaviour commits. Build and
 * run: `make test` in pipelines/. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn/dfn_hop.h"
#include "dfn/dfn_hop_pool.h"
#include "dfn/dfn_rate_bridge.h"
#include "dfn/dfn_res_stage.h"
#include "dfn/dfn_unit_partition.h"
#include "dfn/tests/dfn_test_fixture.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

#define MAXB 513
#define BRIDGE_RATE 16000   /* 16 kHz / 256: 129 bins, hop 128 */
#define BRIDGE_BINS 129
#define BRIDGE_HOP 128

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static DFN2ModelIoDescriptor g_descriptor;

static uint32_t g_lcg;
static float unit(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)(g_lcg >> 8) / 16777216.0f - 0.5f;
}

/* Models. The counting identity: unit mask, zero deep filter; its calls are counted. The half mask:
 * a uniform 0.5 band mask. The hash model: heads that depend on the windows it is shown (by bit
 * pattern) and on its own call count, so a wrong scale, window or order is a different output. */
static long long g_model_calls;
static int counting_identity_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    ++g_model_calls;
    return dfn_fixture_identity_infer(user, in, out);
}

static uint32_t hash_step(uint32_t h, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    return (h ^ bits) * 16777619u;
}

/* Per-context state of the hash model: its call count and what it does on chosen calls. */
typedef struct HashUser {
    uint32_t calls;
    int fail_mode;   /* 0 none; 1 reports failure on calls 10, 11 and 25; 2 leaves the heads unwritten there */
} HashUser;

static int hash_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    HashUser* u = (HashUser*)user;
    uint32_t call = u->calls++;
    uint32_t h = call * 2654435761u;
    size_t i;
    for (i = 0; i < in->erb_window_elements; ++i) h = hash_step(h, in->erb_window[i]);
    for (i = 0; i < in->spec_window_elements; ++i) h = hash_step(h, in->spec_window[i]);
    if (u->fail_mode && (call == 10 || call == 11 || call == 25)) return u->fail_mode == 1 ? -1 : 0;
    for (i = 0; i < out->erb_mask_elements; ++i) {
        h = h * 1664525u + 1013904223u;
        out->erb_mask[i] = 0.2f + 0.7f * (float)((h >> 8) & 0xffff) / 65535.0f;
    }
    for (i = 0; i < out->coefs_elements; ++i) {
        h = h * 1664525u + 1013904223u;
        out->coefs[i] = 0.05f * (float)((int)((h >> 8) & 0xffff) - 32768) / 32768.0f;
    }
    for (i = 0; i < out->alpha_elements; ++i) {
        h = h * 1664525u + 1013904223u;
        out->alpha[i] = 0.5f * (float)((h >> 8) & 0xffff) / 65535.0f;
    }
    return 0;
}

/* Writes every head, then reports failure: the hop must take the frame as the identity whatever was written. */
static int half_mask_then_fail_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    (void)dfn_fixture_half_mask_infer(user, in, out);
    return -1;
}

/* A callback that ran nothing, and one whose run failed. */
static int not_started_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    (void)user; (void)in; (void)out;
    return STREAM_RUN_NOT_STARTED;
}
static int failed_run_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    (void)user; (void)in; (void)out;
    return STREAM_RUN_FAILED;
}

static DFN2Model model_of(int (*infer)(void*, const DFN2PrepostInputs*, DFN2PrepostOutputs*), void* user) {
    DFN2Model m;
    memset(&m, 0, sizeof m);
    m.user = user;
    m.infer = infer;
    m.io_descriptor = &g_descriptor;
    return m;
}

static int aligned_alloc16(void** p, size_t n) {
    return posix_memalign(p, 16, n ? n : 16) == 0 ? 0 : -1;
}

/* One hop under test, built from the pool plan. */
typedef struct Rig {
    int n, hop;
    DfnHop h;
    void* pool;
} Rig;

static void plan_for(DfnHopPoolPlan* plan, DFN2Model model) {
    dfn_hop_pool_plan_init(plan, dfn_test_grid());
    plan->stream_cfg.erb_fwd = g_erb_fwd;
    plan->stream_cfg.erb_inv = g_erb_inv;
    plan->model = model;
}

/* The check_finite argument rig_init_grid() hands to dfn_hop_pool_build(). */
static int g_build_check_finite;

/* host_rate == the model grid's: the direct path; else the bridge on that host grid (fft 0: the rate's default). */
static int rig_init_grid(Rig* r, int host_rate, int fft, DFN2Model model) {
    DfnHopPoolPlan plan;
    memset(r, 0, sizeof *r);
    plan_for(&plan, model);
    if (host_rate == DFN2_SR) {
        r->n = DFN2_N_BINS;
        r->hop = DFN2_HOP_LEN;
    } else {
        r->n = (fft ? fft : 256) / 2 + 1;
        r->hop = (fft ? fft : 256) / 2;
    }
    if (dfn_hop_pool_plan(&plan, host_rate, fft) != 0) return -1;
    if (aligned_alloc16(&r->pool, dfn_hop_pool_bytes(&plan)) != 0) return -1;
    return dfn_hop_pool_build(&plan, (unsigned char*)r->pool, r->n, r->hop, g_build_check_finite, &r->h);
}

static int rig_init(Rig* r, int bridge_mode, DFN2Model model) {
    return rig_init_grid(r, bridge_mode ? BRIDGE_RATE : DFN2_SR, 0, model);
}

static void rig_free(Rig* r) {
    free(r->pool);
    r->pool = NULL;
}

/* Frame-dependent E and a P that differs from it. */
static void make_frame(Complex* e, Complex* p, int n) {
    int k;
    for (k = 0; k < n; ++k) {
        e[k].r = unit() * 0.2f; e[k].i = unit() * 0.2f;
        p[k].r = e[k].r * 0.7f; p[k].i = e[k].i * 0.7f + 0.01f * unit();
    }
}

/* --- twin 1: the recipe written out on a second stream ------------------------------------------ */

typedef struct Twin {
    DFN2Stream* stream;
    void* pool;
    float w[6 * MAXB];
    int n;
    float scale, unscale;
} Twin;

static int twin_init(Twin* t) {
    DFN2StreamConfig c;
    DFN2StreamMemReq req;
    memset(t, 0, sizeof *t);
    memset(&c, 0, sizeof c);
    c.grid = *dfn_test_grid();
    c.erb_fwd = g_erb_fwd;
    c.erb_inv = g_erb_inv;
    t->n = c.grid.n_bins;
    t->scale = dfn2_stft_scale(c.grid.n_fft);
    t->unscale = dfn2_stft_unscale(c.grid.n_fft);
    if (dfn2_stream_get_mem_size(&c, &req) != 0 || aligned_alloc16(&t->pool, (size_t)req.bytes) != 0) return -1;
    t->stream = dfn2_stream_init_ex(&c, t->pool, (size_t)req.bytes, &req);
    return t->stream ? 0 : -1;
}

/* One model frame by the documented recipe: scale, pre, model (only when the hop has a job), post, unscale.
 * Zero when no spectrum is due. */
static int twin_frame(Twin* t, const Complex* e, const Complex* p, DFN2Model model, Complex* out) {
    float *er = t->w, *ei = t->w + t->n, *pr = t->w + 2 * t->n, *pi_ = t->w + 3 * t->n;
    float *or_ = t->w + 4 * t->n, *oi = t->w + 5 * t->n;
    DFN2HopIO io;
    int k, valid = 0;
    for (k = 0; k < t->n; ++k) {
        er[k] = e[k].r * t->scale; ei[k] = e[k].i * t->scale;
        pr[k] = p[k].r * t->scale; pi_[k] = p[k].i * t->scale;
    }
    if (dfn2_stream_pre_process_freq_dual(t->stream, er, ei, pr, pi_, &io) != 1) return -1;
    if (dfn2_stream_post_process_freq(t->stream,
                                      io.job_ready ? model.infer(model.user, &io.inputs, &io.outputs)
                                                   : STREAM_RUN_NOT_STARTED,
                                      or_, oi, &valid) != 0)
        return -1;
    for (k = 0; k < t->n; ++k) {
        out[k].r = valid ? or_[k] * t->unscale : 0.0f;
        out[k].i = valid ? oi[k] * t->unscale : 0.0f;
    }
    return 0;
}

static void twin_free(Twin* t) {
    free(t->pool);
    t->pool = NULL;
}

/* --- twin 2: the stage and the rate bridge the archive still ships ------------------------------ */

static DfnResStageConfig legacy_stage_config(DFN2Model model) {
    DfnResStageConfig cfg = dfn_res_stage_default_config(dfn_test_grid());
    cfg.erb_fwd = g_erb_fwd;
    cfg.erb_inv = g_erb_inv;
    cfg.model = model;
    return cfg;
}

enum { SC_PLAIN, SC_ATTEN_START, SC_MODEL_FAIL, SC_UNWRITTEN, SC_RESET, SC_ATTEN_CHANGE, SC_COUNT };
static const char* const SC_NAME[SC_COUNT] = { "plain", "attenuation limit 12 dB from the start",
    "model reports failure (3 frames)", "heads left unwritten (3 frames)", "reset at hop 20",
    "attenuation change at hop 20" };

/* The hop under test against both twins over `hops` hops with the hash model (each context has its
 * own call counter) on one host grid and scenario. Returns the number of mismatching hops. */
static int compare_with_twins(int host_rate, int fft, int scenario, int hops, int* energy_seen, int* first_nonzero) {
    const int bridge_mode = host_rate != DFN2_SR;
    Rig a;
    Twin t;
    HashUser ua = {0, 0}, ut = {0, 0}, ul = {0, 0};
    Complex e[MAXB], p[MAXB], tmp[MAXB];
    static float oa[MAXB], ot[MAXB], ol[MAXB];
    int bad = 0, h, k;
    DfnResStage* stage = NULL;
    DfnRateBridge* bridge = NULL;
    AcIstft* istft_t = NULL;
    AcIstft* istft_l = NULL;
    void* istft_mem_t = NULL;
    void* istft_mem_l = NULL;
    AinrBridge* tb = NULL;
    void* tb_mem = NULL;
    DFN2Model mt;
    DfnResStageConfig lcfg;
    AcStftConfig scfg;
    size_t isz;

    if (scenario == SC_MODEL_FAIL) ua.fail_mode = ut.fail_mode = ul.fail_mode = 1;
    if (scenario == SC_UNWRITTEN) ua.fail_mode = ut.fail_mode = ul.fail_mode = 2;
    mt = model_of(hash_infer, &ut);
    lcfg = legacy_stage_config(model_of(hash_infer, &ul));
    *energy_seen = 0;
    *first_nonzero = -1;
    if (rig_init_grid(&a, host_rate, fft, model_of(hash_infer, &ua)) != 0 || twin_init(&t) != 0) return 1000;
    if (bridge_mode) {
        DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(host_rate, dfn_test_grid());
        AinrBridgeConfig tc = ainr_bridge_default_config(host_rate, dfn_hop_lookahead_frames(dfn_test_grid()));
        AinrBridgeMemReq treq;
        bc.fft_size = fft;
        tc.fft_size = fft;
        bc.stage = lcfg;
        bridge = dfn_rate_bridge_create(&bc);
        if (!bridge || ainr_bridge_get_mem_requirements(&tc, &treq) != 0 ||
            aligned_alloc16(&tb_mem, (size_t)treq.bytes) != 0)
            return 1000;
        tb = ainr_bridge_init_ex(tb_mem, (size_t)treq.bytes, &tc, &treq);
        if (!tb) return 1000;
    } else {
        stage = dfn_res_stage_create(&lcfg);
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        isz = ac_istft_get_mem_size(&scfg);
        if (!stage || aligned_alloc16(&istft_mem_t, isz) != 0 || aligned_alloc16(&istft_mem_l, isz) != 0) return 1000;
        istft_t = ac_istft_init(istft_mem_t, isz, &scfg);
        istft_l = ac_istft_init(istft_mem_l, isz, &scfg);
        if (!istft_t || !istft_l) return 1000;
    }
    if (scenario == SC_ATTEN_START) {
        if (dfn_hop_set_atten_lim(&a.h, 12.0f) != 0 || dfn2_stream_set_atten_lim(t.stream, 12.0f) != 0 ||
            dfn_res_stage_set_atten_lim(bridge ? dfn_rate_bridge_stage(bridge) : stage, 12.0f) != 0)
            return 1000;
    }

    g_lcg = 99u;
    for (h = 0; h < hops; ++h) {
        int bad_here = 0;
        make_frame(e, p, a.n);
        if (h == 20 && scenario == SC_RESET) {
            dfn_hop_reset(&a.h);
            dfn2_stream_reset(t.stream);
            if (bridge_mode) { ainr_bridge_reset(tb); dfn_rate_bridge_reset(bridge); }
            else { ac_istft_reset(istft_t); dfn_res_stage_reset(stage); ac_istft_reset(istft_l); }
        }
        if (h == 20 && scenario == SC_ATTEN_CHANGE) {
            if (dfn_hop_set_atten_lim(&a.h, 9.0f) != 0 || dfn2_stream_set_atten_lim(t.stream, 9.0f) != 0 ||
                dfn_res_stage_set_atten_lim(bridge ? dfn_rate_bridge_stage(bridge) : stage, 9.0f) != 0)
                bad_here = 1;
        }
        if (dfn_hop_process(&a.h, e, p, oa) != 0) bad_here = 1;
        if (bridge_mode) {
            /* twin 1: bridge + stream with an explicit frame loop; twin 2: the shipped rate bridge */
            AinrBridgeFrame f;
            int r;
            if (ainr_bridge_begin(tb, e, p) != 0) bad_here = 1;
            while ((r = ainr_bridge_model_begin(tb, &f)) == 1) {
                if (twin_frame(&t, f.estimate_spec, f.apply_spec, mt, f.out_spec) != 0) bad_here = 1;
                if (ainr_bridge_model_end(tb) != 0) bad_here = 1;
            }
            if (r < 0 || ainr_bridge_finish(tb, ot) != 0) bad_here = 1;
            if (dfn_rate_bridge_process(bridge, e, p, ol) != 0) bad_here = 1;
        } else {
            if (twin_frame(&t, e, p, mt, tmp) != 0) bad_here = 1;
            (void)ac_istft_push(istft_t, tmp, ot);
            if (dfn_res_stage_process(stage, e, p, tmp) < 0) bad_here = 1;
            (void)ac_istft_push(istft_l, tmp, ol);
        }
        if (memcmp(oa, ot, (size_t)a.hop * sizeof(float)) != 0) bad_here = 1;
        if (memcmp(oa, ol, (size_t)a.hop * sizeof(float)) != 0) bad_here = 1;
        for (k = 0; k < a.hop; ++k)
            if (oa[k] != 0.0f) {
                *energy_seen = 1;
                if (*first_nonzero < 0) *first_nonzero = h;
            }
        bad += bad_here;
    }
    if (ua.calls != ut.calls || ua.calls != ul.calls) ++bad;   /* every context ran the model equally often */
    if (bridge) dfn_rate_bridge_destroy(bridge);
    if (stage) dfn_res_stage_destroy(stage);
    free(istft_mem_t); free(istft_mem_l); free(tb_mem);
    twin_free(&t);
    rig_free(&a);
    return bad;
}

typedef enum { B_RUN, B_FAIL, B_OPEN, B_UNFETCHED } Behaviour;

/* One hop in the given phased behaviour; returns the dfn_hop_end status. */
static int phased(Rig* r, const Complex* e, const Complex* p, Behaviour b, float* out, int* frames) {
    DfnHopJob job;
    int rc, f = 0;
    rc = dfn_hop_begin(&r->h, e, p);
    if (rc != 0) return rc;
    if (b != B_UNFETCHED) {
        while ((rc = dfn_hop_nn_begin(&r->h, &job)) == 1) {
            f++;
            if (b == B_OPEN) { (void)dfn_hop_infer(&r->h, &job); break; }   /* heads written, frame left open */
            rc = dfn_hop_nn_end(&r->h, b == B_RUN ? dfn_hop_infer(&r->h, &job) : STREAM_RUN_FAILED);
            if (rc != 0) return rc;
        }
        if (rc < 0) return rc;
    }
    if (frames) *frames = f;
    return dfn_hop_end(&r->h, out);
}

/* The four behaviours over N hops with `infer`; seq[b][h] receives the hops. run_frames receives the
 * frames the B_RUN behaviour opened, run_calls the model runs and gens[b] the state generation behaviour b
 * ended on. */
enum { N_BEH = 40 };
static int run_behaviours(int bridge_mode, int (*infer)(void*, const DFN2PrepostInputs*, DFN2PrepostOutputs*),
                          float seq[4][N_BEH][MAXB], long long* run_frames, long long* run_calls,
                          unsigned gens[4], int* max_frames, int* first_hop_frames) {
    Complex e[MAXB], p[MAXB];
    int bi, h, bad = 0, frames;
    *max_frames = 0;
    for (bi = 0; bi < 4; ++bi) {
        Rig r;
        if (rig_init(&r, bridge_mode, model_of(infer, NULL)) != 0) return 1000;
        g_lcg = 99u;
        g_model_calls = 0;
        if (bi == B_RUN) *run_frames = 0;
        for (h = 0; h < N_BEH; ++h) {
            make_frame(e, p, r.n);
            frames = 0;
            if (phased(&r, e, p, (Behaviour)bi, seq[bi][h], &frames) != 0) bad++;
            if (frames > *max_frames) *max_frames = frames;
            if (bi == B_RUN) {
                *run_frames += frames;
                if (h == 0) *first_hop_frames = frames;
            }
        }
        if (bi == B_RUN) *run_calls = g_model_calls;
        gens[bi] = dfn_hop_state_generation(&r.h);
        rig_free(&r);
    }
    return bad;
}

static void test_path(int bridge_mode) {
    int bad_process, energy_seen, first_nonzero, h, k;
    const char* tag = bridge_mode ? "bridge" : "stage";
    char msg[112];
    Complex e[MAXB], p[MAXB];

    /* dfn_hop_process against both twins, bit for bit, on a model that reads its windows: every host
     * grid of the path, every scenario */
    {
        static const int grids[4][2] = { { DFN2_SR, 0 }, { 16000, 256 }, { 16000, 512 }, { 8000, 256 } };
        int g, sc;
        for (g = bridge_mode ? 1 : 0; g < (bridge_mode ? 4 : 1); ++g)
            for (sc = 0; sc < SC_COUNT; ++sc) {
                bad_process = compare_with_twins(grids[g][0], grids[g][1], sc, 60, &energy_seen, &first_nonzero);
                snprintf(msg, sizeof msg, "%s %d Hz, %s: dfn_hop_process equals the stream recipe and the shipped stage / bridge bit for bit",
                         tag, grids[g][0], SC_NAME[sc]);
                CHECK(bad_process == 0, msg);
                CHECK(energy_seen, "the output carries signal");
                /* the model's lookahead is silent: output hop f is the frame built L hops earlier */
                CHECK(bridge_mode || sc == SC_RESET ||
                      first_nonzero == dfn_hop_lookahead_frames(dfn_test_grid()),
                      "direct path: the first lookahead hops are exactly zero, the next one carries signal");
            }
    }

    /* the four phased behaviours produce the same hops on the unit partition */
    {
        static float seq[4][N_BEH][MAXB], seq_half[4][N_BEH][MAXB];
        long long run_frames, run_calls, rf2, rc2;
        unsigned gens[4], gens2[4];
        int max_frames, first_hop_frames = -1, mf2, fh2, bad;
        bad = run_behaviours(bridge_mode, counting_identity_infer, seq, &run_frames, &run_calls, gens,
                             &max_frames, &first_hop_frames);
        CHECK(bad == 0, "phased calls succeed in every behaviour");
        CHECK(memcmp(seq[0], seq[1], sizeof seq[0]) == 0 && memcmp(seq[0], seq[2], sizeof seq[0]) == 0 &&
              memcmp(seq[0], seq[3], sizeof seq[0]) == 0,
              "run / fail / left open / never fetched all equal the identity on the unit partition");
        CHECK(max_frames >= 1 && max_frames <= (bridge_mode ? 2 : 1),
              "a hop carries at most one frame (two through the bridge)");
        /* the direct path opens one model frame per hop from hop mask_lookahead on, none before */
        CHECK(bridge_mode || (first_hop_frames == (DFN2_MASK_LOOKAHEAD > 0 ? 0 : 1) && max_frames == 1),
              "direct path: one model frame per hop, from hop mask_lookahead");
        CHECK(run_calls == run_frames && run_frames > 0, "the model runs once per frame it is shown");
        CHECK(bridge_mode || run_frames == N_BEH - DFN2_MASK_LOOKAHEAD, "direct path: one model run per hop with a frame");
        CHECK(gens[B_RUN] == 1u,
              "the state generation stays 1 (no reset, no refused commit, no run that failed)");
        /* the run status decides the generation: a failed run (STREAM_RUN_FAILED) and a frame left open move it
         * once per real frame, a frame never fetched (STREAM_RUN_NOT_STARTED) not at all */
        CHECK(gens[B_UNFETCHED] == gens[B_RUN],
              "a frame never fetched is STREAM_RUN_NOT_STARTED: the generation is the run behaviour's");
        CHECK(gens[B_FAIL] > gens[B_RUN] && gens[B_OPEN] > gens[B_RUN],
              "a reported failure and a frame left open advance the generation");
        CHECK(bridge_mode || (gens[B_FAIL] == 1u + (unsigned)(N_BEH - DFN2_MASK_LOOKAHEAD) && gens[B_OPEN] == gens[B_FAIL]),
              "direct path: one step per model frame for a failure and for a frame left open");

        /* a model that changes the spectrum: only the run behaviour commits its heads */
        bad = run_behaviours(bridge_mode, dfn_fixture_half_mask_infer, seq_half, &rf2, &rc2, gens2, &mf2, &fh2);
        CHECK(bad == 0, "phased calls succeed with the half-mask model");
        CHECK(memcmp(seq_half[B_RUN], seq[B_RUN], sizeof seq_half[0]) != 0,
              "the half-mask model changes the output of the run behaviour");
        CHECK(memcmp(seq_half[B_FAIL], seq[B_RUN], sizeof seq[0]) == 0 &&
              memcmp(seq_half[B_OPEN], seq[B_RUN], sizeof seq[0]) == 0 &&
              memcmp(seq_half[B_UNFETCHED], seq[B_RUN], sizeof seq[0]) == 0,
              "fail / left open / never fetched stay the identity whatever the model would have written");
        /* the model's own result decides: written heads with a failure code are not committed */
        bad = run_behaviours(bridge_mode, half_mask_then_fail_infer, seq_half, &rf2, &rc2, gens2, &mf2, &fh2);
        CHECK(bad == 0, "phased calls succeed with a model that writes and then fails");
        CHECK(memcmp(seq_half[B_RUN], seq[B_RUN], sizeof seq_half[0]) == 0,
              "a model that writes its heads and reports failure is the identity, not the half mask");
    }

    /* sequence guards */
    {
        DfnHopJob job;
        Rig r;
        float out[MAXB];
        CHECK(rig_init(&r, bridge_mode, model_of(counting_identity_infer, NULL)) == 0, "guard rig");
        make_frame(e, p, r.n);
        CHECK(dfn_hop_nn_begin(&r.h, &job) == DFN_HOP_SEQUENCE_ERROR, "nn_begin before begin");
        CHECK(dfn_hop_nn_end(&r.h, 0) == DFN_HOP_SEQUENCE_ERROR, "nn_end without an open frame");
        CHECK(dfn_hop_end(&r.h, out) == DFN_HOP_SEQUENCE_ERROR, "end before begin");
        CHECK(dfn_hop_begin(NULL, e, p) == DFN_HOP_INVALID && dfn_hop_begin(&r.h, NULL, p) == DFN_HOP_INVALID &&
              dfn_hop_begin(&r.h, e, NULL) == DFN_HOP_INVALID, "NULL arguments");
        CHECK(dfn_hop_begin(&r.h, e, p) == 0, "begin");
        CHECK(dfn_hop_begin(&r.h, e, p) == DFN_HOP_SEQUENCE_ERROR, "a second begin in the same hop");
        CHECK(dfn_hop_end(&r.h, NULL) == DFN_HOP_INVALID, "end with a NULL output");
        CHECK(dfn_hop_end(&r.h, out) == 0, "end");
        {   /* a refused block leaves the hop idle, on the direct path and through the bridge alike */
            unsigned gen = dfn_hop_state_generation(&r.h);
            p[3].r = NAN;
            CHECK(dfn_hop_begin(&r.h, e, p) == DFN_HOP_INVALID, "a non-finite P is refused");
            p[3].r = 1e21f;
            CHECK(dfn_hop_begin(&r.h, e, p) == DFN_HOP_INVALID, "a bin beyond 2^63 at model scale is refused");
            e[5].i = 1e21f;
            p[3].r = 0.0f;
            CHECK(dfn_hop_begin(&r.h, e, p) == DFN_HOP_INVALID, "so is such a bin in E");
            CHECK(dfn_hop_state_generation(&r.h) == gen, "a refused block leaves the model state untouched");
            e[5].i = 0.0f;
            CHECK(dfn_hop_begin(&r.h, e, p) == 0 && dfn_hop_end(&r.h, out) == 0,
                  "the hop is usable after a refusal, with no reset");
        }
        rig_free(&r);
    }

    /* reset restores a fresh instance; check_finite does not change a valid stream */
    {
        Rig r, fresh;
        static float o1[MAXB], o2[MAXB];
        int bad_reset = 0;
        HashUser c1 = {0, 0}, c2 = {0, 0};
        CHECK(rig_init(&r, bridge_mode, model_of(hash_infer, &c1)) == 0 &&
              rig_init(&fresh, bridge_mode, model_of(hash_infer, &c2)) == 0, "reset rig");
        r.h.check_finite = 1;
        g_lcg = 5u;
        for (h = 0; h < 25; ++h) { make_frame(e, p, r.n); (void)dfn_hop_process(&r.h, e, p, o1); }
        dfn_hop_reset(&r.h);
        c1.calls = 0;   /* the model's own counter is the callback's to reset */
        g_lcg = 77u;
        for (h = 0; h < 30; ++h) {
            make_frame(e, p, r.n);
            if (dfn_hop_process(&r.h, e, p, o1) != 0 || dfn_hop_process(&fresh.h, e, p, o2) != 0) bad_reset++;
            if (memcmp(o1, o2, (size_t)r.hop * sizeof(float)) != 0) bad_reset++;
        }
        CHECK(bad_reset == 0, "reset equals a fresh instance (and check_finite leaves a valid stream alone)");
        for (k = 0; k < r.hop; ++k) if (o1[k] != 0.0f) break;
        CHECK(k < r.hop, "the compared stream carries signal");
        rig_free(&r);
        rig_free(&fresh);
    }
}

/* Writes a finite mask far beyond DFN2_PREPOST_HEAD_LIMIT: frame_finish refuses it, so the hop takes the identity. */
static int huge_mask_infer(void* user, const DFN2PrepostInputs* in, DFN2PrepostOutputs* out) {
    size_t i;
    (void)user;
    (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 3e30f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    memset(out->alpha, 0, out->alpha_elements * sizeof(float));
    return 0;
}

/* An enormous finite head never reaches the output, with or without check_finite: the library refuses the commit
 * (state restarted, generation advanced), the hop takes the identity and every output stays finite. */
static void test_check_finite(void) {
    int pass;
    for (pass = 0; pass < 2; ++pass) {
        Rig r;
        Complex e[MAXB], p[MAXB];
        float out[MAXB];
        int h, k, refused = 0, inf_seen = 0;
        unsigned gen0;
        g_build_check_finite = pass;   /* through the pool build, as the 4-channel example asks for it */
        CHECK(rig_init(&r, 0, model_of(huge_mask_infer, NULL)) == 0, "finite rig");
        g_build_check_finite = 0;
        CHECK(r.h.check_finite == pass, "the pool build hands check_finite to the hop");
        gen0 = dfn_hop_state_generation(&r.h);
        g_lcg = 11u;
        for (h = 0; h < 12; ++h) {
            make_frame(e, p, r.n);
            for (k = 0; k < r.n; ++k) { e[k].r *= 1e13f; e[k].i *= 1e13f; p[k].r *= 1e13f; p[k].i *= 1e13f; }
            if (dfn_hop_process(&r.h, e, p, out) != 0) ++refused;
            else for (k = 0; k < r.hop; ++k) if (!isfinite(out[k])) inf_seen = 1;
        }
        CHECK(refused == 0 && !inf_seen, "an enormous finite mask is refused by the library: the output is finite");
        CHECK(dfn_hop_state_generation(&r.h) > gen0, "each refused commit advances the state generation");
        rig_free(&r);
    }
}

/* model.reset runs on dfn_hop_reset, and only then. */
static int g_reset_calls;
static void counting_reset(void* user) { (void)user; ++g_reset_calls; }

static void test_model_hooks(void) {
    Rig r;
    DFN2Model m = model_of(counting_identity_infer, NULL);
    Complex e[MAXB], p[MAXB];
    float out[MAXB];
    DfnHopPoolPlan plan;
    DFN2Model foreign;
    DFN2ModelIoDescriptor d;
    m.reset = counting_reset;
    g_reset_calls = 0;
    CHECK(rig_init(&r, 0, m) == 0, "hook rig");
    CHECK(dfn_hop_state_generation(&r.h) == 1u, "a fresh hop starts at state generation 1");
    g_lcg = 3u;
    make_frame(e, p, r.n);
    CHECK(dfn_hop_process(&r.h, e, p, out) == 0 && g_reset_calls == 0, "processing does not call model.reset");
    {   /* the first mask_lookahead hops carry no model frame and do not move the generation */
        DfnHopJob job;
        int hh;
        for (hh = 1; hh < DFN2_MASK_LOOKAHEAD; ++hh) {
            make_frame(e, p, r.n);
            CHECK(dfn_hop_process(&r.h, e, p, out) == 0, "a hop without a model frame closes");
        }
        CHECK(dfn_hop_state_generation(&r.h) == 1u, "the state generation does not move on hops without a model frame");
        make_frame(e, p, r.n);
        CHECK(dfn_hop_begin(&r.h, e, p) == 0 && dfn_hop_nn_begin(&r.h, &job) == 1 &&
              job.state_generation == dfn_hop_state_generation(&r.h),
              "the job carries the generation the accelerator's state must match");
        CHECK(dfn_hop_nn_end(&r.h, dfn_hop_infer(&r.h, &job)) == 0 && dfn_hop_end(&r.h, out) == 0, "the hop closes");
        {
            const unsigned before = dfn_hop_state_generation(&r.h);
            dfn_hop_reset(&r.h);
            CHECK(g_reset_calls == 1, "dfn_hop_reset calls model.reset once");
            CHECK(dfn_hop_state_generation(&r.h) == before + 1u, "a reset advances the state generation by one");
        }
    }
    rig_free(&r);

    {   /* the run status of the model callback decides the generation: no model and a callback that ran nothing
         * (STREAM_RUN_NOT_STARTED) leave it, a callback whose run failed moves it once per real frame */
        static const int HOPS = 10;
        int k, h;
        for (k = 0; k < 3; ++k) {
            DFN2Model mm;
            DfnHopJob job;
            memset(&mm, 0, sizeof mm);
            if (k == 1) mm = model_of(not_started_infer, NULL);
            if (k == 2) mm = model_of(failed_run_infer, NULL);
            CHECK(rig_init(&r, 0, mm) == 0, "status rig");
            g_lcg = 5u;
            for (h = 0; h < HOPS; ++h) {
                make_frame(e, p, r.n);
                CHECK(dfn_hop_process(&r.h, e, p, out) == 0, "a hop with a failing or absent model still closes");
            }
            CHECK(dfn_hop_state_generation(&r.h) ==
                  1u + (k == 2 ? (unsigned)(HOPS - DFN2_MASK_LOOKAHEAD) : 0u),
                  "model status: absent and not-started leave the generation, a failed run advances it per model frame");
            make_frame(e, p, r.n);
            CHECK(dfn_hop_begin(&r.h, e, p) == 0 && dfn_hop_nn_begin(&r.h, &job) == 1, "status rig: a frame opens");
            CHECK(dfn_hop_infer(&r.h, &job) == (k == 0 ? STREAM_RUN_NOT_STARTED : k == 1 ? STREAM_RUN_NOT_STARTED
                                                                                      : STREAM_RUN_FAILED),
                  "dfn_hop_infer reports the callback's status, STREAM_RUN_NOT_STARTED without one");
            rig_free(&r);
        }
    }

    /* the pool plan refuses what the shipped stage refused */
    plan_for(&plan, model_of(counting_identity_infer, NULL));
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == 0, "plan: the shipped configuration");
    {   /* the plan's stream is a freq_only one: smaller than the stream with the PCM entries (the Twin's kind) */
        DFN2StreamConfig full = plan.stream_cfg, slim = plan.stream_cfg;
        DFN2StreamMemReq full_req, slim_req;
        full.freq_only = 0;
        slim.freq_only = 1;
        CHECK(dfn2_stream_get_mem_size(&full, &full_req) == 0 && dfn2_stream_get_mem_size(&slim, &slim_req) == 0 &&
              plan.stream_bytes == (size_t)slim_req.bytes && slim_req.bytes < full_req.bytes &&
              memcmp(&plan.stream_req, &slim_req, sizeof(slim_req)) == 0,
              "plan: the model stream is sized without its PCM half");
    }
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, DFN2_N_FFT * 2) == -1, "plan: another fft size on the model's own rate");
    plan_for(&plan, model_of(counting_identity_infer, NULL));
    plan.stream_cfg.erb_inv = NULL;
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: a missing ERB matrix");
    plan.stream_cfg.erb_fwd = NULL;   /* neither: the stream's file mode is not a plan */
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: no ERB matrix at all");
    {   /* an inverse matrix that is not a partition of unity (one bin off by a relative 1e-4): the plan
         * sizes it, the build refuses it; the same pool builds with the matrix restored */
        void* pool = NULL;
        DfnHop hop;
        plan_for(&plan, model_of(counting_identity_infer, NULL));
        CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == 0 &&
              aligned_alloc16(&pool, dfn_hop_pool_bytes(&plan)) == 0, "partition rig");
        g_erb_inv[7] *= 1.0001f;
        CHECK(dfn_hop_pool_build(&plan, (unsigned char*)pool, DFN2_N_BINS, DFN2_HOP_LEN, 0, &hop) == -1,
              "build: an inverse ERB matrix that is not a partition of unity");
        dfn_unit_partition(g_erb_fwd, g_erb_inv, dfn_test_grid());
        CHECK(dfn_hop_pool_build(&plan, (unsigned char*)pool, DFN2_N_BINS, DFN2_HOP_LEN, 0, &hop) == 0,
              "build: the same pool with the partition restored");
        CHECK(dfn_hop_pool_build(&plan, (unsigned char*)pool, DFN2_N_BINS + 1, DFN2_HOP_LEN, 0, &hop) == -1 &&
              dfn_hop_pool_build(&plan, (unsigned char*)pool, DFN2_N_BINS, DFN2_HOP_LEN / 2, 0, &hop) == -1,
              "build: a front-end grid that is not the model's");
        free(pool);
    }
    plan_for(&plan, model_of(counting_identity_infer, NULL));
    plan.stream_cfg.atten_lim_db = NAN;
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: a non-finite attenuation limit");
    {   /* no model: nothing else can refuse a grid the stream itself accepts */
        DFN2GridSpec spec = dfn_test_spec_of(dfn_test_grid());
        DFN2Grid packed;
        spec.frames_per_call = 2;
        if (dfn2_grid_make(&packed, &spec) == DFN2_GRID_OK) {
            DFN2Model none;
            memset(&none, 0, sizeof none);
            plan_for(&plan, none);
            plan.stream_cfg.grid = packed;
            CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: more than one frame per call");
        } else {
            CHECK(0, "the library makes a two-frame grid");
        }
    }
    plan_for(&plan, model_of(counting_identity_infer, NULL));
    plan.model.io_descriptor = NULL;
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: a model that infers without a descriptor");
    plan_for(&plan, model_of(NULL, NULL));
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: no model but a descriptor");
    plan.model.io_descriptor = NULL;
    CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == 0, "plan: no model at all is the identity");
    {   /* a model built for another grid, lookahead pair or frame count, one external field at a time */
        int which;
        for (which = 0; which < 5; ++which) {
            if (dfn_fixture_foreign_descriptor(dfn_test_grid(), &d, which) != 0) continue;
            foreign = model_of(counting_identity_infer, NULL);
            foreign.io_descriptor = &d;
            plan_for(&plan, foreign);
            CHECK(dfn_hop_pool_plan(&plan, DFN2_SR, 0) == -1, "plan: a model whose descriptor is for another grid");
        }
    }
    {   /* the bridge needs the 48 kHz / 1024 model and a host grid it offers */
        plan_for(&plan, model_of(counting_identity_infer, NULL));
        if (DFN2_SR == 48000 && DFN2_N_FFT == 1024) {
            CHECK(dfn_hop_pool_plan(&plan, 16000, 0) == 0 && plan.use_bridge, "plan: 16 kHz host through the bridge");
            CHECK(dfn_hop_pool_plan(&plan, 16000, 512) == 0, "plan: 16 kHz / 512");
            CHECK(dfn_hop_pool_plan(&plan, 16000, 1024) == -1, "plan: a host fft the bridge does not offer");
            CHECK(dfn_hop_pool_plan(&plan, 32000, 0) == -1, "plan: a host rate the bridge does not offer");
            {   /* the front end's grid must be the bridge's */
                void* pool = NULL;
                DfnHop hop;
                CHECK(dfn_hop_pool_plan(&plan, 16000, 256) == 0 &&
                      aligned_alloc16(&pool, dfn_hop_pool_bytes(&plan)) == 0, "bridge build rig");
                CHECK(dfn_hop_pool_build(&plan, (unsigned char*)pool, BRIDGE_BINS + 1, BRIDGE_HOP, 0, &hop) == -1 &&
                      dfn_hop_pool_build(&plan, (unsigned char*)pool, BRIDGE_BINS, BRIDGE_HOP * 2, 0, &hop) == -1,
                      "build: a front-end grid that is not the bridge's");
                CHECK(dfn_hop_pool_build(&plan, (unsigned char*)pool, BRIDGE_BINS, BRIDGE_HOP, 0, &hop) == 0 &&
                      hop.bridge != NULL && hop.istft == NULL, "build: the bridge's own grid");
                free(pool);
            }
        }
    }
    {   /* a model grid other than 48 kHz / 1024 has no bridge, whatever the test grid is */
        static float fwd[MAXB * 32], inv[32 * MAXB];
        DFN2GridSpec spec = dfn_test_spec(16000, 256, 24, 96, 0, 0);
        DFN2Grid g16;
        if (dfn2_grid_make(&g16, &spec) == DFN2_GRID_OK) {
            dfn_unit_partition(fwd, inv, &g16);
            dfn_hop_pool_plan_init(&plan, &g16);
            plan.stream_cfg.erb_fwd = fwd;
            plan.stream_cfg.erb_inv = inv;
            CHECK(dfn_hop_pool_plan(&plan, 16000, 0) == 0 && !plan.use_bridge, "plan: a 16 kHz model on a 16 kHz host");
            CHECK(dfn_hop_pool_plan(&plan, 8000, 0) == -1, "plan: no bridge for a model grid other than 48 kHz / 1024");
        }
    }
}

/* The delay helper is dfn2_prepost_latency_frames(), and both inits refuse a grid of more than one frame per
 * call (and NULL) without writing the hop. */
static void test_frames_per_call_guard(void) {
    DFN2Grid one = *dfn_test_grid();
    DFN2Grid three = one;
    DFN2Model model = model_of(NULL, NULL);
    static float planar[6 * MAXB];
    static Complex spec[MAXB];
    DfnHop h, before;
    three.frames_per_call = 3;
    CHECK(dfn_hop_lookahead_frames(&one) == one.mask_lookahead + one.df_lookahead, "lookahead: L + D at N = 1");
    CHECK(dfn_hop_lookahead_frames(&three) == dfn2_prepost_latency_frames(&three) &&
          dfn_hop_lookahead_frames(&three) == three.mask_lookahead + three.df_lookahead + 2,
          "lookahead: dfn2_prepost_latency_frames (L + D + N - 1) at N = 3");
    CHECK(dfn_hop_lookahead_frames(NULL) == 0, "lookahead: 0 for NULL");

    memset(&h, 0xA5, sizeof h);
    before = h;
    CHECK(dfn_hop_init_direct(&h, NULL, &three, &model, NULL, spec, planar) == -1 &&
          memcmp(&h, &before, sizeof h) == 0, "init_direct: N = 3 refused, hop untouched");
    CHECK(dfn_hop_init_bridge(&h, NULL, &three, &model, NULL, planar) == -1 &&
          memcmp(&h, &before, sizeof h) == 0, "init_bridge: N = 3 refused, hop untouched");
    CHECK(dfn_hop_init_direct(&h, NULL, NULL, &model, NULL, spec, planar) == -1 &&
          dfn_hop_init_bridge(&h, NULL, &one, NULL, NULL, planar) == -1 &&
          dfn_hop_init_direct(NULL, NULL, &one, &model, NULL, spec, planar) == -1,
          "init: NULL hop, grid or model refused");
    CHECK(dfn_hop_init_direct(&h, NULL, &one, &model, NULL, spec, planar) == 0 && h.n_bins == one.n_bins,
          "init_direct: N = 1 accepted");
    CHECK(dfn_hop_init_bridge(&h, NULL, &one, &model, NULL, planar) == 0 && h.n_bins == one.n_bins,
          "init_bridge: N = 1 accepted");
}

int main(void) {
    dfn_unit_partition(g_erb_fwd, g_erb_inv, dfn_test_grid());
    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &g_descriptor) != 0) {
        printf("FAIL: model descriptor\n");
        return 1;
    }
    dfn_test_report_pair("dfn_hop", -1);
    test_path(0);
    if (DFN2_SR == 48000 && DFN2_N_FFT == 1024) test_path(1);
    else printf("dfn_hop: the test grid is not 48 kHz / 1024, so the bridge path (which needs it) is skipped\n");
    test_model_hooks();
    test_check_finite();
    test_frames_per_call_guard();
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
