/* Tests of dfn/dfn_hop.h: the 48 kHz stage path and the rate-bridge path against direct
 * calls on a twin stage, the four phased behaviours (run, fail, left open, never fetched),
 * the sequence guards and reset. The unit ERB partition makes every fail-open path an exact
 * identity, so all four behaviours must agree bit for bit. Build and run: `make test` in
 * pipelines/. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn/dfn_hop.h"
#include "dfn/dfn_unit_partition.h"
#include "dfn/tests/dfn_test_fixture.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

#define MAXB 513

static float g_erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float g_erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static DFN2ModelIoDescriptor g_descriptor;

static uint32_t g_lcg;
static float unit(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)(g_lcg >> 8) / 16777216.0f - 0.5f;
}

static DfnResStageConfig stage_config(void) {
    DfnResStageConfig cfg = dfn_res_stage_default_config(DFN2_SR);
    cfg.erb_fwd = g_erb_fwd;
    cfg.erb_inv = g_erb_inv;
    cfg.model.infer = dfn_fixture_identity_infer;
    cfg.model.io_descriptor = &g_descriptor;
    return cfg;
}

/* One grid under test: 48 kHz drives the stage, 16 kHz/256 the bridge. */
typedef struct Rig {
    int bridge_mode, n, hop;
    DfnHop h;
    DfnResStage* stage;
    DfnRateBridge* bridge;
    AcIstft* istft;
    void* istft_mem;
    Complex* out_spec;
} Rig;

static int rig_init(Rig* r, int bridge_mode) {
    memset(r, 0, sizeof *r);
    r->bridge_mode = bridge_mode;
    if (bridge_mode) {
        DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(16000);
        bc.stage = stage_config();
        r->bridge = dfn_rate_bridge_create(&bc);
        if (!r->bridge) return -1;
        r->n = dfn_rate_bridge_n_freqs(r->bridge);
        r->hop = dfn_rate_bridge_hop_size(r->bridge);
        dfn_hop_init_bridge(&r->h, r->bridge);
    } else {
        DfnResStageConfig sc = stage_config();
        AcStftConfig scfg;
        size_t b;
        r->stage = dfn_res_stage_create(&sc);
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        b = ac_istft_get_mem_size(&scfg);
        if (posix_memalign(&r->istft_mem, 16, b) != 0) return -1;
        r->istft = ac_istft_init(r->istft_mem, b, &scfg);
        r->out_spec = (Complex*)calloc(DFN2_N_BINS, sizeof(Complex));
        if (!r->stage || !r->istft || !r->out_spec) return -1;
        r->n = DFN2_N_BINS;
        r->hop = DFN2_HOP_LEN;
        dfn_hop_init_stage(&r->h, r->stage, r->istft, r->out_spec);
    }
    return 0;
}

static void rig_free(Rig* r) {
    if (r->bridge) dfn_rate_bridge_destroy(r->bridge);
    if (r->stage) dfn_res_stage_destroy(r->stage);
    free(r->istft_mem);
    free(r->out_spec);
}

/* Frame-dependent E and a P that differs from it. */
static void make_frame(Complex* e, Complex* p, int n) {
    int k;
    for (k = 0; k < n; ++k) {
        e[k].r = unit() * 0.2f; e[k].i = unit() * 0.2f;
        p[k].r = e[k].r * 0.7f; p[k].i = e[k].i * 0.7f + 0.01f * unit();
    }
}

typedef enum { B_RUN, B_FAIL, B_OPEN, B_UNFETCHED } Behaviour;

/* One hop in the given phased behaviour; returns the dfn_hop_end status. */
static int phased(Rig* r, const Complex* e, const Complex* p, Behaviour b, float* out, int* frames) {
    DfnResStageJob job;
    int rc, f = 0;
    rc = dfn_hop_begin(&r->h, e, p);
    if (rc != 0) return rc;
    if (b != B_UNFETCHED) {
        while ((rc = dfn_hop_nn_begin(&r->h, &job)) == 1) {
            f++;
            if (b == B_OPEN) break;
            rc = dfn_hop_nn_end(&r->h, b == B_RUN ? dfn_res_stage_infer(dfn_hop_stage(&r->h), &job) : -1);
            if (rc != 0) return rc;
        }
        if (rc < 0) return rc;
    }
    if (frames) *frames = f;
    return dfn_hop_end(&r->h, out);
}

static void test_path(int bridge_mode) {
    enum { N = 40 };
    Rig a, ref;
    static Complex e[MAXB], p[MAXB];
    static float oa[MAXB], orf[MAXB];
    int h, k, bad_process = 0, bad_phased = 0, frames, max_frames = 0, energy_seen = 0;
    const char* tag = bridge_mode ? "bridge" : "stage";
    char msg[96];
    CHECK(rig_init(&a, bridge_mode) == 0 && rig_init(&ref, bridge_mode) == 0, "setup");
    if (!a.h.stage) return;

    /* dfn_hop_process against direct calls on a twin */
    g_lcg = 99u;
    for (h = 0; h < N; ++h) {
        make_frame(e, p, a.n);
        if (dfn_hop_process(&a.h, e, p, oa) != 0) bad_process++;
        if (bridge_mode) {
            if (dfn_rate_bridge_process(ref.bridge, e, p, orf) != 0) bad_process++;
        } else {
            if (dfn_res_stage_process(ref.stage, e, p, ref.out_spec) < 0) bad_process++;
            (void)ac_istft_push(ref.istft, ref.out_spec, orf);
        }
        if (memcmp(oa, orf, (size_t)a.hop * sizeof(float)) != 0) bad_process++;
        for (k = 0; k < a.hop; ++k) if (oa[k] != 0.0f) energy_seen = 1;
    }
    snprintf(msg, sizeof msg, "%s: dfn_hop_process equals direct calls bit for bit", tag);
    CHECK(bad_process == 0, msg);
    CHECK(energy_seen, "the output carries signal");

    /* the four phased behaviours produce the same hops on the unit partition */
    {
        static float seq[4][N][MAXB];
        long long fin[4], commits[4], skips[4];
        int bi;
        for (bi = 0; bi < 4; ++bi) {
            Rig r;
            CHECK(rig_init(&r, bridge_mode) == 0, "behaviour rig");
            g_lcg = 99u;
            for (h = 0; h < N; ++h) {
                make_frame(e, p, r.n);
                frames = 0;
                if (phased(&r, e, p, (Behaviour)bi, seq[bi][h], &frames) != 0) bad_phased++;
                if (frames > max_frames) max_frames = frames;
            }
            if (bridge_mode) {
                int last;
                dfn_rate_bridge_get_counters(r.bridge, &fin[bi], &last);
                commits[bi] = 0; skips[bi] = 0;
                dfn_res_stage_get_counters(dfn_hop_stage(&r.h), &fin[bi], &commits[bi], &skips[bi]);
            } else {
                dfn_res_stage_get_counters(r.stage, &fin[bi], &commits[bi], &skips[bi]);
            }
            rig_free(&r);
        }
        CHECK(bad_phased == 0, "phased calls succeed in every behaviour");
        CHECK(memcmp(seq[0], seq[1], sizeof seq[0]) == 0 && memcmp(seq[0], seq[2], sizeof seq[0]) == 0 &&
              memcmp(seq[0], seq[3], sizeof seq[0]) == 0,
              "run / fail / left open / never fetched all equal the identity on the unit partition");
        CHECK(commits[0] > 0 && commits[1] == 0 && commits[2] == 0 && commits[3] == 0,
              "only the run behaviour commits model frames");
        CHECK(max_frames >= 1 && max_frames <= (bridge_mode ? 2 : 1),
              "a hop carries at most one frame (two through the bridge)");
        (void)fin; (void)skips;
    }

    /* sequence guards */
    {
        DfnResStageJob job;
        Rig r;
        float out[MAXB];
        CHECK(rig_init(&r, bridge_mode) == 0, "guard rig");
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
        if (!bridge_mode) {   /* the stage validates its inputs; the bridge takes them into its WOLA */
            p[3].r = NAN;
            CHECK(dfn_hop_begin(&r.h, e, p) == DFN_HOP_INVALID, "a non-finite P is refused by the stage");
            dfn_hop_reset(&r.h);
            p[3].r = 0.0f;
            CHECK(dfn_hop_begin(&r.h, e, p) == 0 && dfn_hop_end(&r.h, out) == 0, "usable after reset");
        }
        rig_free(&r);
    }

    /* reset restores a fresh instance; check_finite does not change a valid stream */
    {
        Rig r, fresh;
        static float o1[MAXB], o2[MAXB];
        int bad_reset = 0;
        CHECK(rig_init(&r, bridge_mode) == 0 && rig_init(&fresh, bridge_mode) == 0, "reset rig");
        r.h.check_finite = 1;
        g_lcg = 5u;
        for (h = 0; h < 25; ++h) { make_frame(e, p, r.n); (void)dfn_hop_process(&r.h, e, p, o1); }
        dfn_hop_reset(&r.h);
        g_lcg = 77u;
        for (h = 0; h < 30; ++h) {
            make_frame(e, p, r.n);
            if (dfn_hop_process(&r.h, e, p, o1) != 0 || dfn_hop_process(&fresh.h, e, p, o2) != 0) bad_reset++;
            if (memcmp(o1, o2, (size_t)r.hop * sizeof(float)) != 0) bad_reset++;
        }
        CHECK(bad_reset == 0, "reset equals a fresh instance (and check_finite leaves a valid stream alone)");
        rig_free(&r);
        rig_free(&fresh);
    }
    rig_free(&a);
    rig_free(&ref);
}

int main(void) {
    dfn_unit_partition(g_erb_fwd, g_erb_inv);
    if (dfn2_model_io_descriptor_default(&g_descriptor) != 0) {
        printf("FAIL: model descriptor\n");
        return 1;
    }
    test_path(0);
    test_path(1);
    if (g_fail) { printf("%d check(s) FAILED\n", g_fail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
