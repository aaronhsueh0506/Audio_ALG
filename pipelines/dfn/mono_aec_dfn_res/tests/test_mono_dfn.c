/* mono AEC -> RES -> DeepFilterNet2 composition test: lib/aec's Aec with its own RES gain,
 * P = E * min(1, G_res) (+ comfort noise) and the DFN2 hop (dfn/dfn_hop.h), wired as
 * dfn/mono_aec_dfn_res/main.c wires them.
 * 48 kHz: with the identity model the composition equals the hand-written mono chain
 * (tests/conv_chain_mono.h with enable_nr=0), delayed by exactly DFN_RES_STAGE_LOOKAHEAD_FRAMES
 * hops (two for the shipped 1/1 pair), byte for byte,
 * comfort noise OFF and ON. The echo (384-sample delay, matched estimator) must make the
 * estimator report a delay change (the "no reset on a delay change" contract) and the RES cut
 * (min G_res < 0.9).
 * 8/16 kHz: the stage runs through the bridge; the output equals the NR-off output delayed by
 * the bridge's added delay within the resamplers' pass-band error, a near-end impulse pins the
 * delay, and the per-hop stage schedule is the documented one.
 * The phased hop (begin / nn_begin / nn_end / end) against the callback-driven one, in the four
 * behaviours run / fail / left open / never fetched. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"           /* aec_cng_fill */
#include "dfn/dfn_hop.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn/tests/dfn_test_fixture.h"
#include "mono_aec_nr_res/tests/conv_chain_mono.h"

#define TEST_FRAMES 64
#define CNG_NR_GAIN_FLOOR 0.31622777f
#define CNG_RNG_SEED 0x9e3779b9u
#define ECHO_DELAY 384

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static float far_all[TEST_FRAMES * DFN2_HOP_LEN];
static float far_tones[TEST_FRAMES * 4 * 256];
static float saved[TEST_FRAMES][DFN2_HOP_LEN];

static void make_partition(void) {
    dfn_unit_partition(erb_fwd, erb_inv);
}

static void make_far(void) {
    dfn_fixture_lcg_far(far_all, (size_t)TEST_FRAMES * DFN2_HOP_LEN, 0x1234567u);
}

static void make_hop(float *mic, float *ref, int frame) {
    int i;
    for (i = 0; i < DFN2_HOP_LEN; ++i) {
        int t = frame * DFN2_HOP_LEN + i;
        float echo = t >= ECHO_DELAY ? 0.5f * far_all[t - ECHO_DELAY] : 0.0f;
        ref[i] = far_all[t];
        mic[i] = echo + 0.01f * sinf((float)t * 0.071f);
    }
}

/* ---------------------------------------------------------------------------------------
 * The composition under test, built from heap pieces as main.c builds it from one pool.
 * --------------------------------------------------------------------------------------- */

typedef struct MonoDfn {
    Aec *aec;
    float *unity, *total;        /* the RES gain path: min(1, G_res) */
    Complex *post;               /* P */
    void *post_mem;
    uint32_t rng;
    int n_freqs, enable_cng;
    DfnResStage *stage;          /* driven directly at 48 kHz; the bridge's own below 48 kHz */
    DfnRateBridge *bridge;
    AcIstft *istft;
    void *istft_mem;
    Complex *stage_out;
    DfnHop hop;
    int hop_size;
} MonoDfn;

static void mdfn_destroy(MonoDfn *m) {
    if (!m) return;
    if (m->bridge) dfn_rate_bridge_destroy(m->bridge);
    else if (m->stage) dfn_res_stage_destroy(m->stage);
    if (m->aec) { aec_destroy(m->aec); free(m->aec); }
    free(m->post_mem);
    free(m->istft_mem);
    free(m->stage_out);
    free(m);
}

/* fft_size 0: the rate's default grid. */
static MonoDfn *mdfn_create(int sample_rate, int fft_size, int enable_cng, const DFN2Model *model) {
    MonoDfn *m = (MonoDfn *)calloc(1, sizeof(*m));
    AecSignalGrid grid;
    AecConfig lane;
    DfnResStageConfig stage_cfg = dfn_res_stage_default_config(DFN2_SR);
    size_t vec;
    int k;
    if (!m) return NULL;
    if (!aec_resolve_signal_grid(sample_rate,
                                 fft_size ? fft_size : aec_default_fft_size(sample_rate), &grid))
        goto fail;
    aec_config_from_preset(&lane, AEC_PRESET_BALANCED, sample_rate);
    lane.fft_size = grid.fft_size;
    lane.delay_mode = AEC_DELAY_MATCHED;
    lane.delay_num_filters = DA_NUM_FILTERS;
    lane.fixed_delay_samples = -1;
    lane.enable_delay_est = 1;
    lane.enable_res = 0;
    lane.return_res_context = 1;
    m->hop_size = grid.hop_size;

    m->aec = (Aec *)calloc(1, sizeof(Aec));
    if (!m->aec || aec_create(m->aec, &lane) != 0) { free(m->aec); m->aec = NULL; goto fail; }

    m->n_freqs = grid.n_freqs;
    m->enable_cng = enable_cng;
    vec = ALIGN16((size_t)grid.n_freqs * sizeof(float));
    if (posix_memalign(&m->post_mem, 16, 2 * vec + ALIGN16((size_t)grid.n_freqs * sizeof(Complex))) != 0) {
        m->post_mem = NULL;
        goto fail;
    }
    m->unity = (float *)m->post_mem;
    m->total = (float *)((unsigned char *)m->post_mem + vec);
    m->post = (Complex *)((unsigned char *)m->post_mem + 2 * vec);
    for (k = 0; k < m->n_freqs; ++k) m->unity[k] = 1.0f;
    m->rng = CNG_RNG_SEED;

    stage_cfg.erb_fwd = erb_fwd;
    stage_cfg.erb_inv = erb_inv;
    if (model) stage_cfg.model = *model;
    if (sample_rate != DFN2_SR) {
        DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(sample_rate);
        bc.fft_size = fft_size;
        bc.stage = stage_cfg;
        m->bridge = dfn_rate_bridge_create(&bc);
        if (!m->bridge || dfn_rate_bridge_n_freqs(m->bridge) != grid.n_freqs ||
            dfn_rate_bridge_hop_size(m->bridge) != grid.hop_size)
            goto fail;
        m->stage = dfn_rate_bridge_stage(m->bridge);
        dfn_hop_init_bridge(&m->hop, m->bridge);
    } else {
        AcStftConfig scfg;
        size_t istft_bytes;
        if (grid.n_freqs != DFN2_N_BINS || grid.hop_size != DFN2_HOP_LEN) goto fail;
        m->stage = dfn_res_stage_create(&stage_cfg);
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        istft_bytes = ac_istft_get_mem_size(&scfg);
        if (!m->stage || posix_memalign(&m->istft_mem, 16, istft_bytes) != 0) goto fail;
        m->istft = ac_istft_init(m->istft_mem, istft_bytes, &scfg);
        m->stage_out = (Complex *)calloc(DFN2_N_BINS, sizeof(Complex));
        if (!m->istft || !m->stage_out) goto fail;
        dfn_hop_init_stage(&m->hop, m->stage, m->istft, m->stage_out);
    }
    return m;
fail:
    mdfn_destroy(m);
    return NULL;
}

/* AEC, RES gain, P; the DFN hop begins on (E, P). */
static int mdfn_begin(MonoDfn *m, const float *mic, const float *ref) {
    AecResContext c;
    aec_process_context(m->aec, mic, ref);
    aec_get_res_context(m->aec, &c);
    if (!c.error_spec || !c.res_gain) return -1;
    sk_min_f32(m->total, m->unity, c.res_gain, m->n_freqs);
    sk_capply_gain_f32(m->post, c.error_spec, m->total, m->n_freqs);
    if (m->enable_cng && c.comfort_noise)
        aec_cng_fill(&m->rng, m->post, c.comfort_noise, c.res_gain, m->unity, CNG_NR_GAIN_FLOOR,
                     m->n_freqs);
    return dfn_hop_begin(&m->hop, c.error_spec, m->post) == 0 ? 0 : -1;
}

static int mdfn_process(MonoDfn *m, const float *mic, const float *ref, float *out) {
    DfnResStageJob job;
    int r;
    if (mdfn_begin(m, mic, ref) != 0) return -1;
    while ((r = dfn_hop_nn_begin(&m->hop, &job)) == 1)
        if (dfn_hop_nn_end(&m->hop, dfn_res_stage_infer(m->stage, &job)) != 0) return -1;
    if (r < 0) return -1;
    return dfn_hop_end(&m->hop, out) == 0 ? 0 : -1;
}

static int run_identity(int enable_cng) {
    MonoDfn *dfn = mdfn_create(DFN2_SR, DFN2_N_FFT, enable_cng, NULL);
    ConvMonoConfig bcfg = conv_mono_default_config(DFN2_SR, DFN2_N_FFT);
    ConvMono *baseline;
    float mic[DFN2_HOP_LEN], ref[DFN2_HOP_LEN], out[DFN2_HOP_LEN];
    long long frames = -1, commits = -1, skips = -1;
    float min_res = 1.0f;
    int frame, i, k, delay_changes = 0, nonzero_hops = 0;

    if (!dfn || dfn->hop_size != DFN2_HOP_LEN || dfn->bridge != NULL)
        return 1;

    bcfg.enable_nr = 0;
    bcfg.enable_res = 1;
    bcfg.enable_cng = enable_cng;
    baseline = conv_mono_create(&bcfg);
    if (!baseline) return 2;

    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        AecLinearContext linear;
        AecResContext ctx;
        int any = 0;
        make_hop(mic, ref, frame);
        if (mdfn_process(dfn, mic, ref, out) != 0 ||
            conv_mono_hop(baseline, mic, ref, saved[frame]) != 0)
            return 10 + frame;
        aec_get_linear_context(dfn->aec, &linear);
        if (linear.delay_state == AEC_LINEAR_DELAY_CHANGED) ++delay_changes;
        aec_get_res_context(dfn->aec, &ctx);
        if (ctx.res_gain)
            for (k = 0; k < ctx.n_freqs; ++k)
                if (ctx.res_gain[k] < min_res) min_res = ctx.res_gain[k];
        for (i = 0; i < DFN2_HOP_LEN; ++i) {
            float expected = frame < DFN_RES_STAGE_LOOKAHEAD_FRAMES
                ? 0.0f : saved[frame - DFN_RES_STAGE_LOOKAHEAD_FRAMES][i];
            if (out[i] != expected) return 100 + frame;
            if (frame >= DFN_RES_STAGE_LOOKAHEAD_FRAMES && expected != 0.0f) any = 1;
        }
        nonzero_hops += any;
    }
    if (delay_changes < 1) return 3;          /* the contract was never exercised */
    if (!(min_res < 0.9f)) return 4;          /* RES never cut: A == B */
    if (nonzero_hops < TEST_FRAMES / 2) return 5;
    dfn_res_stage_get_counters(dfn->stage, &frames, &commits, &skips);
    if (frames != TEST_FRAMES || commits != 0 || skips != TEST_FRAMES - DFN2_MASK_LOOKAHEAD)
        return 6;                             /* a reset would leave a gap */
    if (aec_set_preset(dfn->aec, AEC_PRESET_AGGRESSIVE, 0.0f) != 0)
        return 7;
    if (dfn_res_stage_set_atten_lim(dfn->stage, 12.0f) != 0 ||
        dfn_res_stage_set_atten_lim(dfn->stage, NAN) != -1)
        return 8;
    printf("mono_aec_dfn_res: cng=%d identity over %d hops, %d delay change(s), "
           "min G_res %.3f\n", enable_cng, TEST_FRAMES, delay_changes, min_res);
    conv_mono_destroy(baseline);
    mdfn_destroy(dfn);
    return 0;
}

/* Native grid (8 or 16 kHz): host on its own grid, the stage through the
 * bridge. The hand-written chain's NR-off output delayed by the bridge's
 * added delay must match within the resampler round-trip error; a near-end
 * impulse pins the delay; the 48 kHz frame schedule is the documented one. */
static int run_native(int sample_rate, int fft_size, int expected_sixths) {
    MonoDfn *dfn = mdfn_create(sample_rate, fft_size, 0, NULL);
    ConvMonoConfig bcfg = conv_mono_default_config(sample_rate, fft_size);
    ConvMono *baseline;
    static float mic[512], ref[512], out[512];
    static float conv[TEST_FRAMES * 4 * 256];
    static float mine[TEST_FRAMES * 4 * 256];
    static float conv_lp[TEST_FRAMES * 4 * 256];
    static float conv_ref[TEST_FRAMES * 4 * 256];
    static float mine_lp[TEST_FRAMES * 4 * 256];
    int hop, frames = TEST_FRAMES * 4, frame, i, n, added, peak = -1, imp;
    int num = 0, den = 0;
    float max_err = 0.0f, max_sig = 0.0f, peak_abs = 0.0f;
    int schedule[8], last, max_err_at = -1;
    long long total_frames = 0;

    if (!dfn || dfn->bridge == NULL) return 1;
    hop = dfn->hop_size;
    added = dfn_rate_bridge_added_delay_samples(dfn->bridge);
    if (dfn_rate_bridge_added_delay_exact(dfn->bridge, &num, &den) != 0 ||
        den != DFN_RATE_BRIDGE_DELAY_DENOMINATOR || num != expected_sixths ||
        added != num / den)
        return 2;
    bcfg.enable_nr = 0;
    bcfg.enable_res = 1;
    bcfg.enable_cng = 0;
    baseline = conv_mono_create(&bcfg);
    if (!baseline || baseline->hop != hop) return 3;
    n = frames * hop;

    /* Band-limited echo stimulus at the native rate (delay 384 samples
     * scaled), plus a near-end impulse in the second half for the delay
     * pin. */
    dfn_fixture_multitone(far_tones, (size_t)n, sample_rate);
    imp = (frames * 3 / 4) * hop + 5;
    for (frame = 0; frame < frames; ++frame) {
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            int d = ECHO_DELAY * sample_rate / DFN2_SR;
            float echo = t >= d ? 0.5f * far_tones[t - d] : 0.0f;
            ref[i] = far_tones[t];
            mic[i] = echo + 0.01f * sinf((float)t * 0.071f) + (t == imp ? 1.0f : 0.0f);
        }
        if (mdfn_process(dfn, mic, ref, out) != 0 ||
            conv_mono_hop(baseline, mic, ref, conv + frame * hop) != 0)
            return 10 + frame;
        memcpy(mine + frame * hop, out, (size_t)hop * sizeof(float));
        dfn_rate_bridge_get_counters(dfn->bridge, &total_frames, &last);
        if (frame < 8) schedule[frame] = last;
    }
    /* Schedule per native hop: 16 kHz/hop 128 -> 0,1,1,1; 16 kHz/hop 256 and
     * 8 kHz/hop 128 -> 1,2. */
    for (frame = 0; frame < 8; ++frame) {
        int want = (sample_rate == 16000 && hop == 128)
            ? (frame % 4 == 0 ? 0 : 1) : (frame % 2 == 0 ? 1 : 2);
        if (schedule[frame] != want) return 4;
    }
    if (total_frames != ((long long)n * DFN2_SR / sample_rate) / DFN2_HOP_LEN) return 5;
    /* Tolerance identity against the hand-written chain's output, after warm-up
     * and excluding the impulse's neighbourhood. Compare inside the pass band:
     * both signals through the same low-pass (cut-off 0.35 fs), the conventional
     * one delayed by the exact added delay. */
    dfn_fixture_lowpass(conv, conv_lp, n, 129, 0.35f);
    dfn_fixture_lowpass(mine, mine_lp, n, 129, 0.35f);
    dfn_fixture_delay_sixths(conv_lp, conv_ref, n, num);
    /* The test low-pass truncates at both ends: keep 64 samples clear. */
    for (i = added + 8 * hop + 64; i < n - 64; ++i) {
        float ref_s = conv_ref[i], err;
        if (abs(i - (imp + hop + added)) < 4 * hop + 128) continue;
        err = fabsf(mine_lp[i] - ref_s);
        if (err > max_err) { max_err = err; max_err_at = i; }
        if (fabsf(ref_s) > max_sig) max_sig = fabsf(ref_s);
    }
    /* Delay pin: the impulse leaves the hand-written chain at imp + hop and
     * the bridge path at imp + hop + added. */
    for (i = imp + hop + added - 2 * hop; i < imp + hop + added + 2 * hop && i < n; ++i) {
        float a = fabsf(mine[i]);
        if (a > peak_abs) { peak_abs = a; peak = i; }
    }
    printf("mono_aec_dfn_res: host %d Hz/fft %d: added delay %d + %d/%d, pass-band "
           "max error %.2e at %d (of %d; impulse at %d; signal peak %.2f), "
           "impulse peak at +%d\n",
           sample_rate, fft_size, added, num - added * den, den, (double)max_err, max_err_at, n, imp,
           (double)max_sig, peak - imp - hop);
    /* The impulse peak is the sample nearest to the exact position imp + hop + num / den. */
    if (abs(den * (peak - imp - hop) - num) * 2 >= den) return 6;
    if (!(max_err < 1e-3f)) return 7;
    conv_mono_destroy(baseline);
    mdfn_destroy(dfn);
    return 0;
}

/* The lane grid the composition starts from (libaec's) refuses what no DFN2 host grid can be. */
static int run_rejections(void) {
    AecSignalGrid grid;
    DfnRateBridgeConfig bc;
    DfnRateBridgeMemReq bridge_req;

    if (aec_resolve_signal_grid(44100, aec_default_fft_size(44100), &grid)) return 1;
    if (aec_resolve_signal_grid(DFN2_SR, 512, &grid)) return 2;   /* not a 48 kHz grid */
    if (!aec_resolve_signal_grid(DFN2_SR, aec_default_fft_size(DFN2_SR), &grid) ||
        grid.fft_size != DFN2_N_FFT)
        return 3;
    /* a bridge is for hosts below 48 kHz and needs the partition through its stage config */
    bc = dfn_rate_bridge_default_config(16000);
    bc.stage.erb_fwd = erb_fwd;
    bc.stage.erb_inv = erb_inv;
    if (dfn_rate_bridge_get_mem_requirements(&bc, &bridge_req) != 0) return 4;
    bc.stage.erb_inv = NULL;
    if (dfn_rate_bridge_get_mem_requirements(&bc, &bridge_req) == 0) return 5;
    return 0;
}

/* The phased hop (begin / nn_begin ... nn_end / end) against the callback-driven one: a
 * non-identity model (a uniform 0.5 band mask) run by the caller reproduces the configured
 * callback byte for byte at 48 kHz and through the rate bridge, with comfort noise on; a
 * failed run, a frame left open and a frame never fetched equal the model-free composition. */

static int run_phased(int sample_rate, int fft_size, int behaviour) {
    DFN2Model cb_model;
    DFN2ModelIoDescriptor descriptor;
    MonoDfn *ph, *cb;
    static float mic[512], ref[512], oa[512], ob[512];
    static float tones[TEST_FRAMES * 4 * 256 * 2];
    DfnResStageJob job;
    int hop, frame, i, r, nonzero = 0, ran = 0, frames = 80;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    memset(&cb_model, 0, sizeof(cb_model));
    if (behaviour == BP_RUN) {
        cb_model.infer = dfn_fixture_half_mask_infer;
        cb_model.io_descriptor = &descriptor;
    }
    ph = mdfn_create(sample_rate, fft_size, 1, NULL);
    cb = mdfn_create(sample_rate, fft_size, 1, &cb_model);
    if (!ph || !cb) return 2;
    hop = ph->hop_size;
    dfn_fixture_multitone(tones, (size_t)frames * hop, sample_rate);
    for (frame = 0; frame < frames; ++frame) {
        int e = (ECHO_DELAY * sample_rate) / DFN2_SR;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = tones[t];
            mic[i] = (t >= e ? 0.5f * tones[t - e] : 0.0f) + 0.01f * sinf((float)t * 0.071f);
        }
        if (mdfn_process(cb, mic, ref, ob) != 0) return 3;
        if (mdfn_begin(ph, mic, ref) != 0) return 4;
        if (behaviour != BP_UNFETCHED) {
            while ((r = dfn_hop_nn_begin(&ph->hop, &job)) == 1) {
                int rc;
                ++ran;
                if (behaviour == BP_OPEN) break;
                rc = behaviour == BP_RUN
                    ? dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs) : -1;
                if (dfn_hop_nn_end(&ph->hop, rc) != 0) return 5;
            }
            if (r < 0) return 6;
        }
        if (dfn_hop_end(&ph->hop, oa) != 0) return 7;
        if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) return 10 + frame;
        for (i = 0; i < hop; ++i) if (oa[i] != 0.0f) nonzero = 1;
    }
    if (!nonzero) return 8;
    if (behaviour != BP_UNFETCHED && ran == 0) return 9;
    mdfn_destroy(ph);
    mdfn_destroy(cb);
    return 0;
}

int main(void) {
    int rc;
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();   /* this unit's lookahead pair is the archive's */
    make_partition();
    make_far();
    rc = run_rejections();
    if (rc) { fprintf(stderr, "mono DFN rejection row failed: %d\n", rc); return 1; }
    rc = run_identity(0);
    if (rc) { fprintf(stderr, "mono DFN identity (cng off) failed: %d\n", rc); return 1; }
    rc = run_identity(1);
    if (rc) { fprintf(stderr, "mono DFN identity (cng on) failed: %d\n", rc); return 1; }
    rc = run_native(16000, 256, dfn_fixture_bridge_added_delay_sixths(16000, 128));
    if (rc) { fprintf(stderr, "mono DFN 16 kHz/256 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(16000, 512, dfn_fixture_bridge_added_delay_sixths(16000, 256));
    if (rc) { fprintf(stderr, "mono DFN 16 kHz/512 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(8000, 256, dfn_fixture_bridge_added_delay_sixths(8000, 128));
    if (rc) { fprintf(stderr, "mono DFN 8 kHz/256 bridge gate failed: %d\n", rc); return 1; }
    {
        static const int grids[4][2] = {{48000, 1024}, {16000, 256}, {16000, 512}, {8000, 256}};
        int g, b;
        for (g = 0; g < 4; ++g)
            for (b = 0; b < 4; ++b) {
                rc = run_phased(grids[g][0], grids[g][1], b);
                if (rc) {
                    fprintf(stderr, "mono DFN phased %d Hz/%d behaviour %d failed: %d\n",
                            grids[g][0], grids[g][1], b, rc);
                    return 1;
                }
            }
    }
    puts("mono_aec_dfn_res: PASS");
    return 0;
}
