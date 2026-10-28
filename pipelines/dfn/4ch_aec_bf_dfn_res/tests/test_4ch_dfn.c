/* 4-channel AEC -> BF -> RES -> DeepFilterNet2 composition test: the linear AEC + BF unit
 * (the mcab flow with the sample SRP-PHAT/BF helper, mcab_sample_beam.h), the post-beam RES (an AecRes the composition builds with the libaec
 * functions), P = E * min(1, G_res) (+ comfort noise) and the DFN2 hop (dfn/dfn_hop.h),
 * wired as dfn/4ch_aec_bf_dfn_res/main.c wires them.
 * 48 kHz: with the identity model the composition equals the hand-written conventional chain with
 * the denoiser off (tests/conv_chain_mcab.h, its own front end and AecRes, fed the unit's beam
 * weights and beamformed spectrum), delayed by exactly DFN_RES_STAGE_LOOKAHEAD_FRAMES hops (two for
 * the shipped 1/1 pair), byte for byte, with comfort noise
 * OFF and ON.
 * The stimulus is a 384-sample echo under the shared matched delay estimator; the test asserts
 * the estimator reported a delay change (the stage must not reset on one: its frame count keeps
 * growing with no commit and one skipped frame per model frame) and that the RES cuts.
 * 16 kHz: the stage runs through the bridge; the output equals the NR-off output delayed by the
 * bridge's added delay within the resamplers' pass-band error, a near-end impulse pins the delay
 * and the per-hop stage schedule is the documented one.
 * The phased hop (begin / nn_begin / nn_end / end) against the callback-driven one, in the four
 * behaviours run / fail / left open / never fetched, on every grid and both error entries. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MCAB_TEST_RIG_WITH_BEAM 1
#include "4ch_aec_bf_nr_res/tests/mcab_test_rig.h"
#include "dfn/dfn_hop.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "dfn/tests/dfn_test_fixture.h"
#include "4ch_aec_bf_nr_res/tests/conv_chain_mcab.h"
#include "4ch_aec_bf_nr_res/mcab_sample_post.h"

#define TEST_FRAMES 64
#define ECHO_DELAY 384

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];
static float far_all[TEST_FRAMES * DFN2_HOP_LEN];
static float far_tones[TEST_FRAMES * 4 * 256];
static float conventional_out[TEST_FRAMES][DFN2_HOP_LEN];

static void make_partition(void) {
    dfn_unit_partition(erb_fwd, erb_inv);
}

static void make_far(void) {
    dfn_fixture_lcg_far(far_all, (size_t)TEST_FRAMES * DFN2_HOP_LEN, 0x7654321u);
}

static void make_weights(Complex *weights) {
    int ch, k;
    for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
        for (k = 0; k < DFN2_N_BINS; ++k) {
            Complex *w = weights + (size_t)ch * DFN2_N_BINS + k;
            w->r = 0.25f;
            w->i = 0.0f;
        }
    }
}

static void make_hop(float *mics, float *ref, int frame) {
    int i, ch;
    for (i = 0; i < DFN2_HOP_LEN; ++i) {
        int t = frame * DFN2_HOP_LEN + i;
        ref[i] = far_all[t];
        for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
            int d = ECHO_DELAY + 3 * ch;
            float echo = t >= d ? 0.5f * far_all[t - d] : 0.0f;
            mics[(size_t)i * MCAB_CHANNELS + ch] =
                echo + 0.01f * sinf((float)t * 0.071f + (float)ch);
        }
    }
}

/* The core every oracle in this file starts from. */
static McabConfig core_config(int sample_rate, int fft_size) {
    McabConfig cfg = mcab_default_config(sample_rate);
    cfg.fft_size = fft_size;
    return cfg;
}

/* The unit of the composition: its rate default grid. */
static McabTestUnitConfig unit_config(int sample_rate, int fft_size) {
    McabTestUnitConfig cfg = mcab_test_unit_default_config(sample_rate);
    cfg.core.fft_size = fft_size;
    return cfg;
}

/* The hand-written chain with the denoiser off: RES on, comfort noise as asked. */
static ConvChain *oracle_create(const McabConfig *cfg, int enable_res, int enable_cng) {
    ConvChainOptions opt = conv_default_options();
    opt.enable_res = enable_res;
    opt.enable_nr = 0;
    opt.enable_cng = enable_cng;
    return conv_create(cfg, &opt);
}

/* ---------------------------------------------------------------------------------------
 * The composition under test, built from heap pieces as main.c builds it from one pool.
 * --------------------------------------------------------------------------------------- */

/* Deliberate defects the flow-reset row plants to show it can fail. */
enum { FDFN_SAB_NO_CLEAR = 1, FDFN_SAB_LATE_CLEAR = 2 };

typedef struct FourDfn {
    McabSampleBeam *unit;
    McabSampleBeamResult beam;   /* the last hop's beam, for the oracle */
    McabSampleBeamInfo info;
    AecRes *res;                 /* the post-beam RES, balanced, on the unit's grid */
    void *res_mem;
    const float *res_gain;       /* the last hop's G_res */
    float *unity, *total;
    Complex *post;
    void *post_mem;
    uint32_t rng;
    uint32_t reset_handled;   /* McabFrame.post_reset_serial last followed */
    long reset_hops;          /* hops on which the serial had moved */
    const Complex *estimate;  /* E of the last hop (mcab memory, until the next hop) */
    int sabotage;             /* negative controls (FDFN_SAB_*), 0 in the composition */
    int bridge_clear_due;     /* FDFN_SAB_LATE_CLEAR only: the event of the previous hop */
    int n_freqs, enable_cng;
    DfnResStage *stage;          /* driven directly at 48 kHz; the bridge's own below 48 kHz */
    DfnRateBridge *bridge;
    AcIstft *istft;
    void *istft_mem;
    Complex *stage_out;
    DfnHop hop;
    int hop_size;
} FourDfn;

static void fdfn_destroy(FourDfn *m) {
    if (!m) return;
    if (m->bridge) dfn_rate_bridge_destroy(m->bridge);
    else if (m->stage) dfn_res_stage_destroy(m->stage);
    if (m->unit) mcab_test_beam_destroy(m->unit);
    free(m->res_mem);
    free(m->post_mem);
    free(m->istft_mem);
    free(m->stage_out);
    free(m);
}

/* fft_size 0: the core's rate default grid. bf_enable 0 bypasses the BF (lane 0's error, unit
 * weights), which keeps an impulse's timing unchanged by the spatial filter. */
static FourDfn *fdfn_create(int sample_rate, int fft_size, int enable_cng, const DFN2Model *model,
                            int bf_enable) {
    FourDfn *m = (FourDfn *)calloc(1, sizeof(*m));
    McabTestUnitConfig ucfg = unit_config(sample_rate, fft_size);
    DfnResStageConfig stage_cfg = dfn_res_stage_default_config(DFN2_SR);
    AecConfig res_cfg;
    size_t vec, res_bytes;
    int n_freqs, k;
    if (!m) return NULL;
    ucfg.beam.bf_enable = bf_enable;
    m->unit = mcab_test_beam_create(&ucfg);
    if (!m->unit) goto fail;
    m->hop_size = mcab_hop_size(mcab_test_beam_flow(m->unit));
    n_freqs = mcab_n_freqs(mcab_test_beam_flow(m->unit));

    aec_config_from_level(&res_cfg, AEC_LEVEL_BALANCED, sample_rate);
    res_cfg.fft_size = mcab_fft_size(mcab_test_beam_flow(m->unit));
    res_bytes = aec_res_get_mem_size(&res_cfg);
    if (res_bytes == 0 || posix_memalign(&m->res_mem, 16, res_bytes) != 0) {
        m->res_mem = NULL;
        goto fail;
    }
    m->res = aec_res_init(m->res_mem, res_bytes, &res_cfg);
    if (!m->res || aec_res_n_bins(m->res) != n_freqs || aec_res_hop_size(m->res) != m->hop_size)
        goto fail;

    m->n_freqs = n_freqs;
    m->enable_cng = enable_cng;
    vec = ALIGN16((size_t)n_freqs * sizeof(float));
    if (posix_memalign(&m->post_mem, 16, 2 * vec + ALIGN16((size_t)n_freqs * sizeof(Complex))) != 0) {
        m->post_mem = NULL;
        goto fail;
    }
    m->unity = (float *)m->post_mem;
    m->total = (float *)((unsigned char *)m->post_mem + vec);
    m->post = (Complex *)((unsigned char *)m->post_mem + 2 * vec);
    for (k = 0; k < n_freqs; ++k) m->unity[k] = 1.0f;
    m->rng = MCAB_SAMPLE_CNG_RNG_SEED;

    stage_cfg.erb_fwd = erb_fwd;
    stage_cfg.erb_inv = erb_inv;
    if (model) stage_cfg.model = *model;
    if (sample_rate != DFN2_SR) {
        DfnRateBridgeConfig bc = dfn_rate_bridge_default_config(sample_rate);
        bc.fft_size = fft_size;
        bc.stage = stage_cfg;
        m->bridge = dfn_rate_bridge_create(&bc);
        if (!m->bridge || dfn_rate_bridge_n_freqs(m->bridge) != n_freqs ||
            dfn_rate_bridge_hop_size(m->bridge) != m->hop_size)
            goto fail;
        m->stage = dfn_rate_bridge_stage(m->bridge);
        dfn_hop_init_bridge(&m->hop, m->bridge);
    } else {
        AcStftConfig scfg;
        size_t istft_bytes;
        if (n_freqs != DFN2_N_BINS || m->hop_size != DFN2_HOP_LEN) goto fail;
        m->stage = dfn_res_stage_create(&stage_cfg);
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        istft_bytes = ac_istft_get_mem_size(&scfg);
        if (!m->stage || posix_memalign(&m->istft_mem, 16, istft_bytes) != 0) {
            m->istft_mem = NULL;
            goto fail;
        }
        m->istft = ac_istft_init(m->istft_mem, istft_bytes, &scfg);
        m->stage_out = (Complex *)calloc(DFN2_N_BINS, sizeof(Complex));
        if (!m->istft || !m->stage_out) goto fail;
        dfn_hop_init_stage(&m->hop, m->stage, m->istft, m->stage_out);
        m->hop.check_finite = 1;
    }
    return m;
fail:
    fdfn_destroy(m);
    return NULL;
}

/* What an integrator does after a refused call or to start over: every stage together. */
static void fdfn_reset(FourDfn *m) {
    mcab_sample_beam_reset(m->unit);
    aec_res_reset(m->res);
    m->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    dfn_hop_reset(&m->hop);
}

/* Unit (front end, SRP-PHAT, BF), fuse, the RES gain, P; the DFN hop begins on (E, P). */
static int fdfn_begin(FourDfn *m, const float *mics, const float *ref) {
    McabFrame f;
    AecResInput in;
    int due;
    if (mcab_sample_beam_process(m->unit, mics, ref, &m->beam, &m->info) != MCAB_OK ||
        mcab_sample_beam_fuse(m->unit, &f) != MCAB_OK || f.g_res_ctx != NULL)
        return -1;
    mcab_sample_res_input(&f, &in);
    if (!(m->res_gain = aec_res_process(m->res, &in))) return -1;
    m->estimate = f.e;
    due = mcab_sample_post_reset_due(&m->reset_handled, &f);
    if (due) ++m->reset_hops;
    if (m->bridge) {
        /* the bridge synthesises the host-grid spectra at the top of its begin(): the clear goes in on the hop
         * of the event, with no delay line, and DFN2 keeps running */
        int clear = due;
        if (m->sabotage & FDFN_SAB_LATE_CLEAR) { clear = m->bridge_clear_due; m->bridge_clear_due = due; }
        if (clear && !(m->sabotage & FDFN_SAB_NO_CLEAR) && dfn_rate_bridge_reset_synthesis(m->bridge) != 0)
            return -1;
    } else if (due) {
        dfn_hop_reset(&m->hop);
    }
    if (due) m->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    sk_min_f32(m->total, m->unity, m->res_gain, m->n_freqs);
    sk_capply_gain_f32(m->post, f.b, m->total, m->n_freqs);
    if (m->enable_cng && f.cn)
        aec_cng_fill(&m->rng, m->post, f.cn, m->res_gain, m->unity, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR,
                     m->n_freqs);
    return dfn_hop_begin(&m->hop, f.e, m->post) == 0 ? 0 : -1;
}

static int fdfn_process(FourDfn *m, const float *mics, const float *ref, float *out) {
    DfnResStageJob job;
    int r;
    if (fdfn_begin(m, mics, ref) != 0) return -1;
    while ((r = dfn_hop_nn_begin(&m->hop, &job)) == 1)
        if (dfn_hop_nn_end(&m->hop, dfn_res_stage_infer(m->stage, &job)) != 0) return -1;
    if (r < 0) return -1;
    return dfn_hop_end(&m->hop, out) == 0 ? 0 : -1;
}

static int run_identity(int enable_cng) {
    FourDfn *dfn = fdfn_create(DFN2_SR, DFN2_N_FFT, enable_cng, NULL, 1);
    McabConfig ccfg = core_config(DFN2_SR, DFN2_N_FFT);
    ConvChain *conventional;
    float mics[DFN2_HOP_LEN * MCAB_CHANNELS];
    float ref[DFN2_HOP_LEN];
    float dfn_out[DFN2_HOP_LEN];
    long long frames = -1, commits = -1, skips = -1;
    float min_res = 1.0f, live0 = 0.0f, live1 = 0.0f;
    int frame, i, k, delay_changes = 0, nonzero_hops = 0;

    if (!dfn || dfn->hop_size != DFN2_HOP_LEN || dfn->bridge != NULL)
        return 1;
    conventional = oracle_create(&ccfg, 1, enable_cng);
    if (!conventional) return 2;

    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        int any = 0;
        make_hop(mics, ref, frame);
        if (fdfn_process(dfn, mics, ref, dfn_out) != 0) return 10 + frame;
        if (dfn->info.delay.changed) ++delay_changes;
        for (k = 0; k < dfn->n_freqs; ++k)
            if (dfn->res_gain[k] < min_res) min_res = dfn->res_gain[k];
        if (conv_hop(conventional, mics, ref, dfn->beam.weights, dfn->beam.spectrum,
                     conventional_out[frame]) != 0)
            return 200 + frame;
        for (i = 0; i < DFN2_HOP_LEN; ++i) {
            float expected = frame < DFN_RES_STAGE_LOOKAHEAD_FRAMES
                ? 0.0f : conventional_out[frame - DFN_RES_STAGE_LOOKAHEAD_FRAMES][i];
            if (dfn_out[i] != expected) return 300 + frame;
            if (frame >= DFN_RES_STAGE_LOOKAHEAD_FRAMES && expected != 0.0f) any = 1;
        }
        nonzero_hops += any;
    }
    if (delay_changes < 1) return 3;          /* the contract was never exercised */
    if (!(min_res < 0.9f)) return 9;          /* RES never cut: A == B */
    if (nonzero_hops < TEST_FRAMES / 2) return 4;
    dfn_res_stage_get_counters(dfn->stage, &frames, &commits, &skips);
    if (frames != TEST_FRAMES || commits != 0 || skips != TEST_FRAMES - DFN2_MASK_LOOKAHEAD)
        return 5;                             /* a reset would leave a gap */
    if (aec_res_split_floor(dfn->res, &live0, NULL) != 0 ||
        aec_res_set_level(dfn->res, AEC_LEVEL_AGGRESSIVE, 0.0f) != 0 ||
        aec_res_split_floor(dfn->res, &live1, NULL) != 0 || live1 == live0)
        return 6;
    if (dfn_res_stage_set_atten_lim(dfn->stage, 12.0f) != 0 ||
        dfn_res_stage_set_atten_lim(dfn->stage, NAN) != -1)
        return 7;
    printf("4ch_aec_bf_dfn_res: cng=%d identity over %d hops, "
           "%d delay change(s), min G_res %.3f\n", enable_cng, TEST_FRAMES,
           delay_changes, min_res);
    conv_destroy(conventional);
    fdfn_destroy(dfn);
    return 0;
}

/* The stimulus must make the post-beam RES cut: a RES-off chain on the same input has to differ
 * from the RES-on chain after the warm-up hops. */
static int run_res_is_active(void) {
    McabConfig cfg = core_config(DFN2_SR, DFN2_N_FFT);
    ConvChain *on = oracle_create(&cfg, 1, 0);
    ConvChain *off = oracle_create(&cfg, 0, 0);
    Complex weights[MCAB_CHANNELS * DFN2_N_BINS];
    float mics[DFN2_HOP_LEN * MCAB_CHANNELS];
    float ref[DFN2_HOP_LEN], a[DFN2_HOP_LEN], b[DFN2_HOP_LEN];
    int frame, differing = 0;
    if (!on || !off) return 1;
    make_weights(weights);
    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        make_hop(mics, ref, frame);
        if (conv_hop(on, mics, ref, weights, NULL, a) != 0 ||
            conv_hop(off, mics, ref, weights, NULL, b) != 0)
            return 2;
        if (memcmp(a, b, sizeof(a)) != 0) ++differing;
    }
    conv_destroy(on);
    conv_destroy(off);
    return differing > 0 ? 0 : 3;
}

/* Native grid (the 16 kHz default): the unit on its own grid, the stage through the bridge. The
 * conventional chain's NR-off output (fed the unit's beam, BF bypassed) delayed by the bridge's
 * added delay must match within the resampler round-trip error; a near-end impulse pins the
 * delay; the 48 kHz schedule is the documented one. */
static int run_native(int sample_rate, int fft_size, int expected_sixths) {
    FourDfn *dfn = fdfn_create(sample_rate, fft_size, 0, NULL, 0);
    McabConfig ccfg = core_config(sample_rate, fft_size);
    ConvChain *conventional;
    static float mics[512 * MCAB_CHANNELS];
    static float ref[512], out[512];
    static float conv[TEST_FRAMES * 4 * 256];
    static float mine[TEST_FRAMES * 4 * 256];
    static float conv_lp[TEST_FRAMES * 4 * 256];
    static float conv_ref[TEST_FRAMES * 4 * 256];
    static float mine_lp[TEST_FRAMES * 4 * 256];
    int hop, frames = TEST_FRAMES * 4, frame, i, ch, n, added, peak = -1, imp;
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
    conventional = oracle_create(&ccfg, 1, 0);
    if (!conventional || conventional->hop != hop) return 3;
    n = frames * hop;
    dfn_fixture_multitone(far_tones, (size_t)n, sample_rate);
    imp = (frames * 3 / 4) * hop + 5;
    for (frame = 0; frame < frames; ++frame) {
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = far_tones[t];
            for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
                int d = (ECHO_DELAY + 3 * ch) * sample_rate / DFN2_SR;
                float echo = t >= d ? 0.5f * far_tones[t - d] : 0.0f;
                mics[(size_t)i * MCAB_CHANNELS + ch] =
                    echo + 0.01f * sinf((float)t * 0.071f + (float)ch) +
                    (t == imp ? 1.0f : 0.0f);
            }
        }
        if (fdfn_process(dfn, mics, ref, out) != 0 ||
            conv_hop(conventional, mics, ref, dfn->beam.weights, dfn->beam.spectrum,
                     conv + frame * hop) != 0)
            return 10 + frame;
        memcpy(mine + frame * hop, out, (size_t)hop * sizeof(float));
        dfn_rate_bridge_get_counters(dfn->bridge, &total_frames, &last);
        if (frame < 8) schedule[frame] = last;
    }
    for (frame = 0; frame < 8; ++frame) {
        int want = (sample_rate == 16000 && hop == 128)
            ? (frame % 4 == 0 ? 0 : 1) : (frame % 2 == 0 ? 1 : 2);
        if (schedule[frame] != want) return 4;
    }
    if (total_frames != ((long long)n * DFN2_SR / sample_rate) / DFN2_HOP_LEN) return 5;
    /* Compare inside the pass band: both signals through the same low-pass
     * (cut-off 0.35 fs), the conventional one delayed by the exact added delay. */
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
    for (i = imp + hop + added - 2 * hop; i < imp + hop + added + 2 * hop && i < n; ++i) {
        float a = fabsf(mine[i]);
        if (a > peak_abs) { peak_abs = a; peak = i; }
    }
    printf("4ch_aec_bf_dfn_res: core %d Hz/hop %d: added delay %d + %d/%d, pass-band max "
           "error %.2e at %d (of %d; impulse at %d; signal peak %.2f), impulse "
           "peak at +%d\n", sample_rate, hop, added, num - added * den, den, (double)max_err, max_err_at,
           n, imp, (double)max_sig, peak - imp - hop);
    /* The impulse peak is the sample nearest to the exact position imp + hop + num / den. */
    if (abs(den * (peak - imp - hop) - num) * 2 >= den) return 6;
    if (!(max_err < 1e-3f)) return 7;
    conv_destroy(conventional);
    fdfn_destroy(dfn);
    return 0;
}

/* The flow reset at 16 kHz: the application restarts the unit and the RES (a lane's hard post-chain
 * reset, which moves McabFrame.post_reset_serial) and nothing else; the hop, DFN2 and the bridge keep
 * running. The bridge's native-grid synthesis (dfn_rate_bridge_get_host_hops) must equal the
 * hand-written chain (restarted the same way) bit for bit on every hop, the reset hop included, and a
 * bridge that is not cleared, or cleared one hop late, must not. */
static int run_flow_reset(int sample_rate, int fft_size, int sabotage, long *diff_hops, long *resets) {
    FourDfn *dfn = fdfn_create(sample_rate, fft_size, 1, NULL, 1);
    McabConfig ccfg = core_config(sample_rate, fft_size);
    ConvChain *conventional;
    AcStftConfig scfg;
    AcIstft *probe = NULL;
    void *probe_mem = NULL;
    static float probe_hop[512];
    long long stage_frames = 0, commits, skips;
    static float mics[512 * MCAB_CHANNELS], ref[512], conv_out[512], out[512];
    static float tones[160 * 512];
    int hop, frame, i, ch, e, frames = 160, reset_at = 81;
    *diff_hops = 0;
    if (!dfn || !dfn->bridge) return 1;
    dfn->sabotage = sabotage;
    conventional = oracle_create(&ccfg, 1, 1);
    if (!conventional) return 2;
    hop = dfn->hop_size;
    /* the estimate lane's reference: audiolib's default-window synthesis of E, cleared on the event */
    ac_stft_config_default(&scfg, fft_size ? fft_size : 2 * hop);
    if (posix_memalign(&probe_mem, 16, ac_istft_get_mem_size(&scfg)) != 0) return 6;
    probe = ac_istft_init(probe_mem, ac_istft_get_mem_size(&scfg), &scfg);
    if (!probe) return 7;
    dfn_fixture_multitone(tones, (size_t)frames * hop, sample_rate);
    e = (ECHO_DELAY * sample_rate) / DFN2_SR;
    for (frame = 0; frame < frames; ++frame) {
        const float *tap_e, *tap_p;
        long resets_before = dfn->reset_hops;
        DfnResStageJob job;
        int r;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = tones[t];
            for (ch = 0; ch < MCAB_CHANNELS; ++ch)
                mics[(size_t)i * MCAB_CHANNELS + (size_t)ch] =
                    (t >= e ? 0.5f * tones[t - e] : 0.0f) + 0.01f * sinf((float)t * 0.071f + (float)ch);
        }
        if (frame == reset_at) {          /* the application restarts the unit and the RES, nothing downstream */
            mcab_sample_beam_reset(dfn->unit);
            aec_res_reset(dfn->res);
            conv_reset(conventional);
        }
        if (fdfn_begin(dfn, mics, ref) != 0) return 10 + frame;
        if (dfn_rate_bridge_get_host_hops(dfn->bridge, &tap_e, &tap_p) != 0) return 3;
        if (conv_hop(conventional, mics, ref, dfn->beam.weights, dfn->beam.spectrum, conv_out) != 0)
            return 200 + frame;
        if (dfn->reset_hops != resets_before) ac_istft_reset(probe);
        (void)ac_istft_push(probe, dfn->estimate, probe_hop);
        if (memcmp(tap_p, conv_out, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(tap_e, probe_hop, (size_t)hop * sizeof(float)) != 0)
            ++*diff_hops;
        while ((r = dfn_hop_nn_begin(&dfn->hop, &job)) == 1)
            if (dfn_hop_nn_end(&dfn->hop, dfn_res_stage_infer(dfn->stage, &job)) != 0) return 4;
        if (r < 0 || dfn_hop_end(&dfn->hop, out) != 0) return 5;
    }
    *resets = dfn->reset_hops;
    /* DFN2 kept running through the event: its frame count is the whole stream's, with no gap */
    dfn_res_stage_get_counters(dfn->stage, &stage_frames, &commits, &skips);
    if (stage_frames != ((long long)frames * hop * DFN2_SR / sample_rate) / DFN2_HOP_LEN) return 8;
    free(probe_mem);
    conv_destroy(conventional);
    fdfn_destroy(dfn);
    return 0;
}

/* The unit serves 16 and 48 kHz only: the composition has no 8 kHz or off-grid host. */
static int run_rejections(void) {
    McabTestUnitConfig cfg = unit_config(8000, 0);
    McabMemReq req;
    if (mcab_test_unit_get_mem_requirements(&cfg, &req) == 0) return 1;
    cfg = unit_config(44100, 0);
    if (mcab_test_unit_get_mem_requirements(&cfg, &req) == 0) return 2;
    cfg = unit_config(DFN2_SR, 512);   /* not a 48 kHz grid */
    if (mcab_test_unit_get_mem_requirements(&cfg, &req) == 0) return 3;
    cfg = unit_config(DFN2_SR, 0);
    if (mcab_test_unit_get_mem_requirements(&cfg, &req) != 0) return 4;
    return 0;
}

/* Long enough for the RES to leave its initial state, where its gain no longer hides what a
 * missing aec_res_reset() leaves behind. */
#define RESET_FRAMES 200

/* A reset of every stage (unit, RES, comfort-noise seed, hop) returns the composition to
 * the state of a fresh one:
 * the first hops after it reproduce the first hops of a new instance bit for bit, the second pass
 * of a loop equals the first. */
static int run_reset_equals_fresh(int sample_rate, int fft_size) {
    FourDfn *dfn = fdfn_create(sample_rate, fft_size, 1, NULL, 1);
    static float mics[512 * MCAB_CHANNELS], ref[512];
    static float first[RESET_FRAMES][512], again[RESET_FRAMES][512];
    static float tones[RESET_FRAMES * 512];
    int hop, frame, pass, i, ch, e;
    if (!dfn) return 1;
    hop = dfn->hop_size;
    dfn_fixture_multitone(tones, (size_t)RESET_FRAMES * hop, sample_rate);
    e = (ECHO_DELAY * sample_rate) / DFN2_SR;
    for (pass = 0; pass < 2; ++pass) {
        for (frame = 0; frame < RESET_FRAMES; ++frame) {
            for (i = 0; i < hop; ++i) {
                int t = frame * hop + i;
                ref[i] = tones[t];
                for (ch = 0; ch < MCAB_CHANNELS; ++ch)
                    mics[(size_t)i * MCAB_CHANNELS + (size_t)ch] =
                        (t >= e ? 0.5f * tones[t - e] : 0.0f) +
                        0.01f * sinf((float)t * 0.071f + (float)ch);
            }
            if (fdfn_process(dfn, mics, ref, pass ? again[frame] : first[frame]) != 0)
                return 2;
        }
        if (pass == 0) fdfn_reset(dfn);
    }
    for (frame = 0; frame < RESET_FRAMES; ++frame)
        if (memcmp(first[frame], again[frame], (size_t)hop * sizeof(float)) != 0)
            return 10 + frame;
    fdfn_destroy(dfn);
    return 0;
}

/* The phased hop (begin / nn_begin ... nn_end / end) against the callback-driven one: a
 * non-identity model (a uniform 0.5 band mask) run by the caller reproduces the configured
 * callback byte for byte at 48 kHz and through the rate bridge, comfort noise on; a failed run,
 * a frame left open and a frame never fetched equal the model-free composition. */

static int run_phased(int sample_rate, int fft_size, int behaviour) {
    DFN2Model cb_model;
    DFN2ModelIoDescriptor descriptor;
    FourDfn *ph, *cb;
    static float mics[512 * MCAB_CHANNELS], ref[512], oa[512], ob[512];
    static float tones[100 * 512];
    DfnResStageJob job;
    int hop, frame, i, ch, r, nonzero = 0, ran = 0, frames = 80;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    memset(&cb_model, 0, sizeof(cb_model));
    if (behaviour == BP_RUN) {
        cb_model.infer = dfn_fixture_half_mask_infer;
        cb_model.io_descriptor = &descriptor;
    }
    ph = fdfn_create(sample_rate, fft_size, 1, NULL, 1);
    cb = fdfn_create(sample_rate, fft_size, 1, &cb_model, 1);
    if (!ph || !cb) return 2;
    hop = ph->hop_size;
    dfn_fixture_multitone(tones, (size_t)frames * hop, sample_rate);
    for (frame = 0; frame < frames; ++frame) {
        int e = (ECHO_DELAY * sample_rate) / DFN2_SR;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = tones[t];
            for (ch = 0; ch < MCAB_CHANNELS; ++ch)
                mics[(size_t)i * MCAB_CHANNELS + (size_t)ch] =
                    (t >= e ? 0.5f * tones[t - e] : 0.0f) +
                    0.01f * sinf((float)t * 0.071f + (float)ch);
        }
        if (fdfn_process(cb, mics, ref, ob) != 0) return 5;
        if (fdfn_begin(ph, mics, ref) != 0) return 6;
        if (behaviour != BP_UNFETCHED) {
            while ((r = dfn_hop_nn_begin(&ph->hop, &job)) == 1) {
                int rc;
                ++ran;
                if (behaviour == BP_OPEN) break;
                rc = behaviour == BP_RUN
                    ? dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs) : -1;
                if (dfn_hop_nn_end(&ph->hop, rc) != 0) return 7;
            }
            if (r < 0) return 8;
        }
        if (dfn_hop_end(&ph->hop, oa) != 0) return 9;
        if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) return 100 + frame;
        for (i = 0; i < hop; ++i) if (oa[i] != 0.0f) nonzero = 1;
    }
    if (!nonzero) return 10;
    if (behaviour != BP_UNFETCHED && ran == 0) return 11;
    fdfn_destroy(ph);
    fdfn_destroy(cb);
    return 0;
}

int main(void) {
    int rc, cng;
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();   /* this unit's lookahead pair is the archive's */
    make_partition();
    make_far();
    rc = run_rejections();
    if (rc) { fprintf(stderr, "4ch DFN grid rejection row failed: %d\n", rc); return 1; }
    rc = run_res_is_active();
    if (rc) { fprintf(stderr, "4ch DFN stimulus never engaged RES: %d\n", rc); return 1; }
    for (cng = 0; cng <= 1; ++cng) {
        rc = run_identity(cng);
        if (rc != 0) {
            fprintf(stderr, "4ch DFN identity cng=%d failed: %d\n", cng, rc);
            return 1;
        }
    }
    rc = run_native(16000, 256, dfn_fixture_bridge_added_delay_sixths(16000, 128));
    if (rc) { fprintf(stderr, "4ch DFN 16 kHz/256 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(16000, 512, dfn_fixture_bridge_added_delay_sixths(16000, 256));
    if (rc) { fprintf(stderr, "4ch DFN 16 kHz/512 bridge gate failed: %d\n", rc); return 1; }
    {
        static const int grids[2][2] = {{16000, 256}, {16000, 512}};
        static const int sabs[] = { FDFN_SAB_NO_CLEAR, FDFN_SAB_LATE_CLEAR };
        int g;
        size_t c;
        for (g = 0; g < 2; ++g) {
            long diff, resets;
            rc = run_flow_reset(grids[g][0], grids[g][1], 0, &diff, &resets);
            if (rc || diff != 0 || resets < 1) {
                fprintf(stderr, "4ch DFN flow reset %d Hz/%d: rc %d, %ld hops differ, %ld reset hop(s)\n",
                        grids[g][0], grids[g][1], rc, diff, resets);
                return 1;
            }
            printf("4ch_aec_bf_dfn_res: flow reset %d Hz/%d: the bridge's host synthesis, cleared on the hop of the event, equals the "
                   "hand-written chain bit for bit (%ld reset hop(s))\n", grids[g][0], grids[g][1], resets);
            for (c = 0; c < sizeof sabs / sizeof sabs[0]; ++c) {
                rc = run_flow_reset(grids[g][0], grids[g][1], sabs[c], &diff, &resets);
                if (rc || diff == 0) {
                    fprintf(stderr, "4ch DFN flow reset %d Hz/%d: planted defect %d not detected (rc %d)\n",
                            grids[g][0], grids[g][1], sabs[c], rc);
                    return 1;
                }
            }
        }
    }
    {
        static const int grids[3][2] = {{48000, 1024}, {16000, 256}, {16000, 512}};
        int g, b;
        for (g = 0; g < 3; ++g) {
            rc = run_reset_equals_fresh(grids[g][0], grids[g][1]);
            if (rc) {
                fprintf(stderr, "4ch DFN reset %d Hz/%d failed: %d\n",
                        grids[g][0], grids[g][1], rc);
                return 1;
            }
            for (b = 0; b < 4; ++b) {
                rc = run_phased(grids[g][0], grids[g][1], b);
                if (rc) {
                    fprintf(stderr, "4ch DFN phased %d Hz/%d behaviour %d failed: %d\n",
                            grids[g][0], grids[g][1], b, rc);
                    return 1;
                }
            }
        }
    }
    puts("4ch_aec_bf_dfn_res: PASS");
    return 0;
}
