/* 4-channel AEC -> RES -> DeepFilterNet2 composition test: the four-lane core (4aec_nr_res.h), the
 * RES gain chain (samples/se_sample_chain.h, no denoiser, no synthesis) and the DFN2 hop
 * (samples/dfn_hop.h), wired as 4ch_aec_bf_dfn_res/main.c wires them.
 * 48 kHz: with the identity model the composition equals the hand-written conventional chain with
 * the denoiser off (tests/conv_chain_4ch.h), delayed by exactly two hops, byte for byte, over both
 * beamformed-error entries (reconstructed and trusted spectrum) and with comfort noise OFF and ON.
 * The stimulus is a 384-sample echo under the shared matched delay estimator; the test asserts
 * the estimator reported a delay change (the stage must not reset on one: its frame count keeps
 * growing with no commit and exactly one skipped frame) and that the RES cuts.
 * 16 kHz: the stage runs through the bridge; the output equals the NR-off output delayed by the
 * bridge's added delay within the resamplers' pass-band error, a near-end impulse pins the delay
 * and the per-hop stage schedule is the documented one.
 * The phased hop (begin / nn_begin / nn_end / end) against the callback-driven one, in the four
 * behaviours run / fail / left open / never fetched, on every grid and both error entries. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_nr_res.h"
#include "samples/dfn_hop.h"
#include "samples/se_sample_chain.h"
#include "utility/mem_align.h"
#include "tests/dfn_test_fixture.h"
#include "../../4ch_aec_bf_nr_res/tests/conv_chain_4ch.h"

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
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
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
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            int d = ECHO_DELAY + 3 * ch;
            float echo = t >= d ? 0.5f * far_all[t - d] : 0.0f;
            mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                echo + 0.01f * sinf((float)t * 0.071f + (float)ch);
        }
    }
}

static void form_trusted(const FourAecFront *pre,
                         const Complex *weights, Complex *out) {
    int ch, k;
    memset(out, 0, (size_t)DFN2_N_BINS * sizeof(*out));
    for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
        for (k = 0; k < DFN2_N_BINS; ++k) {
            const Complex *w = weights + (size_t)ch * DFN2_N_BINS + k;
            const Complex *x = pre->linear_spectra[ch] + k;
            out[k].r += w->r * x->r - w->i * x->i;
            out[k].i += w->r * x->i + w->i * x->r;
        }
    }
}

/* The core every composition and every oracle in this file starts from. */
static FourAecNrResConfig core_config(int sample_rate, int fft_size) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    return cfg;
}

/* The hand-written chain with the denoiser off: RES on, comfort noise as asked. */
static ConvChain *oracle_create(const FourAecNrResConfig *cfg, int enable_res, int enable_cng) {
    ConvChainOptions opt = conv_default_options();
    opt.enable_res = enable_res;
    opt.enable_nr = 0;
    opt.enable_cng = enable_cng;
    return conv_create(cfg, &opt);
}

/* ---------------------------------------------------------------------------------------
 * The composition under test, built from heap pieces as main.c builds it from one pool.
 * --------------------------------------------------------------------------------------- */

typedef struct FourDfn {
    FourAecNrRes *core;
    SeSampleChain *chain;
    void *chain_mem;
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
    if (m->chain) se_sample_chain_destroy(m->chain);
    if (m->core) four_aec_nr_res_destroy(m->core);
    free(m->chain_mem);
    free(m->istft_mem);
    free(m->stage_out);
    free(m);
}

/* fft_size 0: the core's rate default grid. */
static FourDfn *fdfn_create(int sample_rate, int fft_size, int enable_cng, const DFN2Model *model) {
    FourDfn *m = (FourDfn *)calloc(1, sizeof(*m));
    FourAecNrResConfig ccfg = core_config(sample_rate, fft_size);
    SeSampleChainConfig sc;
    DfnResStageConfig stage_cfg = dfn_res_stage_default_config(DFN2_SR);
    size_t chain_bytes;
    int n_freqs;
    if (!m) return NULL;
    m->core = four_aec_nr_res_create(&ccfg);
    if (!m->core) goto fail;
    m->hop_size = four_aec_nr_res_hop_size(m->core);
    n_freqs = four_aec_nr_res_n_freqs(m->core);

    sc = se_sample_chain_default_config(sample_rate, four_aec_nr_res_fft_size(m->core));
    sc.res = SE_SAMPLE_RES_AEC_RES;
    sc.res_preset = AEC_PRESET_BALANCED;
    sc.enable_nr = 0;
    sc.enable_cng = enable_cng;
    sc.with_synthesis = 0;
    chain_bytes = se_sample_chain_mem_size(&sc);
    if (chain_bytes == 0 || posix_memalign(&m->chain_mem, 16, chain_bytes) != 0) {
        m->chain_mem = NULL;
        goto fail;
    }
    m->chain = se_sample_chain_init(m->chain_mem, chain_bytes, &sc);
    if (!m->chain) goto fail;

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
    four_aec_nr_res_reset(m->core);
    se_sample_chain_reset(m->chain);
    dfn_hop_reset(&m->hop);
}

/* Front end, fuse (trusted spectrum from the weights, or rebuilt from them), RES, P; the DFN
 * hop begins on (E, P). `front` receives the front frame. */
static int fdfn_begin(FourDfn *m, const float *mics, const float *ref, const Complex *weights,
                      int trusted_mode, FourAecFront *front) {
    static Complex trusted[DFN2_N_BINS];
    SeSpecFrame frame;
    const Complex *post;
    if (four_aec_front_process(m->core, mics, ref, front) != FOUR_AEC_NR_RES_OK) return -1;
    if (trusted_mode) form_trusted(front, weights, trusted);
    if (four_aec_fuse(m->core, weights, trusted_mode ? trusted : NULL, &frame) !=
        FOUR_AEC_NR_RES_OK)
        return -1;
    post = se_sample_chain_form(m->chain, &frame);
    if (!post) return -1;
    return dfn_hop_begin(&m->hop, frame.e, post) == 0 ? 0 : -1;
}

static int fdfn_process(FourDfn *m, const float *mics, const float *ref, const Complex *weights,
                        int trusted_mode, FourAecFront *front, float *out) {
    DfnResStageJob job;
    int r;
    if (fdfn_begin(m, mics, ref, weights, trusted_mode, front) != 0) return -1;
    while ((r = dfn_hop_nn_begin(&m->hop, &job)) == 1)
        if (dfn_hop_nn_end(&m->hop, dfn_res_stage_infer(m->stage, &job)) != 0) return -1;
    if (r < 0) return -1;
    return dfn_hop_end(&m->hop, out) == 0 ? 0 : -1;
}

static int run_identity(int trusted_mode, int enable_cng) {
    FourDfn *dfn = fdfn_create(DFN2_SR, DFN2_N_FFT, enable_cng, NULL);
    FourAecNrResConfig ccfg = core_config(DFN2_SR, DFN2_N_FFT);
    ConvChain *conventional;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    float mics[DFN2_HOP_LEN * FOUR_AEC_NR_RES_CHANNELS];
    float ref[DFN2_HOP_LEN];
    float dfn_out[DFN2_HOP_LEN];
    Complex trusted[DFN2_N_BINS];
    long long frames = -1, commits = -1, skips = -1;
    float min_res = 1.0f;
    int frame, i, k, delay_changes = 0, nonzero_hops = 0;

    if (!dfn || dfn->hop_size != DFN2_HOP_LEN || dfn->bridge != NULL ||
        (DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1) * DFN2_HOP_LEN != 3 * DFN2_HOP_LEN)
        return 1;
    conventional = oracle_create(&ccfg, 1, enable_cng);
    if (!conventional) return 2;
    make_weights(weights);

    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        FourAecFront dp;
        const float *res_gain;
        int any = 0;
        make_hop(mics, ref, frame);
        if (fdfn_process(dfn, mics, ref, weights, trusted_mode, &dp, dfn_out) != 0 ||
            !dp.linear_spectra[0])
            return 10 + frame;
        if (dp.delay.changed) ++delay_changes;
        res_gain = se_sample_chain_last_res_gain(dfn->chain);
        if (res_gain)
            for (k = 0; k < dp.n_freqs; ++k)
                if (res_gain[k] < min_res) min_res = res_gain[k];
        if (trusted_mode) form_trusted(&dp, weights, trusted);
        if (conv_hop(conventional, mics, ref, weights, trusted_mode ? trusted : NULL,
                     conventional_out[frame]) != 0)
            return 200 + frame;
        for (i = 0; i < DFN2_HOP_LEN; ++i) {
            float expected = frame < 2 ? 0.0f : conventional_out[frame - 2][i];
            if (dfn_out[i] != expected) return 300 + frame;
            if (frame >= 2 && expected != 0.0f) any = 1;
        }
        nonzero_hops += any;
    }
    if (delay_changes < 1) return 3;          /* the contract was never exercised */
    if (!(min_res < 0.9f)) return 9;          /* RES never cut: A == B */
    if (nonzero_hops < TEST_FRAMES / 2) return 4;
    dfn_res_stage_get_counters(dfn->stage, &frames, &commits, &skips);
    if (frames != TEST_FRAMES || commits != 0 || skips != TEST_FRAMES - 1)
        return 5;                             /* a reset would leave a gap */
    if (aec_res_set_preset(se_sample_chain_res(dfn->chain), AEC_PRESET_AGGRESSIVE, 0.0f) != 0)
        return 6;
    if (dfn_res_stage_set_atten_lim(dfn->stage, 12.0f) != 0 ||
        dfn_res_stage_set_atten_lim(dfn->stage, NAN) != -1)
        return 7;
    printf("4ch_aec_bf_dfn_res: trusted=%d cng=%d identity over %d hops, "
           "%d delay change(s), min G_res %.3f\n", trusted_mode, enable_cng, TEST_FRAMES,
           delay_changes, min_res);
    conv_destroy(conventional);
    fdfn_destroy(dfn);
    return 0;
}

/* The stimulus must make the post-beam RES cut: a RES-off chain on the same input has to differ
 * from the RES-on chain after the warm-up hops. */
static int run_res_is_active(void) {
    FourAecNrResConfig cfg = core_config(DFN2_SR, DFN2_N_FFT);
    ConvChain *on = oracle_create(&cfg, 1, 0);
    ConvChain *off = oracle_create(&cfg, 0, 0);
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    float mics[DFN2_HOP_LEN * FOUR_AEC_NR_RES_CHANNELS];
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

/* Native grid (the 16 kHz default): the core on its own grid, the stage through the bridge. The
 * conventional chain's NR-off output delayed by the bridge's added delay must match within the
 * resampler round-trip error; a near-end impulse pins the delay; the 48 kHz schedule is the
 * documented one. Reconstructed error entry, weights uniform. */
static int run_native(int sample_rate, int fft_size, int expected_added) {
    FourDfn *dfn = fdfn_create(sample_rate, fft_size, 0, NULL);
    FourAecNrResConfig ccfg = core_config(sample_rate, fft_size);
    ConvChain *conventional;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    static float mics[512 * FOUR_AEC_NR_RES_CHANNELS];
    static float ref[512], out[512];
    static float conv[TEST_FRAMES * 4 * 256];
    static float mine[TEST_FRAMES * 4 * 256];
    static float conv_lp[TEST_FRAMES * 4 * 256];
    static float mine_lp[TEST_FRAMES * 4 * 256];
    int hop, frames = TEST_FRAMES * 4, frame, i, ch, n, added, peak = -1, imp;
    float max_err = 0.0f, max_sig = 0.0f, peak_abs = 0.0f;
    int schedule[8], last, max_err_at = -1;
    long long total_frames = 0;

    if (!dfn || dfn->bridge == NULL) return 1;
    hop = dfn->hop_size;
    added = dfn_rate_bridge_added_delay_samples(dfn->bridge);
    if (added != expected_added) return 2;
    conventional = oracle_create(&ccfg, 1, 0);
    if (!conventional || conventional->hop != hop) return 3;
    make_weights(weights);
    n = frames * hop;
    dfn_fixture_multitone(far_tones, (size_t)n, sample_rate);
    imp = (frames * 3 / 4) * hop + 5;
    for (frame = 0; frame < frames; ++frame) {
        FourAecFront dp;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = far_tones[t];
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                int d = (ECHO_DELAY + 3 * ch) * sample_rate / DFN2_SR;
                float echo = t >= d ? 0.5f * far_tones[t - d] : 0.0f;
                mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo + 0.01f * sinf((float)t * 0.071f + (float)ch) +
                    (t == imp ? 1.0f : 0.0f);
            }
        }
        if (fdfn_process(dfn, mics, ref, weights, 0, &dp, out) != 0 ||
            conv_hop(conventional, mics, ref, weights, NULL, conv + frame * hop) != 0)
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
     * (cut-off 0.35 fs), the conventional one delayed by `added`. */
    dfn_fixture_lowpass(conv, conv_lp, n, 129, 0.35f);
    dfn_fixture_lowpass(mine, mine_lp, n, 129, 0.35f);
    /* The test low-pass truncates at both ends: keep 64 samples clear. */
    for (i = added + 8 * hop + 64; i < n - 64; ++i) {
        float ref_s = conv_lp[i - added], err;
        if (abs(i - (imp + hop + added)) < 4 * hop + 128) continue;
        err = fabsf(mine_lp[i] - ref_s);
        if (err > max_err) { max_err = err; max_err_at = i; }
        if (fabsf(ref_s) > max_sig) max_sig = fabsf(ref_s);
    }
    for (i = imp + hop + added - 2 * hop; i < imp + hop + added + 2 * hop && i < n; ++i) {
        float a = fabsf(mine[i]);
        if (a > peak_abs) { peak_abs = a; peak = i; }
    }
    printf("4ch_aec_bf_dfn_res: core %d Hz/hop %d: added delay %d, pass-band max "
           "error %.2e at %d (of %d; impulse at %d; signal peak %.2f), impulse "
           "peak at +%d\n", sample_rate, hop, added, (double)max_err, max_err_at,
           n, imp, (double)max_sig, peak - imp - hop);
    if (peak != imp + hop + added) return 6;
    if (!(max_err < 1e-3f)) return 7;
    conv_destroy(conventional);
    fdfn_destroy(dfn);
    return 0;
}

/* The core serves 16 and 48 kHz only: the composition has no 8 kHz or off-grid host. */
static int run_rejections(void) {
    FourAecNrResConfig cfg = core_config(8000, 0);
    FourAecNrResMemReq req;
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) == 0) return 1;
    cfg = core_config(44100, 0);
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) == 0) return 2;
    cfg = core_config(DFN2_SR, 512);   /* not a 48 kHz grid */
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) == 0) return 3;
    cfg = core_config(DFN2_SR, 0);
    if (four_aec_nr_res_get_mem_requirements(&cfg, &req) != 0) return 4;
    return 0;
}

/* A reset of every stage (core, chain, hop) returns the composition to the state of a fresh one:
 * the first hops after it reproduce the first hops of a new instance bit for bit, the second pass
 * of a loop equals the first. */
static int run_reset_equals_fresh(int sample_rate, int fft_size) {
    FourDfn *dfn = fdfn_create(sample_rate, fft_size, 1, NULL);
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    static float mics[512 * FOUR_AEC_NR_RES_CHANNELS], ref[512];
    static float first[40][512], again[40][512];
    static float tones[40 * 512];
    int hop, frame, pass, i, ch, e;
    if (!dfn) return 1;
    hop = dfn->hop_size;
    make_weights(weights);
    dfn_fixture_multitone(tones, (size_t)40 * hop, sample_rate);
    e = (ECHO_DELAY * sample_rate) / DFN2_SR;
    for (pass = 0; pass < 2; ++pass) {
        for (frame = 0; frame < 40; ++frame) {
            FourAecFront dp;
            for (i = 0; i < hop; ++i) {
                int t = frame * hop + i;
                ref[i] = tones[t];
                for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                    mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
                        (t >= e ? 0.5f * tones[t - e] : 0.0f) +
                        0.01f * sinf((float)t * 0.071f + (float)ch);
            }
            if (fdfn_process(dfn, mics, ref, weights, 0, &dp, pass ? again[frame] : first[frame]) != 0)
                return 2;
        }
        if (pass == 0) fdfn_reset(dfn);
    }
    for (frame = 0; frame < 40; ++frame)
        if (memcmp(first[frame], again[frame], (size_t)hop * sizeof(float)) != 0)
            return 10 + frame;
    fdfn_destroy(dfn);
    return 0;
}

/* The phased hop (begin / nn_begin ... nn_end / end) against the callback-driven one: a
 * non-identity model (a uniform 0.5 band mask) run by the caller reproduces the configured
 * callback byte for byte at 48 kHz and through the rate bridge, comfort noise on, reconstructed
 * and trusted error spectra; a failed run, a frame left open and a frame never fetched equal the
 * model-free composition. */
static int half_mask_infer(void *user, const DFN2PrepostInputs *in,
                           DFN2PrepostOutputs *out) {
    size_t i;
    (void)user; (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.5f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    return 0;
}

enum { BP_RUN, BP_FAIL, BP_OPEN, BP_UNFETCHED };

static int run_phased(int sample_rate, int fft_size, int behaviour, int trusted_mode) {
    DFN2Model cb_model;
    DFN2ModelIoDescriptor descriptor;
    FourDfn *ph, *cb;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    static float mics[512 * FOUR_AEC_NR_RES_CHANNELS], ref[512], oa[512], ob[512];
    static float tones[100 * 512];
    FourAecFront fa, fb;
    DfnResStageJob job;
    int hop, frame, i, ch, r, nonzero = 0, ran = 0, frames = 80;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    memset(&cb_model, 0, sizeof(cb_model));
    if (behaviour == BP_RUN) {
        cb_model.infer = half_mask_infer;
        cb_model.io_descriptor = &descriptor;
    }
    ph = fdfn_create(sample_rate, fft_size, 1, NULL);
    cb = fdfn_create(sample_rate, fft_size, 1, &cb_model);
    if (!ph || !cb) return 2;
    hop = ph->hop_size;
    make_weights(weights);
    dfn_fixture_multitone(tones, (size_t)frames * hop, sample_rate);
    for (frame = 0; frame < frames; ++frame) {
        int e = (ECHO_DELAY * sample_rate) / DFN2_SR;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = tones[t];
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
                    (t >= e ? 0.5f * tones[t - e] : 0.0f) +
                    0.01f * sinf((float)t * 0.071f + (float)ch);
        }
        if (fdfn_process(cb, mics, ref, weights, trusted_mode, &fb, ob) != 0) return 5;
        if (fdfn_begin(ph, mics, ref, weights, trusted_mode, &fa) != 0) return 6;
        if (behaviour != BP_UNFETCHED) {
            while ((r = dfn_hop_nn_begin(&ph->hop, &job)) == 1) {
                int rc;
                ++ran;
                if (behaviour == BP_OPEN) break;
                rc = behaviour == BP_RUN ? half_mask_infer(NULL, &job.inputs, &job.outputs) : -1;
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
    int rc, trusted, cng;
    make_partition();
    make_far();
    rc = run_rejections();
    if (rc) { fprintf(stderr, "4ch DFN grid rejection row failed: %d\n", rc); return 1; }
    rc = run_res_is_active();
    if (rc) { fprintf(stderr, "4ch DFN stimulus never engaged RES: %d\n", rc); return 1; }
    for (trusted = 0; trusted <= 1; ++trusted) {
        for (cng = 0; cng <= 1; ++cng) {
            rc = run_identity(trusted, cng);
            if (rc != 0) {
                fprintf(stderr, "4ch DFN identity trusted=%d cng=%d failed: %d\n",
                        trusted, cng, rc);
                return 1;
            }
        }
    }
    rc = run_native(16000, 256, 672);
    if (rc) { fprintf(stderr, "4ch DFN 16 kHz/256 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(16000, 512, 629);
    if (rc) { fprintf(stderr, "4ch DFN 16 kHz/512 bridge gate failed: %d\n", rc); return 1; }
    {
        static const int grids[3][2] = {{48000, 1024}, {16000, 256}, {16000, 512}};
        int g, b, t;
        for (g = 0; g < 3; ++g) {
            rc = run_reset_equals_fresh(grids[g][0], grids[g][1]);
            if (rc) {
                fprintf(stderr, "4ch DFN reset %d Hz/%d failed: %d\n",
                        grids[g][0], grids[g][1], rc);
                return 1;
            }
            for (b = 0; b < 4; ++b)
                for (t = 0; t <= 1; ++t) {
                    rc = run_phased(grids[g][0], grids[g][1], b, t);
                    if (rc) {
                        fprintf(stderr, "4ch DFN phased %d Hz/%d behaviour %d trusted %d failed: %d\n",
                                grids[g][0], grids[g][1], b, t, rc);
                        return 1;
                    }
                }
        }
    }
    puts("4ch_aec_bf_dfn_res: PASS");
    return 0;
}
