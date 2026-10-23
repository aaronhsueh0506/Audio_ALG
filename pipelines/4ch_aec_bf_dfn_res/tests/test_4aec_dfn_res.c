/* 4aec_dfn_res acceptance test. Backbone gate: with the identity model (no callback) the
 * wrapper must equal the conventional chain with the denoiser off (ConvChain), delayed by
 * exactly two hops, byte for byte, over both beamformed-error entries (reconstructed and
 * trusted spectrum) and with comfort noise OFF and ON. The stimulus is a 384-sample echo
 * under the shared matched delay estimator; the test asserts the estimator reported a
 * delay change (the "no reset on a delay change" contract is exercised) and that the
 * post-beam RES cuts (a RES-off core produces a different output). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_dfn_res.h"
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

static FourAecNrResConfig core_config(int enable_res, int enable_cng) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(DFN2_SR);
    cfg.fft_size = DFN2_N_FFT;
    cfg.sample_rate = DFN2_SR;
    cfg.enable_post = 1;
    cfg.enable_res = enable_res;
    cfg.enable_nr = 0;
    cfg.enable_cng = enable_cng;
    return cfg;
}

static int run_identity(int trusted_mode, int enable_cng) {
    FourAecDfnResConfig dcfg = four_aec_dfn_res_default_config();
    FourAecNrResConfig ccfg = core_config(1, enable_cng);
    FourAecDfnRes *dfn;
    ConvChain *conventional;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    float mics[DFN2_HOP_LEN * FOUR_AEC_NR_RES_CHANNELS];
    float ref[DFN2_HOP_LEN];
    float dfn_out[DFN2_HOP_LEN];
    Complex trusted[DFN2_N_BINS];
    long long frames = -1, commits = -1, skips = -1;
    int frame, i, delay_changes = 0, nonzero_hops = 0;

    dcfg.front_end.sample_rate = DFN2_SR;
    dcfg.front_end.fft_size = DFN2_N_FFT;
    dcfg.front_end.enable_cng = enable_cng;
    dcfg.erb_fwd = erb_fwd;
    dcfg.erb_inv = erb_inv;
    dfn = four_aec_dfn_res_create(&dcfg);
    if (!dfn || four_aec_dfn_res_hop_size(dfn) != DFN2_HOP_LEN ||
        four_aec_dfn_res_lookahead_samples(dfn) != 3 * DFN2_HOP_LEN ||
        four_aec_dfn_res_get_bridge(dfn) != NULL)
        return 1;
    conventional = conv_create(&ccfg);
    if (!conventional) return 2;
    make_weights(weights);

    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        FourAecFront dp;
        int any = 0;
        make_hop(mics, ref, frame);
        if (four_aec_dfn_res_front_process(dfn, mics, ref, &dp) != 0 ||
            !dp.linear_spectra[0])
            return 10 + frame;
        if (dp.delay.changed) ++delay_changes;
        if (trusted_mode) form_trusted(&dp, weights, trusted);
        if (four_aec_dfn_res_process(dfn, weights, trusted_mode ? trusted : NULL,
                                     dfn_out) != 0)
            return 100 + frame;
        if (conv_hop(conventional, mics, ref, weights,
                     trusted_mode ? trusted : NULL, conventional_out[frame]) != 0)
            return 200 + frame;
        for (i = 0; i < DFN2_HOP_LEN; ++i) {
            float expected = frame < 2 ? 0.0f : conventional_out[frame - 2][i];
            if (dfn_out[i] != expected) return 300 + frame;
            if (frame >= 2 && expected != 0.0f) any = 1;
        }
        nonzero_hops += any;
    }
    if (delay_changes < 1) return 3;
    if (nonzero_hops < TEST_FRAMES / 2) return 4;
    dfn_res_stage_get_counters(four_aec_dfn_res_get_stage(dfn),
                               &frames, &commits, &skips);
    if (frames != TEST_FRAMES || commits != 0 || skips != TEST_FRAMES - 1)
        return 5;
    if (four_aec_dfn_res_set_aec_preset(dfn, AEC_PRESET_AGGRESSIVE, 0.0f) != 0)
        return 6;
    if (four_aec_dfn_res_set_atten_lim(dfn, 12.0f) != 0 ||
        four_aec_dfn_res_set_atten_lim(dfn, NAN) != -1)
        return 7;
    printf("4aec_dfn_res: trusted=%d cng=%d identity over %d hops, "
           "%d delay change(s)\n", trusted_mode, enable_cng, TEST_FRAMES,
           delay_changes);
    conv_destroy(conventional);
    four_aec_dfn_res_destroy(dfn);
    return 0;
}

/* The stimulus must make the post-beam RES cut: a RES-off core on the same
 * input has to differ from the RES-on core after the warm-up hops. */
static int run_res_is_active(void) {
    FourAecNrResConfig on_cfg = core_config(1, 0);
    FourAecNrResConfig off_cfg = core_config(0, 0);
    ConvChain *on = conv_create(&on_cfg);
    ConvChain *off = conv_create(&off_cfg);
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

/* Native grid (the 16 kHz default and 8 kHz): the core on its own grid, the
 * stage through the bridge. The conventional chain's NR-off output delayed by
 * the bridge's added delay must match within the resampler round-trip error;
 * a near-end impulse pins the delay; the 48 kHz schedule is the documented
 * one. Reconstructed post entry, weights uniform. */
static int run_native(int sample_rate, int fft_size, int expected_added) {
    FourAecDfnResConfig dcfg = four_aec_dfn_res_default_config();
    FourAecNrResConfig ccfg = four_aec_nr_res_default_config(sample_rate);
    FourAecDfnRes *dfn;
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

    dcfg.front_end.sample_rate = sample_rate;
    dcfg.front_end.fft_size = fft_size;
    dcfg.erb_fwd = erb_fwd;
    dcfg.erb_inv = erb_inv;
    dfn = four_aec_dfn_res_create(&dcfg);
    if (!dfn || four_aec_dfn_res_get_bridge(dfn) == NULL) return 1;
    hop = four_aec_dfn_res_hop_size(dfn);
    added = dfn_rate_bridge_added_delay_samples(four_aec_dfn_res_get_bridge(dfn));
    if (added != expected_added ||
        four_aec_dfn_res_lookahead_samples(dfn) != hop + added)
        return 2;
    ccfg.fft_size = fft_size;
    ccfg.enable_post = 1; ccfg.enable_res = 1; ccfg.enable_nr = 0; ccfg.enable_cng = 0;
    conventional = conv_create(&ccfg);
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
        if (four_aec_dfn_res_front_process(dfn, mics, ref, &dp) != 0 ||
            four_aec_dfn_res_process(dfn, weights, NULL, out) != 0 ||
            conv_hop(conventional, mics, ref, weights, NULL, conv + frame * hop) != 0)
            return 10 + frame;
        memcpy(mine + frame * hop, out, (size_t)hop * sizeof(float));
        dfn_rate_bridge_get_counters(four_aec_dfn_res_get_bridge(dfn),
                                     &total_frames, &last);
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
    printf("4aec_dfn_res: core %d Hz/hop %d: added delay %d, pass-band max "
           "error %.2e at %d (of %d; impulse at %d; signal peak %.2f), impulse "
           "peak at +%d\n", sample_rate, hop, added, (double)max_err, max_err_at,
           n, imp, (double)max_sig, peak - imp - hop);
    if (peak != imp + hop + added) return 6;
    if (!(max_err < 1e-3f)) return 7;
    conv_destroy(conventional);
    four_aec_dfn_res_destroy(dfn);
    return 0;
}

static int run_rejections(void) {
    FourAecDfnResConfig cfg;
    FourAecDfnResMemReq req, bad;
    void *mem = NULL;
    FourAecDfnRes *p;

    cfg = four_aec_dfn_res_default_config();
    cfg.erb_fwd = erb_fwd; cfg.erb_inv = erb_inv;
    cfg.front_end.enable_nr = 1;
    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 1;
    cfg.front_end.enable_nr = 0; cfg.front_end.enable_res = 0;
    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 2;
    cfg.front_end.enable_res = 1; cfg.front_end.sample_rate = 44100;
    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 3;
    cfg = four_aec_dfn_res_default_config();
    cfg.front_end.sample_rate = DFN2_SR; cfg.front_end.fft_size = DFN2_N_FFT;
    cfg.erb_fwd = erb_fwd; cfg.erb_inv = erb_inv;
    erb_inv[9] = 0.999f;
    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) { make_partition(); return 4; }
    make_partition();

    if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) != 0) return 5;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return 6;
    p = four_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!p) { free(mem); return 7; }
    four_aec_dfn_res_destroy(p);
    bad = req; bad.build_flags_hash ^= 1u;
    if (four_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 8; }
    bad = req; bad.layout_version += 1u;
    if (four_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 9; }
    if (four_aec_dfn_res_init(mem, (size_t)req.bytes - 16u, &cfg)) { free(mem); return 10; }
    printf("4aec_dfn_res: 48 kHz host DSP pool %llu bytes (model weights external)\n",
           (unsigned long long)req.bytes);
    free(mem);
    return 0;
}

/* The phased hop (process_begin / nn_begin ... nn_end / process_end) against
 * process(): a non-identity model (a uniform 0.5 band mask) run by the caller
 * reproduces the configured callback byte for byte at 48 kHz and through the
 * rate bridge, comfort noise on, reconstructed and trusted error spectra; a
 * failed run, a frame left open and a frame never fetched equal the
 * model-free pipeline. */
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
    FourAecDfnResConfig ph_cfg = four_aec_dfn_res_default_config();
    FourAecDfnResConfig cb_cfg;
    DFN2ModelIoDescriptor descriptor;
    FourAecDfnRes *ph, *cb;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    static Complex trusted[DFN2_N_BINS];
    static float mics[512 * FOUR_AEC_NR_RES_CHANNELS], ref[512], oa[512], ob[512];
    static float tones[100 * 512];
    FourAecFront fa, fb;
    DfnResStageJob job;
    int hop, frame, i, ch, r, nonzero = 0, ran = 0, frames = 80;

    ph_cfg.front_end.sample_rate = sample_rate;
    ph_cfg.front_end.fft_size = fft_size;
    ph_cfg.front_end.enable_cng = 1;
    ph_cfg.erb_fwd = erb_fwd;
    ph_cfg.erb_inv = erb_inv;
    cb_cfg = ph_cfg;
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    if (behaviour == BP_RUN) {
        cb_cfg.model.infer = half_mask_infer;
        cb_cfg.model.io_descriptor = &descriptor;
    }
    ph = four_aec_dfn_res_create(&ph_cfg);
    cb = four_aec_dfn_res_create(&cb_cfg);
    if (!ph || !cb) return 2;
    hop = four_aec_dfn_res_hop_size(ph);
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
        if (four_aec_dfn_res_front_process(cb, mics, ref, &fb) != 0) return 3;
        if (four_aec_dfn_res_front_process(ph, mics, ref, &fa) != 0) return 4;
        if (trusted_mode) {
            form_trusted(&fb, weights, trusted);
            if (four_aec_dfn_res_process(cb, weights, trusted, ob) != 0) return 5;
            form_trusted(&fa, weights, trusted);
            if (four_aec_dfn_res_process_begin(ph, weights, trusted) != 0) return 6;
        } else {
            if (four_aec_dfn_res_process(cb, weights, NULL, ob) != 0) return 5;
            if (four_aec_dfn_res_process_begin(ph, weights, NULL) != 0) return 6;
        }
        if (behaviour != BP_UNFETCHED) {
            while ((r = four_aec_dfn_res_nn_begin(ph, &job)) == 1) {
                int rc;
                ++ran;
                if (behaviour == BP_OPEN) break;
                rc = behaviour == BP_RUN ? half_mask_infer(NULL, &job.inputs, &job.outputs) : -1;
                if (four_aec_dfn_res_nn_end(ph, rc) != 0) return 7;
            }
            if (r < 0) return 8;
        }
        if (four_aec_dfn_res_process_end(ph, oa) != 0) return 9;
        if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) return 100 + frame;
        for (i = 0; i < hop; ++i) if (oa[i] != 0.0f) nonzero = 1;
    }
    if (!nonzero) return 10;
    if (behaviour != BP_UNFETCHED && ran == 0) return 11;
    four_aec_dfn_res_destroy(ph);
    four_aec_dfn_res_destroy(cb);
    return 0;
}

static int run_phased_sequence(void) {
    FourAecDfnResConfig cfg = four_aec_dfn_res_default_config();
    FourAecDfnRes *p;
    Complex weights[FOUR_AEC_NR_RES_CHANNELS * DFN2_N_BINS];
    static float mics[DFN2_HOP_LEN * FOUR_AEC_NR_RES_CHANNELS], ref[DFN2_HOP_LEN], out[DFN2_HOP_LEN];
    FourAecFront front;
    DfnResStageJob job;
    cfg.front_end.sample_rate = DFN2_SR;
    cfg.front_end.fft_size = DFN2_N_FFT;
    cfg.erb_fwd = erb_fwd; cfg.erb_inv = erb_inv;
    p = four_aec_dfn_res_create(&cfg);
    if (!p) return 1;
    make_weights(weights);
    if (four_aec_dfn_res_process_begin(p, weights, NULL) != FOUR_AEC_NR_RES_SEQUENCE_ERROR)
        return 2;                                   /* no front frame yet */
    if (four_aec_dfn_res_nn_begin(p, &job) != FOUR_AEC_NR_RES_SEQUENCE_ERROR ||
        four_aec_dfn_res_nn_end(p, 0) != FOUR_AEC_NR_RES_SEQUENCE_ERROR ||
        four_aec_dfn_res_process_end(p, out) != FOUR_AEC_NR_RES_SEQUENCE_ERROR)
        return 3;
    if (four_aec_dfn_res_front_process(p, mics, ref, &front) != 0) return 4;
    if (four_aec_dfn_res_process_begin(NULL, weights, NULL) != FOUR_AEC_NR_RES_INVALID_ARGUMENT)
        return 5;
    if (four_aec_dfn_res_process_begin(p, weights, NULL) != 0 ||
        four_aec_dfn_res_process_begin(p, weights, NULL) != FOUR_AEC_NR_RES_SEQUENCE_ERROR)
        return 6;
    if (four_aec_dfn_res_process_end(p, out) != 0 ||
        four_aec_dfn_res_process_end(p, out) != FOUR_AEC_NR_RES_SEQUENCE_ERROR)
        return 7;
    four_aec_dfn_res_destroy(p);
    return 0;
}

int main(void) {
    int rc, trusted, cng;
    make_partition();
    make_far();
    rc = run_rejections();
    if (rc) { fprintf(stderr, "4ch DFN rejection row failed: %d\n", rc); return 1; }
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
    {   /* 8 kHz is not a core grid, so the wrapper must refuse it too. */
        FourAecDfnResConfig cfg = four_aec_dfn_res_default_config();
        FourAecDfnResMemReq req;
        cfg.erb_fwd = erb_fwd; cfg.erb_inv = erb_inv;
        cfg.front_end.sample_rate = 8000;
        if (four_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) {
            fputs("4ch DFN accepted 8 kHz, which the core does not offer\n", stderr);
            return 1;
        }
    }
    {
        static const int grids[3][2] = {{48000, 1024}, {16000, 256}, {16000, 512}};
        int g, b, t;
        for (g = 0; g < 3; ++g)
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
    rc = run_phased_sequence();
    if (rc) { fprintf(stderr, "4ch DFN phased sequence row failed: %d\n", rc); return 1; }
    puts("4aec_dfn_res: PASS");
    return 0;
}
