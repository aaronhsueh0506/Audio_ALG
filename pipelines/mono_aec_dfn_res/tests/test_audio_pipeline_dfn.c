/* mono_aec_dfn_res acceptance test.
 *
 * 48 kHz backbone gate: with the identity model (no callback) the wrapper
 * must be the conventional pipeline with enable_nr=0, delayed by exactly two
 * hops, byte for byte, with comfort noise OFF and ON. The stimulus is an
 * echo delayed by 384 samples under the matched delay estimator, and the
 * test asserts that (a) the estimator reported a delay change during the run
 * and (b) the residual suppressor actually cut (min G_res < 0.9): without
 * (a) the "no reset on a delay change" contract would be unexercised,
 * without (b) the post-RES spectrum would equal the pre-RES one and an
 * estimate/apply swap would be invisible.
 *
 * 16 kHz (the default) and 8 kHz gates: the host runs on its own grid and
 * the bridge takes the stage to 48 kHz and back, so the output is the
 * conventional NR-off output delayed by the bridge's added delay within the
 * resamplers' pass-band error (a tolerance, not a memcmp), the delay itself
 * is pinned by a near-end impulse, and the stage's per-hop schedule is the
 * documented one. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline.h"
#include "audio_pipeline_dfn.h"
#include "tests/dfn_test_fixture.h"

#define TEST_FRAMES 64
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

static MonoAecDfnResConfig dfn_config(int enable_cng) {
    MonoAecDfnResConfig cfg = mono_aec_dfn_res_default_config();
    cfg.host.sample_rate = DFN2_SR;
    cfg.host.fft_size = DFN2_N_FFT;
    cfg.host.enable_cng = enable_cng;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    return cfg;
}

static int run_identity(int enable_cng) {
    MonoAecDfnResConfig dcfg = dfn_config(enable_cng);
    AudioPipelineConfig bcfg = audio_pipeline_default_config(DFN2_SR);
    MonoAecDfnRes *dfn;
    AudioPipeline *baseline;
    float mic[DFN2_HOP_LEN], ref[DFN2_HOP_LEN], out[DFN2_HOP_LEN];
    long long frames = -1, commits = -1, skips = -1;
    float min_res = 1.0f;
    int frame, i, k, delay_changes = 0, nonzero_hops = 0;

    dfn = mono_aec_dfn_res_create(&dcfg);
    if (!dfn || mono_aec_dfn_res_hop_size(dfn) != DFN2_HOP_LEN ||
        mono_aec_dfn_res_lookahead_samples(dfn) != 3 * DFN2_HOP_LEN ||
        mono_aec_dfn_res_get_bridge(dfn) != NULL)
        return 1;

    bcfg = audio_pipeline_default_config(DFN2_SR);
    bcfg.fft_size = DFN2_N_FFT;
    bcfg.enable_nr = 0;
    bcfg.enable_res = 1;
    bcfg.enable_cng = enable_cng;
    baseline = audio_pipeline_create(&bcfg);
    if (!baseline) return 2;

    for (frame = 0; frame < TEST_FRAMES; ++frame) {
        AecLinearContext linear;
        AecResContext ctx;
        int any = 0;
        make_hop(mic, ref, frame);
        if (mono_aec_dfn_res_process(dfn, mic, ref, out) != 0 ||
            audio_pipeline_process(baseline, mic, ref, saved[frame]) != 0)
            return 10 + frame;
        aec_get_linear_context(mono_aec_dfn_res_get_aec(dfn), &linear);
        if (linear.delay_state == AEC_LINEAR_DELAY_CHANGED) ++delay_changes;
        aec_get_res_context(mono_aec_dfn_res_get_aec(dfn), &ctx);
        if (ctx.res_gain)
            for (k = 0; k < ctx.n_freqs; ++k)
                if (ctx.res_gain[k] < min_res) min_res = ctx.res_gain[k];
        for (i = 0; i < DFN2_HOP_LEN; ++i) {
            float expected = frame < 2 ? 0.0f : saved[frame - 2][i];
            if (out[i] != expected) return 100 + frame;
            if (frame >= 2 && expected != 0.0f) any = 1;
        }
        nonzero_hops += any;
    }
    if (delay_changes < 1) return 3;          /* the contract was never exercised */
    if (!(min_res < 0.9f)) return 4;          /* RES never cut: A == B */
    if (nonzero_hops < TEST_FRAMES / 2) return 5;
    dfn_res_stage_get_counters(mono_aec_dfn_res_get_stage(dfn),
                               &frames, &commits, &skips);
    if (frames != TEST_FRAMES || commits != 0 || skips != TEST_FRAMES - 1)
        return 6;                             /* a reset would leave a gap */
    if (mono_aec_dfn_res_set_aec_preset(dfn, AEC_PRESET_AGGRESSIVE, 0.0f) != 0)
        return 7;
    if (mono_aec_dfn_res_set_atten_lim(dfn, 12.0f) != 0 ||
        mono_aec_dfn_res_set_atten_lim(dfn, NAN) != -1)
        return 8;
    printf("mono_aec_dfn_res: cng=%d identity over %d hops, %d delay change(s), "
           "min G_res %.3f\n", enable_cng, TEST_FRAMES, delay_changes, min_res);
    audio_pipeline_destroy(baseline);
    mono_aec_dfn_res_destroy(dfn);
    return 0;
}

/* Native grid (8 or 16 kHz): host on its own grid, the stage through the
 * bridge. The conventional twin's NR-off output delayed by the bridge's
 * added delay must match within the resampler round-trip error; a near-end
 * impulse pins the delay; the 48 kHz frame schedule is the documented one. */
static int run_native(int sample_rate, int fft_size, int expected_added) {
    MonoAecDfnResConfig dcfg = mono_aec_dfn_res_default_config();
    AudioPipelineConfig bcfg = audio_pipeline_default_config(sample_rate);
    MonoAecDfnRes *dfn;
    AudioPipeline *baseline;
    static float mic[512], ref[512], out[512];
    static float conv[TEST_FRAMES * 4 * 256];
    static float mine[TEST_FRAMES * 4 * 256];
    static float conv_lp[TEST_FRAMES * 4 * 256];
    static float mine_lp[TEST_FRAMES * 4 * 256];
    int hop, frames = TEST_FRAMES * 4, frame, i, n, added, peak = -1, imp;
    float max_err = 0.0f, max_sig = 0.0f, peak_abs = 0.0f;
    int schedule[8], last, max_err_at = -1;
    long long total_frames = 0;

    dcfg.host.sample_rate = sample_rate;
    dcfg.host.fft_size = fft_size;
    dcfg.erb_fwd = erb_fwd;
    dcfg.erb_inv = erb_inv;
    dfn = mono_aec_dfn_res_create(&dcfg);
    if (!dfn || mono_aec_dfn_res_get_bridge(dfn) == NULL) return 1;
    hop = mono_aec_dfn_res_hop_size(dfn);
    added = dfn_rate_bridge_added_delay_samples(mono_aec_dfn_res_get_bridge(dfn));
    if (added != expected_added ||
        mono_aec_dfn_res_lookahead_samples(dfn) != hop + added)
        return 2;
    bcfg.fft_size = fft_size;
    bcfg.enable_nr = 0;
    bcfg.enable_res = 1;
    bcfg.enable_cng = 0;
    baseline = audio_pipeline_create(&bcfg);
    if (!baseline || audio_pipeline_hop_size(baseline) != hop) return 3;
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
        if (mono_aec_dfn_res_process(dfn, mic, ref, out) != 0 ||
            audio_pipeline_process(baseline, mic, ref, conv + frame * hop) != 0)
            return 10 + frame;
        memcpy(mine + frame * hop, out, (size_t)hop * sizeof(float));
        dfn_rate_bridge_get_counters(mono_aec_dfn_res_get_bridge(dfn),
                                     &total_frames, &last);
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
    /* Tolerance identity against the conventional output, after warm-up
     * and excluding the impulse's neighbourhood. */
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
    /* Delay pin: the impulse leaves the conventional host at imp + hop and
     * the bridge at imp + hop + added. */
    for (i = imp + hop + added - 2 * hop; i < imp + hop + added + 2 * hop && i < n; ++i) {
        float a = fabsf(mine[i]);
        if (a > peak_abs) { peak_abs = a; peak = i; }
    }
    printf("mono_aec_dfn_res: host %d Hz/fft %d: added delay %d, pass-band "
           "max error %.2e at %d (of %d; impulse at %d; signal peak %.2f), "
           "impulse peak at +%d\n",
           sample_rate, fft_size, added, (double)max_err, max_err_at, n, imp,
           (double)max_sig, peak - imp - hop);
    if (peak != imp + hop + added) return 6;
    if (!(max_err < 1e-3f)) return 7;
    audio_pipeline_destroy(baseline);
    mono_aec_dfn_res_destroy(dfn);
    return 0;
}

static int run_rejections(void) {
    MonoAecDfnResConfig cfg;
    MonoAecDfnResMemReq req, bad;
    void *mem = NULL;
    MonoAecDfnRes *p;

    cfg = dfn_config(1); cfg.host.sample_rate = 44100;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 1;
    cfg = dfn_config(1); cfg.host.fft_size = 512;   /* not a 48 kHz grid */
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 2;
    cfg = dfn_config(2);   /* the host's own boolean gate refuses a 2 */
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 3;
    cfg = dfn_config(1); cfg.host.enable_nr = 1;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 12;
    cfg = dfn_config(1); cfg.host.enable_res = 0;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 13;
    cfg = dfn_config(1); cfg.host.aec_only = 1;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 14;
    cfg = dfn_config(1); cfg.erb_inv = NULL;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) return 4;
    cfg = dfn_config(1); erb_inv[9] = 0.999f;
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) == 0) { make_partition(); return 5; }
    make_partition();

    cfg = dfn_config(1);
    if (mono_aec_dfn_res_get_mem_requirements(&cfg, &req) != 0) return 6;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return 7;
    p = mono_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!p) { free(mem); return 8; }
    mono_aec_dfn_res_destroy(p);
    bad = req; bad.build_flags_hash ^= 1u;
    if (mono_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 9; }
    bad = req; bad.layout_version += 1u;
    if (mono_aec_dfn_res_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 10; }
    if (mono_aec_dfn_res_init(mem, (size_t)req.bytes - 16u, &cfg)) { free(mem); return 11; }
    printf("mono_aec_dfn_res: host DSP pool %llu bytes (model weights external)\n",
           (unsigned long long)req.bytes);
    free(mem);
    return 0;
}

/* The phased hop (process_begin / nn_begin ... nn_end / process_end) against
 * process(): a non-identity model (a uniform 0.5 band mask) run by the caller
 * reproduces the configured callback byte for byte at 48 kHz and through the
 * rate bridge, with comfort noise on; a failed run, a frame left open and a
 * frame never fetched equal the model-free pipeline. */
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

static int run_phased(int sample_rate, int fft_size, int behaviour) {
    MonoAecDfnResConfig ph_cfg = dfn_config(1);
    MonoAecDfnResConfig cb_cfg;
    DFN2ModelIoDescriptor descriptor;
    MonoAecDfnRes *ph, *cb;
    static float mic[512], ref[512], oa[512], ob[512];
    static float tones[TEST_FRAMES * 4 * 256 * 2];
    DfnResStageJob job;
    int hop, frame, i, r, nonzero = 0, ran = 0, frames = 80;

    ph_cfg.host.sample_rate = sample_rate;
    ph_cfg.host.fft_size = fft_size;
    cb_cfg = ph_cfg;
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    if (behaviour == BP_RUN) {
        cb_cfg.model.infer = half_mask_infer;
        cb_cfg.model.io_descriptor = &descriptor;
    }
    ph = mono_aec_dfn_res_create(&ph_cfg);
    cb = mono_aec_dfn_res_create(&cb_cfg);
    if (!ph || !cb) return 2;
    hop = mono_aec_dfn_res_hop_size(ph);
    dfn_fixture_multitone(tones, (size_t)frames * hop, sample_rate);
    for (frame = 0; frame < frames; ++frame) {
        int e = (ECHO_DELAY * sample_rate) / DFN2_SR;
        for (i = 0; i < hop; ++i) {
            int t = frame * hop + i;
            ref[i] = tones[t];
            mic[i] = (t >= e ? 0.5f * tones[t - e] : 0.0f) + 0.01f * sinf((float)t * 0.071f);
        }
        if (mono_aec_dfn_res_process(cb, mic, ref, ob) != 0) return 3;
        if (mono_aec_dfn_res_process_begin(ph, mic, ref) != 0) return 4;
        if (behaviour != BP_UNFETCHED) {
            while ((r = mono_aec_dfn_res_nn_begin(ph, &job)) == 1) {
                int rc;
                ++ran;
                if (behaviour == BP_OPEN) break;
                rc = behaviour == BP_RUN ? half_mask_infer(NULL, &job.inputs, &job.outputs) : -1;
                if (mono_aec_dfn_res_nn_end(ph, rc) != 0) return 5;
            }
            if (r < 0) return 6;
        }
        if (mono_aec_dfn_res_process_end(ph, oa) != 0) return 7;
        if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) return 10 + frame;
        for (i = 0; i < hop; ++i) if (oa[i] != 0.0f) nonzero = 1;
    }
    if (!nonzero) return 8;
    if (behaviour != BP_UNFETCHED && ran == 0) return 9;
    mono_aec_dfn_res_destroy(ph);
    mono_aec_dfn_res_destroy(cb);
    return 0;
}

static int run_phased_sequence(void) {
    MonoAecDfnResConfig cfg = dfn_config(0);
    MonoAecDfnRes *p = mono_aec_dfn_res_create(&cfg);
    static float mic[DFN2_HOP_LEN], ref[DFN2_HOP_LEN], out[DFN2_HOP_LEN];
    DfnResStageJob job;
    if (!p) return 1;
    if (mono_aec_dfn_res_nn_begin(p, &job) != -1 || mono_aec_dfn_res_nn_end(p, 0) != -1 ||
        mono_aec_dfn_res_process_end(p, out) != -1)
        return 2;
    if (mono_aec_dfn_res_process_begin(NULL, mic, ref) != -1 ||
        mono_aec_dfn_res_process_begin(p, NULL, ref) != -1)
        return 3;
    if (mono_aec_dfn_res_process_begin(p, mic, ref) != 0 ||
        mono_aec_dfn_res_process_begin(p, mic, ref) != -1)
        return 4;
    if (mono_aec_dfn_res_process_end(p, out) != 0 ||
        mono_aec_dfn_res_process_end(p, out) != -1)
        return 5;
    mono_aec_dfn_res_destroy(p);
    return 0;
}

int main(void) {
    int rc;
    make_partition();
    make_far();
    rc = run_rejections();
    if (rc) { fprintf(stderr, "mono DFN rejection row failed: %d\n", rc); return 1; }
    rc = run_identity(0);
    if (rc) { fprintf(stderr, "mono DFN identity (cng off) failed: %d\n", rc); return 1; }
    rc = run_identity(1);
    if (rc) { fprintf(stderr, "mono DFN identity (cng on) failed: %d\n", rc); return 1; }
    rc = run_native(16000, 256, 672);
    if (rc) { fprintf(stderr, "mono DFN 16 kHz/256 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(16000, 512, 629);
    if (rc) { fprintf(stderr, "mono DFN 16 kHz/512 bridge gate failed: %d\n", rc); return 1; }
    rc = run_native(8000, 256, 330);
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
    rc = run_phased_sequence();
    if (rc) { fprintf(stderr, "mono DFN phased sequence row failed: %d\n", rc); return 1; }
    puts("mono_aec_dfn_res: PASS");
    return 0;
}
