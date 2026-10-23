/* dfn_res_stage acceptance test (no AEC, no synthesis): the two-hop identity
 * through the cascade with distinct estimate/apply inputs, the absolute scale
 * at the model boundary, the fail-open path, reset parity, the reject-first
 * configuration rows and the stale-pool gate. The two inputs differ in every
 * frame, so an estimate/apply swap inside the stage would fail the identity
 * rows. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_res_stage.h"
#include "audio_window.h"
#include "tests/dfn_test_fixture.h"

#define FRAMES 6
#define SCALE_FRAMES 8
#define CAP_ERB  (DFN2_MODEL_INPUT_FRAMES * DFN2_N_ERB)
#define CAP_SPEC (2 * DFN2_MODEL_INPUT_FRAMES * DFN2_DF_BINS)

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];

static void make_partition(void) {
    dfn_unit_partition(erb_fwd, erb_inv);
}

static int failing_infer(void *user, const DFN2PrepostInputs *in,
                         DFN2PrepostOutputs *out) {
    (void)user; (void)in; (void)out;
    return -1;
}

/* Frame-dependent, mutually distinct estimate and apply spectra. */
static void fill_frame(Complex *estimate, Complex *apply, int frame) {
    int k;
    for (k = 0; k < DFN2_N_BINS; ++k) {
        estimate[k].r = (float)(frame * 1000 + k) * 0.125f;
        estimate[k].i = (float)(frame * 1000 - k) * 0.0625f;
        apply[k].r = (float)(frame * 2000 + k) * 0.25f;
        apply[k].i = (float)(frame * 2000 - k) * 0.125f;
    }
}

/* ---- absolute scale at the model boundary --------------------------------
 * The stage hands DFN2 the host's audio-scale spectrum times 2^-5. The fail-open and
 * identity paths reproduce their input at any scale (IN_SCALE * OUT_SCALE == 1), so only
 * this row sees a wrong factor: it drives one signal down the product path and DFN2's own
 * C TIME-mode analysis and compares the ERB and complex feature windows the model
 * receives. That pins the scale factor, the analysis window and the frame assembly
 * ([previous hop | this hop], 50% overlap, no centring). */
typedef struct ScaleCapture {
    float erb[SCALE_FRAMES][CAP_ERB];
    float spec[SCALE_FRAMES][CAP_SPEC];
    size_t erb_elements;
    size_t spec_elements;
    int frames;
    int overflow;
} ScaleCapture;

static float g_pcm[SCALE_FRAMES * DFN2_HOP_LEN];

static void make_pcm(void) {
    unsigned lcg = 20260915u;
    int n;
    for (n = 0; n < SCALE_FRAMES * DFN2_HOP_LEN; ++n) {
        float s = (float)n / (float)DFN2_SR;
        lcg = lcg * 1664525u + 1013904223u;
        g_pcm[n] = 0.30f * sinf(2.0f * (float)M_PI * 300.0f * s) +
                   0.15f * sinf(2.0f * (float)M_PI * 1200.0f * s) +
                   0.08f * sinf(2.0f * (float)M_PI * 4000.0f * s) +
                   0.02f * ((float)(lcg >> 8) / 16777216.0f - 0.5f);
    }
}

/* The shared periodic sqrt-Hann used by the host front ends. */
static void host_root_hann(float *window, int n) {
    ac_window_build(window, n);
}

static int capturing_infer(void *user, const DFN2PrepostInputs *in,
                           DFN2PrepostOutputs *out) {
    ScaleCapture *cap = (ScaleCapture *)user;
    if (in->erb_window_elements > CAP_ERB || in->spec_window_elements > CAP_SPEC ||
        cap->frames >= SCALE_FRAMES)
        cap->overflow = 1;
    else if (cap->frames < SCALE_FRAMES) {
        memcpy(cap->erb[cap->frames], in->erb_window,
               in->erb_window_elements * sizeof(float));
        memcpy(cap->spec[cap->frames], in->spec_window,
               in->spec_window_elements * sizeof(float));
        cap->erb_elements = in->erb_window_elements;
        cap->spec_elements = in->spec_window_elements;
        ++cap->frames;
    }
    return dfn_fixture_identity_infer(NULL, in, out);
}

static DfnResStageConfig base_config(void) {
    DfnResStageConfig cfg = dfn_res_stage_default_config(DFN2_SR);
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    return cfg;
}

/* The product path: the host's analysis, audio scale, through the stage. */
static int capture_via_stage(ScaleCapture *cap, FftHandle *fft) {
    static float prev[DFN2_HOP_LEN], frame[DFN2_N_FFT], win[DFN2_N_FFT];
    static Complex spec[DFN2_N_BINS], out[DFN2_N_BINS];
    DFN2ModelIoDescriptor descriptor;
    DfnResStageConfig cfg = base_config();
    DfnResStage *s;
    int hop, k;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    cfg.model.infer = capturing_infer;
    cfg.model.user = cap;
    cfg.model.io_descriptor = &descriptor;
    s = dfn_res_stage_create(&cfg);
    if (!s) return 2;
    host_root_hann(win, DFN2_N_FFT);
    memset(prev, 0, sizeof(prev));
    for (hop = 0; hop < SCALE_FRAMES; ++hop) {
        const float *cur = g_pcm + (size_t)hop * DFN2_HOP_LEN;
        for (k = 0; k < DFN2_HOP_LEN; ++k) {
            frame[k] = prev[k] * win[k];
            frame[DFN2_HOP_LEN + k] = cur[k] * win[DFN2_HOP_LEN + k];
        }
        fft_forward(fft, frame, spec);
        memcpy(prev, cur, sizeof(prev));
        if (dfn_res_stage_process(s, spec, spec, out) < 0) {
            dfn_res_stage_destroy(s);
            return 3;
        }
    }
    dfn_res_stage_destroy(s);
    return 0;
}

/* The reference: DFN2's own TIME-mode analysis of the same audio. */
static int capture_via_time_mode(ScaleCapture *cap, FftHandle *fft) {
    static float out_hop[DFN2_HOP_LEN];
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    int hop;

    if (dfn2_prepost_config_defaults(&cfg, DFN2_IO_TIME) != 0) return 1;
    cfg.fft = fft;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    p = dfn2_prepost_create(&cfg);
    if (!p) return 2;
    for (hop = 0; hop < SCALE_FRAMES; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int valid = -1;
        int need = dfn2_prepost_pre_process(p, g_pcm + (size_t)hop * DFN2_HOP_LEN);
        if (need < 0) { dfn2_prepost_destroy(p); return 3; }
        if (need == 1) {
            if (dfn2_prepost_frame_inputs(p, &inputs, &outputs) != 0) {
                dfn2_prepost_destroy(p);
                return 4;
            }
            /* Each recurrent state output is the same buffer as its input. */
            if ((const void *)inputs.gru_hidden !=
                    (const void *)outputs.gru_hidden_next ||
                (const void *)inputs.df_convp_window !=
                    (const void *)outputs.df_convp_window_next) {
                dfn2_prepost_destroy(p);
                return 6;
            }
            if (capturing_infer(cap, &inputs, &outputs) != 0 ||
                dfn2_prepost_frame_commit(p) != 0) {
                dfn2_prepost_destroy(p);
                return 4;
            }
        }
        if (dfn2_prepost_post_process(p, out_hop, &valid) != 0) {
            dfn2_prepost_destroy(p);
            return 5;
        }
    }
    dfn2_prepost_destroy(p);
    return 0;
}

static int run_absolute_scale(void) {
    static ScaleCapture via_stage, via_time;
    FftHandle *fft = fft_create(DFN2_N_FFT);
    int rc, f;
    size_t i;

    if (!fft) return 1;
    make_pcm();
    memset(&via_stage, 0, sizeof(via_stage));
    memset(&via_time, 0, sizeof(via_time));
    rc = capture_via_stage(&via_stage, fft);
    if (rc) { fft_destroy(fft); return 10 + rc; }
    rc = capture_via_time_mode(&via_time, fft);
    fft_destroy(fft);
    if (rc) return 20 + rc;

    if (via_stage.overflow || via_time.overflow) return 2;
    /* Both clocks emit a frame on every hop but the first. */
    if (via_stage.frames != SCALE_FRAMES - 1 ||
        via_time.frames != via_stage.frames) return 3;
    if (via_stage.erb_elements != via_time.erb_elements ||
        via_stage.spec_elements != via_time.spec_elements) return 4;
    for (f = 0; f < via_stage.frames; ++f) {
        for (i = 0; i < via_stage.erb_elements; ++i) {
            if (via_stage.erb[f][i] != via_time.erb[f][i]) {
                fprintf(stderr, "  erb frame %d element %zu: stage %.9g, "
                        "DFN2 TIME mode %.9g\n", f, i,
                        (double)via_stage.erb[f][i], (double)via_time.erb[f][i]);
                return 5;
            }
        }
        for (i = 0; i < via_stage.spec_elements; ++i) {
            if (via_stage.spec[f][i] != via_time.spec[f][i]) {
                fprintf(stderr, "  spec frame %d element %zu: stage %.9g, "
                        "DFN2 TIME mode %.9g\n", f, i,
                        (double)via_stage.spec[f][i], (double)via_time.spec[f][i]);
                return 6;
            }
        }
    }
    printf("dfn_res_stage: absolute scale -- %d frames of ERB (%zu) and "
           "complex (%zu) features bit-identical to DFN2's own analysis\n",
           via_stage.frames, via_stage.erb_elements, via_stage.spec_elements);
    return 0;
}

/* 0 = NULL model, 1 = identity callback, 2 = failing callback. */
static int run_identity_case(int callback_kind) {
    DfnResStageConfig cfg = base_config();
    DfnResStageMemReq req;
    DFN2ModelIoDescriptor descriptor;
    DfnResStage *stage;
    Complex estimate[DFN2_N_BINS];
    Complex apply[DFN2_N_BINS];
    static Complex saved[FRAMES][DFN2_N_BINS];
    Complex out[DFN2_N_BINS];
    long long frames = -1, commits = -1, skips = -1, source = -1;
    int frame, k, rc;

    if (callback_kind != 0) {
        if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
        cfg.model.infer = callback_kind == 1 ? dfn_fixture_identity_infer : failing_infer;
        cfg.model.io_descriptor = &descriptor;
    }
    if (dfn_res_stage_get_mem_requirements(&cfg, &req) != 0) return 2;
    stage = dfn_res_stage_create(&cfg);
    if (!stage) return 3;

    for (frame = 0; frame < FRAMES; ++frame) {
        fill_frame(estimate, apply, frame);
        memcpy(saved[frame], apply, sizeof(apply));
        rc = dfn_res_stage_process(stage, estimate, apply, out);
        if (rc != (frame < 2 ? 0 : 1)) return 10 + frame;
        if (frame < 2) {
            for (k = 0; k < DFN2_N_BINS; ++k)
                if (out[k].r != 0.0f || out[k].i != 0.0f) return 20 + frame;
        } else {
            int src = frame - 2;
            for (k = 0; k < DFN2_N_BINS; ++k)
                if (out[k].r != saved[src][k].r || out[k].i != saved[src][k].i)
                    return 30 + frame;
            if (dfn_res_stage_output_frame_index(stage, &source) != 0 ||
                source != src) return 40 + frame;
        }
    }
    dfn_res_stage_get_counters(stage, &frames, &commits, &skips);
    if (frames != FRAMES) return 50;
    if (callback_kind == 1 && (commits != FRAMES - 1 || skips != 0)) return 51;
    if (callback_kind != 1 && (commits != 0 || skips != FRAMES - 1)) return 52;
    dfn_res_stage_reset(stage);
    dfn_res_stage_get_counters(stage, &frames, &commits, &skips);
    if (frames != 0 || commits != 0 || skips != 0) return 53;
    dfn_res_stage_destroy(stage);
    return 0;
}

/* N hops, reset, N hops == a fresh instance's N hops, byte for byte. */
static int run_reset_parity(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStage *a = dfn_res_stage_create(&cfg);
    DfnResStage *b = dfn_res_stage_create(&cfg);
    Complex estimate[DFN2_N_BINS], apply[DFN2_N_BINS];
    static Complex first[FRAMES][DFN2_N_BINS];
    Complex out[DFN2_N_BINS];
    int frame, pass;
    if (!a || !b) return 1;
    for (pass = 0; pass < 2; ++pass) {
        for (frame = 0; frame < FRAMES; ++frame) {
            fill_frame(estimate, apply, frame + 3);
            if (dfn_res_stage_process(a, estimate, apply, out) < 0) return 2;
            if (pass == 0) memcpy(first[frame], out, sizeof(out));
            else if (memcmp(first[frame], out, sizeof(out)) != 0) return 3;
        }
        dfn_res_stage_reset(a);
    }
    for (frame = 0; frame < FRAMES; ++frame) {
        fill_frame(estimate, apply, frame + 3);
        if (dfn_res_stage_process(b, estimate, apply, out) < 0) return 4;
        if (memcmp(first[frame], out, sizeof(out)) != 0) return 5;
    }
    dfn_res_stage_destroy(a);
    dfn_res_stage_destroy(b);
    return 0;
}

static int rejects(const DfnResStageConfig *cfg) {
    DfnResStageMemReq req;
    return dfn_res_stage_get_mem_requirements(cfg, &req) != 0 &&
           dfn_res_stage_create(cfg) == NULL;
}

static int run_rejections(void) {
    DfnResStageConfig cfg;
    DFN2ModelIoDescriptor descriptor;
    float saved;
    int k;

    cfg = base_config(); cfg.sample_rate = 16000;
    if (!rejects(&cfg)) return 1;
    cfg = base_config(); cfg.fft_size = 512;
    if (!rejects(&cfg)) return 2;
    cfg = base_config(); cfg.erb_inv = NULL;
    if (!rejects(&cfg)) return 3;
    cfg = base_config(); cfg.erb_fwd = NULL;
    if (!rejects(&cfg)) return 4;
    cfg = base_config(); cfg.atten_lim_db = NAN;
    if (!rejects(&cfg)) return 5;
    cfg = base_config(); cfg.model.infer = dfn_fixture_identity_infer;   /* no descriptor */
    if (!rejects(&cfg)) return 6;
    cfg = base_config();
    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 7;
    descriptor.layout_version += 1u;
    cfg.model.infer = dfn_fixture_identity_infer; cfg.model.io_descriptor = &descriptor;
    if (!rejects(&cfg)) return 8;

    /* One bin of the partition off by a relative 1e-4: refused. */
    cfg = base_config();
    saved = erb_inv[5];
    erb_inv[5] = 1.0001f;
    if (!rejects(&cfg)) { erb_inv[5] = saved; return 9; }
    erb_inv[5] = saved;
    /* Two bands sharing a bin as 0.5 + 0.5 (an exact float sum): accepted,
     * so the gate is the unit-sum property, not "band 0 owns everything". */
    erb_inv[7] = 0.5f;
    erb_inv[(size_t)1 * DFN2_N_BINS + 7] = 0.5f;
    if (rejects(&cfg)) { make_partition(); return 10; }
    make_partition();
    for (k = 0; k < DFN2_N_BINS; ++k)
        if (erb_inv[k] != 1.0f) return 11;
    return 0;
}

/* The 8-point stale-pool gate and the pool-size floor. */
static int run_stale_pool(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStageMemReq req, bad;
    void *mem = NULL;
    DfnResStage *s;
    if (dfn_res_stage_get_mem_requirements(&cfg, &req) != 0) return 1;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        return 2;
    s = dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!s) { free(mem); return 3; }
    dfn_res_stage_destroy(s);
    bad = req; bad.build_flags_hash ^= 1u;
    if (dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 4; }
    bad = req; bad.layout_version += 1u;
    if (dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 5; }
    bad = req; bad.backend_id = 0u;
    if (dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 6; }
    bad = req; bad.reserved = 1u;
    if (dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) { free(mem); return 7; }
    if (dfn_res_stage_init(mem, (size_t)req.bytes - 16u, &cfg)) { free(mem); return 8; }
    if (dfn_res_stage_init((unsigned char *)mem + 4, (size_t)req.bytes - 4u, &cfg)) { free(mem); return 9; }
    s = dfn_res_stage_init(mem, (size_t)req.bytes, &cfg);
    if (!s) { free(mem); return 10; }
    dfn_res_stage_destroy(s);
    free(mem);
    return 0;
}

/* A non-finite input is refused before it can reach the recurrent state:
 * out_spec untouched, the frame clock not advanced, the next hop normal. */
static int run_non_finite_input(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStage *s = dfn_res_stage_create(&cfg);
    Complex estimate[DFN2_N_BINS], apply[DFN2_N_BINS], out[DFN2_N_BINS];
    long long frames = -1;
    int k;
    if (!s) return 1;
    fill_frame(estimate, apply, 0);
    for (k = 0; k < DFN2_N_BINS; ++k) { out[k].r = 7.0f; out[k].i = -7.0f; }
    apply[100].r = NAN;
    if (dfn_res_stage_process(s, estimate, apply, out) != -1) return 2;
    for (k = 0; k < DFN2_N_BINS; ++k)
        if (out[k].r != 7.0f || out[k].i != -7.0f) return 3;
    dfn_res_stage_get_counters(s, &frames, NULL, NULL);
    if (frames != 0) return 4;
    apply[100].r = 0.0f;
    estimate[3].i = INFINITY;
    if (dfn_res_stage_process(s, estimate, apply, out) != -1) return 5;
    estimate[3].i = 0.0f;
    if (dfn_res_stage_process(s, estimate, apply, out) != 0) return 6;
    if (dfn_res_stage_process(s, NULL, apply, out) != -1) return 7;
    if (dfn_res_stage_process(s, estimate, apply, NULL) != -1) return 8;
    dfn_res_stage_destroy(s);
    return 0;
}

/* The phased hop (begin / model / end) against the one-call process(): a
 * non-identity model (a uniform 0.5 band mask, so the output differs from the
 * input) run by the caller between begin and end reproduces the configured
 * callback byte for byte. The fail-open rows: a nonzero model result, a frame
 * left open (end without a model result is fed -1 by the caller), and a
 * non-finite head all take the identity; the counters follow. */
static int half_mask_infer(void *user, const DFN2PrepostInputs *in,
                           DFN2PrepostOutputs *out) {
    size_t i;
    (void)user; (void)in;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.5f;
    memset(out->coefs, 0, out->coefs_elements * sizeof(float));
    out->alpha[0] = 0.0f;
    return 0;
}

static int nan_head_infer(void *user, const DFN2PrepostInputs *in,
                          DFN2PrepostOutputs *out) {
    half_mask_infer(user, in, out);
    out->erb_mask[3] = (float)NAN;
    return 0;
}

/* Writes every output validly, then reports failure: the result code alone
 * must send the frame down the identity path. */
static int half_mask_then_fail(void *user, const DFN2PrepostInputs *in,
                               DFN2PrepostOutputs *out) {
    half_mask_infer(user, in, out);
    return -1;
}

/* mode 0: caller runs half_mask_infer; 1: caller reports -1; 2: the model
 * writes a NaN head and reports success; 3: no model at all (config) and the
 * caller reports 0 on a frame it never ran (the commit refuses it); 4: the model
 * writes finite outputs and reports -1. */
static int run_phased_case(int mode) {
    DFN2ModelIoDescriptor descriptor;
    DfnResStageConfig phased_cfg = base_config();
    DfnResStageConfig cb_cfg = base_config();
    DfnResStage *ph, *cb;
    static Complex est[DFN2_N_BINS], app[DFN2_N_BINS], a[DFN2_N_BINS], b[DFN2_N_BINS];
    DfnResStageJob job;
    long long fa, ca, sa, fb, cb_commits, sb;
    int frame, differs = 0, any = 0;

    if (dfn2_model_io_descriptor_default(&descriptor) != 0) return 1;
    cb_cfg.model.io_descriptor = &descriptor;
    cb_cfg.model.infer = mode == 0 ? half_mask_infer
                       : mode == 1 ? failing_infer
                       : mode == 2 ? nan_head_infer
                       : mode == 4 ? half_mask_then_fail : NULL;
    if (mode == 3) cb_cfg.model.io_descriptor = NULL;
    ph = dfn_res_stage_create(&phased_cfg);
    cb = dfn_res_stage_create(&cb_cfg);
    if (!ph || !cb) return 2;
    for (frame = 0; frame < FRAMES + 4; ++frame) {
        int rc_b, rc_a;
        fill_frame(est, app, frame);
        rc_b = dfn_res_stage_process(cb, est, app, b);
        rc_a = dfn_res_stage_begin(ph, est, app, &job);
        if (rc_a < 0 || rc_b < 0) return 3;
        if (rc_a == 1) {
            int model_rc = 0;
            if (mode == 0) model_rc = half_mask_infer(NULL, &job.inputs, &job.outputs);
            else if (mode == 1) model_rc = -1;
            else if (mode == 2) model_rc = nan_head_infer(NULL, &job.inputs, &job.outputs);
            else if (mode == 4) model_rc = half_mask_then_fail(NULL, &job.inputs, &job.outputs);
            else model_rc = 0;   /* nothing was written: the NaN heads refuse the commit */
            if (dfn_res_stage_end(ph, model_rc, a) != rc_b) return 4;
        } else if (dfn_res_stage_end(ph, 0, a) != rc_b) {
            return 5;
        }
        if (memcmp(a, b, sizeof(a)) != 0) return 10 + frame;
        if (frame >= 2) {
            int k;
            for (k = 0; k < DFN2_N_BINS; ++k) {
                if (a[k].r != 0.0f) any = 1;
                if (a[k].r != app[k].r && frame == FRAMES + 3) differs = 1;
            }
        }
    }
    dfn_res_stage_get_counters(ph, &fa, &ca, &sa);
    dfn_res_stage_get_counters(cb, &fb, &cb_commits, &sb);
    if (fa != fb || ca != cb_commits || sa != sb) return 6;
    if (!any) return 7;
    if (mode == 0 && (ca == 0 || !differs)) return 8;   /* the model really shaped it */
    if (mode != 0 && ca != 0) return 9;                 /* every other row is the identity */
    dfn_res_stage_destroy(ph);
    dfn_res_stage_destroy(cb);
    return 0;
}

static int run_phased_sequence(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStage *s = dfn_res_stage_create(&cfg);
    static Complex est[DFN2_N_BINS], app[DFN2_N_BINS], out[DFN2_N_BINS];
    DfnResStageJob job;
    int frame;
    if (!s) return 1;
    fill_frame(est, app, 0);
    if (dfn_res_stage_end(s, 0, out) != -1) return 2;           /* end before begin */
    if (dfn_res_stage_begin(NULL, est, app, &job) != -1 ||
        dfn_res_stage_begin(s, est, app, NULL) != -1)
        return 3;
    if (dfn_res_stage_begin(s, est, app, &job) != 0) return 4;  /* warm-up: no model frame */
    if (dfn_res_stage_begin(s, est, app, &job) != -1) return 5; /* a hop is already begun */
    if (dfn_res_stage_end(s, 0, NULL) != -1) return 6;
    if (dfn_res_stage_end(s, 0, out) != 0) return 7;
    if (dfn_res_stage_end(s, 0, out) != -1) return 8;          /* already closed */
    for (frame = 1; frame < 4; ++frame) {
        int r;
        fill_frame(est, app, frame);
        r = dfn_res_stage_begin(s, est, app, &job);
        if (r < 0) return 9;
        if (r == 1) {
            if (job.inputs.erb_window == NULL || job.outputs.erb_mask == NULL) return 10;
            half_mask_infer(NULL, &job.inputs, &job.outputs);
        }
        if (dfn_res_stage_end(s, 0, out) < 0) return 11;
    }
    dfn_res_stage_reset(s);
    fill_frame(est, app, 0);
    if (dfn_res_stage_begin(s, est, app, &job) != 0) return 12; /* reset restarts the warm-up */
    dfn_res_stage_destroy(s);
    return 0;
}

int main(void) {
    int rc;
    make_partition();
    rc = run_identity_case(0);
    if (rc) { fprintf(stderr, "NULL-model identity failed: %d\n", rc); return 1; }
    rc = run_identity_case(1);
    if (rc) { fprintf(stderr, "callback identity failed: %d\n", rc); return 1; }
    rc = run_identity_case(2);
    if (rc) { fprintf(stderr, "fail-open identity failed: %d\n", rc); return 1; }
    rc = run_absolute_scale();
    if (rc) { fprintf(stderr, "absolute-scale row failed: %d\n", rc); return 1; }
    rc = run_reset_parity();
    if (rc) { fprintf(stderr, "reset parity failed: %d\n", rc); return 1; }
    rc = run_rejections();
    if (rc) { fprintf(stderr, "rejection row failed: %d\n", rc); return 1; }
    rc = run_stale_pool();
    if (rc) { fprintf(stderr, "stale-pool gate failed: %d\n", rc); return 1; }
    rc = run_non_finite_input();
    if (rc) { fprintf(stderr, "non-finite input row failed: %d\n", rc); return 1; }
    for (rc = 0; rc < 5; ++rc) {
        int r = run_phased_case(rc);
        if (r) { fprintf(stderr, "phased case %d failed: %d\n", rc, r); return 1; }
    }
    rc = run_phased_sequence();
    if (rc) { fprintf(stderr, "phased sequence row failed: %d\n", rc); return 1; }
    puts("dfn_res_stage: PASS");
    return 0;
}
