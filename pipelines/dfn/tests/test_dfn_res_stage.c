/* dfn_res_stage acceptance test (no AEC, no synthesis): the L-hop identity
 * (L = mask + df lookahead of the test grid's pair, dfn_test_grid.h) through the cascade with
 * distinct estimate/apply inputs, the absolute scale at the model boundary, the fail-open path,
 * reset parity, the state generation, the reject-first configuration rows (an invalid grid, one
 * frame per call only, a model of another grid or pair) and the stale-pool gate. The two inputs
 * differ in every frame, so an estimate/apply swap inside the stage would fail the identity rows.
 * The stage takes its grid at run time: the last rows build the three grids of the board
 * samples' table and all 15 lookahead pairs with dfn2_grid_make() and run each one through the
 * same stage code, whatever grid this program was built for. A refused call is proven to change
 * nothing by comparing the caller's pool bytes before and after. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn/dfn_res_stage.h"
#include "audio_stft.h"
#include "dfn/tests/dfn_test_fixture.h"

#define FRAMES 6
#define L_FRAMES DFN_TEST_LOOKAHEAD_FRAMES        /* output delay of the test grid's pair */
#define M_FRAMES DFN2_MASK_LOOKAHEAD              /* frames before the first model frame */
#define SCALE_FRAMES 8
#define WINDOW_FRAMES (DFN2_MODEL_INPUT_FRAMES + DFN_TEST_FRAMES_PER_CALL - 1)
#define CAP_ERB  (WINDOW_FRAMES * DFN2_N_ERB)
#define CAP_SPEC (2 * WINDOW_FRAMES * DFN2_DF_BINS)
/* Sizes of the recurrent-state buffers the reference instance registers: any non-empty buffers,
 * the library only zeroes and finite-checks them. */
#define REG_GRU_ELEMS 48
#define REG_CONVP_ELEMS 32

static float erb_fwd[DFN2_N_BINS * DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB * DFN2_N_BINS];

/* The identity paths scale a spectrum by the grid's stft_scale on the way in and stft_unscale on the
 * way out. Where n_fft is a power of four (256, 1024, 4096) both are powers of two and the round trip
 * is exact; elsewhere (512) they are correctly rounded floats and the round trip is within two float
 * ulps. */
static int scaled_equal(float got, float want, int n_fft) {
    int exact = n_fft == 256 || n_fft == 1024 || n_fft == 4096;
    return exact ? got == want : fabsf(got - want) <= 2.4e-7f * fabsf(want);
}

static void make_partition(void) {
    dfn_unit_partition(erb_fwd, erb_inv, dfn_test_grid());
}

static int failing_infer(void *user, const DFN2PrepostInputs *in,
                         DFN2PrepostOutputs *out) {
    (void)user; (void)in; (void)out;
    return -1;
}

/* Heads that are finite except one ERB mask element, reported as a success: the commit refuses. */
static int nan_head_infer(void *user, const DFN2PrepostInputs *in,
                          DFN2PrepostOutputs *out) {
    dfn_fixture_half_mask_infer(user, in, out);
    out->erb_mask[3] = (float)NAN;
    return 0;
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
 * The stage hands DFN2 the host's audio-scale spectrum times the grid's stft_scale (2^-5 on the
 * 48 kHz grid). The fail-open and identity paths reproduce their input at any scale
 * (stft_scale * stft_unscale == 1), so only this row sees a wrong factor: it drives one signal down
 * the product path (hand-assembled frames, unscaled FFT, through the stage) and through a
 * test-owned AcStft with spec_scale = stft_scale into dfn2_prepost_pre_process_freq, and compares
 * the ERB and complex feature windows the model receives bit for bit. That pins the scale factor,
 * the analysis window and the frame assembly ([previous hop | this hop], 50% overlap, no
 * centring). The reference instance registers state buffers, so its input/output aliasing check
 * compares real addresses. */
typedef struct ScaleCapture {
    float erb[SCALE_FRAMES][CAP_ERB];
    float spec[SCALE_FRAMES][CAP_SPEC];
    size_t want_erb;       /* element counts the grid publishes */
    size_t want_spec;
    int frames;
    int bad;               /* a window of another size, or a frame beyond SCALE_FRAMES */
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

static int capturing_infer(void *user, const DFN2PrepostInputs *in,
                           DFN2PrepostOutputs *out) {
    ScaleCapture *cap = (ScaleCapture *)user;
    if (in->erb_window_elements != cap->want_erb || in->spec_window_elements != cap->want_spec ||
        cap->want_erb > CAP_ERB || cap->want_spec > CAP_SPEC || cap->frames >= SCALE_FRAMES) {
        cap->bad = 1;
    } else {
        memcpy(cap->erb[cap->frames], in->erb_window, in->erb_window_elements * sizeof(float));
        memcpy(cap->spec[cap->frames], in->spec_window, in->spec_window_elements * sizeof(float));
        ++cap->frames;
    }
    return dfn_fixture_identity_infer(NULL, in, out);
}

static DfnResStageConfig base_config(void) {
    DfnResStageConfig cfg = dfn_res_stage_default_config(dfn_test_grid());
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

    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
    cfg.model.infer = capturing_infer;
    cfg.model.user = cap;
    cfg.model.io_descriptor = &descriptor;
    s = dfn_res_stage_create(&cfg);
    if (!s) return 2;
    dfn_fixture_root_hann(win, DFN2_N_FFT);
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

/* The reference: the library fed the spectra of an AcStft whose spec_scale is the grid's
 * stft_scale, the framing contract of dfn2_process.h. */
static int capture_via_reference(ScaleCapture *cap) {
    static float re[DFN2_N_BINS], im[DFN2_N_BINS], out_re[DFN2_N_BINS], out_im[DFN2_N_BINS];
    static float reg_gru[REG_GRU_ELEMS], reg_convp[REG_CONVP_ELEMS];
    const DFN2Grid *grid = dfn_test_grid();
    DFN2PrepostConfig cfg;
    DFN2Prepost *p = NULL;
    AcStftConfig scfg;
    AcStft *stft = NULL;
    void *stft_mem = NULL;
    size_t stft_bytes, i;
    int hop, rc = 0;

    if (dfn2_prepost_config_defaults(&cfg, grid) != 0) return 1;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.gru_hidden = reg_gru;
    cfg.gru_hidden_elements = REG_GRU_ELEMS;
    cfg.df_convp_history = reg_convp;
    cfg.df_convp_history_elements = REG_CONVP_ELEMS;
    for (i = 0; i < REG_GRU_ELEMS; ++i) reg_gru[i] = 7.0f;
    for (i = 0; i < REG_CONVP_ELEMS; ++i) reg_convp[i] = 7.0f;
    p = dfn2_prepost_create(&cfg);
    if (!p) return 2;
    /* The instance zeroes what is registered; unregistered state would leave the 7s. */
    for (i = 0; i < REG_GRU_ELEMS; ++i) if (reg_gru[i] != 0.0f) rc = 3;
    for (i = 0; i < REG_CONVP_ELEMS; ++i) if (reg_convp[i] != 0.0f) rc = 3;
    ac_stft_config_default(&scfg, grid->n_fft);
    scfg.spec_scale = grid->stft_scale;
    stft_bytes = ac_stft_get_mem_size(&scfg);
    if (rc == 0 && (stft_bytes == 0 || posix_memalign(&stft_mem, 16u, stft_bytes) != 0)) rc = 4;
    if (rc == 0) {
        stft = ac_stft_init(stft_mem, stft_bytes, &scfg);
        if (!stft) rc = 4;
    }
    for (hop = 0; rc == 0 && hop < SCALE_FRAMES; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int valid = -1, need;
        if (ac_stft_push_planar(stft, g_pcm + (size_t)hop * DFN2_HOP_LEN, re, im) != 1) { rc = 5; break; }
        need = dfn2_prepost_pre_process_freq(p, re, im);
        if (need < 0) { rc = 5; break; }
        if (need == 1) {
            if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0) { rc = 6; break; }
            /* Each recurrent state output is the registered buffer its input is. */
            if (inputs.gru_hidden != reg_gru || outputs.gru_hidden_next != reg_gru ||
                inputs.df_convp_window != reg_convp || outputs.df_convp_window_next != reg_convp ||
                inputs.gru_hidden_elements != REG_GRU_ELEMS ||
                outputs.gru_hidden_elements != REG_GRU_ELEMS ||
                inputs.df_convp_window_elements != REG_CONVP_ELEMS ||
                outputs.df_convp_window_elements != REG_CONVP_ELEMS) { rc = 7; break; }
            if (capturing_infer(cap, &inputs, &outputs) != 0 ||
                dfn2_prepost_frame_finish(p) != 0) { rc = 6; break; }
        }
        if (dfn2_prepost_post_process_freq(p, out_re, out_im, &valid) != 0) rc = 8;
    }
    dfn2_prepost_destroy(p);
    free(stft_mem);
    return rc;
}

static int run_absolute_scale(void) {
    static ScaleCapture via_stage, via_ref;
    const DFN2Grid *grid = dfn_test_grid();
    FftHandle *fft = fft_create(DFN2_N_FFT);
    int rc, f;
    size_t i;

    if (!fft) return 1;
    make_pcm();
    memset(&via_stage, 0, sizeof(via_stage));
    memset(&via_ref, 0, sizeof(via_ref));
    via_stage.want_erb = via_ref.want_erb = dfn2_prepost_erb_window_elements(grid);
    via_stage.want_spec = via_ref.want_spec = dfn2_prepost_spec_window_elements(grid);
    if (via_stage.want_erb != CAP_ERB || via_stage.want_spec != CAP_SPEC) { fft_destroy(fft); return 9; }
    rc = capture_via_stage(&via_stage, fft);
    if (rc) { fft_destroy(fft); return 10 + rc; }
    rc = capture_via_reference(&via_ref);
    fft_destroy(fft);
    if (rc) return 20 + rc;

    if (via_stage.bad || via_ref.bad) return 2;
    /* Both clocks run the model on every hop, the first mask_lookahead hops being warm-up frames. */
    if (via_stage.frames != SCALE_FRAMES || via_ref.frames != via_stage.frames) return 3;
    /* Hop 0 shows the zero-initialised window with the first hop's features in its newest slot:
     * every earlier time slot of both windows is zero, the newest is not. */
    {
        const int past = DFN2_MODEL_INPUT_FRAMES - 1;
        int newest_erb = 0, newest_spec = 0, ch, t;
        for (i = 0; i < via_stage.want_erb; ++i) {
            const int slot = (int)(i / DFN2_N_ERB);
            if (slot < past && via_stage.erb[0][i] != 0.0f) return 7;
            if (slot == past && via_stage.erb[0][i] != 0.0f) newest_erb = 1;
        }
        for (ch = 0; ch < 2; ++ch)
            for (t = 0; t < DFN2_MODEL_INPUT_FRAMES; ++t)
                for (i = 0; i < DFN2_DF_BINS; ++i) {
                    const float v = via_stage.spec[0][((size_t)ch * DFN2_MODEL_INPUT_FRAMES + (size_t)t) * DFN2_DF_BINS + i];
                    if (t < past && v != 0.0f) return 8;
                    if (t == past && v != 0.0f) newest_spec = 1;
                }
        if (!newest_erb || !newest_spec) return 9;
    }
    for (f = 0; f < via_stage.frames; ++f) {
        for (i = 0; i < via_stage.want_erb; ++i) {
            if (via_stage.erb[f][i] != via_ref.erb[f][i]) {
                fprintf(stderr, "  erb frame %d element %zu: stage %.9g, "
                        "reference %.9g\n", f, i,
                        (double)via_stage.erb[f][i], (double)via_ref.erb[f][i]);
                return 5;
            }
        }
        for (i = 0; i < via_stage.want_spec; ++i) {
            if (via_stage.spec[f][i] != via_ref.spec[f][i]) {
                fprintf(stderr, "  spec frame %d element %zu: stage %.9g, "
                        "reference %.9g\n", f, i,
                        (double)via_stage.spec[f][i], (double)via_ref.spec[f][i]);
                return 6;
            }
        }
    }
    printf("dfn_res_stage: absolute scale -- %d frames (warm-up frames included) of ERB (%zu) and "
           "complex (%zu) features bit-identical to the library fed an AcStft with "
           "spec_scale = stft_scale\n", via_stage.frames, via_stage.want_erb, via_stage.want_spec);
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
        if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
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
        if (rc != (frame < L_FRAMES ? 0 : 1)) return 10 + frame;
        if (frame < L_FRAMES) {
            for (k = 0; k < DFN2_N_BINS; ++k)
                if (out[k].r != 0.0f || out[k].i != 0.0f) return 20 + frame;
        } else {
            int src = frame - L_FRAMES;
            for (k = 0; k < DFN2_N_BINS; ++k)
                if (!scaled_equal(out[k].r, saved[src][k].r, DFN2_N_FFT) ||
                    !scaled_equal(out[k].i, saved[src][k].i, DFN2_N_FFT))
                    return 30 + frame;
            if (dfn_res_stage_output_frame_index(stage, &source) != 0 ||
                source != src) return 40 + frame;
        }
    }
    dfn_res_stage_get_counters(stage, &frames, &commits, &skips);
    if (frames != FRAMES) return 50;
    if (callback_kind == 1 && (commits != FRAMES - M_FRAMES || skips != 0)) return 51;
    if (callback_kind != 1 && (commits != 0 || skips != FRAMES - M_FRAMES)) return 52;
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

    /* Control: the unedited config is accepted, so each row below refuses on its one edit. */
    cfg = base_config();
    if (rejects(&cfg)) return 99;
    /* The grid is part of the config and the library validates it: a zeroed grid (what a NULL
     * grid leaves), a hand-edited derived field, a band count the model cannot have, a
     * non-finite alpha. */
    cfg = base_config(); memset(&cfg.grid, 0, sizeof(cfg.grid));
    if (!rejects(&cfg)) return 1;
    cfg = base_config(); cfg.grid.hop += 1;
    if (!rejects(&cfg)) return 2;
    cfg = dfn_res_stage_default_config(NULL); cfg.erb_fwd = erb_fwd; cfg.erb_inv = erb_inv;
    if (!rejects(&cfg)) return 12;
    cfg = base_config(); cfg.grid.n_erb += 1;
    if (!rejects(&cfg)) return 13;
    cfg = base_config(); cfg.grid.n_fft *= 2;
    if (!rejects(&cfg)) return 14;
    cfg = base_config(); cfg.grid.erb_norm_alpha = (float)NAN;
    if (!rejects(&cfg)) return 15;
    /* A bin or band count far beyond any matrix must be refused before the stage reads one. */
    cfg = base_config(); cfg.grid.n_bins = 1 << 20;
    if (!rejects(&cfg)) return 16;
    cfg = base_config(); cfg.grid.n_erb = 1 << 20;
    if (!rejects(&cfg)) return 17;
    /* A tau edited with its alpha kept, a zero normaliser scale or start value, a zero stft scale. */
    cfg = base_config(); cfg.grid.erb_norm_tau_sec = 0.5f;
    if (!rejects(&cfg)) return 18;
    cfg = base_config(); cfg.grid.spec_norm_tau_sec = 0.5f;
    if (!rejects(&cfg)) return 19;
    cfg = base_config(); cfg.grid.erb_norm_scale_db = 0.0f;
    if (!rejects(&cfg)) return 50;
    cfg = base_config(); cfg.grid.spec_norm_init_lo = 0.0f;
    if (!rejects(&cfg)) return 51;
    cfg = base_config(); cfg.grid.spec_norm_init_hi = -1.0f;
    if (!rejects(&cfg)) return 52;
    cfg = base_config(); cfg.grid.stft_scale = 0.0f;
    if (!rejects(&cfg)) return 53;
    cfg = base_config(); cfg.grid.stft_unscale = 0.0f;
    if (!rejects(&cfg)) return 54;
    /* Two frames per call: a grid the library makes and runs, refused by the stage alone (its
     * lookahead and the emitted frame index are exact for one frame per call). */
    {
        DFN2GridSpec spec = dfn_test_spec_of(dfn_test_grid());
        DFN2Grid two;
        DFN2PrepostConfig pcfg;
        DFN2PrepostMemReq preq;
        spec.frames_per_call = 2;
        if (dfn2_grid_make(&two, &spec) != DFN2_GRID_OK) return 55;
        if (dfn2_prepost_config_defaults(&pcfg, &two) != 0) return 56;
        pcfg.erb_fwd = erb_fwd;
        pcfg.erb_inv = erb_inv;
        if (dfn2_prepost_get_mem_size(&pcfg, &preq) != 0) return 57;
        cfg = base_config(); cfg.grid = two;
        if (!rejects(&cfg)) return 58;
        cfg = base_config();
        if (rejects(&cfg)) return 59;
    }
    cfg = base_config(); cfg.erb_inv = NULL;
    if (!rejects(&cfg)) return 3;
    cfg = base_config(); cfg.erb_fwd = NULL;
    if (!rejects(&cfg)) return 4;
    cfg = base_config(); cfg.atten_lim_db = NAN;
    if (!rejects(&cfg)) return 5;
    cfg = base_config(); cfg.model.infer = dfn_fixture_identity_infer;   /* no descriptor */
    if (!rejects(&cfg)) return 6;
    /* A model built for another lookahead pair, grid, frame count or deep-filter order is refused
     * from its descriptor at run time. The control row, the test grid's own descriptor, is
     * accepted, so the rows above refuse on the grid and nothing else. */
    {
        int which;
        for (which = 0; which < 5; ++which) {
            if (dfn_fixture_foreign_descriptor(dfn_test_grid(), &descriptor, which) != 0)
                return 20 + which;
            cfg = base_config();
            cfg.model.infer = dfn_fixture_identity_infer; cfg.model.io_descriptor = &descriptor;
            if (!rejects(&cfg)) return 30 + which;
        }
        cfg = base_config();
        if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 40;
        cfg.model.infer = dfn_fixture_identity_infer; cfg.model.io_descriptor = &descriptor;
        if (rejects(&cfg)) return 41;
    }
    cfg = base_config();
    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 7;
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

/* ---- the stage at run time: the three grids of the board samples' table and every lookahead
 * pair, whatever grid this program was built for ---------------------------------------------
 * Each row builds its grid with dfn2_grid_make() from a spec that carries all its columns, a unit
 * partition of its own size, the model descriptor of that grid and an identity model, and runs
 * hops of its own bin count through the same stage code. The output is the apply spectrum delayed
 * by exactly the grid's lookahead (mask + df), finite, with the warm-up hops all zero; the emitted
 * frame index and the measured first-output hop say the same; the grid the stage reports is the
 * one it was given. Across rows a model descriptor of one grid is refused by a stage of another,
 * and a pool recorded for one grid is refused by a stage of another on the hash alone. */
static const DFN2GridSpec TABLE[] = {
    { .sample_rate = 16000, .n_fft = 256, .n_erb = 24, .df_bins = 96,
      .df_order = DFN_TEST_DF_ORDER, .mask_lookahead = 0, .df_lookahead = 0,
      .frames_per_call = DFN_TEST_FRAMES_PER_CALL,
      .erb_norm_tau_sec = DFN_TEST_ERB_NORM_TAU_SEC, .erb_norm_init_lo_db = DFN_TEST_ERB_NORM_INIT_LO_DB,
      .erb_norm_init_hi_db = DFN_TEST_ERB_NORM_INIT_HI_DB, .erb_norm_scale_db = DFN_TEST_ERB_NORM_SCALE_DB,
      .spec_norm_tau_sec = DFN_TEST_SPEC_NORM_TAU_SEC, .spec_norm_init_lo = DFN_TEST_SPEC_NORM_INIT_LO,
      .spec_norm_init_hi = DFN_TEST_SPEC_NORM_INIT_HI },
    { .sample_rate = 32000, .n_fft = 512, .n_erb = 28, .df_bins = 96,   /* provisional: nothing is trained for it */
      .df_order = DFN_TEST_DF_ORDER, .mask_lookahead = 0, .df_lookahead = 0,
      .frames_per_call = DFN_TEST_FRAMES_PER_CALL,
      .erb_norm_tau_sec = DFN_TEST_ERB_NORM_TAU_SEC, .erb_norm_init_lo_db = DFN_TEST_ERB_NORM_INIT_LO_DB,
      .erb_norm_init_hi_db = DFN_TEST_ERB_NORM_INIT_HI_DB, .erb_norm_scale_db = DFN_TEST_ERB_NORM_SCALE_DB,
      .spec_norm_tau_sec = DFN_TEST_SPEC_NORM_TAU_SEC, .spec_norm_init_lo = DFN_TEST_SPEC_NORM_INIT_LO,
      .spec_norm_init_hi = DFN_TEST_SPEC_NORM_INIT_HI },
    { .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96,
      .df_order = DFN_TEST_DF_ORDER, .mask_lookahead = 1, .df_lookahead = 1,
      .frames_per_call = DFN_TEST_FRAMES_PER_CALL,
      .erb_norm_tau_sec = DFN_TEST_ERB_NORM_TAU_SEC, .erb_norm_init_lo_db = DFN_TEST_ERB_NORM_INIT_LO_DB,
      .erb_norm_init_hi_db = DFN_TEST_ERB_NORM_INIT_HI_DB, .erb_norm_scale_db = DFN_TEST_ERB_NORM_SCALE_DB,
      .spec_norm_tau_sec = DFN_TEST_SPEC_NORM_TAU_SEC, .spec_norm_init_lo = DFN_TEST_SPEC_NORM_INIT_LO,
      .spec_norm_init_hi = DFN_TEST_SPEC_NORM_INIT_HI },
};
#define TABLE_ROWS ((int)(sizeof(TABLE) / sizeof(TABLE[0])))
#define TG_HOPS 12

static float tg_fwd[DFN_TEST_TABLE_MAX_BINS * DFN_TEST_TABLE_MAX_ERB];
static float tg_inv[DFN_TEST_TABLE_MAX_ERB * DFN_TEST_TABLE_MAX_BINS];
static Complex tg_est[DFN_TEST_TABLE_MAX_BINS], tg_app[DFN_TEST_TABLE_MAX_BINS];
static Complex tg_out[DFN_TEST_TABLE_MAX_BINS];
static Complex tg_apply_saved[TG_HOPS][DFN_TEST_TABLE_MAX_BINS];

/* Frame-dependent, mutually distinct estimate and apply spectra of n_bins bins. */
static void tg_fill(int n_bins, int frame) {
    int k;
    for (k = 0; k < n_bins; ++k) {
        tg_est[k].r = (float)(frame * 1000 + k) * 0.125f;
        tg_est[k].i = (float)(frame * 1000 - k) * 0.0625f;
        tg_app[k].r = (float)(frame * 2000 + k) * 0.25f;
        tg_app[k].i = (float)(frame * 2000 - k) * 0.125f;
    }
}

/* Runs `grid` through a stage with the identity model. Returns 0 and the measured stage latency in
 * hops (first emitted hop, minus the source frame it carries) in *measured, or a nonzero code. */
static int run_stage_on_grid(const DFN2Grid *grid, int expect_lookahead, int *measured,
                             DfnResStageMemReq *req_out) {
    DFN2ModelIoDescriptor d;
    DfnResStageConfig cfg = dfn_res_stage_default_config(grid);
    DfnResStageMemReq req;
    DfnResStage *s;
    long long source = -1, frames = 0, commits = 0, skips = 0;
    int hop, k, first = -1, first_source = -1;
    const int n = grid->n_bins;

    if (!dfn_test_table_fits(grid)) return 6;
    dfn_unit_partition(tg_fwd, tg_inv, grid);
    if (dfn2_model_io_descriptor_default(grid, &d) != 0) return 1;
    cfg.erb_fwd = tg_fwd;
    cfg.erb_inv = tg_inv;
    cfg.model.infer = dfn_fixture_identity_infer;
    cfg.model.io_descriptor = &d;
    if (dfn_res_stage_get_mem_requirements(&cfg, &req) != 0) return 2;
    s = dfn_res_stage_create(&cfg);
    if (!s) return 3;
    if (!dfn_res_stage_grid(s) || memcmp(dfn_res_stage_grid(s), grid, sizeof(*grid)) != 0) return 4;
    if (dfn_res_stage_lookahead_frames(grid) != expect_lookahead) return 5;
    for (hop = 0; hop < TG_HOPS; ++hop) {
        int rc;
        tg_fill(n, hop);
        memcpy(tg_apply_saved[hop], tg_app, (size_t)n * sizeof(Complex));
        for (k = 0; k < n; ++k) { tg_out[k].r = 7.0f; tg_out[k].i = -7.0f; }
        rc = dfn_res_stage_process(s, tg_est, tg_app, tg_out);
        if (rc != (hop < expect_lookahead ? 0 : 1)) return 10 + hop;
        for (k = 0; k < n; ++k) {
            if (!isfinite(tg_out[k].r) || !isfinite(tg_out[k].i)) return 30;
            if (hop < expect_lookahead) {
                if (tg_out[k].r != 0.0f || tg_out[k].i != 0.0f) return 31;
            } else {
                const Complex want = tg_apply_saved[hop - expect_lookahead][k];
                if (!scaled_equal(tg_out[k].r, want.r, grid->n_fft) ||
                    !scaled_equal(tg_out[k].i, want.i, grid->n_fft))
                    return 32;
            }
        }
        if (rc == 1) {
            if (dfn_res_stage_output_frame_index(s, &source) != 0 ||
                source != hop - expect_lookahead)
                return 40;
            if (first < 0) { first = hop; first_source = (int)source; }
        }
    }
    dfn_res_stage_get_counters(s, &frames, &commits, &skips);
    if (frames != TG_HOPS || commits != TG_HOPS - grid->mask_lookahead || skips != 0) return 41;
    if (first < 0) return 42;
    /* Each warm-up frame advances the generation once; committed frames leave it alone. */
    if (dfn_res_stage_state_generation(s) != 1u + (unsigned int)grid->mask_lookahead) return 43;
    *measured = first - first_source;
    if (req_out) *req_out = req;
    dfn_res_stage_destroy(s);
    return 0;
}

static int run_table_grids(void) {
    DFN2Grid grid[TABLE_ROWS];
    DfnResStageMemReq req[TABLE_ROWS];
    int row, other, pair_m, pair_d, rc, measured, pairs_run = 0;

    for (row = 0; row < TABLE_ROWS; ++row) {
        const DFN2GridSpec *t = &TABLE[row];
        if (dfn2_grid_make(&grid[row], t) != DFN2_GRID_OK) return 1;
        rc = run_stage_on_grid(&grid[row], t->mask_lookahead + t->df_lookahead, &measured, &req[row]);
        if (rc) {
            fprintf(stderr, "  table grid %d/%d: stage row %d\n", t->sample_rate, t->n_fft, rc);
            return 100 + row;
        }
        if (measured != t->mask_lookahead + t->df_lookahead) return 110 + row;
        printf("dfn_res_stage: grid %d Hz / FFT %d (n_erb %d, df_bins %d, lookahead %d/%d): pool %llu "
               "bytes, hash %08x, output = apply delayed %d hops, all finite\n", t->sample_rate,
               t->n_fft, t->n_erb, t->df_bins, t->mask_lookahead, t->df_lookahead,
               (unsigned long long)req[row].bytes, req[row].build_flags_hash, measured);
    }
    /* A model of one grid on a stage of another, and a pool recorded for another grid. */
    for (row = 0; row < TABLE_ROWS; ++row) {
        for (other = 0; other < TABLE_ROWS; ++other) {
            DFN2ModelIoDescriptor foreign;
            DfnResStageConfig cfg = dfn_res_stage_default_config(&grid[row]);
            DfnResStageMemReq claimed = req[other];
            void *mem = NULL;
            if (row == other) continue;
            dfn_unit_partition(tg_fwd, tg_inv, &grid[row]);
            cfg.erb_fwd = tg_fwd;
            cfg.erb_inv = tg_inv;
            if (dfn2_model_io_descriptor_default(&grid[other], &foreign) != 0) return 120;
            cfg.model.infer = dfn_fixture_identity_infer;
            cfg.model.io_descriptor = &foreign;
            if (!rejects(&cfg)) return 121;
            cfg.model.infer = NULL;
            cfg.model.io_descriptor = NULL;
            if (req[row].build_flags_hash == req[other].build_flags_hash) return 122;
            claimed.bytes = req[row].bytes;   /* only the hash differs */
            if (posix_memalign(&mem, (size_t)req[row].alignment, (size_t)req[row].bytes) != 0)
                return 123;
            if (dfn_res_stage_init_ex(mem, (size_t)req[row].bytes, &cfg, &claimed)) {
                free(mem);
                return 124;
            }
            free(mem);
        }
    }
    /* The lookahead pair is a run-time value too: all 15 pairs (mask 0..2, df 0..df_order - 1) run
     * on the 48 kHz grid with their own latency in one binary. */
    for (pair_m = 0; pair_m <= DFN2_GRID_MASK_LOOKAHEAD_MAX; ++pair_m) {
        for (pair_d = 0; pair_d < DFN_TEST_DF_ORDER; ++pair_d) {
            DFN2Grid g;
            DFN2GridSpec spec = dfn_test_spec(48000, 1024, 32, 96, pair_m, pair_d);
            if (dfn2_grid_make(&g, &spec) != DFN2_GRID_OK) return 131;
            rc = run_stage_on_grid(&g, pair_m + pair_d, &measured, NULL);
            if (rc) { fprintf(stderr, "  pair %d/%d: stage row %d\n", pair_m, pair_d, rc); return 132; }
            if (measured != pair_m + pair_d) return 133;
            ++pairs_run;
        }
    }
    if (pairs_run != 15 || pairs_run != (DFN2_GRID_MASK_LOOKAHEAD_MAX + 1) * DFN_TEST_DF_ORDER) return 134;
    printf("dfn_res_stage: every table grid and all %d lookahead pairs run at run time; a model or "
           "pool of another grid is refused\n", pairs_run);
    return 0;
}

/* The stale-pool gate and the pool-size floor. A refused init leaves the caller's pool bytes as
 * they were. */
static int run_stale_pool(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStageMemReq req, bad;
    unsigned char *snapshot;
    void *mem = NULL;
    DfnResStage *s;
    int rc = 0, i;
    if (dfn_res_stage_get_mem_requirements(&cfg, &req) != 0) return 1;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0)
        return 2;
    snapshot = (unsigned char *)malloc((size_t)req.bytes);
    if (!snapshot) { free(mem); return 2; }
    s = dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &req);
    if (!s) { free(snapshot); free(mem); return 3; }
    dfn_res_stage_destroy(s);
    memset(mem, 0xA5, (size_t)req.bytes);   /* what the caller's pool holds when the init is refused */
    memcpy(snapshot, mem, (size_t)req.bytes);
    for (i = 0; i < 7 && !rc; ++i) {
        bad = req;
        switch (i) {
        case 0: bad.build_flags_hash ^= 1u; break;
        case 1: bad.layout_version += 1u; break;
        case 2: bad.backend_id = 0u; break;
        case 3: bad.reserved = 1u; break;
        case 4: bad.bytes -= 16u; break;
        case 5: bad.alignment = 0u; break;
        default: break;
        }
        if (i < 6) {
            if (dfn_res_stage_init_ex(mem, (size_t)req.bytes, &cfg, &bad)) rc = 4 + i;
        } else if (dfn_res_stage_init(mem, (size_t)req.bytes - 16u, &cfg)) {
            rc = 10;
        }
        if (!rc && memcmp(snapshot, mem, (size_t)req.bytes) != 0) rc = 20 + i;
    }
    if (!rc && dfn_res_stage_init((unsigned char *)mem + 4, (size_t)req.bytes - 4u, &cfg)) rc = 9;
    if (!rc && memcmp(snapshot, mem, (size_t)req.bytes) != 0) rc = 30;
    if (!rc) {
        s = dfn_res_stage_init(mem, (size_t)req.bytes, &cfg);
        if (!s) rc = 11;
        else dfn_res_stage_destroy(s);
    }
    free(snapshot);
    free(mem);
    return rc;
}

/* A refused call leaves the stage alone: non-finite or missing spectra, a missing output, a missing
 * job and an end without a begin are refused with out_spec untouched, the frame count and the state
 * generation unchanged and every byte of the caller's pool as it was; the next hop is normal. */
static int run_refused_calls(void) {
    DfnResStageConfig cfg = base_config();
    DfnResStageMemReq req;
    Complex estimate[DFN2_N_BINS], apply[DFN2_N_BINS], out[DFN2_N_BINS];
    DfnResStageJob job;
    unsigned char *snapshot;
    void *mem = NULL;
    DfnResStage *s;
    long long frames = -1, after = -1;
    unsigned int generation;
    int k, rc = 0;

    if (dfn_res_stage_get_mem_requirements(&cfg, &req) != 0) return 1;
    if (posix_memalign(&mem, (size_t)req.alignment, (size_t)req.bytes) != 0) return 2;
    snapshot = (unsigned char *)malloc((size_t)req.bytes);
    s = snapshot ? dfn_res_stage_init(mem, (size_t)req.bytes, &cfg) : NULL;
    if (!s) { free(snapshot); free(mem); return 3; }
    for (k = 0; k < 2; ++k) {   /* state worth protecting */
        fill_frame(estimate, apply, k);
        if (dfn_res_stage_process(s, estimate, apply, out) < 0) rc = 4;
    }
    fill_frame(estimate, apply, 2);
    for (k = 0; k < DFN2_N_BINS; ++k) { out[k].r = 7.0f; out[k].i = -7.0f; }
    memcpy(snapshot, mem, (size_t)req.bytes);
    dfn_res_stage_get_counters(s, &frames, NULL, NULL);
    generation = dfn_res_stage_state_generation(s);

    if (!rc) {
        apply[100].r = NAN;
        if (dfn_res_stage_process(s, estimate, apply, out) != -1) rc = 5;
        apply[100].r = 0.0f;
        estimate[3].i = INFINITY;
        if (dfn_res_stage_process(s, estimate, apply, out) != -1) rc = 6;
        estimate[3].i = 0.0f;
        if (dfn_res_stage_process(s, NULL, apply, out) != -1) rc = 7;
        if (dfn_res_stage_process(s, estimate, NULL, out) != -1) rc = 8;
        if (dfn_res_stage_process(s, estimate, apply, NULL) != -1) rc = 9;
        if (dfn_res_stage_begin(s, estimate, apply, NULL) != -1) rc = 10;
        if (dfn_res_stage_end(s, 0, out) != -1) rc = 11;           /* end before begin */
    }
    for (k = 0; k < DFN2_N_BINS; ++k)
        if (out[k].r != 7.0f || out[k].i != -7.0f) rc = rc ? rc : 12;
    dfn_res_stage_get_counters(s, &after, NULL, NULL);
    if (!rc && after != frames) rc = 13;
    if (!rc && dfn_res_stage_state_generation(s) != generation) rc = 14;
    if (!rc && memcmp(snapshot, mem, (size_t)req.bytes) != 0) rc = 15;
    if (!rc) {   /* the next hop is normal */
        int r = dfn_res_stage_begin(s, estimate, apply, &job);
        if (r < 0) rc = 16;
        else if (dfn_res_stage_end(s, r == 1 ? dfn_res_stage_infer(s, &job) : 0, out) < 0) rc = 17;
        dfn_res_stage_get_counters(s, &after, NULL, NULL);
        if (!rc && after != frames + 1) rc = 18;
    }
    dfn_res_stage_destroy(s);
    free(snapshot);
    free(mem);
    return rc;
}

/* Generation of the model's recurrent state: 1 after init, plus one per reset, one per warm-up frame
 * (whatever the model wrote) and one per refused commit of a real frame (a head the commit finds
 * non-finite); a model that reports failure and a committed real frame leave it alone. */
static int run_state_generation(void) {
    DFN2ModelIoDescriptor descriptor;
    DfnResStageConfig cfg = base_config();
    DfnResStage *s;
    Complex estimate[DFN2_N_BINS], apply[DFN2_N_BINS], out[DFN2_N_BINS];
    long long commits = 0, skips = 0;
    unsigned int expect;
    int kind, frame;

    if (dfn_res_stage_state_generation(NULL) != 0u) return 1;
    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 2;
    cfg.model.io_descriptor = &descriptor;
    for (kind = 0; kind < 3; ++kind) {   /* committed frames, a failing model, a refused commit */
        cfg.model.infer = kind == 0 ? dfn_fixture_identity_infer
                        : kind == 1 ? failing_infer : nan_head_infer;
        s = dfn_res_stage_create(&cfg);
        if (!s) return 3;
        expect = 1u;
        if (dfn_res_stage_state_generation(s) != expect) return 4;
        for (frame = 0; frame < FRAMES; ++frame) {
            fill_frame(estimate, apply, frame);
            if (dfn_res_stage_process(s, estimate, apply, out) < 0) return 5;
            /* a warm-up frame discards its heads, NaN or not: exactly one; a real frame: only a refusal */
            if (frame < M_FRAMES || kind == 2) ++expect;
            if (dfn_res_stage_state_generation(s) != expect) return 6 + kind;
        }
        dfn_res_stage_get_counters(s, NULL, &commits, &skips);
        if (kind == 0 && (commits != FRAMES - M_FRAMES || skips != 0)) return 10;
        if (kind != 0 && (commits != 0 || skips != FRAMES - M_FRAMES)) return 11;
        dfn_res_stage_reset(s);
        if (dfn_res_stage_state_generation(s) != expect + 1u) return 12;
        dfn_res_stage_reset(s);
        if (dfn_res_stage_state_generation(s) != expect + 2u) return 13;
        dfn_res_stage_destroy(s);
    }
    return 0;
}

/* The phased hop (begin / model / end) against the one-call process(): a
 * non-identity model (a uniform 0.5 band mask, so the output differs from the
 * input) run by the caller between begin and end reproduces the configured
 * callback byte for byte. The fail-open rows: a nonzero model result, a frame
 * left open (end without a model result is fed -1 by the caller), and a
 * non-finite head all take the identity; the counters follow. */

/* Writes every output validly, then reports failure: the result code alone
 * must send the frame down the identity path. */
static int half_mask_then_fail(void *user, const DFN2PrepostInputs *in,
                               DFN2PrepostOutputs *out) {
    dfn_fixture_half_mask_infer(user, in, out);
    return -1;
}

/* mode 0: caller runs dfn_fixture_half_mask_infer; 1: caller reports -1; 2: the model
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
    int frame, differs = 0, any = 0, opened = 0;

    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
    cb_cfg.model.io_descriptor = &descriptor;
    cb_cfg.model.infer = mode == 0 ? dfn_fixture_half_mask_infer
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
            ++opened;
            if (mode == 0) model_rc = dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs);
            else if (mode == 1) model_rc = -1;
            else if (mode == 2) model_rc = nan_head_infer(NULL, &job.inputs, &job.outputs);
            else if (mode == 4) model_rc = half_mask_then_fail(NULL, &job.inputs, &job.outputs);
            else model_rc = 0;   /* nothing was written: the NaN heads refuse the commit */
            if (dfn_res_stage_end(ph, model_rc, a) != rc_b) return 4;
        } else {
            return 5;   /* one model frame on every hop */
        }
        if (memcmp(a, b, sizeof(a)) != 0) return 10 + frame;
        if (frame >= L_FRAMES) {
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
    if (opened != FRAMES + 4) return 12;
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
    /* the first hop opens a model frame whatever the mask lookahead (a warm-up frame when it is > 0) */
    if (dfn_res_stage_begin(s, est, app, &job) != 1 || job.inputs.erb_window == NULL ||
        job.outputs.erb_mask == NULL)
        return 4;
    if (dfn_res_stage_begin(s, est, app, &job) != -1) return 5; /* a hop is already begun */
    if (dfn_res_stage_end(s, 0, NULL) != -1) return 6;
    if (dfn_res_stage_end(s, 0, out) != (L_FRAMES > 0 ? 0 : 1)) return 7;
    if (dfn_res_stage_end(s, 0, out) != -1) return 8;          /* already closed */
    for (frame = 1; frame < 4; ++frame) {
        fill_frame(est, app, frame);
        if (dfn_res_stage_begin(s, est, app, &job) != 1) return 9;
        if (job.inputs.erb_window == NULL || job.outputs.erb_mask == NULL) return 10;
        /* The stage registers no state: the model callback owns it. */
        if (job.inputs.gru_hidden != NULL || job.inputs.gru_hidden_elements != 0 ||
            job.outputs.gru_hidden_next != NULL || job.inputs.df_convp_window != NULL ||
            job.outputs.df_convp_window_next != NULL) return 13;
        if (job.inputs.erb_window_elements != dfn2_prepost_erb_window_elements(dfn_test_grid()) ||
            job.outputs.erb_mask_elements != dfn2_prepost_erb_mask_elements(dfn_test_grid()))
            return 14;
        dfn_fixture_half_mask_infer(NULL, &job.inputs, &job.outputs);
        if (dfn_res_stage_end(s, 0, out) < 0) return 11;
    }
    dfn_res_stage_reset(s);
    fill_frame(est, app, 0);
    /* reset restarts the warm-up: the next hop opens a model frame again */
    if (dfn_res_stage_begin(s, est, app, &job) != 1) return 12;
    dfn_res_stage_destroy(s);
    return 0;
}

/* ---- warm-up frames ------------------------------------------------------------------
 * The first mask_lookahead hops run the model like any other, discard its heads and leave the
 * model state to restart, so the audio equals a stream whose model starts at its first real frame.
 * A mock model with state that shapes the ERB mask stands for a board accelerator:
 *   mode 1  reloads zero whenever dfn_res_stage_state_generation() changes (the contract)
 *   mode 2  reloads zero only at its start (breaks the contract: warm-up frames leak into the state)
 *   mode 3  ignores the generation and zeroes exactly when its first real frame arrives (the
 *           reference: a model that never saw a warm-up frame)
 * `garbage_calls` first calls write a NaN and an enormous head and still report success. */
#define WARM_HOPS 12

typedef struct WarmModel {
    const DfnResStage *stage;
    int mode;
    int garbage_calls;
    int calls;
    unsigned int seen_generation;
    float state;
} WarmModel;

static int warm_infer(void *user, const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    WarmModel *m = (WarmModel *)user;
    float mask;
    size_t i;
    dfn_fixture_half_mask_infer(NULL, in, out);
    if (m->mode == 1) {
        const unsigned int generation = dfn_res_stage_state_generation(m->stage);
        if (generation != m->seen_generation) { m->state = 0.0f; m->seen_generation = generation; }
    } else if (m->mode == 3 && m->calls == M_FRAMES) {
        m->state = 0.0f;
    }
    mask = 0.5f - 0.25f * m->state;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = mask;
    m->state = 0.5f * m->state + 0.5f;
    if (m->calls < m->garbage_calls) { out->erb_mask[3] = (float)NAN; out->erb_mask[0] = 1e30f; }
    ++m->calls;
    return 0;
}

/* One stream of `passes` * WARM_HOPS hops (a reset between passes) through the mock. `out` receives
 * the first pass's spectra. Returns 0, or a nonzero code when a counter disagrees with the contract. */
static int run_warm_stream(int mode, int garbage_calls, int passes, Complex out[][DFN2_N_BINS],
                           unsigned int *generation, int *calls) {
    DFN2ModelIoDescriptor descriptor;
    DfnResStageConfig cfg = base_config();
    WarmModel m;
    DfnResStage *s;
    Complex estimate[DFN2_N_BINS], apply[DFN2_N_BINS], again[DFN2_N_BINS];
    long long frames, commits, skips;
    int pass, hop;

    memset(&m, 0, sizeof(m));
    m.mode = mode;
    m.garbage_calls = garbage_calls;
    if (dfn2_model_io_descriptor_default(dfn_test_grid(), &descriptor) != 0) return 1;
    cfg.model.infer = warm_infer;
    cfg.model.user = &m;
    cfg.model.io_descriptor = &descriptor;
    s = dfn_res_stage_create(&cfg);
    if (!s) return 2;
    m.stage = s;
    m.seen_generation = dfn_res_stage_state_generation(s);
    for (pass = 0; pass < passes; ++pass) {
        const int calls_before = m.calls;
        for (hop = 0; hop < WARM_HOPS; ++hop) {
            fill_frame(estimate, apply, hop);
            if (dfn_res_stage_process(s, estimate, apply, pass == 0 ? out[hop] : again) < 0) return 3;
            if (pass > 0 && memcmp(again, out[hop], sizeof(again)) != 0 && mode == 1) return 4;
            dfn_res_stage_get_counters(s, &frames, &commits, &skips);
            /* the model runs on every hop; a warm-up hop counts as a frame pushed and as nothing else */
            if (m.calls - calls_before != hop + 1 || frames != hop + 1) return 5;
            if (hop < M_FRAMES && (commits != 0 || skips != 0)) return 6;
            if (commits + skips != (hop + 1 > M_FRAMES ? hop + 1 - M_FRAMES : 0)) return 7;
        }
        if (pass + 1 < passes) dfn_res_stage_reset(s);   /* the mock notices it by the generation alone */
    }
    *generation = dfn_res_stage_state_generation(s);
    *calls = m.calls;
    dfn_res_stage_destroy(s);
    return 0;
}

static Complex g_warm_a[WARM_HOPS][DFN2_N_BINS], g_warm_b[WARM_HOPS][DFN2_N_BINS],
               g_warm_c[WARM_HOPS][DFN2_N_BINS];

static int run_warmup_frames(void) {
    unsigned int gen;
    int calls, rc;

    /* The contract (mode 1) with garbage heads on every warm-up call: the heads are discarded, no
     * refusal (generation exactly +1 per warm-up frame, never +2), one model call per hop. */
    rc = run_warm_stream(1, M_FRAMES, 1, g_warm_a, &gen, &calls);
    if (rc) return 10 + rc;
    if (gen != 1u + M_FRAMES || calls != WARM_HOPS) return 20;
    /* The same model with valid heads on every call: identical audio (the heads never mattered). */
    rc = run_warm_stream(1, 0, 1, g_warm_b, &gen, &calls);
    if (rc) return 30 + rc;
    if (gen != 1u + M_FRAMES || calls != WARM_HOPS) return 40;
    if (memcmp(g_warm_a, g_warm_b, sizeof(g_warm_a)) != 0) return 41;
    /* The reference: a model that starts at its first real frame. */
    rc = run_warm_stream(3, 0, 1, g_warm_c, &gen, &calls);
    if (rc) return 50 + rc;
    if (memcmp(g_warm_a, g_warm_c, sizeof(g_warm_a)) != 0) return 51;
    /* A model that reloads only at its start sees its warm-up frames in its state. */
    rc = run_warm_stream(2, 0, 1, g_warm_b, &gen, &calls);
    if (rc) return 60 + rc;
    if ((memcmp(g_warm_b, g_warm_c, sizeof(g_warm_b)) != 0) != (M_FRAMES > 0)) return 61;
    /* A reset restarts the warm-up: the second pass reproduces the first through the generation. */
    rc = run_warm_stream(1, 0, 2, g_warm_b, &gen, &calls);
    if (rc) return 70 + rc;
    if (gen != 2u + 2u * M_FRAMES || calls != 2 * WARM_HOPS) return 71;
    if (memcmp(g_warm_b, g_warm_c, sizeof(g_warm_b)) != 0) return 72;
    return 0;
}

int main(void) {
    int rc;
    DFN2_REQUIRE_VARIANT_LINK();   /* does not link against an archive of the other DFN2 variant */
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
    rc = run_table_grids();
    if (rc) { fprintf(stderr, "run-time table grid row failed: %d\n", rc); return 1; }
    rc = run_stale_pool();
    if (rc) { fprintf(stderr, "stale-pool gate failed: %d\n", rc); return 1; }
    rc = run_refused_calls();
    if (rc) { fprintf(stderr, "refused-call row failed: %d\n", rc); return 1; }
    rc = run_state_generation();
    if (rc) { fprintf(stderr, "state-generation row failed: %d\n", rc); return 1; }
    for (rc = 0; rc < 5; ++rc) {
        int r = run_phased_case(rc);
        if (r) { fprintf(stderr, "phased case %d failed: %d\n", rc, r); return 1; }
    }
    rc = run_phased_sequence();
    if (rc) { fprintf(stderr, "phased sequence row failed: %d\n", rc); return 1; }
    rc = run_warmup_frames();
    if (rc) { fprintf(stderr, "warm-up row failed: %d\n", rc); return 1; }
    {   /* the pair this program was built with, as the stage measures it (make test-pairs reads it) */
        int measured = -1;
        rc = run_stage_on_grid(dfn_test_grid(), DFN_TEST_LOOKAHEAD_FRAMES, &measured, NULL);
        if (rc) { fprintf(stderr, "test grid latency row failed: %d\n", rc); return 1; }
        dfn_test_report_pair("dfn_res_stage", measured);
    }
    puts("dfn_res_stage: PASS");
    return 0;
}
