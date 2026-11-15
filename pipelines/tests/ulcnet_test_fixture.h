/* Shared fixture of the Align-ULCNet pipeline tests (mono and 4ch): the grids they run, the grid under test, the
 * stand-in accelerator, and the rig that drives the spectrum entries of stream/aiaec/ulcnet with it the way the
 * example mains do. Header-only; define nothing it names before including it. */

#ifndef ULCNET_TEST_FIXTURE_H
#define ULCNET_TEST_FIXTURE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "ulcnet_stream.h"

enum { HOP_MAX = 512, BINS_MAX = 513 };
enum { REPRIME_FRAMES = (int)ULCNET_STREAM_REPRIME_FRAMES };

static const UlcnetGridSpec k_grids[] = {
    { .sample_rate = 16000, .n_fft = 512,  .delay_depth = 8 },
    { .sample_rate = 48000, .n_fft = 1024, .delay_depth = 8 },
};

static UlcnetGrid g_grid;   /* the grid under test */
#define HOP (g_grid.hop)
#define BINS (g_grid.n_bins)

/* The descriptor of the grid under test, as a graph exported for it publishes. */
static inline const UlcnetModelIoDescriptor* test_io_descriptor(void) {
    static UlcnetModelIoDescriptor d;
    return ulcnet_model_io_descriptor_default(&g_grid, &d) == 0 ? &d : NULL;
}

/* The table row that is not the grid under test. */
static inline UlcnetGrid other_grid(void) {
    UlcnetGrid other;
    memset(&other, 0, sizeof(other));
    for (size_t i = 0; i < sizeof(k_grids) / sizeof(k_grids[0]); ++i)
        if (k_grids[i].n_fft != g_grid.n_fft) (void)ulcnet_grid_make(&other, &k_grids[i]);
    return other;
}

/* Reprime hops a block of `frames` hops skips out of the `*armed` still owed. */
static inline int reprime_take(int* armed, int frames) {
    int skipped = *armed < frames ? *armed : frames;
    *armed -= skipped;
    return skipped;
}

static inline float spec_peak(const float* re, const float* im) {
    float m = 0.0f;
    for (int k = 0; k < BINS; ++k) {
        float a = fabsf(re[k]);
        if (a > m) m = a;
        a = fabsf(im[k]);
        if (a > m) m = a;
    }
    return m;
}

/* ---- the graph's arithmetic, written again here (scalar, unfused like the library) ---- */

/* The real mask value that scales a spectrum by `gain` (the mask acts on the power-law compressed spectrum). */
static inline float mask_of_gain(float gain) { return powf(gain, ULCNET_MODEL_IO_COMPRESSION_EXP); }

static inline void mask_fill(float* mask_re, float* mask_im, float value) {
    for (int k = 0; k < BINS; ++k) { mask_re[k] = value; mask_im[k] = 0.0f; }
}

/* What an accepted hop is: the compressed error times the complex mask, then the inverse compression. */
static inline void enhanced_of_mask(const float* err_re, const float* err_im, const float* mask_re, const float* mask_im,
                                    float* out_re, float* out_im) {
    const float inverse = 1.0f / ULCNET_MODEL_IO_COMPRESSION_EXP;
    for (int k = 0; k < BINS; ++k) {
        const float c_re = ulcnet_model_io_signed_pow(err_re[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float c_im = ulcnet_model_io_signed_pow(err_im[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        out_re[k] = ulcnet_model_io_signed_pow(c_re * mask_re[k] - c_im * mask_im[k], inverse);
        out_im[k] = ulcnet_model_io_signed_pow(c_re * mask_im[k] + c_im * mask_re[k], inverse);
    }
}

/* As enhanced_of_mask() for a real mask that is mask_of_gain(gain) in every bin. */
static inline void enhanced_of_gain(const float* err_re, const float* err_im, float gain, float* out_re, float* out_im) {
    float mask_re[BINS_MAX], mask_im[BINS_MAX];
    mask_fill(mask_re, mask_im, mask_of_gain(gain));
    enhanced_of_mask(err_re, err_im, mask_re, mask_im, out_re, out_im);
}

/* 1 when the four signal tensors are, bit for bit, the front end of these spectra. */
static inline int tensors_are_front_end_of(const UlcnetModelIoInputs* in, const float* err_re, const float* err_im,
                                           const float* far_re, const float* far_im) {
    if (in->spectrum_bins_elements != (size_t)BINS) return 0;
    for (int k = 0; k < BINS; ++k) {
        const float c_re = ulcnet_model_io_signed_pow(err_re[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float c_im = ulcnet_model_io_signed_pow(err_im[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float f_re = ulcnet_model_io_signed_pow(far_re[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float f_im = ulcnet_model_io_signed_pow(far_im[k], ULCNET_MODEL_IO_COMPRESSION_EXP);
        const float norm = hypotf(c_re, c_im);
        const float want[4] = { sqrtf(c_re * c_re + c_im * c_im + 1e-12f), sqrtf(f_re * f_re + f_im * f_im + 1e-12f),
                                norm > 0.0f ? c_re / norm : 1.0f, norm > 0.0f ? c_im / norm : 0.0f };
        const float got[4] = { in->error_mag[k], in->far_mag[k], in->error_cos[k], in->error_sin[k] };
        if (memcmp(want, got, sizeof(want)) != 0) return 0;
    }
    return 1;
}

/* ---- the stand-in accelerator ---- */

/* One graph run at spectrum level. infer(): gets the hop's linear-error and raw-far spectra (the rig checks that the
 * tensors are their front end) and writes the complex mask, BINS per plane, NaN-prefilled; 0 requires every bin.
 * reset(): the state restart of an alignment boundary or a pipeline reset. io_descriptor: published beside the graph,
 * NULL for none. An all-zero model never runs: every hop closes as a failed run, as in the example mains. */
typedef struct UlcTestModel {
    void* user;
    int (*infer)(void* user, const float* err_re, const float* err_im, const float* far_re, const float* far_im,
                 float* mask_re, float* mask_im);
    void (*reset)(void* user);
    const UlcnetModelIoDescriptor* io_descriptor;
} UlcTestModel;

/* One graph run in flight: the views infer() gets. */
typedef struct UlcTestJob {
    const float *err_re, *err_im, *far_re, *far_im;
    float *mask_re, *mask_im;
} UlcTestJob;

/* The mask is 1: the graph changes nothing but the rounding of the compression. */
static inline int unit_mask_infer(void* user, const float* err_re, const float* err_im, const float* far_re,
                                  const float* far_im, float* mask_re, float* mask_im) {
    (void)user; (void)err_re; (void)err_im; (void)far_re; (void)far_im;
    mask_fill(mask_re, mask_im, 1.0f);
    return 0;
}

/* Records the far spectrum the frame carries (unit mask); `called` is cleared by the test before each hop. */
typedef struct UlcFarCapture {
    float re[BINS_MAX], im[BINS_MAX];
    int called;
} UlcFarCapture;

static inline int far_capture_infer(void* user, const float* err_re, const float* err_im, const float* far_re,
                                    const float* far_im, float* mask_re, float* mask_im) {
    UlcFarCapture* st = (UlcFarCapture*)user;
    (void)err_re; (void)err_im;
    memcpy(st->re, far_re, (size_t)BINS * sizeof(float));
    memcpy(st->im, far_im, (size_t)BINS * sizeof(float));
    st->called = 1;
    mask_fill(mask_re, mask_im, 1.0f);
    return 0;
}

/* ---- the rig: one UlcnetStream on its spectrum entries, the model, and what the tests count ---- */

typedef struct UlcTestRig {
    UlcnetStream* stream;
    UlcTestModel model;
    const float *err_re, *err_im, *far_re, *far_im;   /* the spectra handed to the hop in flight */
    UlcnetHopIO io;
    int begun;       /* a hop is between begin and end */
    int frame;       /* 0 no model frame (reprime, or closed), 1 open and not fetched, 2 fetched */
    int ai_rc;       /* what the hop is closed with: NOT_STARTED until the frame is fetched, FAILED until a job reports */
    long edges;      /* hops on which the stream restarted the model before opening the frame */
    long refused;    /* hops closed with a result the stream refused or a dirty status (the generation moved while closing) */
    long broken;     /* contract violations seen by the rig; every test expects 0 (see ulc_rig_begin) */
} UlcTestRig;

/* Pool bytes of the stream for `grid` (spectrum-only, cfg.freq_only: the rig drives the _freq entries), 0 for a grid it
 * refuses. */
static inline size_t ulc_rig_stream_bytes(const UlcnetGrid* grid) {
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    if (!grid) return 0;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = *grid;
    cfg.freq_only = 1;
    return ulcnet_stream_get_mem_size(&cfg, &req) == 0 ? (size_t)req.bytes : 0;
}

/* Builds the stream in `mem` (16-byte aligned, ulc_rig_stream_bytes() bytes) with the stale-pool gate; no model. */
static inline int ulc_rig_init(UlcTestRig* r, void* mem, size_t bytes, const UlcnetGrid* grid) {
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    memset(r, 0, sizeof(*r));
    if (!grid) return -1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = *grid;
    cfg.freq_only = 1;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) return -1;
    r->stream = ulcnet_stream_init_ex(&cfg, mem, bytes, &req);
    return r->stream ? 0 : -1;
}

/* Installs the model (NULL clears it). As the example mains do with the graph's metadata, a published descriptor is
 * checked against the grid first: -1 leaves the installed model in place. */
static inline int ulc_rig_set_model(UlcTestRig* r, const UlcnetGrid* grid, const UlcTestModel* model) {
    if (!r) return -1;
    if (model && model->io_descriptor &&
        (ulcnet_model_io_descriptor_validate(grid, model->io_descriptor) != 0 ||
         ulcnet_model_io_descriptor_frames(model->io_descriptor) != 1))
        return -1;
    if (model) r->model = *model;
    else memset(&r->model, 0, sizeof(r->model));
    return 0;
}

static inline void ulc_rig_reset(UlcTestRig* r) {
    if (!r) return;
    ulcnet_stream_reset(r->stream);
    r->begun = 0;
    r->frame = 0;
    if (r->model.reset) r->model.reset(r->model.user);
}

/* Starts a hop on the caller's spectra (BINS floats per plane, valid until ulc_rig_end()). 0, or -1 when the stream
 * refuses. Counted as broken: a generation step that is not one edge, a frame whose generation is not the stream's,
 * state views, a mask staging that is not NaN-filled, tensors that are not the front end of these spectra. */
static inline int ulc_rig_begin(UlcTestRig* r, const float* err_re, const float* err_im, const float* far_re,
                                const float* far_im, UlcnetStreamDelay delay) {
    uint32_t before, after;
    int pre;
    if (!r) return -1;
    before = ulcnet_stream_state_generation(r->stream);
    pre = ulcnet_stream_pre_process_freq(r->stream, err_re, err_im, far_re, far_im, delay, &r->io);
    if (pre < 0) return -1;
    after = ulcnet_stream_state_generation(r->stream);
    if (after != before) {
        if (after != before + 1u) r->broken++;
        r->edges++;
        if (r->model.reset) r->model.reset(r->model.user);
    }
    r->err_re = err_re; r->err_im = err_im; r->far_re = far_re; r->far_im = far_im;
    r->begun = 1;
    r->frame = pre == 1;
    r->ai_rc = STREAM_RUN_NOT_STARTED;
    if (pre == 1) {
        int nan_filled = r->io.outputs.output != NULL && r->io.outputs.spectrum_ri_elements == 2u * (size_t)BINS;
        for (int k = 0; nan_filled && k < 2 * BINS; ++k) nan_filled = isnan(r->io.outputs.output[k]);
        if (r->io.state_generation != after || r->io.inputs.h_gru != NULL || r->io.outputs.h_gru_out != NULL ||
            !nan_filled || !tensors_are_front_end_of(&r->io.inputs, err_re, err_im, far_re, far_im))
            r->broken++;
    } else if (after == before) {
        r->broken++;   /* a hop without a frame that is not a reprime hop */
    }
    return 0;
}

/* 1 with the frame's views in *job (run the graph, then ulc_rig_job_done), 0 when the hop has no frame to run (a
 * reprime hop, or already fetched), -1 outside a hop. */
static inline int ulc_rig_job(UlcTestRig* r, UlcTestJob* job) {
    if (!r || !job || !r->begun) return -1;
    if (r->frame != 1) return 0;
    job->err_re = r->err_re; job->err_im = r->err_im;
    job->far_re = r->far_re; job->far_im = r->far_im;
    job->mask_re = r->io.outputs.output;
    job->mask_im = r->io.outputs.output + BINS;
    r->ai_rc = STREAM_RUN_FAILED;
    r->frame = 2;
    return 1;
}

/* The graph's result for the fetched frame; -1 when none is fetched. */
static inline int ulc_rig_job_done(UlcTestRig* r, int model_rc) {
    if (!r || !r->begun || r->frame != 2) return -1;
    r->ai_rc = model_rc;
    r->frame = 0;
    return 0;
}

/* Closes the hop: the enhanced spectrum of an accepted frame, else the error spectrum; a frame never fetched closes as
 * a run that never started, one fetched and never reported as a failed run. 0, or -1 when the stream refuses (the hop
 * then stays as it was). Counted as broken: a step that is not one for a dirty status or none for NOT_STARTED. */
static inline int ulc_rig_end(UlcTestRig* r, float* out_re, float* out_im) {
    uint32_t before, step;
    if (!r) return -1;
    before = ulcnet_stream_state_generation(r->stream);
    if (ulcnet_stream_post_process_freq(r->stream, r->ai_rc, out_re, out_im) != 1) return -1;
    step = ulcnet_stream_state_generation(r->stream) - before;
    if (step > 1u || (r->ai_rc == STREAM_RUN_NOT_STARTED && step != 0u) ||
        (r->ai_rc != STREAM_RUN_OK && r->ai_rc != STREAM_RUN_NOT_STARTED && step != 1u))
        r->broken++;
    if (step != 0u) r->refused++;
    r->begun = 0;
    r->frame = 0;
    r->ai_rc = STREAM_RUN_NOT_STARTED;
    return 0;
}

/* After ulc_rig_begin(): the installed model on the hop's frame, then ulc_rig_end(). */
static inline int ulc_rig_process(UlcTestRig* r, float* out_re, float* out_im) {
    UlcTestJob job;
    if (r && r->model.infer && ulc_rig_job(r, &job) == 1)
        (void)ulc_rig_job_done(r, r->model.infer(r->model.user, job.err_re, job.err_im, job.far_re, job.far_im,
                                                 job.mask_re, job.mask_im));
    return ulc_rig_end(r, out_re, out_im);
}

#endif /* ULCNET_TEST_FIXTURE_H */
