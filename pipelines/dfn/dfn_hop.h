/* dfn_hop.h -- example: one DeepFilterNet2 hop behind an AEC + RES chain. Header-only, no ABI:
 * copy and adapt.
 *
 * The host forms E (the linear-AEC error) and P (E after the RES gain, plus comfort noise when
 * enabled; the DFN example mains show it) before the hop begins. DFN2 then runs on (E, P) through
 * the model-stream layer (stream/ainr/dfn2: dfn2_stream_pre_process_freq_dual /
 * dfn2_stream_post_process_freq in memory mode); this header adds what that layer leaves to its
 * caller: the audio-scale spectra in and out (dfn2_stft_scale in, dfn2_stft_unscale out), the
 * model callback, and the synthesis.
 *   host rate == the model grid's rate   DfnHop synthesises through the caller's AcIstft
 *   other host rates                     stream/ainr/bridge's AinrBridge puts the hop on its 48 kHz
 *                                        grid, runs the model there and synthesises the host hop
 *
 *     dfn_hop_begin(&h, E, P);
 *     while (dfn_hop_nn_begin(&h, &job) == 1)       // a model frame is open
 *         dfn_hop_nn_end(&h, run_model(&job));      // run status (stream_run.h): 0 = every output written
 *     dfn_hop_end(&h, out);                         // one host hop
 *
 * The direct path opens one model frame per hop from hop mask_lookahead on (the first mask_lookahead hops after
 * a start have none: nn_begin returns 0 and the model does not run), the bridge path 0, 1 or 2 by its
 * schedule. A frame the model refuses or leaves unwritten, a frame left open and a frame never fetched all
 * take the identity through the cascade.
 * Run status and state generation: STREAM_RUN_* in stream_run.h. A frame left open reports
 * STREAM_RUN_FAILED, a frame never fetched, a hop with no model frame and the identity model STREAM_RUN_NOT_STARTED.
 *
 * dfn_hop_process() does the same with the model callback of the pool plan. The pointers are not
 * owned; the stream, bridge, synthesis and spectra come from the caller's pool (dfn_hop_pool.h).
 * The hop does not reset on an AEC delay change. It adds dfn_hop_lookahead_frames() hops of delay
 * on the model's own grid (2 for the shipped 1/1 pair at 48 kHz) and
 * ainr_bridge_added_delay_samples() through the bridge; the first output hops are zero. An AEC
 * post-chain reset event (aec_post_reset_serial() moved) is followed by the caller before the
 * hop with dfn_reset_follow(): through the bridge it clears the host-grid synthesis on the hop of the
 * event (that synthesis is inside the bridge and runs at the top of dfn_hop_begin()); with the model
 * grid driven directly it clears the hop's AcIstft dfn_hop_lookahead_frames() hops later. A negative return leaves recovery to the caller: reset the AEC front
 * end, the comfort-noise generator and DfnHop together. */
#ifndef PIPELINES_DFN_DFN_HOP_H
#define PIPELINES_DFN_DFN_HOP_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "ainr_bridge.h"
#include "audio_stft.h"
#include "dfn2_stream.h"
#include "utility/simd_kernel_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DFN_HOP_INVALID = -1,         /* NULL argument, a refused block, or a non-finite hop */
    DFN_HOP_SEQUENCE_ERROR = -2   /* a call out of begin / nn_begin / nn_end / end order */
};

/* Doubles as the model's per-frame views: job.inputs / job.outputs, job.state_generation. */
typedef DFN2HopIO DfnHopJob;

typedef struct DfnHop {
    DFN2Stream* stream;
    AinrBridge* bridge;           /* other host rates: the bridge around the stream; else NULL */
    DFN2Model model;              /* by value; no infer = every frame the identity */
    AcIstft* istft;               /* direct path: synthesis of out_spec */
    Complex* out_spec;            /* direct path: n_bins bins, audio scale */
    float* planar;                /* 6 * n_bins: E and P (re, im) at model scale, then the output */
    int n_bins;                   /* the model grid's */
    int hop_size;                 /* direct path: the model grid's hop */
    float scale, unscale;         /* dfn2_stft_scale / dfn2_stft_unscale of the model grid's fft */
    int check_finite;             /* direct path: refuse a non-finite out_spec or hop at end */

    int phase;                    /* 0 idle, 1 hop begun, 2 model frame open */
    DfnHopJob job;                /* the frame begun, until fetched */
    int job_pending;              /* direct path: job holds an unfetched frame */
    int frame_ended;              /* direct path: the frame is closed */
    int frame_open;               /* bridge path: a model frame is open */
    Complex* frame_out;           /* bridge path: the open frame's output spectrum */
} DfnHop;

/* Frames the emitted spectrum trails the newest input by: dfn2_prepost_latency_frames(), L + D. 0 for NULL. */
static inline int dfn_hop_lookahead_frames(const DFN2Grid* grid) {
    return grid ? dfn2_prepost_latency_frames(grid) : 0;
}

/* 1 when the model is usable on `grid`: no infer needs no reset and no descriptor; an infer needs a
 * descriptor valid for the grid (another grid, lookahead pair or frame count is refused). */
static inline int dfn_hop_model_valid(const DFN2Model* model, const DFN2Grid* grid) {
    if (!model || !grid) return 0;
    if (!model->infer) return model->reset == NULL && model->io_descriptor == NULL;
    return model->io_descriptor != NULL &&
           dfn2_model_io_descriptor_validate(grid, model->io_descriptor) == 0;
}

static inline unsigned int dfn_hop_state_generation(const DfnHop* h) {
    return h ? dfn2_stream_state_generation(h->stream) : 0u;
}

/* Between hops only; 0, or -1 on NULL, a non-finite value or (direct path) a hop begun and not yet ended. */
static inline int dfn_hop_set_atten_lim(DfnHop* h, float atten_lim_db) {
    return h ? dfn2_stream_set_atten_lim(h->stream, atten_lim_db) : -1;
}

/* The model callback on an open frame: its run status, or STREAM_RUN_NOT_STARTED when none is set. */
static inline int dfn_hop_infer(const DfnHop* h, DfnHopJob* job) {
    if (!h || !job || !h->model.infer) return STREAM_RUN_NOT_STARTED;
    return h->model.infer(h->model.user, &job->inputs, &job->outputs);
}

/* The fields both paths share; 0, or -1 (h untouched) for NULL h / grid / model or frames_per_call != 1. */
static inline int dfn_hop_init_common(DfnHop* h, DFN2Stream* stream, const DFN2Grid* grid, const DFN2Model* model,
                                      float* planar) {
    if (!h || !grid || !model || grid->frames_per_call != 1) return -1;
    memset(h, 0, sizeof(*h));
    h->stream = stream;
    h->model = *model;
    h->planar = planar;
    h->n_bins = grid->n_bins;
    h->scale = dfn2_stft_scale(grid->n_fft);
    h->unscale = dfn2_stft_unscale(grid->n_fft);
    memset(planar, 0, (size_t)6 * (size_t)grid->n_bins * sizeof(float));
    return 0;
}

/* Host on the model's grid. `planar` holds 6 * n_bins floats, `out_spec` n_bins bins; both are
 * zeroed here. 0, or -1 (h untouched) for NULL h / grid / model or frames_per_call != 1. */
static inline int dfn_hop_init_direct(DfnHop* h, DFN2Stream* stream, const DFN2Grid* grid,
                                      const DFN2Model* model, AcIstft* istft, Complex* out_spec,
                                      float* planar) {
    if (dfn_hop_init_common(h, stream, grid, model, planar) != 0) return -1;
    h->istft = istft;
    h->out_spec = out_spec;
    h->hop_size = grid->hop;
    memset(out_spec, 0, (size_t)grid->n_bins * sizeof(Complex));
    return 0;
}

/* Host rate differs from the model grid (bridge path). 0, or -1 as dfn_hop_init_direct(). */
static inline int dfn_hop_init_bridge(DfnHop* h, DFN2Stream* stream, const DFN2Grid* grid,
                                      const DFN2Model* model, AinrBridge* bridge, float* planar) {
    if (dfn_hop_init_common(h, stream, grid, model, planar) != 0) return -1;
    h->bridge = bridge;
    return 0;
}

/* Back to hop #0: the stream, the model (reset callback), the bridge or the synthesis, and the
 * output spectrum. */
static inline void dfn_hop_reset(DfnHop* h) {
    if (!h) return;
    h->phase = 0;
    h->job_pending = 0;
    h->frame_ended = 0;
    h->frame_open = 0;
    dfn2_stream_reset(h->stream);
    if (h->model.reset) h->model.reset(h->model.user);
    if (h->bridge) {
        ainr_bridge_reset(h->bridge);
        return;
    }
    if (h->istft) ac_istft_reset(h->istft);
    memset(h->out_spec, 0, (size_t)h->n_bins * sizeof(Complex));
}

/* Takes the audio-scale spectra e (estimated on) and p (applied on), n_bins bins each, scaled to the model's spectra;
 * job->action is STREAM_WORK_BYPASS for the first mask_lookahead hops. A bin non-finite or beyond 2^63 at model scale
 * refuses the block whole and leaves the stream untouched. 0, or -1. */
static inline int dfn_hop_open_frame(DfnHop* h, const Complex* e, const Complex* p, DfnHopJob* job) {
    float* w = h->planar;
    const int n = h->n_bins;
    skn_deinterleave_scale_cf32(e, w, w + n, n, h->scale);
    skn_deinterleave_scale_cf32(p, w + 2 * n, w + 3 * n, n, h->scale);
    return dfn2_stream_pre_process_freq_dual(h->stream, w, w + n, w + 2 * n, w + 3 * n, job) == 1 ? 0 : -1;
}

/* Closes the open frame into `out` (n_bins bins, audio scale; zeros while no spectrum is due):
 * the model's output is committed only for run status 0 and a finite output, anything else is
 * the identity. 0, or -1. */
static inline int dfn_hop_close_frame(DfnHop* h, int model_rc, Complex* out) {
    float* w = h->planar + 4 * h->n_bins;
    int valid = 0;
    if (dfn2_stream_post_process_freq(h->stream, model_rc, w, w + h->n_bins, &valid) != 0) return -1;
    if (valid) skn_interleave_scale_cf32(w, w + h->n_bins, out, h->n_bins, h->unscale);
    else memset(out, 0, (size_t)h->n_bins * sizeof(Complex));
    return 0;
}

/* Starts a hop on the host-grid spectra E and P (borrowed for this call only). A non-finite bin, or one beyond the
 * bound of the path (|bin| * scale > 2^63; on the bridge path ainr_bridge_begin's 2^68), is refused with the hop still
 * idle (DFN_HOP_INVALID). On the bridge path a model-grid frame that the resamplers push beyond the bound is not
 * run: its output spectrum is zero and the hop completes. */
static inline int dfn_hop_begin(DfnHop* h, const Complex* e, const Complex* p) {
    if (!h || !e || !p) return DFN_HOP_INVALID;
    if (h->phase != 0) return DFN_HOP_SEQUENCE_ERROR;
    h->job_pending = 0;
    h->frame_ended = 0;
    if (h->bridge) {
        if (ainr_bridge_begin(h->bridge, e, p) != 0) return DFN_HOP_INVALID;
    } else {
        if (dfn_hop_open_frame(h, e, p, &h->job) != 0) return DFN_HOP_INVALID;
        h->job_pending = h->job.action == STREAM_WORK_SUBMIT;
    }
    h->phase = 1;
    return 0;
}

/* Opens the bridge's next model frame: 1, 0 when the hop has no further frame, or -1. */
static inline int dfn_hop_bridge_frame(DfnHop* h) {
    AinrBridgeFrame frame;
    for (;;) {
        int r = ainr_bridge_model_begin(h->bridge, &frame);
        if (r != 1) return r;
        if (dfn_hop_open_frame(h, frame.estimate_spec, frame.apply_spec, &h->job) != 0) return -1;
        if (h->job.action == STREAM_WORK_SUBMIT) {
            h->frame_out = frame.out_spec;
            h->frame_open = 1;
            return 1;
        }
        /* a frame the model does not see (the first mask_lookahead): closed here, the next one is fetched */
        if (dfn_hop_close_frame(h, STREAM_RUN_NOT_STARTED, frame.out_spec) != 0 || ainr_bridge_model_end(h->bridge) != 0)
            return -1;
    }
}

static inline int dfn_hop_bridge_frame_end(DfnHop* h, int model_rc) {
    h->frame_open = 0;
    if (dfn_hop_close_frame(h, model_rc, h->frame_out) != 0) return -1;
    return ainr_bridge_model_end(h->bridge);
}

/* 1 with the open frame's tensors in *job, 0 when the hop has no further frame, or a negative
 * status. */
static inline int dfn_hop_nn_begin(DfnHop* h, DfnHopJob* job) {
    int r;
    if (!h || !job) return DFN_HOP_INVALID;
    if (h->phase != 1) return DFN_HOP_SEQUENCE_ERROR;
    if (h->bridge) {
        r = dfn_hop_bridge_frame(h);
        if (r < 0) return DFN_HOP_INVALID;
    } else if (h->job_pending) {
        h->job_pending = 0;
        r = 1;
    } else {
        r = 0;
    }
    if (r == 1) {
        *job = h->job;
        h->phase = 2;
    }
    return r;
}

/* Closes the open frame with the model's run status: the output is committed only for status 0 and a finite
 * output; anything else takes the frame as the identity. */
static inline int dfn_hop_nn_end(DfnHop* h, int model_rc) {
    if (!h) return DFN_HOP_INVALID;
    if (h->phase != 2) return DFN_HOP_SEQUENCE_ERROR;
    h->phase = 1;
    if (h->bridge) {
        if (dfn_hop_bridge_frame_end(h, model_rc) != 0) return DFN_HOP_INVALID;
    } else {
        if (dfn_hop_close_frame(h, model_rc, h->out_spec) != 0) return DFN_HOP_INVALID;
        h->frame_ended = 1;
    }
    return 0;
}

/* Emits one host hop into out. A frame left open (STREAM_RUN_FAILED: the caller may have run it) or never fetched
 * (STREAM_RUN_NOT_STARTED) is the identity; a hop with no model frame needs no nn_begin or nn_end. */
static inline int dfn_hop_end(DfnHop* h, float* out) {
    int k;
    if (!h || !out) return DFN_HOP_INVALID;
    if (h->phase == 0) return DFN_HOP_SEQUENCE_ERROR;
    if (h->bridge) {
        int r;
        h->phase = 0;
        if (h->frame_open && dfn_hop_bridge_frame_end(h, STREAM_RUN_FAILED) != 0) return DFN_HOP_INVALID;
        while ((r = dfn_hop_bridge_frame(h)) == 1)
            if (dfn_hop_bridge_frame_end(h, STREAM_RUN_NOT_STARTED) != 0) return DFN_HOP_INVALID;
        if (r < 0) return DFN_HOP_INVALID;
        return ainr_bridge_finish(h->bridge, out) != 0 ? DFN_HOP_INVALID : 0;
    }
    if (!h->frame_ended &&
        dfn_hop_close_frame(h,
                            h->job_pending || h->job.action != STREAM_WORK_SUBMIT ? STREAM_RUN_NOT_STARTED
                                                                                  : STREAM_RUN_FAILED,
                            h->out_spec) != 0) {
        h->phase = 0;
        return DFN_HOP_INVALID;
    }
    h->phase = 0;
    h->job_pending = 0;
    if (h->check_finite) {
        for (k = 0; k < h->n_bins; ++k)
            if (!isfinite(h->out_spec[k].r) || !isfinite(h->out_spec[k].i))
                return DFN_HOP_INVALID;
    }
    (void)ac_istft_push(h->istft, h->out_spec, out);
    if (h->check_finite) {
        for (k = 0; k < h->hop_size; ++k)
            if (!isfinite(out[k])) return DFN_HOP_INVALID;
    }
    return 0;
}

/* The whole hop with the pool plan's model callback (identity when unset). */
static inline int dfn_hop_process(DfnHop* h, const Complex* e, const Complex* p, float* out) {
    DfnHopJob job;
    int r;
    if (!h || !out) return DFN_HOP_INVALID;
    r = dfn_hop_begin(h, e, p);
    if (r != 0) return r;
    while ((r = dfn_hop_nn_begin(h, &job)) == 1) {
        int nr = dfn_hop_nn_end(h, dfn_hop_infer(h, &job));
        if (nr != 0) return nr;
    }
    if (r < 0) return r;
    return dfn_hop_end(h, out);
}

/* Follows an upstream hard post-chain reset event. Direct path: the frame built on the hop of the event reaches the
 * synthesis dfn_hop_lookahead_frames() hops later, so the hop's AcIstft is cleared just before that push and DFN2 keeps
 * running. Bridge path: its host-grid synthesis is cleared on the hop of the event. */
typedef struct DfnResetFollow {
    uint32_t pending;             /* bit k: an event was seen k hops ago */
    int delay;                    /* hops from the event to the clear: the model's lookahead, below 32 */
} DfnResetFollow;

static inline void dfn_reset_follow_init(DfnResetFollow* f, const DFN2Grid* grid) {
    f->pending = 0u;
    f->delay = dfn_hop_lookahead_frames(grid);
}

/* After dfn_hop_reset(): no event is pending. */
static inline void dfn_reset_follow_clear(DfnResetFollow* f) {
    f->pending = 0u;
}

/* Once per hop before dfn_hop_begin(); `moved` is 1 when the event is on this hop. Returns 1 when it cleared a
 * synthesis, 0 when not, or the bridge's refusal (negative; it cannot refuse between hops). */
static inline int dfn_reset_follow(DfnResetFollow* f, DfnHop* h, int moved) {
    if (h->bridge) return moved ? (ainr_bridge_reset_synthesis(h->bridge) == 0 ? 1 : DFN_HOP_INVALID) : 0;
    if (!h->istft) return 0;
    f->pending = (f->pending << 1) | (uint32_t)(moved != 0);
    if (!((f->pending >> f->delay) & 1u)) return 0;
    ac_istft_reset(h->istft);
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_DFN_DFN_HOP_H */
