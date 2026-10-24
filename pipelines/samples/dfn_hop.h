/* dfn_hop.h -- example: one DeepFilterNet2 hop behind an AEC + RES chain. Header-only, no ABI:
 * copy and adapt.
 *
 * The host forms E (the linear-AEC error) and P (E after the RES gain, plus comfort noise when
 * enabled; the DFN example mains show it) before the hop begins. DFN2 then runs on (E, P):
 *   48 kHz       the stage (dfn_res_stage.h) on the host grid; DfnHop synthesises its output
 *                through the caller's AcIstft
 *   other rates  dfn_rate_bridge.h runs the same stage on its 48 kHz grid and synthesises
 *                the host hop itself
 *
 *     dfn_hop_begin(&h, E, P);
 *     while (dfn_hop_nn_begin(&h, &job) == 1)       // a model frame is open (0, 1 or 2 per hop)
 *         dfn_hop_nn_end(&h, run_model(&job));      // 0 = every output written
 *     dfn_hop_end(&h, out);                         // one host hop
 *
 * dfn_hop_process() does the same with the stage's configured model callback. The pointers
 * are not owned; the stage, bridge, synthesis and output spectrum come from the caller's
 * pool. The stage does not reset on an AEC delay change. It adds two hops of delay at 48 kHz
 * and dfn_rate_bridge_added_delay_samples() otherwise; the first output hops are zero. A
 * negative return leaves recovery to the caller: reset the AEC front end, the comfort-noise
 * generator and DfnHop together. */
#ifndef PIPELINES_SAMPLES_DFN_HOP_H
#define PIPELINES_SAMPLES_DFN_HOP_H

#include <math.h>
#include <string.h>

#include "audio_stft.h"
#include "dfn_rate_bridge.h"
#include "dfn_res_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DFN_HOP_INVALID = -1,         /* NULL argument, a refused stage contract, or a non-finite hop */
    DFN_HOP_SEQUENCE_ERROR = -2   /* a call out of begin / nn_begin / nn_end / end order */
};

typedef struct DfnHop {
    DfnResStage* stage;           /* 48 kHz: driven directly (bridge == NULL) */
    DfnRateBridge* bridge;        /* other rates: the stage inside the bridge */
    AcIstft* istft;               /* stage path: synthesis of stage_out */
    Complex* stage_out;           /* stage path: DFN2_N_BINS bins */
    int check_finite;             /* stage path: refuse a non-finite stage_out or hop at end */

    int phase;                    /* 0 idle, 1 hop begun, 2 model frame open */
    DfnResStageJob job;           /* stage path: the frame begun, until fetched */
    int job_pending;              /* stage path: job holds an unfetched frame */
    int stage_ended;              /* stage path: the stage frame is closed */
} DfnHop;

/* 48 kHz host. `stage_out` holds DFN2_N_BINS bins. */
static inline void dfn_hop_init_stage(DfnHop* h, DfnResStage* stage, AcIstft* istft,
                                      Complex* stage_out) {
    memset(h, 0, sizeof(*h));
    h->stage = stage;
    h->istft = istft;
    h->stage_out = stage_out;
}

/* Host below 48 kHz. */
static inline void dfn_hop_init_bridge(DfnHop* h, DfnRateBridge* bridge) {
    memset(h, 0, sizeof(*h));
    h->bridge = bridge;
    h->stage = dfn_rate_bridge_stage(bridge);
}

/* Back to hop #0: the stage or bridge (and its stage), the synthesis and the output spectrum. */
static inline void dfn_hop_reset(DfnHop* h) {
    if (!h) return;
    h->phase = 0;
    h->job_pending = 0;
    h->stage_ended = 0;
    if (h->bridge) {
        dfn_rate_bridge_reset(h->bridge);   /* resets its stage too */
        return;
    }
    dfn_res_stage_reset(h->stage);
    if (h->istft) ac_istft_reset(h->istft);
    memset(h->stage_out, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
}

/* Starts a hop on the host-grid spectra E and P (borrowed for this call only). */
static inline int dfn_hop_begin(DfnHop* h, const Complex* e, const Complex* p) {
    int r;
    if (!h || !e || !p) return DFN_HOP_INVALID;
    if (h->phase != 0) return DFN_HOP_SEQUENCE_ERROR;
    h->job_pending = 0;
    h->stage_ended = 0;
    if (h->bridge) {
        if (dfn_rate_bridge_begin(h->bridge, e, p) != 0) return DFN_HOP_INVALID;
    } else {
        r = dfn_res_stage_begin(h->stage, e, p, &h->job);
        if (r < 0) return DFN_HOP_INVALID;
        h->job_pending = r == 1;
    }
    h->phase = 1;
    return 0;
}

/* 1 with the open frame's tensors in *job, 0 when the hop has no further frame, or a negative
 * status. */
static inline int dfn_hop_nn_begin(DfnHop* h, DfnResStageJob* job) {
    int r;
    if (!h || !job) return DFN_HOP_INVALID;
    if (h->phase != 1) return DFN_HOP_SEQUENCE_ERROR;
    if (h->bridge) {
        r = dfn_rate_bridge_nn_begin(h->bridge, job);
        if (r < 0) return DFN_HOP_INVALID;
    } else if (h->job_pending) {
        *job = h->job;
        h->job_pending = 0;
        r = 1;
    } else {
        r = 0;
    }
    if (r == 1) h->phase = 2;
    return r;
}

/* Closes the open frame: the model output is committed only for model_rc == 0 and a finite
 * output; anything else takes the frame as the identity. */
static inline int dfn_hop_nn_end(DfnHop* h, int model_rc) {
    if (!h) return DFN_HOP_INVALID;
    if (h->phase != 2) return DFN_HOP_SEQUENCE_ERROR;
    h->phase = 1;
    if (h->bridge) {
        if (dfn_rate_bridge_nn_end(h->bridge, model_rc) != 0) return DFN_HOP_INVALID;
    } else {
        if (dfn_res_stage_end(h->stage, model_rc, h->stage_out) < 0) return DFN_HOP_INVALID;
        h->stage_ended = 1;
    }
    return 0;
}

/* Emits one host hop into out. A frame left open or never fetched is the identity. */
static inline int dfn_hop_end(DfnHop* h, float* out) {
    int k;
    if (!h || !out) return DFN_HOP_INVALID;
    if (h->phase == 0) return DFN_HOP_SEQUENCE_ERROR;
    if (h->bridge) {
        h->phase = 0;
        return dfn_rate_bridge_finish(h->bridge, out) != 0 ? DFN_HOP_INVALID : 0;
    }
    if (!h->stage_ended) {
        int open = h->job_pending || h->phase == 2;
        if (dfn_res_stage_end(h->stage, open ? -1 : 0, h->stage_out) < 0) {
            h->phase = 0;
            return DFN_HOP_INVALID;
        }
    }
    h->phase = 0;
    h->job_pending = 0;
    if (h->check_finite) {
        for (k = 0; k < DFN2_N_BINS; ++k)
            if (!isfinite(h->stage_out[k].r) || !isfinite(h->stage_out[k].i))
                return DFN_HOP_INVALID;
    }
    (void)ac_istft_push(h->istft, h->stage_out, out);
    if (h->check_finite) {
        for (k = 0; k < DFN2_HOP_LEN; ++k)
            if (!isfinite(out[k])) return DFN_HOP_INVALID;
    }
    return 0;
}

/* The whole hop with the stage's configured model callback (identity when unset). */
static inline int dfn_hop_process(DfnHop* h, const Complex* e, const Complex* p, float* out) {
    DfnResStageJob job;
    int r;
    if (!h || !out) return DFN_HOP_INVALID;
    r = dfn_hop_begin(h, e, p);
    if (r != 0) return r;
    while ((r = dfn_hop_nn_begin(h, &job)) == 1) {
        int nr = dfn_hop_nn_end(h, dfn_res_stage_infer(h->stage, &job));
        if (nr != 0) return nr;
    }
    if (r < 0) return r;
    return dfn_hop_end(h, out);
}

/* The stage, for set_atten_lim and counters. */
static inline DfnResStage* dfn_hop_stage(const DfnHop* h) {
    return h ? h->stage : NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_SAMPLES_DFN_HOP_H */
