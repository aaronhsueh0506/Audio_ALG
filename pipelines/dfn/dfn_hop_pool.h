/* dfn_hop_pool.h -- example: the DeepFilterNet2 tail of an example program's pool, sized and
 * carved in one place: the model stream (stream/ainr/dfn2, freq_only), the audio-scale
 * conversion buffers, and the synthesis with the output spectrum when the host runs the model's
 * grid, or the rate bridge (stream/ainr/bridge) when the host rate differs (the model is then on
 * its 48 kHz grid). Header-only, no ABI: copy and adapt.
 *
 *     dfn_hop_pool_plan_init(&plan, &model_grid);             // zeroed plan on the model grid
 *     plan.stream_cfg.atten_lim_db = ...;                     // ERB matrices, model, attenuation limit
 *     plan.stream_cfg.erb_fwd = ...;  plan.stream_cfg.erb_inv = ...;  plan.model = ...;
 *     dfn_hop_pool_plan(&plan, host_rate, fft_size);          // sizes
 *     total = <front end and P buffers> + dfn_hop_pool_bytes(&plan);
 *     dfn_hop_pool_build(&plan, cur, n_freqs, hop, &hop_driver);
 *
 * The tail follows the caller's front end and P buffers in the pool; its order is the stream, the
 * conversion buffers, then the bridge, or the synthesis and the output spectrum. Nothing in it
 * allocates: the pool is released by freeing the caller's block. */
#ifndef PIPELINES_DFN_DFN_HOP_POOL_H
#define PIPELINES_DFN_DFN_HOP_POOL_H

#include <stddef.h>
#include <string.h>

#include "ainr_bridge.h"
#include "audio_stft.h"
#include "dfn2_stream.h"
#include "dfn_hop.h"
#include "utility/mem_align.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DfnHopPoolPlan {
    DFN2StreamConfig stream_cfg;      /* model grid, attenuation limit, ERB matrices (borrowed) */
    DFN2Model model;                  /* the callback, its reset and its descriptor */
    AinrBridgeConfig bridge_cfg;
    int use_bridge;                   /* host rate differs from the model grid's (stream_cfg.grid) */
    DFN2StreamMemReq stream_req;
    AinrBridgeMemReq bridge_req;
    size_t stream_bytes, planar_bytes, bridge_bytes, istft_bytes, spec_bytes;
} DfnHopPoolPlan;

/* An empty plan on `grid`: no attenuation limit, no matrices, no model (the identity). */
static inline void dfn_hop_pool_plan_init(DfnHopPoolPlan* p, const DFN2Grid* grid) {
    memset(p, 0, sizeof(*p));
    p->stream_cfg.grid = *grid;
}

/* Sizes the tail for a host at `host_rate`; `fft_size` is the host grid (0 = the rate's default;
 * on the model's own rate only the model's own n_fft). 0, or -1 for a configuration nothing
 * accepts: more than one frame per call, a model its descriptor does not fit, a missing matrix, a
 * non-finite attenuation limit, a bridge for a model grid other than 48 kHz / 1024 or for a host
 * grid the bridge does not offer. The matrices' content is checked by dfn_hop_pool_build(). */
static inline int dfn_hop_pool_plan(DfnHopPoolPlan* p, int host_rate, int fft_size) {
    AcStftConfig scfg;
    const DFN2Grid* grid = &p->stream_cfg.grid;
    p->use_bridge = host_rate != grid->sample_rate;
    p->istft_bytes = 0;
    p->bridge_bytes = 0;
    if (grid->frames_per_call != 1 || !p->stream_cfg.erb_fwd || !p->stream_cfg.erb_inv ||
        !dfn_hop_model_valid(&p->model, grid))
        return -1;
    p->stream_cfg.freq_only = 1;   /* the hop drives the _freq entries only */
    if (dfn2_stream_get_mem_size(&p->stream_cfg, &p->stream_req) != 0) return -1;
    p->stream_bytes = (size_t)p->stream_req.bytes;
    p->planar_bytes = (size_t)6 * (size_t)grid->n_bins * sizeof(float);
    p->spec_bytes = 0;
    if (p->use_bridge) {
        if (grid->sample_rate != AINR_BRIDGE_MODEL_RATE || grid->n_fft != AINR_BRIDGE_MODEL_FFT) return -1;
        p->bridge_cfg.sample_rate = host_rate;
        p->bridge_cfg.fft_size = fft_size;
        p->bridge_cfg.lookahead_frames = dfn_hop_lookahead_frames(grid);
        if (ainr_bridge_get_mem_requirements(&p->bridge_cfg, &p->bridge_req) != 0) return -1;
        p->bridge_bytes = (size_t)p->bridge_req.bytes;
    } else {
        if (fft_size != 0 && fft_size != grid->n_fft) return -1;
        ac_stft_config_default(&scfg, grid->n_fft);
        p->istft_bytes = ac_istft_get_mem_size(&scfg);
        if (p->istft_bytes == 0) return -1;
        p->spec_bytes = (size_t)grid->n_bins * sizeof(Complex);
    }
    return 0;
}

/* Bytes of the tail, each part 16-byte aligned. */
static inline size_t dfn_hop_pool_bytes(const DfnHopPoolPlan* p) {
    return ALIGN16(p->stream_bytes) + ALIGN16(p->planar_bytes) + ALIGN16(p->bridge_bytes) +
           ALIGN16(p->istft_bytes) + ALIGN16(p->spec_bytes);
}

/* Carves the tail at `cur` and wires `hop`; the front end's grid must be the bridge's, or the
 * model's when the host runs the model's grid. 0, or -1 when an init is refused: a pool or grid
 * that does not match the plan, or ERB matrices that fail dfn2_erb_tables_valid() (dfn2_process.h). */
static inline int dfn_hop_pool_build(const DfnHopPoolPlan* p, unsigned char* cur, int n_freqs,
                                     int hop_size, DfnHop* hop) {
    const DFN2Grid* grid = &p->stream_cfg.grid;
    DFN2Stream* stream;
    float* planar;
    memset(hop, 0, sizeof(*hop));
    stream = dfn2_stream_init_ex(&p->stream_cfg, cur, p->stream_bytes, &p->stream_req);
    cur += ALIGN16(p->stream_bytes);
    if (!stream) return -1;
    planar = (float*)cur;
    cur += ALIGN16(p->planar_bytes);
    if (p->use_bridge) {
        AinrBridge* bridge = ainr_bridge_init_ex(cur, p->bridge_bytes, &p->bridge_cfg, &p->bridge_req);
        if (!bridge || ainr_bridge_n_freqs(bridge) != n_freqs ||
            ainr_bridge_hop_size(bridge) != hop_size)
            return -1;
        return dfn_hop_init_bridge(hop, stream, grid, &p->model, bridge, planar);
    } else {
        AcStftConfig scfg;
        AcIstft* istft;
        Complex* out_spec;
        if (n_freqs != grid->n_bins || hop_size != grid->hop) return -1;
        ac_stft_config_default(&scfg, grid->n_fft);   /* scales stay 1: the hop applies the grid's */
        istft = ac_istft_init(cur, p->istft_bytes, &scfg);
        cur += ALIGN16(p->istft_bytes);
        if (!istft) return -1;
        out_spec = (Complex*)cur;
        if (dfn_hop_init_direct(hop, stream, grid, &p->model, istft, out_spec, planar) != 0) return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_DFN_DFN_HOP_POOL_H */
