/* dfn_hop_pool.h -- example: the DeepFilterNet2 tail of an example program's pool, sized and
 * carved in one place: the stage and its 48 kHz synthesis and output spectrum, or the rate bridge
 * below 48 kHz. Header-only, no ABI: copy and adapt.
 *
 *     plan.stage_cfg = ...;                                   // ERB matrices, model, atten
 *     dfn_hop_pool_plan(&plan, host_rate, fft_size);          // sizes
 *     total = <front end and P buffers> + dfn_hop_pool_bytes(&plan);
 *     dfn_hop_pool_build(&plan, cur, n_freqs, hop, check_finite, &hop_driver, &stage, &bridge, &istft);
 *
 * The tail follows the caller's front end and P buffers in the pool; its order is the stage (or
 * bridge), the synthesis, then the stage output spectrum. */
#ifndef PIPELINES_DFN_DFN_HOP_POOL_H
#define PIPELINES_DFN_DFN_HOP_POOL_H

#include <stddef.h>
#include <string.h>

#include "audio_stft.h"
#include "dfn_hop.h"
#include "dfn_rate_bridge.h"
#include "dfn_res_stage.h"
#include "utility/mem_align.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DfnHopPoolPlan {
    DfnResStageConfig stage_cfg;      /* filled by the caller before dfn_hop_pool_plan() */
    DfnRateBridgeConfig bridge_cfg;
    int use_bridge;                   /* host below 48 kHz */
    size_t stage_bytes, istft_bytes;
} DfnHopPoolPlan;

/* Sizes the tail for a host at `host_rate`; `fft_size` is the host grid (0 = the rate's default;
 * at 48 kHz only the stage's own grid). 0, or -1 for a configuration no stage accepts. */
static inline int dfn_hop_pool_plan(DfnHopPoolPlan* p, int host_rate, int fft_size) {
    DfnResStageMemReq stage_req;
    DfnRateBridgeMemReq bridge_req;
    AcStftConfig scfg;
    p->use_bridge = host_rate != DFN2_SR;
    p->istft_bytes = 0;
    if (p->use_bridge) {
        p->bridge_cfg = dfn_rate_bridge_default_config(host_rate);
        p->bridge_cfg.fft_size = fft_size;
        p->bridge_cfg.stage = p->stage_cfg;
        if (dfn_rate_bridge_get_mem_requirements(&p->bridge_cfg, &bridge_req) != 0) return -1;
        p->stage_bytes = (size_t)bridge_req.bytes;
    } else {
        if (fft_size != 0 && fft_size != DFN2_N_FFT) return -1;
        if (dfn_res_stage_get_mem_requirements(&p->stage_cfg, &stage_req) != 0) return -1;
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        p->istft_bytes = ac_istft_get_mem_size(&scfg);
        if (p->istft_bytes == 0) return -1;
        p->stage_bytes = (size_t)stage_req.bytes;
    }
    return 0;
}

/* Bytes of the tail, each part 16-byte aligned. */
static inline size_t dfn_hop_pool_bytes(const DfnHopPoolPlan* p) {
    return ALIGN16(p->stage_bytes) + ALIGN16(p->istft_bytes) +
           (p->use_bridge ? 0 : ALIGN16((size_t)DFN2_N_BINS * sizeof(Complex)));
}

/* Carves the tail at `cur` and wires `hop`; the front end's grid must be the bridge's, or the
 * stage's at 48 kHz. `check_finite` applies to the stage path. 0, or -1 when an init is refused:
 * *stage / *bridge then hold whatever the caller must destroy. */
static inline int dfn_hop_pool_build(const DfnHopPoolPlan* p, unsigned char* cur, int n_freqs,
                                     int hop_size, int check_finite, DfnHop* hop,
                                     DfnResStage** stage, DfnRateBridge** bridge,
                                     AcIstft** istft) {
    if (p->use_bridge) {
        *bridge = dfn_rate_bridge_init(cur, p->stage_bytes, &p->bridge_cfg);
        if (!*bridge || dfn_rate_bridge_n_freqs(*bridge) != n_freqs ||
            dfn_rate_bridge_hop_size(*bridge) != hop_size)
            return -1;
        *stage = dfn_rate_bridge_stage(*bridge);
        dfn_hop_init_bridge(hop, *bridge);
    } else {
        AcStftConfig scfg;
        Complex* stage_out;
        if (n_freqs != DFN2_N_BINS || hop_size != DFN2_HOP_LEN) return -1;
        *stage = dfn_res_stage_init(cur, p->stage_bytes, &p->stage_cfg);
        cur += ALIGN16(p->stage_bytes);
        if (!*stage) return -1;
        ac_stft_config_default(&scfg, DFN2_N_FFT);
        *istft = ac_istft_init(cur, p->istft_bytes, &scfg);
        cur += ALIGN16(p->istft_bytes);
        if (!*istft) return -1;
        stage_out = (Complex*)cur;
        memset(stage_out, 0, (size_t)DFN2_N_BINS * sizeof(Complex));
        dfn_hop_init_stage(hop, *stage, *istft, stage_out);
        hop->check_finite = check_finite;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_DFN_DFN_HOP_POOL_H */
