/* Mono AEC -> RES -> DeepFilterNet2 pipeline.
 * The conventional mono pipeline (audio_pipeline.h) on the product grid (8, 16 or
 * 48 kHz; 16 kHz default) with its denoiser disabled and DFN2 inserted between the
 * post-RES spectrum and synthesis. Per hop:
 *   audio_pipeline_process_spec()   E (linear-AEC error), G_res, comfort noise;
 *                                   the application forms P = E*G_res (+ comfort noise)
 *   48 kHz:    dfn_res_stage_process(E, P), then audio_pipeline_synthesize()
 *   8/16 kHz:  dfn_rate_bridge_process(E, P): the same stage on its 48 kHz grid
 *              (dfn_rate_bridge.h)
 * Only the DFN2 stage runs at 48 kHz. With no model (or a failing one) the output
 * is the conventional pipeline with enable_nr=0, delayed by two hops at 48 kHz and
 * by dfn_rate_bridge_added_delay_samples() otherwise (16 kHz, hop 128: 672).
 * Link with libaudio_pipeline.a (the host), libaec and libaudio_common; the model
 * callback stays on the board side. */
#ifndef MONO_AEC_DFN_RES_PIPELINE_H
#define MONO_AEC_DFN_RES_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "audio_pipeline.h"
#include "dfn_rate_bridge.h"
#include "dfn_res_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MONO_AEC_DFN_RES_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or
 * descriptor content, including the composed host, bridge or stage pool. */
#define MONO_AEC_DFN_RES_LAYOUT_VERSION 5u

/* `host` is the conventional pipeline's config, validated by it. This wrapper pins
 * sample_rate 8000/16000/48000, aec_only=0, enable_nr=0, enable_res=1. enable_cng
 * is the host's comfort-noise fill (off by default). */
typedef struct MonoAecDfnResConfig {
    AudioPipelineConfig host;
    float atten_lim_db;            /* DFN2 attenuation limit in dB; 0 = unlimited */
    const float *erb_fwd;
    const float *erb_inv;
    DFN2Model model;
} MonoAecDfnResConfig;

typedef struct MonoAecDfnResMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} MonoAecDfnResMemReq;

_Static_assert(sizeof(MonoAecDfnResMemReq) == 32,
               "MonoAecDfnResMemReq must be exactly 32 bytes");

typedef struct MonoAecDfnRes MonoAecDfnRes;

/* The host's 16 kHz defaults (fft 256, balanced AEC, matched delay with the
 * five-filter bank) with NR off, RES on, comfort noise off; no attenuation limit,
 * no matrices, identity model. */
MonoAecDfnResConfig mono_aec_dfn_res_default_config(void);

/* -1 on NULL, a host config outside the invariants above or refused by the host,
 * or any stage rejection (dfn_res_stage.h). The descriptor folds the host's and
 * the stage's layout/hash. */
int mono_aec_dfn_res_get_mem_requirements(const MonoAecDfnResConfig *cfg,
                                          MonoAecDfnResMemReq *out);
MonoAecDfnRes *mono_aec_dfn_res_init(void *mem, size_t bytes,
                                     const MonoAecDfnResConfig *cfg);
/* init plus the stale-pool descriptor gate against `expected`. */
MonoAecDfnRes *mono_aec_dfn_res_init_ex(void *mem, size_t bytes,
                                        const MonoAecDfnResConfig *cfg,
                                        const MonoAecDfnResMemReq *expected);
MonoAecDfnRes *mono_aec_dfn_res_create(const MonoAecDfnResConfig *cfg);
void mono_aec_dfn_res_destroy(MonoAecDfnRes *p);
/* Resets host, stage and model together. Whole-pipeline resets only. */
void mono_aec_dfn_res_reset(MonoAecDfnRes *p);

/* Consumes and produces exactly one host hop (mono_aec_dfn_res_hop_size()). The
 * first output hops are zero: two hops at 48 kHz, the bridge's added delay
 * otherwise. The model is the config's DFN2Model (all-zero = identity). Returns -1
 * on NULL arguments or a stage contract error (the pipeline is then reset).
 * Phased use, for a caller that runs the model itself:
 *     mono_aec_dfn_res_process_begin(p, mic, ref);     // AEC, RES, P, stage input
 *     while (mono_aec_dfn_res_nn_begin(p, &job) > 0) { // a model frame is open
 *         rc = run_npu(&job);                          // 0 = every output written
 *         mono_aec_dfn_res_nn_end(p, rc);              // commit, or the identity
 *     }
 *     mono_aec_dfn_res_process_end(p, out);            // synthesis, one host hop
 * A hop carries at most one model frame at 48 kHz and two through the bridge. A
 * frame left open or never fetched takes the identity in process_end(). All
 * functions return 0 (nn_begin: 1 when a frame is open, 0 when none is left) or
 * -1; any contract error resets the pipeline. */
int mono_aec_dfn_res_process_begin(MonoAecDfnRes *p, const float *mic,
                                   const float *ref);
int mono_aec_dfn_res_nn_begin(MonoAecDfnRes *p, DfnResStageJob *job);
int mono_aec_dfn_res_nn_end(MonoAecDfnRes *p, int model_rc);
int mono_aec_dfn_res_process_end(MonoAecDfnRes *p, float *out);
int mono_aec_dfn_res_process(MonoAecDfnRes *p, const float *mic,
                             const float *ref, float *out);

/* Runtime strength: the host's AEC preset ramp and the stage's attenuation limit.
 * Between hops only; -1 on NULL or a refused value. */
int mono_aec_dfn_res_set_aec_preset(MonoAecDfnRes *p, AecPreset preset,
                                    float ramp_ms);
int mono_aec_dfn_res_set_atten_lim(MonoAecDfnRes *p, float atten_lim_db);

int mono_aec_dfn_res_hop_size(const MonoAecDfnRes *p);
/* Total algorithmic latency in host samples, or -1: host WOLA hop plus two model
 * hops at 48 kHz (1536 = 32 ms); host WOLA hop plus the bridge's added delay
 * otherwise (16 kHz, hop 128: 800 = 50 ms). */
int mono_aec_dfn_res_lookahead_samples(const MonoAecDfnRes *p);
/* Read-only diagnostics access; do not reset or destroy through these. */
AudioPipeline *mono_aec_dfn_res_get_host(const MonoAecDfnRes *p);
Aec *mono_aec_dfn_res_get_aec(const MonoAecDfnRes *p);
DfnResStage *mono_aec_dfn_res_get_stage(const MonoAecDfnRes *p);
/* The rate bridge, or NULL at 48 kHz where the stage is driven directly. */
DfnRateBridge *mono_aec_dfn_res_get_bridge(const MonoAecDfnRes *p);

#ifdef __cplusplus
}
#endif
#endif
