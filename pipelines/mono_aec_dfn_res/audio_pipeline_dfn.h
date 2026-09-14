/* Mono AEC -> RES -> DeepFilterNet2 pipeline.
 *
 * The conventional mono pipeline (audio_pipeline.h) hosted on the product's
 * own grid (8, 16 or 48 kHz; 16 kHz by default) with its MMSE-LSA denoiser
 * disabled, and DFN2 inserted between its post-RES spectrum and its
 * synthesis. Per hop:
 *
 *   audio_pipeline_process_post_view()   E (linear-AEC error), P = E*G_res
 *                                        (+ comfort noise when enabled)
 *   48 kHz:  dfn_res_stage_process(E, P) DFN2 estimated on E, applied on P,
 *            audio_pipeline_synthesize_external()   the host's own WOLA
 *   8/16 kHz: dfn_rate_bridge_process(E, P)  the same stage on its 48 kHz
 *            grid: native WOLA of E and P, resample up, 48 kHz analysis,
 *            stage, 48 kHz WOLA, resample down (dfn_rate_bridge.h)
 *
 * Only the DFN2 stage runs at 48 kHz; the AEC, RES and comfort noise stay
 * at the product rate. With no model (or a failing one) the output is the
 * conventional pipeline with enable_nr=0 delayed: bit for bit by two hops
 * at 48 kHz, and through the resampler round trip by
 * dfn_rate_bridge_added_delay_samples() otherwise (16 kHz, hop 128: 672).
 *
 * Link with libaudio_pipeline.a (the host), libaec and libaudio_common; the
 * model callback stays on the board side.
 */
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
#define MONO_AEC_DFN_RES_LAYOUT_VERSION 1u

/* `host` is the conventional pipeline's own config, validated by it. This
 * wrapper only pins the invariants the seam needs: sample_rate 8000, 16000
 * or 48000 with the host's own fft choices, aec_only=0, enable_nr=0,
 * enable_res=1. Its enable_cng is the host's comfort-noise fill, off in the
 * default config (dfn_res_stage.h explains); it stays a switch. */
typedef struct MonoAecDfnResConfig {
    AudioPipelineConfig host;
    float atten_lim_db;            /* DFN2 attenuation limit; 0 = unlimited */
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
 * five-filter bank) with NR off, RES on, comfort noise off; no attenuation
 * limit, no matrices, identity model. */
MonoAecDfnResConfig mono_aec_dfn_res_default_config(void);

/* -1 on NULL, a host config outside the invariants above or refused by the
 * host itself, or any stage rejection (see dfn_res_stage.h). The descriptor
 * folds the host's and the stage's layout/hash, so a carve change in either
 * reaches this pipeline's descriptor. */
int mono_aec_dfn_res_get_mem_requirements(const MonoAecDfnResConfig *cfg,
                                          MonoAecDfnResMemReq *out);
MonoAecDfnRes *mono_aec_dfn_res_init(void *mem, size_t bytes,
                                     const MonoAecDfnResConfig *cfg);
/* init plus the 8-point stale-pool gate against `expected`. */
MonoAecDfnRes *mono_aec_dfn_res_init_ex(void *mem, size_t bytes,
                                        const MonoAecDfnResConfig *cfg,
                                        const MonoAecDfnResMemReq *expected);
MonoAecDfnRes *mono_aec_dfn_res_create(const MonoAecDfnResConfig *cfg);
void mono_aec_dfn_res_destroy(MonoAecDfnRes *p);
/* Host, stage and model together. Whole-pipeline resets only: an AEC delay
 * change does not reset the stage (dfn_res_stage.h, RESET). */
void mono_aec_dfn_res_reset(MonoAecDfnRes *p);

/* Consume and produce exactly one host hop (mono_aec_dfn_res_hop_size()).
 * The first hops of output are zero: two hops at 48 kHz (the model
 * lookahead), the bridge's added delay otherwise. -1 on NULL arguments or a
 * stage contract error (the pipeline is then reset). */
int mono_aec_dfn_res_process(MonoAecDfnRes *p, const float *mic,
                             const float *ref, float *out);

/* Runtime strength: the host's AEC preset ramp and the stage's attenuation
 * limit. Between hops only; -1 on NULL or a refused value. */
int mono_aec_dfn_res_set_aec_preset(MonoAecDfnRes *p, AecPreset preset,
                                    float ramp_ms);
int mono_aec_dfn_res_set_atten_lim(MonoAecDfnRes *p, float atten_lim_db);

int mono_aec_dfn_res_hop_size(const MonoAecDfnRes *p);
/* Total algorithmic latency in host samples, or -1: the host's WOLA hop plus
 * two model hops at 48 kHz (1536 = 32 ms); the host's WOLA hop plus the
 * bridge's added delay otherwise (16 kHz, hop 128: 800 = 50 ms). */
int mono_aec_dfn_res_lookahead_samples(const MonoAecDfnRes *p);
/* Read-only diagnostics (audio_pipeline_get_last_timing,
 * aec_get_res_context, dfn_res_stage_get_counters ...). Do not reset or
 * destroy through these. */
AudioPipeline *mono_aec_dfn_res_get_host(const MonoAecDfnRes *p);
Aec *mono_aec_dfn_res_get_aec(const MonoAecDfnRes *p);
DfnResStage *mono_aec_dfn_res_get_stage(const MonoAecDfnRes *p);
/* The rate bridge, or NULL at 48 kHz where the stage is driven directly. */
DfnRateBridge *mono_aec_dfn_res_get_bridge(const MonoAecDfnRes *p);

#ifdef __cplusplus
}
#endif
#endif
