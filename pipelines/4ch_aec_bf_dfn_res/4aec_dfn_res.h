/* Four-channel AEC -> external BF/GSC -> RES -> DFN2 pipeline.
 *
 * The conventional four-channel core on the product's own grid (16 or
 * 48 kHz, the core's rates; 16 kHz by default) with its post-beam MMSE-LSA
 * replaced by DFN2.
 * The external beamformer seam is identical to FourAecNrRes: process_pre()
 * exposes the four linear lanes, then process_post() accepts the effective
 * weights and the optional trusted GSC spectrum. The core's
 * process_post_view() then yields the beamformed error E and the post-RES
 * spectrum P = E * G_res (+ comfort noise when enabled). At 48 kHz the
 * shared dfn_res_stage is applied to P directly and the core's
 * synthesize_external() finishes the hop; at 16 kHz the dfn_rate_bridge
 * runs the same stage on its 48 kHz grid (native WOLA, resample up, 48 kHz
 * analysis, stage, 48 kHz WOLA, resample down) so only DFN2 leaves the
 * product rate. With no model (or a failing one) the output is the
 * conventional core with enable_nr=0 delayed: bit for bit by two hops at
 * 48 kHz, and through the resampler round trip by
 * dfn_rate_bridge_added_delay_samples() otherwise.
 */
#ifndef FOUR_AEC_DFN_RES_H
#define FOUR_AEC_DFN_RES_H

#include <stddef.h>
#include <stdint.h>

#include "4aec_nr_res.h"
#include "dfn_rate_bridge.h"
#include "dfn_res_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FOUR_AEC_DFN_RES_DESCRIPTOR_VERSION 1u
#define FOUR_AEC_DFN_RES_LAYOUT_VERSION 1u

/* front_end must carry enable_post=1, enable_res=1, enable_nr=0; its
 * enable_cng is the core's own comfort-noise fill, which lands in the
 * post-RES spectrum the stage is applied to. The default config turns it
 * off for this variant (dfn_res_stage.h explains); it stays a switch. */
typedef struct FourAecDfnResConfig {
    FourAecNrResConfig front_end;
    float atten_lim_db;
    const float *erb_fwd;
    const float *erb_inv;
    DFN2Model model;
} FourAecDfnResConfig;

typedef struct FourAecDfnResMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} FourAecDfnResMemReq;

_Static_assert(sizeof(FourAecDfnResMemReq) == 32,
               "FourAecDfnResMemReq must be exactly 32 bytes");

typedef struct FourAecDfnRes FourAecDfnRes;

FourAecDfnResConfig four_aec_dfn_res_default_config(void);
int four_aec_dfn_res_get_mem_requirements(const FourAecDfnResConfig *cfg,
                                          FourAecDfnResMemReq *out);
FourAecDfnRes *four_aec_dfn_res_init(void *mem, size_t bytes,
                                     const FourAecDfnResConfig *cfg);
FourAecDfnRes *four_aec_dfn_res_init_ex(void *mem, size_t bytes,
                                        const FourAecDfnResConfig *cfg,
                                        const FourAecDfnResMemReq *expected);
FourAecDfnRes *four_aec_dfn_res_create(const FourAecDfnResConfig *cfg);
void four_aec_dfn_res_destroy(FourAecDfnRes *p);
void four_aec_dfn_res_reset(FourAecDfnRes *p);

int four_aec_dfn_res_process_pre(FourAecDfnRes *p,
                                 const float *microphones_interleaved,
                                 const float *ref,
                                 FourAecNrResPreFrame *out);
int four_aec_dfn_res_process_post(FourAecDfnRes *p,
                                  const FourAecNrResFrameToken *token,
                                  const Complex *weights,
                                  float *out);
int four_aec_dfn_res_process_post_trusted_spectrum(
    FourAecDfnRes *p,
    const FourAecNrResFrameToken *token,
    const Complex *weights,
    const Complex *beamformed_error,
    float *out);

/* Runtime strength: the core's AEC preset ramp and the stage's attenuation
 * limit. Between hops only; -1 on NULL or a refused value. */
int four_aec_dfn_res_set_aec_preset(FourAecDfnRes *p, AecPreset preset,
                                    float ramp_ms);
int four_aec_dfn_res_set_atten_lim(FourAecDfnRes *p, float atten_lim_db);

int four_aec_dfn_res_hop_size(const FourAecDfnRes *p);
/* Total algorithmic latency in core samples, or -1: the core's WOLA hop plus
 * two model hops at 48 kHz (1536); the core's WOLA hop plus the bridge's
 * added delay otherwise (16 kHz, hop 128: 800). */
int four_aec_dfn_res_lookahead_samples(const FourAecDfnRes *p);
FourAecNrRes *four_aec_dfn_res_get_front_end(const FourAecDfnRes *p);
DfnResStage *four_aec_dfn_res_get_stage(const FourAecDfnRes *p);
/* The rate bridge, or NULL at 48 kHz where the stage is driven directly. */
DfnRateBridge *four_aec_dfn_res_get_bridge(const FourAecDfnRes *p);

#ifdef __cplusplus
}
#endif
#endif
