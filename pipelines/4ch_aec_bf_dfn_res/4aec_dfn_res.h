/* Four-channel AEC -> external beamformer (bf) -> RES -> DFN2 pipeline.
 * The four-channel core on the product grid (16 or 48 kHz; 16 kHz default) with
 * its post-beam denoiser replaced by DFN2. front_process() exposes the four linear
 * lanes; process() takes the effective beamformer weights and the optional trusted
 * BF spectrum, fuses the lanes into the beamformed error E and forms
 * P = E * G_res (+ comfort noise when enabled). At 48 kHz the shared
 * dfn_res_stage is applied to P directly; at 16 kHz dfn_rate_bridge runs the same
 * stage on its 48 kHz grid, so only DFN2 leaves the product rate. With no model
 * (or a failing one) the output is the core chain with the denoiser off, delayed
 * by two hops at 48 kHz and by dfn_rate_bridge_added_delay_samples() otherwise. */
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
/* Carve layout version; bumped on any change of pool size, carve order or
 * descriptor content, including the composed core or stage pool. */
#define FOUR_AEC_DFN_RES_LAYOUT_VERSION 6u

/* front_end must carry enable_post=1, enable_res=1, enable_nr=0. Its enable_cng is
 * the core's comfort-noise fill (off in the default config). */
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

/* One hop around the external SRP-PHAT/BF. front_process() is four_aec_front_process() on the front end (it exposes the four linear lanes, FourAecFront.linear_spectra). The beamformer then supplies the effective weights and optionally the beamformed error (NULL reconstructs it); process() fuses the lanes (four_aec_fuse()), applies RES gain -> min -> comfort noise, runs the DFN2 stage on (E, P) and synthesises one hop. Status codes are FOUR_AEC_NR_RES_*; a stage contract error resets the pipeline and returns DSP_ERROR.
 * Phased use, for a caller that runs the model itself:
 *     four_aec_dfn_res_process_begin(p, weights, error);   // fuse, RES, P, stage input
 *     while (four_aec_dfn_res_nn_begin(p, &job) > 0) {     // a model frame is open
 *         rc = run_npu(&job);                              // 0 = every output written
 *         four_aec_dfn_res_nn_end(p, rc);                  // commit, or the identity
 *     }
 *     four_aec_dfn_res_process_end(p, out);                // synthesis, one hop
 * A hop carries at most one model frame at 48 kHz and two through the bridge. A frame left open or never fetched takes the identity. nn_begin returns 1 (a frame is open), 0 (none left) or a negative status. */
int four_aec_dfn_res_front_process(FourAecDfnRes *p,
                                   const float *microphones_interleaved,
                                   const float *ref,
                                   FourAecFront *out);
int four_aec_dfn_res_process_begin(FourAecDfnRes *p, const Complex *weights,
                                   const Complex *beamformed_error);
int four_aec_dfn_res_nn_begin(FourAecDfnRes *p, DfnResStageJob *job);
int four_aec_dfn_res_nn_end(FourAecDfnRes *p, int model_rc);
int four_aec_dfn_res_process_end(FourAecDfnRes *p, float *out);
int four_aec_dfn_res_process(FourAecDfnRes *p, const Complex *weights,
                             const Complex *beamformed_error, float *out);

/* Runtime strength: the core's AEC preset ramp and the stage's attenuation limit.
 * Between hops only; -1 on NULL or a refused value. */
int four_aec_dfn_res_set_aec_preset(FourAecDfnRes *p, AecPreset preset,
                                    float ramp_ms);
int four_aec_dfn_res_set_atten_lim(FourAecDfnRes *p, float atten_lim_db);

int four_aec_dfn_res_hop_size(const FourAecDfnRes *p);
/* Total algorithmic latency in core samples, or -1: core WOLA hop plus two model
 * hops at 48 kHz (1536); core WOLA hop plus the bridge's added delay otherwise
 * (16 kHz, hop 128: 800). */
int four_aec_dfn_res_lookahead_samples(const FourAecDfnRes *p);
FourAecNrRes *four_aec_dfn_res_get_front_end(const FourAecDfnRes *p);
DfnResStage *four_aec_dfn_res_get_stage(const FourAecDfnRes *p);
/* The rate bridge, or NULL at 48 kHz where the stage is driven directly. */
DfnRateBridge *four_aec_dfn_res_get_bridge(const FourAecDfnRes *p);

#ifdef __cplusplus
}
#endif
#endif
