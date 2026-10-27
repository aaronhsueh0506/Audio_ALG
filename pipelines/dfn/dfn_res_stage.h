/* Post-RES DeepFilterNet2 stage shared by the mono and 4-channel "AEC + RES + DFN2" pipelines. One hop in, one spectrum out: A = E * 2^-5 (E: pre-RES spectrum the RES gain was estimated from), B = P * 2^-5 (P: post-RES spectrum, E * G_res + comfort noise), heads = DFN2(estimated on A), Y = compose(heads, applied on B), out = Y * 2^5. The model estimates from the pre-RES spectrum and applies to the post-RES one; with the comfort-noise fill off E and P share phase, so the deep-filter taps fitted on A apply to B (the hosting pipelines keep the fill off by default).
 * Timing: the emitted spectrum belongs to source frame t - L, L = DFN_RES_STAGE_LOOKAHEAD_FRAMES (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD, 2 for the shipped 1/1 pair); the first L hops return 0 with an all-zero spectrum, so the caller synthesises unconditionally.
 * Fail-open: no model (model.infer == NULL), a nonzero infer() result or an unwritten/non-finite output takes the frame as an identity through the cascade (clocks advance, recurrent state unchanged); the output is P delayed by L hops.
 * Reset: whole-pipeline resets only; an AEC delay change must not reset the stage. Grid: 48 kHz / 1024 / 512 only. Memory: static pool with the 32-byte descriptor; the DFN2 class pool and its hash are folded in; no heap after init, no stdio. */

#ifndef DFN_RES_STAGE_H
#define DFN_RES_STAGE_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_prepost.h"   /* DFN2Model, DFN2ModelIoDescriptor, the grid */
#include "fft_wrapper.h"    /* Complex */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN_RES_STAGE_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or
 * descriptor content, including the composed DFN2 pre/post pool. */
#define DFN_RES_STAGE_LAYOUT_VERSION     3u
#define DFN_RES_STAGE_BACKEND_NE10       2u   /* id 1 is never returned */

/* Audio-scale spectra <-> DFN2 normalized=True spectra (exact powers of two). */
#define DFN_RES_STAGE_IN_SCALE  0x1p-5f
#define DFN_RES_STAGE_OUT_SCALE 0x1p5f

/* Frames the emitted spectrum trails the newest input by (the DFN2 cascade of the build's lookahead pair). */
#define DFN_RES_STAGE_LOOKAHEAD_FRAMES (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

typedef struct DfnResStageConfig {
    int          sample_rate;   /* 48000 only                                */
    int          fft_size;      /* 0 or 1024                                 */
    float        atten_lim_db;  /* DFN2 attenuation limit; 0 = unlimited     */
    const float *erb_fwd;       /* borrowed, bin-major [DFN2_N_BINS][DFN2_N_ERB] */
    const float *erb_inv;       /* borrowed, band-major, partition of unity  */
    DFN2Model    model;         /* by value; all-zero = identity             */
} DfnResStageConfig;

typedef struct {
    uint32_t descriptor_version;  /* DFN_RES_STAGE_DESCRIPTOR_VERSION         */
    uint32_t layout_version;      /* DFN_RES_STAGE_LAYOUT_VERSION             */
    uint32_t backend_id;          /* DFN_RES_STAGE_BACKEND_NE10               */
    uint32_t build_flags_hash;    /* FNV-1a-32, folds the DFN2 class hash     */
    uint32_t alignment;           /* 16                                       */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DfnResStageMemReq;

_Static_assert(sizeof(DfnResStageMemReq) == 32,
               "DfnResStageMemReq must be exactly 32 bytes");

typedef struct DfnResStage DfnResStage;

/* Defaults: 48 kHz, fft 1024, no attenuation limit, no matrices, identity model. */
DfnResStageConfig dfn_res_stage_default_config(int sample_rate);

/* Returns -1 on NULL, a rate other than 48000, an fft_size other than 0/1024, a missing ERB matrix, an erb_inv whose unit band mask does not expand to exactly 1.0f in every bin, a non-finite attenuation limit, a model that infers without a valid descriptor (dfn2_model_io_descriptor_validate()), or a build outside pipelines/Makefile. */
int dfn_res_stage_get_mem_requirements(const DfnResStageConfig *cfg,
                                       DfnResStageMemReq *out);

/* Constructs inside caller memory (16-byte aligned, >= req.bytes), applying every
 * get_mem_requirements() rejection; init_ex also checks the stale-pool descriptor
 * against `expected`. Starts reset. */
DfnResStage *dfn_res_stage_init(void *mem, size_t bytes,
                                const DfnResStageConfig *cfg);
DfnResStage *dfn_res_stage_init_ex(void *mem, size_t bytes,
                                   const DfnResStageConfig *cfg,
                                   const DfnResStageMemReq *expected);
DfnResStage *dfn_res_stage_create(const DfnResStageConfig *cfg);
void dfn_res_stage_destroy(DfnResStage *s);   /* frees only create()'s heap */

/* Resets the DFN2 class and the hop counters; calls model.reset when present.
 * Whole-pipeline resets only. */
void dfn_res_stage_reset(DfnResStage *s);

/* Between hops only; -1 on NULL or a non-finite value. */
int dfn_res_stage_set_atten_lim(DfnResStage *s, float atten_lim_db);

/* Source frame of the last emitted spectrum; -1 before the first emission. */
int dfn_res_stage_output_frame_index(const DfnResStage *s, long long *frame);

/* Hop counters: frames pushed, head evaluations committed, frames taken as identity. */
void dfn_res_stage_get_counters(const DfnResStage *s, long long *frames_in,
                                long long *commits, long long *skips);

/* The model's tensor views of one open frame (DFN2PrepostInputs/Outputs; layout in
 * dfn2_prepost.h). Valid until dfn_res_stage_end(). */
typedef struct DfnResStageJob {
    DFN2PrepostInputs  inputs;
    DFN2PrepostOutputs outputs;
} DfnResStageJob;

/* One hop in two calls around the model. `estimate_spec` is the pre-RES spectrum (E), `apply_spec` the post-RES spectrum (P); both are borrowed for the begin call only. begin returns 1 when the model has a frame to run (*job holds its views; run the model, then call end), 0 during the first DFN2_MASK_LOOKAHEAD hops (end still produces the output), -1 on a contract error (NULL or non-finite arguments, a hop already begun, a frame the DFN2 class refuses). end closes the hop: model_rc is the model's result (0 = every output written); anything else, or an unwritten or non-finite output, takes the frame as the identity (ignored when begin returned 0). `out_spec` is always fully written (zeros during warm-up). Returns 1 when out_spec carries an emitted frame, 0 during warm-up, -1 on a contract error. */
int dfn_res_stage_begin(DfnResStage *s,
                        const Complex *estimate_spec,
                        const Complex *apply_spec,
                        DfnResStageJob *job);
int dfn_res_stage_end(DfnResStage *s, int model_rc, Complex *out_spec);

/* Runs the configured model callback on an open frame: its result (0 = success), or
 * -1 when no callback is set. */
int dfn_res_stage_infer(const DfnResStage *s, DfnResStageJob *job);

/* begin -> configured model callback (identity when unset) -> end in one call. */
int dfn_res_stage_process(DfnResStage *s,
                          const Complex *estimate_spec,
                          const Complex *apply_spec,
                          Complex *out_spec);

#ifdef __cplusplus
}
#endif

#endif /* DFN_RES_STAGE_H */
