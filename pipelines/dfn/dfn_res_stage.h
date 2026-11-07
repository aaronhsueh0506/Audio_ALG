/* Post-RES DeepFilterNet2 stage shared by the mono and 4-channel "AEC + RES + DFN2" pipelines. One
 * hop in, one spectrum out: A = E * grid.stft_scale (E: pre-RES spectrum the RES gain was estimated
 * from), B = P * grid.stft_scale (P: post-RES spectrum, E * G_res + comfort noise), heads =
 * DFN2(estimated on A), Y = heads applied on B, out = Y * grid.stft_unscale. The model estimates from
 * the pre-RES spectrum and applies to the post-RES one; with the comfort-noise fill off E and P share
 * phase, so the deep-filter taps fitted on A apply to B (the hosting pipelines keep the fill off by
 * default).
 *
 * Timing: the emitted spectrum belongs to source frame t - L, L = dfn_res_stage_lookahead_frames()
 * (grid.mask_lookahead + grid.df_lookahead, 2 for the shipped 1/1 pair); the first L hops return 0
 * with an all-zero spectrum, so the caller synthesises unconditionally. Every hop runs the same
 * begin, model, end flow: the first grid.mask_lookahead hops are warm-up frames (the model runs, its
 * heads are discarded, its state restarts), so the audio equals a stream whose model starts at its
 * first real frame.
 *
 * Fail-open: no model (model.infer == NULL), a nonzero infer() result, or an unwritten/non-finite
 * output takes the frame as an identity through the cascade (clocks advance either way); the
 * output is P delayed by L hops. Recurrent state is unchanged only for no-model and a nonzero
 * infer() result (frame_bypass(), never reached for heads at all). An unwritten/non-finite output
 * is different: that is a refused commit (frame_finish()'s own finite check), which zeroes the
 * registered state and advances dfn_res_stage_state_generation() by one, the same as a warm-up
 * frame -- see that function's own contract below.
 *
 * Reset: whole-pipeline resets only; an AEC delay change must not reset the stage. Grid: the
 * DFN2Grid in the config (dfn2_grid_make(); the shipped one is 48 kHz / 1024 / 512, hop 512, 513
 * bins) with one frame per call, handed to the library as it is; a model built for another grid or
 * lookahead pair is refused by its descriptor at run time. Memory: static pool with the 32-byte
 * descriptor; the DFN2 class pool and its hash are folded in; no heap after init, no stdio. */

#ifndef DFN_RES_STAGE_H
#define DFN_RES_STAGE_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_prepost.h"   /* DFN2Grid, DFN2Model, DFN2ModelIoDescriptor */
#include "fft_wrapper.h"    /* Complex */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN_RES_STAGE_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or
 * descriptor content, the DFN2 pre/post pool included. */
#define DFN_RES_STAGE_LAYOUT_VERSION     5u
#define DFN_RES_STAGE_BACKEND_NE10       2u   /* id 1 is never returned */

/* Frames the emitted spectrum trails the newest input by: the DFN2 cascade of the grid's
 * lookahead pair (the stage runs one frame per call). The audio-scale spectra the stage takes
 * and returns are scaled to and from DFN2's normalized=True spectra by the grid's stft_scale
 * (1/sqrt(n_fft)) and stft_unscale (sqrt(n_fft)): exact powers of two where n_fft is a power of
 * four (256, 1024, 4096), correctly rounded floats elsewhere (512). The caller's AcStft and
 * AcIstft keep audiolib's default spec_scale / ispec_scale of 1: the stage applies the grid's. */
static inline int dfn_res_stage_lookahead_frames(const DFN2Grid *grid)
{
    return grid ? grid->mask_lookahead + grid->df_lookahead : 0;
}

typedef struct DfnResStageConfig {
    DFN2Grid     grid;          /* the model's grid (dfn2_grid_make), copied at init */
    float        atten_lim_db;  /* DFN2 attenuation limit; 0 = unlimited     */
    const float *erb_fwd;       /* borrowed, bin-major [grid.n_bins][grid.n_erb] */
    const float *erb_inv;       /* borrowed, band-major [grid.n_erb][grid.n_bins], partition of unity */
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

/* Defaults for the model grid `grid` (copied; NULL leaves a zeroed grid, which the stage refuses): no
 * attenuation limit, no matrices, identity model. */
DfnResStageConfig dfn_res_stage_default_config(const DFN2Grid *grid);

/* Returns -1 on NULL, a grid the library refuses (dfn2_grid_validate(): a zeroed or hand-edited one
 * included), grid.frames_per_call other than 1, a missing ERB matrix, an erb_inv whose unit band mask
 * does not expand to exactly 1.0f in every bin, a non-finite attenuation limit, a model that infers
 * without a descriptor valid for the config's grid (dfn2_model_io_descriptor_validate(): another
 * grid, lookahead pair or frame count is refused here), or a build outside pipelines/Makefile. */
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

/* The grid the stage runs (the config's, as the library holds it); NULL for NULL. */
const DFN2Grid *dfn_res_stage_grid(const DfnResStage *s);

/* Generation of the model's recurrent state: 1 after init, plus one per reset, per warm-up frame and
 * per refused commit; 0 for NULL. The stage registers no state (it belongs to the model callback),
 * so a callback that holds it refills it with quantized zeros whenever this value changes, a warm-up
 * frame's included. */
unsigned int dfn_res_stage_state_generation(const DfnResStage *s);

/* Source frame of the last emitted spectrum; -1 before the first emission. */
int dfn_res_stage_output_frame_index(const DfnResStage *s, long long *frame);

/* Hop counters: frames pushed, head evaluations committed, frames taken as identity; the warm-up
 * frames count in frames_in only. */
void dfn_res_stage_get_counters(const DfnResStage *s, long long *frames_in,
                                long long *commits, long long *skips);

/* The model's tensor views of one open frame (DFN2PrepostInputs/Outputs; layout in
 * dfn2_prepost.h). Valid until dfn_res_stage_end(). */
typedef struct DfnResStageJob {
    DFN2PrepostInputs  inputs;
    DFN2PrepostOutputs outputs;
} DfnResStageJob;

/* One hop in two calls around the model. `estimate_spec` is the pre-RES spectrum (E), `apply_spec` the post-RES spectrum (P); both are borrowed for the begin call only. begin returns 1 with the open frame's views in *job (run the model, then call end), or -1 on a contract error (NULL or non-finite arguments, a hop already begun, a frame the DFN2 class refuses). end closes the hop: model_rc is the model's result (0 = every output written); anything else, or an unwritten or non-finite output, takes the frame as the identity. `out_spec` is always fully written (zeros while no spectrum is due). end returns 1 when out_spec carries an emitted frame, 0 when none is due yet (the first L hops), -1 on a contract error. One frame per call: frames_per_call > 1 is refused; with mask_lookahead 0 there is no warm-up frame. */
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
