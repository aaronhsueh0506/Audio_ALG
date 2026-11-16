/* Post-RES DeepFilterNet2 stage shared by the mono and 4-channel "AEC + RES + DFN2" pipelines. One
 * hop in, one spectrum out: A = E * dfn2_stft_scale(n_fft) (E: pre-RES spectrum the RES gain was
 * estimated from), B = P * dfn2_stft_scale(n_fft) (P: post-RES spectrum, E * G_res + comfort
 * noise), heads = DFN2(estimated on A), Y = heads applied on B, out = Y * dfn2_stft_unscale(n_fft).
 * The model estimates from the pre-RES spectrum and applies to the post-RES one; with the
 * comfort-noise fill off E and P share phase, so the deep-filter taps fitted on A apply to B (the
 * hosting pipelines keep the fill off by default).
 *
 * Timing: the emitted spectrum belongs to source frame t - L, L = dfn_res_stage_lookahead_frames()
 * (grid.mask_lookahead + grid.df_lookahead, 2 for the shipped 1/1 pair); the first L hops return 0
 * with an all-zero spectrum, so the caller synthesises unconditionally. Every hop runs begin, model, end,
 * except that the first grid.mask_lookahead hops after a start need no model frame (begin returns 0, the
 * model does not run): the audio is that of a stream whose model starts at its first real frame.
 *
 * Fail-open: no model (model.infer == NULL), a nonzero run status, or an unwritten/non-finite/over-limit
 * output (DFN2_PREPOST_HEAD_LIMIT) takes the frame as an identity through the cascade (clocks advance either way); the
 * output is P delayed by L hops. Run status (stream_run.h): 0 = every head written;
 * STREAM_RUN_NOT_STARTED = the device state was not touched (no model, or a callback that ran nothing); any other
 * value = it may be dirty. The library advances the state generation whenever the device state may no longer be
 * the one to continue from (reset, refused commit, a dirty status, a model restart). Before a run, zero the
 * accelerator's state when the generation differs from the value at the last zeroing. An unwritten/non-finite/over-limit
 * output is a refused commit (frame_finish()'s own check).
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
#define DFN_RES_STAGE_LAYOUT_VERSION     6u
#define DFN_RES_STAGE_BACKEND_NE10       2u   /* id 1 is never returned */

/* Frames the emitted spectrum trails the newest input by: the DFN2 cascade of the grid's
 * lookahead pair (the stage runs one frame per call). The audio-scale spectra the stage takes
 * and returns are scaled to and from DFN2's normalized=True spectra by dfn2_stft_scale(n_fft)
 * (1/sqrt(n_fft)) and dfn2_stft_unscale(n_fft) (sqrt(n_fft)): exact powers of two where n_fft is
 * a power of four (256, 1024, 4096), correctly rounded floats elsewhere (512). The caller's
 * AcStft and AcIstft keep audiolib's default spec_scale / ispec_scale of 1: the stage applies
 * these itself. */
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

/* Between hops only; -1 on NULL, a non-finite value or a hop in progress (begun, not yet ended). */
int dfn_res_stage_set_atten_lim(DfnResStage *s, float atten_lim_db);

/* The grid the stage runs (the config's, as the library holds it); NULL for NULL. */
const DFN2Grid *dfn_res_stage_grid(const DfnResStage *s);

/* Generation of the model's recurrent state (rule above): 1 after init, plus one per reset, refused
 * commit and dirty run status; 0 for NULL. The stage registers no state: the model callback zeroes its own when it
 * changes. */
unsigned int dfn_res_stage_state_generation(const DfnResStage *s);

/* Source frame of the last emitted spectrum; -1 before the first emission. */
int dfn_res_stage_output_frame_index(const DfnResStage *s, long long *frame);

/* Hop counters: frames pushed, head evaluations committed, frames taken as identity; the hops with no
 * model frame count in frames_in only. */
void dfn_res_stage_get_counters(const DfnResStage *s, long long *frames_in,
                                long long *commits, long long *skips);

/* The model's tensor views of one open frame (DFN2PrepostInputs/Outputs; layout in
 * dfn2_prepost.h). Valid until dfn_res_stage_end(). */
typedef struct DfnResStageJob {
    DFN2PrepostInputs  inputs;
    DFN2PrepostOutputs outputs;
} DfnResStageJob;

/* One hop in two calls around the model. `estimate_spec` is the pre-RES spectrum (E), `apply_spec` the post-RES spectrum (P); both are borrowed for the begin call only. begin returns 1 with the open frame's views in *job (run the model, then call end), 0 when the hop needs no model frame (the first mask_lookahead hops after a start; call end all the same, its model_rc is ignored), or -1 on a contract error (NULL arguments, a bin that is non-finite or whose product with dfn2_stft_scale(n_fft) exceeds 2^63 -- the block is refused whole and the stage is untouched --, a hop already begun, a frame the DFN2 class refuses). end closes the hop: model_rc is the run status (stream_run.h: STREAM_RUN_OK = every output written, STREAM_RUN_NOT_STARTED = nothing ran, any other value a started run that failed); anything but STREAM_RUN_OK, or an unwritten or non-finite output, takes the frame as the identity. `out_spec` is always fully written (zeros while no spectrum is due). end returns 1 when out_spec carries an emitted frame, 0 when none is due yet (the first L hops), -1 on a contract error. One frame per call: frames_per_call > 1 is refused; with mask_lookahead 0 every hop has a model frame. */
int dfn_res_stage_begin(DfnResStage *s,
                        const Complex *estimate_spec,
                        const Complex *apply_spec,
                        DfnResStageJob *job);
int dfn_res_stage_end(DfnResStage *s, int model_rc, Complex *out_spec);

/* Runs the configured model callback on an open frame: its run status, or STREAM_RUN_NOT_STARTED when no callback
 * is set. */
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
