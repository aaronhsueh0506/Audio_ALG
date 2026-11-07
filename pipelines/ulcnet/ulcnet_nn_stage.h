/* ulcnet_nn_stage.h -- example: the Align-ULCNet post-filter stage behind a linear AEC (no ABI,
 * no layout version: copy and adapt). Per hop it takes the linear-error spectrum, the same hop's
 * raw far samples and the AEC's alignment events, runs one UlcnetModel call (ulcnet_process.h)
 * and emits one WOLA hop. The grid is the application's UlcnetGrid: spectra have grid.n_bins
 * bins, hops grid.hop samples. Latency is one hop; hop #0 emits zeros. The error is the AEC's
 * analysis seam, never the rendering base of aec_get_output_base(), so the identity, fail-open
 * and reprime frames are the unguarded residual and may exceed the capture.
 *
 *     ulcnet_nn_stage_begin(stage, err_re, err_im, far, delay);   // or _complex / _time
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)             // a model frame is open
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));       // 0 = every bin written
 *     ulcnet_nn_stage_end(stage, out);                            // WOLA, one hop
 */
#ifndef PIPELINES_ULCNET_ULCNET_NN_STAGE_H
#define PIPELINES_ULCNET_ULCNET_NN_STAGE_H

#include <stddef.h>

#include "audio_stft.h"
#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Identity frames after an alignment boundary: each frame spans two hops, and both branches are
 * framed from the current hop, so only the boundary hop's frame straddles it. */
enum { ULCNET_NN_STAGE_REPRIME_FRAMES = 1 };

enum {
    ULCNET_NN_STAGE_OK = 0,
    ULCNET_NN_STAGE_INVALID_ARGUMENT = -1,
    ULCNET_NN_STAGE_SEQUENCE_ERROR = -2,   /* a call out of begin / nn_begin / nn_end / end order */
    ULCNET_NN_STAGE_DSP_ERROR = -3
};

/* Alignment events of the hop. A 4-channel unit passes delay.changed and delay.solid of its
 * front frame; a mono lane maps AecLinearContext.delay_state (changed = CHANGED, solid = not
 * UNLOCKED). A change, or the first transition to solid after the first hop, calls
 * model->reset and arms ULCNET_NN_STAGE_REPRIME_FRAMES identity frames. */
typedef struct UlcnetNnDelay {
    int changed;   /* a new alignment generation starts with this hop */
    int solid;     /* a usable alignment is in force */
} UlcnetNnDelay;

typedef struct UlcnetNnStage UlcnetNnStage;

/* Pool bytes for `grid` (ulcnet_grid_make). With err_analysis = 1 the stage also carries an STFT
 * for the time-domain error entry (ulcnet_nn_stage_begin_time). 0 for a NULL or invalid grid. */
size_t ulcnet_nn_stage_mem_size(const UlcnetGrid* grid, int err_analysis);

/* Builds the stage in a 16-byte aligned block of at least ulcnet_nn_stage_mem_size() bytes; the
 * block may hold stale data and the grid is copied. The model starts unset (identity). NULL on
 * failure; a refused argument leaves the block untouched. */
UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, const UlcnetGrid* grid,
                                    int err_analysis);

/* The grid the stage runs; NULL for a NULL stage. Valid for the stage's lifetime. */
const UlcnetGrid* ulcnet_nn_stage_grid(const UlcnetNnStage* s);

/* Installs (or clears, with NULL) the model, copied by value; model->user and a descriptor must
 * stay valid until replaced. No reset is issued: install a freshly reset runtime, or call
 * reset() after swapping mid-stream. A model that infers must publish a raw-far descriptor that
 * ulcnet_model_io_descriptor_validate() accepts for the stage's grid. Returns 0, or -1 for a
 * NULL stage or a refused model (the installed one stays). */
int ulcnet_nn_stage_set_model(UlcnetNnStage* s, const UlcnetModel* model);

/* Back to hop #0: framing history, alignment memory and a pending reprime are cleared and
 * model->reset is called. The installed model is kept. */
void ulcnet_nn_stage_reset(UlcnetNnStage* s);

/* Starts a hop from a planar error spectrum (grid.n_bins bins each; copied). far_hop is the
 * caller's raw far hop (grid.hop samples), never the aligned copy. */
int ulcnet_nn_stage_begin(UlcnetNnStage* s, const float* err_re, const float* err_im,
                          const float* far_hop, UlcnetNnDelay delay);

/* The same from an interleaved spectrum of at least grid.n_bins bins. */
int ulcnet_nn_stage_begin_complex(UlcnetNnStage* s, const Complex* err,
                                  const float* far_hop, UlcnetNnDelay delay);

/* The same from the time-domain hop underlying the error (grid.hop samples), analysed by the
 * stage's own rolling STFT; needs a stage built with err_analysis = 1. */
int ulcnet_nn_stage_begin_time(UlcnetNnStage* s, const float* err_hop,
                               const float* far_hop, UlcnetNnDelay delay);

/* 1 with the frame's tensors in *job (run the model, then nn_end), 0 when the hop has no model
 * frame (reprime, or already fetched), or a negative status. */
int ulcnet_nn_stage_nn_begin(UlcnetNnStage* s, UlcnetNnJob* job);

/* Closes the open frame: the model output replaces the error only when model_rc == 0 and every
 * bin is finite. */
int ulcnet_nn_stage_nn_end(UlcnetNnStage* s, int model_rc);

/* WOLA of the hop into out[grid.hop]; hop #0 is zeros. A frame left open or never fetched is
 * the identity. A negative return leaves recovery to the caller: reset the AEC unit and the
 * stage together. */
int ulcnet_nn_stage_end(UlcnetNnStage* s, float* out);

/* After begin: runs the installed model between nn_begin and end, on every emitted frame except
 * the reprime frames. No model, a nonzero result or a non-finite bin passes the error spectrum
 * through. */
int ulcnet_nn_stage_process(UlcnetNnStage* s, float* out);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_ULCNET_ULCNET_NN_STAGE_H */
