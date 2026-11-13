/* ulcnet_nn_stage.h -- example: the Align-ULCNet post-filter stage behind a linear AEC (no ABI,
 * no layout version: copy and adapt). Per hop it takes the linear-error spectrum, the spectrum of
 * the same hop's raw far samples and the AEC's alignment events, runs one UlcnetModel call
 * (ulcnet_process.h) and hands back one output spectrum. The stage performs no STFT and no ISTFT:
 * the application analyses and synthesises with its own AcStft / AcIstft. The grid is the
 * application's UlcnetGrid: spectra have grid.n_bins bins (Complex, audio scale), hops grid.hop
 * samples.
 *
 * The application's framing, the one the model was trained on: AcStft from
 * ac_stft_config_default(&cfg, grid.n_fft) (sqrt-Hann, centre 0, one frame per hop), one for the raw
 * far hop, and an AcIstft with
 * cfg.ola = AC_OLA_ACCUMULATE for the output spectrum. Each hop is the frame's inverse FFT times the
 * synthesis window, first half, plus the previous frame's saved second half; hop #0 is real output
 * (its first half over a zero history). Latency is one hop in total. The error is the AEC's analysis seam (AecResidualView.formed_hop), never the
 * rendering base (base_spec), so the identity, fail-open and reprime frames are the unguarded residual
 * and may exceed the capture.
 *
 *     ac_stft_push(far_stft, far_hop, far_spec);                              // raw far, never the aligned copy
 *     ulcnet_nn_stage_begin_spec(stage, err_spec, far_spec, delay);
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)                         // a model frame is open
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));                   // 0 = every bin written
 *     ulcnet_nn_stage_end_spec(stage, out_spec);                              // accepted model frame, else err_spec
 *     ac_istft_push(synthesis, out_spec, out_hop);                            // OLA, one hop
 *
 * Reset the stage together with the application's objects: ulcnet_nn_stage_reset(stage);
 * ac_stft_reset(far_stft); ac_stft_reset(err_stft); ac_istft_reset(synthesis). The stage keeps no
 * analysis or synthesis history, so its reset leaves those to the caller.
 */
#ifndef PIPELINES_ULCNET_ULCNET_NN_STAGE_H
#define PIPELINES_ULCNET_ULCNET_NN_STAGE_H

#include <stddef.h>

#include "audio_stft.h"         /* AcStft, AcIstft, Complex */
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
    ULCNET_NN_STAGE_SEQUENCE_ERROR = -2    /* a call out of begin_spec / nn_begin / nn_end / end_spec order */
};

/* Alignment events of the hop. A 4-channel unit passes delay.changed and delay.solid of its
 * front frame; a mono lane maps AecLinearView.delay_state (changed = CHANGED, solid = not
 * UNLOCKED). A change, or the first transition to solid after the first hop, calls
 * model->reset and arms ULCNET_NN_STAGE_REPRIME_FRAMES identity frames. */
typedef struct UlcnetNnDelay {
    int changed;   /* a new alignment generation starts with this hop */
    int solid;     /* a usable alignment is in force */
} UlcnetNnDelay;

typedef struct UlcnetNnStage UlcnetNnStage;

/* Pool bytes for `grid` (ulcnet_grid_make); 0 for a NULL or invalid grid. */
size_t ulcnet_nn_stage_mem_size(const UlcnetGrid* grid);

/* Builds the stage in a 16-byte aligned block of at least ulcnet_nn_stage_mem_size() bytes; the
 * block may hold stale data and the grid is copied. The model starts unset (identity). NULL on
 * failure; a refused argument leaves the block untouched. */
UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, const UlcnetGrid* grid);

/* The grid the stage runs; NULL for a NULL stage. Valid for the stage's lifetime. */
const UlcnetGrid* ulcnet_nn_stage_grid(const UlcnetNnStage* s);

/* Installs (or clears, with NULL) the model, copied by value; model->user and a descriptor must
 * stay valid until replaced. No reset is issued: install a freshly reset runtime, or call
 * reset() after swapping mid-stream. A model that infers must publish a raw-far descriptor that
 * ulcnet_model_io_descriptor_validate() accepts for the stage's grid. Returns 0, or -1 for a
 * NULL stage or a refused model (the installed one stays). */
int ulcnet_nn_stage_set_model(UlcnetNnStage* s, const UlcnetModel* model);

/* Back to hop #0: alignment memory, the frame index and a pending reprime are cleared and
 * model->reset is called. The installed model is kept. The caller's AcStft / AcIstft are not the
 * stage's: reset them with it (see the usage block above). */
void ulcnet_nn_stage_reset(UlcnetNnStage* s);

/* Starts a hop. err_spec is the linear-error spectrum and far_spec the spectrum of the same hop's
 * raw far samples, grid.n_bins Complex bins each (audio scale, the framing above); both are copied. */
int ulcnet_nn_stage_begin_spec(UlcnetNnStage* s, const Complex* err_spec, const Complex* far_spec,
                               UlcnetNnDelay delay);

/* 1 with the frame's tensors in *job (run the model, then nn_end), 0 when the hop has no model
 * frame (reprime, or already fetched), or a negative status. */
int ulcnet_nn_stage_nn_begin(UlcnetNnStage* s, UlcnetNnJob* job);

/* Closes the open frame: the model output replaces the error only when model_rc == 0 and every
 * bin is finite. */
int ulcnet_nn_stage_nn_end(UlcnetNnStage* s, int model_rc);

/* Ends the hop: writes grid.n_bins Complex bins to out_spec, the accepted model spectrum or else
 * the error spectrum (identity), for the caller's AcIstft; every hop, hop #0 included. A frame left
 * open or never fetched is the identity. A negative return leaves recovery to the caller: reset the
 * AEC unit, the stage and the AcStft / AcIstft together. */
int ulcnet_nn_stage_end_spec(UlcnetNnStage* s, Complex* out_spec);

/* After begin_spec: runs the installed model between nn_begin and end_spec, on every emitted frame
 * except the reprime frames, then ulcnet_nn_stage_end_spec(). No model, a nonzero result or a
 * non-finite bin passes the error spectrum through. */
int ulcnet_nn_stage_process(UlcnetNnStage* s, Complex* out_spec);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_ULCNET_ULCNET_NN_STAGE_H */
