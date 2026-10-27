/* ulcnet_nn_stage.h -- example: the Align-ULCNet post-filter stage behind a linear AEC.
 * No ABI, no layout version: copy and adapt.
 *
 * The stage takes the linear-error spectrum of a hop (a 4-channel unit's beamformed spectrum or
 * the STFT of a mono lane's formed hop), the same hop's raw far-end samples and the alignment
 * events of the AEC, runs one model call per hop through the UlcnetModel boundary
 * (ulcnet_process.h) and emits the Align-ULCNet WOLA hop. Per hop:
 *
 *     ulcnet_nn_stage_begin(stage, err_re, err_im, far, delay);   // or _complex / _time
 *     if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)             // a model frame is open
 *         ulcnet_nn_stage_nn_end(stage, run_network(&job));       // 0 = every bin written
 *     ulcnet_nn_stage_end(stage, out);                            // WOLA, one hop
 *
 * ulcnet_nn_stage_process() runs the installed model between begin and end. Timing: one hop of
 * added latency; hop #0 of the stream emits zeros. The far input must be the caller's raw far
 * hop, never the aligned copy (a published model descriptor must carry ULCNET_FAR_RAW).
 * Policy: the model runs for every emitted frame except the identity reprime. No model, a
 * nonzero model result or a non-finite output bin passes the error spectrum through. A new
 * alignment generation (delay.changed, or the first transition to solid after the first hop)
 * calls model->reset and arms ULCNET_NN_STAGE_REPRIME_FRAMES identity frames, because the
 * windows in flight straddle the boundary. reset() also calls model->reset.
 *
 * A negative return leaves recovery to the caller: reset the AEC unit and the stage together.
 * One build serves one grid (16 kHz / fft 512 or 48 kHz / fft 1024, ULCNET_MODEL_IO_SR and
 * ULCNET_MODEL_IO_N_FFT). */
#ifndef PIPELINES_ULCNET_ULCNET_NN_STAGE_H
#define PIPELINES_ULCNET_ULCNET_NN_STAGE_H

#include <stddef.h>

#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Emitted frames that still straddle an alignment boundary. Both branches are framed from the
 * current input hop and each frame spans two hops, so only the boundary hop's frame does. */
enum { ULCNET_NN_STAGE_REPRIME_FRAMES = 1 };

enum {
    ULCNET_NN_STAGE_OK = 0,
    ULCNET_NN_STAGE_INVALID_ARGUMENT = -1,
    ULCNET_NN_STAGE_SEQUENCE_ERROR = -2,   /* a call out of begin / nn_begin / nn_end / end order */
    ULCNET_NN_STAGE_DSP_ERROR = -3
};

/* Alignment events of the hop. A 4-channel unit passes delay.changed and delay.solid of its
 * front frame. A mono lane maps AecLinearContext.delay_state: changed = CHANGED,
 * solid = not UNLOCKED. */
typedef struct UlcnetNnDelay {
    int changed;   /* a new alignment generation starts with this hop */
    int solid;     /* a usable alignment is in force */
} UlcnetNnDelay;

typedef struct UlcnetNnStage UlcnetNnStage;

/* Pool bytes. With err_analysis = 1 the stage also carries an STFT for the time-domain error
 * entry (ulcnet_nn_stage_begin_time). 0 for an unsupported build grid. */
size_t ulcnet_nn_stage_mem_size(int err_analysis);

/* Builds the stage in a 16-byte aligned block of at least ulcnet_nn_stage_mem_size() bytes;
 * the block may hold stale data. The model starts unset (identity). NULL on failure. */
UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, int err_analysis);

/* Installs (or clears, with NULL) the model, copied by value; model->user and a descriptor
 * must stay valid until replaced. No reset is issued: install a freshly reset runtime, or
 * call reset() after swapping mid-stream. A model that infers must publish a valid
 * descriptor. Returns 0, or -1 for a NULL stage or a refused model (the installed one stays). */
int ulcnet_nn_stage_set_model(UlcnetNnStage* s, const UlcnetModel* model);

/* Back to hop #0: framing history, alignment memory and a pending reprime are cleared and
 * model->reset is called. The installed model is kept. */
void ulcnet_nn_stage_reset(UlcnetNnStage* s);

/* Starts a hop from a planar error spectrum (ULCNET_BINS bins each; copied). */
int ulcnet_nn_stage_begin(UlcnetNnStage* s, const float* err_re, const float* err_im,
                          const float* far_hop, UlcnetNnDelay delay);

/* The same from an interleaved spectrum of at least ULCNET_BINS bins. */
int ulcnet_nn_stage_begin_complex(UlcnetNnStage* s, const Complex* err,
                                  const float* far_hop, UlcnetNnDelay delay);

/* The same from the time-domain hop underlying the error (ULCNET_HOP samples), analysed by
 * the stage's own rolling STFT; needs a stage built with err_analysis = 1. */
int ulcnet_nn_stage_begin_time(UlcnetNnStage* s, const float* err_hop,
                               const float* far_hop, UlcnetNnDelay delay);

/* 1 with the frame's tensors in *job (run the model, then nn_end), 0 when the hop has no model
 * frame (reprime, or already fetched), or a negative status. */
int ulcnet_nn_stage_nn_begin(UlcnetNnStage* s, UlcnetNnJob* job);

/* Closes the open frame: the model output replaces the error only when model_rc == 0 and every
 * bin is finite. */
int ulcnet_nn_stage_nn_end(UlcnetNnStage* s, int model_rc);

/* WOLA of the hop into out[ULCNET_HOP]; hop #0 is zeros. A frame left open or never fetched is
 * the identity. */
int ulcnet_nn_stage_end(UlcnetNnStage* s, float* out);

/* After begin: the installed model (none: identity) between nn_begin and end. */
int ulcnet_nn_stage_process(UlcnetNnStage* s, float* out);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_ULCNET_ULCNET_NN_STAGE_H */
