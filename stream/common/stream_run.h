/* Run contract of the AI model streams (DFN2, DeepVQE-S, Align-ULCNet): the work a preprocess hands to the
 * accelerator side and the status its postprocess takes back. Header-only, SDK-free. */

#ifndef STREAM_COMMON_RUN_H
#define STREAM_COMMON_RUN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STREAM_WORK_MAX_INPUTS  5   /* Align-ULCNet N > 1: four spectra and frame_reset */
#define STREAM_WORK_MAX_OUTPUTS 3   /* DFN2 full: erb_mask, df_coefs, df_alpha */

/* Row-major float32 view in the ONNX logical order [b][c][h][w], w innermost; elements == b * c * h * w. */
typedef struct StreamTensorIn {
    const float *data;
    size_t elements;
    unsigned int w, h, c, b;
} StreamTensorIn;

/* As StreamTensorIn, written by the accelerator; preprocess NaN-fills it, so an unwritten element is non-finite. */
typedef struct StreamTensorOut {
    float *data;
    size_t elements;
    unsigned int w, h, c, b;
} StreamTensorOut;

typedef enum StreamWorkAction {
    STREAM_WORK_BYPASS = 0,   /* no accelerator call for this block; postprocess is still called once */
    STREAM_WORK_SUBMIT = 1    /* run the graph once on inputs[], fill outputs[] */
} StreamWorkAction;

/* One block: inputs[] are the graph's signals in order, outputs[] its heads; the recurrent state stays in the
 * accelerator. The views are valid until the paired postprocess; do not write the inputs. Before a run, zero the
 * accelerator's state when state_generation differs from its value at the last zeroing (0 on a BYPASS block). */
typedef struct StreamWork {
    StreamWorkAction action;
    StreamTensorIn   inputs[STREAM_WORK_MAX_INPUTS];
    int              input_count;
    StreamTensorOut  outputs[STREAM_WORK_MAX_OUTPUTS];
    int              output_count;
    unsigned int     state_generation;
} StreamWork;

/* Status of a SUBMIT block, passed to postprocess (ignored on BYPASS): OK = every head written by a completed
 * run; NOT_STARTED = no run started, the accelerator state untouched; any other value = the state may be dirty. */
enum {
    STREAM_RUN_OK          = 0,
    STREAM_RUN_NOT_STARTED = 0x4E53,
    STREAM_RUN_FAILED      = -1
};

#ifdef __cplusplus
}
#endif

#endif /* STREAM_COMMON_RUN_H */
