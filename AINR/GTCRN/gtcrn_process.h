#ifndef GTCRN_PROCESS_H
#define GTCRN_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#define GTCRN_SR       16000
#define GTCRN_N_FFT    512
#define GTCRN_N_BINS   257
#define GTCRN_WIN_LEN  512
#define GTCRN_HOP_LEN  256

/* Explicit model-state tensors in export_onnx.py. The accelerator retains
 * nothing between invocations; its three *_out tensors must be committed here
 * and returned as the next call's inputs. */
#define GTCRN_MODEL_LAYOUT_VERSION 1
#define GTCRN_MODEL_CONV_SIDES     2
#define GTCRN_MODEL_CONV_CHANNELS  16
#define GTCRN_MODEL_CONV_TIME      16
#define GTCRN_MODEL_CONV_FREQ      33
#define GTCRN_MODEL_TRA_SIDES      2
#define GTCRN_MODEL_TRA_BLOCKS     3
#define GTCRN_MODEL_TRA_HIDDEN     16
#define GTCRN_MODEL_INTER_LAYERS   2
#define GTCRN_MODEL_INTER_FREQ     33
#define GTCRN_MODEL_INTER_HIDDEN   16

typedef struct {
    /* ONNX: conv_cache[2,1,16,16,33]. */
    float conv_cache[GTCRN_MODEL_CONV_SIDES][1]
                    [GTCRN_MODEL_CONV_CHANNELS]
                    [GTCRN_MODEL_CONV_TIME]
                    [GTCRN_MODEL_CONV_FREQ];
    /* ONNX: tra_cache[2,3,1,1,16]. */
    float tra_cache[GTCRN_MODEL_TRA_SIDES][GTCRN_MODEL_TRA_BLOCKS]
                   [1][1][GTCRN_MODEL_TRA_HIDDEN];
    /* ONNX: inter_cache[2,1,33,16]. */
    float inter_cache[GTCRN_MODEL_INTER_LAYERS][1]
                     [GTCRN_MODEL_INTER_FREQ]
                     [GTCRN_MODEL_INTER_HIDDEN];
} GTCRNModelState;

typedef struct {
    float analysis_buf[GTCRN_WIN_LEN];
    float synthesis_buf[GTCRN_WIN_LEN];
    float window[GTCRN_WIN_LEN];
    float scratch_re[GTCRN_N_FFT];
    float scratch_im[GTCRN_N_FFT];
} GTCRNProcessState;

void gtcrn_process_init(GTCRNProcessState* state);

void gtcrn_model_state_init(GTCRNModelState* state);

/* Copy the accelerator's updated state outputs into the next-call inputs.
 *
 * Transactional: every element of all three caches is checked first, and a
 * single NaN or Inf anywhere refuses the whole commit with -1, leaving the
 * previous state byte-identical so the caller can retry or reset. Returns 0
 * on success and -1 on a null argument or a non-finite element. A caller that
 * ignores the result keeps replaying the last good state, which is the safe
 * direction; a partial write would not be. */
int gtcrn_model_state_commit(GTCRNModelState* state,
                             const float* conv_cache_out,
                             const float* tra_cache_out,
                             const float* inter_cache_out);

/* HOP_LEN new samples -> one unnormalised complex RFFT frame. The network
 * input layout is bin-major [re,im], matching model.py's [F,T,2]. */
void gtcrn_analysis(GTCRNProcessState* state, const float* input,
                    float output[GTCRN_N_BINS][2]);

/* One enhanced [F,2] network-output frame -> HOP_LEN WOLA samples. */
void gtcrn_synthesis(GTCRNProcessState* state,
                     const float input[GTCRN_N_BINS][2],
                     float* output);

const char* gtcrn_simd_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_PROCESS_H */
