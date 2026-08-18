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
 * nothing between invocations; its *_out state tensors must be committed
 * here and returned as the next call's inputs. Every GRU hidden is its own
 * h_* tensor; only the temporal conv history is a combined cache. */
/* Version 3 dropped conv_cache's size-1 batch dim from the graph tensor
 * ([2,16,16,33] instead of [2,1,16,16,33]); the bytes in this struct are
 * unchanged, but the tensor rank is part of the binding contract. Version 4
 * moved the magnitude feature to the host: the graph input is
 * [1,GTCRN_N_BINS,1,3] = [mag, re, im] built by gtcrn_model_input(), so the
 * sqrt never enters the quantized graph. */
#define GTCRN_MODEL_LAYOUT_VERSION 4
#define GTCRN_MODEL_CONV_SIDES     2
#define GTCRN_MODEL_CONV_CHANNELS  16
#define GTCRN_MODEL_CONV_TIME      16
#define GTCRN_MODEL_CONV_FREQ      33
#define GTCRN_MODEL_TRA_GRUS       6
#define GTCRN_MODEL_TRA_HIDDEN     16
#define GTCRN_MODEL_DPGRNN_GRUS    2
#define GTCRN_MODEL_DPGRNN_FREQ    33
#define GTCRN_MODEL_DPGRNN_HIDDEN  16

typedef struct {
    /* ONNX: conv_cache[2,16,16,33]. */
    float conv_cache[GTCRN_MODEL_CONV_SIDES]
                    [GTCRN_MODEL_CONV_CHANNELS]
                    [GTCRN_MODEL_CONV_TIME]
                    [GTCRN_MODEL_CONV_FREQ];
    /* ONNX: h_tra_enc0..2 then h_tra_dec0..2, each [1,1,16]. */
    float h_tra[GTCRN_MODEL_TRA_GRUS][1][1][GTCRN_MODEL_TRA_HIDDEN];
    /* ONNX: h_dpgrnn1..2, each [1,33,16]; this GRU batches the frequency
     * lanes, so the middle extent is the lane count, not a batch of one. */
    float h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS][1]
                  [GTCRN_MODEL_DPGRNN_FREQ]
                  [GTCRN_MODEL_DPGRNN_HIDDEN];
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

/* One model-input frame from one analysis frame: features[k] =
 * [sqrt(re^2 + im^2 + 1e-12), re, im]. The magnitude runs HERE, in fp32,
 * so the quantized graph starts at the ERB matmul (model layout v4). */
void gtcrn_model_input(const float spectrum[GTCRN_N_BINS][2],
                       float features[GTCRN_N_BINS][3]);

/* Copy the accelerator's updated state outputs into the next-call inputs.
 *
 * ``h_tra_out`` holds the six TRA GRU hiddens in graph order (encoder blocks
 * then decoder blocks) and ``h_dpgrnn_out`` the two DPGRNN hiddens.
 *
 * Transactional: every element of every state tensor is checked first, and a
 * single NaN or Inf anywhere refuses the whole commit with -1, leaving the
 * previous state byte-identical so the caller can retry or reset. Returns 0
 * on success and -1 on a null argument or a non-finite element. A caller that
 * ignores the result keeps replaying the last good state, which is the safe
 * direction; a partial write would not be. */
int gtcrn_model_state_commit(GTCRNModelState* state,
                             const float* conv_cache_out,
                             const float* const h_tra_out[GTCRN_MODEL_TRA_GRUS],
                             const float* const h_dpgrnn_out[GTCRN_MODEL_DPGRNN_GRUS]);

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
