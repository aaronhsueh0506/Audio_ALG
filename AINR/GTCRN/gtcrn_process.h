#ifndef GTCRN_PROCESS_H
#define GTCRN_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fft_wrapper.h"

#define GTCRN_SR       16000
#define GTCRN_N_FFT    512
#define GTCRN_N_BINS   257
#define GTCRN_WIN_LEN  512
#define GTCRN_HOP_LEN  256

/* Explicit model-state tensors in export_onnx.py. The accelerator retains
 * nothing between invocations. GTCRNModelState below IS the state buffer:
 * bind each graph state input AND its matching *_out output to the same
 * struct field, so the updated state is the next call's input with no copy,
 * and call gtcrn_model_state_validate() after the invocation. The runtime
 * must read every state input before it writes any state output at the same
 * address. Binding table (graph name -> field, every tensor rank 4 and the
 * same shape at input and output):
 *   conv_enc0 / conv_enc0_out ... conv_dec2 / conv_dec2_out
 *       [1, 16, window, 33]   -> &state->conv_enc0 ... &state->conv_dec2
 *   h_tra / h_tra_out         [1, 6, 1, 16]   -> &state->h_tra
 *   h_dpgrnn / h_dpgrnn_out   [1, 4, 33, 8]   -> &state->h_dpgrnn
 *
 * Each conv tensor is the full causal window of one depthwise convolution
 * (the previous call's window with its oldest frame dropped, plus the new
 * frame). h_tra rows are the six TRA attention GRUs (encoder 0-2, decoder
 * 0-2); h_dpgrnn rows are the inter GRUs h_dpgrnn1_0, h_dpgrnn1_1,
 * h_dpgrnn2_0, h_dpgrnn2_1, whose hidden batch is the frequency axis.
 *
 * Layout versions 3 to 7 are retired. */
#define GTCRN_MODEL_LAYOUT_VERSION 8
#define GTCRN_MODEL_ERB_KEPT       65   /* low bins passed through          */
#define GTCRN_MODEL_ERB_HIGH_BANDS 64   /* compressed high bands            */
#define GTCRN_MODEL_ERB_BANDS      129  /* KEPT + HIGH_BANDS = E            */
#define GTCRN_MODEL_ERB_HIGH_BINS  192  /* N_BINS - KEPT                    */
#define GTCRN_MODEL_CONV_STATES    6
#define GTCRN_MODEL_CONV_CHANNELS  16
#define GTCRN_MODEL_CONV_WINDOW_0  3
#define GTCRN_MODEL_CONV_WINDOW_1  5
#define GTCRN_MODEL_CONV_WINDOW_2  11
#define GTCRN_MODEL_CONV_FREQ      33
#define GTCRN_MODEL_TRA_GRUS       6
#define GTCRN_MODEL_TRA_HIDDEN     16
#define GTCRN_MODEL_DPGRNN_GRUS    4
#define GTCRN_MODEL_DPGRNN_FREQ    33
#define GTCRN_MODEL_DPGRNN_HIDDEN  8

/* state_generation advances each time this library zeroes the state (reset,
 * or a refused validation). The accelerator's fixed-point copy of the state
 * is not visible here, so a runtime refills it with quantized zeros whenever
 * gtcrn_model_state_generation() differs from the value it saw last. */
typedef struct {
    /* ONNX conv_enc0..2 then conv_dec0..2. Decoder windows reverse the
     * encoder's [3,5,11] dilation windows. The leading graph batch extent is
     * one and therefore occupies no extra C dimension. */
    float conv_enc0[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_0][GTCRN_MODEL_CONV_FREQ];
    float conv_enc1[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_1][GTCRN_MODEL_CONV_FREQ];
    float conv_enc2[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_2][GTCRN_MODEL_CONV_FREQ];
    float conv_dec0[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_2][GTCRN_MODEL_CONV_FREQ];
    float conv_dec1[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_1][GTCRN_MODEL_CONV_FREQ];
    float conv_dec2[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_WINDOW_0][GTCRN_MODEL_CONV_FREQ];
    /* ONNX h_tra [1,6,1,16]. */
    float h_tra[GTCRN_MODEL_TRA_GRUS][1][GTCRN_MODEL_TRA_HIDDEN];
    /* ONNX h_dpgrnn [1,4,33,8]. Each grouped inter-RNN contains two real
     * GRUs; frequency lanes are their batch extent. */
    float h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS][GTCRN_MODEL_DPGRNN_FREQ]
                  [GTCRN_MODEL_DPGRNN_HIDDEN];
    unsigned int state_generation;
} GTCRNModelState;

typedef struct {
    float analysis_buf[GTCRN_WIN_LEN];
    float synthesis_buf[GTCRN_WIN_LEN];
    float window[GTCRN_WIN_LEN];
    float scratch_time[GTCRN_N_FFT];
    Complex scratch_freq[GTCRN_N_BINS];
    /* Caller-owned audio_common FFT handle (fft_create/fft_init for
     * GTCRN_N_FFT); the state only borrows it, mirroring the .bin matrix
     * pointers -- the loader owns every resource. */
    FftHandle* fft;
} GTCRNProcessState;

void gtcrn_process_init(GTCRNProcessState* state, FftHandle* fft);

/* First construction: zero the state and start the generation at 1. A state
 * already in use goes through gtcrn_model_state_reset(). The caller fills the
 * accelerator's state with zeros once after this. */
void gtcrn_model_state_init(GTCRNModelState* state);

/* Zero every state tensor and advance the generation. NULL is ignored. */
void gtcrn_model_state_reset(GTCRNModelState* state);

/* The generation: 1 after init, plus one for each reset and each refused
 * validation, modulo 2^32. 0 for NULL. */
unsigned int gtcrn_model_state_generation(const GTCRNModelState* state);

/* The three model-input frames from one analysis frame:
 * [sqrt(re^2 + im^2 + 1e-12), re, im] each banded through the exported ERB
 * forward matrix -- low bins pass through, high bins compress to bands.
 * Everything fixed runs HERE in fp32; the quantized graph concatenates the
 * three tensors and holds learned compute only.
 *
 * erb_fwd is the CALLER-LOADED matrix from export_erb_matrix.py's
 * erb_fwd.bin: raw float32 little-endian, bin-major
 * [GTCRN_MODEL_ERB_HIGH_BINS][GTCRN_MODEL_ERB_HIGH_BANDS]. The library
 * never touches the filesystem, so the .bin can be swapped at any time by
 * the loader that owns it. */
void gtcrn_model_input(const float spectrum[GTCRN_N_BINS][2],
                       const float* erb_fwd,
                       float mag[GTCRN_MODEL_ERB_BANDS],
                       float real_part[GTCRN_MODEL_ERB_BANDS],
                       float imag_part[GTCRN_MODEL_ERB_BANDS]);

/* The fixed back end: expand the graph's ERB-domain
 * complex mask through the caller-loaded inverse matrix (erb_inv.bin: raw
 * float32 little-endian, band-major
 * [GTCRN_MODEL_ERB_HIGH_BANDS][GTCRN_MODEL_ERB_HIGH_BINS]) and apply the
 * complex ratio mask to the SAME analysis frame the inputs came from. */
void gtcrn_model_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                        const float* erb_inv,
                        const float spectrum[GTCRN_N_BINS][2],
                        float enhanced[GTCRN_N_BINS][2]);

/* Finite-check the state in place after an invocation.
 *
 * The graph's *_out outputs are bound to the same fields as its state
 * inputs (see the binding table above), so the updated state is already
 * where the next call reads it and nothing is copied. Returns 0 when every
 * element of the six conv windows, h_tra and h_dpgrnn is finite. A null
 * argument returns -1. A single NaN or Inf anywhere zeroes the whole state
 * (the previous value no longer exists to restore), advances the generation
 * and returns -1, so the next invocation starts from the reset state instead
 * of replaying poison. The check is finite-only: a partial or mixed write
 * that is still finite cannot be detected. */
int gtcrn_model_state_validate(GTCRNModelState* state);

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
