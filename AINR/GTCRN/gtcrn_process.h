#ifndef GTCRN_PROCESS_H
#define GTCRN_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "audio_stft.h"

#define GTCRN_SR       16000
#define GTCRN_N_FFT    512
#define GTCRN_N_BINS   257
#define GTCRN_WIN_LEN  512
#define GTCRN_HOP_LEN  256

/* Explicit model-state tensors (export_onnx.py). The accelerator retains nothing
 * between invocations: GTCRNModelState is the state buffer. Bind each graph state
 * input and its matching *_out output to the same struct field (read every state
 * input before writing any state output) and call gtcrn_model_state_validate()
 * after each invocation. Every tensor is rank 4 with the same shape at input and
 * output:
 *   conv_enc0 / conv_enc0_out ... conv_dec2 / conv_dec2_out
 *       [1, 16, window, 33]   -> &state->conv_enc0 ... &state->conv_dec2
 *   h_tra / h_tra_out         [1, 6, 1, 16]   -> &state->h_tra
 *   h_dpgrnn / h_dpgrnn_out   [1, 4, 33, 8]   -> &state->h_dpgrnn
 * Each conv tensor is the full causal window of one depthwise convolution. h_tra
 * rows are the six TRA attention GRUs (encoder 0-2, decoder 0-2); h_dpgrnn rows
 * are h_dpgrnn1_0, h_dpgrnn1_1, h_dpgrnn2_0, h_dpgrnn2_1, with the frequency axis
 * as the GRU batch. Layout versions 3-7 are never reused. */
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

/* state_generation advances whenever this library zeroes the state (reset, or a
 * refused validation); a runtime holding a fixed-point copy refills it with
 * quantized zeros when gtcrn_model_state_generation() changes. */
typedef struct {
    /* conv_enc0..2 then conv_dec0..2; decoder windows reverse the encoder's
     * [3,5,11]. The leading batch extent is one and occupies no C dimension. */
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
    /* ONNX h_dpgrnn [1,4,33,8]: two real GRUs per grouped inter-RNN, frequency
     * lanes as batch. */
    float h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS][GTCRN_MODEL_DPGRNN_FREQ]
                  [GTCRN_MODEL_DPGRNN_HIDDEN];
    unsigned int state_generation;
} GTCRNModelState;

/* Framing objects, caller-owned and borrowed; the loader owns every resource. */
typedef struct {
    AcStft*  stft;
    AcIstft* istft;
} GTCRNProcessState;

/* Config of the two framing objects, for ac_stft/ac_istft get_mem_size and init:
 * 512-point sqrt-Hann, hop 256, no centering, unscaled spectra. `fft` is a
 * GTCRN_N_FFT handle shared by both objects; `window` is a shared GTCRN_WIN_LEN
 * table, or NULL for the analysis object to carve one that the synthesis object
 * borrows. Borrowed tables stay alive and unchanged as long as the objects. */
void gtcrn_framing_config(AcStftConfig* cfg, FftHandle* fft,
                          const float* window);

/* Bind the objects (either may be NULL where only the other direction is
 * used) and reset them. */
void gtcrn_process_init(GTCRNProcessState* state, AcStft* stft,
                        AcIstft* istft);

/* First construction: zero the state and start the generation at 1. A state in
 * use goes through gtcrn_model_state_reset(). The caller zeroes the accelerator
 * state once afterwards. */
void gtcrn_model_state_init(GTCRNModelState* state);

/* Zero every state tensor and advance the generation. NULL is ignored. */
void gtcrn_model_state_reset(GTCRNModelState* state);

/* The generation: 1 after init, plus one for each reset and each refused
 * validation, modulo 2^32. 0 for NULL. */
unsigned int gtcrn_model_state_generation(const GTCRNModelState* state);

/* The three model-input frames from one analysis frame: [sqrt(re^2 + im^2 +
 * 1e-12), re, im], each banded through the exported ERB forward matrix (low bins
 * pass through, high bins compress to bands). All fixed compute runs here in
 * fp32; the quantized graph holds learned compute only.
 *
 * erb_fwd is the caller-loaded erb_fwd.bin from export_erb_matrix.py: raw float32
 * little-endian, bin-major [GTCRN_MODEL_ERB_HIGH_BINS][GTCRN_MODEL_ERB_HIGH_BANDS].
 * The library never touches the filesystem. */
void gtcrn_model_input(const float spectrum[GTCRN_N_BINS][2],
                       const float* erb_fwd,
                       float mag[GTCRN_MODEL_ERB_BANDS],
                       float real_part[GTCRN_MODEL_ERB_BANDS],
                       float imag_part[GTCRN_MODEL_ERB_BANDS]);

/* The fixed back end: expand the graph's ERB-domain complex mask through the
 * caller-loaded erb_inv.bin (raw float32 little-endian, band-major
 * [GTCRN_MODEL_ERB_HIGH_BANDS][GTCRN_MODEL_ERB_HIGH_BINS]) and apply the complex
 * ratio mask to the same analysis frame the inputs came from. */
void gtcrn_model_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                        const float* erb_inv,
                        const float spectrum[GTCRN_N_BINS][2],
                        float enhanced[GTCRN_N_BINS][2]);

/* Finite-check the state in place after an invocation (the *_out outputs are
 * bound to the same fields as the inputs, so nothing is copied). Returns 0 when
 * every element of the conv windows, h_tra and h_dpgrnn is finite, -1 on a NULL
 * argument. A NaN or Inf anywhere zeroes the whole state, advances the generation
 * and returns -1. A finite partial write is not detectable. */
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
