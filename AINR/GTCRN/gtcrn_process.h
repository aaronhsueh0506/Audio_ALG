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
 * nothing between invocations. GTCRNModelState below IS the state buffer.
 * Normal path: the runtime writes each graph state *_out into its own output
 * tensor and gtcrn_model_state_inherit() copies the finite ones into the
 * struct. Optimized path: bind each graph state input AND its matching *_out
 * output to the same struct field, so the updated state is the next call's
 * input with no copy, and call only gtcrn_model_state_validate() (optional).
 * Every stateful GRU and every temporal-convolution block owns one graph
 * tensor. Binding table (graph name -> field):
 *   conv_enc0 / conv_enc0_out ... conv_dec2 / conv_dec2_out
 *                                  -> &state->conv_enc0 ... &state->conv_dec2
 *   h_tra_enc0..2, h_tra_dec0..2 (+ _out)
 *                                  -> &state->h_tra[0..5]
 *   h_dpgrnn1_0, h_dpgrnn1_1, h_dpgrnn2_0, h_dpgrnn2_1 (+ _out)
 *                                  -> &state->h_dpgrnn[0..3]
 * Aliasing precondition (optimized path): the runtime must read every state
 * input before it writes any state output at the same address. */
/* Version 3 dropped conv_cache's size-1 batch dim from the graph tensor
 * ([2,16,16,33] instead of [2,1,16,16,33]); the bytes in this struct are
 * unchanged, but the tensor rank is part of the binding contract. Version 4
 * moved the magnitude feature to the host. Version 5 moves the WHOLE fixed
 * front/back end out: the host bands [mag, re, im] through the exported ERB
 * forward matrix and feeds THREE separate [1,GTCRN_MODEL_ERB_BANDS,1]
 * inputs (independent quantization scales); the graph returns the
 * ERB-domain complex mask and the host applies ERB inverse + CRM. Version 6
 * splits the packed convolution history and grouped DPGRNN hidden tensors
 * into their actual block/GRU slots. Total state bytes are unchanged; graph
 * Slice/Gather/Concat state packing is removed. */
/* ⚠ Version 7 is RESERVED, not free: export_onnx.py's experimental
 * 'combined' state layout publishes it (the sixteen slots regrouped by shape
 * into conv_cache [2,C,sum(pads),F], h_tra and h_dpgrnn). Total state bytes
 * are again unchanged and the struct below still describes them, but the
 * tensor names and ranks a runtime binds are not. Nothing here binds that
 * layout, so a board built against this header refuses such a graph -- which
 * is the intent. The next real bump of this constant must therefore go to 8,
 * or the two files would disagree about what 7 means. */
#define GTCRN_MODEL_LAYOUT_VERSION 6
#define GTCRN_MODEL_ERB_KEPT       65   /* low bins passed through          */
#define GTCRN_MODEL_ERB_HIGH_BANDS 64   /* compressed high bands            */
#define GTCRN_MODEL_ERB_BANDS      129  /* KEPT + HIGH_BANDS = E            */
#define GTCRN_MODEL_ERB_HIGH_BINS  192  /* N_BINS - KEPT                    */
#define GTCRN_MODEL_CONV_STATES    6
#define GTCRN_MODEL_CONV_CHANNELS  16
#define GTCRN_MODEL_CONV_TIME_0    2
#define GTCRN_MODEL_CONV_TIME_1    4
#define GTCRN_MODEL_CONV_TIME_2    10
#define GTCRN_MODEL_CONV_FREQ      33
#define GTCRN_MODEL_TRA_GRUS       6
#define GTCRN_MODEL_TRA_HIDDEN     16
#define GTCRN_MODEL_DPGRNN_GRUS    4
#define GTCRN_MODEL_DPGRNN_FREQ    33
#define GTCRN_MODEL_DPGRNN_HIDDEN  8

typedef struct {
    /* ONNX conv_enc0..2 then conv_dec0..2. Decoder depths reverse the
     * encoder's [2,4,10] dilation history. The leading graph batch extent is
     * one and therefore occupies no extra C dimension. */
    float conv_enc0[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_0][GTCRN_MODEL_CONV_FREQ];
    float conv_enc1[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_1][GTCRN_MODEL_CONV_FREQ];
    float conv_enc2[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_2][GTCRN_MODEL_CONV_FREQ];
    float conv_dec0[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_2][GTCRN_MODEL_CONV_FREQ];
    float conv_dec1[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_1][GTCRN_MODEL_CONV_FREQ];
    float conv_dec2[GTCRN_MODEL_CONV_CHANNELS]
                   [GTCRN_MODEL_CONV_TIME_0][GTCRN_MODEL_CONV_FREQ];
    /* ONNX: h_tra_enc0..2 then h_tra_dec0..2, each [1,1,16]. */
    float h_tra[GTCRN_MODEL_TRA_GRUS][1][1][GTCRN_MODEL_TRA_HIDDEN];
    /* ONNX: h_dpgrnn1_0, h_dpgrnn1_1, h_dpgrnn2_0, h_dpgrnn2_1, each
     * [1,33,8]. Each grouped inter-RNN contains two real GRUs; frequency
     * lanes are their batch extent. */
    float h_dpgrnn[GTCRN_MODEL_DPGRNN_GRUS][1]
                  [GTCRN_MODEL_DPGRNN_FREQ]
                  [GTCRN_MODEL_DPGRNN_HIDDEN];
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

void gtcrn_model_state_init(GTCRNModelState* state);

/* The three model-input frames from one analysis frame (model layout v5):
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

/* The fixed back end (model layout v5): expand the graph's ERB-domain
 * complex mask through the caller-loaded inverse matrix (erb_inv.bin: raw
 * float32 little-endian, band-major
 * [GTCRN_MODEL_ERB_HIGH_BANDS][GTCRN_MODEL_ERB_HIGH_BINS]) and apply the
 * complex ratio mask to the SAME analysis frame the inputs came from. */
void gtcrn_model_output(const float mask_erb[GTCRN_MODEL_ERB_BANDS][2],
                        const float* erb_inv,
                        const float spectrum[GTCRN_N_BINS][2],
                        float enhanced[GTCRN_N_BINS][2]);

/* Optimized path: finite-check the state in place after an invocation.
 *
 * The graph's *_out outputs are bound to the same fields as its state
 * inputs (see the binding table above), so the updated state is already
 * where the next call reads it and nothing is copied. Calling this is
 * optional for a trusted runtime. Returns 0 when every element of the six
 * conv, six h_tra and four h_dpgrnn tensors is finite. A null argument
 * returns -1. A single NaN or Inf anywhere zeroes the whole state (the
 * previous value no longer exists to restore) and returns -1, so the next
 * invocation starts from the reset state instead of replaying poison.
 * The check is finite-only: a partial or mixed write that is still finite
 * cannot be detected. */
int gtcrn_model_state_validate(GTCRNModelState* state);

/* Normal path: take the runtime's own state output tensors into the state.
 *
 * conv_out[i] (conv_enc0..2, conv_dec0..2), h_tra_out[i] and h_dpgrnn_out[i]
 * point at the graph's *_out tensors, laid out as the matching struct
 * fields. A tensor whose pointer already equals its field was written in
 * place: it is left alone (not copied, not checked; use
 * gtcrn_model_state_validate() for those). Every other tensor is finite-
 * checked first, and only if all of them pass are they copied. Returns 0 on
 * success. A null state, array or element, or a single NaN or Inf in any
 * tensor that would be copied, returns -1 and copies nothing: a refused
 * inherit leaves the state intact, so the caller treats the frame as failed.
 * (The in-place path differs: gtcrn_model_state_validate() restarts the
 * state from zero.) */
int gtcrn_model_state_inherit(
    GTCRNModelState* state,
    const float* const conv_out[GTCRN_MODEL_CONV_STATES],
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
