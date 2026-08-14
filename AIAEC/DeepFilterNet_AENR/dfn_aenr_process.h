/* DeepFilterNet-AENR host boundary.
 *
 * The model consumes independently normalised linear-error and far-reference
 * features, predicts DFN2 heads, and composes those heads onto linear_error.
 * Keeping two independent DFN2State objects is intentional: sharing either
 * normalisation EMA changes the feature contract.  This reference favours
 * correctness; a target port may split the feature-only fields out of the far
 * state after proving parity and updating its memory contract.
 */
#ifndef DFN_AENR_PROCESS_H
#define DFN_AENR_PROCESS_H

#include "dfn2_process.h"
#include "fft_wrapper.h"

typedef struct DfnAenrAnalysis {
    const float *window;
    FftHandle *fft;
    float history[DFN2_N_FFT];
    long hops_seen;
    float segment[DFN2_N_FFT];
    Complex spectrum[DFN2_N_BINS];
} DfnAenrAnalysis;

typedef struct DfnAenrSynthesis {
    const float *window;
    FftHandle *fft;
    float overlap[DFN2_N_FFT];
    float envelope[DFN2_N_FFT];
    long frames_seen;
    Complex spectrum[DFN2_N_BINS];
    float time[DFN2_N_FFT];
} DfnAenrSynthesis;

typedef struct DfnAenrProcessState {
    DFN2State error;
    DFN2State far;
    DfnAenrAnalysis error_analysis;
    DfnAenrAnalysis far_analysis;
    DfnAenrSynthesis synthesis;
} DfnAenrProcessState;

void dfn_aenr_make_window(float window[DFN2_N_FFT]);

/* fft/window are caller-owned and may be shared by the two analyses and
 * synthesis because their transforms are invoked sequentially. */
int dfn_aenr_process_init(DfnAenrProcessState *state, FftHandle *fft,
                          const float window[DFN2_N_FFT]);

/* Centered analysis matching AIAEC's StreamSTFT and normalized=True:
 * push #1 returns 0 frames, push #2 returns 2, later pushes return 1. */
int dfn_aenr_analysis_push(
    DfnAenrProcessState *state,
    const float error_hop[DFN2_HOP_LEN],
    const float far_hop[DFN2_HOP_LEN],
    float error_re[2][DFN2_N_BINS],
    float error_im[2][DFN2_N_BINS],
    float far_re[2][DFN2_N_BINS],
    float far_im[2][DFN2_N_BINS]);
int dfn_aenr_analysis_flush(
    DfnAenrProcessState *state,
    float error_re[2][DFN2_N_BINS],
    float error_im[2][DFN2_N_BINS],
    float far_re[2][DFN2_N_BINS],
    float far_im[2][DFN2_N_BINS]);
void dfn_aenr_compute_features(DfnAenrProcessState *state,
                               const float error_re[DFN2_N_BINS],
                               const float error_im[DFN2_N_BINS],
                               const float far_re[DFN2_N_BINS],
                               const float far_im[DFN2_N_BINS],
                               float error_erb[DFN2_N_ERB],
                               float error_spec[2 * DFN2_DF_BINS],
                               float far_erb[DFN2_N_ERB],
                               float far_spec[2 * DFN2_DF_BINS]);
int dfn_aenr_compose_stream(DfnAenrProcessState *state,
                            const float error_re[DFN2_N_BINS],
                            const float error_im[DFN2_N_BINS],
                            int heads_valid,
                            const float erb_mask[DFN2_N_ERB],
                            const float coefs[DFN2_DF_BINS * DFN2_DF_ORDER * 2],
                            float alpha, float atten_lim_db,
                            float output_re[DFN2_N_BINS],
                            float output_im[DFN2_N_BINS],
                            long long *output_frame_index);

/* Centered normalized=True WOLA. The first spectrum emits no samples;
 * subsequent spectra emit one hop. */
int dfn_aenr_synthesis_push(DfnAenrProcessState *state,
                            const float real[DFN2_N_BINS],
                            const float imag[DFN2_N_BINS],
                            float output[DFN2_HOP_LEN]);
int dfn_aenr_synthesis_flush(DfnAenrProcessState *state,
                             float output[DFN2_N_FFT]);

#endif
