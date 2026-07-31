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

typedef struct {
    float analysis_buf[GTCRN_WIN_LEN];
    float synthesis_buf[GTCRN_WIN_LEN];
    float window[GTCRN_WIN_LEN];
    float scratch_re[GTCRN_N_FFT];
    float scratch_im[GTCRN_N_FFT];
} GTCRNProcessState;

void gtcrn_process_init(GTCRNProcessState* state);

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
