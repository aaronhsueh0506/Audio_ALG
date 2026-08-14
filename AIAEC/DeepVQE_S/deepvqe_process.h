/* DeepVQE-S CCM post-processing.  STFT/WOLA is AIAEC/aiaec_process.h. */
#ifndef DEEPVQE_PROCESS_H
#define DEEPVQE_PROCESS_H

#include "aiaec_process.h"

#define DEEPVQE_TIME_ORDER 3
#define DEEPVQE_FREQ_TAPS  3

typedef struct DeepVqeCcmState {
    float spectrum_re[DEEPVQE_TIME_ORDER][AIAEC_N_BINS];
    float spectrum_im[DEEPVQE_TIME_ORDER][AIAEC_N_BINS];
} DeepVqeCcmState;

void deepvqe_ccm_init(DeepVqeCcmState *state);

/* taps layout: [bin][time_order current-to-past][freq -1,0,+1][re,im]. */
void deepvqe_ccm_process(
    DeepVqeCcmState *state,
    const float input_re[AIAEC_N_BINS],
    const float input_im[AIAEC_N_BINS],
    const float taps[AIAEC_N_BINS][DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2],
    float output_re[AIAEC_N_BINS],
    float output_im[AIAEC_N_BINS]);

#endif
