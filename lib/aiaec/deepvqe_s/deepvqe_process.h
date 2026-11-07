/* DeepVQE-S CCM post-processing: the head's complex taps applied to a ring of the last
 * DEEPVQE_TIME_ORDER raw microphone spectra; the spectra hold n_bins values (the caller's grid). */
#ifndef DEEPVQE_PROCESS_H
#define DEEPVQE_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_TIME_ORDER 3
#define DEEPVQE_FREQ_TAPS  3

/* The two rings are borrowed (they must outlive the state), DEEPVQE_TIME_ORDER * n_bins floats
 * each (real, imaginary), newest spectrum first. */
typedef struct DeepVqeCcmState {
    int    n_bins;
    float *spectrum_re;
    float *spectrum_im;
} DeepVqeCcmState;

/* Bind the rings and zero them. Returns 0, or -1 on NULL or n_bins < 3. */
int deepvqe_ccm_init(DeepVqeCcmState *state, int n_bins,
                     float *spectrum_re, float *spectrum_im);

/* Zero both rings; n_bins and the pointers stay. NULL or an unbound state is ignored. */
void deepvqe_ccm_reset(DeepVqeCcmState *state);

/* Push the input spectrum into the ring and write the convolved spectrum. `taps` is
 * [bin][time_order current-to-past][freq -1,0,+1][re,im], flat; the outputs must not alias
 * the ring, the input or the taps. A NULL argument or an unbound state is ignored. */
void deepvqe_ccm_process(DeepVqeCcmState *state,
                         const float *input_re, const float *input_im,
                         const float *taps,
                         float *output_re, float *output_im);

#ifdef __cplusplus
}
#endif

#endif
