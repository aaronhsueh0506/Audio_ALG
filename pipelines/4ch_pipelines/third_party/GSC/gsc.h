#ifndef GSC_H
#define GSC_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GSC_USE_PROJECTION_BLOCKING
#define GSC_USE_PROJECTION_BLOCKING 1
#endif

typedef struct {
    int enable;                  /* 0: bypass GSC, 1: enable GSC */
    float lambda;
    float mu;

    /* fix beam config */
    int enable_fix_mode;          /* 0: AUTO DOA, 1: FIX BEAM */
    float fixed_doa_rad;          /* only used when enable_fix_mode == 1 */
    int fixed_align_notebook;     /* 1: fixed mode matches Blocking_DSP_GSC.ipynb */

    /* adaptive update interval */
    int adapt_interval;           /* update RLS every N frames, <=0 means 1 */
} GSC_Config;

typedef struct {
    int enable;

    int M;
    int F;
    int num_angles;

    float lambda;
    float mu;

    /* fix beam config */
    int enable_fix_mode;
    float fixed_doa_rad;
    int fixed_align_notebook;

    /* adaptive update interval */
    int adapt_interval;

    /* steering */
    Complex*** a_array;

    /* RLS */
    Complex*** P;   // (F,M,M)
    Complex** wa;   // (M,F)

    /* Persistent per-hop work area. Keeping the F- and M*F-sized arrays
     * here avoids a ~29 KiB stack spike at the 48 kHz / 1024-FFT grid. */
    Complex* scratch;
    Complex* scratch_das;
    Complex* scratch_wu;
    Complex* scratch_spec;
    Complex* scratch_u;
#if !GSC_USE_PROJECTION_BLOCKING
    Complex* scratch_b;
#endif

    /* state */
    int initialized;
    int first_doa_found;
    int first_doa_frame;
    float current_doa;
    int frame_idx;

    /* log */
    float doa_used;
    int adaptive;

} GSC;

/* create */
GSC* gsc_create(int M, int F, int num_angles,
                Complex*** a_array,
                const GSC_Config* cfg);

void gsc_process(GSC* g,
                 Complex** X,
                 float doa_s,
                 int allow_adapt_in,
                 const int* mask,
                 Complex* gsc_out);

/*
 * Same GSC processing and state update as gsc_process(), plus the exact
 * pre-update effective coefficients under:
 *
 *   gsc_out[f] = sum(effective_weights[m,f] * X[m,f])
 *
 * (no conjugation at the call site).  The output buffer may be NULL only
 * when the corresponding result is not needed; gsc_out remains required.
 */
void gsc_process_with_weights(GSC* g,
                              Complex** X,
                              float doa_s,
                              int allow_adapt_in,
                              const int* mask,
                              Complex* gsc_out,
                              Complex* effective_weights);

void gsc_reset(GSC* g);
void gsc_destroy(GSC* g);
float gsc_get_doa_used(const GSC* g);
int gsc_get_adaptive(const GSC* g);

/*
 * The per-hop leak factor applied to the adaptive weight state `wa` during
 * an active, unmasked RLS update (see gsc.c's GSC_WA_LEAK comment). Exposed
 * read-only so callers/tests can reference the authoritative constant
 * instead of duplicating its numeric value.
 */
float gsc_wa_leak_factor(void);

#ifdef __cplusplus
}
#endif

#endif
