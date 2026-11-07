/* Align-ULCNet outside the network; the application frames with audiolib on a run-time UlcnetGrid (ulcnet_model_io.h;
 * Python reference aiaec_streaming). Trained against: sqrt-Hann (ac_window_build default); analysis AcStft from
 * ac_stft_config_default(&sc, grid.n_fft), hop = grid.hop, spec_scale 1, center 0 (one frame per push; the Python
 * reference is centered: 0, 2, then 1); synthesis AcIstft, same config, ola AC_OLA_ENV_NORM, ispec_scale 1, first
 * frame emits nothing, latency grid.hop. FftHandle and window table: caller-owned, shareable. */

#ifndef ULCNET_PROCESS_H
#define ULCNET_PROCESS_H

#include "ulcnet_model_io.h" /* UlcnetModelIoDescriptor */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_COMPRESSION_EXP ULCNET_MODEL_IO_COMPRESSION_EXP

/* A nonzero infer return or non-finite output fails open to the linear error: the AEC analysis seam
 * (AecResContext.formed_hop; 4ch beam.spectrum), never aec_get_output_base(); on a guard hop it can exceed the
 * capture. With infer, io_descriptor is required (NULL only for the memset-zero identity model), checked against the
 * pipeline's grid, layout, raw far and D range (not D against the graph: delay_depth must equal the export depth). */
typedef struct UlcnetModel {
    void *user;
    int (*infer)(void *user,
                 const float *err_re, const float *err_im,
                 const float *far_re, const float *far_im,
                 float *out_re, float *out_im);   /* spectrum_bins each, NaN-prefilled; 0 requires all written */
    void (*reset)(void *user);   /* delay change or pipeline reset: clears the K/V, logit, GRU state; may be NULL */
    const UlcnetModelIoDescriptor *io_descriptor;  /* NULL = not published; borrowed, must outlive the pipelines */
} UlcnetModel;

/* One model frame in flight (between *_nn_begin and *_nn_end); write all n_bins of the NaN-prefilled out_*. */
typedef struct UlcnetNnJob {
    const float *err_re, *err_im;   /* [n_bins] linear-error spectrum */
    const float *far_re, *far_im;   /* [n_bins] raw far spectrum, same frame */
    float *out_re, *out_im;         /* [n_bins] enhanced spectrum, NaN-prefilled */
} UlcnetNnJob;

/* 1 when all n_bins values of re and im are finite, else 0 (also on NULL, n_bins < 1). */
int ulcnet_frame_is_finite(const float *re, const float *im, int n_bins);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PROCESS_H */
