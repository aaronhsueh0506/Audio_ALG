/* Align-ULCNet signal path outside the network.
 * Python reference: aiaec_streaming.StreamSTFT / StreamISTFT.
 *
 *   C  : hop (ULCNET_HOP samples) -> sqrt-Hann STFT -> RI spectra (linear error
 *        and raw far; the production descriptor is fixed to raw far)
 *   NN : (error_ri, far_ri, explicit states) -> enhanced_ri; the accelerator is
 *        stateless and the states are held by the CPU (ulcnet_model_io)
 *   C  : enhanced_ri -> WOLA (sqrt-Hann, 50% overlap, envelope normalised,
 *        half-window trim) -> hop
 *
 * Framing is audiolib AcStft/AcIstft. The centered schedule (center 1) emits 0
 * frames on the first push, 2 on the second, then 1 per push; the rolling
 * schedule (center 0) emits 1 per push. Algorithmic latency is ULCNET_HOP
 * samples (the synthesis half-window trim).
 *
 * Memory: framing state lives in AcStft/AcIstft objects on caller memory (no
 * heap, no globals). The FftHandle and window table are caller-owned, shared by
 * all objects and must outlive them. Not included: the NPU graph and its states
 * (ulcnet_model_io), the AEC linear-error seam (aec_get_linear_context) and the
 * delay state machine (pipeline layer). */

#ifndef ULCNET_PROCESS_H
#define ULCNET_PROCESS_H

#include "audio_stft.h"      /* AcStft, AcIstft, FftHandle, Complex (audiolib) */
#include "ulcnet_model_io.h" /* UlcnetModelIoDescriptor, UlcnetFarInputMode */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Grid (fixed at build time, one grid per build) ----
 * Defined in ulcnet_model_io.h; these names are aliases. */
#define ULCNET_SR       ULCNET_MODEL_IO_SR
#define ULCNET_N_FFT    ULCNET_MODEL_IO_N_FFT
#define ULCNET_HOP      ULCNET_MODEL_IO_HOP    /* = N_FFT/2, 50% overlap (COLA) */
#define ULCNET_BINS     ULCNET_MODEL_IO_BINS   /* N_FFT/2 + 1 */

/* "<SR>x<N_FFT>" string literal, for folding the grid into a build-flags hash. */
#define ULCNET_STRINGIFY_(x) #x
#define ULCNET_STRINGIFY(x)  ULCNET_STRINGIFY_(x)
#define ULCNET_GRID_TOKEN \
    ULCNET_STRINGIFY(ULCNET_SR) "x" ULCNET_STRINGIFY(ULCNET_N_FFT)

/* The other supported grid, which this build must reject. */
#define ULCNET_OTHER_SR     (ULCNET_SR == 16000 ? 48000 : 16000)
#define ULCNET_OTHER_N_FFT  (ULCNET_N_FFT == 512 ? 1024 : 512)

/* Power-law compression exponent; defined in ulcnet_model_io.h. */
#define ULCNET_COMPRESSION_EXP ULCNET_MODEL_IO_COMPRESSION_EXP

/* ---- Framing: audiolib AcStft / AcIstft ----
 * Analysis is a sqrt-Hann STFT, centered (center 1) or rolling (center 0).
 * Synthesis is AC_OLA_ENV_NORM (squared-window envelope divided out, torch.istft
 * semantics); the first frame lies in the trimmed half window and emits nothing.
 *
 * `fft` is a ULCNET_N_FFT-point handle (fft_get_n_freqs(fft) == ULCNET_BINS) and
 * `window` a ULCNET_N_FFT sqrt-Hann table (ac_window_build). Both are
 * caller-owned, may be shared by all objects of a pipeline and must outlive
 * them. Pass them in the config of every ac_stft/ac_istft get_mem_size and init
 * call; NULL makes the object carve its own. */
void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft,
                           const float *window, int center);

/* ---- NPU model callback boundary (shared by the pipeline variants) ----
 * The pipeline does not own the NPU runtime; inference is one callback per
 * frame. The accelerator is stateless: K/V rings, logit history and GRU hidden
 * are CPU-owned (ulcnet_model_io.h). reset runs on a delay change or pipeline
 * reset and clears all of them; it may be NULL. infer returns 0 on success; any
 * other value makes the pipeline fail open to the linear error.
 *
 * Full-write contract: a return of 0 requires infer to have written all
 * ULCNET_BINS entries of out_re and out_im. The pipelines NaN-fill them before
 * every call, and a non-finite output falls open to the identity path.
 *
 * io_descriptor: a model with an `infer` callback must publish its
 * ulcnet_model_io descriptor (the accelerator adapter does); the pipelines
 * refuse it otherwise. They check the grid, layout version, raw-far contract
 * and D within [ULCNET_MODEL_IO_MIN_D, ULCNET_MODEL_IO_MAX_D], but cannot check
 * D against the graph: delay_depth must equal the export depth. NULL is allowed
 * only for the memset-zero identity model. The descriptor must outlive every
 * pipeline the model is installed into (pipelines copy the struct only). */
typedef struct UlcnetModel {
    void *user;
    int (*infer)(void *user,
                 const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                 const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                 float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]);
    void (*reset)(void *user);
    const UlcnetModelIoDescriptor *io_descriptor;  /* NULL = not published */
} UlcnetModel;

/* One model frame of a hop in flight, for callers that run the model themselves
 * (the *_nn_begin / *_nn_end pair). The spectra are the frame handed to
 * UlcnetModel.infer(); the output staging is NaN-prefilled and all ULCNET_BINS
 * entries of both outputs must be written. Valid until the matching *_nn_end(). */
typedef struct UlcnetNnJob {
    const float *err_re, *err_im;   /* [ULCNET_BINS] linear-error spectrum */
    const float *far_re, *far_im;   /* [ULCNET_BINS] raw far spectrum, same frame */
    float *out_re, *out_im;         /* [ULCNET_BINS] enhanced spectrum, NaN-prefilled */
} UlcnetNnJob;

/* 1 when every bin of (re, im) is finite. */
int ulcnet_frame_is_finite(const float re[ULCNET_BINS], const float im[ULCNET_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PROCESS_H */
