/* AIAEC customer interface (DeepVQE-S / Align-ULCNet). The caller sets {sr, fft, hop_in}; the product build binds
 * one model (aiaec_product.h) and the compiled-in catalog resolves N = hop_in / (fft / 2), the delay depth and the
 * graph. The accelerator (open the graph, move tensors, run, read back) is the sample's:
 *
 *     AiaecConfig cfg = { sr, fft, hop_in };
 *     aiaec_get_mem_size(&cfg, &req);  ...allocate...  aiaec_init(&cfg, pool, req.bytes, &ctx, &info);   // no SDK
 *     ...open info.graph_path, check its ports against the sample's model configuration, seed the device state...
 *     per block of hop_in samples:
 *         aiaec_preprocess(ctx, &in, &work);        // work.action: SUBMIT or BYPASS
 *         ...SUBMIT: the sample's accelerator run on work; BYPASS: no accelerator call...
 *         aiaec_postprocess(ctx, run_status, pcm_out);   // once per successful preprocess, failed run or BYPASS included
 *     the output trails the input by one model hop (fft / 2 samples), both models, every N.
 *
 * Run status and state generation: STREAM_RUN_* in stream_run.h. The status of a BYPASS block is
 * not read. STREAM_RUN_NOT_STARTED leaves the generation alone, except for an Align-ULCNet packet with an event frame:
 * the restart the graph never applied advances it once.
 *
 * Work (StreamWork): inputs are the signals in graph order, outputs the head (NaN-filled, so a head the accelerator
 * leaves unwritten counts as non-finite); a BYPASS block's work is all zero (an Align-ULCNet N = 1 event hop).
 * DeepVQE-S: mic, far [1,2,N,bins]; CCM taps [1,N,bins,18]. Align-ULCNet: error_mag, far_mag, error_cos, error_sin
 * [1,1,N,bins], at N > 1 frame_reset [1,1,1,N]; complex mask [1,2,N,bins]. An Align-ULCNet event hop restarts only the
 * model: the analysis and synthesis keep running and its pcm_out is the hop's error spectrum through them.
 *
 * Refusals of a preprocess, in this order, before any state change and with no block opened: NULL / SEQUENCE,
 * INPUT_KIND, INPUT_SIZE, NULL input, NONFINITE_INPUT (a NaN or Inf in any plane of a _freq input, either model; for the
 * Align-ULCNet float PCM input a sample beyond AIAEC_PCM_LIMIT_BITS too; DeepVQE-S int16 PCM needs no scan).
 *
 * Align-ULCNet takes the linear AEC's float residual, the raw far of the same hop and one AiaecAlignEvent per model
 * hop: a pipeline with its own AEC passes them directly; a standalone sample links front/ (libaiaec_front.a) and
 * calls aiaec_front_preprocess(). A configuration with no registered grid, hop_in or graph is refused by
 * aiaec_get_mem_size() and aiaec_init() before any SDK call. The PCM (no suffix) and "_freq" families
 * share configuration, catalog and AiaecInfo; both fill a StreamWork (stream_run.h). A context is used from one
 * thread at a time. */

#ifndef AIAEC_H
#define AIAEC_H

#include <stddef.h>
#include <stdint.h>

#include "stream_mem.h"    /* StreamMemReq; include roots: stream/common and audiolib common/ */
#include "stream_run.h"    /* StreamWork, STREAM_RUN_* */

#ifdef __cplusplus
extern "C" {
#endif

#define AIAEC_ALIGNMENT 16u

typedef struct AiaecConfig {
    int sr;        /* Hz */
    int fft;       /* FFT size; model_hop = fft / 2 */
    int hop_in;    /* new samples per channel per call: a positive multiple of model_hop; N = hop_in / model_hop */
} AiaecConfig;

enum {
    AIAEC_OK = 0,
    AIAEC_ERR_NULL = -1,
    AIAEC_ERR_MODEL_ID = -2,          /* the product's bound model is unknown or not in this build */
    AIAEC_ERR_GRID = -3,              /* no catalog entry for (sr, fft) */
    AIAEC_ERR_IN_HOP = -4,            /* hop_in is not a positive multiple of model_hop */
    AIAEC_ERR_FRAMES_PER_CALL = -5,   /* resolved N is not in the catalog or has no delivered graph */
    AIAEC_ERR_POOL = -6,              /* pool too small or misaligned */
    AIAEC_ERR_SEQUENCE = -7,          /* preprocess/postprocess not called in pairs */
    AIAEC_ERR_INPUT_KIND = -8,        /* AiaecInput.kind does not match the product's model (before any state write) */
    AIAEC_ERR_INPUT_SIZE = -9,        /* samples != hop_in, or (Align-ULCNet) event_count != N (before any state write) */
    AIAEC_ERR_NONFINITE_INPUT = -10   /* NaN/Inf in an input plane, or an Align-ULCNet PCM sample beyond AIAEC_PCM_LIMIT_BITS
                                       * (before any state write: no block is open, the generation does not move) */
};

/* Largest accepted |sample| of Align-ULCNet's float PCM residual_in and far_in, as IEEE bits: 2^112, so the analysis
 * spectra (at most fft * |sample|) stay finite for every grid. Spectrum bins of the _freq family only have to be finite. */
#define AIAEC_PCM_LIMIT_BITS 0x77800000u

/* Short text for a status code, for a log line. Never NULL. */
const char *aiaec_status_text(int code);

/* Align-ULCNet's per-model-hop event from the linear AEC. An event hop is changed, or solid going false -> true after an
 * earlier hop (previous solid carried across blocks); the model restarts there and the hop's frame is the error spectrum. */
typedef struct AiaecAlignEvent {
    int changed;
    int solid;
} AiaecAlignEvent;

/* The signal shape an AiaecInput carries; a mismatch with the product's model is AIAEC_ERR_INPUT_KIND. */
typedef enum AiaecInputKind {
    AIAEC_INPUT_DEEPVQE_S    = 1,
    AIAEC_INPUT_ALIGN_ULCNET = 2
} AiaecInputKind;

/* One block's input, tagged by kind; samples per channel must equal cfg->hop_in. DeepVQE-S: int16 mic and far.
 * Align-ULCNet: full-scale float residual_in (the AEC's error hop, not via int16), far_in (raw far), events oldest
 * first with event_count == N. */
typedef struct AiaecInput {
    AiaecInputKind kind;
    int                 samples;
    union {
        struct {
            const int16_t *mic_in;
            const int16_t *far_in;
        } deepvqe_s;
        struct {
            const float             *residual_in;
            const float             *far_in;
            const AiaecAlignEvent   *events;
            int                      event_count;
        } align_ulcnet;
    } u;
} AiaecInput;

/* ---------------------------------------------------------------------------------------------
 * What one init selects: the graph.
 * ------------------------------------------------------------------------------------------- */

/* The graph the init resolved for this configuration. */
typedef struct AiaecInfo {
    const char  *graph_path;           /* model.bin for this configuration; a static string, never NULL here */
} AiaecInfo;

/* ---------------------------------------------------------------------------------------------
 * PCM block family: PCM in, PCM out, both analysis STFTs and the synthesis ISTFT inside.
 * ------------------------------------------------------------------------------------------- */

typedef struct AiaecCtx AiaecCtx;

/* Exact pool size for cfg: AIAEC_OK (req filled) or an AIAEC_ERR_* (req untouched). Check order: the product's model,
 * the grid, hop_in divisibility, a shipped graph for the resolved N. */
int aiaec_get_mem_size(const AiaecConfig *cfg, StreamMemReq *req);

/* Constructs in caller memory (aligned to req.alignment, pool_bytes >= get_mem_size()) and fills info: AIAEC_OK with
 * *ctx set, else an AIAEC_ERR_* with *ctx NULL and info untouched. Same check order, then AIAEC_ERR_POOL; no file I/O. */
int aiaec_init(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecCtx **ctx, AiaecInfo *info);

/* Back to the start: state zeroed, generation advanced, an open block discarded. */
void aiaec_reset(AiaecCtx *ctx);

/* Current state generation; 0 for a NULL ctx. */
unsigned int aiaec_state_generation(const AiaecCtx *ctx);

/* One block in. AIAEC_OK: work is filled and exactly one aiaec_postprocess() must follow; a refusal (above) opens no block. */
int aiaec_preprocess(AiaecCtx *ctx, const AiaecInput *input, StreamWork *work);

/* Closes the block with its run status (read on SUBMIT only; ERR_SEQUENCE writes nothing); pcm_out gets hop_in samples.
 * A failed run or non-finite result is bypassed (DeepVQE-S silence, Align-ULCNet the error spectrum); the generation
 * moves as described above, and for a non-finite result too. */
int aiaec_postprocess(AiaecCtx *ctx, int run_status, int16_t *pcm_out);

/* ---------------------------------------------------------------------------------------------
 * _freq family (AiaecFreqInput spectra in, no STFT / ISTFT): the same hop_in, catalog, AiaecInfo and StreamWork as the
 * PCM family; the enhanced spectrum comes out.
 * ------------------------------------------------------------------------------------------- */

/* Spectra: frame-major, oldest first, N frames of n_bins = fft / 2 + 1 floats per plane, real and imaginary separate, in
 * audio scale (AcStft / AcIstft defaults: int16 / 32768, sqrt-Hann, unnormalised rfft). */

/* One block's spectra, tagged by kind; frames must equal N, every plane holds frames * n_bins floats. DeepVQE-S: mic, far.
 * Align-ULCNet: err_* is the linear AEC's error spectrum (its analysis seam, not the raw capture), far_* the raw far;
 * event_count == N, oldest first. */
typedef struct AiaecFreqInput {
    AiaecInputKind kind;
    int                 frames;
    union {
        struct {
            const float *mic_re, *mic_im;
            const float *far_re, *far_im;
        } deepvqe_s;
        struct {
            const float             *err_re, *err_im;
            const float             *far_re, *far_im;
            const AiaecAlignEvent   *events;
            int                      event_count;
        } align_ulcnet;
    } u;
} AiaecFreqInput;

typedef struct AiaecFreqCtx AiaecFreqCtx;

/* As aiaec_get_mem_size(), for the _freq family; the pool holds no FFT, STFT / ISTFT or PCM scratch. */
int aiaec_get_mem_size_freq(const AiaecConfig *cfg, StreamMemReq *req);

/* As aiaec_init(), for the _freq family. */
int aiaec_init_freq(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecFreqCtx **ctx, AiaecInfo *info);

void aiaec_reset_freq(AiaecFreqCtx *ctx);
unsigned int aiaec_state_generation_freq(const AiaecFreqCtx *ctx);

/* As aiaec_preprocess() with frames for samples; the NONFINITE scan covers the four planes (N * n_bins floats each). An
 * Align-ULCNet event restarts only the model: the caller's analysers and ISTFT keep running (do not reset the ISTFT), and
 * the output equals the PCM family's through the same AcStft / AcIstft. */
int aiaec_preprocess_freq(AiaecFreqCtx *ctx, const AiaecFreqInput *input, StreamWork *work);

/* As aiaec_postprocess(), writing the block's N enhanced frames to out_re / out_im (N * n_bins floats each), for the
 * block opened by aiaec_preprocess_freq(). Through an AcIstft(AC_OLA_ACCUMULATE, ispec_scale 1) they equal the PCM family's
 * pcm_out. */
int aiaec_postprocess_freq(AiaecFreqCtx *ctx, int run_status, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* AIAEC_H */
