/* AIAEC customer interface (DeepVQE-S / Align-ULCNet). The caller sets {sr, fft, hop_in}; the product build binds
 * one model (aiaec_product.h) and the compiled-in catalog resolves N = hop_in / (fft / 2), the delay depth and the
 * graph. The accelerator (open the graph, move tensors, run, read back) is the sample's:
 *
 *     AiaecConfig cfg = { sr, fft, hop_in };
 *     aiaec_get_mem_size(&cfg, &req);  ...allocate...  aiaec_init(&cfg, pool, req.bytes, &ctx, &info);   // no SDK
 *     ...open info.graph_path, check its ports against the sample's model configuration, seed the device state...
 *     per block of hop_in samples:
 *         aiaec_preprocess(ctx, &in, &work);        // work.action: SUBMIT or BYPASS
 *         ...SUBMIT: the sample's AI3 run on work; BYPASS: no accelerator call...
 *         aiaec_postprocess(ctx, ai_rc, pcm_out);   // once per successful preprocess, failed run or BYPASS included
 *     after the last real block (zero-padded to hop_in): info.tail_zero_blocks all-zero blocks through the same
 *     sequence, then drop info.trim_samples from the start of the output.
 *
 * Align-ULCNet takes the linear AEC's float residual, the raw far of the same hop and one AiaecAlignEvent per model
 * hop: a pipeline with its own AEC passes them directly; a standalone sample links front/ (libaiaec_front.a) and
 * calls aiaec_front_preprocess(). A configuration with no registered grid, hop_in or graph is refused by
 * aiaec_get_mem_size() and aiaec_init() before any SDK call. The PCM (no suffix), "_spec" and "_freq" families
 * share configuration, catalog, AiaecInfo and AiaecWork. */

#ifndef AIAEC_H
#define AIAEC_H

#include <stddef.h>
#include <stdint.h>

#include "stream_mem.h"    /* StreamMemReq */

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
    AIAEC_ERR_INPUT_SIZE = -9         /* samples != hop_in, or (Align-ULCNet) event_count != N (before any state write) */
};

/* Short text for a status code, for a log line. Never NULL. */
const char *aiaec_status_text(int code);

typedef struct AiaecTensorIn {
    const float *data;
    size_t elements;
    unsigned int w, h, c, b;   /* tensor shape, w innermost (the ONNX logical [b,c,h,w]) */
} AiaecTensorIn;

typedef struct AiaecTensorOut {
    float *data;
    size_t elements;
    unsigned int w, h, c, b;
} AiaecTensorOut;

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

#define AIAEC_MAX_INPUT_TENSORS  5   /* DeepVQE-S: mic, far. Align-ULCNet: 4 spectra, plus frame_reset at N > 1 */
#define AIAEC_MAX_OUTPUT_TENSORS 1   /* DeepVQE-S: CCM taps. Align-ULCNet: complex mask, planar */

/* Whether this block carries an accelerator job. A BYPASS block's AiaecWork is zero except the synthesis restart fields. */
typedef enum AiaecAction {
    AIAEC_BYPASS = 0,   /* no accelerator call (an Align-ULCNet N = 1 event hop); aiaec_postprocess() is still called once */
    AIAEC_SUBMIT = 1    /* run the accelerator, pass the outcome to aiaec_postprocess() */
} AiaecAction;

/* Per-block tensor views, valid until the paired aiaec_postprocess(): signals then the head in graph order (the recurrent
 * state stays in the accelerator). DeepVQE-S: mic, far [1,2,N,bins]; CCM taps [1,N,bins,18]. Align-ULCNet: error_mag,
 * far_mag, error_cos, error_sin [1,1,N,bins], at N > 1 frame_reset [1,1,1,N]; complex mask [1,2,N,bins]. */
typedef struct AiaecWork {
    AiaecAction action;
    AiaecTensorIn    inputs[AIAEC_MAX_INPUT_TENSORS];
    int              input_count;
    AiaecTensorOut   outputs[AIAEC_MAX_OUTPUT_TENSORS];
    int              output_count;
    /* Sampled every SUBMIT block: zero the accelerator's state whenever it differs from the last value seen, and before
     * the next SUBMIT after a failed or rejected run. 0 on a BYPASS block. */
    unsigned int     state_generation;
    /* Align-ULCNet PCM and _spec: bit k = model hop k (oldest first, N up to 64) restarted the stream's framing with the
     * model state; synthesis_restarted = any bit set. A _spec caller calls ac_istft_reset right before that hop's frame.
     * Kept on BYPASS blocks; always 0 for _freq and DeepVQE-S. */
    uint64_t         synthesis_restart_mask;
    int              synthesis_restarted;
} AiaecWork;

/* ---------------------------------------------------------------------------------------------
 * What one init selects: graph, trim / queue / tail length.
 * ------------------------------------------------------------------------------------------- */

/* The graph and stream timing the init resolved for this configuration. */
typedef struct AiaecInfo {
    const char  *graph_path;           /* model.bin for this configuration; never NULL here */
    int          trim_samples;         /* samples to drop from the start of the final PCM (after the caller's ISTFT for
                                        * _spec/_freq), all families: DeepVQE-S N * model_hop, Align-ULCNet model_hop */
    int          queue_frames;         /* model-queue frames between a frame in and its enhanced frame out:
                                        * DeepVQE-S N - 1, Align-ULCNet 0 */
    int          tail_zero_blocks;     /* all-zero hop_in blocks after the last real block: ceil(trim_samples / hop_in); no
                                        * family flushes: _spec also pushes them through the caller's ISTFT, _freq's analyser
                                        * must first deliver the final frames; queue_frames zero hops alone lose the last hop */
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

/* One block in. AIAEC_OK: work is filled; call aiaec_postprocess() exactly once before the next preprocess. Errors
 * (NULL / SEQUENCE, then INPUT_KIND / INPUT_SIZE) come before any state change and open no block. Align-ULCNet: an event hop
 * also restarts the stream's framing, so its pcm_out is exactly zero and the previous hop's audio is lost (one hop per event). */
int aiaec_preprocess(AiaecCtx *ctx, const AiaecInput *input, AiaecWork *work);

/* Closes the block; infer_status is read on SUBMIT only; ERR_SEQUENCE (no open block) writes nothing. pcm_out gets hop_in
 * samples. infer_status != 0 or a non-finite output bypasses the block: DeepVQE-S silence (fails CLOSED), Align-ULCNet the error
 * spectrum (fails OPEN). Only a non-finite output also zeroes the model state and advances the generation. */
int aiaec_postprocess(AiaecCtx *ctx, int infer_status, int16_t *pcm_out);

/* ---------------------------------------------------------------------------------------------
 * _spec (PCM in, the caller runs its own ISTFT) and _freq (AiaecFreqInput spectra in, no STFT / ISTFT) families: the
 * same hop_in, catalog, AiaecInfo and AiaecWork as the PCM family; the enhanced spectrum comes out.
 * ------------------------------------------------------------------------------------------- */

/* Spectra: frame-major, oldest first, N frames of n_bins = fft / 2 + 1 floats per plane, real and imaginary separate, in
 * audio scale (AcStft / AcIstft defaults: int16 / 32768, sqrt-Hann, unnormalised rfft). */
typedef struct AiaecSpecCtx AiaecSpecCtx;

/* As aiaec_get_mem_size(), for the _spec family. */
int aiaec_get_mem_size_spec(const AiaecConfig *cfg, StreamMemReq *req);

/* As aiaec_init(), for the _spec family. */
int aiaec_init_spec(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecSpecCtx **ctx, AiaecInfo *info);

/* Clears the family's own history (_spec: analysers, model state; _freq: model state) and advances the generation. */
void aiaec_reset_spec(AiaecSpecCtx *ctx);
unsigned int aiaec_state_generation_spec(const AiaecSpecCtx *ctx);

/* As aiaec_preprocess(). Align-ULCNet event hop: N = 1 a BYPASS block; N > 1 one SUBMIT job (frame_reset 1.0, generation
 * unchanged). */
int aiaec_preprocess_spec(AiaecSpecCtx *ctx, const AiaecInput *input, AiaecWork *work);

/* As aiaec_postprocess(), writing the block's N enhanced frames to out_re / out_im (N * n_bins floats each). Through an
 * AcIstft(AC_OLA_ACCUMULATE, ispec_scale 1) they equal the PCM family's pcm_out (DeepVQE-S's first block starts with N - 1
 * zero frames; Align-ULCNet _spec: ISTFT reset at the synthesis_restart_mask hops, one hop of audio lost per event). */
int aiaec_postprocess_spec(AiaecSpecCtx *ctx, int infer_status, float *out_re, float *out_im);

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

/* As aiaec_get_mem_size(), for the _freq family. */
int aiaec_get_mem_size_freq(const AiaecConfig *cfg, StreamMemReq *req);

/* As aiaec_init(), for the _freq family. */
int aiaec_init_freq(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecFreqCtx **ctx, AiaecInfo *info);

void aiaec_reset_freq(AiaecFreqCtx *ctx);
unsigned int aiaec_state_generation_freq(const AiaecFreqCtx *ctx);

/* As aiaec_preprocess() with frames for samples: INPUT_KIND, INPUT_SIZE (frames or event_count != N), then NULL for a NULL plane.
 * An Align-ULCNet event restarts only the model: the caller's analysers and ISTFT keep running (do not reset the ISTFT),
 * and the output equals _spec's on every other hop. */
int aiaec_preprocess_freq(AiaecFreqCtx *ctx, const AiaecFreqInput *input, AiaecWork *work);

/* As aiaec_postprocess_spec(), for the block opened by aiaec_preprocess_freq(). */
int aiaec_postprocess_freq(AiaecFreqCtx *ctx, int infer_status, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* AIAEC_H */
