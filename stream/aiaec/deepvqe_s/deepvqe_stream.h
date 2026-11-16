/* DeepVQE-S PCM stream: two analysis AcStft (mic, far) and one AcIstft (AC_OLA_ACCUMULATE) around lib/aiaec/deepvqe_s's
 * DeepVqePrepost per-frame functions; the stream sequences them. One call carries grid.frames_per_call = N frames, N in [1, 64], and one accelerator job:
 *
 *     rc = deepvqe_stream_pre_process_packet(ctx, mic_in, far_in, &io);   // N * hop int16 each; -1: contract error
 *     device_state_sync(io.state_generation); ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *     deepvqe_stream_post_process_packet(ctx, ai_rc, pcm_out);              // N * hop samples
 *
 * pre analyses the N hops, runs pre_frame on each and publishes the views; post closes the call (ai_rc 0 commits, otherwise
 * it fails closed) and synthesises its N frames: a call's output is its own N frames, nothing is queued across calls, and there is no start-up
 * case. The PCM delay is one hop of framing whatever N. The one-hop entries (pre_process, post_process, _freq) are the N == 1
 * form of the same sequence and return -1 on a stream with N > 1.
 *
 * Run status (stream_run.h): 0 = every head written; STREAM_RUN_NOT_STARTED = the device state was not touched; any other
 * value = it may be dirty. The library advances the state generation whenever the device state may no longer be the one to
 * continue from (reset, refused commit, a dirty status); each advance also empties the CCM ring. Before a run, zero the
 * accelerator's state when io.state_generation differs from the value at the last zeroing; it is written every call, also
 * with the optional registered state.
 * A status other than 0 fails CLOSED: the batch's spectra are zero, but the overlap-add still carries the previous hop, so
 * pcm_out is not fully zero except on the first bypass. */

#ifndef DEEPVQE_STREAM_H
#define DEEPVQE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_prepost.h"   /* DeepVqeGrid, DeepVqePrepostInputs/Outputs, DeepVqeStateId */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_STREAM_DESCRIPTOR_VERSION 1u
#define DEEPVQE_STREAM_LAYOUT_VERSION     6u   /* bump with build_flags_hash on any pool / carve / descriptor change */
#define DEEPVQE_STREAM_ALIGNMENT          16u

/* Range of the _freq input planes: every |bin| at most the float with these IEEE bits (FLT_MAX); a NaN, an Inf or anything
 * beyond is refused with DEEPVQE_STREAM_ERR_RANGE before the stream moves. */
#define DEEPVQE_STREAM_BIN_LIMIT_BITS     0x7F7FFFFFu
#define DEEPVQE_STREAM_ERR_RANGE          (-2)

typedef struct DeepVqeStreamConfig {
    DeepVqeGrid grid;   /* deepvqe_grid_make(); frames_per_call in [1, 64] */
    /* Non-zero: a spectrum-only stream. Only the _freq entries run (the PCM entries return -1 and change nothing) and the
     * pool holds no FFT, STFT / ISTFT or PCM scratch. 0: both entry families. */
    int freq_only;
    /* Optional borrowed state registration by DeepVqeStateId (see deepvqe_prepost.h): every pointer non-NULL with a
     * nonzero count, or all NULL / 0 (the accelerator owns its state). Not in the pool size; the mode is in the hash. */
    float  *state[DEEPVQE_STATE_COUNT];
    size_t  state_elements[DEEPVQE_STATE_COUNT];
} DeepVqeStreamConfig;

/* Fixed 32-byte descriptor; init refuses a pool sized for another grid, registration mode, freq_only or build. */
typedef struct DeepVqeStreamMemReq {
    uint32_t descriptor_version;  /* DEEPVQE_STREAM_DESCRIPTOR_VERSION       */
    uint32_t layout_version;      /* DEEPVQE_STREAM_LAYOUT_VERSION           */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, state-registration mode, freq_only and the DeepVqePrepost hash */
    uint32_t alignment;           /* DEEPVQE_STREAM_ALIGNMENT                 */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DeepVqeStreamMemReq;

typedef struct DeepVqeStream DeepVqeStream;

/* Exact pool size for cfg. -1 (req untouched) on NULL, an invalid grid, or a partial state registration. */
int deepvqe_stream_get_mem_size(const DeepVqeStreamConfig *cfg, DeepVqeStreamMemReq *req);

/* Constructs in caller memory (16-byte aligned, >= get_mem_size() bytes), starts reset; no heap, no stdio. NULL on a bad
 * config or a misaligned / undersized pool. */
DeepVqeStream *deepvqe_stream_init(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As deepvqe_stream_init, refusing a pool carved for another grid, registration mode, freq_only or layout; `expected` NULL
 * skips it. */
DeepVqeStream *deepvqe_stream_init_ex(const DeepVqeStreamConfig *cfg, void *pool, size_t pool_bytes,
                                     const DeepVqeStreamMemReq *expected);

/* Back to the start: framing (when the stream has it) and DeepVqePrepost reset (registered state zeroed, generation
 * advanced, CCM ring cleared). */
void deepvqe_stream_reset(DeepVqeStream *s);

/* Generation of the recurrent state; 0 for NULL. */
uint32_t deepvqe_stream_state_generation(const DeepVqeStream *s);

int deepvqe_stream_hop_size(const DeepVqeStream *s);              /* samples per PCM hop; -1 on NULL */

/* The tensor views of one open batch and the generation the accelerator's state must match; valid until post_process. */
typedef struct DeepVqeHopIO {
    DeepVqePrepostInputs  inputs;
    DeepVqePrepostOutputs outputs;
    uint32_t              state_generation;
} DeepVqeHopIO;

/* N hops in (mic_in, far_in: N * hop_size samples each, oldest first, no HPF): analysis, pre_frame per frame, views.
 * Returns 1 with io filled, -1 on a contract error (NULL args, a freq_only stream, or a call before the matching post_process). */
int deepvqe_stream_pre_process_packet(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in, DeepVqeHopIO *io);

/* Closes the call (ai_rc 0 commits, else it fails closed), writes its N frames, oldest first, to pcm_out (N * hop_size
 * samples). Returns 1; -1 with zeros on a synthesis error, -1 untouched with no pre_process pending or on a freq_only stream. */
int deepvqe_stream_post_process_packet(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);

/* Spectrum-domain entries, for a caller that keeps analysis and / or synthesis outside: the stream's own scale (unnormalised
 * rfft of a sqrt-Hann n_fft block of full-scale float audio), N frames of n_bins floats per plane, oldest first; the same
 * conventions as the PCM pair, which may be paired with them. cfg.freq_only keeps the FFT, STFT / ISTFT and PCM scratch out. */

/* As deepvqe_stream_pre_process_packet(), with the N frames' mic and far spectra supplied (the analysis STFTs are not touched).
 * DEEPVQE_STREAM_ERR_RANGE (-2), nothing changed, when a bin of the four planes is not finite; -1 is a contract error. */
int deepvqe_stream_pre_process_packet_freq(DeepVqeStream *s, const float *mic_re, const float *mic_im,
                                           const float *far_re, const float *far_im, DeepVqeHopIO *io);

/* As deepvqe_stream_post_process_packet(), but re / im (N * n_bins floats each) receive the enhanced spectra and no ISTFT runs. */
int deepvqe_stream_post_process_packet_freq(DeepVqeStream *s, int ai_rc, float *re, float *im);

/* The packet entries of a one-frame stream (N == 1); -1, nothing touched, when N > 1. */
int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in, DeepVqeHopIO *io);
int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);
int deepvqe_stream_pre_process_freq(DeepVqeStream *s, const float *mic_re, const float *mic_im,
                                    const float *far_re, const float *far_im, DeepVqeHopIO *io);
int deepvqe_stream_post_process_freq(DeepVqeStream *s, int ai_rc, float *re, float *im);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_STREAM_H */
