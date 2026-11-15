/* DeepVQE-S PCM stream: two analysis AcStft (mic, far) and one AcIstft (AC_OLA_ACCUMULATE) around lib/aiaec/deepvqe_s's
 * DeepVqePrepost lifecycle; one hop of int16 PCM per call, grid.frames_per_call N in [1, 64]. N > 1: the accelerator runs
 * once every N hops and pre_process's return value says which (1 job, 0 none); pre/post_process run every hop.
 *
 *     per hop: rc = deepvqe_stream_pre_process(ctx, mic_in, far_in, &io);             // -1: contract error
 *              if (rc == 1) { device_state_sync(io.state_generation); ai_rc = run_accelerator(&io.inputs, &io.outputs); }
 *              deepvqe_stream_post_process(ctx, ai_rc, pcm_out);
 *
 * Run status (stream_run.h): 0 = every head written; STREAM_RUN_NOT_STARTED = the device state was not touched; any other
 * value = it may be dirty. The library advances the state generation whenever the device state may no longer be the one to
 * continue from (reset, refused commit, a dirty status); each advance also empties the CCM ring. Before a run, zero the
 * accelerator's state when io.state_generation differs from the value at the last zeroing; it is written every hop, also with
 * the optional registered state.
 * A status other than 0 fails CLOSED: the batch's spectrum is zero, but the overlap-add still carries the previous hop, so
 * pcm_out is not fully zero except on the first bypass. No warm-up: a frame is ready from hop 0 (N == 1) or hop N - 1, over a
 * zero-filled CCM ring. The zero hops are the N - 1 pre-roll hops of the first batch. PCM delay is N * grid.hop samples. */

#ifndef DEEPVQE_STREAM_H
#define DEEPVQE_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_prepost.h"   /* DeepVqeGrid, DeepVqePrepostInputs/Outputs, DeepVqeStateId */

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPVQE_STREAM_DESCRIPTOR_VERSION 1u
#define DEEPVQE_STREAM_LAYOUT_VERSION     4u   /* bump with build_flags_hash on any pool / carve / descriptor change */
#define DEEPVQE_STREAM_ALIGNMENT          16u

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

/* The tensor views of one open hop and the generation the accelerator's state must match; valid until post_process. */
typedef struct DeepVqeHopIO {
    DeepVqePrepostInputs  inputs;      /* valid only when pre_process returned 1 */
    DeepVqePrepostOutputs outputs;     /* valid only when pre_process returned 1 */
    uint32_t              state_generation;   /* sampled every hop, job or not */
} DeepVqeHopIO;

/* One PCM hop in (mic_in, far_in: hop_size samples each, no HPF): analysis, pre_process_freq, frame_open if a job is due.
 * Returns 1 with io filled, 0 with io zeroed (N > 1, no job this hop; state_generation still written), -1 on a contract
 * error (NULL args, a freq_only stream, or a call before the matching post_process). */
int deepvqe_stream_pre_process(DeepVqeStream *s, const int16_t *mic_in, const int16_t *far_in,
                               DeepVqeHopIO *io);

/* Closes the hop: ai_rc is read only if pre_process returned 1 (0 commits; otherwise the batch fails CLOSED, see above). pcm_out is a
 * full hop: return 1; 0 with zeros during the N - 1 pre-roll hops (N > 1); -1 with zeros on a post_process_freq() error, or
 * -1 untouched if no pre_process is pending or the stream is freq_only (the hop stays due: post_process_freq() closes it).
 * Once per pre_process, every hop. */
int deepvqe_stream_post_process(DeepVqeStream *s, int ai_rc, int16_t *pcm_out);

/* Spectrum-domain entries, for a caller that keeps analysis and / or synthesis outside: the stream's own scale
 * (unnormalised rfft of a sqrt-Hann n_fft block of full-scale float audio, n_bins floats per plane); same pending / job /
 * return conventions as the PCM pair, which may be paired with the _freq entries across. They run on either kind of
 * stream; cfg.freq_only keeps the FFT, STFT / ISTFT and PCM scratch out of the pool. */

/* As deepvqe_stream_pre_process(), with this hop's mic and far spectra supplied (the analysis STFTs are not touched). */
int deepvqe_stream_pre_process_freq(DeepVqeStream *s, const float *mic_re, const float *mic_im,
                                    const float *far_re, const float *far_im, DeepVqeHopIO *io);

/* As deepvqe_stream_post_process(), but re / im (n_bins floats each) receive the enhanced spectrum and no ISTFT runs.
 * Returns 1 real spectrum, 0 zero spectrum (pre-roll), -1 contract error (nothing written without a matching pre_process). */
int deepvqe_stream_post_process_freq(DeepVqeStream *s, int ai_rc, float *re, float *im);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_STREAM_H */
