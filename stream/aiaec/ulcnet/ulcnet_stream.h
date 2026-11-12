/* Align-ULCNet PCM stream: two analysis AcStft (linear error, raw far), one AcIstft (AC_OLA_ACCUMULATE), lib/aiaec/ulcnet's
 * UlcnetPrepost lifecycle and alignment reset / reprime; one hop of PCM per call, no HPF. A post-filter BEHIND a linear
 * AEC: the inputs are its linear-error hop and the raw far, never raw mic.
 *
 *     per hop: pre_rc = ulcnet_stream_pre_process(ctx, err_in, far_in, delay, &io);   // -1: contract error
 *              if (pre_rc == 1) { device_state_sync(io.state_generation); ai_rc = run_accelerator(&io.inputs, &io.outputs);
 *                                 if (ai_rc != 0) device_state_refill_zero(); }
 *              ulcnet_stream_post_process(ctx, ai_rc, pcm_out);
 *
 * Alignment: delay.changed, or delay.solid going false -> true after an earlier hop, resets the stream and arms
 * ULCNET_STREAM_REPRIME_FRAMES identity hops (the hop is bypassed, pre_process returns 0). Fail-OPEN: a bypassed frame is the
 * linear-error spectrum unchanged.
 *
 * Packet streams (cfg.frames_per_call N >= 2, graph layout ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION): each packet pre_process is
 * one accelerator job (return 1) with one post_process, N * hop samples out (_freq: N * n_bins per plane); the per-hop entries
 * refuse such a stream and the packet entries a one-hop stream. delay[k] is hop k's event, oldest first, edge rule as above
 * with the previous solid carried across packets; the graph resets the model before an event hop and bypasses it (generation
 * unchanged). A failed or non-finite job passes all N error spectra through. */

#ifndef ULCNET_STREAM_H
#define ULCNET_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_prepost.h"   /* UlcnetGrid, UlcnetModelIoInputs/Outputs */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_STREAM_DESCRIPTOR_VERSION 1u
#define ULCNET_STREAM_LAYOUT_VERSION     2u   /* bump with build_flags_hash on any pool/carve/descriptor change */
#define ULCNET_STREAM_ALIGNMENT          16u
/* Identity hops after an alignment boundary; equals ULCNET_NN_STAGE_REPRIME_FRAMES (pipelines/ulcnet/ulcnet_nn_stage.h). */
#define ULCNET_STREAM_REPRIME_FRAMES     1u

/* Alignment events of the hop from the caller's linear AEC (the mapping of UlcnetNnDelay, ulcnet_nn_stage.h). */
typedef struct UlcnetStreamDelay {
    int changed;
    int solid;
} UlcnetStreamDelay;

typedef struct UlcnetStreamConfig {
    UlcnetGrid grid;   /* ulcnet_grid_make() */
    /* 0 or 1: one model hop per call. 2..ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL: a packet stream, N hops per accelerator job. */
    int frames_per_call;
    /* Optional borrowed state registration, all four or none (see ulcnet_prepost.h): every pointer non-NULL with a
     * nonzero count, or all NULL / 0 (the accelerator owns its state). Only on / off enters build_flags_hash. */
    float  *key_history;
    size_t  key_history_elements;
    float  *value_history;
    size_t  value_history_elements;
    float  *logit_history;
    size_t  logit_history_elements;
    float  *h_gru;
    size_t  gru_hidden_elements;
} UlcnetStreamConfig;

/* Fixed 32-byte descriptor; init refuses a pool sized for another grid, registration mode or build. */
typedef struct UlcnetStreamMemReq {
    uint32_t descriptor_version;  /* ULCNET_STREAM_DESCRIPTOR_VERSION        */
    uint32_t layout_version;      /* ULCNET_STREAM_LAYOUT_VERSION            */
    uint32_t reserved0;           /* must be 0                                */
    uint32_t build_flags_hash;    /* FNV-1a-32 over grid, state-registration mode and the UlcnetPrepost hash */
    uint32_t alignment;           /* ULCNET_STREAM_ALIGNMENT                  */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} UlcnetStreamMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(UlcnetStreamMemReq) == 32,
               "UlcnetStreamMemReq must stay 32 bytes");
#endif
#endif

typedef struct UlcnetStream UlcnetStream;

/* Exact pool size for cfg. -1 (req untouched) on NULL, an invalid grid, or a partial state registration. */
int ulcnet_stream_get_mem_size(const UlcnetStreamConfig *cfg, UlcnetStreamMemReq *req);

/* Constructs in caller memory (16-byte aligned, >= get_mem_size() bytes), starts reset; no heap, no stdio. NULL on a bad
 * config or a misaligned / undersized pool. */
UlcnetStream *ulcnet_stream_init(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes);

/* As ulcnet_stream_init, refusing a pool carved for another grid, registration mode or layout; `expected` NULL skips it. */
UlcnetStream *ulcnet_stream_init_ex(const UlcnetStreamConfig *cfg, void *pool, size_t pool_bytes,
                                    const UlcnetStreamMemReq *expected);

/* Back to the start: framing and UlcnetPrepost reset (registered state zeroed, generation advanced), reprime countdown
 * cleared. The alignment-edge detector runs the same operation. */
void ulcnet_stream_reset(UlcnetStream *s);

/* Generation of the recurrent state; 0 for NULL. */
uint32_t ulcnet_stream_state_generation(const UlcnetStream *s);

const UlcnetGrid *ulcnet_stream_grid(const UlcnetStream *s);   /* NULL for NULL */
int ulcnet_stream_hop_size(const UlcnetStream *s);             /* samples per PCM hop; -1 on NULL */
int ulcnet_stream_frames_per_call(const UlcnetStream *s);      /* 1 for a one-hop stream; -1 on NULL */
/* The graph layout version this instance is bound to (ulcnet_prepost_descriptor()); -1 on NULL. */
int ulcnet_stream_layout_version(const UlcnetStream *s);

/* Hops of the last pre_process at which the stream restarted its own analysis framing (bit k = hop k; bit 0 only for a
 * one-hop stream). Set by the f32 / int16 entries; 0 after the freq entries, a reset and for NULL. A caller with its own
 * synthesis resets its ISTFT right before pushing each such hop. */
uint64_t ulcnet_stream_restart_mask(const UlcnetStream *s);

/* The tensor views of one open hop and the generation the accelerator's state must match. Valid only when pre_process
 * returned 1; zeroed on a reprime hop. */
typedef struct UlcnetHopIO {
    UlcnetModelIoInputs  inputs;
    UlcnetModelIoOutputs outputs;
    uint32_t             state_generation;
} UlcnetHopIO;

/* One hop in as full-scale float err_in and far_in (hop_size samples each): alignment-edge check (may reset), analysis,
 * pre_process_freq, then frame_bypass (reprime hop: io zeroed, returns 0) or frame_open (returns 1). -1: contract error. The
 * int16 entry below round-trips float lanes through int16 (up to 4 LSB of PCM). */
int ulcnet_stream_pre_process_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                  UlcnetStreamDelay delay, UlcnetHopIO *io);

/* int16 wrapper: converts (/32768.0f), then the f32 path; same returns. */
int ulcnet_stream_pre_process(UlcnetStream *s, const int16_t *err_in, const int16_t *far_in,
                              UlcnetStreamDelay delay, UlcnetHopIO *io);

/* Closes the hop: ai_rc is read only if pre_process returned 1 (0 commits, else bypass, fail OPEN); a reprime hop is already
 * bypassed. pcm_out is a full hop (return 1). An alignment-event hop restarts the stream: its pcm_out is exactly zero and
 * the previous hop's audio is not output. -1 with zeros on a post_process_freq() error. Once per pre_process. */
int ulcnet_stream_post_process(UlcnetStream *s, int ai_rc, int16_t *pcm_out);

/* N hops of full-scale float err_in and far_in (N * hop samples each), analysed hop by hop, then the job; an event hop also
 * restarts the stream's framing (exactly zero output, the previous hop's audio lost; ulcnet_stream_restart_mask()), so the PCM
 * equals N one-hop calls. Returns 1, or -1 on a contract error (NULL args, a one-hop stream, a call before the matching post). */
int ulcnet_stream_pre_process_packet_f32(UlcnetStream *s, const float *err_in, const float *far_in,
                                         const UlcnetStreamDelay *delay, UlcnetHopIO *io);

/* As the f32 packet entry with the N hops' spectra supplied (frame-major, N * n_bins per plane, the stream's scale); restarts
 * no framing. ulcnet_stream_reset() restarts everything, generation included. */
int ulcnet_stream_pre_process_packet_freq(UlcnetStream *s, const float *err_re, const float *err_im,
                                          const float *far_re, const float *far_im,
                                          const UlcnetStreamDelay *delay, UlcnetHopIO *io);

/* As ulcnet_stream_pre_process_f32(), with this hop's linear-error and raw-far spectra supplied (the stream's scale:
 * unnormalised rfft of a sqrt-Hann n_fft block of full-scale float audio, n_bins floats per plane). An edge restarts only the
 * model (state, one identity hop); the caller's analysers and ISTFT are not restarted. */
int ulcnet_stream_pre_process_freq(UlcnetStream *s, const float *err_re, const float *err_im,
                                   const float *far_re, const float *far_im,
                                   UlcnetStreamDelay delay, UlcnetHopIO *io);

/* As ulcnet_stream_post_process(), but re / im (n_bins floats each) receive the enhanced spectrum and no ISTFT runs.
 * Returns 1, or -1 on a contract error (nothing written without a matching pre_process). */
int ulcnet_stream_post_process_freq(UlcnetStream *s, int ai_rc, float *re, float *im);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_STREAM_H */
