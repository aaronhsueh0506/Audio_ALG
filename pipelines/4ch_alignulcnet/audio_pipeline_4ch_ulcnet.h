/** Four-channel spatial pipeline with the Align-ULCNet neural post-filter as its post stage. Per hop (compile-time ULCNET_HOP samples): 4x linear AEC lanes + one shared far aligner (four_aec_front_process) -> SRP-PHAT DOA -> BF effective weights -> beamformed linear-error spectrum, used directly as the model's error frame -> the same hop's raw far through a rolling AcStft -> one UlcnetModel inference per hop from hop #0 -> Align-ULCNet synthesis (ulcnet_process.h) -> enhanced mono hop. The core runs front-only (cfg.core.enable_post = 0).
 * Timing: added latency is one compiled-grid hop (16 ms at 16 kHz/512, 10.67 ms at 48 kHz/1024). process() emits zeros for hop #0 only; hop #p (p >= 1) carries the beamformed error of input hop p-1.
 * Far input: the model receives the caller's raw `far_reference` hop (a published descriptor must carry ULCNET_FAR_RAW); every PBFDKF lane uses the shared delay ring's aligned copy to form its error.
 * Model policy: infer() runs for every emitted frame whenever a callback is installed, except during the identity reprime. No model, infer == NULL, a nonzero infer() result or a non-finite output bin passes the error spectrum through unchanged. On delay.changed (FIXED: the first transition to a usable aligned error) model->reset (if set) is called and the next AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES frames take the identity path without stepping the model, since windows in flight straddle the boundary; a second boundary re-arms the counter. reset() also calls model->reset.
 * VAD: only process_with_activity() (external VAD) exists; the auto_vad_* fields are validated but unused. A caller that derives a per-bin mask and VAD from the lane spectra of the same hop splits process_begin() into process_front() and process_beam().
 * Grid: one build serves one compile-time grid, 16 kHz / fft 512 / hop 256 or 48 kHz / fft 1024 / hop 512; other rates and core fft sizes are rejected. Lifecycle as audio_pipeline_4ch.h: caller-owned pool via get_mem_requirements()/init_ex(), heap via create(). */
#ifndef AUDIO_PIPELINE_4CH_ULCNET_H
#define AUDIO_PIPELINE_4CH_ULCNET_H

#include "audio_pipeline_4ch.h"
#include "ulcnet_process.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Emitted frames that still straddle an alignment boundary, i.e. the length of the
 * identity reprime armed at every generation change. Both branches are framed from
 * the current input hop and each frame spans two hops, so only the boundary hop's
 * frame straddles. */
enum { AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES = 1 };

#define AUDIO_PIPELINE_4CH_ULCNET_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or descriptor content, including the composed core's pool or layout. Carve order: self, core, srp, bf, bf_spectrum, bf_weights, fft. */
#define AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION 23u

/** Fixed 32-byte descriptor with the fields of FourAecNrResMemReq (4aec_nr_res.h); build_flags_hash folds in the core's hash. */
typedef struct AudioPipeline4ChUlcnetMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} AudioPipeline4ChUlcnetMemReq;

_Static_assert(
    sizeof(AudioPipeline4ChUlcnetMemReq) == 32,
    "AudioPipeline4ChUlcnetMemReq must be exactly 32 bytes");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, descriptor_version) == 0,
    "descriptor_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, layout_version) == 4,
    "layout_version offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, backend_id) == 8,
    "backend_id offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, build_flags_hash) == 12,
    "build_flags_hash offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, alignment) == 16,
    "alignment offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, reserved) == 20,
    "reserved offset");
_Static_assert(
    offsetof(AudioPipeline4ChUlcnetMemReq, bytes) == 24,
    "bytes offset");

typedef struct AudioPipeline4ChUlcnet AudioPipeline4ChUlcnet;

/* Defaults for the pre-only profile: core.fft_size = ULCNET_N_FFT and core.enable_post/enable_nr/enable_res/enable_cng/legacy_amin = 0, core.nr_mode = ANR_MODE_BALANCED (required sentinel), auto_vad_* at the audio_pipeline_4ch defaults. These post-only fields are rejected at init if changed; the pre-stage fields (delay_*, aec_preset, filter_length, capture_proxy_channel, max_delay_ms) are live. */
AudioPipeline4ChConfig audio_pipeline_4ch_ulcnet_default_config(void);

/** Queries the 16-byte-aligned caller pool required by cfg; no allocation. Returns 0 on success. */
int audio_pipeline_4ch_ulcnet_get_mem_requirements(
    const AudioPipeline4ChConfig* cfg,
    AudioPipeline4ChUlcnetMemReq* out);

/** Places the pipeline in caller-owned memory that satisfies the queried alignment and size; no heap, a dirty pool is accepted. The model callback starts unset (identity); see set_model(). */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg);

/** init plus the stale-descriptor gate of four_aec_nr_res_init_ex(); NULL expected is the same as init(). */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_init_ex(
    void* mem,
    size_t bytes,
    const AudioPipeline4ChConfig* cfg,
    const AudioPipeline4ChUlcnetMemReq* expected);

/** Heap convenience over get_mem_requirements() + init(); NULL on invalid config or allocation failure. */
AudioPipeline4ChUlcnet* audio_pipeline_4ch_ulcnet_create(
    const AudioPipeline4ChConfig* cfg);

/** Installs (or clears, with NULL) the NPU runtime callbacks, copied by value; model->user must stay valid until replaced or destroy. No reset is issued: install a freshly reset runtime, or call reset() after swapping mid-stream. A model with io_descriptor != NULL (which must outlive the pipeline) must match the raw-far production ABI. Returns 0, or nonzero on a NULL/destroyed pipeline or invalid descriptor. */
int audio_pipeline_4ch_ulcnet_set_model(
    AudioPipeline4ChUlcnet* p,
    const UlcnetModel* model);

/** Completes one hop with activity from the product VAD. microphones_interleaved is [hop][4]; far_reference and out are [hop]. vad_external gates SRP observation and freezes BF adaptation. out is zeros for hop #0, then the enhanced (or identity) beamformed error one hop behind the input; every hop is one inference. Returns FOUR_AEC_NR_RES_OK or the core's status codes (the wrapper resets itself on internal DSP errors). */
int audio_pipeline_4ch_ulcnet_process_with_activity(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external,
    float* out);

/** The same hop in phases, for a caller that runs the network itself (a reset callback, if set, is still called):
 *     audio_pipeline_4ch_ulcnet_process_begin(p, mics, far, vad);  // lanes, DOA, BF, far analysis
 *     if (audio_pipeline_4ch_ulcnet_nn_begin(p, &job) > 0) {       // a frame to run
 *         rc = run_network(&job);                                  // 0 = all bins written
 *         audio_pipeline_4ch_ulcnet_nn_end(p, rc);                 // accept, or the identity
 *     }
 *     audio_pipeline_4ch_ulcnet_process_end(p, out);               // WOLA, one hop
 * nn_begin returns 1 with the frame in *job (BF spectrum as the error frame, the same hop's raw far), 0 when the hop has no model frame (identity reprime), or a negative FOUR_AEC_NR_RES_* status. nn_end accepts the output only when rc == 0 and every bin is finite. A frame left open or never fetched is the identity. */
int audio_pipeline_4ch_ulcnet_process_begin(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    int vad_external);
int audio_pipeline_4ch_ulcnet_nn_begin(AudioPipeline4ChUlcnet* p,
                                       UlcnetNnJob* job);
int audio_pipeline_4ch_ulcnet_nn_end(AudioPipeline4ChUlcnet* p, int model_rc);
int audio_pipeline_4ch_ulcnet_process_end(AudioPipeline4ChUlcnet* p, float* out);

/** For a caller that derives the beamformer mask and VAD from the lane spectra of the same hop, process_begin() splits in two:
 *     audio_pipeline_4ch_ulcnet_process_front(p, mics, far, &front);   // lanes, far analysis
 *     ... mask, vad_raw, vad_out from front.lane_spectra[0] ...
 *     audio_pipeline_4ch_ulcnet_process_beam(p, &front, vad_raw, vad_out, mask);  // DOA, BF
 * process_begin(p, mics, far, v) is process_front followed by process_beam(front, v, v, NULL). */
typedef struct AudioPipeline4ChUlcnetFront {
    int n_freqs;
    /* Read-only AEC error spectra of this hop, one per channel, n_freqs bins; valid
     * until the next process_front()/process_begin()/reset()/destroy(). */
    const Complex* lane_spectra[FOUR_AEC_NR_RES_CHANNELS];
} AudioPipeline4ChUlcnetFront;

/** First half of the hop: shared delay, the four AEC lanes, model-reset and reprime bookkeeping, and the far analysis; fills *front. Afterwards only process_beam() (or reset/destroy) advances the hop. Returns FOUR_AEC_NR_RES_OK, INVALID_ARGUMENT, SEQUENCE_ERROR (a hop is already in flight) or the core's status codes. */
int audio_pipeline_4ch_ulcnet_process_front(
    AudioPipeline4ChUlcnet* p,
    const float* microphones_interleaved,
    const float* far_reference,
    AudioPipeline4ChUlcnetFront* front);

/** Second half of the hop: SRP-PHAT DOA and the beamformer on the lane spectra of `front`; the BF output becomes the model's error frame.
 *   vad_raw  0/1  SRP analyses this hop only when 1 (and the update interval allows).
 *   vad_out  0/1  DOA smoother activity input; 1 freezes BF adaptation.
 *   mask     NULL, or n_freqs ints (read during this call): SRP uses bins with mask[f] != 0 and BF freezes adaptation there; NULL = all bins used, none frozen.
 * Returns FOUR_AEC_NR_RES_OK, INVALID_ARGUMENT (nothing changed) or SEQUENCE_ERROR (no front half pending). */
int audio_pipeline_4ch_ulcnet_process_beam(
    AudioPipeline4ChUlcnet* p,
    const AudioPipeline4ChUlcnetFront* front,
    int vad_raw,
    int vad_out,
    const int* mask);

/** DOA and BF state of the most recent process_beam()/process_begin(), valid until the
 * next hop's beam half. */
typedef struct AudioPipeline4ChUlcnetBeamState {
    float doa_raw_rad;      /* fresh SRP result of that hop; NaN when SRP did not run  */
    float doa_smooth_rad;   /* smoothed, held DOA given to BF; NaN until the first valid one */
    float doa_used_rad;     /* DOA the BF steered to; NaN when it used none          */
    int   bf_adaptive;     /* 1 when BF adapted its weights on that hop, else 0      */
} AudioPipeline4ChUlcnetBeamState;

/** Returns 0 on success, nonzero on a NULL out or NULL/destroyed handle. */
int audio_pipeline_4ch_ulcnet_beam_state(
    const AudioPipeline4ChUlcnet* p,
    AudioPipeline4ChUlcnetBeamState* out);

/** Stage timings (microseconds) of the core's most recent hop. A NULL/destroyed
 * handle zeroes *out; `out` must not be NULL. */
void audio_pipeline_4ch_ulcnet_last_core_timing(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResLastTiming* out);

/** Resets all core/SRP/BF/ULCNet state and calls model->reset (if set); a pending reprime is dropped and the model installation is kept. */
void audio_pipeline_4ch_ulcnet_reset(AudioPipeline4ChUlcnet* p);

/** Destroys an instance: idempotent for caller-owned memory (the pool is not freed); a heap-created handle frees its pool (call once). */
void audio_pipeline_4ch_ulcnet_destroy(AudioPipeline4ChUlcnet* p);

int audio_pipeline_4ch_ulcnet_hop_size(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_fft_size(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_n_freqs(const AudioPipeline4ChUlcnet* p);
int audio_pipeline_4ch_ulcnet_sample_rate(const AudioPipeline4ChUlcnet* p);

/** Copies the shared delay state of the most recent successful process call. Returns 0, or nonzero on a NULL out or NULL/destroyed handle. */
int audio_pipeline_4ch_ulcnet_last_delay(
    const AudioPipeline4ChUlcnet* p,
    FourAecNrResDelayState* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_4CH_ULCNET_H */
