/**
 * aec_4ch/4aec_nr_res.h
 *
 * Four-microphone linear-AEC front end with one shared delay matcher and one
 * post-beam NR/RES path.
 *
 * The external owner supplies SRP-PHAT/GSC. This module deliberately owns:
 *
 *   1 shared DelayAec3 matcher
 *   4 independent linear AEC adaptive filters
 *   1 coherent post-beam residual suppressor
 *   1 mono MMSE-LSA denoiser
 *   1 final inverse FFT / overlap-add path
 *
 * It does not select one microphone's RES gain and does not replicate NR/RES
 * per channel.
 */
#ifndef FOUR_AEC_NR_RES_H
#define FOUR_AEC_NR_RES_H

#include <stdint.h>

#include "aec.h"
#include "mmse_lsa_denoiser.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FOUR_AEC_NR_RES_CHANNELS 4

typedef enum FourAecNrResStatus {
    FOUR_AEC_NR_RES_OK = 0,
    FOUR_AEC_NR_RES_INVALID_ARGUMENT = -1,
    FOUR_AEC_NR_RES_SEQUENCE_ERROR = -2,
    FOUR_AEC_NR_RES_DSP_ERROR = -3
} FourAecNrResStatus;

typedef struct FourAecNrResConfig {
    int sample_rate;              /* 16000 or 48000                         */
    int fft_size;                 /* 0=rate default; 512 @16k, 1024 @48k   */
    int filter_length;            /* 0=rate default                         */
    int capture_proxy_channel;    /* shared matcher input, [0,3]            */
    float max_delay_ms;           /* shared reference delay-line capacity    */
    AecPreset aec_preset;
    MmseLsaNrMode nr_mode;
    int enable_cng;               /* bool                                   */
    int legacy_amin;              /* bool: do not fold R2 into NR prior      */
} FourAecNrResConfig;

typedef struct FourAecNrResDelayState {
    int delay_samples;
    float confidence;
    int solid;
    int changed;
    uint64_t estimator_calls;
    int estimator_updates;
} FourAecNrResDelayState;

/**
 * Opaque ownership/ordering token returned by process_pre().
 * Callers must copy it without modification and return it to process_post().
 */
typedef struct FourAecNrResFrameToken {
    uint64_t frame_index;
    uint64_t generation;
    uintptr_t owner_cookie;
} FourAecNrResFrameToken;

typedef struct FourAecNrResPreFrame {
    FourAecNrResFrameToken token;
    FourAecNrResDelayState delay;
    int hop_size;
    int n_channels;               /* always FOUR_AEC_NR_RES_CHANNELS        */

    /**
     * Read-only interleaved linear-AEC output [hop_size][4]:
     * linear_interleaved[sample * 4 + channel].
     *
     * Valid until process_post(), reset(), or destroy(). Only one pre frame
     * may be in flight.
     */
    const float* linear_interleaved;
} FourAecNrResPreFrame;

typedef struct FourAecNrRes FourAecNrRes;

/** Fill a validated default: 16 kHz, 512/256, balanced AEC/NR, CNG on. */
void four_aec_nr_res_config_defaults(FourAecNrResConfig* cfg,
                                     int sample_rate);

/**
 * Heap construction. All memory is allocated here; process_pre/post perform
 * no allocation. Returns NULL for an invalid config or allocation failure.
 */
FourAecNrRes* four_aec_nr_res_create(const FourAecNrResConfig* cfg);

/**
 * Reset all delay/AEC/NR/RES/OLA state. Invalidates a pending pre-frame token.
 */
void four_aec_nr_res_reset(FourAecNrRes* self);

/** Destroy a heap-created instance. NULL-safe; call once for a live handle. */
void four_aec_nr_res_destroy(FourAecNrRes* self);

/**
 * Run the shared matcher, one shared reference delay line, and four linear
 * AEC filters.
 *
 * microphones_interleaved is [hop][4], ref is [hop]. Both are read only.
 * A second pre() before the matching post() returns SEQUENCE_ERROR.
 */
int four_aec_nr_res_process_pre(
    FourAecNrRes* self,
    const float* microphones_interleaved,
    const float* ref,
    FourAecNrResPreFrame* out);

/**
 * Resume one pre frame after the external SRP-PHAT/GSC has updated its
 * effective frequency-domain weights.
 *
 * weights is channel-major Complex[4][n_freqs] and uses:
 *
 *   output[bin] = sum(weights[channel,bin] * input[channel,bin])
 *
 * without conjugation. The module coherently projects the four AEC spectra,
 * calculates one post-beam RES gain, fuses it with one NR gain, and writes
 * one final mono hop to out.
 *
 * The external beamformer does not need to return a duplicate mono hop:
 * synthesis is performed from the coherently weighted spectra here. For a
 * multi-frame/time-domain GSC, weights must be its exact effective response
 * for this frame.
 */
int four_aec_nr_res_process_post(
    FourAecNrRes* self,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    float* out);

int four_aec_nr_res_hop_size(const FourAecNrRes* self);
int four_aec_nr_res_fft_size(const FourAecNrRes* self);
int four_aec_nr_res_n_freqs(const FourAecNrRes* self);
int four_aec_nr_res_sample_rate(const FourAecNrRes* self);

/* Structural audit hooks. Values are 1 / 4 / 1 / 1 for a valid handle. */
int four_aec_nr_res_matched_filter_count(const FourAecNrRes* self);
int four_aec_nr_res_linear_aec_count(const FourAecNrRes* self);
int four_aec_nr_res_nr_count(const FourAecNrRes* self);
int four_aec_nr_res_post_res_count(const FourAecNrRes* self);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_AEC_NR_RES_H */
