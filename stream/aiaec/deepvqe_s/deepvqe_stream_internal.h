/* Private to deepvqe_stream.c (no stdio, no malloc); not part of the public surface. */

#ifndef DEEPVQE_STREAM_INTERNAL_H
#define DEEPVQE_STREAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

struct DeepVqeStream {
    DeepVqeGrid grid;

    FftHandle *fft;     /* shared by stft_mic / stft_far / istft */
    AcStft    *stft_mic;
    AcStft    *stft_far;
    AcIstft   *istft;
    DeepVqePrepost *prepost;

    /* Per-hop scratch, one slab: mic_f / far_f / hop_out [hop], mic_re / mic_im / far_re / far_im / enh_re / enh_im [n_bins]. */
    float *mic_f;
    float *far_f;
    float *hop_out;
    float *mic_re;
    float *mic_im;
    float *far_re;
    float *far_im;
    float *enh_re;
    float *enh_im;

    int open;    /* 1: a DeepVqePrepost frame is open, post_process must finish / bypass it */
    int pending; /* 1: pre_process succeeded and post_process is due (every hop, job or not) */
    /* 0 until the first batch has finished or bypassed (N > 1 pre-roll hops synthesize silence); reset by deepvqe_stream_reset(). */
    int result_ready;
};

/* Field-by-field DeepVqeStreamMemReq comparison for the stale-pool gate. */
int deepvqe_stream_descriptor_matches(const DeepVqeStreamMemReq *expected,
                                      const DeepVqeStreamMemReq *req);

typedef struct DeepVqeStreamLayout {
    DeepVqeStream *self;
    void *fft_mem;
    void *stft_mic_mem;
    void *stft_far_mem;
    void *istft_mem;
    void *prepost_mem;
    size_t prepost_bytes;
    void *scratch_mem;   /* the per-hop scratch slab */
    uint32_t prepost_build_hash;   /* DeepVqePrepostMemReq.build_flags_hash, folded into this stream's hash */
} DeepVqeStreamLayout;

/* One carve walk: sizing (base == NULL, *out pointers stay NULL) or carving; *total must match across both with one cfg.
 * -1 on a bad config or an overflowing carve. Order (bump LAYOUT_VERSION and build_flags_hash on a change): self, fft,
 * stft_mic, stft_far, istft, prepost, scratch. */
int deepvqe_stream_layout(const DeepVqeStreamConfig *cfg, unsigned char *base,
                          DeepVqeStreamLayout *out, size_t *total);

/* Constructs every carved object; NULL leaves no valid instance. Called after deepvqe_stream_layout() and the caller's
 * size / alignment / stale-pool checks. */
DeepVqeStream *deepvqe_stream_finish_init(const DeepVqeStreamConfig *cfg, unsigned char *base,
                                          const DeepVqeStreamLayout *layout);

#endif /* DEEPVQE_STREAM_INTERNAL_H */
