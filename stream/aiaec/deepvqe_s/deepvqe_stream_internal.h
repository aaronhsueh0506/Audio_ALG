/* deepvqe_stream internal layout -- not part of the public SDK surface (deepvqe_stream.h), never
 * installed with a release package. One translation unit uses this: deepvqe_stream.c (no stdio,
 * no malloc; this model has no file-loader TU, unlike DFN2's ERB matrices). */

#ifndef DEEPVQE_STREAM_INTERNAL_H
#define DEEPVQE_STREAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "deepvqe_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

struct DeepVqeStream {
    DeepVqeGrid grid;

    FftHandle *fft;     /* carved once, shared by stft_mic/stft_far/istft */
    AcStft    *stft_mic;
    AcStft    *stft_far;
    AcIstft   *istft;
    DeepVqePrepost *prepost;

    /* Per-hop scratch, one slab: mic_f/far_f/hop_out [hop] each, mic_re/mic_im/far_re/far_im/
     * enh_re/enh_im [n_bins] each (mic_in/far_in/pcm_out are the caller's own buffers, passed
     * by pointer). */
    float *mic_f;
    float *far_f;
    float *hop_out;
    float *mic_re;
    float *mic_im;
    float *far_re;
    float *far_im;
    float *enh_re;
    float *enh_im;

    int open;   /* 0 closed, 1: pre_process succeeded and post_process is due */
};

/* Field-by-field DeepVqeStreamMemReq comparison, shared wherever a stale-pool gate is needed. */
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
    void *scratch_mem;   /* the mic_f/far_f/hop_out/mic_re/mic_im/far_re/far_im/enh_re/enh_im slab */
    uint32_t prepost_build_hash;   /* DeepVqePrepost's own DeepVqePrepostMemReq.build_flags_hash,
                                     * folded into this stream's own hash by get_mem_size() */
} DeepVqeStreamLayout;

/* One carve walk: sizing (base == NULL, every pointer in *out stays NULL) or carving (base !=
 * NULL, *out holds real addresses into `base`); get_mem_size()/init() agreement is `*total`
 * matching across both calls with the same cfg. -1 on a bad config or an overflowing carve.
 * Carve order (part of DEEPVQE_STREAM_LAYOUT_VERSION/build_flags_hash -- bump both if this
 * changes): self, fft, stft_mic, stft_far, istft, prepost, scratch slab. */
int deepvqe_stream_layout(const DeepVqeStreamConfig *cfg, unsigned char *base,
                          DeepVqeStreamLayout *out, size_t *total);

/* Constructs every carved object. Returns the live DeepVqeStream*, or NULL leaving nothing that
 * looks like a valid instance; only called after deepvqe_stream_layout() and the caller's own
 * size/alignment/stale-pool checks already passed. */
DeepVqeStream *deepvqe_stream_finish_init(const DeepVqeStreamConfig *cfg, unsigned char *base,
                                          const DeepVqeStreamLayout *layout);

#endif /* DEEPVQE_STREAM_INTERNAL_H */
