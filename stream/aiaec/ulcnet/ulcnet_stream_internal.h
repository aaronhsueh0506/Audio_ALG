/* ulcnet_stream internal layout -- not part of the public SDK surface (ulcnet_stream.h), never
 * installed with a release package. One translation unit uses this: ulcnet_stream.c (no stdio,
 * no malloc; this model has no file-loader TU). */

#ifndef ULCNET_STREAM_INTERNAL_H
#define ULCNET_STREAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

/* IDLE: no pending hop (pre_process must run next; post_process refuses with -1). FRAME_OPEN:
 * frame_open() was called this hop; post_process must dispatch frame_finish/frame_bypass on
 * ai_rc. FRAME_CLOSED: a reprime hop -- frame_bypass() was already called directly inside
 * pre_process (no frame_open attempted); post_process skips the finish/bypass dispatch entirely
 * (ai_rc ignored) and goes straight to post_process_freq + synthesis. */
enum {
    ULCNET_STREAM_IDLE = 0,
    ULCNET_STREAM_FRAME_OPEN = 1,
    ULCNET_STREAM_FRAME_CLOSED = 2
};

struct UlcnetStream {
    UlcnetGrid grid;

    FftHandle *fft;     /* carved once, shared by stft_err/stft_far/istft */
    AcStft    *stft_err;
    AcStft    *stft_far;
    AcIstft   *istft;
    UlcnetPrepost *prepost;

    /* Per-hop scratch, one slab: err_f/far_f/hop_out [hop] each, err_re/err_im/far_re/far_im/
     * enh_re/enh_im [n_bins] each (err_in/far_in/pcm_out are the caller's own buffers, passed
     * by pointer). */
    float *err_f;
    float *far_f;
    float *hop_out;
    float *err_re;
    float *err_im;
    float *far_re;
    float *far_im;
    float *enh_re;
    float *enh_im;

    int hop_state;              /* ULCNET_STREAM_IDLE / _FRAME_OPEN / _FRAME_CLOSED */
    int had_hop;                 /* 0 until the first pre_process call completes */
    int prev_solid;               /* delay.solid as of the previous hop */
    unsigned reprime_remaining;  /* identity hops still owed after an alignment-edge reset */
};

/* Field-by-field UlcnetStreamMemReq comparison, shared wherever a stale-pool gate is needed. */
int ulcnet_stream_descriptor_matches(const UlcnetStreamMemReq *expected,
                                     const UlcnetStreamMemReq *req);

typedef struct UlcnetStreamLayout {
    UlcnetStream *self;
    void *fft_mem;
    void *stft_err_mem;
    void *stft_far_mem;
    void *istft_mem;
    void *prepost_mem;
    size_t prepost_bytes;
    void *scratch_mem;   /* the err_f/far_f/hop_out/err_re/err_im/far_re/far_im/enh_re/enh_im slab */
    uint32_t prepost_build_hash;   /* UlcnetPrepost's own UlcnetPrepostMemReq.build_flags_hash,
                                     * folded into this stream's own hash by get_mem_size() */
} UlcnetStreamLayout;

/* One carve walk: sizing (base == NULL, every pointer in *out stays NULL) or carving (base !=
 * NULL, *out holds real addresses into `base`); get_mem_size()/init() agreement is `*total`
 * matching across both calls with the same cfg. -1 on a bad config or an overflowing carve.
 * Carve order (part of ULCNET_STREAM_LAYOUT_VERSION/build_flags_hash -- bump both if this
 * changes): self, fft, stft_err, stft_far, istft, prepost, scratch slab. */
int ulcnet_stream_layout(const UlcnetStreamConfig *cfg, unsigned char *base,
                         UlcnetStreamLayout *out, size_t *total);

/* Constructs every carved object. Returns the live UlcnetStream*, or NULL leaving nothing that
 * looks like a valid instance; only called after ulcnet_stream_layout() and the caller's own
 * size/alignment/stale-pool checks already passed. */
UlcnetStream *ulcnet_stream_finish_init(const UlcnetStreamConfig *cfg, unsigned char *base,
                                        const UlcnetStreamLayout *layout);

#endif /* ULCNET_STREAM_INTERNAL_H */
