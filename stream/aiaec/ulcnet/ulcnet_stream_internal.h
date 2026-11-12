/* Private to ulcnet_stream.c (no stdio, no malloc); not part of the public surface. */

#ifndef ULCNET_STREAM_INTERNAL_H
#define ULCNET_STREAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

/* IDLE: no pending hop (post_process refuses with -1). FRAME_OPEN: frame_open() ran; post_process dispatches
 * frame_finish / frame_bypass on ai_rc. FRAME_CLOSED: a reprime hop, already bypassed in pre_process; ai_rc is ignored. */
enum {
    ULCNET_STREAM_IDLE = 0,
    ULCNET_STREAM_FRAME_OPEN = 1,
    ULCNET_STREAM_FRAME_CLOSED = 2
};

struct UlcnetStream {
    UlcnetGrid grid;
    int frames;         /* hops per accelerator job: 1, or the packet size */

    FftHandle *fft;     /* shared by stft_err / stft_far / istft */
    AcStft    *stft_err;
    AcStft    *stft_far;
    AcIstft   *istft;
    UlcnetPrepost *prepost;

    /* Per-job scratch, one slab: err_f / far_f / hop_out [hop], err_re / err_im / far_re / far_im / enh_re / enh_im [frames * n_bins]. */
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
    uint64_t restart_mask;       /* bit k: hop k of the last PCM-entry pre_process restarted the stream's own framing */
};

/* Field-by-field UlcnetStreamMemReq comparison for the stale-pool gate. */
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
    void *scratch_mem;   /* the per-job scratch slab */
    uint32_t prepost_build_hash;   /* UlcnetPrepostMemReq.build_flags_hash, folded into this stream's hash */
} UlcnetStreamLayout;

/* One carve walk: sizing (base == NULL, *out pointers stay NULL) or carving; *total must match across both with one cfg.
 * -1 on a bad config or an overflowing carve. Order (bump LAYOUT_VERSION and build_flags_hash on a change): self, fft,
 * stft_err, stft_far, istft, prepost, scratch. */
int ulcnet_stream_layout(const UlcnetStreamConfig *cfg, unsigned char *base,
                         UlcnetStreamLayout *out, size_t *total);

/* Constructs every carved object; NULL leaves no valid instance. Called after ulcnet_stream_layout() and the caller's
 * size / alignment / stale-pool checks. */
UlcnetStream *ulcnet_stream_finish_init(const UlcnetStreamConfig *cfg, unsigned char *base,
                                        const UlcnetStreamLayout *layout);

#endif /* ULCNET_STREAM_INTERNAL_H */
