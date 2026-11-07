/* dfn2_stream internal layout -- not part of the public SDK surface (dfn2_stream.h), never
 * installed with a release package. Shared between dfn2_stream.c (no stdio: everything except the
 * file loader) and dfn2_stream_erb_file.c (the only stdio TU: the file-init entry point). */

#ifndef DFN2_STREAM_INTERNAL_H
#define DFN2_STREAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_stream.h"
#include "hpf.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

struct DFN2Stream {
    DFN2Grid grid;
    float    atten_lim_db;
    int      hpf_enable;

    Hpf        *hpf;     /* NULL when !hpf_enable */
    FftHandle  *fft;     /* carved once, shared by stft and istft (mirrors the board sample) */
    AcStft     *stft;
    AcIstft    *istft;
    DFN2Prepost *prepost;

    /* Per-hop scratch, one slab: hop_in/hop_out [hop] each, spec_re/spec_im/enh_re/enh_im [n_bins]
     * each -- the float-domain counterpart of the board sample's grid_buffers_alloc(), minus the
     * two int16 arrays (pcm_in/pcm_out are the caller's own buffers, passed by pointer). */
    float *hop_in;
    float *hop_out;
    float *spec_re;
    float *spec_im;
    float *enh_re;
    float *enh_im;

    int   open;        /* 0 closed, 1: pre_process succeeded and post_process is due */
    void *heap_base;   /* non-NULL only for a to-be-added _create() convenience; unused by init/init_from_erb_file */
};

/* cfg->erb_fwd and cfg->erb_inv both NULL (file-init mode intended); cfg itself may be NULL (0). */
int dfn2_stream_is_file_mode(const DFN2StreamConfig *cfg);

/* Field-by-field DFN2StreamMemReq comparison, shared by dfn2_stream_init_ex() and
 * dfn2_stream_init_from_erb_file_ex() so a future descriptor field only needs updating once. */
int dfn2_stream_descriptor_matches(const DFN2StreamMemReq *expected, const DFN2StreamMemReq *req);

typedef struct DFN2StreamLayout {
    DFN2Stream *self;
    void       *hpf_mem;      /* NULL region when !cfg->hpf_enable */
    void       *fft_mem;
    void       *stft_mem;
    void       *istft_mem;
    void       *prepost_mem;
    size_t      prepost_bytes;
    void       *scratch_mem;  /* the hop_in/hop_out/spec_re/spec_im/enh_re/enh_im slab */
    /* File mode only (both NULL in memory mode): */
    float      *erb_fwd_storage;   /* [n_bins][n_erb], bin-major, kept */
    float      *erb_inv_storage;   /* [n_erb][n_bins], band-major, kept */
    float      *erb_scratch;       /* [n_bins][n_erb], bin-major, transient load scratch only */
    uint32_t    prepost_build_hash; /* dfn2_prepost's own DFN2PrepostMemReq.build_flags_hash, folded
                                      * into this stream's own hash by dfn2_stream_get_mem_size() */
} DFN2StreamLayout;

/* One carve walk: sizing (base == NULL, every pointer in *out stays NULL) or carving (base !=
 * NULL, *out holds real addresses into `base`); get_mem_size()/init() agreement is `*total`
 * matching across both calls with the same cfg. -1 on a bad config or an overflowing carve.
 * Carve order (part of DFN2_STREAM_LAYOUT_VERSION/build_flags_hash -- bump both if this changes):
 * self, hpf (if enabled), fft, stft, istft, prepost, scratch slab, then (file mode only)
 * erb_fwd_storage, erb_inv_storage, erb_scratch. */
int dfn2_stream_layout(const DFN2StreamConfig *cfg, unsigned char *base,
                       DFN2StreamLayout *out, size_t *total);

/* Constructs every carved object except the ERB tables, which must already hold valid data by the
 * time this runs (the caller's own borrowed pointers in memory mode; `layout`'s pool-owned,
 * already-loaded storage in file mode) -- `erb_fwd`/`erb_inv` are exactly those two pointers.
 * Returns the live DFN2Stream*, or NULL leaving nothing that looks like a valid instance; only
 * called after dfn2_stream_layout() and the caller's own size/alignment/stale-pool checks already
 * passed. */
DFN2Stream *dfn2_stream_finish_init(const DFN2StreamConfig *cfg, unsigned char *base,
                                    const DFN2StreamLayout *layout,
                                    const float *erb_fwd, const float *erb_inv);

#endif /* DFN2_STREAM_INTERNAL_H */
