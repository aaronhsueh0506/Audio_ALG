/** Sample-level assembly shared by the 4ch_raw and 4ch_static programs: the array instance with the
 * beamformer helper (the unit), and the stages behind the fuse (denoiser, comfort noise,
 * synthesis). Header-only, no ABI: copy and adapt. Not part of a delivered archive. */
#ifndef ARRAY_SAMPLE_UNIT_H
#define ARRAY_SAMPLE_UNIT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "aec.h"
#include "anr.h"
#include "array_sample_beam.h"
#include "array_sample_post.h"
#include "audio_stft.h"
#include "utility/mem_align.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The array and the helper, carved in this order from one pool of `total` bytes aligned to req.alignment. */
typedef struct ArraySampleUnit {
    Aec* aec;
    ArraySampleBeam* beam;
    AecMemReq req;
    size_t aec_bytes, beam_bytes, total;
} ArraySampleUnit;

/** 0, or -1 for an invalid config or parameter set. */
static inline int array_sample_unit_plan(ArraySampleUnit* u, const AecConfig* cfg,
                                         const ArraySampleBeamParams* bp) {
    memset(u, 0, sizeof(*u));
    if (aec_get_mem_requirements(cfg, &u->req) != 0 ||
        array_sample_beam_get_mem_size(cfg, bp, &u->beam_bytes) != 0) return -1;
    u->aec_bytes = ALIGN16((size_t)u->req.bytes);
    u->total = u->aec_bytes + u->beam_bytes;
    return 0;
}

/** Places the array and the helper in `mem` (>= total bytes). 0, or -1 with nothing usable. */
static inline int array_sample_unit_init(ArraySampleUnit* u, void* mem, const AecConfig* cfg,
                                         const ArraySampleBeamParams* bp) {
    u->aec = aec_init_ex(mem, u->aec_bytes, cfg, &u->req);
    if (!u->aec) return -1;
    u->beam = array_sample_beam_init((unsigned char*)mem + u->aec_bytes, u->beam_bytes, cfg, bp, u->aec);
    return u->beam ? 0 : -1;
}

/** The helper tears down before the array it borrows. Idempotent. */
static inline void array_sample_unit_destroy(ArraySampleUnit* u) {
    array_sample_beam_destroy(u->beam);
    u->beam = NULL;
    aec_destroy(u->aec);
    u->aec = NULL;
}

/** The stages after the fuse, carved in this order from one 16-byte aligned pool of `total` bytes:
 * denoiser, synthesis, NR prior, output spectrum. */
typedef struct ArraySampleStages {
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t nr_bytes, synth_bytes, vec_bytes, spec_bytes, total;
    AnrDenoiser* nr;
    AcIstft* istft;
    float* extra_noise;
    Complex* spec;
    uint32_t reset_handled;   /* AecResidualView.post_reset_serial the synthesis last followed */
} ArraySampleStages;

/** 0, or -1 for a grid no stage accepts. */
static inline int array_sample_stages_plan(ArraySampleStages* ps, int sample_rate, int fft_size) {
    int n_freqs = fft_size / 2 + 1;
    ps->nr_cfg = anr_config_for_level_grid(sample_rate, fft_size, ANR_LEVEL_BALANCED);
    ac_stft_config_default(&ps->synth_cfg, fft_size);
    ps->nr_bytes = anr_get_mem_size(&ps->nr_cfg);
    ps->synth_bytes = ac_istft_get_mem_size(&ps->synth_cfg);
    ps->vec_bytes = ALIGN16((size_t)n_freqs * sizeof(float));
    ps->spec_bytes = ALIGN16((size_t)n_freqs * sizeof(Complex));
    if (ps->nr_bytes == 0 || ps->synth_bytes == 0) return -1;
    ps->total = ALIGN16(ps->nr_bytes) + ALIGN16(ps->synth_bytes) + ps->vec_bytes + ps->spec_bytes;
    return 0;
}

/** Places the stages in `mem` (>= total bytes). 0, or -1 when the denoiser or the synthesis refuses. */
static inline int array_sample_stages_init(ArraySampleStages* ps, void* mem) {
    unsigned char* cur = (unsigned char*)mem;
    if (!cur) return -1;
    ps->nr = anr_init(cur, ps->nr_bytes, &ps->nr_cfg);
    cur += ALIGN16(ps->nr_bytes);
    ps->istft = ac_istft_init(cur, ps->synth_bytes, &ps->synth_cfg);
    cur += ALIGN16(ps->synth_bytes);
    ps->extra_noise = (float*)cur;
    cur += ps->vec_bytes;
    ps->spec = (Complex*)cur;
    ps->reset_handled = 0u;
    return ps->nr && ps->istft ? 0 : -1;
}

/** One hop after the fuse into `out`: G_res (the array's), G_nr (estimated on B), P = B * min(G_nr, G_res)
 * over the rendering base B, comfort noise on the bins the gain cut, synthesis (cleared first when a lane
 * had a post-chain reset). 0, or -1 when the denoiser or the array refuses the hop. */
static inline int array_sample_stages_chain(ArraySampleStages* ps, Aec* aec, const AecResidualView* v,
                                            float* out) {
    const float* nr = array_sample_anr_gain(ps->nr, v, ps->extra_noise, 1);
    if (!nr) return -1;
    if (array_sample_post_reset_due(&ps->reset_handled, v)) ac_istft_reset(ps->istft);
    if (aec_apply_gain(aec, nr, ps->spec) != AEC_OK ||
        aec_apply_cng(aec, nr, ps->spec) != AEC_OK) return -1;
    (void)ac_istft_push(ps->istft, ps->spec, out);
    return 0;
}

/** Back to a fresh denoiser and synthesis. */
static inline void array_sample_stages_reset(ArraySampleStages* ps) {
    anr_reset(ps->nr);
    ac_istft_reset(ps->istft);
}

#ifdef __cplusplus
}
#endif
#endif /* ARRAY_SAMPLE_UNIT_H */
