/* pipelines/common/se_anr_stage.c -- see se_anr_stage.h. */
#include "se_anr_stage.h"

#include "simd_kernels.h"

/* out[k] = r2[k] / SE_PSD_SCALE, as the exact reciprocal multiply. */
static inline void se_psd_scale_scalar(float* out, const float* r2, int n) {
    int k;
    for (k = 0; k < n; ++k) out[k] = r2[k] * SE_PSD_SCALE_INV;
}

#if SK_HAVE_NEON
static inline void se_psd_scale(float* out, const float* r2, int n) {
    const float32x4_t inv = vdupq_n_f32(SE_PSD_SCALE_INV);
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        vst1q_f32(out + k, vmulq_f32(vld1q_f32(r2 + k), inv));
        vst1q_f32(out + k + 4, vmulq_f32(vld1q_f32(r2 + k + 4), inv));
        vst1q_f32(out + k + 8, vmulq_f32(vld1q_f32(r2 + k + 8), inv));
        vst1q_f32(out + k + 12, vmulq_f32(vld1q_f32(r2 + k + 12), inv));
    }
    for (; k + 4 <= n; k += 4)
        vst1q_f32(out + k, vmulq_f32(vld1q_f32(r2 + k), inv));
    se_psd_scale_scalar(out + k, r2 + k, n - k);
}
#else
static inline void se_psd_scale(float* out, const float* r2, int n) {
    se_psd_scale_scalar(out, r2, n);
}
#endif

AnrConfig se_anr_compose_config(int sample_rate, int fft_size, int hop_size, AnrMode mode) {
    AnrConfig cfg = anr_config_for_mode_grid(sample_rate, fft_size, mode);
    cfg.L = anr_retime_frames(150, sample_rate, hop_size);
    cfg.alpha_decay = cfg.alpha_g;
    return cfg;
}

size_t se_anr_stage_mem_size(const AnrConfig* cfg) {
    return anr_get_mem_size(cfg);
}

AnrDenoiser* se_anr_stage_init(void* mem, size_t bytes, const AnrConfig* cfg) {
    return anr_init(mem, bytes, cfg);
}

const float* se_anr_stage_gain(AnrDenoiser* nr, const SeSpecFrame* frame,
                               float* scratch, int use_r2) {
    const float* extra = NULL;
    if (!nr || !frame || !frame->e) return NULL;
    if (use_r2 && frame->r2 && scratch) {
        se_psd_scale(scratch, frame->r2, frame->n_freqs);
        extra = scratch;
    }
    if (anr_process_gain(nr, frame->e, extra, NULL) != 0) return NULL;
    return anr_get_gain(nr, NULL);
}

int se_anr_stage_set_mode(AnrDenoiser* nr, int sample_rate, int fft_size,
                          int hop_size, AnrMode mode) {
    AnrConfig target;
    if (!nr || !anr_nr_mode_is_valid(mode)) return -1;
    target = se_anr_compose_config(sample_rate, fft_size, hop_size, mode);
    return anr_reconfigure(nr, &target);
}

void se_anr_stage_reset(AnrDenoiser* nr) {
    anr_reset(nr);
}

void se_anr_stage_destroy(AnrDenoiser* nr) {
    anr_destroy(nr);
}
