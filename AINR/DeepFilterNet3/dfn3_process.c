#include "dfn3_process.h"

#include <math.h>
#include <string.h>

#include "../dfn_process_common.h"

void dfn3_set_erb_matrices(DFN3State* st,
                             const float* erb_fwd,
                             const float* erb_inv)
{
    if (!st) return;
    st->erb_fwd = erb_fwd;
    st->erb_inv = erb_inv;
}

void dfn3_state_init(DFN3State* st)
{
    if (!st) return;
    memset(st, 0, sizeof(*st));
    /* ERB matrices arrive via dfn3_set_erb_matrices(): caller-loaded
     * erb_fwd.bin / erb_inv.bin from export_erb_matrix.py --runtime-bins. */
    df_common_make_root_hann(st->window, DFN3_WIN_LEN);
    for (int b = 0; b < DFN3_N_ERB; ++b) {
        float position = (float)b / (float)(DFN3_N_ERB - 1);
        st->erb_norm_state[b] = DFN3_ERB_NORM_INIT_LO_DB + position *
            (DFN3_ERB_NORM_INIT_HI_DB - DFN3_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < DFN3_DF_BINS; ++k) {
        float position = (float)k / (float)(DFN3_DF_BINS - 1);
        st->spec_norm_state[k] = DFN3_SPEC_NORM_INIT_LO + position *
            (DFN3_SPEC_NORM_INIT_HI - DFN3_SPEC_NORM_INIT_LO);
    }
}

void dfn3_analysis(DFN3State* st, const float* frame,
                   float* out_re, float* out_im)
{
    const float normalization = 1.0f / sqrtf((float)DFN3_N_FFT);
    if (!st || !frame || !out_re || !out_im) return;
    df_common_analysis(st->analysis_buf, st->window,
                       st->scratch_re, st->scratch_im, frame,
                       DFN3_N_FFT, DFN3_HOP_LEN, normalization,
                       out_re, out_im);
}

void dfn3_compute_features(DFN3State* st,
                           const float* spec_re, const float* spec_im,
                           float* feat_erb, float* feat_spec)
{
    if (!st || !spec_re || !spec_im || !feat_erb || !feat_spec) return;
    df_common_features(
        spec_re, spec_im, st->erb_fwd,
        DFN3_N_BINS, DFN3_N_ERB, DFN3_DF_BINS,
        DFN3_ANALYSIS_SCALE, DFN3_ERB_LOG_FLOOR,
        DFN3_ERB_NORM_ALPHA, DFN3_ERB_NORM_SCALE_DB,
        st->erb_norm_state,
        DFN3_SPEC_NORM_ALPHA, DFN3_SPEC_NORM_EPS,
        st->spec_norm_state,
        st->scratch_power, st->scratch_erb_db, feat_erb, feat_spec);
}

void dfn3_apply_atten_lim(const float* noisy_re, const float* noisy_im,
                          float* enh_re, float* enh_im,
                          float atten_lim_db)
{
    if (!noisy_re || !noisy_im || !enh_re || !enh_im ||
        !isfinite(atten_lim_db) || atten_lim_db == 0.0f) return;
    df_common_atten_lim(noisy_re, noisy_im, enh_re, enh_im,
                        DFN3_N_BINS, atten_lim_db);
}

int dfn3_compose(DFN3State* st,
                 const float* spec_re, const float* spec_im,
                 const float* erb_mask, const float* coefs,
                 float* out_re, float* out_im)
{
    int slot;
    int target;
    if (!st || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im) return 0;
    /* DFN3 owns low bins with the raw spectrum; ERB masking owns highs. */
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN3_N_BINS, DFN3_N_ERB,
                          st->scratch_power);
    slot = st->df_ring_idx;
    memcpy(st->df_ring_re[slot], spec_re,
           sizeof(st->df_ring_re[slot]));
    memcpy(st->df_ring_im[slot], spec_im,
           sizeof(st->df_ring_im[slot]));
    for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
        int high = k - DFN3_DF_BINS;
        st->hi_delay_re[slot][high] = spec_re[k] * st->scratch_power[k];
        st->hi_delay_im[slot][high] = spec_im[k] * st->scratch_power[k];
    }
    memcpy(st->coef_ring[slot], coefs,
           sizeof(st->coef_ring[slot]));
    memcpy(st->noisy_ring_re[slot], spec_re,
           sizeof(st->noisy_ring_re[slot]));
    memcpy(st->noisy_ring_im[slot], spec_im,
           sizeof(st->noisy_ring_im[slot]));
    st->df_ring_idx = (slot + 1) % DFN3_DF_RING;
    if (st->df_ring_count < DFN3_DF_RING) ++st->df_ring_count;
    if (st->df_ring_count <= DFN3_DF_LOOKAHEAD) return 0;

    target = (slot - DFN3_DF_LOOKAHEAD + DFN3_DF_RING) % DFN3_DF_RING;
    for (int k = 0; k < DFN3_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN3_DF_ORDER; ++tap) {
            int source = (slot + tap - (DFN3_DF_ORDER - 1) +
                          DFN3_DF_RING) % DFN3_DF_RING;
            float xr = st->df_ring_re[source][k];
            float xi = st->df_ring_im[source][k];
            float cr = st->coef_ring[target][k][tap][0];
            float ci = st->coef_ring[target][k][tap][1];
            filtered_re += xr * cr - xi * ci;
            filtered_im += xi * cr + xr * ci;
        }
        out_re[k] = filtered_re;
        out_im[k] = filtered_im;
    }
    for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
        int high = k - DFN3_DF_BINS;
        out_re[k] = st->hi_delay_re[target][high];
        out_im[k] = st->hi_delay_im[target][high];
    }
#if DFN3_MASK_PF
    dfn3_post_filter(st->noisy_ring_re[target], st->noisy_ring_im[target],
                     out_re, out_im, DFN3_PF_BETA);
#endif
    return 1;
}

int dfn3_compose_stream(DFN3State* st,
                        const float* current_spec_re,
                        const float* current_spec_im,
                        int heads_valid,
                        const float* erb_mask,
                        const float* coefs,
                        float atten_lim_db,
                        float* out_re,
                        float* out_im,
                        long long* output_frame_index)
{
    long long current;
    long long target_frame;
    int current_slot;
    int target_slot;

    if (!st || !current_spec_re || !current_spec_im || !out_re || !out_im)
        return -1;
    current = st->stream_frame_index;
    if (current < DFN3_MASK_LOOKAHEAD) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !coefs ||
               !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % DFN3_DF_RING);
    memcpy(st->df_ring_re[current_slot], current_spec_re,
           sizeof(st->df_ring_re[current_slot]));
    memcpy(st->df_ring_im[current_slot], current_spec_im,
           sizeof(st->df_ring_im[current_slot]));
    memcpy(st->noisy_ring_re[current_slot], current_spec_re,
           sizeof(st->noisy_ring_re[current_slot]));
    memcpy(st->noisy_ring_im[current_slot], current_spec_im,
           sizeof(st->noisy_ring_im[current_slot]));

    if (current < DFN3_MASK_LOOKAHEAD) return 0;
    target_frame = current - DFN3_MASK_LOOKAHEAD;
    target_slot = (int)(target_frame % DFN3_DF_RING);
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN3_N_BINS, DFN3_N_ERB,
                          st->scratch_power);

    for (int k = 0; k < DFN3_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN3_DF_ORDER; ++tap) {
            long long source_frame =
                target_frame - DFN3_DF_HISTORY + tap;
            float xr = 0.0f;
            float xi = 0.0f;
            if (source_frame >= 0) {
                int source_slot = (int)(source_frame % DFN3_DF_RING);
                xr = st->df_ring_re[source_slot][k];
                xi = st->df_ring_im[source_slot][k];
            }
            {
                size_t coef_index =
                    ((size_t)k * DFN3_DF_ORDER + (size_t)tap) * 2u;
                float cr = coefs[coef_index];
                float ci = coefs[coef_index + 1u];
                filtered_re += xr * cr - xi * ci;
                filtered_im += xi * cr + xr * ci;
            }
        }
        out_re[k] = filtered_re;
        out_im[k] = filtered_im;
    }
    for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
        out_re[k] = st->noisy_ring_re[target_slot][k] * st->scratch_power[k];
        out_im[k] = st->noisy_ring_im[target_slot][k] * st->scratch_power[k];
    }
#if DFN3_MASK_PF
    dfn3_post_filter(st->noisy_ring_re[target_slot],
                     st->noisy_ring_im[target_slot],
                     out_re, out_im, DFN3_PF_BETA);
#endif
    dfn3_apply_atten_lim(st->noisy_ring_re[target_slot],
                         st->noisy_ring_im[target_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
}

void dfn3_post_filter(const float* spec_re, const float* spec_im,
                      float* enh_re, float* enh_im, float beta)
{
    if (!spec_re || !spec_im || !enh_re || !enh_im) return;
    df_common_post_filter(
        spec_re, spec_im, enh_re, enh_im, DFN3_N_BINS, beta);
}

void dfn3_synthesis(DFN3State* st,
                    const float* spec_re, const float* spec_im,
                    float* out_frame)
{
    const float normalization = sqrtf((float)DFN3_N_FFT);
    if (!st || !spec_re || !spec_im || !out_frame) return;
    df_common_synthesis(st->synthesis_buf, st->window,
                        st->scratch_re, st->scratch_im,
                        spec_re, spec_im, DFN3_N_FFT, DFN3_HOP_LEN,
                        normalization, out_frame);
}

const char* dfn3_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
