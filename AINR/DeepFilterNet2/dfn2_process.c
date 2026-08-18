#include "dfn2_process.h"

#include <math.h>
#include <string.h>

#include "../dfn_process_common.h"

void dfn2_set_erb_matrices(DFN2State* st,
                             const float* erb_fwd,
                             const float* erb_inv)
{
    if (!st) return;
    st->erb_fwd = erb_fwd;
    st->erb_inv = erb_inv;
}

void dfn2_state_init(DFN2State* st)
{
    if (!st) return;
    memset(st, 0, sizeof(*st));
    /* ERB matrices arrive via dfn2_set_erb_matrices(): caller-loaded
     * erb_fwd.bin / erb_inv.bin from export_erb_matrix.py --runtime-bins. */
    df_common_make_root_hann(st->window, DFN2_WIN_LEN);
    for (int b = 0; b < DFN2_N_ERB; ++b) {
        float position = (float)b / (float)(DFN2_N_ERB - 1);
        st->erb_norm_state[b] = DFN2_ERB_NORM_INIT_LO_DB + position *
            (DFN2_ERB_NORM_INIT_HI_DB - DFN2_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float position = (float)k / (float)(DFN2_DF_BINS - 1);
        st->spec_norm_state[k] = DFN2_SPEC_NORM_INIT_LO + position *
            (DFN2_SPEC_NORM_INIT_HI - DFN2_SPEC_NORM_INIT_LO);
    }
}

void dfn2_analysis(DFN2State* st, const float* frame,
                   float* out_re, float* out_im)
{
    const float normalization = 1.0f / sqrtf((float)DFN2_N_FFT);
    if (!st || !frame || !out_re || !out_im) return;
    df_common_analysis(st->analysis_buf, st->window,
                       st->scratch_re, st->scratch_im, frame,
                       DFN2_N_FFT, DFN2_HOP_LEN, normalization,
                       out_re, out_im);
}

void dfn2_compute_features(DFN2State* st,
                           const float* spec_re, const float* spec_im,
                           float* feat_erb, float* feat_spec)
{
    if (!st || !spec_re || !spec_im || !feat_erb || !feat_spec) return;
    df_common_features(
        spec_re, spec_im, st->erb_fwd,
        DFN2_N_BINS, DFN2_N_ERB, DFN2_DF_BINS,
        DFN2_ANALYSIS_SCALE, DFN2_ERB_LOG_FLOOR,
        DFN2_ERB_NORM_ALPHA, DFN2_ERB_NORM_SCALE_DB,
        st->erb_norm_state,
        DFN2_SPEC_NORM_ALPHA, DFN2_SPEC_NORM_EPS,
        st->spec_norm_state,
        st->scratch_power, st->scratch_erb_db, feat_erb, feat_spec);
}

void dfn2_apply_atten_lim(const float* noisy_re, const float* noisy_im,
                          float* enh_re, float* enh_im,
                          float atten_lim_db)
{
    if (!noisy_re || !noisy_im || !enh_re || !enh_im ||
        !isfinite(atten_lim_db) || atten_lim_db == 0.0f) return;
    df_common_atten_lim(noisy_re, noisy_im, enh_re, enh_im,
                        DFN2_N_BINS, atten_lim_db);
}

int dfn2_compose(DFN2State* st,
                 const float* spec_re, const float* spec_im,
                 const float* erb_mask, const float* coefs, float alpha,
                 float* out_re, float* out_im)
{
    int slot;
    int target;
    if (!st || !spec_re || !spec_im || !erb_mask || !coefs ||
        !out_re || !out_im || !isfinite(alpha)) return 0;
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN2_N_BINS, DFN2_N_ERB,
                          st->scratch_bin_gain);
    slot = st->df_ring_idx;
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        st->df_ring_re[slot][k] = spec_re[k] * st->scratch_bin_gain[k];
        st->df_ring_im[slot][k] = spec_im[k] * st->scratch_bin_gain[k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        st->hi_delay_re[slot][high] =
            spec_re[k] * st->scratch_bin_gain[k];
        st->hi_delay_im[slot][high] =
            spec_im[k] * st->scratch_bin_gain[k];
    }
    memcpy(st->coef_ring[slot], coefs,
           sizeof(st->coef_ring[slot]));
    memcpy(st->noisy_ring_re[slot], spec_re,
           sizeof(st->noisy_ring_re[slot]));
    memcpy(st->noisy_ring_im[slot], spec_im,
           sizeof(st->noisy_ring_im[slot]));
    st->alpha_ring[slot] = alpha;
    st->df_ring_idx = (slot + 1) % DFN2_DF_RING;
    if (st->df_ring_count < DFN2_DF_RING) ++st->df_ring_count;
    if (st->df_ring_count <= DFN2_DF_LOOKAHEAD) return 0;

    target = (slot - DFN2_DF_LOOKAHEAD + DFN2_DF_RING) % DFN2_DF_RING;
    alpha = st->alpha_ring[target];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
            int source = (slot + tap - (DFN2_DF_ORDER - 1) +
                          DFN2_DF_RING) % DFN2_DF_RING;
            float xr = st->df_ring_re[source][k];
            float xi = st->df_ring_im[source][k];
            float cr = st->coef_ring[target][k][tap][0];
            float ci = st->coef_ring[target][k][tap][1];
            filtered_re += xr * cr - xi * ci;
            filtered_im += xi * cr + xr * ci;
        }
        out_re[k] = alpha * filtered_re +
                    (1.0f - alpha) * st->df_ring_re[target][k];
        out_im[k] = alpha * filtered_im +
                    (1.0f - alpha) * st->df_ring_im[target][k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        out_re[k] = st->hi_delay_re[target][high];
        out_im[k] = st->hi_delay_im[target][high];
    }
#if DFN2_MASK_PF
    dfn2_post_filter(st->noisy_ring_re[target], st->noisy_ring_im[target],
                     out_re, out_im, DFN2_PF_BETA);
#endif
    return 1;
}

int dfn2_compose_stream(DFN2State* st,
                        const float* current_spec_re,
                        const float* current_spec_im,
                        int heads_valid,
                        const float* erb_mask,
                        const float* coefs,
                        float alpha,
                        float atten_lim_db,
                        float* out_re,
                        float* out_im,
                        long long* output_frame_index)
{
    long long current;
    long long head_frame;
    long long target_frame;
    int current_slot;
    int head_slot;
    int target_slot;

    if (!st || !current_spec_re || !current_spec_im || !out_re || !out_im)
        return -1;
    current = st->stream_frame_index;
    if (current < DFN2_MASK_LOOKAHEAD) {
        if (heads_valid) return -1;
    } else if (!heads_valid || !erb_mask || !coefs || !isfinite(alpha) ||
               !isfinite(atten_lim_db)) {
        return -1;
    }
    ++st->stream_frame_index;
    current_slot = (int)(current % DFN2_DF_RING);
    memcpy(st->noisy_ring_re[current_slot], current_spec_re,
           sizeof(st->noisy_ring_re[current_slot]));
    memcpy(st->noisy_ring_im[current_slot], current_spec_im,
           sizeof(st->noisy_ring_im[current_slot]));

    /* A lookahead network cannot return frame 0's heads until input frame
     * MASK_LOOKAHEAD has arrived.  Enforce this alignment here: silently
     * treating a returned frame-(n-L) mask as frame n is audible but finite,
     * so ordinary NaN/output-smoke tests cannot catch the mistake. */
    if (current < DFN2_MASK_LOOKAHEAD) {
        return 0;
    }

    head_frame = current - DFN2_MASK_LOOKAHEAD;
    head_slot = (int)(head_frame % DFN2_DF_RING);
    df_common_expand_mask(erb_mask, st->erb_inv,
                          DFN2_N_BINS, DFN2_N_ERB,
                          st->scratch_bin_gain);
    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        st->df_ring_re[head_slot][k] =
            st->noisy_ring_re[head_slot][k] * st->scratch_bin_gain[k];
        st->df_ring_im[head_slot][k] =
            st->noisy_ring_im[head_slot][k] * st->scratch_bin_gain[k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        st->hi_delay_re[head_slot][high] =
            st->noisy_ring_re[head_slot][k] * st->scratch_bin_gain[k];
        st->hi_delay_im[head_slot][high] =
            st->noisy_ring_im[head_slot][k] * st->scratch_bin_gain[k];
    }
    memcpy(st->coef_ring[head_slot], coefs,
           sizeof(st->coef_ring[head_slot]));
    st->alpha_ring[head_slot] = alpha;

    /* In a cascade the newest usable masked source is head_frame, hence the
     * output target is one DF lookahead behind that head. */
    target_frame = head_frame - DFN2_DF_LOOKAHEAD;
    if (target_frame < 0) return 0;
    target_slot = (int)(target_frame % DFN2_DF_RING);
    alpha = st->alpha_ring[target_slot];
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    for (int k = 0; k < DFN2_DF_BINS; ++k) {
        float filtered_re = 0.0f;
        float filtered_im = 0.0f;
        for (int tap = 0; tap < DFN2_DF_ORDER; ++tap) {
            long long source_frame =
                target_frame - DFN2_DF_HISTORY + tap;
            float xr = 0.0f;
            float xi = 0.0f;
            if (source_frame >= 0) {
                int source_slot = (int)(source_frame % DFN2_DF_RING);
                xr = st->df_ring_re[source_slot][k];
                xi = st->df_ring_im[source_slot][k];
            }
            {
                float cr = st->coef_ring[target_slot][k][tap][0];
                float ci = st->coef_ring[target_slot][k][tap][1];
                filtered_re += xr * cr - xi * ci;
                filtered_im += xi * cr + xr * ci;
            }
        }
        out_re[k] = alpha * filtered_re +
                    (1.0f - alpha) * st->df_ring_re[target_slot][k];
        out_im[k] = alpha * filtered_im +
                    (1.0f - alpha) * st->df_ring_im[target_slot][k];
    }
    for (int k = DFN2_DF_BINS; k < DFN2_N_BINS; ++k) {
        int high = k - DFN2_DF_BINS;
        out_re[k] = st->hi_delay_re[target_slot][high];
        out_im[k] = st->hi_delay_im[target_slot][high];
    }
#if DFN2_MASK_PF
    dfn2_post_filter(st->noisy_ring_re[target_slot],
                     st->noisy_ring_im[target_slot],
                     out_re, out_im, DFN2_PF_BETA);
#endif
    dfn2_apply_atten_lim(st->noisy_ring_re[target_slot],
                         st->noisy_ring_im[target_slot],
                         out_re, out_im, atten_lim_db);
    if (output_frame_index) *output_frame_index = target_frame;
    return 1;
}

void dfn2_post_filter(const float* spec_re, const float* spec_im,
                      float* enh_re, float* enh_im, float beta)
{
    if (!spec_re || !spec_im || !enh_re || !enh_im) return;
    df_common_post_filter(
        spec_re, spec_im, enh_re, enh_im, DFN2_N_BINS, beta);
}

void dfn2_synthesis(DFN2State* st,
                    const float* spec_re, const float* spec_im,
                    float* out_frame)
{
    const float normalization = sqrtf((float)DFN2_N_FFT);
    if (!st || !spec_re || !spec_im || !out_frame) return;
    df_common_synthesis(st->synthesis_buf, st->window,
                        st->scratch_re, st->scratch_im,
                        spec_re, spec_im, DFN2_N_FFT, DFN2_HOP_LEN,
                        normalization, out_frame);
}

const char* dfn2_simd_backend(void)
{
    return DF_COMMON_HAVE_NEON ? "aarch64-neon" : "scalar";
}
