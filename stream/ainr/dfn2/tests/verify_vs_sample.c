/* Verifies dfn2_stream_pre_process/post_process against an independent re-implementation of the
 * board sample's own per-hop loop (alg_ai3_dfn.c's simple_thread(): compute_features ->
 * dfn2_prepost_frame_open -> NPU -> dfn2_prepost_frame_finish/frame_bypass -> post_process_freq ->
 * ac_istft_push_planar -> int16 saturate), built directly against lib/ainr/dfn2/dfn2_prepost.h and
 * audiolib, never against dfn2_stream.c. The two drivers run the SAME deterministic fake
 * accelerator (a pure function of a global hop counter, ignoring its own input argument so its
 * output cannot accidentally depend on which driver is calling it) and must produce byte-identical
 * PCM and an identical (pre_rc, generation, valid) trace, hop for hop.
 *
 * Board sample read-only reference: ~/Desktop/exp/audio_sample/alg_ai3_dfn.c lines 505-520
 * (compute_features), 1750-1838 (the per-hop dispatch + synthesis). Never edited; this file
 * re-implements the same sequence against the public lib/ainr/dfn2 API instead of calling into it.
 *
 * INJECT_BUG_SHIFT (compile-time, off by default): delays the comparison's view of driver B's PCM
 * by one hop, to prove this harness's comparison can actually detect a divergence (project rule:
 * a verification gate must be shown capable of failing). It injects nothing into dfn2_stream.c
 * itself -- that file is never touched by this test -- it only mis-feeds the comparator, which is
 * exactly what a real one-hop OLA bug in the implementation would look like to this harness. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dfn2_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"
#include "hpf.h"

/* ---- shared deterministic fake accelerator: pure function of (global hop, force_fail) ---- */
static int shared_fake(long long ghop, int force_fail, DFN2PrepostOutputs *out) {
    size_t i;
    if (force_fail) return -1;   /* writes nothing: mirrors fetch_heads() leaving the NaN prefill */
    for (i = 0; i < out->erb_mask_elements; ++i)
        out->erb_mask[i] = 0.1f + 0.001f * (float)((ghop + (long long)i) % 97);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i)
        out->coefs[i] = 0.01f * (float)(((ghop * 7 + (long long)i) % 53) - 26);
    for (i = 0; i < out->alpha_elements; ++i)
        out->alpha[i] = 0.2f;
#endif
    for (i = 0; i < out->gru_hidden_elements; ++i)
        out->gru_hidden_next[i] = 0.05f * (float)(((ghop + (long long)i) % 41) - 20);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->df_convp_window_elements; ++i)
        out->df_convp_window_next[i] = 0.02f * (float)(((ghop * 3 + (long long)i) % 31) - 15);
#endif
    return 0;
}

/* ---- driver A: independent re-implementation of the board sample's loop ---- */
typedef struct BoardDriver {
    DFN2Grid grid;
    FftHandle *fft;
    AcStft *stft; void *stft_pool;
    AcIstft *istft; void *istft_pool;
    DFN2Prepost *prepost; void *prepost_pool;
    Hpf *hpf;   /* NULL when !hpf_enable; board sample uses hpf_create(), fine for this host driver */
    float *spec_re, *spec_im, *enh_re, *enh_im, *hop_in, *hop_out;
} BoardDriver;

static int board_init(BoardDriver *d, const DFN2Grid *grid, const float *erb_fwd, const float *erb_inv,
                      int hpf_enable, float hpf_cutoff_hz) {
    AcStftConfig scfg;
    DFN2PrepostConfig pcfg;
    DFN2PrepostMemReq preq;
    size_t n_bins = (size_t)grid->n_bins, hop = (size_t)grid->hop;
    size_t sb, ib;

    memset(d, 0, sizeof(*d));
    d->grid = *grid;
    d->fft = fft_create(grid->n_fft);
    if (!d->fft) return -1;

    if (hpf_enable) {
        d->hpf = hpf_create(hpf_cutoff_hz, grid->sample_rate);
        if (!d->hpf) return -1;
    }

    ac_stft_config_default(&scfg, grid->n_fft);
    scfg.spec_scale = grid->stft_scale;
    scfg.ispec_scale = grid->stft_unscale;
    scfg.fft = d->fft;

    sb = ac_stft_get_mem_size(&scfg);
    if (sb == 0 || posix_memalign(&d->stft_pool, 16, sb) != 0) return -1;
    d->stft = ac_stft_init(d->stft_pool, sb, &scfg);
    if (!d->stft) return -1;

    ib = ac_istft_get_mem_size(&scfg);
    if (ib == 0 || posix_memalign(&d->istft_pool, 16, ib) != 0) return -1;
    d->istft = ac_istft_init(d->istft_pool, ib, &scfg);
    if (!d->istft) return -1;

    if (dfn2_prepost_config_defaults(&pcfg, grid) != 0) return -1;
    pcfg.erb_fwd = erb_fwd;
    pcfg.erb_inv = erb_inv;
    if (dfn2_prepost_get_mem_size(&pcfg, &preq) != 0) return -1;
    if (posix_memalign(&d->prepost_pool, (size_t)preq.alignment, (size_t)preq.bytes) != 0) return -1;
    d->prepost = dfn2_prepost_init(d->prepost_pool, (size_t)preq.bytes, &pcfg);
    if (!d->prepost) return -1;

    d->spec_re = malloc(n_bins * sizeof(float));
    d->spec_im = malloc(n_bins * sizeof(float));
    d->enh_re = malloc(n_bins * sizeof(float));
    d->enh_im = malloc(n_bins * sizeof(float));
    d->hop_in = malloc(hop * sizeof(float));
    d->hop_out = malloc(hop * sizeof(float));
    if (!d->spec_re || !d->spec_im || !d->enh_re || !d->enh_im || !d->hop_in || !d->hop_out) return -1;
    return 0;
}

static int board_pre(BoardDriver *d, const int16_t *pcm_in, DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    int hop = d->grid.hop, k;
    for (k = 0; k < hop; ++k) d->hop_in[k] = (float)pcm_in[k] / 32768.0f;
    if (d->hpf) hpf_process(d->hpf, d->hop_in, hop);
    ac_stft_push_planar(d->stft, d->hop_in, d->spec_re, d->spec_im);
    if (dfn2_prepost_pre_process_freq(d->prepost, d->spec_re, d->spec_im) != 1) return -1;
    if (dfn2_prepost_frame_open(d->prepost, in, out) != 0) return -1;
    return 1;
}

static int board_post(BoardDriver *d, int ai_rc, int16_t *pcm_out, int *valid_out) {
    int rc, prc, valid = 0, hop = d->grid.hop, k;
    rc = (ai_rc == 0) ? dfn2_prepost_frame_finish(d->prepost) : -1;
    if (rc != 0) dfn2_prepost_frame_bypass(d->prepost);
    prc = dfn2_prepost_post_process_freq(d->prepost, d->enh_re, d->enh_im, &valid);
    if (prc == 0) {
        ac_istft_push_planar(d->istft, d->enh_re, d->enh_im, d->hop_out);
    } else {
        memset(d->hop_out, 0, (size_t)hop * sizeof(float));
        valid = 0;
    }
    for (k = 0; k < hop; ++k) {
        float v = d->hop_out[k] * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm_out[k] = (int16_t)v;
    }
    if (valid_out) *valid_out = valid;
    return valid ? 1 : 0;
}

static void board_reset(BoardDriver *d) {
    if (d->hpf) hpf_reset(d->hpf);
    ac_stft_reset(d->stft);
    ac_istft_reset(d->istft);
    dfn2_prepost_reset(d->prepost);
}

static unsigned int board_generation(const BoardDriver *d) {
    return dfn2_prepost_state_generation(d->prepost);
}

/* ---- comparison run over one grid/variant ---- */

typedef struct Stats {
    long long hops;
    long long mismatches;
    long long feat_mismatches;
    long long first_valid_a, first_valid_b;
} Stats;

static int cmp_floats_exact(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

/* inject_shift: 0 = correct comparison; 1 = deliberately compare driver B's PCM against the
 * PREVIOUS hop's buffer (simulating a one-hop synthesis delay) to prove this harness detects it. */
static int run_matrix(const DFN2GridSpec *spec, const char *label, int inject_shift,
                      int hpf_enable, float hpf_cutoff_hz) {
    DFN2Grid grid;
    BoardDriver A;
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    DFN2Stream *B;
    void *pool_b;
    float *erb_fwd, *erb_inv;
    size_t n_bins, n_erb;
    int hop;
    long long ghop = 0;
    Stats st;
    int16_t *pcm_in, *pcm_out_a, *pcm_out_b, *prev_pcm_out_b;
    long long L, D;
    long long warm_hops, post_reset_hops, drain_hops;
    long long fault_hop_index = -1;

    memset(&st, 0, sizeof(st));
    st.first_valid_a = st.first_valid_b = -1;

    if (dfn2_grid_make(&grid, spec) != DFN2_GRID_OK) {
        fprintf(stderr, "[%s] dfn2_grid_make failed\n", label);
        return -1;
    }
    n_bins = (size_t)grid.n_bins;
    n_erb = (size_t)grid.n_erb;
    hop = grid.hop;
    L = grid.mask_lookahead;
    D = grid.df_lookahead;

    erb_fwd = malloc(n_bins * n_erb * sizeof(float));
    erb_inv = malloc(n_erb * n_bins * sizeof(float));
    {
        size_t i;
        for (i = 0; i < n_bins * n_erb; ++i) erb_fwd[i] = 1.0f / (float)n_erb;
        for (i = 0; i < n_erb * n_bins; ++i) erb_inv[i] = 1.0f;
    }

    if (board_init(&A, &grid, erb_fwd, erb_inv, hpf_enable, hpf_cutoff_hz) != 0) {
        fprintf(stderr, "[%s] board_init failed\n", label);
        return -1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.hpf_enable = hpf_enable;
    cfg.hpf_cutoff_hz = hpf_cutoff_hz;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0) {
        fprintf(stderr, "[%s] dfn2_stream_get_mem_size failed\n", label);
        return -1;
    }
    if (posix_memalign(&pool_b, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    B = dfn2_stream_init_ex(&cfg, pool_b, (size_t)req.bytes, &req);
    if (!B) {
        fprintf(stderr, "[%s] dfn2_stream_init_ex failed\n", label);
        return -1;
    }

    pcm_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_out_a = calloc((size_t)hop, sizeof(int16_t));
    pcm_out_b = calloc((size_t)hop, sizeof(int16_t));
    prev_pcm_out_b = calloc((size_t)hop, sizeof(int16_t));

    warm_hops = L + 5;          /* warm-up (if any) + 5 steady hops, one of them fault-injected */
    fault_hop_index = L + 2;    /* a steady hop, well past warm-up */
    post_reset_hops = L + 4;    /* after reset: warm-up again + 4 steady hops */
    drain_hops = L + D + 1;     /* documented tail-drain length */

    {
        long long phase, phase_hops, hop_in_phase;
        int did_reset = 0;
        long long total_phases = 2; /* pre-reset run, then reset + post-reset run; drain appended after */
        for (phase = 0; phase < total_phases + 1; ++phase) {
            if (phase == 0) phase_hops = warm_hops;
            else if (phase == 1) { board_reset(&A); dfn2_stream_reset(B); did_reset = 1; phase_hops = post_reset_hops; }
            else phase_hops = drain_hops;
            (void)did_reset;

            for (hop_in_phase = 0; hop_in_phase < phase_hops; ++hop_in_phase, ++ghop) {
                DFN2HopIO io_b;
                DFN2PrepostInputs in_a;
                DFN2PrepostOutputs out_a;
                int pre_rc_a, pre_rc_b, post_rc_a, post_rc_b, ai_rc;
                int valid_a = 0, valid_b = 0;
                int force_fail = (phase == 0 && hop_in_phase == fault_hop_index) ? 1 : 0;
                int zero_hop = (phase == 2);
                int k;

                if (zero_hop) memset(pcm_in, 0, (size_t)hop * sizeof(int16_t));
                else for (k = 0; k < hop; ++k)
                    pcm_in[k] = (int16_t)(((long long)k * 7 + ghop * 3) % 101 - 50);

                pre_rc_a = board_pre(&A, pcm_in, &in_a, &out_a);
                pre_rc_b = dfn2_stream_pre_process(B, pcm_in, &io_b);

                if (pre_rc_a != pre_rc_b) {
                    fprintf(stderr, "[%s] hop %lld: pre_rc mismatch A=%d B=%d\n", label, ghop, pre_rc_a, pre_rc_b);
                    st.mismatches++;
                }
                if (pre_rc_a == 1 && pre_rc_b == 1) {
                    if (in_a.erb_window_elements != io_b.inputs.erb_window_elements ||
                        !cmp_floats_exact(in_a.erb_window, io_b.inputs.erb_window, in_a.erb_window_elements)) {
                        fprintf(stderr, "[%s] hop %lld: feature window (erb_window) mismatch A vs B\n", label, ghop);
                        st.feat_mismatches++;
                    }
                }

                /* Same fake, called once per side, both with the identical (ghop, force_fail). */
                ai_rc = shared_fake(ghop, force_fail, &out_a);
                {
                    /* Re-run the identical deterministic fake against B's own output views too --
                     * same ghop/force_fail, so both sides get field-for-field identical values. */
                    int ai_rc_b = shared_fake(ghop, force_fail, &io_b.outputs);
                    if (ai_rc_b != ai_rc) { fprintf(stderr, "[%s] hop %lld: fake rc mismatch\n", label, ghop); st.mismatches++; }
                }

                post_rc_a = board_post(&A, ai_rc, pcm_out_a, &valid_a);
                post_rc_b = dfn2_stream_post_process(B, ai_rc, pcm_out_b);
                valid_b = (post_rc_b == 1);

                if (st.first_valid_a < 0 && valid_a) st.first_valid_a = ghop;
                if (st.first_valid_b < 0 && valid_b) st.first_valid_b = ghop;

                if (post_rc_a != post_rc_b) {
                    fprintf(stderr, "[%s] hop %lld: post_rc mismatch A=%d B=%d\n", label, ghop, post_rc_a, post_rc_b);
                    st.mismatches++;
                }

                {
                    const int16_t *b_for_compare = inject_shift ? prev_pcm_out_b : pcm_out_b;
                    if (memcmp(pcm_out_a, b_for_compare, (size_t)hop * sizeof(int16_t)) != 0) {
                        fprintf(stderr, "[%s] hop %lld: PCM mismatch (inject_shift=%d)\n", label, ghop, inject_shift);
                        st.mismatches++;
                    }
                }
                memcpy(prev_pcm_out_b, pcm_out_b, (size_t)hop * sizeof(int16_t));

                if (board_generation(&A) != dfn2_stream_state_generation(B)) {
                    fprintf(stderr, "[%s] hop %lld: generation mismatch A=%u B=%u\n",
                           label, ghop, board_generation(&A), dfn2_stream_state_generation(B));
                    st.mismatches++;
                }
                st.hops++;
            }
        }
    }

    printf("[%s] hops=%lld mismatches=%lld feat_mismatches=%lld first_valid A=%lld B=%lld "
          "(L=%lld D=%lld hpf=%d) %s\n",
          label, st.hops, st.mismatches, st.feat_mismatches, st.first_valid_a, st.first_valid_b,
          L, D, hpf_enable, (st.mismatches == 0 && st.feat_mismatches == 0) ? "PASS" : "FAIL");

    free(pcm_in); free(pcm_out_a); free(pcm_out_b); free(prev_pcm_out_b);
    free(erb_fwd); free(erb_inv);
    return (st.mismatches == 0 && st.feat_mismatches == 0) ? 0 : 1;
}

int main(int argc, char **argv) {
    DFN2GridSpec row16k, row48k;
    int inject_shift = (argc > 1 && strcmp(argv[1], "--inject-shift") == 0);
    int failures = 0;

    memset(&row16k, 0, sizeof(row16k));
    row16k.sample_rate = 16000; row16k.n_fft = 256; row16k.n_erb = 24; row16k.df_bins = 96;
    row16k.df_order = 5; row16k.mask_lookahead = 0; row16k.df_lookahead = 0; row16k.frames_per_call = 1;
    row16k.erb_norm_tau_sec = 1.0f; row16k.erb_norm_init_lo_db = -60.0f; row16k.erb_norm_init_hi_db = -90.0f;
    row16k.erb_norm_scale_db = 40.0f; row16k.spec_norm_tau_sec = 1.0f; row16k.spec_norm_init_lo = 0.001f;
    row16k.spec_norm_init_hi = 0.0001f;

    memset(&row48k, 0, sizeof(row48k));
    row48k.sample_rate = 48000; row48k.n_fft = 1024; row48k.n_erb = 32; row48k.df_bins = 96;
    row48k.mask_lookahead = 1; row48k.frames_per_call = 1;
#ifdef DFN2_MASK_ONLY
    row48k.df_order = 0; row48k.df_lookahead = 0;
#else
    row48k.df_order = 5; row48k.df_lookahead = 1;
#endif
    row48k.erb_norm_tau_sec = 1.0f; row48k.erb_norm_init_lo_db = -60.0f; row48k.erb_norm_init_hi_db = -90.0f;
    row48k.erb_norm_scale_db = 40.0f; row48k.spec_norm_tau_sec = 1.0f; row48k.spec_norm_init_lo = 0.001f;
    row48k.spec_norm_init_hi = 0.0001f;

    failures += (run_matrix(&row16k, "16k/256 L0D0", inject_shift, 0, 0.0f) != 0);
    failures += (run_matrix(&row48k, "48k/1024 L1D(0or1)", inject_shift, 0, 0.0f) != 0);
    /* HPF-enabled coverage (project rule: a carved-but-never-exercised path is not covered;
     * code review flagged dfn2_stream's hpf_enable=1 path -- carve, hpf_init, hpf_process,
     * hpf_reset -- as untested by every binary in this directory before this run). 80 Hz matches
     * the board sample's own --hpf default cutoff. */
    failures += (run_matrix(&row48k, "48k/1024 L1D(0or1) hpf80", inject_shift, 1, 80.0f) != 0);

    if (inject_shift) {
        /* This run is EXPECTED to fail -- it proves the comparator has teeth. */
        if (failures == 0) {
            fprintf(stderr, "INJECT-SHIFT CHECK FAILED: harness stayed green under a deliberately "
                            "injected one-hop comparison shift -- the comparison has no teeth\n");
            return 1;
        }
        printf("inject-shift check: comparator correctly went RED under the injected shift\n");
        return 0;
    }

    return failures ? 1 : 0;
}
