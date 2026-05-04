/**
 * aec_nr_pipeline.c — Linear AEC -> NR -> RES pipeline (AEC v3.8.2 API).
 *
 * Three-stage speech enhancement (HPF is internal to AEC):
 *   Stage 1: Linear AEC (PBFDKF + shadow, internal HPF, RES skipped)
 *            -> error + AecResContext
 *   Stage 2: NR (MMSE-LSA + MCRA + SPP) — denoised + per-bin gain
 *   Stage 3: External RES (NR-gain-corrected echo PSD) — final output
 *
 * All modules use unified 10 ms hop (160 samples @ 16 kHz).
 *
 * Build:
 *   make libs && make
 *
 * Usage:
 *   ./aec_nr_pipeline <mic.wav> <ref.wav> <out.wav> [preset] [--aec-only] [--nr-gain dB]
 *   preset: mild | balanced (default) | aggressive | maximum
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* AEC v3.8.2 — single header for both Aec + ResFilter + Complex types.
 * HPF is built into AEC (enable_highpass=1 default), so no external HPF. */
#include "aec.h"
#include "res_filter.h"

/* NR */
#include "mmse_lsa_denoiser.h"
#include "mmse_lsa_types.h"

/* WAV I/O lives in AEC example/. */
#include "wav_io.h"

static AecPreset parse_preset(const char* name) {
    if (strcmp(name, "mild")       == 0) return AEC_PRESET_MILD;
    if (strcmp(name, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    if (strcmp(name, "maximum")    == 0) return AEC_PRESET_MAXIMUM;
    return AEC_PRESET_BALANCED;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_MILD:       return "mild";
        case AEC_PRESET_BALANCED:   return "balanced";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        case AEC_PRESET_MAXIMUM:    return "maximum";
        default:                    return "unknown";
    }
}

/* Build a ResFilterConfig that mirrors the AEC's internal RES wiring,
 * so the standalone RES sees the same per-preset tuning.            */
static void build_res_cfg(const AecConfig* cfg, int hop, int blk, int K,
                          ResFilterConfig* rcfg) {
    memset(rcfg, 0, sizeof(*rcfg));
    rcfg->block_size            = blk;
    rcfg->n_freqs               = K;
    rcfg->frame_size            = blk;
    rcfg->hop_size              = hop;
    rcfg->sample_rate           = cfg->sample_rate;
    rcfg->g_min_db              = cfg->res_g_min_db;
    rcfg->over_sub              = cfg->res_over_sub_base;
    rcfg->alpha                 = 0.8f;
    rcfg->enable_cng            = cfg->enable_cng;
    rcfg->max_drop_db_per_frame = 6.0f;
    rcfg->max_rise_db_per_frame = 6.0f;
    rcfg->enable_spectral_floor = 1;
    rcfg->spectral_floor_db     = cfg->res_spectral_floor_db;
    rcfg->ne_protect_db         = cfg->res_ne_protect_db;
    rcfg->enable_reverb         = 1;
    rcfg->reverb_decay          = cfg->res_reverb_decay;
    rcfg->reverb_gain           = cfg->res_reverb_gain;
    rcfg->alpha_echo_psd        = cfg->res_alpha_echo_psd;
    rcfg->alpha_error_psd       = cfg->res_alpha_error_psd;
    rcfg->enr_scale             = cfg->res_enr_scale;
    rcfg->render_min_ne_factor  = 0.5f;
    rcfg->render_dt_gain_ceil   = 0.6f;
}

static void print_usage(const char* prog) {
    printf("Linear AEC -> NR -> RES Pipeline (AEC v3.8.2)\n");
    printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [preset] [options]\n\n", prog);
    printf("Presets: mild, balanced (default), aggressive, maximum\n");
    printf("Options:\n");
    printf("  --aec-only    Skip NR + external RES (AEC linear out only)\n");
    printf("  --nr-gain dB  NR minimum gain (default: -15)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) { print_usage(argv[0]); return 1; }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecPreset preset = AEC_PRESET_BALANCED;
    int   aec_only    = 0;
    float nr_g_min_db = -15.0f;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--aec-only") == 0) {
            aec_only = 1;
        } else if (strcmp(argv[i], "--nr-gain") == 0 && i + 1 < argc) {
            nr_g_min_db = (float)atof(argv[++i]);
        } else {
            preset = parse_preset(argv[i]);
        }
    }

    /* === Open input WAVs === */
    WavReader* mr = wav_open_read(mic_path);
    WavReader* rr = wav_open_read(ref_path);
    if (!mr || !rr) {
        fprintf(stderr, "Error: cannot open input WAVs\n");
        return 1;
    }
    if (mr->info.sample_rate != rr->info.sample_rate) {
        fprintf(stderr, "Error: SR mismatch (%d vs %d)\n",
                mr->info.sample_rate, rr->info.sample_rate);
        wav_close_read(mr); wav_close_read(rr);
        return 1;
    }
    int sr = mr->info.sample_rate;
    int n  = (mr->info.num_samples < rr->info.num_samples)
              ? mr->info.num_samples : rr->info.num_samples;

    printf("Linear AEC -> NR -> RES Pipeline (AEC v3.8.2)\n");
    printf("=============================================\n");
    printf("Input:  %s (%d samples, %.2fs)\n", mic_path, n, (float)n / sr);
    printf("Ref:    %s\n", ref_path);
    printf("Output: %s\n", out_path);
    printf("Rate:   %d Hz\n", sr);
    printf("Preset: %s\n", preset_name(preset));
    if (aec_only) printf("Mode:   AEC-only (no NR/RES)\n");
    else          printf("NR:     g_min=%.1f dB\n", nr_g_min_db);
    printf("\n");

    /* === Configure AEC (linear-only: skip built-in RES, run external) === */
    AecConfig cfg;
    aec_config_from_preset(&cfg, preset, sr);
    cfg.enable_residual_filter = 0;     /* linear-only path */

    Aec aec;
    if (aec_create(&aec, &cfg) != 0) {
        fprintf(stderr, "Error: aec_create failed\n");
        wav_close_read(mr); wav_close_read(rr);
        return 1;
    }
    int hop = aec_hop_size(&aec);
    int K   = aec.n_freqs;
    int blk = aec.block_size;

    /* === NR === */
    MmseLsaConfig nr_cfg = mmse_lsa_default_config(sr);
    nr_cfg.g_min_db = nr_g_min_db;
    MmseLsaDenoiser* nr = NULL;
    if (!aec_only) {
        nr = mmse_lsa_create(&nr_cfg);
        if (!nr) { fprintf(stderr, "Error: mmse_lsa_create failed\n"); aec_destroy(&aec); return 1; }
    }

    /* === External RES (init in place) === */
    ResFilter res;
    int has_res = 0;
    if (!aec_only) {
        ResFilterConfig rcfg;
        build_res_cfg(&cfg, hop, blk, K, &rcfg);
        res_filter_init(&res, &rcfg);
        has_res = 1;
    }

    /* === Output WAV === */
    WavWriter* ww = wav_open_write(out_path, sr, 1);
    if (!ww) {
        fprintf(stderr, "Error: cannot create output WAV\n");
        if (has_res) res_filter_free(&res);
        if (nr) mmse_lsa_destroy(nr);
        aec_destroy(&aec);
        return 1;
    }

    /* Buffers */
    float*   mic_buf       = (float*)malloc((size_t)hop * sizeof(float));
    float*   ref_buf       = (float*)malloc((size_t)hop * sizeof(float));
    float*   linear_out    = (float*)malloc((size_t)hop * sizeof(float));
    float*   nr_out        = (float*)malloc((size_t)hop * sizeof(float));
    float*   res_out       = (float*)malloc((size_t)hop * sizeof(float));
    Complex* corrected_echo = (Complex*)malloc((size_t)K * sizeof(Complex));

    /* Hold previous-frame ctx so RES sees the alignment NR sees (NR has
     * 1-frame OLA latency, so its output lags AEC by 1 hop). */
    AecResContext ctx_now, ctx_prev;
    memset(&ctx_prev, 0, sizeof(ctx_prev));
    int have_prev = 0;

    int processed = 0;
    int frame_idx = 0;

    printf("Processing");
    fflush(stdout);

    while (processed + hop <= n) {
        wav_read_float(mr, mic_buf, hop);
        wav_read_float(rr, ref_buf, hop);

        /* Stage 1: AEC applies internal HPF then linear filter; takes a
         * snapshot of context. linear_out holds pre-RES error (mic -
         * filter echo, post-HPF). */
        aec_process(&aec, mic_buf, ref_buf, linear_out);
        aec_get_res_context(&aec, &ctx_now);

        if (aec_only) {
            wav_write_float(ww, linear_out, hop);
        } else {
            /* Stage 2: NR on linear AEC out */
            mmse_lsa_process(nr, linear_out, nr_out);

            /* Stage 3: external RES with NR-gain-corrected echo PSD.
             * NR has 1-frame latency, so we feed its output paired with
             * the *previous* AEC context. First frame: skip RES (NR
             * output is initial-state silence anyway). */
            if (have_prev) {
                const float* g_nr = mmse_lsa_get_gain(nr, NULL);
                if (g_nr) {
                    /* Multiply previous-frame's echo spectrum by NR gain
                     * so RES sees the residual echo *after* NR has
                     * already attenuated those bins. */
                    for (int k = 0; k < K; k++) {
                        corrected_echo[k].r = ctx_prev.echo_spec[k].r * g_nr[k];
                        corrected_echo[k].i = ctx_prev.echo_spec[k].i * g_nr[k];
                    }
                    ResProcessInputs in = {0};
                    in.error_hop              = nr_out;
                    in.echo_spec              = corrected_echo;
                    in.far_spec               = ctx_prev.far_spec;
                    in.near_spec              = ctx_prev.near_spec;
                    in.error_spec_from_filter = NULL;
                    in.far_power              = ctx_prev.far_power;
                    in.erle_factor            = ctx_prev.erle_factor;
                    in.dt_indicator           = ctx_prev.dt_indicator;
                    in.divergence             = ctx_prev.divergence;
                    in.is_stationary_dt       = ctx_prev.is_stationary_dt;
                    in.saturation_level       = ctx_prev.saturation_level;
                    in.epc_active             = ctx_prev.epc_active;
                    in.shadow_dt              = ctx_prev.shadow_dt;
                    in.erl_estimate           = ctx_prev.erl_estimate;
                    in.filter_converged       = ctx_prev.filter_converged;
                    in.filter_once_converged  = ctx_prev.filter_once_converged;
                    res_filter_process(&res, &in, res_out);
                    wav_write_float(ww, res_out, hop);
                } else {
                    wav_write_float(ww, nr_out, hop);
                }
            } else {
                wav_write_float(ww, nr_out, hop);
            }

            /* Cache current ctx for next frame's RES alignment. The
             * Complex* targets are persistent in Aec (echo_spec /
             * far_spec / near_spec live in main_filter.base.*), so the
             * pointer copy survives one extra aec_process call. */
            ctx_prev = ctx_now;
            have_prev = 1;
        }

        frame_idx++;
        processed += hop;
        if (processed % sr == 0) { printf("."); fflush(stdout); }
    }

    printf(" Done!\n\n");
    printf("Results:\n");
    printf("  Processed: %d frames (%.2f s)\n", frame_idx, (float)processed / sr);
    printf("  Final filter_converged: %s\n", ctx_now.filter_converged ? "yes" : "no");
    printf("  Final erle_factor: %.3f\n", ctx_now.erle_factor);

    /* Cleanup */
    free(mic_buf); free(ref_buf);
    free(linear_out); free(nr_out); free(res_out);
    free(corrected_echo);
    if (has_res) res_filter_free(&res);
    if (nr) mmse_lsa_destroy(nr);
    aec_destroy(&aec);
    wav_close_read(mr); wav_close_read(rr);
    wav_close_write(ww);

    printf("\nOutput written to: %s\n", out_path);
    return 0;
}
