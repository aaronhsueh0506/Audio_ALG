/**
 * aec_nr_pipeline.c - Linear AEC -> NR -> RES Pipeline (Version A: malloc)
 *
 * Three-stage speech enhancement:
 *   Stage 1: Linear AEC (PBFDKF + shadow, no RES) -> error signal + context
 *   Stage 2: NR (MMSE-LSA + MCRA + SPP) -> denoised + per-bin gain
 *   Stage 3: RES (NR-gain-corrected echo PSD) -> final output
 *
 * All modules use unified 10ms hop (160 samples @ 16kHz).
 *
 * Compile:
 *   make -C ../lib/aec/c_impl lib
 *   make -C ../lib/nr/c_impl lib
 *   make aec_nr_pipeline
 *
 * Usage:
 *   ./aec_nr_pipeline <mic.wav> <ref.wav> <output.wav> [preset]
 *   preset: mild, balanced (default), aggressive, maximum
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* AEC */
#include "aec.h"
#include "aec_types.h"
#include "res_filter.h"

/* NR */
#include "mmse_lsa_denoiser.h"
#include "mmse_lsa_types.h"

/* WAV I/O (from AEC example) */
#include "wav_io.h"

static AecPreset parse_preset(const char* name) {
    if (strcmp(name, "mild") == 0)       return AEC_PRESET_MILD;
    if (strcmp(name, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    if (strcmp(name, "maximum") == 0)    return AEC_PRESET_MAXIMUM;
    return AEC_PRESET_BALANCED;
}

static void print_usage(const char* prog) {
    printf("Linear AEC -> NR -> RES Pipeline\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav> [preset]\n\n", prog);
    printf("Presets: mild, balanced (default), aggressive, maximum\n");
    printf("Options:\n");
    printf("  --aec-only    Run AEC only (skip NR + RES)\n");
    printf("  --nr-gain dB  NR minimum gain in dB (default: -15)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    /* Parse options */
    AecPreset preset = AEC_PRESET_BALANCED;
    int aec_only = 0;
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

    /* Open WAV files */
    WavReader* mic_reader = wav_open_read(mic_path);
    WavReader* ref_reader = wav_open_read(ref_path);
    if (!mic_reader || !ref_reader) {
        fprintf(stderr, "Error: Failed to open input files\n");
        return 1;
    }

    if (mic_reader->info.sample_rate != ref_reader->info.sample_rate) {
        fprintf(stderr, "Error: Sample rate mismatch (%d vs %d)\n",
                mic_reader->info.sample_rate, ref_reader->info.sample_rate);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int sample_rate = mic_reader->info.sample_rate;
    int num_samples = (mic_reader->info.num_samples < ref_reader->info.num_samples)
                      ? mic_reader->info.num_samples : ref_reader->info.num_samples;

    printf("Linear AEC -> NR -> RES Pipeline\n");
    printf("=================================\n");
    printf("Input:  %s (%d samples, %.2fs)\n", mic_path, num_samples,
           (float)num_samples / sample_rate);
    printf("Ref:    %s\n", ref_path);
    printf("Output: %s\n", out_path);
    printf("Rate:   %d Hz\n", sample_rate);
    printf("Preset: %s\n", aec_preset_name(preset));
    if (!aec_only)
        printf("NR:     g_min=%.1f dB\n", nr_g_min_db);
    else
        printf("Mode:   AEC only\n");
    printf("\n");

    /* === Create AEC (linear mode: no internal RES) === */
    AecConfig aec_cfg = aec_config_from_preset(preset, sample_rate);
    aec_cfg.enable_res = 0;  /* Linear mode — RES handled externally */
    Aec* aec = aec_create(&aec_cfg);
    if (!aec) {
        fprintf(stderr, "Error: Failed to create AEC\n");
        return 1;
    }

    /* Context for external RES */
    AecResContext* ctx = aec_context_create(aec);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create AEC context\n");
        aec_destroy(aec);
        return 1;
    }

    /* === Create NR (10ms hop, matching AEC) === */
    MmseLsaConfig nr_cfg = mmse_lsa_default_config(sample_rate);
    nr_cfg.g_min_db = nr_g_min_db;
    MmseLsaDenoiser* nr = NULL;
    if (!aec_only) {
        nr = mmse_lsa_create(&nr_cfg);
        if (!nr) {
            fprintf(stderr, "Error: Failed to create NR\n");
            aec_context_destroy(ctx);
            aec_destroy(aec);
            return 1;
        }
    }

    /* === Create standalone RES === */
    ResConfig res_cfg = res_config_from_aec(&aec_cfg);
    ResFilter* res = NULL;
    if (!aec_only) {
        res = res_create(&res_cfg);
        if (!res) {
            fprintf(stderr, "Error: Failed to create RES\n");
            mmse_lsa_destroy(nr);
            aec_context_destroy(ctx);
            aec_destroy(aec);
            return 1;
        }
    }

    int hop = aec_get_hop_size(aec);
    int nf = aec_cfg.n_freqs;

    /* Open output WAV */
    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: Failed to create output file\n");
        if (res) res_destroy(res);
        if (nr) mmse_lsa_destroy(nr);
        aec_context_destroy(ctx);
        aec_destroy(aec);
        return 1;
    }

    /* Allocate processing buffers */
    float* mic_buf = (float*)malloc(hop * sizeof(float));
    float* ref_buf = (float*)malloc(hop * sizeof(float));
    float* aec_out = (float*)malloc(hop * sizeof(float));
    float* nr_out  = (float*)malloc(hop * sizeof(float));
    float* res_out = (float*)malloc(hop * sizeof(float));
    Complex* corrected_echo = (Complex*)malloc(nf * sizeof(Complex));

    /* NR OLA delay: save previous AEC context for alignment */
    AecResContext* prev_ctx = aec_context_create(aec);
    int have_prev_ctx = 0;

    /* === Processing loop === */
    int processed = 0;
    float max_erle = 0.0f;

    printf("Processing");
    fflush(stdout);

    while (processed + hop <= num_samples) {
        /* Read input */
        wav_read_float(mic_reader, mic_buf, hop);
        wav_read_float(ref_reader, ref_buf, hop);

        /* Stage 1: Linear AEC (with context output) */
        aec_process_ex(aec, mic_buf, ref_buf, aec_out, ctx);

        if (aec_only) {
            /* AEC only mode */
            wav_write_float(writer, aec_out, hop);
        } else {
            /* Stage 2: NR (MMSE-LSA) */
            mmse_lsa_process(nr, aec_out, nr_out);

            /* Stage 3: RES with NR-gain-corrected echo PSD
             *
             * NR has 1-frame OLA delay: nr_out[i] corresponds to aec frame[i-1].
             * Use prev_ctx (delayed by 1 frame) for alignment.
             */
            if (have_prev_ctx) {
                const float* gain = mmse_lsa_get_gain(nr, NULL);
                if (gain) {
                    /* Correct echo spectrum: NR already attenuated these frequencies */
                    for (int k = 0; k < nf; k++) {
                        corrected_echo[k].r = prev_ctx->echo_spec_re[k] * gain[k];
                        corrected_echo[k].i = prev_ctx->echo_spec_im[k] * gain[k];
                    }

                    /* Build far_spec and near_spec as Complex from prev_ctx */
                    Complex* far_spec_c = (Complex*)malloc(nf * sizeof(Complex));
                    Complex* near_spec_c = (Complex*)malloc(nf * sizeof(Complex));
                    for (int k = 0; k < nf; k++) {
                        far_spec_c[k].r = prev_ctx->far_spec_re[k];
                        far_spec_c[k].i = prev_ctx->far_spec_im[k];
                        near_spec_c[k].r = prev_ctx->near_spec_re[k];
                        near_spec_c[k].i = prev_ctx->near_spec_im[k];
                    }

                    res_process(res, nr_out,
                                corrected_echo,
                                far_spec_c,
                                near_spec_c,
                                prev_ctx->far_power,
                                prev_ctx->filter_converged,
                                prev_ctx->erle_factor,
                                prev_ctx->dt_indicator,
                                prev_ctx->over_sub,
                                res_out);

                    free(far_spec_c);
                    free(near_spec_c);

                    wav_write_float(writer, res_out, hop);
                } else {
                    /* No gain available yet — pass NR output directly */
                    wav_write_float(writer, nr_out, hop);
                }
            } else {
                /* First frame: NR outputs silence/initial, just write it */
                wav_write_float(writer, nr_out, hop);
            }

            /* Swap ctx → prev_ctx for next frame alignment */
            AecResContext* tmp = prev_ctx;
            prev_ctx = ctx;
            ctx = tmp;
            have_prev_ctx = 1;
        }

        float erle = aec_get_erle(aec);
        if (erle > max_erle) max_erle = erle;

        processed += hop;
        if (processed % sample_rate == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    printf(" Done!\n\n");
    printf("Results:\n");
    printf("  Processed: %d samples (%.2fs)\n", processed, (float)processed / sample_rate);
    printf("  ERLE:      %.1f dB\n", aec_get_erle(aec));
    printf("  Max ERLE:  %.1f dB\n", max_erle);
    printf("  Converged: %s\n", aec_is_converged(aec) ? "yes" : "no");

    /* Cleanup */
    free(corrected_echo);
    free(mic_buf);
    free(ref_buf);
    free(aec_out);
    free(nr_out);
    free(res_out);

    aec_context_destroy(prev_ctx);
    aec_context_destroy(ctx);
    if (res) res_destroy(res);
    if (nr) mmse_lsa_destroy(nr);
    aec_destroy(aec);

    wav_close_read(mic_reader);
    wav_close_read(ref_reader);
    wav_close_write(writer);

    printf("\nOutput written to: %s\n", out_path);
    return 0;
}
