/**
 * aec_nr_pipeline_static.c - Linear AEC -> NR -> RES Pipeline
 *                            (Version B: Static Memory)
 *
 * Same processing as Version A (aec_nr_pipeline.c), but all modules
 * are placed in a single pre-allocated memory pool.
 *
 * Memory management:
 *   1. Query each module: _get_mem_size()
 *   2. Allocate one contiguous pool (malloc on desktop, PA/VA on Novatek)
 *   3. Slice pool via pointer arithmetic: _init()
 *   4. Process frames (identical to Version A)
 *   5. Free the single pool (or nvt_mem_free on Novatek)
 *
 * No internal malloc is called by any module when using _init().
 *
 * Compile:
 *   make -C ../lib/aec/c_impl lib
 *   make -C ../lib/nr/c_impl lib
 *   make aec_nr_pipeline_static
 *
 * Usage:
 *   ./aec_nr_pipeline_static <mic.wav> <ref.wav> <output.wav> [preset]
 *   ./aec_nr_pipeline_static --print-mem-size [preset]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* AEC */
#include "aec.h"
#include "aec_types.h"
#include "res_filter.h"
#include "fft_wrapper.h"  /* ALIGN16 */

/* NR */
#include "mmse_lsa_denoiser.h"
#include "mmse_lsa_types.h"

/* HPF */
#include "hpf.h"

/* WAV I/O (from AEC example) */
#include "wav_io.h"

static AecPreset parse_preset(const char* name) {
    if (strcmp(name, "mild") == 0)       return AEC_PRESET_MILD;
    if (strcmp(name, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    if (strcmp(name, "maximum") == 0)    return AEC_PRESET_MAXIMUM;
    return AEC_PRESET_BALANCED;
}

static void print_usage(const char* prog) {
    printf("Linear AEC -> NR -> RES Pipeline (Static Memory)\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav> [preset]\n\n", prog);
    printf("Presets: mild, balanced (default), aggressive, maximum\n");
    printf("Options:\n");
    printf("  --aec-only        Run AEC only (skip NR + RES)\n");
    printf("  --nr-gain dB      NR minimum gain in dB (default: -15)\n");
    printf("  --print-mem-size  Print memory budget and exit\n");
}

/**
 * Compute memory sizes and print budget.
 * Returns total bytes needed.
 */
static size_t compute_memory_budget(const AecConfig* aec_cfg,
                                    const MmseLsaConfig* nr_cfg,
                                    const ResConfig* res_cfg,
                                    int n_freqs,
                                    int hop,
                                    int aec_only,
                                    int print)
{
    size_t hpf_size = hpf_get_mem_size() * 2;  /* mic + ref HPF */
    size_t aec_size = aec_get_mem_size(aec_cfg);
    size_t ctx_size = aec_context_get_mem_size(n_freqs);
    size_t ctx2_size = ctx_size;  /* prev_ctx for NR delay */
    size_t nr_size = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);
    size_t res_size = aec_only ? 0 : res_get_mem_size(res_cfg);

    /* Pipeline working buffers */
    size_t pipe_size = 0;
    pipe_size += ALIGN16(hop * sizeof(float));   /* mic_buf */
    pipe_size += ALIGN16(hop * sizeof(float));   /* ref_buf */
    pipe_size += ALIGN16(hop * sizeof(float));   /* aec_out */
    pipe_size += ALIGN16(hop * sizeof(float));   /* nr_out */
    pipe_size += ALIGN16(hop * sizeof(float));   /* res_out */
    pipe_size += ALIGN16(n_freqs * sizeof(Complex)); /* corrected_echo */
    pipe_size += ALIGN16(n_freqs * sizeof(Complex)); /* far_spec_c */
    pipe_size += ALIGN16(n_freqs * sizeof(Complex)); /* near_spec_c */

    size_t total = ALIGN16(hpf_size)
                 + ALIGN16(aec_size)
                 + ALIGN16(ctx_size)
                 + ALIGN16(ctx2_size)
                 + ALIGN16(nr_size)
                 + ALIGN16(res_size)
                 + ALIGN16(pipe_size);

    if (print) {
        printf("Memory Budget (Static Memory Version B)\n");
        printf("========================================\n");
        printf("  HPF (mic + ref):  %6zu bytes (%5.1f KB)\n", hpf_size, hpf_size / 1024.0);
        printf("  AEC (linear):     %6zu bytes (%5.1f KB)\n", aec_size, aec_size / 1024.0);
        printf("  AEC Context x2:   %6zu bytes (%5.1f KB)\n", ctx_size + ctx2_size, (ctx_size + ctx2_size) / 1024.0);
        if (!aec_only) {
            printf("  NR (MMSE-LSA):    %6zu bytes (%5.1f KB)\n", nr_size, nr_size / 1024.0);
            printf("  RES (standalone): %6zu bytes (%5.1f KB)\n", res_size, res_size / 1024.0);
        }
        printf("  Pipeline buffers: %6zu bytes (%5.1f KB)\n", pipe_size, pipe_size / 1024.0);
        printf("  ----------------------------------------\n");
        printf("  Total (aligned):  %6zu bytes (%5.1f KB)\n", total, total / 1024.0);
        printf("\n");
    }

    return total;
}

int main(int argc, char* argv[]) {
    /* Parse options first (may be --print-mem-size only) */
    AecPreset preset = AEC_PRESET_BALANCED;
    int aec_only = 0;
    float nr_g_min_db = -15.0f;
    int print_mem_only = 0;
    int sample_rate = 16000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_mem_only = 1;
        } else if (strcmp(argv[i], "--aec-only") == 0) {
            aec_only = 1;
        } else if (strcmp(argv[i], "--nr-gain") == 0 && i + 1 < argc) {
            nr_g_min_db = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
            sample_rate = atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !print_mem_only) {
            /* positional args handled below */
        } else if (argv[i][0] != '-') {
            preset = parse_preset(argv[i]);
        }
    }

    /* Build configs */
    AecConfig aec_cfg = aec_config_from_preset(preset, sample_rate);
    aec_cfg.enable_res = 0;  /* Linear mode — RES handled externally */

    MmseLsaConfig nr_cfg = mmse_lsa_default_config(sample_rate);
    nr_cfg.g_min_db = nr_g_min_db;

    ResConfig res_cfg = res_config_from_aec(&aec_cfg);

    int hop = aec_cfg.hop_size;
    int nf = aec_cfg.n_freqs;

    /* --print-mem-size mode */
    if (print_mem_only) {
        compute_memory_budget(&aec_cfg, &nr_cfg, &res_cfg, nf, hop, aec_only, 1);
        return 0;
    }

    /* Normal mode: need mic, ref, output paths */
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    /* Re-parse positional options after paths */
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--aec-only") == 0) {
            aec_only = 1;
        } else if (strcmp(argv[i], "--nr-gain") == 0 && i + 1 < argc) {
            nr_g_min_db = (float)atof(argv[++i]);
            nr_cfg.g_min_db = nr_g_min_db;
        } else if (strcmp(argv[i], "--print-mem-size") == 0) {
            /* already handled */
        } else {
            preset = parse_preset(argv[i]);
            aec_cfg = aec_config_from_preset(preset, sample_rate);
            aec_cfg.enable_res = 0;
            res_cfg = res_config_from_aec(&aec_cfg);
            hop = aec_cfg.hop_size;
            nf = aec_cfg.n_freqs;
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

    sample_rate = mic_reader->info.sample_rate;
    int num_samples = (mic_reader->info.num_samples < ref_reader->info.num_samples)
                      ? mic_reader->info.num_samples : ref_reader->info.num_samples;

    /* Recompute configs with actual sample rate */
    aec_cfg = aec_config_from_preset(preset, sample_rate);
    aec_cfg.enable_res = 0;
    nr_cfg = mmse_lsa_default_config(sample_rate);
    nr_cfg.g_min_db = nr_g_min_db;
    res_cfg = res_config_from_aec(&aec_cfg);
    hop = aec_cfg.hop_size;
    nf = aec_cfg.n_freqs;

    printf("Linear AEC -> NR -> RES Pipeline (Static Memory)\n");
    printf("=================================================\n");
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

    /* === Compute memory budget === */
    size_t total_mem = compute_memory_budget(&aec_cfg, &nr_cfg, &res_cfg,
                                             nf, hop, aec_only, 1);

    /* === Allocate single pool === */
    /*
     * Desktop: malloc
     * Novatek: uint32_t pa; void* pool = nvt_mem_alloc(total_mem, &pa);
     */
    void* pool = malloc(total_mem);
    if (!pool) {
        fprintf(stderr, "Error: Failed to allocate memory pool (%zu bytes)\n", total_mem);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }
    memset(pool, 0, total_mem);

    /* === Slice pool into modules === */
    uint8_t* ptr = (uint8_t*)pool;

    /* HPF (mic + ref, applied before AEC) */
    size_t hpf_sz = hpf_get_mem_size();
    Hpf* hp_mic = hpf_init(ptr, hpf_sz, 80.0f, sample_rate);
    ptr += hpf_sz;
    Hpf* hp_ref = hpf_init(ptr, hpf_sz, 80.0f, sample_rate);
    ptr += hpf_sz;
    if (!hp_mic || !hp_ref) {
        fprintf(stderr, "Error: Failed to init HPF in pool\n");
        free(pool);
        return 1;
    }

    /* AEC */
    size_t aec_size = aec_get_mem_size(&aec_cfg);
    Aec* aec = aec_init(ptr, aec_size, &aec_cfg);
    ptr += ALIGN16(aec_size);
    if (!aec) {
        fprintf(stderr, "Error: Failed to init AEC in pool\n");
        free(pool);
        return 1;
    }

    /* AEC Context (current) */
    size_t ctx_size = aec_context_get_mem_size(nf);
    AecResContext* ctx = aec_context_init(ptr, ctx_size, nf);
    ptr += ALIGN16(ctx_size);

    /* AEC Context (previous — for NR OLA delay alignment) */
    AecResContext* prev_ctx = aec_context_init(ptr, ctx_size, nf);
    ptr += ALIGN16(ctx_size);

    if (!ctx || !prev_ctx) {
        fprintf(stderr, "Error: Failed to init AEC contexts in pool\n");
        free(pool);
        return 1;
    }

    /* NR */
    MmseLsaDenoiser* nr = NULL;
    if (!aec_only) {
        size_t nr_size = mmse_lsa_get_mem_size(&nr_cfg);
        nr = mmse_lsa_init(ptr, nr_size, &nr_cfg);
        ptr += ALIGN16(nr_size);
        if (!nr) {
            fprintf(stderr, "Error: Failed to init NR in pool\n");
            free(pool);
            return 1;
        }
    }

    /* RES */
    ResFilter* res = NULL;
    if (!aec_only) {
        size_t res_size_val = res_get_mem_size(&res_cfg);
        res = res_init(ptr, res_size_val, &res_cfg);
        ptr += ALIGN16(res_size_val);
        if (!res) {
            fprintf(stderr, "Error: Failed to init RES in pool\n");
            free(pool);
            return 1;
        }
    }

    /* Pipeline working buffers (also from pool) */
    float* mic_buf = (float*)ptr;         ptr += ALIGN16(hop * sizeof(float));
    float* ref_buf = (float*)ptr;         ptr += ALIGN16(hop * sizeof(float));
    float* aec_out = (float*)ptr;         ptr += ALIGN16(hop * sizeof(float));
    float* nr_out  = (float*)ptr;         ptr += ALIGN16(hop * sizeof(float));
    float* res_out = (float*)ptr;         ptr += ALIGN16(hop * sizeof(float));
    Complex* corrected_echo = (Complex*)ptr; ptr += ALIGN16(nf * sizeof(Complex));
    Complex* far_spec_c = (Complex*)ptr;  ptr += ALIGN16(nf * sizeof(Complex));
    Complex* near_spec_c = (Complex*)ptr; ptr += ALIGN16(nf * sizeof(Complex));

    /* Open output WAV */
    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: Failed to create output file\n");
        free(pool);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    /* === Processing loop (identical logic to Version A) === */
    int processed = 0;
    float max_erle = 0.0f;
    int have_prev_ctx = 0;

    printf("Processing");
    fflush(stdout);

    while (processed + hop <= num_samples) {
        /* Read input */
        wav_read_float(mic_reader, mic_buf, hop);
        wav_read_float(ref_reader, ref_buf, hop);

        /* Stage 0: HPF (remove DC/hum before AEC) */
        hpf_process(hp_mic, mic_buf, hop);
        hpf_process(hp_ref, ref_buf, hop);

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
                                prev_ctx->divergence,
                                res_out);

                    wav_write_float(writer, res_out, hop);
                } else {
                    /* No gain available yet — pass NR output directly */
                    wav_write_float(writer, nr_out, hop);
                }
            } else {
                /* First frame: NR outputs silence/initial, just write it */
                wav_write_float(writer, nr_out, hop);
            }

            /* Swap ctx <-> prev_ctx for next frame alignment.
             * Since both are in static memory, swap the pointers only. */
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

    /* === Cleanup === */
    /* No per-module free needed — all in one pool.
     * _destroy() is no-op for static modules (is_static=1). */
    wav_close_read(mic_reader);
    wav_close_read(ref_reader);
    wav_close_write(writer);

    /*
     * Desktop: free the pool
     * Novatek: nvt_mem_free(pool, pa);
     */
    free(pool);

    printf("\nOutput written to: %s\n", out_path);
    return 0;
}
