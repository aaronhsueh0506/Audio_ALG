/**
 * aec_nr_pipeline_static.c — AEC + NR Pipeline (Version B: static memory)
 *
 * Same processing as Version A, but all modules are placed in a single
 * pre-allocated memory pool — no internal malloc is called.
 *
 * Two-stage speech enhancement:
 *   Stage 0: AEC with built-in AEC3 post-filter (echo cancel + suppress)
 *             AEC applies mic HPF @ 80 Hz internally (enable_highpass=1 default).
 *   Stage 1: NR (MMSE-LSA, freq-domain OLA managed by caller)
 *
 * Memory management:
 *   1. Query pool size: compute_pool_size()
 *   2. Allocate one contiguous buffer (malloc on host, PA/VA on Novatek)
 *   3. Slice via pointer arithmetic: _init() functions place each module
 *   4. Process frames
 *   5. Free the single pool
 *
 * Requires SE/ repos with static-memory API (AEC_SE_DIR, NR_SE_DIR):
 *   aec_get_mem_size() + aec_init()    — Track H (SE/AEC)
 *   mmse_lsa_get_mem_size() + mmse_lsa_init() — SE/NR
 *   fft_get_mem_size() + fft_init()    — SE/AEC fft_wrapper.h
 *
 * Build:
 *   make libs-static
 *   make aec_nr_pipeline_static
 *
 * Usage:
 *   ./aec_nr_pipeline_static <mic.wav> <ref.wav> <out.wav> [preset]
 *   ./aec_nr_pipeline_static --print-mem-size [preset]
 *   preset: gentle, balanced (default), aggressive
 *   options: --aec-only   --nr-gain <dB>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "aec.h"               /* AEC + fft_wrapper.h (ALIGN16, Complex) */
#include "fft_wrapper.h"       /* fft_get_mem_size, fft_init, fft_forward/inverse */
#include "mmse_lsa_denoiser.h" /* freq-domain NR */
#include "wav_io.h"

/* ------------------------------------------------------------------ */
/* helpers */
/* ------------------------------------------------------------------ */

static AecPreset parse_preset(const char* s) {
    if (strcmp(s, "gentle") == 0)     return AEC_PRESET_GENTLE;
    if (strcmp(s, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    return AEC_PRESET_BALANCED;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_GENTLE:     return "gentle";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        default:                     return "balanced";
    }
}

/* sqrt-Hann window, length n. */
static void make_sqrt_hann(float* w, int n) {
    for (int k = 0; k < n; k++) {
        float h = sinf((float)M_PI * (k + 0.5f) / n);
        w[k] = sqrtf(h);
    }
}

/* Derive standard 10ms-hop frame dimensions from sample rate. */
static void frame_dims(int sr, int* hop, int* frame_sz, int* fft_sz, int* n_freqs) {
    *hop      = (int)(0.01f * sr);
    *frame_sz = 2 * (*hop);
    *fft_sz   = 512;
    while (*fft_sz < *frame_sz) *fft_sz *= 2;
    *n_freqs  = *fft_sz / 2 + 1;
}

/* ------------------------------------------------------------------ */
/* memory budget                                                       */
/* ------------------------------------------------------------------ */

static size_t compute_pool_size(const AecConfig* aec_cfg,
                                const MmseLsaConfig* nr_cfg,
                                int aec_only, int print) {
    int hop, frame_sz, fft_sz, n_freqs;
    frame_dims(aec_cfg->sample_rate, &hop, &frame_sz, &fft_sz, &n_freqs);

    size_t aec_sz  = aec_get_mem_size(aec_cfg);
    size_t fft_mem = aec_only ? 0 : fft_get_mem_size(fft_sz);
    size_t nr_sz   = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);

    size_t pipe = 0;
    pipe += ALIGN16(hop      * sizeof(float));   /* mic_buf */
    pipe += ALIGN16(hop      * sizeof(float));   /* ref_buf */
    pipe += ALIGN16(hop      * sizeof(float));   /* aec_out */
    pipe += ALIGN16(hop      * sizeof(float));   /* out_buf */
    if (!aec_only) {
        pipe += ALIGN16(frame_sz * sizeof(float));   /* analysis_buf */
        pipe += ALIGN16(frame_sz * sizeof(float));   /* synth_buf (OLA) */
        pipe += ALIGN16(frame_sz * sizeof(float));   /* window (sqrt-Hann) */
        pipe += ALIGN16(frame_sz * sizeof(float));   /* fw (windowed frame) */
        pipe += ALIGN16(frame_sz * sizeof(float));   /* ifft_buf */
        pipe += ALIGN16(n_freqs  * sizeof(Complex)); /* spec_in */
        pipe += ALIGN16(n_freqs  * sizeof(Complex)); /* spec_out */
    }

    size_t total = ALIGN16(aec_sz)
                 + ALIGN16(fft_mem)
                 + ALIGN16(nr_sz)
                 + ALIGN16(pipe);

    if (print) {
        printf("Memory Budget (Static Pipeline)\n");
        printf("================================\n");
        printf("  AEC:            %6zu bytes (%5.1f KB)\n",
               aec_sz, aec_sz / 1024.0);
        if (!aec_only) {
            printf("  FFT (NR OLA):   %6zu bytes (%5.1f KB)\n",
                   fft_mem, fft_mem / 1024.0);
            printf("  NR (MMSE-LSA):  %6zu bytes (%5.1f KB)\n",
                   nr_sz, nr_sz / 1024.0);
        }
        printf("  Pipeline bufs:  %6zu bytes (%5.1f KB)\n",
               pipe, pipe / 1024.0);
        printf("  --------------------------------\n");
        printf("  Total:          %6zu bytes (%5.1f KB)\n",
               total, total / 1024.0);
        printf("\n");
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    AecPreset preset      = AEC_PRESET_BALANCED;
    int       aec_only    = 0;
    float     nr_g_min_db = -15.0f;
    int       print_mem   = 0;
    int       sample_rate = 16000;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--print-mem-size") == 0) print_mem = 1;
        else if (strcmp(argv[i], "--aec-only") == 0)       aec_only  = 1;
        else if (strcmp(argv[i], "--nr-gain") == 0 && i+1 < argc)
            nr_g_min_db = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--sample-rate") == 0 && i+1 < argc)
            sample_rate = atoi(argv[++i]);
        else if (argv[i][0] != '-' && print_mem)
            preset = parse_preset(argv[i]);
    }

    AecConfig aec_cfg;
    aec_config_from_preset(&aec_cfg, preset, sample_rate);
    aec_cfg.enable_res = 1;

    MmseLsaConfig nr_cfg = mmse_lsa_default_config(sample_rate);
    nr_cfg.g_min_db = nr_g_min_db;

    if (print_mem) {
        compute_pool_size(&aec_cfg, &nr_cfg, aec_only, 1);
        return 0;
    }

    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [preset]\n", argv[0]);
        printf("       %s --print-mem-size [preset]\n", argv[0]);
        printf("Presets: gentle, balanced (default), aggressive\n");
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0) aec_only = 1;
        else if (strcmp(argv[i], "--nr-gain") == 0 && i+1 < argc) {
            nr_g_min_db = (float)atof(argv[++i]);
            nr_cfg.g_min_db = nr_g_min_db;
        } else if (argv[i][0] != '-') {
            preset = parse_preset(argv[i]);
            aec_config_from_preset(&aec_cfg, preset, sample_rate);
            aec_cfg.enable_res = 1;
        }
    }

    /* Open inputs */
    WavReader* mic_r = wav_open_read(mic_path);
    WavReader* ref_r = wav_open_read(ref_path);
    if (!mic_r || !ref_r) { fprintf(stderr, "Error: cannot open inputs\n"); return 1; }
    if (mic_r->info.sample_rate != ref_r->info.sample_rate) {
        fprintf(stderr, "Error: sample-rate mismatch\n");
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    sample_rate = mic_r->info.sample_rate;
    int n_samples = (mic_r->info.num_samples < ref_r->info.num_samples)
                  ? mic_r->info.num_samples : ref_r->info.num_samples;

    /* Recompute configs with actual sample rate */
    aec_config_from_preset(&aec_cfg, preset, sample_rate);
    aec_cfg.enable_res = 1;
    nr_cfg = mmse_lsa_default_config(sample_rate);
    nr_cfg.g_min_db = nr_g_min_db;

    int hop, frame_sz, fft_sz, n_freqs;
    frame_dims(sample_rate, &hop, &frame_sz, &fft_sz, &n_freqs);

    printf("AEC+NR Pipeline (Static Memory)\n");
    printf("  Preset: %s   NR: %.0f dB min-gain\n\n",
           preset_name(preset), (double)nr_g_min_db);

    /* === Allocate single pool === */
    size_t total = compute_pool_size(&aec_cfg, &nr_cfg, aec_only, 1);
    void*  pool  = malloc(total);
    if (!pool) { fprintf(stderr, "Error: malloc failed (%zu bytes)\n", total); return 1; }
    memset(pool, 0, total);
    uint8_t* ptr = (uint8_t*)pool;

    /* === Slice pool into modules === */

    /* AEC (Track H static API: returns Aec* placed at ptr[0]) */
    /* Note: AEC applies mic HPF @ 80 Hz internally (enable_highpass=1). */
    size_t aec_sz = aec_get_mem_size(&aec_cfg);
    Aec* aec = aec_init(ptr, aec_sz, &aec_cfg);               ptr += ALIGN16(aec_sz);

    /* FFT + NR (for OLA bridge) */
    FftHandle*       fft = NULL;
    MmseLsaDenoiser* nr  = NULL;
    if (!aec_only) {
        size_t fft_mem = fft_get_mem_size(fft_sz);
        fft = fft_init(ptr, fft_mem, fft_sz);                 ptr += ALIGN16(fft_mem);
        size_t nr_sz = mmse_lsa_get_mem_size(&nr_cfg);
        nr  = mmse_lsa_init(ptr, nr_sz, &nr_cfg);             ptr += ALIGN16(nr_sz);
    }

    /* Working buffers */
    float* mic_buf = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));
    float* ref_buf = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));
    float* aec_out = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));
    float* out_buf = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));

    /* NR OLA buffers */
    float*   analysis = NULL, *synth = NULL, *window = NULL, *fw = NULL, *ifft_buf = NULL;
    Complex* spec_in  = NULL, *spec_out = NULL;
    if (!aec_only) {
        analysis = (float*)ptr;   ptr += ALIGN16(frame_sz * sizeof(float));
        synth    = (float*)ptr;   ptr += ALIGN16(frame_sz * sizeof(float));
        window   = (float*)ptr;   ptr += ALIGN16(frame_sz * sizeof(float));
        fw       = (float*)ptr;   ptr += ALIGN16(frame_sz * sizeof(float));
        ifft_buf = (float*)ptr;   ptr += ALIGN16(frame_sz * sizeof(float));
        spec_in  = (Complex*)ptr; ptr += ALIGN16(n_freqs * sizeof(Complex));
        spec_out = (Complex*)ptr; ptr += ALIGN16(n_freqs * sizeof(Complex));
        make_sqrt_hann(window, frame_sz);
    }

    if (!aec || (!aec_only && (!fft || !nr))) {
        fprintf(stderr, "Error: module init failed\n");
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* Output WAV */
    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: cannot create output\n");
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* === Processing loop === */
    int processed = 0;
    while (processed + hop <= n_samples) {
        wav_read_float(mic_r, mic_buf, hop);
        wav_read_float(ref_r, ref_buf, hop);

        /* AEC + built-in AEC3 post-filter (mic HPF applied internally) */
        aec_process(aec, mic_buf, ref_buf, aec_out);

        if (aec_only) {
            wav_write_float(writer, aec_out, hop);
        } else {
            /* Stage 2: NR with freq-domain OLA
             *
             * Maintain a 2-hop (frame_sz) overlap-add analysis buffer.
             * Each hop:
             *   1. Shift analysis_buf left by hop, append new hop
             *   2. Apply sqrt-Hann window → fw, FFT → spec_in
             *   3. mmse_lsa_process: apply NR gain → spec_out
             *   4. IFFT → ifft_buf, apply synthesis window
             *   5. OLA: accumulate ifft_buf into synth
             *   6. Output first hop of synth, shift synth left
             *
             * sqrt-Hann on both analysis and synthesis windows gives perfect
             * reconstruction when NR gain = 1.0 (WOLA property).
             */

            /* 1. OLA analysis: shift + append new hop */
            memmove(analysis, analysis + hop, (size_t)hop * sizeof(float));
            memcpy(analysis + hop, aec_out, (size_t)hop * sizeof(float));

            /* 2. Window + FFT */
            for (int k = 0; k < frame_sz; k++) fw[k] = analysis[k] * window[k];
            fft_forward(fft, fw, spec_in);

            /* 3. NR: freq-domain gain estimation and application */
            mmse_lsa_process(nr, spec_in, spec_out);

            /* 4. IFFT + synthesis window */
            fft_inverse(fft, spec_out, ifft_buf);
            for (int k = 0; k < frame_sz; k++) ifft_buf[k] *= window[k];

            /* 5. OLA accumulation */
            for (int k = 0; k < frame_sz; k++) synth[k] += ifft_buf[k];

            /* 6. Output first hop, shift OLA accumulator */
            memcpy(out_buf, synth, (size_t)hop * sizeof(float));
            memmove(synth, synth + hop, (size_t)hop * sizeof(float));
            memset(synth + hop, 0, (size_t)hop * sizeof(float));

            wav_write_float(writer, out_buf, hop);
        }

        processed += hop;
        if (processed % sample_rate == 0) { printf("."); fflush(stdout); }
    }

    printf("\nProcessed: %d samples (%.2fs)\n", processed, (float)processed / sample_rate);
    printf("Output: %s\n", out_path);

    /* === Cleanup === */
    /* All modules are in the pool — no per-module free needed.
     * aec_destroy() is a no-op for static AEC (is_static=1). */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    free(pool);
    return 0;
}
