/**
 * aec_nr_pipeline.c — AEC + NR Pipeline (Version A: malloc)
 *
 * Two-stage speech enhancement:
 *   Stage 0: HPF (80 Hz Butterworth) — remove DC/hum
 *   Stage 1: AEC with built-in AEC3 post-filter (echo cancel + suppress)
 *   Stage 2: NR (MMSE-LSA, freq-domain OLA managed by caller)
 *
 * Build:
 *   make libs
 *   make aec_nr_pipeline
 *
 * Usage:
 *   ./aec_nr_pipeline <mic.wav> <ref.wav> <output.wav> [preset]
 *   preset: gentle, balanced (default), aggressive
 *   options: --aec-only   --nr-gain <dB>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "aec.h"               /* AEC + fft_wrapper.h (includes ALIGN16, Complex) */
#include "fft_wrapper.h"       /* FftHandle, Complex, ALIGN16 */
#include "mmse_lsa_denoiser.h" /* freq-domain NR */
#include "wav_io.h"

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

/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [preset] [--aec-only] [--nr-gain dB]\n",
               argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecPreset preset  = AEC_PRESET_BALANCED;
    int       aec_only    = 0;
    float     nr_g_min_db = -15.0f;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0)           aec_only = 1;
        else if (strcmp(argv[i], "--nr-gain") == 0 && i+1 < argc)
            nr_g_min_db = (float)atof(argv[++i]);
        else if (argv[i][0] != '-')
            preset = parse_preset(argv[i]);
    }

    /* Open inputs */
    WavReader* mic_r = wav_open_read(mic_path);
    WavReader* ref_r = wav_open_read(ref_path);
    if (!mic_r || !ref_r) { fprintf(stderr, "Error: cannot open inputs\n"); return 1; }
    if (mic_r->info.sample_rate != ref_r->info.sample_rate) {
        fprintf(stderr, "Error: sample-rate mismatch\n");
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int sr         = mic_r->info.sample_rate;
    int n_samples  = (mic_r->info.num_samples < ref_r->info.num_samples)
                   ? mic_r->info.num_samples : ref_r->info.num_samples;

    /* AEC config */
    AecConfig aec_cfg;
    aec_config_from_preset(&aec_cfg, preset, sr);
    aec_cfg.enable_res = 1;   /* built-in AEC3 post-filter (suppression + CNG) */

    /* NR config */
    MmseLsaConfig nr_cfg = mmse_lsa_default_config(sr);
    nr_cfg.g_min_db = nr_g_min_db;

    /* Frame dimensions */
    int hop      = (int)(0.01f * sr);         /* 160 @ 16 kHz */
    int frame_sz = 2 * hop;                   /* 320 */
    int fft_sz   = 512;                        /* next_pow2(frame_sz) */
    while (fft_sz < frame_sz) fft_sz *= 2;
    int n_freqs  = fft_sz / 2 + 1;           /* 257 */

    printf("AEC+NR Pipeline\n");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    printf("  Preset: %s   NR: %.0f dB\n\n", preset_name(preset), (double)nr_g_min_db);

    /* Create AEC (mic HPF @ 80 Hz applied internally via enable_highpass=1) */
    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec || aec_create(aec, &aec_cfg) != 0) {
        fprintf(stderr, "Error: AEC create failed\n"); return 1;
    }

    /* Create FFT + NR (for OLA bridge) */
    FftHandle*       fft = NULL;
    MmseLsaDenoiser* nr  = NULL;
    float* analysis = NULL, *synth = NULL, *window = NULL, *fw = NULL, *ifft_buf = NULL;
    Complex* spec_in = NULL, *spec_out = NULL;

    if (!aec_only) {
        fft = fft_create(fft_sz);
        nr  = mmse_lsa_create(&nr_cfg);
        if (!fft || !nr) { fprintf(stderr, "Error: NR/FFT create failed\n"); return 1; }

        analysis = (float*)calloc((size_t)frame_sz, sizeof(float));
        synth    = (float*)calloc((size_t)frame_sz, sizeof(float));
        window   = (float*)malloc((size_t)frame_sz * sizeof(float));
        fw       = (float*)malloc((size_t)frame_sz * sizeof(float));
        ifft_buf = (float*)malloc((size_t)frame_sz * sizeof(float));
        spec_in  = (Complex*)malloc((size_t)n_freqs * sizeof(Complex));
        spec_out = (Complex*)malloc((size_t)n_freqs * sizeof(Complex));
        if (!analysis || !synth || !window || !fw || !ifft_buf || !spec_in || !spec_out) {
            fprintf(stderr, "Error: OLA buffer alloc failed\n"); return 1;
        }
        make_sqrt_hann(window, frame_sz);
    }

    /* Working buffers */
    float* mic_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* aec_out = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));

    /* Output WAV */
    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) { fprintf(stderr, "Error: cannot create output\n"); return 1; }

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
             * Each hop: shift analysis window, window+FFT, NR, IFFT, OLA.
             * OLA with 50% overlap (sqrt-Hann) gives perfect reconstruction
             * when NR gain = 1.
             */

            /* Shift analysis buffer and append new hop */
            memmove(analysis, analysis + hop, (size_t)hop * sizeof(float));
            memcpy(analysis + hop, aec_out, (size_t)hop * sizeof(float));

            /* Window + FFT */
            for (int k = 0; k < frame_sz; k++) fw[k] = analysis[k] * window[k];
            fft_forward(fft, fw, spec_in);

            /* NR: apply gain in frequency domain */
            mmse_lsa_process(nr, spec_in, spec_out);

            /* IFFT + synthesis window */
            fft_inverse(fft, spec_out, ifft_buf);
            for (int k = 0; k < frame_sz; k++) ifft_buf[k] *= window[k];

            /* OLA: accumulate, output first hop, shift */
            for (int k = 0; k < frame_sz; k++) synth[k] += ifft_buf[k];
            memcpy(out_buf, synth, (size_t)hop * sizeof(float));
            memmove(synth, synth + hop, (size_t)hop * sizeof(float));
            memset(synth + hop, 0, (size_t)hop * sizeof(float));

            wav_write_float(writer, out_buf, hop);
        }

        processed += hop;
        if (processed % sr == 0) { printf("."); fflush(stdout); }
    }

    printf("\nProcessed: %d samples (%.2fs)\n", processed, (float)processed / sr);
    printf("Output: %s\n", out_path);

    /* Cleanup */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    free(mic_buf); free(ref_buf); free(aec_out); free(out_buf);
    if (!aec_only) {
        free(analysis); free(synth); free(window); free(fw); free(ifft_buf);
        free(spec_in);  free(spec_out);
        mmse_lsa_destroy(nr);
        fft_destroy(fft);
    }
    aec_destroy(aec);
    free(aec);
    return 0;
}
