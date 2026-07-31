/**
 * Host-only raw-float runner for recording validation.
 *
 * Input mic: interleaved float32 [frames][4]
 * Input ref: mono float32 [frames]
 * Optional VAD: uint8 [complete_hops], nonzero = target speech
 * Output: mono float32 [complete_hops * hop]
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_doa_gsc.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct Options {
    const char* mic_path;
    const char* ref_path;
    const char* output_path;
    const char* vad_path;
    int sample_rate;
    int fft_size;
    float uca_radius_m;
    int fixed_doa;
    float fixed_doa_rad;
} Options;

/* ------------------------------------------------------------------ */

static void usage(const char* program) {
    fprintf(stderr,
            "Usage: %s --mic-raw MIC.f32 --ref-raw REF.f32 "
            "--output-raw OUT.f32 --sample-rate 16000|48000 "
            "[--fft-size 256|512|1024] "
            "[--vad-u8 VAD.u8] [--uca-radius-m 0.035] "
            "[--fixed-doa-deg DEG]\n",
            program);
}

static int parse_int(const char* text, int* value) {
    char* end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || parsed < -2147483647L ||
        parsed > 2147483647L) return 0;
    *value = (int)parsed;
    return 1;
}

static int parse_float(const char* text, float* value) {
    char* end = NULL;
    float parsed;
    errno = 0;
    parsed = strtof(text, &end);
    if (errno || !end || *end != '\0' || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static void json_float(char* output, size_t capacity, float value) {
    if (!output || capacity == 0) return;
    if (isfinite(value)) {
        (void)snprintf(output, capacity, "%.9g", value);
    } else {
        (void)snprintf(output, capacity, "null");
    }
}

static int parse_options(int argc, char** argv, Options* options) {
    memset(options, 0, sizeof(*options));
    options->uca_radius_m = 0.035f;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 0;
        if (strcmp(argv[i], "--mic-raw") == 0) {
            options->mic_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-raw") == 0) {
            options->ref_path = argv[++i];
        } else if (strcmp(argv[i], "--output-raw") == 0) {
            options->output_path = argv[++i];
        } else if (strcmp(argv[i], "--vad-u8") == 0) {
            options->vad_path = argv[++i];
        } else if (strcmp(argv[i], "--sample-rate") == 0) {
            if (!parse_int(argv[++i], &options->sample_rate)) return 0;
        } else if (strcmp(argv[i], "--fft-size") == 0) {
            if (!parse_int(argv[++i], &options->fft_size)) return 0;
        } else if (strcmp(argv[i], "--uca-radius-m") == 0) {
            if (!parse_float(argv[++i], &options->uca_radius_m)) return 0;
        } else if (strcmp(argv[i], "--fixed-doa-deg") == 0) {
            float degrees;
            if (!parse_float(argv[++i], &degrees)) return 0;
            options->fixed_doa = 1;
            options->fixed_doa_rad =
                degrees * (float)(M_PI / 180.0);
        } else {
            return 0;
        }
    }
    if (!options->mic_path || !options->ref_path ||
        !options->output_path ||
        !isfinite(options->uca_radius_m) ||
        options->uca_radius_m <= 0.0f) return 0;
    if (options->sample_rate == 16000)
        return options->fft_size == 0 ||
               options->fft_size == 256 ||
               options->fft_size == 512;
    if (options->sample_rate == 48000)
        return options->fft_size == 0 ||
               options->fft_size == 1024;
    return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    Options options;
    FourAecDoaGscConfig cfg;
    FourAecDoaGsc* p = NULL;
    FourAecDoaGscFrameInfo info;
    FILE* mic_file = NULL;
    FILE* ref_file = NULL;
    FILE* output_file = NULL;
    FILE* vad_file = NULL;
    float* microphones = NULL;
    float* reference = NULL;
    float* output = NULL;
    int hop;
    int processed = 0;
    int adaptive_frames = 0;
    int doa_frames = 0;
    int doa_analysis_frames = 0;
    int result = 1;
    char final_raw[32];
    char final_smooth[32];
    char final_used[32];

    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return 2;
    }
    cfg = four_aec_doa_gsc_default_config(options.sample_rate);
    cfg.core.fft_size = options.fft_size;
    cfg.uca_radius_m = options.uca_radius_m;
    if (options.fixed_doa) {
        cfg.gsc_fixed_mode = 1;
        cfg.gsc_fixed_doa_rad = options.fixed_doa_rad;
    }
    p = four_aec_doa_gsc_create(&cfg);
    if (!p) {
        fprintf(stderr, "failed to create 4ch DOA/GSC pipeline\n");
        goto cleanup;
    }
    hop = four_aec_doa_gsc_hop_size(p);
    microphones =
        (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS *
                       sizeof(float));
    reference = (float*)malloc((size_t)hop * sizeof(float));
    output = (float*)malloc((size_t)hop * sizeof(float));
    if (!microphones || !reference || !output) {
        fprintf(stderr, "failed to allocate hop buffers\n");
        goto cleanup;
    }
    mic_file = fopen(options.mic_path, "rb");
    ref_file = fopen(options.ref_path, "rb");
    output_file = fopen(options.output_path, "wb");
    if (options.vad_path) vad_file = fopen(options.vad_path, "rb");
    if (!mic_file || !ref_file || !output_file ||
        (options.vad_path && !vad_file)) {
        fprintf(stderr, "failed to open raw input/output\n");
        goto cleanup;
    }

    for (;;) {
        size_t mic_count = fread(
            microphones, sizeof(float),
            (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, mic_file);
        size_t ref_count =
            fread(reference, sizeof(float), (size_t)hop, ref_file);
        int status;
        if (mic_count == 0 && ref_count == 0) break;
        if (mic_count != (size_t)hop * FOUR_AEC_NR_RES_CHANNELS ||
            ref_count != (size_t)hop) {
            break; /* trailing incomplete hop is intentionally ignored */
        }
        if (vad_file) {
            unsigned char activity;
            if (fread(&activity, 1, 1, vad_file) != 1) {
                fprintf(stderr, "VAD file ended before audio\n");
                goto cleanup;
            }
            status = four_aec_doa_gsc_process_with_activity(
                p, microphones, reference,
                activity != 0, activity != 0, NULL, output, &info);
        } else {
            status = four_aec_doa_gsc_process(
                p, microphones, reference, output, &info);
        }
        if (status != FOUR_AEC_NR_RES_OK) {
            fprintf(stderr, "pipeline failed at frame %d: %d\n",
                    processed, status);
            goto cleanup;
        }
        if (fwrite(output, sizeof(float), (size_t)hop, output_file) !=
            (size_t)hop) {
            fprintf(stderr, "failed to write output frame\n");
            goto cleanup;
        }
        adaptive_frames += info.gsc_adaptive != 0;
        doa_frames += isfinite(info.doa_raw_rad);
        doa_analysis_frames += info.doa_analysis_frames;
        processed += 1;
    }
    if (ferror(mic_file) || ferror(ref_file) || ferror(output_file)) {
        fprintf(stderr, "raw file I/O failure\n");
        goto cleanup;
    }
    json_float(
        final_raw, sizeof(final_raw),
        processed > 0 ? info.doa_raw_rad : NAN);
    json_float(
        final_smooth, sizeof(final_smooth),
        processed > 0 ? info.doa_smooth_rad : NAN);
    json_float(
        final_used, sizeof(final_used),
        processed > 0 ? info.doa_used_rad : NAN);

    printf("{\"frames\":%d,\"hop\":%d,\"frame_size\":%d,"
           "\"fft_size\":%d,\"n_freqs\":%d,\"sample_rate\":%d,"
           "\"doa_sample_rate\":%d,\"doa_frame_size\":%d,"
           "\"doa_hop_size\":%d,\"doa_fft_size\":%d,"
           "\"gsc_sample_rate\":%d,\"gsc_frame_size\":%d,"
           "\"gsc_hop_size\":%d,\"gsc_fft_size\":%d,"
           "\"matched_filters\":%d,\"linear_aecs\":%d,\"nr\":%d,"
           "\"post_res\":%d,\"spatial_backend\":\"%s\","
           "\"doa_analysis_frames\":%d,\"doa_update_frames\":%d,"
           "\"gsc_adaptive_frames\":%d,"
           "\"final_delay_samples\":%d,\"final_delay_solid\":%d,"
           "\"final_doa_raw_rad\":%s,\"final_doa_smooth_rad\":%s,"
           "\"final_doa_used_rad\":%s}\n",
           processed, hop,
           four_aec_doa_gsc_frame_size(p),
           four_aec_doa_gsc_fft_size(p),
           four_aec_doa_gsc_n_freqs(p),
           options.sample_rate,
           four_aec_doa_gsc_doa_sample_rate(p),
           four_aec_doa_gsc_doa_frame_size(p),
           four_aec_doa_gsc_doa_hop_size(p),
           four_aec_doa_gsc_doa_fft_size(p),
           four_aec_doa_gsc_gsc_sample_rate(p),
           four_aec_doa_gsc_gsc_frame_size(p),
           four_aec_doa_gsc_gsc_hop_size(p),
           four_aec_doa_gsc_gsc_fft_size(p),
           four_aec_doa_gsc_matched_filter_count(p),
           four_aec_doa_gsc_linear_aec_count(p),
           four_aec_doa_gsc_nr_count(p),
           four_aec_doa_gsc_post_res_count(p),
           four_aec_doa_gsc_spatial_backend(),
           doa_analysis_frames, doa_frames, adaptive_frames,
           processed > 0 ? info.delay.delay_samples : 0,
           processed > 0 ? info.delay.solid : 0,
           final_raw, final_smooth, final_used);
    result = processed > 0 ? 0 : 1;

cleanup:
    if (vad_file) fclose(vad_file);
    if (output_file) fclose(output_file);
    if (ref_file) fclose(ref_file);
    if (mic_file) fclose(mic_file);
    free(output);
    free(reference);
    free(microphones);
    four_aec_doa_gsc_destroy(p);
    return result;
}
