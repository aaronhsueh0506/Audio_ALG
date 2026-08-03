/**
 * audio_pipeline_4ch_static.c — complete four-channel spatial pipeline
 *                                (core + SRP-PHAT + GSC), static memory.
 *
 * Thin host reference over audio_pipeline_4ch.h. Follows the same
 * query -> allocate -> init_ex -> process -> destroy -> release sequence as
 * 4aec_nr_res_static.c (the core-only demo) and pipelines/
 * aec_nr_pipeline_static.c (the mono demo), but drives the COMPLETE wrapper
 * -- the real SRP-PHAT DOA + GSC beamformer run inside this process, not an
 * externally-supplied fixed-weight stand-in. This is the full end-to-end
 * static-memory smoke reference; audio_pipeline_4ch_raw.c (the
 * recording-validation CLI) intentionally stays on the heap `create()` path
 * since it is a one-shot host tool, not a board-deployment demo.
 *
 * Build:
 *   make -C pipelines/4ch_pipelines audio_pipeline_4ch_static
 *
 * Usage:
 *   ./audio_pipeline_4ch_static [--sample-rate 16000|48000]
 *       [--fft-size 256|512|1024] [--uca-radius-m 0.035]
 *       [--fixed-doa-deg DEG]
 *   ./audio_pipeline_4ch_static --print-mem-size [same options]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_pipeline_4ch.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */

static const char* backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case FOUR_AEC_NR_RES_BACKEND_KISS: return "kiss";
        case FOUR_AEC_NR_RES_BACKEND_NE10: return "ne10";
        default:                           return "unknown";
    }
}

static int print_mem_budget(const AudioPipeline4ChConfig* cfg) {
    AudioPipeline4ChMemReq req;

    if (audio_pipeline_4ch_get_mem_requirements(cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid 4ch spatial pipeline config\n");
        return -1;
    }

    printf("Memory Budget (complete 4ch pipeline: core + SRP-PHAT + GSC)\n");
    printf("=============================================================\n");
    /* cfg->core.fft_size may be the unresolved 0 sentinel (caller didn't
     * override --fft-size); print the grid that will actually be used,
     * matching audio_pipeline_4ch_raw.c's print_mem_budget(). */
    printf("  sample_rate=%d fft=%d geometry=%d num_angles=%d\n",
           cfg->core.sample_rate,
           cfg->core.fft_size ? cfg->core.fft_size
                              : (cfg->core.sample_rate == 16000 ? 256 : 1024),
           (int)cfg->geometry, cfg->num_angles);
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           (unsigned long long)req.bytes, (float)req.bytes / 1024.0f);
    printf("  Descriptor:     descriptor_version=%u alignment=%u "
           "layout_version=%u backend_id=%u (%s) "
           "build_flags_hash=0x%08x\n\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, backend_id_name(req.backend_id),
           req.build_flags_hash);
    return 0;
}

static void fill_hop(float* microphones, float* reference, int hop,
                     int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int absolute = frame_index * hop + i;
        float phase =
            2.0f * 3.14159265358979323846f * 440.0f *
            (float)absolute / (float)sample_rate;
        reference[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            /* Slightly different per-channel scale + a tiny per-channel
             * delay stand-in (phase offset) so SRP-PHAT/GSC see a
             * non-degenerate four-channel signal, not four identical
             * copies. */
            float ch_phase = phase - 0.05f * (float)ch;
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                (0.35f + 0.05f * (float)ch) * sinf(ch_phase);
        }
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    int sample_rate = 16000;
    int fft_size = 0;
    float uca_radius_m = 0.035f;
    int fixed_doa = 0;
    float fixed_doa_rad = 0.0f;
    int print_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--print-mem-size") == 0) {
            print_only = 1;
        } else if (strcmp(argv[i], "--sample-rate") == 0 &&
                   i + 1 < argc) {
            sample_rate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fft-size") == 0 &&
                   i + 1 < argc) {
            fft_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--uca-radius-m") == 0 &&
                   i + 1 < argc) {
            uca_radius_m = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--fixed-doa-deg") == 0 &&
                   i + 1 < argc) {
            fixed_doa = 1;
            fixed_doa_rad = (float)atof(argv[++i]) * (float)(M_PI / 180.0);
        } else {
            fprintf(stderr,
                    "Usage: %s [--sample-rate 16000|48000] "
                    "[--fft-size 256|512|1024] [--uca-radius-m 0.035] "
                    "[--fixed-doa-deg DEG] [--print-mem-size]\n",
                    argv[0]);
            return 1;
        }
    }

    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    if (fft_size != 0) cfg.core.fft_size = fft_size;
    cfg.uca_radius_m = uca_radius_m;
    if (fixed_doa) {
        cfg.gsc_fixed_mode = 1;
        cfg.gsc_fixed_doa_rad = fixed_doa_rad;
    }

    printf("Complete 4ch pipeline: core + SRP-PHAT + GSC (static memory)\n");
    printf("  Geometry: UCA radius=%.4f m   num_angles=%d\n",
           cfg.uca_radius_m, cfg.num_angles);

    if (print_mem_budget(&cfg) != 0) return 1;

    /* Query -> allocate -> init_ex: same caller-pool sequence as
     * 4aec_nr_res_static.c / pipelines/aec_nr_pipeline_static.c. malloc is a
     * host stand-in for the board memory manager; the pipeline owns none of
     * this memory. */
    AudioPipeline4ChMemReq req;
    if (audio_pipeline_4ch_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        return 1;
    }

    void* pool = malloc((size_t)req.bytes);
    if (!pool) {
        fprintf(stderr, "Error: malloc failed (%llu bytes)\n",
                (unsigned long long)req.bytes);
        return 1;
    }
    if (((uintptr_t)pool) % req.alignment != 0) {
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n",
                req.alignment, pool);
        free(pool);
        return 1;
    }

    AudioPipeline4Ch* p =
        audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    if (!p) {
        fprintf(stderr, "Error: audio_pipeline_4ch_init_ex failed\n");
        free(pool);
        return 1;
    }

    int hop = audio_pipeline_4ch_hop_size(p);
    printf("hop=%d fft=%d n_freqs=%d at %d Hz\n", hop,
           audio_pipeline_4ch_fft_size(p), audio_pipeline_4ch_n_freqs(p),
           sample_rate);
    if (print_only) {
        audio_pipeline_4ch_destroy(p);
        free(pool);
        return 0;
    }

    float* mic_buf =
        (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        audio_pipeline_4ch_destroy(p);
        free(pool);
        free(mic_buf);
        free(ref_buf);
        free(out_buf);
        return 1;
    }

    /* === Processing loop ===
     * Uses process_with_activity() with an explicit, product-VAD-style
     * schedule instead of process()'s auto-VAD fallback: a short warmup
     * with vad_raw=vad_out=1 (SRP locks a DOA while GSC stays frozen, as
     * it would during genuine target speech), then vad_raw=1/vad_out=0 for
     * the remainder (SRP keeps observing so the DOA stays valid, while GSC
     * -- which by design only adapts while vad_out==0 AND a valid DOA
     * exists, see gsc.c's DOA-logic comment -- gets a real chance to run).
     * vad_raw/vad_out are DECOUPLED parameters here (unlike process()'s
     * auto-VAD fallback, which always passes the same value for both), so
     * this reaches a state process()'s own slow noise-floor-tracking EMA
     * would take hundreds of frames to reach on a sustained tone. Also
     * demonstrates the entry point this header recommends for production
     * integrations that own a VAD. 40 post-warmup frames is long enough to
     * see doa_update_interval's periodic SRP refresh (default 2) lock in
     * and GSC's RLS update (gsc_adapt_interval, default 1) actually run --
     * a short 6-frame run only ever sampled the LAST frame's diagnostics,
     * which could land on an off-interval (deliberately NaN doa_raw, see
     * srp.c) frame and never actually prove SRP/GSC reached a valid
     * adaptive state, just that the pool lifecycle didn't crash. */
    const int warmup_frames = 8;
    const int total_frames = warmup_frames + 40;
    int frame;
    int saw_finite_doa_raw = 0;
    int saw_gsc_adaptive = 0;
    for (frame = 0; frame < total_frames; ++frame) {
        AudioPipeline4ChFrameInfo info;
        int vad_active = frame < warmup_frames ? 1 : 0;
        fill_hop(mic_buf, ref_buf, hop, sample_rate, frame);
        if (audio_pipeline_4ch_process_with_activity(
                p, mic_buf, ref_buf, /*vad_raw=*/1, /*vad_out=*/vad_active,
                NULL, out_buf, &info) != 0) {
            fprintf(stderr,
                    "Error: audio_pipeline_4ch_process_with_activity failed\n");
            audio_pipeline_4ch_destroy(p);
            free(pool);
            free(mic_buf);
            free(ref_buf);
            free(out_buf);
            return 1;
        }
        int k;
        for (k = 0; k < hop; ++k) {
            if (!isfinite(out_buf[k])) {
                fprintf(stderr, "Error: non-finite output\n");
                audio_pipeline_4ch_destroy(p);
                free(pool);
                free(mic_buf);
                free(ref_buf);
                free(out_buf);
                return 1;
            }
        }
        if (isfinite(info.doa_raw_rad)) saw_finite_doa_raw = 1;
        if (info.gsc_adaptive) saw_gsc_adaptive = 1;
        if (frame == total_frames - 1) {
            printf("final frame: doa_raw=%.4f doa_smooth=%.4f doa_used=%.4f "
                   "vad_out=%d gsc_adaptive=%d\n",
                   info.doa_raw_rad, info.doa_smooth_rad, info.doa_used_rad,
                   info.vad_out, info.gsc_adaptive);
        }
    }

    printf("Processed: %d frames (%d samples)\n", frame, frame * hop);
    printf("  saw a finite doa_raw at least once: %s\n",
           saw_finite_doa_raw ? "yes" : "no");
    printf("  saw gsc_adaptive=1 at least once:   %s\n",
           saw_gsc_adaptive ? "yes" : "no");
    if (!saw_finite_doa_raw || !saw_gsc_adaptive) {
        fprintf(stderr,
                "Error: SRP/GSC never reached the expected adaptive state "
                "over %d frames (finite doa_raw=%d, gsc_adaptive=%d) -- "
                "the pipeline ran without crashing, but that alone does "
                "not prove SRP-PHAT/GSC actually did anything\n",
                frame, saw_finite_doa_raw, saw_gsc_adaptive);
        audio_pipeline_4ch_destroy(p);
        free(pool);
        free(mic_buf);
        free(ref_buf);
        free(out_buf);
        return 1;
    }
    printf("audio_pipeline_4ch_static: smoke PASS\n");

    /* === Cleanup === */
    audio_pipeline_4ch_destroy(p);
    free(pool);
    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    return 0;
}
