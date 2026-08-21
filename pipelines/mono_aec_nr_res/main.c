/**
 * aec_nr_pipeline.c — AEC(linear) → echo-aware NR → RES  (Version A: malloc)
 *
 * Thin CLI shell over mono_aec_nr_res/audio_pipeline.h. Freq-domain
 * A_min_pl pipeline, a C port of pipelines/aec_nr_pipeline.py (the 2026-06-23
 * re-tune). Per hop:
 *   Stage 1: AEC in LINEAR mode (enable_res=0, return_res_context=1) — the AEC3
 *            post block still runs and exposes, via AecResContext, the windowed
 *            linear error spectrum E(f), the AEC3 suppression gain G_res(f), the
 *            residual-echo PSD R²(f) and the comfort-noise PSD N²(f); the AEC
 *            time output stays the linear residual (no suppression applied).
 *   Stage 2: echo-aware NR on E(f). MMSE-LSA gain G_nr(f) with R² folded into the
 *            noise floor (ξ = S²/(N²+R²), the unified Speex/Habets gain).
 *   Stage 3: external RES — g_total = min(G_nr, G_res), a far-activity + near-VAD
 *            gated near-end floor lift, then S(f) = E(f)·g_total (+ comfort noise
 *            on the cut bins) and one sqrt-Hann OLA.
 *
 * All of the above now lives in audio_pipeline.c (audio_pipeline_process()) —
 * this file is arg parsing + WAV I/O + the `--debug` / `DUMP_CTX` diagnostics
 * around a single audio_pipeline_create()/destroy() pair (the heap
 * convenience path; see mono_aec_nr_res/static_main.c for the explicit
 * caller-pool flavor of the same API).
 *
 * This mirrors run_aec_linear → run_nr_spectrum(inject_echo_psd) → run_res
 * (combine='min', ne_floor far/near gate). --legacy-amin restores the prior
 * min-only A_min_pl (noise-only NR, scalar ne_floor=0.4).
 *
 * Build:  make libs && make aec_nr_pipeline
 * Usage:
 *   ./aec_nr_pipeline <mic.wav> <ref.wav> <out.wav> [aec-preset]
 *                     [--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin]
 *                     [--debug]
 *
 * --debug: once per second of processed audio, print one compact status line
 *   to stderr combining aec_debug_status() (lib/aec) + mmse_lsa_debug_status()
 *   (lib/nr) — read-only snapshots, no DSP-state or output changes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "audio_pipeline.h"
#include "wav_io.h"

/* ------------------------------------------------------------------ */

static AecPreset parse_preset(const char* s) {
    if (strcmp(s, "mild") == 0)     return AEC_PRESET_MILD;
    if (strcmp(s, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    return AEC_PRESET_BALANCED;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_MILD:     return "mild";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        default:                     return "balanced";
    }
}

static MmseLsaNrMode parse_nr_mode(const char* s) {
    if (strcmp(s, "mild") == 0)       return MMSE_LSA_NR_MILD;
    if (strcmp(s, "aggressive") == 0) return MMSE_LSA_NR_AGGRESSIVE;
    return MMSE_LSA_NR_BALANCED;
}

static const char* nr_mode_name(MmseLsaNrMode m) {
    switch (m) {
        case MMSE_LSA_NR_MILD:       return "mild";
        case MMSE_LSA_NR_AGGRESSIVE: return "aggressive";
        default:                     return "balanced";
    }
}

/* --debug: one compact status line/sec to stderr, combining both libraries'
 * read-only diagnostic queries (aec_debug_status() / mmse_lsa_debug_status())
 * via the handles audio_pipeline_get_aec()/audio_pipeline_get_nr() expose.
 * Neither call mutates DSP state or fast_math-approximates anything (both
 * use standard logf/log10f), so this is safe to call every second regardless
 * of preset/backend. aec_only (or nr==NULL) omits the NR half — no denoiser
 * exists in that mode.
 *
 * CAVEAT: this pipeline always runs AEC in linear mode (enable_res=0,
 * return_res_context=1 — the external NR/RES seam above), and aec.c only
 * caches last_erle_windowed when cfg.enable_res is true. So the "erle="
 * field below always reads 0.0dB here — expected given this pipeline's
 * config, not a broken query (see pipelines/README.md "Debugging &
 * Performance Flags"). */
static void print_debug_status(const AudioPipeline* pipe, int aec_only, float seconds) {
    AecDebugStatus a;
    aec_debug_status(audio_pipeline_get_aec(pipe), &a);
    const MmseLsaDenoiser* nr = audio_pipeline_get_nr(pipe);
    if (aec_only || !nr) {
        fprintf(stderr,
            "[dbg %5.1fs] aec: delay=%d conf=%.1f upd=%d erle=%.1fdB lin=%d conv=%d "
            "near=%.2e out=%.2e | nr: n/a (--aec-only)\n",
            seconds, a.delay_samples, a.delay_confidence, a.delay_updates,
            a.erle_windowed_db, a.usable_linear, a.filter_converged,
            a.near_power, a.out_power);
        return;
    }
    MmseLsaDebugStatus n;
    mmse_lsa_debug_status(nr, &n);
    fprintf(stderr,
        "[dbg %5.1fs] aec: delay=%d conf=%.1f upd=%d erle=%.1fdB lin=%d conv=%d "
        "near=%.2e out=%.2e | nr: init=%d gain=%.1f/%.1fdB spp=%.2f noise=%.1fdB\n",
        seconds, a.delay_samples, a.delay_confidence, a.delay_updates,
        a.erle_windowed_db, a.usable_linear, a.filter_converged,
        a.near_power, a.out_power,
        n.initialized, n.mean_gain_db, n.min_gain_db, n.mean_spp, n.noise_floor_db);
}

/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ *
 * --timing: per-stage cost accumulation.
 *
 * Summed over the run and reported as a mean per hop, because a per-hop
 * print at 100 hops/s is noise and a single hop is not a measurement. The
 * UNATTRIBUTED line is not decoration: the stages deliberately do not sum to
 * the call (audio_pipeline.h documents what sits in the gap), so a table
 * that printed only the parts would imply a whole it never measured.
 * ------------------------------------------------------------------ */
typedef struct {
    uint64_t delay, frontend, linear, res, nr, post, synth, wall;
    uint64_t hops;
} StageAccum;

static void stage_accum_add(StageAccum* a, const AudioPipelineLastTiming* t,
                            uint32_t wall_us) {
    a->delay    += t->aec.delay_us;
    a->frontend += t->aec.frontend_us;
    a->linear   += t->aec.linear_us;
    a->res      += t->aec.res_us;
    a->nr       += t->nr_us;
    a->post     += t->post_us;
    a->synth    += t->synth_us;
    a->wall     += wall_us;
    a->hops     += 1u;
}

static void stage_accum_row(const char* name, uint64_t total,
                            const StageAccum* a) {
    double per_hop = a->hops ? (double)total / (double)a->hops : 0.0;
    double share   = a->wall ? 100.0 * (double)total / (double)a->wall : 0.0;
    printf("  %-14s %8.2f  %6.1f%%\n", name, per_hop, share);
}

static void print_stage_timing(const StageAccum* a) {
    uint64_t measured = a->delay + a->frontend + a->linear + a->res
                      + a->nr + a->post + a->synth;
    if (!a->hops) { printf("\nStage timing: no hops processed.\n"); return; }
    printf("\nStage timing over %llu hops (mean us/hop, share of wall):\n",
           (unsigned long long)a->hops);
    if (measured == 0u) {
        printf("  every stage reads 0 -- this binary was built WITHOUT the\n"
               "  timing flags. Rebuild with `make PROFILE=1` (or pass\n"
               "  -DAUDIO_PIPELINE_STAGE_TIMING=1 -DAEC_STAGE_TIMING=1) to\n"
               "  measure; the display flag alone cannot turn it on.\n");
        return;
    }
    stage_accum_row("AEC delay", a->delay, a);
    stage_accum_row("AEC frontend", a->frontend, a);
    stage_accum_row("AEC linear", a->linear, a);
    stage_accum_row("AEC post/RES", a->res, a);
    stage_accum_row("NR gain", a->nr, a);
    stage_accum_row("post arith", a->post, a);
    stage_accum_row("synthesis", a->synth, a);
    printf("  ------------------------------------\n");
    stage_accum_row("measured", measured, a);
    stage_accum_row("unattributed",
                    a->wall > measured ? a->wall - measured : 0u, a);
    stage_accum_row("call wall", a->wall, a);
}

static uint32_t timing_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}


int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [aec-preset] "
               "[--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin] "
               "[--fft-size 256|512|1024] [--debug] [--timing]\n",
               argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecPreset     preset   = AEC_PRESET_BALANCED;
    MmseLsaNrMode nr_mode  = MMSE_LSA_NR_BALANCED;
    int           aec_only = 0;
    int           legacy   = 0;   /* --legacy-amin → prior min-only A_min_pl */
    int           no_cng   = 0;   /* --no-cng → disable comfort noise (parity) */
    int           debug_status = 0; /* --debug → periodic aec+nr status line   */
    int           show_timing  = 0; /* --timing → per-stage cost summary at exit */
    StageAccum    acc; memset(&acc, 0, sizeof(acc));
    int           fft_size = 0;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0)    aec_only = 1;
        else if (strcmp(argv[i], "--legacy-amin") == 0) legacy = 1;
        else if (strcmp(argv[i], "--no-cng") == 0)      no_cng = 1;
        else if (strcmp(argv[i], "--debug") == 0)       debug_status = 1;
        else if (strcmp(argv[i], "--timing") == 0)      show_timing = 1;
        else if (strcmp(argv[i], "--fft-size") == 0 && i+1 < argc)
            fft_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nr-preset") == 0 && i+1 < argc)
            nr_mode = parse_nr_mode(argv[++i]);
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
    /* CLI rate whitelist: audio_pipeline_create() would reject an unsupported
     * rate anyway (aec_is_valid_sample_rate() under the hood) — fail earlier
     * and with a clearer message here, before any buffers are sized off it. */
    if (!aec_is_valid_sample_rate(mic_r->info.sample_rate)) {
        fprintf(stderr, "Error: unsupported sample rate %d Hz (supported: 8000, 16000, 48000)\n",
                mic_r->info.sample_rate);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int sr        = mic_r->info.sample_rate;
    int n_samples = (mic_r->info.num_samples < ref_r->info.num_samples)
                  ? mic_r->info.num_samples : ref_r->info.num_samples;

    AudioPipelineConfig cfg = audio_pipeline_default_config(sr);
    cfg.fft_size    = fft_size;
    cfg.aec_preset  = preset;
    cfg.nr_mode     = nr_mode;
    cfg.aec_only    = aec_only;
    cfg.enable_cng  = !no_cng;
    cfg.legacy_amin = legacy;

    printf("AEC(linear) -> echo-aware NR -> RES  (freq A_min_pl%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n\n",
           preset_name(preset), nr_mode_name(nr_mode), !no_cng ? "on" : "off");

    /* Create the pipeline (heap convenience: get_mem_requirements +
     * posix_memalign + init, all inside audio_pipeline_create()). */
    AudioPipeline* pipe = audio_pipeline_create(&cfg);
    if (!pipe) {
        fprintf(stderr, "Error: pipeline create failed\n");
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int hop = audio_pipeline_hop_size(pipe);
    int n_freqs = audio_pipeline_n_freqs(pipe);
    float* mic_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        audio_pipeline_destroy(pipe);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) {
        fprintf(stderr, "Error: cannot create output\n");
        audio_pipeline_destroy(pipe);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* Optional per-frame context dump (DUMP_CTX=<path>) for port parity tests:
     * header [n_freqs, hop]; then per frame error_spec(2·nf) res_gain(nf) r2(nf)
     * comfort_noise(nf) far_power(1) g_nr(nf) out_hop(hop), all float32. Read
     * via the audio_pipeline_get_aec()/audio_pipeline_get_nr() handles — a
     * dump record is only ever written for a hop whose freq seam was
     * actually available, mirroring audio_pipeline_process()'s own internal
     * fallback check. */
    FILE* dctx = NULL;
    const char* dpath = getenv("DUMP_CTX");
    if (dpath && !aec_only) {
        dctx = fopen(dpath, "wb");
        if (dctx) { int hdr[2] = { n_freqs, hop }; fwrite(hdr, sizeof(int), 2, dctx); }
    }

    Aec* aec_handle = audio_pipeline_get_aec(pipe);
    MmseLsaDenoiser* nr_handle = audio_pipeline_get_nr(pipe);

    /* === Processing loop === */
    int processed = 0;
    while (processed + hop <= n_samples) {
        wav_read_float(mic_r, mic_buf, hop);
        wav_read_float(ref_r, ref_buf, hop);

        if (show_timing) {
            AudioPipelineLastTiming t;
            uint32_t t_in = timing_now_us();
            audio_pipeline_process(pipe, mic_buf, ref_buf, out_buf);
            {
                uint32_t wall = timing_now_us() - t_in;
                audio_pipeline_get_last_timing(pipe, &t);
                stage_accum_add(&acc, &t, wall);
            }
        } else {
            audio_pipeline_process(pipe, mic_buf, ref_buf, out_buf);
        }
        wav_write_float(writer, out_buf, hop);

        if (dctx) {
            AecResContext ctx;
            aec_get_res_context(aec_handle, &ctx);
            if (ctx.error_spec && ctx.res_gain) {
                const float* g_nr = mmse_lsa_get_gain(nr_handle, NULL);
                fwrite(ctx.error_spec, sizeof(Complex), n_freqs, dctx);
                fwrite(ctx.res_gain, sizeof(float), n_freqs, dctx);
                fwrite(ctx.r2, sizeof(float), n_freqs, dctx);
                fwrite(ctx.comfort_noise, sizeof(float), n_freqs, dctx);
                float fp = ctx.far_power; fwrite(&fp, sizeof(float), 1, dctx);
                fwrite(g_nr, sizeof(float), n_freqs, dctx);
                fwrite(out_buf, sizeof(float), hop, dctx);
            }
        }

        processed += hop;
        if (processed % sr == 0) {
            printf("."); fflush(stdout);
            if (debug_status) print_debug_status(pipe, aec_only, (float)processed / sr);
        }
    }
    if (dctx) fclose(dctx);

    printf("\nProcessed: %d samples (%.2fs)\n", processed, (float)processed / sr);
    if (show_timing) print_stage_timing(&acc);
    printf("Output: %s\n", out_path);

    /* Cleanup */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    audio_pipeline_destroy(pipe);
    free(mic_buf); free(ref_buf); free(out_buf);
    return 0;
}
