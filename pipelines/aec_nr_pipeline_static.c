/**
 * aec_nr_pipeline_static.c — AEC(linear) -> echo-aware NR -> RES  (Version B: static memory)
 *
 * Thin CLI shell over pipelines/audio_pipeline.h (review F20). All
 * pool-sizing/carving/per-hop-processing logic that used to live here as
 * file-local `static` functions (`pipeline_pool_size`, `pipeline_build`,
 * `pipeline_destroy`, and the processing while-loop body inlined in
 * `main()`) has moved into audio_pipeline.c, which is now linkable on its
 * own (see pipelines/README.md "Board Integration"). This file keeps
 * exactly what a CLI is actually responsible for: argv parsing, WAV I/O,
 * the explicit caller-owned-pool dance (`malloc` here standing in for a
 * platform allocator — see audio_pipeline.h's audio_pipeline_init doc), and
 * the `--debug` / `DUMP_CTX` / `--print-mem-size` diagnostics.
 *
 * `aec_nr_pipeline_static out.wav` must be BYTE-IDENTICAL to
 * `aec_nr_pipeline out.wav` for the same inputs/options/backend (both are
 * now the SAME audio_pipeline_process() call underneath — one reached via
 * an explicit caller pool, the other via audio_pipeline_create()'s heap
 * convenience wrapper).
 *
 * Build:  make -C pipelines            (builds both binaries)
 * Usage:
 *   ./aec_nr_pipeline_static <mic.wav> <ref.wav> <out.wav> [aec-preset]
 *                     [--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin]
 *                     [--debug]
 *   ./aec_nr_pipeline_static --print-mem-size [preset] [--nr-preset ...] [--aec-only]
 *                     [--sample-rate <hz>]
 *
 * --debug: identical semantics to the malloc pipeline (see
 *   aec_nr_pipeline.c's header) — prints one combined AEC+NR status
 *   line/sec to stderr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
 * via the handles audio_pipeline_get_aec()/audio_pipeline_get_nr() expose —
 * byte-identical helper to the malloc pipeline's copy.
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

/* Human-readable name for a descriptor backend_id -- CLI-side convenience
 * only (this is trusted local code turning OUR OWN integer back into a
 * string for a diagnostic table; not the caller-data %s/strcmp hazard the
 * library itself now avoids -- see audio_pipeline.h's AudioPipelineMemReq.
 * backend_id doc). */
static const char* backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case AUDIO_PIPELINE_BACKEND_KISS: return "kiss";
        case AUDIO_PIPELINE_BACKEND_NE10: return "ne10";
        default:                          return "unknown";
    }
}

/* --print-mem-size diagnostic table. Backed by audio_pipeline_get_mem_breakdown
 * (per-module AEC/FFT/NR/pipeline-buffer split) + audio_pipeline_get_mem_requirements
 * (the descriptor V2 struct, review B06: total bytes incl. the AudioPipeline
 * control block itself, descriptor_version, layout_version, backend_id,
 * build_flags_hash, alignment). Same table shape as the pre-F20 static CLI,
 * plus the descriptor fields the old bare size_t couldn't carry. */
static void print_mem_budget(const AudioPipelineConfig* cfg) {
    AudioPipelineMemBreakdown b;
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_breakdown(cfg, &b) != 0 ||
        audio_pipeline_get_mem_requirements(cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid config for memory budget query\n");
        return;
    }

    printf("Memory Budget (Static Pipeline)\n");
    printf("================================\n");
    printf("  sample_rate=%d hop=%d frame_sz=%d fft_sz=%d n_freqs=%d\n",
           cfg->sample_rate, b.hop, b.frame_sz, b.fft_sz, b.n_freqs);
    printf("  AEC:            %7zu bytes (%6.1f KB)\n", b.aec_bytes, (float)b.aec_bytes / 1024.0f);
    if (!cfg->aec_only) {
        printf("  FFT (OLA):      %7zu bytes (%6.1f KB)\n", b.fft_bytes, (float)b.fft_bytes / 1024.0f);
        printf("  NR (MMSE-LSA):  %7zu bytes (%6.1f KB)\n", b.nr_bytes, (float)b.nr_bytes / 1024.0f);
    }
    printf("  Pipeline bufs:  %7zu bytes (%6.1f KB)\n", b.pipeline_bytes, (float)b.pipeline_bytes / 1024.0f);
    printf("  --------------------------------\n");
    printf("  Total:          %7llu bytes (%6.1f KB)  [incl. %llu B AudioPipeline control block]\n",
           (unsigned long long)req.bytes, (float)req.bytes / 1024.0f,
           (unsigned long long)req.bytes - b.aec_bytes - b.fft_bytes - b.nr_bytes - b.pipeline_bytes);
    printf("  Descriptor:     descriptor_version=%u alignment=%u layout_version=%u "
           "backend_id=%u (%s) build_flags_hash=0x%08x\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, backend_id_name(req.backend_id), req.build_flags_hash);
    printf("\n");
}

/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    /* --print-mem-size diagnostic mode: report the pool byte budget (and
     * exercise a real audio_pipeline_init(), same guard coverage the old
     * pipeline_build() had) without any WAV I/O. */
    if (argc >= 2 && strcmp(argv[1], "--print-mem-size") == 0) {
        AecPreset     preset      = AEC_PRESET_BALANCED;
        MmseLsaNrMode nr_mode     = MMSE_LSA_NR_BALANCED;
        int           aec_only    = 0;
        int           sample_rate = 16000;
        int           fft_size    = 0;

        for (int i = 2; i < argc; i++) {
            if      (strcmp(argv[i], "--aec-only") == 0) aec_only = 1;
            else if (strcmp(argv[i], "--nr-preset") == 0 && i + 1 < argc)
                nr_mode = parse_nr_mode(argv[++i]);
            else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc)
                sample_rate = atoi(argv[++i]);
            else if (strcmp(argv[i], "--fft-size") == 0 && i + 1 < argc)
                fft_size = atoi(argv[++i]);
            else if (argv[i][0] != '-')
                preset = parse_preset(argv[i]);
        }

        AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
        cfg.fft_size  = fft_size;
        cfg.aec_preset = preset;
        cfg.nr_mode    = nr_mode;
        cfg.aec_only   = aec_only;

        print_mem_budget(&cfg);

        AudioPipelineMemReq req;
        if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
            fprintf(stderr, "Error: invalid config (sample_rate=%d?)\n", sample_rate);
            return 1;
        }
        void* pool = malloc((size_t)req.bytes);   /* host stand-in for a platform memory block */
        if (!pool) { fprintf(stderr, "Error: malloc failed (%llu bytes)\n", (unsigned long long)req.bytes); return 1; }
        if (((uintptr_t)pool) % req.alignment != 0) {
            fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n", req.alignment, pool);
            free(pool); return 1;
        }

        AudioPipeline* pipe = audio_pipeline_init(pool, (size_t)req.bytes, &cfg);
        if (!pipe) { free(pool); return 1; }
        printf("n_freqs agreement OK (n_freqs=%d, hop=%d) at %d Hz\n",
               audio_pipeline_n_freqs(pipe), audio_pipeline_hop_size(pipe), sample_rate);
        audio_pipeline_destroy(pipe);
        free(pool);
        return 0;
    }

    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [aec-preset] "
               "[--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin] "
               "[--fft-size 256|512|1024] [--debug]\n",
               argv[0]);
        printf("       %s --print-mem-size [preset] [--nr-preset ...] [--aec-only] "
               "[--sample-rate <hz>] [--fft-size <n>]\n", argv[0]);
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
    int           fft_size = 0;

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0)    aec_only = 1;
        else if (strcmp(argv[i], "--legacy-amin") == 0) legacy = 1;
        else if (strcmp(argv[i], "--no-cng") == 0)      no_cng = 1;
        else if (strcmp(argv[i], "--debug") == 0)       debug_status = 1;
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
    /* CLI rate whitelist: audio_pipeline_get_mem_requirements() would reject
     * an unsupported rate anyway (aec_is_valid_sample_rate() under the hood)
     * — fail earlier and with a clearer message here, before any buffers/pool
     * are sized off it. */
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

    printf("AEC(linear) -> echo-aware NR -> RES  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n\n",
           preset_name(preset), nr_mode_name(nr_mode), !no_cng ? "on" : "off");

    /* === Query the descriptor, then allocate + hand in the single static
     * pool (the ONE allocation — host stand-in for a platform memory
     * block). This is the board story: a real integrator's memory manager
     * replaces this malloc with its own allocator, keeping everything else
     * identical. === */
    print_mem_budget(&cfg);

    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid config\n");
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    void* pool = malloc((size_t)req.bytes);
    if (!pool) { fprintf(stderr, "Error: malloc failed (%llu bytes)\n", (unsigned long long)req.bytes); return 1; }
    if (((uintptr_t)pool) % req.alignment != 0) {
        /* ALIGN16 contract (mem_align.h): don't rely on allocator luck. */
        fprintf(stderr, "Error: pool not %u-byte aligned (%p)\n", req.alignment, pool);
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    AudioPipeline* pipe = audio_pipeline_init(pool, (size_t)req.bytes, &cfg);
    if (!pipe) {
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    int hop = audio_pipeline_hop_size(pipe);
    int n_freqs = audio_pipeline_n_freqs(pipe);
    float* mic_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: hop-buffer alloc failed\n");
        audio_pipeline_destroy(pipe); free(pool);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) {
        fprintf(stderr, "Error: cannot create output\n");
        audio_pipeline_destroy(pipe); free(pool);
        free(mic_buf); free(ref_buf); free(out_buf);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* Optional per-frame context dump (DUMP_CTX=<path>) for port parity tests —
     * identical layout/semantics to the malloc pipeline: header [n_freqs, hop];
     * then per frame error_spec(2·nf) res_gain(nf) r2(nf) comfort_noise(nf)
     * far_power(1) g_nr(nf) out_hop(hop), all float32. Read via the
     * audio_pipeline_get_aec()/audio_pipeline_get_nr() handles — a dump record
     * is only ever written for a hop whose freq seam was actually available,
     * mirroring audio_pipeline_process()'s own internal fallback check. */
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

        audio_pipeline_process(pipe, mic_buf, ref_buf, out_buf);
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
    printf("Output: %s\n", out_path);

    /* === Cleanup ===
     * All modules and pipeline scratch live in `pool` — a single free()
     * releases everything after audio_pipeline_destroy() (teardown order
     * NR -> pipeline FFT -> AEC, all no-ops on this pool-resident instance —
     * see audio_pipeline.h). */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    audio_pipeline_destroy(pipe);
    free(pool);
    free(mic_buf); free(ref_buf); free(out_buf);
    return 0;
}
