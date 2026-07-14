/**
 * aec_nr_pipeline_static.c — AEC(linear) -> echo-aware NR -> RES  (Version B: static memory)
 *
 * STATIC-MEMORY MIRROR of pipelines/aec_nr_pipeline.c (Version A, malloc). The
 * DSP chain, framing, AEC preset handling, NR wiring (mmse_lsa_process_gain +
 * R²-folded noise floor) and RES combine/OLA are IDENTICAL — this file only
 * changes WHERE the module state and pipeline scratch buffers live: every
 * array comes from ONE caller-provided memory pool via
 *
 *   aec_get_mem_size()/aec_init()            — SE/AEC static API
 *   mmse_lsa_get_mem_size()/mmse_lsa_init()   — SE/NR static API
 *   fft_get_mem_size()/fft_init()             — audio_common static API
 *
 * plus the pipeline's own OLA/gain scratch, carved from the same pool by
 * pointer arithmetic (ALIGN16 bumps, mem_align.h). No malloc after the single
 * pool allocation in main() (the host stand-in for a platform memory block).
 *
 * `aec_nr_pipeline_static out.wav` must be BYTE-IDENTICAL to
 * `aec_nr_pipeline out.wav` for the same inputs/options/backend.
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
#include <math.h>

#include "aec.h"               /* AEC + fft_wrapper.h (Complex, FftHandle, ALIGN16) */
#include "fft_wrapper.h"       /* fft_get_mem_size/fft_init, fft_forward/inverse    */
#include "mmse_lsa_denoiser.h" /* freq-domain NR + mmse_lsa_process_gain            */
#include "wav_io.h"
#include "simd_kernels.h"      /* sk_min_f32 / sk_capply_gain_f32                    */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* Production recipe constants (mirror Python PROD_* in aec_nr_pipeline.py and
 * the malloc pipeline's own copy — must stay byte-identical). */
#define PROD_NE_FLOOR             0.4f
#define PROD_NE_FLOOR_FAR_ACTIVE  0.2f
#define PROD_FAR_GATE_THRESH      1e-4f
#define PROD_NEAR_GATE_THRESH     1e-3f
#define PROD_NEAR_HANGOVER        8
#define PSD_SCALE                 (32768.0f * 32768.0f)  /* int16² (Python _PSD_SCALE) */

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
 * read-only diagnostic queries (aec_debug_status() / mmse_lsa_debug_status()) —
 * byte-identical helper to the malloc pipeline's copy. Neither call mutates
 * DSP state or fast_math-approximates anything, so this is safe to call every
 * second regardless of preset/backend. aec_only (or nr==NULL) omits the NR
 * half — no denoiser exists in that mode.
 *
 * CAVEAT: this pipeline always runs AEC in linear mode (enable_res=0,
 * return_res_context=1 — the external NR/RES seam above), and aec.c only
 * caches last_erle_windowed when cfg.enable_res is true. So the "erle="
 * field below always reads 0.0dB here — expected given this pipeline's
 * config, not a broken query (see pipelines/README.md "Debugging &
 * Performance Flags"). */
static void print_debug_status(const Aec* aec, const MmseLsaDenoiser* nr,
                                int aec_only, float seconds) {
    AecDebugStatus a;
    aec_debug_status(aec, &a);
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

/* Deterministic standard-normal source for comfort noise — same generator as
 * the malloc pipeline (independent per-process global; not cross-binary
 * bit-identical to numpy, but reproducible and level-matched). */
static uint32_t g_rng = 0x9e3779b9u;
static float rng_uniform(void) {                    /* (0,1) */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((g_rng >> 8) + 0.5f) * (1.0f / 16777216.0f);
}
static float rng_gauss(void) {                      /* Box-Muller */
    float u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI_F * u2);
}

/* ------------------------------------------------------------------ */
/* Frame dimensions (shared 10ms-hop grid). hop/frame_sz use the SAME literal
 * expressions as the malloc pipeline's inline computation (byte-for-byte the
 * same values at every sample rate the two binaries are compared at).
 *
 * fft_sz is the TRUE next-pow2(frame_sz) — starting the doubling from 1, not
 * from a hardcoded 512 (the malloc pipeline's `int fft_sz = 512; while
 * (fft_sz < frame_sz) fft_sz *= 2;` — a starting constant that happens to
 * equal next-pow2 for every frame_sz > 256, i.e. sr >= ~12.8 kHz, but
 * OVERSHOOTS below that: at 8 kHz frame_sz=160 so the hardcoded-512 loop
 * never doubles and stays at 512 (257 bins) while AEC's own internal grid
 * (aec.c next_pow2(block_size)) correctly lands on 256 (129 bins) — the "8
 * kHz FFT mismatch" this pipeline must not carry. Starting from 1 matches
 * AEC's derivation exactly at EVERY sample rate, and is IDENTICAL to the
 * malloc pipeline's result whenever frame_sz > 256 (512 @ 16 kHz, 1024 @
 * 48 kHz — verified byte-identical below), so this is a strict fix with zero
 * risk to the byte-identical requirement at the rates actually compared. */
static void compute_frame_dims(int sr, int* o_hop, int* o_frame_sz,
                                int* o_fft_sz, int* o_n_freqs) {
    int hop      = (int)(0.01f * sr);
    int frame_sz = 2 * hop;
    int fft_sz   = 1;
    while (fft_sz < frame_sz) fft_sz *= 2;
    *o_hop = hop; *o_frame_sz = frame_sz; *o_fft_sz = fft_sz;
    *o_n_freqs = fft_sz / 2 + 1;
}

/* ------------------------------------------------------------------ */
/* Static-memory pool — one module set (AEC / FFT / NR) + pipeline scratch.  */

typedef struct {
    void*  pool;
    size_t pool_size;

    Aec*             aec;
    FftHandle*       fft;
    MmseLsaDenoiser* nr;

    float* mic_buf;
    float* ref_buf;
    float* aec_out;
    float* out_buf;
    float* synth_win;
    float* ola;
    float* ifft_buf;      /* irfft -> fft_sz samples (NOT frame_sz — B1 fix) */
    float* g_nr;
    float* g_total;
    float* g_aec;
    float* extra;
    float* e2;            /* |E(f)|² scratch (hoisted out of the RES loops) */
    Complex* spec;
} Pipeline;

/* Total pool bytes for the given configs/dims. aec_cfg->sample_rate and
 * nr_cfg->fft_size/frame_size/hop_size MUST already be finalised (forced onto
 * the shared grid — see main()) before calling this. */
static size_t pipeline_pool_size(const AecConfig* aec_cfg,
                                  const MmseLsaConfig* nr_cfg,
                                  int hop, int frame_sz, int fft_sz, int n_freqs,
                                  int aec_only) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_sz_mem = aec_only ? 0 : fft_get_mem_size(fft_sz);
    size_t nr_sz  = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);

    size_t pipe = 0;
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* mic_buf */
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* ref_buf */
    pipe += ALIGN16((size_t)hop * sizeof(float));   /* aec_out */
    if (!aec_only) {
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* synth_win */
        pipe += ALIGN16((size_t)frame_sz * sizeof(float));   /* ola       */
        pipe += ALIGN16((size_t)fft_sz   * sizeof(float));   /* ifft_buf (fft_sz, B1 fix) */
        pipe += ALIGN16((size_t)hop      * sizeof(float));   /* out_buf   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_nr      */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_total   */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* g_aec     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* extra     */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(float));   /* e2        */
        pipe += ALIGN16((size_t)n_freqs  * sizeof(Complex)); /* spec      */
    }

    return ALIGN16(aec_sz) + ALIGN16(fft_sz_mem) + ALIGN16(nr_sz) + pipe;
}

static void print_mem_budget(const AecConfig* aec_cfg, const MmseLsaConfig* nr_cfg,
                              int hop, int frame_sz, int fft_sz, int n_freqs,
                              int aec_only) {
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    size_t fft_mem = aec_only ? 0 : fft_get_mem_size(fft_sz);
    size_t nr_sz  = aec_only ? 0 : mmse_lsa_get_mem_size(nr_cfg);
    size_t total  = pipeline_pool_size(aec_cfg, nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);
    size_t pipe   = total - ALIGN16(aec_sz) - ALIGN16(fft_mem) - ALIGN16(nr_sz);

    printf("Memory Budget (Static Pipeline)\n");
    printf("================================\n");
    printf("  sample_rate=%d hop=%d frame_sz=%d fft_sz=%d n_freqs=%d\n",
           aec_cfg->sample_rate, hop, frame_sz, fft_sz, n_freqs);
    printf("  AEC:            %7zu bytes (%6.1f KB)\n", aec_sz, (float)aec_sz / 1024.0f);
    if (!aec_only) {
        printf("  FFT (OLA):      %7zu bytes (%6.1f KB)\n", fft_mem, (float)fft_mem / 1024.0f);
        printf("  NR (MMSE-LSA):  %7zu bytes (%6.1f KB)\n", nr_sz, (float)nr_sz / 1024.0f);
    }
    printf("  Pipeline bufs:  %7zu bytes (%6.1f KB)\n", pipe, (float)pipe / 1024.0f);
    printf("  --------------------------------\n");
    printf("  Total:          %7zu bytes (%6.1f KB)\n", total, (float)total / 1024.0f);
    printf("\n");
}

/* Slice `pool` (>= pipeline_pool_size(...) bytes, 16-byte aligned) into every
 * module + pipeline scratch buffer. Returns 0 on success, <0 on failure
 * (undersized pool, module init failure, or an n_freqs/hop disagreement
 * between AEC / NR / FFT — the 8kHz-grid guard). No malloc is called here. */
static int pipeline_build(Pipeline* p, void* pool, size_t pool_size,
                          const AecConfig* aec_cfg, const MmseLsaConfig* nr_cfg,
                          int hop, int frame_sz, int fft_sz, int n_freqs,
                          int aec_only) {
    memset(p, 0, sizeof(*p));
    size_t needed = pipeline_pool_size(aec_cfg, nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);
    if (!pool || pool_size < needed) {
        fprintf(stderr, "Error: pool too small (%zu < %zu)\n", pool_size, needed);
        return -1;
    }
    p->pool = pool; p->pool_size = pool_size;
    uint8_t* ptr = (uint8_t*)pool;

    /* AEC (its own internal post_fft is sized/placed inside aec_get_mem_size /
     * aec_init — a separate FFT instance from the pipeline's own `fft` below). */
    size_t aec_sz = aec_get_mem_size(aec_cfg);
    p->aec = aec_init(ptr, aec_sz, aec_cfg);
    ptr += ALIGN16(aec_sz);
    if (!p->aec) { fprintf(stderr, "Error: aec_init failed\n"); return -1; }

    if (!aec_only) {
        size_t fft_mem = fft_get_mem_size(fft_sz);
        p->fft = fft_init(ptr, fft_mem, fft_sz);
        ptr += ALIGN16(fft_mem);

        size_t nr_sz = mmse_lsa_get_mem_size(nr_cfg);
        p->nr = mmse_lsa_init(ptr, nr_sz, nr_cfg);
        ptr += ALIGN16(nr_sz);

        if (!p->fft || !p->nr) { fprintf(stderr, "Error: NR/FFT init failed\n"); return -1; }
    }

    p->mic_buf = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    p->ref_buf = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    p->aec_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));

    if (!aec_only) {
        p->synth_win = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ola       = (float*)ptr;   ptr += ALIGN16((size_t)frame_sz * sizeof(float));
        p->ifft_buf  = (float*)ptr;   ptr += ALIGN16((size_t)fft_sz   * sizeof(float)); /* B1 fix */
        p->out_buf   = (float*)ptr;   ptr += ALIGN16((size_t)hop      * sizeof(float));
        p->g_nr      = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->g_total   = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->g_aec     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->extra     = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->e2        = (float*)ptr;   ptr += ALIGN16((size_t)n_freqs  * sizeof(float));
        p->spec      = (Complex*)ptr; ptr += ALIGN16((size_t)n_freqs  * sizeof(Complex));

        /* sqrt of periodic Hann (denom = block_size) — matches Python run_res
         * synth_win = sqrt(0.5·(1 - cos(2π k / block_size))), byte-identical
         * to the malloc pipeline's synthesis window. */
        for (int k = 0; k < frame_sz; k++)
            p->synth_win[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * M_PI_F * k / frame_sz)));
    }

    /* --- n_freqs/hop agreement guard (the "8 kHz FFT mismatch" fix) ---
     * aec_get_res_context() is readable right after aec_init (a->n_freqs /
     * a->hop_size are set at init time, before any frame is processed), so
     * this check runs at INIT, before a single sample is read — a real
     * mismatch aborts loudly instead of silently overrunning ctx buffers. */
    AecResContext ctx0;
    aec_get_res_context(p->aec, &ctx0);
    if (ctx0.n_freqs != n_freqs || ctx0.hop_size != hop) {
        fprintf(stderr, "FATAL: AEC/pipeline grid mismatch — pipeline n_freqs=%d hop=%d, "
                        "AEC n_freqs=%d hop=%d\n", n_freqs, hop, ctx0.n_freqs, ctx0.hop_size);
        return -1;
    }
    if (!aec_only) {
        int fft_nf = fft_get_n_freqs(p->fft);
        int nr_nf  = mmse_lsa_get_n_freqs(p->nr);
        if (fft_nf != n_freqs || nr_nf != n_freqs) {
            fprintf(stderr, "FATAL: FFT/NR grid mismatch — pipeline n_freqs=%d, "
                            "fft n_freqs=%d, nr n_freqs=%d\n", n_freqs, fft_nf, nr_nf);
            return -1;
        }
    }
    return 0;
}

static void pipeline_destroy(Pipeline* p) {
    if (!p) return;
    if (p->nr)  mmse_lsa_destroy(p->nr);    /* no-op for static instances */
    if (p->fft) fft_destroy(p->fft);        /* no-op for static instances */
    if (p->aec) aec_destroy(p->aec);        /* no-op for static instances */
    /* p->pool itself is owned by the caller (main()) — freed there. */
}

/* ------------------------------------------------------------------ */

int main(int argc, char* argv[]) {
    /* --print-mem-size diagnostic mode: report the static-pool byte budget
     * (and exercise the full init + n_freqs/hop guard) without any WAV I/O.
     * Kept separate from the mirrored mic/ref/out path below so option
     * parsing for the real run stays byte-for-byte identical to the malloc
     * pipeline. */
    if (argc >= 2 && strcmp(argv[1], "--print-mem-size") == 0) {
        AecPreset     preset      = AEC_PRESET_BALANCED;
        MmseLsaNrMode nr_mode     = MMSE_LSA_NR_BALANCED;
        int           aec_only    = 0;
        int           sample_rate = 16000;

        for (int i = 2; i < argc; i++) {
            if      (strcmp(argv[i], "--aec-only") == 0) aec_only = 1;
            else if (strcmp(argv[i], "--nr-preset") == 0 && i + 1 < argc)
                nr_mode = parse_nr_mode(argv[++i]);
            else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc)
                sample_rate = atoi(argv[++i]);
            else if (argv[i][0] != '-')
                preset = parse_preset(argv[i]);
        }

        AecConfig aec_cfg;
        aec_config_from_preset(&aec_cfg, preset, sample_rate);
        aec_cfg.enable_res = 0;
        aec_cfg.return_res_context = 1;

        MmseLsaConfig nr_cfg = mmse_lsa_config_for_mode(sample_rate, nr_mode);
        nr_cfg.L = 150; nr_cfg.alpha_d = 0.95f;
        nr_cfg.alpha_attack = 0.3f; nr_cfg.alpha_decay = nr_cfg.alpha_g;

        int hop, frame_sz, fft_sz, n_freqs;
        compute_frame_dims(sample_rate, &hop, &frame_sz, &fft_sz, &n_freqs);
        /* Force the NR config onto the SAME fft grid as the pipeline/AEC
         * (the "8 kHz FFT mismatch" fix — otherwise mmse_lsa_default_config's
         * own fft_size derivation (seeded at 256) can diverge from this
         * pipeline's (seeded at 512) below 12.8 kHz). */
        nr_cfg.fft_size = fft_sz; nr_cfg.frame_size = frame_sz; nr_cfg.hop_size = hop;

        print_mem_budget(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);

        size_t total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);
        void* pool = malloc(total);
        if (!pool) { fprintf(stderr, "Error: malloc failed (%zu bytes)\n", total); return 1; }
        if (((uintptr_t)pool) % 16 != 0) {
            fprintf(stderr, "Error: pool not 16-byte aligned (%p)\n", pool);
            free(pool); return 1;
        }
        memset(pool, 0, total);

        Pipeline pipe;
        int rc = pipeline_build(&pipe, pool, total, &aec_cfg, &nr_cfg,
                                hop, frame_sz, fft_sz, n_freqs, aec_only);
        if (rc != 0) { free(pool); return 1; }
        printf("n_freqs agreement OK (pipeline=%d, AEC=%s, FFT/NR=%s) at %d Hz\n",
               n_freqs, "n_freqs", aec_only ? "skipped(--aec-only)" : "n_freqs", sample_rate);
        pipeline_destroy(&pipe);
        free(pool);
        return 0;
    }

    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [aec-preset] "
               "[--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin] "
               "[--debug]\n",
               argv[0]);
        printf("       %s --print-mem-size [preset] [--nr-preset ...] [--aec-only] "
               "[--sample-rate <hz>]\n", argv[0]);
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

    for (int i = 4; i < argc; i++) {
        if      (strcmp(argv[i], "--aec-only") == 0)    aec_only = 1;
        else if (strcmp(argv[i], "--legacy-amin") == 0) legacy = 1;
        else if (strcmp(argv[i], "--no-cng") == 0)      no_cng = 1;
        else if (strcmp(argv[i], "--debug") == 0)       debug_status = 1;
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

    int sr        = mic_r->info.sample_rate;
    int n_samples = (mic_r->info.num_samples < ref_r->info.num_samples)
                  ? mic_r->info.num_samples : ref_r->info.num_samples;

    /* AEC config: LINEAR mode + res-context seam (identical to the malloc
     * pipeline — enable_res=0, return_res_context=1). */
    AecConfig aec_cfg;
    aec_config_from_preset(&aec_cfg, preset, sr);
    aec_cfg.enable_res         = 0;
    aec_cfg.return_res_context = 1;
    int enable_cng = aec_cfg.enable_cng && !no_cng;   /* preset default (1) */

    MmseLsaConfig nr_cfg = mmse_lsa_config_for_mode(sr, nr_mode);
    /* Match the Python pipeline _build_denoiser STRUCTURAL tuning (identical
     * to the malloc pipeline): L=150, alpha_d=0.95, alpha_attack/alpha_decay
     * pinned off the C-only per-mode values. See aec_nr_pipeline.py:_build_denoiser. */
    nr_cfg.L       = 150;
    nr_cfg.alpha_d = 0.95f;
    nr_cfg.alpha_attack = 0.3f;
    nr_cfg.alpha_decay  = nr_cfg.alpha_g;

    /* Frame dimensions (shared 10ms-hop grid). */
    int hop, frame_sz, fft_sz, n_freqs;
    compute_frame_dims(sr, &hop, &frame_sz, &fft_sz, &n_freqs);
    /* Force the NR config onto the SAME fft grid the pipeline/AEC derive
     * (closes the 8 kHz mmse_lsa_default_config-seeded-at-256-vs-pipeline-
     * seeded-at-512 gap; at 16 kHz this is a no-op — both already give 512). */
    nr_cfg.fft_size = fft_sz; nr_cfg.frame_size = frame_sz; nr_cfg.hop_size = hop;

    printf("AEC(linear) -> echo-aware NR -> RES  (static memory%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n\n",
           preset_name(preset), nr_mode_name(nr_mode), enable_cng ? "on" : "off");

    /* === Allocate + slice the single static pool === */
    size_t total = pipeline_pool_size(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);
    print_mem_budget(&aec_cfg, &nr_cfg, hop, frame_sz, fft_sz, n_freqs, aec_only);

    void* pool = malloc(total);   /* the ONE allocation — host stand-in for a platform block */
    if (!pool) { fprintf(stderr, "Error: malloc failed (%zu bytes)\n", total); return 1; }
    if (((uintptr_t)pool) % 16 != 0) {
        /* ALIGN16 contract (mem_align.h): don't rely on allocator luck. */
        fprintf(stderr, "Error: pool not 16-byte aligned (%p)\n", pool);
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }
    memset(pool, 0, total);

    Pipeline P;
    if (pipeline_build(&P, pool, total, &aec_cfg, &nr_cfg,
                       hop, frame_sz, fft_sz, n_freqs, aec_only) != 0) {
        free(pool); wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }
    Aec*             aec = P.aec;
    FftHandle*       fft = P.fft;
    MmseLsaDenoiser* nr  = P.nr;
    float* mic_buf   = P.mic_buf;
    float* ref_buf   = P.ref_buf;
    float* aec_out   = P.aec_out;
    float* out_buf   = P.out_buf;
    float* synth_win = P.synth_win;
    float* ola       = P.ola;
    float* ifft_buf  = P.ifft_buf;
    float* g_nr      = P.g_nr;
    float* g_total   = P.g_total;
    float* g_aec     = P.g_aec;
    float* extra     = P.extra;
    float* e2        = P.e2;
    Complex* spec    = P.spec;

    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) {
        fprintf(stderr, "Error: cannot create output\n");
        pipeline_destroy(&P); free(pool);
        wav_close_read(mic_r); wav_close_read(ref_r); return 1;
    }

    /* Optional per-frame context dump (DUMP_CTX=<path>) for port parity tests —
     * identical layout/semantics to the malloc pipeline: header [n_freqs, hop];
     * then per frame error_spec(2·nf) res_gain(nf) r2(nf) comfort_noise(nf)
     * far_power(1) g_nr(nf) out_hop(hop), all float32. */
    FILE* dctx = NULL;
    const char* dpath = getenv("DUMP_CTX");
    if (dpath && !aec_only) {
        dctx = fopen(dpath, "wb");
        if (dctx) { int hdr[2] = { n_freqs, hop }; fwrite(hdr, sizeof(int), 2, dctx); }
    }

    int near_hang = 0;   /* near-activity hangover counter (gated floor) */

    /* === Processing loop (identical to the malloc pipeline) === */
    int processed = 0;
    while (processed + hop <= n_samples) {
        wav_read_float(mic_r, mic_buf, hop);
        wav_read_float(ref_r, ref_buf, hop);

        /* Stage 1: AEC (linear residual in aec_out; freq seam in ctx). */
        aec_process(aec, mic_buf, ref_buf, aec_out);

        if (aec_only) {
            wav_write_float(writer, aec_out, hop);
            processed += hop;
            if (processed % sr == 0) {
                printf("."); fflush(stdout);
                if (debug_status) print_debug_status(aec, NULL, 1, (float)processed / sr);
            }
            continue;
        }

        AecResContext ctx;
        aec_get_res_context(aec, &ctx);
        if (!ctx.error_spec || !ctx.res_gain) {       /* seam unavailable → linear */
            wav_write_float(writer, aec_out, hop);
            processed += hop;
            continue;
        }

        /* Stage 2: echo-aware NR gain. extra = R²/PSD_SCALE folds the residual
         * echo into the noise floor (ξ = S²/(N²+R²)); off in legacy. */
        const float* nr_extra = NULL;
        if (!legacy && ctx.r2) {
            for (int k = 0; k < n_freqs; k++)
                extra[k] = ctx.r2[k] / PSD_SCALE;
            nr_extra = extra;
        }
        mmse_lsa_process_gain(nr, ctx.error_spec, nr_extra, g_nr);

        /* Stage 3a: g_total = min(G_nr, G_res). g_aec (= G_res, pre-min) sets the
         * comfort-noise level so CNG reflects AEC suppression only. */
        memcpy(g_aec, ctx.res_gain, (size_t)n_freqs * sizeof(float));
        sk_min_f32(g_total, g_nr, g_aec, n_freqs);

        /* |E(f)|² scratch hoist: both the near-energy mean below and the
         * echo-gated lift loop need re*re+im*im per bin — compute it once
         * here (exact same expression text as both original inline sites)
         * instead of twice. */
        for (int k = 0; k < n_freqs; k++) {
            float re = ctx.error_spec[k].r, im = ctx.error_spec[k].i;
            e2[k] = re * re + im * im;
        }

        /* Stage 3b: far-activity + near-VAD gated near-end floor strength. */
        float nf_eff = PROD_NE_FLOOR;
        if (!legacy) {
            int far_active = ctx.far_power > PROD_FAR_GATE_THRESH;
            /* near_energy = mean |E(f)|² (≈ near+noise when far-silent). */
            float ne = 0.0f;
            for (int k = 0; k < n_freqs; k++) {
                ne += e2[k];
            }
            ne /= (float)n_freqs;
            if (ne > PROD_NEAR_GATE_THRESH) near_hang = PROD_NEAR_HANGOVER;
            int near_active = near_hang > 0;
            if (near_hang > 0) near_hang--;
            int protect = (!far_active) && near_active;
            nf_eff = protect ? PROD_NE_FLOOR : PROD_NE_FLOOR_FAR_ACTIVE;
        }

        /* Per-bin echo-gated near-end lift (ne_gate='both': G_res·(1-echo_frac)). */
        if (nf_eff > 0.0f && ctx.r2) {
            for (int k = 0; k < n_freqs; k++) {
                float r2_nr = ctx.r2[k] / PSD_SCALE;
                float echo_frac = r2_nr / (e2[k] + 1e-12f);
                if (echo_frac < 0.0f) echo_frac = 0.0f;
                if (echo_frac > 1.0f) echo_frac = 1.0f;
                float no_echo = ctx.res_gain[k] * (1.0f - echo_frac);
                float lift = nf_eff * no_echo;
                g_total[k] = (1.0f - lift) * g_total[k] + lift; /* blend toward 1 */
            }
        }

        /* S(f) = E(f) · g_total */
        sk_capply_gain_f32(spec, ctx.error_spec, g_total, n_freqs);

        /* Comfort noise on the cut bins: level = sqrt(N²/PSD_SCALE), scaled by
         * sqrt(1 - g_aec²) so it fills only what the AEC suppressed (bins 1..N-2). */
        if (enable_cng && ctx.comfort_noise) {
            for (int k = 1; k < n_freqs - 1; k++) {
                float n_amp = ctx.comfort_noise[k] / PSD_SCALE;
                n_amp = (n_amp > 0.0f) ? sqrtf(n_amp) : 0.0f;
                float ng2 = 1.0f - g_aec[k] * g_aec[k];
                float noise_gain = (ng2 > 0.0f) ? sqrtf(ng2) : 0.0f;
                float a = noise_gain * n_amp;
                spec[k].r += a * rng_gauss();
                spec[k].i += a * rng_gauss();
            }
        }

        /* irfft → sqrt-Hann OLA → output one hop. */
        fft_inverse(fft, spec, ifft_buf);
        for (int k = 0; k < frame_sz; k++) ola[k] += ifft_buf[k] * synth_win[k];
        memcpy(out_buf, ola, (size_t)hop * sizeof(float));
        memmove(ola, ola + hop, (size_t)(frame_sz - hop) * sizeof(float));
        memset(ola + (frame_sz - hop), 0, (size_t)hop * sizeof(float));

        wav_write_float(writer, out_buf, hop);

        if (dctx) {
            fwrite(ctx.error_spec, sizeof(Complex), n_freqs, dctx);
            fwrite(ctx.res_gain, sizeof(float), n_freqs, dctx);
            fwrite(ctx.r2, sizeof(float), n_freqs, dctx);
            fwrite(ctx.comfort_noise, sizeof(float), n_freqs, dctx);
            float fp = ctx.far_power; fwrite(&fp, sizeof(float), 1, dctx);
            fwrite(g_nr, sizeof(float), n_freqs, dctx);
            fwrite(out_buf, sizeof(float), hop, dctx);
        }

        processed += hop;
        if (processed % sr == 0) {
            printf("."); fflush(stdout);
            if (debug_status) print_debug_status(aec, nr, aec_only, (float)processed / sr);
        }
    }
    if (dctx) fclose(dctx);

    printf("\nProcessed: %d samples (%.2fs)\n", processed, (float)processed / sr);
    printf("Output: %s\n", out_path);

    /* === Cleanup ===
     * All modules and pipeline scratch live in `pool` — a single free()
     * releases everything. aec_destroy/mmse_lsa_destroy/fft_destroy are
     * no-ops on static instances (is_static=1) but are still called for
     * hygiene/parity with the malloc pipeline's cleanup shape. */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    pipeline_destroy(&P);
    free(pool);
    return 0;
}
