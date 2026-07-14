/**
 * aec_nr_pipeline.c — AEC(linear) → echo-aware NR → RES  (Version A: malloc)
 *
 * Freq-domain A_min_pl pipeline, a C port of pipelines/aec_nr_pipeline.py
 * (the 2026-06-23 re-tune). Per hop:
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
#include <math.h>

#include "aec.h"               /* AEC + fft_wrapper.h (Complex, FftHandle)   */
#include "fft_wrapper.h"       /* fft_create/forward/inverse                 */
#include "mmse_lsa_denoiser.h" /* freq-domain NR + mmse_lsa_process_gain     */
#include "wav_io.h"
#include "simd_kernels.h"      /* sk_min_f32 / sk_capply_gain_f32             */

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* Production recipe constants (mirror Python PROD_* in aec_nr_pipeline.py). */
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
 * read-only diagnostic queries (aec_debug_status() / mmse_lsa_debug_status()).
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

/* Deterministic standard-normal source for comfort noise. The Python pipeline
 * uses np.random.RandomState(0); the float kiss-FFT already prevents bit-exact
 * parity, so this is an independent (reproducible, perceptually equivalent)
 * realisation at the SAME per-bin level — it is not sample-identical to numpy. */
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

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <mic.wav> <ref.wav> <out.wav> [aec-preset] "
               "[--nr-preset mild|balanced|aggressive] [--aec-only] [--legacy-amin] "
               "[--debug]\n",
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

    /* AEC config: LINEAR mode + res-context seam. enable_res=0 keeps the time
     * output linear; return_res_context=1 makes the AEC3 post block compute and
     * expose E/G_res/R²/CNG (mirrors Python run_aec_linear). */
    AecConfig aec_cfg;
    aec_config_from_preset(&aec_cfg, preset, sr);
    aec_cfg.enable_res         = 0;
    aec_cfg.return_res_context = 1;
    int enable_cng = aec_cfg.enable_cng && !no_cng;   /* preset default (1) */

    MmseLsaConfig nr_cfg = mmse_lsa_config_for_mode(sr, nr_mode);
    /* Match the Python pipeline _build_denoiser STRUCTURAL tuning, which
     * overrides the NR v3_2 defaults (config_for_mode) for the AEC-residual
     * grid: L=150 (1.5s minima window vs 32) and alpha_d=0.95 (slower noise
     * tracking at 10ms hop vs 0.7). The strength quartet {g_min,q,xi_min,alpha_g}
     * stays from the preset. See pipelines/aec_nr_pipeline.py:_build_denoiser. */
    nr_cfg.L       = 150;
    nr_cfg.alpha_d = 0.95f;
    /* Python _build_denoiser leaves gain-smoothing at the v3_2 defaults for ALL
     * presets (does NOT pass the C-only per-mode alpha_attack/alpha_decay):
     * alpha_attack=0.3 fixed, alpha_decay=None→alpha_g. config_for_mode's
     * MILD/AGGRESSIVE values (0.4/0.92, 0.15/0.85) are C-standalone-only and
     * diverge from the pipeline's Python reference — override them here so the
     * pipeline C matches Python on every preset (BALANCED already equal). */
    nr_cfg.alpha_attack = 0.3f;
    nr_cfg.alpha_decay  = nr_cfg.alpha_g;

    /* Frame dimensions (shared 10ms-hop grid) */
    int hop      = (int)(0.01f * sr);   /* 160 @ 16k */
    int frame_sz = 2 * hop;             /* 320       */
    int fft_sz   = 512;
    while (fft_sz < frame_sz) fft_sz *= 2;
    int n_freqs  = fft_sz / 2 + 1;      /* 257       */

    printf("AEC(linear) -> echo-aware NR -> RES  (freq A_min_pl%s)\n",
           legacy ? ", legacy min-only" : "");
    printf("  Input:  %s (%.2fs)\n", mic_path, (float)n_samples / sr);
    printf("  AEC preset: %s   NR preset: %s   CNG: %s\n\n",
           preset_name(preset), nr_mode_name(nr_mode), enable_cng ? "on" : "off");

    /* Create AEC */
    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec || aec_create(aec, &aec_cfg) != 0) {
        fprintf(stderr, "Error: AEC create failed\n"); return 1;
    }

    /* Create FFT (irfft bridge) + NR */
    FftHandle*       fft = NULL;
    MmseLsaDenoiser* nr  = NULL;
    float   *synth_win = NULL, *ola = NULL, *ifft_buf = NULL, *out_buf = NULL;
    float   *g_nr = NULL, *g_total = NULL, *g_aec = NULL, *extra = NULL, *e2 = NULL;
    Complex *spec = NULL;

    if (!aec_only) {
        fft = fft_create(fft_sz);
        nr  = mmse_lsa_create(&nr_cfg);
        if (!fft || !nr) { fprintf(stderr, "Error: NR/FFT create failed\n"); return 1; }

        synth_win = (float*)malloc((size_t)frame_sz * sizeof(float));
        ola       = (float*)calloc((size_t)frame_sz, sizeof(float));
        ifft_buf  = (float*)malloc((size_t)fft_sz   * sizeof(float)); /* irfft → fft_sz */
        out_buf   = (float*)malloc((size_t)hop      * sizeof(float));
        g_nr      = (float*)malloc((size_t)n_freqs  * sizeof(float));
        g_total   = (float*)malloc((size_t)n_freqs  * sizeof(float));
        g_aec     = (float*)malloc((size_t)n_freqs  * sizeof(float));
        extra     = (float*)malloc((size_t)n_freqs  * sizeof(float));
        e2        = (float*)malloc((size_t)n_freqs  * sizeof(float)); /* |E(f)|² scratch */
        spec      = (Complex*)malloc((size_t)n_freqs * sizeof(Complex));
        if (!synth_win || !ola || !ifft_buf || !out_buf || !g_nr || !g_total
            || !g_aec || !extra || !e2 || !spec) {
            fprintf(stderr, "Error: pipeline buffer alloc failed\n"); return 1;
        }
        /* sqrt of periodic Hann (denom = block_size) — matches Python run_res
         * synth_win = sqrt(0.5·(1 - cos(2π k / block_size))). */
        for (int k = 0; k < frame_sz; k++)
            synth_win[k] = sqrtf(0.5f * (1.0f - cosf(2.0f * M_PI_F * k / frame_sz)));
    }

    float* mic_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* ref_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* aec_out = (float*)malloc((size_t)hop * sizeof(float));

    WavWriter* writer = wav_open_write(out_path, sr, 1);
    if (!writer) { fprintf(stderr, "Error: cannot create output\n"); return 1; }

    /* Optional per-frame context dump (DUMP_CTX=<path>) for port parity tests:
     * header [n_freqs, hop]; then per frame error_spec(2·nf) res_gain(nf) r2(nf)
     * comfort_noise(nf) far_power(1) g_nr(nf) out_hop(hop), all float32. */
    FILE* dctx = NULL;
    const char* dpath = getenv("DUMP_CTX");
    if (dpath && !aec_only) {
        dctx = fopen(dpath, "wb");
        if (dctx) { int hdr[2] = { n_freqs, hop }; fwrite(hdr, sizeof(int), 2, dctx); }
    }

    int near_hang = 0;   /* near-activity hangover counter (gated floor) */

    /* === Processing loop === */
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

    /* Cleanup */
    wav_close_read(mic_r);
    wav_close_read(ref_r);
    wav_close_write(writer);
    free(mic_buf); free(ref_buf); free(aec_out);
    if (!aec_only) {
        free(synth_win); free(ola); free(ifft_buf); free(out_buf);
        free(g_nr); free(g_total); free(g_aec); free(extra); free(e2); free(spec);
        mmse_lsa_destroy(nr);
        fft_destroy(fft);
    }
    aec_destroy(aec);
    free(aec);
    return 0;
}
