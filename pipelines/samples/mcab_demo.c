/* mcab_demo.c -- reference wiring for independent AEC lanes -> SRP -> BF.
 *
 * Production uses 4aec_nr_res; this example omits its shared matched-filter
 * duty-cycle scheduling, backward-delay quarantine, shared alignment and
 * far-FFT reuse. Each lane here owns its own delay estimate and filter.
 *
 * The legacy aec_process() convenience entry point supplies time audio for
 * the per-lane RMS report. A subsequent aec_get_res_context() on that SAME
 * hop supplies the formed spectrum for SRP/BF without processing again.
 * enable_res=0 / return_res_context=1 disables applied RES gain but keeps
 * the context calculation. error_spec follows the formed selector, including
 * its capture fallback; it must not be assumed identical to emitted time
 * audio on every hop. This demo reports the BF spectrum and does no final
 * BF inverse FFT/OLA. Production applications own that synthesis step.
 *
 * All objects use heap convenience constructors; BF borrows SRP's steering
 * table, so teardown destroys BF before SRP. The deterministic test signal
 * combines a planted near-end plane wave, per-lane delayed far-end echo and
 * sensor noise. Its tau(theta)=-(x*cos(theta)+y*sin(theta))/c convention
 * matches audiolib/doa's steering vectors. No corpus is required. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "fft_wrapper.h" /* Complex { float r; float i; } */
#include "srp.h"
#include "bf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MCAB_NUM_LANES   4
#define MCAB_SAMPLE_RATE 16000
#define MCAB_NUM_ANGLES  72    /* 5-degree grid, same resolution the
                                   production pipeline's own default
                                   (DEFAULT_ANGLES) uses */
#define MCAB_NUM_HOPS    400   /* enough hops, at any of this grid's hop
                                   sizes, for the matched-filter delay
                                   estimator and the SRP/BF smoother to
                                   settle (see MCAB_LEADIN_HOPS below) */
#define MCAB_LEADIN_HOPS 200   /* far-end-only lead-in: lets each lane's
                                   matched-filter delay estimator converge
                                   on the echo path BEFORE the near-end
                                   (planted-DOA) tone turns on, instead of
                                   asking the AEC to adapt under continuous
                                   double-talk from sample 0 -- the single
                                   hardest case for a linear echo canceller */
#define MCAB_DOA_REPORT_WINDOW 20 /* final DOA/BF-RMS summary is a circular
                                   mean over the last N hops, not just the
                                   very last one, since any single hop's
                                   raw SRP-PHAT estimate is noisy */
#define MCAB_SPEED_OF_SOUND_M_S 343.0f

#define MCAB_PLANTED_DOA_DEG 60.0f
#define MCAB_NEAR_FREQ_HZ    1200.0f
#define MCAB_FAR_FREQ_HZ      300.0f
#define MCAB_NEAR_GAIN  0.35f
#define MCAB_ECHO_GAIN  0.40f
#define MCAB_FAR_GAIN   0.60f
#define MCAB_NOISE_AMP  0.02f

/* Per-lane echo path: a fixed integer-sample delay (different per mic,
 * standing in for different acoustic path lengths) applied to the same
 * far-end tone every lane's AEC must cancel. */
static const int MCAB_ECHO_DELAY_SAMPLES[MCAB_NUM_LANES] = {6, 9, 12, 15};

/* 4-mic square array (5 cm edge), matching the tau(theta) convention
 * audiolib/doa/srp.c's a_array construction uses: tau_m(theta) =
 * -(x_m*cos(theta) + y_m*sin(theta)) / c. A COLLINEAR (e.g. all y=0)
 * array cannot distinguish theta from -theta (cos is the same for both),
 * so this demo deliberately spans both x and y to give the planted DOA a
 * unique SRP-PHAT peak over the full 0..360 degree grid. */
static const float MCAB_GEOM_X[MCAB_NUM_LANES] = {
    -0.025f, 0.025f, 0.025f, -0.025f};
static const float MCAB_GEOM_Y[MCAB_NUM_LANES] = {
    -0.025f, -0.025f, 0.025f, 0.025f};

static float mcab_rand_unit(unsigned int* state) {
    /* Small deterministic PRNG (xorshift32) instead of rand()/srand(), so
     * the demo's synthetic signal does not depend on, or perturb, any
     * global PRNG state a test harness might also use. */
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return ((float)(x & 0xFFFFFFu) / (float)0xFFFFFFu) * 2.0f - 1.0f;
}

static double mcab_rms(const float* x, int n) {
    double acc = 0.0;
    int i;
    for (i = 0; i < n; ++i) acc += (double)x[i] * (double)x[i];
    return sqrt(acc / (double)n);
}

static double mcab_spectrum_rms(const Complex* x, int n_freqs) {
    double acc = 0.0;
    int f;
    for (f = 0; f < n_freqs; ++f) {
        double re = (double)x[f].r, im = (double)x[f].i;
        acc += re * re + im * im;
    }
    return sqrt(acc / (double)n_freqs);
}

int main(void) {
    AecConfig cfg;
    AecSignalGrid grid;
    Aec lanes[MCAB_NUM_LANES];
    int lane_created[MCAB_NUM_LANES] = {0};
    int m, h, k;
    int rc = 1;

    SRP_Config srp_cfg;
    BF_Config bf_cfg;
    ArrayGeometry geom;
    SRP* srp = NULL;
    BF* bf = NULL;

    float planted_doa_rad = MCAB_PLANTED_DOA_DEG * (float)M_PI / 180.0f;
    float cos_t = cosf(planted_doa_rad);
    float sin_t = sinf(planted_doa_rad);
    float tau[MCAB_NUM_LANES];

    float* mic_hop[MCAB_NUM_LANES];
    float* out_hop[MCAB_NUM_LANES];
    float* ref_hop = NULL;
    Complex** lane_spec = NULL; /* Complex* const* X handed to SRP/BF */
    Complex* bf_out = NULL;

    double pre_rms_acc[MCAB_NUM_LANES];
    double post_rms_acc[MCAB_NUM_LANES];
    double leadin_pre_rms_acc[MCAB_NUM_LANES];
    double leadin_post_rms_acc[MCAB_NUM_LANES];
    long steady_state_samples = 0;
    long leadin_steady_samples = 0; /* 2nd half of the lead-in only, once
                                        the matched filter has converged */
    long total_samples = 0;
    unsigned int noise_state = 0xC0FFEEu;

    double report_sin_sum = 0.0, report_cos_sum = 0.0, report_bf_rms_sum = 0.0;
    int report_count = 0;
    double final_doa_deg = NAN;
    double final_bf_rms = NAN;

    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        mic_hop[m] = NULL;
        out_hop[m] = NULL;
        pre_rms_acc[m] = 0.0;
        post_rms_acc[m] = 0.0;
        leadin_pre_rms_acc[m] = 0.0;
        leadin_post_rms_acc[m] = 0.0;
    }

    /* ---- 1. AEC lanes: aec_create() per mic, each fully independent --- */
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, MCAB_SAMPLE_RATE);
    cfg.enable_res = 0;          /* see top-of-file doc: context-only seam */
    cfg.return_res_context = 1;  /* -> error_spec populated, out == linear
                                     residual */
    if (!aec_resolve_signal_grid(MCAB_SAMPLE_RATE, cfg.fft_size, &grid)) {
        fprintf(stderr, "mcab_demo: FATAL: unresolved AEC signal grid\n");
        goto cleanup;
    }

    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        if (aec_create(&lanes[m], &cfg) != 0) {
            fprintf(stderr, "mcab_demo: FATAL: aec_create() lane %d rejected cfg\n", m);
            goto cleanup;
        }
        lane_created[m] = 1;
        tau[m] = -(MCAB_GEOM_X[m] * cos_t + MCAB_GEOM_Y[m] * sin_t) /
                 MCAB_SPEED_OF_SOUND_M_S;
    }

    /* ---- 2. SRP-PHAT DOA estimator, heap convenience constructor ------ */
    memset(&srp_cfg, 0, sizeof(srp_cfg));
    srp_cfg.M = MCAB_NUM_LANES;
    srp_cfg.F = grid.n_freqs;
    srp_cfg.num_angles = MCAB_NUM_ANGLES;
    srp_cfg.sr = (float)MCAB_SAMPLE_RATE;
    srp_cfg.NFFT = (float)grid.fft_size;
    srp_cfg.c = MCAB_SPEED_OF_SOUND_M_S;
    srp_cfg.low_freq = 300.0f;
    srp_cfg.high_freq = 3400.0f;
    srp_cfg.enable_smoothing = 1;
    srp_cfg.switch_consec = 5;     /* a new angle must win 5 consecutive
                                       hops before the held estimate moves */
    srp_cfg.angle_tol = 2.0f * (float)M_PI / MCAB_NUM_ANGLES * 2.0f;
                                    /* +/- 2 grid steps counts as "the same
                                       angle" for hold/switch purposes */
    srp_cfg.update_interval = 1;   /* run SRP every hop */

    geom.M = MCAB_NUM_LANES;
    geom.x = (float*)MCAB_GEOM_X;
    geom.y = (float*)MCAB_GEOM_Y;

    srp = srp_create_from_geometry(&srp_cfg, &geom);
    if (!srp) {
        fprintf(stderr, "mcab_demo: FATAL: srp_create_from_geometry() rejected cfg\n");
        goto cleanup;
    }

    /* ---- 3. Beamformer, steered by the SRP-estimated DOA --------------- */
    memset(&bf_cfg, 0, sizeof(bf_cfg));
    bf_cfg.enable = 1;
    bf_cfg.lambda = 0.995f;
    bf_cfg.mu = 0.1f;
    bf_cfg.enable_fix_mode = 0;      /* AUTO DOA: follow SRP's estimate,
                                         not a fixed, hand-fed angle */
    bf_cfg.fixed_doa_rad = 0.0f;
    bf_cfg.fixed_align_notebook = 0;
    bf_cfg.adapt_interval = 1;

    /* a_array is borrowed from the SRP instance, per bf.h's own doc. */
    bf = bf_create(MCAB_NUM_LANES, grid.n_freqs, MCAB_NUM_ANGLES,
                    srp->a_array, &bf_cfg);
    if (!bf) {
        fprintf(stderr, "mcab_demo: FATAL: bf_create() rejected cfg\n");
        goto cleanup;
    }

    /* ---- 4. Per-hop scratch -------------------------------------------- */
    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        mic_hop[m] = (float*)malloc((size_t)grid.hop_size * sizeof(float));
        out_hop[m] = (float*)malloc((size_t)grid.hop_size * sizeof(float));
        if (!mic_hop[m] || !out_hop[m]) {
            fprintf(stderr, "mcab_demo: FATAL: malloc failed\n");
            goto cleanup;
        }
    }
    ref_hop = (float*)malloc((size_t)grid.hop_size * sizeof(float));
    lane_spec = (Complex**)malloc((size_t)MCAB_NUM_LANES * sizeof(Complex*));
    bf_out = (Complex*)malloc((size_t)grid.n_freqs * sizeof(Complex));
    if (!ref_hop || !lane_spec || !bf_out) {
        fprintf(stderr, "mcab_demo: FATAL: malloc failed\n");
        goto cleanup;
    }

    printf("mcab_demo: %d lanes, sr=%d fft=%d hop=%d n_freqs=%d, "
           "planted DOA=%.1f deg\n",
           MCAB_NUM_LANES, grid.sample_rate, grid.fft_size, grid.hop_size,
           grid.n_freqs, (double)MCAB_PLANTED_DOA_DEG);

    /* ---- 5. Per-hop loop: synth -> AEC x N -> DOA -> beamform ---------- */
    for (h = 0; h < MCAB_NUM_HOPS; ++h) {
        for (k = 0; k < grid.hop_size; ++k) {
            long n = total_samples + k;
            float t = (float)n / (float)MCAB_SAMPLE_RATE;
            ref_hop[k] = MCAB_FAR_GAIN * sinf(2.0f * (float)M_PI *
                                               MCAB_FAR_FREQ_HZ * t);
        }

        for (m = 0; m < MCAB_NUM_LANES; ++m) {
            AecResContext ctx;

            for (k = 0; k < grid.hop_size; ++k) {
                long n = total_samples + k;
                float t = (float)n / (float)MCAB_SAMPLE_RATE;
                float near_s = (h >= MCAB_LEADIN_HOPS) ? MCAB_NEAR_GAIN *
                    sinf(2.0f * (float)M_PI * MCAB_NEAR_FREQ_HZ *
                         (t - tau[m])) : 0.0f;
                long echo_n = n - MCAB_ECHO_DELAY_SAMPLES[m];
                float echo_s = 0.0f;
                if (echo_n >= 0) {
                    float echo_t = (float)echo_n / (float)MCAB_SAMPLE_RATE;
                    echo_s = MCAB_ECHO_GAIN * MCAB_FAR_GAIN *
                        sinf(2.0f * (float)M_PI * MCAB_FAR_FREQ_HZ * echo_t);
                }
                mic_hop[m][k] = near_s + echo_s +
                    MCAB_NOISE_AMP * mcab_rand_unit(&noise_state);
            }

            /* aec_process(): lockstep time-domain AEC (see top-of-file doc
             * for why this, not aec_process_context(), was picked). */
            aec_process(&lanes[m], mic_hop[m], ref_hop, out_hop[m]);

            for (k = 0; k < grid.hop_size; ++k) {
                if (!isfinite(out_hop[m][k])) {
                    fprintf(stderr,
                            "mcab_demo: FATAL: non-finite AEC output, "
                            "lane %d hop %d sample %d\n", m, h, k);
                    goto cleanup;
                }
            }
            /* Pre/post RMS is measured only over the double-talk period
             * (after the lead-in): that is the steady-state regime once
             * the matched-filter delay estimator has already converged on
             * the echo path, and is the regime the reduction number below
             * is meant to describe. */
            if (h >= MCAB_LEADIN_HOPS) {
                double mic_rms = mcab_rms(mic_hop[m], grid.hop_size);
                double out_rms = mcab_rms(out_hop[m], grid.hop_size);
                pre_rms_acc[m] += mic_rms * mic_rms * (double)grid.hop_size;
                post_rms_acc[m] += out_rms * out_rms * (double)grid.hop_size;
            } else if (h >= MCAB_LEADIN_HOPS / 2) {
                /* 2nd half of the far-end-only (single-talk) lead-in --
                 * reported separately from the double-talk window below so
                 * the printed numbers are not an average across two very
                 * different regimes. */
                double mic_rms = mcab_rms(mic_hop[m], grid.hop_size);
                double out_rms = mcab_rms(out_hop[m], grid.hop_size);
                leadin_pre_rms_acc[m] +=
                    mic_rms * mic_rms * (double)grid.hop_size;
                leadin_post_rms_acc[m] +=
                    out_rms * out_rms * (double)grid.hop_size;
            }

            /* Same hop, same instance: read the context AEC_process() just
             * advanced, for the matching frequency-domain spectrum. */
            aec_get_res_context(&lanes[m], &ctx);
            if (!ctx.error_spec) {
                fprintf(stderr,
                        "mcab_demo: FATAL: lane %d error_spec is NULL "
                        "(enable_res/return_res_context misconfigured)\n", m);
                goto cleanup;
            }
            lane_spec[m] = (Complex*)ctx.error_spec;
        }

        total_samples += grid.hop_size;
        if (h >= MCAB_LEADIN_HOPS) {
            steady_state_samples += grid.hop_size;
        } else if (h >= MCAB_LEADIN_HOPS / 2) {
            leadin_steady_samples += grid.hop_size;
        }

        /* SRP/BF only run once the near-end (planted-DOA) source is
         * actually present -- there is nothing to localize or beamform
         * toward during the far-end-only lead-in, so this demo does not
         * waste cycles seeding the DOA smoother from pure noise/echo
         * residual. (A production caller that must localize from hop 0
         * would instead rely on a VAD gate -- doa_step()'s own vad_raw
         * argument -- rather than a fixed hop count; the fixed count here
         * is this demo's own synthetic-signal schedule, not a library
         * requirement.) */
        if (h >= MCAB_LEADIN_HOPS) {
            /* ---- DOA estimate from this hop's N post-AEC spectra ------- */
            doa_step(srp, (const Complex* const*)lane_spec, NULL, 1, 1);

            /* ---- Beamform toward the just-estimated DOA ----------------- */
            bf_process(bf, (const Complex* const*)lane_spec,
                       doa_get_smooth(srp), 1, NULL, bf_out);

            for (k = 0; k < grid.n_freqs; ++k) {
                if (!isfinite(bf_out[k].r) || !isfinite(bf_out[k].i)) {
                    fprintf(stderr,
                            "mcab_demo: FATAL: non-finite beamformer output, "
                            "hop %d bin %d\n", h, k);
                    goto cleanup;
                }
            }

            if (h == MCAB_NUM_HOPS - 1 || ((h - MCAB_LEADIN_HOPS) % 20) == 0) {
                printf("mcab_demo: hop %3d  doa_raw=%6.1f deg  doa_smooth=%6.1f "
                       "deg  bf_rms=%.6f\n",
                       h,
                       (double)(doa_get_raw(srp) * 180.0f / (float)M_PI),
                       (double)(doa_get_smooth(srp) * 180.0f / (float)M_PI),
                       mcab_spectrum_rms(bf_out, grid.n_freqs));
            }

            if (h >= MCAB_NUM_HOPS - MCAB_DOA_REPORT_WINDOW) {
                float doa_s = doa_get_smooth(srp);
                report_sin_sum += sinf(doa_s);
                report_cos_sum += cosf(doa_s);
                report_bf_rms_sum += mcab_spectrum_rms(bf_out, grid.n_freqs);
                report_count++;
            }
        }
    }

    /* Final DOA/BF-RMS is a circular mean over the last
     * MCAB_DOA_REPORT_WINDOW hops (all of them in steady state, well after
     * MCAB_LEADIN_HOPS), not the single last hop -- a single hop's raw
     * SRP-PHAT estimate is noisy even once the smoother has locked on. */
    final_doa_deg = (double)(atan2f((float)(report_sin_sum / report_count),
                                     (float)(report_cos_sum / report_count)) *
                              180.0f / (float)M_PI);
    if (final_doa_deg < 0.0) final_doa_deg += 360.0;
    final_bf_rms = report_bf_rms_sum / (double)report_count;

    printf("mcab_demo: ---- summary ----\n");
    printf("mcab_demo: planted DOA = %.1f deg, estimated DOA = %.1f deg "
           "(|diff| = %.1f deg, mean over last %d hops)\n",
           (double)MCAB_PLANTED_DOA_DEG, final_doa_deg,
           fabs(final_doa_deg - (double)MCAB_PLANTED_DOA_DEG),
           MCAB_DOA_REPORT_WINDOW);
    /* These two blocks are a factual pre/post-RMS measurement, not a
     * performance claim: this AEC lane's own near-end protection (see
     * aec.h's DominantNearend/shadow/EPC machinery) can and does hold
     * cancellation back once it judges the near-end to be dominant, so a
     * lane is not expected to show the same delta in both regimes -- the
     * second block (double-talk) is the regime the DOA/beamform numbers
     * above were actually computed from. */
    printf("mcab_demo: -- per-lane AEC pre/post RMS, far-end-only "
           "lead-in --\n");
    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        double pre = sqrt(leadin_pre_rms_acc[m] / (double)leadin_steady_samples);
        double post = sqrt(leadin_post_rms_acc[m] / (double)leadin_steady_samples);
        double delta_db = (post > 0.0) ? 20.0 * log10(pre / post) : 0.0;
        printf("mcab_demo: lane %d  pre_rms=%.6f  post_rms=%.6f  "
               "delta=%.2f dB\n", m, pre, post, delta_db);
    }
    printf("mcab_demo: -- per-lane AEC pre/post RMS, double-talk steady "
           "state (feeds SRP/BF above) --\n");
    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        double pre = sqrt(pre_rms_acc[m] / (double)steady_state_samples);
        double post = sqrt(post_rms_acc[m] / (double)steady_state_samples);
        double delta_db = (post > 0.0) ? 20.0 * log10(pre / post) : 0.0;
        printf("mcab_demo: lane %d  pre_rms=%.6f  post_rms=%.6f  "
               "delta=%.2f dB\n", m, pre, post, delta_db);
    }
    printf("mcab_demo: final beamformer output RMS = %.6f\n", final_bf_rms);

    if (!isfinite(final_doa_deg) || !isfinite(final_bf_rms) ||
        final_bf_rms <= 0.0) {
        fprintf(stderr,
                "mcab_demo: FATAL: degenerate final output (NaN or zero)\n");
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (bf) bf_destroy(bf);
    if (srp) srp_destroy(srp);
    for (m = 0; m < MCAB_NUM_LANES; ++m) {
        if (lane_created[m]) aec_destroy(&lanes[m]);
        free(mic_hop[m]);
        free(out_hop[m]);
    }
    free(ref_hop);
    free(lane_spec);
    free(bf_out);
    return rc;
}
