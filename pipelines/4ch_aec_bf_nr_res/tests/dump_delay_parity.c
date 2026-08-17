/**
 * dump_delay_parity.c — per-hop shared-alignment CSV for the C/Python
 * reference parity test (tests/test_delay_parity.py).
 *
 * Drives one deterministic synthetic scene (xorshift32 far noise plus a
 * half-amplitude echo at a known bulk delay) through the real 4ch core and
 * prints, for every hop, the published shared-delay state and a hash of the
 * aligned far the four lanes actually consumed. pipeline.py generates the
 * SAME scene and must reproduce every column, so any divergence between the
 * C core's acquisition/ring-fill rules and the Python reference's shows up
 * as a per-hop mismatch instead of as an audio-quality drift nobody bisects.
 *
 * printf lives in THIS test binary only -- 4aec_nr_res.c and everything it
 * links stay stdio-free (see the Makefile's audit-no-stdio target).
 *
 * Build:
 *   make -C pipelines/4ch_aec_bf_nr_res dump_delay_parity
 *
 * Usage:
 *   ./dump_delay_parity [--mode matched|fixed] [--delay SAMPLES]
 *                       [--shift-at HOP --shift-delay SAMPLES]
 *                       [--hops N] [--sample-rate 16000|48000]
 *                       [--fft-size 256|512|1024] [--seed HEX]
 *                       [--backward-quarantine] [--quarantine-s SECONDS]
 *                       [--proxy-noise GAIN]
 *
 * --seed reseeds the far/echo stream ONLY; the --proxy-noise stream keeps its
 * own fixed seed, so a reseeded run still gets the same proxy interference.
 *
 * --backward-quarantine turns on
 * FourAecNrResConfig::delay_backward_quarantine_enabled (default off, as in
 * production) so the parity test can drive the quarantined acceptance rule as
 * well as the unquarantined one -- the quarantine reads live lane state and
 * carries its own countdown, so a mirror that only agreed with it OFF would
 * leave the whole rule uncompared. --quarantine-s overrides the window
 * (default 1.0 s), which is what makes the countdown itself comparable rather
 * than only its enabled/disabled ends.
 *
 * --proxy-noise adds an independent noise stream to the capture-proxy channel
 * ONLY, leaving the other three microphones on the clean echo. That asymmetry
 * is the only way to tell "judge the estimator's own lane" apart from "judge
 * any lane": with it, the proxy lane cannot cancel while the other three can,
 * so the two rules disagree about whether to quarantine. Without it every
 * channel carries identical audio and the two rules are indistinguishable.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "4aec_nr_res.h"

/* xorshift32 -> [-0.125, 0.125). Every operation is exactly representable in
 * float32 (the 24-bit mantissa feed is exact, the two scale factors are
 * powers of two), so pipeline.py's numpy mirror of this generator is
 * bit-identical rather than merely close. */
static float noise_next(uint32_t* state) {
    uint32_t s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    return 0.25f * (((float)(s >> 8) * (1.0f / 16777216.0f)) - 0.5f);
}

/* The far/echo stream, and a second independent one for --proxy-noise: same
 * generator, a different fixed seed, so the two never correlate and the
 * Python mirror reproduces both bit-for-bit. */
static uint32_t g_rng = 0x1234567u;
static uint32_t g_rng_proxy = 0x89ABCDEFu;

/* FNV-1a over the raw sample bytes: byte equality of a whole hop, reported
 * in one CSV column. */
static uint64_t hash_hop(const float* samples, int count) {
    const unsigned char* bytes = (const unsigned char*)samples;
    size_t n = (size_t)count * sizeof(float);
    uint64_t h = 1469598103934665603ULL;
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= (uint64_t)bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    FourAecNrResConfig cfg;
    FourAecNrRes* p = NULL;
    float* far_hist = NULL;
    float* proxy_hist = NULL;
    float* mic = NULL;
    float* far = NULL;
    const char* mode = "matched";
    int sample_rate = 16000;
    int fft_size = 512;
    int delay = 2000;
    int shift_delay = -1;
    int shift_at = -1;
    int hops = 300;
    int quarantine = 0;
    float quarantine_s = -1.0f;
    float proxy_noise = 0.0f;
    int hop, total, pad;
    int h, i, ch;
    int status = 0;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            delay = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shift-at") == 0 && i + 1 < argc) {
            shift_at = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shift-delay") == 0 && i + 1 < argc) {
            shift_delay = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hops") == 0 && i + 1 < argc) {
            hops = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sample-rate") == 0 && i + 1 < argc) {
            sample_rate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fft-size") == 0 && i + 1 < argc) {
            fft_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backward-quarantine") == 0) {
            quarantine = 1;
        } else if (strcmp(argv[i], "--quarantine-s") == 0 && i + 1 < argc) {
            quarantine_s = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--proxy-noise") == 0 && i + 1 < argc) {
            proxy_noise = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_rng = (uint32_t)strtoul(argv[++i], NULL, 0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (delay < 0 || hops <= 0 || g_rng == 0) {
        fprintf(stderr, "invalid --delay/--hops/--seed\n");
        return 2;
    }
    if ((shift_at >= 0) != (shift_delay >= 0)) {
        fprintf(stderr, "--shift-at and --shift-delay come as a pair\n");
        return 2;
    }
    if (shift_delay < 0) shift_delay = delay;

    cfg = four_aec_nr_res_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.enable_cng = 0;
    cfg.enable_post = 0;          /* the alignment seam lives in process_pre */
    cfg.delay_backward_quarantine_enabled = quarantine;
    if (quarantine_s >= 0.0f) cfg.delay_backward_quarantine_s = quarantine_s;
    if (strcmp(mode, "fixed") == 0) {
        cfg.delay_mode = AEC_DELAY_FIXED;
        cfg.fixed_delay_samples = delay;
    } else if (strcmp(mode, "matched") != 0) {
        fprintf(stderr, "unknown --mode: %s\n", mode);
        return 2;
    }

    p = four_aec_nr_res_create(&cfg);
    if (!p) {
        fprintf(stderr, "four_aec_nr_res_create failed\n");
        return 3;
    }
    hop = four_aec_nr_res_hop_size(p);

    /* `pad` samples of pre-history so the echo is valid from the first
     * streamed sample, at either bulk delay. */
    pad = delay > shift_delay ? delay : shift_delay;
    total = pad + hops * hop;
    far_hist = (float*)malloc((size_t)total * sizeof(float));
    mic = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS *
                         sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    if (!far_hist || !mic || !far) {
        fprintf(stderr, "scene allocation failed\n");
        status = 4;
        goto cleanup;
    }
    for (i = 0; i < total; ++i) far_hist[i] = noise_next(&g_rng);
    if (proxy_noise != 0.0f) {
        proxy_hist = (float*)malloc((size_t)total * sizeof(float));
        if (!proxy_hist) {
            fprintf(stderr, "proxy noise allocation failed\n");
            status = 4;
            goto cleanup;
        }
        for (i = 0; i < total; ++i)
            proxy_hist[i] = proxy_noise * noise_next(&g_rng_proxy);
    }

    printf("# hop_size=%d mode=%s delay=%d hops=%d shift_at=%d shift_delay=%d"
           " quarantine=%d quarantine_s=%.4f proxy_noise=%.4f\n",
           hop, mode, delay, hops, shift_at, shift_delay, quarantine,
           (double)cfg.delay_backward_quarantine_s, (double)proxy_noise);
    printf("hop,delay_samples,solid,changed,far_hash,aligned_hash,"
           "aligned_first\n");
    for (h = 0; h < hops; ++h) {
        FourAecNrResPreFrame pre;
        int base = pad + h * hop;
        int echo_delay = (shift_at >= 0 && h >= shift_at) ? shift_delay : delay;
        for (i = 0; i < hop; ++i) {
            float echo = 0.5f * far_hist[base + i - echo_delay];
            far[i] = far_hist[base + i];
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = echo;
            if (proxy_hist)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + cfg.capture_proxy_channel] +=
                    proxy_hist[base + i];
        }
        if (four_aec_nr_res_process_pre(p, mic, far, &pre) !=
            FOUR_AEC_NR_RES_OK) {
            fprintf(stderr, "process_pre failed at hop %d\n", h);
            status = 5;
            goto cleanup;
        }
        printf("%d,%d,%d,%d,%016llx,%016llx,%a\n",
               h, pre.delay.delay_samples, pre.delay.solid ? 1 : 0,
               pre.delay.changed ? 1 : 0,
               (unsigned long long)hash_hop(far, hop),
               (unsigned long long)hash_hop(pre.aligned_ref, hop),
               (double)pre.aligned_ref[0]);
        if (four_aec_nr_res_abandon_pre(p, &pre.token) !=
            FOUR_AEC_NR_RES_OK) {
            fprintf(stderr, "abandon_pre failed at hop %d\n", h);
            status = 6;
            goto cleanup;
        }
    }

cleanup:
    free(far_hist);
    free(proxy_hist);
    free(mic);
    free(far);
    four_aec_nr_res_destroy(p);
    return status;
}
