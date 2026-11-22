/* The stream latency through both families, against a bypassing stand-in accelerator
 * (run_status != 0: unit mask, alpha 0 -- the near-identity pass-through). A single 16000-count
 * sample is placed at input index PULSE_AT; the output carries it back at index PULSE_AT + latency, where
 * latency = (queue_frames + 1) * model_hop is the literal of the catalog row (ainr.h). The stream is
 * fed real blocks only.
 *
 *   PCM family   ainr_preprocess/ainr_postprocess, internal ISTFT.
 *   _freq        an upstream AcStft feeds E == P; an external AcIstft (ACCUMULATE, scale 1) synthesises.
 *
 * Covers full (16 kHz/256 N=1 and N=8, 48 kHz/1024 N=1) or lite (48 kHz/1024 N=1) depending on the
 * archive variant. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_test_fixture.h"
#include "audio_stft.h"

#define PULSE 16000
#define PULSE_AT 1030
#define BLOCKS 12

typedef struct Ola { AcIstft *istft; void *mem; } Ola;

static Ola ola_make(int fft) {
    AcStftConfig sc;
    Ola o;
    ac_stft_config_default(&sc, fft);
    o.mem = alloc_aligned(ac_istft_get_mem_size(&sc), 16);
    o.istft = ac_istft_init(o.mem, ac_istft_get_mem_size(&sc), &sc);
    if (!o.istft) { fprintf(stderr, "istft init fail\n"); exit(1); }
    return o;
}

/* Where the pulse comes back and how loud, for the stream out[0..n). Full scale float -> counts. */
static int check_pulse(const float *out, size_t n, size_t idx, const char *label, double *value_out) {
    size_t i, peak = 0;
    double v;
    if (idx >= n) { fprintf(stderr, "%s: pulse index %zu beyond the %zu produced samples\n", label, idx, n); return 1; }
    for (i = 0; i < n; ++i) if (fabsf(out[i]) > fabsf(out[peak])) peak = i;
    v = (double)out[idx] * 32768.0;
    if (value_out) *value_out = v;
    if (fabs(v - (double)PULSE) > 3.0 || peak != idx) {
        fprintf(stderr, "%s: latency off -- out[%zu] = %.1f (want %d +-3), loudest sample at %zu\n",
                label, idx, v, PULSE, peak);
        return 1;
    }
    return 0;
}

/* want_queue / want_latency are the literal values of the catalog row (L, D lookahead hops of its grid):
 * queue_frames = L + D (whatever N is), latency = (queue_frames + 1) * hop samples -- spelled out per row,
 * not derived from the library's own formula. */
static int run_case(int sr, int fft, int n_erb, int n, int want_queue, int want_latency, const char *label) {
    AinrConfig cfg = {0};
    StreamMemReq req;
    AinrInfo bi_pcm, bi_freq;
    int hop = fft / 2, n_bins = fft / 2 + 1, hop_in = n * hop;
    int b, f, k, bad = 0;
    size_t total = (size_t)BLOCKS * (size_t)hop_in, idx = (size_t)(PULSE_AT + want_latency);
    int16_t *pcm, *out16;
    float *outf, *fre, *fim, *are, *aim, *hopf;
    void *pool;
    AinrCtx *cp;
    AinrFreqCtx *cf;
    Ola ola;
    double v;

    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = hop_in;
    ErbTables erb = erb_tables_identity(n_bins, n_erb);

    /* PCM family: init reports the queue; the internal ISTFT synthesises. */
    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "%s: get_mem_size fail\n", label); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_t(&cfg, &erb, pool, (size_t)req.bytes, &cp, &bi_pcm) != AINR_OK || !cp) {
        fprintf(stderr, "%s: init fail\n", label); return 1;
    }
    {   /* the _freq family reports the same queue */
        StreamMemReq rf;
        void *pool_f;
        const AinrInfo *all[2];
        int fi;
        if (ainr_get_mem_size_freq(&cfg, &rf) != AINR_OK) {
            fprintf(stderr, "%s: freq size fail\n", label); return 1;
        }
        pool_f = alloc_aligned((size_t)rf.bytes, rf.alignment);
        if (ainr_init_freq_t(&cfg, &erb, pool_f, (size_t)rf.bytes, &cf, &bi_freq) != AINR_OK) {
            fprintf(stderr, "%s: freq init fail\n", label); return 1;
        }
        free(pool_f);
        all[0] = &bi_pcm; all[1] = &bi_freq;
        for (fi = 0; fi < 2; ++fi) {
            if (all[fi]->queue_frames != want_queue) {
                fprintf(stderr, "%s: family %d info queue %d, want %d\n", label, fi, all[fi]->queue_frames, want_queue);
                return 1;
            }
        }
    }
    if (idx >= total) { fprintf(stderr, "%s: pulse index %zu beyond the %zu produced samples\n", label, idx, total); return 1; }

    pcm = (int16_t *)calloc(total, sizeof(int16_t));
    out16 = (int16_t *)calloc(total, sizeof(int16_t));
    outf = (float *)calloc(total, sizeof(float));
    fre = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    fim = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    are = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    aim = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    hopf = (float *)malloc((size_t)hop * sizeof(float));
    pcm[PULSE_AT] = PULSE;

    for (b = 0; b < BLOCKS; ++b) {
        StreamWork io;
        if (ainr_preprocess(cp, pcm + (size_t)b * (size_t)hop_in, &io) != AINR_OK ||
            ainr_postprocess(cp, 1, out16 + (size_t)b * (size_t)hop_in) != AINR_OK) {
            fprintf(stderr, "%s: PCM family block %d failed\n", label, b); return 1;
        }
    }
    for (k = 0; (size_t)k < total; ++k) outf[k] = (float)out16[k] / 32768.0f;
    bad |= check_pulse(outf, total, idx, label, &v);
    if (!bad) printf("%s: PCM family latency OK, out[%zu] = %.1f\n", label, idx, v);
    free(pool);

    /* _freq family: the upstream analyser sees the same PCM. */
    {
        StreamMemReq rf;
        AcStftConfig sc;
        AcStft *st;
        void *stm;
        ac_stft_config_default(&sc, fft);
        stm = alloc_aligned(ac_stft_get_mem_size(&sc), 16);
        st = ac_stft_init(stm, ac_stft_get_mem_size(&sc), &sc);
        if (!st || ainr_get_mem_size_freq(&cfg, &rf) != AINR_OK) { fprintf(stderr, "%s: freq setup fail\n", label); return 1; }
        pool = alloc_aligned((size_t)rf.bytes, rf.alignment);
        if (ainr_init_freq_t(&cfg, &erb, pool, (size_t)rf.bytes, &cf, &bi_freq) != AINR_OK || !cf) {
            fprintf(stderr, "%s: freq init fail\n", label); return 1;
        }
        ola = ola_make(fft);
        memset(outf, 0, total * sizeof(float));
        for (b = 0; b < BLOCKS; ++b) {
            StreamWork io;
            for (f = 0; f < n; ++f) {
                for (k = 0; k < hop; ++k)
                    hopf[k] = (float)pcm[(size_t)b * (size_t)hop_in + (size_t)f * (size_t)hop + (size_t)k] / 32768.0f;
                ac_stft_push_planar(st, hopf, are + (size_t)f * (size_t)n_bins, aim + (size_t)f * (size_t)n_bins);
            }
            if (ainr_preprocess_freq(cf, are, aim, are, aim, &io) != AINR_OK ||
                ainr_postprocess_freq(cf, 1, fre, fim) != AINR_OK) { fprintf(stderr, "%s: freq block %d failed\n", label, b); return 1; }
            for (f = 0; f < n; ++f)
                ac_istft_push_planar(ola.istft, fre + (size_t)f * (size_t)n_bins, fim + (size_t)f * (size_t)n_bins,
                                     outf + ((size_t)b * (size_t)n + (size_t)f) * (size_t)hop);
        }
        bad |= check_pulse(outf, total, idx, label, &v);
        if (!bad) printf("%s: _freq (upstream analyser + external ISTFT) latency OK, out[%zu] = %.1f\n", label, idx, v);
        free(pool); free(ola.mem); free(stm);
    }

    erb_tables_free(&erb);
    free(pcm); free(out16); free(outf); free(fre); free(fim); free(are); free(aim); free(hopf);
    return bad;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, 1, 1024, "latency[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 0, 128, "latency[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 0, 128, "latency[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 2, 1536, "latency[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_latency: FAIL\n"); return 1; }
    printf("test_ainr_latency: PASS\n");
    return 0;
}
