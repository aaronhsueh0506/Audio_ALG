/* The end-of-input tail through all three families, against a bypassing stand-in accelerator
 * (infer_status != 0: unit mask, alpha 0 -- the near-identity pass-through). A single 16000-count
 * sample is placed at the very last real input sample; the test reads it back at output index
 * (real length - 1 + trim_samples) after the documented drain (ainr.h "Tail"): tail_zero_blocks
 * all-zero hop_in blocks after the last real block.
 *
 *   PCM family   ainr_preprocess/ainr_postprocess, internal ISTFT.
 *   _spec        ainr_preprocess_spec/ainr_postprocess_spec, external AcIstft (ACCUMULATE, scale 1).
 *   _freq        an upstream AcStft feeds E == P; same external AcIstft.
 *
 * It also reproduces the failing procedure -- draining only queue_frames zero hops of PCM
 * and then one zero SPECTRUM -- and asserts that it loses the pulse. Covers full (16 kHz/256 N=1
 * and N=8, 48 kHz/1024 N=1) or lite (48 kHz/1024 N=1) depending on the archive variant. Not built
 * under NO_STDIO. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "audio_stft.h"

#define PULSE 16000
#define BLOCKS_REAL 4

/* ERB tables that make the bypass an exact identity: every bin belongs to one contiguous band
 * (forward = band mean, inverse = copy the band's gain back), so a unit mask is a unit gain. */
static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb;
    float *fwd = (float *)calloc(elems, sizeof(float));
    float *inv = (float *)calloc(elems, sizeof(float));
    int *size = (int *)calloc((size_t)n_erb, sizeof(int));
    int32_t hdr[2];
    FILE *fp;
    int k, e;
    if (!fwd || !inv || !size) { fprintf(stderr, "calloc fail\n"); exit(1); }
    for (k = 0; k < n_bins; ++k) { e = (int)((long long)k * n_erb / n_bins); size[e]++; }
    for (k = 0; k < n_bins; ++k) {
        e = (int)((long long)k * n_erb / n_bins);
        fwd[(size_t)k * (size_t)n_erb + (size_t)e] = 1.0f / (float)size[e];
        inv[(size_t)k * (size_t)n_erb + (size_t)e] = 1.0f;
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins; hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd); free(inv); free(size);
}

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

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
        fprintf(stderr, "%s: tail lost -- out[%zu] = %.1f (want %d +-3), loudest sample at %zu\n",
                label, idx, v, PULSE, peak);
        return 1;
    }
    return 0;
}

/* want_queue / want_trim / want_tail are the literal values of the catalog row (L, D lookahead hops of
 * its grid): queue_frames = L + D + N - 1, trim_samples = (queue_frames + 1) * hop, tail_zero_blocks =
 * ceil(trim_samples / hop_in) -- spelled out per row, not derived from the library's own formula. */
static int run_case(int sr, int fft, int n_erb, int n, int want_queue, int want_trim, int want_tail,
                    const char *label) {
    AinrConfig cfg;
    StreamMemReq req;
    AinrInfo bi_pcm, bi_spec, bi_freq;
    int hop = fft / 2, n_bins = fft / 2 + 1, hop_in = n * hop;
    int tail, total_blocks, b, f, k, bad = 0;
    size_t s_real = (size_t)BLOCKS_REAL * (size_t)hop_in, total = 0, idx;
    int16_t *pcm, *out16;
    float *outf, *fre, *fim, *are, *aim, *hopf;
    char erb[160];
    void *pool;
    AinrCtx *cp;
    AinrSpecCtx *cs;
    AinrFreqCtx *cf;
    Ola ola;
    double v;

    cfg.sr = sr; cfg.fft = fft; cfg.hop_in = hop_in;
    snprintf(erb, sizeof(erb), "/tmp/ainr_tail_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb, n_bins, n_erb);

    /* PCM family: init reports the lengths; the internal ISTFT drains with the documented zero blocks. */
    if (ainr_get_mem_size(&cfg, &req) != AINR_OK) { fprintf(stderr, "%s: get_mem_size fail\n", label); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    if (ainr_init_from_erb_file(&cfg, erb, pool, (size_t)req.bytes, &cp, &bi_pcm) != AINR_OK || !cp) {
        fprintf(stderr, "%s: init fail\n", label); return 1;
    }
    tail = bi_pcm.tail_zero_blocks;
    {   /* the other two families report the same lengths */
        StreamMemReq rs, rf;
        void *pool_s, *pool_f;
        const AinrInfo *all[3];
        int fi;
        if (ainr_get_mem_size_spec(&cfg, &rs) != AINR_OK || ainr_get_mem_size_freq(&cfg, &rf) != AINR_OK) {
            fprintf(stderr, "%s: spec/freq size fail\n", label); return 1;
        }
        pool_s = alloc_aligned((size_t)rs.bytes, rs.alignment);
        pool_f = alloc_aligned((size_t)rf.bytes, rf.alignment);
        if (ainr_init_spec_from_erb_file(&cfg, erb, pool_s, (size_t)rs.bytes, &cs, &bi_spec) != AINR_OK ||
            ainr_init_freq_from_erb_file(&cfg, erb, pool_f, (size_t)rf.bytes, &cf, &bi_freq) != AINR_OK) {
            fprintf(stderr, "%s: spec/freq init fail\n", label); return 1;
        }
        free(pool_s); free(pool_f);
        all[0] = &bi_pcm; all[1] = &bi_spec; all[2] = &bi_freq;
        for (fi = 0; fi < 3; ++fi) {
            if (all[fi]->queue_frames != want_queue || all[fi]->trim_samples != want_trim ||
                all[fi]->tail_zero_blocks != want_tail) {
                fprintf(stderr, "%s: family %d info queue %d trim %d tail %d, want %d / %d / %d\n", label, fi,
                        all[fi]->queue_frames, all[fi]->trim_samples, all[fi]->tail_zero_blocks,
                        want_queue, want_trim, want_tail);
                return 1;
            }
        }
    }
    total_blocks = BLOCKS_REAL + tail;
    total = (size_t)total_blocks * (size_t)hop_in;
    idx = s_real - 1 + (size_t)bi_pcm.trim_samples;

    pcm = (int16_t *)calloc(total, sizeof(int16_t));
    out16 = (int16_t *)calloc(total, sizeof(int16_t));
    outf = (float *)calloc(total, sizeof(float));
    fre = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    fim = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    are = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    aim = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    hopf = (float *)malloc((size_t)hop * sizeof(float));
    pcm[s_real - 1] = PULSE;

    for (b = 0; b < total_blocks; ++b) {
        AinrWork io;
        if (ainr_preprocess(cp, pcm + (size_t)b * (size_t)hop_in, &io) != AINR_OK ||
            ainr_postprocess(cp, 1, out16 + (size_t)b * (size_t)hop_in) != AINR_OK) {
            fprintf(stderr, "%s: PCM family block %d failed\n", label, b); return 1;
        }
    }
    for (k = 0; (size_t)k < total; ++k) outf[k] = (float)out16[k] / 32768.0f;
    bad |= check_pulse(outf, total, idx, label, &v);
    if (!bad) printf("%s: PCM family tail OK, out[%zu] = %.1f (tail blocks %d)\n", label, idx, v, tail);
    free(pool);

    /* _spec family + external ISTFT. */
    {
        StreamMemReq rs;
        int rc;
        if (ainr_get_mem_size_spec(&cfg, &rs) != AINR_OK) { fprintf(stderr, "%s: spec size fail\n", label); return 1; }
        pool = alloc_aligned((size_t)rs.bytes, rs.alignment);
        if (ainr_init_spec_from_erb_file(&cfg, erb, pool, (size_t)rs.bytes, &cs, &bi_spec) != AINR_OK || !cs) {
            fprintf(stderr, "%s: spec init fail\n", label); return 1;
        }
        ola = ola_make(fft);
        memset(outf, 0, total * sizeof(float));
        for (b = 0; b < total_blocks; ++b) {
            AinrWork io;
            rc = ainr_preprocess_spec(cs, pcm + (size_t)b * (size_t)hop_in, &io);
            if (rc != AINR_OK) { fprintf(stderr, "%s: spec pre block %d rc=%d\n", label, b, rc); return 1; }
            rc = ainr_postprocess_spec(cs, 1, fre, fim);
            if (rc != AINR_OK) { fprintf(stderr, "%s: spec post block %d rc=%d\n", label, b, rc); return 1; }
            for (f = 0; f < n; ++f)
                ac_istft_push_planar(ola.istft, fre + (size_t)f * (size_t)n_bins, fim + (size_t)f * (size_t)n_bins,
                                     outf + ((size_t)b * (size_t)n + (size_t)f) * (size_t)hop);
        }
        bad |= check_pulse(outf, total, idx, label, &v);
        if (!bad) printf("%s: _spec + external ISTFT tail OK, out[%zu] = %.1f\n", label, idx, v);
        free(pool); free(ola.mem);
    }

    /* The failing procedure (N=1): only the reported _spec delay in zero PCM hops, then one zero
     * SPECTRUM to the external ISTFT -- the last real hop is never analysed. */
    if (n == 1) {
        StreamMemReq rs;
        int zero_hops = bi_spec.queue_frames;
        float *zre = (float *)calloc((size_t)n_bins, sizeof(float));
        float *zim = (float *)calloc((size_t)n_bins, sizeof(float));
        if (ainr_get_mem_size_spec(&cfg, &rs) != AINR_OK) return 1;
        pool = alloc_aligned((size_t)rs.bytes, rs.alignment);
        if (ainr_init_spec_from_erb_file(&cfg, erb, pool, (size_t)rs.bytes, &cs, &bi_spec) != AINR_OK || !cs) return 1;
        ola = ola_make(fft);
        memset(outf, 0, total * sizeof(float));
        for (b = 0; b < BLOCKS_REAL + zero_hops; ++b) {
            AinrWork io;
            if (ainr_preprocess_spec(cs, pcm + (size_t)b * (size_t)hop_in, &io) != AINR_OK ||
                ainr_postprocess_spec(cs, 1, fre, fim) != AINR_OK) return 1;
            ac_istft_push_planar(ola.istft, fre, fim, outf + (size_t)b * (size_t)hop);
        }
        ac_istft_push_planar(ola.istft, zre, zim, outf + (size_t)b * (size_t)hop);
        v = (double)outf[idx] * 32768.0;
        if (fabs(v - (double)PULSE) < 0.5 * PULSE) {
            fprintf(stderr, "%s: the reported-delay-only drain unexpectedly kept the pulse (%.1f)\n", label, v);
            bad = 1;
        } else {
            printf("%s: reported-delay drain + zero spectrum loses the tail: out[%zu] = %.1f (want %d)\n",
                   label, idx, v, PULSE);
        }
        free(pool); free(ola.mem); free(zre); free(zim);
    }

    /* _freq family: the upstream analyser sees the same zero blocks of PCM. */
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
        if (ainr_init_freq_from_erb_file(&cfg, erb, pool, (size_t)rf.bytes, &cf, &bi_freq) != AINR_OK || !cf) {
            fprintf(stderr, "%s: freq init fail\n", label); return 1;
        }
        ola = ola_make(fft);
        memset(outf, 0, total * sizeof(float));
        for (b = 0; b < total_blocks; ++b) {
            AinrWork io;
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
        if (!bad) printf("%s: _freq (upstream analyser + external ISTFT) tail OK, out[%zu] = %.1f\n", label, idx, v);
        free(pool); free(ola.mem); free(stm);
    }

    unlink(erb);
    free(pcm); free(out16); free(outf); free(fre); free(fim); free(are); free(aim); free(hopf);
    return bad;
}

int main(void) {
    int rc = 0;
#ifdef DFN2_MASK_ONLY
    rc |= run_case(48000, 1024, 32, 1, 1, 1024, 2, "tail[lite 48k/1024 N=1]");
#else
    rc |= run_case(16000, 256, 24, 1, 0, 128, 1, "tail[full 16k/256 N=1]");
    rc |= run_case(16000, 256, 24, 8, 7, 1024, 1, "tail[full 16k/256 N=8]");
    rc |= run_case(48000, 1024, 32, 1, 2, 1536, 3, "tail[full 48k/1024 N=1]");
#endif
    if (rc) { printf("test_ainr_tail: FAIL\n"); return 1; }
    printf("test_ainr_tail: PASS\n");
    return 0;
}
