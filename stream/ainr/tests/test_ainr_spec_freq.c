/* Host-only regression suite for ainr.h/ainr.c's two additive, spectrum-delivering families
 * ("_spec": PCM in, enhanced spectrum out; "_freq": spectrum in, enhanced spectrum out), against
 * a stand-in accelerator (no NPU) -- same convention as tests/test_ainr_dfn2.c. Covers:
 *
 *   1. N=1, both shipped catalog grids (16kHz/256, 48kHz/1024): ainr_preprocess()/
 *      ainr_postprocess() (PCM family) vs ainr_preprocess_spec()/ainr_postprocess_spec() (this
 *      test does its own ac_istft_push_planar, default scale 1.0, AC_OLA_ACCUMULATE) on the same
 *      PCM and the same fake-accelerator fill, AND ainr_preprocess_freq()/ainr_postprocess_freq()
 *      fed the degenerate E == P case (an independent, default-scale AcStft analysing the same
 *      PCM) against the _spec family's own enhanced spectrum at the same hop -- both with a
 *      forced bypass hop (infer_status != 0) mixed into the run.
 *   2. N > 1 stash/ordering through the public _spec/_freq API on the catalog's 16 kHz/256 N=8 row:
 *      a committed block followed by a bypassed one, and _freq (E == P) against _spec bit-exact.
 *   3. Trim / queue labeling: AinrInfo.queue_frames and trim_samples of all three families against
 *      the lookahead hops of the grid, spelled out per row.
 *
 * Not built under NO_STDIO (ainr.c's only entry points, including the new families', are
 * file-mode only). MASK_ONLY-only concepts (coefs/alpha outputs) are already guarded by ainr.h's
 * own AinrWork contract (output_count differs), so this file needs no #ifdef DFN2_MASK_ONLY of
 * its own beyond what fill_accel() uses, matching test_ainr_dfn2.c's own convention. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "audio_stft.h"
#include "stream_mem.h"

/* ---- synthetic ERB file, same convention as tests/test_ainr_dfn2.c ---- */

static void write_synthetic_erb(const char *path, int n_bins, int n_erb) {
    size_t elems = (size_t)n_bins * (size_t)n_erb;
    float *fwd = (float *)malloc(elems * sizeof(float));
    float *inv_bin_major = (float *)malloc(elems * sizeof(float));
    size_t i;
    int32_t hdr[2];
    FILE *fp;

    if (!fwd || !inv_bin_major) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i = 0; i < elems; ++i) {
        fwd[i] = (float)(i % 17) * 0.01f;
        inv_bin_major[i] = (float)((i + 3) % 11) * 0.02f;
    }
    fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "fopen write fail (%s)\n", path); exit(1); }
    hdr[0] = (int32_t)n_bins;
    hdr[1] = (int32_t)n_erb;
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(fwd, sizeof(float), elems, fp);
    fwrite(inv_bin_major, sizeof(float), elems, fp);
    fclose(fp);
    free(fwd);
    free(inv_bin_major);
}

static AinrInfo g_info;

static void *alloc_aligned(size_t bytes, unsigned int alignment) {
    void *p = NULL;
    if (posix_memalign(&p, (size_t)alignment, bytes) != 0) { fprintf(stderr, "posix_memalign fail\n"); exit(1); }
    return p;
}

static void fill_pcm(int16_t *pcm, int n, int seed) {
    int i;
    for (i = 0; i < n; ++i) pcm[i] = (int16_t)((i * 13 + seed * 7) % 211 - 100);
}

/* Deterministic, non-constant fake accelerator fill: varies with hop_idx and element index, so an
 * indexing bug (wrong frame, wrong bin, cross-family data confusion) shows up as a mismatch
 * instead of being hidden behind constant values. Operates on AinrWork directly -- the same
 * tensor binding regardless of family (ainr.h's own documented contract), so one function serves
 * the PCM, _spec and _freq paths alike. */
static void fill_accel(AinrWork *io, int hop_idx) {
    size_t i;
    for (i = 0; i < io->outputs[0].elements; ++i)
        io->outputs[0].data[i] = 0.1f + 0.01f * (float)((i + (size_t)hop_idx) % 7);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < io->outputs[1].elements; ++i)
        io->outputs[1].data[i] = 0.02f * (float)((i + (size_t)hop_idx) % 5) - 0.05f;
    for (i = 0; i < io->outputs[2].elements; ++i)
        io->outputs[2].data[i] = 0.3f;
#endif
}

/* ---- Test 1: N=1 catalog grid -- PCM vs _spec (own ISTFT), and _spec vs _freq (E == P). ---- */

static int run_family_consistency(int sr, int fft, int n_erb, int hops,
                                   int bypass_hop, const char *label) {
    AinrConfig cfg;
    StreamMemReq req_pcm, req_spec, req_freq;
    void *pool_pcm, *pool_spec, *pool_freq;
    AinrCtx *ctx_pcm;
    AinrSpecCtx *ctx_spec;
    AinrFreqCtx *ctx_freq;
    char erb_path[160];
    int hop = fft / 2, n_bins = fft / 2 + 1;
    int16_t *pcm_in, *pcm_ref, *pcm_from_spec_i16;
    float *spec_re, *spec_im, *freq_re, *freq_im, *af_re, *af_im, *hop_f, *own_pcm_out;
    AcStftConfig scfg;
    AcStft *test_stft;
    AcIstft *test_istft;
    void *stft_mem, *istft_mem;
    int i, k, bad = 0;

    cfg.sr = sr;
    cfg.fft = fft;
    cfg.hop_in = hop;   /* N = 1 */

    if (ainr_get_mem_size(&cfg, &req_pcm) != AINR_OK) { fprintf(stderr, "%s: get_mem_size fail\n", label); return 1; }
    if (ainr_get_mem_size_spec(&cfg, &req_spec) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_spec fail\n", label); return 1; }
    if (ainr_get_mem_size_freq(&cfg, &req_freq) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_freq fail\n", label); return 1; }

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb_path, n_bins, n_erb);

    pool_pcm = alloc_aligned((size_t)req_pcm.bytes, req_pcm.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool_pcm, (size_t)req_pcm.bytes, &ctx_pcm, &g_info);
    pool_spec = alloc_aligned((size_t)req_spec.bytes, req_spec.alignment);
    ainr_init_spec_from_erb_file(&cfg, erb_path, pool_spec, (size_t)req_spec.bytes, &ctx_spec, &g_info);
    pool_freq = alloc_aligned((size_t)req_freq.bytes, req_freq.alignment);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_freq, (size_t)req_freq.bytes, &ctx_freq, &g_info);
    if (!ctx_pcm || !ctx_spec || !ctx_freq) { fprintf(stderr, "%s: init fail\n", label); return 1; }

    /* Test-local AcStft/AcIstft, both DEFAULT scale 1.0 (audio scale throughout) -- the config a
     * real caller of ainr_preprocess_spec()/ainr_postprocess_spec() is documented to synthesize
     * with, and the audio-scale analysis this test independently computes to feed
     * ainr_preprocess_freq()'s E/P arguments. */
    ac_stft_config_default(&scfg, fft);
    stft_mem = alloc_aligned(ac_stft_get_mem_size(&scfg), 16);
    test_stft = ac_stft_init(stft_mem, ac_stft_get_mem_size(&scfg), &scfg);
    istft_mem = alloc_aligned(ac_istft_get_mem_size(&scfg), 16);
    test_istft = ac_istft_init(istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
    if (!test_stft || !test_istft) { fprintf(stderr, "%s: test stft/istft init fail\n", label); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    pcm_ref = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    pcm_from_spec_i16 = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    spec_re = (float *)malloc((size_t)n_bins * sizeof(float));
    spec_im = (float *)malloc((size_t)n_bins * sizeof(float));
    freq_re = (float *)malloc((size_t)n_bins * sizeof(float));
    freq_im = (float *)malloc((size_t)n_bins * sizeof(float));
    af_re = (float *)malloc((size_t)n_bins * sizeof(float));
    af_im = (float *)malloc((size_t)n_bins * sizeof(float));
    hop_f = (float *)malloc((size_t)hop * sizeof(float));
    own_pcm_out = (float *)malloc((size_t)hop * sizeof(float));
    fill_pcm(pcm_in, hops * hop, 7);

    for (i = 0; i < hops && !bad; ++i) {
        AinrWork io_pcm, io_spec, io_freq;
        int infer_status = (i == bypass_hop) ? -1 : 0;
        int rc;

        /* PCM family (reference). */
        rc = ainr_preprocess(ctx_pcm, pcm_in + (size_t)i * (size_t)hop, &io_pcm);
        if (rc != 0) { fprintf(stderr, "%s: pcm preprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        fill_accel(&io_pcm, i);
        rc = ainr_postprocess(ctx_pcm, infer_status, pcm_ref + (size_t)i * (size_t)hop);
        if (rc != AINR_OK) { fprintf(stderr, "%s: pcm postprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }

        /* 1a: _spec family, same PCM in, same fill -- then this test's own ISTFT. */
        rc = ainr_preprocess_spec(ctx_spec, pcm_in + (size_t)i * (size_t)hop, &io_spec);
        if (rc != 0) { fprintf(stderr, "%s: spec preprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        fill_accel(&io_spec, i);
        rc = ainr_postprocess_spec(ctx_spec, infer_status, spec_re, spec_im);
        if (rc != AINR_OK) { fprintf(stderr, "%s: spec postprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        ac_istft_push_planar(test_istft, spec_re, spec_im, own_pcm_out);
        for (k = 0; k < hop; ++k)
            pcm_from_spec_i16[(size_t)i * (size_t)hop + k] = stream_saturate_s16(own_pcm_out[k]);

        /* 1b: _freq family, E == P from an independent default-scale AcStft fed the SAME PCM hop
         * -- must reproduce _spec's own enhanced spectrum for this hop exactly. */
        for (k = 0; k < hop; ++k) hop_f[k] = (float)pcm_in[(size_t)i * (size_t)hop + k] / 32768.0f;
        ac_stft_push_planar(test_stft, hop_f, af_re, af_im);

        rc = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io_freq);
        if (rc != 0) { fprintf(stderr, "%s: freq preprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        fill_accel(&io_freq, i);
        rc = ainr_postprocess_freq(ctx_freq, infer_status, freq_re, freq_im);
        if (rc != AINR_OK) { fprintf(stderr, "%s: freq postprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }

        for (k = 0; k < n_bins; ++k) {
            if (freq_re[k] != spec_re[k] || freq_im[k] != spec_im[k]) {
                fprintf(stderr, "%s: 1b mismatch at hop %d bin %d: freq=(%.9g,%.9g) spec=(%.9g,%.9g)\n",
                       label, i, k, (double)freq_re[k], (double)freq_im[k],
                       (double)spec_re[k], (double)spec_im[k]);
                bad = 1;
                break;
            }
        }
    }

    if (!bad && memcmp(pcm_ref, pcm_from_spec_i16, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "%s: 1a PCM-vs-_spec(own ISTFT) PCM mismatch\n", label);
        bad = 1;
    }

    /* Success is exactly 0, every refusal a negative AINR_ERR_*: a second preprocess while a block is
     * still open is refused in all three families. */
    if (!bad) {
        AinrWork io;
        int r1 = ainr_preprocess(ctx_pcm, pcm_in, &io), r2 = ainr_preprocess(ctx_pcm, pcm_in, &io);
        int s1 = ainr_preprocess_spec(ctx_spec, pcm_in, &io), s2 = ainr_preprocess_spec(ctx_spec, pcm_in, &io);
        int f1 = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io);
        int f2 = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io);
        if (r1 != 0 || s1 != 0 || f1 != 0 || r2 != AINR_ERR_SEQUENCE || s2 != AINR_ERR_SEQUENCE ||
            f2 != AINR_ERR_SEQUENCE || r2 >= 0 || s2 >= 0 || f2 >= 0) {
            fprintf(stderr, "%s: preprocess return values pcm %d/%d spec %d/%d freq %d/%d, want 0 then AINR_ERR_SEQUENCE\n",
                    label, r1, r2, s1, s2, f1, f2);
            bad = 1;
        }
    }

    if (!bad) printf("%s: PASS (1a PCM vs _spec bit-exact, 1b _spec vs _freq E==P bit-exact, "
                     "bypass hop %d)\n", label, bypass_hop);

    free(pcm_in); free(pcm_ref); free(pcm_from_spec_i16);
    free(spec_re); free(spec_im); free(freq_re); free(freq_im); free(af_re); free(af_im);
    free(hop_f); free(own_pcm_out);
    free(pool_pcm); free(pool_spec); free(pool_freq);
    free(stft_mem); free(istft_mem);
    unlink(erb_path);
    return bad;
}

/* ---- Test 2: N > 1 stash/ordering through the public API (catalog 16k/256 N=8). ---- */

#define T2_SR 16000
#define T2_FFT 256
#define T2_N 8
#define T2_ERB 24

/* Two consecutive blocks through the _spec family: block 1 committed, block 2 BYPASSED
 * (infer_status != 0). Confirms (a) block 2's oldest frame is not block 1's newest frame repeated
 * (the ordering bug this kind of stash/deliver logic is prone to) and (b) a bypassed block still
 * delivers a full N-frame output of real (non-silent) spectra, not an error or a short block. */
static int test_core_ordering_and_bypass(void) {
    const int n = T2_N, hop = T2_FFT / 2, n_bins = T2_FFT / 2 + 1;
    AinrConfig cfg = { T2_SR, T2_FFT, T2_N * (T2_FFT / 2) };
    StreamMemReq req;
    AinrSpecCtx *ctx;
    AinrWork io;
    char erb_path[160];
    void *pool;
    int16_t *pcm_in = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    float *re1 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *im1 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *re2 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *im2 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    int rc, k, f, same;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_n8_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, n_bins, T2_ERB);
    if (ainr_get_mem_size_spec(&cfg, &req) != AINR_OK) { fprintf(stderr, "test_core_ordering: get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_spec_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx) { fprintf(stderr, "test_core_ordering: init fail\n"); return 1; }

    /* Block 1: real job, committed. */
    fill_pcm(pcm_in, n * hop, 101);
    rc = ainr_preprocess_spec(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block1 pre rc=%d\n", rc); return 1; }
    fill_accel(&io, 1);
    rc = ainr_postprocess_spec(ctx, 0, re1, im1);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block1 post rc=%d\n", rc); return 1; }

    /* Block 2: real job, BYPASSED (infer_status != 0) -- must still deliver n frames. */
    fill_pcm(pcm_in, n * hop, 202);
    rc = ainr_preprocess_spec(ctx, pcm_in, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block2 pre rc=%d\n", rc); return 1; }
    fill_accel(&io, 2);
    rc = ainr_postprocess_spec(ctx, -1, re2, im2);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block2 post (bypass) rc=%d\n", rc); return 1; }

    same = 1;
    for (k = 0; k < n_bins; ++k) {
        if (re2[k] != re1[(size_t)(n - 1) * (size_t)n_bins + k] ||
            im2[k] != im1[(size_t)(n - 1) * (size_t)n_bins + k]) { same = 0; break; }
    }
    if (same) {
        fprintf(stderr, "test_core_ordering: block2 frame 0 equals block1 frame N-1 -- "
                        "stash/deliver ordering bug\n");
        return 1;
    }
    for (f = 0; f < n; ++f) {
        double e = 0.0;
        for (k = 0; k < n_bins; ++k) {
            double a = re2[(size_t)f * (size_t)n_bins + k], b = im2[(size_t)f * (size_t)n_bins + k];
            e += a * a + b * b;
        }
        if (!(e > 0.0)) { fprintf(stderr, "test_core_ordering: bypassed block frame %d is silent\n", f); return 1; }
    }

    printf("test_core_ordering_and_bypass: PASS (N=%d, block2 bypassed but still delivered %d frames)\n",
           n, n);
    free(pcm_in); free(re1); free(im1); free(re2); free(im2); free(pool);
    unlink(erb_path);
    return 0;
}

/* _freq (E == P audio-scale spectra from a test-local default-scale AcStft of the SAME PCM) against
 * _spec at N=8, bit-exact across two consecutive blocks (block 1 committed, block 2 bypassed) --
 * exercises the stash branch of the _freq loop (the first N-1 frames of a block, the
 * scratch-reuse-across-frames path that the N=1 coverage never reaches). */
static int test_freq_core_matches_spec_core(void) {
    const int n = T2_N, hop = T2_FFT / 2, n_bins = T2_FFT / 2 + 1;
    AinrConfig cfg = { T2_SR, T2_FFT, T2_N * (T2_FFT / 2) };
    StreamMemReq req_s, req_f;
    AinrSpecCtx *ctx_s;
    AinrFreqCtx *ctx_f;
    char erb_path[160];
    void *pool_s, *pool_f, *stft_mem;
    int16_t *pcm_in = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    float *hop_f = (float *)malloc((size_t)hop * sizeof(float));
    float *est_re = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *est_im = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_re_spec = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_im_spec = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_re_freq = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_im_freq = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    AcStftConfig scfg;
    AcStft *test_stft;
    int block, f, k, rc;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_n8f_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, n_bins, T2_ERB);
    if (ainr_get_mem_size_spec(&cfg, &req_s) != AINR_OK || ainr_get_mem_size_freq(&cfg, &req_f) != AINR_OK) {
        fprintf(stderr, "test_freq_core_matches_spec_core: get_mem_size fail\n"); return 1;
    }
    pool_s = alloc_aligned((size_t)req_s.bytes, req_s.alignment);
    pool_f = alloc_aligned((size_t)req_f.bytes, req_f.alignment);
    ainr_init_spec_from_erb_file(&cfg, erb_path, pool_s, (size_t)req_s.bytes, &ctx_s, &g_info);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_f, (size_t)req_f.bytes, &ctx_f, &g_info);
    if (!ctx_s || !ctx_f) { fprintf(stderr, "test_freq_core_matches_spec_core: init fail\n"); return 1; }

    ac_stft_config_default(&scfg, T2_FFT);
    stft_mem = alloc_aligned(ac_stft_get_mem_size(&scfg), 16);
    test_stft = ac_stft_init(stft_mem, ac_stft_get_mem_size(&scfg), &scfg);
    if (!test_stft) { fprintf(stderr, "test_freq_core_matches_spec_core: stft init fail\n"); return 1; }

    for (block = 0; block < 2; ++block) {
        int infer_status = (block == 0) ? 0 : -1;   /* block 0 committed, block 1 bypassed */
        AinrWork io_s, io_f;

        fill_pcm(pcm_in, n * hop, 300 + block);

        /* Audio-scale E == P spectra for this block's N frames, oldest first, from the SAME PCM the
         * _spec family consumes directly below. */
        for (f = 0; f < n; ++f) {
            for (k = 0; k < hop; ++k)
                hop_f[k] = (float)pcm_in[(size_t)f * (size_t)hop + k] / 32768.0f;
            ac_stft_push_planar(test_stft, hop_f,
                                est_re + (size_t)f * (size_t)n_bins, est_im + (size_t)f * (size_t)n_bins);
        }

        rc = ainr_preprocess_spec(ctx_s, pcm_in, &io_s);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_spec_core: spec pre rc=%d (block %d)\n", rc, block); return 1; }
        fill_accel(&io_s, 10 + block);
        rc = ainr_postprocess_spec(ctx_s, infer_status, out_re_spec, out_im_spec);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_spec_core: spec post rc=%d (block %d)\n", rc, block); return 1; }

        rc = ainr_preprocess_freq(ctx_f, est_re, est_im, est_re, est_im, &io_f);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_spec_core: freq pre rc=%d (block %d)\n", rc, block); return 1; }
        fill_accel(&io_f, 10 + block);
        rc = ainr_postprocess_freq(ctx_f, infer_status, out_re_freq, out_im_freq);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_spec_core: freq post rc=%d (block %d)\n", rc, block); return 1; }

        if (memcmp(out_re_spec, out_re_freq, (size_t)n * (size_t)n_bins * sizeof(float)) != 0 ||
            memcmp(out_im_spec, out_im_freq, (size_t)n * (size_t)n_bins * sizeof(float)) != 0) {
            fprintf(stderr, "test_freq_core_matches_spec_core: block %d mismatch (infer_status=%d)\n",
                   block, infer_status);
            return 1;
        }
    }

    printf("test_freq_core_matches_spec_core: PASS (N=%d, 2 blocks, bit-exact vs _spec)\n", n);
    free(pcm_in); free(hop_f); free(est_re); free(est_im);
    free(out_re_spec); free(out_im_spec); free(out_re_freq); free(out_im_freq);
    free(pool_s); free(pool_f); free(stft_mem);
    unlink(erb_path);
    return 0;
}

/* ---- Test 3: trim / queue labeling. ---- */

/* L / D are the mask / deep-filter lookahead hops of the catalog row. Expected values are spelled
 * out here (queue_frames = L + D + N - 1; trim_samples = (queue_frames + 1) * hop), not read from
 * the library, and must be identical for the three families' init info. */
static int run_delay_labeling(int sr, int fft, int n_erb, int n, int L, int D, const char *label) {
    AinrConfig cfg;
    StreamMemReq req_pcm, req_spec, req_freq;
    void *pool_pcm, *pool_spec, *pool_freq;
    AinrCtx *ctx_pcm;
    AinrSpecCtx *ctx_spec;
    AinrFreqCtx *ctx_freq;
    char erb_path[160];
    AinrInfo bi[3];
    int hop = fft / 2, n_bins = fft / 2 + 1, i;
    int want_queue = L + D + n - 1;
    int want_trim = (want_queue + 1) * hop;
    int want_tail = (want_trim + n * hop - 1) / (n * hop);

    cfg.sr = sr;
    cfg.fft = fft;
    cfg.hop_in = n * hop;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_delay_erb_%d_%d_%d.bin",
             sr, fft, (int)getpid());
    write_synthetic_erb(erb_path, n_bins, n_erb);

    if (ainr_get_mem_size(&cfg, &req_pcm) != AINR_OK) { fprintf(stderr, "%s: get_mem_size fail\n", label); return 1; }
    pool_pcm = alloc_aligned((size_t)req_pcm.bytes, req_pcm.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool_pcm, (size_t)req_pcm.bytes, &ctx_pcm, &bi[0]);

    if (ainr_get_mem_size_spec(&cfg, &req_spec) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_spec fail\n", label); return 1; }
    pool_spec = alloc_aligned((size_t)req_spec.bytes, req_spec.alignment);
    ainr_init_spec_from_erb_file(&cfg, erb_path, pool_spec, (size_t)req_spec.bytes, &ctx_spec, &bi[1]);

    if (ainr_get_mem_size_freq(&cfg, &req_freq) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_freq fail\n", label); return 1; }
    pool_freq = alloc_aligned((size_t)req_freq.bytes, req_freq.alignment);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_freq, (size_t)req_freq.bytes, &ctx_freq, &bi[2]);

    if (!ctx_pcm || !ctx_spec || !ctx_freq) { fprintf(stderr, "%s: init fail\n", label); return 1; }

    for (i = 0; i < 3; ++i) {
        if (bi[i].graph_path != bi[0].graph_path) {
            fprintf(stderr, "%s: [%d] graph differs from the PCM family's\n", label, i);
            return 1;
        }
        if (bi[i].queue_frames != want_queue || bi[i].trim_samples != want_trim ||
            bi[i].tail_zero_blocks != want_tail) {
            fprintf(stderr, "%s: [%d] queue=%d trim=%d tail=%d, want %d / %d / %d\n", label, i,
                    bi[i].queue_frames, bi[i].trim_samples, bi[i].tail_zero_blocks,
                    want_queue, want_trim, want_tail);
            return 1;
        }
    }

    printf("%s: PASS (queue_frames=%d trim_samples=%d tail_zero_blocks=%d, hop=%d)\n",
           label, want_queue, want_trim, want_tail, hop);
    free(pool_pcm); free(pool_spec); free(pool_freq);
    unlink(erb_path);
    return 0;
}

static int test_delay_labeling(void) {
    int rc = 0;
    /* 16k/256: L = D = 0 -> queue 0, trim one hop (128): the PCM family's own framing hop only. */
    rc |= run_delay_labeling(16000, 256, 24, 1, 0, 0, "test_delay_labeling[16k/256]");
    /* 48k/1024: L = D = 1 -> queue 2, trim 1536 (three hops) -- both lookaheads nonzero. */
    rc |= run_delay_labeling(48000, 1024, 32, 1, 1, 1, "test_delay_labeling[48k/1024]");
    /* 16k/256 N=8: queue 7, trim 1024; the whole trim is one hop_in block. */
    rc |= run_delay_labeling(16000, 256, 24, 8, 0, 0, "test_delay_labeling[16k/256 N=8]");
    return rc;
}

int main(void) {
    int rc = 0;

    rc |= run_family_consistency(16000, 256, 24, 20, 5, "16k/256 N=1");
    rc |= run_family_consistency(48000, 1024, 32, 20, 5, "48k/1024 N=1");
    rc |= test_core_ordering_and_bypass();
    rc |= test_freq_core_matches_spec_core();
    rc |= test_delay_labeling();

    if (rc) return 1;
    printf("test_ainr_spec_freq: ALL PASS\n");
    return 0;
}
