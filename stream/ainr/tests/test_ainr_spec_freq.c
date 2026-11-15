/* Host-only regression suite for ainr.h/ainr.c's spectrum-delivering family ("_freq": spectrum in, enhanced
 * spectrum out) against the PCM family, with a stand-in accelerator (no NPU) -- same convention as
 * tests/test_ainr_dfn2.c. The analysis and synthesis are the caller's: an audiolib AcStft (default config,
 * scale 1.0, PCM as int16 / 32768) feeds E == P, and an AcIstft (the same config, AC_OLA_ACCUMULATE) turns the
 * enhanced spectra back into PCM. (The file and its make target keep the name of the retired "_spec" family,
 * which did the analysis inside the library; its checks are these, on the AcStft path.) Covers:
 *
 *   1. N=1, both shipped catalog grids (16kHz/256, 48kHz/1024): ainr_preprocess()/ainr_postprocess() (PCM
 *      family) vs ainr_preprocess_freq()/ainr_postprocess_freq() through the test's own AcStft and AcIstft on
 *      the same PCM and the same fake-accelerator fill -- the PCM written back is bit-exact, with a forced
 *      bypass hop (infer_status != 0) mixed into the run.
 *   2. N > 1 stash/ordering through the public _freq API on the catalog's 16 kHz/256 N=8 row: a committed block
 *      followed by a bypassed one, and the PCM family against _freq + AcIstft bit-exact over the same blocks.
 *   3. Trim / queue labeling: AinrInfo.queue_frames and trim_samples of both families against the lookahead
 *      hops of the grid, spelled out per row.
 *
 * Not built under NO_STDIO (ainr.c's only entry points, including the new families', are
 * file-mode only). MASK_ONLY-only concepts (coefs/alpha outputs) are already guarded by ainr.h's
 * own StreamWork contract (output_count differs), so this file needs no #ifdef DFN2_MASK_ONLY of
 * its own beyond what fill_accel() uses, matching test_ainr_dfn2.c's own convention. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "ainr_test_fixture.h"
#include "audio_stft.h"
#include "stream_mem.h"

/* ---- synthetic ERB file, same convention as tests/test_ainr_dfn2.c ---- */

static AinrInfo g_info;

static void fill_pcm(int16_t *pcm, int n, int seed) {
    int i;
    for (i = 0; i < n; ++i) pcm[i] = (int16_t)((i * 13 + seed * 7) % 211 - 100);
}

/* Deterministic, non-constant fake accelerator fill: varies with hop_idx and element index, so an
 * indexing bug (wrong frame, wrong bin, cross-family data confusion) shows up as a mismatch
 * instead of being hidden behind constant values. Operates on StreamWork directly -- the same
 * tensor binding regardless of family (ainr.h's own documented contract), so one function serves
 * the PCM, _spec and _freq paths alike. */
static void fill_accel(StreamWork *io, int hop_idx) {
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

/* The caller's side of the _freq family: an AcStft analysing int16 PCM as float (/ 32768) at scale 1.0, and an
 * AcIstft synthesising the enhanced spectra, both the library default config of the grid. */
typedef struct Audio {
    AcStft *stft;
    AcIstft *istft;
    void *stft_mem, *istft_mem;
    int fft, hop, n_bins;
    float *hop_f, *hop_out;
} Audio;

static int audio_open(Audio *a, int fft) {
    AcStftConfig scfg;
    memset(a, 0, sizeof(*a));
    a->fft = fft; a->hop = fft / 2; a->n_bins = fft / 2 + 1;
    ac_stft_config_default(&scfg, fft);
    a->stft_mem = alloc_aligned(ac_stft_get_mem_size(&scfg), 16);
    a->istft_mem = alloc_aligned(ac_istft_get_mem_size(&scfg), 16);
    a->stft = ac_stft_init(a->stft_mem, ac_stft_get_mem_size(&scfg), &scfg);
    a->istft = ac_istft_init(a->istft_mem, ac_istft_get_mem_size(&scfg), &scfg);
    a->hop_f = (float *)malloc((size_t)a->hop * sizeof(float));
    a->hop_out = (float *)malloc((size_t)a->hop * sizeof(float));
    return (a->stft && a->istft && a->hop_f && a->hop_out) ? 0 : 1;
}

static void audio_close(Audio *a) {
    free(a->stft_mem); free(a->istft_mem); free(a->hop_f); free(a->hop_out);
}

/* n frames of audio-scale spectra (frame-major, oldest first) of the next n hops of pcm. */
static void audio_analyse(Audio *a, const int16_t *pcm, int n, float *re, float *im) {
    int f, k;
    for (f = 0; f < n; ++f) {
        for (k = 0; k < a->hop; ++k) a->hop_f[k] = (float)pcm[(size_t)f * (size_t)a->hop + (size_t)k] / 32768.0f;
        ac_stft_push_planar(a->stft, a->hop_f, re + (size_t)f * (size_t)a->n_bins, im + (size_t)f * (size_t)a->n_bins);
    }
}

/* n frames of enhanced spectra to n hops of int16 PCM. */
static void audio_synthesise(Audio *a, int n, const float *re, const float *im, int16_t *pcm) {
    int f, k;
    for (f = 0; f < n; ++f) {
        ac_istft_push_planar(a->istft, re + (size_t)f * (size_t)a->n_bins, im + (size_t)f * (size_t)a->n_bins, a->hop_out);
        for (k = 0; k < a->hop; ++k) pcm[(size_t)f * (size_t)a->hop + (size_t)k] = stream_saturate_s16(a->hop_out[k]);
    }
}

/* ---- Test 1: N=1 catalog grid -- the PCM family vs _freq (E == P) through the caller's AcStft / AcIstft. ---- */

static int run_family_consistency(int sr, int fft, int n_erb, int hops,
                                   int bypass_hop, const char *label) {
    AinrConfig cfg;
    StreamMemReq req_pcm, req_freq;
    void *pool_pcm, *pool_freq;
    AinrCtx *ctx_pcm;
    AinrFreqCtx *ctx_freq;
    char erb_path[160];
    int hop = fft / 2, n_bins = fft / 2 + 1;
    int16_t *pcm_in, *pcm_ref, *pcm_from_freq;
    float *freq_re, *freq_im, *af_re, *af_im;
    Audio audio;
    int i, bad = 0;

    cfg.sr = sr;
    cfg.fft = fft;
    cfg.hop_in = hop;   /* N = 1 */

    if (ainr_get_mem_size(&cfg, &req_pcm) != AINR_OK) { fprintf(stderr, "%s: get_mem_size fail\n", label); return 1; }
    if (ainr_get_mem_size_freq(&cfg, &req_freq) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_freq fail\n", label); return 1; }

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_erb_%d_%d_%d.bin", sr, fft, (int)getpid());
    write_synthetic_erb(erb_path, n_bins, n_erb);

    pool_pcm = alloc_aligned((size_t)req_pcm.bytes, req_pcm.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool_pcm, (size_t)req_pcm.bytes, &ctx_pcm, &g_info);
    pool_freq = alloc_aligned((size_t)req_freq.bytes, req_freq.alignment);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_freq, (size_t)req_freq.bytes, &ctx_freq, &g_info);
    if (!ctx_pcm || !ctx_freq) { fprintf(stderr, "%s: init fail\n", label); return 1; }
    if (audio_open(&audio, fft)) { fprintf(stderr, "%s: test stft/istft init fail\n", label); return 1; }

    pcm_in = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    pcm_ref = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    pcm_from_freq = (int16_t *)malloc((size_t)hops * (size_t)hop * sizeof(int16_t));
    freq_re = (float *)malloc((size_t)n_bins * sizeof(float));
    freq_im = (float *)malloc((size_t)n_bins * sizeof(float));
    af_re = (float *)malloc((size_t)n_bins * sizeof(float));
    af_im = (float *)malloc((size_t)n_bins * sizeof(float));
    fill_pcm(pcm_in, hops * hop, 7);

    for (i = 0; i < hops && !bad; ++i) {
        StreamWork io_pcm, io_freq;
        int infer_status = (i == bypass_hop) ? -1 : 0;
        int rc;

        /* PCM family (reference). */
        rc = ainr_preprocess(ctx_pcm, pcm_in + (size_t)i * (size_t)hop, &io_pcm);
        if (rc != 0) { fprintf(stderr, "%s: pcm preprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        fill_accel(&io_pcm, i);
        rc = ainr_postprocess(ctx_pcm, infer_status, pcm_ref + (size_t)i * (size_t)hop);
        if (rc != AINR_OK) { fprintf(stderr, "%s: pcm postprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }

        /* _freq family, E == P from the caller's AcStft of the SAME PCM hop, same fill -- then the caller's ISTFT. */
        audio_analyse(&audio, pcm_in + (size_t)i * (size_t)hop, 1, af_re, af_im);
        rc = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io_freq);
        if (rc != 0) { fprintf(stderr, "%s: freq preprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        fill_accel(&io_freq, i);
        rc = ainr_postprocess_freq(ctx_freq, infer_status, freq_re, freq_im);
        if (rc != AINR_OK) { fprintf(stderr, "%s: freq postprocess rc=%d at hop %d\n", label, rc, i); bad = 1; break; }
        audio_synthesise(&audio, 1, freq_re, freq_im, pcm_from_freq + (size_t)i * (size_t)hop);
    }

    if (!bad && memcmp(pcm_ref, pcm_from_freq, (size_t)hops * (size_t)hop * sizeof(int16_t)) != 0) {
        fprintf(stderr, "%s: PCM-vs-_freq(AcStft + AcIstft) PCM mismatch\n", label);
        bad = 1;
    }

    /* Success is exactly 0, every refusal a negative AINR_ERR_*: a second preprocess while a block is
     * still open is refused in both families. */
    if (!bad) {
        StreamWork io;
        int r1 = ainr_preprocess(ctx_pcm, pcm_in, &io), r2 = ainr_preprocess(ctx_pcm, pcm_in, &io);
        int f1 = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io);
        int f2 = ainr_preprocess_freq(ctx_freq, af_re, af_im, af_re, af_im, &io);
        if (r1 != 0 || f1 != 0 || r2 != AINR_ERR_SEQUENCE || f2 != AINR_ERR_SEQUENCE || r2 >= 0 || f2 >= 0) {
            fprintf(stderr, "%s: preprocess return values pcm %d/%d freq %d/%d, want 0 then AINR_ERR_SEQUENCE\n",
                    label, r1, r2, f1, f2);
            bad = 1;
        }
    }

    if (!bad) printf("%s: PASS (PCM vs _freq E==P through AcStft/AcIstft bit-exact, bypass hop %d)\n", label, bypass_hop);

    free(pcm_in); free(pcm_ref); free(pcm_from_freq);
    free(freq_re); free(freq_im); free(af_re); free(af_im);
    audio_close(&audio);
    free(pool_pcm); free(pool_freq);
    unlink(erb_path);
    return bad;
}

/* ---- Test 2: N > 1 stash/ordering through the public API (catalog 16k/256 N=8). ---- */

#define T2_SR 16000
#define T2_FFT 256
#define T2_N 8
#define T2_ERB 24

/* Two consecutive blocks through the _freq family: block 1 committed, block 2 BYPASSED
 * (infer_status != 0). Confirms (a) block 2's oldest frame is not block 1's newest frame repeated
 * (the ordering bug this kind of stash/deliver logic is prone to) and (b) a bypassed block still
 * delivers a full N-frame output of real (non-silent) spectra, not an error or a short block. */
static int test_core_ordering_and_bypass(void) {
    const int n = T2_N, hop = T2_FFT / 2, n_bins = T2_FFT / 2 + 1;
    AinrConfig cfg = { T2_SR, T2_FFT, T2_N * (T2_FFT / 2) };
    StreamMemReq req;
    AinrFreqCtx *ctx;
    StreamWork io;
    Audio audio;
    char erb_path[160];
    void *pool;
    int16_t *pcm_in = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    float *in_re = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *in_im = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *re1 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *im1 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *re2 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *im2 = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    int rc, k, f, same;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_n8_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, n_bins, T2_ERB);
    if (ainr_get_mem_size_freq(&cfg, &req) != AINR_OK) { fprintf(stderr, "test_core_ordering: get_mem_size fail\n"); return 1; }
    pool = alloc_aligned((size_t)req.bytes, req.alignment);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool, (size_t)req.bytes, &ctx, &g_info);
    if (!ctx || audio_open(&audio, T2_FFT)) { fprintf(stderr, "test_core_ordering: init fail\n"); return 1; }

    /* Block 1: real job, committed. */
    fill_pcm(pcm_in, n * hop, 101);
    audio_analyse(&audio, pcm_in, n, in_re, in_im);
    rc = ainr_preprocess_freq(ctx, in_re, in_im, in_re, in_im, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block1 pre rc=%d\n", rc); return 1; }
    fill_accel(&io, 1);
    rc = ainr_postprocess_freq(ctx, 0, re1, im1);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block1 post rc=%d\n", rc); return 1; }

    /* Block 2: real job, BYPASSED (infer_status != 0) -- must still deliver n frames. */
    fill_pcm(pcm_in, n * hop, 202);
    audio_analyse(&audio, pcm_in, n, in_re, in_im);
    rc = ainr_preprocess_freq(ctx, in_re, in_im, in_re, in_im, &io);
    if (rc != AINR_OK) { fprintf(stderr, "test_core_ordering: block2 pre rc=%d\n", rc); return 1; }
    fill_accel(&io, 2);
    rc = ainr_postprocess_freq(ctx, -1, re2, im2);
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
    free(pcm_in); free(in_re); free(in_im); free(re1); free(im1); free(re2); free(im2); free(pool);
    audio_close(&audio);
    unlink(erb_path);
    return 0;
}

/* The PCM family at N=8 against _freq (E == P from the caller's AcStft) + the caller's AcIstft, bit-exact over
 * two consecutive blocks (block 1 committed, block 2 bypassed) -- exercises the stash branch of the _freq loop
 * (the first N-1 frames of a block, the scratch-reuse-across-frames path that the N=1 coverage never reaches). */
static int test_freq_core_matches_pcm_core(void) {
    const int n = T2_N, hop = T2_FFT / 2, n_bins = T2_FFT / 2 + 1;
    AinrConfig cfg = { T2_SR, T2_FFT, T2_N * (T2_FFT / 2) };
    StreamMemReq req_p, req_f;
    AinrCtx *ctx_p;
    AinrFreqCtx *ctx_f;
    Audio audio;
    char erb_path[160];
    void *pool_p, *pool_f;
    int16_t *pcm_in = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    int16_t *pcm_ref = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    int16_t *pcm_freq = (int16_t *)malloc((size_t)n * (size_t)hop * sizeof(int16_t));
    float *est_re = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *est_im = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_re_freq = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    float *out_im_freq = (float *)malloc((size_t)n * (size_t)n_bins * sizeof(float));
    int block, rc;

    snprintf(erb_path, sizeof(erb_path), "/tmp/ainr_spec_freq_n8f_erb_%d.bin", (int)getpid());
    write_synthetic_erb(erb_path, n_bins, T2_ERB);
    if (ainr_get_mem_size(&cfg, &req_p) != AINR_OK || ainr_get_mem_size_freq(&cfg, &req_f) != AINR_OK) {
        fprintf(stderr, "test_freq_core_matches_pcm_core: get_mem_size fail\n"); return 1;
    }
    pool_p = alloc_aligned((size_t)req_p.bytes, req_p.alignment);
    pool_f = alloc_aligned((size_t)req_f.bytes, req_f.alignment);
    ainr_init_from_erb_file(&cfg, erb_path, pool_p, (size_t)req_p.bytes, &ctx_p, &g_info);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_f, (size_t)req_f.bytes, &ctx_f, &g_info);
    if (!ctx_p || !ctx_f || audio_open(&audio, T2_FFT)) { fprintf(stderr, "test_freq_core_matches_pcm_core: init fail\n"); return 1; }

    for (block = 0; block < 2; ++block) {
        int infer_status = (block == 0) ? 0 : -1;   /* block 0 committed, block 1 bypassed */
        StreamWork io_p, io_f;

        fill_pcm(pcm_in, n * hop, 300 + block);

        rc = ainr_preprocess(ctx_p, pcm_in, &io_p);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_pcm_core: pcm pre rc=%d (block %d)\n", rc, block); return 1; }
        fill_accel(&io_p, 10 + block);
        rc = ainr_postprocess(ctx_p, infer_status, pcm_ref);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_pcm_core: pcm post rc=%d (block %d)\n", rc, block); return 1; }

        /* Audio-scale E == P spectra for this block's N frames, oldest first, from the SAME PCM. */
        audio_analyse(&audio, pcm_in, n, est_re, est_im);
        rc = ainr_preprocess_freq(ctx_f, est_re, est_im, est_re, est_im, &io_f);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_pcm_core: freq pre rc=%d (block %d)\n", rc, block); return 1; }
        fill_accel(&io_f, 10 + block);
        rc = ainr_postprocess_freq(ctx_f, infer_status, out_re_freq, out_im_freq);
        if (rc != AINR_OK) { fprintf(stderr, "test_freq_core_matches_pcm_core: freq post rc=%d (block %d)\n", rc, block); return 1; }
        audio_synthesise(&audio, n, out_re_freq, out_im_freq, pcm_freq);

        if (memcmp(pcm_ref, pcm_freq, (size_t)n * (size_t)hop * sizeof(int16_t)) != 0) {
            fprintf(stderr, "test_freq_core_matches_pcm_core: block %d mismatch (infer_status=%d)\n",
                   block, infer_status);
            return 1;
        }
    }

    printf("test_freq_core_matches_pcm_core: PASS (N=%d, 2 blocks, _freq + AcIstft bit-exact vs the PCM family)\n", n);
    free(pcm_in); free(pcm_ref); free(pcm_freq); free(est_re); free(est_im);
    free(out_re_freq); free(out_im_freq);
    free(pool_p); free(pool_f);
    audio_close(&audio);
    unlink(erb_path);
    return 0;
}

/* ---- Test 3: trim / queue labeling. ---- */

/* L / D are the mask / deep-filter lookahead hops of the catalog row. Expected values are spelled
 * out here (queue_frames = L + D + N - 1; trim_samples = (queue_frames + 1) * hop), not read from
 * the library, and must be identical for both families' init info. */
static int run_delay_labeling(int sr, int fft, int n_erb, int n, int L, int D, const char *label) {
    AinrConfig cfg;
    StreamMemReq req_pcm, req_freq;
    void *pool_pcm, *pool_freq;
    AinrCtx *ctx_pcm;
    AinrFreqCtx *ctx_freq;
    char erb_path[160];
    AinrInfo bi[2];
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

    if (ainr_get_mem_size_freq(&cfg, &req_freq) != AINR_OK) { fprintf(stderr, "%s: get_mem_size_freq fail\n", label); return 1; }
    pool_freq = alloc_aligned((size_t)req_freq.bytes, req_freq.alignment);
    ainr_init_freq_from_erb_file(&cfg, erb_path, pool_freq, (size_t)req_freq.bytes, &ctx_freq, &bi[1]);

    if (!ctx_pcm || !ctx_freq) { fprintf(stderr, "%s: init fail\n", label); return 1; }

    for (i = 0; i < 2; ++i) {
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
    free(pool_pcm); free(pool_freq);
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
    rc |= test_freq_core_matches_pcm_core();
    rc |= test_delay_labeling();

    if (rc) return 1;
    printf("test_ainr_spec_freq: ALL PASS\n");
    return 0;
}
