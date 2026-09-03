"""Equivalence and contract gates for the Align-ULCNet pre/post class.

``ulcnet_prepost.c`` owns no signal processing: it composes the two already
parity-tested translation units (``ulcnet_process.c`` for the centered
sqrt-Hann STFT/WOLA, ``ulcnet_model_io.c`` for the feature front end and the
recurrent rings) behind one object. So the gate that matters is not a
tolerance against a Python reference -- the parity of the parts is covered by
test_ulcnet_process_c.py and test_ulcnet_model_io_c.py -- but that the class
reproduces the hand-composed legacy path SAMPLE FOR SAMPLE. Every comparison
here is therefore ``memcmp`` on fp32, never a tolerance: the class is only
allowed to reorder nothing.

Cases, all in ONE driver selected by argv[1] so audio_common and the three
model TUs are compiled once for the module:

  equiv     ULCNET_IO_TIME vs analysis + model-I/O transaction + synthesis,
            hand-composed. The reference deliberately uses ONLY the two
            composed TUs and not ulcnet_accelerator_adapter.c, which is
            itself implemented on top of the class -- comparing against it
            would be circular.
  freq      ULCNET_IO_FREQ driven by the caller's own analysis vs IO_TIME
  freqpool  the FREQ pool is smaller, and the stale-pool gate both accepts
            the matching MemReq and refuses a mismatched one
  lifecycle _create/_destroy against _init on a caller pool
  reject    _get_mem_size / _init reject-first validation
  reset     _reset really clears state (re-run reproduces the first run)
  skip      _frame_skip is the identity and does NOT step the rings
  guard     the frame state machine: a hop on top of an unfinished frame,
            a commit with no frame_inputs behind it, and a retry after a
            refused commit are all refused; skip needs no frame_inputs

The accelerator stand-in is deterministic and input-dependent, so the K/V/
logit rings and the GRU hidden swap genuinely influence later frames -- a
class that dropped a ring advance would diverge rather than agree.
"""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'Align_ULCNet')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_SOURCES = ('ulcnet_prepost.c', 'ulcnet_process.c', 'ulcnet_model_io.c')

_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define D          8
#define HOPS       400
#define SHORT_HOPS 40

/* Deterministic stand-in for the accelerator. Satisfies the FULL-WRITE
 * contract (every element of every output tensor) and is made
 * input-dependent so the rings and the GRU swap actually influence later
 * frames. */
static int fake_run(void *user, const UlcnetModelIoInputs *in,
                    UlcnetModelIoOutputs *out) {
    size_t i;
    double acc = 0.0;
    (void)user;
    for (i = 0; i < in->spectrum_bins_elements; ++i)
        acc += in->error_mag[i] + 0.5 * in->far_mag[i];
    for (i = 0; i < in->key_history_elements; ++i)
        acc += 1e-3 * in->key_history[i];
    for (i = 0; i < in->gru_hidden_elements; ++i)
        acc += 1e-3 * in->h_gru0[i];
    for (i = 0; i < out->spectrum_ri_elements; ++i)
        out->output[i] = in->error_ri[i] * (float)(0.5 + 0.25 * sin(acc + (double)i));
    for (i = 0; i < out->key_now_elements; ++i)
        out->key_now[i] = (float)(0.01 * sin(acc + 1.0 + (double)i));
    for (i = 0; i < out->value_now_elements; ++i)
        out->value_now[i] = (float)(0.01 * cos(acc + 2.0 + (double)i));
    for (i = 0; i < out->logit_now_elements; ++i)
        out->logit_now[i] = (float)(0.01 * sin(acc + 3.0 + (double)i));
    for (i = 0; i < out->gru_hidden_elements; ++i) {
        out->h_gru0_out[i] = (float)(0.5 * sin(acc + 4.0 + (double)i));
        out->h_gru1_out[i] = (float)(0.5 * cos(acc + 5.0 + (double)i));
    }
    return 0;
}

static void fill_pcm(float *error, float *far, int samples) {
    unsigned state = 12345u;
    int i;
    for (i = 0; i < samples; ++i) {
        state = state * 1103515245u + 12345u;
        error[i] = (float)((int)((state >> 16) & 0x7fff) - 16384) / 32768.0f;
        state = state * 1103515245u + 12345u;
        far[i] = (float)((int)((state >> 16) & 0x7fff) - 16384) / 32768.0f;
    }
}

/* aligned_alloc wants a size that is a multiple of the alignment. */
static void *alloc_aligned(size_t alignment, size_t bytes) {
    size_t rounded = (bytes + alignment - 1u) / alignment * alignment;
    return aligned_alloc(alignment, rounded);
}

static FftHandle *make_fft(int fft_size, void **mem_out) {
    size_t bytes = fft_get_mem_size(fft_size);
    void *mem;
    if (bytes == 0) return NULL;
    mem = alloc_aligned(16u, bytes);
    if (!mem) return NULL;
    *mem_out = mem;
    return fft_init(mem, bytes, fft_size);
}

/* The canonical per-hop loop from ulcnet_prepost.h's contract block. */
static int drive_time(UlcnetPrepost *p, const float *error, const float *far,
                      float *out, int hops) {
    int hop;
    for (hop = 0; hop < hops; ++hop) {
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        /* THE contract this class exists to guarantee: one hop in, exactly
         * one accelerator invocation, from the very first hop. A caller
         * whose runtime infers once per hop must never see anything else,
         * so this is asserted rather than looped over. */
        if (ulcnet_prepost_pre_process(p, error + (size_t)hop * ULCNET_HOP,
                                       far + (size_t)hop * ULCNET_HOP) != 1)
            return -1;
        if (ulcnet_prepost_frame_inputs(p, &inputs, &outputs) != 0) return -1;
        if (fake_run(NULL, &inputs, &outputs) != 0) return -1;
        if (ulcnet_prepost_frame_commit(p) != 0) return -1;
        if (ulcnet_prepost_post_process(p, out + (size_t)hop * ULCNET_HOP,
                                        NULL) != 0) return -1;
    }
    return 0;
}

static int identical(const float *a, const float *b, int count) {
    return memcmp(a, b, (size_t)count * sizeof(float)) == 0;
}

static float pcm_error[HOPS * ULCNET_HOP];
static float pcm_far[HOPS * ULCNET_HOP];
static float out_a[HOPS * ULCNET_HOP];
static float out_b[HOPS * ULCNET_HOP];
static float window[ULCNET_N_FFT];


/* center=False rolling analysis: one frame per hop, over the last N_FFT
 * samples. Deliberately expression-for-expression the same as
 * ulcnet_process.c's ulcnet_rfft, so this reference is independent of the
 * class yet bit-comparable with it. */
static void ref_roll(FftHandle *fft, float *history, const float *hop_in,
                     float *out_re, float *out_im) {
    static float seg[ULCNET_N_FFT];
    static Complex spec[ULCNET_BINS];
    int k;
    memmove(history, history + ULCNET_HOP,
            (size_t)(ULCNET_N_FFT - ULCNET_HOP) * sizeof(float));
    memcpy(history + ULCNET_N_FFT - ULCNET_HOP, hop_in,
           (size_t)ULCNET_HOP * sizeof(float));
    for (k = 0; k < ULCNET_N_FFT; ++k) seg[k] = history[k] * window[k];
    fft_forward_scratch(fft, seg, spec);
    for (k = 0; k < ULCNET_BINS; ++k) {
        out_re[k] = spec[k].r;
        out_im[k] = spec[k].i;
    }
}

/* ---- equiv: the class vs the hand-composed reference path ------------ */

static int case_equiv(FftHandle *fft) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoMemReq io_req;
    UlcnetModelIoState *state;
    void *state_mem, *pool;
    static float hist_error[ULCNET_N_FFT], hist_far[ULCNET_N_FFT];
    static UlcnetSynthesis synthesis;
    static UlcnetAnalysis centered;
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    int hop;

    /* path A -- analysis + the model-I/O transaction + synthesis, composed
     * by hand from the two TUs the class merely orchestrates. */
    CHECK(ulcnet_model_io_descriptor_default(D, &descriptor) == 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) == 0);
    state_mem = alloc_aligned(io_req.alignment, io_req.bytes);
    CHECK(state_mem != NULL);
    state = ulcnet_model_io_init(state_mem, io_req.bytes, &descriptor);
    CHECK(state != NULL);
    memset(hist_error, 0, sizeof hist_error);
    memset(hist_far, 0, sizeof hist_far);
    CHECK(ulcnet_synthesis_init(&synthesis, fft, window) == 0);
    CHECK(ulcnet_analysis_init(&centered, fft, window) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        static float error_re[ULCNET_BINS], error_im[ULCNET_BINS];
        static float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
        static float enhanced_re[ULCNET_BINS], enhanced_im[ULCNET_BINS];
        static float c_re[2][ULCNET_BINS], c_im[2][ULCNET_BINS];
        float *slot = out_a + (size_t)hop * ULCNET_HOP;
        int centered_frames;
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        int wrote, bin;
        /* One frame per hop, both streams from the SAME hop index. */
        ref_roll(fft, hist_error, pcm_error + (size_t)hop * ULCNET_HOP,
                 error_re, error_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * ULCNET_HOP,
                 far_re, far_im);
        /* The FRAMING claim the headers make ("centered frame k is
         * bit-identical to non-centered frame k-1"), pinned here rather
         * than asserted: the centered analysis's LAST frame on every hop is
         * memcmp-equal to the rolling frame of the same hop, on the real
         * ulcnet_analysis_push and its documented 0 / 2 / 1 schedule. */
        centered_frames = ulcnet_analysis_push(
            &centered, pcm_error + (size_t)hop * ULCNET_HOP, c_re, c_im);
        CHECK(centered_frames == (hop == 0 ? 0 : hop == 1 ? 2 : 1));
        if (centered_frames > 0) {
            CHECK(identical(c_re[centered_frames - 1], error_re, ULCNET_BINS));
            CHECK(identical(c_im[centered_frames - 1], error_im, ULCNET_BINS));
        }
        for (bin = 0; bin < ULCNET_BINS; ++bin) {
            enhanced_re[bin] = NAN;
            enhanced_im[bin] = NAN;
        }
        CHECK(ulcnet_model_io_prepare(state, error_re, error_im,
                                      far_re, far_im,
                                      &inputs, &outputs) == 0);
        CHECK(fake_run(NULL, &inputs, &outputs) == 0);
        if (ulcnet_model_io_commit(state, enhanced_re, enhanced_im) != 0) {
            memcpy(enhanced_re, error_re, sizeof enhanced_re);
            memcpy(enhanced_im, error_im, sizeof enhanced_im);
        }
        /* One frame per hop: nothing on hop 0 (its block lies inside the
         * trimmed half window), a full hop from hop 1 on. */
        wrote = ulcnet_synthesis_push(&synthesis, enhanced_re,
                                      enhanced_im, slot);
        CHECK(wrote == (hop == 0 ? 0 : ULCNET_HOP));
        if (wrote == 0) memset(slot, 0, (size_t)ULCNET_HOP * sizeof(float));
    }

    /* path B -- the class */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL);
    CHECK(ulcnet_prepost_io_mode(p) == ULCNET_IO_TIME);
    CHECK(ulcnet_prepost_hop_size(p) == ULCNET_HOP);
    CHECK(ulcnet_prepost_num_bins(p) == ULCNET_BINS);
    CHECK(ulcnet_prepost_descriptor(p) != NULL);
    CHECK(ulcnet_prepost_descriptor(p)->delay_depth == D);
    CHECK(drive_time(p, pcm_error, pcm_far, out_b, HOPS) == 0);

    printf("pool: model I/O %zu B   class %llu B (io_mode=TIME, D=%d)\n",
           io_req.bytes, (unsigned long long)req.bytes, D);
    /* The reference is not silence -- otherwise byte-identity is vacuous. */
    CHECK(!identical(out_a, out_b + 1, HOPS * ULCNET_HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * ULCNET_HOP));

    ulcnet_prepost_destroy(p);
    free(pool);
    free(state_mem);
    return 0;
}

/* ---- freq: IO_FREQ on the caller's own transform vs IO_TIME ---------- */

static int case_freq(FftHandle *fft) {
    UlcnetPrepostConfig cfg_time, cfg_freq;
    UlcnetPrepostMemReq req_time, req_freq;
    UlcnetPrepost *p_time, *p_freq;
    void *pool_time, *pool_freq;
    static float hist_error[ULCNET_N_FFT], hist_far[ULCNET_N_FFT];
    static UlcnetSynthesis synthesis;
    int hop;

    CHECK(ulcnet_prepost_config_defaults(&cfg_time, ULCNET_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(ulcnet_prepost_config_defaults(&cfg_freq, ULCNET_IO_FREQ, D) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);
    p_time = ulcnet_prepost_init(pool_time, (size_t)req_time.bytes, &cfg_time);
    p_freq = ulcnet_prepost_init(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(p_time != NULL && p_freq != NULL);
    CHECK(ulcnet_prepost_io_mode(p_freq) == ULCNET_IO_FREQ);

    CHECK(drive_time(p_time, pcm_error, pcm_far, out_a, HOPS) == 0);

    /* In IO_FREQ the CALLER owns analysis, framing clock and synthesis --
     * one frame per hop, the same center=False convention the class uses. */
    memset(hist_error, 0, sizeof hist_error);
    memset(hist_far, 0, sizeof hist_far);
    CHECK(ulcnet_synthesis_init(&synthesis, fft, window) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        static float error_re[ULCNET_BINS], error_im[ULCNET_BINS];
        static float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
        static float enhanced_re[ULCNET_BINS], enhanced_im[ULCNET_BINS];
        float *slot = out_b + (size_t)hop * ULCNET_HOP;
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        int wrote;
        ref_roll(fft, hist_error, pcm_error + (size_t)hop * ULCNET_HOP,
                 error_re, error_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * ULCNET_HOP,
                 far_re, far_im);
        CHECK(ulcnet_prepost_pre_process_freq(
                  p_freq, error_re, error_im, far_re, far_im) == 1);
        CHECK(ulcnet_prepost_frame_inputs(p_freq, &inputs, &outputs) == 0);
        CHECK(fake_run(NULL, &inputs, &outputs) == 0);
        CHECK(ulcnet_prepost_frame_commit(p_freq) == 0);
        CHECK(ulcnet_prepost_post_process_freq(p_freq, enhanced_re,
                                               enhanced_im) == 0);
        /* One frame per hop: nothing on hop 0 (its block lies inside the
         * trimmed half window), a full hop from hop 1 on. */
        wrote = ulcnet_synthesis_push(&synthesis, enhanced_re,
                                      enhanced_im, slot);
        CHECK(wrote == (hop == 0 ? 0 : ULCNET_HOP));
        if (wrote == 0) memset(slot, 0, (size_t)ULCNET_HOP * sizeof(float));
    }

    /* Cross-mode calls are refused, not silently reinterpreted. */
    CHECK(ulcnet_prepost_pre_process(p_freq, pcm_error, pcm_far) == -1);
    CHECK(ulcnet_prepost_post_process(p_freq, out_b, NULL) == -1);
    CHECK(ulcnet_prepost_pre_process_freq(p_time, pcm_error, pcm_error,
                                          pcm_far, pcm_far) == -1);

    CHECK(!identical(out_a, out_b + 1, HOPS * ULCNET_HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * ULCNET_HOP));

    ulcnet_prepost_destroy(p_time);
    ulcnet_prepost_destroy(p_freq);
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- freqpool: pool saving + the stale-pool gate --------------------- */

static int case_freqpool(FftHandle *fft) {
    UlcnetPrepostConfig cfg_time, cfg_freq, cfg_deep;
    UlcnetPrepostMemReq req_time, req_freq, req_deep;
    void *pool_time, *pool_freq;

    CHECK(ulcnet_prepost_config_defaults(&cfg_time, ULCNET_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(ulcnet_prepost_config_defaults(&cfg_freq, ULCNET_IO_FREQ, D) == 0);
    CHECK(ulcnet_prepost_config_defaults(&cfg_deep, ULCNET_IO_TIME, 2 * D) == 0);
    cfg_deep.fft = fft;
    cfg_deep.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg_deep, &req_deep) == 0);

    printf("pool TIME = %llu B\npool FREQ = %llu B   (saves %llu B, %.1f%%)\n",
           (unsigned long long)req_time.bytes,
           (unsigned long long)req_freq.bytes,
           (unsigned long long)(req_time.bytes - req_freq.bytes),
           100.0 * (double)(req_time.bytes - req_freq.bytes) /
               (double)req_time.bytes);

    /* IO_FREQ must not pay for the framing machinery it never runs. */
    CHECK(req_freq.bytes < req_time.bytes);
    CHECK(req_freq.io_mode == (uint32_t)ULCNET_IO_FREQ);
    CHECK(req_time.io_mode == (uint32_t)ULCNET_IO_TIME);
    CHECK(req_freq.build_flags_hash != req_time.build_flags_hash);
    CHECK(req_deep.build_flags_hash != req_time.build_flags_hash);
    CHECK(req_time.descriptor_version == ULCNET_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_time.layout_version == ULCNET_MODEL_IO_LAYOUT_VERSION);
    CHECK(req_time.reserved == 0u);

    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes, &cfg_time,
                                 &req_time) != NULL);
    CHECK(ulcnet_prepost_init_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq,
                                 &req_freq) != NULL);
    /* ...and refuse a pool sized for another io_mode or another D. */
    CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes, &cfg_time,
                                 &req_freq) == NULL);
    CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes, &cfg_time,
                                 &req_deep) == NULL);
    {
        UlcnetPrepostMemReq stale = req_time;
        ++stale.descriptor_version;
        CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                     &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.layout_version;
        CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                     &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.build_flags_hash;
        CHECK(ulcnet_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                     &cfg_time, &stale) == NULL);
    }
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(FftHandle *fft) {
    UlcnetPrepostConfig cfg, bad;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *heap, *stack;
    void *pool;

    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;

    heap = ulcnet_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(ulcnet_prepost_io_mode(heap) == ULCNET_IO_TIME);
    CHECK(ulcnet_prepost_hop_size(heap) == ULCNET_HOP);
    CHECK(ulcnet_prepost_num_bins(heap) == ULCNET_BINS);
    CHECK(ulcnet_prepost_descriptor(heap)->delay_depth == D);
    CHECK(drive_time(heap, pcm_error, pcm_far, out_a, SHORT_HOPS) == 0);

    /* A created instance and a pool instance are the same object. */
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    stack = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(stack != NULL);
    CHECK(drive_time(stack, pcm_error, pcm_far, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * ULCNET_HOP));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still
     * runs afterwards. (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    ulcnet_prepost_destroy(stack);
    ulcnet_prepost_destroy(stack);
    ulcnet_prepost_reset(stack);
    CHECK(drive_time(stack, pcm_error, pcm_far, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * ULCNET_HOP));

    ulcnet_prepost_destroy(heap);
    ulcnet_prepost_destroy(NULL);
    free(pool);

    /* A heap instance in the mode that needs no transform at all. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_FREQ, D) == 0);
    heap = ulcnet_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(ulcnet_prepost_io_mode(heap) == ULCNET_IO_FREQ);
    ulcnet_prepost_destroy(heap);

    /* _create refuses what _get_mem_size refuses, and allocates nothing. */
    CHECK(ulcnet_prepost_create(NULL) == NULL);
    CHECK(ulcnet_prepost_config_defaults(&bad, ULCNET_IO_TIME, D) == 0);
    CHECK(ulcnet_prepost_create(&bad) == NULL);   /* fft/window still NULL */

    /* Accessors on NULL. */
    CHECK(ulcnet_prepost_hop_size(NULL) == -1);
    CHECK(ulcnet_prepost_num_bins(NULL) == -1);
    CHECK(ulcnet_prepost_io_mode(NULL) == -1);
    CHECK(ulcnet_prepost_descriptor(NULL) == NULL);
    ulcnet_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

static int case_reject(FftHandle *fft) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req, guard;
    void *wrong_mem = NULL;
    FftHandle *wrong_fft;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(ulcnet_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, NULL) == -1);
    CHECK(ulcnet_prepost_config_defaults(NULL, ULCNET_IO_TIME, D) == -1);

    /* Unknown io_mode. *req is left untouched on a refusal. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    cfg.io_mode = 2;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.io_mode = -1;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, 2, D) == -1);
    cfg.io_mode = ULCNET_IO_TIME;

    /* D outside [MIN_D, MAX_D]. */
    cfg.delay_depth = ULCNET_MODEL_IO_MIN_D - 1;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.delay_depth = ULCNET_MODEL_IO_MAX_D + 1;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.delay_depth = 0;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME,
                                         ULCNET_MODEL_IO_MIN_D - 1) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME,
                                         ULCNET_MODEL_IO_MAX_D + 1) == -1);
    /* Both ends of the range are accepted. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME,
                                         ULCNET_MODEL_IO_MIN_D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME,
                                         ULCNET_MODEL_IO_MAX_D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);

    /* IO_TIME without a usable transform. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.window = window;
    cfg.fft = NULL;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.fft = fft;
    cfg.window = NULL;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);

    /* A correctly typed but WRONG-SIZE FftHandle: 129 bins, not 257. */
    wrong_fft = make_fft(ULCNET_N_FFT / 2, &wrong_mem);
    CHECK(wrong_fft != NULL);
    CHECK(fft_get_n_freqs(wrong_fft) != ULCNET_BINS);
    cfg.fft = wrong_fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(ulcnet_prepost_init(NULL, 0, &cfg) == NULL);

    /* IO_FREQ ignores fft/window entirely -- NULL is legal there. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_FREQ, D) == 0);
    CHECK(cfg.fft == NULL && cfg.window == NULL);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);

    /* _init: NULL pool, undersized pool, misaligned pool. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes + req.alignment);
    CHECK(pool != NULL);
    CHECK(ulcnet_prepost_init(NULL, (size_t)req.bytes, &cfg) == NULL);
    CHECK(ulcnet_prepost_init(pool, (size_t)req.bytes - 1u, &cfg) == NULL);
    CHECK(ulcnet_prepost_init((unsigned char *)pool + 1,
                              (size_t)req.bytes, &cfg) == NULL);
    CHECK(ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg) != NULL);

    /* Stage calls on NULL, and post_process while a frame is still open. */
    {
        UlcnetPrepost *p = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        int hop, opened = 0;
        CHECK(p != NULL);
        CHECK(ulcnet_prepost_pre_process(NULL, pcm_error, pcm_far) == -1);
        CHECK(ulcnet_prepost_pre_process(p, NULL, pcm_far) == -1);
        CHECK(ulcnet_prepost_pre_process(p, pcm_error, NULL) == -1);
        CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(ulcnet_prepost_frame_commit(p) == -1);
        CHECK(ulcnet_prepost_frame_skip(p) == -1);
        CHECK(ulcnet_prepost_post_process(p, out_a, NULL) == 0);
        for (hop = 0; hop < 4 && !opened; ++hop) {
            int n = ulcnet_prepost_pre_process(p, pcm_error, pcm_far);
            CHECK(n >= 0);
            opened = n > 0;
        }
        CHECK(opened);
        CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(ulcnet_prepost_post_process(p, out_a, NULL) == -1);
        /* A partial accelerator write is refused and moves nothing. */
        outputs.output[0] = 0.0f;
        CHECK(ulcnet_prepost_frame_commit(p) == -1);
        CHECK(ulcnet_prepost_frame_skip(p) == 0);
    }

    free(pool);
    free(wrong_mem);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(FftHandle *fft) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    void *pool;

    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    CHECK(drive_time(p, pcm_error, pcm_far, out_a, SHORT_HOPS) == 0);
    /* Without the reset the second run continues from a warm state, so this
     * comparison has teeth: assert that it does NOT match first. */
    CHECK(drive_time(p, pcm_error, pcm_far, out_b, SHORT_HOPS) == 0);
    CHECK(!identical(out_a, out_b, SHORT_HOPS * ULCNET_HOP));

    ulcnet_prepost_reset(p);
    /* One inference per hop, from the very first hop. */
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == 1);
    ulcnet_prepost_reset(p);
    CHECK(drive_time(p, pcm_error, pcm_far, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * ULCNET_HOP));

    /* Reset mid-hop drops the open frame instead of stranding it. */
    {
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        int hop, opened = 0;
        for (hop = 0; hop < 4 && !opened; ++hop)
            opened = ulcnet_prepost_pre_process(p, pcm_error, pcm_far) > 0;
        CHECK(opened);
        CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        ulcnet_prepost_reset(p);
        CHECK(ulcnet_prepost_frame_commit(p) == -1);
        CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(drive_time(p, pcm_error, pcm_far, out_b, SHORT_HOPS) == 0);
        CHECK(identical(out_a, out_b, SHORT_HOPS * ULCNET_HOP));
    }

    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- skip: the identity, and no ring advance ------------------------- */

static int case_skip(void) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    void *pool;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    static float error_re[ULCNET_BINS], error_im[ULCNET_BINS];
    static float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
    static float got_re[ULCNET_BINS], got_im[ULCNET_BINS];
    float *key = NULL, *value = NULL, *logit = NULL;
    float *gru0 = NULL, *gru1 = NULL;
    size_t key_n, value_n, logit_n, gru_n;
    int bin;

    /* IO_FREQ so the ring state is observable without the framing warm-up. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_FREQ, D) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    for (bin = 0; bin < ULCNET_BINS; ++bin) {
        error_re[bin] = 0.5f - (float)bin / (float)ULCNET_BINS;
        error_im[bin] = (float)bin / (float)ULCNET_BINS - 0.25f;
        far_re[bin] = 0.125f + 0.5f * (float)bin / (float)ULCNET_BINS;
        far_im[bin] = 0.75f - (float)bin / (float)ULCNET_BINS;
    }

    /* Frame 1: a real commit, so the rings are NON-ZERO for the check. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_commit(p) == 0);

    /* Frame 2: snapshot the rings, then take the identity even though the
     * accelerator produced a COMPLETE, perfectly committable result.
     *
     * The full write is the point. Skipping after a partial write would
     * prove nothing -- commit() would refuse that frame on its own -- so a
     * skip that quietly committed would still look correct. Here the only
     * thing standing between the rings and an advance is frame_skip's
     * contract. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    key_n = inputs.key_history_elements;
    value_n = inputs.value_history_elements;
    logit_n = inputs.logit_history_elements;
    gru_n = inputs.gru_hidden_elements;
    key = (float *)malloc(key_n * sizeof(float));
    value = (float *)malloc(value_n * sizeof(float));
    logit = (float *)malloc(logit_n * sizeof(float));
    gru0 = (float *)malloc(gru_n * sizeof(float));
    gru1 = (float *)malloc(gru_n * sizeof(float));
    CHECK(key && value && logit && gru0 && gru1);
    memcpy(key, inputs.key_history, key_n * sizeof(float));
    memcpy(value, inputs.value_history, value_n * sizeof(float));
    memcpy(logit, inputs.logit_history, logit_n * sizeof(float));
    memcpy(gru0, inputs.h_gru0, gru_n * sizeof(float));
    memcpy(gru1, inputs.h_gru1, gru_n * sizeof(float));
    /* The snapshot must not be all zeros, or "unchanged" proves nothing. */
    {
        size_t i;
        int nonzero = 0;
        for (i = 0; i < key_n; ++i) if (key[i] != 0.0f) nonzero = 1;
        for (i = 0; i < gru_n; ++i) if (gru0[i] != 0.0f) nonzero = 1;
        CHECK(nonzero);
    }

    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* Identity: the error spectrum passes through UNCHANGED, bit for bit --
     * the accelerator's enhanced output is discarded, not applied. */
    CHECK(identical(error_re, got_re, ULCNET_BINS));
    CHECK(identical(error_im, got_im, ULCNET_BINS));

    /* Frame 3: the rings must be exactly what frame 2 saw. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(inputs.key_history_elements == key_n);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.value_history, value, value_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.logit_history, logit, logit_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.h_gru0, gru0, gru_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.h_gru1, gru1, gru_n * sizeof(float)) == 0);

    /* The other reason to skip: the accelerator wrote NOTHING at all. Same
     * identity, same frozen rings. */
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    CHECK(identical(error_re, got_re, ULCNET_BINS));
    CHECK(identical(error_im, got_im, ULCNET_BINS));

    /* Frame 4: still frozen -- and a commit here DOES move the rings, so
     * the three comparisons above are not measuring a constant. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.h_gru0, gru0, gru_n * sizeof(float)) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_commit(p) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) != 0);

    free(key);
    free(value);
    free(logit);
    free(gru0);
    free(gru1);
    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- guard: the frame state machine refuses what it must -------------- */

static int case_guard(FftHandle *fft) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    void *pool;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    static float error_re[ULCNET_BINS], error_im[ULCNET_BINS];
    static float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
    static float got_re[ULCNET_BINS], got_im[ULCNET_BINS];
    size_t i;
    int bin;

    /* (1) TIME: a hop on top of an unfinished one is refused, and the open
     * frame is still the caller's to finish. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == 1);
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == -1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == -1);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_commit(p) == 0);
    CHECK(ulcnet_prepost_post_process(p, out_a, NULL) == 0);
    /* ...and once finished -- by commit or by skip -- the next hop is
     * accepted again. */
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == 1);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    CHECK(ulcnet_prepost_pre_process(p, pcm_error, pcm_far) == 1);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    ulcnet_prepost_destroy(p);
    free(pool);

    /* (2)+(3) FREQ: the same refusal, then the transaction gates. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, ULCNET_IO_FREQ, D) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    for (bin = 0; bin < ULCNET_BINS; ++bin) {
        error_re[bin] = 0.5f - (float)bin / (float)ULCNET_BINS;
        error_im[bin] = (float)bin / (float)ULCNET_BINS - 0.25f;
        far_re[bin] = 0.125f + 0.5f * (float)bin / (float)ULCNET_BINS;
        far_im[bin] = 0.75f - (float)bin / (float)ULCNET_BINS;
    }
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == -1);
    /* (2) A frame is open but the accelerator was never handed it: commit
     * is refused rather than passing untouched buffers off as a result. */
    CHECK(ulcnet_prepost_frame_commit(p) == -1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_commit(p) == 0);
    /* (3) A commit refused on a non-finite output disarms the transaction:
     * a retry without a fresh frame_inputs is refused too, the frame is
     * still open, and the fresh frame_inputs re-fills EVERY accelerator
     * output with NaN before the accelerator is asked again. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    outputs.h_gru1_out[outputs.gru_hidden_elements - 1u] = NAN;
    CHECK(ulcnet_prepost_frame_commit(p) == -1);
    CHECK(ulcnet_prepost_frame_commit(p) == -1);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == -1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    for (i = 0; i < outputs.spectrum_ri_elements; ++i)
        CHECK(isnan(outputs.output[i]));
    for (i = 0; i < outputs.key_now_elements; ++i)
        CHECK(isnan(outputs.key_now[i]));
    for (i = 0; i < outputs.value_now_elements; ++i)
        CHECK(isnan(outputs.value_now[i]));
    for (i = 0; i < outputs.logit_now_elements; ++i)
        CHECK(isnan(outputs.logit_now[i]));
    for (i = 0; i < outputs.gru_hidden_elements; ++i) {
        CHECK(isnan(outputs.h_gru0_out[i]));
        CHECK(isnan(outputs.h_gru1_out[i]));
    }
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_commit(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* (4) Skipping never needed the accelerator, so it needs no
     * frame_inputs either. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    /* (5) ...and a skip taken AFTER frame_inputs armed the accelerator must
     * not leave that arming behind: the next frame's commit, with no
     * frame_inputs of its own, is still refused. This is the state in which
     * a latch borrowed from model_io alone lets stale outputs through, so
     * (2) above cannot stand in for it. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_commit(p) == -1);
    CHECK(ulcnet_prepost_frame_skip(p) == 0);

    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

int main(int argc, char **argv) {
    void *fft_mem = NULL;
    FftHandle *fft;
    int status;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <case>\n", argv[0]);
        return 2;
    }
    ulcnet_make_window(window);
    fill_pcm(pcm_error, pcm_far, HOPS * ULCNET_HOP);
    fft = make_fft(ULCNET_N_FFT, &fft_mem);
    CHECK(fft != NULL);

    if (strcmp(argv[1], "equiv") == 0)          status = case_equiv(fft);
    else if (strcmp(argv[1], "freq") == 0)      status = case_freq(fft);
    else if (strcmp(argv[1], "freqpool") == 0)  status = case_freqpool(fft);
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle(fft);
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject(fft);
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset(fft);
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard(fft);
    else {
        fprintf(stderr, "unknown case: %s\n", argv[1]);
        status = 2;
    }
    free(fft_mem);
    return status;
}
'''


@pytest.fixture(scope='module')
def audio_common_lib():
    """audio_common's NE10 static lib (the default backend since
    2026-08-18), built and located the way the other C tests document it."""
    if shutil.which('make') is None:
        pytest.skip('no make available')
    subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'lib'],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True,
    )
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    """One executable for every case: the class plus the three TUs it
    composes, compiled at the house flags with -Werror."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('ulcnet_prepost_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c11',
         '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off',
         '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, str(source),
         *[os.path.join(_ULCNET_DIR, name) for name in _SOURCES],
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


def _run(driver, case):
    done = subprocess.run([str(driver), case], capture_output=True, text=True)
    assert done.returncode == 0, (
        'case %s failed (rc=%d)\n%s' % (case, done.returncode, done.stderr))
    return done.stdout


def test_time_mode_matches_the_hand_composed_path(driver):
    """The class vs analysis + the model-I/O transaction + synthesis wired
    up by hand, byte-identical over 400 hops. This is the gate that folding
    the composition into an object changed nothing at all. The same case
    pins the framing claim: the centered analysis's last frame per hop is
    bit-identical to the class's rolling frame."""
    assert 'io_mode=TIME' in _run(driver, 'equiv')


def test_freq_mode_matches_time_mode(driver):
    """ULCNET_IO_FREQ fed from the caller's OWN analysis, re-synthesised by
    the caller, must reproduce ULCNET_IO_TIME sample for sample -- the two
    modes may differ in who owns the transform, never in the result."""
    _run(driver, 'freq')


def test_freq_pool_is_smaller_and_stale_pool_gate_holds(driver):
    """IO_FREQ must not carve the framing machinery it never runs, and
    _init_ex must refuse a pool recorded for another mode, D or build --
    while still accepting the matching one."""
    stdout = _run(driver, 'freqpool')
    assert 'pool FREQ' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe, and on a
    pool instance it is the no-op the contract promises."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size / _init refuse NULLs, an unknown io_mode, a D outside
    [MIN_D, MAX_D], IO_TIME without a usable transform, a wrong-size
    FftHandle, and an undersized or misaligned pool."""
    _run(driver, 'reject')


def test_reset_clears_state(driver):
    """N hops, reset, the same N hops again -> byte-identical. Asserted
    against a NO-reset second run first, so the test cannot pass by the
    output being state-independent."""
    _run(driver, 'reset')


def test_frame_skip_is_identity_and_does_not_step_the_rings(driver):
    """_frame_skip passes the error spectrum through unchanged and leaves
    the K/V/logit rings and both GRU hidden tensors exactly as they were --
    checked against non-zero state, and against a commit that does move
    them."""
    _run(driver, 'skip')


def test_frame_state_machine_guards(driver):
    """A hop on top of an unfinished frame is refused (and the open frame
    stays the caller's to finish), a commit with no frame_inputs behind it
    is refused, and a commit refused on a non-finite output disarms the
    transaction until a fresh frame_inputs -- which NaN-refills every
    accelerator output. Skipping needs no frame_inputs at all."""
    _run(driver, 'guard')
