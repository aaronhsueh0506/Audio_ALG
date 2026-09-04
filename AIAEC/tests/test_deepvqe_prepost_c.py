"""Equivalence and contract gates for the DeepVQE-S pre/post class.

``DeepVQE_S/deepvqe_prepost.c`` owns no signal processing: it composes two
already parity-tested translation units (``aiaec_process.c`` for the 16 kHz
sqrt-Hann STFT/WOLA -- itself a thin wrapper over ``ulcnet_process.c`` -- and
``DeepVQE_S/deepvqe_process.c`` for the CCM kernel) behind one object. So the
gate that matters is not a tolerance against a Python reference -- the parity
of the parts is covered elsewhere -- but that the class reproduces the
hand-composed path SAMPLE FOR SAMPLE. Every numeric comparison here is
therefore ``memcmp`` on fp32, never a tolerance: the class is allowed to
reorder nothing. Every "identical" claim is paired with a "not trivially
identical" one, so no gate can pass by both sides being silence.

Cases, all in ONE driver selected by argv[1] so audio_common and the four
model TUs are compiled once for the module:

  equiv      DEEPVQE_IO_TIME vs a rolling analysis + two hand-held state
             banks + deepvqe_ccm_process + aiaec_synthesis_push, wired up by
             hand from ONLY the composed TUs
  freq       DEEPVQE_IO_FREQ driven by the caller's own transform vs IO_TIME
  freqpool   the FREQ pool is smaller, and the stale-pool gate both accepts
             the matching MemReq and refuses every mismatched one
  lifecycle  _create/_destroy against _init on a caller pool
  reject     _get_mem_size / _init / the stage calls, reject-first
  reset      _reset really clears state (a re-run reproduces the first run)
  skip       _frame_skip MUTES (fail-closed) and freezes the banks AND the
             CCM ring
  guard      the three contract gates: no double pre_process, no commit
             without frame_inputs, no re-commit after a refused one
  boundary   the explicit state boundary -- names, shapes, element counts,
             and the descriptor validator's 12 refusable fields
  descriptor a shape/validate dump driven from argv, so the Python side can
             feed it a descriptor MEASURED from the built graph

The accelerator stand-in is deterministic and input-dependent -- it sums both
signal inputs and all sixteen state tensors -- so a bank swap that failed to
happen, or happened when it should not have, diverges rather than agrees.

THE SKIP POLICY IS DELIBERATE. DeepVQE-S's stream 0 is the RAW microphone,
not a residual, so the pass-through identity a post-filter takes would emit
the full uncancelled echo. ``deepvqe_prepost_frame_skip`` therefore emits
SILENCE, and this suite asserts the mute rather than a pass-through.
"""

import json
import os
import re
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_ULCNET_DIR = os.path.join(_AIAEC_DIR, 'Align_ULCNet')
_DEEPVQE_DIR = os.path.join(_AIAEC_DIR, 'DeepVQE_S')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_PREPOST_HEADER = os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.h')

_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c'),
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
    os.path.join(_AIAEC_DIR, 'aiaec_process.c'),
    os.path.join(_ULCNET_DIR, 'ulcnet_process.c'),
)

# The descriptor's 13 fields, in the order the `descriptor` case reads them
# off argv. Fixed here and in the C driver together: a reordering on one side
# only would silently validate a permuted descriptor.
_DESCRIPTOR_KEYS = (
    'layout_version',
    'delay_depth',
    'sample_rate',
    'fft_size',
    'hop_size',
    'spectrum_bins',
    'time_order',
    'freq_taps',
    'conv_history_frames',
    'score_history_frames',
    'gru_layers',
    'gru_hidden',
    'state_tensor_count',
)


_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

/* D=8 everywhere except where a case explicitly needs the shipped default:
 * every state tensor's size scales with it, and the fake accelerator walks
 * all of them on every frame. */
#define D          8
#define HOPS       400
#define SHORT_HOPS 40

#define TAPS_PER_BIN (DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2)

/* Deterministic stand-in for the accelerator. It satisfies the FULL-WRITE
 * contract -- every element of `taps` and of all sixteen state_out tensors --
 * and is made input-dependent on BOTH signal inputs and EVERY state tensor,
 * so a bank swap that did not happen (or happened when it should not have)
 * changes every later frame instead of going unnoticed. */
static int fake_run(const DeepVqePrepostInputs *in,
                    DeepVqePrepostOutputs *out) {
    double acc = 0.0;
    size_t i;
    int id;

    for (i = 0; i < in->spectrum_ri_elements; ++i)
        acc += (double)in->mic[i] + 0.5 * (double)in->far[i];
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (i = 0; i < in->state_elements[id]; ++i)
            acc += 1e-3 * (double)in->state[id][i];

    for (i = 0; i < out->taps_elements; ++i) {
        /* taps: [bin][time][freq -1,0,+1][re,im]. The centre tap is
         * time 0 / freq 0, i.e. offsets 2 and 3 of each bin's 18. */
        size_t slot = i % (size_t)TAPS_PER_BIN;
        out->taps[i] = (slot == 2u || slot == 3u)
            ? (float)(0.3 * sin(acc + (double)i))
            : (float)(0.05 * cos(acc + (double)i));
    }
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (i = 0; i < out->state_elements[id]; ++i)
            out->state_out[id][i] =
                (float)(0.5 * sin(acc + (double)id + 1e-2 * (double)i));
    return 0;
}

static void fill_pcm(float *mic, float *far, int samples) {
    unsigned state = 12345u;
    int i;
    for (i = 0; i < samples; ++i) {
        state = state * 1103515245u + 12345u;
        mic[i] = (float)((int)((state >> 16) & 0x7fff) - 16384) / 32768.0f;
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

static int identical(const float *a, const float *b, int count) {
    return memcmp(a, b, (size_t)count * sizeof(float)) == 0;
}

static int all_zero(const float *values, int count) {
    int i;
    for (i = 0; i < count; ++i) if (values[i] != 0.0f) return 0;
    return 1;
}

static float pcm_mic[HOPS * AIAEC_HOP];
static float pcm_far[HOPS * AIAEC_HOP];
static float out_a[HOPS * AIAEC_HOP];
static float out_b[HOPS * AIAEC_HOP];
static float window[AIAEC_N_FFT];

/* The canonical per-hop loop from deepvqe_prepost.h's contract block.
 * `cold` asserts the synthesis warm-up schedule: the first frame lands
 * entirely inside the trimmed half-window, so `written` is 0 on hop 0 and
 * AIAEC_HOP from then on. A run continued on a WARM instance emits a full
 * hop immediately, which is why the schedule is a parameter and not a
 * constant. */
static int drive_time(DeepVqePrepost *p, const float *mic, const float *far,
                      float *out, int hops, int cold) {
    int hop;
    for (hop = 0; hop < hops; ++hop) {
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        int written = -1;
        int expect = (cold && hop == 0) ? 0 : AIAEC_HOP;
        /* THE contract this class exists to guarantee: one hop in, exactly
         * one accelerator invocation, from the very first hop. It is
         * asserted every hop rather than looped over. */
        if (deepvqe_prepost_pre_process(p, mic + (size_t)hop * AIAEC_HOP,
                                        far + (size_t)hop * AIAEC_HOP) != 1)
            return -1;
        if (deepvqe_prepost_frame_inputs(p, &inputs, &outputs) != 0) return -1;
        if (fake_run(&inputs, &outputs) != 0) return -1;
        if (deepvqe_prepost_frame_commit(p) != 0) return -1;
        if (deepvqe_prepost_post_process(p, out + (size_t)hop * AIAEC_HOP,
                                         &written) != 0) return -1;
        if (written != expect) return -1;
    }
    return 0;
}

/* center=False rolling analysis: one frame per hop over the last N_FFT
 * samples. Deliberately expression-for-expression the same as
 * ulcnet_process.c's ulcnet_rfft (the transform behind the
 * aiaec_analysis_push_frame the class drives), so this reference is
 * independent of deepvqe_prepost.c yet bit-comparable with it. */
static void ref_roll(FftHandle *fft, float *history, const float *hop_in,
                     float *out_re, float *out_im) {
    static float seg[AIAEC_N_FFT];
    static Complex spec[AIAEC_N_BINS];
    int k;
    memmove(history, history + AIAEC_HOP,
            (size_t)(AIAEC_N_FFT - AIAEC_HOP) * sizeof(float));
    memcpy(history + AIAEC_N_FFT - AIAEC_HOP, hop_in,
           (size_t)AIAEC_HOP * sizeof(float));
    for (k = 0; k < AIAEC_N_FFT; ++k) seg[k] = history[k] * window[k];
    fft_forward_scratch(fft, seg, spec);
    for (k = 0; k < AIAEC_N_BINS; ++k) {
        out_re[k] = spec[k].r;
        out_im[k] = spec[k].i;
    }
}

static void interleave_ri(const float *re, const float *im, float *ri) {
    int bin;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        ri[2 * bin] = re[bin];
        ri[2 * bin + 1] = im[bin];
    }
}

/* ---- equiv: the class vs the hand-composed reference path ------------ */

static int case_equiv(FftHandle *fft) {
    /* Path A -- the two composed TUs plus two state banks held by hand.
     * deepvqe_prepost.c contributes nothing here except the pure shape
     * helper, so this reference cannot be circular. */
    static float hist_mic[AIAEC_N_FFT], hist_far[AIAEC_N_FFT];
    static AiaecSynthesis synthesis;
    static DeepVqeCcmState ccm;
    static float mic_ri[2 * AIAEC_N_BINS], far_ri[2 * AIAEC_N_BINS];
    static float taps_ref[DEEPVQE_TAPS_ELEMENTS];
    float *bank[2][DEEPVQE_STATE_COUNT];
    size_t elements[DEEPVQE_STATE_COUNT];
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    void *pool;
    int front = 0;
    int bank_index, id, hop;

    for (bank_index = 0; bank_index < 2; ++bank_index) {
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            elements[id] = deepvqe_prepost_state_elements(id, D);
            CHECK(elements[id] > 0u);
            bank[bank_index][id] =
                (float *)calloc(elements[id], sizeof(float));
            CHECK(bank[bank_index][id] != NULL);
        }
    }
    memset(hist_mic, 0, sizeof hist_mic);
    memset(hist_far, 0, sizeof hist_far);
    deepvqe_ccm_init(&ccm);
    CHECK(aiaec_synthesis_init(&synthesis, fft, window) == 0);

    for (hop = 0; hop < HOPS; ++hop) {
        static float mic_re[AIAEC_N_BINS], mic_im[AIAEC_N_BINS];
        static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
        static float enh_re[AIAEC_N_BINS], enh_im[AIAEC_N_BINS];
        float *slot = out_a + (size_t)hop * AIAEC_HOP;
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        int wrote;

        ref_roll(fft, hist_mic, pcm_mic + (size_t)hop * AIAEC_HOP,
                 mic_re, mic_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * AIAEC_HOP,
                 far_re, far_im);
        interleave_ri(mic_re, mic_im, mic_ri);
        interleave_ri(far_re, far_im, far_ri);

        memset(&inputs, 0, sizeof inputs);
        memset(&outputs, 0, sizeof outputs);
        inputs.mic = mic_ri;
        inputs.far = far_ri;
        inputs.spectrum_ri_elements = 2u * (size_t)AIAEC_N_BINS;
        outputs.taps = taps_ref;
        outputs.taps_elements = DEEPVQE_TAPS_ELEMENTS;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            inputs.state[id] = bank[front][id];
            inputs.state_elements[id] = elements[id];
            outputs.state_out[id] = bank[front ^ 1][id];
            outputs.state_elements[id] = elements[id];
        }
        CHECK(fake_run(&inputs, &outputs) == 0);
        front ^= 1;   /* the graph returns the FULL next state: swap banks */

        deepvqe_ccm_process(
            &ccm, mic_re, mic_im,
            (const float (*)[DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2])
                taps_ref,
            enh_re, enh_im);
        wrote = aiaec_synthesis_push(&synthesis, enh_re, enh_im, slot);
        /* The synthesis half-window trim, restated on the reference side. */
        CHECK(wrote == (hop == 0 ? 0 : AIAEC_HOP));
        if (wrote < AIAEC_HOP)
            memset(slot + wrote, 0,
                   (size_t)(AIAEC_HOP - wrote) * sizeof(float));
    }

    /* Path B -- the class. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = deepvqe_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL);
    CHECK(deepvqe_prepost_io_mode(p) == DEEPVQE_IO_TIME);
    CHECK(deepvqe_prepost_hop_size(p) == AIAEC_HOP);
    CHECK(deepvqe_prepost_num_bins(p) == AIAEC_N_BINS);
    CHECK(deepvqe_prepost_descriptor(p) != NULL);
    CHECK(deepvqe_prepost_descriptor(p)->delay_depth == D);
    CHECK(drive_time(p, pcm_mic, pcm_far, out_b, HOPS, 1) == 0);

    printf("pool: class %llu B (io_mode=TIME, D=%d)\n",
           (unsigned long long)req.bytes, D);
    /* The reference is not silence and not a constant -- otherwise
     * byte-identity below would be vacuous. */
    CHECK(!all_zero(out_a, HOPS * AIAEC_HOP));
    CHECK(!identical(out_a, out_b + 1, HOPS * AIAEC_HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * AIAEC_HOP));

    deepvqe_prepost_destroy(p);
    free(pool);
    for (bank_index = 0; bank_index < 2; ++bank_index)
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            free(bank[bank_index][id]);
    return 0;
}

/* ---- freq: IO_FREQ on the caller's own transform vs IO_TIME ---------- */

static int case_freq(FftHandle *fft) {
    DeepVqePrepostConfig cfg_time, cfg_freq;
    DeepVqePrepostMemReq req_time, req_freq;
    DeepVqePrepost *p_time, *p_freq;
    void *pool_time, *pool_freq;
    static float hist_mic[AIAEC_N_FFT], hist_far[AIAEC_N_FFT];
    static AiaecSynthesis synthesis;
    int hop;

    CHECK(deepvqe_prepost_config_defaults(&cfg_time, DEEPVQE_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);
    p_time = deepvqe_prepost_init(pool_time, (size_t)req_time.bytes,
                                  &cfg_time);
    p_freq = deepvqe_prepost_init(pool_freq, (size_t)req_freq.bytes,
                                  &cfg_freq);
    CHECK(p_time != NULL && p_freq != NULL);
    CHECK(deepvqe_prepost_io_mode(p_freq) == DEEPVQE_IO_FREQ);

    CHECK(drive_time(p_time, pcm_mic, pcm_far, out_a, HOPS, 1) == 0);

    /* In IO_FREQ the CALLER owns analysis, framing clock and synthesis --
     * one frame per hop, the same center=False convention the class uses. */
    memset(hist_mic, 0, sizeof hist_mic);
    memset(hist_far, 0, sizeof hist_far);
    CHECK(aiaec_synthesis_init(&synthesis, fft, window) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        static float mic_re[AIAEC_N_BINS], mic_im[AIAEC_N_BINS];
        static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
        static float enh_re[AIAEC_N_BINS], enh_im[AIAEC_N_BINS];
        float *slot = out_b + (size_t)hop * AIAEC_HOP;
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        int wrote;
        ref_roll(fft, hist_mic, pcm_mic + (size_t)hop * AIAEC_HOP,
                 mic_re, mic_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * AIAEC_HOP,
                 far_re, far_im);
        CHECK(deepvqe_prepost_pre_process_freq(p_freq, mic_re, mic_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p_freq, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_commit(p_freq) == 0);
        CHECK(deepvqe_prepost_post_process_freq(p_freq, enh_re, enh_im) == 0);
        wrote = aiaec_synthesis_push(&synthesis, enh_re, enh_im, slot);
        CHECK(wrote == (hop == 0 ? 0 : AIAEC_HOP));
        if (wrote < AIAEC_HOP)
            memset(slot + wrote, 0,
                   (size_t)(AIAEC_HOP - wrote) * sizeof(float));
    }

    /* Cross-mode calls are refused, not silently reinterpreted. */
    CHECK(deepvqe_prepost_pre_process(p_freq, pcm_mic, pcm_far) == -1);
    CHECK(deepvqe_prepost_post_process(p_freq, out_b, NULL) == -1);
    CHECK(deepvqe_prepost_pre_process_freq(p_time, pcm_mic, pcm_mic,
                                           pcm_far, pcm_far) == -1);
    CHECK(deepvqe_prepost_post_process_freq(p_time, out_b, out_b) == -1);

    CHECK(!all_zero(out_a, HOPS * AIAEC_HOP));
    CHECK(!identical(out_a, out_b + 1, HOPS * AIAEC_HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * AIAEC_HOP));

    deepvqe_prepost_destroy(p_time);
    deepvqe_prepost_destroy(p_freq);
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- freqpool: pool saving + the stale-pool gate --------------------- */

static int case_freqpool(FftHandle *fft) {
    DeepVqePrepostConfig cfg_time, cfg_freq, cfg_deep;
    DeepVqePrepostConfig cfg_time63, cfg_freq63;
    DeepVqePrepostMemReq req_time, req_freq, req_deep;
    DeepVqePrepostMemReq req_time63, req_freq63;
    void *pool_time, *pool_freq;

    CHECK(deepvqe_prepost_config_defaults(&cfg_time, DEEPVQE_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_config_defaults(&cfg_deep, DEEPVQE_IO_TIME,
                                          2 * D) == 0);
    cfg_deep.fft = fft;
    cfg_deep.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_time63, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_DEFAULT_D) == 0);
    cfg_time63.fft = fft;
    cfg_time63.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq63, DEEPVQE_IO_FREQ,
                                          DEEPVQE_PREPOST_DEFAULT_D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_deep, &req_deep) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time63, &req_time63) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq63, &req_freq63) == 0);

    printf("pool TIME D=%d  = %llu B\n"
           "pool FREQ D=%d  = %llu B   (saves %llu B, %.2f%%)\n"
           "pool TIME D=%d = %llu B\n"
           "pool FREQ D=%d = %llu B   (saves %llu B, %.2f%%)\n",
           D, (unsigned long long)req_time.bytes,
           D, (unsigned long long)req_freq.bytes,
           (unsigned long long)(req_time.bytes - req_freq.bytes),
           100.0 * (double)(req_time.bytes - req_freq.bytes) /
               (double)req_time.bytes,
           DEEPVQE_PREPOST_DEFAULT_D, (unsigned long long)req_time63.bytes,
           DEEPVQE_PREPOST_DEFAULT_D, (unsigned long long)req_freq63.bytes,
           (unsigned long long)(req_time63.bytes - req_freq63.bytes),
           100.0 * (double)(req_time63.bytes - req_freq63.bytes) /
               (double)req_time63.bytes);

    /* IO_FREQ must not pay for the framing machinery it never runs. */
    CHECK(req_freq.bytes < req_time.bytes);
    CHECK(req_freq63.bytes < req_time63.bytes);
    /* D is a pool-size parameter: the attention rings scale with it. */
    CHECK(req_time63.bytes > req_time.bytes);
    CHECK(req_freq.io_mode == (uint32_t)DEEPVQE_IO_FREQ);
    CHECK(req_time.io_mode == (uint32_t)DEEPVQE_IO_TIME);
    CHECK(req_freq.build_flags_hash != req_time.build_flags_hash);
    CHECK(req_deep.build_flags_hash != req_time.build_flags_hash);
    CHECK(req_time.descriptor_version == DEEPVQE_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_time.layout_version == DEEPVQE_PREPOST_LAYOUT_VERSION);
    CHECK(req_freq.descriptor_version == DEEPVQE_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_freq.layout_version == DEEPVQE_PREPOST_LAYOUT_VERSION);
    CHECK(req_time.reserved == 0u);
    CHECK(req_freq.reserved == 0u);
    CHECK(req_time.alignment == DEEPVQE_PREPOST_ALIGNMENT);

    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                  &cfg_time, &req_time) != NULL);
    CHECK(deepvqe_prepost_init_ex(pool_freq, (size_t)req_freq.bytes,
                                  &cfg_freq, &req_freq) != NULL);
    /* ...and refuse a pool recorded for another io_mode or another D. */
    CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                  &cfg_time, &req_freq) == NULL);
    CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                  &cfg_time, &req_deep) == NULL);
    {
        DeepVqePrepostMemReq stale;
        stale = req_time;
        ++stale.descriptor_version;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.layout_version;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.build_flags_hash;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.bytes;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        stale = req_time;
        ++stale.alignment;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        /* Unperturbed again, so the five refusals above are attributable to
         * the perturbation and not to the pool going stale on its own. */
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &req_time) != NULL);
    }
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(FftHandle *fft) {
    DeepVqePrepostConfig cfg, bad;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *heap, *stack;
    void *pool;

    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;

    heap = deepvqe_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(deepvqe_prepost_io_mode(heap) == DEEPVQE_IO_TIME);
    CHECK(deepvqe_prepost_hop_size(heap) == AIAEC_HOP);
    CHECK(deepvqe_prepost_num_bins(heap) == AIAEC_N_BINS);
    CHECK(deepvqe_prepost_descriptor(heap) != NULL);
    CHECK(deepvqe_prepost_descriptor(heap)->delay_depth == D);
    CHECK(deepvqe_prepost_descriptor(heap)->layout_version ==
          DEEPVQE_PREPOST_LAYOUT_VERSION);
    CHECK(drive_time(heap, pcm_mic, pcm_far, out_a, SHORT_HOPS, 1) == 0);
    CHECK(!all_zero(out_a, SHORT_HOPS * AIAEC_HOP));

    /* A created instance and a pool instance are the same object. */
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    stack = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(stack != NULL);
    CHECK(drive_time(stack, pcm_mic, pcm_far, out_b, SHORT_HOPS, 1) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * AIAEC_HOP));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still runs
     * afterwards. (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    deepvqe_prepost_destroy(stack);
    deepvqe_prepost_destroy(stack);
    deepvqe_prepost_reset(stack);
    memset(out_b, 0, sizeof out_b);
    CHECK(drive_time(stack, pcm_mic, pcm_far, out_b, SHORT_HOPS, 1) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * AIAEC_HOP));

    deepvqe_prepost_destroy(heap);
    deepvqe_prepost_destroy(NULL);
    free(pool);

    /* A heap instance in the mode that needs no transform at all. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) == 0);
    heap = deepvqe_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(deepvqe_prepost_io_mode(heap) == DEEPVQE_IO_FREQ);
    CHECK(deepvqe_prepost_descriptor(heap)->delay_depth == D);
    deepvqe_prepost_destroy(heap);

    /* _create refuses what _get_mem_size refuses, and allocates nothing. */
    CHECK(deepvqe_prepost_create(NULL) == NULL);
    CHECK(deepvqe_prepost_config_defaults(&bad, DEEPVQE_IO_TIME, D) == 0);
    CHECK(deepvqe_prepost_create(&bad) == NULL);   /* fft/window still NULL */
    bad.io_mode = 2;
    CHECK(deepvqe_prepost_create(&bad) == NULL);

    /* Accessors on NULL. */
    CHECK(deepvqe_prepost_hop_size(NULL) == -1);
    CHECK(deepvqe_prepost_num_bins(NULL) == -1);
    CHECK(deepvqe_prepost_io_mode(NULL) == -1);
    CHECK(deepvqe_prepost_descriptor(NULL) == NULL);
    deepvqe_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

static int case_reject(FftHandle *fft) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req, guard;
    void *wrong_mem = NULL;
    FftHandle *wrong_fft;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(deepvqe_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, NULL) == -1);
    CHECK(deepvqe_prepost_config_defaults(NULL, DEEPVQE_IO_TIME, D) == -1);

    /* Unknown io_mode. *req is left untouched on a refusal. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    cfg.io_mode = 2;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.io_mode = -1;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    CHECK(deepvqe_prepost_config_defaults(&cfg, 2, D) == -1);
    CHECK(deepvqe_prepost_config_defaults(&cfg, -1, D) == -1);

    /* D outside [MIN_D, MAX_D]. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    cfg.delay_depth = DEEPVQE_PREPOST_MIN_D - 1;
    req = guard;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.delay_depth = DEEPVQE_PREPOST_MAX_D + 1;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_MIN_D - 1) == -1);
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_MAX_D + 1) == -1);
    /* Both ends of the range are accepted. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_MIN_D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_MAX_D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);

    /* IO_TIME without a usable transform. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.window = window;
    cfg.fft = NULL;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.fft = fft;
    cfg.window = NULL;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);

    /* A correctly typed but WRONG-SIZE FftHandle: 129 bins, not 257. */
    wrong_fft = make_fft(AIAEC_N_FFT / 2, &wrong_mem);
    CHECK(wrong_fft != NULL);
    CHECK(fft_get_n_freqs(wrong_fft) != AIAEC_N_BINS);
    cfg.fft = wrong_fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(deepvqe_prepost_init(NULL, 0, &cfg) == NULL);

    /* IO_FREQ ignores fft/window entirely -- NULL is legal there. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(cfg.fft == NULL && cfg.window == NULL);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);

    /* _init: NULL pool, undersized pool, misaligned pool, then success. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes + req.alignment);
    CHECK(pool != NULL);
    CHECK(deepvqe_prepost_init(NULL, (size_t)req.bytes, &cfg) == NULL);
    CHECK(deepvqe_prepost_init(pool, (size_t)req.bytes - 1u, &cfg) == NULL);
    CHECK(deepvqe_prepost_init((unsigned char *)pool + 1,
                               (size_t)req.bytes, &cfg) == NULL);
    CHECK(deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg) != NULL);

    /* Stage calls on NULL, out of order, and after a partial write. */
    {
        DeepVqePrepost *p = deepvqe_prepost_init(pool, (size_t)req.bytes,
                                                 &cfg);
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        int written = -1;
        CHECK(p != NULL);
        CHECK(deepvqe_prepost_pre_process(NULL, pcm_mic, pcm_far) == -1);
        CHECK(deepvqe_prepost_pre_process(p, NULL, pcm_far) == -1);
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, NULL) == -1);
        CHECK(deepvqe_prepost_pre_process_freq(NULL, pcm_mic, pcm_mic,
                                               pcm_far, pcm_far) == -1);
        CHECK(deepvqe_prepost_post_process(NULL, out_a, NULL) == -1);
        CHECK(deepvqe_prepost_post_process(p, NULL, NULL) == -1);
        CHECK(deepvqe_prepost_post_process_freq(NULL, out_a, out_b) == -1);
        /* Nothing is open yet: the three frame stages all refuse. */
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_inputs(NULL, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_commit(NULL) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == -1);
        CHECK(deepvqe_prepost_frame_skip(NULL) == -1);
        /* With no frame open, post_process still emits a defined hop. */
        CHECK(deepvqe_prepost_post_process(p, out_a, &written) == 0);
        CHECK(written == 0);
        CHECK(all_zero(out_a, AIAEC_HOP));

        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, NULL) == -1);
        /* A frame is open: the hop cannot be emitted yet. */
        CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == -1);
        /* A partial accelerator write is refused and moves nothing; the
         * frame stays open so the caller can fail closed. */
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.taps[DEEPVQE_TAPS_ELEMENTS / 3] = NAN;
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == 0);

        /* The same refusal when the hole is in ONE state tensor rather than
         * in the taps -- the finite check walks the whole state, not just
         * the head. */
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_commit(p) == 0);   /* teeth: it can pass */
        CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.state_out[DEEPVQE_STATE_MIC3_HISTORY][5] = NAN;
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == 0);
    }

    free(pool);
    free(wrong_mem);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(FftHandle *fft) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    void *pool;

    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_TIME, D) == 0);
    cfg.fft = fft;
    cfg.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    CHECK(drive_time(p, pcm_mic, pcm_far, out_a, SHORT_HOPS, 1) == 0);
    /* Without the reset the second run continues from a warm state, so this
     * comparison has teeth: assert that it does NOT match first. */
    CHECK(drive_time(p, pcm_mic, pcm_far, out_b, SHORT_HOPS, 0) == 0);
    CHECK(!identical(out_a, out_b, SHORT_HOPS * AIAEC_HOP));

    deepvqe_prepost_reset(p);
    CHECK(drive_time(p, pcm_mic, pcm_far, out_b, SHORT_HOPS, 1) == 0);
    CHECK(identical(out_a, out_b, SHORT_HOPS * AIAEC_HOP));

    /* Reset mid-frame drops the open frame instead of stranding it. */
    {
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        deepvqe_prepost_reset(p);
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == -1);
        memset(out_b, 0, sizeof out_b);
        CHECK(drive_time(p, pcm_mic, pcm_far, out_b, SHORT_HOPS, 1) == 0);
        CHECK(identical(out_a, out_b, SHORT_HOPS * AIAEC_HOP));
    }

    deepvqe_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- skip: FAIL CLOSED, and no state or ring advance ----------------- */

static void make_frame(float *re, float *im, float seed) {
    int bin;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        re[bin] = seed + 0.5f - (float)bin / (float)AIAEC_N_BINS;
        im[bin] = (float)bin / (float)AIAEC_N_BINS - 0.25f - seed;
    }
}

/* One committed FREQ frame. */
static int commit_freq(DeepVqePrepost *p, const float *mic_re,
                       const float *mic_im, const float *far_re,
                       const float *far_im, float *out_re, float *out_im) {
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    if (deepvqe_prepost_pre_process_freq(p, mic_re, mic_im,
                                         far_re, far_im) != 1) return -1;
    if (deepvqe_prepost_frame_inputs(p, &inputs, &outputs) != 0) return -1;
    if (fake_run(&inputs, &outputs) != 0) return -1;
    if (deepvqe_prepost_frame_commit(p) != 0) return -1;
    if (deepvqe_prepost_post_process_freq(p, out_re, out_im) != 0) return -1;
    return 0;
}

static int case_skip(void) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    void *pool;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float f1_re[AIAEC_N_BINS], f1_im[AIAEC_N_BINS];
    static float f2_re[AIAEC_N_BINS], f2_im[AIAEC_N_BINS];
    static float f3_re[AIAEC_N_BINS], f3_im[AIAEC_N_BINS];
    static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
    static float got_re[AIAEC_N_BINS], got_im[AIAEC_N_BINS];
    float *snapshot[DEEPVQE_STATE_COUNT];
    size_t elements[DEEPVQE_STATE_COUNT];
    int id;
    size_t i;

    CHECK(strcmp(deepvqe_prepost_skip_policy_name(),
                 "mute_fail_closed") == 0);

    make_frame(f1_re, f1_im, 0.0f);
    make_frame(f2_re, f2_im, 0.125f);
    make_frame(f3_re, f3_im, -0.375f);
    make_frame(far_re, far_im, 0.0625f);

    /* IO_FREQ so the boundary is observable without the framing warm-up. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    /* Frame 1: a real commit, so the state banks are NON-ZERO below. */
    CHECK(commit_freq(p, f1_re, f1_im, far_re, far_im, got_re, got_im) == 0);
    /* Teeth for the mute assertion: a committed frame is NOT all zeros. */
    CHECK(!all_zero(got_re, AIAEC_N_BINS) || !all_zero(got_im, AIAEC_N_BINS));

    /* Frame 2: snapshot all sixteen state views, then take the fail-closed
     * identity even though the accelerator produced a COMPLETE, perfectly
     * committable result.
     *
     * The full write is the point. Skipping after a partial write would
     * prove nothing -- commit() refuses that frame on its own -- so a skip
     * that quietly committed would still look correct. Here the only thing
     * standing between the banks and a swap is frame_skip's contract. */
    CHECK(deepvqe_prepost_pre_process_freq(p, f2_re, f2_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    {
        int nonzero = 0;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            elements[id] = inputs.state_elements[id];
            CHECK(elements[id] == deepvqe_prepost_state_elements(id, D));
            snapshot[id] = (float *)malloc(elements[id] * sizeof(float));
            CHECK(snapshot[id] != NULL);
            memcpy(snapshot[id], inputs.state[id],
                   elements[id] * sizeof(float));
            for (i = 0; i < elements[id]; ++i)
                if (snapshot[id][i] != 0.0f) nonzero = 1;
        }
        /* "unchanged" proves nothing against an all-zero snapshot. */
        CHECK(nonzero);
    }
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_skip(p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* FAIL CLOSED: silence, NOT the microphone passed through. Passing the
     * raw microphone would emit the full uncancelled echo -- see the warning
     * at the top of deepvqe_prepost.h. */
    CHECK(all_zero(got_re, AIAEC_N_BINS));
    CHECK(all_zero(got_im, AIAEC_N_BINS));
    CHECK(!identical(got_re, f2_re, AIAEC_N_BINS));

    /* Frame 3: the state banks must be exactly what frame 2 saw. */
    CHECK(deepvqe_prepost_pre_process_freq(p, f3_re, f3_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        CHECK(inputs.state_elements[id] == elements[id]);
        CHECK(memcmp(inputs.state[id], snapshot[id],
                     elements[id] * sizeof(float)) == 0);
    }
    /* A skip WITHOUT frame_inputs behind it is legal: the accelerator may
     * have failed before it was ever handed the tensors. */
    CHECK(deepvqe_prepost_frame_skip(p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
    CHECK(all_zero(got_re, AIAEC_N_BINS));
    CHECK(all_zero(got_im, AIAEC_N_BINS));

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(snapshot[id]);
    deepvqe_prepost_destroy(p);
    free(pool);

    /* The CCM spectrum ring is HOST state and must not advance on a skip
     * either. Three instances fed the same frames:
     *   A: commit(f1), skip(f2),   commit(f3)
     *   B: commit(f1),             commit(f3)   -- f2 never presented
     *   C: commit(f1), commit(f2), commit(f3)
     * A must equal B (the skipped frame left no trace anywhere) and must
     * differ from C (a committed f2 does move both the banks and the ring),
     * so the equality is not measuring a constant. */
    {
        DeepVqePrepost *a, *b, *c;
        void *pool_a, *pool_b, *pool_c;
        static float a_re[AIAEC_N_BINS], a_im[AIAEC_N_BINS];
        static float b_re[AIAEC_N_BINS], b_im[AIAEC_N_BINS];
        static float c_re[AIAEC_N_BINS], c_im[AIAEC_N_BINS];

        pool_a = alloc_aligned(req.alignment, (size_t)req.bytes);
        pool_b = alloc_aligned(req.alignment, (size_t)req.bytes);
        pool_c = alloc_aligned(req.alignment, (size_t)req.bytes);
        CHECK(pool_a && pool_b && pool_c);
        a = deepvqe_prepost_init(pool_a, (size_t)req.bytes, &cfg);
        b = deepvqe_prepost_init(pool_b, (size_t)req.bytes, &cfg);
        c = deepvqe_prepost_init(pool_c, (size_t)req.bytes, &cfg);
        CHECK(a && b && c);

        CHECK(commit_freq(a, f1_re, f1_im, far_re, far_im,
                          a_re, a_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(a, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(a, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_skip(a) == 0);
        CHECK(deepvqe_prepost_post_process_freq(a, a_re, a_im) == 0);
        CHECK(commit_freq(a, f3_re, f3_im, far_re, far_im,
                          a_re, a_im) == 0);

        CHECK(commit_freq(b, f1_re, f1_im, far_re, far_im,
                          b_re, b_im) == 0);
        CHECK(commit_freq(b, f3_re, f3_im, far_re, far_im,
                          b_re, b_im) == 0);

        CHECK(commit_freq(c, f1_re, f1_im, far_re, far_im,
                          c_re, c_im) == 0);
        CHECK(commit_freq(c, f2_re, f2_im, far_re, far_im,
                          c_re, c_im) == 0);
        CHECK(commit_freq(c, f3_re, f3_im, far_re, far_im,
                          c_re, c_im) == 0);

        CHECK(!all_zero(a_re, AIAEC_N_BINS));
        CHECK(identical(a_re, b_re, AIAEC_N_BINS));
        CHECK(identical(a_im, b_im, AIAEC_N_BINS));
        CHECK(!identical(a_re, c_re, AIAEC_N_BINS));
        CHECK(!identical(a_im, c_im, AIAEC_N_BINS));

        deepvqe_prepost_destroy(a);
        deepvqe_prepost_destroy(b);
        deepvqe_prepost_destroy(c);
        free(pool_a);
        free(pool_b);
        free(pool_c);
    }
    return 0;
}

/* ---- guard: the three transaction gates ------------------------------ */

static int case_guard(FftHandle *fft) {
    DeepVqePrepostConfig cfg_time, cfg_freq;
    DeepVqePrepostMemReq req_time, req_freq;
    DeepVqePrepost *p, *q;
    void *pool_time, *pool_freq;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float f_re[AIAEC_N_BINS], f_im[AIAEC_N_BINS];
    static float g_re[AIAEC_N_BINS], g_im[AIAEC_N_BINS];
    int id;
    size_t i;

    CHECK(deepvqe_prepost_config_defaults(&cfg_time, DEEPVQE_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    CHECK(pool_time != NULL);
    p = deepvqe_prepost_init(pool_time, (size_t)req_time.bytes, &cfg_time);
    CHECK(p != NULL);

    /* (1) A hop is refused rather than stacked on top of an unfinished one,
     * and the refusal does not damage the frame that IS open. */
    CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
    CHECK(deepvqe_prepost_pre_process(p, pcm_mic + AIAEC_HOP,
                                      pcm_far + AIAEC_HOP) == -1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_commit(p) == 0);
    CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);

    /* (2) A commit with no frame_inputs behind it is refused, so an
     * accelerator that never ran cannot pass untouched buffers off as a
     * result -- and the same frame commits normally once it has. */
    CHECK(deepvqe_prepost_pre_process(p, pcm_mic + AIAEC_HOP,
                                      pcm_far + AIAEC_HOP) == 1);
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_commit(p) == 0);
    CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);

    /* (3) A refused commit DISARMS the transaction: re-committing without a
     * fresh frame_inputs must not succeed by the second walk happening to
     * find the same buffers. The fresh frame_inputs then re-fills EVERY
     * accelerator output with NaN, which is what makes the finite check a
     * partial-write detector rather than a stale-value detector. */
    CHECK(deepvqe_prepost_pre_process(p, pcm_mic + 2 * AIAEC_HOP,
                                      pcm_far + 2 * AIAEC_HOP) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    outputs.state_out[DEEPVQE_STATE_UP3_HISTORY][0] = NAN;
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    {
        size_t defined = 0;
        for (i = 0; i < outputs.taps_elements; ++i)
            if (!isnan(outputs.taps[i])) ++defined;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            for (i = 0; i < outputs.state_elements[id]; ++i)
                if (!isnan(outputs.state_out[id][i])) ++defined;
        CHECK(defined == 0u);
    }
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_commit(p) == 0);
    CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);

    deepvqe_prepost_destroy(p);
    free(pool_time);

    /* (1) again on the other side of the transform: pre_process_freq. */
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_freq != NULL);
    q = deepvqe_prepost_init(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(q != NULL);
    make_frame(f_re, f_im, 0.0f);
    make_frame(g_re, g_im, 0.25f);
    CHECK(deepvqe_prepost_pre_process_freq(q, f_re, f_im, g_re, g_im) == 1);
    CHECK(deepvqe_prepost_pre_process_freq(q, g_re, g_im, f_re, f_im) == -1);
    CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_commit(q) == 0);
    CHECK(deepvqe_prepost_post_process_freq(q, f_re, f_im) == 0);

    /* (2) again in IO_FREQ. */
    CHECK(deepvqe_prepost_pre_process_freq(q, g_re, g_im, f_re, f_im) == 1);
    CHECK(deepvqe_prepost_frame_commit(q) == -1);
    CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_commit(q) == 0);

    deepvqe_prepost_destroy(q);
    free(pool_freq);
    return 0;
}

/* ---- boundary: names, shapes, element counts, descriptor validation --- */

/* The twelve descriptor fields that are COMPILED CONSTANTS. delay_depth is
 * deliberately absent: it is the one export-time deployment parameter, so
 * validate() range-checks it instead of pinning it, and a +1 there is
 * legitimately accepted. */
static const char *const dv_constant_field_names[12] = {
    "layout_version", "sample_rate", "fft_size", "hop_size", "spectrum_bins",
    "time_order", "freq_taps", "conv_history_frames", "score_history_frames",
    "gru_layers", "gru_hidden", "state_tensor_count"
};

static int perturbation_refused(const DeepVqePrepostDescriptor *base,
                                int field) {
    DeepVqePrepostDescriptor probe = *base;
    switch (field) {
    case 0:  probe.layout_version += 1u;      break;
    case 1:  probe.sample_rate += 1;          break;
    case 2:  probe.fft_size += 1;             break;
    case 3:  probe.hop_size += 1;             break;
    case 4:  probe.spectrum_bins += 1;        break;
    case 5:  probe.time_order += 1;           break;
    case 6:  probe.freq_taps += 1;            break;
    case 7:  probe.conv_history_frames += 1;  break;
    case 8:  probe.score_history_frames += 1; break;
    case 9:  probe.gru_layers += 1;           break;
    case 10: probe.gru_hidden += 1;           break;
    case 11: probe.state_tensor_count += 1;   break;
    default: return 0;
    }
    if (deepvqe_prepost_descriptor_validate(&probe) == -1) return 1;
    fprintf(stderr, "descriptor field '%s' +1 was ACCEPTED\n",
            dv_constant_field_names[field]);
    return 0;
}

static int case_boundary(void) {
    DeepVqePrepostDescriptor base, probe;
    int dims[DEEPVQE_STATE_MAX_RANK];
    int id, field;

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        const char *name = deepvqe_prepost_state_name(id);
        const char *name_out = deepvqe_prepost_state_name_out(id);
        size_t length;
        int rank, axis;
        size_t elements = 1;

        CHECK(name != NULL);
        CHECK(name_out != NULL);
        length = strlen(name);
        /* The graph's handoff convention: the output tensor is the input
         * tensor's name plus "_out". */
        CHECK(strncmp(name_out, name, length) == 0);
        CHECK(strcmp(name_out + length, "_out") == 0);

        memset(dims, -1, sizeof dims);
        rank = deepvqe_prepost_state_shape(id, D, dims);
        if (id == DEEPVQE_STATE_H_GRU) {
            /* The one rank-3 tensor at this boundary. */
            CHECK(rank == 3);
            CHECK(dims[0] == 1 && dims[1] == 1 && dims[2] == 192);
            CHECK(dims[3] == 0);   /* entries beyond the shape are zeroed */
        } else {
            CHECK(rank == 4);
            CHECK(dims[0] == 1);
        }
        for (axis = 0; axis < rank; ++axis) {
            CHECK(dims[axis] > 0);
            elements *= (size_t)dims[axis];
        }
        CHECK(deepvqe_prepost_state_elements(id, D) == elements);
    }

    /* Bad ids and out-of-range depths, on all three accessors. */
    CHECK(deepvqe_prepost_state_name(-1) == NULL);
    CHECK(deepvqe_prepost_state_name(DEEPVQE_STATE_COUNT) == NULL);
    CHECK(deepvqe_prepost_state_name_out(-1) == NULL);
    CHECK(deepvqe_prepost_state_name_out(DEEPVQE_STATE_COUNT) == NULL);
    CHECK(deepvqe_prepost_state_shape(-1, D, dims) == -1);
    CHECK(deepvqe_prepost_state_shape(DEEPVQE_STATE_COUNT, D, dims) == -1);
    CHECK(deepvqe_prepost_state_shape(0, D, NULL) == -1);
    CHECK(deepvqe_prepost_state_elements(-1, D) == 0u);
    CHECK(deepvqe_prepost_state_elements(DEEPVQE_STATE_COUNT, D) == 0u);
    CHECK(deepvqe_prepost_state_shape(0, DEEPVQE_PREPOST_MIN_D - 1,
                                      dims) == -1);
    CHECK(deepvqe_prepost_state_shape(0, DEEPVQE_PREPOST_MAX_D + 1,
                                      dims) == -1);
    CHECK(deepvqe_prepost_state_elements(0, DEEPVQE_PREPOST_MIN_D - 1) == 0u);
    CHECK(deepvqe_prepost_state_elements(0, DEEPVQE_PREPOST_MAX_D + 1) == 0u);
    /* Both ends of the range still produce a shape. */
    CHECK(deepvqe_prepost_state_shape(0, DEEPVQE_PREPOST_MIN_D, dims) == 4);
    CHECK(deepvqe_prepost_state_shape(0, DEEPVQE_PREPOST_MAX_D, dims) == 4);

    /* The validator. */
    CHECK(deepvqe_prepost_descriptor_default(D, &base) == 0);
    CHECK(deepvqe_prepost_descriptor_validate(&base) == 0);
    CHECK(deepvqe_prepost_descriptor_default(D, NULL) == -1);
    CHECK(deepvqe_prepost_descriptor_default(DEEPVQE_PREPOST_MIN_D - 1,
                                             &base) == -1);
    CHECK(deepvqe_prepost_descriptor_default(DEEPVQE_PREPOST_MAX_D + 1,
                                             &base) == -1);
    CHECK(deepvqe_prepost_descriptor_default(D, &base) == 0);
    CHECK(deepvqe_prepost_descriptor_validate(NULL) == -1);
    probe = base;
    probe.delay_depth = DEEPVQE_PREPOST_MIN_D - 1;
    CHECK(deepvqe_prepost_descriptor_validate(&probe) == -1);
    probe = base;
    probe.delay_depth = DEEPVQE_PREPOST_MAX_D + 1;
    CHECK(deepvqe_prepost_descriptor_validate(&probe) == -1);

    for (field = 0; field < 12; ++field) {
        CHECK(perturbation_refused(&base, field));
        /* The unperturbed descriptor is accepted again, so the refusal
         * above is attributable to the perturbation. */
        CHECK(deepvqe_prepost_descriptor_validate(&base) == 0);
    }
    return 0;
}

/* ---- descriptor: validate + shape dump from argv ---------------------- */

static int case_descriptor(char **argv) {
    DeepVqePrepostDescriptor descriptor;
    int dims[DEEPVQE_STATE_MAX_RANK];
    int id;

    memset(&descriptor, 0, sizeof descriptor);
    descriptor.layout_version        = (uint32_t)strtoul(argv[2], NULL, 10);
    descriptor.delay_depth           = atoi(argv[3]);
    descriptor.sample_rate           = atoi(argv[4]);
    descriptor.fft_size              = atoi(argv[5]);
    descriptor.hop_size              = atoi(argv[6]);
    descriptor.spectrum_bins         = atoi(argv[7]);
    descriptor.time_order            = atoi(argv[8]);
    descriptor.freq_taps             = atoi(argv[9]);
    descriptor.conv_history_frames   = atoi(argv[10]);
    descriptor.score_history_frames  = atoi(argv[11]);
    descriptor.gru_layers            = atoi(argv[12]);
    descriptor.gru_hidden            = atoi(argv[13]);
    descriptor.state_tensor_count    = atoi(argv[14]);

    printf("validate=%d\n", deepvqe_prepost_descriptor_validate(&descriptor));
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        int rank;
        memset(dims, 0, sizeof dims);
        rank = deepvqe_prepost_state_shape(id, descriptor.delay_depth, dims);
        printf("%s %d %d %d %d %d\n", deepvqe_prepost_state_name(id), rank,
               dims[0], dims[1], dims[2], dims[3]);
    }
    return 0;
}

int main(int argc, char **argv) {
    void *fft_mem = NULL;
    FftHandle *fft;
    int status;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <case> [descriptor fields...]\n", argv[0]);
        return 2;
    }
    aiaec_make_window(window);
    fill_pcm(pcm_mic, pcm_far, HOPS * AIAEC_HOP);

    if (strcmp(argv[1], "descriptor") == 0) {
        if (argc != 15) {
            fprintf(stderr, "descriptor needs 13 fields, got %d\n", argc - 2);
            return 2;
        }
        return case_descriptor(argv);
    }

    fft = make_fft(AIAEC_N_FFT, &fft_mem);
    CHECK(fft != NULL);
    CHECK(fft_get_n_freqs(fft) == AIAEC_N_BINS);

    if (strcmp(argv[1], "equiv") == 0)          status = case_equiv(fft);
    else if (strcmp(argv[1], "freq") == 0)      status = case_freq(fft);
    else if (strcmp(argv[1], "freqpool") == 0)  status = case_freqpool(fft);
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle(fft);
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject(fft);
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset(fft);
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard(fft);
    else if (strcmp(argv[1], "boundary") == 0)  status = case_boundary();
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
    """One executable for every case: the class plus the three TUs behind it
    (aiaec_process.c is a wrapper over ulcnet_process.c, so both are needed),
    compiled at the house flags with -Werror."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('deepvqe_prepost_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c11',
         '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off',
         '-I', _AIAEC_DIR, '-I', _ULCNET_DIR, '-I', _DEEPVQE_DIR,
         '-I', _AC_INCLUDE, str(source), *_SOURCES,
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


def _run(driver, *args):
    done = subprocess.run([str(driver), *args], capture_output=True,
                          text=True)
    assert done.returncode == 0, (
        'case %s failed (rc=%d)\n%s' % (args[0], done.returncode, done.stderr))
    return done.stdout


# ---- the per-case gates ------------------------------------------------


def test_time_mode_matches_the_hand_composed_path(driver):
    """DEEPVQE_IO_TIME vs a rolling center=False analysis, two hand-held
    state banks, deepvqe_ccm_process and aiaec_synthesis_push wired up from
    ONLY the composed TUs -- byte-identical over 400 hops, with `written`
    following the synthesis half-window schedule (0, then one full hop).
    This is the gate that folding the composition into an object changed
    nothing at all."""
    assert 'io_mode=TIME' in _run(driver, 'equiv')


def test_freq_mode_matches_time_mode(driver):
    """DEEPVQE_IO_FREQ fed from the caller's OWN transform and re-synthesised
    by the caller must reproduce DEEPVQE_IO_TIME sample for sample: the two
    modes may differ in who owns the transform, never in the result. Every
    cross-mode call is refused rather than reinterpreted."""
    _run(driver, 'freq')


def test_freq_pool_is_smaller_and_stale_pool_gate_holds(driver):
    """IO_FREQ must not carve the framing machinery it never runs, D must
    move the pool size, and _init_ex must refuse a pool recorded for another
    mode, another D or a perturbed descriptor/layout/hash/size/alignment --
    while still accepting the matching one."""
    stdout = _run(driver, 'freqpool')
    assert 'pool FREQ' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe, and on a
    pool instance it is the repeatable no-op the contract promises."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size / _init refuse NULLs, an unknown io_mode, a D outside
    [MIN_D, MAX_D], IO_TIME without a usable transform, a wrong-size
    FftHandle, and an undersized or misaligned pool -- leaving *req
    untouched. The stage calls refuse being run out of order, and a partial
    accelerator write is refused whether the hole is in the taps or in one
    state tensor."""
    _run(driver, 'reject')


def test_reset_clears_state(driver):
    """N hops, reset, the same N hops again -> byte-identical. Asserted
    against a NO-reset second run first, so the test cannot pass by the
    output being state-independent. A reset mid-frame drops the open frame
    rather than stranding it."""
    _run(driver, 'reset')


def test_frame_skip_mutes_and_freezes_the_state_and_the_ring(driver):
    """FAIL CLOSED. DeepVQE-S's stream 0 is the RAW microphone, so the
    pass-through identity a post-filter takes would emit the full uncancelled
    echo; _frame_skip emits SILENCE instead and the policy names itself
    "mute_fail_closed". It also freezes both halves of the model's time: the
    sixteen state banks do not swap, and the CCM spectrum ring does not
    advance -- proved by an instance that skipped a frame agreeing with one
    that never saw it, and differing from one that committed it."""
    _run(driver, 'skip')


def test_transaction_guards(driver):
    """The three gates that make the boundary a transaction: a second
    pre_process while a frame is open is refused without damaging that frame,
    a commit with no frame_inputs behind it is refused (an accelerator that
    never ran cannot pass untouched buffers off as a result), and a refused
    commit disarms until a fresh frame_inputs -- which re-fills every tap and
    every state_out element with NaN."""
    _run(driver, 'guard')


def test_state_boundary_names_shapes_and_descriptor_validation(driver):
    """The explicit boundary: sixteen named state tensors whose output name
    is the input name plus "_out", rank 4 everywhere except h_gru's rank-3
    [1,1,192], element counts that are the product of the published dims, and
    a descriptor validator that refuses each of the twelve compiled-constant
    fields when perturbed by one -- while still accepting the unperturbed
    descriptor, and still range-checking (not pinning) delay_depth."""
    _run(driver, 'boundary')


# ---- the Python/C boundary agreement -----------------------------------


def test_layout_version_matches_c_header():
    """The C header is the source of truth for DEEPVQE_PREPOST_LAYOUT_VERSION
    and _streaming_export.py writes it into every artifact. If the two drift,
    a board refuses a graph it can bind or -- far worse -- binds one it
    cannot. A source-text check, kept as the cheap no-torch gate;
    test_c_descriptor_matches_the_built_graph proves the same thing against
    the COMPILED constant through the C validator."""
    from AIAEC._streaming_export import DEEPVQE_C_LAYOUT_VERSION

    with open(_PREPOST_HEADER, encoding='utf-8') as stream:
        text = stream.read()
    found = re.findall(
        r'^#define\s+DEEPVQE_PREPOST_LAYOUT_VERSION\s+(\d+)u?\s*$',
        text, re.MULTILINE)
    assert len(found) == 1, found
    assert int(found[0]) == DEEPVQE_C_LAYOUT_VERSION


@pytest.fixture(scope='module')
def deepvqe_built():
    """The streaming export wrapper for a freshly constructed DeepVQE-S, plus
    the descriptor MEASURED off it. Module-scoped: three tests read it and
    building the graph twice would prove nothing extra."""
    torch = pytest.importorskip('torch')
    from AIAEC.aiaec_common import SignalGrid
    from AIAEC.DeepVQE_S import DeepVQES
    from AIAEC._streaming_export import _build, c_descriptor

    grid = SignalGrid(16000, 512, 512, 256)
    torch.manual_seed(0)
    model = DeepVQES(grid).eval()
    built = _build('DeepVQE_S', model)
    with torch.no_grad():
        outputs = built[0](*built[1])
    return model, built, c_descriptor('DeepVQE_S', built, outputs)


def _descriptor_argv(descriptor):
    return [str(int(descriptor[key])) for key in _DESCRIPTOR_KEYS]


def _parse_shapes(stdout):
    lines = stdout.strip().splitlines()
    assert lines[0].startswith('validate='), stdout
    rows = []
    for line in lines[1:]:
        parts = line.split()
        assert len(parts) == 6, line
        rank = int(parts[1])
        dims = tuple(int(value) for value in parts[2:])
        rows.append((parts[0], rank, dims))
    return lines[0], rows


def test_c_descriptor_matches_the_built_graph(driver, deepvqe_built):
    """The C class's compiled ABI against the graph the exporter actually
    builds: the descriptor measured from the boundary tensors must validate,
    and every one of the sixteen state tensors must have the same NAME, in
    the same ORDER, with the same SHAPE on both sides. A channel-schedule or
    frequency-ladder change in model.py surfaces here rather than as a
    silently wrong binding on the board."""
    model, built, descriptor = deepvqe_built
    _wrapper, inputs, input_names, _output_names, _split = built

    stdout = _run(driver, 'descriptor', *_descriptor_argv(descriptor))
    verdict, rows = _parse_shapes(stdout)
    assert verdict == 'validate=0', stdout

    state_names = input_names[2:]
    assert len(rows) == len(state_names) == 16
    for index, (name, rank, dims) in enumerate(rows):
        assert name == state_names[index], (index, name, state_names[index])
        assert dims[:rank] == tuple(inputs[2 + index].shape)

    assert descriptor['delay_depth'] == model.max_delay_frames == 63
    assert descriptor['state_tensor_count'] == 16


def test_export_metadata_carries_the_c_descriptor(driver, deepvqe_built,
                                                  tmp_path):
    """A shipped artifact must carry the boundary its C side binds. Both the
    ONNX metadata and the JSON written beside it must record
    state_layout_version 2 and the measured c_descriptor -- the very dict
    test_c_descriptor_matches_the_built_graph already fed to the C
    validator, so nothing between the graph and the board restates the ABI
    by hand."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    torch = pytest.importorskip('torch')
    from AIAEC.dataset_gen import AecGrid
    from AIAEC._streaming_export import (
        DEEPVQE_C_LAYOUT_VERSION,
        export_graph,
    )

    _model, built, descriptor = deepvqe_built
    grid = AecGrid(16000, 512, 512, 256)
    checkpoint = tmp_path / 'deepvqe_s.pth'
    checkpoint.write_bytes(b'not a real checkpoint; only hashed')
    output = tmp_path / 'graph.onnx'

    with torch.no_grad():
        metadata = export_graph(grid, built, str(checkpoint), str(output),
                                63, verify=False)

    assert metadata['state_layout_version'] == DEEPVQE_C_LAYOUT_VERSION
    assert metadata['c_descriptor'] == descriptor
    assert metadata['output_schema']['output'] == [1, 1, 257, 18]

    with open(str(output.with_suffix('.json')), encoding='utf-8') as stream:
        written = json.load(stream)
    assert written['state_layout_version'] == DEEPVQE_C_LAYOUT_VERSION
    assert written['c_descriptor'] == descriptor
    assert written['output_schema']['output'] == [1, 1, 257, 18]
