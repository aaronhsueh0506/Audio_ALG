"""Equivalence and contract gates for the DeepVQE-S pre/post class.

``DeepVQE_S/deepvqe_prepost.c`` owns no signal processing: it composes two
parity-tested translation units (``aiaec_process.c`` for the 16 kHz sqrt-Hann
STFT/WOLA, a thin wrapper over ``ulcnet_process.c``, and
``DeepVQE_S/deepvqe_process.c`` for the CCM kernel) behind one object, so the
gate is that the class reproduces the hand-composed path SAMPLE FOR SAMPLE:
every numeric comparison is ``memcmp`` on fp32, and every "identical" claim is
paired with a "not trivially identical" one so no gate passes on silence.

Cases, all in ONE driver selected by argv[1] (audiolib and the four model TUs
are compiled once for the module):

  equiv      DEEPVQE_IO_TIME vs a rolling analysis + one hand-held in-place
             state buffer set + deepvqe_ccm_process + a hand-built AcIstft
  freq       DEEPVQE_IO_FREQ driven by the caller's own transform vs IO_TIME
  freqpool   the FREQ pool is smaller; the stale-pool gate accepts the matching
             MemReq and refuses every mismatched one
  lifecycle  _create/_destroy against _init on a caller pool
  reject     _get_mem_size / _init / the stage calls, reject-first
  reset      _reset really clears state (a re-run reproduces the first run)
  skip       _frame_skip MUTES (fail-closed), leaves the state as the
             accelerator wrote it and does not advance the CCM ring
  inplace    one buffer per state tensor (input and output are the same pointer,
             stable for the instance's life), only the head NaN-prefilled, and a
             non-finite write to the newest frame of any state tensor zeroes all
             sixteen and leaves the next frame healthy
  guard      no double pre_process, no commit without frame_inputs, no re-commit
             after a refused one
  newest     commit checks the newest frame of every window and h_gru whole; a
             non-finite value in an older or dropped frame is not seen
  generation state_generation starts at 0 and advances on reset and on a
             refused commit, and on nothing else
  boundary   state names, shapes (all rank 4), element counts and the descriptor
             validator's 12 refusable fields
  descriptor a shape/validate dump driven from argv, so the Python side can feed
             it a descriptor MEASURED from the built graph

The accelerator stand-in is deterministic and input-dependent (it sums both
signal inputs and all sixteen state tensors and writes its next state back in
place), so a state handoff that failed to happen diverges rather than agrees.

THE SKIP POLICY IS DELIBERATE: stream 0 is the RAW microphone, not a residual,
so a pass-through would emit the full uncancelled echo; ``frame_skip`` emits
SILENCE and this suite asserts the mute.
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
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
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
    'conv_window_frames',
    'score_window_frames',
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
 * so a state handoff that did not happen (or happened when it should not
 * have) changes every later frame instead of going unnoticed.
 *
 * state_out aliases state: the whole old state is reduced into `acc` before
 * the first output element is written, so every state input is read before any
 * state output is written, as the in-place contract requires. */
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

/* The reference synthesis object, built the way a host builds it: the shared
 * handle and window table in the config, one aligned block. */
static AcIstft *make_synthesis(FftHandle *fft) {
    AcStftConfig sc;
    size_t bytes;
    ac_stft_config_default(&sc, AIAEC_N_FFT);
    sc.ola = AC_OLA_ENV_NORM;
    sc.fft = fft;
    sc.window = window;
    bytes = ac_istft_get_mem_size(&sc);
    return bytes ? ac_istft_init(alloc_aligned(16u, bytes), bytes, &sc) : NULL;
}

/* The canonical per-hop loop from deepvqe_prepost.h's contract block.
 * `cold` asserts the synthesis warm-up schedule: the first frame lands
 * entirely inside the trimmed half-window, so `written` is 0 on hop 0 and
 * AIAEC_HOP from then on. A run continued on a WARM instance emits a full
 * hop immediately, which is why the schedule is a parameter and not a
 * constant. The accelerator writes the pool's own pointers (state_out ==
 * state). */
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
        if (deepvqe_prepost_post_process(
                p, out + (size_t)hop * AIAEC_HOP, &written) != 0) return -1;
        if (written != expect) return -1;
    }
    return 0;
}

/* Where commit looks. Every state tensor is [1,C,T,F]: a ring (attention
 * key/value) is newest-first, so its newest frame is slot 0; a conv or score
 * window is oldest-first, so its newest frame is slot T-1. h_gru is checked
 * whole. These helpers locate elements by direct indexing on the published
 * shape, independent of the class's own loop. */
static int is_ring(int id) {
    return id == DEEPVQE_STATE_ALIGN_KEY_RING ||
           id == DEEPVQE_STATE_ALIGN_VALUE_RING;
}

static size_t frame_element(int id, size_t channel, size_t frame, size_t bin) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    if (deepvqe_prepost_state_shape(id, D, dims) != 4) return 0;
    return ((channel * (size_t)dims[2]) + frame) * (size_t)dims[3] + bin;
}

static size_t newest_frame(int id) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    if (is_ring(id)) return 0u;
    if (deepvqe_prepost_state_shape(id, D, dims) != 4) return 0;
    return (size_t)dims[2] - 1u;
}

/* An element the commit check reads: bin 0 of the newest frame of channel 0
 * (`last` == 0) or the last bin of the newest frame of the last channel. */
static size_t newest_element(int id, int last) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    if (id == DEEPVQE_STATE_H_GRU)
        return last ? deepvqe_prepost_state_elements(id, D) - 1u : 0u;
    if (deepvqe_prepost_state_shape(id, D, dims) != 4) return 0;
    return last ? frame_element(id, (size_t)dims[1] - 1u, newest_frame(id),
                                (size_t)dims[3] - 1u)
                : frame_element(id, 0u, newest_frame(id), 0u);
}

/* An element the commit check does NOT read: bin 0 of the oldest frame of
 * channel 0 -- the slot a window drops (slot D-1 of a ring, frame 0 of a
 * conv or score window). */
static size_t dropped_element(int id) {
    int dims[DEEPVQE_STATE_MAX_RANK];
    if (deepvqe_prepost_state_shape(id, D, dims) != 4) return 0;
    return frame_element(id, 0u, is_ring(id) ? (size_t)dims[2] - 1u : 0u, 0u);
}

/* center=False rolling analysis: one frame per hop over the last N_FFT
 * samples. Deliberately expression-for-expression the same as
 * audio_stft.c's rolling analysis (the transform behind the AcStft the class
 * drives), so this reference is independent of deepvqe_prepost.c yet
 * bit-comparable with it. */
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

/* The model's compressed_ri_feature for one spectrum, bin by bin and in the
 * same fp32 operations the class runs, planar [real plane, imaginary plane].
 * case_compression checks the formula itself against double precision. */
static void compress_ref(const float *re, const float *im, float *ri) {
    int bin;
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        const float mag = sqrtf(re[bin] * re[bin] + im[bin] * im[bin] + 1e-12f);
        const float scale = powf(mag, DEEPVQE_COMPRESSION_EXP - 1.0f);
        ri[bin] = re[bin] * scale;
        ri[AIAEC_N_BINS + bin] = im[bin] * scale;
    }
}

/* ---- compression: the host front end vs double precision -------------- */

static int case_compression(void) {
    static float mic_re[AIAEC_N_BINS], mic_im[AIAEC_N_BINS];
    static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    DeepVqePrepost *p;
    void *pool;
    int bin, stream;

    /* Zero, denormal-scale, ordinary, negative and very large bins: the
     * 1e-12 floor under the sqrt is what keeps a silent bin finite. */
    for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
        const float sign = (bin % 3 == 0) ? -1.0f : 1.0f;
        mic_re[bin] = sign * (float)bin * 0.37f;
        mic_im[bin] = -sign * (float)(AIAEC_N_BINS - bin) * 0.011f;
        far_re[bin] = (bin % 5 == 0) ? 0.0f : sign * 1e-7f * (float)bin;
        far_im[bin] = (bin % 7 == 0) ? 0.0f : sign * 3000.0f / (float)(bin + 1);
    }
    mic_re[10] = mic_im[10] = 0.0f;
    far_re[3] = 1.0e4f;

    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(deepvqe_prepost_pre_process_freq(p, mic_re, mic_im, far_re,
                                           far_im) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(inputs.spectrum_ri_elements == 2u * (size_t)AIAEC_N_BINS);

    for (stream = 0; stream < 2; ++stream) {
        const float *re = stream ? far_re : mic_re;
        const float *im = stream ? far_im : mic_im;
        const float *got = stream ? inputs.far : inputs.mic;
        for (bin = 0; bin < AIAEC_N_BINS; ++bin) {
            const double power = (double)re[bin] * (double)re[bin] +
                                 (double)im[bin] * (double)im[bin] + 1e-12;
            const double scale = pow(sqrt(power),
                                     (double)DEEPVQE_COMPRESSION_EXP - 1.0);
            const double want_re = (double)re[bin] * scale;
            const double want_im = (double)im[bin] * scale;
            CHECK(fabs((double)got[bin] - want_re) <=
                  4e-6 * fabs(want_re) + 1e-30);
            CHECK(fabs((double)got[AIAEC_N_BINS + bin] - want_im) <=
                  4e-6 * fabs(want_im) + 1e-30);
        }
    }
    deepvqe_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- equiv: the class vs the hand-composed reference path ------------ */

static int case_equiv(FftHandle *fft) {
    /* Path A -- the two composed TUs plus one in-place state buffer set held
     * by hand.
     * deepvqe_prepost.c contributes nothing here except the pure shape
     * helper, so this reference cannot be circular. */
    static float hist_mic[AIAEC_N_FFT], hist_far[AIAEC_N_FFT];
    AcIstft *synthesis;
    static DeepVqeCcmState ccm;
    static float mic_ri[2 * AIAEC_N_BINS], far_ri[2 * AIAEC_N_BINS];
    static float taps_ref[DEEPVQE_TAPS_ELEMENTS];
    float *state[DEEPVQE_STATE_COUNT];
    size_t elements[DEEPVQE_STATE_COUNT];
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *p;
    void *pool;
    int id, hop;

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        elements[id] = deepvqe_prepost_state_elements(id, D);
        CHECK(elements[id] > 0u);
        state[id] = (float *)calloc(elements[id], sizeof(float));
        CHECK(state[id] != NULL);
    }
    memset(hist_mic, 0, sizeof hist_mic);
    memset(hist_far, 0, sizeof hist_far);
    deepvqe_ccm_init(&ccm);
    synthesis = make_synthesis(fft);
    CHECK(synthesis != NULL);

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
        compress_ref(mic_re, mic_im, mic_ri);
        compress_ref(far_re, far_im, far_ri);

        memset(&inputs, 0, sizeof inputs);
        memset(&outputs, 0, sizeof outputs);
        inputs.mic = mic_ri;
        inputs.far = far_ri;
        inputs.spectrum_ri_elements = 2u * (size_t)AIAEC_N_BINS;
        outputs.taps = taps_ref;
        outputs.taps_elements = DEEPVQE_TAPS_ELEMENTS;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            inputs.state[id] = state[id];
            inputs.state_elements[id] = elements[id];
            outputs.state_out[id] = state[id];   /* in place */
            outputs.state_elements[id] = elements[id];
        }
        CHECK(fake_run(&inputs, &outputs) == 0);

        deepvqe_ccm_process(
            &ccm, mic_re, mic_im,
            (const float (*)[DEEPVQE_TIME_ORDER][DEEPVQE_FREQ_TAPS][2])
                taps_ref,
            enh_re, enh_im);
        wrote = ac_istft_push_planar(synthesis, enh_re, enh_im, slot);
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
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(state[id]);
    return 0;
}

/* ---- freq: IO_FREQ on the caller's own transform vs IO_TIME ---------- */

static int case_freq(FftHandle *fft) {
    DeepVqePrepostConfig cfg_time, cfg_freq;
    DeepVqePrepostMemReq req_time, req_freq;
    DeepVqePrepost *p_time, *p_freq;
    void *pool_time, *pool_freq;
    static float hist_mic[AIAEC_N_FFT], hist_far[AIAEC_N_FFT];
    AcIstft *synthesis;
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
    synthesis = make_synthesis(fft);
    CHECK(synthesis != NULL);
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
        wrote = ac_istft_push_planar(synthesis, enh_re, enh_im, slot);
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
    DeepVqePrepostConfig cfg_time_default, cfg_freq_default;
    DeepVqePrepostMemReq req_time, req_freq, req_deep;
    DeepVqePrepostMemReq req_time_default, req_freq_default;
    void *pool_time, *pool_freq;

    CHECK(deepvqe_prepost_config_defaults(&cfg_time, DEEPVQE_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_config_defaults(&cfg_deep, DEEPVQE_IO_TIME,
                                          2 * D) == 0);
    cfg_deep.fft = fft;
    cfg_deep.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_time_default, DEEPVQE_IO_TIME,
                                          DEEPVQE_PREPOST_DEFAULT_D) == 0);
    cfg_time_default.fft = fft;
    cfg_time_default.window = window;
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq_default, DEEPVQE_IO_FREQ,
                                          DEEPVQE_PREPOST_DEFAULT_D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_deep, &req_deep) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time_default, &req_time_default) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq_default, &req_freq_default) == 0);

    printf("pool TIME D=%d  = %llu B\n"
           "pool FREQ D=%d  = %llu B   (saves %llu B, %.2f%%)\n"
           "pool TIME D=%d = %llu B\n"
           "pool FREQ D=%d = %llu B   (saves %llu B, %.2f%%)\n",
           D, (unsigned long long)req_time.bytes,
           D, (unsigned long long)req_freq.bytes,
           (unsigned long long)(req_time.bytes - req_freq.bytes),
           100.0 * (double)(req_time.bytes - req_freq.bytes) /
               (double)req_time.bytes,
           DEEPVQE_PREPOST_DEFAULT_D, (unsigned long long)req_time_default.bytes,
           DEEPVQE_PREPOST_DEFAULT_D, (unsigned long long)req_freq_default.bytes,
           (unsigned long long)(req_time_default.bytes - req_freq_default.bytes),
           100.0 * (double)(req_time_default.bytes - req_freq_default.bytes) /
               (double)req_time_default.bytes);

    /* IO_FREQ must not pay for the framing machinery it never runs: the
     * difference is exactly the output hop staging and the three framing
     * objects, sized here by audiolib with the caller's handle and
     * window. */
    {
        AcStftConfig sc;
        size_t a, s;
        ac_stft_config_default(&sc, AIAEC_N_FFT);
        sc.ola = AC_OLA_ENV_NORM;
        sc.fft = fft;
        sc.window = window;
        a = ac_stft_get_mem_size(&sc);
        s = ac_istft_get_mem_size(&sc);
        CHECK(a != 0 && s != 0);
#define A16(x) (((size_t)(x) + 15u) / 16u * 16u)
        CHECK(req_time.bytes - req_freq.bytes ==
              A16((size_t)AIAEC_HOP * sizeof(float)) + 2u * A16(a) + A16(s));
#undef A16
    }

    /* ONE copy of the state per tensor: doubling D grows the pool by the
     * state's own growth (alignment padding aside), not by twice that. */
    {
        size_t state_d = 0, state_2d = 0;
        int sid;
        for (sid = 0; sid < DEEPVQE_STATE_COUNT; ++sid) {
            state_d += deepvqe_prepost_state_elements(sid, D) * sizeof(float);
            state_2d +=
                deepvqe_prepost_state_elements(sid, 2 * D) * sizeof(float);
        }
        CHECK(state_2d > state_d);
        CHECK(req_deep.bytes - req_time.bytes >= state_2d - state_d);
        CHECK(req_deep.bytes - req_time.bytes <
              state_2d - state_d + 16u * DEEPVQE_STATE_COUNT);
    }

    /* IO_FREQ must not pay for the framing machinery it never runs. */
    CHECK(req_freq.bytes < req_time.bytes);
    CHECK(req_freq_default.bytes < req_time_default.bytes);
    /* D is a pool-size parameter: the attention rings scale with it. */
    CHECK(req_time_default.bytes > req_time.bytes);
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
        stale = req_time;
        stale.io_mode = (uint32_t)DEEPVQE_IO_FREQ;
        CHECK(deepvqe_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                      &cfg_time, &stale) == NULL);
        /* Unperturbed again, so the six refusals above are attributable to
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
        int id;
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
        /* Teeth: the state is non-zero going in, so "zero" below is the
         * commit's doing. */
        CHECK(!all_zero(outputs.state_out[DEEPVQE_STATE_H_GRU],
                        (int)outputs.state_elements[DEEPVQE_STATE_H_GRU]));
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == 0);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id],
                           (int)outputs.state_elements[id]));

        /* The same refusal when the hole is in the newest frame of ONE state
         * tensor rather than in the taps -- the finite check walks the state,
         * not just the head. */
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_commit(p) == 0);   /* teeth: it can pass */
        CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic, pcm_far) == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.state_out[DEEPVQE_STATE_MIC3_HISTORY][
            newest_element(DEEPVQE_STATE_MIC3_HISTORY, 0) + 5u] = NAN;
        CHECK(deepvqe_prepost_frame_commit(p) == -1);
        CHECK(deepvqe_prepost_frame_skip(p) == 0);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id],
                           (int)outputs.state_elements[id]));
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
    float *before[DEEPVQE_STATE_COUNT];    /* state the accelerator read   */
    float *written[DEEPVQE_STATE_COUNT];   /* state the accelerator wrote  */
    size_t elements[DEEPVQE_STATE_COUNT];
    int id;
    int moved;
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

    /* Frame 1: a real commit, so the state tensors are NON-ZERO below. */
    CHECK(commit_freq(p, f1_re, f1_im, far_re, far_im, got_re, got_im) == 0);
    /* Teeth for the mute assertion: a committed frame is NOT all zeros. */
    CHECK(!all_zero(got_re, AIAEC_N_BINS) || !all_zero(got_im, AIAEC_N_BINS));

    /* Frame 2: the accelerator produces a COMPLETE, perfectly committable
     * result, and the host takes the fail-closed identity anyway.
     *
     * The full write is the point. Skipping after a partial write would
     * prove nothing -- commit() refuses that frame on its own -- so a skip
     * that quietly committed would still look correct. Here the only thing
     * standing between the run and a commit is frame_skip's contract. The
     * state is in place, so it is whatever the accelerator wrote: skip does
     * not undo it. */
    CHECK(deepvqe_prepost_pre_process_freq(p, f2_re, f2_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    {
        int nonzero = 0;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            elements[id] = inputs.state_elements[id];
            CHECK(elements[id] == deepvqe_prepost_state_elements(id, D));
            before[id] = (float *)malloc(elements[id] * sizeof(float));
            written[id] = (float *)malloc(elements[id] * sizeof(float));
            CHECK(before[id] != NULL && written[id] != NULL);
            memcpy(before[id], inputs.state[id],
                   elements[id] * sizeof(float));
            for (i = 0; i < elements[id]; ++i)
                if (before[id][i] != 0.0f) nonzero = 1;
        }
        /* "unchanged" proves nothing against an all-zero snapshot. */
        CHECK(nonzero);
    }
    CHECK(fake_run(&inputs, &outputs) == 0);
    moved = 0;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        memcpy(written[id], outputs.state_out[id],
               elements[id] * sizeof(float));
        if (memcmp(written[id], before[id],
                   elements[id] * sizeof(float)) != 0) moved = 1;
    }
    CHECK(moved);   /* the run really changed the state */
    CHECK(deepvqe_prepost_frame_skip(p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* FAIL CLOSED: silence, NOT the microphone passed through. Passing the
     * raw microphone would emit the full uncancelled echo -- see the warning
     * at the top of deepvqe_prepost.h. */
    CHECK(all_zero(got_re, AIAEC_N_BINS));
    CHECK(all_zero(got_im, AIAEC_N_BINS));
    CHECK(!identical(got_re, f2_re, AIAEC_N_BINS));

    /* Frame 3: the state is exactly what the accelerator wrote in frame 2
     * -- the skip neither rolled it back nor zeroed it. */
    CHECK(deepvqe_prepost_pre_process_freq(p, f3_re, f3_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        CHECK(inputs.state_elements[id] == elements[id]);
        CHECK(memcmp(inputs.state[id], written[id],
                     elements[id] * sizeof(float)) == 0);
    }
    /* A skip WITHOUT frame_inputs behind it is legal: the accelerator may
     * have failed before it was ever handed the tensors. */
    CHECK(deepvqe_prepost_frame_skip(p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
    CHECK(all_zero(got_re, AIAEC_N_BINS));
    CHECK(all_zero(got_im, AIAEC_N_BINS));

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        free(before[id]);
        free(written[id]);
    }
    deepvqe_prepost_destroy(p);
    free(pool);

    /* The CCM spectrum ring is HOST state and must not advance on a skip
     * either. Four instances fed the same frames:
     *   A: commit(f1), run + skip(f2), commit(f3)  -- accelerator wrote f2
     *   D: commit(f1), skip(f2),       commit(f3)  -- accelerator wrote nothing
     *   B: commit(f1),                 commit(f3)  -- f2 never presented
     *   C: commit(f1), commit(f2),     commit(f3)
     * D must equal B: a skip after a failed run leaves no trace in the state
     * or in the ring (a ring that advanced on skip would split them). A must
     * differ from B: the state is in place, so a skip keeps what the
     * accelerator wrote. Both must differ from C, so the equality is not
     * measuring a constant. */
    {
        DeepVqePrepost *a, *b, *c, *d;
        void *pool_a, *pool_b, *pool_c, *pool_d;
        static float a_re[AIAEC_N_BINS], a_im[AIAEC_N_BINS];
        static float b_re[AIAEC_N_BINS], b_im[AIAEC_N_BINS];
        static float c_re[AIAEC_N_BINS], c_im[AIAEC_N_BINS];
        static float d_re[AIAEC_N_BINS], d_im[AIAEC_N_BINS];

        pool_a = alloc_aligned(req.alignment, (size_t)req.bytes);
        pool_b = alloc_aligned(req.alignment, (size_t)req.bytes);
        pool_c = alloc_aligned(req.alignment, (size_t)req.bytes);
        pool_d = alloc_aligned(req.alignment, (size_t)req.bytes);
        CHECK(pool_a && pool_b && pool_c && pool_d);
        a = deepvqe_prepost_init(pool_a, (size_t)req.bytes, &cfg);
        b = deepvqe_prepost_init(pool_b, (size_t)req.bytes, &cfg);
        c = deepvqe_prepost_init(pool_c, (size_t)req.bytes, &cfg);
        d = deepvqe_prepost_init(pool_d, (size_t)req.bytes, &cfg);
        CHECK(a && b && c && d);

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

        CHECK(commit_freq(d, f1_re, f1_im, far_re, far_im,
                          d_re, d_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(d, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(d, &inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_skip(d) == 0);   /* no fake_run */
        CHECK(deepvqe_prepost_post_process_freq(d, d_re, d_im) == 0);
        CHECK(commit_freq(d, f3_re, f3_im, far_re, far_im,
                          d_re, d_im) == 0);

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

        CHECK(!all_zero(d_re, AIAEC_N_BINS));
        CHECK(identical(d_re, b_re, AIAEC_N_BINS));
        CHECK(identical(d_im, b_im, AIAEC_N_BINS));
        CHECK(!identical(a_re, b_re, AIAEC_N_BINS));
        CHECK(!identical(a_re, c_re, AIAEC_N_BINS));
        CHECK(!identical(a_im, c_im, AIAEC_N_BINS));
        CHECK(!identical(d_re, c_re, AIAEC_N_BINS));
        CHECK(!identical(d_im, c_im, AIAEC_N_BINS));

        deepvqe_prepost_destroy(a);
        deepvqe_prepost_destroy(b);
        deepvqe_prepost_destroy(c);
        deepvqe_prepost_destroy(d);
        free(pool_a);
        free(pool_b);
        free(pool_c);
        free(pool_d);
    }
    return 0;
}

/* ---- inplace: one buffer per state tensor, validate-only commit ------- */

static int state_all_zero(const DeepVqePrepostInputs *in) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (!all_zero(in->state[id], (int)in->state_elements[id])) return 0;
    return 1;
}

static int case_inplace(FftHandle *fft) {
    DeepVqePrepostConfig cfg_time, cfg_freq;
    DeepVqePrepostMemReq req_time, req_freq;
    DeepVqePrepost *p, *q;
    void *pool_time, *pool_freq;
    DeepVqePrepostInputs inputs, first_in;
    DeepVqePrepostOutputs outputs, first_out;
    static float f1_re[AIAEC_N_BINS], f1_im[AIAEC_N_BINS];
    static float f2_re[AIAEC_N_BINS], f2_im[AIAEC_N_BINS];
    static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
    static float got_re[AIAEC_N_BINS], got_im[AIAEC_N_BINS];
    float *expect[DEEPVQE_STATE_COUNT];
    int id, hop, target;
    size_t i;

    /* (1) Stream: through a real TIME-mode stream the state input and output
     * are one pointer per tensor, the same pointers on every frame, and the
     * frame-t output is what frame t+1 reads. Healthy-stream numerics are the
     * equiv/freq/reset cases above (byte-identical to the hand-composed
     * reference); this pins the aliasing itself. */
    CHECK(deepvqe_prepost_config_defaults(&cfg_time, DEEPVQE_IO_TIME, D) == 0);
    cfg_time.fft = fft;
    cfg_time.window = window;
    CHECK(deepvqe_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    CHECK(pool_time != NULL);
    p = deepvqe_prepost_init(pool_time, (size_t)req_time.bytes, &cfg_time);
    CHECK(p != NULL);
    memset(&first_in, 0, sizeof first_in);
    memset(&first_out, 0, sizeof first_out);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        expect[id] = (float *)malloc(
            deepvqe_prepost_state_elements(id, D) * sizeof(float));
        CHECK(expect[id] != NULL);
    }
    for (hop = 0; hop < SHORT_HOPS; ++hop) {
        CHECK(deepvqe_prepost_pre_process(p, pcm_mic + (size_t)hop * AIAEC_HOP,
                                          pcm_far + (size_t)hop * AIAEC_HOP)
              == 1);
        CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        if (hop == 0) {
            first_in = inputs;
            first_out = outputs;
            CHECK(state_all_zero(&inputs));      /* starts cold */
        }
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            CHECK(inputs.state[id] == (const float *)outputs.state_out[id]);
            CHECK(inputs.state[id] == first_in.state[id]);
            CHECK(outputs.state_out[id] == first_out.state_out[id]);
            CHECK(inputs.state_elements[id] == outputs.state_elements[id]);
            if (hop > 0)   /* the last output IS this input: no copy, no swap */
                CHECK(memcmp(inputs.state[id], expect[id],
                             inputs.state_elements[id] * sizeof(float)) == 0);
        }
        /* Only the head is NaN-prefilled; the state is left as the input. */
        for (i = 0; i < outputs.taps_elements; ++i)
            CHECK(isnan(outputs.taps[i]));
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            for (i = 0; i < inputs.state_elements[id]; ++i)
                CHECK(!isnan(inputs.state[id][i]));
        CHECK(fake_run(&inputs, &outputs) == 0);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            memcpy(expect[id], outputs.state_out[id],
                   inputs.state_elements[id] * sizeof(float));
        CHECK(deepvqe_prepost_frame_commit(p) == 0);
        /* commit leaves the state exactly as written. */
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(memcmp(first_in.state[id], expect[id],
                         first_in.state_elements[id] * sizeof(float)) == 0);
        CHECK(deepvqe_prepost_post_process(p, out_a, NULL) == 0);
    }
    CHECK(!state_all_zero(&first_in));   /* the stream really moved the state */
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(expect[id]);

    deepvqe_prepost_destroy(p);
    free(pool_time);

    /* (2) MUTATION: one non-finite element in ONE state tensor, injected
     * through the in-place pointer, must make commit return -1, leave ALL
     * sixteen tensors zero, not advance the CCM ring, and leave the next
     * healthy frame working. Every tensor takes the injection in turn (first
     * and last element, NaN and Inf), and the same frame WITHOUT the
     * injection commits, so the refusal is attributable to the injection. */
    CHECK(deepvqe_prepost_config_defaults(&cfg_freq, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_freq != NULL);
    q = deepvqe_prepost_init(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(q != NULL);
    make_frame(f1_re, f1_im, 0.0f);
    make_frame(f2_re, f2_im, 0.125f);
    make_frame(far_re, far_im, 0.0625f);

    for (target = 0; target < 2 * DEEPVQE_STATE_COUNT; ++target) {
        int tensor = target % DEEPVQE_STATE_COUNT;
        int last = target >= DEEPVQE_STATE_COUNT;   /* first half: elem 0 */
        float poison = (target % 2) ? INFINITY : NAN;
        float *victim;
        size_t index;

        deepvqe_prepost_reset(q);
        CHECK(commit_freq(q, f1_re, f1_im, far_re, far_im,
                          got_re, got_im) == 0);

        /* Control: the identical frame, uninjected, commits. */
        CHECK(deepvqe_prepost_pre_process_freq(q, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(!state_all_zero(&inputs));
        CHECK(deepvqe_prepost_frame_commit(q) == 0);
        CHECK(deepvqe_prepost_post_process_freq(q, got_re, got_im) == 0);
        CHECK(!all_zero(got_re, AIAEC_N_BINS));

        /* Same history, now poisoned. */
        deepvqe_prepost_reset(q);
        CHECK(commit_freq(q, f1_re, f1_im, far_re, far_im,
                          got_re, got_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(q, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        index = newest_element(tensor, last);
        victim = outputs.state_out[tensor];
        victim[index] = poison;
        CHECK(deepvqe_prepost_frame_commit(q) == -1);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id],
                           (int)outputs.state_elements[id]));
        CHECK(state_all_zero(&inputs));
        CHECK(deepvqe_prepost_frame_commit(q) == -1);   /* disarmed */

        /* The next healthy frame works: retry through a fresh frame_inputs
         * from the cold state. */
        CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
        CHECK(state_all_zero(&inputs));
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_commit(q) == 0);
        CHECK(deepvqe_prepost_post_process_freq(q, got_re, got_im) == 0);
        CHECK(!all_zero(got_re, AIAEC_N_BINS));
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            for (i = 0; i < outputs.state_elements[id]; ++i)
                CHECK(isfinite(outputs.state_out[id][i]));
    }

    /* (3) A refused commit does not advance the CCM ring. X: poisoned f1
     * refused, skipped (muted), then a healthy f2. Y: a fresh instance fed
     * only the healthy f2. Both reach f2 from the cold state with an empty
     * ring, so their outputs agree byte for byte -- a ring that had taken f1
     * on the refused commit would split them. */
    {
        DeepVqePrepost *y;
        void *pool_y;
        static float x_re[AIAEC_N_BINS], x_im[AIAEC_N_BINS];
        static float y_re[AIAEC_N_BINS], y_im[AIAEC_N_BINS];

        pool_y = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
        CHECK(pool_y != NULL);
        y = deepvqe_prepost_init(pool_y, (size_t)req_freq.bytes, &cfg_freq);
        CHECK(y != NULL);

        deepvqe_prepost_reset(q);
        CHECK(deepvqe_prepost_pre_process_freq(q, f1_re, f1_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_inputs(q, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.state_out[DEEPVQE_STATE_H_GRU][3] = NAN;
        CHECK(deepvqe_prepost_frame_commit(q) == -1);
        CHECK(deepvqe_prepost_frame_skip(q) == 0);
        CHECK(deepvqe_prepost_post_process_freq(q, x_re, x_im) == 0);
        CHECK(all_zero(x_re, AIAEC_N_BINS));
        CHECK(commit_freq(q, f2_re, f2_im, far_re, far_im, x_re, x_im) == 0);

        CHECK(commit_freq(y, f2_re, f2_im, far_re, far_im, y_re, y_im) == 0);
        CHECK(!all_zero(y_re, AIAEC_N_BINS));
        CHECK(identical(x_re, y_re, AIAEC_N_BINS));
        CHECK(identical(x_im, y_im, AIAEC_N_BINS));

        deepvqe_prepost_destroy(y);
        free(pool_y);
    }

    deepvqe_prepost_destroy(q);
    free(pool_freq);
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
     * find the same buffers. The fresh frame_inputs then re-fills the head
     * output with NaN, which is what makes the finite check a partial-write
     * detector rather than a stale-value detector on the taps. The state
     * tensors are NOT re-filled: they are the accelerator's input, and after
     * the refused commit they are the cold (all-zero) state. */
    CHECK(deepvqe_prepost_pre_process(p, pcm_mic + 2 * AIAEC_HOP,
                                      pcm_far + 2 * AIAEC_HOP) == 1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    outputs.state_out[DEEPVQE_STATE_UP3_HISTORY][newest_element(
        DEEPVQE_STATE_UP3_HISTORY, 0)] = NAN;
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    {
        size_t defined = 0;
        for (i = 0; i < outputs.taps_elements; ++i)
            if (!isnan(outputs.taps[i])) ++defined;
        CHECK(defined == 0u);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            CHECK(inputs.state[id] == outputs.state_out[id]);
            CHECK(all_zero(inputs.state[id], (int)inputs.state_elements[id]));
        }
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

/* ---- newest: commit reads the newest frame of every window ---------- */

static DeepVqePrepost *new_freq_instance(void **pool_out) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    void *pool;
    if (deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) != 0)
        return NULL;
    if (deepvqe_prepost_get_mem_size(&cfg, &req) != 0) return NULL;
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    if (!pool) return NULL;
    *pool_out = pool;
    return deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
}

/* Open one FREQ frame, run the stand-in, and return the views so the caller
 * can edit what the "accelerator" wrote before committing. */
static int open_and_run(DeepVqePrepost *p, DeepVqePrepostInputs *inputs,
                        DeepVqePrepostOutputs *outputs, float seed) {
    static float re[AIAEC_N_BINS], im[AIAEC_N_BINS];
    static float far_re[AIAEC_N_BINS], far_im[AIAEC_N_BINS];
    make_frame(re, im, seed);
    make_frame(far_re, far_im, seed + 0.0625f);
    if (deepvqe_prepost_pre_process_freq(p, re, im, far_re, far_im) != 1)
        return -1;
    if (deepvqe_prepost_frame_inputs(p, inputs, outputs) != 0) return -1;
    return fake_run(inputs, outputs);
}

static int case_newest(void) {
    DeepVqePrepost *p;
    void *pool = NULL;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    int id, which;

    p = new_freq_instance(&pool);
    CHECK(p != NULL);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        if (id != DEEPVQE_STATE_H_GRU) {
            /* The frame a window drops, and the oldest frames in general,
             * are not read: a poisoned value there is committed. */
            for (which = 0; which < 2; ++which) {
                float *state;
                size_t at = dropped_element(id);
                CHECK(open_and_run(p, &inputs, &outputs, 0.125f) == 0);
                state = outputs.state_out[id];
                state[at] = which ? INFINITY : NAN;
                CHECK(deepvqe_prepost_frame_commit(p) == 0);
                /* ...and the poison is the caller's to flush: reset. */
                deepvqe_prepost_reset(p);
            }
        }
        /* The newest frame is read: poison at its first and last element is
         * refused and takes every tensor with it. */
        for (which = 0; which < 2; ++which) {
            size_t at = newest_element(id, which);
            uint32_t before = deepvqe_prepost_state_generation(p);
            CHECK(open_and_run(p, &inputs, &outputs, 0.25f) == 0);
            CHECK(!all_zero(outputs.state_out[id],
                            (int)outputs.state_elements[id]));
            outputs.state_out[id][at] = NAN;
            CHECK(deepvqe_prepost_frame_commit(p) == -1);
            CHECK(deepvqe_prepost_state_generation(p) == before + 1u);
            {
                int other;
                for (other = 0; other < DEEPVQE_STATE_COUNT; ++other)
                    CHECK(all_zero(outputs.state_out[other],
                                   (int)outputs.state_elements[other]));
            }
            CHECK(deepvqe_prepost_frame_skip(p) == 0);
        }
    }
    /* h_gru is checked whole, not only in some slot. */
    {
        size_t last = outputs.state_elements[DEEPVQE_STATE_H_GRU] - 1u;
        size_t middle = last / 2u;
        size_t picks[2];
        picks[0] = middle;
        picks[1] = last;
        for (which = 0; which < 2; ++which) {
            CHECK(open_and_run(p, &inputs, &outputs, 0.375f) == 0);
            outputs.state_out[DEEPVQE_STATE_H_GRU][picks[which]] = INFINITY;
            CHECK(deepvqe_prepost_frame_commit(p) == -1);
            CHECK(deepvqe_prepost_frame_skip(p) == 0);
        }
    }
    deepvqe_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- generation: the restart counter ---------------------------------- */

static int case_generation(void) {
    DeepVqePrepost *p;
    void *pool = NULL;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float got_re[AIAEC_N_BINS], got_im[AIAEC_N_BINS];

    CHECK(deepvqe_prepost_state_generation(NULL) == 0u);
    p = new_freq_instance(&pool);
    CHECK(p != NULL);
    CHECK(deepvqe_prepost_state_generation(p) == 0u);

    /* Healthy commits, skips and new frames change nothing. */
    CHECK(open_and_run(p, &inputs, &outputs, 0.0f) == 0);
    CHECK(deepvqe_prepost_frame_commit(p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
    CHECK(open_and_run(p, &inputs, &outputs, 0.5f) == 0);
    CHECK(deepvqe_prepost_frame_skip(p) == 0);
    CHECK(deepvqe_prepost_state_generation(p) == 0u);

    /* reset advances it, once per call. */
    deepvqe_prepost_reset(p);
    CHECK(deepvqe_prepost_state_generation(p) == 1u);
    deepvqe_prepost_reset(p);
    CHECK(deepvqe_prepost_state_generation(p) == 2u);

    /* A refused commit advances it once; the refused retry does not. */
    CHECK(open_and_run(p, &inputs, &outputs, 0.25f) == 0);
    outputs.taps[DEEPVQE_TAPS_ELEMENTS / 2] = NAN;
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_state_generation(p) == 3u);
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_state_generation(p) == 3u);
    CHECK(deepvqe_prepost_frame_skip(p) == 0);

    /* A commit refused for a hole in a state tensor counts the same. */
    CHECK(open_and_run(p, &inputs, &outputs, 0.75f) == 0);
    outputs.state_out[DEEPVQE_STATE_UP3_HISTORY][newest_element(
        DEEPVQE_STATE_UP3_HISTORY, 0)] = NAN;
    CHECK(deepvqe_prepost_frame_commit(p) == -1);
    CHECK(deepvqe_prepost_state_generation(p) == 4u);
    CHECK(deepvqe_prepost_frame_skip(p) == 0);

    /* ...and the next healthy frame leaves it alone. */
    CHECK(open_and_run(p, &inputs, &outputs, 0.125f) == 0);
    CHECK(deepvqe_prepost_frame_commit(p) == 0);
    CHECK(deepvqe_prepost_state_generation(p) == 4u);
    deepvqe_prepost_destroy(p);
    free(pool);
    return 0;
}
/* ---- boundary: names, shapes, element counts, descriptor validation --- */

/* The twelve descriptor fields that are COMPILED CONSTANTS. delay_depth is
 * deliberately absent: it is the one export-time deployment parameter, so
 * validate() range-checks it instead of pinning it, and a +1 there is
 * legitimately accepted. */
static const char *const dv_constant_field_names[12] = {
    "layout_version", "sample_rate", "fft_size", "hop_size", "spectrum_bins",
    "time_order", "freq_taps", "conv_window_frames", "score_window_frames",
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
    case 7:  probe.conv_window_frames += 1;   break;
    case 8:  probe.score_window_frames += 1;  break;
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
        CHECK(rank == 4);
        CHECK(dims[0] == 1);
        if (id == DEEPVQE_STATE_H_GRU) {
            /* [1, GRU_LAYERS, 1, GRU_HIDDEN] */
            CHECK(dims[1] == 1 && dims[2] == 1 && dims[3] == 192);
        } else if (is_ring(id)) {
            CHECK(dims[2] == D);              /* a ring holds D frames */
        } else if (id == DEEPVQE_STATE_ALIGN_SCORE_HISTORY) {
            CHECK(dims[2] == 5 && dims[3] == D);
        } else {
            CHECK(dims[2] == 4);              /* a conv window holds 4 */
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
    descriptor.conv_window_frames    = atoi(argv[10]);
    descriptor.score_window_frames   = atoi(argv[11]);
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

/* ---- hash: the stale-pool identity of this build ---------------------- */

static int case_hash(void) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    CHECK(deepvqe_prepost_config_defaults(&cfg, DEEPVQE_IO_FREQ, D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    /* The ABI token folded into the hash follows the C target, which a
     * cross-built driver run under an emulator does not share with the
     * Python host. */
    printf("hash=%u layout=%u ptr_bytes=%zu\n", req.build_flags_hash,
           req.layout_version, sizeof(void *));
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
    ac_window_build(window, AIAEC_N_FFT);
    fill_pcm(pcm_mic, pcm_far, HOPS * AIAEC_HOP);

    if (strcmp(argv[1], "hash") == 0) return case_hash();
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
    else if (strcmp(argv[1], "compression") == 0)
                                                status = case_compression();
    else if (strcmp(argv[1], "freqpool") == 0)  status = case_freqpool(fft);
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle(fft);
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject(fft);
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset(fft);
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard(fft);
    else if (strcmp(argv[1], "inplace") == 0)   status = case_inplace(fft);
    else if (strcmp(argv[1], "newest") == 0)    status = case_newest();
    else if (strcmp(argv[1], "generation") == 0) status = case_generation();
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
    """audiolib's NE10 static lib (the default backend),
    built and located the way the other C tests document it."""
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
         '-I', _AC_INCLUDE, '-I', _AC_DIR, str(source), *_SOURCES,
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
    """DEEPVQE_IO_TIME vs a rolling center=False analysis, one hand-held
    in-place state buffer set, deepvqe_ccm_process and a hand-built AcIstft
    wired up from ONLY the composed TUs -- byte-identical over 400 hops, with
    `written` following the synthesis half-window schedule (0, then one full
    hop)."""
    assert 'io_mode=TIME' in _run(driver, 'equiv')


def test_host_front_end_is_the_models_compression(driver):
    """The planar tensors the class hands the graph are x * |x|^(e-1) with
    |x| = sqrt(re^2 + im^2 + 1e-12), the model's compressed_ri_feature, for
    zero, tiny, negative and large bins -- checked against double precision,
    so the equivalence cases (which mirror the fp32 expression) cannot hide a
    wrong formula."""
    _run(driver, 'compression')


def test_compression_exponent_matches_the_exporter():
    from AIAEC._streaming_export import DEEPVQE_COMPRESSION_EXPONENT

    with open(_PREPOST_HEADER, encoding='utf-8') as stream:
        text = stream.read()
    found = re.findall(r'^#define\s+DEEPVQE_COMPRESSION_EXP\s+([0-9.]+)f\s*$',
                       text, re.MULTILINE)
    assert len(found) == 1, found
    assert float(found[0]) == DEEPVQE_COMPRESSION_EXPONENT


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


def test_frame_skip_mutes_keeps_the_written_state_and_freezes_the_ring(driver):
    """FAIL CLOSED. DeepVQE-S's stream 0 is the RAW microphone, so the
    pass-through identity a post-filter takes would emit the full uncancelled
    echo; _frame_skip emits SILENCE instead and the policy names itself
    "mute_fail_closed". The state is in place, so a skip leaves it exactly as
    the accelerator wrote it, and the CCM spectrum ring does not advance --
    proved by an instance whose accelerator wrote nothing agreeing with one
    that never saw the frame, and both differing from one that committed it."""
    _run(driver, 'skip')


def test_state_is_in_place_and_a_non_finite_write_zeroes_it(driver):
    """One buffer per state tensor: the input and `_out` pointers are equal
    and stable, the frame-t output is the frame-t+1 input with no copy, and
    only the head is NaN-prefilled. MUTATION: one NaN/Inf injected into the
    newest frame of any one of the sixteen tensors (first and last element)
    through the in-place pointer makes _frame_commit return -1, zeroes ALL sixteen tensors, and
    the next healthy frame still commits -- while the identical uninjected
    frame commits, so the refusal is attributable to the injection. A refused
    commit also leaves the CCM ring where it was."""
    _run(driver, 'inplace')


def test_commit_reads_the_newest_frame_of_every_window(driver):
    """A window is written back whole but only its newest frame is new: slot
    0 of an attention ring, the last frame of a conv or score window, and
    h_gru whole. A NaN/Inf there is refused and zeroes every tensor; the same
    poison in the oldest frame (the one a window drops) is committed."""
    _run(driver, 'newest')


def test_state_generation_counts_restarts(driver):
    """0 at init; reset and a refused commit advance it, once each; healthy
    commits, skips and a refused retry do not. NULL reads 0. A board with a
    fixed-point accelerator state refills it with quantized zeros when the
    value changes."""
    _run(driver, 'generation')


def test_inherit_path_is_gone():
    for path in (_PREPOST_HEADER,
                 os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c')):
        with open(path, encoding='utf-8') as stream:
            assert 'inherit' not in stream.read().replace('inherited', ''), path


def test_transaction_guards(driver):
    """The three gates that make the boundary a transaction: a second
    pre_process while a frame is open is refused without damaging that frame,
    a commit with no frame_inputs behind it is refused (an accelerator that
    never ran cannot pass untouched buffers off as a result), and a refused
    commit disarms until a fresh frame_inputs -- which re-fills every tap with
    NaN and presents the (now cold) state tensors unfilled."""
    _run(driver, 'guard')


def test_state_boundary_names_shapes_and_descriptor_validation(driver):
    """The explicit boundary: sixteen named state tensors whose output name
    is the input name plus "_out", rank 4 everywhere (h_gru is [1,1,1,192],
    conv windows hold 4 frames, the score window 5, rings D), element counts
    that are the product of the published dims, and
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


def test_shipped_delay_depth_agrees_across_header_config_and_grid():
    """The half-second search range is stated three times: the C default
    (DEEPVQE_PREPOST_DEFAULT_D), the trainer config's max_delay_seconds and the
    grid's frame conversion. They must name the same D, or a board built with
    the default asks for a pool the shipped graph does not fit."""
    import configparser

    from AIAEC.aiaec_common import SignalGrid

    with open(_PREPOST_HEADER, encoding='utf-8') as stream:
        found = re.findall(
            r'^#define\s+DEEPVQE_PREPOST_DEFAULT_D\s+(\d+)\s*$',
            stream.read(), re.MULTILINE)
    assert len(found) == 1, found
    config = configparser.ConfigParser()
    assert config.read(os.path.join(_DEEPVQE_DIR, 'config.ini'))
    seconds = config.getfloat('model', 'max_delay_seconds')
    assert seconds == pytest.approx(0.5)
    grid = SignalGrid(16000, 512, 512, 256)
    assert grid.delay_frames(seconds) == int(found[0]) == 32


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
    model = DeepVQES(grid, max_delay_seconds=0.5).eval()
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

    assert descriptor['delay_depth'] == model.max_delay_frames == 32
    assert descriptor['state_tensor_count'] == 16


def test_export_metadata_carries_the_c_descriptor(driver, deepvqe_built,
                                                  tmp_path):
    """A shipped artifact must carry the boundary its C side binds. Both the
    ONNX metadata and the JSON written beside it must record
    state_layout_version 4 and the measured c_descriptor -- the very dict
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
                                32, verify=False)

    assert metadata['state_layout_version'] == DEEPVQE_C_LAYOUT_VERSION
    assert metadata['c_descriptor'] == descriptor
    assert metadata['output_schema']['output'] == [1, 1, 257, 18]

    with open(str(output.with_suffix('.json')), encoding='utf-8') as stream:
        written = json.load(stream)
    assert written['state_layout_version'] == DEEPVQE_C_LAYOUT_VERSION
    assert written['c_descriptor'] == descriptor
    assert written['output_schema']['output'] == [1, 1, 257, 18]


def _fnv1a(values):
    h = 2166136261
    for value in values:
        for shift in (0, 8, 16, 24):
            h ^= (value >> shift) & 0xff
            h = (h * 16777619) & 0xffffffff
    return h


def test_pool_identity_moved_with_the_full_window_carve(driver):
    """The stale-pool gate keys on (grid, layout, carve version, mode, D).
    A pool requested under layout 3 / carve 3 (conv and score windows one
    frame shorter, rank-3 h_gru) must hash differently from the shipped one;
    the hash is recomputed here from the published constants."""
    out = _run(driver, 'hash')
    c_hash = int(re.search(r'hash=(\d+)', out).group(1))
    layout = int(re.search(r'layout=(\d+)', out).group(1))
    with open(_PREPOST_HEADER, encoding='utf-8') as stream:
        carve = int(re.search(
            r'^#define\s+DEEPVQE_PREPOST_CARVE_VERSION\s+(\d+)u',
            stream.read(), re.MULTILINE).group(1))
    assert (layout, carve) == (4, 5)
    expected = _fnv1a([16000, 512, layout, carve, 0, 8])
    ptr_bytes = int(re.search(r'ptr_bytes=(\d+)', out).group(1))
    if ptr_bytes == 4:
        for byte in b'|abi:ilp32':
            expected = ((expected ^ byte) * 16777619) & 0xffffffff
    assert c_hash == expected
    assert c_hash != _fnv1a([16000, 512, 3, 3, 0, 8])
