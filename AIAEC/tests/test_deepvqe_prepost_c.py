"""Equivalence and contract gates for the DeepVQE-S pre/post class.

``DeepVQE_S/deepvqe_prepost.c`` owns no signal processing: it composes the CCM
kernel (``DeepVQE_S/deepvqe_process.c``) behind one frequency-domain object, so
the gate is that the class reproduces the hand-composed path SAMPLE FOR SAMPLE:
every numeric comparison is ``memcmp`` on fp32, and every "identical" claim is
paired with a "not trivially identical" one so no gate passes on silence. The
framing is the application's, built from audiolib directly (sqrt-Hann, center
0, hop = n_fft / 2, envelope-normalised synthesis). The recurrent state is the
application's too: the driver is a float host that owns the sixteen NaN-filled
buffers of every instance and registers them; the state shapes are a table of
the driver (the library knows none), pinned to the exported graph below.

Cases, all in ONE driver selected by argv[1], the grid by argv[2]
(``grid=<sample rate>,<n_fft>``; audiolib and the model TUs are compiled once
for the module):

  equiv      the class vs a rolling analysis + one hand-held in-place state
             buffer set + deepvqe_ccm_process + a hand-built AcIstft
  stalepool  the stale-pool gate accepts the matching MemReq and refuses every
             mismatched one; the pool follows the grid, not D
  lifecycle  _create/_destroy against _init on a caller pool
  reject     _get_mem_size / _init / the stage calls, reject-first
  reset      _reset really clears state (a re-run reproduces the first run)
  skip       _frame_bypass MUTES (fail-closed), leaves the state as the
             accelerator wrote it and does not advance the CCM ring
  inplace    one buffer per state tensor (input and output are the same pointer,
             the registered buffer, stable for the instance's life), only the
             head NaN-prefilled, and a non-finite write to the newest frame of
             any state tensor zeroes all sixteen and leaves the next frame
             healthy
  guard      no double pre_process, no commit without frame_open, no re-commit
             after a refused one
  whole      commit checks the whole of every registered buffer; a non-finite
             value in the newest, an older or the dropped frame is refused
  generation state_generation starts at 0 and advances on reset and on a
             refused commit, and on nothing else
  boundary   state names and the descriptor validator: the eight external
             fields are refused when perturbed, the four model-interior fields
             are not read
  descriptor a validate/shape dump driven from argv, so the Python side can feed
             it a descriptor MEASURED from the built graph
  board      nothing registered (the accelerator owns the state): empty state
             views, generation 0 at init, a refused commit advances it by one
             and the next hop recovers, the pool fences stay intact
  registration  all sixteen state pairs or none: every partial registration is
             refused and changes nothing, then the full one is accepted and runs
  grid       deepvqe_grid_make / deepvqe_grid_validate: each documented status

The accelerator stand-in is deterministic and input-dependent (it sums both
signal inputs and all sixteen state tensors and writes its next state back in
place), so a state handoff that failed to happen diverges rather than agrees.

THE SKIP POLICY IS DELIBERATE: stream 0 is the RAW microphone, not a residual,
so a pass-through would emit the full uncancelled echo; ``frame_bypass`` emits
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
_DEEPVQE_DIR = os.path.join(_AIAEC_DIR, '..', 'lib', 'aiaec', 'deepvqe_s')
# deepvqe_*.c/.h moved to lib/aiaec/deepvqe_s/ in the single-model-release restructuring;
# config.ini (the Python training side) stayed at AIAEC/DeepVQE_S/.
_DEEPVQE_PY_DIR = os.path.join(_AIAEC_DIR, 'DeepVQE_S')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_PREPOST_HEADER = os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.h')

_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c'),
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
)

_GRID_16K = 'grid=16000,512'
_GRID_48K = 'grid=48000,1024'

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

#include "audio_stft.h"
#include "deepvqe_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

/* D=8 everywhere except where a case explicitly needs the shipped depth:
 * every state tensor's size scales with it, and the fake accelerator walks
 * all of them on every frame. */
#define D          8
#define SHIPPED_D  32
#define HOPS       400
#define SHORT_HOPS 40
#define BINS_MAX   513
#define HOP_MAX    512
#define N_FFT_MAX  1024

#define TAPS_PER_BIN (DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2)

/* The grid under test (run-time): argument grid=<sample rate>,<n_fft>. */
static DeepVqeGrid g_grid;
#define BINS   ((size_t)g_grid.n_bins)
#define HOP    ((size_t)g_grid.hop)
#define N_FFT  g_grid.n_fft

static DeepVqeGrid grid_with_depth(int depth) {
    const DeepVqeGridSpec spec = { .sample_rate = g_grid.sample_rate,
                                   .n_fft = g_grid.n_fft, .delay_depth = depth };
    DeepVqeGrid made;
    if (deepvqe_grid_make(&made, &spec) != DEEPVQE_GRID_OK) made.n_fft = 0;
    return made;
}

/* BEGIN state table: the sixteen state tensors of the shipped graph at the grid under test, as
 * [1][channels][frames][width] (pinned to the exporter by test_deepvqe_prepost_c.py). */
#define F0 (g_grid.n_bins)
#define F1 ((F0 + 1) / 2)
#define F2 ((F1 + 1) / 2)
#define F3 ((F2 + 1) / 2)
#define F4 ((F3 + 1) / 2)
#define CONV_WINDOW 4
#define SCORE_WINDOW 5

static int state_dims(int id, int depth, int dims[4])
{
    int c = 0, t = 0, f = 0;
    switch (id) {
    case DEEPVQE_STATE_ALIGN_KEY_RING:      c = 4;  t = depth; f = F2; break;
    case DEEPVQE_STATE_ALIGN_VALUE_RING:    c = 24; t = depth; f = F2; break;
    case DEEPVQE_STATE_ALIGN_SCORE_HISTORY: c = 4;  t = SCORE_WINDOW; f = depth; break;
    case DEEPVQE_STATE_CCM_UP_HISTORY:      c = 32; t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_FAR1_HISTORY:        c = 2;  t = CONV_WINDOW; f = F0; break;
    case DEEPVQE_STATE_FAR2_HISTORY:        c = 8;  t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_H_GRU:
        dims[0] = 1; dims[1] = 1; dims[2] = 1; dims[3] = 192;
        return 4;
    case DEEPVQE_STATE_MIC1_HISTORY:        c = 2;  t = CONV_WINDOW; f = F0; break;
    case DEEPVQE_STATE_MIC2_HISTORY:        c = 16; t = CONV_WINDOW; f = F1; break;
    case DEEPVQE_STATE_MIC3_HISTORY:        c = 64; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_MIC4_HISTORY:        c = 56; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_RES2_HISTORY:        c = 32; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_RES3_HISTORY:        c = 40; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_UP1_HISTORY:         c = 32; t = CONV_WINDOW; f = F2; break;
    case DEEPVQE_STATE_UP2_HISTORY:         c = 40; t = CONV_WINDOW; f = F3; break;
    case DEEPVQE_STATE_UP3_HISTORY:         c = 24; t = CONV_WINDOW; f = F4; break;
    default: return -1;
    }
    dims[0] = 1; dims[1] = c; dims[2] = t; dims[3] = f;
    return 4;
}
/* END state table */

static size_t state_count(int id, int depth)
{
    int dims[4];
    size_t n = 1;
    if (state_dims(id, depth, dims) != 4) return 0;
    for (int k = 0; k < 4; ++k) n *= (size_t)dims[k];
    return n;
}

/* Where the commit looks: the whole of every registered buffer. Every state tensor is
 * [1,C,T,F]: a ring (attention key/value) is newest-first, so its newest frame is slot
 * 0; a conv or score window is oldest-first, so its newest frame is slot T-1. h_gru is
 * one flat tensor. These helpers locate elements by direct indexing on the table. */
static int is_ring(int id) {
    return id == DEEPVQE_STATE_ALIGN_KEY_RING ||
           id == DEEPVQE_STATE_ALIGN_VALUE_RING;
}

static size_t frame_element(int id, size_t channel, size_t frame, size_t bin) {
    int dims[4];
    if (state_dims(id, D, dims) != 4) return 0;
    return ((channel * (size_t)dims[2]) + frame) * (size_t)dims[3] + bin;
}

static size_t newest_frame(int id) {
    int dims[4];
    if (is_ring(id)) return 0u;
    if (state_dims(id, D, dims) != 4) return 0;
    return (size_t)dims[2] - 1u;
}

/* bin 0 of the newest frame of channel 0 (`last` == 0) or the last bin of the newest
 * frame of the last channel. */
static size_t newest_element(int id, int last) {
    int dims[4];
    if (id == DEEPVQE_STATE_H_GRU)
        return last ? state_count(id, D) - 1u : 0u;
    if (state_dims(id, D, dims) != 4) return 0;
    return last ? frame_element(id, (size_t)dims[1] - 1u, newest_frame(id),
                                (size_t)dims[3] - 1u)
                : frame_element(id, 0u, newest_frame(id), 0u);
}

/* bin 0 of the oldest frame of channel 0 -- the slot a window drops (slot D-1 of a
 * ring, frame 0 of a conv or score window). */
static size_t dropped_element(int id) {
    int dims[4];
    if (state_dims(id, D, dims) != 4) return 0;
    return frame_element(id, 0u, is_ring(id) ? (size_t)dims[2] - 1u : 0u, 0u);
}

/* Deterministic stand-in for the accelerator. It satisfies the FULL-WRITE
 * contract -- every element of `taps` and of all sixteen state_out tensors --
 * and is made input-dependent on BOTH signal inputs and EVERY state tensor,
 * so a state handoff that did not happen (or happened when it should not
 * have) changes every later frame instead of going unnoticed.
 *
 * state_out aliases state: the whole old state is reduced into `acc` before
 * the first output element is written, so every state input is read before any
 * state output is written, as the in-place contract requires. The state views
 * are the registered buffers: a missing or short registration fails the run, it
 * does not become an empty loop. */
static int fake_run(const DeepVqePrepostInputs *in,
                    DeepVqePrepostOutputs *out) {
    double acc = 0.0;
    size_t i;
    int id;

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (!in->state[id] || in->state_elements[id] != state_count(id, D))
            return -1;
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

static int identical(const float *a, const float *b, size_t count) {
    return memcmp(a, b, count * sizeof(float)) == 0;
}

static int all_zero(const float *values, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) if (values[i] != 0.0f) return 0;
    return 1;
}

static float pcm_mic[HOPS * HOP_MAX];
static float pcm_far[HOPS * HOP_MAX];
static float out_a[HOPS * HOP_MAX];
static float out_b[HOPS * HOP_MAX];
static float window[N_FFT_MAX];

/* One analysed hop per stream, and the spectra the class hands back. */
typedef struct HopSpectra {
    float mr[BINS_MAX], mi[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX];
} HopSpectra;
typedef struct HopOut {
    float re[BINS_MAX], im[BINS_MAX];
} HopOut;
static HopSpectra specs[HOPS];
static HopOut outs_a[HOPS], outs_b[HOPS];

/* The application's framing, built from audiolib directly (deepvqe_prepost.h): sqrt-Hann,
 * center 0, hop = n_fft / 2, unnormalised spectra, AC_OLA_ENV_NORM synthesis. */
static void framing_config(AcStftConfig *sc) {
    ac_stft_config_default(sc, N_FFT);
    sc->ola = AC_OLA_ENV_NORM;
}

static AcStft *make_analysis(void) {
    AcStftConfig sc;
    size_t bytes;
    framing_config(&sc);
    bytes = ac_stft_get_mem_size(&sc);
    return bytes ? ac_stft_init(alloc_aligned(16u, bytes), bytes, &sc) : NULL;
}

static AcIstft *make_synthesis(void) {
    AcStftConfig sc;
    size_t bytes;
    framing_config(&sc);
    bytes = ac_istft_get_mem_size(&sc);
    return bytes ? ac_istft_init(alloc_aligned(16u, bytes), bytes, &sc) : NULL;
}

/* center=False rolling analysis: one frame per hop over the last N_FFT
 * samples. Deliberately expression-for-expression the same as
 * audio_stft.c's rolling analysis, so this reference is independent of
 * deepvqe_prepost.c yet bit-comparable with it. */
static void ref_roll(FftHandle *fft, float *history, const float *hop_in,
                     float *out_re, float *out_im) {
    static float seg[N_FFT_MAX];
    static Complex spec[BINS_MAX];
    size_t k;
    memmove(history, history + HOP, (size_t)(N_FFT - g_grid.hop) * sizeof(float));
    memcpy(history + N_FFT - g_grid.hop, hop_in, HOP * sizeof(float));
    for (k = 0; k < (size_t)N_FFT; ++k) seg[k] = history[k] * window[k];
    fft_forward_scratch(fft, seg, spec);
    for (k = 0; k < BINS; ++k) {
        out_re[k] = spec[k].r;
        out_im[k] = spec[k].i;
    }
}

/* Analyse every hop of the test signal once, with audiolib's rolling analysis. */
static int build_specs(void) {
    AcStft *mic = make_analysis(), *far = make_analysis();
    int hop;
    if (!mic || !far) return -1;
    for (hop = 0; hop < HOPS; ++hop) {
        if (ac_stft_push_planar(mic, pcm_mic + (size_t)hop * HOP,
                                specs[hop].mr, specs[hop].mi) != 1) return -1;
        if (ac_stft_push_planar(far, pcm_far + (size_t)hop * HOP,
                                specs[hop].fr, specs[hop].fi) != 1) return -1;
    }
    return 0;
}

/* The model's compressed_ri_feature for one spectrum, bin by bin and in the
 * same fp32 operations the class runs, planar [real plane, imaginary plane].
 * case_compression checks the formula itself against double precision. */
static void compress_ref(const float *re, const float *im, float *ri) {
    size_t bin;
    for (bin = 0; bin < BINS; ++bin) {
        const float mag = sqrtf(re[bin] * re[bin] + im[bin] * im[bin] + 1e-12f);
        const float scale = powf(mag, DEEPVQE_COMPRESSION_EXP - 1.0f);
        ri[bin] = re[bin] * scale;
        ri[BINS + bin] = im[bin] * scale;
    }
}

/* An instance with the application's sixteen state buffers (NaN-filled, so an
 * instance that did not zero what it was handed shows up) registered. */
typedef struct Inst {
    DeepVqePrepost *p;
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    void *pool;
    float *buf[DEEPVQE_STATE_COUNT];
    int heap;
} Inst;

static int inst_config(Inst *in, int depth) {
    DeepVqeGrid grid = grid_with_depth(depth);
    size_t i;
    int id;
    memset(in, 0, sizeof *in);
    if (grid.n_fft == 0 || deepvqe_prepost_config_defaults(&in->cfg, &grid) != 0) return -1;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        const size_t n = state_count(id, depth);
        in->buf[id] = (float *)malloc(n * sizeof(float));
        if (!in->buf[id]) return -1;
        for (i = 0; i < n; ++i) in->buf[id][i] = NAN;
        in->cfg.state[id] = in->buf[id];
        in->cfg.state_elements[id] = n;
    }
    return 0;
}

/* A pool instance (heap == 0) or a _create instance (heap == 1). */
static int inst_init(Inst *in, int heap) {
    in->heap = heap;
    if (deepvqe_prepost_get_mem_size(&in->cfg, &in->req) != 0) return -1;
    if (heap) {
        in->p = deepvqe_prepost_create(&in->cfg);
    } else {
        in->pool = alloc_aligned(in->req.alignment, (size_t)in->req.bytes);
        if (!in->pool) return -1;
        in->p = deepvqe_prepost_init_ex(in->pool, (size_t)in->req.bytes, &in->cfg,
                                        &in->req);
    }
    return in->p ? 0 : -1;
}

static int inst_make(Inst *in) {
    return inst_config(in, D) == 0 && inst_init(in, 0) == 0 ? 0 : -1;
}

static void inst_free(Inst *in) {
    int id;
    deepvqe_prepost_destroy(in->p);
    free(in->pool);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(in->buf[id]);
}

static int inst_state_zero(const Inst *in) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (!all_zero(in->buf[id], state_count(id, D))) return 0;
    return 1;
}

static int bytes_are(const void *base, size_t count, unsigned char value) {
    const unsigned char *bytes = (const unsigned char *)base;
    size_t i;
    for (i = 0; i < count; ++i) if (bytes[i] != value) return 0;
    return 1;
}

/* Copies of an instance's registered buffers, to prove a refused call changed none of them. */
typedef struct Snap { float *copy[DEEPVQE_STATE_COUNT]; } Snap;

static int snap_take(Snap *snap, const Inst *in) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        const size_t bytes = in->cfg.state_elements[id] * sizeof(float);
        snap->copy[id] = (float *)malloc(bytes);
        if (!snap->copy[id]) return -1;
        memcpy(snap->copy[id], in->buf[id], bytes);
    }
    return 0;
}

static int snap_same(const Snap *snap, const Inst *in) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (memcmp(snap->copy[id], in->buf[id],
                   in->cfg.state_elements[id] * sizeof(float)) != 0) return 0;
    return 1;
}

static void snap_free(Snap *snap) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(snap->copy[id]);
}

/* One accelerator transaction through the class for the analysed hops
 * [first, first + count); the enhanced spectra land in `out`. */
static int drive_spec(DeepVqePrepost *p, int first, int count, HopOut *out) {
    int hop;
    for (hop = first; hop < first + count; ++hop) {
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        /* THE contract this class exists to guarantee: one spectrum in, exactly
         * one accelerator invocation, from the very first hop. */
        if (deepvqe_prepost_pre_process_freq(p, specs[hop].mr, specs[hop].mi,
                                             specs[hop].fr, specs[hop].fi) != 1)
            return -1;
        if (deepvqe_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
        if (fake_run(&inputs, &outputs) != 0) return -1;
        if (deepvqe_prepost_frame_finish(p) != 0) return -1;
        if (deepvqe_prepost_post_process_freq(p, out[hop - first].re,
                                              out[hop - first].im) != 0) return -1;
    }
    return 0;
}

static int outs_identical(const HopOut *a, const HopOut *b, int count) {
    int hop;
    for (hop = 0; hop < count; ++hop)
        if (!identical(a[hop].re, b[hop].re, BINS) ||
            !identical(a[hop].im, b[hop].im, BINS)) return 0;
    return 1;
}

static int outs_nonzero(const HopOut *a, int count) {
    int hop;
    for (hop = 0; hop < count; ++hop)
        if (!all_zero(a[hop].re, BINS) || !all_zero(a[hop].im, BINS)) return 1;
    return 0;
}

/* ---- compression: the host front end vs double precision -------------- */

static int case_compression(void) {
    static float mic_re[BINS_MAX], mic_im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    DeepVqePrepost *p;
    void *pool;
    size_t bin;
    int stream;

    /* Zero, denormal-scale, ordinary, negative and very large bins: the
     * 1e-12 floor under the sqrt is what keeps a silent bin finite. */
    for (bin = 0; bin < BINS; ++bin) {
        const float sign = (bin % 3 == 0) ? -1.0f : 1.0f;
        mic_re[bin] = sign * (float)bin * 0.37f;
        mic_im[bin] = -sign * (float)(BINS - bin) * 0.011f;
        far_re[bin] = (bin % 5 == 0) ? 0.0f : sign * 1e-7f * (float)bin;
        far_im[bin] = (bin % 7 == 0) ? 0.0f : sign * 3000.0f / (float)(bin + 1);
    }
    mic_re[10] = mic_im[10] = 0.0f;
    far_re[3] = 1.0e4f;

    /* The front end needs no recurrent state: nothing registered. */
    CHECK(deepvqe_prepost_config_defaults(&cfg, &g_grid) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(deepvqe_prepost_pre_process_freq(p, mic_re, mic_im, far_re,
                                           far_im) == 1);
    CHECK(deepvqe_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(inputs.spectrum_ri_elements == 2u * BINS);

    for (stream = 0; stream < 2; ++stream) {
        const float *re = stream ? far_re : mic_re;
        const float *im = stream ? far_im : mic_im;
        const float *got = stream ? inputs.far : inputs.mic;
        for (bin = 0; bin < BINS; ++bin) {
            const double power = (double)re[bin] * (double)re[bin] +
                                 (double)im[bin] * (double)im[bin] + 1e-12;
            const double scale = pow(sqrt(power),
                                     (double)DEEPVQE_COMPRESSION_EXP - 1.0);
            const double want_re = (double)re[bin] * scale;
            const double want_im = (double)im[bin] * scale;
            CHECK(fabs((double)got[bin] - want_re) <=
                  4e-6 * fabs(want_re) + 1e-30);
            CHECK(fabs((double)got[BINS + bin] - want_im) <=
                  4e-6 * fabs(want_im) + 1e-30);
        }
    }
    deepvqe_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- equiv: the class vs the hand-composed reference path ------------ */

static int case_equiv(FftHandle *fft) {
    /* Path A -- the CCM kernel plus one in-place state buffer set held by
     * hand. The class contributes nothing here, so this reference cannot be
     * circular. */
    static float hist_mic[N_FFT_MAX], hist_far[N_FFT_MAX];
    AcIstft *synthesis_a, *synthesis_b;
    static DeepVqeCcmState ccm;
    static float ring_re[DEEPVQE_TIME_ORDER * BINS_MAX], ring_im[DEEPVQE_TIME_ORDER * BINS_MAX];
    static float mic_ri[2 * BINS_MAX], far_ri[2 * BINS_MAX];
    static float taps_ref[BINS_MAX * TAPS_PER_BIN];
    float *state[DEEPVQE_STATE_COUNT];
    size_t elements[DEEPVQE_STATE_COUNT];
    Inst inst;
    int id, hop;

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        elements[id] = state_count(id, D);
        CHECK(elements[id] > 0u);
        state[id] = (float *)calloc(elements[id], sizeof(float));
        CHECK(state[id] != NULL);
    }
    memset(hist_mic, 0, sizeof hist_mic);
    memset(hist_far, 0, sizeof hist_far);
    CHECK(deepvqe_ccm_init(&ccm, g_grid.n_bins, ring_re, ring_im) == 0);
    synthesis_a = make_synthesis();
    synthesis_b = make_synthesis();
    CHECK(synthesis_a != NULL && synthesis_b != NULL);

    for (hop = 0; hop < HOPS; ++hop) {
        static float mic_re[BINS_MAX], mic_im[BINS_MAX];
        static float far_re[BINS_MAX], far_im[BINS_MAX];
        static float enh_re[BINS_MAX], enh_im[BINS_MAX];
        float *slot = out_a + (size_t)hop * HOP;
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        int wrote;

        ref_roll(fft, hist_mic, pcm_mic + (size_t)hop * HOP, mic_re, mic_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * HOP, far_re, far_im);
        /* audiolib's rolling analysis is that reference, bit for bit. */
        CHECK(identical(specs[hop].mr, mic_re, BINS) && identical(specs[hop].mi, mic_im, BINS));
        CHECK(identical(specs[hop].fr, far_re, BINS) && identical(specs[hop].fi, far_im, BINS));
        compress_ref(mic_re, mic_im, mic_ri);
        compress_ref(far_re, far_im, far_ri);

        memset(&inputs, 0, sizeof inputs);
        memset(&outputs, 0, sizeof outputs);
        inputs.mic = mic_ri;
        inputs.far = far_ri;
        inputs.spectrum_ri_elements = 2u * BINS;
        outputs.taps = taps_ref;
        outputs.taps_elements = deepvqe_prepost_taps_elements(&g_grid);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            inputs.state[id] = state[id];
            inputs.state_elements[id] = elements[id];
            outputs.state_out[id] = state[id];   /* in place */
            outputs.state_elements[id] = elements[id];
        }
        CHECK(fake_run(&inputs, &outputs) == 0);

        deepvqe_ccm_process(&ccm, mic_re, mic_im, taps_ref, enh_re, enh_im);
        wrote = ac_istft_push_planar(synthesis_a, enh_re, enh_im, slot);
        /* The synthesis half-window trim, restated on the reference side. */
        CHECK(wrote == (hop == 0 ? 0 : g_grid.hop));
        if ((size_t)wrote < HOP)
            memset(slot + wrote, 0, (HOP - (size_t)wrote) * sizeof(float));
    }

    /* Path B -- the class, fed the same analysed spectra. */
    CHECK(inst_make(&inst) == 0);
    CHECK(deepvqe_prepost_hop_size(inst.p) == g_grid.hop);
    CHECK(deepvqe_prepost_num_bins(inst.p) == g_grid.n_bins);
    CHECK(deepvqe_prepost_descriptor(inst.p) != NULL);
    CHECK(deepvqe_prepost_descriptor(inst.p)->delay_depth == D);
    CHECK(drive_spec(inst.p, 0, HOPS, outs_b) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        float *slot = out_b + (size_t)hop * HOP;
        int wrote = ac_istft_push_planar(synthesis_b, outs_b[hop].re, outs_b[hop].im, slot);
        CHECK(wrote == (hop == 0 ? 0 : g_grid.hop));
        if ((size_t)wrote < HOP)
            memset(slot + wrote, 0, (HOP - (size_t)wrote) * sizeof(float));
    }

    printf("pool: class %llu B (D=%d, %d bins)\n",
           (unsigned long long)inst.req.bytes, D, g_grid.n_bins);
    /* The reference is not silence and not a constant -- otherwise
     * byte-identity below would be vacuous. */
    CHECK(!all_zero(out_a, HOPS * HOP));
    CHECK(!identical(out_a, out_b + 1, HOPS * HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * HOP));

    inst_free(&inst);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(state[id]);
    return 0;
}

/* ---- stalepool: the stale-pool gate ----------------------------------- */

/* A refused _init_ex returns NULL and leaves the pool and the registered buffers byte for byte unchanged. */
static int stale_refused(void *pool, size_t bytes, const Inst *in,
                         const DeepVqePrepostMemReq *expected, const Snap *snap) {
    return deepvqe_prepost_init_ex(pool, bytes, &in->cfg, expected) == NULL &&
           bytes_are(pool, bytes, 0xa5u) && snap_same(snap, in);
}

/* The pool request for another grid. */
static int req_for_grid(const DeepVqeGridSpec *spec, DeepVqePrepostMemReq *req) {
    DeepVqePrepostConfig other_cfg;
    DeepVqeGrid other;
    if (deepvqe_grid_make(&other, spec) != DEEPVQE_GRID_OK ||
        deepvqe_prepost_config_defaults(&other_cfg, &other) != 0) return -1;
    return deepvqe_prepost_get_mem_size(&other_cfg, req);
}

static int case_stalepool(void) {
    Inst a, deep, shipped;
    DeepVqePrepostMemReq req_rate, req_fft;
    const DeepVqeGridSpec rate_spec = { .sample_rate = 2 * g_grid.sample_rate,
                                        .n_fft = g_grid.n_fft, .delay_depth = D };
    const DeepVqeGridSpec fft_spec = { .sample_rate = g_grid.sample_rate,
                                       .n_fft = 2 * g_grid.n_fft, .delay_depth = D };
    Snap snap;
    void *pool;
    int id;

    CHECK(inst_config(&a, D) == 0);
    CHECK(inst_config(&deep, 2 * D) == 0);
    CHECK(inst_config(&shipped, SHIPPED_D) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&a.cfg, &a.req) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&deep.cfg, &deep.req) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&shipped.cfg, &shipped.req) == 0);
    CHECK(req_for_grid(&rate_spec, &req_rate) == 0);
    CHECK(req_for_grid(&fft_spec, &req_fft) == 0);

    printf("pool D=%d = %llu B\npool D=%d = %llu B\n", D,
           (unsigned long long)a.req.bytes, SHIPPED_D,
           (unsigned long long)shipped.req.bytes);

    /* The pool holds the signal staging only: it follows the grid, not D (the
     * attention rings are state tensors, the application's). D moves the stale-pool
     * identity. */
    CHECK(deep.req.bytes == a.req.bytes && shipped.req.bytes == a.req.bytes);
    CHECK(deep.req.build_flags_hash != a.req.build_flags_hash);
    CHECK(shipped.req.build_flags_hash != a.req.build_flags_hash);
    /* Another sample rate keeps the size and moves the hash; another n_fft moves both. */
    CHECK(req_rate.bytes == a.req.bytes && req_rate.build_flags_hash != a.req.build_flags_hash);
    CHECK(req_fft.bytes != a.req.bytes && req_fft.build_flags_hash != a.req.build_flags_hash);
    CHECK(a.req.descriptor_version == DEEPVQE_PREPOST_DESCRIPTOR_VERSION);
    CHECK(a.req.layout_version == DEEPVQE_PREPOST_LAYOUT_VERSION);
    CHECK(a.req.reserved0 == 0u && a.req.reserved1 == 0u);
    CHECK(a.req.alignment == DEEPVQE_PREPOST_ALIGNMENT);

    pool = alloc_aligned(a.req.alignment, (size_t)a.req.bytes);
    CHECK(pool != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(deepvqe_prepost_init_ex(pool, (size_t)a.req.bytes, &a.cfg, &a.req) != NULL);
    CHECK(!bytes_are(pool, (size_t)a.req.bytes, 0xa5u) && inst_state_zero(&a));
    /* Poison the pool and the registered buffers: a refusal must leave both as they are. */
    memset(pool, 0xa5, (size_t)a.req.bytes);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (size_t i = 0; i < a.cfg.state_elements[id]; ++i) a.buf[id][i] = NAN;
    CHECK(snap_take(&snap, &a) == 0);
    /* ...and refuse a pool recorded for another D or another grid. */
    CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &deep.req, &snap));
    CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &req_rate, &snap));
    CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &req_fft, &snap));
    {
        DeepVqePrepostMemReq stale;
        stale = a.req;
        ++stale.descriptor_version;
        CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &stale, &snap));
        stale = a.req;
        ++stale.layout_version;
        CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &stale, &snap));
        stale = a.req;
        ++stale.build_flags_hash;
        CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &stale, &snap));
        stale = a.req;
        ++stale.bytes;
        CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &stale, &snap));
        stale = a.req;
        ++stale.alignment;
        CHECK(stale_refused(pool, (size_t)a.req.bytes, &a, &stale, &snap));
        /* Unperturbed again, so the five refusals above are attributable to
         * the perturbation and not to the pool going stale on its own. */
        CHECK(deepvqe_prepost_init_ex(pool, (size_t)a.req.bytes, &a.cfg, &a.req) != NULL);
        CHECK(!bytes_are(pool, (size_t)a.req.bytes, 0xa5u) && inst_state_zero(&a));
    }
    snap_free(&snap);
    free(pool);
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(void) {
    Inst heap, stack;
    DeepVqePrepostConfig bad;

    CHECK(inst_config(&heap, D) == 0 && inst_init(&heap, 1) == 0);
    CHECK(deepvqe_prepost_hop_size(heap.p) == g_grid.hop);
    CHECK(deepvqe_prepost_num_bins(heap.p) == g_grid.n_bins);
    CHECK(deepvqe_prepost_descriptor(heap.p) != NULL);
    CHECK(deepvqe_prepost_descriptor(heap.p)->delay_depth == D);
    CHECK(deepvqe_prepost_descriptor(heap.p)->layout_version ==
          DEEPVQE_PREPOST_LAYOUT_VERSION);
    CHECK(drive_spec(heap.p, 0, SHORT_HOPS, outs_a) == 0);
    CHECK(outs_nonzero(outs_a, SHORT_HOPS));

    /* A created instance and a pool instance are the same object. */
    CHECK(inst_config(&stack, D) == 0 && inst_init(&stack, 0) == 0);
    CHECK(drive_spec(stack.p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still runs
     * afterwards. (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    deepvqe_prepost_destroy(stack.p);
    deepvqe_prepost_destroy(stack.p);
    deepvqe_prepost_reset(stack.p);
    memset(outs_b, 0, sizeof outs_b);
    CHECK(drive_spec(stack.p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));

    deepvqe_prepost_destroy(heap.p);
    deepvqe_prepost_destroy(NULL);
    heap.p = NULL;
    inst_free(&heap);
    inst_free(&stack);

    /* _create refuses what _get_mem_size refuses, and allocates nothing. */
    CHECK(deepvqe_prepost_create(NULL) == NULL);
    CHECK(deepvqe_prepost_config_defaults(&bad, &g_grid) == 0);
    bad.grid.delay_depth = 0;
    CHECK(deepvqe_prepost_create(&bad) == NULL);

    /* Accessors on NULL. */
    CHECK(deepvqe_prepost_hop_size(NULL) == -1);
    CHECK(deepvqe_prepost_num_bins(NULL) == -1);
    CHECK(deepvqe_prepost_descriptor(NULL) == NULL);
    deepvqe_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

static int case_reject(void) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req, guard;
    DeepVqeGrid bad_grid, edge;
    Inst inst;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    CHECK(inst_config(&inst, D) == 0);
    cfg = inst.cfg;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(deepvqe_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, NULL) == -1);
    CHECK(deepvqe_prepost_config_defaults(NULL, &g_grid) == -1);
    CHECK(deepvqe_prepost_config_defaults(&cfg, NULL) == -1);
    cfg = inst.cfg;

    /* An inconsistent grid. *req is left untouched on a refusal. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    bad_grid = g_grid;
    bad_grid.n_bins += 1;
    CHECK(deepvqe_prepost_config_defaults(&cfg, &bad_grid) == -1);
    cfg = inst.cfg;
    cfg.grid = bad_grid;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);

    /* D outside [MIN_D, MAX_D]. */
    bad_grid = g_grid;
    bad_grid.delay_depth = DEEPVQE_PREPOST_MIN_D - 1;
    cfg.grid = bad_grid;
    req = guard;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    bad_grid.delay_depth = DEEPVQE_PREPOST_MAX_D + 1;
    cfg.grid = bad_grid;
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(deepvqe_prepost_config_defaults(&cfg, &bad_grid) == -1);
    /* Both ends of the range are accepted. */
    edge = grid_with_depth(DEEPVQE_PREPOST_MIN_D);
    CHECK(edge.n_fft != 0 && deepvqe_prepost_config_defaults(&cfg, &edge) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    edge = grid_with_depth(DEEPVQE_PREPOST_MAX_D);
    CHECK(edge.n_fft != 0 && deepvqe_prepost_config_defaults(&cfg, &edge) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);

    /* _init: NULL pool, undersized pool, misaligned pool, then success. */
    cfg = inst.cfg;
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
        DeepVqePrepost *p = deepvqe_prepost_init(pool, (size_t)req.bytes, &cfg);
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        static float got_re[BINS_MAX], got_im[BINS_MAX];
        int id;
        CHECK(p != NULL);
        CHECK(deepvqe_prepost_pre_process_freq(NULL, specs[0].mr, specs[0].mi,
                                               specs[0].fr, specs[0].fi) == -1);
        CHECK(deepvqe_prepost_pre_process_freq(p, NULL, specs[0].mi,
                                               specs[0].fr, specs[0].fi) == -1);
        CHECK(deepvqe_prepost_pre_process_freq(p, specs[0].mr, specs[0].mi,
                                               specs[0].fr, NULL) == -1);
        CHECK(deepvqe_prepost_post_process_freq(NULL, got_re, got_im) == -1);
        CHECK(deepvqe_prepost_post_process_freq(p, NULL, got_im) == -1);
        /* Nothing is open yet: the three frame stages all refuse. */
        CHECK(deepvqe_prepost_frame_open(p, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_open(NULL, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_finish(p) == -1);
        CHECK(deepvqe_prepost_frame_finish(NULL) == -1);
        CHECK(deepvqe_prepost_frame_bypass(p) == -1);
        CHECK(deepvqe_prepost_frame_bypass(NULL) == -1);
        /* With no frame open, post_process still emits a defined (silent) hop. */
        CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
        CHECK(all_zero(got_re, BINS) && all_zero(got_im, BINS));

        CHECK(deepvqe_prepost_pre_process_freq(p, specs[0].mr, specs[0].mi,
                                               specs[0].fr, specs[0].fi) == 1);
        CHECK(deepvqe_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_open(p, &inputs, NULL) == -1);
        /* A frame is open: the spectrum cannot be emitted yet. */
        CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == -1);
        /* A partial accelerator write is refused and moves nothing; the
         * frame stays open so the caller can fail closed. */
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.taps[outputs.taps_elements / 3] = NAN;
        /* Teeth: the state is non-zero going in, so "zero" below is the
         * commit's doing. */
        CHECK(!all_zero(outputs.state_out[DEEPVQE_STATE_H_GRU],
                        outputs.state_elements[DEEPVQE_STATE_H_GRU]));
        CHECK(deepvqe_prepost_frame_finish(p) == -1);
        CHECK(deepvqe_prepost_frame_bypass(p) == 0);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id], outputs.state_elements[id]));

        /* The same refusal when the hole is in the newest frame of ONE state
         * tensor rather than in the taps -- the finite check walks the state,
         * not just the head. */
        CHECK(deepvqe_prepost_pre_process_freq(p, specs[1].mr, specs[1].mi,
                                               specs[1].fr, specs[1].fi) == 1);
        CHECK(deepvqe_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_finish(p) == 0);   /* teeth: it can pass */
        CHECK(deepvqe_prepost_post_process_freq(p, got_re, got_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(p, specs[2].mr, specs[2].mi,
                                               specs[2].fr, specs[2].fi) == 1);
        CHECK(deepvqe_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.state_out[DEEPVQE_STATE_MIC3_HISTORY][
            newest_element(DEEPVQE_STATE_MIC3_HISTORY, 0) + 5u] = NAN;
        CHECK(deepvqe_prepost_frame_finish(p) == -1);
        CHECK(deepvqe_prepost_frame_bypass(p) == 0);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id], outputs.state_elements[id]));
    }

    free(pool);
    inst_free(&inst);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(void) {
    Inst inst;

    CHECK(inst_make(&inst) == 0);
    CHECK(drive_spec(inst.p, 0, SHORT_HOPS, outs_a) == 0);
    CHECK(outs_nonzero(outs_a, SHORT_HOPS));
    /* Without the reset the second run continues from a warm state, so this
     * comparison has teeth: assert that it does NOT match first. */
    CHECK(drive_spec(inst.p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(!outs_identical(outs_a, outs_b, SHORT_HOPS));

    deepvqe_prepost_reset(inst.p);
    CHECK(inst_state_zero(&inst));
    CHECK(drive_spec(inst.p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));

    /* Reset mid-frame drops the open frame instead of stranding it. */
    {
        DeepVqePrepostInputs inputs;
        DeepVqePrepostOutputs outputs;
        CHECK(deepvqe_prepost_pre_process_freq(inst.p, specs[0].mr, specs[0].mi,
                                               specs[0].fr, specs[0].fi) == 1);
        CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        deepvqe_prepost_reset(inst.p);
        CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
        CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == -1);
        CHECK(deepvqe_prepost_frame_bypass(inst.p) == -1);
        memset(outs_b, 0, sizeof outs_b);
        CHECK(drive_spec(inst.p, 0, SHORT_HOPS, outs_b) == 0);
        CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));
    }

    inst_free(&inst);
    return 0;
}

/* ---- skip: FAIL CLOSED, and no state or ring advance ----------------- */

static void make_frame(float *re, float *im, float seed) {
    size_t bin;
    for (bin = 0; bin < BINS; ++bin) {
        re[bin] = seed + 0.5f - (float)bin / (float)BINS;
        im[bin] = (float)bin / (float)BINS - 0.25f - seed;
    }
}

/* One committed frame. */
static int commit_freq(DeepVqePrepost *p, const float *mic_re,
                       const float *mic_im, const float *far_re,
                       const float *far_im, float *out_re, float *out_im) {
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    if (deepvqe_prepost_pre_process_freq(p, mic_re, mic_im,
                                         far_re, far_im) != 1) return -1;
    if (deepvqe_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
    if (fake_run(&inputs, &outputs) != 0) return -1;
    if (deepvqe_prepost_frame_finish(p) != 0) return -1;
    if (deepvqe_prepost_post_process_freq(p, out_re, out_im) != 0) return -1;
    return 0;
}

static int case_skip(void) {
    Inst inst;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float f1_re[BINS_MAX], f1_im[BINS_MAX];
    static float f2_re[BINS_MAX], f2_im[BINS_MAX];
    static float f3_re[BINS_MAX], f3_im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    static float got_re[BINS_MAX], got_im[BINS_MAX];
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

    CHECK(inst_make(&inst) == 0);

    /* Frame 1: a real commit, so the state tensors are NON-ZERO below. */
    CHECK(commit_freq(inst.p, f1_re, f1_im, far_re, far_im, got_re, got_im) == 0);
    /* Teeth for the mute assertion: a committed frame is NOT all zeros. */
    CHECK(!all_zero(got_re, BINS) || !all_zero(got_im, BINS));

    /* Frame 2: the accelerator produces a COMPLETE, perfectly committable
     * result, and the host takes the fail-closed identity anyway.
     *
     * The full write is the point. Skipping after a partial write would
     * prove nothing -- commit() refuses that frame on its own -- so a skip
     * that quietly committed would still look correct. Here the only thing
     * standing between the run and a commit is frame_bypass's contract. The
     * state is in place, so it is whatever the accelerator wrote: skip does
     * not undo it. */
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, f2_re, f2_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    {
        int nonzero = 0;
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            elements[id] = inputs.state_elements[id];
            CHECK(elements[id] == state_count(id, D));
            CHECK(inputs.state[id] == inst.buf[id]);
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
    CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, got_re, got_im) == 0);
    /* FAIL CLOSED: silence, NOT the microphone passed through. Passing the
     * raw microphone would emit the full uncancelled echo -- see
     * deepvqe_prepost_frame_bypass() in deepvqe_prepost.h. */
    CHECK(all_zero(got_re, BINS));
    CHECK(all_zero(got_im, BINS));
    CHECK(!identical(got_re, f2_re, BINS));

    /* Frame 3: the state is exactly what the accelerator wrote in frame 2
     * -- the skip neither rolled it back nor zeroed it. */
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, f3_re, f3_im,
                                           far_re, far_im) == 1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        CHECK(inputs.state_elements[id] == elements[id]);
        CHECK(memcmp(inputs.state[id], written[id],
                     elements[id] * sizeof(float)) == 0);
    }
    /* A skip WITHOUT frame_open behind it is legal: the accelerator may
     * have failed before it was ever handed the tensors. */
    CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, got_re, got_im) == 0);
    CHECK(all_zero(got_re, BINS));
    CHECK(all_zero(got_im, BINS));

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        free(before[id]);
        free(written[id]);
    }
    inst_free(&inst);

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
     * measuring a constant. Each instance has its own state buffers. */
    {
        Inst a, b, c, d;
        static float a_re[BINS_MAX], a_im[BINS_MAX];
        static float b_re[BINS_MAX], b_im[BINS_MAX];
        static float c_re[BINS_MAX], c_im[BINS_MAX];
        static float d_re[BINS_MAX], d_im[BINS_MAX];

        CHECK(inst_make(&a) == 0 && inst_make(&b) == 0);
        CHECK(inst_make(&c) == 0 && inst_make(&d) == 0);

        CHECK(commit_freq(a.p, f1_re, f1_im, far_re, far_im,
                          a_re, a_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(a.p, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_open(a.p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_bypass(a.p) == 0);
        CHECK(deepvqe_prepost_post_process_freq(a.p, a_re, a_im) == 0);
        CHECK(commit_freq(a.p, f3_re, f3_im, far_re, far_im,
                          a_re, a_im) == 0);

        CHECK(commit_freq(d.p, f1_re, f1_im, far_re, far_im,
                          d_re, d_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(d.p, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_open(d.p, &inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_bypass(d.p) == 0);   /* no fake_run */
        CHECK(deepvqe_prepost_post_process_freq(d.p, d_re, d_im) == 0);
        CHECK(commit_freq(d.p, f3_re, f3_im, far_re, far_im,
                          d_re, d_im) == 0);

        CHECK(commit_freq(b.p, f1_re, f1_im, far_re, far_im,
                          b_re, b_im) == 0);
        CHECK(commit_freq(b.p, f3_re, f3_im, far_re, far_im,
                          b_re, b_im) == 0);

        CHECK(commit_freq(c.p, f1_re, f1_im, far_re, far_im,
                          c_re, c_im) == 0);
        CHECK(commit_freq(c.p, f2_re, f2_im, far_re, far_im,
                          c_re, c_im) == 0);
        CHECK(commit_freq(c.p, f3_re, f3_im, far_re, far_im,
                          c_re, c_im) == 0);

        CHECK(!all_zero(d_re, BINS));
        CHECK(identical(d_re, b_re, BINS));
        CHECK(identical(d_im, b_im, BINS));
        CHECK(!identical(a_re, b_re, BINS));
        CHECK(!identical(a_re, c_re, BINS));
        CHECK(!identical(a_im, c_im, BINS));
        CHECK(!identical(d_re, c_re, BINS));
        CHECK(!identical(d_im, c_im, BINS));

        inst_free(&a);
        inst_free(&b);
        inst_free(&c);
        inst_free(&d);
    }
    return 0;
}

/* ---- inplace: one buffer per state tensor, validate-only commit ------- */

static int state_all_zero(const DeepVqePrepostInputs *in) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (!all_zero(in->state[id], in->state_elements[id])) return 0;
    return 1;
}

static int case_inplace(void) {
    Inst inst, other;
    DeepVqePrepostInputs inputs, first_in;
    DeepVqePrepostOutputs outputs, first_out;
    static float f1_re[BINS_MAX], f1_im[BINS_MAX];
    static float f2_re[BINS_MAX], f2_im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    static float got_re[BINS_MAX], got_im[BINS_MAX];
    float *expect[DEEPVQE_STATE_COUNT];
    int id, hop, target;
    size_t i;

    /* (1) Stream: through a real stream the state input and output are one
     * pointer per tensor -- the registered buffer, the same on every frame --
     * and the frame-t output is what frame t+1 reads. Healthy-stream numerics
     * are the equiv/lifecycle/reset cases (byte-identical to the hand-composed
     * reference); this pins the aliasing itself. */
    CHECK(inst_make(&inst) == 0);
    memset(&first_in, 0, sizeof first_in);
    memset(&first_out, 0, sizeof first_out);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        expect[id] = (float *)malloc(state_count(id, D) * sizeof(float));
        CHECK(expect[id] != NULL);
    }
    for (hop = 0; hop < SHORT_HOPS; ++hop) {
        CHECK(deepvqe_prepost_pre_process_freq(inst.p, specs[hop].mr, specs[hop].mi,
                                               specs[hop].fr, specs[hop].fi) == 1);
        CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
        if (hop == 0) {
            first_in = inputs;
            first_out = outputs;
            CHECK(state_all_zero(&inputs));      /* starts cold */
        }
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            CHECK(inputs.state[id] == (const float *)outputs.state_out[id]);
            CHECK(inputs.state[id] == inst.buf[id]);
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
        CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
        /* commit leaves the state exactly as written. */
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(memcmp(first_in.state[id], expect[id],
                         first_in.state_elements[id] * sizeof(float)) == 0);
        CHECK(deepvqe_prepost_post_process_freq(inst.p, got_re, got_im) == 0);
    }
    CHECK(!state_all_zero(&first_in));   /* the stream really moved the state */
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(expect[id]);
    inst_free(&inst);

    /* (2) MUTATION: one non-finite element in ONE state tensor, injected
     * through the in-place pointer, must make commit return -1, leave ALL
     * sixteen tensors zero, not advance the CCM ring, and leave the next
     * healthy frame working. Every tensor takes the injection in turn (first
     * and last element, NaN and Inf), and the same frame WITHOUT the
     * injection commits, so the refusal is attributable to the injection. */
    CHECK(inst_make(&other) == 0);
    make_frame(f1_re, f1_im, 0.0f);
    make_frame(f2_re, f2_im, 0.125f);
    make_frame(far_re, far_im, 0.0625f);

    for (target = 0; target < 2 * DEEPVQE_STATE_COUNT; ++target) {
        int tensor = target % DEEPVQE_STATE_COUNT;
        int last = target >= DEEPVQE_STATE_COUNT;   /* first half: elem 0 */
        float poison = (target % 2) ? INFINITY : NAN;
        float *victim;
        size_t index;

        deepvqe_prepost_reset(other.p);
        CHECK(commit_freq(other.p, f1_re, f1_im, far_re, far_im,
                          got_re, got_im) == 0);

        /* Control: the identical frame, uninjected, commits. */
        CHECK(deepvqe_prepost_pre_process_freq(other.p, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_open(other.p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(!state_all_zero(&inputs));
        CHECK(deepvqe_prepost_frame_finish(other.p) == 0);
        CHECK(deepvqe_prepost_post_process_freq(other.p, got_re, got_im) == 0);
        CHECK(!all_zero(got_re, BINS));

        /* Same history, now poisoned. */
        deepvqe_prepost_reset(other.p);
        CHECK(commit_freq(other.p, f1_re, f1_im, far_re, far_im,
                          got_re, got_im) == 0);
        CHECK(deepvqe_prepost_pre_process_freq(other.p, f2_re, f2_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_open(other.p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        index = newest_element(tensor, last);
        victim = outputs.state_out[tensor];
        victim[index] = poison;
        CHECK(deepvqe_prepost_frame_finish(other.p) == -1);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
            CHECK(all_zero(outputs.state_out[id], outputs.state_elements[id]));
        CHECK(state_all_zero(&inputs));
        CHECK(deepvqe_prepost_frame_finish(other.p) == -1);   /* disarmed */

        /* The next healthy frame works: retry through a fresh frame_open
         * from the cold state. */
        CHECK(deepvqe_prepost_frame_open(other.p, &inputs, &outputs) == 0);
        CHECK(state_all_zero(&inputs));
        CHECK(fake_run(&inputs, &outputs) == 0);
        CHECK(deepvqe_prepost_frame_finish(other.p) == 0);
        CHECK(deepvqe_prepost_post_process_freq(other.p, got_re, got_im) == 0);
        CHECK(!all_zero(got_re, BINS));
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
        Inst y;
        static float x_re[BINS_MAX], x_im[BINS_MAX];
        static float y_re[BINS_MAX], y_im[BINS_MAX];

        CHECK(inst_make(&y) == 0);

        deepvqe_prepost_reset(other.p);
        CHECK(deepvqe_prepost_pre_process_freq(other.p, f1_re, f1_im,
                                               far_re, far_im) == 1);
        CHECK(deepvqe_prepost_frame_open(other.p, &inputs, &outputs) == 0);
        CHECK(fake_run(&inputs, &outputs) == 0);
        outputs.state_out[DEEPVQE_STATE_H_GRU][3] = NAN;
        CHECK(deepvqe_prepost_frame_finish(other.p) == -1);
        CHECK(deepvqe_prepost_frame_bypass(other.p) == 0);
        CHECK(deepvqe_prepost_post_process_freq(other.p, x_re, x_im) == 0);
        CHECK(all_zero(x_re, BINS));
        CHECK(commit_freq(other.p, f2_re, f2_im, far_re, far_im, x_re, x_im) == 0);

        CHECK(commit_freq(y.p, f2_re, f2_im, far_re, far_im, y_re, y_im) == 0);
        CHECK(!all_zero(y_re, BINS));
        CHECK(identical(x_re, y_re, BINS));
        CHECK(identical(x_im, y_im, BINS));

        inst_free(&y);
    }

    inst_free(&other);
    return 0;
}

/* ---- guard: the three transaction gates ------------------------------ */

static int case_guard(void) {
    Inst inst;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float f_re[BINS_MAX], f_im[BINS_MAX];
    static float g_re[BINS_MAX], g_im[BINS_MAX];
    int id;
    size_t i;

    CHECK(inst_make(&inst) == 0);
    make_frame(f_re, f_im, 0.0f);
    make_frame(g_re, g_im, 0.25f);

    /* (1) A spectrum is refused rather than stacked on top of an unfinished
     * frame, and the refusal does not damage the frame that IS open. */
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, f_re, f_im, g_re, g_im) == 1);
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, g_re, g_im, f_re, f_im) == -1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, f_re, f_im) == 0);

    /* (2) A commit with no frame_open behind it is refused, so an
     * accelerator that never ran cannot pass untouched buffers off as a
     * result -- and the same frame commits normally once it has. */
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, g_re, g_im, f_re, f_im) == 1);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, f_re, f_im) == 0);

    /* (3) A refused commit DISARMS the transaction: re-committing without a
     * fresh frame_open must not succeed by the second walk happening to
     * find the same buffers. The fresh frame_open then re-fills the head
     * output with NaN, which is what makes the finite check a partial-write
     * detector rather than a stale-value detector on the taps. The state
     * tensors are NOT re-filled: they are the accelerator's input, and after
     * the refused commit they are the cold (all-zero) state. */
    CHECK(deepvqe_prepost_pre_process_freq(inst.p, f_re, f_im, g_re, g_im) == 1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    CHECK(fake_run(&inputs, &outputs) == 0);
    outputs.state_out[DEEPVQE_STATE_UP3_HISTORY][newest_element(
        DEEPVQE_STATE_UP3_HISTORY, 0)] = NAN;
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_frame_open(inst.p, &inputs, &outputs) == 0);
    {
        size_t defined = 0;
        for (i = 0; i < outputs.taps_elements; ++i)
            if (!isnan(outputs.taps[i])) ++defined;
        CHECK(defined == 0u);
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            CHECK(inputs.state[id] == outputs.state_out[id]);
            CHECK(all_zero(inputs.state[id], inputs.state_elements[id]));
        }
        CHECK(inst_state_zero(&inst));
    }
    CHECK(fake_run(&inputs, &outputs) == 0);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, f_re, f_im) == 0);

    inst_free(&inst);
    return 0;
}

/* ---- whole: commit reads the whole of every registered window -------- */

/* Open one frame, run the stand-in, and return the views so the caller
 * can edit what the "accelerator" wrote before committing. */
static int open_and_run(DeepVqePrepost *p, DeepVqePrepostInputs *inputs,
                        DeepVqePrepostOutputs *outputs, float seed) {
    static float re[BINS_MAX], im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    make_frame(re, im, seed);
    make_frame(far_re, far_im, seed + 0.0625f);
    if (deepvqe_prepost_pre_process_freq(p, re, im, far_re, far_im) != 1)
        return -1;
    if (deepvqe_prepost_frame_open(p, inputs, outputs) != 0) return -1;
    return fake_run(inputs, outputs);
}

static int case_whole(void) {
    Inst inst;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    int id, which, other;

    CHECK(inst_make(&inst) == 0);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        /* The whole buffer is read: a poisoned value in the oldest frame (the
         * slot a window drops) is refused like one in the newest, and takes
         * every tensor with it. */
        for (which = 0; which < 2; ++which) {
            size_t at = id == DEEPVQE_STATE_H_GRU ? state_count(id, D) / 2u
                                                  : dropped_element(id);
            uint32_t before = deepvqe_prepost_state_generation(inst.p);
            CHECK(open_and_run(inst.p, &inputs, &outputs, 0.125f) == 0);
            CHECK(!all_zero(outputs.state_out[id], outputs.state_elements[id]));
            outputs.state_out[id][at] = which ? INFINITY : NAN;
            CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
            CHECK(deepvqe_prepost_state_generation(inst.p) == before + 1u);
            for (other = 0; other < DEEPVQE_STATE_COUNT; ++other)
                CHECK(all_zero(outputs.state_out[other],
                               outputs.state_elements[other]));
            CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
        }
        /* The newest frame is read: poison at its first and last element is
         * refused and takes every tensor with it. */
        for (which = 0; which < 2; ++which) {
            size_t at = newest_element(id, which);
            uint32_t before = deepvqe_prepost_state_generation(inst.p);
            CHECK(open_and_run(inst.p, &inputs, &outputs, 0.25f) == 0);
            CHECK(!all_zero(outputs.state_out[id], outputs.state_elements[id]));
            outputs.state_out[id][at] = NAN;
            CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
            CHECK(deepvqe_prepost_state_generation(inst.p) == before + 1u);
            for (other = 0; other < DEEPVQE_STATE_COUNT; ++other)
                CHECK(all_zero(outputs.state_out[other],
                               outputs.state_elements[other]));
            CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
        }
    }
    /* h_gru is checked whole, in the middle as well as at the end. */
    {
        size_t last = outputs.state_elements[DEEPVQE_STATE_H_GRU] - 1u;
        size_t middle = last / 2u;
        size_t picks[2];
        picks[0] = middle;
        picks[1] = last;
        for (which = 0; which < 2; ++which) {
            CHECK(open_and_run(inst.p, &inputs, &outputs, 0.375f) == 0);
            outputs.state_out[DEEPVQE_STATE_H_GRU][picks[which]] = INFINITY;
            CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
            CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
        }
    }
    inst_free(&inst);
    return 0;
}

/* ---- generation: the restart counter ---------------------------------- */

static int case_generation(void) {
    Inst inst;
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    static float got_re[BINS_MAX], got_im[BINS_MAX];

    CHECK(deepvqe_prepost_state_generation(NULL) == 0u);
    CHECK(inst_make(&inst) == 0);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 0u);

    /* Healthy commits, skips and new frames change nothing. */
    CHECK(open_and_run(inst.p, &inputs, &outputs, 0.0f) == 0);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(inst.p, got_re, got_im) == 0);
    CHECK(open_and_run(inst.p, &inputs, &outputs, 0.5f) == 0);
    CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 0u);

    /* reset advances it, once per call. */
    deepvqe_prepost_reset(inst.p);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 1u);
    deepvqe_prepost_reset(inst.p);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 2u);

    /* A refused commit advances it once; the refused retry does not. */
    CHECK(open_and_run(inst.p, &inputs, &outputs, 0.25f) == 0);
    outputs.taps[outputs.taps_elements / 2] = NAN;
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 3u);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 3u);
    CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);

    /* A commit refused for a hole in a state tensor counts the same. */
    CHECK(open_and_run(inst.p, &inputs, &outputs, 0.75f) == 0);
    outputs.state_out[DEEPVQE_STATE_UP3_HISTORY][newest_element(
        DEEPVQE_STATE_UP3_HISTORY, 0)] = NAN;
    CHECK(deepvqe_prepost_frame_finish(inst.p) == -1);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 4u);
    CHECK(deepvqe_prepost_frame_bypass(inst.p) == 0);

    /* ...and the next healthy frame leaves it alone. */
    CHECK(open_and_run(inst.p, &inputs, &outputs, 0.125f) == 0);
    CHECK(deepvqe_prepost_frame_finish(inst.p) == 0);
    CHECK(deepvqe_prepost_state_generation(inst.p) == 4u);
    inst_free(&inst);
    return 0;
}

/* ---- boundary: names and descriptor validation ------------------------ */

/* The eight descriptor fields that are the library's: validated against the grid. */
static const char *const dv_external_field_names[8] = {
    "layout_version", "sample_rate", "fft_size", "hop_size", "spectrum_bins",
    "time_order", "freq_taps", "state_tensor_count"
};

static int perturbation_refused(const DeepVqePrepostDescriptor *base, int field) {
    DeepVqePrepostDescriptor probe = *base;
    switch (field) {
    case 0:  probe.layout_version += 1u;      break;
    case 1:  probe.sample_rate += 1;          break;
    case 2:  probe.fft_size += 1;             break;
    case 3:  probe.hop_size += 1;             break;
    case 4:  probe.spectrum_bins += 1;        break;
    case 5:  probe.time_order += 1;           break;
    case 6:  probe.freq_taps += 1;            break;
    case 7:  probe.state_tensor_count += 1;   break;
    default: return 0;
    }
    if (deepvqe_prepost_descriptor_validate(&g_grid, &probe) == -1) return 1;
    fprintf(stderr, "descriptor field '%s' +1 was ACCEPTED\n",
            dv_external_field_names[field]);
    return 0;
}

static int case_boundary(void) {
    DeepVqePrepostDescriptor base, probe;
    int id, field;

    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        const char *name = deepvqe_prepost_state_name(id);
        const char *name_out = deepvqe_prepost_state_name_out(id);
        size_t length;

        CHECK(name != NULL);
        CHECK(name_out != NULL);
        length = strlen(name);
        /* The graph's handoff convention: the output tensor is the input
         * tensor's name plus "_out". */
        CHECK(strncmp(name_out, name, length) == 0);
        CHECK(strcmp(name_out + length, "_out") == 0);
    }

    /* Bad ids. */
    CHECK(deepvqe_prepost_state_name(-1) == NULL);
    CHECK(deepvqe_prepost_state_name(DEEPVQE_STATE_COUNT) == NULL);
    CHECK(deepvqe_prepost_state_name_out(-1) == NULL);
    CHECK(deepvqe_prepost_state_name_out(DEEPVQE_STATE_COUNT) == NULL);

    /* The validator. */
    CHECK(deepvqe_prepost_descriptor_default(&g_grid, &base) == 0);
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &base) == 0);
    CHECK(deepvqe_prepost_descriptor_default(&g_grid, NULL) == -1);
    CHECK(deepvqe_prepost_descriptor_default(NULL, &base) == -1);
    CHECK(deepvqe_prepost_descriptor_default(&g_grid, &base) == 0);
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, NULL) == -1);
    CHECK(deepvqe_prepost_descriptor_validate(NULL, &base) == -1);
    probe = base;
    probe.delay_depth = DEEPVQE_PREPOST_MIN_D - 1;
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &probe) == -1);
    probe = base;
    probe.delay_depth = DEEPVQE_PREPOST_MAX_D + 1;
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &probe) == -1);
    /* delay_depth is range-checked, not pinned: the export depth is the
     * application's to know. */
    probe = base;
    probe.delay_depth = base.delay_depth + 1;
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &probe) == 0);

    for (field = 0; field < 8; ++field) {
        CHECK(perturbation_refused(&base, field));
        /* The unperturbed descriptor is accepted again, so the refusal
         * above is attributable to the perturbation. */
        CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &base) == 0);
    }

    /* The four model-interior fields are the graph's: zero here, never read. */
    CHECK(base.conv_window_frames == 0 && base.score_window_frames == 0 &&
          base.gru_layers == 0 && base.gru_hidden == 0);
    probe = base;
    probe.conv_window_frames = 4;
    probe.score_window_frames = 5;
    probe.gru_layers = 1;
    probe.gru_hidden = 192;
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &probe) == 0);
    probe.conv_window_frames += 1;
    probe.score_window_frames += 1;
    probe.gru_layers += 1;
    probe.gru_hidden += 1;
    CHECK(deepvqe_prepost_descriptor_validate(&g_grid, &probe) == 0);
    return 0;
}

/* ---- descriptor: validate + shape dump from argv ---------------------- */

/* The descriptor is a graph's own, measured by the Python side. The grid it describes
 * is the one under test; the state shapes printed are the test-owned table's, which the
 * Python side compares with the graph's tensors. */
static int case_descriptor(char **argv) {
    DeepVqePrepostDescriptor descriptor;
    int dims[4];
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

    printf("validate=%d\n", deepvqe_prepost_descriptor_validate(&g_grid, &descriptor));
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        int rank;
        memset(dims, 0, sizeof dims);
        rank = state_dims(id, descriptor.delay_depth, dims);
        printf("%s %d %d %d %d %d\n", deepvqe_prepost_state_name(id), rank,
               dims[0], dims[1], dims[2], dims[3]);
    }
    return 0;
}

/* ---- hash: the stale-pool identity of this build ---------------------- */

static int case_hash(void) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    CHECK(deepvqe_prepost_config_defaults(&cfg, &g_grid) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    /* The ABI token folded into the hash follows the C target, which a
     * cross-built driver run under an emulator does not share with the
     * Python host. */
    printf("hash=%u layout=%u ptr_bytes=%zu\n", req.build_flags_hash,
           req.layout_version, sizeof(void *));
    return 0;
}

/* ---- board: nothing registered, the accelerator owns the state ------- */

#define FENCE 64u

/* The state views of a board instance are empty on both sides. */
static int views_empty(const DeepVqePrepostInputs *in, const DeepVqePrepostOutputs *out) {
    int id;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        if (in->state[id] || in->state_elements[id] ||
            out->state_out[id] || out->state_elements[id]) return 0;
    return 1;
}

/* One hop through a head-only stand-in (the taps depend on the hop only): returns frame_finish's
 * status, or 2 when the boundary itself is wrong. `poison` leaves one tap NaN. */
static int head_hop(DeepVqePrepost *p, int hop, int empty, int poison) {
    DeepVqePrepostInputs in;
    DeepVqePrepostOutputs out;
    size_t i;
    if (deepvqe_prepost_pre_process_freq(p, specs[hop].mr, specs[hop].mi,
                                         specs[hop].fr, specs[hop].fi) != 1) return 2;
    if (deepvqe_prepost_frame_open(p, &in, &out) != 0) return 2;
    for (i = 0; i < out.taps_elements; ++i) if (!isnan(out.taps[i])) return 2;
    if (empty && !views_empty(&in, &out)) return 2;
    for (i = 0; i < out.taps_elements; ++i) {
        const size_t slot = i % (size_t)TAPS_PER_BIN;
        out.taps[i] = (slot == 2u || slot == 3u)
            ? (float)(0.3 * sin((double)(i + (size_t)hop)))
            : (float)(0.05 * cos((double)(i + (size_t)hop)));
    }
    if (poison) out.taps[out.taps_elements / 3u] = NAN;
    return deepvqe_prepost_frame_finish(p);
}

static int fences_intact(const unsigned char *raw, size_t bytes) {
    return bytes_are(raw, FENCE, 0xa5u) && bytes_are(raw + FENCE + bytes, FENCE, 0xa5u);
}

static int case_board(void) {
    DeepVqePrepostConfig cfg;
    DeepVqePrepostMemReq req;
    DeepVqePrepost *board;
    Inst twin;
    static HopOut got[4], ref[4];
    unsigned char *raw;
    size_t bytes;

    CHECK(deepvqe_prepost_config_defaults(&cfg, &g_grid) == 0);   /* nothing registered */
    CHECK(deepvqe_prepost_get_mem_size(&cfg, &req) == 0);
    bytes = (size_t)req.bytes;
    raw = (unsigned char *)alloc_aligned(req.alignment, bytes + 2u * FENCE);
    CHECK(raw != NULL);
    memset(raw, 0xa5, bytes + 2u * FENCE);
    board = deepvqe_prepost_init_ex(raw + FENCE, bytes, &cfg, &req);
    CHECK(board != NULL);
    /* The same hops through a float host with registered state: the signal path is the same. */
    CHECK(inst_make(&twin) == 0);
    CHECK(deepvqe_prepost_state_generation(board) == 0u);
    CHECK(deepvqe_prepost_state_generation(twin.p) == 0u);

    /* Two healthy hops. */
    CHECK(head_hop(board, 0, 1, 0) == 0 && head_hop(twin.p, 0, 0, 0) == 0);
    CHECK(deepvqe_prepost_post_process_freq(board, got[0].re, got[0].im) == 0);
    CHECK(deepvqe_prepost_post_process_freq(twin.p, ref[0].re, ref[0].im) == 0);
    CHECK(head_hop(board, 1, 1, 0) == 0 && head_hop(twin.p, 1, 0, 0) == 0);
    CHECK(deepvqe_prepost_post_process_freq(board, got[1].re, got[1].im) == 0);
    CHECK(deepvqe_prepost_post_process_freq(twin.p, ref[1].re, ref[1].im) == 0);
    CHECK(outs_identical(got, ref, 2) && outs_nonzero(got, 2));
    CHECK(!identical(got[0].re, got[1].re, BINS));
    CHECK(deepvqe_prepost_state_generation(board) == 0u && fences_intact(raw, bytes));

    /* A non-finite tap: refused, the generation moves by exactly one (nothing registered
     * included), a retry is refused without moving it again, the frame stays open for bypass. */
    CHECK(head_hop(board, 2, 1, 1) == -1 && head_hop(twin.p, 2, 0, 1) == -1);
    CHECK(deepvqe_prepost_state_generation(board) == 1u);
    CHECK(deepvqe_prepost_state_generation(twin.p) == 1u);
    CHECK(deepvqe_prepost_frame_finish(board) == -1);
    CHECK(deepvqe_prepost_state_generation(board) == 1u);
    CHECK(deepvqe_prepost_post_process_freq(board, got[2].re, got[2].im) == -1);
    CHECK(deepvqe_prepost_frame_bypass(board) == 0 && deepvqe_prepost_frame_bypass(twin.p) == 0);
    CHECK(deepvqe_prepost_post_process_freq(board, got[2].re, got[2].im) == 0);
    CHECK(all_zero(got[2].re, BINS) && all_zero(got[2].im, BINS));   /* fail closed: silence */
    CHECK(deepvqe_prepost_state_generation(board) == 1u && fences_intact(raw, bytes));

    /* The next hop recovers and matches the float host again. */
    CHECK(head_hop(board, 3, 1, 0) == 0 && head_hop(twin.p, 3, 0, 0) == 0);
    CHECK(deepvqe_prepost_post_process_freq(board, got[3].re, got[3].im) == 0);
    CHECK(deepvqe_prepost_post_process_freq(twin.p, ref[3].re, ref[3].im) == 0);
    CHECK(identical(got[3].re, ref[3].re, BINS) && identical(got[3].im, ref[3].im, BINS));
    CHECK(!all_zero(got[3].re, BINS));
    CHECK(deepvqe_prepost_state_generation(board) == 1u && fences_intact(raw, bytes));

    free(raw);
    inst_free(&twin);
    return 0;
}

/* ---- registration: all sixteen pairs or none -------------------------- */

/* shape 0: only pair `id`; 1: its pointer without a count; 2: its count without a pointer; 3: all but it. */
static void partial_registration(DeepVqePrepostConfig *cfg, int shape, int id) {
    int j;
    for (j = 0; j < DEEPVQE_STATE_COUNT; ++j) {
        const int mine = j == id;
        if (shape == 1 && mine) cfg->state_elements[j] = 0u;
        else if (shape == 2 && mine) cfg->state[j] = NULL;
        else if ((shape == 0 && !mine) || (shape == 3 && mine)) {
            cfg->state[j] = NULL;
            cfg->state_elements[j] = 0u;
        }
    }
}

/* Every entry refuses with its documented status and changes nothing. */
static int registration_refused(const DeepVqePrepostConfig *cfg, void *pool, size_t bytes,
                                const DeepVqePrepostMemReq *good, const Inst *in,
                                const Snap *snap) {
    DeepVqePrepostMemReq req, guard;
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    return deepvqe_prepost_get_mem_size(cfg, &req) == -1 &&
           memcmp(&req, &guard, sizeof req) == 0 &&
           deepvqe_prepost_init(pool, bytes, cfg) == NULL &&
           deepvqe_prepost_init_ex(pool, bytes, cfg, good) == NULL &&
           deepvqe_prepost_create(cfg) == NULL &&
           bytes_are(pool, bytes, 0xa5u) && snap_same(snap, in);
}

static int case_registration(void) {
    Inst in;
    DeepVqePrepostConfig none, cfg;
    DeepVqePrepostMemReq req, req_none;
    Snap snap;
    size_t bytes;
    int shape, id;

    CHECK(inst_config(&in, D) == 0);   /* the full registration, NaN-filled buffers */
    CHECK(deepvqe_prepost_config_defaults(&none, &in.cfg.grid) == 0);
    /* The two accepted extremes first: the registration is in neither the pool size nor the hash. */
    CHECK(deepvqe_prepost_get_mem_size(&in.cfg, &req) == 0);
    CHECK(deepvqe_prepost_get_mem_size(&none, &req_none) == 0);
    CHECK(memcmp(&req, &req_none, sizeof req) == 0);
    bytes = (size_t)req.bytes;
    in.pool = alloc_aligned(req.alignment, bytes);
    CHECK(in.pool != NULL);
    memset(in.pool, 0xa5, bytes);
    CHECK(snap_take(&snap, &in) == 0);

    for (shape = 0; shape < 4; ++shape) {
        for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
            cfg = in.cfg;
            partial_registration(&cfg, shape, id);
            CHECK(registration_refused(&cfg, in.pool, bytes, &req, &in, &snap));
        }
    }

    /* The full registration is accepted afterwards: buffers zeroed, generation 0, a frame runs. */
    in.p = deepvqe_prepost_init_ex(in.pool, bytes, &in.cfg, &req);
    CHECK(in.p != NULL);
    CHECK(inst_state_zero(&in));
    CHECK(deepvqe_prepost_state_generation(in.p) == 0u);
    CHECK(drive_spec(in.p, 0, 3, outs_a) == 0);
    CHECK(!inst_state_zero(&in));
    snap_free(&snap);
    inst_free(&in);
    return 0;
}

/* ---- grid: deepvqe_grid_make / deepvqe_grid_validate ------------------ */

/* A refused _make returns the status and leaves the grid unchanged. */
#define MAKE_REFUSED(status, ...) do { \
    const DeepVqeGridSpec refused_spec = { __VA_ARGS__ }; \
    DeepVqeGrid refused_out, refused_guard; \
    memset(&refused_out, 0xa5, sizeof refused_out); \
    refused_guard = refused_out; \
    CHECK(deepvqe_grid_make(&refused_out, &refused_spec) == (status)); \
    CHECK(memcmp(&refused_out, &refused_guard, sizeof refused_out) == 0); \
} while (0)

#define VALIDATE_AS(status, field, value) do { \
    DeepVqeGrid bent = made; \
    bent.field = (value); \
    CHECK(deepvqe_grid_validate(&bent) == (status)); \
} while (0)

static int audiolib_builds(int n_fft) {
    AcStftConfig sc;
    ac_stft_config_default(&sc, n_fft);
    return ac_stft_get_mem_size(&sc) != 0u;
}

static int case_grid(void) {
    static const int bad_fft[] = { 0, -512, 8, 513, 1000, 16384 };
    DeepVqeGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = D };
    DeepVqeGrid made;
    size_t i;

    /* Accepted, both ends of the depth range included; the derived fields are the rule. */
    CHECK(audiolib_builds(512) && audiolib_builds(1024));
    CHECK(deepvqe_grid_make(&made, &spec) == DEEPVQE_GRID_OK);
    CHECK(made.sample_rate == 16000 && made.n_fft == 512 && made.hop == 256 &&
          made.n_bins == 257 && made.delay_depth == D);
    spec.n_fft = 1024;
    spec.delay_depth = DEEPVQE_PREPOST_MIN_D;
    CHECK(deepvqe_grid_make(&made, &spec) == DEEPVQE_GRID_OK);
    CHECK(made.hop == 512 && made.n_bins == 513 && made.delay_depth == DEEPVQE_PREPOST_MIN_D);
    spec.delay_depth = DEEPVQE_PREPOST_MAX_D;
    CHECK(deepvqe_grid_make(&made, &spec) == DEEPVQE_GRID_OK);
    CHECK(made.delay_depth == DEEPVQE_PREPOST_MAX_D);

    CHECK(deepvqe_grid_make(NULL, &spec) == DEEPVQE_GRID_ERR_NULL);
    MAKE_REFUSED(DEEPVQE_GRID_ERR_SAMPLE_RATE, .sample_rate = 0, .n_fft = 512, .delay_depth = D);
    MAKE_REFUSED(DEEPVQE_GRID_ERR_SAMPLE_RATE, .sample_rate = -16000, .n_fft = 512, .delay_depth = D);
    for (i = 0; i < sizeof bad_fft / sizeof bad_fft[0]; ++i) {
        CHECK(!audiolib_builds(bad_fft[i]));
        MAKE_REFUSED(DEEPVQE_GRID_ERR_N_FFT, .sample_rate = 16000, .n_fft = bad_fft[i],
                     .delay_depth = D);
    }
    MAKE_REFUSED(DEEPVQE_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512,
                 .delay_depth = DEEPVQE_PREPOST_MIN_D - 1);
    MAKE_REFUSED(DEEPVQE_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512,
                 .delay_depth = DEEPVQE_PREPOST_MAX_D + 1);
    MAKE_REFUSED(DEEPVQE_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512, .delay_depth = -1);
    /* The first failing check wins: sample rate, then n_fft, then the depth. */
    MAKE_REFUSED(DEEPVQE_GRID_ERR_SAMPLE_RATE, .sample_rate = 0, .n_fft = 0, .delay_depth = 0);
    MAKE_REFUSED(DEEPVQE_GRID_ERR_N_FFT, .sample_rate = 16000, .n_fft = 0, .delay_depth = 0);
    {
        DeepVqeGrid untouched;
        memset(&untouched, 0xa5, sizeof untouched);
        made = untouched;
        CHECK(deepvqe_grid_make(&made, NULL) == DEEPVQE_GRID_ERR_NULL);
        CHECK(memcmp(&made, &untouched, sizeof made) == 0);
    }

    /* _validate: the same checks, then hop and n_bins must be what _make derives. */
    spec.n_fft = 512;
    spec.delay_depth = D;
    CHECK(deepvqe_grid_make(&made, &spec) == DEEPVQE_GRID_OK);
    CHECK(deepvqe_grid_validate(&made) == DEEPVQE_GRID_OK);
    CHECK(deepvqe_grid_validate(NULL) == DEEPVQE_GRID_ERR_NULL);
    VALIDATE_AS(DEEPVQE_GRID_ERR_SAMPLE_RATE, sample_rate, 0);
    VALIDATE_AS(DEEPVQE_GRID_ERR_N_FFT, n_fft, 0);
    VALIDATE_AS(DEEPVQE_GRID_ERR_N_FFT, n_fft, 513);
    VALIDATE_AS(DEEPVQE_GRID_ERR_DELAY_DEPTH, delay_depth, DEEPVQE_PREPOST_MIN_D - 1);
    VALIDATE_AS(DEEPVQE_GRID_ERR_DELAY_DEPTH, delay_depth, DEEPVQE_PREPOST_MAX_D + 1);
    VALIDATE_AS(DEEPVQE_GRID_ERR_INCONSISTENT, hop, made.hop + 1);
    VALIDATE_AS(DEEPVQE_GRID_ERR_INCONSISTENT, hop, made.n_fft);
    VALIDATE_AS(DEEPVQE_GRID_ERR_INCONSISTENT, n_bins, made.n_bins + 1);
    VALIDATE_AS(DEEPVQE_GRID_ERR_INCONSISTENT, n_bins, made.n_bins - 1);
    return 0;
}

int main(int argc, char **argv) {
    void *fft_mem = NULL;
    FftHandle *fft;
    int status;
    int sample_rate = 0, n_fft = 0;

    if (argc < 3 || sscanf(argv[2], "grid=%d,%d", &sample_rate, &n_fft) != 2) {
        fprintf(stderr, "usage: %s <case> grid=<sample rate>,<n_fft> [descriptor fields...]\n",
                argv[0]);
        return 2;
    }
    {
        const DeepVqeGridSpec spec = { .sample_rate = sample_rate, .n_fft = n_fft,
                                       .delay_depth = D };
        CHECK(deepvqe_grid_make(&g_grid, &spec) == DEEPVQE_GRID_OK);
    }
    CHECK(g_grid.n_bins <= BINS_MAX && g_grid.hop <= HOP_MAX);
    ac_window_build(window, N_FFT);
    fill_pcm(pcm_mic, pcm_far, (int)(HOPS * HOP));

    if (strcmp(argv[1], "hash") == 0) return case_hash();
    if (strcmp(argv[1], "descriptor") == 0) {
        if (argc != 16) {
            fprintf(stderr, "descriptor needs 13 fields, got %d\n", argc - 3);
            return 2;
        }
        /* argv[2] is the grid; the 13 fields follow. */
        {
            char *shifted[15];
            int k;
            shifted[0] = argv[0];
            shifted[1] = argv[1];
            for (k = 2; k < 15; ++k) shifted[k] = argv[k + 1];
            return case_descriptor(shifted);
        }
    }

    fft = make_fft(N_FFT, &fft_mem);
    CHECK(fft != NULL);
    CHECK(fft_get_n_freqs(fft) == g_grid.n_bins);
    CHECK(build_specs() == 0);

    if (strcmp(argv[1], "equiv") == 0)          status = case_equiv(fft);
    else if (strcmp(argv[1], "compression") == 0)
                                                status = case_compression();
    else if (strcmp(argv[1], "stalepool") == 0) status = case_stalepool();
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle();
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject();
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset();
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard();
    else if (strcmp(argv[1], "inplace") == 0)   status = case_inplace();
    else if (strcmp(argv[1], "whole") == 0)     status = case_whole();
    else if (strcmp(argv[1], "generation") == 0) status = case_generation();
    else if (strcmp(argv[1], "boundary") == 0)  status = case_boundary();
    else if (strcmp(argv[1], "board") == 0)     status = case_board();
    else if (strcmp(argv[1], "registration") == 0) status = case_registration();
    else if (strcmp(argv[1], "grid") == 0)      status = case_grid();
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


def _build_driver(tmp_path_factory, audio_common_lib, name, *flags):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp(name)
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c11',
         '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off', *flags,
         '-I', _DEEPVQE_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(source), *_SOURCES,
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    """One executable for every case: the class plus the CCM kernel behind it,
    compiled at the house flags with -Werror."""
    return _build_driver(tmp_path_factory, audio_common_lib, 'deepvqe_prepost_c')


@pytest.fixture(scope='module')
def driver_scalar(tmp_path_factory, audio_common_lib):
    """The same driver with the SIMD kernels forced to their scalar twins."""
    return _build_driver(tmp_path_factory, audio_common_lib,
                         'deepvqe_prepost_c_scalar', '-DSIMD_KERNELS_FORCE_SCALAR')


def _run(driver, case, grid=_GRID_16K, *fields):
    done = subprocess.run([str(driver), case, grid, *fields],
                          capture_output=True, text=True)
    assert done.returncode == 0, (
        'case %s (%s) failed (rc=%d)\n%s' % (case, grid, done.returncode,
                                             done.stderr))
    return done.stdout


# ---- the per-case gates ------------------------------------------------


@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_class_matches_the_hand_composed_path(driver, grid):
    """The class vs a rolling center=False analysis, one hand-held in-place
    state buffer set, deepvqe_ccm_process and a hand-built AcIstft wired up
    from ONLY the CCM kernel -- byte-identical over 400 hops, with `written`
    following the synthesis half-window schedule (0, then one full hop). The
    same case pins audiolib's rolling analysis to the hand-written reference.
    The shipped grid and a second one."""
    assert 'pool: class' in _run(driver, 'equiv', grid)


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


def test_stale_pool_gate_holds_and_the_pool_follows_the_grid(driver):
    """D does not move the pool size (the attention rings are the
    application's state tensors) but it does move the stale-pool identity, and
    _init_ex must refuse a pool recorded for another D, another sample rate or
    n_fft, or a perturbed descriptor/layout/hash/size/alignment -- leaving the
    pool and the registered buffers byte for byte unchanged -- while still
    accepting the matching one."""
    stdout = _run(driver, 'stalepool')
    assert 'pool D=' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe, and on a
    pool instance it is the repeatable no-op the contract promises."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size / _init refuse NULLs, an inconsistent grid, a D outside
    [MIN_D, MAX_D], and an undersized or misaligned pool -- leaving *req
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


def test_frame_bypass_mutes_keeps_the_written_state_and_freezes_the_ring(driver):
    """FAIL CLOSED. DeepVQE-S's stream 0 is the RAW microphone, so the
    pass-through identity a post-filter takes would emit the full uncancelled
    echo; _frame_bypass emits SILENCE instead and the policy names itself
    "mute_fail_closed". The state is in place, so a skip leaves it exactly as
    the accelerator wrote it, and the CCM spectrum ring does not advance --
    proved by an instance whose accelerator wrote nothing agreeing with one
    that never saw the frame, and both differing from one that committed it."""
    _run(driver, 'skip')


def test_state_is_in_place_and_a_non_finite_write_zeroes_it(driver):
    """One buffer per state tensor: the input and `_out` pointers are equal,
    the registered buffer, and stable, the frame-t output is the frame-t+1
    input with no copy, and only the head is NaN-prefilled. MUTATION: one
    NaN/Inf injected into the newest frame of any one of the sixteen tensors
    (first and last element) through the in-place pointer makes _frame_finish
    return -1, zeroes ALL sixteen tensors, and the next healthy frame still
    commits -- while the identical uninjected frame commits, so the refusal is
    attributable to the injection. A refused commit also leaves the CCM ring
    where it was."""
    _run(driver, 'inplace')


def test_commit_reads_the_whole_of_every_registered_buffer(driver):
    """The accelerator rewrites each registered buffer whole, and a non-finite
    value anywhere in it reaches a later frame: the newest frame, an older
    frame and the frame a window drops are all refused, zero every tensor and
    advance the generation; h_gru is checked whole."""
    _run(driver, 'whole')


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
    a commit with no frame_open behind it is refused (an accelerator that
    never ran cannot pass untouched buffers off as a result), and a refused
    commit disarms until a fresh frame_open -- which re-fills every tap with
    NaN and presents the (now cold) state tensors unfilled."""
    _run(driver, 'guard')


def test_state_boundary_names_and_descriptor_validation(driver):
    """The explicit boundary: sixteen named state tensors whose output name
    is the input name plus "_out", and a descriptor validator that refuses
    each of the eight external fields when perturbed -- while still accepting
    the unperturbed descriptor, range-checking (not pinning) delay_depth and
    not reading the four model-interior fields."""
    _run(driver, 'boundary')


@pytest.mark.parametrize('build', ['driver', 'driver_scalar'])
@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_board_form_nothing_registered_runs_and_counts_restarts(
        request, build, grid):
    """The board configuration: no state registered, the accelerator owns it.
    All sixteen state views are empty, the generation starts at 0, a commit
    refused on a non-finite tap moves it by exactly one (a retry does not),
    bypass emits silence, the next hop commits again and equals a float host's
    output, and the bytes around the pool are never written. Run on the NEON
    and the scalar build, on both grids."""
    _run(request.getfixturevalue(build), 'board', grid)


def test_state_registration_is_all_or_none(driver):
    """Every partial registration of the sixteen state pairs -- one pair alone,
    all but one, a pointer without a count, a count without a pointer -- is
    refused by _get_mem_size (req untouched), _init, _init_ex and _create,
    leaving the pool and the registered buffers byte for byte unchanged; the
    full registration is accepted afterwards and runs. Control flow only: no
    SIMD twin to compare."""
    _run(driver, 'registration')


def test_grid_make_and_validate_report_each_documented_status(driver):
    """deepvqe_grid_make: NULL, sample rate <= 0, an n_fft audiolib's framing
    does not build, a delay depth outside [MIN_D, MAX_D], the first failing
    check winning, the grid untouched on every refusal, both ends of the depth
    range accepted. deepvqe_grid_validate: the same statuses plus a hop or
    n_bins that is not what _make derives."""
    _run(driver, 'grid')


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


def test_shipped_delay_depth_agrees_across_driver_config_and_grid():
    """The half-second search range is stated in the trainer config's
    max_delay_seconds, in the grid's frame conversion and in the shipped depth
    the C tests run at (SHIPPED_D here, the second chain depth there). They
    must name the same D."""
    import configparser

    from AIAEC.aiaec_common import SignalGrid

    found = re.findall(r'^#define\s+SHIPPED_D\s+(\d+)\s*$', _DRIVER,
                       re.MULTILINE)
    assert len(found) == 1, found
    config = configparser.ConfigParser()
    assert config.read(os.path.join(_DEEPVQE_PY_DIR, 'config.ini'))
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
    the same ORDER, with the same SHAPE on both sides. The shapes are the
    driver's own table (the library knows none): this is the probe that pins it
    to the graph, so a channel-schedule or frequency-ladder change in model.py
    surfaces here rather than as a silently wrong binding on the board. The
    chain driver carries a copy of the table, required to be text-identical."""
    model, built, descriptor = deepvqe_built
    _wrapper, inputs, input_names, _output_names, _split = built

    stdout = _run(driver, 'descriptor', _GRID_16K,
                  *_descriptor_argv(descriptor))
    verdict, rows = _parse_shapes(stdout)
    assert verdict == 'validate=0', stdout

    state_names = input_names[2:]
    assert len(rows) == len(state_names) == 16
    for index, (name, rank, dims) in enumerate(rows):
        assert name == state_names[index], (index, name, state_names[index])
        assert dims[:rank] == tuple(inputs[2 + index].shape)

    assert descriptor['delay_depth'] == model.max_delay_frames == 32
    assert descriptor['state_tensor_count'] == 16

    def table(text):
        match = re.search(r'/\* BEGIN state table.*?/\* END state table \*/',
                          text, re.DOTALL)
        assert match is not None
        return match.group(0)

    with open(os.path.join(_THIS_DIR, 'test_prepost_chains.c'),
              encoding='utf-8') as stream:
        assert table(stream.read()) == table(_DRIVER)


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
    """The stale-pool gate keys on (grid, layout, carve version, D).
    A pool requested under layout 3 / carve 3 (conv and score windows one
    frame shorter, rank-3 h_gru) or under another carve must hash differently
    from the shipped one; the hash is recomputed here from the published
    constants."""
    out = _run(driver, 'hash')
    c_hash = int(re.search(r'hash=(\d+)', out).group(1))
    layout = int(re.search(r'layout=(\d+)', out).group(1))
    with open(_PREPOST_HEADER, encoding='utf-8') as stream:
        carve = int(re.search(
            r'^#define\s+DEEPVQE_PREPOST_CARVE_VERSION\s+(\d+)u',
            stream.read(), re.MULTILINE).group(1))
    assert (layout, carve) == (4, 8)
    expected = _fnv1a([16000, 512, layout, carve, 8])
    ptr_bytes = int(re.search(r'ptr_bytes=(\d+)', out).group(1))
    if ptr_bytes == 4:
        for byte in b'|abi:ilp32':
            expected = ((expected ^ byte) * 16777619) & 0xffffffff
    assert c_hash == expected
    assert c_hash != _fnv1a([16000, 512, 3, 3, 8])
    assert c_hash != _fnv1a([16000, 512, layout, carve - 1, 8])
    assert c_hash != _fnv1a([16000, 512, layout, carve, 32])
    assert c_hash != _fnv1a([48000, 1024, layout, carve, 8])
