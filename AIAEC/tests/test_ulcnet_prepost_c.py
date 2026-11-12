"""Equivalence and contract gates for the Align-ULCNet pre/post class.

``ulcnet_prepost.c`` owns no signal processing: it composes the model-I/O
translation unit (``ulcnet_model_io.c``: the feature front end and the tensor
views over the application's recurrent state) behind one frequency-domain
object, so the gate is that the class reproduces the hand-composed path SAMPLE
FOR SAMPLE: every comparison is ``memcmp`` on fp32, never a tolerance. The
framing is the application's, built from audiolib directly (sqrt-Hann, center
0, hop = n_fft / 2, envelope-normalised synthesis). The state is the
application's too: the driver is a float host that owns four NaN-filled
buffers per instance and registers them.

Cases, all in ONE driver selected by argv[1], the grid by argv[2]
(``grid=<sample rate>,<n_fft>``; audiolib and the model TUs are compiled once
for the module):

  equiv     the class vs analysis + model-I/O transaction + synthesis,
            hand-composed from ONLY the model-I/O TU (not
            ulcnet_accelerator_adapter.c, which is built on the class)
  stalepool the stale-pool gate accepts the matching MemReq and refuses every
            mismatched one; the pool follows the grid, not D
  lifecycle _create/_destroy against _init on a caller pool
  reject    _get_mem_size / _init reject-first validation
  reset     _reset really clears state (re-run reproduces the first run)
  skip      _frame_bypass is the identity and leaves the state as written
  guard     the frame state machine: a spectrum on top of an unfinished frame,
            a commit with no frame_open and a retry after a refused commit are
            refused; skip needs no frame_open
  board     nothing registered (the accelerator owns the state): empty state
            views, generation 0 at init, a refused commit advances it by one
            and the next hop recovers, the pool fences stay intact
  registration  all four state pairs or none: every partial registration is
            refused and changes nothing, then the full one is accepted and runs
  grid      ulcnet_grid_make / ulcnet_grid_validate: each documented status

The accelerator stand-in is deterministic and input-dependent, so a class that
dropped a ring advance would diverge rather than agree.
"""

import os
import re
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ULCNET_DIR = os.path.join(os.path.dirname(_THIS_DIR), '..', 'lib', 'aiaec', 'ulcnet')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_SOURCES = ('ulcnet_prepost.c', 'ulcnet_model_io.c')

_GRID_16K = 'grid=16000,512'
_GRID_48K = 'grid=48000,1024'

_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "ulcnet_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

/* State table of the shipped graph (pinned to the exporter by
 * test_export_streaming_ulcnet.py): the library knows none of it, the host does. */
#define STATE_CHANNELS 32
#define STATE_SCORE_FRAMES 5
#define STATE_GRU_ROWS 4
#define STATE_GRU_HIDDEN 128
#define STATE_TA_BINS(n_bins) (((n_bins) + 9) / 10)

#define D          8
#define HOPS       400
#define SHORT_HOPS 40
#define BINS_MAX   513
#define HOP_MAX    512
#define N_FFT_MAX  1024
#define TA_MAX     52

/* The grid under test (run-time): argument grid=<sample rate>,<n_fft>. */
static UlcnetGrid grid;
#define BINS   ((size_t)grid.n_bins)
#define HOP    ((size_t)grid.hop)
#define N_FFT  grid.n_fft
#define FEATURE ((size_t)STATE_TA_BINS(grid.n_bins))

static UlcnetGrid grid_with_depth(int depth) {
    const UlcnetGridSpec spec = { .sample_rate = grid.sample_rate,
                                  .n_fft = grid.n_fft, .delay_depth = depth };
    UlcnetGrid made;
    if (ulcnet_grid_make(&made, &spec) != ULCNET_GRID_OK) made.n_fft = 0;
    return made;
}

/* The application's recurrent state: four buffers filled with NaN, so an
 * instance that did not zero what it was handed shows up. Every instance of a
 * case has its own. */
typedef struct Host {
    float key[STATE_CHANNELS * D * TA_MAX];
    float value[STATE_CHANNELS * D * TA_MAX];
    float logit[STATE_CHANNELS * STATE_SCORE_FRAMES * D];
    float gru[STATE_GRU_ROWS * STATE_GRU_HIDDEN];
} Host;

static void host_make(Host *host) {
    size_t i;
    for (i = 0; i < sizeof host->key / sizeof host->key[0]; ++i) {
        host->key[i] = NAN;
        host->value[i] = NAN;
    }
    for (i = 0; i < sizeof host->logit / sizeof host->logit[0]; ++i) host->logit[i] = NAN;
    for (i = 0; i < sizeof host->gru / sizeof host->gru[0]; ++i) host->gru[i] = NAN;
}

static void host_register(Host *host, UlcnetPrepostConfig *cfg) {
    cfg->key_history = host->key;
    cfg->key_history_elements = (size_t)STATE_CHANNELS * D * FEATURE;
    cfg->value_history = host->value;
    cfg->value_history_elements = (size_t)STATE_CHANNELS * D * FEATURE;
    cfg->logit_history = host->logit;
    cfg->logit_history_elements = (size_t)STATE_CHANNELS * STATE_SCORE_FRAMES * D;
    cfg->h_gru = host->gru;
    cfg->gru_hidden_elements = (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN;
}

static void host_buffers(Host *host, UlcnetModelIoStateBuffers *buffers) {
    buffers->key_history = host->key;
    buffers->key_history_elements = (size_t)STATE_CHANNELS * D * FEATURE;
    buffers->value_history = host->value;
    buffers->value_history_elements = (size_t)STATE_CHANNELS * D * FEATURE;
    buffers->logit_history = host->logit;
    buffers->logit_history_elements = (size_t)STATE_CHANNELS * STATE_SCORE_FRAMES * D;
    buffers->h_gru = host->gru;
    buffers->gru_hidden_elements = (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN;
}

/* The graph's window shift, performed by the stand-in accelerator: every
 * state input is read (the caller stages `now` from them) before any state
 * output is written, and each *_out is the input's own memory. K/V windows
 * are newest-first (X = cat(new, W[0:D-1])), the logit window oldest-first
 * (X = cat(W[1:5], new)). */
static void push_history(const UlcnetModelIoOutputs *out,
                         const float *key_now, const float *value_now,
                         const float *logit_now, size_t ta_bins,
                         size_t depth) {
    const size_t channels = STATE_CHANNELS;
    const size_t score_frames = STATE_SCORE_FRAMES;
    size_t channel;
    for (channel = 0; channel < channels; ++channel) {
        float *key = out->key_history_out + channel * depth * ta_bins;
        float *value = out->value_history_out + channel * depth * ta_bins;
        float *logit = out->logit_history_out +
            channel * score_frames * depth;
        memmove(key + ta_bins, key, (depth - 1u) * ta_bins * sizeof(float));
        memcpy(key, key_now + channel * ta_bins, ta_bins * sizeof(float));
        memmove(value + ta_bins, value,
                (depth - 1u) * ta_bins * sizeof(float));
        memcpy(value, value_now + channel * ta_bins,
               ta_bins * sizeof(float));
        memmove(logit, logit + depth,
                (score_frames - 1u) * depth * sizeof(float));
        memcpy(logit + (score_frames - 1u) * depth,
               logit_now + channel * depth, depth * sizeof(float));
    }
}

/* Deterministic stand-in for the accelerator. Satisfies the FULL-WRITE
 * contract (every element of every output tensor) and is made
 * input-dependent so the rings and the in-place GRU hiddens actually
 * influence later frames. It reads every input into `acc` before it writes
 * any output, which is what an in-place runtime must do. The state views are
 * the registered buffers: an unregistered one fails the run. */
static int fake_run(void *user, const UlcnetModelIoInputs *in,
                    UlcnetModelIoOutputs *out) {
    static float key_now[STATE_CHANNELS * TA_MAX];
    static float value_now[STATE_CHANNELS * TA_MAX];
    static float logit_now[STATE_CHANNELS * D];
    size_t i;
    double acc = 0.0;
    (void)user;
    if (!in->key_history || !in->value_history || !in->logit_history || !in->h_gru ||
        in->key_history_elements != (size_t)STATE_CHANNELS * D * FEATURE ||
        in->gru_hidden_elements != (size_t)STATE_GRU_ROWS * STATE_GRU_HIDDEN)
        return -1;
    for (i = 0; i < in->spectrum_bins_elements; ++i)
        acc += in->error_mag[i] + 0.5 * in->far_mag[i];
    for (i = 0; i < in->key_history_elements; ++i)
        acc += 1e-3 * in->key_history[i];
    for (i = 0; i < in->value_history_elements; ++i)
        acc += 1e-3 * in->value_history[i];
    for (i = 0; i < in->logit_history_elements; ++i)
        acc += 1e-3 * in->logit_history[i];
    for (i = 0; i < in->gru_hidden_elements; ++i)
        acc += 1e-3 * in->h_gru[i];
    /* The graph's output is the planar complex mask: real plane, then
     * imaginary plane. */
    for (i = 0; i < BINS; ++i) {
        out->output[i] = (float)(0.5 + 0.25 * sin(acc + (double)i));
        out->output[BINS + i] = (float)(0.125 * cos(acc + (double)i));
    }
    for (i = 0; i < (size_t)STATE_CHANNELS * FEATURE; ++i) {
        key_now[i] = (float)(0.01 * sin(acc + 1.0 + (double)i));
        value_now[i] = (float)(0.01 * cos(acc + 2.0 + (double)i));
    }
    for (i = 0; i < (size_t)STATE_CHANNELS * D; ++i)
        logit_now[i] = (float)(0.01 * sin(acc + 3.0 + (double)i));
    for (i = 0; i < out->gru_hidden_elements; ++i) {
        out->h_gru_out[i] = (float)(0.5 * ((i & 1u) ?
            cos(acc + 5.0 + (double)i) : sin(acc + 4.0 + (double)i)));
    }
    push_history(out, key_now, value_now, logit_now, FEATURE, D);
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

static int identical(const float *a, const float *b, size_t count) {
    return memcmp(a, b, count * sizeof(float)) == 0;
}

static float pcm_error[HOPS * HOP_MAX];
static float pcm_far[HOPS * HOP_MAX];
static float out_a[HOPS * HOP_MAX];
static float out_b[HOPS * HOP_MAX];
static float window[N_FFT_MAX];

/* One analysed hop per stream, and the spectra the class hands back. */
typedef struct HopSpectra {
    float er[BINS_MAX], ei[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX];
} HopSpectra;
typedef struct HopOut {
    float re[BINS_MAX], im[BINS_MAX];
} HopOut;
static HopSpectra specs[HOPS];
static HopOut outs_a[HOPS], outs_b[HOPS];

/* The application's framing, built from audiolib directly: the documented
 * contract (ulcnet_process.h) is sqrt-Hann, center 0, hop = n_fft / 2,
 * unnormalised spectra, AC_OLA_ENV_NORM synthesis. */
typedef struct Framing {
    AcStft *error, *far;
    AcIstft *synthesis;
} Framing;

static void framing_config(AcStftConfig *sc, int center) {
    ac_stft_config_default(sc, N_FFT);
    sc->center = center;
    sc->ola = AC_OLA_ENV_NORM;
}

static AcStft *make_analysis(int center) {
    AcStftConfig sc;
    size_t bytes;
    framing_config(&sc, center);
    bytes = ac_stft_get_mem_size(&sc);
    return bytes ? ac_stft_init(alloc_aligned(16u, bytes), bytes, &sc) : NULL;
}

static AcIstft *make_synthesis(void) {
    AcStftConfig sc;
    size_t bytes;
    framing_config(&sc, 0);
    bytes = ac_istft_get_mem_size(&sc);
    return bytes ? ac_istft_init(alloc_aligned(16u, bytes), bytes, &sc) : NULL;
}

static int framing_make(Framing *f) {
    f->error = make_analysis(0);
    f->far = make_analysis(0);
    f->synthesis = make_synthesis();
    return f->error != NULL && f->far != NULL && f->synthesis != NULL;
}

/* center=False rolling analysis: one frame per hop, over the last N_FFT
 * samples. Deliberately expression-for-expression the same as
 * the center=False analysis of audiolib's AcStft, so this reference is
 * independent of it yet bit-comparable with it. */
static void ref_roll(FftHandle *fft, float *history, const float *hop_in,
                     float *out_re, float *out_im) {
    static float seg[N_FFT_MAX];
    static Complex spec[BINS_MAX];
    size_t k;
    memmove(history, history + HOP, (size_t)(N_FFT - grid.hop) * sizeof(float));
    memcpy(history + N_FFT - grid.hop, hop_in, HOP * sizeof(float));
    for (k = 0; k < (size_t)N_FFT; ++k) seg[k] = history[k] * window[k];
    fft_forward_scratch(fft, seg, spec);
    for (k = 0; k < BINS; ++k) {
        out_re[k] = spec[k].r;
        out_im[k] = spec[k].i;
    }
}

/* Analyse every hop of the test signal once, with audiolib's rolling analysis. */
static int build_specs(void) {
    Framing f;
    int hop;
    if (!framing_make(&f)) return -1;
    for (hop = 0; hop < HOPS; ++hop) {
        if (ac_stft_push_planar(f.error, pcm_error + (size_t)hop * HOP,
                                specs[hop].er, specs[hop].ei) != 1) return -1;
        if (ac_stft_push_planar(f.far, pcm_far + (size_t)hop * HOP,
                                specs[hop].fr, specs[hop].fi) != 1) return -1;
    }
    return 0;
}

/* One accelerator transaction through the class for the analysed hops
 * [first, first + count); the enhanced spectra land in `out`. */
static int drive_spec(UlcnetPrepost *p, int first, int count, HopOut *out) {
    int hop;
    for (hop = first; hop < first + count; ++hop) {
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        /* THE contract this class exists to guarantee: one spectrum in, exactly
         * one accelerator invocation, from the very first hop. */
        if (ulcnet_prepost_pre_process_freq(p, specs[hop].er, specs[hop].ei,
                                            specs[hop].fr, specs[hop].fi) != 1)
            return -1;
        if (ulcnet_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
        if (fake_run(NULL, &inputs, &outputs) != 0) return -1;
        if (ulcnet_prepost_frame_finish(p) != 0) return -1;
        if (ulcnet_prepost_post_process_freq(p, out[hop - first].re,
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

/* Class for the grid under test with a registered Host. */
static UlcnetPrepost *class_make(Host *host, void **pool_out,
                                 UlcnetPrepostMemReq *req_out) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    void *pool;
    UlcnetPrepost *p;
    if (ulcnet_prepost_config_defaults(&cfg, &grid) != 0) return NULL;
    host_make(host);
    host_register(host, &cfg);
    if (ulcnet_prepost_get_mem_size(&cfg, &req) != 0) return NULL;
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    if (!pool) return NULL;
    p = ulcnet_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    if (pool_out) *pool_out = pool;
    if (req_out) *req_out = req;
    return p;
}

static int bytes_are(const void *base, size_t count, unsigned char value) {
    const unsigned char *bytes = (const unsigned char *)base;
    size_t i;
    for (i = 0; i < count; ++i) if (bytes[i] != value) return 0;
    return 1;
}

/* ---- equiv: the class vs the hand-composed reference path ------------ */

static int case_equiv(FftHandle *fft) {
    UlcnetModelIoDescriptor descriptor;
    UlcnetModelIoMemReq io_req;
    UlcnetModelIoStateBuffers buffers;
    UlcnetModelIoState *state;
    void *state_mem, *pool;
    static Host host_a, host_b;
    static float hist_error[N_FFT_MAX], hist_far[N_FFT_MAX];
    Framing fa, fb;
    AcStft *centered;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *p;
    int hop;

    /* path A -- the analysis, the model-I/O transaction and the synthesis,
     * composed by hand from the one TU the class merely orchestrates. */
    CHECK(ulcnet_model_io_descriptor_default(&grid, &descriptor) == 0);
    CHECK(ulcnet_model_io_get_mem_requirements(&descriptor, &io_req) == 0);
    state_mem = alloc_aligned(io_req.alignment, io_req.bytes);
    CHECK(state_mem != NULL);
    host_make(&host_a);
    host_buffers(&host_a, &buffers);
    state = ulcnet_model_io_init(state_mem, io_req.bytes, &descriptor, &buffers);
    CHECK(state != NULL);
    memset(hist_error, 0, sizeof hist_error);
    memset(hist_far, 0, sizeof hist_far);
    CHECK(framing_make(&fa) && framing_make(&fb));
    centered = make_analysis(1);
    CHECK(centered != NULL);
    for (hop = 0; hop < HOPS; ++hop) {
        static float error_re[BINS_MAX], error_im[BINS_MAX];
        static float far_re[BINS_MAX], far_im[BINS_MAX];
        static float enhanced_re[BINS_MAX], enhanced_im[BINS_MAX];
        static float c_re[2 * BINS_MAX], c_im[2 * BINS_MAX];
        float *slot = out_a + (size_t)hop * HOP;
        int centered_frames;
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        int wrote, bin;
        /* One frame per hop, both streams from the SAME hop index. */
        ref_roll(fft, hist_error, pcm_error + (size_t)hop * HOP,
                 error_re, error_im);
        ref_roll(fft, hist_far, pcm_far + (size_t)hop * HOP, far_re, far_im);
        /* audiolib's rolling analysis is that reference, bit for bit. */
        CHECK(identical(specs[hop].er, error_re, BINS));
        CHECK(identical(specs[hop].ei, error_im, BINS));
        CHECK(identical(specs[hop].fr, far_re, BINS));
        CHECK(identical(specs[hop].fi, far_im, BINS));
        /* The FRAMING claim the headers make ("centered frame k is
         * bit-identical to non-centered frame k-1"), pinned here rather
         * than asserted: the centered analysis's LAST frame on every hop is
         * memcmp-equal to the rolling frame of the same hop, on the real
         * centered AcStft and its documented 0 / 2 / 1 schedule. */
        centered_frames = ac_stft_push_planar(
            centered, pcm_error + (size_t)hop * HOP, c_re, c_im);
        CHECK(centered_frames == (hop == 0 ? 0 : hop == 1 ? 2 : 1));
        if (centered_frames > 0) {
            /* frame f of a push lands at f * n_bins */
            CHECK(identical(c_re + (size_t)(centered_frames - 1) * BINS, error_re, BINS));
            CHECK(identical(c_im + (size_t)(centered_frames - 1) * BINS, error_im, BINS));
        }
        for (bin = 0; bin < grid.n_bins; ++bin) {
            enhanced_re[bin] = NAN;
            enhanced_im[bin] = NAN;
        }
        CHECK(ulcnet_model_io_prepare(state, error_re, error_im,
                                      far_re, far_im,
                                      &inputs, &outputs) == 0);
        CHECK(fake_run(NULL, &inputs, &outputs) == 0);
        if (ulcnet_model_io_enhance_spectrum(state, enhanced_re, enhanced_im) != 0) {
            memcpy(enhanced_re, error_re, sizeof enhanced_re);
            memcpy(enhanced_im, error_im, sizeof enhanced_im);
        }
        /* One frame per hop: nothing on hop 0 (its block lies inside the
         * trimmed half window), a full hop from hop 1 on. */
        wrote = ac_istft_push_planar(fa.synthesis, enhanced_re,
                                     enhanced_im, slot);
        CHECK(wrote == (hop == 0 ? 0 : grid.hop));
        if (wrote == 0) memset(slot, 0, HOP * sizeof(float));
    }

    /* path B -- the class, fed the same analysed spectra */
    p = class_make(&host_b, &pool, &req);
    CHECK(p != NULL);
    CHECK(ulcnet_prepost_hop_size(p) == grid.hop);
    CHECK(ulcnet_prepost_num_bins(p) == grid.n_bins);
    CHECK(ulcnet_prepost_descriptor(p) != NULL);
    CHECK(ulcnet_prepost_descriptor(p)->delay_depth == D);
    for (hop = 0; hop < HOPS; ++hop) {
        static float enhanced_re[BINS_MAX], enhanced_im[BINS_MAX];
        float *slot = out_b + (size_t)hop * HOP;
        int wrote;
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        CHECK(ulcnet_prepost_pre_process_freq(p, specs[hop].er, specs[hop].ei,
                                              specs[hop].fr, specs[hop].fi) == 1);
        CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(fake_run(NULL, &inputs, &outputs) == 0);
        CHECK(ulcnet_prepost_frame_finish(p) == 0);
        CHECK(ulcnet_prepost_post_process_freq(p, enhanced_re, enhanced_im) == 0);
        wrote = ac_istft_push_planar(fb.synthesis, enhanced_re, enhanced_im, slot);
        CHECK(wrote == (hop == 0 ? 0 : grid.hop));
        if (wrote == 0) memset(slot, 0, HOP * sizeof(float));
    }

    printf("pool: model I/O %zu B   class %llu B (D=%d, %d bins)\n",
           io_req.bytes, (unsigned long long)req.bytes, D, grid.n_bins);
    /* The reference is not silence -- otherwise byte-identity is vacuous. */
    CHECK(!identical(out_a, out_b + 1, HOPS * HOP - 1));
    CHECK(identical(out_a, out_b, HOPS * HOP));

    ulcnet_prepost_destroy(p);
    free(pool);
    free(state_mem);
    return 0;
}

/* ---- stale pool: the stale-pool gate --------------------------------- */

/* A refused _init_ex returns NULL and leaves the pool and the registered buffers byte for byte unchanged. */
static int stale_refused(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg,
                         const UlcnetPrepostMemReq *expected, const Host *live,
                         const Host *snapshot) {
    return ulcnet_prepost_init_ex(pool, bytes, cfg, expected) == NULL &&
           bytes_are(pool, bytes, 0xa5u) &&
           memcmp(live, snapshot, sizeof *live) == 0;
}

/* The pool request of a registered config on another grid. */
static int req_for_grid(const UlcnetGridSpec *spec, Host *host,
                        UlcnetPrepostMemReq *req) {
    UlcnetPrepostConfig other_cfg;
    UlcnetGrid other;
    if (ulcnet_grid_make(&other, spec) != ULCNET_GRID_OK ||
        ulcnet_prepost_config_defaults(&other_cfg, &other) != 0) return -1;
    host_register(host, &other_cfg);
    return ulcnet_prepost_get_mem_size(&other_cfg, req);
}

static int case_stalepool(void) {
    UlcnetPrepostConfig cfg, cfg_deep;
    UlcnetPrepostMemReq req, req_deep, req_rate, req_fft;
    UlcnetGrid grid_deep = grid_with_depth(2 * D);
    const UlcnetGridSpec rate_spec = { .sample_rate = 2 * grid.sample_rate,
                                       .n_fft = grid.n_fft, .delay_depth = D };
    const UlcnetGridSpec fft_spec = { .sample_rate = grid.sample_rate,
                                      .n_fft = 2 * grid.n_fft, .delay_depth = D };
    static Host host, snapshot;
    void *pool;

    CHECK(grid_deep.n_fft != 0);
    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    CHECK(ulcnet_prepost_config_defaults(&cfg_deep, &grid_deep) == 0);
    host_make(&host);
    host_register(&host, &cfg);
    host_register(&host, &cfg_deep);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg_deep, &req_deep) == 0);
    CHECK(req_for_grid(&rate_spec, &host, &req_rate) == 0);
    CHECK(req_for_grid(&fft_spec, &host, &req_fft) == 0);

    /* The pool follows the grid, not D; D moves the stale-pool identity. */
    printf("pool = %llu B\n", (unsigned long long)req.bytes);
    CHECK(req_deep.bytes == req.bytes);
    CHECK(req_deep.build_flags_hash != req.build_flags_hash);
    /* Another sample rate keeps the size and moves the hash; another n_fft moves both. */
    CHECK(req_rate.bytes == req.bytes && req_rate.build_flags_hash != req.build_flags_hash);
    CHECK(req_fft.bytes != req.bytes && req_fft.build_flags_hash != req.build_flags_hash);
    CHECK(req.descriptor_version == ULCNET_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req.layout_version == ULCNET_MODEL_IO_LAYOUT_VERSION);
    CHECK(req.reserved0 == 0u && req.reserved1 == 0u);

    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(ulcnet_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req) != NULL);
    CHECK(!bytes_are(pool, (size_t)req.bytes, 0xa5u) && host.gru[0] == 0.0f);
    /* Poison the pool and the registered buffers: a refusal must leave both as they are. */
    memset(pool, 0xa5, (size_t)req.bytes);
    host_make(&host);
    snapshot = host;
    /* ...and refuse a pool sized for another D or another grid. */
    CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &req_deep, &host, &snapshot));
    CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &req_rate, &host, &snapshot));
    CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &req_fft, &host, &snapshot));
    {
        UlcnetPrepostMemReq stale = req;
        ++stale.descriptor_version;
        CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &stale, &host, &snapshot));
        stale = req;
        ++stale.layout_version;
        CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &stale, &host, &snapshot));
        stale = req;
        ++stale.build_flags_hash;
        CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &stale, &host, &snapshot));
        stale = req;
        ++stale.bytes;
        CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &stale, &host, &snapshot));
        stale = req;
        stale.alignment *= 2u;
        CHECK(stale_refused(pool, (size_t)req.bytes, &cfg, &stale, &host, &snapshot));
        /* Unperturbed again, so the refusals above are attributable to the
         * perturbation and not to the pool going stale on its own. */
        CHECK(ulcnet_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req) != NULL);
        CHECK(!bytes_are(pool, (size_t)req.bytes, 0xa5u) && host.gru[0] == 0.0f);
    }
    free(pool);
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(void) {
    UlcnetPrepostConfig cfg, bad;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *heap, *stack;
    static Host host_heap, host_stack;
    void *pool;

    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    host_make(&host_heap);
    host_register(&host_heap, &cfg);
    heap = ulcnet_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(ulcnet_prepost_hop_size(heap) == grid.hop);
    CHECK(ulcnet_prepost_num_bins(heap) == grid.n_bins);
    CHECK(ulcnet_prepost_descriptor(heap)->delay_depth == D);
    CHECK(drive_spec(heap, 0, SHORT_HOPS, outs_a) == 0);

    /* A created instance and a pool instance are the same object. */
    host_make(&host_stack);
    host_register(&host_stack, &cfg);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    stack = ulcnet_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(stack != NULL);
    CHECK(drive_spec(stack, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));
    /* ...and the output is not silence or constant. */
    CHECK(!identical(outs_a[SHORT_HOPS - 1].re, outs_a[SHORT_HOPS - 2].re, BINS));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still
     * runs afterwards. (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    ulcnet_prepost_destroy(stack);
    ulcnet_prepost_destroy(stack);
    ulcnet_prepost_reset(stack);
    CHECK(drive_spec(stack, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));

    ulcnet_prepost_destroy(heap);
    ulcnet_prepost_destroy(NULL);
    free(pool);

    /* _create refuses what _get_mem_size refuses, and allocates nothing. */
    CHECK(ulcnet_prepost_create(NULL) == NULL);
    CHECK(ulcnet_prepost_config_defaults(&bad, &grid) == 0);
    bad.grid.delay_depth = 0;
    CHECK(ulcnet_prepost_create(&bad) == NULL);

    /* Accessors on NULL. */
    CHECK(ulcnet_prepost_hop_size(NULL) == -1);
    CHECK(ulcnet_prepost_num_bins(NULL) == -1);
    CHECK(ulcnet_prepost_descriptor(NULL) == NULL);
    ulcnet_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

static int case_reject(void) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req, guard;
    UlcnetGrid bad_grid, edge;
    static Host host;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    host_make(&host);
    host_register(&host, &cfg);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(ulcnet_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, NULL) == -1);
    CHECK(ulcnet_prepost_config_defaults(NULL, &grid) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, NULL) == -1);
    host_register(&host, &cfg);

    /* An invalid grid. *req is left untouched on a refusal. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    bad_grid = grid;
    bad_grid.n_bins += 1;
    CHECK(ulcnet_prepost_config_defaults(&cfg, &bad_grid) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    host_register(&host, &cfg);
    cfg.grid = bad_grid;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.grid = grid;

    /* D outside [MIN_D, MAX_D]. */
    bad_grid = grid;
    bad_grid.delay_depth = ULCNET_MODEL_IO_MIN_D - 1;
    cfg.grid = bad_grid;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    bad_grid.delay_depth = ULCNET_MODEL_IO_MAX_D + 1;
    cfg.grid = bad_grid;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    bad_grid.delay_depth = 0;
    cfg.grid = bad_grid;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(ulcnet_prepost_config_defaults(&cfg, &bad_grid) == -1);
    /* Both ends of the range are accepted. */
    edge = grid_with_depth(ULCNET_MODEL_IO_MIN_D);
    CHECK(edge.n_fft != 0 && ulcnet_prepost_config_defaults(&cfg, &edge) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    edge = grid_with_depth(ULCNET_MODEL_IO_MAX_D);
    CHECK(edge.n_fft != 0 && ulcnet_prepost_config_defaults(&cfg, &edge) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);

    /* _init: NULL pool, undersized pool, misaligned pool. */
    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    host_register(&host, &cfg);
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
        static float got_re[BINS_MAX], got_im[BINS_MAX];
        CHECK(p != NULL);
        CHECK(ulcnet_prepost_pre_process_freq(NULL, specs[0].er, specs[0].ei,
                                              specs[0].fr, specs[0].fi) == -1);
        CHECK(ulcnet_prepost_pre_process_freq(p, NULL, specs[0].ei,
                                              specs[0].fr, specs[0].fi) == -1);
        CHECK(ulcnet_prepost_pre_process_freq(p, specs[0].er, specs[0].ei,
                                              specs[0].fr, NULL) == -1);
        CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == -1);
        CHECK(ulcnet_prepost_frame_finish(p) == -1);
        CHECK(ulcnet_prepost_frame_bypass(p) == -1);
        CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
        CHECK(ulcnet_prepost_pre_process_freq(p, specs[0].er, specs[0].ei,
                                              specs[0].fr, specs[0].fi) == 1);
        CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == -1);
        /* A partial accelerator write is refused and moves nothing. */
        outputs.output[0] = 0.0f;
        CHECK(ulcnet_prepost_frame_finish(p) == -1);
        CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    }

    free(pool);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(void) {
    UlcnetPrepost *p;
    static Host host;
    void *pool;

    p = class_make(&host, &pool, NULL);
    CHECK(p != NULL);

    CHECK(drive_spec(p, 0, SHORT_HOPS, outs_a) == 0);
    /* Without the reset the second run continues from a warm state, so this
     * comparison has teeth: assert that it does NOT match first. */
    CHECK(drive_spec(p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(!outs_identical(outs_a, outs_b, SHORT_HOPS));

    CHECK(ulcnet_prepost_state_generation(p) == 0u);
    CHECK(ulcnet_prepost_state_generation(NULL) == 0u);
    ulcnet_prepost_reset(p);
    CHECK(ulcnet_prepost_state_generation(p) == 1u);
    /* One inference per spectrum, from the very first hop. */
    CHECK(ulcnet_prepost_pre_process_freq(p, specs[0].er, specs[0].ei,
                                          specs[0].fr, specs[0].fi) == 1);
    ulcnet_prepost_reset(p);
    CHECK(ulcnet_prepost_state_generation(p) == 2u);
    CHECK(drive_spec(p, 0, SHORT_HOPS, outs_b) == 0);
    CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));

    /* Reset mid-hop drops the open frame instead of stranding it. */
    {
        UlcnetModelIoInputs inputs;
        UlcnetModelIoOutputs outputs;
        CHECK(ulcnet_prepost_pre_process_freq(p, specs[0].er, specs[0].ei,
                                              specs[0].fr, specs[0].fi) == 1);
        CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
        ulcnet_prepost_reset(p);
        CHECK(ulcnet_prepost_frame_finish(p) == -1);
        CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == -1);
        CHECK(drive_spec(p, 0, SHORT_HOPS, outs_b) == 0);
        CHECK(outs_identical(outs_a, outs_b, SHORT_HOPS));
    }

    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- skip: the identity, and no ring advance ------------------------- */

static int case_skip(void) {
    UlcnetPrepost *p;
    static Host host;
    void *pool;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    static float error_re[BINS_MAX], error_im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    static float got_re[BINS_MAX], got_im[BINS_MAX];
    float *key = NULL, *value = NULL, *logit = NULL;
    float *gru = NULL;
    size_t key_n, value_n, logit_n, gru_n;
    size_t bin;

    p = class_make(&host, &pool, NULL);
    CHECK(p != NULL);

    for (bin = 0; bin < BINS; ++bin) {
        error_re[bin] = 0.5f - (float)bin / (float)BINS;
        error_im[bin] = (float)bin / (float)BINS - 0.25f;
        far_re[bin] = 0.125f + 0.5f * (float)bin / (float)BINS;
        far_im[bin] = 0.75f - (float)bin / (float)BINS;
    }

    /* Frame 1: a real commit, so the rings are NON-ZERO for the check. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_finish(p) == 0);

    /* Frame 2: take the identity even though the accelerator produced a
     * COMPLETE, perfectly committable result.
     *
     * The full write is the point. Skipping after a partial write would
     * prove nothing -- commit() would refuse that frame on its own -- so a
     * skip that quietly committed would still look correct. Here the only
     * thing standing between the frame and the pipeline is frame_bypass's
     * contract. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    key_n = inputs.key_history_elements;
    value_n = inputs.value_history_elements;
    logit_n = inputs.logit_history_elements;
    gru_n = inputs.gru_hidden_elements;
    /* The views are the registered buffers: a missing registration cannot pass for
     * an empty window. */
    CHECK(key_n == (size_t)STATE_CHANNELS * D * FEATURE && gru_n != 0u);
    CHECK(inputs.key_history == host.key && inputs.h_gru == host.gru);
    key = (float *)malloc(key_n * sizeof(float));
    value = (float *)malloc(value_n * sizeof(float));
    logit = (float *)malloc(logit_n * sizeof(float));
    gru = (float *)malloc(gru_n * sizeof(float));
    CHECK(key && value && logit && gru);
    memcpy(key, inputs.key_history, key_n * sizeof(float));
    /* The pre-run snapshot must not be all zeros, or "moved" proves
     * nothing. */
    {
        size_t i;
        int nonzero = 0;
        for (i = 0; i < key_n; ++i) if (key[i] != 0.0f) nonzero = 1;
        CHECK(nonzero);
    }

    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    /* Every state tensor is in-place: the run has already written it where
     * the next frame reads it, and skip leaves it there. Snapshot it AFTER
     * the run. */
    CHECK(outputs.key_history_out == inputs.key_history);
    CHECK(outputs.value_history_out == inputs.value_history);
    CHECK(outputs.logit_history_out == inputs.logit_history);
    CHECK(outputs.h_gru_out == inputs.h_gru);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) != 0);
    memcpy(key, inputs.key_history, key_n * sizeof(float));
    memcpy(value, inputs.value_history, value_n * sizeof(float));
    memcpy(logit, inputs.logit_history, logit_n * sizeof(float));
    memcpy(gru, inputs.h_gru, gru_n * sizeof(float));
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* Identity: the error spectrum passes through UNCHANGED, bit for bit --
     * the accelerator's enhanced output is discarded, not applied. */
    CHECK(identical(error_re, got_re, BINS));
    CHECK(identical(error_im, got_im, BINS));

    /* Frame 3: the state must be exactly what frame 2's run left. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(inputs.key_history_elements == key_n);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.value_history, value, value_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.logit_history, logit, logit_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.h_gru, gru, gru_n * sizeof(float)) == 0);

    /* The other reason to skip: the accelerator wrote NOTHING at all. Same
     * identity, same state. */
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    CHECK(identical(error_re, got_re, BINS));
    CHECK(identical(error_im, got_im, BINS));

    /* Frame 4: still as written -- and a run plus commit here DOES move
     * the state, so the comparisons above are not measuring a constant. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) == 0);
    CHECK(memcmp(inputs.h_gru, gru, gru_n * sizeof(float)) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_finish(p) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(memcmp(inputs.key_history, key, key_n * sizeof(float)) != 0);

    free(key);
    free(value);
    free(logit);
    free(gru);
    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- guard: the frame state machine refuses what it must -------------- */

static int case_guard(void) {
    UlcnetPrepost *p;
    static Host host;
    void *pool;
    UlcnetModelIoInputs inputs;
    UlcnetModelIoOutputs outputs;
    static float error_re[BINS_MAX], error_im[BINS_MAX];
    static float far_re[BINS_MAX], far_im[BINS_MAX];
    static float got_re[BINS_MAX], got_im[BINS_MAX];
    size_t i;
    size_t bin;
    uint32_t generation;

    p = class_make(&host, &pool, NULL);
    CHECK(p != NULL);
    for (bin = 0; bin < BINS; ++bin) {
        error_re[bin] = 0.5f - (float)bin / (float)BINS;
        error_im[bin] = (float)bin / (float)BINS - 0.25f;
        far_re[bin] = 0.125f + 0.5f * (float)bin / (float)BINS;
        far_im[bin] = 0.75f - (float)bin / (float)BINS;
    }
    /* (1) A spectrum on top of an unfinished one is refused, and the open
     * frame is still the caller's to finish. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == -1);
    /* (2) A frame is open but the accelerator was never handed it: commit
     * is refused rather than passing untouched buffers off as a result. */
    CHECK(ulcnet_prepost_frame_finish(p) == -1);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == -1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == -1);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_finish(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* ...and once finished -- by commit or by skip -- the next spectrum is
     * accepted again. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);

    /* (3) A commit refused on a non-finite output disarms the transaction:
     * a retry without a fresh frame_open is refused too, the frame is
     * still open, and the fresh frame_open re-fills the mask with
     * NaN before the accelerator is asked again. The in-place state is not
     * refilled: the refusal has already restarted it from zero. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    outputs.h_gru_out[outputs.gru_hidden_elements - 1u] = NAN;
    generation = ulcnet_prepost_state_generation(p);
    CHECK(ulcnet_prepost_frame_finish(p) == -1);
    CHECK(ulcnet_prepost_state_generation(p) == generation + 1u);
    CHECK(ulcnet_prepost_frame_finish(p) == -1);
    CHECK(ulcnet_prepost_state_generation(p) == generation + 1u);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == -1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    for (i = 0; i < outputs.spectrum_ri_elements; ++i)
        CHECK(isnan(outputs.output[i]));
    CHECK(outputs.h_gru_out == inputs.h_gru);
    CHECK(outputs.key_history_out == inputs.key_history);
    CHECK(outputs.value_history_out == inputs.value_history);
    CHECK(outputs.logit_history_out == inputs.logit_history);
    /* The refusal zeroed the registered buffers themselves. */
    for (i = 0; i < inputs.key_history_elements; ++i)
        CHECK(host.key[i] == 0.0f);
    for (i = 0; i < inputs.value_history_elements; ++i)
        CHECK(host.value[i] == 0.0f);
    for (i = 0; i < inputs.logit_history_elements; ++i)
        CHECK(host.logit[i] == 0.0f);
    for (i = 0; i < outputs.gru_hidden_elements; ++i)
        CHECK(host.gru[i] == 0.0f);
    CHECK(inputs.key_history_elements != 0u);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_finish(p) == 0);
    CHECK(ulcnet_prepost_post_process_freq(p, got_re, got_im) == 0);
    /* (4) Skipping never needed the accelerator, so it needs no
     * frame_open either. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    /* (5) ...and a skip taken AFTER frame_open armed the accelerator must
     * not leave that arming behind: the next frame's commit, with no
     * frame_open of its own, is still refused. This is the state in which
     * a latch borrowed from model_io alone lets stale outputs through, so
     * (2) above cannot stand in for it. */
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(fake_run(NULL, &inputs, &outputs) == 0);
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);
    CHECK(ulcnet_prepost_pre_process_freq(p, error_re, error_im,
                                          far_re, far_im) == 1);
    CHECK(ulcnet_prepost_frame_finish(p) == -1);
    CHECK(ulcnet_prepost_frame_bypass(p) == 0);

    ulcnet_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- layout: the config's model_layout_version ------------------------ */

/* 0 resolves to this build's layout and an explicit match is the same pool;
 * any other value is refused before a pool is sized. */
static int case_layout(void) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq base, explicit_match, other;
    UlcnetPrepost *p;
    static Host host;
    void *pool;
    uint32_t refused;

    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    host_make(&host);
    host_register(&host, &cfg);
    CHECK(cfg.model_layout_version == 0u);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &base) == 0);
    CHECK(base.layout_version == ULCNET_MODEL_IO_LAYOUT_VERSION);

    cfg.model_layout_version = ULCNET_MODEL_IO_LAYOUT_VERSION;
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &explicit_match) == 0);
    CHECK(memcmp(&base, &explicit_match, sizeof(base)) == 0);
    pool = alloc_aligned(explicit_match.alignment, (size_t)explicit_match.bytes);
    CHECK(pool != NULL);
    p = ulcnet_prepost_init_ex(pool, (size_t)explicit_match.bytes, &cfg,
                               &explicit_match);
    CHECK(p != NULL);
    CHECK(ulcnet_prepost_descriptor(p)->layout_version ==
          ULCNET_MODEL_IO_LAYOUT_VERSION);
    ulcnet_prepost_destroy(p);

    /* Every retired layout (3-17) and a number that was never assigned are
     * refused. */
    for (refused = 3u; refused <= 19u; ++refused) {
        if (refused == ULCNET_MODEL_IO_LAYOUT_VERSION) continue;
        cfg.model_layout_version = refused;
        CHECK(ulcnet_prepost_get_mem_size(&cfg, &other) != 0);
        CHECK(ulcnet_prepost_init_ex(pool, (size_t)explicit_match.bytes, &cfg,
                                     &explicit_match) == NULL);
    }
    free(pool);
    return 0;
}

/* ---- hash: the stale-pool identity of this build ---------------------- */

static int case_hash(void) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
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
static int views_empty(const UlcnetModelIoInputs *in, const UlcnetModelIoOutputs *out) {
    return !in->key_history && !in->value_history && !in->logit_history && !in->h_gru &&
           !in->key_history_elements && !in->value_history_elements &&
           !in->logit_history_elements && !in->gru_hidden_elements &&
           !out->key_history_out && !out->value_history_out && !out->logit_history_out &&
           !out->h_gru_out && !out->key_history_elements && !out->value_history_elements &&
           !out->logit_history_elements && !out->gru_hidden_elements;
}

/* One hop through a head-only stand-in (the mask depends on the hop only): returns frame_finish's
 * status, or 2 when the boundary itself is wrong. `poison` leaves one mask element NaN. */
static int head_hop(UlcnetPrepost *p, int hop, int empty, int poison) {
    UlcnetModelIoInputs in;
    UlcnetModelIoOutputs out;
    size_t i;
    if (ulcnet_prepost_pre_process_freq(p, specs[hop].er, specs[hop].ei,
                                        specs[hop].fr, specs[hop].fi) != 1) return 2;
    if (ulcnet_prepost_frame_open(p, &in, &out) != 0) return 2;
    if (out.spectrum_ri_elements != 2u * BINS || !isnan(out.output[0]) ||
        !isnan(out.output[2u * BINS - 1u])) return 2;
    if (empty && !views_empty(&in, &out)) return 2;
    for (i = 0; i < BINS; ++i) {
        out.output[i] = 0.5f + 0.25f * sinf((float)(i + (size_t)hop));
        out.output[BINS + i] = 0.125f * cosf((float)(i + (size_t)hop));
    }
    if (poison) out.output[BINS / 2u] = NAN;
    return ulcnet_prepost_frame_finish(p);
}

static int fences_intact(const unsigned char *raw, size_t bytes) {
    return bytes_are(raw, FENCE, 0xa5u) && bytes_are(raw + FENCE + bytes, FENCE, 0xa5u);
}

static int case_board(void) {
    UlcnetPrepostConfig cfg;
    UlcnetPrepostMemReq req;
    UlcnetPrepost *board, *twin;
    static Host host;
    static HopOut got[4], ref[4];
    void *twin_pool;
    unsigned char *raw;
    size_t bytes;

    CHECK(ulcnet_prepost_config_defaults(&cfg, &grid) == 0);   /* nothing registered */
    CHECK(ulcnet_prepost_get_mem_size(&cfg, &req) == 0);
    bytes = (size_t)req.bytes;
    raw = (unsigned char *)alloc_aligned(req.alignment, bytes + 2u * FENCE);
    CHECK(raw != NULL);
    memset(raw, 0xa5, bytes + 2u * FENCE);
    board = ulcnet_prepost_init_ex(raw + FENCE, bytes, &cfg, &req);
    CHECK(board != NULL);
    /* The same hops through a float host with registered state: the signal path is the same. */
    twin = class_make(&host, &twin_pool, NULL);
    CHECK(twin != NULL);
    CHECK(ulcnet_prepost_state_generation(board) == 0u);
    CHECK(ulcnet_prepost_state_generation(twin) == 0u);

    /* Two healthy hops. */
    CHECK(head_hop(board, 0, 1, 0) == 0 && head_hop(twin, 0, 0, 0) == 0);
    CHECK(ulcnet_prepost_post_process_freq(board, got[0].re, got[0].im) == 0);
    CHECK(ulcnet_prepost_post_process_freq(twin, ref[0].re, ref[0].im) == 0);
    CHECK(head_hop(board, 1, 1, 0) == 0 && head_hop(twin, 1, 0, 0) == 0);
    CHECK(ulcnet_prepost_post_process_freq(board, got[1].re, got[1].im) == 0);
    CHECK(ulcnet_prepost_post_process_freq(twin, ref[1].re, ref[1].im) == 0);
    CHECK(outs_identical(got, ref, 2));
    CHECK(!identical(got[0].re, got[1].re, BINS) && !identical(got[0].re, specs[0].er, BINS));
    CHECK(ulcnet_prepost_state_generation(board) == 0u && fences_intact(raw, bytes));

    /* A non-finite mask: refused, the generation moves by exactly one (nothing registered
     * included), a retry is refused without moving it again, the frame stays open for bypass. */
    CHECK(head_hop(board, 2, 1, 1) == -1 && head_hop(twin, 2, 0, 1) == -1);
    CHECK(ulcnet_prepost_state_generation(board) == 1u);
    CHECK(ulcnet_prepost_state_generation(twin) == 1u);
    CHECK(ulcnet_prepost_frame_finish(board) == -1);
    CHECK(ulcnet_prepost_state_generation(board) == 1u);
    CHECK(ulcnet_prepost_post_process_freq(board, got[2].re, got[2].im) == -1);
    CHECK(ulcnet_prepost_frame_bypass(board) == 0 && ulcnet_prepost_frame_bypass(twin) == 0);
    CHECK(ulcnet_prepost_post_process_freq(board, got[2].re, got[2].im) == 0);
    CHECK(identical(got[2].re, specs[2].er, BINS) && identical(got[2].im, specs[2].ei, BINS));
    CHECK(ulcnet_prepost_state_generation(board) == 1u && fences_intact(raw, bytes));

    /* The next hop recovers and matches the float host again. */
    CHECK(head_hop(board, 3, 1, 0) == 0 && head_hop(twin, 3, 0, 0) == 0);
    CHECK(ulcnet_prepost_post_process_freq(board, got[3].re, got[3].im) == 0);
    CHECK(ulcnet_prepost_post_process_freq(twin, ref[3].re, ref[3].im) == 0);
    CHECK(identical(got[3].re, ref[3].re, BINS) && identical(got[3].im, ref[3].im, BINS));
    CHECK(!identical(got[3].re, specs[3].er, BINS));
    CHECK(ulcnet_prepost_state_generation(board) == 1u && fences_intact(raw, bytes));

    free(raw);
    free(twin_pool);
    return 0;
}

/* ---- registration: all four pairs or none ----------------------------- */

typedef struct Pair { float **pointer; size_t *count; } Pair;

static void pairs_of(UlcnetPrepostConfig *cfg, Pair pairs[4]) {
    pairs[0] = (Pair){ &cfg->key_history, &cfg->key_history_elements };
    pairs[1] = (Pair){ &cfg->value_history, &cfg->value_history_elements };
    pairs[2] = (Pair){ &cfg->logit_history, &cfg->logit_history_elements };
    pairs[3] = (Pair){ &cfg->h_gru, &cfg->gru_hidden_elements };
}

static UlcnetModelIoStateBuffers buffers_of(const UlcnetPrepostConfig *cfg) {
    const UlcnetModelIoStateBuffers buffers = {
        .key_history = cfg->key_history, .key_history_elements = cfg->key_history_elements,
        .value_history = cfg->value_history, .value_history_elements = cfg->value_history_elements,
        .logit_history = cfg->logit_history, .logit_history_elements = cfg->logit_history_elements,
        .h_gru = cfg->h_gru, .gru_hidden_elements = cfg->gru_hidden_elements,
    };
    return buffers;
}

/* shape 0: only pair k; 1: pair k pointer without count; 2: pair k count without pointer; 3: all but pair k. */
static void partial_registration(UlcnetPrepostConfig *cfg, int shape, int k) {
    Pair pairs[4];
    int j;
    pairs_of(cfg, pairs);
    for (j = 0; j < 4; ++j) {
        const int mine = j == k;
        if (shape == 1 && mine) *pairs[j].count = 0u;
        else if (shape == 2 && mine) *pairs[j].pointer = NULL;
        else if ((shape == 0 && !mine) || (shape == 3 && mine)) {
            *pairs[j].pointer = NULL;
            *pairs[j].count = 0u;
        }
    }
}

/* Every entry refuses with its documented status and changes nothing. */
static int registration_refused(const UlcnetPrepostConfig *cfg, void *pool, size_t bytes,
                                const UlcnetPrepostMemReq *good, const Host *live,
                                const Host *snapshot) {
    UlcnetPrepostMemReq req, guard;
    const UlcnetModelIoStateBuffers buffers = buffers_of(cfg);
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    return ulcnet_prepost_get_mem_size(cfg, &req) == -1 &&
           memcmp(&req, &guard, sizeof req) == 0 &&
           ulcnet_model_io_state_buffers_validate(&buffers) == -1 &&
           ulcnet_prepost_init(pool, bytes, cfg) == NULL &&
           ulcnet_prepost_init_ex(pool, bytes, cfg, good) == NULL &&
           ulcnet_prepost_create(cfg) == NULL &&
           bytes_are(pool, bytes, 0xa5u) && memcmp(live, snapshot, sizeof *live) == 0;
}

static int case_registration(void) {
    UlcnetPrepostConfig full, none, cfg;
    UlcnetPrepostMemReq req, req_none;
    UlcnetModelIoStateBuffers all, empty;
    static Host host, snapshot;
    UlcnetPrepost *p;
    void *pool;
    size_t bytes;
    int shape, k;

    CHECK(ulcnet_prepost_config_defaults(&full, &grid) == 0);
    none = full;
    host_make(&host);
    host_register(&host, &full);
    /* The two accepted extremes first: the registration is in neither the pool size nor the hash. */
    all = buffers_of(&full);
    empty = buffers_of(&none);
    CHECK(ulcnet_model_io_state_buffers_validate(&all) == 0);
    CHECK(ulcnet_model_io_state_buffers_validate(&empty) == 0);
    CHECK(ulcnet_model_io_state_buffers_validate(NULL) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&full, &req) == 0);
    CHECK(ulcnet_prepost_get_mem_size(&none, &req_none) == 0);
    CHECK(memcmp(&req, &req_none, sizeof req) == 0);
    bytes = (size_t)req.bytes;
    pool = alloc_aligned(req.alignment, bytes);
    CHECK(pool != NULL);
    memset(pool, 0xa5, bytes);
    snapshot = host;

    for (shape = 0; shape < 4; ++shape) {
        for (k = 0; k < 4; ++k) {
            cfg = full;
            partial_registration(&cfg, shape, k);
            CHECK(registration_refused(&cfg, pool, bytes, &req, &host, &snapshot));
        }
    }

    /* The full registration is accepted afterwards: buffers zeroed, generation 0, a frame runs. */
    p = ulcnet_prepost_init_ex(pool, bytes, &full, &req);
    CHECK(p != NULL);
    CHECK(bytes_are(host.key, full.key_history_elements * sizeof(float), 0u));
    CHECK(bytes_are(host.gru, full.gru_hidden_elements * sizeof(float), 0u));
    CHECK(ulcnet_prepost_state_generation(p) == 0u);
    CHECK(drive_spec(p, 0, 3, outs_a) == 0);
    CHECK(!bytes_are(host.gru, full.gru_hidden_elements * sizeof(float), 0u));
    free(pool);
    return 0;
}

/* ---- grid: ulcnet_grid_make / ulcnet_grid_validate -------------------- */

/* A refused _make returns the status and leaves the grid unchanged. */
#define MAKE_REFUSED(status, ...) do { \
    const UlcnetGridSpec refused_spec = { __VA_ARGS__ }; \
    UlcnetGrid refused_out, refused_guard; \
    memset(&refused_out, 0xa5, sizeof refused_out); \
    refused_guard = refused_out; \
    CHECK(ulcnet_grid_make(&refused_out, &refused_spec) == (status)); \
    CHECK(memcmp(&refused_out, &refused_guard, sizeof refused_out) == 0); \
} while (0)

#define VALIDATE_AS(status, field, value) do { \
    UlcnetGrid bent = made; \
    bent.field = (value); \
    CHECK(ulcnet_grid_validate(&bent) == (status)); \
} while (0)

static int audiolib_builds(int n_fft) {
    AcStftConfig sc;
    ac_stft_config_default(&sc, n_fft);
    return ac_stft_get_mem_size(&sc) != 0u;
}

static int case_grid(void) {
    static const int bad_fft[] = { 0, -512, 8, 513, 1000, 16384 };
    UlcnetGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = D };
    UlcnetGrid made;
    size_t i;

    /* Accepted, both ends of the depth range included; the derived fields are the rule. */
    CHECK(audiolib_builds(512) && audiolib_builds(1024));
    CHECK(ulcnet_grid_make(&made, &spec) == ULCNET_GRID_OK);
    CHECK(made.sample_rate == 16000 && made.n_fft == 512 && made.hop == 256 &&
          made.n_bins == 257 && made.delay_depth == D);
    spec.n_fft = 1024;
    spec.delay_depth = ULCNET_MODEL_IO_MIN_D;
    CHECK(ulcnet_grid_make(&made, &spec) == ULCNET_GRID_OK);
    CHECK(made.hop == 512 && made.n_bins == 513 && made.delay_depth == ULCNET_MODEL_IO_MIN_D);
    spec.delay_depth = ULCNET_MODEL_IO_MAX_D;
    CHECK(ulcnet_grid_make(&made, &spec) == ULCNET_GRID_OK);
    CHECK(made.delay_depth == ULCNET_MODEL_IO_MAX_D);

    CHECK(ulcnet_grid_make(NULL, &spec) == ULCNET_GRID_ERR_NULL);
    MAKE_REFUSED(ULCNET_GRID_ERR_SAMPLE_RATE, .sample_rate = 0, .n_fft = 512, .delay_depth = D);
    MAKE_REFUSED(ULCNET_GRID_ERR_SAMPLE_RATE, .sample_rate = -16000, .n_fft = 512, .delay_depth = D);
    for (i = 0; i < sizeof bad_fft / sizeof bad_fft[0]; ++i) {
        CHECK(!audiolib_builds(bad_fft[i]));
        MAKE_REFUSED(ULCNET_GRID_ERR_N_FFT, .sample_rate = 16000, .n_fft = bad_fft[i],
                     .delay_depth = D);
    }
    MAKE_REFUSED(ULCNET_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512,
                 .delay_depth = ULCNET_MODEL_IO_MIN_D - 1);
    MAKE_REFUSED(ULCNET_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512,
                 .delay_depth = ULCNET_MODEL_IO_MAX_D + 1);
    MAKE_REFUSED(ULCNET_GRID_ERR_DELAY_DEPTH, .sample_rate = 16000, .n_fft = 512, .delay_depth = 0);
    /* The first failing check wins: sample rate, then n_fft, then the depth. */
    MAKE_REFUSED(ULCNET_GRID_ERR_SAMPLE_RATE, .sample_rate = 0, .n_fft = 0, .delay_depth = 0);
    MAKE_REFUSED(ULCNET_GRID_ERR_N_FFT, .sample_rate = 16000, .n_fft = 0, .delay_depth = 0);
    {
        UlcnetGrid untouched;
        memset(&untouched, 0xa5, sizeof untouched);
        made = untouched;
        CHECK(ulcnet_grid_make(&made, NULL) == ULCNET_GRID_ERR_NULL);
        CHECK(memcmp(&made, &untouched, sizeof made) == 0);
    }

    /* _validate: the same checks, then hop and n_bins must be what _make derives. */
    spec.n_fft = 512;
    spec.delay_depth = D;
    CHECK(ulcnet_grid_make(&made, &spec) == ULCNET_GRID_OK);
    CHECK(ulcnet_grid_validate(&made) == ULCNET_GRID_OK);
    CHECK(ulcnet_grid_validate(NULL) == ULCNET_GRID_ERR_NULL);
    VALIDATE_AS(ULCNET_GRID_ERR_SAMPLE_RATE, sample_rate, 0);
    VALIDATE_AS(ULCNET_GRID_ERR_N_FFT, n_fft, 0);
    VALIDATE_AS(ULCNET_GRID_ERR_N_FFT, n_fft, 513);
    VALIDATE_AS(ULCNET_GRID_ERR_DELAY_DEPTH, delay_depth, ULCNET_MODEL_IO_MIN_D - 1);
    VALIDATE_AS(ULCNET_GRID_ERR_DELAY_DEPTH, delay_depth, ULCNET_MODEL_IO_MAX_D + 1);
    VALIDATE_AS(ULCNET_GRID_ERR_INCONSISTENT, hop, made.hop + 1);
    VALIDATE_AS(ULCNET_GRID_ERR_INCONSISTENT, hop, made.n_fft);
    VALIDATE_AS(ULCNET_GRID_ERR_INCONSISTENT, n_bins, made.n_bins + 1);
    VALIDATE_AS(ULCNET_GRID_ERR_INCONSISTENT, n_bins, made.n_bins - 1);
    return 0;
}

int main(int argc, char **argv) {
    void *fft_mem = NULL;
    FftHandle *fft;
    int status;
    int sample_rate = 0, n_fft = 0;

    if (argc != 3 || sscanf(argv[2], "grid=%d,%d", &sample_rate, &n_fft) != 2) {
        fprintf(stderr, "usage: %s <case> grid=<sample rate>,<n_fft>\n", argv[0]);
        return 2;
    }
    {
        const UlcnetGridSpec spec = { .sample_rate = sample_rate, .n_fft = n_fft,
                                      .delay_depth = D };
        CHECK(ulcnet_grid_make(&grid, &spec) == ULCNET_GRID_OK);
    }
    CHECK(grid.n_bins <= BINS_MAX && grid.hop <= HOP_MAX && FEATURE <= TA_MAX);
    ac_window_build(window, N_FFT);
    fill_pcm(pcm_error, pcm_far, (int)(HOPS * HOP));
    fft = make_fft(N_FFT, &fft_mem);
    CHECK(fft != NULL);
    CHECK(build_specs() == 0);

    if (strcmp(argv[1], "equiv") == 0)          status = case_equiv(fft);
    else if (strcmp(argv[1], "stalepool") == 0) status = case_stalepool();
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle();
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject();
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset();
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard();
    else if (strcmp(argv[1], "layout") == 0)    status = case_layout();
    else if (strcmp(argv[1], "hash") == 0)      status = case_hash();
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
         '-I', _ULCNET_DIR, '-I', _AC_INCLUDE, '-I', _AC_DIR, str(source),
         *[os.path.join(_ULCNET_DIR, unit) for unit in _SOURCES],
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    """One executable for every case: the class plus the three TUs it
    composes, compiled at the house flags with -Werror."""
    return _build_driver(tmp_path_factory, audio_common_lib, 'ulcnet_prepost_c')


@pytest.fixture(scope='module')
def driver_scalar(tmp_path_factory, audio_common_lib):
    """The same driver with the SIMD kernels forced to their scalar twins."""
    return _build_driver(tmp_path_factory, audio_common_lib,
                         'ulcnet_prepost_c_scalar', '-DSIMD_KERNELS_FORCE_SCALAR')


def _run(driver, case, grid=_GRID_16K):
    done = subprocess.run([str(driver), case, grid], capture_output=True,
                          text=True)
    assert done.returncode == 0, (
        'case %s (%s) failed (rc=%d)\n%s' % (case, grid, done.returncode,
                                             done.stderr))
    return done.stdout


@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_class_matches_the_hand_composed_path(driver, grid):
    """The class vs analysis + the model-I/O transaction + synthesis wired up
    by hand, byte-identical over 400 hops, on the shipped grid and a second
    one. The same case pins the framing claim: audiolib's rolling analysis is
    the hand-written reference bit for bit, and the centered analysis's last
    frame per hop is bit-identical to the rolling frame."""
    assert 'pool: model I/O' in _run(driver, 'equiv', grid)


def test_stale_pool_gate_holds_and_the_pool_follows_the_grid(driver):
    """_init_ex must refuse a pool recorded for another D, another sample rate
    or n_fft, or a perturbed descriptor/layout/hash/size/alignment -- leaving
    the pool and the registered buffers byte for byte unchanged -- while
    still accepting the matching one; the pool size depends on the grid, not
    on D."""
    stdout = _run(driver, 'stalepool')
    assert 'pool =' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe, and on a
    pool instance it is the no-op the contract promises."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size / _init refuse NULLs, an inconsistent grid, a D outside
    [MIN_D, MAX_D] and an undersized or misaligned pool, leaving *req
    untouched; the stage calls refuse being run out of order."""
    _run(driver, 'reject')


def test_reset_clears_state(driver):
    """N hops, reset, the same N hops again -> byte-identical. Asserted
    against a NO-reset second run first, so the test cannot pass by the
    output being state-independent."""
    _run(driver, 'reset')


def test_frame_bypass_is_identity_and_does_not_step_the_rings(driver):
    """_frame_bypass passes the error spectrum through unchanged and leaves
    the K/V/logit rings and both GRU hidden tensors exactly as they were --
    checked against non-zero state, and against a commit that does move
    them."""
    _run(driver, 'skip')


def test_frame_state_machine_guards(driver):
    """A spectrum on top of an unfinished frame is refused (and the open frame
    stays the caller's to finish), a commit with no frame_open behind it
    is refused, and a commit refused on a non-finite output disarms the
    transaction until a fresh frame_open -- which NaN-refills every
    accelerator output. Skipping needs no frame_open at all."""
    _run(driver, 'guard')


def test_model_layout_version_field_accepts_only_the_built_layout(driver):
    """0 means this build's layout and an explicit match is the identical
    pool; every other number -- retired layouts, unassigned numbers -- is refused
    before sizing and by a pool-init that is handed a stale request."""
    _run(driver, 'layout')


def _fnv1a(values):
    """FNV-1a-32 over the little-endian bytes of each 32-bit value, as the
    class's pp_build_hash folds them."""
    h = 2166136261
    for value in values:
        for shift in (0, 8, 16, 24):
            h ^= (value >> shift) & 0xff
            h = (h * 16777619) & 0xffffffff
    return h


def test_pool_identity_moved_with_the_full_window_carve(driver):
    """The stale-pool gate keys on (grid, layout, carve version, D).

    The carve sets the size and meaning of the pool, so a pool requested under
    another carve must hash differently. The hash is recomputed here from the
    published constants and compared with what the C returns, then with what
    other carves would have produced."""
    out = _run(driver, 'hash')
    c_hash = int(re.search(r'hash=(\d+)', out).group(1))
    layout = int(re.search(r'layout=(\d+)', out).group(1))
    assert layout == 18
    with open(os.path.join(_ULCNET_DIR, 'ulcnet_prepost.h'),
              encoding='utf-8') as stream:
        carve = int(re.search(
            r'^#define\s+ULCNET_PREPOST_CARVE_VERSION\s+(\d+)u',
            stream.read(), re.MULTILINE).group(1))
    assert carve == 9
    # grid 16000 / 512, D == 8 (this driver's grid).
    expected = _fnv1a([16000, 512, layout, carve, 8])
    ptr_bytes = int(re.search(r'ptr_bytes=(\d+)', out).group(1))
    if ptr_bytes == 4:
        for byte in b'|abi:ilp32':
            expected = ((expected ^ byte) * 16777619) & 0xffffffff
    assert c_hash == expected
    assert c_hash != _fnv1a([16000, 512, layout, carve - 1, 8])
    assert c_hash != _fnv1a([16000, 512, layout, carve, 5])
    assert c_hash != _fnv1a([48000, 1024, layout, carve, 8])


@pytest.mark.parametrize('build', ['driver', 'driver_scalar'])
@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_board_form_nothing_registered_runs_and_counts_restarts(
        request, build, grid):
    """The board configuration: no state registered, the accelerator owns it.
    The state views are empty, the generation starts at 0, a commit refused on
    a non-finite mask moves it by exactly one (a retry does not), bypass is the
    identity, the next hop commits again and equals a float host's output, and
    the bytes around the pool are never written. Run on the NEON and the scalar
    build, on both grids."""
    _run(request.getfixturevalue(build), 'board', grid)


def test_state_registration_is_all_or_none(driver):
    """Every partial registration of the four state pairs -- one pair alone,
    all but one, a pointer without a count, a count without a pointer -- is
    refused by _get_mem_size (req untouched), _init, _init_ex and _create, and
    by the model-I/O validator, leaving the pool and the registered buffers
    byte for byte unchanged; the full registration is accepted afterwards
    and runs. Control flow only: no SIMD twin to compare."""
    _run(driver, 'registration')


def test_grid_make_and_validate_report_each_documented_status(driver):
    """ulcnet_grid_make: NULL, sample rate <= 0, an n_fft audiolib's framing
    does not build, a delay depth outside [MIN_D, MAX_D], the first failing
    check winning, the grid untouched on every refusal, both ends of the depth
    range accepted. ulcnet_grid_validate: the same statuses plus a hop or
    n_bins that is not what _make derives."""
    _run(driver, 'grid')
