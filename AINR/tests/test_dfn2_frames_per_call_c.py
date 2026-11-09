"""The DFN2 C host for N frames per call equals the one-frame host.

A grid with ``frames_per_call = N`` runs the accelerator once per N hops on an ``N + 2`` frame
window and composes the N returned frames in one burst. The stand-in accelerator below is a pure
function of (three-frame window slice, state) -> (heads, next state), applied to the N slices in
order the way the exported graph is, so the N-frame host and the one-frame host see the same
arithmetic and must agree BYTE FOR BYTE. One driver binary takes N as an argument:

  stream       the head sequence the accelerator produced, the final state of both recurrent
               tensors, and every output hop (the N-frame host trails by N - 1 hops: its hop
               h + N - 1 is the one-frame host's hop h), for a stream length that is not a
               multiple of N, with the single and the dual entry mixed
  reset        a reset in the middle of a call; both segments agree
  refuse       a call whose head or state comes back non-finite is refused, the state is
               zeroed and the N frames pass as the identity, the same as N one-frame refusals
  contract     readiness (N == 1: every hop, the first L being warm-up frames whose heads stay
               NaN and whose state the stand-in scribbles on; N > 1: L + N, then every N;
               unchanged), the window rows at the grid's stride, the
               descriptor's input_frames (N + 2), the latency accessor, element counts, a
               descriptor made for another grid refused before infer() runs, and a pool of the
               one-frame grid or of another grid refused
  abi          the 72-byte descriptor, filled and checked through a grid of either N: nothing
               written or read past it, a descriptor of 3 input frames refused by N > 1

A stand-in reading the wrong window slice fails the comparison, so it is not vacuous.

The one-frame host (N == 1) runs the warm-up frames of its stream start through the accelerator and
discards them (the stand-in leaves their heads NaN and overwrites the state: finish must neither
read the heads nor keep the state), so its output still equals the N-frame host's; N > 1 is
unchanged and is the regression gate for the older flow (no warm-up frame, no generation step).

The recurrent state is the host's: the driver registers the two buffers. The driver runs the
grid of ``dfn2_test_grid.h`` (the shipped 48 kHz grid unless ``-DDFN2_GRID_KHZ`` says otherwise)
with N frames per call; the 16 kHz and 32 kHz grids and a custom grid, whose windows are packed
at a smaller stride, are run as well. N outside 1..64 is refused at run time by
``dfn2_grid_make`` (test_dfn2_grid.py).
"""

import subprocess

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
_SOURCES = ('prepost', 'process', 'model_io')
PAIRS = ((1, 1), (0, 0), (2, 4), (0, 1), (2, 0))
FRAMES = (2, 3, 5, 10)
# Grids other than the shipped 48 kHz one: the 16 and 32 kHz presets of
# dfn2_test_grid.h and a custom grid (-DDFN2_SR and the three others) whose band
# count and deep-filter bin count are both below the shipped model's, so every
# window is packed at another stride.
_GRID_FLAGS = {
    48: [],
    32: ['-DDFN2_GRID_KHZ=32'],
    16: ['-DDFN2_GRID_KHZ=16'],
    'custom': ['-DDFN2_SR=16000', '-DDFN2_N_FFT=256', '-DDFN2_N_ERB=20',
               '-DDFN2_DF_BINS=64'],
}
OTHER_GRIDS = (16, 32, 'custom')
REAL_HOPS = 37
_pair_id = dfn_pairs.pair_id

_DRIVER = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_test_grid.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

/* The frames per call is the driver's first argument; the grid it names is g_grid. */
static DFN2Grid g_grid;
#define NF    (g_grid.frames_per_call)
#define LA    DFN2_MASK_LOOKAHEAD
#define DA    DFN2_DF_LOOKAHEAD
#define WF    (g_grid.input_frames)
#define E     DFN2_N_ERB
#define F     DFN2_DF_BINS
#define COEF1 (DFN2_DF_BINS * DFN2_DF_ORDER * 2)
#define ROWS  DFN2_TEST_GRU_ROWS
#define HID   DFN2_TEST_GRU_HIDDEN
#define CH    DFN2_TEST_CONVP_CHANNELS
#define PW    DFN2_TEST_CONVP_WINDOW
/* The window slice head i reads is (i + g_slice_offset) % N: 0 is the correct stand-in. */
static int g_slice_offset;

/* ---- ERB matrices: erb_inv a partition of unity. The shipped 48 kHz grid has
 * its border table; any other grid gets evenly spaced borders. */
static int erb_borders[E];
static float erb_fwd[DFN2_N_BINS][E];
static float erb_inv[E][DFN2_N_BINS];

/* The host's recurrent state, registered with every instance. */
static float host_gru[DFN2_TEST_GRU_ELEMENTS];
static float host_convp[DFN2_TEST_CONVP_ELEMENTS];

static void build_erb(void) {
    int segment = 0, k;
#if DFN2_N_FFT == 1024 && DFN2_N_ERB == 32
    static const int shipped[E] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
        50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
        325, 378, 440, 513
    };
    memcpy(erb_borders, shipped, sizeof shipped);
#else
    for (k = 0; k < E; ++k)
        erb_borders[k] = (int)((long)k * DFN2_N_BINS / (E - 1));
#endif
    for (k = 0; k < DFN2_N_BINS; ++k) {
        int lo, hi;
        float right, left;
        while (segment + 1 < E - 1 && k >= erb_borders[segment + 1]) ++segment;
        lo = erb_borders[segment];
        hi = erb_borders[segment + 1];
        right = (float)(k - lo) / (float)(hi - lo);
        left = 1.0f - right;
        erb_fwd[k][segment] = segment == 0 ? 2.0f * left : left;
        erb_fwd[k][segment + 1] = segment + 1 == E - 1 ? 2.0f * right : right;
        erb_inv[segment][k] = left;
        erb_inv[segment + 1][k] = right;
    }
}

/* ---- log: 24-byte records, then `count` floats ---------------------- */
typedef struct { int32_t tag, a, b, count; int64_t c; } Rec;
static FILE *log_file;
static void rec(char tag, int a, int b, long long c, const float *v, int n) {
    Rec r;
    if (!log_file) return;
    r.tag = tag; r.a = a; r.b = b; r.count = n; r.c = c;
    fwrite(&r, sizeof r, 1, log_file);
    if (n) fwrite(v, sizeof(float), (size_t)n, log_file);
}

/* ---- the stand-in accelerator --------------------------------------
 * One frame: heads and the next state from the three-frame window slice and
 * the state, updated in place (each element is read before it is written).
 * Every head depends on the slice and on both state tensors, and the pathway
 * window on all five of its frames. */
static void stub_frame(const float *erb, const float *spec0,
                       const float *spec1, float (*gru)[HID], float *hist,
                       float *mask, float *coefs, float *alpha) {
    /* erb: [3][E], spec0/spec1: [3][F], hist: [CH][PW][F], all packed at the
     * grid's stride */
#define HIST(ch, w, k) hist[(((size_t)(ch)) * PW + (w)) * F + (k)]
    float e = 0.0f, sp = 0.0f;
    int t, b, r, j, ch, k;
    for (t = 0; t < 3; ++t)
        for (b = 0; b < E; ++b)
            e += erb[t * E + b] * (0.01f * (float)(t + 1) + 0.001f * (float)b);
    for (t = 0; t < 3; ++t)
        for (k = 0; k < F; ++k)
            sp += spec0[t * F + k] * 0.002f * (float)(t + 1) -
                  spec1[t * F + k] * 0.001f;
    for (r = 0; r < ROWS; ++r)
        for (j = 0; j < HID; ++j)
            gru[r][j] = tanhf(0.9f * gru[r][j] + 0.05f * e +
                              0.03f * (float)(r + 1) * sp + 0.001f * (float)j);
    for (ch = 0; ch < CH; ++ch) {
        memmove(&HIST(ch, 0, 0), &HIST(ch, 1, 0),
                (PW - 1) * (size_t)F * sizeof(float));
        for (k = 0; k < F; ++k)
            HIST(ch, PW - 1, k) = tanhf(
                0.1f * spec0[1 * F + k] * (float)(ch + 1) / (float)CH +
                0.5f * gru[ch % ROWS][k] + 0.2f * HIST(ch, 0, k));
    }
    for (b = 0; b < E; ++b)
        mask[b] = 1.0f / (1.0f + expf(-(erb[1 * E + b] + gru[1][b])));
    for (j = 0; j < COEF1; ++j)
        coefs[j] = 0.1f * tanhf(HIST(j % CH, (j / CH) % PW, j % F) +
                                gru[3][j % HID]);
    *alpha = 1.0f / (1.0f + expf(-gru[4][0]));
#undef HIST
}

typedef struct {
    long long next_head;          /* head index of the call's first frame */
    long long poison_lo, poison_hi;
    int poison_kind;              /* 1: a NaN head, 2: a NaN state element */
    int calls;
    int bad_views;
    int warm_left;                /* warm-up frames of a one-frame grid still to come */
} Stub;

static int stub_infer(void *user, const DFN2PrepostInputs *in,
                      DFN2PrepostOutputs *out) {
    Stub *s = (Stub *)user;
    int i, poisoned = 0;
    ++s->calls;
    if (in->erb_window_elements != (size_t)WF * E ||
        in->spec_window_elements != (size_t)2 * WF * F ||
        out->erb_mask_elements != (size_t)NF * E ||
        out->coefs_elements != (size_t)NF * COEF1 ||
        out->alpha_elements != (size_t)NF ||
        in->df_convp_window_elements != DFN2_TEST_CONVP_ELEMENTS ||
        in->gru_hidden_elements != DFN2_TEST_GRU_ELEMENTS ||
        in->gru_hidden != host_gru || out->gru_hidden_next != host_gru ||
        in->df_convp_window != host_convp || out->df_convp_window_next != host_convp) {
        ++s->bad_views;
        return -1;
    }
    if (s->warm_left > 0) {       /* warm-up frame: heads left NaN, nothing logged, state overwritten */
        --s->warm_left;
        for (i = 0; i < (int)in->gru_hidden_elements; ++i) host_gru[i] = 99.0f;
        return 0;
    }
    for (i = 0; i < NF; ++i) {
        const int at = (i + g_slice_offset) % NF;
        long long head = s->next_head + i;
        stub_frame(in->erb_window + (size_t)at * E,
                   in->spec_window + ((size_t)0 * WF + at) * F,
                   in->spec_window + ((size_t)1 * WF + at) * F,
                   (float (*)[HID])out->gru_hidden_next, out->df_convp_window_next,
                   out->erb_mask + (size_t)i * E,
                   out->coefs + (size_t)i * COEF1, out->alpha + i);
        rec('H', (int)head, 0, head, out->erb_mask + (size_t)i * E, E);
        rec('C', (int)head, 0, head, out->coefs + (size_t)i * COEF1, COEF1);
        rec('A', (int)head, 0, head, out->alpha + i, 1);
        if (head >= s->poison_lo && head < s->poison_hi) poisoned = 1;
    }
    if (poisoned && s->poison_kind == 1) out->alpha[NF - 1] = (float)NAN;
    if (poisoned && s->poison_kind == 2)
        out->gru_hidden_next[(size_t)(ROWS - 1) * HID + HID - 1] = (float)NAN;
    s->next_head += NF;
    return 0;
}

static float noise(long long hop, int i) {
    uint32_t x = (uint32_t)(hop * 2654435761u) ^ (uint32_t)(i * 40503u + 17u);
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u;
    x ^= x >> 16;
    return ((float)(x & 0xffffu) / 65535.0f - 0.5f) * 0.2f;
}

static DFN2Prepost *make(void) {
    DFN2PrepostConfig cfg;
    if (dfn2_prepost_config_defaults(&cfg, &g_grid) != 0) return NULL;
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    cfg.gru_hidden = host_gru;
    cfg.gru_hidden_elements = DFN2_TEST_GRU_ELEMENTS;
    cfg.df_convp_history = host_convp;
    cfg.df_convp_history_elements = DFN2_TEST_CONVP_ELEMENTS;
    return dfn2_prepost_create(&cfg);
}

static DFN2PrepostInputs views_in;
static DFN2PrepostOutputs views_out;

/* freq, reset, refuse: one stream through dfn2_model_run_frame. */
static int run_stream(int real, int hops, int reset_hop, long long plo, long long phi, int pkind) {
    DFN2Prepost *p;
    DFN2ModelIoDescriptor descriptor;
    DFN2Model model;
    Stub stub;
    static float est_re[DFN2_N_BINS], est_im[DFN2_N_BINS];
    static float app_re[DFN2_N_BINS], app_im[DFN2_N_BINS];
    static float out_spec[2 * DFN2_N_BINS];
    int h, k;

    p = make();
    CHECK(p != NULL);
    CHECK(dfn2_model_io_descriptor_default(&g_grid, &descriptor) == 0);
    memset(&stub, 0, sizeof stub);
    stub.warm_left = NF == 1 ? LA : 0;
    stub.poison_lo = plo;
    stub.poison_hi = phi;
    stub.poison_kind = pkind;
    model.user = &stub;
    model.infer = stub_infer;
    model.reset = NULL;
    model.io_descriptor = &descriptor;

    for (h = 0; h < hops; ++h) {
        const int live = h < real;
        int rc, written = 0;
        long long frame = -1;
        if (h == reset_hop) {
            dfn2_prepost_reset(p);
            stub.next_head = 0;
            stub.warm_left = NF == 1 ? LA : 0;
            rec('R', h, 0, 0, NULL, 0);
        }
        for (k = 0; k < DFN2_N_BINS; ++k) {
            est_re[k] = live ? noise(h, 2 * k) : 0.0f;
            est_im[k] = live ? noise(h, 2 * k + 1) : 0.0f;
            app_re[k] = 0.5f * est_re[k] + (live ? noise(h + 7, k) : 0.0f);
            app_im[k] = 0.5f * est_im[k];
        }
        rc = h % 3 == 0
            ? dfn2_prepost_pre_process_freq_dual(p, est_re, est_im, app_re, app_im)
            : dfn2_prepost_pre_process_freq(p, est_re, est_im);
        CHECK(rc == 0 || rc == 1);
        if (rc == 1) CHECK(dfn2_model_run_frame(&model, p) >= 0);
        CHECK(dfn2_prepost_post_process_freq(
                  p, out_spec, out_spec + DFN2_N_BINS, &written) == 0);
        (void)dfn2_prepost_output_frame_index(p, &frame);
        rec('O', h, (written ? 1 : 0) | (rc << 1), frame, out_spec, 2 * DFN2_N_BINS);
    }
    CHECK(stub.bad_views == 0);
    /* The final recurrent state, read through the views of the next open
     * frame: zero hops are pushed (they touch only the feature windows) until
     * one opens. */
    {
        int rc;
        for (;;) {
            memset(est_re, 0, sizeof est_re);
            memset(est_im, 0, sizeof est_im);
            rc = dfn2_prepost_pre_process_freq(p, est_re, est_im);
            CHECK(rc == 0 || rc == 1);
            if (rc == 1) break;
            CHECK(dfn2_prepost_post_process_freq(
                      p, out_spec, out_spec + DFN2_N_BINS, NULL) == 0);
        }
        CHECK(dfn2_prepost_frame_open(p, &views_in, &views_out) == 0);
        rec('S', 0, 0, 0, views_in.gru_hidden, (int)views_in.gru_hidden_elements);
        rec('W', 0, 0, 0, views_in.df_convp_window,
            (int)views_in.df_convp_window_elements);
    }
    rec('G', (int)dfn2_prepost_state_generation(p), 0, 0, NULL, 0);
    printf("calls=%d generation=%u\n", stub.calls,
           dfn2_prepost_state_generation(p));
    dfn2_prepost_destroy(p);
    return 0;
}

static int case_contract(void) {
    const DFN2Grid *grid = &g_grid;
    DFN2Grid other;
    DFN2GridSpec other_spec;
    DFN2ModelIoDescriptor d, bad, other_d;
    DFN2ModelIOState io;
    DFN2PrepostMemReq freq_req;
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    DFN2Model model;
    Stub stub;
    const size_t io_bytes = dfn2_model_io_mem_size(grid);
    void *io_block = aligned_alloc(16u, io_bytes);
    static float erb[E], spec[2][F], est_re[DFN2_N_BINS], est_im[DFN2_N_BINS];
    static float out_spec[2 * DFN2_N_BINS];
    int f, s, b, h, written, first_written = -1;

    CHECK(io_block != NULL);
    CHECK(dfn2_model_io_descriptor_default(grid, &d) == 0);
    CHECK(d.input_frames == WF && d.input_frames - 2 == NF);
    CHECK(dfn2_model_io_descriptor_validate(grid, &d) == 0);
    bad = d; ++bad.input_frames;
    CHECK(dfn2_model_io_descriptor_validate(grid, &bad) == -1);
    bad = d; bad.input_frames = DFN2_MODEL_INPUT_FRAMES + (NF == 1);
    CHECK(dfn2_model_io_descriptor_validate(grid, &bad) == -1);
    /* The grid is part of the contract: a descriptor made for another grid
     * (fewer ERB bands) is not this one. */
    other_spec = dfn2_test_grid_spec();
    other_spec.n_erb = DFN2_N_ERB - 4;
    other_spec.frames_per_call = NF;
    CHECK(dfn2_grid_make(&other, &other_spec) == DFN2_GRID_OK);
    CHECK(dfn2_model_io_descriptor_default(&other, &other_d) == 0);
    CHECK(other_d.erb_bands == DFN2_N_ERB - 4);
    CHECK(dfn2_model_io_descriptor_validate(grid, &other_d) == -1);
    CHECK(dfn2_model_io_descriptor_validate(&other, &d) == -1);

    /* Readiness, slots and the window rows, frame by frame. */
    CHECK(dfn2_model_io_ready_frames(NULL) == 0);
    CHECK(dfn2_model_io_init(&io, io_block, io_bytes, grid, NULL, 0, NULL, 0) == 0);
    CHECK(dfn2_model_io_ready_frames(&io) == (unsigned int)(LA + NF));
    CHECK(dfn2_model_io_frame_slot(&io) == -1);
    CHECK(dfn2_model_io_frame_slot(NULL) == -1);
    for (f = 0; f < LA + 6 * NF + 3; ++f) {
        int ready, want;
        for (b = 0; b < E; ++b) erb[b] = (float)(1000 * (f + 1) + b);
        for (b = 0; b < F; ++b) {
            spec[0][b] = (float)(1000 * (f + 1) + b) + 0.25f;
            spec[1][b] = -(float)(1000 * (f + 1) + b);
        }
        ready = dfn2_model_io_push_features(&io, erb, &spec[0][0]);
        want = f + 1 >= LA + NF && (f + 1 - LA) % NF == 0;
        CHECK(ready == want);
        CHECK(dfn2_model_io_frame_slot(&io) == (f < LA ? -1 : (f - LA) % NF));
        for (s = 0; s < WF; ++s) {
            const int frame = f - (WF - 1) + s;
            for (b = 0; b < E; ++b)
                CHECK(io.erb_window[s * E + b] ==
                      (frame < 0 ? 0.0f : (float)(1000 * (frame + 1) + b)));
            CHECK(io.spec_window[(0 * WF + s) * F + 3] ==
                  (frame < 0 ? 0.0f : (float)(1000 * (frame + 1) + 3) + 0.25f));
            CHECK(io.spec_window[(1 * WF + s) * F + F - 1] ==
                  (frame < 0 ? 0.0f : -(float)(1000 * (frame + 1) + F - 1)));
        }
    }
    free(io_block);

    /* The class: pre_process readiness, views, latency, first output. */
    p = make();
    CHECK(p != NULL);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == LA + DA + NF - 1);
    CHECK(dfn2_prepost_latency_frames(grid) == LA + DA + NF - 1);
    CHECK(dfn2_prepost_grid(p) != NULL && dfn2_prepost_grid(p)->n_erb == E &&
          dfn2_prepost_grid(p)->df_bins == F &&
          dfn2_prepost_grid(p)->frames_per_call == NF);
    memset(&stub, 0, sizeof stub);
    stub.poison_lo = stub.poison_hi = -1;
    stub.warm_left = NF == 1 ? LA : 0;   /* the warm-up frames' heads stay NaN: finish must not read them */
    model.user = &stub;
    model.infer = stub_infer;
    model.reset = NULL;
    bad = d;
    bad.input_frames = DFN2_MODEL_INPUT_FRAMES + (NF == 1);
    for (h = 0; h < LA + DA + 6 * NF + 4; ++h) {
        int rc, want;
        for (b = 0; b < DFN2_N_BINS; ++b) {
            est_re[b] = noise(h, 2 * b);
            est_im[b] = noise(h, 2 * b + 1);
        }
        rc = dfn2_prepost_pre_process_freq(p, est_re, est_im);
        want = NF == 1 ? 1 : (h + 1 >= LA + NF && (h + 1 - LA) % NF == 0);
        CHECK(rc == want);
        if (rc == 1) {
            int calls = stub.calls;
            /* Another N's descriptor is refused before infer() runs. */
            model.io_descriptor = &bad;
            CHECK(dfn2_model_run_frame(&model, p) == -1);
            CHECK(stub.calls == calls);
            /* ... and so is one made for another grid. */
            model.io_descriptor = &other_d;
            CHECK(dfn2_model_run_frame(&model, p) == -1);
            CHECK(stub.calls == calls);
            CHECK(dfn2_prepost_frame_open(p, &views_in, &views_out) == 0);
            CHECK(views_in.erb_window_elements ==
                  dfn2_prepost_erb_window_elements(grid));
            CHECK(dfn2_prepost_erb_window_elements(grid) == (size_t)WF * E);
            CHECK(dfn2_prepost_spec_window_elements(grid) == (size_t)2 * WF * F);
            CHECK(dfn2_prepost_erb_mask_elements(grid) == (size_t)NF * E);
            CHECK(dfn2_prepost_coefs_elements(grid) == (size_t)NF * COEF1);
            CHECK(views_in.df_convp_window_elements == (size_t)CH * PW * F);
            CHECK(views_out.alpha_elements == (size_t)NF);
            CHECK(isnan(views_out.alpha[NF - 1]));
            model.io_descriptor = &d;
            CHECK(dfn2_model_run_frame(&model, p) == 1);
            CHECK(stub.calls == calls + 1);
        }
        CHECK(dfn2_prepost_post_process_freq(p, out_spec, out_spec + DFN2_N_BINS,
                                             &written) == 0);
        if (written && first_written < 0) first_written = h;
    }
    CHECK(first_written == LA + DA + NF - 1);
    /* one generation step per warm-up frame (N == 1 only); with N > 1 the frames before the first call
     * open nothing and move nothing */
    CHECK(dfn2_prepost_state_generation(p) == (unsigned)(1 + (NF == 1 ? LA : 0)));
    dfn2_prepost_destroy(p);

    CHECK(dfn2_prepost_config_defaults(&cfg, grid) == 0);
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    CHECK(dfn2_prepost_get_mem_size(&cfg, &freq_req) == 0);
    printf("req freq=%llu:%08x\n", (unsigned long long)freq_req.bytes,
           (unsigned)freq_req.build_flags_hash);
    return 0;
}

/* A pool descriptor from another grid or frame count is refused; this grid's own is
 * accepted. argv: bytes hash. */
static int case_req_check(unsigned long long bytes, unsigned hash) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, other;
    size_t room;
    void *pool;
    CHECK(dfn2_prepost_config_defaults(&cfg, &g_grid) == 0);
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    other = req;
    other.bytes = bytes;
    other.build_flags_hash = hash;
    room = (size_t)(req.bytes > bytes ? req.bytes : bytes);
    room = (room + 15u) / 16u * 16u;
    pool = aligned_alloc(16u, room);
    CHECK(pool != NULL);
    CHECK(dfn2_prepost_init_ex(pool, room, &cfg, &req) != NULL);
    CHECK(dfn2_prepost_init_ex(pool, room, &cfg, &other) == NULL);
    free(pool);
    printf("req_check refused\n");
    return 0;
}

/* The grid of this program with `frames` frames per call, or exit 2. */
static void set_frames(const char *text) {
    DFN2GridSpec spec = dfn2_test_grid_spec();
    spec.frames_per_call = atoi(text);
    if (dfn2_grid_make(&g_grid, &spec) != DFN2_GRID_OK) exit(2);
}

/* argv: contract N | req_check N bytes hash |
 *       N log real hops reset poison_lo poison_hi kind [slice_offset] */
int main(int argc, char **argv) {
    build_erb();
    if (argc == 3 && strcmp(argv[1], "contract") == 0) {
        set_frames(argv[2]);
        return case_contract();
    }
    if (argc == 5 && strcmp(argv[1], "req_check") == 0) {
        set_frames(argv[2]);
        return case_req_check(strtoull(argv[3], NULL, 10), (unsigned)strtoul(argv[4], NULL, 16));
    }
    if (argc == 9 || argc == 10) {
        int rc;
        set_frames(argv[1]);
        if (argc == 10) g_slice_offset = atoi(argv[9]);
        log_file = fopen(argv[2], "wb");
        if (!log_file) return 2;
        rc = run_stream(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoll(argv[6]),
                        atoll(argv[7]), atoi(argv[8]));
        fclose(log_file);
        return rc;
    }
    return 2;
}
'''

_REC = np.dtype([('tag', '<i4'), ('a', '<i4'), ('b', '<i4'),
                 ('count', '<i4'), ('c', '<i8')])


def _defines(pair, khz=48):
    return list(dfn_pairs.defines(FAMILY, pair)) + _GRID_FLAGS[khz]


def _build(work, audio_common_lib, pair, khz=48):
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    exe = work / 'driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), *_defines(pair, khz),
         '-I', str(dfn_pairs.AINR / 'tests'), str(source),
         *[str(dfn_pairs.source_path(FAMILY, kind)) for kind in _SOURCES],
         audio_common_lib, '-lm', '-o', str(exe)],
        check=True, capture_output=True, text=True)
    return exe


@pytest.fixture(scope='module')
def audio_common_lib():
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    return dfn_pairs.audio_common_lib()


@pytest.fixture(scope='module')
def drivers(tmp_path_factory, audio_common_lib):
    """One driver binary per (pair, grid); the frame count is its argument."""
    built = {}

    def get(pair, khz=48):
        key = (pair, khz)
        if key not in built:
            work = tmp_path_factory.mktemp('dfn2_%s_%s' % (khz, _pair_id(pair)))
            built[key] = _build(work, audio_common_lib, pair, khz)
        return built[key]
    return get


def _parse(path):
    raw = path.read_bytes()
    at, records = 0, []
    while at < len(raw):
        head = np.frombuffer(raw, _REC, 1, at)[0]
        at += _REC.itemsize
        count = int(head['count'])
        values = np.frombuffer(raw, '<f4', count, at)
        at += 4 * count
        records.append((chr(head['tag']), int(head['a']), int(head['b']),
                        int(head['c']), values))
    return records


def _stream(driver, tmp_path, frames, hops, reset=-1, poison=(-1, -1), kind=0,
            name='log', slice_offset=0):
    log = tmp_path / ('%s.bin' % name)
    done = subprocess.run(
        [str(driver), str(frames), str(log), str(REAL_HOPS), str(hops), str(reset),
         str(poison[0]), str(poison[1]), str(kind), str(slice_offset)],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return _parse(log), done.stdout


def _segments(records):
    """Split at reset markers: per segment, the heads (index -> bytes of
    mask, coefs, alpha), the output records and the final state."""
    segments = [{'heads': {}, 'out': [], 'state': None}]
    for tag, a, b, c, values in records:
        seg = segments[-1]
        if tag == 'R':
            segments.append({'heads': {}, 'out': [], 'state': None})
        elif tag in 'HCA':
            seg['heads'].setdefault(a, {})[tag] = values.tobytes()
        elif tag == 'O':
            seg['out'].append((b & 1, c, values.tobytes(), b >> 1))
        elif tag in 'SW':
            seg.setdefault('final', {})[tag] = values.tobytes()
    return segments


def _hops(pair, frames, real=REAL_HOPS):
    """Hops that emit every real frame from the N-frame host and leave both
    hosts with the same number of heads: L + N * K."""
    mask, df = pair
    need = real + mask + df + frames - 1
    calls = -(-(need - mask) // frames)
    return mask + frames * calls


def _compare(one, many, frames, poison=None):
    """The N-frame segment `many` against the one-frame segment `one`."""
    heads = sorted(many['heads'])
    assert heads == list(range(len(heads)))
    skip = set(range(*poison)) if poison else set()
    for index in heads:
        if index not in skip:
            assert many['heads'][index] == one['heads'][index], index
    delay = frames - 1
    out_many, out_one = many['out'], one['out']
    assert len(out_many) == len(out_one)
    for hop in range(len(out_one) - delay):
        assert out_many[hop + delay][:3] == out_one[hop][:3], hop
    for hop in range(delay):
        assert out_many[hop][0] == 0, 'nothing is emitted in the first N-1 hops'
    return len(heads)


@pytest.mark.parametrize('frames', FRAMES)
@pytest.mark.parametrize('pair', PAIRS, ids=_pair_id)
def test_chunk_host_equals_the_frame_host(drivers, tmp_path, pair, frames):
    hops = _hops(pair, frames)
    driver = drivers(pair)
    one, _ = _stream(driver, tmp_path, 1, hops, name='one')
    many, out = _stream(driver, tmp_path, frames, hops, name='many')
    one, many = _segments(one)[0], _segments(many)[0]
    count = _compare(one, many, frames)
    assert count == hops - pair[0] == len(one['heads'])
    assert many['final'] == one['final']
    assert 'calls=%d ' % (count // frames) in out
    emitted = [entry for entry in many['out'] if entry[0]]
    assert many['out'].index(emitted[0]) == sum(pair) + frames - 1
    assert [entry[1] for entry in emitted][:REAL_HOPS] == list(
        range(REAL_HOPS))
    # Not a vacuous pass: the heads change the signal.
    assert len({entry[2] for entry in emitted}) > REAL_HOPS // 2


@pytest.mark.parametrize('frames', [3, 5])
@pytest.mark.parametrize('khz', OTHER_GRIDS)
def test_chunk_host_equals_the_frame_host_on_other_grids(drivers, tmp_path,
                                                         khz, frames):
    """The windows are packed at the grid's stride (24, 28 or 20 bands and, for
    the custom grid, 64 bins, not the shipped model's 32 and 96): the N-frame host
    still equals the one-frame host byte for byte, and the stand-in reads the
    same slices from the packed windows."""
    pair = (1, 1)
    hops = _hops(pair, frames)
    driver = drivers(pair, khz=khz)
    one, _ = _stream(driver, tmp_path, 1, hops, name='one')
    many, out = _stream(driver, tmp_path, frames, hops, name='many')
    one, many = _segments(one)[0], _segments(many)[0]
    count = _compare(one, many, frames)
    assert count == hops - pair[0] == len(one['heads'])
    assert many['final'] == one['final']
    assert 'calls=%d ' % (count // frames) in out
    emitted = [entry for entry in many['out'] if entry[0]]
    assert [entry[1] for entry in emitted][:REAL_HOPS] == list(
        range(REAL_HOPS))
    assert len({entry[2] for entry in emitted}) > REAL_HOPS // 2


@pytest.mark.parametrize('frames', [3, 5])
@pytest.mark.parametrize('pair', [(1, 1), (2, 4), (0, 1)], ids=_pair_id)
def test_a_reset_in_the_middle_of_a_call(drivers, tmp_path, pair, frames):
    """Reset with a call half full: the pending frames are dropped with the
    rest of the stream state, and each segment agrees with the one-frame host
    on everything the N-frame host had emitted."""
    reset = pair[0] + 2 * frames + 1
    hops = reset + _hops(pair, frames, real=12)
    driver = drivers(pair)
    one, _ = _stream(driver, tmp_path, 1, hops, reset=reset, name='one')
    many, out = _stream(driver, tmp_path, frames, hops, reset=reset, name='many')
    one, many = _segments(one), _segments(many)
    assert len(one) == len(many) == 2
    # Before the reset the N-frame host ran only its complete calls and had
    # emitted up to the hop of the reset.
    first = many[0]
    assert sorted(first['heads']) == list(range(2 * frames))
    for index in first['heads']:
        assert first['heads'][index] == one[0]['heads'][index]
    for hop in range(reset - (frames - 1)):
        assert first['out'][hop + frames - 1][:3] == one[0]['out'][hop][:3]
    count = _compare(one[1], many[1], frames)
    assert count == len(one[1]['heads'])
    assert many[1]['final'] == one[1]['final']
    assert 'generation=2' in out


@pytest.mark.parametrize('kind', [1, 2], ids=['nan_head', 'nan_state'])
@pytest.mark.parametrize('frames', [2, 5])
@pytest.mark.parametrize('pair', [(1, 1), (0, 0), (2, 4)], ids=_pair_id)
def test_a_refused_call_is_n_refused_frames(drivers, tmp_path, pair, frames,
                                            kind):
    """A call whose head (kind 1) or state (kind 2) comes back non-finite is
    refused: both recurrent tensors are zeroed, its N frames take the
    identity, the generation advances once. The one-frame host refusing the
    same N frames one by one ends in the same zero state and the same
    output."""
    hops = _hops(pair, frames)
    poison = (3 * frames, 4 * frames)
    driver = drivers(pair)
    one, out_one = _stream(driver, tmp_path, 1, hops, poison=poison, kind=kind,
                           name='one')
    many, out_many = _stream(driver, tmp_path, frames, hops, poison=poison,
                             kind=kind, name='many')
    one, many = _segments(one)[0], _segments(many)[0]
    _compare(one, many, frames, poison=poison)
    assert many['final'] == one['final']
    # N > 1: one refused call, no warm-up frame (unchanged). N == 1: `frames` refused frames and the
    # pair[0] warm-up frames of the stream start, one step each.
    assert 'generation=2' in out_many
    assert 'generation=%d' % (1 + frames + pair[0]) in out_one
    # The identity reached the output: the refused frames differ from a run
    # without the refusal.
    clean, _ = _stream(driver, tmp_path, frames, hops, name='clean')
    clean = _segments(clean)[0]
    assert [e[:3] for e in clean['out']] != [e[:3] for e in many['out']]


def test_a_stand_in_on_the_wrong_slice_fails_the_comparison(drivers,
                                                            tmp_path):
    """The comparison reads the window slices: an accelerator that takes
    slice i + 1 for head i diverges from the one-frame host."""
    pair, frames = (1, 1), 3
    hops = _hops(pair, frames)
    one, _ = _stream(drivers(pair), tmp_path, 1, hops, name='one')
    many, _ = _stream(drivers(pair), tmp_path, frames, hops, name='many', slice_offset=1)
    with pytest.raises(AssertionError):
        _compare(_segments(one)[0], _segments(many)[0], frames)


def _contract(driver, frames):
    done = subprocess.run([str(driver), 'contract', str(frames)],
                          capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    line = [item for item in done.stdout.splitlines()
            if item.startswith('req ')][0]
    fields = dict(item.split('=') for item in line.split()[1:])
    return {key: (int(value.split(':')[0]), value.split(':')[1])
            for key, value in fields.items()}


@pytest.mark.parametrize('frames', (1,) + FRAMES)
@pytest.mark.parametrize('pair', PAIRS, ids=_pair_id)
def test_contract_readiness_window_descriptor_and_latency(drivers, pair,
                                                          frames):
    _contract(drivers(pair), frames)


@pytest.mark.parametrize('frames', (1, 3, 5))
@pytest.mark.parametrize('khz', OTHER_GRIDS)
def test_contract_on_other_grids(drivers, khz, frames):
    _contract(drivers((1, 1), khz=khz), frames)


@pytest.mark.parametrize('frames', FRAMES)
def test_the_pool_of_another_frame_count_is_refused(drivers, frames):
    pair = (1, 1)
    driver = drivers(pair)
    single = _contract(driver, 1)
    chunked = _contract(driver, frames)
    assert chunked['freq'][1] != single['freq'][1]
    assert chunked['freq'][0] > single['freq'][0]
    bytes_, hash_ = single['freq']
    done = subprocess.run(
        [str(driver), 'req_check', str(frames), str(bytes_), hash_],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)


@pytest.mark.parametrize('frames', (1, 3))
@pytest.mark.parametrize('khz', OTHER_GRIDS)
def test_the_pool_of_another_grid_is_refused(drivers, khz, frames):
    """The grid is folded into the pool descriptor: the 48 kHz host refuses
    the descriptor (size and hash) a host of another grid computed, and
    accepts its own."""
    pair = (1, 1)
    shipped_driver = drivers(pair)
    shipped = _contract(shipped_driver, frames)
    other = _contract(drivers(pair, khz=khz), frames)
    assert other['freq'][1] != shipped['freq'][1]
    # the hash alone is enough to refuse it: same size, other hash
    bytes_, hash_ = shipped['freq']
    done = subprocess.run(
        [str(shipped_driver), 'req_check', str(frames), str(bytes_), other['freq'][1]],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    # and the size alone: the other grid's pool size with this grid's hash
    done = subprocess.run(
        [str(shipped_driver), 'req_check', str(frames), str(other['freq'][0]), hash_],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)


# ---------------------------------------------------------------------------
# the descriptor's size
# ---------------------------------------------------------------------------

# A unit that declares the descriptor itself: the 18 fields, 72 bytes, with a guard band behind
# it, passed to the library through the public prototypes only. The grid comes from
# dfn2_grid.h, which carries no model-I/O declarations; argv[1] is the frames per call.
_ABI_DRIVER = r"""
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_grid.h"

typedef struct {
    unsigned int layout_version;
    int sample_rate;
    int fft_size;
    int hop_size;
    int spectrum_bins;
    int erb_bands;
    int df_bins;
    int df_order;
    int mask_lookahead;
    int df_lookahead;
    int input_frames;
    int encoder_gru_layers;
    int erb_gru_layers;
    int df_gru_layers;
    int gru_hidden;
    int encoder_channels;
    int df_pathway_window;
    int state_tensor_count;
} OldDescriptor;

typedef struct DFN2ModelIoDescriptor DFN2ModelIoDescriptor;
int dfn2_model_io_descriptor_default(const DFN2Grid *grid,
                                     DFN2ModelIoDescriptor *descriptor);
int dfn2_model_io_descriptor_validate(
    const DFN2Grid *grid, const DFN2ModelIoDescriptor *descriptor);

typedef struct {
    OldDescriptor d;
    unsigned char guard[16];
} Box;
static Box box;

_Static_assert(sizeof(OldDescriptor) == 72 &&
                   offsetof(Box, guard) == 72,
               "the guard band starts right after the descriptor");

static int guard_intact(void)
{
    size_t i;
    for (i = 0; i < sizeof box.guard; ++i)
        if (box.guard[i] != 0xA5) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    DFN2ModelIoDescriptor *d = (DFN2ModelIoDescriptor *)&box.d;
    DFN2GridSpec spec = {
        .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = @ORDER@,
        .mask_lookahead = 1, .df_lookahead = 1, .frames_per_call = 1,
    };
    DFN2Grid grid;
    int written, filled, as_filled, old_frames, made;
    if (argc != 2) return 2;
    spec.frames_per_call = atoi(argv[1]);
    made = dfn2_grid_make(&grid, &spec);
    memset(&box, 0xA5, sizeof box);
    filled = dfn2_model_io_descriptor_default(&grid, d);
    written = guard_intact();
    memset(box.guard, 0xA5, sizeof box.guard);
    as_filled = dfn2_model_io_descriptor_validate(&grid, d);
    printf("grid=%d default=%d input_frames=%d guard_after_default=%d "
           "validate_default=%d", made, filled, box.d.input_frames, written,
           as_filled);
    box.d.input_frames = 3;
    old_frames = dfn2_model_io_descriptor_validate(&grid, d);
    printf(" validate_three=%d guard_end=%d null=%d,%d", old_frames,
           guard_intact(), dfn2_model_io_descriptor_default(&grid, NULL),
           dfn2_model_io_descriptor_validate(&grid, NULL));
    /* No grid: refused, nothing is written past the descriptor, and what is
     * left in it is not a descriptor any grid accepts. */
    memset(&box, 0xA5, sizeof box);
    printf(" no_grid=%d,%d", dfn2_model_io_descriptor_default(NULL, d),
           dfn2_model_io_descriptor_validate(NULL, d));
    printf(" guard_no_grid=%d leftover=%d\n", guard_intact(),
           dfn2_model_io_descriptor_validate(&grid, d));
    return 0;
}
"""


def _abi_source():
    values = dfn_pairs.dfn2_spec_values()
    return _ABI_DRIVER.replace('@ORDER@', str(values['df_order']))


@pytest.fixture(scope='module')
def abi_driver(tmp_path_factory, audio_common_lib):
    work = tmp_path_factory.mktemp('dfn2_abi')
    source = work / 'abi_driver.c'
    source.write_text(_abi_source(), encoding='utf-8')
    exe = work / 'abi_driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), str(source),
         *[str(dfn_pairs.source_path(FAMILY, kind))
           for kind in ('model_io', 'process')],
         audio_common_lib, '-lm', '-o', str(exe)],
        check=True, capture_output=True, text=True)
    return exe


@pytest.mark.parametrize('frames', [1, 3])
def test_a_descriptor_of_72_bytes_is_safe_for_any_frame_count(abi_driver, frames):
    """default() and validate() touch exactly the 72 bytes of the descriptor; an
    input_frames of 3 passes a grid of one frame per call and is refused by a grid of
    three. A missing grid is refused and writes nothing past the descriptor."""
    done = subprocess.run([str(abi_driver), str(frames)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    fields = dict(item.split('=') for item in done.stdout.split())
    assert fields == {
        'grid': '0', 'default': '0', 'input_frames': str(frames + 2),
        'guard_after_default': '1', 'validate_default': '0',
        'validate_three': '0' if frames == 1 else '-1', 'guard_end': '1',
        'null': '-1,-1', 'no_grid': '-1,-1', 'guard_no_grid': '1',
        'leftover': '-1'}, done.stdout
