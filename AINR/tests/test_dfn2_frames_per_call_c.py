"""The DFN2 C host for N frames per call equals the one-frame host.

A build with ``-DDFN2_FRAMES_PER_CALL=N`` runs the accelerator once per N hops
on an ``N + 2`` frame window and composes the N returned frames in one burst.
The stand-in accelerator below is a pure function of (three-frame window
slice, state) -> (heads, next state), applied to the N slices in order the
way the exported graph is, so the N-frame host and the one-frame host see the
same arithmetic and must agree BYTE FOR BYTE:

  time / freq  the head sequence the accelerator produced, the final state of
               both recurrent tensors, and every output hop (the N-frame host
               trails by N - 1 hops: its hop h + N - 1 is the one-frame host's
               hop h), for a stream length that is not a multiple of N, in
               DFN2_IO_TIME and in DFN2_IO_FREQ with the single and the dual
               entry mixed
  reset        a reset in the middle of a call; both segments agree
  refuse       a call whose head or state comes back non-finite is refused,
               the state is zeroed and the N frames pass as the identity, the
               same as N one-frame refusals
  contract     readiness (L + N, then every N), the window rows, the
               descriptor's input_frames (N + 2), the latency accessor,
               element counts, and a pool of the one-frame build refused
  abi          the 72-byte descriptor of the one-frame header, filled and
               checked through the library of either N: nothing written or
               read past it, a one-frame descriptor refused by N > 1
  build        N outside 1..64 does not compile, a one-frame unit does not
               compile for N > 1, mixed-N objects do not link, and the
               Makefile validates and forwards DFN2_FRAMES_PER_CALL

A stand-in reading the wrong window slice fails the comparison, so it is not
vacuous.
"""

import subprocess

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
_SOURCES = ('prepost', 'process', 'model_io')
PAIRS = ((1, 1), (0, 0), (2, 4), (0, 1), (2, 0))
FRAMES = (2, 3, 5, 10)
REAL_HOPS = 37
_pair_id = dfn_pairs.pair_id

_DRIVER = r'''
#define DFN2_MULTI_FRAME_HOST
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define NF    DFN2_FRAMES_PER_CALL
#define LA    DFN2_MASK_LOOKAHEAD
#define DA    DFN2_DF_LOOKAHEAD
#define WF    DFN2_MODEL_WINDOW_FRAMES
#define E     DFN2_N_ERB
#define F     DFN2_DF_BINS
#define COEF1 (DFN2_DF_BINS * DFN2_DF_ORDER * 2)
#define ROWS  DFN2_MODEL_GRU_ROWS
#define HID   DFN2_MODEL_GRU_HIDDEN
#define CH    DFN2_MODEL_ENCODER_CHANNELS
#define PW    DFN2_MODEL_DF_PATHWAY_WINDOW
#ifndef STUB_SLICE_OFFSET
#define STUB_SLICE_OFFSET 0
#endif

/* ---- ERB matrices: the shipped border table, erb_inv a partition of unity */
static const int erb_borders[E] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
    50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
    325, 378, 440, 513
};
static float erb_fwd[DFN2_N_BINS][E];
static float erb_inv[E][DFN2_N_BINS];

static void build_erb(void) {
    int segment = 0, k;
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
static void stub_frame(const float (*erb)[E], const float (*spec0)[F],
                       const float (*spec1)[F], float (*gru)[HID],
                       float (*hist)[PW][F], float *mask, float *coefs,
                       float *alpha) {
    float e = 0.0f, sp = 0.0f;
    int t, b, r, j, ch, k;
    for (t = 0; t < 3; ++t)
        for (b = 0; b < E; ++b)
            e += erb[t][b] * (0.01f * (float)(t + 1) + 0.001f * (float)b);
    for (t = 0; t < 3; ++t)
        for (k = 0; k < F; ++k)
            sp += spec0[t][k] * 0.002f * (float)(t + 1) - spec1[t][k] * 0.001f;
    for (r = 0; r < ROWS; ++r)
        for (j = 0; j < HID; ++j)
            gru[r][j] = tanhf(0.9f * gru[r][j] + 0.05f * e +
                              0.03f * (float)(r + 1) * sp + 0.001f * (float)j);
    for (ch = 0; ch < CH; ++ch) {
        memmove(hist[ch][0], hist[ch][1], (PW - 1) * sizeof(hist[ch][0]));
        for (k = 0; k < F; ++k)
            hist[ch][PW - 1][k] = tanhf(
                0.1f * spec0[1][k] * (float)(ch + 1) / (float)CH +
                0.5f * gru[ch % ROWS][k] + 0.2f * hist[ch][0][k]);
    }
    for (b = 0; b < E; ++b)
        mask[b] = 1.0f / (1.0f + expf(-(erb[1][b] + gru[1][b])));
    for (j = 0; j < COEF1; ++j)
        coefs[j] = 0.1f * tanhf(hist[j % CH][(j / CH) % PW][j % F] +
                                gru[3][j % HID]);
    *alpha = 1.0f / (1.0f + expf(-gru[4][0]));
}

typedef struct {
    long long next_head;          /* head index of the call's first frame */
    long long poison_lo, poison_hi;
    int poison_kind;              /* 1: a NaN head, 2: a NaN state element */
    int calls;
    int bad_views;
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
        (const float *)in->gru_hidden != &out->gru_hidden_next[0][0] ||
        (const float *)in->df_convp_window !=
            &out->df_convp_window_next[0][0][0]) {
        ++s->bad_views;
        return -1;
    }
    for (i = 0; i < NF; ++i) {
        const int at = (i + STUB_SLICE_OFFSET) % NF;
        long long head = s->next_head + i;
        stub_frame(in->erb_window + at,
                   (const float (*)[F])in->spec_window[0][at],
                   (const float (*)[F])in->spec_window[1][at],
                   out->gru_hidden_next, out->df_convp_window_next,
                   out->erb_mask + (size_t)i * E,
                   out->coefs + (size_t)i * COEF1, out->alpha + i);
        rec('H', (int)head, 0, head, out->erb_mask + (size_t)i * E, E);
        rec('C', (int)head, 0, head, out->coefs + (size_t)i * COEF1, COEF1);
        rec('A', (int)head, 0, head, out->alpha + i, 1);
        if (head >= s->poison_lo && head < s->poison_hi) poisoned = 1;
    }
    if (poisoned && s->poison_kind == 1) out->alpha[NF - 1] = (float)NAN;
    if (poisoned && s->poison_kind == 2)
        out->gru_hidden_next[ROWS - 1][HID - 1] = (float)NAN;
    s->next_head += NF;
    return 0;
}

static float noise(long long hop, int i) {
    uint32_t x = (uint32_t)(hop * 2654435761u) ^ (uint32_t)(i * 40503u + 17u);
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u;
    x ^= x >> 16;
    return ((float)(x & 0xffffu) / 65535.0f - 0.5f) * 0.2f;
}

static DFN2Prepost *make(int io_mode, FftHandle *fft) {
    DFN2PrepostConfig cfg;
    if (dfn2_prepost_config_defaults(&cfg, io_mode) != 0) return NULL;
    cfg.fft = io_mode == DFN2_IO_TIME ? fft : NULL;
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    return dfn2_prepost_create(&cfg);
}

static DFN2PrepostInputs views_in;
static DFN2PrepostOutputs views_out;

/* time, freq, reset, refuse: one stream through dfn2_model_run_frame. */
static int run_stream(const char *mode, int real, int hops, int reset_hop,
                      long long plo, long long phi, int pkind) {
    const int freq = strcmp(mode, "freq") == 0;
    FftHandle *fft = fft_create(DFN2_N_FFT);
    DFN2Prepost *p;
    DFN2ModelIoDescriptor descriptor;
    DFN2Model model;
    Stub stub;
    static float in_hop[DFN2_HOP_LEN], out_hop[DFN2_HOP_LEN];
    static float est_re[DFN2_N_BINS], est_im[DFN2_N_BINS];
    static float app_re[DFN2_N_BINS], app_im[DFN2_N_BINS];
    static float out_spec[2 * DFN2_N_BINS];
    int h, k;

    CHECK(fft != NULL);
    p = make(freq ? DFN2_IO_FREQ : DFN2_IO_TIME, fft);
    CHECK(p != NULL);
    CHECK(dfn2_model_io_descriptor_default(&descriptor) == 0);
    memset(&stub, 0, sizeof stub);
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
            rec('R', h, 0, 0, NULL, 0);
        }
        if (freq) {
            for (k = 0; k < DFN2_N_BINS; ++k) {
                est_re[k] = live ? noise(h, 2 * k) : 0.0f;
                est_im[k] = live ? noise(h, 2 * k + 1) : 0.0f;
                app_re[k] = 0.5f * est_re[k] + (live ? noise(h + 7, k) : 0.0f);
                app_im[k] = 0.5f * est_im[k];
            }
            rc = h % 3 == 0
                ? dfn2_prepost_pre_process_freq_dual(p, est_re, est_im,
                                                     app_re, app_im)
                : dfn2_prepost_pre_process_freq(p, est_re, est_im);
        } else {
            for (k = 0; k < DFN2_HOP_LEN; ++k)
                in_hop[k] = live ? noise(h, k) : 0.0f;
            rc = dfn2_prepost_pre_process(p, in_hop);
        }
        CHECK(rc == 0 || rc == 1);
        if (rc == 1) CHECK(dfn2_model_run_frame(&model, p) >= 0);
        if (freq) {
            CHECK(dfn2_prepost_post_process_freq(
                      p, out_spec, out_spec + DFN2_N_BINS, &written) == 0);
        } else {
            CHECK(dfn2_prepost_post_process(p, out_hop, &written) == 0);
        }
        (void)dfn2_prepost_output_frame_index(p, &frame);
        rec('O', h, (written ? 1 : 0) | (rc << 1), frame,
            freq ? out_spec : out_hop, freq ? 2 * DFN2_N_BINS : DFN2_HOP_LEN);
    }
    CHECK(stub.bad_views == 0);
    /* The final recurrent state, read through the views of the next open
     * frame: zero hops are pushed (they touch only the feature windows and the
     * staging) until one opens. */
    {
        int rc;
        for (;;) {
            if (freq) {
                memset(est_re, 0, sizeof est_re);
                memset(est_im, 0, sizeof est_im);
                rc = dfn2_prepost_pre_process_freq(p, est_re, est_im);
            } else {
                memset(in_hop, 0, sizeof in_hop);
                rc = dfn2_prepost_pre_process(p, in_hop);
            }
            CHECK(rc == 0 || rc == 1);
            if (rc == 1) break;
            if (freq)
                CHECK(dfn2_prepost_post_process_freq(
                          p, out_spec, out_spec + DFN2_N_BINS, NULL) == 0);
            else
                CHECK(dfn2_prepost_post_process(p, out_hop, NULL) == 0);
        }
        CHECK(dfn2_prepost_frame_inputs(p, &views_in, &views_out) == 0);
        rec('S', 0, 0, 0, &views_in.gru_hidden[0][0],
            (int)views_in.gru_hidden_elements);
        rec('W', 0, 0, 0, &views_in.df_convp_window[0][0][0],
            (int)views_in.df_convp_window_elements);
    }
    rec('G', (int)dfn2_prepost_state_generation(p), 0, 0, NULL, 0);
    printf("calls=%d generation=%u\n", stub.calls,
           dfn2_prepost_state_generation(p));
    dfn2_prepost_destroy(p);
    fft_destroy(fft);
    return 0;
}

static int case_contract(void) {
    FftHandle *fft = fft_create(DFN2_N_FFT);
    DFN2ModelIoDescriptor d, bad;
    DFN2ModelIOState io;
    DFN2PrepostMemReq time_req, freq_req;
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    DFN2Model model;
    Stub stub;
    static float erb[E], spec[2][F], in_hop[DFN2_HOP_LEN], out_hop[DFN2_HOP_LEN];
    int f, s, b, h, written, first_written = -1;

    CHECK(fft != NULL);
    CHECK(dfn2_model_io_descriptor_default(&d) == 0);
    CHECK(d.input_frames == WF && d.input_frames - 2 == NF);
    CHECK(dfn2_model_io_descriptor_validate(&d) == 0);
    bad = d; ++bad.input_frames;
    CHECK(dfn2_model_io_descriptor_validate(&bad) == -1);
    bad = d; bad.input_frames = DFN2_MODEL_INPUT_FRAMES + (NF == 1);
    CHECK(dfn2_model_io_descriptor_validate(&bad) == -1);

    /* Readiness, slots and the window rows, frame by frame. */
    dfn2_model_io_init(&io);
    CHECK(dfn2_model_io_frame_slot(&io) == -1);
    CHECK(dfn2_model_io_frame_slot(NULL) == -1);
    for (f = 0; f < LA + 6 * NF + 3; ++f) {
        int ready, want;
        for (b = 0; b < E; ++b) erb[b] = (float)(1000 * (f + 1) + b);
        for (b = 0; b < F; ++b) {
            spec[0][b] = (float)(1000 * (f + 1) + b) + 0.25f;
            spec[1][b] = -(float)(1000 * (f + 1) + b);
        }
        ready = dfn2_model_io_push_features(&io, erb, (const float (*)[F])spec);
        want = f + 1 >= LA + NF && (f + 1 - LA) % NF == 0;
        CHECK(ready == want);
        CHECK(dfn2_model_io_frame_slot(&io) == (f < LA ? -1 : (f - LA) % NF));
        for (s = 0; s < WF; ++s) {
            const int frame = f - (WF - 1) + s;
            for (b = 0; b < E; ++b)
                CHECK(io.erb_window[s][b] ==
                      (frame < 0 ? 0.0f : (float)(1000 * (frame + 1) + b)));
            CHECK(io.spec_window[0][s][3] ==
                  (frame < 0 ? 0.0f : (float)(1000 * (frame + 1) + 3) + 0.25f));
            CHECK(io.spec_window[1][s][F - 1] ==
                  (frame < 0 ? 0.0f : -(float)(1000 * (frame + 1) + F - 1)));
        }
    }

    /* The class: pre_process readiness, views, latency, first output. */
    CHECK(dfn2_prepost_config_defaults(&cfg, DFN2_IO_TIME) == 0);
    cfg.fft = fft;
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    CHECK(dfn2_prepost_get_mem_size(&cfg, &time_req) == 0);
    p = dfn2_prepost_create(&cfg);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == LA + DA + NF - 1);
    CHECK(DFN2_PREPOST_LATENCY_FRAMES == LA + DA + NF - 1);
    memset(&stub, 0, sizeof stub);
    stub.poison_lo = stub.poison_hi = -1;
    model.user = &stub;
    model.infer = stub_infer;
    model.reset = NULL;
    bad = d;
    bad.input_frames = DFN2_MODEL_INPUT_FRAMES + (NF == 1);
    for (h = 0; h < LA + DA + 6 * NF + 4; ++h) {
        int rc, want;
        for (b = 0; b < DFN2_HOP_LEN; ++b) in_hop[b] = noise(h, b);
        rc = dfn2_prepost_pre_process(p, in_hop);
        want = h + 1 >= LA + NF && (h + 1 - LA) % NF == 0;
        CHECK(rc == want);
        if (rc == 1) {
            int calls = stub.calls;
            /* Another N's descriptor is refused before infer() runs. */
            model.io_descriptor = &bad;
            CHECK(dfn2_model_run_frame(&model, p) == -1);
            CHECK(stub.calls == calls);
            CHECK(dfn2_prepost_frame_inputs(p, &views_in, &views_out) == 0);
            CHECK(views_in.erb_window_elements ==
                  (size_t)DFN2_PREPOST_ERB_WINDOW_ELEMENTS);
            CHECK(DFN2_PREPOST_ERB_WINDOW_ELEMENTS == WF * E);
            CHECK(DFN2_PREPOST_SPEC_WINDOW_ELEMENTS == 2 * WF * F);
            CHECK(views_out.alpha_elements == (size_t)NF);
            CHECK(isnan(views_out.alpha[NF - 1]));
            model.io_descriptor = &d;
            CHECK(dfn2_model_run_frame(&model, p) == 1);
            CHECK(stub.calls == calls + 1);
        }
        CHECK(dfn2_prepost_post_process(p, out_hop, &written) == 0);
        if (written && first_written < 0) first_written = h;
    }
    CHECK(first_written == LA + DA + NF - 1);
    dfn2_prepost_destroy(p);

    CHECK(dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) == 0);
    cfg.erb_fwd = &erb_fwd[0][0];
    cfg.erb_inv = &erb_inv[0][0];
    CHECK(dfn2_prepost_get_mem_size(&cfg, &freq_req) == 0);
    printf("req time=%llu:%08x freq=%llu:%08x\n",
           (unsigned long long)time_req.bytes,
           (unsigned)time_req.build_flags_hash,
           (unsigned long long)freq_req.bytes,
           (unsigned)freq_req.build_flags_hash);
    fft_destroy(fft);
    return 0;
}

/* A FREQ pool descriptor from another build is refused; this build's own is
 * accepted. argv: bytes hash. */
static int case_req_check(unsigned long long bytes, unsigned hash) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, other;
    size_t room;
    void *pool;
    CHECK(dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) == 0);
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

int main(int argc, char **argv) {
    build_erb();
    if (argc >= 2 && strcmp(argv[1], "contract") == 0) return case_contract();
    if (argc == 4 && strcmp(argv[1], "req_check") == 0)
        return case_req_check(strtoull(argv[2], NULL, 10),
                              (unsigned)strtoul(argv[3], NULL, 16));
    if (argc == 9) {
        int rc;
        log_file = fopen(argv[2], "wb");
        if (!log_file) return 2;
        rc = run_stream(argv[1], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                        atoll(argv[6]), atoll(argv[7]), atoi(argv[8]));
        fclose(log_file);
        return rc;
    }
    return 2;
}
'''

_REC = np.dtype([('tag', '<i4'), ('a', '<i4'), ('b', '<i4'),
                 ('count', '<i4'), ('c', '<i8')])


def _defines(pair, frames, extra=()):
    flags = list(dfn_pairs.defines(FAMILY, pair))
    if frames != 1:
        flags.append('-DDFN2_FRAMES_PER_CALL=%d' % frames)
    return flags + list(extra)


def _build(work, audio_common_lib, pair, frames, extra=()):
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    exe = work / 'driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), *_defines(pair, frames, extra),
         str(source),
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
    built = {}

    def get(pair, frames, extra=()):
        key = (pair, frames, tuple(extra))
        if key not in built:
            work = tmp_path_factory.mktemp(
                'dfn2_n%d_%s' % (frames, _pair_id(pair)))
            built[key] = _build(work, audio_common_lib, pair, frames, extra)
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


def _stream(driver, tmp_path, mode, hops, reset=-1, poison=(-1, -1), kind=0,
            name='log'):
    log = tmp_path / ('%s.bin' % name)
    done = subprocess.run(
        [str(driver), mode, str(log), str(REAL_HOPS), str(hops), str(reset),
         str(poison[0]), str(poison[1]), str(kind)],
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


@pytest.mark.parametrize('mode', ['time', 'freq'])
@pytest.mark.parametrize('frames', FRAMES)
@pytest.mark.parametrize('pair', PAIRS, ids=_pair_id)
def test_chunk_host_equals_the_frame_host(drivers, tmp_path, pair, frames,
                                          mode):
    hops = _hops(pair, frames)
    one, _ = _stream(drivers(pair, 1), tmp_path, mode, hops, name='one')
    many, out = _stream(drivers(pair, frames), tmp_path, mode, hops,
                        name='many')
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
@pytest.mark.parametrize('pair', [(1, 1), (2, 4), (0, 1)], ids=_pair_id)
def test_a_reset_in_the_middle_of_a_call(drivers, tmp_path, pair, frames):
    """Reset with a call half full: the pending frames are dropped with the
    rest of the stream state, and each segment agrees with the one-frame host
    on everything the N-frame host had emitted."""
    reset = pair[0] + 2 * frames + 1
    hops = reset + _hops(pair, frames, real=12)
    one, _ = _stream(drivers(pair, 1), tmp_path, 'time', hops, reset=reset,
                     name='one')
    many, out = _stream(drivers(pair, frames), tmp_path, 'time', hops,
                        reset=reset, name='many')
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
    one, out_one = _stream(drivers(pair, 1), tmp_path, 'time', hops,
                           poison=poison, kind=kind, name='one')
    many, out_many = _stream(drivers(pair, frames), tmp_path, 'time', hops,
                             poison=poison, kind=kind, name='many')
    one, many = _segments(one)[0], _segments(many)[0]
    _compare(one, many, frames, poison=poison)
    assert many['final'] == one['final']
    assert 'generation=2' in out_many
    assert 'generation=%d' % (1 + frames) in out_one
    # The identity reached the output: the refused frames differ from a run
    # without the refusal.
    clean, _ = _stream(drivers(pair, frames), tmp_path, 'time', hops,
                       name='clean')
    clean = _segments(clean)[0]
    assert [e[:3] for e in clean['out']] != [e[:3] for e in many['out']]


def test_a_stand_in_on_the_wrong_slice_fails_the_comparison(drivers,
                                                            tmp_path):
    """The comparison reads the window slices: an accelerator that takes
    slice i + 1 for head i diverges from the one-frame host."""
    pair, frames = (1, 1), 3
    hops = _hops(pair, frames)
    one, _ = _stream(drivers(pair, 1), tmp_path, 'time', hops, name='one')
    many, _ = _stream(drivers(pair, frames, ('-DSTUB_SLICE_OFFSET=1',)),
                      tmp_path, 'time', hops, name='many')
    with pytest.raises(AssertionError):
        _compare(_segments(one)[0], _segments(many)[0], frames)


def _contract(driver):
    done = subprocess.run([str(driver), 'contract'], capture_output=True,
                          text=True)
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
    _contract(drivers(pair, frames))


@pytest.mark.parametrize('frames', FRAMES)
def test_the_pool_of_another_frame_count_is_refused(drivers, frames):
    pair = (1, 1)
    single = _contract(drivers(pair, 1))
    chunked = _contract(drivers(pair, frames))
    for mode in ('time', 'freq'):
        assert chunked[mode][1] != single[mode][1]
        assert chunked[mode][0] > single[mode][0]
    bytes_, hash_ = single['freq']
    done = subprocess.run(
        [str(drivers(pair, frames)), 'req_check', str(bytes_), hash_],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)


# ---------------------------------------------------------------------------
# build-time guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('std', ['c11', 'gnu99', 'c++11'])
@pytest.mark.parametrize('frames', [0, -1, 65])
def test_frames_per_call_outside_the_range_does_not_compile(
        tmp_path, std, frames):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    done = dfn_pairs.header_compiles(
        FAMILY, (1, 1), std, tmp_path,
        extra=('-DDFN2_FRAMES_PER_CALL=%d' % frames,
               '-DDFN2_MULTI_FRAME_HOST'))
    assert done.returncode != 0
    assert 'DFN2_FRAMES_PER_CALL must be in [1, 64]' in done.stderr


@pytest.mark.parametrize('std', ['c11', 'gnu99', 'c++11'])
def test_a_one_frame_unit_does_not_compile_for_more_frames(tmp_path, std):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    refused = dfn_pairs.header_compiles(
        FAMILY, (1, 1), std, tmp_path, extra=('-DDFN2_FRAMES_PER_CALL=4',))
    assert refused.returncode != 0
    assert 'DFN2_MULTI_FRAME_HOST' in refused.stderr
    for extra in (('-DDFN2_FRAMES_PER_CALL=4', '-DDFN2_MULTI_FRAME_HOST'),
                  ('-DDFN2_FRAMES_PER_CALL=1',), ()):
        done = dfn_pairs.header_compiles(FAMILY, (1, 1), std, tmp_path,
                                         extra=extra)
        assert done.returncode == 0, (extra, done.stderr)


def test_objects_of_different_frame_counts_do_not_link(tmp_path,
                                                       audio_common_lib):
    """dfn2_prepost.o of one N against dfn2_model_io.o of another leaves the
    prepost object's frames-per-call symbol undefined."""
    objects = []
    for kind, frames in (('process', 1), ('model_io', 4), ('prepost', 1)):
        obj = tmp_path / ('%s.o' % kind)
        subprocess.run(
            ['cc', *dfn_pairs.house_cflags(), *_defines((1, 1), frames),
             '-c', str(dfn_pairs.source_path(FAMILY, kind)), '-o', str(obj)],
            check=True, capture_output=True)
        objects.append(str(obj))
    main = tmp_path / 'main.c'
    main.write_text('#include "DeepFilterNet2/dfn2_prepost.h"\n'
                    'int main(void) { DFN2PrepostMemReq r;\n'
                    '  return dfn2_prepost_get_mem_size(NULL, &r); }\n')
    linked = subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), str(main), *objects,
         audio_common_lib, '-lm', '-o', str(tmp_path / 'mixed')],
        capture_output=True, text=True)
    assert linked.returncode != 0
    assert 'dfn2_frames_per_call1' in linked.stderr
    same = subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), '-DDFN2_FRAMES_PER_CALL=4',
         '-DDFN2_MULTI_FRAME_HOST', str(main),
         *[str(dfn_pairs.source_path(FAMILY, kind)) for kind in _SOURCES],
         audio_common_lib, '-lm', '-o', str(tmp_path / 'same')],
        capture_output=True, text=True)
    assert same.returncode == 0, same.stderr


# A unit compiled against the one-frame header: the descriptor exactly as that
# header declared it (18 fields, 72 bytes) with a guard band behind it, passed
# to the library through the public prototypes only.
_ABI_DRIVER = r'''
#include <stddef.h>
#include <stdio.h>
#include <string.h>

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
int dfn2_model_io_descriptor_default(DFN2ModelIoDescriptor *descriptor);
int dfn2_model_io_descriptor_validate(
    const DFN2ModelIoDescriptor *descriptor);

typedef struct {
    OldDescriptor d;
    unsigned char guard[16];
} Box;
static Box box;

_Static_assert(sizeof(OldDescriptor) == 72 &&
                   offsetof(Box, guard) == 72,
               "the guard band starts right after the old descriptor");

static int guard_intact(void)
{
    size_t i;
    for (i = 0; i < sizeof box.guard; ++i)
        if (box.guard[i] != 0xA5) return 0;
    return 1;
}

int main(void)
{
    DFN2ModelIoDescriptor *d = (DFN2ModelIoDescriptor *)&box.d;
    int written, filled, as_filled, old_frames;
    memset(&box, 0xA5, sizeof box);
    filled = dfn2_model_io_descriptor_default(d);
    written = guard_intact();
    memset(box.guard, 0xA5, sizeof box.guard);
    as_filled = dfn2_model_io_descriptor_validate(d);
    printf("default=%d input_frames=%d guard_after_default=%d "
           "validate_default=%d", filled, box.d.input_frames, written,
           as_filled);
    box.d.input_frames = 3;
    old_frames = dfn2_model_io_descriptor_validate(d);
    printf(" validate_three=%d guard_end=%d null=%d,%d\n", old_frames,
           guard_intact(), dfn2_model_io_descriptor_default(NULL),
           dfn2_model_io_descriptor_validate(NULL));
    return 0;
}
'''


@pytest.mark.parametrize('frames', [1, 3])
def test_a_descriptor_of_the_one_frame_header_is_safe_for_any_frame_count(
        tmp_path, audio_common_lib, frames):
    """default() and validate() touch exactly the 72 bytes of the one-frame
    header's descriptor; its input_frames of 3 passes an N = 1 library and is
    refused by an N = 3 one."""
    source = tmp_path / 'old_adapter.c'
    source.write_text(_ABI_DRIVER, encoding='utf-8')
    exe = tmp_path / 'old_adapter'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), *_defines((1, 1), frames), str(source),
         *[str(dfn_pairs.source_path(FAMILY, kind))
           for kind in ('model_io', 'process')],
         audio_common_lib, '-lm', '-o', str(exe)],
        check=True, capture_output=True, text=True)
    done = subprocess.run([str(exe)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    fields = dict(item.split('=') for item in done.stdout.split())
    assert fields == {
        'default': '0', 'input_frames': str(frames + 2),
        'guard_after_default': '1', 'validate_default': '0',
        'validate_three': '0' if frames == 1 else '-1', 'guard_end': '1',
        'null': '-1,-1'}, done.stdout


@pytest.mark.parametrize('value', ['0', '65', '2 3', 'x', '04', '-1'])
def test_the_makefile_refuses_a_frame_count_outside_the_range(value):
    if dfn_pairs.require_toolchain():
        pytest.skip('make unavailable')
    done = subprocess.run(
        ['make', '-s', '-C', str(dfn_pairs.AINR),
         'DFN2_FRAMES_PER_CALL=%s' % value, 'print-cflags'],
        capture_output=True, text=True)
    assert done.returncode != 0
    assert 'DFN2_FRAMES_PER_CALL must be one plain decimal in 1..64' in (
        done.stderr)


def test_the_makefile_forwards_the_frame_count_only_when_not_one():
    if dfn_pairs.require_toolchain():
        pytest.skip('make unavailable')

    def cflags(*args):
        return subprocess.run(
            ['make', '-s', '-C', str(dfn_pairs.AINR), *args, 'print-cflags'],
            check=True, capture_output=True, text=True).stdout
    default = cflags()
    assert 'DFN2_FRAMES_PER_CALL' not in default
    assert cflags('DFN2_FRAMES_PER_CALL=1') == default
    assert '-DDFN2_FRAMES_PER_CALL=10' in cflags('DFN2_FRAMES_PER_CALL=10')
