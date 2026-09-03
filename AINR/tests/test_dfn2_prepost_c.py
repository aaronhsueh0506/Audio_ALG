"""Equivalence and contract gates for the DeepFilterNet2 pre/post class.

``dfn2_prepost.c`` owns no signal processing: it composes the two already
parity-tested translation units (``dfn2_process.c`` for the center=False
sqrt-Hann STFT/WOLA, the ERB/complex feature front end and the compose stage,
``dfn2_model_io.c`` for the graph feature windows and the recurrent caches)
behind one object.  So the gate that matters is not a tolerance against a
Python reference -- the parity of the parts is covered by
test_python_c_prepost_parity.py and tests/test_c_prepost.c -- but that the
class reproduces the hand-composed path SAMPLE FOR SAMPLE.  Every comparison
here is therefore ``memcmp`` on fp32, never a tolerance: the class is only
allowed to reorder nothing.

Cases, all in ONE driver selected by argv[1] so audio_common and the three
model TUs are compiled once for the module:

  equiv     DFN2_IO_TIME vs analysis + features + model-I/O + compose +
            synthesis, hand-composed.  The reference deliberately uses ONLY
            the two composed TUs, never dfn2_prepost.c, so the comparison is
            not circular.  Run twice: attenuation limit off and at -20 dB.
  freq      DFN2_IO_FREQ driven by the caller's own dfn2_analysis on a
            separate DFN2State and re-synthesised by the caller, vs IO_TIME
  freqpool  the FREQ pool is smaller, and the stale-pool gate both accepts
            the matching MemReq and refuses a perturbed one
  lifecycle _create/_destroy against _init on a caller pool
  reject    _get_mem_size / _init / stage-call reject-first validation
  reset     _reset really clears state (a re-run reproduces the first run)
  skip      _frame_skip is the exact identity and does NOT step the state
  guard     the four frame-state-machine gates the contract block promises
  txn       "on failure NOTHING moves, the frame stays open", proved for
            every one of the seven accelerator output tensors

The accelerator stand-in is deterministic and input-dependent, so the two
feature windows, the three GRU hidden tensors and the deep-filter pathway
history genuinely influence later frames -- a class that dropped a state
advance would diverge rather than agree.
"""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DFN_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'DeepFilterNet2')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audio_common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_SOURCES = ('dfn2_prepost.c', 'dfn2_process.c', 'dfn2_model_io.c')

_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define HOPS        400
#define SHORT_HOPS  40
#define LOOKAHEAD   (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

/* skip: frames driven purely on frame_skip, then four more for the freeze. */
#define SKIP_FRAMES 24
#define SKIP_TOTAL  (SKIP_FRAMES + 4)

/* txn: K normal hops, the poisoned hop, then enough hops for the divergence
 * against the "committed at K" reference to be unmissable. */
#define TXN_PRE   20
#define TXN_POST  40
#define TXN_HOPS  (TXN_PRE + 1 + TXN_POST)

#define N_OUT_TENSORS 7

/* ---- synthetic ERB matrices ------------------------------------------
 * Built exactly the way tests/test_c_prepost.c's dfn_test_build_erb() does,
 * from the shipped 48 kHz / 1024 / 32 border table.  erb_inv is a partition
 * of unity per bin, which is what makes frame_skip an EXACT identity rather
 * than an approximate one -- main() asserts that property before any case
 * runs, in the same accumulation order df_common_expand_mask() uses. */
static const int erb_borders[DFN2_N_ERB] = {
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
    50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
    325, 378, 440, 513
};
static float erb_fwd[DFN2_N_BINS][DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB][DFN2_N_BINS];

static void build_erb(void) {
    int segment = 0;
    int k;
    memset(erb_fwd, 0, sizeof erb_fwd);
    memset(erb_inv, 0, sizeof erb_inv);
    for (k = 0; k < DFN2_N_BINS; ++k) {
        int lo, hi, width, offset;
        float right, left, fleft, fright;
        while (segment + 1 < DFN2_N_ERB - 1 && k >= erb_borders[segment + 1])
            ++segment;
        lo = erb_borders[segment];
        hi = erb_borders[segment + 1];
        width = hi - lo;
        offset = k - lo;
        right = (float)offset / (float)width;
        left = 1.0f - right;
        fleft = left;
        fright = right;
        if (segment == 0) fleft *= 2.0f;
        if (segment + 1 == DFN2_N_ERB - 1) fright *= 2.0f;
        erb_fwd[k][segment] = fleft;
        erb_fwd[k][segment + 1] = fright;
        erb_inv[segment][k] = left;
        erb_inv[segment + 1][k] = right;
    }
}

static int erb_inv_is_partition_of_unity(void) {
    int k, b;
    for (k = 0; k < DFN2_N_BINS; ++k) {
        float sum = 0.0f;
        for (b = 0; b < DFN2_N_ERB; ++b) sum += erb_inv[b][k];
        if (sum != 1.0f) return 0;
    }
    return 1;
}

/* ---- deterministic accelerator stand-in -------------------------------
 * Split into an accumulate half and a fill half so the SAME arithmetic can
 * be driven either through the class's published views or straight off a
 * hand-composed DFN2ModelIOState.  Every element of all seven outputs is
 * written, as the commit contract demands, and every value depends on both
 * feature windows and on all four recurrent tensors, so a dropped state
 * advance shows up as a divergence rather than as agreement. */
static double fake_acc(const float *erb_window, const float *spec_window,
                       const float *encoder_hidden, const float *erb_hidden,
                       const float *df_hidden, const float *convp) {
    double acc = 0.0;
    size_t i;
    for (i = 0; i < (size_t)DFN2_PREPOST_ERB_WINDOW_ELEMENTS; ++i)
        acc += erb_window[i];
    for (i = 0; i < (size_t)DFN2_PREPOST_SPEC_WINDOW_ELEMENTS; ++i)
        acc += 0.5 * spec_window[i];
    for (i = 0; i < (size_t)DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS; ++i)
        acc += 1e-3 * encoder_hidden[i];
    for (i = 0; i < (size_t)DFN2_PREPOST_ERB_HIDDEN_ELEMENTS; ++i)
        acc += 1e-3 * erb_hidden[i];
    for (i = 0; i < (size_t)DFN2_PREPOST_DF_HIDDEN_ELEMENTS; ++i)
        acc += 1e-3 * df_hidden[i];
    for (i = 0; i < (size_t)DFN2_PREPOST_CONVP_HISTORY_ELEMENTS; ++i)
        acc += 1e-4 * convp[i];
    return acc;
}

static void fake_fill(double acc, float *erb_mask, float *coefs, float *alpha,
                      float *encoder_next, float *erb_next, float *df_next,
                      float *convp_next) {
    size_t i;
    /* Kept inside (0,1) so it is a plausible sigmoid mask, not a value the
     * compose stage would clamp. */
    for (i = 0; i < (size_t)DFN2_PREPOST_ERB_MASK_ELEMENTS; ++i)
        erb_mask[i] = (float)(0.5 + 0.4 * sin(acc + (double)i));
    for (i = 0; i < (size_t)DFN2_PREPOST_COEFS_ELEMENTS; ++i)
        coefs[i] = (float)(0.05 * cos(acc + 1.0 + (double)i));
    for (i = 0; i < (size_t)DFN2_PREPOST_ALPHA_ELEMENTS; ++i)
        alpha[i] = (float)(0.5 + 0.3 * sin(acc + 2.0));
    for (i = 0; i < (size_t)DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS; ++i)
        encoder_next[i] = (float)(0.5 * sin(acc + 3.0 + (double)i));
    for (i = 0; i < (size_t)DFN2_PREPOST_ERB_HIDDEN_ELEMENTS; ++i)
        erb_next[i] = (float)(0.5 * sin(acc + 4.0 + (double)i));
    for (i = 0; i < (size_t)DFN2_PREPOST_DF_HIDDEN_ELEMENTS; ++i)
        df_next[i] = (float)(0.5 * sin(acc + 5.0 + (double)i));
    for (i = 0; i < (size_t)DFN2_PREPOST_CONVP_HISTORY_ELEMENTS; ++i)
        convp_next[i] = (float)(0.1 * cos(acc + 5.0 + (double)i));
}

static void fake_run(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    double acc = fake_acc(&in->erb_window[0][0],
                          &in->spec_window[0][0][0],
                          &in->encoder_gru_hidden[0][0],
                          &in->erb_gru_hidden[0][0],
                          &in->df_gru_hidden[0][0],
                          &in->df_convp_history[0][0][0]);
    fake_fill(acc, out->erb_mask, out->coefs, out->alpha,
              &out->encoder_gru_hidden_next[0][0],
              &out->erb_gru_hidden_next[0][0],
              &out->df_gru_hidden_next[0][0],
              &out->df_convp_history_next[0][0][0]);
}

/* The same seven tensors, laid out for the hand-composed reference path. */
typedef struct {
    float erb_mask[DFN2_PREPOST_ERB_MASK_ELEMENTS];
    float coefs[DFN2_PREPOST_COEFS_ELEMENTS];
    float alpha[DFN2_PREPOST_ALPHA_ELEMENTS];
    float encoder_next[DFN2_MODEL_ENCODER_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN];
    float erb_next[DFN2_MODEL_ERB_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN];
    float df_next[DFN2_MODEL_DF_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN];
    float convp_next[DFN2_MODEL_ENCODER_CHANNELS]
                    [DFN2_MODEL_DF_PATHWAY_HISTORY][DFN2_DF_BINS];
} RefHeads;

static void fake_run_reference(const DFN2ModelIOState *io, RefHeads *heads) {
    double acc = fake_acc(&io->erb_window[0][0],
                          &io->spec_window[0][0][0],
                          &io->encoder_gru_hidden[0][0],
                          &io->erb_gru_hidden[0][0],
                          &io->df_gru_hidden[0][0],
                          &io->df_convp_history[0][0][0]);
    fake_fill(acc, heads->erb_mask, heads->coefs, heads->alpha,
              &heads->encoder_next[0][0], &heads->erb_next[0][0],
              &heads->df_next[0][0], &heads->convp_next[0][0][0]);
}

/* ---- small helpers ---------------------------------------------------- */

static int identical(const float *a, const float *b, size_t count) {
    return memcmp(a, b, count * sizeof(float)) == 0;
}

static int any_nonzero(const float *v, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) if (v[i] != 0.0f) return 1;
    return 0;
}

static int all_zero(const float *v, size_t count) {
    return !any_nonzero(v, count);
}

static int all_nan(const float *v, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) if (!isnan(v[i])) return 0;
    return 1;
}

static int outputs_all_nan(const DFN2PrepostOutputs *out) {
    return all_nan(out->erb_mask, out->erb_mask_elements) &&
           all_nan(out->coefs, out->coefs_elements) &&
           all_nan(out->alpha, out->alpha_elements) &&
           all_nan(&out->encoder_gru_hidden_next[0][0],
                   out->encoder_gru_hidden_elements) &&
           all_nan(&out->erb_gru_hidden_next[0][0],
                   out->erb_gru_hidden_elements) &&
           all_nan(&out->df_gru_hidden_next[0][0],
                   out->df_gru_hidden_elements) &&
           all_nan(&out->df_convp_history_next[0][0][0],
                   out->df_convp_history_elements);
}

/* One of the seven accelerator output tensors, by index, for the txn sweep. */
static float *output_tensor(DFN2PrepostOutputs *out, int which,
                            size_t *count) {
    switch (which) {
    case 0: *count = out->erb_mask_elements; return out->erb_mask;
    case 1: *count = out->coefs_elements;    return out->coefs;
    case 2: *count = out->alpha_elements;    return out->alpha;
    case 3: *count = out->encoder_gru_hidden_elements;
            return &out->encoder_gru_hidden_next[0][0];
    case 4: *count = out->erb_gru_hidden_elements;
            return &out->erb_gru_hidden_next[0][0];
    case 5: *count = out->df_gru_hidden_elements;
            return &out->df_gru_hidden_next[0][0];
    default: *count = out->df_convp_history_elements;
            return &out->df_convp_history_next[0][0][0];
    }
}

static void fill_pcm(float *pcm, int samples) {
    unsigned state = 12345u;
    int i;
    for (i = 0; i < samples; ++i) {
        state = state * 1103515245u + 12345u;
        pcm[i] = (float)((int)((state >> 16) & 0x7fff) - 16384) / 32768.0f;
    }
}

/* A spectrum at the model's own framing, varied per frame so the feature
 * windows demonstrably slide. */
static void frame_spectrum(int t, float *re, float *im) {
    int k;
    for (k = 0; k < DFN2_N_BINS; ++k) {
        re[k] = 0.25f * sinf(0.017f * (float)k + 0.11f * (float)t) +
                0.01f * (float)(t + 1);
        im[k] = 0.19f * cosf(0.023f * (float)k - 0.07f * (float)t) -
                0.005f * (float)(t + 1);
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

static void config_time(DFN2PrepostConfig *cfg, FftHandle *fft) {
    (void)dfn2_prepost_config_defaults(cfg, DFN2_IO_TIME);
    cfg->fft = fft;
    cfg->erb_fwd = &erb_fwd[0][0];
    cfg->erb_inv = &erb_inv[0][0];
}

static void config_freq(DFN2PrepostConfig *cfg) {
    (void)dfn2_prepost_config_defaults(cfg, DFN2_IO_FREQ);
    cfg->erb_fwd = &erb_fwd[0][0];
    cfg->erb_inv = &erb_inv[0][0];
}

static float pcm_in[HOPS * DFN2_HOP_LEN];
static float out_a[HOPS * DFN2_HOP_LEN];
static float out_b[HOPS * DFN2_HOP_LEN];
static float out_c[HOPS * DFN2_HOP_LEN];
static int written_log[HOPS];
static long long frame_log[HOPS];

/* ---- the canonical per-hop loop from dfn2_prepost.h -------------------
 * `n` is 0 on the very first hop of a cold instance and 1 on every hop after
 * -- never 2 -- so that is asserted per hop rather than merely looped over. */
static int drive_time_ex(DFN2Prepost *p, const float *in, float *out,
                         int hops, int cold, int *written_out,
                         long long *frame_out) {
    int hop;
    for (hop = 0; hop < hops; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n, written = -1;
        long long frame = -1;
        n = dfn2_prepost_pre_process(p, in + (size_t)hop * DFN2_HOP_LEN);
        if (n != ((cold && hop == 0) ? 0 : 1)) return -1;
        if (n == 1) {
            if (dfn2_prepost_frame_inputs(p, &inputs, &outputs) != 0)
                return -1;
            fake_run(&inputs, &outputs);
            if (dfn2_prepost_frame_commit(p) != 0) return -1;
        }
        if (dfn2_prepost_post_process(p, out + (size_t)hop * DFN2_HOP_LEN,
                                      &written) != 0) return -1;
        if (written_out) written_out[hop] = written;
        if (frame_out) {
            if (dfn2_prepost_output_frame_index(p, &frame) != 0) frame = -1;
            frame_out[hop] = frame;
        }
    }
    return 0;
}

static int drive_time(DFN2Prepost *p, const float *in, float *out, int hops) {
    return drive_time_ex(p, in, out, hops, 1, NULL, NULL);
}

/* ---- equiv: the class vs the hand-composed path ---------------------- */

static int equiv_once(FftHandle *fft, float atten_lim_db) {
    static DFN2State ref_st;
    static DFN2ModelIOState ref_io;
    static RefHeads heads;
    static float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    static float feat_erb[DFN2_N_ERB], feat_spec[2 * DFN2_DF_BINS];
    static float enh_re[DFN2_N_BINS], enh_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;
    int hop;

    /* path A -- analysis, features, the model-I/O transaction, the compose
     * stage and the synthesis, wired by hand from the two TUs the class
     * merely orchestrates.  dfn2_prepost.c is deliberately not involved. */
    dfn2_state_init(&ref_st, fft);
    dfn2_set_erb_matrices(&ref_st, &erb_fwd[0][0], &erb_inv[0][0]);
    dfn2_model_io_init(&ref_io);
    for (hop = 0; hop < HOPS; ++hop) {
        float *slot = out_a + (size_t)hop * DFN2_HOP_LEN;
        int needed, emitted;
        dfn2_analysis(&ref_st, pcm_in + (size_t)hop * DFN2_HOP_LEN,
                      spec_re, spec_im);
        dfn2_compute_features(&ref_st, spec_re, spec_im, feat_erb, feat_spec);
        needed = dfn2_model_io_push_features(
            &ref_io, feat_erb, (const float (*)[DFN2_DF_BINS])feat_spec);
        CHECK(needed >= 0);
        if (needed == 0) {
            /* Left warm-up: the graph has no right-hand neighbour yet, so
             * the compose clock advances with heads_valid = 0. */
            emitted = dfn2_compose_stream(&ref_st, spec_re, spec_im, 0,
                                          NULL, NULL, 0.0f, atten_lim_db,
                                          enh_re, enh_im, NULL);
        } else {
            fake_run_reference(&ref_io, &heads);
            emitted = dfn2_compose_stream(&ref_st, spec_re, spec_im, 1,
                                          heads.erb_mask, heads.coefs,
                                          heads.alpha[0], atten_lim_db,
                                          enh_re, enh_im, NULL);
            CHECK(dfn2_model_io_commit_state(
                      &ref_io,
                      (const float (*)[DFN2_MODEL_GRU_HIDDEN])
                          heads.encoder_next,
                      (const float (*)[DFN2_MODEL_GRU_HIDDEN])
                          heads.erb_next,
                      (const float (*)[DFN2_MODEL_GRU_HIDDEN])
                          heads.df_next,
                      (const float (*)[DFN2_MODEL_DF_PATHWAY_HISTORY]
                                      [DFN2_DF_BINS])heads.convp_next) == 0);
        }
        CHECK(emitted >= 0);
        if (emitted == 1) {
            dfn2_synthesis(&ref_st, enh_re, enh_im, slot);
        } else {
            memset(slot, 0, (size_t)DFN2_HOP_LEN * sizeof(float));
        }
    }

    /* path B -- the class */
    config_time(&cfg, fft);
    cfg.atten_lim_db = atten_lim_db;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = dfn2_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_io_mode(p) == DFN2_IO_TIME);
    CHECK(dfn2_prepost_hop_size(p) == DFN2_HOP_LEN);
    CHECK(dfn2_prepost_num_bins(p) == DFN2_N_BINS);
    CHECK(dfn2_prepost_layout_version(p) == DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == LOOKAHEAD);
    CHECK(drive_time_ex(p, pcm_in, out_b, HOPS, 1,
                        written_log, frame_log) == 0);

    printf("equiv: io_mode=TIME atten_lim_db=%.1f pool=%llu B\n",
           (double)atten_lim_db, (unsigned long long)req.bytes);

    /* The two lookahead hops are paid up front and only there: `written` is
     * 0 for exactly the first LOOKAHEAD hops, DFN2_HOP_LEN afterwards, and
     * the frame id is unavailable until the first emission and then counts
     * source frames from 0 without a gap. */
    for (hop = 0; hop < HOPS; ++hop) {
        if (hop < LOOKAHEAD) {
            CHECK(written_log[hop] == 0);
            CHECK(frame_log[hop] == -1);
        } else {
            CHECK(written_log[hop] == DFN2_HOP_LEN);
            CHECK(frame_log[hop] == (long long)(hop - LOOKAHEAD));
        }
    }

    /* The reference is neither silence nor a constant, or byte-identity
     * below would be vacuous. */
    CHECK(any_nonzero(out_a, (size_t)HOPS * DFN2_HOP_LEN));
    CHECK(!identical(out_a, out_b + 1, (size_t)HOPS * DFN2_HOP_LEN - 1));
    CHECK(identical(out_a, out_b, (size_t)HOPS * DFN2_HOP_LEN));

    dfn2_prepost_destroy(p);
    free(pool);
    return 0;
}

static int case_equiv(FftHandle *fft) {
    /* Once with the attenuation limit disabled, once with it engaged, so the
     * limit path inside the compose stage is covered by the same identity. */
    if (equiv_once(fft, 0.0f) != 0) return 1;
    if (equiv_once(fft, -20.0f) != 0) return 1;
    return 0;
}

/* ---- freq: IO_FREQ on the caller's own transform vs IO_TIME ---------- */

static int case_freq(FftHandle *fft) {
    static DFN2State caller_st;
    static float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    static float enh_re[DFN2_N_BINS], enh_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg_time, cfg_freq;
    DFN2PrepostMemReq req_time, req_freq;
    DFN2Prepost *p_time, *p_freq;
    void *pool_time, *pool_freq;
    int hop;

    config_time(&cfg_time, fft);
    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);
    p_time = dfn2_prepost_init(pool_time, (size_t)req_time.bytes, &cfg_time);
    p_freq = dfn2_prepost_init(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(p_time != NULL && p_freq != NULL);
    CHECK(dfn2_prepost_io_mode(p_freq) == DFN2_IO_FREQ);

    CHECK(drive_time(p_time, pcm_in, out_a, HOPS) == 0);

    /* In DFN2_IO_FREQ the CALLER owns the transform on both sides: one
     * dfn2_analysis per hop into the class, one dfn2_synthesis per emitted
     * frame out of it, on the same DFN2State the class would have used. */
    dfn2_state_init(&caller_st, fft);
    dfn2_set_erb_matrices(&caller_st, &erb_fwd[0][0], &erb_inv[0][0]);
    for (hop = 0; hop < HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        float *slot = out_b + (size_t)hop * DFN2_HOP_LEN;
        int n, valid = -1;
        dfn2_analysis(&caller_st, pcm_in + (size_t)hop * DFN2_HOP_LEN,
                      spec_re, spec_im);
        n = dfn2_prepost_pre_process_freq(p_freq, spec_re, spec_im);
        CHECK(n == (hop == 0 ? 0 : 1));
        if (n == 1) {
            CHECK(dfn2_prepost_frame_inputs(p_freq, &inputs, &outputs) == 0);
            fake_run(&inputs, &outputs);
            CHECK(dfn2_prepost_frame_commit(p_freq) == 0);
        }
        CHECK(dfn2_prepost_post_process_freq(p_freq, enh_re, enh_im,
                                             &valid) == 0);
        if (hop < LOOKAHEAD) {
            /* Warm-up: not merely "ignore this", but a defined all-zero
             * spectrum, so a caller that forwards it anyway emits silence
             * rather than the previous frame. */
            CHECK(valid == 0);
            CHECK(all_zero(enh_re, DFN2_N_BINS));
            CHECK(all_zero(enh_im, DFN2_N_BINS));
            memset(slot, 0, (size_t)DFN2_HOP_LEN * sizeof(float));
        } else {
            CHECK(valid == 1);
            dfn2_synthesis(&caller_st, enh_re, enh_im, slot);
        }
    }

    /* Cross-mode calls are refused, never silently reinterpreted. */
    CHECK(dfn2_prepost_pre_process(p_freq, pcm_in) == -1);
    CHECK(dfn2_prepost_post_process(p_freq, out_c, NULL) == -1);
    CHECK(dfn2_prepost_pre_process_freq(p_time, spec_re, spec_im) == -1);
    CHECK(dfn2_prepost_post_process_freq(p_time, enh_re, enh_im, NULL) == -1);

    CHECK(any_nonzero(out_a, (size_t)HOPS * DFN2_HOP_LEN));
    CHECK(!identical(out_a, out_b + 1, (size_t)HOPS * DFN2_HOP_LEN - 1));
    CHECK(identical(out_a, out_b, (size_t)HOPS * DFN2_HOP_LEN));

    dfn2_prepost_destroy(p_time);
    dfn2_prepost_destroy(p_freq);
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- freqpool: pool saving + the stale-pool gate --------------------- */

static int case_freqpool(FftHandle *fft) {
    DFN2PrepostConfig cfg_time, cfg_freq;
    DFN2PrepostMemReq req_time, req_freq;
    void *pool_time, *pool_freq;

    config_time(&cfg_time, fft);
    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);

    printf("pool TIME = %llu B\npool FREQ = %llu B   (saves %llu B, %.2f%%)\n",
           (unsigned long long)req_time.bytes,
           (unsigned long long)req_freq.bytes,
           (unsigned long long)(req_time.bytes - req_freq.bytes),
           100.0 * (double)(req_time.bytes - req_freq.bytes) /
               (double)req_time.bytes);

    /* IO_FREQ does not carve the output hop staging it never uses.  The
     * header is honest that this is the ONLY difference -- the saving is
     * small, but it must not be zero or negative. */
    CHECK(req_freq.bytes < req_time.bytes);
    CHECK(req_time.io_mode == (uint32_t)DFN2_IO_TIME);
    CHECK(req_freq.io_mode == (uint32_t)DFN2_IO_FREQ);
    CHECK(req_time.build_flags_hash != req_freq.build_flags_hash);
    CHECK(req_time.descriptor_version == DFN2_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_freq.descriptor_version == DFN2_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_time.layout_version == (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(req_freq.layout_version == (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(req_time.alignment == DFN2_PREPOST_ALIGNMENT);
    CHECK(req_time.reserved == 0u);
    CHECK(req_freq.reserved == 0u);

    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes, &cfg_time,
                               &req_time) != NULL);
    CHECK(dfn2_prepost_init_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq,
                               &req_freq) != NULL);
    /* ...and refuse a pool recorded for the other mode. */
    CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes, &cfg_time,
                               &req_freq) == NULL);
    CHECK(dfn2_prepost_init_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq,
                               &req_time) == NULL);
    {
        DFN2PrepostMemReq stale;
        stale = req_time; ++stale.descriptor_version;
        CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                   &cfg_time, &stale) == NULL);
        stale = req_time; ++stale.layout_version;
        CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                   &cfg_time, &stale) == NULL);
        stale = req_time; ++stale.build_flags_hash;
        CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                   &cfg_time, &stale) == NULL);
        stale = req_time; ++stale.bytes;
        CHECK(dfn2_prepost_init_ex(pool_time, (size_t)req_time.bytes,
                                   &cfg_time, &stale) == NULL);
    }
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(FftHandle *fft) {
    DFN2PrepostConfig cfg, bad;
    DFN2PrepostMemReq req;
    DFN2Prepost *heap, *stack, *tmp;
    void *pool;
    long long frame = 0;

    config_time(&cfg, fft);
    heap = dfn2_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(dfn2_prepost_io_mode(heap) == DFN2_IO_TIME);
    CHECK(dfn2_prepost_hop_size(heap) == DFN2_HOP_LEN);
    CHECK(dfn2_prepost_num_bins(heap) == DFN2_N_BINS);
    CHECK(dfn2_prepost_layout_version(heap) == DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(dfn2_prepost_model_lookahead_frames(heap) == LOOKAHEAD);
    CHECK(drive_time(heap, pcm_in, out_a, SHORT_HOPS) == 0);
    CHECK(any_nonzero(out_a, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* A created instance and a pool instance are the same object. */
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    stack = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(stack != NULL);
    CHECK(drive_time(stack, pcm_in, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still runs
     * afterwards.  (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    dfn2_prepost_destroy(stack);
    dfn2_prepost_destroy(stack);
    dfn2_prepost_reset(stack);
    CHECK(drive_time(stack, pcm_in, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* Runtime setters on a live instance: reject-first, then accept. */
    CHECK(dfn2_prepost_set_erb_matrices(NULL, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_erb_matrices(stack, NULL, &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_erb_matrices(stack, &erb_fwd[0][0], NULL) == -1);
    CHECK(dfn2_prepost_set_erb_matrices(stack, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == 0);
    CHECK(dfn2_prepost_set_atten_lim(NULL, 0.0f) == -1);
    CHECK(dfn2_prepost_set_atten_lim(stack, (float)NAN) == -1);
    CHECK(dfn2_prepost_set_atten_lim(stack, (float)INFINITY) == -1);
    CHECK(dfn2_prepost_set_atten_lim(stack, -(float)INFINITY) == -1);
    CHECK(dfn2_prepost_set_atten_lim(stack, -20.0f) == 0);
    CHECK(dfn2_prepost_set_atten_lim(stack, 0.0f) == 0);

    dfn2_prepost_destroy(heap);
    dfn2_prepost_destroy(NULL);
    free(pool);

    /* A heap instance in the mode that needs no transform at all. */
    config_freq(&cfg);
    heap = dfn2_prepost_create(&cfg);
    CHECK(heap != NULL);
    CHECK(dfn2_prepost_io_mode(heap) == DFN2_IO_FREQ);
    CHECK(dfn2_prepost_hop_size(heap) == DFN2_HOP_LEN);
    dfn2_prepost_destroy(heap);

    /* _create refuses exactly what _get_mem_size refuses, and allocates
     * nothing when it does. */
    CHECK(dfn2_prepost_create(NULL) == NULL);
    CHECK(dfn2_prepost_config_defaults(&bad, DFN2_IO_TIME) == 0);
    CHECK(dfn2_prepost_create(&bad) == NULL);      /* no fft, no ERB       */
    bad.erb_fwd = &erb_fwd[0][0];
    bad.erb_inv = &erb_inv[0][0];
    CHECK(dfn2_prepost_create(&bad) == NULL);      /* TIME without fft     */
    bad.fft = fft;
    bad.erb_fwd = NULL;
    CHECK(dfn2_prepost_create(&bad) == NULL);      /* erb_fwd missing      */
    bad.erb_fwd = &erb_fwd[0][0];
    bad.erb_inv = NULL;
    CHECK(dfn2_prepost_create(&bad) == NULL);      /* erb_inv missing      */
    bad.erb_inv = &erb_inv[0][0];
    tmp = dfn2_prepost_create(&bad);               /* ...and now accepts   */
    CHECK(tmp != NULL);
    dfn2_prepost_destroy(tmp);

    /* Accessors on NULL. */
    CHECK(dfn2_prepost_hop_size(NULL) == -1);
    CHECK(dfn2_prepost_num_bins(NULL) == -1);
    CHECK(dfn2_prepost_io_mode(NULL) == -1);
    CHECK(dfn2_prepost_model_lookahead_frames(NULL) == -1);
    CHECK(dfn2_prepost_layout_version(NULL) == -1);
    CHECK(dfn2_prepost_output_frame_index(NULL, &frame) == -1);
    dfn2_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

static int case_reject(FftHandle *fft) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, guard;
    void *wrong_mem = NULL;
    FftHandle *wrong_fft;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    config_time(&cfg, fft);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(dfn2_prepost_config_defaults(NULL, DFN2_IO_TIME) == -1);
    CHECK(dfn2_prepost_config_defaults(&cfg, 2) == -1);
    CHECK(dfn2_prepost_config_defaults(&cfg, -1) == -1);

    /* *req is left untouched on every refusal, not half-filled. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    config_time(&cfg, fft);
    CHECK(dfn2_prepost_get_mem_size(&cfg, NULL) == -1);

    cfg.io_mode = 2;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.io_mode = -1;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);

    /* The ERB matrices are required in BOTH modes -- mask expansion and the
     * ERB feature branch live on the spectrum side, not the framing side. */
    config_time(&cfg, fft);
    cfg.erb_fwd = NULL;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    config_time(&cfg, fft);
    cfg.erb_inv = NULL;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    config_freq(&cfg);
    cfg.erb_fwd = NULL;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    config_freq(&cfg);
    cfg.erb_inv = NULL;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);

    /* A non-finite attenuation limit. */
    config_time(&cfg, fft);
    cfg.atten_lim_db = (float)NAN;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.atten_lim_db = (float)INFINITY;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.atten_lim_db = -(float)INFINITY;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);

    /* DFN2_IO_TIME without a transform, and with a correctly typed but
     * WRONG-SIZE one: 257 bins, not 513. */
    config_time(&cfg, fft);
    cfg.fft = NULL;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    wrong_fft = make_fft(DFN2_N_FFT / 2, &wrong_mem);
    CHECK(wrong_fft != NULL);
    CHECK(fft_get_n_freqs(wrong_fft) != DFN2_N_BINS);
    cfg.fft = wrong_fft;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    CHECK(dfn2_prepost_init(NULL, 0, &cfg) == NULL);

    /* DFN2_IO_FREQ ignores the transform entirely -- NULL is legal there. */
    config_freq(&cfg);
    CHECK(cfg.fft == NULL);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);

    /* _init: NULL pool, undersized pool, misaligned pool, then the good one. */
    config_time(&cfg, fft);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes + req.alignment);
    CHECK(pool != NULL);
    CHECK(dfn2_prepost_init(NULL, (size_t)req.bytes, &cfg) == NULL);
    CHECK(dfn2_prepost_init(pool, (size_t)req.bytes - 1u, &cfg) == NULL);
    CHECK(dfn2_prepost_init((unsigned char *)pool + 1,
                            (size_t)req.bytes, &cfg) == NULL);
    CHECK(dfn2_prepost_init(pool, (size_t)req.bytes, &cfg) != NULL);

    /* Stage calls: NULLs, out-of-order calls, and the two "no frame open"
     * shapes the state machine must distinguish. */
    {
        DFN2Prepost *p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int written = -1;
        CHECK(p != NULL);
        CHECK(dfn2_prepost_pre_process(NULL, pcm_in) == -1);
        CHECK(dfn2_prepost_pre_process(p, NULL) == -1);
        CHECK(dfn2_prepost_pre_process_freq(NULL, pcm_in, pcm_in) == -1);
        CHECK(dfn2_prepost_frame_inputs(NULL, &inputs, &outputs) == -1);
        CHECK(dfn2_prepost_frame_inputs(p, NULL, &outputs) == -1);
        CHECK(dfn2_prepost_frame_inputs(p, &inputs, NULL) == -1);
        CHECK(dfn2_prepost_frame_commit(NULL) == -1);
        CHECK(dfn2_prepost_frame_skip(NULL) == -1);
        CHECK(dfn2_prepost_post_process(NULL, out_a, NULL) == -1);
        CHECK(dfn2_prepost_post_process(p, NULL, NULL) == -1);

        /* Nothing is open yet. */
        CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(dfn2_prepost_frame_commit(p) == -1);
        CHECK(dfn2_prepost_frame_skip(p) == -1);

        /* post_process before any frame ran: a full, defined, silent hop. */
        memset(out_a, 0x5a, (size_t)DFN2_HOP_LEN * sizeof(float));
        CHECK(dfn2_prepost_post_process(p, out_a, &written) == 0);
        CHECK(written == 0);
        CHECK(all_zero(out_a, DFN2_HOP_LEN));

        /* Open a frame, then refuse the emit that would strand it. */
        CHECK(dfn2_prepost_pre_process(p, pcm_in) == 0);
        CHECK(dfn2_prepost_pre_process(p, pcm_in + DFN2_HOP_LEN) == 1);
        CHECK(dfn2_prepost_post_process(p, out_a, NULL) == -1);
        /* An accelerator that never wrote leaves the NaN prefill in place,
         * so the transaction refuses rather than committing garbage. */
        CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        CHECK(dfn2_prepost_frame_commit(p) == -1);
        CHECK(dfn2_prepost_frame_skip(p) == 0);
    }

    free(pool);
    free(wrong_mem);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(FftHandle *fft) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;

    config_time(&cfg, fft);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    CHECK(drive_time(p, pcm_in, out_a, SHORT_HOPS) == 0);
    /* Without a reset the second run continues from a warm state -- assert
     * that it does NOT match, or the comparison after the reset has no
     * teeth.  A warm instance also emits a frame on its very first hop. */
    CHECK(drive_time_ex(p, pcm_in, out_b, SHORT_HOPS, 0, NULL, NULL) == 0);
    CHECK(!identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    dfn2_prepost_reset(p);
    /* The framing clock is cold again: the first hop asks for no inference. */
    CHECK(dfn2_prepost_pre_process(p, pcm_in) == 0);
    dfn2_prepost_reset(p);
    CHECK(drive_time(p, pcm_in, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* A reset mid-frame drops the open frame instead of stranding it. */
    {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        dfn2_prepost_reset(p);
        CHECK(dfn2_prepost_pre_process(p, pcm_in) == 0);
        CHECK(dfn2_prepost_pre_process(p, pcm_in + DFN2_HOP_LEN) == 1);
        CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
        dfn2_prepost_reset(p);
        CHECK(dfn2_prepost_frame_commit(p) == -1);
        CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == -1);
        CHECK(drive_time(p, pcm_in, out_b, SHORT_HOPS) == 0);
        CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));
    }

    dfn2_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- skip: the exact identity, and no state advance ------------------ */

static int case_skip(void) {
    static float in_re[SKIP_TOTAL][DFN2_N_BINS];
    static float in_im[SKIP_TOTAL][DFN2_N_BINS];
    static float got_re[DFN2_N_BINS], got_im[DFN2_N_BINS];
    static float snap_encoder[DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS];
    static float snap_erb[DFN2_PREPOST_ERB_HIDDEN_ELEMENTS];
    static float snap_df[DFN2_PREPOST_DF_HIDDEN_ELEMENTS];
    static float snap_convp[DFN2_PREPOST_CONVP_HISTORY_ELEMENTS];
    static float snap_erb_window[DFN2_PREPOST_ERB_WINDOW_ELEMENTS];
    static float snap_spec_window[DFN2_PREPOST_SPEC_WINDOW_ELEMENTS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    DFN2Prepost *p;
    void *pool;
    int t;

    /* DFN2_IO_FREQ so the identity is observable on the spectrum itself,
     * without a WOLA round trip in between. */
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    for (t = 0; t < SKIP_TOTAL; ++t) frame_spectrum(t, in_re[t], in_im[t]);
    CHECK(any_nonzero(in_re[0], DFN2_N_BINS));

    /* Every frame taken on frame_skip: the accelerator never runs at all,
     * and the class must still be a bit-exact delay line of exactly
     * MASK_LOOKAHEAD + DF_LOOKAHEAD frames. */
    for (t = 0; t < SKIP_FRAMES; ++t) {
        int n, valid = -1;
        n = dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]);
        CHECK(n == (t == 0 ? 0 : 1));
        if (n == 1) CHECK(dfn2_prepost_frame_skip(p) == 0);
        CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &valid) == 0);
        if (t < LOOKAHEAD) {
            CHECK(valid == 0);
            CHECK(all_zero(got_re, DFN2_N_BINS));
            CHECK(all_zero(got_im, DFN2_N_BINS));
        } else {
            CHECK(valid == 1);
            CHECK(identical(got_re, in_re[t - LOOKAHEAD], DFN2_N_BINS));
            CHECK(identical(got_im, in_im[t - LOOKAHEAD], DFN2_N_BINS));
        }
    }

    /* One REAL commit, so the recurrent tensors are non-zero and "unchanged"
     * below means something.
     *
     * The frame this hop EMITS is still the identity, and that is the
     * cascade's timing rather than a coincidence: the heads shaping source
     * frame t - LOOKAHEAD were supplied one hop EARLIER (they land in the
     * ring at head_frame = hop - MASK_LOOKAHEAD), and that hop was a skip.
     * A commit therefore never reaches back and re-masks the frame it
     * emits. */
    t = SKIP_FRAMES;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    CHECK(identical(got_re, in_re[t - LOOKAHEAD], DFN2_N_BINS));
    CHECK(identical(got_im, in_im[t - LOOKAHEAD], DFN2_N_BINS));

    /* Snapshot, then skip even though the accelerator produced a COMPLETE,
     * perfectly committable result.  The full write is the point: skipping
     * after a partial write would prove nothing, because commit() would
     * refuse that frame on its own. */
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    memcpy(snap_encoder, &inputs.encoder_gru_hidden[0][0],
           sizeof snap_encoder);
    memcpy(snap_erb, &inputs.erb_gru_hidden[0][0], sizeof snap_erb);
    memcpy(snap_df, &inputs.df_gru_hidden[0][0], sizeof snap_df);
    memcpy(snap_convp, &inputs.df_convp_history[0][0][0], sizeof snap_convp);
    memcpy(snap_erb_window, &inputs.erb_window[0][0], sizeof snap_erb_window);
    memcpy(snap_spec_window, &inputs.spec_window[0][0][0],
           sizeof snap_spec_window);
    CHECK(any_nonzero(snap_encoder, DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS));
    CHECK(any_nonzero(snap_erb, DFN2_PREPOST_ERB_HIDDEN_ELEMENTS));
    CHECK(any_nonzero(snap_df, DFN2_PREPOST_DF_HIDDEN_ELEMENTS));
    CHECK(any_nonzero(snap_convp, DFN2_PREPOST_CONVP_HISTORY_ELEMENTS));
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_skip(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    /* This hop's emission is NOT the identity, and asserting the difference
     * is the point: source frame t - LOOKAHEAD was masked by the REAL heads
     * committed one hop earlier, so a mismatch here proves that commit
     * actually reached the compose stage. A skip only takes the identity for
     * the frame whose heads IT supplies, which is emitted one hop later. */
    CHECK(!identical(got_re, in_re[t - LOOKAHEAD], DFN2_N_BINS));
    CHECK(!identical(got_im, in_im[t - LOOKAHEAD], DFN2_N_BINS));

    /* The four recurrent tensors are frozen; the two feature windows are
     * NOT -- they are the framing clock and must still have slid. */
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(identical(&inputs.encoder_gru_hidden[0][0], snap_encoder,
                    inputs.encoder_gru_hidden_elements));
    CHECK(identical(&inputs.erb_gru_hidden[0][0], snap_erb,
                    inputs.erb_gru_hidden_elements));
    CHECK(identical(&inputs.df_gru_hidden[0][0], snap_df,
                    inputs.df_gru_hidden_elements));
    CHECK(identical(&inputs.df_convp_history[0][0][0], snap_convp,
                    inputs.df_convp_history_elements));
    CHECK(!identical(&inputs.erb_window[0][0], snap_erb_window,
                     inputs.erb_window_elements));
    CHECK(!identical(&inputs.spec_window[0][0][0], snap_spec_window,
                     inputs.spec_window_elements));

    /* ...and a commit here DOES move them, so the freeze above is not
     * measuring a constant. */
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
    CHECK(!identical(&inputs.encoder_gru_hidden[0][0], snap_encoder,
                     inputs.encoder_gru_hidden_elements));
    CHECK(!identical(&inputs.erb_gru_hidden[0][0], snap_erb,
                     inputs.erb_gru_hidden_elements));
    CHECK(!identical(&inputs.df_gru_hidden[0][0], snap_df,
                     inputs.df_gru_hidden_elements));
    CHECK(!identical(&inputs.df_convp_history[0][0][0], snap_convp,
                     inputs.df_convp_history_elements));
    CHECK(dfn2_prepost_frame_skip(p) == 0);

    dfn2_prepost_destroy(p);
    free(pool);
    return 0;
}

/* ---- guard: the frame-state-machine contract gates ------------------- */

static int case_guard(FftHandle *fft) {
    static float re_a[DFN2_N_BINS], im_a[DFN2_N_BINS];
    static float re_b[DFN2_N_BINS], im_b[DFN2_N_BINS];
    static float hop_scratch[DFN2_HOP_LEN];
    DFN2PrepostConfig cfg_time, cfg_freq;
    DFN2PrepostMemReq req_time, req_freq;
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    DFN2Prepost *pt, *pf;
    void *pool_time, *pool_freq;

    config_time(&cfg_time, fft);
    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);
    pt = dfn2_prepost_init(pool_time, (size_t)req_time.bytes, &cfg_time);
    pf = dfn2_prepost_init(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(pt != NULL && pf != NULL);
    frame_spectrum(0, re_a, im_a);
    frame_spectrum(1, re_b, im_b);

    /* (1) A second pre_process with a frame open is refused, and the refusal
     * does not eat the open frame: it is still committable afterwards. */
    CHECK(dfn2_prepost_pre_process(pt, pcm_in) == 0);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in + DFN2_HOP_LEN) == 1);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in + 2 * DFN2_HOP_LEN) == -1);
    /* The setters are between-hops operations. With the frame open they
     * are refused, so a swap can never land inside one transaction (between
     * the features just taken and the compose still pending), and the
     * refusal leaves the frame committable. */
    CHECK(dfn2_prepost_set_erb_matrices(pt, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_atten_lim(pt, -12.0f) == -1);
    CHECK(dfn2_prepost_frame_inputs(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(pt) == 0);
    CHECK(dfn2_prepost_set_erb_matrices(pt, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == 0);
    CHECK(dfn2_prepost_set_atten_lim(pt, 0.0f) == 0);

    CHECK(dfn2_prepost_pre_process_freq(pf, re_a, im_a) == 0);
    CHECK(dfn2_prepost_pre_process_freq(pf, re_b, im_b) == 1);
    CHECK(dfn2_prepost_pre_process_freq(pf, re_a, im_a) == -1);
    /* Same refusal in FREQ mode: the guard is keyed to the frame, not to
     * who owns the transform. */
    CHECK(dfn2_prepost_set_erb_matrices(pf, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_atten_lim(pf, -12.0f) == -1);
    CHECK(dfn2_prepost_frame_inputs(pf, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(pf) == 0);
    CHECK(dfn2_prepost_set_atten_lim(pf, 0.0f) == 0);

    /* (2) A commit with no frame_inputs behind it is refused: an accelerator
     * that never ran must not be able to pass untouched buffers off as a
     * result.  The frame stays open and commits normally afterwards. */
    dfn2_prepost_reset(pt);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in) == 0);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in + DFN2_HOP_LEN) == 1);
    CHECK(dfn2_prepost_frame_commit(pt) == -1);
    CHECK(dfn2_prepost_frame_inputs(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(pt) == 0);

    /* (3) A commit that failed on a non-finite output DISARMS the
     * transaction: retrying it without a fresh frame_inputs is refused,
     * and the fresh frame_inputs NaN-refills every writable element. */
    dfn2_prepost_reset(pt);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in) == 0);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in + DFN2_HOP_LEN) == 1);
    CHECK(dfn2_prepost_frame_inputs(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    {
        /* Keep the good value so the retry below can be made perfectly
         * committable again.  Retrying while the NaN is still in place
         * would be refused by the finiteness preflight whether or not the
         * transaction was disarmed, and would prove nothing. */
        const size_t poison = DFN2_PREPOST_COEFS_ELEMENTS / 2;
        float good = outputs.coefs[poison];
        CHECK(isfinite(good));
        outputs.coefs[poison] = (float)NAN;
        CHECK(dfn2_prepost_frame_commit(pt) == -1);
        /* Repaired in place through the stale view: the only thing left
         * that can refuse this commit is the disarm itself. */
        outputs.coefs[poison] = good;
        CHECK(dfn2_prepost_frame_commit(pt) == -1);
    }
    CHECK(dfn2_prepost_frame_inputs(pt, &inputs, &outputs) == 0);
    CHECK(outputs_all_nan(&outputs));
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_commit(pt) == 0);

    /* (4) frame_skip is exactly the accelerator-never-ran path, so it needs
     * no frame_inputs behind it. */
    dfn2_prepost_reset(pt);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in) == 0);
    CHECK(dfn2_prepost_pre_process(pt, pcm_in + DFN2_HOP_LEN) == 1);
    CHECK(dfn2_prepost_frame_skip(pt) == 0);
    CHECK(dfn2_prepost_post_process(pt, hop_scratch, NULL) == 0);

    dfn2_prepost_destroy(pt);
    dfn2_prepost_destroy(pf);
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- txn: on failure NOTHING moves, the frame stays open ------------- */

/* Reference stream: at hop TXN_PRE either a plain frame_skip (never running
 * the accelerator) or an ordinary commit of the un-poisoned result. */
static int txn_reference(DFN2Prepost *p, float *out, int commit_at_k) {
    int hop;
    for (hop = 0; hop < TXN_HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n = dfn2_prepost_pre_process(
            p, pcm_in + (size_t)hop * DFN2_HOP_LEN);
        if (n != (hop == 0 ? 0 : 1)) return -1;
        if (n == 1) {
            if (hop == TXN_PRE && !commit_at_k) {
                if (dfn2_prepost_frame_skip(p) != 0) return -1;
            } else {
                if (dfn2_prepost_frame_inputs(p, &inputs, &outputs) != 0)
                    return -1;
                fake_run(&inputs, &outputs);
                if (dfn2_prepost_frame_commit(p) != 0) return -1;
            }
        }
        if (dfn2_prepost_post_process(p, out + (size_t)hop * DFN2_HOP_LEN,
                                      NULL) != 0) return -1;
    }
    return 0;
}

static int txn_poisoned(DFN2Prepost *p, float *out, int tensor, size_t index) {
    static float snap_encoder[DFN2_PREPOST_ENCODER_HIDDEN_ELEMENTS];
    static float snap_erb[DFN2_PREPOST_ERB_HIDDEN_ELEMENTS];
    static float snap_df[DFN2_PREPOST_DF_HIDDEN_ELEMENTS];
    static float snap_convp[DFN2_PREPOST_CONVP_HISTORY_ELEMENTS];
    static float snap_erb_window[DFN2_PREPOST_ERB_WINDOW_ELEMENTS];
    static float snap_spec_window[DFN2_PREPOST_SPEC_WINDOW_ELEMENTS];
    static float hop_scratch[DFN2_HOP_LEN];
    int hop;

    for (hop = 0; hop < TXN_HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n = dfn2_prepost_pre_process(
            p, pcm_in + (size_t)hop * DFN2_HOP_LEN);
        CHECK(n == (hop == 0 ? 0 : 1));
        if (n == 1 && hop == TXN_PRE) {
            float *target;
            size_t count = 0;
            CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
            memcpy(snap_encoder, &inputs.encoder_gru_hidden[0][0],
                   sizeof snap_encoder);
            memcpy(snap_erb, &inputs.erb_gru_hidden[0][0], sizeof snap_erb);
            memcpy(snap_df, &inputs.df_gru_hidden[0][0], sizeof snap_df);
            memcpy(snap_convp, &inputs.df_convp_history[0][0][0],
                   sizeof snap_convp);
            memcpy(snap_erb_window, &inputs.erb_window[0][0],
                   sizeof snap_erb_window);
            memcpy(snap_spec_window, &inputs.spec_window[0][0][0],
                   sizeof snap_spec_window);
            fake_run(&inputs, &outputs);
            target = output_tensor(&outputs, tensor, &count);
            CHECK(index < count);
            target[index] = (float)NAN;
            CHECK(dfn2_prepost_frame_commit(p) == -1);
            /* (a) the frame is STILL OPEN -- neither emitting nor opening
             * the next one is allowed while it is. */
            CHECK(dfn2_prepost_post_process(p, hop_scratch, NULL) == -1);
            CHECK(dfn2_prepost_pre_process(
                      p, pcm_in + (size_t)hop * DFN2_HOP_LEN) == -1);
            /* (b) nothing persistent moved: the same views come back. */
            CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
            CHECK(identical(&inputs.encoder_gru_hidden[0][0], snap_encoder,
                            inputs.encoder_gru_hidden_elements));
            CHECK(identical(&inputs.erb_gru_hidden[0][0], snap_erb,
                            inputs.erb_gru_hidden_elements));
            CHECK(identical(&inputs.df_gru_hidden[0][0], snap_df,
                            inputs.df_gru_hidden_elements));
            CHECK(identical(&inputs.df_convp_history[0][0][0], snap_convp,
                            inputs.df_convp_history_elements));
            CHECK(identical(&inputs.erb_window[0][0], snap_erb_window,
                            inputs.erb_window_elements));
            CHECK(identical(&inputs.spec_window[0][0][0], snap_spec_window,
                            inputs.spec_window_elements));
            /* (c) the documented recovery: take the identity. */
            CHECK(dfn2_prepost_frame_skip(p) == 0);
        } else if (n == 1) {
            CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
            fake_run(&inputs, &outputs);
            CHECK(dfn2_prepost_frame_commit(p) == 0);
        }
        CHECK(dfn2_prepost_post_process(p, out + (size_t)hop * DFN2_HOP_LEN,
                                        NULL) == 0);
    }
    return 0;
}

static int case_txn(FftHandle *fft) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;
    const size_t stream = (size_t)TXN_HOPS * DFN2_HOP_LEN;
    int tensor, which;

    config_time(&cfg, fft);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);

    p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(txn_reference(p, out_a, 0) == 0);      /* skipped at hop K   */
    p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(txn_reference(p, out_b, 1) == 0);      /* committed at hop K */
    /* The teeth: the two references must genuinely differ, otherwise
     * "identical to the skipped run" would be satisfied by anything. */
    CHECK(any_nonzero(out_a, stream));
    CHECK(!identical(out_a, out_b, stream));

    for (tensor = 0; tensor < N_OUT_TENSORS; ++tensor) {
        for (which = 0; which < 3; ++which) {
            DFN2PrepostOutputs shape;
            DFN2PrepostInputs probe;
            size_t count = 0, index;
            DFN2Prepost *probe_p;

            /* Ask the class itself for this tensor's length rather than
             * recomputing it here, so a shape change cannot drift. */
            probe_p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
            CHECK(probe_p != NULL);
            CHECK(dfn2_prepost_pre_process(probe_p, pcm_in) == 0);
            CHECK(dfn2_prepost_pre_process(probe_p,
                                           pcm_in + DFN2_HOP_LEN) == 1);
            CHECK(dfn2_prepost_frame_inputs(probe_p, &probe, &shape) == 0);
            (void)output_tensor(&shape, tensor, &count);
            CHECK(count > 0u);
            index = (which == 0) ? 0u
                  : (which == 1) ? count - 1u
                                 : count / 2u;

            p = dfn2_prepost_init(pool, (size_t)req.bytes, &cfg);
            CHECK(p != NULL);
            CHECK(txn_poisoned(p, out_c, tensor, index) == 0);
            CHECK(identical(out_a, out_c, stream));
            CHECK(!identical(out_b, out_c, stream));
        }
    }

    printf("txn: %d tensors x 3 indices, %d hops each\n",
           N_OUT_TENSORS, TXN_HOPS);
    free(pool);
    return 0;
}

/* ---------------------------------------------------------------------- */

int main(int argc, char **argv) {
    void *fft_mem = NULL;
    FftHandle *fft;
    int status;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <case>\n", argv[0]);
        return 2;
    }
    build_erb();
    /* frame_skip's identity is exact only because a unit band mask expands
     * to a unit bin gain.  Pin that here, in the same accumulation order
     * df_common_expand_mask() uses, so a broken fixture fails loudly
     * instead of turning the skip case into an approximation. */
    CHECK(erb_inv_is_partition_of_unity());
    fill_pcm(pcm_in, HOPS * DFN2_HOP_LEN);
    fft = make_fft(DFN2_N_FFT, &fft_mem);
    CHECK(fft != NULL);
    CHECK(fft_get_n_freqs(fft) == DFN2_N_BINS);

    if (strcmp(argv[1], "equiv") == 0)          status = case_equiv(fft);
    else if (strcmp(argv[1], "freq") == 0)      status = case_freq(fft);
    else if (strcmp(argv[1], "freqpool") == 0)  status = case_freqpool(fft);
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle(fft);
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject(fft);
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset(fft);
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard(fft);
    else if (strcmp(argv[1], "txn") == 0)       status = case_txn(fft);
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
    2026-08-18), built and located the way the other AINR C tests do it."""
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
    """One executable for every case: the class plus the two TUs it composes,
    compiled at the house flags with -Werror."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp('dfn2_prepost_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c11',
         '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off',
         '-I', _DFN_DIR, '-I', _AC_INCLUDE, str(source),
         *[os.path.join(_DFN_DIR, name) for name in _SOURCES],
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
    """The class vs dfn2_analysis + dfn2_compute_features + the model-I/O
    transaction + dfn2_compose_stream + dfn2_synthesis wired up by hand from
    the two composed TUs, byte-identical over 400 hops -- with the
    attenuation limit both disabled and at -20 dB. Also pins the two-hop
    model lookahead: `written` is 0 for exactly the first two hops and
    DFN2_HOP_LEN after, and the reported output frame index is unavailable
    until the first emission and then counts source frames without a gap."""
    assert 'io_mode=TIME' in _run(driver, 'equiv')


def test_freq_mode_matches_time_mode(driver):
    """DFN2_IO_FREQ fed from the caller's OWN dfn2_analysis on a separate
    DFN2State and re-synthesised by the caller must reproduce DFN2_IO_TIME
    sample for sample; the warm-up emissions report valid == 0 and carry an
    all-zero spectrum; and every cross-mode stage call is refused."""
    _run(driver, 'freq')


def test_freq_pool_is_smaller_and_stale_pool_gate_holds(driver):
    """DFN2_IO_FREQ must not carve the output hop staging it never uses, the
    MemReq must carry this build's identity, and _init_ex must accept the
    matching descriptor while refusing the other mode's and any single-field
    perturbation of its own."""
    stdout = _run(driver, 'freqpool')
    assert 'pool FREQ' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe and a genuine
    no-op on a pool instance, the accessors report the graph contract this
    build binds, and the runtime setters reject NULL/non-finite input while
    accepting good values."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size leaves *req untouched when it refuses (NULL args, an
    unknown io_mode, a missing ERB matrix in either mode, a non-finite
    attenuation limit, DFN2_IO_TIME without or with a wrong-size FftHandle);
    _init refuses a NULL, undersized or misaligned pool; and the stage calls
    refuse NULLs and out-of-order use."""
    _run(driver, 'reject')


def test_reset_clears_state(driver):
    """N hops, reset, the same N hops again -> byte-identical, asserted
    against a NO-reset second run first so it cannot pass by the output being
    state-independent. A reset mid-frame drops the open frame rather than
    stranding it."""
    _run(driver, 'reset')


def test_frame_skip_is_exact_identity_and_freezes_the_state(driver):
    """Driven entirely on frame_skip, the class is a bit-exact delay line of
    MASK_LOOKAHEAD + DF_LOOKAHEAD frames. A skip taken over a COMPLETE
    accelerator result leaves all four recurrent tensors byte-identical while
    the two feature windows still slide, and a following commit does move
    them -- so the freeze is not measuring a constant."""
    _run(driver, 'skip')


def test_frame_state_machine_guards(driver):
    """The contract gates: a second pre_process with a frame open is refused
    without eating the frame; the ERB and attenuation setters are refused
    while a frame is open (a swap can never land inside a transaction) and
    accepted again after it closes; a commit with no frame_inputs behind it
    is refused; a
    commit that failed on a non-finite output disarms the transaction until
    a fresh frame_inputs NaN-refills every writable element; and frame_skip
    needs no frame_inputs behind it."""
    _run(driver, 'guard')


def test_failed_commit_moves_nothing_and_leaves_the_frame_open(driver):
    """Poison one element of one accelerator output tensor and commit: the
    refusal must leave the frame open, every recurrent tensor and both
    feature windows byte-identical to a snapshot taken before it, and
    frame_skip available as the documented recovery -- so the whole 61-hop
    stream matches a run that plainly skipped that hop, and differs from one
    that committed it. Swept over all seven output tensors at three
    indices."""
    assert 'txn:' in _run(driver, 'txn')
