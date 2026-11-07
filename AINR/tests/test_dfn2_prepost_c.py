"""Equivalence and contract gates for the DeepFilterNet2 pre/post class.

``dfn2_prepost.c`` owns no signal processing: it composes two parity-tested
translation units (``dfn2_process.c`` for the ERB/complex feature front end and the compose
stage, ``dfn2_model_io.c`` for the graph feature windows) behind one object, and it is
frequency-domain only. The gate is that the class reproduces the hand-composed path SAMPLE FOR
SAMPLE, with the framing the application owns (AcStft / AcIstft, tests/ainr_test_hop.h) around
both: every comparison is ``memcmp`` on fp32, never a tolerance.

Cases, all in ONE driver selected by argv[1] (audiolib and the three model TUs
are compiled once for the module):

  equiv     the class (spectrum entries, hop-in / hop-out through the host's framing) vs
            analysis + features + model-I/O + compose + synthesis, hand-composed from ONLY
            the two composed TUs (never dfn2_prepost.c). Run twice: attenuation limit off
            and at -20 dB.
  freq      the class driven by a framing of the caller's own on a separate instance vs the
            same class behind the wrapper; warm-up reports valid == 0 with an all-zero spectrum
  freqpool  the stale-pool gate accepts the matching MemReq and refuses a perturbed one, a
            grid that differs in any one value, and a different tau
  lifecycle _create/_destroy against _init on a caller pool
  reject    _get_mem_size / _init / stage-call reject-first validation, and the
            run-time refusals of a grid: dfn2_grid_make() status per failing
            value, a grid built by hand with a derived field edited, and every object
            that takes a grid
  reset     _reset really clears state (a re-run reproduces the first run)
  skip      _frame_bypass is the exact identity, never steps the state itself and
            leaves it as the accelerator wrote it
  guard     the four frame-state-machine gates the contract block promises
  txn       a refused commit zeroes the two recurrent tensors (written in place,
            so there is nothing to roll back to), keeps the frame open and leaves
            windows and clocks alone, proved for each of the five accelerator
            output tensors
  state_inplace  one buffer per recurrent tensor: input and `*_next` views are
            the same address on every frame
  inplace_nan    a NaN injected through the in-place pointer makes commit refuse
            and leaves both tensors zero
  generation     the state generation is 1 after init and advances once per reset,
            once per warm-up frame and once per refused commit, never on commit or
            skip of a real frame
  warmup    the first mask_lookahead frames are opened like any other; their heads
            are discarded, the registered state is zeroed, the generation advances
            by one each and nothing is emitted
  warmup_synthesis  the synthesis may be pushed on every hop

The recurrent state belongs to the application: the driver registers two float buffers per
instance (h_gru and the pathway window) and the instance zeroes and finite-checks them. The
accelerator stand-in is deterministic and input-dependent, so a class that dropped a state
advance would diverge rather than agree.

The library holds no grid: the driver takes its grid from ``tests/dfn2_test_grid.h``
(the 48 kHz grid, whose ERB border table it carries), builds it once with
``dfn2_grid_make()`` and hands that ``DFN2Grid`` to every object it creates.
The header's macros only size the driver's own buffers; main() checks them
against the grid and against the library's ``dfn2_prepost_*_elements()``.

The driver is built for the shipped (mask, df) lookahead pair with no override
and, in the last tests, for every other pair in ``dfn_pairs.c_build_pairs`` with
-DDFN2_MASK_LOOKAHEAD / -DDFN2_DF_LOOKAHEAD: every warm-up length,
readiness count and delay-line length in it follows the pair in the grid. The
pool identity (build_flags_hash) is checked to differ between pairs, and a pool
of another pair is refused by _init_ex. Inside one binary the same refusal is
checked for grids that differ in any single value (frequency, frame size, band
count, deep-filter bins, either lookahead, either time constant), for the pool
identity and for the graph descriptor.
"""

import re
import subprocess

import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
_SOURCES = ('prepost', 'process', 'model_io')

_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests/dfn2_test_grid.h"   /* the grid this program runs, and the library header */
#include "utility/mem_align.h"
#include "tests/ainr_test_hop.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define HOPS        400
#define SHORT_HOPS  40
#define LOOKAHEAD   (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

/* Flat element counts of the accelerator tensors. The library gives the window, mask and
 * coefficient counts as functions of the grid; the arrays below need them as constants, so
 * they are written out here from the test grid's macros and main() checks them against
 * dfn2_prepost_*_elements(dfn2_test_grid()). The recurrent state is the host's: its counts
 * are the test grid's table. */
#define TEST_ERB_WINDOW_ELEMENTS  (DFN2_MODEL_INPUT_FRAMES * DFN2_N_ERB)
#define TEST_SPEC_WINDOW_ELEMENTS (2 * DFN2_MODEL_INPUT_FRAMES * DFN2_DF_BINS)
#define TEST_ERB_MASK_ELEMENTS    (1 * DFN2_N_ERB)
#define TEST_CONVP_WINDOW_ELEMENTS  DFN2_TEST_CONVP_ELEMENTS
#define TEST_GRU_ELEMENTS           DFN2_TEST_GRU_ELEMENTS
#define TEST_ALPHA_ELEMENTS         1
#define TEST_COEFS_ELEMENTS \
    (1 * DFN2_DF_BINS * DFN2_DF_ORDER * 2)

/* What pre_process returns on stream hop t of a cold instance: 1 on every hop. The first
 * DFN2_MASK_LOOKAHEAD hops are warm-up frames (their heads belong to a frame before the
 * stream start and are discarded); the window of heads[0] is complete on hop
 * DFN2_MASK_LOOKAHEAD. */
#define PRE_RET(t)  1

/* skip: frames driven purely on frame_bypass, then five more for the freeze. */
#define SKIP_FRAMES 24
#define SKIP_TOTAL  (SKIP_FRAMES + 5)

/* txn: K normal hops, the poisoned hop, then enough hops for the divergence
 * against the "committed at K" reference to be unmissable. */
#define TXN_PRE   20
#define TXN_POST  40
#define TXN_HOPS  (TXN_PRE + 1 + TXN_POST)

#define N_OUT_TENSORS 5

/* ---- synthetic ERB matrices ------------------------------------------
 * Built exactly the way tests/test_c_prepost.c's dfn_test_build_erb() does, over the band
 * border table of dfn2_test_band_borders() (dfn2_test_grid.h; the shipped 48 kHz / 1024 / 32
 * table on this driver's grid).  erb_inv is a partition of unity per bin, which is what makes
 * frame_bypass an EXACT identity rather than an approximate one -- main() asserts that property
 * before any case runs, in the same accumulation order df_common_expand_mask() uses. */
_Static_assert(DFN2_N_FFT == 1024 && DFN2_N_ERB == 32,
               "this driver is built for the shipped grid and only varies the lookahead pair");
static int erb_borders[DFN2_N_ERB];
static float erb_fwd[DFN2_N_BINS][DFN2_N_ERB];
static float erb_inv[DFN2_N_ERB][DFN2_N_BINS];

static void build_erb(void) {
    int segment = 0;
    int k;
    dfn2_test_band_borders(erb_borders);
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
 * hand-composed DFN2ModelIOState.  Every element of all five outputs is
 * written, as the commit contract demands, and every value depends on both
 * feature windows and on both recurrent tensors, so a dropped state
 * advance shows up as a divergence rather than as agreement. */
static double fake_acc(const float *erb_window, const float *spec_window,
                       const float *gru_hidden, const float *convp) {
    const DFN2Grid *grid = dfn2_test_grid();
    double acc = 0.0;
    size_t i;
    for (i = 0; i < dfn2_prepost_erb_window_elements(grid); ++i)
        acc += erb_window[i];
    for (i = 0; i < dfn2_prepost_spec_window_elements(grid); ++i)
        acc += 0.5 * spec_window[i];
    for (i = 0; i < (size_t)TEST_GRU_ELEMENTS; ++i)
        acc += 1e-3 * gru_hidden[i];
    for (i = 0; i < (size_t)TEST_CONVP_WINDOW_ELEMENTS; ++i)
        acc += 1e-4 * convp[i];
    return acc;
}

static void fake_fill(double acc, float *erb_mask, float *coefs, float *alpha,
                      float *gru_next, float *convp_next) {
    const DFN2Grid *grid = dfn2_test_grid();
    size_t i;
    /* Kept inside (0,1) so it is a plausible sigmoid mask, not a value the
     * compose stage would clamp. */
    for (i = 0; i < dfn2_prepost_erb_mask_elements(grid); ++i)
        erb_mask[i] = (float)(0.5 + 0.4 * sin(acc + (double)i));
    for (i = 0; i < dfn2_prepost_coefs_elements(grid); ++i)
        coefs[i] = (float)(0.05 * cos(acc + 1.0 + (double)i));
    for (i = 0; i < (size_t)TEST_ALPHA_ELEMENTS; ++i)
        alpha[i] = (float)(0.5 + 0.3 * sin(acc + 2.0));
    for (i = 0; i < (size_t)TEST_GRU_ELEMENTS; ++i)
        gru_next[i] = (float)(0.5 * sin(acc + 3.0 + (double)i));
    for (i = 0; i < (size_t)TEST_CONVP_WINDOW_ELEMENTS; ++i)
        convp_next[i] = (float)(0.1 * cos(acc + 5.0 + (double)i));
}

static void fake_run(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    double acc = fake_acc(in->erb_window,
                          in->spec_window,
                          in->gru_hidden,
                          in->df_convp_window);
    fake_fill(acc, out->erb_mask, out->coefs, out->alpha,
              out->gru_hidden_next,
              out->df_convp_window_next);
}

/* The three heads, laid out for the hand-composed reference path. */
typedef struct {
    float erb_mask[TEST_ERB_MASK_ELEMENTS];
    float coefs[TEST_COEFS_ELEMENTS];
    float alpha[TEST_ALPHA_ELEMENTS];
} RefHeads;

/* The reference accelerator writes the recurrent state in place too: fake_acc
 * reads every state input before fake_fill overwrites any of them. */
static void fake_run_reference(DFN2ModelIOState *io, RefHeads *heads) {
    double acc = fake_acc(io->erb_window,
                          io->spec_window,
                          io->gru_hidden,
                          io->df_convp_window);
    fake_fill(acc, heads->erb_mask, heads->coefs, heads->alpha,
              io->gru_hidden, io->df_convp_window);
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

/* Only the three heads are NaN-prefilled; the recurrent state is the
 * accelerator's input and is never prefilled. */
static int heads_all_nan(const DFN2PrepostOutputs *out) {
    return all_nan(out->erb_mask, out->erb_mask_elements) &&
           all_nan(out->coefs, out->coefs_elements) &&
           all_nan(out->alpha, out->alpha_elements);
}

/* Each recurrent state output is the same buffer as its input. */
static int state_aliased(const DFN2PrepostInputs *in,
                         const DFN2PrepostOutputs *out) {
    return (const void *)in->gru_hidden ==
               (const void *)out->gru_hidden_next &&
           (const void *)in->df_convp_window ==
               (const void *)out->df_convp_window_next;
}

static int state_all_zero(const DFN2PrepostInputs *in) {
    return all_zero(in->gru_hidden, in->gru_hidden_elements) &&
           all_zero(in->df_convp_window,
                    in->df_convp_window_elements);
}

/* One warm-up frame through the stand-in, on an open frame: the stand-in writes the state, finish
 * discards the heads and leaves the registered state zero. 0 when both held. */
static int run_warm_frame(DFN2Prepost *p) {
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    if (dfn2_prepost_frame_open(p, &in, &out) != 0) return -1;
    fake_run(&in, &out);
    if (!any_nonzero(in.gru_hidden, in.gru_hidden_elements)) return -1;
    return dfn2_prepost_frame_finish(p) == 0 && state_all_zero(&in) ? 0 : -1;
}

/* One of the five accelerator output tensors, by index, for the txn sweep. */
static float *output_tensor(DFN2PrepostOutputs *out, int which,
                            size_t *count) {
    switch (which) {
    case 0: *count = out->erb_mask_elements; return out->erb_mask;
    case 1: *count = out->coefs_elements;    return out->coefs;
    case 2: *count = out->alpha_elements;    return out->alpha;
    case 3: *count = out->gru_hidden_elements;
            return out->gru_hidden_next;
    default: *count = out->df_convp_window_elements;
            return out->df_convp_window_next;
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

/* ---- the host's recurrent state and the framing it owns ----------------
 * The library holds no framing and no recurrent state: every instance below registers a
 * pair of float buffers (h_gru and the pathway window) and, for the hop-in / hop-out
 * entries the cases drive, the driver runs AcStft / AcIstft around it (ainr_test_hop.h).
 * The buffers come from a ring of arena slots, poisoned before each start so an init that
 * leaves a registered element unzeroed is visible; an instance in use is never more than a
 * few starts old. */
typedef struct {
    float gru[DFN2_TEST_GRU_ELEMENTS];
    float convp[DFN2_TEST_CONVP_ELEMENTS];
} StateBufs;

#define N_ARENA 64
static StateBufs arena[N_ARENA];
static int arena_next;

static StateBufs *next_bufs(void) {
    StateBufs *b = &arena[arena_next++ % N_ARENA];
    memset(b, 0xA5, sizeof *b);
    return b;
}

static void register_state(DFN2PrepostConfig *cfg, StateBufs *b) {
    cfg->gru_hidden = b->gru;
    cfg->gru_hidden_elements = DFN2_TEST_GRU_ELEMENTS;
    cfg->df_convp_history = b->convp;
    cfg->df_convp_history_elements = DFN2_TEST_CONVP_ELEMENTS;
}

/* The hop-in / hop-out framing of an instance, opened on first use and dropped when the
 * instance is (re)started or stopped. */
typedef struct {
    DFN2Prepost *p;
    AinrTestHop hop;
} HopSlot;
static HopSlot hop_slots[16];

static void hop_forget(DFN2Prepost *p) {
    int i;
    for (i = 0; i < 16; ++i)
        if (hop_slots[i].p == p) {
            ainr_test_hop_close(&hop_slots[i].hop);
            hop_slots[i].p = NULL;
        }
}

static AinrTestHop *hop_of(DFN2Prepost *p) {
    int i;
    for (i = 0; i < 16; ++i)
        if (hop_slots[i].p == p) return &hop_slots[i].hop;
    for (i = 0; i < 16; ++i)
        if (hop_slots[i].p == NULL) {
            const DFN2Grid *g = dfn2_test_grid();
            if (ainr_test_hop_open(&hop_slots[i].hop, g->n_fft, g->stft_scale,
                                   g->stft_unscale) != 0) abort();
            hop_slots[i].p = p;
            return &hop_slots[i].hop;
        }
    abort();
}

/* _init / _init_ex / _create with the state registered (a copy of the caller's config, so
 * a refused call is refused for the config and not for the registration). */
static DFN2Prepost *start_ex(void *pool, size_t bytes, const DFN2PrepostConfig *cfg,
                             const DFN2PrepostMemReq *expected) {
    DFN2PrepostConfig c = *cfg;
    DFN2Prepost *p;
    register_state(&c, next_bufs());
    p = dfn2_prepost_init_ex(pool, bytes, &c, expected);
    if (p) hop_forget(p);
    return p;
}

static DFN2Prepost *start(void *pool, size_t bytes, const DFN2PrepostConfig *cfg) {
    return start_ex(pool, bytes, cfg, NULL);
}

static DFN2Prepost *make_class(const DFN2PrepostConfig *cfg) {
    DFN2PrepostConfig c = *cfg;
    DFN2Prepost *p;
    register_state(&c, next_bufs());
    p = dfn2_prepost_create(&c);
    if (p) hop_forget(p);
    return p;
}

static void stop(DFN2Prepost *p) {
    hop_forget(p);
    dfn2_prepost_destroy(p);
}

/* reset of the instance and of the framing the host owns around it. */
static void treset(DFN2Prepost *p) {
    int i;
    dfn2_prepost_reset(p);
    for (i = 0; i < 16; ++i)
        if (p && hop_slots[i].p == p) {
            ac_stft_reset(hop_slots[i].hop.stft);
            ac_istft_reset(hop_slots[i].hop.istft);
        }
}

/* hop in: one analysis frame into the instance (the _freq entry); returns its result. */
static int tpre(DFN2Prepost *p, const float *pcm) {
    static float re[DFN2_N_BINS], im[DFN2_N_BINS];
    AinrTestHop *h = hop_of(p);
    ainr_test_hop_analysis(h, pcm, re, im);
    return dfn2_prepost_pre_process_freq(p, re, im);
}

/* hop out: the next enhanced frame through the synthesis, a silent hop while warming up;
 * `*written` is the hop length when a frame was emitted, else 0. */
static int tpost(DFN2Prepost *p, float *out, int *written) {
    static float re[DFN2_N_BINS], im[DFN2_N_BINS];
    AinrTestHop *h = hop_of(p);
    int valid = 0;
    if (dfn2_prepost_post_process_freq(p, re, im, &valid) != 0) return -1;
    if (valid) ainr_test_hop_synthesis(h, re, im, out);
    else memset(out, 0, (size_t)DFN2_HOP_LEN * sizeof(float));
    if (written) *written = valid ? DFN2_HOP_LEN : 0;
    return 0;
}

static void config_freq(DFN2PrepostConfig *cfg) {
    if (dfn2_prepost_config_defaults(cfg, dfn2_test_grid()) != 0)
        abort();
    cfg->erb_fwd = &erb_fwd[0][0];
    cfg->erb_inv = &erb_inv[0][0];
}

static float pcm_in[HOPS * DFN2_HOP_LEN];
static float out_a[HOPS * DFN2_HOP_LEN];
static float out_b[HOPS * DFN2_HOP_LEN];
static float out_c[HOPS * DFN2_HOP_LEN];
static float out_d[HOPS * DFN2_HOP_LEN];
static int written_log[HOPS];
static long long frame_log[HOPS];

/* ---- the canonical per-hop loop from dfn2_prepost.h -------------------
 * `n` is 1 on every hop, the first DFN2_MASK_LOOKAHEAD of a cold instance being warm-up
 * frames -- never 0 or 2 -- so that is asserted per hop rather than merely looped over. */
static int drive_time_ex(DFN2Prepost *p, const float *in, float *out,
                         int hops, int cold, int *written_out,
                         long long *frame_out) {
    int hop;
    for (hop = 0; hop < hops; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n, written = -1;
        long long frame = -1;
        n = tpre(p, in + (size_t)hop * DFN2_HOP_LEN);
        if (n != (cold ? PRE_RET(hop) : 1)) return -1;
        if (n == 1) {
            if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0)
                return -1;
            fake_run(&inputs, &outputs);
            if (dfn2_prepost_frame_finish(p) != 0) return -1;
        }
        if (tpost(p, out + (size_t)hop * DFN2_HOP_LEN,
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

/* Drive a cold instance, through the host framing, through its warm-up hops (frames whose heads
 * are discarded) and the first hop with usable heads, left open. Returns the index of the
 * next hop, or -1 if the instance disagreed. */
static int open_first_frame_time(DFN2Prepost *p) {
    int t;
    for (t = 0; t < DFN2_MASK_LOOKAHEAD; ++t) {
        float hop_out[DFN2_HOP_LEN];
        if (tpre(p, pcm_in + (size_t)t * DFN2_HOP_LEN) != 1) return -1;
        if (run_warm_frame(p) != 0 || tpost(p, hop_out, NULL) != 0) return -1;
    }
    if (tpre(p, pcm_in + (size_t)t * DFN2_HOP_LEN) != 1)
        return -1;
    return t + 1;
}

/* The spectrum-entry twin, feeding the same spectrum on every hop. */
static int open_first_frame_freq(DFN2Prepost *p, const float *re,
                                 const float *im) {
    int t;
    for (t = 0; t < DFN2_MASK_LOOKAHEAD; ++t) {
        float out_re[DFN2_N_BINS], out_im[DFN2_N_BINS];
        if (dfn2_prepost_pre_process_freq(p, re, im) != 1) return -1;
        if (run_warm_frame(p) != 0 ||
            dfn2_prepost_post_process_freq(p, out_re, out_im, NULL) != 0) return -1;
    }
    return dfn2_prepost_pre_process_freq(p, re, im) == 1 ? t + 1 : -1;
}

/* ---- equiv: the class vs the hand-composed path ---------------------- */

static int equiv_once(float atten_lim_db) {
    const DFN2Grid *grid = dfn2_test_grid();
    static DFN2State ref_st;
    AinrTestHop ref_hop;
    static DFN2ModelIOState ref_io;
    static float ref_gru[DFN2_TEST_GRU_ELEMENTS], ref_convp[DFN2_TEST_CONVP_ELEMENTS];
    static RefHeads heads;
    static float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    static float feat_erb[DFN2_N_ERB], feat_spec[2 * DFN2_DF_BINS];
    static float enh_re[DFN2_N_BINS], enh_im[DFN2_N_BINS];
    const size_t state_bytes = dfn2_state_mem_size(grid);
    const size_t io_bytes = dfn2_model_io_mem_size(grid);
    void *state_block = alloc_aligned(16u, state_bytes);
    void *io_block = alloc_aligned(16u, io_bytes);
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;
    int hop;

    /* path A -- analysis, features, the model-I/O transaction, the compose
     * stage and the synthesis, wired by hand from the two TUs the class
     * merely orchestrates.  dfn2_prepost.c is deliberately not involved. */
    CHECK(state_block != NULL && io_block != NULL);
    CHECK(ainr_test_hop_open(&ref_hop, grid->n_fft, grid->stft_scale, grid->stft_unscale) == 0);
    CHECK(dfn2_state_init(&ref_st, state_block, state_bytes, grid) == 0);
    dfn2_set_erb_matrices(&ref_st, &erb_fwd[0][0], &erb_inv[0][0]);
    CHECK(dfn2_model_io_init(&ref_io, io_block, io_bytes, grid, ref_gru,
                             DFN2_TEST_GRU_ELEMENTS, ref_convp,
                             DFN2_TEST_CONVP_ELEMENTS) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        float *slot = out_a + (size_t)hop * DFN2_HOP_LEN;
        int needed, emitted;
        ainr_test_hop_analysis(&ref_hop, pcm_in + (size_t)hop * DFN2_HOP_LEN,
                               spec_re, spec_im);
        dfn2_compute_features(&ref_st, spec_re, spec_im, feat_erb, feat_spec);
        needed = dfn2_model_io_push_features(&ref_io, feat_erb, feat_spec);
        CHECK(needed >= 0);
        if (needed == 0) {
            /* Left warm-up: the graph has no right-hand neighbour yet, so
             * the compose clock advances with heads_valid = 0. */
            emitted = dfn2_enhance_spectrum_stream(&ref_st, spec_re, spec_im, 0,
                                          NULL, NULL, 0.0f, atten_lim_db,
                                          enh_re, enh_im, NULL);
        } else {
            fake_run_reference(&ref_io, &heads);
            emitted = dfn2_enhance_spectrum_stream(&ref_st, spec_re, spec_im, 1,
                                          heads.erb_mask, heads.coefs,
                                          heads.alpha[0], atten_lim_db,
                                          enh_re, enh_im, NULL);
            CHECK(dfn2_model_io_validate_state(&ref_io) == 0);
        }
        CHECK(emitted >= 0);
        if (emitted == 1) {
            ainr_test_hop_synthesis(&ref_hop, enh_re, enh_im, slot);
        } else {
            memset(slot, 0, (size_t)DFN2_HOP_LEN * sizeof(float));
        }
    }
    ainr_test_hop_close(&ref_hop);

    /* path B -- the class, with the same framing around it */
    config_freq(&cfg);
    cfg.atten_lim_db = atten_lim_db;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = start_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_hop_size(p) == DFN2_HOP_LEN);
    CHECK(dfn2_prepost_num_bins(p) == DFN2_N_BINS);
    CHECK(dfn2_prepost_grid(p) != NULL);
    CHECK(memcmp(dfn2_prepost_grid(p), dfn2_test_grid(), sizeof(DFN2Grid)) == 0);
    CHECK(dfn2_prepost_layout_version(p) == DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == LOOKAHEAD);
    CHECK(drive_time_ex(p, pcm_in, out_b, HOPS, 1,
                        written_log, frame_log) == 0);

    printf("equiv: atten_lim_db=%.1f pool=%llu B\n",
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

    stop(p);
    free(pool);
    free(state_block);
    free(io_block);
    return 0;
}

static int case_equiv(void) {
    /* Once with the attenuation limit disabled, once with it engaged, so the
     * limit path inside the compose stage is covered by the same identity. */
    if (equiv_once(0.0f) != 0) return 1;
    if (equiv_once(-20.0f) != 0) return 1;
    return 0;
}

/* ---- freq: the entries driven by the caller's own transform ----------- */

static int case_freq(void) {
    AinrTestHop caller_hop;
    static float spec_re[DFN2_N_BINS], spec_im[DFN2_N_BINS];
    static float enh_re[DFN2_N_BINS], enh_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p_wrap, *p_freq;
    void *pool_wrap, *pool_freq;
    int hop;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool_wrap = alloc_aligned(req.alignment, (size_t)req.bytes);
    pool_freq = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool_wrap != NULL && pool_freq != NULL);
    p_wrap = start(pool_wrap, (size_t)req.bytes, &cfg);
    p_freq = start(pool_freq, (size_t)req.bytes, &cfg);
    CHECK(p_wrap != NULL && p_freq != NULL);

    CHECK(drive_time(p_wrap, pcm_in, out_a, HOPS) == 0);

    /* The caller owns the transform on both sides: one analysis per hop into the class,
     * one synthesis per emitted frame out of it, on a framing of its own. */
    CHECK(ainr_test_hop_open(&caller_hop, DFN2_N_FFT, dfn2_test_grid()->stft_scale,
                             dfn2_test_grid()->stft_unscale) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        float *slot = out_b + (size_t)hop * DFN2_HOP_LEN;
        int n, valid = -1;
        ainr_test_hop_analysis(&caller_hop, pcm_in + (size_t)hop * DFN2_HOP_LEN,
                               spec_re, spec_im);
        n = dfn2_prepost_pre_process_freq(p_freq, spec_re, spec_im);
        CHECK(n == PRE_RET(hop));
        if (n == 1) {
            CHECK(dfn2_prepost_frame_open(p_freq, &inputs, &outputs) == 0);
            fake_run(&inputs, &outputs);
            CHECK(dfn2_prepost_frame_finish(p_freq) == 0);
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
            ainr_test_hop_synthesis(&caller_hop, enh_re, enh_im, slot);
        }
    }
    ainr_test_hop_close(&caller_hop);

    CHECK(any_nonzero(out_a, (size_t)HOPS * DFN2_HOP_LEN));
    CHECK(!identical(out_a, out_b + 1, (size_t)HOPS * DFN2_HOP_LEN - 1));
    CHECK(identical(out_a, out_b, (size_t)HOPS * DFN2_HOP_LEN));

    stop(p_wrap);
    stop(p_freq);
    free(pool_wrap);
    free(pool_freq);
    return 0;
}

/* ---- grid variants ---------------------------------------------------- */

/* The library holds no grid, so a different grid is a different VALUE, not a different
 * build: these are valid grids that differ from the test grid in exactly one value, made
 * with dfn2_grid_make() inside the same binary (the two time constants stand for a model
 * trained with another tau). Every object that records the grid must tell them apart. */
enum {
    VARIANT_SAMPLE_RATE, VARIANT_N_FFT, VARIANT_N_ERB, VARIANT_DF_BINS,
    VARIANT_MASK_LOOKAHEAD, VARIANT_DF_LOOKAHEAD, VARIANT_ERB_TAU,
    VARIANT_SPEC_TAU, N_VARIANTS
};

static int make_variant(DFN2Grid *grid, int which) {
    const DFN2Grid *base = dfn2_test_grid();
    DFN2GridSpec spec = dfn2_test_grid_spec();
    switch (which) {
    case VARIANT_SAMPLE_RATE:    spec.sample_rate += 1000; break;
    case VARIANT_N_FFT:          spec.n_fft /= 2; break;
    case VARIANT_N_ERB:          spec.n_erb -= 4; break;
    case VARIANT_DF_BINS:        spec.df_bins -= 4; break;
    case VARIANT_MASK_LOOKAHEAD: spec.mask_lookahead = base->mask_lookahead > 0
                                     ? base->mask_lookahead - 1 : 1; break;
    case VARIANT_DF_LOOKAHEAD:   spec.df_lookahead = base->df_lookahead > 0
                                     ? base->df_lookahead - 1 : 1; break;
    case VARIANT_ERB_TAU:        spec.erb_norm_tau_sec = 2.0f; break;
    case VARIANT_SPEC_TAU:       spec.spec_norm_tau_sec = 2.0f; break;
    default: break;
    }
    if (dfn2_grid_make(grid, &spec) != DFN2_GRID_OK) return -1;
    /* The alphas follow the sample rate, the hop and the taus (dfn2_norm_alpha), so a
     * variant of the rate or the frame size moves them with it. */
    return memcmp(grid, base, sizeof *grid) != 0 &&
           dfn2_grid_validate(grid) == DFN2_GRID_OK ? 0 : -1;
}

/* ---- freqpool: the stale-pool gate ------------------------------------ */

/* The pool identity records the whole grid, and the grid arrives at run time: a
 * MemReq sized for a grid that differs in any one value is refused by init_ex() in this
 * very binary, by identity alone (size and alignment forged equal), while each one is
 * accepted for its own grid. A value dropped from the hash makes the matching variant
 * pass here. */
static int freqpool_grid_identity(const DFN2PrepostConfig *cfg_freq,
                                  const DFN2PrepostMemReq *req_freq) {
    DFN2PrepostConfig cfg_v;
    DFN2PrepostMemReq req_v[N_VARIANTS], forged;
    uint64_t most = req_freq->bytes;
    void *pool;
    unsigned char *before;
    int v, w;

    for (v = 0; v < N_VARIANTS; ++v) {
        DFN2Grid grid;
        CHECK(make_variant(&grid, v) == 0);
        CHECK(dfn2_prepost_config_defaults(&cfg_v, &grid) == 0);
        cfg_v.erb_fwd = cfg_freq->erb_fwd;
        cfg_v.erb_inv = cfg_freq->erb_inv;
        CHECK(dfn2_prepost_get_mem_size(&cfg_v, &req_v[v]) == 0);
        CHECK(req_v[v].build_flags_hash != req_freq->build_flags_hash);
        for (w = 0; w < v; ++w)
            CHECK(req_v[v].build_flags_hash != req_v[w].build_flags_hash);
        if (req_v[v].bytes > most) most = req_v[v].bytes;
    }
    pool = alloc_aligned(req_freq->alignment, (size_t)most);
    before = malloc((size_t)most);
    CHECK(pool != NULL && before != NULL);
    for (v = 0; v < N_VARIANTS; ++v) {
        DFN2Grid grid;
        CHECK(make_variant(&grid, v) == 0);
        CHECK(dfn2_prepost_config_defaults(&cfg_v, &grid) == 0);
        cfg_v.erb_fwd = cfg_freq->erb_fwd;
        cfg_v.erb_inv = cfg_freq->erb_inv;
        /* accepted for its own grid... */
        CHECK(start_ex(pool, (size_t)most, &cfg_v, &req_v[v]) != NULL);
        memcpy(before, pool, (size_t)most);
        /* ...refused for the test grid, and the other way round... */
        CHECK(start_ex(pool, (size_t)most, cfg_freq, &req_v[v]) == NULL);
        CHECK(start_ex(pool, (size_t)most, &cfg_v, req_freq) == NULL);
        /* ...also when the request is given the test grid's size and alignment */
        forged = req_v[v];
        forged.bytes = req_freq->bytes;
        forged.alignment = req_freq->alignment;
        CHECK(start_ex(pool, (size_t)most, cfg_freq, &forged) == NULL);
        /* ...and a refusal writes nothing: the live instance's pool is byte-identical */
        CHECK(memcmp(pool, before, (size_t)most) == 0);
    }
    free(before);
    free(pool);
    return 0;
}

static int case_freqpool(void) {
    DFN2PrepostConfig cfg_freq;
    DFN2PrepostMemReq req_freq;
    void *pool_freq;

    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    printf("pool = %llu B\n", (unsigned long long)req_freq.bytes);

    CHECK(req_freq.descriptor_version == DFN2_PREPOST_DESCRIPTOR_VERSION);
    CHECK(req_freq.layout_version == (uint32_t)DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(req_freq.alignment == DFN2_PREPOST_ALIGNMENT);
    CHECK(req_freq.reserved0 == 0u);
    CHECK(req_freq.reserved == 0u);

    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_freq != NULL);

    /* The gate must be able to PASS, or every refusal below is vacuous. */
    CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &req_freq) != NULL);
    {
        DFN2PrepostMemReq stale;
        unsigned char *live = (unsigned char *)malloc((size_t)req_freq.bytes);
        CHECK(live != NULL);
        memcpy(live, pool_freq, (size_t)req_freq.bytes);
        stale = req_freq; ++stale.descriptor_version;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        stale = req_freq; ++stale.layout_version;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        stale = req_freq; ++stale.build_flags_hash;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        stale = req_freq; ++stale.bytes;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        stale = req_freq; stale.alignment *= 2u;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        stale = req_freq; stale.reserved0 = 1u;
        CHECK(start_ex(pool_freq, (size_t)req_freq.bytes, &cfg_freq, &stale) == NULL);
        /* a refusal writes nothing: the pool of the instance started above is byte-identical */
        CHECK(memcmp(pool_freq, live, (size_t)req_freq.bytes) == 0);
        free(live);
    }
    CHECK(freqpool_grid_identity(&cfg_freq, &req_freq) == 0);
    free(pool_freq);
    return 0;
}

/* ---- dual: the dual-input FREQ entry point ---------------------------- */

/* Drive one FREQ hop: pre-stage by the caller, then the model transaction. */
static int dual_step(DFN2Prepost *p, int n, float *out_re, float *out_im,
                     int *valid) {
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    if (n == 1) {
        if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0) return -1;
        fake_run(&inputs, &outputs);
        if (dfn2_prepost_frame_finish(p) != 0) return -1;
    }
    return dfn2_prepost_post_process_freq(p, out_re, out_im, valid);
}

/* est == app must be the single-input entry, byte for byte, including the
 * head bin gain the class publishes and the emitted frame index. */
static int case_dual_identity(void) {
    static float in_re[SHORT_HOPS][DFN2_N_BINS];
    static float in_im[SHORT_HOPS][DFN2_N_BINS];
    static float got_a_re[DFN2_N_BINS], got_a_im[DFN2_N_BINS];
    static float got_b_re[DFN2_N_BINS], got_b_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *pa, *pb;
    void *pool_a, *pool_b;
    int t, emitted = 0;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool_a = alloc_aligned(req.alignment, (size_t)req.bytes);
    pool_b = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool_a != NULL && pool_b != NULL);
    pa = start(pool_a, (size_t)req.bytes, &cfg);
    pb = start(pool_b, (size_t)req.bytes, &cfg);
    CHECK(pa != NULL && pb != NULL);
    for (t = 0; t < SHORT_HOPS; ++t) frame_spectrum(t, in_re[t], in_im[t]);

    for (t = 0; t < SHORT_HOPS; ++t) {
        int na, nb, va = -1, vb = -1;
        long long fa = -2, fb = -2;
        na = dfn2_prepost_pre_process_freq(pa, in_re[t], in_im[t]);
        nb = dfn2_prepost_pre_process_freq_dual(pb, in_re[t], in_im[t],
                                                in_re[t], in_im[t]);
        CHECK(na == PRE_RET(t) && nb == na);
        CHECK(dual_step(pa, na, got_a_re, got_a_im, &va) == 0);
        CHECK(dual_step(pb, nb, got_b_re, got_b_im, &vb) == 0);
        CHECK(va == vb);
        CHECK(identical(got_a_re, got_b_re, DFN2_N_BINS));
        CHECK(identical(got_a_im, got_b_im, DFN2_N_BINS));
        (void)dfn2_prepost_output_frame_index(pa, &fa);
        (void)dfn2_prepost_output_frame_index(pb, &fb);
        CHECK(fa == fb);
        /* The heads are a real mask, not the identity: an emitted frame must
         * differ from the source spectrum it was composed from, otherwise
         * single == dual would hold for a class that ignores its heads. */
        if (va == 1)
            CHECK(!identical(got_a_re, in_re[t - LOOKAHEAD], DFN2_N_BINS));
        if (va == 1) ++emitted;
    }
    CHECK(emitted == SHORT_HOPS - LOOKAHEAD);
    stop(pa);
    stop(pb);
    free(pool_a);
    free(pool_b);
    return 0;
}

/* The directional gate: app = est with the high band zeroed.  The emitted
 * high band must be exactly zero (it comes from app), the low band and the
 * published gain must equal the single-input run's (the mask follows est,
 * and the deep filter reads only low bins, which the two spectra share).
 * A build that swapped the two arguments would emit est's non-zero high band
 * and, because the features would see the zeroed spectrum, a different
 * mask. */
static int case_dual_split(void) {
    static float est_re[SHORT_HOPS][DFN2_N_BINS];
    static float est_im[SHORT_HOPS][DFN2_N_BINS];
    static float app_re[DFN2_N_BINS], app_im[DFN2_N_BINS];
    static float got_s_re[DFN2_N_BINS], got_s_im[DFN2_N_BINS];
    static float got_d_re[DFN2_N_BINS], got_d_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *ps, *pd;
    void *pool_s, *pool_d;
    int t, checked = 0;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool_s = alloc_aligned(req.alignment, (size_t)req.bytes);
    pool_d = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool_s != NULL && pool_d != NULL);
    ps = start(pool_s, (size_t)req.bytes, &cfg);
    pd = start(pool_d, (size_t)req.bytes, &cfg);
    CHECK(ps != NULL && pd != NULL);
    for (t = 0; t < SHORT_HOPS; ++t) frame_spectrum(t, est_re[t], est_im[t]);

    for (t = 0; t < SHORT_HOPS; ++t) {
        int ns, nd, vs = -1, vd = -1;
        memcpy(app_re, est_re[t], sizeof(app_re));
        memcpy(app_im, est_im[t], sizeof(app_im));
        memset(app_re + DFN2_DF_BINS, 0,
               (size_t)(DFN2_N_BINS - DFN2_DF_BINS) * sizeof(float));
        memset(app_im + DFN2_DF_BINS, 0,
               (size_t)(DFN2_N_BINS - DFN2_DF_BINS) * sizeof(float));
        ns = dfn2_prepost_pre_process_freq(ps, est_re[t], est_im[t]);
        nd = dfn2_prepost_pre_process_freq_dual(pd, est_re[t], est_im[t],
                                                app_re, app_im);
        CHECK(ns == nd && ns == PRE_RET(t));
        CHECK(dual_step(ps, ns, got_s_re, got_s_im, &vs) == 0);
        CHECK(dual_step(pd, nd, got_d_re, got_d_im, &vd) == 0);
        CHECK(vs == vd);
        if (vs == 1) {
            CHECK(any_nonzero(got_s_re + DFN2_DF_BINS,
                              DFN2_N_BINS - DFN2_DF_BINS));
            CHECK(all_zero(got_d_re + DFN2_DF_BINS,
                           DFN2_N_BINS - DFN2_DF_BINS));
            CHECK(all_zero(got_d_im + DFN2_DF_BINS,
                           DFN2_N_BINS - DFN2_DF_BINS));
            CHECK(identical(got_s_re, got_d_re, DFN2_DF_BINS));
            CHECK(identical(got_s_im, got_d_im, DFN2_DF_BINS));
            ++checked;
        }
    }
    CHECK(checked == SHORT_HOPS - LOOKAHEAD);
    stop(ps);
    stop(pd);
    free(pool_s);
    free(pool_d);
    return 0;
}

/* Contract gates around the dual entry and the gain accessor. */
static int case_dual_guards(void) {
    static float re[DFN2_N_BINS], im[DFN2_N_BINS];
    static float got_re[DFN2_N_BINS], got_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg_freq;
    DFN2PrepostMemReq req_freq;
    DFN2Prepost *pf;
    void *pool_f;

    CHECK(DFN2_PREPOST_CARVE_VERSION == 11u);

    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_f = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_f != NULL);
    pf = start(pool_f, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(pf != NULL);
    frame_spectrum(0, re, im);

    /* NULL arguments. */
    CHECK(dfn2_prepost_pre_process_freq_dual(NULL, re, im, re, im) == -1);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, NULL, im, re, im) == -1);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, NULL, re, im) == -1);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, NULL, im) == -1);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, re, NULL) == -1);
    /* The warm-up hops open a frame like any other: a second pre-stage of either kind is refused
     * with it open, the bypass closes it, and the hop emits nothing. */
    {
        int w, valid = -1;
        for (w = 0; w < DFN2_MASK_LOOKAHEAD; ++w) {
            CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, re, im) == 1);
            CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, re, im) == -1);
            CHECK(dfn2_prepost_pre_process_freq(pf, re, im) == -1);
            CHECK(dfn2_prepost_post_process_freq(pf, got_re, got_im, NULL) == -1);
            CHECK(dfn2_prepost_frame_bypass(pf) == 0);
            CHECK(dfn2_prepost_post_process_freq(pf, got_re, got_im, &valid) == 0 && valid == 0);
        }
    }
    /* The first hop with heads: a frame open refuses a second pre-stage of
     * either kind; taken as the identity the hop closes normally. */
    frame_spectrum(1, re, im);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, re, im) == 1);
    CHECK(dfn2_prepost_pre_process_freq_dual(pf, re, im, re, im) == -1);
    CHECK(dfn2_prepost_pre_process_freq(pf, re, im) == -1);
    CHECK(dfn2_prepost_frame_bypass(pf) == 0);
    CHECK(dfn2_prepost_post_process_freq(pf, got_re, got_im, NULL) == 0);

    stop(pf);
    free(pool_f);
    return 0;
}

/* ---- model_run: the DFN2Model callback boundary ------------------------ */

typedef struct {
    int mode;     /* 0 full write, 1 refuse (rc -1), 2 leave alpha unwritten
                   * after writing the state in place                       */
    int calls;
    int resets;
} StubModel;

static int stub_infer(void *user, const DFN2PrepostInputs *in,
                      DFN2PrepostOutputs *out) {
    StubModel *m = (StubModel *)user;
    ++m->calls;
    if (m->mode == 1) return -1;
    fake_run(in, out);
    if (m->mode == 2) out->alpha[0] = (float)NAN;   /* one output unwritten */
    return 0;
}

static void stub_reset(void *user) { ++((StubModel *)user)->resets; }

/* Copy the two recurrent tensors out of the published input views. */
static void snapshot_state(DFN2Prepost *p, float *gru, float *convp) {
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    (void)dfn2_prepost_frame_open(p, &in, &out);
    memcpy(gru, in.gru_hidden,
           (size_t)TEST_GRU_ELEMENTS * sizeof(float));
    memcpy(convp, in.df_convp_window,
           (size_t)TEST_CONVP_WINDOW_ELEMENTS * sizeof(float));
}

static int case_model_run(void) {
    static float in_re[SHORT_HOPS][DFN2_N_BINS];
    static float in_im[SHORT_HOPS][DFN2_N_BINS];
    static float got_re[DFN2_N_BINS], got_im[DFN2_N_BINS];
    static float ref_re[DFN2_N_BINS], ref_im[DFN2_N_BINS];
    static float gru0[TEST_GRU_ELEMENTS];
    static float convp0[TEST_CONVP_WINDOW_ELEMENTS];
    static float gru1[TEST_GRU_ELEMENTS];
    static float convp1[TEST_CONVP_WINDOW_ELEMENTS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p, *q;
    void *pool_p, *pool_q;
    StubModel stub = {0, 0, 0};
    DFN2Model model;
    DFN2Model bad;
    DFN2ModelIoDescriptor descriptor;
    int t, v, valid = -1, calls;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool_p = alloc_aligned(req.alignment, (size_t)req.bytes);
    pool_q = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool_p != NULL && pool_q != NULL);
    p = start(pool_p, (size_t)req.bytes, &cfg);
    q = start(pool_q, (size_t)req.bytes, &cfg);
    CHECK(p != NULL && q != NULL);
    for (t = 0; t < SHORT_HOPS; ++t) frame_spectrum(t, in_re[t], in_im[t]);
    memset(&model, 0, sizeof(model));
    model.user = &stub;
    model.infer = stub_infer;
    model.reset = stub_reset;
    CHECK(dfn2_model_io_descriptor_default(dfn2_test_grid(), &descriptor) == 0);
    model.io_descriptor = &descriptor;

    /* Contract errors: no frame open, NULL model, no infer callback. */
    CHECK(dfn2_model_run_frame(&model, p) == -1);
    for (t = 0; t < DFN2_MASK_LOOKAHEAD; ++t) {
        CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
        CHECK(dfn2_model_run_frame(&model, p) == 1);   /* warm-up frame: the model runs, the heads are discarded */
        CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &valid) == 0 && valid == 0);
    }
    CHECK(stub.calls == DFN2_MASK_LOOKAHEAD);
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_model_run_frame(NULL, p) == -1);
    CHECK(dfn2_model_run_frame(&model, NULL) == -1);
    memset(&bad, 0, sizeof(bad));
    CHECK(dfn2_model_run_frame(&bad, p) == -1);
    /* No published graph contract, or one of another lookahead pair: refused
     * before infer() runs, the frame stays open. */
    bad = model;
    bad.io_descriptor = NULL;
    CHECK(dfn2_model_run_frame(&bad, p) == -1);
    descriptor.mask_lookahead ^= 1;
    CHECK(dfn2_model_run_frame(&model, p) == -1);
    descriptor.mask_lookahead ^= 1;
    descriptor.df_lookahead ^= 1;
    CHECK(dfn2_model_run_frame(&model, p) == -1);
    descriptor.df_lookahead ^= 1;
    /* The instance holds its grid as a run-time value: a descriptor made for a
     * valid grid that differs in any one value is refused the same way, and a
     * descriptor of the instance's own grid is what lets it through. */
    for (v = 0; v < N_VARIANTS; ++v) {
        DFN2Grid grid;
        DFN2ModelIoDescriptor other;
        CHECK(make_variant(&grid, v) == 0);
        CHECK(dfn2_model_io_descriptor_default(&grid, &other) == 0);
        bad = model;
        bad.io_descriptor = &other;
        /* the two alphas are not part of the graph contract: only the other
         * six values can make the descriptor differ */
        if (v == VARIANT_ERB_TAU || v == VARIANT_SPEC_TAU)
            CHECK(memcmp(&other, &descriptor, sizeof other) == 0);
        else
            CHECK(dfn2_model_run_frame(&bad, p) == -1);
    }
    CHECK(stub.calls == DFN2_MASK_LOOKAHEAD);
    /* A full write commits and steps the recurrent state. */
    snapshot_state(p, gru0, convp0);
    CHECK(dfn2_model_run_frame(&model, p) == 1);
    CHECK(stub.calls == DFN2_MASK_LOOKAHEAD + 1);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &valid) == 0);
    CHECK(valid == (t >= LOOKAHEAD));   /* the lookahead warm-up emits nothing */
    /* Committed hops until the next hop emits, so the refused ones below are
     * emitting hops whatever the lookahead pair. */
    while (t + 1 < LOOKAHEAD) {
        ++t;
        CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
        CHECK(dfn2_model_run_frame(&model, p) == 1);
    }
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    snapshot_state(p, gru1, convp1);
    CHECK(!identical(gru0, gru1, TEST_GRU_ELEMENTS));
    /* A refused inference (nonzero return, nothing written) is taken as the
     * identity: state frozen, frame closed, the hop still emits. */
    stub.mode = 1;
    calls = stub.calls;
    CHECK(dfn2_model_run_frame(&model, p) == 0);
    CHECK(stub.calls == calls + 1);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &valid) == 0);
    CHECK(valid == 1);
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    snapshot_state(p, gru0, convp0);
    CHECK(identical(gru0, gru1, TEST_GRU_ELEMENTS));
    CHECK(identical(convp0, convp1, TEST_CONVP_WINDOW_ELEMENTS));
    /* A partial write (one head left unwritten) is refused by commit and
     * taken as the identity the same way -- but the accelerator already wrote
     * the state in place, so the refusal zeroes both recurrent tensors. */
    stub.mode = 2;
    calls = stub.calls;
    CHECK(dfn2_model_run_frame(&model, p) == 0);
    CHECK(stub.calls == calls + 1);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &valid) == 0);
    CHECK(valid == 1);
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    snapshot_state(p, gru0, convp0);
    CHECK(all_zero(gru0, TEST_GRU_ELEMENTS));
    CHECK(all_zero(convp0, TEST_CONVP_WINDOW_ELEMENTS));
    CHECK(!identical(gru0, gru1, TEST_GRU_ELEMENTS));
    CHECK(dfn2_prepost_frame_bypass(p) == 0);

    /* A model that always refuses is exactly the all-skip stream. */
    treset(p);
    treset(q);
    stub.mode = 1;
    for (t = 0; t < SHORT_HOPS; ++t) {
        int np, nq, vp = -1, vq = -1;
        np = dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]);
        nq = dfn2_prepost_pre_process_freq(q, in_re[t], in_im[t]);
        CHECK(np == nq);
        if (np == 1) {
            CHECK(dfn2_model_run_frame(&model, p) == 0);
            CHECK(dfn2_prepost_frame_bypass(q) == 0);
        }
        CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, &vp) == 0);
        CHECK(dfn2_prepost_post_process_freq(q, ref_re, ref_im, &vq) == 0);
        CHECK(vp == vq);
        CHECK(identical(got_re, ref_re, DFN2_N_BINS));
        CHECK(identical(got_im, ref_im, DFN2_N_BINS));
    }
    printf("model_run: calls=%d\n", stub.calls);
    stop(p);
    stop(q);
    free(pool_p);
    free(pool_q);
    return 0;
}

/* ---- descriptor: the graph contract value ----------------------------- */

static int case_descriptor(void) {
    const DFN2Grid *grid = dfn2_test_grid();
    DFN2ModelIoDescriptor base, d;
    int *fields[17];
    int i;

    CHECK(dfn2_model_io_descriptor_default(grid, NULL) == -1);
    CHECK(dfn2_model_io_descriptor_default(NULL, &base) == -1);
    CHECK(dfn2_model_io_descriptor_validate(grid, NULL) == -1);
    CHECK(dfn2_model_io_descriptor_default(grid, &base) == 0);
    CHECK(dfn2_model_io_descriptor_validate(NULL, &base) == -1);
    CHECK(dfn2_model_io_descriptor_validate(grid, &base) == 0);
    CHECK(base.layout_version == (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(base.sample_rate == DFN2_SR && base.fft_size == DFN2_N_FFT &&
          base.hop_size == DFN2_HOP_LEN && base.spectrum_bins == DFN2_N_BINS);
    CHECK(base.erb_bands == DFN2_N_ERB && base.df_bins == DFN2_DF_BINS &&
          base.df_order == DFN2_DF_ORDER);
    CHECK(base.mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          base.df_lookahead == DFN2_DF_LOOKAHEAD);
    /* The six interior fields are caller-filled bookkeeping: 0 by default. */
    CHECK(base.encoder_gru_layers == 0 && base.erb_gru_layers == 0 &&
          base.df_gru_layers == 0 && base.gru_hidden == 0 &&
          base.encoder_channels == 0 && base.df_pathway_window == 0);
    CHECK(base.state_tensor_count == 2);

    d = base;
    ++d.layout_version;
    CHECK(dfn2_model_io_descriptor_validate(grid, &d) == -1);
    fields[0] = &d.sample_rate;        fields[1] = &d.fft_size;
    fields[2] = &d.hop_size;           fields[3] = &d.spectrum_bins;
    fields[4] = &d.erb_bands;          fields[5] = &d.df_bins;
    fields[6] = &d.df_order;           fields[7] = &d.mask_lookahead;
    fields[8] = &d.df_lookahead;       fields[9] = &d.input_frames;
    fields[10] = &d.encoder_gru_layers; fields[11] = &d.erb_gru_layers;
    fields[12] = &d.df_gru_layers;     fields[13] = &d.gru_hidden;
    fields[14] = &d.encoder_channels;  fields[15] = &d.df_pathway_window;
    fields[16] = &d.state_tensor_count;
    for (i = 0; i < 17; ++i) {
        d = base;
        *fields[i] += 1;
        /* fields 10..15 are the interior ones, which validate does not read */
        CHECK(dfn2_model_io_descriptor_validate(grid, &d) ==
              (i >= 10 && i <= 15 ? 0 : -1));
    }
    d = base;
    CHECK(dfn2_model_io_descriptor_validate(grid, &d) == 0);

    /* The descriptor is a function of the grid the application hands in: for every
     * valid grid that differs in one value it is the descriptor of THAT grid, and
     * the pair of grid and descriptor must agree. A grid value left out of the
     * descriptor leaves the two descriptors equal and the cross check below
     * passing where it must be refused. */
    for (i = 0; i < N_VARIANTS; ++i) {
        DFN2Grid other;
        DFN2ModelIoDescriptor other_descriptor;
        CHECK(make_variant(&other, i) == 0);
        CHECK(dfn2_model_io_descriptor_default(&other, &other_descriptor) == 0);
        CHECK(dfn2_model_io_descriptor_validate(&other, &other_descriptor) == 0);
        if (i == VARIANT_ERB_TAU || i == VARIANT_SPEC_TAU) {
            /* the EMA coefficients are not part of the graph contract */
            CHECK(memcmp(&other_descriptor, &base, sizeof base) == 0);
            continue;
        }
        CHECK(memcmp(&other_descriptor, &base, sizeof base) != 0);
        CHECK(dfn2_model_io_descriptor_validate(grid, &other_descriptor) == -1);
        CHECK(dfn2_model_io_descriptor_validate(&other, &base) == -1);
    }

    /* A grid the library refuses has no descriptor, and a descriptor is not
     * validated against one. */
    {
        DFN2Grid broken = *grid;
        ++broken.hop;                       /* derived field does not match */
        CHECK(dfn2_model_io_descriptor_default(&broken, &d) == -1);
        CHECK(dfn2_model_io_descriptor_validate(&broken, &base) == -1);
    }
    return 0;
}

/* ---- lifecycle: _create/_destroy against _init on a caller pool ------ */

static int case_lifecycle(void) {
    DFN2PrepostConfig cfg, bad;
    DFN2PrepostMemReq req;
    DFN2Prepost *heap, *stack, *tmp;
    void *pool;
    long long frame = 0;

    config_freq(&cfg);
    heap = make_class(&cfg);
    CHECK(heap != NULL);
    CHECK(dfn2_prepost_hop_size(heap) == DFN2_HOP_LEN);
    CHECK(dfn2_prepost_num_bins(heap) == DFN2_N_BINS);
    /* The instance reports the grid it was configured with, by value. */
    CHECK(dfn2_prepost_grid(heap) != NULL);
    CHECK(dfn2_prepost_grid(heap) != dfn2_test_grid());
    CHECK(memcmp(dfn2_prepost_grid(heap), dfn2_test_grid(), sizeof(DFN2Grid)) == 0);
    CHECK(dfn2_prepost_layout_version(heap) == DFN2_MODEL_IO_LAYOUT_VERSION);
    CHECK(dfn2_prepost_model_lookahead_frames(heap) == LOOKAHEAD);
    CHECK(drive_time(heap, pcm_in, out_a, SHORT_HOPS) == 0);
    CHECK(any_nonzero(out_a, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* A created instance and a pool instance are the same object. */
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    stack = start(pool, (size_t)req.bytes, &cfg);
    CHECK(stack != NULL);
    CHECK(drive_time(stack, pcm_in, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* _destroy frees only what _create allocated, so on a pool instance it
     * is a genuine no-op and therefore idempotent -- the instance still runs
     * afterwards.  (Repeating it on a CREATED instance would be a
     * use-after-free, exactly as for fft_destroy; see fft_wrapper.h.) */
    dfn2_prepost_destroy(stack);
    dfn2_prepost_destroy(stack);
    treset(stack);
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

    stop(heap);
    stop(NULL);
    free(pool);

    /* _create refuses exactly what _get_mem_size refuses, and allocates
     * nothing when it does. */
    CHECK(dfn2_prepost_create(NULL) == NULL);
    CHECK(dfn2_prepost_config_defaults(&bad, dfn2_test_grid()) == 0);
    CHECK(make_class(&bad) == NULL);               /* no ERB               */
    bad.erb_fwd = &erb_fwd[0][0];
    CHECK(make_class(&bad) == NULL);               /* erb_inv missing      */
    bad.erb_fwd = NULL;
    bad.erb_inv = &erb_inv[0][0];
    CHECK(make_class(&bad) == NULL);               /* erb_fwd missing      */
    bad.erb_fwd = &erb_fwd[0][0];
    tmp = make_class(&bad);                        /* ...and accepts       */
    CHECK(tmp != NULL);
    stop(tmp);

    /* Accessors on NULL. */
    CHECK(dfn2_prepost_hop_size(NULL) == -1);
    CHECK(dfn2_prepost_num_bins(NULL) == -1);
    CHECK(dfn2_prepost_model_lookahead_frames(NULL) == -1);
    CHECK(dfn2_prepost_layout_version(NULL) == -1);
    CHECK(dfn2_prepost_grid(NULL) == NULL);
    CHECK(dfn2_prepost_output_frame_index(NULL, &frame) == -1);
    dfn2_prepost_reset(NULL);
    return 0;
}

/* ---- reject: reject-first validation --------------------------------- */

/* The grid is a run-time value, so everything that does not fit (a lookahead pair outside
 * the model's range, a size the framing cannot build, a hand-edited derived field) is
 * refused by the library when the value arrives: by dfn2_grid_make() with the status of
 * the first failing check, by dfn2_grid_validate() for a grid built by hand, and by every
 * object that takes a grid. */
typedef struct {
    int sample_rate, n_fft, n_erb, df_bins, mask, df;
    int want;
} GridCase;

static void tamper_grid(DFN2Grid *grid, int which) {
    *grid = *dfn2_test_grid();
    switch (which) {
    case 0: ++grid->hop; break;
    case 1: ++grid->n_bins; break;
    case 2: ++grid->df_history; break;
    case 3: ++grid->ring; break;
    case 4: grid->stft_scale *= 2.0f; break;
    case 5: grid->stft_unscale *= 2.0f; break;
    case 6: grid->erb_norm_alpha = 1.0f; break;
    case 7: grid->spec_norm_alpha = (float)NAN; break;
    default: grid->erb_norm_alpha = 0.0f; break;
    }
}
#define N_TAMPERS 9

static DFN2GridSpec case_spec(const GridCase *c) {
    DFN2GridSpec spec = dfn2_test_grid_spec();
    spec.sample_rate = c->sample_rate;
    spec.n_fft = c->n_fft;
    spec.n_erb = c->n_erb;
    spec.df_bins = c->df_bins;
    spec.mask_lookahead = c->mask;
    spec.df_lookahead = c->df;
    return spec;
}

static int check_grid_refusals(void) {
    static DFN2State state;
    static DFN2ModelIOState io;
    static float erb_row[DFN2_N_ERB], spec_row[2 * DFN2_DF_BINS];
    const DFN2Grid *grid = dfn2_test_grid();
    const int sr = DFN2_SR, nf = DFN2_N_FFT, ne = DFN2_N_ERB, db = DFN2_DF_BINS;
    const int m = DFN2_MASK_LOOKAHEAD, d = DFN2_DF_LOOKAHEAD;
    const GridCase cases[] = {
        {0,  nf, ne, db, m, d, DFN2_GRID_ERR_SAMPLE_RATE},
        {-1, nf, ne, db, m, d, DFN2_GRID_ERR_SAMPLE_RATE},
        {sr, 960,   ne, db, m, d, DFN2_GRID_ERR_N_FFT},      /* not a power of two */
        {sr, 8,     ne, db, m, d, DFN2_GRID_ERR_N_FFT},
        {sr, 16384, ne, db, m, d, DFN2_GRID_ERR_N_FFT},      /* above audiolib's limit */
        {sr, nf, 30, db, m, d, DFN2_GRID_ERR_N_ERB},         /* not a multiple of 4 */
        {sr, nf, 0,  db, m, d, DFN2_GRID_ERR_N_ERB},
        {sr, 64, 20, db, m, d, DFN2_GRID_ERR_N_ERB},         /* a band under two bins */
        {sr, nf, ne, 1,  m, d, DFN2_GRID_ERR_DF_BINS},
        {sr, nf, ne, nf / 2 + 1, m, d, DFN2_GRID_ERR_DF_BINS}, /* no bin left to mask */
        {sr, nf, ne, db, 3,  d, DFN2_GRID_ERR_MASK_LOOKAHEAD},
        {sr, nf, ne, db, -1, d, DFN2_GRID_ERR_MASK_LOOKAHEAD},
        {sr, nf, ne, db, m, DFN2_DF_ORDER, DFN2_GRID_ERR_DF_LOOKAHEAD},
        {sr, nf, ne, db, m, -1, DFN2_GRID_ERR_DF_LOOKAHEAD},
        /* no capacity: these sizes are grids */
        {sr, 2048, ne, db,  m, d, DFN2_GRID_OK},
        {sr, nf, 36, db,        m, d, DFN2_GRID_OK},
        {sr, nf, ne, 100,       m, d, DFN2_GRID_OK},
        {sr, nf, ne, db, 2, 4, DFN2_GRID_OK},                /* ring 7 */
        {sr, nf, ne, db, 2, 3, DFN2_GRID_OK},                /* ring 6 */
        {sr, nf, ne, db, 2, 2, DFN2_GRID_OK},                /* ring 5 */
    };
    DFN2Grid out, untouched;
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, guard;
    size_t i;
    int t;

    memset(&untouched, 0xa5, sizeof untouched);
    for (i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        const DFN2GridSpec spec = case_spec(&cases[i]);
        out = untouched;
        if (dfn2_grid_make(&out, &spec) != cases[i].want) {
            fprintf(stderr, "grid case %zu: wrong status\n", i);
            return 1;
        }
        /* a refusal writes nothing; an accepted value is a valid grid */
        if (cases[i].want != DFN2_GRID_OK)
            CHECK(memcmp(&out, &untouched, sizeof out) == 0);
        else
            CHECK(dfn2_grid_validate(&out) == DFN2_GRID_OK);
    }
    {
        const DFN2GridSpec spec = dfn2_test_grid_spec();
        CHECK(dfn2_grid_make(NULL, &spec) == DFN2_GRID_ERR_NULL);
        CHECK(dfn2_grid_make(&out, NULL) == DFN2_GRID_ERR_NULL);
    }
    CHECK(dfn2_grid_validate(NULL) == DFN2_GRID_ERR_NULL);
    CHECK(dfn2_grid_validate(grid) == DFN2_GRID_OK);

    /* Each hand-edited derived field (and an alpha outside (0, 1)) is
     * INCONSISTENT, and no consumer takes the grid. */
    memset(&guard, 0xa5, sizeof guard);
    memset(erb_row, 0, sizeof erb_row);
    memset(spec_row, 0, sizeof spec_row);
    for (t = 0; t < N_TAMPERS; ++t) {
        DFN2Grid bad;
        void *pool, *state_block, *io_block;
        const size_t state_bytes = dfn2_state_mem_size(grid);
        const size_t io_bytes = dfn2_model_io_mem_size(grid);
        tamper_grid(&bad, t);
        CHECK(memcmp(&bad, grid, sizeof bad) != 0);
        CHECK(dfn2_grid_validate(&bad) == DFN2_GRID_ERR_INCONSISTENT);

        CHECK(dfn2_prepost_config_defaults(&cfg, &bad) == -1);
        config_freq(&cfg);
        cfg.grid = bad;
        req = guard;
        CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
        CHECK(memcmp(&req, &guard, sizeof req) == 0);
        CHECK(make_class(&cfg) == NULL);
        config_freq(&cfg);
        CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
        pool = alloc_aligned(req.alignment, (size_t)req.bytes);
        CHECK(pool != NULL);
        cfg.grid = bad;
        CHECK(start(pool, (size_t)req.bytes, &cfg) == NULL);
        free(pool);

        /* The blocks are the right size for the unedited grid, so the grid is what is
         * refused. */
        state_block = alloc_aligned(16u, state_bytes);
        io_block = alloc_aligned(16u, io_bytes);
        CHECK(state_block != NULL && io_block != NULL);
        CHECK(dfn2_state_init(&state, state_block, state_bytes, &bad) == -1);
        CHECK(dfn2_model_io_init(&io, io_block, io_bytes, &bad, NULL, 0, NULL, 0) == -1);
        /* a state that has no grid runs nothing */
        CHECK(dfn2_model_io_ready_frames(&io) == 0u);
        CHECK(dfn2_model_io_push_features(&io, erb_row, spec_row) == -1);
        free(state_block);
        free(io_block);
    }
    /* ...and the same calls accept the test grid, or the refusals above prove
     * nothing. */
    {
        const size_t state_bytes = dfn2_state_mem_size(grid);
        const size_t io_bytes = dfn2_model_io_mem_size(grid);
        void *state_block = alloc_aligned(16u, state_bytes);
        void *io_block = alloc_aligned(16u, io_bytes);
        CHECK(state_block != NULL && io_block != NULL);
        CHECK(dfn2_state_init(&state, state_block, state_bytes, grid) == 0);
        CHECK(dfn2_model_io_init(&io, io_block, io_bytes, grid, NULL, 0, NULL, 0) == 0);
        CHECK(dfn2_model_io_ready_frames(&io) ==
              (unsigned)(DFN2_MASK_LOOKAHEAD + grid->frames_per_call));
        CHECK(dfn2_model_io_ready_frames(NULL) == 0u);
        CHECK(dfn2_state_init(NULL, state_block, state_bytes, grid) == -1);
        CHECK(dfn2_model_io_init(NULL, io_block, io_bytes, grid, NULL, 0, NULL, 0) == -1);
        free(state_block);
        free(io_block);
    }
    return 0;
}

static int case_reject(void) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, guard;
    void *pool;

    /* The accepting case first, so the refusals below are not vacuous. */
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes > 0u);

    CHECK(check_grid_refusals() == 0);
    CHECK(dfn2_prepost_config_defaults(NULL, dfn2_test_grid()) == -1);
    CHECK(dfn2_prepost_config_defaults(&cfg, NULL) == -1);

    /* *req is left untouched on every refusal, not half-filled. */
    memset(&guard, 0xa5, sizeof guard);
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(NULL, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, NULL) == -1);

    /* The ERB matrices are required: mask expansion and the ERB feature
     * branch live on the spectrum side. */
    config_freq(&cfg);
    cfg.erb_fwd = NULL;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    config_freq(&cfg);
    cfg.erb_inv = NULL;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);

    /* A non-finite attenuation limit. */
    config_freq(&cfg);
    cfg.atten_lim_db = (float)NAN;
    req = guard;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    CHECK(memcmp(&req, &guard, sizeof req) == 0);
    cfg.atten_lim_db = (float)INFINITY;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);
    cfg.atten_lim_db = -(float)INFINITY;
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == -1);

    /* _init: NULL pool, undersized pool, misaligned pool, then the good one. */
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes + req.alignment);
    CHECK(pool != NULL);
    CHECK(start(NULL, (size_t)req.bytes, &cfg) == NULL);
    CHECK(start(pool, (size_t)req.bytes - 1u, &cfg) == NULL);
    CHECK(start((unsigned char *)pool + 1,
                (size_t)req.bytes, &cfg) == NULL);
    CHECK(start(pool, (size_t)req.bytes, &cfg) != NULL);

    /* Stage calls: NULLs, out-of-order calls, and the two "no frame open"
     * shapes the state machine must distinguish. */
    {
        DFN2Prepost *p = start(pool, (size_t)req.bytes, &cfg);
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        CHECK(p != NULL);
        CHECK(dfn2_prepost_pre_process_freq(NULL, pcm_in, pcm_in) == -1);
        CHECK(dfn2_prepost_frame_open(NULL, &inputs, &outputs) == -1);
        CHECK(dfn2_prepost_frame_open(p, NULL, &outputs) == -1);
        CHECK(dfn2_prepost_frame_open(p, &inputs, NULL) == -1);
        CHECK(dfn2_prepost_frame_finish(NULL) == -1);
        CHECK(dfn2_prepost_frame_bypass(NULL) == -1);

        /* Nothing is open yet. */
        CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == -1);
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(dfn2_prepost_frame_bypass(p) == -1);

        /* post_process before any frame ran: a full, defined, silent hop. */
        {
            int valid = -1;
            memset(out_a, 0x5a, (size_t)DFN2_HOP_LEN * sizeof(float));
            CHECK(tpost(p, out_a, &valid) == 0);
            CHECK(valid == 0);
            CHECK(all_zero(out_a, DFN2_HOP_LEN));
        }

        /* Open a frame, then refuse the emit that would strand it. */
        CHECK(open_first_frame_time(p) >= 0);
        CHECK(tpost(p, out_a, NULL) == -1);
        /* An accelerator that never wrote leaves the NaN prefill in place,
         * so the transaction refuses rather than committing garbage. */
        CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(dfn2_prepost_frame_bypass(p) == 0);
    }

    free(pool);
    return 0;
}

/* ---- reset: state really cleared ------------------------------------- */

static int case_reset(void) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = start(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);

    CHECK(drive_time(p, pcm_in, out_a, SHORT_HOPS) == 0);
    /* Without a reset the second run continues from a warm state -- assert
     * that it does NOT match, or the comparison after the reset has no
     * teeth.  A warm instance also emits a frame on its very first hop. */
    CHECK(drive_time_ex(p, pcm_in, out_b, SHORT_HOPS, 0, NULL, NULL) == 0);
    CHECK(!identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    treset(p);
    /* The framing clock is cold again: the first hop asks for no inference
     * unless the window of heads[0] is already complete. */
    CHECK(tpre(p, pcm_in) == PRE_RET(0));
    treset(p);
    CHECK(drive_time(p, pcm_in, out_b, SHORT_HOPS) == 0);
    CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));

    /* A reset mid-frame drops the open frame instead of stranding it. */
    {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        treset(p);
        CHECK(open_first_frame_time(p) >= 0);
        CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
        treset(p);
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == -1);
        CHECK(drive_time(p, pcm_in, out_b, SHORT_HOPS) == 0);
        CHECK(identical(out_a, out_b, (size_t)SHORT_HOPS * DFN2_HOP_LEN));
    }

    stop(p);
    free(pool);
    return 0;
}

/* ---- skip: the exact identity, and no state advance ------------------ */

/* Hop h emits source frame h - LOOKAHEAD. Its mask and blend weight come from
 * the heads of hop h - DFN2_DF_LOOKAHEAD (hop k's heads describe frame
 * k - DFN2_MASK_LOOKAHEAD, which hop k + DFN2_DF_LOOKAHEAD emits), so the
 * emission is the plain delayed noisy frame unless one of the hops committed
 * for real (real_a, real_b) supplied them. */
static int skip_emission_ok(int h, int real_a, int real_b,
                            const float *got_re, const float *got_im,
                            float (*in_re)[DFN2_N_BINS],
                            float (*in_im)[DFN2_N_BINS]) {
    const int shaped = h - DFN2_DF_LOOKAHEAD == real_a ||
                       h - DFN2_DF_LOOKAHEAD == real_b;
    const int same = identical(got_re, in_re[h - LOOKAHEAD], DFN2_N_BINS) &&
                     identical(got_im, in_im[h - LOOKAHEAD], DFN2_N_BINS);
    return shaped ? !same : same;
}

static int case_skip(void) {
    static float in_re[SKIP_TOTAL][DFN2_N_BINS];
    static float in_im[SKIP_TOTAL][DFN2_N_BINS];
    static float got_re[DFN2_N_BINS], got_im[DFN2_N_BINS];
    static float snap_gru[TEST_GRU_ELEMENTS];
    static float snap_convp[TEST_CONVP_WINDOW_ELEMENTS];
    static float snap_erb_window[TEST_ERB_WINDOW_ELEMENTS];
    static float snap_spec_window[TEST_SPEC_WINDOW_ELEMENTS];
    static float wrote_gru[TEST_GRU_ELEMENTS];
    static float wrote_convp[TEST_CONVP_WINDOW_ELEMENTS];
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    DFN2Prepost *p;
    void *pool;
    int t;

    /* The spectrum entries, so the identity is observable on the spectrum itself,
     * without a WOLA round trip in between. */
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = start(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    for (t = 0; t < SKIP_TOTAL; ++t) frame_spectrum(t, in_re[t], in_im[t]);
    CHECK(any_nonzero(in_re[0], DFN2_N_BINS));

    /* Every frame taken on frame_bypass: the accelerator never runs at all,
     * and the class must still be a bit-exact delay line of exactly
     * MASK_LOOKAHEAD + DF_LOOKAHEAD frames. */
    for (t = 0; t < SKIP_FRAMES; ++t) {
        int n, valid = -1;
        n = dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]);
        CHECK(n == PRE_RET(t));
        if (n == 1) CHECK(dfn2_prepost_frame_bypass(p) == 0);
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
     * Whether the frame this hop EMITS is still the identity is the
     * cascade's timing rather than a coincidence: the heads shaping source
     * frame t - LOOKAHEAD were supplied DFN2_DF_LOOKAHEAD hops EARLIER (they
     * land in the ring at head_frame = hop - MASK_LOOKAHEAD), and with a
     * positive DF lookahead that hop was a skip. A commit never reaches back
     * and re-masks a frame already emitted; it shapes the emission
     * DFN2_DF_LOOKAHEAD hops later (skip_emission_ok). */
    t = SKIP_FRAMES;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    CHECK(skip_emission_ok(t, SKIP_FRAMES, SKIP_FRAMES + 3,
                           got_re, got_im, in_re, in_im));

    /* Snapshot, then skip even though the accelerator produced a COMPLETE,
     * perfectly committable result.  The full write is the point: skipping
     * after a partial write would prove nothing, because commit() would
     * refuse that frame on its own.  The accelerator writes the recurrent
     * state in place, so the skip cannot roll it back: it keeps what the
     * accelerator wrote. */
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
    memcpy(snap_gru, inputs.gru_hidden, sizeof snap_gru);
    memcpy(snap_convp, inputs.df_convp_window, sizeof snap_convp);
    memcpy(snap_erb_window, inputs.erb_window, sizeof snap_erb_window);
    memcpy(snap_spec_window, inputs.spec_window,
           sizeof snap_spec_window);
    CHECK(any_nonzero(snap_gru, TEST_GRU_ELEMENTS));
    CHECK(any_nonzero(snap_convp, TEST_CONVP_WINDOW_ELEMENTS));
    fake_run(&inputs, &outputs);
    CHECK(state_aliased(&inputs, &outputs));
    memcpy(wrote_gru, inputs.gru_hidden, sizeof wrote_gru);
    memcpy(wrote_convp, inputs.df_convp_window, sizeof wrote_convp);
    /* The in-place write moved every state tensor off the snapshot. */
    CHECK(!identical(wrote_gru, snap_gru, TEST_GRU_ELEMENTS));
    CHECK(!identical(wrote_convp, snap_convp,
                     TEST_CONVP_WINDOW_ELEMENTS));
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    /* Whether this hop's emission is the identity follows the same rule, and
     * asserting the difference where the rule predicts one is the point: the
     * emission that source frame was masked into by the REAL heads proves
     * that commit actually reached the compose stage. A skip only takes the
     * identity for the frame whose heads IT supplies. */
    CHECK(skip_emission_ok(t, SKIP_FRAMES, SKIP_FRAMES + 3,
                           got_re, got_im, in_re, in_im));

    /* The skip left the two recurrent tensors exactly as the accelerator
     * wrote them -- not restored to the snapshot, not zeroed -- while the two
     * feature windows are the framing clock and must still have slid. */
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(identical(inputs.gru_hidden, wrote_gru,
                    inputs.gru_hidden_elements));
    CHECK(identical(inputs.df_convp_window, wrote_convp,
                    inputs.df_convp_window_elements));
    CHECK(!identical(inputs.gru_hidden, snap_gru,
                     inputs.gru_hidden_elements));
    CHECK(!identical(inputs.df_convp_window, snap_convp,
                     inputs.df_convp_window_elements));
    CHECK(!identical(inputs.erb_window, snap_erb_window,
                     inputs.erb_window_elements));
    CHECK(!identical(inputs.spec_window, snap_spec_window,
                     inputs.spec_window_elements));

    /* A skip over a frame on which the accelerator never ran writes nothing,
     * so the state stays frozen exactly as it is. */
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    CHECK(skip_emission_ok(t, SKIP_FRAMES, SKIP_FRAMES + 3,
                           got_re, got_im, in_re, in_im));
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(identical(inputs.gru_hidden, wrote_gru,
                    inputs.gru_hidden_elements));
    CHECK(identical(inputs.df_convp_window, wrote_convp,
                    inputs.df_convp_window_elements));

    /* ...and a commit here DOES move them, so the freeze above is not
     * measuring a constant. */
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(p) == 0);
    CHECK(dfn2_prepost_post_process_freq(p, got_re, got_im, NULL) == 0);
    CHECK(skip_emission_ok(t, SKIP_FRAMES, SKIP_FRAMES + 3,
                           got_re, got_im, in_re, in_im));
    ++t;
    CHECK(dfn2_prepost_pre_process_freq(p, in_re[t], in_im[t]) == 1);
    CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
    CHECK(!identical(inputs.gru_hidden, wrote_gru,
                     inputs.gru_hidden_elements));
    CHECK(!identical(inputs.df_convp_window, wrote_convp,
                     inputs.df_convp_window_elements));
    CHECK(dfn2_prepost_frame_bypass(p) == 0);

    stop(p);
    free(pool);
    return 0;
}

/* ---- guard: the frame-state-machine contract gates ------------------- */

static int case_guard(void) {
    static float re_a[DFN2_N_BINS], im_a[DFN2_N_BINS];
    static float re_b[DFN2_N_BINS], im_b[DFN2_N_BINS];
    static float hop_scratch[DFN2_HOP_LEN];
    DFN2PrepostConfig cfg_time, cfg_freq;
    DFN2PrepostMemReq req_time, req_freq;
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    DFN2Prepost *pt, *pf;
    void *pool_time, *pool_freq;

    /* Two instances: pt is driven through the host's hop-in / hop-out framing, pf
     * through spectra it supplies itself. */
    config_freq(&cfg_time);
    config_freq(&cfg_freq);
    CHECK(dfn2_prepost_get_mem_size(&cfg_time, &req_time) == 0);
    CHECK(dfn2_prepost_get_mem_size(&cfg_freq, &req_freq) == 0);
    pool_time = alloc_aligned(req_time.alignment, (size_t)req_time.bytes);
    pool_freq = alloc_aligned(req_freq.alignment, (size_t)req_freq.bytes);
    CHECK(pool_time != NULL && pool_freq != NULL);
    pt = start(pool_time, (size_t)req_time.bytes, &cfg_time);
    pf = start(pool_freq, (size_t)req_freq.bytes, &cfg_freq);
    CHECK(pt != NULL && pf != NULL);
    frame_spectrum(0, re_a, im_a);
    frame_spectrum(1, re_b, im_b);

    /* (1) A second pre_process with a frame open is refused, and the refusal
     * does not eat the open frame: it is still committable afterwards. */
    {
        int next = open_first_frame_time(pt);
        CHECK(next >= 0);
        CHECK(tpre(pt, pcm_in + (size_t)next * DFN2_HOP_LEN) == -1);
    }
    /* The setters are between-hops operations. With the frame open they
     * are refused, so a swap can never land inside one transaction (between
     * the features just taken and the compose still pending), and the
     * refusal leaves the frame committable. */
    CHECK(dfn2_prepost_set_erb_matrices(pt, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_atten_lim(pt, -12.0f) == -1);
    CHECK(dfn2_prepost_frame_open(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(pt) == 0);
    CHECK(dfn2_prepost_set_erb_matrices(pt, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == 0);
    CHECK(dfn2_prepost_set_atten_lim(pt, 0.0f) == 0);

    CHECK(open_first_frame_freq(pf, re_b, im_b) >= 0);
    CHECK(dfn2_prepost_pre_process_freq(pf, re_a, im_a) == -1);
    /* Same refusal on the spectrum entry: the guard is keyed to the frame, not to
     * who owns the transform. */
    CHECK(dfn2_prepost_set_erb_matrices(pf, &erb_fwd[0][0],
                                        &erb_inv[0][0]) == -1);
    CHECK(dfn2_prepost_set_atten_lim(pf, -12.0f) == -1);
    CHECK(dfn2_prepost_frame_open(pf, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(pf) == 0);
    CHECK(dfn2_prepost_set_atten_lim(pf, 0.0f) == 0);

    /* (2) A commit with no frame_open behind it is refused: an accelerator
     * that never ran must not be able to pass untouched buffers off as a
     * result.  The frame stays open and commits normally afterwards. */
    treset(pt);
    CHECK(open_first_frame_time(pt) >= 0);
    CHECK(dfn2_prepost_frame_finish(pt) == -1);
    CHECK(dfn2_prepost_frame_open(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(pt) == 0);

    /* (3) A commit that failed on a non-finite output DISARMS the
     * transaction: retrying it without a fresh frame_open is refused,
     * and the fresh frame_open NaN-refills every writable element. */
    treset(pt);
    CHECK(open_first_frame_time(pt) >= 0);
    CHECK(dfn2_prepost_frame_open(pt, &inputs, &outputs) == 0);
    fake_run(&inputs, &outputs);
    {
        /* Keep the good value so the retry below can be made perfectly
         * committable again.  Retrying while the NaN is still in place
         * would be refused by the finiteness preflight whether or not the
         * transaction was disarmed, and would prove nothing. */
        const size_t poison = TEST_COEFS_ELEMENTS / 2;
        float good = outputs.coefs[poison];
        CHECK(isfinite(good));
        outputs.coefs[poison] = (float)NAN;
        CHECK(dfn2_prepost_frame_finish(pt) == -1);
        /* Repaired in place through the stale view: the only thing left
         * that can refuse this commit is the disarm itself. */
        outputs.coefs[poison] = good;
        CHECK(dfn2_prepost_frame_finish(pt) == -1);
    }
    CHECK(dfn2_prepost_frame_open(pt, &inputs, &outputs) == 0);
    /* The fresh frame_open NaN-refills the heads only. The state is the
     * accelerator's input, so it is not prefilled: the refused commit left it
     * zero, and it is the same memory as the outputs' state tensors. */
    CHECK(heads_all_nan(&outputs));
    CHECK(state_aliased(&inputs, &outputs));
    CHECK(state_all_zero(&inputs));
    fake_run(&inputs, &outputs);
    CHECK(dfn2_prepost_frame_finish(pt) == 0);

    /* (4) frame_bypass is exactly the accelerator-never-ran path, so it needs
     * no frame_open behind it. */
    treset(pt);
    CHECK(open_first_frame_time(pt) >= 0);
    CHECK(dfn2_prepost_frame_bypass(pt) == 0);
    CHECK(tpost(pt, hop_scratch, NULL) == 0);

    stop(pt);
    stop(pf);
    free(pool_time);
    free(pool_freq);
    return 0;
}

/* ---- txn: on failure NOTHING moves, the frame stays open ------------- */

/* Reference stream. At hop TXN_PRE: mode 0 is a plain frame_bypass (the
 * accelerator never runs), mode 1 an ordinary commit of the un-poisoned
 * result, mode 2 an accelerator run whose state is then zeroed by hand and the
 * frame taken with frame_bypass -- the stream a refused commit must reproduce. */
static int txn_reference(DFN2Prepost *p, float *out, int mode) {
    int hop;
    for (hop = 0; hop < TXN_HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n = tpre(
            p, pcm_in + (size_t)hop * DFN2_HOP_LEN);
        if (n != PRE_RET(hop)) return -1;
        if (n == 1) {
            if (hop == TXN_PRE && mode != 1) {
                if (mode == 2) {
                    if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0)
                        return -1;
                    fake_run(&inputs, &outputs);
                    memset(outputs.gru_hidden_next, 0,
                           outputs.gru_hidden_elements * sizeof(float));
                    memset(outputs.df_convp_window_next, 0,
                           outputs.df_convp_window_elements * sizeof(float));
                }
                if (dfn2_prepost_frame_bypass(p) != 0) return -1;
            } else {
                if (dfn2_prepost_frame_open(p, &inputs, &outputs) != 0)
                    return -1;
                fake_run(&inputs, &outputs);
                if (dfn2_prepost_frame_finish(p) != 0) return -1;
            }
        }
        if (tpost(p, out + (size_t)hop * DFN2_HOP_LEN,
                                      NULL) != 0) return -1;
    }
    return 0;
}

static int txn_poisoned(DFN2Prepost *p, float *out, int tensor, size_t index) {
    static float snap_erb_window[TEST_ERB_WINDOW_ELEMENTS];
    static float snap_spec_window[TEST_SPEC_WINDOW_ELEMENTS];
    static float hop_scratch[DFN2_HOP_LEN];
    int hop;

    for (hop = 0; hop < TXN_HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        int n = tpre(
            p, pcm_in + (size_t)hop * DFN2_HOP_LEN);
        CHECK(n == PRE_RET(hop));
        if (n == 1 && hop == TXN_PRE) {
            float *target;
            size_t count = 0;
            CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
            memcpy(snap_erb_window, inputs.erb_window,
                   sizeof snap_erb_window);
            memcpy(snap_spec_window, inputs.spec_window,
                   sizeof snap_spec_window);
            fake_run(&inputs, &outputs);
            target = output_tensor(&outputs, tensor, &count);
            CHECK(index < count);
            target[index] = (float)NAN;
            CHECK(dfn2_prepost_frame_finish(p) == -1);
            /* (a) the frame is STILL OPEN -- neither emitting nor opening
             * the next one is allowed while it is. */
            CHECK(tpost(p, hop_scratch, NULL) == -1);
            CHECK(tpre(
                      p, pcm_in + (size_t)hop * DFN2_HOP_LEN) == -1);
            /* (b) the accelerator overwrote the state in place, so the
             * refusal cannot restore it: both recurrent tensors are zero
             * whichever of the five outputs was poisoned. The feature
             * windows are CPU-owned and unchanged. */
            CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
            CHECK(state_aliased(&inputs, &outputs));
            CHECK(state_all_zero(&inputs));
            CHECK(identical(inputs.erb_window, snap_erb_window,
                            inputs.erb_window_elements));
            CHECK(identical(inputs.spec_window, snap_spec_window,
                            inputs.spec_window_elements));
            /* (c) the documented recovery: take the identity. */
            CHECK(dfn2_prepost_frame_bypass(p) == 0);
        } else if (n == 1) {
            CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
            fake_run(&inputs, &outputs);
            CHECK(dfn2_prepost_frame_finish(p) == 0);
        }
        CHECK(tpost(p, out + (size_t)hop * DFN2_HOP_LEN,
                                        NULL) == 0);
    }
    return 0;
}

static int case_txn(void) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    DFN2Prepost *p;
    void *pool;
    const size_t stream = (size_t)TXN_HOPS * DFN2_HOP_LEN;
    int tensor, which;

    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);

    p = start(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(txn_reference(p, out_a, 0) == 0);      /* skipped at hop K   */
    p = start(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(txn_reference(p, out_b, 1) == 0);      /* committed at hop K */
    p = start(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL);
    CHECK(txn_reference(p, out_d, 2) == 0);      /* zeroed state, skipped */
    /* The teeth: the three references must genuinely differ, otherwise
     * "identical to the zeroed-state run" would be satisfied by anything --
     * in particular, zeroing the state must be observable against a skip
     * that leaves it. */
    CHECK(any_nonzero(out_d, stream));
    CHECK(!identical(out_a, out_b, stream));
    CHECK(!identical(out_d, out_b, stream));
    CHECK(!identical(out_d, out_a, stream));

    for (tensor = 0; tensor < N_OUT_TENSORS; ++tensor) {
        for (which = 0; which < 3; ++which) {
            DFN2PrepostOutputs shape;
            DFN2PrepostInputs probe;
            size_t count = 0, index;
            DFN2Prepost *probe_p;

            /* Ask the class itself for this tensor's length rather than
             * recomputing it here, so a shape change cannot drift. */
            probe_p = start(pool, (size_t)req.bytes, &cfg);
            CHECK(probe_p != NULL);
            CHECK(open_first_frame_time(probe_p) >= 0);
            CHECK(dfn2_prepost_frame_open(probe_p, &probe, &shape) == 0);
            (void)output_tensor(&shape, tensor, &count);
            CHECK(count > 0u);
            index = (which == 0) ? 0u
                  : (which == 1) ? count - 1u
                                 : count / 2u;

            p = start(pool, (size_t)req.bytes, &cfg);
            CHECK(p != NULL);
            CHECK(txn_poisoned(p, out_c, tensor, index) == 0);
            CHECK(identical(out_d, out_c, stream));
            CHECK(!identical(out_b, out_c, stream));
            CHECK(!identical(out_a, out_c, stream));
        }
    }

    printf("txn: %d tensors x 3 indices, %d hops each\n",
           N_OUT_TENSORS, TXN_HOPS);
    free(pool);
    return 0;
}

/* ---------------------------------------------------------------------- */

/* The recurrent state is ONE buffer per tensor: the input view and the
 * `*_next` view are the same address, on every frame, across commit, a refused
 * commit, skip and reset, and the accelerator's in-place write is what the
 * next frame reads. */
static int case_state_inplace(void) {
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    float re[DFN2_N_BINS] = {0}, im[DFN2_N_BINS] = {0};
    const float *a;
    int frame;
    config_freq(&cfg);
    p = make_class(&cfg);
    CHECK(p != NULL);
    CHECK(open_first_frame_freq(p, re, im) >= 0);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    a = in.df_convp_window;
    CHECK(state_aliased(&in, &out));
    CHECK(in.df_convp_window_elements == out.df_convp_window_elements);
    CHECK(state_all_zero(&in));
    fake_run(&in, &out);
    /* The write through the output view is already visible through the input
     * view: there is no second copy to commit. */
    CHECK(any_nonzero(a, in.df_convp_window_elements));
    CHECK(any_nonzero(in.gru_hidden, in.gru_hidden_elements));
    CHECK(dfn2_prepost_frame_finish(p) == 0);
    CHECK(any_nonzero(a, in.df_convp_window_elements));
    for (frame = 0; frame < 3; ++frame) {
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        CHECK(in.df_convp_window == a);
        CHECK(out.df_convp_window_next == a);
        CHECK(state_aliased(&in, &out));
        if (frame == 0) {
            /* A refused commit zeroes the state in place; no address moves. */
            CHECK(dfn2_prepost_frame_finish(p) == -1);
            CHECK(state_all_zero(&in));
            CHECK(dfn2_prepost_frame_bypass(p) == 0);
        } else if (frame == 1) {
            fake_run(&in, &out);
            CHECK(dfn2_prepost_frame_finish(p) == 0);
        } else {
            CHECK(dfn2_prepost_frame_bypass(p) == 0);
        }
    }
    treset(p);
    CHECK(open_first_frame_freq(p, re, im) >= 0);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    CHECK(in.df_convp_window == a);
    CHECK(state_aliased(&in, &out));
    CHECK(state_all_zero(&in));
    stop(p);
    return 0;
}

/* Mutation gate for the finite check. One NaN injected into one element of one
 * recurrent tensor THROUGH THE IN-PLACE POINTER must make frame_finish refuse,
 * leave both tensors zero, and leave the instance able to run the next
 * healthy frame. Run for each of the four poisoned places. */
static int case_inplace_nan(void) {
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    float re[DFN2_N_BINS] = {0}, im[DFN2_N_BINS] = {0};
    int tensor;

    config_freq(&cfg);
    for (tensor = 0; tensor < 4; ++tensor) {
        float *poison;
        p = make_class(&cfg);
        CHECK(p != NULL);
        CHECK(open_first_frame_freq(p, re, im) >= 0);
        /* A healthy frame first, so the state is non-zero going in. */
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(!state_all_zero(&in));
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        switch (tensor) {
        case 0: poison = &out.gru_hidden_next[(size_t)DFN2_TEST_GRU_ROW_ENCODER *
                                              DFN2_TEST_GRU_HIDDEN + 3];
                break;
        case 1: poison = &out.gru_hidden_next[(size_t)(DFN2_TEST_GRU_ROW_ERB + 1) *
                                              DFN2_TEST_GRU_HIDDEN +
                                              DFN2_TEST_GRU_HIDDEN - 1];
                break;
        case 2: poison = &out.gru_hidden_next[(size_t)(DFN2_TEST_GRU_ROW_DF + 1) *
                                              DFN2_TEST_GRU_HIDDEN];
                break;
        default: {
            /* the last element of the packed [channels][window][df_bins] run */
            const size_t last =
                (((size_t)DFN2_TEST_CONVP_CHANNELS - 1u) *
                     DFN2_TEST_CONVP_WINDOW +
                 (DFN2_TEST_CONVP_WINDOW - 1u)) * DFN2_DF_BINS +
                (DFN2_DF_BINS - 1u);
            CHECK(last + 1u == out.df_convp_window_elements);
            poison = &out.df_convp_window_next[last];
            break;
        }
        }
        CHECK(isfinite(*poison));
        *poison = (float)NAN;
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(state_all_zero(&in));
        /* Zeroed through the same memory the accelerator wrote. */
        CHECK(*poison == 0.0f);
        /* The frame is still open, and the next healthy frame works. */
        CHECK(dfn2_prepost_frame_bypass(p) == 0);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        CHECK(state_all_zero(&in));
        fake_run(&in, &out);
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(!state_all_zero(&in));
        stop(p);
    }
    return 0;
}

/* ---- generation: the counter that tells a runtime to refill its state -- */

/* Advances by exactly one for every event that zeroes the recurrent tensors
 * behind the accelerator's back (reset, each warm-up frame, a commit refused over a bad state, a
 * commit refused over a bad head), and never for a commit or a skip of a real frame. */
static int case_generation(void) {
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    float re[DFN2_N_BINS] = {0}, im[DFN2_N_BINS] = {0};
    unsigned int g;
    int hop;

    CHECK(dfn2_prepost_state_generation(NULL) == 0u);
    config_freq(&cfg);
    p = make_class(&cfg);
    CHECK(p != NULL);
    g = dfn2_prepost_state_generation(p);
    CHECK(g == 1u);

    /* Each warm-up frame: exactly one, whatever the heads hold. */
    for (hop = 0; hop < DFN2_MASK_LOOKAHEAD; ++hop) {
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_state_generation(p) == g);   /* analysis moves nothing */
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        out.alpha[0] = (float)NAN;   /* a warm-up frame does not read its heads */
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(state_all_zero(&in));
        ++g;
        CHECK(dfn2_prepost_state_generation(p) == g);
    }
    /* After a reset the warm-up frames are due again; closed by frame_bypass they count the same. */
    treset(p);
    ++g;
    CHECK(dfn2_prepost_state_generation(p) == g);
    for (hop = 0; hop < DFN2_MASK_LOOKAHEAD; ++hop) {
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        CHECK(dfn2_prepost_frame_bypass(p) == 0);
        CHECK(state_all_zero(&in));
        ++g;
        CHECK(dfn2_prepost_state_generation(p) == g);
    }
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    fake_run(&in, &out);
    CHECK(dfn2_prepost_frame_finish(p) == 0);
    CHECK(dfn2_prepost_state_generation(p) == g);   /* a real frame's commit moves nothing */
    CHECK(!state_all_zero(&in));
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_state_generation(p) == g);

    /* A non-finite state element. */
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    fake_run(&in, &out);
    out.gru_hidden_next[(size_t)DFN2_TEST_GRU_ROW_DF * DFN2_TEST_GRU_HIDDEN] = (float)NAN;
    CHECK(dfn2_prepost_frame_finish(p) == -1);
    CHECK(state_all_zero(&in));
    ++g;
    CHECK(dfn2_prepost_state_generation(p) == g);
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_state_generation(p) == g);

    /* A non-finite head over a finite, non-zero state still restarts it. */
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    fake_run(&in, &out);
    CHECK(!state_all_zero(&in));
    out.alpha[0] = (float)NAN;
    CHECK(dfn2_prepost_frame_finish(p) == -1);
    CHECK(state_all_zero(&in));
    ++g;
    CHECK(dfn2_prepost_state_generation(p) == g);
    CHECK(dfn2_prepost_frame_bypass(p) == 0);

    /* A second refusal of the same open frame counts again, a skip does not. */
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    CHECK(dfn2_prepost_frame_finish(p) == -1);   /* heads still NaN */
    ++g;
    CHECK(dfn2_prepost_state_generation(p) == g);
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_state_generation(p) == g);

    treset(p);
    ++g;
    CHECK(dfn2_prepost_state_generation(p) == g);
    stop(p);
    return 0;
}

/* ---- warmup: the first mask_lookahead frames are frames like any other, minus their heads -- */

/* A registered-state instance, or an unregistered one when `registered` is 0. */
static DFN2Prepost *warm_instance(int registered) {
    DFN2PrepostConfig cfg;
    config_freq(&cfg);
    return registered ? make_class(&cfg) : dfn2_prepost_create(&cfg);
}

/* Set every element of the registered recurrent state through the accelerator's in-place views. */
static void state_set(const DFN2PrepostOutputs *out, float v) {
    size_t i;
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = v;
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = v;
}

static int state_is(const DFN2PrepostInputs *in, float v) {
    size_t i;
    for (i = 0; i < in->gru_hidden_elements; ++i) if (in->gru_hidden[i] != v) return 0;
    for (i = 0; i < in->df_convp_window_elements; ++i) if (in->df_convp_window[i] != v) return 0;
    return 1;
}

/* The row `back` rows before the newest of an input window of rows of `width` elements. */
static const float *window_row(const float *window, size_t width, int back) {
    return window + (size_t)(DFN2_MODEL_INPUT_FRAMES - 1 - back) * width;
}

/* Warm-up frame t: opened like a real one (window [0, .., 0, x0..xt], heads NaN), closed without
 * reading its heads, state zero afterwards, generation exactly +1, nothing emitted. Every run
 * of this case is for N == 1; with mask_lookahead == 0 there is no warm-up frame and the loops
 * below are empty, while the real-frame checks still hold. */
static int case_warmup(void) {
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    DFN2Prepost *p;
    float re[DFN2_N_BINS], im[DFN2_N_BINS], hop_re[DFN2_N_BINS], hop_im[DFN2_N_BINS];
    unsigned int g;
    long long frame = -1;
    int t, k, valid = -1;
    const int m = DFN2_MASK_LOOKAHEAD;
    const DFN2Grid *grid = dfn2_test_grid();
    /* the hand-driven reference: the feature front end and the graph windows only, as before the
     * warm-up frames were opened (dfn2_prepost.c not involved) */
    static DFN2State ref_st;
    static DFN2ModelIOState ref_io;
    static float ref_gru[DFN2_TEST_GRU_ELEMENTS], ref_convp[DFN2_TEST_CONVP_ELEMENTS];
    static float feat_erb[DFN2_N_ERB], feat_spec[2 * DFN2_DF_BINS];
    void *state_block = alloc_aligned(16u, dfn2_state_mem_size(grid));
    void *io_block = alloc_aligned(16u, dfn2_model_io_mem_size(grid));
    CHECK(state_block != NULL && io_block != NULL);
    CHECK(dfn2_state_init(&ref_st, state_block, dfn2_state_mem_size(grid), grid) == 0);
    dfn2_set_erb_matrices(&ref_st, &erb_fwd[0][0], &erb_inv[0][0]);
    CHECK(dfn2_model_io_init(&ref_io, io_block, dfn2_model_io_mem_size(grid), grid, ref_gru,
                             DFN2_TEST_GRU_ELEMENTS, ref_convp, DFN2_TEST_CONVP_ELEMENTS) == 0);

    /* The warm-up frame as a frame: pre returns 1, the window holds the frames seen so far with
     * zeros before them, the heads are NaN; garbage heads and garbage state do not matter. */
    p = warm_instance(1);
    CHECK(p != NULL);
    g = dfn2_prepost_state_generation(p);
    CHECK(g == 1u);
    for (t = 0; t < m; ++t) {
        frame_spectrum(t, re, im);
        dfn2_compute_features(&ref_st, re, im, feat_erb, feat_spec);
        CHECK(dfn2_model_io_push_features(&ref_io, feat_erb, feat_spec) == 0);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        CHECK(identical(in.erb_window, ref_io.erb_window, in.erb_window_elements));
        CHECK(identical(in.spec_window, ref_io.spec_window, in.spec_window_elements));
        CHECK(heads_all_nan(&out));
        CHECK(any_nonzero(window_row(in.erb_window, DFN2_N_ERB, 0), DFN2_N_ERB));
        CHECK(any_nonzero(window_row(in.spec_window, DFN2_DF_BINS, 0), DFN2_DF_BINS));
        for (k = t + 1; k < DFN2_MODEL_INPUT_FRAMES; ++k) {
            CHECK(all_zero(window_row(in.erb_window, DFN2_N_ERB, k), DFN2_N_ERB));
            CHECK(all_zero(window_row(in.spec_window, DFN2_DF_BINS, k), DFN2_DF_BINS));
        }
        state_set(&out, 123.0f);                       /* the accelerator wrote the state in place */
        CHECK(state_is(&in, 123.0f));
        CHECK(dfn2_prepost_frame_finish(p) == 0);       /* heads still NaN: not looked at */
        CHECK(state_all_zero(&in));
        CHECK(dfn2_prepost_state_generation(p) == g + 1u);
        ++g;
        /* the frame is closed: a second finish or bypass is refused and moves nothing */
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(dfn2_prepost_frame_bypass(p) == -1);
        CHECK(dfn2_prepost_state_generation(p) == g);
        CHECK(dfn2_prepost_post_process_freq(p, hop_re, hop_im, &valid) == 0);
        CHECK(valid == 0 && all_zero(hop_re, DFN2_N_BINS) && all_zero(hop_im, DFN2_N_BINS));
        CHECK(dfn2_prepost_output_frame_index(p, &frame) == -1);
    }
    /* The first real frame: its windows are byte-identical to the hand-driven reference's, its
     * state is zero (as it was before the warm-up frames were opened), its heads ARE read and it
     * is not a warm-up frame (a NaN head is refused). */
    frame_spectrum(m, re, im);
    dfn2_compute_features(&ref_st, re, im, feat_erb, feat_spec);
    CHECK(dfn2_model_io_push_features(&ref_io, feat_erb, feat_spec) == 1);
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    CHECK(state_all_zero(&in));
    CHECK(identical(in.erb_window, ref_io.erb_window, in.erb_window_elements));
    CHECK(identical(in.spec_window, ref_io.spec_window, in.spec_window_elements));
    CHECK(any_nonzero(window_row(in.erb_window, DFN2_N_ERB, 0), DFN2_N_ERB));
    CHECK(dfn2_prepost_frame_finish(p) == -1);        /* heads NaN: a real frame reads them */
    CHECK(dfn2_prepost_state_generation(p) == g + 1u);
    ++g;
    CHECK(dfn2_prepost_frame_bypass(p) == 0);
    CHECK(dfn2_prepost_state_generation(p) == g);
    stop(p);

    /* The warm-up frame closed by frame_bypass, and through dfn2_model_run_frame. */
    p = warm_instance(1);
    CHECK(p != NULL);
    g = dfn2_prepost_state_generation(p);
    for (t = 0; t < m; ++t) {
        frame_spectrum(t, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        state_set(&out, 7.0f);
        CHECK(dfn2_prepost_frame_bypass(p) == 0);
        CHECK(state_all_zero(&in));                     /* bypass of a real frame leaves the state */
        CHECK(dfn2_prepost_state_generation(p) == ++g);
        CHECK(dfn2_prepost_post_process_freq(p, hop_re, hop_im, &valid) == 0 && valid == 0);
    }
    stop(p);
    {
        StubModel stub;
        DFN2Model model;
        DFN2ModelIoDescriptor descriptor;
        memset(&stub, 0, sizeof stub);
        memset(&model, 0, sizeof model);
        CHECK(dfn2_model_io_descriptor_default(dfn2_test_grid(), &descriptor) == 0);
        model.user = &stub;
        model.infer = stub_infer;
        model.io_descriptor = &descriptor;
        p = warm_instance(1);
        CHECK(p != NULL);
        g = dfn2_prepost_state_generation(p);
        for (t = 0; t < m; ++t) {
            frame_spectrum(t, re, im);
            stub.mode = (t & 1) ? 1 : 0;                /* alternate: infer ok (1) / infer failed (0) */
            CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
            CHECK(dfn2_model_run_frame(&model, p) == ((t & 1) ? 0 : 1));
            CHECK(stub.calls == t + 1);                 /* one accelerator call per warm-up frame */
            CHECK(dfn2_prepost_state_generation(p) == ++g);
            CHECK(dfn2_prepost_post_process_freq(p, hop_re, hop_im, &valid) == 0 && valid == 0);
        }
        stop(p);
    }

    /* Nothing registered: only the generation moves; a non-finite registered state still counts once. */
    p = warm_instance(0);
    CHECK(p != NULL);
    g = dfn2_prepost_state_generation(p);
    for (t = 0; t < m; ++t) {
        frame_spectrum(t, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        CHECK(in.gru_hidden == NULL && in.gru_hidden_elements == 0u);
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(dfn2_prepost_state_generation(p) == ++g);
    }
    stop(p);
    p = warm_instance(1);
    CHECK(p != NULL);
    g = dfn2_prepost_state_generation(p);
    for (t = 0; t < m; ++t) {
        frame_spectrum(t, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        state_set(&out, 1.0f);
        out.gru_hidden_next[0] = (float)NAN;
        out.df_convp_window_next[out.df_convp_window_elements - 1] = (float)INFINITY;
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(state_all_zero(&in));
        CHECK(dfn2_prepost_state_generation(p) == ++g);   /* +1, not +2 */
    }
    stop(p);

    /* A reset restarts the warm-up: M frames again, each +1, the state of a frame after them is kept. A reset in
     * the middle of a warm-up frame drops it: its finish and bypass are refused. */
    p = warm_instance(1);
    CHECK(p != NULL);
    for (t = 0; t < m + 3; ++t) {
        frame_spectrum(t, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(t < m ? state_all_zero(&in) : !state_all_zero(&in));
    }
    g = dfn2_prepost_state_generation(p);
    treset(p);
    CHECK(dfn2_prepost_state_generation(p) == ++g);
    for (t = 0; t < m; ++t) {
        frame_spectrum(t, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        CHECK(state_all_zero(&in));
        CHECK(dfn2_prepost_state_generation(p) == ++g);
    }
    frame_spectrum(m, re, im);
    CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
    fake_run(&in, &out);
    CHECK(dfn2_prepost_frame_finish(p) == 0);
    CHECK(!state_all_zero(&in));
    CHECK(dfn2_prepost_state_generation(p) == g);
    treset(p);
    ++g;
    if (m > 0) {
        frame_spectrum(0, re, im);
        CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
        fake_run(&in, &out);
        treset(p);
        ++g;
        CHECK(dfn2_prepost_frame_finish(p) == -1);
        CHECK(dfn2_prepost_frame_bypass(p) == -1);
        CHECK(dfn2_prepost_state_generation(p) == g);
        /* the stream starts over: the warm-up is due in full again */
        for (t = 0; t < m; ++t) {
            frame_spectrum(t, re, im);
            CHECK(dfn2_prepost_pre_process_freq(p, re, im) == 1);
            CHECK(dfn2_prepost_frame_open(p, &in, &out) == 0);
            fake_run(&in, &out);
            CHECK(dfn2_prepost_frame_finish(p) == 0);
            CHECK(state_all_zero(&in));
            CHECK(dfn2_prepost_state_generation(p) == ++g);
        }
    }
    stop(p);
    free(state_block);
    free(io_block);
    return 0;
}

/* The caller pushes the synthesis on every hop, valid or not: the warm-up hops carry a zero
 * spectrum, which plain overlap-add turns into zeros, so the stream equals the one that pushes
 * only valid hops. */
static int case_warmup_synthesis(void) {
    AinrTestHop caller_hop;
    static float enh_re[DFN2_N_BINS], enh_im[DFN2_N_BINS];
    DFN2PrepostConfig cfg;
    DFN2Prepost *p_valid, *p_all;
    int hop;

    config_freq(&cfg);
    p_valid = make_class(&cfg);
    p_all = make_class(&cfg);
    CHECK(p_valid != NULL && p_all != NULL);
    CHECK(drive_time(p_valid, pcm_in, out_a, HOPS) == 0);
    CHECK(ainr_test_hop_open(&caller_hop, DFN2_N_FFT, dfn2_test_grid()->stft_scale,
                             dfn2_test_grid()->stft_unscale) == 0);
    for (hop = 0; hop < HOPS; ++hop) {
        DFN2PrepostInputs inputs;
        DFN2PrepostOutputs outputs;
        float *slot = out_b + (size_t)hop * DFN2_HOP_LEN;
        int valid = -1;
        ainr_test_hop_analysis(&caller_hop, pcm_in + (size_t)hop * DFN2_HOP_LEN, enh_re, enh_im);
        CHECK(dfn2_prepost_pre_process_freq(p_all, enh_re, enh_im) == 1);
        CHECK(dfn2_prepost_frame_open(p_all, &inputs, &outputs) == 0);
        fake_run(&inputs, &outputs);
        CHECK(dfn2_prepost_frame_finish(p_all) == 0);
        CHECK(dfn2_prepost_post_process_freq(p_all, enh_re, enh_im, &valid) == 0);
        CHECK(valid == (hop >= LOOKAHEAD));
        ainr_test_hop_synthesis(&caller_hop, enh_re, enh_im, slot);   /* no valid test */
        if (!valid) CHECK(all_zero(slot, DFN2_HOP_LEN));
    }
    ainr_test_hop_close(&caller_hop);
    CHECK(any_nonzero(out_a, (size_t)HOPS * DFN2_HOP_LEN));
    CHECK(identical(out_a, out_b, (size_t)HOPS * DFN2_HOP_LEN));
    stop(p_valid);
    stop(p_all);
    return 0;
}

/* ---- identity: the pool identity carries the lookahead pair ----------- */

/* A pool sized by a build for another (mask, df) pair is refused by this one:
 * req_dump writes this build's MemReq, req_check hands a dumped one to
 * _init_ex as the expected identity. Mode 1 expects it accepted, 0 refused, and
 * 2 refused although size and alignment were made equal to this build's, so
 * only the build identity can tell the pools apart. A refusal leaves the pool
 * bytes as they were. */
static int case_req_dump(const char *path) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    FILE *file;
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(&req, sizeof req, 1, file) == 1);
    CHECK(fclose(file) == 0);
    printf("req: mask=%d df=%d hash=%u\n", DFN2_MASK_LOOKAHEAD,
           DFN2_DF_LOOKAHEAD, (unsigned)req.build_flags_hash);
    return 0;
}

static int case_req_check(const char *path, int mode) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req, expected;
    DFN2Prepost *p;
    void *pool;
    unsigned char *before;
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fread(&expected, sizeof expected, 1, file) == 1);
    CHECK(fclose(file) == 0);
    config_freq(&cfg);
    CHECK(dfn2_prepost_get_mem_size(&cfg, &req) == 0);
    pool = alloc_aligned(req.alignment, (size_t)req.bytes);
    before = malloc((size_t)req.bytes);
    CHECK(pool != NULL && before != NULL);
    memset(pool, 0xa5, (size_t)req.bytes);
    memcpy(before, pool, (size_t)req.bytes);
    if (mode == 2) {
        expected.bytes = req.bytes;
        expected.alignment = req.alignment;
    }
    p = start_ex(pool, (size_t)req.bytes, &cfg, &expected);
    CHECK((p != NULL) == (mode == 1));
    if (p) stop(p);
    else CHECK(memcmp(pool, before, (size_t)req.bytes) == 0);
    free(before);
    free(pool);
    return 0;
}

int main(int argc, char **argv) {
    int status;

    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <case> [req_dump file | req_check file 0|1|2]\n",
                argv[0]);
        return 2;
    }
    /* The arrays of this driver are sized by the test grid's macros; the library
     * derives the same numbers from the grid it is handed. They must agree, or a
     * tensor would overrun a buffer here without any case noticing. */
    {
        const DFN2Grid *grid = dfn2_test_grid();
        CHECK(dfn2_grid_validate(grid) == DFN2_GRID_OK);
        CHECK(grid->sample_rate == DFN2_SR && grid->n_fft == DFN2_N_FFT);
        CHECK(grid->hop == DFN2_HOP_LEN && grid->n_bins == DFN2_N_BINS);
        CHECK(grid->n_erb == DFN2_N_ERB && grid->df_bins == DFN2_DF_BINS);
        CHECK(grid->mask_lookahead == DFN2_MASK_LOOKAHEAD);
        CHECK(grid->df_lookahead == DFN2_DF_LOOKAHEAD);
        CHECK(grid->df_history == DFN2_DF_HISTORY && grid->ring == DFN2_DF_RING);
        CHECK(grid->df_order == DFN2_DF_ORDER && grid->frames_per_call == 1);
        CHECK(grid->stft_scale == DFN2_NORMALIZED_STFT_SCALE);
        CHECK(grid->stft_unscale == DFN2_NORMALIZED_STFT_UNSCALE);
        CHECK(dfn2_prepost_erb_window_elements(grid) ==
              (size_t)TEST_ERB_WINDOW_ELEMENTS);
        CHECK(dfn2_prepost_spec_window_elements(grid) ==
              (size_t)TEST_SPEC_WINDOW_ELEMENTS);
        CHECK(dfn2_prepost_erb_mask_elements(grid) ==
              (size_t)TEST_ERB_MASK_ELEMENTS);
        CHECK(dfn2_prepost_coefs_elements(grid) == (size_t)TEST_COEFS_ELEMENTS);
        CHECK(dfn2_prepost_alpha_elements(grid) == (size_t)TEST_ALPHA_ELEMENTS);
        CHECK(dfn2_prepost_latency_frames(grid) == LOOKAHEAD);
    }
    build_erb();
    /* frame_bypass's identity is exact only because a unit band mask expands
     * to a unit bin gain.  Pin that here, in the same accumulation order
     * df_common_expand_mask() uses, so a broken fixture fails loudly
     * instead of turning the skip case into an approximation. */
    CHECK(erb_inv_is_partition_of_unity());
    fill_pcm(pcm_in, HOPS * DFN2_HOP_LEN);

    if (argc == 3 && !strcmp(argv[1], "req_dump"))
        status = case_req_dump(argv[2]);
    else if (argc == 4 && !strcmp(argv[1], "req_check"))
        status = case_req_check(argv[2], atoi(argv[3]));
    else if (strcmp(argv[1], "equiv") == 0)     status = case_equiv();
    else if (strcmp(argv[1], "freq") == 0)      status = case_freq();
    else if (strcmp(argv[1], "freqpool") == 0)  status = case_freqpool();
    else if (strcmp(argv[1], "lifecycle") == 0) status = case_lifecycle();
    else if (strcmp(argv[1], "reject") == 0)    status = case_reject();
    else if (strcmp(argv[1], "reset") == 0)     status = case_reset();
    else if (strcmp(argv[1], "skip") == 0)      status = case_skip();
    else if (strcmp(argv[1], "guard") == 0)     status = case_guard();
    else if (strcmp(argv[1], "txn") == 0)       status = case_txn();
    else if (strcmp(argv[1], "dual_identity") == 0) status = case_dual_identity();
    else if (strcmp(argv[1], "dual_split") == 0)    status = case_dual_split();
    else if (strcmp(argv[1], "dual_guards") == 0)   status = case_dual_guards();
    else if (strcmp(argv[1], "model_run") == 0)     status = case_model_run();
    else if (strcmp(argv[1], "descriptor") == 0)    status = case_descriptor();
    else if (strcmp(argv[1], "state_inplace") == 0) status = case_state_inplace();
    else if (strcmp(argv[1], "inplace_nan") == 0)   status = case_inplace_nan();
    else if (strcmp(argv[1], "generation") == 0)    status = case_generation();
    else if (strcmp(argv[1], "warmup") == 0)        status = case_warmup();
    else if (strcmp(argv[1], "warmup_synthesis") == 0) status = case_warmup_synthesis();
    else {
        fprintf(stderr, "unknown case: %s\n", argv[1]);
        status = 2;
    }
    return status;
}
'''


@pytest.fixture(scope='module')
def audio_common_lib():
    """audiolib's NE10 static lib (the default backend), built and located the way the other AINR C tests do it."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    return dfn_pairs.audio_common_lib()


# Every (DFN2_MASK_LOOKAHEAD, DFN2_DF_LOOKAHEAD) pair built here is one the model
# accepts. The shipped pair is the default of tests/dfn2_test_grid.h and is built with
# no override; the others are built with -DDFN2_MASK_LOOKAHEAD / -DDFN2_DF_LOOKAHEAD,
# which only that test header reads. The library reads neither macro: it takes the pair
# from the DFN2Grid its caller builds, and a board sample builds that grid with
# dfn2_grid_make() from its own table.
_LOOKAHEAD_PAIRS = dfn_pairs.c_build_pairs(FAMILY)
_SHIPPED_PAIR = dfn_pairs.shipped_pair(FAMILY)
# The argument-free cases the driver dispatches on argv[1]; every case the
# driver defines must be among them, so a case written another way cannot be
# silently left out of the per-pair run.
_CASES = tuple(re.findall(r'strcmp\(argv\[1\], "(\w+)"\) == 0\)', _DRIVER))
assert set(_CASES) == (
    set(re.findall(r'^static int case_(\w+)\(', _DRIVER, re.M))
    - {'req_dump', 'req_check'})


def _build_driver(tmp_path_factory, audio_common_lib, pair=None):
    """One executable for every case: the class plus the two TUs it composes,
    compiled at the house flags with -Werror. `pair` overrides the lookahead
    macros of the test grid header; None builds that header's defaults."""
    work = tmp_path_factory.mktemp('dfn2_prepost_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), '-I', str(dfn_pairs.AINR / 'tests'),
         *([] if pair is None else dfn_pairs.defines(FAMILY, pair)),
         str(source),
         *[str(dfn_pairs.source_path(FAMILY, kind)) for kind in _SOURCES],
         audio_common_lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


@pytest.fixture(scope='module')
def driver(tmp_path_factory, audio_common_lib):
    return _build_driver(tmp_path_factory, audio_common_lib)


_OTHER_PAIRS = [pair for pair in _LOOKAHEAD_PAIRS if pair != _SHIPPED_PAIR]


@pytest.fixture(scope='module')
def other_pair_drivers(tmp_path_factory, audio_common_lib):
    return {pair: _build_driver(tmp_path_factory, audio_common_lib, pair)
            for pair in _OTHER_PAIRS}


@pytest.fixture(params=_OTHER_PAIRS, ids=dfn_pairs.pair_id)
def pair_driver(request, other_pair_drivers):
    return other_pair_drivers[request.param]


def _run(driver, case):
    done = subprocess.run([str(driver), case], capture_output=True, text=True)
    assert done.returncode == 0, (
        'case %s failed (rc=%d)\n%s' % (case, done.returncode, done.stderr))
    return done.stdout


def test_class_matches_the_hand_composed_path(driver):
    """The class vs analysis + dfn2_compute_features + the model-I/O
    transaction + dfn2_enhance_spectrum_stream + synthesis wired up by hand from
    the two composed TUs, byte-identical over 400 hops -- with the
    attenuation limit both disabled and at -20 dB. Also pins the model
    lookahead (MASK + DF hops): `written` is 0 for exactly the first MASK + DF hops and
    DFN2_HOP_LEN after, and the reported output frame index is unavailable
    until the first emission and then counts source frames without a gap."""
    assert 'equiv: atten_lim_db' in _run(driver, 'equiv')


def test_freq_entries_driven_by_the_callers_own_transform(driver):
    """The spectrum entries fed from a framing of the caller's own and
    re-synthesised by the caller must reproduce the same class behind the
    host's hop-in / hop-out wrapper sample for sample; the warm-up emissions
    report valid == 0 and carry an all-zero spectrum."""
    _run(driver, 'freq')


def test_stale_pool_gate_holds(driver):
    """The MemReq must carry the instance's identity, and _init_ex must accept
    the matching descriptor while refusing any single-field perturbation of its
    own. The identity folds in the whole grid: a request sized for a grid that
    differs in any one value (with size and alignment forged equal) is refused
    in the same binary."""
    stdout = _run(driver, 'freqpool')
    assert 'pool = ' in stdout


def test_create_destroy_roundtrip_and_pool_init_agree(driver):
    """Both house lifecycles: _create/_destroy on the heap and _init on a
    caller pool produce identical output, _destroy is NULL-safe and a genuine
    no-op on a pool instance, the accessors report the graph contract this
    build binds, and the runtime setters reject NULL/non-finite input while
    accepting good values."""
    _run(driver, 'lifecycle')


def test_reject_first_validation(driver):
    """_get_mem_size leaves *req untouched when it refuses (NULL args, a
    missing ERB matrix, a non-finite attenuation limit, a grid that is NULL
    or hand-edited); _init refuses a NULL, undersized or misaligned pool; and
    the stage calls refuse NULLs and out-of-order use. dfn2_grid_make() returns
    the status of the first failing value (sizes no build bounds are
    accepted), and no object takes a grid dfn2_grid_validate() refuses."""
    _run(driver, 'reject')


def test_reset_clears_state(driver):
    """N hops, reset, the same N hops again -> byte-identical, asserted
    against a NO-reset second run first so it cannot pass by the output being
    state-independent. A reset mid-frame drops the open frame rather than
    stranding it."""
    _run(driver, 'reset')


def test_frame_bypass_is_exact_identity_and_freezes_the_state(driver):
    """Driven entirely on frame_bypass, the class is a bit-exact delay line of
    MASK_LOOKAHEAD + DF_LOOKAHEAD frames. The state is written in place, so a
    skip taken over a COMPLETE accelerator result leaves both recurrent
    tensors exactly as the accelerator wrote them (neither restored nor
    zeroed) while the two feature windows still slide; a skip over a frame the
    accelerator never ran writes nothing; and a following commit does move
    the state -- so the freeze is not measuring a constant."""
    _run(driver, 'skip')


def test_frame_state_machine_guards(driver):
    """The contract gates: a second pre-stage with a frame open is refused
    without eating the frame; the ERB and attenuation setters are refused while
    a frame is open and accepted again after it closes; a commit with no
    frame_open behind it is refused; a commit that failed on a non-finite
    output disarms the transaction until a fresh frame_open NaN-refills the
    three heads (the state is the accelerator's input and is not prefilled);
    and frame_bypass needs no frame_open behind it."""
    _run(driver, 'guard')


def test_failed_commit_zeroes_the_state_and_leaves_the_frame_open(driver):
    """Poison one element of one accelerator output tensor and commit: the
    accelerator wrote the recurrent state in place, so the refusal cannot roll
    it back -- both recurrent tensors are zero, the two feature windows are
    byte-identical to a snapshot taken before it, the frame stays open and
    frame_bypass is the documented recovery. The whole 61-hop stream matches a
    run that zeroed the state by hand and skipped that hop, and differs from
    one that committed it and from one that skipped without zeroing. Swept
    over all five output tensors at three indices."""
    assert 'txn:' in _run(driver, 'txn')


def test_dual_entry_with_one_spectrum_is_the_single_entry(driver):
    """pre_process_freq_dual(x, x) must be pre_process_freq(x) byte for byte:
    every emitted spectrum, the emitted frame index and the published head
    bin gain. Catches a dual path that perturbs the feature EMA, the graph
    window or the compose clock; it cannot catch an est/app swap (see the
    split case for that)."""
    _run(driver, 'dual_identity')


def test_dual_entry_estimates_on_est_and_applies_on_app(driver):
    """app = est with the high band zeroed: the emitted high band is exactly
    zero (it comes from app) while the low band and the published gain equal
    the single-input run's (the mask follows est; the deep filter reads only
    the low band the two share). A swapped build emits est's non-zero high
    band and a different mask."""
    _run(driver, 'dual_split')


def test_dual_entry_guards_and_gain_accessor(driver):
    """The dual entry is refused on any NULL argument and with a frame open
    and a frame_bypass closes it; the carve version is 11."""
    _run(driver, 'dual_guards')


def test_model_run_frame_finishes_or_takes_the_identity(driver):
    """dfn2_model_run_frame: contract errors (-1) for no open frame, a NULL
    model/instance, no infer callback, a NULL io_descriptor or one whose
    mask_lookahead or df_lookahead differs from the instance's grid, or one
    made for a grid that differs in any other value; a full write returns 1 and steps the
    recurrent state; a refused inference (nothing written) returns 0, leaves
    every recurrent tensor byte-identical and still emits; a partial write
    (state written in place, one head unwritten) returns 0 with both
    recurrent tensors zeroed and still emits; a model that always refuses is
    exactly the all-frame_bypass stream."""
    assert 'model_run:' in _run(driver, 'model_run')


def test_model_io_descriptor_default_validates_and_every_field_is_checked(driver):
    """validate(grid, default(grid)) == 0 and a +1 perturbation of every one of
    the 12 external fields is refused, so a field added to the struct but omitted
    from validate cannot pass (the six interior fields are bookkeeping the
    library does not read); the descriptor of a grid that differs in one value is
    refused against this grid, and a refused grid has no descriptor."""
    _run(driver, 'descriptor')


def test_recurrent_state_is_one_buffer_updated_in_place(driver):
    """The input and `*_next` views of each recurrent tensor are the same
    address on every frame -- across commit, a refused commit, skip and reset
    -- and the accelerator's in-place write is what the next frame reads."""
    _run(driver, 'state_inplace')


def test_nan_through_the_in_place_pointer_is_refused_and_zeroes_the_state(driver):
    """Mutation gate for the finite check: one NaN written into one element of
    each recurrent tensor through the in-place pointer makes frame_finish
    return -1 and leaves both tensors zero, and the next healthy frame
    commits. Weakening the finite check makes this case fail."""
    _run(driver, 'inplace_nan')


def test_state_generation_counts_every_zeroing_and_nothing_else(driver):
    """The generation is 1 after init and advances by exactly one for each
    reset, each warm-up frame (finish or bypass) and each refused commit (bad
    state element or bad head); commit and skip of a real frame never move it.
    A runtime refills its fixed-point state from it."""
    _run(driver, 'generation')


def test_warmup_frames_are_opened_and_their_heads_discarded(driver):
    """pre_process returns 1 on the first mask_lookahead hops too (window
    [0, .., 0, x0..xt], heads NaN, windows byte-identical to a hand-driven
    feature front end). frame_finish of such a frame never reads its heads (NaN
    heads, garbage or non-finite state: still 0), leaves the registered state
    zero, advances the generation by exactly one (also unregistered, also via
    frame_bypass and dfn2_model_run_frame, which calls the accelerator once per
    warm-up frame) and emits nothing; a second finish or bypass is refused and
    moves nothing. The first real frame starts from zero state, reads its heads
    (a NaN head is refused) and a reset makes the warm-up due again; a reset
    inside a warm-up frame drops it. With mask_lookahead 0 nothing changes."""
    _run(driver, 'warmup')


def test_synthesis_may_be_pushed_on_every_hop(driver):
    """Pushing the post_process spectrum into an AC_OLA_ACCUMULATE AcIstft on
    every hop, the warm-up hops included, gives the stream that pushes only the
    valid hops, sample for sample: the warm-up spectrum is all zero."""
    _run(driver, 'warmup_synthesis')


@pytest.mark.parametrize('case', _CASES)
def test_every_case_holds_for_every_other_lookahead_pair(pair_driver, case):
    """The whole driver, rebuilt for each non-shipped (mask, df) lookahead
    pair: warm-up length, delay-line length, readiness and the frame clocks
    all follow the compiled constants rather than the shipped 1/1."""
    _run(pair_driver, case)


def test_a_pool_identity_from_another_lookahead_pair_is_refused(
        tmp_path, driver, other_pair_drivers):
    """The pool identity folds in the grid's lookahead pair: a host whose grid
    names one trained pair refuses a pool (and so a graph description) sized for
    another: refused outright, and still refused when the dumped request is
    given this host's size and alignment (identity alone)."""
    builds = {_SHIPPED_PAIR: driver, **other_pair_drivers}
    dumps, hashes = {}, set()
    for pair, exe in builds.items():
        path = tmp_path / ('req_%d%d.bin' % pair)
        out = subprocess.run([str(exe), 'req_dump', str(path)],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        dumps[pair] = path
        hashes.add(re.search(r'hash=(\d+)', out.stdout).group(1))
    assert len(hashes) == len(_LOOKAHEAD_PAIRS)
    for host, exe in builds.items():
        for pool, path in dumps.items():
            for mode in ((1,) if host == pool else (0, 2)):
                done = subprocess.run(
                    [str(exe), 'req_check', str(path), str(mode)],
                    capture_output=True, text=True)
                assert done.returncode == 0, (host, pool, mode, done.stderr)
