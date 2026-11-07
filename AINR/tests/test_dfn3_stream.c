/* DFN3 public streaming and lifecycle regression on spectra. No weights or accelerator are required: known
 * heads make tap selection, timing and state observable. The lookahead pair is a compile-time preset of this
 * program (-DDFN3_MASK_LOOKAHEAD=m -DDFN3_DF_LOOKAHEAD=d, default 1/1; test_dfn3_stream_c.py), the rest of the
 * grid is the shipped model's, and every expectation derives from the pair. DELAY frames separate a capture
 * from its output; every hop runs the model, the first MASK being warm-up frames whose heads are discarded, and
 * with DF > MASK some hops emit nothing yet. A refused call must leave the carved state block (or the whole
 * pool) byte-identical. The registration rule, the board form (nothing registered), the stale-pool gate, the
 * scan of the registered state and the warm-up frames have their own functions below. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "DeepFilterNet3/dfn3_prepost.h"

#ifndef DFN3_MASK_LOOKAHEAD
#define DFN3_MASK_LOOKAHEAD 1
#endif
#ifndef DFN3_DF_LOOKAHEAD
#define DFN3_DF_LOOKAHEAD 1
#endif

/* The shipped model: config.ini [signal], [model], [feature] and the exported state tensors; test_dfn3_contract.py
 * pins every value. */
#define DFN3_TEST_SAMPLE_RATE         48000
#define DFN3_TEST_N_FFT               1024
#define DFN3_TEST_N_ERB               32
#define DFN3_TEST_DF_BINS             96
#define DFN3_TEST_DF_ORDER            5
#define DFN3_TEST_ERB_NORM_TAU_SEC    1.0f
#define DFN3_TEST_ERB_NORM_INIT_LO_DB -60.0f
#define DFN3_TEST_ERB_NORM_INIT_HI_DB -90.0f
#define DFN3_TEST_ERB_NORM_SCALE_DB   40.0f
#define DFN3_TEST_SPEC_NORM_TAU_SEC   1.0f
#define DFN3_TEST_SPEC_NORM_INIT_LO   0.001f
#define DFN3_TEST_SPEC_NORM_INIT_HI   0.0001f
#define DFN3_TEST_GRU_ROWS            5     /* h_gru [1, rows, 1, hidden] */
#define DFN3_TEST_GRU_HIDDEN          256
#define DFN3_TEST_ENCODER_CHANNELS    64    /* df_convp_history [1, channels, window, df_bins] */
#define DFN3_TEST_DF_PATHWAY_WINDOW   5

enum {
    N_BINS = DFN3_TEST_N_FFT / 2 + 1,
    GRU_ELEMENTS = DFN3_TEST_GRU_ROWS * DFN3_TEST_GRU_HIDDEN,
    CONVP_ELEMENTS = DFN3_TEST_ENCODER_CHANNELS * DFN3_TEST_DF_PATHWAY_WINDOW * DFN3_TEST_DF_BINS
};
#define DELAY   (DFN3_MASK_LOOKAHEAD > DFN3_DF_LOOKAHEAD ? DFN3_MASK_LOOKAHEAD : DFN3_DF_LOOKAHEAD)
#define HISTORY (DFN3_TEST_DF_ORDER - DFN3_DF_LOOKAHEAD - 1)
#define WARM    ((unsigned int)DFN3_MASK_LOOKAHEAD)   /* warm-up frames after every init or reset */

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "DFN3 check failed at %d: %s\n", __LINE__, #x); \
    return 1; } } while (0)
#define CLOSE(a,b) CHECK(fabsf((a)-(b)) < 3e-5f)

static DFN3Grid G;
static uint32_t digest = 2166136261u;
static uint32_t build_hash;   /* pool identity; differs for every lookahead pair */
static void digest_floats(const float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof(bits));
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            digest ^= (bits >> shift) & 0xffu;
            digest *= 16777619u;
        }
    }
}
static float fwd[N_BINS * DFN3_TEST_N_ERB], inv[DFN3_TEST_N_ERB * N_BINS];
static float re[N_BINS], im[N_BINS];
static float out_re[N_BINS], out_im[N_BINS];
static float mask[DFN3_TEST_N_ERB];
static float coefs[DFN3_TEST_DF_BINS * DFN3_TEST_DF_ORDER * 2];
static float gru_buf[GRU_ELEMENTS], convp_buf[CONVP_ELEMENTS];
#define COEF(k, tap, c) coefs[((size_t)(k) * (size_t)G.df_order + (size_t)(tap)) * 2u + (size_t)(c)]

static void matrices(void) {
    for (int k = 0; k < G.n_bins; ++k) {
        int b = k * G.n_erb / G.n_bins;
        fwd[(size_t)k * (size_t)G.n_erb + (size_t)b] = 1.0f;
        inv[(size_t)b * (size_t)G.n_bins + (size_t)k] = 1.0f;
    }
}
/* Spectrum of a source frame; frames before the stream start are zero, like
 * the zero rows the deep filter reads there. */
static float ref_re(int frame, int k) {
    return frame < 0 ? 0.0f : 0.1f * (float)(frame + 1) + 0.0001f * (float)k;
}
static float ref_im(int frame, int k) {
    return frame < 0 ? 0.0f : -0.03f * (float)(frame + 1) + 0.0002f * (float)k;
}
static void spectrum(int frame) {
    for (int k = 0; k < G.n_bins; ++k) {
        re[k] = ref_re(frame, k);
        im[k] = ref_im(frame, k);
    }
}
static void identity_heads(DFN3PrepostOutputs *o) {
    for (size_t i = 0; i < o->erb_mask_elements; ++i) o->erb_mask[i] = 1.0f;
    memset(o->coefs, 0, o->coefs_elements * sizeof(float));
    for (int k = 0; k < G.df_bins; ++k)
        o->coefs[((size_t)k * (size_t)G.df_order + (size_t)HISTORY) * 2u] = 1.0f;
}
static int callback(void *user, const DFN3PrepostInputs *i,
                     DFN3PrepostOutputs *o) {
    (void)user; (void)i;
    identity_heads(o);
    return 0;
}
static DFN3GridSpec test_spec(void) {
    const DFN3GridSpec spec = {
        .sample_rate = DFN3_TEST_SAMPLE_RATE,
        .n_fft = DFN3_TEST_N_FFT,
        .n_erb = DFN3_TEST_N_ERB,
        .df_bins = DFN3_TEST_DF_BINS,
        .df_order = DFN3_TEST_DF_ORDER,
        .mask_lookahead = DFN3_MASK_LOOKAHEAD,
        .df_lookahead = DFN3_DF_LOOKAHEAD,
        .erb_norm_tau_sec = DFN3_TEST_ERB_NORM_TAU_SEC,
        .erb_norm_init_lo_db = DFN3_TEST_ERB_NORM_INIT_LO_DB,
        .erb_norm_init_hi_db = DFN3_TEST_ERB_NORM_INIT_HI_DB,
        .erb_norm_scale_db = DFN3_TEST_ERB_NORM_SCALE_DB,
        .spec_norm_tau_sec = DFN3_TEST_SPEC_NORM_TAU_SEC,
        .spec_norm_init_lo = DFN3_TEST_SPEC_NORM_INIT_LO,
        .spec_norm_init_hi = DFN3_TEST_SPEC_NORM_INIT_HI,
    };
    return spec;
}

static int config(DFN3PrepostConfig *cfg, int registered) {
    if (dfn3_prepost_config_defaults(cfg, &G) != 0) return -1;
    cfg->erb_fwd = fwd; cfg->erb_inv = inv;
    if (registered) {
        cfg->gru_hidden = gru_buf; cfg->gru_hidden_elements = GRU_ELEMENTS;
        cfg->df_convp_history = convp_buf; cfg->df_convp_history_elements = CONVP_ELEMENTS;
    }
    return 0;
}

/* A state header with its carved block, and a byte copy of both: the header holds pointers and clocks, the
 * block holds every ring and normaliser a refused call could touch. */
typedef struct { DFN3State st; void *block; size_t bytes; } Dsp;
typedef struct { DFN3State st; unsigned char *block; } Snap;
static int dsp_make(Dsp *d) {
    d->bytes = dfn3_state_mem_size(&G);
    d->block = aligned_alloc(16u, d->bytes);
    if (!d->block) return -1;
    if (dfn3_state_init(&d->st, d->block, d->bytes, &G) != 0) return -1;
    dfn3_set_erb_matrices(&d->st, fwd, inv);
    return 0;
}
static void dsp_free(Dsp *d) { free(d->block); }
static int snap_save(Snap *s, const Dsp *d) {
    memcpy(&s->st, &d->st, sizeof d->st);
    s->block = (unsigned char *)malloc(d->bytes);
    if (!s->block) return -1;
    memcpy(s->block, d->block, d->bytes);
    return 0;
}
static int snap_same(const Snap *s, const Dsp *d) {
    return memcmp(&s->st, &d->st, sizeof d->st) == 0 &&
           memcmp(s->block, d->block, d->bytes) == 0;
}
static void snap_free(Snap *s) { free(s->block); s->block = NULL; }

/* A pure newest tap proves DF is fed the raw spectrum DF frames after its
 * target and that high bins use the mask of target = current-DELAY. This fails
 * a cascade or a duplicated delay. */
static int tap_timing(void) {
    Dsp d;
    long long index = -99;
    CHECK(dsp_make(&d) == 0);
    memset(coefs, 0, sizeof(coefs));
    for (int k = 0; k < G.df_bins; ++k)
        COEF(k, G.df_order - 1, 0) = 1.0f;
    for (int b = 0; b < G.n_erb; ++b) mask[b] = 0.1f;
    spectrum(0);
    {
        float erb[DFN3_TEST_N_ERB], spec[2 * DFN3_TEST_DF_BINS];
        dfn3_compute_features(&d.st, re, im, erb, spec);
        for (int k = 0; k < G.n_erb; ++k) CHECK(isfinite(erb[k]));
        for (int k = 0; k < 2 * G.df_bins; ++k) CHECK(isfinite(spec[k]));
        digest_floats(erb, (size_t)G.n_erb);
        digest_floats(spec, 2u * (size_t)G.df_bins);
    }
    /* Heads are refused while the first head's window is not complete, and
     * required after. */
    for (int t = 0; t < DFN3_MASK_LOOKAHEAD; ++t) {
        Snap s;
        spectrum(t);
        CHECK(snap_save(&s, &d) == 0);
        CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask, coefs,
                                  0, out_re, out_im, &index) == -1);
        CHECK(snap_same(&s, &d));
        snap_free(&s);
        CHECK(index == -99);
        CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 0, NULL, NULL,
                                  0, out_re, out_im, &index) == 0);
    }
    for (int t = DFN3_MASK_LOOKAHEAD; t < 16 + DELAY; ++t) {
        const int target = t - DELAY;
        Snap s;
        spectrum(t);
        mask[0] = (float)NAN;
        CHECK(snap_save(&s, &d) == 0);
        CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask, coefs,
                                  0, out_re, out_im, &index) == -1);
        CHECK(snap_same(&s, &d));
        snap_free(&s);
        mask[0] = 0.1f;
        if (target < 0) {   /* heads taken, first output frame not complete */
            CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask,
                                      coefs, 0, out_re, out_im,
                                      &index) == 0);
            continue;
        }
        CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask, coefs,
                                  0, out_re, out_im, &index) == 1);
        CHECK(index == target);
        digest_floats(out_re, (size_t)G.n_bins);
        digest_floats(out_im, (size_t)G.n_bins);
        for (int k = 0; k < G.df_bins; ++k) {
            const int source = target + DFN3_DF_LOOKAHEAD;
            CLOSE(out_re[k], ref_re(source, k));
            CLOSE(out_im[k], ref_im(source, k));
        }
        for (int k = G.df_bins; k < G.n_bins; ++k) {
            CLOSE(out_re[k], ref_re(target, k) * 0.1f);
            CLOSE(out_im[k], ref_im(target, k) * 0.1f);
        }
    }
    {
        Snap s;
        CHECK(snap_save(&s, &d) == 0);
        re[0] = (float)INFINITY;
        CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask, coefs,
                                  0, out_re, out_im, NULL) == -1);
        CHECK(snap_same(&s, &d));
        snap_free(&s);
    }
    dsp_free(&d);
    return 0;
}

/* Each tap in turn is the only nonzero one: the delayed frame's window is
 * [target-HISTORY, ..., target+DF], including the oldest tap, which is the
 * first to be lost if the frame ring is one slot short. Frames before the
 * stream start read as zero. */
static int every_tap_reads_its_frame(void) {
    long long index = -99;
    for (int b = 0; b < G.n_erb; ++b) mask[b] = 1.0f;
    for (int tap = 0; tap < G.df_order; ++tap) {
        Dsp d;
        CHECK(dsp_make(&d) == 0);
        memset(coefs, 0, sizeof(coefs));
        for (int k = 0; k < G.df_bins; ++k) COEF(k, tap, 0) = 1.0f;
        for (int t = 0; t < 24; ++t) {
            spectrum(t);
            if (t < DFN3_MASK_LOOKAHEAD) {
                CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 0, NULL, NULL, 0,
                                          out_re, out_im, &index) == 0);
                continue;
            }
            if (t < DELAY) {
                CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask,
                                          coefs, 0, out_re, out_im,
                                          &index) == 0);
                continue;
            }
            CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask,
                                      coefs, 0, out_re, out_im,
                                      &index) == 1);
            {
                const int target = t - DELAY;
                const int source = target - HISTORY + tap;
                CHECK(index == target);
                for (int k = 0; k < G.df_bins; ++k) {
                    CLOSE(out_re[k], ref_re(source, k));
                    CLOSE(out_im[k], ref_im(source, k));
                }
            }
        }
        dsp_free(&d);
    }
    return 0;
}

/* Heads that differ per frame: the mask level and the chosen tap of frame h
 * must come out at frame h's own output. The heads of frame h arrive at call
 * h+MASK; when DF > MASK they wait in the ring until call h+DF. */
static float level_of(int frame) { return 0.1f + 0.15f * (float)(frame % 5); }
static int tap_of(int frame) { return (frame * 3 + 1) % DFN3_TEST_DF_ORDER; }

static int heads_stay_with_their_frame(void) {
    Dsp d;
    long long index = -99;
    CHECK(dsp_make(&d) == 0);
    for (int t = 0; t < 30; ++t) {
        const int head = t - DFN3_MASK_LOOKAHEAD;
        const int target = t - DELAY;
        int emitted;
        spectrum(t);
        if (head < 0) {
            CHECK(dfn3_enhance_spectrum_stream(&d.st, re, im, 0, NULL, NULL, 0,
                                      out_re, out_im, &index) == 0);
            continue;
        }
        for (int b = 0; b < G.n_erb; ++b) mask[b] = level_of(head);
        memset(coefs, 0, sizeof(coefs));
        for (int k = 0; k < G.df_bins; ++k) COEF(k, tap_of(head), 0) = 1.0f;
        emitted = dfn3_enhance_spectrum_stream(&d.st, re, im, 1, mask,
                                      coefs, 0, out_re, out_im,
                                      &index);
        CHECK(emitted == (target >= 0));
        if (target < 0) continue;
        CHECK(index == target);
        for (int k = 0; k < G.df_bins; ++k) {
            const int source = target - HISTORY + tap_of(target);
            CLOSE(out_re[k], ref_re(source, k));
            CLOSE(out_im[k], ref_im(source, k));
        }
        for (int k = G.df_bins; k < G.n_bins; ++k) {
            CLOSE(out_re[k], ref_re(target, k) * level_of(target));
            CLOSE(out_im[k], ref_im(target, k) * level_of(target));
        }
    }
    dsp_free(&d);
    return 0;
}

/* The graph window of heads[t] is [t-(2-MASK), ..., t+MASK] with zeros before
 * the stream start; the first MASK pushes do not complete a window. The windows
 * are flat runs: erb_window [3][n_erb], spec_window [2][3][df_bins]. */
static int model_io_window_alignment(void) {
    DFN3ModelIOState io;
    const size_t bytes = dfn3_model_io_mem_size(&G);
    void *block = aligned_alloc(16u, bytes);
    const int ne = G.n_erb, df = G.df_bins, rows = DFN3_MODEL_INPUT_FRAMES;
    float erb[DFN3_TEST_N_ERB], spec[2 * DFN3_TEST_DF_BINS];
    CHECK(bytes != 0 && block != NULL);
    CHECK(dfn3_model_io_init(&io, block, bytes, &G, NULL, 0, NULL, 0) == 0);
    for (int n = 0; n < 8; ++n) {
        for (int b = 0; b < ne; ++b) erb[b] = (float)(n + 1);
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < df; ++k)
                spec[c * df + k] = (float)((n + 1) * (c + 1));
        CHECK(dfn3_model_io_push_features(&io, erb, spec) ==
              (n >= DFN3_MASK_LOOKAHEAD ? 1 : 0));
        for (int row = 0; row < rows; ++row) {
            const int frame = n - (rows - 1) + row;
            const float want = frame < 0 ? 0.0f : (float)(frame + 1);
            CHECK(io.erb_window[row * ne] == want);
            CHECK(io.erb_window[row * ne + ne - 1] == want);
            CHECK(io.spec_window[(0 * rows + row) * df] == want);
            CHECK(io.spec_window[(1 * rows + row) * df + df - 1] == 2.0f * want);
        }
        /* heads[t] with t = n - MASK has frame t-(2-MASK)+row in row `row`. */
        if (n >= DFN3_MASK_LOOKAHEAD) {
            const int t = n - DFN3_MASK_LOOKAHEAD;
            CHECK(io.erb_window[0] ==
                  (t - (2 - DFN3_MASK_LOOKAHEAD) < 0 ? 0.0f :
                   (float)(t - (2 - DFN3_MASK_LOOKAHEAD) + 1)));
            CHECK(io.erb_window[(rows - 1) * ne] ==
                  (float)(t + DFN3_MASK_LOOKAHEAD + 1));
        }
    }
    dfn3_model_io_reset(&io);
    CHECK(io.feature_frames_seen == 0);
    for (int b = 0; b < ne; ++b) erb[b] = 1.0f;
    memset(spec, 0, sizeof(spec));
    CHECK(dfn3_model_io_push_features(&io, erb, spec) ==
          (DFN3_MASK_LOOKAHEAD == 0 ? 1 : 0));
    free(block);
    return 0;
}

/* Every hop opens a frame; the model is taken as the identity here (frame_bypass). The first MASK frames are
 * warm-up frames, each one moving the state generation by one; the next DELAY - MASK hops (DF > MASK) are real
 * frames that still emit nothing. After this, hop DELAY is the first with output. */
static int warm_up(DFN3Prepost *p) {
    int valid;
    for (int t = 0; t < DELAY; ++t) {
        const unsigned int gen = dfn3_prepost_state_generation(p);
        spectrum(t);
        CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn3_prepost_frame_bypass(p) == 0);
        CHECK(dfn3_prepost_state_generation(p) == gen + (t < DFN3_MASK_LOOKAHEAD ? 1u : 0u));
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == 0);
    }
    return 0;
}

/* The caller's pool bytes before a refused call. */
static unsigned char *pool_copy;
static int pool_snap(const void *pool, size_t bytes) {
    free(pool_copy);
    pool_copy = (unsigned char *)malloc(bytes);
    if (!pool_copy) return -1;
    memcpy(pool_copy, pool, bytes);
    return 0;
}
static int pool_same(const void *pool, size_t bytes) {
    return memcmp(pool_copy, pool, bytes) == 0;
}

/* registered = 1: a float host that registers both state buffers; 0: the board form, nothing registered. */
static int lifecycle_and_transactions(int registered) {
    DFN3PrepostConfig cfg;
    DFN3PrepostMemReq req, stale;
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    DFN3ModelIoDescriptor descriptor;
    DFN3Prepost *p;
    void *pool;
    unsigned int gen;
    int valid;
    long long index;
    CHECK(config(&cfg, registered) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes % req.alignment == 0);
    CHECK(req.layout_version == 9);
    if (registered) build_hash = req.build_flags_hash;
    pool = aligned_alloc(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    memset(pool, 0xa5, (size_t)req.bytes);
    CHECK(dfn3_prepost_init(pool, (size_t)req.bytes - 1, &cfg) == NULL);
    CHECK(dfn3_prepost_init((char *)pool + 1, (size_t)req.bytes, &cfg) == NULL);
    stale = req; --stale.layout_version;
    CHECK(dfn3_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &stale) == NULL);
    p = dfn3_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(p != NULL);
    CHECK(dfn3_prepost_layout_version(p) == 9);
    CHECK(dfn3_prepost_model_lookahead_frames(p) == DELAY);
    CHECK(dfn3_prepost_state_generation(p) == 1);
    CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
    CHECK(dfn3_model_io_descriptor_validate(&G, &descriptor) == 0);
    descriptor.layout_version = 8;
    CHECK(dfn3_model_io_descriptor_validate(&G, &descriptor) == -1);
    /* A graph exported for another lookahead pair is refused. */
    CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
    CHECK(descriptor.mask_lookahead == DFN3_MASK_LOOKAHEAD &&
          descriptor.df_lookahead == DFN3_DF_LOOKAHEAD);
    descriptor.mask_lookahead ^= 1;
    CHECK(dfn3_model_io_descriptor_validate(&G, &descriptor) == -1);
    CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
    descriptor.df_lookahead ^= 1;
    CHECK(dfn3_model_io_descriptor_validate(&G, &descriptor) == -1);
    CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == -1);
    CHECK(warm_up(p) == 0);
    spectrum(DELAY);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == -1);
    CHECK(dfn3_prepost_frame_finish(p) == -1);
    CHECK(dfn3_prepost_set_atten_lim(p, -20) == -1);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    /* The state tensors alias, and the aliased run is the registered buffer: a host that forgot to register
     * would compare NULL with NULL. */
    CHECK((const void *)in.gru_hidden == (void *)out.gru_hidden_next);
    CHECK((const void *)in.df_convp_window == (void *)out.df_convp_window_next);
    CHECK(in.gru_hidden == (registered ? gru_buf : NULL));
    CHECK(in.df_convp_window == (registered ? convp_buf : NULL));
    CHECK(in.gru_hidden_elements == (registered ? (size_t)GRU_ELEMENTS : 0u));
    CHECK(in.df_convp_window_elements == (registered ? (size_t)CONVP_ELEMENTS : 0u));
    for (size_t i = 0; i < out.coefs_elements; ++i) CHECK(isnan(out.coefs[i]));
    identity_heads(&out);
    if (registered) {
        out.gru_hidden_next[0] = 3.0f;
        out.df_convp_window_next[0] = 4.0f;
    }
    out.coefs[out.coefs_elements-1] = (float)NAN;
    gen = dfn3_prepost_state_generation(p);
    CHECK(dfn3_prepost_frame_finish(p) == -1);
    CHECK(dfn3_prepost_state_generation(p) == gen + 1u);
    for (size_t i = 0; i < in.gru_hidden_elements; ++i)
        CHECK(out.gru_hidden_next[i] == 0.0f);
    for (size_t i = 0; i < in.df_convp_window_elements; ++i)
        CHECK(out.df_convp_window_next[i] == 0.0f);
    CHECK(dfn3_prepost_frame_finish(p) == -1);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    identity_heads(&out);
    CHECK(dfn3_prepost_frame_finish(p) == 0);
    CHECK(dfn3_prepost_frame_bypass(p) == -1);
    CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
    CHECK(valid == 1);
    CHECK(dfn3_prepost_output_frame_index(p, &index) == 0 && index == 0);
    spectrum(0);
    for (int k = 0; k < G.n_bins; ++k) {
        CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
    }
    /* Reject non-finite input without consuming a frame or poisoning EMA. */
    spectrum(DELAY + 1);
    re[0] = (float)NAN;
    CHECK(pool_snap(pool, (size_t)req.bytes) == 0);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == -1);
    CHECK(pool_same(pool, (size_t)req.bytes));
    spectrum(DELAY + 1);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    identity_heads(&out);
    if (registered) out.gru_hidden_next[0] = (float)INFINITY;
    else out.erb_mask[0] = (float)INFINITY;
    CHECK(dfn3_prepost_frame_finish(p) == -1);
    CHECK(dfn3_prepost_frame_bypass(p) == 0);
    CHECK(dfn3_prepost_output_frame_index(p, &index) == 0 && index == 1);
    gen = dfn3_prepost_state_generation(p);
    dfn3_prepost_reset(p);
    CHECK(dfn3_prepost_state_generation(p) == gen + 1u);
    CHECK(dfn3_prepost_output_frame_index(p, &index) == -1);
    /* Caller pool destruction is repeatable and leaves its contents usable. */
    dfn3_prepost_destroy(p); dfn3_prepost_destroy(p);
    CHECK(warm_up(p) == 0);
    spectrum(DELAY);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    {
        DFN3Model model = {NULL, callback, NULL, &descriptor};
        descriptor.layout_version = 8;
        CHECK(dfn3_model_run_frame(&model, p) == -1);
        descriptor.layout_version = 9;
        CHECK(dfn3_model_run_frame(&model, p) == 1);
    }
    free(pool);
    return 0;
}

/* ---- registration, board form, stale pool, registered-state scan ---- */

static void fill_state(float value) {
    for (size_t i = 0; i < GRU_ELEMENTS; ++i) gru_buf[i] = value;
    for (size_t i = 0; i < CONVP_ELEMENTS; ++i) convp_buf[i] = value;
}
static int state_is(float value) {
    for (size_t i = 0; i < GRU_ELEMENTS; ++i) if (gru_buf[i] != value) return 0;
    for (size_t i = 0; i < CONVP_ELEMENTS; ++i) if (convp_buf[i] != value) return 0;
    return 1;
}

/* The hop after the warm-up: the accelerator must run. */
static int hop_ready(DFN3Prepost *p, int hop) {
    spectrum(hop);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    return 0;
}
static int hop_emit(DFN3Prepost *p) {
    int valid = 0;
    CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0 && valid == 1);
    return 0;
}

/* A cold instance warms up, runs one identity frame and emits it; the views are the registered buffers or NULL. */
static int frame_commits(DFN3Prepost *p, int registered) {
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    CHECK(warm_up(p) == 0);
    CHECK(hop_ready(p, DELAY) == 0);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    CHECK(in.gru_hidden == (registered ? gru_buf : NULL));
    CHECK(in.df_convp_window == (registered ? convp_buf : NULL));
    identity_heads(&out);
    CHECK(dfn3_prepost_frame_finish(p) == 0);
    return hop_emit(p);
}

/* Both pairs set or all four values NULL/0: of the 16 combinations of the four values only those two are accepted,
 * and a refusal leaves the request, the pool and the registered buffers as they were. */
static int registration_is_all_or_none(void) {
    DFN3PrepostConfig good;
    DFN3PrepostMemReq base;
    void *pool;
    CHECK(config(&good, 0) == 0);
    CHECK(dfn3_prepost_get_mem_size(&good, &base) == 0);
    pool = aligned_alloc(base.alignment, (size_t)base.bytes);
    CHECK(pool != NULL);
    for (unsigned int bits = 0; bits < 16u; ++bits) {
        const int accepted = bits == 0u || bits == 15u;
        DFN3PrepostConfig cfg = good;
        DFN3PrepostMemReq req, untouched;
        DFN3Prepost *p;
        cfg.gru_hidden = (bits & 1u) ? gru_buf : NULL;
        cfg.gru_hidden_elements = (bits & 2u) ? GRU_ELEMENTS : 0u;
        cfg.df_convp_history = (bits & 4u) ? convp_buf : NULL;
        cfg.df_convp_history_elements = (bits & 8u) ? CONVP_ELEMENTS : 0u;
        memset(&untouched, 0xa5, sizeof untouched);
        req = untouched;
        fill_state(7.0f);
        memset(pool, 0xa5, (size_t)base.bytes);
        CHECK(pool_snap(pool, (size_t)base.bytes) == 0);
        CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == (accepted ? 0 : -1));
        p = dfn3_prepost_init(pool, (size_t)base.bytes, &cfg);
        CHECK((p != NULL) == accepted);
        if (!accepted) {
            CHECK(memcmp(&req, &untouched, sizeof req) == 0);
            CHECK(pool_same(pool, (size_t)base.bytes) && state_is(7.0f));
            CHECK(dfn3_prepost_create(&cfg) == NULL);
            continue;
        }
        /* Registration is in neither the pool size nor the identity. */
        CHECK(req.bytes == base.bytes && req.build_flags_hash == base.build_flags_hash);
        CHECK(dfn3_prepost_state_generation(p) == 1u);
        CHECK(state_is(bits == 15u ? 0.0f : 7.0f));
        CHECK(frame_commits(p, bits == 15u) == 0);
        CHECK(state_is(bits == 15u ? 0.0f : 7.0f));
    }
    free(pool);
    return 0;
}

static int failing_callback(void *user, const DFN3PrepostInputs *i, DFN3PrepostOutputs *o) {
    (void)user; (void)i; (void)o;
    return -1;
}
static int nan_head_callback(void *user, const DFN3PrepostInputs *i, DFN3PrepostOutputs *o) {
    (void)user; (void)i;
    identity_heads(o);
    o->erb_mask[0] = (float)NAN;
    return 0;
}

/* Nothing registered: the views hold NULL and 0, the generation starts at 1 and moves by one per reset, per
 * warm-up frame (WARM of them after each start) and per refused finish only, and no buffer the library was not
 * given is touched, not even one registered by an earlier instance on the same pool. */
static int board_form_follows_the_documented_generation(void) {
    static struct {
        unsigned char head[32];
        float gru[GRU_ELEMENTS], convp[CONVP_ELEMENTS];
        unsigned char tail[32];
    } foreign, foreign_before;
    DFN3PrepostConfig cfg, registered;
    DFN3PrepostMemReq req;
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    DFN3ModelIoDescriptor descriptor;
    DFN3Prepost *p;
    void *pool;
    int hop = DELAY;
    memset(&cfg, 0xa5, sizeof cfg);
    CHECK(dfn3_prepost_config_defaults(&cfg, &G) == 0);
    CHECK(cfg.gru_hidden == NULL && cfg.gru_hidden_elements == 0u &&
          cfg.df_convp_history == NULL && cfg.df_convp_history_elements == 0u);
    cfg.erb_fwd = fwd; cfg.erb_inv = inv;
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    pool = aligned_alloc(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    CHECK(config(&registered, 1) == 0);
    CHECK(dfn3_prepost_init(pool, (size_t)req.bytes, &registered) != NULL);
    fill_state(5.0f);
    memset(&foreign, 0x5a, sizeof foreign);
    memset(foreign.head, 0xa5, sizeof foreign.head);
    memset(foreign.tail, 0xa5, sizeof foreign.tail);
    memcpy(&foreign_before, &foreign, sizeof foreign);

    p = dfn3_prepost_init(pool, (size_t)req.bytes, &cfg);
    CHECK(p != NULL && dfn3_prepost_state_generation(p) == 1u);
    CHECK(warm_up(p) == 0);
    CHECK(hop_ready(p, hop++) == 0);
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    CHECK(in.gru_hidden == NULL && in.df_convp_window == NULL &&
          in.gru_hidden_elements == 0u && in.df_convp_window_elements == 0u);
    CHECK(out.gru_hidden_next == NULL && out.df_convp_window_next == NULL);
    identity_heads(&out);
    CHECK(dfn3_prepost_frame_finish(p) == 0 && dfn3_prepost_state_generation(p) == 1u + WARM);
    CHECK(hop_emit(p) == 0);
    CHECK(hop_ready(p, hop++) == 0);                       /* a bypassed frame */
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    CHECK(dfn3_prepost_frame_bypass(p) == 0 && dfn3_prepost_state_generation(p) == 1u + WARM);
    CHECK(hop_ready(p, hop++) == 0);                       /* a refused frame: once, then disarmed */
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    identity_heads(&out);
    out.erb_mask[0] = (float)NAN;
    CHECK(dfn3_prepost_frame_finish(p) == -1 && dfn3_prepost_state_generation(p) == 2u + WARM);
    CHECK(dfn3_prepost_frame_finish(p) == -1 && dfn3_prepost_state_generation(p) == 2u + WARM);
    CHECK(dfn3_prepost_frame_bypass(p) == 0 && dfn3_prepost_state_generation(p) == 2u + WARM);
    CHECK(hop_ready(p, hop++) == 0);                       /* the next frame commits again */
    CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
    identity_heads(&out);
    CHECK(dfn3_prepost_frame_finish(p) == 0 && dfn3_prepost_state_generation(p) == 2u + WARM);
    dfn3_prepost_reset(p);
    CHECK(dfn3_prepost_state_generation(p) == 3u + WARM);
    CHECK(frame_commits(p, 0) == 0 && dfn3_prepost_state_generation(p) == 3u + 2u * WARM);
    {
        DFN3Model model = {.user = NULL, .infer = failing_callback, .reset = NULL,
                           .io_descriptor = &descriptor};
        CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
        CHECK(hop_ready(p, hop++) == 0);
        CHECK(dfn3_model_run_frame(&model, p) == 0 && dfn3_prepost_state_generation(p) == 3u + 2u * WARM);
        model.infer = nan_head_callback;
        CHECK(hop_ready(p, hop++) == 0);
        CHECK(dfn3_model_run_frame(&model, p) == 0 && dfn3_prepost_state_generation(p) == 4u + 2u * WARM);
        model.infer = callback;
        CHECK(hop_ready(p, hop++) == 0);
        CHECK(dfn3_model_run_frame(&model, p) == 1 && dfn3_prepost_state_generation(p) == 4u + 2u * WARM);
    }
    CHECK(memcmp(&foreign, &foreign_before, sizeof foreign) == 0);
    for (size_t i = 0; i < GRU_ELEMENTS; ++i) CHECK(gru_buf[i] == 5.0f);
    for (size_t i = 0; i < CONVP_ELEMENTS; ++i) CHECK(convp_buf[i] == 5.0f);
    free(pool);
    return 0;
}

static int variant_grid(DFN3Grid *grid, int which) {
    DFN3GridSpec spec = test_spec();
    switch (which) {
    case 0: spec.n_erb -= 4; break;
    case 1: spec.n_fft /= 2; break;
    case 2: spec.df_bins -= 4; break;
    case 3: spec.mask_lookahead = DFN3_MASK_LOOKAHEAD ? 0 : 1; break;
    case 4: spec.sample_rate += 100; break;
    default: spec.erb_norm_tau_sec *= 2.0f; break;
    }
    return dfn3_grid_make(grid, &spec) == DFN3_GRID_OK ? 0 : -1;
}
enum { N_VARIANTS = 6 };

/* A pool made for another grid, or offered with a descriptor that differs in any compared field, is refused by
 * init_ex() before it writes anything: the pool (here a live instance of the other grid) and the registered buffers
 * are byte-identical afterwards. The matching descriptor is accepted and the instance runs. */
static int stale_pool_is_refused_untouched(void) {
    DFN3PrepostConfig cfg, other;
    DFN3PrepostMemReq req, req_other[N_VARIANTS], bad;
    size_t most;
    void *pool;
    DFN3Prepost *p;
    CHECK(config(&cfg, 1) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    most = (size_t)req.bytes;
    for (int v = 0; v < N_VARIANTS; ++v) {
        DFN3Grid grid;
        CHECK(variant_grid(&grid, v) == 0);
        CHECK(config(&other, 1) == 0);
        CHECK(dfn3_prepost_config_defaults(&other, &grid) == 0);
        other.erb_fwd = fwd; other.erb_inv = inv;
        other.gru_hidden = gru_buf; other.gru_hidden_elements = GRU_ELEMENTS;
        other.df_convp_history = convp_buf; other.df_convp_history_elements = CONVP_ELEMENTS;
        CHECK(dfn3_prepost_get_mem_size(&other, &req_other[v]) == 0);
        CHECK(req_other[v].build_flags_hash != req.build_flags_hash);
        if (req_other[v].bytes > most) most = (size_t)req_other[v].bytes;
    }
    pool = aligned_alloc(16u, (most + 15u) / 16u * 16u);
    CHECK(pool != NULL);
    for (int v = 0; v < N_VARIANTS; ++v) {
        DFN3Grid grid;
        DFN3Prepost *live;
        CHECK(variant_grid(&grid, v) == 0);
        CHECK(config(&other, 1) == 0);
        CHECK(dfn3_prepost_config_defaults(&other, &grid) == 0);
        other.erb_fwd = fwd; other.erb_inv = inv;
        other.gru_hidden = gru_buf; other.gru_hidden_elements = GRU_ELEMENTS;
        other.df_convp_history = convp_buf; other.df_convp_history_elements = CONVP_ELEMENTS;
        memset(pool, 0xa5, most);
        live = dfn3_prepost_init_ex(pool, most, &other, &req_other[v]);
        CHECK(live != NULL);                                  /* the control: its own descriptor passes */
        fill_state(7.0f);
        CHECK(pool_snap(pool, most) == 0);
        CHECK(dfn3_prepost_init_ex(pool, most, &cfg, &req_other[v]) == NULL);
        CHECK(pool_same(pool, most) && state_is(7.0f));
        bad = req_other[v]; bad.bytes = req.bytes; bad.alignment = req.alignment;
        CHECK(dfn3_prepost_init_ex(pool, most, &cfg, &bad) == NULL);   /* the identity alone */
        CHECK(pool_same(pool, most) && state_is(7.0f));
        CHECK(dfn3_prepost_init_ex(pool, most, &other, &req) == NULL);
        CHECK(pool_same(pool, most) && state_is(7.0f));
        CHECK(dfn3_prepost_grid(live)->n_erb == grid.n_erb && dfn3_prepost_hop_size(live) == grid.hop);
    }
    memset(pool, 0xa5, most);
    p = dfn3_prepost_init_ex(pool, most, &cfg, &req);
    CHECK(p != NULL);
    for (int field = 0; field < 6; ++field) {
        bad = req;
        switch (field) {
        case 0: ++bad.descriptor_version; break;
        case 1: ++bad.layout_version; break;
        case 2: bad.reserved0 = 1u; break;
        case 3: bad.build_flags_hash ^= 1u; break;
        case 4: bad.alignment *= 2u; break;
        default: ++bad.bytes; break;
        }
        fill_state(7.0f);
        CHECK(pool_snap(pool, most) == 0);
        CHECK(dfn3_prepost_init_ex(pool, most, &cfg, &bad) == NULL);
        CHECK(pool_same(pool, most) && state_is(7.0f));
    }
    p = dfn3_prepost_init_ex(pool, most, &cfg, &req);
    CHECK(p != NULL && state_is(0.0f));
    CHECK(frame_commits(p, 1) == 0);
    free(pool);
    return 0;
}

/* The library owns the registered state: zeroed at init and at reset, and scanned whole at every finish. A
 * non-finite value in any frame of the pathway window [channels][window][df_bins], the oldest included, refuses
 * the frame, zeroes both buffers and moves the generation once; a finite state is left alone. */
static int registered_state_is_zeroed_and_scanned_whole(void) {
    enum { CH = DFN3_TEST_ENCODER_CHANNELS, K = DFN3_TEST_DF_PATHWAY_WINDOW, BINS = DFN3_TEST_DF_BINS };
    static const struct { int channel, slot, bin; float poison; } at[] = {
        {0, 0, 0, (float)NAN}, {17, 1, 50, (float)NAN}, {40, 2, 95, (float)NAN},
        {63, 3, 7, -(float)INFINITY}, {63, K - 1, BINS - 1, (float)NAN},
    };
    DFN3PrepostConfig cfg;
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    DFN3Prepost *p;
    unsigned int gen;
    int hop;
    CHECK(config(&cfg, 1) == 0);
    fill_state(7.0f);
    p = dfn3_prepost_create(&cfg);
    CHECK(p != NULL && state_is(0.0f));
    fill_state(7.0f);
    gen = dfn3_prepost_state_generation(p);
    dfn3_prepost_reset(p);
    CHECK(state_is(0.0f) && dfn3_prepost_state_generation(p) == gen + 1u);
    CHECK(warm_up(p) == 0);
    hop = DELAY;
    for (size_t n = 0; n < sizeof at / sizeof at[0]; ++n) {
        const size_t index = ((size_t)at[n].channel * K + (size_t)at[n].slot) * BINS + (size_t)at[n].bin;
        CHECK(hop_ready(p, hop++) == 0);
        CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
        identity_heads(&out);
        fill_state(7.0f);
        out.df_convp_window_next[index] = at[n].poison;
        gen = dfn3_prepost_state_generation(p);
        CHECK(dfn3_prepost_frame_finish(p) == -1);
        CHECK(dfn3_prepost_state_generation(p) == gen + 1u && state_is(0.0f));
        CHECK(dfn3_prepost_frame_finish(p) == -1 && dfn3_prepost_state_generation(p) == gen + 1u);
        CHECK(dfn3_prepost_frame_bypass(p) == 0 && hop_emit(p) == 0);
        /* The same state, finite, commits and stays as the accelerator wrote it. */
        CHECK(hop_ready(p, hop++) == 0);
        CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
        identity_heads(&out);
        fill_state(7.0f);
        CHECK(dfn3_prepost_frame_finish(p) == 0 && state_is(7.0f));
        CHECK(dfn3_prepost_state_generation(p) == gen + 1u && hop_emit(p) == 0);
    }
    dfn3_prepost_destroy(p);
    return 0;
}

/* The scan covers exactly the registered count: the last registered element is checked, the one after it is not. */
static int registered_scan_stops_at_the_registered_count(void) {
    DFN3PrepostConfig cfg;
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    DFN3Prepost *p;
    const size_t count = CONVP_ELEMENTS - 5u;
    CHECK(config(&cfg, 1) == 0);
    cfg.df_convp_history_elements = count;
    p = dfn3_prepost_create(&cfg);
    CHECK(p != NULL && warm_up(p) == 0);
    for (int outside = 0; outside < 2; ++outside) {
        fill_state(7.0f);
        CHECK(hop_ready(p, DELAY + outside) == 0);
        CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
        identity_heads(&out);
        convp_buf[outside ? count : count - 1u] = (float)NAN;
        CHECK(dfn3_prepost_frame_finish(p) == (outside ? 0 : -1));
        if (outside) {
            CHECK(isnan(convp_buf[count]) && gru_buf[0] == 7.0f && convp_buf[0] == 7.0f);
        } else {
            CHECK(dfn3_prepost_frame_bypass(p) == 0);
            CHECK(convp_buf[count - 1u] == 0.0f && convp_buf[count] == 7.0f && gru_buf[0] == 0.0f);
        }
        CHECK(hop_emit(p) == 0);
    }
    dfn3_prepost_destroy(p);
    return 0;
}

/* The first MASK frames after init or reset are warm-up frames: opened and run like any other, but their heads
 * (NaN or garbage) are never read, and whatever the accelerator wrote into the registered state is zeroed again,
 * one generation step per frame (also when the state is non-finite), so the first real frame starts from zero state.
 * The frame is closed by finish, bypass or dfn3_model_run_frame() alike. With MASK == 0 there is none. */
static int stand_in_garbage(void *user, const DFN3PrepostInputs *i, DFN3PrepostOutputs *o) {
    (void)user; (void)i;
    fill_state(7.0f);
    o->erb_mask[0] = (float)NAN;
    return 0;
}
static int stand_in_garbage_then_fail(void *user, const DFN3PrepostInputs *i, DFN3PrepostOutputs *o) {
    (void)user; (void)i; (void)o;
    fill_state(7.0f);
    return -1;
}
static int warm_up_frames_discard_heads_and_zero_state(void) {
    enum { VIA_FINISH_NAN_HEADS, VIA_FINISH_NAN_STATE, VIA_BYPASS, VIA_RUN, VIA_RUN_FAILS, N_VIA };
    DFN3PrepostConfig cfg;
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    DFN3ModelIoDescriptor descriptor;
    DFN3Model model = {.user = NULL, .infer = stand_in_garbage, .reset = NULL, .io_descriptor = &descriptor};
    DFN3Prepost *p;
    long long index;
    int valid;
    CHECK(config(&cfg, 1) == 0);
    CHECK(dfn3_model_io_descriptor_default(&G, &descriptor) == 0);
    p = dfn3_prepost_create(&cfg);
    CHECK(p != NULL);
    for (int via = 0; via < N_VIA; ++via) {
        unsigned int gen0;
        fill_state(7.0f);
        dfn3_prepost_reset(p);
        CHECK(state_is(0.0f));
        gen0 = dfn3_prepost_state_generation(p);
        for (int t = 0; t < DFN3_MASK_LOOKAHEAD; ++t) {
            spectrum(t);
            CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);       /* a warm-up frame is a frame */
            if (via == VIA_RUN || via == VIA_RUN_FAILS) {
                model.infer = via == VIA_RUN ? stand_in_garbage : stand_in_garbage_then_fail;
                CHECK(dfn3_model_run_frame(&model, p) == (via == VIA_RUN ? 1 : 0));
            } else {
                CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
                CHECK(in.gru_hidden == gru_buf && in.df_convp_window == convp_buf);
                CHECK(isnan(out.erb_mask[0]) && isnan(out.coefs[out.coefs_elements - 1u]));
                fill_state(via == VIA_FINISH_NAN_STATE ? (float)NAN : 7.0f);
                if (via == VIA_BYPASS) CHECK(dfn3_prepost_frame_bypass(p) == 0);
                else CHECK(dfn3_prepost_frame_finish(p) == 0);          /* the NaN heads are not read */
            }
            CHECK(state_is(0.0f));
            CHECK(dfn3_prepost_state_generation(p) == gen0 + (unsigned int)t + 1u);   /* exactly one */
            CHECK(dfn3_prepost_frame_finish(p) == -1 && dfn3_prepost_frame_bypass(p) == -1);   /* closed */
            CHECK(dfn3_prepost_state_generation(p) == gen0 + (unsigned int)t + 1u);
            CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0 && valid == 0);
            for (int k = 0; k < G.n_bins; ++k) CHECK(out_re[k] == 0.0f && out_im[k] == 0.0f);
            CHECK(dfn3_prepost_output_frame_index(p, &index) == -1);
        }
        /* The first real frame reads its heads: a NaN head is refused, and the state it starts from is zero. */
        spectrum(DFN3_MASK_LOOKAHEAD);
        CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
        CHECK(state_is(0.0f) && isnan(out.erb_mask[0]));
        CHECK(dfn3_prepost_frame_finish(p) == -1);
        CHECK(dfn3_prepost_state_generation(p) == gen0 + WARM + 1u);
        CHECK(dfn3_prepost_frame_bypass(p) == 0);
        CHECK(dfn3_prepost_state_generation(p) == gen0 + WARM + 1u);
    }
    /* A reset restarts the warm-up, also inside a warm-up frame, which it drops. */
    if (DFN3_MASK_LOOKAHEAD > 0) {
        unsigned int gen;
        dfn3_prepost_reset(p);
        spectrum(0);
        CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
        gen = dfn3_prepost_state_generation(p);
        dfn3_prepost_reset(p);
        CHECK(dfn3_prepost_state_generation(p) == gen + 1u);
        CHECK(dfn3_prepost_frame_finish(p) == -1 && dfn3_prepost_frame_bypass(p) == -1);
        for (int t = 0; t < DFN3_MASK_LOOKAHEAD; ++t) {
            spectrum(t);
            CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
            CHECK(dfn3_prepost_frame_bypass(p) == 0);
            CHECK(dfn3_prepost_state_generation(p) == gen + 2u + (unsigned int)t);
        }
    }
    dfn3_prepost_destroy(p);
    return 0;
}

/* A stand-in accelerator with memory: its registered state advances on every run and its heads follow the state
 * and the newest window row. Run 0 runs it on the warm-up frames too (writing garbage heads), run 1 only closes them
 * by frame_bypass without running it. Both give the same windows, state at every real frame and output spectra,
 * bit for bit, so the real frames do not depend on whether the accelerator ran on the warm-up frames. */
static uint32_t chain_digest;
static void chain_hash(const float *values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, values + i, sizeof bits);
        chain_digest = (chain_digest ^ bits) * 16777619u;
    }
}
static void stand_in_step(const DFN3PrepostInputs *in, DFN3PrepostOutputs *o, int garbage) {
    const float *newest = in->erb_window + 2u * (size_t)G.n_erb;
    gru_buf[0] += 1.0f;
    convp_buf[0] += 0.5f;
    for (size_t b = 0; b < o->erb_mask_elements; ++b)
        o->erb_mask[b] = 0.5f + 0.4f * sinf(gru_buf[0] * 0.7f + (float)b * 0.1f + newest[b]);
    memset(o->coefs, 0, o->coefs_elements * sizeof(float));
    for (int k = 0; k < G.df_bins; ++k)
        o->coefs[((size_t)k * (size_t)G.df_order + (size_t)HISTORY) * 2u] =
            1.0f + 0.01f * convp_buf[0] * (float)(k % 3);
    if (garbage) { o->erb_mask[0] = (float)NAN; o->coefs[0] = 1e30f; }
}
static int stand_in_stream(DFN3Prepost *p, int run_warm_up, int garbage, float start_state) {
    DFN3PrepostInputs in;
    DFN3PrepostOutputs out;
    int valid;
    chain_digest = 2166136261u;
    fill_state(0.0f);
    dfn3_prepost_reset(p);
    for (int t = 0; t < DELAY + 8; ++t) {
        const int warm = t < DFN3_MASK_LOOKAHEAD;
        spectrum(t);
        CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
        CHECK(dfn3_prepost_frame_open(p, &in, &out) == 0);
        if (warm && !run_warm_up) {
            CHECK(dfn3_prepost_frame_bypass(p) == 0);
        } else {
            if (!warm) {
                if (t == DFN3_MASK_LOOKAHEAD && start_state != 0.0f) gru_buf[0] = start_state;
                chain_hash(in.erb_window, in.erb_window_elements);
                chain_hash(in.spec_window, in.spec_window_elements);
                chain_hash(gru_buf, 1u);
                chain_hash(convp_buf, 1u);
            }
            stand_in_step(&in, &out, warm && garbage);
            CHECK(dfn3_prepost_frame_finish(p) == 0);
        }
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == (t >= DELAY));
        chain_hash(out_re, (size_t)G.n_bins);
        chain_hash(out_im, (size_t)G.n_bins);
    }
    return 0;
}
static int warm_up_runs_do_not_change_the_real_frames(void) {
    DFN3PrepostConfig cfg;
    DFN3Prepost *p;
    uint32_t reference;
    CHECK(config(&cfg, 1) == 0);
    p = dfn3_prepost_create(&cfg);
    CHECK(p != NULL);
    CHECK(stand_in_stream(p, 0, 0, 0.0f) == 0);
    reference = chain_digest;
    CHECK(stand_in_stream(p, 1, 0, 0.0f) == 0 && chain_digest == reference);
    CHECK(stand_in_stream(p, 1, 1, 0.0f) == 0 && chain_digest == reference);
    /* Not vacuous: a nonzero state at the first real frame changes the digest. */
    CHECK(stand_in_stream(p, 1, 0, 3.0f) == 0 && chain_digest != reference);
    dfn3_prepost_destroy(p);
    return 0;
}

static int dual_input(void) {
    DFN3PrepostConfig cfg;
    DFN3Prepost *p;
    float est_re[N_BINS], est_im[N_BINS];
    int valid;
    CHECK(config(&cfg, 1) == 0);
    p = dfn3_prepost_create(&cfg); CHECK(p != NULL);
    memset(est_re, 0, sizeof(est_re)); memset(est_im, 0, sizeof(est_im));
    for (int t = 0; t < 12; ++t) {
        spectrum(t);
        CHECK(dfn3_prepost_pre_process_freq_dual(p, est_re, est_im, re, im) == 1);
        CHECK(dfn3_prepost_frame_bypass(p) == 0);
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == (t >= DELAY));
        if (valid) {
            spectrum(t - DELAY);
            for (int k = 0; k < G.n_bins; ++k) {
                CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
            }
        }
    }
    dfn3_prepost_destroy(p);
    return 0;
}

/* With identity heads the class between a caller-owned AcStft and AcIstft (default sqrt-Hann, analysis
 * spec_scale grid.stft_scale, synthesis ispec_scale grid.stft_unscale) delays the audio by one framing hop
 * plus DELAY frames and nothing else; a second pass after a reset repeats it. */
static int framed_identity(void) {
    enum { INPUT_HOPS = 18 };
    const int hop = G.hop, delay_hops = 1 + DELAY, total_hops = INPUT_HOPS + delay_hops;
    DFN3PrepostConfig cfg;
    DFN3Prepost *p;
    AcStftConfig sc;
    AcStft *stft;
    AcIstft *istft;
    size_t stft_bytes, istft_bytes;
    void *stft_mem, *istft_mem;
    float *audio = (float *)calloc((size_t)total_hops * (size_t)hop, sizeof(float));
    float *output = (float *)calloc((size_t)hop, sizeof(float));
    float peak = 0;
    int valid;
    CHECK(audio != NULL && output != NULL);
    ac_stft_config_default(&sc, G.n_fft);
    sc.spec_scale = G.stft_scale;
    sc.ispec_scale = G.stft_unscale;
    stft_bytes = ac_stft_get_mem_size(&sc);
    istft_bytes = ac_istft_get_mem_size(&sc);
    CHECK(stft_bytes != 0 && istft_bytes != 0);
    stft_mem = aligned_alloc(16u, (stft_bytes + 15u) / 16u * 16u);
    istft_mem = aligned_alloc(16u, (istft_bytes + 15u) / 16u * 16u);
    CHECK(stft_mem != NULL && istft_mem != NULL);
    stft = ac_stft_init(stft_mem, stft_bytes, &sc);
    istft = ac_istft_init(istft_mem, istft_bytes, &sc);
    CHECK(stft != NULL && istft != NULL);
    for (int t = 0; t < INPUT_HOPS; ++t)
        for (int n = 0; n < hop; ++n)
            audio[(size_t)t * (size_t)hop + (size_t)n] =
                0.2f * sinf((float)(t * hop + n) * 0.043f);
    CHECK(config(&cfg, 1) == 0);
    p = dfn3_prepost_create(&cfg); CHECK(p != NULL);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            dfn3_prepost_reset(p);
            ac_stft_reset(stft);
            ac_istft_reset(istft);
            memset(output, 0, (size_t)hop * sizeof(float));
        }
        for (int t = 0; t < total_hops; ++t) {
            CHECK(ac_stft_push_planar(stft, audio + (size_t)t * (size_t)hop, re, im) == 1);
            CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
            CHECK(dfn3_prepost_frame_bypass(p) == 0);
            CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
            CHECK(valid == (t >= DELAY));
            CHECK(ac_istft_push_planar(istft, out_re, out_im, output) == hop);   /* every hop, valid or not */
            for (int n = 0; n < hop; ++n) {
                float expected = t < delay_hops ? 0 :
                    audio[(size_t)(t - delay_hops) * (size_t)hop + (size_t)n];
                float err = fabsf(output[n] - expected);
                if (err > peak) peak = err;
            }
        }
    }
    CHECK(peak < 2e-6f);
    dfn3_prepost_destroy(p);
    free(stft_mem); free(istft_mem); free(audio); free(output);
    return 0;
}

/* The pool identity carries the lookahead pair. req_dump writes this build's
 * MemReq; req_check hands a dumped one to _init_ex as the expected identity and
 * exits 0 when the outcome is the wanted one: mode 1 = accepted, 0 = refused,
 * 2 = refused although size and alignment were made equal to this build's, so
 * only the build identity can tell the pools apart. A refusal leaves the pool
 * bytes as they were. */
static int req_dump(const char *path) {
    DFN3PrepostConfig cfg;
    DFN3PrepostMemReq req;
    FILE *file;
    CHECK(config(&cfg, 0) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(&req, sizeof req, 1, file) == 1);
    CHECK(fclose(file) == 0);
    printf("req: mask=%d df=%d hash=%u\n", DFN3_MASK_LOOKAHEAD,
           DFN3_DF_LOOKAHEAD, (unsigned)req.build_flags_hash);
    return 0;
}

static int req_check(const char *path, int mode) {
    DFN3PrepostConfig cfg;
    DFN3PrepostMemReq req, expected;
    DFN3Prepost *p;
    void *pool;
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fread(&expected, sizeof expected, 1, file) == 1);
    CHECK(fclose(file) == 0);
    CHECK(config(&cfg, 0) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    if (mode == 2) {
        expected.bytes = req.bytes;
        expected.alignment = req.alignment;
    }
    pool = aligned_alloc(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    memset(pool, 0xa5, (size_t)req.bytes);
    CHECK(pool_snap(pool, (size_t)req.bytes) == 0);
    p = dfn3_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &expected);
    CHECK((p != NULL) == (mode == 1));
    if (p) dfn3_prepost_destroy(p);
    else CHECK(pool_same(pool, (size_t)req.bytes));
    free(pool);
    return 0;
}

/* A span wide enough to cancel in float (lo + (hi - lo) rounding to 0) keeps both configured ends of the
 * spec normaliser, from init and from reset, whichever end is the larger. */
static int norm_seed_keeps_both_ends(void) {
    const float ends[2][2] = {{1.0f, 1e-8f}, {1e-8f, 1.0f}};
    for (int e = 0; e < 2; ++e) {
        DFN3GridSpec spec = test_spec();
        DFN3Grid wide;
        DFN3State st;
        spec.spec_norm_init_lo = ends[e][0];
        spec.spec_norm_init_hi = ends[e][1];
        CHECK(dfn3_grid_make(&wide, &spec) == DFN3_GRID_OK);
        size_t bytes = dfn3_state_mem_size(&wide);
        void *block = aligned_alloc(16u, bytes);
        CHECK(block != NULL && dfn3_state_init(&st, block, bytes, &wide) == 0);
        for (int pass = 0; pass < 2; ++pass) {
            const float *s = st.spec_norm_state;
            CHECK(s[0] == ends[e][0] && s[wide.df_bins - 1] == ends[e][1]);
            for (int k = 0; k < wide.df_bins; ++k)
                CHECK(s[k] > 0.0f && s[k] >= fminf(ends[e][0], ends[e][1]) &&
                      s[k] <= fmaxf(ends[e][0], ends[e][1]));
            memset(st.spec_norm_state, 0, sizeof(float) * (size_t)wide.df_bins);
            dfn3_state_reset(&st);                     /* the second pass reads the reset seeds */
        }
        free(block);
    }
    return 0;
}

int main(int argc, char **argv) {
    const DFN3GridSpec spec = test_spec();
    CHECK(dfn3_grid_make(&G, &spec) == DFN3_GRID_OK);
    CHECK(G.n_bins == N_BINS && G.output_delay == DELAY && G.df_history == HISTORY);
    matrices();
    if (argc == 3 && strcmp(argv[1], "req_dump") == 0)
        return req_dump(argv[2]);
    if (argc == 4 && strcmp(argv[1], "req_check") == 0)
        return req_check(argv[2], atoi(argv[3]));
    CHECK(argc == 1);
    CHECK(tap_timing() == 0);
    CHECK(every_tap_reads_its_frame() == 0);
    CHECK(heads_stay_with_their_frame() == 0);
    CHECK(model_io_window_alignment() == 0);
    CHECK(lifecycle_and_transactions(1) == 0);
    CHECK(lifecycle_and_transactions(0) == 0);
    CHECK(registration_is_all_or_none() == 0);
    CHECK(board_form_follows_the_documented_generation() == 0);
    CHECK(stale_pool_is_refused_untouched() == 0);
    CHECK(registered_state_is_zeroed_and_scanned_whole() == 0);
    CHECK(registered_scan_stops_at_the_registered_count() == 0);
    CHECK(warm_up_frames_discard_heads_and_zero_state() == 0);
    CHECK(warm_up_runs_do_not_change_the_real_frames() == 0);
    CHECK(dual_input() == 0);
    CHECK(framed_identity() == 0);
    CHECK(norm_seed_keeps_both_ends() == 0);
    printf("DFN3 %s mask=%d df=%d: raw future taps, layout9, transactions, "
           "reset, dual input, identity delay=%d samples PASS build=%08x "
           "digest=%08x\n",
           dfn3_simd_backend(), DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD,
           (1 + DELAY) * G.hop, build_hash, digest);
    return 0;
}
