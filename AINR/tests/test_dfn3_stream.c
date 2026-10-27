/* DFN3 public streaming/lifecycle regression. No weights or accelerator are
 * required: known heads make exact tap selection, timing and state observable.
 * Every expectation is derived from DFN3_MASK_LOOKAHEAD / DFN3_DF_LOOKAHEAD, so
 * the same source is built and run for each pair the trainer accepts
 * (-DDFN3_MASK_LOOKAHEAD=m -DDFN3_DF_LOOKAHEAD=d; test_dfn3_stream_c.py).
 * DELAY is the frames between a capture and its enhanced output; heads exist
 * from frame MASK on, so with DF > MASK some hops run the model and emit
 * nothing yet. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DeepFilterNet3/dfn3_prepost.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "DFN3 check failed at %d: %s\n", __LINE__, #x); \
    return 1; } } while (0)
#define CLOSE(a,b) CHECK(fabsf((a)-(b)) < 3e-5f)
#define DELAY DFN3_OUTPUT_DELAY

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
static float fwd[DFN3_N_BINS][DFN3_N_ERB];
static float inv[DFN3_N_ERB][DFN3_N_BINS];
static float re[DFN3_N_BINS], im[DFN3_N_BINS];
static float out_re[DFN3_N_BINS], out_im[DFN3_N_BINS];
static float mask[DFN3_N_ERB];
static float coefs[DFN3_DF_BINS][DFN3_DF_ORDER][2];

static void matrices(void) {
    for (int k = 0; k < DFN3_N_BINS; ++k) {
        int b = k * DFN3_N_ERB / DFN3_N_BINS;
        fwd[k][b] = 1.0f;
        inv[b][k] = 1.0f;
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
    for (int k = 0; k < DFN3_N_BINS; ++k) {
        re[k] = ref_re(frame, k);
        im[k] = ref_im(frame, k);
    }
}
static void identity_heads(DFN3PrepostOutputs *o) {
    for (size_t i = 0; i < o->erb_mask_elements; ++i) o->erb_mask[i] = 1.0f;
    memset(o->coefs, 0, o->coefs_elements * sizeof(float));
    for (int k = 0; k < DFN3_DF_BINS; ++k)
        o->coefs[(k * DFN3_DF_ORDER + DFN3_DF_HISTORY) * 2] = 1.0f;
}
static int callback(void *user, const DFN3PrepostInputs *i,
                     DFN3PrepostOutputs *o) {
    (void)user; (void)i;
    identity_heads(o);
    return 0;
}
static int config(DFN3PrepostConfig *cfg, int mode, FftHandle *fft) {
    if (dfn3_prepost_config_defaults(cfg, mode) != 0) return -1;
    cfg->erb_fwd = &fwd[0][0]; cfg->erb_inv = &inv[0][0]; cfg->fft = fft;
    return 0;
}

/* A pure newest tap proves DF is fed the raw spectrum DF frames after its
 * target and that high bins use the mask of target = current-DELAY. This fails
 * a cascade or a duplicated delay. */
static int tap_timing(void) {
    static DFN3State state, snapshot;
    long long index = -99;
    dfn3_state_init(&state, NULL, NULL);
    dfn3_set_erb_matrices(&state, &fwd[0][0], &inv[0][0]);
    memset(coefs, 0, sizeof(coefs));
    for (int k = 0; k < DFN3_DF_BINS; ++k)
        coefs[k][DFN3_DF_ORDER-1][0] = 1.0f;
    for (int b = 0; b < DFN3_N_ERB; ++b) mask[b] = 0.1f;
    spectrum(0);
    {
        float erb[DFN3_N_ERB], spec[2 * DFN3_DF_BINS];
        dfn3_compute_features(&state, re, im, erb, spec);
        for (int k = 0; k < DFN3_N_ERB; ++k) CHECK(isfinite(erb[k]));
        for (int k = 0; k < 2 * DFN3_DF_BINS; ++k) CHECK(isfinite(spec[k]));
        digest_floats(erb, DFN3_N_ERB);
        digest_floats(spec, 2 * DFN3_DF_BINS);
    }
    /* Heads are refused while the first head's window is not complete, and
     * required after. */
    for (int t = 0; t < DFN3_MASK_LOOKAHEAD; ++t) {
        spectrum(t);
        snapshot = state;
        CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                                  0, out_re, out_im, &index) == -1);
        CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
        CHECK(index == -99);
        CHECK(dfn3_compose_stream(&state, re, im, 0, NULL, NULL,
                                  0, out_re, out_im, &index) == 0);
    }
    for (int t = DFN3_MASK_LOOKAHEAD; t < 16 + DELAY; ++t) {
        const int target = t - DELAY;
        spectrum(t);
        mask[0] = (float)NAN;
        snapshot = state;
        CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                                  0, out_re, out_im, &index) == -1);
        CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
        mask[0] = 0.1f;
        if (target < 0) {   /* heads taken, first output frame not complete */
            CHECK(dfn3_compose_stream(&state, re, im, 1, mask,
                                      &coefs[0][0][0], 0, out_re, out_im,
                                      &index) == 0);
            continue;
        }
        CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                                  0, out_re, out_im, &index) == 1);
        CHECK(index == target);
        digest_floats(out_re, DFN3_N_BINS);
        digest_floats(out_im, DFN3_N_BINS);
        for (int k = 0; k < DFN3_DF_BINS; ++k) {
            const int source = target + DFN3_DF_LOOKAHEAD;
            CLOSE(out_re[k], ref_re(source, k));
            CLOSE(out_im[k], ref_im(source, k));
        }
        for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
            CLOSE(out_re[k], ref_re(target, k) * 0.1f);
            CLOSE(out_im[k], ref_im(target, k) * 0.1f);
        }
    }
    snapshot = state;
    re[0] = (float)INFINITY;
    CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                              0, out_re, out_im, NULL) == -1);
    CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
    return 0;
}

/* Each tap in turn is the only nonzero one: the delayed frame's window is
 * [target-HISTORY, ..., target+DF], including the oldest tap, which is the
 * first to be lost if the frame ring is one slot short. Frames before the
 * stream start read as zero. */
static int every_tap_reads_its_frame(void) {
    static DFN3State state;
    long long index = -99;
    for (int b = 0; b < DFN3_N_ERB; ++b) mask[b] = 1.0f;
    for (int tap = 0; tap < DFN3_DF_ORDER; ++tap) {
        dfn3_state_init(&state, NULL, NULL);
        dfn3_set_erb_matrices(&state, &fwd[0][0], &inv[0][0]);
        memset(coefs, 0, sizeof(coefs));
        for (int k = 0; k < DFN3_DF_BINS; ++k) coefs[k][tap][0] = 1.0f;
        for (int t = 0; t < 24; ++t) {
            spectrum(t);
            if (t < DFN3_MASK_LOOKAHEAD) {
                CHECK(dfn3_compose_stream(&state, re, im, 0, NULL, NULL, 0,
                                          out_re, out_im, &index) == 0);
                continue;
            }
            if (t < DELAY) {
                CHECK(dfn3_compose_stream(&state, re, im, 1, mask,
                                          &coefs[0][0][0], 0, out_re, out_im,
                                          &index) == 0);
                continue;
            }
            CHECK(dfn3_compose_stream(&state, re, im, 1, mask,
                                      &coefs[0][0][0], 0, out_re, out_im,
                                      &index) == 1);
            const int target = t - DELAY;
            const int source = target - DFN3_DF_HISTORY + tap;
            CHECK(index == target);
            for (int k = 0; k < DFN3_DF_BINS; ++k) {
                CLOSE(out_re[k], ref_re(source, k));
                CLOSE(out_im[k], ref_im(source, k));
            }
        }
    }
    return 0;
}

/* Heads that differ per frame: the mask level and the chosen tap of frame h
 * must come out at frame h's own output. The heads of frame h arrive at call
 * h+MASK; when DF > MASK they wait in the ring until call h+DF. */
static float level_of(int frame) { return 0.1f + 0.15f * (float)(frame % 5); }
static int tap_of(int frame) { return (frame * 3 + 1) % DFN3_DF_ORDER; }

static int heads_stay_with_their_frame(void) {
    static DFN3State state;
    long long index = -99;
    dfn3_state_init(&state, NULL, NULL);
    dfn3_set_erb_matrices(&state, &fwd[0][0], &inv[0][0]);
    for (int t = 0; t < 30; ++t) {
        const int head = t - DFN3_MASK_LOOKAHEAD;
        const int target = t - DELAY;
        int emitted;
        spectrum(t);
        if (head < 0) {
            CHECK(dfn3_compose_stream(&state, re, im, 0, NULL, NULL, 0,
                                      out_re, out_im, &index) == 0);
            continue;
        }
        for (int b = 0; b < DFN3_N_ERB; ++b) mask[b] = level_of(head);
        memset(coefs, 0, sizeof(coefs));
        for (int k = 0; k < DFN3_DF_BINS; ++k) coefs[k][tap_of(head)][0] = 1.0f;
        emitted = dfn3_compose_stream(&state, re, im, 1, mask,
                                      &coefs[0][0][0], 0, out_re, out_im,
                                      &index);
        CHECK(emitted == (target >= 0));
        if (target < 0) continue;
        CHECK(index == target);
        for (int k = 0; k < DFN3_DF_BINS; ++k) {
            const int source = target - DFN3_DF_HISTORY + tap_of(target);
            CLOSE(out_re[k], ref_re(source, k));
            CLOSE(out_im[k], ref_im(source, k));
        }
        for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
            CLOSE(out_re[k], ref_re(target, k) * level_of(target));
            CLOSE(out_im[k], ref_im(target, k) * level_of(target));
        }
    }
    return 0;
}

/* The graph window of heads[t] is [t-(2-MASK), ..., t+MASK] with zeros before
 * the stream start; the first MASK pushes do not complete a window. */
static int model_io_window_alignment(void) {
    static DFN3ModelIOState io;
    float erb[DFN3_N_ERB], spec[2][DFN3_DF_BINS];
    dfn3_model_io_reset(&io);
    for (int n = 0; n < 8; ++n) {
        for (int b = 0; b < DFN3_N_ERB; ++b) erb[b] = (float)(n + 1);
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < DFN3_DF_BINS; ++k)
                spec[c][k] = (float)((n + 1) * (c + 1));
        CHECK(dfn3_model_io_push_features(&io, erb,
                  (const float (*)[DFN3_DF_BINS])spec) ==
              (n >= DFN3_MASK_LOOKAHEAD ? 1 : 0));
        for (int row = 0; row < DFN3_MODEL_INPUT_FRAMES; ++row) {
            const int frame = n - (DFN3_MODEL_INPUT_FRAMES - 1) + row;
            const float want = frame < 0 ? 0.0f : (float)(frame + 1);
            CHECK(io.erb_window[row][0] == want);
            CHECK(io.erb_window[row][DFN3_N_ERB - 1] == want);
            CHECK(io.spec_window[0][row][0] == want);
            CHECK(io.spec_window[1][row][DFN3_DF_BINS - 1] == 2.0f * want);
        }
        /* heads[t] with t = n - MASK has frame t-(2-MASK)+row in row `row`. */
        if (n >= DFN3_MASK_LOOKAHEAD) {
            const int t = n - DFN3_MASK_LOOKAHEAD;
            CHECK(io.erb_window[0][0] ==
                  (t - (2 - DFN3_MASK_LOOKAHEAD) < 0 ? 0.0f :
                   (float)(t - (2 - DFN3_MASK_LOOKAHEAD) + 1)));
            CHECK(io.erb_window[DFN3_MODEL_INPUT_FRAMES - 1][0] ==
                  (float)(t + DFN3_MASK_LOOKAHEAD + 1));
        }
    }
    dfn3_model_io_reset(&io);
    CHECK(io.feature_frames_seen == 0);
    for (int b = 0; b < DFN3_N_ERB; ++b) erb[b] = 1.0f;
    memset(spec, 0, sizeof(spec));
    CHECK(dfn3_model_io_push_features(&io, erb,
              (const float (*)[DFN3_DF_BINS])spec) ==
          (DFN3_MASK_LOOKAHEAD == 0 ? 1 : 0));
    return 0;
}

/* The first MASK hops only fill the window: no accelerator run, no output. The
 * next DELAY - MASK hops (DF > MASK) run the model (taken as the identity here)
 * and still emit nothing. After this, hop DELAY is the first with output. */
static int warm_up(DFN3Prepost *p) {
    int valid;
    for (int t = 0; t < DELAY; ++t) {
        spectrum(t);
        CHECK(dfn3_prepost_pre_process_freq(p, re, im) ==
              (t >= DFN3_MASK_LOOKAHEAD));
        if (t >= DFN3_MASK_LOOKAHEAD)
            CHECK(dfn3_prepost_frame_skip(p) == 0);
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == 0);
    }
    return 0;
}

static int lifecycle_and_transactions(void) {
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
    CHECK(config(&cfg, DFN3_IO_FREQ, NULL) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    CHECK(req.bytes % req.alignment == 0);
    CHECK(req.layout_version == 9);
    build_hash = req.build_flags_hash;
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
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == 0);
    descriptor.layout_version = 8;
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == -1);
    /* A graph exported for another lookahead pair is refused. */
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    CHECK(descriptor.mask_lookahead == DFN3_MASK_LOOKAHEAD &&
          descriptor.df_lookahead == DFN3_DF_LOOKAHEAD);
    descriptor.mask_lookahead ^= 1;
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == -1);
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    descriptor.df_lookahead ^= 1;
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == -1);
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    CHECK(dfn3_prepost_frame_inputs(p, &in, &out) == -1);
    CHECK(warm_up(p) == 0);
    spectrum(DELAY);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == -1);
    CHECK(dfn3_prepost_frame_commit(p) == -1);
    CHECK(dfn3_prepost_set_atten_lim(p, -20) == -1);
    CHECK(dfn3_prepost_frame_inputs(p, &in, &out) == 0);
    CHECK((const void *)in.gru_hidden == (void *)out.gru_hidden_next);
    CHECK((const void *)in.df_convp_window == (void *)out.df_convp_window_next);
    for (size_t i = 0; i < out.coefs_elements; ++i) CHECK(isnan(out.coefs[i]));
    identity_heads(&out);
    out.gru_hidden_next[0][0] = 3.0f;
    out.df_convp_window_next[0][0][0] = 4.0f;
    out.coefs[out.coefs_elements-1] = (float)NAN;
    gen = dfn3_prepost_state_generation(p);
    CHECK(dfn3_prepost_frame_commit(p) == -1);
    CHECK(dfn3_prepost_state_generation(p) == gen + 1u);
    for (size_t i = 0; i < in.gru_hidden_elements; ++i)
        CHECK(((float *)out.gru_hidden_next)[i] == 0.0f);
    for (size_t i = 0; i < in.df_convp_window_elements; ++i)
        CHECK(((float *)out.df_convp_window_next)[i] == 0.0f);
    CHECK(dfn3_prepost_frame_commit(p) == -1);
    CHECK(dfn3_prepost_frame_inputs(p, &in, &out) == 0);
    identity_heads(&out);
    CHECK(dfn3_prepost_frame_commit(p) == 0);
    CHECK(dfn3_prepost_frame_skip(p) == -1);
    CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
    CHECK(valid == 1);
    CHECK(dfn3_prepost_output_frame_index(p, &index) == 0 && index == 0);
    spectrum(0);
    for (int k = 0; k < DFN3_N_BINS; ++k) {
        CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
    }
    /* Reject non-finite input without consuming a frame or poisoning EMA. */
    re[0] = (float)NAN;
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == -1);
    spectrum(DELAY + 1);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 1);
    CHECK(dfn3_prepost_frame_inputs(p, &in, &out) == 0);
    identity_heads(&out);
    out.gru_hidden_next[0][0] = (float)INFINITY;
    CHECK(dfn3_prepost_frame_commit(p) == -1);
    CHECK(dfn3_prepost_frame_skip(p) == 0);
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

static int dual_input(void) {
    DFN3PrepostConfig cfg;
    DFN3Prepost *p;
    float est_re[DFN3_N_BINS], est_im[DFN3_N_BINS];
    int valid;
    CHECK(config(&cfg, DFN3_IO_FREQ, NULL) == 0);
    p = dfn3_prepost_create(&cfg); CHECK(p != NULL);
    memset(est_re, 0, sizeof(est_re)); memset(est_im, 0, sizeof(est_im));
    for (int t = 0; t < 12; ++t) {
        spectrum(t);
        CHECK(dfn3_prepost_pre_process_freq_dual(p, est_re, est_im, re, im)
              == (t >= DFN3_MASK_LOOKAHEAD));
        if (t >= DFN3_MASK_LOOKAHEAD) CHECK(dfn3_prepost_frame_skip(p) == 0);
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == (t >= DELAY));
        if (valid) {
            spectrum(t - DELAY);
            for (int k = 0; k < DFN3_N_BINS; ++k) {
                CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
            }
        }
    }
    dfn3_prepost_destroy(p);
    return 0;
}

static int time_identity(void) {
    enum { INPUT_HOPS = 18,
           DELAY_HOPS = 1 + DELAY,   /* framing + model lookahead */
           TOTAL_HOPS = INPUT_HOPS + DELAY_HOPS };
    DFN3PrepostConfig cfg;
    DFN3Prepost *p;
    FftHandle *fft = fft_create(DFN3_N_FFT);
    float audio[TOTAL_HOPS][DFN3_HOP_LEN], output[DFN3_HOP_LEN];
    float peak = 0;
    int written;
    CHECK(fft != NULL);
    memset(audio, 0, sizeof(audio));
    for (int t = 0; t < INPUT_HOPS; ++t)
        for (int n = 0; n < DFN3_HOP_LEN; ++n)
            audio[t][n] = 0.2f * sinf((float)(t * DFN3_HOP_LEN + n) * 0.043f);
    CHECK(config(&cfg, DFN3_IO_TIME, fft) == 0);
    p = dfn3_prepost_create(&cfg); CHECK(p != NULL);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) dfn3_prepost_reset(p);
        for (int t = 0; t < TOTAL_HOPS; ++t) {
            CHECK(dfn3_prepost_pre_process(p, audio[t]) ==
                  (t >= DFN3_MASK_LOOKAHEAD));
            if (t >= DFN3_MASK_LOOKAHEAD)
                CHECK(dfn3_prepost_frame_skip(p) == 0);
            CHECK(dfn3_prepost_post_process(p, output, &written) == 0);
            CHECK(written == (t >= DELAY ? DFN3_HOP_LEN : 0));
            for (int n = 0; n < DFN3_HOP_LEN; ++n) {
                float expected = t < DELAY_HOPS ? 0 : audio[t-DELAY_HOPS][n];
                float err = fabsf(output[n]-expected);
                if (err > peak) peak = err;
            }
        }
    }
    CHECK(peak < 2e-6f);
    dfn3_prepost_destroy(p);
    fft_destroy(fft);
    return 0;
}

/* The pool identity carries the lookahead pair. req_dump writes this build's
 * MemReq; req_check hands a dumped one to _init_ex as the expected identity and
 * exits 0 when the outcome is the wanted one: mode 1 = accepted, 0 = refused,
 * 2 = refused although size and alignment were made equal to this build's, so
 * only the build identity can tell the pools apart. */
static int req_dump(const char *path) {
    DFN3PrepostConfig cfg;
    DFN3PrepostMemReq req;
    FILE *file;
    CHECK(config(&cfg, DFN3_IO_FREQ, NULL) == 0);
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
    CHECK(config(&cfg, DFN3_IO_FREQ, NULL) == 0);
    CHECK(dfn3_prepost_get_mem_size(&cfg, &req) == 0);
    if (mode == 2) {
        expected.bytes = req.bytes;
        expected.alignment = req.alignment;
    }
    pool = aligned_alloc(req.alignment, (size_t)req.bytes);
    CHECK(pool != NULL);
    p = dfn3_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &expected);
    CHECK((p != NULL) == (mode == 1));
    if (p) dfn3_prepost_destroy(p);
    free(pool);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "req_dump") == 0)
        return req_dump(argv[2]);
    if (argc == 4 && strcmp(argv[1], "req_check") == 0)
        return req_check(argv[2], atoi(argv[3]));
    CHECK(argc == 1);
    matrices();
    CHECK(tap_timing() == 0);
    CHECK(every_tap_reads_its_frame() == 0);
    CHECK(heads_stay_with_their_frame() == 0);
    CHECK(model_io_window_alignment() == 0);
    CHECK(lifecycle_and_transactions() == 0);
    CHECK(dual_input() == 0);
    CHECK(time_identity() == 0);
    printf("DFN3 %s mask=%d df=%d: raw future taps, layout9, transactions, "
           "reset, dual input, identity delay=%d samples PASS build=%08x "
           "digest=%08x\n",
           dfn3_simd_backend(), DFN3_MASK_LOOKAHEAD, DFN3_DF_LOOKAHEAD,
           (1 + DELAY) * DFN3_HOP_LEN, build_hash, digest);
    return 0;
}
