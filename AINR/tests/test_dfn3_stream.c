/* DFN3 public streaming/lifecycle regression. No weights or accelerator are
 * required: known heads make exact tap selection, timing and state observable. */
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

static uint32_t digest = 2166136261u;
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
static void spectrum(int frame) {
    for (int k = 0; k < DFN3_N_BINS; ++k) {
        re[k] = 0.1f * (float)(frame + 1) + 0.0001f * (float)k;
        im[k] = -0.03f * (float)(frame + 1) + 0.0002f * (float)k;
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

/* A pure future tap proves DF is fed the raw current frame and that high bins
 * use the mask of target=current-1. This fails a cascade or a duplicated delay. */
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
    snapshot = state;
    CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                              0, out_re, out_im, &index) == -1);
    CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
    CHECK(index == -99);
    CHECK(dfn3_compose_stream(&state, re, im, 0, NULL, NULL,
                              0, out_re, out_im, &index) == 0);
    for (int t = 1; t < 17; ++t) {
        spectrum(t);
        mask[0] = (float)NAN;
        snapshot = state;
        CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                                  0, out_re, out_im, &index) == -1);
        CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
        mask[0] = 0.1f;
        CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                                  0, out_re, out_im, &index) == 1);
        CHECK(index == t - 1);
        digest_floats(out_re, DFN3_N_BINS);
        digest_floats(out_im, DFN3_N_BINS);
        for (int k = 0; k < DFN3_DF_BINS; ++k) {
            CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
        }
        for (int k = DFN3_DF_BINS; k < DFN3_N_BINS; ++k) {
            CLOSE(out_re[k], (0.1f * t + 0.0001f * k) * 0.1f);
            CLOSE(out_im[k], (-0.03f * t + 0.0002f * k) * 0.1f);
        }
    }
    snapshot = state;
    re[0] = (float)INFINITY;
    CHECK(dfn3_compose_stream(&state, re, im, 1, mask, &coefs[0][0][0],
                              0, out_re, out_im, NULL) == -1);
    CHECK(memcmp(&snapshot, &state, sizeof(state)) == 0);
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
    CHECK(dfn3_prepost_model_lookahead_frames(p) == 1);
    CHECK(dfn3_prepost_state_generation(p) == 1);
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == 0);
    descriptor.layout_version = 8;
    CHECK(dfn3_model_io_descriptor_validate(&descriptor) == -1);
    CHECK(dfn3_model_io_descriptor_default(&descriptor) == 0);
    CHECK(dfn3_prepost_frame_inputs(p, &in, &out) == -1);
    spectrum(0);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 0);
    CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
    CHECK(valid == 0);
    spectrum(1);
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
    spectrum(2);
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
    spectrum(0);
    CHECK(dfn3_prepost_pre_process_freq(p, re, im) == 0);
    spectrum(1);
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
              == (t > 0));
        if (t > 0) CHECK(dfn3_prepost_frame_skip(p) == 0);
        CHECK(dfn3_prepost_post_process_freq(p, out_re, out_im, &valid) == 0);
        CHECK(valid == (t > 0));
        if (valid) {
            spectrum(t-1);
            for (int k = 0; k < DFN3_N_BINS; ++k) {
                CLOSE(out_re[k], re[k]); CLOSE(out_im[k], im[k]);
            }
        }
    }
    dfn3_prepost_destroy(p);
    return 0;
}

static int time_identity(void) {
    enum { INPUT_HOPS = 18, TOTAL_HOPS = INPUT_HOPS + 2 };
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
            CHECK(dfn3_prepost_pre_process(p, audio[t]) == (t > 0));
            if (t > 0) CHECK(dfn3_prepost_frame_skip(p) == 0);
            CHECK(dfn3_prepost_post_process(p, output, &written) == 0);
            CHECK(written == (t > 0 ? DFN3_HOP_LEN : 0));
            for (int n = 0; n < DFN3_HOP_LEN; ++n) {
                float expected = t < 2 ? 0 : audio[t-2][n];
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

int main(void) {
    matrices();
    CHECK(tap_timing() == 0);
    CHECK(lifecycle_and_transactions() == 0);
    CHECK(dual_input() == 0);
    CHECK(time_identity() == 0);
    printf("DFN3 %s: raw future taps, layout9, transactions, reset, dual input, "
           "identity delay=1024 samples PASS digest=%08x\n",
           dfn3_simd_backend(), digest);
    return 0;
}
