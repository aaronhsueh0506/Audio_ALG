/* Independent C reference test for log_erb_abs_cplx_0_4k_v2. */

#include "process.h"
#include "rnnoise_tables_gen.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_FRAMES 4096
#define TOL 2e-5f

typedef struct {
    float spec_norm[RNNOISE_SPEC_BINS];
    float erb_history[3][RNNOISE_N_BANDS];
    float spec_history[3][2][RNNOISE_SPEC_BINS];
    int index;
    int count;
} RefState;

static void fill_stationary_spectrum(float *re, float *im) {
    for (int k = 0; k < RNNOISE_N_BINS; ++k) {
        re[k] = 0.0007f * (float)(1 + (k * 17) % 29);
        im[k] = 0.0003f * (float)((k * 11) % 23);
    }
    im[0] = 0.0f;
    im[RNNOISE_N_BINS - 1] = 0.0f;
}

static int close_enough(float got, float want, const char *what,
                        int frame, int index) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(want)));
    if (fabsf(got - want) <= TOL * scale) return 1;
    printf("FAIL: %s frame=%d index=%d got=%.9g want=%.9g\n",
           what, frame, index, (double)got, (double)want);
    return 0;
}

static void ref_init(RefState *st) {
    memset(st, 0, sizeof(*st));
    for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
        float pos = (float)k / (float)(RNNOISE_SPEC_BINS - 1);
        st->spec_norm[k] = RNNOISE_SPEC_NORM_INIT_LO +
            pos * (RNNOISE_SPEC_NORM_INIT_HI - RNNOISE_SPEC_NORM_INIT_LO);
    }
}

static int ref_step(RefState *st, const float *re, const float *im,
                    float out_erb[3][RNNOISE_N_BANDS],
                    float out_spec[3][2][RNNOISE_SPEC_BINS]) {
    const float alpha = (float)exp(
        -((double)RNNOISE_HOP_LEN / (double)RNNOISE_SR) /
        (double)RNNOISE_SPEC_NORM_TAU_SEC);
    const int idx = st->index;

    for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
        float energy = 0.0f;
        for (int k = 0; k < RNNOISE_N_BINS; ++k) {
            float power = re[k] * re[k] + im[k] * im[k];
            energy += power * rnn_erb_fwd[k][b];
        }
        float erb_db = 10.0f * log10f(energy + 1e-10f);
        float feat = (erb_db - RNNOISE_ERB_CENTER_DB) / RNNOISE_ERB_SCALE_DB;
        if (feat > RNNOISE_ERB_CLIP) feat = RNNOISE_ERB_CLIP;
        if (feat < -RNNOISE_ERB_CLIP) feat = -RNNOISE_ERB_CLIP;
        st->erb_history[idx][b] = feat;
    }

    for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
        float magnitude = sqrtf(re[k] * re[k] + im[k] * im[k]);
        float state = alpha * st->spec_norm[k] + (1.0f - alpha) * magnitude;
        float denom = sqrtf(state + RNNOISE_SPEC_NORM_EPS);
        float re_norm = re[k] / denom;
        float im_norm = im[k] / denom;
        if (re_norm > RNNOISE_SPEC_CLIP) re_norm = RNNOISE_SPEC_CLIP;
        if (re_norm < -RNNOISE_SPEC_CLIP) re_norm = -RNNOISE_SPEC_CLIP;
        if (im_norm > RNNOISE_SPEC_CLIP) im_norm = RNNOISE_SPEC_CLIP;
        if (im_norm < -RNNOISE_SPEC_CLIP) im_norm = -RNNOISE_SPEC_CLIP;
        st->spec_norm[k] = state;
        st->spec_history[idx][0][k] = re_norm;
        st->spec_history[idx][1][k] = im_norm;
    }

    st->index = (idx + 1) % 3;
    if (st->count < 3) ++st->count;
    if (st->count < 3) return 0;

    for (int f = 0; f < 3; ++f) {
        int src = (st->index + f) % 3;
        memcpy(out_erb[f], st->erb_history[src], sizeof(out_erb[f]));
        memcpy(out_spec[f], st->spec_history[src], sizeof(out_spec[f]));
    }
    return 1;
}

int main(void) {
    RNNoiseState actual;
    RefState ref;
    float re[RNNOISE_N_BINS], im[RNNOISE_N_BINS];
    float got_erb[3][RNNOISE_N_BANDS], want_erb[3][RNNOISE_N_BANDS];
    float got_spec[3][2][RNNOISE_SPEC_BINS];
    float want_spec[3][2][RNNOISE_SPEC_BINS];
    int ok = 1;

    rnnoise_state_init(&actual);
    ref_init(&ref);
    fill_stationary_spectrum(re, im);

    for (int t = 0; t < TEST_FRAMES; ++t) {
        int got_ready = rnnoise_compute_features(
            &actual, re, im, got_erb, got_spec);
        int want_ready = ref_step(&ref, re, im, want_erb, want_spec);
        if (got_ready != want_ready) {
            printf("FAIL: ready mismatch at frame %d: got=%d want=%d\n",
                   t, got_ready, want_ready);
            ok = 0;
            break;
        }
        if (got_ready) {
            for (int f = 0; f < 3; ++f) {
                for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
                    if (!close_enough(got_erb[f][b], want_erb[f][b],
                                      "erb", t, b)) ok = 0;
                }
                for (int c = 0; c < 2; ++c) {
                    for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
                        if (!close_enough(got_spec[f][c][k], want_spec[f][c][k],
                                          "complex", t, k)) ok = 0;
                    }
                }
            }
        }
        for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
            if (!close_enough(actual.spec_norm_state[k], ref.spec_norm[k],
                              "spec_norm_state", t, k)) ok = 0;
        }
        if (!ok) break;
    }

    if (ok) {
        float min_erb = got_erb[2][0], max_erb = got_erb[2][0];
        float complex_abs_sum = 0.0f;
        for (int b = 1; b < RNNOISE_N_BANDS; ++b) {
            if (got_erb[2][b] < min_erb) min_erb = got_erb[2][b];
            if (got_erb[2][b] > max_erb) max_erb = got_erb[2][b];
        }
        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
                complex_abs_sum += fabsf(got_spec[2][c][k]);
            }
        }
        if (!(max_erb - min_erb > 0.25f)) {
            printf("FAIL: stationary ERB envelope collapsed\n");
            ok = 0;
        }
        if (!(complex_abs_sum / (2.0f * RNNOISE_SPEC_BINS) > 0.01f)) {
            printf("FAIL: stationary complex feature collapsed\n");
            ok = 0;
        }
    }

    if (ok) {
        printf("PASS: %s matches independent reference; stationary ERB and "
               "complex features remain observable after %d frames\n",
               RNNOISE_FEATURE_VERSION, TEST_FRAMES);
    }
    return ok ? 0 : 1;
}
