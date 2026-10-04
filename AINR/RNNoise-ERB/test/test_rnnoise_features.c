/* Independent C reference test for log_erb_dfn_mean_cplx_unit_0_4k_v3. */

#include "process.h"
#include "rnnoise_tables_gen.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_FRAMES 4096
#define TOL 2e-5f

typedef struct {
    float erb_norm[RNNOISE_N_BANDS];
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
    for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
        float pos = (float)b / (float)(RNNOISE_N_BANDS - 1);
        st->erb_norm[b] = RNNOISE_ERB_NORM_INIT_LO_DB +
            pos * (RNNOISE_ERB_NORM_INIT_HI_DB - RNNOISE_ERB_NORM_INIT_LO_DB);
    }
    for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
        float pos = (float)k / (float)(RNNOISE_SPEC_BINS - 1);
        st->spec_norm[k] = RNNOISE_SPEC_NORM_INIT_LO +
            pos * (RNNOISE_SPEC_NORM_INIT_HI - RNNOISE_SPEC_NORM_INIT_LO);
    }
}

static int ref_step(RefState *st, const float *re, const float *im,
                    float out_erb[3][RNNOISE_N_BANDS],
                    float out_spec[3][2][RNNOISE_SPEC_BINS]) {
    const float erb_alpha = RNNOISE_ERB_NORM_ALPHA;
    const float spec_alpha = RNNOISE_SPEC_NORM_ALPHA;
    const int idx = st->index;

    for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
        float energy = 0.0f;
        for (int k = 0; k < RNNOISE_N_BINS; ++k) {
            float power = re[k] * re[k] + im[k] * im[k];
            energy += power * rnn_erb_fwd[k][b];
        }
        float erb_db = 10.0f * log10f(energy + 1e-10f);
        float mean = erb_alpha * st->erb_norm[b] +
            (1.0f - erb_alpha) * erb_db;
        float feat = (erb_db - mean) / RNNOISE_ERB_NORM_SCALE_DB;
        st->erb_norm[b] = mean;
        st->erb_history[idx][b] = feat;
    }

    for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
        float magnitude = sqrtf(re[k] * re[k] + im[k] * im[k]);
        float state = spec_alpha * st->spec_norm[k] +
            (1.0f - spec_alpha) * magnitude;
        float denom = sqrtf(state + RNNOISE_SPEC_NORM_EPS);
        float re_norm = re[k] / denom;
        float im_norm = im[k] / denom;
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
    RNNoiseModelState model_state;
    RNNoiseModelState previous_model_state;
    RefState ref;
    float re[RNNOISE_N_BINS], im[RNNOISE_N_BINS];
    float got_erb[3][RNNOISE_N_BANDS], want_erb[3][RNNOISE_N_BANDS];
    float got_spec[3][2][RNNOISE_SPEC_BINS];
    float want_spec[3][2][RNNOISE_SPEC_BINS];
    int ok = 1;

    rnnoise_state_init(&actual);
    rnnoise_model_state_init(&model_state);
    /* The state struct is the buffer the graph's h*_out outputs write into,
     * so a healthy state validates in place and is left byte-identical. */
    memset(model_state.hidden, 0x3d, sizeof(model_state.hidden));
    previous_model_state = model_state;
    if (rnnoise_model_state_validate(&model_state) != 0 ||
        memcmp(&model_state, &previous_model_state,
               sizeof(model_state)) != 0) {
        printf("FAIL: RNNoise validate changed or rejected a finite state\n");
        ok = 0;
    }
    if (rnnoise_model_state_validate(NULL) != -1) {
        printf("FAIL: RNNoise validate accepted a NULL state\n");
        ok = 0;
    }
    /* A bad element in any layer, first and last position included, must
     * zero the WHOLE state (every other element is nonzero beforehand). */
    {
        static const int bad_layer[] = {0, 1, RNNOISE_MODEL_GRU_COUNT - 1,
                                        RNNOISE_MODEL_GRU_COUNT - 1};
        static const int bad_index[] = {0, 17, 5, RNNOISE_MODEL_GRU_SIZE - 1};
        static const float bad_value[] = {NAN, INFINITY, -INFINITY, NAN};
        RNNoiseModelState zeros;
        memset(&zeros, 0, sizeof(zeros));
        for (int c = 0; c < 4; ++c) {
            memset(model_state.hidden, 0x41, sizeof(model_state.hidden));
            model_state.hidden[bad_layer[c]][bad_index[c]] = bad_value[c];
            if (rnnoise_model_state_validate(&model_state) != -1 ||
                memcmp(&model_state, &zeros, sizeof(model_state)) != 0) {
                printf("FAIL: RNNoise non-finite hidden[%d][%d] did not "
                       "return -1 and zero the state\n",
                       bad_layer[c], bad_index[c]);
                ok = 0;
            }
        }
    }
    /* A finite state must still validate after a refusal: the check is per
     * call, not a latch. */
    memset(model_state.hidden, 0x41, sizeof(model_state.hidden));
    previous_model_state = model_state;
    if (rnnoise_model_state_validate(&model_state) != 0 ||
        memcmp(&model_state, &previous_model_state,
               sizeof(model_state)) != 0) {
        printf("FAIL: RNNoise refused a finite state after a refusal\n");
        ok = 0;
    }
    /* Copy path: the runtime's own output block is finite-checked and
     * copied into the state. */
    {
        enum { INHERIT_FRAMES = 64 };
        static float out[RNNOISE_MODEL_GRU_COUNT][RNNOISE_MODEL_GRU_SIZE];
        const float (*cout)[RNNOISE_MODEL_GRU_SIZE] =
            (const float (*)[RNNOISE_MODEL_GRU_SIZE])out;
        RNNoiseModelState in_place;
        RNNoiseModelState copied;
        RNNoiseModelState before;
        static const float bad_value[] = {NAN, INFINITY, -INFINITY};
        int stream_ok = 1;
        int refuse_ok = 1;

        /* Deterministic pseudo-state: the in-place state A has the frame
         * written straight into the struct, the copy state B goes through
         * private tensors and inherit; they must stay byte-identical. */
        rnnoise_model_state_init(&in_place);
        rnnoise_model_state_init(&copied);
        for (int t = 0; t < INHERIT_FRAMES; ++t) {
            for (int layer = 0; layer < RNNOISE_MODEL_GRU_COUNT; ++layer) {
                for (int index = 0; index < RNNOISE_MODEL_GRU_SIZE; ++index) {
                    float v = sinf(0.37f * (float)(t + 1) +
                                   0.011f * (float)(layer * 131 + index));
                    in_place.hidden[layer][index] = v;
                    out[layer][index] = v;
                }
            }
            stream_ok &= rnnoise_model_state_validate(&in_place) == 0 &&
                         rnnoise_model_state_inherit(&copied, cout) == 0 &&
                         memcmp(&in_place, &copied, sizeof(in_place)) == 0 &&
                         memcmp(copied.hidden, out, sizeof(out)) == 0;
        }
        if (!stream_ok) {
            printf("FAIL: RNNoise inherit stream differs from in-place "
                   "state\n");
            ok = 0;
        }

        /* An aliased pointer means the runtime wrote in place: no-op, even
         * for a value inherit does not check. */
        memset(copied.hidden, 0x41, sizeof(copied.hidden));
        copied.hidden[1][9] = NAN;
        before = copied;
        if (rnnoise_model_state_inherit(
                &copied, (const float (*)[RNNOISE_MODEL_GRU_SIZE])
                         copied.hidden) != 0 ||
            memcmp(&copied, &before, sizeof(copied)) != 0) {
            printf("FAIL: RNNoise inherit of an aliased block was not a "
                   "no-op returning 0\n");
            ok = 0;
        }

        /* A bad element in any layer, first or last position, returns -1
         * and leaves the previous state untouched. */
        for (int layer = 0; layer < RNNOISE_MODEL_GRU_COUNT; ++layer) {
            for (int c = 0; c < 6; ++c) {
                int index = (c & 1) ? RNNOISE_MODEL_GRU_SIZE - 1 : 0;
                memset(out, 0x3d, sizeof(out));
                out[layer][index] = bad_value[c / 2];
                memset(copied.hidden, 0x41, sizeof(copied.hidden));
                before = copied;
                refuse_ok &= rnnoise_model_state_inherit(&copied, cout) == -1 &&
                             memcmp(&copied, &before, sizeof(copied)) == 0;
            }
        }
        if (!refuse_ok) {
            printf("FAIL: RNNoise inherit accepted or partly copied a "
                   "non-finite block\n");
            ok = 0;
        }

        memset(out, 0x3d, sizeof(out));
        if (rnnoise_model_state_inherit(NULL, cout) != -1 ||
            rnnoise_model_state_inherit(&copied, NULL) != -1) {
            printf("FAIL: RNNoise inherit accepted a NULL argument\n");
            ok = 0;
        }
    }
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
        for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
            if (!close_enough(actual.erb_norm_state[b], ref.erb_norm[b],
                              "erb_norm_state", t, b)) ok = 0;
        }
        if (!ok) break;
    }

    if (ok) {
        float erb_abs_max = fabsf(got_erb[2][0]);
        float complex_abs_sum = 0.0f;
        for (int b = 1; b < RNNOISE_N_BANDS; ++b) {
            if (fabsf(got_erb[2][b]) > erb_abs_max) {
                erb_abs_max = fabsf(got_erb[2][b]);
            }
        }
        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < RNNOISE_SPEC_BINS; ++k) {
                complex_abs_sum += fabsf(got_spec[2][c][k]);
            }
        }
        if (!(erb_abs_max < 2e-5f)) {
            printf("FAIL: stationary ERB mean norm did not converge to zero: %.9g\n",
                   (double)erb_abs_max);
            ok = 0;
        }
        if (!(complex_abs_sum / (2.0f * RNNOISE_SPEC_BINS) > 0.01f)) {
            printf("FAIL: stationary complex feature collapsed\n");
            ok = 0;
        }
    }

    if (ok) {
        printf("PASS: %s matches independent reference; stationary ERB mean "
               "converges to zero while complex features remain observable after %d frames\n",
               RNNOISE_FEATURE_VERSION, TEST_FRAMES);
    }
    return ok ? 0 : 1;
}
