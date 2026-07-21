/* Feature-state contract test for log_erb_shared_online_cmvn_v1.
 *
 * This independently repeats the scalar normalisation recurrence around the
 * public rnnoise_compute_features() API.  It also runs a stationary, non-flat
 * spectrum long enough for the runtime state to converge and verifies that the
 * ERB spectral envelope remains non-zero -- the regression that per-band EMA
 * subtraction used to violate.
 */

#include "process.h"
#include "rnnoise_tables_gen.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_FRAMES 4096
#define TOL 2e-5f

typedef struct {
    float mean;
    float var;
    float history[3][RNNOISE_N_BANDS];
    int index;
    int count;
} RefState;

static void fill_stationary_spectrum(float *re, float *im) {
    for (int k = 0; k < RNNOISE_N_BINS; ++k) {
        /* Deterministic and intentionally non-flat across frequency. */
        re[k] = 0.0007f * (float)(1 + (k * 17) % 29);
        im[k] = 0.0003f * (float)((k * 11) % 23);
    }
    im[0] = 0.0f;
    im[RNNOISE_N_BINS - 1] = 0.0f;
}

static int close_enough(float got, float want, const char *what, int frame, int band) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(want)));
    if (fabsf(got - want) <= TOL * scale) return 1;
    printf("FAIL: %s frame=%d band=%d got=%.9g want=%.9g\n",
           what, frame, band, (double)got, (double)want);
    return 0;
}

static int ref_step(RefState *st, const float *re, const float *im,
                    float out[3][RNNOISE_N_BANDS]) {
    float erb_db[RNNOISE_N_BANDS];
    float level = 0.0f;
    const float alpha = (float)exp(
        -((double)RNNOISE_HOP_LEN / (double)RNNOISE_SR) /
        (double)RNNOISE_NORM_TAU_SEC);
    const float denom = sqrtf(st->var + RNNOISE_NORM_VAR_FLOOR_DB2);
    const int idx = st->index;

    for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
        float energy = 0.0f;
        for (int k = 0; k < RNNOISE_N_BINS; ++k) {
            float power = re[k] * re[k] + im[k] * im[k];
            energy += power * rnn_erb_fwd[k][b];
        }
        erb_db[b] = 10.0f * log10f(energy + 1e-10f);
        float feat = (erb_db[b] - st->mean) / denom;
        if (feat > RNNOISE_NORM_CLIP) feat = RNNOISE_NORM_CLIP;
        if (feat < -RNNOISE_NORM_CLIP) feat = -RNNOISE_NORM_CLIP;
        st->history[idx][b] = feat;
        level += erb_db[b];
    }
    level /= RNNOISE_N_BANDS;

    {
        float delta = level - st->mean;
        float new_mean = st->mean + (1.0f - alpha) * delta;
        float new_var = alpha * st->var +
            (1.0f - alpha) * delta * (level - new_mean);
        st->mean = new_mean;
        st->var = new_var > 0.0f ? new_var : 0.0f;
    }

    st->index = (idx + 1) % 3;
    if (st->count < 3) ++st->count;
    if (st->count < 3) return 0;

    for (int f = 0; f < 3; ++f) {
        int src = (st->index + f) % 3;
        memcpy(out[f], st->history[src], sizeof(out[f]));
    }
    return 1;
}

int main(void) {
    RNNoiseState actual;
    RefState ref;
    float re[RNNOISE_N_BINS], im[RNNOISE_N_BINS];
    float got[3][RNNOISE_N_BANDS];
    float want[3][RNNOISE_N_BANDS];
    int ok = 1;

    rnnoise_state_init(&actual);
    memset(&ref, 0, sizeof(ref));
    ref.mean = RNNOISE_NORM_MEAN_INIT_DB;
    ref.var = RNNOISE_NORM_VAR_INIT_DB2;
    fill_stationary_spectrum(re, im);

    for (int t = 0; t < TEST_FRAMES; ++t) {
        int got_ready = rnnoise_compute_features(&actual, re, im, got);
        int want_ready = ref_step(&ref, re, im, want);
        if (got_ready != want_ready) {
            printf("FAIL: ready mismatch at frame %d: got=%d want=%d\n",
                   t, got_ready, want_ready);
            ok = 0;
            break;
        }
        if (got_ready) {
            for (int f = 0; f < 3; ++f) {
                for (int b = 0; b < RNNOISE_N_BANDS; ++b) {
                    if (!close_enough(got[f][b], want[f][b],
                                      "feature", t, b)) ok = 0;
                }
            }
        }
        if (!close_enough(actual.norm_mean, ref.mean, "norm_mean", t, -1)) ok = 0;
        if (!close_enough(actual.norm_var, ref.var, "norm_var", t, -1)) ok = 0;
        if (!ok) break;
    }

    if (ok) {
        float min_feat = got[2][0], max_feat = got[2][0];
        for (int b = 1; b < RNNOISE_N_BANDS; ++b) {
            if (got[2][b] < min_feat) min_feat = got[2][b];
            if (got[2][b] > max_feat) max_feat = got[2][b];
        }
        if (!(max_feat - min_feat > 0.25f)) {
            printf("FAIL: stationary ERB envelope collapsed: min=%.9g max=%.9g\n",
                   (double)min_feat, (double)max_feat);
            ok = 0;
        }
    }

    if (ok) {
        printf("PASS: %s recurrence matches independent reference; "
               "stationary spectral envelope remains non-zero after %d frames\n",
               RNNOISE_FEATURE_VERSION, TEST_FRAMES);
    }
    return ok ? 0 : 1;
}
