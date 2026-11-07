#define _POSIX_C_SOURCE 200809L
/* The framing contract of process.h on several grids: audiolib's AcStft with spec_scale =
 * grid.stft_scale and AcIstft with its defaults, joined by rnnoise_apply_gains(), reproduce the
 * input one hop later at unity gain. A scale applied twice or not at all breaks that; the chain
 * is also run with each wrong scale to show the check can fail. The ERB matrices are zeros:
 * no feature or expansion call is made. */

#include "rnnoise_test_support.h"

#include <math.h>

#define HOPS 12
#define TOL 1e-5f

static int g_fail;
#define CHECK(name, cond) do { if (!(cond)) { ++g_fail; \
    printf("FAIL [%s] line %d: %s\n", (name), __LINE__, #cond); } } while (0)

typedef struct {
    int sample_rate, n_fft, n_bands, spec_max_hz;
} GridRow;

static const GridRow GRIDS[] = {
    {.sample_rate = 16000, .n_fft = 512,  .n_bands = 22, .spec_max_hz = 4000},
    {.sample_rate = 16000, .n_fft = 256,  .n_bands = 18, .spec_max_hz = 4000},
    {.sample_rate = 48000, .n_fft = 1024, .n_bands = 32, .spec_max_hz = 4000},
    {.sample_rate = 8000,  .n_fft = 128,  .n_bands = 12, .spec_max_hz = 3000},
};

static uint32_t rng_state = 0x2718281du;

static float next_uniform(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)((rng_state >> 8) & 0x00ffffffu) / 8388608.0f - 1.0f);
}

/* Largest |out(t) - in(t-1)| over the hops after the first. */
static float chain_error(const RNNoiseState *st, const AcStftConfig *cfg,
                         const float *x, int hop, int n_bins)
{
    RnnFraming fr;
    float *re = (float *)malloc(sizeof(float) * (size_t)n_bins);
    float *im = (float *)malloc(sizeof(float) * (size_t)n_bins);
    float *ones = (float *)malloc(sizeof(float) * (size_t)n_bins);
    float *out = (float *)malloc(sizeof(float) * (size_t)hop);
    float worst = 0.0f;
    if (!re || !im || !ones || !out) { worst = INFINITY; goto done; }
    for (int k = 0; k < n_bins; ++k) ones[k] = 1.0f;
    if (rnn_framing_open(&fr, cfg) != 0) { worst = INFINITY; goto done; }
    for (int t = 0; t < HOPS; ++t) {
        if (ac_stft_push_planar(fr.stft, x + (size_t)t * (size_t)hop, re, im) != 1 ||
            rnnoise_apply_gains(st, re, im, ones) != 0 ||
            ac_istft_push_planar(fr.istft, re, im, out) != hop) {
            worst = INFINITY;
            break;
        }
        for (int i = 0; t > 0 && i < hop; ++i) {
            float e = fabsf(out[i] - x[(size_t)(t - 1) * (size_t)hop + (size_t)i]);
            if (e > worst) worst = e;
        }
    }
    rnn_framing_close(&fr);
done:
    free(re); free(im); free(ones); free(out);
    return worst;
}

static void check_grid(const GridRow *row)
{
    RNNoiseGridSpec spec = RNN_TEST_SHIPPED;
    RNNoiseGrid grid;
    RnnPool pool;
    AcStftConfig sc, wrong;
    RnnFraming fr;
    float *zeros, *x, *re, *im, *gain, *ref_re, *ref_im;
    const float *w;
    const int failed_before = g_fail;
    int n_bins, hop;

    spec.sample_rate = row->sample_rate;
    spec.n_fft = row->n_fft;
    spec.n_bands = row->n_bands;
    spec.spec_max_hz = row->spec_max_hz;
    CHECK("grid.make", rnnoise_grid_make(&grid, &spec) == RNNOISE_GRID_OK);
    if (g_fail != failed_before) return;
    n_bins = grid.n_bins;
    hop = grid.hop;
    CHECK("grid.hop", hop * 2 == grid.n_fft);
    CHECK("grid.scales", grid.stft_scale == 1.0f / sqrtf((float)grid.n_fft) &&
          grid.stft_unscale == sqrtf((float)grid.n_fft));

    zeros = (float *)calloc((size_t)n_bins * (size_t)grid.n_bands, sizeof(float));
    CHECK("alloc.zeros", zeros != NULL);
    if (!zeros) return;
    memset(&pool, 0, sizeof(pool));
    rnnoise_config_defaults(&pool.cfg, &grid);
    pool.cfg.erb_fwd = zeros;
    pool.cfg.erb_inv = zeros;
    CHECK("pool.open", rnn_pool_open(&pool) == 0);
    if (g_fail != failed_before) { free(zeros); return; }

    x = (float *)malloc(sizeof(float) * (size_t)HOPS * (size_t)hop);
    CHECK("alloc.signal", x != NULL);
    if (!x) { free(zeros); free(pool.pool); return; }
    for (int i = 0; i < HOPS * hop; ++i) x[i] = 0.2f * next_uniform();

    /* the prescribed configuration: unity gain returns the input one hop later */
    rnn_framing_config(&sc, &grid);
    CHECK("config.defaults", sc.center == 0 && sc.ola == AC_OLA_ACCUMULATE &&
          sc.ispec_scale == 1.0f && sc.window == NULL && sc.fft == NULL);
    CHECK("chain.identity", chain_error(pool.st, &sc, x, hop, n_bins) < TOL);

    /* the same chain with each scale wrong must be seen */
    wrong = sc;
    wrong.spec_scale = 1.0f;
    CHECK("chain.no_analysis_scale", chain_error(pool.st, &wrong, x, hop, n_bins) > 0.1f);
    wrong = sc;
    wrong.ispec_scale = grid.stft_unscale;
    CHECK("chain.double_unscale", chain_error(pool.st, &wrong, x, hop, n_bins) > 0.1f);

    /* the default window is the periodic sqrt-Hann (tolerance: audiolib builds it in float) */
    CHECK("window.open", rnn_framing_open(&fr, &sc) == 0);
    w = ac_stft_window(fr.stft);
    CHECK("window.zero", w[0] == 0.0f);
    for (int i = 0; i < grid.n_fft; ++i) {
        double want = sqrt(0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * (double)i / (double)grid.n_fft)));
        CHECK("window.value", fabs((double)w[i] - want) < 1e-5);
    }
    rnn_framing_close(&fr);

    /* apply_gains: spec *= gain * stft_unscale, bit for bit, inside n_bins only */
    re = (float *)malloc(sizeof(float) * (size_t)(n_bins + 4));
    im = (float *)malloc(sizeof(float) * (size_t)(n_bins + 4));
    ref_re = (float *)malloc(sizeof(float) * (size_t)(n_bins + 4));
    ref_im = (float *)malloc(sizeof(float) * (size_t)(n_bins + 4));
    gain = (float *)malloc(sizeof(float) * (size_t)n_bins);
    CHECK("alloc.frames", re && im && ref_re && ref_im && gain);
    if (!(re && im && ref_re && ref_im && gain)) {
        free(re); free(im); free(ref_re); free(ref_im); free(gain);
        free(x); free(zeros); free(pool.pool);
        return;
    }
    for (int k = 0; k < n_bins + 4; ++k) {
        re[k] = ref_re[k] = k < n_bins ? 0.1f * next_uniform() : 123.0f;
        im[k] = ref_im[k] = k < n_bins ? 0.1f * next_uniform() : 456.0f;
    }
    for (int k = 0; k < n_bins; ++k) {
        gain[k] = 0.05f + 0.95f * (0.5f + 0.5f * next_uniform());
        ref_re[k] = ref_re[k] * (gain[k] * grid.stft_unscale);
        ref_im[k] = ref_im[k] * (gain[k] * grid.stft_unscale);
    }
    CHECK("apply.ok", rnnoise_apply_gains(pool.st, re, im, gain) == 0);
    CHECK("apply.values", memcmp(re, ref_re, sizeof(float) * (size_t)(n_bins + 4)) == 0 &&
          memcmp(im, ref_im, sizeof(float) * (size_t)(n_bins + 4)) == 0);   /* includes the tail */
    CHECK("apply.null", rnnoise_apply_gains(NULL, re, im, gain) == -1 &&
          rnnoise_apply_gains(pool.st, NULL, im, gain) == -1 &&
          rnnoise_apply_gains(pool.st, re, NULL, gain) == -1 &&
          rnnoise_apply_gains(pool.st, re, im, NULL) == -1);
    CHECK("apply.null_untouched", memcmp(re, ref_re, sizeof(float) * (size_t)(n_bins + 4)) == 0 &&
          memcmp(im, ref_im, sizeof(float) * (size_t)(n_bins + 4)) == 0);

    free(re); free(im); free(ref_re); free(ref_im); free(gain);
    free(x); free(zeros); free(pool.pool);
}

int main(void)
{
    for (size_t i = 0; i < sizeof(GRIDS) / sizeof(GRIDS[0]); ++i) check_grid(&GRIDS[i]);
    if (g_fail != 0) return 1;
    puts("PASS: framing contract (scaled sqrt-Hann analysis, unity-gain chain delayed by one hop, "
         "default synthesis, gain product) holds on 4 grids");
    return 0;
}
