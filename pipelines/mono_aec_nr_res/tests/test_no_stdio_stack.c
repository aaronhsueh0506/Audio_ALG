/* tests/test_no_stdio_stack.c -- full-stack no-stdio link gate. A minimal main() runs the
 * delivered stack as the example programs compose it (query -> aec_init + anr_init +
 * ac_istft_init -> one hop of aec_process_context / aec_get_res_context / anr_process_gain / min /
 * apply / comfort noise / ac_istft_push -> reset -> destroy) using no stdio itself; the exit code
 * is the only output.
 * `make audit-no-stdio-stack` links it against the NO_STDIO=1 archives (libaec.a, libanr.a,
 * libaudio_common.a) and `nm`-gates the executable, so a stdio symbol from any archive fails the
 * audit.
 * Exit codes: 0 = every step succeeded; 1..6 identify the failing step. */
#include <stdint.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"           /* aec_cng_fill */
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#define CNG_NR_GAIN_FLOOR 0.31622777f
#define CNG_RNG_SEED 0x9e3779b9u
#define PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))   /* int16^2 -> audio scale, exact */

/* Static pool: sized for the 16 kHz configs this gate runs; the runtime check against the
 * plan keeps this honest if a future config outgrows it. */
static uint8_t g_pool[1u << 20] __attribute__((aligned(16)));

typedef struct Post {
    AnrDenoiser* nr;
    AcIstft* istft;
    float *total, *extra;
    Complex* spec;
    uint32_t rng;
} Post;

/* One hop of the example chain; 0, or -1 when a stage refuses. */
static int hop(Aec* aec, Post* q, const float* mic, const float* ref, float* out) {
    AecResContext c;
    const float* extra = NULL;
    const float* nr;
    aec_process_context(aec, mic, ref);
    aec_get_res_context(aec, &c);
    if (!c.error_spec || !c.res_gain) return -1;
    if (c.r2 && c.usable_linear) {
        sk_scale_f32(q->extra, c.r2, PSD_SCALE_INV, c.n_freqs);
        extra = q->extra;
    }
    if (anr_process_gain(q->nr, c.error_spec, extra, NULL) != 0) return -1;
    if (!(nr = anr_get_gain(q->nr, NULL))) return -1;
    sk_min_f32(q->total, nr, c.res_gain, c.n_freqs);
    sk_capply_gain_f32(q->spec, c.error_spec, q->total, c.n_freqs);
    if (c.comfort_noise)
        aec_cng_fill(&q->rng, q->spec, c.comfort_noise, c.res_gain, nr, CNG_NR_GAIN_FLOOR,
                     c.n_freqs);
    (void)ac_istft_push(q->istft, q->spec, out);
    return 0;
}

int main(void) {
    AecSignalGrid grid;
    AecConfig aec_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t aec_bytes, nr_bytes, synth_bytes, vec, total;
    uint8_t* cur;
    Aec* aec;
    Post q;

    if (!aec_resolve_signal_grid(16000, aec_default_fft_size(16000), &grid)) return 1;
    aec_config_from_level(&aec_cfg, AEC_LEVEL_BALANCED, 16000);
    aec_cfg.fft_size = grid.fft_size;
    aec_cfg.delay_mode = AEC_DELAY_MATCHED;
    aec_cfg.delay_num_filters = DA_NUM_FILTERS;
    aec_cfg.fixed_delay_samples = -1;
    aec_cfg.enable_delay_est = 1;
    aec_cfg.enable_res = 0;
    aec_cfg.return_res_context = 1;
    nr_cfg = anr_config_for_level_grid(16000, grid.fft_size, ANR_LEVEL_BALANCED);
    ac_stft_config_default(&synth_cfg, grid.fft_size);
    aec_bytes = aec_get_mem_size(&aec_cfg);
    nr_bytes = anr_get_mem_size(&nr_cfg);
    synth_bytes = ac_istft_get_mem_size(&synth_cfg);
    vec = ALIGN16((size_t)grid.n_freqs * sizeof(float));
    if (aec_bytes == 0 || nr_bytes == 0 || synth_bytes == 0) return 1;
    total = ALIGN16(aec_bytes) + ALIGN16(nr_bytes) + ALIGN16(synth_bytes) + 2 * vec +
            ALIGN16((size_t)grid.n_freqs * sizeof(Complex));
    if (total > sizeof(g_pool)) return 2;

    cur = g_pool;
    aec = aec_init(cur, aec_bytes, &aec_cfg);
    cur += ALIGN16(aec_bytes);
    q.nr = anr_init(cur, nr_bytes, &nr_cfg);
    cur += ALIGN16(nr_bytes);
    q.istft = ac_istft_init(cur, synth_bytes, &synth_cfg);
    cur += ALIGN16(synth_bytes);
    if (!aec || !q.nr || !q.istft) return 3;
    q.total = (float*)cur;  cur += vec;
    q.extra = (float*)cur;  cur += vec;
    q.spec = (Complex*)cur;
    q.rng = CNG_RNG_SEED;

    /* One hop of silence through the full AEC->NR->RES chain. */
    static float mic[480], ref[480], out[480];
    memset(mic, 0, sizeof mic); memset(ref, 0, sizeof ref);
    if (hop(aec, &q, mic, ref, out) != 0) return 4;

    aec_reset(aec);
    anr_reset(q.nr);
    ac_istft_reset(q.istft);
    q.rng = CNG_RNG_SEED;
    if (hop(aec, &q, mic, ref, out) != 0) return 5;

    anr_destroy(q.nr);
    aec_destroy(aec);
    anr_destroy(q.nr);
    aec_destroy(aec); /* double-destroy must stay safe */
    (void)out;
    return 0;
}
