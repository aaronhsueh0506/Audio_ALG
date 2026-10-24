/* tests/test_no_stdio_stack.c -- full-stack no-stdio link gate. A minimal main() runs the
 * delivered stack as the example programs compose it (query -> aec_init + anr_init +
 * ac_istft_init -> one hop of aec_process_context / se_frame_from_aec / se_anr_stage_gain / min /
 * apply / comfort noise / ac_istft_push -> reset -> destroy) using no stdio itself; the exit code
 * is the only output.
 * `make audit-no-stdio-stack` links it against the NO_STDIO=1 archives (libaec.a, libanr.a,
 * libaudio_common.a) and se_config.o and `nm`-gates the executable, so a stdio symbol from any
 * archive fails the audit.
 * Exit codes: 0 = every step succeeded; 1..6 identify the failing step. */
#include <stdint.h>
#include <string.h>

#include "aec.h"
#include "anr.h"
#include "audio_stft.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_anr_stage.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

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
    SeSpecFrame f;
    const float* nr;
    aec_process_context(aec, mic, ref);
    if (se_frame_from_aec(aec, &f) != 0 || !f.g_res_ctx) return -1;
    if (!(nr = se_anr_stage_gain(q->nr, &f, q->extra, 1))) return -1;
    sk_min_f32(q->total, nr, f.g_res_ctx, f.n_freqs);
    sk_capply_gain_f32(q->spec, f.e, q->total, f.n_freqs);
    if (f.cn) aec_cng_fill(&q->rng, q->spec, f.cn, f.g_res_ctx, nr, SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
    (void)ac_istft_push(q->istft, q->spec, out);
    return 0;
}

int main(void) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    SeGrid grid;
    AecConfig aec_cfg;
    AnrConfig nr_cfg;
    AcStftConfig synth_cfg;
    size_t aec_bytes, nr_bytes, synth_bytes, vec, total;
    uint8_t* cur;
    Aec* aec;
    Post q;

    memset(&req, 0, sizeof req);
    req.sample_rate = 16000;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = DA_NUM_FILTERS;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &grid, &aec_cfg) != 0) return 1;
    nr_cfg = anr_config_for_mode_grid(16000, grid.fft_size, ANR_MODE_BALANCED);
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
    q.rng = SE_CNG_RNG_SEED;

    /* One hop of silence through the full AEC->NR->RES chain. */
    static float mic[480], ref[480], out[480];
    memset(mic, 0, sizeof mic); memset(ref, 0, sizeof ref);
    if (hop(aec, &q, mic, ref, out) != 0) return 4;

    aec_reset(aec);
    anr_reset(q.nr);
    ac_istft_reset(q.istft);
    q.rng = SE_CNG_RNG_SEED;
    if (hop(aec, &q, mic, ref, out) != 0) return 5;

    anr_destroy(q.nr);
    aec_destroy(aec);
    anr_destroy(q.nr);
    aec_destroy(aec); /* double-destroy must stay safe */
    (void)out;
    return 0;
}
