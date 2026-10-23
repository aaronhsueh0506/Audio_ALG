/* tests/test_no_stdio_stack.c -- full-stack no-stdio link gate. A minimal main() runs the
 * delivered stack as the example programs compose it (query -> aec_init + se_sample_chain_init
 * -> one hop of aec_process_context / se_frame_from_aec / se_sample_chain_form / synthesize ->
 * reset -> destroy) using no stdio itself; the exit code is the only output.
 * `make audit-no-stdio-stack` links it against the NO_STDIO=1 archives (libaec.a, libanr.a,
 * libaudio_common.a) and se_config.o and `nm`-gates the executable, so a stdio symbol from any
 * archive fails the audit.
 * Exit codes: 0 = every step succeeded; 1..6 identify the failing step. */
#include <stdint.h>
#include <string.h>

#include "aec.h"
#include "common/se_config.h"
#include "common/se_stage.h"
#include "samples/se_sample_chain.h"

/* Static pool: sized for the 16 kHz configs this gate runs; the runtime check against the
 * plan keeps this honest if a future config outgrows it. */
static uint8_t g_pool[1u << 20] __attribute__((aligned(16)));

int main(void) {
    static const SeRules rules = { .allow_8k = 1, .spatial_context = SE_CONTEXT_PRESET };
    SeRequest req;
    SeGrid grid;
    AecConfig aec_cfg;
    SeSampleChainConfig chain_cfg;
    SeSpecFrame frame;
    const Complex* spec;
    size_t aec_bytes, chain_bytes;
    Aec* aec;
    SeSampleChain* chain;

    memset(&req, 0, sizeof req);
    req.sample_rate = 16000;
    req.delay_mode = AEC_DELAY_MATCHED;
    req.delay_num_filters = DA_NUM_FILTERS;
    req.fixed_delay_samples = -1;
    req.aec_preset = AEC_PRESET_BALANCED;
    if (se_derive(&req, &rules, &grid, &aec_cfg) != 0) return 1;
    chain_cfg = se_sample_chain_default_config(16000, grid.fft_size);
    chain_cfg.res = SE_SAMPLE_RES_FRAME_GAIN;
    aec_bytes = aec_get_mem_size(&aec_cfg);
    chain_bytes = se_sample_chain_mem_size(&chain_cfg);
    if (aec_bytes == 0 || chain_bytes == 0) return 1;
    if (ALIGN16(aec_bytes) + chain_bytes > sizeof(g_pool)) return 2;

    aec = aec_init(g_pool, aec_bytes, &aec_cfg);
    if (!aec) return 3;
    chain = se_sample_chain_init(g_pool + ALIGN16(aec_bytes), chain_bytes, &chain_cfg);
    if (!chain) return 3;

    /* One hop of silence through the full AEC->NR->RES chain. */
    static float mic[480], ref[480], out[480];
    memset(mic, 0, sizeof mic); memset(ref, 0, sizeof ref);
    aec_process_context(aec, mic, ref);
    if (se_frame_from_aec(aec, &frame) != 0) return 4;
    spec = se_sample_chain_form(chain, &frame);
    if (!spec || se_sample_chain_synthesize(chain, spec, out) != 0) return 4;

    aec_reset(aec);
    se_sample_chain_reset(chain);
    aec_process_context(aec, mic, ref);
    if (se_frame_from_aec(aec, &frame) != 0) return 5;
    spec = se_sample_chain_form(chain, &frame);
    if (!spec || se_sample_chain_synthesize(chain, spec, out) != 0) return 5;

    se_sample_chain_destroy(chain);
    aec_destroy(aec);
    se_sample_chain_destroy(chain);
    aec_destroy(aec); /* double-destroy must stay safe */
    (void)out;
    return 0;
}
