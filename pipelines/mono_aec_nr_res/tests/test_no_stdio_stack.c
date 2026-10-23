/* tests/test_no_stdio_stack.c -- full-stack no-stdio link gate. A minimal main() runs the
 * delivered stack (query -> audio_pipeline_init_ex -> one hop of audio_pipeline_process ->
 * reset -> destroy) using no stdio itself; the exit code is the only output.
 * `make audit-no-stdio-stack` links it against the four NO_STDIO=1 archives
 * (libaudio_pipeline.a, libaec.a, libanr.a, libaudio_common.a) and `nm`-gates the
 * executable, so a stdio symbol from any archive fails the audit.
 * Exit codes: 0 = every step succeeded; 1..6 identify the failing step. */
#include <stdint.h>
#include <string.h>
#include "audio_pipeline.h"

/* Static pool: sized for the 16 kHz configs this gate runs (both backends
 * fit well under 1 MiB); the runtime check against req.bytes keeps this
 * honest if a future config outgrows it. */
static uint8_t g_pool[1u << 20] __attribute__((aligned(16)));

int main(void) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
    AudioPipelineMemReq req;
    if (audio_pipeline_get_mem_requirements(&cfg, &req) != 0) return 1;
    if (req.bytes > sizeof(g_pool)) return 2;

    AudioPipeline* p = audio_pipeline_init_ex(g_pool, (size_t)req.bytes, &cfg, &req);
    if (!p) return 3;

    /* One hop of silence through the full AEC->NR->RES chain. */
    static float mic[480], ref[480], out[480];
    memset(mic, 0, sizeof mic); memset(ref, 0, sizeof ref);
    if (audio_pipeline_process(p, mic, ref, out) != 0) { audio_pipeline_destroy(p); return 4; }

    audio_pipeline_reset(p);
    if (audio_pipeline_process(p, mic, ref, out) != 0) { audio_pipeline_destroy(p); return 5; }

    audio_pipeline_destroy(p);
    audio_pipeline_destroy(p); /* double-destroy must stay safe */
    (void)out;
    return 0;
}
