/* test_no_stdio_stack.c — full-stack no-stdio link gate.
 *
 * A minimal main() that exercises the WHOLE delivered stack -- query ->
 * audio_pipeline_init_ex -> one hop of audio_pipeline_process -> reset ->
 * destroy -- while itself using NO stdio whatsoever: the exit code is the
 * only output. `make audit-no-stdio-stack` links this against the four
 * NO_STDIO=1 archives (libaudio_pipeline.a + libaec.a + libmmse_lsa.a +
 * libaudio_common.a) and then `nm`-gates the resulting EXECUTABLE, so a
 * stdio symbol smuggled in by ANY archive (not just the pipeline's own TU)
 * fails the audit -- a per-object gate alone can pass while the delivered
 * image still drags in stderr machinery.
 *
 * Exit codes: 0 = every step succeeded; 1..6 identify the failing step so a
 * board bring-up can tell them apart without any logging.
 */
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
