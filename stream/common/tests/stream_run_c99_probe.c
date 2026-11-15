/* stream_run.h as the only include, C99: the types and constants a stream user needs. Included twice: the guard holds. */
#include "stream_run.h"
#include "stream_run.h"

int stream_run_probe(StreamWork *w, const float *in, float *out, int status)
{
    w->action = STREAM_WORK_SUBMIT;
    w->inputs[STREAM_WORK_MAX_INPUTS - 1].data = in;
    w->outputs[STREAM_WORK_MAX_OUTPUTS - 1].data = out;
    w->input_count = w->output_count = 1;
    w->state_generation = 0u;
    if (status == STREAM_RUN_OK) return 0;
    return status == STREAM_RUN_NOT_STARTED ? 1 : status == STREAM_RUN_FAILED ? 2 : 3;
}
