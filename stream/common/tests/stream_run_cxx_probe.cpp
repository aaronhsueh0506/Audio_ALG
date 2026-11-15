/* stream_run.h as the only include, C++11: compiled and run. A value-initialised work is an empty BYPASS block, and the
 * status constants keep their values under C++. */
#include "stream_run.h"

int main()
{
    StreamWork w = StreamWork();
    StreamTensorIn in = StreamTensorIn();
    StreamTensorOut out = StreamTensorOut();
    const int not_started = STREAM_RUN_NOT_STARTED;

    if (w.action != STREAM_WORK_BYPASS || w.input_count != 0 || w.output_count != 0 || w.state_generation != 0u) return 1;
    if (in.data != 0 || out.data != 0 || in.elements != 0u || out.elements != 0u) return 2;
    w.action = STREAM_WORK_SUBMIT;
    if (w.action != 1 || not_started != 0x4E53 || STREAM_RUN_OK != 0 || STREAM_RUN_FAILED != -1) return 3;
    if (sizeof w.inputs / sizeof w.inputs[0] != 5u || sizeof w.outputs / sizeof w.outputs[0] != 3u) return 4;
    return 0;
}
