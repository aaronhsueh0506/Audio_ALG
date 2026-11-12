/* Prints every row of ai3_model_io_dfn2.h, expanded, for AINR/tests/test_dfn2_port_order.py to compare with a fresh
 * export. Built by `make model-io-dumper`.
 *   row <id> layout <v> counts <signals> <heads> <inputs> <outputs>
 *   in <i> <b> <c> <h> <w> <elements>      inputs in order: signals, then states
 *   out <i> <b> <c> <h> <w> <elements>     outputs in order: heads, then states
 * <id> is dfn2_<full|lite>_<sample_rate>_<fft>_n<N>. */

#include <stdio.h>

#include "dfn2_model_io.h"
#include "ai3_model_io.h"
#include "ai3_model_io_dfn2.h"

_Static_assert(AI3_MODEL_IO_DFN2_LAYOUT_FULL == DFN2_MODEL_IO_LAYOUT_VERSION_FULL, "full layout version");
_Static_assert(AI3_MODEL_IO_DFN2_LAYOUT_LITE == DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY, "lite layout version");

int main(void)
{
    int r, i;

    for (r = 0; r < AI3_MODEL_IO_DFN2_ROW_COUNT; ++r) {
        Ai3ModelIoConfig c;
        const StreamWiring *w = &c.wiring;

        if (ai3_model_io_dfn2_row(r, &c) != AI3_MODEL_IO_OK) return 1;
        printf("row dfn2_%s_%d_%d_n%d layout %d counts %d %d %d %d\n", c.model == AI3_MODEL_IO_DFN2_LITE ? "lite" : "full",
               c.sample_rate, c.fft, c.n, c.layout_version, w->signal_in_count, w->head_out_count, w->input_count,
               w->output_count);
        for (i = 0; i < w->input_count; ++i)
            printf("in %d %u %u %u %u %zu\n", i, w->inputs[i].b, w->inputs[i].c, w->inputs[i].h, w->inputs[i].w,
                   w->inputs[i].elements);
        for (i = 0; i < w->output_count; ++i)
            printf("out %d %u %u %u %u %zu\n", i, w->outputs[i].b, w->outputs[i].c, w->outputs[i].h, w->outputs[i].w,
                   w->outputs[i].elements);
    }
    return 0;
}
