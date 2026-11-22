/* Prints every row of graph_io_dfn2.h, expanded, for AINR/tests/test_dfn2_port_order.py to compare with a fresh
 * export. Built by `make graph-io-dumper`.
 *   row <id> layout <v> counts <signals> <heads> <inputs> <outputs>
 *   file <mask_lookahead> <graph file> <erb file>
 *   in <i> <b> <c> <h> <w> <elements>      inputs in order: signals, then states
 *   out <i> <b> <c> <h> <w> <elements>     outputs in order: heads, then states
 * <id> is dfn2_<full|lite>_<sample_rate>_<fft>_n<N>_l<mask_lookahead>: the 48 kHz N = 1 and N = 2 rows exist for
 * mask_lookahead 1 and 0, so the id carries it. */

#include <stdio.h>

#include "dfn2_model_io.h"
#include "graph_io.h"
#include "graph_io_dfn2.h"

int main(void)
{
    int r, i;

    /* The rows carry the layout numbers of dfn2_model_io.h; a stale table must not reach the Python pin. */
    if (GRAPH_IO_DFN2_LAYOUT_FULL != DFN2_MODEL_IO_LAYOUT_VERSION_FULL) {
        fprintf(stderr, "full layout version: table %d, library %d\n", (int)GRAPH_IO_DFN2_LAYOUT_FULL,
                (int)DFN2_MODEL_IO_LAYOUT_VERSION_FULL);
        return 2;
    }
    if (GRAPH_IO_DFN2_LAYOUT_LITE != DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY) {
        fprintf(stderr, "lite layout version: table %d, library %d\n", (int)GRAPH_IO_DFN2_LAYOUT_LITE,
                (int)DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
        return 2;
    }

    for (r = 0; r < GRAPH_IO_DFN2_ROW_COUNT; ++r) {
        GraphIoConfig c;
        const GraphIoDfn2Row *meta = &GRAPH_IO_DFN2_ROWS[r];
        const StreamWiring *w = &c.wiring;

        if (graph_io_dfn2_row(r, &c) != GRAPH_IO_OK) return 1;
        printf("row dfn2_%s_%d_%d_n%d_l%d layout %d counts %d %d %d %d\n", c.model == GRAPH_IO_DFN2_LITE ? "lite" : "full",
               c.sample_rate, c.fft, c.n, meta->mask_lookahead, c.layout_version, w->signal_in_count, w->head_out_count,
               w->input_count, w->output_count);
        printf("file %d %s %s\n", meta->mask_lookahead, c.file, meta->erb_file);
        for (i = 0; i < w->input_count; ++i)
            printf("in %d %u %u %u %u %zu\n", i, w->inputs[i].b, w->inputs[i].c, w->inputs[i].h, w->inputs[i].w,
                   w->inputs[i].elements);
        for (i = 0; i < w->output_count; ++i)
            printf("out %d %u %u %u %u %zu\n", i, w->outputs[i].b, w->outputs[i].c, w->outputs[i].h, w->outputs[i].w,
                   w->outputs[i].elements);
    }
    return 0;
}
