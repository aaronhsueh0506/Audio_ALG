/* graph_io.h and GRAPH_IO_CHECK_WORK as C++11: a work struct of the same shape as StreamWork (make check-graph-io-cxx). */
#include "graph_io.h"

int probe(const GraphIoRow *r, GraphIoConfig *c) {
    struct W {
        int input_count, output_count;
        struct { const float *data; size_t elements; unsigned w, h, c, b; } inputs[1], outputs[1];
    } wk = {0, 0, {{0, 0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0, 0}}};
    int rc = graph_io_expand(r, c);
    GRAPH_IO_CHECK_WORK(rc, c, &wk);
    return rc + graph_io_frames(256, 128) + graph_io_state_count(c) + (int)graph_io_input_bytes(c, 0, 4);
}
