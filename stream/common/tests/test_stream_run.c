/* stream_run.h alone: constants, struct layout and constness of the run contract, and the work type driven through
 * GRAPH_IO_CHECK_WORK over a synthetic row (no sample tables, no SDK). Each CHECK counts one assertion; the exit
 * status is non-zero on any failure. */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "graph_io.h"
#include "stream_run.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; printf("FAIL (line %d): ", __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } while (0)

static void test_constants(void)
{
    const int not_started = STREAM_RUN_NOT_STARTED;
    static const int plausible[] = { 0, 1, -1, 7, -2, 2, 255, -100, 0x4E52, 0x4E54 };
    size_t i;

    CHECK(STREAM_WORK_BYPASS == 0 && STREAM_WORK_SUBMIT == 1, "action values %d / %d", (int)STREAM_WORK_BYPASS,
          (int)STREAM_WORK_SUBMIT);
    CHECK(STREAM_RUN_OK == 0 && STREAM_RUN_FAILED == -1, "status values %d / %d", (int)STREAM_RUN_OK,
          (int)STREAM_RUN_FAILED);
    CHECK(not_started == 0x4E53 && not_started > 0, "NOT_STARTED is the positive 0x4E53 (got %d)", not_started);
    for (i = 0; i < sizeof plausible / sizeof plausible[0]; ++i)
        CHECK(not_started != plausible[i], "NOT_STARTED collides with %d", plausible[i]);
    CHECK(not_started != GRAPH_IO_ERR_NULL && not_started != GRAPH_IO_ERR_INDEX &&
          STREAM_RUN_FAILED != GRAPH_IO_ERR_NULL, "status values clear of GRAPH_IO_ERR_*");
    CHECK(STREAM_WORK_MAX_INPUTS == 5 && STREAM_WORK_MAX_OUTPUTS == 3, "work capacities %d / %d",
          STREAM_WORK_MAX_INPUTS, STREAM_WORK_MAX_OUTPUTS);
    CHECK(STREAM_WORK_MAX_INPUTS == GRAPH_IO_MAX_SIGNALS && STREAM_WORK_MAX_OUTPUTS == GRAPH_IO_MAX_HEADS,
          "work capacities equal the graph_io signal / head tables");
}

static void test_layout(void)
{
    StreamWork w;

    CHECK(offsetof(StreamTensorIn, data) == 0 && offsetof(StreamTensorOut, data) == 0, "data first");
    CHECK(offsetof(StreamTensorIn, elements) == offsetof(StreamTensorOut, elements), "elements offset differs");
    CHECK(offsetof(StreamTensorIn, w) == offsetof(StreamTensorOut, w) && offsetof(StreamTensorIn, h) == offsetof(StreamTensorOut, h) &&
          offsetof(StreamTensorIn, c) == offsetof(StreamTensorOut, c) && offsetof(StreamTensorIn, b) == offsetof(StreamTensorOut, b),
          "shape offsets differ");
    CHECK(offsetof(StreamTensorIn, elements) > offsetof(StreamTensorIn, data) &&
          offsetof(StreamTensorIn, w) > offsetof(StreamTensorIn, elements) &&
          offsetof(StreamTensorIn, h) == offsetof(StreamTensorIn, w) + sizeof(unsigned int) &&
          offsetof(StreamTensorIn, c) == offsetof(StreamTensorIn, h) + sizeof(unsigned int) &&
          offsetof(StreamTensorIn, b) == offsetof(StreamTensorIn, c) + sizeof(unsigned int),
          "member order is data, elements, w, h, c, b");
    CHECK(sizeof(StreamTensorIn) == sizeof(StreamTensorOut), "tensor sizes %zu / %zu", sizeof(StreamTensorIn),
          sizeof(StreamTensorOut));
    CHECK(offsetof(StreamWork, action) == 0 && offsetof(StreamWork, inputs) > offsetof(StreamWork, action) &&
          offsetof(StreamWork, input_count) >= offsetof(StreamWork, inputs) + STREAM_WORK_MAX_INPUTS * sizeof(StreamTensorIn) &&
          offsetof(StreamWork, outputs) > offsetof(StreamWork, input_count) &&
          offsetof(StreamWork, output_count) >= offsetof(StreamWork, outputs) + STREAM_WORK_MAX_OUTPUTS * sizeof(StreamTensorOut) &&
          offsetof(StreamWork, state_generation) > offsetof(StreamWork, output_count),
          "StreamWork member order");
    CHECK(sizeof(w.inputs) == STREAM_WORK_MAX_INPUTS * sizeof(StreamTensorIn) &&
          sizeof(w.outputs) == STREAM_WORK_MAX_OUTPUTS * sizeof(StreamTensorOut), "array lengths");
    CHECK(sizeof(StreamWork) - (offsetof(StreamWork, state_generation) + sizeof(unsigned int)) < sizeof(void *),
          "StreamWork has a member after state_generation (size %zu)", sizeof(StreamWork));
    printf("stream_run: sizeof StreamTensorIn %zu, StreamWork %zu\n", sizeof(StreamTensorIn), sizeof(StreamWork));

    memset(&w, 0, sizeof w);
    CHECK(w.action == STREAM_WORK_BYPASS && w.input_count == 0 && w.output_count == 0 && w.state_generation == 0u,
          "an all-zero work is an empty BYPASS block");

    /* the input view is read-only, the output view is written by the accelerator: a pointer to each member of exactly
     * this type compiles without a diagnostic only while the member has that type (-Werror in the Makefile) */
    {
        const float **in_data = &w.inputs[0].data;
        float **out_data = &w.outputs[0].data;
        size_t *in_elements = &w.inputs[0].elements, *out_elements = &w.outputs[0].elements;
        unsigned int *shape[] = { &w.inputs[0].w, &w.inputs[0].h, &w.inputs[0].c, &w.inputs[0].b,
                                  &w.outputs[0].w, &w.outputs[0].h, &w.outputs[0].c, &w.outputs[0].b,
                                  &w.state_generation };
        int *counts[] = { &w.input_count, &w.output_count };

        CHECK(in_data == &w.inputs[0].data && out_data == &w.outputs[0].data, "data members");
        CHECK(in_elements == &w.inputs[0].elements && out_elements == &w.outputs[0].elements, "elements members");
        CHECK(shape[0] == &w.inputs[0].w && counts[1] == &w.output_count, "shape, generation and count members");
    }
}

/* Row with distinct axes (a swapped w/h/c/b is visible): 2 signals, 3 heads, 1 state. */
static GraphIoRow row_two_three(void)
{
    GraphIoRow r;
    memset(&r, 0, sizeof r);
    r.file = "dfn2_48k_fft1024_n1.bin"; r.model = GRAPH_IO_DFN2_FULL; r.sample_rate = 48000; r.fft = 1024; r.n = 1; r.layout_version = 1;
    r.signal_count = 2; r.head_count = 3; r.state_count = 1;
    r.signal[0][0] = 1; r.signal[0][1] = 1; r.signal[0][2] = 2; r.signal[0][3] = 5;
    r.signal[1][0] = 1; r.signal[1][1] = 3; r.signal[1][2] = 4; r.signal[1][3] = 7;
    r.head[0][0] = 1; r.head[0][1] = 1; r.head[0][2] = 2; r.head[0][3] = 6;
    r.head[1][0] = 1; r.head[1][1] = 2; r.head[1][2] = 3; r.head[1][3] = 4;
    r.head[2][0] = 1; r.head[2][1] = 1; r.head[2][2] = 1; r.head[2][3] = 9;
    r.state[0][0] = 1; r.state[0][1] = 1; r.state[0][2] = 2; r.state[0][3] = 2;
    return r;
}

/* The table limits: STREAM_WORK_MAX_INPUTS signals, STREAM_WORK_MAX_OUTPUTS heads. */
static GraphIoRow row_widest(void)
{
    GraphIoRow r;
    int i;
    memset(&r, 0, sizeof r);
    r.file = "alignulcnet_16k_fft256_n8.bin"; r.model = GRAPH_IO_ALIGNULCNET; r.sample_rate = 16000; r.fft = 256; r.n = 8; r.layout_version = 1;
    r.signal_count = STREAM_WORK_MAX_INPUTS; r.head_count = STREAM_WORK_MAX_OUTPUTS; r.state_count = 2;
    for (i = 0; i < r.signal_count; ++i) { r.signal[i][0] = 1; r.signal[i][1] = 1; r.signal[i][2] = (unsigned)i + 1; r.signal[i][3] = 3; }
    for (i = 0; i < r.head_count; ++i) { r.head[i][0] = 1; r.head[i][1] = 1; r.head[i][2] = (unsigned)i + 2; r.head[i][3] = 4; }
    for (i = 0; i < r.state_count; ++i) { r.state[i][0] = 1; r.state[i][1] = 1; r.state[i][2] = 1; r.state[i][3] = 8; }
    return r;
}

static float g_in[64], g_out[64];

/* A SUBMIT block shaped like cfg's signals and heads. */
static void fill_work(const GraphIoConfig *cfg, StreamWork *w)
{
    const StreamWiring *wi = &cfg->wiring;
    int i;
    memset(w, 0, sizeof *w);
    w->action = STREAM_WORK_SUBMIT;
    w->input_count = wi->signal_in_count;
    for (i = 0; i < w->input_count; ++i) {
        w->inputs[i].data = g_in; w->inputs[i].elements = wi->inputs[i].elements;
        w->inputs[i].b = wi->inputs[i].b; w->inputs[i].c = wi->inputs[i].c;
        w->inputs[i].h = wi->inputs[i].h; w->inputs[i].w = wi->inputs[i].w;
    }
    w->output_count = wi->head_out_count;
    for (i = 0; i < w->output_count; ++i) {
        w->outputs[i].data = g_out; w->outputs[i].elements = wi->outputs[i].elements;
        w->outputs[i].b = wi->outputs[i].b; w->outputs[i].c = wi->outputs[i].c;
        w->outputs[i].h = wi->outputs[i].h; w->outputs[i].w = wi->outputs[i].w;
    }
}

static int check_work(const GraphIoConfig *cfg, const StreamWork *w)
{
    int rc;
    GRAPH_IO_CHECK_WORK(rc, cfg, w);
    return rc;
}

static void test_graph_io_check_work(void)
{
    GraphIoRow r = row_two_three(), rw = row_widest();
    GraphIoConfig c, cw;
    StreamWork w, v;
    int i;

    CHECK(graph_io_expand(&r, &c) == GRAPH_IO_OK && graph_io_expand(&rw, &cw) == GRAPH_IO_OK, "rows expand");
    CHECK(graph_io_fits(&c, STREAM_WORK_MAX_INPUTS, STREAM_WORK_MAX_OUTPUTS) == GRAPH_IO_OK &&
          graph_io_fits(&cw, STREAM_WORK_MAX_INPUTS, STREAM_WORK_MAX_OUTPUTS) == GRAPH_IO_OK, "rows fit the work arrays");

    fill_work(&c, &w);
    CHECK(check_work(&c, &w) == GRAPH_IO_OK, "a matching work passes");
    w.action = STREAM_WORK_BYPASS; w.state_generation = 77u;
    CHECK(check_work(&c, &w) == GRAPH_IO_OK, "action and generation are not part of the shape check");

    fill_work(&c, &w); w.input_count = 3;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_COUNT, "one input too many");
    fill_work(&c, &w); w.input_count = 1;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_COUNT, "one input short");
    fill_work(&c, &w); w.output_count = 4;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_COUNT, "one output too many");
    fill_work(&c, &w); w.output_count = 2;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_COUNT, "one output short");
    fill_work(&c, &w); w.input_count = 0; w.output_count = 0;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_COUNT, "an empty (BYPASS-shaped) work does not match a graph");

    fill_work(&c, &w); w.inputs[1].data = NULL;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_NULL, "input without data");
    fill_work(&c, &w); w.outputs[2].data = NULL;
    CHECK(check_work(&c, &w) == GRAPH_IO_ERR_NULL, "output without data");

    /* each axis of each tensor wrong alone, element count kept: the axes are compared, not only the product */
    for (i = 0; i < 2; ++i) {
        fill_work(&c, &w); v = w; v.inputs[i].w = w.inputs[i].h; v.inputs[i].h = w.inputs[i].w;
        if (w.inputs[i].w != w.inputs[i].h) CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "input %d: w and h swapped", i);
        v = w; v.inputs[i].c = w.inputs[i].w; v.inputs[i].w = w.inputs[i].c;
        if (w.inputs[i].w != w.inputs[i].c) CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "input %d: w and c swapped", i);
        v = w; v.inputs[i].b = w.inputs[i].w; v.inputs[i].w = w.inputs[i].b;
        if (w.inputs[i].w != w.inputs[i].b) CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "input %d: w and b swapped", i);
        v = w; v.inputs[i].elements += 1;
        CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "input %d: one element more", i);
    }
    for (i = 0; i < 3; ++i) {
        fill_work(&c, &w); v = w; v.outputs[i].elements -= 1;
        CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "output %d: one element fewer", i);
        v = w; v.outputs[i].w = w.outputs[i].h; v.outputs[i].h = w.outputs[i].w;
        if (w.outputs[i].w != w.outputs[i].h) CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "output %d: w and h swapped", i);
        v = w; v.outputs[i].c = w.outputs[i].w; v.outputs[i].w = w.outputs[i].c;
        if (w.outputs[i].w != w.outputs[i].c) CHECK(check_work(&c, &v) == GRAPH_IO_ERR_SHAPE, "output %d: w and c swapped", i);
    }

    /* the widest graph fills every slot of both arrays */
    fill_work(&cw, &w);
    CHECK(w.input_count == STREAM_WORK_MAX_INPUTS && w.output_count == STREAM_WORK_MAX_OUTPUTS, "widest work counts");
    CHECK(check_work(&cw, &w) == GRAPH_IO_OK, "a work with every slot used passes");
    v = w; v.inputs[STREAM_WORK_MAX_INPUTS - 1].h += 1;
    CHECK(check_work(&cw, &v) == GRAPH_IO_ERR_SHAPE, "the last input slot is checked");
    v = w; v.outputs[STREAM_WORK_MAX_OUTPUTS - 1].c += 1;
    CHECK(check_work(&cw, &v) == GRAPH_IO_ERR_SHAPE, "the last output slot is checked");
}

int main(void)
{
    test_constants();
    test_layout();
    test_graph_io_check_work();
    if (g_fail) { printf("test_stream_run: %d of %d checks FAILED\n", g_fail, g_checks); return 1; }
    printf("test_stream_run: PASS (%d checks)\n", g_checks);
    return 0;
}
