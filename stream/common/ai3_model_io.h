/* Shared model I/O wiring of the board samples: row expansion, state pairing, buffer sizing, the pre -> graph
 * boundary check and the opened-graph port check. SDK-free, header-only; the per-graph rows are in
 * ai3_model_io_<model>.h. Inputs = signals, then states; outputs = heads, then states; state k pairs input
 * signal_in_count + k with output head_out_count + k. Link stream_bind.c for the port check. */

#ifndef AI3_MODEL_IO_H
#define AI3_MODEL_IO_H

#include <stddef.h>
#include <string.h>

#include "stream_bind.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AI3_MODEL_IO_MAX_SIGNALS 5
#define AI3_MODEL_IO_MAX_HEADS   3
#define AI3_MODEL_IO_MAX_STATES  16

/* Row keys of this table (Ai3ModelIoRow.model), not the library's AINR_/AIAEC_PRODUCT_MODEL_ID values. */
enum {
    AI3_MODEL_IO_DFN2_FULL = 1,
    AI3_MODEL_IO_DFN2_LITE = 2,
    AI3_MODEL_IO_DEEPVQE_S = 3,
    AI3_MODEL_IO_ALIGN_ULCNET = 4
};

/* Distinct from the STREAM_PORT_ERR_* codes and from each library's own codes. */
enum {
    AI3_MODEL_IO_OK = 0,
    AI3_MODEL_IO_ERR_NULL = -301,      /* NULL argument or work tensor without data */
    AI3_MODEL_IO_ERR_NO_ROW = -302,    /* no row for the (model, sample rate, fft, N) of the selected graph */
    AI3_MODEL_IO_ERR_ROW = -303,       /* row malformed, or too large for the caller's tables */
    AI3_MODEL_IO_ERR_COUNT = -304,     /* the work's signal or head count differs from the graph's */
    AI3_MODEL_IO_ERR_SHAPE = -305,     /* a work tensor's shape or element count differs from the graph's */
    AI3_MODEL_IO_ERR_INDEX = -306      /* port or state index out of range */
};

/* One graph as exported: shapes are [b, c, h, w] (lower rank: leading axes 1); a state has the same shape at its
 * input and its output. Unused entries are zero. */
typedef struct Ai3ModelIoRow {
    int          model, sample_rate, fft, n;   /* the selected graph's configuration; n = frames per call */
    int          layout_version;
    int          signal_count, head_count, state_count;
    unsigned int signal[AI3_MODEL_IO_MAX_SIGNALS][4];
    unsigned int head[AI3_MODEL_IO_MAX_HEADS][4];
    unsigned int state[AI3_MODEL_IO_MAX_STATES][4];
} Ai3ModelIoRow;

/* A row expanded to the port lists stream_check_ports() takes. */
typedef struct Ai3ModelIoConfig {
    int          model, sample_rate, fft, n;
    int          layout_version;
    StreamWiring wiring;
} Ai3ModelIoConfig;

/* N = hop_in / (fft / 2); 0 when fft or hop_in is not positive or hop_in is not a multiple of fft / 2. */
static inline int ai3_model_io_frames(int fft, int hop_in)
{
    int model_hop = fft / 2;
    if (fft <= 0 || model_hop <= 0 || hop_in <= 0 || hop_in % model_hop != 0) return 0;
    return hop_in / model_hop;
}

/* Expands a row: signals then states as inputs, heads then states as outputs. OK, or ERR_NULL / ERR_ROW
 * (counts outside the tables, a zero axis); cfg is zeroed on failure. */
static inline int ai3_model_io_expand(const Ai3ModelIoRow *row, Ai3ModelIoConfig *cfg)
{
    int i, k;

    if (!row || !cfg) return AI3_MODEL_IO_ERR_NULL;
    memset(cfg, 0, sizeof *cfg);
    if (row->signal_count < 1 || row->signal_count > AI3_MODEL_IO_MAX_SIGNALS ||
        row->head_count < 1 || row->head_count > AI3_MODEL_IO_MAX_HEADS ||
        row->state_count < 0 || row->state_count > AI3_MODEL_IO_MAX_STATES ||
        row->signal_count + row->state_count > STREAM_MAX_IN_PORTS ||
        row->head_count + row->state_count > STREAM_MAX_OUT_PORTS)
        return AI3_MODEL_IO_ERR_ROW;
    for (i = 0; i < row->signal_count; ++i)
        if (!row->signal[i][0] || !row->signal[i][1] || !row->signal[i][2] || !row->signal[i][3])
            return AI3_MODEL_IO_ERR_ROW;
    for (i = 0; i < row->head_count; ++i)
        if (!row->head[i][0] || !row->head[i][1] || !row->head[i][2] || !row->head[i][3])
            return AI3_MODEL_IO_ERR_ROW;
    for (k = 0; k < row->state_count; ++k)
        if (!row->state[k][0] || !row->state[k][1] || !row->state[k][2] || !row->state[k][3])
            return AI3_MODEL_IO_ERR_ROW;

    cfg->model = row->model; cfg->sample_rate = row->sample_rate; cfg->fft = row->fft; cfg->n = row->n;
    cfg->layout_version = row->layout_version;
    cfg->wiring.signal_in_count = row->signal_count;
    cfg->wiring.head_out_count = row->head_count;
    cfg->wiring.input_count = row->signal_count + row->state_count;
    cfg->wiring.output_count = row->head_count + row->state_count;
    for (i = 0; i < row->signal_count; ++i)
        cfg->wiring.inputs[i] = stream_port_shape(row->signal[i][0], row->signal[i][1], row->signal[i][2],
                                                  row->signal[i][3]);
    for (i = 0; i < row->head_count; ++i)
        cfg->wiring.outputs[i] = stream_port_shape(row->head[i][0], row->head[i][1], row->head[i][2],
                                                   row->head[i][3]);
    for (k = 0; k < row->state_count; ++k) {
        StreamPortShape s = stream_port_shape(row->state[k][0], row->state[k][1], row->state[k][2],
                                              row->state[k][3]);
        cfg->wiring.inputs[row->signal_count + k] = s;
        cfg->wiring.outputs[row->head_count + k] = s;
    }
    return AI3_MODEL_IO_OK;
}

/* OK when the graph's signal and head counts fit the caller's tables (the pre/post work's capacities). */
static inline int ai3_model_io_fits(const Ai3ModelIoConfig *cfg, int max_signals, int max_heads)
{
    const StreamWiring *w;

    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    w = &cfg->wiring;
    if (w->signal_in_count < 1 || w->signal_in_count > max_signals ||
        w->head_out_count < 1 || w->head_out_count > max_heads ||
        w->input_count < w->signal_in_count || w->input_count > STREAM_MAX_IN_PORTS ||
        w->output_count - w->head_out_count != w->input_count - w->signal_in_count ||
        w->output_count > STREAM_MAX_OUT_PORTS)
        return AI3_MODEL_IO_ERR_ROW;
    return AI3_MODEL_IO_OK;
}

/* State pairs of the graph: signal-first inputs, head-first outputs; 0 for a NULL cfg. */
static inline int ai3_model_io_state_count(const Ai3ModelIoConfig *cfg)
{
    return cfg ? cfg->wiring.input_count - cfg->wiring.signal_in_count : 0;
}

/* Input port index of state k, or -1 when k is outside [0, state_count). */
static inline int ai3_model_io_state_in(const Ai3ModelIoConfig *cfg, int k)
{
    return (cfg && k >= 0 && k < ai3_model_io_state_count(cfg)) ? cfg->wiring.signal_in_count + k : -1;
}

/* Output port index of state k, or -1 when k is outside [0, state_count). */
static inline int ai3_model_io_state_out(const Ai3ModelIoConfig *cfg, int k)
{
    return (cfg && k >= 0 && k < ai3_model_io_state_count(cfg)) ? cfg->wiring.head_out_count + k : -1;
}

/* State pair of output port i, or -1 for a head or an index past the outputs. */
static inline int ai3_model_io_out_state(const Ai3ModelIoConfig *cfg, int i)
{
    if (!cfg || i < cfg->wiring.head_out_count || i >= cfg->wiring.output_count) return -1;
    return i - cfg->wiring.head_out_count;
}

/* State pair of input port i, or -1 for a signal or an index past the inputs. */
static inline int ai3_model_io_in_state(const Ai3ModelIoConfig *cfg, int i)
{
    if (!cfg || i < cfg->wiring.signal_in_count || i >= cfg->wiring.input_count) return -1;
    return i - cfg->wiring.signal_in_count;
}

/* Host buffer bytes of input / output port i at elem_bytes per element; 0 for a bad index or on overflow. */
static inline size_t ai3_model_io_input_bytes(const Ai3ModelIoConfig *cfg, int i, size_t elem_bytes)
{
    size_t e;
    if (!cfg || i < 0 || i >= cfg->wiring.input_count || !elem_bytes) return 0;
    e = cfg->wiring.inputs[i].elements;
    return e > (size_t)-1 / elem_bytes ? 0 : e * elem_bytes;
}

/* As input_bytes, for output port i. */
static inline size_t ai3_model_io_output_bytes(const Ai3ModelIoConfig *cfg, int i, size_t elem_bytes)
{
    size_t e;
    if (!cfg || i < 0 || i >= cfg->wiring.output_count || !elem_bytes) return 0;
    e = cfg->wiring.outputs[i].elements;
    return e > (size_t)-1 / elem_bytes ? 0 : e * elem_bytes;
}

/* stream_check_ports() of the opened graph against the config: STREAM_PORT_OK or a STREAM_PORT_ERR_*, counts
 * first; msg (may be NULL) gets a one-line reason. Run once after the graph is opened, before the first run. */
static inline int ai3_model_io_check_ports(const Ai3ModelIoConfig *cfg,
                                           const StreamDeviceDesc in[], int n_in,
                                           const StreamDeviceDesc out[], int n_out,
                                           char *msg, size_t msg_len)
{
    if (!cfg) return STREAM_PORT_ERR_NULL;
    return stream_check_ports(&cfg->wiring, in, n_in, out, n_out, msg, msg_len);
}

/* Counts of one pre-processed block's signal and head tensors against the graph. */
static inline int ai3_model_io_check_counts(const Ai3ModelIoConfig *cfg, int input_count, int output_count)
{
    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    if (input_count != cfg->wiring.signal_in_count || output_count != cfg->wiring.head_out_count)
        return AI3_MODEL_IO_ERR_COUNT;
    return AI3_MODEL_IO_OK;
}

/* One work tensor against a port shape: data present, then element count and [b, c, h, w] equal. */
static inline int ai3_model_io_check_tensor(const StreamPortShape *want, const void *data, size_t elements,
                                            unsigned int b, unsigned int c, unsigned int h, unsigned int w)
{
    if (!data) return AI3_MODEL_IO_ERR_NULL;
    if (elements != want->elements || b != want->b || c != want->c || h != want->h || w != want->w)
        return AI3_MODEL_IO_ERR_SHAPE;
    return AI3_MODEL_IO_OK;
}

/* Signal i of the work against the graph's input i: data present, element count and [b, c, h, w] equal;
 * ERR_INDEX outside the signals (a state port is never a work tensor). */
static inline int ai3_model_io_check_input(const Ai3ModelIoConfig *cfg, int i, const void *data, size_t elements,
                                           unsigned int b, unsigned int c, unsigned int h, unsigned int w)
{
    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    if (i < 0 || i >= cfg->wiring.signal_in_count) return AI3_MODEL_IO_ERR_INDEX;
    return ai3_model_io_check_tensor(&cfg->wiring.inputs[i], data, elements, b, c, h, w);
}

/* As check_input, for head j against output j. */
static inline int ai3_model_io_check_output(const Ai3ModelIoConfig *cfg, int j, const void *data, size_t elements,
                                            unsigned int b, unsigned int c, unsigned int h, unsigned int w)
{
    if (!cfg) return AI3_MODEL_IO_ERR_NULL;
    if (j < 0 || j >= cfg->wiring.head_out_count) return AI3_MODEL_IO_ERR_INDEX;
    return ai3_model_io_check_tensor(&cfg->wiring.outputs[j], data, elements, b, c, h, w);
}

/* rc = AI3_MODEL_IO_OK or the first mismatch of a pointer to a work with input_count / inputs[] / output_count /
 * outputs[] (the stream libraries' work types). Run on every block the stream asks to submit, before any SDK
 * call; a mismatch means no accelerator run and a failed block for the post-processing. */
#define AI3_MODEL_IO_CHECK_WORK(rc, cfg, work)                                                              \
    do {                                                                                                    \
        int ai3_model_io_i_;                                                                                \
        (rc) = ai3_model_io_check_counts((cfg), (work)->input_count, (work)->output_count);                 \
        for (ai3_model_io_i_ = 0; (rc) == AI3_MODEL_IO_OK && ai3_model_io_i_ < (work)->input_count;         \
             ++ai3_model_io_i_)                                                                             \
            (rc) = ai3_model_io_check_input((cfg), ai3_model_io_i_, (work)->inputs[ai3_model_io_i_].data,   \
                       (work)->inputs[ai3_model_io_i_].elements, (work)->inputs[ai3_model_io_i_].b,         \
                       (work)->inputs[ai3_model_io_i_].c, (work)->inputs[ai3_model_io_i_].h,                \
                       (work)->inputs[ai3_model_io_i_].w);                                                  \
        for (ai3_model_io_i_ = 0; (rc) == AI3_MODEL_IO_OK && ai3_model_io_i_ < (work)->output_count;        \
             ++ai3_model_io_i_)                                                                             \
            (rc) = ai3_model_io_check_output((cfg), ai3_model_io_i_, (work)->outputs[ai3_model_io_i_].data, \
                       (work)->outputs[ai3_model_io_i_].elements, (work)->outputs[ai3_model_io_i_].b,       \
                       (work)->outputs[ai3_model_io_i_].c, (work)->outputs[ai3_model_io_i_].h,              \
                       (work)->outputs[ai3_model_io_i_].w);                                                 \
    } while (0)

/* Row lookup shared by the model headers: the row of (model, sample_rate, fft, n) in rows[row_count], expanded
 * into cfg. OK, or ERR_NO_ROW (a configuration without a row is refused, never derived) / ERR_NULL / ERR_ROW;
 * cfg is zeroed on failure. */
static inline int ai3_model_io_find_row(const Ai3ModelIoRow *rows, int row_count, int model, int sample_rate,
                                        int fft, int n, Ai3ModelIoConfig *cfg)
{
    int i;

    if (!rows || !cfg) return AI3_MODEL_IO_ERR_NULL;
    memset(cfg, 0, sizeof *cfg);
    for (i = 0; i < row_count; ++i)
        if (rows[i].model == model && rows[i].sample_rate == sample_rate && rows[i].fft == fft && rows[i].n == n)
            return ai3_model_io_expand(&rows[i], cfg);
    return AI3_MODEL_IO_ERR_NO_ROW;
}

#ifdef __cplusplus
}
#endif

#endif /* AI3_MODEL_IO_H */
