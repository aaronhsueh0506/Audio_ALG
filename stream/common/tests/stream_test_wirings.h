/* Synthetic wirings shaped like the graphs stream/common serves (DFN2 full / lite, DeepVQE-S, Align-ULCNet) and
 * the valid device-descriptor sets derived from them; used by test_stream_bind.c. */

#ifndef STREAM_TEST_WIRINGS_H
#define STREAM_TEST_WIRINGS_H

#include <stdint.h>
#include <string.h>

#include "stream_bind.h"

typedef struct Set {
    StreamDeviceDesc in[STREAM_MAX_IN_PORTS + 1];
    StreamDeviceDesc out[STREAM_MAX_OUT_PORTS + 1];
    int n_in, n_out;
} Set;

static inline void add_in(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    w->inputs[w->input_count++] = stream_port_shape(b, c, h, ww);
}

static inline void add_out(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    w->outputs[w->output_count++] = stream_port_shape(b, c, h, ww);
}

static inline void end_signals(StreamWiring *w) {
    w->signal_in_count = w->input_count;
    w->head_out_count = w->output_count;
}

static inline void add_state(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    add_in(w, b, c, h, ww);
    add_out(w, b, c, h, ww);
}

/* DFN2: erb window, spec window; erb_mask (+ df_coefs, df_alpha); h_gru (+ df_convp_history full only). */
static inline void build_dfn2(StreamWiring *w, int full, unsigned n, unsigned df_bins) {
    memset(w, 0, sizeof(*w));
    add_in(w, 1, 1, n + 2u, 32);
    add_in(w, 1, 2, n + 2u, df_bins);
    add_out(w, 1, 1, n, 32);
    if (full) {
        add_out(w, 1, n, df_bins, 10);
        add_out(w, 1, 1, n, 1);
    }
    end_signals(w);
    add_state(w, 1, full ? 5u : 3u, 1, 256);
    if (full) add_state(w, 1, 64, 5, df_bins);
}

/* DeepVQE-S: mic, far; output; 16 states. */
static inline void build_deepvqe(StreamWiring *w, unsigned n, unsigned bins) {
    unsigned f1 = (bins + 1) / 2, f2 = (f1 + 1) / 2, d = bins > 129 ? 32u : 63u;
    int i;
    memset(w, 0, sizeof(*w));
    add_in(w, 1, 2, n, bins);
    add_in(w, 1, 2, n, bins);
    add_out(w, 1, n, bins, 18);
    end_signals(w);
    for (i = 0; i < 16; ++i) {
        unsigned c = 1u + (unsigned)(i % 4) * 8u, h = 4u + (unsigned)(i % 3), ww = (i % 2) ? f2 : d;
        if (i == 8) { c = 1u; h = 1u; ww = 192u; }
        add_state(w, 1, c, h, ww);
    }
}

/* Align-ULCNet: 4 feature tensors (+ frame_reset last at N > 1); output; 4 states. */
static inline void build_align(StreamWiring *w, unsigned n, unsigned bins) {
    unsigned d = bins > 129 ? 32u : 63u, ta = (bins + 9) / 10;
    memset(w, 0, sizeof(*w));
    add_in(w, 1, 1, n, bins);
    add_in(w, 1, 1, n, bins);
    add_in(w, 1, 1, n, bins);
    add_in(w, 1, 1, n, bins);
    if (n > 1) add_in(w, 1, 1, 1, n);
    add_out(w, 1, 2, n, bins);
    end_signals(w);
    add_state(w, 1, 32, d, ta);
    add_state(w, 1, 32, d, ta);
    add_state(w, 1, 32, 5, d);
    add_state(w, 1, 4, 1, 128);
}

static inline void restride(StreamDeviceDesc *d) {
    d->line_ofs = (uint64_t)d->w * (d->elem_bits / 8u);
    d->channel_ofs = d->line_ofs * d->h;
    d->batch_ofs = d->channel_ofs * d->c;
    d->capacity_bytes = d->batch_ofs * d->b;
}

/* Recurrent-state ports carry zero_point 0; the other ports of mode 2 carry a zero point. */
static inline StreamDeviceDesc contiguous(const StreamPortShape *p, int mode, int is_state) {
    StreamDeviceDesc d;
    memset(&d, 0, sizeof(d));
    d.w = p->w; d.h = p->h; d.c = p->c; d.b = p->b;
    d.is_float = (mode == 0);
    d.elem_bits = mode == 0 ? 32u : mode == 1 ? 16u : 8u;
    d.scale_ratio = 0.5f;
    d.zero_point = (mode == 2 && !is_state) ? 3 : 0;
    restride(&d);
    return d;
}

/* mode 0: float32 everywhere; 1: int16 fixed point; 2: int8 fixed point with a zero point on non-state ports. */
static inline Set valid_set(const StreamWiring *w, int mode) {
    Set s;
    int i;
    memset(&s, 0, sizeof(s));
    s.n_in = w->input_count; s.n_out = w->output_count;
    for (i = 0; i < s.n_in; ++i) s.in[i] = contiguous(&w->inputs[i], mode, i >= w->signal_in_count);
    for (i = 0; i < s.n_out; ++i) s.out[i] = contiguous(&w->outputs[i], mode, i >= w->head_out_count);
    return s;
}

#endif /* STREAM_TEST_WIRINGS_H */
