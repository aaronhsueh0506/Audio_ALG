#include "stream_bind.h"

#include <math.h>

/* Messages are assembled by hand: this object has no printf-family reference. */
typedef struct StreamMsg {
    char *buf;
    size_t cap, len;
} StreamMsg;

static void msg_init(StreamMsg *m, char *buf, size_t cap)
{
    m->buf = buf; m->cap = buf ? cap : 0u; m->len = 0u;
    if (m->cap) m->buf[0] = '\0';
}

static StreamMsg *msg_s(StreamMsg *m, const char *s)
{
    while (m->cap && *s && m->len + 1u < m->cap) m->buf[m->len++] = *s++;
    if (m->cap) m->buf[m->len] = '\0';
    return m;
}

static StreamMsg *msg_u(StreamMsg *m, unsigned long long v)
{
    char t[24];
    int n = 0;
    do { t[n++] = (char)('0' + (int)(v % 10u)); v /= 10u; } while (v);
    while (n) { char c[2]; c[0] = t[--n]; c[1] = '\0'; msg_s(m, c); }
    return m;
}

/* "<dir> <idx> (<signal|head|state> <n>): ", ports named by position; the first state_start ports
 * of the list are signals (inputs) or heads (outputs). */
static StreamMsg *msg_port(StreamMsg *m, const char *dir, int idx, int state_start, const char *lead)
{
    msg_s(m, dir); msg_s(m, " "); msg_u(m, (unsigned long long)idx);
    if (idx < state_start) {
        msg_s(m, " ("); msg_s(m, lead); msg_s(m, " "); msg_u(m, (unsigned long long)idx);
    } else {
        msg_s(m, " (state "); msg_u(m, (unsigned long long)(idx - state_start));
    }
    msg_s(m, "): ");
    return m;
}

/* a * b, pinned to UINT64_MAX on overflow. */
static uint64_t mul_sat(uint64_t a, uint64_t b)
{
    if (a != 0u && b > UINT64_MAX / a) return UINT64_MAX;
    return a * b;
}

/* ONNX [b,c,h,w] vs the device's (w,h,c,b): the non-1 axes, outermost first, must be equal. */
static int axes_match(const StreamPortShape *p, const StreamDeviceDesc *d)
{
    unsigned int want[4], have[4];
    int nw = 0, nh = 0, i;
    const unsigned int onnx[4] = { p->b, p->c, p->h, p->w };
    const unsigned int dev[4] = { d->b, d->c, d->h, d->w };

    for (i = 0; i < 4; ++i) {
        if (onnx[i] != 1u) want[nw++] = onnx[i];
        if (dev[i] != 1u) have[nh++] = dev[i];
    }
    if (nw != nh) return 0;
    for (i = 0; i < nw; ++i) if (want[i] != have[i]) return 0;
    return 1;
}

typedef struct PortPos {
    const char *dir;      /* "input" or "output" */
    const char *lead;     /* "signal" or "head" */
    int         idx;
    int         state_start;
} PortPos;

static int check_one(const StreamPortShape *p, const StreamDeviceDesc *d, const PortPos *pos,
                     StreamMsg *m)
{
    uint64_t bytes, line, chan, batch, total;

    if (d->w == 0u || d->h == 0u || d->c == 0u || d->b == 0u) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "zero-sized axis");
        return STREAM_PORT_ERR_AXES;
    }
    if (d->is_float ? (d->elem_bits != 32u) : (d->elem_bits != 8u && d->elem_bits != 16u)) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "unsupported element format (");
        msg_u(m, d->elem_bits); msg_s(m, " bits, float "); msg_u(m, (unsigned long long)(d->is_float != 0));
        msg_s(m, ")");
        return STREAM_PORT_ERR_FORMAT;
    }
    if (!d->is_float && (d->scale_ratio == 0.0f || !isfinite(d->scale_ratio))) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "fixed-point port with scale_ratio 0 or non-finite");
        return STREAM_PORT_ERR_SCALE;
    }
    if (!axes_match(p, d)) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "device axes w");
        msg_u(m, d->w); msg_s(m, " h"); msg_u(m, d->h);
        msg_s(m, " c"); msg_u(m, d->c); msg_s(m, " b"); msg_u(m, d->b);
        msg_s(m, " do not match ONNX [b,c,h,w] = [");
        msg_u(m, p->b); msg_s(m, ","); msg_u(m, p->c); msg_s(m, ",");
        msg_u(m, p->h); msg_s(m, ","); msg_u(m, p->w); msg_s(m, "]");
        return STREAM_PORT_ERR_AXES;
    }
    bytes = (uint64_t)(d->elem_bits / 8u);
    line = (uint64_t)d->w * bytes;
    chan = mul_sat(line, d->h);
    batch = mul_sat(chan, d->c);
    total = mul_sat(batch, d->b);
    if (total == UINT64_MAX) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "tensor size overflows 64 bits");
        return STREAM_PORT_ERR_CAPACITY;
    }
    if (d->line_ofs != line || (d->channel_ofs != 0u && d->channel_ofs != chan) ||
        (d->batch_ofs != 0u && d->batch_ofs != batch)) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "padded layout (line_ofs ");
        msg_u(m, d->line_ofs); msg_s(m, " want "); msg_u(m, line);
        msg_s(m, ", channel_ofs "); msg_u(m, d->channel_ofs);
        msg_s(m, ", batch_ofs "); msg_u(m, d->batch_ofs);
        msg_s(m, "): only contiguous tensors are supported");
        return STREAM_PORT_ERR_STRIDE;
    }
    if (d->capacity_bytes < total) {
        msg_s(msg_port(m, pos->dir, pos->idx, pos->state_start, pos->lead), "buffer ");
        msg_u(m, d->capacity_bytes); msg_s(m, " bytes < tensor ");
        msg_u(m, total); msg_s(m, " bytes");
        return STREAM_PORT_ERR_CAPACITY;
    }
    return STREAM_PORT_OK;
}

/* Argument and wiring validity: a bad wiring is a library defect, not a graph defect. */
static int wiring_valid(const StreamWiring *w, StreamMsg *m)
{
    int k, n_state;

    if (!w) {
        msg_s(m, "null wiring");
        return STREAM_PORT_ERR_NULL;
    }
    /* equal, non-negative state counts also bound input_count, output_count and head_out_count below */
    if (w->input_count > STREAM_MAX_IN_PORTS || w->output_count > STREAM_MAX_OUT_PORTS ||
        w->signal_in_count < 0 || w->head_out_count < 0 || w->signal_in_count > w->input_count ||
        w->input_count - w->signal_in_count != w->output_count - w->head_out_count) {
        msg_s(m, "wiring has an invalid port count");
        return STREAM_PORT_ERR_NULL;
    }
    n_state = w->input_count - w->signal_in_count;
    for (k = 0; k < n_state; ++k) {
        const StreamPortShape *a = &w->inputs[w->signal_in_count + k];
        const StreamPortShape *b = &w->outputs[w->head_out_count + k];
        if (a->b != b->b || a->c != b->c || a->h != b->h || a->w != b->w) {
            msg_s(m, "wiring state "); msg_u(m, (unsigned long long)k);
            msg_s(m, " has different input and output shapes");
            return STREAM_PORT_ERR_NULL;
        }
    }
    return STREAM_PORT_OK;
}

int stream_check_ports(const StreamWiring *w,
                       const StreamDeviceDesc in[], int n_in,
                       const StreamDeviceDesc out[], int n_out,
                       char *msg, size_t msg_len)
{
    StreamMsg m;
    int i, k, n_state, rc;

    msg_init(&m, msg, msg_len);
    rc = wiring_valid(w, &m);
    if (rc != STREAM_PORT_OK) return rc;
    if (n_in != w->input_count) {
        msg_s(&m, "graph has "); msg_u(&m, (unsigned long long)(n_in < 0 ? 0 : n_in));
        msg_s(&m, " input port(s), the wiring has "); msg_u(&m, (unsigned long long)w->input_count);
        return STREAM_PORT_ERR_IN_COUNT;
    }
    if (n_out != w->output_count) {
        msg_s(&m, "graph has "); msg_u(&m, (unsigned long long)(n_out < 0 ? 0 : n_out));
        msg_s(&m, " output port(s), the wiring has "); msg_u(&m, (unsigned long long)w->output_count);
        return STREAM_PORT_ERR_OUT_COUNT;
    }
    if ((n_in > 0 && !in) || (n_out > 0 && !out)) {
        msg_s(&m, "null descriptor array");
        return STREAM_PORT_ERR_NULL;
    }

    for (i = 0; i < n_in; ++i) {
        PortPos pos;
        pos.dir = "input"; pos.lead = "signal"; pos.idx = i; pos.state_start = w->signal_in_count;
        rc = check_one(&w->inputs[i], &in[i], &pos, &m);
        if (rc != STREAM_PORT_OK) return rc;
    }
    for (i = 0; i < n_out; ++i) {
        PortPos pos;
        pos.dir = "output"; pos.lead = "head"; pos.idx = i; pos.state_start = w->head_out_count;
        rc = check_one(&w->outputs[i], &out[i], &pos, &m);
        if (rc != STREAM_PORT_OK) return rc;
    }

    n_state = n_in - w->signal_in_count;
    for (k = 0; k < n_state; ++k) {
        const int ii = w->signal_in_count + k, oi = w->head_out_count + k;
        const StreamDeviceDesc *a = &in[ii];
        const StreamDeviceDesc *b = &out[oi];
        const char *field = (const char *)0;
        if (a->is_float != b->is_float || a->elem_bits != b->elem_bits) field = "format";
        else if (!a->is_float && a->scale_ratio != b->scale_ratio) field = "scale_ratio";
        else if (a->w != b->w || a->h != b->h || a->c != b->c || a->b != b->b) field = "axes";
        else if (a->line_ofs != b->line_ofs || a->channel_ofs != b->channel_ofs ||
                 a->batch_ofs != b->batch_ofs) field = "strides";
        if (field) {
            msg_s(&m, "state "); msg_u(&m, (unsigned long long)k); msg_s(&m, " (input ");
            msg_u(&m, (unsigned long long)ii); msg_s(&m, ", output ");
            msg_u(&m, (unsigned long long)oi); msg_s(&m, ") differ in ");
            msg_s(&m, field); msg_s(&m, ": an in-place binding needs identical descriptors");
            return STREAM_PORT_ERR_STATE_PAIR;
        }
        if (!a->is_float && (a->zero_point != 0 || b->zero_point != 0)) {
            const int in_side = a->zero_point != 0;
            msg_s(&m, "fixed-point state "); msg_u(&m, (unsigned long long)k);
            msg_s(&m, in_side ? " (input " : " (output ");
            msg_u(&m, (unsigned long long)(in_side ? ii : oi));
            msg_s(&m, ") has a non-zero zero_point: a zero fill of the device state would not be the quantized zero");
            return STREAM_PORT_ERR_STATE_ZERO_POINT;
        }
    }
    return STREAM_PORT_OK;
}
