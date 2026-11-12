/* stream_check_ports(): the opened-graph check shared by ainr and aiaec, with synthetic device
 * descriptors (no SDK, no accelerator) over synthetic wirings shaped like the graphs it serves:
 * DFN2 full (2 signals + 2 states / 3 heads + 2 states), DFN2 lite (2 + 1 / 1 + 1), DeepVQE-S
 * (2 + 16 / 1 + 16), Align-ULCNet (4 or 5 + 4 / 1 + 4; frame_reset is the last signal at N > 1).
 * The valid set is derived from the wiring itself (ONNX [b,c,h,w] -> the SDK's w,h,c,b, contiguous)
 * and every defect is one edit of that set; each must be refused with its own code and a message
 * naming the port by position. */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "stream_bind.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

typedef struct Set {
    StreamDeviceDesc in[STREAM_MAX_IN_PORTS + 1];
    StreamDeviceDesc out[STREAM_MAX_OUT_PORTS + 1];
    int n_in, n_out;
} Set;

static void add_in(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    w->inputs[w->input_count++] = stream_port_shape(b, c, h, ww);
}

static void add_out(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    w->outputs[w->output_count++] = stream_port_shape(b, c, h, ww);
}

static void end_signals(StreamWiring *w) {
    w->signal_in_count = w->input_count;
    w->head_out_count = w->output_count;
}

static void add_state(StreamWiring *w, unsigned b, unsigned c, unsigned h, unsigned ww) {
    add_in(w, b, c, h, ww);
    add_out(w, b, c, h, ww);
}

/* DFN2: erb window, spec window; erb_mask (+ df_coefs, df_alpha); h_gru (+ df_convp_history full only). */
static void build_dfn2(StreamWiring *w, int full, unsigned n, unsigned df_bins) {
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
static void build_deepvqe(StreamWiring *w, unsigned n, unsigned bins) {
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
static void build_align(StreamWiring *w, unsigned n, unsigned bins) {
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

static void restride(StreamDeviceDesc *d) {
    d->line_ofs = (uint64_t)d->w * (d->elem_bits / 8u);
    d->channel_ofs = d->line_ofs * d->h;
    d->batch_ofs = d->channel_ofs * d->c;
    d->capacity_bytes = d->batch_ofs * d->b;
}

/* Recurrent-state ports carry zero_point 0; the other ports of mode 2 carry a zero point. */
static StreamDeviceDesc contiguous(const StreamPortShape *p, int mode, int is_state) {
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
static Set valid_set(const StreamWiring *w, int mode) {
    Set s;
    int i;
    memset(&s, 0, sizeof(s));
    s.n_in = w->input_count; s.n_out = w->output_count;
    for (i = 0; i < s.n_in; ++i) s.in[i] = contiguous(&w->inputs[i], mode, i >= w->signal_in_count);
    for (i = 0; i < s.n_out; ++i) s.out[i] = contiguous(&w->outputs[i], mode, i >= w->head_out_count);
    return s;
}

static int run(const StreamWiring *w, const Set *s, char *msg, size_t n) {
    return stream_check_ports(w, s->in, s->n_in, s->out, s->n_out, msg, n);
}

static void expect(const StreamWiring *w, const Set *s, int want, const char *what, const char *needle) {
    char msg[256];
    int rc;
    memset(msg, 0x7f, sizeof(msg));
    msg[sizeof(msg) - 1u] = '\0';
    rc = run(w, s, msg, sizeof(msg));
    CHECK(rc == want, "%s: rc=%d (%s), want %d", what, rc, msg, want);
    if (want != STREAM_PORT_OK)
        CHECK(msg[0] != '\0' && (needle == NULL || strstr(msg, needle) != NULL),
              "%s: message '%s' lacks '%s'", what, msg, needle ? needle : "");
}

/* The position label a message must carry: "input 3 (state 0)", "output 0 (head 0)". */
static const char *label(char *buf, size_t n, int is_input, int idx, int split) {
    if (idx < split) snprintf(buf, n, "%s %d (%s %d)", is_input ? "input" : "output", idx, is_input ? "signal" : "head", idx);
    else snprintf(buf, n, "%s %d (state %d)", is_input ? "input" : "output", idx, idx - split);
    return buf;
}

/* Swaps axes i and j (of b,c,h,w order) when both are > 1 and differ: same element count, contiguous
 * once re-strided, but not the ONNX layout. 0 if not possible. */
static int swap_axes(StreamDeviceDesc *d, int i, int j) {
    unsigned int *ax[4] = { &d->b, &d->c, &d->h, &d->w };
    unsigned int t;
    if (*ax[i] <= 1u || *ax[j] <= 1u || *ax[i] == *ax[j]) return 0;
    t = *ax[i]; *ax[i] = *ax[j]; *ax[j] = t;
    restride(d);
    return 1;
}

/* Moves the first axis of size > 1 to an adjacent slot of size 1 (same memory layout, other slot). */
static int shift_first_axis(StreamDeviceDesc *d) {
    unsigned int *ax[4] = { &d->b, &d->c, &d->h, &d->w };
    int i;
    for (i = 0; i < 4; ++i) {
        if (*ax[i] == 1u) continue;
        if (i + 1 < 4 && *ax[i + 1] == 1u) { *ax[i + 1] = *ax[i]; *ax[i] = 1u; restride(d); return 1; }
        if (i - 1 >= 0 && *ax[i - 1] == 1u) { *ax[i - 1] = *ax[i]; *ax[i] = 1u; restride(d); return 1; }
        return 0;
    }
    return 0;
}

static void run_profile(const StreamWiring *w, const char *name) {
    Set s;
    StreamWiring bad;
    char what[160], msg[256], lb[64], needle[96];
    int i, j, k, mode, swapped = 0, lo, li, n_state;
    const int sig = w->signal_in_count, head = w->head_out_count;

    n_state = w->input_count - sig;
    CHECK(w->input_count <= STREAM_MAX_IN_PORTS && w->output_count <= STREAM_MAX_OUT_PORTS, "%s: port counts", name);
    CHECK(n_state >= 1 && n_state == w->output_count - head, "%s: state counts", name);
    for (i = 0; i < w->input_count; ++i)
        CHECK(w->inputs[i].elements == (size_t)w->inputs[i].b * w->inputs[i].c * w->inputs[i].h * w->inputs[i].w, "%s: input %d elements", name, i);
    lo = w->output_count - 1; li = w->input_count - 1;

    for (mode = 0; mode < 3; ++mode) {
        snprintf(what, sizeof(what), "%s valid set mode %d", name, mode);
        s = valid_set(w, mode);
        expect(w, &s, STREAM_PORT_OK, what, NULL);
    }

    /* the same memory reported with the size-1 axes in other slots is accepted */
    for (i = 0; i < w->output_count; ++i) {
        s = valid_set(w, 0);
        if (shift_first_axis(&s.out[i])) { expect(w, &s, STREAM_PORT_OK, "output with a shifted size-1 axis", NULL); break; }
    }
    s = valid_set(w, 0);
    CHECK(shift_first_axis(&s.in[0]), "%s: signal input 0 has an axis to shift", name);
    expect(w, &s, STREAM_PORT_OK, "input with a shifted size-1 axis", NULL);

    /* counts, refused before any indexing */
    s = valid_set(w, 0); s.n_in -= 1;  expect(w, &s, STREAM_PORT_ERR_IN_COUNT, "missing input", "input port");
    s = valid_set(w, 0); s.n_out -= 1; expect(w, &s, STREAM_PORT_ERR_OUT_COUNT, "missing output", "output port");
    s = valid_set(w, 0); s.in[s.n_in] = s.in[0]; s.n_in += 1;
    expect(w, &s, STREAM_PORT_ERR_IN_COUNT, "extra input", "input port");
    s = valid_set(w, 0); s.out[s.n_out] = s.out[0]; s.n_out += 1;
    expect(w, &s, STREAM_PORT_ERR_OUT_COUNT, "extra output", "output port");
    s = valid_set(w, 0);
    CHECK(stream_check_ports(w, NULL, 0, s.out, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_IN_COUNT, "count before indexing");
    CHECK(stream_check_ports(w, s.in, s.n_in, NULL, 0, msg, sizeof(msg)) == STREAM_PORT_ERR_OUT_COUNT, "output count before indexing");
    CHECK(stream_check_ports(w, s.in, -1, s.out, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_IN_COUNT && strstr(msg, "graph has 0 input port"), "negative device count");
    CHECK(stream_check_ports(w, s.in, s.n_in, s.out, -1, msg, sizeof(msg)) == STREAM_PORT_ERR_OUT_COUNT && strstr(msg, "graph has 0 output port"), "negative device out count");
    CHECK(stream_check_ports(w, NULL, s.n_in, s.out, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_NULL, "NULL input descriptors");
    CHECK(stream_check_ports(w, s.in, s.n_in, NULL, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_NULL, "NULL output descriptors");

    /* equal element count, swapped axes: every port, every pair of axes that can be swapped */
    for (i = 0; i < w->input_count; ++i)
        for (j = 0; j < 4; ++j)
            for (k = j + 1; k < 4; ++k) {
                s = valid_set(w, 0);
                if (!swap_axes(&s.in[i], j, k)) continue;
                expect(w, &s, STREAM_PORT_ERR_AXES, "input axes swapped", label(lb, sizeof(lb), 1, i, sig));
                ++swapped;
            }
    for (i = 0; i < w->output_count; ++i)
        for (j = 0; j < 4; ++j)
            for (k = j + 1; k < 4; ++k) {
                s = valid_set(w, 0);
                if (!swap_axes(&s.out[i], j, k)) continue;
                expect(w, &s, STREAM_PORT_ERR_AXES, "output axes swapped", label(lb, sizeof(lb), 0, i, head));
                ++swapped;
            }
    CHECK(swapped >= w->output_count, "%s: only %d port/axis swaps could be made", name, swapped);
    s = valid_set(w, 0); s.out[lo].w += 1u;
    expect(w, &s, STREAM_PORT_ERR_AXES, "state output dims differ", label(lb, sizeof(lb), 0, lo, head));
    s = valid_set(w, 0); s.in[0].w = 0u;
    expect(w, &s, STREAM_PORT_ERR_AXES, "zero-sized axis", "input 0 (signal 0): zero-sized axis");
    s = valid_set(w, 0); s.out[0].b = 0u;
    expect(w, &s, STREAM_PORT_ERR_AXES, "zero-sized output axis", "output 0 (head 0): zero-sized axis");

    /* padding / stride */
    s = valid_set(w, 0); s.in[0].line_ofs += 4u;
    expect(w, &s, STREAM_PORT_ERR_STRIDE, "padded line_ofs", "padded");
    s = valid_set(w, 0); s.in[1].channel_ofs += 8u;
    expect(w, &s, STREAM_PORT_ERR_STRIDE, "padded channel_ofs", "padded");
    s = valid_set(w, 0); s.in[0].channel_ofs = s.in[0].line_ofs * s.in[0].h + 8u;
    expect(w, &s, STREAM_PORT_ERR_STRIDE, "padded channel_ofs (input 0)", "padded");
    s = valid_set(w, 0); s.out[0].batch_ofs = s.out[0].channel_ofs * s.out[0].c + 64u;
    expect(w, &s, STREAM_PORT_ERR_STRIDE, "padded batch_ofs", "padded");
    s = valid_set(w, 0); s.in[0].line_ofs = 0u;
    expect(w, &s, STREAM_PORT_ERR_STRIDE, "unreported line_ofs", "padded");
    s = valid_set(w, 0); s.in[1].channel_ofs = 0u; s.in[1].batch_ofs = 0u;
    expect(w, &s, STREAM_PORT_OK, "channel/batch offsets unreported", NULL);

    /* capacity */
    s = valid_set(w, 1); s.in[1].capacity_bytes -= 1u;
    expect(w, &s, STREAM_PORT_ERR_CAPACITY, "undersized signal buffer", "buffer");
    s = valid_set(w, 1); s.out[lo].capacity_bytes -= 2u;
    expect(w, &s, STREAM_PORT_ERR_CAPACITY, "undersized state buffer", "buffer");
    s = valid_set(w, 0); s.out[0].capacity_bytes += 1000u;
    expect(w, &s, STREAM_PORT_OK, "oversized buffer", NULL);

    /* state pair disagreement, at every state k */
    for (k = 0; k < n_state; ++k) {
        const int ii = sig + k, oi = head + k;
        snprintf(needle, sizeof(needle), "state %d (input %d, output %d)", k, ii, oi);
        s = valid_set(w, 1); s.out[oi] = contiguous(&w->outputs[oi], 2, 1);
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state in int16, out int8", needle);
        s = valid_set(w, 1); s.out[oi] = contiguous(&w->outputs[oi], 2, 1);
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state in int16, out int8 (format)", "format");
        s = valid_set(w, 1); s.out[oi].scale_ratio = 0.25f;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state scale differs", "scale_ratio");
        s = valid_set(w, 1); s.in[ii].scale_ratio = 0.25f;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state scale differs (input end)", needle);
        s = valid_set(w, 0); s.out[oi] = contiguous(&w->outputs[oi], 1, 1);
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state float in, int16 out", "format");
        s = valid_set(w, 1); s.in[ii] = contiguous(&w->inputs[ii], 0, 1);
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state int16 in, float out", "format");
        s = valid_set(w, 0); s.out[oi].line_ofs = 0u;   /* equal-padding mismatch: the per-port stride rule fires first */
        expect(w, &s, STREAM_PORT_ERR_STRIDE, "state out line_ofs 0", "padded");
        s = valid_set(w, 0);
        if (shift_first_axis(&s.out[oi])) expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state in/out dims differ in slot", "axes");
        else CHECK(0, "%s: state %d has no axis to shift", name, k);
        /* strides differing between pair ends while each is individually valid: unreported vs reported channel/batch */
        s = valid_set(w, 0); s.out[oi].channel_ofs = 0u; s.out[oi].batch_ofs = 0u;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state strides reported on one end only", "strides");
        s = valid_set(w, 0); s.in[ii].channel_ofs = 0u; s.in[ii].batch_ofs = 0u;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state strides reported on the input end only", needle);
        s = valid_set(w, 0); s.out[oi].batch_ofs = 0u;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state batch_ofs reported on one end only", "strides");
        s = valid_set(w, 0); s.in[ii].batch_ofs = 0u;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state batch_ofs reported on the input end only", "strides");
        s = valid_set(w, 0); s.out[oi].channel_ofs = 0u;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "state channel_ofs reported on one end only", "strides");
    }
    if (n_state >= 2) {   /* two broken pairs: the lower k is reported */
        s = valid_set(w, 1);
        s.out[head + 0].scale_ratio = 0.25f;
        s.out[lo].scale_ratio = 0.25f;
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "first broken pair reported", "state 0 (");
    }
    /* a state output only pairs with the state input of the same position */
    if (n_state >= 2) {
        s = valid_set(w, 1);
        s.out[head + 0].scale_ratio = 0.25f; s.out[head + 1].scale_ratio = 0.25f;
        s.in[sig + 0].scale_ratio = 0.25f;   /* pair 0 now agrees; pair 1 does not */
        expect(w, &s, STREAM_PORT_ERR_STATE_PAIR, "pair 1 broken only", "state 1 (");
    }
    CHECK(li == sig + n_state - 1, "%s: last state input", name);

    /* fixed-point state zero_point: 0/0 passes; any non-zero at either end is refused, equal or not */
    for (mode = 1; mode <= 2; ++mode) {
        for (k = 0; k < n_state; ++k) {
            const int o = head + k, in_i = sig + k;
            char m2[256], want_out[48], want_in[48];
            snprintf(want_out, sizeof(want_out), "(output %d)", o);
            snprintf(want_in, sizeof(want_in), "(input %d)", in_i);
            s = valid_set(w, mode);
            expect(w, &s, STREAM_PORT_OK, "state zero_point 0/0", NULL);
            s = valid_set(w, mode); s.in[in_i].zero_point = 5; s.out[o].zero_point = 5;
            expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "state zero_point 5/5", "zero_point");
            s = valid_set(w, mode); s.out[o].zero_point = 5;
            expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "state zero_point 0/5", want_out);
            s = valid_set(w, mode); s.in[in_i].zero_point = 5;
            expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "state zero_point 5/0", want_in);
            s = valid_set(w, mode); s.in[in_i].zero_point = -3; s.out[o].zero_point = -3;
            expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "state zero_point -3/-3", "zero_point");
            s = valid_set(w, mode); s.in[in_i].zero_point = 3; s.out[o].zero_point = 1;
            expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "state zero_point 3/1", "zero_point");
            s = valid_set(w, mode); s.out[o].zero_point = 5;
            (void)run(w, &s, m2, sizeof(m2));
            CHECK(strstr(m2, want_out) != NULL, "%s: zero_point message lacks '%s': %s", name, want_out, m2);
        }
    }
    /* a float state port ignores zero_point; a non-zero zero_point on a signal or head port is not this rule */
    s = valid_set(w, 0); s.in[sig].zero_point = 5; s.out[head].zero_point = 5;
    expect(w, &s, STREAM_PORT_OK, "float state with a zero_point", NULL);
    s = valid_set(w, 1); s.in[0].zero_point = 5;
    expect(w, &s, STREAM_PORT_OK, "signal port with a zero_point", NULL);
    s = valid_set(w, 1); s.out[0].zero_point = -7;
    expect(w, &s, STREAM_PORT_OK, "head port with a zero_point", NULL);
    for (i = 0; i < sig; ++i) {   /* every signal port may carry a zero point; the last signal is not a state */
        s = valid_set(w, 1); s.in[i].zero_point = 5;
        expect(w, &s, STREAM_PORT_OK, "any signal port with a zero_point", NULL);
    }
    for (i = 0; i < head; ++i) {
        s = valid_set(w, 1); s.out[i].zero_point = 5;
        expect(w, &s, STREAM_PORT_OK, "any head port with a zero_point", NULL);
    }

    /* the signal / state split is read from the wiring: moving the first state into the signals and
     * the heads (counts stay equal) makes its zero_point a non-rule */
    bad = *w; bad.signal_in_count += 1; bad.head_out_count += 1;
    s = valid_set(w, 1); s.in[sig].zero_point = 5; s.out[head].zero_point = 5;
    expect(w, &s, STREAM_PORT_ERR_STATE_ZERO_POINT, "zero_point on a state under the real split", "zero_point");
    expect(&bad, &s, STREAM_PORT_OK, "zero_point on a state moved into the signals", NULL);

    /* scale / format */
    s = valid_set(w, 1); s.in[0].scale_ratio = 0.0f;
    expect(w, &s, STREAM_PORT_ERR_SCALE, "zero scale", "scale");
    s = valid_set(w, 1); s.out[0].scale_ratio = 0.0f / 0.0f;
    expect(w, &s, STREAM_PORT_ERR_SCALE, "NaN scale", "scale");
    s = valid_set(w, 1); s.out[0].scale_ratio = 1.0f / 0.0f;
    expect(w, &s, STREAM_PORT_ERR_SCALE, "Inf scale", "scale");
    s = valid_set(w, 0); s.in[0].scale_ratio = 0.0f;
    expect(w, &s, STREAM_PORT_OK, "float port ignores scale_ratio", NULL);
    s = valid_set(w, 0); s.in[0].elem_bits = 24u;
    expect(w, &s, STREAM_PORT_ERR_FORMAT, "24-bit element", "unsupported");
    s = valid_set(w, 1); s.in[0].is_float = 1;
    expect(w, &s, STREAM_PORT_ERR_FORMAT, "float flag with 16 bits", "unsupported");
    s = valid_set(w, 2); s.out[0].elem_bits = 32u;
    expect(w, &s, STREAM_PORT_ERR_FORMAT, "fixed point with 32 bits", "unsupported");

    /* wiring validity: counts and state shapes */
    s = valid_set(w, 0);
    bad = *w; bad.input_count = -1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative input_count", "invalid port count");
    bad = *w; bad.input_count = STREAM_MAX_IN_PORTS + 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "input_count above the maximum", "invalid port count");
    bad = *w; bad.output_count = -1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative output_count", "invalid port count");
    bad = *w; bad.output_count = STREAM_MAX_OUT_PORTS + 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "output_count above the maximum", "invalid port count");
    bad = *w; bad.signal_in_count = -1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative signal_in_count", "invalid port count");
    bad = *w; bad.signal_in_count = w->input_count + 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "signal_in_count above input_count", "invalid port count");
    bad = *w; bad.head_out_count = -1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative head_out_count", "invalid port count");
    bad = *w; bad.head_out_count = w->output_count + 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "head_out_count above output_count", "invalid port count");
    bad = *w; bad.signal_in_count += 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "state counts differ (one more signal)", "invalid port count");
    bad = *w; bad.head_out_count += 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "state counts differ (one more head)", "invalid port count");
    bad = *w; bad.signal_in_count = 0; bad.head_out_count = 0; bad.input_count = w->input_count;
    bad.output_count = w->output_count;
    expect(&bad, &s, w->input_count == w->output_count ? STREAM_PORT_OK : STREAM_PORT_ERR_NULL,
           "all ports states: only equal lists are consistent", NULL);
    /* each bound alone: the state counts stay equal, only one limit is broken */
    bad = *w; bad.input_count = STREAM_MAX_IN_PORTS + 1; bad.signal_in_count = bad.input_count - n_state;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "input_count above the maximum, states equal", "invalid port count");
    bad = *w; bad.output_count = STREAM_MAX_OUT_PORTS + 1; bad.head_out_count = bad.output_count - n_state;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "output_count above the maximum, states equal", "invalid port count");
    bad = *w; bad.signal_in_count = w->input_count + 1; bad.head_out_count = w->output_count + 1;
    expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative state count, equal on both sides", "invalid port count");
    if (w->input_count - w->output_count - 1 >= 0) {
        bad = *w; bad.head_out_count = -1; bad.signal_in_count = w->input_count - w->output_count - 1;
        expect(&bad, &s, STREAM_PORT_ERR_NULL, "negative head_out_count, states equal", "invalid port count");
    }
    for (k = 0; k < n_state; ++k) {
        snprintf(needle, sizeof(needle), "wiring state %d", k);
        bad = *w; bad.outputs[head + k].w += 1u;
        expect(&bad, &s, STREAM_PORT_ERR_NULL, "wiring state output wider than its input", needle);
        bad = *w; bad.inputs[sig + k].c += 1u;
        expect(&bad, &s, STREAM_PORT_ERR_NULL, "wiring state input has more channels", needle);
        bad = *w; bad.outputs[head + k].b += 1u;
        expect(&bad, &s, STREAM_PORT_ERR_NULL, "wiring state output has another batch", needle);
        bad = *w; bad.outputs[head + k].h += 1u;
        expect(&bad, &s, STREAM_PORT_ERR_NULL, "wiring state output is taller than its input", needle);
        bad = *w;   /* same element count, size-1 axis in another slot: the in-place binding needs the same layout */
        bad.outputs[head + k].b = w->outputs[head + k].c; bad.outputs[head + k].c = w->outputs[head + k].b;
        if (bad.outputs[head + k].b != w->outputs[head + k].b)
            expect(&bad, &s, STREAM_PORT_ERR_NULL, "wiring state with the b and c axes exchanged", needle);
    }
    CHECK(stream_check_ports(NULL, s.in, s.n_in, s.out, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_NULL && strstr(msg, "null wiring"),
          "NULL wiring");

    /* a short message buffer, a NULL buffer */
    s = valid_set(w, 0); s.in[0].line_ofs += 4u;
    {
        char tiny[8];
        memset(tiny, 'X', sizeof(tiny));
        CHECK(stream_check_ports(w, s.in, s.n_in, s.out, s.n_out, tiny, sizeof(tiny)) == STREAM_PORT_ERR_STRIDE && strlen(tiny) == 7u,
              "short msg buffer is truncated, terminated");
        CHECK(stream_check_ports(w, s.in, s.n_in, s.out, s.n_out, NULL, 0) == STREAM_PORT_ERR_STRIDE, "NULL msg buffer");
        CHECK(stream_check_ports(w, s.in, s.n_in, s.out, s.n_out, tiny, 0) == STREAM_PORT_ERR_STRIDE, "zero-length msg buffer");
    }
    /* success clears a stale message */
    s = valid_set(w, 0);
    strcpy(msg, "stale");
    CHECK(run(w, &s, msg, sizeof(msg)) == STREAM_PORT_OK && msg[0] == '\0', "%s: success leaves an empty message", name);
}

/* tensor size that wraps 64 bits must not pass the capacity check */
static void run_overflow(void) {
    StreamWiring w;
    Set s;
    memset(&w, 0, sizeof(w));
    w.inputs[0].b = 1u << 22; w.inputs[0].c = 1u << 22; w.inputs[0].h = 1u << 22; w.inputs[0].w = 1u;
    w.input_count = 1; w.signal_in_count = 1;
    memset(&s, 0, sizeof(s));
    s.n_in = 1; s.in[0] = contiguous(&w.inputs[0], 0, 0);
    s.in[0].line_ofs = 4u; s.in[0].channel_ofs = 0u; s.in[0].batch_ofs = 0u; s.in[0].capacity_bytes = 0u;
    expect(&w, &s, STREAM_PORT_ERR_CAPACITY, "tensor size wraps 64 bits", "overflow");
}

/* More outputs than inputs: a negative signal_in_count keeps the state counts equal. */
static void run_negative_signal_count(void) {
    StreamWiring w;
    Set s;
    char msg[64];
    memset(&w, 0, sizeof(w));
    w.input_count = 2; w.output_count = 3; w.signal_in_count = -1; w.head_out_count = 0;
    memset(&s, 0, sizeof(s));
    s.n_in = 2; s.n_out = 3;
    CHECK(stream_check_ports(&w, s.in, s.n_in, s.out, s.n_out, msg, sizeof(msg)) == STREAM_PORT_ERR_NULL, "negative signal_in_count, states equal");
}

/* A wiring with no ports at all is a valid, empty graph. */
static void run_empty(void) {
    StreamWiring w;
    char msg[64];
    memset(&w, 0, sizeof(w));
    CHECK(stream_check_ports(&w, NULL, 0, NULL, 0, msg, sizeof(msg)) == STREAM_PORT_OK, "empty wiring");
    CHECK(stream_check_ports(&w, NULL, 1, NULL, 0, msg, sizeof(msg)) == STREAM_PORT_ERR_IN_COUNT, "empty wiring, one input");
}

int main(void) {
    static StreamWiring w;
    const int codes[] = { STREAM_PORT_ERR_NULL, STREAM_PORT_ERR_IN_COUNT, STREAM_PORT_ERR_OUT_COUNT, STREAM_PORT_ERR_AXES,
                          STREAM_PORT_ERR_FORMAT, STREAM_PORT_ERR_SCALE, STREAM_PORT_ERR_STRIDE, STREAM_PORT_ERR_CAPACITY,
                          STREAM_PORT_ERR_STATE_PAIR, STREAM_PORT_ERR_STATE_ZERO_POINT };
    int a, b;

    build_dfn2(&w, 1, 1, 96);  run_profile(&w, "dfn2 full N=1 16k/256");
    build_dfn2(&w, 1, 8, 96);  run_profile(&w, "dfn2 full N=8 16k/256");
    build_dfn2(&w, 1, 1, 128); run_profile(&w, "dfn2 full N=1 df_bins 128");
    build_dfn2(&w, 0, 1, 96);  run_profile(&w, "dfn2 lite N=1");
    build_dfn2(&w, 0, 8, 96);  run_profile(&w, "dfn2 lite N=8");
    build_deepvqe(&w, 1, 129); run_profile(&w, "deepvqe_s N=1 16k/256");
    build_deepvqe(&w, 8, 129); run_profile(&w, "deepvqe_s N=8 16k/256");
    build_deepvqe(&w, 1, 257); run_profile(&w, "deepvqe_s N=1 16k/512");
    build_align(&w, 1, 129);   run_profile(&w, "align_ulcnet N=1 16k/256");
    build_align(&w, 8, 129);   run_profile(&w, "align_ulcnet N=8 16k/256");
    build_align(&w, 1, 257);   run_profile(&w, "align_ulcnet N=1 16k/512");
    run_overflow();
    run_negative_signal_count();
    run_empty();

    /* the per-model counts of the convention */
    build_dfn2(&w, 1, 1, 96);
    CHECK(w.signal_in_count == 2 && w.head_out_count == 3 && w.input_count == 4 && w.output_count == 5, "dfn2 full counts");
    build_dfn2(&w, 0, 1, 96);
    CHECK(w.signal_in_count == 2 && w.head_out_count == 1 && w.input_count == 3 && w.output_count == 2, "dfn2 lite counts");
    build_deepvqe(&w, 1, 129);
    CHECK(w.signal_in_count == 2 && w.head_out_count == 1 && w.input_count == 18 && w.output_count == 17, "deepvqe_s counts");
    build_align(&w, 1, 129);
    CHECK(w.signal_in_count == 4 && w.head_out_count == 1 && w.input_count == 8 && w.output_count == 5, "align N=1 counts");
    build_align(&w, 8, 129);
    CHECK(w.signal_in_count == 5 && w.head_out_count == 1 && w.input_count == 9 && w.output_count == 5, "align N>1 counts");

    CHECK(STREAM_PORT_OK == 0, "OK is 0");
    for (a = 0; a < (int)(sizeof(codes) / sizeof(codes[0])); ++a) {
        CHECK(codes[a] < 0, "code %d is negative", codes[a]);
        for (b = a + 1; b < (int)(sizeof(codes) / sizeof(codes[0])); ++b)
            CHECK(codes[a] != codes[b], "duplicate port-check code %d", codes[a]);
    }
    CHECK(STREAM_PORT_ERR_NULL == -101 && STREAM_PORT_ERR_STATE_PAIR == -109 && STREAM_PORT_ERR_STATE_ZERO_POINT == -110, "code values");

    {   /* stream_port_shape fills elements */
        StreamPortShape p = stream_port_shape(1u, 2u, 3u, 4u);
        CHECK(p.b == 1u && p.c == 2u && p.h == 3u && p.w == 4u && p.elements == 24u, "stream_port_shape");
    }

    if (g_fail) { printf("test_stream_bind: %d of %d check(s) FAILED\n", g_fail, g_checks); return 1; }
    printf("test_stream_bind: PASS (%d checks)\n", g_checks);
    return 0;
}
