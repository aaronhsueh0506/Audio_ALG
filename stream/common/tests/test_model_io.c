/* ai3_model_io.h alone, over small synthetic rows spelled here (no sample tables, no SDK): row expansion,
 * row lookup, table fit, frame count, state pairing, buffer sizing, the pre -> graph boundary check on
 * stand-in work structs shaped like AinrWork / AiaecWork, and the port check delegated to
 * stream_check_ports(). Each CHECK counts one assertion; the exit status is non-zero on any failure. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ai3_model_io.h"

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; printf("FAIL (line %d): ", __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { M_A = 1, M_B = 2, M_C = 3, M_D = 4 };

/* A: 2 signals, 3 heads, 3 states, every axis distinct so a swap or a mix-up is visible; the counts differ
 * (2 vs 3) so a signal-first and a head-first base cannot be confused. */
static Ai3ModelIoRow row_a(void)
{
    Ai3ModelIoRow r;
    memset(&r, 0, sizeof r);
    r.model = M_A; r.sample_rate = 16000; r.fft = 256; r.n = 1; r.layout_version = 11;
    r.signal_count = 2; r.head_count = 3; r.state_count = 3;
    r.signal[0][0] = 2; r.signal[0][1] = 3; r.signal[0][2] = 4; r.signal[0][3] = 5;
    r.signal[1][0] = 1; r.signal[1][1] = 1; r.signal[1][2] = 1; r.signal[1][3] = 9;
    r.head[0][0] = 1; r.head[0][1] = 2; r.head[0][2] = 3; r.head[0][3] = 4;
    r.head[1][0] = 1; r.head[1][1] = 1; r.head[1][2] = 1; r.head[1][3] = 6;
    r.head[2][0] = 1; r.head[2][1] = 1; r.head[2][2] = 2; r.head[2][3] = 7;
    r.state[0][0] = 1; r.state[0][1] = 2; r.state[0][2] = 3; r.state[0][3] = 5;
    r.state[1][0] = 1; r.state[1][1] = 1; r.state[1][2] = 4; r.state[1][3] = 6;
    r.state[2][0] = 1; r.state[2][1] = 3; r.state[2][2] = 1; r.state[2][3] = 2;
    return r;
}

/* B: 3 signals, 1 head, 1 state. */
static Ai3ModelIoRow row_b(void)
{
    Ai3ModelIoRow r;
    memset(&r, 0, sizeof r);
    r.model = M_B; r.sample_rate = 48000; r.fft = 1024; r.n = 8; r.layout_version = 12;
    r.signal_count = 3; r.head_count = 1; r.state_count = 1;
    r.signal[0][0] = 1; r.signal[0][1] = 1; r.signal[0][2] = 8; r.signal[0][3] = 33;
    r.signal[1][0] = 1; r.signal[1][1] = 2; r.signal[1][2] = 8; r.signal[1][3] = 3;
    r.signal[2][0] = 1; r.signal[2][1] = 1; r.signal[2][2] = 1; r.signal[2][3] = 8;
    r.head[0][0] = 1; r.head[0][1] = 1; r.head[0][2] = 8; r.head[0][3] = 17;
    r.state[0][0] = 1; r.state[0][1] = 1; r.state[0][2] = 5; r.state[0][3] = 2;
    return r;
}

/* C: no state at all. */
static Ai3ModelIoRow row_c(void)
{
    Ai3ModelIoRow r;
    memset(&r, 0, sizeof r);
    r.model = M_C; r.sample_rate = 16000; r.fft = 512; r.n = 1; r.layout_version = 13;
    r.signal_count = 1; r.head_count = 1; r.state_count = 0;
    r.signal[0][0] = 1; r.signal[0][1] = 2; r.signal[0][2] = 3; r.signal[0][3] = 4;
    r.head[0][0] = 1; r.head[0][1] = 2; r.head[0][2] = 3; r.head[0][3] = 4;
    return r;
}

/* D: the table limits, 5 signals, 3 heads, 12 states. */
static Ai3ModelIoRow row_d(void)
{
    Ai3ModelIoRow r;
    int i;
    memset(&r, 0, sizeof r);
    r.model = M_D; r.sample_rate = 16000; r.fft = 256; r.n = 8; r.layout_version = 14;
    r.signal_count = AI3_MODEL_IO_MAX_SIGNALS; r.head_count = AI3_MODEL_IO_MAX_HEADS; r.state_count = 12;
    for (i = 0; i < r.signal_count; ++i) { r.signal[i][0] = 1; r.signal[i][1] = 1; r.signal[i][2] = (unsigned)i + 1; r.signal[i][3] = 2; }
    for (i = 0; i < r.head_count; ++i) { r.head[i][0] = 1; r.head[i][1] = 1; r.head[i][2] = (unsigned)i + 1; r.head[i][3] = 3; }
    for (i = 0; i < r.state_count; ++i) { r.state[i][0] = 1; r.state[i][1] = 1; r.state[i][2] = (unsigned)i + 1; r.state[i][3] = 4; }
    return r;
}

static int same_shape(const StreamPortShape *a, const StreamPortShape *b)
{
    return a->b == b->b && a->c == b->c && a->h == b->h && a->w == b->w && a->elements == b->elements;
}

static int is_zeroed(const Ai3ModelIoConfig *c)
{
    Ai3ModelIoConfig z;
    memset(&z, 0, sizeof z);
    return memcmp(c, &z, sizeof z) == 0;
}

/* The codes are part of the board samples' log vocabulary: pinned, and clear of the -1xx port codes and the
 * -2xx AIAEC front codes. */
static void test_error_codes(void)
{
    CHECK(AI3_MODEL_IO_ERR_NULL == -301 && AI3_MODEL_IO_ERR_NO_ROW == -302 && AI3_MODEL_IO_ERR_ROW == -303 &&
          AI3_MODEL_IO_ERR_COUNT == -304 && AI3_MODEL_IO_ERR_SHAPE == -305 && AI3_MODEL_IO_ERR_INDEX == -306,
          "AI3_MODEL_IO_ERR_* values");
    CHECK(AI3_MODEL_IO_OK == 0, "AI3_MODEL_IO_OK is 0");
    CHECK(AI3_MODEL_IO_ERR_NULL < STREAM_PORT_ERR_STATE_ZERO_POINT && AI3_MODEL_IO_ERR_INDEX < -299,
          "below the -1xx and -2xx ranges");
}

static void test_frames(void)
{
    CHECK(ai3_model_io_frames(256, 128) == 1, "N for hop_in 128");
    CHECK(ai3_model_io_frames(256, 1024) == 8, "N for hop_in 1024");
    CHECK(ai3_model_io_frames(1024, 512) == 1 && ai3_model_io_frames(1024, 4096) == 8, "N at fft 1024");
    CHECK(ai3_model_io_frames(256, 129) == 0, "129 is not a multiple of 128");
    CHECK(ai3_model_io_frames(256, 127) == 0 && ai3_model_io_frames(256, 192) == 0, "non-multiples refused");
    CHECK(ai3_model_io_frames(256, 0) == 0 && ai3_model_io_frames(256, -128) == 0, "non-positive hop_in");
    CHECK(ai3_model_io_frames(0, 128) == 0 && ai3_model_io_frames(1, 128) == 0 && ai3_model_io_frames(-256, 128) == 0,
          "no model hop");
}

static void test_expand(void)
{
    Ai3ModelIoRow ra = row_a(), rb = row_b(), rc_ = row_c(), rd = row_d(), r;
    Ai3ModelIoConfig c;
    const StreamWiring *w = &c.wiring;
    int k, i;

    CHECK(ai3_model_io_expand(&ra, &c) == AI3_MODEL_IO_OK, "row A expands");
    CHECK(c.model == M_A && c.sample_rate == 16000 && c.fft == 256 && c.n == 1 && c.layout_version == 11,
          "row A configuration copied");
    CHECK(w->signal_in_count == 2 && w->head_out_count == 3 && w->input_count == 5 && w->output_count == 6,
          "row A counts %d/%d/%d/%d", w->signal_in_count, w->head_out_count, w->input_count, w->output_count);
    for (i = 0; i < ra.signal_count; ++i) {
        StreamPortShape want = stream_port_shape(ra.signal[i][0], ra.signal[i][1], ra.signal[i][2], ra.signal[i][3]);
        CHECK(same_shape(&w->inputs[i], &want), "signal %d is input %d", i, i);
    }
    for (i = 0; i < ra.head_count; ++i) {
        StreamPortShape want = stream_port_shape(ra.head[i][0], ra.head[i][1], ra.head[i][2], ra.head[i][3]);
        CHECK(same_shape(&w->outputs[i], &want), "head %d is output %d", i, i);
    }
    for (k = 0; k < ra.state_count; ++k) {
        StreamPortShape want = stream_port_shape(ra.state[k][0], ra.state[k][1], ra.state[k][2], ra.state[k][3]);
        CHECK(same_shape(&w->inputs[ra.signal_count + k], &want), "state %d input at signal_count + k", k);
        CHECK(same_shape(&w->outputs[ra.head_count + k], &want), "state %d output at head_count + k, same shape", k);
        CHECK(same_shape(&w->inputs[ra.signal_count + k], &w->outputs[ra.head_count + k]), "state %d mirrored", k);
    }
    CHECK(w->inputs[0].elements == 120 && w->inputs[1].elements == 9 && w->outputs[2].elements == 14 &&
          w->inputs[2].elements == 30, "element counts");

    CHECK(ai3_model_io_expand(&rb, &c) == AI3_MODEL_IO_OK && w->signal_in_count == 3 && w->head_out_count == 1 &&
          w->input_count == 4 && w->output_count == 2 && same_shape(&w->inputs[3], &w->outputs[1]) &&
          w->inputs[3].h == 5 && w->inputs[3].w == 2, "row B");
    CHECK(ai3_model_io_expand(&rc_, &c) == AI3_MODEL_IO_OK && w->input_count == 1 && w->output_count == 1 &&
          ai3_model_io_state_count(&c) == 0, "row C without state");
    CHECK(ai3_model_io_expand(&rd, &c) == AI3_MODEL_IO_OK && w->input_count == 17 && w->output_count == 15 &&
          same_shape(&w->inputs[5 + 11], &w->outputs[3 + 11]) && w->inputs[16].h == 12, "row D at the table limits");

    /* failure zeroes cfg, whatever it held */
    memset(&c, 0x5a, sizeof c);
    r = ra; r.signal[1][2] = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c), "zero signal axis: ERR_ROW, cfg zeroed");

    /* a zero in any of the four axes of any used entry */
    for (i = 0; i < 4; ++i) {
        r = ra; r.signal[0][i] = 0;
        CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c), "signal axis %d zero", i);
        r = ra; r.head[2][i] = 0;
        CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c), "head axis %d zero", i);
        r = ra; r.state[2][i] = 0;
        CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c), "last state axis %d zero", i);
    }
    /* entries past the counts are not read */
    r = ra; r.state_count = 2; r.state[2][0] = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.input_count == 4, "unused state entry ignored");
    r = ra; r.signal_count = 1; r.signal[1][0] = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.input_count == 4, "unused signal entry ignored");
    r = ra; r.head_count = 1; r.head[2][0] = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.output_count == 4, "unused head entry ignored");

    /* count limits */
    r = rd; r.signal_count = AI3_MODEL_IO_MAX_SIGNALS + 1;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c), "6 signals");
    r = rd; r.signal_count = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "0 signals");
    r = rd; r.head_count = AI3_MODEL_IO_MAX_HEADS + 1;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "4 heads");
    r = rd; r.head_count = 0;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "0 heads");
    r = rd; r.state_count = -1;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "negative states");
    r = row_c(); r.state_count = AI3_MODEL_IO_MAX_STATES;
    for (i = 0; i < r.state_count; ++i) { r.state[i][0] = r.state[i][1] = r.state[i][2] = 1; r.state[i][3] = 2; }
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.input_count == 17 && c.wiring.output_count == 17,
          "1 + 16 states fills 17 ports");
    r.state_count = AI3_MODEL_IO_MAX_STATES + 1;
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "17 states");
    r = rd; for (i = 12; i < AI3_MODEL_IO_MAX_STATES; ++i) { r.state[i][0] = r.state[i][1] = r.state[i][2] = 1; r.state[i][3] = 2; }
    r.signal_count = 5; r.state_count = 14;       /* 19 inputs */
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "inputs past STREAM_MAX_IN_PORTS");
    r = rd; for (i = 12; i < AI3_MODEL_IO_MAX_STATES; ++i) { r.state[i][0] = r.state[i][1] = r.state[i][2] = 1; r.state[i][3] = 2; }
    r.signal_count = 5; r.state_count = 13;       /* 18 inputs, 16 outputs */
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.input_count == STREAM_MAX_IN_PORTS,
          "exactly STREAM_MAX_IN_PORTS inputs");
    r = rd; for (i = 12; i < AI3_MODEL_IO_MAX_STATES; ++i) { r.state[i][0] = r.state[i][1] = r.state[i][2] = 1; r.state[i][3] = 2; }
    r.signal_count = 1; r.head_count = 3; r.state_count = 15;   /* 18 outputs */
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "outputs past STREAM_MAX_OUT_PORTS");
    r.state_count = 14;   /* 17 outputs */
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_OK && c.wiring.output_count == STREAM_MAX_OUT_PORTS,
          "exactly STREAM_MAX_OUT_PORTS outputs");

    CHECK(ai3_model_io_expand(NULL, &c) == AI3_MODEL_IO_ERR_NULL && ai3_model_io_expand(&ra, NULL) == AI3_MODEL_IO_ERR_NULL,
          "expand NULL");
    memset(&r, 0, sizeof r);
    CHECK(ai3_model_io_expand(&r, &c) == AI3_MODEL_IO_ERR_ROW, "all-zero row");
}

static void test_find_row(void)
{
    Ai3ModelIoRow rows[4];
    Ai3ModelIoConfig c;
    Ai3ModelIoRow bad;

    rows[0] = row_a(); rows[1] = row_b(); rows[2] = row_c(); rows[3] = row_d();

    CHECK(ai3_model_io_find_row(rows, 4, M_A, 16000, 256, 1, &c) == AI3_MODEL_IO_OK && c.model == M_A &&
          c.layout_version == 11 && c.wiring.input_count == 5, "find A");
    CHECK(ai3_model_io_find_row(rows, 4, M_B, 48000, 1024, 8, &c) == AI3_MODEL_IO_OK && c.layout_version == 12 &&
          c.wiring.input_count == 4, "find B");
    CHECK(ai3_model_io_find_row(rows, 4, M_C, 16000, 512, 1, &c) == AI3_MODEL_IO_OK && c.layout_version == 13, "find C");
    CHECK(ai3_model_io_find_row(rows, 4, M_D, 16000, 256, 8, &c) == AI3_MODEL_IO_OK && c.layout_version == 14 &&
          c.wiring.input_count == 17, "find D (same rate and fft as A, other model and N)");

    /* each key field alone refuses, and cfg is zeroed */
#define NO_ROW(m, s, f, n_, what) do { memset(&c, 0x5a, sizeof c); \
    CHECK(ai3_model_io_find_row(rows, 4, (m), (s), (f), (n_), &c) == AI3_MODEL_IO_ERR_NO_ROW && is_zeroed(&c), what); } while (0)
    NO_ROW(M_A + 10, 16000, 256, 1, "wrong model");
    NO_ROW(M_A, 48000, 256, 1, "wrong sample rate");
    NO_ROW(M_A, 16000, 512, 1, "wrong fft");
    NO_ROW(M_A, 16000, 256, 2, "wrong N");
    NO_ROW(M_A, 16000, 256, 0, "N = 0 (bad hop_in) is never derived");
    NO_ROW(M_B, 48000, 1024, 1, "B only at N = 8");
    NO_ROW(M_D, 16000, 256, 1, "D only at N = 8");
#undef NO_ROW
    CHECK(ai3_model_io_find_row(rows, 0, M_A, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_NO_ROW && is_zeroed(&c), "empty table");
    CHECK(ai3_model_io_find_row(rows, -1, M_A, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_NO_ROW, "negative count");
    CHECK(ai3_model_io_find_row(rows, 1, M_B, 48000, 1024, 8, &c) == AI3_MODEL_IO_ERR_NO_ROW,
          "rows past row_count are not searched");

    CHECK(ai3_model_io_find_row(NULL, 4, M_A, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_NULL &&
          ai3_model_io_find_row(rows, 4, M_A, 16000, 256, 1, NULL) == AI3_MODEL_IO_ERR_NULL, "find NULL");

    bad = row_a(); bad.head[0][2] = 0;
    CHECK(ai3_model_io_find_row(&bad, 1, M_A, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_ROW && is_zeroed(&c),
          "a matched but malformed row is ERR_ROW, cfg zeroed");
}

static void test_fits(void)
{
    Ai3ModelIoRow ra = row_a(), rb = row_b();
    Ai3ModelIoConfig a, b, m;

    ai3_model_io_expand(&ra, &a);
    ai3_model_io_expand(&rb, &b);
    CHECK(ai3_model_io_fits(&a, 2, 3) == AI3_MODEL_IO_OK, "A fits 2/3");
    CHECK(ai3_model_io_fits(&a, 5, 3) == AI3_MODEL_IO_OK && ai3_model_io_fits(&a, 2, 5) == AI3_MODEL_IO_OK, "larger tables");
    CHECK(ai3_model_io_fits(&a, 1, 3) == AI3_MODEL_IO_ERR_ROW, "one signal slot short");
    CHECK(ai3_model_io_fits(&a, 2, 2) == AI3_MODEL_IO_ERR_ROW, "one head slot short");
    CHECK(ai3_model_io_fits(&a, 0, 3) == AI3_MODEL_IO_ERR_ROW && ai3_model_io_fits(&a, 2, 0) == AI3_MODEL_IO_ERR_ROW,
          "zero capacity");
    CHECK(ai3_model_io_fits(&b, 3, 1) == AI3_MODEL_IO_OK && ai3_model_io_fits(&b, 2, 1) == AI3_MODEL_IO_ERR_ROW,
          "B needs three signal slots");
    CHECK(ai3_model_io_fits(NULL, 5, 5) == AI3_MODEL_IO_ERR_NULL, "fits NULL");

    /* a hand-edited wiring that no row can produce */
    m = a; m.wiring.signal_in_count = 0;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "no signals");
    m = a; m.wiring.head_out_count = 0;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "no heads");
    m = a; m.wiring.output_count += 1;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "state counts differ");
    m = a; m.wiring.input_count = STREAM_MAX_IN_PORTS + 1; m.wiring.output_count = m.wiring.head_out_count + m.wiring.input_count - m.wiring.signal_in_count;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "inputs past the port limit");
    m = a; m.wiring.input_count = m.wiring.signal_in_count - 1;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "fewer inputs than signals");
    m = a; m.wiring.output_count = STREAM_MAX_OUT_PORTS + 1; m.wiring.input_count = m.wiring.signal_in_count + m.wiring.output_count - m.wiring.head_out_count;
    CHECK(ai3_model_io_fits(&m, 5, 5) == AI3_MODEL_IO_ERR_ROW, "outputs past the port limit");
}

static void test_state_pairing(void)
{
    Ai3ModelIoRow ra = row_a(), rc_ = row_c(), rd = row_d();
    Ai3ModelIoConfig a, c, d, none;
    const StreamWiring *w;
    int k, i;

    ai3_model_io_expand(&ra, &a);
    ai3_model_io_expand(&rc_, &c);
    ai3_model_io_expand(&rd, &d);
    w = &a.wiring;

    CHECK(ai3_model_io_state_count(&a) == 3 && ai3_model_io_state_count(&c) == 0 && ai3_model_io_state_count(&d) == 12,
          "state counts");
    for (k = 0; k < 3; ++k) {
        int in = ai3_model_io_state_in(&a, k), out = ai3_model_io_state_out(&a, k);
        CHECK(in == 2 + k, "state %d input index %d (signals first)", k, in);
        CHECK(out == 3 + k, "state %d output index %d (heads first)", k, out);
        CHECK(ai3_model_io_in_state(&a, in) == k && ai3_model_io_out_state(&a, out) == k, "inverse of state %d", k);
        CHECK(same_shape(&w->inputs[in], &w->outputs[out]), "state %d pair shapes", k);
    }
    CHECK(ai3_model_io_state_in(&a, 3) == -1 && ai3_model_io_state_in(&a, -1) == -1, "state_in range");
    CHECK(ai3_model_io_state_out(&a, 3) == -1 && ai3_model_io_state_out(&a, -1) == -1, "state_out range");
    for (i = 0; i < w->signal_in_count; ++i) CHECK(ai3_model_io_in_state(&a, i) == -1, "signal %d is no state", i);
    for (i = 0; i < w->head_out_count; ++i) CHECK(ai3_model_io_out_state(&a, i) == -1, "head %d is no state", i);
    CHECK(ai3_model_io_in_state(&a, -1) == -1 && ai3_model_io_in_state(&a, w->input_count) == -1, "in_state range");
    CHECK(ai3_model_io_out_state(&a, -1) == -1 && ai3_model_io_out_state(&a, w->output_count) == -1, "out_state range");
    CHECK(ai3_model_io_state_in(&c, 0) == -1 && ai3_model_io_state_out(&c, 0) == -1 && ai3_model_io_in_state(&c, 0) == -1 &&
          ai3_model_io_out_state(&c, 0) == -1, "a graph without state pairs nothing");
    CHECK(ai3_model_io_state_in(&d, 11) == 16 && ai3_model_io_state_out(&d, 11) == 14 && ai3_model_io_state_in(&d, 12) == -1,
          "last state of D");

    CHECK(ai3_model_io_state_count(NULL) == 0 && ai3_model_io_state_in(NULL, 0) == -1 &&
          ai3_model_io_state_out(NULL, 0) == -1 && ai3_model_io_in_state(NULL, 0) == -1 &&
          ai3_model_io_out_state(NULL, 0) == -1, "NULL cfg");
    memset(&none, 0, sizeof none);
    CHECK(ai3_model_io_state_count(&none) == 0 && ai3_model_io_state_in(&none, 0) == -1 &&
          ai3_model_io_state_out(&none, 0) == -1, "zeroed cfg");
}

static void test_bytes(void)
{
    Ai3ModelIoRow ra = row_a();
    Ai3ModelIoConfig a;
    size_t e;

    ai3_model_io_expand(&ra, &a);
    CHECK(ai3_model_io_input_bytes(&a, 0, 4) == 120u * 4u, "signal 0 float bytes");
    CHECK(ai3_model_io_input_bytes(&a, 0, 1) == 120u && ai3_model_io_input_bytes(&a, 0, 2) == 240u, "int8 / int16 bytes");
    CHECK(ai3_model_io_input_bytes(&a, 1, 4) == 9u * 4u, "signal 1 bytes");
    CHECK(ai3_model_io_input_bytes(&a, 4, 4) == 6u * 4u, "last state input bytes");
    CHECK(ai3_model_io_output_bytes(&a, 0, 4) == 24u * 4u && ai3_model_io_output_bytes(&a, 2, 4) == 14u * 4u, "head bytes");
    CHECK(ai3_model_io_output_bytes(&a, 5, 4) == 6u * 4u && ai3_model_io_output_bytes(&a, 3, 4) == 30u * 4u, "state output bytes");

    /* bad index, zero element size, NULL */
    CHECK(ai3_model_io_input_bytes(&a, 5, 4) == 0 && ai3_model_io_input_bytes(&a, -1, 4) == 0 &&
          ai3_model_io_input_bytes(&a, 0, 0) == 0 && ai3_model_io_input_bytes(NULL, 0, 4) == 0, "input_bytes refusals");
    CHECK(ai3_model_io_output_bytes(&a, 6, 4) == 0 && ai3_model_io_output_bytes(&a, -1, 4) == 0 &&
          ai3_model_io_output_bytes(&a, 0, 0) == 0 && ai3_model_io_output_bytes(NULL, 0, 4) == 0, "output_bytes refusals");
    CHECK(ai3_model_io_input_bytes(&a, 0, 4) != 0, "index 0 is valid, not a refusal");

    /* overflow reports 0, never a wrapped size; the largest exact product is kept */
    CHECK(ai3_model_io_input_bytes(&a, 0, (size_t)-1) == 0 && ai3_model_io_output_bytes(&a, 0, (size_t)-1) == 0,
          "elements * SIZE_MAX overflows");
    e = a.wiring.inputs[0].elements;                     /* 120 */
    CHECK(ai3_model_io_input_bytes(&a, 0, (size_t)-1 / e) == e * ((size_t)-1 / e), "largest exact product kept");
    CHECK(ai3_model_io_input_bytes(&a, 0, (size_t)-1 / e + 1) == 0, "one past the largest product");
    e = a.wiring.outputs[0].elements;                    /* 24 */
    CHECK(ai3_model_io_output_bytes(&a, 0, (size_t)-1 / e) == e * ((size_t)-1 / e) &&
          ai3_model_io_output_bytes(&a, 0, (size_t)-1 / e + 1) == 0, "output bytes at the limit");
    a.wiring.inputs[0].elements = 1;
    CHECK(ai3_model_io_input_bytes(&a, 0, (size_t)-1) == (size_t)-1, "1 * SIZE_MAX is exact");
}

typedef struct { const float *data; size_t elements; unsigned int w, h, c, b; } StandInIn;
typedef struct { float *data; size_t elements; unsigned int w, h, c, b; } StandInOut;

/* A work like AinrWork: counts after the arrays. One spare input slot, so a count one above the widest row
 * still indexes inside the struct. */
typedef struct { StandInIn inputs[AI3_MODEL_IO_MAX_SIGNALS + 1]; int input_count;
                 StandInOut outputs[AI3_MODEL_IO_MAX_HEADS]; int output_count; } WorkX;

/* A work like AiaecWork: other member order, extra members, larger arrays. */
typedef struct { int input_count; int output_count; unsigned long long block_id;
                 StandInIn inputs[8]; StandInOut outputs[8]; } WorkY;

static float g_buf[4];

static void fill_x(const Ai3ModelIoConfig *c, WorkX *wk)
{
    const StreamWiring *w = &c->wiring;
    int i;
    memset(wk, 0, sizeof *wk);
    wk->input_count = w->signal_in_count;
    for (i = 0; i < w->signal_in_count; ++i) {
        wk->inputs[i].data = g_buf; wk->inputs[i].elements = w->inputs[i].elements;
        wk->inputs[i].b = w->inputs[i].b; wk->inputs[i].c = w->inputs[i].c;
        wk->inputs[i].h = w->inputs[i].h; wk->inputs[i].w = w->inputs[i].w;
    }
    wk->output_count = w->head_out_count;
    for (i = 0; i < w->head_out_count; ++i) {
        wk->outputs[i].data = g_buf; wk->outputs[i].elements = w->outputs[i].elements;
        wk->outputs[i].b = w->outputs[i].b; wk->outputs[i].c = w->outputs[i].c;
        wk->outputs[i].h = w->outputs[i].h; wk->outputs[i].w = w->outputs[i].w;
    }
}

static void fill_y(const Ai3ModelIoConfig *c, WorkY *wk)
{
    WorkX x;
    int i;
    fill_x(c, &x);
    memset(wk, 0, sizeof *wk);
    wk->input_count = x.input_count; wk->output_count = x.output_count; wk->block_id = 7;
    for (i = 0; i < x.input_count; ++i) wk->inputs[i] = x.inputs[i];
    for (i = 0; i < x.output_count; ++i) wk->outputs[i] = x.outputs[i];
}

static void test_check_counts_and_tensors(void)
{
    Ai3ModelIoRow ra = row_a();
    Ai3ModelIoConfig a;
    ai3_model_io_expand(&ra, &a);
    CHECK(ai3_model_io_check_counts(&a, 2, 3) == AI3_MODEL_IO_OK, "counts match");
    CHECK(ai3_model_io_check_counts(&a, 3, 3) == AI3_MODEL_IO_ERR_COUNT && ai3_model_io_check_counts(&a, 1, 3) == AI3_MODEL_IO_ERR_COUNT,
          "signal count off by one either way");
    CHECK(ai3_model_io_check_counts(&a, 2, 4) == AI3_MODEL_IO_ERR_COUNT && ai3_model_io_check_counts(&a, 2, 2) == AI3_MODEL_IO_ERR_COUNT,
          "head count off by one either way");
    CHECK(ai3_model_io_check_counts(&a, 5, 6) == AI3_MODEL_IO_ERR_COUNT, "state-inclusive counts are not the work's counts");
    CHECK(ai3_model_io_check_counts(&a, 3, 2) == AI3_MODEL_IO_ERR_COUNT && ai3_model_io_check_counts(&a, 0, 0) == AI3_MODEL_IO_ERR_COUNT,
          "swapped / empty counts");
    CHECK(ai3_model_io_check_counts(NULL, 2, 3) == AI3_MODEL_IO_ERR_NULL, "check_counts NULL");

    /* input 0 = [2,3,4,5], 120 elements */
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 2, 3, 4, 5) == AI3_MODEL_IO_OK, "input 0 exact");
    CHECK(ai3_model_io_check_input(&a, 1, g_buf, 9, 1, 1, 1, 9) == AI3_MODEL_IO_OK, "input 1 exact");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 121, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "one element more");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 119, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "one element fewer");
    /* each axis alone wrong, element count right: the axes are compared, not only the product */
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 3, 3, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "b alone");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 2, 4, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "c alone");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 2, 3, 5, 5) == AI3_MODEL_IO_ERR_SHAPE, "h alone");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 2, 3, 4, 6) == AI3_MODEL_IO_ERR_SHAPE, "w alone");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 2, 3, 5, 4) == AI3_MODEL_IO_ERR_SHAPE, "h <-> w, same element count");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 120, 3, 2, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "b <-> c, same element count");
    CHECK(ai3_model_io_check_input(&a, 0, g_buf, 9, 1, 1, 1, 9) == AI3_MODEL_IO_ERR_SHAPE, "input 1's tensor in slot 0");
    CHECK(ai3_model_io_check_input(&a, 1, g_buf, 120, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_SHAPE, "input 0's tensor in slot 1");
    CHECK(ai3_model_io_check_input(&a, 0, NULL, 120, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_NULL, "no data");
    CHECK(ai3_model_io_check_input(&a, 0, NULL, 1, 1, 1, 1, 1) == AI3_MODEL_IO_ERR_NULL, "no data, wrong shape: NULL");
    CHECK(ai3_model_io_check_input(&a, 2, g_buf, 30, 1, 2, 3, 5) == AI3_MODEL_IO_ERR_INDEX, "state 0's input is not a signal");
    CHECK(ai3_model_io_check_input(&a, a.wiring.signal_in_count, g_buf, 30, 1, 2, 3, 5) == AI3_MODEL_IO_ERR_INDEX,
          "index == signal_in_count");
    CHECK(ai3_model_io_check_input(&a, -1, g_buf, 120, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_INDEX, "negative index");
    CHECK(ai3_model_io_check_input(NULL, 0, g_buf, 120, 2, 3, 4, 5) == AI3_MODEL_IO_ERR_NULL, "check_input NULL cfg");

    /* output 0 = [1,2,3,4], 24; output 2 = [1,1,2,7], 14 */
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 2, 3, 4) == AI3_MODEL_IO_OK, "output 0 exact");
    CHECK(ai3_model_io_check_output(&a, 2, g_buf, 14, 1, 1, 2, 7) == AI3_MODEL_IO_OK, "output 2 exact (last head)");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 25, 1, 2, 3, 4) == AI3_MODEL_IO_ERR_SHAPE &&
          ai3_model_io_check_output(&a, 0, g_buf, 23, 1, 2, 3, 4) == AI3_MODEL_IO_ERR_SHAPE, "output element count");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 2, 2, 3, 4) == AI3_MODEL_IO_ERR_SHAPE, "output b alone");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 3, 3, 4) == AI3_MODEL_IO_ERR_SHAPE, "output c alone");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 2, 4, 4) == AI3_MODEL_IO_ERR_SHAPE, "output h alone");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 2, 3, 5) == AI3_MODEL_IO_ERR_SHAPE, "output w alone");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 2, 4, 3) == AI3_MODEL_IO_ERR_SHAPE, "output h <-> w");
    CHECK(ai3_model_io_check_output(&a, 0, g_buf, 24, 1, 3, 2, 4) == AI3_MODEL_IO_ERR_SHAPE, "output c <-> h");
    CHECK(ai3_model_io_check_output(&a, 0, NULL, 24, 1, 2, 3, 4) == AI3_MODEL_IO_ERR_NULL, "output without data");
    CHECK(ai3_model_io_check_output(&a, 3, g_buf, 30, 1, 2, 3, 5) == AI3_MODEL_IO_ERR_INDEX, "state 0's output is not a head");
    CHECK(ai3_model_io_check_output(&a, a.wiring.head_out_count, g_buf, 14, 1, 1, 2, 7) == AI3_MODEL_IO_ERR_INDEX,
          "index == head_out_count");
    CHECK(ai3_model_io_check_output(&a, -1, g_buf, 24, 1, 2, 3, 4) == AI3_MODEL_IO_ERR_INDEX, "negative head index");
    CHECK(ai3_model_io_check_output(NULL, 0, g_buf, 24, 1, 2, 3, 4) == AI3_MODEL_IO_ERR_NULL, "check_output NULL cfg");
}

/* The macro over both stand-in work types. */
#define WORK_CASES(WORK, FILL) do {                                                                          \
    Ai3ModelIoConfig cs[4]; Ai3ModelIoRow rows[4]; int r, i, rc, ax;                                         \
    WORK wk;                                                                                                 \
    rows[0] = row_a(); rows[1] = row_b(); rows[2] = row_c(); rows[3] = row_d();                              \
    for (r = 0; r < 4; ++r) ai3_model_io_expand(&rows[r], &cs[r]);                                           \
    for (r = 0; r < 4; ++r) {                                                                                \
        const Ai3ModelIoConfig *c = &cs[r];                                                                  \
        FILL(c, &wk); AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                                                   \
        CHECK(rc == AI3_MODEL_IO_OK, #WORK " row %d matching work rc %d", r, rc);                            \
        FILL(c, &wk); wk.input_count += 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                              \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d one signal too many rc %d", r, rc);               \
        FILL(c, &wk); wk.input_count -= 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                              \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d one signal short rc %d", r, rc);                  \
        FILL(c, &wk); wk.output_count += 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                             \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d one head too many rc %d", r, rc);                 \
        FILL(c, &wk); wk.output_count -= 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                             \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d one head short rc %d", r, rc);                    \
        FILL(c, &wk); wk.input_count = 0; wk.output_count = 0; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);          \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d empty work rc %d", r, rc);                        \
        FILL(c, &wk); wk.input_count += 1; wk.inputs[wk.input_count - 1].data = NULL;                        \
        AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                                                                 \
        CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " row %d counts are checked before tensors rc %d", r, rc);\
        for (i = 0; i < c->wiring.signal_in_count; ++i) {                                                    \
            unsigned int *axes[4];                                                                           \
            FILL(c, &wk); wk.inputs[i].elements += 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                   \
            CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d signal %d one element more rc %d", r, i, rc);\
            FILL(c, &wk); wk.inputs[i].elements -= 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                   \
            CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d signal %d one element fewer rc %d", r, i, rc);\
            FILL(c, &wk); wk.inputs[i].data = NULL; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                     \
            CHECK(rc == AI3_MODEL_IO_ERR_NULL, #WORK " row %d signal %d no data rc %d", r, i, rc);           \
            for (ax = 0; ax < 4; ++ax) {                                                                     \
                FILL(c, &wk);                                                                                \
                axes[0] = &wk.inputs[i].b; axes[1] = &wk.inputs[i].c;                                        \
                axes[2] = &wk.inputs[i].h; axes[3] = &wk.inputs[i].w;                                        \
                *axes[ax] += 1;   /* element count unchanged */                                              \
                AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                                                         \
                CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d signal %d axis %d alone rc %d", r, i, ax, rc);\
            }                                                                                                \
        }                                                                                                    \
        for (i = 0; i < c->wiring.head_out_count; ++i) {                                                     \
            unsigned int *axes[4];                                                                           \
            FILL(c, &wk); wk.outputs[i].elements += 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                  \
            CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d head %d one element more rc %d", r, i, rc);  \
            FILL(c, &wk); wk.outputs[i].elements -= 1; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                  \
            CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d head %d one element fewer rc %d", r, i, rc); \
            FILL(c, &wk); wk.outputs[i].data = NULL; AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                    \
            CHECK(rc == AI3_MODEL_IO_ERR_NULL, #WORK " row %d head %d no data rc %d", r, i, rc);             \
            for (ax = 0; ax < 4; ++ax) {                                                                     \
                FILL(c, &wk);                                                                                \
                axes[0] = &wk.outputs[i].b; axes[1] = &wk.outputs[i].c;                                      \
                axes[2] = &wk.outputs[i].h; axes[3] = &wk.outputs[i].w;                                      \
                *axes[ax] += 1;                                                                              \
                AI3_MODEL_IO_CHECK_WORK(rc, c, &wk);                                                         \
                CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " row %d head %d axis %d alone rc %d", r, i, ax, rc);\
            }                                                                                                \
        }                                                                                                    \
    }                                                                                                        \
    /* A: tensors in each other's slots, and the first mismatch wins (inputs before outputs) */              \
    FILL(&cs[0], &wk); { StandInIn t = wk.inputs[0]; wk.inputs[0] = wk.inputs[1]; wk.inputs[1] = t; }         \
    AI3_MODEL_IO_CHECK_WORK(rc, &cs[0], &wk);                                                                \
    CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " swapped signal slots rc %d", rc);                            \
    FILL(&cs[0], &wk); { StandInOut t = wk.outputs[0]; wk.outputs[0] = wk.outputs[2]; wk.outputs[2] = t; }   \
    AI3_MODEL_IO_CHECK_WORK(rc, &cs[0], &wk);                                                                \
    CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " swapped head slots rc %d", rc);                              \
    FILL(&cs[0], &wk); wk.inputs[1].elements += 1; wk.outputs[0].data = NULL;                                \
    AI3_MODEL_IO_CHECK_WORK(rc, &cs[0], &wk);                                                                \
    CHECK(rc == AI3_MODEL_IO_ERR_SHAPE, #WORK " input mismatch reported before the head's rc %d", rc);       \
    FILL(&cs[0], &wk); wk.inputs[0].data = NULL; wk.inputs[1].elements += 1;                                 \
    AI3_MODEL_IO_CHECK_WORK(rc, &cs[0], &wk);                                                                \
    CHECK(rc == AI3_MODEL_IO_ERR_NULL, #WORK " first mismatch wins rc %d", rc);                              \
    /* the work's state slots are not part of the work: a cfg with states accepts exactly the signals */    \
    FILL(&cs[0], &wk); wk.input_count = cs[0].wiring.input_count;                                            \
    AI3_MODEL_IO_CHECK_WORK(rc, &cs[0], &wk);                                                                \
    CHECK(rc == AI3_MODEL_IO_ERR_COUNT, #WORK " state-inclusive input count refused rc %d", rc);             \
    CHECK(ai3_model_io_check_counts(NULL, 1, 1) == AI3_MODEL_IO_ERR_NULL, "NULL cfg");                       \
} while (0)

static void test_check_work(void)
{
    WORK_CASES(WorkX, fill_x);
    WORK_CASES(WorkY, fill_y);
}

/* Device descriptors that describe a shape list exactly: float32, contiguous, capacity = tensor. */
static void descs_of(const StreamPortShape *p, int n, StreamDeviceDesc *d)
{
    int i;
    memset(d, 0, sizeof(*d) * (size_t)n);
    for (i = 0; i < n; ++i) {
        d[i].w = p[i].w; d[i].h = p[i].h; d[i].c = p[i].c; d[i].b = p[i].b;
        d[i].elem_bits = 32; d[i].is_float = 1;
        d[i].line_ofs = (uint64_t)p[i].w * 4u;
        d[i].channel_ofs = d[i].line_ofs * p[i].h;
        d[i].batch_ofs = d[i].channel_ofs * p[i].c;
        d[i].capacity_bytes = (uint64_t)p[i].elements * 4u;
    }
}

static void test_check_ports(void)
{
    Ai3ModelIoRow ra = row_a(), rc_ = row_c();
    Ai3ModelIoConfig a, c;
    const StreamWiring *w;
    StreamDeviceDesc din[STREAM_MAX_IN_PORTS], dout[STREAM_MAX_OUT_PORTS], keep;
    char msg[160];
    int i;

    ai3_model_io_expand(&ra, &a);
    ai3_model_io_expand(&rc_, &c);
    w = &a.wiring;
    descs_of(w->inputs, w->input_count, din);
    descs_of(w->outputs, w->output_count, dout);

    msg[0] = 'x';
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) == STREAM_PORT_OK,
          "exact ports: %s", msg);
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, NULL, 0) == STREAM_PORT_OK,
          "msg may be NULL");
    msg[0] = '\0';
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count - 1, dout, w->output_count, msg, sizeof msg) ==
          STREAM_PORT_ERR_IN_COUNT && msg[0] != '\0', "one input short, with a reason");
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count + 1, msg, sizeof msg) ==
          STREAM_PORT_ERR_OUT_COUNT, "one output too many");

    /* counts are the wiring's, the work's signal counts are not: a graph with only signals fails here */
    CHECK(ai3_model_io_check_ports(&a, din, w->signal_in_count, dout, w->head_out_count, msg, sizeof msg) ==
          STREAM_PORT_ERR_IN_COUNT, "signals-only port lists refused");

    /* the first signal has two non-unit, distinct axes: swap them with the same element count */
    keep = din[0];
    din[0].h = keep.w; din[0].w = keep.h;
    din[0].line_ofs = (uint64_t)din[0].w * 4u; din[0].channel_ofs = din[0].line_ofs * din[0].h;
    din[0].batch_ofs = din[0].channel_ofs * din[0].c;
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) ==
          STREAM_PORT_ERR_AXES, "axes swapped on signal 0");
    din[0] = keep;

    i = ai3_model_io_state_in(&a, 0);
    din[i].capacity_bytes -= 4u;
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) ==
          STREAM_PORT_ERR_CAPACITY, "state buffer one element short");
    din[i].capacity_bytes += 4u;

    i = ai3_model_io_state_out(&a, 2);
    keep = dout[i];
    dout[i].w += 1u; dout[i].line_ofs += 4u; dout[i].capacity_bytes += 4096u * 4096u;
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) != STREAM_PORT_OK,
          "state output differs from its input");
    dout[i] = keep;

    keep = dout[0];
    dout[0].elem_bits = 12;
    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) ==
          STREAM_PORT_ERR_FORMAT, "unsupported element format on a head");
    dout[0] = keep;

    CHECK(ai3_model_io_check_ports(&a, din, w->input_count, dout, w->output_count, msg, sizeof msg) == STREAM_PORT_OK,
          "restored");
    CHECK(ai3_model_io_check_ports(NULL, din, w->input_count, dout, w->output_count, msg, sizeof msg) == STREAM_PORT_ERR_NULL,
          "NULL cfg");
    CHECK(ai3_model_io_check_ports(&a, NULL, w->input_count, dout, w->output_count, msg, sizeof msg) == STREAM_PORT_ERR_NULL,
          "NULL descriptor array");

    descs_of(c.wiring.inputs, c.wiring.input_count, din);
    descs_of(c.wiring.outputs, c.wiring.output_count, dout);
    CHECK(ai3_model_io_check_ports(&c, din, 1, dout, 1, msg, sizeof msg) == STREAM_PORT_OK, "graph without state");
    CHECK(ai3_model_io_check_ports(&c, din, 2, dout, 1, msg, sizeof msg) == STREAM_PORT_ERR_IN_COUNT, "extra input");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_error_codes();
    test_frames();
    test_expand();
    test_find_row();
    test_fits();
    test_state_pairing();
    test_bytes();
    test_check_counts_and_tensors();
    test_check_work();
    test_check_ports();
    printf("test_model_io: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
