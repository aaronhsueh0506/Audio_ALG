/* SDK-free graph wiring and opened-graph check, used by the board samples. Every converted
 * graph has inputs = signals, then states; outputs = heads, then states; state k is input
 * signal_in_count + k <-> output head_out_count + k. A StreamWiring describes the graph, a
 * StreamDeviceDesc each port the SDK opened; stream_check_ports() compares them before the first run.
 * No stdio: messages are assembled by hand (no printf-family reference). */

#ifndef STREAM_COMMON_BIND_H
#define STREAM_COMMON_BIND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Largest port lists; stream_check_ports() refuses a larger count. */
#define STREAM_MAX_IN_PORTS  18
#define STREAM_MAX_OUT_PORTS 17

/* ONNX logical shape of one port (lower rank: leading axes 1); elements = b * c * h * w. */
typedef struct StreamPortShape {
    unsigned int b, c, h, w;
    size_t       elements;
} StreamPortShape;

/* The graph's ports by position: signal_in_count signals then states; head_out_count heads then states.
 * The state counts are equal and state k has the same shape at its input and its output. */
typedef struct StreamWiring {
    int             signal_in_count;
    int             head_out_count;
    int             input_count;
    int             output_count;
    StreamPortShape inputs[STREAM_MAX_IN_PORTS];
    StreamPortShape outputs[STREAM_MAX_OUT_PORTS];
} StreamWiring;

static inline StreamPortShape stream_port_shape(unsigned int b, unsigned int c, unsigned int h,
                                                unsigned int w)
{
    StreamPortShape s;
    s.b = b; s.c = c; s.h = h; s.w = w;
    s.elements = (size_t)b * c * h * w;
    return s;
}

/* What the SDK reports for one opened port; the sample fills one per port in SDK order. */
typedef struct StreamDeviceDesc {
    unsigned int w, h, c, b;      /* the SDK's axes, w innermost */
    unsigned int elem_bits;       /* 32, 16 or 8 */
    int          is_float;        /* 1: float32 (scale/zero_point ignored); 0: fixed point */
    float        scale_ratio;
    int          zero_point;
    uint64_t     line_ofs;        /* bytes between rows; must be w * bytes per element */
    uint64_t     channel_ofs;     /* bytes between channels; 0 (unreported) or line_ofs * h */
    uint64_t     batch_ofs;       /* bytes between batches; 0 (unreported) or channel_ofs * c */
    uint64_t     capacity_bytes;  /* usable bytes of the device buffer */
} StreamDeviceDesc;

/* Port check codes (distinct from each library's own error codes). */
enum {
    STREAM_PORT_OK = 0,
    STREAM_PORT_ERR_NULL = -101,        /* NULL or invalid argument / wiring (counts, state shapes) */
    STREAM_PORT_ERR_IN_COUNT = -102,    /* wrong number of input ports */
    STREAM_PORT_ERR_OUT_COUNT = -103,   /* wrong number of output ports */
    STREAM_PORT_ERR_AXES = -104,        /* axes do not match the ONNX logical shape */
    STREAM_PORT_ERR_FORMAT = -105,      /* unsupported element format */
    STREAM_PORT_ERR_SCALE = -106,       /* fixed-point port with a zero or non-finite scale_ratio */
    STREAM_PORT_ERR_STRIDE = -107,      /* padded / non-contiguous layout */
    STREAM_PORT_ERR_CAPACITY = -108,    /* device buffer smaller than the tensor */
    STREAM_PORT_ERR_STATE_PAIR = -109,  /* a state input and its output disagree */
    STREAM_PORT_ERR_STATE_ZERO_POINT = -110   /* a fixed-point state port (either end) has zero_point != 0 */
};

/* Checks the opened graph (in[n_in], out[n_out]) against w before the first run: 0 or a STREAM_PORT_ERR_*, msg (may
 * be NULL) gets a one-line reason naming the port ("input 3 (state 0)"). Counts first; per port: zero axis, format,
 * scale, axes (size-1 ignored), contiguous stride, capacity; per state pair: same at both ends, zero_point 0 if fixed. */
int stream_check_ports(const StreamWiring *w,
                       const StreamDeviceDesc in[], int n_in,
                       const StreamDeviceDesc out[], int n_out,
                       char *msg, size_t msg_len);

#ifdef __cplusplus
}
#endif

#endif /* STREAM_COMMON_BIND_H */
