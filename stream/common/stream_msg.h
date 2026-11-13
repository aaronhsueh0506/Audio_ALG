/* One-line messages for the SDK-free stream checkers, assembled by hand: no printf-family reference.
 * Internal to stream/common. */

#ifndef STREAM_COMMON_MSG_H
#define STREAM_COMMON_MSG_H

#include <stddef.h>

typedef struct StreamMsg {
    char *buf;
    size_t cap, len;
} StreamMsg;

static inline void msg_init(StreamMsg *m, char *buf, size_t cap)
{
    m->buf = buf; m->cap = buf ? cap : 0u; m->len = 0u;
    if (m->cap) m->buf[0] = '\0';
}

static inline StreamMsg *msg_s(StreamMsg *m, const char *s)
{
    while (m->cap && *s && m->len + 1u < m->cap) m->buf[m->len++] = *s++;
    if (m->cap) m->buf[m->len] = '\0';
    return m;
}

static inline StreamMsg *msg_u(StreamMsg *m, unsigned long long v)
{
    char t[24];
    int n = 0;
    do { t[n++] = (char)('0' + (int)(v % 10u)); v /= 10u; } while (v);
    while (n) { char c[2]; c[0] = t[--n]; c[1] = '\0'; msg_s(m, c); }
    return m;
}

/* "<dir> <idx> (<signal|head|state> <n>): ", ports named by position; the first state_start ports
 * of the list are signals (inputs) or heads (outputs). */
static inline StreamMsg *msg_port(StreamMsg *m, const char *dir, int idx, int state_start, const char *lead)
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

#endif /* STREAM_COMMON_MSG_H */
