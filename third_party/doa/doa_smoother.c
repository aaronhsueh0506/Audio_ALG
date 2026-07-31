#include <stdlib.h>
#include <math.h>
#include "doa_smoother.h"

static int same_angle(float a, float b, float tol)
{
    if (isnan(a) || isnan(b)) return 0;
    return fabsf(a - b) <= tol;
}


float doa_smoother_update(
    DOA_Smoother* s,
    float doa_raw,
    int vad
)
{
    if (!vad || isnan(doa_raw)) {
        s->pending = NAN;
        s->cnt = 0;
        return s->null_value;
    }

    float cur = doa_raw;

    if (!s->initialized) {
        s->last = cur;
        s->initialized = 1;
        return s->last;
    }

    if (same_angle(cur, s->last, s->angle_tol)) {
        s->pending = NAN;
        s->cnt = 0;
        return s->last;
    }

    if (isnan(s->pending) || !same_angle(cur, s->pending, s->angle_tol)) {
        s->pending = cur;
        s->cnt = 1;
    } else {
        s->cnt++;
    }

    if (s->cnt >= s->switch_consec) {
        s->last = s->pending;
        s->pending = NAN;
        s->cnt = 0;
    }

    return s->last;
}
