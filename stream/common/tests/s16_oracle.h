/* Test oracle of the float PCM -> int16 output rule, computed in double and independent of stream_saturate_s16:
 * scale by 32768, NaN is 0, clip to [-32768, 32767], round half away from zero. Header-only; not part of the library. */

#ifndef S16_ORACLE_H
#define S16_ORACLE_H

#include <math.h>
#include <stdint.h>

static inline int16_t s16_oracle(float x)
{
    double s = (double)x * 32768.0, r;
    if (isnan(s)) return 0;
    if (s > 32767.0) s = 32767.0;
    if (s < -32768.0) s = -32768.0;
    r = floor(fabs(s) + 0.5);
    return (int16_t)(s < 0.0 ? -r : r);
}

#endif /* S16_ORACLE_H */
