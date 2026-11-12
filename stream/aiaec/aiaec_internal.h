/* Private to stream/aiaec: defined in libaiaec.a, called by libaiaec_front.a. */

#ifndef AIAEC_INTERNAL_H
#define AIAEC_INTERNAL_H

#include "aiaec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* aiaec_preprocess()'s sequence, kind and size gates for `kind`, `samples` and `event_count` (Align-ULCNet only), touching
 * no state: AIAEC_OK, or the status the call would return. */
int aiaec_precheck(const AiaecCtx *ctx, AiaecInputKind kind, int samples, int event_count);

#ifdef __cplusplus
}
#endif

#endif /* AIAEC_INTERNAL_H */
