/* aiaec_front_preprocess() and aiaec_front_status_text() in their own object, so the front's lower-level steps link
 * without libaiaec.a. */

#include <string.h>

#include "aiaec_front_internal.h"
#include "aiaec_internal.h"

int aiaec_front_preprocess(AiaecFront *f, AiaecCtx *ctx, const int16_t *mic, const int16_t *far,
                           AiaecWork *work) {
    AiaecFrontOut out;
    AiaecInput in;
    int rc;

    if (work) memset(work, 0, sizeof(*work));
    if (!f || !ctx || !mic || !far || !work) return AIAEC_ERR_NULL;
    rc = aiaec_precheck(ctx, AIAEC_INPUT_ALIGN_ULCNET, f->hop_in, f->frames);
    if (rc != AIAEC_OK) return rc;
    rc = aiaec_front_process_i16(f, mic, far, f->hop_in, &out);
    if (rc != AIAEC_OK) return rc;
    aiaec_front_fill_align_input(&out, &in);
    return aiaec_preprocess(ctx, &in, work);
}

const char *aiaec_front_status_text(int code) {
    switch (code) {
    case AIAEC_FRONT_ERR_DELAY: return "illegal delay policy";
    case AIAEC_FRONT_ERR_AEC: return "the AEC library refused the resolved configuration";
    default: return aiaec_status_text(code);
    }
}
