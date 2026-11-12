/* Private to libaiaec_front.a. */

#ifndef AIAEC_FRONT_INTERNAL_H
#define AIAEC_FRONT_INTERNAL_H

#include "aec.h"
#include "aiaec_front.h"

struct AiaecFront {
    Aec            *aec;
    float          *mic_f;
    float          *far_f;
    float          *formed;
    AiaecAlignEvent *events;
    int             hop;
    int             frames;
    int             hop_in;
};

#endif /* AIAEC_FRONT_INTERNAL_H */
