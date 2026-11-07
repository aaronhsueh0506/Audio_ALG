/* Compatibility forwarding header: the three mask helpers this used to declare moved to their
 * owning model's own core (lib/aiaec/{cagcrn,align_cruse,ulcnet}/), split by the single-model
 * release restructuring. Kept so any existing `#include "aiaec_process.h"` still resolves all
 * three declarations unchanged. */
#ifndef AIAEC_PROCESS_H
#define AIAEC_PROCESS_H

#include "../lib/aiaec/cagcrn/cagcrn_process.h"
#include "../lib/aiaec/align_cruse/align_cruse_process.h"
#include "../lib/aiaec/ulcnet/ulcnet_mask_process.h"

#endif
