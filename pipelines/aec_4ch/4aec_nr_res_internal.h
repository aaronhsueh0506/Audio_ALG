/**
 * Internal seam shared only by the complete SRP/GSC wrapper and its tests.
 * Public external beamformers must use four_aec_nr_res_process_post(), which
 * reconstructs the mono error from the supplied weights and therefore does
 * not trust a second independently supplied spectrum.
 */
#ifndef FOUR_AEC_NR_RES_INTERNAL_H
#define FOUR_AEC_NR_RES_INTERNAL_H

#include "4aec_nr_res.h"

int four_aec_nr_res_process_post_trusted_spectrum(
    FourAecNrRes* p,
    const FourAecNrResFrameToken* token,
    const Complex* weights,
    const Complex* beamformed_error,
    float* out);

#endif /* FOUR_AEC_NR_RES_INTERNAL_H */
