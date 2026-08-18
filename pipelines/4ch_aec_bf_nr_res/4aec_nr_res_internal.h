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

/* ============================================================================
 * Shared-delay change admission (lib/aec Path-B mirror)
 *
 * A change to an alignment already in force is admitted only when the
 * movement exceeds FOUR_DELAY_CHANGE_MIN_SAMPLES and the same value is still
 * being offered, within FOUR_DELAY_CHANGE_CONFIRM_SAMPLES, before the held
 * candidate ages out. Same three numbers as lib/aec's own Path B (aec.c:
 * `abs(new_delay - current_delay) > 32`, `abs(new_delay - pending_delay) <
 * 16`, `pending_delay_ttl = 3`), in native samples and hops.
 *
 * Exposed here rather than kept static because the TTL is not reachable from
 * a stream: DelayAec3 re-offers a movement on every hop once it has one, so
 * a candidate is always resolved on the very next eligible hop and never
 * ages. Driving this state machine directly is the only way to test expiry
 * without pretending a synthetic scene produced it.
 * ========================================================================== */

#define FOUR_DELAY_CHANGE_MIN_SAMPLES      32
#define FOUR_DELAY_CHANGE_CONFIRM_SAMPLES  16
#define FOUR_DELAY_CHANGE_CANDIDATE_TTL     3

/* `ttl` is both the countdown and the "a candidate is held" flag: lib/aec
 * spells those as pending_delay_ttl plus has_pending, but sets and clears the
 * two together on every path, so one counter cannot disagree with itself. */
typedef struct FourAecDelayAdmission {
    int candidate;
    int ttl;
} FourAecDelayAdmission;

/* One hop of life spent, whether or not this hop had a usable estimate --
 * lib/aec ages outside its own eligibility test too, so an estimate that
 * stops being usable spends the candidate's life exactly like a differing
 * one does. Call once per hop, before offer(). */
void four_aec_nr_res_admission_age(FourAecDelayAdmission* admission);

/* Offer this hop's eligible estimate against the alignment in force. Returns
 * 1 when the movement is admitted (the caller applies it and realigns), 0
 * when it is absorbed as too small, or held/replaced as a candidate. */
int four_aec_nr_res_admission_offer(
    FourAecDelayAdmission* admission, int accepted_delay, int estimated);

#endif /* FOUR_AEC_NR_RES_INTERNAL_H */
