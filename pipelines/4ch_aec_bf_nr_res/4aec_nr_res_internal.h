/** Internal delay-admission state machine and diagnostic hooks, non-static so they can be driven directly. Public entry points: 4aec_nr_res.h. */
#ifndef FOUR_AEC_NR_RES_INTERNAL_H
#define FOUR_AEC_NR_RES_INTERNAL_H

#include "4aec_nr_res.h"
#include <limits.h>

static inline long four_aec_count_add_sat(long total, long add) {
    if (total < 0 || add < 0 || add > LONG_MAX - total) return LONG_MAX;
    return total + add;
}

static inline int four_aec_far_fft_delta_matches(long before, long after,
                                                 long expected) {
    if (before < 0 || after < before || after == LONG_MAX) return 0;
    return after - before == expected;
}

/* Shared-delay change admission (lib/aec's Path-B rule): a change to an alignment in force is admitted only when the movement exceeds FOUR_DELAY_CHANGE_MIN_SAMPLES and is offered again, within FOUR_DELAY_CHANGE_CONFIRM_SAMPLES, before the held candidate ages out (TTL in hops). */

#define FOUR_DELAY_CHANGE_MIN_SAMPLES      32
#define FOUR_DELAY_CHANGE_CONFIRM_SAMPLES  16
#define FOUR_DELAY_CHANGE_CANDIDATE_TTL     3

typedef struct FourAecDelayAdmission {
    int candidate;
    int ttl;
} FourAecDelayAdmission;

/* One hop of life spent, whether or not this hop had a usable estimate. Call once per
 * hop, before offer(). */
void four_aec_nr_res_admission_age(FourAecDelayAdmission* admission);

/* Offers this hop's eligible estimate against the alignment in force. Returns 1 when
 * the movement is admitted (the caller applies it and realigns), 0 when it is absorbed
 * as too small or held/replaced as a candidate. */
int four_aec_nr_res_admission_offer(
    FourAecDelayAdmission* admission, int accepted_delay, int estimated);

/* Cross-lane far-end agreement (fuse-stage precondition). A mixed warm/soft-realign hop skips this predicate (p->far_spec_mixed_hop), so a stream only exercises its accepting branch. */

/* 1 = this lane may be folded into the beam, 0 = it carries a different far-end
 * spectrum and the fuse must reject the frame. shared_far_provenance is the per-hop
 * evidence that the four lanes consumed one spectrum (four_aec_front_process()); with
 * it the answer is immediate, without it every bin is compared. */
int four_aec_nr_res_far_spec_agrees(int shared_far_provenance,
                                    const Complex* lane_far_spec,
                                    const Complex* reference_far_spec,
                                    int n_freqs);

int four_aec_nr_res_far_spec_provenance(const FourAecNrRes* p);

/* Matched-filter duty cycle pinned at full rate: removes only the C-only analysis decimation. */

/* Analyses every hop for the rest of this core's life; idempotent, inert outside AEC_DELAY_MATCHED, not undone by reset. */
void four_aec_nr_res_pin_duty_full_rate(FourAecNrRes* p);

/* Per-hop leak the grid implies; the authored per-hop literal on a non-positive grid. */
float four_aec_nr_res_duty_leak_db_for_grid(int hop_size, int sample_rate);

/* The value this core converted for its own grid at init; 0.0f on NULL or a destroyed
 * core. */
float four_aec_nr_res_duty_erle_leak_db(const FourAecNrRes* p);

/* The leaky peak itself; 0.0f on NULL or a destroyed core. */
float four_aec_nr_res_duty_erle_peak(const FourAecNrRes* p);

#endif /* FOUR_AEC_NR_RES_INTERNAL_H */
