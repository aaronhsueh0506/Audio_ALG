/**
 * Internal delay-admission state machine, exposed non-static so the core's
 * own tests can drive TTL expiry directly. Public processing entry points,
 * including the atomic spectrum+weights post seam, live in 4aec_nr_res.h.
 */
#ifndef FOUR_AEC_NR_RES_INTERNAL_H
#define FOUR_AEC_NR_RES_INTERNAL_H

#include "4aec_nr_res.h"

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

/* ============================================================================
 * Cross-lane far-end agreement (fuse-stage precondition)
 *
 * Exposed for the same reason the admission machine is: the rejecting half
 * is unreachable from a stream. This file's own lanes always share one far
 * spectrum by construction, so a test driving process_pre()/process_post()
 * can only ever exercise the accepting branch -- the divergent case, which
 * is what the rejection exists for, has to be handed to the predicate
 * directly.
 * ========================================================================== */

/* 1 = this lane may be folded into the beam, 0 = it carries a different
 * far-end spectrum and the fuse must reject the frame.
 *
 * shared_far_provenance is the per-hop evidence that the four lanes consumed
 * ONE spectrum (see four_aec_nr_res_process_pre()); with it the answer is
 * immediate, without it every bin is compared. */
int four_aec_nr_res_far_spec_agrees(int shared_far_provenance,
                                    const Complex* lane_far_spec,
                                    const Complex* reference_far_spec,
                                    int n_freqs);

/* The provenance the LAST accepted process_pre() established, so a test can
 * assert that the cheap path is the one production actually takes (and that
 * a bailed-out hop leaves the expensive one). Read-only. */
int four_aec_nr_res_far_spec_provenance(const FourAecNrRes* p);

/* ============================================================================
 * Matched-filter duty cycle, pinned at full rate
 *
 * The duty machine in update_shared_delay() is a C-only SCHEDULE: it decides
 * on which hops the matched-filter analysis runs, and pipeline.py -- the
 * reference for the shared alignment's MATH -- has no counterpart for it
 * (delay_aec3.h says so of the underlying delay_aec3_accumulate_ex()). A
 * per-hop C/Python comparison therefore has to neutralise the schedule
 * first, or it measures the divergence the schedule IS instead of the
 * arithmetic it is supposed to gate.
 *
 * This is that neutraliser, and it is the only one that keeps the rest of
 * the core intact: the estimator, the admission machine, the quarantine and
 * the ring all run exactly as they do in production, and only the
 * decimation of the analysis is removed. Not reachable from the public
 * header, and nothing in the signal path calls it -- production keeps the
 * duty cycle it was built with. tests/dump_delay_parity.c is the caller.
 * ========================================================================== */

/* Analyse every hop for the rest of this core's life. Idempotent, and inert
 * outside AEC_DELAY_MATCHED (no matched filter is built there to decimate).
 * four_aec_nr_res_reset() does not undo it: the pin is a property of the
 * harness, not of the alignment generation the reset abandons. */
void four_aec_nr_res_pin_duty_full_rate(FourAecNrRes* p);

/* ============================================================================
 * Watchdog leak, and the conversion behind it
 *
 * The leak is the duty machine's only wall-clock RATE: 0.1 dB per second,
 * carried in whatever per-hop amount this core's grid makes that. Three
 * seams, because the claim has three parts and no one of them implies the
 * others. All read-only; nothing in the signal path calls any of them.
 *
 * _for_grid() is the conversion itself, pure, so a test can evaluate it at
 * the grid the constant was CALIBRATED on -- hop 160 at 16 kHz, where it must
 * return the authored literal bit-for-bit or the retime is a rewrite. That
 * grid has a 320-sample frame, which this pipeline does not build, so no
 * core can be constructed carrying that value and no reader can observe it.
 *
 * _duty_erle_leak_db() is what the core actually stored at init, so a test
 * can hold the shipped path to the conversion rather than to a restatement
 * of it.
 *
 * _duty_erle_peak() is the state the watchdog applies it to, so a test can
 * prove the branch READS the stored value -- the part neither of the other
 * two can see.
 * ========================================================================== */

/* The per-hop leak the given grid implies. Falls back to the authored
 * per-hop literal on a non-positive grid ("not converted"), rather than
 * dividing by zero. */
float four_aec_nr_res_duty_leak_db_for_grid(int hop_size, int sample_rate);

/* The value this core converted for its own grid at init. 0.0f on NULL or a
 * destroyed core. */
float four_aec_nr_res_duty_erle_leak_db(const FourAecNrRes* p);

/* The leaky peak itself, so a test can reconstruct the subtraction from the
 * trajectory. Without it a correct leak can sit in the control block while
 * the branch that is supposed to apply it keeps subtracting the raw literal,
 * and every value-level assertion stays green. 0.0f on NULL or a destroyed
 * core. */
float four_aec_nr_res_duty_erle_peak(const FourAecNrRes* p);

#endif /* FOUR_AEC_NR_RES_INTERNAL_H */
