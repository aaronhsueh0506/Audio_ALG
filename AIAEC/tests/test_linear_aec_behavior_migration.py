"""Exactly one, rate-scoped migration connects recorded data to this frontend.

`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` holds the composed fresh-instance-reset and
48-kHz FilterAnalyzer-correction pair, and nothing else. It is admitted only at
16 kHz: measured formed-output bytes are identical there, while the live
detector window intentionally changes at 48 kHz.

What must NOT happen is the pre-dominant-peak entries coming back. The
dominant-matched-filter-peak change MOVES `linear_error`, so their byte-identity
evidence describes no frontend after it; retargeting one at a live build would
declare an old waveform compatible with a build that does not produce it. They
stay in `RETIRED_BEHAVIOR_HASHES` and are refused with an instruction to
rematerialize rather than with a bare hash mismatch.

Run:
    python3 -m pytest AIAEC/tests/test_linear_aec_behavior_migration.py
"""
from __future__ import annotations

import dataclasses

import pytest

from AIAEC.dataset_gen.aec_behavior_hash import aec_python_behavior_hash
from AIAEC.dataset_gen.linear_aec import (
    ACCEPTED_BEHAVIOR_HASH_MIGRATIONS,
    BEHAVIOR_HASH_SCHEMA,
    RETIRED_BEHAVIOR_HASHES,
    make_linear_aec_contract,
    require_linear_aec_contract,
)


# Two identities that must stay refused: the frontend the shipped corpus and
# the Align-ULCNet checkpoints were built on, and one of the identities that
# had been verified equivalent to it. Both are spelled out rather than read
# from RETIRED_BEHAVIOR_HASHES: a test that takes its inputs from the table it
# is checking would still pass if the table were emptied.
DEPLOYED_OLD_HASH = (
    "abd8aa04272d8f833e948999fb54c5a9b2348d0f57f38efb21f43a15bbe6a239")
EARLIER_MIGRATED_HASH = (
    "eda3c3be25b4bb69762572b22447db7e004f870a395d7cf84ad1ce02ddd28cfe")

# The one identity that IS carried forward: the frontend the 200-hour corpus
# was materialized under, before aec_reset() was made to return a fresh
# instance. Spelled out for the same reason as the two above -- reading it
# back out of ACCEPTED_BEHAVIOR_HASH_MIGRATIONS would make every assertion
# below satisfied by whatever the table happens to contain.
MIGRATED_FROM_HASH = (
    "37ed5ad9b75ce42902361d8195fcf04a650b940744ec036a16c8736dec9d5061")


def _contract_recorded_as(hash_value):
    current = make_linear_aec_contract(16000, preset="balanced")
    return dataclasses.replace(current, aec_behavior_hash=hash_value).as_dict()


def test_the_frontend_identity_actually_changed():
    """The premise. If the hash still matched the old one, every refusal below
    would be vacuous."""
    assert aec_python_behavior_hash() != DEPLOYED_OLD_HASH


def test_the_migration_table_holds_exactly_the_one_admitted_pair():
    assert ACCEPTED_BEHAVIOR_HASH_MIGRATIONS == {
        MIGRATED_FROM_HASH: aec_python_behavior_hash()
    }, "every pair in this table needs its own admission evidence"


def test_no_retired_identity_migrates_to_anything():
    """Guards the specific mistake: retargeting a retired entry at a live hash
    instead of leaving it retired. A pair is admissible only on evidence that
    the stem renders byte-identical across it, and no such evidence can exist
    for a frontend that moves `linear_error`."""
    for retired in RETIRED_BEHAVIOR_HASHES:
        assert retired not in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS
    assert not (set(RETIRED_BEHAVIOR_HASHES)
                & set(ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.values()))
    assert aec_python_behavior_hash() not in RETIRED_BEHAVIOR_HASHES


def test_the_admitted_pair_is_accepted_with_a_warning():
    """The remote flow: a corpus materialized under the recorded identity,
    packed and validated by a build that computes the new one. Accepting
    SILENTLY would be as wrong as refusing -- the operator has to see which
    frontend identity was let through."""
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    with pytest.warns(RuntimeWarning, match="frontend-equivalent migration"):
        # (actual=this build, expected=the contract the data recorded)
        require_linear_aec_contract(
            current, _contract_recorded_as(MIGRATED_FROM_HASH),
            context="fixture")


def test_the_admitted_hash_pair_is_refused_at_48khz():
    """Hash equality evidence at 16 kHz must not pardon old 48-kHz data."""
    current = make_linear_aec_contract(
        48000, preset="balanced", frame_size=1024).as_dict()
    recorded = dict(current, aec_behavior_hash=MIGRATED_FROM_HASH)
    with pytest.raises(ValueError, match="aec_behavior_hash"):
        require_linear_aec_contract(current, recorded, context="fixture")


def test_the_admitted_pair_is_refused_in_reverse():
    """One direction by construction. Data recorded under the NEW identity run
    against the OLD build is a genuine downgrade, and the table is not
    consulted backwards."""
    recorded_new = _contract_recorded_as(aec_python_behavior_hash())
    running_old = _contract_recorded_as(MIGRATED_FROM_HASH)
    with pytest.raises(ValueError, match="contract mismatch"):
        require_linear_aec_contract(running_old, recorded_new,
                                    context="fixture")


def test_an_unlisted_identity_is_still_refused():
    """The table is an explicit, per-pair exemption, not a general amnesty."""
    with pytest.raises(ValueError):
        require_linear_aec_contract(
            make_linear_aec_contract(16000, preset="balanced").as_dict(),
            _contract_recorded_as("f" * 64), context="fixture")


@pytest.mark.parametrize("recorded", (DEPLOYED_OLD_HASH, EARLIER_MIGRATED_HASH))
def test_an_old_frontend_identity_is_refused_with_what_to_do(recorded):
    """Fails CLOSED, and says rematerialize -- not a bare hash mismatch."""
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    with pytest.raises(ValueError) as caught:
        # (actual=this build, expected=the contract the data recorded)
        require_linear_aec_contract(current, _contract_recorded_as(recorded),
                                    context="fixture")
    message = str(caught.value)
    assert recorded in message
    assert "rematerialize_linear_aec" in message
    assert "--resume" in message
    assert "retrain" in message


@pytest.mark.parametrize("recorded", (DEPLOYED_OLD_HASH, EARLIER_MIGRATED_HASH))
def test_an_old_identity_never_warns_that_it_is_equivalent(recorded, recwarn):
    """The failure mode this forbids: passing with a 'frontend-equivalent,
    no rematerialization required' warning, which is what the table used to
    emit for exactly these hashes."""
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    with pytest.raises(ValueError):
        # (actual=this build, expected=the contract the data recorded)
        require_linear_aec_contract(current, _contract_recorded_as(recorded),
                                    context="fixture")
    assert not [w for w in recwarn.list
                if "No rematerialization" in str(w.message)]


def test_the_hash_schema_is_unchanged():
    """A behaviour change, not a canonicalizer change: bumping the schema would
    invalidate every recorded hash for the wrong reason."""
    assert BEHAVIOR_HASH_SCHEMA == "canon-ast-1"
