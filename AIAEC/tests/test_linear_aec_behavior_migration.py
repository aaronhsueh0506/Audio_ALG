"""No migration connects recorded data to this frontend.

`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` is empty: the shadow-copy baseline retime
and the C restart evidence clearing (2026-09-04) move `linear_error`, so the
pair that carried the 200-hour corpus forward (37ed5ad9 -> 19dd4f90) was
retired with the identities before it.

What must NOT happen is a retired entry coming back. Byte-identity evidence
describes no frontend after a change that moves the stem; retargeting one at
a live build would declare an old waveform compatible with a build that does
not produce it. Retired identities are refused with an instruction to
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

# The pair retired most recently: the frontend the 200-hour corpus was
# materialized under and the identity it had been carried forward to. Spelled
# out for the same reason as the two above.
CORPUS_HASH = (
    "37ed5ad9b75ce42902361d8195fcf04a650b940744ec036a16c8736dec9d5061")
LAST_MIGRATED_TO_HASH = (
    "19dd4f90f482e15072d535964ac9816cdc21cae2c350b98de12a0e9ab561ff45")
REFUSED_HASHES = (DEPLOYED_OLD_HASH, EARLIER_MIGRATED_HASH, CORPUS_HASH,
                  LAST_MIGRATED_TO_HASH)


def _contract_recorded_as(hash_value):
    current = make_linear_aec_contract(16000, preset="balanced")
    return dataclasses.replace(current, aec_behavior_hash=hash_value).as_dict()


def test_the_frontend_identity_actually_changed():
    """The premise. If the hash still matched the old one, every refusal below
    would be vacuous."""
    assert aec_python_behavior_hash() not in REFUSED_HASHES


def test_the_migration_table_is_empty():
    assert ACCEPTED_BEHAVIOR_HASH_MIGRATIONS == {}, (
        "every pair in this table needs its own admission evidence")


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


def test_an_unlisted_identity_is_still_refused():
    """The table is an explicit, per-pair exemption, not a general amnesty."""
    with pytest.raises(ValueError):
        require_linear_aec_contract(
            make_linear_aec_contract(16000, preset="balanced").as_dict(),
            _contract_recorded_as("f" * 64), context="fixture")


@pytest.mark.parametrize("recorded", REFUSED_HASHES)
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


@pytest.mark.parametrize("recorded", REFUSED_HASHES)
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
