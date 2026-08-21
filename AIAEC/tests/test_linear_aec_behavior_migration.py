"""No migration connects the old linear_error waveform to this frontend.

The dominant-matched-filter-peak change MOVES `linear_error`. Every entry
`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` used to hold was admitted on evidence that
the stem rendered byte-identical across the pair, and that evidence does not
describe this build. Retargeting those entries would declare an old waveform
compatible with a build that does not produce it -- so they are retired, and
the identities they named are refused with an instruction to rematerialize
rather than with a bare hash mismatch.

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


# The frontend the shipped corpus and the Align-ULCNet checkpoints were built
# on, and one of the identities that had been verified equivalent to it. Both
# are spelled out rather than read from RETIRED_BEHAVIOR_HASHES: a test that
# takes its inputs from the table it is checking would still pass if the table
# were emptied.
DEPLOYED_OLD_HASH = (
    "abd8aa04272d8f833e948999fb54c5a9b2348d0f57f38efb21f43a15bbe6a239")
EARLIER_MIGRATED_HASH = (
    "eda3c3be25b4bb69762572b22447db7e004f870a395d7cf84ad1ce02ddd28cfe")


def _contract_recorded_as(hash_value):
    current = make_linear_aec_contract(16000, preset="balanced")
    return dataclasses.replace(current, aec_behavior_hash=hash_value).as_dict()


def test_the_frontend_identity_actually_changed():
    """The premise. If the hash still matched the old one, every refusal below
    would be vacuous."""
    assert aec_python_behavior_hash() != DEPLOYED_OLD_HASH


def test_the_migration_table_is_empty():
    assert ACCEPTED_BEHAVIOR_HASH_MIGRATIONS == {}, (
        "no pair may connect the old linear_error waveform to this frontend"
    )


def test_no_migration_targets_this_build():
    """Guards the specific mistake: retargeting the retired entries at the new
    hash instead of retiring them."""
    current = aec_python_behavior_hash()
    assert current not in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.values()
    assert current not in RETIRED_BEHAVIOR_HASHES


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
