"""No migration connects recorded data to this frontend.

`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` is empty: the shadow-copy baseline retime,
the C restart evidence clearing and the analysis-seam / rendering-base split of
lib/aec move `linear_error`, so every earlier identity pair is retired. A
retired entry must not come back: retargeting one at a live build would declare
an old waveform compatible with a build that does not produce it. Retired
identities are refused with an instruction to rematerialize rather than a bare
hash mismatch.

The released inference-only checkpoint pointer is retired with them
(`RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY` is empty): the released
checkpoint is refused for inference and training alike, never re-pointed at
this build's hash.

Run:
    python3 -m pytest AIAEC/tests/test_linear_aec_behavior_migration.py
"""
from __future__ import annotations

import dataclasses

import pytest

from AIAEC.dataset_gen.aec_behavior_hash import aec_python_behavior_hash
from AIAEC.dataset_gen import linear_aec
from AIAEC.dataset_gen.linear_aec import (
    ACCEPTED_BEHAVIOR_HASH_MIGRATIONS,
    BEHAVIOR_HASH_SCHEMA,
    LINEAR_AEC_CONTRACT_VERSION,
    RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY,
    RELEASED_INFERENCE_POINTER_BUMPS,
    RETIRED_BEHAVIOR_HASHES,
    RETIRED_CONTRACT_VERSIONS,
    LinearAecContract,
    make_linear_aec_contract,
    require_inference_linear_aec_contract,
    require_linear_aec_contract,
)


# Two retired identities that must stay refused: the frontend the shipped
# corpus and the Align-ULCNet checkpoints were built on, and one verified
# equivalent to it. Spelled out rather than read from RETIRED_BEHAVIOR_HASHES:
# a test that takes its inputs from the table it checks would pass if the table
# were emptied.
DEPLOYED_OLD_HASH = (
    "abd8aa04272d8f833e948999fb54c5a9b2348d0f57f38efb21f43a15bbe6a239")
EARLIER_MIGRATED_HASH = (
    "eda3c3be25b4bb69762572b22447db7e004f870a395d7cf84ad1ce02ddd28cfe")

# The most recently retired pair: the frontend the shipped corpus was
# materialized under and the identity it was carried forward to. Spelled out
# for the same reason.
CORPUS_HASH = (
    "37ed5ad9b75ce42902361d8195fcf04a650b940744ec036a16c8736dec9d5061")
LAST_MIGRATED_TO_HASH = (
    "19dd4f90f482e15072d535964ac9816cdc21cae2c350b98de12a0e9ab561ff45")
RELEASED_CHECKPOINT_HASH = (
    "c1b1f5433fa244a2f7369938992bff95f00d1bdc394c4f21a5862f2dc547f786")
# The frontend (lib/aec 115e10c) before the analysis-seam / rendering-base
# split. The released checkpoint's pointer used to target it. Its output is NOT
# this build's, and it must not be registered as a byte-equivalent migration.
PREVIOUS_FRONTEND_HASH = (
    "9fa7a2dec85cef8f123cc4dd02e7eb3c5dc0c1b45d7ec0f32b766cf9cda245e5")
REFUSED_HASHES = (DEPLOYED_OLD_HASH, EARLIER_MIGRATED_HASH, CORPUS_HASH,
                  LAST_MIGRATED_TO_HASH, RELEASED_CHECKPOINT_HASH,
                  PREVIOUS_FRONTEND_HASH)


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


def test_the_previous_frontend_is_retired_and_not_migrated():
    assert PREVIOUS_FRONTEND_HASH in RETIRED_BEHAVIOR_HASHES
    assert PREVIOUS_FRONTEND_HASH not in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS
    assert PREVIOUS_FRONTEND_HASH not in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.values()
    assert aec_python_behavior_hash() != PREVIOUS_FRONTEND_HASH


def test_released_checkpoint_pointer_is_retired_not_repointed():
    """The released checkpoint (recorded at c1b1f5...) used to point at the
    previous frontend. It must not point at this build, nor at anything."""
    assert RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY == {}
    assert RELEASED_INFERENCE_POINTER_BUMPS == ()
    assert linear_aec._RELEASED_INFERENCE_SAMPLE_RATES == {}
    assert RELEASED_CHECKPOINT_HASH in RETIRED_BEHAVIOR_HASHES
    assert RELEASED_CHECKPOINT_HASH not in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS


@pytest.mark.parametrize("sample_rate, frame_size", ((16000, 512), (48000, 1024)))
@pytest.mark.parametrize(
    "recorded", (RELEASED_CHECKPOINT_HASH, PREVIOUS_FRONTEND_HASH))
def test_released_checkpoint_is_refused_for_inference_and_training(
        recorded, sample_rate, frame_size, recwarn):
    """Neither the inference gate nor the strict gate admits it, at either
    rate, and nothing warns that it was admitted."""
    current = make_linear_aec_contract(
        sample_rate, preset="balanced", frame_size=frame_size).as_dict()
    stale = dict(current, aec_behavior_hash=recorded)
    for gate in (require_inference_linear_aec_contract,
                 require_linear_aec_contract):
        with pytest.raises(ValueError, match="rematerialize_linear_aec") as caught:
            gate(current, stale, context="released checkpoint")
        assert recorded in str(caught.value)
    assert not [w for w in recwarn.list if "inference-only" in str(w.message)]


def test_released_checkpoint_is_refused_in_the_reverse_direction_too():
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    older_build = dict(current, aec_behavior_hash=RELEASED_CHECKPOINT_HASH)
    with pytest.raises(ValueError):
        require_inference_linear_aec_contract(
            older_build, current, context="reverse direction")


def test_the_empty_pointer_table_is_what_refuses_the_released_checkpoint(
        monkeypatch):
    """Can-fail proof for the retirement tests above. Re-pointing the released
    checkpoint at this build -- the silent fix this change forbids -- would make
    inference admit it with a warning; so the refusal is the table's emptiness,
    not an accident of the gate."""
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    stale = dict(current, aec_behavior_hash=RELEASED_CHECKPOINT_HASH)
    monkeypatch.setitem(
        RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY,
        RELEASED_CHECKPOINT_HASH, aec_python_behavior_hash())
    monkeypatch.setitem(
        linear_aec._RELEASED_INFERENCE_SAMPLE_RATES,
        RELEASED_CHECKPOINT_HASH, frozenset({16000}))
    with pytest.warns(RuntimeWarning, match="inference-only.*may differ"):
        require_inference_linear_aec_contract(
            current, stale, context="re-pointed")
    # ... and the training gate still refuses the re-pointed pair.
    with pytest.raises(ValueError, match="rematerialize_linear_aec"):
        require_linear_aec_contract(current, stale, context="re-pointed")


def test_a_new_hash_artifact_is_accepted_silently(recwarn):
    """The contract a corpus or checkpoint recorded under THIS build's
    frontend: accepted by both gates, without a migration warning."""
    current = make_linear_aec_contract(16000, preset="balanced").as_dict()
    # Provenance (commit, raw-text hash) may differ; behaviour is the same.
    recorded = dataclasses.replace(
        make_linear_aec_contract(16000, preset="balanced"),
        aec_commit="b" * 40, aec_source_hash="0" * 64).as_dict()
    assert recorded["aec_behavior_hash"] == aec_python_behavior_hash()
    require_linear_aec_contract(current, recorded, context="new corpus")
    require_inference_linear_aec_contract(
        current, recorded, context="new checkpoint")
    assert not recwarn.list


def test_the_current_contract_version_is_not_a_retired_one():
    assert "aiaec-linear-error-v3" in RETIRED_CONTRACT_VERSIONS
    assert LINEAR_AEC_CONTRACT_VERSION not in RETIRED_CONTRACT_VERSIONS
    assert LINEAR_AEC_CONTRACT_VERSION == "aiaec-linear-error-v4"


# Spelled out, not read from RETIRED_CONTRACT_VERSIONS: an emptied table would
# otherwise parametrize this to nothing and pass.
@pytest.mark.parametrize("retired", ("aiaec-linear-error-v3",))
def test_a_retired_contract_version_is_refused_with_what_to_do(retired):
    """A manifest, shard or checkpoint recorded under the previous contract
    version is refused on load -- before any hash comparison -- and the
    message says rematerialize, without --resume, repack, retrain."""
    recorded = dict(make_linear_aec_contract(16000, preset="balanced").as_dict(),
                    version=retired)
    with pytest.raises(ValueError) as caught:
        LinearAecContract.from_dict(recorded)
    message = str(caught.value)
    assert retired in message
    assert "rematerialize_linear_aec" in message
    assert "--resume" in message and "retrain" in message


def test_an_unknown_contract_version_is_still_a_plain_mismatch():
    recorded = dict(make_linear_aec_contract(16000, preset="balanced").as_dict(),
                    version="aiaec-linear-error-v0")
    with pytest.raises(ValueError, match="expected"):
        LinearAecContract.from_dict(recorded)


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
    no rematerialization required' warning."""
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
