"""Frozen Python-PBFDKF frontend used to materialize AIAEC ``linear_error``.

The adaptive filter is stateful.  Call :meth:`LinearAecProcessor.process`
once per complete parent sequence, then split its returned waveform into
training chunks.  Resetting or reconstructing the processor at a chunk
boundary changes the data contract and is deliberately not supported here.
"""

from __future__ import annotations

import dataclasses
import configparser
import hashlib
import json
import os
import subprocess
import sys
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch


_AUDIO_ALG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_AEC_ROOT = os.path.join(_AUDIO_ALG_ROOT, "lib", "aec")
if not os.path.isdir(os.path.join(_AEC_ROOT, "python")):
    # Deployment/training copies carry only AINR/, AIAEC/ and the standalone
    # AEC/ checkout side by side (no git tree, so no lib/aec submodule): fall
    # back to a sibling AEC/ beside this root, then to the SE-tree layout one
    # level up. In a full checkout lib/aec exists and nothing changes.
    for _candidate in (
        os.path.join(_AUDIO_ALG_ROOT, "AEC"),
        os.path.join(os.path.dirname(_AUDIO_ALG_ROOT), "AEC"),
    ):
        if os.path.isdir(os.path.join(_candidate, "python")):
            _AEC_ROOT = _candidate
            break
_AEC_PYTHON = os.path.join(_AEC_ROOT, "python")
if _AEC_PYTHON not in sys.path:
    sys.path.insert(0, _AEC_PYTHON)

from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402

# Behaviour/provenance hashing lives in a dependency-free sibling so the
# cross-CPython parity test can import it under interpreters without torch.
from .aec_behavior_hash import (  # noqa: E402
    BEHAVIOR_HASH_SCHEMA,
    aec_python_behavior_hash,
    aec_python_source_hash,
)


LINEAR_AEC_CONTRACT_VERSION = "aiaec-linear-error-v4"
# The signal is get_formed_output(): the shadow/main-selected, crossfaded,
# WOLA-formed residual, with RES/CNG disabled and independent of any
# product-level output dynamics applied after the AEC chain. It is the ANALYSIS
# seam of lib/aec: always the real selected linear residual, never replaced by
# the capture, and it pairs with the C `formed_hop` the board pipelines feed the
# network. On a hop where lib/aec's rendering guard fires, the residual can carry
# more energy than the capture; the guard protects the rendered audio, not this
# signal. Compatibility condition: `aec_behavior_hash` (normalized AST,
# comment/format insensitive); `aec_commit`/`aec_source_hash` are provenance only.

# Contract versions refused by name, so the refusal says what to do instead of
# reporting a bare version mismatch. There is no migration from any of them.
RETIRED_CONTRACT_VERSIONS: Dict[str, str] = {
    "aiaec-linear-error-v3":
        "its linear_error was written by a front end that predates the "
        "analysis-seam / rendering-base split of lib/aec, whose context-only "
        "formed seam could carry the capture instead of the selected residual",
}

# The linear-AEC frontend is a frozen, versioned contract: its (frame_size,
# hop_size) per sample rate is a pinned choice of this dataset, not meant to
# track lib/aec's own production preset default. Every caller that resolves a
# frame_size (the contract validator, the config-driven path and the factories'
# fallback) reads this one map.
FROZEN_FRAME_HOP_BY_SR = {16000: (512, 256), 48000: (1024, 512)}

# Matched-filter bank size every AIAEC corpus is, and stays, materialized
# with. It is NOT part of the contract dict: the contract records the SIGNAL
# agreement between the frontend and the data, and this value never varies
# across the dataset, so recording it would only add a field that can never
# differ. It lives here as the single literal the generation path resolves,
# and as the number a diagnostic override is reported against.
DATASET_DELAY_NUM_FILTERS = 5
# The bank size is a deployment/diagnostic knob, not a data-contract one:
# dataset generation always runs the value above (`materialize_linear_error`
# and the config-driven contract builder take no bank-size argument), while
# inference/diagnostics may pass `delay_num_filters=n` to `LinearAecEngine` or
# `LinearAecProcessor` without changing the recorded contract or its
# fingerprint. lib/aec sizes its rings and histograms at construction and has
# no runtime setter, so the override arrives at init.
DELAY_NUM_FILTERS_RANGE = (1, 5)

# Reliable bulk-delay reach of an n-filter matched bank, in ms: lib/aec's
# contract value ((n-1)*384 + 501 downsampled samples at 0.25 ms each), not a
# geometric span -- derived from that arithmetic (125/221/317/413/509 ms)
# rather than copied as literals, mirroring the C tests' KD_RELIABLE_SAMPLES.
# lib/aec resamples 48 kHz internally and rescales the reported delay, so the
# ms table holds at both supported rates. It lives here, beside the range it
# is defined over, because QA verdicts (not just display) depend on it.
MATCHED_REACH_MS = {
    n: ((n - 1) * 384 + 501) * 0.25
    for n in range(DELAY_NUM_FILTERS_RANGE[0], DELAY_NUM_FILTERS_RANGE[1] + 1)
}


def check_delay_num_filters(value: object) -> int:
    """Validate a matched-filter bank size, refusing anything out of range.

    Rejected rather than clamped, and ``0`` is not an "off" switch: the bank
    is what searches for the bulk delay at all, so a caller asking for a size
    the geometry does not have holds a wrong mental model of the search rather
    than a value to round into range.
    """
    low, high = DELAY_NUM_FILTERS_RANGE
    msg = (
        f"delay_num_filters must be an integer in [{low}, {high}], got "
        f"{value!r}"
    )
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(msg) from None
    # ``int(1.5)`` silently truncates. This is a public Python API as well as
    # an argparse validator, so direct callers must not accidentally deploy a
    # different bank from the value they supplied.
    try:
        exact = float(value) == resolved
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or not low <= resolved <= high:
        raise ValueError(msg)
    return resolved


def engine_delay_num_filters(engine: AEC) -> int:
    """The bank size a LIVE engine actually allocated.

    Counts the matched filters the estimator holds instead of echoing back a
    configured scalar: the point of reading this is to catch an override that
    was accepted and then not honoured, and a value read out of the same field
    that was written cannot report that.
    """
    estimator = getattr(getattr(engine, "delay_est", None), "_estimator", None)
    matched = getattr(estimator, "_matched_filter", None)
    filters = getattr(matched, "_filters", None)
    if filters is None:
        raise RuntimeError(
            "engine has no matched-filter bank to read back; the linear AEC "
            "frontend requires the MATCHED delay estimator"
        )
    return len(filters)


def _git_commit(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except Exception:
        pass
    return "unknown"


@dataclasses.dataclass(frozen=True)
class LinearAecContract:
    version: str
    engine: str
    preset: str
    sample_rate: int
    frame_size: int
    hop_size: int
    filter_length: int
    mode: str
    enable_res: bool
    enable_cng: bool
    enable_shadow: bool
    enable_delay_est: bool
    enable_highpass: bool
    output_seam: str
    # Pinned by ``__post_init__``.
    limiter: bool
    # Provenance: exactly which bytes produced this data. Not compared for
    # compatibility -- see aec_python_source_hash()/aec_python_behavior_hash().
    aec_commit: str
    aec_source_hash: str
    # Compatibility: the AEC's CODE identity, insensitive to comments/format.
    aec_behavior_hash: str
    # Which canonicalization rule produced aec_behavior_hash. Compared so that
    # changing the serializer reports itself by name instead of masquerading as
    # an AEC code change. Also folded into the digest, so the two agree.
    behavior_hash_schema: str = BEHAVIOR_HASH_SCHEMA

    def __post_init__(self) -> None:
        supported = FROZEN_FRAME_HOP_BY_SR
        if self.engine != "python_pbfdkf":
            raise ValueError(f"unsupported linear AEC engine {self.engine!r}")
        if self.preset != "balanced":
            raise ValueError(
                f"linear AEC preset must be 'balanced', got {self.preset!r}"
            )
        if self.sample_rate not in supported:
            raise ValueError(
                "linear AEC sample_rate must be 16000 or 48000, got "
                f"{self.sample_rate}"
            )
        expected_frame, expected_hop = supported[self.sample_rate]
        if (self.frame_size, self.hop_size) != (expected_frame, expected_hop):
            raise ValueError(
                f"linear AEC contract for sr={self.sample_rate} requires "
                f"frame/hop={expected_frame}/{expected_hop}, got "
                f"{self.frame_size}/{self.hop_size}"
            )
        expected_flags = {
            "mode": AecMode.PBFDKF.value,
            "enable_res": False,
            "enable_cng": False,
            "enable_shadow": True,
            "enable_delay_est": True,
            "enable_highpass": True,
            "output_seam": "formed_output",
            "limiter": False,
        }
        for name, expected in expected_flags.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(
                    f"linear AEC contract {name}={actual!r}, expected {expected!r}"
                )
        if self.filter_length <= 0:
            raise ValueError(
                f"linear AEC filter_length must be positive, got {self.filter_length}"
            )

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def compatibility_dict(self) -> Dict:
        """Fields that decide whether recorded data may be used with this
        build.

        Drops ONLY ``aec_commit``/``aec_source_hash`` (raw-text provenance that
        changes on a comment reflow). ``aec_behavior_hash`` stays: without the
        AEC's code identity the dict is fully determined by values both call
        sites echo from the recorded contract, i.e. a tautology.
        """
        value = self.as_dict()
        value.pop("aec_commit")
        value.pop("aec_source_hash")
        return value

    def fingerprint(self) -> str:
        """Integrity hash of the complete recorded contract and provenance.

        Includes provenance on purpose: resume/repack must notice that a
        different build produced a shard even when behaviour is unchanged.
        """
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, value: Dict) -> "LinearAecContract":
        if not isinstance(value, dict):
            raise ValueError("linear_aec contract must be a dict")
        expected = {field.name for field in dataclasses.fields(cls)}
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        # A v2 contract lacks both v3 compatibility fields.
        # `behavior_hash_schema` alone missing is not v2 and falls through to
        # the generic error.
        if "aec_behavior_hash" in missing and not extra and set(missing) <= {
                "aec_behavior_hash", "behavior_hash_schema"}:
            # v2 has deliberately NO automatic migration: it records only a
            # raw-text source hash, so the producing build's BEHAVIOUR hash is
            # unrecoverable and stamping the current one would assert a
            # compatibility nobody verified. Shards can be re-stamped by
            # re-running the materializer; a v2 checkpoint must be retrained.
            raise ValueError(
                "linear_aec contract is v2 (no 'aec_behavior_hash'). v2 -> v3 "
                "has no safe automatic migration: v2 records only a raw-text "
                "source hash, so the producing build's behaviour identity is "
                "unrecoverable. Re-stamp a dataset by re-running "
                "rematerialize_linear_aec.py; a v2 checkpoint must be retrained."
            )
        if missing or extra:
            raise ValueError(
                f"linear_aec contract fields differ: missing={missing}, extra={extra}"
            )
        contract = cls(**value)
        if contract.version in RETIRED_CONTRACT_VERSIONS:
            raise ValueError(
                f"linear_aec contract version={contract.version!r} is retired "
                f"(this build writes {LINEAR_AEC_CONTRACT_VERSION!r}): "
                f"{RETIRED_CONTRACT_VERSIONS[contract.version]}. There is no "
                "migration. Recompute the corpus's linear_error channel "
                "(AIAEC.dataset_gen.rematerialize_linear_aec, without "
                "--resume), repack, and retrain any checkpoint trained on the "
                "old data."
            )
        if contract.version != LINEAR_AEC_CONTRACT_VERSION:
            raise ValueError(
                f"linear_aec contract version={contract.version!r}, expected "
                f"{LINEAR_AEC_CONTRACT_VERSION!r}"
            )
        if contract.behavior_hash_schema != BEHAVIOR_HASH_SCHEMA:
            # Reported by name here so the far commoner failure -- an actual
            # lib/aec code change -- is not confused with a serializer change.
            # The schema is folded into the digest too, so the hashes would also
            # differ; this check exists only to make the reason legible.
            raise ValueError(
                "linear_aec behaviour hash was produced by schema "
                f"{contract.behavior_hash_schema!r}, this build canonicalizes "
                f"with {BEHAVIOR_HASH_SCHEMA!r}. The hashes are not comparable; "
                "re-stamp the dataset by re-running rematerialize_linear_aec.py."
            )
        return contract


def make_linear_aec_config(
    sample_rate: int,
    *,
    preset: str = "balanced",
    frame_size: Optional[int] = None,
    filter_length: Optional[int] = None,
    delay_num_filters: Optional[int] = None,
) -> AecConfig:
    """Build the frozen frontend's ``AecConfig``.

    ``delay_num_filters`` is the diagnostic/inference-only matched-filter bank
    size (see ``DATASET_DELAY_NUM_FILTERS``). ``None`` (dataset generation and
    every contract-driven call site) leaves the key out of the overrides, so
    the config is exactly the frozen one.
    """
    if AecPreset(preset) is not AecPreset.BALANCED:
        raise ValueError(f"linear AEC preset must be 'balanced', got {preset!r}")
    if frame_size is None:
        # Default to THIS contract's own frozen frame size, never to whatever
        # AecConfig.from_preset would otherwise pick on its own -- see
        # FROZEN_FRAME_HOP_BY_SR's docstring.
        pair = FROZEN_FRAME_HOP_BY_SR.get(int(sample_rate))
        if pair is not None:
            frame_size = pair[0]
    overrides = {
        "sample_rate": int(sample_rate),
        "mode": AecMode.PBFDKF,
        "enable_res": False,
        "enable_cng": False,
        # The AEC3 post chain has to run for AecState (usable verdict, ERLE,
        # error and echo PSDs) to advance on the real selected residual. The
        # output guard only picks the rendering base and never touches the
        # formed seam read below. This is also the mode the shipped C pipeline
        # uses (enable_res=0 with the context seam on), so the training signal
        # and the deployed one come off the same branch.
        "return_res_context": True,
        "enable_shadow": True,
        "enable_delay_est": True,
        "enable_highpass": True,
        # Seam: selected/crossfaded WOLA output, not raw process() output.
        "return_formed_output": True,
    }
    if frame_size is not None:
        overrides["frame_size"] = int(frame_size)
    if filter_length is not None:
        overrides["filter_length"] = int(filter_length)
    if delay_num_filters is not None:
        overrides["delay_num_filters"] = check_delay_num_filters(
            delay_num_filters
        )
    return AecConfig.from_preset(AecPreset(preset), **overrides)


def make_linear_aec_contract(
    sample_rate: int,
    *,
    preset: str = "balanced",
    frame_size: Optional[int] = None,
    filter_length: Optional[int] = None,
) -> LinearAecContract:
    cfg = make_linear_aec_config(
        sample_rate,
        preset=preset,
        frame_size=frame_size,
        filter_length=filter_length,
    )
    return LinearAecContract(
        version=LINEAR_AEC_CONTRACT_VERSION,
        engine="python_pbfdkf",
        preset=AecPreset(preset).value,
        sample_rate=cfg.sample_rate,
        frame_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        filter_length=cfg.filter_length,
        mode=cfg.mode.value,
        enable_res=cfg.enable_res,
        enable_cng=cfg.enable_cng,
        enable_shadow=cfg.enable_shadow,
        enable_delay_est=cfg.enable_delay_est,
        enable_highpass=cfg.enable_highpass,
        output_seam="formed_output",
        limiter=False,
        aec_commit=_git_commit(_AEC_ROOT),
        aec_source_hash=aec_python_source_hash(),
        aec_behavior_hash=aec_python_behavior_hash(),
    )


def linear_aec_contract_from_config(
    cfg: configparser.ConfigParser,
) -> LinearAecContract:
    sample_rate = cfg.getint("signal", "sr")
    n_fft = cfg.getint("signal", "n_fft")
    win_len = cfg.getint("signal", "win_len", fallback=n_fft)
    hop_len = cfg.getint("signal", "hop_len", fallback=win_len // 2)
    preset = cfg.get("linear_aec", "preset", fallback="balanced")
    frame_size = cfg.getint(
        "linear_aec", "frame_size",
        fallback=n_fft,
    )
    supported = FROZEN_FRAME_HOP_BY_SR
    if sample_rate not in supported:
        raise ValueError(
            f"linear AEC dataset supports sr=16000 or 48000, got {sample_rate}"
        )
    expected_frame, expected_hop = supported[sample_rate]
    if (frame_size, hop_len) != (expected_frame, expected_hop):
        raise ValueError(
            f"linear AEC grid for sr={sample_rate} must be "
            f"frame/hop={expected_frame}/{expected_hop}, got "
            f"{frame_size}/{hop_len}"
        )
    if frame_size != n_fft or win_len != n_fft or hop_len != frame_size // 2:
        raise ValueError(
            "dataset/PBFDKF grid mismatch: require "
            "linear_aec.frame_size == signal.n_fft == signal.win_len and "
            "signal.hop_len == frame_size/2, got "
            f"frame={frame_size}, n_fft={n_fft}, win={win_len}, hop={hop_len}"
        )
    # The delay profile is deliberately not configurable here. A config file
    # naming it would read as a supported way to materialize a corpus at
    # another bank size, and the resulting `linear_error` would differ
    # wherever the bulk delay leaves the smaller reach while every recorded
    # contract still compared EQUAL -- the shards would be indistinguishable
    # from full-bank ones. Refuse the key instead of ignoring it.
    unsupported = sorted(
        key for key in ("delay_mode", "delay_num_filters",
                        "fixed_delay_samples")
        if cfg.has_option("linear_aec", key)
    )
    if unsupported:
        raise ValueError(
            "linear_aec config option(s) " + ", ".join(unsupported) + " are "
            "not supported: dataset generation is frozen at the matched "
            f"{DATASET_DELAY_NUM_FILTERS}-filter bank and the delay profile "
            "is not part of the recorded contract. A different bank size is a "
            "deployment/diagnostic override (sweep_delay_depth.py "
            "--delay-num-filters), not a corpus setting."
        )
    configured_filter_length = cfg.getint(
        "linear_aec", "filter_length", fallback=-1
    )
    return make_linear_aec_contract(
        sample_rate,
        preset=preset,
        frame_size=frame_size,
        filter_length=(
            None if configured_filter_length < 0 else configured_filter_length
        ),
    )


# Recorded-hash -> current-hash pairs that are the SAME materialized signal.
# `aec_behavior_hash` is deliberately conservative (any expression, constant or
# control-flow edit under the signal scope changes it), so this table is the
# narrow escape for a checkpoint that cannot be rebuilt. A pair is admitted
# only if (1) `aec_behavior_hash` is the sole differing field, (2) the lib/aec
# change is inert on this path, (3) inertness is MEASURED byte-identical on the
# frozen frontend over a scene that reaches the changed code and (4) the same
# harness is shown able to fail. One hop, one direction, not transitive.
ACCEPTED_BEHAVIOR_HASH_MIGRATIONS: Dict[str, str] = {
    # Empty: no pair is admitted.
}

# Inference-only pointers: recorded-hash -> current-hash pairs for released
# checkpoints that LinearAecEngine may load with a "may differ" warning. Empty:
# the released 16-kHz Align-ULCNet checkpoint (recorded at c1b1f543...) used to
# point at 9fa7a2de..., a frontend that differed only in the TD-constraint
# window. The analysis/rendering split of lib/aec changes `linear_error` itself
# and retires 9fa7a2de..., so that checkpoint is retired and is NOT re-pointed
# at the current hash. A legacy front end is a build identity, not an entry of
# this build: running it means checking out the lib/aec whose behaviour hash is
# the recorded one (9fa7a2de... is lib/aec 115e10c). A new entry needs its own
# evidence and its own refusal and warning tests.
RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY: Dict[str, str] = {}

# Bump history of the table above, read by released_pointer_bump_guard.py: one
# row (date, old_value, new_value) per bump while an entry exists, each `old`
# copied from the previous row's `new`. Empty with the table; git has the rest.
RELEASED_INFERENCE_POINTER_BUMPS: Tuple[Tuple[str, str, str], ...] = ()

_RELEASED_INFERENCE_SAMPLE_RATES: Dict[str, frozenset] = {}

# A migration pair can be frontend-equivalent at one sample rate and
# intentionally different at another (e.g. a correction that is a no-op at 16
# kHz but changes the live 48-kHz detector window), so this is never inferred
# from the hash alone: such material must fail closed at that rate.
_BEHAVIOR_MIGRATION_SAMPLE_RATES: Dict[str, frozenset] = {
    # Keyed like ACCEPTED_BEHAVIOR_HASH_MIGRATIONS; empty while it is.
}


def _behavior_migration_applies(recorded: str, current: str,
                                sample_rate: int) -> bool:
    return (
        ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.get(recorded) == current
        and int(sample_rate) in _BEHAVIOR_MIGRATION_SAMPLE_RATES.get(
            recorded, frozenset())
    )

# Provenance of every build each accepted migration's SOURCE identity ran on. A
# legacy rematerialization ledger records only `fingerprint()`, which folds in
# `aec_commit` and `aec_source_hash` and is one-way, so a build computing the
# migration TARGET cannot re-derive the SOURCE fingerprint without this table.
# A behaviour hash is carried by a RANGE of revisions (docs-only commits keep
# it), so each maps to the `{"aec_commit": ..., "aec_source_hash": ...}` rows
# of every revision that carried it. An entry grants nothing on its own: it is
# consulted only after ACCEPTED_BEHAVIOR_HASH_MIGRATIONS admitted the pair.
MIGRATED_SOURCE_PROVENANCE: Dict[str, tuple] = {
    # Empty while no migration is admitted.
}


def migrated_ledger_fingerprints(contract: "LinearAecContract") -> tuple:
    """Fingerprints a migration SOURCE build would have written for ``contract``.

    The source differs from ``contract`` only in the three recorded provenance
    fields, so one candidate per recorded REVISION is reconstructible exactly.
    Yields ``(behaviour_hash, aec_commit, fingerprint)``; empty when nothing
    migrates to this build or the source has no recorded provenance.
    """
    out = []
    for recorded, current in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.items():
        if not _behavior_migration_applies(
                recorded, contract.aec_behavior_hash, contract.sample_rate):
            continue
        for revision in MIGRATED_SOURCE_PROVENANCE.get(recorded, ()):
            source = dataclasses.replace(
                contract,
                aec_behavior_hash=recorded,
                aec_commit=revision["aec_commit"],
                aec_source_hash=revision["aec_source_hash"],
            )
            out.append((recorded, revision["aec_commit"], source.fingerprint()))
    return tuple(out)


# Frontend identities that produced a DIFFERENT `linear_error` from the current
# build. Listed so the refusal can say what to do about it. Nothing may migrate
# from these: they are named to be rejected clearly, not to be accepted.
RETIRED_BEHAVIOR_HASHES = frozenset({
    "abd8aa04272d8f833e948999fb54c5a9b2348d0f57f38efb21f43a15bbe6a239",
    "eda3c3be25b4bb69762572b22447db7e004f870a395d7cf84ad1ce02ddd28cfe",
    "ffc2e044d031f06a9d685ee77159e5a8d7d3075e5e3eaea8156746c416c1dd4e",
    "8198bd0e29e4530f16a1ada6eafb86148e09ec749d10e84771bf2c86598143cd",
    "7c9249dee507f31837e6025c91ca855c05c59febcdee1cad5189cba7ce49cf88",
    # `linear_error` moved by the shadow-copy baseline retention retime and C's
    # restart evidence clearing.
    "37ed5ad9b75ce42902361d8195fcf04a650b940744ec036a16c8736dec9d5061",
    "19dd4f90f482e15072d535964ac9816cdc21cae2c350b98de12a0e9ab561ff45",
    # The frontend before the causal-half TD-constraint window; its corpus is
    # retired because the corrected window moves linear_error. The released
    # 16-kHz checkpoint recorded at this hash is retired too.
    "c1b1f5433fa244a2f7369938992bff95f00d1bdc394c4f21a5862f2dc547f786",
    # The frontend before the analysis/rendering split of lib/aec: `linear_error`
    # moves because the formed seam no longer carries the capture. Not a
    # migration source; corpora are rematerialized, checkpoints retrained.
    "9fa7a2dec85cef8f123cc4dd02e7eb3c5dc0c1b45d7ec0f32b766cf9cda245e5",
})


def require_linear_aec_contract(actual: Dict, expected: Dict, context: str) -> None:
    """Refuse a linear-AEC frontend that differs from the recorded one.

    The comparison MUST retain ``aec_behavior_hash``: every other field is
    pinned or echoed from the recorded contract, so without it a changed PBFDKF
    step size would silently feed inference a ``linear_error`` from another
    filter. ``aec_commit``/``aec_source_hash`` are NOT compared. The one
    exception is ``ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`` (warns).
    """
    got = LinearAecContract.from_dict(actual)
    want = LinearAecContract.from_dict(expected)
    got_dict = got.compatibility_dict()
    want_dict = want.compatibility_dict()
    if got_dict == want_dict:
        return
    differing = [key for key in want_dict if got_dict[key] != want_dict[key]]
    recorded_hash = want_dict["aec_behavior_hash"]
    current_hash = got_dict["aec_behavior_hash"]
    if (differing == ["aec_behavior_hash"]
            and _behavior_migration_applies(
                recorded_hash, current_hash, got.sample_rate)):
        warnings.warn(
            f"{context} linear_aec was materialized by AEC behaviour "
            f"{recorded_hash} and this build is {current_hash}; accepting via "
            "the verified frontend-equivalent migration table "
            "(ACCEPTED_BEHAVIOR_HASH_MIGRATIONS in dataset_gen/linear_aec.py), "
            "which records the byte-identical linear_error evidence for this "
            f"pair at {got.sample_rate} Hz. No rematerialization or retraining "
            "is required.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if differing == ["aec_behavior_hash"] and recorded_hash in RETIRED_BEHAVIOR_HASHES:
        raise ValueError(
            f"{context} linear_aec was materialized by AEC behaviour "
            f"{recorded_hash}, which this build ({current_hash}) does not "
            "reproduce. That identity is retired because it produces a "
            "different linear_error; there is no byte-equivalent migration "
            "for this pair and none may be added. Recompute the corpus's "
            "linear_error channel "
            "(AIAEC.dataset_gen.rematerialize_linear_aec, without --resume), "
            "repack, and retrain any checkpoint trained on the old data."
        )
    mismatches = [
        f"{key}: got {got_dict[key]!r}, expected {want_dict[key]!r}"
        for key in differing
    ]
    raise ValueError(f"{context} linear_aec contract mismatch: " + "; ".join(mismatches))


def require_inference_linear_aec_contract(
        actual: Dict, expected: Dict, context: str) -> None:
    """Inference-only compatibility for an explicitly released checkpoint.

    Unlike :func:`require_linear_aec_contract`, this may admit a known
    output-changing frontend transition listed in
    ``RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY``. That table is empty, so
    today it is the strict gate: a checkpoint recorded under any retired
    identity, the released pre-constraint one included, is refused. It is
    intentionally used only by the live ``LinearAecEngine``; dataset
    generation, packing, rematerialization and training never consult it.
    """
    got = LinearAecContract.from_dict(actual)
    want = LinearAecContract.from_dict(expected)
    got_dict = got.compatibility_dict()
    want_dict = want.compatibility_dict()
    differing = [key for key in want_dict if got_dict[key] != want_dict[key]]
    recorded_hash = want_dict["aec_behavior_hash"]
    current_hash = got_dict["aec_behavior_hash"]
    if (differing == ["aec_behavior_hash"]
            and RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY.get(
                recorded_hash) == current_hash
            and got.sample_rate in _RELEASED_INFERENCE_SAMPLE_RATES.get(
                recorded_hash, frozenset())):
        warnings.warn(
            f"{context} is loading a released {got.sample_rate}-Hz checkpoint "
            f"trained with AEC behaviour {recorded_hash} on this build "
            f"({current_hash}). This inference-only exception is not "
            "frontend-equivalent: linear_error and model output may differ. "
            "Old dataset material remains rejected and must be regenerated "
            "before training a new checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    require_linear_aec_contract(actual, expected, context)


class LinearAecProcessor:
    """One stateful PBFDKF instance for one complete parent sequence.

    ``delay_num_filters`` deploys a matched-filter bank other than the
    ``DATASET_DELAY_NUM_FILTERS`` one the corpus is materialized with. It is a
    DIAGNOSTIC/INFERENCE knob: it changes what the engine searches, never what
    ``contract`` says, so an overridden run must never be written back into a
    dataset (``materialize_linear_error`` takes no bank size at all).
    """

    def __init__(
        self,
        contract: LinearAecContract,
        *,
        delay_num_filters: Optional[int] = None,
    ):
        self.contract = contract
        if contract.engine != "python_pbfdkf":
            raise ValueError(f"unsupported linear AEC engine {contract.engine!r}")
        cfg = make_linear_aec_config(
            contract.sample_rate,
            preset=contract.preset,
            frame_size=contract.frame_size,
            filter_length=contract.filter_length,
            delay_num_filters=delay_num_filters,
        )
        runtime = make_linear_aec_contract(
            contract.sample_rate,
            preset=contract.preset,
            frame_size=contract.frame_size,
            filter_length=contract.filter_length,
        )
        if runtime != contract:
            require_linear_aec_contract(
                runtime.as_dict(), contract.as_dict(), "runtime"
            )
        self._engine = AEC(cfg)
        # Read the bank back off the constructed engine. The contract cannot
        # carry this check -- it does not record the bank size, by design --
        # so an override that is accepted and then dropped somewhere between
        # here and the estimator has to be caught against the live object.
        self.delay_num_filters = engine_delay_num_filters(self._engine)
        # `requested` is derived from the ARGUMENT, deliberately not from cfg:
        # the guard exists to catch the config layer swallowing the request,
        # so it must not trust that layer's output (its mutation test patches
        # make_linear_aec_config to drop the forwarding).
        requested = (
            DATASET_DELAY_NUM_FILTERS if delay_num_filters is None
            else check_delay_num_filters(delay_num_filters)
        )
        if self.delay_num_filters != requested:
            raise ValueError(
                f"linear AEC built a {self.delay_num_filters}-filter matched "
                f"bank, but {requested} was requested; the bank size is not "
                "reaching the estimator"
            )

    @property
    def hop_size(self) -> int:
        return int(self.contract.hop_size)

    def process_numpy(
        self,
        microphone: np.ndarray,
        far_end: np.ndarray,
        *,
        hop_tap: Optional[Callable[[AEC, int], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the hop loop; optionally observe the engine after each hop.

        ``hop_tap(engine, hop_index)`` is called after each hop has been
        processed and its formed output read. It exists so diagnostic taps
        (applied delay, confidence, the consumed far) run THIS loop instead of
        maintaining a second copy of it; ``None`` -- the materialization path
        -- changes nothing.
        """
        microphone = np.asarray(microphone, dtype=np.float32)
        far_end = np.asarray(far_end, dtype=np.float32)
        if microphone.ndim != 1 or microphone.shape != far_end.shape:
            raise ValueError(
                "microphone/far_end must be equal-length 1-D waveforms, got "
                f"{microphone.shape} and {far_end.shape}"
            )
        if microphone.size % self.hop_size:
            raise ValueError(
                f"sequence length {microphone.size} is not divisible by PBFDKF "
                f"hop {self.hop_size}; materialize before choosing chunk boundaries "
                "or pad the complete sequence explicitly"
            )

        error = np.empty_like(microphone, dtype=np.float32)
        for start in range(0, microphone.size, self.hop_size):
            stop = start + self.hop_size
            # The model contract consumes the selected/crossfaded WOLA seam,
            # not process()'s raw main-filter return.
            self._engine.process(
                np.ascontiguousarray(microphone[start:stop]),
                np.ascontiguousarray(far_end[start:stop]),
            )
            error[start:stop] = self._engine.get_formed_output()
            if hop_tap is not None:
                hop_tap(self._engine, start // self.hop_size)
        if not np.isfinite(error).all():
            raise ValueError("linear AEC produced non-finite samples")
        echo_estimate = microphone - error
        return error, echo_estimate

    def process(
        self, microphone: torch.Tensor, far_end: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if microphone.device.type != "cpu" or far_end.device.type != "cpu":
            raise ValueError("dataset materialization requires CPU tensors")
        error, echo = self.process_numpy(
            microphone.detach().numpy(), far_end.detach().numpy()
        )
        return (
            torch.from_numpy(error).to(dtype=microphone.dtype),
            torch.from_numpy(echo).to(dtype=microphone.dtype),
        )


def materialize_linear_error(
    microphone: torch.Tensor,
    far_end: torch.Tensor,
    contract: LinearAecContract,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize one complete sequence with a fresh PBFDKF state.

    Takes no bank size: corpus data is produced at
    ``DATASET_DELAY_NUM_FILTERS`` and nothing else, and the contract records
    no profile that could tell a differently-searched shard apart afterwards.
    """
    return LinearAecProcessor(contract).process(microphone, far_end)


__all__ = [
    "ACCEPTED_BEHAVIOR_HASH_MIGRATIONS",
    "RELEASED_INFERENCE_BEHAVIOR_HASH_COMPATIBILITY",
    "RELEASED_INFERENCE_POINTER_BUMPS",
    "MIGRATED_SOURCE_PROVENANCE",
    "RETIRED_BEHAVIOR_HASHES",
    "RETIRED_CONTRACT_VERSIONS",
    "BEHAVIOR_HASH_SCHEMA",
    "DATASET_DELAY_NUM_FILTERS",
    "DELAY_NUM_FILTERS_RANGE",
    "LINEAR_AEC_CONTRACT_VERSION",
    "LinearAecContract",
    "LinearAecProcessor",
    "aec_python_behavior_hash",
    "aec_python_source_hash",
    "check_delay_num_filters",
    "engine_delay_num_filters",
    "linear_aec_contract_from_config",
    "make_linear_aec_config",
    "make_linear_aec_contract",
    "materialize_linear_error",
    "migrated_ledger_fingerprints",
    "require_linear_aec_contract",
    "require_inference_linear_aec_contract",
]
