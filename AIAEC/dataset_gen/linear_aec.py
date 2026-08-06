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
from typing import Dict, Optional, Tuple

import numpy as np
import torch


_AUDIO_ALG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_AEC_ROOT = os.path.join(_AUDIO_ALG_ROOT, "lib", "aec")
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


LINEAR_AEC_CONTRACT_VERSION = "aiaec-linear-error-v3"
# v2 uses get_formed_output(): the shadow/main-selected, crossfaded,
# WOLA-formed residual. It keeps RES/CNG disabled and is independent of any
# product-level output dynamics applied after the AEC chain.
# v1 data was materialized from a different seam. Old (v1) linear_error/ch5
# data must be regenerated, not mixed with later output in the same packed
# dataset.
# v2 -> v3: split provenance from compatibility. v2 compared the raw-text
# `aec_source_hash`, so reflowing a comment in lib/aec invalidated
# byte-identical shards AND refused already-trained checkpoints that no
# rematerialization could repair. v3 adds `aec_behavior_hash` (normalized AST,
# comment/format insensitive) as the compatibility condition and demotes
# `aec_commit`/`aec_source_hash` to provenance. The signal contract itself is
# unchanged, so v2 DATA is still valid -- but v2 contract dicts lack the new
# field and are rejected by from_dict; rerun the contract stamp, not the audio.

# The linear-AEC frontend is a frozen, versioned contract: its (frame_size,
# hop_size) per sample rate is a deliberate, pinned choice of this dataset,
# not meant to track whatever lib/aec's own production preset currently
# defaults to -- that default has already changed once (512/256 -> 256/128)
# and will again. Every caller that resolves a frame_size (the contract
# validator below, the config-driven path, and the factories' own fallback
# when a caller omits frame_size) reads this one map, so the frontend cannot
# silently drift with an unrelated upstream default change.
FROZEN_FRAME_HOP_BY_SR = {16000: (512, 256), 48000: (1024, 512)}


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
    # Retained in the v2 serialized contract for backward compatibility.
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
        """Fields that decide whether recorded data may be used with this build.

        Drops ONLY ``aec_commit``/``aec_source_hash`` -- raw-text provenance
        that changes on a comment reflow. ``aec_behavior_hash`` stays, and is
        the one field carrying the AEC's actual code identity: without it this
        dict is fully determined by the four values both call sites echo out of
        the recorded contract plus eleven ``__post_init__`` literals, i.e. a
        tautology. Do not remove it.
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
        # A v2 contract lacks both v3 compatibility fields. `behavior_hash_schema`
        # alone missing is NOT v2 -- it is a v3 dict written before the schema
        # field existed, which never left this repo (no stamped artifact has ever
        # carried a v3 contract), so it falls through to the generic error rather
        # than getting a migration path built for a state that cannot exist.
        if "aec_behavior_hash" in missing and not extra and set(missing) <= {
                "aec_behavior_hash", "behavior_hash_schema"}:
            # v2 contract. There is deliberately NO automatic migration: a v2
            # contract records only a raw-text source hash, and once lib/aec has
            # moved on there is no way to recover what the producing build's
            # BEHAVIOUR hash was, so stamping the current one would assert a
            # compatibility nobody verified. Shards can be re-stamped by
            # re-running the materializer; a v2 CHECKPOINT cannot be repaired at
            # all and must be retrained against a v3 corpus.
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
) -> AecConfig:
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
        "enable_shadow": True,
        "enable_delay_est": True,
        "enable_highpass": True,
        # v2 seam: selected/crossfaded WOLA output, not raw process() output.
        "return_formed_output": True,
    }
    if frame_size is not None:
        overrides["frame_size"] = int(frame_size)
    if filter_length is not None:
        overrides["filter_length"] = int(filter_length)
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


def require_linear_aec_contract(actual: Dict, expected: Dict, context: str) -> None:
    """Refuse a linear-AEC frontend that differs from the recorded one.

    The comparison MUST retain ``aec_behavior_hash``. Every other compared field
    is either pinned to a literal by ``__post_init__`` (engine, mode, the
    enable_* flags, output_seam, limiter, version) or echoed out of the recorded
    contract by both call sites, which build their ``runtime`` contract from
    ``contract.sample_rate/preset/frame_size/filter_length``. Drop the hash and
    the comparison becomes a tautology that cannot fail -- a changed PBFDKF step
    size, delay tolerance or formed-output crossfade would then feed inference a
    ``linear_error`` from a different filter than the one the checkpoint was
    trained on, silently. That regression shipped once (2026-08-06); the
    ``test_contract_comparison_is_not_vacuous`` test exists to stop it
    recurring.

    ``aec_commit``/``aec_source_hash`` are deliberately NOT compared here: they
    are raw-text provenance and would reject byte-identical data over a comment
    reflow. They remain in ``fingerprint()`` for resume/integrity.
    """
    got = LinearAecContract.from_dict(actual)
    want = LinearAecContract.from_dict(expected)
    got_dict = got.compatibility_dict()
    want_dict = want.compatibility_dict()
    if got_dict != want_dict:
        mismatches = [
            f"{key}: got {got_dict[key]!r}, expected {want_dict[key]!r}"
            for key in want_dict
            if got_dict[key] != want_dict[key]
        ]
        raise ValueError(f"{context} linear_aec contract mismatch: " + "; ".join(mismatches))


class LinearAecProcessor:
    """One stateful PBFDKF instance for one complete parent sequence."""

    def __init__(self, contract: LinearAecContract):
        self.contract = contract
        if contract.engine != "python_pbfdkf":
            raise ValueError(f"unsupported linear AEC engine {contract.engine!r}")
        cfg = make_linear_aec_config(
            contract.sample_rate,
            preset=contract.preset,
            frame_size=contract.frame_size,
            filter_length=contract.filter_length,
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

    @property
    def hop_size(self) -> int:
        return int(self.contract.hop_size)

    def process_numpy(
        self, microphone: np.ndarray, far_end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    """Materialize one complete sequence with a fresh PBFDKF state."""
    return LinearAecProcessor(contract).process(microphone, far_end)


__all__ = [
    "BEHAVIOR_HASH_SCHEMA",
    "LINEAR_AEC_CONTRACT_VERSION",
    "LinearAecContract",
    "LinearAecProcessor",
    "aec_python_behavior_hash",
    "aec_python_source_hash",
    "linear_aec_contract_from_config",
    "make_linear_aec_config",
    "make_linear_aec_contract",
    "materialize_linear_error",
    "require_linear_aec_contract",
]
