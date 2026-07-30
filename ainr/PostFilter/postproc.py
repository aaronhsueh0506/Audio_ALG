"""Caller-side bounds on the PostFilter's gain.

WHY THIS IS A SEPARATE FILE
---------------------------
The network emits its suppression decision and nothing else.  The preset floor
and the attenuation cap are product settings: they are what lets one trained
checkpoint ship as a gentle preset in one product and an aggressive one in
another.  Folding them into ``forward`` would make every preset a retrain, and
would also mean the loss was computed on a floored gain, so the model would
learn to fight its own floor.

⚠ Nothing here is differentiable-by-design or used during training.  If one of
these functions ever appears inside the training loop, the separation above has
been lost.

All dB values act on a GAIN, so they convert with /20, not /10.  (The classical
NR shipped a /10 bug here once; the presets are -20/-25/-30/-40 dB of gain.)
"""

import math

import torch


__all__ = [
    'GainPostProcessor',
    'apply_attenuation_cap',
    'apply_gain_floor',
    'db_to_gain',
    'smooth_gain',
]


def db_to_gain(db: float) -> float:
    """dB on a gain -> linear.  ⚠ /20, not /10."""
    return float(10.0 ** (db / 20.0))


def apply_gain_floor(gain: torch.Tensor, floor_db) -> torch.Tensor:
    """Clamp the gain from below at ``floor_db`` (the tuning preset's g_min).

    ``None`` or ``-inf`` disables it.  Accepts a real gain or a complex mask; a
    complex mask is rescaled so its MAGNITUDE meets the floor and its phase is
    untouched -- flooring the real and imaginary parts separately would rotate
    the mask, which is a different operation wearing the same name.
    """
    if floor_db is None or not math.isfinite(float(floor_db)):
        return gain
    return _floor_magnitude(gain, db_to_gain(float(floor_db)))


def apply_attenuation_cap(gain: torch.Tensor, cap_db) -> torch.Tensor:
    """Refuse to attenuate by more than ``cap_db`` dB.

    Numerically this is the same clamp as :func:`apply_gain_floor` with
    ``floor_db = -cap_db``.  Both exist because they are configured from
    different places -- the floor comes from the tuning preset table, the cap
    from a product decision about how much of the near end may be lost -- and a
    caller that sets both gets the LESS aggressive of the two, which is what
    :class:`GainPostProcessor` does.  ``0`` or ``None`` disables it.
    """
    if cap_db is None or float(cap_db) <= 0.0 or not math.isfinite(float(cap_db)):
        return gain
    return _floor_magnitude(gain, db_to_gain(-float(cap_db)))


def _floor_magnitude(gain: torch.Tensor, floor: float) -> torch.Tensor:
    if torch.is_complex(gain):
        magnitude = gain.abs()
        scale = torch.clamp(magnitude, min=floor) / magnitude.clamp_min(1e-12)
        return gain * scale.to(gain.dtype)
    return torch.clamp(gain, min=floor)


def smooth_gain(gain: torch.Tensor, previous, alpha_attack: float,
                alpha_release: float):
    """One-pole gain smoothing along the LAST axis (time).

    ``gain``: ``(..., T)``.  ``previous``: ``(...)`` carried from the previous
    block, or ``None`` to start from the first frame.  Returns
    ``(smoothed, last_value)``.

    "Attack" is the direction in which the gain FALLS, i.e. suppression coming
    on.  ⚠ Slowing the attack trades echo leakage at onsets for fewer
    gain-modulation artefacts.  It cannot improve both, and no value of it fixes
    a model that decided wrongly.
    """
    if alpha_attack <= 0.0 and alpha_release <= 0.0:
        last = gain[..., -1] if gain.shape[-1] else previous
        return gain, last
    magnitude = gain.abs() if torch.is_complex(gain) else gain
    state = magnitude[..., 0] if previous is None else previous
    smoothed = []
    for t in range(magnitude.shape[-1]):
        target = magnitude[..., t]
        alpha = torch.where(target < state,
                            torch.full_like(target, alpha_attack),
                            torch.full_like(target, alpha_release))
        state = alpha * state + (1.0 - alpha) * target
        smoothed.append(state)
    smoothed = torch.stack(smoothed, dim=-1)
    if torch.is_complex(gain):
        scale = smoothed / magnitude.clamp_min(1e-12)
        return gain * scale.to(gain.dtype), state
    return smoothed, state


class GainPostProcessor:
    """The caller's policy: floor, cap and smoothing, in that order.

    Holds the one-pole state so a streaming caller can drive it block by block.
    Built from the config's ``[inference]`` section by :meth:`from_config`, so
    denoise.py and a C port read the same numbers from the same place.
    """

    def __init__(self, gain_floor_db=None, attenuation_cap_db=None,
                 attack_tau_sec=0.0, release_tau_sec=0.0,
                 hop_len=None, sr=None):
        self.gain_floor_db = gain_floor_db
        self.attenuation_cap_db = attenuation_cap_db
        self.attack_tau_sec = float(attack_tau_sec)
        self.release_tau_sec = float(release_tau_sec)
        if (self.attack_tau_sec > 0.0 or self.release_tau_sec > 0.0):
            if hop_len is None or sr is None:
                raise ValueError(
                    "smoothing needs hop_len and sr: the time constants are in "
                    "SECONDS and a frame-count alpha would silently change "
                    "meaning at 48 kHz")
            # Imported lazily so postproc stays usable without the dataset layer
            # on a machine that only runs inference.
            from dataset_gen.aec import alpha_from_tau
            self.alpha_attack = alpha_from_tau(self.attack_tau_sec, hop_len, sr)
            self.alpha_release = alpha_from_tau(self.release_tau_sec, hop_len, sr)
        else:
            self.alpha_attack = self.alpha_release = 0.0
        self._state = None

    @classmethod
    def from_config(cls, cfg, grid=None):
        section = 'inference'
        if not cfg.has_section(section):
            return cls()
        hop_len = grid.hop_len if grid is not None else None
        sr = grid.sr if grid is not None else None
        return cls(
            gain_floor_db=cfg.getfloat(section, 'gain_floor_db', fallback=None),
            attenuation_cap_db=cfg.getfloat(section, 'attenuation_cap_db',
                                            fallback=None),
            attack_tau_sec=cfg.getfloat(section, 'attack_tau_sec', fallback=0.0),
            release_tau_sec=cfg.getfloat(section, 'release_tau_sec', fallback=0.0),
            hop_len=hop_len, sr=sr,
        )

    def reset(self):
        self._state = None

    def __call__(self, gain: torch.Tensor) -> torch.Tensor:
        gain = apply_gain_floor(gain, self.gain_floor_db)
        gain = apply_attenuation_cap(gain, self.attenuation_cap_db)
        gain, self._state = smooth_gain(gain, self._state,
                                        self.alpha_attack, self.alpha_release)
        # ⚠ Smoothing can only raise a floored gain (it is a convex combination
        # of values that already meet the floor), so the floor survives it and
        # the order above is safe.  Reversing it would not be.
        return gain

    def describe(self) -> str:
        floor = ('none' if self.gain_floor_db is None
                 else f"{self.gain_floor_db:g} dB")
        cap = ('none' if not self.attenuation_cap_db
               else f"{self.attenuation_cap_db:g} dB")
        return (f"gain floor={floor}, attenuation cap={cap}, "
                f"attack={self.attack_tau_sec:g}s, release={self.release_tau_sec:g}s")
