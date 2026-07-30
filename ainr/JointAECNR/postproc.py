"""Optional post-processing a caller MAY apply to JointAECNR's output.

POLICY
------
``JointOutputs.speech_spec`` is the DEFAULT FINAL OUTPUT.  Nothing in this file
is required, nothing here is applied by ``model.py``, and none of it is
enforced by the type system -- a caller that just plays ``speech_spec`` is
using the model correctly.

What this file exists for is the three things a classical chain used to provide
downstream of the AEC, which a joint model removes:

  * a safety attenuation limiter, because the joint model's implied gain is
    unbounded from below and a single bad frame can mute a talker;
  * comfort noise, because the classical CNG was driven by ``g_res`` and there
    is no ``g_res`` any more -- ``aux_noise_psd_head`` is its replacement;
  * a fallback to the classical chain, for the cases where the model is not
    trusted (cold start, NaN, a scenario it was not trained for).

⚠ THE ONE THING THAT MUST NOT BE CHAINED IS A SECOND FULL NOISE-SUPPRESSION
STAGE.  JointAECNR already did the noise suppression; running a classical NR
after it applies two independent gains to the same spectrum, and the audible
result -- pumping, musical noise, a hollowed-out talker -- looks exactly like a
bad model rather than a bad integration.  :class:`PostProcessChain` refuses it
at RUNTIME with a clear error.  It is deliberately not made impossible at
construction time: a caller with a genuine reason (an A/B against the cascade,
say) can pass ``allow_double_suppression=True`` and own the decision, and that
argument is greppable in a way a type-system workaround would not be.
"""

import dataclasses
import math
from typing import Callable, List, Optional

import torch


__all__ = [
    'DoubleSuppressionError',
    'PostProcessChain',
    'PostProcessStage',
    'apply_safety_attenuation',
    'classical_fallback_blend',
    'comfort_noise_from_log_psd',
]


class DoubleSuppressionError(RuntimeError):
    """A second full noise-suppression stage was chained after the model."""


# ============================================================
# Individual helpers
# ============================================================

def apply_safety_attenuation(speech_spec: torch.Tensor, mic_spec: torch.Tensor,
                             max_attenuation_db: float = 30.0,
                             max_gain_db: float = 0.0) -> torch.Tensor:
    """Clamp the gain the model implied, per bin and frame.

    The model's implied gain is ``|S_hat| / |Y|``.  This bounds it into
    ``[10^(-max_attenuation_db/20), 10^(max_gain_db/20)]`` while keeping the
    model's phase, which is what makes it a limiter rather than a second mask:
    it can only walk back an extreme decision, never make a new one.

    ⚠ ``max_attenuation_db`` is a NEAR-END PRESERVATION control, not an echo
    control.  Lowering it (less attenuation allowed) leaks more residual echo;
    raising it lets the model mute a quiet talker during double talk.  There is
    no setting that is good at both, which is why it is a caller's decision and
    not a model hyper-parameter.

    ``max_gain_db = 0`` forbids amplification outright.  Raise it only if the
    model is expected to restore level, which this one is not trained to do.
    """
    if max_attenuation_db < 0:
        raise ValueError(
            f"max_attenuation_db is a positive number of dB of allowed "
            f"attenuation, got {max_attenuation_db}")
    floor = 10.0 ** (-max_attenuation_db / 20.0)
    ceiling = 10.0 ** (max_gain_db / 20.0)

    mic_mag = mic_spec.abs()
    gain = speech_spec.abs() / mic_mag.clamp_min(1e-12)
    limited = gain.clamp(floor, ceiling)
    # Where the mic itself is silent there is no gain to speak of; leave the
    # model's output alone rather than inventing floor * 0.
    scale = torch.where(mic_mag > 1e-12,
                        limited / gain.clamp_min(1e-12),
                        torch.ones_like(gain))
    return speech_spec * scale


def comfort_noise_from_log_psd(speech_spec: torch.Tensor,
                               mic_spec: torch.Tensor,
                               noise_log_psd: torch.Tensor,
                               cng_level_db: float = -6.0,
                               attenuation_threshold_db: float = 6.0,
                               generator: Optional[torch.Generator] = None
                               ) -> torch.Tensor:
    """Refill bins the model emptied, at the level ``aux_noise_psd_head`` predicts.

    ``noise_log_psd`` is log10 of the local-noise power per bin and frame, i.e.
    exactly what the auxiliary head emits.  Comfort noise is added only where
    the model attenuated by more than ``attenuation_threshold_db``, at
    ``cng_level_db`` relative to the predicted noise floor, with random phase.

    ⚠ This is the replacement for the classical CNG, which was driven by
    ``g_res`` -- a quantity a joint model does not produce.  If
    ``aux_noise_psd_head`` is switched off there is no honest way to run this;
    guessing the floor from the model's own output is circular, because that
    output is where the floor was just removed.

    ⚠ Comfort noise is cosmetic.  It cannot be measured by any echo or speech
    metric and it will not improve one; it exists because a spectrum with holes
    punched in it sounds broken to a listener even when it scores well.
    """
    if noise_log_psd.shape != speech_spec.shape:
        raise ValueError(
            f"noise_log_psd {tuple(noise_log_psd.shape)} must match the "
            f"spectrum {tuple(speech_spec.shape)}")

    attenuation = speech_spec.abs() / mic_spec.abs().clamp_min(1e-12)
    emptied = attenuation < 10.0 ** (-attenuation_threshold_db / 20.0)

    magnitude = torch.sqrt(torch.pow(10.0, noise_log_psd).clamp_min(0.0))
    magnitude = magnitude * (10.0 ** (cng_level_db / 20.0))

    phase = torch.rand(speech_spec.shape, generator=generator,
                       device=speech_spec.device,
                       dtype=magnitude.dtype) * (2.0 * math.pi)
    noise = torch.polar(magnitude, phase).to(speech_spec.dtype)
    return speech_spec + noise * emptied


def classical_fallback_blend(joint_spec: torch.Tensor,
                             classical_spec: torch.Tensor,
                             joint_weight: torch.Tensor) -> torch.Tensor:
    """Cross-fade between the joint output and a classical chain's output.

    ``joint_weight`` is broadcast against the spectra and is 1 where the joint
    model is trusted, 0 where it is not.  A caller supplies whatever policy it
    wants there -- a ramp over the first seconds, a NaN guard, a scenario
    detector.

    ⚠ The classical chain's output is used as a REPLACEMENT, never as a second
    stage.  Blending is not chaining: at every bin exactly one of the two
    suppressors is in effect, so no bin gets two gains.  Feeding the joint
    output INTO a classical chain is the thing :class:`PostProcessChain`
    refuses.
    """
    if joint_spec.shape != classical_spec.shape:
        raise ValueError(
            f"joint {tuple(joint_spec.shape)} and classical "
            f"{tuple(classical_spec.shape)} outputs must have the same shape")
    weight = joint_weight.clamp(0.0, 1.0).to(joint_spec.real.dtype)
    return joint_spec * weight + classical_spec * (1.0 - weight)


# ============================================================
# Chain, with the runtime double-suppression check
# ============================================================

@dataclasses.dataclass(frozen=True)
class PostProcessStage:
    """One post-processing step.

    ``suppresses`` marks a stage that applies its own broadband spectral gain
    for noise reduction.  ⚠ Mark it honestly: the flag is the only thing
    standing between a caller and double suppression, and the failure it
    prevents is audible but not obviously an integration bug.
    """

    name: str
    fn: Callable[..., torch.Tensor]
    suppresses: bool = False


class PostProcessChain:
    """Apply post-processing stages in order, refusing double suppression.

    ``JointAECNR`` is itself stage zero and it IS a noise suppressor, so the
    first stage added with ``suppresses=True`` is already the second one in the
    signal path.  That is why it raises immediately rather than on the second.
    """

    def __init__(self, allow_double_suppression: bool = False):
        self.stages: List[PostProcessStage] = []
        self.allow_double_suppression = bool(allow_double_suppression)

    def add(self, name: str, fn: Callable[..., torch.Tensor],
            suppresses: bool = False) -> 'PostProcessChain':
        if suppresses and not self.allow_double_suppression:
            raise DoubleSuppressionError(
                f"stage {name!r} is a noise-suppression stage, but JointAECNR "
                f"has already suppressed noise -- chaining a second broadband "
                f"gain onto the same spectrum causes pumping and musical noise "
                f"that looks like a bad model, not a bad integration.  Use "
                f"classical_fallback_blend() to REPLACE the joint output where "
                f"it is not trusted, or pass "
                f"PostProcessChain(allow_double_suppression=True) and own the "
                f"decision."
            )
        self.stages.append(PostProcessStage(name, fn, suppresses))
        return self

    def __call__(self, speech_spec: torch.Tensor, **context) -> torch.Tensor:
        out = speech_spec
        for stage in self.stages:
            out = stage.fn(out, **context)
        return out

    def __repr__(self):
        names = ', '.join(stage.name for stage in self.stages) or '(none)'
        return f"PostProcessChain[{names}]"
