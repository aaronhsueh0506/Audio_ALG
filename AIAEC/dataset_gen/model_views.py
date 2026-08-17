"""One authoritative mapping from AEC stems to each candidate's task.

Residual-postfilter models require the same frozen linear AEC that will precede
them in production.  This module refuses to synthesize an oracle residual from
the clean echo stem: doing that produces unrealistically easy training data and
silently changes the front-end contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from .aec_features import AecGrid, AecStems, stft


@dataclass(frozen=True)
class ModelView:
    model_name: str
    task: str
    inputs: Dict[str, Tensor]
    target: Tensor
    echo_estimate: Optional[Tensor] = None


@dataclass(frozen=True)
class SpectralModelView:
    """Exact keyword inputs accepted by one candidate's ``forward`` method."""

    model_name: str
    task: str
    inputs: Dict[str, Tensor]
    target: Tensor
    feature_state: Optional[Dict[str, Any]] = None


MODEL_TASKS = {
    "Align_CRUSE": "end_to_end_aec_res_nr_dereverb",
    "Align_ULCNet": "linear_aec_postfilter_res_nr_dereverb",
    "DeepVQE_S": "end_to_end_aec_res_nr_dereverb",
    "CAGCRN": "end_to_end_aec_res_nr_dereverb",
}


def build_model_view(stems: AecStems, model_name: str,
                     sample_rate: Optional[int] = None) -> ModelView:
    """Create waveform inputs/target without duplicating task logic in trainers."""
    try:
        task = MODEL_TASKS[model_name]
    except KeyError:
        raise ValueError(
            f"unknown AIAEC model {model_name!r}; expected {sorted(MODEL_TASKS)}"
        ) from None

    if task == "end_to_end_aec_res_nr_dereverb":
        # Align-CRUSE previously ran under its own "direct_aec_preserve_noise"
        # task (echo cancellation only, noise left untouched for a later
        # independent NR stage) -- that route was retired in favour of this
        # one joint AEC+RES+NR+dereverberation task.
        return ModelView(
            model_name, task,
            {"microphone": stems.mic_postclip, "far_end": stems.far_render},
            stems.near_target,
        )

    linear_error = stems.linear_error
    echo_estimate = stems.D_hat
    if linear_error.shape != stems.mic_postclip.shape:
        raise ValueError("linear AEC error shape differs from microphone")
    if echo_estimate.shape != stems.mic_postclip.shape:
        raise ValueError("linear AEC echo-estimate shape differs from microphone")
    if not torch.isfinite(linear_error).all() or not torch.isfinite(echo_estimate).all():
        raise ValueError("materialized linear AEC stems contain non-finite samples")
    return ModelView(
        model_name, task,
        {"linear_error": linear_error, "far_end": stems.far_render},
        stems.near_target,
        echo_estimate=echo_estimate,
    )


def _as_batch(waveform: Tensor) -> Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(
            f"model view waveform must be [T] or [B,T], got {tuple(waveform.shape)}"
        )
    return waveform


def _standard_spectrum(waveform: Tensor, grid: AecGrid) -> Tensor:
    # Shared helper is [B,F,T]; every AIAEC public model boundary is [B,T,F].
    return stft(_as_batch(waveform), grid).transpose(-2, -1)


def build_spectral_model_view(view: ModelView,
                              grid: AecGrid) -> SpectralModelView:
    """Convert a waveform view into exact model ``forward(**inputs)`` tensors."""
    if grid.sr <= 0:
        raise ValueError("invalid AEC grid")
    target_wave = _as_batch(view.target)

    inputs = {
        name: _standard_spectrum(waveform, grid)
        for name, waveform in view.inputs.items()
    }
    return SpectralModelView(
        view.model_name, view.task,
        inputs,
        _standard_spectrum(target_wave, grid),
    )
