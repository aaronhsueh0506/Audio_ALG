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
    "Align_CRUSE": "direct_aec_preserve_noise",
    "Align_ULCNet": "linear_aec_postfilter_res_nr",
    "GTCRN_AENR": "linear_aec_postfilter_res_nr",
    "DeepFilterNet_AENR": "linear_aec_postfilter_res_nr",
    "DeepVQE_S": "end_to_end_aec_res_nr",
    "CAGCRN": "end_to_end_aec_res_nr",
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

    if task == "direct_aec_preserve_noise":
        # only AEC/RES: background noise is explicitly desired signal.
        return ModelView(
            model_name, task,
            {"microphone": stems.mic_postclip, "far_end": stems.far_render},
            stems.near_speech + stems.local_noise,
        )

    if task == "end_to_end_aec_res_nr":
        # DeepVQE's published task includes dereverberation; CAGCRN's does not.
        # Keeping both targets in the corpus prevents a silent task change.
        target = (stems.near_target if model_name == "DeepVQE_S"
                  else stems.near_speech)
        return ModelView(
            model_name, task,
            {"microphone": stems.mic_postclip, "far_end": stems.far_render},
            target,
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
        stems.near_speech,
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


def build_spectral_model_view(
        view: ModelView, grid: AecGrid, *, dfn_model=None,
        dfn_feature_config: Optional[Dict[str, Any]] = None,
        dfn_feature_state: Optional[Dict[str, Any]] = None,
        ) -> SpectralModelView:
    """Convert a waveform view into exact model ``forward(**inputs)`` tensors.

    DeepFilterNet is intentionally special: its feature path uses normalized
    STFT coefficients and two independent causal EMA states for linear-error
    and far-end features.  Reusing one state is rejected by construction.
    """
    if grid.sr <= 0:
        raise ValueError("invalid AEC grid")
    target_wave = _as_batch(view.target)

    if view.model_name != "DeepFilterNet_AENR":
        inputs = {
            name: _standard_spectrum(waveform, grid)
            for name, waveform in view.inputs.items()
        }
        return SpectralModelView(
            view.model_name, view.task, inputs,
            _standard_spectrum(target_wave, grid),
        )

    if dfn_model is None or dfn_feature_config is None:
        raise ValueError(
            "DeepFilterNet_AENR spectral views require dfn_model and the "
            "exact read_feature_config() result used by that checkpoint"
        )
    if set(view.inputs) != {"linear_error", "far_end"}:
        raise ValueError("DeepFilterNet_AENR waveform view contract is corrupt")
    if dfn_feature_state is not None:
        if set(dfn_feature_state) != {"error", "far"}:
            raise ValueError("DFN state must contain independent 'error' and 'far' states")
        if dfn_feature_state["error"] is dfn_feature_state["far"]:
            raise ValueError("error and far DFN EMA states must not be shared")

    def normalized_stft(waveform: Tensor) -> Tensor:
        waveform = _as_batch(waveform)
        return torch.stft(
            waveform, grid.n_fft, grid.hop_len, grid.win_len,
            window=grid.window(device=waveform.device, dtype=waveform.dtype),
            normalized=True, return_complex=True,
        )

    # Import locally so non-DFN dataset consumers do not acquire trainer
    # dependencies or configuration side effects.
    from AINR.DeepFilterNet2.train import extract_dfn2_features

    error_spec = normalized_stft(view.inputs["linear_error"])
    far_spec = normalized_stft(view.inputs["far_end"])
    error_state = None if dfn_feature_state is None else dfn_feature_state["error"]
    far_state = None if dfn_feature_state is None else dfn_feature_state["far"]
    _, error_erb, error_feat, next_error = extract_dfn2_features(
        error_spec, dfn_model.erb_fb, dfn_model.df_bins,
        feature_cfg=dfn_feature_config, ema_state=error_state,
    )
    _, far_erb, far_feat, next_far = extract_dfn2_features(
        far_spec, dfn_model.erb_fb, dfn_model.df_bins,
        feature_cfg=dfn_feature_config, ema_state=far_state,
    )
    return SpectralModelView(
        view.model_name, view.task,
        {
            "linear_error": error_spec.transpose(1, 2),
            "error_erb": error_erb,
            "error_spec": error_feat,
            "far_erb": far_erb,
            "far_spec": far_feat,
        },
        normalized_stft(target_wave).transpose(1, 2),
        feature_state={"error": next_error, "far": next_far},
    )
