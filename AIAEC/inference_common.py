"""Shared audio input handling for AIAEC inference entry points."""

from __future__ import annotations

from typing import Tuple
import warnings

import soundfile as sf
import torch
import torchaudio
from torch import Tensor


# torchaudio's high-quality Kaiser-sinc preset.  Inference resampling is done
# once per file, so preserving the band edge is more important than throughput.
_RESAMPLE_KWARGS = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}

def _read_mono(path: str, label: str) -> Tuple[Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"{label} must be mono, got {audio.ndim} dimensions")
    if audio.size == 0:
        raise ValueError(f"{label} is empty: {path}")
    waveform = torch.from_numpy(audio).contiguous()
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{label} contains NaN or Inf: {path}")
    return waveform, int(sample_rate)


def _fit_reference_duration(
    primary: Tensor,
    primary_sample_rate: int,
    reference: Tensor,
    reference_sample_rate: int,
    primary_label: str,
) -> Tensor:
    """Pad/crop a time-aligned reference to the primary signal's duration."""
    target_length = int(round(
        primary.numel() * reference_sample_rate / primary_sample_rate
    ))
    difference = reference.numel() - target_length
    if difference == 0:
        return reference
    duration_error_ms = abs(difference) * 1000.0 / reference_sample_rate
    action = "cropping" if difference > 0 else "zero-padding"
    warnings.warn(
        f"{primary_label}/far tails differ by {duration_error_ms:.2f} ms; "
        f"{action} far to the {primary_label}'s duration",
        RuntimeWarning,
        stacklevel=3,
    )
    if difference > 0:
        return reference[:target_length]
    return torch.nn.functional.pad(reference, (0, -difference))


def _resample(waveform: Tensor, source_rate: int, model_rate: int) -> Tensor:
    if source_rate == model_rate:
        return waveform
    return torchaudio.functional.resample(
        waveform, source_rate, model_rate, **_RESAMPLE_KWARGS
    )


def _match_output_length(primary: Tensor, reference: Tensor) -> Tensor:
    """Resolve at most one rational-resampler rounding sample."""
    difference = reference.numel() - primary.numel()
    if abs(difference) > 1:
        raise RuntimeError(
            f"resampled primary/far lengths differ "
            f"({primary.numel()}/{reference.numel()})"
        )
    if difference > 0:
        return reference[:primary.numel()]
    if difference < 0:
        return torch.nn.functional.pad(reference, (0, -difference))
    return reference


def load_mic_far(
    mic_path: str,
    far_path: str,
    model_sample_rate: int,
) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
    """Load an aligned mono pair and convert both signals to the model rate.

    Resampling happens before every AEC/model operation.  This is load-bearing
    for RES+NR candidates: running PBFDKF at the capture rate and resampling
    only its residual would not reproduce the linear-AEC frontend used during
    training.

    Returns two ``[1, samples]`` CPU tensors plus the original mic/far rates.
    The synchronized source files must use the same rate and represent the
    same start time. The microphone owns the output timeline: a short far-end
    tail is zero-padded and a long one is cropped before resampling.
    """
    if model_sample_rate <= 0:
        raise ValueError(
            f"model_sample_rate must be positive, got {model_sample_rate}"
        )

    mic, mic_sr = _read_mono(mic_path, "mic")
    far, far_sr = _read_mono(far_path, "far")
    if mic_sr != far_sr:
        raise ValueError(
            f"mic/far sample rates differ ({mic_sr}/{far_sr}); provide an "
            "aligned pair from the same capture clock"
        )

    far = _fit_reference_duration(mic, mic_sr, far, far_sr, "mic")
    mic = _resample(mic, mic_sr, model_sample_rate)
    far = _resample(far, far_sr, model_sample_rate)
    far = _match_output_length(mic, far)
    mic = mic.unsqueeze(0).contiguous()
    far = far.unsqueeze(0).contiguous()
    return mic, far, (mic_sr, far_sr)


def load_linear_error_far(
    linear_error_path: str,
    far_path: str,
    model_sample_rate: int,
) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
    """Load a precomputed KF/AEC error and far reference for NN-only tests.

    Unlike :func:`load_mic_far`, the source rates may differ: published AEC
    demos commonly provide a 16 kHz KF residual beside a 48 kHz loopback. The
    error signal owns the output timeline and both tensors leave at the model
    rate. This helper deliberately does not claim that the external KF matches
    the frozen PBFDKF used to train this repository's checkpoints.
    """
    if model_sample_rate <= 0:
        raise ValueError(
            f"model_sample_rate must be positive, got {model_sample_rate}"
        )
    error, error_sr = _read_mono(linear_error_path, "linear error")
    far, far_sr = _read_mono(far_path, "far")
    far = _fit_reference_duration(
        error, error_sr, far, far_sr, "linear error"
    )
    error = _resample(error, error_sr, model_sample_rate)
    far = _resample(far, far_sr, model_sample_rate)
    far = _match_output_length(error, far)
    return (
        error.unsqueeze(0).contiguous(),
        far.unsqueeze(0).contiguous(),
        (error_sr, far_sr),
    )


__all__ = ["load_linear_error_far", "load_mic_far"]
