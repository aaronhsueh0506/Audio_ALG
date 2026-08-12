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

# Capture/export pipelines commonly leave one frame of tail in only one file.
# Trimming that tail preserves the shared start time and is safe for offline
# AEC.  A larger mismatch is more likely to mean the wrong pair was supplied.
_MAX_TAIL_MISMATCH_SECONDS = 0.100


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
    same start time. A tail mismatch of at most 100 ms is trimmed before
    resampling; larger differences are rejected as a likely wrong pair.
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

    source_length_difference = abs(mic.numel() - far.numel())
    duration_error_seconds = source_length_difference / mic_sr
    if duration_error_seconds > _MAX_TAIL_MISMATCH_SECONDS + 1e-12:
        raise ValueError(
            f"mic/far tails differ by {duration_error_seconds * 1000:.2f} ms "
            f"({source_length_difference} samples at {mic_sr} Hz), exceeding "
            f"the {_MAX_TAIL_MISMATCH_SECONDS * 1000:.0f} ms safety limit; "
            "verify that both files belong to the same aligned capture"
        )
    if source_length_difference:
        common_source_length = min(mic.numel(), far.numel())
        warnings.warn(
            f"mic/far tails differ by {duration_error_seconds * 1000:.2f} ms; "
            f"truncating both to {common_source_length} source samples",
            RuntimeWarning,
            stacklevel=2,
        )
        mic = mic[:common_source_length]
        far = far[:common_source_length]

    if mic_sr != model_sample_rate:
        mic = torchaudio.functional.resample(
            mic, mic_sr, model_sample_rate, **_RESAMPLE_KWARGS
        )
    if far_sr != model_sample_rate:
        far = torchaudio.functional.resample(
            far, far_sr, model_sample_rate, **_RESAMPLE_KWARGS
        )

    # Both resamplers saw the same source length. Keep a final min() for the
    # backend's rational output-length rounding contract.
    common_length = min(mic.numel(), far.numel())
    mic = mic[:common_length].unsqueeze(0).contiguous()
    far = far[:common_length].unsqueeze(0).contiguous()
    return mic, far, (mic_sr, far_sr)


__all__ = ["load_mic_far"]
