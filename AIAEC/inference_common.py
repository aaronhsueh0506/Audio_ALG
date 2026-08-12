"""Shared audio input handling for AIAEC inference entry points."""

from __future__ import annotations

from typing import Tuple

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
    same duration to within one sample at ``model_sample_rate``.
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

    duration_error_samples = abs(
        mic.numel() * model_sample_rate / mic_sr
        - far.numel() * model_sample_rate / far_sr
    )
    if duration_error_samples > 1.0 + 1e-6:
        raise ValueError(
            "mic/far durations differ by "
            f"{duration_error_samples:.2f} samples at {model_sample_rate} Hz; "
            "AEC inputs must be time-aligned"
        )

    if mic_sr != model_sample_rate:
        mic = torchaudio.functional.resample(
            mic, mic_sr, model_sample_rate, **_RESAMPLE_KWARGS
        )
    if far_sr != model_sample_rate:
        far = torchaudio.functional.resample(
            far, far_sr, model_sample_rate, **_RESAMPLE_KWARGS
        )

    # Source files from the same clock can still differ by one input sample.
    # Trim only the corresponding permitted output rounding sample; larger
    # differences were rejected above.
    common_length = min(mic.numel(), far.numel())
    mic = mic[:common_length].unsqueeze(0).contiguous()
    far = far[:common_length].unsqueeze(0).contiguous()
    return mic, far, (mic_sr, far_sr)


__all__ = ["load_mic_far"]
