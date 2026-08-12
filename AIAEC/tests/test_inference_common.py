from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.inference_common import load_mic_far


def _write(path, audio, sample_rate):
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate,
             subtype="FLOAT")


def test_48k_pair_is_downsampled_to_16k_before_inference(tmp_path):
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mic = np.sin(2 * np.pi * 1000 * time)
    far = np.sin(2 * np.pi * 700 * time)
    _write(tmp_path / "mic.wav", mic, sample_rate)
    _write(tmp_path / "far.wav", far, sample_rate)

    mic_out, far_out, source_rates = load_mic_far(
        str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
    )

    assert source_rates == (48000, 48000)
    assert mic_out.shape == far_out.shape == (1, 16000)
    assert torch.isfinite(mic_out).all()
    assert torch.isfinite(far_out).all()


def test_downsampling_is_anti_aliased(tmp_path):
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    in_band = np.sin(2 * np.pi * 1000 * time)
    above_new_nyquist = np.sin(2 * np.pi * 12000 * time)
    _write(tmp_path / "mic.wav", in_band, sample_rate)
    _write(tmp_path / "far.wav", above_new_nyquist, sample_rate)

    mic_out, far_out, _ = load_mic_far(
        str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
    )

    # Ignore the boundary transient.  A stride-pick downsampler would alias
    # 12 kHz into the passband at essentially full level.
    interior = slice(256, -256)
    mic_rms = mic_out[0, interior].square().mean().sqrt()
    far_rms = far_out[0, interior].square().mean().sqrt()
    assert far_rms < mic_rms * 1e-3


def test_different_source_rates_are_rejected(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros(48000), 48000)
    _write(tmp_path / "far.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="sample rates differ"):
        load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )


def test_duration_mismatch_is_not_silently_truncated(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros(48000), 48000)
    _write(tmp_path / "far.wav", np.zeros(47700), 48000)
    with pytest.raises(ValueError, match="durations differ"):
        load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )


def test_stereo_input_is_rejected(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros((16000, 2)), 16000)
    _write(tmp_path / "far.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="mic must be mono"):
        load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )


@pytest.mark.parametrize("candidate", (
    "Align_CRUSE",
    "Align_ULCNet",
    "CAGCRN",
    "DeepVQE_S",
    "GTCRN_AENR",
    "DeepFilterNet_AENR",
))
def test_every_denoise_entry_uses_shared_resampling(candidate):
    source = (
        Path(__file__).parents[1] / candidate / "denoise.py"
    ).read_text(encoding="utf-8")
    assert "from AIAEC.inference_common import load_mic_far" in source
    assert "load_mic_far(" in source
    assert "must equal the" not in source
