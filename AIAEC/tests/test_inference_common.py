from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.inference_common import load_linear_error_far, load_mic_far


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


def test_short_far_tail_is_zero_padded_to_preserve_mic_timeline(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros(48000), 48000)
    _write(tmp_path / "far.wav", np.zeros(47520), 48000)  # 10 ms shorter
    with pytest.warns(RuntimeWarning, match="zero-padding far"):
        mic, far, _ = load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )
    assert mic.shape == far.shape == (1, 16000)
    assert torch.count_nonzero(far[:, -160:]) == 0


def test_official_demo_sized_short_far_tail_is_supported(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros(48000), 48000)
    _write(tmp_path / "far.wav", np.zeros(19200), 48000)  # 600 ms shorter
    with pytest.warns(RuntimeWarning, match="600.00 ms; zero-padding far"):
        mic, far, _ = load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )
    assert mic.shape == far.shape == (1, 16000)
    assert torch.count_nonzero(far[:, 6400:]) == 0


@pytest.mark.parametrize(
    ("tail_samples", "tail_ms"),
    ((11520, "240.00"), (115200, "2400.00")),
)
def test_reported_align_ulcnet_tail_mismatches_are_supported(
    tmp_path, tail_samples, tail_ms
):
    # 240 ms at 48 kHz is 11,520 samples; keep the reported 115,200-sample
    # variant as well so either form of the external demo export is covered.
    mic_length = 144000
    _write(tmp_path / "mic.wav", np.zeros(mic_length), 48000)
    _write(
        tmp_path / "far.wav",
        np.zeros(mic_length - tail_samples),
        48000,
    )
    with pytest.warns(
        RuntimeWarning, match=rf"{tail_ms} ms; zero-padding far"
    ):
        mic, far, _ = load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )
    assert mic.shape == far.shape == (1, 48000)


def test_long_far_tail_is_cropped_to_mic_timeline(tmp_path):
    _write(tmp_path / "mic.wav", np.zeros(48000), 48000)
    _write(tmp_path / "far.wav", np.zeros(48480), 48000)
    with pytest.warns(RuntimeWarning, match="cropping far"):
        mic, far, _ = load_mic_far(
            str(tmp_path / "mic.wav"), str(tmp_path / "far.wav"), 16000
        )
    assert mic.shape == far.shape == (1, 16000)


def test_published_16k_error_and_48k_far_can_bypass_linear_aec(tmp_path):
    _write(tmp_path / "error.wav", np.zeros(16000), 16000)
    _write(tmp_path / "far.wav", np.zeros(48000), 48000)
    error, far, rates = load_linear_error_far(
        str(tmp_path / "error.wav"), str(tmp_path / "far.wav"), 16000
    )
    assert rates == (16000, 48000)
    assert error.shape == far.shape == (1, 16000)


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
