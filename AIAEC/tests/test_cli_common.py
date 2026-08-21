import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC._cli_common import resolve_directory_jobs
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
))
def test_every_inference_entry_uses_shared_resampling(candidate):
    """No candidate may grow its own WAV reader/resampler.

    The hop loop is whichever module actually defines that candidate's
    ``_streaming.main`` -- the shared one in ``_cli_common`` for the
    mic/far candidates, Align-ULCNet's own file for the PBFDKF path.  The
    assertion is on the function object, not on an import line's text, so a
    private copy fails here even if the shared name is still imported.
    """
    import sys

    import AIAEC.inference_common as shared

    streaming = importlib.import_module("AIAEC.%s._streaming" % candidate)
    hop_loop = sys.modules[streaming.main.__module__]
    assert hop_loop.load_mic_far is shared.load_mic_far


@pytest.mark.parametrize("candidate", ("Align_CRUSE", "CAGCRN", "DeepVQE_S"))
def test_shared_hop_loop_reproduces_the_offline_forward(candidate, tmp_path):
    """The generic CLIs share ONE hop loop, and it must still be the reference.

    Three near-identical schedules were folded into ``_cli_common``;
    nothing else drives that loop end to end.  This runs the real
    ``_streaming.main`` over real WAVs -- with only the checkpoint loader
    injected, so no checkpoint fixture is needed -- and compares the file it
    wrote against the whole-utterance forward.  That is the comparison
    ``--verify`` prints without gating, so a resampling, framing, flush or
    state-handoff mistake in the shared loop fails here instead of shipping.
    """
    from types import SimpleNamespace

    from AIAEC.aiaec_common import SignalGrid
    from AIAEC.Align_CRUSE import AlignCRUSE
    from AIAEC.CAGCRN import CAGCRN
    from AIAEC.DeepVQE_S import DeepVQES
    from AIAEC.dataset_gen import AecGrid, istft, stft

    grid = AecGrid(16000, 512, 512, 256)
    torch.manual_seed(5)
    model = {"Align_CRUSE": AlignCRUSE, "CAGCRN": CAGCRN,
             "DeepVQE_S": DeepVQES}[candidate](
        SignalGrid(16000, 512, 512, 256)).eval()

    # 96 frames, comfortably past every candidate's alignment depth: below it
    # the offline forward is deliberately NOT the streaming reference.
    length = 24576
    generator = torch.Generator().manual_seed(11)
    mic = 0.1 * torch.randn(1, length, generator=generator)
    far = 0.1 * torch.randn(1, length, generator=generator)
    _write(tmp_path / "mic.wav", mic[0].numpy(), grid.sr)
    _write(tmp_path / "far.wav", far[0].numpy(), grid.sr)

    out_wav = tmp_path / "out.wav"
    streaming = importlib.import_module("AIAEC.%s._streaming" % candidate)
    streaming.main(
        SimpleNamespace(
            checkpoint="unused", mic_wav=str(tmp_path / "mic.wav"),
            far_wav=str(tmp_path / "far.wav"), out_wav=str(out_wav),
            device="cpu", verify=False,
        ),
        load_model_fn=lambda checkpoint, device: (model, grid),
    )

    streamed, rate = sf.read(str(out_wav), dtype="float32")
    assert rate == grid.sr
    assert streamed.shape == (length,)
    with torch.no_grad():
        offline = model(microphone=stft(mic, grid).transpose(-2, -1),
                        far_end=stft(far, grid).transpose(-2, -1))
    reference = istft(
        offline.enhanced.transpose(-2, -1), grid, length=length
    )[0].numpy()
    # Measured max-abs on this fixture: <2e-7 (GEMM ordering noise only).
    assert np.abs(streamed - reference).max() < 1e-5


@pytest.mark.parametrize("candidate", (
    "Align_CRUSE",
    "Align_ULCNet",
    "CAGCRN",
    "DeepVQE_S",
))
def test_every_public_inference_entry_delegates_to_streaming(
        candidate, monkeypatch):
    inference = importlib.import_module("AIAEC.%s.inference" % candidate)
    streaming = importlib.import_module("AIAEC.%s._streaming" % candidate)
    seen = []
    monkeypatch.setattr(
        streaming, "main",
        lambda args, load_model_fn=None: seen.append((args, load_model_fn)),
    )
    # mic_dir=None is the single-pair form; directory mode is covered by the
    # resolve/pairing tests below.
    args = SimpleNamespace(candidate=candidate, mic_dir=None)
    inference.main(args)
    assert seen == [(args, inference.load_model)]


# ---------- directory mode ----------

def _job_args(tmp_path, ref_dir=None):
    return SimpleNamespace(
        mic_dir=str(tmp_path / "mic"),
        ref_dir=None if ref_dir is None else str(ref_dir),
        out_dir=str(tmp_path / "out"),
    )


def _tree(tmp_path, mic_names, ref_names=()):
    for sub_dir, names in (("mic", mic_names), ("ref", ref_names)):
        for name in names:
            path = tmp_path / sub_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, np.zeros(1600), 16000)
    return tmp_path / "ref"


def test_no_far_end_run_uses_digital_silence(tmp_path):
    """Omitting the reference must give a silent one of the mic's own shape --
    that is what turns the run into pure noise reduction."""
    _write(tmp_path / "mic.wav", np.sin(np.arange(16000) / 20.0), 16000)
    mic, far, rates = load_mic_far(str(tmp_path / "mic.wav"), None, 16000)
    assert mic.shape == far.shape == (1, 16000)
    assert torch.count_nonzero(far) == 0
    # No second file was read, so the far rate reported is the mic's own.
    assert rates == (16000, 16000)
    # Written so it can FAIL: an all-zero mic would satisfy the check above.
    assert torch.count_nonzero(mic) > 0


def test_a_missing_reference_is_an_error_not_a_silent_noise_reduction_run(
        tmp_path):
    """The failure mode this refuses: a mistyped --ref-dir silently turning an
    AEC evaluation into noise reduction, with nothing in the output to say so.
    """
    ref_dir = _tree(tmp_path, ["a69_mic.wav", "a70_mic.wav"],
                    ["a69_lpb.wav"])
    args = _job_args(tmp_path, ref_dir)
    with pytest.raises(ValueError, match="a70_mic.wav"):
        resolve_directory_jobs(args)


def test_a_reference_matching_no_microphone_is_also_an_error(tmp_path):
    """Reported from the far side too: an orphan there means the trees do not
    describe the same capture, which a per-file lookup would never notice."""
    ref_dir = _tree(tmp_path, ["a69_mic.wav"],
                    ["a69_lpb.wav", "a70_lpb.wav"])
    with pytest.raises(ValueError, match="a70_lpb.wav"):
        resolve_directory_jobs(_job_args(tmp_path, ref_dir))


def test_directory_jobs_resolve_before_the_first_file_runs(tmp_path):
    ref_dir = _tree(tmp_path, ["b/a70_mic.wav", "a69_mic.wav"],
                    ["b/a70_lpb.wav", "a69_lpb.wav"])
    jobs = resolve_directory_jobs(_job_args(tmp_path, ref_dir))
    assert [os.path.basename(mic) for mic, _ref, _out in jobs] == [
        "a69_mic.wav", "a70_mic.wav"], "jobs run in a stable sorted order"
    for mic, ref, out in jobs:
        assert os.path.basename(ref) == \
            os.path.basename(mic).replace("mic", "lpb")
        # The output tree mirrors the input tree and keeps the mic's name.
        assert os.path.relpath(out, str(tmp_path / "out")) == \
            os.path.relpath(mic, str(tmp_path / "mic"))


def test_without_ref_dir_every_job_carries_no_reference(tmp_path):
    _tree(tmp_path, ["a69_mic.wav", "not_named_by_the_rule.wav"])
    jobs = resolve_directory_jobs(_job_args(tmp_path))
    assert len(jobs) == 2
    assert all(ref is None for _mic, ref, _out in jobs), (
        "no --ref-dir means no pairing, so the naming rule must not apply"
    )


def test_writing_results_over_the_inputs_is_refused(tmp_path):
    _tree(tmp_path, ["a69_mic.wav"])
    args = _job_args(tmp_path)
    args.out_dir = args.mic_dir
    with pytest.raises(ValueError, match="must differ"):
        resolve_directory_jobs(args)


def test_an_empty_or_missing_mic_dir_is_refused(tmp_path):
    (tmp_path / "mic").mkdir()
    with pytest.raises(ValueError, match="no WAV files"):
        resolve_directory_jobs(_job_args(tmp_path))
    args = _job_args(tmp_path)
    args.mic_dir = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="does not exist"):
        resolve_directory_jobs(args)
