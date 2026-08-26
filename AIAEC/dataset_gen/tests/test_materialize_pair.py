"""materialize_pair is the loose-WAV diagnostic frontend.

The property that earns it a place beside rematerialize_linear_aec.py is that
it runs when a checkpoint-carrying path refuses: it builds its contract from
the runtime, so an installed engine whose behavior hash has moved is not a
disagreement it can have. ``test_runs_when_a_recorded_contract_would_refuse``
is that property, written so it fails if the tool ever starts carrying someone
else's contract.
"""

import dataclasses
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from AIAEC.dataset_gen.linear_aec import (
    LinearAecProcessor,
    make_linear_aec_contract,
)
from AIAEC.dataset_gen.materialize_pair import build_parser, materialize_pair
from AIAEC.dataset_gen.measure_align_residual import synth_echo_scene

SR = 16000
HOP = 256


def _write_pair(tmp_path, samples):
    mic, far = synth_echo_scene(samples, bulk_delay=0, seed=20260826)
    mic_p, far_p = tmp_path / "mic.wav", tmp_path / "far.wav"
    sf.write(mic_p, mic, SR, subtype="FLOAT")
    sf.write(far_p, far, SR, subtype="FLOAT")
    return str(mic_p), str(far_p)


def _run(tmp_path, samples, extra=()):
    """Returns (out_path, mic_path, far_path)."""
    mic_p, far_p = _write_pair(tmp_path, samples)
    out = str(tmp_path / "error.wav")
    materialize_pair(build_parser().parse_args([mic_p, far_p, out, *extra]))
    return out, mic_p, far_p


def test_output_keeps_the_microphone_length(tmp_path):
    """Padding to the hop boundary must not reach the caller."""
    out, _, _ = _run(tmp_path, 70001)
    error, rate = sf.read(out, dtype="float32")
    assert rate == SR
    assert error.shape == (70001,)
    assert np.isfinite(error).all()


def test_a_hop_aligned_length_is_not_padded(tmp_path):
    out, _, _ = _run(tmp_path, HOP * 40)
    error, _ = sf.read(out, dtype="float32")
    assert error.shape == (HOP * 40,)


def test_matches_the_materializer_it_wraps(tmp_path):
    """The CLI must add no signal processing of its own.

    Anything it did to the waveform beyond padding and trimming would show up
    here as a mismatch against the library call it exists to expose.
    """
    out, mic_p, far_p = _run(tmp_path, HOP * 30)
    written, _ = sf.read(out, dtype="float32")

    mic, _ = sf.read(mic_p, dtype="float32")
    far, _ = sf.read(far_p, dtype="float32")
    direct, _echo = LinearAecProcessor(make_linear_aec_contract(SR)).process(
        torch.from_numpy(mic), torch.from_numpy(far))
    assert np.array_equal(written, direct.numpy())


def test_echo_estimate_is_the_microphone_minus_the_error(tmp_path):
    echo_p = str(tmp_path / "echo.wav")
    out, mic_p, _ = _run(tmp_path, HOP * 20, ("--echo-estimate", echo_p))
    mic, _ = sf.read(mic_p, dtype="float32")
    error, _ = sf.read(out, dtype="float32")
    echo, _ = sf.read(echo_p, dtype="float32")
    assert np.allclose(echo, mic - error, atol=1e-6)


def test_refuses_to_overwrite_without_the_flag(tmp_path):
    mic_p, far_p = _write_pair(tmp_path, HOP * 10)
    out = tmp_path / "error.wav"
    out.write_bytes(b"not a wav")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        materialize_pair(build_parser().parse_args([mic_p, far_p, str(out)]))
    assert out.read_bytes() == b"not a wav"

    materialize_pair(build_parser().parse_args(
        [mic_p, far_p, str(out), "--overwrite"]))
    assert out.read_bytes() != b"not a wav"


def test_refuses_to_overwrite_the_echo_estimate_too(tmp_path):
    """The guard has to cover every output, not just the first one."""
    mic_p, far_p = _write_pair(tmp_path, HOP * 10)
    echo_p = tmp_path / "echo.wav"
    echo_p.write_bytes(b"not a wav")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        materialize_pair(build_parser().parse_args(
            [mic_p, far_p, str(tmp_path / "error.wav"),
             "--echo-estimate", str(echo_p)]))
    assert echo_p.read_bytes() == b"not a wav"


def test_runs_when_a_recorded_contract_would_refuse(tmp_path, capsys):
    """The reason this tool exists: no checkpoint contract to disagree with.

    A contract recorded before a frontend change carries the old behavior
    hash, and handing that to LinearAecProcessor is refused -- that guard is
    what the hash is for. This tool builds its contract from the runtime, so
    the same engine that refuses there succeeds here.
    """
    stale = dataclasses.replace(
        make_linear_aec_contract(SR), aec_behavior_hash="0" * 64)
    with pytest.raises(ValueError):
        LinearAecProcessor(stale)

    out, _, _ = _run(tmp_path, HOP * 20)
    assert sf.read(out, dtype="float32")[0].shape == (HOP * 20,)
    printed = capsys.readouterr().out
    assert make_linear_aec_contract(SR).aec_behavior_hash in printed


def test_reports_the_frontend_identity(tmp_path, capsys):
    """The operator has to be able to record WHICH frontend produced a WAV."""
    _run(tmp_path, HOP * 10)
    printed = capsys.readouterr().out
    contract = make_linear_aec_contract(SR)
    assert contract.aec_behavior_hash in printed
    assert contract.fingerprint() in printed
    assert "python_pbfdkf" in printed


def test_float32_precision_reaches_the_file(tmp_path):
    """float32 is the default and must arrive as real 32-bit float.

    soundfile's WAV default is PCM_16 whatever the input dtype, so writing
    this the obvious way quantises the signal the tool exists to preserve --
    see gen_aec_dataset.WAV_ENCODINGS for what that costs downstream.
    """
    out, _, _ = _run(tmp_path, HOP * 10)
    info = torchaudio.info(out)
    assert (info.encoding, info.bits_per_sample) == ("PCM_F", 32)


def test_int16_is_available_but_not_the_default(tmp_path):
    out, _, _ = _run(tmp_path, HOP * 10, ("--wav-encoding", "int16"))
    info = torchaudio.info(out)
    assert (info.encoding, info.bits_per_sample) == ("PCM_S", 16)


def test_a_build_that_ignores_the_encoding_is_refused(tmp_path, monkeypatch):
    """verify_wav_io must stop the run, and stop it BEFORE the filter runs."""
    from AIAEC.dataset_gen import materialize_pair as mp

    real_save = torchaudio.save

    def downgrading_save(path, waveform, rate, **kwargs):
        return real_save(path, waveform, rate,
                         encoding="PCM_S", bits_per_sample=16)

    monkeypatch.setattr(mp.torchaudio, "save", downgrading_save)
    monkeypatch.setattr(
        mp, "materialize_linear_error",
        lambda *a, **k: pytest.fail("filter ran before the encoding check"))
    with pytest.raises(RuntimeError, match="ignoring the requested encoding"):
        _run(tmp_path, HOP * 10)


def test_the_frozen_grid_is_not_exposed_as_a_knob(tmp_path):
    """preset and frame size are frozen per rate; offering them as flags would
    only hand the caller a raw traceback from the contract's own validation."""
    flags = {action.dest for action in build_parser()._actions}
    assert "preset" not in flags
    assert "frame_size" not in flags
    assert "filter_length" in flags


def test_module_runs_as_a_script(tmp_path):
    """`python3 -m AIAEC.dataset_gen.materialize_pair` is the documented form."""
    mic_p, far_p = _write_pair(tmp_path, HOP * 10)
    out = tmp_path / "error.wav"
    result = subprocess.run(
        [sys.executable, "-m", "AIAEC.dataset_gen.materialize_pair",
         mic_p, far_p, str(out)],
        capture_output=True, text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[3]))
    assert result.returncode == 0, result.stderr
    assert out.exists()
