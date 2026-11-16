"""float_io_compare.py 的失敗閘門：有設限值時，無法比較的兩次執行（長度不同、空檔、短於一個 hop 的 head 傾印）不得通過。"""

import array
import pathlib
import struct
import subprocess
import sys
import wave

import pytest

_TOOL = pathlib.Path(__file__).resolve().parents[2] / 'docs' / 'maintenance' / 'float_io_compare.py'


def _wav(path, samples):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack('<%dh' % len(samples), *samples))
    return str(path)


def _f32(path, values):
    array.array('f', values).tofile(open(path, 'wb'))
    return str(path)


def _run(*args):
    return subprocess.run([sys.executable, '-I', str(_TOOL), *args], capture_output=True, text=True)


def test_audio_identical_passes(tmp_path):
    a = _wav(tmp_path / 'a.wav', [1, 2, 3, 4])
    b = _wav(tmp_path / 'b.wav', [1, 2, 3, 4])
    assert _run('--audio-a', a, '--audio-b', b, '--fail-audio-lsb', '0').returncode == 0


@pytest.mark.parametrize('b_samples', [[], [1, 2]])
def test_audio_truncated_or_empty_run_fails_a_limit(tmp_path, b_samples):
    a = _wav(tmp_path / 'a.wav', [1, 2, 3, 4])
    b = _wav(tmp_path / 'b.wav', b_samples)
    r = _run('--audio-a', a, '--audio-b', b, '--fail-audio-lsb', '5')
    assert r.returncode == 1, r.stdout
    assert _run('--audio-a', a, '--audio-b', b).returncode == 0     # no limit: report only


@pytest.mark.parametrize('b_values', [[], [1.0] * 3, [1.0] * 4])
def test_heads_truncated_dump_fails_a_limit(tmp_path, b_values):
    a = _f32(tmp_path / 'a.f32', [1.0] * 8)
    b = _f32(tmp_path / 'b.f32', b_values)
    r = _run('--heads-a', a, '--heads-b', b, '--head-elems', '4', '--fail-max-abs', '1')
    assert r.returncode == (0 if len(b_values) == 8 else 1), r.stdout
    assert _run('--heads-a', a, '--heads-b', b, '--head-elems', '4').returncode == 0


def test_heads_identical_passes(tmp_path):
    a = _f32(tmp_path / 'a.f32', [1.0] * 8)
    b = _f32(tmp_path / 'b.f32', [1.0] * 8)
    assert _run('--heads-a', a, '--heads-b', b, '--head-elems', '4', '--fail-max-abs', '0').returncode == 0


@pytest.mark.parametrize('a_values, b_values, rc', [
    ([float('nan'), 1.0], [1.0, float('nan')], 1),            # the NaN moved to another element
    ([float('nan'), 1.0], [1.0, 1.0], 1),                      # present on one side only
    ([1.0, 1.0], [float('nan'), 1.0], 1),
    ([float('nan'), 1.0], [float('inf'), 1.0], 1),             # another kind at the same position
    ([float('inf'), 1.0], [float('-inf'), 1.0], 1),
    ([float('nan'), 1.0], [float('nan'), 1.0], 0),             # the same non-finite value in the same place
    ([float('inf'), -1.0], [float('inf'), -1.0], 0),
])
def test_heads_nonfinite_values_must_agree_in_kind_and_position(tmp_path, a_values, b_values, rc):
    a = _f32(tmp_path / 'a.f32', a_values)
    b = _f32(tmp_path / 'b.f32', b_values)
    r = _run('--heads-a', a, '--heads-b', b, '--head-elems', '2', '--fail-max-abs', '0')
    assert r.returncode == rc, r.stdout
