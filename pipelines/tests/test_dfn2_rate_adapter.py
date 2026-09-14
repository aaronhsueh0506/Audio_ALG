"""The Python resampler twin against the C it mirrors, and the round-trip
delay constant the non-48 kHz DFN2 path relies on."""
import ctypes
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.dfn2_rate_adapter import AudioResampler, CResampler, round_trip  # noqa: E402

AC = ROOT.parent / 'audio_common'


@pytest.fixture(scope='module')
def c_lib():
    """The adapter's own ctypes binding of the scalar C kernel."""
    if not (shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')):
        pytest.skip('no C compiler')
    return CResampler.library()


def _c_resample(lib, rate_in, rate_out, signal):
    handle = lib.audio_resampler_create(rate_in, rate_out, 1)
    assert handle
    try:
        latency = lib.audio_resampler_latency_input_frames(handle)
        chunk = 160
        out_parts = []
        for start in range(0, len(signal), chunk):
            block = np.ascontiguousarray(signal[start:start + chunk], np.float32)
            bound = lib.audio_resampler_output_bound(handle, len(block))
            out = np.zeros(max(bound, 1), np.float32)
            consumed = ctypes.c_int(0)
            produced = ctypes.c_int(0)
            rc = lib.audio_resampler_process(
                handle, block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(block),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(out),
                ctypes.byref(consumed), ctypes.byref(produced))
            assert rc == 0 and consumed.value == len(block)
            out_parts.append(out[:produced.value])
        return np.concatenate(out_parts), latency
    finally:
        lib.audio_resampler_destroy(handle)


@pytest.mark.parametrize('rate_in,rate_out', [
    (8000, 48000), (48000, 8000),
    (16000, 48000), (48000, 16000),
    (24000, 48000), (48000, 24000),
    (32000, 48000), (48000, 32000),
    (48000, 48000),
])
def test_python_twin_matches_the_scalar_c_kernel(c_lib, rate_in, rate_out):
    rng = np.random.RandomState(3)
    signal = (rng.randn(2400) * 0.3).astype(np.float32)
    c_out, c_latency = _c_resample(c_lib, rate_in, rate_out, signal)
    twin = AudioResampler(rate_in, rate_out)
    py_out = twin.process(signal)
    assert twin.latency_input_frames == c_latency
    assert len(py_out) == len(c_out)
    # Same design, same float32 op order: bit-exact on the scalar C kernel.
    assert np.array_equal(py_out, c_out)


@pytest.mark.parametrize('native,expected_delay', [
    (8000, 32), (16000, 32), (24000, 32), (32000, 32), (48000, 0),
])
def test_round_trip_delay_matches_the_rate_pair(native, expected_delay):
    n = 4000
    impulse = np.zeros(n, np.float32)
    impulse[1000] = 1.0
    back, delay = round_trip(impulse, native)
    assert delay == expected_delay
    assert int(np.argmax(np.abs(back))) == 1000 + expected_delay


def test_up_direction_produces_exactly_up_outputs_per_input():
    twin = AudioResampler(16000, 48000)
    out = twin.process(np.zeros(100, np.float32) + 0.1)
    assert len(out) == 300
    twin8 = AudioResampler(8000, 48000)
    assert len(twin8.process(np.zeros(50, np.float32))) == 300
