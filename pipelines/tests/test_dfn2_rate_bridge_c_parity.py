"""The Python rate bridge against the C bridge it mirrors.

Both bridges get identical native spectra (a host-style sqrt-Hann analysis of
the same 16 kHz / 8 kHz signals) and the identity model.  The resamplers are
the same C kernel on both sides, so the only differences are the FFT
backends; the outputs must agree to float32 round-off, and the calibrated
prefill / added delay must be the same integers."""
import ctypes
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip('torch')

from pipelines.dfn2_rate_adapter import Dfn2RateBridge, _root_hann  # noqa: E402
from pipelines.dfn2_stage import Dfn2Stage, IdentityHeads, load_dfn2  # noqa: E402
from pipelines.tests.c_shim import build_shim, float_pointer as _fp  # noqa: E402

DFN = ROOT / 'AINR' / 'DeepFilterNet2'
PIPELINES = ROOT / 'pipelines'

_SHIM = r'''
#include <string.h>
#include "dfn_rate_bridge.h"

void *bridge_parity_create(int sample_rate, int fft_size, const float *fwd,
                           const float *inv) {
    DfnRateBridgeConfig cfg = dfn_rate_bridge_default_config(sample_rate);
    cfg.fft_size = fft_size;
    cfg.stage.erb_fwd = fwd;
    cfg.stage.erb_inv = inv;
    return dfn_rate_bridge_create(&cfg);
}
void bridge_parity_destroy(void *b) { dfn_rate_bridge_destroy((DfnRateBridge *)b); }
int bridge_parity_hop(void *b) { return dfn_rate_bridge_hop_size((DfnRateBridge *)b); }
int bridge_parity_added_delay(void *b) {
    return dfn_rate_bridge_added_delay_samples((DfnRateBridge *)b);
}
int bridge_parity_process(void *b, const float *est_re, const float *est_im,
                          const float *app_re, const float *app_im, float *out) {
    Complex est[257], app[257];
    int n = dfn_rate_bridge_n_freqs((DfnRateBridge *)b), k;
    for (k = 0; k < n; ++k) {
        est[k].r = est_re[k]; est[k].i = est_im[k];
        app[k].r = app_re[k]; app[k].i = app_im[k];
    }
    return dfn_rate_bridge_process((DfnRateBridge *)b, est, app, out);
}
'''


@pytest.fixture(scope='module')
def assets():
    return load_dfn2(seed=0)


@pytest.fixture(scope='module')
def c_lib(tmp_path_factory):
    lib = build_shim(
        tmp_path_factory.mktemp('dfn2_rate_bridge_c_parity'), 'shim', _SHIM,
        extra_sources=[PIPELINES / 'dfn_rate_bridge.c', PIPELINES / 'dfn_res_stage.c',
                       DFN / 'dfn2_process.c', DFN / 'dfn2_model_io.c',
                       DFN / 'dfn2_prepost.c'],
        extra_includes=[PIPELINES, DFN],
        defines=['AUDIO_PIPELINE_BACKEND_STR="ne10"'])
    fp = ctypes.POINTER(ctypes.c_float)
    lib.bridge_parity_create.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp]
    lib.bridge_parity_create.restype = ctypes.c_void_p
    lib.bridge_parity_destroy.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_hop.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_hop.restype = ctypes.c_int
    lib.bridge_parity_added_delay.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_added_delay.restype = ctypes.c_int
    lib.bridge_parity_process.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, fp]
    lib.bridge_parity_process.restype = ctypes.c_int
    return lib


def _host_spectra(signal, fft):
    """The host's analysis: frame = [previous hop, this hop] * sqrt-Hann."""
    hop = fft // 2
    win = _root_hann(fft)
    prev = np.zeros(hop, np.float32)
    out = []
    for start in range(0, len(signal) - hop + 1, hop):
        cur = signal[start:start + hop]
        frame = np.concatenate([prev, cur]).astype(np.float32) * win
        out.append(np.fft.rfft(frame).astype(np.complex64))
        prev = cur
    return out


def _multitone(count, sample_rate):
    # The same 16 tones and phases as dfn_fixture_multitone (tests/dfn_test_fixture.h).
    tones = [211, 347, 503, 659, 811, 977, 1123, 1301,
             1487, 1693, 1901, 2129, 2357, 2609, 2803, 2999]
    t = np.arange(count, dtype=np.float64) / sample_rate
    return sum(0.06 * np.sin(2 * np.pi * f * t + k) for k, f in enumerate(tones)
               ).astype(np.float32)


@pytest.mark.parametrize('sample_rate,fft', [(16000, 256), (16000, 512), (8000, 256)])
def test_python_bridge_matches_the_c_bridge(c_lib, assets, sample_rate, fft):
    hop = fft // 2
    hops = 120
    n = hops * hop
    p_signal = _multitone(n, sample_rate)
    e_signal = (0.5 * np.sin(2 * np.pi * 700.0 * np.arange(n) / sample_rate)).astype(np.float32)
    p_signal[40 * hop + 7] += 1.0
    e_spec = _host_spectra(e_signal, fft)
    p_spec = _host_spectra(p_signal, fft)

    # C wants the forward bank bin-major [513][32]; the assets keep it
    # band-major (n_erb, n_bins). erb_inv is band-major on both sides.
    fwd = np.ascontiguousarray(assets.erb_fb.detach().cpu().numpy().T, np.float32)
    inv = np.ascontiguousarray(assets.erb_inv, np.float32)
    handle = c_lib.bridge_parity_create(sample_rate, fft, _fp(fwd), _fp(inv))
    assert handle
    py = Dfn2RateBridge(Dfn2Stage(assets, heads=IdentityHeads()), sample_rate, fft)
    try:
        assert c_lib.bridge_parity_hop(handle) == hop == py.hop
        assert c_lib.bridge_parity_added_delay(handle) == py.added_delay
        c_out = np.zeros(n, np.float32)
        py_out = np.zeros(n, np.float32)
        for t, (e, p) in enumerate(zip(e_spec, p_spec)):
            out = np.zeros(hop, np.float32)
            rc = c_lib.bridge_parity_process(
                handle, _fp(np.ascontiguousarray(e.real, np.float32)),
                _fp(np.ascontiguousarray(e.imag, np.float32)),
                _fp(np.ascontiguousarray(p.real, np.float32)),
                _fp(np.ascontiguousarray(p.imag, np.float32)), _fp(out))
            assert rc == 0
            c_out[t * hop:(t + 1) * hop] = out
            py_out[t * hop:(t + 1) * hop] = py.push(e, p)
    finally:
        c_lib.bridge_parity_destroy(handle)
    assert np.max(np.abs(c_out)) > 0.1
    np.testing.assert_allclose(py_out, c_out, rtol=2e-4, atol=2e-6)
    # And both are the P signal delayed by hop + added delay, in the pass band.
    delay = hop + py.added_delay
    peak = int(np.argmax(np.abs(c_out)))
    assert peak == 40 * hop + 7 + delay
