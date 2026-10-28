"""The Python rate bridge against the C bridge it mirrors.

Both bridges get identical native spectra (a host-style sqrt-Hann analysis of
the same 16 kHz / 8 kHz signals) and the identity model.  The resamplers are
the same C kernel on both sides, so the only differences are the FFT
backends; the outputs must agree to float32 round-off, and the calibrated
prefill / added delay (floored and exact) must be the same integers.  Runs for the shipped
lookahead pair and two others (``C_PAIRS``), the C rebuilt with the pair's
``-D`` overrides."""
import ctypes
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip('torch')

from pipelines.dfn.dfn2_rate_adapter import Dfn2RateBridge, _root_hann  # noqa: E402
from pipelines.dfn.dfn2_stage import Dfn2Stage, IdentityHeads, load_dfn2  # noqa: E402
from pipelines.tests.c_shim import build_shim, float_pointer as _fp  # noqa: E402
from pipelines.dfn.tests.dfn2_pairs import C_PAIRS, SHIPPED, pair_config, pair_defines, pair_id  # noqa: E402

DFN = ROOT / 'AINR' / 'DeepFilterNet2'
PIPELINES = ROOT / 'pipelines'

_SHIM = r'''
#include <string.h>
#include "dfn/dfn_rate_bridge.h"

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
int bridge_parity_added_delay_exact(void *b, int *numerator, int *denominator) {
    return dfn_rate_bridge_added_delay_exact((DfnRateBridge *)b, numerator, denominator);
}
int bridge_parity_reset_synthesis(void *b) {
    return dfn_rate_bridge_reset_synthesis((DfnRateBridge *)b);
}
int bridge_parity_host_hops(void *b, float *estimate, float *apply) {
    const float *e, *p;
    int hop = dfn_rate_bridge_hop_size((DfnRateBridge *)b);
    if (dfn_rate_bridge_get_host_hops((DfnRateBridge *)b, &e, &p) != 0) return -1;
    memcpy(estimate, e, (size_t)hop * sizeof(float));
    memcpy(apply, p, (size_t)hop * sizeof(float));
    return 0;
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


@pytest.fixture(scope='module', params=C_PAIRS, ids=pair_id)
def pair(request):
    return request.param


@pytest.fixture(scope='module')
def assets(pair, tmp_path_factory):
    if pair == SHIPPED:
        return load_dfn2(seed=0)
    return load_dfn2(config_path=pair_config(tmp_path_factory.mktemp('dfn2_pair'), pair),
                     seed=0)


@pytest.fixture(scope='module')
def c_lib(pair, tmp_path_factory):
    lib = build_shim(
        tmp_path_factory.mktemp('dfn2_rate_bridge_c_parity'), 'shim', _SHIM,
        extra_sources=[PIPELINES / 'dfn' / 'dfn_rate_bridge.c', PIPELINES / 'dfn' / 'dfn_res_stage.c',
                       DFN / 'dfn2_process.c', DFN / 'dfn2_model_io.c',
                       DFN / 'dfn2_prepost.c'],
        extra_includes=[PIPELINES, DFN],
        defines=['AUDIO_PIPELINE_BACKEND_STR="ne10"'] + pair_defines(pair))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.bridge_parity_create.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp]
    lib.bridge_parity_create.restype = ctypes.c_void_p
    lib.bridge_parity_destroy.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_hop.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_hop.restype = ctypes.c_int
    lib.bridge_parity_added_delay.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_added_delay.restype = ctypes.c_int
    lib.bridge_parity_added_delay_exact.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.bridge_parity_added_delay_exact.restype = ctypes.c_int
    lib.bridge_parity_process.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, fp]
    lib.bridge_parity_process.restype = ctypes.c_int
    lib.bridge_parity_reset_synthesis.argtypes = [ctypes.c_void_p]
    lib.bridge_parity_reset_synthesis.restype = ctypes.c_int
    lib.bridge_parity_host_hops.argtypes = [ctypes.c_void_p, fp, fp]
    lib.bridge_parity_host_hops.restype = ctypes.c_int
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
    # The same 16 tones and phases as dfn_fixture_multitone (dfn/tests/dfn_test_fixture.h).
    tones = [211, 347, 503, 659, 811, 977, 1123, 1301,
             1487, 1693, 1901, 2129, 2357, 2609, 2803, 2999]
    t = np.arange(count, dtype=np.float64) / sample_rate
    return sum(0.06 * np.sin(2 * np.pi * f * t + k) for k, f in enumerate(tones)
               ).astype(np.float32)


def _run_c_and_py(c_lib, assets, sample_rate, fft, e_signal, p_signal, reset_at=(), taps=None):
    """Both bridges on the same host-analysed signals: (c_out, py_out, python bridge).
    `reset_at`: hops before which both bridges get their synthesis-only reset; `taps`, a dict, receives
    the native-grid hops each bridge synthesised ('c_e', 'c_p', 'py_e', 'py_p': one array per hop)."""
    hop = fft // 2
    n = len(p_signal)
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
        num, den = ctypes.c_int(0), ctypes.c_int(0)
        assert c_lib.bridge_parity_added_delay_exact(
            handle, ctypes.byref(num), ctypes.byref(den)) == 0
        assert (num.value, den.value) == py.added_delay_exact
        assert num.value // den.value == py.added_delay
        c_out = np.zeros(n, np.float32)
        py_out = np.zeros(n, np.float32)
        for t, (e, p) in enumerate(zip(e_spec, p_spec)):
            out = np.zeros(hop, np.float32)
            if t in reset_at:
                assert c_lib.bridge_parity_reset_synthesis(handle) == 0
                py.reset_synthesis()
            rc = c_lib.bridge_parity_process(
                handle, _fp(np.ascontiguousarray(e.real, np.float32)),
                _fp(np.ascontiguousarray(e.imag, np.float32)),
                _fp(np.ascontiguousarray(p.real, np.float32)),
                _fp(np.ascontiguousarray(p.imag, np.float32)), _fp(out))
            assert rc == 0
            c_out[t * hop:(t + 1) * hop] = out
            py_out[t * hop:(t + 1) * hop] = py.push(e, p)
            if taps is not None:
                te, tp = np.zeros(hop, np.float32), np.zeros(hop, np.float32)
                assert c_lib.bridge_parity_host_hops(handle, _fp(te), _fp(tp)) == 0
                for key, value in (('c_e', te), ('c_p', tp), ('py_e', py.host_hops[0]), ('py_p', py.host_hops[1])):
                    taps.setdefault(key, []).append(np.array(value, np.float32))
    finally:
        c_lib.bridge_parity_destroy(handle)
    return c_out, py_out, py


@pytest.mark.parametrize('sample_rate,fft', [(16000, 256), (16000, 512), (8000, 256)])
def test_python_bridge_matches_the_c_bridge(c_lib, assets, sample_rate, fft):
    hop = fft // 2
    n = 120 * hop
    p_signal = _multitone(n, sample_rate)
    e_signal = (0.5 * np.sin(2 * np.pi * 700.0 * np.arange(n) / sample_rate)).astype(np.float32)
    p_signal[40 * hop + 7] += 1.0
    c_out, py_out, _ = _run_c_and_py(c_lib, assets, sample_rate, fft, e_signal, p_signal)
    assert np.max(np.abs(c_out)) > 0.1
    np.testing.assert_allclose(py_out, c_out, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize('sample_rate,fft', [(16000, 256), (16000, 512), (8000, 256)])
def test_an_impulse_lands_at_the_exact_added_delay(c_lib, assets, sample_rate, fft):
    """The P signal comes back hop + the exact added delay later in both bridges. The round
    trip is a symmetric band-limited kernel, so its centroid sits at that position to a
    fraction of a sample, where the floored delay is off by up to 2/3 of a sample for the
    pairs with L + 1 not a multiple of 3."""
    hop = fft // 2
    n = 120 * hop
    at = 40 * hop + 7
    p_signal = np.zeros(n, np.float32)
    p_signal[at] = 1.0
    c_out, py_out, py = _run_c_and_py(c_lib, assets, sample_rate, fft,
                                      np.zeros(n, np.float32), p_signal)
    num, den = py.added_delay_exact
    expected = at + hop + num / den
    for out in (c_out, py_out):
        peak = int(np.argmax(np.abs(out)))
        window = slice(peak - 24, peak + 25)
        centroid = float((np.arange(n)[window] * out[window]).sum() / out[window].sum())
        assert abs(centroid - expected) < 0.01, (centroid, expected)
        assert abs(peak - expected) < 0.5


@pytest.mark.parametrize('sample_rate,fft', [(16000, 256), (16000, 512), (8000, 256)])
def test_the_synthesis_only_reset_matches_the_c_bridge(c_lib, assets, sample_rate, fft):
    """dfn_rate_bridge_reset_synthesis() / Dfn2RateBridge.reset_synthesis() before hops 41 and 77: the native-grid
    hops and the output agree to float round-off in both bridges, the clear lands on the hop it names (the host
    hops differ from the uncleared run there and nowhere else), and the Python bridge keeps its frame counters
    (the reset leaves DFN2, the resamplers and the FIFOs running)."""
    hop = fft // 2
    n = 120 * hop
    resets = (41, 77)
    p_signal = _multitone(n, sample_rate)
    e_signal = (0.5 * np.sin(2 * np.pi * 700.0 * np.arange(n) / sample_rate)).astype(np.float32)
    taps, taps_ref = {}, {}
    c_out, py_out, py = _run_c_and_py(c_lib, assets, sample_rate, fft, e_signal, p_signal,
                                      reset_at=resets, taps=taps)
    c_ref, py_ref, py_unreset = _run_c_and_py(c_lib, assets, sample_rate, fft, e_signal, p_signal,
                                              taps=taps_ref)
    np.testing.assert_allclose(py_out, c_out, rtol=2e-4, atol=2e-6)
    for key in ('e', 'p'):
        np.testing.assert_allclose(np.concatenate(taps['py_' + key]), np.concatenate(taps['c_' + key]),
                                   rtol=2e-4, atol=2e-6)
    for who in ('c', 'py'):
        for key in ('e', 'p'):
            changed = [t for t in range(120)
                       if not np.allclose(taps[f'{who}_{key}'][t], taps_ref[f'{who}_{key}'][t], atol=1e-6)]
            assert changed == list(resets), (who, key, changed)
    assert np.max(np.abs(c_out - c_ref)) > 1e-2           # the clear is audible in the output
    assert (py.frames48, py.last_hop_frames) == (py_unreset.frames48, py_unreset.last_hop_frames)
    assert py.added_delay_exact == py_unreset.added_delay_exact
