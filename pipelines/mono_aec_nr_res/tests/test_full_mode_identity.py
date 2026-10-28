"""The Python mono chain with the denoiser off IS the AEC's own full output.

``run_aec_linear`` (context seam) followed by ``run_res(use_nr=False)`` renders ``ctx.output_base_spec``
times the RES gain, adds comfort noise from the AEC library's generator and synthesises with the AEC's
sqrt-Hann OLA; ``AEC.process()`` of a second instance in full mode (``enable_res``) is the independent
oracle.  The two must agree bit for bit on streams that fire the output guard and move
``post_reset_serial`` (a delay change restarting the filter, an application ``reset()``), with comfort
noise on and off.  Planted defects (rendering from the analysis seam, ignoring the reset event, the old
comfort-noise seed) must each break the comparison, so the gate can fail.
"""
import dataclasses

import numpy as np
import pytest

from pipelines.mono_aec_nr_res import pipeline as P   # puts lib/aec/python on sys.path
from lib.aec.python.aec import AEC, AecConfig, AecMode, AecPreset, AEC_CNG_SEED  # noqa: E402

GRIDS = [(16000, 256, 6.0), (16000, 512, 6.0)]


def _scene(sr, seconds, seed=1, speech_far=True, echo=True, near_burst=True, silent_head=0.0):
    """Speech-like far-end bursts, a piecewise echo path (delay changes at 1/3 and 2/3), optional near burst."""
    rs = np.random.RandomState(seed)
    n = int(sr * seconds)
    seg = sr // 4
    far = np.zeros(n, np.float32)
    env, lp = 0.2, 0.0
    lpf = rs.randn(n).astype(np.float32)
    for i0 in range(0, n, seg):
        r = rs.uniform(-1, 1)
        env = (0.35 if r > 0.3 else (0.12 if r > -0.4 else 0.0)) if speech_far else 0.45
        seg_len = min(seg, n - i0)
        x = np.empty(seg_len, np.float32)
        for k in range(seg_len):
            lp = 0.6 * lp + 0.4 * lpf[i0 + k]
            x[k] = env * lp * 0.7
        far[i0:i0 + seg_len] = x
    far[:int(silent_head * sr)] = 0.0
    near = (0.004 * rs.randn(n)).astype(np.float32)
    if near_burst:
        near[n // 3:n // 3 + n // 6] = (0.15 * rs.randn(n // 6)).astype(np.float32)
    mic = near.copy()
    if echo:
        d = [480 * sr // 16000, 1100 * sr // 16000, 240 * sr // 16000]
        for lo, hi, dd in ((0, n // 3, d[0]), (n // 3, 2 * n // 3, d[1]), (2 * n // 3, n, d[2])):
            for i in range(lo, hi):
                if i - dd >= 0:
                    mic[i] += 0.45 * far[i - dd]
                if i - dd - 37 >= 0:
                    mic[i] += 0.2 * far[i - dd - 37]
    return mic.astype(np.float32), far.astype(np.float32)


def _config(sr, fft, enable_res, enable_cng):
    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr, frame_size=fft, hop_size=fft // 2,
                                mode=AecMode.PBFDKF, mu=0.3, enable_res=enable_res, enable_cng=enable_cng)
    return cfg


def _run(sr, fft, mic, far, enable_cng, reset_at=()):
    """(full-mode output, context-seam contexts, config) for the same input; the application's
    reset() hops are applied to both instances."""
    hop = fft // 2
    full = AEC(_config(sr, fft, True, enable_cng))
    cfg = _config(sr, fft, False, enable_cng)
    cfg.return_res_context = True
    split = AEC(cfg)
    n_hops = min(len(mic), len(far)) // hop
    out = np.zeros(n_hops * hop, np.float32)
    ctxs = []
    for h in range(n_hops):
        if h in reset_at:
            full.reset()
            split.reset()
        sl = slice(h * hop, (h + 1) * hop)
        out[sl] = full.process(mic[sl].copy(), far[sl].copy())
        _o, ctx = split.process(mic[sl].copy(), far[sl].copy())
        ctxs.append(ctx)
    return out, ctxs, cfg


def _render(ctxs, cfg, enable_cng, **kw):
    cfg = dataclasses.replace(cfg, enable_cng=enable_cng)
    n = len(ctxs) * cfg.hop_size
    tail = np.zeros(n, np.float32)
    ones = np.ones((len(ctxs), cfg.fft_size // 2 + 1), np.float32)
    return P.run_res(tail, ones, ctxs, cfg, use_nr=False, use_res=True, combine='min', **kw)


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _events(ctxs):
    guard = sum(int(c.guard_active) for c in ctxs)
    resets = sum(1 for a, b in zip([0] + [c.post_reset_serial for c in ctxs[:-1]],
                                   [c.post_reset_serial for c in ctxs]) if a != b)
    return guard, resets


@pytest.fixture(scope='module', params=GRIDS, ids=lambda g: '%dk_%d' % (g[0] // 1000, g[1]))
def streams(request):
    sr, fft, seconds = request.param
    hop = fft // 2
    mic, far = _scene(sr, seconds)
    resets = (int(seconds * sr / hop) // 2,)
    data = {}
    for cng in (True, False):
        out, ctxs, cfg = _run(sr, fft, mic, far, cng, reset_at=resets)
        data[cng] = (out, ctxs, cfg)
    return sr, fft, data


@pytest.mark.parametrize('cng', [True, False])
def test_nr_off_chain_equals_full_mode_bit_for_bit(streams, cng):
    _sr, _fft, data = streams
    out, ctxs, cfg = data[cng]
    mine = _render(ctxs, cfg, cng)
    assert np.array_equal(_bits(mine), _bits(out)), \
        'first differing hop %d' % (np.nonzero(_bits(mine) != _bits(out))[0][0] // cfg.hop_size)


def test_the_stream_exercises_the_guard_and_the_reset_event(streams):
    _sr, _fft, data = streams
    guard, resets = _events(data[True][1])
    assert guard > 0, 'the output guard never fired: the identity above would be vacuous'
    assert resets >= 2, 'no hard post-chain reset besides the first: the reset path is not exercised'


def test_planted_defects_are_detected(streams):
    _sr, _fft, data = streams
    out, ctxs, cfg = data[True]
    ref = _bits(out)
    # 1. rendering the analysis seam instead of the rendering base
    seam = [dataclasses.replace(c, output_base_spec=c.error_spec) for c in ctxs]
    assert not np.array_equal(_bits(_render(seam, cfg, True)), ref)
    # 2. ignoring the reset event (no OLA clear, no reseed)
    blind = [dataclasses.replace(c, post_reset_serial=0) for c in ctxs]
    assert not np.array_equal(_bits(_render(blind, cfg, True)), ref)
    # 3. the retired comfort-noise seed
    old = P.AEC_CNG_SEED
    try:
        P.AEC_CNG_SEED = 0x9e3779b9
        assert not np.array_equal(_bits(_render(ctxs, cfg, True)), ref)
    finally:
        P.AEC_CNG_SEED = old
    assert old == AEC_CNG_SEED == 42
