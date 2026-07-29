#!/usr/bin/env python3
"""The normaliser-init calibration must recover a ramp it was given.

``erb_norm_init_lo_db``/``hi_db`` are the two ends of a ``torch.linspace``
across FREQUENCY (train.py ``normalize_log_erb``), so calibrating them means
fitting the frequency profile.  The original script pooled every band and
every frame into one distribution and reported its median and 5th percentile.
That measures how levels vary over time and loudness, not over frequency: on
the synthetic case below it reports a 20.8 dB span for a signal whose true
frequency ramp is 16.0 dB, with the difference coming entirely from per-frame
level jitter.

This test pins the property that actually matters -- given data built from a
known ramp, the calibration returns that ramp.
"""

import contextlib
import io
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Each of the three model projects has its own top-level ``train.py``; drop any
# cached copy so this resolves against the ROOT just inserted above.
for _stale in ('train', 'denoise', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'calibrate_norm_init', ROOT / 'calibrate_norm_init.py')
calibrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate)


TRUE_LO, TRUE_HI = -58.0, -74.0
N_BANDS, N_FRAMES = 22, 5000
LEVEL_JITTER_DB = 12.0


def _synthetic_bands(seed=0):
    """Frames whose per-band mean IS a known ramp, plus frequency-flat jitter.

    The jitter is applied equally to every band, so it carries no frequency
    information at all -- any method that reports it as part of the ramp is
    reading the wrong axis.
    """
    torch.manual_seed(seed)
    ramp = torch.linspace(TRUE_LO, TRUE_HI, N_BANDS)
    level = torch.randn(N_FRAMES, 1) * LEVEL_JITTER_DB
    return ramp.unsqueeze(0) + level + torch.randn(N_FRAMES, N_BANDS) * 1.5


def test_fit_recovers_the_ramp_it_was_given():
    x = _synthetic_bands()
    with contextlib.redirect_stdout(io.StringIO()):
        lo, hi, residual = calibrate._fit_ramp(x, 'ERB band', 'dB')
    assert abs(lo - TRUE_LO) < 0.5, (lo, TRUE_LO)
    assert abs(hi - TRUE_HI) < 0.5, (hi, TRUE_HI)
    # The ramp is exactly linear here, so almost nothing should be left over.
    assert residual < 0.5, residual


def test_frequency_flat_jitter_does_not_enter_the_ramp():
    """Tripling the level jitter must not change the fitted endpoints.

    This is the specific failure of the pooled-quantile approach: its endpoints
    are driven by the jitter, so they move when the jitter does even though the
    frequency profile is untouched.
    """
    torch.manual_seed(0)
    ramp = torch.linspace(TRUE_LO, TRUE_HI, N_BANDS)
    base = torch.randn(N_FRAMES, 1)
    noise = torch.randn(N_FRAMES, N_BANDS) * 1.5

    out = []
    for scale in (LEVEL_JITTER_DB, LEVEL_JITTER_DB * 3):
        x = ramp.unsqueeze(0) + base * scale + noise
        with contextlib.redirect_stdout(io.StringIO()):
            out.append(calibrate._fit_ramp(x, 'ERB band', 'dB')[:2])
    (lo1, hi1), (lo2, hi2) = out
    assert abs(lo1 - lo2) < 0.5 and abs(hi1 - hi2) < 0.5, out


def test_pooled_quantiles_would_have_failed():
    """Guard the reason this test file exists.

    If someone reverts to pooled median/p05 endpoints, this documents by how
    much they are wrong -- and fails if the old approach ever silently becomes
    the accepted one.
    """
    x = _synthetic_bands()
    pooled_lo = calibrate._quantile(x, 0.5)
    pooled_hi = calibrate._quantile(x, 0.05)
    assert abs(pooled_lo - TRUE_LO) > 3.0, pooled_lo
    assert abs(pooled_hi - TRUE_HI) > 3.0, pooled_hi
    true_span = abs(TRUE_HI - TRUE_LO)
    assert abs(abs(pooled_hi - pooled_lo) - true_span) > 2.0


def test_quantile_helper_handles_oversized_input():
    """torch.quantile rejects >~16.7M elements; the default --clips exceeds it."""
    big = torch.arange(17_000_001, dtype=torch.float32)
    got = calibrate._quantile(big, 0.5)
    assert abs(got - 8_500_000) < 10_000, got


if __name__ == '__main__':
    tests = [
        test_fit_recovers_the_ramp_it_was_given,
        test_frequency_flat_jitter_does_not_enter_the_ramp,
        test_pooled_quantiles_would_have_failed,
        test_quantile_helper_handles_oversized_input,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
