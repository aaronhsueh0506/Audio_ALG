"""Tests for the Phase-0 alignment-residual meter.

Two layers, mirroring the tool's own self-proof requirement:

* Pure-meter tests drive :func:`measure_residual_windows` with synthetic
  numpy signals whose ground-truth lag is known by construction -- they prove
  the correlation core recovers magnitude AND sign, and that the silence /
  low-correlation gates actually drop windows (a meter that cannot fail
  measures nothing).
* Engine-backed tests run the real frozen PBFDKF frontend (read-only, from
  ``lib/aec``) through :func:`run_self_test` on short white-noise scenes with
  known bulk delays, asserting the post-lock residual is positive and
  bounded, that an injected +N shift of the tool's own aligned-far tap moves
  the measurement by ~N, and that an injected negative lag is reported
  negative.

No corpus, no fixtures on disk: everything is synthesized in-process.
"""

import numpy as np
import pytest

from AIAEC.dataset_gen.measure_align_residual import (
    FileResult,
    WindowMeasurement,
    aggregate_results,
    count_delay_change_events,
    first_lock_hop,
    measure_residual_windows,
    run_self_test,
    shift_signal,
)

SR = 16000


def _noise(n: int, seed: int) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------------------
# Pure meter: known-lag recovery, sign convention, gating
# ---------------------------------------------------------------------------

def test_meter_recovers_known_positive_lag():
    """D_hat = far delayed through a short tail => lag exactly +delay."""
    n = 4 * SR
    far = _noise(n, 7)
    tail = np.zeros(64)
    tail[0], tail[10] = 1.0, 0.3
    dhat = np.convolve(shift_signal(far, -37), tail)[:n]
    windows, skipped = measure_residual_windows(far, dhat, SR)
    assert len(windows) >= 3
    assert all(w.lag_samples == 37 for w in windows)
    assert all(w.lag_ms == pytest.approx(37 * 1000.0 / SR) for w in windows)
    assert skipped == {"far_silent": 0, "low_corr": 0}


def test_meter_reports_negative_lag():
    """D_hat LEADING the reference must come out negative, not folded."""
    n = 4 * SR
    far = _noise(n, 8)
    dhat = shift_signal(far, 90)  # dhat(t) = far(t + 90): far is LATE
    windows, _ = measure_residual_windows(far, dhat, SR)
    assert len(windows) >= 3
    assert all(w.lag_samples == -90 for w in windows)


def test_meter_tracks_injected_far_tap_shift():
    """far_tap_shift=+N must move the measured lag by exactly N here."""
    n = 4 * SR
    far = _noise(n, 9)
    dhat = shift_signal(far, -37)
    base, _ = measure_residual_windows(far, dhat, SR)
    shifted, _ = measure_residual_windows(far, dhat, SR, far_tap_shift=100)
    assert [w.lag_samples for w in base] == [37] * len(base)
    assert [w.lag_samples for w in shifted] == [137] * len(shifted)


def test_meter_skips_silent_far_windows():
    n = 4 * SR
    far = _noise(n, 10)
    far[16512:32512] = 0.0  # exactly the second analysis window
    dhat = shift_signal(far, -37)
    windows, skipped = measure_residual_windows(far, dhat, SR)
    assert skipped["far_silent"] == 1
    assert [w.start_sample for w in windows] == [512, 32512]


def test_meter_skips_uncorrelated_windows():
    """Independent noise explains nothing: every window must be dropped."""
    n = 4 * SR
    windows, skipped = measure_residual_windows(
        _noise(n, 11), _noise(n, 12), SR
    )
    assert windows == []
    assert skipped["low_corr"] == 3


def test_shift_signal_conventions():
    x = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(
        shift_signal(x, 2), [2, 3, 4, 5, 6, 7, 0, 0]
    )
    np.testing.assert_array_equal(
        shift_signal(x, -2), [0, 0, 0, 1, 2, 3, 4, 5]
    )
    assert shift_signal(x, 0) is x


def test_lock_and_event_helpers():
    d = np.array([-1, -1, 100, 100, 132, 132, 132])
    assert first_lock_hop(d) == 2
    # Acquisition (-1 -> 100) is lock time, not an event; 100 -> 132 is one.
    assert count_delay_change_events(d) == 1
    assert first_lock_hop(np.array([-1, -1])) == -1
    assert count_delay_change_events(np.array([-1, -1])) == 0


def test_aggregate_results_percentiles_and_negative_fraction():
    def win(start, lag, locked=True):
        return WindowMeasurement(
            start_sample=start,
            lag_samples=lag,
            lag_ms=lag * 1000.0 / SR,
            peak_corr=0.9,
            locked=locked,
        )

    result = FileResult(
        name="synthetic",
        n_hops=100,
        lock_hop=5,
        lock_ms=80.0,
        delay_change_events=2,
        applied_delay_final=1536,
        confidence_final=1.0,
        windows=[
            win(512, 64), win(16512, 64), win(32512, -32),
            win(48512, 96), win(64512, 480, locked=False),
        ],
        skipped={"far_silent": 1, "low_corr": 2},
    )
    agg = aggregate_results([result])
    # The unlocked (pre-lock) window must not contaminate the statistics.
    assert agg["n_windows"] == 4
    assert agg["n_windows_prelock"] == 1
    assert agg["max_ms"] == pytest.approx(96 * 1000.0 / SR)
    assert agg["frac_negative"] == pytest.approx(0.25)
    assert agg["p50_ms"] == pytest.approx(64 * 1000.0 / SR)
    assert agg["delay_change_events_total"] == 2
    assert agg["lock_ms_median"] == pytest.approx(80.0)
    assert agg["n_windows_skipped_far_silent"] == 1
    assert agg["n_windows_skipped_low_corr"] == 2


def test_aggregate_results_empty():
    agg = aggregate_results([])
    assert agg["n_windows"] == 0
    assert agg["p50_ms"] is None
    assert agg["frac_negative"] is None


# ---------------------------------------------------------------------------
# Engine-backed self-test (real frozen frontend, short synthetic scenes)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selftest_report():
    """One real self-test run shared by the assertions below.

    Two bulk delays -- one short, one near the estimator's upper coverage -- at
    6 s each keep the wall time to a few seconds while still exercising
    acquisition, the ring-buffer compensation, and both injection proofs.
    """
    return run_self_test(
        bulk_delays=(1600, 7000), duration_s=6.0, seed=1234
    )


def test_self_test_passes(selftest_report):
    assert selftest_report["passed"], selftest_report


def test_residual_positive_and_bounded_after_lock(selftest_report):
    for case in selftest_report["cases"]:
        assert case["checks"]["locked"], case
        assert case["checks"]["has_post_lock_windows"], case
        assert case["residual_ms"] > 0.0, case
        assert case["residual_ms"] < 12.0, case


def test_applied_delay_close_to_ground_truth(selftest_report):
    """Residual = bulk - applied must agree with the meter's own reading."""
    for case in selftest_report["cases"]:
        expected = case["bulk_delay"] - case["applied_delay_final"]
        assert 0 < expected < 512, case
        assert case["residual_samples"] == pytest.approx(expected, abs=8), case


def test_injected_shift_tracks(selftest_report):
    for case in selftest_report["cases"]:
        assert case["checks"]["injection_tracks"], case
        assert case["injected_delta_samples"] == pytest.approx(
            case["injected_shift"], abs=16
        ), case


def test_injected_negative_lag_reported_negative(selftest_report):
    for case in selftest_report["cases"]:
        assert case["checks"]["negative_lag_reported_negative"], case
        assert case["negative_lag_samples"] < 0, case
