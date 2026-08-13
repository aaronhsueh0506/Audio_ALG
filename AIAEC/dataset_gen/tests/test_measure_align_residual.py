"""Tests for the Phase-0 alignment-residual meter.

Three layers, mirroring the tool's own self-proof requirement:

* Pure-meter tests drive :func:`measure_residual_windows` with synthetic
  numpy signals whose ground-truth lag is known by construction -- they prove
  the correlation core recovers magnitude AND sign, that the silence /
  low-correlation gates actually drop windows, that a peak on the +-max_lag
  search boundary is flagged, and that the locked-window gate (stable delay +
  solid confidence + settle time) admits and rejects exactly what it claims
  (a meter that cannot fail measures nothing).
* Engine-backed tests run the real frozen PBFDKF frontend (read-only, from
  ``lib/aec``) through :func:`run_self_test` on short white-noise scenes with
  known bulk delays, asserting the post-lock residual vs the D_hat PROXY is
  positive and bounded, that an injected +N shift of the tool's own
  aligned-far tap moves the measurement by ~N, and that an injected negative
  lag is reported negative.
* Generator-backed tests render short clean sequences fully in memory with
  the repo's own ``AecSequenceRenderer`` and assert the residual against the
  TRUE echo (``RenderedSequence.audit['echo']``) obeys the same bounds, and
  that the D_hat proxy agrees with the true echo within one histogram
  quantum.

No corpus, no fixtures on disk (the generator-backed layer writes only
throwaway source WAVs to an ephemeral temp dir): everything is synthesized
in-process.
"""

import numpy as np
import pytest

from AIAEC.dataset_gen.measure_align_residual import (
    DEFAULT_MAX_LAG,
    PROXY_SOURCE,
    PROXY_TRUE_AGREEMENT_QUANTUM,
    TRUE_ECHO_SOURCE,
    FileResult,
    WindowMeasurement,
    _print_self_test,
    _print_synthetic_echo_test,
    _print_table,
    aggregate_results,
    count_delay_change_events,
    first_lock_hop,
    last_change_hops,
    measure_residual_windows,
    run_self_test,
    run_synthetic_echo_test,
    shift_signal,
    solid_confidence_threshold,
    undecidable_gate,
)

SR = 16000
HOP = 256


def _noise(n: int, seed: int) -> np.ndarray:
    return 0.1 * np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------------------
# Pure meter: known-lag recovery, sign convention, gating
# ---------------------------------------------------------------------------

def test_default_search_range_bounds_d8_span():
    """+-2048 samples = 128 ms at 16 kHz >= the D=8 hop span (112 ms)."""
    assert DEFAULT_MAX_LAG == 2048
    assert DEFAULT_MAX_LAG * 1000.0 / SR >= 112.0


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
    assert all(not w.boundary_peak for w in windows)
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
    # Exactly the second analysis window ([2048 + 16000, 2048 + 2*16000)).
    far[18048:34048] = 0.0
    dhat = shift_signal(far, -37)
    windows, skipped = measure_residual_windows(far, dhat, SR)
    assert skipped["far_silent"] == 1
    assert [w.start_sample for w in windows] == [2048, 34048]


def test_meter_skips_uncorrelated_windows():
    """Independent noise explains nothing: every window must be dropped."""
    n = 4 * SR
    windows, skipped = measure_residual_windows(
        _noise(n, 11), _noise(n, 12), SR
    )
    assert windows == []
    assert skipped["low_corr"] == 3


def test_meter_flags_boundary_peaks():
    """A peak exactly ON +-max_lag must be flagged; one inside must not.

    A boundary peak is indistinguishable from a clipped out-of-range peak,
    so 'frac_boundary_peak > 0' is the report's own proof that the search
    range cannot bound the residual.
    """
    n = 4 * SR
    far = _noise(n, 13)
    on_boundary, _ = measure_residual_windows(
        far, shift_signal(far, -256), SR, max_lag=256
    )
    assert len(on_boundary) >= 3
    assert all(w.lag_samples == 256 for w in on_boundary)
    assert all(w.boundary_peak for w in on_boundary)

    inside, _ = measure_residual_windows(
        far, shift_signal(far, -200), SR, max_lag=256
    )
    assert all(w.lag_samples == 200 for w in inside)
    assert all(not w.boundary_peak for w in inside)


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


def test_last_change_hops():
    d = np.array([-1, -1, 100, 100, 132, 132, 132])
    # Hop 0 is a change point (no pre-run history); acquisition at hop 2 and
    # the 100 -> 132 step at hop 4 each reset the settle clock.
    np.testing.assert_array_equal(
        last_change_hops(d), [0, 0, 2, 2, 4, 4, 4]
    )
    np.testing.assert_array_equal(last_change_hops(np.array([7])), [0])
    assert last_change_hops(np.array([])).size == 0


def test_solid_confidence_threshold_scale_detection():
    # The run's scale reaches 1.0 -> only 1.0 (refined) counts as solid.
    assert solid_confidence_threshold(np.array([0.0, 0.5, 1.0])) == 1.0
    # The run never reports 1.0 -> 0.5 is the best available and counts.
    assert solid_confidence_threshold(np.array([0.0, 0.5, 0.5])) == 0.5
    # No confidence signal at all -> falls back to the coarse threshold.
    assert solid_confidence_threshold(np.array([np.nan, np.nan])) == 0.5


# ---------------------------------------------------------------------------
# Locked-window gate: stable delay + settle time + solid confidence
# ---------------------------------------------------------------------------

def _gated_windows(delay, conf=None, settle_s=0.5, n=4 * SR, seed=20):
    far = _noise(n, seed)
    dhat = shift_signal(far, -37)
    windows, _ = measure_residual_windows(
        far, dhat, SR,
        hop_size=HOP, delay_samples=delay, confidence=conf,
        settle_s=settle_s,
    )
    # Default geometry: three 1 s (16000-sample) windows.
    assert [w.start_sample for w in windows] == [2048, 18048, 34048]
    return [w.locked for w in windows]


def test_gate_requires_settle_after_run_start():
    """A constant delay still needs settle_s of history before window 1."""
    delay = np.full(4 * SR // HOP, 800, dtype=np.int64)
    # settle 0.5 s = 8000 samples: window at 2048 is inside the settle
    # window, the later two are not.
    assert _gated_windows(delay, settle_s=0.5) == [False, True, True]
    assert _gated_windows(delay, settle_s=0.0) == [True, True, True]


def test_gate_requires_stable_delay_and_settle_after_change():
    delay = np.full(4 * SR // HOP, 800, dtype=np.int64)
    delay[:40] = -1        # acquisition at hop 40 (sample 10240)
    delay[100:] = 832      # delay change at hop 100 (sample 25600)
    # Window 1 overlaps unacquired hops; window 2 straddles the 800 -> 832
    # change; window 3 (start 34048) has 8448 samples >= 0.5 s of settle.
    assert _gated_windows(delay, settle_s=0.5) == [False, False, True]
    # A longer settle requirement must also reject window 3
    # (34048 - 25600 = 8448 samples = 0.528 s < 1.0 s).
    assert _gated_windows(delay, settle_s=1.0) == [False, False, False]


def test_gate_requires_solid_confidence():
    n_hops = 4 * SR // HOP
    delay = np.full(n_hops, 800, dtype=np.int64)

    solid = np.full(n_hops, 1.0)
    assert _gated_windows(delay, conf=solid, settle_s=0.0) == [
        True, True, True,
    ]

    # The run's scale reaches 1.0, so 0.5 (coarse) hops are NOT solid.
    coarse_after_refined = np.full(n_hops, 0.5)
    coarse_after_refined[0] = 1.0
    locked = _gated_windows(delay, conf=coarse_after_refined, settle_s=0.0)
    assert locked == [False, False, False]

    # The run never reports 1.0: 0.5 is the best available and counts.
    coarse_only = np.full(n_hops, 0.5)
    assert _gated_windows(delay, conf=coarse_only, settle_s=0.0) == [
        True, True, True,
    ]

    # No confidence signal at all cannot be gated on.
    unavailable = np.full(n_hops, np.nan)
    assert _gated_windows(delay, conf=unavailable, settle_s=0.0) == [
        True, True, True,
    ]


# ---------------------------------------------------------------------------
# Aggregation and the undecidable-fraction gate
# ---------------------------------------------------------------------------

def _win(start, lag, locked=True, boundary_peak=False):
    return WindowMeasurement(
        start_sample=start,
        lag_samples=lag,
        lag_ms=lag * 1000.0 / SR,
        peak_corr=0.9,
        locked=locked,
        boundary_peak=boundary_peak,
    )


def _file_result(windows, skipped):
    return FileResult(
        name="synthetic",
        n_hops=100,
        lock_hop=5,
        lock_ms=80.0,
        delay_change_events=2,
        applied_delay_final=1536,
        confidence_final=1.0,
        windows=windows,
        skipped=skipped,
    )


def test_aggregate_results_percentiles_and_negative_fraction():
    result = _file_result(
        windows=[
            _win(2048, 64), _win(18432, 64), _win(34816, -32),
            _win(51200, 96), _win(67584, 480, locked=False),
        ],
        skipped={"far_silent": 1, "low_corr": 2},
    )
    agg = aggregate_results([result])
    # The unsettled window must not contaminate the statistics.
    assert agg["n_windows"] == 4
    assert agg["n_windows_unsettled"] == 1
    assert agg["max_ms"] == pytest.approx(96 * 1000.0 / SR)
    assert agg["frac_negative"] == pytest.approx(0.25)
    assert agg["p50_ms"] == pytest.approx(64 * 1000.0 / SR)
    assert agg["delay_change_events_total"] == 2
    assert agg["lock_ms_median"] == pytest.approx(80.0)
    assert agg["n_windows_skipped_far_silent"] == 1
    assert agg["n_windows_skipped_low_corr"] == 2
    # 5 measured windows, none on the boundary.
    assert agg["n_windows_boundary_peak"] == 0
    assert agg["frac_boundary_peak"] == pytest.approx(0.0)
    # (silent 1 + low_corr 2 + unsettled 1) / (5 measured + 3 skipped).
    assert agg["frac_undecidable"] == pytest.approx(4.0 / 8.0)


def test_aggregate_counts_boundary_peaks():
    result = _file_result(
        windows=[
            _win(2048, 128, boundary_peak=True),
            _win(18432, 64),
            _win(34816, -128, locked=False, boundary_peak=True),
            _win(51200, 96),
        ],
        skipped={"far_silent": 0, "low_corr": 0},
    )
    agg = aggregate_results([result])
    # Counted over ALL measured windows (locked or not): a clipped peak
    # anywhere means the search range is too small.
    assert agg["n_windows_boundary_peak"] == 2
    assert agg["frac_boundary_peak"] == pytest.approx(0.5)


def test_aggregate_results_empty():
    agg = aggregate_results([])
    assert agg["n_windows"] == 0
    assert agg["p50_ms"] is None
    assert agg["frac_negative"] is None
    assert agg["frac_boundary_peak"] is None
    assert agg["frac_undecidable"] is None


def test_undecidable_gate():
    ok, _ = undecidable_gate({"frac_undecidable": 0.4}, 0.5)
    assert ok
    # The gate is strictly 'greater than': exactly at the limit passes.
    ok, _ = undecidable_gate({"frac_undecidable": 0.5}, 0.5)
    assert ok
    ok, message = undecidable_gate(
        {
            "frac_undecidable": 0.6,
            "n_windows": 2,
            "n_windows_unsettled": 1,
            "n_windows_skipped_far_silent": 1,
            "n_windows_skipped_low_corr": 1,
        },
        0.5,
    )
    assert not ok
    assert "UNDECIDABLE" in message
    # No windows at all must FAIL, not pass vacuously.
    ok, message = undecidable_gate({"frac_undecidable": None}, 0.5)
    assert not ok
    assert "no analysis windows" in message


def test_undecidable_gate_fires_on_all_unsettled_corpus():
    """End to end through aggregate_results: a corpus where every window is
    unsettled or skipped must fail the default gate."""
    result = _file_result(
        windows=[_win(2048, 64, locked=False), _win(18432, 64, locked=False)],
        skipped={"far_silent": 2, "low_corr": 3},
    )
    agg = aggregate_results([result])
    assert agg["frac_undecidable"] == pytest.approx(1.0)
    ok, _ = undecidable_gate(agg)
    assert not ok


# ---------------------------------------------------------------------------
# Printed labels: every number states its measurement source
# ---------------------------------------------------------------------------

def test_table_labels_proxy_source(capsys):
    _print_table([], aggregate_results([]))
    out = capsys.readouterr().out
    assert "D_hat proxy" in out
    assert "frac_boundary_peak" in out
    assert "frac_undecidable" in out


def test_table_warns_on_boundary_peaks(capsys):
    result = _file_result(
        windows=[_win(2048, 128, boundary_peak=True)],
        skipped={"far_silent": 0, "low_corr": 0},
    )
    _print_table([result], aggregate_results([result]))
    assert "search range is too small" in capsys.readouterr().out


def test_self_test_print_labels_proxy(capsys):
    _print_self_test({"passed": True, "cases": []})
    assert "D_hat proxy" in capsys.readouterr().out


def test_synthetic_echo_print_labels_both_sources(capsys):
    _print_synthetic_echo_test({"passed": True, "cases": []})
    out = capsys.readouterr().out
    assert "true echo" in out
    assert "D_hat proxy" in out


# ---------------------------------------------------------------------------
# Engine-backed self-test (real frozen frontend, short synthetic scenes,
# residuals vs the D_hat proxy)
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
    assert selftest_report["residual_source"] == PROXY_SOURCE


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


# ---------------------------------------------------------------------------
# Generator-backed true-echo test (renders in memory with the repo's own
# AecSequenceRenderer; residual vs audit['echo'] ground truth)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_echo_report():
    """Two short (~6 s) clean far-only sequences, rendered in memory."""
    return run_synthetic_echo_test(n_sequences=2, n_chunks=6, seed=7)


def test_synthetic_echo_test_passes(synthetic_echo_report):
    assert synthetic_echo_report["passed"], synthetic_echo_report
    assert synthetic_echo_report["residual_sources"] == [
        TRUE_ECHO_SOURCE, PROXY_SOURCE,
    ]


def test_true_echo_residual_positive_and_bounded(synthetic_echo_report):
    """Same bounds as the D_hat self-test, but against ground-truth echo."""
    for case in synthetic_echo_report["cases"]:
        assert case["checks"]["locked"], case
        assert case["checks"]["has_post_lock_windows"], case
        assert case["residual_ms_true_echo"] > 0.0, case
        assert case["residual_ms_true_echo"] < 12.0, case


def test_proxy_agrees_with_true_echo(synthetic_echo_report):
    """D_hat is only a proxy; on clean converged scenes it must agree with
    the true echo within one histogram quantum (64 samples)."""
    for case in synthetic_echo_report["cases"]:
        assert case["checks"]["proxy_agrees_with_true_echo"], case
        assert abs(case["proxy_true_delta_samples"]) <= (
            PROXY_TRUE_AGREEMENT_QUANTUM
        ), case
        # Both numbers are reported, each under a source-labelled key.
        assert "residual_ms_proxy" in case
        assert "residual_ms_true_echo" in case


def test_synthetic_echo_windows_stay_off_boundary(synthetic_echo_report):
    for case in synthetic_echo_report["cases"]:
        assert case["checks"]["no_boundary_peaks"], case
