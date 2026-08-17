import argparse
import dataclasses

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet.sweep_delay_depth import (
    QA_MAX_HEADROOM_MS,
    OfflineBulkDelay,
    _alignment_summary,
    _delay_summary,
    _write_alignment_trace,
    alignment_qa,
    build_parser,
    check_argument_conflicts,
    measure_offline_bulk_delay,
    parse_delay_num_filters,
    parse_depths,
    resolve_delay_num_filters,
    run_streaming_frames,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen.linear_aec import (
    DATASET_DELAY_NUM_FILTERS,
    check_delay_num_filters,
)
from AIAEC.dataset_gen.measure_align_residual import EngineRun


GRID = SignalGrid(16000, 512, 512, 256)


def _spec(frames, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(1, frames, GRID.n_freqs, generator=generator),
        torch.randn(1, frames, GRID.n_freqs, generator=generator),
    )


def test_parse_depths_preserves_order_and_removes_duplicates():
    assert parse_depths('64, 16,8,16') == [64, 16, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths('64,zero')
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths('8,0')
    with pytest.raises(argparse.ArgumentTypeError):
        parse_depths(' , ')


def test_streaming_sweep_uses_requested_depth_and_reports_state_memory():
    torch.manual_seed(0)
    large = AlignULCNet(GRID, max_delay_frames=8).eval()
    small = AlignULCNet(GRID, max_delay_frames=4).eval()
    small.load_state_dict(large.state_dict(), strict=True)
    error = _spec(12, 1)
    far = _spec(12, 2)

    result_large = run_streaming_frames(large, error, far)
    result_small = run_streaming_frames(small, error, far)

    assert result_large.enhanced.shape == error.shape
    assert result_small.enhanced.shape == error.shape
    assert result_large.delay_distribution.shape == (1, 12, 8)
    assert result_small.delay_distribution.shape == (1, 12, 4)
    assert result_large.state_bytes > result_small.state_bytes > 0
    assert result_large.elapsed_seconds >= 0.0
    assert torch.isfinite(result_small.enhanced.real).all()

    summary = _delay_summary(
        result_small.delay_distribution, depth=4,
        hop_seconds=GRID.hop_len / GRID.sample_rate,
    )
    assert summary['evaluated_frames'] == 9.0
    assert 0.0 <= summary['argmax_at_max_depth_rate'] <= 1.0
    assert 0.0 <= summary['max_depth_probability_mean'] <= 1.0


def test_alignment_trace_reports_applied_delay_and_acquisition(tmp_path):
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=np.zeros(4 * 256, dtype=np.float32),
        echo_estimate=np.zeros(4 * 256, dtype=np.float32),
        aligned_far=np.zeros(4 * 256, dtype=np.float32),
        delay_samples=np.asarray([-1, 7936, 7936, 8000], dtype=np.int64),
        confidence=np.asarray([0.0, 0.5, 1.0, 1.0], dtype=np.float64),
    )

    summary = _alignment_summary(run)
    assert summary['aec_delay_acquired'] is True
    assert summary['aec_first_acquired_ms'] == 32.0
    assert summary['aec_initial_delay_samples'] == 7936
    assert summary['aec_final_delay_samples'] == 8000
    assert summary['aec_final_delay_ms'] == 500.0
    assert summary['aec_delay_change_events'] == 1
    assert summary['aec_final_confidence'] == 1.0

    path = tmp_path / 'aec_alignment.csv'
    _write_alignment_trace(path, run)
    rows = path.read_text(encoding='utf-8').splitlines()
    assert rows[0].startswith('hop,time_ms,applied_delay_samples')
    assert rows[1].split(',')[2:] == ['-1', 'nan', '0.000000000', '0']
    assert rows[-1].split(',')[1:4] == ['64.000000', '8000', '500.000000']


def test_parse_delay_num_filters_enforces_the_bank_range():
    assert [parse_delay_num_filters(str(n)) for n in (1, 3, 5)] == [1, 3, 5]
    for bad in ('0', '6', '-1', 'five', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_delay_num_filters(bad)
    # The canonical validator itself, asserted directly: values that int()
    # would silently truncate or that only masquerade as integers.
    for bad in (True, np.bool_(True), 1.5, np.float32(2.5), float('inf')):
        with pytest.raises(ValueError):
            check_delay_num_filters(bad)


def test_delay_num_filters_defaults_to_the_corpus_bank_size():
    """An unflagged run must reproduce the frontend the corpus was made with.

    The parsed default stays ``None`` rather than the number itself: the tool
    has to be able to tell "did not ask" from "asked for 5" so that combining
    the flag with the PBFDKF bypass can be refused, and the resolution to the
    corpus size happens afterwards.
    """
    args = build_parser().parse_args(['c.pth', 'm.wav', 'f.wav', 'out'])
    assert args.delay_num_filters is None
    assert resolve_delay_num_filters(None) == DATASET_DELAY_NUM_FILTERS
    assert resolve_delay_num_filters(2) == 2
    # Programmatic (non-argparse) callers get the validator's own ValueError;
    # only the argparse `type=` adapter re-dresses it as ArgumentTypeError.
    with pytest.raises(ValueError):
        resolve_delay_num_filters(6)


def test_bank_size_override_is_refused_with_the_pbfdkf_bypass():
    """--input-is-linear-error has no bank to size, so the pair must fail fast.

    Both flags parse -- the conflict is between two individually valid ones --
    so it is checked before anything is loaded, and asserted here through the
    same entry ``main`` calls first.
    """
    parser = build_parser()

    conflicting = parser.parse_args([
        'c.pth', 'm.wav', 'f.wav', 'out',
        '--input-is-linear-error', '--delay-num-filters', '2',
    ])
    with pytest.raises(ValueError, match='delay-num-filters'):
        check_argument_conflicts(conflicting)

    # Either flag alone is legitimate, so the check must not reject them.
    for argv in (
        ['--input-is-linear-error'],
        ['--delay-num-filters', '2'],
        [],
    ):
        check_argument_conflicts(
            parser.parse_args(['c.pth', 'm.wav', 'f.wav', 'out'] + argv)
        )


def test_offline_bulk_delay_recovers_a_known_delay_without_the_estimator():
    rng = np.random.default_rng(5)
    sample_rate = 16000
    far = (0.2 * rng.standard_normal(6 * sample_rate)).astype(np.float32)
    delay = 1600
    mic = np.zeros_like(far)
    mic[delay:] = 0.5 * far[:-delay]

    measured = measure_offline_bulk_delay(
        mic, far, sample_rate, max_lag=8192,
    )
    assert measured.bulk_delay_samples == delay
    assert measured.n_boundary_peak == 0
    assert measured.n_windows >= 2


def _engine_run_with_applied_delay(applied: int, aligned_lead: int):
    """A synthetic EngineRun whose aligned far leads the echo by a known lag."""
    rng = np.random.default_rng(9)
    sample_rate, hop = 16000, 256
    n = 6 * sample_rate
    far = (0.2 * rng.standard_normal(n)).astype(np.float32)
    echo = np.zeros_like(far)
    echo[aligned_lead:] = 0.5 * far[:n - aligned_lead]
    return EngineRun(
        sample_rate=sample_rate,
        hop_size=hop,
        error=np.zeros(n, dtype=np.float32),
        echo_estimate=echo,
        aligned_far=far,
        delay_samples=np.full(n // hop, applied, dtype=np.int64),
        confidence=np.ones(n // hop, dtype=np.float64),
    ), far, echo


def test_alignment_qa_accepts_a_delay_that_agrees_with_the_offline_truth():
    true_delay = 1600
    run, far, echo = _engine_run_with_applied_delay(
        applied=true_delay - 64, aligned_lead=64,
    )
    # The mic the offline measurement sees is the echo delayed by the bulk
    # delay the estimator claims to have removed.
    mic = np.zeros_like(far)
    mic[true_delay - 64:] = echo[:far.size - (true_delay - 64)]

    qa = alignment_qa(run, mic, far, max_lag=8192)
    assert qa['qa_status'] == 'ok', qa
    assert qa['qa_valid'] is True
    assert 0.0 < qa['qa_residual_p50_ms'] < 10.0
    assert abs(qa['qa_applied_vs_offline_ms']) <= QA_MAX_HEADROOM_MS


def test_alignment_qa_marks_a_confidently_wrong_delay_invalid():
    """The gate must reject a lock the raw signals contradict.

    Same clip as above, but the estimator reports a delay far from the one the
    raw correlation measures -- the shape of the mis-lock this QA exists to
    keep out of a delay-profile statistic. Both directions are checked: a
    reference left too early (a large positive residual the model's attention
    could not span) and one pushed too late (a negative residual no causal
    filter can explain).
    """
    true_delay = 1600
    offline = None
    for applied in (200, true_delay + 1600):
        run, far, echo = _engine_run_with_applied_delay(
            applied=applied, aligned_lead=64,
        )
        mic = np.zeros_like(far)
        mic[true_delay - 64:] = echo[:far.size - (true_delay - 64)]

        # The raw signals are identical across the loop (fixed seed); only
        # the claimed applied delay varies, so measure offline truth once.
        if offline is None:
            offline = measure_offline_bulk_delay(mic, far, 16000, max_lag=8192)
        qa = alignment_qa(run, mic, far, max_lag=8192, offline=offline)
        assert qa['qa_status'] == 'mislock', (applied, qa)
        assert qa['qa_valid'] is False


def test_alignment_qa_rejects_failure_to_acquire_an_in_range_delay():
    rng = np.random.default_rng(12)
    sample_rate, hop, true_delay = 16000, 256, 1280
    far = (0.2 * rng.standard_normal(6 * sample_rate)).astype(np.float32)
    mic = np.zeros_like(far)
    mic[true_delay:] = 0.5 * far[:-true_delay]
    run = EngineRun(
        sample_rate=sample_rate,
        hop_size=hop,
        error=np.zeros_like(far),
        echo_estimate=mic.copy(),
        aligned_far=far.copy(),
        delay_samples=np.full(far.size // hop, -1, dtype=np.int64),
        confidence=np.zeros(far.size // hop, dtype=np.float64),
        delay_num_filters=1,  # reliable reach 125 ms; true delay is 80 ms
    )

    qa = alignment_qa(run, mic, far, max_lag=8192)
    assert qa['qa_status'] == 'not_acquired_in_range', qa
    assert qa['qa_valid'] is False


def test_alignment_qa_fails_closed_when_the_bank_size_is_unknown():
    """A run with no legal bank size has no reach to judge against.

    Synthetic ``EngineRun``s default to ``delay_num_filters=0``. A
    never-acquired clip on such a run must come out ``undecidable``/invalid,
    not fall through to the VALID ``not_acquired`` arm -- that would average
    an unjudgeable clip into the profile statistic as a fail-open
    observation.
    """
    silence = np.zeros(1024, dtype=np.float32)
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=silence.copy(),
        echo_estimate=silence.copy(),
        aligned_far=silence.copy(),
        delay_samples=np.full(4, -1, dtype=np.int64),
        confidence=np.zeros(4, dtype=np.float64),
    )
    offline = OfflineBulkDelay(
        n_windows=3, n_boundary_peak=0, bulk_delay_samples=1280,
    )

    qa = alignment_qa(run, silence, silence, max_lag=8192, offline=offline)
    assert qa['qa_expected_in_range'] is False
    assert qa['qa_status'] == 'undecidable', qa
    assert qa['qa_valid'] is False


def test_alignment_qa_does_not_treat_negative_bulk_lag_as_in_range():
    silence = np.zeros(1024, dtype=np.float32)
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=silence.copy(),
        echo_estimate=silence.copy(),
        aligned_far=silence.copy(),
        delay_samples=np.full(4, -1, dtype=np.int64),
        confidence=np.zeros(4, dtype=np.float64),
        delay_num_filters=5,
    )
    offline = OfflineBulkDelay(
        n_windows=3, n_boundary_peak=0, bulk_delay_samples=-160,
    )

    qa = alignment_qa(run, silence, silence, max_lag=8192, offline=offline)
    assert qa['qa_expected_in_range'] is False
    assert qa['qa_status'] == 'undecidable'
    assert qa['qa_valid'] is False

    # Same negative offline verdict with an ACQUIRED delay: also undecidable
    # (before the negative-lag guard this arm reported 'mislock'; either way
    # qa_valid stays False, but the changed diagnosis in summary.csv is
    # intended, so pin it).
    acquired_run = dataclasses.replace(
        run, delay_samples=np.full(4, 160, dtype=np.int64)
    )
    acquired = alignment_qa(
        acquired_run, silence, silence, max_lag=8192, offline=offline,
    )
    assert acquired['qa_status'] == 'undecidable'
    assert acquired['qa_valid'] is False


def test_alignment_summary_reports_never_acquired():
    run = EngineRun(
        sample_rate=16000,
        hop_size=256,
        error=np.zeros(512, dtype=np.float32),
        echo_estimate=np.zeros(512, dtype=np.float32),
        aligned_far=np.zeros(512, dtype=np.float32),
        delay_samples=np.asarray([-1, -1], dtype=np.int64),
        confidence=np.asarray([0.0, 0.0], dtype=np.float64),
    )
    summary = _alignment_summary(run)
    assert summary['aec_delay_acquired'] is False
    assert summary['aec_final_delay_samples'] == -1
    assert np.isnan(summary['aec_first_acquired_ms'])
    assert np.isnan(summary['aec_final_delay_ms'])
