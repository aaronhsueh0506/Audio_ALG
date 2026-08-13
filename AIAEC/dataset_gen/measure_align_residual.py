"""Phase-0 alignment-residual meter for the AEC -> NN seam.

The AIAEC models consume the frozen Python PBFDKF frontend's ``linear_error``
together with a far-end reference. The frontend aligns that reference itself
(delay estimator + ring-buffer compensation), so the question this tool
answers is: after the frontend's own alignment, HOW MUCH residual lag is left
between the far-end the filter actually consumed and the echo path -- i.e.
how far into the past (or, pathologically, the future) would a causal NN have
to look to explain the echo.

Per processed hop it records, read-only from the engine:

* the applied delay (orchestrator ``_current_delay``; -1 = not yet acquired),
* the delay-estimator confidence (0 / 0.5 / 1),
* the ACTUAL aligned far-end the PBFDKF consumed this hop
  (``engine.filter.far_buffer[-hop:]`` -- the post-ring-compensation samples,
  which is the whole point of tapping here instead of re-deriving alignment
  from the reported delay).

Per ~1 s analysis window it cross-correlates that aligned far-end against the
echo estimate ``D_hat = mic - linear_error`` over lags of +-``max_lag``
samples and reports the peak lag.

Sign convention (load-bearing -- keep consistent everywhere):
    ``lag > 0``  means the aligned far-end is EARLY relative to the echo
    path: ``D_hat(t)`` is best explained by ``aligned_far(t - lag)``, so a
    causal NN looks ``lag`` samples into PAST far-end context. This is the
    healthy direction (delay headroom + estimator quantization).
    ``lag < 0``  means the aligned far-end is LATE: explaining the echo would
    require FUTURE far-end samples, which neither the linear filter nor a
    causal NN can see. Negative-lag windows are the alignment failures this
    meter exists to count.

The engine under ``lib/aec`` is used strictly read-only (public ``process``/
``get_formed_output`` plus attribute reads); nothing in ``lib/`` is imported
for mutation and nothing there is modified.

Self-proof: ``--self-test`` builds synthetic scenes internally (white-noise
far, mic = delayed far + short echo tail), asserts the measured residual is
positive and bounded after delay lock, that artificially shifting the tool's
own aligned-far tap by +N samples moves the measurement by ~N, and that an
injected negative lag is reported negative. It needs no corpus and exits
nonzero on failure.

Usage:
    python3 -m AIAEC.dataset_gen.measure_align_residual --self-test
    python3 -m AIAEC.dataset_gen.measure_align_residual \
        --pairs-dir out/train --config AIAEC/dataset_gen/config.ini \
        --limit 20 --json residuals.json
    python3 -m AIAEC.dataset_gen.measure_align_residual \
        --mic mic.wav --far far.wav
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aec_features import STEM_ORDER
from .linear_aec import (
    LinearAecContract,
    LinearAecProcessor,
    linear_aec_contract_from_config,
    make_linear_aec_contract,
)
from .seq_layout import scan_chunks


# Defaults for the residual meter itself (not part of the data contract).
DEFAULT_WINDOW_S = 1.0
DEFAULT_MAX_LAG = 512
# A window whose aligned-far mean-square is below this fraction of the
# whole-sequence mean-square (or below an absolute floor) carries no echo
# excitation and would correlate against noise; skip it.
FAR_ENERGY_RATIO = 1e-4
FAR_ENERGY_FLOOR = 1e-10
# Peak |normalized correlation| below this means the echo estimate in the
# window is not explained by the far-end at ANY searched lag (silence, near
# speech only, or pre-convergence); the lag would be noise, so skip it.
MIN_PEAK_CORR = 0.2


# ---------------------------------------------------------------------------
# Engine hop loop with read-only taps
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EngineRun:
    """Everything tapped from one full-sequence run of the frozen frontend."""

    sample_rate: int
    hop_size: int
    error: np.ndarray            # linear_error (formed output), float32 [n]
    echo_estimate: np.ndarray    # D_hat = mic - error, float32 [n]
    aligned_far: np.ndarray      # far the PBFDKF consumed, float32 [n]
    delay_samples: np.ndarray    # per hop, int64; -1 = not yet acquired
    confidence: np.ndarray       # per hop, float64; NaN if unavailable


def run_linear_aec_with_taps(
    microphone: np.ndarray,
    far_end: np.ndarray,
    contract: LinearAecContract,
) -> EngineRun:
    """Run the frozen PBFDKF hop loop, tapping per-hop alignment state.

    Same seam as ``LinearAecProcessor.process_numpy`` (formed output), but
    with per-hop reads of the applied delay, estimator confidence, and the
    filter's actual aligned far-end input. Inputs are zero-padded at the tail
    to a whole number of hops.
    """
    processor = LinearAecProcessor(contract)
    engine = processor._engine
    hop = processor.hop_size

    microphone = np.asarray(microphone, dtype=np.float32)
    far_end = np.asarray(far_end, dtype=np.float32)
    if microphone.ndim != 1 or microphone.shape != far_end.shape:
        raise ValueError(
            "microphone/far_end must be equal-length 1-D waveforms, got "
            f"{microphone.shape} and {far_end.shape}"
        )
    pad = (-microphone.size) % hop
    if pad:
        microphone = np.pad(microphone, (0, pad))
        far_end = np.pad(far_end, (0, pad))

    n = microphone.size
    n_hops = n // hop
    error = np.empty(n, dtype=np.float32)
    aligned_far = np.empty(n, dtype=np.float32)
    delay_samples = np.empty(n_hops, dtype=np.int64)
    confidence = np.empty(n_hops, dtype=np.float64)

    main_filter = getattr(engine, "filter", None)
    if main_filter is None or not hasattr(main_filter, "far_buffer"):
        raise RuntimeError(
            "engine has no main filter with a far_buffer to tap; the residual "
            "meter requires the PBFDKF frontend"
        )

    for i in range(n_hops):
        start = i * hop
        stop = start + hop
        engine.process(
            np.ascontiguousarray(microphone[start:stop]),
            np.ascontiguousarray(far_end[start:stop]),
        )
        error[start:stop] = engine.get_formed_output()
        # The last hop of the filter's overlap-save far buffer is exactly the
        # aligned (ring-compensated) far-end block this hop's update consumed.
        aligned_far[start:stop] = main_filter.far_buffer[-hop:]
        delay_samples[i] = _applied_delay(engine)
        confidence[i] = _estimator_confidence(engine)

    if not np.isfinite(error).all():
        raise ValueError("linear AEC produced non-finite samples")
    return EngineRun(
        sample_rate=contract.sample_rate,
        hop_size=hop,
        error=error,
        echo_estimate=microphone - error,
        aligned_far=aligned_far,
        delay_samples=delay_samples,
        confidence=confidence,
    )


def _applied_delay(engine) -> int:
    """The orchestrator's currently applied delay in samples (-1 = none)."""
    value = getattr(engine, "_current_delay", None)
    if value is not None:
        return int(value)
    # Fallback: stats report (maps "not acquired" to 0, so prefer the
    # attribute above whenever it exists).
    stats = engine.get_stats()
    return int(getattr(stats, "delay_samples", -1))


def _estimator_confidence(engine) -> float:
    estimator = getattr(engine, "delay_est", None)
    if estimator is None or not hasattr(estimator, "confidence"):
        return float("nan")
    return float(estimator.confidence)


# ---------------------------------------------------------------------------
# Residual-lag measurement (pure numpy; independently testable)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WindowMeasurement:
    start_sample: int
    lag_samples: int
    lag_ms: float
    peak_corr: float
    locked: bool  # every hop overlapping the window had an acquired delay


def shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    """Advance (shift > 0) or delay (shift < 0) a signal, zero-filling.

    ``y[t] = x[t + shift]`` where defined. Advancing the aligned-far tap by
    +N makes the reference N samples earlier, so the measured residual lag
    (see module docstring sign convention) grows by ~N.
    """
    if shift == 0:
        return x
    y = np.zeros_like(x)
    if shift > 0:
        if shift < x.size:
            y[:-shift] = x[shift:]
    else:
        if -shift < x.size:
            y[-shift:] = x[:shift]
    return y


def measure_residual_windows(
    aligned_far: np.ndarray,
    echo_estimate: np.ndarray,
    sample_rate: int,
    *,
    hop_size: int = 0,
    delay_samples: Optional[np.ndarray] = None,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag: int = DEFAULT_MAX_LAG,
    far_tap_shift: int = 0,
    min_peak_corr: float = MIN_PEAK_CORR,
) -> Tuple[List[WindowMeasurement], Dict[str, int]]:
    """Windowed residual lag between the aligned far tap and D_hat.

    Returns ``(measurements, skipped)`` where ``skipped`` counts windows
    dropped for ``far_silent`` (no reference excitation) or ``low_corr``
    (echo estimate not explained by the reference at any searched lag).

    ``far_tap_shift`` artificially shifts the tool's OWN aligned-far tap (see
    :func:`shift_signal`) before measuring; it exists so the self-test can
    prove the meter tracks an injected alignment error. It never touches the
    engine.
    """
    far = np.asarray(aligned_far, dtype=np.float64)
    dhat = np.asarray(echo_estimate, dtype=np.float64)
    if far.shape != dhat.shape or far.ndim != 1:
        raise ValueError("aligned_far/echo_estimate must be equal-length 1-D")
    if far_tap_shift:
        far = shift_signal(far, int(far_tap_shift))

    n = far.size
    window = int(round(window_s * sample_rate))
    if window <= 0:
        raise ValueError(f"window_s={window_s} yields empty window")
    max_lag = int(max_lag)
    global_ms = float(np.mean(far ** 2)) if n else 0.0
    energy_gate = max(FAR_ENERGY_FLOOR, FAR_ENERGY_RATIO * global_ms)

    results: List[WindowMeasurement] = []
    skipped = {"far_silent": 0, "low_corr": 0}
    start = max_lag
    while start + window + max_lag <= n:
        fwin = far[start:start + window]
        far_ms = float(np.mean(fwin ** 2))
        if far_ms < energy_gate:
            skipped["far_silent"] += 1
            start += window
            continue
        dwin = dhat[start:start + window]
        fseg = far[start - max_lag:start + window + max_lag]
        # c[k] = sum_t fseg[k + t] * dwin[t], k in [0, 2*max_lag];
        # lag = max_lag - k, so index 0 is lag=+max_lag (far earliest).
        c = np.correlate(fseg, dwin, mode="valid")
        # Per-lag far energy for normalization (sliding window over fseg).
        csum = np.concatenate(([0.0], np.cumsum(fseg ** 2)))
        e_far = csum[window:] - csum[:-window]
        e_d = float(np.sum(dwin ** 2))
        denom = np.sqrt(np.maximum(e_far * e_d, 1e-30))
        ncc = c / denom
        k = int(np.argmax(np.abs(ncc)))
        peak = float(ncc[k])
        if abs(peak) < min_peak_corr:
            skipped["low_corr"] += 1
            start += window
            continue
        lag = max_lag - k
        locked = False
        if delay_samples is not None and hop_size > 0:
            first_hop = start // hop_size
            last_hop = (start + window - 1) // hop_size
            span = np.asarray(delay_samples)[first_hop:last_hop + 1]
            locked = bool(span.size) and bool(np.all(span >= 0))
        results.append(WindowMeasurement(
            start_sample=start,
            lag_samples=int(lag),
            lag_ms=lag * 1000.0 / sample_rate,
            peak_corr=peak,
            locked=locked,
        ))
        start += window
    return results, skipped


def first_lock_hop(delay_samples: np.ndarray) -> int:
    """First hop index with an acquired delay (>= 0); -1 if never."""
    idx = np.nonzero(np.asarray(delay_samples) >= 0)[0]
    return int(idx[0]) if idx.size else -1


def count_delay_change_events(delay_samples: np.ndarray) -> int:
    """Hops where the applied delay changed AFTER first acquisition.

    Derived by differencing the per-hop applied delay (the orchestrator has
    no cheap per-hop event diagnostic). The initial -1 -> first-value
    transition is the acquisition, reported separately as lock time, so it is
    not counted here.
    """
    d = np.asarray(delay_samples)
    if d.size < 2:
        return 0
    prev, cur = d[:-1], d[1:]
    return int(np.count_nonzero((prev >= 0) & (cur != prev)))


# ---------------------------------------------------------------------------
# Per-file measurement + aggregation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FileResult:
    name: str
    n_hops: int
    lock_hop: int
    lock_ms: float
    delay_change_events: int
    applied_delay_final: int
    confidence_final: float
    windows: List[WindowMeasurement]
    skipped: Dict[str, int]


def measure_pair(
    name: str,
    microphone: np.ndarray,
    far_end: np.ndarray,
    contract: LinearAecContract,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag: int = DEFAULT_MAX_LAG,
) -> FileResult:
    run = run_linear_aec_with_taps(microphone, far_end, contract)
    windows, skipped = measure_residual_windows(
        run.aligned_far,
        run.echo_estimate,
        run.sample_rate,
        hop_size=run.hop_size,
        delay_samples=run.delay_samples,
        window_s=window_s,
        max_lag=max_lag,
    )
    lock = first_lock_hop(run.delay_samples)
    lock_ms = (
        lock * run.hop_size * 1000.0 / run.sample_rate if lock >= 0 else -1.0
    )
    finite_conf = run.confidence[np.isfinite(run.confidence)]
    return FileResult(
        name=name,
        n_hops=int(run.delay_samples.size),
        lock_hop=lock,
        lock_ms=lock_ms,
        delay_change_events=count_delay_change_events(run.delay_samples),
        applied_delay_final=int(run.delay_samples[-1]) if run.delay_samples.size else -1,
        confidence_final=float(finite_conf[-1]) if finite_conf.size else float("nan"),
        windows=windows,
        skipped=skipped,
    )


def aggregate_results(results: Sequence[FileResult]) -> Dict:
    """Corpus-level residual statistics over LOCKED windows.

    Pre-lock windows measure the estimator's acquisition transient, not the
    steady-state seam the NN trains on; they are excluded here and accounted
    for via lock time and the pre-lock window count.
    """
    locked_lags_ms = [
        w.lag_ms for r in results for w in r.windows if w.locked
    ]
    n_prelock = sum(
        1 for r in results for w in r.windows if not w.locked
    )
    lags = np.asarray(locked_lags_ms, dtype=np.float64)
    if lags.size:
        p50, p90, p99 = np.percentile(lags, [50, 90, 99])
        stats = {
            "n_windows": int(lags.size),
            "p50_ms": float(p50),
            "p90_ms": float(p90),
            "p99_ms": float(p99),
            "max_ms": float(np.max(lags)),
            "frac_negative": float(np.mean(lags < 0)),
        }
    else:
        stats = {
            "n_windows": 0,
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "frac_negative": None,
        }
    lock_ms = [r.lock_ms for r in results if r.lock_hop >= 0]
    stats.update({
        "n_files": len(results),
        "n_windows_prelock": n_prelock,
        "n_files_never_locked": sum(1 for r in results if r.lock_hop < 0),
        "lock_ms_median": float(np.median(lock_ms)) if lock_ms else None,
        "lock_ms_max": float(np.max(lock_ms)) if lock_ms else None,
        "delay_change_events_total": int(
            sum(r.delay_change_events for r in results)
        ),
        "n_windows_skipped_far_silent": int(
            sum(r.skipped.get("far_silent", 0) for r in results)
        ),
        "n_windows_skipped_low_corr": int(
            sum(r.skipped.get("low_corr", 0) for r in results)
        ),
    })
    return stats


def _file_result_to_dict(r: FileResult) -> Dict:
    return {
        "name": r.name,
        "n_hops": r.n_hops,
        "lock_hop": r.lock_hop,
        "lock_ms": r.lock_ms,
        "delay_change_events": r.delay_change_events,
        "applied_delay_final": r.applied_delay_final,
        "confidence_final": r.confidence_final,
        "skipped": dict(r.skipped),
        "windows": [dataclasses.asdict(w) for w in r.windows],
    }


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV as float32 [channels, time] plus sample rate."""
    import torchaudio

    audio, sr = torchaudio.load(path)
    return audio.numpy().astype(np.float32), int(sr)


def _resolve_seqs_dir(pairs_dir: str) -> str:
    seqs = os.path.join(pairs_dir, "seqs")
    return seqs if os.path.isdir(seqs) else pairs_dir


def _stem_channels() -> Tuple[int, int]:
    """(far_render, mic_postclip) channel indices from STEM_ORDER."""
    return STEM_ORDER.index("far_render"), STEM_ORDER.index("mic_postclip")


def load_pairs_dir(
    pairs_dir: str, limit: int, expected_sr: int
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(name, mic, far)`` per parent sequence from the WAV layout.

    Chunk files (``SSSSSS_CCC.wav``) are concatenated per sequence so the
    stateful frontend sees each complete parent sequence, matching how
    ``linear_error`` was materialized. Directories of arbitrary 5-channel
    WAVs are accepted too, each file as its own sequence.
    """
    seqs_dir = _resolve_seqs_dir(pairs_dir)
    far_idx, mic_idx = _stem_channels()
    sequences = scan_chunks(seqs_dir)
    entries: List[Tuple[str, List[str]]] = []
    if sequences:
        for sequence_id, paths in sorted(sequences.items()):
            entries.append((f"{sequence_id:06d}", paths))
    else:
        wavs = sorted(
            os.path.join(seqs_dir, f)
            for f in os.listdir(seqs_dir)
            if f.lower().endswith(".wav")
        )
        if not wavs:
            raise FileNotFoundError(f"no .wav files under {seqs_dir}")
        entries = [(os.path.basename(p), [p]) for p in wavs]
    if limit > 0:
        entries = entries[:limit]

    out = []
    for name, paths in entries:
        mics, fars = [], []
        for path in paths:
            audio, sr = _load_wav(path)
            if sr != expected_sr:
                raise ValueError(
                    f"{path}: sample rate {sr} != contract {expected_sr}"
                )
            if audio.shape[0] != len(STEM_ORDER):
                raise ValueError(
                    f"{path}: expected {len(STEM_ORDER)} channels "
                    f"({list(STEM_ORDER)}), got {audio.shape[0]}"
                )
            fars.append(audio[far_idx])
            mics.append(audio[mic_idx])
        out.append((name, np.concatenate(mics), np.concatenate(fars)))
    return out


# ---------------------------------------------------------------------------
# Self-test: the meter must be able to fail
# ---------------------------------------------------------------------------

def synth_echo_scene(
    n_samples: int,
    bulk_delay: int,
    *,
    echo_gain: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """White-noise far-end; mic = far delayed by ``bulk_delay`` through a
    short decaying echo tail, plus a tiny noise floor."""
    rng = np.random.default_rng(seed)
    far = (0.1 * rng.standard_normal(n_samples)).astype(np.float32)
    # Short early-reflection tail (~3 ms) so the scene is not a pure delta.
    tail = np.zeros(64, dtype=np.float32)
    for tap, gain in ((0, 1.0), (12, 0.35), (25, 0.18), (40, 0.08)):
        tail[tap] = gain
    echo = np.convolve(far, tail)[:n_samples].astype(np.float32)
    mic = np.zeros(n_samples, dtype=np.float32)
    if bulk_delay < n_samples:
        mic[bulk_delay:] = echo_gain * echo[:n_samples - bulk_delay]
    mic += (1e-5 * rng.standard_normal(n_samples)).astype(np.float32)
    return mic, far


def _post_lock_windows(
    windows: Sequence[WindowMeasurement],
    lock_sample: int,
    settle_s: float,
    sample_rate: int,
) -> List[WindowMeasurement]:
    threshold = lock_sample + int(settle_s * sample_rate)
    return [w for w in windows if w.locked and w.start_sample >= threshold]


def run_self_test(
    *,
    bulk_delays: Sequence[int] = (1600, 3000, 7000),
    duration_s: float = 8.0,
    sample_rate: int = 16000,
    injected_shift: int = 100,
    settle_s: float = 1.0,
    max_residual_ms: float = 12.0,
    shift_tolerance: int = 16,
    seed: int = 0,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Dict:
    """Prove the meter on synthetic scenes with known ground truth.

    Three properties per bulk delay (see module docstring for context):
      1. after delay lock (+ settle), the measured residual is positive and
         below ``max_residual_ms``;
      2. shifting the tool's own aligned-far tap by +N samples moves the
         measured residual by ~N (the meter tracks an injected error);
      3. an injected negative lag (delaying the tap past the true residual)
         is reported negative.
    Needs no corpus. Returns a report dict with ``passed``.
    """
    contract = make_linear_aec_contract(sample_rate)
    n = int(duration_s * sample_rate)
    cases = []
    passed = True
    for case_index, bulk in enumerate(bulk_delays):
        mic, far = synth_echo_scene(n, int(bulk), seed=seed + case_index)
        run = run_linear_aec_with_taps(mic, far, contract)

        def _measure(tap_shift: int) -> List[WindowMeasurement]:
            windows, _ = measure_residual_windows(
                run.aligned_far,
                run.echo_estimate,
                run.sample_rate,
                hop_size=run.hop_size,
                delay_samples=run.delay_samples,
                window_s=window_s,
                max_lag=max_lag,
                far_tap_shift=tap_shift,
            )
            return windows

        lock = first_lock_hop(run.delay_samples)
        checks: Dict[str, bool] = {"locked": lock >= 0}
        case: Dict = {
            "bulk_delay": int(bulk),
            "lock_hop": lock,
            "applied_delay_final": int(run.delay_samples[-1]),
            "checks": checks,
        }
        if lock >= 0:
            lock_sample = lock * run.hop_size
            base = _post_lock_windows(
                _measure(0), lock_sample, settle_s, sample_rate
            )
            checks["has_post_lock_windows"] = len(base) >= 2
            if base:
                base_lags = np.array([w.lag_samples for w in base])
                residual = float(np.median(base_lags))
                case["residual_samples"] = residual
                case["residual_ms"] = residual * 1000.0 / sample_rate
                checks["residual_positive"] = residual > 0
                checks["residual_below_max"] = (
                    case["residual_ms"] < max_residual_ms
                )

                # (2) meter tracks an injected +N tap shift by ~N.
                shifted = _post_lock_windows(
                    _measure(injected_shift), lock_sample, settle_s,
                    sample_rate,
                )
                if shifted:
                    delta = float(
                        np.median([w.lag_samples for w in shifted])
                    ) - residual
                    case["injected_shift"] = injected_shift
                    case["injected_delta_samples"] = delta
                    checks["injection_tracks"] = (
                        abs(delta - injected_shift) <= shift_tolerance
                    )
                else:
                    checks["injection_tracks"] = False

                # (3) an injected negative lag is reported negative.
                negative_shift = -(int(round(residual)) + 256)
                negated = _post_lock_windows(
                    _measure(negative_shift), lock_sample, settle_s,
                    sample_rate,
                )
                if negated:
                    neg_lag = float(
                        np.median([w.lag_samples for w in negated])
                    )
                    case["negative_shift"] = negative_shift
                    case["negative_lag_samples"] = neg_lag
                    checks["negative_lag_reported_negative"] = neg_lag < 0
                else:
                    checks["negative_lag_reported_negative"] = False
        case_ok = bool(checks) and all(checks.values())
        case["passed"] = case_ok
        passed = passed and case_ok
        cases.append(case)
    return {
        "passed": passed,
        "params": {
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "injected_shift": injected_shift,
            "settle_s": settle_s,
            "max_residual_ms": max_residual_ms,
            "window_s": window_s,
            "max_lag": max_lag,
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(results: Sequence[FileResult], aggregate: Dict) -> None:
    header = (
        f"{'file':<20} {'hops':>6} {'lock_ms':>8} {'events':>6} "
        f"{'windows':>7} {'p50_ms':>8} {'p90_ms':>8} {'max_ms':>8} "
        f"{'neg%':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        lags = np.array([w.lag_ms for w in r.windows if w.locked])
        if lags.size:
            p50 = f"{np.percentile(lags, 50):8.2f}"
            p90 = f"{np.percentile(lags, 90):8.2f}"
            mx = f"{np.max(lags):8.2f}"
            neg = f"{100.0 * np.mean(lags < 0):5.1f}%"
        else:
            p50 = p90 = mx = f"{'-':>8}"
            neg = f"{'-':>6}"
        lock = f"{r.lock_ms:8.1f}" if r.lock_hop >= 0 else f"{'never':>8}"
        print(
            f"{r.name:<20} {r.n_hops:>6} {lock} {r.delay_change_events:>6} "
            f"{lags.size:>7} {p50} {p90} {mx} {neg}"
        )
    print("-" * len(header))
    print("aggregate (locked windows only):")
    for key in (
        "n_files", "n_windows", "p50_ms", "p90_ms", "p99_ms", "max_ms",
        "frac_negative", "lock_ms_median", "lock_ms_max",
        "delay_change_events_total", "n_windows_prelock",
        "n_files_never_locked", "n_windows_skipped_far_silent",
        "n_windows_skipped_low_corr",
    ):
        value = aggregate.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"  {key:<30} {value}")


def _print_self_test(report: Dict) -> None:
    print("self-test:", "PASS" if report["passed"] else "FAIL")
    for case in report["cases"]:
        print(
            f"  bulk_delay={case['bulk_delay']:>5}  "
            f"applied={case.get('applied_delay_final', -1):>5}  "
            f"residual_ms={case.get('residual_ms', float('nan')):6.2f}  "
            f"injected_delta={case.get('injected_delta_samples', float('nan')):7.1f}  "
            f"negative_lag={case.get('negative_lag_samples', float('nan')):7.1f}"
        )
        for name, ok in case["checks"].items():
            if not ok:
                print(f"    FAILED check: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the residual alignment lag at the AEC -> NN seam "
            "(aligned far-end vs echo estimate). See module docstring for "
            "the sign convention."
        )
    )
    source = parser.add_argument_group("input (choose one)")
    source.add_argument(
        "--pairs-dir",
        help="split directory (or its seqs/) with 5-channel dataset WAVs",
    )
    source.add_argument("--mic", help="microphone WAV (with --far)")
    source.add_argument("--far", help="far-end reference WAV (with --mic)")
    parser.add_argument(
        "--config",
        help=(
            "generation config INI; rebuilds the frozen linear-AEC contract "
            "exactly the way the packer does (recommended for --pairs-dir)"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="cap the number of sequences/files measured (0 = all)",
    )
    parser.add_argument("--json", help="write machine-readable results here")
    parser.add_argument(
        "--window-s", type=float, default=DEFAULT_WINDOW_S,
        help="cross-correlation window length in seconds",
    )
    parser.add_argument(
        "--max-lag", type=int, default=DEFAULT_MAX_LAG,
        help="residual lag search range in samples (+-)",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the built-in synthetic proof (no corpus needed)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        report = run_self_test(window_s=args.window_s, max_lag=args.max_lag)
        _print_self_test(report)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)
        return 0 if report["passed"] else 1

    if bool(args.mic) != bool(args.far):
        print("--mic and --far must be given together", file=sys.stderr)
        return 2
    if not args.pairs_dir and not args.mic:
        print(
            "one of --pairs-dir, --mic/--far, or --self-test is required",
            file=sys.stderr,
        )
        return 2
    if args.pairs_dir and args.mic:
        print("--pairs-dir and --mic/--far are exclusive", file=sys.stderr)
        return 2

    if args.config:
        cfg = configparser.ConfigParser()
        if not cfg.read(args.config):
            print(f"cannot read config {args.config}", file=sys.stderr)
            return 2
        contract = linear_aec_contract_from_config(cfg)
    else:
        contract = None  # derived from the first file's sample rate

    pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if args.mic:
        mic_audio, mic_sr = _load_wav(args.mic)
        far_audio, far_sr = _load_wav(args.far)
        if mic_sr != far_sr:
            print(
                f"sample-rate mismatch: mic {mic_sr} vs far {far_sr}",
                file=sys.stderr,
            )
            return 2
        if contract is None:
            contract = make_linear_aec_contract(mic_sr)
        elif contract.sample_rate != mic_sr:
            print(
                f"wav sample rate {mic_sr} != contract "
                f"{contract.sample_rate}",
                file=sys.stderr,
            )
            return 2
        mic, far = mic_audio[0], far_audio[0]
        length = min(mic.size, far.size)
        pairs.append((os.path.basename(args.mic), mic[:length], far[:length]))
    else:
        if contract is None:
            # Peek the first WAV for the rate, then reuse the frozen map.
            seqs_dir = _resolve_seqs_dir(args.pairs_dir)
            wavs = sorted(
                f for f in os.listdir(seqs_dir) if f.lower().endswith(".wav")
            )
            if not wavs:
                print(f"no .wav files under {seqs_dir}", file=sys.stderr)
                return 2
            _, sr = _load_wav(os.path.join(seqs_dir, wavs[0]))
            contract = make_linear_aec_contract(sr)
        pairs = load_pairs_dir(
            args.pairs_dir, args.limit, contract.sample_rate
        )

    if args.limit > 0:
        pairs = pairs[:args.limit]

    results = [
        measure_pair(
            name, mic, far, contract,
            window_s=args.window_s, max_lag=args.max_lag,
        )
        for name, mic, far in pairs
    ]
    aggregate = aggregate_results(results)
    _print_table(results, aggregate)

    if args.json:
        payload = {
            "contract": {
                "sample_rate": contract.sample_rate,
                "frame_size": contract.frame_size,
                "hop_size": contract.hop_size,
                "filter_length": contract.filter_length,
                "version": contract.version,
            },
            "params": {
                "window_s": args.window_s,
                "max_lag": args.max_lag,
            },
            "files": [_file_result_to_dict(r) for r in results],
            "aggregate": aggregate,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
