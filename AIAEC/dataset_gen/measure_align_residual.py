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

Per ~1 s analysis window it cross-correlates that aligned far-end against an
echo target over lags of +-``max_lag`` samples (default 2048 = +-128 ms at
16 kHz, wide enough to bound a D=8 hop attention span) and reports the peak
lag.

MEASUREMENT SOURCES (every reported number is labelled with one of these):

* ``d_hat_proxy`` -- the target is ``D_hat = mic - linear_error``, the
  PBFDKF's OWN echo estimate. This is what the corpus/--mic modes and
  ``--self-test`` measure; it is a valid proxy only once the filter has
  converged, which is exactly why the tool also has:
* ``true_echo`` -- ``--synthetic-echo-test`` renders short sequences in
  memory with the repo's own generator (``aec_dataset.AecSequenceRenderer``)
  and measures the SAME aligned-far tap against the rendered ground-truth
  echo (``RenderedSequence.audit['echo']``), asserting the post-lock residual
  is positive and < 12 ms and that the proxy agrees with the true echo within
  one histogram quantum (64 samples). Source WAV fixtures live in an
  ephemeral temporary directory; no corpus is written to disk.

Sign convention (load-bearing -- keep consistent everywhere):
    ``lag > 0``  means the aligned far-end is EARLY relative to the echo
    path: the target is best explained by ``aligned_far(t - lag)``, so a
    causal NN looks ``lag`` samples into PAST far-end context. This is the
    healthy direction (delay headroom + estimator quantization).
    ``lag < 0``  means the aligned far-end is LATE: explaining the echo would
    require FUTURE far-end samples, which neither the linear filter nor a
    causal NN can see. Negative-lag windows are the alignment failures this
    meter exists to count.

TRUST GATES (a report full of unmeasurable windows must never read as a pass):

* A window only counts as LOCKED when the estimator confidence is solid
  (>= 1.0 when the run's confidence scale reaches 1.0, else >= 0.5), the
  applied delay is unchanged across the whole window, AND at least
  ``--settle-s`` (default 0.5 s) has elapsed since the last delay change.
  Windows that measured a lag but fail this gate are counted as
  ``n_windows_unsettled`` and excluded from the residual statistics.
* ``frac_boundary_peak`` is the fraction of measured windows whose
  correlation peak lands ON the +-``max_lag`` search boundary; any nonzero
  value means the search range is too small and the report says so.
* ``frac_undecidable`` = (silent + low-correlation + unsettled) / total
  windows. ``--max-undecidable`` (default 0.5) makes the CLI exit nonzero
  when it is exceeded.

The engine under ``lib/aec`` is used strictly read-only (public ``process``/
``get_formed_output`` plus attribute reads); nothing in ``lib/`` is imported
for mutation and nothing there is modified.

Self-proof: ``--self-test`` builds synthetic scenes internally (white-noise
far, mic = delayed far + short echo tail), asserts the measured residual
(vs the D_hat proxy) is positive and bounded after delay lock, that
artificially shifting the tool's own aligned-far tap by +N samples moves the
measurement by ~N, and that an injected negative lag is reported negative.
It needs no corpus and exits nonzero on failure.

Usage:
    python3 -m AIAEC.dataset_gen.measure_align_residual --self-test
    python3 -m AIAEC.dataset_gen.measure_align_residual --synthetic-echo-test
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
# +-2048 samples = +-128 ms at 16 kHz: wide enough to bound a D=8 hop
# (8 * 256 = 2048 samples = 128 ms) attention span. The old +-512 could not.
DEFAULT_MAX_LAG = 2048
# Locked-window gate: minimum time since the last applied-delay change.
DEFAULT_SETTLE_S = 0.5
# CLI gate: maximum tolerated fraction of undecidable windows.
DEFAULT_MAX_UNDECIDABLE = 0.5
# One histogram quantum: the proxy and true-echo residuals must agree within
# this many samples on clean synthetic scenes for the proxy to be trusted.
PROXY_TRUE_AGREEMENT_QUANTUM = 64
# A window whose aligned-far mean-square is below this fraction of the
# whole-sequence mean-square (or below an absolute floor) carries no echo
# excitation and would correlate against noise; skip it.
FAR_ENERGY_RATIO = 1e-4
FAR_ENERGY_FLOOR = 1e-10
# Peak |normalized correlation| below this means the echo target in the
# window is not explained by the far-end at ANY searched lag (silence, near
# speech only, or pre-convergence); the lag would be noise, so skip it.
MIN_PEAK_CORR = 0.2

# Measurement-source labels: every reported number carries one of these.
PROXY_SOURCE = "d_hat_proxy"
TRUE_ECHO_SOURCE = "true_echo"
PROXY_SOURCE_HUMAN = "D_hat proxy (mic - linear_error)"
TRUE_ECHO_SOURCE_HUMAN = "true echo (renderer audit['echo'])"


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
    locked: bool         # passed the confidence + stable-delay + settle gate
    boundary_peak: bool  # correlation peak sits ON the +-max_lag boundary


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


def last_change_hops(delay_samples: np.ndarray) -> np.ndarray:
    """Per hop, the most recent hop index at which the applied delay changed.

    Hop 0 always counts as a change point: the run carries no history before
    its first hop, so the settle clock can start no earlier than the run
    itself. Acquisition (-1 -> value) and every later value change reset the
    clock too.
    """
    d = np.asarray(delay_samples)
    if d.size == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.ones(d.size, dtype=bool)
    if d.size > 1:
        change[1:] = d[1:] != d[:-1]
    idx = np.where(change, np.arange(d.size, dtype=np.int64), 0)
    return np.maximum.accumulate(idx)


def solid_confidence_threshold(confidence: np.ndarray) -> float:
    """The per-hop confidence a locked window must sustain.

    The estimator reports 0 / 0.5 (coarse) / 1.0 (refined). Require >= 1.0
    whenever this run's confidence scale demonstrably reaches 1.0; on an
    engine whose confidence never reports 1.0, fall back to >= 0.5 so the
    gate does not silently reject every window.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    finite = conf[np.isfinite(conf)]
    if finite.size and float(np.max(finite)) >= 1.0:
        return 1.0
    return 0.5


def measure_residual_windows(
    aligned_far: np.ndarray,
    echo_target: np.ndarray,
    sample_rate: int,
    *,
    hop_size: int = 0,
    delay_samples: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag: int = DEFAULT_MAX_LAG,
    far_tap_shift: int = 0,
    min_peak_corr: float = MIN_PEAK_CORR,
    settle_s: float = DEFAULT_SETTLE_S,
) -> Tuple[List[WindowMeasurement], Dict[str, int]]:
    """Windowed residual lag between the aligned far tap and ``echo_target``.

    ``echo_target`` is either the engine's own ``D_hat`` (proxy) or a
    rendered true echo -- the measurement is identical; the CALLER is
    responsible for labelling the result with its source.

    Returns ``(measurements, skipped)`` where ``skipped`` counts windows
    dropped for ``far_silent`` (no reference excitation) or ``low_corr``
    (echo target not explained by the reference at any searched lag).

    A window is ``locked`` only when, over every hop it overlaps: the applied
    delay is acquired AND constant, the estimator confidence is solid (see
    :func:`solid_confidence_threshold`; hops with no confidence signal cannot
    be gated and pass vacuously), and at least ``settle_s`` has elapsed
    between the last applied-delay change and the window start.

    ``far_tap_shift`` artificially shifts the tool's OWN aligned-far tap (see
    :func:`shift_signal`) before measuring; it exists so the self-test can
    prove the meter tracks an injected alignment error. It never touches the
    engine.
    """
    far = np.asarray(aligned_far, dtype=np.float64)
    dhat = np.asarray(echo_target, dtype=np.float64)
    if far.shape != dhat.shape or far.ndim != 1:
        raise ValueError("aligned_far/echo_target must be equal-length 1-D")
    if far_tap_shift:
        far = shift_signal(far, int(far_tap_shift))

    n = far.size
    window = int(round(window_s * sample_rate))
    if window <= 0:
        raise ValueError(f"window_s={window_s} yields empty window")
    max_lag = int(max_lag)
    global_ms = float(np.mean(far ** 2)) if n else 0.0
    energy_gate = max(FAR_ENERGY_FLOOR, FAR_ENERGY_RATIO * global_ms)

    gate_ready = delay_samples is not None and hop_size > 0
    if gate_ready:
        delays = np.asarray(delay_samples)
        change_hops = last_change_hops(delays)
        settle_samples = int(round(settle_s * sample_rate))
        conf = None
        solid = 0.0
        if confidence is not None:
            conf = np.asarray(confidence, dtype=np.float64)
            if conf.shape != delays.shape:
                raise ValueError(
                    "confidence and delay_samples must be per-hop arrays of "
                    f"equal length, got {conf.shape} and {delays.shape}"
                )
            solid = solid_confidence_threshold(conf)

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
        # A peak sitting exactly ON the search boundary is indistinguishable
        # from a clipped out-of-range peak; any such window means the search
        # range cannot be trusted to bound the residual.
        boundary_peak = abs(lag) == max_lag
        locked = False
        if gate_ready:
            first_hop = start // hop_size
            last_hop = (start + window - 1) // hop_size
            span = delays[first_hop:last_hop + 1]
            # Acquired AND constant across the whole window.
            locked = (
                bool(span.size)
                and bool(np.all(span >= 0))
                and int(np.max(span)) == int(np.min(span))
            )
            if locked:
                # Settle: the delay in force must have been established at
                # least settle_s before the window starts.
                change_sample = int(change_hops[first_hop]) * hop_size
                locked = (start - change_sample) >= settle_samples
            if locked and conf is not None:
                cwin = conf[first_hop:last_hop + 1]
                finite = cwin[np.isfinite(cwin)]
                # Hops with no confidence signal cannot be gated on it.
                locked = finite.size == 0 or bool(np.all(finite >= solid))
        results.append(WindowMeasurement(
            start_sample=start,
            lag_samples=int(lag),
            lag_ms=lag * 1000.0 / sample_rate,
            peak_corr=peak,
            locked=locked,
            boundary_peak=boundary_peak,
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
    settle_s: float = DEFAULT_SETTLE_S,
) -> FileResult:
    run = run_linear_aec_with_taps(microphone, far_end, contract)
    windows, skipped = measure_residual_windows(
        run.aligned_far,
        run.echo_estimate,
        run.sample_rate,
        hop_size=run.hop_size,
        delay_samples=run.delay_samples,
        confidence=run.confidence,
        window_s=window_s,
        max_lag=max_lag,
        settle_s=settle_s,
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

    Unsettled windows (pre-lock, mid-delay-change, low confidence, or inside
    the post-change settle time) measure the estimator's transient, not the
    steady-state seam the NN trains on; they are excluded from the lag
    statistics and accounted for via ``n_windows_unsettled`` and
    ``frac_undecidable``.

    The aggregation itself is source-agnostic: the caller labels the result
    with whichever echo target (D_hat proxy or true echo) produced it.
    """
    measured = [w for r in results for w in r.windows]
    locked_lags_ms = [w.lag_ms for w in measured if w.locked]
    n_unsettled = sum(1 for w in measured if not w.locked)
    n_boundary = sum(1 for w in measured if w.boundary_peak)
    n_silent = int(sum(r.skipped.get("far_silent", 0) for r in results))
    n_low_corr = int(sum(r.skipped.get("low_corr", 0) for r in results))
    n_total = len(measured) + n_silent + n_low_corr

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
        "n_windows_unsettled": int(n_unsettled),
        "n_files_never_locked": sum(1 for r in results if r.lock_hop < 0),
        "lock_ms_median": float(np.median(lock_ms)) if lock_ms else None,
        "lock_ms_max": float(np.max(lock_ms)) if lock_ms else None,
        "delay_change_events_total": int(
            sum(r.delay_change_events for r in results)
        ),
        "n_windows_skipped_far_silent": n_silent,
        "n_windows_skipped_low_corr": n_low_corr,
        # Fraction of MEASURED windows whose peak sits on the +-max_lag
        # boundary; nonzero means the search range is too small.
        "n_windows_boundary_peak": int(n_boundary),
        "frac_boundary_peak": (
            float(n_boundary) / len(measured) if measured else None
        ),
        # Windows that produced no trustworthy number, over everything the
        # meter looked at. None when there were no windows at all -- which
        # the --max-undecidable gate treats as a failure, not a pass.
        "frac_undecidable": (
            float(n_silent + n_low_corr + n_unsettled) / n_total
            if n_total else None
        ),
    })
    return stats


def undecidable_gate(
    aggregate: Dict, max_undecidable: float = DEFAULT_MAX_UNDECIDABLE
) -> Tuple[bool, str]:
    """(ok, message) for the undecidable-fraction gate.

    Fails when ``frac_undecidable`` exceeds ``max_undecidable`` and also when
    there were no analysis windows at all -- either way the report carries too
    few trustworthy numbers to be read as a pass.
    """
    frac = aggregate.get("frac_undecidable")
    if frac is None:
        return False, (
            "UNDECIDABLE GATE FAILED: no analysis windows were measured at "
            "all; this report cannot justify any residual bound."
        )
    if frac > max_undecidable:
        return False, (
            f"UNDECIDABLE GATE FAILED: frac_undecidable={frac:.3f} > "
            f"--max-undecidable {max_undecidable:.3f} "
            f"(far_silent={aggregate.get('n_windows_skipped_far_silent')}, "
            f"low_corr={aggregate.get('n_windows_skipped_low_corr')}, "
            f"unsettled={aggregate.get('n_windows_unsettled')}, "
            f"locked={aggregate.get('n_windows')}). Most windows carry no "
            "measurable residual; this report must not be read as a pass."
        )
    return True, (
        f"undecidable fraction {frac:.3f} <= {max_undecidable:.3f}"
    )


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

    Residuals here are measured against the D_HAT PROXY (mic - linear_error),
    the same seam the corpus modes use; ``run_synthetic_echo_test`` is the
    true-echo counterpart. Three properties per bulk delay (see module
    docstring for context):
      1. after the locked-window gate (stable delay + solid confidence +
         settle), the measured residual is positive and below
         ``max_residual_ms``;
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

        def _measure_locked(tap_shift: int) -> List[WindowMeasurement]:
            windows, _ = measure_residual_windows(
                run.aligned_far,
                run.echo_estimate,
                run.sample_rate,
                hop_size=run.hop_size,
                delay_samples=run.delay_samples,
                confidence=run.confidence,
                window_s=window_s,
                max_lag=max_lag,
                far_tap_shift=tap_shift,
                settle_s=settle_s,
            )
            return [w for w in windows if w.locked]

        lock = first_lock_hop(run.delay_samples)
        checks: Dict[str, bool] = {"locked": lock >= 0}
        case: Dict = {
            "bulk_delay": int(bulk),
            "lock_hop": lock,
            "applied_delay_final": int(run.delay_samples[-1]),
            "checks": checks,
        }
        if lock >= 0:
            base = _measure_locked(0)
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
                shifted = _measure_locked(injected_shift)
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
                negated = _measure_locked(negative_shift)
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
        "residual_source": PROXY_SOURCE,
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
# Synthetic true-echo test: prove the D_hat proxy against rendered ground
# truth from the repo's own generator (RenderedSequence.audit['echo'])
# ---------------------------------------------------------------------------

def _write_synthetic_sources(root: str, sample_rate: int) -> None:
    """A handful of tiny source WAVs (speech / noise / RIR) for the renderer.

    These are throwaway EXCITATION fixtures in an ephemeral temp directory --
    the rendered sequences themselves never touch disk. Recipes mirror the
    generator's own test fixtures: bursty band-limited noise as speech, plain
    noise beds, and an exponentially decaying RIR with a unit direct path.
    """
    import torch
    import torchaudio

    def write(rel: str, audio: "torch.Tensor") -> None:
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(
            path, audio.unsqueeze(0), sample_rate,
            encoding="PCM_F", bits_per_sample=32,
        )

    generator = torch.Generator().manual_seed(20260813)

    def speechlike(n: int) -> "torch.Tensor":
        base = torch.randn(n, generator=generator)
        smooth = torch.nn.functional.avg_pool1d(
            base.view(1, 1, -1), kernel_size=5, stride=1, padding=2
        ).view(-1)
        t = torch.arange(n, dtype=torch.float32) / sample_rate
        envelope = (
            0.5 + 0.5 * torch.sin(2 * math.pi * 1.7 * t)
        ).clamp_min(0.05)
        return (smooth * envelope * 0.2)[:n]

    def rir(n: int, rt60: float) -> "torch.Tensor":
        t = torch.arange(n, dtype=torch.float32) / sample_rate
        decay = torch.exp(-6.9078 * t / rt60)
        out = torch.randn(n, generator=generator) * decay
        out[0] = 1.0  # a clear direct path
        return out * 0.5

    for speaker in range(2):
        write(
            os.path.join("speech", f"reader_{speaker:03d}", "take_0.wav"),
            speechlike(8 * sample_rate),
        )
    write(
        os.path.join("noise", "noise_00.wav"),
        torch.randn(4 * sample_rate, generator=generator) * 0.05,
    )
    write(os.path.join("rir", "room_00", "rir_0.wav"),
          rir(int(0.25 * sample_rate), 0.2))


def _synthetic_render_config(
    root: str, sample_rate: int
) -> configparser.ConfigParser:
    """The generator config for the true-echo proof: clean, linear scenes.

    Starts from the package's own config.example.ini so the renderer runs
    under its real schema, then pins the knobs that make the case CLEAN --
    linear loudspeakers, no clipping/AGC, near-silent noise bed, strong echo
    -- because the proof is about alignment, not robustness.
    """
    cfg = configparser.ConfigParser()
    example = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.example.ini"
    )
    if not cfg.read(example):
        raise FileNotFoundError(f"cannot read {example}")
    cfg.set("signal", "sr", str(sample_rate))
    cfg.set("paths", "speech_dir", os.path.join(root, "speech"))
    cfg.set("paths", "noise_dir", os.path.join(root, "noise"))
    cfg.set("paths", "rir_dir", os.path.join(root, "rir"))
    # 1.024 s chunks stay hop-exact on both frozen grids (256 @16k, 512 @48k).
    cfg.set("sequence", "seq_sec_min", "2.048")
    cfg.set("sequence", "seq_sec_max", "8.192")
    cfg.set("sequence", "chunk_sec", "1.024")
    cfg.set("rir", "rt60_min", "0.05")
    cfg.set("rir", "rt60_max", "2.0")
    # Clean-case pins: the true-echo bound is only claimed for scenes the
    # linear filter can actually converge on.
    cfg.set("devices", "nonlinear_models", "linear")
    cfg.set("mic", "p_clipping", "0")
    cfg.set("mic", "p_agc", "0")
    cfg.set("levels", "noise_level_dbfs_min", "-70")
    cfg.set("levels", "noise_level_dbfs_max", "-68")
    cfg.set("levels", "erl_db_min", "5")
    cfg.set("levels", "erl_db_max", "12")
    cfg.set("activity", "far_talk_sec_mean", "8.0")
    cfg.set("activity", "far_gap_sec_mean", "0.4")
    cfg.set("echo_path", "bulk_delay_ms_min", "40")
    cfg.set("echo_path", "bulk_delay_ms_max", "120")
    return cfg


def render_synthetic_sequences(
    n_sequences: int, n_chunks: int, sample_rate: int, seed: int
) -> List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray,
                LinearAecContract]]:
    """Render short far-only sequences fully in memory via the repo generator.

    Returns ``(name, bulk_delay_samples, mic, far, true_echo, contract)`` per
    sequence, where ``true_echo`` is ``RenderedSequence.audit['echo']`` --
    the ground-truth echo actually mixed into the mic. Source fixtures live
    in a temporary directory that is deleted before returning; no corpus is
    written to disk and nothing under ``lib/`` is touched.
    """
    # Deferred: the renderer pulls in torch/torchaudio, which are slow to
    # import and unneeded by every other mode of this tool.
    import tempfile

    from .aec_dataset import AecSequenceRenderer, SequencePlan, stable_seed
    from .manifest import build_unified_manifest, pools_for_split

    far_idx, mic_idx = _stem_channels()
    out = []
    with tempfile.TemporaryDirectory(prefix="align_residual_fixture_") as tmp:
        _write_synthetic_sources(tmp, sample_rate)
        cfg = _synthetic_render_config(tmp, sample_rate)
        manifest = build_unified_manifest(cfg, seed=seed, progress=False)
        renderer = AecSequenceRenderer(
            cfg, pools_for_split(manifest, "all"), corpus_seed=seed
        )
        for index in range(n_sequences):
            plan = SequencePlan(
                sequence_id=index,
                n_chunks=n_chunks,
                # far_only: the mic is echo + a near-silent noise bed, so the
                # true-echo correlation is not diluted by near speech.
                scenario="far_only",
                seed=stable_seed(seed, "align-residual-true-echo", index),
            )
            rendered = renderer.render(plan)
            out.append((
                f"synthetic_{index:03d}",
                int(rendered.chunk_meta[0]["bulk_delay_samples"]),
                rendered.stems[mic_idx].numpy().astype(np.float32),
                rendered.stems[far_idx].numpy().astype(np.float32),
                rendered.audit["echo"].numpy().astype(np.float32),
                renderer.linear_aec_contract,
            ))
    return out


def run_synthetic_echo_test(
    *,
    n_sequences: int = 2,
    n_chunks: int = 6,
    sample_rate: int = 16000,
    seed: int = 7,
    settle_s: float = 1.0,
    max_residual_ms: float = 12.0,
    agreement_quantum: int = PROXY_TRUE_AGREEMENT_QUANTUM,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Dict:
    """Prove the residual bound against TRUE echo, not just the D_hat proxy.

    The corpus modes correlate aligned_far against D_hat = mic -
    linear_error, the PBFDKF's OWN estimate -- a valid proxy only when the
    filter converged. This mode renders a few short clean sequences with the
    repo's own generator and measures the SAME aligned-far tap against the
    rendered ground-truth echo (``RenderedSequence.audit['echo']``).

    Per sequence it asserts, over locked windows:
      1. the TRUE-ECHO residual is positive and below ``max_residual_ms``
         (the same bounds the D_hat self-test enforces);
      2. the D_hat-proxy residual agrees with the true-echo residual within
         ``agreement_quantum`` samples (one histogram quantum);
      3. no measured window peaks on the +-max_lag search boundary.
    Both numbers are reported, each labelled with its source.
    """
    cases: List[Dict] = []
    passed = True
    for name, bulk, mic, far, true_echo, contract in render_synthetic_sequences(
        n_sequences, n_chunks, sample_rate, seed
    ):
        run = run_linear_aec_with_taps(mic, far, contract)

        def _measure(target: np.ndarray) -> List[WindowMeasurement]:
            windows, _ = measure_residual_windows(
                run.aligned_far,
                target,
                run.sample_rate,
                hop_size=run.hop_size,
                delay_samples=run.delay_samples,
                confidence=run.confidence,
                window_s=window_s,
                max_lag=max_lag,
                settle_s=settle_s,
            )
            return windows

        # One engine run, two targets: the proxy and the rendered truth.
        proxy_windows = _measure(run.echo_estimate)
        true_windows = _measure(true_echo)
        proxy_locked = [w for w in proxy_windows if w.locked]
        true_locked = [w for w in true_windows if w.locked]

        lock = first_lock_hop(run.delay_samples)
        checks: Dict[str, bool] = {
            "locked": lock >= 0,
            "has_post_lock_windows": (
                len(true_locked) >= 2 and len(proxy_locked) >= 2
            ),
            "no_boundary_peaks": not any(
                w.boundary_peak for w in proxy_windows + true_windows
            ),
        }
        case: Dict = {
            "name": name,
            "bulk_delay_samples": bulk,
            "lock_hop": lock,
            "applied_delay_final": int(run.delay_samples[-1]),
            "checks": checks,
        }
        if true_locked and proxy_locked:
            true_residual = float(
                np.median([w.lag_samples for w in true_locked])
            )
            proxy_residual = float(
                np.median([w.lag_samples for w in proxy_locked])
            )
            case["residual_samples_true_echo"] = true_residual
            case["residual_ms_true_echo"] = (
                true_residual * 1000.0 / sample_rate
            )
            case["residual_samples_proxy"] = proxy_residual
            case["residual_ms_proxy"] = proxy_residual * 1000.0 / sample_rate
            case["proxy_true_delta_samples"] = proxy_residual - true_residual
            checks["true_echo_residual_positive"] = true_residual > 0
            checks["true_echo_residual_below_max"] = (
                case["residual_ms_true_echo"] < max_residual_ms
            )
            checks["proxy_agrees_with_true_echo"] = (
                abs(proxy_residual - true_residual) <= agreement_quantum
            )
        case_ok = bool(checks) and all(checks.values())
        case["passed"] = case_ok
        passed = passed and case_ok
        cases.append(case)
    return {
        "passed": passed,
        "residual_sources": [TRUE_ECHO_SOURCE, PROXY_SOURCE],
        "params": {
            "n_sequences": n_sequences,
            "n_chunks": n_chunks,
            "sample_rate": sample_rate,
            "seed": seed,
            "settle_s": settle_s,
            "max_residual_ms": max_residual_ms,
            "agreement_quantum": agreement_quantum,
            "window_s": window_s,
            "max_lag": max_lag,
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(results: Sequence[FileResult], aggregate: Dict) -> None:
    print(f"residual lag measured against: {PROXY_SOURCE_HUMAN}")
    header = (
        f"{'file':<20} {'hops':>6} {'lock_ms':>8} {'events':>6} "
        f"{'windows':>7} {'unsett':>6} {'p50_ms':>8} {'p90_ms':>8} "
        f"{'max_ms':>8} {'neg%':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        lags = np.array([w.lag_ms for w in r.windows if w.locked])
        unsettled = sum(1 for w in r.windows if not w.locked)
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
            f"{lags.size:>7} {unsettled:>6} {p50} {p90} {mx} {neg}"
        )
    print("-" * len(header))
    print(f"aggregate (locked windows only, vs {PROXY_SOURCE_HUMAN}):")
    for key in (
        "n_files", "n_windows", "p50_ms", "p90_ms", "p99_ms", "max_ms",
        "frac_negative", "lock_ms_median", "lock_ms_max",
        "delay_change_events_total", "n_windows_unsettled",
        "n_files_never_locked", "n_windows_skipped_far_silent",
        "n_windows_skipped_low_corr", "frac_boundary_peak",
        "frac_undecidable",
    ):
        value = aggregate.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        print(f"  {key:<30} {value}")
    if aggregate.get("frac_boundary_peak"):
        print(
            f"  WARNING: {aggregate.get('n_windows_boundary_peak')} window(s) "
            "peak ON the +-max_lag search boundary -- the search range is too "
            "small to bound the residual; raise --max-lag and re-run."
        )


def _print_self_test(report: Dict) -> None:
    print(
        "self-test (residual vs "
        f"{PROXY_SOURCE_HUMAN}):",
        "PASS" if report["passed"] else "FAIL",
    )
    for case in report["cases"]:
        print(
            f"  bulk_delay={case['bulk_delay']:>5}  "
            f"applied={case.get('applied_delay_final', -1):>5}  "
            f"residual_ms(proxy)={case.get('residual_ms', float('nan')):6.2f}  "
            f"injected_delta={case.get('injected_delta_samples', float('nan')):7.1f}  "
            f"negative_lag={case.get('negative_lag_samples', float('nan')):7.1f}"
        )
        for name, ok in case["checks"].items():
            if not ok:
                print(f"    FAILED check: {name}")


def _print_synthetic_echo_test(report: Dict) -> None:
    print(
        "synthetic-echo-test (residual vs "
        f"{TRUE_ECHO_SOURCE_HUMAN}, cross-checked against "
        f"{PROXY_SOURCE_HUMAN}):",
        "PASS" if report["passed"] else "FAIL",
    )
    for case in report["cases"]:
        print(
            f"  {case['name']}  bulk={case['bulk_delay_samples']:>5}  "
            f"applied={case.get('applied_delay_final', -1):>5}  "
            "residual_ms(true echo)="
            f"{case.get('residual_ms_true_echo', float('nan')):6.2f}  "
            "residual_ms(proxy)="
            f"{case.get('residual_ms_proxy', float('nan')):6.2f}  "
            "proxy-true delta="
            f"{case.get('proxy_true_delta_samples', float('nan')):6.1f} samples"
        )
        for name, ok in case["checks"].items():
            if not ok:
                print(f"    FAILED check: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the residual alignment lag at the AEC -> NN seam "
            "(aligned far-end vs an echo target). See module docstring for "
            "the sign convention and the D_hat-proxy vs true-echo labels."
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
        "--settle-s", type=float, default=None,
        help=(
            "locked-window gate: minimum seconds since the last applied-"
            f"delay change (default {DEFAULT_SETTLE_S} for corpus modes; the "
            "built-in synthetic proofs default to their own 1.0 s)"
        ),
    )
    parser.add_argument(
        "--max-undecidable", type=float, default=DEFAULT_MAX_UNDECIDABLE,
        help=(
            "fail (nonzero exit) when the undecidable window fraction "
            "(silent + low-correlation + unsettled) exceeds this"
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help=(
            "run the built-in synthetic proof against the D_hat proxy "
            "(no corpus needed)"
        ),
    )
    parser.add_argument(
        "--synthetic-echo-test", action="store_true",
        help=(
            "render short sequences in memory with the repo generator and "
            "prove the residual against TRUE echo (audit['echo']); asserts "
            "the D_hat proxy agrees within one histogram quantum"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test and args.synthetic_echo_test:
        print(
            "--self-test and --synthetic-echo-test are separate proofs; "
            "run them one at a time",
            file=sys.stderr,
        )
        return 2

    if args.self_test:
        kwargs = dict(window_s=args.window_s, max_lag=args.max_lag)
        if args.settle_s is not None:
            kwargs["settle_s"] = args.settle_s
        report = run_self_test(**kwargs)
        _print_self_test(report)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)
        return 0 if report["passed"] else 1

    if args.synthetic_echo_test:
        kwargs = dict(window_s=args.window_s, max_lag=args.max_lag)
        if args.settle_s is not None:
            kwargs["settle_s"] = args.settle_s
        report = run_synthetic_echo_test(**kwargs)
        _print_synthetic_echo_test(report)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)
        return 0 if report["passed"] else 1

    if bool(args.mic) != bool(args.far):
        print("--mic and --far must be given together", file=sys.stderr)
        return 2
    if not args.pairs_dir and not args.mic:
        print(
            "one of --pairs-dir, --mic/--far, --self-test, or "
            "--synthetic-echo-test is required",
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

    settle_s = args.settle_s if args.settle_s is not None else DEFAULT_SETTLE_S
    results = [
        measure_pair(
            name, mic, far, contract,
            window_s=args.window_s, max_lag=args.max_lag, settle_s=settle_s,
        )
        for name, mic, far in pairs
    ]
    aggregate = aggregate_results(results)
    _print_table(results, aggregate)
    gate_ok, gate_message = undecidable_gate(aggregate, args.max_undecidable)

    if args.json:
        payload = {
            "contract": {
                "sample_rate": contract.sample_rate,
                "frame_size": contract.frame_size,
                "hop_size": contract.hop_size,
                "filter_length": contract.filter_length,
                "version": contract.version,
            },
            "residual_source": PROXY_SOURCE,
            "params": {
                "window_s": args.window_s,
                "max_lag": args.max_lag,
                "settle_s": settle_s,
                "max_undecidable": args.max_undecidable,
            },
            "files": [_file_result_to_dict(r) for r in results],
            "aggregate": aggregate,
            "undecidable_gate_ok": gate_ok,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)

    if not gate_ok:
        print(gate_message, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
