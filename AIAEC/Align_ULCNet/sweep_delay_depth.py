#!/usr/bin/env python3
"""Compare Align-ULCNet streaming inference across a delay profile.

A delay profile has two independent halves and this tool drives both:

* the AEC half, ``--delay-num-filters`` (n): how far the PBFDKF frontend's
  matched-filter bank searches for the bulk far-to-mic delay, and how much AEC
  pool that costs. Reliable reach is 125/221/317/413/509 ms for n=1..5.
* the model half, ``--depths`` (D): how many past frames the temporal
  alignment attention keeps, and how much model state that costs. One hop is
  16 ms on the fixed 16 kHz / 512 / 256 grid.

They do not add up into one delay budget. Each layer must satisfy the input
condition the previous one delivers, which is why n and D are reported side by
side in every summary row rather than summed.

Both are DEPLOYMENT knobs and neither is a data-contract change. n is a
runtime AEC init override applied to the instance this tool builds; D is an
export-time model shape. The checkpoint's recorded ``linear_aec`` contract is
read and honoured unchanged either way -- corpora are always materialized at
the frozen ``DATASET_DELAY_NUM_FILTERS`` bank, so a run at another n is a
diagnostic of the frontend, reported as departing from the corpus the
checkpoint was trained on.

The microphone/far frontend is evaluated once, with a tapped hop loop that
records the delay the PBFDKF actually applied and the far it actually
consumed. Every requested depth then loads the same checkpoint weights into an
otherwise identical model and runs ``forward_stream()`` one STFT frame at a
time. The tool writes one float WAV per depth, a summary CSV, a per-frame
delay trace, and the AEC alignment trace.

Every clip is also QA'd against an estimator-INDEPENDENT offline measurement
of the bulk delay (see :func:`alignment_qa`). A clip whose applied delay
disagrees with that measurement is marked invalid so a mis-lock cannot be
averaged into a delay-profile statistic.

Examples::

    python3 sweep_delay_depth.py model.pth mic.wav far.wav d_sweep \
        --depths 64,32,16,8,4 --device cuda

    python3 sweep_delay_depth.py model.pth mic.wav far.wav short_route \
        --far-input-mode aligned_far --delay-num-filters 2 --depths 8,4

    python3 sweep_delay_depth.py model.pth kf_error.wav far.wav d_sweep \
        --input-is-linear-error --depths 64,16,8 \
        --target-wav clean_target.wav

The Python real-time factor is useful only for relative comparisons between
depths on the same machine.  It is not an estimate of NPU performance.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet.denoise import load_model
from AIAEC.dataset_gen import istft, stft
from AIAEC.dataset_gen.linear_aec import (
    DATASET_DELAY_NUM_FILTERS,
    DELAY_NUM_FILTERS_RANGE,
    LinearAecContract,
    check_delay_num_filters,
)
from AIAEC.dataset_gen.measure_align_residual import (
    DEFAULT_MAX_LAG,
    DEFAULT_SETTLE_S,
    DEFAULT_WINDOW_S,
    EngineRun,
    count_delay_change_events,
    first_lock_hop,
    measure_residual_windows,
    run_linear_aec_with_taps,
)
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC.training_common import auto_device


@dataclass
class StreamResult:
    enhanced: Tensor
    delay_distribution: Tensor
    elapsed_seconds: float
    state_bytes: int


def parse_depths(value: str) -> List[int]:
    """Parse a comma-separated, positive, duplicate-free D list."""
    depths: List[int] = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            depth = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid delay depth {item!r}; expected comma-separated integers"
            ) from exc
        if depth <= 0:
            raise argparse.ArgumentTypeError(
                f"delay depths must be positive, got {depth}"
            )
        if depth not in depths:
            depths.append(depth)
    if not depths:
        raise argparse.ArgumentTypeError("at least one delay depth is required")
    return depths


def parse_delay_num_filters(value: str) -> int:
    """argparse adapter over :func:`check_delay_num_filters` -- one validator,
    two exception dialects."""
    try:
        return check_delay_num_filters(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# Reliable bulk-delay reach of an n-filter matched bank, in ms: lib/aec's
# contract value ((n-1)*384 + 501 downsampled samples at 0.25 ms each), not a
# geometric span -- derived from that arithmetic (125/221/317/413/509 ms)
# rather than copied as literals, mirroring the C tests' KD_RELIABLE_SAMPLES.
# Printed so a run that fails to lock says whether the delay was ever inside
# reach.
MATCHED_REACH_MS = {
    n: ((n - 1) * 384 + 501) * 0.25
    for n in range(DELAY_NUM_FILTERS_RANGE[0], DELAY_NUM_FILTERS_RANGE[1] + 1)
}

# QA gate for the applied delay. The offline cross-correlation of RAW far
# against the mic measures the true bulk delay independently of the estimator;
# a healthy applied delay sits at or just before it (the aligned far leads the
# echo by the alignment headroom, never trails it). These bounds are wide
# enough to absorb estimator quantisation and headroom, and orders of
# magnitude tighter than the failure class they exist to catch -- a corpus
# tail-length difference reported as a multi-second delay.
QA_MAX_HEADROOM_MS = 32.0
QA_MAX_OVERSHOOT_MS = 8.0
QA_MIN_WINDOWS = 2


def resolve_delay_num_filters(requested: Optional[int]) -> int:
    """The bank size this run deploys.

    ``None`` -- the flag omitted -- resolves to the size every corpus is
    materialized at, so an unflagged run reproduces exactly the frontend the
    checkpoint was trained against. Resolved here rather than as an argparse
    default so the tool can still tell "asked for the corpus size" apart from
    "did not ask", which is what the ``--input-is-linear-error`` conflict check
    needs.
    """
    if requested is None:
        return DATASET_DELAY_NUM_FILTERS
    return check_delay_num_filters(requested)


def check_argument_conflicts(args: argparse.Namespace) -> None:
    """Refuse flag pairs that individually parse but cannot both apply.

    Checked before anything is loaded, so a run that could only produce a
    misleading report stops at the command line rather than after minutes of
    inference.
    """
    if args.input_is_linear_error and args.delay_num_filters is not None:
        raise ValueError(
            "--delay-num-filters has no effect with --input-is-linear-error: "
            "that bypass skips this project's PBFDKF entirely, so there is no "
            "matched-filter bank to size. Drop one of the two flags."
        )


def _default_depths(checkpoint_depth: int) -> List[int]:
    candidates = [checkpoint_depth, 64, 32, 16, 8, 4]
    return list(dict.fromkeys(d for d in candidates if d <= checkpoint_depth))


def _sync_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def _state_nbytes(state: Dict[str, object]) -> int:
    total = 0
    for cell in state.values():
        if hasattr(cell, 'state_tensors'):
            tensors = cell.state_tensors().values()
        elif isinstance(cell, Tensor):
            tensors = (cell,)
        else:
            tensors = ()
        total += sum(t.numel() * t.element_size() for t in tensors)
    return int(total)


def run_streaming_frames(
    model,
    error_spec: Tensor,
    far_spec: Tensor,
) -> StreamResult:
    """Run the actual one-frame model path and retain its delay decisions."""
    if error_spec.shape != far_spec.shape:
        raise ValueError("error_spec and far_spec must have identical shapes")
    if error_spec.ndim != 3 or error_spec.shape[1] == 0:
        raise ValueError("streaming spectra must have shape [B,T,F] with T > 0")

    state = model.create_stream_state()
    enhanced: List[Tensor] = []
    delays: List[Tensor] = []
    device = error_spec.device
    _sync_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        for frame in range(error_spec.shape[1]):
            output = model.forward_stream(
                linear_error=error_spec[:, frame:frame + 1],
                far_end=far_spec[:, frame:frame + 1],
                state=state,
            )
            enhanced.append(output.enhanced)
            delays.append(output.delay_distribution)
    _sync_device(device)
    elapsed = time.perf_counter() - started
    enhanced_tensor = torch.cat(enhanced, dim=1)
    delay_tensor = torch.cat(delays, dim=1)
    if not torch.isfinite(enhanced_tensor.real).all() or not torch.isfinite(
            enhanced_tensor.imag).all():
        raise RuntimeError("streaming model produced NaN or Inf")
    if not torch.isfinite(delay_tensor).all():
        raise RuntimeError("delay distribution contains NaN or Inf")
    return StreamResult(
        enhanced=enhanced_tensor,
        delay_distribution=delay_tensor,
        elapsed_seconds=elapsed,
        state_bytes=_state_nbytes(state),
    )


def _db(value: float, floor: float = 1e-12) -> float:
    return 20.0 * math.log10(max(value, floor))


def _snr_db(reference: Tensor, estimate: Tensor) -> float:
    error = estimate - reference
    numerator = reference.square().mean().item()
    denominator = error.square().mean().item()
    if numerator <= 1e-20:
        return float('nan')
    if denominator <= 1e-20:
        return float('inf')
    return 10.0 * math.log10(numerator / denominator)


def _si_sdr_db(target: Tensor, estimate: Tensor) -> float:
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    target_energy = target.square().sum()
    if target_energy.item() <= 1e-20:
        return float('nan')
    projection = target * ((estimate * target).sum() / target_energy)
    noise = estimate - projection
    return float(10.0 * torch.log10(
        projection.square().sum().clamp_min(1e-20)
        / noise.square().sum().clamp_min(1e-20)
    ).item())


def _delay_summary(distribution: Tensor, depth: int, hop_seconds: float) -> Dict[str, float]:
    # Ignore the causal ring fill when the recording is long enough.  On a
    # shorter test clip use every frame rather than returning an empty report.
    warmup = depth - 1 if distribution.shape[1] > depth else 0
    active = distribution[:, warmup:]
    indices = torch.arange(depth, device=active.device, dtype=active.dtype)
    expected = (active * indices).sum(dim=-1)
    argmax = active.argmax(dim=-1)
    boundary = active[..., -1]
    return {
        'evaluated_frames': float(active.numel() // depth),
        'expected_delay_frames_mean': float(expected.mean().item()),
        'expected_delay_ms_mean': float(expected.mean().item() * hop_seconds * 1000.0),
        'argmax_delay_frames_p50': float(torch.quantile(argmax.float(), 0.50).item()),
        'argmax_delay_frames_p90': float(torch.quantile(argmax.float(), 0.90).item()),
        'uniform_boundary_rate': 1.0 / depth,
        'argmax_at_max_depth_rate': float((argmax == depth - 1).float().mean().item()),
        'max_depth_probability_mean': float(boundary.mean().item()),
    }


def _write_delay_trace(path: Path, distribution: Tensor, hop_seconds: float) -> None:
    values = distribution.detach().cpu()[0]
    indices = torch.arange(values.shape[-1], dtype=values.dtype)
    expected = (values * indices).sum(dim=-1)
    argmax = values.argmax(dim=-1)
    maximum = values.max(dim=-1).values
    boundary = values[:, -1]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'frame', 'time_ms', 'argmax_delay_frames',
            'expected_delay_frames', 'max_probability',
            'max_depth_probability',
        ])
        for frame in range(values.shape[0]):
            writer.writerow([
                frame,
                f"{frame * hop_seconds * 1000.0:.6f}",
                int(argmax[frame]),
                f"{expected[frame].item():.6f}",
                f"{maximum[frame].item():.9f}",
                f"{boundary[frame].item():.9f}",
            ])


def _alignment_summary(run: EngineRun) -> Dict[str, object]:
    """Summarize the delay actually applied by the PBFDKF frontend."""
    delays = np.asarray(run.delay_samples, dtype=np.int64)
    confidence = np.asarray(run.confidence, dtype=np.float64)
    first = first_lock_hop(delays)
    if delays.size == 0:
        final_delay = -1
        final_confidence = float('nan')
    else:
        final_delay = int(delays[-1])
        final_confidence = float(confidence[-1])
    first_at_ms = (
        (first + 1) * run.hop_size * 1000.0 / run.sample_rate
        if first >= 0 else float('nan')
    )
    return {
        'aec_delay_acquired': first >= 0,
        'aec_first_acquired_ms': first_at_ms,
        'aec_initial_delay_samples': int(delays[first]) if first >= 0 else -1,
        'aec_final_delay_samples': final_delay,
        'aec_final_delay_ms': (
            final_delay * 1000.0 / run.sample_rate
            if final_delay >= 0 else float('nan')
        ),
        'aec_delay_change_events': count_delay_change_events(delays),
        'aec_final_confidence': final_confidence,
    }


class OfflineBulkDelay(NamedTuple):
    """What an estimator-independent raw-correlation measurement concluded."""

    n_windows: int
    n_boundary_peak: int
    bulk_delay_samples: int     # -1 when no window produced a measurement


def measure_offline_bulk_delay(
    microphone: np.ndarray,
    far_end: np.ndarray,
    sample_rate: int,
    *,
    max_lag: int,
    window_s: float = DEFAULT_WINDOW_S,
) -> OfflineBulkDelay:
    """Bulk far-to-mic delay from RAW signals, independent of the estimator.

    Reuses the residual meter's windowed, energy-gated, normalized
    cross-correlation with the RAW far as the reference and the microphone as
    the target, so ``lag > 0`` means the far leads the mic by that many
    samples -- the bulk delay. Windows with no far excitation or no
    explanatory peak are skipped rather than contributing a noise lag, and a
    peak sitting on the search boundary is reported instead of being read as a
    measurement: that combination is what separates a real delay from the
    corpus tail-length artefact that once read as 3744 ms.

    The lock gate is deliberately not applied here. This measurement must stay
    independent of the very estimator decision it is used to audit.
    """
    windows, _skipped = measure_residual_windows(
        far_end, microphone, sample_rate,
        window_s=window_s, max_lag=max_lag,
    )
    lags = np.asarray([w.lag_samples for w in windows], dtype=np.float64)
    return OfflineBulkDelay(
        n_windows=int(lags.size),
        n_boundary_peak=int(sum(1 for w in windows if w.boundary_peak)),
        bulk_delay_samples=(
            int(round(float(np.median(lags)))) if lags.size else -1
        ),
    )


def alignment_qa(
    run: EngineRun,
    microphone: np.ndarray,
    raw_far: np.ndarray,
    *,
    max_lag: int,
    settle_s: float = DEFAULT_SETTLE_S,
    offline: Optional[OfflineBulkDelay] = None,
) -> Dict[str, object]:
    """Per-clip QA of the estimator's decision against offline ground truth.

    Two independent verdicts, both reported:

    * ``applied vs offline bulk delay`` -- the estimator's applied delay must
      land at, or just before, the delay an estimator-independent correlation
      of the raw signals measures. Anything else is a mis-lock and the clip is
      marked INVALID so it cannot be averaged into a delay-profile statistic.
    * ``aligned-far residual`` -- over settled, locked windows the far the
      PBFDKF actually consumed must lead the echo (positive lag) by a bounded
      amount. This is the seam the NN sees in ``aligned_far`` mode.

    ``status`` is one of ``ok`` / ``mislock`` / ``not_acquired`` /
    ``undecidable``. ``not_acquired`` is an HONEST outcome, not a defect: a
    bank too small to reach the clip's delay is supposed to stay unlocked and
    let the pipeline fail open. It is reported separately from ``mislock``
    precisely so the two are never averaged together.

    ``offline`` accepts a precomputed measurement: it depends only on the raw
    signals, so callers that QA several runs of one clip measure once.
    """
    if offline is None:
        offline = measure_offline_bulk_delay(
            microphone, raw_far, run.sample_rate, max_lag=max_lag,
        )
    residual_windows, _ = measure_residual_windows(
        run.aligned_far, run.echo_estimate, run.sample_rate,
        hop_size=run.hop_size,
        delay_samples=run.delay_samples,
        confidence=run.confidence,
        settle_s=settle_s,
    )
    locked = [w.lag_ms for w in residual_windows if w.locked]
    applied = int(run.delay_samples[-1]) if run.delay_samples.size else -1
    to_ms = 1000.0 / run.sample_rate

    decidable = (
        offline.n_windows >= QA_MIN_WINDOWS
        and offline.n_boundary_peak == 0
    )
    headroom_ms = float('nan')
    if applied < 0:
        status = 'not_acquired' if decidable else 'undecidable'
    elif not decidable:
        status = 'undecidable'
    else:
        headroom_ms = (offline.bulk_delay_samples - applied) * to_ms
        status = (
            'ok'
            if -QA_MAX_OVERSHOOT_MS <= headroom_ms <= QA_MAX_HEADROOM_MS
            else 'mislock'
        )
    return {
        'qa_status': status,
        # Only a measured disagreement invalidates a clip. A clip whose delay
        # was never inside the bank's reach is a valid observation OF that.
        'qa_valid': status in ('ok', 'not_acquired'),
        'qa_offline_bulk_delay_samples': offline.bulk_delay_samples,
        'qa_offline_bulk_delay_ms': (
            offline.bulk_delay_samples * to_ms
            if offline.bulk_delay_samples >= 0 else float('nan')
        ),
        'qa_offline_windows': offline.n_windows,
        'qa_offline_boundary_peaks': offline.n_boundary_peak,
        'qa_applied_vs_offline_ms': headroom_ms,
        'qa_residual_locked_windows': len(locked),
        'qa_residual_p50_ms': (
            float(np.percentile(locked, 50)) if locked else float('nan')
        ),
        'qa_residual_max_ms': float(np.max(locked)) if locked else float('nan'),
        'qa_residual_min_ms': float(np.min(locked)) if locked else float('nan'),
    }


def _write_alignment_trace(path: Path, run: EngineRun) -> None:
    """Write the post-hop PBFDKF delay state used to produce aligned far."""
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'hop', 'time_ms', 'applied_delay_samples', 'applied_delay_ms',
            'confidence', 'acquired',
        ])
        for hop, (delay, confidence) in enumerate(zip(
                run.delay_samples, run.confidence)):
            # State is sampled after this hop has been processed, so report
            # the hop-end time rather than its input start time.
            time_ms = (hop + 1) * run.hop_size * 1000.0 / run.sample_rate
            delay_ms = (
                float(delay) * 1000.0 / run.sample_rate
                if delay >= 0 else float('nan')
            )
            writer.writerow([
                hop,
                f"{time_ms:.6f}",
                int(delay),
                f"{delay_ms:.6f}",
                f"{float(confidence):.9f}",
                int(delay >= 0),
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav', metavar='mic_or_linear_error_wav')
    parser.add_argument('far_wav')
    parser.add_argument('output_dir')
    parser.add_argument(
        '--depths', type=parse_depths, default=None,
        help='Comma-separated D values. Default: checkpoint D followed by '
             'smaller candidates from 64,32,16,8,4.',
    )
    parser.add_argument(
        '--reference-depth', type=int, default=None,
        help='D used as the waveform-difference reference (default: the '
             'checkpoint contract depth). It is added to --depths if needed.',
    )
    parser.add_argument('--device', default=None,
                        help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument(
        '--input-is-linear-error', action='store_true',
        help='Treat the first WAV as an existing KF/AEC error and bypass PBFDKF.',
    )
    parser.add_argument(
        '--far-input-mode', choices=('raw_far', 'aligned_far'),
        default='raw_far',
        help='Far stream presented to the NN. raw_far reproduces existing '
             'checkpoint training. aligned_far is a diagnostic, out-of-'
             'distribution experiment for current raw-far checkpoints: it '
             'taps the far hop actually consumed by PBFDKF and writes '
             'aec_alignment.csv; with --input-is-linear-error, the supplied '
             'far WAV is assumed to be already aligned and no AEC delay trace '
             'is available.',
    )
    parser.add_argument(
        '--delay-num-filters', type=parse_delay_num_filters, default=None,
        metavar='N',
        help='Matched-filter bank size n (1..5) the PBFDKF frontend runs '
             f'with (default {DATASET_DELAY_NUM_FILTERS}). This is the '
             'AEC-side half of a delay profile: it sets how far the '
             'bulk-delay search reaches (125/221/317/413/509 ms for n=1..5) '
             'and how much AEC pool it costs, and it is independent of the NN '
             'depth D set by --depths. It is an AEC init parameter applied to '
             'this diagnostic run only: it is not written back to any '
             'contract, and corpora are always materialized at the default, '
             'so any other value is reported as departing from the corpus the '
             'checkpoint was trained on.',
    )
    parser.add_argument(
        '--qa-max-lag', type=int, default=DEFAULT_MAX_LAG * 8,
        help='Search range (+- samples) for the estimator-independent '
             'offline bulk-delay QA. The default spans 1024 ms at 16 kHz, '
             'wide enough to bound the n=5 reach with margin.',
    )
    parser.add_argument(
        '--target-wav', default=None,
        help='Optional aligned clean target. Adds SNR/SI-SDR columns; it is '
             'resampled to the checkpoint rate and must match the primary '
             'timeline after resampling.',
    )
    parser.add_argument(
        '--max-seconds', type=float, default=None,
        help='Evaluate only the beginning of the stream (useful for a quick '
             'smoke test; PBFDKF is still cold-started at sample zero).',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Allow replacement of an existing summary/WAV/trace set.',
    )
    return parser


def _require_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        preview = ', '.join(existing[:3])
        raise FileExistsError(
            f"output already exists ({preview}); pass --overwrite to replace it"
        )


def _load_optional_target(
    path: str,
    sample_rate: int,
    length: int,
    allow_prefix: bool,
) -> Tensor:
    # Reuse the public resampling path without inventing another filter.  The
    # same file is supplied as both synchronized streams; only the first is
    # retained.
    target, _copy, _rates = load_mic_far(path, path, sample_rate)
    if allow_prefix and target.shape[-1] >= length:
        return target[..., :length]
    if target.shape[-1] != length:
        raise ValueError(
            f"target length {target.shape[-1]} does not match primary length "
            f"{length} after resampling; provide a time-aligned target"
        )
    return target


def main(args: argparse.Namespace) -> None:
    check_argument_conflicts(args)
    device_name = auto_device(args.device)
    device = torch.device(device_name)
    checkpoint_model, grid, linear_contract = load_model(
        args.checkpoint, device_name
    )
    checkpoint_depth = int(checkpoint_model.max_delay_frames)
    depths = list(args.depths) if args.depths is not None else _default_depths(
        checkpoint_depth
    )
    reference_depth = (checkpoint_depth if args.reference_depth is None
                       else int(args.reference_depth))
    if reference_depth <= 0:
        raise ValueError("--reference-depth must be positive")
    if reference_depth not in depths:
        depths.insert(0, reference_depth)

    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")

    # The recorded contract is used exactly as stamped; the bank size is a
    # runtime AEC init override this tool applies on top of it and never a
    # contract field, so there is no second, "runtime" contract to build.
    frontend_contract = LinearAecContract.from_dict(linear_contract)
    delay_num_filters = resolve_delay_num_filters(args.delay_num_filters)
    if not args.input_is_linear_error:
        print(f"PBFDKF matched-filter bank: n={delay_num_filters} "
              f"(reliable bulk-delay reach "
              f"{MATCHED_REACH_MS[delay_num_filters]:.0f} ms), contract "
              f"{frontend_contract.version} unchanged")
        if delay_num_filters != DATASET_DELAY_NUM_FILTERS:
            print("WARNING: this bank size is a deployment override. Every "
                  f"corpus is materialized at n={DATASET_DELAY_NUM_FILTERS}, "
                  "so the linear_error the model sees here is not the one it "
                  "was trained on; treat the result as a candidate profile "
                  "measurement, not a release comparison.")

    alignment_run: Optional[EngineRun] = None
    alignment_qa_metrics: Dict[str, object] = {}
    if args.input_is_linear_error:
        error, far, source_rates = load_linear_error_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        print("using external linear-error input; PBFDKF bypassed")
        if args.far_input_mode == 'aligned_far':
            print("aligned_far evaluation: treating the supplied far WAV as "
                  "already aligned (no PBFDKF tap is available in bypass mode)")
            print("WARNING: current checkpoints are trained with raw_far; "
                  "aligned_far is an out-of-distribution diagnostic")
        if args.max_seconds is not None:
            limit = min(
                error.shape[-1], max(1, int(round(args.max_seconds * grid.sr)))
            )
            error = error[..., :limit]
            far = far[..., :limit]
    else:
        mic, far, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        if args.max_seconds is not None:
            limit = min(
                mic.shape[-1], max(1, int(round(args.max_seconds * grid.sr)))
            )
            mic = mic[..., :limit]
            far = far[..., :limit]
        # One tapped hop loop for BOTH far modes. The frontend and its
        # linear_error are identical either way -- only which far stream is
        # handed to the NN differs -- and running the tapped loop
        # unconditionally is what makes the delay timeline, the aligned-far
        # residual and the offline QA available for raw_far too, instead of
        # only for the mode that happens to consume the tap.
        original_length = mic.shape[-1]
        mic_numpy = mic.squeeze(0).numpy()
        raw_far_numpy = far.squeeze(0).numpy()
        tapped = run_linear_aec_with_taps(
            mic_numpy, raw_far_numpy, frontend_contract,
            delay_num_filters=delay_num_filters,
        )
        alignment_run = tapped
        alignment_qa_metrics = alignment_qa(
            tapped, mic_numpy, raw_far_numpy, max_lag=args.qa_max_lag,
        )
        error = torch.from_numpy(
            tapped.error[:original_length]
        ).unsqueeze(0).contiguous()
        if args.far_input_mode == 'aligned_far':
            far = torch.from_numpy(
                tapped.aligned_far[:original_length]
            ).unsqueeze(0).contiguous()
            print("aligned_far evaluation: using the post-delay-buffer far "
                  "samples actually consumed by PBFDKF")
            print("WARNING: current checkpoints are trained with raw_far; "
                  "aligned_far is an out-of-distribution diagnostic")
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled primary/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")

    length = error.shape[-1]
    if length == 0:
        raise ValueError("evaluation stream is empty")
    target = (_load_optional_target(
                  args.target_wav, grid.sr, length,
                  allow_prefix=args.max_seconds is not None,
              )
              if args.target_wav else None)

    error = error.to(device)
    far = far.to(device)
    error_spec = stft(error, grid).transpose(-2, -1)
    far_spec = stft(far, grid).transpose(-2, -1)
    audio_seconds = length / grid.sr
    hop_seconds = grid.hop_len / grid.sr

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / 'summary.csv']
    if alignment_run is not None:
        paths.append(output_dir / 'aec_alignment.csv')
    for depth in depths:
        paths.extend((
            output_dir / f'D{depth:03d}.wav',
            output_dir / f'D{depth:03d}_delay.csv',
        ))
    _require_output_paths(paths, args.overwrite)

    alignment_metrics: Dict[str, object] = {}
    if alignment_run is not None:
        alignment_metrics = _alignment_summary(alignment_run)
        _write_alignment_trace(
            output_dir / 'aec_alignment.csv', alignment_run,
        )
        if alignment_metrics['aec_delay_acquired']:
            print(
                "PBFDKF delay: first acquired at "
                f"{alignment_metrics['aec_first_acquired_ms']:.1f} ms, "
                f"initial={alignment_metrics['aec_initial_delay_samples']} "
                "samples, final="
                f"{alignment_metrics['aec_final_delay_samples']} samples "
                f"({alignment_metrics['aec_final_delay_ms']:.2f} ms), "
                f"changes={alignment_metrics['aec_delay_change_events']}, "
                "final confidence="
                f"{alignment_metrics['aec_final_confidence']:.3f}"
            )
        else:
            print(
                "PBFDKF delay: NOT ACQUIRED during this recording (n="
                f"{alignment_run.delay_num_filters} reaches "
                f"{MATCHED_REACH_MS[alignment_run.delay_num_filters]:.0f} ms)"
            )
        print(f"wrote {output_dir / 'aec_alignment.csv'}")
    if alignment_qa_metrics:
        print(
            f"delay QA: {alignment_qa_metrics['qa_status'].upper()} -- "
            "offline bulk delay "
            f"{alignment_qa_metrics['qa_offline_bulk_delay_ms']:.1f} ms over "
            f"{alignment_qa_metrics['qa_offline_windows']} window(s), applied "
            "is earlier by "
            f"{alignment_qa_metrics['qa_applied_vs_offline_ms']:.1f} ms; "
            "aligned-far residual p50="
            f"{alignment_qa_metrics['qa_residual_p50_ms']:.2f} ms over "
            f"{alignment_qa_metrics['qa_residual_locked_windows']} locked "
            "window(s)"
        )
        if (args.far_input_mode == 'aligned_far'
                and alignment_qa_metrics['qa_status'] == 'not_acquired'):
            print(
                "NOTE: no delay was ever acquired, so the 'aligned' far this "
                "tool fed the model is the raw far. A deployed ALIGNED "
                "pipeline FAILS OPEN while unlocked and applies no model "
                "output at all, so these WAVs do not represent what that "
                "pipeline would emit for this clip."
            )
        if not alignment_qa_metrics['qa_valid']:
            print(
                "WARNING: this clip is marked INVALID for delay-profile "
                "statistics -- the applied delay does not agree with the "
                "estimator-independent offline measurement, so any residual "
                "or lock number from it describes a mis-lock, not the "
                "profile under test."
            )

    # Bank size the run actually deployed, read back off the engine rather
    # than echoed from the flag; 0 when PBFDKF was bypassed and there was no
    # bank at all. A summary row must not be able to name a profile the run
    # did not execute.
    deployed_bank = (
        alignment_run.delay_num_filters if alignment_run is not None else 0
    )
    bank_label = f"n={deployed_bank}" if deployed_bank else "PBFDKF bypassed"

    target_cpu = target.cpu() if target is not None else None
    reference_waveform: Optional[Tensor] = None
    metrics_by_depth: Dict[int, Dict[str, object]] = {}
    ordered_depths = [reference_depth] + [d for d in depths if d != reference_depth]
    for depth in ordered_depths:
        if depth == checkpoint_depth:
            model = checkpoint_model
        else:
            model, other_grid, _ = load_model(
                args.checkpoint, device_name, max_delay_frames=depth
            )
            if other_grid != grid:
                raise RuntimeError("model grid changed while sweeping D")
        print(f"running streaming D={depth} over {error_spec.shape[1]} frames")
        result = run_streaming_frames(model, error_spec, far_spec)
        waveform = istft(
            result.enhanced.transpose(-2, -1), grid, length=length
        ).detach().cpu()
        if reference_waveform is None:
            reference_waveform = waveform
        difference = waveform - reference_waveform
        sf.write(
            str(output_dir / f'D{depth:03d}.wav'),
            waveform.squeeze(0).numpy(), grid.sr, subtype='FLOAT',
        )
        _write_delay_trace(
            output_dir / f'D{depth:03d}_delay.csv',
            result.delay_distribution,
            hop_seconds,
        )
        metrics: Dict[str, object] = {
            'far_input_mode': args.far_input_mode,
            # Both halves of the deployed profile in every row, so a summary
            # can never be read back without knowing which (n, D) produced it.
            # Diagnostic columns: n is a runtime override, not part of the
            # recorded contract, so nothing downstream may key off them.
            'aec_delay_num_filters': deployed_bank,
            'aec_matched_reach_ms': MATCHED_REACH_MS.get(
                deployed_bank, float('nan')),
            # The grid every millisecond in this row is denominated in. A
            # delay-profile CSV read without it is unreadable: D is a count of
            # hops, and one hop is only 16 ms because of these three numbers.
            'grid_sample_rate': grid.sample_rate,
            'grid_fft_size': grid.n_fft,
            'grid_hop_size': grid.hop_len,
            'depth_frames': depth,
            'buffer_span_ms': depth * hop_seconds * 1000.0,
            'max_delay_ms': (depth - 1) * hop_seconds * 1000.0,
            'state_kib': result.state_bytes / 1024.0,
            'python_seconds': result.elapsed_seconds,
            'python_rtf': result.elapsed_seconds / audio_seconds,
            'output_peak': waveform.abs().max().item(),
            'output_rms_dbfs': _db(waveform.square().mean().sqrt().item()),
            'delta_vs_reference_max_abs': difference.abs().max().item(),
            'delta_vs_reference_rms': difference.square().mean().sqrt().item(),
            'snr_vs_reference_db': _snr_db(reference_waveform, waveform),
        }
        metrics.update(_delay_summary(
            result.delay_distribution, depth, hop_seconds
        ))
        metrics.update(alignment_metrics)
        metrics.update(alignment_qa_metrics)
        if target_cpu is not None:
            metrics['target_snr_db'] = _snr_db(target_cpu, waveform)
            metrics['target_si_sdr_db'] = _si_sdr_db(target_cpu, waveform)
        metrics_by_depth[depth] = metrics
        # WAV/trace/summary are now materialised on the host.  Do not retain
        # one full complex spectrum and delay matrix per D on the accelerator.
        del result, waveform
        if depth != checkpoint_depth:
            del model

    rows = [metrics_by_depth[depth] for depth in depths]

    fieldnames = list(rows[0].keys())
    with (output_dir / 'summary.csv').open(
            'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nD sweep summary at far={args.far_input_mode}, {bank_label} "
          "(Python timing is relative, not an NPU estimate):")
    print("  D   max lag  state KiB  RTF     boundary   SNR vs ref")
    for row in rows:
        print(
            f"  {row['depth_frames']:>3d} "
            f"{row['max_delay_ms']:>8.1f}ms "
            f"{row['state_kib']:>9.1f} "
            f"{row['python_rtf']:>7.3f} "
            f"{row['argmax_at_max_depth_rate']:>9.3%} "
            f"{row['snr_vs_reference_db']:>10.2f} dB"
        )
    print(f"wrote {output_dir / 'summary.csv'} and {len(rows)} WAV/trace pairs")


if __name__ == '__main__':
    main(build_parser().parse_args())
