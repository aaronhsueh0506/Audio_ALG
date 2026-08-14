#!/usr/bin/env python3
"""Compare Align-ULCNet streaming inference at several delay depths.

The microphone/far frontend is evaluated once.  Every requested depth then
loads the same checkpoint weights into an otherwise identical model and runs
``forward_stream()`` one STFT frame at a time.  The tool writes one float WAV
per depth, a summary CSV, and a per-frame delay trace for listening and
inspection before choosing the fixed D of an embedded ONNX export.

Examples::

    python3 sweep_delay_depth.py model.pth mic.wav far.wav d_sweep \
        --depths 64,32,16,8,4 --device cuda

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
from typing import Dict, Iterable, List, Optional

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
from AIAEC.dataset_gen.linear_aec import LinearAecContract
from AIAEC.dataset_gen.measure_align_residual import (
    EngineRun,
    count_delay_change_events,
    first_lock_hop,
    run_linear_aec_with_taps,
)
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC.training_common import LinearAecEngine, auto_device


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

    alignment_run: Optional[EngineRun] = None
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
        if args.far_input_mode == 'aligned_far':
            original_length = mic.shape[-1]
            tapped = run_linear_aec_with_taps(
                mic.squeeze(0).numpy(),
                far.squeeze(0).numpy(),
                LinearAecContract.from_dict(linear_contract),
            )
            error = torch.from_numpy(
                tapped.error[:original_length]
            ).unsqueeze(0).contiguous()
            far = torch.from_numpy(
                tapped.aligned_far[:original_length]
            ).unsqueeze(0).contiguous()
            alignment_run = tapped
            print("aligned_far evaluation: using the post-delay-buffer far "
                  "samples actually consumed by PBFDKF")
            print("WARNING: current checkpoints are trained with raw_far; "
                  "aligned_far is an out-of-distribution diagnostic")
        else:
            linear_aec = LinearAecEngine(
                n_lanes=1, sample_rate=grid.sr, contract=linear_contract
            )
            error, _echo_estimate = linear_aec(mic, far, grid.sr)
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
            print("PBFDKF delay: NOT ACQUIRED during this recording")
        print(f"wrote {output_dir / 'aec_alignment.csv'}")

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

    print("\nD sweep summary (Python timing is relative, not an NPU estimate):")
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
