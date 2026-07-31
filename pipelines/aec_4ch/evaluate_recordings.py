"""Reproducible offline validation for the two four-channel recordings.

The reference stems share one source timeline, but that timeline and the
four-channel capture do not share a recording start.  This tool estimates the
single file-level fixture offset before exercising the streaming pipeline.
That offline synchronization does not remove the physical echo delay; the
pipeline still owns exactly one live matched-filter instance that must acquire
that delay.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy import signal

from lib.aec.python.modules.config import AecConfig
from pipelines.aec_4ch import (
    EqualWeightBeamformer,
    FourChannelAecConfig,
    FourChannelAecPipeline,
)
from pipelines.aec_nr_pipeline import run_nr_spectrum, run_res


def estimate_file_offset(capture: np.ndarray, source: np.ndarray) -> tuple[int, float]:
    """Return lag where ``source[0]`` belongs on the capture timeline."""
    capture = np.asarray(capture, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    capture = capture - float(np.mean(capture))
    source = source - float(np.mean(source))
    correlation = signal.correlate(capture, source, mode="full", method="fft")
    lags = signal.correlation_lags(capture.size, source.size, mode="full")
    index = int(np.argmax(np.abs(correlation)))
    denom = np.sqrt(np.dot(capture, capture) * np.dot(source, source)) + 1e-30
    return int(lags[index]), float(abs(correlation[index]) / denom)


def place_on_timeline(source: np.ndarray, lag: int, length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    dst_start = max(0, int(lag))
    src_start = max(0, -int(lag))
    count = min(length - dst_start, source.size - src_start)
    if count > 0:
        output[dst_start : dst_start + count] = source[src_start : src_start + count]
    return output


def _frame_power(x: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    x = np.asarray(x[: n_frames * hop], dtype=np.float64)
    return np.mean(x.reshape(n_frames, hop) ** 2, axis=1)


def _activity(power: np.ndarray) -> np.ndarray:
    reference = float(np.percentile(power, 95.0)) if power.size else 0.0
    return power > max(reference * 1e-4, 1e-10)


def _attenuation_db(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not np.any(mask):
        return None
    before_power = float(np.mean(before[mask]))
    after_power = float(np.mean(after[mask]))
    return float(10.0 * np.log10((before_power + 1e-20) / (after_power + 1e-20)))


def evaluate_case(case_dir: Path, output_dir: Optional[Path] = None) -> dict:
    microphones, sample_rate = sf.read(
        case_dir / "unprocessed_4ch.wav", always_2d=True, dtype="float32"
    )
    far_path = case_dir / "woman(ref).wav"
    if not far_path.exists():
        # Accept the spelling used in some fixture handoffs without making it
        # a second data contract.
        far_path = case_dir / "women(ref).wav"
    far, far_rate = sf.read(far_path, always_2d=True, dtype="float32")
    near, near_rate = sf.read(case_dir / "man.wav", always_2d=True, dtype="float32")
    if sample_rate != far_rate or sample_rate != near_rate:
        raise ValueError("fixture sample rates do not match")
    if microphones.shape[1] != 4:
        raise ValueError("unprocessed_4ch.wav must contain four channels")

    far_lag, far_correlation = estimate_file_offset(microphones[:, 0], far[:, 0])
    near_lag, near_correlation = estimate_file_offset(microphones[:, 0], near[:, 0])
    if far.shape[0] != near.shape[0]:
        raise ValueError(
            "far/near fixture tracks must share one source timeline; their "
            "lengths differ, so one recording-start offset is not valid"
        )
    # man.wav and woman(ref).wav are stems from one source timeline. The near
    # path estimates only that timeline's recording-start offset. Applying the
    # far-to-mic lag here would also remove the physical echo/system delay and
    # reduce the live shared matcher test to a zero-delay case.
    fixture_lag = near_lag
    far_aligned = place_on_timeline(far[:, 0], fixture_lag, microphones.shape[0])
    near_aligned = place_on_timeline(near[:, 0], near_lag, microphones.shape[0])

    config = FourChannelAecConfig(sample_rate=sample_rate)
    # Explicit fixture adapter. Production leaves the beamformer unset and
    # drives process_pre_beamformer()/process_post_beamformer() around the
    # externally owned SRP-PHAT/GSC implementation.
    pipeline = FourChannelAecPipeline(config)
    fixture_beamformer = EqualWeightBeamformer()
    n_complete = min(microphones.shape[0], far_aligned.size) // pipeline.hop_size
    linear = np.zeros(n_complete * pipeline.hop_size, dtype=np.float32)
    contexts = []
    delays = []
    for frame_index in range(n_complete):
        start = frame_index * pipeline.hop_size
        stop = start + pipeline.hop_size
        pre = pipeline.process_pre_beamformer(
            microphones[start:stop], far_aligned[start:stop]
        )
        # Deliberately cross the same ownership seam the external algorithm
        # will use.  The adapter supplies no SRP/GSC behavior; it only makes
        # both sides of our contract executable in this repository.
        external = fixture_beamformer.process(pre.linear_hops, pre.contexts)
        result = pipeline.process_post_beamformer(pre, external)
        linear[start:stop] = result.beamformed
        contexts.append(result.context)
        delays.append(result.delay)
    n_frames = len(contexts)
    n_samples = linear.size
    input_mono = np.mean(microphones[:n_samples], axis=1, dtype=np.float32)

    # The reused pipeline helpers are human-facing CLI utilities and emit a
    # progress line on stdout.  Keep this evaluator's stdout machine-readable
    # JSON and send those progress messages to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        nr_gain = run_nr_spectrum(
            contexts, sample_rate, nr_preset="balanced", inject_echo_psd=True
        )
    frame, hop = config.resolved_grid()
    post_config = AecConfig(
        sample_rate=sample_rate,
        frame_size=frame,
        hop_size=hop,
        enable_delay_est=False,
        enable_res=False,
        return_res_context=True,
    )
    with contextlib.redirect_stdout(sys.stderr):
        post = run_res(
            linear,
            nr_gain,
            contexts,
            post_config,
            use_nr=True,
            use_res=True,
            combine="min",
        )

    input_power = _frame_power(input_mono, hop, n_frames)
    linear_power = _frame_power(linear, hop, n_frames)
    post_power = _frame_power(post, hop, n_frames)
    far_active = _activity(_frame_power(far_aligned, hop, n_frames))
    near_active = _activity(_frame_power(near_aligned, hop, n_frames))
    cohorts = {
        "far_only": far_active & ~near_active,
        "near_only": near_active & ~far_active,
        "double_talk": far_active & near_active,
        "silence": ~far_active & ~near_active,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_dir / f"{case_dir.name}_linear_bf.wav", linear, sample_rate)
        sf.write(output_dir / f"{case_dir.name}_nr_res.wav", post, sample_rate)

    finite = bool(
        np.all(np.isfinite(linear))
        and np.all(np.isfinite(post))
        and all(np.all(np.isfinite(c.error_spec)) for c in contexts)
    )
    final_delay = asdict(delays[-1]) if delays else None
    expected_echo_delay = int(far_lag - fixture_lag)
    final_delay_samples = int(delays[-1].delay_samples) if delays else 0
    delay_error = final_delay_samples - expected_echo_delay
    first_solid_frame = next(
        (index for index, state in enumerate(delays) if state.solid), None
    )
    first_nonzero_frame = next(
        (index for index, state in enumerate(delays) if state.delay_samples > 0),
        None,
    )
    changed_frames = [
        index for index, state in enumerate(delays) if state.changed
    ]
    last_change_frame = changed_frames[-1] if changed_frames else None
    return {
        "case": case_dir.name,
        "sample_rate": int(sample_rate),
        "input_samples": int(microphones.shape[0]),
        "processed_samples": int(n_samples),
        "frame_size_fft": int(frame),
        "hop_size": int(hop),
        "matched_filter_instances": pipeline.matched_filter_instance_count,
        "linear_filter_instances": pipeline.linear_filter_instance_count,
        "residual_suppressor_instances": pipeline.residual_suppressor_instance_count,
        "beamformer_configured": pipeline.beamformer_configured,
        "source_timeline_offset_samples": int(fixture_lag),
        "source_timeline_offset_ms": float(fixture_lag * 1000.0 / sample_rate),
        "far_total_correlation_lag_samples": int(far_lag),
        "far_total_correlation_lag_ms": float(far_lag * 1000.0 / sample_rate),
        "far_alignment_correlation": far_correlation,
        "expected_echo_delay_samples": expected_echo_delay,
        "expected_echo_delay_ms": float(
            expected_echo_delay * 1000.0 / sample_rate
        ),
        "final_delay_error_samples": int(delay_error),
        "final_delay_error_ms": float(delay_error * 1000.0 / sample_rate),
        "first_solid_delay_frame": first_solid_frame,
        "first_solid_delay_ms": (
            None
            if first_solid_frame is None
            else float(first_solid_frame * hop * 1000.0 / sample_rate)
        ),
        "first_nonzero_delay_frame": first_nonzero_frame,
        "first_nonzero_delay_ms": (
            None
            if first_nonzero_frame is None
            else float(first_nonzero_frame * hop * 1000.0 / sample_rate)
        ),
        "delay_change_count": len(changed_frames),
        "last_delay_change_frame": last_change_frame,
        "last_delay_change_ms": (
            None
            if last_change_frame is None
            else float(last_change_frame * hop * 1000.0 / sample_rate)
        ),
        "near_alignment_correlation": near_correlation,
        "final_shared_delay": final_delay,
        "finite": finite,
        "input_rms": float(np.sqrt(np.mean(input_mono.astype(np.float64) ** 2))),
        "linear_bf_rms": float(np.sqrt(np.mean(linear.astype(np.float64) ** 2))),
        "nr_res_rms": float(np.sqrt(np.mean(post.astype(np.float64) ** 2))),
        "cohorts": {
            name: {
                "frames": int(np.count_nonzero(mask)),
                "linear_attenuation_db": _attenuation_db(input_power, linear_power, mask),
                "nr_res_attenuation_db": _attenuation_db(input_power, post_power, mask),
            }
            for name, mask in cohorts.items()
        },
    }


def validate_recording_contract(result: dict) -> None:
    """Fail the recording test on structural or delay-acquisition regressions.

    Output-quality thresholds are deliberately excluded: this executable uses
    the equal-weight test beamformer, not the production external beamformer,
    and the checked-in stems are not a clean microphone-domain target.
    """
    failures = []
    if not result.get("finite"):
        failures.append("pipeline output/context contains NaN or Inf")
    expected_resources = {
        "matched_filter_instances": 1,
        "linear_filter_instances": 4,
        "residual_suppressor_instances": 1,
        "beamformer_configured": False,
    }
    for name, expected in expected_resources.items():
        if result.get(name) != expected:
            failures.append(f"{name}={result.get(name)!r}, expected {expected}")
    final_delay = result.get("final_shared_delay") or {}
    if not final_delay.get("solid"):
        failures.append("shared matched-filter delay never became solid")
    if result.get("first_nonzero_delay_frame") is None:
        failures.append("shared matched-filter never accepted a nonzero delay")
    delay_error = abs(int(result.get("final_delay_error_samples", 1 << 30)))
    tolerance = int(result["hop_size"]) // 2
    if delay_error > tolerance:
        failures.append(
            f"final shared delay error {delay_error} samples exceeds "
            f"half-hop tolerance {tolerance}"
        )
    if result.get("processed_samples", 0) <= 0:
        failures.append("no complete hops were processed")
    if failures:
        raise RuntimeError(
            f"{result.get('case', '<unknown>')} recording contract failed: "
            + "; ".join(failures)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root", type=Path, default=Path(__file__).resolve().parents[3] / "datasets"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-contract-check", action="store_true",
        help="emit diagnostics without failing structural/delay assertions",
    )
    args = parser.parse_args()
    results = [
        evaluate_case(args.datasets_root / name, args.output_dir)
        for name in ("aec_take_turn", "aec_together")
    ]
    if not args.no_contract_check:
        for result in results:
            validate_recording_contract(result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
