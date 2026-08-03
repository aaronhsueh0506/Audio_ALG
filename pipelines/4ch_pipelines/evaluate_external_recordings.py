"""Run the complete C SRP-PHAT/GSC pipeline on both checked-in recordings."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .evaluate_recordings import (
    _activity,
    _attenuation_db,
    _frame_power,
    estimate_file_offset,
    place_on_timeline,
)


# ---------------------------------------------------------------------------
# Recording alignment and C-runner helpers
# ---------------------------------------------------------------------------

def _default_binary() -> Optional[Path]:
    pipeline_root = Path(__file__).resolve().parent
    candidates = sorted(
        pipeline_root.glob("bin/*/audio_pipeline_4ch_raw"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_case(case_dir: Path):
    microphones, sample_rate = sf.read(
        case_dir / "unprocessed_4ch.wav", always_2d=True, dtype="float32"
    )
    far_path = case_dir / "woman(ref).wav"
    if not far_path.exists():
        far_path = case_dir / "women(ref).wav"
    far, far_rate = sf.read(far_path, always_2d=True, dtype="float32")
    near, near_rate = sf.read(
        case_dir / "man.wav", always_2d=True, dtype="float32"
    )
    if microphones.shape[1] != 4:
        raise ValueError(f"{case_dir}: unprocessed_4ch.wav is not four-channel")
    if far_rate != sample_rate or near_rate != sample_rate:
        raise ValueError(f"{case_dir}: sample-rate mismatch")
    if far.shape[0] != near.shape[0]:
        raise ValueError(f"{case_dir}: far/near source timelines differ")
    far_lag, far_corr = estimate_file_offset(microphones[:, 0], far[:, 0])
    near_lag, near_corr = estimate_file_offset(microphones[:, 0], near[:, 0])
    far_aligned = place_on_timeline(far[:, 0], near_lag, microphones.shape[0])
    near_aligned = place_on_timeline(near[:, 0], near_lag, microphones.shape[0])
    return (
        microphones,
        far_aligned,
        near_aligned,
        int(sample_rate),
        int(far_lag),
        float(far_corr),
        int(near_lag),
        float(near_corr),
    )


# ---------------------------------------------------------------------------
# Evaluation and acceptance contract
# ---------------------------------------------------------------------------

def evaluate_case(
    case_dir: Path,
    binary: Path,
    output_dir: Optional[Path],
    uca_radius_m: float,
    fft_size: int,
) -> dict:
    (
        microphones,
        far,
        near,
        sample_rate,
        far_lag,
        far_corr,
        timeline_lag,
        near_corr,
    ) = _load_case(case_dir)
    # 16 kHz rate default is 256/128 (8ms hop) as of 2026-08-02/03 (AEC's
    # python/modules/config.py and NR's core/signal_grid.py both default
    # here now); 512/256 remains a supported, explicit alternate grid.
    selected_fft = fft_size or (256 if sample_rate == 16000 else 1024)
    if (
        sample_rate == 16000
        and selected_fft not in (256, 512)
    ) or (
        sample_rate == 48000
        and selected_fft != 1024
    ):
        raise ValueError(
            f"unsupported spatial grid: {sample_rate} Hz / FFT {selected_fft}"
        )
    hop = selected_fft // 2
    n_frames = microphones.shape[0] // hop
    n_samples = n_frames * hop
    microphones = np.ascontiguousarray(microphones[:n_samples], dtype=np.float32)
    far = np.ascontiguousarray(far[:n_samples], dtype=np.float32)
    near = np.ascontiguousarray(near[:n_samples], dtype=np.float32)
    near_activity = _activity(_frame_power(near, hop, n_frames)).astype(np.uint8)

    with tempfile.TemporaryDirectory(prefix="four-aec-gsc-") as directory:
        temporary = Path(directory)
        mic_raw = temporary / "mic.f32"
        ref_raw = temporary / "ref.f32"
        vad_raw = temporary / "vad.u8"
        out_raw = temporary / "out.f32"
        microphones.tofile(mic_raw)
        far.tofile(ref_raw)
        near_activity.tofile(vad_raw)
        command = [
            str(binary),
            "--mic-raw",
            str(mic_raw),
            "--ref-raw",
            str(ref_raw),
            "--vad-u8",
            str(vad_raw),
            "--output-raw",
            str(out_raw),
            "--sample-rate",
            str(sample_rate),
            "--fft-size",
            str(selected_fft),
            "--uca-radius-m",
            str(uca_radius_m),
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        output = np.fromfile(out_raw, dtype=np.float32)

    if output.size != n_samples:
        raise RuntimeError(
            f"{case_dir.name}: C output has {output.size}, expected {n_samples}"
        )
    input_mono = np.mean(microphones, axis=1, dtype=np.float32)
    input_power = _frame_power(input_mono, hop, n_frames)
    output_power = _frame_power(output, hop, n_frames)
    far_active = _activity(_frame_power(far, hop, n_frames))
    near_active = near_activity.astype(bool)
    cohorts = {
        "far_only": far_active & ~near_active,
        "near_only": near_active & ~far_active,
        "double_talk": far_active & near_active,
        "silence": ~far_active & ~near_active,
    }
    expected_delay = far_lag - timeline_lag
    delay_error = int(summary["final_delay_samples"]) - expected_delay
    # The source-file correlation is only an approximate recording offset
    # (and is deliberately independent of the online matcher).  Keep at
    # least 5 ms of tolerance so choosing the smaller 128-sample hop does not
    # turn the same physical delay estimate into a false failure.
    delay_tolerance = max(hop // 2, sample_rate // 200)
    result = {
        "case": case_dir.name,
        "sample_rate": sample_rate,
        "input_samples": int(n_samples),
        "finite": bool(np.all(np.isfinite(output))),
        "input_rms": float(np.sqrt(np.mean(input_mono.astype(np.float64) ** 2))),
        "output_rms": float(np.sqrt(np.mean(output.astype(np.float64) ** 2))),
        "far_alignment_correlation": far_corr,
        "near_alignment_correlation": near_corr,
        "source_timeline_offset_samples": timeline_lag,
        "expected_echo_delay_samples": int(expected_delay),
        "final_delay_error_samples": delay_error,
        "delay_tolerance_samples": delay_tolerance,
        "frame_size": selected_fft,
        "fft_size": selected_fft,
        "hop_size": hop,
        "c_pipeline": summary,
        "cohorts": {
            name: {
                "frames": int(np.count_nonzero(mask)),
                "attenuation_db": _attenuation_db(
                    input_power, output_power, mask
                ),
            }
            for name, mask in cohorts.items()
        },
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(
            output_dir / f"{case_dir.name}_c_srp_gsc_nr_res.wav",
            output,
            sample_rate,
        )
    return result


def validate(result: dict) -> None:
    summary = result["c_pipeline"]
    failures = []
    if not result["finite"]:
        failures.append("non-finite output")
    expected = {
        "matched_filters": 1,
        "linear_aecs": 4,
        "nr": 1,
        "post_res": 1,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{key}={summary.get(key)}, expected {value}")
    expected_grid = {
        "sample_rate": result["sample_rate"],
        "frame_size": result["frame_size"],
        "fft_size": result["fft_size"],
        "hop": result["hop_size"],
        "n_freqs": result["fft_size"] // 2 + 1,
        "doa_sample_rate": result["sample_rate"],
        "doa_frame_size": result["frame_size"],
        "doa_hop_size": result["hop_size"],
        "doa_fft_size": result["fft_size"],
        "gsc_sample_rate": result["sample_rate"],
        "gsc_frame_size": result["frame_size"],
        "gsc_hop_size": result["hop_size"],
        "gsc_fft_size": result["fft_size"],
    }
    for key, value in expected_grid.items():
        if summary.get(key) != value:
            failures.append(f"{key}={summary.get(key)}, expected {value}")
    if not summary.get("final_delay_solid"):
        failures.append("shared delay never became solid")
    if summary.get("doa_analysis_frames", 0) <= 0:
        failures.append("DOA analysis never consumed a frame")
    if summary.get("doa_update_frames", 0) <= 0:
        failures.append("SRP-PHAT never updated")
    if summary.get("gsc_adaptive_frames", 0) <= 0:
        failures.append("GSC never adapted")
    if abs(result["final_delay_error_samples"]) > result[
        "delay_tolerance_samples"
    ]:
        failures.append(
            f"delay error {result['final_delay_error_samples']} exceeds "
            f"tolerance {result['delay_tolerance_samples']}"
        )
    if failures:
        raise RuntimeError(f"{result['case']}: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "datasets",
    )
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--uca-radius-m", type=float, default=0.035)
    parser.add_argument(
        "--fft-size",
        type=int,
        default=0,
        help="0=rate default; 16 kHz accepts 256/512, 48 kHz accepts 1024",
    )
    parser.add_argument("--no-contract-check", action="store_true")
    args = parser.parse_args()
    if args.binary is None or not args.binary.exists():
        raise SystemExit(
            "audio_pipeline_4ch_raw not found; build it with "
            "`make -C Audio_ALG/pipelines/4ch_pipelines audio_pipeline_4ch_raw`"
        )
    results = [
        evaluate_case(
            args.datasets_root / name,
            args.binary,
            args.output_dir,
            args.uca_radius_m,
            args.fft_size,
        )
        for name in ("aec_take_turn", "aec_together")
    ]
    if not args.no_contract_check:
        for result in results:
            validate(result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
