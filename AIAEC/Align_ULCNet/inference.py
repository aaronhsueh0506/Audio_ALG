#!/usr/bin/env python3
"""Align-ULCNet streaming inference on one mic/far-end pair.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --device cpu
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify
    python3 inference.py checkpoint.pth kf_error.wav far.wav out.wav \\
        --input-is-linear-error

目錄批次 (--mic-dir 取代三個位置參數):
    # 沒帶 --ref-dir: reference 為數位靜音, 等同純降噪
    python3 inference.py checkpoint.pth --mic-dir /path/to/mic \\
        --out-dir /path/to/out
    # 帶了 --ref-dir: 逐檔以檔名配對, stem 的 mic 換成 lpb
    python3 inference.py checkpoint.pth --mic-dir /path/to/mic \\
        --ref-dir /path/to/lpb --out-dir /path/to/out

Calibration (also exports the ONNX graph the tensors bind to; the default
graph path is <output>.onnx, override with --onnx):
    python3 inference.py calib --checkpoint checkpoint.pth \\
        --primary-dir /path/to/linear_error --far-dir /path/to/raw_far \\
        --frames 8192 --max-delay-frames 8 --format bin \\
        --output calib/align_ulcnet_d8

Calibration straight from RAW microphone recordings: add --primary-is-mic
and point --primary-dir at the mic WAVs. The checkpoint-matched frozen
PBFDKF derives the linear-error stems in-process and persists them beside
the artifact (<output>_linear_error/); the manifest records
primary_source=raw_mic_via_frozen_pbfdkf:
    python3 inference.py calib --checkpoint checkpoint.pth \\
        --primary-dir /path/to/raw_mic --far-dir /path/to/raw_far \\
        --primary-is-mic --frames 8192 --max-delay-frames 8 --format bin \\
        --output calib/align_ulcnet_d8

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before the linear AEC when needed; output stays at that model rate.

This candidate's input is the FROZEN PRODUCTION LINEAR AEC's error, not the
raw microphone -- see AIAEC/training_common.py's LinearAecEngine. This
script runs that same reference Python engine (RES+CNG disabled) one hop at a
time, followed by incremental STFT, one-frame model inference, and incremental
ISTFT.  A fresh process starts with cold AEC/model state; state then persists
for the complete file exactly as it does in deployment.

Output is the common denoised, echo-free, early/dereverberated near-end speech
estimate used by every selected AIAEC candidate.

config.ini is not read for model shape: every shape-relevant setting is
recovered from the checkpoint's own contract, including the exact materialized
Python-PBFDKF frontend. Inference refuses a missing or drifted AEC contract.

``--input-is-linear-error`` is an evaluation-only bypass for published demos
that already provide KF residual Z. It does not make an external KF equivalent
to the frozen PBFDKF used to train this repository's checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC._cli_common import (
    DEVICE_HELP,
    VERIFY_HELP,
    add_directory_arguments,
    require_single_or_directory,
    run_pipeline,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid
from AIAEC.training_common import (
    checkpoint_far_input_mode,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
)


def build_parser() -> argparse.ArgumentParser:
    # Raw: the docstring is a usage block whose line breaks carry meaning.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument(
        'mic_wav', metavar='mic_or_linear_error_wav', nargs='?',
        help='Microphone input, or precomputed error with '
             '--input-is-linear-error',
    )
    parser.add_argument('far_wav', nargs='?')
    parser.add_argument('out_wav', nargs='?')
    parser.add_argument('--device', default=None, help=DEVICE_HELP)
    parser.add_argument(
        '--input-is-linear-error', action='store_true',
        help='Evaluation only: first WAV is an existing KF/AEC error Z; '
             'bypass this project\'s PBFDKF and run only the neural post-filter. '
             'The supplied far WAV must already be aligned to that error.',
    )
    parser.add_argument(
        '--max-delay-frames', type=int, default=None,
        help='Deployment override for the alignment search depth D. The '
             'checkpoint contract stays the source of truth; when this '
             'differs, only the alignment depth is rebuilt (weights are '
             'D-agnostic, but the output is NOT numerically identical '
             'across D).',
    )
    parser.add_argument('--verify', action='store_true', help=VERIFY_HELP)
    add_directory_arguments(parser)
    return parser


def load_model(checkpoint_path: str, device: str,
               max_delay_frames: int | None = None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    require_checkpoint_model_identity(contract, 'Align_ULCNet')
    # Missing field defaults to 'raw_far' (legacy checkpoints); an unknown
    # recorded mode is rejected here, before any weights load. _streaming.py
    # shares this loader, so both CLIs print the mode at load time.
    print("checkpoint training far_input_mode: "
          f"{checkpoint_far_input_mode(contract)}; deployment: aligned_far")
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    linear_aec_contract = require_checkpoint_linear_aec(contract, aec_grid)
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    # The checkpoint contract stays the source of truth for the model shape.
    # Build from the contract first so the checkpoint's own (possibly
    # grid-derived) alignment depth D is known before any override applies.
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    if (max_delay_frames is not None
            and int(max_delay_frames) != model.max_delay_frames):
        # Deployment-only override of D. It rebuilds the model with ONLY
        # max_delay_frames replaced; every other constructor argument still
        # comes from the contract, so nothing else can silently drift. D
        # never enters weight shapes (the delay attention projections and
        # score conv are D-agnostic), so the strict load below still
        # verifies every tensor.
        print(f"deployment override: max_delay_frames "
              f"{model.max_delay_frames} -> {int(max_delay_frames)} "
              f"(weights are D-agnostic; output is NOT numerically "
              f"identical across D)")
        override_kwargs = dict(model_kwargs,
                               max_delay_frames=int(max_delay_frames))
        model = AlignULCNet(model_grid, **override_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model, aec_grid, linear_aec_contract


def _run_one(args):
    # Keep checkpoint construction here as the single source used by export
    # and calibration.  The audio schedule itself lives in _streaming.py so
    # the public CLI executes the same hop-by-hop implementation used by tests.
    from AIAEC.Align_ULCNet._streaming import main as streaming_main
    streaming_main(args, load_model_fn=load_model)


def main(args):
    run_pipeline(args, _run_one)


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'calib':
        del sys.argv[1]
        from AIAEC._streaming_calibration import main as calibration_main
        calibration_main('Align_ULCNet')
        return
    parser = build_parser()
    args = parser.parse_args()
    require_single_or_directory(parser, args)
    main(args)


if __name__ == '__main__':
    cli()
