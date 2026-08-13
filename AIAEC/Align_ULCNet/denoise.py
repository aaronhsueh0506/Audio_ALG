#!/usr/bin/env python3
"""Align-ULCNet inference -- run a trained checkpoint on one mic/far-end pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu
    python3 denoise.py checkpoint.pth kf_error.wav far.wav out.wav \\
        --input-is-linear-error

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before the linear AEC when needed; output stays at that model rate.

This candidate's input is the FROZEN PRODUCTION LINEAR AEC's error, not the
raw microphone -- see AIAEC/training_common.py's LinearAecEngine. This
script runs that same reference Python engine (RES+CNG disabled) over the
whole file as ONE continuous stream before calling the model, which is the
correct single-utterance case for LinearAecEngine (a fresh engine per call
IS a cold start, matching the file's own beginning -- no cross-call state to
carry, unlike training's per-lane persistence across chunks).

Output is the common denoised, echo-free, early/dereverberated near-end speech
estimate used by every selected AIAEC candidate.

config.ini is not read for model shape: every shape-relevant setting is
recovered from the checkpoint's own contract, including the exact materialized
Python-PBFDKF frontend. Inference refuses a missing or drifted AEC contract.

``--input-is-linear-error`` is an evaluation-only bypass for published demos
that already provide KF residual Z. It does not make an external KF equivalent
to the frozen PBFDKF used to train this repository's checkpoint.
"""

import argparse
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, istft, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.inference_common import load_linear_error_far
from AIAEC.training_common import (
    LinearAecEngine,
    auto_device,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument(
        'mic_wav', metavar='mic_or_linear_error_wav',
        help='Microphone input, or precomputed error with '
             '--input-is-linear-error',
    )
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument(
        '--input-is-linear-error', action='store_true',
        help='Evaluation only: first WAV is an existing KF/AEC error Z; '
             'bypass this project\'s PBFDKF and run only the neural post-filter',
    )
    return parser


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    require_checkpoint_model_identity(contract, 'Align_ULCNet')
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    linear_aec_contract = require_checkpoint_linear_aec(contract, aec_grid)
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, aec_grid, linear_aec_contract


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(args.checkpoint, device)

    if args.input_is_linear_error:
        error, far_t, source_rates = load_linear_error_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        print("using external linear-error input; PBFDKF bypassed")
    else:
        mic_t, far_t, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        linear_aec = LinearAecEngine(
            n_lanes=1, sample_rate=grid.sr, contract=linear_contract
        )
        error, _echo_estimate = linear_aec(
            mic_t, far_t, grid.sr
        )
    error = error.to(device)
    far_t = far_t.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled primary/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    length = error.shape[-1]

    error_spec = stft(error, grid).transpose(-2, -1)   # [B,T,F], the public model boundary
    far_spec = stft(far_t, grid).transpose(-2, -1)

    with torch.no_grad():
        output = model(linear_error=error_spec, far_end=far_spec)

    enhanced = istft(output.enhanced.transpose(-2, -1), grid, length=length)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
