#!/usr/bin/env python3
"""Align-ULCNet inference -- run a trained checkpoint on one mic/far-end pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu --preset balanced

mic.wav / far.wav must be mono and at the checkpoint's sample rate (resample
first if your capture is at a different rate).

This candidate's input is the FROZEN PRODUCTION LINEAR AEC's error, not the
raw microphone -- see AIAEC/training_common.py's LinearAecEngine. This
script runs that same reference Python engine (RES+CNG disabled) over the
whole file as ONE continuous stream before calling the model, which is the
correct single-utterance case for LinearAecEngine (a fresh engine per call
IS a cold start, matching the file's own beginning -- no cross-call state to
carry, unlike training's per-lane persistence across chunks).

config.ini is not read for model shape: every shape-relevant setting is
recovered from the checkpoint's own contract (train.py's
make_checkpoint_contract), so inference cannot silently drift from what the
weights were trained with. ``--preset`` selects which linear-AEC preset runs
in front, independent of the checkpoint (the preset is a deployment choice,
not a trained parameter).
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
from AIAEC.training_common import LinearAecEngine, auto_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--preset', default='balanced',
                        choices=('mild', 'balanced', 'aggressive'),
                        help='Frozen linear-AEC preset run in front of the model')
    return parser


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, aec_grid


def main(args):
    device = auto_device(args.device)
    model, grid = load_model(args.checkpoint, device)

    mic, mic_sr = sf.read(args.mic_wav, dtype='float32')
    far, far_sr = sf.read(args.far_wav, dtype='float32')
    if mic_sr != grid.sr or far_sr != grid.sr:
        raise ValueError(
            f"mic/far sample rate ({mic_sr}/{far_sr}) must equal the "
            f"checkpoint's grid rate ({grid.sr}); resample before calling this")
    if mic.ndim > 1 or far.ndim > 1:
        raise ValueError("mic/far must be mono")

    mic_t = torch.from_numpy(mic).unsqueeze(0).to(device)
    far_t = torch.from_numpy(far).unsqueeze(0).to(device)
    length = mic_t.shape[-1]

    linear_aec = LinearAecEngine(n_lanes=1, sample_rate=grid.sr, preset=args.preset)
    error, _echo_estimate = linear_aec(mic_t, far_t, grid.sr)   # same length as mic_t

    error_spec = stft(error, grid).transpose(-2, -1)   # [B,T,F], the public model boundary
    far_spec = stft(far_t, grid).transpose(-2, -1)

    with torch.no_grad():
        output = model(linear_error=error_spec, far_end=far_spec)

    enhanced = istft(output.enhanced.transpose(-2, -1), grid, length=length)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
