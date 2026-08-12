#!/usr/bin/env python3
"""Align-CRUSE inference -- run a trained checkpoint on one mic/far-end pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before inference when needed; output stays at that model rate. Output is
the joint end-to-end AEC+RES+NR estimate of near_target --
denoised, dereverberated and echo-cancelled near speech (see ../README.md's
decision matrix). This candidate's earlier AEC-only, noise-preserving route
was retired.

config.ini is not read here: every shape-relevant setting (grid, model
kwargs) is recovered from the checkpoint's own contract (see train.py /
AIAEC/training_common.py's make_checkpoint_contract), so inference cannot
silently drift from what the weights were trained with.
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

from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, istft, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import auto_device, require_checkpoint_model_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    return parser


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    require_checkpoint_model_identity(contract, 'Align_CRUSE')
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    model = AlignCRUSE(model_grid, **model_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, aec_grid


def main(args):
    device = auto_device(args.device)
    model, grid = load_model(args.checkpoint, device)

    mic_t, far_t, source_rates = load_mic_far(
        args.mic_wav, args.far_wav, grid.sr
    )
    mic_t = mic_t.to(device)
    far_t = far_t.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled mic/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    length = mic_t.shape[-1]

    mic_spec = stft(mic_t, grid).transpose(-2, -1)   # [B,T,F], the public model boundary
    far_spec = stft(far_t, grid).transpose(-2, -1)

    with torch.no_grad():
        output = model(microphone=mic_spec, far_end=far_spec)

    enhanced = istft(output.enhanced.transpose(-2, -1), grid, length=length)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
