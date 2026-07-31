#!/usr/bin/env python3
"""Align-CRUSE inference -- run a trained checkpoint on one mic/far-end pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu

mic.wav / far.wav must be mono and at the checkpoint's sample rate (the
checkpoint's contract sr; resample first if your capture is at a different
rate). Output is the AEC/RES-only estimate of near_speech + local_noise --
this candidate deliberately preserves background noise for a later,
independent NR stage (see ../README.md's decision matrix); it is not a
"clean speech" denoiser.

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
from AIAEC.training_common import auto_device


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

    mic_spec = stft(mic_t, grid).transpose(-2, -1)   # [B,T,F], the public model boundary
    far_spec = stft(far_t, grid).transpose(-2, -1)

    with torch.no_grad():
        output = model(microphone=mic_spec, far_end=far_spec)

    enhanced = istft(output.enhanced.transpose(-2, -1), grid, length=length)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
