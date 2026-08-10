#!/usr/bin/env python3
"""DeepFilterNet-AENR inference -- run a trained checkpoint on one mic/far pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu

mic.wav / far.wav must be mono and at the checkpoint's sample rate (the
checkpoint's contract sr, 48 kHz or 16 kHz -- resample first if your capture
is at a different rate).

Like ../GTCRN_AENR/denoise.py, this candidate's input is the FROZEN
PRODUCTION LINEAR AEC's error, not the raw microphone: this script runs that
reference engine over the whole file as one continuous cold-start stream (a
fresh LinearAecEngine per call IS the correct single-utterance case -- see
GTCRN_AENR/denoise.py's note).

The DFN feature normalisers (extract_dfn2_features) run with a fresh EMA
state at the start of the file, same as training (see train.py's top-of-file
note on why cross-chunk state threading is out of scope) -- error and far
each get their OWN independent state, never a shared one.

Output is the common denoised, echo-free, early/dereverberated near-end speech
estimate used by every selected AIAEC candidate.

config.ini is not read for model shape: every shape-relevant setting is
recovered from the checkpoint's own contract (train.py's
make_checkpoint_contract), so inference cannot silently drift from what the
weights were trained with. The [feature] normalisation constants likewise
travel with the checkpoint via feature_version -- this script recomputes
them the same way train.py did (AINR.DeepFilterNet2.train.read_feature_config
against its own config.ini), and refuses to run on a mismatched
feature_version rather than silently using the wrong normalisation.
"""

import argparse
import configparser
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AINR.DeepFilterNet2.train import FEATURE_VERSION, extract_dfn2_features, read_feature_config
from AIAEC.DeepFilterNet_AENR import DeepFilterNetAENR
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, istft
from AIAEC.training_common import (
    LinearAecEngine,
    auto_device,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--config', default='config.ini',
                        help='Where to read [feature] from (must match training)')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    return parser


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    require_checkpoint_model_identity(contract, 'DeepFilterNet_AENR')
    if contract.get('feature_version') != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature_version={contract.get('feature_version')!r}, "
            f"this code expects {FEATURE_VERSION!r}; the normalisation "
            f"formula changed since this checkpoint was trained")
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    linear_aec_contract = require_checkpoint_linear_aec(contract, aec_grid)
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    model = DeepFilterNetAENR(model_grid, **model_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, aec_grid, linear_aec_contract


def _normalized_stft(waveform: torch.Tensor, grid: AecGrid) -> torch.Tensor:
    """DFN's own STFT convention -- NOT AIAEC.dataset_gen.stft.

    Matches AIAEC/dataset_gen/model_views.py's private ``normalized_stft``:
    ``torch.stft(normalized=True)`` rather than the plain sqrt-Hann every
    other candidate's public boundary uses. Duplicated here (5 lines) rather
    than imported, since it is a private helper of model_views.py's DFN
    branch, not part of the public dataset_gen API.
    """
    window = grid.window(device=waveform.device, dtype=waveform.dtype)
    return torch.stft(waveform, grid.n_fft, grid.hop_len, grid.win_len,
                      window=window, normalized=True, return_complex=True)


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(args.checkpoint, device)

    feature_cfg_ini = configparser.ConfigParser()
    if not feature_cfg_ini.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    feature_cfg = read_feature_config(feature_cfg_ini, grid.sr, grid.hop_len)

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

    linear_aec = LinearAecEngine(
        n_lanes=1, sample_rate=grid.sr, contract=linear_contract
    )
    error, _echo_estimate = linear_aec(mic_t, far_t, grid.sr)

    error_spec = _normalized_stft(error, grid)
    far_spec = _normalized_stft(far_t[:, :error.shape[-1]], grid)
    _, error_erb, error_feat, _ = extract_dfn2_features(
        error_spec, model.erb_fb, model.df_bins, feature_cfg=feature_cfg, ema_state=None)
    _, far_erb, far_feat, _ = extract_dfn2_features(
        far_spec, model.erb_fb, model.df_bins, feature_cfg=feature_cfg, ema_state=None)

    with torch.no_grad():
        output = model(
            linear_error=error_spec.transpose(1, 2),   # [B,T,F], the public model boundary
            error_erb=error_erb, error_spec=error_feat,
            far_erb=far_erb, far_spec=far_feat,
        )

    # normalized=True mirrors _normalized_stft() above; without it the output
    # is silently attenuated by 1/sqrt(n_fft).
    enhanced = istft(output.enhanced.transpose(-2, -1), grid, length=length,
                     normalized=True)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
