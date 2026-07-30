"""
AECNet inference -- cancel the echo in one recording.

    python3 denoise.py --config config.ini --model output/aecnet_best.pth \
        --mic mic.wav --ref far_end.wav --output aec_out.wav

    # or a single interleaved file, channel 0 = microphone, channel 1 = reference
    python3 denoise.py --config config.ini --model output/aecnet_best.pth \
        --input mic_and_ref.wav --output aec_out.wav --echo-out echo_estimate.wav

⚠ The network emits D_hat, the ESTIMATED ECHO.  What this script writes to
``--output`` is ``E = Y - D_hat``, computed here by SUBTRACTION.  ``--echo-out``
writes D_hat itself, which is the thing to listen to when deciding whether the
canceller is modelling the echo or merely gating the microphone.

⚠ This is a linear canceller only.  E still contains the residual echo
``R = D - D_hat`` and all of the local noise; a residual suppressor and a noise
reducer are separate downstream stages, deliberately.
"""

import argparse
import configparser
import os
import sys
from typing import Tuple

import torch
import torchaudio

from model import AecNet, AecNetConfig, build_model
from train import build_contract, read_loss_config, require_checkpoint_contract

_AIAEC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AINR = os.path.join(os.path.dirname(_AIAEC), 'ainr')
sys.path.insert(0, _AIAEC)   # dataset_gen_aec: the AEC corpus, owned by AIAEC/
sys.path.insert(0, _AINR)    # dataset_gen: the SHARED loader/split/seed + DSP
# ⚠ AIAEC/ deliberately depends on ainr/dataset_gen and must not fork it.  Two
# things live there that cannot be duplicated: the augmentation DSP the AEC corpus
# reuses (RIR, RT60, biquad, clipping), and the train/val split + seeder that every
# model in the repo shares.  A second copy of the split is how two models being
# compared silently end up trained on different corpora -- see dataset_gen/loader.py.
# The package is named dataset_gen_aec, NOT dataset_gen, because both directories
# sit on this sys.path and a same-named package would shadow whichever came second.
from dataset_gen_aec import AecGrid, istft, stft  # noqa: E402


def load_model(config_path: str, model_path: str,
               device=torch.device('cpu')) -> Tuple[AecNet, AecGrid]:
    cfg = configparser.ConfigParser()
    if not cfg.read(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")
    grid = AecGrid.from_config(cfg)
    model_cfg = AecNetConfig.from_config(cfg, grid.frame_rate)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # The same gate the resume path applies.  Shape mismatches are caught by
    # load_state_dict, but an n_fft, lookahead or compress_exponent change that
    # leaves the shapes intact would otherwise run silently on the wrong grid --
    # and a compression mismatch in particular produces output that is merely
    # bad, not obviously broken.
    require_checkpoint_contract(
        ckpt, build_contract(cfg, grid, model_cfg, read_loss_config(cfg)),
        context=model_path)

    model = build_model(cfg, grid).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, grid


def _load_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav[0]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def estimate_echo(model: AecNet, y_spec: torch.Tensor, x_spec: torch.Tensor,
                  block_frames: int = 0) -> torch.Tensor:
    """D_hat for the whole recording, optionally through the streaming path.

    ``block_frames > 0`` feeds the model in blocks while carrying its state,
    which is exactly what a real-time implementation does.  It produces the same
    result as one call -- ``tests/test_aecnet_model.py`` asserts that -- so the
    flag exists to exercise the streaming path on real audio, not to change the
    answer.
    """
    if block_frames <= 0:
        d_hat, _ = model.forward_spec(y_spec, x_spec)
        return d_hat
    state = None
    pieces = []
    for start in range(0, y_spec.shape[-1], block_frames):
        stop = start + block_frames
        piece, state = model.forward_spec(
            y_spec[..., start:stop], x_spec[..., start:stop], state)
        pieces.append(piece)
    return torch.cat(pieces, dim=-1)


def cancel(model: AecNet, grid: AecGrid, mic: torch.Tensor, ref: torch.Tensor,
           block_frames: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E, D_hat)`` as waveforms, both the length of ``mic``."""
    n_samples = mic.shape[-1]
    if ref.shape[-1] < n_samples:
        ref = torch.nn.functional.pad(ref, (0, n_samples - ref.shape[-1]))
    ref = ref[..., :n_samples]

    y_spec = stft(mic.unsqueeze(0), grid)
    x_spec = stft(ref.unsqueeze(0), grid)
    with torch.no_grad():
        d_hat = estimate_echo(model, y_spec, x_spec, block_frames)

    look = model.lookahead
    e_spec = y_spec.clone()
    if look:
        # ⚠ With lookahead L the model's output frame i belongs to input frame
        # i - L (see AecNet.forward).  The last L frames of the recording have
        # no estimate yet and are passed through uncancelled, which is the
        # honest thing to do and matches what a streaming implementation would
        # emit at end-of-stream.
        keep = e_spec.shape[-1] - look
        e_spec[..., :keep] = y_spec[..., :keep] - d_hat[..., look:]
        aligned = torch.zeros_like(y_spec)
        aligned[..., :keep] = d_hat[..., look:]
        d_hat = aligned
    else:
        e_spec = y_spec - d_hat

    e_wav = istft(e_spec, grid, length=n_samples).squeeze(0)
    d_wav = istft(d_hat, grid, length=n_samples).squeeze(0)
    return e_wav, d_wav


def _resolve_inputs(args, sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.input:
        wav, file_sr = torchaudio.load(args.input)
        if wav.shape[0] < 2:
            raise ValueError(
                f"{args.input} has {wav.shape[0]} channel(s); --input expects "
                f"channel 0 = microphone and channel 1 = far-end reference. Use "
                f"--mic/--ref for two separate files.")
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        return wav[0], wav[1]
    if not (args.mic and args.ref):
        raise ValueError("give --input, or both --mic and --ref")
    return _load_mono(args.mic, sr), _load_mono(args.ref, sr)


def main():
    parser = argparse.ArgumentParser(description='AECNet inference')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model', required=True, help='checkpoint .pth')
    parser.add_argument('--mic', default=None, help='microphone wav (Y)')
    parser.add_argument('--ref', default=None, help='far-end reference wav (X)')
    parser.add_argument('--input', default=None,
                        help='2-channel wav: ch0 = microphone, ch1 = reference')
    parser.add_argument('--output', required=True,
                        help='where to write E = Y - D_hat')
    parser.add_argument('--echo-out', default=None,
                        help='optional: write the echo ESTIMATE D_hat itself')
    parser.add_argument('--block-frames', type=int, default=0,
                        help='process in blocks of this many frames, carrying '
                             'the recurrent state (0 = one call). The result is '
                             'the same either way; this exercises the streaming '
                             'path.')
    args = parser.parse_args()

    model, grid = load_model(args.config, args.model)
    mic, ref = _resolve_inputs(args, grid.sr)
    e_wav, d_wav = cancel(model, grid, mic, ref, args.block_frames)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                exist_ok=True)
    torchaudio.save(args.output, e_wav.unsqueeze(0), grid.sr)
    print(f"AEC output (E = Y - D_hat): {args.output}")
    if args.echo_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.echo_out)) or '.',
                    exist_ok=True)
        torchaudio.save(args.echo_out, d_wav.unsqueeze(0), grid.sr)
        print(f"echo estimate (D_hat)     : {args.echo_out}")

    erle = 10.0 * torch.log10(
        mic.pow(2).sum() / (e_wav.pow(2).sum() + 1e-20) + 1e-20)
    print(f"mic-to-output level drop  : {float(erle):.1f} dB")
    print("  ⚠ this is NOT ERLE. It is the total level drop, which includes any "
          "near speech the canceller removed. Judge cancellation on a far-only "
          "recording, or against the separated stems.")


if __name__ == '__main__':
    main()
