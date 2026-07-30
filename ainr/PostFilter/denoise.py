"""PostFilter inference.

Two ways in, because the front-end may or may not be this repo's:

  1. mic + reference -- run the configured front-end, then the post-filter:
        python denoise.py --config config.ini --model output/postfilter_best.pth \
            --mic mic.wav --ref far.wav --output enhanced.wav

  2. already have (E, D_hat) -- e.g. the shipping C canceller produced them:
        python denoise.py --config config.ini --model output/postfilter_best.pth \
            --aec-out e.wav --echo-est d_hat.wav [--ref far.wav] \
            --output enhanced.wav

⚠ Path 2 does NOT check that the supplied (E, D_hat) came from the front-end the
checkpoint was trained behind -- nothing in a wav file records that.  It prints
the checkpoint's frontend_id so the operator can.  Path 1 enforces it.

This script IS the caller, so it applies the preset floor, the attenuation cap
and the gain smoothing from ``[inference]`` (postproc.py).  The network's own
output is unbounded below by design.
"""

import argparse
import configparser
import os
import sys

import torch
import torchaudio

from frontends import build_frontend
from model import build_model, mask_magnitude
from postproc import GainPostProcessor
from train import build_contract, require_checkpoint_contract

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.aec import AecGrid, istft, stft  # noqa: E402


def load_wav(path, sr):
    audio, file_sr = torchaudio.load(path)
    audio = audio[0]
    if file_sr != sr:
        # ⚠ Resampling the reference and the mic independently is fine only
        # because they are resampled identically; the echo delay between them is
        # preserved to within the resampler's own group delay.
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    return audio


def align_lengths(*signals):
    """Truncate to the shortest.  ⚠ Zero-padding instead would invent silence at
    the end of the reference, which reads to the model as a reference dropout."""
    n = min(int(s.shape[-1]) for s in signals)
    return [s[..., :n] for s in signals]


def load_model(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    grid = AecGrid.from_config(cfg)

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    model = build_model(cfg, grid)
    # Same gate as --resume, minus the loss fields: inference does not care what
    # the objective was, but it very much cares about the grid, the feature
    # definition and the mask resolution.
    require_checkpoint_contract(ckpt, build_contract(cfg, grid, model),
                                context=args.model, require_loss=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print(f"PostFilter: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  trained behind frontend_id = {ckpt.get('frontend_id')!r}")
    if ckpt.get('frontend_history'):
        print(f"  ⚠ frontend_history = {ckpt['frontend_history']} -- this "
              f"checkpoint has an out-of-distribution lineage")
    return cfg, grid, model, ckpt


def enhance(args):
    cfg, grid, model, ckpt = load_model(args)
    postproc = GainPostProcessor.from_config(cfg, grid)
    print(f"  caller-side post-processing: {postproc.describe()}")

    if args.aec_out:
        e_wav = load_wav(args.aec_out, grid.sr)
        d_wav = (load_wav(args.echo_est, grid.sr) if args.echo_est
                 else torch.zeros_like(e_wav))
        x_wav = load_wav(args.ref, grid.sr) if args.ref else torch.zeros_like(e_wav)
        e_wav, d_wav, x_wav = align_lengths(e_wav, d_wav, x_wav)
        length = e_wav.shape[-1]
        e_spec = stft(e_wav.unsqueeze(0), grid)
        d_spec = stft(d_wav.unsqueeze(0), grid)
        x_spec = stft(x_wav.unsqueeze(0), grid)
        print(f"  front-end: supplied externally; checkpoint expected "
              f"{ckpt.get('frontend_id')!r} -- unverifiable from wav files")
    else:
        frontend = build_frontend(cfg, grid)
        if frontend.frontend_id != ckpt.get('frontend_id'):
            raise ValueError(
                f"config builds front-end {frontend.frontend_id!r} but the "
                f"checkpoint was trained behind {ckpt.get('frontend_id')!r}; "
                f"the residual distribution differs, so this is not the system "
                f"that was trained. Use --aec-out/--echo-est if you mean to "
                f"drive the model from a different front-end deliberately.")
        y_wav = load_wav(args.mic, grid.sr)
        x_wav = load_wav(args.ref, grid.sr)
        y_wav, x_wav = align_lengths(y_wav, x_wav)
        length = y_wav.shape[-1]
        y_spec = stft(y_wav.unsqueeze(0), grid)
        x_spec = stft(x_wav.unsqueeze(0), grid)
        with torch.no_grad():
            e_spec, d_spec, _ = frontend.process(y_spec, x_spec, None)

    with torch.no_grad():
        mask, _ = model(e_spec, d_spec, x_spec, None)
        gain = model.expand_to_bins(mask)
        gain = postproc(gain)
        enhanced = gain.to(e_spec.dtype) * e_spec if torch.is_complex(gain) \
            else gain.to(e_spec.real.dtype) * e_spec
        wav = istft(enhanced, grid, length=length)

    magnitude = mask_magnitude(mask)
    print(f"  |gain|: mean={magnitude.mean():.3f} "
          f"min={magnitude.min():.3f} max={magnitude.max():.3f} "
          f"(before the caller-side floor)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                exist_ok=True)
    torchaudio.save(args.output, wav.reshape(1, -1), grid.sr)
    print(f"Written: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PostFilter inference')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model', required=True, help='checkpoint .pth')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mic', default=None, help='microphone wav (path 1)')
    parser.add_argument('--ref', default=None, help='far-end reference wav')
    parser.add_argument('--aec-out', default=None,
                        help='E, an existing AEC output (path 2)')
    parser.add_argument('--echo-est', default=None,
                        help='D_hat that goes with --aec-out; omitted means '
                             'zero, i.e. the model runs as a pure denoiser')
    args = parser.parse_args()

    if args.aec_out:
        if args.mic:
            parser.error('--aec-out and --mic are the two alternative inputs; '
                         'pass one')
    elif not (args.mic and args.ref):
        parser.error('pass --mic + --ref, or --aec-out [+ --echo-est]')
    enhance(args)
