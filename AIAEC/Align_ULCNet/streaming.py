#!/usr/bin/env python3
"""Align-ULCNet frame-by-frame streaming inference -- deployment reference.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify
    python3 streaming.py checkpoint.pth kf_error.wav far.wav out.wav \\
        --input-is-linear-error

Checkpoint loading and the audio frontend are identical to denoise.py, far
branch included: the model is fed the linear AEC's own aligned-far seam, not
the raw far WAV.  The frozen PBFDKF linear AEC still runs OFFLINE over the
whole file: streaming that engine is a separate C-side seam (the production
linear AEC is already a streaming C implementation), so this CLI verifies the
NN streaming path only.

Everything after the linear AEC is strictly incremental: hop-sized sample
chunks feed StreamSTFT, every finished frame goes through forward_stream (one
complex [B,1,F] frame in, one out), and StreamISTFT reconstructs samples as
they become final.  On startup the per-invocation I/O table and the state
inventory (the deployment RAM contract) are printed.  ``--verify`` also runs
the offline whole-utterance forward and reports the waveform difference.
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

from AIAEC.Align_ULCNet.denoise import load_model
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft, stft
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC.training_common import LinearAecEngine, auto_device


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
    parser.add_argument('--device', default=None,
                        help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument(
        '--input-is-linear-error', action='store_true',
        help='Evaluation only: first WAV is an existing KF/AEC error Z; '
             'bypass this project\'s PBFDKF and run only the neural post-filter',
    )
    parser.add_argument(
        '--max-delay-frames', type=int, default=None,
        help='Deployment override for the alignment search depth D. The '
             'checkpoint contract stays the source of truth; when this '
             'differs, only the alignment depth is rebuilt (weights are '
             'D-agnostic, but the output is NOT numerically identical '
             'across D).',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Also run the offline whole-utterance forward and print the '
             'max-abs and RMS difference between the two output waveforms',
    )
    return parser


def _print_io_table(inputs, output):
    def row(name, tensor):
        print(f"  {name:<28s} {str(tuple(tensor.shape)):<18s} "
              f"{str(tensor.dtype).replace('torch.', '')}")

    print("per-invocation I/O (one forward_stream step):")
    for name, tensor in inputs.items():
        row(f"in  {name}", tensor)
    row("out enhanced", output.enhanced)
    row("out mask", output.mask)
    row("out delay_distribution", output.delay_distribution)
    for name, tensor in output.auxiliary.items():
        row(f"out auxiliary[{name}]", tensor)


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(
        args.checkpoint, device, max_delay_frames=args.max_delay_frames
    )

    if args.input_is_linear_error:
        error, far, source_rates = load_linear_error_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        print("using external linear-error input; PBFDKF bypassed")
    else:
        mic, far, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        linear_aec = LinearAecEngine(
            n_lanes=1, sample_rate=grid.sr, contract=linear_contract
        )
        error, _echo_estimate = linear_aec(mic, far, grid.sr)
        # Same seam denoise.py uses, and the same one the production C
        # pipelines expose: the exact far hop PBFDKF consumed (raw until the
        # alignment ring can serve the applied delay, ring-aligned after).
        far = linear_aec.get_aligned_far()
    error = error.to(device)
    far = far.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled primary/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    if error.shape != far.shape:
        raise RuntimeError(
            f"error/far lengths differ ({error.shape[-1]}/{far.shape[-1]})"
        )
    length = error.shape[-1]

    window = grid.window(device=device)
    stft_error = StreamSTFT(grid.n_fft, grid.hop_len, window)
    stft_far = StreamSTFT(grid.n_fft, grid.hop_len, window)
    istft_out = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    pieces = []
    emitted = 0
    frames_done = 0

    def run_frames(error_frames, far_frames):
        nonlocal emitted, frames_done
        # Hard check, not `assert` (must survive python -O): zip() would
        # silently truncate to the shorter list and drop frames.
        if len(error_frames) != len(far_frames):
            raise RuntimeError(
                f"error/far frame lists diverged: {len(error_frames)} error "
                f"frames vs {len(far_frames)} far frames"
            )
        for error_frame, far_frame in zip(error_frames, far_frames):
            inputs = {
                "linear_error": error_frame.unsqueeze(1),   # complex [B,1,F]
                "far_end": far_frame.unsqueeze(1),
            }
            with torch.no_grad():
                out = model.forward_stream(state=state, **inputs)
            if frames_done == 0:
                _print_io_table(inputs, out)
            samples = istft_out.push(out.enhanced[:, 0])
            emitted += samples.shape[-1]
            pieces.append(samples)
            frames_done += 1
            if frames_done == 1:
                print("streaming state after the first frame:")
                print(state_report({
                    **state,
                    "stft_error": stft_error,
                    "stft_far": stft_far,
                    "istft": istft_out,
                }))

    # Hop-sized sample chunks: one new STFT frame per chunk once primed.
    for start in range(0, length, grid.hop_len):
        stop = start + grid.hop_len
        run_frames(stft_error.push(error[:, start:stop]),
                   stft_far.push(far[:, start:stop]))
    run_frames(stft_error.flush(), stft_far.flush())
    pieces.append(istft_out.flush(length=length, already_emitted=emitted))
    streamed = torch.cat(pieces, dim=-1)[:, :length]

    sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames_done} frames)")

    if args.verify:
        error_spec = stft(error, grid).transpose(-2, -1)
        far_spec = stft(far, grid).transpose(-2, -1)
        with torch.no_grad():
            reference = model(linear_error=error_spec, far_end=far_spec)
        offline = istft(reference.enhanced.transpose(-2, -1), grid,
                        length=length)
        difference = streamed - offline
        print(f"verify vs offline forward: max-abs "
              f"{difference.abs().max().item():.3e}, RMS "
              f"{difference.square().mean().sqrt().item():.3e}")


if __name__ == '__main__':
    main(build_parser().parse_args())
