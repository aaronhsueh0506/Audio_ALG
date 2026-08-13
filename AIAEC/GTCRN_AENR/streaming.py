#!/usr/bin/env python3
"""GTCRN-AENR frame-by-frame (N=1) streaming inference.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify

Same arguments and checkpoint handling as denoise.py, but the NN runs one STFT
frame at a time: StreamSTFT -> forward_stream -> StreamISTFT, with all
time-context state held in the dict returned by create_stream_state().

The PBFDKF linear-AEC frontend still runs offline over the whole file exactly
as in denoise.py.  Streaming that frontend is a separate C-side seam (the
production linear AEC is already a streaming C implementation); this CLI
verifies the NN streaming decomposition only.

--verify additionally runs the offline whole-wav forward and reports the
max-abs and RMS difference between the two output waveforms.
"""

import argparse
import os
import sys
from collections import deque

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.GTCRN_AENR.denoise import load_model
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft, sqrt_hann_window, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import LinearAecEngine, auto_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--verify', action='store_true',
                        help='also run the offline whole-wav forward and print '
                             'max-abs / RMS waveform differences')
    return parser


def _print_io_table(inputs, outputs):
    print('per-invocation I/O (one forward_stream step):')
    for direction, table in (('in ', inputs), ('out', outputs)):
        for name, tensor in table:
            print(f"  {direction} {name:<22s} {str(tuple(tensor.shape)):<18s} "
                  f"{str(tensor.dtype).replace('torch.', '')}")


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(args.checkpoint, device)

    mic_t, far_t, source_rates = load_mic_far(
        args.mic_wav, args.far_wav, grid.sr
    )
    mic_t = mic_t.to(device)
    far_t = far_t.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled mic/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    length = mic_t.shape[-1]

    # Offline frontend, identical to denoise.py (see module docstring).
    linear_aec = LinearAecEngine(
        n_lanes=1, sample_rate=grid.sr, contract=linear_contract
    )
    error, _echo_estimate = linear_aec(mic_t, far_t, grid.sr)

    window = sqrt_hann_window(grid.win_len, device=device)
    stft_err = StreamSTFT(grid.n_fft, grid.hop_len, window)
    stft_far = StreamSTFT(grid.n_fft, grid.hop_len, window)
    istft_out = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    err_frames = deque()
    far_frames = deque()
    emitted_chunks = []
    emitted_samples = 0
    frames_done = 0

    def drain():
        nonlocal emitted_samples, frames_done
        while err_frames and far_frames:
            err_frame = err_frames.popleft().unsqueeze(1)   # [B,1,F]
            far_frame = far_frames.popleft().unsqueeze(1)
            with torch.no_grad():
                out = model.forward_stream(
                    linear_error=err_frame, far_end=far_frame, state=state)
            samples = istft_out.push(out.enhanced[:, 0])
            if samples.shape[-1]:
                emitted_chunks.append(samples)
                emitted_samples += samples.shape[-1]
            frames_done += 1
            if frames_done == 1:
                _print_io_table(
                    [('linear_error', err_frame), ('far_end', far_frame)],
                    [('enhanced', out.enhanced), ('mask', out.mask),
                     ('erb_complex_mask', out.auxiliary['erb_complex_mask'])],
                )
                print('stream state after first frame:')
                print(state_report(state))

    for start in range(0, length, grid.hop_len):
        stop = start + grid.hop_len
        err_frames.extend(stft_err.push(error[:, start:stop]))
        far_frames.extend(stft_far.push(far_t[:, start:stop]))
        drain()
    err_frames.extend(stft_err.flush())
    far_frames.extend(stft_far.flush())
    drain()
    tail = istft_out.flush(length=length, already_emitted=emitted_samples)
    if tail.shape[-1]:
        emitted_chunks.append(tail)

    streamed = torch.cat(emitted_chunks, dim=-1)[:, :length]
    sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames_done} frames streamed)")

    if args.verify:
        error_spec = stft(error, grid).transpose(-2, -1)
        far_spec = stft(far_t, grid).transpose(-2, -1)
        with torch.no_grad():
            output = model(linear_error=error_spec, far_end=far_spec)
        offline = istft(output.enhanced.transpose(-2, -1), grid, length=length)
        diff = (offline - streamed).abs()
        rms = (offline - streamed).square().mean().sqrt()
        print(f"verify vs offline forward: max-abs {diff.max().item():.3e}, "
              f"RMS {rms.item():.3e}")


if __name__ == '__main__':
    main(build_parser().parse_args())
