#!/usr/bin/env python3
"""CAGCRN streaming inference -- frame-by-frame twin of denoise.py.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify

Loads the checkpoint exactly like denoise.py, then runs the complete
STFT -> model -> ISTFT chain one hop at a time through StreamSTFT /
forward_stream / StreamISTFT.  On startup it prints the per-invocation I/O
table and, after the first frame, the state inventory -- together these are
the deployment contract an NPU/C port must reproduce.

CAGCRN is an end-to-end candidate: denoise.py feeds the microphone and
far-end reference straight into the network, so there is no linear-AEC
(PBFDKF) frontend to run here.  For candidates that do consume a PBFDKF
error signal, streaming that frontend is a separate C-side seam -- this CLI
verifies the NN streaming.

--verify additionally runs the offline whole-wav forward and prints the
max-abs and RMS difference between the two output waveforms.
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

from AIAEC.CAGCRN.denoise import load_model
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft, sqrt_hann_window, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import auto_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None,
                        help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--verify', action='store_true',
                        help='also run the offline forward and report the '
                             'max-abs / RMS waveform difference')
    return parser


def _print_io_table(mic_frame, far_frame, out):
    rows = [('in ', 'microphone', mic_frame),
            ('in ', 'far_end', far_frame),
            ('out', 'enhanced', out.enhanced),
            ('out', 'mask', out.mask),
            ('out', 'delay_distribution', out.delay_distribution)]
    rows += [('out', f'auxiliary[{key}]', value)
             for key, value in out.auxiliary.items()]
    print('per-invocation I/O (one STFT frame):')
    for direction, name, tensor in rows:
        print(f'  {direction} {name:<28s} {str(tuple(tensor.shape)):<22s} '
              f'{str(tensor.dtype).replace("torch.", "")}')


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

    window = sqrt_hann_window(grid.win_len, device=device)
    mic_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
    far_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
    synthesis = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    pending_mic, pending_far = [], []
    out_chunks = []
    emitted = 0
    frames_done = 0
    printed = False

    def process_pairs():
        nonlocal emitted, frames_done, printed
        # Mic and far are equal-length, so both analyses emit in lockstep.
        while pending_mic and pending_far:
            mic_frame = pending_mic.pop(0).unsqueeze(1)   # complex [B,1,F]
            far_frame = pending_far.pop(0).unsqueeze(1)
            with torch.no_grad():
                out = model.forward_stream(
                    microphone=mic_frame, far_end=far_frame, state=state)
            if not printed:
                _print_io_table(mic_frame, far_frame, out)
                print('state after first frame:')
                print(state_report(state))
                printed = True
            samples = synthesis.push(out.enhanced[:, 0])
            emitted += samples.shape[-1]
            out_chunks.append(samples)
            frames_done += 1

    for start in range(0, length, grid.hop_len):
        stop = min(start + grid.hop_len, length)
        pending_mic.extend(mic_stft.push(mic_t[:, start:stop]))
        pending_far.extend(far_stft.push(far_t[:, start:stop]))
        process_pairs()
    pending_mic.extend(mic_stft.flush())
    pending_far.extend(far_stft.flush())
    process_pairs()
    out_chunks.append(synthesis.flush(length=length, already_emitted=emitted))

    enhanced = torch.cat(out_chunks, dim=-1)[:, :length]
    if enhanced.shape[-1] != length:
        raise RuntimeError(
            f"streamed {enhanced.shape[-1]} samples, expected {length}")
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames_done} frames)")

    if args.verify:
        mic_spec = stft(mic_t, grid).transpose(-2, -1)
        far_spec = stft(far_t, grid).transpose(-2, -1)
        with torch.no_grad():
            offline = model(microphone=mic_spec, far_end=far_spec)
        offline_wav = istft(
            offline.enhanced.transpose(-2, -1), grid, length=length)
        difference = (offline_wav - enhanced).abs()
        print(f"verify vs offline forward: max-abs "
              f"{difference.max().item():.3e}, "
              f"RMS {difference.square().mean().sqrt().item():.3e}")


if __name__ == '__main__':
    main(build_parser().parse_args())
