#!/usr/bin/env python3
"""DeepVQE-S frame-by-frame streaming inference on one mic/far-end pair.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify

Same contract as denoise.py (checkpoint-recovered grid, mono inputs resampled
to the model rate), but STFT -> model -> ISTFT runs fully frame-by-frame:
audio is fed one hop at a time through StreamSTFT, every emitted frame goes
through DeepVQES.forward_stream with an explicit state dict, and StreamISTFT
reconstructs samples incrementally.

DeepVQE-S is end-to-end: unlike the RES+NR candidates there is no linear-AEC
(PBFDKF) frontend stage before the network.  For candidates that carry such a
frontend, streaming that filter is a separate C-side seam and would run
offline here regardless -- this CLI verifies the NN streaming only.

On startup the CLI prints the per-invocation I/O contract (one step's tensor
names, shapes, dtypes) and the persistent-state inventory; with --verify it
also runs the offline whole-wav forward and reports max-abs / RMS waveform
difference against the streamed output.
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

from AIAEC.DeepVQE_S.denoise import load_model
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import auto_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('mic_wav')
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--verify', action='store_true',
                        help='also run the offline forward and print the '
                             'max-abs / RMS waveform difference')
    return parser


def _print_io_table(mic_frame, far_frame, output):
    rows = [
        ('in  microphone', mic_frame),
        ('in  far_end', far_frame),
        ('out enhanced', output.enhanced),
        ('out delay_distribution', output.delay_distribution),
        ('out ccm_taps', output.auxiliary['ccm_taps']),
    ]
    print('per-invocation I/O (one STFT frame per step):')
    for name, tensor in rows:
        dtype = str(tensor.dtype).replace('torch.', '')
        print(f'  {name:<24s} {str(tuple(tensor.shape)):<20s} {dtype}')


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

    window = grid.window(device=device)
    mic_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
    far_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
    out_istft = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    pieces = []
    emitted = 0
    printed_contract = False

    def run_frames(mic_frames, far_frames):
        nonlocal emitted, printed_contract
        assert len(mic_frames) == len(far_frames)
        for mic_frame, far_frame in zip(mic_frames, far_frames):
            microphone = mic_frame.unsqueeze(1)   # [B,1,F] complex
            far_end = far_frame.unsqueeze(1)
            output = model.forward_stream(microphone, far_end, state)
            if not printed_contract:
                _print_io_table(microphone, far_end, output)
                print('persistent state after first frame:')
                print(state_report(state))
                printed_contract = True
            samples = out_istft.push(output.enhanced[:, 0])
            emitted += samples.shape[-1]
            pieces.append(samples)

    with torch.no_grad():
        # One hop of audio per push mirrors the deployment cadence; both
        # analyses see identical chunk boundaries so frames arrive in lockstep.
        for start in range(0, length, grid.hop_len):
            stop = min(start + grid.hop_len, length)
            run_frames(mic_stft.push(mic_t[:, start:stop]),
                       far_stft.push(far_t[:, start:stop]))
        run_frames(mic_stft.flush(), far_stft.flush())
        tail = out_istft.flush(length=length, already_emitted=emitted)
        pieces.append(tail)

    streamed = torch.cat(pieces, dim=-1)[:, :length]

    if args.verify:
        mic_spec = stft(mic_t, grid).transpose(-2, -1)
        far_spec = stft(far_t, grid).transpose(-2, -1)
        with torch.no_grad():
            offline = model(microphone=mic_spec, far_end=far_spec)
        reference = istft(offline.enhanced.transpose(-2, -1), grid,
                          length=length)
        diff = (streamed - reference).abs()
        rms = diff.square().mean().sqrt()
        print(f"verify vs offline: max-abs {diff.max().item():.3e}  "
              f"RMS {rms.item():.3e}")

    sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"streamed {out_istft.frames_seen} frames)")


if __name__ == '__main__':
    main(build_parser().parse_args())
