#!/usr/bin/env python3
"""Align-CRUSE frame-by-frame streaming inference -- one STFT hop at a time.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify

Same arguments and checkpoint handling as denoise.py; the difference is the
execution schedule.  Audio is consumed in hop-sized chunks, converted with the
incremental StreamSTFT, run through ``AlignCRUSE.forward_stream`` one frame at
a time (all time context lives in the explicit state cells), and reconstructed
with the incremental StreamISTFT.  ``stream_output_delay`` is 0: each hop's
mask is emitted immediately.

This candidate consumes raw unaligned mic/far spectra -- it has no linear-AEC
(PBFDKF) frontend, so the entire model path streams here.  For the fronted
candidates the PBFDKF frontend is a separate C-side streaming seam; this CLI
family verifies the NN streaming only.

On startup it prints the per-invocation I/O table (tensor name/shape/dtype for
one step) and the state inventory after the first frame -- together these are
the deployment RAM/IO contract.  With --verify it also runs the offline
whole-utterance forward and reports max-abs / RMS waveform differences
(expected within ~1e-5 for utterances of at least max_delay_frames hops; see
forward_stream's short-utterance caveat).
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

from AIAEC.Align_CRUSE.denoise import load_model
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
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--verify', action='store_true',
                        help='also run the offline forward and print waveform differences')
    return parser


def _io_table(step_inputs, output) -> str:
    rows = ["  per-invocation I/O (one forward_stream step):"]
    entries = list(step_inputs.items())
    entries += [("enhanced", output.enhanced), ("mask", output.mask),
                ("delay_distribution", output.delay_distribution)]
    for name, tensor in entries:
        rows.append(f"    {name:<22s} {str(tuple(tensor.shape)):<16s} "
                    f"{str(tensor.dtype).replace('torch.', '')}")
    return "\n".join(rows)


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
    synth = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    chunks = []
    emitted = 0
    frames = 0
    mic_pending, far_pending = [], []

    def run_ready():
        nonlocal emitted, frames
        while mic_pending and far_pending:
            mic_frame = mic_pending.pop(0).unsqueeze(1)   # complex [B,1,F]
            far_frame = far_pending.pop(0).unsqueeze(1)
            with torch.no_grad():
                output = model.forward_stream(mic_frame, far_frame, state)
            if frames == 0:
                print(_io_table({"microphone": mic_frame,
                                 "far_end": far_frame}, output))
                print("  state inventory after first frame:")
                print(state_report(state))
            frames += 1
            piece = synth.push(output.enhanced[:, 0])
            emitted += piece.shape[-1]
            chunks.append(piece)

    for start in range(0, length, grid.hop_len):
        stop = min(start + grid.hop_len, length)
        mic_pending += mic_stft.push(mic_t[:, start:stop])
        far_pending += far_stft.push(far_t[:, start:stop])
        run_ready()
    mic_pending += mic_stft.flush()
    far_pending += far_stft.flush()
    run_ready()
    chunks.append(synth.flush(length=length, already_emitted=emitted))

    enhanced = torch.cat(chunks, dim=-1)[:, :length]
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames} frames streamed)")

    if args.verify:
        mic_spec = stft(mic_t, grid).transpose(-2, -1)
        far_spec = stft(far_t, grid).transpose(-2, -1)
        with torch.no_grad():
            offline = model(microphone=mic_spec, far_end=far_spec)
        reference = istft(offline.enhanced.transpose(-2, -1), grid,
                          length=length)
        diff = (enhanced - reference).abs()
        rms = (enhanced - reference).square().mean().sqrt()
        print(f"verify vs offline: max-abs {diff.max().item():.3e}, "
              f"RMS {rms.item():.3e}")


if __name__ == '__main__':
    main(build_parser().parse_args())
