#!/usr/bin/env python3
"""Internal entry-point machinery shared by the AIAEC inference CLIs.

Three of the four candidates take the same microphone/far-end WAV pair, load
their weights the same way and run the same hop-by-hop STFT -> forward_stream
-> ISTFT schedule; only the model class and the prose differ.  Both halves of
that CLI live here once: ``make_inference_cli`` builds the public entry point
and ``make_streaming_main`` builds the deployment-reference schedule behind it.

Align-ULCNet keeps its own pair of files because a PBFDKF frontend runs inside
its hop loop; it reuses the pieces below that do not depend on that seam.

This is the CLI factory.  ``AIAEC/inference_common.py`` -- no leading
underscore -- is the separate module holding the WAV loading/resampling
helpers both hop loops call.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(_THIS_DIR)
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.aiaec_common import SignalGrid
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import AecGrid, istft, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import auto_device, require_checkpoint_model_identity


DEVICE_HELP = 'cuda / cpu / mps (default: auto-detect)'

# --verify is a diagnostic print, deliberately not a gate: it has no tolerance
# and never changes the exit code.  Streaming equivalence is gated by the
# test_stream_matches_offline tests, which pin measured tolerances.
VERIFY_HELP = (
    'diagnostics only: also run the whole-utterance offline forward and print '
    'the max-abs / RMS waveform difference. No tolerance is applied and the '
    'exit code never changes -- the streaming-equivalence gate is the '
    'test_stream_matches_offline test, not this flag'
)


def require_matched_frames(primary_frames, far_frames,
                           primary_label='mic', far_label='far'):
    """Refuse a primary/far frame-count divergence before ``zip`` hides it.

    A hard check, not ``assert`` (it must survive ``python -O``): ``zip``
    truncates to the shorter list and would silently drop frames.
    """
    if len(primary_frames) != len(far_frames):
        raise RuntimeError(
            f"{primary_label}/{far_label} frame lists diverged: "
            f"{len(primary_frames)} {primary_label} frames vs "
            f"{len(far_frames)} {far_label} frames"
        )


def print_io_table(inputs, output):
    """Print one streaming step's tensor contract (the deployment I/O table)."""
    rows = [('in ', name, tensor) for name, tensor in inputs.items()]
    rows.append(('out', 'enhanced', output.enhanced))
    if output.mask is not None:
        rows.append(('out', 'mask', output.mask))
    if output.delay_distribution is not None:
        rows.append(('out', 'delay_distribution', output.delay_distribution))
    rows += [('out', f'auxiliary[{key}]', value)
             for key, value in output.auxiliary.items()]
    print('per-invocation I/O (one forward_stream step):')
    for direction, name, tensor in rows:
        print(f"  {direction} {name:<26s} {str(tuple(tensor.shape)):<20s} "
              f"{str(tensor.dtype).replace('torch.', '')}")


def make_streaming_main(model_name):
    """Build the hop-by-hop deployment reference for a mic/far candidate.

    Audio is consumed one hop at a time through StreamSTFT, every emitted STFT
    frame goes through ``forward_stream`` with an explicit state dict, and
    StreamISTFT reconstructs samples incrementally -- the schedule a C/NPU port
    has to reproduce.  The per-invocation I/O table and the state inventory
    after the first frame are printed as that deployment contract.
    """

    def main(args, load_model_fn=None):
        if load_model_fn is None:
            load_model_fn = importlib.import_module(
                'AIAEC.%s.inference' % model_name
            ).load_model
        device = auto_device(args.device)
        model, grid = load_model_fn(args.checkpoint, device)

        microphone, far_end, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        microphone = microphone.to(device)
        far_end = far_end.to(device)
        if source_rates != (grid.sr, grid.sr):
            print(f"resampled mic/far {source_rates[0]}/{source_rates[1]} -> "
                  f"{grid.sr} Hz")
        length = microphone.shape[-1]

        window = grid.window(device=device)
        mic_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
        far_stft = StreamSTFT(grid.n_fft, grid.hop_len, window)
        synthesis = StreamISTFT(grid.n_fft, grid.hop_len, window)
        state = model.create_stream_state()

        pieces = []
        emitted = 0
        frames = 0

        def run_frames(mic_frames, far_frames):
            nonlocal emitted, frames
            require_matched_frames(mic_frames, far_frames)
            for mic_frame, far_frame in zip(mic_frames, far_frames):
                inputs = {
                    'microphone': mic_frame.unsqueeze(1),   # complex [B,1,F]
                    'far_end': far_frame.unsqueeze(1),
                }
                with torch.no_grad():
                    output = model.forward_stream(
                        inputs['microphone'], inputs['far_end'], state
                    )
                if frames == 0:
                    print_io_table(inputs, output)
                    print('state inventory after the first frame:')
                    print(state_report(state))
                frames += 1
                samples = synthesis.push(output.enhanced[:, 0])
                emitted += samples.shape[-1]
                pieces.append(samples)

        # One hop of audio per push mirrors the deployment cadence; both
        # analyses see identical chunk boundaries so frames arrive in lockstep.
        for start in range(0, length, grid.hop_len):
            stop = min(start + grid.hop_len, length)
            run_frames(mic_stft.push(microphone[:, start:stop]),
                       far_stft.push(far_end[:, start:stop]))
        run_frames(mic_stft.flush(), far_stft.flush())
        pieces.append(synthesis.flush(length=length, already_emitted=emitted))

        streamed = torch.cat(pieces, dim=-1)[:, :length]
        if streamed.shape[-1] != length:
            raise RuntimeError(
                f"streamed {streamed.shape[-1]} samples, expected {length}")
        sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
                 subtype='FLOAT')
        print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
              f"{frames} frames)")

        if args.verify:
            mic_spec = stft(microphone, grid).transpose(-2, -1)
            far_spec = stft(far_end, grid).transpose(-2, -1)
            with torch.no_grad():
                offline = model(microphone=mic_spec, far_end=far_spec)
            reference = istft(offline.enhanced.transpose(-2, -1), grid,
                              length=length)
            difference = streamed - reference
            print(f"verify vs offline forward (diagnostics only): max-abs "
                  f"{difference.abs().max().item():.3e}, RMS "
                  f"{difference.square().mean().sqrt().item():.3e}")

    return main


def make_inference_cli(model_name, model_class, description):
    """Build one candidate's public CLI: parser, loader, main and dispatcher.

    ``load_model`` is the single checkpoint-construction site shared by the
    streaming CLI, the exporters and the calibration recorder, so a contract
    check added here reaches all of them.  config.ini is deliberately not read:
    every shape-relevant setting comes from the checkpoint's own contract.

    Returns ``(build_parser, load_model, main, cli)``; each model's
    ``inference.py`` binds them as its module-level entry points.
    """

    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('checkpoint')
        parser.add_argument('mic_wav')
        parser.add_argument('far_wav')
        parser.add_argument('out_wav')
        parser.add_argument('--device', default=None, help=DEVICE_HELP)
        parser.add_argument('--verify', action='store_true', help=VERIFY_HELP)
        return parser

    def load_model(checkpoint_path: str, device: str):
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        contract = ckpt['contract']
        require_checkpoint_model_identity(contract, model_name)
        aec_grid = AecGrid(contract['sr'], contract['n_fft'],
                           contract['win_len'], contract['hop_len'])
        model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft,
                                aec_grid.win_len, aec_grid.hop_len)
        model_kwargs = {
            key[len('ctor_'):]: value for key, value in contract.items()
            if key.startswith('ctor_')
        }
        model = model_class(model_grid, **model_kwargs).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, aec_grid

    def main(args):
        # The audio schedule lives in _streaming.py so the public CLI executes
        # the same hop-by-hop implementation the streaming tests drive.
        streaming = importlib.import_module(
            'AIAEC.%s._streaming' % model_name
        )
        streaming.main(args, load_model_fn=load_model)

    def cli():
        if len(sys.argv) > 1 and sys.argv[1] == 'calib':
            del sys.argv[1]
            from AIAEC._streaming_calibration import main as calibration_main
            calibration_main(model_name)
            return
        main(build_parser().parse_args())

    return build_parser, load_model, main, cli
