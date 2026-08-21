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
import copy
import importlib
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(_THIS_DIR)
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._calibration_common import discover_pairs, wav_inventory
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


# Directory mode pairs microphone and far-end files by the same explicit
# token rule the calibration recorder already exposes as
# --pair-replace mic:lpb, through the same discover_pairs() walk. One rule and
# one walk, so a tree that pairs for calibration also pairs for inference.
MIC_TO_REFERENCE = ('mic', 'lpb')

MIC_DIR_HELP = (
    'run every .wav under this directory instead of the positional trio '
    '(recursive). Requires --out-dir'
)
REF_DIR_HELP = (
    "far-end tree for --mic-dir; omit it to run with a silent reference, "
    "which makes the run pure noise reduction. Files pair by relative name, "
    "with '%s' in a microphone name also matching '%s' here. Anything "
    "unpaired on either side is an error, never a silent fallback" %
    MIC_TO_REFERENCE
)
OUT_DIR_HELP = (
    'where --mic-dir results are written, mirroring the input tree and '
    "keeping each microphone file's own name"
)


def add_directory_arguments(parser: argparse.ArgumentParser) -> None:
    """The batch form of the positional mic/far/out trio."""
    group = parser.add_argument_group('directory mode')
    group.add_argument('--mic-dir', default=None, help=MIC_DIR_HELP)
    group.add_argument('--ref-dir', default=None, help=REF_DIR_HELP)
    group.add_argument('--out-dir', default=None, help=OUT_DIR_HELP)


def resolve_directory_jobs(args):
    """``(mic_path, ref_path_or_None, out_path)`` per microphone file.

    NAMES are resolved for the whole directory before the first file runs, so
    a pairing mistake in the last file fails before anything has been written,
    and discover_pairs reports every mismatch on both sides at once. Audio-
    level problems -- a stereo file, a rate disagreement, a non-finite sample
    -- are still found when that file is read, and abort the batch.
    """
    mic_dir = os.path.abspath(args.mic_dir)
    out_dir = os.path.abspath(args.out_dir)
    if out_dir == mic_dir:
        raise ValueError(
            '--out-dir must differ from --mic-dir: results keep their '
            'microphone file names and would overwrite the inputs'
        )
    if args.ref_dir:
        found = discover_pairs(mic_dir, args.ref_dir, MIC_TO_REFERENCE)
    else:
        found = [(name, path, None)
                 for name, path in sorted(wav_inventory(mic_dir).items())]
    return [(str(mic), None if ref is None else str(ref),
             os.path.join(out_dir, *name.split('/')))
            for name, mic, ref in found]


def run_directory(args, run_one) -> None:
    """Run ``run_one`` over every job, one file at a time.

    Each file gets its own COPY of the parsed arguments with the positional
    trio filled in, so the per-file path is the same code the single-pair CLI
    runs by construction rather than by audit -- directory mode adds a loop,
    not a second implementation.
    """
    jobs = resolve_directory_jobs(args)
    print('%d file(s) from %s%s' % (
        len(jobs), args.mic_dir,
        '' if args.ref_dir else ' with a silent reference (noise reduction)'))
    for index, (mic_path, reference, out_path) in enumerate(jobs, 1):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print('[%d/%d] %s' % (index, len(jobs), os.path.basename(mic_path)))
        job = copy.copy(args)
        job.mic_wav, job.far_wav, job.out_wav = mic_path, reference, out_path
        run_one(job)


def require_single_or_directory(parser, args) -> None:
    """Refuse an invocation that is neither form, or is both.

    Argument SHAPE only, and here rather than in resolve_directory_jobs
    because this is the layer that can answer with a usage message instead of
    a traceback. Whether the directories exist and pair up is a filesystem
    question, and stays where the filesystem is read.
    """
    positional = (args.mic_wav, args.far_wav, args.out_wav)
    if args.mic_dir:
        if any(value is not None for value in positional):
            parser.error('--mic-dir replaces the positional mic/far/out '
                         'arguments; pass one form or the other')
        if not args.out_dir:
            parser.error('--mic-dir requires --out-dir')
        return
    if args.ref_dir or args.out_dir:
        parser.error('--ref-dir/--out-dir apply to --mic-dir only')
    if any(value is None for value in positional):
        parser.error('the mic/far/out arguments are required without '
                     '--mic-dir')


def run_pipeline(args, run_one) -> None:
    """Run one pair or a whole directory, whichever this invocation is.

    The single place that branches, so the four CLIs cannot answer the
    question differently.
    """
    if getattr(args, 'mic_dir', None):
        run_directory(args, run_one)
    else:
        run_one(args)


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
        # Raw: the description is a usage block with line breaks that carry
        # meaning. The default formatter reflows it into one paragraph, so
        # every example the docstring spells out arrives unreadable.
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('checkpoint')
        # Optional so --mic-dir can replace the trio; presence is checked by
        # require_single_or_directory(), which can name the two forms.
        parser.add_argument('mic_wav', nargs='?')
        parser.add_argument('far_wav', nargs='?')
        parser.add_argument('out_wav', nargs='?')
        parser.add_argument('--device', default=None, help=DEVICE_HELP)
        parser.add_argument('--verify', action='store_true', help=VERIFY_HELP)
        add_directory_arguments(parser)
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

    def run_one(args):
        # The audio schedule lives in _streaming.py so the public CLI executes
        # the same hop-by-hop implementation the streaming tests drive.
        streaming = importlib.import_module(
            'AIAEC.%s._streaming' % model_name
        )
        streaming.main(args, load_model_fn=load_model)

    def main(args):
        run_pipeline(args, run_one)

    def cli():
        if len(sys.argv) > 1 and sys.argv[1] == 'calib':
            del sys.argv[1]
            from AIAEC._streaming_calibration import main as calibration_main
            calibration_main(model_name)
            return
        parser = build_parser()
        args = parser.parse_args()
        require_single_or_directory(parser, args)
        main(args)

    return build_parser, load_model, main, cli
