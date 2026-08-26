#!/usr/bin/env python3
"""Materialize one mic/far pair's ``linear_error`` with the current frontend.

DIAGNOSTIC ONLY. This exists to feed
``inference.py --input-is-linear-error``, whose point is to answer "what does
this checkpoint do on a signal the CURRENT frontend produces" without
retraining first. rematerialize_linear_aec.py is the corpus tool: it walks a
generated split directory and rewrites dataset chunks. Neither substitutes for
the other, and neither of them served a loose pair of WAVs.

The contract here is built from the runtime, not read from a checkpoint, which
is what makes this runnable when a checkpoint-carrying path is not: after a
frontend change a recorded ``aec_behavior_hash`` no longer describes the
installed engine, and every path that carries one is supposed to refuse. This
tool carries no checkpoint, so it has nothing to refuse.

⚠ The output must never be written back into a corpus. A corpus is materialized
as complete sequences and chunked afterwards; this pads one waveform to a hop
boundary and trims the result back, so its tail is not what the corpus path
would produce there. No contract comparison can tell the two apart -- the
contract is identical -- which is why this is a warning and not a check.

    python3 -m AIAEC.dataset_gen.materialize_pair mic.wav far.wav error.wav
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

if __package__ in (None, ""):  # direct-script invocation
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))

from AIAEC.dataset_gen.gen_aec_dataset import (  # noqa: E402
    WAV_ENCODINGS,
    verify_wav_io,
)
from AIAEC.dataset_gen.linear_aec import (  # noqa: E402
    FROZEN_FRAME_HOP_BY_SR,
    make_linear_aec_contract,
    materialize_linear_error,
)
from AIAEC.inference_common import load_mic_far  # noqa: E402


def materialize_pair(args: argparse.Namespace) -> None:
    outputs = [p for p in (args.out_wav, args.echo_estimate) if p]
    for path in outputs:
        if os.path.exists(path) and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path} (use --overwrite)")

    # Before the filter runs, not after: this proves the torchaudio build still
    # honours the requested precision, and PBFDKF over a long pair is not worth
    # paying to then discover the file cannot hold the result.
    verify_wav_io(os.path.dirname(os.path.abspath(args.out_wav)) or ".",
                  args.wav_encoding)

    contract = make_linear_aec_contract(
        args.sample_rate, filter_length=args.filter_length)
    # load_mic_far owns rate conversion, mono folding and the mic-owns-the-
    # timeline rule. Running PBFDKF at the capture rate and resampling only its
    # residual would not reproduce the training frontend -- see its docstring.
    mic, far, source_rates = load_mic_far(
        args.mic_wav, args.far_wav, contract.sample_rate)
    mic, far = mic.squeeze(0), far.squeeze(0)

    # The materializer takes complete sequences and refuses a partial hop. A
    # loose pair rarely lands on a hop boundary, so pad and trim back rather
    # than making the caller do it -- and report it, because that padded tail
    # is what makes this output unfit for a corpus.
    length = int(mic.shape[-1])
    padding = (-length) % contract.hop_size
    if padding:
        mic = torch.nn.functional.pad(mic, (0, padding))
        far = torch.nn.functional.pad(far, (0, padding))

    error, echo_estimate = materialize_linear_error(mic, far, contract)
    error, echo_estimate = error[:length], echo_estimate[:length]

    spec = WAV_ENCODINGS[args.wav_encoding]
    torchaudio.save(args.out_wav, error.unsqueeze(0),
                    contract.sample_rate, **spec)
    if args.echo_estimate:
        torchaudio.save(args.echo_estimate, echo_estimate.unsqueeze(0),
                        contract.sample_rate, **spec)

    rms = lambda x: float(np.sqrt(np.mean(np.square(x.numpy()))))  # noqa: E731
    print(f"frontend    : {contract.engine} {contract.preset} "
          f"@ {contract.sample_rate} Hz, frame {contract.frame_size}, "
          f"hop {contract.hop_size}, taps {contract.filter_length}")
    print(f"behavior    : {contract.aec_behavior_hash}")
    print(f"fingerprint : {contract.fingerprint()}")
    print(f"source rates: mic {source_rates[0]} Hz, far {source_rates[1]} Hz")
    if padding:
        print(f"padded      : {padding} sample(s) to the hop boundary, "
              f"trimmed back to {length}")
    for path in outputs:
        print(f"wrote       : {path} ({length} samples, {args.wav_encoding})")
    print(f"levels      : mic rms {rms(mic[:length]):.6f}, "
          f"error rms {rms(error):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mic_wav", help="Microphone capture")
    parser.add_argument(
        "far_wav",
        help="Far-end reference, synchronized with the microphone and "
             "starting at the same time. Pass the RAW reference: the model's "
             "own alignment consumes it, and the training contract pairs "
             "linear_error with raw far, not with an aligned copy",
    )
    parser.add_argument("out_wav", help="Where to write linear_error")
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        choices=tuple(sorted(FROZEN_FRAME_HOP_BY_SR)),
        help="Frontend grid rate (default 16000). Must equal the consuming "
             "checkpoint's rate; both inputs are resampled to it first. The "
             "frame/hop pair and the preset follow from it and are frozen, so "
             "neither is a knob here",
    )
    parser.add_argument(
        "--filter-length", type=int, default=None,
        help="Override the PBFDKF tap count the preset would select",
    )
    parser.add_argument(
        "--echo-estimate", default=None, metavar="PATH",
        help="Also write mic - error, the frontend's implied echo estimate",
    )
    parser.add_argument(
        "--wav-encoding", default="float32", choices=tuple(WAV_ENCODINGS),
        help="Output precision (default float32). int16 halves the file and "
             "is fine for listening, but it perturbs any arithmetic that "
             "recombines stems -- including mic - error -- by ~1e-4",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing outputs",
    )
    return parser


if __name__ == "__main__":
    materialize_pair(build_parser().parse_args())
