#!/usr/bin/env python3
"""Internal Align-CRUSE frame-by-frame deployment reference.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify

This module binds inference.py's execution schedule (the shared hop loop in
AIAEC/_cli_common.py) to this candidate: audio is consumed in hop-sized chunks,
converted with incremental StreamSTFT, run through
``AlignCRUSE.forward_stream`` one frame at a time (all time context lives in
the explicit state cells) and reconstructed with incremental StreamISTFT.
``stream_output_delay`` is 0. The candidate consumes raw unaligned mic/far
spectra with no linear-AEC (PBFDKF) frontend, so the entire model path streams
here.

On startup it prints the per-invocation I/O table and, after the first frame,
the state inventory (the deployment RAM/IO contract). ``--verify`` also runs
the offline whole-utterance forward and prints the max-abs / RMS waveform
difference (expected within ~1e-5 for utterances of at least max_delay_frames
hops; see forward_stream's short-utterance caveat) as DIAGNOSTICS ONLY: no
tolerance is applied and the exit code never changes. The streaming equivalence
gate is
tests/test_streaming_align_cruse.py::test_stream_matches_offline_on_long_utterance.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._cli_common import make_streaming_main


main = make_streaming_main('Align_CRUSE')
