#!/usr/bin/env python3
"""Internal CAGCRN frame-by-frame deployment reference.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify

This module binds inference.py's complete STFT -> model -> ISTFT chain (the
shared hop loop in AIAEC/_cli_common.py) to this candidate, one hop at a
time through StreamSTFT / forward_stream / StreamISTFT. On startup it prints
the per-invocation I/O table and, after the first frame, the state inventory --
together these are the deployment contract an NPU/C port must reproduce.

CAGCRN is an end-to-end candidate: inference.py feeds the microphone and
far-end reference straight into the network, so there is no linear-AEC
(PBFDKF) frontend to run here.  For candidates that do consume a PBFDKF
error signal, streaming that frontend is a separate C-side seam -- this CLI
verifies the NN streaming.

``--verify`` additionally runs the offline whole-wav forward and prints the
max-abs / RMS difference between the two output waveforms.  It is DIAGNOSTICS
ONLY: no tolerance is applied and the exit code never changes.  The streaming
equivalence gate is
tests/test_streaming_cagcrn.py::test_stream_matches_offline, not this flag.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._cli_common import make_streaming_main


main = make_streaming_main('CAGCRN')
