#!/usr/bin/env python3
"""Internal DeepVQE-S frame-by-frame deployment reference.

Binds inference.py's schedule (the shared hop loop in AIAEC/_cli_common.py) to
this candidate: audio is fed one hop at a time through StreamSTFT, every emitted
frame goes through DeepVQES.forward_stream with an explicit state dict, and
StreamISTFT reconstructs samples incrementally. There is no linear-AEC frontend
before the network.

On startup it prints the per-invocation I/O contract (tensor names, shapes,
dtypes) and the persistent-state inventory. ``--verify`` also runs the offline
whole-wav forward and reports the max-abs / RMS waveform difference as
diagnostics only (no tolerance; the exit code never changes).
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._cli_common import make_streaming_main


main = make_streaming_main('DeepVQE_S')
