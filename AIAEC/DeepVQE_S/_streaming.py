#!/usr/bin/env python3
"""Internal DeepVQE-S frame-by-frame deployment reference.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify

This module binds inference.py's schedule (the shared hop loop in
AIAEC/_cli_common.py) to this candidate, using its checkpoint-recovered
grid and resampled mono inputs. STFT -> model -> ISTFT runs frame-by-frame:
audio is fed one hop at a time through StreamSTFT, every emitted frame goes
through DeepVQES.forward_stream with an explicit state dict, and StreamISTFT
reconstructs samples incrementally.

DeepVQE-S is end-to-end: unlike the RES+NR candidates there is no linear-AEC
(PBFDKF) frontend stage before the network.  For candidates that carry such a
frontend, streaming that filter is a separate C-side seam and would run
offline here regardless -- this CLI verifies the NN streaming only.

On startup the CLI prints the per-invocation I/O contract (one step's tensor
names, shapes, dtypes) and the persistent-state inventory.  ``--verify``
additionally runs the offline whole-wav forward and reports the max-abs / RMS
waveform difference against the streamed output.  It is DIAGNOSTICS ONLY: no
tolerance is applied and the exit code never changes.  The streaming
equivalence gate is
tests/test_streaming_deepvqe_s.py::test_stream_matches_offline, not this flag.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._cli_common import make_streaming_main


main = make_streaming_main('DeepVQE_S')
