#!/usr/bin/env python3
"""Align-CRUSE streaming inference on one mic/far-end pair.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --device cpu
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify

Calibration:
    python3 inference.py calib --checkpoint checkpoint.pth \\
        --primary-dir /path/to/mic --far-dir /path/to/far \\
        --frames 8192 --format bin --output calib/align_cruse

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before inference when needed; output stays at that model rate. Output is
the joint end-to-end AEC+RES+NR estimate of near_target --
denoised, dereverberated and echo-cancelled near speech (see ../README.md's
decision matrix). This candidate's earlier AEC-only, noise-preserving route
was retired.

config.ini is not read here: every shape-relevant setting (grid, model
kwargs) is recovered from the checkpoint's own contract (see train.py /
AIAEC/training_common.py's make_checkpoint_contract), so inference cannot
silently drift from what the weights were trained with.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC._cli_common import make_inference_cli


build_parser, load_model, main, cli = make_inference_cli(
    'Align_CRUSE', AlignCRUSE, __doc__
)


if __name__ == '__main__':
    cli()
