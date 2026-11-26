#!/usr/bin/env python3
"""Cross-layer equalization of a DeepVQE-S checkpoint before PTQ.

    python3 equalize.py output_16k_512/deepvqe_s_best.pth \
        -o output_16k_512/deepvqe_s_best_cle.pth

See README.md, "Cross-layer equalization".
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._equalize import main

if __name__ == '__main__':
    main('DeepVQE_S')
