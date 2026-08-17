#!/usr/bin/env python3
"""Export DeepVQE-S as a stateless one-frame ONNX graph.

    python3 export_onnx.py --checkpoint checkpoint.pth \
        --output output/deepvqe_s.onnx --verify
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC._streaming_export import main

if __name__ == '__main__':
    main('DeepVQE_S')
