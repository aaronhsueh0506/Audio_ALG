"""Expose the import roots AINR tests share with the CLIs.

``calibration_io`` and ``onnx_streaming_contract`` live in ``AINR/`` so an
AINR release does not depend on its parent directory.  Trainers/tests import
AINR's own ``dataset_gen`` bare; all three resolve by putting ``AINR`` on the
path, exactly as every entry script's own shim does. The ``Audio_ALG`` root is
added only for cross-package ``AIAEC.*`` imports used by integration tests.
"""

import os
import sys

_AINR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG = os.path.dirname(_AINR)

if _AUDIO_ALG not in sys.path:
    sys.path.insert(0, _AUDIO_ALG)
if _AINR not in sys.path:
    sys.path.insert(0, _AINR)
