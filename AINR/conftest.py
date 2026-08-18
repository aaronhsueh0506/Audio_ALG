"""Expose the import roots AINR tests share with the CLIs.

``calibration_io`` lives in ``AINR/`` (so a deployment that copies only the
model folders still carries it), and the trainers/tests import AINR's own
``dataset_gen`` bare -- both resolve by putting the ``AINR`` directory on the
path, exactly as every entry script's own shim does. The ``Audio_ALG`` root
is added for the cross-package ``AIAEC.*`` imports.
"""

import os
import sys

_AINR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG = os.path.dirname(_AINR)

if _AUDIO_ALG not in sys.path:
    sys.path.insert(0, _AUDIO_ALG)
if _AINR not in sys.path:
    sys.path.insert(0, _AINR)
