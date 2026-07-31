"""Expose package-qualified AIAEC and pure-AINR imports during tests.

AEC data generation reuses generic augmentation DSP from ``AINR``. The
GTCRN-AENR and DeepFilterNet-AENR project variants also import their audited
single-input bases from there; all AEC conditioning stays in AIAEC.

Only the ``Audio_ALG`` root is added. Adding ``AIAEC`` and ``AINR`` themselves
would create two ambiguous top-level packages both named ``dataset_gen``.
"""

import os
import sys

_AIAEC = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG = os.path.dirname(_AIAEC)

if _AUDIO_ALG not in sys.path:
    sys.path.insert(0, _AUDIO_ALG)
