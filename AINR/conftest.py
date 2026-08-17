"""Expose root-level ``Audio_ALG`` imports during AINR tests.

Every model's calibration entry point shares ``calibration_io`` from the
repository root, so the tests must resolve it the same way the CLIs do.

Only the ``Audio_ALG`` root is added. Adding ``AINR`` itself would create two
ambiguous top-level packages both named ``dataset_gen``.
"""

import os
import sys

_AINR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG = os.path.dirname(_AINR)

if _AUDIO_ALG not in sys.path:
    sys.path.insert(0, _AUDIO_ALG)
