"""Expose package-qualified AIAEC and pure-AINR imports during tests.

AEC data generation reuses generic augmentation DSP from ``AINR``.

The ``Audio_ALG`` root serves the package-qualified imports; the ``AINR``
directory is added as well because ``calibration_io`` lives there (so a
deployment that copies only the model folders still carries it). ``AIAEC``
itself is deliberately NOT added -- that would shadow AINR's bare
``dataset_gen`` with a second top-level package of the same name.
"""

import os
import sys

_AIAEC = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG = os.path.dirname(_AIAEC)

if _AUDIO_ALG not in sys.path:
    sys.path.insert(0, _AUDIO_ALG)
_AINR = os.path.join(_AUDIO_ALG, 'AINR')
if _AINR not in sys.path:
    sys.path.insert(0, _AINR)
