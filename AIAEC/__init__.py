"""Neural acoustic echo-control model candidates.

All model front-ends use complex STFT tensors in ``[batch, time, frequency]``
layout.  The package deliberately keeps AEC-specific conditioning outside the
``AINR`` package.
"""

from .aiaec_common import AecOutput, SignalGrid

__all__ = ["AecOutput", "SignalGrid"]
