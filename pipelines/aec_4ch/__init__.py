"""Four-microphone linear-AEC integration path.

The hard resource boundary is part of the public contract:

* exactly one shared matched-filter delay estimator;
* exactly four linear adaptive filters;
* beamform to mono before the existing NR + RES post path.
"""

from .pipeline import (
    Beamformer,
    BeamformerFrame,
    EqualWeightBeamformer,
    FourChannelAecConfig,
    FourChannelAecPipeline,
    FourChannelFrame,
    PreBeamformerFrame,
    SharedDelayState,
    SharedMatchedDelayEstimator,
)

__all__ = [
    "Beamformer",
    "BeamformerFrame",
    "EqualWeightBeamformer",
    "FourChannelAecConfig",
    "FourChannelAecPipeline",
    "FourChannelFrame",
    "PreBeamformerFrame",
    "SharedDelayState",
    "SharedMatchedDelayEstimator",
]
