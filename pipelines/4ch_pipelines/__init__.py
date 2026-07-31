"""Four-microphone linear-AEC integration path.

The hard resource boundary is part of the public contract:

* exactly one shared matched-filter delay estimator;
* exactly four linear adaptive filters;
* beamform to mono before the existing NR + RES post path.
"""

from importlib import import_module

# ``4ch_pipelines`` is the project directory name requested for the C/Python
# integration surface.  A leading digit is accepted by importlib and
# ``python -m`` but cannot appear in a normal Python ``from`` statement.
# Resolve the implementation by string so this file also remains importable
# when pytest collects it as a path-level ``__init__`` module.
_pipeline = import_module("pipelines.4ch_pipelines.pipeline")
Beamformer = _pipeline.Beamformer
BeamformerFrame = _pipeline.BeamformerFrame
EqualWeightBeamformer = _pipeline.EqualWeightBeamformer
FourChannelAecConfig = _pipeline.FourChannelAecConfig
FourChannelAecPipeline = _pipeline.FourChannelAecPipeline
FourChannelFrame = _pipeline.FourChannelFrame
PreBeamformerFrame = _pipeline.PreBeamformerFrame
SharedDelayState = _pipeline.SharedDelayState
SharedMatchedDelayEstimator = _pipeline.SharedMatchedDelayEstimator

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
