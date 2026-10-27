"""Four-microphone linear-AEC integration path.

The hard resource boundary is part of the public contract:

* exactly one shared matched-filter delay estimator;
* exactly four linear adaptive filters;
* beamform to mono before the NR + RES post path.
"""

from importlib import import_module

# ``4ch_aec_bf_nr_res`` starts with a digit: importlib and ``python -m`` accept
# it but a ``from`` statement cannot, so the implementation is resolved by
# string (which also keeps this file importable when pytest collects it as a
# path-level ``__init__`` module).
_pipeline = import_module("pipelines.4ch_aec_bf_nr_res.pipeline")
Beamformer = _pipeline.Beamformer
McabBeam = _pipeline.McabBeam
EqualWeightBeamformer = _pipeline.EqualWeightBeamformer
McabConfig = _pipeline.McabConfig
McabPipeline = _pipeline.McabPipeline
McabFrame = _pipeline.McabFrame
McabFront = _pipeline.McabFront
PostBeamResidualSuppressor = _pipeline.PostBeamResidualSuppressor
McabDelayState = _pipeline.McabDelayState
SharedMatchedDelayEstimator = _pipeline.SharedMatchedDelayEstimator

__all__ = [
    "Beamformer",
    "McabBeam",
    "EqualWeightBeamformer",
    "McabConfig",
    "McabPipeline",
    "McabFrame",
    "McabFront",
    "PostBeamResidualSuppressor",
    "McabDelayState",
    "SharedMatchedDelayEstimator",
]
