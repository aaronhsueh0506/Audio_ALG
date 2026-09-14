"""
4-channel AEC + beamformer + RES + DeepFilterNet2 (Python reference)

The four-channel front end (shared delay, four linear AEC lanes, the fused
post-beam residual gain) is the existing ``4ch_aec_bf_nr_res/pipeline.py``
reference, unchanged.  This module adds the post half that has no Python twin
today -- the spectrum-to-hop tail after ``res_gain`` -- in two flavours:

* ``post_conventional``: MMSE-LSA on the fused pre-RES spectrum,
  ``min(G_nr, G_res)``, comfort noise, WOLA -- the C core's
  ``run_post_res_and_nr`` (``4aec_nr_res.c``), giving evaluation variant (a)
  a 4-channel Python counterpart;
* ``post_dfn2``: ``P = E * G_res`` plus comfort noise (the conventional
  post spectrum with the denoiser off), the dual-input DFN2 stage estimated
  on ``E`` (the beamformed spectrum) and applied on ``P``, WOLA -- the
  ``4ch_aec_bf_dfn_res`` C wrapper.

Both consume the same fused ``AecResContext`` stream, so the two tails differ
only in the stage under test.  The C wrapper hands the stage the GSC output
directly (``trusted_beamformed_error``); here ``E`` is the coherent projection
``sum_ch w_ch * error_spec_ch`` the Python reference builds from the adapter's
weights, which is the same signal when the adapter's weights are the ones the
beamformer actually applied.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_AEC_PY = os.path.join(_ROOT, 'lib', 'aec', 'python')
if _AEC_PY not in sys.path:
    sys.path.insert(0, _AEC_PY)

from lib.aec.python.aec import AecConfig, AecMode, AecResContext  # noqa: E402
from pipelines.aec_dfn2_res_pipeline import run_dfn2_res  # noqa: E402
from pipelines.aec_nr_pipeline import run_nr_spectrum, run_res  # noqa: E402
from pipelines.dfn2_stage import Dfn2Stage  # noqa: E402


def _load_four_channel_reference():
    """``pipelines/4ch_aec_bf_nr_res/pipeline.py`` (a directory name that is
    not an importable package name), loaded by path."""
    path = os.path.join(_ROOT, 'pipelines', '4ch_aec_bf_nr_res', 'pipeline.py')
    name = 'four_channel_aec_reference'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


four_channel = _load_four_channel_reference()
FourChannelAecConfig = four_channel.FourChannelAecConfig
FourChannelAecPipeline = four_channel.FourChannelAecPipeline
EqualWeightBeamformer = four_channel.EqualWeightBeamformer


def run_front_end(microphones: np.ndarray, render: np.ndarray,
                  config: Optional[FourChannelAecConfig] = None,
                  beamformer=None) -> List[AecResContext]:
    """Shared delay + four lanes + beamformer adapter + fused post-beam RES.

    Returns the fused per-hop contexts (``error_spec`` = the beamformed
    pre-RES spectrum, ``res_gain`` = the one post-beam suppression gain,
    ``comfort_noise``/``r2`` fused), which both tails consume.
    """
    pipeline = FourChannelAecPipeline(config, beamformer or EqualWeightBeamformer())
    hop = pipeline.hop_size
    n = min(len(render), microphones.shape[0])
    contexts: List[AecResContext] = []
    for start in range(0, n - hop + 1, hop):
        frame = pipeline.process(microphones[start:start + hop], render[start:start + hop])
        contexts.append(frame.context)
    return contexts


def _post_config(sample_rate: int, hop: int, enable_cng: bool) -> AecConfig:
    return AecConfig(sample_rate=sample_rate, frame_size=2 * hop, hop_size=hop,
                     mode=AecMode.PBFDKF, enable_cng=enable_cng)


def post_conventional(contexts: Sequence[AecResContext], sample_rate: int,
                      nr_preset: str = 'balanced', enable_cng: bool = True,
                      inject_echo_psd: bool = True) -> np.ndarray:
    """MMSE-LSA on E, min(G_nr, G_res), comfort noise, WOLA."""
    hop = int(contexts[0].error_spec.size - 1)
    nr_gains = run_nr_spectrum(contexts, sample_rate, nr_preset=nr_preset,
                               inject_echo_psd=inject_echo_psd)
    tail = np.zeros(len(contexts) * hop, np.float32)
    return run_res(tail, nr_gains, contexts, _post_config(sample_rate, hop, enable_cng),
                   use_nr=True, use_res=True, combine='min')


def post_dfn2(contexts: Sequence[AecResContext], sample_rate: int, stage: Dfn2Stage,
              estimate_source: str = 'pre_res',
              enable_cng: bool = False, realtime_timing: bool = False) -> np.ndarray:
    """P = E * G_res (+ comfort noise when enabled), DFN2 estimated on E and
    applied on P, WOLA. Comfort noise is off by default, as in the C wrapper."""
    hop = int(contexts[0].error_spec.size - 1)
    return run_dfn2_res(contexts, _post_config(sample_rate, hop, enable_cng), stage,
                        estimate_source=estimate_source,
                        enable_cng=enable_cng, realtime_timing=realtime_timing,
                        output_length=len(contexts) * hop)
