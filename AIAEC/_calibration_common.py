"""Shared WAV-pair loading for model-local AIAEC calibration commands.

``--primary-dir`` means microphone for the end-to-end models and materialized
linear-AEC error for RES+NR models.  ``--far-dir`` must have identical relative
WAV paths.  Inputs are resampled and transformed with the same project helpers
as inference; no random tensors or zero-only recurrent history are used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.dataset_gen import stft
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC.training_common import (
    CALIBRATION_ONLY_FAR_INPUT_MODE,
    DEPLOYED_FAR_INPUT_MODE,
)


LINEAR_ERROR_MODELS = {'Align_ULCNet'}


def far_mode_provenance(model_name):
    """Describe calibration data separately from the deployment seam.

    Lives here rather than in the streaming recorder because both recorders
    read the same far directory through ``blocks_from_pair`` and so must
    describe it the same way. Returns ``(calibration, deployment)``.
    """
    if model_name == 'Align_ULCNet':
        # The far WAVs are the raw rendered reference; deployment feeds the
        # aligned far the linear AEC produced. Recording only one of the two
        # would let a consumer assume the ranges were measured on the signal
        # the board will actually present.
        return 'raw_far', DEPLOYED_FAR_INPUT_MODE
    return CALIBRATION_ONLY_FAR_INPUT_MODE, CALIBRATION_ONLY_FAR_INPUT_MODE


def discover_pairs(primary_dir, far_dir):
    def inventory(root):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError('directory does not exist: %s' % root)
        result = {path.relative_to(root).as_posix(): path
                  for path in sorted(root.rglob('*'))
                  if path.is_file() and path.suffix.lower() == '.wav'}
        if not result:
            raise ValueError('no WAV files under %s' % root)
        return result
    primary = inventory(primary_dir)
    far = inventory(far_dir)
    if set(primary) != set(far):
        raise ValueError('primary/far relative WAV sets differ: missing=%d extra=%d' %
                         (len(set(primary) - set(far)),
                          len(set(far) - set(primary))))
    return [(name, primary[name], far[name]) for name in sorted(primary)]


def _ri(spec):
    return torch.view_as_real(spec.transpose(-2, -1)).contiguous()


def blocks_from_pair(model_name, model, grid, primary_path, far_path,
                     frames, feature_config=None):
    loader = (load_linear_error_far if model_name in LINEAR_ERROR_MODELS
              else load_mic_far)
    primary, far, _ = loader(str(primary_path), str(far_path), grid.sr)
    tensors = {
        ('error' if model_name in LINEAR_ERROR_MODELS
         else 'mic'): _ri(stft(primary, grid)),
        'far': _ri(stft(far, grid)),
    }
    time_axis = {name: 1 for name in tensors}
    total = tensors[next(iter(tensors))].shape[time_axis[next(iter(tensors))]]
    for start in range(0, total - frames + 1, frames):
        block = {}
        for name, tensor in tensors.items():
            axis = time_axis[name]
            slices = [slice(None)] * tensor.ndim
            slices[axis] = slice(start, start + frames)
            block[name] = tensor[tuple(slices)][0].cpu().numpy().astype(
                np.float32, copy=False)
        yield block

