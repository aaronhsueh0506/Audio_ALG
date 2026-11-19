"""The constructor keyword arguments of a model's shipped per-grid config.

DeepVQE-S and Align-ULCNet each ship one self-contained training config per
board grid (``AIAEC/<model>/config_16k_256.ini``, ``config_16k_512.ini``,
``config_48k.ini``); every row of stream/aiaec/graph_io_{deepvqes,alignulcnet}.h
follows the config of its grid. Tests that export a model build it from these
kwargs, read through the trainer's own path, so they follow the config.
Align_CRUSE and CAGCRN keep a single ``config.ini``.
"""

import contextlib
import io
from pathlib import Path

from AIAEC.training_common import (
    contract_model_kwargs,
    read_config,
    read_model_kwargs,
)

AIAEC_DIR = Path(__file__).resolve().parents[1]
GRID_CONFIGS = {
    (16000, 256): 'config_16k_256.ini',
    (16000, 512): 'config_16k_512.ini',
    (48000, 1024): 'config_48k.ini',
}
SHIPPED_GRID = (16000, 256)
PER_GRID_MODELS = ('DeepVQE_S', 'Align_ULCNet')


def config_path(model_name, grid=SHIPPED_GRID):
    """The shipped config of ``model_name`` for ``grid`` (a model with one
    ``config.ini`` returns it for any grid)."""
    if model_name not in PER_GRID_MODELS:
        return AIAEC_DIR / model_name / 'config.ini'
    return AIAEC_DIR / model_name / GRID_CONFIGS[tuple(grid)]


def config_paths(model_name):
    """Every shipped training config of ``model_name``."""
    if model_name not in PER_GRID_MODELS:
        return [AIAEC_DIR / model_name / 'config.ini']
    return [AIAEC_DIR / model_name / name for name in GRID_CONFIGS.values()]


def shipped_model_kwargs(model_name, model_cls, grid=SHIPPED_GRID):
    """``model_cls`` kwargs from the ``[model]`` of the shipped config of
    ``grid`` (read_config refuses a missing file); the file's ``[signal]``
    must be that grid."""
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = read_config([str(config_path(model_name, grid))])
    assert (cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft')) == tuple(grid)
    return read_model_kwargs(cfg, model_cls)


def ctor_contract(model_name, model_cls, grid=SHIPPED_GRID):
    """The ``ctor_*`` checkpoint-contract keys the trainer writes for the
    shipped kwargs, so ``inference.load_model`` rebuilds the same model."""
    kwargs = contract_model_kwargs(
        model_cls, shipped_model_kwargs(model_name, model_cls, grid))
    return {'ctor_' + name: value for name, value in kwargs.items()}
