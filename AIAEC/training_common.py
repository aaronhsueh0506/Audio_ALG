"""Shared training infrastructure for every AIAEC candidate trainer.

Four model directories -- Align_CRUSE, Align_ULCNet, DeepVQE_S and CAGCRN --
each own a ``train.py`` / ``config.ini`` / ``inference.py`` (see any
of their top-of-file docstrings for the exact usage). What they do NOT each
keep a private copy of lives here, for the same reason
``AIAEC/dataset_gen/aec_features.py`` holds the signal grid and
``AINR/dataset_gen`` holds the seeder: a knob that drifts between four
near-identical copies is a checkpoint- or corpus-comparability bug that
surfaces only as "one model is mysteriously worse."

Provided here:

* ``set_seed``                                -- re-exported from AINR.dataset_gen
* ``read_model_kwargs``                       -- ``[model]`` config -> constructor
  kwargs, generalising AINR/DeepFilterNet2's ``read_model_config`` (there:
  one model class; here: four)
* ``make_checkpoint_contract`` /
  ``require_checkpoint_contract``             -- reject a checkpoint whose grid,
  task, model kwargs or loss version differ from the running config
* ``scan_non_finite`` / ``NonFiniteTraining`` / ``GradNormLog`` /
  ``halt_on_non_finite`` / ``WeightScaleGuard`` / ``VanishedWeights``
                                              -- the halt machinery is
  re-exported from ``AINR.training_common``, not reimplemented a second time.
  ``WeightScaleGuard`` covers the failure ``scan_non_finite`` structurally
  cannot: weights that decay to denormal magnitudes stay FINITE
* ``make_optimizer``                          -- fallback optimizer selector for
  candidates whose paper does not prescribe a different recipe
* ``make_scheduler`` / ``fast_forward_scheduler`` -- the legacy project
  warmup/cosine recipe, retained for old checkpoints rather than imposed on
  models whose papers prescribe another schedule
* ``compressed_spectral_loss``                -- project fallback for candidates
  whose paper does not publish a training objective
* ``component_compressed_mse_loss``           -- Align-ULCNet/ULCNet paper
  objective: component-wise signed power compression followed by complex MSE
* ``power_compressed_complex_mse_loss``       -- Align-CRUSE PLCPA objective
* ``si_snr_loss``                             -- waveform SI-SNR term used by
  CAGCRN
* ``stft_consistent_spectrum`` / ``spectrum_to_waveform`` -- differentiable
  shared synthesis boundaries for objectives defined after iSTFT
* ``LinearAecEngine``                         -- inference-only continuous-file
  wrapper around the same frozen Python PBFDKF whose output is materialized as
  dataset stem five (`linear_error`). Trainers never execute it.
* ``split_dataset_by_sample`` / ``build_plain_loaders`` -- deterministic
  per-chunk train/val split plus epoch-level shuffle for the unified corpus.

Do not add a fifth copy of generic infrastructure into a candidate's
``train.py``. A paper-specific optimizer or scheduler remains in that
candidate, is recorded in its checkpoint contract, and must not be presented
as a cross-model comparison rule. The same reasoning
``AIAEC/dataset_gen/README.md`` gives for ``aec_features.py``.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(_THIS_DIR)
_LIB_AEC_PYTHON = os.path.join(_AUDIO_ALG_ROOT, 'lib', 'aec', 'python')
for _path in (_AUDIO_ALG_ROOT, _LIB_AEC_PYTHON):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from AINR.dataset_gen import set_seed, subsets_from_indices  # noqa: E402
# The NaN-halt machinery and the LR schedule factory are AINR's shared trainer
# module, not any one trainer's -- importing a MODEL to borrow infrastructure
# meant a DeepFilterNet2 edit could break every AIAEC trainer.
from AINR.training_common import (  # noqa: E402
    GradNormLog,
    NonFiniteTraining,
    VanishedWeights,
    WeightScaleGuard,
    fast_forward_scheduler,
    halt_on_non_finite as _halt_on_non_finite,
    make_scheduler,
    scan_non_finite,
)

# lib/aec/python/aec.py re-exports the engine; its own modules use bare
# `from modules.xxx import ...`, which resolves only with lib/aec/python
# itself (not lib/aec) on sys.path -- the same setup run_one_case.py and
# eval_aec_challenge.py use.
from aec import AEC  # noqa: E402

from AIAEC.aiaec_common import SignalGrid, safe_abs  # noqa: E402
from AIAEC.dataset_gen import (  # noqa: E402
    AecGrid,
    MODEL_TASKS,
    PackedAecDataset,
    aec_collate,
    istft,
    stft,
)
from AIAEC.dataset_gen.linear_aec import (  # noqa: E402
    DATASET_DELAY_NUM_FILTERS,
    LinearAecContract,
    check_delay_num_filters,
    engine_delay_num_filters,
    make_linear_aec_config,
    make_linear_aec_contract,
    require_linear_aec_contract,
)


__all__ = [
    'set_seed',
    'build_arg_parser',
    'auto_device',
    'training_progress',
    'read_grids',
    'split_dataset_by_sample',
    'build_plain_loaders',
    'read_model_kwargs',
    'FAR_INPUT_MODES',
    'FAR_INPUT_MODE_C_VALUES',
    'DEPLOYED_FAR_INPUT_MODE',
    'CALIBRATION_ONLY_FAR_INPUT_MODE',
    'far_input_mode_c_value',
    'checkpoint_far_input_mode',
    'make_checkpoint_contract',
    'require_checkpoint_contract',
    'require_checkpoint_model_identity',
    'require_checkpoint_linear_aec',
    'scan_non_finite',
    'NonFiniteTraining',
    'GradNormLog',
    'halt_on_non_finite',
    'VanishedWeights',
    'WeightScaleGuard',
    'DEFAULT_OPTIMIZER',
    'make_optimizer',
    'compressed_spectral_loss',
    'component_compressed_mse_loss',
    'power_compressed_complex_mse_loss',
    'si_snr_loss',
    'stft_consistent_spectrum',
    'spectrum_to_waveform',
    'LinearAecEngine',
    'make_scheduler',
    'fast_forward_scheduler',
]


# ============================================================
# CLI
# ============================================================

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """The common flags every AIAEC train.py accepts, so they stop drifting.

    A candidate whose training genuinely needs an extra flag adds it to the
    parser this returns -- see any train.py's ``build_parser()`` for the
    pattern -- rather than re-declaring these five.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument(
        '--packed-dir', default=None,
        help='Packed shard file/directory; overrides [data] packed_dir',
    )
    parser.add_argument(
        '--mmap', action='store_true',
        help='Memory-map packed tensors instead of loading them fully into RAM',
    )
    parser.add_argument(
        '--gpu', type=int, default=None,
        help='CUDA GPU index (for example --gpu 0); takes precedence over --device',
    )
    parser.add_argument('--device', default=None,
                        help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint path to resume training from')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='With --resume, load model weights only and build '
                             'a fresh optimizer/epoch counter (e.g. to change '
                             'the LR schedule mid-run without diagnostic-resume '
                             'semantics)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser


def auto_device(requested: Optional[str], gpu: Optional[int] = None) -> str:
    if gpu is not None:
        if gpu < 0:
            raise ValueError(f"--gpu must be non-negative, got {gpu}")
        return f'cuda:{gpu}'
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def training_progress(loader, *, training: bool, epoch: int,
                      max_epochs: Optional[int] = None):
    """Return AINR-style tqdm progress for training and a plain val loader.

    AINR shows one ``Epoch current/total`` bar for the training loader and
    leaves validation quiet. Keeping that policy here avoids four subtly
    different progress implementations while preserving normal DataLoader
    behaviour for evaluation.
    """
    if not training:
        return loader
    current = int(epoch) + 1  # AIAEC checkpoints/loops store zero-based epochs.
    desc = (f"Epoch {current}/{int(max_epochs)}"
            if max_epochs is not None else f"Epoch {current}")
    return tqdm.tqdm(loader, desc=desc)


# ============================================================
# Signal grid
# ============================================================

def read_grids(cfg, section: str = 'signal') -> Tuple[AecGrid, 'SignalGrid']:
    """One ``[signal]`` section -> both grid types the two layers need.

    ``AecGrid`` (``dataset_gen/aec_features.py``, field ``sr``) drives the
    dataset/STFT boundary; ``SignalGrid`` (``aiaec_common.py``, field
    ``sample_rate``) is what every model constructor takes. They describe the
    identical grid under two names because they were written for two
    different layers; building both from one config read here means they
    cannot drift apart the way two independent ``cfg.getint`` call sites
    could.
    """
    aec_grid = AecGrid.from_config(cfg, section=section)
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    return aec_grid, model_grid


# ============================================================
# Train/val split at load time (paired with --split all / build_unified_manifest)
# ============================================================

def split_dataset_by_sample(dataset, val_fraction: float,
                            seed: int = 42) -> Tuple[List[int], List[int]]:
    """Deterministically random-split individual materialized chunks.

    The frozen PBFDKF has already run over each complete parent sequence before
    the `linear_error` stem was cut into chunks. Consequently each stored item is now a fixed
    model sample: the trainer may place any chunk on either side and shuffle
    train order without changing adaptive-filter state. Sharing a sequence,
    speaker or RIR across train/validation is an explicit in-distribution
    validation choice, matching the standalone NR loaders.

    Returned indices are sorted only for shard/mmap locality. The assignment is
    drawn from a dedicated RNG and depends exclusively on dataset length and
    ``seed``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    size = len(dataset)
    if val_fraction > 0.0 and size < 2:
        raise ValueError(
            f"only {size} sample(s) in this dataset; a held-out split needs "
            "at least 2 so both sides are non-empty"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    n_val = max(1, int(round(size * val_fraction))) if val_fraction > 0.0 else 0
    if n_val >= size:
        raise ValueError(
            f"val split ({n_val}) would consume the whole dataset ({size} samples)"
        )
    return sorted(order[n_val:]), sorted(order[:n_val])


def build_plain_loaders(cfg, aec_grid, seed: int = 42,
                        section: str = 'data', *,
                        packed_dir: Optional[str] = None,
                        mmap: bool = False) -> Tuple[
                            DataLoader, Optional[DataLoader], Dict]:
    """Build the shared per-chunk split/loaders and its checkpoint contract.

    Every candidate uses this exact path. Train samples reshuffle every epoch;
    validation order stays stable. The returned JSON-serializable data
    contract records the exact corpus and split used for resume checks.
    """
    resolved_packed_dir = packed_dir or cfg.get(
        section, 'packed_dir', fallback=None
    )
    if not resolved_packed_dir:
        raise ValueError("--packed-dir or [data] packed_dir required")
    dataset = PackedAecDataset(
        resolved_packed_dir, expected_sr=aec_grid.sr, mmap=mmap
    )
    val_fraction = cfg.getfloat(section, 'val_fraction', fallback=0.1)
    train_indices, val_indices = split_dataset_by_sample(
        dataset, val_fraction, seed=seed
    )
    train_subset, val_subset = subsets_from_indices(dataset, train_indices, val_indices)

    batch_size = cfg.getint(section, 'batch_size')
    num_workers = cfg.getint(section, 'num_workers', fallback=0)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=aec_collate, drop_last=False,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=aec_collate, drop_last=False,
        )
    data_contract = {
        'dataset_fingerprint': dataset.fingerprint(),
        'linear_aec': dataset.linear_aec_contract.as_dict(),
        'linear_aec_contract_hash': dataset.linear_aec_contract_hash,
        'split_kind': 'random_chunk',
        'split_seed': int(seed),
        'val_fraction': float(val_fraction),
        'train_indices': train_indices,
        'val_indices': val_indices,
    }
    return train_loader, val_loader, data_contract


# ============================================================
# Config -> constructor kwargs
# ============================================================

def read_model_kwargs(cfg, model_cls, section: str = 'model',
                      aliases: Optional[Dict[str, str]] = None) -> Dict:
    """Every ``model_cls.__init__`` keyword argument, overlaid with ``[model]``.

    ``grid`` (and ``self``) are excluded: the signal grid comes from
    ``[signal]`` via ``AecGrid.from_config``, not from a second copy in
    ``[model]``. Unknown ``[model]`` keys raise, and the parsed type follows
    the constructor's own default -- exactly
    ``AINR/DeepFilterNet2/train.py``'s ``read_model_config``, generalised from
    one model class to whichever ``model_cls`` a candidate's train.py passes.

    """
    aliases = aliases or {}
    sig = inspect.signature(model_cls.__init__)
    kwargs = {
        name: param.default for name, param in sig.parameters.items()
        if name not in ('self', 'grid') and param.default is not inspect.Parameter.empty
    }
    if not cfg.has_section(section):
        return kwargs
    for name in cfg.options(section):
        kwarg = aliases.get(name, name)
        if kwarg not in kwargs:
            raise ValueError(
                f"[{section}] {name!r} is not a {model_cls.__name__} constructor "
                f"argument (known: {', '.join(sorted(kwargs))})")
        default = kwargs[kwarg]
        raw = cfg.get(section, name)
        if isinstance(default, bool):          # before int -- bool subclasses int
            kwargs[kwarg] = cfg.getboolean(section, name)
        elif isinstance(default, int):
            kwargs[kwarg] = int(raw)
        elif isinstance(default, float):
            kwargs[kwarg] = float(raw)
        elif isinstance(default, tuple):
            kwargs[kwarg] = tuple(int(v) for v in raw.split(','))
        elif default is None:
            # Optional[int] constructor args (e.g. df_bins, max_delay_frames):
            # 'none'/'' keeps the model's own derived default.
            kwargs[kwarg] = None if raw.strip().lower() in ('none', '') else int(raw)
        else:
            kwargs[kwarg] = raw
    return kwargs


# ============================================================
# Checkpoint contract
# ============================================================

# The far-end input every current trainer feeds its model: the RAW rendered
# far-end reference (``stems.far_render`` in dataset_gen/model_views.py, both
# task families), with no alignment or preprocessing. Recorded in the
# checkpoint contract so a checkpoint trained on a future preprocessed far
# variant cannot be loaded into a seam that feeds raw far.
FAR_INPUT_MODES = ('raw_far',)

# The C-side UlcnetFarInputMode enumeration (AIAEC/Align_ULCNet/
# ulcnet_model_io.h), which is what a board compares against: the exporter
# writes the numeric value beside the string so no integrator has to keep a
# second copy of this table. It deliberately spans MORE modes than
# FAR_INPUT_MODES above: FAR_INPUT_MODES is what this training seam can
# produce today, while this is the full C ABI enumeration, so an
# 'aligned_far' checkpoint would still map to a defined descriptor value
# rather than to an unnamed integer.
FAR_INPUT_MODE_C_VALUES = {'raw_far': 0, 'aligned_far': 1}

# The far-end signal the production seam presents to the model, fixed for
# every deployed alignment candidate: the far the linear AEC has already
# aligned. Deliberately NOT derived from FAR_INPUT_MODES -- that names what
# training feeds, and the whole point of recording both is that they differ.
# Raw/aligned comparison survives only in sweep_delay_depth.py.
DEPLOYED_FAR_INPUT_MODE = 'aligned_far'

# Calibration-seam-only name, used when a model has no separate far seam at
# all: the recorder fed it exactly the far its own graph consumes, so there is
# nothing to distinguish. It is deliberately ABSENT from
# FAR_INPUT_MODE_C_VALUES -- no C enumerator exists for it and none should,
# because it never describes a deployed wiring; far_input_mode_c_value()
# rejecting it is the intended behaviour, not an oversight.
CALIBRATION_ONLY_FAR_INPUT_MODE = 'model_native_far'


def far_input_mode_c_value(mode: str) -> int:
    """Map a far-input mode name to its C ``UlcnetFarInputMode`` value."""
    if mode not in FAR_INPUT_MODE_C_VALUES:
        raise ValueError(
            f"far_input_mode={mode!r} has no C enum value "
            f"(known: {', '.join(sorted(FAR_INPUT_MODE_C_VALUES))})"
        )
    return FAR_INPUT_MODE_C_VALUES[mode]


def checkpoint_far_input_mode(contract: Dict) -> str:
    """Return the contract's far-input mode; the ONE place the default lives.

    Checkpoints written before ``far_input_mode`` existed all trained on raw
    far by construction (see ``FAR_INPUT_MODES``), so a missing field reads as
    ``'raw_far'`` and legacy checkpoints stay loadable. A recorded mode this
    code does not know is rejected: it means the checkpoint's far
    preprocessing has no implementation at this seam.
    """
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no valid contract")
    mode = contract.get('far_input_mode', 'raw_far')
    if mode not in FAR_INPUT_MODES:
        raise ValueError(
            f"checkpoint far_input_mode={mode!r} is not supported here "
            f"(known: {', '.join(FAR_INPUT_MODES)})"
        )
    return mode


def make_checkpoint_contract(*, model_name: str, task: str, grid, model_kwargs: Dict,
                             loss_version: str, feature_version: Optional[str] = None,
                             data_contract: Optional[Dict] = None,
                             optimizer: Optional[Dict] = None) -> Dict:
    """The fields a checkpoint must match to be resumed or loaded for inference.

    ``model_kwargs`` should be exactly what ``read_model_kwargs`` returned
    (or what was passed to the model constructor), so a config edit that
    changes the model's shape is caught before ``load_state_dict`` turns it
    into a cryptic size-mismatch error.

    ``optimizer`` is the resolved training recipe -- the optimizer name, its
    numeric arguments and the schedule.  Since the four candidates diverged
    onto their own papers' recipes these differ per model AND per run, and
    without them two runs of one model at different learning rates produce
    identical contracts and resume into each other in silence.  Recording the
    recipe here rather than encoding it in ``loss_version`` keeps the loss
    string about the loss.  Optional so a caller with no optimizer (an export
    or an evaluation) can still build a contract.

    ``grid`` accepts either ``AIAEC.aiaec_common.SignalGrid`` (``.sample_rate``,
    the model-boundary type) or ``AIAEC.dataset_gen.AecGrid`` (``.sr``, the
    dataset-boundary type) -- two dataclasses for the same grid, named
    differently at the two layers; duck-typing here means a trainer never has
    to convert between them just to record a checkpoint contract.

    Model constructor kwargs are recorded under a ``ctor_`` prefix (not
    ``model_``) so they cannot collide with the fixed ``model_name`` field --
    a candidate could otherwise legitimately have a constructor keyword
    literally named ``name``.
    """
    sr = getattr(grid, 'sr', None)
    if sr is None:
        sr = grid.sample_rate
    contract = {
        'model_name': model_name,
        'task': task,
        'sr': sr, 'n_fft': grid.n_fft, 'win_len': grid.win_len, 'hop_len': grid.hop_len,
        'loss_version': loss_version,
        # Every current trainer feeds raw far (dataset_gen/model_views.py);
        # readers default a missing field to this via
        # checkpoint_far_input_mode, so the field is additive.
        'far_input_mode': 'raw_far',
    }
    if optimizer is not None:
        contract['optimizer'] = dict(optimizer)
    if feature_version is not None:
        contract['feature_version'] = feature_version
    if data_contract is not None:
        contract['data'] = data_contract
        contract['linear_aec'] = data_contract['linear_aec']
        contract['linear_aec_contract_hash'] = data_contract[
            'linear_aec_contract_hash'
        ]
    contract.update({f'ctor_{k}': v for k, v in sorted(model_kwargs.items())})
    return contract


def require_checkpoint_contract(ckpt: Dict, expected: Dict, context: str = 'checkpoint') -> None:
    """Reject a checkpoint whose recorded contract disagrees with ``expected``."""
    saved = ckpt.get('contract', {})
    if isinstance(saved, dict):
        # Legacy checkpoints predate 'far_input_mode'; the single defaulting
        # authority is checkpoint_far_input_mode (which also rejects an
        # unknown recorded mode), so resume does not fail on the added field.
        saved = dict(saved, far_input_mode=checkpoint_far_input_mode(saved))
    def compare(got, want, path):
        if isinstance(want, dict):
            if not isinstance(got, dict):
                return path, got, want
            for key, child_want in want.items():
                if key not in got:
                    return f"{path}.{key}", '<missing>', child_want
                mismatch = compare(got[key], child_want, f"{path}.{key}")
                if mismatch is not None:
                    return mismatch
            return None
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = math.isclose(float(got), want, rel_tol=1e-7, abs_tol=1e-7)
        else:
            ok = got == want
        return None if ok else (path, got, want)

    mismatch = compare(saved, expected, 'contract')
    if mismatch is not None:
        path, got, want = mismatch
        if isinstance(got, list):
            got = f"<list len={len(got)}>"
        if isinstance(want, list):
            want = f"<list len={len(want)}>"
        raise ValueError(
            f"{context} {path}={got!r}, but the running config requires "
            f"{want!r}. Retrain, or fix config.ini before resuming/loading "
            f"for inference."
        )


def require_checkpoint_model_identity(contract: Dict, model_name: str) -> None:
    """Reject a checkpoint from another model or an obsolete task target."""
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no valid contract")
    expected_task = MODEL_TASKS[model_name]
    if contract.get('model_name') != model_name:
        raise ValueError(
            f"checkpoint model_name={contract.get('model_name')!r}, expected "
            f"{model_name!r}"
        )
    if contract.get('task') != expected_task:
        raise ValueError(
            f"checkpoint task={contract.get('task')!r}, expected "
            f"{expected_task!r}; this checkpoint used a different training "
            "target and must not be used for inference"
        )


def require_checkpoint_linear_aec(contract: Dict, grid) -> Dict:
    """Validate and return the materialized PBFDKF contract for inference."""
    if 'linear_aec' not in contract:
        raise ValueError("checkpoint has no materialized linear_aec contract")
    if 'linear_aec_contract_hash' not in contract:
        raise ValueError("checkpoint has no linear_aec_contract_hash")
    linear = LinearAecContract.from_dict(contract['linear_aec'])
    if contract['linear_aec_contract_hash'] != linear.fingerprint():
        raise ValueError(
            "checkpoint linear_aec_contract_hash does not match linear_aec"
        )
    sr = getattr(grid, 'sr', getattr(grid, 'sample_rate', None))
    expected = (int(sr), int(grid.n_fft), int(grid.hop_len))
    actual = (linear.sample_rate, linear.frame_size, linear.hop_size)
    if actual != expected:
        raise ValueError(
            "checkpoint model/PBFDKF grid mismatch: "
            f"model sr/frame/hop={expected}, linear_aec={actual}"
        )
    return linear.as_dict()


# ============================================================
# Loss
# ============================================================

def compressed_spectral_loss(estimate: Tensor, target: Tensor, *,
                             compression: float = 0.3,
                             magnitude_weight: float = 1.0,
                             complex_weight: float = 1.0,
                             eps: float = 1e-12) -> Tensor:
    """Power-law compressed magnitude + complex L1 loss.

    This is the project fallback for candidates without a published objective.
    It is not the ULCNet frequency-domain loss: ULCNet compresses real and
    imaginary components independently and applies MSE, which is implemented
    by :func:`component_compressed_mse_loss`.
    """
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate/target shape mismatch: {tuple(estimate.shape)} vs "
            f"{tuple(target.shape)}")
    if not torch.is_complex(estimate) or not torch.is_complex(target):
        raise ValueError("compressed_spectral_loss expects complex spectra")
    mag_e = safe_abs(estimate, eps)
    mag_t = safe_abs(target, eps)
    comp_e = estimate * mag_e.pow(compression - 1.0)
    comp_t = target * mag_t.pow(compression - 1.0)
    complex_term = ((comp_e.real - comp_t.real).abs().mean()
                    + (comp_e.imag - comp_t.imag).abs().mean())
    magnitude_term = (mag_e.pow(compression) - mag_t.pow(compression)).abs().mean()
    return complex_weight * complex_term + magnitude_weight * magnitude_term


def component_compressed_mse_loss(estimate: Tensor, target: Tensor, *,
                                  compression: float = 0.3,
                                  eps: float = 1e-12) -> Tensor:
    """ULCNet frequency-domain MSE after component-wise compression.

    ULCNet applies ``sign(x) * abs(x) ** alpha`` independently to the real and
    imaginary STFT components.  This differs from conventional PLCPA, which
    compresses the complex magnitude while retaining the original phase.
    """
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate/target shape mismatch: {tuple(estimate.shape)} vs "
            f"{tuple(target.shape)}")
    if not torch.is_complex(estimate) or not torch.is_complex(target):
        raise ValueError(
            "component_compressed_mse_loss expects complex spectra")
    if not 0.0 < compression <= 1.0:
        raise ValueError("compression must be in (0, 1]")

    def compress_component(x: Tensor) -> Tensor:
        # Clamp only inside the selected nonzero branch.  abs(x)**c has an
        # infinite derivative at zero for c<1; the literal formula therefore
        # turns exact-zero bins into NaN gradients even though their value is
        # well defined.  This keeps f(0)=0 while making backward finite.
        magnitude = x.abs()
        # A 0-d zero broadcasts, so the false branch costs nothing; zeros_like
        # would allocate and fill a full [B,T,F] frame on each of the four
        # calls this loss makes per batch.
        return torch.where(
            magnitude > 0,
            x.sign() * magnitude.clamp_min(eps).pow(compression),
            x.new_zeros(()),
        )

    estimate_real = compress_component(estimate.real)
    estimate_imag = compress_component(estimate.imag)
    target_real = compress_component(target.real)
    target_imag = compress_component(target.imag)
    return ((estimate_real - target_real).square()
            + (estimate_imag - target_imag).square()).mean()


def power_compressed_complex_mse_loss(
        estimate: Tensor, target: Tensor, *, compression: float = 0.3,
        complex_weight: float = 0.7, eps: float = 1e-12) -> Tensor:
    """Align-CRUSE power-compressed magnitude/complex MSE (PLCPA).

    ``complex_weight`` is the paper's beta: beta weights the phase-aware
    complex error and ``1-beta`` weights the compressed magnitude error.
    """
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate/target shape mismatch: {tuple(estimate.shape)} vs "
            f"{tuple(target.shape)}")
    if not torch.is_complex(estimate) or not torch.is_complex(target):
        raise ValueError(
            "power_compressed_complex_mse_loss expects complex spectra")
    if not 0.0 < compression <= 1.0:
        raise ValueError("compression must be in (0, 1]")
    if not 0.0 <= complex_weight <= 1.0:
        raise ValueError("complex_weight must be in [0, 1]")

    mag_e = safe_abs(estimate, eps)
    mag_t = safe_abs(target, eps)
    mag_e_c = mag_e.pow(compression)
    mag_t_c = mag_t.pow(compression)
    complex_e = estimate * mag_e.pow(compression - 1.0)
    complex_t = target * mag_t.pow(compression - 1.0)
    magnitude_mse = (mag_e_c - mag_t_c).square().mean()
    complex_mse = ((complex_e.real - complex_t.real).square()
                   + (complex_e.imag - complex_t.imag).square()).mean()
    return ((1.0 - complex_weight) * magnitude_mse
            + complex_weight * complex_mse)


def si_snr_loss(estimate: Tensor, target: Tensor,
                eps: float = 1e-8) -> Tensor:
    """Negative scale-invariant SNR, averaged over the leading dimensions."""
    if estimate.shape != target.shape:
        raise ValueError(
            f"estimate/target shape mismatch: {tuple(estimate.shape)} vs "
            f"{tuple(target.shape)}")
    if estimate.ndim < 1 or torch.is_complex(estimate) or torch.is_complex(target):
        raise ValueError("si_snr_loss expects real waveforms ending in time")
    estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)
    projection = ((estimate_zm * target_zm).sum(dim=-1, keepdim=True)
                  * target_zm
                  / target_zm.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    noise = estimate_zm - projection
    ratio = (projection.square().sum(dim=-1)
             / noise.square().sum(dim=-1).clamp_min(eps))
    return -10.0 * torch.log10(ratio.clamp_min(eps)).mean()


def spectrum_to_waveform(spectrum: Tensor, grid: AecGrid) -> Tensor:
    """Public ``[B,T,F]`` complex spectrum -> differentiable waveform."""
    if spectrum.ndim != 3 or not torch.is_complex(spectrum):
        raise ValueError("spectrum_to_waveform expects complex [B,T,F]")
    if spectrum.shape[-1] != grid.n_freqs:
        raise ValueError(
            f"spectrum has {spectrum.shape[-1]} bins, expected {grid.n_freqs}")
    return istft(spectrum.transpose(-2, -1), grid)


def stft_consistent_spectrum(spectrum: Tensor, grid: AecGrid) -> Tensor:
    """Project a public ``[B,T,F]`` spectrum through iSTFT -> STFT."""
    waveform = spectrum_to_waveform(spectrum, grid)
    consistent = stft(waveform, grid).transpose(-2, -1)
    if consistent.shape != spectrum.shape:
        raise RuntimeError(
            "STFT consistency changed shape: "
            f"{tuple(spectrum.shape)} -> {tuple(consistent.shape)}")
    return consistent


# ============================================================
# Optimizer
# ============================================================

#: ``[training] optimizer`` -> class.  AdamW is the default because plain Adam's
#: COUPLED L2 term is a weight-killer on a branch whose gradient has gone quiet:
#: ``weight_decay * w`` is then the only consistently signed contribution left in
#: that tensor's gradient, and Adam divides it by ``sqrt(v_hat)`` of that same
#: contribution, so the normalised step is ~1 and the weight moves a full ~lr
#: per step -- through zero and on down into denormal magnitudes.  AdamW applies
#: the identical ``weight_decay`` OUTSIDE the moment normaliser, a bounded
#: ``(1 - lr * weight_decay)`` shrink per step that no vanishing gradient can
#: amplify.  ``adam`` stays selectable so a run that predates the switch can be
#: reproduced exactly.
OPTIMIZERS = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
}
DEFAULT_OPTIMIZER = 'adamw'


def make_optimizer(cfg, params, *, lr: float) -> torch.optim.Optimizer:
    """Build the legacy configurable optimizer used by older experiments.

    New paper-reference trainers construct the optimizer named by their own
    recipe directly.  This selector remains for checkpoint reproduction and
    for candidates without a published optimizer; an unknown name is refused.

    ``weight_decay`` keeps its nominal 1e-4 across the switch.  Decoupled, that
    number is a much gentler regulariser than the same number was coupled, and
    that is the intent: bounded shrink instead of an unbounded push once a
    tensor's gradient vanishes.
    """
    name = cfg.get('training', 'optimizer', fallback=DEFAULT_OPTIMIZER)
    name = name.strip().lower()
    if name not in OPTIMIZERS:
        raise ValueError(
            f"[training] optimizer must be one of {sorted(OPTIMIZERS)}, "
            f"got {name!r}"
        )
    return OPTIMIZERS[name](
        params, lr=lr,
        weight_decay=cfg.getfloat('training', 'weight_decay', fallback=1e-4),
        amsgrad=cfg.getboolean('training', 'amsgrad', fallback=True),
    )


# ============================================================
# NaN-halt machinery (thin AEC-flavoured wrapper; see the docstring above)
# ============================================================

def halt_on_non_finite(reason: str, *, model, mic: Tensor, target: Tensor,
                       epoch: int, batch_idx: int, global_step: int,
                       loss_value: float, total_norm: float, output_dir: str,
                       sr: int, checkpoint: Dict, enhanced: Optional[Tensor] = None) -> None:
    """Dump evidence and raise, delegating to ``AINR.training_common``.

    ``mic``/``target`` are whatever the caller's loss operates on -- for every
    AIAEC trainer that is a COMPLEX SPECTRUM, not a waveform, because the
    model-specific spectral objectives run in that domain. The delegate's
    WAV dump therefore raises inside its own try/except and is skipped (it
    tries ``torchaudio.save`` on a complex tensor); the ``batch.pt`` tensor
    dump and ``lanes.txt`` per-lane summary, which do not assume a waveform
    dtype, still capture full evidence. Threading the pre-STFT waveform
    through for a real audio dump is a straightforward follow-up, not done
    here to avoid a seventh per-trainer plumbing path.

    """
    _halt_on_non_finite(
        reason, model=model,
        noisy=mic, clean=target, epoch=epoch, batch_idx=batch_idx,
        global_step=global_step, loss_value=loss_value, total_norm=total_norm,
        output_dir=output_dir, sr=sr, checkpoint=checkpoint, enhanced=enhanced,
    )


# ============================================================
# Frozen linear AEC for offline/file inference
# ============================================================

class LinearAecEngine:
    """Continuous Python-PBFDKF wrapper for RES+NR file inference.

    Training reads the already-materialized `linear_error` stem and never calls
    this class. Inference constructs it from the checkpoint's exact
    ``linear_aec`` contract, processes the whole file as one stateful stream,
    and recovers ``D_hat`` by subtraction.

    The multi-lane/reset surface is retained for diagnostics and parity tests,
    not for trainer batching.
    """

    def __init__(self, n_lanes: int, sample_rate: int, preset: str = 'balanced',
                 filter_length: Optional[int] = None,
                 frame_size: Optional[int] = None,
                 contract: Optional[Dict] = None,
                 delay_num_filters: Optional[int] = None):
        if n_lanes <= 0:
            raise ValueError(f"n_lanes must be positive, got {n_lanes}")
        self.n_lanes = int(n_lanes)
        self.sample_rate = int(sample_rate)
        self.contract = (
            LinearAecContract.from_dict(contract)
            if contract is not None else
            make_linear_aec_contract(
                self.sample_rate, preset=preset, frame_size=frame_size,
                filter_length=filter_length,
            )
        )
        if self.contract.sample_rate != self.sample_rate:
            raise ValueError(
                f"linear AEC contract sr={self.contract.sample_rate}, requested "
                f"sample_rate={self.sample_rate}"
            )
        self._requested_delay_num_filters = (
            None if delay_num_filters is None
            else check_delay_num_filters(delay_num_filters)
        )
        runtime_contract = make_linear_aec_contract(
            self.contract.sample_rate,
            preset=self.contract.preset,
            frame_size=self.contract.frame_size,
            filter_length=self.contract.filter_length,
        )
        require_linear_aec_contract(
            runtime_contract.as_dict(), self.contract.as_dict(), "inference runtime"
        )
        self._engines: List[AEC] = [self._new_engine() for _ in range(self.n_lanes)]
        self.delay_num_filters = engine_delay_num_filters(self._engines[0])
        expected_filters = (
            DATASET_DELAY_NUM_FILTERS
            if self._requested_delay_num_filters is None
            else self._requested_delay_num_filters
        )
        if self.delay_num_filters != expected_filters:
            raise ValueError(
                f"linear AEC built a {self.delay_num_filters}-filter matched "
                f"bank, but {expected_filters} was requested"
            )
        self._pending_reset: Optional[List[bool]] = None
        self._last_aligned_far: Optional[Tensor] = None

    def _new_engine(self) -> AEC:
        cfg = make_linear_aec_config(
            self.contract.sample_rate,
            preset=self.contract.preset,
            frame_size=self.contract.frame_size,
            filter_length=self.contract.filter_length,
            delay_num_filters=self._requested_delay_num_filters,
        )
        return AEC(cfg)

    def reset_lane(self, lane: int) -> None:
        self._engines[lane] = self._new_engine()

    def arm_reset(self, reset_mask: Sequence[bool]) -> None:
        """Mark which lanes must restart cold on the NEXT call."""
        if len(reset_mask) != self.n_lanes:
            raise ValueError(
                f"reset_mask has {len(reset_mask)} entries, expected "
                f"{self.n_lanes} (one per lane)")
        self._pending_reset = [bool(r) for r in reset_mask]

    def _process_numpy(
        self, mic: np.ndarray, far: np.ndarray, reset_mask: Sequence[bool]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns ``(error, echo_estimate, aligned_far)`` at ``mic.shape``.

        The returned signal must stay the same shape as the microphone, so
        this cannot truncate to a whole number of hops. A file may end with
        a sub-hop remainder. That remainder is too short to feed
        ``AEC.process()`` at all, so it
        passes through with no cancellation applied (``error = mic``,
        ``echo_estimate = 0`` there) -- the honest answer for a fragment the
        engine never got to see, not an invented one.
        """
        if mic.shape != far.shape or mic.ndim != 2 or mic.shape[0] != self.n_lanes:
            raise ValueError(
                f"mic/far must both be ({self.n_lanes}, T), got {mic.shape} vs "
                f"{far.shape}")
        for lane, reset in enumerate(reset_mask):
            if reset:
                self.reset_lane(lane)

        hop = self._engines[0].hop_size
        n_samples = mic.shape[1]
        used = (n_samples // hop) * hop
        error = np.zeros((self.n_lanes, n_samples), dtype=np.float32)
        # A sub-hop tail is not processed by AEC and therefore remains the
        # caller's raw far, matching the error tail's pass-through policy.
        aligned_far = np.ascontiguousarray(far, dtype=np.float32).copy()
        for lane in range(self.n_lanes):
            engine = self._engines[lane]
            mic_lane = np.ascontiguousarray(mic[lane, :used], dtype=np.float32)
            far_lane = np.ascontiguousarray(far[lane, :used], dtype=np.float32)
            for start in range(0, used, hop):
                # Match the offline materializer: use the selected/crossfaded
                # WOLA seam, not process()'s raw main-filter return.
                engine.process(mic_lane[start:start + hop],
                               far_lane[start:start + hop])
                error[lane, start:start + hop] = engine.get_formed_output()
                # Exact Python equivalent of C AecLinearContext's
                # aligned_far_hop: the hop PBFDKF just consumed.
                aligned_far[lane, start:start + hop] = (
                    engine.filter.far_buffer[-hop:]
                )
            if used < n_samples:
                error[lane, used:] = mic[lane, used:]
        echo_estimate = mic.astype(np.float32) - error
        return error, echo_estimate, aligned_far

    def get_aligned_far(self) -> Tensor:
        """Far seam from the most recent call, on that call's device/dtype."""
        if self._last_aligned_far is None:
            raise RuntimeError("LinearAecEngine has not processed audio yet")
        return self._last_aligned_far

    def __call__(self, mic: Tensor, far: Tensor, sample_rate: int) -> Tuple[Tensor, Tensor]:
        """``LinearAecFrontend`` signature: ``(mic, far, sample_rate) -> (error, echo_estimate)``."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"engine was built for sample_rate={self.sample_rate}, got "
                f"{sample_rate}")
        reset_mask = self._pending_reset
        self._pending_reset = None
        if reset_mask is None:
            reset_mask = [False] * self.n_lanes
        mic_np = mic.detach().cpu().numpy()
        far_np = far.detach().cpu().numpy()
        error_np, echo_np, aligned_np = self._process_numpy(
            mic_np, far_np, reset_mask
        )
        error = torch.from_numpy(error_np).to(device=mic.device, dtype=mic.dtype)
        echo_estimate = torch.from_numpy(echo_np).to(device=mic.device, dtype=mic.dtype)
        self._last_aligned_far = torch.from_numpy(aligned_np).to(
            device=far.device, dtype=far.dtype
        )
        return error, echo_estimate
