"""Shared training infrastructure for every AIAEC candidate trainer.

Six model directories -- Align_CRUSE, Align_ULCNet, GTCRN_AENR,
DeepFilterNet_AENR, DeepVQE_S, CAGCRN -- each own a ``train.py`` /
``config.ini`` / ``denoise.py`` mirroring AINR's per-project layout (see any
of their top-of-file docstrings for the exact usage). What they do NOT each
keep a private copy of lives here, for the same reason
``AIAEC/dataset_gen/aec_features.py`` holds the signal grid and
``AINR/dataset_gen`` holds the seeder: a knob that drifts between six
near-identical copies is a checkpoint- or corpus-comparability bug that
surfaces only as "one model is mysteriously worse."

Provided here:

* ``set_seed``                                -- re-exported from AINR.dataset_gen
* ``read_model_kwargs``                       -- ``[model]`` config -> constructor
  kwargs, generalising AINR/DeepFilterNet2's ``read_model_config`` (there:
  one model class; here: six)
* ``make_checkpoint_contract`` /
  ``require_checkpoint_contract``             -- reject a checkpoint whose grid,
  task, model kwargs or loss version differ from the running config
* ``scan_non_finite`` / ``NonFiniteTraining`` / ``GradNormLog`` /
  ``halt_on_non_finite``                      -- the NaN-halt machinery
  DeepFilterNet2's alignment pass built is reused via
  ``AINR.DeepFilterNet2.train``, not reimplemented a second time
* ``compressed_spectral_loss``                -- none of the six candidates'
  papers publish a loss (see ``docs/ai_aec_candidate_matrix.md`` and the
  DeepVQE_S/CAGCRN READMEs' "did not publish ... loss details"); this is the
  one loss every trainer uses, so scores stay comparable across candidates
* ``LinearAecEngine``                         -- per-lane frozen production
  linear AEC (``lib/aec``, RES+CNG disabled) for the three
  "linear AEC -> RES+NR" candidates (Align_ULCNet, GTCRN_AENR,
  DeepFilterNet_AENR)

Do not add a seventh copy of any of this into a candidate's ``train.py``. If a
candidate genuinely needs different behaviour, change the signature here so
every trainer can see the choice was made -- the same reasoning
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
from torch import Tensor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(_THIS_DIR)
_LIB_AEC_PYTHON = os.path.join(_AUDIO_ALG_ROOT, 'lib', 'aec', 'python')
for _path in (_AUDIO_ALG_ROOT, _LIB_AEC_PYTHON):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from AINR.dataset_gen import set_seed  # noqa: E402
from AINR.DeepFilterNet2.train import (  # noqa: E402
    GradNormLog,
    NonFiniteTraining,
    dump_batch as _dump_batch,
    halt_on_non_finite as _halt_on_non_finite,
    scan_non_finite,
)

# lib/aec/python/aec.py re-exports the engine; its own modules use bare
# `from modules.xxx import ...`, which resolves only with lib/aec/python
# itself (not lib/aec) on sys.path -- the same setup run_one_case.py and
# eval_aec_challenge.py use.
from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402

from AIAEC.aiaec_common import SignalGrid, safe_abs  # noqa: E402
from AIAEC.dataset_gen import AecGrid  # noqa: E402


__all__ = [
    'set_seed',
    'build_arg_parser',
    'auto_device',
    'read_grids',
    'read_model_kwargs',
    'make_checkpoint_contract',
    'require_checkpoint_contract',
    'scan_non_finite',
    'NonFiniteTraining',
    'GradNormLog',
    'halt_on_non_finite',
    'compressed_spectral_loss',
    'LinearAecEngine',
]


# ============================================================
# CLI
# ============================================================

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """The five flags every AIAEC train.py accepts, so they stop drifting.

    A candidate whose training genuinely needs an extra flag adds it to the
    parser this returns -- see any train.py's ``build_parser()`` for the
    pattern -- rather than re-declaring these five.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='config.ini', help='Config file path')
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


def auto_device(requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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
# Config -> constructor kwargs
# ============================================================

def read_model_kwargs(cfg, model_cls, section: str = 'model',
                      aliases: Optional[Dict[str, str]] = None,
                      extra_bases: Sequence[type] = (),
                      exclude: Sequence[str] = ()) -> Dict:
    """Every ``model_cls.__init__`` keyword argument, overlaid with ``[model]``.

    ``grid`` (and ``self``) are excluded: the signal grid comes from
    ``[signal]`` via ``AecGrid.from_config``, not from a second copy in
    ``[model]``. Unknown ``[model]`` keys raise, and the parsed type follows
    the constructor's own default -- exactly
    ``AINR/DeepFilterNet2/train.py``'s ``read_model_config``, generalised from
    one model class to whichever ``model_cls`` a candidate's train.py passes.

    ``extra_bases`` covers a subclass whose own ``__init__`` forwards a
    ``**kwargs`` catch-all to a base constructor (``DeepFilterNetAENR`` ->
    ``DeepFilterNet2``, see that trainer): those base keywords are invisible
    to plain introspection of the subclass, so pass the base class here to
    have its defaults merged in too. A name already found on ``model_cls`` or
    an earlier base wins -- later bases never override -- and ``exclude``
    drops base keywords the subclass already binds explicitly (e.g. ``n_erb``,
    which ``DeepFilterNetAENR.__init__`` takes itself and passes through
    positionally; leaving it merged in would make it configurable under a
    name the constructor call would then pass twice).
    """
    aliases = aliases or {}
    sig = inspect.signature(model_cls.__init__)
    kwargs = {
        name: param.default for name, param in sig.parameters.items()
        if name not in ('self', 'grid') and param.default is not inspect.Parameter.empty
    }
    for base in extra_bases:
        base_sig = inspect.signature(base.__init__)
        for name, param in base_sig.parameters.items():
            if name in ('self', 'grid') or name in exclude or name in kwargs:
                continue
            if param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
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

def make_checkpoint_contract(*, model_name: str, task: str, grid, model_kwargs: Dict,
                             loss_version: str, feature_version: Optional[str] = None) -> Dict:
    """The fields a checkpoint must match to be resumed or loaded for inference.

    ``model_kwargs`` should be exactly what ``read_model_kwargs`` returned
    (or what was passed to the model constructor), so a config edit that
    changes the model's shape is caught before ``load_state_dict`` turns it
    into a cryptic size-mismatch error.

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
    }
    if feature_version is not None:
        contract['feature_version'] = feature_version
    contract.update({f'ctor_{k}': v for k, v in sorted(model_kwargs.items())})
    return contract


def require_checkpoint_contract(ckpt: Dict, expected: Dict, context: str = 'checkpoint') -> None:
    """Reject a checkpoint whose recorded contract disagrees with ``expected``."""
    saved = ckpt.get('contract', {})
    for key, want in expected.items():
        got = saved.get(key, '<missing>')
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = math.isclose(float(got), want, rel_tol=1e-7, abs_tol=1e-7)
        else:
            ok = got == want
        if not ok:
            raise ValueError(
                f"{context} contract {key}={got!r}, but the running config "
                f"requires {want!r}. Retrain, or fix config.ini before resuming/"
                f"loading for inference."
            )


# ============================================================
# Loss
# ============================================================

def compressed_spectral_loss(estimate: Tensor, target: Tensor, *,
                             compression: float = 0.3,
                             magnitude_weight: float = 1.0,
                             complex_weight: float = 1.0,
                             eps: float = 1e-12) -> Tensor:
    """Power-law compressed magnitude + complex L1 loss.

    No AIAEC candidate paper publishes a loss function -- see
    ``docs/ai_aec_candidate_matrix.md`` and the DeepVQE_S / CAGCRN READMEs
    ("did not publish ... loss details"). This is therefore a project choice
    used identically by all six trainers, so scores stay comparable across
    candidates instead of each optimising a different objective.
    ``compression=0.3`` matches the compression exponent Align-ULCNet and
    DeepVQE-S already use for their own INPUT features (see
    ``aiaec_common.compressed_ri_feature``), so the loss and the feature
    domain share one compression law rather than two unrelated ones.
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


# ============================================================
# NaN-halt machinery (thin AEC-flavoured wrapper; see the docstring above)
# ============================================================

def halt_on_non_finite(reason: str, *, model, optimizer, mic: Tensor, target: Tensor,
                       epoch: int, batch_idx: int, global_step: int,
                       loss_value: float, total_norm: float, output_dir: str,
                       sr: int, checkpoint: Dict, enhanced: Optional[Tensor] = None) -> None:
    """Dump evidence and raise, delegating to DeepFilterNet2's implementation.

    ``mic``/``target`` are whatever the caller's loss operates on -- for every
    AIAEC trainer that is a COMPLEX SPECTRUM, not a waveform, because
    ``compressed_spectral_loss`` runs in the spectral domain. The delegate's
    WAV dump therefore raises inside its own try/except and is skipped (it
    tries ``torchaudio.save`` on a complex tensor); the ``batch.pt`` tensor
    dump and ``lanes.txt`` per-lane summary, which do not assume a waveform
    dtype, still capture full evidence. Threading the pre-STFT waveform
    through for a real audio dump is a straightforward follow-up, not done
    here to avoid a seventh per-trainer plumbing path.

    No trainer here has a scheduler (see each train.py's own note on why);
    ``scheduler=None`` is passed through unconditionally.
    """
    _halt_on_non_finite(
        reason, model=model, optimizer=optimizer, scheduler=None,
        noisy=mic, clean=target, epoch=epoch, batch_idx=batch_idx,
        global_step=global_step, loss_value=loss_value, total_norm=total_norm,
        output_dir=output_dir, sr=sr, checkpoint=checkpoint, enhanced=enhanced,
    )


# ============================================================
# Frozen production linear AEC (for the three "-> RES+NR" candidates)
# ============================================================

class LinearAecEngine:
    """Per-lane frozen production linear AEC, RES and CNG disabled.

    Wraps the reference Python engine in ``lib/aec/python`` -- the same
    engine the AEC repo 800-case-benches and treats as the fp64 algorithm
    spec for the float32 C production port (``lib/aec/CLAUDE.md``).
    ``enable_res=False, enable_cng=False`` makes ``AEC.process()`` return the
    linear PBFDKF residual before any suppression gain -- documented in
    ``lib/aec/CLAUDE.md``: "running with --enable-res 0 emits the linear
    residual at PBFDKF output" -- i.e. exactly ``E = Y - D_hat``. The echo
    estimate is recovered by subtraction, ``D_hat = Y - E``, never a second
    read of internal filter state; this matches the signal model documented
    in ``AIAEC/dataset_gen/aec_features.py`` and ``dataset_gen/README.md``.

    One ``AEC`` instance per lane, because the filter's convergence state --
    cold at a sequence's first chunk, progressively converged afterwards --
    is exactly the thing the corpus's long stateful sequences and
    ``SequenceChunkSampler`` exist to make realistic (see
    ``dataset_gen/README.md``, "Why sequences are long"). A lane resets to a
    brand new ``AEC`` object -- not a partial ``.reset()`` of unaudited scope
    -- whenever the caller's ``reset_mask`` says that lane's sequence just
    started, the same reset signal ``lane_reset_mask`` gives a model's own
    recurrent state.

    Usage (see a candidate's ``train.py`` for the full loop): the
    ``model_views.build_model_view`` ``linear_aec`` callback has a fixed
    ``(mic, far, sample_rate) -> (error, echo_estimate)`` signature with no
    room for per-call reset information, so the reset mask is armed
    separately and consumed by the next call:

        engine.arm_reset(lane_reset_mask(meta_chunk_indices).tolist())
        view = build_model_view(stems, model_name, grid.sr, linear_aec=engine)

    ⚠ This is the reference PYTHON engine, not the shipped C build. It is
    CPU-costly per step (a Python per-hop loop, times n_lanes, every training
    step) and, being frozen and deterministic, repeats identical work every
    epoch. The AI-AEC plan's Milestone 2 (docs/archive/
    ai_aec_candidate_matrix_2026_07_30.md) calls for precomputing this once
    with the shipped C build and caching it, stratified by filter state; that
    caching layer does not exist yet, so today it runs live. Budget batch
    size / lane count accordingly.
    """

    def __init__(self, n_lanes: int, sample_rate: int, preset: str = 'balanced',
                 filter_length: Optional[int] = None):
        if n_lanes <= 0:
            raise ValueError(f"n_lanes must be positive, got {n_lanes}")
        self.n_lanes = int(n_lanes)
        self.sample_rate = int(sample_rate)
        self._preset = AecPreset(preset)
        self._filter_length = filter_length
        self._engines: List[AEC] = [self._new_engine() for _ in range(self.n_lanes)]
        self._pending_reset: Optional[List[bool]] = None

    def _new_engine(self) -> AEC:
        overrides = {}
        if self._filter_length is not None:
            overrides['filter_length'] = self._filter_length
        cfg = AecConfig.from_preset(
            self._preset, sample_rate=self.sample_rate, mode=AecMode.PBFDKF,
            enable_res=False, enable_cng=False, enable_shadow=True, **overrides,
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

    def _process_numpy(self, mic: np.ndarray, far: np.ndarray,
                       reset_mask: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(error, echo_estimate)``, both EXACTLY ``mic.shape``.

        ``model_views.build_model_view`` requires the linear-AEC error to be
        the same shape as the microphone (it rejects a shorter one), so this
        cannot truncate to a whole number of hops the way
        ``process_wav_files`` does. A 3 s chunk at 16 kHz/hop 256 is 187.5
        hops -- there is always a sub-hop remainder for SOME chunk length,
        this project's included. The trailing remainder (< one hop, at most
        ~16 ms at 16 kHz) is too short to feed ``AEC.process()`` at all, so it
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
        for lane in range(self.n_lanes):
            engine = self._engines[lane]
            mic_lane = np.ascontiguousarray(mic[lane, :used], dtype=np.float32)
            far_lane = np.ascontiguousarray(far[lane, :used], dtype=np.float32)
            for start in range(0, used, hop):
                block = engine.process(mic_lane[start:start + hop],
                                       far_lane[start:start + hop])
                error[lane, start:start + hop] = block
            if used < n_samples:
                error[lane, used:] = mic[lane, used:]
        echo_estimate = mic.astype(np.float32) - error
        return error, echo_estimate

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
        error_np, echo_np = self._process_numpy(mic_np, far_np, reset_mask)
        error = torch.from_numpy(error_np).to(device=mic.device, dtype=mic.dtype)
        echo_estimate = torch.from_numpy(echo_np).to(device=mic.device, dtype=mic.dtype)
        return error, echo_estimate
