"""
AECNet training script -- stage-1 echo estimation.

Usage:
    python3 train.py --config config.ini --packed-dir data_aec/packed/train \
                     --val-packed-dir data_aec/packed/val --gpu 0
    python3 train.py --config config.ini --packed-dir data_aec/packed/train \
                     --resume output/aecnet_best.pth

The model's target is the ECHO, D.  ``E = Y - D_hat`` is a subtraction done by
whatever consumes this stage, not something the network emits; see model.py.
"""

import argparse
import configparser
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import tqdm

from model import (
    _MAG_EPS,
    AecNet,
    AecNetConfig,
    build_model,
    compress_spec,
    describe,
    safe_mag,
    zero_reference_leak_db,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The split, the seeder and the worker kwargs are shared by every model in
# ainr/ -- see dataset_gen/loader.py for what happened the one time they were
# copied into each trainer (5% held out on one side, 10% on the other, so two
# models being compared trained on different corpora).  Nothing in this file
# re-declares them; ainr/tests/test_bakeoff_protocol.py enforces that rule for
# the NR trainers and tests/test_aecnet_contract.py mirrors it here.
from dataset_gen import (  # noqa: E402
    DEFAULT_VAL_FRACTION,
    dataloader_worker_kwargs,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
    subsets_from_indices,
)
from dataset_gen.aec import (  # noqa: E402
    AecGrid,
    AecStems,
    PackedAecDataset,
    SequenceChunkSampler,
    aec_collate,
    lane_reset_mask,
    stft,
)


# ============================================================
# Checkpoint contract
# ============================================================
#
# Bumped whenever a change makes previously-trained weights invalid or
# meaningless to resume.  Same gate as GTCRN/train.py and DeepFilterNet2/train.py.
MODEL_VERSION = 'aecnet_crn_groupedgru_echo_regression_v1'
FEATURE_VERSION = 'aecnet_compressed_complex_yx_stack_v1'
# The loss string names every term, because three of the four are weighted by
# config knobs whose value does not appear in the weights: a checkpoint trained
# with lambda_idle = 0 is a different model from one trained with 0.5 and
# nothing but this string plus the contract fields can tell them apart.
LOSS_VERSION = 'aecnet_echo_mae_plus_output_plus_nearpres_plus_guarded_idle_v2'
# v2: the ECHO term became MAE (L1).  v1 used a squared compressed-spectral
# distance for it, which Braun & Valero (IWAENC 2022) report causes
# "significant under-estimation of the echo" -- the one failure mode this
# project cannot tolerate, because an under-estimated D_hat silently pushes
# work into stage 2 while every stage-1 metric still looks fine.  See
# EchoEstimationLoss._pair.  Selectable via [training] echo_norm.

_VERSIONS = {
    'model_version': MODEL_VERSION,
    'feature_version': FEATURE_VERSION,
    'loss_version': LOSS_VERSION,
}

# ⚠ DERIVED from the version dict, never restated.  Adding a fourth version
# string above therefore cannot leave it unvalidated, which is the failure the
# hand-written tuple in GTCRN/train.py is one edit away from.
_VERSION_FIELDS = tuple(_VERSIONS)


def read_loss_config(cfg) -> Dict[str, float]:
    """The ``[training]`` keys that define the objective, in one place."""
    loss_cfg = {
        'mag_weight': cfg.getfloat('training', 'mag_weight', fallback=1.0),
        'lambda_out': cfg.getfloat('training', 'lambda_out', fallback=1.0),
        'lambda_near': cfg.getfloat('training', 'lambda_near', fallback=0.5),
        'lambda_idle': cfg.getfloat('training', 'lambda_idle', fallback=1.0),
        'idle_guard_sec': cfg.getfloat('training', 'idle_guard_sec', fallback=1.5),
        'far_active_dbfs': cfg.getfloat('training', 'far_active_dbfs', fallback=-60.0),
        'near_active_dbfs': cfg.getfloat('training', 'near_active_dbfs', fallback=-50.0),
        'echo_norm': cfg.get('training', 'echo_norm', fallback='l1').strip().lower(),
    }
    for key in ('mag_weight', 'lambda_out', 'lambda_near', 'lambda_idle'):
        if loss_cfg[key] < 0.0:
            raise ValueError(f"[training] {key} must be >= 0, got {loss_cfg[key]}")
    if loss_cfg['idle_guard_sec'] < 0.0:
        raise ValueError("[training] idle_guard_sec must be >= 0")
    if loss_cfg['echo_norm'] not in ('l1', 'l2'):
        raise ValueError(
            f"[training] echo_norm must be 'l1' or 'l2', got "
            f"{loss_cfg['echo_norm']!r}")
    return loss_cfg


def build_contract(cfg, grid: AecGrid, model_cfg: AecNetConfig,
                   loss_cfg: Dict[str, float]) -> Dict[str, object]:
    """Everything that changes what the weights mean."""
    contract = dict(_VERSIONS)
    contract.update({
        'sr': grid.sr,
        'n_fft': grid.n_fft,
        'win_len': grid.win_len,
        'hop_len': grid.hop_len,
        'n_freqs': grid.n_freqs,
    })
    contract.update(model_cfg.as_contract())
    # The loss weights are part of the contract, not only of LOSS_VERSION:
    # resuming a lambda_idle = 1.0 run under lambda_idle = 0.0 is a different
    # objective on the same weights and would look like a mysterious regression.
    contract.update({f'loss_{k}': v for k, v in loss_cfg.items()})
    return contract


def require_checkpoint_contract(ckpt: dict, contract: Dict[str, object],
                                context: str = 'checkpoint') -> None:
    """Refuse to resume or infer across a semantic change.

    There is no ``allow_missing`` escape hatch here, unlike GTCRN: this project
    vendors no upstream checkpoints, so every checkpoint that exists was written
    by this file and must carry a contract.  A pre-contract checkpoint is a
    checkpoint from a code state nobody can identify.
    """
    for key in _VERSION_FIELDS:
        got = ckpt.get(key)
        if got != contract[key]:
            shown = repr(got) if got is not None else 'missing (pre-contract checkpoint)'
            raise ValueError(
                f"{context} {key}={shown}, expected {contract[key]!r}. "
                f"Resuming across this change would train on incompatible "
                f"weights; start a fresh run instead."
            )
    for key, want in contract.items():
        if key in _VERSION_FIELDS:
            continue
        if key not in ckpt:
            raise ValueError(
                f"{context} is missing contract field {key!r} (expected {want!r}); "
                f"it predates this field being recorded, so its value cannot be "
                f"verified -- start a fresh run.")
        got = ckpt[key]
        if isinstance(want, str):
            matches = str(got) == want
        else:
            matches = got is not None and math.isclose(
                float(got), float(want), rel_tol=1e-9, abs_tol=1e-12)
        if not matches:
            raise ValueError(
                f"{context} {key}={got!r}, but config requires {want!r}.")


# ============================================================
# Activity, in dBFS, on the model's own frame grid
# ============================================================

def frame_level_db(wav: torch.Tensor, grid: AecGrid) -> torch.Tensor:
    """Per-frame level of a waveform, in dBFS, aligned with ``stft(center=True)``.

    Measured on the WAVEFORM rather than the spectrum on purpose: a threshold in
    dBFS is only meaningful against a signal whose scale is the sample scale,
    and ``torch.stft`` output carries the window's gain, so "-60 dB" of a
    spectrum is not -60 dBFS of anything.

    ⚠ No frame count appears here.  The pooling geometry is win_len/hop_len, so
    the returned length is exactly ``grid.n_frames(T)`` on any grid.
    """
    if wav.ndim != 2:
        raise ValueError(f"expected (B, T), got {tuple(wav.shape)}")
    pad = grid.win_len // 2
    power = F.pad(wav.pow(2).unsqueeze(1), (pad, pad))
    power = F.avg_pool1d(power, grid.win_len, grid.hop_len).squeeze(1)
    return 10.0 * torch.log10(power + 1e-20)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``(B, F, T)`` over the frames selected by ``(B, T)``.

    Returns exactly 0 when the mask selects nothing, so a batch with no idle
    frames contributes no gradient rather than a NaN.
    """
    weight = mask.unsqueeze(1)
    total = mask.sum() * value.shape[1]
    if float(total) <= 0.0:
        return value.sum() * 0.0
    return (value * weight).sum() / total


# ============================================================
# Loss
# ============================================================

class EchoEstimationLoss(nn.Module):
    """
        L = L_echo(D_hat, D)
          + lambda_out  * L_output(Y - D_hat, S + N)
          + lambda_near * near_end_preservation
          + lambda_idle * ||D_hat||   on frames where the reference is inactive

    Every spectral term lives in the compressed domain |Z|^c with the phase
    intact (c = the model's ``compress_exponent``, 0.3 by default).  ONE
    exponent, shared with the model, because the network predicts in that domain
    -- so ``L_echo`` is a plain MSE on the network's own output.  A separate
    loss exponent would put a c-th root between the parameters and the objective
    and reintroduce the dynamic-range problem the compression removes.

    Term by term:

    ``L_echo``  the primary objective: complex MSE plus a magnitude-only MSE.
        The magnitude term matters because a complex MSE alone is minimised by
        shrinking |D_hat| toward zero whenever the phase is uncertain, which is
        precisely the failure mode "the canceller does nothing" wears.

    ``L_output``  the same form on E = Y - D_hat against S + N.  It is what
        stops the model from trading a better echo fit for damage to the near
        path, and it is the ONLY place the subtraction appears -- the network
        still never sees S + N as its own target.

    ``near_end_preservation``  ASYMMETRIC, and deliberately so.  It penalises
        only the DEFICIT ``relu(|S+N|^c - |E|^c)`` on frames where the near
        talker is active: energy removed from the near path, never energy left
        behind.  ⚠ The asymmetry is the whole point.  Residual echo left in E
        can still be removed by a downstream residual suppressor; near speech
        that this stage has cancelled is gone for good and nothing downstream
        can restore it.  Treating the two symmetrically prices an irreversible
        loss the same as a recoverable one.

    ``idle``  compressed energy of D_hat on frames where the reference is
        silent, and it is what makes "reference silent => produce no echo
        estimate" trainable.  ⚠ It is GUARDED: a frame counts as idle only if
        the reference has been silent for ``idle_guard_sec`` CONTINUOUSLY.
        Without the guard, every brief pause in far-end speech would be labelled
        idle and the model would be punished for the echo TAIL that is still
        physically arriving -- i.e. trained to truncate reverberant echo, which
        is the opposite of what it is for.  The guard runs across chunk
        boundaries via ``far_tail``; without that, the first frames of every
        4 s chunk would look idle no matter how loud the previous chunk ended.

    ``lookahead`` must be the model's.  The model's output is delayed by that
    many frames (see ``AecNet.forward``), and this class realigns before
    comparing.  ⚠ A mismatch here trains against a target shifted in time, which
    presents as a model that simply will not converge.
    """

    def __init__(self, grid: AecGrid, compress_exponent: float,
                 mag_weight: float, lambda_out: float, lambda_near: float,
                 lambda_idle: float, idle_guard_sec: float,
                 far_active_dbfs: float, near_active_dbfs: float,
                 lookahead: int = 0, echo_norm: str = 'l1'):
        super().__init__()
        self.grid = grid
        self.c = compress_exponent
        self.echo_norm = echo_norm
        self.mag_weight = mag_weight
        self.lambda_out = lambda_out
        self.lambda_near = lambda_near
        self.lambda_idle = lambda_idle
        self.far_active_dbfs = far_active_dbfs
        self.near_active_dbfs = near_active_dbfs
        self.lookahead = int(lookahead)
        # SECONDS -> frames through the grid's own frame rate.  ⚠ Never a
        # literal frame count: 1.5 s is 94 frames at 16 kHz/hop 256 and 94
        # frames at 48 kHz/hop 512 only because this line does the conversion.
        self.guard_frames = max(1, int(round(idle_guard_sec * grid.frame_rate)))

    def guard_tail_width(self) -> int:
        return self.guard_frames - 1

    def _pair(self, pred: torch.Tensor, target: torch.Tensor,
              norm: str = 'l2') -> Tuple[torch.Tensor, torch.Tensor]:
        """Compressed-spectral distance between two complex spectra.

        ``norm`` selects L2 (squared error) or L1 (absolute error).  This is not
        a style knob.  Braun & Valero (IWAENC 2022), who introduced the
        echo-target + subtraction design this project follows, report that the
        obvious choice fails:

            "We found in preliminary experiments that using compressed spectral
             distances similar to (4) for the echo component results in
             significant under-estimation of the echo.  Therefore, we propose to
             use the mean absolute error (MAE), which provides accurate echo
             estimates."

        An under-estimated echo is the worst possible failure here, because the
        whole point of stage 1 is that whatever it leaves behind is handed to
        stage 2 as residual: a systematically small D_hat quietly moves work
        downstream while every stage-1 metric still looks reasonable.  So the
        ECHO term uses L1; the output term keeps L2, matching their eq. (4).
        """
        pred_c = compress_spec(pred, self.c)
        target_c = compress_spec(target, self.c)
        diff = pred_c - target_c
        if norm == 'l1':
            # Complex modulus |D_hat - D|.  ⚠ Floored INSIDE the sqrt for the
            # same reason _MAG_EPS exists at all: the idle term drives the
            # output to exact zero, where d|z|/dz is undefined and the gradient
            # would be NaN rather than merely large.
            spec = torch.sqrt(diff.real.pow(2) + diff.imag.pow(2)
                              + _MAG_EPS).mean()
            mag = (safe_mag(pred_c) - safe_mag(target_c)).abs().mean()
        elif norm == 'l2':
            spec = (diff.real.pow(2) + diff.imag.pow(2)).mean()
            mag = (safe_mag(pred_c) - safe_mag(target_c)).pow(2).mean()
        else:
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        return spec + self.mag_weight * mag, pred_c

    def forward(self, d_hat: torch.Tensor, y_spec: torch.Tensor,
                echo_spec: torch.Tensor, out_spec: torch.Tensor,
                far_wav: torch.Tensor, near_wav: torch.Tensor,
                far_tail: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """All spectra complex ``(B, F, T)``; waveforms ``(B, samples)``."""
        far_db = frame_level_db(far_wav, self.grid)
        near_db = frame_level_db(near_wav, self.grid)
        n_frames = d_hat.shape[-1]
        if far_db.shape[-1] != n_frames:
            raise ValueError(
                f"waveform gives {far_db.shape[-1]} frames but the spectra have "
                f"{n_frames}; the loss and the model are on different grids")

        far_active = (far_db > self.far_active_dbfs).to(d_hat.real.dtype)
        near_mask = (near_db > self.near_active_dbfs).to(d_hat.real.dtype)

        guard = self.guard_frames
        if guard > 1:
            if far_tail is None:
                far_tail = far_active.new_zeros(far_active.shape[0], guard - 1)
            padded = torch.cat([far_tail, far_active], dim=1)
            recent = F.max_pool1d(padded.unsqueeze(1), guard, 1).squeeze(1)
            new_tail = padded[:, -(guard - 1):].detach()
        else:
            recent = far_active
            new_tail = far_active.new_zeros(far_active.shape[0], 0)
        idle_mask = 1.0 - recent

        look = self.lookahead
        if look:
            keep = n_frames - look
            if keep <= 0:
                raise ValueError(
                    f"chunk is {n_frames} frames but the model's lookahead is "
                    f"{look}; nothing is left to score")
            d_hat = d_hat[..., look:]
            y_spec = y_spec[..., :keep]
            echo_spec = echo_spec[..., :keep]
            out_spec = out_spec[..., :keep]
            near_mask = near_mask[..., :keep]
            idle_mask = idle_mask[..., :keep]

        l_echo, d_hat_c = self._pair(d_hat, echo_spec, norm=self.echo_norm)

        residual = y_spec - d_hat
        l_out, residual_c = self._pair(residual, out_spec)

        deficit = F.relu(safe_mag(compress_spec(out_spec, self.c))
                         - safe_mag(residual_c))
        l_near = _masked_mean(deficit.pow(2), near_mask)

        l_idle = _masked_mean(
            d_hat_c.real.pow(2) + d_hat_c.imag.pow(2), idle_mask)

        total = (l_echo
                 + self.lambda_out * l_out
                 + self.lambda_near * l_near
                 + self.lambda_idle * l_idle)
        parts = {
            'echo': float(l_echo.detach()),
            'out': float(l_out.detach()),
            'near': float(l_near.detach()),
            'idle': float(l_idle.detach()),
            'idle_frames': float(idle_mask.sum().detach()),
        }
        return total, parts, new_tail


# ============================================================
# Splitting and batching
# ============================================================

def sequence_level_split(dataset: PackedAecDataset, seed: int,
                         val_fraction: float = DEFAULT_VAL_FRACTION
                         ) -> Tuple[List[int], List[int]]:
    """Hold out whole SEQUENCES, drawn by the shared split function.

    ⚠ Two things make a clip-level split wrong here, and only the first is
    obvious:

    1. Chunks of one sequence are near-duplicates of each other -- same room,
       same device, same talker, seconds apart.  Splitting over clips puts both
       halves of that pair on opposite sides of the fence and validation then
       measures memorisation.
    2. ``SequenceChunkSampler`` requires every sequence it sees to have a
       COMPLETE, contiguous run of chunk_index values.  A clip-level split
       shatters sequences and the sampler raises rather than silently walking a
       sequence with holes in it.

    The permutation still comes from ``locality_preserving_random_split``; only
    the UNIT it is drawn over changes, from clips to sequences.  Re-deriving the
    permutation here is what the bake-off drift guard forbids, and rightly.

    ⚠ This is the FALLBACK path.  The AEC corpus is generated with a
    source-disjoint split already decided (dataset_gen/aec/manifest.py), and
    passing --val-packed-dir uses it.  Splitting one corpus, however carefully,
    still shares speakers, rooms and loudspeakers across the fence.
    """
    seq_ids = sorted(set(dataset.sequence_ids()))
    n_train, n_val = split_sizes(seq_ids, val_fraction)
    train_subset, val_subset = locality_preserving_random_split(
        seq_ids, n_train, n_val, seed)
    train_seqs = {seq_ids[i] for i in train_subset.indices}
    val_seqs = {seq_ids[i] for i in val_subset.indices}
    all_ids = dataset.sequence_ids()
    train_indices = [i for i, s in enumerate(all_ids) if s in train_seqs]
    val_indices = [i for i, s in enumerate(all_ids) if s in val_seqs]
    return train_indices, val_indices


def sampler_for(dataset, n_lanes: int, seed: int, shuffle: bool
                ) -> SequenceChunkSampler:
    """A lane-per-sequence batch sampler over a dataset or one of its Subsets.

    The sampler indexes whatever the DataLoader is given, so for a ``Subset``
    the sequence/chunk metadata must be looked up globally and re-indexed
    locally -- handing it the global metadata would emit indices past the end of
    the subset.
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        seq_all, chunk_all = base.sequence_ids(), base.chunk_indices()
        seq_ids = [seq_all[i] for i in dataset.indices]
        chunk_ids = [chunk_all[i] for i in dataset.indices]
    else:
        seq_ids, chunk_ids = dataset.sequence_ids(), dataset.chunk_indices()
    return SequenceChunkSampler(seq_ids, chunk_ids, n_lanes=n_lanes,
                                shuffle=shuffle, seed=seed)


def _lanes_for(dataset, requested: int, name: str) -> int:
    """Clamp the lane count to the number of sequences available."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        seq_all = base.sequence_ids()
        n_seq = len({seq_all[i] for i in dataset.indices})
    else:
        n_seq = dataset.n_sequences()
    if requested <= n_seq:
        return requested
    print(f"  ⚠ {name}: batch_size {requested} exceeds the {n_seq} sequence(s) "
          f"available; using {n_seq} lanes. Every lane needs its own sequence or "
          f"two lanes would replay the same recurrent state.")
    return n_seq


# ============================================================
# One pass over a loader
# ============================================================

def _stems_to_spectra(stems: torch.Tensor, grid: AecGrid):
    """``(B, 6, T)`` stems -> the four spectra the loss needs, in one STFT."""
    view = AecStems(stems)
    bundle = torch.stack(
        [view.Y, view.X, view.D, view.S + view.N], dim=1)   # (B, 4, T)
    spec = stft(bundle, grid)                               # (B, 4, F, T)
    return spec[:, 0], spec[:, 1], spec[:, 2], spec[:, 3], view


def run_epoch(model: AecNet, loader, criterion: EchoEstimationLoss,
              grid: AecGrid, device, optimizer=None, grad_clip: float = 5.0,
              max_steps: int = 0, desc: str = '') -> Dict[str, float]:
    """Walk the loader keeping per-lane recurrent state across consecutive chunks.

    ⚠ The state carry is the reason this is not a plain ``for batch in loader``.
    Lane k of batch b+1 holds the NEXT chunk of the sequence lane k held in
    batch b, so the recurrent state is valid across the boundary and a whole
    20-60 s sequence is seen unbroken.  A trainer that reset state every chunk
    would look identical on the loss curve and would be unable to demonstrate
    convergence from cold, recovery after an echo-path change, or drift.
    """
    training = optimizer is not None
    model.train(training)
    totals = {'loss': 0.0, 'echo': 0.0, 'out': 0.0, 'near': 0.0, 'idle': 0.0}
    idle_frames = 0.0
    steps = 0
    state = None
    far_tail = None

    progress = tqdm.tqdm(loader, desc=desc, leave=False)
    for stems, meta in progress:
        reset = lane_reset_mask([m['chunk_index'] for m in meta]).to(device)
        state = model.reset_state(state, reset)
        if far_tail is not None:
            far_tail = far_tail.masked_fill(reset.view(-1, 1), 0.0)

        stems = stems.to(device=device, dtype=torch.float32)
        y_spec, x_spec, d_spec, o_spec, view = _stems_to_spectra(stems, grid)

        with torch.set_grad_enabled(training):
            d_hat, state = model.forward_spec(y_spec, x_spec, state)
            loss, parts, far_tail = criterion(
                d_hat, y_spec, d_spec, o_spec, view.X, view.S, far_tail)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ⚠ Truncated BPTT.  The state is carried forward but detached, so the
        # graph covers one chunk.  Without this the graph grows for the whole
        # sequence and the first backward pass runs out of memory.
        state = AecNet.detach_state(state)

        totals['loss'] += float(loss.detach())
        for key in ('echo', 'out', 'near', 'idle'):
            totals[key] += parts[key]
        idle_frames += parts['idle_frames']
        steps += 1
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if max_steps and steps >= max_steps:
            break
    progress.close()

    if steps == 0:
        raise RuntimeError("loader produced no batches")
    result = {key: value / steps for key, value in totals.items()}
    result['steps'] = steps
    result['idle_frames'] = idle_frames
    return result


# ============================================================
# Train
# ============================================================

def train(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    # Seed before anything draws randomness, so two runs of the same config are
    # comparable and --resume does not redraw the split.
    set_seed(args.seed)

    grid = AecGrid.from_config(cfg)
    model_cfg = AecNetConfig.from_config(cfg, grid.frame_rate)
    loss_cfg = read_loss_config(cfg)

    epochs = cfg.getint('training', 'epochs', fallback=100)
    batch_size = cfg.getint('training', 'batch_size', fallback=8)
    lr = cfg.getfloat('training', 'lr', fallback=1e-3)
    min_lr = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    lr_patience = cfg.getint('training', 'lr_patience', fallback=4)
    patience = cfg.getint('training', 'early_stop_patience', fallback=20)
    steps_per_epoch = cfg.getint('training', 'steps_per_epoch', fallback=0)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=5.0)
    num_workers = cfg.getint('training', 'num_workers', fallback=2)
    prefetch_factor = cfg.getint('training', 'prefetch_factor', fallback=2)
    leak_gate_db = cfg.getfloat('training', 'zero_ref_leak_db_max', fallback=-40.0)
    output_dir = cfg.get('paths', 'output_dir', fallback='output')

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be greater than zero")

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(
            args.device or cfg.get('training', 'device', fallback='cpu'))

    packed_dir = args.packed_dir or cfg.get('paths', 'packed_dir', fallback=None)
    if not packed_dir:
        raise ValueError("--packed-dir or [paths] packed_dir required")
    val_packed_dir = args.val_packed_dir or cfg.get(
        'paths', 'val_packed_dir', fallback=None)

    dataset = PackedAecDataset(packed_dir, expected_sr=grid.sr, mmap=args.mmap)

    contract = build_contract(cfg, grid, model_cfg, loss_cfg)
    resume_ckpt = None
    if args.resume:
        print(f"Resuming: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device,
                                 weights_only=False)
        require_checkpoint_contract(resume_ckpt, contract, context=args.resume)

    # -- the split ------------------------------------------------------
    if val_packed_dir:
        split_mode = 'source_disjoint_corpora'
        train_set = dataset
        val_set = PackedAecDataset(val_packed_dir, expected_sr=grid.sr,
                                   mmap=args.mmap)
        train_indices = val_indices = None
        if resume_ckpt is not None and resume_ckpt.get('train_indices'):
            print("  ⚠ checkpoint stores a within-corpus split but two corpora "
                  "were given; the stored indices are ignored.")
    else:
        split_mode = 'sequence_level_within_corpus'
        if resume_ckpt is not None and resume_ckpt.get('train_indices'):
            train_indices = list(resume_ckpt['train_indices'])
            val_indices = list(resume_ckpt['val_indices'])
            print(f"  restored split from checkpoint: {len(train_indices)} train "
                  f"/ {len(val_indices)} val chunks")
        else:
            train_indices, val_indices = sequence_level_split(dataset, args.seed)
            if resume_ckpt is not None:
                print("  ⚠ checkpoint has no stored split; redrawing from --seed "
                      f"{args.seed} (validation may be contaminated)")
        train_set, val_set = subsets_from_indices(
            dataset, train_indices, val_indices)

    train_lanes = _lanes_for(train_set, batch_size, 'train')
    val_lanes = _lanes_for(val_set, batch_size, 'val')
    train_sampler = sampler_for(train_set, train_lanes, args.seed, shuffle=True)
    # Validation walks in a fixed order so epoch-to-epoch numbers are comparable.
    val_sampler = sampler_for(val_set, val_lanes, args.seed, shuffle=False)

    pin_memory = device.type == 'cuda'
    worker_kwargs = dataloader_worker_kwargs(num_workers, pin_memory,
                                             prefetch_factor)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=aec_collate, **worker_kwargs)
    val_loader = DataLoader(
        val_set, batch_sampler=val_sampler, collate_fn=aec_collate,
        **dataloader_worker_kwargs(min(num_workers, 2), pin_memory,
                                   prefetch_factor))

    model = build_model(cfg, grid).to(device)
    criterion = EchoEstimationLoss(
        grid, model_cfg.compress_exponent, lookahead=model_cfg.lookahead,
        **loss_cfg).to(device)

    # -- banner ---------------------------------------------------------
    print("=" * 68)
    print("AECNet -- stage 1: ECHO ESTIMATION (output is D_hat, not speech)")
    print("=" * 68)
    print(f"grid          : sr={grid.sr} n_fft={grid.n_fft} win={grid.win_len} "
          f"hop={grid.hop_len} -> {grid.n_freqs} bins @ {grid.frame_rate:.4g} fps")
    for line in describe(model, grid.frame_rate):
        print(f"{line}")
    print(f"parameters    : {model.n_parameters():,}")
    print(f"model_version : {MODEL_VERSION}")
    print(f"feature_version: {FEATURE_VERSION}")
    print(f"loss_version  : {LOSS_VERSION}")
    print(f"loss weights  : out={loss_cfg['lambda_out']} "
          f"near={loss_cfg['lambda_near']} idle={loss_cfg['lambda_idle']} "
          f"mag={loss_cfg['mag_weight']}")
    print(f"echo norm     : {loss_cfg['echo_norm'].upper()}"
          f"{'  (MAE -- guards against echo under-estimation)' if loss_cfg['echo_norm'] == 'l1' else '  ⚠ L2 under-estimates the echo'}")
    print(f"idle guard    : {loss_cfg['idle_guard_sec']} s = "
          f"{criterion.guard_frames} frames on this grid")
    print(f"split         : {split_mode}  "
          f"({len(train_set)} train / {len(val_set)} val chunks)")
    print(f"lanes         : train={train_lanes} val={val_lanes}  "
          f"({len(train_sampler)} train batches/epoch)")
    print(f"device        : {device}   seed={args.seed}")
    counts = dataset.scenario_counts()
    print(f"scenarios     : {counts}")
    dropout_share = counts.get('ref_dropout', 0) / max(1, len(dataset))
    if loss_cfg['lambda_idle'] > 0 and dropout_share < 0.01:
        print(f"  ⚠ only {dropout_share:.2%} of chunks are 'ref_dropout'; the "
              f"idle term and the zero-reference gate are being supervised by "
              f"almost nothing. Raise [dropout] ref_dropout_chunks_max in the "
              f"generator config -- longer dropouts, not more sequences.")
    if steps_per_epoch:
        print(f"  ⚠ steps_per_epoch={steps_per_epoch} truncates every epoch, so "
              f"only the FIRST {steps_per_epoch} chunks of each sequence are "
              f"ever seen. Use it to smoke-test, not to train.")
    print("=" * 68)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    no_improve = 0

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['state_dict'])
        if 'optimizer' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if 'scheduler' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler'])
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_loss = resume_ckpt.get('best_val_loss', float('inf'))
        # Without this, early stopping restarts its patience window on every
        # resume and can never fire.
        no_improve = resume_ckpt.get('no_improve', 0)
        print(f"  Resumed epoch {start_epoch - 1}, best={best_val_loss:.5f}, "
              f"no_improve={no_improve}")

    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)
        tr = run_epoch(model, train_loader, criterion, grid, device,
                       optimizer=optimizer, grad_clip=grad_clip,
                       max_steps=steps_per_epoch,
                       desc=f"Epoch {epoch}/{epochs}")
        va = run_epoch(model, val_loader, criterion, grid, device,
                       desc=f"Val {epoch}/{epochs}")

        # The hard gate, measured rather than asserted: a training run should
        # report it every epoch and a shipped checkpoint should be asserted on
        # with model.assert_zero_reference_gate.
        leak_db = _measure_zero_reference_leak(model, val_loader, grid, device)
        gate = 'PASS' if leak_db <= leak_gate_db else 'FAIL'

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={tr['loss']:.4f} "
              f"(echo {tr['echo']:.4f} out {tr['out']:.4f} near {tr['near']:.4f} "
              f"idle {tr['idle']:.5f} over {tr['idle_frames']:.0f} idle frames)  "
              f"val={va['loss']:.4f}  lr={lr_now:.2e}  "
              f"zero-ref leak {leak_db:.1f} dB [{gate}]")
        # ⚠ An idle term of 0.00000 reads like "converged" and usually means
        # "never supervised".  Say which.
        for name, stats in (('training', tr), ('validation', va)):
            if loss_cfg['lambda_idle'] > 0 and stats['idle_frames'] == 0:
                print(f"  ⚠ the {name} pass contained NO idle frames, so the "
                      f"idle term contributed nothing and its 0.00000 means "
                      f"'unsupervised', not 'converged'. Either the corpus has "
                      f"no ref_dropout/near_only chunks, or chunk_sec is "
                      f"shorter than idle_guard_sec "
                      f"({criterion.guard_frames} frames) and the guard "
                      f"consumes every chunk whole.")
        scheduler.step(va['loss'])

        is_best = va['loss'] < best_val_loss
        if is_best:
            best_val_loss = va['loss']
            no_improve = 0
        else:
            no_improve += 1

        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
            'seed': args.seed,
            'split_mode': split_mode,
            # Exact split, so a resume cannot leak trained chunks into val.
            'train_indices': train_indices,
            'val_indices': val_indices,
            'zero_ref_leak_db': leak_db,
            **contract,
        }
        torch.save(ckpt, os.path.join(output_dir, 'aecnet_last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'aecnet_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        elif patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"Training done. Best val loss: {best_val_loss:.5f}")


def _measure_zero_reference_leak(model: AecNet, loader, grid: AecGrid,
                                 device) -> float:
    """Zero-reference leak on the first validation batch, in dB.

    One batch is enough: this measures a property of the network, not of the
    data, and the whole point is that the answer should not depend on which
    microphone signal it is asked about.
    """
    for stems, _ in loader:
        stems = stems.to(device=device, dtype=torch.float32)
        view = AecStems(stems)
        y_spec = stft(view.Y, grid)
        return zero_reference_leak_db(model, y_spec)
    raise RuntimeError("validation loader produced no batches")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AECNet training (echo estimation)')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='directory of packed AEC .pt shards (pack_aec_dataset.py output)')
    parser.add_argument('--val-packed-dir', default=None,
                        help='separate packed corpus for validation. Strongly '
                             'preferred: the generator already produced a '
                             'SOURCE-disjoint split, and using it means the val '
                             'score is not measuring memorised rooms.')
    parser.add_argument('--mmap', action='store_true',
                        help='Memory-map the .pt shards (low RAM, disk-backed)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; also fixes the train/val split.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    train(args)
