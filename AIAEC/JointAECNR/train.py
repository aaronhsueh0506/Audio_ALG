"""JointAECNR training.

Usage:
    python train.py --config config.ini --packed-dir data/aec_packed --gpu 0
    python train.py --config config.ini --packed-dir data/aec_packed \
        --resume output/joint_aecnr_best.pth

Consumes the packed AEC corpus written by ``AIAEC/dataset_gen_aec/pack_aec_dataset.py``
(six stems per chunk, sequences cut into consecutive chunks).  Produces S_hat
directly from (Y, X).
"""

import argparse
import configparser
import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import tqdm

from model import JointAECNR, detach_state, idle_gate_report, reset_state

_AIAEC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AINR = os.path.join(os.path.dirname(_AIAEC), 'ainr')
sys.path.insert(0, _AIAEC)   # dataset_gen_aec: the AEC corpus, owned by AIAEC/
sys.path.insert(0, _AINR)    # dataset_gen: the SHARED loader/split/seed + DSP
# ⚠ AIAEC/ deliberately depends on ainr/dataset_gen and must not fork it.  Two
# things live there that cannot be duplicated: the augmentation DSP the AEC corpus
# reuses (RIR, RT60, biquad, clipping), and the train/val split + seeder that every
# model in the repo shares.  A second copy of the split is how two models being
# compared silently end up trained on different corpora -- see dataset_gen/loader.py.
# The package is named dataset_gen_aec, NOT dataset_gen, because both directories
# sit on this sys.path and a same-named package would shadow whichever came second.
# ⚠ The split, the seeder and the DataLoader worker policy are imported, never
# re-declared.  ``dataset_gen/loader.py`` records what happened the last time
# each trainer carried its own copy: the held-out fraction drifted from 5% to
# 10% and two models that were being compared had silently been trained on
# different corpora.  ``ainr/tests/test_bakeoff_protocol.py`` asserts the
# absence of local re-declarations.
from dataset_gen import (  # noqa: E402
    dataloader_worker_kwargs,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
    subsets_from_indices,
)
from dataset_gen_aec import (  # noqa: E402
    AecGrid,
    AecStems,
    PackedAecDataset,
    SequenceChunkSampler,
    aec_collate,
    alpha_from_tau,
    istft,
    lane_reset_mask,
    stft,
)


# ============================================================
# Checkpoint contract
# ============================================================
#
# Bumped whenever a change makes previously-trained weights invalid or
# meaningless to resume.  Same gate as GTCRN/train.py and
# DeepFilterNet2/train.py; the extra risk here is that this model has THREE
# output heads whose presence is a config switch, and a checkpoint trained with
# the auxiliary heads off loads perfectly into a model built with them on --
# load_state_dict would simply leave the new heads at their random init and
# training would look like it resumed.
MODEL_VERSION = 'joint_aecnr_dualenc_refgated_echo_head_v1'
FEATURE_VERSION = 'compressed_complex_zero_preserving_v1'
LOSS_VERSION = 'joint_spec30_mag70_sisnr_echo_noisepsd_idle_v1'

# ⚠ Derived from the dict that build_contract() returns them from, not restated.
# A restated tuple is how a newly added version string ends up recorded in the
# checkpoint but never actually compared against.
_VERSIONS = {
    'model_version': MODEL_VERSION,
    'feature_version': FEATURE_VERSION,
    'loss_version': LOSS_VERSION,
}
_VERSION_FIELDS = tuple(_VERSIONS)


def build_contract(cfg: configparser.ConfigParser, grid: AecGrid,
                   model: JointAECNR) -> dict:
    """Everything that changes what the weights MEAN.

    The resolved integers are recorded, not the seconds they came from: a
    checkpoint trained at 16 kHz with ``ref_context_sec = 0.25`` must not
    silently resume on a 48 kHz grid where the same 0.25 s is a different
    number of taps -- and the tap count is what the weight shapes encode.
    """
    return {
        **_VERSIONS,
        'sr': grid.sr,
        'n_fft': grid.n_fft,
        'win_len': grid.win_len,
        'hop_len': grid.hop_len,
        'center': cfg.getboolean('signal', 'center', fallback=False),
        'enc_channels': model.enc_channels,
        'enc_stages': model.enc_stages,
        'rnn_hidden': model.rnn_hidden,
        'rnn_layers': model.rnn_layers,
        'lookahead_frames': model.lookahead_frames,
        'ref_context_frames': model.ref_context_frames,
        'echo_gate_memory_frames': model.echo_gate_memory_frames,
        'compress_exponent': model.compress_exponent,
        'mask_max': model.mask_max,
        'aux_echo_head': model.aux_echo_head,
        'echo_head_ref_gated': model.echo_head_ref_gated,
        'aux_noise_psd_head': model.aux_noise_psd_head,
        'use_deep_filter': model.use_deep_filter,
        'df_bins': model.df_bins,
        'df_order': model.df_order,
        'df_lookahead': model.df_lookahead,
    }


def require_checkpoint_contract(ckpt: dict, contract: dict,
                                context: str = 'checkpoint',
                                allow_missing: bool = False) -> None:
    """Refuse to load a checkpoint across a semantic change.

    ``allow_missing=True`` exempts a checkpoint that records no contract at all
    (there are none today, but inference must be able to accept a hand-made
    state_dict without pretending it verified anything).
    """
    if allow_missing and not any(key in ckpt for key in _VERSION_FIELDS):
        return
    for key in _VERSION_FIELDS:
        got = ckpt.get(key)
        if got != contract[key]:
            shown = repr(got) if got is not None else 'missing (pre-contract checkpoint)'
            raise ValueError(
                f"{context} {key}={shown}, expected {contract[key]!r}. "
                f"Resuming across this change would train on incompatible "
                f"weights; start a fresh run instead."
            )
    # Past this point the checkpoint is known to carry a contract, so every
    # field must be PRESENT and match.  "compare only if present" would let a
    # checkpoint that recorded nothing but the version strings satisfy the whole
    # gate -- and the aux-head switches are exactly the fields that leave weight
    # shapes intact while changing what the model is.
    for key in contract:
        if key in _VERSION_FIELDS:
            continue
        if key not in ckpt:
            raise ValueError(
                f"{context} is missing contract field {key!r} (expected "
                f"{contract[key]!r}); it predates this field being recorded, so "
                f"its value cannot be verified -- start a fresh run."
            )
        if ckpt[key] != contract[key]:
            raise ValueError(
                f"{context} {key}={ckpt[key]!r}, but config requires "
                f"{contract[key]!r}."
            )


# ============================================================
# Loss
# ============================================================

_EPS = 1e-12


def _magnitude(spec: torch.Tensor) -> torch.Tensor:
    """``|spec|`` with the floor INSIDE the square root.

    ⚠ Not ``spec.abs()``, and the reason is the COMPRESSION that follows it, not
    the abs itself (``abs()`` is well behaved: torch defines its gradient at
    zero as 0).  Every use of this value is raised to a fractional power --
    ``pow(gamma)`` with gamma = 0.3 in the magnitude term, ``pow(gamma - 1)`` in
    the complex term -- and ``d/dx x^0.3`` at x = 0 is INFINITE.  One infinite
    gradient times a zero upstream gradient is NaN, and it reaches every
    parameter through the shared trunk on the first optimizer step.

    THIS MODEL PRODUCES EXACT ZEROS BY DESIGN: the reference-gated echo head
    emits literally 0.0 on every ref-idle frame, and the corpus is full of them
    (``ref_dropout``, ``near_only``).  So this is a first-step killer, not a
    rare edge -- ``tests/test_train_contract.py`` pins it.  Same construction as
    GTCRN's HybridLoss, which floors inside the sqrt for the same reason.
    """
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + _EPS)


def _power(spec: torch.Tensor) -> torch.Tensor:
    """``|spec|^2`` without going through ``abs()``; see :func:`_magnitude`."""
    return spec.real.pow(2) + spec.imag.pow(2)


def _compress(spec: torch.Tensor, gamma: float) -> torch.Tensor:
    """Magnitude-compressed complex spectrum, phase untouched."""
    return spec * _magnitude(spec).pow(gamma - 1.0)


def waveform_for_loss(spec: torch.Tensor, grid: AecGrid) -> torch.Tensor:
    """Time-domain view of a spectrum, for comparing two spectra only.

    ⚠ ``center=True`` on the INVERSE regardless of how the analysis was done,
    and that is not a typo.  With ``center=False`` analysis, the first and last
    half-window of a WOLA reconstruction receive a contribution from exactly one
    frame, and sqrt-Hann is zero at its edges -- the overlap-add envelope
    touches zero there and ``torch.istft`` refuses to invert it (its NOLA
    check).  ``center=True`` tells the inverse to drop those two incomplete
    half-windows and return the interior, which is the only part that was ever
    reconstructible.

    The prediction and the target go through this identical call, so the SI-SNR
    term compares the same span of both and the dropped edges cannot bias it.
    ⚠ Do NOT reuse this to write audio: it is shorter than the input and offset
    from it.  ``denoise.reconstruct`` is the function for that.
    """
    return istft(spec, grid, center=True)


def si_snr(pred: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR in the log domain.  ``(B, T)`` -> ``(B,)``."""
    projection = (
        (pred * target).sum(dim=-1, keepdim=True)
        / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
        * target
    )
    residual = pred - projection
    return torch.log10(
        projection.pow(2).sum(dim=-1) / (residual.pow(2).sum(dim=-1) + eps) + eps
    )


def causal_ema(power: torch.Tensor, alpha: float) -> torch.Tensor:
    """One-pole smoothing along the LAST axis (frames).

    ⚠ Used only to build a TARGET, so it runs under no_grad and the python loop
    costs nothing that matters.  ``alpha`` must come from
    ``dataset_gen_aec.alpha_from_tau`` -- a literal here would be a
    frame-rate-dependent constant in disguise and the 48 kHz variant would
    smooth over a different physical duration.
    """
    with torch.no_grad():
        out = torch.empty_like(power)
        running = power[..., 0]
        out[..., 0] = running
        for frame in range(1, power.shape[-1]):
            running = alpha * running + (1.0 - alpha) * power[..., frame]
            out[..., frame] = running
        return out


class JointLoss(nn.Module):
    """Main objective on S_hat vs S, plus the two auxiliary supervisions.

    ⚠ The auxiliary terms are not regularisers, they are the attribution
    mechanism.  Setting ``echo_weight = 0`` leaves ``aux_echo_head`` present but
    untrained, which is worse than switching it off: it still emits a D_hat, and
    that D_hat means nothing.
    """

    def __init__(self, grid: AecGrid, cfg: configparser.ConfigParser):
        super().__init__()
        self.grid = grid
        self.spec_weight = cfg.getfloat('loss', 'spec_weight')
        self.mag_weight = cfg.getfloat('loss', 'mag_weight')
        self.gamma = cfg.getfloat('loss', 'compress_gamma')
        self.sisnr_weight = cfg.getfloat('loss', 'sisnr_weight')
        self.echo_weight = cfg.getfloat('loss', 'echo_weight')
        self.noise_psd_weight = cfg.getfloat('loss', 'noise_psd_weight')
        self.idle_weight = cfg.getfloat('loss', 'idle_weight')
        self.noise_psd_alpha = alpha_from_tau(
            cfg.getfloat('loss', 'noise_psd_tau_sec'), grid.hop_len, grid.sr)
        self.noise_psd_floor = 10.0 ** (
            cfg.getfloat('loss', 'noise_psd_floor_db') / 10.0)

    def spectral(self, pred: torch.Tensor, target: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_c, target_c = _compress(pred, self.gamma), _compress(target, self.gamma)
        spec = (F.mse_loss(pred_c.real, target_c.real)
                + F.mse_loss(pred_c.imag, target_c.imag))
        mag = F.mse_loss(_magnitude(pred).pow(self.gamma),
                         _magnitude(target).pow(self.gamma))
        return spec, mag

    def forward(self, outputs, targets: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # ⚠ Every value put into `parts` is detached.  It is a reporting number;
        # keeping the graph alive for it would pin a whole batch's activations
        # for as long as the running-average dict lives.
        speech = targets['near_speech']
        spec, mag = self.spectral(outputs.speech_spec, speech)
        total = self.spec_weight * spec + self.mag_weight * mag
        parts = {'spec': spec.item(), 'mag': mag.item()}

        if self.sisnr_weight:
            # ⚠ Both spectra are ISTFT'd here by the SAME call.  Comparing an
            # ISTFT'd prediction against a RAW waveform puts a floor under this
            # term that no model can reach, because WOLA does not reconstruct
            # the edge frames exactly (GTCRN/train.py hit this).
            pred_wav = waveform_for_loss(outputs.speech_spec, self.grid)
            true_wav = waveform_for_loss(speech, self.grid)
            sisnr = -si_snr(pred_wav, true_wav).mean()
            total = total + self.sisnr_weight * sisnr
            parts['sisnr'] = sisnr.item()

        if outputs.echo_spec is not None and self.echo_weight:
            echo_spec, echo_mag = self.spectral(outputs.echo_spec, targets['echo'])
            echo = echo_spec + echo_mag
            total = total + self.echo_weight * echo
            parts['echo'] = echo.item()

            if self.idle_weight:
                # Ref-idle frames: the activity gate is below 0.5 exactly when
                # the reference power (max over the gate's memory) is under the
                # configured silence floor.
                #
                # ⚠ This is the SAME error as the echo term, reweighted -- not a
                # "D_hat must be zero" term.  On ref_dropout and near_only
                # chunks the corpus really does have D == 0 (the generator cuts
                # X and D together), so there it does mean "emit nothing".  In
                # an ordinary gap between far-end bursts the true echo is a
                # decaying tail, and a term that demanded zero there would be
                # training the model to abandon the tail it is supposed to
                # cancel.
                idle = (outputs.ref_gate < 0.5).unsqueeze(1)
                if idle.any():
                    error = _power(outputs.echo_spec - targets['echo'])
                    idle_loss = (error * idle).sum() / idle.expand_as(error).sum()
                    total = total + self.idle_weight * idle_loss
                    parts['idle'] = idle_loss.item()

        if outputs.noise_log_psd is not None and self.noise_psd_weight:
            power = _power(targets['local_noise'])
            smoothed = causal_ema(power, self.noise_psd_alpha)
            target_log = torch.log10(smoothed.clamp_min(self.noise_psd_floor))
            psd = F.l1_loss(outputs.noise_log_psd, target_log)
            total = total + self.noise_psd_weight * psd
            parts['noise_psd'] = psd.item()

        parts['total'] = total.item()
        return total, parts


# ============================================================
# Sequence-level split
# ============================================================

def sequence_level_split(dataset: PackedAecDataset, seed: int
                         ) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Hold out whole SEQUENCES, not chunks.

    ⚠ A chunk-level split is wrong here twice over.  It leaks -- chunks 0..4 of
    a sequence would train while chunk 5, the same talker in the same room two
    seconds later, is validated on.  And it is not even representable:
    ``SequenceChunkSampler`` requires each sequence's ``chunk_index`` to be a
    complete ``0..n-1`` run and refuses a subset that breaks it.

    The permutation still comes from the shared
    ``locality_preserving_random_split``, applied to the sequence-id list
    instead of the chunk list, so the split is drawn by exactly one
    implementation and the held-out fraction is the shared one.
    """
    all_sequences = dataset.sequence_ids()
    unique = sorted(set(all_sequences))
    n_train, n_val = split_sizes(unique)
    train_subset, val_subset = locality_preserving_random_split(
        unique, n_train, n_val, seed)
    train_sequences = [unique[i] for i in train_subset.indices]
    val_sequences = [unique[i] for i in val_subset.indices]

    train_set, val_set = set(train_sequences), set(val_sequences)
    train_indices = [i for i, s in enumerate(all_sequences) if s in train_set]
    val_indices = [i for i, s in enumerate(all_sequences) if s in val_set]
    return train_indices, val_indices, train_sequences, val_sequences


def make_sequence_loader(dataset: PackedAecDataset, indices: Sequence[int],
                         n_lanes: int, seed: int, shuffle: bool,
                         worker_kwargs: dict
                         ) -> Tuple[DataLoader, SequenceChunkSampler]:
    """DataLoader whose batch lanes each walk one sequence in order."""
    sequence_ids = dataset.sequence_ids()
    chunk_indices = dataset.chunk_indices()
    try:
        sampler = SequenceChunkSampler(
            [sequence_ids[i] for i in indices],
            [chunk_indices[i] for i in indices],
            n_lanes=n_lanes, shuffle=shuffle, seed=seed, drop_last=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}\n(batch_size is the number of sequence LANES for this "
            f"model, so it cannot exceed the number of sequences in the split)"
        ) from None
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_sampler=sampler, collate_fn=aec_collate,
                        **worker_kwargs)
    return loader, sampler


# ============================================================
# One pass over a split
# ============================================================

def stem_spectra(stems: AecStems, grid: AecGrid, center: bool
                 ) -> Dict[str, torch.Tensor]:
    """All six stems transformed in one call, keyed by the DECLARED order."""
    spec = stft(stems.as_tensor(), grid, center=center)   # (B, 6, F, T)
    return {name: spec[:, i] for i, name in enumerate(stems.order)}


def run_epoch(model: JointAECNR, loader: DataLoader,
              sampler: SequenceChunkSampler, criterion: JointLoss,
              grid: AecGrid, center: bool, device: torch.device,
              optimizer=None, grad_clip: float = 5.0, max_steps: int = 0,
              desc: str = '') -> Dict[str, float]:
    """Walk the split with per-lane recurrent state carried across batches."""
    training = optimizer is not None
    model.train(training)
    n_lanes = sampler.n_lanes
    state = model.init_state(n_lanes, device=device)
    totals: Dict[str, float] = {}
    steps = 0

    progress = tqdm.tqdm(loader, desc=desc, total=(
        min(len(sampler), max_steps) if max_steps else len(sampler)))
    for stems_batch, metas in progress:
        stems = AecStems(stems_batch.to(device=device, dtype=torch.float32))
        specs = stem_spectra(stems, grid, center)

        # ⚠ Detach BEFORE the step, not after: truncated BPTT means the graph
        # must not span batches, and the state a lane receives is the previous
        # batch's output.  Then zero the lanes that have moved on to a new
        # sequence -- carrying one talker's adaptation into the next sequence
        # would make cold-start convergence look better than it is.
        reset = lane_reset_mask([m['chunk_index'] for m in metas]).to(device)
        state = reset_state(detach_state(state), reset)

        with torch.set_grad_enabled(training):
            outputs, state = model(specs['mic_postclip'], specs['far_render'],
                                   state)
            loss, parts = criterion(outputs, specs)

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        progress.set_postfix(loss=f"{parts['total']:.4f}")
        if max_steps and steps >= max_steps:
            break
    progress.close()

    if steps == 0:
        raise RuntimeError(
            "the split produced no batches; with drop_last=True every lane "
            "needs at least one sequence, so batch_size may exceed the number "
            "of sequences available")
    return {key: value / steps for key, value in totals.items()}


# ============================================================
# Banner
# ============================================================

def print_banner(grid, model, dataset, train_indices, val_indices,
                 train_sequences, val_sequences, train_sampler, val_sampler,
                 device, args, contract):
    def ms(frames):
        return frames * grid.hop_len / grid.sr * 1000.0

    n_params = sum(p.numel() for p in model.parameters())
    ref_params = sum(p.numel() for p in model.reference_pathway_parameters())

    print("=" * 68)
    print("JointAECNR  --  joint AEC + RES + NR   (in: Y, X   out: S_hat)")
    print("=" * 68)
    print(f"  model_version   : {MODEL_VERSION}")
    print(f"  feature_version : {FEATURE_VERSION}")
    print(f"  loss_version    : {LOSS_VERSION}")
    print("-" * 68)
    print(f"  grid            : sr={grid.sr}  n_fft={grid.n_fft}  "
          f"win={grid.win_len}  hop={grid.hop_len}")
    print(f"                    n_freqs={grid.n_freqs}  "
          f"frame_rate={grid.frame_rate:.4g} fps  center={contract['center']}")
    print(f"  parameters      : {n_params:,}  "
          f"({ref_params:,} in the reference pathway)")
    print(f"  encoder         : {model.enc_channels} ch x {model.enc_stages} "
          f"stages, freqs {model.freqs}")
    print(f"  bottleneck      : GRU {model.rnn_hidden} x {model.rnn_layers}")
    print(f"  lookahead       : {model.lookahead_frames} frames "
          f"({ms(model.lookahead_frames):.1f} ms)")
    print(f"  ref context     : {model.ref_context_frames} frames "
          f"({ms(model.ref_context_frames):.1f} ms)   "
          f"ref receptive field {model.reference_receptive_field_frames} frames")
    print(f"  echo gate       : {model.echo_gate_memory_frames} frames "
          f"({ms(model.echo_gate_memory_frames):.0f} ms) memory, "
          f"gated={model.echo_head_ref_gated}")
    if model.use_deep_filter:
        print(f"  deep filter     : {model.df_bins} bins "
              f"(<= {model.df_bins * grid.sr / grid.n_fft:.0f} Hz), order "
              f"{model.df_order} ({ms(model.df_order):.0f} ms), lookahead "
              f"{model.df_lookahead} ({ms(model.df_lookahead):.1f} ms)")
    else:
        print("  deep filter     : OFF")
    total_delay = ms(model.lookahead_frames + model.df_lookahead)
    print(f"  algorithmic delay: {total_delay:.1f} ms "
          f"(model lookahead + deep-filter lookahead)")
    print("-" * 68)
    print(f"  aux_echo_head     : {model.aux_echo_head}")
    print(f"  aux_noise_psd_head: {model.aux_noise_psd_head}")
    if not (model.aux_echo_head and model.aux_noise_psd_head):
        print("  ⚠ an auxiliary head is OFF.  They are what make this joint "
              "model comparable")
        print("    against a #4 + #3 cascade; without them a bad score cannot "
              "be split into")
        print("    'failed to cancel' and 'failed to suppress', and there is no "
              "PSD reference")
        print("    for a downstream comfort-noise generator.")
    print("-" * 68)
    print(f"  corpus          : {len(dataset)} chunks / "
          f"{dataset.n_sequences()} sequences, T={dataset.chunk_samples} "
          f"({dataset.chunk_samples / grid.sr:.2f} s)")
    print(f"  split           : {len(train_sequences)} train / "
          f"{len(val_sequences)} val SEQUENCES  "
          f"({len(train_indices)} / {len(val_indices)} chunks)")
    print(f"  scenarios       : {dataset.scenario_counts()}")
    print(f"  lanes           : {train_sampler.n_lanes} train / "
          f"{val_sampler.n_lanes} val "
          f"(a lane = one sequence walked in order, state carried)")
    # ⚠ Printed, not hidden.  Every lane is the same width, so the sampler stops
    # when its SHORTEST lane runs out and the tail chunks of the longer lanes
    # are never seen.  The validation schedule is fixed (shuffle=False), so the
    # val number is comparable across epochs and across models -- but it is a
    # number over this many chunks, not over the whole split.
    scored = len(val_sampler) * val_sampler.n_lanes
    print(f"  val coverage    : {scored}/{len(val_indices)} chunks scored "
          f"per epoch ({100.0 * scored / max(1, len(val_indices)):.0f}%)")
    print(f"  device          : {device}   seed: {args.seed}")
    print("=" * 68)


# ============================================================
# Train
# ============================================================

def train(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    # Seed before anything that draws randomness; the split and the lane layout
    # both depend on it, so two runs of one config must agree.
    set_seed(args.seed)

    grid = AecGrid.from_config(cfg)
    center = cfg.getboolean('signal', 'center', fallback=False)

    epochs = cfg.getint('training', 'epochs')
    batch_size = cfg.getint('training', 'batch_size')
    lr = cfg.getfloat('training', 'lr')
    min_lr = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    lr_patience = cfg.getint('training', 'lr_patience', fallback=4)
    patience = cfg.getint('training', 'early_stop_patience', fallback=20)
    epoch_size = cfg.getint('training', 'epoch_size', fallback=0)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=5.0)
    num_workers = cfg.getint('training', 'num_workers', fallback=2)
    prefetch_factor = cfg.getint('training', 'prefetch_factor', fallback=2)
    output_dir = cfg.get('paths', 'output_dir', fallback='output')

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(
            args.device or cfg.get('training', 'device', fallback='cpu'))

    packed_dir = args.packed_dir or cfg.get('paths', 'packed_dir', fallback=None)
    if not packed_dir:
        raise ValueError("--packed-dir or [paths] packed_dir required")

    mmap = args.mmap or cfg.getboolean('training', 'mmap', fallback=False)
    dataset = PackedAecDataset(packed_dir, expected_sr=grid.sr, mmap=mmap)

    model = JointAECNR.from_config(cfg, grid).to(device)
    contract = build_contract(cfg, grid, model)

    resume_ckpt = None
    if args.resume:
        print(f"Resuming: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device,
                                 weights_only=False)
        require_checkpoint_contract(resume_ckpt, contract, context=args.resume)

    if resume_ckpt is not None and 'train_indices' in resume_ckpt:
        train_indices = list(resume_ckpt['train_indices'])
        val_indices = list(resume_ckpt['val_indices'])
        train_sequences = list(resume_ckpt.get('train_sequences', []))
        val_sequences = list(resume_ckpt.get('val_sequences', []))
        # Rebuilt only to prove the recorded chunk indices still address the
        # same rows; the loaders below index the dataset directly.
        subsets_from_indices(dataset, train_indices, val_indices)
        print(f"  restored split from checkpoint: {len(train_indices)} train / "
              f"{len(val_indices)} val chunks")
    else:
        (train_indices, val_indices,
         train_sequences, val_sequences) = sequence_level_split(dataset, args.seed)
        if resume_ckpt is not None:
            print("  ⚠ checkpoint has no stored split; redrawing from --seed "
                  f"{args.seed} (validation may be contaminated)")

    pin_memory = device.type == 'cuda'
    worker_kwargs = dataloader_worker_kwargs(num_workers, pin_memory,
                                             prefetch_factor)
    train_loader, train_sampler = make_sequence_loader(
        dataset, train_indices, batch_size, args.seed, True, worker_kwargs)
    # ⚠ The validation split holds a small fraction of the SEQUENCES, and every
    # lane needs one of its own -- so validation runs with as many lanes as it
    # has sequences, up to batch_size.  It is not the same width as training,
    # and it does not need to be: nothing is being back-propagated.
    val_lanes = min(batch_size, len(val_sequences))
    val_loader, val_sampler = make_sequence_loader(
        dataset, val_indices, val_lanes, args.seed, False,
        dataloader_worker_kwargs(min(num_workers, 2), pin_memory,
                                 prefetch_factor))

    criterion = JointLoss(grid, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)

    print_banner(grid, model, dataset, train_indices, val_indices,
                 train_sequences, val_sequences, train_sampler, val_sampler,
                 device, args, contract)

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
        # Without restoring this, early stopping restarts its patience window on
        # every resume and can never fire.
        no_improve = resume_ckpt.get('no_improve', 0)
        print(f"  Resumed epoch {start_epoch - 1}, "
              f"best_val_loss={best_val_loss:.5f}, no_improve={no_improve}")

    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)
        train_parts = run_epoch(
            model, train_loader, train_sampler, criterion, grid, center, device,
            optimizer=optimizer, grad_clip=grad_clip, max_steps=epoch_size,
            desc=f"Epoch {epoch}/{epochs}")
        val_parts = run_epoch(
            model, val_loader, val_sampler, criterion, grid, center, device,
            desc=f"  val {epoch}")

        val_loss = val_parts['total']
        lr_now = optimizer.param_groups[0]['lr']
        detail = '  '.join(f"{k}={v:.4f}" for k, v in sorted(val_parts.items())
                           if k != 'total')
        print(f"Epoch {epoch}: train={train_parts['total']:.4f}  "
              f"val={val_loss:.4f}  lr={lr_now:.2e}")
        print(f"           val parts: {detail}")
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
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
            'config': dict(cfg['signal']),
            # The exact split, so a resume cannot leak trained sequences into
            # validation.  Sequence ids are stored alongside the chunk indices
            # because the chunk indices are only meaningful for THIS corpus.
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'seed': args.seed,
            **contract,
        }
        torch.save(ckpt, os.path.join(output_dir, 'joint_aecnr_last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'joint_aecnr_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        elif patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # The hard gate, reported once at the end on a silent-reference probe built
    # from the validation corpus.  ⚠ It is a REPORT, not a pass/fail: what
    # counts as passing depends on how noisy the probe mic is, and this model is
    # supposed to change a noisy mic.
    probe_stems = AecStems(dataset[val_indices[0]][0].unsqueeze(0).to(device))
    probe = stft(probe_stems.near_speech, grid, center=center)
    print(f"Idle gate (near-speech-only probe, X == 0): "
          f"{idle_gate_report(model, probe)}")
    print(f"Training done. Best val loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JointAECNR training')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='directory of .pt shards from pack_aec_dataset.py')
    parser.add_argument('--mmap', action='store_true',
                        help='memory-map the shards (low RAM, disk-backed)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; also fixes the sequence-level split. '
                             'Must match the other AEC models for a comparable '
                             'run.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    train(parser.parse_args())
