"""PostFilter training: joint residual-echo + noise suppression.

Usage:
    python train.py --config config.ini --packed-dir data/aec_packed --gpu 0
    python train.py --config config.ini --packed-dir data/aec_packed \
        --resume output/postfilter_best.pth

Consumes the packed AEC corpus (dataset_gen/aec/pack_aec_dataset.py).  From each
chunk it derives

    Y = mic_postclip     X = far_render      D = echo      S = near_speech

runs the FROZEN front-end to get ``(E, D_hat)``, and trains the mask to turn E
into S.

⚠ THREE THINGS IN THIS FILE ARE LOAD-BEARING AND EASY TO BREAK SILENTLY

1. State carries along a lane.  ``SequenceChunkSampler`` puts consecutive chunks
   of ONE sequence in lane k of consecutive batches; the GRU state, the feature
   EMAs, the front-end's adaptive filter and the STFT overlap tail all persist
   along that lane and are reset only where ``chunk_index == 0``.  A trainer that
   reinitialises per chunk still converges -- it just cannot show convergence,
   echo-path-change recovery or long-term drift, which is what the 20-60 s
   sequences exist for.

2. ``center=False`` with an explicit overlap tail.  ``center=True`` pads half a
   window of zeros at both ends of every chunk, which puts a fabricated
   discontinuity into the middle of a sequence every four seconds and makes the
   front-end re-converge across it.

3. The front-end is frozen and its identity is gated.  See frontends.py.
"""

import argparse
import configparser
import itertools
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import tqdm

from frontends import build_frontend
from model import build_model, mask_magnitude

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The split, the seeder and the worker kwargs are shared by every model in
# ainr/ -- see dataset_gen/loader.py for what happened the last time each
# trainer carried its own copy (5% held out on one side, 10% on the other).
from dataset_gen import (  # noqa: E402
    dataloader_worker_kwargs,
    locality_preserving_random_split,
    set_seed,
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
# meaningless to resume.
#
# MODEL_VERSION   architecture and what the mask multiplies.
# FEATURE_VERSION what the network sees.  ⚠ The scale-invariant normalisation is
#                 the whole point of this model; a change to it is not a tweak.
# LOSS_VERSION    what "better" means.
MODEL_VERSION = 'postfilter_erb_gain_convgru_v1'
FEATURE_VERSION = 'ref_normalised_ratio_coherence_contrast_v1'
LOSS_VERSION = 'gamma_mag_complex_sisdr_scenario_weighted_v1'

# ⚠ ONE dict.  _VERSION_FIELDS is derived from it and build_contract() splats it,
# so adding a version string here automatically makes it gated AND recorded.
# Restating the names in a second tuple is how a version ends up recorded but
# never checked.
_VERSIONS = {
    'model_version': MODEL_VERSION,
    'feature_version': FEATURE_VERSION,
    'loss_version': LOSS_VERSION,
}
_VERSION_FIELDS = tuple(_VERSIONS)

# Loss settings live in the contract too, so "resume" cannot quietly mean
# "continue the same weights under a different objective".  denoise.py passes
# require_loss=False because inference does not care.
_LOSS_FIELDS = ('gamma', 'mag_weight', 'complex_weight', 'sisdr_weight',
                'echo_leak_weight', 'idle_weight', 'dt_weight')


def build_contract(cfg, grid: AecGrid, model) -> dict:
    """Everything that changes the meaning of the weights.

    ``enc_downsamples`` is taken from the BUILT model rather than the config
    because the config may say ``auto``; two configs that both say ``auto`` but
    differ in ``n_bands`` resolve to different encoders, and the contract has to
    see the resolved value.
    """
    return {
        **_VERSIONS,
        'sr': grid.sr,
        'n_fft': grid.n_fft,
        'win_len': grid.win_len,
        'hop_len': grid.hop_len,
        'mask_resolution': model.mask_resolution,
        'output_type': model.output_type,
        'n_bands': model.n_out,
        'use_reference': bool(model.features.use_reference),
        'use_mic': bool(model.features.use_mic),
        'include_absolute_level': bool(model.features.include_absolute_level),
        'include_absolute_level': bool(model.features.include_absolute_level),
        'coherence_tau_sec': model.features.coherence_tau_sec,
        'level_tau_sec': model.features.level_tau_sec,
        'enc_channels': model.enc_channels,
        'enc_kernel_t': model.enc_kernel_t,
        'enc_kernel_f': model.enc_kernel_f,
        'enc_downsamples': model.n_down,
        'gru_hidden': model.gru_hidden,
        'gru_layers': model.gru_layers,
        'dec_hidden': model.dec_hidden,
        'lookahead_frames': model.lookahead_frames,
        'gamma': cfg.getfloat('loss', 'gamma', fallback=0.3),
        'mag_weight': cfg.getfloat('loss', 'mag_weight', fallback=500.0),
        'complex_weight': cfg.getfloat('loss', 'complex_weight', fallback=500.0),
        'sisdr_weight': cfg.getfloat('loss', 'sisdr_weight', fallback=1.0),
        'echo_leak_weight': cfg.getfloat('loss', 'echo_leak_weight', fallback=0.0),
        'idle_weight': cfg.getfloat('loss', 'idle_weight', fallback=3.0),
        'dt_weight': cfg.getfloat('loss', 'dt_weight', fallback=1.0),
    }


def require_checkpoint_contract(ckpt, contract, context='checkpoint',
                                require_loss=True):
    """Refuse to reuse a checkpoint across a semantic change."""
    for key in _VERSION_FIELDS:
        got = ckpt.get(key)
        if got != contract[key]:
            shown = repr(got) if got is not None else 'missing (pre-contract checkpoint)'
            raise ValueError(
                f"{context} {key}={shown}, expected {contract[key]!r}. "
                f"Reusing these weights would train or run against an "
                f"incompatible definition; start a fresh run instead.")
    for key, want in contract.items():
        if key in _VERSION_FIELDS:
            continue
        if not require_loss and key in _LOSS_FIELDS:
            continue
        if key not in ckpt:
            raise ValueError(
                f"{context} is missing contract field {key!r} (expected "
                f"{want!r}); it predates this field being recorded, so its "
                f"value cannot be verified -- start a fresh run.")
        got = ckpt[key]
        if isinstance(want, float):
            matches = math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-12)
        else:
            matches = got == want
        if not matches:
            raise ValueError(
                f"{context} {key}={got!r}, but this config requires {want!r}.")


def require_frontend_match(ckpt, frontend_id, allow_change, context='checkpoint'):
    """Gate on the identity of the FROZEN upstream stage.

    ⚠ A checkpoint trained behind one front-end and attached to another is a
    valid out-of-distribution experiment and NOT a valid result.  Passing
    ``--allow-frontend-change`` permits it and appends to a ``frontend_history``
    that every later checkpoint inherits, so no descendant can later be mistaken
    for a matched run.
    """
    stored = ckpt.get('frontend_id')
    history = list(ckpt.get('frontend_history', []))
    if stored == frontend_id:
        return history
    if not allow_change:
        shown = repr(stored) if stored is not None else 'missing'
        raise ValueError(
            f"{context} frontend_id={shown}, but this config builds "
            f"{frontend_id!r}. (E, D_hat) from a different front-end have a "
            f"different residual distribution, so these weights were not "
            f"trained for this input. Pass --allow-frontend-change to run it "
            f"anyway as an OOD experiment -- the result is not comparable.")
    print(f"  ⚠ FRONT-END CHANGED: {stored!r} -> {frontend_id!r}")
    print("    This run is an OUT-OF-DISTRIBUTION experiment. Every checkpoint")
    print("    it produces carries frontend_history and must not be reported")
    print("    as a matched-front-end result.")
    return history + [stored]


# ============================================================
# Loss
# ============================================================

# Chunks where the far end is silent: nothing to cancel, so any suppression is
# pure regression.  ⚠ 'ref_dropout' is the only place the TRANSITION into and
# out of idle appears, and it is ~2.5% of chunks -- see the generator config.
IDLE_SCENARIOS = ('ref_dropout', 'near_only')
DOUBLE_TALK_SCENARIOS = ('double_talk',)


def scenario_weights(meta, idle_weight, dt_weight, device):
    """Per-lane loss multipliers from the chunk metadata.

    ⚠ This reweights the corpus.  Two runs with different weights are not the
    same experiment, which is why both numbers are in the contract.
    """
    weights = []
    for entry in meta:
        scenario = entry.get('scenario')
        if scenario in IDLE_SCENARIOS:
            weights.append(idle_weight)
        elif scenario in DOUBLE_TALK_SCENARIOS:
            weights.append(dt_weight)
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _compress(spec, gamma):
    """``|x|^gamma * x/|x|`` -- magnitude compression that keeps the phase."""
    magnitude = spec.abs().clamp_min(1e-12)
    return (magnitude ** gamma) * (spec / magnitude)


def si_sdr(pred, target, eps=1e-8):
    """Scale-invariant SDR in dB, per lane.  Shapes ``(B, L)`` -> ``(B,)``."""
    alpha = ((pred * target).sum(-1, keepdim=True)
             / (target.square().sum(-1, keepdim=True) + eps))
    projection = alpha * target
    noise = pred - projection
    return 10.0 * torch.log10(
        projection.square().sum(-1) / (noise.square().sum(-1) + eps) + eps)


class PostFilterLoss(nn.Module):
    """Compressed magnitude + complex MSE, plus SI-SDR, plus an optional leak term.

    ⚠ SI-SDR compares the ISTFT of the PREDICTED spectrum against the ISTFT of
    the TARGET spectrum -- never against the raw waveform.  sqrt-Hann WOLA does
    not reconstruct the first and last half-frame exactly, so comparing against
    the raw waveform puts a floor on the term that the model can never reach.

    ⚠ SI-SDR is skipped on lanes whose target is silent.  ``far_only`` chunks
    have S == 0, where SI-SDR is a constant with zero gradient that would still
    dominate the printed loss and make two runs with different scenario mixes
    look different for no reason.  The spectral terms supervise those lanes.
    """

    def __init__(self, grid: AecGrid, gamma=0.3, mag_weight=500.0,
                 complex_weight=500.0, sisdr_weight=1.0, echo_leak_weight=0.0,
                 silence_floor=1e-10):
        super().__init__()
        self.grid = grid
        self.gamma = float(gamma)
        self.mag_weight = float(mag_weight)
        self.complex_weight = float(complex_weight)
        self.sisdr_weight = float(sisdr_weight)
        self.echo_leak_weight = float(echo_leak_weight)
        self.silence_floor = float(silence_floor)
        # .clone(): sqrt_hann_window() hands back a CACHED tensor, and a buffer
        # is a handle a caller could mutate in place.
        self.register_buffer('window', grid.window().clone())

    def _istft(self, spec):
        """Overlap-add with the incomplete first/last half-window dropped.

        ⚠ ``center=True`` here does NOT contradict the ``center=False`` analysis
        in stream_stft.  On this grid it means exactly "overlap-add, then trim
        win_len/2 from each end", and the trim is required: sqrt-Hann (periodic)
        has ``w[0] == 0``, so over the first hop the synthesis envelope is zero
        and a strict inverse is undefined there -- torch refuses it with a NOLA
        error.  The interior reconstructs to 7e-7.

        Both the prediction and the target go through this identical operator,
        which is what keeps the SI-SDR term free of a reconstruction floor.
        """
        return torch.istft(
            spec, n_fft=self.grid.n_fft, hop_length=self.grid.hop_len,
            win_length=self.grid.win_len, window=self.window, center=True)

    def forward(self, pred_spec, target_spec, mask_bins=None, residual_spec=None,
                weights=None):
        pred_c = _compress(pred_spec, self.gamma)
        target_c = _compress(target_spec, self.gamma)

        # Per-lane means, so the scenario weighting below is a weighting of
        # LANES and not of however many frames each happened to contribute.
        per_lane = (self.mag_weight
                    * (pred_c.abs() - target_c.abs()).square().mean(dim=(-2, -1)))
        per_lane = per_lane + self.complex_weight * (
            (pred_c - target_c).abs().square().mean(dim=(-2, -1)))

        parts = {'mag_complex': per_lane.detach().mean().item()}

        if self.sisdr_weight != 0.0:
            pred_wav = self._istft(pred_spec)
            target_wav = self._istft(target_spec)
            active = target_wav.square().mean(-1) > self.silence_floor
            term = -si_sdr(pred_wav, target_wav) * active.to(pred_wav.dtype)
            per_lane = per_lane + self.sisdr_weight * term
            parts['sisdr'] = term.detach().mean().item()

        if self.echo_leak_weight != 0.0:
            if mask_bins is None or residual_spec is None:
                raise ValueError(
                    "echo_leak_weight != 0 needs mask_bins and residual_spec")
            # ⚠ R = D - D_hat is a PENALTY here, never a target.  Raising this
            # weight moves the operating point along the echo/near-end Pareto
            # curve; it does not improve the curve.  Report the value used.
            leaked = _compress(mask_bins.to(residual_spec.dtype) * residual_spec,
                               self.gamma)
            term = leaked.abs().square().mean(dim=(-2, -1))
            per_lane = per_lane + self.echo_leak_weight * term
            parts['echo_leak'] = term.detach().mean().item()

        if weights is None:
            return per_lane.mean(), parts
        return (per_lane * weights).sum() / weights.sum().clamp_min(1e-8), parts


# ============================================================
# Streaming STFT with an overlap tail
# ============================================================

def stream_stft(wave, grid: AecGrid, tail=None):
    """``(B, C, T)`` -> ``((B, C, F, T_f), new_tail)`` on a CONTINUOUS frame grid.

    ``tail`` is the last ``win_len - hop_len`` samples of the previous chunk of
    the same sequence.  Prepending it and analysing with ``center=False`` makes
    the frame grid of chunk n+1 the exact continuation of chunk n's.

    ⚠ With ``center=True`` (torch's default, and what the NR trainers use) each
    chunk is zero-padded by half a window at both ends.  Inside a sequence that
    is a fabricated 32 ms gap every four seconds: the adaptive front-end sees an
    echo-path discontinuity, re-converges, and the corpus's real echo-path
    changes become indistinguishable from chunk boundaries.
    """
    overlap = grid.win_len - grid.hop_len
    if tail is None:
        tail = wave.new_zeros(wave.shape[0], wave.shape[1], overlap)
    x = torch.cat([tail, wave], dim=-1)
    length = x.shape[-1]
    n_frames = (length - grid.win_len) // grid.hop_len + 1
    if n_frames < 1:
        raise ValueError(
            f"chunk of {wave.shape[-1]} samples is shorter than one window "
            f"({grid.win_len}) once the {overlap}-sample tail is included")
    spec = stft(x, grid, center=False)
    keep = length - n_frames * grid.hop_len
    return spec, x[..., length - keep:].detach()


# ============================================================
# Data
# ============================================================

def _sequence_level_split(dataset, indices, val_fraction, seed):
    """Hold out whole SEQUENCES, using the shared split implementation.

    ⚠ Splitting chunks would put the same speaker, room and echo path on both
    sides, and the validation curve would measure memorisation.  The shared
    ``locality_preserving_random_split`` is applied to the list of sequence ids
    rather than to the chunks, so the definition of "the split" stays single
    even though what is being split is different.
    """
    order = []
    seen = set()
    for index in indices:
        sid = int(dataset.meta(index)['sequence_id'])
        if sid not in seen:
            seen.add(sid)
            order.append(sid)
    n_val = max(1, int(round(len(order) * val_fraction)))
    if n_val >= len(order):
        raise ValueError(
            f"val_fraction {val_fraction} would consume all {len(order)} "
            f"sequences")
    train_ids, _ = locality_preserving_random_split(
        order, len(order) - n_val, n_val, seed)
    train_set = {order[i] for i in train_ids.indices}
    train_indices, val_indices = [], []
    for index in indices:
        sid = int(dataset.meta(index)['sequence_id'])
        (train_indices if sid in train_set else val_indices).append(index)
    if not train_indices or not val_indices:
        raise ValueError(
            f"sequence-level split over {len(order)} sequences left one side "
            f"empty (train={len(train_indices)}, val={len(val_indices)})")
    return train_indices, val_indices


def resolve_splits(dataset, val_dataset, cfg, args, resume_ckpt):
    """``(train_dataset, train_indices, val_dataset, val_indices, description)``."""
    all_indices = list(range(len(dataset)))

    if resume_ckpt is not None and 'train_indices' in resume_ckpt:
        train_indices = list(resume_ckpt['train_indices'])
        val_indices = list(resume_ckpt['val_indices'])
        source = 'checkpoint'
        if resume_ckpt.get('val_from_separate_dir') and val_dataset is None:
            raise ValueError(
                "checkpoint was trained with a separate validation corpus; "
                "pass --val-packed-dir so the split means the same thing")
        if val_dataset is not None:
            return dataset, train_indices, val_dataset, val_indices, source
        return dataset, train_indices, dataset, val_indices, source

    if val_dataset is not None:
        return (dataset, all_indices, val_dataset,
                list(range(len(val_dataset))), 'separate --val-packed-dir')

    # The generator can produce a SOURCE-DISJOINT split (disjoint speakers,
    # noises, rooms and loudspeakers).  ⚠ Prefer it whenever it is present: a
    # random split over the same shards puts the same speaker and the same
    # device on both sides, which is a strictly easier and less honest question.
    tagged = {dataset.meta(i).get('split') for i in all_indices}
    if {'train', 'val'} <= tagged:
        train_indices = [i for i in all_indices if dataset.meta(i)['split'] == 'train']
        val_indices = [i for i in all_indices if dataset.meta(i)['split'] == 'val']
        return (dataset, train_indices, dataset, val_indices,
                'source-disjoint split recorded in the corpus')

    val_fraction = cfg.getfloat('training', 'val_fraction', fallback=0.1)
    train_indices, val_indices = _sequence_level_split(
        dataset, all_indices, val_fraction, args.seed)
    return (dataset, train_indices, dataset, val_indices,
            f'⚠ sequence-level random split (val_fraction={val_fraction}); the '
            f'corpus carries no source-disjoint split, so validation shares '
            f'speakers and devices with training')


def make_loader(dataset, indices, lanes, shuffle, seed, workers, prefetch,
                pin_memory, label):
    """Sequence-aware loader.  Returns ``(loader, lanes_used)``."""
    sequence_ids = [int(dataset.meta(i)['sequence_id']) for i in indices]
    chunk_indices = [int(dataset.meta(i)['chunk_index']) for i in indices]
    n_sequences = len(set(sequence_ids))
    if lanes > n_sequences:
        print(f"  ⚠ {label}: lanes {lanes} > {n_sequences} sequences; "
              f"clamping to {n_sequences}. Every lane needs its own sequence or "
              f"two lanes would replay the same state.")
        lanes = n_sequences
    # ⚠ SequenceChunkSampler rejects a subset whose sequences are incomplete, so
    # this call is also the guard that the split above was made over sequences
    # and not over chunks.
    sampler = SequenceChunkSampler(sequence_ids, chunk_indices, n_lanes=lanes,
                                   shuffle=shuffle, seed=seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_sampler=sampler, collate_fn=aec_collate,
        **dataloader_worker_kwargs(workers, pin_memory, prefetch))
    return loader, sampler, lanes


# ============================================================
# One pass over a loader
# ============================================================

def run_pass(model, frontend, criterion, loader, sampler, grid, device,
             optimizer=None, grad_clip=5.0, idle_weight=3.0, dt_weight=1.0,
             max_steps=None, desc='train', pin_memory=False):
    """Walk the loader keeping per-lane state.  ``optimizer=None`` -> validation."""
    training = optimizer is not None
    model.train(training)

    state = frontend_state = tail = None
    total, count = 0.0, 0
    steps = len(sampler) if max_steps is None else min(max_steps, len(sampler))
    iterator = itertools.islice(loader, steps)

    context = torch.enable_grad() if training else torch.no_grad()
    with context, tqdm.tqdm(iterator, total=steps, desc=desc, leave=False) as pbar:
        for stems, meta in pbar:
            stems = stems.to(device=device, dtype=torch.float32,
                             non_blocking=pin_memory)
            batch = stems.shape[0]
            named = AecStems(stems)

            reset = lane_reset_mask([m['chunk_index'] for m in meta]).to(device)
            if state is None:
                state = model.init_state(batch, device=device)
                frontend_state = frontend.init_state(batch, device=device)
                tail = None
            state = model.reset_lanes(state, reset)
            frontend_state = frontend.reset_lanes(frontend_state, reset)
            if tail is not None:
                tail = tail * (~reset).to(tail.dtype).view(-1, 1, 1)

            wave = torch.stack([named.Y, named.X, named.D, named.S], dim=1)
            spec, tail = stream_stft(wave, grid, tail)
            y_spec, x_spec, d_spec, s_spec = spec.unbind(dim=1)

            # ⚠ Frozen. No gradient reaches the front-end, ever.
            with torch.no_grad():
                e_spec, d_hat, frontend_state = frontend.process(
                    y_spec, x_spec, frontend_state, D=d_spec)
            e_spec, d_hat = e_spec.detach(), d_hat.detach()

            mask, state = model(e_spec, d_hat, x_spec, state)
            enhanced = model.apply_mask(mask, e_spec)

            mask_bins = residual = None
            if criterion.echo_leak_weight != 0.0:
                mask_bins = model.expand_to_bins(mask)
                residual = d_spec - d_hat

            weights = scenario_weights(meta, idle_weight, dt_weight, device)
            loss, parts = criterion(enhanced, s_spec, mask_bins=mask_bins,
                                    residual_spec=residual, weights=weights)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            state = model.detach_state(state)
            total += loss.item()
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             g=f"{mask_magnitude(mask).mean().item():.3f}",
                             **{k: f"{v:.3f}" for k, v in parts.items()
                                if k != 'mag_complex'})

    if count == 0:
        raise RuntimeError(f"{desc}: the loader produced no batches")
    return total / count


# ============================================================
# Train
# ============================================================

def train(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    # Seed before anything draws randomness, so the split and the lane layout
    # depend on --seed alone.
    set_seed(args.seed)

    grid = AecGrid.from_config(cfg)
    epochs = cfg.getint('training', 'epochs', fallback=100)
    lanes = cfg.getint('training', 'lanes', fallback=16)
    lr = cfg.getfloat('training', 'lr', fallback=1e-3)
    min_lr = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    lr_patience = cfg.getint('training', 'lr_patience', fallback=4)
    patience = cfg.getint('training', 'early_stop_patience', fallback=20)
    grad_clip = cfg.getfloat('training', 'grad_clip', fallback=5.0)
    epoch_size = cfg.getint('training', 'epoch_size', fallback=0)
    workers = cfg.getint('training', 'num_workers', fallback=2)
    prefetch = cfg.getint('training', 'prefetch_factor', fallback=2)
    output_dir = cfg.get('paths', 'output_dir', fallback='output')

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device
                              or cfg.get('training', 'device', fallback='cpu'))
    pin_memory = device.type == 'cuda'

    packed_dir = args.packed_dir or cfg.get('paths', 'packed_dir', fallback=None)
    if not packed_dir:
        raise ValueError("--packed-dir or [paths] packed_dir required")
    val_dir = args.val_packed_dir or cfg.get('paths', 'val_packed_dir',
                                             fallback=None) or None

    dataset = PackedAecDataset(packed_dir, expected_sr=grid.sr, mmap=args.mmap)
    val_dataset = (PackedAecDataset(val_dir, expected_sr=grid.sr, mmap=args.mmap)
                   if val_dir else None)

    model = build_model(cfg, grid).to(device)
    frontend = build_frontend(cfg, grid, device=device)
    criterion = PostFilterLoss(
        grid,
        gamma=cfg.getfloat('loss', 'gamma', fallback=0.3),
        mag_weight=cfg.getfloat('loss', 'mag_weight', fallback=500.0),
        complex_weight=cfg.getfloat('loss', 'complex_weight', fallback=500.0),
        sisdr_weight=cfg.getfloat('loss', 'sisdr_weight', fallback=1.0),
        echo_leak_weight=cfg.getfloat('loss', 'echo_leak_weight', fallback=0.0),
    ).to(device)
    idle_weight = cfg.getfloat('loss', 'idle_weight', fallback=3.0)
    dt_weight = cfg.getfloat('loss', 'dt_weight', fallback=1.0)

    # The contract must exist before the resume checkpoint is read, and the
    # split must come FROM the checkpoint -- redrawing it would move previously
    # trained sequences into validation.
    contract = build_contract(cfg, grid, model)
    resume_ckpt = None
    frontend_history = []
    if args.resume:
        print(f"Resuming: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device,
                                 weights_only=False)
        require_checkpoint_contract(resume_ckpt, contract, context=args.resume)
        frontend_history = require_frontend_match(
            resume_ckpt, frontend.frontend_id, args.allow_frontend_change,
            context=args.resume)

    (train_ds, train_indices, val_ds, val_indices,
     split_description) = resolve_splits(dataset, val_dataset, cfg, args,
                                         resume_ckpt)

    train_loader, train_sampler, train_lanes = make_loader(
        train_ds, train_indices, lanes, True, args.seed, workers, prefetch,
        pin_memory, 'train')
    val_loader, val_sampler, val_lanes = make_loader(
        val_ds, val_indices, lanes, False, args.seed, min(workers, 2), prefetch,
        pin_memory, 'val')
    # epoch_size is in CHUNKS; batches are lanes wide.  max(1, ...) so a small
    # epoch_size shortens the epoch instead of producing zero batches and a
    # confusing "the loader produced no batches".
    max_steps = max(1, epoch_size // train_lanes) if epoch_size > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)

    # ---------------- banner ----------------
    print("=" * 72)
    print("PostFilter -- joint residual-echo + noise suppressor")
    print("=" * 72)
    print(f"  grid: sr={grid.sr} n_fft={grid.n_fft} win={grid.win_len} "
          f"hop={grid.hop_len} -> n_freqs={grid.n_freqs}, "
          f"{grid.frame_rate:.4g} fps")
    print(f"  {model.describe()}")
    print(f"  model_version   = {MODEL_VERSION}")
    print(f"  feature_version = {FEATURE_VERSION}")
    print(f"  loss_version    = {LOSS_VERSION}")
    print(f"  frontend_id     = {frontend.frontend_id}")
    if frontend_history:
        print(f"  ⚠ frontend_history = {frontend_history}  (OOD lineage)")
    print(f"  loss: gamma={criterion.gamma} mag={criterion.mag_weight:g} "
          f"complex={criterion.complex_weight:g} sisdr={criterion.sisdr_weight:g} "
          f"echo_leak={criterion.echo_leak_weight:g}")
    print(f"  scenario weights: idle={idle_weight:g} (ref_dropout, near_only), "
          f"double_talk={dt_weight:g}")
    print(f"  split: {split_description}")
    print(f"    train {len(train_indices)} chunks / "
          f"{len({train_ds.meta(i)['sequence_id'] for i in train_indices})} sequences")
    print(f"    val   {len(val_indices)} chunks / "
          f"{len({val_ds.meta(i)['sequence_id'] for i in val_indices})} sequences")
    print(f"  lanes: train={train_lanes} val={val_lanes}, "
          f"{len(train_sampler)} train batches/epoch"
          + (f" (capped to {max_steps})" if max_steps else ""))
    print(f"  device={device}, lr={lr}, grad_clip={grad_clip}, seed={args.seed}")
    print("=" * 72)

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
        if start_epoch > epochs:
            # ⚠ Otherwise the loop body never runs, no checkpoint is written,
            # and the run looks successful while having done nothing.
            print(f"  ⚠ nothing to do: the checkpoint is already at epoch "
                  f"{start_epoch - 1} and [training] epochs = {epochs}. "
                  f"Raise epochs to continue training.")

    for epoch in range(start_epoch, epochs + 1):
        # Reshuffle which sequence lands in which lane.  ⚠ Not calling this
        # would give every epoch the same lane layout, so the model would see
        # the same sequences in the same order for the whole run.
        train_sampler.set_epoch(epoch)

        train_loss = run_pass(
            model, frontend, criterion, train_loader, train_sampler, grid,
            device, optimizer=optimizer, grad_clip=grad_clip,
            idle_weight=idle_weight, dt_weight=dt_weight, max_steps=max_steps,
            desc=f"Epoch {epoch}/{epochs}", pin_memory=pin_memory)
        val_loss = run_pass(
            model, frontend, criterion, val_loader, val_sampler, grid, device,
            idle_weight=idle_weight, dt_weight=dt_weight, desc='val',
            pin_memory=pin_memory)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}")
        scheduler.step(val_loss)

        # Update the early-stopping counter BEFORE writing the checkpoint, so
        # the saved no_improve matches what a resume needs to restore.
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, no_improve = val_loss, 0
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
            'frontend_id': frontend.frontend_id,
            'frontend_history': frontend_history,
            'split_description': split_description,
            'val_from_separate_dir': val_ds is not train_ds,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'config': {section: dict(cfg[section]) for section in cfg.sections()},
            **contract,
        }
        torch.save(ckpt, os.path.join(output_dir, 'postfilter_last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'postfilter_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        elif patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"Training done. Best val loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PostFilter training')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='directory of packed AEC .pt shards, or one shard')
    parser.add_argument('--val-packed-dir', default=None,
                        help='separately generated validation corpus '
                             '(gen_aec_dataset.py --split val)')
    parser.add_argument('--mmap', action='store_true',
                        help='memory-map the shards (low RAM, disk-backed)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--allow-frontend-change', action='store_true',
                        help='⚠ resume a checkpoint trained behind a DIFFERENT '
                             'front-end. Valid as an OOD experiment, not as a '
                             'comparable result; the lineage is recorded.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; also fixes the train/val split and the '
                             'lane layout.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    train(parser.parse_args())
