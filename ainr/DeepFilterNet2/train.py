"""
DeepFilterNet2 訓練腳本

用法:
    python train.py --config config.ini --packed-dir data_48k/packed.pt --gpu 0
    python train.py --config config.ini --packed-dir data_48k/packed.pt \
        --resume output/dfn2_best.pth

Dataset format: dataset_gen/pack_dataset.py output containing
{'data': (N, 2, T), 'sr': 48000}.
"""

import argparse
import configparser
import inspect
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import tqdm

from model import DeepFilterNet2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# PackedDataset, the sampler, the seeder and the train/val split are shared by
# all three models -- see dataset_gen/loader.py for why the split in particular
# must not be re-implemented per trainer.
from dataset_gen import (  # noqa: E402
    BlockShuffleSampler,
    PackedDataset,
    dataloader_worker_kwargs,
    load_packed_dataset,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
)


# v3: _build_erb_fb() (model.py) rewritten to the exact triangular
# construction from this project's own aaronhsueh0506/DeepFilterNet-Keras
# bandERB.ipynb (ERBBand()/ERB_pro_matrix()) -- same as RNNoise-ERB's
# train.py -- replacing a different ERB-rate formula (21.4/0.00437 instead
# of the correct 9.265/24.7), a different band-width enforcement, and a
# non-doubled forward matrix. erb_fb/erb_inv are registered buffers (part of
# state_dict), so old checkpoints carry stale values that load_state_dict
# would silently restore; bump forces a fresh training run instead.
#
# v4: the architecture was realigned to released DeepFilterNet3 (minus lsnr) --
# PReLU->ReLU, dense->GroupedLinearEinsum, GRU layers moved from the encoder to
# the ERB decoder, and the mask/DF cascade replaced by DFN3's parallel band
# split.  That last change costs ZERO parameters, so neither load_state_dict
# nor a parameter count can reject a v3 checkpoint; only this string can.
#
# v5 completes the realignment with two more parameter-count-invisible fixes:
# the encoder now flattens frequency-major (B,T,F,C) like upstream instead of
# channel-major, which changes which elements GroupedLinearEinsum groups
# together; and the ERB decoder takes encoder skips through 1x1 pathway convs
# and ADDS them, instead of concatenating.  The concat form and the pathway
# form both total 14,210 conv parameters here -- identical numbers, different
# functions, which is precisely why the version string has to carry it.
MODEL_VERSION = 'dfn3_fmajor_flatten_pathway_add_no_lsnr_v5'
FEATURE_VERSION = 'dfn2_dual_ema_state_v2'
LOSS_VERSION = 'dfn_mrsl_mag_complex_gamma_v2'


# ============================================================
# Multi-resolution STFT loss
# ============================================================

class _SafeAngle(torch.autograd.Function):
    """DeepFilterNet-compatible complex angle with a finite zero gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.angle(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        inv_power = grad / (x.real.square() + x.imag.square()).clamp_min(1e-10)
        return torch.complex(-x.imag * inv_power, x.real * inv_power)


class _LossStft(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, waveform):
        return torch.stft(
            waveform,
            self.n_fft,
            self.hop,
            window=self.window,
            normalized=True,
            return_complex=True,
        )


class MultiResSpecLoss(nn.Module):
    """DeepFilterNet phase-aware multi-resolution spectrogram objective.

    Every resolution contributes both gamma-compressed magnitude MSE and
    gamma-compressed complex MSE.  Resolution losses are summed, matching the
    upstream implementation.  ``clean=0`` remains a finite pure-noise target.
    """

    def __init__(self, fft_sizes=(256, 512, 1024, 2048), gamma=0.3,
                 factor=500.0, factor_complex=500.0):
        super().__init__()
        self.gamma = gamma
        self.factor = factor
        self.factor_complex = factor_complex
        self.stfts = nn.ModuleDict({str(n): _LossStft(n) for n in fft_sizes})

    def forward(self, enhanced, clean):
        total = torch.zeros((), device=enhanced.device, dtype=enhanced.dtype)
        for stft_fn in self.stfts.values():
            y = stft_fn(enhanced)
            s = stft_fn(clean)
            y_abs = y.abs()
            s_abs = s.abs()
            if self.gamma != 1:
                y_abs = y_abs.clamp_min(1e-12).pow(self.gamma)
                s_abs = s_abs.clamp_min(1e-12).pow(self.gamma)
            total = total + F.mse_loss(y_abs, s_abs) * self.factor
            if self.factor_complex != 0:
                if self.gamma != 1:
                    y = y_abs * torch.exp(1j * _SafeAngle.apply(y))
                    s = s_abs * torch.exp(1j * _SafeAngle.apply(s))
                total = total + F.mse_loss(
                    torch.view_as_real(y),
                    torch.view_as_real(s),
                ) * self.factor_complex
        return total


def read_loss_config(cfg):
    section = 'multi_res_spec_loss'
    fft_sizes = tuple(
        int(value.strip())
        for value in cfg.get(
            section, 'fft_sizes', fallback='256,512,1024,2048'
        ).split(',')
        if value.strip()
    )
    loss_cfg = {
        'fft_sizes': fft_sizes,
        'gamma': cfg.getfloat(section, 'gamma', fallback=0.3),
        'factor': cfg.getfloat(section, 'factor', fallback=500.0),
        'factor_complex': cfg.getfloat(
            section, 'factor_complex', fallback=500.0
        ),
    }
    if (
        not fft_sizes
        or any(n <= 0 or n % 2 for n in fft_sizes)
        or not 0 < loss_cfg['gamma'] <= 1
        or loss_cfg['factor'] < 0
        or loss_cfg['factor_complex'] < 0
        or loss_cfg['factor'] + loss_cfg['factor_complex'] == 0
    ):
        raise ValueError('invalid DeepFilterNet MultiResSpecLoss configuration')
    return loss_cfg



def make_norm_alpha(sr, hop_len, tau):
    """Match DeepFilterNet's stable rounded EMA coefficient."""
    exact = math.exp(-(hop_len / sr) / tau)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(exact, precision)
        precision += 1
    return alpha


# ``[model] enc_channels`` predates the constructor's ``enc_ch``; every other
# key already matches its keyword argument by name.
_MODEL_KWARG_ALIASES = {'enc_channels': 'enc_ch'}


def read_model_config(cfg):
    """Every DeepFilterNet2 constructor argument, defaults overlaid with [model].

    Defaults come from ``DeepFilterNet2.__init__``'s own signature rather than
    from a ``fallback=`` on each read.  They used to be restated in train.py
    *and* denoise.py, so a knob whose fallback was updated in one place built a
    differently-shaped model in the other -- which surfaces only as a
    load_state_dict failure at inference time, long after the training run.

    Unknown ``[model]`` keys raise rather than being silently ignored, and the
    parsed type follows the signature default, so a new constructor argument
    becomes configurable without touching this function.
    """
    kwargs = {
        name: param.default
        for name, param in inspect.signature(DeepFilterNet2.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    kwargs['sr'] = cfg.getint('signal', 'sr')
    kwargs['n_fft'] = cfg.getint('signal', 'n_fft')
    for name in cfg.options('model'):
        kwarg = _MODEL_KWARG_ALIASES.get(name, name)
        if kwarg not in kwargs:
            raise ValueError(
                f"[model] {name!r} is not a DeepFilterNet2 constructor argument "
                f"(known: {', '.join(sorted(kwargs))})")
        default = kwargs[kwarg]
        if isinstance(default, bool):          # before int -- bool subclasses int
            kwargs[kwarg] = cfg.getboolean('model', name)
        elif isinstance(default, int):
            kwargs[kwarg] = cfg.getint('model', name)
        elif isinstance(default, float):
            kwargs[kwarg] = cfg.getfloat('model', name)
        elif isinstance(default, tuple):
            kwargs[kwarg] = tuple(int(v) for v in cfg.get('model', name).split(','))
        else:
            kwargs[kwarg] = cfg.get('model', name)
    return kwargs


def read_feature_config(cfg, sr, hop_len):
    section = 'feature'
    erb_tau = cfg.getfloat(section, 'erb_norm_tau_sec', fallback=1.0)
    spec_tau = cfg.getfloat(section, 'spec_norm_tau_sec', fallback=1.0)
    feature_cfg = {
        'erb_tau_sec': erb_tau,
        'erb_alpha': make_norm_alpha(sr, hop_len, erb_tau),
        'erb_init_lo_db': cfg.getfloat(
            section, 'erb_norm_init_lo_db', fallback=-60.0
        ),
        'erb_init_hi_db': cfg.getfloat(
            section, 'erb_norm_init_hi_db', fallback=-90.0
        ),
        'erb_scale_db': cfg.getfloat(
            section, 'erb_norm_scale_db', fallback=40.0
        ),
        'spec_tau_sec': spec_tau,
        'spec_alpha': make_norm_alpha(sr, hop_len, spec_tau),
        'spec_init_lo': cfg.getfloat(
            section, 'spec_norm_init_lo', fallback=0.001
        ),
        'spec_init_hi': cfg.getfloat(
            section, 'spec_norm_init_hi', fallback=0.0001
        ),
        'spec_eps': cfg.getfloat(section, 'spec_norm_eps', fallback=1e-12),
    }
    if (
        erb_tau <= 0
        or spec_tau <= 0
        or feature_cfg['erb_scale_db'] <= 0
        or feature_cfg['spec_eps'] <= 0
    ):
        raise ValueError('invalid DeepFilterNet feature-normalization configuration')
    return feature_cfg


def make_checkpoint_contract(
    sr,
    n_fft,
    win_len,
    hop_len,
    n_erb,
    df_bins,
    df_order,
    mask_lookahead,
    df_lookahead,
    feature_cfg,
    loss_cfg,
):
    return {
        'sr': sr,
        'n_fft': n_fft,
        'win_len': win_len,
        'hop_len': hop_len,
        'n_erb': n_erb,
        'df_bins': df_bins,
        'df_order': df_order,
        'mask_lookahead': mask_lookahead,
        'df_lookahead': df_lookahead,
        'erb_norm_tau_sec': feature_cfg['erb_tau_sec'],
        'erb_norm_alpha': feature_cfg['erb_alpha'],
        'erb_norm_init_lo_db': feature_cfg['erb_init_lo_db'],
        'erb_norm_init_hi_db': feature_cfg['erb_init_hi_db'],
        'erb_norm_scale_db': feature_cfg['erb_scale_db'],
        'spec_norm_tau_sec': feature_cfg['spec_tau_sec'],
        'spec_norm_alpha': feature_cfg['spec_alpha'],
        'spec_norm_init_lo': feature_cfg['spec_init_lo'],
        'spec_norm_init_hi': feature_cfg['spec_init_hi'],
        'spec_norm_eps': feature_cfg['spec_eps'],
        'loss_fft_sizes': ','.join(str(n) for n in loss_cfg['fft_sizes']),
        'loss_gamma': loss_cfg['gamma'],
        'loss_factor': loss_cfg['factor'],
        'loss_factor_complex': loss_cfg['factor_complex'],
    }


def require_checkpoint_contract(
    ckpt,
    expected,
    context='checkpoint',
    require_loss=True,
):
    versions = {
        'model_version': MODEL_VERSION,
        'feature_version': FEATURE_VERSION,
    }
    if require_loss:
        versions['loss_version'] = LOSS_VERSION
    for key, want in versions.items():
        got = ckpt.get(key)
        if got != want:
            shown = repr(got) if got is not None else 'missing (legacy contract)'
            raise ValueError(
                f"{context} {key}={shown}, expected {want!r}; "
                "this change requires a fresh training run."
            )

    saved = ckpt.get('contract', {})
    for key, want in expected.items():
        if not require_loss and key.startswith('loss_'):
            continue
        got = saved.get(key)
        if isinstance(want, str):
            matches = str(got) == want
        else:
            matches = got is not None and math.isclose(
                float(got), float(want), rel_tol=1e-7, abs_tol=1e-7
            )
        if not matches:
            raise ValueError(
                f"{context} {key}={got!r}, runtime requires {want!r}; "
                "use the training config that belongs to this checkpoint."
            )


# ============================================================
# Feature extraction
# ============================================================

def causal_ema_db_norm(
    erb_db,
    norm_state=None,
    alpha=0.989,
    mean_norm_init=(-60.0, -90.0),
    scale_db=40.0,
):
    """
    DeepFilterNet band_mean_norm_erb: per-band causal EMA of dB, subtract running mean, /40.
    State init = linspace(MEAN_NORM_INIT) = -60..-90 dB (NOT first-frame). Requires the STFT
    to be normalized=True (fft^-0.5) so the -60/-90 init is calibrated.
    erb_db    : (B, T, n_erb)
    Returns:
        normed   : (B, T, n_erb)
        norm_state: updated tensor (B, 1, n_erb)
    """
    B, T, n_erb = erb_db.shape
    device = erb_db.device

    state_ok = (
        norm_state is not None
        and tuple(norm_state.shape) == (B, 1, n_erb)
    )
    if not state_ok:
        lo_i, hi_i = mean_norm_init
        mu = torch.linspace(lo_i, hi_i, n_erb, device=device, dtype=erb_db.dtype
                            ).view(1, 1, n_erb).expand(B, 1, n_erb).clone()
    else:
        mu = norm_state.to(device=device, dtype=erb_db.dtype)

    frames = []
    for t in range(T):
        mu = alpha * mu + (1 - alpha) * erb_db[:, t:t + 1, :]
        frames.append((erb_db[:, t:t + 1, :] - mu) / scale_db)
    normed = torch.cat(frames, dim=1)

    return normed, mu.detach()


def causal_ema_mag_norm(spec_low, norm_state=None, alpha=0.989, eps=1e-12,
                        unit_norm_init=(0.001, 0.0001)):
    """
    DeepFilterNet band_unit_norm (libDF lib.rs): per-bin EMA of |x|, divide by SQRT(EMA).
        s = |x|*(1-a) + s*a ;  x = x / sqrt(s + eps)
    State init = linspace(UNIT_NORM_INIT) = 0.001..0.0001 across bins (NOT first-frame).
    spec_low  : (B, T, df_bins) complex
    Returns: normed (B, T, df_bins) complex and state (B, 1, df_bins)
    """
    B, T, df_bins = spec_low.shape
    device = spec_low.device

    state_ok = (
        norm_state is not None
        and tuple(norm_state.shape) == (B, 1, df_bins)
    )
    if not state_ok:
        lo_i, hi_i = unit_norm_init
        mu = torch.linspace(lo_i, hi_i, df_bins, device=device, dtype=spec_low.real.dtype
                            ).view(1, 1, df_bins).expand(B, 1, df_bins).clone()
    else:
        mu = norm_state.to(device=device, dtype=spec_low.real.dtype)

    frames = []
    for t in range(T):
        mag = spec_low[:, t:t + 1, :].abs()
        mu = alpha * mu + (1 - alpha) * mag
        frames.append(spec_low[:, t:t + 1, :] / torch.sqrt(mu + eps))
    normed = torch.cat(frames, dim=1)

    return normed, mu.detach()


def extract_dfn2_features(
    spec_c,
    erb_fb,
    df_bins,
    feature_cfg=None,
    ema_state=None,
):
    """
    Extract DFN2 input features from complex spectrum.

    Args:
        spec_c   : (B, n_bins, T) complex  (return_complex=True convention)
        erb_fb   : (n_erb, n_bins) tensor on same device
        df_bins  : int
        feature_cfg: normalization constants returned by ``read_feature_config``
        ema_state: ``{'erb': ..., 'spec': ...}`` or None.  The two paths must
            remain independent across streaming chunks.

    Returns:
        spec_c   : unchanged, (B, n_bins, T) complex
        feat_erb : (B, 1, T, n_erb)   DFN2 encoder expects [B, 1, T, Fe] — time before freq
        feat_spec: (B, 2, T, df_bins) DFN2 encoder expects [B, 2, T, Fc]
        ema_state: updated state
    """
    if feature_cfg is None:
        feature_cfg = {
            'erb_alpha': 0.989,
            'erb_init_lo_db': -60.0,
            'erb_init_hi_db': -90.0,
            'erb_scale_db': 40.0,
            'spec_alpha': 0.989,
            'spec_init_lo': 0.001,
            'spec_init_hi': 0.0001,
            'spec_eps': 1e-12,
        }
    if ema_state is not None and not isinstance(ema_state, dict):
        raise ValueError("ema_state must be None or a dict with 'erb'/'spec'")
    erb_state_in = None if ema_state is None else ema_state.get('erb')
    spec_state_in = None if ema_state is None else ema_state.get('spec')

    spec_BTC = spec_c.permute(0, 2, 1)                         # (B, T, n_bins)

    # ERB features: dB + causal EMA normalisation
    erb_power = spec_BTC.abs().pow(2).matmul(erb_fb.T)         # (B, T, n_erb)
    erb_db = (erb_power + 1e-10).log10() * 10
    feat_erb_BTE, erb_state = causal_ema_db_norm(
        erb_db,
        norm_state=erb_state_in,
        alpha=feature_cfg['erb_alpha'],
        mean_norm_init=(
            feature_cfg['erb_init_lo_db'],
            feature_cfg['erb_init_hi_db'],
        ),
        scale_db=feature_cfg['erb_scale_db'],
    )
    feat_erb = feat_erb_BTE.unsqueeze(1)                        # (B, 1, T, n_erb)

    # DF features: unit-norm magnitude + view_as_real
    spec_low = spec_BTC[:, :, :df_bins]                        # (B, T, df_bins) complex
    unit_s, spec_state = causal_ema_mag_norm(
        spec_low,
        norm_state=spec_state_in,
        alpha=feature_cfg['spec_alpha'],
        eps=feature_cfg['spec_eps'],
        unit_norm_init=(
            feature_cfg['spec_init_lo'],
            feature_cfg['spec_init_hi'],
        ),
    )
    feat_spec = torch.view_as_real(unit_s)                     # (B, T, df_bins, 2)
    feat_spec = feat_spec.permute(0, 3, 1, 2)                  # (B, 2, T, df_bins)

    return spec_c, feat_erb, feat_spec, {
        'erb': erb_state,
        'spec': spec_state,
    }


# ============================================================
# Scheduler
# ============================================================

def make_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    base_lr,
    min_lr,
    warmup_lr,
):
    start_factor = warmup_lr / base_lr
    if not 0 < start_factor <= 1:
        raise ValueError('lr_warmup must be in (0, lr]')
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, warmup_steps)],
    )


# ============================================================
# Train
# ============================================================

def train(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed: {args.seed}")

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR      = cfg.getint('signal', 'sr')
    N_FFT   = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)

    model_cfg = read_model_config(cfg)
    N_ERB          = model_cfg['n_erb']
    DF_BINS        = model_cfg['df_bins']
    DF_ORDER       = model_cfg['df_order']
    MASK_LOOKAHEAD = model_cfg['mask_lookahead']
    DF_LOOKAHEAD   = model_cfg['df_lookahead']

    epochs       = cfg.getint('training', 'epochs')
    batch_size   = cfg.getint('training', 'batch_size')
    lr           = cfg.getfloat('training', 'lr')
    min_lr       = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    warmup_lr    = cfg.getfloat('training', 'lr_warmup', fallback=1e-4)
    warmup_ep    = cfg.getint('training', 'warmup_epochs', fallback=3)
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=1e-12)
    weight_decay_end = cfg.getfloat(
        'training', 'weight_decay_end', fallback=0.05
    )
    grad_clip    = cfg.getfloat('training', 'grad_clip', fallback=1.0)
    patience     = cfg.getint('training', 'early_stop_patience', fallback=20)
    epoch_size   = cfg.getint('training', 'epoch_size', fallback=0)
    mmap_block_size = cfg.getint('training', 'mmap_block_size', fallback=256)
    mmap_workers = cfg.getint('training', 'mmap_num_workers', fallback=2)
    prefetch_factor = cfg.getint('training', 'prefetch_factor', fallback=2)
    output_dir   = cfg.get('paths', 'output_dir', fallback='output')

    if mmap_workers < 0:
        raise ValueError("mmap_num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be greater than zero")
    if not 0 < WIN_LEN <= N_FFT:
        raise ValueError('win_len must be in (0, n_fft]')
    if not 0 < HOP_LEN <= WIN_LEN:
        raise ValueError('hop_len must be in (0, win_len]')
    if not 0 < DF_BINS <= N_FFT // 2 + 1:
        raise ValueError('df_bins exceeds the available STFT bins')
    if N_ERB <= 0 or N_ERB % 4:
        raise ValueError('n_erb must be positive and divisible by four')
    if not 0 <= MASK_LOOKAHEAD <= 2:
        raise ValueError('mask_lookahead must be in [0, 2]')
    if not 0 <= DF_LOOKAHEAD < DF_ORDER:
        raise ValueError('df_lookahead must be in [0, df_order)')
    if weight_decay < 0 or weight_decay_end < 0:
        raise ValueError('weight decay values must be non-negative')
    if grad_clip <= 0:
        raise ValueError('grad_clip must be positive')

    feature_cfg = read_feature_config(cfg, SR, HOP_LEN)
    loss_cfg = read_loss_config(cfg)
    contract = make_checkpoint_contract(
        SR,
        N_FFT,
        WIN_LEN,
        HOP_LEN,
        N_ERB,
        DF_BINS,
        DF_ORDER,
        MASK_LOOKAHEAD,
        DF_LOOKAHEAD,
        feature_cfg,
        loss_cfg,
    )

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device_str = args.device or cfg.get('training', 'device', fallback='cpu')
        device = torch.device(device_str)

    packed_dir = args.packed_dir or cfg.get('paths', 'packed_dir', fallback=None)
    if not packed_dir:
        raise ValueError("--packed-dir or [paths] packed_dir required")

    # Accepts either a directory (scans for *.pt) or a direct .pt path.
    dataset = load_packed_dataset(packed_dir, expected_sr=SR, mmap=args.mmap)
    n_train, n_val = split_sizes(dataset)
    train_set, val_set = locality_preserving_random_split(
        dataset, n_train, n_val, args.seed)

    pin_memory = device.type == 'cuda'
    train_workers = mmap_workers if args.mmap else 4
    train_kwargs = dataloader_worker_kwargs(
        train_workers, pin_memory, prefetch_factor
    )
    sample_count = epoch_size if 0 < epoch_size < len(train_set) else None
    if args.mmap:
        sampler = BlockShuffleSampler(
            train_set, block_size=mmap_block_size, num_samples=sample_count
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=sampler, **train_kwargs
        )
    elif sample_count is not None:
        sampler = RandomSampler(
            train_set, replacement=False, num_samples=sample_count
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=sampler, **train_kwargs
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **train_kwargs
        )

    val_workers = min(train_workers, 2)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        **dataloader_worker_kwargs(val_workers, pin_memory, prefetch_factor),
    )

    model = DeepFilterNet2(**model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(warmup_ep * len(train_loader), total_steps - 1)
    scheduler = make_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        lr,
        min_lr,
        warmup_lr,
    )
    loss_fn = MultiResSpecLoss(**loss_cfg).to(device)

    stft_window = torch.hann_window(WIN_LEN).pow(0.5).to(device)

    print(f"DeepFilterNet2 training: SR={SR}, N_FFT={N_FFT}, WIN={WIN_LEN}, HOP={HOP_LEN}")
    print(f"  n_erb={N_ERB}, df_bins={DF_BINS}, df_order={DF_ORDER}")
    print(f"  mask_lookahead={MASK_LOOKAHEAD}, df_lookahead={DF_LOOKAHEAD}")
    print(f"  batch={batch_size}, lr={lr}, device={device}")
    print(
        f"  EMA: erb_alpha={feature_cfg['erb_alpha']:.6f}, "
        f"spec_alpha={feature_cfg['spec_alpha']:.6f}, independent states"
    )
    print(
        f"  loss={LOSS_VERSION}: fft_sizes={loss_cfg['fft_sizes']}, "
        f"gamma={loss_cfg['gamma']}, magnitude_factor={loss_cfg['factor']}, "
        f"complex_factor={loss_cfg['factor_complex']}"
    )
    print(
        f"  weight_decay={weight_decay:g}->{weight_decay_end:g}, "
        f"per-step LR/WD schedule, grad_clip={grad_clip:g}"
    )
    if args.mmap:
        print(f"  mmap: block={mmap_block_size}, workers={train_workers}, "
              f"prefetch={prefetch_factor}, packed_dtype_preserved=True")

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    no_improve = 0
    global_step = 0

    if args.resume:
        print(f"Resuming: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_contract(
            ckpt, contract, context=args.resume, require_loss=True
        )
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get(
            'global_step', (start_epoch - 1) * len(train_loader)
        )
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed epoch {start_epoch - 1}, best_val_loss={best_val_loss:.5f}")

    for epoch in range(start_epoch, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for noisy, clean in pbar:
                noisy = noisy.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)   # (B, T)
                clean = clean.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)
                T = noisy.shape[-1]

                spec_c = torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True, normalized=True,
                )  # (B, n_bins, T_f)

                spec_c, feat_erb, feat_spec, _ = extract_dfn2_features(
                    spec_c, model.erb_fb, DF_BINS,
                    feature_cfg=feature_cfg,
                )

                enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)

                # ISTFT
                enhanced_wav = torch.istft(
                    enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T, normalized=True,
                )

                loss = loss_fn(enhanced_wav, clean)

                progress = min(global_step, total_steps - 1) / max(
                    total_steps - 1, 1
                )
                current_wd = float(
                    weight_decay_end
                    + 0.5
                    * (weight_decay - weight_decay_end)
                    * (1.0 + math.cos(math.pi * progress))
                )
                for group in optimizer.param_groups:
                    group['weight_decay'] = current_wd
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)
                clean = clean.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)
                T = noisy.shape[-1]

                spec_c = torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True, normalized=True,
                )
                spec_c, feat_erb, feat_spec, _ = extract_dfn2_features(
                    spec_c, model.erb_fb, DF_BINS,
                    feature_cfg=feature_cfg,
                )
                enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)
                enhanced_wav = torch.istft(
                    enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T, normalized=True,
                )
                val_loss += loss_fn(enhanced_wav, clean).item()

        val_loss /= len(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        is_best = val_loss < best_val_loss
        checkpoint_best = min(best_val_loss, val_loss)
        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': checkpoint_best,
            'model_version': MODEL_VERSION,
            'feature_version': FEATURE_VERSION,
            'loss_version': LOSS_VERSION,
            'contract': contract,
            'config': {k: dict(v) for k, v in cfg.items() if k != 'DEFAULT'},
        }
        torch.save(ckpt, os.path.join(output_dir, 'dfn2_last.pth'))

        if is_best:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, os.path.join(output_dir, 'dfn2_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training done. Best val loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFilterNet2 Training')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='packed.pt file or directory containing *.pt (pack_dataset.py output)')
    parser.add_argument('--mmap', action='store_true',
                        help='Memory-map .pt tensors (low RAM, disk-backed; needs PyTorch>=2.0)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42; use -1 to disable)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    train(args)
