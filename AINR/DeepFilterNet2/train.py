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
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import tqdm

try:  # Package import for library consumers and tests.
    from .model import DeepFilterNet2
except ImportError:  # Direct ``python train.py`` execution.
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
from training_common import (  # noqa: E402
    GradNormLog,
    NonFiniteTraining,
    fast_forward_scheduler,
    halt_on_non_finite,
    make_scheduler,
    scan_non_finite,
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
#
# v6 intentionally restores DFN2's output composition while retaining the v5
# encoder/decoder widths and all training-safety changes: ERB mask over every
# bin, DF on the masked low-frequency spectrum, then the learned sigmoid alpha
# blend against that masked residual.  The alpha parameters already existed in
# v5 but were unused, so old band-split checkpoints are not valid for this graph.
#
# ⚠ lookahead stays 1/1 and the ERB filterbank stays triangular/overlapping/sum.
# Both are operator decisions, NOT oversights -- see config.ini [model] and
# UPSTREAM_ALIGNMENT.md.  Everything else in the port is aligned to upstream.
#
# FEATURE v5: three changes, all toward upstream.
#   1. The normaliser inits are back to libDF's literal MEAN_NORM_INIT
#      (-60/-90 dB) and UNIT_NORM_INIT (0.001/0.0001), reverting the corpus
#      calibration shipped as ...calibrated_v3 in 37db9df.  That commit was the
#      only behavioural change between a run that trained fine and a run that
#      diverged into NaN, on an identical corpus, both from scratch.
#   2. spec_norm_eps is GONE.  libDF divides by a bare sqrt(state); the eps was
#      this port's own addition.  See config.ini [feature] for the measurement
#      showing removal is safe in fp32 and for the one case where it is not.
#   3. The analysis spectrum now carries libDF's wnorm (see analysis_scale), so
#      the normalisers see the scale their constants were calibrated against.
# ⚠ This is NOT a return to '...dual_ema_state_v2': that contract carried
# spec_norm_eps, so reusing the string would let a v2 checkpoint load into a
# model whose feature path no longer matches it.
MODEL_VERSION = 'dfn2_fmajor_pathway_cascade_alpha_no_lsnr_v6'
FEATURE_VERSION = 'dfn2_libdf_wnorm_upstream_init_no_eps_v5'
LOSS_VERSION = 'dfn_mrsl_mag_complex_gamma_v2'


# ============================================================
# Multi-resolution STFT loss
# ============================================================

#: The clamp inside _SafeAngle.backward.  Same value upstream (df/utils.py:74).
SAFE_ANGLE_CLAMP = 1e-10
#: Gradient gain of _SafeAngle is |x| / max(|x|^2, SAFE_ANGLE_CLAMP).  It peaks
#: at |x| = sqrt(clamp) with value 1/sqrt(clamp) -- so the worst case is 1e5, NOT
#: 1/clamp = 1e10, and at x = 0 exactly the gain falls to ZERO.  Measured against
#: autograd at 12 magnitudes; see tests/test_dfn2_contract.py.
#:
#: ⚠ The consequence for diagnosis: an exactly-silent target is NOT itself the
#: hazard.  The hazard is a PREDICTION whose STFT magnitude sits near 1e-5, and a
#: silent target matters only because it drives |y| down THROUGH that band.  With
#: gamma=0.3 a zero target compresses to clamp_min(1e-12)**0.3 = 2.51e-4, so the
#: prediction is pulled toward 2.51e-4, not toward zero.
SAFE_ANGLE_WORST_MAG = SAFE_ANGLE_CLAMP ** 0.5          # 1e-5; peak gain is 1/this


class _SafeAngle(torch.autograd.Function):
    """DeepFilterNet-compatible complex angle with a finite zero gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.angle(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        inv_power = grad / (
            x.real.square() + x.imag.square()
        ).clamp_min(SAFE_ANGLE_CLAMP)
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
        # Upstream treats factor == 0 as "MRSL off" outright (loss.py:718-722),
        # even with factor_complex > 0, so factor=0/factor_complex=500 is not a
        # configuration upstream can express.  Reject it rather than silently
        # running a complex-only objective upstream never trains with.
        or (loss_cfg['factor'] == 0 and loss_cfg['factor_complex'] > 0)
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
    *and* inference.py, so a knob whose fallback was updated in one place built a
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


def validate_signal_config(n_fft, win_len, hop_len, n_erb, df_bins, df_order,
                          mask_lookahead, df_lookahead):
    """Reject grid/model combinations the port cannot build. Raises ValueError.

    Extracted from ``train()`` so it is reachable from a test: while it lived
    inline, a test could only re-implement the arithmetic and assert on its own
    copy, which passes whatever the real rule says.
    """
    if not 0 < win_len <= n_fft:
        raise ValueError('win_len must be in (0, n_fft]')
    if not 0 < hop_len <= win_len:
        raise ValueError('hop_len must be in (0, win_len]')
    if not 0 < df_bins <= n_fft // 2 + 1:
        raise ValueError('df_bins exceeds the available STFT bins')
    # ⚠ THE divisibility guard for this port -- model.py carries no equivalent
    # assert, so do not delete it assuming otherwise.  The two stride-2 stages
    # only need % 4, but upstream asserts % 8 and this port follows it: 20 and 36
    # pass a % 4 check and are rejected upstream.
    if n_erb <= 0 or n_erb % 8:
        raise ValueError('n_erb must be positive and divisible by eight')
    if not 0 <= mask_lookahead <= 2:
        raise ValueError('mask_lookahead must be in [0, 2]')
    if not 0 <= df_lookahead < df_order:
        raise ValueError('df_lookahead must be in [0, df_order)')
    # The df_lookahead <= mask_lookahead relation is enforced ONCE, in
    # DeepFilterNet2.__init__ -- every entry point constructs the model, so
    # re-checking it here would be a fourth independent copy of the same rule.


def analysis_scale(n_fft, win_len, hop_len):
    """Factor that turns a ``normalized=True`` STFT into libDF's analysis scale.

    libDF multiplies its analysis spectrum by
    ``wnorm = 1 / (window_size^2 / (2 * frame_size))`` = ``2*hop / win^2``
    (lib.rs:133, applied at :390-392), and EVERYTHING calibrated against that
    spectrum -- compute_band_corr, the +1e-10 floor, MEAN_NORM_INIT,
    UNIT_NORM_INIT -- assumes it.  ``torch.stft(normalized=True)`` instead applies
    ``1/sqrt(n_fft)``, so this returns the residual ``wnorm * sqrt(n_fft)``
    (= 1/32 at 1024/1024/512, i.e. libDF's own 1/1024 in total).

    ⚠ It is not cosmetic on the DF path.  ``band_unit_norm`` is
    ``x / sqrt(EMA|x|)``, so scaling x by c scales its OUTPUT by sqrt(c) -- a
    permanent factor, not a transient.  Left unapplied, feat_spec entered the DF
    branch at sqrt(32) = 5.66x upstream's regime forever while feat_erb entered at
    upstream's, mis-scaling the two encoder branches relative to each other.
    """
    return (2.0 * hop_len / (win_len ** 2)) * math.sqrt(n_fft)


def erb_band_db(spec_BTC, erb_fb, analysis_scale):
    """Band energies in dB, exactly as the model is fed them.

    ⚠ SHARED with calibrate_norm_init.py on purpose.  This used to be four lines
    hand-copied there, and they drifted twice -- once on the log floor
    (clamp_min(1e-16) vs +1e-10) and once on the power expression -- each time
    silently biasing the init values the calibrator recommends.  The sibling
    project still carries that drift.  One call site each makes it unrepresentable
    rather than merely tested.

    ``analysis_scale`` is applied to the POWER as s**2 rather than to the spectrum
    as s: identical result (s is an exact power of two at every supported grid, so
    the exponent shift is lossless -- verified bit-equal by torch.equal), but it
    touches (B, T, n_erb) floats instead of (B, T, n_bins) complex.

    The +1e-10 floor is upstream's literal (lib.rs:209).  ⚠ It was never
    re-derived for this port's band scale; see config.ini [feature] for the
    measurement and for why raising it is the wrong repair.
    """
    power = spec_BTC.real.square() + spec_BTC.imag.square()
    band = power.matmul(erb_fb.T) * (analysis_scale * analysis_scale)
    return (band + 1e-10).log10() * 10


def read_feature_config(cfg, sr, hop_len):
    section = 'feature'
    known = {
        'erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
        'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
        'spec_norm_init_hi',
    }
    # Reject unknown keys the way read_model_config does.  Silently ignoring them
    # let a stale `spec_norm_eps = 1e-12` read as though the guard were honoured
    # while the code divided by a bare sqrt.
    for name in cfg.options(section):
        if name not in known:
            raise ValueError(
                f"[{section}] {name!r} is not a feature-normalisation key "
                f"(known: {', '.join(sorted(known))})")
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    erb_tau = cfg.getfloat(section, 'erb_norm_tau_sec', fallback=1.0)
    spec_tau = cfg.getfloat(section, 'spec_norm_tau_sec', fallback=1.0)
    feature_cfg = {
        'analysis_scale': analysis_scale(n_fft, win_len, hop_len),
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
    }
    if (
        erb_tau <= 0
        or spec_tau <= 0
        or feature_cfg['erb_scale_db'] <= 0
        # The unit-norm state is a magnitude the model divides by as
        # x / sqrt(state); a non-positive init makes the first frames NaN.
        # Easy to paste in by accident: a least-squares ramp fitted to a convex
        # magnitude profile extrapolates below zero at the high-frequency edge, which is
        # exactly what calibrate_norm_init.py produced on a real corpus before
        # it grew a positivity guard.  RNNoise-ERB has always rejected this;
        # this project silently accepted it.
        or feature_cfg['spec_init_lo'] <= 0
        or feature_cfg['spec_init_hi'] <= 0
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
    mask_pf,
    pf_beta,
    feature_cfg,
    loss_cfg,
):
    return {
        'composition': 'dfn2_cascade_alpha',
        'sr': sr,
        'n_fft': n_fft,
        'win_len': win_len,
        'hop_len': hop_len,
        'n_erb': n_erb,
        'df_bins': df_bins,
        'df_order': df_order,
        'mask_lookahead': mask_lookahead,
        'mask_pf': mask_pf,
        'pf_beta': pf_beta,
        'df_lookahead': df_lookahead,
        # No 'analysis_scale': it is analysis_scale(n_fft, win_len, hop_len) and
        # all three are keys above, so it cannot mismatch independently.
        'erb_norm_tau_sec': feature_cfg['erb_tau_sec'],
        'erb_norm_alpha': feature_cfg['erb_alpha'],
        'erb_norm_init_lo_db': feature_cfg['erb_init_lo_db'],
        'erb_norm_init_hi_db': feature_cfg['erb_init_hi_db'],
        'erb_norm_scale_db': feature_cfg['erb_scale_db'],
        'spec_norm_tau_sec': feature_cfg['spec_tau_sec'],
        'spec_norm_alpha': feature_cfg['spec_alpha'],
        'spec_norm_init_lo': feature_cfg['spec_init_lo'],
        'spec_norm_init_hi': feature_cfg['spec_init_hi'],
        'loss_fft_sizes': ','.join(str(n) for n in loss_cfg['fft_sizes']),
        'loss_gamma': loss_cfg['gamma'],
        'loss_factor': loss_cfg['factor'],
        'loss_factor_complex': loss_cfg['factor_complex'],
    }


#: Contract keys that record how the weights were TRAINED but must not constrain
#: how a trained checkpoint may be RENDERED.  The loss configuration cannot affect
#: inference at all.  The post-filter is an output-shaping choice the operator is
#: entitled to change on an existing checkpoint -- ⚠ it IS recorded, because DFN3's
#: form is ungated and therefore does change the training objective, and it IS
#: still enforced when resuming training; it is only waived on the inference path.
#: Without this, flipping `mask_pf = true` to try the post-filter made inference.py
#: refuse to load the checkpoint, i.e. the knob could not be exercised at all.
RENDER_ONLY_CONTRACT_KEYS = ('mask_pf', 'pf_beta')


def require_checkpoint_contract(
    ckpt,
    expected,
    context='checkpoint',
    for_training=True,
):
    versions = {
        'model_version': MODEL_VERSION,
        'feature_version': FEATURE_VERSION,
    }
    if for_training:
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
        if not for_training and (
            key.startswith('loss_') or key in RENDER_ONLY_CONTRACT_KEYS
        ):
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
    State init = linspace(mean_norm_init) across bands (NOT first-frame).

    ⚠ The init IS libDF's MEAN_NORM_INIT = -60..-90 dB, restored deliberately.
    It is NOT calibrated for this port: the triangular / overlapping / energy-SUM
    bank plus normalized=True puts measured band levels ~+39 dB higher than
    libDF's (composite +32.8..+48.4 dB per band), and the fitted values for this
    corpus are -17.35/-57.81.  Those were shipped and then reverted, because
    training diverged under them and converges under these.  config.ini [feature]
    carries the full reasoning and the suspected mechanism (the +1e-10 log floor,
    which these values keep at upstream's -1.00 bound by coincidence).
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


def causal_ema_mag_norm(spec_low, norm_state=None, alpha=0.989,
                        unit_norm_init=(0.001, 0.0001)):
    """
    DeepFilterNet band_unit_norm (libDF lib.rs): per-bin EMA of |x|, divide by SQRT(EMA).
        s = |x|*(1-a) + s*a ;  x = x / sqrt(s)
    State init = linspace(unit_norm_init) across bins (NOT first-frame); the
    default IS libDF's UNIT_NORM_INIT (lib.rs:13).  See config.ini [feature] for
    why the corpus-fitted pair was shipped and then reverted.
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
        frames.append(spec_low[:, t:t + 1, :] / torch.sqrt(mu))
    normed = torch.cat(frames, dim=1)

    return normed, mu.detach()


def extract_dfn2_features(
    spec_c,
    erb_fb,
    df_bins,
    feature_cfg,
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
    if ema_state is not None and not isinstance(ema_state, dict):
        raise ValueError("ema_state must be None or a dict with 'erb'/'spec'")
    erb_state_in = None if ema_state is None else ema_state.get('erb')
    spec_state_in = None if ema_state is None else ema_state.get('spec')

    # ⚠ FEATURES ONLY.  The returned spec_c keeps the analysis scale the ISTFT
    # inverts, so the round trip stays unity; only the normaliser inputs move to
    # libDF's wnorm.
    scale = feature_cfg['analysis_scale']
    spec_BTC = spec_c.permute(0, 2, 1)                         # view, no copy
    erb_db = erb_band_db(spec_BTC, erb_fb, scale)
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
    # ⚠ The DF path takes RAW COMPLEX, so it needs the scale in amplitude form --
    # a filterbank-only fold would leave it unscaled and band_unit_norm's
    # x/sqrt(EMA|x|) would then be off by sqrt(1/32) permanently.
    spec_low = spec_BTC[:, :, :df_bins] * scale                # (B, T, df_bins) complex
    unit_s, spec_state = causal_ema_mag_norm(
        spec_low,
        norm_state=spec_state_in,
        alpha=feature_cfg['spec_alpha'],
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
        'training', 'weight_decay_end', fallback=0.01
    )
    grad_clip    = cfg.getfloat('training', 'grad_clip', fallback=1.0)
    amsgrad      = cfg.getboolean('training', 'amsgrad', fallback=True)
    patience     = cfg.getint('training', 'early_stop_patience', fallback=25)
    epoch_size   = cfg.getint('training', 'epoch_size', fallback=0)
    mmap_block_size = cfg.getint('training', 'mmap_block_size', fallback=256)
    mmap_workers = cfg.getint('training', 'mmap_num_workers', fallback=2)
    prefetch_factor = cfg.getint('training', 'prefetch_factor', fallback=2)
    output_dir   = cfg.get('paths', 'output_dir', fallback='output')

    if mmap_workers < 0:
        raise ValueError("mmap_num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be greater than zero")
    validate_signal_config(N_FFT, WIN_LEN, HOP_LEN, N_ERB, DF_BINS, DF_ORDER,
                           MASK_LOOKAHEAD, DF_LOOKAHEAD)
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
        model_cfg['mask_pf'],
        model_cfg['pf_beta'],
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

    # amsgrad=True matches upstream, which hardcodes it on its adamw branch
    # (df/train.py:495) rather than exposing it.  Without the running max of the
    # second moment, beta2=0.999 gives the denominator a ~1000-step memory, so a
    # single gradient spike is followed by oversized steps until v recovers --
    # the amplifier that turns one bad batch into a multi-epoch runaway.  Not a
    # tuning knob here: it is an alignment fix.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=weight_decay,
                                  amsgrad=amsgrad)
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
        f"per-step LR/WD schedule, grad_clip={grad_clip:g}, "
        f"amsgrad={amsgrad}"
    )
    print(
        f"  feature init: erb={feature_cfg['erb_init_lo_db']:g}/"
        f"{feature_cfg['erb_init_hi_db']:g} dB  "
        f"spec={feature_cfg['spec_init_lo']:g}/{feature_cfg['spec_init_hi']:g}"
        f"  [{FEATURE_VERSION}]"
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
            ckpt, contract, context=args.resume, for_training=True
        )
        model.load_state_dict(ckpt['state_dict'])
        poisoned = scan_non_finite(model)
        if poisoned:
            print('  ⚠ the resumed WEIGHTS are already non-finite:')
            for name, n_nan, n_inf, numel in poisoned[:10]:
                print(f'      {name}: {n_nan} NaN, {n_inf} inf of {numel}')
            raise NonFiniteTraining(
                f'{args.resume} contains non-finite weights; training from it '
                'can only produce NaN. Resume from an earlier checkpoint.'
            )
        if args.reset_optimizer:
            print('  --reset-optimizer: model weights only, fresh optimizer '
                  f'(amsgrad={optimizer.param_groups[0]["amsgrad"]})')
        elif 'optimizer' in ckpt:
            ckpt_amsgrad = ckpt['optimizer']['param_groups'][0].get('amsgrad')
            optimizer.load_state_dict(ckpt['optimizer'])
            # ⚠ load_state_dict restores param_group HYPERPARAMETERS too, so a
            # checkpoint written before amsgrad was turned on silently reverts it
            # to False -- verified on torch 2.8: the flag flips back, no
            # max_exp_avg_sq is ever allocated, and nothing warns.
            #
            # ⚠ Do NOT paper over that by force-enabling amsgrad here.  Flipping
            # the optimizer mid-run destroys the one thing a diagnostic resume is
            # for: it removes the very variable under test, so a run that then
            # survives tells you nothing about whether the batch or the optimizer
            # was to blame.  Make the fork explicit instead.
            if ckpt_amsgrad is False and amsgrad:
                print()
                print('  ' + '-' * 64)
                print('  ⚠ CHECKPOINT OPTIMIZER PREDATES amsgrad=True')
                print('  ' + '-' * 64)
                print('  The restored optimizer state carries amsgrad=False and')
                print('  has no max_exp_avg_sq buffers, so this run continues')
                print('  WITHOUT amsgrad -- deliberately, and it is now visible')
                print('  instead of silent.')
                print()
                print('  This is the right setting for a DIAGNOSTIC resume: one')
                print('  variable changes (the guard + the norm trace), so a')
                print('  spike in grad_norm.csv is attributable.')
                print()
                print('  For a real amsgrad run use --reset-optimizer (loads the')
                print('  weights, builds a fresh optimizer).  Never carry moments')
                print('  across an optimizer change and call it the same run.')
                print('  ' + '-' * 64)
                print()
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get(
            'global_step', (start_epoch - 1) * len(train_loader)
        )
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        # Rebuilt, never restored -- fast_forward_scheduler()'s docstring has
        # the measured reason.
        resumed_lr = fast_forward_scheduler(scheduler, global_step)
        print(f"  Resumed epoch {start_epoch - 1}, best_val_loss={best_val_loss:.5f}")
        print(f"  scheduler rebuilt for epochs={epochs} and fast-forwarded "
              f"{global_step} steps (lr={resumed_lr:.4e})")

    # Append mode, so a resumed run continues the same trace instead of starting
    # a second one that cannot be compared against the first.
    grad_log = GradNormLog(os.path.join(output_dir, 'grad_norm.csv'), SR,
                           hazard_mag=SAFE_ANGLE_WORST_MAG)

    def make_halt_context(epoch):
        """Assemble halt_on_non_finite's arguments -- called ONLY when halting.

        Everything here (state_dict copies, a full config walk) is too expensive
        to build per batch, and the halt path runs at most once per process.
        Model/optimizer/scheduler state is still pre-step at every call site, so
        the checkpoint it captures is uncontaminated.
        """
        def context(batch_idx, noisy, clean, enhanced):
            return {
                'model': model,
                'noisy': noisy, 'clean': clean, 'enhanced': enhanced,
                'epoch': epoch,
                'batch_idx': batch_idx, 'global_step': global_step,
                'output_dir': output_dir, 'sr': SR,
                'hazard_mag': SAFE_ANGLE_WORST_MAG,
                'checkpoint': {
                    'epoch': epoch - 1,
                    'global_step': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_version': MODEL_VERSION,
                    'feature_version': FEATURE_VERSION,
                    'loss_version': LOSS_VERSION,
                    'contract': contract,
                    'config': {
                        k: dict(v) for k, v in cfg.items() if k != 'DEFAULT'
                    },
                },
            }
        return context

    for epoch in range(start_epoch, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        halt_context = make_halt_context(epoch)
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch_idx, (noisy, clean) in enumerate(pbar):
                noisy = noisy.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)   # (B, T)
                clean = clean.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)
                T = noisy.shape[-1]

                # ⚠ center / pad_mode are left at torch's defaults (center=True,
                # pad_mode='reflect'), which is NOT libDF's analysis.  libDF
                # zero-primes analysis_mem (lib.rs:118-119) and its offline
                # front-end DROPS the primed frames (transforms.rs:162,187), so
                # its frame 0 begins in silence where ours begins in a
                # time-reversed copy of the clip -- and that lands inside the
                # init-dominated EMA transient the norm-init was calibrated for
                # (3*tau = 273 frames vs 281 frames per 3 s segment).  In steady
                # state center=True is only a fixed half-window frame-timing
                # offset that a streaming implementation reproduces exactly, and
                # train/inference here are mutually consistent (inference.py uses
                # the same defaults), so this is a documented offset, not a
                # latency bug.
                # ⚠ MEASURED: center=False is NOT a one-line alignment.  At
                # win=1024 / hop=512 torch.istft REFUSES it outright --
                # "window overlap add min: 1" -- because the first and last
                # frames lack OLA coverage.  Aligning means replicating libDF's
                # framing (zero-prime analysis_mem, then DROP the primed frames),
                # not flipping a flag.
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
                # One device->host sync for the loss, here.  The finiteness check
                # below reads this float instead of doing `if not
                # torch.isfinite(loss)`, which was a second sync at the same
                # point in the stream.
                loss_value = loss.item()

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

                # ⚠ Check the LOSS separately from the gradient.  A non-finite
                # loss is a forward-side fault (a division or log inside the
                # objective) and diagnoses differently from a finite loss with an
                # exploding gradient, which is a backward-side fault.  Upstream
                # also separates them (df/train.py:380-390 vs :392-419).
                if not math.isfinite(loss_value):
                    halt_on_non_finite(
                        'loss is non-finite before backward '
                        '(forward-side fault)',
                        loss_value=loss_value, total_norm=None,
                        **halt_context(batch_idx, noisy, clean, enhanced_wav),
                    )

                optimizer.zero_grad()
                loss.backward()
                # ⚠ error_if_nonfinite=True is what stops clipping from CREATING
                # the NaN.  Without it, total_norm=inf gives
                # clip_coef = 1.0/(inf+1e-6) = 0.0, and inf*0.0 = NaN -- which
                # optimizer.step() then writes into the weights AND into Adam's
                # exp_avg/exp_avg_sq, so no later clean batch can recover it.
                # Verified: 20 clean batches after one inf gradient leave the
                # weight still NaN.  With the flag, the raise happens BEFORE any
                # scaling, so gradients and weights are untouched.
                try:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    halt_on_non_finite(
                        f'non-finite gradient (backward-side fault): {exc}',
                        loss_value=loss_value, total_norm='non-finite',
                        **halt_context(batch_idx, noisy, clean, enhanced_wav),
                    )
                optimizer.step()
                scheduler.step()
                # One sync for the norm, here, reused by both the CSV and the
                # progress bar.  Formatting `float(total_norm)` in set_postfix
                # copied the same device scalar a second time.
                norm_value = float(total_norm)
                grad_log.record(
                    norm_value, epoch=epoch, batch_idx=batch_idx,
                    global_step=global_step, loss_value=loss_value,
                    noisy=noisy, clean=clean, enhanced=enhanced_wav,
                    output_dir=output_dir,
                )

                # After record(): the CSV row and a halt report must name the
                # same step for the same batch.
                global_step += 1

                train_loss += loss_value
                # refresh=False lets tqdm's own mininterval throttle the redraw;
                # the default forces a terminal write every step.
                pbar.set_postfix(loss=f"{loss_value:.4f}",
                                 gnorm=f"{norm_value:.2e}", refresh=False)

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

        # ⚠ Never overwrite dfn2_last.pth with a poisoned state.  The gradient
        # guard covers the path where a NaN arrives through backward, but not
        # every path: a non-finite value can enter a BUFFER (the two EMA norm
        # states are carried outside autograd), and a finite loss over already-
        # NaN weights would step right past the guard.  Without this check the
        # only good checkpoint gets replaced by a dead one and the run becomes
        # unrecoverable -- exactly the failure this whole exercise is about.
        poisoned = scan_non_finite(model)
        optim_bad = [
            (pid, key)
            for pid, entry in optimizer.state_dict()['state'].items()
            for key, value in entry.items()
            if torch.is_tensor(value) and value.is_floating_point()
            and not torch.isfinite(value).all()
        ]
        if poisoned or optim_bad or not math.isfinite(val_loss):
            print('  ⚠ REFUSING to write a checkpoint: state is non-finite')
            for name, n_nan, n_inf, numel in poisoned[:10]:
                print(f'      weight {name}: {n_nan} NaN, {n_inf} inf of {numel}')
            for pid, key in optim_bad[:10]:
                print(f'      optim  param {pid} / {key}')
            if not math.isfinite(val_loss):
                print(f'      val_loss = {val_loss}')
            print(f'      {os.path.join(output_dir, "dfn2_last.pth")} left as-is')
            raise NonFiniteTraining(
                f'non-finite state at end of epoch {epoch}; the last good '
                'checkpoint was preserved rather than overwritten'
            )

        is_best = val_loss < best_val_loss
        checkpoint_best = min(best_val_loss, val_loss)
        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # No 'scheduler': rebuilt from epochs/steps and fast-forwarded on
            # resume, so a stored T_max cannot survive an epochs change.
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

    # No try/finally around the epoch loop: the trace is line-buffered, so every
    # step is already on disk and a halt or a hard kill loses nothing.  This close
    # is tidiness, not durability.
    grad_log.close()
    print(f"Training done. Best val loss: {best_val_loss:.5f}")
    print(f"  gradient-norm trace: "
          f"{os.path.join(output_dir, 'grad_norm.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFilterNet2 Training')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='packed.pt file or directory containing *.pt (pack_dataset.py output)')
    parser.add_argument('--mmap', action='store_true',
                        help='Memory-map .pt tensors (low RAM, disk-backed; needs PyTorch>=2.0)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='with --resume: load model weights but build a '
                             'fresh optimizer. Required to actually get '
                             'amsgrad=True from a checkpoint written without '
                             'it, because load_state_dict restores the flag. '
                             'Omit for a diagnostic resume, where changing the '
                             'optimizer would remove the variable under test.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    # ⚠ 42, NOT upstream's 43 -- deliberately, and do not "align" it.  A seed is
    # arbitrary: matching upstream's reproduces nothing of upstream's run (different
    # corpus, different data order, different RNG), so the alignment buys zero. All
    # standalone NR trainers sharing one seed keeps their comparison split stable.
    # Enforced by
    # AINR/tests/test_bakeoff_protocol.py::test_seed_defaults_match.
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42; use -1 to disable)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    train(args)
