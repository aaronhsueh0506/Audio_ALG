"""
RNNoise v0.2-inspired 噪音抑制模型 — 訓練腳本
採用 Conv+GRU 骨架，並改為 config-driven / ERB+complex dual-input /
無 pitch 的本地架構

用法:
    python train.py --config config.ini
    python train.py --config config.ini --device cpu
    python train.py --config config.ini --resume output/rnnoise_epoch5.pth
    python train.py --config config.ini --seed 123
"""

import argparse
import configparser
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AINR_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _AINR_ROOT)

# The sampler, seeder and train/val split live in dataset_gen because GTCRN
# trains on the same packed corpus and the two are compared directly -- which
# samples each model gets is part of that protocol.  The copies that used to
# live here had already drifted to a 10% held-out fraction against GTCRN's 5%.
from dataset_gen import (  # noqa: E402
    BlockShuffleSampler,
    PackedDataset,
    dataloader_worker_kwargs,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
)


# Feature semantics are intentionally versioned.  v3 keeps the two input shapes
# from v2, but aligns their causal normalisation with the original DeepFilterNet:
# per-band EMA mean-normalised log-ERB and per-bin EMA-normalised complex bins.
# It is incompatible with all earlier checkpoints even though both input/output
# dimensions are unchanged.
# v4 removes the erb_norm_clip/spec_clip deployment safety clamp that v3 kept
# on top of the DeepFilterNet formula: verified against both the upstream
# Rikorose/DeepFilterNet libDF/src/lib.rs (band_mean_norm_erb/band_unit_norm,
# neither clips) and this repo's own ainr/DeepFilterNet2 port (same formulas,
# no clip either) that no clip is part of the original algorithm at all.
# v5 fixes erb_bandborder()'s minimum-band-width enforcement: the previous
# "every-OTHER-band-pair >= 2 bins" rule (ported verbatim from
# aaronhsueh0506/DeepFilterNet-Keras's ERBBand() notebook) did not actually
# guarantee every individual band is >= 2 bins wide (a genuine 1-bin band
# was verified to occur at this file's own sr/n_fft/n_bands). Replaced with
# a strict greedy-forward minimum, same style as this file's own
# compute_erb_bands(). This changes the ERB filterbank matrices
# (erb_fwd/erb_inv / nfftborder), hence the resulting features, even though
# input/output dimensions are unchanged.
FEATURE_VERSION = 'log_erb_dfn_mean_cplx_unit_0_4k_v7'
LOSS_VERSION = 'erb_irm_only_v4'
# v2 (2026-07-29), three changes that together invalidate v1 checkpoints:
#   * fft_sizes 256,512,1024,2048 -> 128,256,512,1024, i.e. {n_fft/4, n_fft/2,
#     n_fft, n_fft*2} for THIS model's 512-point FFT.  The old integers were
#     upstream's, chosen against its own 960-point FFT, and applying them here
#     shifted every resolution one octave up (16/32/64/128 ms).
#   * direct ERB IRM term restored alongside MRSL, with the undefined-band mask
#     and (1 + 5*vad) activity weighting that xiph/rnnoise has and this port
#     never had.
#   * training STFT moved to center=False so framing matches deployment.



def require_checkpoint_feature_version(ckpt, context='checkpoint'):
    """Reject checkpoints whose input semantics do not match this code."""
    version = ckpt.get('feature_version', ckpt.get('config', {}).get('feature_version'))
    if version != FEATURE_VERSION:
        shown = repr(version) if version is not None else 'missing (legacy feature contract)'
        raise ValueError(
            f"{context} feature_version={shown}, expected {FEATURE_VERSION!r}. "
            "Feature dimensions match but semantics do not; retrain this model."
        )


def require_checkpoint_feature_config(ckpt, feature_cfg, context='checkpoint'):
    """Require the checkpoint and runtime to use identical normaliser constants."""
    require_checkpoint_feature_version(ckpt, context=context)
    saved = ckpt.get('config', {})
    expected = {
        'sr': feature_cfg['sr'],
        'n_fft': feature_cfg['n_fft'],
        'win_len': feature_cfg['win_len'],
        'hop_len': feature_cfg['hop_len'],
        'lookahead_frames': feature_cfg['lookahead_frames'],
        'n_bands': feature_cfg['n_bands'],
        # min_bins_per_band moves the ERB band borders, hence the whole
        # filterbank and the generated C tables.  config.ini and process.h were
        # guarded against each other but nothing tied a CHECKPOINT to the
        # borders it was trained on, so old weights could be paired with new
        # tables silently.
        'min_bins_per_band': feature_cfg['min_bins_per_band'],
        'feature_erb_norm_tau_sec': feature_cfg['erb_tau_sec'],
        'feature_erb_norm_alpha': feature_cfg['erb_alpha'],
        'feature_erb_norm_init_lo_db': feature_cfg['erb_norm_init_lo_db'],
        'feature_erb_norm_init_hi_db': feature_cfg['erb_norm_init_hi_db'],
        'feature_erb_norm_scale_db': feature_cfg['erb_norm_scale_db'],
        'feature_spec_max_hz': feature_cfg['spec_max_hz'],
        'feature_spec_bins': feature_cfg['spec_bins'],
        'feature_spec_norm_tau_sec': feature_cfg['spec_tau_sec'],
        'feature_spec_norm_alpha': feature_cfg['spec_alpha'],
        'feature_spec_norm_init_lo': feature_cfg['spec_norm_init_lo'],
        'feature_spec_norm_init_hi': feature_cfg['spec_norm_init_hi'],
        'feature_spec_norm_eps': feature_cfg['spec_norm_eps'],
    }
    for key, want in expected.items():
        if key not in saved or not math.isclose(float(saved[key]), float(want),
                                                 rel_tol=1e-7, abs_tol=1e-7):
            got = saved.get(key, 'missing')
            raise ValueError(
                f"{context} {key}={got!r}, but runtime config requires {want!r}; "
                "use the training config or retrain before inference/export."
            )


def model_capacity_from_checkpoint(ckpt):
    """RNNoiseModel shape kwargs, taken from the checkpoint rather than config.

    Capacity is a property of the weights.  A config.ini that disagrees would
    build a model that either fails to load or -- worse -- loads and means
    something else, so both denoise.py and export_onnx.py construct from here.
    (Feature/signal constants are the opposite case: those must match the
    runtime config exactly, which is require_checkpoint_feature_config's job.)

    ``use_complex_input`` is read from the recorded config, falling back to
    probing the weights for checkpoints written before the flag existed.  The
    spec_* shapes are only read once that is known to be true -- they do not
    exist in a pure-ERB checkpoint, and reading them unconditionally is a
    KeyError, not a graceful degradation.
    """
    sd = ckpt['state_dict']
    saved = ckpt.get('config', {})
    use_complex_input = bool(saved.get(
        'use_complex_input', any(k.startswith('spec_conv1.') for k in sd)))
    kwargs = {
        'n_bands': sd['erb_conv.weight'].shape[1],
        'cond_size': sd['erb_conv.weight'].shape[0],
        'gru_size': sd['gru1.weight_ih_l0'].shape[0] // 3,
        'use_complex_input': use_complex_input,
    }
    if use_complex_input:
        kwargs['spec_conv_channels'] = sd['spec_conv1.weight'].shape[0]
        kwargs['spec_embed_size'] = sd['spec_proj.weight'].shape[0]
    return kwargs


def require_checkpoint_loss_config(ckpt, loss_cfg, irm_cfg, context='checkpoint'):
    """Do not resume optimizer state trained with the previous IRM-mixed objective."""
    version = ckpt.get('loss_version', ckpt.get('config', {}).get('loss_version'))
    if version != LOSS_VERSION:
        shown = repr(version) if version is not None else 'missing (legacy loss contract)'
        raise ValueError(
            f"{context} loss_version={shown}, expected {LOSS_VERSION!r}. "
            "The training objective is MRSL + direct ERB-IRM supervision, and "
            "the MRSL resolutions moved to {n_fft/4..n_fft*2}; "
            "start a fresh training run instead of resuming this optimizer state."
        )
    saved = ckpt.get('config', {})
    expected = {
        'loss_fft_sizes': ','.join(str(n) for n in loss_cfg['fft_sizes']),
        'loss_gamma': loss_cfg['gamma'],
        'loss_factor': loss_cfg['factor'],
        'loss_factor_complex': loss_cfg['factor_complex'],
        # The IRM term is the whole objective now, so its settings are at
        # least as load-bearing as MRSL's.  They were absent entirely: a
        # checkpoint trained at gamma=0.25 resumed happily at gamma=0.5.
        'irm_factor': irm_cfg['factor'],
        'irm_gamma': irm_cfg['gamma'],
        'irm_energy_floor': irm_cfg['energy_floor'],
    }
    for key, want in expected.items():
        got = saved.get(key)
        matches = (str(got) == want if isinstance(want, str) else
                   got is not None and math.isclose(float(got), float(want),
                                                    rel_tol=1e-7, abs_tol=1e-7))
        if not matches:
            raise ValueError(
                f"{context} {key}={got!r}, but runtime loss config requires {want!r}; "
                "start a fresh training run or use the checkpoint's original config."
            )


def read_loss_config(cfg):
    """Read the DeepFilterNet 3 production MultiResSpecLoss subset used here."""
    section = 'multi_res_spec_loss'
    fft_sizes = tuple(
        int(x.strip()) for x in
        cfg.get(section, 'fft_sizes', fallback='256,512,1024,2048').split(',')
        if x.strip()
    )
    loss_cfg = {
        'fft_sizes': fft_sizes,
        'gamma': cfg.getfloat(section, 'gamma', fallback=0.3),
        'factor': cfg.getfloat(section, 'factor', fallback=500.0),
        'factor_complex': cfg.getfloat(section, 'factor_complex', fallback=500.0),
    }
    if (not fft_sizes or any(n <= 0 or n % 2 for n in fft_sizes) or
            not 0 < loss_cfg['gamma'] <= 1 or loss_cfg['factor'] < 0 or
            loss_cfg['factor_complex'] < 0):
        raise ValueError('invalid DeepFilterNet MultiResSpecLoss configuration')
    # factor == factor_complex == 0 is legal and means "MRSL off": the ERB-IRM
    # term is then the whole objective.  It used to raise, which is why the
    # imbalanced MRSL+IRM mix was the only reachable configuration.
    # (The dict stays exactly MultiResSpecLoss's kwargs -- callers ask
    # mrsl_is_enabled() rather than reading a non-kwarg flag out of it.)
    return loss_cfg


def mrsl_is_enabled(loss_cfg):
    """False when both MRSL factors are zero, i.e. IRM is the whole objective."""
    return loss_cfg['factor'] + loss_cfg['factor_complex'] > 0


def read_irm_loss_config(cfg):
    """Read the direct ERB-gain (IRM) loss settings."""
    section = 'erb_irm_loss'
    irm_cfg = {
        'factor': cfg.getfloat(section, 'factor', fallback=1.0),
        'gamma': cfg.getfloat(section, 'gamma', fallback=0.25),
        'energy_floor': cfg.getfloat(section, 'energy_floor', fallback=1e-9),
    }
    if (irm_cfg['factor'] < 0 or not 0 < irm_cfg['gamma'] <= 1 or
            irm_cfg['energy_floor'] <= 0):
        raise ValueError('invalid ERB IRM loss configuration')
    return irm_cfg


def read_feature_config(cfg, sr, hop_len, n_fft, win_len=None):
    """Read the dual-input feature contract shared by train/denoise/C."""
    version = cfg.get('feature', 'version', fallback=FEATURE_VERSION)
    if version != FEATURE_VERSION:
        raise ValueError(
            f"config feature version {version!r} is unsupported; expected {FEATURE_VERSION!r}"
        )
    if win_len is None:
        win_len = n_fft
    erb_tau_sec = cfg.getfloat('feature', 'erb_norm_tau_sec', fallback=1.0)
    erb_norm_init_lo_db = cfg.getfloat(
        'feature', 'erb_norm_init_lo_db', fallback=-60.0)
    erb_norm_init_hi_db = cfg.getfloat(
        'feature', 'erb_norm_init_hi_db', fallback=-90.0)
    erb_norm_scale_db = cfg.getfloat(
        'feature', 'erb_norm_scale_db', fallback=40.0)
    spec_max_hz = cfg.getint('feature', 'spec_max_hz', fallback=4000)
    spec_tau_sec = cfg.getfloat('feature', 'spec_norm_tau_sec', fallback=1.0)
    spec_norm_init_lo = cfg.getfloat('feature', 'spec_norm_init_lo', fallback=0.001)
    spec_norm_init_hi = cfg.getfloat('feature', 'spec_norm_init_hi', fallback=0.0001)
    spec_norm_eps = cfg.getfloat('feature', 'spec_norm_eps', fallback=1e-12)
    if (win_len <= 0 or win_len > n_fft or hop_len <= 0 or
            erb_tau_sec <= 0 or erb_norm_scale_db <= 0 or
            spec_max_hz <= 0 or
            spec_max_hz > sr // 2 or spec_tau_sec <= 0 or
            spec_norm_init_lo <= 0 or spec_norm_init_hi <= 0 or
            spec_norm_eps <= 0):
        raise ValueError('invalid DeepFilterNet-style ERB/complex feature configuration')
    spec_bins = spec_max_hz * n_fft // sr + 1
    # An explicit alpha, when given, wins over the tau-derived one.
    #
    # alpha is the per-FRAME decay, so pinning it keeps the normaliser's memory
    # fixed at 1/(1-alpha) frames no matter what sr/hop become.  That is the
    # invariant the GRU actually cares about: its learned time constants are in
    # frames, so a fixed alpha keeps "normaliser adaptation speed vs state
    # update rate" constant across grid changes and the learned dynamics stay
    # valid.  Deriving alpha from tau instead pins the memory in SECONDS, which
    # is the signal-processing invariant but lets the frame-domain relationship
    # drift (63 frames at hop 256 vs 126 at hop 128).
    #
    # The cost of a longer alpha is a longer warm-up, and that only hurts while
    # the init values are wrong; once they are calibrated to the steady state
    # (calibrate_norm_init.py) there is no transient to sit through.
    erb_alpha = cfg.getfloat('feature', 'erb_norm_alpha', fallback=0.0)
    spec_alpha = cfg.getfloat('feature', 'spec_norm_alpha', fallback=0.0)
    if not 0.0 <= erb_alpha < 1.0 or not 0.0 <= spec_alpha < 1.0:
        raise ValueError('norm alpha must be in [0, 1); 0 means "derive from tau"')
    if erb_alpha == 0.0:
        erb_alpha = make_norm_alpha(sr, hop_len, erb_tau_sec)
    if spec_alpha == 0.0:
        spec_alpha = make_norm_alpha(sr, hop_len, spec_tau_sec)
    return dict(
        version=version,
        sr=sr,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        lookahead_frames=cfg.getint('signal', 'lookahead_frames', fallback=0),
        n_bands=cfg.getint('signal', 'n_bands'),
        min_bins_per_band=cfg.getint('signal', 'min_bins_per_band', fallback=2),
        erb_tau_sec=erb_tau_sec,
        erb_alpha=erb_alpha,
        erb_norm_init_lo_db=erb_norm_init_lo_db,
        erb_norm_init_hi_db=erb_norm_init_hi_db,
        erb_norm_scale_db=erb_norm_scale_db,
        spec_max_hz=spec_max_hz,
        spec_bins=spec_bins,
        spec_tau_sec=spec_tau_sec,
        spec_alpha=spec_alpha,
        spec_norm_init_lo=spec_norm_init_lo,
        spec_norm_init_hi=spec_norm_init_hi,
        spec_norm_eps=spec_norm_eps,
    )

# ============================================================
# ERB Band 工具
# ============================================================

def erb_rate(f):
    """頻率 (Hz) → ERB-rate (Glasberg & Moore 1990)"""
    return 21.4 * np.log10(0.00437 * f + 1)

def erb_inv(e):
    """ERB-rate → 頻率 (Hz)"""
    return (10 ** (e / 21.4) - 1) / 0.00437

def compute_erb_bands(n_fft, sr, n_bands, min_bins_per_band):
    """
    計算 ERB band 的 FFT bin 邊界，回傳 shape=(n_bands+1,) 的整數陣列。

    min_bins_per_band: 每個 band 至少 N 個 FFT bins (DFN3 風格, 預設 2)。
                       不夠寬的 band 向後 "借" bin (相當於整體往高頻擠壓)。
    """
    n_bins = n_fft // 2 + 1
    e_low = erb_rate(0)
    e_high = erb_rate(sr / 2)
    erb_edges = np.linspace(e_low, e_high, n_bands + 1)
    freq_edges = erb_inv(erb_edges)
    ideal = np.round(freq_edges / (sr / n_fft)).astype(int)

    # Greedy forward: next_edge = max(ideal_edge, prev_edge + min_bins)
    # 確保每 band 至少 min_bins, 不夠寬的向後借, 高頻原本寬的 band 自動緩衝
    bin_edges = [0]
    for i in range(n_bands):
        next_edge = max(int(ideal[i + 1]), bin_edges[-1] + min_bins_per_band)
        next_edge = min(next_edge, n_bins)
        bin_edges.append(next_edge)
    bin_edges[-1] = n_bins
    return np.array(bin_edges, dtype=int)



def freq2erb(freq_hz):
    """Hz → ERB-number (Glasberg-Moore; DeepFilterNet / DeepFilterNet-Keras constants)."""
    return 9.265 * np.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """ERB-number → Hz (inverse of freq2erb)."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft, min_bins_per_band):
    """Port of DeepFilterNet(-Keras) ERBBand(): returns nfftborder, a length-N
    (N = n_bands) int array of FFT-bin band borders on the Glasberg-Moore ERB scale.
    nfftborder[0]=0 (DC), nfftborder[-1]=n_fft//2+1 (Nyquist+1). N borders → N ERB bands
    (one matrix column each):
        cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N))
        border  = round((cutoff + bw/2) / bw),   bw = (sr/2)/(n_fft/2) = sr/n_fft

    Band-width enforcement: strict greedy-forward minimum -- every consecutive
    border pair is nudged to be at least `min_bins_per_band` bins apart (never
    below the ideal position, so we only ever borrow forward, never move a
    border backward). This replaces the original aaronhsueh0506/
    DeepFilterNet-Keras `ERBBand()` notebook's "every-OTHER-band-pair >= 2"
    rule (`nb[i+2]-nb[i] >= 2`, checked only for i, i+2 -- not i, i+1), which
    does NOT actually guarantee every individual band is >= 2 bins wide:
    verified empirically to produce a genuine 1-bin-wide band at
    sr=16000/n_fft=512/n_bands=22. The greedy-forward style itself mirrors
    this file's own compute_erb_bands() (used by the disabled hybrid-band
    path), just applied to this function's N-border/edge-doubled convention
    instead of that function's N+1-border one.
    """
    n_bins = n_fft // 2 + 1
    high_lim = sr / 2.0
    bw = high_lim / (n_fft / 2.0)                       # freqRangePerBin = sr / n_fft
    erb_lims = np.linspace(freq2erb(0.0), freq2erb(high_lim), n_bands)
    cutoffs = erb2freq(erb_lims)
    ideal = np.round((cutoffs + bw / 2.0) / bw).astype(int)
    nb = [0]
    for i in range(1, n_bands):
        nxt = max(int(ideal[i]), nb[-1] + min_bins_per_band)
        nxt = min(nxt, n_bins)
        nb.append(nxt)
    nb = np.array(nb, dtype=int)
    # Pin the last border to Nyquist+1 = n_bins (at 48k the rounding lands on
    # n_bins exactly, but at other sr/n_fft it can fall a bin short and leave
    # the Nyquist bin uncovered → mode1 partition of unity would break).
    nb[-1] = n_bins
    return nb


def compute_erb_matrix(nfftborder, n_fft, mode=0):
    """Faithful port of DeepFilterNet(-Keras) ERB_pro_matrix(nfftborder, NFFT, mode).
    Triangular ERB filterbank, shape (n_bins, N) with N = len(nfftborder) bands. The
    range(N-1) blocks lie BETWEEN consecutive borders, so column 0 (falling ramp only)
    and column N-1 (rising ramp only) are ONE-SIDED; interior columns are full triangles.
        mode=0 (forward / feature ERBB): x2 on the two one-sided edge columns so their
               band energy is comparable to interior bands.
        mode=1 (inverse / mask expansion): no x2 — a clean partition of unity, so
               mask=1 maps to bin_gain=1 with NO row normalisation.
    """
    n_bins = n_fft // 2 + 1
    N = len(nfftborder)
    W = np.zeros((n_bins, N), dtype=np.float32)
    for i in range(N - 1):
        lo, hi = int(nfftborder[i]), int(nfftborder[i + 1])
        bs = hi - lo
        for j in range(bs):
            W[lo + j, i] = 1.0 - j / bs
            W[lo + j, i + 1] = j / bs
    if mode == 0:
        W[:, 0] *= 2.0
        W[:, N - 1] *= 2.0
    return W


# ============================================================
# 模型 (RNNoise v0.2-inspired 本地架構)
# ============================================================

class RNNoiseModel(nn.Module):
    """
    RNNoise/DeepFilterNet-inspired ERB-gain model.

    The ERB path observes short-term deviations from a causal per-band mean.
    An optional complex path (``use_complex_input``, OFF by default) preserves
    fine low-frequency magnitude/phase structure; see __init__ for why it is
    off.  Both paths are causal apart from caller-controlled lookahead padding.

    ``forward`` always accepts ``spec_features`` so the ONNX signature, the C
    feature contract and every call site stay stable; with the complex path
    disabled the argument is simply unused.
    """
    def __init__(self, n_bands, spec_bins, cond_size=64, gru_size=128,
                 spec_conv_channels=8, spec_embed_size=64, dropout=0.0,
                 use_complex_input=False):
        super().__init__()
        self.n_bands = n_bands
        self.spec_bins = spec_bins
        self.gru_size = gru_size
        self.spec_conv_channels = spec_conv_channels
        self.spec_embed_size = spec_embed_size

        # Three-frame mean-normalised log-ERB path.
        self.erb_conv = nn.Conv1d(n_bands, cond_size, kernel_size=3, padding=0)

        # Per-frame complex spectrum encoder.  Frequency is reduced by 4x;
        # temporal context is applied only after each frame has been embedded.
        #
        # Disabled by default: the model's OUTPUT is 22 real ERB band gains, so
        # everything this branch learns about within-band structure is discarded
        # at the output anyway.  It cost 46,952 params (12.5% of the model) to
        # extract fine-grained 0-4 kHz detail that the coarse output cannot
        # express.  Turning it off makes the model pure-ERB on both ends, which
        # is what the RNNoise-ERB vs GTCRN comparison is meant to isolate:
        # GTCRN has hybrid-resolution input AND a complex mask, so the two
        # models become clean opposite ends rather than a muddled middle.
        self.use_complex_input = use_complex_input
        if use_complex_input:
            self.spec_conv1 = nn.Conv1d(2, spec_conv_channels, kernel_size=5,
                                        stride=2, padding=2)
            self.spec_conv2 = nn.Conv1d(spec_conv_channels, 2 * spec_conv_channels,
                                        kernel_size=5, stride=2, padding=2)
            reduced_bins = (spec_bins + 3) // 4
            self.spec_proj = nn.Linear(2 * spec_conv_channels * reduced_bins,
                                       spec_embed_size)
            self.spec_temporal = nn.Conv1d(spec_embed_size, spec_embed_size,
                                           kernel_size=3, padding=0)
            fuse_in = cond_size + spec_embed_size
        else:
            fuse_in = cond_size
        self.fuse = nn.Linear(fuse_in, gru_size)

        # Dropout (GRU 層間)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 3 層 GRU
        self.gru1 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru2 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru3 = nn.GRU(gru_size, gru_size, batch_first=True)

        # Output: ERB-band gains only.  Speech activity is used solely as a
        # training loss weight and is not part of the runtime model contract.
        self.dense_out = nn.Linear(4 * gru_size, n_bands)

        # 初始化 GRU hidden weights 為 orthogonal
        for gru in [self.gru1, self.gru2, self.gru3]:
            for name, param in gru.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {n_params:,} parameters (dropout={dropout})")

    def forward(self, erb_features, spec_features, states=None):
        """
        erb_features:  (batch, seq_len, n_bands)
        spec_features: (batch, seq_len, 2, spec_bins), or None when the complex
              branch is disabled.  The argument stays in the signature either
              way so the ONNX graph and the C contract do not change shape with
              a training-time flag; when unused it is simply not read.
        states: [h1, h2, h3] 或 None
        回傳: gains, new_states
              seq_len' = seq_len - 2 (conv1 kernel=3 valid 減 2 frame)
        """
        if not torch.jit.is_tracing():
            if erb_features.ndim != 3:
                raise ValueError('expected ERB features [B,T,E]')
            if self.use_complex_input:
                if spec_features is None or spec_features.ndim != 4:
                    raise ValueError(
                        'use_complex_input=True requires a complex spectrum '
                        '[B,T,2,F]')
                if spec_features.shape[2:] != (2, self.spec_bins):
                    raise ValueError(
                        f'complex feature shape {tuple(spec_features.shape[2:])}, '
                        f'expected (2, {self.spec_bins})')

        device = erb_features.device
        batch, seq_len, _ = erb_features.shape

        if states is None:
            h1 = torch.zeros(1, batch, self.gru_size, device=device)
            h2 = torch.zeros(1, batch, self.gru_size, device=device)
            h3 = torch.zeros(1, batch, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        erb = torch.tanh(self.erb_conv(erb_features.permute(0, 2, 1)))
        erb = erb.permute(0, 2, 1)  # (B, T-2, cond_size)

        if self.use_complex_input:
            spec = spec_features.reshape(batch * seq_len, 2, self.spec_bins)
            spec = torch.tanh(self.spec_conv1(spec))
            spec = torch.tanh(self.spec_conv2(spec))
            spec = spec.flatten(1)
            spec = torch.tanh(self.spec_proj(spec))
            spec = spec.reshape(batch, seq_len, self.spec_embed_size)
            spec = torch.tanh(self.spec_temporal(spec.permute(0, 2, 1)))
            spec = spec.permute(0, 2, 1)  # (B, T-2, spec_embed_size)
            fused = torch.tanh(self.fuse(torch.cat([erb, spec], dim=-1)))
        else:
            fused = torch.tanh(self.fuse(erb))

        # 3 層 GRU + dropout
        gru1_out, h1 = self.gru1(fused, h1)
        gru1_out = self.dropout(gru1_out)
        gru2_out, h2 = self.gru2(gru1_out, h2)
        gru2_out = self.dropout(gru2_out)
        gru3_out, h3 = self.gru3(gru2_out, h3)

        # 本地 multi-layer skip: concat conv 與三層 GRU 輸出
        cat = torch.cat([fused, gru1_out, gru2_out, gru3_out], dim=-1)
        gains = torch.sigmoid(self.dense_out(cat))

        return gains, [h1, h2, h3]

# ============================================================
# Perceptual loss helpers (WAV-data mode)
# ============================================================

def make_norm_alpha(sr: int, hop_len: int, tau: float = 10.0) -> float:
    """Original DeepFilterNet decay calculation, including stable rounding.

    DeepFilterNet rounds from three decimal places upward until alpha is below
    one.  Besides matching its train/runtime contract, this makes alpha a fixed
    C constant instead of requiring exp() in the per-frame embedded hot path.
    """
    exact = math.exp(-(hop_len / sr) / tau)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(exact, precision)
        precision += 1
    return alpha


def stft(wav, n_fft, hop_len, win_len, window):
    """STFT with normalized=True (= fft_size^-0.5, matching DeepFilterNet-Keras
    stft(normalize=True)). Pairs with istft() for perfect reconstruction.

    center=False so that training framing equals the deployed framing.  The C
    streaming path is center=False-equivalent by construction (process.h:20-23),
    and with center=True every frame differed from deployment by n_fft/2 = 256
    samples.  That header calls it a boundary-only effect, which holds for the
    spectrum but NOT for the causal EMA normaliser: its state starts fresh at
    frame 0 and, because tau=1 s against 3 s segments means the whole segment is
    still warming up, a frame-0 difference propagates through the entire example.
    """
    return torch.stft(wav, n_fft, hop_len, win_len, window=window,
                      return_complex=True, center=False, normalized=True)


def istft(spec, n_fft, hop_len, win_len, window, length=None):
    """Weighted overlap-add synthesis, matching the C streaming path.

    Deliberately NOT torch.istft: that divides by the window envelope, which is
    exactly zero at the first and last sample once center=False (sqrt-Hann
    starts at 0 and only one frame covers the edge), so it refuses to run.  The
    deployed path has no such division -- sqrt-Hann analysis times sqrt-Hann
    synthesis is Hann, which sums to 1 at 50% overlap (COLA), so plain
    overlap-add reconstructs exactly in the interior.  The edges are genuinely
    attenuated in deployment too; ``valid_region`` below reports the span where
    that is not the case, and the loss trims to it.

    normalized=True in the analysis scales by n_fft^-0.5, so undo it here.
    """
    n_frames = spec.shape[-1]
    # normalized=True in the analysis scales by n_fft^-0.5; undo it by scaling
    # the window rather than the (n_fft, T) frame tensor -- same product, but a
    # win_len-element multiply instead of a full-size one on the gradient path.
    frames = torch.fft.irfft(spec, n=n_fft, dim=-2)          # (..., n_fft, T)
    frames = frames[..., :win_len, :] * (window * math.sqrt(n_fft)).view(-1, 1)
    out_len = (n_frames - 1) * hop_len + win_len
    wav = torch.nn.functional.fold(
        frames, output_size=(1, out_len), kernel_size=(1, win_len),
        stride=(1, hop_len),
    ).reshape(*spec.shape[:-2], out_len)
    if length is not None:
        wav = wav[..., :length] if wav.shape[-1] >= length else F.pad(
            wav, (0, length - wav.shape[-1]))
    return wav


def valid_region(win_len, hop_len):
    """Samples to trim from each end before computing a waveform loss.

    Only the first/last (win_len - hop_len) samples see fewer than the full set
    of overlapping frames, so their amplitude is a framing artefact rather than
    something the model can fix.
    """
    return win_len - hop_len


def normalize_log_erb(erb_db, norm_state=None, norm_alpha: float = 0.984,
                      init_lo_db: float = -60.0, init_hi_db: float = -90.0,
                      scale_db: float = 40.0):
    """Original DeepFilterNet causal per-band mean normalisation.

    The EMA is updated with the current frame before subtraction, matching
    Rikorose/DeepFilterNet ``libDF::band_mean_norm_erb`` exactly (verified
    against libDF/src/lib.rs and this repo's own ainr/DeepFilterNet2 port,
    neither of which clips the output). No variance or absolute-level
    side channel is used; the complex branch remains observable for stationary
    inputs and retains partial level information.
    """
    if erb_db.ndim != 3:
        raise ValueError('erb_db must have shape [B,T,E]')
    batch, n_frames, n_bands = erb_db.shape
    state_ok = norm_state is not None and norm_state.shape == (batch, 1, n_bands)
    if state_ok:
        mean = norm_state.to(device=erb_db.device, dtype=erb_db.dtype)
    else:
        mean = torch.linspace(init_lo_db, init_hi_db, n_bands,
                              device=erb_db.device, dtype=erb_db.dtype).view(1, 1, n_bands)
        mean = mean.expand(batch, 1, n_bands).clone()

    frames = []
    for t in range(n_frames):
        frame = erb_db[:, t:t + 1, :]
        mean = norm_alpha * mean + (1.0 - norm_alpha) * frame
        frames.append((frame - mean) / scale_db)
    return torch.cat(frames, dim=1), mean.detach()


def normalize_complex_spectrum(spec_low, norm_state=None, norm_alpha: float = 0.984,
                               init_lo: float = 0.001, init_hi: float = 0.0001,
                               eps: float = 1e-12):
    """Original DeepFilterNet per-bin magnitude EMA norm.

    This matches ``libDF::band_unit_norm`` exactly (verified against
    libDF/src/lib.rs and this repo's own ainr/DeepFilterNet2 port, neither of
    which clips the output): each bin updates its own state from
    ``abs(X[k])`` and divides the complex value by ``sqrt(state[k])``.  It is
    deliberately not fully gain-invariant: at steady state, scaling X by ``a``
    scales this feature by approximately ``sqrt(a)``.
    """
    if spec_low.ndim != 3:
        raise ValueError('spec_low must have shape [B,T,F]')
    batch, n_frames, n_bins = spec_low.shape
    state_ok = norm_state is not None and norm_state.shape == (batch, 1, n_bins)
    if state_ok:
        mag_mean = norm_state.to(device=spec_low.device, dtype=spec_low.real.dtype)
    else:
        mag_mean = torch.linspace(init_lo, init_hi, n_bins, device=spec_low.device,
                                  dtype=spec_low.real.dtype).view(1, 1, n_bins)
        mag_mean = mag_mean.expand(batch, 1, n_bins).clone()

    frames = []
    for t in range(n_frames):
        frame = spec_low[:, t:t + 1, :]
        mag_mean = norm_alpha * mag_mean + (1.0 - norm_alpha) * frame.abs()
        normed = frame / torch.sqrt(mag_mean + eps)
        frames.append(torch.view_as_real(normed))
    features = torch.cat(frames, dim=1).permute(0, 1, 3, 2)  # [B,T,2,F]
    return features, mag_mean.detach()


def extract_model_features(spec, erb_matrix, feature_cfg, norm_state=None,
                           return_debug=False, need_spec=True):
    """
    spec: (B, n_bins, T) complex, from normalized=True STFT.
    erb_matrix: (n_bins, n_bands) triangular forward ERB filterbank.
    Returns mean-normalised ERB features, complex low-bin features and both
    updated causal normalisation states.

    ``need_spec=False`` skips the complex branch entirely and returns ``None``
    in its place.  A pure-ERB model discards that tensor, but producing it
    costs a Python loop over every frame (the EMA is causal), so the training
    loop must not pay for it.  The C path and the ONNX signature are unaffected
    -- they are separate code, and the default stays True so every other caller
    (parity tests, denoise) is unchanged.
    """
    spec_btf = spec.permute(0, 2, 1)
    energy = spec_btf.abs().pow(2) @ erb_matrix
    erb_db = 10.0 * torch.log10(energy + 1e-10)
    if norm_state is not None and not isinstance(norm_state, dict):
        raise ValueError('norm_state must be None or a dict with erb/spec entries')
    erb_state_in = None if norm_state is None else norm_state.get('erb')
    spec_state_in = None if norm_state is None else norm_state.get('spec')
    erb_features, erb_state = normalize_log_erb(
        erb_db, norm_state=erb_state_in, norm_alpha=feature_cfg['erb_alpha'],
        init_lo_db=feature_cfg['erb_norm_init_lo_db'],
        init_hi_db=feature_cfg['erb_norm_init_hi_db'],
        scale_db=feature_cfg['erb_norm_scale_db'])
    spec_low = spec_btf[..., :feature_cfg['spec_bins']]
    if need_spec or return_debug:
        spec_features, norm_state = normalize_complex_spectrum(
            spec_low, norm_state=spec_state_in, norm_alpha=feature_cfg['spec_alpha'],
            init_lo=feature_cfg['spec_norm_init_lo'],
            init_hi=feature_cfg['spec_norm_init_hi'], eps=feature_cfg['spec_norm_eps'])
        if not need_spec:
            spec_features = None
    else:
        spec_features, norm_state = None, spec_state_in
    debug = None
    if return_debug:
        debug = {'erb_db': erb_db.detach(), 'spec_magnitude': spec_low.abs().detach()}
    return erb_features, spec_features, {'erb': erb_state, 'spec': norm_state}, debug


def apply_erb_gains_batch(noisy_spec, gains, erb_inv, lookahead=0):
    """
    noisy_spec : (B, n_bins, n_frames) complex
    gains      : (B, n_frames_out, n_bands)
    erb_inv    : (n_bins, n_bands) mode=1 inverse ERB FB (no edge x2, partition of unity)
    Keras mask expansion: bin_gains = gains @ erb_inv.T (= mask @ iERB_Matrix). Because
    mode=1 is a partition of unity, gains=1 → bin_gains=1 with NO row normalisation.

    Mode 自動判定:
    - n_frames_out == n_frames : padded 模式 (training), gains 與 spec frame 1:1 對齊
    - 否則                      : streaming 模式 (inference), 邊界 frame 留 gain=1
    """
    B, n_bins, n_frames = noisy_spec.shape
    n_frames_out = gains.shape[1]
    bin_g = (gains @ erb_inv.t()).transpose(1, 2)         # (B, n_bins, n_frames_out)
    if n_frames_out == n_frames:
        return noisy_spec * bin_g
    bin_gains = torch.ones(B, n_bins, n_frames,
                           device=gains.device, dtype=gains.dtype)
    off = 2 - lookahead
    bin_gains[:, :, off:off + n_frames_out] = bin_g
    return noisy_spec * bin_gains


class _SafeAngle(torch.autograd.Function):
    """DeepFilterNet's angle op with finite gradients at zero magnitude."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        inv_power = grad / (x.real.square() + x.imag.square()).clamp_min(1e-10)
        return torch.complex(-x.imag * inv_power, x.real * inv_power)


class _LossStft(nn.Module):
    """STFT module used by DeepFilterNet's loss, including its cached window."""

    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, waveform):
        return torch.stft(
            waveform, self.n_fft, self.hop, window=self.window,
            normalized=True, return_complex=True)


class ErbIrmLoss(nn.Module):
    """Direct supervision of the 22 ERB band gains against the ideal ratio mask.

    Why this exists: with MultiResSpecLoss alone the gains are only supervised
    *through* the synthesis, so nothing pushes a specific gain value in a band
    the waveform loss barely notices.  This trains them directly.

    target  g* = sqrt(clamp(E_clean / E_noisy, 0, 1))   per band, amplitude domain
    loss    mean( (g^gamma - g*^gamma)^2 )  over DEFINED bands only

    NO speech-activity weighting.  A ``(1 + 5*vad)`` term used to sit here by
    analogy with xiph/rnnoise, but xiph computes it from a real VAD *output
    head* trained against a VAD target; this model has no such head and emits
    nothing but 22 band gains, so the term was weighting a loss for a quantity
    the network never predicts.  It was also inverted in the case that matters
    most: for a noise-only example ``clean_frame_e`` is 0 and the peak
    reference clamps to 1e-12, so ``10*log10(0/1e-12)`` evaluated to 0 dB --
    above any sane threshold -- and every pure-noise frame was scored as
    speech-active and up-weighted 6x.

    The undefined-band mask is kept: a band with essentially no energy in the
    MIXTURE has no defined ratio at all, so its gradient is pure noise rather
    than a weighting choice.
    """

    def __init__(self, factor=1.0, gamma=0.25, energy_floor=1e-9):
        super().__init__()
        self.factor = factor
        self.gamma = gamma
        self.energy_floor = energy_floor

    @staticmethod
    def _band_energy(spec, erb_fwd):
        """spec: (B, n_bins, T) complex; erb_fwd: (n_bins, n_bands)
        -> (B, T, n_bands)"""
        power = spec.real.pow(2) + spec.imag.pow(2)      # (B, n_bins, T)
        return power.transpose(1, 2).matmul(erb_fwd)     # (B, T, n_bands)

    def forward(self, pred_gains, noisy_spec, clean_spec, erb_fwd):
        if self.factor == 0:
            return torch.zeros((), device=pred_gains.device, dtype=pred_gains.dtype)

        e_noisy = self._band_energy(noisy_spec, erb_fwd)
        e_clean = self._band_energy(clean_spec, erb_fwd)

        # Align the gain's frame axis with the spectra (the model may emit
        # fewer frames than the STFT produced, depending on lookahead padding).
        n = min(pred_gains.shape[1], e_noisy.shape[1])
        pred_gains = pred_gains[:, :n]
        e_noisy, e_clean = e_noisy[:, :n], e_clean[:, :n]

        target = torch.sqrt(torch.clamp(e_clean / (e_noisy + 1e-10), 0.0, 1.0))

        # Undefined-band mask: where the band carries essentially no energy in
        # the mixture, the ratio is numerically meaningless and its gradient is
        # pure noise.  The original RNNoise masks these out rather than letting
        # them dominate (most bands are near-silent at low SNR).
        defined = (e_noisy > self.energy_floor).to(pred_gains.dtype)

        weight = defined

        g = pred_gains.clamp_min(1e-12).pow(self.gamma)
        t = target.clamp_min(1e-12).pow(self.gamma)
        num = (weight * (g - t).pow(2)).sum()
        den = weight.sum().clamp_min(1.0)
        return self.factor * num / den


class MultiResSpecLoss(nn.Module):
    """
    DeepFilterNet 3 MultiResSpecLoss port. Each resolution uses a plain Hann
    window, hop=n_fft//4 and normalized STFT. Both magnitude and complex spectra
    are gamma-compressed before MSE, and resolution losses are summed (not
    averaged), exactly like the upstream implementation.

    There is no division by clean energy, activity weighting, or IRM target, so
    clean=0 (pure-noise samples) is a normal finite target.
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
            Y = stft_fn(enhanced)
            S = stft_fn(clean)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            total = total + F.mse_loss(Y_abs, S_abs) * self.factor
            if self.factor_complex != 0:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * _SafeAngle.apply(Y))
                    S = S_abs * torch.exp(1j * _SafeAngle.apply(S))
                total = total + F.mse_loss(
                    torch.view_as_real(Y),
                    torch.view_as_real(S),
                ) * self.factor_complex
        return total


# ============================================================
# 訓練
# ============================================================


def train(args):
    # Seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Load config
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # Signal params
    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)
    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    MIN_BINS_PER_BAND = cfg.getint('signal', 'min_bins_per_band', fallback=2)

    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        raise NotImplementedError(
            "hybrid bands are not supported with the faithful DFN/Keras ERB "
            "filterbank; set hybrid_cutoff_hz=0 to use pure ERB")
    N_BANDS = cfg.getint('signal', 'n_bands')

    LOOKAHEAD = cfg.getint('signal', 'lookahead_frames', fallback=0)
    assert 0 <= LOOKAHEAD <= 2, "lookahead_frames 只支援 0~2"
    FEATURE_CFG = read_feature_config(cfg, SR, HOP_LEN, N_FFT, WIN_LEN)

    # Training params
    epochs = cfg.getint('training', 'epochs')
    batch_size = cfg.getint('training', 'batch_size')
    lr = cfg.getfloat('training', 'lr')
    mmap_block_size = cfg.getint('training', 'mmap_block_size', fallback=256)
    mmap_workers = cfg.getint('training', 'mmap_num_workers', fallback=2)
    prefetch_factor = cfg.getint('training', 'prefetch_factor', fallback=2)
    if mmap_workers < 0:
        raise ValueError("mmap_num_workers cannot be negative")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be greater than zero")
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device_str = args.device or cfg.get('training', 'device', fallback='cpu')
        device = torch.device(device_str)
    output_dir = cfg.get('paths', 'output_dir')

    # ERB band borders (faithful DeepFilterNet/Keras ERBBand, config-driven)
    NFFTBORDER = erb_bandborder(N_BANDS, SR, N_FFT, MIN_BINS_PER_BAND)  # (N_BANDS,) ints, [0 .. n_fft//2+1]

    # Dataset
    use_online = False
    use_wav = False
    if args.packed_dir or args.packed_data:
        pt_files = []
        if args.packed_dir:
            pt_files += sorted(glob.glob(os.path.join(args.packed_dir, '*.pt')))
        if args.packed_data:
            pt_files += args.packed_data
        if not pt_files:
            raise FileNotFoundError(f"在 {args.packed_dir} 找不到任何 .pt 檔案")
        parts = [
            PackedDataset(p, mmap=args.mmap, expected_sr=SR) for p in pt_files
        ]
        dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]
        use_wav = True
    else:
        raise ValueError("RNNoise-ERB 訓練僅支援 wav-data：請用 --packed-dir 或 --packed-data")

    n_train, n_val = split_sizes(dataset)
    train_set, val_set = locality_preserving_random_split(
        dataset, n_train, n_val, args.seed)
    print(f"  split: {n_train} train / {n_val} val "
          f"(shared fraction, split seed {args.seed})")

    # epoch_size 可限制每 epoch 的 sample 數；mmap 模式仍以局部區塊取樣。
    epoch_size = cfg.getint('training', 'epoch_size', fallback=0)
    # RAM-backed packed tensors do not need worker processes. mmap uses a small
    # worker pool and shallow prefetch so it does not consume shared-server RAM.
    n_workers = mmap_workers if args.mmap else 0
    pin_memory = device.type == 'cuda'
    common_kwargs = dataloader_worker_kwargs(
        n_workers, pin_memory, prefetch_factor
    )
    sample_count = epoch_size if 0 < epoch_size < len(train_set) else None

    if args.mmap:
        train_sampler = BlockShuffleSampler(
            train_set, block_size=mmap_block_size, num_samples=sample_count
        )
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, **common_kwargs)
    elif sample_count is not None:
        train_sampler = RandomSampler(train_set, replacement=False, num_samples=epoch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, **common_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, **common_kwargs)

    val_workers = min(n_workers, 2)
    val_kwargs = dataloader_worker_kwargs(
        val_workers, pin_memory, prefetch_factor
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, **val_kwargs)

    # Regularization
    dropout = cfg.getfloat('training', 'dropout', fallback=0.0)
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=0.05)
    # DFN-style weight-decay schedule: 若 weight_decay_end >= 0 → cosine 從 weight_decay
    # 排到 weight_decay_end (無 warmup);否則 (-1) = 常數 weight_decay。
    weight_decay_end = cfg.getfloat('training', 'weight_decay_end', fallback=-1.0)
    wd_scheduled = weight_decay_end >= 0.0
    min_lr = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    warmup_epochs = cfg.getint('training', 'warmup_epochs', fallback=3)
    patience = cfg.getint('training', 'early_stop_patience', fallback=0)

    # 模型 (容量 config-driven; 官方 v0.2 = 128/256)
    COND_SIZE = cfg.getint('model', 'cond_size', fallback=64)
    USE_COMPLEX_INPUT = cfg.getboolean('model', 'use_complex_input', fallback=False)
    GRU_SIZE = cfg.getint('model', 'gru_size', fallback=128)
    SPEC_CONV_CHANNELS = cfg.getint('model', 'spec_conv_channels', fallback=8)
    SPEC_EMBED_SIZE = cfg.getint('model', 'spec_embed_size', fallback=64)
    model = RNNoiseModel(
        n_bands=N_BANDS, spec_bins=FEATURE_CFG['spec_bins'],
        cond_size=COND_SIZE, gru_size=GRU_SIZE,
        spec_conv_channels=SPEC_CONV_CHANNELS,
        spec_embed_size=SPEC_EMBED_SIZE, dropout=dropout,
        use_complex_input=USE_COMPLEX_INPUT).to(device)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay, amsgrad=True)
    # Linear warmup → cosine annealing (DFN2 style)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, warmup_epochs * len(train_loader))
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_steps])

    loss_cfg = read_loss_config(cfg)
    loss_fn = MultiResSpecLoss(**loss_cfg).to(device)
    irm_cfg = read_irm_loss_config(cfg)
    irm_loss_fn = ErbIrmLoss(**irm_cfg).to(device)

    # Forward ERBB (mode=0, edge x2) for features; inverse (mode=1, partition of unity)
    # for mask→bin expansion — exactly the DFN/Keras forward/inverse split.
    ERB_FWD = torch.from_numpy(
        compute_erb_matrix(NFFTBORDER, N_FFT, mode=0)).to(device)  # (n_bins, n_bands)
    ERB_INV = torch.from_numpy(
        compute_erb_matrix(NFFTBORDER, N_FFT, mode=1)).to(device)  # (n_bins, n_bands)
    print(f"  feature={FEATURE_CFG['version']}")
    print(f"  ERB mean norm: alpha={FEATURE_CFG['erb_alpha']:.6f} "
          f"(tau={FEATURE_CFG['erb_tau_sec']:g}s), "
          f"init={FEATURE_CFG['erb_norm_init_lo_db']:g}.."
          f"{FEATURE_CFG['erb_norm_init_hi_db']:g}dB, "
          f"scale={FEATURE_CFG['erb_norm_scale_db']:g}dB")
    print(f"  complex: 0..{FEATURE_CFG['spec_max_hz']}Hz "
          f"({FEATURE_CFG['spec_bins']} bins), per-bin unit-norm alpha="
          f"{FEATURE_CFG['spec_alpha']:.6f} (tau={FEATURE_CFG['spec_tau_sec']:g}s)")

    # Window for on-the-fly STFT (wav-data mode) — created once, moved to device
    stft_window = torch.sqrt(torch.hann_window(WIN_LEN)).to(device)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    no_improve_count = 0  # early stopping 計數器

    # Resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_feature_config(ckpt, FEATURE_CFG, context=args.resume)
        require_checkpoint_loss_config(ckpt, loss_cfg, irm_cfg, context=args.resume)
        resume_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        resume_model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('loss', float('inf')))
        print(f"  Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.5f}")

    print(f"Training: SR={SR}, N_FFT={N_FFT}, N_BANDS={N_BANDS}")
    print(f"  WIN_LEN={WIN_LEN}, HOP_LEN={HOP_LEN} (root Hann window)")
    print(f"  lookahead_frames={LOOKAHEAD} ({LOOKAHEAD * HOP_LEN / SR * 1000:.1f} ms extra latency)")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")
    if args.mmap:
        print(f"  mmap: block={mmap_block_size}, workers={n_workers}, "
              f"prefetch={prefetch_factor}, packed_dtype_preserved=True")
    wd_note = f"{weight_decay}→{weight_decay_end} (cosine)" if wd_scheduled else f"{weight_decay} (const)"
    print(f"  dropout={dropout}, weight_decay={wd_note}")
    print(f"  loss={LOSS_VERSION}: fft_sizes={loss_cfg['fft_sizes']}, "
          f"gamma={loss_cfg['gamma']}, magnitude_factor={loss_cfg['factor']}, "
          f"complex_factor={loss_cfg['factor_complex']}")
    if patience > 0:
        print(f"  early stopping: patience={patience}")
    print(f"  device={device}")

    # 學習曲線 CSV (epoch,train_loss,val_loss,lr)。resume 時保留舊紀錄 (append)，
    # 否則寫 header。畫圖用 plot_curve.py。
    log_csv = os.path.join(output_dir, 'train_log.csv')
    if not os.path.isfile(log_csv):
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr\n')

    mrsl_enabled = mrsl_is_enabled(loss_cfg)

    def batch_loss(noisy_wav, clean_wav):
        """One forward pass and its objective, shared by the train and val loops.

        These were two verbatim copies sixty lines apart, so every change to the
        objective had to be made twice.
        """
        noisy_spec = stft(noisy_wav, N_FFT, HOP_LEN, WIN_LEN, stft_window)

        # Zero-pad time so both k=3 branches emit one gain per spectrum frame:
        # pad_left = 2-LOOKAHEAD (過去), pad_right = LOOKAHEAD (未來).
        # Dataset clips are independent and shuffled, so no normalisation/GRU
        # state is carried across files; each call still evolves causally
        # across every frame within the clip.
        erb_features, spec_features, _, _ = extract_model_features(
            noisy_spec, ERB_FWD, FEATURE_CFG, need_spec=model.use_complex_input)
        pad_left, pad_right = 2 - LOOKAHEAD, LOOKAHEAD
        erb_features = F.pad(erb_features, (0, 0, pad_left, pad_right))
        if spec_features is not None:
            spec_features = F.pad(
                spec_features, (0, 0, 0, 0, pad_left, pad_right))
        pred_gains, _ = model(erb_features, spec_features)

        # Direct supervision of the band gains against the ideal ratio mask.
        clean_spec = stft(clean_wav, N_FFT, HOP_LEN, WIN_LEN, stft_window)
        loss = irm_loss_fn(pred_gains, noisy_spec, clean_spec, ERB_FWD)
        if not mrsl_enabled:
            # With MRSL off the synthesis is not on the gradient path at all,
            # so the gain application and the ISTFT are skipped rather than
            # computed and discarded.
            return loss

        enhanced_spec = apply_erb_gains_batch(
            noisy_spec, pred_gains, ERB_INV, LOOKAHEAD)
        enhanced_wav = istft(enhanced_spec, N_FFT, HOP_LEN, WIN_LEN, stft_window)

        # Trim the partial-overlap edges: with center=False only the interior
        # sees the full set of overlapping frames, and the taper there is a
        # framing artefact the model cannot fix (deployment has it too).
        trim = valid_region(WIN_LEN, HOP_LEN)
        n = min(enhanced_wav.size(-1), clean_wav.size(-1))
        return loss + loss_fn(enhanced_wav[..., trim:n - trim],
                              clean_wav[..., trim:n - trim])

    global_step = (start_epoch - 1) * len(train_loader)   # resume-safe (給 wd schedule)
    last_wd = weight_decay
    for epoch in range(start_epoch, epochs + 1):
        # 每 epoch 重新 shuffle dataset indices (僅 online 模式)
        if use_online:
            dataset._shuffle_indices()

        # --- Train ---
        model.train()
        train_loss_sum = 0
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            if use_wav:
                for noisy_wav, clean_wav in pbar:
                    noisy_wav = noisy_wav.to(
                        device=device, dtype=torch.float32, non_blocking=pin_memory)
                    clean_wav = clean_wav.to(
                        device=device, dtype=torch.float32, non_blocking=pin_memory)

                    loss = batch_loss(noisy_wav, clean_wav)

                    # weight-decay 排程 (DFN-style cosine, 套在這步 optimizer.step 前)
                    if wd_scheduled:
                        t = min(global_step, total_steps - 1) / max(total_steps - 1, 1)
                        last_wd = float(weight_decay_end + 0.5 * (weight_decay - weight_decay_end)
                                        * (1.0 + np.cos(np.pi * t)))
                        for pg in optimizer.param_groups:
                            pg['weight_decay'] = last_wd

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    train_loss_sum += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_train = train_loss_sum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            if use_wav:
                for noisy_wav, clean_wav in val_loader:
                    noisy_wav = noisy_wav.to(
                        device=device, dtype=torch.float32, non_blocking=pin_memory)
                    clean_wav = clean_wav.to(
                        device=device, dtype=torch.float32, non_blocking=pin_memory)

                    loss = batch_loss(noisy_wav, clean_wav)
                    val_loss_sum += loss.item()

        avg_val = val_loss_sum / max(len(val_loader), 1)
        cur_lr = scheduler.get_last_lr()[0]
        wd_str = f"  wd={last_wd:.2e}" if wd_scheduled else ""
        print(f"  train_loss={avg_train:.5f}  val_loss={avg_val:.5f}  lr={cur_lr:.2e}{wd_str}")
        with open(log_csv, 'a') as f:
            f.write(f'{epoch},{avg_train:.6f},{avg_val:.6f},{cur_lr:.6e}\n')

        # 儲存 checkpoint (compiled model 要取 _orig_mod 避免 key 有 _orig_mod. 前綴)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        is_best = avg_val < best_val_loss
        checkpoint_best = min(best_val_loss, avg_val)
        ckpt = {
            'epoch': epoch,
            'state_dict': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_val,
            'best_val_loss': checkpoint_best,
            'nfftborder': NFFTBORDER.tolist(),
            'feature_version': FEATURE_VERSION,
            'loss_version': LOSS_VERSION,
            'config': {
                'sr': SR, 'n_fft': N_FFT, 'win_len': WIN_LEN,
                'hop_len': HOP_LEN, 'n_bands': N_BANDS,
                'lookahead_frames': LOOKAHEAD,
                'cond_size': COND_SIZE, 'gru_size': GRU_SIZE,
                'use_complex_input': USE_COMPLEX_INPUT,
                'spec_conv_channels': SPEC_CONV_CHANNELS,
                'spec_embed_size': SPEC_EMBED_SIZE,
                'feature_version': FEATURE_VERSION,
                'loss_version': LOSS_VERSION,
                'loss_fft_sizes': ','.join(str(n) for n in loss_cfg['fft_sizes']),
                'loss_gamma': loss_cfg['gamma'],
                'loss_factor': loss_cfg['factor'],
                'loss_factor_complex': loss_cfg['factor_complex'],
                'irm_factor': irm_cfg['factor'],
                'irm_gamma': irm_cfg['gamma'],
                'irm_energy_floor': irm_cfg['energy_floor'],
                'min_bins_per_band': MIN_BINS_PER_BAND,
                'feature_erb_norm_tau_sec': FEATURE_CFG['erb_tau_sec'],
                'feature_erb_norm_alpha': FEATURE_CFG['erb_alpha'],
                'feature_erb_norm_init_lo_db': FEATURE_CFG['erb_norm_init_lo_db'],
                'feature_erb_norm_init_hi_db': FEATURE_CFG['erb_norm_init_hi_db'],
                'feature_erb_norm_scale_db': FEATURE_CFG['erb_norm_scale_db'],
                'feature_spec_max_hz': FEATURE_CFG['spec_max_hz'],
                'feature_spec_bins': FEATURE_CFG['spec_bins'],
                'feature_spec_norm_tau_sec': FEATURE_CFG['spec_tau_sec'],
                'feature_spec_norm_alpha': FEATURE_CFG['spec_alpha'],
                'feature_spec_norm_init_lo': FEATURE_CFG['spec_norm_init_lo'],
                'feature_spec_norm_init_hi': FEATURE_CFG['spec_norm_init_hi'],
                'feature_spec_norm_eps': FEATURE_CFG['spec_norm_eps'],
            },
        }
        torch.save(ckpt, os.path.join(output_dir, f'rnnoise_epoch{epoch}.pth'))

        if is_best:
            best_val_loss = avg_val
            no_improve_count = 0
            torch.save(ckpt, os.path.join(output_dir, 'rnnoise_best.pth'))
            print(f"  ✓ best model saved (val_loss={avg_val:.5f})")
        else:
            no_improve_count += 1
            if patience > 0:
                print(f"  no improvement {no_improve_count}/{patience}")
                if no_improve_count >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNNoise v0.2-inspired 訓練 (config-driven, ERB bands)')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--device', default=None,
                        help='覆蓋 config 中的 device 設定')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定 GPU ID (例: --gpu 0)')
    parser.add_argument('--packed-dir', default=None,
                        help='包含 .pt 檔的資料夾，自動掃描全部 '
                             '(../dataset_gen/pack_dataset.py 產生)')
    parser.add_argument('--packed-data', default=None, nargs='+',
                        help='指定 .pt 檔，可多個；與 --packed-dir 可同時使用')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint 路徑，從斷點續訓')
    parser.add_argument('--mmap', action='store_true',
                        help='Memory-map .pt tensors (low RAM, disk-backed; needs PyTorch>=2.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42, 設 -1 關閉)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    train(args)
