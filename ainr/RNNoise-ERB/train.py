import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
RNNoise v0.2-inspired 噪音抑制模型 — 訓練腳本
採用 Conv+GRU 骨架，並改為 config-driven / ERB-only / 無 pitch 的本地架構

用法:
    python train.py --config config.ini
    python train.py --config config.ini --device cpu
    python train.py --config config.ini --resume output/rnnoise_epoch5.pth
    python train.py --config config.ini --seed 123
"""

import argparse
import configparser
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, random_split
import tqdm

from dataset import PackedDataset


# Feature semantics are intentionally versioned because old and new models both use
# 22 inputs while assigning completely different meanings to those 22 values.  Never
# silently load a checkpoint trained with the former per-band temporal EMA.
FEATURE_VERSION = 'log_erb_shared_online_cmvn_v1'


def require_checkpoint_feature_version(ckpt, context='checkpoint'):
    """Reject checkpoints whose 22-D input semantics do not match this code."""
    version = ckpt.get('feature_version', ckpt.get('config', {}).get('feature_version'))
    if version != FEATURE_VERSION:
        shown = repr(version) if version is not None else 'missing (legacy per-band EMA)'
        raise ValueError(
            f"{context} feature_version={shown}, expected {FEATURE_VERSION!r}. "
            "Feature dimensions match but semantics do not; retrain this model."
        )


def require_checkpoint_feature_config(ckpt, feature_cfg, context='checkpoint'):
    """Require the checkpoint and runtime to use identical normaliser constants."""
    require_checkpoint_feature_version(ckpt, context=context)
    saved = ckpt.get('config', {})
    expected = {
        'feature_norm_tau_sec': feature_cfg['tau_sec'],
        'feature_mean_init_db': feature_cfg['mean_init_db'],
        'feature_std_init_db': feature_cfg['var_init_db2'] ** 0.5,
        'feature_std_floor_db': feature_cfg['var_floor_db2'] ** 0.5,
        'feature_clip': feature_cfg['clip'],
    }
    import math
    for key, want in expected.items():
        if key not in saved or not math.isclose(float(saved[key]), float(want),
                                                 rel_tol=1e-7, abs_tol=1e-7):
            got = saved.get(key, 'missing')
            raise ValueError(
                f"{context} {key}={got!r}, but runtime config requires {want!r}; "
                "use the training config or retrain before inference/export."
            )


def read_feature_config(cfg, sr, hop_len):
    """Read the runtime-normalisation contract shared by train/denoise/C."""
    version = cfg.get('feature', 'version', fallback=FEATURE_VERSION)
    if version != FEATURE_VERSION:
        raise ValueError(
            f"config feature version {version!r} is unsupported; expected {FEATURE_VERSION!r}"
        )
    tau_sec = cfg.getfloat('feature', 'norm_tau_sec', fallback=10.0)
    mean_init_db = cfg.getfloat('feature', 'mean_init_db', fallback=-75.0)
    std_init_db = cfg.getfloat('feature', 'std_init_db', fallback=20.0)
    std_floor_db = cfg.getfloat('feature', 'std_floor_db', fallback=6.0)
    clip = cfg.getfloat('feature', 'clip', fallback=5.0)
    if tau_sec <= 0 or std_init_db <= 0 or std_floor_db <= 0 or clip <= 0:
        raise ValueError('feature tau/std/clip values must all be positive')
    alpha = make_norm_alpha(sr, hop_len, tau_sec)
    return dict(
        version=version,
        tau_sec=tau_sec,
        alpha=alpha,
        mean_init_db=mean_init_db,
        var_init_db2=std_init_db * std_init_db,
        var_floor_db2=std_floor_db * std_floor_db,
        clip=clip,
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

def compute_erb_bands(n_fft, sr, n_bands, min_bins_per_band=2):
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


def compute_hybrid_bands(n_fft, sr, n_erb_high, cutoff_hz, min_bins_per_band=2):
    """
    Hybrid frequency bands (ref: GTCRN):
      - 0 ~ cutoff_hz: 每個 FFT bin 一個 band (原始解析度, 刻意 1 bin/band)
      - cutoff_hz ~ Nyquist: n_erb_high 個 ERB bands (套 min_bins_per_band)
    回傳: bin_edges (n_bands+1,), n_bands
    """
    n_bins = n_fft // 2 + 1
    bin_res = sr / n_fft
    cutoff_bin = int(round(cutoff_hz / bin_res))
    cutoff_bin = min(cutoff_bin, n_bins - 1)

    # Part 1: individual bins [0, 1, 2, ..., cutoff_bin] (high resolution, 故意)
    low_edges = list(range(cutoff_bin + 1))

    # Part 2: ERB bands above cutoff (greedy forward with min_bins)
    e_cut = erb_rate(cutoff_hz)
    e_high = erb_rate(sr / 2)
    erb_edges = np.linspace(e_cut, e_high, n_erb_high + 1)
    freq_edges = erb_inv(erb_edges)
    ideal = np.round(freq_edges / bin_res).astype(int)
    ideal = np.clip(ideal, cutoff_bin, n_bins)

    high = [cutoff_bin]
    for i in range(n_erb_high):
        next_edge = max(int(ideal[i + 1]), high[-1] + min_bins_per_band)
        next_edge = min(next_edge, n_bins)
        high.append(next_edge)
    high[-1] = n_bins

    # 合併: low_edges[-1] == high[0] == cutoff_bin
    all_edges = np.array(low_edges + list(high[1:]), dtype=int)
    n_bands = len(all_edges) - 1
    return all_edges, n_bands


def freq2erb(freq_hz):
    """Hz → ERB-number (Glasberg-Moore; DeepFilterNet / DeepFilterNet-Keras constants)."""
    return 9.265 * np.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """ERB-number → Hz (inverse of freq2erb)."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft):
    """Faithful port of DeepFilterNet(-Keras) ERBBand(): returns nfftborder, a length-N
    (N = n_bands) int array of FFT-bin band borders on the Glasberg-Moore ERB scale.
    nfftborder[0]=0 (DC), nfftborder[-1]=n_fft//2+1 (Nyquist+1). N borders → N ERB bands
    (one matrix column each):
        cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N))
        border  = round((cutoff + bw/2) / bw),   bw = (sr/2)/(n_fft/2) = sr/n_fft
    then Keras's 'every-other-band span >= 2 bins' enforcement.
    """
    high_lim = sr / 2.0
    bw = high_lim / (n_fft / 2.0)                       # freqRangePerBin = sr / n_fft
    erb_lims = np.linspace(freq2erb(0.0), freq2erb(high_lim), n_bands)
    cutoffs = erb2freq(erb_lims)
    nb = np.round((cutoffs + bw / 2.0) / bw)
    for i in range(n_bands - 2):
        if (nb[i + 2] - nb[i]) < 2:
            nb[i + 2] += (2 - (nb[i + 2] - nb[i]))
    # Pin endpoints to DC and Nyquist+1 = n_bins (Keras's intent; at 48k the rounding
    # lands on n_bins exactly, but at other sr/n_fft it can fall a bin short and leave
    # the Nyquist bin uncovered → mode1 partition of unity would break).
    nb[0] = 0
    nb[-1] = n_fft // 2 + 1
    return nb.astype(int)


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
    RNNoise v0.2-inspired: Conv1d 前處理 + 3 層 GRU。
    本地差異包含 ERB-only input、conv2 k=1、concat 各層輸出、無 VAD/稀疏化。
    """
    def __init__(self, n_bands, cond_size=64, gru_size=128, dropout=0.0):
        super().__init__()
        self.n_bands = n_bands
        self.gru_size = gru_size

        # Conv1d 前處理 (k=3 + k=1, 減少 latency)
        self.conv1 = nn.Conv1d(n_bands, cond_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=1)

        # Dropout (GRU 層間)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 3 層 GRU
        self.gru1 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru2 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru3 = nn.GRU(gru_size, gru_size, batch_first=True)

        # 輸出: concat(conv_out, gru1, gru2, gru3) → gains
        self.dense_out = nn.Linear(4 * gru_size, n_bands)

        # 初始化 GRU hidden weights 為 orthogonal
        for gru in [self.gru1, self.gru2, self.gru3]:
            for name, param in gru.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {n_params:,} parameters (dropout={dropout})")

    def forward(self, x, states=None):
        """
        x: (batch, seq_len, n_bands)
        states: [h1, h2, h3] 或 None
        回傳: gains (batch, seq_len', n_bands), new_states
              seq_len' = seq_len - 2 (conv1 kernel=3 valid 減 2 frame)
        """
        device = x.device
        batch = x.size(0)

        if states is None:
            h1 = torch.zeros(1, batch, self.gru_size, device=device)
            h2 = torch.zeros(1, batch, self.gru_size, device=device)
            h3 = torch.zeros(1, batch, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        # Conv1d 前處理: (B, T, C) → (B, C, T) → conv → (B, C, T') → (B, T', C)
        tmp = x.permute(0, 2, 1)
        tmp = torch.tanh(self.conv1(tmp))
        tmp = torch.tanh(self.conv2(tmp))
        conv_out = tmp.permute(0, 2, 1)  # (B, T-2, gru_size)

        # 3 層 GRU + dropout
        gru1_out, h1 = self.gru1(conv_out, h1)
        gru1_out = self.dropout(gru1_out)
        gru2_out, h2 = self.gru2(gru1_out, h2)
        gru2_out = self.dropout(gru2_out)
        gru3_out, h3 = self.gru3(gru2_out, h3)

        # 本地 multi-layer skip: concat conv 與三層 GRU 輸出
        cat = torch.cat([conv_out, gru1_out, gru2_out, gru3_out], dim=-1)
        # Plain sigmoid (DFN3 style): 軟輸出, sharpening 留給 inference-only post-filter
        gains = torch.sigmoid(self.dense_out(cat))

        return gains, [h1, h2, h3]

# ============================================================
# Perceptual loss helpers (WAV-data mode)
# ============================================================

def make_norm_alpha(sr: int, hop_len: int, tau: float = 10.0) -> float:
    """Causal runtime-statistics decay: alpha = exp(-frame_period/tau)."""
    import math
    return math.exp(-(hop_len / sr) / tau)


def stft(wav, n_fft, hop_len, win_len, window):
    """STFT with normalized=True (= fft_size^-0.5, matching DeepFilterNet-Keras
    stft(normalize=True)). Pairs with istft() for perfect reconstruction."""
    return torch.stft(wav, n_fft, hop_len, win_len, window=window,
                      return_complex=True, center=True, normalized=True)


def istft(spec, n_fft, hop_len, win_len, window, length):
    """Inverse of stft() (normalized=True → istft normalized=True inverts it exactly)."""
    return torch.istft(spec, n_fft, hop_len, win_len,
                       window=window, length=length, normalized=True)


def normalize_log_erb(erb_db, norm_state=None, norm_alpha: float = 0.9984,
                      mean_init_db: float = -75.0,
                      var_init_db2: float = 400.0,
                      var_floor_db2: float = 36.0,
                      clip: float = 5.0):
    """
    Normalise log-ERB values with ONE broadband runtime mean/variance per stream.

    The same scalar statistics are shared by every ERB band, so runtime gain drift is
    removed while a stationary signal's cross-band spectral envelope remains visible.
    Features are computed from the previous state, then the state is updated causally.

    erb_db: (..., T, C), norm_state: {'mean': (B, 1), 'var': (B, 1)} or None.
    """
    shape = erb_db.shape
    x = erb_db.reshape(-1, shape[-2], shape[-1])            # (B, T, C)
    batch = x.size(0)

    state_ok = (
        norm_state is not None and
        'mean' in norm_state and 'var' in norm_state and
        norm_state['mean'].shape == (batch, 1) and
        norm_state['var'].shape == (batch, 1)
    )
    if state_ok:
        mean = norm_state['mean'].to(device=x.device, dtype=x.dtype)
        var = norm_state['var'].to(device=x.device, dtype=x.dtype)
    else:
        mean = torch.full((batch, 1), mean_init_db, device=x.device, dtype=x.dtype)
        var = torch.full((batch, 1), var_init_db2, device=x.device, dtype=x.dtype)

    out = []
    for t in range(x.size(1)):
        frame = x[:, t, :]
        denom = torch.sqrt(var + var_floor_db2)
        out.append(torch.clamp((frame - mean) / denom, -clip, clip))

        level = frame.mean(dim=-1, keepdim=True)
        delta = level - mean
        new_mean = mean + (1.0 - norm_alpha) * delta
        var = norm_alpha * var + (1.0 - norm_alpha) * delta * (level - new_mean)
        mean = new_mean
        var = var.clamp_min(0.0)

    features = torch.stack(out, dim=1).reshape(shape)
    return features, {'mean': mean.detach(), 'var': var.detach()}


def extract_erb_features(power_spec, erb_matrix, norm_state=None, **norm_kwargs):
    """
    power_spec: (..., T, n_bins), from normalized=True STFT.
    erb_matrix: (n_bins, n_bands) triangular forward ERB filterbank.
    Returns (features, updated_state), with features shape (..., T, n_bands).
    """
    energy = power_spec @ erb_matrix
    erb_db = 10.0 * torch.log10(energy + 1e-10)
    return normalize_log_erb(erb_db, norm_state=norm_state, **norm_kwargs)


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


def multi_res_stft_loss(enhanced, clean, fft_sizes=(512, 256, 1024), gamma=0.3,
                        factor=1.0, f_under=1.0):
    """
    DeepFilterNet MultiResSpecLoss 忠實 port (df/loss.py)。每個 resolution 用獨立
    Stft = plain Hann window + hop n_fft//4 (75% overlap) + normalized=True;magnitude
    做 γ-power compression (clamp_min(1e-12)) 再算 MSE。magnitude-only:real-gain mask
    的 ∠enhanced≡∠noisy,相位無梯度 → 不加 complex 項 (那是 DFN deep-filter stage 用的)。
    無除法 → 對 silence target (noise-only sample, target=0) 也穩定有界。

    factor : 全域縮放 (DFN MultiResSpecLoss factor=500)。⚠ 單一 loss 項 + AdamW 下
             ≈ no-op (m̂/√v̂ 約掉常數);唯一耦合 = grad_clip_norm(1.0) 會更常觸發。
    f_under: under-suppression 非對稱權重。E_c>C_c (enhanced 比 clean 大 = 殘留噪音沒壓掉)
             的誤差乘 f_under。f_under=1 → 對稱 (= F.mse_loss,行為與舊版逐位元相同);
             f_under>1 → 把「留噪音」罰更重,直接逼模型壓 (改 incentive,非全域縮放)。
    """
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4          # DFN Stft default = n_fft//4 (75% overlap)
        win = torch.hann_window(n_fft, device=enhanced.device)   # DFN: plain Hann
        E = torch.stft(enhanced, n_fft, hop, window=win,
                       return_complex=True, normalized=True).abs()   # (B, F, T')
        C = torch.stft(clean,    n_fft, hop, window=win,
                       return_complex=True, normalized=True).abs()

        E_c = E.clamp_min(1e-12).pow(gamma)   # DFN compression (clamp_min, 非 additive eps)
        C_c = C.clamp_min(1e-12).pow(gamma)
        diff = E_c - C_c
        if f_under == 1.0:
            total = total + diff.pow(2).mean()                 # == F.mse_loss(E_c, C_c)
        else:
            w = 1.0 + (f_under - 1.0) * (diff > 0).to(diff.dtype)   # 留噪音 (diff>0) 罰 f_under 倍
            total = total + (w * diff.pow(2)).mean()

    return factor * total / len(fft_sizes)


# ============================================================
# 訓練
# ============================================================

def set_seed(seed):
    """固定所有隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        # Hybrid mode: raw bins below cutoff + ERB above
        _, N_BANDS = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
    else:
        N_BANDS = cfg.getint('signal', 'n_bands')

    LOOKAHEAD = cfg.getint('signal', 'lookahead_frames', fallback=0)
    assert 0 <= LOOKAHEAD <= 2, "lookahead_frames 只支援 0~2"
    FEATURE_CFG = read_feature_config(cfg, SR, HOP_LEN)

    # Training params
    epochs = cfg.getint('training', 'epochs')
    batch_size = cfg.getint('training', 'batch_size')
    lr = cfg.getfloat('training', 'lr')
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device_str = args.device or cfg.get('training', 'device', fallback='cpu')
        device = torch.device(device_str)
    output_dir = cfg.get('paths', 'output_dir')

    # ERB band borders (faithful DeepFilterNet/Keras ERBBand, config-driven)
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        raise NotImplementedError(
            "hybrid bands are not supported with the faithful DFN/Keras ERB filterbank; "
            "set hybrid_cutoff_hz=0 to use pure ERB")
    NFFTBORDER = erb_bandborder(N_BANDS, SR, N_FFT)   # (N_BANDS,) ints, [0 .. n_fft//2+1]

    # Dataset
    use_online = False
    use_wav = False
    if args.packed_dir or args.packed_data:
        from torch.utils.data import ConcatDataset
        pt_files = []
        if args.packed_dir:
            pt_files += sorted(glob.glob(os.path.join(args.packed_dir, '*.pt')))
        if args.packed_data:
            pt_files += args.packed_data
        if not pt_files:
            raise FileNotFoundError(f"在 {args.packed_dir} 找不到任何 .pt 檔案")
        parts = [PackedDataset(p, mmap=args.mmap) for p in pt_files]
        dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]
        use_wav = True
    else:
        raise ValueError("RNNoise-ERB 訓練僅支援 wav-data：請用 --packed-dir 或 --packed-data")

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # epoch_size: precomputed/wav-data 模式下用 RandomSampler 限制每 epoch 的 sample 數
    # online 模式由 DNS4Dataset._shuffle_indices() 處理
    epoch_size = cfg.getint('training', 'epoch_size', fallback=0)
    # packed-data 整包進 RAM (mmap=False) 時 indexing 是 RAM，0 worker 最省；
    # 但 --mmap 時資料在磁碟，需 worker 並行 IO 才不會卡主執行緒 (否則超慢)。
    use_packed = args.packed_data is not None
    packed_in_ram = use_packed and not args.mmap
    n_workers = 0 if (packed_in_ram or not (use_online or use_wav)) else 4
    common_kwargs = dict(num_workers=n_workers, pin_memory=True)
    if n_workers > 0:
        common_kwargs.update(prefetch_factor=4, persistent_workers=True)

    if not use_online and epoch_size > 0 and epoch_size < len(train_set):
        train_sampler = RandomSampler(train_set, replacement=False, num_samples=epoch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, **common_kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, **common_kwargs)

    val_workers = min(n_workers, 2)
    val_kwargs = dict(num_workers=val_workers, pin_memory=True)
    if val_workers > 0:
        val_kwargs.update(prefetch_factor=4, persistent_workers=True)
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
    GRU_SIZE = cfg.getint('model', 'gru_size', fallback=128)
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=COND_SIZE, gru_size=GRU_SIZE,
                         dropout=dropout).to(device)
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

    # Perceptual loss params
    fft_sizes_str = cfg.get('perceptual_loss', 'fft_sizes', fallback='512,256,1024')
    fft_sizes = tuple(int(x.strip()) for x in fft_sizes_str.split(','))
    perc_gamma = cfg.getfloat('perceptual_loss', 'gamma', fallback=0.3)
    perc_factor = cfg.getfloat('perceptual_loss', 'factor', fallback=1.0)
    perc_f_under = cfg.getfloat('perceptual_loss', 'f_under', fallback=1.0)

    # Forward ERBB (mode=0, edge x2) for features; inverse (mode=1, partition of unity)
    # for mask→bin expansion — exactly the DFN/Keras forward/inverse split.
    ERB_FWD = torch.from_numpy(
        compute_erb_matrix(NFFTBORDER, N_FFT, mode=0)).to(device)  # (n_bins, n_bands)
    ERB_INV = torch.from_numpy(
        compute_erb_matrix(NFFTBORDER, N_FFT, mode=1)).to(device)  # (n_bins, n_bands)
    print(f"  feature={FEATURE_CFG['version']}")
    print(f"  shared norm alpha={FEATURE_CFG['alpha']:.6f} "
          f"(tau={FEATURE_CFG['tau_sec']:g}s, dt={HOP_LEN/SR*1000:.1f}ms, "
          f"mean_init={FEATURE_CFG['mean_init_db']:g}dB, "
          f"std_init={FEATURE_CFG['var_init_db2'] ** 0.5:g}dB, "
          f"std_floor={FEATURE_CFG['var_floor_db2'] ** 0.5:g}dB)")

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
        model.load_state_dict(ckpt['state_dict'])
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
    wd_note = f"{weight_decay}→{weight_decay_end} (cosine)" if wd_scheduled else f"{weight_decay} (const)"
    print(f"  dropout={dropout}, weight_decay={wd_note}")
    print(f"  loss: multi-res STFT (perceptual), fft_sizes={fft_sizes}, "
          f"factor={perc_factor}, f_under={perc_f_under}")
    if patience > 0:
        print(f"  early stopping: patience={patience}")
    print(f"  device={device}")

    # 學習曲線 CSV (epoch,train_loss,val_loss,lr)。resume 時保留舊紀錄 (append)，
    # 否則寫 header。畫圖用 plot_curve.py。
    log_csv = os.path.join(output_dir, 'train_log.csv')
    if not os.path.isfile(log_csv):
        with open(log_csv, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr\n')

    global_step = (start_epoch - 1) * len(train_loader)   # resume-safe (給 wd schedule)
    last_wd = weight_decay
    for epoch in range(start_epoch, epochs + 1):
        # 每 epoch 重新 shuffle dataset indices (僅 online 模式)
        if use_online:
            dataset._shuffle_indices()

        # --- Train ---
        model.train()
        train_loss_sum = 0
        # Each batch lane behaves like a long-running stream.  Carrying the
        # normaliser across independent 3-s clips exposes the model to mature
        # statistics and abrupt source changes instead of restarting every clip.
        train_norm_state = None
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            if use_wav:
                for noisy_wav, clean_wav in pbar:
                    noisy_wav = noisy_wav.to(device)
                    clean_wav = clean_wav.to(device)

                    # On-the-fly STFT
                    noisy_spec = stft(noisy_wav, N_FFT, HOP_LEN, WIN_LEN, stft_window)
                    # (B, n_bins, n_frames)

                    # ERB features + zero-pad on T dim 讓 conv 輸出與 spec frame 1:1 對齊
                    # pad_left = 2-LOOKAHEAD (補過去), pad_right = LOOKAHEAD (補未來)
                    power = noisy_spec.abs().pow(2).permute(0, 2, 1)  # (B, T, F)
                    features, train_norm_state = extract_erb_features(
                        power, ERB_FWD, norm_state=train_norm_state,
                        norm_alpha=FEATURE_CFG['alpha'],
                        mean_init_db=FEATURE_CFG['mean_init_db'],
                        var_init_db2=FEATURE_CFG['var_init_db2'],
                        var_floor_db2=FEATURE_CFG['var_floor_db2'],
                        clip=FEATURE_CFG['clip'])  # (B, T, n_bands)
                    features = F.pad(features, (0, 0, 2 - LOOKAHEAD, LOOKAHEAD))

                    pred_gains, _ = model(features)  # (B, T, n_bands)

                    # Apply ERB gains → enhanced STFT → ISTFT
                    enhanced_spec = apply_erb_gains_batch(
                        noisy_spec, pred_gains, ERB_INV, LOOKAHEAD)
                    enhanced_wav = istft(enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                                         stft_window, noisy_wav.size(-1))

                    loss = multi_res_stft_loss(enhanced_wav, clean_wav, fft_sizes, perc_gamma,
                                               factor=perc_factor, f_under=perc_f_under)

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
        val_norm_state = None
        with torch.no_grad():
            if use_wav:
                for noisy_wav, clean_wav in val_loader:
                    noisy_wav = noisy_wav.to(device)
                    clean_wav = clean_wav.to(device)

                    noisy_spec = stft(noisy_wav, N_FFT, HOP_LEN, WIN_LEN, stft_window)
                    power = noisy_spec.abs().pow(2).permute(0, 2, 1)
                    features, val_norm_state = extract_erb_features(
                        power, ERB_FWD, norm_state=val_norm_state,
                        norm_alpha=FEATURE_CFG['alpha'],
                        mean_init_db=FEATURE_CFG['mean_init_db'],
                        var_init_db2=FEATURE_CFG['var_init_db2'],
                        var_floor_db2=FEATURE_CFG['var_floor_db2'],
                        clip=FEATURE_CFG['clip'])
                    features = F.pad(features, (0, 0, 2 - LOOKAHEAD, LOOKAHEAD))
                    pred_gains, _ = model(features)
                    enhanced_spec = apply_erb_gains_batch(
                        noisy_spec, pred_gains, ERB_INV, LOOKAHEAD)
                    enhanced_wav = istft(enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                                         stft_window, noisy_wav.size(-1))
                    val_loss_sum += multi_res_stft_loss(
                        enhanced_wav, clean_wav, fft_sizes, perc_gamma,
                        factor=perc_factor, f_under=perc_f_under).item()

        avg_val = val_loss_sum / max(len(val_loader), 1)
        cur_lr = scheduler.get_last_lr()[0]
        wd_str = f"  wd={last_wd:.2e}" if wd_scheduled else ""
        print(f"  train_loss={avg_train:.5f}  val_loss={avg_val:.5f}  lr={cur_lr:.2e}{wd_str}")
        with open(log_csv, 'a') as f:
            f.write(f'{epoch},{avg_train:.6f},{avg_val:.6f},{cur_lr:.6e}\n')

        # 儲存 checkpoint (compiled model 要取 _orig_mod 避免 key 有 _orig_mod. 前綴)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        ckpt = {
            'epoch': epoch,
            'state_dict': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_val,
            'best_val_loss': best_val_loss,
            'nfftborder': NFFTBORDER.tolist(),
            'feature_version': FEATURE_VERSION,
            'config': {
                'sr': SR, 'n_fft': N_FFT, 'win_len': WIN_LEN,
                'hop_len': HOP_LEN, 'n_bands': N_BANDS,
                'lookahead_frames': LOOKAHEAD,
                'cond_size': COND_SIZE, 'gru_size': GRU_SIZE,
                'feature_version': FEATURE_VERSION,
                'feature_norm_tau_sec': FEATURE_CFG['tau_sec'],
                'feature_mean_init_db': FEATURE_CFG['mean_init_db'],
                'feature_std_init_db': FEATURE_CFG['var_init_db2'] ** 0.5,
                'feature_std_floor_db': FEATURE_CFG['var_floor_db2'] ** 0.5,
                'feature_clip': FEATURE_CFG['clip'],
            },
        }
        torch.save(ckpt, os.path.join(output_dir, f'rnnoise_epoch{epoch}.pth'))

        if avg_val < best_val_loss:
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
    parser.add_argument('--precomputed', default=None,
                        help='預生成資料目錄 (.pt shard 格式, 舊版)')
    parser.add_argument('--packed-dir', default=None,
                        help='包含 .pt 檔的資料夾，自動掃描全部 (pack_dataset.py 產生)')
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
