import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
RNNoise v0.2 風格噪音抑制模型 — 訓練腳本
基於官方 xiph/rnnoise torch 版本架構，適配 config-driven / ERB bands / 無 pitch

用法:
    python train.py --config config.ini
    python train.py --config config.ini --device cpu
    python train.py --config config.ini --resume output/rnnoise_epoch5.pth
    python train.py --config config.ini --seed 123
"""

import argparse
import configparser
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, random_split
import tqdm

from dataset import DNS4Dataset, PrecomputedDataset, WavPairDataset

# ============================================================
# ERB Band 工具
# ============================================================

def erb_rate(f):
    """頻率 (Hz) → ERB-rate (Glasberg & Moore 1990)"""
    return 21.4 * np.log10(0.00437 * f + 1)

def erb_inv(e):
    """ERB-rate → 頻率 (Hz)"""
    return (10 ** (e / 21.4) - 1) / 0.00437

def compute_erb_bands(n_fft, sr, n_bands):
    """計算 ERB band 的 FFT bin 邊界，回傳 shape=(n_bands+1,) 的整數陣列"""
    n_bins = n_fft // 2 + 1
    e_low = erb_rate(0)
    e_high = erb_rate(sr / 2)
    erb_edges = np.linspace(e_low, e_high, n_bands + 1)
    freq_edges = erb_inv(erb_edges)
    bin_edges = np.round(freq_edges / (sr / n_fft)).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_bins - 1)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1
    bin_edges[-1] = min(bin_edges[-1], n_bins)
    return bin_edges


def compute_hybrid_bands(n_fft, sr, n_erb_high, cutoff_hz):
    """
    Hybrid frequency bands (ref: GTCRN):
      - 0 ~ cutoff_hz: 每個 FFT bin 一個 band (原始解析度)
      - cutoff_hz ~ Nyquist: n_erb_high 個 ERB bands
    回傳: bin_edges (n_bands+1,), n_bands
    """
    n_bins = n_fft // 2 + 1
    bin_res = sr / n_fft
    cutoff_bin = int(round(cutoff_hz / bin_res))
    cutoff_bin = min(cutoff_bin, n_bins - 1)

    # Part 1: individual bins [0, 1, 2, ..., cutoff_bin]
    low_edges = list(range(cutoff_bin + 1))

    # Part 2: ERB bands above cutoff
    e_cut = erb_rate(cutoff_hz)
    e_high = erb_rate(sr / 2)
    erb_edges = np.linspace(e_cut, e_high, n_erb_high + 1)
    freq_edges = erb_inv(erb_edges)
    high_edges = np.round(freq_edges / bin_res).astype(int)
    high_edges = np.clip(high_edges, cutoff_bin, n_bins)
    for i in range(1, len(high_edges)):
        if high_edges[i] <= high_edges[i - 1]:
            high_edges[i] = high_edges[i - 1] + 1
    high_edges[-1] = min(high_edges[-1], n_bins)

    # 合併: low_edges[-1] == high_edges[0] == cutoff_bin
    all_edges = np.array(low_edges + list(high_edges[1:]), dtype=int)
    n_bands = len(all_edges) - 1
    return all_edges, n_bands


def compute_erb_matrix(bin_edges, n_fft, n_bands):
    """
    建構 ERB 轉換矩陣 W, shape = (n_bins, n_bands)
    W[bin, band] = 1.0 if bin 屬於 band, else 0.0
    """
    n_bins = n_fft // 2 + 1
    W = np.zeros((n_bins, n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        W[lo:hi, b] = 1.0
    return W


# ============================================================
# 模型 (基於 RNNoise v0.2 官方 PyTorch 架構)
# ============================================================

class RNNoiseModel(nn.Module):
    """
    架構沿用官方 v0.2: Conv1d 前處理 + 3 層 GRU + concat 全層輸出
    差異: 輸入改為 ERB band log energy, 無 VAD, 無 sparsification
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

        # Concat 全層輸出 (同官方 v0.2)
        cat = torch.cat([conv_out, gru1_out, gru2_out, gru3_out], dim=-1)
        # sin²(π/2 · σ(x)): 低 gain 壓更低、高 gain 不變、過渡更平滑 (ref: RNNoise)
        gains = torch.sin(torch.sigmoid(self.dense_out(cat)) * (3.14159265 / 2)).pow(2)

        return gains, [h1, h2, h3]

# ============================================================
# Perceptual loss helpers (WAV-data mode)
# ============================================================

def extract_erb_features(power_spec, bin_edges):
    """
    power_spec: (..., n_frames, n_bins)  — works for both (T, F) and (B, T, F)
    Returns: (..., n_frames, n_bands)  normalized log ERB energy
    """
    bands = []
    for b in range(len(bin_edges) - 1):
        lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
        bands.append(power_spec[..., lo:hi].sum(dim=-1))
    energy = torch.stack(bands, dim=-1)
    log_energy = torch.log(energy + 1e-10)
    mean = log_energy.mean(dim=-2, keepdim=True)
    std = log_energy.std(dim=-2, keepdim=True) + 1e-8
    return (log_energy - mean) / std


def apply_erb_gains_batch(noisy_spec, gains, bin_edges, lookahead=0):
    """
    noisy_spec : (B, n_bins, n_frames) complex
    gains      : (B, n_frames_out, n_bands)
    Returns    : enhanced_spec (B, n_bins, n_frames) complex
    """
    B, n_bins, n_frames = noisy_spec.shape
    n_frames_out = gains.shape[1]
    bin_gains = torch.ones(B, n_bins, n_frames,
                           device=gains.device, dtype=gains.dtype)
    gain_offset = 2 - lookahead
    for b in range(len(bin_edges) - 1):
        lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
        # gains[:, :, b]: (B, n_frames_out) → unsqueeze → (B, 1, n_frames_out)
        bin_gains[:, lo:hi, gain_offset:gain_offset + n_frames_out] = \
            gains[:, :, b].unsqueeze(1)
    return noisy_spec * bin_gains


def multi_res_stft_loss(enhanced, clean, fft_sizes=(512, 256, 1024)):
    """
    DeepFilterNet 風格 multi-resolution STFT loss.
    enhanced, clean: (B, T)
    組合 spectral convergence loss + log magnitude loss。
    """
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=enhanced.device)
        E = torch.stft(enhanced, n_fft, hop, window=win,
                       return_complex=True).abs()   # (B, F, T')
        C = torch.stft(clean,    n_fft, hop, window=win,
                       return_complex=True).abs()

        # spectral convergence
        sc = (E - C).norm(dim=(-2, -1)) / (C.norm(dim=(-2, -1)) + 1e-8)
        # log magnitude
        log_mag = (torch.log(E + 1e-8) - torch.log(C + 1e-8)).abs().mean(dim=(-2, -1))

        total = total + sc.mean() + log_mag.mean()

    return total / len(fft_sizes)


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

    # Band edges
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        BIN_EDGES, N_BANDS = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
        print(f"  Hybrid bands: {HYBRID_CUTOFF}Hz cutoff, "
              f"{BIN_EDGES[0]}..{int(round(HYBRID_CUTOFF/(SR/N_FFT)))} raw bins + "
              f"{N_ERB_HIGH} ERB = {N_BANDS} total")
    else:
        BIN_EDGES = compute_erb_bands(N_FFT, SR, N_BANDS)

    # Dataset
    use_online = False
    use_wav = False
    if args.wav_data:
        dataset = WavPairDataset(args.wav_data)
        use_wav = True
    elif args.precomputed:
        dataset = PrecomputedDataset(args.precomputed)
    else:
        dataset = DNS4Dataset(cfg)
        use_online = True

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # epoch_size: precomputed/wav-data 模式下用 RandomSampler 限制每 epoch 的 sample 數
    # online 模式由 DNS4Dataset._shuffle_indices() 處理
    epoch_size = cfg.getint('training', 'epoch_size', fallback=0)
    n_workers = 4 if (use_online or use_wav) else 0
    if not use_online and epoch_size > 0 and epoch_size < len(train_set):
        train_sampler = RandomSampler(train_set, replacement=False, num_samples=epoch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=n_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=min(n_workers, 2))

    # Regularization
    dropout = cfg.getfloat('training', 'dropout', fallback=0.0)
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=0.01)
    patience = cfg.getint('training', 'early_stop_patience', fallback=0)

    # 模型
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=64, gru_size=128,
                         dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.98),
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (1.0 + 5e-5 * step)
    )

    gamma             = cfg.getfloat('training', 'gamma',             fallback=0.5)
    loss_over_weight  = cfg.getfloat('training', 'loss_over_weight',  fallback=2.5)
    loss_under_weight = cfg.getfloat('training', 'loss_under_weight', fallback=1.0)
    noise_frame_boost = cfg.getfloat('training', 'noise_frame_boost', fallback=3.0)
    speech_frame_scale = cfg.getfloat('training', 'speech_frame_scale', fallback=2.0)

    # Perceptual loss FFT sizes (wav-data mode)
    fft_sizes_str = cfg.get('perceptual_loss', 'fft_sizes', fallback='512,256,1024')
    fft_sizes = tuple(int(x.strip()) for x in fft_sizes_str.split(','))

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
    print(f"  dropout={dropout}, weight_decay={weight_decay}")
    if use_wav:
        print(f"  loss: multi-res STFT (perceptual), fft_sizes={fft_sizes}")
    else:
        print(f"  loss: gamma={gamma}, over={loss_over_weight}, under={loss_under_weight}, "
              f"noise_boost={noise_frame_boost}, speech_scale={speech_frame_scale}")
    if patience > 0:
        print(f"  early stopping: patience={patience}")
    print(f"  device={device}")

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
                    noisy_wav = noisy_wav.to(device)
                    clean_wav = clean_wav.to(device)

                    # On-the-fly STFT
                    noisy_spec = torch.stft(
                        noisy_wav, N_FFT, HOP_LEN, WIN_LEN,
                        window=stft_window, return_complex=True, center=True)
                    # (B, n_bins, n_frames)

                    # ERB features
                    power = noisy_spec.abs().pow(2).permute(0, 2, 1)  # (B, T, F)
                    features = extract_erb_features(power, BIN_EDGES)  # (B, T, n_bands)

                    pred_gains, _ = model(features)  # (B, T-2, n_bands)

                    # Apply ERB gains → enhanced STFT → ISTFT
                    enhanced_spec = apply_erb_gains_batch(
                        noisy_spec, pred_gains, BIN_EDGES, LOOKAHEAD)
                    enhanced_wav = torch.istft(
                        enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                        window=stft_window, length=noisy_wav.size(-1))

                    loss = multi_res_stft_loss(enhanced_wav, clean_wav, fft_sizes)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    train_loss_sum += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.5f}")
            else:
                for features, targets in pbar:
                    features = features.to(device)
                    targets = targets.to(device)

                    pred_gains, _ = model(features)
                    # Conv1d valid padding 減少 2 個 frame; lookahead 決定 target 對齊位置
                    # lookahead=0: output[i] 對應 input[i+2] → targets[:, 2:   ]
                    # lookahead=1: output[i] 對應 input[i+1] → targets[:, 1:-1 ]
                    t_end = -LOOKAHEAD if LOOKAHEAD > 0 else None
                    targets = targets[:, 2 - LOOKAHEAD : t_end, :]

                    # Speech-weighted asymmetric loss
                    speech_weight = targets.mean(dim=-1, keepdim=True)
                    nb = torch.where(speech_weight < 0.1,
                                     torch.tensor(noise_frame_boost), torch.ones(1))
                    frame_weight = nb + speech_frame_scale * speech_weight

                    pred_g = pred_gains ** gamma
                    targ_g = targets ** gamma
                    error = pred_g - targ_g
                    # error > 0: noise leak (over-estimation); error < 0: over-suppression
                    asym_weight = torch.where(error > 0, loss_over_weight, loss_under_weight)

                    loss = (frame_weight * asym_weight * error.pow(2)).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    train_loss_sum += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_train = train_loss_sum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            if use_wav:
                for noisy_wav, clean_wav in val_loader:
                    noisy_wav = noisy_wav.to(device)
                    clean_wav = clean_wav.to(device)

                    noisy_spec = torch.stft(
                        noisy_wav, N_FFT, HOP_LEN, WIN_LEN,
                        window=stft_window, return_complex=True, center=True)
                    power = noisy_spec.abs().pow(2).permute(0, 2, 1)
                    features = extract_erb_features(power, BIN_EDGES)
                    pred_gains, _ = model(features)
                    enhanced_spec = apply_erb_gains_batch(
                        noisy_spec, pred_gains, BIN_EDGES, LOOKAHEAD)
                    enhanced_wav = torch.istft(
                        enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                        window=stft_window, length=noisy_wav.size(-1))
                    val_loss_sum += multi_res_stft_loss(
                        enhanced_wav, clean_wav, fft_sizes).item()
            else:
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    pred_gains, _ = model(features)
                    t_end = -LOOKAHEAD if LOOKAHEAD > 0 else None
                    targets = targets[:, 2 - LOOKAHEAD : t_end, :]
                    sw = targets.mean(dim=-1, keepdim=True)
                    nb = torch.where(sw < 0.1,
                                     torch.tensor(noise_frame_boost), torch.ones(1))
                    fw = nb + speech_frame_scale * sw
                    pg = pred_gains ** gamma
                    tg = targets ** gamma
                    err = pg - tg
                    aw = torch.where(err > 0, loss_over_weight, loss_under_weight)
                    val_loss_sum += (fw * aw * err.pow(2)).mean().item()

        avg_val = val_loss_sum / max(len(val_loader), 1)
        print(f"  train_loss={avg_train:.5f}  val_loss={avg_val:.5f}")

        # 儲存 checkpoint
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_val,
            'best_val_loss': best_val_loss,
            'bin_edges': BIN_EDGES.tolist(),
            'config': {
                'sr': SR, 'n_fft': N_FFT, 'win_len': WIN_LEN,
                'hop_len': HOP_LEN, 'n_bands': N_BANDS,
                'lookahead_frames': LOOKAHEAD,
                'gamma': gamma, 'loss_over_weight': loss_over_weight,
                'loss_under_weight': loss_under_weight,
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
        description='RNNoise v0.2 訓練 (config-driven, ERB bands)')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--device', default=None,
                        help='覆蓋 config 中的 device 設定')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定 GPU ID (例: --gpu 0)')
    parser.add_argument('--precomputed', default=None,
                        help='預生成資料目錄 (.pt shard 格式, 舊版)')
    parser.add_argument('--wav-data', default=None,
                        help='WAV pair 資料目錄 (gen_dataset.py WAV 模式產生, 使用 perceptual loss)')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint 路徑，從斷點續訓')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42, 設 -1 關閉)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    train(args)
