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
from torch.utils.data import DataLoader, random_split
import tqdm

from dataset import DNS4Dataset

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
    def __init__(self, n_bands, cond_size=64, gru_size=128):
        super().__init__()
        self.n_bands = n_bands
        self.gru_size = gru_size

        # Conv1d 前處理 (k=3 + k=1, 減少 latency)
        self.conv1 = nn.Conv1d(n_bands, cond_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=1)

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
        print(f"Model: {n_params:,} parameters")

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

        # 3 層 GRU
        gru1_out, h1 = self.gru1(conv_out, h1)
        gru2_out, h2 = self.gru2(gru1_out, h2)
        gru3_out, h3 = self.gru3(gru2_out, h3)

        # Concat 全層輸出 (同官方 v0.2)
        cat = torch.cat([conv_out, gru1_out, gru2_out, gru3_out], dim=-1)
        gains = torch.sigmoid(self.dense_out(cat))

        return gains, [h1, h2, h3]

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
    N_BANDS = cfg.getint('signal', 'n_bands')

    # Training params
    epochs = cfg.getint('training', 'epochs')
    batch_size = cfg.getint('training', 'batch_size')
    lr = cfg.getfloat('training', 'lr')
    device_str = args.device or cfg.get('training', 'device', fallback='cpu')
    device = torch.device(device_str)
    output_dir = cfg.get('paths', 'output_dir')

    # ERB
    BIN_EDGES = compute_erb_bands(N_FFT, SR, N_BANDS)

    # Dataset
    dataset = DNS4Dataset(cfg)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2)

    # 模型
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=64, gru_size=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (1.0 + 5e-5 * step)
    )

    gamma = 0.5  # perceptual exponent

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1

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
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  device={device}")

    for epoch in range(start_epoch, epochs + 1):
        # 每 epoch 重新 shuffle dataset indices
        dataset._shuffle_indices()

        # --- Train ---
        model.train()
        train_loss_sum = 0
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for features, targets in pbar:
                features = features.to(device)
                targets = targets.to(device)

                pred_gains, _ = model(features)
                # Conv1d valid padding 會減少 2 個 frame
                targets = targets[:, 1:-1, :]

                loss = F.mse_loss(pred_gains ** gamma, targets ** gamma)

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
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                pred_gains, _ = model(features)
                targets = targets[:, 1:-1, :]
                val_loss_sum += F.mse_loss(pred_gains ** gamma, targets ** gamma).item()

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
            },
        }
        torch.save(ckpt, os.path.join(output_dir, f'rnnoise_epoch{epoch}.pth'))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, os.path.join(output_dir, 'rnnoise_best.pth'))
            print(f"  ✓ best model saved (val_loss={avg_val:.5f})")

# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNNoise v0.2 訓練 (config-driven, ERB bands)')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--device', default=None,
                        help='覆蓋 config 中的 device 設定')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint 路徑，從斷點續訓')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42, 設 -1 關閉)')
    args = parser.parse_args()
    if args.seed == -1:
        args.seed = None
    train(args)
