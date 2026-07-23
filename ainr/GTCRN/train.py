"""
GTCRN 訓練腳本

用法:
    python train.py --config config.ini --packed-dir data/ --gpu 0
    python train.py --config config.ini --packed-dir data/ --resume output/gtcrn_best.pth
"""

import argparse
import configparser
import glob
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, Subset
import tqdm

from model import GTCRN


# ============================================================
# Dataset
# ============================================================

class PackedDataset(Dataset):
    """
    Loads a packed .pt file produced by pack_dataset.py.
    Format: {'data': Tensor(N, 2, T)}  ch0=noisy, ch1=clean.

    Pass mmap=True on shared servers to keep data on disk (OS page cache)
    instead of loading the full tensor into RAM.
    """
    def __init__(self, pt_path: str, mmap: bool = False, expected_sr: int = None):
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"Packed dataset not found: {pt_path}")
        print(f"PackedDataset: loading {pt_path} (mmap={mmap}) ...")
        obj = torch.load(pt_path, map_location='cpu', mmap=mmap, weights_only=True)
        if 'sr' not in obj:
            raise ValueError(f"Packed dataset has no sample-rate metadata: {pt_path}")
        self.sr = int(obj['sr'])
        if expected_sr is not None and self.sr != expected_sr:
            raise ValueError(
                f"Packed dataset SR={self.sr}, but config requires SR={expected_sr}: {pt_path}"
            )
        self.data = obj['data']   # (N, 2, T)
        if self.data.ndim != 3 or self.data.shape[1] != 2:
            raise ValueError(
                f"Packed dataset must have shape (N, 2, T), got {tuple(self.data.shape)}"
            )
        N, _, T = self.data.shape
        size_mb = self.data.nbytes / 1024 ** 2
        print(f"PackedDataset: {N} pairs, T={T}, SR={self.sr}, {size_mb:.0f} MB")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pair = self.data[idx]   # (2, T)
        # Keep packed float16 samples compact while mmap/DataLoader prefetches.
        # The complete batch is converted to float32 after transfer to device.
        return pair[0], pair[1]   # noisy, clean


class BlockShuffleSampler(Sampler):
    """Shuffle mmap data in local blocks instead of causing random page faults."""

    def __init__(self, data_source, block_size=256, num_samples=None):
        self.data_source = data_source
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("mmap_block_size must be greater than zero")
        size = len(data_source)
        self.num_samples = size if num_samples is None else min(int(num_samples), size)

    def __iter__(self):
        size = len(self.data_source)
        block_starts = list(range(0, size, self.block_size))
        emitted = 0
        for block_idx in torch.randperm(len(block_starts)).tolist():
            start = block_starts[block_idx]
            end = min(start + self.block_size, size)
            for offset in torch.randperm(end - start).tolist():
                if emitted >= self.num_samples:
                    return
                yield start + offset
                emitted += 1

    def __len__(self):
        return self.num_samples


def locality_preserving_random_split(dataset, n_train, n_val):
    """Randomly assign samples, then sort each subset for mmap-local indexing."""
    indices = torch.randperm(len(dataset)).tolist()
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:n_val + n_train])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def dataloader_worker_kwargs(num_workers, pin_memory, prefetch_factor):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    if num_workers > 0:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    return kwargs


# ============================================================
# Loss
# ============================================================

def si_snr(pred, target, eps=1e-8):
    """GTCRN paper/GitHub scale-invariant SNR. Input/return shape: (B, T)/(B,)."""
    s_target = (
        (pred * target).sum(dim=-1, keepdim=True)
        / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
        * target
    )
    e_noise = pred - s_target
    return torch.log10(
        s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + eps) + eps
    )


class HybridLoss(nn.Module):
    """
    Paper-faithful GTCRN loss:
        30 * (mag_norm_re_mse + mag_norm_im_mse) + 70 * mag^0.3_mse + SI-SNR
    """
    def forward(self, pred_spec, true_spec, pred_wav, true_wav):
        # pred_spec, true_spec: (B, F, T, 2)
        pred_mag = torch.sqrt(pred_spec[..., 0] ** 2 + pred_spec[..., 1] ** 2 + 1e-12)
        true_mag = torch.sqrt(true_spec[..., 0] ** 2 + true_spec[..., 1] ** 2 + 1e-12)

        # Official GTCRN complex compression: S / |S|^0.7.
        pred_real_n = pred_spec[..., 0] / pred_mag.pow(0.7)
        true_real_n = true_spec[..., 0] / true_mag.pow(0.7)
        pred_imag_n = pred_spec[..., 1] / pred_mag.pow(0.7)
        true_imag_n = true_spec[..., 1] / true_mag.pow(0.7)

        spec_loss = (
            F.mse_loss(pred_real_n, true_real_n)
            + F.mse_loss(pred_imag_n, true_imag_n)
        )
        mag_loss = F.mse_loss(pred_mag ** 0.3, true_mag ** 0.3)
        sisnr_loss = -si_snr(pred_wav, true_wav).mean()

        return 30 * spec_loss + 70 * mag_loss + sisnr_loss


# ============================================================
# Scheduler
# ============================================================

def make_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Train
# ============================================================

def train(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR      = cfg.getint('signal', 'sr')
    N_FFT   = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)

    ERB_SUB1 = cfg.getint('model', 'erb_subband_1', fallback=65)
    ERB_SUB2 = cfg.getint('model', 'erb_subband_2', fallback=64)

    epochs       = cfg.getint('training', 'epochs')
    batch_size   = cfg.getint('training', 'batch_size')
    lr           = cfg.getfloat('training', 'lr')
    min_lr       = cfg.getfloat('training', 'min_lr', fallback=1e-6)
    warmup_ep    = cfg.getint('training', 'warmup_epochs', fallback=3)
    weight_decay = cfg.getfloat('training', 'weight_decay', fallback=0.05)
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

    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device_str = args.device or cfg.get('training', 'device', fallback='cpu')
        device = torch.device(device_str)

    packed_dir = args.packed_dir or cfg.get('paths', 'packed_dir', fallback=None)
    if not packed_dir:
        raise ValueError("--packed-dir or [paths] packed_dir required")

    # Accept either a directory (scans for *.pt) or a direct .pt path
    if os.path.isdir(packed_dir):
        pt_files = sorted(glob.glob(os.path.join(packed_dir, '*.pt')))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {packed_dir}")
        if len(pt_files) > 1:
            from torch.utils.data import ConcatDataset
            dataset = ConcatDataset([
                PackedDataset(p, mmap=args.mmap, expected_sr=SR) for p in pt_files
            ])
        else:
            dataset = PackedDataset(pt_files[0], mmap=args.mmap, expected_sr=SR)
    else:
        dataset = PackedDataset(packed_dir, mmap=args.mmap, expected_sr=SR)
    n_val = max(2, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_set, val_set = locality_preserving_random_split(dataset, n_train, n_val)

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

    model = GTCRN(erb_subband_1=ERB_SUB1, erb_subband_2=ERB_SUB2,
                  nfft=N_FFT, fs=SR).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GTCRN: {n_params:,} parameters")
    print(f"  SR={SR}, N_FFT={N_FFT}, WIN={WIN_LEN}, HOP={HOP_LEN}")
    print(f"  batch={batch_size}, lr={lr}, device={device}")
    if args.mmap:
        print(f"  mmap: block={mmap_block_size}, workers={train_workers}, "
              f"prefetch={prefetch_factor}, packed_dtype_preserved=True")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_ep, epochs, lr, min_lr)
    criterion = HybridLoss()

    stft_window = torch.hann_window(WIN_LEN).pow(0.5).to(device)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    no_improve = 0

    if args.resume:
        print(f"Resuming: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
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

                noisy_spec = torch.view_as_real(torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))  # (B, F, T_f, 2)
                clean_spec = torch.view_as_real(torch.stft(
                    clean, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))

                enhanced_spec = model(noisy_spec)   # (B, F, T_f, 2)

                # ISTFT for SI-SNR: permute → view_as_complex → permute → istft
                enh_c = torch.view_as_complex(
                    enhanced_spec.permute(0, 2, 1, 3).contiguous()
                )                                   # (B, T_f, F)
                enh_c = enh_c.permute(0, 2, 1)     # (B, F, T_f)
                enhanced_wav = torch.istft(
                    enh_c, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T,
                )

                loss = criterion(enhanced_spec, clean_spec, enhanced_wav, clean)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        scheduler.step()

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

                noisy_spec = torch.view_as_real(torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))
                clean_spec = torch.view_as_real(torch.stft(
                    clean, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))
                enhanced_spec = model(noisy_spec)

                enh_c = torch.view_as_complex(
                    enhanced_spec.permute(0, 2, 1, 3).contiguous()
                ).permute(0, 2, 1)
                enhanced_wav = torch.istft(
                    enh_c, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T,
                )

                val_loss += criterion(enhanced_spec, clean_spec, enhanced_wav, clean).item()

        val_loss /= len(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        checkpoint_best = min(best_val_loss, val_loss)
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': checkpoint_best,
            'config': dict(cfg['signal']),
        }
        torch.save(ckpt, os.path.join(output_dir, 'gtcrn_last.pth'))

        if is_best:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, os.path.join(output_dir, 'gtcrn_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"Training done. Best val loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTCRN Training')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--packed-dir', default=None,
                        help='packed.pt file or directory containing *.pt (pack_dataset.py output)')
    parser.add_argument('--mmap', action='store_true',
                        help='Memory-map .pt tensors (low RAM, disk-backed; needs PyTorch>=2.0)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    train(args)
