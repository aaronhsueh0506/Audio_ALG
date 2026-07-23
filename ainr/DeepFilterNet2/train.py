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
import glob
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, Subset
import tqdm

from model import DeepFilterNet2


# ============================================================
# Multi-resolution STFT loss
# ============================================================

def _stft_loss_single(pred, target, n_fft, hop_size, win_size, gamma=0.3):
    window = torch.hann_window(win_size, device=pred.device)
    P = torch.stft(pred,   n_fft, hop_size, win_size, window, return_complex=True).abs()
    T = torch.stft(target, n_fft, hop_size, win_size, window, return_complex=True).abs()
    # Clamp before fractional power: d(x**gamma)/dx is singular at x=0 for gamma<1.
    P = P.clamp_min(1e-12).pow(gamma)
    T = T.clamp_min(1e-12).pow(gamma)
    return F.mse_loss(P, T)


def multi_res_stft_loss(pred, target, fft_sizes=(256, 512, 1024, 2048),
                        hop_sizes=None, win_sizes=None, gamma=0.3):
    if hop_sizes is None:
        hop_sizes = [s // 4 for s in fft_sizes]
    if win_sizes is None:
        win_sizes = list(fft_sizes)
    total = sum(
        _stft_loss_single(pred, target, n, h, w, gamma)
        for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
    )
    return total / len(fft_sizes)


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
        # Preserve packed dtype (normally float16) until the complete batch is
        # copied to the accelerator.  Per-sample float32 conversion defeats
        # mmap's low-RAM benefit and doubles DataLoader prefetch memory.
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
# Feature extraction
# ============================================================

def causal_ema_db_norm(erb_db, ema_state, alpha=0.99, mean_norm_init=(-60.0, -90.0)):
    """
    DeepFilterNet band_mean_norm_erb: per-band causal EMA of dB, subtract running mean, /40.
    State init = linspace(MEAN_NORM_INIT) = -60..-90 dB (NOT first-frame). Requires the STFT
    to be normalized=True (fft^-0.5) so the -60/-90 init is calibrated.
    erb_db    : (B, T, n_erb)
    Returns:
        normed   : (B, T, n_erb)
        ema_state: updated state dict {'erb_mean': (B, 1, n_erb)}
    """
    B, T, n_erb = erb_db.shape
    device = erb_db.device

    if ema_state is None or 'erb_mean' not in ema_state:
        lo_i, hi_i = mean_norm_init
        mu = torch.linspace(lo_i, hi_i, n_erb, device=device, dtype=erb_db.dtype
                            ).view(1, 1, n_erb).expand(B, 1, n_erb).clone()
    else:
        mu = ema_state['erb_mean'].to(device)

    frames = []
    for t in range(T):
        mu = alpha * mu + (1 - alpha) * erb_db[:, t:t + 1, :]
        frames.append((erb_db[:, t:t + 1, :] - mu) / 40.0)
    normed = torch.cat(frames, dim=1)

    return normed, {'erb_mean': mu.detach()}


def causal_ema_mag_norm(spec_low, ema_state, alpha=0.99, eps=1e-12,
                        unit_norm_init=(0.001, 0.0001)):
    """
    DeepFilterNet band_unit_norm (libDF lib.rs): per-bin EMA of |x|, divide by SQRT(EMA).
        s = |x|*(1-a) + s*a ;  x = x / (sqrt(s) + eps)
    State init = linspace(UNIT_NORM_INIT) = 0.001..0.0001 across bins (NOT first-frame).
    spec_low  : (B, T, df_bins) complex
    Returns: normed (B, T, df_bins) complex, ema_state {'mag_mean': (B, 1, df_bins)}
    """
    B, T, df_bins = spec_low.shape
    device = spec_low.device

    if ema_state is None or 'mag_mean' not in ema_state:
        lo_i, hi_i = unit_norm_init
        mu = torch.linspace(lo_i, hi_i, df_bins, device=device, dtype=spec_low.real.dtype
                            ).view(1, 1, df_bins).expand(B, 1, df_bins).clone()
    else:
        mu = ema_state['mag_mean'].to(device)

    frames = []
    for t in range(T):
        mag = spec_low[:, t:t + 1, :].abs()
        mu = alpha * mu + (1 - alpha) * mag
        frames.append(spec_low[:, t:t + 1, :] / (mu.sqrt() + eps))
    normed = torch.cat(frames, dim=1)

    return normed, {'mag_mean': mu.detach()}


def extract_dfn2_features(spec_c, erb_fb, df_bins, ema_state=None):
    """
    Extract DFN2 input features from complex spectrum.

    Args:
        spec_c   : (B, n_bins, T) complex  (return_complex=True convention)
        erb_fb   : (n_erb, n_bins) tensor on same device
        df_bins  : int
        ema_state: dict or None (stateful EMA across calls, reset per segment during training)

    Returns:
        spec_c   : unchanged, (B, n_bins, T) complex
        feat_erb : (B, 1, T, n_erb)   DFN2 encoder expects [B, 1, T, Fe] — time before freq
        feat_spec: (B, 2, T, df_bins) DFN2 encoder expects [B, 2, T, Fc]
        ema_state: updated state
    """
    spec_BTC = spec_c.permute(0, 2, 1)                         # (B, T, n_bins)

    # ERB features: dB + causal EMA normalisation
    erb_power = spec_BTC.abs().pow(2).matmul(erb_fb.T)         # (B, T, n_erb)
    erb_db = (erb_power + 1e-10).log10() * 10
    feat_erb_BTE, ema_state = causal_ema_db_norm(erb_db, ema_state)
    feat_erb = feat_erb_BTE.unsqueeze(1)                        # (B, 1, T, n_erb)

    # DF features: unit-norm magnitude + view_as_real
    spec_low = spec_BTC[:, :, :df_bins]                        # (B, T, df_bins) complex
    unit_s, ema_state = causal_ema_mag_norm(spec_low, ema_state)
    feat_spec = torch.view_as_real(unit_s)                     # (B, T, df_bins, 2)
    feat_spec = feat_spec.permute(0, 3, 1, 2)                  # (B, 2, T, df_bins)

    return spec_c, feat_erb, feat_spec, ema_state


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

    N_ERB      = cfg.getint('model', 'n_erb',       fallback=32)
    DF_BINS    = cfg.getint('model', 'df_bins',     fallback=64)
    DF_ORDER   = cfg.getint('model', 'df_order',    fallback=5)
    EMB_SIZE   = cfg.getint('model', 'emb_size',    fallback=256)
    ENC_CH     = cfg.getint('model', 'enc_channels', fallback=16)
    GRU_GROUPS = cfg.getint('model', 'gru_groups',  fallback=1)

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

    # Multi-res STFT loss params
    fft_sizes = [int(s) for s in cfg.get('perceptual_loss', 'fft_sizes',
                                          fallback='256,512,1024,2048').split(',')]
    gamma     = cfg.getfloat('perceptual_loss', 'gamma', fallback=0.3)

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

    model = DeepFilterNet2(
        n_fft=N_FFT, sr=SR, n_erb=N_ERB, df_bins=DF_BINS, df_order=DF_ORDER,
        enc_ch=ENC_CH, emb_size=EMB_SIZE, gru_groups=GRU_GROUPS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_ep, epochs, lr, min_lr)

    stft_window = torch.hann_window(WIN_LEN).pow(0.5).to(device)

    print(f"DeepFilterNet2 training: SR={SR}, N_FFT={N_FFT}, WIN={WIN_LEN}, HOP={HOP_LEN}")
    print(f"  n_erb={N_ERB}, df_bins={DF_BINS}, df_order={DF_ORDER}")
    print(f"  batch={batch_size}, lr={lr}, device={device}")
    if args.mmap:
        print(f"  mmap: block={mmap_block_size}, workers={train_workers}, "
              f"prefetch={prefetch_factor}, packed_dtype_preserved=True")

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

                spec_c = torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True, normalized=True,
                )  # (B, n_bins, T_f)

                spec_c, feat_erb, feat_spec, _ = extract_dfn2_features(
                    spec_c, model.erb_fb, DF_BINS
                )

                enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)

                # ISTFT
                enhanced_wav = torch.istft(
                    enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T, normalized=True,
                )

                loss = multi_res_stft_loss(
                    enhanced_wav, clean,
                    fft_sizes=fft_sizes,
                    hop_sizes=[s // 4 for s in fft_sizes],
                    win_sizes=fft_sizes,
                    gamma=gamma,
                )

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

                spec_c = torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True, normalized=True,
                )
                spec_c, feat_erb, feat_spec, _ = extract_dfn2_features(
                    spec_c, model.erb_fb, DF_BINS
                )
                enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)
                enhanced_wav = torch.istft(
                    enhanced_spec, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, length=T, normalized=True,
                )
                val_loss += multi_res_stft_loss(
                    enhanced_wav, clean,
                    fft_sizes=fft_sizes,
                    hop_sizes=[s // 4 for s in fft_sizes],
                    win_sizes=fft_sizes,
                    gamma=gamma,
                ).item()

        val_loss /= len(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        is_best = val_loss < best_val_loss
        checkpoint_best = min(best_val_loss, val_loss)
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': checkpoint_best,
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
    args = parser.parse_args()
    train(args)
