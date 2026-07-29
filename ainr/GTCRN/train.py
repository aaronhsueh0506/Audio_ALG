"""
GTCRN 訓練腳本

用法:
    python train.py --config config.ini --packed-dir data/ --gpu 0
    python train.py --config config.ini --packed-dir data/ --resume output/gtcrn_best.pth
"""

import argparse
import configparser
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import tqdm

from checkpoint_utils import extract_state_dict
from model import GTCRN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen import (  # noqa: E402
    BlockShuffleSampler,
    dataloader_worker_kwargs,
    load_packed_dataset,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
    subsets_from_indices,
)


# ``PackedDataset``, the sampler, the seeder and the train/val split all live
# in ``dataset_gen`` and are shared by all three models.  They used to be
# copied into each trainer, and they drifted: GTCRN held out 5% while
# RNNoise-ERB held out 10%, so the two models being compared were trained on
# different corpora.  Which samples each model gets is part of the comparison
# protocol, so there is exactly one definition of it.


# ============================================================
# Checkpoint contract
# ============================================================
#
# Bumped whenever a change makes previously-trained weights invalid or
# meaningless to resume.  Mirrors the gates in RNNoise-ERB/train.py:53-54 and
# DeepFilterNet2/train.py:39-40, which GTCRN previously lacked entirely.
MODEL_VERSION = 'gtcrn_upstream_bitexact_v1'
LOSS_VERSION = 'gtcrn_hybrid_30spec_70mag_sisnr_istft_both_v1'

# Every non-version key in build_contract() is a field that changes the meaning
# of the weights (``erb_subband_1/2`` included -- they alter the model's input
# width yet live in [model], which pre-contract checkpoints never recorded).
# Derived rather than restated so adding a field to build_contract() cannot
# silently leave it unvalidated.
_VERSION_FIELDS = ('model_version', 'loss_version')


def build_contract(cfg, win_len, hop_len):
    return {
        'model_version': MODEL_VERSION,
        'loss_version': LOSS_VERSION,
        'sr': cfg.getint('signal', 'sr'),
        'n_fft': cfg.getint('signal', 'n_fft'),
        'win_len': win_len,
        'hop_len': hop_len,
        'erb_subband_1': cfg.getint('model', 'erb_subband_1', fallback=65),
        'erb_subband_2': cfg.getint('model', 'erb_subband_2', fallback=64),
    }


def require_checkpoint_contract(ckpt, contract, context='checkpoint',
                                allow_missing=False):
    """Refuse to resume across a semantic change.

    Upstream ships no trainer, so this is our own hygiene — but without it a
    resume across an n_fft or erb_subband change silently succeeds and trains
    garbage, which is exactly what the old code did.

    ``allow_missing=True`` skips the version gate for checkpoints that predate
    it -- the vendored upstream tars carry no contract at all, so inference
    must still accept them while enforcing the contract on anything that does
    record one.
    """
    if allow_missing and not any(key in ckpt for key in _VERSION_FIELDS):
        return
    for key in _VERSION_FIELDS:
        got = ckpt.get(key)
        if got != contract[key]:
            shown = repr(got) if got is not None else 'missing (pre-contract checkpoint)'
            raise ValueError(
                f"{context} {key}={shown}, expected {contract[key]!r}. "
                f"Resuming across this change would train on incompatible weights; "
                f"start a fresh run instead."
            )
    # Beyond this point the checkpoint is known to carry a contract, so every
    # field must be present AND match.  "compare only if present" let a
    # checkpoint that recorded nothing but the two version strings satisfy the
    # whole gate; vendored upstream tars are already handled above by
    # allow_missing, so nothing needs that leniency.
    for key in contract:
        if key in _VERSION_FIELDS:
            continue
        if key not in ckpt:
            raise ValueError(
                f"{context} is missing contract field {key!r} (expected "
                f"{contract[key]!r}); it predates this field being recorded, "
                f"so its value cannot be verified -- start a fresh run."
            )
        if ckpt[key] != contract[key]:
            raise ValueError(
                f"{context} {key}={ckpt[key]!r}, but config requires "
                f"{contract[key]!r}."
            )


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
    Paper-faithful GTCRN loss (paper Eq. 1 with alpha=0.01, beta=0.3, scaled
    x100 exactly as gtcrn_github/loss.py does):
        30 * (mag_norm_re_mse + mag_norm_im_mse) + 70 * mag^0.3_mse + SI-SNR

    Both waveforms are obtained by ISTFT of the two spectra, matching upstream
    gtcrn_github/loss.py:24-25.  This repo previously compared an ISTFT'd
    prediction against the RAW clean waveform; because sqrt-Hann WOLA does not
    reconstruct the first/last half-frame exactly, that put a floor on the
    SI-SNR term that the model could never reach.
    """

    def __init__(self, n_fft=512, hop_len=256, win_len=512):
        super().__init__()
        self.n_fft, self.hop_len, self.win_len = n_fft, hop_len, win_len
        self.register_buffer('window', torch.hann_window(win_len).pow(0.5))

    def _istft(self, spec):
        # spec: (B, F, T, 2) real-valued -> (B, T_samples)
        return torch.istft(
            torch.view_as_complex(spec.contiguous()),
            self.n_fft, self.hop_len, self.win_len, window=self.window,
        )

    def forward(self, pred_spec, true_spec):
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
        sisnr_loss = -si_snr(self._istft(pred_spec), self._istft(true_spec)).mean()

        return 30 * spec_loss + 70 * mag_loss + sisnr_loss


# ============================================================
# Train
# ============================================================

def train(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # Seed before anything that draws randomness.  Without this the train/val
    # split was redrawn every run, so two runs of the same config were not
    # comparable and --resume leaked training data into validation.
    set_seed(args.seed)

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
    lr_patience  = cfg.getint('training', 'lr_patience', fallback=5)
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

    # Accepts either a directory (scans for *.pt) or a direct .pt path.
    dataset = load_packed_dataset(packed_dir, expected_sr=SR, mmap=args.mmap)
    n_train, n_val = split_sizes(dataset)

    # The contract must be built before the resume checkpoint is read, and the
    # split must come from the checkpoint when resuming — redrawing it would
    # put previously-trained samples into validation.
    contract = build_contract(cfg, WIN_LEN, HOP_LEN)
    resume_ckpt = None
    if args.resume:
        print(f"Resuming: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_contract(resume_ckpt, contract, context=args.resume)

    if resume_ckpt is not None and 'train_indices' in resume_ckpt:
        train_set, val_set = subsets_from_indices(
            dataset, resume_ckpt['train_indices'], resume_ckpt['val_indices']
        )
        print(f"  restored split from checkpoint: "
              f"{len(train_set)} train / {len(val_set)} val")
    else:
        train_set, val_set = locality_preserving_random_split(
            dataset, n_train, n_val, args.seed
        )
        if resume_ckpt is not None:
            print("  ⚠ checkpoint has no stored split; redrawing from --seed "
                  f"{args.seed} (validation may be contaminated)")

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

    # GTCRN paper §3.2: "The models are trained by Adam Optimizer with an
    # initial learning rate of 0.001.  The learning rate will be halved if the
    # validation loss does not decrease for 5 consecutive epochs."
    # This repo previously used AdamW(wd=0.05) + cosine-with-warmup, which is
    # neither the paper's optimizer nor its schedule.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr,
    )
    criterion = HybridLoss(n_fft=N_FFT, hop_len=HOP_LEN, win_len=WIN_LEN).to(device)

    stft_window = criterion.window   # same sqrt-Hann, already on device

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    no_improve = 0

    if resume_ckpt is not None:
        model.load_state_dict(extract_state_dict(resume_ckpt, args.resume))
        if 'optimizer' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if 'scheduler' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler'])
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_loss = resume_ckpt.get('best_val_loss', float('inf'))
        # Without this, early stopping restarts its patience window on every
        # resume and can never fire.
        no_improve = resume_ckpt.get('no_improve', 0)
        print(f"  Resumed epoch {start_epoch - 1}, best_val_loss={best_val_loss:.5f}, "
              f"no_improve={no_improve}")

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

                noisy_spec = torch.view_as_real(torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))  # (B, F, T_f, 2)
                clean_spec = torch.view_as_real(torch.stft(
                    clean, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))

                enhanced_spec = model(noisy_spec)   # (B, F, T_f, 2)

                # The loss ISTFTs both spectra itself (upstream loss.py:24-25).
                loss = criterion(enhanced_spec, clean_spec)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        # ReduceLROnPlateau steps on the VALIDATION loss, so it is advanced
        # after the validation pass below, not here.

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)
                clean = clean.to(device=device, dtype=torch.float32,
                                 non_blocking=pin_memory)

                noisy_spec = torch.view_as_real(torch.stft(
                    noisy, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))
                clean_spec = torch.view_as_real(torch.stft(
                    clean, N_FFT, HOP_LEN, WIN_LEN,
                    window=stft_window, return_complex=True,
                ))
                enhanced_spec = model(noisy_spec)

                val_loss += criterion(enhanced_spec, clean_spec).item()

        val_loss /= len(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")
        scheduler.step(val_loss)

        # Update the early-stopping counter BEFORE writing the checkpoint, so
        # the saved no_improve matches the state a resume needs to restore.
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
            # Exact split, so a resume cannot leak trained samples into val.
            'train_indices': train_set.indices,
            'val_indices': val_set.indices,
            'seed': args.seed,
            **contract,
        }
        torch.save(ckpt, os.path.join(output_dir, 'gtcrn_last.pth'))

        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'gtcrn_best.pth'))
            print(f"  ✓ New best: {best_val_loss:.5f}")
        elif patience > 0 and no_improve >= patience:
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
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; also fixes the train/val split. '
                             'Must match RNNoise-ERB for a comparable run.')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    train(args)
