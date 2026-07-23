"""Memory-mapped loader for packed noisy/clean waveform pairs."""

import os

import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """
    Load ``pack_dataset.py`` output with shape ``(N, 2, T)``.

    The packed dtype is preserved during indexing and DataLoader prefetch.
    Training code should convert the complete batch to float32 after moving it
    to the target device.
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
                f"Packed dataset SR={self.sr}, but config requires "
                f"SR={expected_sr}: {pt_path}"
            )

        self.data = obj['data']
        if self.data.ndim != 3 or self.data.shape[1] != 2:
            raise ValueError(
                f"Packed dataset must have shape (N, 2, T), "
                f"got {tuple(self.data.shape)}"
            )

        n_pairs, _, n_samples = self.data.shape
        size_mb = self.data.nbytes / 1024 ** 2
        storage = "disk-backed" if mmap else "in RAM"
        print(
            f"PackedDataset: {n_pairs} pairs, T={n_samples}, SR={self.sr}, "
            f"dtype={self.data.dtype}, {size_mb:.0f} MB ({storage})"
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pair = self.data[idx]
        return pair[0], pair[1]
