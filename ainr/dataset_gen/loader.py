"""Trainer-side loading primitives shared by every model in ``ainr/``.

WHY THESE LIVE HERE
-------------------
RNNoise-ERB and GTCRN are trained on the same packed 16 kHz corpus and then
compared against each other, so *which samples each one gets* is part of the
comparison protocol, not an implementation detail of either trainer.  When
these functions were copied into each ``train.py`` they drifted immediately:
GTCRN held out ``max(2, 5%)`` while RNNoise-ERB held out ``max(1, 10%)``, so
the two models trained on different corpora and scored on different validation
sets -- and no amount of seed agreement could have fixed it.  ``PackedDataset``
was hoisted here for exactly this reason; the split belongs with it.

``DEFAULT_VAL_FRACTION`` is therefore the single definition of the held-out
fraction.  A trainer may override it, but then it is opting out of the bake-off
and should say so.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler, Subset

from .packed_dataset import PackedDataset


DEFAULT_VAL_FRACTION = 0.05
MIN_VAL_SAMPLES = 2

# The split is part of the comparison protocol, not part of a training run's
# randomness, so it stays fixed even when a trainer disables seeding entirely
# (``--seed -1``).  Passing ``seed=None`` therefore still yields a repeatable
# split rather than an arbitrary one.
DEFAULT_SPLIT_SEED = 42


__all__ = [
    'DEFAULT_SPLIT_SEED',
    'DEFAULT_VAL_FRACTION',
    'MIN_VAL_SAMPLES',
    'BlockShuffleSampler',
    'dataloader_worker_kwargs',
    'load_packed_dataset',
    'locality_preserving_random_split',
    'set_seed',
    'split_sizes',
    'subsets_from_indices',
]


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


def set_seed(seed):
    """Seed every RNG the trainers touch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sizes(dataset, val_fraction=DEFAULT_VAL_FRACTION):
    """(n_train, n_val) for a dataset, using the shared held-out fraction."""
    n_val = max(MIN_VAL_SAMPLES, int(len(dataset) * val_fraction))
    if n_val >= len(dataset):
        raise ValueError(
            f"val split ({n_val}) would consume the whole dataset "
            f"({len(dataset)} samples)")
    return len(dataset) - n_val, n_val


def locality_preserving_random_split(dataset, n_train, n_val, seed=None):
    """Randomly assign samples, then sort each subset for mmap-local indexing.

    The permutation is drawn from a *dedicated* generator seeded independently
    of the global RNG, so the split depends only on (len(dataset), seed) and
    not on how much randomness the rest of setup happened to consume.  That is
    what lets a second model reproduce the exact same split.
    """
    generator = torch.Generator().manual_seed(
        DEFAULT_SPLIT_SEED if seed is None else int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:n_val + n_train])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def subsets_from_indices(dataset, train_indices, val_indices):
    """Rebuild the exact split recorded in a checkpoint."""
    return Subset(dataset, list(train_indices)), Subset(dataset, list(val_indices))


def dataloader_worker_kwargs(num_workers, pin_memory, prefetch_factor):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    if num_workers > 0:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    return kwargs


def load_packed_dataset(path, expected_sr=None, mmap=False):
    """Open a packed ``.pt`` file, or concatenate every ``.pt`` in a directory."""
    if os.path.isdir(path):
        shards = sorted(glob.glob(os.path.join(path, '*.pt')))
        if not shards:
            raise FileNotFoundError(f"no .pt files under {path}")
        if len(shards) > 1:
            return ConcatDataset([
                PackedDataset(shard, expected_sr=expected_sr, mmap=mmap)
                for shard in shards
            ])
        path = shards[0]
    return PackedDataset(path, expected_sr=expected_sr, mmap=mmap)
