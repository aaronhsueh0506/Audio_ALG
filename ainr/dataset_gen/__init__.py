"""Shared dataset generation and packed-pair loading utilities."""

from .calibration import fit_ramp, robust_quantile
from .loader import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    BlockShuffleSampler,
    dataloader_worker_kwargs,
    load_packed_dataset,
    locality_preserving_random_split,
    set_seed,
    split_sizes,
    subsets_from_indices,
)
from .packed_dataset import PackedDataset

__all__ = [
    'DEFAULT_SPLIT_SEED',
    'DEFAULT_VAL_FRACTION',
    'BlockShuffleSampler',
    'PackedDataset',
    'dataloader_worker_kwargs',
    'fit_ramp',
    'load_packed_dataset',
    'locality_preserving_random_split',
    'set_seed',
    'robust_quantile',
    'split_sizes',
    'subsets_from_indices',
]
