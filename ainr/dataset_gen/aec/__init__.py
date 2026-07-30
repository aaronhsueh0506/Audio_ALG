"""AEC scenario dataset generation and the shared primitives models import.

The three AEC model projects import from here and nowhere else:

    from dataset_gen.aec import (
        AecGrid, AecStems, SequenceChunkSampler, PackedAecDataset,
        STEM_ORDER, alpha_from_tau, stft, istft,
    )

⚠ A project that re-declares any of these is opting out of the comparison.
``ainr/tests/test_bakeoff_protocol.py`` already guards the NR side of that rule
for split/sampler/seed; the same reasoning applies to the STFT grid and the
stem channel order, which are equally silent when they drift.
"""

from .aec_dataset import (
    NONLINEAR_MODELS,
    SCENARIOS,
    AecSequenceRenderer,
    DeviceModel,
    RenderedSequence,
    SequencePlan,
    plan_sequences,
)
from .aec_features import (
    STEM_ORDER,
    AecGrid,
    AecStems,
    SequenceChunkSampler,
    alpha_from_tau,
    frames_from_seconds,
    istft,
    lane_reset_mask,
    sqrt_hann_window,
    stft,
)
from .manifest import (
    SourcePools,
    assert_source_disjoint,
    build_manifest,
    config_hash,
    load_manifest,
    pools_for_split,
    save_manifest,
)
from .packed_aec_dataset import PackedAecDataset, aec_collate

__all__ = [
    'NONLINEAR_MODELS',
    'SCENARIOS',
    'STEM_ORDER',
    'AecGrid',
    'AecSequenceRenderer',
    'AecStems',
    'DeviceModel',
    'PackedAecDataset',
    'RenderedSequence',
    'SequenceChunkSampler',
    'SequencePlan',
    'SourcePools',
    'aec_collate',
    'alpha_from_tau',
    'assert_source_disjoint',
    'build_manifest',
    'config_hash',
    'frames_from_seconds',
    'istft',
    'lane_reset_mask',
    'load_manifest',
    'plan_sequences',
    'pools_for_split',
    'save_manifest',
    'sqrt_hann_window',
    'stft',
]
