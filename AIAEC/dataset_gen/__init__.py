"""AEC scenario dataset generation and the shared primitives models import.

The four AEC model projects import from here and nowhere else:

    from AIAEC.dataset_gen import (
        AecGrid, AecStems, SequenceChunkSampler, PackedAecDataset,
        PACKED_STEM_ORDER, alpha_from_tau, stft, istft,
    )

⚠ A project that re-declares any of these is opting out of the comparison.
``AINR/tests/test_bakeoff_protocol.py`` already guards the NR side of that rule
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
    BASE_STEM_ORDER,
    PACKED_STEM_ORDER,
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
from .linear_aec import (
    ACCEPTED_BEHAVIOR_HASH_MIGRATIONS,
    LINEAR_AEC_CONTRACT_VERSION,
    LinearAecContract,
    LinearAecProcessor,
    linear_aec_contract_from_config,
    make_linear_aec_config,
    make_linear_aec_contract,
    materialize_linear_error,
    require_linear_aec_contract,
)
from .manifest import (
    ALL_SPLIT_NAMES,
    SourcePools,
    assert_source_disjoint,
    build_manifest,
    build_unified_manifest,
    config_hash,
    load_manifest,
    pools_for_split,
    save_manifest,
)
from .packed_aec_dataset import PackedAecDataset, aec_collate
from .model_views import (
    MODEL_TASKS,
    ModelView,
    SpectralModelView,
    build_model_view,
    build_spectral_model_view,
)

__all__ = [
    'ACCEPTED_BEHAVIOR_HASH_MIGRATIONS',
    'ALL_SPLIT_NAMES',
    'BASE_STEM_ORDER',
    'PACKED_STEM_ORDER',
    'LINEAR_AEC_CONTRACT_VERSION',
    'NONLINEAR_MODELS',
    'SCENARIOS',
    'STEM_ORDER',
    'AecGrid',
    'AecSequenceRenderer',
    'AecStems',
    'DeviceModel',
    'PackedAecDataset',
    'MODEL_TASKS',
    'ModelView',
    'LinearAecContract',
    'LinearAecProcessor',
    'SpectralModelView',
    'RenderedSequence',
    'SequenceChunkSampler',
    'SequencePlan',
    'SourcePools',
    'aec_collate',
    'alpha_from_tau',
    'assert_source_disjoint',
    'build_manifest',
    'build_unified_manifest',
    'build_model_view',
    'build_spectral_model_view',
    'config_hash',
    'frames_from_seconds',
    'istft',
    'lane_reset_mask',
    'load_manifest',
    'linear_aec_contract_from_config',
    'make_linear_aec_config',
    'make_linear_aec_contract',
    'materialize_linear_error',
    'plan_sequences',
    'pools_for_split',
    'save_manifest',
    'require_linear_aec_contract',
    'sqrt_hann_window',
    'stft',
]
