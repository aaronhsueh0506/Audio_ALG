"""Adding a 16-kHz grid must not relabel legacy corpora or checkpoints."""

import configparser
import copy
import json

import pytest

from AIAEC.dataset_gen.aec_features import AecGrid
from AIAEC.dataset_gen.linear_aec import (
    FROZEN_FRAME_HOP_BY_SR,
    SUPPORTED_FRAME_HOP_BY_SR,
    LinearAecContract,
    linear_aec_contract_from_config,
    make_linear_aec_config,
    make_linear_aec_contract,
    require_linear_aec_contract,
)
from AIAEC.dataset_gen.model_views import MODEL_TASKS
from AIAEC.training_common import (
    make_checkpoint_contract,
    require_checkpoint_contract,
    require_checkpoint_linear_aec,
)


SUPPORTED_GRIDS = ((16000, 512, 256), (16000, 256, 128), (48000, 1024, 512))

# A previously serializable v4 contract with synthetic, non-retired provenance.
# Pin its digest independently of the checked-out AEC's changing source hash.
LEGACY_CONTRACT = {
    'version': 'aiaec-linear-error-v4',
    'engine': 'python_pbfdkf',
    'preset': 'balanced',
    'sample_rate': 16000,
    'frame_size': 512,
    'hop_size': 256,
    'filter_length': 832,
    'mode': 'pbfdkf',
    'enable_res': False,
    'enable_cng': False,
    'enable_shadow': True,
    'enable_delay_est': True,
    'enable_highpass': True,
    'output_seam': 'formed_output',
    'limiter': False,
    'aec_commit': 'legacy-materializer',
    'aec_source_hash': '1' * 64,
    'aec_behavior_hash': '2' * 64,
    'behavior_hash_schema': 'canon-ast-1',
}
LEGACY_FINGERPRINT = '031937366c38e83c575f73b21ba912ef5bb4111fc76f5a657f42664ccf90a90d'


def _config(sample_rate, frame, hop):
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        'signal': {
            'sr': str(sample_rate), 'n_fft': str(frame),
            'win_len': str(frame), 'hop_len': str(hop),
        },
        'linear_aec': {'preset': 'balanced'},
    })
    return cfg


def _checkpoint(linear):
    grid = AecGrid(linear.sample_rate, linear.frame_size,
                   linear.frame_size, linear.hop_size)
    data = {
        'dataset_fingerprint': f'grid-corpus-{linear.sample_rate}-{linear.frame_size}',
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'split_kind': 'random_chunk', 'split_seed': 42,
        'val_fraction': 0.1, 'train_indices': [0, 1], 'val_indices': [2],
    }
    contract = make_checkpoint_contract(
        model_name='Align_ULCNet', task=MODEL_TASKS['Align_ULCNet'],
        grid=grid, model_kwargs={'max_delay_frames': 32}, loss_version='grid-test-v1',
        data_contract=data,
    )
    return contract, grid


def test_supported_grids_preserve_the_legacy_factory_defaults():
    assert FROZEN_FRAME_HOP_BY_SR == {16000: (512, 256), 48000: (1024, 512)}
    assert SUPPORTED_FRAME_HOP_BY_SR == {
        16000: ((512, 256), (256, 128)),
        48000: ((1024, 512),),
    }


@pytest.mark.parametrize('sample_rate,frame,hop', SUPPORTED_GRIDS)
def test_config_and_serialized_contract_accept_each_supported_grid(
        sample_rate, frame, hop):
    contract = linear_aec_contract_from_config(_config(sample_rate, frame, hop))
    assert (contract.sample_rate, contract.frame_size, contract.hop_size) == (
        sample_rate, frame, hop)
    restored = LinearAecContract.from_dict(json.loads(json.dumps(contract.as_dict())))
    assert restored == contract
    assert restored.fingerprint() == contract.fingerprint()


@pytest.mark.parametrize('sample_rate,frame,hop', SUPPORTED_GRIDS)
def test_omitted_window_and_hop_resolve_from_the_selected_fft(
        sample_rate, frame, hop):
    cfg = _config(sample_rate, frame, hop)
    cfg.remove_option('signal', 'win_len')
    cfg.remove_option('signal', 'hop_len')
    contract = linear_aec_contract_from_config(cfg)
    assert (contract.frame_size, contract.hop_size) == (frame, hop)


def test_legacy_serialized_contract_keeps_its_fields_and_fingerprint():
    restored = LinearAecContract.from_dict(copy.deepcopy(LEGACY_CONTRACT))
    assert restored.as_dict() == LEGACY_CONTRACT
    assert restored.fingerprint() == LEGACY_FINGERPRINT


@pytest.mark.parametrize('sample_rate,frame,hop', ((16000, 512, 256), (48000, 1024, 512)))
def test_factory_without_a_frame_keeps_the_legacy_grid(sample_rate, frame, hop):
    config = make_linear_aec_config(sample_rate)
    contract = make_linear_aec_contract(sample_rate)
    explicit = make_linear_aec_contract(sample_rate, frame_size=frame)
    assert (config.frame_size, config.hop_size) == (frame, hop)
    assert contract == explicit
    assert contract.fingerprint() == explicit.fingerprint()


@pytest.mark.parametrize('sample_rate,frame,hop', (
    (16000, 512, 128),  # 75% overlap is not the supported 256/128 grid.
    (16000, 256, 256),
    (16000, 128, 64),
    (48000, 512, 256),
))
def test_config_still_rejects_unlisted_frame_hop_pairs(sample_rate, frame, hop):
    with pytest.raises(ValueError, match='frame/hop'):
        linear_aec_contract_from_config(_config(sample_rate, frame, hop))


def test_explicit_linear_frame_must_match_the_model_fft():
    cfg = _config(16000, 256, 128)
    cfg.set('linear_aec', 'frame_size', '512')
    with pytest.raises(ValueError, match='frame/hop|grid mismatch'):
        linear_aec_contract_from_config(cfg)


def test_serialized_contract_still_rejects_512_frame_with_128_hop():
    bad = dict(LEGACY_CONTRACT, hop_size=128)
    with pytest.raises(ValueError, match='frame/hop'):
        LinearAecContract.from_dict(bad)


def test_both_16k_grids_have_distinct_contract_fingerprints():
    legacy = make_linear_aec_contract(16000, frame_size=512)
    compact = make_linear_aec_contract(16000, frame_size=256)
    assert legacy.fingerprint() != compact.fingerprint()
    # Supporting another grid does not grant a frontend-equivalent migration.
    for actual, expected in ((legacy, compact), (compact, legacy)):
        with pytest.raises(ValueError, match='frame_size|hop_size'):
            require_linear_aec_contract(actual.as_dict(), expected.as_dict(), 'grid test')


@pytest.mark.parametrize('sample_rate,frame,hop', SUPPORTED_GRIDS)
def test_checkpoint_accepts_its_own_model_and_frontend_grid(sample_rate, frame, hop):
    linear = make_linear_aec_contract(sample_rate, frame_size=frame)
    contract, grid = _checkpoint(linear)
    assert require_checkpoint_linear_aec(contract, grid) == linear.as_dict()
    require_checkpoint_contract({'contract': copy.deepcopy(contract)}, contract)


@pytest.mark.parametrize('saved_frame,running_frame', ((512, 256), (256, 512)))
def test_checkpoint_frontend_rejects_the_other_supported_model_grid(
        saved_frame, running_frame):
    saved_linear = make_linear_aec_contract(16000, frame_size=saved_frame)
    contract, _ = _checkpoint(saved_linear)
    running_grid = AecGrid(16000, running_frame, running_frame, running_frame // 2)
    with pytest.raises(ValueError, match='model/PBFDKF grid mismatch'):
        require_checkpoint_linear_aec(contract, running_grid)


@pytest.mark.parametrize('frame', (512, 256))
def test_checkpoint_frontend_rejects_a_forged_fingerprint(frame):
    contract, grid = _checkpoint(make_linear_aec_contract(16000, frame_size=frame))
    contract['linear_aec_contract_hash'] = '0' * 64
    with pytest.raises(ValueError, match='contract_hash does not match'):
        require_checkpoint_linear_aec(contract, grid)


@pytest.mark.parametrize('saved_frame,running_frame', ((512, 256), (256, 512)))
def test_training_resume_rejects_the_other_supported_grid(saved_frame, running_frame):
    saved, _ = _checkpoint(make_linear_aec_contract(16000, frame_size=saved_frame))
    running, _ = _checkpoint(make_linear_aec_contract(16000, frame_size=running_frame))
    with pytest.raises(ValueError, match=r'contract\.n_fft'):
        require_checkpoint_contract({'contract': saved}, running)


def test_training_resume_rejects_frontend_swap_with_unchanged_model_grid():
    expected, _ = _checkpoint(make_linear_aec_contract(16000, frame_size=512))
    forged = copy.deepcopy(expected)
    compact = make_linear_aec_contract(16000, frame_size=256)
    for target in (forged, forged['data']):
        target['linear_aec'] = compact.as_dict()
        target['linear_aec_contract_hash'] = compact.fingerprint()
    with pytest.raises(ValueError, match=r'contract\.data\.linear_aec\.frame_size'):
        require_checkpoint_contract({'contract': forged}, expected)
