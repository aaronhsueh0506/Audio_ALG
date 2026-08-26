import configparser
import copy
import dataclasses
import importlib
import pathlib

import pytest
import torch

import AIAEC.training_common as training_common
from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.aiaec_common import SignalGrid
from AIAEC.training_common import (
    CALIBRATION_ONLY_FAR_INPUT_MODE,
    DEPLOYED_FAR_INPUT_MODE,
    FAR_INPUT_MODE_C_VALUES,
    LinearAecEngine,
    auto_device,
    build_arg_parser,
    build_plain_loaders,
    checkpoint_far_input_mode,
    far_input_mode_c_value,
    compressed_spectral_loss,
    make_checkpoint_contract,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
    split_dataset_by_sample,
)
from AIAEC.dataset_gen import (
    ACCEPTED_BEHAVIOR_HASH_MIGRATIONS,
    LinearAecContract,
    LinearAecProcessor,
    MODEL_TASKS,
    make_linear_aec_contract,
    require_linear_aec_contract,
)
from AIAEC.dataset_gen.aec_behavior_hash import aec_python_behavior_hash


def _cfg(text: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


@pytest.mark.parametrize('module_name', [
    'AIAEC.Align_CRUSE.train',
    'AIAEC.Align_ULCNet.train',
    'AIAEC.DeepVQE_S.train',
    'AIAEC.CAGCRN.train',
])
def test_all_trainers_accept_packed_gpu_and_mmap_args(module_name):
    trainer = importlib.import_module(module_name)
    args = trainer.build_parser().parse_args([
        '--packed-dir', '/datasets/aec-packed', '--gpu', '2', '--mmap',
    ])
    assert args.packed_dir == '/datasets/aec-packed'
    assert args.gpu == 2
    assert args.mmap is True


def test_gpu_arg_takes_precedence_over_device():
    args = build_arg_parser('test').parse_args([
        '--device', 'cpu', '--gpu', '3',
    ])
    assert auto_device(args.device, args.gpu) == 'cuda:3'
    with pytest.raises(ValueError, match='non-negative'):
        auto_device(None, -1)


def test_training_progress_matches_ainr_epoch_style(monkeypatch):
    calls = []

    class FakeBar:
        def __init__(self, loader, desc):
            self.loader = loader
            self.desc = desc

        def __iter__(self):
            return iter(self.loader)

        def set_postfix(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        training_common.tqdm, 'tqdm',
        lambda loader, desc: FakeBar(loader, desc),
    )
    loader = [1, 2]
    bar = training_common.training_progress(
        loader, training=True, epoch=2, max_epochs=10,
    )
    assert bar.desc == 'Epoch 3/10'
    assert list(bar) == loader
    bar.set_postfix(loss='0.1234', gnorm='1.00e-02', refresh=False)
    assert calls == [{
        'loss': '0.1234', 'gnorm': '1.00e-02', 'refresh': False,
    }]
    assert training_common.training_progress(
        loader, training=False, epoch=2, max_epochs=10,
    ) is loader


@pytest.mark.parametrize('module_name', [
    'AIAEC.Align_CRUSE.train',
    'AIAEC.Align_ULCNet.train',
    'AIAEC.DeepVQE_S.train',
    'AIAEC.CAGCRN.train',
])
def test_all_train_loops_use_shared_progress_and_postfix(module_name):
    trainer = importlib.import_module(module_name)
    names = set(trainer.run_epoch.__code__.co_names)
    assert 'training_progress' in names
    assert 'set_postfix' in names


def test_loader_cli_packed_dir_override_and_mmap_reach_dataset(monkeypatch):
    observed = {}
    linear = make_linear_aec_contract(16000, frame_size=512)

    class FakePackedDataset:
        def __init__(self, path, expected_sr=None, mmap=False):
            observed.update(path=path, expected_sr=expected_sr, mmap=mmap)
            self.linear_aec_contract = linear
            self.linear_aec_contract_hash = linear.fingerprint()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

        def fingerprint(self):
            return 'fake-corpus'

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    cfg = _cfg(
        '[data]\n'
        'packed_dir = from-config\n'
        'val_fraction = 0.25\n'
        'batch_size = 2\n'
        'num_workers = 0\n'
    )
    aec_grid, _ = read_grids(_cfg('[signal]\nsr=16000\nn_fft=512\n'))
    _train, _val, contract = build_plain_loaders(
        cfg, aec_grid, packed_dir='from-cli', mmap=True,
    )
    assert observed == {
        'path': 'from-cli', 'expected_sr': 16000, 'mmap': True,
    }
    assert contract['dataset_fingerprint'] == 'fake-corpus'


def test_read_grids_builds_matching_aec_and_signal_grids():
    aec_grid, model_grid = read_grids(_cfg('[signal]\nsr=16000\nn_fft=512\n'))
    assert (aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len) == (16000, 512, 512, 256)
    assert (model_grid.sample_rate, model_grid.n_fft, model_grid.win_len, model_grid.hop_len) == \
        (16000, 512, 512, 256)


def test_read_model_kwargs_overlays_only_declared_keys():
    kwargs = read_model_kwargs(_cfg('[model]\ngru_hidden = 96\n'), AlignCRUSE)
    assert kwargs['gru_hidden'] == 96
    assert kwargs['alignment_mode'] == 'causal_running'  # deployment-safe default


def test_read_model_kwargs_rejects_unknown_key():
    with pytest.raises(ValueError, match='not a AlignCRUSE constructor argument'):
        read_model_kwargs(_cfg('[model]\nnot_a_real_arg = 1\n'), AlignCRUSE)


def test_checkpoint_contract_roundtrip_and_mismatch():
    grid = SignalGrid(16000, 512, 512, 256)
    kwargs = {'gru_hidden': 64}
    contract = make_checkpoint_contract(
        model_name='Align_CRUSE', task=MODEL_TASKS['Align_CRUSE'], grid=grid,
        model_kwargs=kwargs, loss_version='v1',
    )
    require_checkpoint_contract({'contract': contract}, contract)   # must not raise
    with pytest.raises(ValueError, match=r'contract\.sr'):
        require_checkpoint_contract({'contract': contract}, {**contract, 'sr': 48000})


def test_far_input_mode_recorded_legacy_default_and_unknown_rejected():
    grid = SignalGrid(16000, 512, 512, 256)
    contract = make_checkpoint_contract(
        model_name='Align_CRUSE', task=MODEL_TASKS['Align_CRUSE'], grid=grid,
        model_kwargs={}, loss_version='v1',
    )
    # New contracts record the training far-input mode explicitly.
    assert contract['far_input_mode'] == 'raw_far'
    assert checkpoint_far_input_mode(contract) == 'raw_far'

    # Legacy checkpoint (written before the field existed): the ONE defaulting
    # helper reads it as raw_far, and resuming it against a new expected
    # contract still passes.
    legacy = {k: v for k, v in contract.items() if k != 'far_input_mode'}
    assert checkpoint_far_input_mode(legacy) == 'raw_far'
    require_checkpoint_contract({'contract': legacy}, contract)  # must not raise

    # An unknown recorded mode is rejected on both the helper and resume path.
    unknown = dict(contract, far_input_mode='aligned_far')
    with pytest.raises(ValueError, match='far_input_mode'):
        checkpoint_far_input_mode(unknown)
    with pytest.raises(ValueError, match='far_input_mode'):
        require_checkpoint_contract({'contract': unknown}, contract)


def test_deployed_far_mode_is_a_real_c_enumerator():
    """The deployed seam name must map to a C value; a typo cannot.

    Both exporters stamp this string beside its numeric enumerator, and a
    board rejects a descriptor whose two halves disagree.
    """
    assert DEPLOYED_FAR_INPUT_MODE in FAR_INPUT_MODE_C_VALUES
    assert far_input_mode_c_value(DEPLOYED_FAR_INPUT_MODE) == 1
    # Deployment feeds something training never produced; if these ever
    # coincide the whole two-field record has stopped saying anything.
    assert DEPLOYED_FAR_INPUT_MODE not in training_common.FAR_INPUT_MODES


def test_calibration_only_far_mode_has_no_c_value_on_purpose():
    """``model_native_far`` describes recording, never a deployed wiring.

    It exists only so a calibration report can say "this model has no
    separate far seam" instead of borrowing a deployment name. Refusing it a
    C enumerator is the intended behaviour: an integrator who somehow got it
    into a descriptor must fail loudly rather than land on enumerator 0
    (raw_far) by default.
    """
    assert CALIBRATION_ONLY_FAR_INPUT_MODE not in FAR_INPUT_MODE_C_VALUES
    with pytest.raises(ValueError, match='has no C enum value'):
        far_input_mode_c_value(CALIBRATION_ONLY_FAR_INPUT_MODE)
    # And it is not a trainable mode either.
    assert (CALIBRATION_ONLY_FAR_INPUT_MODE
            not in training_common.FAR_INPUT_MODES)


def test_checkpoint_contract_rejects_changed_data_split_indices():
    grid = SignalGrid(16000, 512, 512, 256)
    linear = make_linear_aec_contract(16000, frame_size=512)
    data_contract = {
        'dataset_fingerprint': 'corpus-a',
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'split_kind': 'random_chunk',
        'split_seed': 42,
        'val_fraction': 0.1,
        'train_indices': [0, 2, 3],
        'val_indices': [1],
    }
    contract = make_checkpoint_contract(
        model_name='Align_ULCNet', task=MODEL_TASKS['Align_ULCNet'],
        grid=grid, model_kwargs={}, loss_version='v1',
        data_contract=data_contract,
    )
    require_checkpoint_contract({'contract': contract}, contract)
    changed = copy.deepcopy(contract)
    changed['data']['val_indices'] = [2]
    with pytest.raises(ValueError, match=r'contract\.data\.val_indices'):
        require_checkpoint_contract({'contract': contract}, changed)


def test_checkpoint_linear_aec_rejects_hash_and_model_grid_mismatch():
    linear = make_linear_aec_contract(16000, frame_size=512)
    grid = SignalGrid(16000, 512, 512, 256)
    checkpoint_contract = {
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
    }
    assert require_checkpoint_linear_aec(
        checkpoint_contract, grid
    ) == linear.as_dict()

    bad_hash = {**checkpoint_contract, 'linear_aec_contract_hash': 'bad'}
    with pytest.raises(ValueError, match='contract_hash'):
        require_checkpoint_linear_aec(bad_hash, grid)
    with pytest.raises(ValueError, match='grid mismatch'):
        require_checkpoint_linear_aec(
            checkpoint_contract, SignalGrid(48000, 1024, 1024, 512)
        )


def test_checkpoint_contract_ctor_prefix_does_not_collide_with_model_name():
    grid = SignalGrid(16000, 512, 512, 256)
    contract = make_checkpoint_contract(
        model_name='Align_CRUSE', task=MODEL_TASKS['Align_CRUSE'], grid=grid,
        model_kwargs={'gru_hidden': 64}, loss_version='v1',
    )
    assert contract['model_name'] == 'Align_CRUSE'
    assert contract['ctor_gru_hidden'] == 64


def test_inference_checkpoint_identity_rejects_old_target_and_wrong_model():
    valid = {
        'model_name': 'CAGCRN',
        'task': MODEL_TASKS['CAGCRN'],
    }
    require_checkpoint_model_identity(valid, 'CAGCRN')
    with pytest.raises(ValueError, match='different training target'):
        require_checkpoint_model_identity(
            {**valid, 'task': 'end_to_end_aec_res_nr'}, 'CAGCRN'
        )
    with pytest.raises(ValueError, match='model_name'):
        require_checkpoint_model_identity(valid, 'DeepVQE_S')


def test_compressed_spectral_loss_zero_for_identical_spectra():
    spec = torch.complex(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
    assert float(compressed_spectral_loss(spec, spec)) == pytest.approx(0.0, abs=1e-6)


def test_linear_aec_engine_full_length_output_and_reset():
    engine = LinearAecEngine(n_lanes=2, sample_rate=16000, preset='balanced',
                             frame_size=512)
    mic = torch.randn(2, 16000) * 0.05
    far = torch.randn(2, 16000) * 0.05
    error, echo_estimate = engine(mic, far, 16000)
    assert error.shape == mic.shape
    assert echo_estimate.shape == mic.shape
    aligned_far = engine.get_aligned_far()
    assert aligned_far.shape == far.shape
    assert torch.isfinite(aligned_far).all()
    hop = engine._engines[0].hop_size
    used = (far.shape[-1] // hop) * hop
    # A sub-hop tail is never fed to the engine, so it passes through as the
    # caller's own far -- same policy as the error tail above. (What the tap
    # CONTAINS on the processed span is pinned separately, against a shift
    # this test's uncorrelated mic/far could not produce:
    # test_linear_aec_engine_aligned_far_is_the_shifted_far.)
    torch.testing.assert_close(aligned_far[:, used:], far[:, used:])
    assert torch.isfinite(error).all() and torch.isfinite(echo_estimate).all()
    torch.testing.assert_close(mic - error, echo_estimate)

    engine.arm_reset([True, False])
    error2, _ = engine(mic, far, 16000)
    assert error2.shape == mic.shape


def test_linear_aec_engine_honours_deployment_filter_bank_override():
    engine = LinearAecEngine(
        n_lanes=1, sample_rate=16000, frame_size=512,
        delay_num_filters=2,
    )
    assert engine.delay_num_filters == 2
    assert len(engine._engines[0].delay_est._estimator._matched_filter._filters) == 2

    # A reset must rebuild the same deployment profile, not fall back to the
    # corpus default of five filters.
    engine.reset_lane(0)
    assert len(engine._engines[0].delay_est._estimator._matched_filter._filters) == 2


def test_linear_aec_engine_aligned_far_is_the_shifted_far():
    """The aligned-far tap must BE the shifted far, not merely self-consistent.

    Both inference CLIs feed this tap to the model as the far branch, so what
    it contains is a contract, not an internal detail. The scene has a known
    bulk delay; the applied alignment is read from the engine's public stats
    seam and the expected content is then built INDEPENDENTLY here by shifting
    the caller's own far by that many samples -- nothing is compared against
    the buffer the implementation copied from.
    """
    samples, true_delay = 32768, 1024
    generator = torch.Generator().manual_seed(23)
    far = (torch.randn(samples, generator=generator) * 0.05).unsqueeze(0)
    mic = torch.zeros(1, samples)
    mic[0, true_delay:] = 0.5 * far[0, :samples - true_delay]
    mic += torch.randn(1, samples, generator=generator) * 0.005

    engine = LinearAecEngine(n_lanes=1, sample_rate=16000, frame_size=512)
    engine(mic, far, 16000)
    aligned_far = engine.get_aligned_far()

    lane = engine._engines[0]
    hop = lane.hop_size
    applied = lane.get_stats().delay_samples
    # Non-vacuous on both counts: the alignment really moved, and it moved to
    # the echo (early-or-exact, never late -- the same contract the C
    # known-delay suite asserts).
    assert applied > 0
    assert 0 <= true_delay - applied <= 128
    assert not torch.equal(aligned_far, far)

    # Steady state, well past the acquisition hop: the tap is the caller's
    # own far delayed by exactly `applied` samples.
    used = (samples // hop) * hop
    span = slice(used - 8 * hop, used)
    shifted = slice(span.start - applied, span.stop - applied)
    assert torch.equal(aligned_far[0, span], far[0, shifted])
    # ... and before anything is accepted the tap is the RAW far, which is
    # what production's pre-lock seam serves too.
    assert torch.equal(aligned_far[0, :hop], far[0, :hop])


def test_linear_aec_engine_rejects_sample_rate_mismatch():
    engine = LinearAecEngine(n_lanes=1, sample_rate=16000, frame_size=512)
    mic = torch.randn(1, 16000) * 0.05
    with pytest.raises(ValueError, match='sample_rate'):
        engine(mic, mic, 48000)


def test_linear_aec_engine_aligned_far_requires_a_processed_stream():
    engine = LinearAecEngine(n_lanes=1, sample_rate=16000, frame_size=512)
    with pytest.raises(RuntimeError, match='has not processed audio'):
        engine.get_aligned_far()


def test_inference_linear_aec_matches_offline_materializer_exactly():
    contract = make_linear_aec_contract(16000, frame_size=512)
    generator = torch.Generator().manual_seed(19)
    far = torch.randn(32768, generator=generator) * 0.05
    mic = torch.randn(32768, generator=generator) * 0.02 + far * 0.3

    offline, offline_echo = LinearAecProcessor(contract).process(mic, far)
    inference = LinearAecEngine(
        n_lanes=1, sample_rate=16000, contract=contract.as_dict()
    )
    online, online_echo = inference(
        mic.unsqueeze(0), far.unsqueeze(0), sample_rate=16000
    )
    torch.testing.assert_close(online.squeeze(0), offline, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        online_echo.squeeze(0), offline_echo, rtol=0.0, atol=0.0
    )


def test_inference_rejects_a_different_linear_aec_build():
    """A dataset materialized by a different lib/aec must not load silently.

    ``aec_behavior_hash`` is the only contract field that carries any
    information about the AEC implementation, so this is the single test
    standing between a changed PBFDKF/delay/formed-output seam and inference
    quietly running a different frontend than the checkpoint was trained on.
    """
    contract = make_linear_aec_contract(16000, frame_size=512).as_dict()
    contract['aec_behavior_hash'] = '0' * 64
    with pytest.raises(ValueError, match='aec_behavior_hash'):
        LinearAecEngine(n_lanes=1, sample_rate=16000, contract=contract)


def test_inference_accepts_comment_only_provenance_drift():
    """A comment/format-only edit to lib/aec must NOT invalidate a checkpoint.

    Raw-text provenance moves on a comment reflow; behaviour does not. Only the
    latter gates compatibility, so a doc cleanup cannot strand an already-trained
    checkpoint that no rematerialization could repair.
    """
    contract = make_linear_aec_contract(16000, frame_size=512).as_dict()
    contract['aec_source_hash'] = '0' * 64
    contract['aec_commit'] = 'a-comment-only-cleanup-commit'
    LinearAecEngine(n_lanes=1, sample_rate=16000, contract=contract)


def test_contract_comparison_rejects_behavioral_linear_aec_drift():
    expected = make_linear_aec_contract(16000, frame_size=512).as_dict()
    actual = dict(expected)
    actual['filter_length'] += actual['hop_size']
    with pytest.raises(ValueError, match='filter_length'):
        require_linear_aec_contract(actual, expected, 'test')


def test_contract_comparison_is_not_vacuous():
    """Guard against the comparison silently losing its only real condition.

    Both call sites build the ``runtime`` contract by echoing the recorded
    contract's sample_rate/preset/frame_size/filter_length, and every remaining
    non-provenance field is pinned to a literal by ``__post_init__``. So if the
    provenance fields are ever excluded from the comparison again, no field can
    differ on any real path and the check becomes a tautology -- which is
    exactly what this asserts is not the case.
    """
    recorded = make_linear_aec_contract(16000, frame_size=512)
    runtime = make_linear_aec_contract(
        recorded.sample_rate, preset=recorded.preset,
        frame_size=recorded.frame_size, filter_length=recorded.filter_length,
    )
    fields = {f.name for f in dataclasses.fields(LinearAecContract)}
    provenance = {'aec_commit', 'aec_source_hash', 'aec_behavior_hash'}
    differing = {
        name for name in fields - provenance
        if getattr(runtime, name) != getattr(recorded, name)
    }
    assert not differing, (
        'a non-provenance field varies between an echoed runtime contract and '
        f'the recorded one ({differing}); update this test if that is intended'
    )
    stale = dataclasses.replace(recorded, aec_behavior_hash='0' * 64)
    with pytest.raises(ValueError, match='aec_behavior_hash'):
        require_linear_aec_contract(runtime.as_dict(), stale.as_dict(), 'test')


# ---- verified frontend-equivalent behaviour-hash migrations ----------------
#
# The rows here are the STRUCTURAL invariants the lookup relies on whatever
# the table contains, and they are what a new entry is admitted against. They
# hold for an empty table too, which is why they stayed when it was emptied.
#
# The per-pair rows -- accepted-with-a-warning, refused-in-reverse, the pair
# actually targeting the live build -- live in
# AIAEC/tests/test_linear_aec_behavior_migration.py, together with the
# retired identities that must never come back. That file is the one to read
# for what the table currently claims; this one only checks the shape of the
# claim.

def _contract_pair(pair):
    """(current, recorded) contract dicts, identical except the hash field."""
    base = make_linear_aec_contract(16000, frame_size=512).as_dict()
    old, new = pair
    return dict(base, aec_behavior_hash=new), dict(base, aec_behavior_hash=old)


def test_migration_table_is_one_way_and_not_chained():
    """Structural invariants the lookup relies on.

    The lookup is a single hop and is never re-applied to its own output, so a
    value appearing as another entry's key would read as a two-step migration
    that silently does not work. An identity entry would be dead weight: equal
    dicts return before the table is consulted at all.
    """
    table = ACCEPTED_BEHAVIOR_HASH_MIGRATIONS
    assert not set(table.values()) & set(table), (
        'a migration target is also a migration source; the table is not '
        'applied transitively, so this pair would silently never resolve'
    )
    for old, new in table.items():
        assert old != new
        assert len(old) == 64 and len(new) == 64


@pytest.mark.parametrize(
    "pair", sorted(ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.items())
)
def test_whitelisted_migration_does_not_excuse_a_second_difference(pair):
    """The hash must be the ONLY differing field.

    Otherwise a migration entry becomes a general-purpose amnesty: a real
    frontend change (here a different filter length) would ride along with an
    accepted hash pair and feed inference a `linear_error` from another filter.
    """
    current, recorded = _contract_pair(pair)
    current['filter_length'] += current['hop_size']
    with pytest.raises(ValueError, match='filter_length'):
        require_linear_aec_contract(current, recorded, 'test')


def test_unlisted_behaviour_hash_is_still_refused():
    """An identity in neither the migration table nor the retired list is
    refused on the plain mismatch path -- built without a table entry, so it
    keeps working whatever the table contains."""
    base = make_linear_aec_contract(16000, frame_size=512).as_dict()
    unlisted = dict(base, aec_behavior_hash='0' * 64)
    with pytest.raises(ValueError, match='aec_behavior_hash'):
        require_linear_aec_contract(base, unlisted, 'test')


def test_resnr_trainers_do_not_import_or_execute_live_linear_aec():
    aiaec_root = pathlib.Path(__file__).parents[1]
    for model_name in ('Align_ULCNet',):
        source = (aiaec_root / model_name / 'train.py').read_text()
        assert 'LinearAecEngine' not in source
        assert 'build_sequence_loaders' not in source
        assert 'SequenceChunkSampler' not in source
        assert 'lane_reset_mask' not in source


class _FakeDataset:
    """Minimal dataset for the deterministic per-sample split."""

    def __init__(self, chunks_per_sequence):
        self._sequence_ids = []
        self._chunk_indices = []
        for sequence_id, n_chunks in enumerate(chunks_per_sequence):
            for chunk_index in range(n_chunks):
                self._sequence_ids.append(sequence_id)
                self._chunk_indices.append(chunk_index)

    def sequence_ids(self):
        return list(self._sequence_ids)

    def chunk_indices(self):
        return list(self._chunk_indices)

    def meta(self, index):
        return {'sequence_id': self._sequence_ids[index], 'chunk_index': self._chunk_indices[index]}

    def __len__(self):
        return len(self._sequence_ids)

    def __getitem__(self, index):
        return self._sequence_ids[index], self._chunk_indices[index]


def test_split_dataset_by_sample_can_straddle_sequences_and_covers_everything():
    dataset = _FakeDataset([3] * 20)
    train_indices, val_indices = split_dataset_by_sample(
        dataset, val_fraction=0.2, seed=42
    )

    assert sorted(train_indices + val_indices) == list(range(len(dataset)))
    assert not (set(train_indices) & set(val_indices))
    train_sequences = {dataset.sequence_ids()[i] for i in train_indices}
    val_sequences = {dataset.sequence_ids()[i] for i in val_indices}
    assert train_sequences & val_sequences, "per-chunk split should allow straddling"
    assert len(val_indices) == 12


def test_split_dataset_by_sample_is_deterministic_given_seed():
    dataset = _FakeDataset([2] * 10)
    a = split_dataset_by_sample(dataset, val_fraction=0.3, seed=7)
    b = split_dataset_by_sample(dataset, val_fraction=0.3, seed=7)
    assert a == b
    assert a != split_dataset_by_sample(dataset, val_fraction=0.3, seed=8)


def test_split_dataset_by_sample_zero_val_fraction_is_all_train():
    dataset = _FakeDataset([2] * 5)
    train_indices, val_indices = split_dataset_by_sample(
        dataset, val_fraction=0.0, seed=1
    )
    assert val_indices == []
    assert sorted(train_indices) == list(range(len(dataset)))


def test_split_dataset_by_sample_rejects_out_of_range_fraction():
    dataset = _FakeDataset([1] * 3)
    with pytest.raises(ValueError, match='val_fraction'):
        split_dataset_by_sample(dataset, val_fraction=1.0, seed=1)


AIAEC_TRAINER_SOURCES = [
    pathlib.Path(training_common.__file__).parent / name / 'train.py'
    for name in sorted(MODEL_TASKS)
]


def test_clipping_a_nonfinite_norm_without_the_flag_creates_the_nan():
    """Why ``error_if_nonfinite=True`` is not optional in any trainer.

    ⚠ Written so it can FAIL.  If clip_grad_norm_ ever stopped scaling by
    ``max_norm / (total_norm + eps)``, the unflagged branch would leave the
    gradient finite and the first assertion would say so, instead of this test
    passing while proving nothing.
    """
    exploded = torch.nn.Parameter(torch.zeros(2))
    exploded.grad = torch.tensor([float('inf'), 1.0])
    total = torch.nn.utils.clip_grad_norm_([exploded], 1.0)
    # clip_coef = 1.0 / (inf + 1e-6) = 0.0, and inf * 0.0 = NaN.  optimizer.step()
    # would write that into the weights AND into Adam's moments.
    assert torch.isinf(total)
    assert torch.isnan(exploded.grad).any(), exploded.grad.tolist()

    guarded = torch.nn.Parameter(torch.zeros(2))
    guarded.grad = torch.tensor([float('inf'), 1.0])
    with pytest.raises(RuntimeError):
        torch.nn.utils.clip_grad_norm_([guarded], 1.0, error_if_nonfinite=True)
    # Untouched: the raise lands before any scaling, so a halt dump describes
    # what backward produced rather than what the clip did to it.
    assert not torch.isnan(guarded.grad).any(), guarded.grad.tolist()
    assert guarded.grad.tolist() == [float('inf'), 1.0]


@pytest.mark.parametrize('source_path', AIAEC_TRAINER_SOURCES,
                         ids=lambda p: p.parent.name)
def test_all_trainers_halt_instead_of_clipping_a_nonfinite_norm(source_path):
    source = source_path.read_text(encoding='utf-8')
    assert 'error_if_nonfinite=True' in source
    assert 'halt_on_non_finite(' in source
    assert 'GradNormLog(' in source


@pytest.mark.parametrize('source_path', AIAEC_TRAINER_SOURCES,
                         ids=lambda p: p.parent.name)
def test_all_trainers_step_a_schedule_they_rebuild_on_resume(source_path):
    """A constant LR was this family's actual state: no scheduler at all."""
    source = source_path.read_text(encoding='utf-8')
    assert 'make_scheduler(' in source
    assert 'scheduler.step()' in source
    assert 'fast_forward_scheduler(' in source
    assert 'scheduler.load_state_dict' not in source


def test_the_shared_schedule_actually_reaches_min_lr():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    sched = training_common.make_scheduler(opt, 30, 300, 1e-3, 1e-6, 1e-4)

    seen = []
    for _ in range(300):
        seen.append(opt.param_groups[0]['lr'])
        sched.step()

    assert seen[0] == pytest.approx(1e-4)          # warmup starts low
    assert seen[30] == pytest.approx(1e-3)         # and reaches the base lr
    assert seen[-1] == pytest.approx(1e-6, rel=0.05)
    assert len(set(seen)) > 100, 'a constant LR would collapse to one value'


class _StopAfterSetup(Exception):
    """Sentinel: main() reached its first epoch, so setup completed."""


@pytest.mark.parametrize('model_name', sorted(MODEL_TASKS))
def test_main_builds_its_schedule_before_the_first_epoch(
        model_name, monkeypatch, tmp_path):
    """Run main() through model, optimizer and scheduler construction.

    ⚠ Written because the source-text assertions above CANNOT see this class of
    defect. All four trainers once read ``max_epochs`` to size the cosine period
    on a line ABOVE the one that assigns it -- an UnboundLocalError on the very
    first run -- while every `'make_scheduler(' in source` assertion still
    passed. Nothing short of executing the setup path catches that.

    The dataset and the epoch loop are stubbed; everything between them is the
    trainer's real code reading its real shipped config.ini.
    """
    trainer = importlib.import_module(f'AIAEC.{model_name}.train')
    linear = make_linear_aec_contract(16000, frame_size=512)

    class FakePackedDataset:
        def __init__(self, path, expected_sr=None, mmap=False):
            self.linear_aec_contract = linear
            self.linear_aec_contract_hash = linear.fingerprint()

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return torch.zeros(5, 16000), {}

        def fingerprint(self):
            return 'fake-corpus'

    seen = {}

    def fake_run_epoch(*args, **kwargs):
        seen['scheduler'] = kwargs.get('scheduler')
        raise _StopAfterSetup

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    monkeypatch.setattr(trainer, 'run_epoch', fake_run_epoch)
    monkeypatch.chdir(tmp_path)

    config = pathlib.Path(training_common.__file__).parent / model_name / 'config.ini'
    args = trainer.build_parser().parse_args(
        ['--config', str(config), '--packed-dir', 'fake', '--device', 'cpu']
    )
    with pytest.raises(_StopAfterSetup):
        trainer.main(args)

    scheduler = seen['scheduler']
    assert scheduler is not None, 'run_epoch was called without a scheduler'
    # A schedule that never moves is the state this whole contract replaced.
    before = scheduler.optimizer.param_groups[0]['lr']
    for _ in range(64):
        scheduler.step()
    assert scheduler.optimizer.param_groups[0]['lr'] != before


def test_align_ulcnet_resume_restores_early_stop_counter(
        monkeypatch, tmp_path):
    """A resumed run must not receive a fresh patience window.

    This executes ``main`` through checkpoint loading, scheduler rebuilding,
    one train/validation epoch and checkpoint writing.  The expensive epoch
    body is stubbed, but the state transition under test is the production
    one.  Starting from ``patience - 1`` and observing one non-improvement
    must stop immediately and persist ``patience`` in the new checkpoint.
    """
    trainer = importlib.import_module('AIAEC.Align_ULCNet.train')
    linear = make_linear_aec_contract(16000, frame_size=512)

    class FakePackedDataset:
        def __init__(self, path, expected_sr=None, mmap=False):
            self.linear_aec_contract = linear
            self.linear_aec_contract_hash = linear.fingerprint()

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return torch.zeros(4, 16000), {
                'sequence_id': index // 2, 'chunk_index': index % 2,
            }

        def fingerprint(self):
            return 'resume-test-corpus'

    config_path = tmp_path / 'config.ini'
    config_path.write_text(
        '[signal]\n'
        'sr = 16000\n'
        'n_fft = 512\n'
        '[data]\n'
        'packed_dir = fake\n'
        'batch_size = 2\n'
        'num_workers = 0\n'
        'val_fraction = 0.25\n'
        '[model]\n'
        'max_delay_frames = 2\n'
        '[training]\n'
        f'output_dir = {tmp_path / "output"}\n'
        'lr = 0.001\n'
        'min_lr = 0.000001\n'
        'lr_warmup = 0.0001\n'
        'warmup_epochs = 1\n'
        'max_epochs = 20\n'
        'early_stop_patience = 15\n'
        'grad_clip = 1.0\n'
        '[loss]\n'
        'compression = 0.3\n'
        'magnitude_weight = 1.0\n'
        'complex_weight = 1.0\n',
        encoding='utf-8',
    )

    model = trainer.AlignULCNet(
        SignalGrid(16000, 512, 512, 256), max_delay_frames=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    resume_path = tmp_path / 'resume.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': 3,
        'global_step': 4,
        'best_val': 0.0,
        'no_improve': 14,
        'contract': {},
    }, resume_path)

    calls = []

    def fake_run_epoch(*args, **kwargs):
        calls.append(kwargs.get('optimizer') is not None or len(args) > 5)
        return 1.0, kwargs.get('global_step', 0) + 1

    saved = []
    real_save = torch.save

    def capture_save(obj, path):
        saved.append((copy.deepcopy(obj), str(path)))

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    monkeypatch.setattr(trainer, 'require_checkpoint_contract',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, 'run_epoch', fake_run_epoch)
    monkeypatch.setattr(trainer.torch, 'save', capture_save)
    args = trainer.build_parser().parse_args([
        '--config', str(config_path), '--packed-dir', 'fake', '--device', 'cpu',
        '--resume', str(resume_path),
    ])
    trainer.main(args)

    # One train call and one validation call, then patience is exhausted.
    assert calls == [True, False]
    last = next(obj for obj, path in saved if path.endswith('_last.pth'))
    assert last['best_val'] == 0.0
    assert last['no_improve'] == 15

    # Keep the local name alive until after the monkeypatch has captured the
    # saves; this also makes accidental use of a mocked writer above obvious.
    assert real_save is not capture_save
