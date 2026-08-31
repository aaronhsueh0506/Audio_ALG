import configparser
import copy
import dataclasses
import importlib
import inspect
import io
import pathlib

import pytest
import torch

import AIAEC.training_common as training_common
from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.aiaec_common import SignalGrid
from AIAEC.training_common import (
    CALIBRATION_ONLY_FAR_INPUT_MODE,
    DEFAULT_OPTIMIZER,
    DEPLOYED_FAR_INPUT_MODE,
    FAR_INPUT_MODE_C_VALUES,
    LinearAecEngine,
    auto_device,
    build_arg_parser,
    build_plain_loaders,
    checkpoint_far_input_mode,
    far_input_mode_c_value,
    compressed_spectral_loss,
    component_compressed_mse_loss,
    make_checkpoint_contract,
    make_optimizer,
    power_compressed_complex_mse_loss,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
    si_snr_loss,
    spectrum_to_waveform,
    split_dataset_by_sample,
    stft_consistent_spectrum,
)
from AIAEC.dataset_gen import (
    ACCEPTED_BEHAVIOR_HASH_MIGRATIONS,
    AecGrid,
    LinearAecContract,
    LinearAecProcessor,
    MODEL_TASKS,
    make_linear_aec_contract,
    require_linear_aec_contract,
)
from AIAEC.dataset_gen.aec_features import PACKED_STEM_ORDER
from AIAEC.dataset_gen.linear_aec import (
    MIGRATED_SOURCE_PROVENANCE,
    migrated_ledger_fingerprints,
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


@pytest.mark.parametrize('loss_fn', [
    component_compressed_mse_loss,
    power_compressed_complex_mse_loss,
])
def test_paper_spectral_losses_are_zero_and_finite_at_zero(loss_fn):
    target = torch.zeros(2, 3, 5, dtype=torch.complex64)
    estimate = target.clone().requires_grad_(True)
    loss = loss_fn(estimate, target)
    assert float(loss.detach()) == pytest.approx(0.0)
    loss.backward()
    assert torch.isfinite(estimate.grad).all()


def test_plcpa_beta_weights_the_phase_aware_complex_term():
    target = torch.ones(1, 1, 1, dtype=torch.complex64)
    phase_rotated = torch.full_like(target, 1j)
    magnitude_only = power_compressed_complex_mse_loss(
        phase_rotated, target, complex_weight=0.0,
    )
    phase_aware = power_compressed_complex_mse_loss(
        phase_rotated, target, complex_weight=1.0,
    )
    assert float(magnitude_only) == pytest.approx(0.0)
    assert float(phase_aware) > 0.0


def test_si_snr_loss_prefers_the_clean_target():
    target = torch.randn(2, 800)
    clean = si_snr_loss(target, target)
    noisy = si_snr_loss(target + 0.5 * torch.randn_like(target), target)
    assert clean < noisy


def test_stft_consistency_helpers_preserve_shape_and_gradient():
    grid = AecGrid(16000, 512, 512, 256)
    spectrum = torch.complex(
        torch.randn(2, 9, grid.n_freqs),
        torch.randn(2, 9, grid.n_freqs),
    ).requires_grad_(True)
    waveform = spectrum_to_waveform(spectrum, grid)
    consistent = stft_consistent_spectrum(spectrum, grid)
    assert waveform.shape == (2, 8 * grid.hop_len)
    assert consistent.shape == spectrum.shape
    consistent.abs().mean().backward()
    assert spectrum.grad is not None
    assert torch.isfinite(spectrum.grad).all()


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


def test_migrated_source_provenance_describes_only_accepted_sources():
    """Structural invariants of the legacy-ledger bridge.

    An entry keyed on a behaviour hash nothing migrates from is never
    consulted -- it would look like coverage while granting nothing. And each
    revision has to be a full provenance pair, because the fingerprint is
    reconstructed from it whole.
    """
    for source, revisions in MIGRATED_SOURCE_PROVENANCE.items():
        assert source in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS, (
            'provenance recorded for a behaviour hash no migration admits; '
            'the bridge would never reach it'
        )
        assert revisions, 'an empty revision set covers no corpus'
        commits = [revision['aec_commit'] for revision in revisions]
        assert len(set(commits)) == len(commits), 'duplicate revision'
        for revision in revisions:
            assert set(revision) == {'aec_commit', 'aec_source_hash'}
            assert len(revision['aec_commit']) == 40
            assert len(revision['aec_source_hash']) == 64


def test_every_recorded_revision_reconstructs_its_own_fingerprint():
    """One candidate per revision, and no two revisions collapse into one.

    Several revisions can share an `aec_source_hash` -- a commit that touches
    only non-signal files does not move it -- so distinctness here is what
    proves `aec_commit` is folded into the reconstruction. Drop it and the
    candidates collide, the count falls, and an unrelated build that differs
    only in commit would be accepted.
    """
    current = make_linear_aec_contract(16000, frame_size=512)
    candidates = migrated_ledger_fingerprints(current)
    expected = sum(
        len(MIGRATED_SOURCE_PROVENANCE.get(source, ()))
        for source, target in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS.items()
        if target == current.aec_behavior_hash
    )
    assert len(candidates) == expected
    assert len({fingerprint for _, _, fingerprint in candidates}) == expected
    assert len({commit for _, commit, _ in candidates}) == expected
    for source, _commit, _fingerprint in candidates:
        assert ACCEPTED_BEHAVIOR_HASH_MIGRATIONS[source] == \
            current.aec_behavior_hash


def test_the_legacy_ledger_bridge_is_one_way():
    """A build running a migration SOURCE gets no candidates: the table is
    read forwards only, so a corpus written by the newer frontend is never
    handed to the older one."""
    for source in ACCEPTED_BEHAVIOR_HASH_MIGRATIONS:
        running_source = dataclasses.replace(
            make_linear_aec_contract(16000, frame_size=512),
            aec_behavior_hash=source,
        )
        assert migrated_ledger_fingerprints(running_source) == ()


def test_the_legacy_ledger_bridge_does_not_migrate_48khz():
    current_48k = make_linear_aec_contract(48000, frame_size=1024)
    assert migrated_ledger_fingerprints(current_48k) == ()


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


AIAEC_TRAINER_CONFIGS = [
    pathlib.Path(training_common.__file__).parent / name / 'config.ini'
    for name in sorted(MODEL_TASKS)
]


@pytest.mark.parametrize('source_path', AIAEC_TRAINER_SOURCES,
                         ids=lambda p: p.parent.name)
def test_all_trainers_keep_the_weight_guard_around_checkpoint_writes(source_path):
    """Paper recipes differ; checkpoint corruption protection must not."""
    source = source_path.read_text(encoding='utf-8')
    assert 'WeightScaleGuard(model)' in source

    assert (source.index('WeightScaleGuard(model)')
            > source.index("model.load_state_dict(ckpt['state_dict'])"))
    assert (source.index('weight_guard.check(')
            < source.index("_last.pth"))


def test_shipped_configs_declare_the_model_specific_paper_recipes():
    expected = {
        'Align_ULCNet': ('adam', 'warmup_cosine', 4e-3, 0.0,
                         16, 50, 15),
        'Align_CRUSE': ('adam', 'constant', 1.5e-4, 5e-6,
                        16, 50, 50),
        'DeepVQE_S': ('adamw', 'warmup_cosine', 1.2e-3, 5e-7,
                      8, 50, 50),
        'CAGCRN': ('adamw', 'constant', 1e-3, 5e-7,
                   32, 50, 50),
    }
    declared = {}
    for path in AIAEC_TRAINER_CONFIGS:
        cfg = configparser.ConfigParser()
        assert cfg.read(path), path
        declared[path.parent.name] = (
            cfg.get('training', 'optimizer'),
            cfg.get('training', 'scheduler'),
            cfg.getfloat('training', 'lr'),
            cfg.getfloat('training', 'weight_decay'),
            cfg.getint('data', 'batch_size'),
            cfg.getint('training', 'max_epochs'),
            cfg.getint('training', 'early_stop_patience'),
        )
    assert declared == expected


def test_shipped_loss_and_delay_overrides_are_explicit():
    configs = {}
    for path in AIAEC_TRAINER_CONFIGS:
        cfg = configparser.ConfigParser()
        assert cfg.read(path), path
        configs[path.parent.name] = cfg

    ulcnet = configs['Align_ULCNet']
    assert ulcnet.getint('model', 'max_delay_frames') == 32
    assert ulcnet.getfloat('loss', 'compression') == pytest.approx(0.3)

    for name in ('Align_CRUSE', 'DeepVQE_S'):
        cfg = configs[name]
        assert cfg.getfloat('loss', 'compression') == pytest.approx(0.3)
        assert cfg.getfloat('loss', 'complex_weight') == pytest.approx(0.7)
        assert cfg.getboolean('loss', 'stft_consistency')

    cagcrn = configs['CAGCRN']
    assert cagcrn.getfloat('loss', 'mse_weight') == pytest.approx(1.0)
    assert cagcrn.getfloat('loss', 'si_snr_weight') == pytest.approx(1.0)
    assert cagcrn.getfloat('loss', 'l1_weight') == pytest.approx(1.0)


def test_each_trainer_wires_the_schedule_its_recipe_declares():
    """Three shapes, and the checkpoint handling differs between them.

    Align-ULCNet steps a plateau scheduler on the validation loss.  That state
    cannot be reconstructed from a step count, so it MUST be checkpointed and
    restored.  DeepVQE-S steps the shared warmup/cosine schedule once per
    optimizer step; that state IS reconstructible, so it is deliberately
    rebuilt from the epoch budget and fast-forwarded on resume rather than
    restored -- carrying a stored T_max back is the failure
    ``fast_forward_scheduler`` exists to document.  The remaining two report no
    schedule and must not quietly grow one.
    """
    for path in AIAEC_TRAINER_SOURCES:
        source = path.read_text(encoding='utf-8')
        name = path.parent.name
        if name == 'Align_ULCNet':
            assert 'ReduceLROnPlateau(' in source
            assert 'scheduler.step(val_loss)' in source
            assert "scheduler.load_state_dict(ckpt['scheduler'])" in source
        elif name == 'DeepVQE_S':
            assert 'make_scheduler(' in source
            assert 'scheduler.step()' in source
            assert 'fast_forward_scheduler(scheduler, global_step)' in source
            assert "ckpt['scheduler']" not in source, (
                'the rebuilt schedule must not be restored from the checkpoint')
            # The restore side is only half of it: writing the state is what
            # puts a stale T_max where a later resume can pick it up, and that
            # line contains no `ckpt[` at all. Match on the method instead, so
            # neither quote style nor `.get()` slips past.
            assert 'scheduler.state_dict()' not in source, (
                'the rebuilt schedule must not be written to the checkpoint')
            assert 'ReduceLROnPlateau(' not in source
        else:
            assert 'scheduler.step(' not in source
            assert 'lr_scheduler.' not in source
            assert 'make_scheduler(' not in source


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
def test_main_builds_its_paper_optimizer_before_the_first_epoch(
        model_name, monkeypatch, tmp_path):
    """Execute setup using each shipped config; stub only the epoch body."""
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
        seen['optimizer'] = args[5]
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

    expected_type = {
        'Align_ULCNet': torch.optim.Adam,
        'Align_CRUSE': torch.optim.Adam,
        'DeepVQE_S': torch.optim.AdamW,
        'CAGCRN': torch.optim.AdamW,
    }[model_name]
    assert type(seen['optimizer']) is expected_type


def _drive_main_capturing(model_name, monkeypatch, tmp_path, config_path,
                          capture, scheduler_box=None):
    """Run a trainer's real ``main`` far enough to build its objects."""
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

    def fake_run_epoch(*args, **kwargs):
        capture['contract'] = kwargs['checkpoint_for_halt']['contract']
        raise _StopAfterSetup

    if scheduler_box is not None:
        real_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau

        # A subclass, not a function: SequentialLR does
        # isinstance(child, ReduceLROnPlateau) on every schedule it wraps, and
        # a function there raises TypeError, which would break the schedules
        # this box is not even watching.
        class spy(real_plateau):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                scheduler_box['kwargs'] = kwargs
                scheduler_box['scheduler'] = self

        monkeypatch.setattr(
            torch.optim.lr_scheduler, 'ReduceLROnPlateau', spy)

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    monkeypatch.setattr(trainer, 'run_epoch', fake_run_epoch)
    monkeypatch.chdir(tmp_path)
    args = trainer.build_parser().parse_args(
        ['--config', str(config_path), '--packed-dir', 'fake', '--device', 'cpu']
    )
    with pytest.raises(_StopAfterSetup):
        trainer.main(args)
    return capture['contract']


@pytest.mark.parametrize('model_name', sorted(MODEL_TASKS))
def test_the_checkpoint_contract_records_the_training_recipe(
        model_name, monkeypatch, tmp_path):
    """Two runs of one model at different learning rates must not resume into
    each other.

    ``require_checkpoint_contract`` compares the contract and nothing else, so
    if the recipe is absent those two runs are indistinguishable -- same model,
    same grid, same ``loss_version`` -- and a checkpoint trained at one LR
    silently continues under another.  The four candidates now carry different
    optimizers and different numbers, which is exactly when this stops being
    hypothetical.
    """
    shipped = (pathlib.Path(training_common.__file__).parent
               / model_name / 'config.ini')
    cfg = configparser.ConfigParser()
    assert cfg.read(shipped), shipped

    contract = _drive_main_capturing(
        model_name, monkeypatch, tmp_path, shipped, {})
    recorded = contract['optimizer']
    assert recorded['name'] == cfg.get('training', 'optimizer').lower()
    assert recorded['lr'] == pytest.approx(cfg.getfloat('training', 'lr'))
    assert recorded['weight_decay'] == pytest.approx(
        cfg.getfloat('training', 'weight_decay'))
    assert recorded['schedule'] == cfg.get('training', 'scheduler').lower()

    # The same config with one number changed must produce a contract the
    # comparison refuses -- that is the property, not merely that a field
    # exists.
    cfg.set('training', 'lr', repr(cfg.getfloat('training', 'lr') * 2.0))
    other_path = tmp_path / 'other.ini'
    with open(other_path, 'w', encoding='utf-8') as handle:
        cfg.write(handle)
    other = _drive_main_capturing(
        model_name, monkeypatch, tmp_path, other_path, {})
    assert other != contract
    with pytest.raises(ValueError):
        require_checkpoint_contract({'contract': other}, contract)


def test_align_ulcnet_plateau_reductions_have_a_floor(monkeypatch, tmp_path):
    """The plateau schedule must not be able to anneal the LR to nothing.

    ``reduce_on_plateau`` is no longer what Align-ULCNet ships, but it stays
    selectable, so it keeps its guard: PyTorch's default ``min_lr`` is 0, and a
    run whose validation stalls takes one x0.1 every two epochs while still
    reporting a full campaign.  The config is copied and overridden here rather
    than read as shipped, so this test states which schedule it is exercising
    instead of inheriting whichever one happens to be current.
    """
    shipped = (pathlib.Path(training_common.__file__).parent
               / 'Align_ULCNet' / 'config.ini')
    cfg = configparser.ConfigParser()
    assert cfg.read(shipped), shipped
    cfg.set('training', 'scheduler', 'reduce_on_plateau')
    floor = cfg.getfloat('training', 'min_lr')
    override = tmp_path / 'plateau.ini'
    with io.open(override, 'w', encoding='utf-8') as handle:
        cfg.write(handle)

    box = {}
    _drive_main_capturing(
        'Align_ULCNet', monkeypatch, tmp_path, override, {}, scheduler_box=box)
    assert box['kwargs'].get('min_lr') == pytest.approx(floor), (
        'ReduceLROnPlateau was built without the configured floor')

    scheduler = box['scheduler']
    # Far past max_epochs on purpose. The shipped factor/patience are gentle
    # enough that 50 non-improving epochs do NOT reach the floor -- that is the
    # point of them -- so a 50-epoch loop here would pass with min_lr removed.
    # What this pins is that the reductions CLAMP rather than run to zero.
    for _ in range(500):
        scheduler.step(1.0)          # never improves
        assert scheduler.optimizer.param_groups[0]['lr'] >= floor
    assert scheduler.optimizer.param_groups[0]['lr'] == pytest.approx(floor), (
        'the plateau schedule did not settle on the configured floor')


def test_align_ulcnet_ships_a_step_indexed_schedule(monkeypatch, tmp_path):
    """The shipped schedule must not count epochs.

    This is the whole point of the change away from ``reduce_on_plateau``.
    That scheduler took its patience in EPOCHS, and this project's epoch is
    roughly a tenth of the one the recipe was written against, so the LR
    reached its floor within single-digit epochs and the rest of the campaign
    ran on a dead LR.  Any schedule indexed by optimizer step is structurally
    immune, so what has to be pinned is that the shipped config selects one AND
    that no per-epoch plateau scheduler is constructed alongside it.
    """
    shipped = (pathlib.Path(training_common.__file__).parent
               / 'Align_ULCNet' / 'config.ini')
    cfg = configparser.ConfigParser()
    assert cfg.read(shipped), shipped
    assert cfg.get('training', 'scheduler') == 'warmup_cosine'

    box = {}
    _drive_main_capturing(
        'Align_ULCNet', monkeypatch, tmp_path, shipped, {}, scheduler_box=box)
    assert not box, (
        'the shipped config still builds a ReduceLROnPlateau; its patience is '
        'denominated in epochs and this project redefined the epoch')


@pytest.mark.parametrize('model_name', ['Align_ULCNet', 'DeepVQE_S'])
def test_a_resume_cannot_silently_rebuild_a_different_lr_curve(
        model_name, monkeypatch, tmp_path):
    """Changing the schedule's denominator must change the contract.

    ``warmup_cosine`` is rebuilt from ``max_epochs * len(train_loader)`` and
    fast-forwarded by ``global_step`` -- deliberately, so a saved ``T_max``
    cannot come back on a run with a different horizon.  That makes the rebuild
    only as trustworthy as its two inputs, and ``len(train_loader)`` follows
    ``batch_size``.  Neither was recorded, so either one produced a DIFFERENT
    curve under an IDENTICAL contract, and the resume was accepted.

    Measured on Align-ULCNet's grid, resuming at epoch 20 after batch_size
    16 -> 64 reads 160% progress; and past the horizon the chainable cosine
    walks BACKWARDS, reaching 0.97 x peak LR at 1.75x rather than resting at
    min_lr.  The config names the GPU as the batch-size constraint, which makes
    a hardware change the likeliest reason anyone resumes -- exactly the edit
    this has to refuse.

    Compares real contracts built by the real ``main``, not the source text.
    """
    shipped = (pathlib.Path(training_common.__file__).parent
               / model_name / 'config.ini')

    def contract_for(section=None, key=None, value=None):
        cfg = configparser.ConfigParser()
        assert cfg.read(shipped), shipped
        if section is not None:
            cfg.set(section, key, value)
        path = tmp_path / ('cfg_%d.ini' % len(list(tmp_path.iterdir())))
        with io.open(path, 'w', encoding='utf-8') as handle:
            cfg.write(handle)
        return _drive_main_capturing(
            model_name, monkeypatch, tmp_path, path, {})

    base = contract_for()
    for section, key, value in (
            ('data', 'batch_size', '64'),
            ('training', 'max_epochs', '30'),
    ):
        changed = contract_for(section, key, value)
        assert changed != base, (
            f'{section}.{key} does not reach the checkpoint contract, so a '
            'resume after changing it rebuilds a different LR curve and is '
            'accepted anyway')


def test_align_ulcnet_steps_its_schedule_per_batch_not_per_epoch(
        monkeypatch, tmp_path):
    """The shipped schedule has to be advanced once per optimizer step.

    Selecting ``warmup_cosine`` is not enough. Moving ``scheduler.step()`` out
    of the batch loop and into the epoch loop leaves every string in the file
    unchanged, and leaves ``run_epoch`` still RECEIVING the scheduler, while
    compressing the whole trajectory into ``max_epochs`` steps -- an
    epoch-counted schedule wearing a step-counted name, which is the exact
    defect this model was just moved off.

    So no stubbed ``run_epoch``: this runs one real epoch over a small corpus
    and counts the scheduler's own advances.  Anything other than one per
    training batch fails.
    """
    trainer = importlib.import_module('AIAEC.Align_ULCNet.train')
    shipped = (pathlib.Path(training_common.__file__).parent
               / 'Align_ULCNet' / 'config.ini')
    cfg = configparser.ConfigParser()
    assert cfg.read(shipped), shipped
    # One short epoch on a small corpus: the wiring is what is under test, not
    # convergence.  batch 4 over 36 training chunks makes the per-batch and
    # per-epoch answers differ by 9x rather than by 1.
    cfg.set('training', 'max_epochs', '1')
    cfg.set('training', 'warmup_epochs', '1')
    cfg.set('data', 'batch_size', '4')
    cfg.set('data', 'num_workers', '0')
    cfg.set('model', 'max_delay_frames', '2')
    override = tmp_path / 'one_epoch.ini'
    with io.open(override, 'w', encoding='utf-8') as handle:
        cfg.write(handle)

    linear = make_linear_aec_contract(16000, frame_size=512)

    class FakePackedDataset:
        def __init__(self, path, expected_sr=None, mmap=False):
            self.linear_aec_contract = linear
            self.linear_aec_contract_hash = linear.fingerprint()

        def __len__(self):
            return 40

        def __getitem__(self, index):
            # One row per PACKED_STEM_ORDER entry; a wrong count is rejected by
            # AecStems before the model is ever reached.
            return torch.zeros(len(PACKED_STEM_ORDER), 8192), {}

        def fingerprint(self):
            return 'fake-corpus'

    calls = {'n': 0}
    real_make = trainer.make_scheduler

    def counting_make_scheduler(*args, **kwargs):
        built = real_make(*args, **kwargs)
        real_step = built.step

        def counted(*a, **k):
            calls['n'] += 1
            return real_step(*a, **k)

        built.step = counted
        return built

    batches = {'n': 0}
    real_run_epoch = trainer.run_epoch

    def counting_run_epoch(model, loader, *args, **kwargs):
        # Only the training pass carries a scheduler; the validation pass runs
        # with neither optimizer nor scheduler and must not be counted.
        if kwargs.get('scheduler') is not None:
            batches['n'] = len(loader)
        return real_run_epoch(model, loader, *args, **kwargs)

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    monkeypatch.setattr(trainer, 'make_scheduler', counting_make_scheduler)
    monkeypatch.setattr(trainer, 'run_epoch', counting_run_epoch)
    monkeypatch.chdir(tmp_path)
    args = trainer.build_parser().parse_args(
        ['--config', str(override), '--packed-dir', 'fake', '--device', 'cpu']
    )
    trainer.main(args)

    assert batches['n'] > 1, (
        'the training loader yielded one batch, so per-batch and per-epoch '
        'stepping are indistinguishable and this test proves nothing')
    assert calls['n'] == batches['n'], (
        f"the schedule advanced {calls['n']} times over an epoch of "
        f"{batches['n']} batches; it must advance once per optimizer step, "
        'not once per epoch')


def test_deepvqe_s_steps_its_schedule_per_batch_not_per_epoch(
        monkeypatch, tmp_path):
    """The cosine horizon has to be advanced once per optimizer step.

    The previous version of this test asserted ``'scheduler.step()' in
    inspect.getsource(run_epoch)`` plus a bound on the LR after ``max_epochs``
    advances.  Measured, a mutant that keeps ``scheduler.step()`` inside
    ``run_epoch`` but moves it OUT of the batch loop -- so the schedule
    advances once per epoch, a 4050x re-timing -- passes both assertions: the
    string is still there, and ``main()`` still builds a step-sized horizon so
    the numeric bound still holds.  The whole suite stayed green under it.

    So count the scheduler's own advances over one real epoch instead, exactly
    as the Align-ULCNet twin does.  Anything but one advance per training batch
    fails.
    """
    trainer = importlib.import_module('AIAEC.DeepVQE_S.train')
    shipped = (pathlib.Path(training_common.__file__).parent
               / 'DeepVQE_S' / 'config.ini')
    cfg = configparser.ConfigParser()
    assert cfg.read(shipped), shipped
    cfg.set('training', 'max_epochs', '1')
    cfg.set('training', 'warmup_epochs', '1')
    cfg.set('data', 'batch_size', '4')
    cfg.set('data', 'num_workers', '0')
    override = tmp_path / 'one_epoch.ini'
    with io.open(override, 'w', encoding='utf-8') as handle:
        cfg.write(handle)

    linear = make_linear_aec_contract(16000, frame_size=512)

    class FakePackedDataset:
        def __init__(self, path, expected_sr=None, mmap=False):
            self.linear_aec_contract = linear
            self.linear_aec_contract_hash = linear.fingerprint()

        def __len__(self):
            return 40

        def __getitem__(self, index):
            return torch.zeros(len(PACKED_STEM_ORDER), 8192), {}

        def fingerprint(self):
            return 'fake-corpus'

    calls = {'n': 0}
    real_make = trainer.make_scheduler

    def counting_make_scheduler(*args, **kwargs):
        built = real_make(*args, **kwargs)
        real_step = built.step

        def counted(*a, **k):
            calls['n'] += 1
            return real_step(*a, **k)

        built.step = counted
        return built

    batches = {'n': 0}
    real_run_epoch = trainer.run_epoch

    def counting_run_epoch(model, loader, *args, **kwargs):
        if kwargs.get('scheduler') is not None:
            batches['n'] = len(loader)
        return real_run_epoch(model, loader, *args, **kwargs)

    monkeypatch.setattr(training_common, 'PackedAecDataset', FakePackedDataset)
    monkeypatch.setattr(trainer, 'make_scheduler', counting_make_scheduler)
    monkeypatch.setattr(trainer, 'run_epoch', counting_run_epoch)
    monkeypatch.chdir(tmp_path)
    args = trainer.build_parser().parse_args(
        ['--config', str(override), '--packed-dir', 'fake', '--device', 'cpu']
    )
    trainer.main(args)

    assert batches['n'] > 1, (
        'the training loader yielded one batch, so per-batch and per-epoch '
        'stepping are indistinguishable and this test proves nothing')
    assert calls['n'] == batches['n'], (
        f"the schedule advanced {calls['n']} times over an epoch of "
        f"{batches['n']} batches; it must advance once per optimizer step, "
        'not once per epoch')


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
        'optimizer = adam\n'
        'lr = 0.001\n'
        'weight_decay = 0\n'
        'amsgrad = false\n'
        'scheduler = warmup_cosine\n'
        'lr_warmup = 1e-4\n'
        'warmup_epochs = 1\n'
        'lr_decay_factor = 0.5\n'
        'lr_patience = 5\n'
        'min_lr = 1e-6\n'
        'max_epochs = 20\n'
        'early_stop_patience = 15\n'
        'grad_clip = 1.0\n'
        '[loss]\n'
        'compression = 0.3\n',
        encoding='utf-8',
    )

    model = trainer.AlignULCNet(
        SignalGrid(16000, 512, 512, 256), max_delay_frames=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=1,
    )
    resume_path = tmp_path / 'resume.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
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


# ============================================================
# Optimizer selection
# ============================================================

def test_optimizer_selector_defaults_to_decoupled_decay():
    """⚠ ``type(...) is``, not isinstance: torch 2.8's AdamW SUBCLASSES Adam, so
    ``isinstance(opt, torch.optim.Adam)`` is True for both and an isinstance
    assertion here would pass no matter which one was built.
    """
    params = [torch.nn.Parameter(torch.zeros(2))]

    default = make_optimizer(_cfg('[training]\n'), params, lr=1e-3)
    assert type(default) is torch.optim.AdamW
    assert default.param_groups[0]['lr'] == pytest.approx(1e-3)
    assert default.param_groups[0]['weight_decay'] == pytest.approx(1e-4)
    assert default.param_groups[0]['amsgrad'] is True

    # Kept selectable so a run that predates the switch can be reproduced.
    legacy = make_optimizer(
        _cfg('[training]\noptimizer = Adam\n'), params, lr=1e-3)
    assert type(legacy) is torch.optim.Adam


def test_optimizer_selector_refuses_an_unknown_name():
    """A typo must not silently train something other than what it says."""
    with pytest.raises(ValueError, match='optimizer must be one of'):
        make_optimizer(_cfg('[training]\noptimizer = lion\n'),
                       [torch.nn.Parameter(torch.zeros(2))], lr=1e-3)


def test_coupled_decay_is_what_walks_a_quiet_weight_into_denormals():
    """The mechanism behind the default, demonstrated both ways.

    One parameter whose true gradient is exactly zero -- a branch that has
    stopped receiving signal -- decayed at the shipped lr 1e-3 /
    weight_decay 1e-4 / amsgrad for 5000 steps, about one epoch's worth.
    Coupled, ``weight_decay * w`` is the whole gradient, so
    ``m_hat / sqrt(v_hat)`` normalises it back to ~1 and the weight moves a full
    lr per step: it crosses zero and keeps shrinking, ending nine decades below
    where it started and still falling.  Decoupled, the same number is a
    ``(1 - lr * weight_decay)`` shrink and costs the weight 0.06%.

    ⚠ Written so it can FAIL: if AdamW's decay ever became coupled again, the
    second branch would land on the first branch's value and the final
    assertion would say so instead of this test passing vacuously.
    """
    def decay_a_quiet_weight(cls, steps=5000):
        param = torch.nn.Parameter(torch.tensor([0.5]))
        opt = cls([param], lr=1e-3, weight_decay=1e-4, amsgrad=True)
        for _ in range(steps):
            param.grad = torch.zeros_like(param)
            opt.step()
        return float(param.detach().abs())

    coupled = decay_a_quiet_weight(torch.optim.Adam)
    decoupled = decay_a_quiet_weight(torch.optim.AdamW)

    assert coupled < 1e-9, coupled
    assert decoupled > 0.49, decoupled
    assert decoupled / coupled > 1e8, (decoupled, coupled)
