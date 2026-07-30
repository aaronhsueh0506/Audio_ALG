"""Trainer contracts: checkpoint gate, sequence-level split, one real step.

The end-to-end test builds a real packed shard on disk and runs two optimizer
steps through the real loader, sampler, collate and loss.  ⚠ Keep it: every
individual piece here has a unit test somewhere, and the failure mode this
catches is none of them -- it is the wiring between a sequence-aware sampler, a
per-lane recurrent state and a chunk-level dataset, which is where a
state-carrying trainer actually goes wrong.
"""

import configparser
import os
import sys
import tempfile

import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _stale in ('train', 'denoise', 'model', 'postproc'):
    sys.modules.pop(_stale, None)

from denoise import analysis_pad, reconstruct  # noqa: E402
from model import JointAECNR  # noqa: E402
from train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    JointLoss,
    build_contract,
    causal_ema,
    make_sequence_loader,
    require_checkpoint_contract,
    run_epoch,
    sequence_level_split,
    stem_spectra,
)

sys.path.insert(0, os.path.dirname(ROOT))
from dataset_gen_aec import (  # noqa: E402
    STEM_ORDER,
    AecGrid,
    AecStems,
    PackedAecDataset,
    alpha_from_tau,
    stft,
)


AINR = os.path.dirname(ROOT)


def load_config(**signal_overrides):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    for key, value in signal_overrides.items():
        cfg['signal'][key] = str(value)
    return cfg


# ============================================================
# Checkpoint contract
# ============================================================

def _contract():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    return cfg, grid, JointAECNR.from_config(cfg, grid)


def test_contract_records_the_resolved_integers_not_the_seconds():
    """⚠ 0.25 s is 16 taps at 16 kHz and 23 at 48 kHz; the taps are what the
    weight shapes encode, so the taps are what must be compared."""
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    assert contract['ref_context_frames'] == model.ref_context_frames
    assert contract['df_order'] == model.df_order
    assert 'ref_context_sec' not in contract


def test_contract_accepts_itself():
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    require_checkpoint_contract(dict(contract), contract)


def test_contract_rejects_a_version_change():
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    stale = dict(contract, loss_version='something_else')
    with pytest.raises(ValueError, match='loss_version'):
        require_checkpoint_contract(stale, contract)


def test_contract_rejects_a_head_switch_that_leaves_shapes_intact():
    """The failure the gate exists for: aux_echo_head off -> on loads cleanly."""
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    with pytest.raises(ValueError, match='aux_echo_head'):
        require_checkpoint_contract(dict(contract, aux_echo_head=False), contract)


def test_contract_rejects_a_checkpoint_missing_a_field():
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    partial = {key: contract[key] for key in contract if key != 'df_bins'}
    with pytest.raises(ValueError, match='df_bins'):
        require_checkpoint_contract(partial, contract)


def test_contract_only_exempts_a_checkpoint_that_records_nothing():
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    require_checkpoint_contract({'state_dict': {}}, contract, allow_missing=True)
    with pytest.raises(ValueError):
        require_checkpoint_contract({'state_dict': {}}, contract)


def test_every_version_string_is_in_the_contract():
    cfg, grid, model = _contract()
    contract = build_contract(cfg, grid, model)
    assert contract['model_version'] == MODEL_VERSION
    assert contract['feature_version'] == FEATURE_VERSION
    assert contract['loss_version'] == LOSS_VERSION


# ============================================================
# Split
# ============================================================

class _SequenceStub:
    def __init__(self, sequence_ids):
        self._sequence_ids = list(sequence_ids)

    def sequence_ids(self):
        return list(self._sequence_ids)


def test_split_never_cuts_a_sequence_in_half():
    """⚠ A chunk-level split leaks (same talker, same room, two seconds later)
    AND is unrepresentable -- SequenceChunkSampler refuses a sequence whose
    chunk_index run has holes."""
    dataset = _SequenceStub([sid for sid in range(40) for _ in range(5)])
    train_idx, val_idx, train_seq, val_seq = sequence_level_split(dataset, 42)

    assert set(train_seq) & set(val_seq) == set()
    assert set(train_idx) & set(val_idx) == set()
    ids = dataset.sequence_ids()
    assert {ids[i] for i in train_idx} == set(train_seq)
    assert {ids[i] for i in val_idx} == set(val_seq)
    # every sequence contributes all of its chunks to exactly one side
    for side in (train_idx, val_idx):
        counts = {}
        for i in side:
            counts[ids[i]] = counts.get(ids[i], 0) + 1
        assert set(counts.values()) == {5}


def test_split_is_reproducible_from_the_seed():
    dataset = _SequenceStub([sid for sid in range(30) for _ in range(4)])
    assert sequence_level_split(dataset, 42) == sequence_level_split(dataset, 42)
    assert sequence_level_split(dataset, 42) != sequence_level_split(dataset, 7)


# ============================================================
# Loss pieces
# ============================================================

def _small_model_and_loss():
    cfg = load_config(n_fft=64, win_len=64, hop_len=32)
    cfg['model'].update(enc_channels='8', enc_stages='2', rnn_hidden='16',
                        rnn_layers='1', df_band_hz='2000')
    grid = AecGrid.from_config(cfg)
    return cfg, grid, JointAECNR.from_config(cfg, grid), JointLoss(grid, cfg)


def _stems_with_reference(present: bool, samples=4096):
    torch.manual_seed(2)
    raw = torch.randn(1, len(STEM_ORDER), samples) * 0.05
    index = {name: i for i, name in enumerate(STEM_ORDER)}
    if not present:
        # ⚠ X and D are zeroed TOGETHER, which is what the generator does for
        # ref_dropout (ref_dropout_echo_continues_p = 0).  Zeroing only X would
        # be the pathological case that trains the model to hallucinate.
        raw[:, index['far_render']] = 0.0
        raw[:, index['echo']] = 0.0
    raw[:, index['mic_postclip']] = (raw[:, index['near_speech']]
                                     + raw[:, index['local_noise']]
                                     + raw[:, index['echo']])
    return AecStems(raw)


def test_idle_term_fires_only_when_the_reference_is_silent():
    """⚠ The term is not decorative and it is not vacuous.

    It must appear on ref-idle data (or the `ref_dropout` scenario buys
    nothing) and must NOT appear on ordinary data (or it is really a global
    reweighting of the echo term wearing a different name).
    """
    cfg, grid, model, criterion = _small_model_and_loss()
    center = cfg.getboolean('signal', 'center')

    silent = stem_spectra(_stems_with_reference(False), grid, center)
    outputs, _ = model(silent['mic_postclip'], silent['far_render'])
    _, parts = criterion(outputs, silent)
    assert 'idle' in parts, 'the idle term never fired on a silent reference'

    active = stem_spectra(_stems_with_reference(True), grid, center)
    outputs, _ = model(active['mic_postclip'], active['far_render'])
    _, parts = criterion(outputs, active)
    assert 'idle' not in parts, 'the idle term fired on an active reference'


def test_loss_survives_a_backward_through_exact_zeros():
    """⚠ The reference-gated echo head emits exact 0.0, and the loss raises
    magnitudes to a fractional power, whose derivative at zero is infinite.
    One backward over ref-idle data then reaches every parameter through the
    shared trunk -- a first-step killer, not a rare edge.  Verified to fail
    against the unfloored form."""
    _, grid, model, criterion = _small_model_and_loss()
    specs = stem_spectra(_stems_with_reference(False), grid, center=False)
    outputs, _ = model(specs['mic_postclip'], specs['far_render'])
    loss, _ = criterion(outputs, specs)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f'{name} grad is NaN/inf'


def test_causal_ema_is_causal_and_uses_the_configured_time_constant():
    alpha = alpha_from_tau(0.2, 256, 16000)
    power = torch.zeros(1, 2, 10)
    power[..., 5] = 1.0
    smoothed = causal_ema(power, alpha)
    assert (smoothed[..., :5] == 0).all(), 'a future impulse leaked backwards'
    assert smoothed[0, 0, 5] > 0
    assert smoothed[0, 0, 6] < smoothed[0, 0, 5]


# ============================================================
# End to end
# ============================================================

def _write_shard(path, n_sequences=6, chunks_per_sequence=3, samples=2048,
                 sr=16000):
    torch.manual_seed(0)
    data, meta = [], []
    for sequence in range(n_sequences):
        for chunk in range(chunks_per_sequence):
            stems = torch.randn(len(STEM_ORDER), samples) * 0.05
            # mic_preclip == S + N + D and mic_postclip == mic_preclip, so the
            # shard obeys the same identity the generator guarantees.
            index = {name: i for i, name in enumerate(STEM_ORDER)}
            mic = (stems[index['near_speech']] + stems[index['local_noise']]
                   + stems[index['echo']])
            stems[index['mic_preclip']] = mic
            stems[index['mic_postclip']] = mic
            data.append(stems)
            meta.append({
                'sequence_id': sequence, 'chunk_index': chunk,
                'speaker_id': f's{sequence}', 'noise_id': 'n', 'rir_id': 'r',
                'ser_db': 0.0, 'snr_db': 10.0, 'erl_db': 12.0,
                'bulk_delay_samples': 320, 'delay_jitter': False,
                'sro_ppm': 0.0, 'nonlinear': 'linear', 'clipped': False,
                'scenario': 'double_talk',
            })
    torch.save({'stems': list(STEM_ORDER),
                'data': torch.stack(data).float(), 'sr': sr, 'meta': meta,
                'generator_commit': 'test', 'config_hash': 'test'}, path)


def test_two_real_training_steps():
    cfg = load_config(n_fft=64, win_len=64, hop_len=32)
    cfg['model']['enc_channels'] = '8'
    cfg['model']['enc_stages'] = '2'
    cfg['model']['rnn_hidden'] = '16'
    cfg['model']['rnn_layers'] = '1'
    cfg['model']['df_band_hz'] = '2000'
    grid = AecGrid.from_config(cfg)
    center = cfg.getboolean('signal', 'center')

    with tempfile.TemporaryDirectory() as tmp:
        _write_shard(os.path.join(tmp, 'shard_000.pt'), sr=grid.sr)
        dataset = PackedAecDataset(tmp, expected_sr=grid.sr, verbose=False)
        train_idx, val_idx, _, _ = sequence_level_split(dataset, 42)

        model = JointAECNR.from_config(cfg, grid)
        criterion = JointLoss(grid, cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader, sampler = make_sequence_loader(
            dataset, train_idx, n_lanes=2, seed=42, shuffle=True,
            worker_kwargs={'num_workers': 0, 'pin_memory': False})

        parts = run_epoch(model, loader, sampler, criterion, grid, center,
                          torch.device('cpu'), optimizer=optimizer,
                          max_steps=2, desc='test')

    for term in ('spec', 'mag', 'sisnr', 'echo', 'noise_psd', 'total'):
        assert term in parts, f'loss term {term} never fired'
        assert parts[term] == parts[term], f'{term} went NaN'


def test_lanes_walk_a_sequence_in_order():
    """Batch b holds chunk k of some sequence; batch b+1 holds chunk k+1."""
    cfg = load_config(n_fft=64, win_len=64, hop_len=32)
    grid = AecGrid.from_config(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        _write_shard(os.path.join(tmp, 'shard_000.pt'), sr=grid.sr)
        dataset = PackedAecDataset(tmp, expected_sr=grid.sr, verbose=False)
        train_idx, _, _, _ = sequence_level_split(dataset, 42)
        loader, sampler = make_sequence_loader(
            dataset, train_idx, n_lanes=2, seed=42, shuffle=False,
            worker_kwargs={'num_workers': 0, 'pin_memory': False})

        previous = None
        for _, metas in loader:
            current = [(m['sequence_id'], m['chunk_index']) for m in metas]
            if previous is not None:
                for (old_seq, old_chunk), (new_seq, new_chunk) in zip(previous,
                                                                      current):
                    same = old_seq == new_seq and new_chunk == old_chunk + 1
                    fresh = new_chunk == 0
                    assert same or fresh, (
                        f'lane jumped from {(old_seq, old_chunk)} to '
                        f'{(new_seq, new_chunk)}')
            previous = current


def test_batch_size_above_the_sequence_count_says_why():
    cfg = load_config(n_fft=64, win_len=64, hop_len=32)
    grid = AecGrid.from_config(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        _write_shard(os.path.join(tmp, 'shard_000.pt'), n_sequences=3,
                     sr=grid.sr)
        dataset = PackedAecDataset(tmp, expected_sr=grid.sr, verbose=False)
        with pytest.raises(ValueError, match='sequence LANES'):
            make_sequence_loader(dataset, list(range(len(dataset))), n_lanes=99,
                                 seed=42, shuffle=False,
                                 worker_kwargs={'num_workers': 0,
                                                'pin_memory': False})


def test_stem_spectra_are_keyed_by_the_declared_order():
    stems = AecStems(torch.randn(2, len(STEM_ORDER), 1024))
    grid = AecGrid(sr=16000, n_fft=64, win_len=64, hop_len=32)
    specs = stem_spectra(stems, grid, center=False)
    assert set(specs) == set(STEM_ORDER)
    assert specs['mic_postclip'].shape[1] == grid.n_freqs


# ============================================================
# Reconstruction
# ============================================================

@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('sr,n_fft', [(16000, 512), (48000, 1024)])
@pytest.mark.parametrize('length', [1000, 16003])
def test_denoise_round_trips_sample_for_sample(center, sr, n_fft, length):
    """⚠ The identity case has to be exact before any enhancement claim means
    anything: if the analysis/synthesis pair loses or shifts samples, every
    metric is measuring the framing as well as the model."""
    grid = AecGrid(sr=sr, n_fft=n_fft, win_len=n_fft, hop_len=n_fft // 2)
    signal = torch.randn(1, length)
    pad_front, pad_tail = analysis_pad(length, grid, center)
    padded = torch.nn.functional.pad(signal, (pad_front, pad_tail))
    spec = stft(padded, grid, center=center)
    restored = reconstruct(spec, grid, center, pad_front, length)
    assert restored.shape == signal.shape
    assert torch.allclose(restored, signal, atol=1e-4)


# ============================================================
# Bake-off protocol
# ============================================================

def test_shared_helpers_are_imported_not_redeclared():
    """Same drift guard ainr/tests/test_bakeoff_protocol.py applies to the NR
    trainers: one definition of the split, the seeder and the worker policy."""
    source = open(os.path.join(ROOT, 'train.py')).read()
    for declaration in ('def locality_preserving_random_split', 'def set_seed',
                        'class BlockShuffleSampler',
                        'def dataloader_worker_kwargs'):
        assert declaration not in source, (
            f'train.py re-declares "{declaration}"; import it from dataset_gen')
    assert 'from dataset_gen import' in source
    # The grid, the stem order and the sequence sampler are the AEC-side
    # equivalents and must come from one place too.
    for name in ('AecGrid', 'AecStems', 'SequenceChunkSampler'):
        assert f'class {name}' not in source


def test_seed_default_is_42():
    source = open(os.path.join(ROOT, 'train.py')).read()
    marker = "'--seed', type=int, default="
    assert marker in source
    assert int(source.split(marker, 1)[1].split(',')[0]) == 42


def test_config_states_durations_in_seconds_and_bands_in_hz():
    """⚠ A frame count in config.ini means one thing at 16 kHz and another at
    48 kHz, so the 48 kHz variant would quietly be a different model."""
    cfg = load_config()
    banned = ('lookahead_frames', 'ref_context_frames', 'df_order', 'df_bins',
              'echo_gate_memory_frames', 'noise_psd_alpha')
    for section in cfg.sections():
        for key in cfg[section]:
            assert key not in banned, (
                f'[{section}] {key} is a frame count; express it in seconds '
                f'or hertz and convert with the grid')
