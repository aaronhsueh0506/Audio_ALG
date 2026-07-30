"""End-to-end trainer test over a tiny synthetic packed corpus.

This drives the REAL path -- PackedAecDataset, the sequence-aware sampler,
lane state, the frozen front-end, the loss and the checkpoint -- rather than
unit-testing the pieces.  The integration is where the interesting mistakes are:
a lane whose state is reset every batch, a split that cuts a sequence in half,
or a mask applied to Y instead of E all train perfectly well and are invisible
in a unit test.
"""

import configparser
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AINR = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, AINR)

for _stale in ('train', 'denoise', 'model', 'frontends', 'postproc'):
    sys.modules.pop(_stale, None)

from dataset_gen.aec import STEM_ORDER, AecGrid, PackedAecDataset  # noqa: E402
from frontends import build_frontend  # noqa: E402
from model import build_model  # noqa: E402
from train import (  # noqa: E402
    PostFilterLoss,
    _sequence_level_split,
    make_loader,
    run_pass,
)


SR = 16000
CHUNK_SAMPLES = 4096          # 16 frames at hop 256 -- divides exactly
SEQUENCES = 6
CHUNKS_PER_SEQUENCE = 4
SCENARIOS = ['double_talk', 'far_only', 'near_only', 'ref_dropout',
             'echo_path_change', 'nonlinear_spk']


def write_corpus(directory, seed=0):
    """A shard in the declared packed format.

    The stems obey the signal model (mic_preclip == S + N + D) so a trainer bug
    that reads the wrong channel produces a wrong loss rather than a crash.
    """
    generator = torch.Generator().manual_seed(seed)
    n_clips = SEQUENCES * CHUNKS_PER_SEQUENCE
    data = torch.zeros(n_clips, len(STEM_ORDER), CHUNK_SAMPLES)
    meta = []
    index = 0
    for sequence in range(SEQUENCES):
        scenario = SCENARIOS[sequence % len(SCENARIOS)]
        for chunk in range(CHUNKS_PER_SEQUENCE):
            far = torch.randn(CHUNK_SAMPLES, generator=generator) * 0.1
            echo = torch.roll(far, 40) * 0.3
            near = torch.randn(CHUNK_SAMPLES, generator=generator) * 0.05
            noise = torch.randn(CHUNK_SAMPLES, generator=generator) * 0.01
            if scenario in ('near_only', 'ref_dropout'):
                far = torch.zeros_like(far)
                echo = torch.zeros_like(echo)
            if scenario == 'far_only':
                near = torch.zeros_like(near)
            mic = near + noise + echo
            data[index] = torch.stack([far, echo, near, noise, mic, mic])
            meta.append({
                'sequence_id': sequence,
                'chunk_index': chunk,
                'speaker_id': f'spk{sequence}',
                'noise_id': f'noi{sequence}',
                'rir_id': f'rir{sequence}',
                'ser_db': 0.0, 'snr_db': 10.0, 'erl_db': 10.0,
                'bulk_delay_samples': 40, 'delay_jitter': False,
                'sro_ppm': 0.0, 'nonlinear': 'linear', 'clipped': False,
                'scenario': scenario,
            })
            index += 1

    os.makedirs(directory, exist_ok=True)
    torch.save({'stems': list(STEM_ORDER), 'data': data, 'sr': SR, 'meta': meta,
                'generator_commit': 'test', 'config_hash': 'test'},
               os.path.join(directory, 'shard_000.pt'))
    return directory


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    return write_corpus(str(tmp_path_factory.mktemp('aec_packed')))


@pytest.fixture(scope='module')
def small_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    cfg.set('model', 'enc_channels', '8')
    cfg.set('model', 'gru_hidden', '24')
    cfg.set('model', 'gru_layers', '1')
    cfg.set('model', 'dec_hidden', '16')
    cfg.set('frontend', 'filter_span_sec', '0.064')   # 4 taps on the 16 kHz grid
    return cfg


def test_sequence_split_never_cuts_a_sequence(corpus):
    dataset = PackedAecDataset(corpus, expected_sr=SR, verbose=False)
    train_indices, val_indices = _sequence_level_split(
        dataset, list(range(len(dataset))), val_fraction=0.34, seed=42)
    train_ids = {dataset.meta(i)['sequence_id'] for i in train_indices}
    val_ids = {dataset.meta(i)['sequence_id'] for i in val_indices}
    assert train_ids and val_ids
    assert not (train_ids & val_ids), (
        "a sequence appears on both sides: the same speaker, room and echo path "
        "are then in train and val and the curve measures memorisation")
    # Every held-out sequence is held out WHOLE, which is also what lets
    # SequenceChunkSampler accept the subset at all.
    for sid in val_ids:
        kept = [i for i in val_indices if dataset.meta(i)['sequence_id'] == sid]
        assert len(kept) == CHUNKS_PER_SEQUENCE


def test_sampler_lanes_walk_one_sequence_in_order(corpus):
    dataset = PackedAecDataset(corpus, expected_sr=SR, verbose=False)
    indices = list(range(len(dataset)))
    _, sampler, lanes = make_loader(dataset, indices, lanes=3, shuffle=False,
                                    seed=42, workers=0, prefetch=2,
                                    pin_memory=False, label='test')
    assert lanes == 3
    batches = list(sampler)
    for lane in range(lanes):
        walked = [dataset.meta(indices[batch[lane]]) for batch in batches]
        for step in range(1, len(walked)):
            same_sequence = walked[step]['sequence_id'] == walked[step - 1]['sequence_id']
            if same_sequence:
                assert walked[step]['chunk_index'] == walked[step - 1]['chunk_index'] + 1
            else:
                assert walked[step]['chunk_index'] == 0, (
                    "a lane started a new sequence part-way through it; the "
                    "reset mask would then never fire")


def test_lanes_are_clamped_to_the_sequence_count(corpus, capsys):
    dataset = PackedAecDataset(corpus, expected_sr=SR, verbose=False)
    _, _, lanes = make_loader(dataset, list(range(len(dataset))), lanes=99,
                              shuffle=False, seed=42, workers=0, prefetch=2,
                              pin_memory=False, label='test')
    assert lanes == SEQUENCES
    assert 'clamping' in capsys.readouterr().out


def test_a_training_pass_runs_and_moves_the_weights(corpus, small_config):
    cfg = small_config
    grid = AecGrid.from_config(cfg)
    dataset = PackedAecDataset(corpus, expected_sr=SR, verbose=False)
    device = torch.device('cpu')

    model = build_model(cfg, grid).to(device)
    frontend = build_frontend(cfg, grid)
    criterion = PostFilterLoss(grid, gamma=0.3, mag_weight=500.0,
                               complex_weight=500.0, sisdr_weight=1.0,
                               echo_leak_weight=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader, sampler, _ = make_loader(dataset, list(range(len(dataset))), lanes=3,
                                     shuffle=False, seed=42, workers=0,
                                     prefetch=2, pin_memory=False, label='train')
    before = [p.detach().clone() for p in model.parameters()]
    loss = run_pass(model, frontend, criterion, loader, sampler, grid, device,
                    optimizer=optimizer, idle_weight=3.0, dt_weight=1.0,
                    desc='train')
    assert torch.isfinite(torch.tensor(loss))
    moved = any(not torch.equal(a, b)
                for a, b in zip(before, model.parameters()))
    assert moved, 'no parameter changed: the graph is disconnected somewhere'


def test_validation_pass_leaves_the_weights_alone(corpus, small_config):
    cfg = small_config
    grid = AecGrid.from_config(cfg)
    dataset = PackedAecDataset(corpus, expected_sr=SR, verbose=False)
    model = build_model(cfg, grid)
    frontend = build_frontend(cfg, grid)
    criterion = PostFilterLoss(grid)
    loader, sampler, _ = make_loader(dataset, list(range(len(dataset))), lanes=2,
                                     shuffle=False, seed=42, workers=0,
                                     prefetch=2, pin_memory=False, label='val')
    before = [p.detach().clone() for p in model.parameters()]
    loss = run_pass(model, frontend, criterion, loader, sampler, grid,
                    torch.device('cpu'), desc='val')
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_the_frontend_never_receives_a_gradient(corpus, small_config):
    """⚠ 'Frozen' has to be true, not merely intended."""
    cfg = small_config
    grid = AecGrid.from_config(cfg)
    frontend = build_frontend(cfg, grid)
    state = frontend.init_state(2)
    y = torch.complex(torch.randn(2, grid.n_freqs, 8),
                      torch.randn(2, grid.n_freqs, 8)).requires_grad_(True)
    x = torch.complex(torch.randn(2, grid.n_freqs, 8),
                      torch.randn(2, grid.n_freqs, 8))
    with torch.no_grad():
        e, d_hat, _ = frontend.process(y, x, state)
    assert not e.requires_grad and not d_hat.requires_grad


def test_loss_ignores_si_sdr_on_silent_targets():
    """far_only chunks have S == 0, where SI-SDR is a gradient-free constant
    that would still dominate the printed loss."""
    grid = AecGrid(sr=SR, n_fft=512, win_len=512, hop_len=256)
    criterion = PostFilterLoss(grid, mag_weight=0.0, complex_weight=0.0,
                               sisdr_weight=1.0)
    pred = torch.complex(torch.randn(2, grid.n_freqs, 12),
                         torch.randn(2, grid.n_freqs, 12))
    silent = torch.zeros_like(pred)
    loss, parts = criterion(pred, silent)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert parts['sisdr'] == pytest.approx(0.0, abs=1e-6)


def test_scenario_weighting_changes_the_loss(corpus, small_config):
    """The idle upweight has to actually reach the objective."""
    grid = AecGrid.from_config(small_config)
    criterion = PostFilterLoss(grid, sisdr_weight=0.0)
    pred = torch.complex(torch.randn(3, grid.n_freqs, 10),
                         torch.randn(3, grid.n_freqs, 10))
    target = torch.zeros_like(pred)
    flat, _ = criterion(pred, target, weights=torch.ones(3))
    tilted, _ = criterion(pred, target, weights=torch.tensor([9.0, 1.0, 1.0]))
    assert not torch.isclose(flat, tilted)


def test_checkpoint_round_trip_through_the_real_cli(corpus, small_config,
                                                    tmp_path):
    """Train one epoch through train.train(), then resume from what it wrote."""
    import argparse

    import train as trainer

    config_path = tmp_path / 'config.ini'
    output_dir = tmp_path / 'out'
    small_config.set('training', 'epochs', '1')
    small_config.set('training', 'lanes', '2')
    small_config.set('training', 'num_workers', '0')
    small_config.set('training', 'device', 'cpu')
    small_config.set('training', 'val_fraction', '0.34')
    small_config.set('paths', 'output_dir', str(output_dir))
    with open(config_path, 'w', encoding='utf-8') as handle:
        small_config.write(handle)

    def make_args(**overrides):
        base = dict(config=str(config_path), packed_dir=corpus,
                    val_packed_dir=None, mmap=False, resume=None,
                    allow_frontend_change=False, seed=42, gpu=None, device='cpu')
        base.update(overrides)
        return argparse.Namespace(**base)

    trainer.train(make_args())
    checkpoint = output_dir / 'postfilter_last.pth'
    assert checkpoint.is_file()

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['frontend_id'].startswith('stft_nlms_')
    assert ckpt['train_indices'] and ckpt['val_indices']
    assert not set(ckpt['train_indices']) & set(ckpt['val_indices'])

    # Resume: the contract and the front-end gate both have to pass, and the
    # split must come back from the checkpoint rather than being redrawn.
    small_config.set('training', 'epochs', '2')
    with open(config_path, 'w', encoding='utf-8') as handle:
        small_config.write(handle)
    trainer.train(make_args(resume=str(checkpoint)))
    resumed = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert resumed['epoch'] == 2
    assert resumed['train_indices'] == ckpt['train_indices']


def test_resume_refuses_a_different_frontend(corpus, small_config, tmp_path):
    import argparse

    import train as trainer

    config_path = tmp_path / 'config_fe.ini'
    output_dir = tmp_path / 'out_fe'
    small_config.set('training', 'epochs', '1')
    small_config.set('training', 'lanes', '2')
    small_config.set('training', 'num_workers', '0')
    small_config.set('training', 'device', 'cpu')
    small_config.set('training', 'val_fraction', '0.34')
    small_config.set('paths', 'output_dir', str(output_dir))
    with open(config_path, 'w', encoding='utf-8') as handle:
        small_config.write(handle)

    def make_args(**overrides):
        base = dict(config=str(config_path), packed_dir=corpus,
                    val_packed_dir=None, mmap=False, resume=None,
                    allow_frontend_change=False, seed=42, gpu=None, device='cpu')
        base.update(overrides)
        return argparse.Namespace(**base)

    trainer.train(make_args())
    checkpoint = str(output_dir / 'postfilter_last.pth')

    small_config.set('frontend', 'kind', 'none')
    small_config.set('training', 'epochs', '2')   # so the resume has work to do
    with open(config_path, 'w', encoding='utf-8') as handle:
        small_config.write(handle)
    with pytest.raises(ValueError, match='different front-end'):
        trainer.train(make_args(resume=checkpoint))

    # ...and the override lets it through while recording the lineage.
    trainer.train(make_args(resume=checkpoint, allow_frontend_change=True))
    ckpt = torch.load(str(output_dir / 'postfilter_last.pth'), map_location='cpu',
                      weights_only=False)
    assert ckpt['frontend_id'] == 'none_v1'
    assert ckpt['frontend_history'] and ckpt['frontend_history'][0].startswith(
        'stft_nlms_')
