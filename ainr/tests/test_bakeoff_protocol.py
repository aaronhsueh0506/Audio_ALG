#!/usr/bin/env python3
"""Drift guard for the RNNoise-ERB vs GTCRN comparison protocol.

WHY THIS EXISTS
---------------
The two models are trained on the same packed 16 kHz corpus so that the
difference between their scores is attributable to the models.  Everything that
has to be equal for that to hold used to be equal only because two config files
happened to agree, with a ``⚠`` comment in each asking a human to keep them that
way.  They had already drifted once: the held-out fraction was 5% on one side
and 10% on the other, so the two models were trained on different corpora and
validated on different data.

This is the same guard ``RNNoise-ERB/tests/test_feature_contract.py`` applies to
the config/C constants, pointed at the pair of config files instead.

Deliberate divergences are asserted as divergences, not ignored -- ``center``
and ``batch_size`` differ on purpose, and pinning them here records that the
difference is a decision rather than an oversight.

Dependency-free: parses the .ini files directly, no torch import.
"""

import configparser
import pathlib
import sys


AINR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AINR))


def _load(name):
    cfg = configparser.ConfigParser()
    if not cfg.read(AINR / name / 'config.ini'):
        raise AssertionError(f'missing {name}/config.ini')
    return cfg


# Keys that MUST agree for the comparison to be attributable to the models.
SHARED_SIGNAL = ('sr', 'n_fft')
SHARED_TRAINING = ('epoch_size',)


def test_signal_grid_matches():
    """Both models must analyse the corpus on the same time-frequency grid."""
    rnn, gtcrn = _load('RNNoise-ERB'), _load('GTCRN')
    for key in SHARED_SIGNAL:
        a, b = rnn.getint('signal', key), gtcrn.getint('signal', key)
        assert a == b, f'[signal] {key}: RNNoise-ERB={a} GTCRN={b}'

    # win_len/hop_len are optional in both files and default off n_fft.
    for cfg, name in ((rnn, 'RNNoise-ERB'), (gtcrn, 'GTCRN')):
        n_fft = cfg.getint('signal', 'n_fft')
        win = cfg.getint('signal', 'win_len', fallback=n_fft)
        hop = cfg.getint('signal', 'hop_len', fallback=win // 2)
        assert win == n_fft, f'{name}: win_len {win} != n_fft {n_fft}'
        assert hop == win // 2, f'{name}: hop_len {hop} != win_len//2 {win // 2}'


def test_epoch_budget_matches():
    """Equal epoch_size means "100 epochs" is the same amount of data seen.

    It is NOT the same number of optimizer steps -- batch_size differs by
    design (see test_intentional_divergences_are_still_intentional).
    """
    rnn, gtcrn = _load('RNNoise-ERB'), _load('GTCRN')
    for key in SHARED_TRAINING:
        a = rnn.getint('training', key, fallback=0)
        b = gtcrn.getint('training', key, fallback=0)
        assert a == b, f'[training] {key}: RNNoise-ERB={a} GTCRN={b}'


def test_split_comes_from_one_shared_implementation():
    """Neither trainer may define its own split, sampler or seeder.

    The 5%-vs-10% drift happened because each train.py carried its own copy.
    Importing from dataset_gen is what makes the split a single definition;
    a re-declaration here would silently reintroduce the divergence.
    """
    banned = ('def locality_preserving_random_split',
              'def set_seed',
              'class BlockShuffleSampler',
              'def dataloader_worker_kwargs')
    for name in ('RNNoise-ERB', 'GTCRN', 'DeepFilterNet2'):
        source = (AINR / name / 'train.py').read_text()
        for decl in banned:
            assert decl not in source, (
                f'{name}/train.py re-declares "{decl}"; import it from '
                f'dataset_gen so all three models share one definition')
        assert 'from dataset_gen import' in source, (
            f'{name}/train.py does not import the shared loader')


def test_seed_defaults_match():
    """A different default seed would silently give a different split."""
    seeds = {}
    for name in ('RNNoise-ERB', 'GTCRN'):
        source = (AINR / name / 'train.py').read_text()
        marker = "'--seed', type=int, default="
        assert marker in source, f'{name}: no --seed default found'
        tail = source.split(marker, 1)[1]
        seeds[name] = int(tail.split(',')[0].split(')')[0].strip())
    assert len(set(seeds.values())) == 1, f'--seed defaults differ: {seeds}'


def test_intentional_divergences_are_still_intentional():
    """Pin the differences that are deliberate, so they read as decisions.

    - ``center``: RNNoise-ERB trains with center=False to match its streaming C
      path; GTCRN keeps upstream's center=True.  Unifying them would break one
      of the two against its own reference.
    - ``batch_size``: sized per model, so equal epoch_size gives equal samples
      seen but a different number of optimizer steps.
    """
    rnn = (AINR / 'RNNoise-ERB' / 'train.py').read_text()
    gtcrn = (AINR / 'GTCRN' / 'train.py').read_text()
    assert 'center=False' in rnn, 'RNNoise-ERB lost its center=False STFT'
    assert 'center=False' not in gtcrn, (
        'GTCRN gained center=False; upstream parity assumes center=True')

    sizes = {name: _load(name).getint('training', 'batch_size')
             for name in ('RNNoise-ERB', 'GTCRN')}
    assert len(set(sizes.values())) > 1, (
        f'batch_size is now equal ({sizes}); if that was intended, update this '
        f'test -- equal epoch_size then also means equal optimizer steps')


if __name__ == '__main__':
    tests = [
        test_signal_grid_matches,
        test_epoch_budget_matches,
        test_split_comes_from_one_shared_implementation,
        test_seed_defaults_match,
        test_intentional_divergences_are_still_intentional,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
