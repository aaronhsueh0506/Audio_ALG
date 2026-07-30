"""Contract, config and loss tests for AECNet.

The loss tests are the substantive ones. Three of the four terms are guards
against failure modes that improve the headline number while making the product
worse -- an idle penalty that truncates echo tails, a symmetric near-end term
that prices cancelled speech like leftover echo, and a lookahead misalignment
that is indistinguishable from "the model does not converge".
"""

import configparser
import math
import os
import pathlib
import sys

import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AINR = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(1, AINR)

for _stale in ('train', 'denoise', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)

from model import _MAG_EPS, MODEL_KEYS, AecNetConfig, build_model  # noqa: E402
from train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    EchoEstimationLoss,
    _VERSION_FIELDS,
    _VERSIONS,
    build_contract,
    frame_level_db,
    read_loss_config,
    require_checkpoint_contract,
)

from dataset_gen_aec import AecGrid  # noqa: E402


GRID = AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256)


def load_config():
    cfg = configparser.ConfigParser()
    if not cfg.read(os.path.join(ROOT, 'config.ini')):
        raise AssertionError('AECNet/config.ini is missing')
    return cfg


# ============================================================
# Config round trip
# ============================================================

def test_every_model_key_is_consumed_and_nothing_else_exists():
    """[model] and MODEL_KEYS must agree exactly, in both directions.

    A key the file declares but nobody reads is a knob that silently does
    nothing; a key the code reads but the file omits is a default nobody
    reviewed. Both have shipped in this repo before.
    """
    cfg = load_config()
    declared = set(cfg['model'])
    assert declared == set(MODEL_KEYS), (
        f"config.ini [model] has {sorted(declared - set(MODEL_KEYS))} that the "
        f"model does not read, and is missing "
        f"{sorted(set(MODEL_KEYS) - declared)}")


def test_unknown_model_key_is_rejected():
    cfg = load_config()
    cfg.set('model', 'mystery_knob', '7')
    with pytest.raises(ValueError, match='mystery_knob'):
        AecNetConfig.from_config(cfg, AecGrid.from_config(cfg).frame_rate)


def test_config_round_trips_through_the_contract():
    cfg = load_config()
    model_cfg = AecNetConfig.from_config(cfg, AecGrid.from_config(cfg).frame_rate)
    contract = model_cfg.as_contract()
    assert contract['channels'] == ','.join(
        str(c) for c in model_cfg.channels)
    for key in ('kernel_t', 'kernel_f', 'stride_f', 'gru_layers', 'gru_groups',
                'lookahead', 'compress_exponent'):
        assert key in contract, f"{key} is not in the checkpoint contract"


def test_shipped_config_is_the_16k_grid():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    assert (grid.sr, grid.n_fft, grid.win_len, grid.hop_len) == (16000, 512, 512, 256)
    assert grid.n_freqs == 257
    assert grid.frame_rate == 62.5


def test_lookahead_is_a_duration_not_a_frame_count():
    """⚠ The config states lookahead in SECONDS; frames are derived.

    A config that said ``lookahead = 2`` would mean 32 ms at 16 kHz and 21 ms at
    48 kHz -- the same file describing two different algorithmic latencies. This
    pins that the same seconds buy the same MILLISECONDS on both grids, which is
    the whole reason the knob is a duration.
    """
    cfg = load_config()
    assert not cfg.has_option('model', 'lookahead'), (
        "[model] lookahead is a frame count; use lookahead_sec")
    cfg.set('model', 'lookahead_sec', '0.032')

    at_16k = AecNetConfig.from_config(cfg, AecGrid.from_config(cfg).frame_rate)
    assert at_16k.lookahead == 2          # 0.032 s * 62.5 fps

    cfg.set('signal', 'sr', '48000')
    cfg.set('signal', 'n_fft', '1024')
    cfg.set('signal', 'win_len', '1024')
    cfg.set('signal', 'hop_len', '512')
    grid_48k = AecGrid.from_config(cfg)
    at_48k = AecNetConfig.from_config(cfg, grid_48k.frame_rate)

    assert at_48k.lookahead == 3          # 0.032 s * 93.75 fps
    ms_16k = 1000.0 * at_16k.lookahead / 62.5
    ms_48k = 1000.0 * at_48k.lookahead / grid_48k.frame_rate
    assert abs(ms_16k - ms_48k) < 5.0, (
        f"same lookahead_sec gives {ms_16k:.1f} ms at 16 kHz and "
        f"{ms_48k:.1f} ms at 48 kHz")


# ============================================================
# Checkpoint contract
# ============================================================

def _contract():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    return cfg, grid, build_contract(cfg, grid,
                                     AecNetConfig.from_config(cfg, grid.frame_rate),
                                     read_loss_config(cfg))


def test_version_fields_are_derived_not_restated():
    """⚠ Adding a fourth version string must not be able to leave it unchecked."""
    assert tuple(_VERSIONS) == _VERSION_FIELDS
    assert set(_VERSION_FIELDS) == {
        'model_version', 'feature_version', 'loss_version'}
    assert _VERSIONS == {'model_version': MODEL_VERSION,
                         'feature_version': FEATURE_VERSION,
                         'loss_version': LOSS_VERSION}
    _, _, contract = _contract()
    for field in _VERSION_FIELDS:
        assert field in contract
        assert contract[field] == _VERSIONS[field]


def test_matching_checkpoint_is_accepted():
    _, _, contract = _contract()
    require_checkpoint_contract(dict(contract), contract)


@pytest.mark.parametrize('field,value', [
    ('model_version', 'something_else_v9'),
    ('feature_version', 'something_else_v9'),
    ('loss_version', 'something_else_v9'),
])
def test_mismatched_version_is_rejected(field, value):
    _, _, contract = _contract()
    ckpt = dict(contract)
    ckpt[field] = value
    with pytest.raises(ValueError, match=field):
        require_checkpoint_contract(ckpt, contract)


def test_missing_version_is_rejected():
    """No allow_missing escape hatch: this project vendors no foreign weights."""
    _, _, contract = _contract()
    ckpt = dict(contract)
    del ckpt['model_version']
    with pytest.raises(ValueError, match='pre-contract'):
        require_checkpoint_contract(ckpt, contract)


@pytest.mark.parametrize('field,value', [
    ('n_fft', 1024),
    ('hop_len', 128),
    ('compress_exponent', 0.5),
    ('lookahead', 2),
    ('channels', '16,32'),
    ('loss_lambda_idle', 0.0),
    ('loss_idle_guard_sec', 0.1),
])
def test_semantic_change_is_rejected(field, value):
    """Including the LOSS WEIGHTS.

    Resuming a lambda_idle = 1.0 run under lambda_idle = 0.0 is a different
    objective applied to the same weights; nothing about the shapes or the
    version strings would notice.
    """
    _, _, contract = _contract()
    assert field in contract, f"{field} is not in the contract at all"
    ckpt = dict(contract)
    ckpt[field] = value
    with pytest.raises(ValueError, match=field):
        require_checkpoint_contract(ckpt, contract)


def test_missing_contract_field_is_rejected():
    _, _, contract = _contract()
    ckpt = dict(contract)
    del ckpt['n_freqs']
    with pytest.raises(ValueError, match='n_freqs'):
        require_checkpoint_contract(ckpt, contract)


# ============================================================
# Shared-primitive drift guard (the same rule as ainr/tests)
# ============================================================

def test_train_does_not_redeclare_shared_primitives():
    source = pathlib.Path(ROOT, 'train.py').read_text()
    banned = ('def locality_preserving_random_split',
              'def set_seed',
              'class BlockShuffleSampler',
              'def dataloader_worker_kwargs',
              # The AEC-side equivalents: re-deriving any of these makes this
              # project's corpus subtly different from the other AEC models'.
              'class AecGrid',
              'class SequenceChunkSampler',
              'def stft',
              'STEM_ORDER =')
    for decl in banned:
        assert decl not in source, (
            f'AECNet/train.py re-declares "{decl}"; import it from dataset_gen '
            f'so every model shares one definition')
    assert 'from dataset_gen import' in source
    assert 'from dataset_gen_aec import' in source


def test_seed_default_is_42():
    source = pathlib.Path(ROOT, 'train.py').read_text()
    marker = "'--seed', type=int, default="
    assert marker in source
    tail = source.split(marker, 1)[1]
    assert int(tail.split(',')[0].split(')')[0].strip()) == 42


SHIPPED_FILES = ('config.ini', 'model.py', 'train.py', 'denoise.py', 'README.md',
                 'tests/test_aecnet_model.py', 'tests/test_aecnet_contract.py')


def test_no_absolute_filesystem_paths_are_written_into_any_file():
    """⚠ An absolute path leaks the checkout location, and with it the silicon
    vendor's name -- which must never appear in this repo, in code, comments,
    docs or paths. The platform is referred to only as 'the target platform'.
    Every path here is either relative or comes from a CLI argument, and this
    test is what keeps it that way; it covers the test files too, since a
    hardcoded fixture path is the usual way one slips in.
    """
    # Assembled rather than written out: this file is one of the files it
    # scans, so a literal needle here would make the guard fail on itself.
    roots = ['/' + part + '/' for part in ('Users', 'home', 'mnt', 'opt')]
    for name in SHIPPED_FILES:
        text = pathlib.Path(ROOT, name).read_text()
        for leak in roots:
            assert leak not in text, (
                f'{name} contains the absolute path fragment {leak!r}; use a '
                f'path relative to the file, or a CLI argument')


# ============================================================
# Frame grid
# ============================================================

@pytest.mark.parametrize('grid', [
    AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256),
    AecGrid(sr=48000, n_fft=1024, win_len=1024, hop_len=512),
])
def test_frame_level_db_lands_on_the_stft_grid(grid):
    """⚠ No frame count is hardcoded; the pooling geometry derives it."""
    n_samples = grid.hop_len * 37
    wav = torch.randn(3, n_samples)
    assert frame_level_db(wav, grid).shape == (3, grid.n_frames(n_samples))


def test_frame_level_db_is_in_dbfs():
    wav = torch.full((1, 16000), 0.1)
    db = frame_level_db(wav, GRID)
    # Interior frames are a full window of constant 0.1 -> -20 dBFS.
    assert abs(float(db[0, 20]) + 20.0) < 0.1
    assert float(frame_level_db(torch.zeros(1, 16000), GRID).max()) < -150.0


# ============================================================
# Loss
# ============================================================

def _loss(**overrides):
    kwargs = dict(compress_exponent=0.3, mag_weight=1.0, lambda_out=1.0,
                  lambda_near=0.5, lambda_idle=1.0, idle_guard_sec=0.5,
                  far_active_dbfs=-60.0, near_active_dbfs=-50.0, lookahead=0)
    kwargs.update(overrides)
    return EchoEstimationLoss(GRID, **kwargs)


def _spectra(n_frames, n_freqs=257, batch=2, scale=1.0):
    return torch.randn(batch, n_freqs, n_frames, dtype=torch.complex64) * scale


def _wav(n_frames, amplitude, batch=2):
    n_samples = (n_frames - 1) * GRID.hop_len
    return torch.randn(batch, n_samples) * amplitude


def test_idle_guard_is_expressed_in_seconds_not_frames():
    """⚠ 1.5 s must be 1.5 s of WALL CLOCK on both grids.

    The two grids run at different frame rates -- 62.5 fps at 16 kHz/hop 256 and
    93.75 fps at 48 kHz/hop 512 -- so the same duration is a DIFFERENT number of
    frames. That difference is the whole reason the knob is in seconds: a
    literal frame count copied across would silently shorten the guard by a
    third at 48 kHz and start training echo-tail truncation.
    """
    at16 = EchoEstimationLoss(
        AecGrid(16000, 512, 512, 256), 0.3, 1.0, 1.0, 0.5, 1.0, 1.5, -60, -50)
    at48 = EchoEstimationLoss(
        AecGrid(48000, 1024, 1024, 512), 0.3, 1.0, 1.0, 0.5, 1.0, 1.5, -60, -50)
    assert at16.guard_frames != at48.guard_frames, (
        'the frame counts are equal, so the guard is not being converted at all')
    for loss in (at16, at48):
        seconds = loss.guard_frames / loss.grid.frame_rate
        assert abs(seconds - 1.5) < 0.02, f'guard is {seconds:.3f} s, not 1.5 s'


def _idle_frames(loss, far_wav, n_frames, far_tail=None):
    d_hat = _spectra(n_frames)
    _, parts, tail = loss(d_hat, _spectra(n_frames), _spectra(n_frames),
                          _spectra(n_frames), far_wav, _wav(n_frames, 0.1),
                          far_tail)
    return parts['idle_frames'], tail


def test_idle_mask_extremes():
    loss = _loss()
    n = 40
    batch = 2
    silent, _ = _idle_frames(loss, torch.zeros(batch, (n - 1) * GRID.hop_len), n)
    assert silent == n * batch, "an entirely silent reference must be all idle"
    loud, _ = _idle_frames(loss, _wav(n, 0.2), n)
    assert loud == 0.0, "a continuously active reference must have no idle frame"


def test_idle_guard_delays_idleness_after_the_reference_stops():
    """⚠ The echo TAIL is still arriving when the reference goes quiet.

    Without the guard the model is punished for estimating that tail, i.e.
    trained to truncate reverberant echo. This compares a long guard against
    effectively none, on the same signal.
    """
    n = 150                    # 0.5 s of guard is 31 frames; the chunk must be
    far = torch.zeros(2, (n - 1) * GRID.hop_len)   # comfortably longer than that
    active = far.shape[1] // 3
    far[:, :active] = torch.randn(2, active) * 0.2

    guarded, _ = _idle_frames(_loss(idle_guard_sec=0.5), far, n)
    unguarded, _ = _idle_frames(_loss(idle_guard_sec=0.0), far, n)
    assert 0 < guarded < unguarded, (
        f"guarded={guarded} unguarded={unguarded}: the guard must shrink the "
        f"idle region, not leave it unchanged or erase it")
    # Two lanes, so the shrinkage is 2 x (guard_frames - 1) frames.
    assert unguarded - guarded == pytest.approx(2 * (31 - 1), abs=2)


def test_idle_guard_carries_across_chunk_boundaries():
    """⚠ Without the carried tail, every chunk starts looking idle.

    A chunk of pure silence that FOLLOWS a loud chunk is still inside the guard
    window for its first frames. Dropping the tail would mark them idle and
    penalise the tail of the previous chunk's echo -- once per chunk, forever.
    """
    n = 40
    loss = _loss(idle_guard_sec=0.5)
    loud = _wav(n, 0.2)
    silence = torch.zeros(2, (n - 1) * GRID.hop_len)

    _, tail = _idle_frames(loss, loud, n)
    with_history, _ = _idle_frames(loss, silence, n, far_tail=tail)
    without_history, _ = _idle_frames(loss, silence, n, far_tail=None)
    assert with_history < without_history
    assert tail.shape[1] == loss.guard_tail_width()


def test_near_end_preservation_is_asymmetric():
    """Only energy REMOVED from the near path is penalised.

    Residual echo left in E is recoverable downstream; near speech this stage
    cancelled is not. A symmetric term would price them the same.
    """
    n = 30
    torch.manual_seed(3)
    target = _spectra(n, scale=1.0)          # S + N
    echo = _spectra(n, scale=1.0)            # D
    y_spec = target + echo
    far = _wav(n, 0.2)
    near = _wav(n, 0.2)
    loss = _loss(lambda_near=1.0)

    def near_term(d_hat):
        _, parts, _ = loss(d_hat, y_spec, echo, target, far, near)
        return parts['near']

    perfect = near_term(echo)                      # E == S + N exactly
    over = near_term(echo + 0.5 * target)          # E == 0.5 (S+N): speech removed
    under = near_term(echo - 0.5 * target)         # E == 1.5 (S+N): echo left behind

    assert perfect == pytest.approx(0.0, abs=1e-6)
    assert over > 1e-4, "removing half the near path was not penalised"
    assert under == pytest.approx(0.0, abs=1e-6), (
        "leaving extra energy in E was penalised; the near term must be "
        "one-sided")


# A perfect estimate does NOT drive the L1 echo term to exactly zero: the
# complex modulus is floored inside its sqrt, so the minimum attainable value is
# sqrt(_MAG_EPS) per element.  That floor is deliberate -- sqrt(x^2 + eps) is
# the Charbonnier / pseudo-Huber form, whose gradient is +-1 away from zero (a
# true L1) and smoothly vanishes at zero instead of being undefined there, which
# matters because the idle term drives the output to exact zero by design.
_L1_FLOOR = math.sqrt(_MAG_EPS)


def test_perfect_echo_estimate_drives_every_term_to_its_floor():
    n = 30
    torch.manual_seed(5)
    target = _spectra(n)
    echo = _spectra(n)
    y_spec = target + echo
    loss = _loss()
    total, parts, _ = loss(echo, y_spec, echo, target, _wav(n, 0.2), _wav(n, 0.2))

    # L_output is L2 and really does reach zero.
    assert parts['out'] < 1e-9
    # L_echo is L1 (see LOSS_VERSION v2) and reaches its analytic floor, which
    # is six orders of magnitude below any realistic residual -- assert it is at
    # the floor rather than merely small, so a future change that reintroduces a
    # real error here cannot hide inside a loose threshold.
    # Exactly ONE floor, not two: the magnitude term is |mag(a) - mag(b)|, and
    # for identical inputs that difference is exactly 0 -- safe_mag's own eps
    # floors each magnitude but cancels in the subtraction. Only the complex
    # modulus, which floors INSIDE its sqrt, cannot reach zero.
    assert parts['echo'] == pytest.approx(_L1_FLOOR, rel=1e-3), (
        f"expected the complex-modulus floor sqrt(_MAG_EPS) = {_L1_FLOOR:g}, "
        f"got {parts['echo']}")
    assert float(total) < 1e-4


def test_loss_realigns_for_the_model_lookahead():
    """⚠ A lookahead mismatch is indistinguishable from a model that won't train.

    With lookahead L the model's output frame i belongs to input frame i - L.
    A loss that compares them index-for-index scores a target shifted by L
    frames, so the objective can never reach zero and nothing in the run says
    why.
    """
    n = 24
    look = 3
    torch.manual_seed(7)
    target = _spectra(n)
    echo = _spectra(n)
    y_spec = target + echo

    delayed = torch.zeros_like(echo)
    delayed[..., look:] = echo[..., :n - look]

    aligned = _loss(lookahead=look)
    total, parts, _ = aligned(delayed, y_spec, echo, target,
                              _wav(n, 0.2), _wav(n, 0.2))
    # At the L1 floor, not at zero -- see _L1_FLOOR above.
    assert parts['echo'] == pytest.approx(_L1_FLOOR, rel=1e-3), (
        "the aligned loss did not see a perfect estimate")

    naive = _loss(lookahead=0)
    _, naive_parts, _ = naive(delayed, y_spec, echo, target,
                              _wav(n, 0.2), _wav(n, 0.2))
    assert naive_parts['echo'] > 1e-3, (
        "the unaligned loss scored a shifted estimate as perfect; this test "
        "cannot detect the bug it exists for")


def test_loss_backward_is_finite_through_a_real_model():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    model = build_model(cfg, grid)
    n = 12
    y_spec = _spectra(n, grid.n_freqs, batch=1, scale=0.05)
    x_spec = _spectra(n, grid.n_freqs, batch=1, scale=0.05)
    echo = _spectra(n, grid.n_freqs, batch=1, scale=0.05)
    target = _spectra(n, grid.n_freqs, batch=1, scale=0.05)
    d_hat, _ = model.forward_spec(y_spec, x_spec)
    loss = _loss()
    total, _, _ = loss(d_hat, y_spec, echo, target,
                       _wav(n, 0.2, batch=1), _wav(n, 0.2, batch=1))
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, 'no parameter received a gradient'
    assert all(torch.isfinite(g).all() for g in grads)


def test_read_loss_config_rejects_negative_weights():
    cfg = load_config()
    cfg.set('training', 'lambda_idle', '-1')
    with pytest.raises(ValueError, match='lambda_idle'):
        read_loss_config(cfg)


if __name__ == '__main__':
    raise SystemExit(pytest.main([str(pathlib.Path(__file__)), '-q']))
