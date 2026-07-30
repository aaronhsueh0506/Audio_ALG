"""Checkpoint-contract, front-end-identity and caller-side-bound guards.

Every gate here exists because the failure it prevents is SILENT: a resume
across a feature change trains real-looking curves on meaningless weights, and a
checkpoint attached to the wrong front-end produces a number that looks like a
result and is not one.
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

from dataset_gen_aec import AecGrid  # noqa: E402
from frontends import NullFrontEnd, StftNlmsFrontEnd, build_frontend  # noqa: E402
from model import build_model  # noqa: E402
from postproc import (  # noqa: E402
    GainPostProcessor,
    apply_attenuation_cap,
    apply_gain_floor,
    db_to_gain,
)
from train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    _VERSION_FIELDS,
    build_contract,
    require_checkpoint_contract,
    require_frontend_match,
    scenario_weights,
    stream_stft,
)


def load_config():
    cfg = configparser.ConfigParser()
    assert cfg.read(os.path.join(ROOT, 'config.ini')), 'missing config.ini'
    return cfg


def reference_contract():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    return cfg, grid, build_contract(cfg, grid, build_model(cfg, grid))


# ============================================================
# Contract shape
# ============================================================

def test_version_fields_are_derived_not_restated():
    """Adding a version string must gate it automatically.

    A second hand-written tuple of names is how a version ends up recorded in
    the checkpoint but never compared against.
    """
    _, _, contract = reference_contract()
    from_contract = {k for k in contract if k.endswith('_version')}
    assert set(_VERSION_FIELDS) == from_contract
    assert contract['model_version'] == MODEL_VERSION
    assert contract['feature_version'] == FEATURE_VERSION
    assert contract['loss_version'] == LOSS_VERSION


def test_contract_records_the_resolved_encoder_depth():
    """'auto' must not reach the checkpoint: two configs that both say auto but
    differ in n_bands build different encoders."""
    _, _, contract = reference_contract()
    assert isinstance(contract['enc_downsamples'], int)
    assert contract['enc_downsamples'] >= 2


def test_contract_gate_accepts_its_own_output():
    _, _, contract = reference_contract()
    require_checkpoint_contract(dict(contract), contract, context='self')


@pytest.mark.parametrize('field', ['model_version', 'feature_version',
                                   'loss_version'])
def test_contract_gate_refuses_a_version_change(field):
    _, _, contract = reference_contract()
    ckpt = dict(contract)
    ckpt[field] = 'something_else'
    with pytest.raises(ValueError, match=field):
        require_checkpoint_contract(ckpt, contract, context='ckpt')


@pytest.mark.parametrize('field', ['n_fft', 'hop_len', 'n_bands',
                                   'mask_resolution', 'lookahead_frames',
                                   'coherence_tau_sec'])
def test_contract_gate_refuses_a_shape_or_feature_change(field):
    _, _, contract = reference_contract()
    ckpt = dict(contract)
    ckpt[field] = (contract[field] + 1 if isinstance(contract[field], (int, float))
                   else 'full')
    with pytest.raises(ValueError, match=field):
        require_checkpoint_contract(ckpt, contract, context='ckpt')


def test_contract_gate_refuses_a_missing_field():
    _, _, contract = reference_contract()
    ckpt = dict(contract)
    del ckpt['gru_hidden']
    with pytest.raises(ValueError, match='gru_hidden'):
        require_checkpoint_contract(ckpt, contract, context='ckpt')


def test_inference_may_ignore_the_loss_fields_but_nothing_else():
    """denoise.py runs with require_loss=False: what the objective was does not
    change what the weights compute.  The grid still does."""
    _, _, contract = reference_contract()
    ckpt = dict(contract)
    ckpt['echo_leak_weight'] = 99.0
    require_checkpoint_contract(ckpt, contract, context='ckpt', require_loss=False)
    with pytest.raises(ValueError):
        require_checkpoint_contract(ckpt, contract, context='ckpt', require_loss=True)

    ckpt = dict(contract)
    ckpt['sr'] = 48000
    with pytest.raises(ValueError, match='sr'):
        require_checkpoint_contract(ckpt, contract, context='ckpt',
                                    require_loss=False)


# ============================================================
# Front-end identity
# ============================================================

def test_frontend_gate_refuses_a_mismatched_resume():
    ckpt = {'frontend_id': 'stft_nlms_p16_mu0.35_leak0.9999_gate-60_v1'}
    with pytest.raises(ValueError, match='different front-end'):
        require_frontend_match(ckpt, 'none_v1', allow_change=False, context='ckpt')


def test_frontend_gate_refuses_a_checkpoint_that_records_nothing():
    with pytest.raises(ValueError, match='missing'):
        require_frontend_match({}, 'none_v1', allow_change=False, context='ckpt')


def test_frontend_gate_accepts_a_match_and_keeps_history():
    ckpt = {'frontend_id': 'none_v1', 'frontend_history': ['oracle_v1']}
    assert require_frontend_match(ckpt, 'none_v1', allow_change=False) == ['oracle_v1']


def test_override_records_an_ood_lineage_that_cannot_be_dropped():
    """⚠ The override is allowed; forgetting that it happened is not."""
    ckpt = {'frontend_id': 'oracle_v1'}
    history = require_frontend_match(ckpt, 'none_v1', allow_change=True)
    assert history == ['oracle_v1']
    # A later resume of the descendant inherits the lineage.
    descendant = {'frontend_id': 'none_v1', 'frontend_history': history}
    assert require_frontend_match(descendant, 'none_v1', allow_change=False) == \
        ['oracle_v1']


def test_frontend_ids_are_distinct_and_carry_their_settings():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    ids = {build_frontend(cfg, grid).frontend_id, NullFrontEnd().frontend_id,
           StftNlmsFrontEnd(grid, taps=8).frontend_id,
           StftNlmsFrontEnd(grid, taps=16, mu=0.1).frontend_id}
    assert len(ids) == 4, f"front-end ids collide: {ids}"


def test_configured_frontend_covers_the_corpus_bulk_delay():
    """⚠ The filter span must exceed the generator's bulk_delay_ms_max (120 ms)
    plus the room RIR, or every residual in the corpus is 'the filter could not
    reach the echo' rather than 'the filter did its best'.

    The config states the span in SECONDS and the taps are derived, so this
    holds on BOTH grids without a config edit -- which is exactly what the
    48 kHz leg below pins.  A taps count in the config would pass here at
    16 kHz and silently buy only 171 ms at 48 kHz.
    """
    cfg = load_config()
    assert not cfg.has_option('frontend', 'taps'), (
        "[frontend] taps is a frame count; use filter_span_sec")

    grid = AecGrid.from_config(cfg)
    frontend = build_frontend(cfg, grid)
    span_ms = 1000.0 * frontend.taps * grid.hop_len / grid.sr
    assert span_ms >= 250.0, f"only {span_ms:.0f} ms of echo-path coverage"

    cfg.set('signal', 'sr', '48000')
    cfg.set('signal', 'n_fft', '1024')
    cfg.set('signal', 'win_len', '1024')
    cfg.set('signal', 'hop_len', '512')
    grid_48k = AecGrid.from_config(cfg)
    span_48k_ms = (1000.0 * build_frontend(cfg, grid_48k).taps
                   * grid_48k.hop_len / grid_48k.sr)
    assert span_48k_ms >= 250.0, (
        f"48 kHz by config change alone gives only {span_48k_ms:.0f} ms of "
        f"echo-path coverage; the span stopped being rate-invariant")


# ============================================================
# The shared-implementation rule
# ============================================================

def test_split_seed_and_sampler_come_from_the_shared_module():
    """Same guard ainr/tests/test_bakeoff_protocol.py applies to the NR models."""
    source = open(os.path.join(ROOT, 'train.py'), encoding='utf-8').read()
    for declaration in ('def locality_preserving_random_split', 'def set_seed',
                        'class BlockShuffleSampler', 'def dataloader_worker_kwargs',
                        'class SequenceChunkSampler', 'class AecGrid',
                        'def alpha_from_tau'):
        assert declaration not in source, (
            f"train.py re-declares {declaration!r}; import it so every project "
            f"shares one definition")
    assert 'from dataset_gen import' in source
    assert 'from dataset_gen_aec import' in source


def test_seed_default_is_42():
    source = open(os.path.join(ROOT, 'train.py'), encoding='utf-8').read()
    marker = "'--seed', type=int, default="
    assert marker in source
    assert int(source.split(marker, 1)[1].split(',')[0].strip()) == 42


def test_config_is_a_valid_cola_grid_and_stays_config_driven():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    assert grid.hop_len == grid.win_len // 2
    assert grid.n_freqs == grid.n_fft // 2 + 1
    # The 48 kHz variant must work by config change alone.
    cfg.set('signal', 'sr', '48000')
    cfg.set('signal', 'n_fft', '1024')
    cfg.set('signal', 'win_len', '1024')
    cfg.set('signal', 'hop_len', '512')
    grid48 = AecGrid.from_config(cfg)
    model = build_model(cfg, grid48)
    assert model.grid.n_freqs == 513
    assert model.features.alpha_coh != build_model(load_config(),
                                                   grid).features.alpha_coh, (
        "the coherence alpha did not change with the frame rate; a time "
        "constant has been hardcoded in frames somewhere")


# ============================================================
# Streaming STFT
# ============================================================

def test_stream_stft_is_continuous_across_chunks():
    """Chunk n+1's frame grid must be the exact continuation of chunk n's.

    With center=True (torch's default) it is not, and the front-end sees a
    fabricated echo-path discontinuity every chunk.
    """
    grid = AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256)
    wave = torch.randn(2, 3, 2048)
    whole, _ = stream_stft(wave, grid)
    first, tail = stream_stft(wave[..., :1024], grid)
    second, _ = stream_stft(wave[..., 1024:], grid, tail)
    joined = torch.cat([first, second], dim=-1)
    assert joined.shape == whole.shape
    assert torch.allclose(joined, whole, atol=1e-5)


# ============================================================
# Caller-side bounds
# ============================================================

def test_gain_floor_uses_twenty_log_ten():
    """⚠ dB on a GAIN is /20.  The classical NR shipped the /10 form once."""
    assert db_to_gain(-20.0) == pytest.approx(0.1)
    assert db_to_gain(-6.0) == pytest.approx(0.5011872)


def test_floor_and_cap_are_the_same_clamp_from_two_sides():
    gain = torch.linspace(0.0, 1.0, 21)
    assert torch.allclose(apply_gain_floor(gain, -20.0),
                          apply_attenuation_cap(gain, 20.0))


def test_floor_preserves_the_phase_of_a_complex_mask():
    mask = torch.complex(torch.tensor([0.001, 0.5]), torch.tensor([0.001, -0.2]))
    floored = apply_gain_floor(mask, -20.0)
    assert floored.abs().min() >= db_to_gain(-20.0) - 1e-6
    assert torch.allclose(torch.angle(floored), torch.angle(mask), atol=1e-6)


def test_disabled_bounds_are_identities():
    gain = torch.rand(4, 8)
    assert torch.equal(apply_gain_floor(gain, None), gain)
    assert torch.equal(apply_attenuation_cap(gain, 0), gain)


def test_smoothing_cannot_reintroduce_a_violation_of_the_floor():
    """The order floor -> cap -> smooth is safe because smoothing is a convex
    combination of values that already meet the floor.  Reversing it is not."""
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    processor = GainPostProcessor(gain_floor_db=-25.0, attack_tau_sec=0.05,
                                  release_tau_sec=0.2, hop_len=grid.hop_len,
                                  sr=grid.sr)
    gain = torch.zeros(2, 40)
    gain[:, 10:20] = 1.0
    out = processor(gain)
    assert out.min().item() >= db_to_gain(-25.0) - 1e-6
    assert out.max().item() <= 1.0 + 1e-6


def test_post_processor_reads_the_shipping_preset_from_config():
    cfg = load_config()
    grid = AecGrid.from_config(cfg)
    processor = GainPostProcessor.from_config(cfg, grid)
    assert processor.gain_floor_db == cfg.getfloat('inference', 'gain_floor_db')
    assert 'gain floor' in processor.describe()


def test_smoothing_in_seconds_refuses_to_guess_the_frame_rate():
    with pytest.raises(ValueError, match='SECONDS'):
        GainPostProcessor(attack_tau_sec=0.05)


# ============================================================
# Scenario weighting
# ============================================================

def test_idle_chunks_are_upweighted_and_the_rest_are_not():
    meta = [{'scenario': 'ref_dropout'}, {'scenario': 'near_only'},
            {'scenario': 'double_talk'}, {'scenario': 'far_only'}]
    weights = scenario_weights(meta, idle_weight=3.0, dt_weight=1.5,
                               device=torch.device('cpu'))
    assert weights.tolist() == [3.0, 3.0, 1.5, 1.0]
