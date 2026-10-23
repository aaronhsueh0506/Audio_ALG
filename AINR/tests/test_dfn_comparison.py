"""Keep DFN2/DFN3 comparisons on the same data, features and learned backbone.

Composition and checkpoint identity intentionally differ. A source restoration
must not silently give the two experiments different preprocessing or recipes.
"""

import configparser
from importlib import import_module
from pathlib import Path

import pytest
import torch


AINR = Path(__file__).resolve().parents[1]


def _project(number):
    package = f'AINR.DeepFilterNet{number}'
    cfg = configparser.ConfigParser()
    assert cfg.read(AINR / f'DeepFilterNet{number}' / 'config.ini')
    return cfg, import_module(package + '.train'), import_module(package + '.model')


def _contract(cfg, trainer):
    model = trainer.read_model_config(cfg)
    return trainer.make_checkpoint_contract(
        **{key: cfg.getint('signal', key) for key in
           ('sr', 'n_fft', 'win_len', 'hop_len')},
        **{key: model[key] for key in
           ('n_erb', 'df_bins', 'df_order', 'mask_lookahead',
            'df_lookahead', 'mask_pf', 'pf_beta')},
        feature_cfg=trainer.read_feature_config(
            cfg, cfg.getint('signal', 'sr'), cfg.getint('signal', 'hop_len')),
        loss_cfg=trainer.read_loss_config(cfg),
    )


def test_dfn_default_recipes_match_for_controlled_comparison():
    cfg2, trainer2, _ = _project(2)
    cfg3, trainer3, _ = _project(3)
    for section in ('signal', 'model', 'feature', 'training', 'multi_res_spec_loss'):
        assert dict(cfg2[section]) == dict(cfg3[section]), section
    assert trainer2.FEATURE_VERSION == trainer3.FEATURE_VERSION
    assert trainer2.LOSS_VERSION == trainer3.LOSS_VERSION
    assert trainer2.MODEL_VERSION != trainer3.MODEL_VERSION


def test_dfn_features_match_bit_for_bit_on_the_same_spectrum():
    cfg2, trainer2, module2 = _project(2)
    cfg3, trainer3, module3 = _project(3)
    generator = torch.Generator().manual_seed(182)
    spec = torch.complex(
        torch.randn(2, 513, 17, generator=generator),
        torch.randn(2, 513, 17, generator=generator)) * 0.1
    fb2, _ = module2._build_erb_fb(1024, 48000, 32)
    fb3, _ = module3._build_erb_fb(1024, 48000, 32)
    feat2 = trainer2.extract_dfn2_features(
        spec, fb2, 96, feature_cfg=trainer2.read_feature_config(cfg2, 48000, 512))
    feat3 = trainer3.extract_dfn3_features(
        spec, fb3, 96, feature_cfg=trainer3.read_feature_config(cfg3, 48000, 512))
    torch.testing.assert_close(feat2, feat3, rtol=0, atol=0)


def test_dfn_checkpoint_gates_reject_the_other_composition_in_both_directions():
    projects = [_project(2), _project(3)]
    for cfg, trainer, _ in projects:
        other = projects[1][1] if trainer is projects[0][1] else projects[0][1]
        checkpoint = {
            'model_version': other.MODEL_VERSION,
            'feature_version': other.FEATURE_VERSION,
            'loss_version': other.LOSS_VERSION,
            'contract': _contract(cfg, trainer),
        }
        with pytest.raises(ValueError, match='model_version'):
            trainer.require_checkpoint_contract(checkpoint, _contract(cfg, trainer))


def test_dfn_backbones_produce_identical_heads_before_composition():
    """Controlled weights isolate the output graph from encoder differences.

    This explicit test-only copy is not a supported cross-model checkpoint load.
    A DFN3 checkpoint retains the unused alpha parameters, so even a strict
    state_dict load alone cannot distinguish these differently trained graphs.
    """
    cfg2, trainer2, module2 = _project(2)
    cfg3, trainer3, module3 = _project(3)
    compact = dict(enc_ch=16, emb_size=32, df_hidden=32,
                   lin_groups=4, enc_lin_groups=4)
    torch.manual_seed(18)
    model2 = module2.DeepFilterNet2(**{**trainer2.read_model_config(cfg2), **compact}).eval()
    model3 = module3.DeepFilterNet3(**{**trainer3.read_model_config(cfg3), **compact}).eval()
    model3.load_state_dict(model2.state_dict(), strict=True)
    erb = torch.randn(1, 1, 11, 32)
    cplx = torch.randn(1, 2, 11, 96)
    with torch.no_grad():
        mask2, coefs2, _ = model2.heads(erb, cplx)
        mask3, coefs3 = model3.heads(erb, cplx)
    torch.testing.assert_close(mask2, mask3, rtol=0, atol=0)
    torch.testing.assert_close(coefs2, coefs3, rtol=0, atol=0)
