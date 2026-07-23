#!/usr/bin/env python3
"""Regression tests for the DeepFilterNet 3 MultiResSpecLoss training objective."""

import configparser
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    LOSS_VERSION,
    MultiResSpecLoss,
    RNNoiseModel,
    apply_erb_gains_batch,
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    istft,
    read_feature_config,
    read_loss_config,
    require_checkpoint_loss_config,
    stft,
)


def reference_df3_mrsl(enhanced, clean, fft_sizes, gamma, factor, factor_complex):
    """Direct transcription of Rikorose/DeepFilterNet df/loss.py."""
    loss = torch.zeros((), device=enhanced.device, dtype=enhanced.dtype)
    for n_fft in fft_sizes:
        window = torch.hann_window(n_fft, device=enhanced.device,
                                   dtype=enhanced.dtype)
        y = torch.stft(enhanced, n_fft, n_fft // 4, window=window,
                       normalized=True, return_complex=True)
        s = torch.stft(clean, n_fft, n_fft // 4, window=window,
                       normalized=True, return_complex=True)
        y_abs = y.abs().clamp_min(1e-12).pow(gamma)
        s_abs = s.abs().clamp_min(1e-12).pow(gamma)
        loss = loss + F.mse_loss(y_abs, s_abs) * factor
        y = y_abs * torch.exp(1j * torch.angle(y))
        s = s_abs * torch.exp(1j * torch.angle(s))
        loss = loss + F.mse_loss(
            torch.view_as_real(y), torch.view_as_real(s)) * factor_complex
    return loss


def test_config_is_df3_production_mrsl():
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    loss_cfg = read_loss_config(cfg)
    assert LOSS_VERSION == 'df3_multi_res_spec_only_gamma_0.3_v1'
    assert loss_cfg == {
        'fft_sizes': (256, 512, 1024, 2048),
        'gamma': 0.3,
        'factor': 500.0,
        'factor_complex': 500.0,
    }
    assert not cfg.has_section('perceptual_loss')


def test_matches_upstream_formula():
    torch.manual_seed(7)
    enhanced = torch.randn(2, 4096)
    clean = torch.randn(2, 4096)
    kwargs = dict(fft_sizes=(256, 512, 1024), gamma=0.3,
                  factor=500.0, factor_complex=500.0)
    actual = MultiResSpecLoss(**kwargs)(enhanced, clean)
    expected = reference_df3_mrsl(enhanced, clean, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_checkpoint_gate_rejects_legacy_objective():
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    loss_cfg = read_loss_config(cfg)
    saved = {
        'loss_version': LOSS_VERSION,
        'config': {
            'loss_fft_sizes': '256,512,1024,2048',
            'loss_gamma': 0.3,
            'loss_factor': 500.0,
            'loss_factor_complex': 500.0,
        },
    }
    require_checkpoint_loss_config(saved, loss_cfg, context='test checkpoint')
    try:
        require_checkpoint_loss_config({}, loss_cfg, context='legacy checkpoint')
    except ValueError as exc:
        assert 'legacy loss contract' in str(exc)
    else:
        raise AssertionError('legacy checkpoint objective was accepted')


def test_pure_noise_target_has_finite_nonzero_gradient():
    torch.manual_seed(11)
    enhanced = torch.randn(2, 4096, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    loss = MultiResSpecLoss()(enhanced, clean)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()
    assert enhanced.grad.abs().max().item() > 0


def test_silent_pair_is_zero_and_backward_safe():
    enhanced = torch.zeros(1, 4096, requires_grad=True)
    clean = torch.zeros_like(enhanced)
    loss = MultiResSpecLoss()(enhanced, clean)
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
    loss.backward()
    assert enhanced.grad is not None
    assert torch.isfinite(enhanced.grad).all()


def test_pure_noise_end_to_end_model_backward():
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    n_bands = cfg.getint('signal', 'n_bands')
    lookahead = cfg.getint('signal', 'lookahead_frames')
    feature_cfg = read_feature_config(cfg, sr, hop_len, n_fft, win_len)
    borders = erb_bandborder(n_bands, sr, n_fft)
    erb_fwd = torch.from_numpy(compute_erb_matrix(borders, n_fft, mode=0))
    erb_inv = torch.from_numpy(compute_erb_matrix(borders, n_fft, mode=1))
    model = RNNoiseModel(
        n_bands=n_bands,
        spec_bins=feature_cfg['spec_bins'],
        cond_size=cfg.getint('model', 'cond_size'),
        gru_size=cfg.getint('model', 'gru_size'),
        spec_conv_channels=cfg.getint('model', 'spec_conv_channels'),
        spec_embed_size=cfg.getint('model', 'spec_embed_size'),
        dropout=0.0,
    )
    torch.manual_seed(13)
    noisy = torch.randn(1, 4096) * 0.05
    clean = torch.zeros_like(noisy)
    window = torch.sqrt(torch.hann_window(win_len))
    noisy_spec = stft(noisy, n_fft, hop_len, win_len, window)
    erb, spec, _, _ = extract_model_features(noisy_spec, erb_fwd, feature_cfg)
    erb = F.pad(erb, (0, 0, 2 - lookahead, lookahead))
    spec = F.pad(spec, (0, 0, 0, 0, 2 - lookahead, lookahead))
    gains, _ = model(erb, spec)
    enhanced_spec = apply_erb_gains_batch(noisy_spec, gains, erb_inv, lookahead)
    enhanced = istft(enhanced_spec, n_fft, hop_len, win_len, window, noisy.size(-1))
    loss = MultiResSpecLoss(**read_loss_config(cfg))(enhanced, clean)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max().item() > 0 for g in grads)


if __name__ == '__main__':
    tests = [
        test_config_is_df3_production_mrsl,
        test_matches_upstream_formula,
        test_checkpoint_gate_rejects_legacy_objective,
        test_pure_noise_target_has_finite_nonzero_gradient,
        test_silent_pair_is_zero_and_backward_safe,
        test_pure_noise_end_to_end_model_backward,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
