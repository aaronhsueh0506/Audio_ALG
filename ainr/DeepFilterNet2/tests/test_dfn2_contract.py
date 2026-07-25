"""Regression tests for DFN2 feature, FIR, loss, and checkpoint contracts."""

import configparser
import os
import sys

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import _build_erb_fb, deep_filter_apply  # noqa: E402
from train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    MultiResSpecLoss,
    extract_dfn2_features,
    make_checkpoint_contract,
    read_feature_config,
    read_loss_config,
    require_checkpoint_contract,
)


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    return cfg


def test_feature_chunk_equivalence_and_independent_states():
    torch.manual_seed(4)
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    real = torch.randn(2, 513, 29)
    imag = torch.randn(2, 513, 29)
    spec = torch.complex(real, imag)

    _, erb_full, cplx_full, state_full = extract_dfn2_features(
        spec, erb_fb, 96, feature_cfg=feature_cfg
    )
    _, erb_a, cplx_a, state = extract_dfn2_features(
        spec[..., :11], erb_fb, 96, feature_cfg=feature_cfg
    )
    _, erb_b, cplx_b, state = extract_dfn2_features(
        spec[..., 11:],
        erb_fb,
        96,
        feature_cfg=feature_cfg,
        ema_state=state,
    )

    torch.testing.assert_close(
        torch.cat([erb_a, erb_b], dim=2), erb_full, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        torch.cat([cplx_a, cplx_b], dim=2), cplx_full, rtol=1e-5, atol=1e-6
    )
    assert set(state) == {'erb', 'spec'}
    torch.testing.assert_close(state['erb'], state_full['erb'])
    torch.testing.assert_close(state['spec'], state_full['spec'])


def test_causal_order_five_uses_current_tap_without_extra_delay():
    spec = torch.complex(
        torch.arange(1, 8, dtype=torch.float32).view(1, 1, 7),
        torch.zeros(1, 1, 7),
    )
    coefs = torch.zeros(1, 7, 1, 10)
    # Coefficient index four is the current frame for order=5/lookahead=0.
    coefs[..., 8] = 1.0
    alpha = torch.ones(1, 7, 1)
    actual = deep_filter_apply(
        spec, coefs, alpha, df_bins=1, df_order=5, df_lookahead=0
    )
    torch.testing.assert_close(actual, spec)


def reference_mrsl(enhanced, clean, loss_cfg):
    total = torch.zeros((), dtype=enhanced.dtype)
    for n_fft in loss_cfg['fft_sizes']:
        window = torch.hann_window(n_fft)
        y = torch.stft(
            enhanced, n_fft, n_fft // 4, window=window,
            normalized=True, return_complex=True,
        )
        s = torch.stft(
            clean, n_fft, n_fft // 4, window=window,
            normalized=True, return_complex=True,
        )
        y_abs = y.abs().clamp_min(1e-12).pow(loss_cfg['gamma'])
        s_abs = s.abs().clamp_min(1e-12).pow(loss_cfg['gamma'])
        total += F.mse_loss(y_abs, s_abs) * loss_cfg['factor']
        y = y_abs * torch.exp(1j * torch.angle(y))
        s = s_abs * torch.exp(1j * torch.angle(s))
        total += F.mse_loss(
            torch.view_as_real(y), torch.view_as_real(s)
        ) * loss_cfg['factor_complex']
    return total


def test_phase_aware_loss_matches_reference_and_accepts_pure_noise():
    torch.manual_seed(7)
    cfg = load_config()
    loss_cfg = read_loss_config(cfg)
    enhanced = torch.randn(2, 4096, requires_grad=True)
    clean = torch.randn(2, 4096)
    loss = MultiResSpecLoss(**loss_cfg)(enhanced, clean)
    expected = reference_mrsl(enhanced, clean, loss_cfg)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
    loss.backward()
    assert torch.isfinite(enhanced.grad).all()

    pure_noise_out = torch.randn(1, 4096, requires_grad=True)
    zero_target = torch.zeros_like(pure_noise_out)
    pure_noise_loss = MultiResSpecLoss(**loss_cfg)(
        pure_noise_out, zero_target
    )
    assert torch.isfinite(pure_noise_loss)
    pure_noise_loss.backward()
    assert torch.isfinite(pure_noise_out.grad).all()


def test_checkpoint_contract_rejects_legacy_and_accepts_current():
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    loss_cfg = read_loss_config(cfg)
    contract = make_checkpoint_contract(
        48000, 1024, 1024, 512, 32, 96, 5, 1, 0,
        feature_cfg, loss_cfg,
    )
    current = {
        'model_version': MODEL_VERSION,
        'feature_version': FEATURE_VERSION,
        'loss_version': LOSS_VERSION,
        'contract': contract,
    }
    require_checkpoint_contract(current, contract)

    try:
        require_checkpoint_contract({}, contract)
    except ValueError:
        pass
    else:
        raise AssertionError('legacy checkpoint was accepted')


if __name__ == '__main__':
    tests = [
        test_feature_chunk_equivalence_and_independent_states,
        test_causal_order_five_uses_current_tap_without_extra_delay,
        test_phase_aware_loss_matches_reference_and_accepts_pure_noise,
        test_checkpoint_contract_rejects_legacy_and_accepts_current,
    ]
    for test in tests:
        test()
        print(f'PASS: {test.__name__}')
