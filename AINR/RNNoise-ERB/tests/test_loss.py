#!/usr/bin/env python3
"""Regression tests for the DeepFilterNet 3 MultiResSpecLoss training objective."""

import configparser
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Each of the three model projects has its own top-level ``train.py`` (and
# ``denoise.py``/``model.py``).  Under a single pytest session the first one
# imported wins ``sys.modules``, so a sibling project's tests would silently
# exercise the wrong code.  Dropping the cached entries forces the re-import
# to resolve against the ROOT just inserted above.
for _stale in ('train', 'denoise', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)


from train import (  # noqa: E402
    ErbIrmLoss,
    LOSS_VERSION,
    MultiResSpecLoss,
    mrsl_is_enabled,
    RNNoiseModel,
    apply_erb_gains_batch,
    compute_erb_matrix,
    erb_bandborder,
    extract_model_features,
    istft,
    read_feature_config,
    read_irm_loss_config,
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


def test_config_is_irm_only():
    """The objective is the ERB-IRM term alone; MRSL is configured off.

    Measured on a 3 s sample, d(MRSL)/d(gains) had ~1106x the RMS of
    d(IRM)/d(gains), so running both meant IRM contributed nothing to the
    optimisation.  Turning MRSL off is what makes "train with IRM loss" true.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    loss_cfg = read_loss_config(cfg)
    assert LOSS_VERSION == 'erb_irm_only_v4'
    assert loss_cfg['factor'] == 0.0 and loss_cfg['factor_complex'] == 0.0
    assert not mrsl_is_enabled(loss_cfg)
    # fft_sizes/gamma stay configured so re-enabling is a one-line change.
    assert loss_cfg['fft_sizes'] == (128, 256, 512, 1024)
    assert loss_cfg['gamma'] == 0.3
    assert cfg.getfloat('erb_irm_loss', 'factor') > 0
    assert not cfg.has_option('erb_irm_loss', 'activity_weight')
    assert not cfg.has_section('perceptual_loss')


def test_fft_sizes_are_ratios_of_the_model_fft():
    """The MRSL resolutions must be {n_fft/4, n_fft/2, n_fft, n_fft*2}.

    Upstream's literal 256,512,1024,2048 are those ratios for ITS 960-point
    model FFT; copied verbatim onto a 512-point model they land an octave high
    and leave no resolution below 10 ms.  Pinning the ratio rather than the
    integers keeps them correct if n_fft ever changes.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    n_fft = cfg.getint('signal', 'n_fft')
    assert read_loss_config(cfg)['fft_sizes'] == (
        n_fft // 4, n_fft // 2, n_fft, n_fft * 2)


def test_erb_irm_loss_is_configured_and_well_behaved():
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    assert cfg.has_section('erb_irm_loss')
    irm = ErbIrmLoss(**read_irm_loss_config(cfg))

    torch.manual_seed(0)
    B, T, n_bins, n_bands = 2, 20, 257, 22
    erb_fwd = torch.rand(n_bins, n_bands)
    clean = torch.complex(torch.randn(B, n_bins, T), torch.randn(B, n_bins, T))
    noisy = clean + 0.5 * torch.complex(
        torch.randn(B, n_bins, T), torch.randn(B, n_bins, T))

    e_c = irm._band_energy(clean, erb_fwd)
    e_n = irm._band_energy(noisy, erb_fwd)
    ideal = torch.sqrt(torch.clamp(e_c / (e_n + 1e-10), 0.0, 1.0))

    # The ideal ratio mask is the global minimum.
    assert abs(irm(ideal, noisy, clean, erb_fwd).item()) < 1e-9
    assert irm(torch.ones(B, T, n_bands), noisy, clean, erb_fwd) > 0
    assert irm(torch.full((B, T, n_bands), 1e-6), noisy, clean, erb_fwd) > 0

    # Undefined-band mask: a silent mixture carries no ratio information, so it
    # must contribute exactly nothing rather than a noise gradient.
    silence = torch.zeros(B, n_bins, T, dtype=torch.complex64)
    assert irm(torch.rand(B, T, n_bands), silence, silence, erb_fwd).item() == 0.0

    g = torch.rand(B, T, n_bands, requires_grad=True)
    irm(g, noisy, clean, erb_fwd).backward()
    assert torch.isfinite(g.grad).all()


def test_pure_erb_model_accepts_the_training_loop_call():
    """The training loop passes spec_features=None when use_complex_input=False.

    forward() used to dereference .ndim before checking the flag, so the
    default configuration crashed on the first batch.  Nothing covered this:
    the end-to-end test below builds its own features and always passes a real
    tensor.  The argument must stay accepted (ONNX/C signature stability) while
    None must also work.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    n_bands = cfg.getint('signal', 'n_bands')
    feature_cfg = read_feature_config(
        cfg, cfg.getint('signal', 'sr'), 256,
        cfg.getint('signal', 'n_fft'), cfg.getint('signal', 'n_fft'))
    spec_bins = feature_cfg['spec_bins']
    erb = torch.randn(2, 10, n_bands)
    spec = torch.randn(2, 10, 2, spec_bins)

    pure = RNNoiseModel(n_bands=n_bands, spec_bins=spec_bins,
                        use_complex_input=False)
    assert pure(erb, None)[0].shape == (2, 8, n_bands)
    assert pure(erb, spec)[0].shape == (2, 8, n_bands)

    dual = RNNoiseModel(n_bands=n_bands, spec_bins=spec_bins,
                        use_complex_input=True)
    assert dual(erb, spec)[0].shape == (2, 8, n_bands)
    try:
        dual(erb, None)
    except ValueError:
        pass
    else:
        raise AssertionError('complex model accepted a missing spectrum')


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
    irm_cfg = read_irm_loss_config(cfg)
    saved = {
        'loss_version': LOSS_VERSION,
        'config': {
            'loss_fft_sizes': '128,256,512,1024',
            'loss_gamma': 0.3,
            'loss_factor': 0.0,
            'loss_factor_complex': 0.0,
            'irm_factor': irm_cfg['factor'],
            'irm_gamma': irm_cfg['gamma'],
            'irm_energy_floor': irm_cfg['energy_floor'],
        },
    }
    require_checkpoint_loss_config(saved, loss_cfg, irm_cfg,
                                   context='test checkpoint')
    try:
        require_checkpoint_loss_config({}, loss_cfg, irm_cfg,
                                       context='legacy checkpoint')
    except ValueError as exc:
        assert 'legacy loss contract' in str(exc)
    else:
        raise AssertionError('legacy checkpoint objective was accepted')


def test_checkpoint_gate_covers_the_irm_settings():
    """The IRM term is the entire objective, so its knobs must be contracted.

    They were absent: a run trained at gamma=0.25 resumed without complaint
    against a config asking for 0.5.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    loss_cfg, irm_cfg = read_loss_config(cfg), read_irm_loss_config(cfg)
    base = {
        'loss_fft_sizes': ','.join(str(n) for n in loss_cfg['fft_sizes']),
        'loss_gamma': loss_cfg['gamma'],
        'loss_factor': loss_cfg['factor'],
        'loss_factor_complex': loss_cfg['factor_complex'],
        'irm_factor': irm_cfg['factor'],
        'irm_gamma': irm_cfg['gamma'],
        'irm_energy_floor': irm_cfg['energy_floor'],
    }
    require_checkpoint_loss_config(
        {'loss_version': LOSS_VERSION, 'config': base}, loss_cfg, irm_cfg)
    for key, bad in (('irm_gamma', 0.5), ('irm_factor', 100.0),
                     ('irm_energy_floor', 1e-6)):
        drifted = {'loss_version': LOSS_VERSION, 'config': {**base, key: bad}}
        try:
            require_checkpoint_loss_config(drifted, loss_cfg, irm_cfg)
        except ValueError:
            continue
        raise AssertionError(f'{key} drift was accepted')


def test_irm_loss_has_no_speech_activity_weighting():
    """A noise-only example must not be up-weighted as speech.

    The removed ``(1 + 5*vad)`` term computed vad from the CLEAN energy against
    its own peak; for an all-zero clean signal that was 10*log10(0/1e-12) = 0
    dB, i.e. above threshold, so every pure-noise frame scored as speech and
    got 6x weight -- the exact opposite of the intent.  The model has no VAD
    output head, so there is nothing for such a weight to correspond to.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    irm_cfg = read_irm_loss_config(cfg)
    assert 'activity_weight' not in irm_cfg
    assert 'vad_threshold_db' not in irm_cfg
    assert not cfg.has_option('erb_irm_loss', 'activity_weight')
    irm = ErbIrmLoss(**irm_cfg)
    assert not hasattr(irm, 'activity_weight')

    torch.manual_seed(0)
    n_bins, n_bands, t = 257, 22, 10
    erb_fwd = torch.rand(n_bins, n_bands)
    noisy = torch.complex(torch.randn(1, n_bins, t), torch.randn(1, n_bins, t))
    silent = torch.zeros(1, n_bins, t, dtype=torch.complex64)
    gains = torch.rand(1, t, n_bands, requires_grad=True)
    loss = irm(gains, noisy, silent, erb_fwd)      # noise-only: target is 0
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(gains.grad).all()


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
    """The CONFIGURED objective, through the model, on a pure-noise target.

    This mirrors train.py's batch_loss: pure-ERB features (spec_features=None),
    the configured ERB-IRM term, and MRSL added only when it is enabled.  It
    used to hardcode MRSL, so it kept passing while the configured objective
    was something else entirely.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    n_bands = cfg.getint('signal', 'n_bands')
    lookahead = cfg.getint('signal', 'lookahead_frames')
    min_bins = cfg.getint('signal', 'min_bins_per_band', fallback=2)
    use_complex = cfg.getboolean('model', 'use_complex_input', fallback=False)
    feature_cfg = read_feature_config(cfg, sr, hop_len, n_fft, win_len)
    loss_cfg = read_loss_config(cfg)
    borders = erb_bandborder(n_bands, sr, n_fft, min_bins)
    erb_fwd = torch.from_numpy(compute_erb_matrix(borders, n_fft, mode=0))
    erb_inv = torch.from_numpy(compute_erb_matrix(borders, n_fft, mode=1))
    model = RNNoiseModel(
        n_bands=n_bands,
        spec_bins=feature_cfg['spec_bins'],
        cond_size=cfg.getint('model', 'cond_size'),
        gru_size=cfg.getint('model', 'gru_size'),
        spec_conv_channels=cfg.getint('model', 'spec_conv_channels'),
        spec_embed_size=cfg.getint('model', 'spec_embed_size'),
        use_complex_input=use_complex,
        dropout=0.0,
    )
    torch.manual_seed(13)
    noisy = torch.randn(1, 4096) * 0.05
    clean = torch.zeros_like(noisy)
    window = torch.sqrt(torch.hann_window(win_len))
    noisy_spec = stft(noisy, n_fft, hop_len, win_len, window)
    clean_spec = stft(clean, n_fft, hop_len, win_len, window)
    erb, spec, _, _ = extract_model_features(
        noisy_spec, erb_fwd, feature_cfg, need_spec=use_complex)
    erb = F.pad(erb, (0, 0, 2 - lookahead, lookahead))
    if spec is not None:
        spec = F.pad(spec, (0, 0, 0, 0, 2 - lookahead, lookahead))
    gains, _ = model(erb, spec)

    loss = ErbIrmLoss(**read_irm_loss_config(cfg))(
        gains, noisy_spec, clean_spec, erb_fwd)
    if mrsl_is_enabled(loss_cfg):
        enhanced_spec = apply_erb_gains_batch(noisy_spec, gains, erb_inv, lookahead)
        enhanced = istft(enhanced_spec, n_fft, hop_len, win_len, window,
                         noisy.size(-1))
        loss = loss + MultiResSpecLoss(**loss_cfg)(enhanced, clean)

    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max().item() > 0 for g in grads)


if __name__ == '__main__':
    tests = [
        test_config_is_irm_only,
        test_fft_sizes_are_ratios_of_the_model_fft,
        test_erb_irm_loss_is_configured_and_well_behaved,
        test_pure_erb_model_accepts_the_training_loop_call,
        test_matches_upstream_formula,
        test_checkpoint_gate_rejects_legacy_objective,
        test_checkpoint_gate_covers_the_irm_settings,
        test_irm_loss_has_no_speech_activity_weighting,
        test_pure_noise_target_has_finite_nonzero_gradient,
        test_silent_pair_is_zero_and_backward_safe,
        test_pure_noise_end_to_end_model_backward,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
