"""Regression tests for DFN3 feature, FIR, loss, and checkpoint contracts."""

import configparser
import math
import json
import re
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = pathlib.Path(ROOT)
# Qualified imports keep DFN2 and DFN3 independent in a combined pytest run.
sys.path.insert(0, str(ROOT_PATH.parents[1]))


from AINR.DeepFilterNet3.model import (  # noqa: E402
    DeepFilterNet3,
    SqueezedGRU_S,
    _build_erb_fb,
    deep_filter_apply,
    erb_bandborder,
)
from AINR.DeepFilterNet3.train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    SAFE_ANGLE_CLAMP,
    SAFE_ANGLE_WORST_MAG,
    MultiResSpecLoss,
    _SafeAngle,
    erb_band_db,
    extract_dfn3_features,
    make_checkpoint_contract,
    read_feature_config,
    read_loss_config,
    read_model_config,
    require_checkpoint_contract,
    scan_non_finite,
    validate_signal_config,
)


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    return cfg


def build_shipped_model(cfg, **overrides):
    """Construct the model exactly as config.ini specifies, with overrides.

    Goes through the trainer's own ``read_model_config`` rather than re-reading
    the keys, so tests cannot build ``df_hidden``, ``mask_pf`` or ``pf_beta``
    from constructor defaults while the trainer builds them from config; it
    also inherits the unknown-key rejection.
    """
    return DeepFilterNet3(**{**read_model_config(cfg), **overrides})


def test_feature_chunk_equivalence_and_independent_states():
    torch.manual_seed(4)
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    real = torch.randn(2, 513, 29)
    imag = torch.randn(2, 513, 29)
    spec = torch.complex(real, imag)

    _, erb_full, cplx_full, state_full = extract_dfn3_features(
        spec, erb_fb, 96, feature_cfg=feature_cfg
    )
    _, erb_a, cplx_a, state = extract_dfn3_features(
        spec[..., :11], erb_fb, 96, feature_cfg=feature_cfg
    )
    _, erb_b, cplx_b, state = extract_dfn3_features(
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
    """Tap ordering guard: with only the current-frame tap set, the filter is
    an identity.  Catches a reversed tap order or a stray conjugation, neither
    of which shows up in a parameter count.

    (deep_filter_apply lost its `alpha` argument when the composition moved to
    DFN3's parallel band split: the deep filter now owns the lowest df_bins
    outright instead of being alpha-blended against the masked spectrum.)
    """
    spec = torch.complex(
        torch.arange(1, 8, dtype=torch.float32).view(1, 1, 7),
        torch.zeros(1, 1, 7),
    )
    coefs = torch.zeros(1, 7, 1, 10)
    # Coefficient index four is the current frame for order=5/lookahead=0;
    # index 8 is its real part in the interleaved (order, re/im) layout.
    coefs[..., 8] = 1.0
    actual = deep_filter_apply(spec, coefs, df_bins=1, df_order=5, df_lookahead=0)
    torch.testing.assert_close(actual, spec)


def test_deep_filter_leaves_bins_above_df_bins_untouched():
    """The DF stage must only write the lowest df_bins; everything above is
    supplied by the ERB mask in the parallel band split."""
    torch.manual_seed(0)
    n_bins, T, df_bins, df_order = 8, 5, 3, 5
    spec = torch.complex(torch.randn(1, n_bins, T), torch.randn(1, n_bins, T))
    coefs = torch.randn(1, T, df_bins, df_order * 2)
    out = deep_filter_apply(spec.clone(), coefs, df_bins, df_order, 0)
    torch.testing.assert_close(out[:, df_bins:], spec[:, df_bins:])
    assert not torch.allclose(out[:, :df_bins], spec[:, :df_bins])


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


def test_erb_fb_matches_banderb_notebook_construction():
    # erb_bandborder(): every band >= 2 bins (the original notebook's
    # "every-OTHER-band-pair >= 2" rule does not guarantee it), endpoints
    # pinned, right band count.
    for n_bands, sr, n_fft in [(32, 48000, 1024), (22, 16000, 512),
                               (10, 8000, 256)]:
        border = erb_bandborder(n_bands, sr, n_fft)
        widths = np.diff(border)
        assert (widths >= 2).all(), (n_bands, sr, n_fft, widths.tolist())
        assert border[0] == 0
        assert border[-1] == n_fft // 2 + 1
        assert len(border) == n_bands

    # _build_erb_fb(): erb_inv is an exact partition of unity (no row
    # normalisation needed); erb_fb's two edge columns are exactly 2x erb_inv's
    # (mode=0 vs mode=1), interior columns match exactly.
    fb, inv = _build_erb_fb(1024, 48000, 32)
    assert fb.shape == (32, 513)
    assert inv.shape == (32, 513)
    colsum = inv.sum(dim=0)
    torch.testing.assert_close(colsum, torch.ones_like(colsum), rtol=0, atol=1e-6)
    torch.testing.assert_close(fb[0], 2.0 * inv[0])
    torch.testing.assert_close(fb[-1], 2.0 * inv[-1])
    torch.testing.assert_close(fb[15], inv[15])


def test_checkpoint_contract_rejects_legacy_and_accepts_current():
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    loss_cfg = read_loss_config(cfg)
    contract = make_checkpoint_contract(
        sr=48000, n_fft=1024, win_len=1024, hop_len=512, n_erb=32, df_bins=96,
        df_order=5, mask_lookahead=1, df_lookahead=1, mask_pf=False,
        pf_beta=0.02, feature_cfg=feature_cfg, loss_cfg=loss_cfg,
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
        test_erb_fb_matches_banderb_notebook_construction,
        test_checkpoint_contract_rejects_legacy_and_accepts_current,
    ]
    for test in tests:
        test()
        print(f'PASS: {test.__name__}')


# ============================================================
# Gradient hazard around _SafeAngle's clamp
#
# ⚠ The pure-noise assertion above uses randn, whose STFT magnitudes are O(1) --
# nowhere near where the complex-angle gradient is actually amplified.  It proves
# the loss is finite on a zero target and nothing about the hazard.  The tests
# below target the region deliberately.
# ============================================================

def test_safe_angle_gain_peaks_at_sqrt_clamp_not_at_zero():
    """The gain is |x|/max(|x|^2, clamp): peak 1/sqrt(clamp), and ZERO at x=0.

    Guards a specific wrong belief -- that the clamp value 1e-10 is itself the
    amplification, i.e. 1e10, and that an exactly-zero input is the worst case.
    Both are false, and acting on them sends the search for a gradient spike to
    the wrong signals.
    """
    assert _SafeAngle_gain(0.0) == 0.0

    peak_mag = SAFE_ANGLE_CLAMP ** 0.5
    assert peak_mag == SAFE_ANGLE_WORST_MAG
    peak_gain = _SafeAngle_gain(peak_mag)
    assert peak_gain == pytest.approx(1.0 / SAFE_ANGLE_WORST_MAG, rel=1e-6)
    assert peak_gain == pytest.approx(1e5, rel=1e-6)
    # Emphatically not 1/clamp.
    assert peak_gain < 1e-3 * (1.0 / SAFE_ANGLE_CLAMP)

    # Unimodal, peaking at sqrt(clamp): both sides fall away.
    for mag in (peak_mag / 10, peak_mag / 3, peak_mag * 3, peak_mag * 10):
        assert _SafeAngle_gain(mag) < peak_gain

    # Analytic agreement across four decades either side of the peak.
    for exponent in range(-12, -1):
        mag = 10.0 ** exponent
        expected = mag / max(mag * mag, SAFE_ANGLE_CLAMP)
        assert _SafeAngle_gain(mag) == pytest.approx(expected, rel=1e-5)


def _SafeAngle_gain(mag):
    """|d angle/dx| measured through autograd at complex magnitude ``mag``."""
    x = torch.complex(
        torch.tensor([mag * 0.6]), torch.tensor([mag * 0.8])
    ).requires_grad_(True)
    _SafeAngle.apply(x).backward(torch.ones(1))
    return float(x.grad.abs())


def test_zero_target_compresses_to_a_floor_not_to_complex_zero():
    """gamma=0.3 turns a silent target into magnitude 2.51e-4, not 0.

    So the prediction is pulled toward that floor rather than toward zero, and on
    the way it traverses the peak-gain band.  This is why 'zero target' is not
    itself the hazard.
    """
    gamma = read_loss_config(load_config())['gamma']
    floor = torch.zeros(1).clamp_min(1e-12).pow(gamma)
    # rel=1e-6, not tighter: the loss path is float32 and pow() there differs
    # from the float64 literal in the 9th significant digit.  The claim under
    # test is the magnitude, not bit-exactness.
    assert float(floor) == pytest.approx(1e-12 ** gamma, rel=1e-6)
    assert float(floor) == pytest.approx(2.5119e-4, rel=1e-3)
    assert float(floor) > SAFE_ANGLE_WORST_MAG


@pytest.mark.parametrize('scale', [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
def test_loss_gradient_finite_across_the_hazard_sweep(scale):
    """Finite loss AND finite gradient for predictions swept through the band.

    Both target polarities: a silent target (the full-suppression case) and a
    non-zero one.  ``scale=0`` is the exactly-zero prediction, which must give a
    finite -- specifically zero-angle-gradient -- result rather than NaN.
    """
    torch.manual_seed(11)
    loss_cfg = read_loss_config(load_config())
    loss_fn = MultiResSpecLoss(**loss_cfg)

    base = torch.randn(1, 4096)
    for target in (torch.zeros(1, 4096), torch.randn(1, 4096) * 1e-3):
        enhanced = (base * scale).clone().requires_grad_(True)
        loss = loss_fn(enhanced, target)
        assert torch.isfinite(loss), f'loss non-finite at scale={scale}'
        loss.backward()
        assert torch.isfinite(enhanced.grad).all(), (
            f'gradient non-finite at scale={scale}'
        )


def test_model_to_loss_backward_is_finite_on_a_silent_target():
    """Full path -- model -> ISTFT -> MRSL -> backward -- not just the loss.

    A loss-only test cannot catch a non-finite value introduced by the feature
    normalisers or the deep-filter operator, and the EMA states live outside
    autograd, so they are exactly the kind of thing a loss test misses.
    """
    torch.manual_seed(13)
    cfg = load_config()
    n_fft, hop, win = 1024, 512, 1024
    model = build_shipped_model(cfg)
    window = torch.sqrt(torch.hann_window(win))
    feature_cfg = read_feature_config(cfg, 48000, hop)

    # Near-silent input: the regime that produces near-silent predictions.
    noisy = torch.randn(1, 4096) * 1e-6
    clean = torch.zeros(1, 4096)

    spec = torch.stft(noisy, n_fft, hop, win, window=window,
                      return_complex=True, normalized=True)
    spec, feat_erb, feat_spec, _ = extract_dfn3_features(
        spec, model.erb_fb, 96, feature_cfg=feature_cfg
    )
    enhanced_spec, _ = model(spec, feat_erb, feat_spec)
    enhanced = torch.istft(enhanced_spec, n_fft, hop, win, window=window,
                           length=noisy.shape[-1], normalized=True)
    assert torch.isfinite(enhanced).all()

    loss = MultiResSpecLoss(**read_loss_config(cfg))(enhanced, clean)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, 'no parameter received a gradient'
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_non_finite_gradient_is_refused_before_it_touches_the_model():
    """error_if_nonfinite must raise BEFORE clipping scales anything.

    Without the flag, total_norm=inf gives clip_coef=1/(inf+1e-6)=0 and inf*0
    becomes NaN, which the next step writes into the weights and into Adam's
    moments -- unrecoverable.  This asserts the ordering, not just that an
    exception appears: after the raise, gradients and weights are untouched.
    """
    model = torch.nn.Linear(4, 1)
    before = [p.detach().clone() for p in model.parameters()]
    (model(torch.ones(1, 4)).sum() * float('inf')).backward()

    with pytest.raises(RuntimeError):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )

    assert any(not torch.isfinite(p.grad).all() for p in model.parameters()), (
        'gradients were modified; the raise must happen before scaling'
    )
    for param, original in zip(model.parameters(), before):
        assert torch.equal(param.detach(), original)
    assert scan_non_finite(model) == []


def test_scan_non_finite_excludes_buffers_by_default():
    """⚠ Buffers must be OFF by default, and that default is load-bearing.

    BatchNorm writes running_mean/running_var during forward() in train mode,
    so a forward-side fault poisons every BN buffer before the loss exists.
    Including them would make the halt report claim "an earlier step already
    wrote NaN into the weights" at global step 0 and put non-finite buffers
    into the checkpoint it tells the operator to resume from, which the resume
    guard rejects.
    """
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    assert scan_non_finite(model) == []

    with torch.no_grad():
        model[1].running_mean[0] = float('nan')
    assert scan_non_finite(model) == [], (
        'a poisoned BN buffer must NOT register as a poisoned parameter'
    )
    with_buffers = scan_non_finite(model, include_buffers=True)
    assert [row[0] for row in with_buffers] == ['1.running_mean']

    with torch.no_grad():
        model[0].weight[0, 0] = float('inf')
    params_only = scan_non_finite(model)
    assert [row[0] for row in params_only] == ['0.weight']
    assert params_only[0][1:3] == (0, 1)


def test_lookahead_relation_is_enforced_in_the_model_constructor():
    """df_lookahead > mask_lookahead must be rejected at the single source.

    train.py and inference.py deliberately do not check it, so if the
    constructor stops enforcing it nothing does.
    """
    cfg = load_config()
    with pytest.raises(ValueError, match='mask_lookahead'):
        build_shipped_model(cfg, mask_lookahead=0, df_lookahead=1)
    # The shipped pairing must remain constructible.
    build_shipped_model(cfg)


def test_enc_concat_true_still_emits_a_single_width_bus():
    """⚠ Dormant path, so nothing else covers it.

    Upstream keeps emb_in_dim and emb_out_dim separate: concatenating the two
    encoder branches doubles what the embedding GRU CONSUMES while what it
    EMITS stays one bus wide for the decoders. One shared width attribute would
    make enc_concat=True emit a double-width bus the ERB decoder cannot accept,
    and enc_concat is an exposed config knob.
    """
    cfg = load_config()
    model = build_shipped_model(cfg, enc_concat=True)
    # One immutable bus width; concatenation widens only the GRU's input.
    assert model.encoder.emb_dim == 64 * (32 // 4)

    n_fft, hop, win = 1024, 512, 1024
    window = torch.sqrt(torch.hann_window(win))
    noisy = torch.randn(1, 8192) * 0.05
    spec = torch.stft(noisy, n_fft, hop, win, window=window,
                      return_complex=True, normalized=True)
    spec, feat_erb, feat_spec, _ = extract_dfn3_features(
        spec, model.erb_fb, 96, feature_cfg=read_feature_config(cfg, 48000, hop)
    )
    enhanced_spec, erb_mask = model(spec, feat_erb, feat_spec)
    assert enhanced_spec.shape == spec.shape
    assert torch.isfinite(enhanced_spec.real).all()
    assert torch.isfinite(erb_mask).all()


def test_n_erb_divisibility_follows_upstreams_stricter_rule():
    """Upstream asserts nb_erb % 8; the two stride-2 stages only need % 4.

    ⚠ Calls the real validator: a local `32 % 8 == 0` assertion would never
    reach train.py and would stay green if the guard reverted to % 4.
    """
    shipped = dict(n_fft=1024, win_len=1024, hop_len=512, n_erb=32, df_bins=96,
                   df_order=5, mask_lookahead=1, df_lookahead=1)
    validate_signal_config(**shipped)          # must not raise

    # 20 and 36 are the interesting cases: legal under %4, illegal under %8.
    for bad in (20, 36):
        assert bad % 4 == 0 and bad % 8 != 0
        with pytest.raises(ValueError, match='divisible by eight'):
            validate_signal_config(**{**shipped, 'n_erb': bad})
    with pytest.raises(ValueError, match='divisible by eight'):
        validate_signal_config(**{**shipped, 'n_erb': 0})


def test_analysis_scale_fold_is_bit_exact():
    """Scaling the POWER by s**2 must equal scaling the SPECTRUM by s.

    ``erb_band_db`` folds libDF's wnorm past the band sum so it touches
    (B, T, n_erb) floats instead of (B, T, n_bins) complex.  That is only free
    because ``analysis_scale`` is an exact power of two at every supported grid --
    if a future grid makes it inexact, this test is what says so.
    """
    torch.manual_seed(29)
    cfg = load_config()
    scale = read_feature_config(cfg, 48000, 512)['analysis_scale']
    assert scale == 2.0 ** round(math.log2(scale)), (
        f'analysis_scale {scale!r} is not a power of two; the fold in '
        f'erb_band_db is no longer lossless'
    )
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    for amplitude in (1e-6, 0.05, 3.0):
        spec = torch.complex(torch.randn(1, 20, 513), torch.randn(1, 20, 513))
        spec = spec * amplitude
        folded = erb_band_db(spec, erb_fb, scale)
        scaled_first = erb_band_db(spec * scale, erb_fb, 1.0)
        assert torch.equal(folded, scaled_first), (
            f'fold is not bit-exact at amplitude {amplitude}'
        )


def test_calibrator_calls_the_trainers_band_db_step():
    """⚠ The calibrator fits init values FOR the trainer, so it must measure
    what the trainer produces.  It calls the same function, so a transcribed
    copy cannot drift on the log floor or the power expression; this asserts
    the call rather than the text.
    """
    source = (ROOT_PATH / 'calibrate_norm_init.py').read_text()
    assert 'erb_band_db(' in source, (
        'calibrate_norm_init.py must call the trainer\'s erb_band_db, not '
        're-implement the band-dB step'
    )
    # And the numbers must actually agree on the same input.
    torch.manual_seed(31)
    cfg = load_config()
    scale = read_feature_config(cfg, 48000, 512)['analysis_scale']
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    spec = torch.complex(torch.randn(1, 513, 20), torch.randn(1, 513, 20)) * 0.05
    calibrator_view = erb_band_db(spec.transpose(1, 2), erb_fb, scale)
    trainer_view = erb_band_db(spec.permute(0, 2, 1), erb_fb, scale)
    assert torch.equal(calibrator_view, trainer_view)


def test_erb_power_uses_the_direct_form_through_the_real_feature_path():
    """Upstream computes re*re + im*im (lib.rs:291), not abs()**2.

    ⚠ Asserts through extract_dfn3_features, not against a local
    re-implementation: a test that built both expressions itself would pass
    whichever form the trainer used, while reverting the trainer to
    abs().pow(2) fails this one.
    """
    torch.manual_seed(5)
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    spec = torch.complex(torch.randn(1, 513, 24), torch.randn(1, 513, 24))

    _, feat_erb, _, _ = extract_dfn3_features(
        spec, erb_fb, 96, feature_cfg=feature_cfg
    )

    def reference(power_expr):
        # libDF's wnorm, exactly as extract_dfn3_features applies it.  Omitting it
        # leaves a constant 10*log10(1/1024)/40 = 0.753 offset in the feature.
        spec_btc = spec.permute(0, 2, 1) * feature_cfg['analysis_scale']
        erb_db = (power_expr(spec_btc).matmul(erb_fb.T) + 1e-10).log10() * 10
        mu = torch.linspace(
            feature_cfg['erb_init_lo_db'], feature_cfg['erb_init_hi_db'], 32
        ).view(1, 1, 32).clone()
        out = []
        for t in range(erb_db.shape[1]):
            mu = feature_cfg['erb_alpha'] * mu + (
                1 - feature_cfg['erb_alpha']
            ) * erb_db[:, t:t + 1, :]
            out.append((erb_db[:, t:t + 1, :] - mu) / feature_cfg['erb_scale_db'])
        return torch.cat(out, dim=1)

    direct = reference(lambda x: x.real.square() + x.imag.square())
    round_trip = reference(lambda x: x.abs().pow(2))

    flat = feat_erb.squeeze(1)
    assert torch.equal(flat, direct), (
        'extract_dfn3_features must use the direct re*re + im*im form'
    )
    assert not torch.equal(direct, round_trip), (
        'inputs chosen so the sqrt round trip actually loses bits; if these are '
        'equal the test cannot detect a regression'
    )


def test_squeezed_gru_skip_closes_over_the_raw_input():
    """⚠ Dormant at every call site, so only a direct construction reaches it.

    Upstream's SqueezedGRU_S adds `gru_skip(input)` -- the raw argument --
    after linear_out (df/modules.py:732-738), not `gru_skip(x)` with `x`
    already rebound by linear_in. All three in-repo sites pass
    gru_skip_op=None, so no end-to-end test can see the difference.
    """
    torch.manual_seed(17)
    block = SqueezedGRU_S(
        8, 4, output_size=8, num_layers=1, linear_groups=1,
        gru_skip_op=lambda: torch.nn.Identity(),
    )
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        out, _ = block(x)
        squeezed = block.linear_in(x)
        gru_out, _ = block.gru(squeezed)
        expanded = block.linear_out(gru_out)

    torch.testing.assert_close(out, expanded + x, rtol=1e-6, atol=1e-6)
    # At unequal widths the wrong form cannot even run -- 8 vs 4 -- so a crash is
    # the detection there.  The case that would be SILENT is hidden == input, so
    # cover it too: both forms are legal and only the values differ.
    torch.manual_seed(19)
    same = SqueezedGRU_S(8, 8, output_size=8, num_layers=1, linear_groups=1,
                         gru_skip_op=lambda: torch.nn.Identity())
    z = torch.randn(1, 3, 8)
    with torch.no_grad():
        same_out, _ = same(z)
        z_squeezed = same.linear_in(z)
        z_gru, _ = same.gru(z_squeezed)
        z_expanded = same.linear_out(z_gru)
    torch.testing.assert_close(same_out, z_expanded + z, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(same_out, z_expanded + z_squeezed,
                              rtol=1e-6, atol=1e-6), (
        'raw-input and squeezed-input skips must be distinguishable here, or '
        'this test cannot detect the mis-wiring in the silent case'
    )


def test_emb_num_layers_one_fails_loudly_instead_of_building_two():
    """Upstream's bare ``emb_num_layers - 1`` is kept (no max(1, ...) clamp).

    ⚠ A clamp would silently build 1 + 1 = 2 GRU layers for emb_num_layers=1
    while the config says 1, breaking the total-depth invariant config.ini
    documents; nn.GRU rejects num_layers=0.
    """
    cfg = load_config()
    with pytest.raises(ValueError, match='num_layers'):
        build_shipped_model(cfg, emb_num_layers=1)

    # The shipped depth still resolves to 1 encoder + 2 ERB + 2 DF = 5.
    model = build_shipped_model(cfg)
    gru_layers = [n for n, _ in model.named_parameters()
                  if '.gru.weight_ih' in n]
    assert len(gru_layers) == 5


# Restored deployment boundary: DFN3 has two heads and caller-owned state.
from AINR.DeepFilterNet3.export_onnx import (  # noqa: E402
    HEAD_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    RETIRED_LAYOUT_VERSIONS,
    STATE_LAYOUT_VERSION,
    StatelessDFN3Heads,
    build_metadata,
    export_graph,
    feature_windows,
)


def _small_dfn3():
    return DeepFilterNet3(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()


def _checkpoint_contract(cfg=None):
    cfg = cfg or load_config()
    model = read_model_config(cfg)
    return make_checkpoint_contract(
        sr=cfg.getint('signal', 'sr'), n_fft=cfg.getint('signal', 'n_fft'),
        win_len=cfg.getint('signal', 'win_len'),
        hop_len=cfg.getint('signal', 'hop_len'),
        **{key: model[key] for key in (
            'n_erb', 'df_bins', 'df_order', 'mask_lookahead',
            'df_lookahead', 'mask_pf', 'pf_beta')},
        feature_cfg=read_feature_config(
            cfg, cfg.getint('signal', 'sr'), cfg.getint('signal', 'hop_len')),
        loss_cfg=read_loss_config(cfg),
    )


def test_restored_checkpoint_loads_historical_v5_but_rejects_dfn2(tmp_path):
    """The two graphs have identical parameter sets; only version/contract can
    prevent accidentally loading alpha/cascade weights into the split graph."""
    from types import SimpleNamespace
    from AINR.DeepFilterNet3.inference import load_model

    cfg = load_config()
    model = build_shipped_model(cfg).eval()
    contract = _checkpoint_contract(cfg)
    checkpoint = {
        'model_version': MODEL_VERSION, 'feature_version': FEATURE_VERSION,
        'loss_version': LOSS_VERSION, 'contract': contract,
        'state_dict': model.state_dict(),
    }
    path = tmp_path / 'dfn3.pth'
    # A v5 checkpoint does not store the composition field; no arithmetic
    # differs.
    historical = {**checkpoint, 'contract': {
        key: value for key, value in contract.items() if key != 'composition'}}
    torch.save(historical, path)
    restored, params = load_model(SimpleNamespace(
        config=str(ROOT_PATH / 'config.ini'), model=str(path)))
    assert params['DF_BINS'] == 96
    assert restored.df_lookahead == 1
    torch.testing.assert_close(restored.erb_fb, model.erb_fb, rtol=0, atol=0)
    require_checkpoint_contract(historical, contract, for_training=True)
    with pytest.raises(ValueError, match='model_version'):
        require_checkpoint_contract({
            **checkpoint,
            'model_version': 'dfn2_fmajor_pathway_cascade_alpha_no_lsnr_v6',
        }, contract, for_training=False)
    with pytest.raises(ValueError, match='composition'):
        require_checkpoint_contract({
            **checkpoint,
            'contract': {**contract, 'composition': 'dfn2_cascade_alpha'},
        }, contract)


def test_dfn3_compose_uses_raw_low_bins_and_masked_high_bins():
    """A zero ERB gain must not zero DF's current raw-spectrum tap."""
    torch.manual_seed(75)
    model = _small_dfn3()
    spectrum = torch.complex(
        torch.randn(1, model.n_bins, 7), torch.randn(1, model.n_bins, 7))
    coefficients = torch.zeros(1, 7, model.df_bins, model.df_order * 2)
    current_tap = model.df_order - model.df_lookahead - 1
    coefficients[..., 2 * current_tap] = 1
    result = model.compose(
        spectrum, torch.zeros(1, 1, 7, model.n_erb), coefficients)
    torch.testing.assert_close(result[:, :model.df_bins],
                               spectrum[:, :model.df_bins], rtol=0, atol=0)
    assert torch.count_nonzero(result[:, model.df_bins:]) == 0


def test_stateless_dfn3_two_heads_replay_offline_and_pathway_history():
    torch.manual_seed(76)
    model = _small_dfn3()
    wrapper = StatelessDFN3Heads(model).eval()
    assert HEAD_OUTPUT_NAMES == ('erb_mask', 'df_coefs')
    assert wrapper.output_names == (
        'erb_mask', 'df_coefs', 'h_gru_out', 'df_convp_history_out')
    erb = torch.randn(1, 1, 11, model.n_erb)
    spec = torch.randn(1, 2, 11, model.df_bins)
    with torch.no_grad():
        reference = model.heads(erb, spec)
        states = wrapper.initial_inputs()[2:]
        masks, coefs, histories = [], [], []
        for erb_window, spec_window in zip(feature_windows(erb),
                                            feature_windows(spec)):
            outputs = wrapper(erb_window, spec_window, *states)
            masks.append(outputs[0])
            coefs.append(outputs[1])
            states = outputs[2:]
            histories.append(states[-1])
    torch.testing.assert_close(torch.cat(masks, 2), reference[0],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.cat(coefs, 1), reference[1],
                               rtol=1e-5, atol=1e-6)
    for before, after in zip(histories, histories[1:]):
        assert torch.equal(before[:, :, 1:], after[:, :, :-1])
    assert histories[-1].abs().max() > 0


def test_fixed_batch_dfn3_replays_independent_streaming_lanes():
    torch.manual_seed(77)
    model = _small_dfn3()
    single = StatelessDFN3Heads(model).eval()
    batch = StatelessDFN3Heads(model, batch_size=3).eval()
    inputs = tuple(torch.randn_like(value) for value in batch.initial_inputs())
    with torch.no_grad():
        actual = batch(*inputs)
        lanes = [single(*(value[lane:lane + 1] for value in inputs))
                 for lane in range(3)]
    for index, output in enumerate(actual):
        torch.testing.assert_close(
            output, torch.cat([lane[index] for lane in lanes], dim=0),
            rtol=1e-5, atol=1e-6)


def test_dfn3_export_state_layout_and_c_dimensions_match(tmp_path):
    header = (ROOT_PATH / 'dfn3_model_io.h').read_text()
    process = (ROOT_PATH / 'dfn3_process.h').read_text()

    def macro(text, name):
        match = re.search(r'^#define\s+' + name + r'\s+(\d+)u?\b',
                          text, flags=re.MULTILINE)
        assert match, name
        return int(match.group(1))

    model = build_shipped_model(load_config()).eval()
    wrapper = StatelessDFN3Heads(model).eval()
    inputs = wrapper.initial_inputs()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'hash only')
    metadata = build_metadata(str(checkpoint),
        {'SR': 48000, 'N_FFT': 1024, 'WIN_LEN': 1024, 'HOP_LEN': 512},
        inputs, outputs)
    assert STATE_LAYOUT_VERSION == macro(header, 'DFN3_MODEL_IO_LAYOUT_VERSION') == 9
    assert STATE_LAYOUT_VERSION not in RETIRED_LAYOUT_VERSIONS
    assert 8 in RETIRED_LAYOUT_VERSIONS
    assert metadata['composition'] == 'dfn3_parallel_split'
    assert tuple(metadata['output_schema']) == OUTPUT_NAMES
    assert metadata['state_handoff'] == {
        'h_gru': 'h_gru_out', 'df_convp_history': 'df_convp_history_out'}
    frames = macro(header, 'DFN3_MODEL_INPUT_FRAMES')
    bins = macro(process, 'DFN3_DF_BINS')
    assert [tuple(value.shape) for value in inputs] == [
        (1, 1, frames, macro(process, 'DFN3_N_ERB')),
        (1, 2, frames, bins),
        (1, macro(header, 'DFN3_MODEL_GRU_ROWS'), 1,
         macro(header, 'DFN3_MODEL_GRU_HIDDEN')),
        (1, macro(header, 'DFN3_MODEL_ENCODER_CHANNELS'),
         macro(header, 'DFN3_MODEL_DF_PATHWAY_WINDOW'), bins),
    ]


def test_runtime_erb_bins_are_exact_dfn3_buffers(tmp_path):
    from AINR.DeepFilterNet3.export_erb_matrix import write_runtime_bins
    write_runtime_bins(str(ROOT_PATH / 'config.ini'), str(tmp_path))
    model = build_shipped_model(load_config()).eval()
    forward = np.fromfile(tmp_path / 'erb_fwd.bin', '<f4').reshape(513, 32)
    inverse = np.fromfile(tmp_path / 'erb_inv.bin', '<f4').reshape(32, 513)
    assert np.array_equal(forward, model.erb_fb.numpy().T)
    assert np.array_equal(inverse, model.erb_inv.numpy())


@pytest.mark.parametrize('batch_size', [1, 3])
def test_dfn3_onnx_executes_multiple_hops_with_explicit_state(tmp_path, batch_size):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    from onnx_streaming_contract import validate_inplace_state_from_metadata

    torch.manual_seed(78)
    wrapper = StatelessDFN3Heads(_small_dfn3(), batch_size=batch_size).eval()
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'hash only')
    path = tmp_path / 'dfn3.onnx'
    metadata = export_graph(wrapper,
        {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        str(checkpoint), str(path), verify=True)
    graph = onnx.load(str(path))
    validate_inplace_state_from_metadata(graph)
    assert {value.name for value in graph.graph.output} == set(OUTPUT_NAMES)
    assert 'df_alpha' not in metadata['output_schema']
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    initial = wrapper.initial_inputs()
    torch_states = initial[2:]
    onnx_states = tuple(value.numpy() for value in torch_states)
    with torch.no_grad():
        for _ in range(4):
            content = tuple(torch.randn_like(value) for value in initial[:2])
            expected = wrapper(*content, *torch_states)
            actual = session.run(None, dict(zip(INPUT_NAMES,
                [value.numpy() for value in content] + list(onnx_states))))
            for got, want in zip(actual, expected):
                np.testing.assert_allclose(got, want.numpy(), rtol=1e-4, atol=2e-5)
            torch_states = expected[2:]
            onnx_states = tuple(actual[2:])
    persisted = json.loads(path.with_suffix('.json').read_text())
    assert persisted['model_family'] == 'DeepFilterNet3'
    assert persisted['state_layout_version'] == 9


@pytest.mark.parametrize('override', [
    {'conv_kernel': (2, 3)}, {'convt_kernel': (2, 3)},
])
def test_stateless_export_rejects_unrepresented_temporal_history(override):
    model = DeepFilterNet3(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4, **override,
    ).eval()
    # These configs are valid offline models, but their intermediate kernels
    # need more state than layout 9 exports. Reject before writing any graph.
    with pytest.raises(ValueError, match='intermediate temporal kernels'):
        StatelessDFN3Heads(model)


@pytest.fixture
def erb_export_checkpoint(tmp_path):
    cfg = load_config()
    for key, value in {
        'enc_channels': '8', 'emb_size': '32', 'df_hidden': '32',
        'lin_groups': '4', 'enc_lin_groups': '4',
    }.items():
        cfg.set('model', key, value)
    config = tmp_path / 'config.ini'
    with config.open('w') as stream:
        cfg.write(stream)
    model = build_shipped_model(cfg).eval()
    # A deliberate checkpoint-buffer difference catches reconstruction from
    # config (or ignoring --model) while retaining inverse partition of unity.
    model.erb_fb.mul_(1.125)
    payload = {
        'model_version': MODEL_VERSION, 'feature_version': FEATURE_VERSION,
        'loss_version': LOSS_VERSION, 'contract': _checkpoint_contract(cfg),
        'state_dict': model.state_dict(),
    }
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save(payload, checkpoint)
    return config, checkpoint, model, payload


@pytest.mark.parametrize('runtime_bins', [False, True])
@pytest.mark.parametrize('bad_version', [
    ('model_version', 'dfn2_fmajor_pathway_cascade_alpha_no_lsnr_v6'),
    ('model_version', 'dfn3_fmajor_flatten_pathway_add_no_lsnr_v4'),
    ('feature_version', 'dfn2_libdf_wnorm_calibrated_v3'),
])
def test_erb_export_cli_rejects_incompatible_checkpoints_before_writing(
        tmp_path, monkeypatch, erb_export_checkpoint, runtime_bins, bad_version):
    from AINR.DeepFilterNet3.export_erb_matrix import main
    config, _, _, payload = erb_export_checkpoint
    key, value = bad_version
    incompatible = tmp_path / 'incompatible.pth'
    torch.save({**payload, key: value}, incompatible)
    output = tmp_path / 'rejected_export'
    mode = ['--runtime-bins', str(output)] if runtime_bins else [
        '--output-dir', str(output), '--format', 'npy']
    monkeypatch.setattr(sys, 'argv', [
        'export_erb_matrix.py', '--config', str(config),
        '--model', str(incompatible), *mode])
    with pytest.raises(ValueError, match=key):
        main()
    assert not output.exists(), 'an incompatible checkpoint must not be relabeled DFN3'


@pytest.mark.parametrize('runtime_bins', [False, True])
def test_erb_export_cli_uses_loaded_checkpoint_buffers_and_hash(
        tmp_path, monkeypatch, erb_export_checkpoint, runtime_bins):
    import hashlib
    from AINR.DeepFilterNet3.export_erb_matrix import main
    config, checkpoint, model, _ = erb_export_checkpoint
    output = tmp_path / 'export'
    mode = ['--runtime-bins', str(output)] if runtime_bins else [
        '--output-dir', str(output), '--format', 'npy']
    monkeypatch.setattr(sys, 'argv', [
        'export_erb_matrix.py', '--config', str(config),
        '--model', str(checkpoint), *mode])
    main()
    if runtime_bins:
        forward = np.fromfile(output / 'erb_fwd.bin', '<f4').reshape(513, 32)
        inverse = np.fromfile(output / 'erb_inv.bin', '<f4').reshape(32, 513)
        manifest = json.loads((output / 'erb_matrices.json').read_text())
    else:
        forward = np.load(output / 'erb_forward.npy')
        inverse = np.load(output / 'erb_inverse.npy').T
        manifest = json.loads((output / 'erb_matrix.json').read_text())
    assert np.array_equal(forward, model.erb_fb.numpy().T)
    assert np.array_equal(inverse, model.erb_inv.numpy())
    assert manifest['checkpoint_sha256'] == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
