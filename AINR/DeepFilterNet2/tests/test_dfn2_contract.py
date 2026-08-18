"""Regression tests for DFN2 feature, FIR, loss, and checkpoint contracts."""

import configparser
import json
import math
import os
import pathlib
import re
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = pathlib.Path(ROOT)
sys.path.insert(0, ROOT)

# Each of the three model projects has its own top-level ``train.py`` (and
# ``inference.py``/``model.py``). Under a single pytest session the first one
# imported wins ``sys.modules``, so a sibling project's tests would silently
# exercise the wrong code.  Dropping the cached entries forces the re-import
# to resolve against the ROOT just inserted above.
for _stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx'):
    sys.modules.pop(_stale, None)


from model import (  # noqa: E402
    DeepFilterNet2,
    SqueezedGRU_S,
    _build_erb_fb,
    deep_filter_apply,
    erb_bandborder,
)
from train import (  # noqa: E402
    FEATURE_VERSION,
    LOSS_VERSION,
    MODEL_VERSION,
    SAFE_ANGLE_CLAMP,
    SAFE_ANGLE_WORST_MAG,
    MultiResSpecLoss,
    _SafeAngle,
    erb_band_db,
    extract_dfn2_features,
    make_checkpoint_contract,
    read_feature_config,
    read_loss_config,
    read_model_config,
    require_checkpoint_contract,
    scan_non_finite,
    validate_signal_config,
)
from export_onnx import (  # noqa: E402
    INPUT_FRAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    STATE_LAYOUT_VERSION,
    StatelessDFN2Heads,
    build_metadata,
    feature_windows,
    initial_inputs as streaming_export_inputs,
)


def c_macro(header_text, name):
    """Read a plain integer ``#define`` out of a shipped C header."""
    match = re.search(
        r'^#define\s+%s\s+(\d+)u?\s*(?:/\*.*)?$' % re.escape(name),
        header_text,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return int(match.group(1))


def model_io_header():
    return (ROOT_PATH / 'dfn2_model_io.h').read_text(encoding='utf-8')


def load_config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    return cfg


def build_shipped_model(cfg, **overrides):
    """Construct the model exactly as config.ini specifies, with overrides.

    Goes through the trainer's own ``read_model_config`` rather than re-reading
    the keys: a local copy is a third restatement of the constructor's defaults
    (train.py and inference.py were the first two, which is why that function
    exists), and it silently omitted ``df_hidden``, ``mask_pf`` and ``pf_beta`` --
    so tests built those three from constructor defaults while the trainer built
    them from config.  It also inherits the unknown-key rejection.
    """
    return DeepFilterNet2(**{**read_model_config(cfg), **overrides})


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
    """Tap ordering guard: with only the current-frame tap set, the filter is
    an identity.  Catches a reversed tap order or a stray conjugation, neither
    of which shows up in a parameter count.

    Alpha belongs to model.compose(), not this parameter-free FIR primitive.
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
    """The DF stage only writes low bins; DFN2 compose preserves masked highs."""
    torch.manual_seed(0)
    n_bins, T, df_bins, df_order = 8, 5, 3, 5
    spec = torch.complex(torch.randn(1, n_bins, T), torch.randn(1, n_bins, T))
    coefs = torch.randn(1, T, df_bins, df_order * 2)
    out = deep_filter_apply(spec.clone(), coefs, df_bins, df_order, 0)
    torch.testing.assert_close(out[:, df_bins:], spec[:, df_bins:])
    assert not torch.allclose(out[:, :df_bins], spec[:, :df_bins])


def test_compose_is_full_band_mask_then_low_band_alpha_blend():
    """Guard the parameter-count-invisible DFN2 composition contract.

    With zero DF coefficients, alpha=0.25 leaves 75% of the already-masked low
    spectrum.  High bins must retain the full-band ERB-mask result.  A DFN3
    parallel band split fails both assertions.
    """
    torch.manual_seed(3)
    model = build_shipped_model(load_config()).eval()
    time = 4
    spec = torch.complex(
        torch.randn(1, model.n_bins, time),
        torch.randn(1, model.n_bins, time),
    )
    erb_mask = torch.full((1, 1, time, model.n_erb), 0.5)
    coefs = torch.zeros(1, time, model.df_bins, model.df_order * 2)
    alpha = torch.full((1, time, 1), 0.25)

    actual = model.compose(spec, erb_mask, coefs, alpha)
    torch.testing.assert_close(
        actual[:, :model.df_bins],
        spec[:, :model.df_bins] * 0.5 * 0.75,
    )
    torch.testing.assert_close(
        actual[:, model.df_bins:],
        spec[:, model.df_bins:] * 0.5,
    )


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
    # erb_bandborder(): every band >= 2 bins (the v3 fix -- the original
    # notebook's "every-OTHER-band-pair >= 2" rule did not actually
    # guarantee this), endpoints pinned, right band count.
    for n_bands, sr, n_fft in [(32, 48000, 1024), (22, 16000, 512),
                               (10, 8000, 256)]:
        border = erb_bandborder(n_bands, sr, n_fft)
        widths = np.diff(border)
        assert (widths >= 2).all(), (n_bands, sr, n_fft, widths.tolist())
        assert border[0] == 0
        assert border[-1] == n_fft // 2 + 1
        assert len(border) == n_bands

    # _build_erb_fb(): erb_inv is an exact partition of unity (no row
    # normalisation needed, unlike the old construction); erb_fb's two edge
    # columns are exactly 2x erb_inv's (mode=0 vs mode=1), interior columns
    # match exactly.
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


def test_stateless_export_replays_heads_with_explicit_state():
    """Three feature frames plus returned state must equal offline heads."""
    torch.manual_seed(73)
    model = DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()
    wrapper = StatelessDFN2Heads(model).eval()
    frames = 9
    erb = torch.randn(1, 1, frames, model.n_erb)
    spec = torch.randn(1, 2, frames, model.df_bins)

    with torch.no_grad():
        reference = model.heads(erb, spec)
        state = streaming_export_inputs(model)[2:]
        streamed = [[], [], []]
        for erb_window, spec_window in zip(
            feature_windows(erb), feature_windows(spec)
        ):
            output = wrapper(erb_window, spec_window, *state)
            for index in range(3):
                streamed[index].append(output[index])
            state = output[3:]

    assembled = (
        torch.cat(streamed[0], dim=2),
        torch.cat(streamed[1], dim=1),
        torch.cat(streamed[2], dim=1),
    )
    for actual, expected in zip(assembled, reference):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stateless_export_df_pathway_cache_is_live():
    """Dropping the kernel-5 c0 history must change the coefficient stream."""
    torch.manual_seed(74)
    model = DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()
    wrapper = StatelessDFN2Heads(model).eval()
    inputs = streaming_export_inputs(model)
    state = inputs[2:]
    with torch.no_grad():
        first = wrapper(*inputs)
        proper = wrapper(inputs[0], inputs[1], *first[3:])[1]
        broken_state = first[3:-1] + (torch.zeros_like(first[-1]),)
        broken = wrapper(inputs[0], inputs[1], *broken_state)[1]
    assert (proper - broken).abs().max().item() > 1e-6


def test_c_model_io_layout_constants_match_the_shipped_export_shapes():
    """dfn2_model_io.h's struct dimensions ARE the graph's state shapes.

    The header hard-codes every tensor extent the accelerator hands back.
    Nothing else compares them to the model the exporter actually builds from
    the shipped config, so an emb_hidden_dim or enc_ch edit would silently
    leave the C side allocating the wrong buffers -- the graph would still
    export and only a board would find out.
    """
    header = model_io_header()
    process = (ROOT_PATH / 'dfn2_process.h').read_text(encoding='utf-8')
    frames = c_macro(header, 'DFN2_MODEL_INPUT_FRAMES')
    hidden = c_macro(header, 'DFN2_MODEL_GRU_HIDDEN')
    n_erb = c_macro(process, 'DFN2_N_ERB')
    df_bins = c_macro(process, 'DFN2_DF_BINS')
    assert frames == INPUT_FRAMES, (
        'the C window depth and the exporter window depth are one contract'
    )

    model = build_shipped_model(load_config()).eval()
    expected = (
        (1, 1, frames, n_erb),
        (1, 2, frames, df_bins),
        (c_macro(header, 'DFN2_MODEL_ENCODER_GRU_LAYERS'), 1, hidden),
        (c_macro(header, 'DFN2_MODEL_ERB_GRU_LAYERS'), 1, hidden),
        (c_macro(header, 'DFN2_MODEL_DF_GRU_LAYERS'), 1, hidden),
        (1, c_macro(header, 'DFN2_MODEL_ENCODER_CHANNELS'),
         c_macro(header, 'DFN2_MODEL_DF_PATHWAY_HISTORY'), df_bins),
    )
    actual = tuple(
        tuple(int(size) for size in value.shape)
        for value in streaming_export_inputs(model)
    )
    assert actual == expected


def _small_dfn2():
    return DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()


def test_calibration_frame_shapes_equal_the_exported_graph_inputs(tmp_path):
    """One recorded calibration frame must BE one graph invocation.

    ``capture_calibration_inputs`` keeps the graph's batch dimension and
    ``np.stack`` adds the calibration-frame axis on top of it.  Dropping the
    batch axis would leave every ``.bin`` one rank short of the input the
    accelerator binds it to, and nothing downstream re-derives the shape --
    the manifest is what a quantizer reads.  So the per-frame shapes are
    compared against the ONNX graph this exporter really produces rather than
    against a literal that could be edited to match a regression.
    """
    onnx = pytest.importorskip('onnx')
    from calibration_io import (
        capture_calibration_inputs,
        write_calibration_artifact,
    )

    torch.manual_seed(75)
    model = _small_dfn2()
    wrapper = StatelessDFN2Heads(model).eval()
    inputs = streaming_export_inputs(model)

    graph_path = os.fspath(tmp_path / 'dfn2_stream.onnx')
    torch.onnx.export(
        wrapper, inputs, graph_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        opset_version=17, do_constant_folding=True,
    )
    graph_shapes = {
        value.name: [int(dim.dim_value)
                     for dim in value.type.tensor_type.shape.dim]
        for value in onnx.load(graph_path).graph.input
    }
    assert set(graph_shapes) == set(INPUT_NAMES)

    # Two invocations, recorded exactly the way calibration_main records them.
    captured = {}
    state = tuple(inputs[2:])
    with torch.no_grad():
        for _ in range(2):
            step = (inputs[0], inputs[1]) + state
            capture_calibration_inputs(captured, INPUT_NAMES, step)
            state = tuple(wrapper(*step)[3:])
    arrays = {name: np.stack(values).astype(np.float32, copy=False)
              for name, values in captured.items()}
    artifact = tmp_path / 'calib'
    write_calibration_artifact(artifact, arrays, {'frames': 2}, 'bin')
    manifest = json.loads((artifact / 'manifest.json').read_text())

    for name in INPUT_NAMES:
        assert manifest['binary_tensors'][name]['frame_shape'] == (
            graph_shapes[name]
        ), name
        # And the bytes on disk really hold one whole graph input.
        blob = np.fromfile(artifact / name / ('%s_0000.bin' % name), '<f4')
        assert blob.size == int(np.prod(graph_shapes[name])), name


def test_state_layout_version_is_pinned_to_the_c_header(tmp_path):
    """The exported metadata must carry the header's layout version.

    A board reads ``state_layout_version`` out of the graph to decide whether
    its ``DFN2ModelIOState`` still matches. Asserting the Python constant
    alone would not catch the metadata key being dropped, so this goes through
    the same builder ``main`` uses.
    """
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    model = DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()
    inputs = streaming_export_inputs(model)
    with torch.no_grad():
        outputs = StatelessDFN2Heads(model).eval()(*inputs)
    metadata = build_metadata(
        str(checkpoint),
        {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        inputs,
        outputs,
    )
    assert metadata['state_layout_version'] == c_macro(
        model_io_header(), 'DFN2_MODEL_IO_LAYOUT_VERSION'
    )
    assert STATE_LAYOUT_VERSION == metadata['state_layout_version']


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
    spec, feat_erb, feat_spec, _ = extract_dfn2_features(
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

    BatchNorm writes running_mean/running_var during forward() in train mode, so a
    forward-side fault poisons every BN buffer before the loss exists.  Including
    them made the halt report claim "an earlier step already wrote NaN into the
    weights" at global step 0 -- the opposite of the truth -- and put non-finite
    buffers into the checkpoint it told the operator to resume from, which the
    resume guard then rejected.
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

    Guards against the invariant drifting back out into per-entry-point copies:
    train.py and inference.py deliberately no longer check it, so if the
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
    encoder branches doubles what the embedding GRU CONSUMES while what it EMITS
    stays one bus wide, because the decoders downstream are built for one bus.
    The port previously used a single attribute, so enc_concat=True emitted a
    double-width bus the ERB decoder could not accept.  Since enc_concat is an
    exposed config knob, that failure was reachable.
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
    spec, feat_erb, feat_spec, _ = extract_dfn2_features(
        spec, model.erb_fb, 96, feature_cfg=read_feature_config(cfg, 48000, hop)
    )
    enhanced_spec, erb_mask = model(spec, feat_erb, feat_spec)
    assert enhanced_spec.shape == spec.shape
    assert torch.isfinite(enhanced_spec.real).all()
    assert torch.isfinite(erb_mask).all()


def test_n_erb_divisibility_follows_upstreams_stricter_rule():
    """Upstream asserts nb_erb % 8; the two stride-2 stages only need % 4.

    ⚠ Calls the real validator.  The previous version asserted `32 % 8 == 0`
    locally and never reached train.py -- mutation-proven: reverting the guard to
    % 4 left it green.
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
    """⚠ The calibrator fits init values FOR the trainer, so it must measure what
    the trainer produces.  It used to transcribe the four lines and drifted twice
    -- on the log floor and on the power expression.  It now calls the same
    function, so this asserts the call rather than the text.
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

    ⚠ Asserts through extract_dfn2_features, not against a local re-implementation.
    The previous version of this test built both expressions itself and compared
    them, so it passed whichever form the trainer actually used -- mutation-proven:
    reverting the trainer to abs().pow(2) left it green.
    """
    torch.manual_seed(5)
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    erb_fb, _ = _build_erb_fb(1024, 48000, 32)
    spec = torch.complex(torch.randn(1, 513, 24), torch.randn(1, 513, 24))

    _, feat_erb, _, _ = extract_dfn2_features(
        spec, erb_fb, 96, feature_cfg=feature_cfg
    )

    def reference(power_expr):
        # libDF's wnorm, exactly as extract_dfn2_features applies it.  Omitting it
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
        'extract_dfn2_features must use the direct re*re + im*im form'
    )
    assert not torch.equal(direct, round_trip), (
        'inputs chosen so the sqrt round trip actually loses bits; if these are '
        'equal the test cannot detect a regression'
    )


def test_squeezed_gru_skip_closes_over_the_raw_input():
    """⚠ Dormant at every call site, so only a direct construction reaches it.

    Upstream's SqueezedGRU_S adds `gru_skip(input)` -- the raw argument -- after
    linear_out (df/modules.py:732-738).  The port had `gru_skip(x)` where `x` had
    already been rebound by linear_in, i.e. the squeezed activation.  All three
    in-repo sites pass gru_skip_op=None, so no end-to-end test can see the
    difference; mutation-proven that the whole suite stayed green with the wrong
    wiring.
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
    """The max(1, emb_num_layers - 1) clamp was removed to match upstream.

    ⚠ With the clamp, emb_num_layers=1 silently built 1 + 1 = 2 GRU layers while
    the config said 1, breaking the total-depth invariant config.ini documents.
    Upstream is a bare subtraction and nn.GRU rejects num_layers=0.
    """
    cfg = load_config()
    with pytest.raises(ValueError, match='num_layers'):
        build_shipped_model(cfg, emb_num_layers=1)

    # The shipped depth still resolves to 1 encoder + 2 ERB + 2 DF = 5.
    model = build_shipped_model(cfg)
    gru_layers = [n for n, _ in model.named_parameters()
                  if '.gru.weight_ih' in n]
    assert len(gru_layers) == 5


def test_runtime_erb_bins_match_both_models(tmp_path):
    """The exported .bin matrices equal DFN2's AND DFN3's frozen buffers.

    The C host consumes caller-loaded pointers in these exact layouts; both
    DF models share one filterbank, so one pair of files must serve both.
    """
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from export_erb_matrix import write_runtime_bins

    out = write_runtime_bins(os.path.join(ROOT, 'config.ini'),
                             os.fspath(tmp_path))
    fwd = np.fromfile(os.path.join(out, 'erb_fwd.bin'),
                      '<f4').reshape(513, 32)
    inv = np.fromfile(os.path.join(out, 'erb_inv.bin'),
                      '<f4').reshape(32, 513)
    model = DeepFilterNet2(**read_model_config(load_config()))
    ref_fwd = model.erb_fb.detach().numpy().astype(np.float32)
    if ref_fwd.shape[0] < ref_fwd.shape[1]:
        ref_fwd = ref_fwd.T
    ref_inv = model.erb_inv.detach().numpy().astype(np.float32)
    if ref_inv.shape[0] > ref_inv.shape[1]:
        ref_inv = ref_inv.T
    assert np.array_equal(fwd, ref_fwd)
    assert np.array_equal(inv, ref_inv)

    import importlib
    for stale in ('train', 'inference', 'model', 'export_onnx'):
        sys.modules.pop(stale, None)
    dfn3_root = os.path.join(os.path.dirname(ROOT), 'DeepFilterNet3')
    _sys.path.insert(0, dfn3_root)
    try:
        dfn3_model_mod = importlib.import_module('model')
        dfn3_train = importlib.import_module('train')
        cfg3 = configparser.ConfigParser()
        assert cfg3.read(os.path.join(dfn3_root, 'config.ini'))
        dfn3 = dfn3_model_mod.DeepFilterNet3(
            **dfn3_train.read_model_config(cfg3))
        fwd3 = dfn3.erb_fb.detach().numpy().astype(np.float32)
        if fwd3.shape[0] < fwd3.shape[1]:
            fwd3 = fwd3.T
        assert np.array_equal(fwd, fwd3)
    finally:
        _sys.path.remove(dfn3_root)
        for stale in ('train', 'inference', 'model', 'export_onnx'):
            sys.modules.pop(stale, None)
