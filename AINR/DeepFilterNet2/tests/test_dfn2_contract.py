"""Tests for DFN2 feature, FIR, loss, and checkpoint contracts."""

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
# AINR/tests holds dfn_pairs; pytest adds it through AINR/conftest.py, a direct run here.
_AINR_TESTS = os.path.join(os.path.dirname(ROOT), 'tests')
if _AINR_TESTS not in sys.path:
    sys.path.insert(0, _AINR_TESTS)

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
    validate_signal_config,
)
from export_onnx import (  # noqa: E402
    INPUT_FRAMES,
    GRU_STATE_NAME,
    GRU_STATE_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    PATHWAY_STATE_NAME,
    RETIRED_LAYOUT_VERSIONS,
    STATE_BINDING,
    STATE_HANDOFF,
    STATE_LAYOUT_VERSION,
    StatelessDFN2Heads,
    build_metadata,
    export_graph,
    feature_windows,
    gru_state_slice_report,
    initial_inputs as streaming_export_inputs,
    window_frames,
    window_alignment,
)

import dfn_pairs  # noqa: E402

# Every (mask_lookahead, df_lookahead) pair the model accepts; each one must
# export and stream.
LOOKAHEAD_PAIRS = list(dfn_pairs.accepted_pairs('DeepFilterNet2'))


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
    the keys, so tests cannot build ``df_hidden``, ``mask_pf`` or ``pf_beta``
    from constructor defaults while the trainer builds them from config; it
    also inherits the unknown-key rejection.
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
    # normalisation needed); erb_fb's two edge
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


@pytest.mark.parametrize('mask_lookahead,df_lookahead', LOOKAHEAD_PAIRS)
def test_stateless_export_replays_heads_with_explicit_state(
        mask_lookahead, df_lookahead):
    """Three feature frames plus returned state must equal offline heads."""
    torch.manual_seed(73)
    model = _small_dfn2(mask_lookahead, df_lookahead)
    wrapper = StatelessDFN2Heads(model).eval()
    frames = 9
    erb = torch.randn(1, 1, frames, model.n_erb)
    spec = torch.randn(1, 2, frames, model.df_bins)

    with torch.no_grad():
        reference = model.heads(erb, spec)
        state = streaming_export_inputs(model)[2:]
        streamed = [[], [], []]
        for erb_window, spec_window in zip(
            feature_windows(erb, mask_lookahead),
            feature_windows(spec, mask_lookahead),
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
    layers = (c_macro(header, 'DFN2_MODEL_ENCODER_GRU_LAYERS'),
              c_macro(header, 'DFN2_MODEL_ERB_GRU_LAYERS'),
              c_macro(header, 'DFN2_MODEL_DF_GRU_LAYERS'))
    assert layers == (1, 2, 2)
    rows = c_macro(header, 'DFN2_MODEL_GRU_ROWS')
    assert rows == sum(layers)
    assert (c_macro(header, 'DFN2_MODEL_GRU_ROW_ENCODER'),
            c_macro(header, 'DFN2_MODEL_GRU_ROW_ERB'),
            c_macro(header, 'DFN2_MODEL_GRU_ROW_DF')) == (
        0, layers[0], layers[0] + layers[1]), (
        'row offsets are the exporter slice starts')

    model = build_shipped_model(load_config()).eval()
    expected = (
        (1, 1, frames, n_erb),
        (1, 2, frames, df_bins),
        (1, rows, 1, hidden),
        (1, c_macro(header, 'DFN2_MODEL_ENCODER_CHANNELS'),
         c_macro(header, 'DFN2_MODEL_DF_PATHWAY_WINDOW'), df_bins),
    )
    actual = tuple(
        tuple(int(size) for size in value.shape)
        for value in streaming_export_inputs(model)
    )
    assert actual == expected
    wrapper = StatelessDFN2Heads(model)
    assert tuple(
        (name, start, stop) for name, start, stop in wrapper.gru_state_slices
    ) == (
        ('h_encoder', c_macro(header, 'DFN2_MODEL_GRU_ROW_ENCODER'),
         c_macro(header, 'DFN2_MODEL_GRU_ROW_ERB')),
        ('h_erb', c_macro(header, 'DFN2_MODEL_GRU_ROW_ERB'),
         c_macro(header, 'DFN2_MODEL_GRU_ROW_DF')),
        ('h_df', c_macro(header, 'DFN2_MODEL_GRU_ROW_DF'), rows),
    )


def _small_dfn2(mask_lookahead=1, df_lookahead=1):
    return DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
        mask_lookahead=mask_lookahead, df_lookahead=df_lookahead,
    ).eval()


def test_calibration_frame_shapes_equal_the_exported_graph_inputs(tmp_path):
    """One recorded calibration frame must BE one graph invocation.

    ``capture_calibration_inputs`` keeps the graph's batch dimension and
    ``np.stack`` adds the calibration-frame axis on top; dropping the batch
    axis would leave every ``.bin`` one rank short of the input the accelerator
    binds it to, with nothing downstream re-deriving the shape.
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
    graph = onnx.load(graph_path)
    graph_shapes = {
        value.name: [int(dim.dim_value)
                     for dim in value.type.tensor_type.shape.dim]
        for value in graph.graph.input
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
        model.mask_lookahead,
        model.df_lookahead,
    )
    header_version = c_macro(model_io_header(),
                             'DFN2_MODEL_IO_LAYOUT_VERSION')
    assert metadata['state_layout_version'] == header_version
    assert STATE_LAYOUT_VERSION == metadata['state_layout_version']
    # Retired numbers are never reused: a C-side bump onto one must fail here
    # rather than ship a header that means two different things to two
    # different graphs.
    assert RETIRED_LAYOUT_VERSIONS == frozenset({4, 5, 6, 7})
    assert header_version not in RETIRED_LAYOUT_VERSIONS, (
        'DFN2_MODEL_IO_LAYOUT_VERSION %d collides with a retired layout'
        % header_version
    )
    assert header_version == 8
    assert metadata['state_binding'] == STATE_BINDING == 'in_place_same_address'
    assert metadata['state_handoff'] == STATE_HANDOFF == {
        'h_gru': 'h_gru_out',
        'df_convp_history': 'df_convp_history_out',
    }


def test_the_tested_pairs_are_the_fifteen_legal_pairs():
    """The legal set is mask_lookahead in [0, 2] and df_lookahead in
    [0, df_order - 1], independent of each other: 15 pairs. The test pairs are
    probed from the model constructor (``dfn_pairs.accepted_pairs``), so this
    is the one place the rule is stated."""
    cfg = load_config()
    df_order = read_model_config(cfg)['df_order']
    assert LOOKAHEAD_PAIRS == [(mask, df) for mask in range(3)
                               for df in range(df_order)]
    assert len(LOOKAHEAD_PAIRS) == 15
    assert dfn_pairs.shipped_pair('DeepFilterNet2') in LOOKAHEAD_PAIRS


def test_window_alignment_names_the_frames_the_host_must_supply():
    """The shipped string is unchanged; the others follow from the padding."""
    assert window_alignment(1) == '[t-1,t,t+1] -> heads[t]'
    assert window_alignment(0) == '[t-2,t-1,t] -> heads[t]'
    assert window_alignment(2) == '[t,t+1,t+2] -> heads[t]'


@pytest.mark.parametrize('mask_lookahead', [0, 1, 2])
def test_feature_windows_hold_frames_t_minus_2_plus_L_through_t_plus_L(
        mask_lookahead):
    """Each slot of window t is frame t-(2-L)+slot, or zero off either end.

    Independent of the model: the frames carry their own index, so any
    off-by-one padding shows as a wrong number in a named slot."""
    frames = 6
    feature = torch.arange(1, frames + 1, dtype=torch.float32)
    feature = feature.view(1, 1, frames, 1).expand(1, 1, frames, 3).contiguous()
    windows = list(feature_windows(feature, mask_lookahead))
    assert len(windows) == frames
    for t, window in enumerate(windows):
        assert tuple(window.shape) == (1, 1, INPUT_FRAMES, 3)
        for slot in range(INPUT_FRAMES):
            source = t - (INPUT_FRAMES - 1 - mask_lookahead) + slot
            want = float(source + 1) if 0 <= source < frames else 0.0
            assert float(window[0, 0, slot, 0]) == want, (t, slot)


def test_mask_lookahead_outside_the_three_frame_window_names_the_contract():
    for bad in (-1, 3):
        with pytest.raises(ValueError, match='model contract'):
            window_frames(bad)
        with pytest.raises(ValueError, match='model contract'):
            window_alignment(bad)
        # Checked when the iterator is built, not when its first window is
        # taken.
        with pytest.raises(ValueError, match='fresh training run'):
            feature_windows(torch.zeros(1, 1, 4, 3), bad)


def test_the_host_window_must_match_the_convolution_padding():
    """The window is derived from mask_lookahead, the padding from the module
    that was built: a model whose recorded lookahead disagrees with its
    convolution is refused rather than exported with a mis-aligned window."""
    model = _small_dfn2(1, 1)
    model.mask_lookahead = 0
    with pytest.raises(ValueError, match='pads time'):
        StatelessDFN2Heads(model)


@pytest.mark.parametrize('mask_lookahead,df_lookahead', LOOKAHEAD_PAIRS)
def test_export_metadata_publishes_the_lookahead_pair(
        tmp_path, mask_lookahead, df_lookahead):
    """The pair and the window it implies are in the graph's metadata, so a
    host built for another pair is detectable from the exported artefact."""
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    model = _small_dfn2(mask_lookahead, df_lookahead)
    wrapper = StatelessDFN2Heads(model).eval()
    inputs = wrapper.initial_inputs()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    metadata = build_metadata(
        str(checkpoint),
        {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        inputs, outputs, wrapper.mask_lookahead, wrapper.df_lookahead,
    )
    assert metadata['mask_lookahead'] == mask_lookahead
    assert metadata['df_lookahead'] == df_lookahead
    assert metadata['input_window_alignment'] == window_alignment(
        mask_lookahead)
    assert metadata['input_feature_frames'] == INPUT_FRAMES


def test_config_lookahead_pair_is_pinned_to_the_c_header_defaults():
    """config.ini names the pair a checkpoint is trained for; the C build's
    default constants must be the same pair, or the shipped host and the
    shipped training config describe different models. A build for another
    pair overrides the macros (-DDFN2_MASK_LOOKAHEAD=..) and is a deliberate
    act; the defaults are what ships."""
    model_cfg = read_model_config(load_config())
    process = (ROOT_PATH / 'dfn2_process.h').read_text(encoding='utf-8')
    assert model_cfg['mask_lookahead'] == c_macro(
        process, 'DFN2_MASK_LOOKAHEAD')
    assert model_cfg['df_lookahead'] == c_macro(process, 'DFN2_DF_LOOKAHEAD')


def test_changing_the_lookahead_pair_of_a_checkpoint_asks_for_retraining():
    cfg = load_config()
    feature_cfg = read_feature_config(cfg, 48000, 512)
    loss_cfg = read_loss_config(cfg)

    def contract(mask, df):
        return make_checkpoint_contract(
            sr=48000, n_fft=1024, win_len=1024, hop_len=512, n_erb=32,
            df_bins=96, df_order=5, mask_lookahead=mask, df_lookahead=df,
            mask_pf=False, pf_beta=0.02, feature_cfg=feature_cfg,
            loss_cfg=loss_cfg,
        )

    trained = {
        'model_version': MODEL_VERSION,
        'feature_version': FEATURE_VERSION,
        'loss_version': LOSS_VERSION,
        'contract': contract(1, 1),
    }
    require_checkpoint_contract(trained, contract(1, 1))
    for mask, df in ((1, 0), (0, 0), (2, 1)):
        with pytest.raises(ValueError, match='fresh training run'):
            require_checkpoint_contract(trained, contract(mask, df))


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
def test_lookahead_ranges_are_enforced_in_the_model_constructor():
    """Each lookahead is checked against its own range, at the single source.

    Guards against the invariant drifting back out into per-entry-point copies:
    train.py and inference.py deliberately do not check it, so if the
    constructor stops enforcing it nothing does. The two are independent: a
    deep filter may look further ahead than the mask path.
    """
    cfg = load_config()
    df_order = read_model_config(cfg)['df_order']
    for bad in ({'mask_lookahead': 3}, {'mask_lookahead': -1},
                {'df_lookahead': df_order}, {'df_lookahead': -1}):
        with pytest.raises(ValueError, match='_lookahead'):
            build_shipped_model(cfg, **bad)
    build_shipped_model(cfg, mask_lookahead=0, df_lookahead=df_order - 1)
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
    spec, feat_erb, feat_spec, _ = extract_dfn2_features(
        spec, model.erb_fb, 96, feature_cfg=read_feature_config(cfg, 48000, hop)
    )
    enhanced_spec, erb_mask = model(spec, feat_erb, feat_spec)
    assert enhanced_spec.shape == spec.shape
    assert torch.isfinite(enhanced_spec.real).all()
    assert torch.isfinite(erb_mask).all()


def test_n_erb_divisibility_follows_upstreams_stricter_rule():
    """Upstream asserts nb_erb % 8; the two stride-2 stages only need % 4.

    ⚠ Calls the real validator.  A test that only asserted `32 % 8 == 0`
    locally would pass whichever rule train.py enforced; reverting the guard to
    % 4 fails this one.
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
    the trainer produces.  It calls the same function rather than transcribing the
    four lines (which can drift on the log floor and on the power expression),
    so this asserts the call rather than the text.
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
    A test that built both expressions itself and compared them would pass
    whichever form the trainer actually used; reverting the trainer to
    abs().pow(2) fails this one.
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
    """There is no max(1, emb_num_layers - 1) clamp, matching upstream.

    ⚠ With a clamp, emb_num_layers=1 would silently build 1 + 1 = 2 GRU layers
    while the config said 1, breaking the total-depth invariant config.ini
    documents. Upstream is a bare subtraction and nn.GRU rejects num_layers=0.
    """
    cfg = load_config()
    with pytest.raises(ValueError, match='num_layers'):
        build_shipped_model(cfg, emb_num_layers=1)

    # The shipped depth still resolves to 1 encoder + 2 ERB + 2 DF = 5.
    model = build_shipped_model(cfg)
    gru_layers = [n for n, _ in model.named_parameters()
                  if '.gru.weight_ih' in n]
    assert len(gru_layers) == 5


def test_runtime_erb_bins_match_the_model(tmp_path):
    """The exported .bin matrices equal DFN2's frozen buffers.

    The C host consumes caller-loaded pointers in these exact layouts, so the
    exported pair must reproduce the model's own filterbank bit for bit.
    """
    import sys as _sys
    for stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx',
                  'stream_model', 'export_erb_matrix'):
        _sys.modules.pop(stale, None)
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


def test_gru_state_rows_are_routed_to_their_own_gru():
    """Each slice of ``h_gru`` must land on the GRU the C header names.

    The rows are ``dfn2_model_io.h``'s memory order (encoder, ERB decoder, DF
    decoder), and the DF and ERB decoders read only the encoder's embedding,
    never each other's hidden state. So perturbing one decoder's rows may move
    only that decoder's rows of ``h_gru_out``; perturbing the encoder row moves
    the embedding and therefore both decoders.
    """
    torch.manual_seed(73)
    model = _small_dfn2()
    wrapper = StatelessDFN2Heads(model).eval()
    assert wrapper.input_names == (
        'erb', 'spec', 'h_gru', 'df_convp_history')
    assert wrapper.output_names[3:] == ('h_gru_out', 'df_convp_history_out')
    assert wrapper.state_layout_version == STATE_LAYOUT_VERSION
    assert tuple(wrapper.initial_inputs()[2].shape) == (1, 5, 1, 32)
    assert tuple(wrapper.initial_inputs()[3].shape) == (
        1, model.encoder.erb_conv0[1].out_channels, 5, model.df_bins)

    erb = torch.randn(1, 1, INPUT_FRAMES, model.n_erb)
    spec = torch.randn(1, 2, INPUT_FRAMES, model.df_bins)
    window = torch.randn_like(wrapper.initial_inputs()[3])
    base = torch.randn(1, 5, 1, 32)
    rows = {name: (start, stop) for name, start, stop
            in wrapper.gru_state_slices}

    def moved(perturbed):
        with torch.no_grad():
            first = wrapper(erb, spec, base, window)[3]
            second = wrapper(erb, spec, perturbed, window)[3]
        return [row for row in range(5)
                if not torch.equal(first[0, row], second[0, row])]

    def bump(name):
        start, stop = rows[name]
        bumped = base.clone()
        bumped[:, start:stop] += 0.5
        return bumped

    assert moved(bump('h_erb')) == [1, 2], 'ERB rows must move only themselves'
    assert moved(bump('h_df')) == [3, 4], 'DF rows must move only themselves'
    assert set(moved(bump('h_encoder'))) >= {0, 1, 2, 3, 4}, (
        'the encoder row feeds the embedding both decoders read')


@pytest.mark.parametrize('mask_lookahead,df_lookahead', LOOKAHEAD_PAIRS)
def test_stateless_graph_chain_is_the_offline_stream_in_both_state_tensors(
        mask_lookahead, df_lookahead):
    """The window the graph returns is the next call's input, unchanged.

    ``h_gru_out`` and ``df_convp_history_out`` are fed straight back for
    several hops; the heads must equal the offline model's, so nothing is lost
    by returning the whole pathway window instead of its tail, and the window
    returned at hop t is exactly the last five frames the pathway has seen.
    """
    torch.manual_seed(74)
    model = _small_dfn2(mask_lookahead, df_lookahead)
    wrapper = StatelessDFN2Heads(model).eval()
    frames = 12
    erb = torch.randn(1, 1, frames, model.n_erb)
    spec = torch.randn(1, 2, frames, model.df_bins)
    with torch.no_grad():
        reference = model.heads(erb, spec)
        state = wrapper.initial_inputs()[2:]
        heads = []
        windows = []
        for erb_window, spec_window in zip(
                feature_windows(erb, mask_lookahead),
                feature_windows(spec, mask_lookahead)):
            output = wrapper(erb_window, spec_window, *state)
            heads.append(output[1])
            windows.append(output[4])
            state = output[3:]
    torch.testing.assert_close(
        torch.cat(heads, dim=1), reference[1], rtol=1e-5, atol=1e-6)
    # Consecutive windows overlap by four frames: window[t][1:] == window[t+1][:-1]
    for before, after in zip(windows, windows[1:]):
        assert torch.equal(before[:, :, 1:], after[:, :, :-1])
    assert float(windows[-1].abs().max()) > 0.0


def _export_small_graph(tmp_path, batch_size=1):
    """Run the real export path (optimiser, folding, validator, metadata)."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    checkpoint = tmp_path / ('ckpt%d.pth' % batch_size)
    checkpoint.write_bytes(b'only hashed')
    torch.manual_seed(73)
    model = _small_dfn2()
    wrapper = StatelessDFN2Heads(model, batch_size=batch_size).eval()
    path = tmp_path / ('small_b%d.onnx' % batch_size)
    metadata = export_graph(
        wrapper,
        {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        str(checkpoint), str(path), verify=True,
    )
    return wrapper, path, metadata


@pytest.mark.parametrize('batch_size', [1, 3])
def test_exported_state_boundary_is_in_place_and_sliced_on_the_input(
        tmp_path, batch_size):
    """What the board binds: rank-4 state pairs of equal shape, the pathway
    window consumed only by one explicit Slice, the GRU rows only by Slices
    on dim 1 that tile it, and no graph output produced by a Slice."""
    import onnx
    from onnx import numpy_helper

    wrapper, path, metadata = _export_small_graph(tmp_path, batch_size)
    graph = onnx.load(str(path))
    shapes = {value.name: [dim.dim_value
                           for dim in value.type.tensor_type.shape.dim]
              for value in list(graph.graph.input) + list(graph.graph.output)}
    for state, back in STATE_HANDOFF.items():
        assert len(shapes[state]) == 4
        assert shapes[state] == shapes[back], (state, back)
    assert shapes[GRU_STATE_NAME] == [batch_size, 5, 1, 32]
    assert shapes[PATHWAY_STATE_NAME][2] == 5
    props = {prop.key: prop.value for prop in graph.metadata_props}
    assert props['state_binding'] == 'in_place_same_address'
    assert metadata['state_binding'] == 'in_place_same_address'
    assert props['state_layout_version'] == str(STATE_LAYOUT_VERSION)

    constants = {item.name: numpy_helper.to_array(item).tolist()
                 for item in graph.graph.initializer}
    users = [node for node in graph.graph.node
             if PATHWAY_STATE_NAME in node.input]
    assert [node.op_type for node in users] == ['Slice']
    starts, ends, axes, steps = (constants[name] for name in users[0].input[1:])
    assert (starts, ends, axes, steps) == ([1], [5], [2], [1])
    gru_users = [node for node in graph.graph.node
                 if GRU_STATE_NAME in node.input]
    assert gru_users and all(node.op_type == 'Slice' for node in gru_users)
    ranges = sorted((constants[node.input[1]][0], constants[node.input[2]][0])
                    for node in gru_users)
    assert ranges == [(0, 1), (1, 3), (3, 5)]
    assert all(constants[node.input[3]] == [1] for node in gru_users)
    produced = {name: node for node in graph.graph.node
                for name in node.output}
    for output in graph.graph.output:
        assert produced[output.name].op_type != 'Slice', output.name


def _drop_the_other_end(model, name):
    """Point the Slice that reads window ``name`` at the opposite end of the
    window, keeping its length: [1:K) becomes [0:K-1) and back."""
    from onnx import numpy_helper

    constants = {item.name: numpy_helper.to_array(item)
                 for item in model.graph.initializer}
    node = next(item for item in model.graph.node if name in item.input)
    start, end = (int(constants[node.input[index]].reshape(-1)[0])
                  for index in (1, 2))
    new = (0, end - 1) if start == 1 else (1, end + 1)
    for index, value in ((1, new[0]), (2, new[1])):
        label = 'flipped_%s_%d' % (name, index)
        model.graph.initializer.append(numpy_helper.from_array(
            np.array([value], dtype=np.int64), label))
        node.input[index] = label
    return model


def test_a_window_that_drops_the_wrong_end_is_refused(tmp_path):
    """The pathway window is time ordered: the graph drops its first frame,
    the metadata says so, and a graph (or a declaration) with the other end
    is refused by the validator the exporter ran."""
    import onnx
    from onnx_streaming_contract import validate_inplace_state_from_metadata

    _wrapper, path, _metadata = _export_small_graph(tmp_path)
    graph = onnx.load(str(path))
    props = {prop.key: prop.value for prop in graph.metadata_props}
    slicing = json.loads(props['state_slicing'])
    assert slicing[PATHWAY_STATE_NAME]['drop'] == 'first'
    assert validate_inplace_state_from_metadata(graph) == len(STATE_HANDOFF)
    mutant = _drop_the_other_end(onnx.load(str(path)), PATHWAY_STATE_NAME)
    with pytest.raises(RuntimeError, match='does not drop the first slot'):
        validate_inplace_state_from_metadata(mutant)
    mutant = onnx.load(str(path))
    for prop in mutant.metadata_props:
        if prop.key == 'state_slicing':
            declared = json.loads(prop.value)
            declared[PATHWAY_STATE_NAME]['drop'] = 'last'
            prop.value = json.dumps(declared)
    with pytest.raises(RuntimeError, match='does not drop the last slot'):
        validate_inplace_state_from_metadata(mutant)


def test_gru_state_reaches_every_gru_through_slices(tmp_path):
    """Each recurrent row block reaches its own GRU nodes and nothing merges
    them first: if a later change let the rows meet at a Concat before the
    GRUs, the three stacks would share a quantization scale in a worse way
    than the single input tensor already makes them. ONNX has no stacked GRU
    op, so a two-layer stack legitimately becomes two GRU nodes fed by a
    further internal Slice; the count below is nodes, not state tensors."""
    import onnx

    wrapper, path, _metadata = _export_small_graph(tmp_path)
    graph = onnx.load(str(path))

    def _reaches_gru(name, depth=0):
        found = []
        for node in graph.graph.node:
            if name not in node.input:
                continue
            if node.op_type == 'GRU':
                found.append(node.name)
            elif node.op_type in ('Slice', 'Squeeze', 'Unsqueeze',
                                  'Identity', 'Reshape') and depth < 6:
                for output_name in node.output:
                    found.extend(_reaches_gru(output_name, depth + 1))
            elif node.op_type == 'Concat':
                raise AssertionError(
                    'h_gru reaches a Concat before its GRUs')
        return found

    reached = _reaches_gru(GRU_STATE_NAME)
    expected = sum(stop - start for _name, start, stop
                   in wrapper.gru_state_slices)
    assert len(set(reached)) == expected, (
        'expected one distinct GRU node per recurrent layer, reached %r'
        % sorted(set(reached))
    )


def test_combined_gru_calibration_report_slices_the_layer_axis():
    """The added graph-batch axis must not shift per-GRU PTQ statistics."""
    captured = np.zeros((2, 1, 5, 1, 3), dtype=np.float32)
    captured[:, :, 0:1] = 1.0
    captured[:, :, 1:3] = 2.0
    captured[:, :, 3:5] = 3.0

    def stats(value):
        return {
            'shape': list(value.shape),
            'min': float(value.min()),
            'max': float(value.max()),
        }

    report = gru_state_slice_report(
        captured, stats,
        (('h_encoder', 0, 1), ('h_erb', 1, 3), ('h_df', 3, 5)),
    )
    assert report['per_layer']['h_encoder']['shape'] == [2, 1, 1, 1, 3]
    assert report['per_layer']['h_erb']['shape'] == [2, 1, 2, 1, 3]
    assert report['per_layer']['h_df']['shape'] == [2, 1, 2, 1, 3]
    assert report['per_layer']['h_encoder']['max'] == 1.0
    assert report['per_layer']['h_erb']['max'] == 2.0
    assert report['per_layer']['h_df']['max'] == 3.0


def test_fixed_batch_graph_matches_independent_streaming_lanes():
    """Batch profiling must not turn adjacent frames into shared state.

    Run three independent batch-one invocations, merge exactly the tensor
    axes published by ``inference_batch.py``, and require the fixed-batch
    wrapper to produce the same result for every head and state tensor.
    """
    from inference_batch import _merge_independent_samples

    torch.manual_seed(809)
    model = _small_dfn2().eval()
    single = StatelessDFN2Heads(model, batch_size=1).eval()
    batch = StatelessDFN2Heads(model, batch_size=3).eval()
    samples = []
    expected = []
    with torch.no_grad():
        for _ in range(3):
            values = tuple(torch.randn_like(value)
                           for value in single.initial_inputs())
            samples.append(values)
            expected.append(single(*values))
        merged = _merge_independent_samples(batch.input_names, samples)
        actual = batch(*merged)

    assert tuple(merged[0].shape[:2]) == (3, 1)
    assert tuple(merged[2].shape) == (3, 5, 1, 32)
    assert merged[3].shape[0] == 3 and merged[3].shape[2] == 5

    for output_index, got in enumerate(actual):
        want = torch.cat(
            tuple(value[output_index] for value in expected), dim=0
        )
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)


def test_fixed_batch_metadata_publishes_static_lane_count(tmp_path):
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'fixed-batch metadata test')
    model = _small_dfn2().eval()
    wrapper = StatelessDFN2Heads(model, batch_size=4).eval()
    inputs = wrapper.initial_inputs()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    metadata = build_metadata(
        str(checkpoint),
        {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        inputs,
        outputs,
        model.mask_lookahead,
        model.df_lookahead,
    )
    assert metadata['fixed_batch_size'] == 4
    assert metadata['batch_semantics'] == 'independent_streaming_lanes'
    assert metadata['input_schema']['erb'][0] == 4
    assert metadata['input_schema']['h_gru'] == [4, 5, 1, 32]


def test_fixed_batch_bin_contains_every_lane(tmp_path):
    from calibration_io import (
        capture_calibration_inputs,
        write_calibration_artifact,
    )
    from inference_batch import _merge_independent_samples

    model = _small_dfn2().eval()
    single = StatelessDFN2Heads(model, batch_size=1).eval()
    batch = StatelessDFN2Heads(model, batch_size=3).eval()
    samples = [single.initial_inputs() for _ in range(3)]
    merged = _merge_independent_samples(batch.input_names, samples)
    captured = {}
    capture_calibration_inputs(captured, batch.input_names, merged)
    arrays = {name: np.stack(values).astype(np.float32, copy=False)
              for name, values in captured.items()}
    artifact = tmp_path / 'batch_calib'
    write_calibration_artifact(
        artifact, arrays, {'frames': 1}, 'bin'
    )
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['binary_tensors']['erb']['frame_shape'][0] == 3
    assert manifest['binary_tensors']['h_gru']['frame_shape'] == [3, 5, 1, 32]
    blob = np.fromfile(
        artifact / 'h_gru' / 'h_gru_0000.bin', '<f4'
    )
    assert blob.size == 3 * 5 * 1 * 32


def test_runtime_erb_bins_are_a_bitwise_partition_of_unity(tmp_path):
    """The erb_inv.bin the C host loads must expand a unit band mask to
    exactly 1.0f per bin in df_common_expand_mask's accumulation order --
    dfn2_prepost_frame_skip's identity, and every host identity gate built
    on it, is exact only under that property. The runtime exporter refuses
    to write a matrix that breaks it, and the check itself must be able to
    fail."""
    from export_erb_matrix import require_partition_of_unity, write_runtime_bins

    out_dir = write_runtime_bins(os.path.join(ROOT, 'config.ini'), str(tmp_path))
    manifest = json.load(open(os.path.join(out_dir, 'erb_matrices.json'), encoding='utf-8'))
    n_erb, n_bins = manifest['erb_inv.bin']['shape']
    inverse = np.fromfile(os.path.join(out_dir, 'erb_inv.bin'), dtype='<f4').reshape(n_erb, n_bins)
    require_partition_of_unity(inverse)
    bin_gain = np.zeros(n_bins, np.float32)
    for row in inverse:
        bin_gain += row
    assert np.array_equal(bin_gain, np.ones(n_bins, np.float32))

    broken = inverse.copy()
    broken[3] *= np.float32(1.0 + 1e-4)
    with pytest.raises(RuntimeError):
        require_partition_of_unity(broken)
    shifted = inverse.copy()
    shifted[5, 100] += np.float32(2 ** -20)
    with pytest.raises(RuntimeError):
        require_partition_of_unity(shifted)


def test_batch1_cleanup_keeps_the_graph_exact_and_rank_four(tmp_path):
    """The grouped linears' rank-5 MatMuls and their Squeeze/Unsqueeze chains
    are batch-1 artefacts: onnx_batch1_optimizer.py (run on the exported graph)
    removes them without changing one output bit."""
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    import copy

    # The cleanup lives in Audio_ALG/PTQ, outside this release unit; the
    # exporter does not call it, so the check skips when AINR is released alone.
    sys.path.append(os.fspath(pathlib.Path(__file__).resolve().parents[3] / 'PTQ'))
    b1 = pytest.importorskip('onnx_batch1_optimizer')

    torch.manual_seed(74)
    model = DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64,
        enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4,
    ).eval()
    wrapper = StatelessDFN2Heads(model).eval()
    path = tmp_path / 'raw.onnx'
    torch.onnx.export(wrapper, wrapper.initial_inputs(), str(path),
                      input_names=list(INPUT_NAMES),
                      output_names=list(OUTPUT_NAMES), opset_version=17,
                      do_constant_folding=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    options.optimized_model_filepath = str(path) + '.fold'
    options.log_severity_level = 3
    ort.InferenceSession(str(path), options,
                         providers=['CPUExecutionProvider'])
    graph = onnx.load(str(path) + '.fold')
    original = copy.deepcopy(graph)
    report = b1.optimize_batch1(graph)
    assert report['high_rank_left'] == [] and report['blocked'] == []
    assert report['nodes_after'] < report['nodes_before']
    assert b1.verify_identical(original, graph) == 0.0
