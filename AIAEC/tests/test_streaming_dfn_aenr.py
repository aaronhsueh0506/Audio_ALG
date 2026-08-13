"""Frame-by-frame streaming equivalence for DeepFilterNet-AENR.

The offline reference is the model's own ``forward``; ``forward_stream`` must
reproduce it exactly two hops late (mask_lookahead + df_lookahead, serial in
the DFN2 cascade), with ``flush_stream`` supplying the final two frames.  The
feature path (extract_dfn2_features) is a per-frame EMA recursion and is
checked at its own boundary.
"""

import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.DeepFilterNet_AENR import DeepFilterNetAENR
from AINR.DeepFilterNet2.model import _build_erb_fb
from AINR.DeepFilterNet2.train import (
    analysis_scale,
    extract_dfn2_features,
    make_norm_alpha,
)

G16 = SignalGrid(16000, 512, 512, 256)

# Measured on the T=90 equivalence run below (float32 CPU): max |delta|
# enhanced 3.4e-7, mask 6.0e-8, coefs 1.3e-7, alpha 6.0e-8.  Pinned with
# >10x headroom, still far inside the 1e-4 target.
TOL = 1e-5


def _model(seed=0):
    torch.manual_seed(seed)
    model = DeepFilterNetAENR(
        G16, enc_ch=16, emb_size=64, df_hidden=64,
        lin_groups=8, enc_lin_groups=8,
    )
    with torch.no_grad():
        # The shipped conditioner init is an error passthrough that IGNORES
        # the far branch; randomize it so streaming equivalence (and the
        # far-shift can-fail test) actually exercise the far path.
        model.erb_condition.weight.normal_(std=0.5)
        model.erb_condition.bias.normal_(std=0.1)
        model.spec_condition.weight.normal_(std=0.5)
        model.spec_condition.bias.normal_(std=0.1)
    return model.eval()


def _inputs(t_frames, seed):
    torch.manual_seed(seed)
    return {
        "linear_error": torch.complex(torch.randn(1, t_frames, G16.n_freqs),
                                      torch.randn(1, t_frames, G16.n_freqs)),
        "error_erb": torch.randn(1, 1, t_frames, 32),
        "error_spec": torch.randn(1, 2, t_frames, 64),
        "far_erb": torch.randn(1, 1, t_frames, 32),
        "far_spec": torch.randn(1, 2, t_frames, 64),
    }


def _run_stream(model, inputs):
    """Feed one frame per call, flush, and concatenate to offline layout."""
    state = model.create_stream_state()
    t_frames = inputs["linear_error"].shape[1]
    per_call = []
    outs, masks, coefs, alphas = [], [], [], []
    with torch.no_grad():
        for t in range(t_frames):
            out = model.forward_stream(
                inputs["linear_error"][:, t:t + 1],
                inputs["error_erb"][:, :, t:t + 1],
                inputs["error_spec"][:, :, t:t + 1],
                inputs["far_erb"][:, :, t:t + 1],
                inputs["far_spec"][:, :, t:t + 1],
                state,
            )
            per_call.append(out.enhanced.shape[1])
            outs.append(out.enhanced)
            masks.append(out.mask)
            coefs.append(out.auxiliary["deep_filter_coefficients"])
            alphas.append(out.auxiliary["deep_filter_alpha"])
        tail = model.flush_stream(state)
    outs.append(tail.enhanced)
    masks.append(tail.mask)
    coefs.append(tail.auxiliary["deep_filter_coefficients"])
    alphas.append(tail.auxiliary["deep_filter_alpha"])
    return {
        "per_call": per_call,
        "tail_frames": tail.enhanced.shape[1],
        "enhanced": torch.cat(outs, dim=1),
        "mask": torch.cat(masks, dim=2),
        "coefs": torch.cat(coefs, dim=1),
        "alpha": torch.cat(alphas, dim=1),
    }


def _offline(model, inputs):
    with torch.no_grad():
        return model(**inputs)


def _feature_cfg():
    alpha = make_norm_alpha(16000, 256, 1.0)
    return {
        "analysis_scale": analysis_scale(512, 512, 256),
        "erb_alpha": alpha,
        "erb_init_lo_db": -60.0,
        "erb_init_hi_db": -90.0,
        "erb_scale_db": 40.0,
        "spec_alpha": alpha,
        "spec_init_lo": 0.001,
        "spec_init_hi": 0.0001,
    }


def test_feature_extraction_streams_frame_by_frame():
    torch.manual_seed(3)
    erb_fb, _ = _build_erb_fb(512, 16000, 32)
    spec = torch.complex(torch.randn(1, 257, 90), torch.randn(1, 257, 90))
    cfg = _feature_cfg()
    _, erb_ref, feat_ref, _ = extract_dfn2_features(
        spec, erb_fb, 64, feature_cfg=cfg, ema_state=None)
    state = None
    erb_frames, feat_frames = [], []
    for t in range(spec.shape[-1]):
        _, erb_t, feat_t, state = extract_dfn2_features(
            spec[:, :, t:t + 1], erb_fb, 64, feature_cfg=cfg, ema_state=state)
        erb_frames.append(erb_t)
        feat_frames.append(feat_t)
    erb_s = torch.cat(erb_frames, dim=2)
    feat_s = torch.cat(feat_frames, dim=2)
    assert (erb_s - erb_ref).abs().max().item() <= 1e-6
    assert (feat_s - feat_ref).abs().max().item() <= 1e-6


def test_forward_stream_matches_offline_two_frames_late():
    model = _model()
    inputs = _inputs(90, seed=1)
    offline = _offline(model, inputs)
    stream = _run_stream(model, inputs)

    # Warm-up contract: the first stream_output_delay calls emit nothing,
    # every later call emits exactly one frame, flush supplies the last two.
    assert model.stream_output_delay == 2
    delay = model.stream_output_delay
    assert stream["per_call"][:delay] == [0] * delay
    assert stream["per_call"][delay:] == [1] * (90 - delay)
    assert stream["tail_frames"] == delay

    assert stream["enhanced"].shape == offline.enhanced.shape
    assert (stream["enhanced"] - offline.enhanced).abs().max().item() <= TOL
    assert (stream["mask"] - offline.mask).abs().max().item() <= TOL
    ref_coefs = offline.auxiliary["deep_filter_coefficients"]
    ref_alpha = offline.auxiliary["deep_filter_alpha"]
    assert (stream["coefs"] - ref_coefs).abs().max().item() <= TOL
    assert (stream["alpha"] - ref_alpha).abs().max().item() <= TOL


def test_far_shift_breaks_equivalence():
    """CAN-FAIL: a one-frame shift of the far reference must be visible."""
    model = _model()
    inputs = _inputs(48, seed=2)
    offline = _offline(model, inputs)
    shifted = dict(inputs)
    for name in ("far_erb", "far_spec"):
        x = inputs[name]
        shifted[name] = torch.cat(
            [torch.zeros_like(x[:, :, :1]), x[:, :, :-1]], dim=2)
    stream = _run_stream(model, shifted)
    assert (stream["enhanced"] - offline.enhanced).abs().max().item() > 1e-3


def test_fresh_states_do_not_cross_contaminate():
    model = _model()
    first = _inputs(44, seed=11)
    second = _inputs(44, seed=12)
    stream_first = _run_stream(model, first)
    stream_second = _run_stream(model, second)
    offline_first = _offline(model, first)
    offline_second = _offline(model, second)
    assert (stream_first["enhanced"]
            - offline_first.enhanced).abs().max().item() <= TOL
    assert (stream_second["enhanced"]
            - offline_second.enhanced).abs().max().item() <= TOL
