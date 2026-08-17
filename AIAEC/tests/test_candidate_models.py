import pathlib

import pytest
import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.Align_ULCNet import AlignULCNet, ChannelSampledReorientation
from AIAEC.CAGCRN import CAGCRN
from AIAEC.DeepVQE_S import DeepVQES
from AIAEC.dataset_gen import MODEL_TASKS


G16 = SignalGrid(16000, 512, 512, 256)
G16_LOW = SignalGrid(16000, 256, 256, 128)
G48 = SignalGrid(48000, 1024, 1024, 512)


def _spec(batch, frames, bins):
    return torch.complex(torch.randn(batch, frames, bins),
                         torch.randn(batch, frames, bins))


def _assert_finite_parameter_gradients(model, loss):
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients, "model forward is detached from every trainable parameter"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("factory", [
    lambda: AlignCRUSE(G16),
    lambda: AlignULCNet(G16),
    lambda: DeepVQES(G16),
    lambda: CAGCRN(G16),
])
def test_waveform_boundary_models_forward_finite(factory):
    model = factory().eval()
    x = _spec(1, 4, G16.n_freqs)
    with torch.no_grad():
        out = model(x, x)
    assert out.enhanced.shape == x.shape
    assert torch.is_complex(out.enhanced)
    assert torch.isfinite(torch.view_as_real(out.enhanced)).all()


@pytest.mark.parametrize("factory", [
    lambda: AlignCRUSE(G16),
    lambda: AlignULCNet(G16),
    lambda: DeepVQES(G16),
    lambda: CAGCRN(G16),
])
def test_waveform_boundary_models_backward_finite(factory):
    model = factory().train()
    x = _spec(2, 3, G16.n_freqs)
    _assert_finite_parameter_gradients(model, model(x, x).enhanced.abs().mean())


@pytest.mark.parametrize("factory", [
    lambda: AlignCRUSE(G48),
    lambda: AlignULCNet(G48),
    lambda: DeepVQES(G48),
    lambda: CAGCRN(G48),
])
def test_grid_adapted_models_accept_48k(factory):
    model = factory().eval()
    x = _spec(1, 2, G48.n_freqs)
    with torch.no_grad():
        out = model(x, x)
    assert out.enhanced.shape == x.shape


def test_cagcrn_soft_window_has_gradient():
    model = CAGCRN(G16).train()
    x = _spec(2, 3, G16.n_freqs)
    model(x, x).enhanced.abs().mean().backward()
    grad = model.cata.raw_window.grad
    assert grad is not None
    assert torch.isfinite(grad)
    assert grad.abs() > 0


def test_align_cruse_defaults_to_streaming_distribution_and_convt_decoder():
    model = AlignCRUSE(G16).eval()
    x = _spec(2, 8, G16.n_freqs)
    with torch.no_grad():
        out = model(x, x)
    assert out.delay_distribution.shape == (2, 8, model.max_delay_frames)
    assert isinstance(model.up3.conv, torch.nn.ConvTranspose2d)
    assert isinstance(model.up2.conv, torch.nn.ConvTranspose2d)
    assert isinstance(model.up1.conv, torch.nn.ConvTranspose2d)
    assert isinstance(model.mask_up.conv, torch.nn.ConvTranspose2d)


def test_align_cruse_paper_global_mode_is_explicitly_available():
    model = AlignCRUSE(G16, alignment_mode="paper_global").eval()
    x = _spec(2, 8, G16.n_freqs)
    with torch.no_grad():
        out = model(x, x)
    assert out.delay_distribution.shape == (2, model.max_delay_frames)


def test_csamfr_samples_two_bin_subbands_not_individual_bins():
    block = ChannelSampledReorientation(257, gamma=5, subband_bins=2)
    source = torch.arange(257.0).reshape(1, 1, 1, 257)
    sampled = block(source)
    assert block.n_subbands == 130
    assert sampled.shape == (1, 5, 1, 52)
    # Set zero: bands 0,5,10... => bins [0,1,10,11,20,21,...].
    torch.testing.assert_close(sampled[0, 0, 0, :6],
                               torch.tensor([0., 1., 10., 11., 20., 21.]))
    torch.testing.assert_close(block.inverse(sampled), source[:, 0])


def test_published_size_classes_are_preserved_on_project_grid():
    counts = {
        "align_cruse": sum(p.numel() for p in AlignCRUSE(G16).parameters()),
        "align_ulcnet": sum(p.numel() for p in AlignULCNet(G16).parameters()),
        "deepvqe_s": sum(p.numel() for p in DeepVQES(G16).parameters()),
    }
    assert 680_000 <= counts["align_cruse"] <= 780_000
    assert 640_000 <= counts["align_ulcnet"] <= 720_000
    assert 580_000 <= counts["deepvqe_s"] <= 680_000

    cag = CAGCRN(G16)
    # The paper's 0.07 M count includes fixed ERB analysis/synthesis weights.
    state_elements = sum(value.numel() for value in cag.state_dict().values())
    assert 60_000 <= state_elements <= 80_000
    assert cag.mic_tfgru is not cag.far_tfgru


@pytest.mark.parametrize("factory", [
    lambda: AlignCRUSE(G16_LOW),
    lambda: AlignULCNet(G16_LOW),
    lambda: DeepVQES(G16_LOW),
    lambda: CAGCRN(G16_LOW),
])
def test_grid_adapted_models_accept_16k_low_latency_grid(factory):
    model = factory().eval()
    x = _spec(1, 2, G16_LOW.n_freqs)
    with torch.no_grad():
        assert model(x, x).enhanced.shape == x.shape


@pytest.mark.parametrize("factory", [
    lambda: AlignCRUSE(G16),
    lambda: AlignULCNet(G16),
    lambda: DeepVQES(G16),
    lambda: CAGCRN(G16),
])
def test_no_future_frame_leakage(factory):
    torch.manual_seed(7)
    model = factory().eval()
    a = _spec(1, 6, G16.n_freqs)
    b = a.clone()
    b[:, 4:] = _spec(1, 2, G16.n_freqs)
    with torch.no_grad():
        ya = model(a, a).enhanced[:, :4]
        yb = model(b, b).enhanced[:, :4]
    torch.testing.assert_close(ya, yb, rtol=1e-5, atol=1e-6)


def test_delay_attention_chunking_is_bit_exact_to_explicit_broadcast():
    """The long-file memory optimization preserves the original arithmetic."""
    from AIAEC.aiaec_common import FrameDelayAttention, causal_delay_stack

    torch.manual_seed(17)
    attention = FrameDelayAttention(3, 4, 5, 6, 7).eval()
    mic = torch.randn(2, 3, 9, 11)
    far = torch.randn(2, 4, 9, 11)
    with torch.no_grad():
        actual_aligned, actual_distribution = attention(mic, far)

        q = attention.query(mic)
        k_delayed = causal_delay_stack(attention.key(far), 7)
        logits = (q.unsqueeze(3) * k_delayed).sum(dim=-1)
        logits = attention.score(logits).squeeze(1)
        expected_distribution = torch.softmax(logits, dim=-1)
        v_delayed = causal_delay_stack(attention.value(far), 7)
        expected_aligned = (
            v_delayed * expected_distribution[:, None, :, :, None]
        ).sum(dim=3)

    assert torch.equal(actual_distribution, expected_distribution)
    assert torch.equal(actual_aligned, expected_aligned)


def test_old_generic_projects_are_removed():
    root = pathlib.Path(__file__).parents[1]
    for stale in ("AECNet", "PostFilter", "JointAECNR"):
        assert not (root / stale).exists()


def test_model_task_attributes_match_the_training_contract():
    classes = {
        'Align_CRUSE': AlignCRUSE,
        'Align_ULCNet': AlignULCNet,
        'DeepVQE_S': DeepVQES,
        'CAGCRN': CAGCRN,
    }
    assert {name: cls.task for name, cls in classes.items()} == MODEL_TASKS
