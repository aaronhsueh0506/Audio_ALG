"""Behavioural tests for the AECNet architecture.

Every test here pins something a consumer cannot notice being broken by reading
a loss curve: an output that is secretly a mask on the microphone, a
convolution that has quietly become non-causal, a recurrent state that is not
actually carried across chunks, or a zero-reference gate that nobody checks.
"""

import configparser
import math
import os
import pathlib
import sys

import pytest
import torch
import torch.nn as nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AINR = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(1, AINR)

# Each of the model projects in ainr/ has its own top-level ``train.py`` /
# ``model.py``.  Under one pytest session the first import wins sys.modules, so
# a sibling project's tests would silently exercise this code (or vice versa).
# Dropping the cached entries forces re-resolution against the ROOT above.
for _stale in ('train', 'denoise', 'model', 'checkpoint_utils'):
    sys.modules.pop(_stale, None)

from model import (  # noqa: E402
    AecNet,
    AecNetConfig,
    assert_zero_reference_gate,
    build_model,
    compress_spec,
    expand_spec,
    zero_reference_leak_db,
)

from dataset_gen.aec import AecGrid  # noqa: E402


SMALL = AecNetConfig(channels=(8, 16), kernel_t=2, kernel_f=5, stride_f=2,
                     gru_layers=1, gru_groups=2, lookahead=0,
                     compress_exponent=0.3)


def _small_model(n_freqs=65, **overrides):
    cfg = SMALL if not overrides else AecNetConfig(
        **{**SMALL.__dict__, **overrides})
    torch.manual_seed(0)
    model = AecNet(n_freqs=n_freqs, config=cfg)
    model.eval()
    return model


# ============================================================
# Shape / finiteness
# ============================================================

def test_untrained_forward_is_shaped_and_finite():
    """An untrained model is only asked for shape and finiteness.

    ⚠ Deliberately NOT asserted here: the zero-reference gate. An untrained
    network fails it by construction, so asserting it on random weights would
    make the test either vacuous or permanently red. The gate is
    ``assert_zero_reference_gate``, and the trained-model evaluation calls it.
    """
    model = _small_model()
    x = torch.randn(3, 4, 11, 65)
    out, state = model(x)
    assert out.shape == (3, 2, 11, 65)
    assert torch.isfinite(out).all()
    assert set(state) == {'enc', 'gru', 'dec'}
    assert all(torch.isfinite(t).all() for t in state['enc'])
    assert torch.isfinite(state['gru']).all()


def test_output_is_an_echo_estimate_not_a_mask_on_the_microphone():
    """THE defining property: D_hat has 2 channels and does not multiply Y.

    With Y == 0 a mask-on-Y architecture can only ever produce exactly zero, no
    matter what the reference does. This model must still be able to emit an
    echo estimate -- and to make it DEPEND on X -- because the echo is a
    function of the reference, not a selected part of the microphone.
    """
    model = _small_model()
    zero_y = torch.zeros(2, 4, 9, 65)

    x_a = zero_y.clone()
    x_a[:, 2:] = torch.randn(2, 2, 9, 65)
    x_b = zero_y.clone()
    x_b[:, 2:] = torch.randn(2, 2, 9, 65) * 3.0

    with torch.no_grad():
        out_a, _ = model(x_a)
        out_b, _ = model(x_b)

    assert out_a.shape[1] == 2, "D_hat must be 2 channels (real, imag)"
    assert float(out_a.abs().max()) > 0.0, (
        "with Y == 0 the model produced exactly zero; that is what a mask on "
        "the microphone would do, and it means the model cannot represent an "
        "echo at all")
    assert float((out_a - out_b).abs().max()) > 0.0, (
        "changing the far-end reference did not change D_hat; the model is not "
        "using X")


def test_grid_is_config_driven_not_hardcoded_at_16k():
    """The 48 kHz variant must build from the grid alone."""
    grid48 = AecGrid(sr=48000, n_fft=1024, win_len=1024, hop_len=512)
    assert grid48.n_freqs == 513
    model = _small_model(n_freqs=grid48.n_freqs)
    out, _ = model(torch.randn(1, 4, 7, 513))
    assert out.shape == (1, 2, 7, 513)
    assert torch.isfinite(out).all()


def test_default_config_model_is_in_budget():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    grid = AecGrid.from_config(cfg)
    model = build_model(cfg, grid)
    n_params = model.n_parameters()
    assert 1_000_000 <= n_params <= 3_000_000, (
        f"{n_params:,} parameters is outside the 1-3 M design budget")
    out, _ = model(torch.randn(1, 4, 5, grid.n_freqs))
    assert out.shape == (1, 2, 5, grid.n_freqs)
    assert torch.isfinite(out).all()


# ============================================================
# Causality
# ============================================================

def _first_changed_frame(model, x, frame, bump=5.0):
    with torch.no_grad():
        base, _ = model(x)
        perturbed = x.clone()
        perturbed[:, :, frame, :] += bump
        other, _ = model(perturbed)
    delta = (base - other).abs().amax(dim=(0, 1, 3))     # per frame
    changed = (delta > 0).nonzero().flatten()
    return (int(changed[0]) if changed.numel() else None), delta


@pytest.mark.parametrize('lookahead', [0, 1, 3])
def test_core_is_strictly_causal_at_every_lookahead(lookahead):
    """Perturbing input frame t must leave every earlier output BIT-IDENTICAL.

    ⚠ This holds for every lookahead, and it is not a weaker statement than it
    sounds. Any model that can be fed in chunks satisfies out[i] = f(x[0..i]) --
    when index i has to be emitted, frame i+1 has not arrived. A right-padded
    "non-causal" conv does not escape that; its streaming implementation buffers,
    and the buffering IS the delay. So the core is causal and the lookahead
    lives entirely in which input frame an output index is an estimate FOR,
    which the next test pins down.
    """
    model = _small_model(lookahead=lookahead)
    n_frames = 13
    frame = 5
    x = torch.randn(1, 4, n_frames, 65)
    first, delta = _first_changed_frame(model, x, frame)
    assert first is not None, "perturbing an input frame changed no output at all"
    assert first == frame, (
        f"lookahead={lookahead}: perturbing input frame {frame} first changed "
        f"output index {first}; the core is not causal. Per-frame delta: "
        f"{delta.tolist()}")
    assert float(delta[:frame].max()) == 0.0


@pytest.mark.parametrize('lookahead', [0, 1, 3])
def test_lookahead_is_exactly_the_declared_output_delay(lookahead):
    """The estimate FOR input frame t sees inputs t..t+lookahead and no further.

    ``align_to_input`` undoes the declared delay, so index t of its result is
    the estimate for input frame t. Perturbing input frame t0 must therefore
    first disturb aligned index ``t0 - lookahead`` -- i.e. the estimate for a
    frame that far in the past is allowed to use t0, and the estimate for
    ``t0 - lookahead - 1`` is not. That is "changing a future frame must not
    change the current output, except by exactly the configured lookahead".
    """
    model = _small_model(lookahead=lookahead)
    n_frames = 13
    frame = 7
    x = torch.randn(1, 4, n_frames, 65)
    with torch.no_grad():
        base = model.align_to_input(model(x)[0])
        perturbed = x.clone()
        perturbed[:, :, frame, :] += 5.0
        other = model.align_to_input(model(perturbed)[0])
    delta = (base - other).abs().amax(dim=(0, 1, 3))
    changed = (delta > 0).nonzero().flatten()
    assert changed.numel(), "no aligned estimate responded to the input at all"
    assert int(changed[0]) == frame - lookahead, (
        f"lookahead={lookahead}: input frame {frame} first changed the estimate "
        f"for input frame {int(changed[0])}, expected {frame - lookahead}")
    assert base.shape[2] == n_frames - lookahead


def test_causality_holds_in_train_mode_too():
    """⚠ The normalisation must not peek at the future.

    BatchNorm2d -- what every published CRUSE implementation uses -- normalises
    over the time axis at training time, so frame t's statistics include frames
    t+1..T-1 and this test fails in train() while passing in eval(). That is
    exactly the kind of test that proves less than it looks like it does, so the
    model uses a per-frame norm and the property is asserted in both modes.
    """
    model = _small_model()
    model.train()
    x = torch.randn(1, 4, 11, 65)
    first, _ = _first_changed_frame(model, x, 4)
    assert first == 4


# ============================================================
# Streaming state
# ============================================================

@pytest.mark.parametrize('lookahead', [0, 2])
def test_chunked_state_matches_one_shot(lookahead):
    """4 s chunks with state carried == one long call.

    This is the guarantee the trainer relies on when it walks a 20-60 s sequence
    across batches, and the guarantee a streaming implementation relies on. If
    it fails, "carrying the state" is a comment rather than a fact.
    """
    model = _small_model(lookahead=lookahead)
    x = torch.randn(2, 4, 21, 65)
    with torch.no_grad():
        whole, _ = model(x)
        state = None
        pieces = []
        for start in range(0, x.shape[2], 4):
            piece, state = model(x[:, :, start:start + 4], state)
            pieces.append(piece)
        streamed = torch.cat(pieces, dim=2)
    torch.testing.assert_close(streamed, whole, rtol=1e-4, atol=1e-6)


def test_reset_state_zeroes_only_the_flagged_lanes():
    model = _small_model()
    x = torch.randn(4, 4, 6, 65)
    with torch.no_grad():
        _, state = model(x)
    reset = torch.tensor([True, False, True, False])
    cleared = AecNet.reset_state(state, reset)
    assert float(cleared['gru'][:, 0].abs().max()) == 0.0
    assert float(cleared['gru'][:, 2].abs().max()) == 0.0
    assert float(cleared['gru'][:, 1].abs().max()) > 0.0
    for before, after in zip(state['enc'], cleared['enc']):
        assert float(after[0].abs().max()) == 0.0
        torch.testing.assert_close(after[1], before[1])


def test_detach_state_breaks_the_graph():
    model = _small_model()
    _, state = model(torch.randn(1, 4, 5, 65))
    assert state['gru'].requires_grad
    detached = AecNet.detach_state(state)
    assert not detached['gru'].requires_grad
    assert all(not t.requires_grad for t in detached['enc'])


# ============================================================
# Spectral compression round trip
# ============================================================

def test_compression_round_trips_and_survives_exact_zero():
    spec = torch.randn(4, 17, 9, dtype=torch.complex64) * 1e-3
    spec[0, 0, 0] = 0.0    # a reference dropout is exactly this
    for c in (0.3, 0.5, 1.0):
        back = expand_spec(compress_spec(spec, c), c)
        torch.testing.assert_close(back, spec, rtol=1e-4, atol=1e-9)

    # ⚠ torch's complex abs() has a NaN gradient at the origin, which genuinely
    # occurs here. safe_mag exists to avoid it; this is the regression test.
    z = torch.zeros(2, 3, 4, dtype=torch.complex64, requires_grad=True)
    compress_spec(z, 0.3).abs().sum().backward()
    assert torch.isfinite(z.grad).all()


# ============================================================
# The zero-reference hard gate
# ============================================================

class _StubModel(nn.Module):
    """Minimal stand-in exposing only what the gate helper touches."""

    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward_spec(self, y_spec, x_spec, state=None):
        return y_spec * self.gain, state


def test_zero_reference_gate_helper():
    y_spec = torch.randn(2, 33, 20, dtype=torch.complex64)

    silent = _StubModel(0.0)
    assert zero_reference_leak_db(silent, y_spec) == -math.inf
    assert assert_zero_reference_gate(silent, y_spec) == -math.inf

    # 1% of the microphone amplitude is -40 dB in power, exactly at the limit.
    borderline = _StubModel(0.01)
    assert abs(zero_reference_leak_db(borderline, y_spec) + 40.0) < 1e-6

    loud = _StubModel(1.0)
    with pytest.raises(AssertionError, match='zero-reference gate FAILED'):
        assert_zero_reference_gate(loud, y_spec, max_leak_db=-40.0)


def test_zero_reference_gate_measures_a_real_model_without_asserting():
    """The gate must be MEASURABLE on an untrained model, just not passing."""
    model = _small_model()
    y_spec = torch.randn(1, 65, 12, dtype=torch.complex64)
    leak = zero_reference_leak_db(model, y_spec)
    assert math.isfinite(leak) or leak == -math.inf


def test_forward_spec_matches_forward():
    model = _small_model()
    y_spec = torch.randn(2, 65, 9, dtype=torch.complex64)
    x_spec = torch.randn(2, 65, 9, dtype=torch.complex64)
    with torch.no_grad():
        d_hat, _ = model.forward_spec(y_spec, x_spec)
        stacked = torch.stack(
            [y_spec.real, y_spec.imag, x_spec.real, x_spec.imag], dim=1
        ).transpose(-2, -1)
        out, _ = model(stacked)
    torch.testing.assert_close(
        d_hat, torch.complex(out[:, 0], out[:, 1]).transpose(-2, -1))


# ============================================================
# Construction-time rejections
# ============================================================

def test_bad_configs_are_rejected_at_construction():
    with pytest.raises(ValueError, match='kernel_f must be odd'):
        AecNet(n_freqs=65, config=AecNetConfig(**{**SMALL.__dict__, 'kernel_f': 4}))
    with pytest.raises(ValueError, match='divisible by'):
        AecNet(n_freqs=65,
               config=AecNetConfig(**{**SMALL.__dict__, 'gru_groups': 3}))
    with pytest.raises(ValueError, match='compress_exponent'):
        AecNet(n_freqs=65,
               config=AecNetConfig(**{**SMALL.__dict__, 'compress_exponent': 0.0}))
    with pytest.raises(ValueError, match='lookahead'):
        AecNet(n_freqs=65,
               config=AecNetConfig(**{**SMALL.__dict__, 'lookahead': -1}))


if __name__ == '__main__':
    raise SystemExit(pytest.main([str(pathlib.Path(__file__)), '-q']))
