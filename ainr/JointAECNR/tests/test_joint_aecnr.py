"""Model-level contracts for JointAECNR.

The four that must never be allowed to rot:
  * the HARD GATE -- a silent reference must leave the microphone alone;
  * the auxiliary heads must be switchable off without breaking the model;
  * causality, including the lookahead the config claims to spend;
  * chunked processing must equal whole-sequence processing.
"""

import os
import sys

import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Each model project has its own top-level train.py/model.py/denoise.py.  Under
# one pytest session the first import wins sys.modules, so a sibling project's
# modules would be exercised instead of these.  Same guard as
# DeepFilterNet2/tests.
for _stale in ('train', 'denoise', 'model', 'postproc'):
    sys.modules.pop(_stale, None)

from model import (  # noqa: E402
    JointAECNR,
    bins_from_hz,
    compress_complex,
    detach_state,
    frames_from_seconds,
    idle_gate_report,
    reference_activity_gate,
    reset_state,
)
from postproc import (  # noqa: E402
    DoubleSuppressionError,
    PostProcessChain,
    apply_safety_attenuation,
    classical_fallback_blend,
    comfort_noise_from_log_psd,
)

sys.path.insert(0, os.path.dirname(ROOT))
from dataset_gen.aec import AecGrid  # noqa: E402


TINY_GRID = AecGrid(sr=16000, n_fft=64, win_len=64, hop_len=32)


def tiny_model(**overrides) -> JointAECNR:
    """Small enough to run in a test, large enough to exercise every path."""
    kwargs = dict(
        grid=TINY_GRID, enc_channels=8, enc_stages=2, rnn_hidden=16,
        rnn_layers=1, lookahead_frames=0, ref_context_frames=4,
        echo_gate_memory_frames=5, df_bins=8, df_order=3,
    )
    kwargs.update(overrides)
    return JointAECNR(**kwargs).eval()


def random_specs(batch=2, frames=24, grid=TINY_GRID, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, grid.n_freqs, frames)
    def draw():
        return torch.complex(torch.randn(shape, generator=generator),
                             torch.randn(shape, generator=generator))
    return draw(), draw()


# ============================================================
# HARD GATE
# ============================================================

def test_zero_reference_makes_the_reference_branch_irrelevant():
    """⚠ HARD GATE. X == 0 must leave the microphone's processing untouched.

    Asserted the strong way: randomise every parameter reachable from x_spec
    and check the output does not move by a single float.  If it moves, some
    module in the reference pathway has acquired a bias or a normalisation
    offset, and the pathway no longer maps a silent reference to zero -- which
    is the whole basis of the idle behaviour, and would otherwise only show up
    as a trained model that quietly damages near-end-only audio.
    """
    model = tiny_model()
    mic, _ = random_specs()
    silent = torch.zeros_like(mic)

    before, _ = model(mic, silent)
    with torch.no_grad():
        for parameter in model.reference_pathway_parameters():
            parameter.normal_(0.0, 3.0)
    after, _ = model(mic, silent)

    assert torch.equal(before.speech_spec, after.speech_spec)


def test_zero_reference_gives_exactly_zero_echo_estimate():
    """The reference-gated echo head is structurally silent on a silent reference."""
    model = tiny_model(aux_echo_head=True, echo_head_ref_gated=True)
    mic, _ = random_specs()
    outputs, _ = model(mic, torch.zeros_like(mic))
    assert outputs.echo_spec is not None
    assert torch.equal(outputs.echo_spec, torch.zeros_like(outputs.echo_spec))
    assert torch.equal(outputs.ref_gate, torch.zeros_like(outputs.ref_gate))


def test_ungated_echo_head_loses_the_structural_guarantee():
    """Pin the cost of echo_head_ref_gated = false, so it reads as a decision."""
    model = tiny_model(echo_head_ref_gated=False)
    mic, _ = random_specs()
    outputs, _ = model(mic, torch.zeros_like(mic))
    assert outputs.echo_spec.abs().sum() > 0


def test_idle_gate_report_shape():
    model = tiny_model()
    mic, _ = random_specs()
    report = idle_gate_report(model, mic)
    assert report['echo_energy_db'] == float('-inf')
    assert report['reference_receptive_field_frames'] >= model.ref_context_frames
    assert torch.isfinite(torch.tensor(report['mic_delta_db']))


def test_reference_activity_gate_needs_the_whole_window_to_go_silent():
    """⚠ The gate must NOT drop the instant X does.

    The echo of the last sample played is still arriving one bulk delay later
    and decays over the room's RT60.  A gate that closed immediately would
    structurally forbid cancelling that tail.
    """
    memory = 5
    reference = torch.zeros(1, 9, 12)
    reference[0, :, 0] = 1.0            # one loud frame, then silence
    gate, _ = reference_activity_gate(reference, memory, floor_power=1e-7)
    assert gate[0, 0] > 0.99
    assert (gate[0, 1:memory] > 0.99).all(), "gate closed before the tail could decay"
    assert (gate[0, memory:] == 0).all(), "gate never closed"


def test_reference_activity_gate_chunked_matches_whole():
    reference = torch.rand(2, 9, 20)
    reference[:, :, 7:13] = 0.0
    whole, _ = reference_activity_gate(reference, 5, 1e-7)
    first, state = reference_activity_gate(reference[..., :11], 5, 1e-7)
    second, _ = reference_activity_gate(reference[..., 11:], 5, 1e-7, state)
    assert torch.allclose(whole, torch.cat([first, second], dim=-1))


# ============================================================
# Auxiliary heads
# ============================================================

@pytest.mark.parametrize('echo', [True, False])
@pytest.mark.parametrize('psd', [True, False])
@pytest.mark.parametrize('deep_filter', [True, False])
def test_heads_and_deep_filter_can_be_switched_off(echo, psd, deep_filter):
    model = tiny_model(aux_echo_head=echo, aux_noise_psd_head=psd,
                       use_deep_filter=deep_filter)
    mic, ref = random_specs()
    outputs, state = model(mic, ref)
    assert outputs.speech_spec.shape == mic.shape
    assert (outputs.echo_spec is not None) == echo
    assert (outputs.noise_log_psd is not None) == psd
    assert ('df' in state) == deep_filter
    if psd:
        assert outputs.noise_log_psd.shape == mic.shape


def test_switching_a_head_off_removes_its_parameters():
    """A head that is off must not be a dead weight in the checkpoint."""
    full = sum(p.numel() for p in tiny_model().parameters())
    bare = sum(p.numel() for p in tiny_model(
        aux_echo_head=False, aux_noise_psd_head=False,
        use_deep_filter=False).parameters())
    assert bare < full


# ============================================================
# Causality
# ============================================================

@pytest.mark.parametrize('lookahead', [0, 2])
def test_output_does_not_depend_on_the_future(lookahead):
    """Output frame t may use input up to t + lookahead, and no further."""
    model = tiny_model(lookahead_frames=lookahead)
    mic, ref = random_specs(batch=1, frames=20)
    baseline, _ = model(mic, ref)

    perturbed = mic.clone()
    perturbed[..., 14] += 50.0
    changed, _ = model(perturbed, ref)

    safe = 14 - lookahead
    assert torch.allclose(baseline.speech_spec[..., :safe],
                          changed.speech_spec[..., :safe], atol=1e-6)
    assert not torch.allclose(baseline.speech_spec[..., 14],
                              changed.speech_spec[..., 14])


def test_reference_future_does_not_leak_either():
    model = tiny_model()
    mic, ref = random_specs(batch=1, frames=20)
    baseline, _ = model(mic, ref)
    perturbed = ref.clone()
    perturbed[..., 12] += 50.0
    changed, _ = model(mic, perturbed)
    assert torch.allclose(baseline.speech_spec[..., :12],
                          changed.speech_spec[..., :12], atol=1e-6)


# ============================================================
# Streaming state
# ============================================================

def test_chunked_processing_equals_whole_sequence():
    """⚠ The state carry must be real, not decorative.

    If this fails, every chunk boundary is a cold start and the trainer's claim
    to walk a 20-60 s sequence with the adaptation intact is false -- which
    would hide exactly the behaviours the long sequences exist to expose
    (convergence, echo-path-change recovery, long-term drift).
    """
    model = tiny_model()
    mic, ref = random_specs(batch=2, frames=30)
    whole, _ = model(mic, ref)

    state = None
    pieces = []
    for start in range(0, 30, 7):
        outputs, state = model(mic[..., start:start + 7],
                               ref[..., start:start + 7], state)
        pieces.append(outputs.speech_spec)
    chunked = torch.cat(pieces, dim=-1)
    assert torch.allclose(whole.speech_spec, chunked, atol=1e-5)


def test_init_state_keys_match_what_forward_returns():
    model = tiny_model()
    mic, ref = random_specs()
    _, produced = model(mic, ref)
    assert set(produced) == set(model.init_state(mic.shape[0]))


def test_reset_state_zeroes_only_the_flagged_lanes():
    model = tiny_model()
    mic, ref = random_specs(batch=3)
    _, state = model(mic, ref)
    reset = torch.tensor([True, False, True])
    cleared = reset_state(detach_state(state), reset)
    for key, value in cleared.items():
        lane_axis = 1 if key == 'rnn' else 0
        for lane, should_reset in enumerate(reset.tolist()):
            slab = value.select(lane_axis, lane)
            if should_reset:
                assert torch.equal(slab, torch.zeros_like(slab)), key
            else:
                assert torch.equal(slab, state[key].select(lane_axis, lane)), key


# ============================================================
# Grid independence
# ============================================================

def test_durations_convert_per_grid_not_per_frame_count():
    """The same seconds must buy different frame counts on different grids."""
    grid_16k = AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256)
    grid_48k = AecGrid(sr=48000, n_fft=1024, win_len=1024, hop_len=512)
    assert frames_from_seconds(0.25, grid_16k.frame_rate) == 16
    assert frames_from_seconds(0.25, grid_48k.frame_rate) == 23
    # ...and the same Hz must buy a band of the same width in Hz.
    for grid in (grid_16k, grid_48k):
        bins = bins_from_hz(1500.0, grid)
        assert abs((bins - 1) * grid.sr / grid.n_fft - 1500.0) < grid.sr / grid.n_fft


def test_model_builds_on_the_48k_grid():
    grid = AecGrid(sr=48000, n_fft=1024, win_len=1024, hop_len=512)
    model = tiny_model(grid=grid, enc_stages=4, df_bins=32)
    mic, ref = random_specs(batch=1, frames=10, grid=grid)
    outputs, _ = model(mic, ref)
    assert outputs.speech_spec.shape == mic.shape


def test_compress_complex_maps_zero_to_zero():
    """⚠ The property the whole zero-reference guarantee rests on."""
    spec = torch.zeros(1, 5, 3, dtype=torch.complex64)
    assert torch.equal(compress_complex(spec, 0.5),
                       torch.zeros(1, 2, 3, 5))


def test_mask_bound_leaves_identity_reachable():
    model = tiny_model(mask_max=2.0)
    mic, ref = random_specs()
    outputs, _ = model(mic, ref)
    assert outputs.mask.abs().max() <= 2.0 + 1e-6


# ============================================================
# Post-processing policy
# ============================================================

def test_second_suppression_stage_is_refused_at_runtime():
    chain = PostProcessChain()
    chain.add('safety', lambda spec, **_: spec)
    with pytest.raises(DoubleSuppressionError, match='pumping'):
        chain.add('classical_nr', lambda spec, **_: spec, suppresses=True)


def test_double_suppression_can_be_opted_into_explicitly():
    chain = PostProcessChain(allow_double_suppression=True)
    chain.add('classical_nr', lambda spec, **_: spec, suppresses=True)
    assert len(chain.stages) == 1


def test_safety_attenuation_limits_both_directions():
    mic = torch.full((1, 4, 3), 1.0, dtype=torch.complex64)
    speech = mic * torch.tensor(1e-4)          # 80 dB of attenuation
    limited = apply_safety_attenuation(speech, mic, max_attenuation_db=20.0)
    assert torch.allclose(limited.abs(), torch.full_like(limited.abs(), 0.1),
                          atol=1e-6)
    boosted = apply_safety_attenuation(mic * 4.0, mic, max_gain_db=0.0)
    assert torch.allclose(boosted.abs(), torch.ones_like(boosted.abs()),
                          atol=1e-6)


def test_comfort_noise_only_fills_bins_the_model_emptied():
    mic = torch.ones(1, 4, 2, dtype=torch.complex64)
    speech = mic.clone()
    speech[:, :2] *= 1e-3                       # only these were emptied
    log_psd = torch.full((1, 4, 2), -2.0)
    filled = comfort_noise_from_log_psd(speech, mic, log_psd,
                                        attenuation_threshold_db=6.0)
    assert torch.equal(filled[:, 2:], speech[:, 2:])
    assert not torch.equal(filled[:, :2], speech[:, :2])


def test_classical_fallback_replaces_rather_than_chains():
    joint = torch.ones(1, 2, 2, dtype=torch.complex64)
    classical = torch.full((1, 2, 2), 3.0, dtype=torch.complex64)
    weight = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]).reshape(1, 2, 2)
    blended = classical_fallback_blend(joint, classical, weight)
    assert torch.allclose(blended.real, torch.tensor([[[1.0, 3.0], [3.0, 1.0]]]))
