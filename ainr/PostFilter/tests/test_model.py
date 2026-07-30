"""PostFilter model, feature and front-end invariants.

The properties here are the ones a consumer cannot notice being broken: a gain
that leaves [0,1] is clipped by the caller and looks like a tuning problem, a
feature set that is not scale-invariant looks like a model that merely
generalises badly to a new front-end, and state that does not carry across
chunks looks like slow convergence.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AINR = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, AINR)

# Each model project has its own top-level train.py/model.py/denoise.py.  Under
# one pytest session the first import wins sys.modules, so a sibling project's
# module would silently be exercised here instead.
for _stale in ('train', 'denoise', 'model', 'frontends', 'postproc'):
    sys.modules.pop(_stale, None)

from dataset_gen.aec import AecGrid  # noqa: E402
from frontends import (  # noqa: E402
    NullFrontEnd,
    OracleFrontEnd,
    StftNlmsFrontEnd,
)
from model import (  # noqa: E402
    PostFilterNet,
    build_band_matrices,
    compute_erb_matrix,
    erb_bandborder,
    mask_magnitude,
)


GRID = AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256)


def make_model(**kwargs):
    """A small but structurally complete model.  eval() everywhere: BatchNorm in
    train mode normalises over the batch, which would make the split-vs-whole
    comparisons below fail for a reason that has nothing to do with state."""
    params = dict(n_bands=32, enc_channels=8, gru_hidden=24, gru_layers=1,
                  dec_hidden=16)
    params.update(kwargs)
    model = PostFilterNet(GRID, **params)
    return model.eval()


def synth_inputs(batch=2, frames=24, seed=0, scale=1.0):
    """(E, D_hat, X) with a plausible structure: E holds the near end plus a
    residual correlated with D_hat, which is itself a filtered X."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, GRID.n_freqs, frames)

    def noise():
        return torch.complex(torch.randn(shape, generator=generator),
                             torch.randn(shape, generator=generator))

    x = noise()
    d_hat = 0.6 * x + 0.1 * noise()
    residual = 0.15 * d_hat
    near = noise() * torch.linspace(1.0, 0.2, GRID.n_freqs).view(1, -1, 1)
    e = near + residual + 0.05 * noise()
    return (e * scale).to(torch.complex64), (d_hat * scale).to(torch.complex64), \
        (x * scale).to(torch.complex64)


# ============================================================
# Gain range
# ============================================================

@pytest.mark.parametrize('resolution,output_type', [
    ('band', 'gain'), ('full', 'gain'), ('full', 'complex'),
])
def test_gain_is_finite_and_in_unit_interval(resolution, output_type):
    model = make_model(mask_resolution=resolution, output_type=output_type)
    e, d, x = synth_inputs()
    with torch.no_grad():
        mask, _ = model(e, d, x)
    magnitude = mask_magnitude(mask)
    assert torch.isfinite(magnitude).all()
    assert magnitude.min() >= 0.0 and magnitude.max() <= 1.0
    # And it survives expansion to bins: the ERB synthesis matrix is a partition
    # of unity, so the expansion is a convex combination.
    bins = mask_magnitude(model.expand_to_bins(mask))
    assert bins.min() >= 0.0 and bins.max() <= 1.0 + 1e-6


def test_band_expansion_is_a_partition_of_unity():
    """gain == 1 in every band must expand to exactly 1 in every bin.

    Anything else means the model cannot express "leave this alone", and the
    residual ripple would be a fixed colouration baked into every output.
    """
    model = make_model()
    ones = torch.ones(1, model.n_out, 5)
    expanded = model.expand_to_bins(ones)
    assert torch.allclose(expanded, torch.ones_like(expanded), atol=1e-6)


def test_erb_bands_are_all_non_empty():
    borders = erb_bandborder(32, GRID.sr, GRID.n_fft)
    widths = borders[1:] - borders[:-1]
    assert widths.min() >= 2, f"degenerate ERB band(s): {widths.tolist()}"
    matrix = compute_erb_matrix(borders, GRID.n_fft, mode=1)
    assert torch.allclose(torch.from_numpy(matrix).sum(dim=1),
                          torch.ones(GRID.n_freqs), atol=1e-6)


def test_erb_filterbank_matches_the_sibling_projects():
    """⚠ model.py copies DeepFilterNet2's ERB construction rather than importing
    it (see the comment there).  Guard the copy instead of trusting it."""
    ours = build_band_matrices(GRID, 32, 'band')

    dfn2 = os.path.join(AINR, 'DeepFilterNet2')
    sys.path.insert(0, dfn2)
    sys.modules.pop('model', None)
    try:
        import model as dfn2_model    # noqa: N813  (the sibling's model.py)
        reference = dfn2_model._build_erb_fb(GRID.n_fft, GRID.sr, 32)
    finally:
        # Put sys.modules['model'] back to OURS, or every later test in this
        # session that imports it lazily gets the sibling project's file.
        sys.path.remove(dfn2)
        sys.modules.pop('model', None)
        sys.path.insert(0, ROOT)
        import model    # noqa: F401

    for mine, theirs, name in zip(ours, reference, ('analysis', 'synthesis')):
        assert torch.equal(mine, theirs), (
            f"{name} ERB matrix has drifted from DeepFilterNet2's; the two are "
            f"meant to be the same filterbank")


# ============================================================
# Scale invariance -- the load-bearing property
# ============================================================

@pytest.mark.parametrize('scale', [1e-3, 0.05, 0.25, 4.0, 20.0, 1e3])
def test_ratio_only_mode_is_invariant_to_a_common_scaling_of_e_and_d_hat(scale):
    """Scaling (E, D_hat) by one constant must not move the predicted gain
    -- IN THE PURE-RATIO MODE (``include_absolute_level=False``).

    ⚠ THIS IS NOT THE SHIPPED DEFAULT, and that is deliberate.  Scale-invariant
    ratio INPUTS are this project's hypothesis, not established practice: a
    survey of twelve published residual-echo suppressors found the dominant
    convention is raw compressed/log spectra at a fixed absolute scale, and the
    one shipped production neural residual-echo estimator hard-codes a 1/32768
    constant with the comment "Trained model expects [-1,1]-scaled signals".
    So the default carries an absolute channel ALONGSIDE the ratios, and this
    test pins the property for the variant that drops it -- which is the
    experiment worth running, since nobody has published it.

    WHY THE TOLERANCE IS 1e-4 AND NOT 0
    Every channel is a ratio taken after E, D_hat and Y are divided by
    ``sqrt(mean(|E|^2 + |D_hat|^2))`` of the same frame, so in exact arithmetic
    the features are IDENTICAL under a common scaling.  What is left is
      * float32 round-off through the band matmul, the log, the EMA recursion,
        the convolutions and the GRU;
      * the eps floors (1e-10 on a normalised band power, 1e-20 on the frame
        reference), which sit ~8 orders of magnitude below the test signal and
        bite only on digital silence.
    Measured over these six decades of scaling the worst deviation is 6e-8 --
    one float32 ULP at a gain near 0.5 -- so 1e-4 leaves three orders of
    magnitude of headroom.  A failure here is an absolute level leaking into a
    feature, not numerical noise.
    """
    model = make_model(include_absolute_level=False)
    e, d, x = synth_inputs(seed=3)
    with torch.no_grad():
        base, _ = model(e, d, x)
        # X is scaled too: in the real system a front-end gain change moves the
        # whole chain.  X is normalised by its OWN frame energy, so it is
        # separately invariant.
        scaled, _ = model(e * scale, d * scale, x * scale)
    deviation = (mask_magnitude(base) - mask_magnitude(scaled)).abs().max().item()
    assert deviation < 1e-4, f"gain moved by {deviation:.2e} under x{scale}"


def test_shipped_config_carries_the_absolute_channel_and_the_mic():
    """Pins the two input decisions that came from the literature, not from us.

    ``include_absolute_level``: ratios ALONGSIDE an absolute channel, because
    ratio-only inputs are unpublished and the shipped production estimator does
    the opposite.  Flipping this back to false is a legitimate experiment, but
    it must be a deliberate one -- hence the pin.

    ``use_mic``: the one side-input both camps in the literature agree on.
    Franzen & Fingscheidt (ICASSP 2022) measure Y added to E lifting noise-only
    dSNR from 14.74 to 22.38 dB, attributing it to Y being unprocessed by the
    AEC.  A ratio against Y does not substitute for it: a ratio keeps the
    relationship and discards the shape.
    """
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    assert cfg.getboolean('feature', 'include_absolute_level') is True
    assert cfg.getboolean('feature', 'use_mic') is True


def test_mic_channel_changes_the_feature_count_and_the_output():
    """The mic channel must be real, not a config key nobody reads."""
    with_mic = make_model(use_mic=True)
    without = make_model(use_mic=False)
    assert with_mic.features.n_channels == without.features.n_channels + 1

    e, d, x = synth_inputs(seed=11)
    with torch.no_grad():
        a, _ = with_mic.features(e, d, x)
        b, _ = without.features(e, d, x)
    assert a.shape[1] == b.shape[1] + 1


def test_absolute_level_channel_really_does_break_invariance():
    """Prove the ⚠ above is a real hazard and not a superstition."""
    model = make_model(include_absolute_level=True)
    e, d, x = synth_inputs(seed=4)
    with torch.no_grad():
        feats_a, _ = model.features(e, d, x)
        feats_b, _ = model.features(e * 10.0, d * 10.0, x * 10.0)
    assert (feats_a - feats_b).abs().max() > 0.1


# ============================================================
# Echo-free operation
# ============================================================

def test_model_still_works_with_a_zero_echo_estimate():
    """D_hat == 0 happens for real: reference dropout, the 'none' front-end, and
    any frame before the canceller has converged.  The model must stay a
    denoiser there rather than depending on echo being present.

    This asserts the mechanism (finite features, a non-degenerate gain, gradient
    reaching every parameter), not the quality -- an untrained model cannot be
    asked to denoise well.
    """
    model = make_model()
    model.train(False)
    e, _, x = synth_inputs(seed=7)
    zero = torch.zeros_like(e)

    feats, _ = model.features(e, zero, x)
    assert torch.isfinite(feats).all()
    # The echo-to-output ratio channel saturates at the documented clamp rather
    # than going to -inf.
    ratio = feats[:, 1]
    assert torch.allclose(ratio, torch.full_like(ratio, -6.0 / 3.0), atol=1e-5)

    mask, _ = model(e, zero, x)
    assert torch.isfinite(mask).all()
    # Not a constant: the gain still responds to the spectral content.
    assert mask.std().item() > 1e-4

    # And it is trainable in this regime: every parameter gets a gradient.
    model.zero_grad()
    mask, _ = model(e, zero, x)
    (mask.mean()).backward()
    missing = [name for name, param in model.named_parameters()
               if param.grad is None or not torch.isfinite(param.grad).all()]
    assert not missing, f"no usable gradient for {missing} when D_hat == 0"


def test_gain_responds_to_the_echo_estimate():
    """The converse of the test above: changing D_hat with everything else fixed
    must change what the model sees, or this is a noise suppressor wearing the
    wrong name.

    The FEATURE assertion is the strong one.  The gain assertion is deliberately
    weak: at random initialisation the sigmoid sits near 0.5 and is nearly flat,
    so an untrained model's output sensitivity says little about the trained
    one.  What it does establish is that the echo channels reach the head at
    all -- and 1e-5 is still ~2 orders of magnitude above the 6e-8 a common
    rescaling produces, so this and the scale-invariance test are not in
    tension.
    """
    model = make_model()
    e, d, x = synth_inputs(seed=11)
    with torch.no_grad():
        feats_off, _ = model.features(e, torch.zeros_like(d), x)
        feats_on, _ = model.features(e, d, x)
        without, _ = model(e, torch.zeros_like(d), x)
        with_echo, _ = model(e, d, x)
    # Channel 1 = log10(|D_hat|^2 / |E|^2); channel 3 = coherence(E, D_hat).
    assert (feats_off[:, 1] - feats_on[:, 1]).abs().max().item() > 0.5
    assert (feats_off[:, 3] - feats_on[:, 3]).abs().max().item() > 0.1
    assert (without - with_echo).abs().max().item() > 1e-5


# ============================================================
# Causality and state
# ============================================================

def test_output_is_causal():
    """With lookahead 0, a change at frame t must not move any output before t."""
    model = make_model(lookahead_frames=0)
    e, d, x = synth_inputs(frames=16, seed=5)
    cut = 9
    e2 = e.clone()
    e2[..., cut:] *= 4.0
    with torch.no_grad():
        a, _ = model(e, d, x)
        b, _ = model(e2, d, x)
    assert torch.allclose(a[..., :cut], b[..., :cut], atol=1e-6)
    assert not torch.allclose(a[..., cut:], b[..., cut:], atol=1e-6)


def test_lookahead_buys_exactly_the_frames_it_claims():
    model = make_model(lookahead_frames=1)
    e, d, x = synth_inputs(frames=16, seed=6)
    cut = 9
    e2 = e.clone()
    e2[..., cut:] *= 4.0
    with torch.no_grad():
        a, _ = model(e, d, x)
        b, _ = model(e2, d, x)
    # One frame of lookahead means frame cut-1 already sees the change.
    assert torch.allclose(a[..., :cut - 1], b[..., :cut - 1], atol=1e-6)
    assert not torch.allclose(a[..., cut - 1], b[..., cut - 1], atol=1e-6)


def test_state_carries_across_chunks():
    """Two consecutive chunks with the state carried must equal one long chunk.

    This is what makes the sequence sampler worth having.  If it fails, the
    model has a hidden per-chunk reset and every convergence measurement over
    the corpus is measuring chunk boundaries.
    """
    model = make_model()
    e, d, x = synth_inputs(frames=20, seed=8)
    half = 11
    with torch.no_grad():
        whole, _ = model(e, d, x)
        first, state = model(e[..., :half], d[..., :half], x[..., :half])
        second, _ = model(e[..., half:], d[..., half:], x[..., half:], state)
    joined = torch.cat([first, second], dim=-1)
    assert torch.allclose(whole, joined, atol=1e-5), (
        f"max deviation {(whole - joined).abs().max().item():.2e}")


def _lane(value, index, key):
    """The GRU state is (layers, batch, hidden); everything else is batch-first."""
    return value[:, index] if key == 'gru' else value[index]


def test_reset_lanes_clears_only_the_flagged_lanes():
    """⚠ Every state a lane owns must be listed in reset_lanes.  A forgotten EMA
    carries the previous sequence's echo path into the next one, which looks
    like a slow-converging model rather than a bug."""
    model = make_model()
    e, d, x = synth_inputs(batch=3, frames=8, seed=9)
    with torch.no_grad():
        _, state = model(e, d, x)
    fresh = model.init_state(3)
    assert set(state) == set(fresh), (
        f"init_state and forward disagree about the state: "
        f"{set(state) ^ set(fresh)}")

    cleared = model.reset_lanes(state, torch.tensor([True, False, True]))
    for key in fresh:
        assert torch.equal(_lane(cleared[key], 0, key), _lane(fresh[key], 0, key)), \
            f"{key} was not cleared on a reset lane"
        assert not torch.equal(_lane(cleared[key], 1, key), _lane(fresh[key], 1, key)), \
            f"{key} was cleared on a lane that did not ask for it"


# ============================================================
# Complexity budget
# ============================================================

@pytest.mark.parametrize('resolution', ['band', 'full'])
def test_default_config_lands_in_the_stated_macs_budget(resolution):
    """config.ini's defaults must actually hit the 100-300 M MACs/s target.

    A config edit that quietly triples the cost is invisible until the target
    platform runs out of cycles, which is much later and much more expensive.
    """
    import configparser
    from model import build_model
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, 'config.ini'))
    cfg.set('model', 'mask_resolution', resolution)
    model = build_model(cfg, GRID)
    macs = model.macs_per_second() / 1e6
    assert 100.0 <= macs <= 300.0, f"{resolution}: {macs:.1f} M MACs/s"


def test_complex_output_requires_full_resolution():
    """⚠ One phase rotation per ERB band is meaningless; the constructor refuses
    it rather than shipping a variant that merely underperforms."""
    with pytest.raises(ValueError, match='phase rotation'):
        make_model(mask_resolution='band', output_type='complex')


# ============================================================
# Front-ends
# ============================================================

def test_null_frontend_is_a_passthrough():
    e, d, x = synth_inputs(seed=12)
    y = e + d
    out_e, out_d, _ = NullFrontEnd().process(y, x)
    assert torch.equal(out_e, y)
    assert torch.count_nonzero(out_d) == 0


def test_oracle_frontend_leaves_no_residual():
    _, d, x = synth_inputs(seed=13)
    y = torch.randn_like(d.real) + 1j * torch.randn_like(d.real) + d
    out_e, out_d, _ = OracleFrontEnd().process(y, x, D=d)
    assert torch.allclose(out_d, d)
    assert torch.allclose(out_e, y - d)


def test_oracle_frontend_refuses_to_guess():
    _, d, x = synth_inputs(seed=13)
    with pytest.raises(ValueError, match='true echo'):
        OracleFrontEnd().process(d, x)


def test_stft_nlms_converges_on_a_stationary_echo_path():
    """The front-end must actually cancel, or every residual in the corpus is
    "the filter never got there" rather than "the filter did its best"."""
    grid = GRID
    frontend = StftNlmsFrontEnd(grid, taps=4, mu=0.5)
    generator = torch.Generator().manual_seed(21)
    frames = 400
    x = torch.complex(torch.randn(1, grid.n_freqs, frames, generator=generator),
                      torch.randn(1, grid.n_freqs, frames, generator=generator))
    # A fixed 2-tap echo path, i.e. exactly representable by the filter.
    path0 = torch.complex(torch.randn(1, grid.n_freqs, 1, generator=generator),
                          torch.randn(1, grid.n_freqs, 1, generator=generator))
    path1 = 0.4 * path0
    echo = path0 * x
    echo[..., 1:] = echo[..., 1:] + path1 * x[..., :-1]
    e, d_hat, _ = frontend.process(echo, x)

    tail = slice(frames // 2, None)
    erle = 10.0 * torch.log10(echo[..., tail].abs().square().mean()
                              / e[..., tail].abs().square().mean())
    assert erle.item() > 20.0, f"only {erle.item():.1f} dB ERLE after convergence"
    assert torch.isfinite(d_hat).all()


def test_stft_nlms_freezes_when_the_reference_is_silent():
    """⚠ A reference dropout must not unlearn the filter: without the activity
    gate, the update is driven by pure near-end speech."""
    grid = GRID
    frontend = StftNlmsFrontEnd(grid, taps=2, mu=0.5)
    generator = torch.Generator().manual_seed(22)
    frames = 200
    x = torch.complex(torch.randn(1, grid.n_freqs, frames, generator=generator),
                      torch.randn(1, grid.n_freqs, frames, generator=generator))
    path = torch.complex(torch.randn(1, grid.n_freqs, 1, generator=generator),
                         torch.randn(1, grid.n_freqs, 1, generator=generator))
    _, _, state = frontend.process(path * x, x)
    converged = state['w'].clone()

    silence = torch.zeros_like(x)
    near = torch.complex(torch.randn(1, grid.n_freqs, frames, generator=generator),
                         torch.randn(1, grid.n_freqs, frames, generator=generator))
    e, d_hat, state = frontend.process(near, silence, state)
    assert torch.allclose(state['w'], converged, atol=1e-6)
    # "ref == 0 implies output == mic", but only once the tap delay line has
    # flushed: for the first `taps` frames the filter is still convolving
    # reference frames from BEFORE the dropout, and the echo of what was already
    # played really is still arriving.  Asserting it from frame 0 would be
    # asserting that echo stops instantly.
    flushed = slice(frontend.taps, None)
    assert torch.count_nonzero(d_hat[..., flushed]) == 0
    assert torch.equal(e[..., flushed], near[..., flushed])


def test_frontend_state_resets_per_lane():
    frontend = StftNlmsFrontEnd(GRID, taps=2)
    state = frontend.init_state(3)
    state['w'] = state['w'] + 1.0
    state['peak'] = state['peak'] + 1.0
    cleared = frontend.reset_lanes(state, torch.tensor([True, False, True]))
    assert torch.count_nonzero(cleared['w'][0]) == 0
    assert torch.count_nonzero(cleared['w'][1]) > 0
    assert cleared['peak'].tolist() == [0.0, 1.0, 0.0]
