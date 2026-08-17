"""Frame-by-frame streaming equivalence for Align-ULCNet.

The offline forward is the reference; ``forward_stream`` must replay it one
STFT frame at a time.  Tolerances below were MEASURED on this fixture (worst
over three input pairs: enhanced 4.1e-8, mask 1.2e-7, magnitude_mask 6.0e-8,
delay 4.0e-7 -- pure GEMM-batching noise from running the GRUs stepwise) and
pinned with ~10x headroom.
"""

import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.aiaec_common import SignalGrid

GRID = SignalGrid(16000, 512, 512, 256)
# Longer than max_delay_frames (64 on this grid) so both delay rings wrap.
T = 96

ENHANCED_TOL = 5e-7
MASK_TOL = 2e-6
MAGMASK_TOL = 1e-6
DELAY_TOL = 5e-6


def _spec(batch, frames, bins, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(batch, frames, bins, generator=g),
                         torch.randn(batch, frames, bins, generator=g))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = AlignULCNet(GRID).eval()
    # At default init the softmax over 64 delay slots is near-uniform and the
    # aligned feature is almost invariant to a one-frame far shift, which
    # would make the can-fail test below vacuous.  Widening only the
    # alignment pathway makes the delay head timing-sensitive (a one-frame
    # far shift moves delay_distribution by ~0.18) without touching the
    # offline/streaming contract under test.
    with torch.no_grad():
        for module in (model.align.query, model.align.key,
                       model.align.score.conv):
            for parameter in module.parameters():
                parameter.mul_(4.0)
    return model


def _stream(model, error, far, state=None):
    if state is None:
        state = model.create_stream_state()
    enhanced, mask, magmask, delay = [], [], [], []
    with torch.no_grad():
        for t in range(error.shape[1]):
            out = model.forward_stream(error[:, t:t + 1], far[:, t:t + 1],
                                       state)
            enhanced.append(out.enhanced)
            mask.append(out.mask)
            magmask.append(out.auxiliary["magnitude_mask"])
            delay.append(out.delay_distribution)
    return (torch.cat(enhanced, 1), torch.cat(mask, 1),
            torch.cat(magmask, 1), torch.cat(delay, 1))


def _offline(model, error, far):
    with torch.no_grad():
        return model(linear_error=error, far_end=far)


def test_stream_matches_offline(model):
    assert model.stream_output_delay == 0
    error = _spec(1, T, GRID.n_freqs, seed=1)
    far = _spec(1, T, GRID.n_freqs, seed=2)
    ref = _offline(model, error, far)
    enhanced, mask, magmask, delay = _stream(model, error, far)
    assert (enhanced - ref.enhanced).abs().max() <= ENHANCED_TOL
    assert (mask - ref.mask).abs().max() <= MASK_TOL
    assert (magmask - ref.auxiliary["magnitude_mask"]).abs().max() <= MAGMASK_TOL
    assert (delay - ref.delay_distribution).abs().max() <= DELAY_TOL


def test_shifted_far_is_detected(model):
    # CAN-FAIL check: stream a far reference delayed by one frame against the
    # unshifted offline run.  The delay head is the output that encodes far
    # timing; at this fixture's init the one-frame shift moves it by ~0.18.
    # (enhanced moves only ~1e-6 here: the near-uniform delay softmax averages
    # a one-frame far shift out of the aligned feature at random init.)
    error = _spec(1, T, GRID.n_freqs, seed=3)
    far = _spec(1, T, GRID.n_freqs, seed=4)
    ref = _offline(model, error, far)
    shifted = torch.cat((torch.zeros_like(far[:, :1]), far[:, :-1]), dim=1)
    _, _, _, delay = _stream(model, error, shifted)
    assert (delay - ref.delay_distribution).abs().max() > 1e-3


def test_fresh_states_do_not_cross_contaminate(model):
    error_a = _spec(1, T, GRID.n_freqs, seed=5)
    far_a = _spec(1, T, GRID.n_freqs, seed=6)
    error_b = _spec(1, T, GRID.n_freqs, seed=7)
    far_b = _spec(1, T, GRID.n_freqs, seed=8)
    ref_a = _offline(model, error_a, far_a)
    ref_b = _offline(model, error_b, far_b)

    # Interleave the two utterances frame-by-frame through two fresh states;
    # each must still reproduce its own offline run.
    state_a = model.create_stream_state()
    state_b = model.create_stream_state()
    enhanced_a, enhanced_b = [], []
    with torch.no_grad():
        for t in range(T):
            enhanced_a.append(model.forward_stream(
                error_a[:, t:t + 1], far_a[:, t:t + 1], state_a).enhanced)
            enhanced_b.append(model.forward_stream(
                error_b[:, t:t + 1], far_b[:, t:t + 1], state_b).enhanced)
    enhanced_a = torch.cat(enhanced_a, 1)
    enhanced_b = torch.cat(enhanced_b, 1)
    assert (enhanced_a - ref_a.enhanced).abs().max() <= ENHANCED_TOL
    assert (enhanced_b - ref_b.enhanced).abs().max() <= ENHANCED_TOL


def test_stream_state_requires_eval_mode(model):
    model.train()
    try:
        with pytest.raises(RuntimeError):
            model.create_stream_state()
    finally:
        model.eval()


def test_state_dict_is_d_agnostic_but_output_is_not(model):
    # Zero-shot D transfer: the alignment depth D never enters weight shapes,
    # so a D=64 checkpoint must strict-load into a D=8 model -- and the two
    # models must NOT be numerically identical (the softmax support and the
    # causal delay stack change).  Measured on this fixture: enhanced max-abs
    # difference 8.9e-6; the bound below leaves ~9x headroom while staying
    # far above the streaming GEMM noise (4.1e-8) pinned in the header.
    assert model.max_delay_frames == 64
    small = AlignULCNet(GRID, max_delay_frames=8).eval()
    small.load_state_dict(model.state_dict(), strict=True)
    error = _spec(1, T, GRID.n_freqs, seed=9)
    far = _spec(1, T, GRID.n_freqs, seed=10)
    ref = _offline(model, error, far)
    out = _offline(small, error, far)
    assert out.delay_distribution.shape[-1] == 8
    assert torch.isfinite(out.enhanced.real).all()
    assert torch.isfinite(out.enhanced.imag).all()
    assert (out.enhanced - ref.enhanced).abs().max() > 1e-6


def test_denoise_stream_path_equals_direct_forward_stream(model):
    # The denoise.py --stream branch calls stream_forward_spec; driving
    # forward_stream directly over the same frames must give the SAME tensor
    # (bit-identical by construction -- same ops in the same order).
    from AIAEC.Align_ULCNet.denoise import stream_forward_spec

    error = _spec(1, T, GRID.n_freqs, seed=11)
    far = _spec(1, T, GRID.n_freqs, seed=12)
    via_cli_helper = stream_forward_spec(model, error, far)
    state = model.create_stream_state()
    direct = []
    with torch.no_grad():
        for t in range(T):
            direct.append(model.forward_stream(
                error[:, t:t + 1], far[:, t:t + 1], state).enhanced)
    direct = torch.cat(direct, dim=1)
    assert torch.equal(via_cli_helper, direct)


def _write_synthetic_checkpoint(path, model):
    """A minimal checkpoint that satisfies load_model's contract checks."""
    from AIAEC.dataset_gen import make_linear_aec_contract

    linear = make_linear_aec_contract(GRID.sample_rate)
    contract = {
        'model_name': 'Align_ULCNet',
        'task': AlignULCNet.task,
        'sr': GRID.sample_rate, 'n_fft': GRID.n_fft,
        'win_len': GRID.win_len, 'hop_len': GRID.hop_len,
        'loss_version': 'test',
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        # None = the contract's own grid-derived depth (64 on this grid),
        # exactly what a config-driven trainer records for the default.
        'ctor_max_delay_frames': None,
    }
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)


def test_load_model_far_input_mode_default_present_and_rejected(
        model, tmp_path, capsys):
    from AIAEC.Align_ULCNet.denoise import load_model

    path = str(tmp_path / 'ckpt.pth')
    _write_synthetic_checkpoint(path, model)

    # _write_synthetic_checkpoint records no far_input_mode -- exactly a
    # legacy checkpoint. It must load, defaulted to raw_far, and the loader
    # must name BOTH sides: the mode the checkpoint trained on and the
    # aligned-far seam deployment feeds it (streaming.py shares this loader
    # and the same seam, so both CLIs print the same line).
    load_model(path, 'cpu')
    assert ('checkpoint training far_input_mode: raw_far; deployment: aligned_far'
            in capsys.readouterr().out)

    # Field present (what every new contract records): loads identically.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    ckpt['contract']['far_input_mode'] = 'raw_far'
    explicit_path = str(tmp_path / 'ckpt_explicit.pth')
    torch.save(ckpt, explicit_path)
    load_model(explicit_path, 'cpu')
    assert ('checkpoint training far_input_mode: raw_far; deployment: aligned_far'
            in capsys.readouterr().out)

    # Unknown mode: rejected before any weights load.
    ckpt['contract']['far_input_mode'] = 'aligned_far'
    unknown_path = str(tmp_path / 'ckpt_unknown.pth')
    torch.save(ckpt, unknown_path)
    with pytest.raises(ValueError, match='far_input_mode'):
        load_model(unknown_path, 'cpu')


def test_load_model_max_delay_override(model, tmp_path, capsys):
    from AIAEC.Align_ULCNet.denoise import load_model

    path = str(tmp_path / 'ckpt.pth')
    _write_synthetic_checkpoint(path, model)

    # Without the flag the contract rules: D stays 64.
    loaded, _, _ = load_model(path, 'cpu')
    assert loaded.max_delay_frames == 64
    assert 'deployment override' not in capsys.readouterr().out

    # Differing override: rebuilt at D=8, unmissable line printed, weights
    # still strict-loaded from the D=64 checkpoint.
    loaded, _, _ = load_model(path, 'cpu', max_delay_frames=8)
    assert loaded.max_delay_frames == 8
    printed = capsys.readouterr().out
    assert ('deployment override: max_delay_frames 64 -> 8 (weights are '
            'D-agnostic; output is NOT numerically identical across D)'
            ) in printed
    assert torch.equal(loaded.align.query.weight, model.align.query.weight)

    # Override equal to the contract value: nothing to do, no override line.
    loaded, _, _ = load_model(path, 'cpu', max_delay_frames=64)
    assert loaded.max_delay_frames == 64
    assert 'deployment override' not in capsys.readouterr().out


def _write_delayed_scene(tmp_path, samples=32768, delay=1024, seed=23):
    """A mic/far pair whose echo sits at a known bulk delay, on disk."""
    import soundfile as sf

    generator = torch.Generator().manual_seed(seed)
    far = torch.randn(samples, generator=generator) * 0.05
    mic = torch.zeros(samples)
    mic[delay:] = 0.5 * far[:samples - delay]
    mic += torch.randn(samples, generator=generator) * 0.005
    mic_path = str(tmp_path / 'mic.wav')
    far_path = str(tmp_path / 'far.wav')
    sf.write(mic_path, mic.numpy(), GRID.sample_rate, subtype='FLOAT')
    sf.write(far_path, far.numpy(), GRID.sample_rate, subtype='FLOAT')
    return mic_path, far_path, far.unsqueeze(0)


def test_denoise_and_streaming_feed_the_model_the_same_aligned_far(
        model, tmp_path, monkeypatch):
    """Both CLIs must build the model's far branch from the SAME seam.

    Production feeds the model the far hop the linear AEC actually consumed
    (raw until the alignment ring can serve the applied delay, ring-aligned
    afterwards), so an offline CLI that fed the raw far WAV instead would be
    evaluating a model on an input distribution deployment never produces.

    The far each CLI hands to its own STFT is captured at that boundary --
    denoise.py through its module-level ``stft``, streaming.py through the
    chunks pushed into its far ``StreamSTFT`` -- and compared against the
    PBFDKF-consumed far recomputed INDEPENDENTLY here from the same contract.
    The raw-far inequality is what keeps this from passing vacuously.
    """
    from AIAEC.Align_ULCNet import denoise as denoise_cli
    from AIAEC.Align_ULCNet import streaming as streaming_cli
    from AIAEC.aiaec_streaming import StreamSTFT
    from AIAEC.dataset_gen import make_linear_aec_contract
    from AIAEC.inference_common import load_mic_far
    from AIAEC.training_common import LinearAecEngine

    checkpoint = str(tmp_path / 'ckpt.pth')
    _write_synthetic_checkpoint(checkpoint, model)
    mic_path, far_path, raw_far = _write_delayed_scene(tmp_path)

    # --- denoise.py: record the tensors handed to the offline STFT. The CLI
    # transforms the error first and the far second, and calls stft exactly
    # twice, so the far branch is call 1.
    stft_inputs = []
    real_stft = denoise_cli.stft

    def recording_stft(signal, grid, *args, **kwargs):
        stft_inputs.append(signal.detach().clone())
        return real_stft(signal, grid, *args, **kwargs)

    monkeypatch.setattr(denoise_cli, 'stft', recording_stft)
    denoise_cli.main(denoise_cli.build_parser().parse_args([
        checkpoint, mic_path, far_path, str(tmp_path / 'denoise_out.wav'),
        '--device', 'cpu',
    ]))
    assert len(stft_inputs) == 2
    denoise_far = stft_inputs[1]

    # --- streaming.py: record every chunk pushed into the far StreamSTFT.
    # The CLI constructs the error transform first and the far one second.
    pushed = {}
    constructed = []

    class RecordingStreamSTFT(StreamSTFT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = len(constructed)
            constructed.append(self)
            pushed[self.index] = []

        def push(self, chunk):
            pushed[self.index].append(chunk.detach().clone())
            return super().push(chunk)

    monkeypatch.setattr(streaming_cli, 'StreamSTFT', RecordingStreamSTFT)
    streaming_cli.main(streaming_cli.build_parser().parse_args([
        checkpoint, mic_path, far_path, str(tmp_path / 'stream_out.wav'),
        '--device', 'cpu',
    ]))
    assert len(constructed) == 2
    streaming_far = torch.cat(pushed[1], dim=-1)

    # --- the independent reference: the same frozen engine, same contract,
    # run here over the same audio.
    contract = make_linear_aec_contract(GRID.sample_rate)
    mic_t, far_t, _rates = load_mic_far(mic_path, far_path, GRID.sample_rate)
    engine = LinearAecEngine(n_lanes=1, sample_rate=GRID.sample_rate,
                             contract=contract.as_dict())
    engine(mic_t, far_t, GRID.sample_rate)
    consumed_far = engine.get_aligned_far()

    assert streaming_far.shape == consumed_far.shape
    assert torch.equal(streaming_far, consumed_far)
    assert torch.equal(denoise_far, consumed_far)
    assert torch.equal(streaming_far, denoise_far)

    # Non-vacuous: the scene really was delayed, so the seam really did move
    # the far. A CLI still feeding the raw WAV would satisfy every equality
    # above only if this failed.
    assert not torch.equal(consumed_far, raw_far)
