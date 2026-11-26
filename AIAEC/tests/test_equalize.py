"""Cross-layer equalization (``AIAEC/_equalize.py``, ``DeepVQE_S/equalize.py``)."""

import copy
import math

import numpy as np
import pytest
import torch

from AIAEC import _equalize as cle
from AIAEC._streaming_export import (
    _fold_batch_norm_into_upsample_conv,
    _fold_three_vector_conv,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.DeepVQE_S.model import DeepVQES

from test_training_options import (
    _distinct_prelu_slopes,
    _export,
    _option_checkpoint,
)

GRID256 = SignalGrid(16000, 256, 256, 128)
# (grid, constructor values): the constructor defaults on each grid, and the
# [model] values of the shipped config_16k_256.ini.
GRIDS = {
    '16k-256': (GRID256, {}),
    '16k-256-shipped': (GRID256, {'similarity_channels': 8,
                                  'gru_hidden': 256, 'score_kernel': (9, 5)}),
    '16k-512': (SignalGrid(16000, 512, 512, 256), {}),
    '48k-1024': (SignalGrid(48000, 1024, 1024, 512), {}),
}


def _uneven_norms(model, seed=5):
    """Trained-like BatchNorms: signed channel gains a few times apart, with
    running statistics, so the folded weights are uneven per channel (a wider
    spread drives the outputs past the exporter's absolute parity bound)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator))
                module.running_var.copy_(
                    torch.rand(n, generator=generator) * 2 + 0.05)
                gain = torch.exp(torch.randn(n, generator=generator) * 0.7)
                sign = torch.randn(n, generator=generator).sign()
                module.weight.copy_(gain * sign)
                module.bias.copy_(torch.randn(n, generator=generator) * 0.3)
    _distinct_prelu_slopes(model)
    return model


def _model(activation, case='16k-256'):
    grid, kwargs = GRIDS[case]
    torch.manual_seed(3)
    return _uneven_norms(DeepVQES(grid, max_delay_seconds=0.5,
                                  activation=activation, **kwargs).eval())


def _equalized(reference):
    model = copy.deepcopy(reference)
    result = cle.equalize(cle.model_groups('DeepVQE_S', model))
    assert result.converged
    return model, result


@pytest.mark.parametrize('grid', sorted(GRIDS))
@pytest.mark.parametrize('activation', ['relu', 'leaky_relu', 'prelu'])
def test_output_unchanged_and_ranges_matched(activation, grid):
    reference = _model(activation, grid)
    model, result = _equalized(reference)
    assert cle.output_difference(reference, model) <= 1e-4
    # Every channel short of the scale bound has its largest written and
    # read weights equal.
    free = math.log(cle.MAX_SCALE) * (1 - 1e-4)
    for group in cle.model_groups('DeepVQE_S', model):
        written, read = group.ranges()
        mismatch = (written / read).log().abs()
        held = result.scales[group.name].log().abs() >= free
        assert float(mismatch[~held].max()) <= 1e-4, group.name
    # A reparameterization: nothing outside the groups moves.
    assert torch.equal(model.gru.weight_hh_l0, reference.gru.weight_hh_l0)
    assert torch.equal(model.align.score.conv.weight,
                       reference.align.score.conv.weight)


def test_an_outlier_channel_stops_costing_per_tensor_weights():
    """The case CLE is for: one channel far above the rest (DFQ, MobileNetV2
    after BatchNorm folding). Without one the change can be either way, which
    is why the CLI prints both granularities before and after."""
    reference = _model('prelu')
    with torch.no_grad():
        reference.mic2.norm.weight[0] *= 50
        reference.up2.norm.weight[1] *= 30
    model, _ = _equalized(reference)
    before = cle.weight_quantization_snr('DeepVQE_S', reference, reference)
    after = cle.weight_quantization_snr('DeepVQE_S', reference, model)
    assert after['per_tensor'] >= before['per_tensor'] + 2.0


def test_near_dead_channel_is_held_at_the_scale_bound():
    """A channel with a tiny gain but a real bias is scaled up whole, bias
    included, so its activation and the state that holds it grow; unbounded
    this one would grow about a thousandfold."""
    reference = _model('prelu')
    with torch.no_grad():
        reference.mic2.norm.weight[5] = 1e-6
        reference.mic2.norm.bias[5] = 3.0
    model, result = _equalized(reference)
    assert result.scales['mic2'][5] == pytest.approx(1 / cle.MAX_SCALE)
    assert cle.at_bound(result) >= 1
    assert cle.output_difference(reference, model) <= 1e-4
    primary, far = cle._random_stream(reference)
    peaks = []
    for net in (reference, model):
        seen = []
        handle = net.mic2.activation.register_forward_hook(
            lambda module, inputs, output: seen.append(output[:, 5].abs().max()))
        with torch.no_grad():
            net(primary, far)
        handle.remove()
        peaks.append(float(seen[0]))
    assert peaks[1] / peaks[0] == pytest.approx(cle.MAX_SCALE, rel=1e-3)
    # The report sees module outputs; this tensor's peak is set by others.
    assert 'mic2.activation' in {
        name for _, name in cle.peak_growth(reference, model)}


@pytest.mark.parametrize('activation', ['relu', 'leaky_relu', 'prelu'])
def test_stream_unchanged(activation):
    reference = _model(activation)
    model, _ = _equalized(reference)
    primary, far = cle._random_stream(reference)
    outputs = []
    for net in (reference, model):
        state = net.create_stream_state()
        with torch.no_grad():
            outputs.append(torch.cat([
                net.forward_stream(primary[:, t:t + 1], far[:, t:t + 1],
                                   state).enhanced
                for t in range(primary.shape[1])], dim=1))
    peak = outputs[0].abs().max()
    assert (outputs[1] - outputs[0]).abs().max() <= 1e-4 * peak


def test_weights_are_the_exported_folds():
    model = _model('prelu')
    weights = {weight.name: weight for weight in cle._deepvqe_weights(model)}
    ccm = model.ccm_up.conv.conv
    assert torch.equal(weights['ccm_up'].get(), _fold_three_vector_conv(
        ccm, ccm.out_channels // 6).weight.detach())
    for name in ('up3', 'up2', 'up1'):
        layer = getattr(model, name)
        torch.testing.assert_close(
            weights[name].get(), _fold_batch_norm_into_upsample_conv(
                layer.conv.conv, layer.norm).weight.detach())
    # A rounded value written back is the value the graph then carries.
    for name in ('ccm_up', 'up2', 'mic3', 'gru.W_ih'):
        value = torch.randn_like(weights[name].get())
        weights[name].put(value)
        torch.testing.assert_close(weights[name].get(), value)
    # The folded CCM rows are not the parameter's rows: no row scale there.
    with pytest.raises(ValueError, match='not the axis'):
        cle.Side(weights['ccm_up'], 0, 2).check(18)


def test_dead_channel_keeps_its_scale():
    reference = _model('prelu')
    with torch.no_grad():
        reference.mic1.norm.weight[3] = 0.0
    model, result = _equalized(reference)
    # Left as it is, not pushed to the bound (its output is the bias alone).
    assert float(result.scales['mic1'][3]) == 1.0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert cle.output_difference(reference, model) <= 1e-4


def test_elu_and_other_models_are_refused():
    with pytest.raises(ValueError, match='relu, leaky_relu or prelu'):
        cle.model_groups('DeepVQE_S', _model('elu'))
    with pytest.raises(ValueError, match='no channel table'):
        cle.model_groups('Align_ULCNet', None)


def _checkpoint(tmp_path, activation):
    path = tmp_path / (activation + '.pth')
    model, contract = _option_checkpoint(
        'DeepVQE_S', {'activation': activation}, path)
    _uneven_norms(model)
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    return path, contract


def _rollout(session, signals):
    """Every frame's outputs of a streaming graph, state fed back."""
    states = [np.zeros(value.shape, np.float32)
              for value in session.get_inputs()[2:]]
    names = [value.name for value in session.get_inputs()]
    outputs = []
    for mic, far in signals:
        result = session.run(None, dict(zip(names, [mic, far] + states)))
        states = result[1:]
        outputs.append(result)
    return outputs


def test_cli_writes_a_checkpoint_that_exports(tmp_path, capsys):
    ort = pytest.importorskip('onnxruntime')
    source, contract = _checkpoint(tmp_path, 'prelu')
    target = tmp_path / 'prelu_cle.pth'
    cle.main('DeepVQE_S', [str(source), '-o', str(target)])
    assert 'one scale per tensor' in capsys.readouterr().out
    saved = torch.load(target, weights_only=False)
    assert set(saved) == {'contract', 'state_dict', 'cross_layer_equalization'}
    assert saved['contract'] == contract
    assert len(saved['cross_layer_equalization']['source_sha256']) == 64

    from AIAEC.DeepVQE_S.inference import load_model
    sessions = []
    for path in (source, target):
        model, _ = load_model(str(path), 'cpu')
        graph = str(path.with_suffix('.onnx'))
        _export('DeepVQE_S', model, path, graph, 1, verify=True)
        sessions.append(ort.InferenceSession(
            graph, providers=['CPUExecutionProvider']))
    rng = np.random.default_rng(0)
    shapes = [value.shape for value in sessions[0].get_inputs()[:2]]
    signals = [[rng.standard_normal(shape).astype(np.float32)
                for shape in shapes] for _ in range(12)]
    original, equalized = (_rollout(session, signals) for session in sessions)
    peak = max(float(np.abs(frame[0]).max()) for frame in original)
    assert all(np.abs(b[0] - a[0]).max() <= 1e-4 * peak
               for a, b in zip(original, equalized))
    # The state inputs carry the rescaled channels.
    assert any(not np.allclose(x, y) for a, b in zip(original, equalized)
               for x, y in zip(a[1:], b[1:]))


def test_cli_refusals(tmp_path, monkeypatch, capsys):
    source, _ = _checkpoint(tmp_path, 'prelu')
    out = tmp_path / 'out.pth'
    with pytest.raises(SystemExit) as refused:
        cle.main('DeepVQE_S', [str(source), '-o', str(source)])
    assert refused.value.code == 2
    elu, _ = _checkpoint(tmp_path, 'elu')
    with pytest.raises(SystemExit) as refused:
        cle.main('DeepVQE_S', [str(elu), '-o', str(out)])
    assert refused.value.code == 2
    assert 'relu, leaky_relu or prelu' in capsys.readouterr().err
    # A source rewritten while it is read (a training run saving over it).
    hashes = iter(['a' * 64, 'b' * 64])
    monkeypatch.setattr(cle, 'file_sha256', lambda path: next(hashes))
    with pytest.raises(SystemExit, match='changed while it was read'):
        cle.main('DeepVQE_S', [str(source), '-o', str(out)])
    assert not out.exists() and not (tmp_path / 'out.pth.partial').exists()
