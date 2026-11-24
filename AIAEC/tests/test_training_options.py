"""Constructor options, their checkpoint contract, the shipped per-grid configs and the
auxiliary STFT loss of DeepVQE-S and Align-ULCNet.

Every option defaults to the existing behaviour; the default path itself is
pinned by the existing model, streaming, export and trainer tests.
"""

import collections
import configparser
import contextlib
import importlib
import io
import pathlib
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

import AIAEC.training_common as training_common
from AIAEC._streaming_export import _build, export_graph as generic_export_graph
from AIAEC.aiaec_common import (
    ACTIVATIONS,
    DEFAULT_SCORE_KERNEL,
    CausalConvBlock,
    FreqUpsample,
    SeparableCausalConv,
    SignalGrid,
)
from AIAEC.Align_ULCNet import export_onnx as ulcnet_export
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.dataset_gen import stft
from AIAEC.dataset_gen.aec_features import AecGrid, PACKED_STEM_ORDER
from AIAEC.dataset_gen.linear_aec import make_linear_aec_contract
from AIAEC.DeepVQE_S.model import DeepVQES
from AIAEC.training_common import (
    AuxStftLoss,
    aux_loss_version,
    aux_recipe,
    aux_stft_loss_from_config,
    build_arg_parser,
    contract_model_kwargs,
    make_checkpoint_contract,
    plcpa_log,
    plcpa_training_loss,
    power_compressed_complex_mse_loss,
    read_config,
    read_model_kwargs,
    require_checkpoint_contract,
    stft_consistent_spectrum,
)
from test_training_common import _drive_main_capturing

AIAEC_DIR = pathlib.Path(training_common.__file__).parent
GRID256 = SignalGrid(16000, 256, 256, 128)
GRID512 = SignalGrid(16000, 512, 512, 256)
AEC256 = AecGrid(16000, 256, 256, 128)
CLASSES = {'DeepVQE_S': DeepVQES, 'Align_ULCNet': AlignULCNet}
# Measured offline-vs-stream worst case ~2e-6 (the GRU's batched and stepped
# kernels differ in summation order); ~10x headroom.
STREAM_ATOL = 2e-5


def _spectrum(frames, bins, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(1, frames, bins, generator=generator),
                         torch.randn(1, frames, bins, generator=generator))


def _cfg(text):
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


def _without_options(model_cls, kwargs):
    """The ctor kwargs a contract recorded before the options existed."""
    return {name: value for name, value in kwargs.items()
            if name not in model_cls.optional_contract_kwargs}


def _distinct_prelu_slopes(model):
    """Give every PReLU channel its own slope: the 0.25 init would hide a
    slope applied on the wrong axis."""
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.PReLU):
                module.weight.copy_(
                    torch.rand(module.weight.shape, generator=generator) - 0.5)


def _option_checkpoint(model_name, ctor, path, recorded=contract_model_kwargs,
                       grid=GRID256, distinct_prelu=False):
    """Build ``model_name`` on ``grid`` (16 kHz/256 unless given) from
    ``[model] max_delay_seconds = 0.5`` plus ``ctor`` and save it under the
    contract the trainer writes, recording ``recorded(model_cls, kwargs)``;
    returns ``(model, contract)``. ``distinct_prelu`` applies
    ``_distinct_prelu_slopes`` before the save."""
    model_cls = CLASSES[model_name]
    cfg = _cfg('[model]\nmax_delay_seconds = 0.5\n' + ''.join(
        '%s = %s\n' % (key, ','.join(map(str, value))
                       if isinstance(value, tuple) else value)
        for key, value in ctor.items()))
    kwargs = read_model_kwargs(cfg, model_cls)
    torch.manual_seed(2)
    model = model_cls(grid, **kwargs).eval()
    if distinct_prelu:
        _distinct_prelu_slopes(model)
    linear = make_linear_aec_contract(grid.sample_rate, frame_size=grid.n_fft)
    contract = make_checkpoint_contract(
        model_name=model_name, task=model_cls.task, grid=grid,
        model_kwargs=recorded(model_cls, kwargs), loss_version='test',
        data_contract={
            'linear_aec': linear.as_dict(),
            'linear_aec_contract_hash': linear.fingerprint()})
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    return model, contract


# ---- models ----------------------------------------------------------------

# relu and leaky_relu are parameterless point-wise ops like elu; prelu carries
# a per-channel weight.
@pytest.mark.parametrize('activation', ['elu', 'prelu'])
def test_deepvqe_options_stream_like_the_offline_forward(activation):
    torch.manual_seed(0)
    model = DeepVQES(GRID256, max_delay_seconds=0.5, similarity_channels=8,
                     gru_hidden=256, score_kernel=(9, 5),
                     activation=activation).eval()
    _distinct_prelu_slopes(model)
    assert model.align.score.conv.kernel_size == (9, 5)
    assert model.align.query.out_channels == 8
    assert model.gru.hidden_size == 256
    mic, far = _spectrum(32, 129, 1), _spectrum(32, 129, 2)
    with torch.no_grad():
        offline = model(microphone=mic, far_end=far).enhanced
        for frames_per_call in (1, 8):
            state = model.create_stream_state()
            streamed = torch.cat([
                model.forward_stream(mic[:, s:s + frames_per_call],
                                     far[:, s:s + frames_per_call],
                                     state).enhanced
                for s in range(0, 32, frames_per_call)], dim=1)
            torch.testing.assert_close(streamed, offline, atol=STREAM_ATOL,
                                       rtol=0)


@pytest.mark.parametrize('score_kernel,activation', [
    ((5, 5), 'elu'), ((9, 5), 'elu'), ((9, 5), 'prelu')])
def test_align_ulcnet_options_stream_like_the_offline_forward(
        score_kernel, activation):
    torch.manual_seed(0)
    model = AlignULCNet(GRID256, max_delay_seconds=0.5,
                        score_kernel=score_kernel, joint2_stride=1,
                        activation=activation).eval()
    _distinct_prelu_slopes(model)
    error, far = _spectrum(32, 129, 3), _spectrum(32, 129, 4)
    with torch.no_grad():
        offline = model(linear_error=error, far_end=far).enhanced
        state = model.create_stream_state()
        one = torch.cat([
            model.forward_stream(error[:, t:t + 1], far[:, t:t + 1],
                                 state).enhanced
            for t in range(32)], dim=1)
        wrapper = ulcnet_export.AlignUlcnetPacketExport(model, 8).eval()
        assert wrapper.score_window == score_kernel[0]
        packet_state = ulcnet_export.dummy_packet_inputs(
            8, wrapper.delay_depth, wrapper.n_freqs, wrapper.ta_bins,
            wrapper.score_window)[wrapper.signal_inputs:]
        packets = []
        for start in range(0, 32, 8):
            enhanced, packet_state = ulcnet_export.run_packet(
                wrapper, model, error[:, start:start + 8],
                far[:, start:start + 8], [], packet_state)
            packets.append(enhanced)
    torch.testing.assert_close(one, offline, atol=STREAM_ATOL, rtol=0)
    torch.testing.assert_close(torch.cat(packets, dim=1), offline,
                               atol=STREAM_ATOL, rtol=0)


def test_joint2_stride_one_keeps_the_512_bottleneck_at_fft_256():
    default = AlignULCNet(GRID256)
    stride_one = AlignULCNet(GRID256, joint2_stride=1)
    reference = AlignULCNet(GRID512)
    assert default.subband_widths == (2, 2)
    assert stride_one.subband_widths == reference.subband_widths == (4, 3)
    assert ([block.gru.input_size for block in stride_one.subband_grus]
            == [block.gru.input_size for block in reference.subband_grus]
            == [256, 192])
    with pytest.raises(ValueError, match='joint2_stride'):
        AlignULCNet(GRID256, joint2_stride=3)


@pytest.mark.parametrize('kernel', [(1, 3), (5, 4), (5, 0), (5,)])
def test_score_kernel_is_validated(kernel):
    with pytest.raises(ValueError, match='score_kernel'):
        DeepVQES(GRID256, score_kernel=kernel)


# activation -> (module type, exported ONNX op)
ACTIVATION_OPS = {'elu': (torch.nn.ELU, 'Elu'),
                  'relu': (torch.nn.ReLU, 'Relu'),
                  'leaky_relu': (torch.nn.LeakyReLU, 'LeakyRelu'),
                  'prelu': (torch.nn.PReLU, 'PRelu')}
NAMES = list(ACTIVATIONS)
# The blocks the paper gives ELU: DeepVQE-S six encoder blocks, three
# upsample and two residual blocks; Align-ULCNet four encoder and two joint
# convs.
ELU_SITES = {'DeepVQE_S': 11, 'Align_ULCNet': 6}
ULCNET_PRELU = ('mask_act', 'stage2_act')


def _count(model, kind):
    return sum(type(module) is kind for module in model.modules())


def _sites(model, kind):
    """The modules of type ``kind``, without Align-ULCNet's fixed PReLUs."""
    return [name for name, module in model.named_modules()
            if type(module) is kind and name not in ULCNET_PRELU]


@pytest.mark.parametrize('activation', NAMES)
@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_activation_option_replaces_every_elu(model_name, activation):
    model_cls = CLASSES[model_name]
    kwargs = read_model_kwargs(_cfg('[model]\nactivation = %s\n' % activation),
                               model_cls)
    assert kwargs['activation'] == activation
    model = model_cls(GRID256, **kwargs)
    sites = _sites(model, ACTIVATION_OPS[activation][0])
    assert len(sites) == ELU_SITES[model_name], sites
    assert activation == 'elu' or _count(model, torch.nn.ELU) == 0
    if model_name == 'DeepVQE_S':
        assert type(model.ccm_up.act) is torch.nn.Identity
    else:
        assert all(type(getattr(model, name)) is torch.nn.PReLU
                   for name in ULCNET_PRELU)
    if activation == 'prelu':       # one slope per channel
        site, width = ((model.mic1.activation, 16) if model_name == 'DeepVQE_S'
                       else (model.joint2.act, 96))
        assert site.weight.shape == (width,)


@pytest.mark.parametrize('build', [
    lambda name: DeepVQES(GRID256, activation=name),
    lambda name: AlignULCNet(GRID256, activation=name),
    lambda name: CausalConvBlock(2, 4, activation=name),
    lambda name: SeparableCausalConv(2, 4, activation=name),
    lambda name: FreqUpsample(2, 4, activation=name),
], ids=('deepvqe', 'ulcnet', 'conv_block', 'separable', 'upsample'))
@pytest.mark.parametrize('name', ['gelu', 'relu6', 'ELU', 'none', ''])
def test_unknown_activation_is_refused(build, name):
    with pytest.raises(ValueError, match='activation must be one of'):
        build(name)


def test_none_is_no_activation_only_for_the_bare_upsample():
    assert type(FreqUpsample(2, 4, activation=None).act) is torch.nn.Identity
    for build in (DeepVQES, AlignULCNet):
        with pytest.raises(ValueError, match='activation must be one of'):
            build(GRID256, activation=None)
    for block in (CausalConvBlock, SeparableCausalConv):
        with pytest.raises(ValueError, match='activation must be one of'):
            block(2, 4, activation=None)


# ---- contract --------------------------------------------------------------

def test_model_section_parses_tuple_options():
    cfg = _cfg('[model]\nscore_kernel = 9,5\njoint2_stride = 1\n')
    kwargs = read_model_kwargs(cfg, AlignULCNet)
    assert kwargs['score_kernel'] == (9, 5)
    assert kwargs['joint2_stride'] == 1
    model = AlignULCNet(GRID256, **kwargs)
    assert model.align.score.conv.kernel_size == (9, 5)


@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_contract_records_an_option_only_when_it_is_set(model_name):
    model_cls = CLASSES[model_name]
    defaults = read_model_kwargs(_cfg('[model]\n'), model_cls)
    recorded = contract_model_kwargs(model_cls, defaults)
    for name in model_cls.optional_contract_kwargs:
        assert name in defaults and name not in recorded
    assert set(defaults) - set(recorded) == set(
        model_cls.optional_contract_kwargs)

    explicit_default = read_model_kwargs(
        _cfg('[model]\nscore_kernel = 5,3\n'), model_cls)
    assert contract_model_kwargs(model_cls, explicit_default) == recorded

    changed = read_model_kwargs(_cfg('[model]\nscore_kernel = 9,5\n'),
                                model_cls)
    assert contract_model_kwargs(model_cls, changed)['score_kernel'] == (9, 5)

    assert contract_model_kwargs(model_cls, read_model_kwargs(
        _cfg('[model]\nactivation = elu\n'), model_cls)) == recorded
    relu = read_model_kwargs(_cfg('[model]\nactivation = relu\n'), model_cls)
    assert contract_model_kwargs(model_cls, relu)['activation'] == 'relu'


@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_a_checkpoint_without_the_option_keys_resumes_and_loads(
        model_name, tmp_path):
    """A contract written before the options existed (no score_kernel /
    joint2_stride / activation key) passes the resume check and loads through
    inference; a changed option refuses the resume both ways, and inference
    rebuilds the checkpoint's activation."""
    path = tmp_path / 'old.pth'
    _, old = _option_checkpoint(model_name, {}, path,
                                recorded=_without_options)
    assert not {'ctor_' + name for name
                in CLASSES[model_name].optional_contract_kwargs} & set(old)
    saved = torch.load(path, weights_only=False)
    _, current = _option_checkpoint(model_name, {}, tmp_path / 'new.pth')
    require_checkpoint_contract(saved, current)

    inference = importlib.import_module(f'AIAEC.{model_name}.inference')
    with contextlib.redirect_stdout(io.StringIO()):
        model = inference.load_model(str(path), 'cpu')[0]
    assert model.align.score.conv.kernel_size == DEFAULT_SCORE_KERNEL
    assert _count(model, torch.nn.ELU) == ELU_SITES[model_name]

    for option, value in (('score_kernel', (9, 5)), ('activation', 'relu'),
                          ('activation', 'prelu')):
        changed_path = tmp_path / 'changed.pth'
        built, changed = _option_checkpoint(
            model_name, {option: value}, changed_path, distinct_prelu=True)
        with pytest.raises(ValueError, match='ctor_' + option):
            require_checkpoint_contract(saved, changed)
        with pytest.raises(ValueError, match='ctor_' + option):
            require_checkpoint_contract(
                torch.load(changed_path, weights_only=False), current)
        if option == 'activation':
            with contextlib.redirect_stdout(io.StringIO()):
                reloaded = inference.load_model(str(changed_path), 'cpu')[0]
            assert len(_sites(reloaded, ACTIVATION_OPS[value][0])) == (
                ELU_SITES[model_name])
            want = built.state_dict()
            assert all(torch.equal(tensor, want[key])
                       for key, tensor in reloaded.state_dict().items())


# ---- export ----------------------------------------------------------------

@pytest.mark.parametrize('frames_per_call', [1, 8])
@pytest.mark.parametrize('grid,kernel,message', [
    (GRID256, (5, 3), 'score_kernel time size 5 != 9, .*16000 Hz / FFT 256'),
    (GRID512, (9, 5), 'score_kernel time size 9 != 5, .*16000 Hz / FFT 512'),
], ids=('fft256-k53', 'fft512-k95'))
def test_align_ulcnet_export_refuses_a_score_window_other_than_its_grid(
        tmp_path, frames_per_call, grid, kernel, message):
    """The window is per grid (SCORE_WINDOW_BY_GRID, the C rows): 9 at
    16 kHz / FFT 256, 5 elsewhere; the refusal names both and writes
    nothing."""
    checkpoint = tmp_path / 'k.pth'
    model, _ = _option_checkpoint(
        'Align_ULCNet', {'max_delay_frames': 8, 'score_kernel': kernel},
        checkpoint, grid=grid)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match=message):
        ulcnet_export.export_graph(model, str(checkpoint),
                                   str(out / 'graph.onnx'),
                                   frames_per_call=frames_per_call)
    assert not out.exists()


@pytest.mark.parametrize('frames_per_call', [1, 8])
def test_align_ulcnet_joint2_stride_changes_no_graph_port(
        tmp_path, frames_per_call):
    """At 16 kHz / FFT 256, joint2_stride 1 exports the same ports as 2 and
    the logit window is the grid's 9 (and the delay size is free)."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    schemas = {}
    for name, ctor in (('stride2', {'score_kernel': (9, 5)}),
                       ('stride1', {'score_kernel': (9, 5),
                                    'joint2_stride': 1}),
                       ('stride1_k93', {'score_kernel': (9, 3),
                                        'joint2_stride': 1})):
        checkpoint = tmp_path / (name + '.pth')
        model, _ = _option_checkpoint(
            'Align_ULCNet', dict(ctor, max_delay_frames=8), checkpoint)
        _wrapper, _inputs, metadata = ulcnet_export.export_graph(
            model, str(checkpoint), str(tmp_path / (name + '.onnx')),
            verify=name == 'stride2', frames_per_call=frames_per_call)
        schemas[name] = (metadata['input_schema'], metadata['output_schema'],
                         metadata['state_layout_version'],
                         metadata['score_window_frames'])
    assert schemas['stride1'] == schemas['stride2'] == schemas['stride1_k93']
    assert schemas['stride2'][3] == 9 == ulcnet_export.SCORE_WINDOW_BY_GRID[
        (16000, 256)]
    assert schemas['stride2'][0]['logit_history'] == [1, 32, 9, 8]


def test_align_ulcnet_export_cli_accepts_an_option_checkpoint(
        tmp_path, monkeypatch):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    checkpoint = tmp_path / 'k95.pth'
    _option_checkpoint('Align_ULCNet', {'max_delay_frames': 8,
                                        'score_kernel': (9, 5),
                                        'joint2_stride': 1}, checkpoint)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'export_onnx.py', '--checkpoint', str(checkpoint),
        '--output', str(out / 'graph.onnx'), '--verify'])
    ulcnet_export.main()
    assert (out / 'graph.onnx').is_file()
    assert (out / 'graph.json').is_file()


def test_deepvqe_options_export_and_verify(tmp_path):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    checkpoint = tmp_path / 'deepvqe.pth'
    model, _ = _option_checkpoint(
        'DeepVQE_S', {'similarity_channels': 8, 'gru_hidden': 256,
                      'score_kernel': (9, 5)}, checkpoint)
    for frames_per_call in (1, 8):
        built = _build('DeepVQE_S', model, frames_per_call)
        metadata = generic_export_graph(
            AEC256, built, str(checkpoint),
            str(tmp_path / ('n%d.onnx' % frames_per_call)), 63,
            verify=True, frames_per_call=frames_per_call)
        descriptor = metadata['c_descriptor']
        assert descriptor['score_window_frames'] == 9
        assert descriptor['gru_hidden'] == 256
        assert metadata['input_schema']['state_align_key_ring'] == [
            1, 8, 63, 33]


# The smallest checkpoint each exporter takes at 16 kHz / FFT 256 (the
# Align-ULCNet export needs the grid's logit window of 9).
EXPORT_CTOR = {'DeepVQE_S': {},
               'Align_ULCNet': {'max_delay_frames': 8, 'score_kernel': (9, 5)}}


def _export(model_name, model, checkpoint, path, frames_per_call, verify):
    if model_name == 'DeepVQE_S':
        built = _build(model_name, model, frames_per_call)
        generic_export_graph(AEC256, built, str(checkpoint), path, 63,
                             verify=verify, frames_per_call=frames_per_call)
    else:
        ulcnet_export.export_graph(model, str(checkpoint), path, verify=verify,
                                   frames_per_call=frames_per_call)


@pytest.mark.parametrize('activation,frames_per_call', [
    ('relu', 1), ('leaky_relu', 1), ('prelu', 1), ('prelu', 8)])
@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_activation_exports_and_verifies(model_name, activation,
                                         frames_per_call, tmp_path):
    """The exported graph replays the model (prelu slopes distinct per
    channel), and each Elu of the default graph becomes the chosen op."""
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    counts = {}
    for name in ('elu', activation):
        checkpoint = tmp_path / (name + '.pth')
        path = str(tmp_path / (name + '.onnx'))
        model, _ = _option_checkpoint(
            model_name, dict(EXPORT_CTOR[model_name], activation=name),
            checkpoint, distinct_prelu=True)
        _export(model_name, model, checkpoint, path, frames_per_call,
                verify=name != 'elu')
        counts[name] = collections.Counter(
            node.op_type for node in onnx.load(path).graph.node)
    op = ACTIVATION_OPS[activation][1]
    assert counts['elu']['Elu'] > 0 and counts[activation]['Elu'] == 0
    assert counts[activation][op] - counts['elu'][op] == counts['elu']['Elu']


CLIP = 16000 + 37 * 3 + 5       # 126 frames at hop 128, not a multiple of 8


@pytest.mark.parametrize('model_name,ctor', [
    ('DeepVQE_S', {'similarity_channels': 8, 'gru_hidden': 256,
                   'score_kernel': (9, 5)}),
    ('Align_ULCNet', {'score_kernel': (9, 5), 'joint2_stride': 1}),
])
def test_option_checkpoints_resume_load_and_run_the_inference_cli(
        model_name, ctor, tmp_path):
    path = tmp_path / 'variant.pth'
    _, contract = _option_checkpoint(model_name, ctor, path)
    for key, value in ctor.items():
        assert contract['ctor_' + key] == value
    require_checkpoint_contract(torch.load(path, weights_only=False), contract)

    inference = importlib.import_module(f'AIAEC.{model_name}.inference')
    model = inference.load_model(str(path), 'cpu')[0]
    assert model.align.score.conv.kernel_size == (9, 5)

    generator = np.random.RandomState(1)
    far = 0.1 * generator.randn(CLIP).astype('f')
    mic = 0.1 * generator.randn(CLIP).astype('f') + 0.3 * np.roll(far, 200)
    sf.write(tmp_path / 'mic.wav', mic, 16000)
    sf.write(tmp_path / 'far.wav', far, 16000)
    outputs = {}
    for frames_per_call in (1, 8):
        out = tmp_path / ('n%d.wav' % frames_per_call)
        args = inference.build_parser().parse_args([
            str(path), str(tmp_path / 'mic.wav'), str(tmp_path / 'far.wav'),
            str(out), '--frames-per-call', str(frames_per_call)])
        with contextlib.redirect_stdout(io.StringIO()):
            inference.main(args)
        outputs[frames_per_call] = sf.read(str(out), dtype='float32')[0]
    scale = float(np.abs(outputs[1]).max())
    assert scale > 0
    assert outputs[8].shape == outputs[1].shape == (CLIP,)
    assert float(np.abs(outputs[8] - outputs[1]).max()) < 1e-5 * scale


# ---- auxiliary STFT loss ---------------------------------------------------

LOSS_CFG = _cfg('[loss]\ncompression = 0.3\ncomplex_weight = 0.7\n'
                'stft_consistency = true\n')['loss']


def test_aux_loss_at_the_grid_fft_equals_the_primary_plcpa():
    generator = torch.Generator().manual_seed(5)
    target_wave = torch.randn(2, 128 * 40, generator=generator)
    target = stft(target_wave, AEC256).transpose(-2, -1)
    enhanced = target + 0.3 * torch.complex(
        torch.randn(target.shape, generator=generator),
        torch.randn(target.shape, generator=generator))
    primary = power_compressed_complex_mse_loss(
        stft_consistent_spectrum(enhanced, AEC256), target,
        compression=0.3, complex_weight=0.7)
    aux = AuxStftLoss(weight=1.0, n_fft=256, compression=0.3,
                      complex_weight=0.7)(enhanced, target, AEC256)
    torch.testing.assert_close(aux, primary, rtol=1e-5, atol=0)
    at_512 = AuxStftLoss(weight=0.5, n_fft=512, compression=0.3,
                         complex_weight=0.7)
    alone = at_512(enhanced, target, AEC256)
    assert torch.isfinite(alone) and alone > 0

    # The trainers' batch loss: the same PLCPA plus the term.
    total, plcpa = plcpa_training_loss(enhanced, target, AEC256, LOSS_CFG,
                                       consistency=True, aux_loss=at_512)
    assert torch.equal(plcpa, primary)
    assert torch.equal(total, primary + alone)
    total, plcpa = plcpa_training_loss(enhanced, target, AEC256, LOSS_CFG,
                                       consistency=True)
    assert total is plcpa and torch.equal(plcpa, primary)


def test_aux_loss_scale_does_not_depend_on_its_fft_size():
    """Two independent white noises: per-bin power, and so the PLCPA, is the
    same at every resolution once scaled (unscaled, 512 vs 256 differs by
    2 ** 0.3 = 1.23)."""
    generator = torch.Generator().manual_seed(9)
    noise = torch.randn(2, 2, 128 * 800, generator=generator)
    estimate = stft(noise[0], AEC256).transpose(-2, -1)
    target = stft(noise[1], AEC256).transpose(-2, -1)
    at_256, at_512, at_1024 = (
        AuxStftLoss(1.0, n_fft, 0.3, 0.7)(estimate, target, AEC256)
        for n_fft in (256, 512, 1024))
    assert float(at_512 / at_256) == pytest.approx(1.0, abs=0.03)
    assert float(at_1024 / at_256) == pytest.approx(1.0, abs=0.03)


def test_aux_loss_config_and_contract_fields():
    assert aux_stft_loss_from_config(_cfg('[loss]\n')) is None
    assert aux_stft_loss_from_config(
        _cfg('[loss]\naux_stft_weight = 0\n')) is None
    aux = aux_stft_loss_from_config(_cfg(
        '[loss]\ncompression = 0.4\ncomplex_weight = 0.6\n'
        'aux_stft_weight = 0.5\n'))
    assert (aux.weight, aux.n_fft, aux.compression, aux.complex_weight) == (
        0.5, 512, 0.4, 0.6)
    for bad in ('aux_stft_weight = -1\n',
                'aux_stft_weight = 0.5\naux_n_fft = 300\n'):
        with pytest.raises(ValueError):
            aux_stft_loss_from_config(_cfg('[loss]\n' + bad))

    assert aux_loss_version('base', None) == 'base'
    assert aux_loss_version('base', aux) == 'base+aux_stft512_w0.5'
    assert aux_recipe(None) == {}
    assert aux_recipe(aux) == {'loss_aux_stft_weight': 0.5,
                               'loss_aux_n_fft': 512}
    assert plcpa_log(train=None, val=None) == ''
    assert plcpa_log(train=1.0, val=None) == ' train_plcpa=1.0000'
    assert plcpa_log(train=1.0, val=2.0) == (
        ' train_plcpa=1.0000 val_plcpa=2.0000')


def test_aux_loss_refuses_a_wrong_stft_backward(monkeypatch):
    def stub(device, samples):
        return types.SimpleNamespace(
            device=types.SimpleNamespace(type=device), requires_grad=True,
            shape=(1, samples))

    bound = training_common._MPS_STFT_BACKWARD_SAFE_SAMPLES
    unsafe = training_common._mps_stft_backward_unsafe
    assert unsafe(stub('mps', bound + 1))
    assert not unsafe(stub('mps', bound))
    assert not unsafe(stub('cpu', bound + 1))

    monkeypatch.setattr(training_common, '_mps_stft_backward_unsafe',
                        lambda waveform: True)
    spectrum = _spectrum(8, 129, 6)
    with pytest.raises(RuntimeError, match='aux_stft_weight'):
        AuxStftLoss(0.5, 512, 0.3, 0.7)(spectrum, spectrum, AEC256)


def _batch(seed=7):
    generator = torch.Generator().manual_seed(seed)
    return (0.3 * torch.randn(2, len(PACKED_STEM_ORDER), 128 * 32,
                              generator=generator)).clamp(-1, 1)


@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_aux_loss_reports_the_plcpa_and_reaches_every_gradient(model_name):
    trainer = importlib.import_module(f'AIAEC.{model_name}.train')
    aux = AuxStftLoss(0.5, 512, 0.3, 0.7)
    loader = [(_batch(), None)]
    torch.manual_seed(0)
    model = CLASSES[model_name](GRID256, max_delay_seconds=0.5)

    off, _, untracked = trainer.run_epoch(model, loader, AEC256, 'cpu',
                                          LOSS_CFG)
    on, _, plcpa = trainer.run_epoch(model, loader, AEC256, 'cpu', LOSS_CFG,
                                     aux_loss=aux)
    assert untracked is None
    assert plcpa == off
    assert on > off

    model.train()
    output, spectral = trainer.forward_batch(model, _batch(), AEC256, 'cpu')
    aux(output.enhanced, spectral.target, AEC256).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(float(p.grad.abs().sum()) > 0 for p in model.parameters())


# ---- shipped grid configs and the trainer contract ------------------------------

def test_config_option_takes_several_files_in_order(tmp_path):
    # A later file overrides the keys it sets; the shipped grid configs are
    # self-contained and are not meant to be stacked.
    assert build_arg_parser('x').parse_args([]).config == ['config.ini']
    assert build_arg_parser('x').parse_args(
        ['--config', 'a.ini', 'b.ini']).config == ['a.ini', 'b.ini']
    base, overlay = tmp_path / 'base.ini', tmp_path / 'overlay.ini'
    base.write_text('[signal]\nsr = 16000\nn_fft = 256\n[data]\nx = 1\n')
    overlay.write_text('[signal]\nn_fft = 512\n')
    cfg = read_config([str(base), str(overlay)])
    assert cfg.getint('signal', 'n_fft') == 512
    assert cfg.getint('data', 'x') == 1
    with pytest.raises(FileNotFoundError, match='missing.ini'):
        read_config([str(base), str(tmp_path / 'missing.ini')])


def _setup(model_name, configs, monkeypatch, tmp_path, packed_dir=None):
    """What a trainer's main built from ``--config configs``."""
    seen = {}
    _drive_main_capturing(model_name, monkeypatch, tmp_path, configs, seen,
                          packed_dir=packed_dir)
    return seen


# grid -> (n_fft, hop, max_delay_frames, packed_dir, output_dir)
GRID_FILES = {
    (16000, 256): ('config_16k_256.ini', 256, 128, 63,
                   'data_aec_16k_256/packed/all', 'output_16k_256'),
    (16000, 512): ('config_16k_512.ini', 512, 256, 32,
                   'data_aec_16k/packed/all', 'output_16k_512'),
    (48000, 1024): ('config_48k.ini', 1024, 512, 47,
                    'data_aec_48k/packed/all', 'output_48k'),
}
# model -> grid -> (ctor_* keys the contract records, lr, aux loss on)
SHIPPED_RECIPE = {
    'DeepVQE_S': {
        (16000, 256): ({'ctor_gru_hidden': 256, 'ctor_similarity_channels': 8,
                        'ctor_score_kernel': (9, 5)}, 1.2e-3, True),
        (16000, 512): ({'ctor_gru_hidden': 192,
                        'ctor_similarity_channels': 4}, 1.2e-3, False),
        (48000, 1024): ({'ctor_gru_hidden': 192,
                         'ctor_similarity_channels': 4}, 1.2e-3, False),
    },
    'Align_ULCNet': {
        (16000, 256): ({'ctor_joint2_stride': 1, 'ctor_score_kernel': (9, 5)},
                       2e-3, False),
        (16000, 512): ({}, 4e-3, False),
        (48000, 1024): ({}, 4e-3, False),
    },
}
GRID_CASES = [(model_name, grid) for model_name in sorted(CLASSES)
              for grid in GRID_FILES]


GRID_CASE_IDS = [f'{model_name}-{sr // 1000}k-{n_fft}'
                 for model_name, (sr, n_fft) in GRID_CASES]


@pytest.mark.parametrize('model_name, grid', GRID_CASES, ids=GRID_CASE_IDS)
def test_each_shipped_grid_config_is_self_contained(
        model_name, grid, monkeypatch, tmp_path):
    """Every board grid has one config that trains alone: its [signal] is the
    grid its file name says, its model searches the board row's delay depth,
    and its corpus and output directories are its own."""
    name, n_fft, hop, depth, packed, output = GRID_FILES[grid]
    config = str(AIAEC_DIR / model_name / name)
    trainer = importlib.import_module(f'AIAEC.{model_name}.train')
    seen = _setup(model_name, [config], monkeypatch, tmp_path)

    cfg = read_config([config])
    assert (cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft')) == grid
    contract = seen['contract']
    assert (contract['n_fft'], contract['win_len'], contract['hop_len']) == (
        n_fft, n_fft, hop)
    assert seen['model'].align.max_delay_frames == depth
    assert seen['packed_dir'] == packed
    assert cfg.get('training', 'output_dir') == output

    ctor, lr, aux = SHIPPED_RECIPE[model_name][grid]
    for key in ('ctor_gru_hidden', 'ctor_similarity_channels',
                'ctor_score_kernel', 'ctor_joint2_stride', 'ctor_activation'):
        if key in ctor:
            value = ctor[key]
            got = contract[key]
            assert (tuple(got) if isinstance(value, tuple) else got) == value, key
        else:
            assert key not in contract, key
    assert contract['optimizer']['lr'] == pytest.approx(lr)
    assert contract['loss_version'].startswith(trainer.LOSS_VERSION)
    assert isinstance(seen['aux_loss'], AuxStftLoss) == aux


def test_the_two_grid_trainers_require_a_config_and_the_others_default_it():
    for model_name in ('DeepVQE_S', 'Align_ULCNet'):
        parser = importlib.import_module(f'AIAEC.{model_name}.train').build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
        assert parser.parse_args(['--config', 'config_16k_512.ini']
                                ).config == ['config_16k_512.ini']
    for model_name in ('Align_CRUSE', 'CAGCRN'):
        parser = importlib.import_module(f'AIAEC.{model_name}.train').build_parser()
        assert parser.parse_args([]).config == ['config.ini']
    with pytest.raises(SystemExit):
        build_arg_parser('x', default_config=None).parse_args([])


@pytest.mark.parametrize('model_name', sorted(CLASSES))
def test_aux_loss_changes_the_contract_so_resume_across_it_is_refused(
        model_name, monkeypatch, tmp_path):
    off = [str(AIAEC_DIR / model_name / 'config_16k_512.ini')]
    on = tmp_path / 'aux.ini'
    on.write_text('[loss]\naux_stft_weight = 0.5\n')
    trainer = importlib.import_module(f'AIAEC.{model_name}.train')
    off_seen = _setup(model_name, off, monkeypatch, tmp_path, 'fake')
    on_seen = _setup(model_name, [*off, str(on)], monkeypatch, tmp_path,
                     'fake')
    off_contract, on_contract = off_seen['contract'], on_seen['contract']
    assert off_contract['loss_version'] == trainer.LOSS_VERSION
    assert 'loss_aux_stft_weight' not in off_contract['optimizer']
    assert on_contract['loss_version'] == (
        trainer.LOSS_VERSION + '+aux_stft512_w0.5')
    assert on_contract['optimizer']['loss_aux_stft_weight'] == 0.5
    assert on_contract['optimizer']['loss_aux_n_fft'] == 512
    assert isinstance(on_seen['aux_loss'], AuxStftLoss)
    for saved, expected in ((off_contract, on_contract),
                            (on_contract, off_contract)):
        with pytest.raises(ValueError):
            require_checkpoint_contract({'contract': saved}, expected)
