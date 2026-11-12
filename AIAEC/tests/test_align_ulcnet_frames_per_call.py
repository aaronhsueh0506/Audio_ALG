"""Align-ULCNet at 16 kHz / FFT 256 / hop 128: N = 1 export, and the packet graph
(frames_per_call N > 1) with per-frame alignment events.

Every model here has RANDOM weights (no trained checkpoint exists for this
grid): the tests prove the boundary -- tensor order and shapes, state hand-off,
event semantics, the zero-completed last packet -- not audio quality.

Event semantics under test (stream/aiaec/ulcnet/ulcnet_stream.c): a boundary
frame resets the model state to zero BEFORE the frame and bypasses it (identity
head, model not stepped, state stays zero); the frame after it continues from
the reset state. The packet graph must equal the N = 1 stream that applies
exactly that at the same frames.
"""

import contextlib
import io
import json
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet import export_onnx as ex
from AIAEC.Align_ULCNet import inference as al_inference
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import make_linear_aec_contract

SR, N_FFT, HOP = 16000, 256, 128
GRID = SignalGrid(SR, N_FFT, N_FFT, HOP)
F = GRID.n_freqs
CLIP = SR + 37 * 3 + 5          # 126 frames: not a multiple of 8


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(1)
    return AlignULCNet(GRID, max_delay_seconds=0.5).eval()


@pytest.fixture(scope='module')
def checkpoint(model, tmp_path_factory):
    work = tmp_path_factory.mktemp('al256')
    linear = make_linear_aec_contract(SR, frame_size=N_FFT)
    contract = {
        'model_name': 'Align_ULCNet', 'task': AlignULCNet.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'ctor_max_delay_seconds': 0.5,
    }
    path = work / 'al256.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    rng = np.random.RandomState(1)
    far = 0.1 * rng.randn(CLIP).astype('f')
    mic = 0.1 * rng.randn(CLIP).astype('f') + 0.3 * np.roll(far, 200)
    sf.write(work / 'mic.wav', mic, SR)
    sf.write(work / 'far.wav', far, SR)
    return work, str(path)


def _complex(generator, frames):
    shape = (1, frames, F)
    return torch.complex(torch.randn(shape, generator=generator),
                         torch.randn(shape, generator=generator))


def _identity_mask():
    return torch.cat((torch.ones(1, 1, 1, F), torch.zeros(1, 1, 1, F)), dim=1)


def sequential_reference(model, error_c, far_c, events, state):
    """The N = 1 stream that applies reset + bypass at the event frames."""
    wrapper = ex.AlignUlcnetStreamingExport(model).eval()
    enhanced = []
    for index in range(error_c.shape[1]):
        *signals, compressed = ex.stream_features(
            model, torch.view_as_real(error_c[:, index:index + 1]),
            torch.view_as_real(far_c[:, index:index + 1]))
        if events[index]:
            state = tuple(torch.zeros_like(value) for value in state)
            mask = _identity_mask()
        else:
            outputs = wrapper(*signals, *state)
            mask, state = outputs[0], ex.next_state(outputs)
        enhanced.append(torch.view_as_complex(
            ex.host_back_end(model, compressed, mask).contiguous()))
    return torch.cat(enhanced, dim=1), state


def _zero_state(model, frames=None):
    wrapper = ex.AlignUlcnetStreamingExport(model)
    return ex.dummy_inputs(wrapper.delay_depth, wrapper.n_freqs,
                           wrapper.ta_bins)[ex.SIGNAL_INPUTS:]


EVENT_STREAMS = {
    'none': [[0] * 8] * 3,
    'frame3': [[0, 0, 1, 0, 0, 0, 0, 0], [0] * 8, [0] * 8],
    'frame1': [[1, 0, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8],
    'frameN': [[0] * 7 + [1], [0] * 8, [0] * 8],
    'two_in_one_packet': [[0, 0, 1, 0, 0, 1, 0, 0], [0] * 8, [0] * 8],
    'adjacent': [[0, 1, 1, 0, 0, 0, 0, 0], [0] * 8, [0] * 8],
    'every_packet': [[0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 1, 0, 1, 0, 0]],
    'all_frames': [[1] * 8, [0] * 8, [0] * 8],
}


@pytest.mark.parametrize('name', sorted(EVENT_STREAMS))
def test_packet_matches_sequential_reset_and_bypass_in_pytorch(model, name):
    packets = EVENT_STREAMS[name]
    wrapper = ex.AlignUlcnetPacketExport(model, 8).eval()
    generator = torch.Generator().manual_seed(11)
    packet_state = sequential_state = _zero_state(model)
    worst_out = worst_state = 0.0
    with torch.no_grad():
        for events in packets:
            error_c, far_c = _complex(generator, 8), _complex(generator, 8)
            got, packet_state = ex.run_packet(
                wrapper, model, error_c, far_c, events, packet_state)
            want, sequential_state = sequential_reference(
                model, error_c, far_c, events, sequential_state)
            worst_out = max(worst_out, (got - want).abs().max().item())
            worst_state = max(worst_state, max(
                (a - b).abs().max().item()
                for a, b in zip(packet_state, sequential_state)))
    print('%s: max |packet - sequential| output %.3g, end-of-packet state %.3g'
          % (name, worst_out, worst_state))
    assert worst_out < 1e-6 and worst_state < 1e-6


def test_event_frame_is_an_exact_identity_bypass_and_state_stays_zero(model):
    wrapper = ex.AlignUlcnetPacketExport(model, 8).eval()
    generator = torch.Generator().manual_seed(5)
    error_c, far_c = _complex(generator, 8), _complex(generator, 8)
    # warm the state up with one normal packet so a reset has something to clear
    _, state = ex.run_packet(wrapper, model, error_c, far_c, [0] * 8,
                             _zero_state(model))
    assert max(value.abs().max().item() for value in state) > 0
    error_c, far_c = _complex(generator, 8), _complex(generator, 8)
    *signals, compressed = ex.stream_features(
        model, torch.view_as_real(error_c), torch.view_as_real(far_c))
    # an event on the LAST frame: the packet ends in the reset state
    outputs = wrapper(*signals, ex.packet_frame_reset([0] * 7 + [1], 8), *state)
    assert outputs[0][:, 0, 7, :].eq(1).all() and outputs[0][:, 1, 7, :].eq(0).all()
    for value in outputs[1:]:
        assert value.abs().max().item() == 0.0       # state stays zero


def test_events_never_reach_back_into_earlier_frames(model):
    """Causality: the frames before an event are identical with or without it."""
    wrapper = ex.AlignUlcnetPacketExport(model, 8).eval()
    generator = torch.Generator().manual_seed(9)
    error_c, far_c = _complex(generator, 8), _complex(generator, 8)
    state = _zero_state(model)
    with torch.no_grad():
        quiet, _ = ex.run_packet(wrapper, model, error_c, far_c, [0] * 8, state)
        event, _ = ex.run_packet(wrapper, model, error_c, far_c,
                                 [0, 0, 0, 1, 0, 0, 0, 0], state)
    assert torch.equal(quiet[:, :3], event[:, :3])
    assert not torch.allclose(quiet[:, 4:], event[:, 4:])


def test_packet_graph_refuses_what_it_cannot_batch(model):
    for bad in (0, 1, 65, -2, True):
        with pytest.raises(ValueError, match='frames_per_call'):
            ex.AlignUlcnetPacketExport(model, bad)
    wrapper = ex.AlignUlcnetPacketExport(model, 4)
    with pytest.raises(ValueError, match='5 events for a 4-frame packet'):
        ex.packet_frame_reset([0] * 5, 4)
    assert wrapper.input_names[4] == 'frame_reset' and wrapper.signal_inputs == 5


# ---- ONNX Runtime ------------------------------------------------------------

@pytest.fixture(scope='module')
def packet_graph(checkpoint, model, tmp_path_factory):
    ort = pytest.importorskip('onnxruntime')
    _work, path = checkpoint
    out = tmp_path_factory.mktemp('graphs')
    graphs = {}
    for n in (1, 8):
        onnx_path = out / ('align_n%d.onnx' % n)
        with contextlib.redirect_stdout(io.StringIO()):
            wrapper, inputs, metadata = ex.export_graph(
                model, path, str(onnx_path), verify=True, frames_per_call=n)
        graphs[n] = (str(onnx_path), wrapper, inputs, metadata,
                     ort.InferenceSession(str(onnx_path),
                                          providers=['CPUExecutionProvider']))
    return graphs


def test_exported_graph_boundaries(packet_graph):
    import onnx
    one_path, _w1, _i1, meta1, _s1 = packet_graph[1]
    eight_path, _w8, _i8, meta8, _s8 = packet_graph[8]
    # N = 1: the unchanged layout-18 boundary, no control input
    assert meta1['state_layout_version'] == ex.LAYOUT_VERSION == 18
    assert 'frames_per_call' not in meta1 and 'control_inputs' not in meta1
    graph1 = onnx.load(one_path).graph
    assert [i.name for i in graph1.input] == list(ex.INPUT_NAMES)
    # N = 8: packet boundary layout 19
    assert meta8['state_layout_version'] == ex.PACKET_LAYOUT_VERSION == 19
    assert meta8['frames_per_call'] == 8 and meta8['frames_per_invocation'] == 8
    assert meta8['control_inputs']['frame_reset']['quantization'] == 'none'
    graph8 = onnx.load(eight_path).graph

    def shape(value):
        return [d.dim_value for d in value.type.tensor_type.shape.dim]

    assert [(i.name, shape(i)) for i in graph8.input] == [
        ('error_mag', [1, 8, F]), ('far_mag', [1, 8, F]),
        ('error_cos', [1, 8, F]), ('error_sin', [1, 8, F]),
        ('frame_reset', [1, 8]),
        ('key_history', [1, 32, 63, 13]), ('value_history', [1, 32, 63, 13]),
        ('logit_history', [1, 32, 5, 63]), ('h_gru', [1, 4, 1, 128])]
    assert [(o.name, shape(o)) for o in graph8.output] == [
        ('output', [1, 2, 8, F]),
        ('key_history_out', [1, 32, 63, 13]),
        ('value_history_out', [1, 32, 63, 13]),
        ('logit_history_out', [1, 32, 5, 63]), ('h_gru_out', [1, 4, 1, 128])]


@pytest.mark.parametrize('name', sorted(EVENT_STREAMS))
def test_onnx_runtime_packet_equals_sequential_stream(model, packet_graph, name):
    """ORT packet graph vs the N = 1 stream with reset + bypass: per-frame
    outputs, end-of-packet state, and the packet after it."""
    _path, wrapper, _inputs, _meta, session = packet_graph[8]
    packets = EVENT_STREAMS[name]
    generator = torch.Generator().manual_seed(21)
    state = sequential_state = _zero_state(model)
    worst_out = worst_state = 0.0
    for events in packets:
        error_c, far_c = _complex(generator, 8), _complex(generator, 8)
        *signals, compressed = ex.stream_features(
            model, torch.view_as_real(error_c), torch.view_as_real(far_c))
        feed = dict(zip(ex.PACKET_INPUT_NAMES,
                        (*signals, ex.packet_frame_reset(events, 8), *state)))
        outputs = session.run(None, {k: v.numpy() for k, v in feed.items()})
        enhanced = torch.view_as_complex(ex.host_back_end(
            model, compressed, torch.from_numpy(outputs[0])).contiguous())
        state = tuple(torch.from_numpy(value) for value in outputs[1:])
        with torch.no_grad():
            want, sequential_state = sequential_reference(
                model, error_c, far_c, events, sequential_state)
        worst_out = max(worst_out, (enhanced - want).abs().max().item())
        worst_state = max(worst_state, max(
            (a - b).abs().max().item()
            for a, b in zip(state, sequential_state)))
    print('ORT %s: max |graph - sequential| output %.3g, state %.3g'
          % (name, worst_out, worst_state))
    assert worst_out < 1e-5 and worst_state < 1e-5


# ---- general-WAV inference and calib -------------------------------------------

def _run_cli(work, checkpoint_path, name, frames_per_call=None):
    argv = [checkpoint_path, str(work / 'mic.wav'), str(work / 'far.wav'),
            str(work / name)]
    if frames_per_call is not None:
        argv += ['--frames-per-call', str(frames_per_call)]
    args = al_inference.build_parser().parse_args(argv)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        al_inference.main(args)
    audio, rate = sf.read(str(work / name), dtype='float32')
    assert rate == SR
    return audio, out.getvalue()


def test_wav_inference_packets_equal_the_one_frame_stream(checkpoint):
    work, path = checkpoint
    default, _ = _run_cli(work, path, 'default.wav')
    assert np.abs(default).max() > 1e-4              # not a trivially silent file
    scale = np.abs(default).max()
    for n in (2, 8):
        packets, text = _run_cli(work, path, 'n%d.wav' % n, n)
        assert packets.shape == default.shape == (CLIP,)
        assert np.abs(packets - default).max() < 1e-5 * scale
        assert np.abs(packets[-300:] - default[-300:]).max() < 1e-5 * scale
        assert '126 frames, %d model calls of %d' % (-(-126 // n), n) in text


def test_wav_inference_frames_per_call_is_validated(checkpoint):
    work, path = checkpoint
    from AIAEC._cli_common import require_frames_per_call
    assert require_frames_per_call('Align_ULCNet', 8) == 8
    for bad in (0, 65):
        with pytest.raises(ValueError, match=r'\[1, 64\]'):
            require_frames_per_call('Align_ULCNet', bad)
    with pytest.raises(SystemExit):                 # parser.error on 65
        sys.argv, old = ['inference.py', path, 'a', 'b', 'c',
                         '--frames-per-call', '65'], sys.argv
        try:
            al_inference.cli()
        finally:
            sys.argv = old


def test_calib_captures_n_frame_packets_and_a_control_input(checkpoint, tmp_path):
    from AIAEC import _streaming_calibration as calibration
    work, path = checkpoint
    err_dir, far_dir = tmp_path / 'err', tmp_path / 'far'
    err_dir.mkdir(), far_dir.mkdir()
    rng = np.random.RandomState(4)
    for directory in (err_dir, far_dir):
        sf.write(directory / 'a.wav', 0.1 * rng.randn(SR * 3 + 10).astype('f'), SR)
    reports = {}
    for n in (1, 8):
        out = tmp_path / ('cal%d.npz' % n)
        argv = ['calib', '--checkpoint', path, '--primary-dir', str(err_dir),
                '--far-dir', str(far_dir), '--allow-unstamped-linear-error',
                '--frames', '4', '--format', 'npz', '--frames-per-call', str(n),
                '--output', str(out)]
        old, sys.argv = sys.argv, argv
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    pytest.warns(RuntimeWarning):
                calibration.main('Align_ULCNet')
        finally:
            sys.argv = old
        data = np.load(out)
        assert data['error_mag'].shape == (4, 1, n, F)      # 4 INVOCATIONS of n frames
        assert data['key_history'].shape == (4, 1, 32, 63, 13)
        reports[n] = json.load(open(tmp_path / ('cal%d.calib.json' % n)))
        if n == 8:
            assert data['frame_reset'].shape == (4, 1, 8) and not data['frame_reset'].any()
        else:
            assert 'frame_reset' not in data.files
    assert reports[1]['state_layout_version'] == 18
    assert 'frames_per_call' not in reports[1]
    assert reports[8]['state_layout_version'] == 19
    assert reports[8]['frames_per_call'] == 8 and reports[8]['input_feature_frames'] == 8
    assert reports[8]['state_precision_policy']['frame_reset'] == 'binary_control_no_ptq'
    assert 'min' not in reports[8]['inputs']['frame_reset']    # no range for a control


# ---- 16 kHz / FFT 256 N = 1 export --------------------------------------------

def test_n1_export_at_16k_256_matches_forward_stream_and_ort(model, packet_graph):
    """The one-frame graph at the new grid: ORT == the PyTorch wrapper (done in
    export_graph's own verify) and the wrapper == model.forward_stream."""
    _path, wrapper, inputs, meta, _session = packet_graph[1]
    assert (meta['sample_rate'], meta['n_fft'], meta['hop_len']) == (SR, N_FFT, HOP)
    assert meta['max_delay_frames'] == 63
    assert meta['input_schema']['key_history'] == [1, 32, 63, 13]
    generator = torch.Generator().manual_seed(2)
    state = model.create_stream_state()
    graph_state = inputs[ex.SIGNAL_INPUTS:]
    with torch.no_grad():
        for _ in range(12):
            error_c, far_c = _complex(generator, 1), _complex(generator, 1)
            streamed = model.forward_stream(error_c, far_c, state).enhanced
            *signals, compressed = ex.stream_features(
                model, torch.view_as_real(error_c), torch.view_as_real(far_c))
            outputs = wrapper(*signals, *graph_state)
            graph_state = ex.next_state(outputs)
            enhanced = torch.view_as_complex(ex.host_back_end(
                model, compressed, outputs[0]).contiguous())
            assert (enhanced - streamed).abs().max().item() < 1e-6
