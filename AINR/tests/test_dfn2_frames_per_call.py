"""A DFN2 graph that emits N frames per call equals the one-frame graph.

``export_onnx.StatelessDFN2Heads(model, frames_per_call=N)`` takes ``N + 2``
feature frames and returns N heads; the three GRUs run over the N frames as
one sequence and hand back the state after the last one, and the deep-filter
pathway window advances by N. Fed a stream in chunks of N with the state
carried between calls, it must reproduce the one-frame graph fed the same
stream frame by frame, for every lookahead pair:

  torch       the wrapper, every accepted pair, N in 1, 2, 3, 5, 6, 10, a
              stream whose length is not a multiple of N (the last call is
              completed with zero frames, as the host flushes), heads and the
              final state of both state tensors
  wrong carry the same comparison with the hidden state taken after the first
              frame, or the pathway window taken from the wrong slice, fails
  onnxruntime the exported graph streamed in chunks, a pair/N subset plus
              every pair at N = 3
  metadata    N = 1 writes exactly the one-frame metadata; N > 1 records
              frames_per_call and the chunk window, and
              ``validate_frames_per_call`` refuses a graph whose metadata and
              shapes disagree
"""

import json
import pathlib
import re
import sys

import numpy as np
import pytest
import torch

import dfn_pairs

FAMILY = 'DeepFilterNet2'
PAIRS = dfn_pairs.accepted_pairs(FAMILY)
FRAMES_PER_CALL = (1, 2, 3, 5, 6, 10)
STREAM = 23
# The pairs the exported graph is streamed for at every N; every pair is also
# streamed at N = 3.
ORT_PAIRS = ((0, 0), (1, 1), (2, 4), (0, 1), (2, 0))
ORT_FRAMES = (2, 3, 5, 10)
_pair_id = dfn_pairs.pair_id


@pytest.fixture(scope='module')
def export():
    return dfn_pairs.import_family_module(FAMILY, 'export_onnx')


@pytest.fixture(scope='module')
def contract():
    import onnx_streaming_contract
    return onnx_streaming_contract


def _model(pair, seed=7):
    model_mod = dfn_pairs.import_family_module(FAMILY, 'model')
    torch.manual_seed(seed)
    model = model_mod.DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64, enc_ch=8, emb_size=32,
        df_hidden=32, lin_groups=4, enc_lin_groups=4,
        mask_lookahead=pair[0], df_lookahead=pair[1])
    # Non-trivial batch-norm statistics, so a window off by one frame cannot
    # hide inside the tolerance.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0.0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def _stream(seed, model, frames=STREAM):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, frames, model.n_erb, generator=generator),
            torch.randn(1, 2, frames, model.df_bins, generator=generator))


def _pad_to(feature, frames):
    return torch.nn.functional.pad(
        feature, (0, 0, 0, frames - feature.shape[2]))


def _run(step, windows, state):
    """Drive ``step`` over the windows; heads concatenated on their time
    axes, and the final state."""
    heads = [[], [], []]
    for erb_window, spec_window in windows:
        output = step(erb_window, spec_window, *state)
        for index in range(3):
            heads[index].append(output[index])
        state = tuple(output[3:])
    return (torch.cat(heads[0], dim=2), torch.cat(heads[1], dim=1),
            torch.cat(heads[2], dim=1)), state


def _frame_reference(export, model, erb, spec, padded):
    """The one-frame graph over the stream zero-padded to ``padded`` frames."""
    wrapper = export.StatelessDFN2Heads(model).eval()
    erb, spec = _pad_to(erb, padded), _pad_to(spec, padded)
    with torch.no_grad():
        return _run(wrapper, zip(
            export.feature_windows(erb, model.mask_lookahead),
            export.feature_windows(spec, model.mask_lookahead)),
            wrapper.initial_inputs()[2:])


def _chunked(export, model, erb, spec, frames_per_call, step=None):
    wrapper = export.StatelessDFN2Heads(
        model, frames_per_call=frames_per_call).eval()
    with torch.no_grad():
        return _run(step or wrapper, zip(
            export.feature_chunk_windows(erb, model.mask_lookahead,
                                         frames_per_call),
            export.feature_chunk_windows(spec, model.mask_lookahead,
                                         frames_per_call)),
            wrapper.initial_inputs()[2:])


def _padded_length(frames_per_call, frames=STREAM):
    return -(-frames // frames_per_call) * frames_per_call


def _gap(got, want):
    return max(float((a - b).abs().max()) for a, b in zip(got, want))


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mask_lookahead', [0, 1, 2])
def test_one_frame_per_call_windows_are_the_frame_windows(
        export, mask_lookahead):
    feature = torch.randn(1, 2, 9, 4)
    chunks = list(export.feature_chunk_windows(feature, mask_lookahead, 1))
    frames = list(export.feature_windows(feature, mask_lookahead))
    assert len(chunks) == len(frames) == 9
    for chunk, frame in zip(chunks, frames):
        assert torch.equal(chunk, frame)


@pytest.mark.parametrize('frames_per_call', [2, 3, 5])
@pytest.mark.parametrize('mask_lookahead', [0, 1, 2])
def test_slice_i_of_a_chunk_window_is_the_window_of_head_i(
        export, mask_lookahead, frames_per_call):
    """Chunk c spans [cN-(2-L) .. cN+N-1+L]: its three frames from slot i are
    the one-frame window of head cN+i, also in the zero-completed last call."""
    feature = torch.randn(1, 2, 11, 4)
    padded = _pad_to(feature, _padded_length(frames_per_call, 11))
    frames = list(export.feature_windows(padded, mask_lookahead))
    chunks = list(export.feature_chunk_windows(feature, mask_lookahead,
                                               frames_per_call))
    assert len(chunks) * frames_per_call == len(frames)
    for index, chunk in enumerate(chunks):
        assert chunk.shape[2] == frames_per_call + 2
        for slot in range(frames_per_call):
            assert torch.equal(chunk[:, :, slot:slot + 3],
                               frames[index * frames_per_call + slot])


def test_chunk_window_alignment_names_the_first_and_last_frame(export):
    assert export.chunk_window_alignment(1, 4) == '[t-1 .. t+4] -> heads[t .. t+3]'
    assert export.chunk_window_alignment(0, 2) == '[t-2 .. t+1] -> heads[t .. t+1]'
    assert export.chunk_window_alignment(2, 10) == '[t .. t+11] -> heads[t .. t+9]'


@pytest.mark.parametrize('bad', [0, -1, True, 1.5, '2'])
def test_frames_per_call_must_be_a_positive_integer(export, bad):
    with pytest.raises(ValueError):
        export.StatelessDFN2Heads(_model((1, 1)), frames_per_call=bad)


def test_the_frames_per_call_cap_is_the_c_hosts(export):
    header = (pathlib.Path(export.__file__).parent / 'dfn2_model_io.h').read_text()
    cap = re.search(r'#define\s+DFN2_MODEL_MAX_FRAMES_PER_CALL\s+(\d+)', header)
    assert cap and export.MAX_FRAMES_PER_CALL == int(cap.group(1)) == 64
    assert export.chunk_input_frames(export.MAX_FRAMES_PER_CALL) == 66


@pytest.mark.parametrize('bad', [65, 1000])
def test_frames_per_call_above_the_cap_is_refused(export, bad):
    with pytest.raises(ValueError, match='64'):
        export.chunk_input_frames(bad)
    with pytest.raises(ValueError, match='64'):
        export.StatelessDFN2Heads(_model((1, 1)), frames_per_call=bad)


@pytest.mark.parametrize('bad', ['0', '65'])
@pytest.mark.parametrize('entry', ['export', 'calibration'])
def test_both_entry_points_refuse_frames_per_call_outside_the_cap(
        export, tmp_path, monkeypatch, capsys, entry, bad):
    """Both command lines stop at argument parsing (exit 2) before loading
    anything, with the cap in the message."""
    args = ['--model', str(tmp_path / 'absent.pth'), '--frames-per-call', bad]
    if entry == 'export':
        monkeypatch.setattr(sys, 'argv', ['export_onnx.py', *args,
                                          '--output', str(tmp_path / 'g.onnx')])
        entry_point = export.main
    else:
        monkeypatch.setattr(sys, 'argv', [
            'inference.py', *args, '--wav-dir', str(tmp_path),
            '--output', str(tmp_path / 'calib')])
        entry_point = dfn_pairs.import_family_module(
            FAMILY, 'inference').calibration_main
    with pytest.raises(SystemExit) as stop:
        entry_point()
    assert stop.value.code == 2
    assert '[1, 64]' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# torch: chunked graph == frame graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frames_per_call', FRAMES_PER_CALL)
@pytest.mark.parametrize('pair', PAIRS, ids=_pair_id)
def test_chunked_wrapper_equals_the_frame_wrapper(export, pair,
                                                  frames_per_call):
    model = _model(pair)
    erb, spec = _stream(11, model)
    padded = _padded_length(frames_per_call)
    want_heads, want_state = _frame_reference(export, model, erb, spec,
                                              padded)
    got_heads, got_state = _chunked(export, model, erb, spec,
                                    frames_per_call)
    assert got_heads[0].shape[2] == padded
    for got, want in zip(got_heads + got_state, want_heads + want_state):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    # The real frames are the offline model's heads too.
    with torch.no_grad():
        offline = model.heads(erb, spec)
    for got, want, axis in zip(got_heads, offline, (2, 1, 1)):
        torch.testing.assert_close(got.narrow(axis, 0, STREAM), want,
                                   rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mutation', ['hidden_after_first_frame',
                                      'pathway_window_oldest_frames'])
def test_a_wrong_state_carry_fails_the_equivalence(export, monkeypatch,
                                                   mutation):
    """The comparison above is not blind to the carry: a graph that hands
    back the hidden state of the first of the N frames, or the oldest five
    frames of the pathway concatenation instead of the newest, misses the
    frame graph by far more than the tolerance."""
    pair, frames_per_call = (1, 1), 3
    model = _model(pair)
    erb, spec = _stream(11, model)
    want_heads, want_state = _frame_reference(
        export, model, erb, spec, _padded_length(frames_per_call))
    right = _gap(*(_chunked(export, model, erb, spec, frames_per_call)[0],
                   want_heads))
    if mutation == 'hidden_after_first_frame':
        original = export._squeezed_gru

        def wrong(module, value, hidden):
            output, _ = original(module, value, hidden)
            _, first = module.gru(module.linear_in(value[:, :1]), hidden)
            return output, first
        monkeypatch.setattr(export, '_squeezed_gru', wrong)
    else:
        original = export._df_pathway_step

        def wrong(module, current, window, current_tail=None):
            output, _ = original(module, current, window, current_tail)
            combined = torch.cat((window[:, :, 1:], current), dim=2)
            return output, combined[:, :, :window.shape[2]]
        monkeypatch.setattr(export, '_df_pathway_step', wrong)
    got_heads, got_state = _chunked(export, model, erb, spec,
                                    frames_per_call)
    wrong_gap = _gap(got_heads, want_heads)
    assert right < 1e-5
    assert wrong_gap > 1e-3, wrong_gap
    with pytest.raises(AssertionError):
        for got, want in zip(got_heads + got_state,
                             want_heads + want_state):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# onnxruntime: the exported graph, streamed in chunks
# ---------------------------------------------------------------------------

_ORT_CASES = sorted(
    {(pair, n) for pair in ORT_PAIRS for n in ORT_FRAMES}
    | {(pair, 3) for pair in PAIRS})


def _export_small(export, tmp_path, model, frames_per_call, verify=True):
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    wrapper = export.StatelessDFN2Heads(
        model, frames_per_call=frames_per_call).eval()
    path = tmp_path / ('stream_n%d.onnx' % frames_per_call)
    metadata = export.export_graph(
        wrapper, {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        str(checkpoint), str(path), verify=verify)
    return path, metadata


@pytest.mark.parametrize('pair,frames_per_call', _ORT_CASES,
                         ids=['%s-n%d' % (_pair_id(p), n)
                              for p, n in _ORT_CASES])
def test_exported_chunk_graph_streams_to_the_frame_graph(
        export, tmp_path, pair, frames_per_call):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    model = _model(pair)
    path, metadata = _export_small(export, tmp_path, model, frames_per_call)
    assert metadata['frames_per_call'] == frames_per_call
    assert metadata['input_feature_frames'] == frames_per_call + 2
    assert metadata['output_frames_per_invocation'] == frames_per_call
    assert metadata['input_schema']['erb'][2] == frames_per_call + 2
    assert metadata['output_schema']['df_coefs'][1] == frames_per_call
    # The state tensors keep their one-frame shapes.
    for name in ('h_gru', 'df_convp_history'):
        assert (metadata['input_schema'][name]
                == metadata['output_schema'][name + '_out'])

    session = ort.InferenceSession(str(path),
                                   providers=['CPUExecutionProvider'])

    def step(erb_window, spec_window, *state):
        outputs = session.run(None, dict(zip(
            export.INPUT_NAMES,
            [erb_window.numpy(), spec_window.numpy()]
            + [value.numpy() if torch.is_tensor(value) else value
               for value in state])))
        return [torch.from_numpy(value) for value in outputs]

    erb, spec = _stream(13, model)
    want_heads, want_state = _frame_reference(
        export, model, erb, spec, _padded_length(frames_per_call))
    got_heads, got_state = _chunked(export, model, erb, spec,
                                    frames_per_call, step=step)
    for got, want in zip(got_heads + tuple(got_state),
                         want_heads + tuple(want_state)):
        np.testing.assert_allclose(got.numpy(), want.numpy(),
                                   rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# metadata and the validator
# ---------------------------------------------------------------------------

def test_one_frame_metadata_has_no_frames_per_call_entry(export, tmp_path):
    model = _model((1, 1))
    wrapper = export.StatelessDFN2Heads(model).eval()
    inputs = wrapper.initial_inputs()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    args = (str(checkpoint), {'SR': 1, 'N_FFT': 2, 'WIN_LEN': 2, 'HOP_LEN': 1},
            inputs, outputs, 1, 1)
    assert export.build_metadata(*args) == export.build_metadata(*args, 1)
    metadata = export.build_metadata(*args)
    assert 'frames_per_call' not in metadata
    assert 'chunk_window_alignment' not in metadata
    assert metadata['input_feature_frames'] == 3
    assert metadata['output_frames_per_invocation'] == 1


@pytest.fixture(scope='module')
def chunk_graph(export, tmp_path_factory):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    tmp_path = tmp_path_factory.mktemp('dfn2_n3')
    path, metadata = _export_small(export, tmp_path, _model((1, 1)), 3)
    assert metadata['chunk_window_alignment'] == (
        '[t-1 .. t+3] -> heads[t .. t+2]')
    return path


def _with_props(path, **changes):
    import onnx
    graph = onnx.load(str(path))
    props = {prop.key: prop.value for prop in graph.metadata_props}
    for key, value in changes.items():
        if value is None:
            props.pop(key, None)
        else:
            props[key] = value
    del graph.metadata_props[:]
    onnx.helper.set_model_props(graph, props)
    return graph


def _validate(export, contract, graph):
    return contract.validate_frames_per_call(
        graph, export.WINDOW_INPUT_TIME_AXES, export.HEAD_OUTPUT_TIME_AXES,
        export.INPUT_FRAMES)


def test_validator_accepts_the_exported_chunk_graph(export, contract,
                                                    chunk_graph):
    assert _validate(export, contract, _with_props(chunk_graph)) == 3


@pytest.mark.parametrize('changes,message', [
    (dict(frames_per_call=None, input_feature_frames=None,
          output_frames_per_invocation=None), 'not 3 for frames_per_call 1'),
    (dict(frames_per_call='1'), 'integer >= 2'),
    (dict(frames_per_call='2'), 'input_feature_frames is 5, not 4'),
    (dict(frames_per_call='three'), 'integer >= 2'),
    (dict(frames_per_call='03'), 'integer >= 2'),
    (dict(input_feature_frames='3'), 'input_feature_frames is 3'),
    (dict(output_frames_per_invocation='1'),
     'output_frames_per_invocation is 1'),
])
def test_validator_refuses_metadata_that_disagrees_with_the_shapes(
        export, contract, chunk_graph, changes, message):
    with pytest.raises(RuntimeError, match=message):
        _validate(export, contract, _with_props(chunk_graph, **changes))


def test_validator_refuses_a_one_frame_graph_that_claims_more(
        export, contract, tmp_path):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    path, metadata = _export_small(export, tmp_path, _model((1, 1)), 1,
                                   verify=False)
    assert 'frames_per_call' not in json.loads(
        path.with_suffix('.json').read_text())
    assert _validate(export, contract, _with_props(path)) == 1
    with pytest.raises(RuntimeError, match='frames_per_call 2'):
        _validate(export, contract, _with_props(
            path, frames_per_call='2', input_feature_frames=None,
            output_frames_per_invocation=None))


# ---------------------------------------------------------------------------
# calibration capture for an N-frame graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pair', [(1, 1), (0, 2), (2, 0)], ids=_pair_id)
def test_calibration_captures_the_invocations_of_the_chunk_graph(
        tmp_path, monkeypatch, pair):
    """``inference.py calib --frames-per-call N`` exports the N-frame graph
    and records its invocations: N + 2 frame windows that advance by N, the
    stream's own first window (2 - L leading zero frames) and a report and
    graph that both name N."""
    pytest.importorskip('soundfile')
    pytest.importorskip('onnxruntime')
    onnx = pytest.importorskip('onnx')
    frames_per_call = 3
    config, checkpoint, wavs = dfn_pairs.write_small_model(tmp_path, FAMILY,
                                                           pair)
    output = tmp_path / 'calib'
    monkeypatch.setattr(sys, 'argv', [
        'inference.py', '--config', str(config), '--model', str(checkpoint),
        '--wav-dir', str(wavs), '--output', str(output), '--format', 'bin',
        '--frames', '4', '--frames-per-call', str(frames_per_call)])
    dfn_pairs.import_family_module(FAMILY, 'inference').calibration_main()
    report = json.loads((output / 'manifest.json').read_text())
    assert report['frames_per_call'] == frames_per_call
    assert report['input_feature_frames'] == frames_per_call + 2
    assert report['frames'] == 4
    shape = report['binary_tensors']['erb']['frame_shape']
    assert shape[2] == frames_per_call + 2
    erb = np.stack([np.fromfile(output / 'erb' / ('erb_%04d.bin' % index),
                                '<f4').reshape(shape) for index in range(4)])
    leading = 2 - pair[0]
    assert not erb[0, 0, 0, :leading].any()
    assert erb[0, 0, 0, leading].any()
    np.testing.assert_array_equal(erb[1:, :, :, :2],
                                  erb[:-1, :, :, frames_per_call:])
    graph = onnx.load(str(output) + '.onnx')
    props = {prop.key: prop.value for prop in graph.metadata_props}
    assert props['frames_per_call'] == str(frames_per_call)
