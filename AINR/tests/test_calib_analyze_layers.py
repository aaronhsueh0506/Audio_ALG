"""``PTQ/calib_analyze.py --onnx``: the tensors inside the graph."""

import json
import os
import sys

import numpy as np
import pytest

from calibration_io import write_calibration_artifact

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

_PTQ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'PTQ')
if _PTQ not in sys.path:
    sys.path.insert(0, _PTQ)

import calib_analyze  # noqa: E402

N = 400
A_MAX, A_MIN = 37, 50           # a and b both at +3 / -3: the input extremes
H_MIN, H_MAX = 300, 301
DIFF_MAX, DIFF_MIN = 123, 200   # ordinary inputs, the extremes of a - b
INPUT_HOLDERS = [A_MAX, A_MIN, H_MIN, H_MAX]


def _arrays():
    """a == b on every frame (a - b is 0) except the two DIFF frames, whose
    inputs stay well inside the range frames A_MAX / A_MIN set."""
    rng = np.random.default_rng(0)
    a = (0.3 * rng.standard_normal((N, 1, 4))).astype(np.float32)
    b = a.copy()
    a[A_MAX] = b[A_MAX] = 3.0
    a[A_MIN] = b[A_MIN] = -3.0
    # A small opposite shift: each input's own mean, spread and peak stay
    # those of an ordinary frame.
    a[DIFF_MAX] += 0.1
    b[DIFF_MAX] -= 0.1
    a[DIFF_MIN] -= 0.03
    b[DIFF_MIN] += 0.03
    h = np.full((N, 1, 2), 0.01, dtype=np.float32)
    h[0] = 0.0
    h[H_MIN], h[H_MAX] = -0.5, 0.5
    return {'a': a, 'b': b, 'h': h, 'step': np.ones((N, 1), dtype=np.int64)}


def _graph(path, props=None, rename=None, width=4):
    """diff = a - b; m = 2 * Reshape(Relu(diff)); s = Sigmoid(50 * diff);
    hs = h * Cast(step)."""
    name = dict({'a': 'a', 'b': 'b'}, **(rename or {}))
    nodes = [
        helper.make_node('Sub', [name['a'], name['b']], ['diff']),
        helper.make_node('Relu', ['diff'], ['r']),
        helper.make_node('Reshape', ['r', 'shape'], ['flat']),
        helper.make_node('Mul', ['flat', 'two'], ['m']),
        helper.make_node('Mul', ['diff', 'fifty'], ['scaled']),
        helper.make_node('Sigmoid', ['scaled'], ['s']),
        helper.make_node('Cast', ['step'], ['stepf'], to=TensorProto.FLOAT),
        helper.make_node('Mul', ['h', 'stepf'], ['hs']),
    ]
    graph = helper.make_graph(
        nodes, 'tiny',
        [helper.make_tensor_value_info(name['a'], TensorProto.FLOAT, [1, width]),
         helper.make_tensor_value_info(name['b'], TensorProto.FLOAT, [1, width]),
         helper.make_tensor_value_info('h', TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info('step', TensorProto.INT64, [1])],
        [helper.make_tensor_value_info('m', TensorProto.FLOAT, [width]),
         helper.make_tensor_value_info('s', TensorProto.FLOAT, [1, width]),
         helper.make_tensor_value_info('hs', TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.array([width], dtype=np.int64), 'shape'),
         numpy_helper.from_array(np.float32(2.0), 'two'),
         numpy_helper.from_array(np.float32(50.0), 'fifty')])
    return _save(graph, path, props)


def _save(graph, path, props=None):
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    if props:
        helper.set_model_props(model, props)
    onnx.save(model, str(path))
    return str(path)


def _single(tmp_path, nodes, x):
    """Artifact of one input ``x`` [frames, 1, 3] and its graph ``nodes``
    (x -> y); returns the artifact and the graph path."""
    out = tmp_path / 'one.npz'
    write_calibration_artifact(out, {'x': x}, {'frames': len(x)}, 'npz')
    _save(helper.make_graph(
        nodes, 'one',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3])]),
        tmp_path / 'one.onnx')
    art = calib_analyze.load_artifact(str(out))
    return art, calib_analyze.graph_path(art)


def _pool(tmp_path, report=None, **graph):
    out = tmp_path / 'pool.npz'
    arrays = _arrays()
    write_calibration_artifact(out, arrays, dict(report or {}, frames=N),
                               'npz')
    _graph(tmp_path / 'pool.onnx', **graph)
    return calib_analyze.load_artifact(str(out))


def _layers(tmp_path, **kwargs):
    art = _pool(tmp_path, **kwargs)
    return art, calib_analyze.layer_stats(
        art, calib_analyze.graph_path(art))


def _select(art, layers, count, **kwargs):
    stats, schemes, _ = calib_analyze.prepare(art, 'asym')
    return calib_analyze.select_patterns(art, stats, schemes, count,
                                         layers=layers, **kwargs)


def test_layer_stats_match_the_graph_computed_by_hand(tmp_path):
    art, layers = _layers(tmp_path)
    a, b, h = (art.arrays[name].astype(np.float32) for name in 'abh')
    diff = a - b
    want = {
        'diff': diff, 'r': np.maximum(diff, 0), 'm': 2 * np.maximum(diff, 0),
        'scaled': 50 * diff, 's': 1 / (1 + np.exp(-50 * diff.astype(np.float64))),
        'stepf': np.ones((N, 1)), 'hs': h,
    }
    # The Reshape output repeats r and is not followed.
    assert layers['names'] == list(want)
    assert layers['ops'] == ['Sub', 'Relu', 'Mul', 'Mul', 'Sigmoid', 'Cast',
                             'Mul']
    for col, value in enumerate(want.values()):
        flat = value.reshape(N, -1)
        np.testing.assert_allclose(layers['frame_min'][:, col], flat.min(1),
                                   rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(layers['frame_max'][:, col], flat.max(1),
                                   rtol=1e-6, atol=1e-7)


def test_select_keeps_an_internal_extreme_the_inputs_do_not_show(tmp_path):
    art, layers = _layers(tmp_path)
    plain = _select(art, None, 10)['keep']
    assert DIFF_MAX not in plain and DIFF_MIN not in plain
    assert set(INPUT_HOLDERS) <= set(plain)

    sel = _select(art, layers, 10)
    assert set(INPUT_HOLDERS + [DIFF_MAX, DIFF_MIN]) <= set(sel['keep'])
    assert sel['keep'].size == 10
    # Relu's 0, the sigmoid's 0.5 and the constant Cast are reached on almost
    # every frame: two frames hold every internal end, not one frame per end.
    assert sel['must_layers'] == 2 and sel['must'] == 6

    kept = calib_analyze.layer_table(layers, sel['keep'])
    assert kept['narrowed'] == 0 and kept['worst'] is None
    assert all(row['kept'] == 1.0 for row in kept['rows'])

    lost = calib_analyze.layer_table(layers, plain)
    by_name = {row['tensor']: row for row in lost['rows']}
    assert lost['narrowed'] == 5        # diff, r, m, scaled, s
    assert by_name['diff']['keep_lo'] == 0.0 == by_name['diff']['keep_hi']
    assert by_name['diff']['kept'] == 0.0
    assert by_name['diff']['frames_outside_pct'] == pytest.approx(100 * 2 / N)
    assert by_name['r']['frames_outside_pct'] == pytest.approx(100 * 1 / N)
    assert by_name['hs']['kept'] == 1.0 == by_name['stepf']['kept']
    assert lost['frames_outside_max'] == pytest.approx(100 * 2 / N)


def test_internal_holders_are_ranked_when_the_count_cannot_take_them(tmp_path):
    art, layers = _layers(tmp_path)
    # Leaving DIFF_MAX out empties Relu's whole range; DIFF_MIN costs less.
    sel = _select(art, layers, 5)
    assert sorted(sel['keep']) == sorted(INPUT_HOLDERS + [DIFF_MAX])
    assert any('exceed --count 5' in note for note in sel['notes'])
    assert (sel['must'], sel['must_layers'], sel['kept_layers']) == (6, 2, 1)
    analysis = calib_analyze.analyze(art, count=5, layers=layers)
    text = ' '.join(item['text'] for item in analysis['conclusions'])
    assert '輸入的 4 個都保留，內部的 2 個依影響大小只保留了 1 個' in text
    assert '為內部 tensor 多留的有 1 個' in text
    # Fewer than the input holders: the input ranges themselves are short.
    short = _select(art, layers, 3)
    assert short['kept_layers'] == 0 and set(short['keep']) < set(INPUT_HOLDERS)
    analysis = calib_analyze.analyze(art, count=3, layers=layers)
    text = ' '.join(item['text'] for item in analysis['conclusions'])
    assert '需要 4 個 frame，超過 --count 3：挑出的 frame 的輸入 range 比整池窄' in text
    assert sorted(_select(art, layers, 6)['keep']) == sorted(
        INPUT_HOLDERS + [DIFF_MAX, DIFF_MIN])


def test_internal_extremes_are_taken_from_the_frames_the_budget_leaves(
        tmp_path):
    art, layers = _layers(tmp_path)
    stats, schemes, _ = calib_analyze.prepare(art, 'asym')
    sel = _select(art, layers, 10, max_clip_pct=0.5)
    assert sel['k_budget'] > 0 and A_MAX not in sel['pool']
    inner = calib_analyze.layer_comparison(art, stats, schemes, layers, sel,
                                           10, 0.5)
    assert inner['selected']['base_frames'] == sel['pool'].size
    assert inner['selected']['narrowed'] == 0
    assert inner['input_only']['narrowed'] > 0
    # The budget ranks by the inputs: what it dropped shows against every
    # frame (here hs, whose extremes are the dropped state frames).
    whole = inner['all_frames']
    assert whole['base_frames'] == N and whole['narrowed'] == 1
    assert whole['worst']['tensor'] == 'hs'
    text = ' '.join(line for _, line in calib_analyze._layer_lines(inner, 6))
    assert '以全部 400 個 frame 為基準：7 個內部 tensor 中有 1 個的 range 變窄' in text
    plain = calib_analyze.layer_comparison(
        art, stats, schemes, layers, _select(art, layers, 10), 10, 0.0)
    assert 'all_frames' not in plain


def test_graph_of_other_inputs_shape_or_checkpoint_is_refused(tmp_path):
    art = _pool(tmp_path, report={'checkpoint_sha256': 'a' * 64},
                props={'checkpoint_sha256': 'a' * 64})
    path = calib_analyze.graph_path(art)
    assert calib_analyze.layer_stats(art, path)['checkpoint_confirmed']

    other = _graph(tmp_path / 'renamed.onnx', rename={'a': 'x'})
    with pytest.raises(ValueError, match='takes inputs'):
        calib_analyze.layer_stats(art, other)
    wide = _graph(tmp_path / 'wide.onnx', width=5)
    with pytest.raises(ValueError, match=r'input a is \[1, 5\]'):
        calib_analyze.layer_stats(art, wide)
    stale = _graph(tmp_path / 'stale.onnx',
                   props={'checkpoint_sha256': 'b' * 64})
    with pytest.raises(ValueError, match='records checkpoint_sha256 bbbb'):
        calib_analyze.layer_stats(art, stale)
    art.report['state_layout_version'] = 15
    layout = _graph(tmp_path / 'layout.onnx',
                    props={'checkpoint_sha256': 'a' * 64,
                           'state_layout_version': '14'})
    with pytest.raises(ValueError, match='records state_layout_version 14'):
        calib_analyze.layer_stats(art, layout)
    del art.report['state_layout_version']
    art.arrays['a'] = art.arrays['a'].astype(np.float64)
    with pytest.raises(ValueError, match='input a is float32 but the artifact '
                                         'holds float64'):
        calib_analyze.layer_stats(art, path)
    art.arrays['a'] = art.arrays['a'].astype(np.float32)
    (tmp_path / 'broken.onnx').write_bytes(b'not a graph')
    with pytest.raises(ValueError, match='cannot read the graph'):
        calib_analyze.layer_stats(art, str(tmp_path / 'broken.onnx'))
    unstamped = _graph(tmp_path / 'unstamped.onnx')
    assert not calib_analyze.layer_stats(art, unstamped)[
        'checkpoint_confirmed']


def test_graph_path_follows_the_report_then_the_sibling_name(tmp_path):
    art = _pool(tmp_path)
    assert calib_analyze.graph_path(art) == str(tmp_path / 'pool.onnx')
    os.rename(tmp_path / 'pool.onnx', tmp_path / 'named.onnx')
    with pytest.raises(ValueError, match='no graph at'):
        calib_analyze.graph_path(art)
    art.report['graph'] = 'elsewhere/named.onnx'    # only the name is used
    assert calib_analyze.graph_path(art) == str(tmp_path / 'named.onnx')
    assert calib_analyze.graph_path(art, str(tmp_path / 'named.onnx')) == str(
        tmp_path / 'named.onnx')


def test_non_finite_internal_tensor_is_refused(tmp_path):
    art, path = _single(tmp_path, [helper.make_node('Log', ['x'], ['y'])],
                        np.full((4, 1, 3), -1.0, dtype=np.float32))
    with pytest.raises(ValueError, match='non-finite values in y'):
        calib_analyze.layer_stats(art, path)


def test_replay_is_compared_with_the_unchanged_graph(tmp_path, monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setattr(calib_analyze.np, 'allclose',
                        lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='changed output'):
        calib_analyze.layer_stats(art, calib_analyze.graph_path(art))


def test_a_head_made_by_a_layout_operator_is_compared_too(tmp_path,
                                                          monkeypatch):
    art, path = _single(tmp_path, [helper.make_node('Relu', ['x'], ['r']),
                                   helper.make_node('Flatten', ['r'], ['y'])],
                        np.ones((4, 1, 3), dtype=np.float32))
    assert calib_analyze.layer_stats(art, path)['names'] == ['r']
    monkeypatch.setattr(calib_analyze.np, 'allclose',
                        lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='changed output y'):
        calib_analyze.layer_stats(art, path)


def test_missing_onnxruntime_is_a_usage_error(tmp_path, capsys, monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    with pytest.raises(SystemExit):
        calib_analyze.main(['select', art.path, '--onnx'])
    assert 'needs the onnx and onnxruntime packages' in capsys.readouterr().err


def test_cli_select_layers_and_analyze_with_the_graph(tmp_path, capsys):
    art = _pool(tmp_path)
    keep_out = str(tmp_path / 'keep.json')
    assert calib_analyze.main(['select', art.path, '--count', '10', '--onnx',
                               '--keep-out', keep_out]) == 0
    out = capsys.readouterr().out
    assert '7 internal tensors followed' in out
    assert 'selected 10 (6 hold a tensor extreme)' in out
    assert ('2 frames hold an extreme of a tensor inside the graph only; 2 of '
            'them are kept') in out
    assert 'selected from the inputs only (10 of 400 frames): 5 of 7' in out
    assert 'selected with the graph (10 of 400 frames): 0 of 7' in out
    with open(keep_out, encoding='utf-8') as stream:
        record = json.load(stream)
    assert record['graph'] == str(tmp_path / 'pool.onnx')
    assert {DIFF_MAX, DIFF_MIN} <= set(record['keep_frames'])

    # An existing selection, here one made without the graph.
    plain_out = str(tmp_path / 'plain.json')
    assert calib_analyze.main(['select', art.path, '--count', '10',
                               '--keep-out', plain_out]) == 0
    with open(plain_out, encoding='utf-8') as stream:
        assert 'graph' not in json.load(stream)
    capsys.readouterr()
    table_out = str(tmp_path / 'layers.json')
    assert calib_analyze.main(['layers', art.path, '--keep', plain_out,
                               '--json', table_out]) == 0
    out = capsys.readouterr().out
    assert 'kept (10 of 400 frames): 5 of 7 internal tensors narrower' in out
    assert 'diff' in out and 'keeps   0.0%' in out
    with open(table_out, encoding='utf-8') as stream:
        assert json.load(stream)['kept']['narrowed'] == 5
    assert calib_analyze.main(['layers', art.path, '--keep', keep_out]) == 0
    assert '0 of 7 internal tensors narrower' in capsys.readouterr().out

    report = str(tmp_path / 'report.html')
    dump = str(tmp_path / 'analysis.json')
    assert calib_analyze.main(['analyze', art.path, '--count', '10', '--onnx',
                               str(tmp_path / 'pool.onnx'), '--out', report,
                               '--json', dump]) == 0
    with open(dump, encoding='utf-8') as stream:
        analysis = json.load(stream)
    inner = analysis['recommend']['layers']
    assert inner['selected']['narrowed'] == 0
    assert inner['input_only']['narrowed'] == 5 and inner['must_layers'] == 2
    assert inner['replayed'] == N and 'all_frames' not in inner
    text = ' '.join(item['text'] for item in analysis['conclusions'])
    assert '7 個內部 tensor 的 range 都和基準相同' in text
    assert '7 個內部 tensor 中有 5 個的 range 變窄' in text
    assert '這個工具看不到' not in text
    with open(report, encoding='utf-8') as stream:
        assert '"layers"' in stream.read()


def test_without_the_graph_nothing_about_layers_is_reported(tmp_path):
    art = _pool(tmp_path)
    analysis = calib_analyze.analyze(art, count=10)
    assert 'layers' not in analysis['recommend']
    text = ' '.join(item['text'] for item in analysis['conclusions'])
    assert '這個工具看不到' in text
    assert _select(art, None, 10)['must_layers'] == 0


def test_missing_graph_is_a_usage_error(tmp_path, capsys):
    art = _pool(tmp_path)
    os.remove(tmp_path / 'pool.onnx')
    with pytest.raises(SystemExit):
        calib_analyze.main(['select', art.path, '--onnx'])
    assert 'no graph at' in capsys.readouterr().err


def test_layers_reads_the_keep_file_before_the_replay(tmp_path, capsys,
                                                      monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setattr(calib_analyze, 'layer_stats', lambda *args: 1 / 0)
    with pytest.raises(SystemExit):
        calib_analyze.main(['layers', art.path, '--keep',
                            str(tmp_path / 'absent.json')])
    assert 'absent.json' in capsys.readouterr().err
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'keep_frames': [N]}))
    with pytest.raises(SystemExit):
        calib_analyze.main(['layers', art.path, '--keep', str(bad)])
    assert 'outside 0..399' in capsys.readouterr().err


def test_a_contiguous_artifact_is_not_replayed(tmp_path, capsys, monkeypatch):
    art = _pool(tmp_path, report={'contiguous_state_frames': N,
                                  'model_family': 'Align_CRUSE'})
    monkeypatch.setattr(calib_analyze, 'layer_stats', lambda *args: 1 / 0)
    assert calib_analyze.main(['analyze', art.path, '--onnx', '--out',
                               str(tmp_path / 'r.html')]) == 0
    assert 'note: --onnx is ignored' in capsys.readouterr().out
