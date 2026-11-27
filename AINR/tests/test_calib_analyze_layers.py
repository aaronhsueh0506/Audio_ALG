"""``PTQ/calib_analyze.py``: the tensors inside the graph, and the scan."""

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


def _pool(tmp_path, report=None, arrays=None, name='pool', **graph):
    out = tmp_path / (name + '.npz')
    arrays = _arrays() if arrays is None else arrays
    write_calibration_artifact(out, arrays, dict(report or {},
                                                 frames=len(arrays['a'])),
                               'npz')
    _graph(tmp_path / (name + '.onnx'), **graph)
    return calib_analyze.load_artifact(str(out))


def _scan(art, path=None, **kwargs):
    return calib_analyze.scan_artifact(
        art, path or calib_analyze.graph_path(art), **kwargs)


def _layers(tmp_path, **kwargs):
    art = _pool(tmp_path, **kwargs)
    return art, _scan(art)


def _rows(table):
    return {row['tensor']: row for row in table['rows']}


def test_layer_stats_match_the_graph_computed_by_hand(tmp_path):
    art, pool = _layers(tmp_path)
    layers = pool.layers
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
    operands = dict(zip(layers['names'], layers['operands']))
    # m reads r through the Reshape; constants are not operands.
    assert operands == {'diff': ['a', 'b'], 'r': ['diff'], 'm': ['r'],
                        'scaled': ['diff'], 's': ['scaled'],
                        'stepf': ['step'], 'hs': ['h', 'stepf']}


def test_select_keeps_an_internal_extreme_the_inputs_do_not_show(tmp_path):
    _, pool = _layers(tmp_path)
    plain = calib_analyze.select_patterns(pool, 10, graph=False)['keep']
    assert DIFF_MAX not in plain and DIFF_MIN not in plain
    assert set(INPUT_HOLDERS) <= set(plain)

    sel = calib_analyze.select_patterns(pool, 10)
    assert set(INPUT_HOLDERS + [DIFF_MAX, DIFF_MIN]) <= set(sel['keep'])
    assert sel['keep'].size == 10
    # Relu's 0, the sigmoid's 0.5 and the constant Cast are reached on almost
    # every frame: two frames hold every internal end, not one frame per end.
    assert sel['must_layers'] == 2 and sel['must'] == 6

    kept = calib_analyze.layer_table(pool.layers, sel['keep'])
    assert kept['narrowed'] == 0 and kept['worst'] is None
    assert all(row['kept'] == 1.0 for row in kept['rows'])

    lost = calib_analyze.layer_table(pool.layers, plain)
    rows = _rows(lost)
    assert lost['narrowed'] == 5        # diff, r, m, scaled, s
    assert rows['diff']['keep_lo'] == 0.0 == rows['diff']['keep_hi']
    assert rows['diff']['kept'] == 0.0
    assert rows['diff']['frames_outside_pct'] == pytest.approx(100 * 2 / N)
    assert rows['r']['frames_outside_pct'] == pytest.approx(100 * 1 / N)
    # The grid holds 0: s stays above 0 at DIFF_MIN, so only DIFF_MAX is
    # past [0, 0.5].
    assert (rows['s']['keep_lo'], rows['s']['keep_hi']) == (0.0, 0.5)
    assert rows['s']['frames_outside_pct'] == pytest.approx(100 * 1 / N)
    assert rows['hs']['kept'] == 1.0 == rows['stepf']['kept']
    assert lost['frames_outside_max'] == pytest.approx(100 * 2 / N)


def test_internal_holders_are_ranked_when_the_count_cannot_take_them(tmp_path):
    _, pool = _layers(tmp_path)
    # Leaving DIFF_MAX out empties Relu's whole range; DIFF_MIN costs less.
    sel = calib_analyze.select_patterns(pool, 5)
    assert sorted(sel['keep']) == sorted(INPUT_HOLDERS + [DIFF_MAX])
    assert any('exceed --count 5' in note for note in sel['notes'])
    assert (sel['must'], sel['must_layers'], sel['kept_layers']) == (6, 2, 1)
    text = ' '.join(item['text'] for item in calib_analyze.analyze(
        pool, count=5)['conclusions'])
    assert '輸入的 4 個都保留，內部的 2 個依影響大小只保留了 1 個' in text
    assert '為內部 tensor 多留的有 1 個' in text
    # Fewer than the input holders: the input ranges themselves are short.
    short = calib_analyze.select_patterns(pool, 3)
    assert short['kept_layers'] == 0 and set(short['keep']) < set(INPUT_HOLDERS)
    text = ' '.join(item['text'] for item in calib_analyze.analyze(
        pool, count=3)['conclusions'])
    assert '需要 4 個 frame，超過 --count 3：挑出的 frame 的輸入 range 比整池窄' in text
    assert sorted(calib_analyze.select_patterns(pool, 6)['keep']) == sorted(
        INPUT_HOLDERS + [DIFF_MAX, DIFF_MIN])


def test_graph_of_other_inputs_shape_or_checkpoint_is_refused(tmp_path):
    art = _pool(tmp_path, report={'checkpoint_sha256': 'a' * 64},
                props={'checkpoint_sha256': 'a' * 64})
    path = calib_analyze.graph_path(art)
    assert _scan(art, path).layers['checkpoint_confirmed']

    other = _graph(tmp_path / 'renamed.onnx', rename={'a': 'x'})
    with pytest.raises(ValueError, match='takes inputs'):
        _scan(art, other)
    wide = _graph(tmp_path / 'wide.onnx', width=5)
    with pytest.raises(ValueError, match=r'input a is \[1, 5\]'):
        _scan(art, wide)
    stale = _graph(tmp_path / 'stale.onnx',
                   props={'checkpoint_sha256': 'b' * 64})
    with pytest.raises(ValueError, match='records checkpoint_sha256 bbbb'):
        _scan(art, stale)
    art.report['state_layout_version'] = 15
    layout = _graph(tmp_path / 'layout.onnx',
                    props={'checkpoint_sha256': 'a' * 64,
                           'state_layout_version': '14'})
    with pytest.raises(ValueError, match='records state_layout_version 14'):
        _scan(art, layout)
    del art.report['state_layout_version']
    arrays = dict(art.arrays, a=art.arrays['a'].astype(np.float64))
    wide_art = calib_analyze.Artifact(art.path, art.fmt, arrays, art.report)
    with pytest.raises(ValueError, match='input a is float32 but the artifact '
                                         'holds float64'):
        _scan(wide_art, path)
    (tmp_path / 'broken.onnx').write_bytes(b'not a graph')
    with pytest.raises(ValueError, match='cannot read the graph'):
        _scan(art, str(tmp_path / 'broken.onnx'))
    unstamped = _graph(tmp_path / 'unstamped.onnx')
    assert not _scan(art, unstamped).layers['checkpoint_confirmed']


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
        _scan(art, path)


def test_replay_is_compared_with_the_unchanged_graph(tmp_path, monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setattr(calib_analyze.np, 'allclose',
                        lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='changed output'):
        _scan(art)


def test_the_last_frame_is_compared_too(tmp_path, monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setattr(calib_analyze, '_check_at', lambda index: False)
    monkeypatch.setattr(calib_analyze.np, 'allclose',
                        lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='at frame %d' % (N - 1)):
        _scan(art)


def test_a_head_made_by_a_layout_operator_is_compared_too(tmp_path,
                                                          monkeypatch):
    art, path = _single(tmp_path, [helper.make_node('Relu', ['x'], ['r']),
                                   helper.make_node('Flatten', ['r'], ['y'])],
                        np.ones((4, 1, 3), dtype=np.float32))
    assert _scan(art, path).layers['names'] == ['r']
    monkeypatch.setattr(calib_analyze.np, 'allclose',
                        lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='changed output y'):
        _scan(art, path)


def test_missing_onnxruntime_is_a_usage_error(tmp_path, capsys, monkeypatch):
    art = _pool(tmp_path)
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', art.path, '--out',
                            str(tmp_path / 'r.html')])
    err = capsys.readouterr().err
    assert 'needs the onnx and onnxruntime packages' in err
    assert '--inputs-only' in err


def test_cli_analyze_replays_the_graph_and_writes_the_selection(tmp_path,
                                                                capsys):
    art = _pool(tmp_path)
    report = str(tmp_path / 'report.html')
    dump = str(tmp_path / 'analysis.json')
    out = str(tmp_path / 'sel.npz')
    assert calib_analyze.main(['analyze', art.path, '--count', '10',
                               '--out', report, '--json', dump,
                               '--output', out]) == 0
    printed = capsys.readouterr().out
    assert '7 internal tensors followed' in printed
    assert 'selected from the inputs only (10 of 400 frames): 5 of 7' in printed
    assert 'selected with the graph (10 of 400 frames): 0 of 7' in printed
    with open(dump, encoding='utf-8') as stream:
        analysis = json.load(stream)
    inner = analysis['recommend']['layers']
    assert inner['selected']['narrowed'] == 0
    assert inner['input_only']['narrowed'] == 5 and inner['must_layers'] == 2
    assert inner['replayed'] == N
    assert analysis['meta']['graph'] == str(tmp_path / 'pool.onnx')
    text = ' '.join(item['text'] for item in analysis['conclusions'])
    assert '7 個內部 tensor 的 range 都和基準相同' in text
    assert '7 個內部 tensor 中有 5 個的 range 變窄' in text
    assert '沒有重跑 graph' not in text
    with open(report, encoding='utf-8') as stream:
        assert '"layers"' in stream.read()
    sel = calib_analyze.load_artifact(out)
    diff = sel.arrays['a'] - sel.arrays['b']
    whole = art.arrays['a'] - art.arrays['b']
    assert diff.max() == whole.max() and diff.min() == whole.min()

    plain = str(tmp_path / 'plain.json')
    assert calib_analyze.main(['analyze', art.path, '--count', '10',
                               '--inputs-only', '--out', report,
                               '--json', plain]) == 0
    with open(plain, encoding='utf-8') as stream:
        analysis = json.load(stream)
    assert 'layers' not in analysis['recommend']
    assert analysis['meta']['graph'] is None
    assert any('這次沒有重跑 graph' in item['text']
               for item in analysis['conclusions'])


def test_a_graph_that_is_not_there(tmp_path, capsys):
    art = _pool(tmp_path)
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', art.path, '--onnx',
                            str(tmp_path / 'absent.onnx')])
    assert 'no graph at' in capsys.readouterr().err
    os.remove(tmp_path / 'pool.onnx')
    assert calib_analyze.main(['analyze', art.path, '--out',
                               str(tmp_path / 'r.html')]) == 0
    assert 'no graph beside the artifact' in capsys.readouterr().out


def test_a_contiguous_artifact_is_not_replayed(tmp_path, capsys, monkeypatch):
    art = _pool(tmp_path, report={'contiguous_state_frames': N,
                                  'model_family': 'Align_CRUSE'})
    monkeypatch.setattr(calib_analyze, '_Replay', lambda *args: 1 / 0)
    assert calib_analyze.main(['analyze', art.path, '--out',
                               str(tmp_path / 'r.html')]) == 0
    assert 'note: the graph is not replayed' in capsys.readouterr().out


# --- coverage of a held-out artifact inside the graph ------------------------

SPLIT = 150     # the held-out artifact's second source starts here


def _held(tmp_path):
    """The pool's frames as two sources: p.wav (0..149, holding DIFF_MAX)
    and q.wav."""
    arrays = _arrays()
    arrays['h'][SPLIT] = 0.0
    out = tmp_path / 'held.npz'
    write_calibration_artifact(out, arrays, {
        'frames': N, 'source_files': ['p.wav', 'q.wav']}, 'npz')
    return calib_analyze.load_artifact(str(out)), arrays


def _calibration(tmp_path, keep):
    """A calibration artifact holding the pool's frames ``keep``."""
    arrays = {name: value[keep] for name, value in _arrays().items()}
    return _pool(tmp_path, arrays=arrays, name='cal')


def _coverage(cal, held):
    path = calib_analyze.graph_path(cal)
    return calib_analyze.layer_coverage(_scan(cal, path), _scan(held, path))


def test_coverage_finds_how_far_a_held_out_file_goes_past_inside(tmp_path):
    keep = np.setdiff1d(np.arange(N), [DIFF_MAX])
    cal = _calibration(tmp_path, keep)
    held, arrays = _held(tmp_path)
    inner = _coverage(cal, held)
    rows = _rows(inner)
    diff = arrays['a'] - arrays['b']
    lo, hi = diff[keep].min(), diff[keep].max()     # [-0.06, 0]: 0 is inside
    assert rows['diff']['overshoot'] == pytest.approx(
        float(diff.max() - hi) / float(hi - lo), rel=1e-5)
    assert rows['diff']['frames_outside_pct'] == pytest.approx(100.0 / N)
    assert (rows['diff']['keep_lo'], rows['diff']['keep_hi']) == (lo, hi)
    # r and m are 0 on every calibration frame.
    assert rows['r']['overshoot'] == rows['m']['overshoot'] == np.inf
    assert rows['stepf']['overshoot'] == rows['hs']['overshoot'] == 0
    assert 'worst_source' not in rows['hs']
    # a and b stay inside on DIFF_MAX: diff is where it starts.
    assert inner['outside'] == 5 and inner['origins'] == ['diff']
    assert {row['worst_source'] for row in inner['rows']
            if row['overshoot'] > 0} == {'p.wav'}
    p, q = inner['sources']
    assert (p['source'], p['frames'], p['outside'], p['origins']) == (
        'p.wav', SPLIT, 5, ['diff'])
    assert p['worst'] == 'r' and p['overshoot'] == np.inf
    assert (q['source'], q['outside'], q['origins'], q['worst'],
            q['overshoot']) == ('q.wav', 0, [], None, 0.0)


def test_coverage_of_frames_inside_the_range_is_a_finding(tmp_path, capsys):
    cal = _calibration(tmp_path, np.arange(N))
    held, _ = _held(tmp_path)
    inner = _coverage(cal, held)
    assert inner['outside'] == 0 and inner['origins'] == []
    calib_analyze.print_layer_coverage(inner)
    out = capsys.readouterr().out
    assert 'every one of the 7 tensors stays within its calibration range' in out
    assert 'p.wav (150 frames): inside' in out


def test_coverage_shows_the_operands_of_an_addition(tmp_path, capsys):
    x = np.tile(np.array([[[0.1, -0.2, 0.3]]], dtype=np.float32), (6, 1, 1))
    art, path = _single(tmp_path, [
        helper.make_node('Relu', ['x'], ['r']),
        helper.make_node('Add', ['r', 'x'], ['y'])], x)
    big = tmp_path / 'big.npz'
    write_calibration_artifact(big, {'x': 4 * x}, {'frames': 6}, 'npz')
    held = calib_analyze.load_artifact(str(big))
    inner = calib_analyze.layer_coverage(_scan(art, path), _scan(held, path))
    assert inner['segments'] == 'no tensor starts from an all-zero state'
    calib_analyze.print_layer_coverage(inner)
    out = capsys.readouterr().out
    # Every frame of the held-out input is past: nothing starts inside.
    assert '2 of 2 tensors go past' in out
    assert 'every operand stays inside: none (' in out
    assert 'operand r ' in out
    assert 'per-source breakdown unavailable' in out


def test_cli_coverage_with_the_graph(tmp_path, capsys, monkeypatch):
    cal = _calibration(tmp_path, np.setdiff1d(np.arange(N), [DIFF_MAX]))
    held, _ = _held(tmp_path)
    dump = str(tmp_path / 'coverage.json')
    assert calib_analyze.main(['coverage', cal.path, '--against', held.path,
                               '--json', dump, '--jobs', '2']) == 0
    out = capsys.readouterr().out
    assert 'replaying 399 frames of %s' % cal.path in out
    assert 'replaying 400 frames of %s' % held.path in out
    assert '5 of 7 tensors go past' in out
    assert 'p.wav (150 frames): 5 tensors past, furthest r by a zero-width ' \
        'range; origins: diff' in out
    assert ' * diff ' in out
    with open(dump, encoding='utf-8') as stream:
        inner = json.load(stream)['layers']
    assert inner['outside'] == 5 and inner['frames'] == N - 1
    assert _rows(inner)['r']['overshoot'] is None       # inf has no JSON value

    plain = str(tmp_path / 'plain.json')
    assert calib_analyze.main(['coverage', cal.path, '--against', held.path,
                               '--inputs-only', '--json', plain]) == 0
    assert 'inside the graph' not in capsys.readouterr().out
    with open(plain, encoding='utf-8') as stream:
        assert 'layers' not in json.load(stream)

    # Another checkpoint is refused before the replay.
    other = tmp_path / 'other.npz'
    write_calibration_artifact(other, _arrays(), {
        'frames': N, 'checkpoint_sha256': 'y'}, 'npz')
    (tmp_path / 'x').mkdir()
    art2 = _pool(tmp_path / 'x', report={'checkpoint_sha256': 'x'})
    monkeypatch.setattr(calib_analyze, 'scan_artifact', lambda *args: 1 / 0)
    with pytest.raises(SystemExit):
        calib_analyze.main(['coverage', art2.path, '--against', str(other)])
    assert 'checkpoint_sha256' in capsys.readouterr().err


def test_an_origin_is_judged_frame_by_frame(tmp_path):
    """y = sum(x), z = -y.  y is an origin where it goes past on a frame whose
    x stays inside the range of the calibration frames, and not where x goes
    past on the same frame, on either side."""
    graph = helper.make_graph(
        [helper.make_node('ReduceSum', ['x'], ['y'], keepdims=1),
         helper.make_node('Neg', ['y'], ['z'])], 'sum',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 1])])
    # x ranges over [-1, 1] and y over [0, 1].
    frames = {'cal': [[1, -1, 0], [0.5, 0.5, 0], [-1, 1, 0]],
              # x past on both sides but y inside; then y past with x inside
              'starts': [[2, -2, 0], [1, 1, 1], [0, 0, 0]],
              # y past only where x is past too: past its top (inside the
              # range of every frame), then past its bottom
              'follows': [[1.2, 0.3, 0], [-1.5, 0.5, -0.5], [0, 0, 0]]}
    pools = {}
    for name, x in frames.items():
        write_calibration_artifact(
            tmp_path / (name + '.npz'),
            {'x': np.array(x, dtype=np.float32)[:, None, :]},
            {'frames': len(x)}, 'npz')
        _save(graph, tmp_path / (name + '.onnx'))
        pools[name] = _scan(calib_analyze.load_artifact(
            str(tmp_path / (name + '.npz'))))
    starts = calib_analyze.layer_coverage(pools['cal'], pools['starts'])
    assert starts['outside'] == 2 and starts['origins'] == ['y']
    rows = _rows(starts)
    assert rows['y']['overshoot'] == pytest.approx(2.0)   # 3 past [0, 1]
    assert rows['y']['frames_outside_pct'] == pytest.approx(100 / 3)
    follows = calib_analyze.layer_coverage(pools['cal'], pools['follows'])
    assert follows['outside'] == 2 and follows['origins'] == []


# --- the scan: a stream, its parts and its threads ---------------------------

def _sources():
    """The pool's frames as two sources, p.wav (0..149) and q.wav."""
    arrays = _arrays()
    arrays['h'][SPLIT] = 0.0
    return arrays, ['p.wav', 'q.wav'], np.r_[np.zeros(SPLIT, int),
                                             np.ones(N - SPLIT, int)]


def _stream(arrays, src, path, select=None, reserve=0):
    """A Scan fed one frame at a time, as a capture run feeds it."""
    specs = [(name, value.shape[1:], value.dtype)
             for name, value in arrays.items()]
    scan = calib_analyze.Scan(specs, graph=path, reserve=reserve)
    for index in range(len(src)) if select is None else select:
        scan.add({name: value[index] for name, value in arrays.items()},
                 int(src[index]))
    return scan, specs


def _assert_same_pool(one, two):
    assert one.count == two.count
    assert np.array_equal(one.candidates, two.candidates)
    for name in one.stats:
        a, b = one.stats[name], two.stats[name]
        assert a['global'] == b['global']
        assert a['concentration'] == b['concentration']
        assert np.array_equal(a['frame_min'], b['frame_min'])
        assert np.array_equal(a['_sample'], b['_sample'])
    assert np.array_equal(one.features, two.features)
    for key in ('frame_min', 'frame_max'):
        assert np.array_equal(one.layers[key], two.layers[key])
    assert one.layers['bands'] == two.layers['bands']
    # The note says where the sources came from (state resets or the
    # capture), which differs between the two routes.
    assert {k: v for k, v in one.sources.items() if k != 'note'} == \
        {k: v for k, v in two.sources.items() if k != 'note'}


def test_a_stream_scanned_frame_by_frame_selects_what_the_stored_pool_does(
        tmp_path, monkeypatch):
    """Fewer frames are kept than the stream holds, so the capture route only
    has the frames it retained; it must choose as the stored route does, and
    the frames it writes must hold every tensor's range of the whole
    stream."""
    monkeypatch.setattr(calib_analyze, 'SAMPLE_FRAMES', 30)
    count = 12
    reserve = calib_analyze.fill_reserve(count)
    assert reserve < N
    arrays, names, src = _sources()
    out = tmp_path / 'pool.npz'
    write_calibration_artifact(out, arrays, {'frames': N, 'source_files': names},
                               'npz')
    path = _graph(tmp_path / 'pool.onnx')
    stored = _scan(calib_analyze.load_artifact(str(out)), reserve=reserve)

    scan, specs = _stream(arrays, src, path, reserve=reserve)
    part = scan.part()
    assert len(part['kept_at']) < N            # the bounded path ran
    streamed = calib_analyze.assemble([part], source_names=names,
                                      reserve=reserve)
    _assert_same_pool(stored, streamed)
    keep = calib_analyze.select_patterns(streamed, count)['keep']
    assert np.array_equal(keep, calib_analyze.select_patterns(stored,
                                                              count)['keep'])
    written = tmp_path / 'sel.npz'
    calib_analyze.write_selection(streamed, keep, str(written), 'npz')
    _graph(tmp_path / 'sel.onnx')
    again = _scan(calib_analyze.load_artifact(str(written)))
    for name in streamed.stats:
        assert again.stats[name]['global']['min'] == \
            streamed.stats[name]['global']['min']
        assert again.stats[name]['global']['max'] == \
            streamed.stats[name]['global']['max']
    assert np.array_equal(again.layers['frame_min'].min(axis=0),
                          streamed.layers['frame_min'].min(axis=0))
    assert np.array_equal(again.layers['frame_max'].max(axis=0),
                          streamed.layers['frame_max'].max(axis=0))


def test_scans_of_separate_sources_merge_into_the_scan_of_all(tmp_path):
    arrays, names, src = _sources()
    path = _graph(tmp_path / 'pool.onnx')
    reserve = 50
    whole, specs = _stream(arrays, src, path, reserve=reserve)
    one = calib_analyze.assemble([whole.part()], source_names=names,
                                 reserve=reserve)
    parts = []
    for k, frames in enumerate((np.flatnonzero(src == 1),
                                np.flatnonzero(src == 0))):
        scan, _ = _stream(arrays, src, path, frames, reserve=reserve)
        folder = tmp_path / ('part%d' % k)
        calib_analyze.save_part(scan.part(), str(folder))
        parts.append(calib_analyze.load_part(str(folder)))
    merged = calib_analyze.assemble(parts, source_names=names,
                                    reserve=reserve)
    _assert_same_pool(one, merged)
    keep = calib_analyze.select_patterns(merged, 12)['keep']
    assert np.array_equal(keep, calib_analyze.select_patterns(one, 12)['keep'])
    for name in arrays:
        np.testing.assert_array_equal(merged.frames(keep, name),
                                      arrays[name][keep])


def test_more_threads_give_the_same_scan(tmp_path):
    art = _pool(tmp_path)
    one = _scan(art, threads=1)
    three = _scan(art, threads=3)
    _assert_same_pool(one, three)


def test_the_tensors_on_the_way_to_a_softmax_are_marked(tmp_path):
    """t = ReduceSum(q * k) + ReduceSum(Relu(k * k)), y = Softmax(t): q, k,
    their product, the Relu's output and the sums reach the Softmax through no
    activation; k * k reaches it only through the Relu."""
    graph = helper.make_graph(
        [helper.make_node('Mul', ['q', 'k'], ['m']),
         helper.make_node('ReduceSum', ['m', 'axes'], ['s'], keepdims=1),
         helper.make_node('Mul', ['k', 'k'], ['p']),
         helper.make_node('Relu', ['p'], ['g']),
         helper.make_node('ReduceSum', ['g', 'axes'], ['gs'], keepdims=1),
         helper.make_node('Add', ['s', 'gs'], ['t']),
         helper.make_node('Softmax', ['t'], ['y'], axis=0)], 'att',
        [helper.make_tensor_value_info('q', TensorProto.FLOAT, [4, 3]),
         helper.make_tensor_value_info('k', TensorProto.FLOAT, [4, 3])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 1])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), 'axes')])
    rng = np.random.default_rng(2)
    arrays = {'q': rng.standard_normal((20, 4, 3)).astype(np.float32),
              'k': rng.standard_normal((20, 4, 3)).astype(np.float32)}
    write_calibration_artifact(tmp_path / 'att.npz', arrays, {'frames': 20},
                               'npz')
    _save(graph, tmp_path / 'att.onnx')
    pool = _scan(calib_analyze.load_artifact(str(tmp_path / 'att.npz')))
    assert pool.layers['softmax'] == ['g', 'gs', 'k', 'm', 'q', 's', 't']
    table = _rows(calib_analyze.layer_table(pool.layers, np.arange(20)))
    assert table['t']['softmax'] and table['g']['softmax']
    assert not table['p']['softmax'] and not table['y']['softmax']


# --- one index dominating a tensor -------------------------------------------

BANDS = 60          # frames of the band artifact
SPIKE = 9           # the one frame where channel 2 of x is large


def _bands(tmp_path):
    """x [1, 8, 3]: small values, except channel 2 on frame SPIKE.
    y = x + c, where c is 100 on channel 1 only: a constant offset every
    frame, so y's channel 1 is large on a typical frame too.  w = x + d, d
    20 on channel 5: large on a typical frame, but SPIKE's channel 2 is
    larger still.  z = x * 1."""
    rng = np.random.default_rng(3)
    x = (0.1 * rng.standard_normal((BANDS, 1, 8, 3))).astype(np.float32)
    x[SPIKE, :, 2] = 50.0
    write_calibration_artifact(tmp_path / 'bands.npz', {'x': x},
                               {'frames': BANDS}, 'npz')
    offset = np.zeros((1, 8, 3), dtype=np.float32)
    offset[:, 1] = 100.0
    small = np.zeros((1, 8, 3), dtype=np.float32)
    small[:, 5] = 20.0
    graph = helper.make_graph(
        [helper.make_node('Add', ['x', 'd'], ['w']),
         helper.make_node('Add', ['x', 'c'], ['y']),
         helper.make_node('Mul', ['x', 'one'], ['z'])], 'bands',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 8, 3])],
        [helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 8, 3]),
         helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 8, 3]),
         helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 8, 3])],
        [numpy_helper.from_array(offset, 'c'), numpy_helper.from_array(small, 'd'),
         numpy_helper.from_array(np.float32(1.0), 'one')])
    _save(graph, tmp_path / 'bands.onnx')
    art = calib_analyze.load_artifact(str(tmp_path / 'bands.npz'))
    return art, calib_analyze.graph_path(art)


def _same(a, b):
    """Nested dicts and lists of numbers equal to float rounding."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return np.allclose(a, b, rtol=1e-9, atol=0)


def test_a_fixed_band_is_told_from_one_loud_frame(tmp_path):
    art, path = _bands(tmp_path)
    pool = _scan(art, path)
    bands = pool.layers['bands']
    fixed = bands['y']['channel']
    assert fixed['argmax'] == fixed['frame_top'] == 1
    assert fixed['frame_share'] == 1.0 and fixed['top_shares'] == [[1, 1.0]]
    assert fixed['ratio'] > 100 and fixed['typical'] > 100
    assert calib_analyze.band_kind(fixed) == 'fixed'
    loud = bands['z']['channel']
    assert loud['argmax'] == 2 and loud['ratio'] > 100
    assert loud['typical'] < calib_analyze.FIXED_BAND_RATIO
    assert loud['frame_share'] < 0.5   # by chance on some frames, not every one
    assert calib_analyze.band_kind(loud) == 'loud'
    # w: the loud frame sets the overall peak, a typical frame names channel 5.
    under = bands['w']['channel']
    assert under['argmax'] == 2 and under['frame_top'] == 5
    assert under['typical'] > 100
    assert under['frame_share'] == pytest.approx((BANDS - 1) / BANDS)
    # z is x: inside the graph and on the input, the same numbers.
    assert _same(bands['z'], pool.stats['x']['concentration'])
    assert _scan(art, path, bands=False).layers['bands'] is None


def test_bands_do_not_depend_on_the_block_size(tmp_path, monkeypatch):
    art, path = _bands(tmp_path)
    whole = _scan(art, path)
    monkeypatch.setattr(calib_analyze, 'BLOCK', 7)  # 60 = 8 * 7 + 4
    part = _scan(art, path)
    assert _same(part.layers['bands'], whole.layers['bands'])
    assert _same(part.stats['x']['concentration'],
                 whole.stats['x']['concentration'])


def test_conclusions_name_a_fixed_band_and_not_a_loud_frame(tmp_path,
                                                            monkeypatch):
    art, path = _bands(tmp_path)
    pool = _scan(art, path)
    texts = [item['text'] for item in calib_analyze.analyze(
        pool, count=20)['conclusions']]
    fixed = [text for text in texts if '固定落在' in text]
    assert len(fixed) == 2      # the larger ratio first
    assert fixed[0].startswith('y（Add） 的極端值固定落在第 1 個channel')
    assert fixed[1].startswith('w（Add） 的極端值固定落在第 5 個channel')
    y = pool.layers['bands']['y']['channel']
    assert '一般 frame 裡最大的是中位數的 %.1f 倍' % y['typical'] in fixed[0]
    assert '最大的峰值是中位數的 %.1f 倍' % y['ratio'] in fixed[0]
    assert '16-bit 時中位數的那一個在峰值只剩約 %d 階' % (32767 / y['ratio']) in fixed[0]
    loud = [text for text in texts if '少數 frame 撐出來的' in text]
    assert len(loud) == 2       # the input x, then z inside the graph
    assert loud[0].startswith('x 的峰值') and loud[1].startswith('z（Mul） 的峰值')
    monkeypatch.setattr(calib_analyze, 'LAYER_BAND_LINES', 1)
    texts = [item['text'] for item in calib_analyze.analyze(
        pool, count=20)['conclusions']]
    shown = [text for text in texts if '的極端值固定落在' in text]
    assert len(shown) == 1 and shown[0].startswith('y（Add） ')
    assert any(text.startswith('另有 1 處也固定落在') and 'w（Add） 的第 5 個channel'
               in text for text in texts)


def test_channels_that_take_turns_are_not_called_fixed():
    """8 channels, 90 frames: three channels take turns at 100, the rest stay
    at 1. Each is largest on a third of the frames; none is fixed."""
    x = np.ones((90, 1, 8, 1), dtype=np.float32)
    for frame in range(90):
        x[frame, 0, (0, 3, 5)[frame % 3], 0] = 100.0
    pool = calib_analyze.scan_artifact(calib_analyze.Artifact(
        'x', 'npz', {'x': x}, {}))
    c = pool.stats['x']['concentration']['channel']
    assert calib_analyze.band_kind(c) == 'rotating'
    assert c['top_shares'] == [[0, pytest.approx(1 / 3)],
                               [3, pytest.approx(1 / 3)],
                               [5, pytest.approx(1 / 3)]]
    texts = [text for _, text in calib_analyze._band_lines(
        {'x': pool.stats['x']['concentration']}, (8, 16))]
    assert not any('固定落在' in t for t in texts)
    turns = [t for t in texts if '輪流出現' in t]
    assert len(turns) == 1
    assert '第 0 個（33%）、第 3 個（33%）、第 5 個（33%）' in turns[0]


def test_all_zero_frames_are_left_out_of_the_typical_frame():
    """A fresh state is all zero: it has no largest index of its own."""
    peaks = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                      [1, 10, 1], [1, 1, 10]], dtype=np.float64)
    c = calib_analyze._concentration(peaks.max(axis=0),
                                     *calib_analyze._frame_peaks(peaks))
    assert c['argmax'] == 1 and c['ratio'] == pytest.approx(1.0)
    assert c['typical'] == pytest.approx(10.0)
    assert c['frame_top'] == 1 and c['frame_share'] == 0.5
    assert c['top_shares'] == [[1, 0.5], [2, 0.5]]
    none = calib_analyze._concentration(np.zeros(3),
                                        *calib_analyze._frame_peaks(np.zeros((2, 3))))
    assert none['typical'] == 0.0 and none['frame_share'] == 0.0
    assert none['top_shares'] == []


def test_channels_at_zero_do_not_make_a_band_fixed():
    """A rectifier with most channels off: five channels stay at 0, three are
    ordinary and one frame is loud on channel 6.  The zeros need no steps, so
    they are left out of the medians, and the loud frame stays a loud frame."""
    rng = np.random.default_rng(5)
    peaks = np.zeros((20, 8))
    peaks[:, 5:] = rng.uniform(0.5, 1.0, (20, 3))
    peaks[3, 6] = 100.0
    c = calib_analyze._concentration(peaks.max(axis=0),
                                     *calib_analyze._frame_peaks(peaks))
    assert c['argmax'] == 6 and 90 < c['ratio'] < 200
    assert c['typical'] < calib_analyze.FIXED_BAND_RATIO
    dead = calib_analyze._concentration(np.zeros(4),
                                        *calib_analyze._frame_peaks(np.zeros((3, 4))))
    assert dead['ratio'] == 0.0 and dead['typical'] == 0.0


def test_the_per_frame_median_is_the_median_of_the_non_zero_indices():
    """The per-frame ratio divides by the median of the non-zero indices; it
    is computed by sorting, and must equal the plain NaN-median of them for
    odd and even counts of non-zero indices alike."""
    rng = np.random.default_rng(8)
    peaks = rng.uniform(0.0, 5.0, (200, 9))
    peaks[rng.uniform(size=peaks.shape) < 0.4] = 0.0     # rectified indices
    peaks[:3] = 0.0                                       # fresh states
    ratio, top = calib_analyze._frame_peaks(peaks)
    live = peaks.max(axis=1) > 0
    nonzero = np.where(peaks[live] > 0, peaks[live], np.nan)
    want = peaks[live].max(axis=1) / (np.nanmedian(nonzero, axis=1)
                                      + calib_analyze.EPS)
    assert np.isnan(ratio[~live]).all()
    np.testing.assert_array_equal(ratio[live], want)
    np.testing.assert_array_equal(top, peaks.argmax(axis=1))
    counts = np.count_nonzero(peaks[live] > 0, axis=1)
    assert (counts % 2 == 0).any() and (counts % 2 == 1).any()
