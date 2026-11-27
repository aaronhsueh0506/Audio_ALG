"""``PTQ/calib_analyze.py`` reads an artifact and the report beside it.

A calibration run exports ``<stem>.onnx`` next to ``<stem>.npz``, so
``<stem>.json`` is the graph's sidecar and the report is ``<stem>.calib.json``.
A BIN directory needs no report: the graph gives each input's shape and type.
"""

import json
import os
import sys

import numpy as np
import pytest

from calibration_io import write_calibration_artifact

_PTQ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'PTQ')
if _PTQ not in sys.path:
    sys.path.insert(0, _PTQ)

import calib_analyze  # noqa: E402


def _archive(tmp_path):
    arrays = {'feature': np.arange(6, dtype=np.float32).reshape(2, 1, 3)}
    output = tmp_path / 'foo.npz'
    write_calibration_artifact(
        output, arrays, {'frames': 2, 'seed': 7}, 'npz'
    )
    return output


def test_report_comes_from_its_own_sidecar_not_the_graph_sidecar(tmp_path):
    output = _archive(tmp_path)
    (tmp_path / 'foo.json').write_text(json.dumps({'input_schema': {}}))
    art = calib_analyze.load_artifact(str(output))
    assert art.report['seed'] == 7
    assert 'input_schema' not in art.report


def test_archive_with_the_report_under_the_old_name_still_loads(tmp_path):
    output = _archive(tmp_path)
    (tmp_path / 'foo.calib.json').rename(tmp_path / 'foo.json')
    art = calib_analyze.load_artifact(str(output))
    assert art.report['seed'] == 7


def test_graph_sidecar_alone_is_not_taken_for_a_report(tmp_path):
    output = _archive(tmp_path)
    (tmp_path / 'foo.calib.json').unlink()
    (tmp_path / 'foo.json').write_text(json.dumps({'input_schema': {}}))
    assert calib_analyze.load_artifact(str(output)).report == {}


# --- BIN directories: the graph gives the shapes -----------------------------

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper  # noqa: E402


def _graph(path, inputs):
    """A graph taking ``inputs`` (name -> (shape, onnx type)); each float
    input goes through a Relu."""
    nodes, outs = [], []
    for name, (shape, kind) in inputs.items():
        if kind == TensorProto.FLOAT:
            nodes.append(helper.make_node('Relu', [name], [name + '_r']))
            outs.append(helper.make_tensor_value_info(name + '_r', kind, shape))
    model = helper.make_model(helper.make_graph(
        nodes, 'g', [helper.make_tensor_value_info(name, kind, shape)
                     for name, (shape, kind) in inputs.items()], outs),
        opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def _bin(tmp_path, frames=12, name='cal'):
    rng = np.random.default_rng(1)
    arrays = {'feature': rng.standard_normal((frames, 1, 4)).astype(np.float32),
              'step': np.arange(frames, dtype=np.int64).reshape(frames, 1)}
    out = tmp_path / name
    write_calibration_artifact(out, arrays, {'frames': frames, 'seed': 3}, 'bin')
    _graph(tmp_path / (name + '.onnx'), {'feature': ([1, 4], TensorProto.FLOAT),
                                        'step': ([1], TensorProto.INT64)})
    return out, arrays


def test_a_bin_directory_without_a_manifest_reads_its_shapes_from_the_graph(
        tmp_path):
    out, arrays = _bin(tmp_path)
    (out / 'manifest.json').unlink()
    # File names another tool wrote: not zero-padded, so 10 sorts after 2.
    for name in arrays:
        for index in range(len(arrays[name])):
            os.rename(out / name / ('%s_%04d.bin' % (name, index)),
                      out / name / ('%d.bin' % index))
    art = calib_analyze.load_artifact(str(out))
    assert art.report == {} and art.fmt == 'bin'
    for name, value in arrays.items():
        np.testing.assert_array_equal(art.arrays[name], value)
    assert art.skipped == {'step': 'non-float dtype int64'}
    pool = calib_analyze.scan_artifact(art)
    texts = [c['text'] for c in calib_analyze.analyze(pool)['conclusions']]
    assert any(t.startswith('沒有校正報告') and '哪些輸入不量化' in t
               for t in texts)


def test_a_bin_directory_with_a_manifest_still_reads_it(tmp_path):
    out, arrays = _bin(tmp_path)
    art = calib_analyze.load_artifact(str(out))
    assert art.report['seed'] == 3
    np.testing.assert_array_equal(art.arrays['feature'], arrays['feature'])
    texts = [c['text'] for c in calib_analyze.analyze(
        calib_analyze.scan_artifact(art))['conclusions']]
    assert not any(t.startswith('沒有校正報告') for t in texts)


def test_a_bin_directory_the_graph_does_not_describe_is_refused(tmp_path):
    out, arrays = _bin(tmp_path)
    os.remove(str(out) + '.onnx')
    # No graph: the shapes manifest.json records; with neither, a refusal.
    np.testing.assert_array_equal(
        calib_analyze.load_artifact(str(out)).arrays['feature'],
        arrays['feature'])
    os.rename(out / 'manifest.json', tmp_path / 'manifest.json')
    with pytest.raises(ValueError, match='record no shape.*--onnx PATH'):
        calib_analyze.load_artifact(str(out))
    os.rename(tmp_path / 'manifest.json', out / 'manifest.json')
    wide = _graph(tmp_path / 'wide.onnx', {
        'feature': ([1, 5], TensorProto.FLOAT), 'step': ([1], TensorProto.INT64)})
    with pytest.raises(ValueError, match='manifest.json records feature'):
        calib_analyze.load_artifact(str(out), wide)
    (out / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='is 16 bytes; feature float32 \\[1, 5\\]'):
        calib_analyze.load_artifact(str(out), wide)
    extra = _graph(tmp_path / 'extra.onnx', {
        'feature': ([1, 4], TensorProto.FLOAT), 'step': ([1], TensorProto.INT64),
        'other': ([1, 4], TensorProto.FLOAT)})
    with pytest.raises(ValueError, match='no folder for graph input other'):
        calib_analyze.load_artifact(str(out), extra)
    good = _graph(tmp_path / 'good.onnx', {
        'feature': ([1, 4], TensorProto.FLOAT), 'step': ([1], TensorProto.INT64)})
    assert calib_analyze.load_artifact(str(out), good).frames == 12
    os.remove(out / 'step' / 'step_0003.bin')
    with pytest.raises(ValueError, match='different frame counts'):
        calib_analyze.load_artifact(str(out), good)


def test_a_bin_directory_is_read_the_same_on_more_threads(tmp_path):
    out, _ = _bin(tmp_path, frames=600)
    one = calib_analyze.load_artifact(str(out))
    four = calib_analyze.load_artifact(str(out), threads=4)
    for name in one.arrays:
        np.testing.assert_array_equal(one.arrays[name], four.arrays[name])


# --- an Align-ULCNet artifact must come from the current AEC front end -------

from AIAEC.dataset_gen.linear_aec import (  # noqa: E402
    LINEAR_AEC_CONTRACT_VERSION,
    RETIRED_BEHAVIOR_HASHES,
    RETIRED_CONTRACT_VERSIONS,
)
from AIAEC.dataset_gen.aec_behavior_hash import (  # noqa: E402
    aec_python_behavior_hash,
)


def _ulcnet_artifact(tmp_path, fmt='npz', **stamp):
    rng = np.random.default_rng(0)
    arrays = {'feature': rng.standard_normal((8, 1, 4)).astype(np.float32)}
    report = {'frames': 8, 'model_family': 'Align_ULCNet'}
    report.update(stamp)
    output = tmp_path / ('ulc.npz' if fmt == 'npz' else 'ulc_bin')
    write_calibration_artifact(output, arrays, report, fmt)
    _graph(tmp_path / ('ulc.onnx' if fmt == 'npz' else 'ulc_bin.onnx'),
           {'feature': ([1, 4], TensorProto.FLOAT)})
    return output


def _current():
    return dict(linear_aec_contract_version=LINEAR_AEC_CONTRACT_VERSION,
                aec_behavior_hash=aec_python_behavior_hash())


def _refusals():
    retired = sorted(RETIRED_BEHAVIOR_HASHES)[0]
    return {
        'no stamp': {},
        'no hash': dict(
            linear_aec_contract_version=LINEAR_AEC_CONTRACT_VERSION),
        'retired hash': dict(_current(), aec_behavior_hash=retired),
        'retired version': dict(
            _current(),
            linear_aec_contract_version=sorted(RETIRED_CONTRACT_VERSIONS)[0]),
    }


@pytest.mark.parametrize('fmt', ['npz', 'bin'])
@pytest.mark.parametrize('case', sorted(_refusals()))
def test_ulcnet_artifact_from_another_front_end_is_refused(
        tmp_path, fmt, case):
    output = _ulcnet_artifact(tmp_path, fmt, **_refusals()[case])
    with pytest.raises(ValueError, match='front-end|front end|retired'):
        calib_analyze.load_artifact(str(output))


def test_cli_analyze_refuses_a_retired_artifact(tmp_path):
    output = _ulcnet_artifact(tmp_path, **_refusals()['retired hash'])
    out = str(tmp_path / 'out.npz')
    for argv in (['analyze', str(output), '--out', str(tmp_path / 'r.html')],
                 ['analyze', str(output), '--count', '4', '--output', out]):
        with pytest.raises(SystemExit):
            calib_analyze.main(argv)
    assert not os.path.exists(out)


def test_write_selection_refuses_a_stale_stamp_on_a_loaded_artifact(tmp_path):
    # A pool built another way (or edited after loading) is checked again at
    # the write, where the stamp would be copied onto a new artifact.
    art = calib_analyze.load_artifact(str(_ulcnet_artifact(
        tmp_path, **_current())))
    pool = calib_analyze.scan_artifact(art)
    pool.report['aec_behavior_hash'] = sorted(RETIRED_BEHAVIOR_HASHES)[0]
    with pytest.raises(ValueError, match='retired'):
        calib_analyze.write_selection(
            pool, np.arange(4), str(tmp_path / 'o.npz'), 'npz')
    assert not (tmp_path / 'o.npz').exists()


@pytest.mark.parametrize('fmt', ['npz', 'bin'])
def test_current_stamp_loads_and_survives_a_selection(tmp_path, fmt):
    output = _ulcnet_artifact(tmp_path, fmt, **_current())
    pool = calib_analyze.scan_artifact(calib_analyze.load_artifact(str(output)))
    target = tmp_path / ('pruned.npz' if fmt == 'npz' else 'pruned_bin')
    calib_analyze.write_selection(pool, np.arange(5), str(target), fmt)
    if fmt == 'bin':
        _graph(str(target) + '.onnx', {'feature': ([1, 4], TensorProto.FLOAT)})
    kept = calib_analyze.load_artifact(str(target))
    assert kept.frames == 5
    for key, value in _current().items():
        assert kept.report[key] == value


def test_other_families_need_no_front_end_stamp(tmp_path):
    arrays = {'feature': np.zeros((4, 1, 3), np.float32)}
    for family in ('DeepFilterNet2', 'Align_CRUSE'):
        output = tmp_path / ('%s.npz' % family)
        write_calibration_artifact(
            output, arrays, {'frames': 4, 'model_family': family}, 'npz')
        assert calib_analyze.load_artifact(str(output)).frames == 4
