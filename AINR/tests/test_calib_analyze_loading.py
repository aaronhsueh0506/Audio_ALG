"""``PTQ/calib_analyze.py`` reads the report that belongs to an NPZ archive.

A calibration run exports ``<stem>.onnx`` next to ``<stem>.npz``, so
``<stem>.json`` is the graph's sidecar and the report is ``<stem>.calib.json``.
"""

import json
import os
import sys

import numpy as np

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


# --- an Align-ULCNet artifact must come from the current AEC front end -------

import pytest  # noqa: E402

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


def test_cli_analyze_select_and_apply_refuse_a_retired_artifact(tmp_path):
    output = _ulcnet_artifact(tmp_path, **_refusals()['retired hash'])
    keep = tmp_path / 'keep.json'
    keep.write_text(json.dumps({'keep_frames': [0, 1, 2]}))
    out = str(tmp_path / 'out.npz')
    for argv in (['analyze', str(output), '--out', str(tmp_path / 'r.html')],
                 ['select', str(output), '--count', '4'],
                 ['apply', str(output), '--keep', str(keep),
                  '--output', out]):
        with pytest.raises(ValueError, match='retired'):
            calib_analyze.main(argv)
    assert not os.path.exists(out)


def test_apply_selection_refuses_a_stale_stamp_on_a_loaded_artifact(tmp_path):
    # An Artifact built another way (or edited after loading) is checked again
    # at the write, where the stamp would be copied onto a new artifact.
    art = calib_analyze.load_artifact(str(_ulcnet_artifact(
        tmp_path, **_current())))
    art.report['aec_behavior_hash'] = sorted(RETIRED_BEHAVIOR_HASHES)[0]
    with pytest.raises(ValueError, match='retired'):
        calib_analyze.apply_selection(
            art, np.arange(4), str(tmp_path / 'o.npz'), 'npz')


@pytest.mark.parametrize('fmt', ['npz', 'bin'])
def test_current_stamp_loads_and_survives_apply(tmp_path, fmt):
    output = _ulcnet_artifact(tmp_path, fmt, **_current())
    art = calib_analyze.load_artifact(str(output))
    target = tmp_path / ('pruned.npz' if fmt == 'npz' else 'pruned_bin')
    calib_analyze.apply_selection(art, np.arange(5), str(target), fmt)
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
