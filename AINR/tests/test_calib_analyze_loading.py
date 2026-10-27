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
