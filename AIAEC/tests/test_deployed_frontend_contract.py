"""Deployment gate: the C lane and the graph metadata against the training front end.

Align-ULCNet is trained on ``linear_error`` from the frozen Python front end
(dataset_gen/linear_aec.py, ``LinearAecContract``) and deployed behind lib/aec's
C AEC. The C model I/O descriptor checks only tensor shapes, grid, D and the
far-input mode, so a graph trained on another front end (a retired v3 corpus,
another behaviour hash, another filter length) with the same tensors passes it.
That shape-only check is NOT sufficient; this file is the gate that is:

* the AEC configuration the deployed mono lane builds
  (pipelines/ulcnet/mono_alignulcnet/main.c ``derive_lane``, compiled into
  tests/deployed_lane_dump.c, not copied) equals the contract on every
  supported grid (frontend_gate.lane_mismatches);
* an exported graph's JSON sidecar carries the compiled
  ULCNET_MODEL_IO_FRONTEND_CONTRACT, the live aec_python_behavior_hash() and
  the sha256 of the graph bytes (frontend_gate.metadata_mismatches);
* each check reports a retired hash, a v3 contract, a wrong filter length,
  grid or flag, and a swapped graph or checkpoint.

The board sample outside this repository builds its lane separately; it is
covered only if its derive_lane is the example's. test_export_streaming_ulcnet.py
pins the header define to LINEAR_AEC_CONTRACT_VERSION and the metadata to the
checkpoint contract; this file does not repeat those.
"""

import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess

import pytest
import torch

from AIAEC.Align_ULCNet.export_onnx import export_graph
from AIAEC.Align_ULCNet.frontend_gate import (
    CONTRACT_FIELDS,
    lane_mismatches,
    metadata_mismatches,
    parse_lane_dump,
)
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen.aec_behavior_hash import aec_python_behavior_hash
from AIAEC.dataset_gen.linear_aec import (
    LinearAecContract,
    make_linear_aec_contract,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_ALG_DIR = os.path.dirname(_AIAEC_DIR)
_ULCNET_DIR = os.path.join(_AIAEC_DIR, 'Align_ULCNet')
_PIPELINES_DIR = os.path.join(_ALG_DIR, 'pipelines')
_LANE_MAIN = os.path.join(_PIPELINES_DIR, 'ulcnet', 'mono_alignulcnet', 'main.c')
_AEC_C_DIR = os.path.join(_ALG_DIR, 'lib', 'aec', 'c_impl')
_AC_DIR = os.path.abspath(os.path.join(_ALG_DIR, '..', 'audiolib', 'common'))

GRIDS = {16000: 512, 48000: 1024}

# Retired identities, spelled out rather than read from the tables the gate
# uses (a check fed from its own table passes when the table is emptied).
RETIRED_VERSION = 'aiaec-linear-error-v3'
PREVIOUS_FRONTEND_HASH = (
    '9fa7a2dec85cef8f123cc4dd02e7eb3c5dc0c1b45d7ec0f32b766cf9cda245e5')


def _extra_cflags():
    """EXTRA_CFLAGS from the environment, forwarded like pipelines/Makefile."""
    return shlex.split(os.environ.get('EXTRA_CFLAGS', ''))


def _producer_lib(directory, *args):
    extra = ['EXTRA_CFLAGS=' + os.environ['EXTRA_CFLAGS']] \
        if os.environ.get('EXTRA_CFLAGS') else []
    subprocess.run(['make', '-s', '-C', directory, *args, *extra, 'lib'],
                   check=True, capture_output=True)
    out = subprocess.run(
        ['make', '-s', '-C', directory, *args, *extra, 'print-lib-path'],
        check=True, capture_output=True, text=True)
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


@pytest.fixture(scope='module')
def producer_libs():
    if shutil.which('make') is None:
        pytest.skip('no make available')
    return (_producer_lib(_AEC_C_DIR),
            _producer_lib(_AC_DIR, 'BACKEND=ne10'))


@pytest.fixture(scope='module')
def lanes(tmp_path_factory, producer_libs):
    """The deployed lane's dump on each supported grid."""
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    sources = [
        os.path.join(_THIS_DIR, 'deployed_lane_dump.c'),
        os.path.join(_PIPELINES_DIR, 'ulcnet', 'ulcnet_nn_stage.c'),
        *(os.path.join(_ULCNET_DIR, name) for name in (
            'ulcnet_accelerator_adapter.c', 'ulcnet_prepost.c',
            'ulcnet_process.c', 'ulcnet_model_io.c')),
    ]
    includes = [_PIPELINES_DIR, os.path.join(_PIPELINES_DIR, 'ulcnet'),
                _ULCNET_DIR, os.path.join(_AEC_C_DIR, 'include'),
                os.path.join(_AEC_C_DIR, 'example'),
                os.path.join(_AC_DIR, 'include'), _AC_DIR]
    dumps = {}
    for sample_rate, n_fft in GRIDS.items():
        exe = tmp_path_factory.mktemp('lane%d' % sample_rate) / 'dump'
        subprocess.run(
            [cc, '-O2', '-std=gnu99', '-ffp-contract=off', *_extra_cflags(),
             '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
             '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
             *('-I' + path for path in includes), *sources, *producer_libs,
             '-lm', '-o', str(exe)],
            check=True, capture_output=True)
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        dumps[sample_rate] = parse_lane_dump(run.stdout)
    return dumps


@pytest.fixture(scope='module')
def contracts():
    return {sr: make_linear_aec_contract(sr) for sr in GRIDS}


def test_every_contract_field_is_mapped():
    """A new LinearAecContract field fails here until the gate maps it."""
    assert set(CONTRACT_FIELDS) == {
        field.name for field in dataclasses.fields(LinearAecContract)}


@pytest.mark.parametrize('sample_rate', sorted(GRIDS))
def test_deployed_lane_matches_the_contract(lanes, contracts, sample_rate):
    lane = lanes[sample_rate]
    assert lane_mismatches(lane, contracts[sample_rate]) == []
    # What makes the lane's inherited enable_cng = 1 acceptable, measured on
    # the deployed lane rather than assumed.
    assert lane['formed_hop_cng_invariant'] == '1'


def test_deployed_stage_reads_the_formed_seam():
    """output_seam is the formed hop only if main.c hands that to the stage."""
    with open(_LANE_MAIN, encoding='utf-8') as stream:
        source = stream.read()
    assert re.search(r'ulcnet_nn_stage_begin_time\(\s*stage,\s*rctx\.formed_hop,',
                     source)


@pytest.mark.parametrize('key, value, field', [
    ('frontend_contract', RETIRED_VERSION, 'version'),
    ('filter_length', '1024', 'filter_length'),
    ('n_partitions', '5', 'n_partitions'),
    ('enable_shadow', '0', 'enable_shadow'),
    ('enable_res', '1', 'enable_res'),
    ('enable_highpass', '0', 'enable_highpass'),
    ('enable_saturation', '0', 'enable_saturation'),
    ('spatial_linear_context', '0', 'LINEAR_ONLY'),
    ('return_res_context', '0', 'return_res_context'),
    ('formed_hop_cng_invariant', '0', 'enable_cng'),
    ('enable_delay_est', '0', 'enable_delay_est'),
    ('delay_mode', '1', 'delay_mode MATCHED'),
    ('delay_num_filters', '1', 'delay_num_filters'),
    ('delay_analysis_every_hop', '0', 'delay_analysis_every_hop'),
    ('live_observer_enabled', '1', 'live_observer_enabled'),
    ('min_gain_floor_far_active_db', '-38', 'preset'),
])
def test_lane_gate_refuses_a_changed_field(lanes, contracts, key, value, field):
    lane = dict(lanes[16000], **{key: value})
    mismatches = lane_mismatches(lane, contracts[16000])
    assert any(field in item for item in mismatches), mismatches


def test_lane_gate_refuses_cng_off_the_analysis_lane(lanes, contracts):
    """enable_cng = 1 is accepted only on a LINEAR_ONLY lane."""
    lane = dict(lanes[16000], spatial_linear_context='1')
    mismatches = lane_mismatches(lane, contracts[16000])
    assert any(item.startswith('enable_cng') for item in mismatches), mismatches


def test_lane_gate_refuses_another_grid(lanes, contracts):
    mismatches = lane_mismatches(lanes[48000], contracts[16000])
    for field in ('sample_rate', 'frame_size', 'hop_size', 'filter_length'):
        assert any(item.startswith(field) for item in mismatches), mismatches


def test_lane_gate_refuses_a_model_trained_on_another_filter_length(lanes):
    """The same tensors, another accepted front end: only the gate tells."""
    custom = make_linear_aec_contract(16000, filter_length=1024)
    mismatches = lane_mismatches(lanes[16000], custom)
    assert any(item.startswith('filter_length') for item in mismatches)


@pytest.fixture(scope='module')
def exported(tmp_path_factory, lanes):
    pytest.importorskip('onnx')
    sample_rate = 16000
    grid = SignalGrid(sample_rate, GRIDS[sample_rate], GRIDS[sample_rate],
                      GRIDS[sample_rate] // 2)
    model = AlignULCNet(grid, max_delay_frames=4).eval()
    linear = make_linear_aec_contract(sample_rate)
    work = tmp_path_factory.mktemp('graph')
    checkpoint = work / 'ckpt.pth'
    torch.save({'contract': {'linear_aec': linear.as_dict(),
                             'linear_aec_contract_hash': linear.fingerprint()},
                'state_dict': model.state_dict()}, checkpoint)
    graph = work / 'model.onnx'
    export_graph(model, str(checkpoint), str(graph))
    sidecar = json.loads(graph.with_suffix('.json').read_text(encoding='utf-8'))
    return graph, checkpoint, sidecar, lanes[sample_rate]


def _metadata_gate(exported, **override):
    graph, checkpoint, sidecar, lane = exported
    kwargs = dict(frontend_contract=lane['frontend_contract'],
                  behavior_hash=aec_python_behavior_hash(),
                  graph_path=str(graph), lane=lane,
                  checkpoint_path=str(checkpoint))
    # A None override removes the key.
    metadata = {key: value for key, value in
                dict(sidecar, **override.pop('metadata', {})).items()
                if value is not None}
    kwargs.update(override)
    return metadata_mismatches(metadata, **kwargs)


def test_exported_graph_matches_the_deployment(exported):
    _graph, _checkpoint, sidecar, lane = exported
    assert sidecar['linear_aec_contract_version'] == lane['frontend_contract']
    assert sidecar['aec_behavior_hash'] == aec_python_behavior_hash()
    assert _metadata_gate(exported) == []


@pytest.mark.parametrize('metadata, field', [
    ({'linear_aec_contract_version': RETIRED_VERSION}, 'retired'),
    ({'aec_behavior_hash': PREVIOUS_FRONTEND_HASH}, 'retired'),
    ({'aec_behavior_hash': 'ab' * 32}, 'aec_behavior_hash'),
    ({'error_input_seam': 'rendering_base'}, 'error_input_seam'),
    ({'onnx_sha256': None}, 'onnx_sha256 missing'),
    ({'onnx_sha256': '00' * 32}, 'onnx_sha256'),
    ({'checkpoint_sha256': '00' * 32}, 'checkpoint_sha256'),
    ({'n_fft': 1024}, 'n_fft'),
])
def test_metadata_gate_refuses_a_changed_field(exported, metadata, field):
    mismatches = _metadata_gate(exported, metadata=metadata)
    assert any(field in item for item in mismatches), mismatches


def test_metadata_gate_refuses_a_swapped_graph(exported, tmp_path):
    graph = exported[0]
    swapped = tmp_path / 'swapped.onnx'
    data = bytearray(graph.read_bytes())
    data[-1] ^= 1
    swapped.write_bytes(bytes(data))
    mismatches = _metadata_gate(exported, graph_path=str(swapped))
    assert any(item.startswith('onnx_sha256') for item in mismatches)


def test_metadata_gate_refuses_another_build(exported):
    mismatches = _metadata_gate(exported, frontend_contract=RETIRED_VERSION)
    assert any('linear_aec_contract_version' in item for item in mismatches)
