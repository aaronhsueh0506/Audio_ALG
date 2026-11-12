"""The AIAEC stream's work views equal what the exporters write.

The signal and head tensors of every block are the AiaecWork views a real aiaec_preprocess*() call returns;
the recurrent state belongs to the accelerator, so the library holds no port table. This test compares those
views with a fresh export of the same configuration. Every converted graph follows one order: inputs are the
signal inputs, then the state inputs; outputs are the head outputs, then the state outputs; state k is input
signal_in + k <-> output head_out + k. The ONNX names are read on the exporter side only, to prove the exporter follows that order (Align-ULCNet at N > 1 has frame_reset as the last signal
input). The graph rows (stream/aiaec/ai3_model_io_{deepvqes,alignulcnet}.h) are pinned below by the same exports.

A tiny C dumper (stream/aiaec/tests/dump_bind_info.c, built once per product by
``make -C stream/aiaec bind-dumpers`` with -DAIAEC_TEST_CATALOG so the rows without a delivered graph still
resolve) prints the AiaecInfo of a real init and the AiaecWork of the first real pre call for each family
(PCM, PCM -> spectrum, spectrum -> spectrum). Random-weight graphs are exported at DeepVQE-S 16k/256 N=1
and N=8, DeepVQE-S 16k/512 N=1, Align-ULCNet 16k/256 N=1 and N=8 (packet graph, layout 19) and 16k/512 N=1.
The dumped signal inputs and head output are compared with the exported graph's leading inputs and outputs
by position and exact padded shape, for all three families, and the three families must print one and the
same views and the same trim / queue / tail. Align-ULCNet's frame_reset values and the BYPASS block of an
N = 1 event hop are pinned as well.

Set AIAEC_PIN_DUMPERS to ``<deepvqe_s dumper><pathsep><align_ulcnet dumper>`` to compare other
builds (the mutation check of this test) instead of building the repository's.

The graph rows are compared with the same exports: ``make -C stream/aiaec model-io-dumper`` prints every row of
the two model-config headers (AIAEC_MODEL_IO_DUMPER compares another build), and each row must equal the fresh
export: layout version, port counts and order, every input and output shape, the state names and their order, each
state's shape at both ends, the state output name ``<input>_out`` and frame_reset as the last signal (Align-ULCNet
N > 1). The row set must be exactly the exported cases.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet import export_onnx as align_export
from AIAEC.DeepVQE_S import DeepVQES
from AIAEC._streaming_export import _build, export_graph as deepvqe_export_graph
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, make_linear_aec_contract

REPO = Path(__file__).resolve().parents[2]
STREAM_AIAEC = REPO / 'stream' / 'aiaec'

# (model, sr, fft, N) -> hop_in is N * fft / 2
CASES = [
    ('deepvqe_s', 16000, 256, 1),
    ('deepvqe_s', 16000, 256, 8),
    ('deepvqe_s', 16000, 512, 1),
    ('align_ulcnet', 16000, 256, 1),
    ('align_ulcnet', 16000, 256, 8),
    ('align_ulcnet', 16000, 512, 1),
]


def signal_names(model_name, n):
    """The signal inputs in graph order: DeepVQE-S mic, far; Align-ULCNet the four features, then frame_reset
    for the packet graph (N > 1)."""
    if model_name == 'deepvqe_s':
        return ['mic', 'far']
    names = ['error_mag', 'far_mag', 'error_cos', 'error_sin']
    return names + ['frame_reset'] if n > 1 else names


def expected_trim_queue(model_name, n, hop):
    """(trim_samples, queue_frames), the same for all three families: DeepVQE-S N * hop and N - 1 frames;
    Align-ULCNet hop and no queue."""
    return (hop, 0) if model_name == 'align_ulcnet' else (n * hop, n - 1)


def _case_id(case):
    model, _sr, fft, n = case
    return '%s-fft%d-N%d' % (model, fft, n)


@pytest.fixture(scope='module')
def dumpers():
    override = os.environ.get('AIAEC_PIN_DUMPERS')
    if override:
        deepvqe_s, align_ulcnet = override.split(os.pathsep)
        return {'deepvqe_s': deepvqe_s, 'align_ulcnet': align_ulcnet}
    if shutil.which('make') is None or shutil.which('cc') is None:
        pytest.skip('no make / C compiler available')
    out = subprocess.run(['make', '-s', '-C', str(STREAM_AIAEC), 'bind-dumpers'],
                         check=True, capture_output=True, text=True)
    paths = out.stdout.strip().splitlines()[-2:]
    assert all(os.path.isfile(p) for p in paths), paths
    return {'deepvqe_s': paths[0], 'align_ulcnet': paths[1]}


FAMILIES = (0, 1, 2)   # AiaecFamily: PCM, PCM -> spectrum, spectrum -> spectrum


def run_dumper(path, sr, fft, hop_in, family=0, event_hop=None):
    """The dumper's init result and the first block's AiaecWork as a dict; views as (b, c, h, w) shapes with
    their element counts. N is not printed: it is hop_in / (fft / 2)."""
    argv = [path, str(sr), str(fft), str(hop_in), str(family)]
    if event_hop is not None:
        argv.append(str(event_hop))
    text = subprocess.run(argv, check=True, capture_output=True, text=True).stdout
    dump = {'in': [], 'out': []}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == 'rc':
            dump['rc'] = int(fields[1])
        elif fields[0] == 'graph':
            dump['graph'] = fields[1]
        elif fields[0] == 'info':
            dump.update(trim=int(fields[2]), queue=int(fields[4]), tail=int(fields[6]))
        elif fields[0] == 'work':
            dump.update(action=int(fields[2]), inputs=int(fields[4]), outputs=int(fields[6]))
        elif fields[0] in ('in', 'out'):
            _kind, index, b, c, h, w, elements = fields
            assert int(index) == len(dump[fields[0]])
            dump[fields[0]].append(((int(b), int(c), int(h), int(w)), int(elements)))
        elif fields[0] == 'frame_reset':
            dump['frame_reset'] = [float(v) for v in fields[1:]]
    return dump


def _dims(value):
    dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
    assert 0 < len(dims) <= 4, (value.name, dims)
    return tuple([1] * (4 - len(dims)) + dims)


def _pad(shape):
    shape = tuple(int(s) for s in shape)
    assert 0 < len(shape) <= 4, shape
    return tuple([1] * (4 - len(shape)) + list(shape))


def delay_depth(sr, fft):
    """The C catalog's rule (AIAEC_MAX_DELAY_SECONDS = 0.5): ceil(0.5 s / hop). Align-ULCNet's own
    default at 16 kHz / FFT 512 is 64 (the paper's value), so its export must pass this depth
    explicitly (the exporter's --max-delay-frames), as the deployed 16 kHz / 512 graph does."""
    return int(np.ceil(0.5 * sr / (fft // 2)))


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    """case -> (ONNX graph, metadata dict), random weights, the catalog's alignment depth."""
    onnx = pytest.importorskip('onnx')
    work = tmp_path_factory.mktemp('aiaec_pin')
    result = {}
    for case in CASES:
        model_name, sr, fft, n = case
        grid = SignalGrid(sr, fft, fft, fft // 2)
        path = work / (_case_id(case) + '.onnx')
        torch.manual_seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            if model_name == 'deepvqe_s':
                model = DeepVQES(grid, max_delay_seconds=0.5).eval()
                assert model.max_delay_frames == delay_depth(sr, fft)
                checkpoint = work / (_case_id(case) + '.pth')
                checkpoint.write_bytes(b'random-weights')
                built = _build('DeepVQE_S', model, n)
                metadata = deepvqe_export_graph(
                    AecGrid(sr, fft, fft, fft // 2), built, str(checkpoint), str(path),
                    model.max_delay_frames, verify=False, frames_per_call=n)
            else:
                model = AlignULCNet(grid, max_delay_frames=delay_depth(sr, fft)).eval()
                linear = make_linear_aec_contract(sr, frame_size=fft)
                checkpoint = work / (_case_id(case) + '.pth')
                torch.save({'contract': {'linear_aec': linear.as_dict(),
                                         'linear_aec_contract_hash': linear.fingerprint()},
                            'state_dict': model.state_dict()}, checkpoint)
                _wrapper, _inputs, metadata = align_export.export_graph(
                    model, str(checkpoint), str(path), verify=False, frames_per_call=n)
        result[case] = (onnx.load(str(path)).graph, metadata)
    return result


@pytest.mark.parametrize('family', FAMILIES, ids=('pcm', 'spec', 'freq'))
@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_c_work_views_equal_the_exported_graph(case, family, dumpers, exports):
    model_name, sr, fft, n = case
    hop = fft // 2
    graph, metadata = exports[case]
    dump = run_dumper(dumpers[model_name], sr, fft, n * hop, family)

    assert dump['rc'] == 0
    want_trim, want_queue = expected_trim_queue(model_name, n, hop)
    assert (dump['trim'], dump['queue'], dump['tail']) == (want_trim, want_queue, 1)
    assert dump['action'] == 1, 'a quiet block is a SUBMIT (no event)'

    # exporter order (names are read only here): signal inputs, then states; heads, then states; the state
    # output of an input is "<input>_out"
    in_names = [v.name for v in graph.input]
    out_names = [v.name for v in graph.output]
    n_signal = len(signal_names(model_name, n))
    assert in_names[:n_signal] == signal_names(model_name, n)
    assert not set(in_names[n_signal:]) & set(signal_names(model_name, 8)), 'a signal name among the states'
    assert out_names[0] == 'output' and len(out_names) - 1 == len(in_names) - n_signal
    assert out_names[1:] == [name + '_out' for name in in_names[n_signal:]]

    # the exporter's own metadata says the same, in the same order
    onnx_in = [_dims(v) for v in graph.input]
    onnx_out = [_dims(v) for v in graph.output]
    assert [_pad(v) for v in metadata['input_schema'].values()] == onnx_in
    assert [_pad(v) for v in metadata['output_schema'].values()] == onnx_out
    assert list(metadata['input_schema']) == in_names and list(metadata['output_schema']) == out_names

    # the work holds the signals and the one head by position; the states are not in it
    assert (dump['inputs'], dump['outputs']) == (n_signal, 1)
    assert [p[0] for p in dump['in']] == onnx_in[:n_signal], 'signal views differ from the exported graph'
    assert [p[0] for p in dump['out']] == onnx_out[:1], 'head view differs from the exported graph'
    for ports in (dump['in'], dump['out']):
        for i, (shape, elements) in enumerate(ports):
            assert elements == shape[0] * shape[1] * shape[2] * shape[3], i

    # state k pairs input signal_in + k with output head_out + k: same padded shape at both ends (exporter side)
    for k in range(len(in_names) - n_signal):
        assert onnx_in[n_signal + k] == onnx_out[1 + k], 'state %d' % k

    if model_name == 'align_ulcnet' and n > 1:
        assert dump['frame_reset'] == [0.0] * n, 'no event: every control value is 0'


def test_the_customer_arithmetic_is_pinned_to_the_exported_n(dumpers, exports):
    """hop_in 128 is N=1, 1024 is N=8 and 129 is refused at 16 kHz / FFT 256; the N it derives is
    the N the graph was exported for (the input time axis of the DeepVQE-S graph)."""
    dv = dumpers['deepvqe_s']
    for hop_in, n in ((128, 1), (1024, 8)):
        dump = run_dumper(dv, 16000, 256, hop_in)
        graph, _meta = exports[('deepvqe_s', 16000, 256, n)]
        assert dump['rc'] == 0
        assert dump['in'][0][0][2] == n == hop_in // (256 // 2) == _dims(graph.input[0])[2]
    assert run_dumper(dv, 16000, 256, 129)['rc'] == -4          # AIAEC_ERR_IN_HOP
    assert run_dumper(dumpers['align_ulcnet'], 16000, 256, 129)['rc'] == -4
    # Align-ULCNet: N=8 is the packet graph; its time axis is the N it was exported for
    dump = run_dumper(dumpers['align_ulcnet'], 16000, 256, 1024)
    graph, _meta = exports[('align_ulcnet', 16000, 256, 8)]
    assert dump['rc'] == 0 and dump['in'][0][0][2] == 8 == _dims(graph.input[0])[2]
    assert dump['inputs'] == 5 and dump['in'][4][0] == (1, 1, 1, 8)        # frame_reset [1, N], the last signal


@pytest.mark.parametrize('family', FAMILIES, ids=('pcm', 'spec', 'freq'))
def test_align_event_hop_3_of_an_n8_packet_puts_frame_reset_00100000_in_the_fifth_signal(family, dumpers, exports):
    """The stream, not the graph or the sample, derives the control values: model hop 2 (the third) marked
    as an alignment change gives 0 0 1 0 0 0 0 0, still one SUBMIT with five signals (the packet graph)."""
    dump = run_dumper(dumpers['align_ulcnet'], 16000, 256, 1024, family, event_hop=2)
    graph, _meta = exports[('align_ulcnet', 16000, 256, 8)]
    assert dump['rc'] == 0 and dump['action'] == 1 and dump['inputs'] == 5
    assert dump['frame_reset'] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [p[0] for p in dump['in']] == [_dims(v) for v in graph.input][:5]


@pytest.mark.parametrize('family', FAMILIES, ids=('pcm', 'spec', 'freq'))
def test_align_event_hop_of_an_n1_block_is_a_bypass_with_no_views(family, dumpers):
    """N = 1 (layout 18) has no frame_reset: an event hop is a BYPASS block with no accelerator job, so the
    sample binds and submits nothing, yet still calls post once."""
    quiet = run_dumper(dumpers['align_ulcnet'], 16000, 256, 128, family)
    event = run_dumper(dumpers['align_ulcnet'], 16000, 256, 128, family, event_hop=0)
    assert quiet['rc'] == event['rc'] == 0
    assert quiet['action'] == 1 and quiet['inputs'] == 4 and quiet['outputs'] == 1
    assert event['action'] == 0 and event['inputs'] == 0 and event['outputs'] == 0
    assert event['in'] == [] and event['out'] == [] and 'frame_reset' not in event


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_the_three_families_share_one_set_of_views(case, dumpers):
    model_name, sr, fft, n = case
    dumps = [run_dumper(dumpers[model_name], sr, fft, n * (fft // 2), f) for f in FAMILIES]
    for other in dumps[1:]:
        assert other == dumps[0]


def test_a_spectrum_family_refuses_what_the_pcm_family_refuses(dumpers):
    for family in FAMILIES:
        assert run_dumper(dumpers['deepvqe_s'], 16000, 256, 129, family)['rc'] == -4
        assert run_dumper(dumpers['align_ulcnet'], 16000, 256, 1032, family)['rc'] == -4
        assert run_dumper(dumpers['align_ulcnet'], 16000, 256, 256, family)['rc'] == -5   # N=2: no graph in the catalog


# ---- the graph rows (ai3_model_io_deepvqes.h, ai3_model_io_alignulcnet.h) against the same exports ----

# State inputs in graph order, spelled out here from the exporters, never from the headers
DEEPVQE_S_STATES = ['state_align_key_ring', 'state_align_value_ring', 'state_align_score_history',
                    'state_ccm_up_history', 'state_far1_history', 'state_far2_history', 'h_gru',
                    'state_mic1_history', 'state_mic2_history', 'state_mic3_history', 'state_mic4_history',
                    'state_res2_history', 'state_res3_history', 'state_up1_history', 'state_up2_history',
                    'state_up3_history']
ALIGN_ULCNET_STATES = ['key_history', 'value_history', 'logit_history', 'h_gru']


@pytest.fixture(scope='module')
def model_io_rows():
    """row id -> {'layout', 'counts' (signals, heads, inputs, outputs), 'in', 'out'} as the headers' rows expand."""
    path = os.environ.get('AIAEC_MODEL_IO_DUMPER')
    if not path:
        if shutil.which('make') is None or shutil.which('cc') is None:
            pytest.skip('no make / C compiler available')
        out = subprocess.run(['make', '-s', '-C', str(STREAM_AIAEC), 'model-io-dumper'],
                             check=True, capture_output=True, text=True)
        path = out.stdout.strip().splitlines()[-1]
    rows = {}
    current = None
    for line in subprocess.run([path], check=True, capture_output=True, text=True).stdout.splitlines():
        fields = line.split()
        if fields[0] == 'row':
            current = {'layout': int(fields[3]), 'counts': tuple(int(v) for v in fields[5:9]), 'in': [], 'out': []}
            assert fields[1] not in rows, 'duplicate row ' + fields[1]
            rows[fields[1]] = current
        else:
            kind, index, b, c, h, w, elements = fields
            assert int(index) == len(current[kind])
            shape = (int(b), int(c), int(h), int(w))
            assert elements == str(shape[0] * shape[1] * shape[2] * shape[3])
            current[kind].append(shape)
    return rows


def _row_id(case):
    model, sr, fft, n = case
    return '%s_%d_%d_n%d' % (model, sr, fft, n)


def test_the_sample_rows_are_exactly_the_exported_cases(model_io_rows):
    assert sorted(model_io_rows) == sorted(_row_id(case) for case in CASES)


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_sample_row_equals_a_fresh_export(case, model_io_rows, exports):
    model_name, _sr, _fft, n = case
    row = model_io_rows[_row_id(case)]
    graph, metadata = exports[case]
    in_names = [v.name for v in graph.input]
    out_names = [v.name for v in graph.output]
    shapes_in = [_dims(v) for v in graph.input]
    shapes_out = [_dims(v) for v in graph.output]
    signals = signal_names(model_name, n)
    states = in_names[len(signals):]

    assert row['layout'] == metadata['state_layout_version']
    assert states == (DEEPVQE_S_STATES if model_name == 'deepvqe_s' else ALIGN_ULCNET_STATES)
    assert out_names == ['output'] + [name + '_out' for name in states]
    assert row['counts'] == (len(signals), 1, len(in_names), len(out_names))
    assert row['in'] == shapes_in, 'input shapes (signals, then states) differ from the exported graph'
    assert row['out'] == shapes_out, 'output shapes (head, then states) differ from the exported graph'
    # state k is input signal_in + k and output head_out + k, equal shape at both ends
    for k, name in enumerate(states):
        assert row['in'][len(signals) + k] == row['out'][1 + k], name
    if model_name == 'align_ulcnet' and n > 1:
        assert in_names[len(signals) - 1] == 'frame_reset' and row['in'][len(signals) - 1] == (1, 1, 1, n)
