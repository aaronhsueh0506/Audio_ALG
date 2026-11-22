"""The DFN2 exporter writes the ports in the one order every converted graph follows, and the stream's work
views are the graph's signal inputs and head outputs.

Inputs are the signal inputs (erb, spec), then the state inputs; outputs are the head outputs, then the state
outputs named ``<input>_out``; state k is input ``signal_in + k`` and output ``head_out + k`` with equal shapes.
The board samples bind by position, so this order is the contract. The library keeps no port list: stream/ainr
hands each block's signals and heads in StreamWork, and the graph's state ports belong to the sample's model
configuration. Random-weight graphs are exported full and mask-only at N = 1 and N = 8 for the order check (the
ONNX names are read here, on the exporter side only), and at the registered catalog grids (16 kHz / 256, 48 kHz /
1024) for the work-view check against ``make -C stream/ainr [MASK_ONLY=1] work-dumper``, which runs a real
preprocess for each registered row; the rows of the mask_lookahead 0 model (32 kHz / 512 at N 1 and 4, 48 kHz / 1024
at N 1 and 2, 16 kHz / 256 at N 1 and 8) come from a dumper bound to that model
(``EXTRA_CFLAGS=-DAINR_PRODUCT_MODEL_ID=3``). The library accepts any N up to 64 at a grid it knows; which N has a graph
is the sample table, so an N without a row is accepted by the dumper (rc 0) and absent from the graph-io dumper's rows,
and N = 65 is the library's refusal (-6). Set AINR_PIN_DUMPERS to
``<full dumper><pathsep><lite dumper>[<pathsep><mask_lookahead 0 dumper>]`` to compare other builds (the mutation
check of this test).

The board samples' DFN2 rows (stream/ainr/graph_io_dfn2.h) are pinned here as well: every row printed by
``make -C stream/ainr graph-io-dumper`` (set AINR_GRAPH_IO_DUMPER to compare another build) is compared with a fresh
export at the production model dimensions: layout version, port counts and order, every input and output shape, the
state shapes at both ends, the state output name ``<input>_out``, the mask_lookahead, the graph file and ERB file
names, and the row set itself (a row id is ``dfn2_<full|lite>_<sr>_<fft>_n<N>_l<mask_lookahead>``).
"""

import os
import pathlib
import shutil
import subprocess

import pytest
import torch

import dfn_pairs

FAMILY = 'DeepFilterNet2'
SIGNALS = ['erb', 'spec']
CASES = [(False, 1), (False, 8), (True, 1), (True, 8)]


@pytest.fixture(scope='module')
def export():
    return dfn_pairs.import_family_module(FAMILY, 'export_onnx')


def _dims(value):
    dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
    assert 0 < len(dims) <= 4, (value.name, dims)
    return tuple([1] * (4 - len(dims)) + dims)


def _export(export, tmp_path, mask_only, frames_per_call):
    onnx = pytest.importorskip('onnx')
    model_mod = dfn_pairs.import_family_module(FAMILY, 'model')
    torch.manual_seed(7)
    model = model_mod.DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64, enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4, mask_lookahead=0 if mask_only else 1,
        df_lookahead=0 if mask_only else 1).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    wrapper = export.StatelessDFN2Heads(model, frames_per_call=frames_per_call, mask_only=mask_only).eval()
    path = tmp_path / 'g.onnx'
    export.export_graph(wrapper, {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
                        str(checkpoint), str(path), verify=False)
    return onnx.load(str(path)).graph


@pytest.mark.parametrize('mask_only,frames_per_call', CASES,
                         ids=['%s-n%d' % ('lite' if m else 'full', n) for m, n in CASES])
def test_ports_are_signals_then_states_and_heads_then_states(export, tmp_path, mask_only, frames_per_call):
    graph = _export(export, tmp_path, mask_only, frames_per_call)
    in_names = [v.name for v in graph.input]
    out_names = [v.name for v in graph.output]
    heads = ['erb_mask'] if mask_only else ['erb_mask', 'df_coefs', 'df_alpha']
    states = in_names[len(SIGNALS):]
    assert in_names[:len(SIGNALS)] == SIGNALS
    assert states == (['h_gru'] if mask_only else ['h_gru', 'df_convp_history'])
    assert out_names == heads + [name + '_out' for name in states]
    # state k pairs input signal_in + k with output head_out + k, equal shape at both ends
    shapes_in = [_dims(v) for v in graph.input]
    shapes_out = [_dims(v) for v in graph.output]
    for k in range(len(states)):
        assert shapes_in[len(SIGNALS) + k] == shapes_out[len(heads) + k], states[k]
    # signal windows hold N + 2 frames, heads hold N
    assert shapes_in[0][2] == shapes_in[1][2] == frames_per_call + 2
    assert shapes_out[0][2] == frames_per_call


REPO = pathlib.Path(__file__).resolve().parents[2]
STREAM_AINR = REPO / 'stream' / 'ainr'

# (lite, sr, fft, n_erb, N): rows of the full model (16 kHz / 256 at N 1 and 8, 48 kHz / 1024 at N 1) and of the lite
# model (48 kHz / 1024 at N 1); df_bins is 96 in every row
ROWS = [(False, 16000, 256, 24, 1), (False, 16000, 256, 24, 8), (False, 48000, 1024, 32, 1),
        (True, 48000, 1024, 32, 1)]
# The mask_lookahead 0 model's grids: full, mask_lookahead 0, df_lookahead 0
L0_ROWS = [(False, 16000, 256, 24, 1), (False, 16000, 256, 24, 8), (False, 32000, 512, 28, 1),
           (False, 32000, 512, 28, 4), (False, 48000, 1024, 32, 1), (False, 48000, 1024, 32, 2)]


def _row_id(row):
    lite, sr, fft, _erb, n = row
    return '%s-%dk-fft%d-n%d' % ('lite' if lite else 'full', sr // 1000, fft, n)


@pytest.fixture(scope='module')
def work_dumpers():
    override = os.environ.get('AINR_PIN_DUMPERS')
    if override:
        fields = override.split(os.pathsep)
        full, lite = fields[:2]
        return {False: full, True: lite, 'l0': fields[2] if len(fields) > 2 else None}
    if shutil.which('make') is None or shutil.which('cc') is None:
        pytest.skip('no make / C compiler available')
    paths = {}
    for lite in (False, True):
        out = subprocess.run(['make', '-s', '-C', str(STREAM_AINR), 'MASK_ONLY=%d' % lite, 'work-dumper'],
                             check=True, capture_output=True, text=True)
        paths[lite] = out.stdout.strip().splitlines()[-1]
    out = subprocess.run(['make', '-s', '-C', str(STREAM_AINR), 'work-dumper',
                          'EXTRA_CFLAGS=-DAINR_PRODUCT_MODEL_ID=3'],
                         check=True, capture_output=True, text=True)
    paths['l0'] = out.stdout.strip().splitlines()[-1]
    return paths


def run_work_dumper(path, sr, fft, hop_in, family=0):
    """rc, queue, mask_lookahead, counts and the work views (b, c, h, w) with element counts of the first real preprocess."""
    text = subprocess.run([path, str(sr), str(fft), str(hop_in), str(family)], check=True,
                          capture_output=True, text=True).stdout
    dump = {'in': [], 'out': []}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == 'rc':
            dump['rc'] = int(fields[1])
        elif fields[0] == 'info':
            dump['queue' if fields[1] == 'queue' else 'mask_lookahead'] = int(fields[2])
        elif fields[0] == 'counts':
            dump['counts'] = (int(fields[2]), int(fields[4]))
        elif fields[0] in ('in', 'out'):
            _kind, index, b, c, h, w, elements = fields
            assert int(index) == len(dump[fields[0]])
            dump[fields[0]].append(((int(b), int(c), int(h), int(w)), int(elements)))
    return dump


def _export_row(export, tmp_path, row, l0=False):
    onnx = pytest.importorskip('onnx')
    lite, sr, fft, n_erb, n = row
    model_mod = dfn_pairs.import_family_module(FAMILY, 'model')
    torch.manual_seed(7)
    model = model_mod.DeepFilterNet2(
        n_fft=fft, sr=sr, n_erb=n_erb, df_bins=96, enc_ch=8, emb_size=32, df_hidden=32,
        lin_groups=4, enc_lin_groups=4, mask_lookahead=0 if lite or l0 else 1,
        df_lookahead=0 if lite or l0 else 1).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    wrapper = export.StatelessDFN2Heads(model, frames_per_call=n, mask_only=lite).eval()
    path = tmp_path / 'g.onnx'
    export.export_graph(wrapper, {'SR': sr, 'N_FFT': fft, 'WIN_LEN': fft, 'HOP_LEN': fft // 2},
                        str(checkpoint), str(path), verify=False)
    return onnx.load(str(path)).graph


@pytest.mark.parametrize('family', (0, 2), ids=('pcm', 'freq'))   # the dumper's family 1 (_spec) is retired
@pytest.mark.parametrize('row', ROWS, ids=_row_id)
def test_library_work_views_are_the_exported_signals_and_heads(export, tmp_path, work_dumpers, row, family):
    lite, sr, fft, _n_erb, n = row
    graph = _export_row(export, tmp_path, row)
    dump = run_work_dumper(work_dumpers[lite], sr, fft, n * (fft // 2), family)
    shapes_in = [_dims(v) for v in graph.input]
    shapes_out = [_dims(v) for v in graph.output]
    n_heads = 1 if lite else 3
    assert dump['rc'] == 0
    assert dump['counts'] == (len(SIGNALS), n_heads)
    assert [p[0] for p in dump['in']] == shapes_in[:len(SIGNALS)], 'signal views differ from the exported graph'
    assert [p[0] for p in dump['out']] == shapes_out[:n_heads], 'head views differ from the exported graph'
    for views in (dump['in'], dump['out']):
        for i, (shape, elements) in enumerate(views):
            assert elements == shape[0] * shape[1] * shape[2] * shape[3], i
    # the graph's state ports are after them in the graph and not in the work
    assert len(shapes_in) - len(SIGNALS) == len(shapes_out) - n_heads == (1 if lite else 2)


@pytest.mark.parametrize('family', (0, 2), ids=('pcm', 'freq'))
@pytest.mark.parametrize('row', L0_ROWS, ids=_row_id)
def test_l0_library_work_views_are_the_exported_signals_and_heads(export, tmp_path, work_dumpers, row, family):
    """The mask_lookahead 0 model's rows: the work of a real preprocess is the exported graph's signals and heads, the
    first block already carries the job (no mask_lookahead blocks before it), and queue_frames is 0."""
    _lite, sr, fft, _n_erb, n = row
    graph = _export_row(export, tmp_path, row, l0=True)
    dump = run_work_dumper(work_dumpers['l0'], sr, fft, n * (fft // 2), family)
    shapes_in = [_dims(v) for v in graph.input]
    shapes_out = [_dims(v) for v in graph.output]
    assert dump['rc'] == 0 and dump['queue'] == 0
    assert dump['counts'] == (len(SIGNALS), 3)
    assert [p[0] for p in dump['in']] == shapes_in[:len(SIGNALS)], 'signal views differ from the exported graph'
    assert [p[0] for p in dump['out']] == shapes_out[:3], 'head views differ from the exported graph'
    for views in (dump['in'], dump['out']):
        for i, (shape, elements) in enumerate(views):
            assert elements == shape[0] * shape[1] * shape[2] * shape[3], i
    assert dump['mask_lookahead'] == 0


def _row_key(lite, sr, fft, n, mask_lookahead):
    return 'dfn2_%s_%d_%d_n%d_l%d' % ('lite' if lite else 'full', sr, fft, n, mask_lookahead)


def test_the_l0_customer_arithmetic(work_dumpers, graph_io_rows):
    """The mask_lookahead 0 model: N = hop_in / (fft / 2) at every grid, any N up to 64 accepted (an N without a graph
    row is the sample's refusal, not the library's), N = 65 refused."""
    l0 = work_dumpers['l0']
    for sr, fft, hop_in, n in ((16000, 256, 1024, 8), (32000, 512, 1024, 4), (48000, 1024, 1024, 2), (48000, 1024, 512, 1)):
        dump = run_work_dumper(l0, sr, fft, hop_in)
        assert dump['rc'] == 0 and dump['in'][0][0][2] == n + 2 and dump['out'][0][0][2] == n
        assert dump['mask_lookahead'] == 0 and dump['queue'] == 0
        assert _row_key(False, sr, fft, n, dump['mask_lookahead']) in graph_io_rows
    for sr, fft, n in ((32000, 512, 3), (48000, 1024, 4), (16000, 256, 2)):   # accepted, no graph row
        dump = run_work_dumper(l0, sr, fft, n * (fft // 2))
        assert dump['rc'] == 0 and dump['in'][0][0][2] == n + 2 and dump['out'][0][0][2] == n
        assert _row_key(False, sr, fft, n, dump['mask_lookahead']) not in graph_io_rows
    assert run_work_dumper(l0, 32000, 512, 65 * 256)['rc'] == -6   # N = 65: AINR_ERR_FRAMES_PER_CALL
    assert run_work_dumper(l0, 32000, 512, 100)['rc'] == -4        # AINR_ERR_IN_HOP
    assert run_work_dumper(l0, 44100, 512, 256)['rc'] == -3        # AINR_ERR_GRID


def test_the_customer_arithmetic_picks_the_exported_n(work_dumpers, graph_io_rows):
    """N = hop_in / (fft / 2) at any mask_lookahead; the time axis of the signals is N + 2 and of the heads N; a
    config with no graph row is accepted by the library (rc 0) and absent from the sample table; N = 65 is refused."""
    full, lite = work_dumpers[False], work_dumpers[True]
    for sr, fft, hop_in, n in ((16000, 256, 128, 1), (16000, 256, 1024, 8), (48000, 1024, 512, 1), (48000, 1024, 1024, 2)):
        dump = run_work_dumper(full, sr, fft, hop_in)
        assert dump['rc'] == 0 and dump['in'][0][0][2] == n + 2 and dump['out'][0][0][2] == n
        assert _row_key(False, sr, fft, n, dump['mask_lookahead']) in graph_io_rows
    for sr, fft, n in ((16000, 256, 2), (48000, 1024, 4)):   # accepted, no graph row
        dump = run_work_dumper(full, sr, fft, n * (fft // 2))
        assert dump['rc'] == 0 and dump['in'][0][0][2] == n + 2 and dump['out'][0][0][2] == n
        assert _row_key(False, sr, fft, n, dump['mask_lookahead']) not in graph_io_rows
    assert run_work_dumper(full, 16000, 256, 65 * 128)['rc'] == -6   # N = 65: AINR_ERR_FRAMES_PER_CALL
    assert run_work_dumper(full, 16000, 256, 100)['rc'] == -4       # AINR_ERR_IN_HOP
    assert run_work_dumper(full, 16000, 512, 256)['rc'] == -3       # AINR_ERR_GRID
    assert run_work_dumper(lite, 16000, 256, 128)['rc'] == -3       # lite has only 48 kHz / 1024
    dump = run_work_dumper(lite, 48000, 1024, 3 * 512)               # lite N = 3: accepted, no row
    assert dump['rc'] == 0 and _row_key(True, 48000, 1024, 3, dump['mask_lookahead']) not in graph_io_rows


# ---- the graph rows (stream/ainr/graph_io_dfn2.h) against a fresh export at the production dimensions ----

# (row id, mask_only, sr, fft, n_erb, mask_lookahead, df_lookahead, N): the rows of the header, spelled out here from
# the exporter's arguments, never from the header. The 48 kHz N = 1 and N = 2 graphs exist for mask_lookahead 1 and 0;
# the shapes do not depend on the lookahead pair, so the two rows of an N share one shape and differ in file and key.
GRAPH_IO_CASES = [
    ('dfn2_full_16000_256_n1_l0', False, 16000, 256, 24, 0, 0, 1),
    ('dfn2_full_16000_256_n8_l0', False, 16000, 256, 24, 0, 0, 8),
    ('dfn2_full_48000_1024_n1_l1', False, 48000, 1024, 32, 1, 1, 1),
    ('dfn2_full_48000_1024_n2_l1', False, 48000, 1024, 32, 1, 1, 2),
    ('dfn2_full_48000_1024_n1_l0', False, 48000, 1024, 32, 0, 0, 1),
    ('dfn2_full_48000_1024_n2_l0', False, 48000, 1024, 32, 0, 0, 2),
    ('dfn2_full_32000_512_n1_l0', False, 32000, 512, 28, 0, 0, 1),
    ('dfn2_full_32000_512_n4_l0', False, 32000, 512, 28, 0, 0, 4),
    ('dfn2_lite_48000_1024_n1_l1', True, 48000, 1024, 32, 1, 0, 1),
]


def _graph_file(mask_only, sr, fft, mask_la, n):
    """dfn2[_l0|_lite]_<sr/1000>k_fft<fft>_n<N>.bin: the mask_lookahead 0 model has its own files except at 16 kHz,
    whose graphs it shares with the full model."""
    tag = '_lite' if mask_only else '_l0' if mask_la == 0 and sr != 16000 else ''
    return 'dfn2%s_%dk_fft%d_n%d.bin' % (tag, sr // 1000, fft, n)


@pytest.fixture(scope='module')
def graph_io_dumper():
    override = os.environ.get('AINR_GRAPH_IO_DUMPER')
    if override:
        return override
    if shutil.which('make') is None or shutil.which('cc') is None:
        pytest.skip('no make / C compiler available')
    out = subprocess.run(['make', '-s', '-C', str(STREAM_AINR), 'graph-io-dumper'],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1]


@pytest.fixture(scope='module')
def graph_io_rows(graph_io_dumper):
    """row id -> {'layout', 'counts' (signals, heads, inputs, outputs), 'mask_lookahead', 'file', 'erb_file', 'in',
    'out'} as the header's rows expand."""
    rows = {}
    current = None
    text = subprocess.run([graph_io_dumper], check=True, capture_output=True, text=True).stdout
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == 'row':
            current = {'layout': int(fields[3]), 'counts': tuple(int(v) for v in fields[5:9]), 'in': [], 'out': []}
            assert fields[1] not in rows, 'duplicate row ' + fields[1]
            rows[fields[1]] = current
        elif fields[0] == 'file':
            current['mask_lookahead'], current['file'], current['erb_file'] = int(fields[1]), fields[2], fields[3]
        else:
            kind, index, b, c, h, w, elements = fields
            assert int(index) == len(current[kind])
            shape = (int(b), int(c), int(h), int(w))
            assert elements == str(shape[0] * shape[1] * shape[2] * shape[3])
            current[kind].append(shape)
    return rows


@pytest.fixture(scope='module')
def graph_io_exports(export, tmp_path_factory):
    """case -> (graph, metadata) of a random-weight export at the production dimensions of config.ini."""
    pytest.importorskip('onnx')
    import onnx
    model_mod = dfn_pairs.import_family_module(FAMILY, 'model')
    work = tmp_path_factory.mktemp('dfn2_graph_io')
    result = {}
    for case in GRAPH_IO_CASES:
        rid, mask_only, sr, fft, n_erb, mask_la, df_la, n = case
        torch.manual_seed(7)
        model = model_mod.DeepFilterNet2(
            n_fft=fft, sr=sr, n_erb=n_erb, df_bins=96, df_order=5, enc_ch=64, emb_size=256, df_hidden=256,
            df_num_layers=2, emb_num_layers=3, lin_groups=16, enc_lin_groups=32, mask_lookahead=mask_la,
            df_lookahead=df_la).eval()
        checkpoint = work / (rid + '.pth')
        checkpoint.write_bytes(b'only hashed')
        wrapper = export.StatelessDFN2Heads(model, frames_per_call=n, mask_only=mask_only).eval()
        path = work / (rid + '.onnx')
        metadata = export.export_graph(wrapper, {'SR': sr, 'N_FFT': fft, 'WIN_LEN': fft, 'HOP_LEN': fft // 2},
                                       str(checkpoint), str(path), verify=False)
        result[case] = (onnx.load(str(path)).graph, metadata)
    return result


def test_the_sample_rows_are_exactly_the_registered_rows(graph_io_rows):
    assert sorted(graph_io_rows) == sorted(case[0] for case in GRAPH_IO_CASES)
    files = [row['file'] for row in graph_io_rows.values()]
    assert len(set(files)) == len(files), 'two rows share a graph file'


@pytest.mark.parametrize('case', GRAPH_IO_CASES, ids=[case[0] for case in GRAPH_IO_CASES])
def test_sample_row_equals_a_fresh_export(case, graph_io_rows, graph_io_exports):
    rid, mask_only, sr, fft, n_erb, mask_la, _dla, n = case
    row = graph_io_rows[rid]
    graph, metadata = graph_io_exports[case]
    in_names = [v.name for v in graph.input]
    out_names = [v.name for v in graph.output]
    shapes_in = [_dims(v) for v in graph.input]
    shapes_out = [_dims(v) for v in graph.output]
    heads = ['erb_mask'] if mask_only else ['erb_mask', 'df_coefs', 'df_alpha']
    states = ['h_gru'] if mask_only else ['h_gru', 'df_convp_history']

    assert row['layout'] == metadata['state_layout_version']
    assert row['mask_lookahead'] == mask_la
    assert row['file'] == _graph_file(mask_only, sr, fft, mask_la, n)
    assert row['erb_file'] == 'fb_%d_%d.bin' % (sr, n_erb)
    assert in_names == SIGNALS + states and out_names == heads + [name + '_out' for name in states]
    assert row['counts'] == (len(SIGNALS), len(heads), len(in_names), len(out_names))
    assert row['in'] == shapes_in, 'input shapes (signals, then states) differ from the exported graph'
    assert row['out'] == shapes_out, 'output shapes (heads, then states) differ from the exported graph'
    # state k is input signal_in + k and output head_out + k, equal shape at both ends; df_alpha is output 2
    for k, name in enumerate(states):
        assert row['in'][len(SIGNALS) + k] == row['out'][len(heads) + k], name
    if not mask_only:
        assert out_names[2] == 'df_alpha' and row['out'][2] == (1, 1, n, 1)
