"""Explicit full-state export contract for streaming Align-ULCNet."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet.export_onnx import (
    AlignUlcnetStreamingExport,
    host_back_end,
    stream_features,
    GRU_HIDDEN,
    GRU_LAYERS,
    GRU_STATE_NAME,
    GRU_SUBBANDS,
    COMPRESSION_EXPONENT,
    INPUT_NAMES,
    LAYOUT_VERSION,
    RETIRED_LAYOUT_VERSIONS,
    SIGNAL_INPUTS,
    MAX_DELAY_DEPTH,
    MIN_DELAY_DEPTH,
    OUTPUT_NAMES,
    SCORE_WINDOW_FRAMES,
    STATE_LAYOUT_VERSION,
    STATE_NAMES,
    STATE_OUTPUT_NAMES,
    SUPPORTED_GRIDS,
    WINDOW_STATE_DROPS,
    WINDOW_STATE_NAMES,
    ta_bins_for,
    TA_CHANNELS,
    _write_metadata,
    dummy_inputs,
    export_graph,
    graph_signals,
    next_state,
    state_shapes,
)
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.Align_ULCNet import export_onnx
from AIAEC.dataset_gen import linear_aec
from AIAEC.dataset_gen.linear_aec import make_linear_aec_contract
from AIAEC.training_common import FAR_INPUT_MODE_C_VALUES


GRID = SignalGrid(16000, 512, 512, 256)
# The 16 kHz values of the two grid-derived quantities. ta_bins_for()
# is the single derivation; test_ta_bins_follows_the_grid pins both.
N_FREQS = GRID.n_freqs
TA_BINS = 26
D = 8


def _checkpoint_contract(**extra):
    """The part of a checkpoint contract the exporter reads."""
    linear = make_linear_aec_contract(GRID.sample_rate)
    return {
        'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        **extra,
    }


def _complex_frame(generator, n_freqs=GRID.n_freqs):
    return torch.complex(
        torch.randn(1, 1, n_freqs, generator=generator),
        torch.randn(1, 1, n_freqs, generator=generator),
    )


def _ri(value):
    return torch.stack((value.real, value.imag), dim=-1)


def test_ta_bins_follows_the_grid():
    """ta_bins is derived, not a constant, and the derivation has a trap.

    At 48 kHz the correct value (52) equals reorient.width at 16 kHz (52),
    so a derivation that read reorient.width instead of ceil(width/2) would
    look right in a 48 kHz trace and be 2x wrong on the shipped grid. Both
    grids are checked here for exactly that reason.
    """
    g48 = SignalGrid(48000, 1024, 1024, 512)
    m16 = AlignULCNet(GRID, max_delay_frames=D)
    m48 = AlignULCNet(g48, max_delay_frames=D)
    assert ta_bins_for(m16) == TA_BINS == 26
    assert ta_bins_for(m48) == 52
    assert m16.reorient.width == 52 and m48.reorient.width == 104
    # The K/V caches carry it, so the state geometry follows the grid too.
    assert state_shapes(D, ta_bins_for(m48))['key_history'][-1] == 52


def test_c_header_enumerates_the_same_grids_as_the_exporter():
    """The legal-grid set is stated in both languages; pin them together.

    export_onnx refuses a foreign grid in Python and ulcnet_model_io.h
    #errors on one in C. Nothing compared the two sets, so a grid could be
    added to one side alone and the mismatch would only surface as a build
    that the exporter is happy to feed and the C cannot accept.
    """
    header = (
        Path(__file__).resolve().parents[1]
        / 'Align_ULCNet' / 'ulcnet_model_io.h'
    ).read_text(encoding='utf-8')
    pairs = re.findall(
        r'ULCNET_MODEL_IO_SR\s*==\s*(\d+)\s*&&\s*'
        r'ULCNET_MODEL_IO_N_FFT\s*==\s*(\d+)',
        header,
    )
    assert pairs, 'no grid pairs found in the header guard'
    assert {(int(sr), int(fft)) for sr, fft in pairs} == set(SUPPORTED_GRIDS)


_TA_BINS_PROBE = """
#include <stdio.h>
#include "ulcnet_model_io.h"
int main(void) {
    printf("%d %d\\n", ULCNET_MODEL_IO_BINS, ULCNET_MODEL_IO_TA_BINS);
    return 0;
}
"""


@pytest.mark.parametrize('sample_rate,n_fft', SUPPORTED_GRIDS)
def test_c_ta_bins_matches_the_python_derivation(tmp_path, sample_rate, n_fft):
    """The C macro and the Python derivation are independent encodings.

    Nothing else compares them: the C driver checks the descriptor against
    the same macro it was built from, which is an identity, and a text match
    on the header survives a changed divisor. Only evaluating the macro and
    comparing it to the model's own width closes that gap -- change the 10 in
    ULCNET_MODEL_IO_TA_BINS to an 11 and this is the test that goes red.
    """
    compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if compiler is None:
        pytest.skip('no C compiler available')
    ulcnet_dir = Path(__file__).resolve().parents[1] / 'Align_ULCNet'
    probe = tmp_path / 'probe.c'
    probe.write_text(_TA_BINS_PROBE, encoding='utf-8')
    executable = tmp_path / 'probe'
    subprocess.run([
        compiler, '-std=c99', '-Wall', '-Werror',
        '-DULCNET_MODEL_IO_SR=%d' % sample_rate,
        '-DULCNET_MODEL_IO_N_FFT=%d' % n_fft,
        '-I', str(ulcnet_dir), str(probe), '-o', str(executable),
    ], check=True, capture_output=True)
    c_bins, c_ta_bins = (
        int(value) for value in
        subprocess.run([str(executable)], check=True, capture_output=True)
        .stdout.split()
    )

    grid = SignalGrid(sample_rate, n_fft, n_fft, n_fft // 2)
    model = AlignULCNet(grid, max_delay_frames=D)
    assert c_bins == grid.n_freqs
    assert c_ta_bins == ta_bins_for(model)


def test_streaming_export_shapes_are_fixed_and_full_window():
    shapes = state_shapes(D, TA_BINS)
    assert shapes['key_history'] == (1, 32, D, 26)
    assert shapes['value_history'] == (1, 32, D, 26)
    assert shapes['logit_history'] == (1, 32, 5, D)
    assert shapes['h_gru'] == (1, 4, 1, 128)
    # Every boundary state is rank-4 NCHW. Written once, as a rule, so a new
    # state cannot be added at the wrong rank without failing here.
    assert all(len(shape) == 4 for shape in shapes.values())
    assert INPUT_NAMES == (
        'error_mag', 'far_mag', 'error_cos', 'error_sin',
        'key_history', 'value_history', 'logit_history', 'h_gru',
    )
    # Every state input has an _out partner, in the order the inputs are
    # bound, so the next state is the output tail unchanged.
    assert OUTPUT_NAMES == (
        'output', 'key_history_out', 'value_history_out',
        'logit_history_out', 'h_gru_out',
    )
    assert OUTPUT_NAMES[1:] == tuple(
        name + '_out' for name in INPUT_NAMES[SIGNAL_INPUTS:])
    assert STATE_NAMES == INPUT_NAMES[SIGNAL_INPUTS:]
    assert STATE_OUTPUT_NAMES == OUTPUT_NAMES[1:]
    assert WINDOW_STATE_NAMES == ('key_history', 'value_history',
                                  'logit_history')
    assert GRU_SUBBANDS * GRU_LAYERS == shapes['h_gru'][1]
    assert SCORE_WINDOW_FRAMES == shapes['logit_history'][2]


def test_layout_version_is_new_and_retired_numbers_stay_retired():
    assert LAYOUT_VERSION == STATE_LAYOUT_VERSION == 18
    # 3-17 each denoted a boundary that is not implemented (split hiddens,
    # in-graph front end, windows one frame short).
    # Reusing one would make a single number mean two different contracts,
    # which no runtime could tell apart. The shipped version must sit above
    # every retired number rather than above a literal floor, which would
    # stay green after the next bump while guarding nothing.
    assert RETIRED_LAYOUT_VERSIONS == frozenset(range(3, 18))
    assert LAYOUT_VERSION not in RETIRED_LAYOUT_VERSIONS
    assert LAYOUT_VERSION > max(RETIRED_LAYOUT_VERSIONS)


def test_retired_boundary_switches_are_gone():
    """The in-graph front end and the split hiddens are not selectable."""
    import AIAEC.Align_ULCNet.export_onnx as module
    for name in ('FEATURE_LAYOUTS', 'GRU_STATE_LAYOUTS', 'resolve_layout',
                 'GRAPH_LAYOUTS', 'LAYOUT_VERSIONS'):
        assert not hasattr(module, name), name
    result = subprocess.run(
        [sys.executable, module.__file__, '--help'],
        capture_output=True, text=True,
        env={'PYTHONDONTWRITEBYTECODE': '1', 'PATH': os.environ['PATH']},
    )
    assert result.returncode == 0, result.stderr
    assert '--feature-layout' not in result.stdout
    assert '--gru-state-layout' not in result.stdout
    with pytest.raises(TypeError):
        AlignUlcnetStreamingExport(
            AlignULCNet(GRID, max_delay_frames=D).eval(),
            gru_state_layout='split')


def test_c_descriptor_constants_match_export_contract():
    header = (
        Path(__file__).resolve().parents[1]
        / 'Align_ULCNet' / 'ulcnet_model_io.h'
    ).read_text(encoding='utf-8')
    expected = {
        'ULCNET_MODEL_IO_LAYOUT_VERSION': STATE_LAYOUT_VERSION,
        'ULCNET_MODEL_IO_MIN_D': MIN_DELAY_DEPTH,
        'ULCNET_MODEL_IO_MAX_D': MAX_DELAY_DEPTH,
        'ULCNET_MODEL_IO_TA_CHANNELS': TA_CHANNELS,
        'ULCNET_MODEL_IO_SCORE_WINDOW': SCORE_WINDOW_FRAMES,
        'ULCNET_MODEL_IO_GRU_SUBBANDS': GRU_SUBBANDS,
        'ULCNET_MODEL_IO_GRU_LAYERS': GRU_LAYERS,
        'ULCNET_MODEL_IO_GRU_HIDDEN': GRU_HIDDEN,
    }
    for name, value in expected.items():
        match = re.search(
            r'^#define\s+%s\s+(\d+)u?\s*$' % re.escape(name),
            header,
            flags=re.MULTILINE,
        )
        assert match is not None, name
        assert int(match.group(1)) == value

    # TA_BINS is derived from the compiled grid rather than transcribed, so
    # it cannot be pinned by value here. Matching the header TEXT would not
    # pin it either -- a changed divisor keeps the same text. The evaluated
    # comparison lives in test_c_ta_bins_matches_the_python_derivation.
    assert '#define ULCNET_MODEL_IO_TA_BINS' in header

    # The far-input enumeration is a two-sided contract: the exporter writes
    # the numeric value into the metadata, the descriptor carries the same
    # field, and both pipelines compare them. Pin the enumerators against the
    # single Python table rather than restating the numbers here.
    for name, value in FAR_INPUT_MODE_C_VALUES.items():
        enumerator = 'ULCNET_FAR_' + (
            'RAW' if name == 'raw_far' else name.replace('_far', '').upper()
        )
        match = re.search(
            r'^\s+%s\s*=\s*(\d+)\s*,?' % re.escape(enumerator),
            header,
            flags=re.MULTILINE,
        )
        assert match is not None, enumerator
        assert int(match.group(1)) == value, enumerator
    assert 'int far_input_mode;' in header

    # The compression exponent is applied outside the graph; the C define and
    # the exporter constant could drift silently, so pin them here (the
    # exporter separately refuses checkpoints with any other exponent).
    match = re.search(
        r'^#define\s+ULCNET_MODEL_IO_COMPRESSION_EXP\s+([0-9.]+)f\s*$',
        header,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert float(match.group(1)) == COMPRESSION_EXPONENT


def test_c_frontend_contract_equals_the_python_contract_version():
    """The board-side constant is the string the checkpoints are stamped with."""
    header = (
        Path(__file__).resolve().parents[1]
        / 'Align_ULCNet' / 'ulcnet_model_io.h'
    ).read_text(encoding='utf-8')
    match = re.search(
        r'^#define\s+ULCNET_MODEL_IO_FRONTEND_CONTRACT\s+"([^"]+)"\s*$',
        header,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert match.group(1) == linear_aec.LINEAR_AEC_CONTRACT_VERSION


def test_exported_metadata_carries_the_checkpoint_frontend(tmp_path):
    onnx = pytest.importorskip('onnx')
    model = AlignULCNet(GRID, max_delay_frames=4).eval()
    contract = _checkpoint_contract()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()},
               checkpoint)
    path = tmp_path / 'frontend.onnx'
    _wrapper, _inputs, metadata = export_graph(model, str(checkpoint), str(path))

    expected = {
        'linear_aec_contract_version': contract['linear_aec']['version'],
        'aec_behavior_hash': contract['linear_aec']['aec_behavior_hash'],
        'error_input_seam': 'analysis_formed',
    }
    assert expected['linear_aec_contract_version'] == (
        linear_aec.LINEAR_AEC_CONTRACT_VERSION)
    for key, value in expected.items():
        assert metadata[key] == value
    written = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    for key, value in expected.items():
        assert written[key] == value
        assert props[key] == value


def test_metadata_frontend_is_copied_from_the_contract_not_hard_coded(tmp_path):
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    checkpoint = tmp_path / 'ckpt.pt'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    inputs = dummy_inputs(D, N_FREQS, TA_BINS)
    outputs = AlignUlcnetStreamingExport(model).eval()(*inputs)

    contract = _checkpoint_contract()
    contract['linear_aec']['aec_behavior_hash'] = 'ab' * 32
    metadata = _write_metadata(
        str(tmp_path / 'patched.onnx'), str(checkpoint), model, contract,
        inputs, outputs,
    )
    assert metadata['aec_behavior_hash'] == 'ab' * 32

    # No recorded front end: nothing to stamp, so no graph.
    absent = _checkpoint_contract()
    del absent['linear_aec']
    with pytest.raises(ValueError, match='linear_aec'):
        _write_metadata(
            str(tmp_path / 'absent.onnx'), str(checkpoint), model, absent,
            inputs, outputs,
        )

    # A retired contract is refused at the metadata step too, so a graph is
    # never written for it even when load_model was bypassed.
    retired = _checkpoint_contract()
    retired['linear_aec']['version'] = 'aiaec-linear-error-v3'
    with pytest.raises(ValueError, match='retired'):
        _write_metadata(
            str(tmp_path / 'retired.onnx'), str(checkpoint), model, retired,
            inputs, outputs,
        )
    assert not (tmp_path / 'retired.json').exists()


def test_v3_checkpoint_still_fails_export_at_load_model(tmp_path):
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    contract = _checkpoint_contract(
        model_name='Align_ULCNet', task=AlignULCNet.task,
        sr=GRID.sample_rate, n_fft=GRID.n_fft, win_len=GRID.win_len,
        hop_len=GRID.hop_len, loss_version='test',
        ctor_max_delay_frames=D,
    )
    ok = tmp_path / 'v4.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, ok)
    export_onnx.load_model(str(ok), 'cpu')

    contract['linear_aec']['version'] = 'aiaec-linear-error-v3'
    old = tmp_path / 'v3.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, old)
    with pytest.raises(ValueError, match='retired'):
        export_onnx.load_model(str(old), 'cpu')
    assert not (tmp_path / 'v3.onnx').exists()


RETIRED_HASH = (
    '9fa7a2dec85cef8f123cc4dd02e7eb3c5dc0c1b45d7ec0f32b766cf9cda245e5')


def _retired_hash_contract(**extra):
    """Current contract version, retired behaviour hash, valid fingerprint."""
    contract = _checkpoint_contract(**extra)
    contract['linear_aec']['aec_behavior_hash'] = RETIRED_HASH
    contract['linear_aec_contract_hash'] = (
        linear_aec.LinearAecContract.from_dict(
            contract['linear_aec']).fingerprint())
    return contract


def test_retired_behavior_hash_writes_no_graph_on_any_export_path(
        tmp_path, monkeypatch):
    """The version string alone must not vouch for a checkpoint.

    LinearAecContract.from_dict checks only the version, so a v4-stamped
    checkpoint recorded under a retired behaviour hash would otherwise export
    (the exporter builds no AEC engine).
    """
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    contract = _retired_hash_contract(
        model_name='Align_ULCNet', task=AlignULCNet.task,
        sr=GRID.sample_rate, n_fft=GRID.n_fft, win_len=GRID.win_len,
        hop_len=GRID.hop_len, loss_version='test',
        ctor_max_delay_frames=D,
    )
    checkpoint = tmp_path / 'retired.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()},
               checkpoint)
    assert contract['linear_aec']['version'] == (
        linear_aec.LINEAR_AEC_CONTRACT_VERSION)
    out = tmp_path / 'out'
    out.mkdir()

    with pytest.raises(ValueError, match='which is retired because it produced a different linear_error'):
        export_onnx.load_model(str(checkpoint), 'cpu')

    monkeypatch.setattr(sys, 'argv', [
        'export_onnx.py', '--checkpoint', str(checkpoint),
        '--output', str(out / 'graph.onnx'),
    ])
    with pytest.raises(ValueError, match='retired'):
        export_onnx.main()

    # Direct callers that bypass load_model are stopped before the graph is
    # written, not after torch.onnx.export has put one on disk.
    with pytest.raises(ValueError, match='retired'):
        export_graph(model, str(checkpoint), str(out / 'direct.onnx'))

    inputs = dummy_inputs(D, N_FREQS, TA_BINS)
    outputs = AlignUlcnetStreamingExport(model).eval()(*inputs)
    with pytest.raises(ValueError, match='retired'):
        _write_metadata(str(out / 'meta.onnx'), str(checkpoint), model,
                        contract, inputs, outputs)

    assert list(out.iterdir()) == []


def test_export_refuses_foreign_compression_exponent(tmp_path):
    model = AlignULCNet(
        GRID, max_delay_frames=D, compression_exponent=0.5
    ).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': {}, 'state_dict': model.state_dict()},
               checkpoint)
    with pytest.raises(ValueError, match='compression_exponent'):
        export_graph(model, str(checkpoint), str(tmp_path / 'graph.onnx'))


def test_metadata_separates_training_provenance_from_deployment(tmp_path):
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    checkpoint = tmp_path / 'ckpt.pt'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    inputs = dummy_inputs(D, N_FREQS, TA_BINS)
    outputs = AlignUlcnetStreamingExport(model).eval()(*inputs)

    metadata = _write_metadata(
        str(tmp_path / 'model.onnx'), str(checkpoint), model,
        _checkpoint_contract(far_input_mode='raw_far'), inputs, outputs,
    )
    assert metadata['training_far_input_mode'] == 'raw_far'
    assert metadata['far_input_mode'] == 'raw_far'
    assert metadata['far_input_mode_c_value'] == FAR_INPUT_MODE_C_VALUES[
        'raw_far'
    ]
    assert metadata['state_layout_version'] == STATE_LAYOUT_VERSION
    assert metadata['state_binding'] == 'in_place_same_address'
    assert metadata['score_window_frames'] == SCORE_WINDOW_FRAMES

    # Checkpoints without the field keep the same raw-far training/deployment
    # contract.
    legacy = _write_metadata(
        str(tmp_path / 'legacy.onnx'), str(checkpoint), model,
        _checkpoint_contract(), inputs, outputs,
    )
    assert legacy['training_far_input_mode'] == 'raw_far'
    assert legacy['far_input_mode'] == 'raw_far'
    assert legacy['far_input_mode_c_value'] == 0


@pytest.mark.parametrize('sample_rate,n_fft,ta_bins',
                         [(16000, 512, 26), (48000, 1024, 52)])
def test_full_window_wrapper_matches_forward_stream_frame_by_frame(
        sample_rate, n_fft, ta_bins):
    grid = SignalGrid(sample_rate, n_fft, n_fft, n_fft // 2)
    torch.manual_seed(20260816)
    model = AlignULCNet(grid, max_delay_frames=D).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    assert wrapper.ta_bins == ta_bins
    explicit = tuple(value.clone() for value in dummy_inputs(
        D, grid.n_freqs, ta_bins)[SIGNAL_INPUTS:])
    reference = model.create_stream_state()
    generator = torch.Generator().manual_seed(73)

    with torch.no_grad():
        for _ in range(2 * D + 5):
            error = _complex_frame(generator, grid.n_freqs)
            far = _complex_frame(generator, grid.n_freqs)
            features = stream_features(model, _ri(error), _ri(far))
            outputs = wrapper(*features[:SIGNAL_INPUTS], *explicit)
            expected = model.forward_stream(error, far, reference)

            # The graph output is the raw mask; the host multiplies the
            # compressed error by it and applies the inverse signed power
            # (host_back_end), completing the chain.
            assert outputs[0].shape == (1, 2, 1, grid.n_freqs)
            assert torch.allclose(
                host_back_end(model, features[4], outputs[0]),
                _ri(expected.enhanced), atol=5e-7, rtol=1e-6
            )
            explicit = next_state(outputs)

            align = reference['align']
            # The boundary carries the cells' own windows whole.
            assert torch.equal(explicit[0], align.key_ring._ring)
            assert torch.equal(explicit[1], align.value_ring._ring)
            assert torch.equal(explicit[2], align.score_cell._window)
            # The boundary carries rank-4 NCHW; the streaming reference holds
            # nn.GRU's rank-3 hidden. Lift the reference rather than squeezing
            # the boundary, so this still pins where the new axis sits, and
            # the two subband hiddens stack on dim 1 in subband order.
            assert explicit[3].shape == (
                1, GRU_SUBBANDS * GRU_LAYERS, 1, GRU_HIDDEN)
            assert torch.equal(
                explicit[3][:, :GRU_LAYERS],
                reference['subband_gru0']._hidden.unsqueeze(0)
            )
            assert torch.equal(
                explicit[3][:, GRU_LAYERS:],
                reference['subband_gru1']._hidden.unsqueeze(0)
            )


def _run_wrapper(wrapper, model, state, generator):
    error_ri = torch.randn(1, 1, N_FREQS, 2, generator=generator)
    far_ri = torch.randn(1, 1, N_FREQS, 2, generator=generator)
    with torch.no_grad():
        return wrapper(*graph_signals(model, error_ri, far_ri), *state)


def test_window_shift_directions():
    """K/V windows are newest-first, the logit window oldest-first.

    The shift is the whole contract of the in-place binding: a window that
    moved the wrong way would still be finite, same-shaped and plausible, and
    only change which frames the attention reads.
    """
    torch.manual_seed(31)
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    state = dummy_inputs(D, N_FREQS, TA_BINS)[SIGNAL_INPUTS:]
    generator = torch.Generator().manual_seed(32)
    previous = None
    for _ in range(D + 3):
        outputs = _run_wrapper(wrapper, model, state, generator)
        state = next_state(outputs)
        key, value, logit, _hidden = state
        if previous is not None:
            old_key, old_value, old_logit = previous
            assert torch.equal(key[:, :, 1:], old_key[:, :, :-1])
            assert torch.equal(value[:, :, 1:], old_value[:, :, :-1])
            assert torch.equal(logit[:, :, :-1], old_logit[:, :, 1:])
        previous = (key, value, logit)
    # Sensitivity: the relations above are not vacuous.
    assert key[:, :, 0].abs().max() > 0 and logit[:, :, -1].abs().max() > 0
    assert not torch.equal(key[:, :, 0], key[:, :, 1])


def test_dropped_window_slot_is_never_read():
    """The slot a window drops only keeps the in and out shapes identical.

    Poisoning it with NaN must change neither the mask nor the next state;
    a graph that read it (or a Slice cutting the wrong end) would not pass.
    """
    torch.manual_seed(33)
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    state = dummy_inputs(D, N_FREQS, TA_BINS)[SIGNAL_INPUTS:]
    generator = torch.Generator().manual_seed(34)
    for _ in range(D + 2):
        outputs = _run_wrapper(wrapper, model, state, generator)
        state = next_state(outputs)
    key, value, logit, hidden = (value.clone() for value in state)
    signals = graph_signals(
        model,
        torch.randn(1, 1, N_FREQS, 2, generator=generator),
        torch.randn(1, 1, N_FREQS, 2, generator=generator))
    with torch.no_grad():
        reference = wrapper(*signals, key, value, logit, hidden)
        key[:, :, D - 1] = float('nan')
        value[:, :, D - 1] = float('nan')
        logit[:, :, 0] = float('nan')
        poisoned = wrapper(*signals, key, value, logit, hidden)
    for got, want in zip(poisoned, reference):
        assert torch.equal(got, want)
    assert all(torch.isfinite(t).all() for t in poisoned)


def test_exported_onnx_state_contract(tmp_path):
    """The declared table is not the contract -- the shipped graph is.

    Goes through export_graph rather than torch.onnx.export directly: the
    optimizer and _pin_static_output_shapes both run there, and raw tracing
    leaves the *_out states with an unresolved middle dimension that only
    that pinning step fixes. Reading the raw graph would pin the wrong thing.
    """
    onnx = pytest.importorskip('onnx')
    from AIAEC._onnx_contract import validate_inplace_state_contract
    depth = 4
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': _checkpoint_contract(),
                'state_dict': model.state_dict()}, checkpoint)
    path = tmp_path / 'rank.onnx'
    export_graph(model, str(checkpoint), str(path))

    loaded = onnx.load(str(path))
    graph = loaded.graph
    shapes = state_shapes(depth, TA_BINS)

    def dims(value):
        return tuple(d.dim_value for d in value.type.tensor_type.shape.dim)

    ins = {}
    for value in graph.input:
        if value.name in shapes:
            assert dims(value) == shapes[value.name], value.name
            ins[value.name] = dims(value)
    outs = {}
    for value in graph.output:
        name = value.name[:-4] if value.name.endswith('_out') else value.name
        if name in shapes:
            assert dims(value) == shapes[name], value.name
            outs[name] = dims(value)
    # Every state is returned whole, so each must have an _out partner of
    # exactly its own shape or the two cannot share one address.
    assert set(outs) == set(ins) == set(shapes)
    assert all(outs[name] == ins[name] for name in outs)

    # The structure that lets the shift run in the graph: each window input
    # has one consumer, a Slice; no state output is produced by a Slice.
    producer = {name: node for node in graph.node for name in node.output}
    for name in WINDOW_STATE_NAMES:
        users = [node for node in graph.node if name in node.input]
        assert [node.op_type for node in users] == ['Slice'], name
        assert producer[name + '_out'].op_type == 'Concat', name
    assert producer['h_gru_out'].op_type != 'Slice'
    assert validate_inplace_state_contract(
        loaded, STATE_NAMES, STATE_OUTPUT_NAMES,
        window_drops=WINDOW_STATE_DROPS) == len(WINDOW_STATE_NAMES)
    props = {p.key: p.value for p in loaded.metadata_props}
    assert props['state_binding'] == 'in_place_same_address'
    assert props['state_layout_version'] == str(LAYOUT_VERSION)


def test_host_graph_ends_at_the_mask_conv(tmp_path):
    """The shipped graph's last compute is the ``complex_mask`` Conv.

    A multiply, gather or concat after it is a chain of tiny elementwise and
    data-movement ops on 2 x n_freqs elements: negligible arithmetic, but each
    is a separate accelerator op. The mask multiply belongs to the host, so
    the exported graph (after the exporter's own optimisation) must have the
    mask Conv as the producer of ``output`` -- no node between them.
    """
    onnx = pytest.importorskip('onnx')
    model = AlignULCNet(GRID, max_delay_frames=4).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': _checkpoint_contract(),
                'state_dict': model.state_dict()}, checkpoint)
    path = tmp_path / 'tail.onnx'
    export_graph(model, str(checkpoint), str(path))

    graph = onnx.load(str(path)).graph
    producer = {name: node for node in graph.node for name in node.output}
    head = producer['output']
    assert head.op_type == 'Conv'
    weight = {init.name: init for init in graph.initializer}[head.input[1]]
    assert tuple(weight.dims) == (2, 32, 1, 1)
    shape = next(v for v in graph.output if v.name == 'output')
    assert tuple(d.dim_value for d in shape.type.tensor_type.shape.dim) == (
        1, 2, 1, N_FREQS)
    assert not {'Gather', 'Mul', 'Sub', 'Concat'} & {
        node.op_type for node in graph.node
        if node.name.startswith('/complex_mask')}


def test_optimized_graph_has_no_conv_pad_op_and_replays_the_wrapper(tmp_path):
    """After the separate batch-1 cleanup the convolutions' frequency padding
    lives in their attributes, and the cleaned file still reproduces the wrapper."""
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    depth = 4
    torch.manual_seed(19)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': _checkpoint_contract(),
                'state_dict': model.state_dict()}, checkpoint)
    path = tmp_path / 'pads.onnx'
    wrapper, initial, _meta = export_graph(
        model, str(checkpoint), str(path))
    import onnx_batch1_optimizer as b1
    exported = onnx.load(str(path))
    b1.optimize_batch1(exported, verify=True)
    onnx.save(exported, str(path))

    graph = onnx.load(str(path)).graph
    # Any Pad left (the reorientation's width fit) feeds no Conv.
    conv_inputs = {name for node in graph.node if node.op_type == 'Conv'
                   for name in node.input}
    assert not [node for node in graph.node
                if node.op_type == 'Pad' and node.output[0] in conv_inputs]
    session = ort.InferenceSession(
        str(path), providers=['CPUExecutionProvider'])
    state = tuple(value.clone() for value in initial[SIGNAL_INPUTS:])
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for _ in range(2 * depth + 2):
            current = graph_signals(
                model,
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
            ) + state
            expected = wrapper(*current)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, current)})
            for got, want in zip(actual, expected):
                assert abs(got - want.numpy()).max() <= 3e-4
            state = next_state(expected)


def test_hidden_stacks_on_the_channel_axis():
    """Pins WHICH axis the two subband hiddens stack on.

    dim 0 is the singleton N. Slicing it returns the whole tensor and an
    empty one without raising, so a wrong axis is silent -- and the audio
    head alone would not show it.
    """
    depth = 4
    torch.manual_seed(5)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    reference = model.create_stream_state()
    state = dummy_inputs(depth, N_FREQS, TA_BINS)[SIGNAL_INPUTS:]
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _ in range(2 * depth + 2):
            error = _complex_frame(generator)
            far = _complex_frame(generator)
            signals = stream_features(model, _ri(error), _ri(far))[
                :SIGNAL_INPUTS]
            state = next_state(wrapper(*signals, *state))
            model.forward_stream(error, far, reference)

            merged = state[-1]
            assert merged.shape == (1, 2 * GRU_LAYERS, 1, GRU_HIDDEN)
            assert torch.equal(merged[:, :GRU_LAYERS],
                               reference['subband_gru0']._hidden.unsqueeze(0))
            assert torch.equal(merged[:, GRU_LAYERS:],
                               reference['subband_gru1']._hidden.unsqueeze(0))
    assert merged.abs().max() > 0


def test_reset_is_all_zero_external_state():
    inputs = dummy_inputs(D, N_FREQS, TA_BINS)
    for state in inputs[SIGNAL_INPUTS:]:
        assert torch.count_nonzero(state) == 0


def test_streaming_onnx_runtime_matches_pytorch(tmp_path):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    depth = 4
    torch.manual_seed(17)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    initial = dummy_inputs(depth, N_FREQS, TA_BINS)
    output_path = tmp_path / 'align_ulcnet_stream.onnx'

    torch.onnx.export(
        wrapper,
        initial,
        str(output_path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=17,
        do_constant_folding=True,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        str(output_path), providers=['CPUExecutionProvider']
    )
    assert tuple(item.name for item in session.get_inputs()) == INPUT_NAMES
    assert tuple(item.name for item in session.get_outputs()) == OUTPUT_NAMES

    state = tuple(value.clone() for value in initial[SIGNAL_INPUTS:])
    generator = torch.Generator().manual_seed(23)
    worst = 0.0
    with torch.no_grad():
        for _ in range(2 * depth + 4):
            current = stream_features(
                model,
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
            )[:SIGNAL_INPUTS] + state
            expected = wrapper(*current)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, current)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.numpy())
                )))
            state = next_state(expected)
    assert worst <= 3e-4


def test_streaming_export_rejects_a_non_verified_grid():
    """The boundary guard must fire before any shape can go wrong.

    SignalGrid admits grids the deployed boundary is not defined for -- 16 kHz
    / 256 is constructible and the model runs on it -- so a checkpoint from one
    has to be rejected here with an actionable message, never crash deep in a
    matmul nor export a graph no C build can bind. Both product grids must be
    accepted, or lifting the rate pin achieved nothing.
    """
    model = AlignULCNet(SignalGrid(16000, 256, 256, 128),
                        max_delay_frames=D).eval()
    with pytest.raises(ValueError, match='streaming export supports'):
        AlignUlcnetStreamingExport(model)

    for grid in (GRID, SignalGrid(48000, 1024, 1024, 512)):
        wrapper = AlignUlcnetStreamingExport(
            AlignULCNet(grid, max_delay_frames=D).eval())
        assert wrapper.n_freqs == grid.n_freqs

    # The C-SamFR sampling pair is not implied by the grid, so it keeps a
    # guard of its own.
    off = AlignULCNet(GRID, max_delay_frames=D, gamma=4).eval()
    with pytest.raises(ValueError, match='gamma, subband_bins'):
        AlignUlcnetStreamingExport(off)
