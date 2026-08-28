"""Explicit delta-state export contract for streaming Align-ULCNet."""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet.export_onnx import (
    AlignUlcnetStreamingExport,
    host_output,
    stream_features,
    GRU_HIDDEN,
    GRU_LAYERS,
    COMBINED_GRU_STATE_NAME,
    COMPRESSION_EXPONENT,
    FEATURE_LAYOUTS,
    GRU_STATE_LAYOUTS,
    INPUT_NAMES,
    LAYOUT_VERSIONS,
    RETIRED_LAYOUT_VERSIONS,
    SIGNAL_INPUTS,
    MAX_DELAY_DEPTH,
    MIN_DELAY_DEPTH,
    OUTPUT_NAMES,
    SCORE_HISTORY_FRAMES,
    STATE_LAYOUT_VERSION,
    SUPPORTED_GRIDS,
    ta_bins_for,
    TA_CHANNELS,
    _write_metadata,
    dummy_inputs,
    export_graph,
    graph_signals,
    next_state,
    resolve_layout,
    state_shapes,
)
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.training_common import FAR_INPUT_MODE_C_VALUES


GRID = SignalGrid(16000, 512, 512, 256)
# The 16 kHz values of the two grid-derived quantities. ta_bins_for()
# is the single derivation; test_ta_bins_follows_the_grid pins both.
N_FREQS = GRID.n_freqs
TA_BINS = 26
D = 8


def _complex_frame(generator):
    return torch.complex(
        torch.randn(1, 1, GRID.n_freqs, generator=generator),
        torch.randn(1, 1, GRID.n_freqs, generator=generator),
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


def test_streaming_export_shapes_are_fixed_and_delta_only():
    shapes = state_shapes(D, TA_BINS)
    assert shapes['key_history'] == (1, 32, D - 1, 26)
    assert shapes['value_history'] == (1, 32, D - 1, 26)
    assert shapes['logit_history'] == (1, 32, 4, D)
    assert shapes['h_gru0'] == (1, 2, 1, 128)
    assert shapes['h_gru1'] == (1, 2, 1, 128)
    assert shapes['h_gru'] == (1, 4, 1, 128)
    # Every boundary state is rank-4 NCHW. Written once, as a rule, so a new
    # state cannot be added at the wrong rank without failing here.
    assert all(len(shape) == 4 for shape in shapes.values())
    assert INPUT_NAMES == (
        'error_mag', 'far_mag', 'error_cos', 'error_sin', 'error_ri',
        'key_history', 'value_history', 'logit_history', 'h_gru0', 'h_gru1',
    )
    assert OUTPUT_NAMES == (
        'output', 'key_now', 'value_now', 'logit_now',
        'h_gru0_out', 'h_gru1_out',
    )


def test_every_boundary_pair_has_its_own_version():
    pairs = {(feature, gru)
             for feature in FEATURE_LAYOUTS for gru in GRU_STATE_LAYOUTS}
    assert set(LAYOUT_VERSIONS) == pairs
    assert len(set(LAYOUT_VERSIONS.values())) == len(LAYOUT_VERSIONS)
    # ulcnet_model_io.h implements exactly one pair; every other pair must
    # therefore carry a version the header rejects.
    # With every version distinct, pinning the shipped pair is enough: no
    # other pair can then carry the version ulcnet_model_io.h implements.
    assert LAYOUT_VERSIONS[('host', 'split')] == STATE_LAYOUT_VERSION
    # 3-7 denoted rank-3 boundaries. Reusing one would make a single number
    # mean two different contracts, which no runtime could tell apart. Tied to
    # the shipped pair rather than a literal floor, which would still pass
    # after the next bump while no longer guarding anything.
    assert not (set(LAYOUT_VERSIONS.values()) & RETIRED_LAYOUT_VERSIONS)
    assert min(LAYOUT_VERSIONS.values()) == STATE_LAYOUT_VERSION


def test_graph_feature_layout_restores_the_pre_host_boundary():
    layout = resolve_layout('graph', 'split')
    assert layout.input_names == (
        'error', 'far',
        'key_history', 'value_history', 'logit_history', 'h_gru0', 'h_gru1',
    )
    assert layout.output_names == OUTPUT_NAMES
    # Same tensor names and semantics as the pre-host boundary, but the
    # recurrent hiddens are rank-4 now, so it cannot reuse that boundary's
    # retired number.
    assert layout.layout_version == 10


@pytest.mark.parametrize('feature', sorted(FEATURE_LAYOUTS))
@pytest.mark.parametrize('gru', sorted(GRU_STATE_LAYOUTS))
def test_every_layout_pair_computes_the_same_frames(feature, gru):
    """The four boundaries differ only in where tensors are cut, so a run
    through any of them must reproduce the shipped pair bit for bit."""
    torch.manual_seed(7)
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    reference = AlignUlcnetStreamingExport(model).eval()
    candidate = AlignUlcnetStreamingExport(
        model, feature_layout=feature, gru_state_layout=gru).eval()

    def start(wrapper):
        return tuple(value.clone() for value in
                     dummy_inputs(D, N_FREQS, TA_BINS, wrapper.layout)[
                         wrapper.layout.signal_inputs:])

    def head(wrapper, outputs):
        # Compare in one domain: the host layout leaves the inverse signed
        # power to host_output(), the graph layout has already applied it.
        if wrapper.layout.in_graph_features:
            return outputs[0]
        return host_output(model, outputs[0])

    reference_state, candidate_state = start(reference), start(candidate)
    generator = torch.Generator().manual_seed(11)
    observed_nonzero = False
    with torch.no_grad():
        for _ in range(2 * D + 5):
            error_ri = _ri(_complex_frame(generator))
            far_ri = _ri(_complex_frame(generator))
            expected = reference(*(
                graph_signals(model, error_ri, far_ri, reference.layout)
                + reference_state))
            actual = candidate(*(
                graph_signals(model, error_ri, far_ri, candidate.layout)
                + candidate_state))
            torch.testing.assert_close(
                head(candidate, actual), head(reference, expected),
                rtol=0, atol=0)
            observed_nonzero = observed_nonzero or bool(
                candidate_state[-1].abs().max() > 0)
            reference_state = next_state(reference_state, expected, D)
            candidate_state = next_state(candidate_state, actual, D)
    assert observed_nonzero, 'state never left its zero initialisation'


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
        'ULCNET_MODEL_IO_SCORE_HISTORY': SCORE_HISTORY_FRAMES,
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

    # v5 moved the compression exponent out of the graph; the C define and
    # the exporter constant can now drift silently, so pin them here (the
    # exporter separately refuses checkpoints with any other exponent).
    match = re.search(
        r'^#define\s+ULCNET_MODEL_IO_COMPRESSION_EXP\s+([0-9.]+)f\s*$',
        header,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert float(match.group(1)) == COMPRESSION_EXPONENT


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
        {'far_input_mode': 'raw_far'}, inputs, outputs,
    )
    assert metadata['training_far_input_mode'] == 'raw_far'
    assert metadata['far_input_mode'] == 'aligned_far'
    assert metadata['far_input_mode_c_value'] == FAR_INPUT_MODE_C_VALUES[
        'aligned_far'
    ]
    assert metadata['state_layout_version'] == STATE_LAYOUT_VERSION

    # Legacy checkpoints retain raw-far provenance while deployment remains
    # fixed to aligned far.
    legacy = _write_metadata(
        str(tmp_path / 'legacy.onnx'), str(checkpoint), model, {},
        inputs, outputs,
    )
    assert legacy['training_far_input_mode'] == 'raw_far'
    assert legacy['far_input_mode'] == 'aligned_far'
    assert legacy['far_input_mode_c_value'] == 1


def test_delta_state_wrapper_matches_forward_stream_frame_by_frame():
    torch.manual_seed(20260816)
    model = AlignULCNet(GRID, max_delay_frames=D).eval()
    wrapper = AlignUlcnetStreamingExport(model).eval()
    explicit = tuple(value.clone() for value in dummy_inputs(D, N_FREQS, TA_BINS)[SIGNAL_INPUTS:])
    reference = model.create_stream_state()
    generator = torch.Generator().manual_seed(73)

    with torch.no_grad():
        for _ in range(2 * D + 5):
            error = _complex_frame(generator)
            far = _complex_frame(generator)
            signals = stream_features(model, _ri(error), _ri(far))
            outputs = wrapper(*signals, *explicit)
            expected = model.forward_stream(error, far, reference)

            # The graph output is the COMPRESSED estimate; the host applies
            # the inverse signed power (host_output), completing the chain.
            assert torch.allclose(
                host_output(model, outputs[0]), _ri(expected.enhanced),
                atol=5e-7, rtol=1e-6
            )
            explicit = next_state(explicit, outputs, D)

            align = reference['align']
            assert torch.equal(
                explicit[0], align.key_ring._ring[:, :, :D - 1]
            )
            assert torch.equal(
                explicit[1], align.value_ring._ring[:, :, :D - 1]
            )
            assert torch.equal(
                explicit[2], align.score_cell._history
            )
            # The boundary carries rank-4 NCHW; the streaming reference holds
            # nn.GRU's rank-3 hidden. Lift the reference rather than squeezing
            # the boundary, so this still pins where the new axis sits.
            assert explicit[3].shape == (1, GRU_LAYERS, 1, GRU_HIDDEN)
            assert explicit[4].shape == (1, GRU_LAYERS, 1, GRU_HIDDEN)
            assert torch.equal(
                explicit[3], reference['subband_gru0']._hidden.unsqueeze(0)
            )
            assert torch.equal(
                explicit[4], reference['subband_gru1']._hidden.unsqueeze(0)
            )


def test_exported_onnx_state_tensors_are_rank_four(tmp_path):
    """The declared table is not the contract -- the shipped graph is.

    Goes through export_graph rather than torch.onnx.export directly: the
    optimizer and _pin_static_output_shapes both run there, and raw tracing
    leaves the *_out states with an unresolved middle dimension that only
    that pinning step fixes. Reading the raw graph would pin the wrong thing.
    """
    onnx = pytest.importorskip('onnx')
    depth = 4
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    torch.save({'contract': {}, 'state_dict': model.state_dict()}, checkpoint)
    path = tmp_path / 'rank.onnx'
    export_graph(model, str(checkpoint), str(path))

    graph = onnx.load(str(path)).graph
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
    # The GRU hiddens are REPLACED whole, so each must have an _out partner of
    # exactly its own shape or the recurrence cannot be fed back. The three
    # caches are deliberately not here: their outputs are the delta-only
    # key_now/value_now/logit_now, one frame rather than the ring.
    assert set(outs) == {'h_gru0', 'h_gru1'}, sorted(outs)
    assert all(outs[name] == ins[name] for name in outs)
    assert set(ins) == set(shapes) - {COMBINED_GRU_STATE_NAME}


def test_combined_state_stacks_on_the_channel_axis():
    """Pins WHICH axis the combined layout stacks on.

    dim 0 is the singleton N. Slicing it returns the whole tensor and an
    empty one without raising, so a wrong axis is silent -- and the
    all-pairs equivalence test only compares the audio head, not the state.
    """
    depth = 4
    torch.manual_seed(5)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    split = AlignUlcnetStreamingExport(model, gru_state_layout='split').eval()
    combined = AlignUlcnetStreamingExport(
        model, gru_state_layout='combined').eval()

    split_state = dummy_inputs(depth, N_FREQS, TA_BINS, split.layout)[SIGNAL_INPUTS:]
    combined_state = dummy_inputs(depth, N_FREQS, TA_BINS, combined.layout)[SIGNAL_INPUTS:]
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _ in range(2 * depth + 2):
            signals = stream_features(
                model,
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
                torch.randn(1, 1, GRID.n_freqs, 2, generator=generator),
            )
            split_out = split(*(signals + split_state))
            combined_out = combined(*(signals + combined_state))
            split_state = next_state(split_state, split_out, depth)
            combined_state = next_state(combined_state, combined_out, depth)

            merged = combined_state[-1]
            assert merged.shape == (1, 2 * GRU_LAYERS, 1, GRU_HIDDEN)
            assert torch.equal(merged[:, :GRU_LAYERS], split_state[-2])
            assert torch.equal(merged[:, GRU_LAYERS:], split_state[-1])


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
            ) + state
            expected = wrapper(*current)
            actual = session.run(None, {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, current)
            })
            for got, want in zip(actual, expected):
                worst = max(worst, float(np.max(
                    np.abs(got - want.numpy())
                )))
            state = next_state(state, expected, depth)
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
