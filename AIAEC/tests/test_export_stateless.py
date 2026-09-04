"""Explicit-state accelerator boundary tests for AIAEC candidates."""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

from AIAEC.aiaec_common import SignalGrid, log_power_feature
from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.CAGCRN import CAGCRN
from AIAEC.DeepVQE_S import DeepVQES
from AIAEC._streaming_export import (
    GRU_STATE_LAYOUTS,
    StatelessOneFrameAIAEC,
    _build,
    requires_contiguous_calibration,
    state_precision_policy,
)
from AIAEC._streaming_calibration import (
    ALL_MODEL_NAMES as CALIBRATION_MODEL_NAMES,
    far_mode_provenance,
)
from AIAEC.training_common import DEPLOYED_FAR_INPUT_MODE


GRID = SignalGrid(16000, 512, 512, 256)


def test_export_log_power_matches_training_floor_semantics():
    """The exported graph must use the model's clamp, not add an epsilon."""
    complex_spec = torch.tensor(
        [[[0.0 + 0.0j, 1.0e-8 + 0.0j, 1.0e-5 + 2.0e-5j]]],
        dtype=torch.complex64,
    )
    ri_spec = torch.view_as_real(complex_spec)
    torch.testing.assert_close(
        StatelessOneFrameAIAEC._log_power(ri_spec),
        log_power_feature(complex_spec),
        rtol=0.0,
        atol=0.0,
    )


def _learned_output(name, output):
    if name == 'Align_CRUSE':
        return output.mask
    if name == 'DeepVQE_S':
        taps = output.auxiliary['ccm_taps']
        return torch.stack((taps.real, taps.imag), dim=-1).flatten(start_dim=3)
    if name == 'CAGCRN':
        return output.mask.permute(0, 2, 3, 1)
    raise AssertionError(name)


@pytest.mark.parametrize('name,factory', (
    ('Align_CRUSE', lambda: AlignCRUSE(GRID)),
    ('DeepVQE_S', lambda: DeepVQES(GRID)),
    ('CAGCRN', lambda: CAGCRN(GRID)),
))
def test_external_state_round_trip_matches_streaming_reference(name, factory):
    torch.manual_seed(81)
    model = factory().eval()
    wrapper, dummy, input_names, output_names, split = _build(name, model)
    assert len(input_names) == len(dummy)
    external_state = dummy[split.signal_inputs:]
    reference_state = model.create_stream_state()
    observed_nonzero_state = False
    with torch.no_grad():
        for _ in range(6):
            primary_ri = torch.randn_like(dummy[0])
            far_ri = torch.randn_like(dummy[1])
            actual = wrapper(primary_ri, far_ri, *external_state)
            reference = model.forward_stream(
                torch.complex(primary_ri[..., 0], primary_ri[..., 1]),
                torch.complex(far_ri[..., 0], far_ri[..., 1]),
                reference_state,
            )
            torch.testing.assert_close(
                actual[0], _learned_output(name, reference),
                rtol=1e-5, atol=2e-6,
            )
            external_state = actual[split.head_outputs:]
            observed_nonzero_state |= any(
                value.dtype.is_floating_point
                and bool(torch.count_nonzero(value))
                for value in external_state
            )
    assert len(output_names) == len(actual)
    assert observed_nonzero_state


def test_deepvqe_head_is_rank4_and_preserves_ccm_memory_order():
    """The NPU boundary is compact without changing a single tap value."""
    torch.manual_seed(91)
    model = DeepVQES(GRID).eval()
    wrapper, inputs, _names, _outputs, _split = _build('DeepVQE_S', model)
    reference_state = model.create_stream_state()
    primary = torch.complex(inputs[0][..., 0], inputs[0][..., 1])
    far = torch.complex(inputs[1][..., 0], inputs[1][..., 1])
    with torch.no_grad():
        actual = wrapper(*inputs)[0]
        reference = model.forward_stream(primary, far, reference_state)
    taps = reference.auxiliary['ccm_taps']
    unpacked = torch.stack((taps.real, taps.imag), dim=-1)
    assert actual.shape == (1, 1, GRID.n_freqs, 18)
    assert torch.equal(actual, unpacked.reshape(1, 1, GRID.n_freqs, 18))


def test_align_cruse_frame_index_is_explicit_int64_state():
    model = AlignCRUSE(GRID).eval()
    _wrapper, inputs, names, _outputs, _split = _build('Align_CRUSE', model)
    index = names.index('state_align_frame_index')
    assert inputs[index].dtype == torch.int64
    assert inputs[index].ndim == 0


def test_align_cruse_cumulative_state_is_excluded_from_integer_ptq():
    assert state_precision_policy('Align_CRUSE') == {
        'state_align_score_sum': 'float32_no_ptq',
        'state_align_frame_index': 'int64_no_ptq',
    }
    assert state_precision_policy('DeepVQE_S') == {}
    # The same cumulative state drives both calibration rules.
    assert requires_contiguous_calibration('Align_CRUSE')
    assert not requires_contiguous_calibration('DeepVQE_S')


def test_precision_policy_names_are_real_graph_inputs():
    """A policy entry naming a tensor the graph does not have is inert.

    Nothing downstream would fail: the exporter writes the policy into the
    metadata verbatim and the calibration recorder only ever looks entries up
    by an existing tensor name, so a renamed state slot would silently leave
    the accumulator inside integer PTQ.
    """
    model = AlignCRUSE(GRID).eval()
    _wrapper, inputs, names, _outputs, _split = _build('Align_CRUSE', model)
    policy = state_precision_policy('Align_CRUSE')
    assert policy
    assert set(policy) <= set(names)

    # score_sum must be a float32 tensor: the policy claims float32_no_ptq,
    # and that claim is only meaningful if the graph input really is float32.
    score_sum = inputs[names.index('state_align_score_sum')]
    assert score_sum.dtype == torch.float32
    assert policy['state_align_score_sum'] == 'float32_no_ptq'


def test_align_ulcnet_calibration_provenance_is_not_deployment_mode():
    assert 'Align_ULCNet' in CALIBRATION_MODEL_NAMES
    assert far_mode_provenance('Align_ULCNet') == (
        'raw_far', 'aligned_far'
    )
    assert far_mode_provenance('DeepVQE_S') == (
        'model_native_far', 'model_native_far'
    )
    root = Path(__file__).resolve().parents[1]
    for model_name in CALIBRATION_MODEL_NAMES:
        model_root = root / model_name
        assert (model_root / 'export_onnx.py').is_file(), model_name
        assert (model_root / 'inference.py').is_file(), model_name


@pytest.mark.parametrize('model_name', CALIBRATION_MODEL_NAMES)
def test_calib_subcommand_records_against_its_own_model(
        model_name, monkeypatch):
    """``inference.py calib`` must reach the recorder naming ITS OWN model.

    Driving the real dispatcher rather than searching the file for the call is
    the point: a source-text match is satisfied by a line that never runs, and
    it cannot tell a model wired to a sibling's name from a correct one.
    """
    from AIAEC import _streaming_calibration

    inference = importlib.import_module('AIAEC.%s.inference' % model_name)
    recorded = []
    monkeypatch.setattr(_streaming_calibration, 'main', recorded.append)
    monkeypatch.setattr(
        sys, 'argv', ['inference.py', 'calib', '--checkpoint', 'unused']
    )
    inference.cli()
    assert recorded == [model_name]


def test_calibration_deployment_mode_equals_the_ulcnet_exporter_literal():
    """Two files must name the SAME deployment seam, so compare them directly.

    The calibration report says what deployment will feed the model; the
    ULCNet exporter stamps the value the board compares against. Asserting
    each against its own literal would let them drift apart while both tests
    stayed green.
    """
    from AIAEC.Align_ULCNet.export_onnx import _write_metadata
    from AIAEC.Align_ULCNet.model import AlignULCNet

    import tempfile

    model = AlignULCNet(GRID, max_delay_frames=2).eval()
    with tempfile.TemporaryDirectory() as work:
        checkpoint = os.path.join(work, 'ckpt.pt')
        with open(checkpoint, 'wb') as stream:
            stream.write(b'not a real checkpoint, only hashed')
        from AIAEC.Align_ULCNet.export_onnx import (
            AlignUlcnetStreamingExport,
            dummy_inputs,
        )
        wrapper = AlignUlcnetStreamingExport(model).eval()
        inputs = dummy_inputs(2, wrapper.n_freqs, wrapper.ta_bins)
        with torch.no_grad():
            outputs = wrapper(*inputs)
        metadata = _write_metadata(
            os.path.join(work, 'model.onnx'), checkpoint, model,
            {'far_input_mode': 'raw_far'}, inputs, outputs,
        )
    exported = metadata['far_input_mode']
    assert exported == DEPLOYED_FAR_INPUT_MODE
    assert far_mode_provenance('Align_ULCNet')[1] == exported
    # And the calibration side must NOT claim the deployment mode describes
    # the recorded data.
    assert far_mode_provenance('Align_ULCNet')[0] != exported


@pytest.mark.parametrize('name,factory', (
    ('Align_CRUSE', lambda: AlignCRUSE(GRID)),
    ('DeepVQE_S', lambda: DeepVQES(GRID)),
    ('CAGCRN', lambda: CAGCRN(GRID)),
))
def test_stateless_graph_really_lowers_to_onnx_and_replays_state(
        name, factory, tmp_path):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from AIAEC._streaming_export import _verify_onnx

    model = factory().eval()
    wrapper, inputs, input_names, output_names, split = _build(name, model)
    path = os.fspath(tmp_path / (name + '.onnx'))
    torch.onnx.export(
        wrapper,
        inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    assert _verify_onnx(
        path, wrapper, inputs, input_names, split, steps=3
    ) < 3e-4


@pytest.mark.parametrize('name,factory,combinable', (
    ('CAGCRN', lambda: CAGCRN(GRID), True),
    ('Align_CRUSE', lambda: AlignCRUSE(GRID), False),
    ('DeepVQE_S', lambda: DeepVQES(GRID), False),
))
def test_combined_gru_state_layout_only_regroups_the_boundary(
        name, factory, combinable):
    """The combined layout must change where tensors are cut, nothing else.

    It also must not claim to combine a model that has one GRU hidden: those
    are served the split layout whatever was asked for, and the wrapper --
    not the request -- is what the exported metadata records.
    """
    assert 'combined' in GRU_STATE_LAYOUTS
    torch.manual_seed(81)
    model = factory().eval()
    split, split_inputs, split_names, _outs, graph = _build(name, model)
    combined, combined_inputs, combined_names, _o, _g = _build(
        name, model, 'combined')

    assert combined.gru_state_layout == ('combined' if combinable else 'split')
    if not combinable:
        assert combined_names == split_names
        return

    hidden = len(combined._gru_slots)
    assert hidden > 1
    assert len(combined_names) == len(split_names) - hidden + 1
    assert combined_names[-1] == combined.COMBINED_GRU_STATE_NAME
    # Regrouping must not change how many state values cross the boundary.
    assert sum(value.numel() for value in combined_inputs[graph.signal_inputs:]) \
        == sum(value.numel() for value in split_inputs[graph.signal_inputs:])

    split_state = tuple(value.clone()
                        for value in split_inputs[graph.signal_inputs:])
    combined_state = tuple(value.clone()
                           for value in combined_inputs[graph.signal_inputs:])
    generator = torch.Generator().manual_seed(11)
    observed_nonzero = False
    with torch.no_grad():
        for _ in range(5):
            signals = tuple(
                torch.randn(value.shape, generator=generator)
                for value in split_inputs[:graph.signal_inputs]
            )
            expected = split(*(signals + split_state))
            actual = combined(*(signals + combined_state))
            torch.testing.assert_close(actual[0], expected[0],
                                       rtol=0, atol=0)
            split_state = expected[1:]
            combined_state = actual[1:]
            observed_nonzero = observed_nonzero or bool(
                combined_state[-1].abs().max() > 0)
    # Written so it can FAIL: zero state would satisfy every comparison above.
    assert observed_nonzero, 'state never left its zero initialisation'
