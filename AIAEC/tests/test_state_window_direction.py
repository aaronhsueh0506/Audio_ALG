"""Each state window has one correct direction, and the validator holds it.

A time-ordered window is ``cat(W[..., 1:K], new)`` and drops its first frame;
a newest-first ring is ``cat(new, W[..., 0:K-1])`` and drops its last one. The
exporter declares the end per state in the graph metadata
(``state_window_drop``) and the validator accepts exactly that end. For each
of the four models, every window of the real exported graph is mutated to
drop the other end (and, separately, declared the other way) and must be
refused.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.Align_ULCNet.export_onnx import (
    STATE_NAMES as ULCNET_STATE_NAMES,
    STATE_OUTPUT_NAMES as ULCNET_STATE_OUTPUT_NAMES,
    export_graph as export_ulcnet,
)
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.CAGCRN import CAGCRN
from AIAEC.DeepVQE_S import DeepVQES
from AIAEC._onnx_contract import validate_inplace_state_contract
from AIAEC._streaming_export import _build, export_graph

GRID = SignalGrid(16000, 512, 512, 256)
OTHER_END = {'first': 'last', 'last': 'first'}

# The newest-first rings, by their state names; every other window is a
# time-ordered convolution or score window.
RINGS = {
    'Align_ULCNet': {'key_history', 'value_history'},
    'Align_CRUSE': {'state_align_key_ring', 'state_align_value_ring'},
    'DeepVQE_S': {'state_align_key_ring', 'state_align_value_ring'},
    'CAGCRN': {'state_cata_ring_ring'},
}


def _export(name, tmp_path):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    torch.manual_seed(91)
    checkpoint = tmp_path / 'ckpt.pth'
    path = tmp_path / (name + '.onnx')
    if name == 'Align_ULCNet':
        model = AlignULCNet(GRID, max_delay_frames=4).eval()
        torch.save({'contract': {}, 'state_dict': model.state_dict()},
                   checkpoint)
        export_ulcnet(model, str(checkpoint), str(path))
    else:
        factory = {'Align_CRUSE': AlignCRUSE, 'DeepVQE_S': DeepVQES,
                   'CAGCRN': CAGCRN}[name]
        built = _build(name, factory(GRID).eval())
        checkpoint.write_bytes(b'hashed only')
        grid = SimpleNamespace(sr=16000, n_fft=512, win_len=512, hop_len=256)
        export_graph(grid, built, str(checkpoint), str(path), 0)
    return onnx.load(str(path))


def _declared(graph, key):
    props = {prop.key: prop.value for prop in graph.metadata_props}
    return json.loads(props[key])


def _flip_slice(graph, name):
    """Point the Slice that reads window ``name`` at the other end of the
    window, keeping its length: [1:K) becomes [0:K-1) and back."""
    from onnx import numpy_helper

    constants = {item.name: numpy_helper.to_array(item)
                 for item in graph.graph.initializer}
    node = next(item for item in graph.graph.node if name in item.input)
    start, end = (int(constants[node.input[index]].reshape(-1)[0])
                  for index in (1, 2))
    new = (0, end - 1) if start == 1 else (1, end + 1)
    for index, value in ((1, new[0]), (2, new[1])):
        label = 'flipped_%s_%d' % (name, index)
        graph.graph.initializer.append(numpy_helper.from_array(
            np.array([value], dtype=np.int64), label))
        node.input[index] = label


@pytest.mark.parametrize('name', sorted(RINGS))
def test_a_window_that_drops_the_wrong_end_is_refused(name, tmp_path):
    graph = _export(name, tmp_path)
    drops = _declared(graph, 'state_window_drop')
    if name == 'Align_ULCNet':
        inputs = list(ULCNET_STATE_NAMES)
        outputs = list(ULCNET_STATE_OUTPUT_NAMES)
    else:
        handoff = _declared(graph, 'state_handoff')
        inputs, outputs = list(handoff), list(handoff.values())

    assert set(drops) <= set(inputs)
    assert {item for item in drops if drops[item] == 'last'} == RINGS[name]
    assert {drops[item] for item in drops} <= {'first', 'last'}
    assert validate_inplace_state_contract(
        graph, inputs, outputs, window_drops=drops) == len(drops)

    for window, drop in drops.items():
        # The graph drops the other end than the one declared.
        mutant = type(graph)()
        mutant.CopyFrom(graph)
        _flip_slice(mutant, window)
        with pytest.raises(
                RuntimeError,
                match='%s: Slice .* does not drop the %s slot'
                % (window, drop)):
            validate_inplace_state_contract(
                mutant, inputs, outputs, window_drops=drops)
        # The declaration names the other end than the graph drops.
        wrong = dict(drops, **{window: OTHER_END[drop]})
        with pytest.raises(
                RuntimeError,
                match='%s: Slice .* does not drop the %s slot'
                % (window, OTHER_END[drop])):
            validate_inplace_state_contract(
                graph, inputs, outputs, window_drops=wrong)
