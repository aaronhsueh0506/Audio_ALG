"""``PTQ/calib_analyze.softmax_path`` on an exported DeepVQE-S graph: the
attention logits are marked, the encoder outputs end the walk."""

import pytest

import calib_analyze

from test_training_options import _export, _option_checkpoint

onnx = pytest.importorskip('onnx')


def test_the_deepvqe_attention_logits_reach_the_softmax_through_no_activation(
        tmp_path):
    checkpoint = tmp_path / 'model.pth'
    model, _ = _option_checkpoint('DeepVQE_S', {'activation': 'prelu'},
                                  checkpoint)
    graph_path = tmp_path / 'model.onnx'
    _export('DeepVQE_S', model, checkpoint, str(graph_path), 1, verify=False)
    graph = onnx.load(str(graph_path)).graph
    path = calib_analyze.softmax_path(graph)
    # The frequency sum of q * k (the score conv's input) and the two states
    # the logits are made from are on the way; the value ring comes after the
    # Softmax.
    assert '/ReduceSum_output_0' in path
    assert {'state_align_key_ring', 'state_align_score_history'} <= path
    assert 'state_align_value_ring' not in path
    # The walk stops at the outputs of the two encoders that make q and k:
    # the microphone and far-end inputs reach the Softmax only through them.
    producer = {name: node for node in graph.node for name in node.output}
    stops = [name for name in path
             if name in producer and producer[name].op_type == 'PRelu']
    assert len(stops) == 2
    assert not {'mic', 'far'} & path
