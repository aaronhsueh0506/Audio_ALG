"""The exported streaming graphs may pad frequency, never time."""

import numpy as np
import pytest

from onnx_streaming_contract import validate_nctf_no_temporal_padding


def _pad_graph(pads):
    onnx = pytest.importorskip('onnx')
    from onnx import TensorProto, helper, numpy_helper

    source = helper.make_tensor_value_info(
        'x', TensorProto.FLOAT, [1, 1, 3, 8]
    )
    output = helper.make_tensor_value_info(
        'y', TensorProto.FLOAT, None
    )
    initializer = numpy_helper.from_array(
        np.asarray(pads, dtype=np.int64), name='pads'
    )
    node = helper.make_node('Pad', ('x', 'pads'), ('y',), name='pad')
    return helper.make_model(helper.make_graph(
        (node,), 'padding-contract', (source,), (output,), (initializer,)
    ))


def _conv_graph(pads):
    onnx = pytest.importorskip('onnx')
    from onnx import TensorProto, helper, numpy_helper

    source = helper.make_tensor_value_info(
        'x', TensorProto.FLOAT, [1, 1, 3, 8]
    )
    output = helper.make_tensor_value_info(
        'y', TensorProto.FLOAT, None
    )
    weight = numpy_helper.from_array(
        np.ones((1, 1, 3, 3), dtype=np.float32), name='weight'
    )
    node = helper.make_node(
        'Conv', ('x', 'weight'), ('y',), name='conv',
        kernel_shape=(3, 3), pads=pads,
    )
    return helper.make_model(helper.make_graph(
        (node,), 'convolution-padding-contract', (source,), (output,),
        (weight,)
    ))


def test_frequency_only_rank4_pad_is_accepted():
    graph = _pad_graph([0, 0, 0, 1, 0, 0, 0, 1])
    assert validate_nctf_no_temporal_padding(graph) == 1


@pytest.mark.parametrize('pads', (
    [0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 1, 1],
))
def test_rank4_time_pad_is_rejected(pads):
    with pytest.raises(RuntimeError, match='pads the time axis'):
        validate_nctf_no_temporal_padding(_pad_graph(pads))


def test_frequency_only_conv_padding_is_accepted():
    assert validate_nctf_no_temporal_padding(
        _conv_graph([0, 1, 0, 1])
    ) == 1


def test_conv_time_padding_is_rejected():
    with pytest.raises(RuntimeError, match='pads the time axis'):
        validate_nctf_no_temporal_padding(_conv_graph([1, 1, 1, 1]))
