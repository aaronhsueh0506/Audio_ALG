#!/usr/bin/env python3
"""Refuse a quantized streaming graph whose state pairs cannot share an address.

A state input and its ``*_out`` output are bound to one buffer. The buffer is
read as fixed point by the next invocation, so the two tensors must agree in
every property that defines the fixed-point encoding: element format,
zero point, scale and dimensions. A quantizer is free to pick different
scales for the two (they are different tensors in the float graph), and that
silently rescales the state at every step.

Two stages:

1. The float ONNX graph: ``validate_inplace_state_from_metadata`` -- rank 4,
   equal shapes, windows and row stacks consumed by explicit Slices, no state
   output produced by a Slice, ``state_binding`` recorded. Always run.
2. The quantization report, when one is given: for every state pair the
   dtype, zero point, scale and shape must be identical.

Quantization report interface (JSON; a toolchain adapter writes it)::

    {"tensors": {"<graph tensor name>": {
        "dtype": "int8",          # element format of the quantized tensor
        "scale": 0.0123,          # or a list for per-channel scales
        "zero_point": 0,          # or a list for per-channel zero points
        "shape": [1, 5, 1, 256]}}}

Every state input and its output must appear in ``tensors``. Exit status 0
when both stages pass, 1 on any difference, 2 on a usage or format error.

Usage::

    python3 check_state_quantization.py graph.onnx [--report quant.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from onnx_streaming_contract import validate_inplace_state_from_metadata  # noqa: E402

_FIELDS = ('dtype', 'scale', 'zero_point', 'shape')


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def check_quantized_state_pairs(handoff, tensors):
    """Return the list of differences between each state input and output.

    ``handoff`` maps a state input name to its output name; ``tensors`` is the
    report's ``tensors`` mapping. An empty list means every pair agrees.
    """
    problems = []
    for name, back in handoff.items():
        missing = [item for item in (name, back) if item not in tensors]
        if missing:
            problems.append('%s -> %s: missing from the report: %s'
                            % (name, back, ', '.join(missing)))
            continue
        left, right = tensors[name], tensors[back]
        for field in _FIELDS:
            if field not in left or field not in right:
                problems.append('%s -> %s: report lacks %s'
                                % (name, back, field))
            elif _as_list(left[field]) != _as_list(right[field]):
                problems.append('%s -> %s: %s differs (%r vs %r)'
                                % (name, back, field, left[field],
                                   right[field]))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     RawDescriptionHelpFormatter)
    parser.add_argument('graph')
    parser.add_argument('--report', default=None,
                        help='quantization report (JSON, see above)')
    args = parser.parse_args(argv)

    import onnx

    graph = onnx.load(args.graph)
    try:
        pairs = validate_inplace_state_from_metadata(graph)
    except RuntimeError as error:
        print('FAIL graph: %s' % error)
        return 1
    print('PASS graph: %d state pairs are bindable in place' % pairs)
    if args.report is None:
        print('NOT_RUN quantized check: no --report given')
        return 0
    try:
        with open(args.report, encoding='utf-8') as stream:
            report = json.load(stream)
        tensors = report['tensors']
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('ERROR report: %s' % error)
        return 2
    props = {prop.key: prop.value for prop in graph.metadata_props}
    problems = check_quantized_state_pairs(
        json.loads(props['state_handoff']), tensors)
    for problem in problems:
        print('FAIL quantized: %s' % problem)
    if problems:
        return 1
    print('PASS quantized: every state pair agrees in dtype, scale, zero '
          'point and shape')
    return 0


if __name__ == '__main__':
    sys.exit(main())
