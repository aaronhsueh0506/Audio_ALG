"""ONNX contract checks for AIAEC's stateless streaming graphs.

The validators are implemented once, in ``AINR/onnx_streaming_contract.py``,
which is the module every exporter uses. ``AINR/`` is a sibling of ``AIAEC/``
in a full checkout and in a standalone AIAEC copy alike.

``validate_inplace_state_contract`` takes AIAEC's name lists instead of the
shared module's ``state_handoff`` mapping. A window state is a history cut by
one Slice over the time axis; a time-ordered window drops its oldest frame, a
newest-first K/V or attention ring drops its last slot, so either end is
accepted.
"""

import os
import sys

_AINR_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AINR')
if os.path.isdir(_AINR_ROOT) and _AINR_ROOT not in sys.path:
    # Appended, not inserted: only this one module is wanted from AINR, and the
    # names AIAEC already resolves must keep resolving where they did.
    sys.path.append(_AINR_ROOT)

from onnx_streaming_contract import (  # noqa: E402
    STATE_BINDING,
    validate_inplace_state_graph,
    validate_nctf_no_temporal_padding,
)

WINDOW_TIME_AXIS = 2


def validate_inplace_state_contract(model, state_inputs, state_outputs,
                                    window_inputs=(), require_rank4=True):
    """Check the same-address state boundary of a streaming graph.

    ``state_inputs`` and ``state_outputs`` are the graph's state tensor names,
    pairwise in the same order. ``window_inputs`` is the subset that are
    history windows (time axis 2). See ``validate_inplace_state_graph`` for
    the rules. Returns the number of window inputs checked.
    """
    state_inputs = list(state_inputs)
    state_outputs = list(state_outputs)
    if len(state_inputs) != len(state_outputs):
        raise RuntimeError('state input/output name lists differ in length')
    if len(set(state_inputs)) != len(state_inputs):
        raise RuntimeError('state input names repeat')
    validate_inplace_state_graph(
        model, dict(zip(state_inputs, state_outputs)),
        windows={name: WINDOW_TIME_AXIS for name in window_inputs},
        require_rank4=require_rank4, window_drop='either')
    return len(tuple(window_inputs))


__all__ = [
    'STATE_BINDING',
    'validate_inplace_state_contract',
    'validate_nctf_no_temporal_padding',
]
