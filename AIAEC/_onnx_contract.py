"""The streaming-ONNX padding validator, as AIAEC's exporters import it.

A full checkout carries ``onnx_streaming_contract`` at the repository root. A
standalone AIAEC copy has ``AINR/`` beside it instead, and ``AINR`` ships its
own self-contained copy of the same module, so fall back to that one.
"""

import os
import sys

try:
    from onnx_streaming_contract import validate_nctf_no_temporal_padding
except ModuleNotFoundError as error:
    _AINR_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AINR')
    if error.name != 'onnx_streaming_contract' or not os.path.isdir(_AINR_ROOT):
        raise
    # Appended, not inserted: only this one module is wanted from AINR, and the
    # names AIAEC already resolves must keep resolving where they did.
    sys.path.append(_AINR_ROOT)
    from onnx_streaming_contract import validate_nctf_no_temporal_padding

__all__ = ['validate_nctf_no_temporal_padding']
