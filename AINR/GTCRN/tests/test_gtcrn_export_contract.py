"""The GTCRN export metadata and gtcrn_process.h are one state contract."""

import os
import pathlib
import re
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PATH = pathlib.Path(ROOT)
sys.path.insert(0, ROOT)

# Each model project has its own top-level ``train.py``/``model.py``/
# ``export_onnx.py``.  Under a single pytest session the first one imported
# wins ``sys.modules``, so dropping the cached entries is what makes this file
# exercise GTCRN's code rather than a sibling project's.
for _stale in ('train', 'denoise', 'model', 'checkpoint_utils', 'export_onnx'):
    sys.modules.pop(_stale, None)


from model import GTCRN  # noqa: E402
from stream_model import StreamGTCRN, initial_inputs  # noqa: E402
from export_onnx import (  # noqa: E402
    INPUT_NAMES,
    OUTPUT_NAMES,
    STATE_LAYOUT_VERSION,
    build_metadata,
)


GRID = {'sr': 16000, 'n_fft': 512, 'win_len': 512, 'hop_len': 256}


def c_macro(name):
    header = (ROOT_PATH / 'gtcrn_process.h').read_text(encoding='utf-8')
    match = re.search(
        r'^#define\s+%s\s+(\d+)u?\s*(?:/\*.*)?$' % re.escape(name),
        header,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return int(match.group(1))


def _metadata(tmp_path):
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'not a real checkpoint, only hashed')
    torch.manual_seed(43)
    stream = StreamGTCRN(GTCRN(65, 64, nfft=512, fs=16000).eval()).eval()
    inputs = initial_inputs()
    with torch.no_grad():
        outputs = stream(*(tensor.clone() for tensor in inputs))
    return build_metadata(str(checkpoint), GRID, inputs, outputs)


def test_state_layout_version_is_pinned_to_the_c_header(tmp_path):
    """A board reads this out of the graph to decide whether its
    ``GTCRNModelState`` still matches. Asserting the Python constant alone
    would not catch the metadata key being dropped, so this goes through the
    same builder ``main`` uses.
    """
    metadata = _metadata(tmp_path)
    assert metadata['state_layout_version'] == c_macro(
        'GTCRN_MODEL_LAYOUT_VERSION'
    )
    assert STATE_LAYOUT_VERSION == metadata['state_layout_version']


def test_input_schema_shapes_match_the_c_cache_struct(tmp_path):
    """The schema must come from the real tensors, not a typed-in string.

    gtcrn_process.h sizes its three caches from these extents. While the
    schema was a hand-written literal, a stream-model cache-shape change would
    have left the metadata describing the old graph and nothing would have
    disagreed.
    """
    metadata = _metadata(tmp_path)
    schema = metadata['input_schema']
    assert set(schema) == set(INPUT_NAMES)
    assert schema['conv_cache'] == [
        c_macro('GTCRN_MODEL_CONV_SIDES'), 1,
        c_macro('GTCRN_MODEL_CONV_CHANNELS'),
        c_macro('GTCRN_MODEL_CONV_TIME'),
        c_macro('GTCRN_MODEL_CONV_FREQ'),
    ]
    assert schema['tra_cache'] == [
        c_macro('GTCRN_MODEL_TRA_SIDES'), c_macro('GTCRN_MODEL_TRA_BLOCKS'),
        1, 1, c_macro('GTCRN_MODEL_TRA_HIDDEN'),
    ]
    assert schema['inter_cache'] == [
        c_macro('GTCRN_MODEL_INTER_LAYERS'), 1,
        c_macro('GTCRN_MODEL_INTER_FREQ'),
        c_macro('GTCRN_MODEL_INTER_HIDDEN'),
    ]

    # Every state output must hand back exactly the shape its input slot
    # expects, or the caller cannot copy one into the other.
    output_schema = metadata['output_schema']
    assert set(output_schema) == set(OUTPUT_NAMES)
    for state_in, state_out in metadata['state_handoff'].items():
        assert output_schema[state_out] == schema[state_in]
