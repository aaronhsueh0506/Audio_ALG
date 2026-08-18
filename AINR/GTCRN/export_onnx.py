#!/usr/bin/env python3
"""Export GTCRN as a one-frame explicit-state streaming ONNX graph.

Run from ``AINR/GTCRN``::

    python3 export_onnx.py --model output/gtcrn_best.pth \
        --output output/gtcrn_stream.onnx --verify

Generate deployment calibration beside this model with::

    python3 inference.py calib --model output/gtcrn_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/gtcrn
"""

import argparse
import configparser
import hashlib
import json
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative imports FIRST: when this module is loaded package-qualified
# (``import AINR.GTCRN.export_onnx`` from a quantization script) the flat
# names must not be touched at all -- another model directory may already
# own ``model``/``train`` in sys.modules. The flat branch serves direct
# ``python export_onnx.py`` execution only.
try:
    from .checkpoint_utils import extract_state_dict
    from .model import GTCRN
    from .train import build_contract, require_checkpoint_contract
    from .stream_model import StreamGTCRN, initial_inputs, stream_features
except ImportError:
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    from checkpoint_utils import extract_state_dict
    from model import GTCRN
    from train import build_contract, require_checkpoint_contract
    from stream_model import StreamGTCRN, initial_inputs, stream_features



# Kept numerically equal to GTCRN_MODEL_LAYOUT_VERSION in gtcrn_process.h,
# which declares the caller-owned cache struct this graph consumes and emits.
# tests/test_gtcrn_export_contract.py pins the two together.
STATE_LAYOUT_VERSION = 4

# One h_* tensor per GRU so each state slot names itself; only the temporal
# conv history stays a combined cache. Encoder TRA GRUs first, then decoder,
# then the two DPGRNN inter GRUs (whose hidden batches the frequency lanes).
INPUT_NAMES = (
    'input', 'conv_cache',
    'h_tra_enc0', 'h_tra_enc1', 'h_tra_enc2',
    'h_tra_dec0', 'h_tra_dec1', 'h_tra_dec2',
    'h_dpgrnn1', 'h_dpgrnn2',
)

OUTPUT_NAMES = ('output',) + tuple(
    name + '_out' for name in INPUT_NAMES[1:]
)


def _schema(names, tensors):
    return {
        name: [int(size) for size in tensor.shape]
        for name, tensor in zip(names, tensors)
    }


def set_onnx_metadata(graph, metadata):
    """Write a Python metadata mapping into an ONNX graph's model props.

    ONNX model properties are string-valued, so structured entries are JSON
    encoded (containers and bools) and scalars use ``str``; a consumer must
    see one rule everywhere.
    """
    import onnx
    onnx.helper.set_model_props(graph, {
        key: (json.dumps(value, sort_keys=True)
              if isinstance(value, (dict, list, bool)) else str(value))
        for key, value in metadata.items()
    })


def optimize_graph_file(path):
    """onnxoptimizer cleanup: drop the tracer's Identity/Constant/dead-end
    noise so the deployed graph carries only real ops. Skipped when the
    package is absent -- the graph is then correct but unoptimized."""
    try:
        import onnxoptimizer
        from onnxoptimizer import (
            get_available_passes,
            get_fuse_and_elimination_passes,
        )
    except ImportError:
        print('[skip] onnxoptimizer not installed; graph left unoptimized')
        return
    import onnx
    wanted = {
        'eliminate_nop_pad', 'eliminate_nop_transpose', 'eliminate_identity',
        'eliminate_deadend', 'eliminate_unused_initializer',
        'fuse_consecutive_transposes', 'fuse_consecutive_squeezes',
        'fuse_consecutive_unsqueezes', 'fuse_matmul_add_bias_into_gemm',
        'fuse_add_bias_into_conv',
    }
    passes = list(set(get_fuse_and_elimination_passes())
                  | (wanted & set(get_available_passes())))
    graph = onnx.load(path)
    before = len(graph.graph.node)
    graph = onnxoptimizer.optimize(graph, passes, fixed_point=True)
    onnx.save(graph, path)
    print('[onnxoptimizer] %d -> %d nodes' % (before, len(graph.graph.node)))


def _pin_static_output_shapes(graph, output_names, outputs):
    """Declare every graph output with its concrete traced shape.

    Shape inference cannot prove some extents static after GRU/Slice/Concat
    and leaves symbolic dim_params, but with every input static the true
    output shapes ARE static -- they are exactly the reference forward's
    tensor shapes. Accelerator toolchains require static I/O declarations,
    so the symbolic dims are overwritten with the measured ones.
    """
    shapes = {name: tuple(int(size) for size in tensor.shape)
              for name, tensor in zip(output_names, outputs)}
    for value_info in graph.graph.output:
        tensor_shape = value_info.type.tensor_type.shape
        tensor_shape.ClearField('dim')
        for size in shapes[value_info.name]:
            tensor_shape.dim.add().dim_value = size


def _resolve_internal_shapes(model):
    """Make every internal value_info static, or drop the annotation.

    torch's shape inference labels edges it cannot prove with symbolic
    unk__N dims even though static inputs fix every extent. onnxruntime's
    symbolic shape inference folds those to integers; any annotation it
    still cannot prove is removed -- a missing annotation is re-inferred by
    the consumer, a symbolic one is read as dynamic.
    """
    try:
        from onnxruntime.tools.symbolic_shape_infer import (
            SymbolicShapeInference,
        )
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    except Exception as error:
        print('[skip] symbolic shape inference unavailable: %s' % error)
    kept = [
        value_info for value_info in model.graph.value_info
        if all(dim.HasField('dim_value')
               for dim in value_info.type.tensor_type.shape.dim)
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    return model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_stream_model(config_path, checkpoint_path):
    cfg = configparser.ConfigParser()
    if not cfg.read(config_path):
        raise FileNotFoundError(config_path)
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    sub1 = cfg.getint('model', 'erb_subband_1')
    sub2 = cfg.getint('model', 'erb_subband_2')
    # No grid is pinned here: every graph and cache extent downstream derives
    # from the model built off this config (see stream_model.initial_inputs),
    # so the config must simply match the checkpoint's training grid. A
    # mismatch fails loudly -- shape-changing drift at strict load_state_dict,
    # shape-preserving drift at the recorded checkpoint contract.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require_checkpoint_contract(checkpoint, build_contract(cfg, win_len, hop_len),
                                context=checkpoint_path, allow_missing=True)
    offline = GTCRN(sub1, sub2, nfft=n_fft, fs=sr).eval()
    offline.load_state_dict(extract_state_dict(checkpoint, checkpoint_path), strict=True)

    stream = StreamGTCRN(offline).eval()
    return stream, {'sr': sr, 'n_fft': n_fft, 'win_len': win_len,
                    'hop_len': hop_len, 'sub1': sub1, 'sub2': sub2}


def build_metadata(checkpoint_path, grid, inputs, outputs):
    """The exported graph's metadata.

    Separate from ``main`` so the contract test can compare
    ``state_layout_version`` against gtcrn_process.h without an ONNX export.
    """
    return {
        'model_family': 'GTCRN',
        'boundary': 'stateless_streaming_explicit_state',
        'state_layout_version': STATE_LAYOUT_VERSION,
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'sample_rate': grid['sr'], 'n_fft': grid['n_fft'],
        'win_len': grid['win_len'], 'hop_len': grid['hop_len'],
        'erb_boundary': 'inside_graph',
        'c_prepost': 'gtcrn_process.c/gtcrn_process.h',
        'input_feature_frames': 1,
        'output_frames_per_invocation': 1,
        'accelerator_persistent_state': False,
        'recurrent_state': 'conv_cache_plus_per_gru_h_explicit_input_output',
        'state_handoff': dict(zip(INPUT_NAMES[1:], OUTPUT_NAMES[1:])),
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
    }


def export_graph(stream, grid, checkpoint_path, output_path, opset=17,
                 verify=False):
    """Write the ONNX graph plus its metadata JSON; optionally verify parity.

    Shared by the export CLI and ``inference.py calib``, so the calibration
    tensors and the graph they bind to always come from the same model
    instance in the same process.
    """
    inputs = initial_inputs(stream.model)
    # Cloned so the schema comes from the real graph tensors without the
    # traced export seeing anything this forward pass touched.
    with torch.no_grad():
        outputs = stream(*(tensor.clone() for tensor in inputs))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        stream, inputs, output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset, do_constant_folding=True,
    )
    optimize_graph_file(output_path)
    import onnx
    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    metadata = build_metadata(checkpoint_path, grid, inputs, outputs)
    set_onnx_metadata(graph, metadata)
    _pin_static_output_shapes(graph, OUTPUT_NAMES, outputs)
    graph = _resolve_internal_shapes(graph)
    onnx.save(graph, output_path)
    with open(os.path.splitext(output_path)[0] + '.json', 'w',
              encoding='utf-8') as fp:
        json.dump({p.key: p.value for p in graph.metadata_props}, fp,
                  indent=2, sort_keys=True)
        fp.write('\n')
    if verify:
        import onnxruntime as ort
        # Verify in the domain the graph actually sees: a random RI spectrum
        # run through the SAME host feature rule (positive magnitude,
        # consistent channels), not raw noise in every channel.
        generator = torch.Generator().manual_seed(20260818)
        verify_inputs = list(initial_inputs(stream.model))
        spectrum_ri = torch.randn(
            verify_inputs[0].shape[:-1] + (2,), generator=generator
        )
        verify_inputs[0] = stream_features(spectrum_ri)
        ort_feed = {name: tensor.detach().numpy().copy()
                    for name, tensor in zip(INPUT_NAMES, verify_inputs)}
        with torch.no_grad():
            expected = stream(*(tensor.clone() for tensor in verify_inputs))
        session = ort.InferenceSession(output_path,
                                       providers=['CPUExecutionProvider'])
        actual = session.run(None, {item.name: ort_feed[item.name]
                                    for item in session.get_inputs()})
        worst = max(float(np.max(np.abs(a - b.detach().numpy())))
                    for a, b in zip(actual, expected))
        # fp32 GRU kernels differ in accumulation order between torch and
        # onnxruntime, and the difference scales with the trained weight
        # magnitude: a real checkpoint measured 2.14e-4 where random-weight
        # fixtures sit at 1e-6. The bound guards against structural breakage
        # (wrong wiring explodes far past this), not kernel rounding.
        if worst > 1e-3:
            raise RuntimeError('ONNX parity failed: max abs error %.6g' % worst)
        print('ONNX parity max_abs=%.6g' % worst)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    model, grid = build_stream_model(args.config, args.model)
    export_graph(model, grid, args.model, args.output,
                 opset=args.opset, verify=args.verify)
    print(args.output)


if __name__ == '__main__':
    main()
