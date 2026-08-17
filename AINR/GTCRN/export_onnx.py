#!/usr/bin/env python3
"""Export GTCRN as a one-frame explicit-state streaming ONNX graph."""

import argparse
import configparser
import hashlib
import json
import os
import sys

import numpy as np
import torch

from checkpoint_utils import extract_state_dict
from model import GTCRN
from train import build_contract, require_checkpoint_contract
from stream_model import StreamGTCRN, initial_inputs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

# Package-qualified so it cannot collide with this project's own flat
# ``model``/``train`` modules. The schema encoder is one contract across every
# stateless exporter in the repo; a hand-typed shape string is a second copy
# that drifts the moment a cache extent changes.
from AINR.DeepFilterNet2.export_onnx import (  # noqa: E402
    _schema,
    set_onnx_metadata,
)


# Kept numerically equal to GTCRN_MODEL_LAYOUT_VERSION in gtcrn_process.h,
# which declares the caller-owned cache struct this graph consumes and emits.
# tests/test_gtcrn_export_contract.py pins the two together.
STATE_LAYOUT_VERSION = 1

INPUT_NAMES = ('mix', 'conv_cache', 'tra_cache', 'inter_cache')

OUTPUT_NAMES = (
    'enhanced', 'conv_cache_out', 'tra_cache_out', 'inter_cache_out',
)


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
    if (sr, n_fft, sub1, sub2) != (16000, 512, 65, 64):
        raise ValueError('the verified stream graph requires sr/n_fft/ERB=16000/512/65/64')
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
        'recurrent_state': 'conv_tra_inter_caches_explicit_input_output',
        'state_handoff': dict(zip(INPUT_NAMES[1:], OUTPUT_NAMES[1:])),
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
    }


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
    inputs = initial_inputs()
    # Cloned so the schema comes from the real graph tensors without the
    # traced export seeing anything this forward pass touched.
    with torch.no_grad():
        outputs = model(*(tensor.clone() for tensor in inputs))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        model, inputs, args.output,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=args.opset, do_constant_folding=True,
    )
    import onnx
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    set_onnx_metadata(graph, build_metadata(args.model, grid, inputs, outputs))
    onnx.save(graph, args.output)
    with open(os.path.splitext(args.output)[0] + '.json', 'w', encoding='utf-8') as fp:
        json.dump({p.key: p.value for p in graph.metadata_props}, fp,
                  indent=2, sort_keys=True)
        fp.write('\n')
    if args.verify:
        import onnxruntime as ort
        verify_inputs = initial_inputs()
        ort_feed = {name: tensor.detach().numpy().copy()
                    for name, tensor in zip(INPUT_NAMES, verify_inputs)}
        with torch.no_grad():
            expected = model(*(tensor.clone() for tensor in verify_inputs))
        session = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        actual = session.run(None, {item.name: ort_feed[item.name]
                                    for item in session.get_inputs()})
        worst = max(float(np.max(np.abs(a - b.detach().numpy())))
                    for a, b in zip(actual, expected))
        if worst > 2e-4:
            raise RuntimeError('ONNX parity failed: max abs error %.6g' % worst)
        print('ONNX parity max_abs=%.6g' % worst)
    print(args.output)


if __name__ == '__main__':
    main()
