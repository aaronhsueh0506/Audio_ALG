#!/usr/bin/env python3
"""Build representative fixed-block ONNX calibration inputs for AIAEC.

``--primary-dir`` means microphone for the end-to-end models and materialized
linear-AEC error for RES+NR models.  ``--far-dir`` must have identical relative
WAV paths.  Inputs are resampled and transformed with the same project helpers
as inference; no random tensors or zero-only recurrent history are used.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.export_onnx import (
    MODEL_NAMES,
    alignment_depth,
    file_sha256,
    load_checkpoint_model,
    set_alignment_depth,
)
from AIAEC.dataset_gen import stft
from AIAEC.inference_common import load_linear_error_far, load_mic_far


LINEAR_ERROR_MODELS = {
    'Align_ULCNet', 'GTCRN_AENR', 'DeepFilterNet_AENR',
}


def discover_pairs(primary_dir, far_dir):
    def inventory(root):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError('directory does not exist: %s' % root)
        result = {path.relative_to(root).as_posix(): path
                  for path in sorted(root.rglob('*'))
                  if path.is_file() and path.suffix.lower() == '.wav'}
        if not result:
            raise ValueError('no WAV files under %s' % root)
        return result
    primary = inventory(primary_dir)
    far = inventory(far_dir)
    if set(primary) != set(far):
        raise ValueError('primary/far relative WAV sets differ: missing=%d extra=%d' %
                         (len(set(primary) - set(far)),
                          len(set(far) - set(primary))))
    return [(name, primary[name], far[name]) for name in sorted(primary)]


def _ri(spec):
    return torch.view_as_real(spec.transpose(-2, -1)).contiguous()


def _dfn_feature_config(path, grid):
    from AINR.DeepFilterNet2.train import read_feature_config
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(path)
    return read_feature_config(config, grid.sr, grid.hop_len)


def blocks_from_pair(model_name, model, grid, primary_path, far_path,
                     frames, feature_config=None):
    loader = (load_linear_error_far if model_name in LINEAR_ERROR_MODELS
              else load_mic_far)
    primary, far, _ = loader(str(primary_path), str(far_path), grid.sr)
    if model_name == 'DeepFilterNet_AENR':
        from AINR.DeepFilterNet2.train import extract_dfn2_features
        window = grid.window(device=primary.device, dtype=primary.dtype)
        primary_spec = torch.stft(
            primary, grid.n_fft, grid.hop_len, grid.win_len, window=window,
            normalized=True, return_complex=True)
        far_spec = torch.stft(
            far, grid.n_fft, grid.hop_len, grid.win_len, window=window,
            normalized=True, return_complex=True)
        _, pe, ps, _ = extract_dfn2_features(
            primary_spec, model.erb_fb, model.df_bins, feature_config, None)
        _, fe, fs, _ = extract_dfn2_features(
            far_spec, model.erb_fb, model.df_bins, feature_config, None)
        tensors = {'error_erb': pe, 'error_spec': ps,
                   'far_erb': fe, 'far_spec': fs}
        time_axis = {'error_erb': 2, 'error_spec': 2,
                     'far_erb': 2, 'far_spec': 2}
    else:
        tensors = {
            ('linear_error_ri' if model_name in LINEAR_ERROR_MODELS
             else 'microphone_ri'): _ri(stft(primary, grid)),
            'far_end_ri': _ri(stft(far, grid)),
        }
        time_axis = {name: 1 for name in tensors}
    total = tensors[next(iter(tensors))].shape[time_axis[next(iter(tensors))]]
    for start in range(0, total - frames + 1, frames):
        block = {}
        for name, tensor in tensors.items():
            axis = time_axis[name]
            slices = [slice(None)] * tensor.ndim
            slices[axis] = slice(start, start + frames)
            block[name] = tensor[tuple(slices)][0].cpu().numpy().astype(
                np.float32, copy=False)
        yield block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_name', choices=MODEL_NAMES)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--primary-dir', required=True)
    parser.add_argument('--far-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, required=True,
                        help='must match export_onnx.py --frames')
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--max-delay-frames', type=int, default=None,
                        help='must match export_onnx.py when D is overridden')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dfn-config',
                        default=os.path.join(
                            _SCRIPT_DIR, 'DeepFilterNet_AENR', 'config.ini'))
    args = parser.parse_args()
    if args.frames <= 0 or args.blocks <= 0:
        parser.error('--frames and --blocks must be positive')
    if not args.output.lower().endswith('.npz'):
        parser.error('--output must end in .npz')

    model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
    checkpoint_delay_depth = alignment_depth(model)
    if args.max_delay_frames is not None:
        try:
            set_alignment_depth(model, args.max_delay_frames)
        except ValueError as error:
            parser.error(str(error))
    delay_depth = alignment_depth(model)
    if delay_depth and args.frames < delay_depth:
        parser.error('--frames=%d is shorter than this checkpoint alignment '
                     'depth D=%d; it would not match export_onnx.py' %
                     (args.frames, delay_depth))
    pairs = discover_pairs(args.primary_dir, args.far_dir)
    random.Random(args.seed).shuffle(pairs)
    feature_config = (_dfn_feature_config(args.dfn_config, grid)
                      if args.model_name == 'DeepFilterNet_AENR' else None)
    captured = {}
    source_files = []
    with torch.no_grad():
        for relative, primary, far in pairs:
            used = False
            for block in blocks_from_pair(
                    args.model_name, model, grid, primary, far, args.frames,
                    feature_config):
                for name, value in block.items():
                    captured.setdefault(name, []).append(value)
                used = True
                if len(next(iter(captured.values()))) >= args.blocks:
                    break
            if used:
                source_files.append(relative)
            if captured and len(next(iter(captured.values()))) >= args.blocks:
                break
    if not captured:
        raise RuntimeError('no complete %d-frame blocks were found' % args.frames)
    arrays = {name: np.stack(values[:args.blocks], axis=0)
              for name, values in captured.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        'schema': 'aiaec-fixed-block-calibration-v1',
        'model_family': args.model_name,
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'sample_rate': int(grid.sr), 'n_fft': int(grid.n_fft),
        'win_len': int(grid.win_len), 'hop_len': int(grid.hop_len),
        'frames_per_block': args.frames,
        'max_delay_frames': delay_depth,
        'checkpoint_max_delay_frames': checkpoint_delay_depth,
        'blocks': int(next(iter(arrays.values())).shape[0]),
        'seed': args.seed, 'source_files': source_files,
        'inputs': {
            name: {'shape': list(value.shape),
                   'min': float(value.min()), 'max': float(value.max()),
                   'p001': float(np.percentile(value, 0.1)),
                   'p999': float(np.percentile(value, 99.9))}
            for name, value in arrays.items()
        },
    }
    with open(os.path.splitext(args.output)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('%s: %d blocks x %d frames' %
          (args.output, report['blocks'], args.frames))


if __name__ == '__main__':
    main()
