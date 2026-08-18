#!/usr/bin/env python3
"""Export GTCRN's exact ERB compression/expansion matrices.

Since model layout v5 the ERB maps run on the HOST: gtcrn_process.c takes
POINTERS to the matrices, and ``--runtime-bins`` writes the raw float32
``erb_fwd.bin``/``erb_inv.bin`` (plus a json manifest) in exactly the layouts
the C loops consume -- the deployment loader owns the files and can swap
them at any time without recompiling. The bin/npy formats remain for
toolchain validation.

    python3 export_erb_matrix.py --runtime-bins output/erb
"""

import argparse
import configparser
import json
import os
import struct

import numpy as np

try:
    from .export_onnx import build_stream_model, file_sha256
except ImportError:  # direct ``python export_erb_matrix.py`` execution
    from export_onnx import build_stream_model, file_sha256


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_matrix(fp, name, matrix):
    rows, cols = matrix.shape
    fp.write('static const float %s[%d][%d] = {\n' % (name, rows, cols))
    for row in matrix:
        fp.write('    {%s},\n' % ', '.join('%.9ef' % float(v) for v in row))
    fp.write('};\n\n')


def write_runtime_bins(config_path, out_dir):
    """Emit the runtime .bin matrices the C host loads.

    Raw float32 little-endian in exactly the layouts gtcrn_process.c
    consumes: erb_fwd.bin bin-major [high_bins][high_bands], erb_inv.bin
    band-major [high_bands][high_bins], plus erb_matrices.json describing
    both. The library takes pointers only, so the deployment loader owns the
    files and can swap them at any time.
    """
    import configparser as _configparser

    import torch

    try:
        from .model import GTCRN
    except ImportError:
        from model import GTCRN

    cfg = _configparser.ConfigParser()
    if not cfg.read(config_path):
        raise FileNotFoundError(config_path)
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    sub1 = cfg.getint('model', 'erb_subband_1')
    sub2 = cfg.getint('model', 'erb_subband_2')
    with torch.no_grad():
        erb = GTCRN(sub1, sub2, nfft=n_fft, fs=sr).erb
        forward = erb.erb_fc.weight.detach().numpy().T.astype('<f4')
        inverse = erb.ierb_fc.weight.detach().numpy().T.astype('<f4')
    os.makedirs(out_dir, exist_ok=True)
    forward.tofile(os.path.join(out_dir, 'erb_fwd.bin'))
    inverse.tofile(os.path.join(out_dir, 'erb_inv.bin'))
    manifest = {
        'dtype': 'float32', 'byte_order': 'little',
        'erb_fwd.bin': {'shape': list(forward.shape),
                        'layout': 'bin_major [high_bins][high_bands]'},
        'erb_inv.bin': {'shape': list(inverse.shape),
                        'layout': 'band_major [high_bands][high_bins]'},
        'grid': {'sr': sr, 'n_fft': n_fft,
                 'erb_subband_1': sub1, 'erb_subband_2': sub2},
    }
    with open(os.path.join(out_dir, 'erb_matrices.json'), 'w',
              encoding='utf-8') as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)
        fp.write('\n')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', default=None,
                        help='checkpoint whose frozen ERB buffers are '
                             'exported (unused by --c-tables)')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--format', choices=('bin', 'npy', 'c', 'all'), default='all')
    parser.add_argument('--runtime-bins', default=None,
                        help='write erb_fwd.bin/erb_inv.bin (+ json) in the '
                             'exact layouts gtcrn_process.c consumes; no '
                             'checkpoint needed')
    args = parser.parse_args()
    if args.runtime_bins:
        print(write_runtime_bins(args.config, args.runtime_bins))
        return
    if not args.model:
        parser.error('--model is required unless --runtime-bins is used')
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)

    stream_model, grid = build_stream_model(args.config, args.model)
    model = stream_model.model
    sr = grid['sr']
    n_fft = grid['n_fft']
    sub1 = grid['sub1']
    sub2 = grid['sub2']
    high_forward = model.erb.erb_fc.weight.detach().numpy().astype('<f4', copy=False)
    high_inverse = model.erb.ierb_fc.weight.detach().numpy().astype('<f4', copy=False)
    n_bins = n_fft // 2 + 1
    n_erb = sub1 + sub2
    forward = np.zeros((n_bins, n_erb), dtype='<f4')
    inverse = np.zeros((n_erb, n_bins), dtype='<f4')
    forward[:sub1, :sub1] = np.eye(sub1, dtype=np.float32)
    forward[sub1:, sub1:] = high_forward.T
    inverse[:sub1, :sub1] = np.eye(sub1, dtype=np.float32)
    inverse[sub1:, sub1:] = high_inverse.T

    output_dir = args.output_dir or cfg.get('paths', 'output_dir', fallback='output')
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        'schema': 'gtcrn-erb-matrix-v1', 'sample_rate': sr, 'n_fft': n_fft,
        'checkpoint_sha256': file_sha256(args.model),
        'n_bins': n_bins, 'n_erb_features': n_erb,
        'low_identity_bins': sub1, 'high_erb_bands': sub2,
        'graph_boundary': 'inside_onnx',
        'warning': 'Do not apply these matrices outside the current ONNX graph.',
        'forward_layout': 'fft_bin_major[n_bins][n_erb_features]',
        'inverse_layout': 'erb_feature_major[n_erb_features][n_bins]',
    }
    with open(os.path.join(output_dir, 'erb_matrix.json'), 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
        fp.write('\n')
    if args.format in ('bin', 'all'):
        with open(os.path.join(output_dir, 'erb_matrix.bin'), 'wb') as fp:
            fp.write(struct.pack('<iiii', n_bins, n_erb, sub1, sub2))
            fp.write(np.ascontiguousarray(forward).tobytes())
            fp.write(np.ascontiguousarray(inverse).tobytes())
    if args.format in ('npy', 'all'):
        np.save(os.path.join(output_dir, 'erb_forward.npy'), forward)
        np.save(os.path.join(output_dir, 'erb_inverse.npy'), inverse)
    if args.format in ('c', 'all'):
        path = os.path.join(output_dir, 'erb_matrix.h')
        with open(path, 'w', encoding='ascii') as fp:
            fp.write('/* Generated by GTCRN/export_erb_matrix.py; graph already embeds this map. */\n')
            fp.write('#ifndef GTCRN_ERB_MATRIX_GENERATED_H\n#define GTCRN_ERB_MATRIX_GENERATED_H\n\n')
            fp.write('#define GTCRN_ERB_MATRIX_BINS %d\n' % n_bins)
            fp.write('#define GTCRN_ERB_MATRIX_FEATURES %d\n\n' % n_erb)
            _write_matrix(fp, 'GTCRN_ERB_FORWARD', forward)
            _write_matrix(fp, 'GTCRN_ERB_INVERSE', inverse)
            fp.write('#endif\n')
        print(path)
    print('forward=%s inverse=%s (already embedded in ONNX)' %
          (forward.shape, inverse.shape))


if __name__ == '__main__':
    main()
