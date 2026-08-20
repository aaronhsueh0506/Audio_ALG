#!/usr/bin/env python3
"""Export the exact DeepFilterNet2 ERB analysis and mask-expansion matrices.

The matrices are registered buffers of :class:`DeepFilterNet2`, so this tool
constructs the model through the same config reader as training instead of
reimplementing the filterbank.  The C runtime uses ``forward`` to build ERB
features and ``inverse`` to expand network ERB masks to FFT-bin gains.
"""

import argparse
import configparser
import hashlib
import json
import os
import struct

import numpy as np
import torch

from model import DeepFilterNet2
from train import read_model_config


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_matrix(fp, name, matrix):
    rows, cols = matrix.shape
    fp.write("static const float %s[%d][%d] = {\n" % (name, rows, cols))
    for row in matrix:
        fp.write("    {%s},\n" % ", ".join("%.9ef" % float(v) for v in row))
    fp.write("};\n\n")


def write_runtime_bins(config_path, out_dir):
    """Emit the runtime .bin matrices the C host loads.

    Raw float32 little-endian in exactly the layouts df_common consumes:
    erb_fwd.bin bin-major [n_bins][n_erb], erb_inv.bin band-major
    [n_erb][n_bins], plus erb_matrices.json. The library takes pointers only
    (dfn2_set_erb_matrices), so the deployment loader owns the files and can
    swap them at any time. The contract test pins the model's own buffers
    against these files.
    """
    import configparser as _configparser

    import torch

    try:
        from .model import DeepFilterNet2
        from .train import read_model_config
    except ImportError:
        from model import DeepFilterNet2
        from train import read_model_config

    cfg = _configparser.ConfigParser()
    if not cfg.read(config_path):
        raise FileNotFoundError(config_path)
    with torch.no_grad():
        model = DeepFilterNet2(**read_model_config(cfg))
        forward = model.erb_fb.detach().numpy().astype('<f4')
        inverse = model.erb_inv.detach().numpy().astype('<f4')
    if forward.shape[0] < forward.shape[1]:
        forward = np.ascontiguousarray(forward.T)
    if inverse.shape[0] > inverse.shape[1]:
        inverse = np.ascontiguousarray(inverse.T)
    os.makedirs(out_dir, exist_ok=True)
    forward.tofile(os.path.join(out_dir, 'erb_fwd.bin'))
    inverse.tofile(os.path.join(out_dir, 'erb_inv.bin'))
    manifest = {
        'dtype': 'float32', 'byte_order': 'little',
        'erb_fwd.bin': {'shape': list(forward.shape),
                        'layout': 'bin_major [n_bins][n_erb]'},
        'erb_inv.bin': {'shape': list(inverse.shape),
                        'layout': 'band_major [n_erb][n_bins]'},
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
                        help='optional checkpoint; validates that exported buffers match it')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--format', choices=('bin', 'npy', 'c', 'all'), default='all')
    parser.add_argument('--runtime-bins', default=None,
                        help='write erb_fwd.bin/erb_inv.bin (+ json) in the '
                             'exact layouts df_common consumes; no '
                             'checkpoint needed')
    args = parser.parse_args()
    if args.runtime_bins:
        print(write_runtime_bins(args.config, args.runtime_bins))
        return

    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)
    model = DeepFilterNet2(**read_model_config(cfg)).eval()
    if args.model:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    # model buffers are (bands, bins); C tables are row-major (bins, bands).
    forward = model.erb_fb.detach().cpu().numpy().T.astype('<f4', copy=False)
    inverse = model.erb_inv.detach().cpu().numpy().T.astype('<f4', copy=False)
    n_bins, n_bands = forward.shape
    output_dir = args.output_dir or cfg.get('paths', 'output_dir', fallback='output')
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'schema': 'dfn2-erb-matrix-v1',
        'sample_rate': cfg.getint('signal', 'sr'),
        'n_fft': cfg.getint('signal', 'n_fft'),
        'n_bins': n_bins,
        'n_bands': n_bands,
        'forward_layout': 'fft_bin_major[n_bins][n_bands]',
        'inverse_layout': 'fft_bin_major[n_bins][n_bands]',
        'forward_use': 'erb_power[band] = sum(power[bin] * forward[bin][band])',
        'inverse_use': 'bin_gain[bin] = sum(inverse[bin][band] * mask[band])',
    }
    if args.model:
        digest = hashlib.sha256()
        with open(args.model, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        metadata['checkpoint_sha256'] = digest.hexdigest()
    with open(os.path.join(output_dir, 'erb_matrix.json'), 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2, sort_keys=True)
        fp.write('\n')

    if args.format in ('bin', 'all'):
        path = os.path.join(output_dir, 'erb_matrix.bin')
        with open(path, 'wb') as fp:
            fp.write(struct.pack('<ii', n_bins, n_bands))
            fp.write(np.ascontiguousarray(forward).tobytes())
            fp.write(np.ascontiguousarray(inverse).tobytes())
        print(path)
    if args.format in ('npy', 'all'):
        np.save(os.path.join(output_dir, 'erb_forward.npy'), forward)
        np.save(os.path.join(output_dir, 'erb_inverse.npy'), inverse)
    if args.format in ('c', 'all'):
        path = os.path.join(output_dir, 'erb_matrix.h')
        with open(path, 'w', encoding='ascii') as fp:
            fp.write("/* Generated by DeepFilterNet2/export_erb_matrix.py. */\n")
            fp.write("#ifndef DFN2_ERB_MATRIX_GENERATED_H\n")
            fp.write("#define DFN2_ERB_MATRIX_GENERATED_H\n\n")
            fp.write("#define DFN2_ERB_MATRIX_BINS %d\n" % n_bins)
            fp.write("#define DFN2_ERB_MATRIX_BANDS %d\n\n" % n_bands)
            _write_matrix(fp, 'DFN2_ERB_FORWARD', forward)
            _write_matrix(fp, 'DFN2_ERB_INVERSE', inverse)
            fp.write("#endif\n")
        print(path)

    unity_error = float(np.max(np.abs(inverse.sum(axis=1) - 1.0)))
    if unity_error > 1e-6:
        raise RuntimeError('inverse ERB matrix is not a partition of unity: %.3g' % unity_error)
    print('shape=%s inverse_unity_max_error=%.3g' % (forward.shape, unity_error))


if __name__ == '__main__':
    main()
