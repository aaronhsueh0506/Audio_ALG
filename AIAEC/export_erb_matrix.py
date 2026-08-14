#!/usr/bin/env python3
"""Export checkpoint-exact ERB maps used by AIAEC candidates.

Only CAGCRN, GTCRN-AENR, and DeepFilterNet-AENR contain an ERB transform.
The other candidates operate directly on FFT bins and are rejected rather
than emitting a meaningless identity table.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AIAEC.export_onnx import file_sha256, load_checkpoint_model


SUPPORTED = ('CAGCRN', 'GTCRN_AENR', 'DeepFilterNet_AENR')


def extract_matrices(model_name, model):
    n_bins = model.grid.n_freqs
    if model_name == 'CAGCRN':
        low = model.erb.low_bins
        bands = model.erb.merge_matrix.shape[0]
        forward = np.zeros((n_bins, low + bands), dtype=np.float32)
        inverse = np.zeros((low + bands, n_bins), dtype=np.float32)
        forward[:low, :low] = np.eye(low, dtype=np.float32)
        inverse[:low, :low] = np.eye(low, dtype=np.float32)
        forward[low:, low:] = model.erb.merge_matrix.detach().cpu().numpy().T
        inverse[low:, low:] = model.erb.split_matrix.detach().cpu().numpy()
        detail = {'low_identity_bins': int(low), 'high_erb_bands': int(bands)}
    elif model_name == 'GTCRN_AENR':
        low = model.erb.erb_subband_1
        bands = model.erb.erb_fc.out_features
        forward = np.zeros((n_bins, low + bands), dtype=np.float32)
        inverse = np.zeros((low + bands, n_bins), dtype=np.float32)
        forward[:low, :low] = np.eye(low, dtype=np.float32)
        inverse[:low, :low] = np.eye(low, dtype=np.float32)
        forward[low:, low:] = model.erb.erb_fc.weight.detach().cpu().numpy().T
        inverse[low:, low:] = model.erb.ierb_fc.weight.detach().cpu().numpy().T
        detail = {'low_identity_bins': int(low), 'high_erb_bands': int(bands)}
    elif model_name == 'DeepFilterNet_AENR':
        forward = model.erb_fb.detach().cpu().numpy().T
        inverse = model.erb_inv.detach().cpu().numpy()
        detail = {'erb_bands': int(forward.shape[1])}
    else:
        raise ValueError('%s has no ERB transform; supported models: %s' %
                         (model_name, ', '.join(SUPPORTED)))
    forward = np.ascontiguousarray(forward, dtype='<f4')
    inverse = np.ascontiguousarray(inverse, dtype='<f4')
    if forward.shape[0] != n_bins or inverse.shape != (forward.shape[1], n_bins):
        raise RuntimeError('invalid ERB matrix shapes %s / %s' %
                           (forward.shape, inverse.shape))
    if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
        raise RuntimeError('ERB matrices contain NaN or Inf')
    return forward, inverse, detail


def _write_c_matrix(stream, name, matrix):
    stream.write('static const float %s[%d][%d] = {\n' %
                 (name, matrix.shape[0], matrix.shape[1]))
    for row in matrix:
        stream.write('    {%s},\n' %
                     ', '.join('%.9gf' % float(value) for value in row))
    stream.write('};\n\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_name', choices=SUPPORTED)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--format', choices=('bin', 'npy', 'c', 'all'),
                        default='bin')
    args = parser.parse_args()

    model, grid = load_checkpoint_model(args.model_name, args.checkpoint)
    forward, inverse, detail = extract_matrices(args.model_name, model)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.model_name.lower() + '_erb'
    metadata = {
        'schema': 'aiaec-erb-matrix-v1',
        'model_family': args.model_name,
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'sample_rate': int(grid.sr),
        'n_fft': int(grid.n_fft),
        'forward_shape': list(forward.shape),
        'inverse_shape': list(inverse.shape),
        'forward_layout': 'fft_bin_major[n_bins][n_features]',
        'inverse_layout': 'feature_major[n_features][n_bins]',
        **detail,
    }
    with open(os.path.join(args.output_dir, stem + '.json'), 'w',
              encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if args.format in ('bin', 'all'):
        with open(os.path.join(args.output_dir, stem + '.bin'), 'wb') as stream:
            stream.write(struct.pack('<8sIII', b'AIAECERB', 1,
                                     forward.shape[0], forward.shape[1]))
            stream.write(forward.tobytes())
            stream.write(inverse.tobytes())
    if args.format in ('npy', 'all'):
        np.save(os.path.join(args.output_dir, stem + '_forward.npy'), forward)
        np.save(os.path.join(args.output_dir, stem + '_inverse.npy'), inverse)
    if args.format in ('c', 'all'):
        guard = ('AIAEC_%s_ERB_GENERATED_H' % args.model_name).replace('-', '_')
        with open(os.path.join(args.output_dir, stem + '.h'), 'w',
                  encoding='ascii') as stream:
            stream.write('/* Generated by AIAEC/export_erb_matrix.py. */\n')
            stream.write('#ifndef %s\n#define %s\n\n' % (guard, guard))
            _write_c_matrix(stream, 'AIAEC_ERB_FORWARD', forward)
            _write_c_matrix(stream, 'AIAEC_ERB_INVERSE', inverse)
            stream.write('#endif\n')
    print('%s: forward=%s inverse=%s' %
          (args.model_name, forward.shape, inverse.shape))


if __name__ == '__main__':
    main()
