#!/usr/bin/env python3
"""Export the learned DeepFilterNet2 heads to a fixed-frame ONNX graph.

The ONNX boundary is intentionally ``model.heads``.  STFT, ERB/complex feature
normalisation, mask expansion, deep-filter composition, post-filter and WOLA
remain in ``dfn2_process.c``.  ``--frames`` is fixed in the graph because the
current PyTorch model carries recurrent state internally; invoking this graph
on independent blocks resets that state.  A deployment must therefore use the
same block size and reset policy used for its checkpoint validation.
"""

import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from denoise import load_model


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat_erb, feat_spec):
        return self.model.heads(feat_erb, feat_spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, required=True,
                        help='fixed number of feature frames per invocation')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')

    model, params = load_model(SimpleNamespace(config=args.config, model=args.model))
    wrapper = Heads(model).eval()
    erb = torch.randn(1, 1, args.frames, model.n_erb)
    spec = torch.randn(1, 2, args.frames, model.df_bins)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        wrapper, (erb, spec), args.output,
        input_names=('feat_erb', 'feat_spec'),
        output_names=('erb_mask', 'df_coefs', 'df_alpha'),
        opset_version=args.opset, do_constant_folding=True,
    )

    import onnx
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.helper.set_model_props(graph, {
        'model_family': 'DeepFilterNet2',
        'checkpoint_sha256': file_sha256(args.model),
        'boundary': 'learned_heads_only',
        'sample_rate': str(params['SR']),
        'n_fft': str(params['N_FFT']),
        'win_len': str(params['WIN_LEN']),
        'hop_len': str(params['HOP_LEN']),
        'frames_per_invocation': str(args.frames),
        'recurrent_state': 'internal_reset_each_invocation',
        'c_prepost': 'dfn2_process.c/dfn2_process.h',
        'input_schema': 'feat_erb[1,1,T,n_erb];feat_spec[1,2,T,df_bins]',
        'output_schema': 'erb_mask[1,1,T,n_erb];df_coefs[1,T,df_bins,2*df_order];df_alpha[1,T,1]',
    })
    onnx.save(graph, args.output)

    manifest = os.path.splitext(args.output)[0] + '.json'
    with open(manifest, 'w', encoding='utf-8') as fp:
        json.dump({p.key: p.value for p in graph.metadata_props}, fp,
                  indent=2, sort_keys=True)
        fp.write('\n')

    if args.verify:
        import onnxruntime as ort
        with torch.no_grad():
            expected = wrapper(erb, spec)
        session = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        actual = session.run(None, {'feat_erb': erb.numpy(), 'feat_spec': spec.numpy()})
        worst = max(float(np.max(np.abs(a - b.detach().numpy())))
                    for a, b in zip(actual, expected))
        if worst > 2e-4:
            raise RuntimeError('ONNX parity failed: max abs error %.6g' % worst)
        print('ONNX parity max_abs=%.6g' % worst)
    print(args.output)


if __name__ == '__main__':
    main()
