#!/usr/bin/env python3
"""Export DFN2 heads as a stateless streaming ONNX graph.

The accelerator owns no persistent state. Each invocation receives the three
feature frames required by the input kernel, all three GRU hidden states, and
the four-frame causal history used by ``df_convp``. It emits the heads for the
centre feature frame together with every updated state tensor.

STFT, feature normalisation, the three-frame feature window, deep-filter
composition and WOLA remain in the host C code.

Run from ``AINR/DeepFilterNet2``::

    python3 export_onnx.py --model output/dfn2_best.pth \
        --output output/dfn2_stream.onnx --verify

Generate deployment calibration beside this model with::

    python3 inference.py calib --model output/dfn2_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/dfn2
"""

import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .inference import load_model
except ImportError:  # direct ``python export_onnx.py`` execution
    from inference import load_model


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FRAMES = 3

# Kept numerically equal to DFN2_MODEL_IO_LAYOUT_VERSION in dfn2_model_io.h,
# which declares the caller-owned state struct this graph consumes and emits.
# tests/test_dfn2_contract.py pins the two together. Tensor names are part of
# this contract: runtimes bind by name.
STATE_LAYOUT_VERSION = 2

# Content inputs carry content names (two feature streams); each GRU hidden
# is its own h_* tensor; the deep-filter pathway history is a combined cache.
INPUT_NAMES = (
    'erb',
    'spec',
    'h_encoder',
    'h_erb',
    'h_df',
    'df_convp_history',
)

OUTPUT_NAMES = (
    'erb_mask',
    'df_coefs',
    'df_alpha',
    'h_encoder_out',
    'h_erb_out',
    'h_df_out',
    'df_convp_history_out',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _lookahead_centre(module, window):
    """Run a LookaheadConv2d on [t-1,t,t+1] and emit only frame t."""
    if window.shape[2] != INPUT_FRAMES:
        raise ValueError('DFN2 expects exactly three input feature frames')
    layers = list(module.children())
    conv = layers[1]
    if conv.kernel_size[0] != INPUT_FRAMES:
        raise ValueError('unsupported DFN2 input temporal kernel')
    pad = layers[0]
    if not isinstance(pad, nn.ConstantPad2d):
        raise ValueError('unsupported DFN2 lookahead padding module')
    # The host supplied the complete temporal receptive field, so only the
    # frequency part of LookaheadConv2d's original padding remains here.
    output = F.pad(window, (pad.padding[0], pad.padding[1], 0, 0))
    for layer in layers[1:]:
        output = layer(output)
    if output.shape[2] != 1:
        raise RuntimeError('DFN2 centre convolution did not emit one frame')
    return output


def _squeezed_gru(module, value, hidden):
    raw = value
    value = module.linear_in(value)
    value, hidden_next = module.gru(value, hidden)
    value = module.linear_out(value)
    if module.gru_skip is not None:
        value = value + module.gru_skip(raw)
    return value, hidden_next


def _df_pathway_step(module, current, history):
    """Run the causal kernel-5 DF residual and return its next history."""
    kernel = module.conv.kernel_size[0]
    if history.shape[2] != kernel - 1:
        raise ValueError('df_convp_history has the wrong temporal depth')
    combined = torch.cat((history, current), dim=2)
    output = module.conv(combined)
    output = module.pw(output)
    output = module.bn(output)
    output = module.act(output)
    if output.shape[2] != 1:
        raise RuntimeError('DF pathway did not emit one frame')
    return output, combined[:, :, 1:]


def feature_windows(feature):
    """Yield every ``[t-1,t,t+1]`` window of a ``(B,C,T,F)`` feature stream.

    The stream is zero-padded on both ends, so window ``t`` is exactly what
    the host pushes into ``dfn2_model_io_push_features`` for frame ``t`` and
    the graph turns into ``heads[t]``.  Every replay site -- calibration
    capture and the contract tests -- must slide the window the same way, or
    a test can agree with itself while disagreeing with the exporter.
    """
    zero = torch.zeros_like(feature[:, :, :1])
    padded = torch.cat((zero, feature, zero), dim=2)
    for index in range(feature.shape[2]):
        yield padded[:, :, index:index + INPUT_FRAMES]


class StatelessDFN2Heads(nn.Module):
    """Functional one-output-frame twin of ``DeepFilterNet2.heads``."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        if model.mask_lookahead != 1:
            raise ValueError(
                'this three-frame export requires mask_lookahead=1'
            )
        if model.encoder.erb_conv0[1].kernel_size[0] != INPUT_FRAMES:
            raise ValueError('this exporter requires a temporal kernel of 3')

    def forward(
        self,
        feat_erb_window,
        feat_spec_window,
        encoder_gru_hidden,
        erb_gru_hidden,
        df_gru_hidden,
        df_convp_history,
    ):
        model = self.model
        enc = model.encoder

        e0 = _lookahead_centre(enc.erb_conv0, feat_erb_window)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = _lookahead_centre(enc.df_conv0, feat_spec_window)
        c1 = enc.df_conv1(c0)

        batch = e3.shape[0]
        e3_flat = e3.permute(0, 2, 3, 1).reshape(batch, 1, -1)
        c1_flat = c1.permute(0, 2, 3, 1).reshape(batch, 1, -1)
        df_embedding = enc.df_fc_emb(c1_flat)
        if enc.enc_concat:
            embedding = torch.cat((e3_flat, df_embedding), dim=-1)
        else:
            embedding = e3_flat + df_embedding
        embedding, encoder_hidden_next = _squeezed_gru(
            enc.emb_gru, embedding, encoder_gru_hidden
        )

        erb = model.erb_dec
        erb_value, erb_hidden_next = _squeezed_gru(
            erb.emb_gru, embedding, erb_gru_hidden
        )
        erb_value = erb_value.reshape(
            batch, 1, erb.n_erb_4, erb.enc_ch
        ).permute(0, 3, 1, 2)
        erb_value = erb.convt3(erb.conv3p(e3) + erb_value)
        erb_value = erb.convt2(erb.conv2p(e2) + erb_value)
        erb_value = erb.convt1(erb.conv1p(e1) + erb_value)
        erb_mask = erb.conv0_out(erb.conv0p(e0) + erb_value)

        decoder = model.df_dec
        df_value, df_hidden_next = _squeezed_gru(
            decoder.df_gru, embedding, df_gru_hidden
        )
        if decoder.df_skip is not None:
            df_value = df_value + decoder.df_skip(embedding)
        c0_residual, pathway_history_next = _df_pathway_step(
            decoder.df_convp, c0, df_convp_history
        )
        c0_residual = c0_residual.permute(0, 2, 3, 1)
        alpha = decoder.df_fc_a(df_value)
        coefficients = decoder.df_out(df_value).view(
            batch, 1, decoder.df_bins, decoder.df_order * 2
        )
        coefficients = coefficients + c0_residual
        return (
            erb_mask,
            coefficients,
            alpha,
            encoder_hidden_next,
            erb_hidden_next,
            df_hidden_next,
            pathway_history_next,
        )


def initial_inputs(model):
    encoder_gru = model.encoder.emb_gru.gru
    erb_gru = model.erb_dec.emb_gru.gru
    df_gru = model.df_dec.df_gru.gru
    enc_ch = model.encoder.erb_conv0[1].out_channels
    pathway_history = model.df_dec.df_convp.conv.kernel_size[0] - 1
    return (
        torch.randn(1, 1, INPUT_FRAMES, model.n_erb),
        torch.randn(1, 2, INPUT_FRAMES, model.df_bins),
        torch.zeros(encoder_gru.num_layers, 1, encoder_gru.hidden_size),
        torch.zeros(erb_gru.num_layers, 1, erb_gru.hidden_size),
        torch.zeros(df_gru.num_layers, 1, df_gru.hidden_size),
        torch.zeros(1, enc_ch, pathway_history, model.df_bins),
    )


def _schema(names, tensors):
    return {
        name: [int(size) for size in tensor.shape]
        for name, tensor in zip(names, tensors)
    }


def set_onnx_metadata(graph, metadata):
    """Write a Python metadata mapping into an ONNX graph's model props.

    ONNX model properties are string-valued, so structured entries have to be
    encoded.  Every exporter that emits a schema dict needs the same rule --
    JSON for containers and bools, ``str`` for scalars -- and a second copy of
    it is a silent divergence: ``str(True)`` is ``'True'`` while
    ``json.dumps(True)`` is ``'true'``, and a consumer parsing one form breaks
    on the other.
    """
    import onnx
    onnx.helper.set_model_props(graph, {
        key: (json.dumps(value, sort_keys=True)
              if isinstance(value, (dict, list, bool)) else str(value))
        for key, value in metadata.items()
    })


def build_metadata(checkpoint_path, params, inputs, outputs):
    """The exported graph's metadata.

    Separate from ``main`` so the contract test can compare
    ``state_layout_version`` against dfn2_model_io.h without an ONNX export.
    """
    return {
        'model_family': 'DeepFilterNet2',
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'boundary': 'stateless_streaming_heads_explicit_state',
        'state_layout_version': STATE_LAYOUT_VERSION,
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        'output_frames_per_invocation': 1,
        'input_window_alignment': '[t-1,t,t+1] -> heads[t]',
        'accelerator_persistent_state': False,
        'host_updates_feature_window': True,
        'state_handoff': {
            name: OUTPUT_NAMES[index + 3]
            for index, name in enumerate(INPUT_NAMES[2:])
        },
        'c_prepost': 'dfn2_process.c/dfn2_process.h',
        'c_model_io': 'dfn2_model_io.c/dfn2_model_io.h',
        'input_schema': _schema(INPUT_NAMES, inputs),
        'output_schema': _schema(OUTPUT_NAMES, outputs),
    }


def export_graph(wrapper, params, checkpoint_path, output_path,
                 opset=17, verify=False):
    """Write the ONNX graph plus its metadata JSON; optionally verify parity.

    Shared by the export CLI and ``inference.py calib``, so the calibration
    tensors and the graph they bind to always come from the same model
    instance in the same process.
    """
    inputs = initial_inputs(wrapper.model)
    with torch.no_grad():
        outputs = wrapper(*inputs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        do_constant_folding=True,
    )

    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(output_path))
    onnx.checker.check_model(graph)
    metadata = build_metadata(checkpoint_path, params, inputs, outputs)
    set_onnx_metadata(graph, metadata)
    onnx.save(graph, output_path)
    with open(os.path.splitext(output_path)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')

    if verify:
        import onnxruntime as ort
        session = ort.InferenceSession(
            output_path, providers=['CPUExecutionProvider']
        )
        actual = session.run(None, {
            name: value.detach().numpy()
            for name, value in zip(INPUT_NAMES, inputs)
        })
        worst = max(
            float(np.max(np.abs(got - want.detach().numpy())))
            for got, want in zip(actual, outputs)
        )
        if worst > 2e-4:
            raise RuntimeError(
                'ONNX parity failed: max abs error %.6g' % worst
            )
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

    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model
    ))
    wrapper = StatelessDFN2Heads(model).eval()
    export_graph(wrapper, params, args.model, args.output,
                 opset=args.opset, verify=args.verify)
    print(args.output)


if __name__ == '__main__':
    main()
