#!/usr/bin/env python3
"""Capture representative inputs for the stateless DFN2 ONNX graph.

The tool replays complete streams. It records three-frame feature windows and
the real GRU/pathway states immediately before each selected invocation, so
PTQ sees the same non-zero state distributions as deployment.
"""

import argparse
import configparser
import glob
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torchaudio

try:
    from .denoise import load_model
    from .export_onnx import (
        INPUT_FRAMES,
        INPUT_NAMES,
        StatelessDFN2Heads,
        feature_windows,
        file_sha256,
        initial_inputs,
    )
    from .train import extract_dfn2_features, read_feature_config
except ImportError:  # direct ``python export_calibration.py`` execution
    from denoise import load_model
    from export_onnx import (
        INPUT_FRAMES,
        INPUT_NAMES,
        StatelessDFN2Heads,
        feature_windows,
        file_sha256,
        initial_inputs,
    )
    from train import extract_dfn2_features, read_feature_config


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_wave(path, sample_rate):
    wave, source_rate = sf.read(path, dtype='float32', always_2d=True)
    wave = torch.from_numpy(wave.mean(axis=1))
    if source_rate != sample_rate:
        wave = torchaudio.functional.resample(wave, source_rate, sample_rate)
    return wave.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--wav-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=256,
                        help='number of streaming invocations to capture')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    if not args.output.lower().endswith('.npz'):
        parser.error('--output must end in .npz')

    config = configparser.ConfigParser()
    if not config.read(args.config):
        raise FileNotFoundError(args.config)
    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model
    ))
    wrapper = StatelessDFN2Heads(model).eval()
    feature = read_feature_config(config, params['SR'], params['HOP_LEN'])
    files = sorted(glob.glob(
        os.path.join(args.wav_dir, '**', '*.wav'), recursive=True
    ))
    if not files:
        raise FileNotFoundError('no WAV files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)

    window = torch.hann_window(params['WIN_LEN']).sqrt()
    captured = {name: [] for name in INPUT_NAMES}
    source_files = []
    with torch.no_grad():
        for path in files:
            wave = _read_wave(path, params['SR'])
            spectrum = torch.stft(
                wave,
                params['N_FFT'],
                params['HOP_LEN'],
                params['WIN_LEN'],
                window=window,
                normalized=True,
                return_complex=True,
            )
            _, erb, spec, _ = extract_dfn2_features(
                spectrum, model.erb_fb, model.df_bins, feature, None
            )
            state = tuple(value for value in initial_inputs(model)[2:])
            used = False
            for erb_window, spec_window in zip(
                feature_windows(erb), feature_windows(spec)
            ):
                inputs = (erb_window, spec_window) + state
                for name, value in zip(INPUT_NAMES, inputs):
                    sample = value.detach().cpu().numpy()
                    # Remove only the graph batch axis from feature windows.
                    if name.startswith('feat_'):
                        sample = sample[0]
                    captured[name].append(sample.copy())
                outputs = wrapper(*inputs)
                state = tuple(outputs[3:])
                used = True
                if len(captured['feat_erb_window']) >= args.frames:
                    break
            if used:
                source_files.append(os.path.relpath(path, args.wav_dir))
            if len(captured['feat_erb_window']) >= args.frames:
                break
    if not captured['feat_erb_window']:
        raise RuntimeError('no calibration frames were produced')

    arrays = {
        name: np.stack(values[:args.frames]).astype(np.float32, copy=False)
        for name, values in captured.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        'schema': 'dfn2-stateless-stream-calibration-v2',
        'checkpoint_sha256': file_sha256(args.model),
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        'frames': int(arrays['feat_erb_window'].shape[0]),
        'seed': args.seed,
        'source_files': source_files,
        'inputs': {
            name: {
                'shape': list(value.shape),
                'min': float(value.min()),
                'max': float(value.max()),
                'p001': float(np.percentile(value, 0.1)),
                'p999': float(np.percentile(value, 99.9)),
            }
            for name, value in arrays.items()
        },
    }
    with open(os.path.splitext(args.output)[0] + '.json', 'w',
              encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('%s: %d streaming frames' %
          (args.output, report['frames']))


if __name__ == '__main__':
    main()
