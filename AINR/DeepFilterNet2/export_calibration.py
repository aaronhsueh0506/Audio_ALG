#!/usr/bin/env python3
"""Export representative DFN2 feature blocks for accelerator calibration.

This is distinct from ``calibrate_norm_init.py``: that script measures EMA
initialisation constants, while this tool serialises the actual ONNX inputs
``feat_erb`` and ``feat_spec`` for INT8/PTQ calibration.  Feature EMA state is
continuous within each WAV and resets at file boundaries.
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

from denoise import load_model
from export_onnx import file_sha256
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
    parser.add_argument('--frames', type=int, required=True,
                        help='must match export_onnx.py --frames')
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.frames <= 0 or args.blocks <= 0:
        parser.error('--frames and --blocks must be positive')
    if not args.output.lower().endswith('.npz'):
        parser.error('--output must end in .npz')

    config = configparser.ConfigParser()
    if not config.read(args.config):
        raise FileNotFoundError(args.config)
    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model))
    feature = read_feature_config(config, params['SR'], params['HOP_LEN'])
    files = sorted(glob.glob(os.path.join(args.wav_dir, '**', '*.wav'),
                             recursive=True))
    if not files:
        raise FileNotFoundError('no WAV files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)
    window = torch.hann_window(params['WIN_LEN']).sqrt()
    captured = {'feat_erb': [], 'feat_spec': []}
    source_files = []
    with torch.no_grad():
        for path in files:
            wave = _read_wave(path, params['SR'])
            spectrum = torch.stft(
                wave, params['N_FFT'], params['HOP_LEN'], params['WIN_LEN'],
                window=window, normalized=True, return_complex=True)
            _, erb, spec, _ = extract_dfn2_features(
                spectrum, model.erb_fb, model.df_bins, feature, None)
            used = False
            for start in range(0, erb.shape[2] - args.frames + 1, args.frames):
                captured['feat_erb'].append(
                    erb[0, :, start:start + args.frames].cpu().numpy().copy())
                captured['feat_spec'].append(
                    spec[0, :, start:start + args.frames].cpu().numpy().copy())
                used = True
                if len(captured['feat_erb']) >= args.blocks:
                    break
            if used:
                source_files.append(os.path.relpath(path, args.wav_dir))
            if len(captured['feat_erb']) >= args.blocks:
                break
    if not captured['feat_erb']:
        raise RuntimeError('no complete %d-frame blocks were found' % args.frames)
    arrays = {name: np.stack(values[:args.blocks], axis=0).astype(
        np.float32, copy=False) for name, values in captured.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        'schema': 'dfn2-heads-calibration-v1',
        'checkpoint_sha256': file_sha256(args.model),
        'sample_rate': params['SR'], 'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'], 'hop_len': params['HOP_LEN'],
        'frames_per_block': args.frames,
        'blocks': int(arrays['feat_erb'].shape[0]),
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
