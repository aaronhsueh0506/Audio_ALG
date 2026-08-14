#!/usr/bin/env python3
"""Create representative GTCRN streaming inputs for NPU INT8 calibration.

Unlike a collection of zero-state spectra, this tool runs the streaming model
and records the cache values *before* each selected invocation.  The resulting
NPZ therefore represents both signal and recurrent-state distributions.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio

from export_onnx import build_stream_model, file_sha256, initial_inputs


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def frames_from_wav(path, sr, n_fft, win_len, hop_len):
    wave, source_sr = sf.read(path, dtype='float32', always_2d=True)
    wave = torch.from_numpy(wave.mean(axis=1))
    if source_sr != sr:
        wave = torchaudio.functional.resample(wave, source_sr, sr)
    # Streaming analysis begins with win-hop zeros; no center=True reflection.
    wave = torch.nn.functional.pad(wave, (win_len - hop_len, 0))
    if wave.numel() < win_len:
        wave = torch.nn.functional.pad(wave, (0, win_len - wave.numel()))
    count = 1 + (wave.numel() - win_len) // hop_len
    starts = torch.arange(count) * hop_len
    window = torch.hann_window(win_len).sqrt()
    chunks = torch.stack([wave[s:s + win_len] for s in starts.tolist()]) * window
    return torch.view_as_real(torch.fft.rfft(chunks, n=n_fft))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        default=os.path.join(_SCRIPT_DIR, 'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--wav-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    if not args.output.lower().endswith('.npz'):
        parser.error('--output must end in .npz')
    files = sorted(glob.glob(os.path.join(args.wav_dir, '**', '*.wav'), recursive=True))
    if not files:
        raise FileNotFoundError('no wav files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)
    model, grid = build_stream_model(args.config, args.model)
    _, conv, tra, inter = initial_inputs()
    captured = {'mix': [], 'conv_cache': [], 'tra_cache': [], 'inter_cache': []}
    source_files = []
    with torch.no_grad():
        for path in files:
            # File boundaries are stream resets, matching one utterance/session.
            _, conv, tra, inter = initial_inputs()
            spectra = frames_from_wav(path, grid['sr'], grid['n_fft'],
                                      grid['win_len'], grid['hop_len'])
            used = False
            for spectrum in spectra:
                mix = spectrum[None, :, None, :]
                # StreamGTCRN updates cache slices in place.  A numpy view
                # would make every previously captured sample track the same
                # tensor and eventually contain the final state.
                captured['mix'].append(mix.detach().cpu().numpy()[0].copy())
                captured['conv_cache'].append(
                    conv.detach().cpu().numpy().copy())
                captured['tra_cache'].append(
                    tra.detach().cpu().numpy().copy())
                captured['inter_cache'].append(
                    inter.detach().cpu().numpy().copy())
                used = True
                _, conv, tra, inter = model(mix, conv, tra, inter)
                if len(captured['mix']) >= args.frames:
                    break
            if used:
                source_files.append(os.path.relpath(path, args.wav_dir))
            if len(captured['mix']) >= args.frames:
                break
    if not captured['mix']:
        raise RuntimeError('no calibration frames were produced')
    arrays = {name: np.stack(values, axis=0).astype(np.float32, copy=False)
              for name, values in captured.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        'schema': 'gtcrn-stream-calibration-v1', 'seed': args.seed,
        'checkpoint_sha256': file_sha256(args.model),
        'frames': int(arrays['mix'].shape[0]),
        'source_files': source_files,
        'sample_rate': grid['sr'], 'n_fft': grid['n_fft'],
        'inputs': {name: {'shape': list(value.shape),
                          'min': float(value.min()), 'max': float(value.max()),
                          'p001': float(np.percentile(value, 0.1)),
                          'p999': float(np.percentile(value, 99.9))}
                   for name, value in arrays.items()},
    }
    with open(os.path.splitext(args.output)[0] + '.json', 'w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2, sort_keys=True)
        fp.write('\n')
    print('%s (%d streaming frames)' % (args.output, arrays['mix'].shape[0]))


if __name__ == '__main__':
    main()
