"""
GTCRN 推論腳本

單檔:
    python inference.py --config config.ini --model output/gtcrn_best.pth \
                      --input noisy.wav --output clean.wav

批次:
    python inference.py --config config.ini --model output/gtcrn_best.pth \
                      --input-dir /path/to/noisy --output-dir /path/to/enhanced

Calibration (also exports the ONNX graph the tensors bind to):
    python inference.py calib --model output/gtcrn_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/gtcrn
"""

import argparse
import configparser
import glob
import os
import sys

import torch
import torchaudio
import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)
_AINR_DIR = os.path.dirname(_THIS_DIR)  # home of calibration_io
if _AINR_DIR not in sys.path:
    sys.path.insert(0, _AINR_DIR)

try:
    from .checkpoint_utils import extract_state_dict
    from .model import GTCRN
    from .train import build_contract, require_checkpoint_contract
except ImportError:  # direct ``python inference.py`` execution
    from checkpoint_utils import extract_state_dict
    from model import GTCRN
    from train import build_contract, require_checkpoint_contract


def load_model(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR      = cfg.getint('signal', 'sr')
    N_FFT   = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)
    ERB_SUB1 = cfg.getint('model', 'erb_subband_1', fallback=65)
    ERB_SUB2 = cfg.getint('model', 'erb_subband_2', fallback=64)

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    # Enforce the same contract the resume path does.  Shape mismatches are
    # caught by load_state_dict, but an n_fft or erb_subband change that keeps
    # the shapes intact would otherwise run silently on the wrong grid.  The
    # vendored upstream tars record no contract, so they are exempted rather
    # than rejected.
    require_checkpoint_contract(ckpt, build_contract(cfg, WIN_LEN, HOP_LEN),
                                context=args.model, allow_missing=True)

    model = GTCRN(erb_subband_1=ERB_SUB1, erb_subband_2=ERB_SUB2,
                  nfft=N_FFT, fs=SR)
    # Local checkpoints (train.py) store weights under 'state_dict'; upstream
    # gtcrn tars store them under 'model' (see gtcrn_github/infer.py:11, whose
    # top-level keys are ['epoch', 'optimizer', 'model']).  Accept both so the
    # vendored published checkpoints load without conversion.
    model.load_state_dict(extract_state_dict(ckpt, args.model))
    model.eval()

    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN)
    return model, params


def process_file(input_path, output_path, model, params):
    SR      = params['SR']
    N_FFT   = params['N_FFT']
    WIN_LEN = params['WIN_LEN']
    HOP_LEN = params['HOP_LEN']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]   # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)
    T = audio.shape[-1]

    window = torch.hann_window(WIN_LEN).pow(0.5)
    noisy_spec = torch.view_as_real(torch.stft(
        audio.unsqueeze(0), N_FFT, HOP_LEN, WIN_LEN,
        window=window, return_complex=True,
    ))  # (1, F, T_f, 2)

    with torch.no_grad():
        enhanced_spec = model(noisy_spec)   # (1, F, T_f, 2)

    enh_c = torch.view_as_complex(
        enhanced_spec.permute(0, 2, 1, 3).contiguous()
    ).permute(0, 2, 1)   # (1, F, T_f)
    enhanced_wav = torch.istft(
        enh_c, N_FFT, HOP_LEN, WIN_LEN, window=window, length=T,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # istft keeps the batch dimension: (1, T), already the 2-D
    # (channels, time) layout required by torchaudio.save.
    torchaudio.save(output_path, enhanced_wav, SR)


def denoise_single(args):
    model, params = load_model(args)
    process_file(args.input, args.output, model, params)
    print(f"降噪完成: {args.output}")


def denoise_batch(args):
    model, params = load_model(args)
    wav_files = sorted(glob.glob(
        os.path.join(args.input_dir, '**', '*.wav'), recursive=True
    ))
    if not wav_files:
        raise FileNotFoundError(f"在 {args.input_dir} 找不到任何 .wav 檔案")

    print(f"共 {len(wav_files)} 個檔案 → {args.output_dir}")
    failed = []
    for input_path in tqdm.tqdm(wav_files):
        rel = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)
        try:
            process_file(input_path, output_path, model, params)
        except Exception as e:
            failed.append((rel, str(e)))

    print(f"完成: {len(wav_files) - len(failed)}/{len(wav_files)} 成功")
    if failed:
        print("失敗:")
        for rel, err in failed:
            print(f"  {rel}: {err}")


def calibration_main():
    """Record real streaming spectra and external caches for PTQ."""
    import random

    import numpy as np
    import soundfile as sf

    from calibration_io import (
        CALIBRATION_FORMATS,
        capture_calibration_inputs,
        resolve_calibration_format,
        sibling_onnx_path,
        write_calibration_artifact,
    )
    try:
        from .export_onnx import (
            INPUT_NAMES, build_stream_model, export_graph, initial_inputs,
            stream_features,
        )
    except ImportError:
        from export_onnx import (
            INPUT_NAMES, build_stream_model, export_graph, initial_inputs,
            stream_features,
        )

    parser = argparse.ArgumentParser(description=calibration_main.__doc__)
    parser.add_argument('--config',
                        default=os.path.join(os.path.dirname(__file__),
                                             'config.ini'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--wav-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--format', choices=CALIBRATION_FORMATS, default=None,
                        help='bin or npz; inferred from --output when omitted')
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--onnx', default=None,
                        help='where to write the graph these tensors bind to '
                             '(default: <output>.onnx)')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    try:
        artifact_format = resolve_calibration_format(args.output, args.format)
    except ValueError as error:
        parser.error(str(error))

    files = sorted(glob.glob(
        os.path.join(args.wav_dir, '**', '*.wav'), recursive=True
    ))
    if not files:
        raise FileNotFoundError('no wav files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)
    stream_model, grid = build_stream_model(args.config, args.model)
    # The graph is exported (and parity-checked) in the same process, from the
    # same model instance the tensors below are recorded against, so the two
    # deployment artifacts cannot drift apart.
    onnx_path = sibling_onnx_path(args.output, args.onnx)
    graph_metadata = export_graph(stream_model, grid, args.model, onnx_path,
                                  verify=True)
    captured = {name: [] for name in INPUT_NAMES}
    source_files = []
    window = torch.hann_window(grid['win_len']).sqrt()

    with torch.no_grad():
        for path in files:
            wave, source_sr = sf.read(
                path, dtype='float32', always_2d=True
            )
            wave = torch.from_numpy(wave.mean(axis=1))
            if source_sr != grid['sr']:
                wave = torchaudio.functional.resample(
                    wave, source_sr, grid['sr']
                )
            wave = torch.nn.functional.pad(
                wave, (grid['win_len'] - grid['hop_len'], 0)
            )
            if wave.numel() < grid['win_len']:
                wave = torch.nn.functional.pad(
                    wave, (0, grid['win_len'] - wave.numel())
                )
            spectra = torch.view_as_real(torch.stft(
                wave, grid['n_fft'], grid['hop_len'], grid['win_len'],
                window=window, center=False, return_complex=True,
            )).permute(1, 0, 2)
            state = initial_inputs(stream_model.model)[3:]
            used = False
            for spectrum in spectra:
                features = stream_features(
                    stream_model.model, spectrum[None, :, None, :]
                )
                inputs = tuple(features) + tuple(state)
                capture_calibration_inputs(captured, INPUT_NAMES, inputs)
                state = stream_model(*inputs)[1:]
                used = True
                if len(captured['mag']) >= args.frames:
                    break
            if used:
                source_files.append(os.path.relpath(path, args.wav_dir))
            if len(captured['mag']) >= args.frames:
                break
    if not captured['mag']:
        raise RuntimeError('no calibration frames were produced')
    arrays = {
        name: np.stack(values[:args.frames]).astype(np.float32, copy=False)
        for name, values in captured.items()
    }
    report = {
        'schema': 'gtcrn-stream-calibration-v1',
        'checkpoint_sha256': graph_metadata['checkpoint_sha256'],
        'graph': os.path.basename(onnx_path),
        'frames': int(arrays['mag'].shape[0]),
        'source_files': source_files,
        'sample_rate': grid['sr'],
        'n_fft': grid['n_fft'],
        'inputs': {
            name: {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'min': float(value.min()),
                'max': float(value.max()),
                'p001': float(np.percentile(value, 0.1)),
                'p999': float(np.percentile(value, 99.9)),
            }
            for name, value in arrays.items()
        },
    }
    write_calibration_artifact(
        args.output, arrays, report, artifact_format
    )
    print('%s (%d streaming frames), graph %s' %
          (args.output, arrays['mag'].shape[0], onnx_path))


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'calib':
        del sys.argv[1]
        calibration_main()
        return
    parser = argparse.ArgumentParser(description='GTCRN 推論')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model', required=True, help='模型 .pth 路徑')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--input-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')


if __name__ == '__main__':
    cli()
