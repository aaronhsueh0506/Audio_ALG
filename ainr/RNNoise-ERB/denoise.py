import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
RNNoise v0.2 風格噪音抑制 — 推論腳本

單檔:
    python denoise.py --config config.ini --model output/rnnoise_best.pth \
                      --input noisy.wav --output clean.wav

批次 (保留子目錄結構):
    python denoise.py --config config.ini --model output/rnnoise_best.pth \
                      --input-dir /path/to/vctk/noisy \
                      --output-dir /path/to/vctk/enhanced

量化校正資料:
    python denoise.py --config config.ini --model output/rnnoise_best.pth \
                      --input noisy.wav --output clean.wav \
                      --dump-calib calib/ --max-frames 200
"""

import argparse
import configparser
import glob
import os

import numpy as np
import torch
import torchaudio
import tqdm

from train import (
    compute_erb_bands, compute_hybrid_bands, RNNoiseModel,
)


def extract_features(power_spec, bin_edges):
    """power spectrum → log ERB band energy (正規化)"""
    bands = []
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bands.append(power_spec[..., lo:hi].sum(dim=-1))
    energy = torch.stack(bands, dim=-1)
    log_energy = torch.log(energy + 1e-10)
    mean = log_energy.mean(dim=-2, keepdim=True)
    std = log_energy.std(dim=-2, keepdim=True) + 1e-8
    return (log_energy - mean) / std


def streaming_forward_with_dump(model, features, dump_dir, max_frames):
    """Streaming 逐幀推論，同時存下每幀的 ONNX 輸入供量化校正。"""
    os.makedirs(dump_dir, exist_ok=True)
    n_frames = features.size(0)
    gru_size = model.gru_size

    h1 = torch.zeros(1, 1, gru_size)
    h2 = torch.zeros(1, 1, gru_size)
    h3 = torch.zeros(1, 1, gru_size)

    saved = 0
    with torch.no_grad():
        for t in range(2, n_frames):
            x = features[t-2:t+1, :].unsqueeze(0)

            if saved < max_frames:
                np.save(os.path.join(dump_dir, f'frame_{saved:04d}.npy'), {
                    'input': x.numpy(),
                    'h1_in': h1.numpy(),
                    'h2_in': h2.numpy(),
                    'h3_in': h3.numpy(),
                })
                saved += 1

            tmp = x.permute(0, 2, 1)
            tmp = torch.tanh(model.conv1(tmp))
            tmp = torch.tanh(model.conv2(tmp))
            conv_out = tmp.permute(0, 2, 1)

            g1, h1 = model.gru1(conv_out, h1)
            g2, h2 = model.gru2(g1, h2)
            g3, h3 = model.gru3(g2, h3)

    print(f"校正資料已存: {dump_dir}/ ({saved} frames)")


def load_model(args):
    """載入 config + model，回傳推論所需的所有參數。"""
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)
    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    LOOKAHEAD = cfg.getint('signal', 'lookahead_frames', fallback=0)

    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        _, N_BANDS = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
    else:
        N_BANDS = cfg.getint('signal', 'n_bands')

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=64, gru_size=128)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if 'bin_edges' in ckpt:
        bin_edges = np.array(ckpt['bin_edges'])
    elif HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        bin_edges, _ = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
    else:
        bin_edges = compute_erb_bands(N_FFT, SR, N_BANDS)

    GAIN_FLOOR    = cfg.getfloat('inference', 'gain_floor',    fallback=0.02)
    ATTACK_ALPHA  = cfg.getfloat('inference', 'attack_alpha',  fallback=0.5)
    RELEASE_ALPHA = cfg.getfloat('inference', 'release_alpha', fallback=0.15)

    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  N_BANDS=N_BANDS, LOOKAHEAD=LOOKAHEAD, bin_edges=bin_edges,
                  GAIN_FLOOR=GAIN_FLOOR, ATTACK_ALPHA=ATTACK_ALPHA,
                  RELEASE_ALPHA=RELEASE_ALPHA)
    return model, params


def process_file(input_path, output_path, model, params, dump_calib=None, max_frames=200):
    """單一 wav 檔案降噪，輸出到 output_path。"""
    SR = params['SR']
    N_FFT = params['N_FFT']
    WIN_LEN = params['WIN_LEN']
    HOP_LEN = params['HOP_LEN']
    N_BANDS       = params['N_BANDS']
    LOOKAHEAD     = params['LOOKAHEAD']
    bin_edges     = params['bin_edges']
    GAIN_FLOOR    = params['GAIN_FLOOR']
    ATTACK_ALPHA  = params['ATTACK_ALPHA']
    RELEASE_ALPHA = params['RELEASE_ALPHA']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]  # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)

    window = torch.sqrt(torch.hann_window(WIN_LEN))
    spec = torch.stft(audio, N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
                      window=window, return_complex=True, center=True)

    power = spec.abs().pow(2).T  # (n_frames, n_bins)
    features = extract_features(power, bin_edges)  # (n_frames, n_bands)

    if dump_calib:
        streaming_forward_with_dump(model, features, dump_calib, max_frames)

    with torch.no_grad():
        gains, _ = model(features.unsqueeze(0))  # (1, n_frames-2, n_bands)
    gains = gains.squeeze(0)

    # Asymmetric temporal gain smoothing
    for t in range(1, gains.size(0)):
        alpha = torch.where(gains[t] > gains[t - 1], ATTACK_ALPHA, RELEASE_ALPHA)
        gains[t] = alpha * gains[t] + (1 - alpha) * gains[t - 1]

    gains = torch.clamp(gains, min=GAIN_FLOOR)

    n_bins = spec.size(0)
    n_frames_out = gains.size(0)
    bin_gains = torch.ones(n_bins, spec.size(1))

    gain_offset = 2 - LOOKAHEAD
    for b in range(N_BANDS):
        lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
        bin_gains[lo:hi, gain_offset:gain_offset + n_frames_out] = gains[:, b].unsqueeze(0)

    filtered = spec * bin_gains
    output = torch.istft(filtered, N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
                         window=window, length=len(audio))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, output.unsqueeze(0), SR)


def denoise_single(args):
    model, params = load_model(args)
    process_file(args.input, args.output, model, params,
                 dump_calib=args.dump_calib, max_frames=args.max_frames)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNNoise v0.2 推論 (config-driven, ERB bands)')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--model', required=True, help='模型 .pth 檔案路徑')

    # 單檔模式
    parser.add_argument('--input', default=None, help='輸入 wav (單檔模式)')
    parser.add_argument('--output', default=None, help='輸出 wav (單檔模式)')
    parser.add_argument('--dump-calib', default=None, help='存量化校正資料的目錄')
    parser.add_argument('--max-frames', type=int, default=200, help='最多存幾幀校正資料')

    # 批次模式
    parser.add_argument('--input-dir', default=None, help='輸入目錄 (批次模式，遞迴掃描 .wav)')
    parser.add_argument('--output-dir', default=None, help='輸出目錄 (批次模式，保留子目錄結構)')

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')
