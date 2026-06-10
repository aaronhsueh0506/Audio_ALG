import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
RNNoise-ERB 推論腳本 (DFN-style inference)

單檔:
    python denoise.py --config config.ini --model output/rnnoise_best.pth \
                      --input noisy.wav --output clean.wav

批次 (保留子目錄結構):
    python denoise.py --config config.ini --model output/rnnoise_best.pth \
                      --input-dir /path/to/noisy --output-dir /path/to/enhanced

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
import torch.nn.functional as F
import torchaudio
import tqdm

from train import (
    compute_erb_bands, compute_hybrid_bands, RNNoiseModel, make_ema_alpha,
)


def build_erb_inv_fb(bin_edges, n_bins):
    """
    ERB → FFT inverse filterbank matrix, shape (n_bands, n_bins).
    DFN-style one-hot: gains.matmul(erb_inv_fb) → per-bin gains.
    """
    n_bands = len(bin_edges) - 1
    fb = torch.zeros(n_bands, n_bins)
    for b in range(n_bands):
        lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
        fb[b, lo:hi] = 1.0
    return fb


def extract_features(power_spec, bin_edges, ema_alpha: float = 0.992,
                     ema_state=None):
    """
    power spectrum → DFN-style causal EMA unit-norm ERB features.
    ema_state: (n_bands,) tensor, pass across calls for true streaming.
    Returns: features (T, n_bands), new_ema_state (n_bands,)
    """
    bands = []
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bands.append(power_spec[..., lo:hi].mean(dim=-1))
    energy = torch.stack(bands, dim=-1).clamp_min(1e-14)  # (T, n_bands)

    state = ema_state if ema_state is not None else energy[0].clone()
    out = []
    for t in range(energy.size(0)):
        e = energy[t]
        state = e * (1 - ema_alpha) + state * ema_alpha
        out.append(e / state.sqrt())
    return torch.stack(out, dim=0), state


def valin_post_filter(mask, beta=0.02):
    """
    Valin post-filter: inference-only mask sharpening.
    Pushes mid-range gains toward 0 or 1 to reduce musical noise.
    beta=0 → identity, DFN default=0.02.
    """
    if beta <= 0:
        return mask
    eps = 1e-12
    mask_sin = mask * torch.sin(np.pi * mask / 2)
    return (1 + beta) * mask / (1 + beta * (mask / mask_sin.clamp_min(eps)).pow(2))


def streaming_forward_with_dump(model, features, dump_dir, max_frames):
    """逐幀推論並儲存 ONNX 量化校正資料。"""
    os.makedirs(dump_dir, exist_ok=True)
    gru_size = model.gru_size
    h1 = torch.zeros(1, 1, gru_size)
    h2 = torch.zeros(1, 1, gru_size)
    h3 = torch.zeros(1, 1, gru_size)

    saved = 0
    with torch.no_grad():
        for t in range(2, features.size(0)):
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
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR        = cfg.getint('signal', 'sr')
    N_FFT     = cfg.getint('signal', 'n_fft')
    WIN_LEN   = cfg.getint('signal', 'win_len',           fallback=N_FFT)
    HOP_LEN   = cfg.getint('signal', 'hop_len',           fallback=WIN_LEN // 2)
    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz',  fallback=0)
    N_ERB_HIGH    = cfg.getint('signal', 'n_erb_high_bands',  fallback=0)
    LOOKAHEAD     = cfg.getint('signal', 'lookahead_frames',  fallback=0)

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

    n_bins = N_FFT // 2 + 1
    erb_inv_fb = build_erb_inv_fb(bin_edges, n_bins)  # (n_bands, n_bins)

    EMA_ALPHA = make_ema_alpha(SR, HOP_LEN)
    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  N_BANDS=N_BANDS, LOOKAHEAD=LOOKAHEAD, bin_edges=bin_edges,
                  erb_inv_fb=erb_inv_fb, EMA_ALPHA=EMA_ALPHA)
    return model, params


def process_file(input_path, output_path, model, params,
                 pf_beta=0.0, dump_calib=None, max_frames=200):
    SR         = params['SR']
    N_FFT      = params['N_FFT']
    WIN_LEN    = params['WIN_LEN']
    HOP_LEN    = params['HOP_LEN']
    LOOKAHEAD  = params['LOOKAHEAD']
    bin_edges  = params['bin_edges']
    erb_inv_fb = params['erb_inv_fb']   # (n_bands, n_bins)
    EMA_ALPHA  = params['EMA_ALPHA']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]  # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)

    window = torch.sqrt(torch.hann_window(WIN_LEN))
    spec = torch.stft(audio, N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
                      window=window, return_complex=True, center=True)
    # spec: (n_bins, n_frames)

    power = spec.abs().pow(2).T  # (n_frames, n_bins)
    features, _ = extract_features(power, bin_edges, EMA_ALPHA)  # (n_frames, n_bands)

    if dump_calib:
        streaming_forward_with_dump(model, features, dump_calib, max_frames)

    features_padded = F.pad(features.unsqueeze(0), (0, 0, 2 - LOOKAHEAD, LOOKAHEAD))
    with torch.no_grad():
        gains, _ = model(features_padded)  # (1, n_frames, n_bands)
    gains = gains.squeeze(0)  # (n_frames, n_bands)

    # DFN-style: optional Valin post-filter, then ERB→FFT via matmul
    if pf_beta > 0:
        gains = valin_post_filter(gains, beta=pf_beta)

    bin_gains = gains.matmul(erb_inv_fb)  # (n_frames, n_bins)
    bin_gains = bin_gains.T               # (n_bins, n_frames)

    output = torch.istft(spec * bin_gains, N_FFT, hop_length=HOP_LEN,
                         win_length=WIN_LEN, window=window, length=len(audio))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, output.unsqueeze(0), SR)


def denoise_single(args):
    model, params = load_model(args)
    process_file(args.input, args.output, model, params,
                 pf_beta=args.pf_beta,
                 dump_calib=args.dump_calib, max_frames=args.max_frames)
    print(f"降噪完成: {args.output}")


def denoise_batch(args):
    model, params = load_model(args)
    wav_files = sorted(glob.glob(
        os.path.join(args.input_dir, '**', '*.wav'), recursive=True))
    if not wav_files:
        raise FileNotFoundError(f"在 {args.input_dir} 找不到任何 .wav 檔案")

    print(f"共 {len(wav_files)} 個檔案 → {args.output_dir}")
    failed = []
    for input_path in tqdm.tqdm(wav_files):
        rel = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)
        try:
            process_file(input_path, output_path, model, params, pf_beta=args.pf_beta)
        except Exception as e:
            failed.append((rel, str(e)))

    print(f"完成: {len(wav_files) - len(failed)}/{len(wav_files)} 成功")
    if failed:
        print("失敗:")
        for rel, err in failed:
            print(f"  {rel}: {err}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNNoise-ERB 推論 (DFN-style, ERB matmul)')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model',  required=True)

    parser.add_argument('--input',  default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--dump-calib', default=None)
    parser.add_argument('--max-frames', type=int, default=200)

    parser.add_argument('--input-dir',  default=None)
    parser.add_argument('--output-dir', default=None)

    parser.add_argument('--pf-beta', type=float, default=0.0,
                        help='Valin post-filter beta (0=off, DFN default=0.02)')

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')
