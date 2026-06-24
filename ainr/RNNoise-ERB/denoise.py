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
    erb_bandborder, compute_hybrid_bands, compute_erb_matrix, RNNoiseModel,
    make_ema_alpha, stft, istft,
)


def extract_features(power_spec, erb_matrix, ema_alpha: float = 0.99,
                     ema_state=None, mean_norm_init=(-60.0, -90.0)):
    """
    power spectrum → DeepFilterNet feat_erb (faithful port). MUST mirror
    train.extract_erb_features (same recurrence) so the model sees the feature
    distribution it trained on:
        erb_db = 10*log10(power @ erb_matrix + 1e-10)      # triangular ERB band energy
        band_mean_norm_erb: state=x*(1-a)+state*a ; out=(x-state)/40
        state init = linspace(-60, -90)  (MEAN_NORM_INIT)
    erb_matrix: (n_bins, n_bands) triangular ERB FB. STFT must be normalized=True (fft^-0.5).
    ema_state: running-mean tensor (n_bands,) to continue across streaming chunks; None = fresh.
    Returns: features (T, n_bands), new_ema_state (n_bands,)
    """
    energy = power_spec @ erb_matrix                      # (T, n_bands)
    erb_db = 10.0 * torch.log10(energy + 1e-10)           # → dB

    if ema_state is not None:
        state = ema_state
    else:
        lo_i, hi_i = mean_norm_init
        state = torch.linspace(lo_i, hi_i, erb_db.shape[-1],
                               device=erb_db.device, dtype=erb_db.dtype)
    out = []
    for t in range(erb_db.size(0)):
        state = erb_db[t] * (1.0 - ema_alpha) + state * ema_alpha
        out.append((erb_db[t] - state) / 40.0)
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
        raise NotImplementedError(
            "hybrid bands are not supported with the faithful DFN/Keras ERB filterbank; "
            "set hybrid_cutoff_hz=0 to use pure ERB")
    N_BANDS = cfg.getint('signal', 'n_bands')

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    # 容量優先讀 ckpt (避免 train/denoise 漂移), 退回 config, 再退回舊預設
    ck_cfg = ckpt.get('config', {})
    cond_size = ck_cfg.get('cond_size', cfg.getint('model', 'cond_size', fallback=64))
    gru_size = ck_cfg.get('gru_size', cfg.getint('model', 'gru_size', fallback=128))
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=cond_size, gru_size=gru_size)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # ERB band borders: prefer the trained checkpoint's; else recompute (config-driven)
    if 'nfftborder' in ckpt:
        nfftborder = np.array(ckpt['nfftborder'])
    else:
        nfftborder = erb_bandborder(N_BANDS, SR, N_FFT)

    # Forward ERBB (mode=0, edge x2) for features; inverse (mode=1) for mask→bin
    erb_fwd = torch.from_numpy(compute_erb_matrix(nfftborder, N_FFT, mode=0))
    erb_inv = torch.from_numpy(compute_erb_matrix(nfftborder, N_FFT, mode=1))

    EMA_ALPHA = make_ema_alpha(SR, HOP_LEN)
    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  N_BANDS=N_BANDS, LOOKAHEAD=LOOKAHEAD, nfftborder=nfftborder,
                  erb_fwd=erb_fwd, erb_inv=erb_inv, EMA_ALPHA=EMA_ALPHA)
    return model, params


def process_file(input_path, output_path, model, params,
                 pf_beta=0.0, dump_calib=None, max_frames=200):
    SR         = params['SR']
    N_FFT      = params['N_FFT']
    WIN_LEN    = params['WIN_LEN']
    HOP_LEN    = params['HOP_LEN']
    LOOKAHEAD  = params['LOOKAHEAD']
    erb_fwd    = params['erb_fwd']      # (n_bins, n_bands) mode=0, features
    erb_inv    = params['erb_inv']      # (n_bins, n_bands) mode=1, mask→bin
    EMA_ALPHA  = params['EMA_ALPHA']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]  # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)

    window = torch.sqrt(torch.hann_window(WIN_LEN))
    spec = stft(audio, N_FFT, HOP_LEN, WIN_LEN, window)
    # spec: (n_bins, n_frames), normalized=True (fft^-0.5)

    power = spec.abs().pow(2).T  # (n_frames, n_bins)
    features, _ = extract_features(power, erb_fwd, EMA_ALPHA)  # (n_frames, n_bands)

    if dump_calib:
        streaming_forward_with_dump(model, features, dump_calib, max_frames)

    features_padded = F.pad(features.unsqueeze(0), (0, 0, 2 - LOOKAHEAD, LOOKAHEAD))
    with torch.no_grad():
        gains, _ = model(features_padded)  # (1, n_frames, n_bands)
    gains = gains.squeeze(0)  # (n_frames, n_bands)

    # DFN-style: optional Valin post-filter, then mask→bin via mode=1 inverse ERB
    if pf_beta > 0:
        gains = valin_post_filter(gains, beta=pf_beta)

    bin_gains = (gains @ erb_inv.t()).T                  # (n_bins, n_frames), no row-norm

    output = istft(spec * bin_gains, N_FFT, HOP_LEN, WIN_LEN, window, len(audio))

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

    parser.add_argument('--pf-beta', type=float, default=0.02,
                        help='Valin post-filter beta (0=off, DFN default=0.02). '
                             'Sharpens low gains toward 0 → deeper steady-state suppression.')

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')
