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
    erb_bandborder, compute_erb_matrix, RNNoiseModel,
    extract_model_features, read_feature_config,
    require_checkpoint_feature_config, stft, istft,
)


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


def streaming_forward_with_dump(model, erb_features, spec_features,
                                dump_dir, max_frames):
    """逐幀推論並儲存 ONNX 量化校正資料。"""
    os.makedirs(dump_dir, exist_ok=True)
    gru_size = model.gru_size
    h1 = torch.zeros(1, 1, gru_size)
    h2 = torch.zeros(1, 1, gru_size)
    h3 = torch.zeros(1, 1, gru_size)

    saved = 0
    with torch.no_grad():
        for t in range(2, erb_features.size(0)):
            erb_x = erb_features[t-2:t+1, :].unsqueeze(0)
            spec_x = spec_features[t-2:t+1, :, :].unsqueeze(0)
            if saved < max_frames:
                np.save(os.path.join(dump_dir, f'frame_{saved:04d}.npy'), {
                    'erb_input': erb_x.numpy(),
                    'spec_input': spec_x.numpy(),
                    'h1_in': h1.numpy(),
                    'h2_in': h2.numpy(),
                    'h3_in': h3.numpy(),
                })
                saved += 1
            _, states = model(erb_x, spec_x, [h1, h2, h3])
            h1, h2, h3 = states
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
    feature_cfg   = read_feature_config(cfg, SR, HOP_LEN, N_FFT)

    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        raise NotImplementedError(
            "hybrid bands are not supported with the faithful DFN/Keras ERB filterbank; "
            "set hybrid_cutoff_hz=0 to use pure ERB")
    N_BANDS = cfg.getint('signal', 'n_bands')

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    require_checkpoint_feature_config(ckpt, feature_cfg, context=args.model)
    # state_dict 是架構容量的權威來源；feature/signal contract
    # 則已在上方與 runtime config 逐項比對。
    sd = ckpt['state_dict']
    trained_n_bands = sd['erb_conv.weight'].shape[1]
    if trained_n_bands != N_BANDS:
        raise ValueError(
            f"{args.model} n_bands={trained_n_bands}, runtime config={N_BANDS}")
    cond_size = sd['erb_conv.weight'].shape[0]
    gru_size = sd['gru1.weight_ih_l0'].shape[0] // 3
    spec_conv_channels = sd['spec_conv1.weight'].shape[0]
    spec_embed_size = sd['spec_proj.weight'].shape[0]
    model = RNNoiseModel(
        n_bands=N_BANDS, spec_bins=feature_cfg['spec_bins'],
        cond_size=cond_size, gru_size=gru_size,
        spec_conv_channels=spec_conv_channels,
        spec_embed_size=spec_embed_size)
    model.load_state_dict(sd)
    model.eval()

    # ERB band borders: prefer the trained checkpoint's; else recompute (config-driven)
    if 'nfftborder' in ckpt:
        nfftborder = np.array(ckpt['nfftborder'])
    else:
        nfftborder = erb_bandborder(N_BANDS, SR, N_FFT)
    if (nfftborder.shape != (N_BANDS,) or nfftborder[0] != 0 or
            nfftborder[-1] != N_FFT // 2 + 1):
        raise ValueError(f"{args.model} contains an invalid ERB band-border table")

    # Forward ERBB (mode=0, edge x2) for features; inverse (mode=1) for mask→bin
    erb_fwd = torch.from_numpy(compute_erb_matrix(nfftborder, N_FFT, mode=0))
    erb_inv = torch.from_numpy(compute_erb_matrix(nfftborder, N_FFT, mode=1))

    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  N_BANDS=N_BANDS, LOOKAHEAD=LOOKAHEAD, nfftborder=nfftborder,
                  erb_fwd=erb_fwd, erb_inv=erb_inv, FEATURE_CFG=feature_cfg)
    return model, params


def process_file(input_path, output_path, model, params,
                 pf_beta=0.0, dump_calib=None, max_frames=200,
                 dump_debug=None):
    SR         = params['SR']
    N_FFT      = params['N_FFT']
    WIN_LEN    = params['WIN_LEN']
    HOP_LEN    = params['HOP_LEN']
    LOOKAHEAD  = params['LOOKAHEAD']
    erb_fwd    = params['erb_fwd']      # (n_bins, n_bands) mode=0, features
    erb_inv    = params['erb_inv']      # (n_bins, n_bands) mode=1, mask→bin
    feature_cfg = params['FEATURE_CFG']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]  # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)

    window = torch.sqrt(torch.hann_window(WIN_LEN))
    spec = stft(audio, N_FFT, HOP_LEN, WIN_LEN, window)
    # spec: (n_bins, n_frames), normalized=True (fft^-0.5)

    erb_features, spec_features, _, debug = extract_model_features(
        spec.unsqueeze(0), erb_fwd, feature_cfg,
        return_debug=dump_debug is not None)

    if dump_calib:
        streaming_forward_with_dump(
            model, erb_features.squeeze(0), spec_features.squeeze(0),
            dump_calib, max_frames)

    pad_left, pad_right = 2 - LOOKAHEAD, LOOKAHEAD
    erb_padded = F.pad(erb_features, (0, 0, pad_left, pad_right))
    spec_padded = F.pad(spec_features, (0, 0, 0, 0, pad_left, pad_right))
    with torch.no_grad():
        gains, _ = model(erb_padded, spec_padded)
    raw_gains = gains.squeeze(0)
    gains = raw_gains

    # DFN-style: optional Valin post-filter, then mask→bin via mode=1 inverse ERB
    if pf_beta > 0:
        gains = valin_post_filter(gains, beta=pf_beta)

    bin_gains = (gains @ erb_inv.t()).T                  # (n_bins, n_frames), no row-norm

    output = istft(spec * bin_gains, N_FFT, HOP_LEN, WIN_LEN, window, len(audio))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torchaudio.save(output_path, output.unsqueeze(0), SR)

    if dump_debug:
        os.makedirs(os.path.dirname(dump_debug) or '.', exist_ok=True)
        np.savez_compressed(
            dump_debug,
            erb_db=debug['erb_db'].squeeze(0).cpu().numpy(),
            erb_features=erb_features.squeeze(0).cpu().numpy(),
            spec_magnitude=debug['spec_magnitude'].squeeze(0).cpu().numpy(),
            spec_features=spec_features.squeeze(0).cpu().numpy(),
            raw_gains=raw_gains.cpu().numpy(),
            post_gains=gains.cpu().numpy(),
            input_audio=audio.cpu().numpy(),
            output_audio=output.cpu().numpy(),
        )
        print(f"Debug dump 已存: {dump_debug}")


def denoise_single(args):
    model, params = load_model(args)
    process_file(args.input, args.output, model, params,
                 pf_beta=args.pf_beta,
                 dump_calib=args.dump_calib, max_frames=args.max_frames,
                 dump_debug=args.dump_debug)
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
    parser.add_argument('--dump-debug', default=None,
                        help='儲存 ERB/complex features 與 raw/post gains 到 .npz')
    parser.add_argument('--max-frames', type=int, default=200)

    parser.add_argument('--input-dir',  default=None)
    parser.add_argument('--output-dir', default=None)

    parser.add_argument('--pf-beta', type=float, default=0.0,
                        help='Valin post-filter beta (0=off, DFN default=0.02). '
                             'Sharpens low gains toward 0 → deeper steady-state suppression.')

    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')
