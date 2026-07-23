"""
DeepFilterNet2 推論腳本

單檔:
    python denoise.py --config config.ini --model output/dfn2_best.pth \
                      --input noisy.wav --output clean.wav

批次:
    python denoise.py --config config.ini --model output/dfn2_best.pth \
                      --input-dir /path/to/noisy --output-dir /path/to/enhanced
"""

import argparse
import configparser
import glob
import os

import torch
import torchaudio
import tqdm

from model import DeepFilterNet2
from train import extract_dfn2_features


def load_model(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR      = cfg.getint('signal', 'sr')
    N_FFT   = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)

    N_ERB      = cfg.getint('model', 'n_erb',       fallback=32)
    DF_BINS    = cfg.getint('model', 'df_bins',     fallback=64)
    DF_ORDER   = cfg.getint('model', 'df_order',    fallback=5)
    EMB_SIZE   = cfg.getint('model', 'emb_size',    fallback=256)
    ENC_CH     = cfg.getint('model', 'enc_channels', fallback=16)
    GRU_GROUPS = cfg.getint('model', 'gru_groups',  fallback=1)

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    model = DeepFilterNet2(
        n_fft=N_FFT, sr=SR, n_erb=N_ERB, df_bins=DF_BINS, df_order=DF_ORDER,
        enc_ch=ENC_CH, emb_size=EMB_SIZE, gru_groups=GRU_GROUPS,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  DF_BINS=DF_BINS)
    return model, params


def process_file(input_path, output_path, model, params):
    SR      = params['SR']
    N_FFT   = params['N_FFT']
    WIN_LEN = params['WIN_LEN']
    HOP_LEN = params['HOP_LEN']
    DF_BINS = params['DF_BINS']

    audio, orig_sr = torchaudio.load(input_path)
    audio = audio[0]   # mono
    if orig_sr != SR:
        audio = torchaudio.functional.resample(audio, orig_sr, SR)
    T = audio.shape[-1]

    window = torch.hann_window(WIN_LEN).pow(0.5)
    spec_c = torch.stft(
        audio.unsqueeze(0), N_FFT, HOP_LEN, WIN_LEN,
        window=window, return_complex=True, normalized=True,
    )  # (1, n_bins, T_f)

    with torch.no_grad():
        spec_c, feat_erb, feat_spec, _ = extract_dfn2_features(
            spec_c, model.erb_fb, DF_BINS
        )
        enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)

    enhanced_wav = torch.istft(
        enhanced_spec, N_FFT, HOP_LEN, WIN_LEN, window=window, length=T, normalized=True,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFilterNet2 推論')
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--model', required=True)
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
