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
from train import (
    extract_dfn2_features,
    make_checkpoint_contract,
    read_feature_config,
    read_loss_config,
    require_checkpoint_contract,
)


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
    MASK_LOOKAHEAD = cfg.getint('model', 'mask_lookahead', fallback=1)
    DF_LOOKAHEAD = cfg.getint('model', 'df_lookahead', fallback=0)
    EMB_SIZE   = cfg.getint('model', 'emb_size',    fallback=256)
    ENC_CH     = cfg.getint('model', 'enc_channels', fallback=16)
    GRU_GROUPS = cfg.getint('model', 'gru_groups',  fallback=1)
    if not 0 < WIN_LEN <= N_FFT:
        raise ValueError('win_len must be in (0, n_fft]')
    if not 0 < HOP_LEN <= WIN_LEN:
        raise ValueError('hop_len must be in (0, win_len]')
    if not 0 <= MASK_LOOKAHEAD <= 2:
        raise ValueError('mask_lookahead must be in [0, 2]')
    if not 0 <= DF_LOOKAHEAD < DF_ORDER:
        raise ValueError('df_lookahead must be in [0, df_order)')

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    feature_cfg = read_feature_config(cfg, SR, HOP_LEN)
    loss_cfg = read_loss_config(cfg)
    contract = make_checkpoint_contract(
        SR,
        N_FFT,
        WIN_LEN,
        HOP_LEN,
        N_ERB,
        DF_BINS,
        DF_ORDER,
        MASK_LOOKAHEAD,
        DF_LOOKAHEAD,
        feature_cfg,
        loss_cfg,
    )
    require_checkpoint_contract(
        ckpt, contract, context=args.model, require_loss=False
    )

    model = DeepFilterNet2(
        n_fft=N_FFT, sr=SR, n_erb=N_ERB, df_bins=DF_BINS, df_order=DF_ORDER,
        enc_ch=ENC_CH, emb_size=EMB_SIZE, gru_groups=GRU_GROUPS,
        mask_lookahead=MASK_LOOKAHEAD, df_lookahead=DF_LOOKAHEAD,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    params = dict(SR=SR, N_FFT=N_FFT, WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN,
                  DF_BINS=DF_BINS, FEATURE_CFG=feature_cfg)
    return model, params


def apply_atten_lim(noisy_spec, enhanced_spec, atten_lim_db):
    """Attenuation limit, ported verbatim from Rikorose/DeepFilterNet's
    ``enhance.py``: ``enhanced = noisy*lim + enhanced*(1-lim)``,
    ``lim = 10**(-|atten_lim_db|/20)``. Unlike RNNoise-ERB (a real-valued
    per-band gain), DFN2's deep-filter output is a genuinely different
    complex spectrum (multi-tap complex combination, not just noisy times a
    real mask), so the mix has to happen on the complex spectra directly
    rather than on some intermediate gain -- this is the exact mechanism the
    upstream CLI's ``--atten-lim``/``-a`` flag uses.
    """
    if atten_lim_db is None or abs(atten_lim_db) == 0:
        return enhanced_spec
    lim = 10 ** (-abs(atten_lim_db) / 20)
    return noisy_spec * lim + enhanced_spec * (1 - lim)


def process_file(input_path, output_path, model, params, atten_lim_db=None):
    SR      = params['SR']
    N_FFT   = params['N_FFT']
    WIN_LEN = params['WIN_LEN']
    HOP_LEN = params['HOP_LEN']
    DF_BINS = params['DF_BINS']
    feature_cfg = params['FEATURE_CFG']

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
            spec_c, model.erb_fb, DF_BINS,
            feature_cfg=feature_cfg,
        )
        enhanced_spec, _ = model(spec_c, feat_erb, feat_spec)
        enhanced_spec = apply_atten_lim(spec_c, enhanced_spec, atten_lim_db)

    enhanced_wav = torch.istft(
        enhanced_spec, N_FFT, HOP_LEN, WIN_LEN, window=window, length=T, normalized=True,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # istft keeps the batch dimension: (1, T), already the 2-D
    # (channels, time) layout required by torchaudio.save.
    torchaudio.save(output_path, enhanced_wav, SR)


def denoise_single(args):
    model, params = load_model(args)
    process_file(args.input, args.output, model, params, atten_lim_db=args.atten_lim)
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
            process_file(input_path, output_path, model, params,
                        atten_lim_db=args.atten_lim)
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
    parser.add_argument('--atten-lim', type=float, default=None,
                        help='Attenuation limit in dB by mixing the enhanced spectrum '
                             'with the noisy spectrum, matching '
                             "Rikorose/DeepFilterNet enhance.py's --atten-lim: e.g. "
                             '12 only suppresses noise by up to 12dB and keeps the '
                             'rest. None/0 disables it (default, max suppression).')
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        denoise_batch(args)
    elif args.input and args.output:
        denoise_single(args)
    else:
        parser.error('請指定 (--input + --output) 或 (--input-dir + --output-dir)')
