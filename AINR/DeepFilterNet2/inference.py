"""
DeepFilterNet2 推論腳本

單檔:
    python inference.py --config config.ini --model output/dfn2_best.pth \
                      --input noisy.wav --output clean.wav

批次:
    python inference.py --config config.ini --model output/dfn2_best.pth \
                      --input-dir /path/to/noisy --output-dir /path/to/enhanced

Calibration (also exports the ONNX graph the tensors bind to):
    python inference.py calib --model output/dfn2_best.pth \
        --wav-dir /path/to/noisy --frames 8192 --format bin \
        --output calib/dfn2
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
    from .model import DeepFilterNet2
    from .train import (
        extract_dfn2_features,
        make_checkpoint_contract,
        read_feature_config,
        read_loss_config,
        require_checkpoint_contract,
        read_model_config,
        validate_signal_config,
    )
except ImportError:  # direct ``python inference.py`` execution
    from model import DeepFilterNet2
    from train import (
        extract_dfn2_features,
        make_checkpoint_contract,
        read_feature_config,
        read_loss_config,
        require_checkpoint_contract,
        read_model_config,
        validate_signal_config,
    )


def load_model(args):
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR      = cfg.getint('signal', 'sr')
    N_FFT   = cfg.getint('signal', 'n_fft')
    WIN_LEN = cfg.getint('signal', 'win_len', fallback=N_FFT)
    HOP_LEN = cfg.getint('signal', 'hop_len', fallback=WIN_LEN // 2)

    model_cfg = read_model_config(cfg)
    N_ERB          = model_cfg['n_erb']
    DF_BINS        = model_cfg['df_bins']
    DF_ORDER       = model_cfg['df_order']
    MASK_LOOKAHEAD = model_cfg['mask_lookahead']
    DF_LOOKAHEAD   = model_cfg['df_lookahead']
    MASK_PF        = model_cfg['mask_pf']
    PF_BETA        = model_cfg['pf_beta']
    validate_signal_config(N_FFT, WIN_LEN, HOP_LEN, N_ERB, DF_BINS, DF_ORDER,
                           MASK_LOOKAHEAD, DF_LOOKAHEAD)

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
        MASK_PF,
        PF_BETA,
        feature_cfg,
        loss_cfg,
    )
    require_checkpoint_contract(
        ckpt, contract, context=args.model, for_training=False
    )

    model = DeepFilterNet2(**model_cfg)
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


def calibration_main():
    """Record DFN2 feature windows and external model state for PTQ."""
    import random
    from types import SimpleNamespace

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
            COMBINED_GRU_STATE_NAME,
            DEFAULT_GRU_STATE_LAYOUT,
            GRU_STATE_LAYOUTS,
            GRU_STATE_NAMES,
            HEAD_OUTPUT_NAMES,
            INPUT_FRAMES,
            StatelessDFN2Heads,
            export_graph,
            feature_windows,
            gru_state_slice_report,
        )
    except ImportError:
        from export_onnx import (
            COMBINED_GRU_STATE_NAME,
            DEFAULT_GRU_STATE_LAYOUT,
            GRU_STATE_LAYOUTS,
            GRU_STATE_NAMES,
            HEAD_OUTPUT_NAMES,
            INPUT_FRAMES,
            StatelessDFN2Heads,
            export_graph,
            feature_windows,
            gru_state_slice_report,
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
    parser.add_argument(
        '--gru-state-layout', choices=sorted(GRU_STATE_LAYOUTS),
        default=DEFAULT_GRU_STATE_LAYOUT,
        help='recurrent-state layout for the graph exported beside these '
             'tensors; see export_onnx.py --gru-state-layout')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    try:
        artifact_format = resolve_calibration_format(args.output, args.format)
    except ValueError as error:
        parser.error(str(error))

    config = configparser.ConfigParser()
    if not config.read(args.config):
        raise FileNotFoundError(args.config)
    files = sorted(glob.glob(
        os.path.join(args.wav_dir, '**', '*.wav'), recursive=True
    ))
    if not files:
        raise FileNotFoundError('no WAV files under %s' % args.wav_dir)
    random.Random(args.seed).shuffle(files)
    model, params = load_model(SimpleNamespace(
        config=args.config, model=args.model
    ))
    wrapper = StatelessDFN2Heads(
        model, gru_state_layout=args.gru_state_layout).eval()
    # The graph is exported (and parity-checked) in the same process, from the
    # same model instance the tensors below are recorded against, so the two
    # deployment artifacts cannot drift apart. The recurrent-state layout comes
    # off that same wrapper below, so the recorded tensor set is the graph's
    # input set by construction rather than by a second reading of the flag.
    onnx_path = sibling_onnx_path(args.output, args.onnx)
    graph_metadata = export_graph(wrapper, params, args.model, onnx_path,
                                  verify=True)
    feature = read_feature_config(
        config, params['SR'], params['HOP_LEN']
    )

    window = torch.hann_window(params['WIN_LEN']).sqrt()
    input_names = wrapper.input_names
    captured = {name: [] for name in input_names}
    source_files = []
    with torch.no_grad():
        for path in files:
            wave, source_rate = sf.read(
                path, dtype='float32', always_2d=True
            )
            wave = torch.from_numpy(wave.mean(axis=1))
            if source_rate != params['SR']:
                wave = torchaudio.functional.resample(
                    wave, source_rate, params['SR']
                )
            spectrum = torch.stft(
                wave.unsqueeze(0),
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
            state = wrapper.initial_inputs()[2:]
            used = False
            for erb_window, spec_window in zip(
                    feature_windows(erb), feature_windows(spec)):
                inputs = (erb_window, spec_window) + state
                capture_calibration_inputs(captured, input_names, inputs)
                outputs = wrapper(*inputs)
                state = tuple(outputs[len(HEAD_OUTPUT_NAMES):])
                used = True
                if len(captured['erb']) >= args.frames:
                    break
            if used:
                source_files.append(os.path.relpath(path, args.wav_dir))
            if len(captured['erb']) >= args.frames:
                break
    if not captured['erb']:
        raise RuntimeError('no calibration frames were produced')
    arrays = {
        name: np.stack(values[:args.frames]).astype(np.float32, copy=False)
        for name, values in captured.items()
    }
    def _stats(value):
        # One percentile call for both quantiles: np.percentile copies and
        # partitions per call, so two calls do that work twice for the same
        # numbers (measured 2x on the captured shapes).
        low, high = np.percentile(value, [0.1, 99.9])
        return {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'min': float(value.min()),
            'max': float(value.max()),
            'p001': float(low),
            'p999': float(high),
        }

    report = {
        # The report FORMAT version, deliberately not the graph's state
        # layout version: the two are independent axes elsewhere in this repo
        # (GTCRN is layout 6 / schema v2, ULCNet layout 5 / schema v1), and
        # the layout is already published on its own line below.
        'schema': 'dfn2-stateless-stream-calibration-v3',
        'gru_state_layout': graph_metadata['gru_state_layout'],
        'checkpoint_sha256': graph_metadata['checkpoint_sha256'],
        'graph': os.path.basename(onnx_path),
        'sample_rate': params['SR'],
        'n_fft': params['N_FFT'],
        'win_len': params['WIN_LEN'],
        'hop_len': params['HOP_LEN'],
        'input_feature_frames': INPUT_FRAMES,
        'frames': int(arrays['erb'].shape[0]),
        'seed': args.seed,
        'source_files': source_files,
        'inputs': {name: _stats(value) for name, value in arrays.items()},
    }
    if wrapper.gru_state_layout.combined:
        report['gru_state_slices'] = gru_state_slice_report(
            arrays[COMBINED_GRU_STATE_NAME], _stats,
            wrapper.gru_state_slices,
        )

    write_calibration_artifact(
        args.output, arrays, report, artifact_format
    )
    print('%s: %d streaming frames, graph %s' %
          (args.output, report['frames'], onnx_path))
    if wrapper.gru_state_layout.combined:
        worst = report['gru_state_slices']['worst_bits_lost']
        print('combined GRU state: one symmetric max-abs int8 scale costs '
              'the narrowest layer %s bits'
              % ('%.2f' % worst if worst is not None else 'n/a'))


def cli():
    if len(sys.argv) > 1 and sys.argv[1] == 'calib':
        del sys.argv[1]
        calibration_main()
        return
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


if __name__ == '__main__':
    cli()
