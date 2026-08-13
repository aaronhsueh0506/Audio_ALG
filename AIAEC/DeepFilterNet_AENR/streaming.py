#!/usr/bin/env python3
"""DeepFilterNet-AENR STREAMING inference -- frame-by-frame NN path.

用法:
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav
    python3 streaming.py checkpoint.pth mic.wav far.wav out.wav --verify

Same arguments, checkpoint handling and output contract as denoise.py; the
difference is HOW the NN half runs.  STFT -> feature normalisers -> model ->
ISTFT all advance one hop at a time through explicit streaming state
(StreamSTFT / extract_dfn2_features with threaded EMA states /
create_stream_state + forward_stream + flush_stream / StreamISTFT).

The FROZEN PRODUCTION LINEAR AEC frontend still runs offline over the whole
file, exactly like denoise.py: streaming the PBFDKF frontend is a separate
C-side seam (it is already a per-hop engine in production), so this CLI
verifies the NN streaming, not the linear frontend.  The C twin of the
compose rings this Python mirrors is documented in
AINR/DeepFilterNet2/dfn2_process.h (dfn2_compose_stream).

The feature path uses DFN's normalized STFT convention: StreamSTFT frames are
scaled by n_fft**-0.5 (verified bit-equal to ``torch.stft(normalized=True)``)
and the inverse scaling is applied before the WOLA ISTFT.

The model's cascade carries stream_output_delay = 2 hops of algorithmic
lookahead (mask 1 + deep filter 1, serial).  The frame pipeline absorbs it
before the ISTFT -- enhanced frame t is pushed the moment it emerges, two
hops later in wall time -- so the written wav stays time-aligned with
denoise.py's output.

--verify additionally runs the offline whole-wav forward (identical to
denoise.py) and prints the max-abs and RMS difference between the streamed
and offline output waveforms.

On startup this prints (a) the per-invocation I/O table -- every tensor
crossing the frame boundary for one steady-state step -- and (b) the
state_report() RAM inventory after the first frame.  Together these are the
per-frame contract an NPU/C port must reproduce.
"""

import argparse
import configparser
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AINR.DeepFilterNet2.train import extract_dfn2_features, read_feature_config
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft
from AIAEC.DeepFilterNet_AENR.denoise import (
    _normalized_stft,
    build_parser as build_denoise_parser,
    load_model,
)
from AIAEC.inference_common import load_mic_far
from AIAEC.training_common import LinearAecEngine, auto_device


def build_parser() -> argparse.ArgumentParser:
    parser = build_denoise_parser()
    parser.description = __doc__
    parser.add_argument('--verify', action='store_true',
                        help='also run the offline whole-wav forward and print '
                             'max-abs/RMS waveform differences')
    return parser


def _io_table(rows) -> str:
    lines = ["  per-invocation streaming I/O (one steady-state step):"]
    for direction, name, tensor in rows:
        lines.append(f"    {direction:<3s} {name:<28s} "
                     f"{str(tuple(tensor.shape)):<20s} "
                     f"{str(tensor.dtype).replace('torch.', '')}")
    return "\n".join(lines)


def _report_view(state, err_ema, far_ema):
    """Flatten the model state plus the two feature EMA states for reporting."""
    view = {}
    for name, value in state.items():
        if name == "pending_spec":
            if value:
                view[name] = torch.cat(value, dim=1)
        elif name == "df_queue":
            for i, entry in enumerate(value):
                for key, tensor in entry.items():
                    view[f"df_queue{i}.{key}"] = tensor
        elif name == "feat_zero":
            continue
        else:
            view[name] = value
    for label, ema in (("error_feature_ema", err_ema),
                       ("far_feature_ema", far_ema)):
        if ema is not None:
            view[f"{label}.erb"] = ema["erb"]
            view[f"{label}.spec"] = ema["spec"]
    return view


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(args.checkpoint, device)

    feature_cfg_ini = configparser.ConfigParser()
    if not feature_cfg_ini.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")
    feature_cfg = read_feature_config(feature_cfg_ini, grid.sr, grid.hop_len)

    mic_t, far_t, source_rates = load_mic_far(
        args.mic_wav, args.far_wav, grid.sr
    )
    mic_t = mic_t.to(device)
    far_t = far_t.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled mic/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    length = mic_t.shape[-1]

    # Offline frontend, identical to denoise.py (see module docstring).
    linear_aec = LinearAecEngine(
        n_lanes=1, sample_rate=grid.sr, contract=linear_contract
    )
    error, _echo_estimate = linear_aec(mic_t, far_t, grid.sr)
    far_cut = far_t[:, :error.shape[-1]]

    n_fft, hop = grid.n_fft, grid.hop_len
    window = grid.window(device=error.device)
    scale = n_fft ** -0.5          # torch.stft(normalized=True) frame scale
    inv_scale = n_fft ** 0.5

    stft_err = StreamSTFT(n_fft, hop, window)
    stft_far = StreamSTFT(n_fft, hop, window)
    istft_out = StreamISTFT(n_fft, hop, window)
    state = model.create_stream_state()
    err_ema = far_ema = None
    out_chunks = []
    printed_state = printed_table = False
    frames_in = frames_out = 0

    def process_pair(err_frame, far_frame):
        nonlocal err_ema, far_ema, printed_state, printed_table
        nonlocal frames_in, frames_out
        err_spec = (err_frame * scale).unsqueeze(-1)     # (B, n_bins, 1)
        far_spec_c = (far_frame * scale).unsqueeze(-1)
        _, error_erb, error_feat, err_ema = extract_dfn2_features(
            err_spec, model.erb_fb, model.df_bins,
            feature_cfg=feature_cfg, ema_state=err_ema)
        _, far_erb, far_feat, far_ema = extract_dfn2_features(
            far_spec_c, model.erb_fb, model.df_bins,
            feature_cfg=feature_cfg, ema_state=far_ema)
        linear_error = err_spec.transpose(1, 2)          # (B, 1, n_bins)
        with torch.no_grad():
            out = model.forward_stream(
                linear_error=linear_error,
                error_erb=error_erb, error_spec=error_feat,
                far_erb=far_erb, far_spec=far_feat, state=state)
        frames_in += 1
        samples = None
        for k in range(out.enhanced.shape[1]):
            frames_out += 1
            samples = istft_out.push(out.enhanced[:, k] * inv_scale)
            out_chunks.append(samples)
        if not printed_state:
            printed_state = True
            print("state_report after the first frame:")
            print(state_report(_report_view(state, err_ema, far_ema)))
        if not printed_table and out.enhanced.shape[1] > 0:
            # First steady-state step (warm-up steps emit T=0 tensors).
            printed_table = True
            print(_io_table([
                ("in", "err_stft_frame", err_frame),
                ("in", "far_stft_frame", far_frame),
                ("in", "linear_error", linear_error),
                ("in", "error_erb", error_erb),
                ("in", "error_spec", error_feat),
                ("in", "far_erb", far_erb),
                ("in", "far_spec", far_feat),
                ("out", "enhanced", out.enhanced),
                ("out", "mask", out.mask),
                ("out", "deep_filter_coefficients",
                 out.auxiliary["deep_filter_coefficients"]),
                ("out", "deep_filter_alpha",
                 out.auxiliary["deep_filter_alpha"]),
                ("out", "wav_samples", samples),
            ]))
            print("state_report at steady state (full RAM contract):")
            print(state_report(_report_view(state, err_ema, far_ema)))

    for start in range(0, error.shape[-1], hop):
        err_frames = stft_err.push(error[:, start:start + hop])
        far_frames = stft_far.push(far_cut[:, start:start + hop])
        # Hard check, not `assert` (must survive python -O): zip() would
        # silently truncate to the shorter list and drop frames.
        if len(err_frames) != len(far_frames):
            raise RuntimeError(
                f"error/far frame lists diverged: {len(err_frames)} error "
                f"frames vs {len(far_frames)} far frames"
            )
        for ef, ff in zip(err_frames, far_frames):
            process_pair(ef, ff)
    err_tail = stft_err.flush()
    far_tail = stft_far.flush()
    if len(err_tail) != len(far_tail):
        raise RuntimeError(
            f"error/far flush tails diverged: {len(err_tail)} error frames "
            f"vs {len(far_tail)} far frames"
        )
    for ef, ff in zip(err_tail, far_tail):
        process_pair(ef, ff)

    with torch.no_grad():
        tail = model.flush_stream(state)
    for k in range(tail.enhanced.shape[1]):
        frames_out += 1
        out_chunks.append(istft_out.push(tail.enhanced[:, k] * inv_scale))
    emitted = sum(chunk.shape[-1] for chunk in out_chunks)
    out_chunks.append(istft_out.flush(length=length, already_emitted=emitted))
    enhanced_wav = torch.cat(out_chunks, dim=-1)[:, :length]
    print(f"streamed {frames_in} frames in / {frames_out} frames out "
          f"(model delay {model.stream_output_delay} hops, absorbed pre-ISTFT)")

    sf.write(args.out_wav, enhanced_wav.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")

    if args.verify:
        # Offline reference: same error waveform, same path as denoise.py.
        error_spec_full = _normalized_stft(error, grid)
        far_spec_full = _normalized_stft(far_cut, grid)
        _, error_erb, error_feat, _ = extract_dfn2_features(
            error_spec_full, model.erb_fb, model.df_bins,
            feature_cfg=feature_cfg, ema_state=None)
        _, far_erb, far_feat, _ = extract_dfn2_features(
            far_spec_full, model.erb_fb, model.df_bins,
            feature_cfg=feature_cfg, ema_state=None)
        with torch.no_grad():
            offline = model(
                linear_error=error_spec_full.transpose(1, 2),
                error_erb=error_erb, error_spec=error_feat,
                far_erb=far_erb, far_spec=far_feat,
            )
        offline_wav = istft(offline.enhanced.transpose(-2, -1), grid,
                            length=length, normalized=True)
        diff = enhanced_wav - offline_wav
        rms = diff.square().mean().sqrt().item()
        print(f"--verify streamed vs offline: max-abs "
              f"{diff.abs().max().item():.3e}, RMS {rms:.3e}")


if __name__ == '__main__':
    main(build_parser().parse_args())
