#!/usr/bin/env python3
"""Align-ULCNet inference -- run a trained checkpoint on one mic/far-end pair.

用法:
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav
    python3 denoise.py checkpoint.pth mic.wav far.wav out.wav --device cpu
    python3 denoise.py checkpoint.pth kf_error.wav far.wav out.wav \\
        --input-is-linear-error

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before the linear AEC when needed; output stays at that model rate.

This candidate's input is the FROZEN PRODUCTION LINEAR AEC's error, not the
raw microphone -- see AIAEC/training_common.py's LinearAecEngine. This
script runs that same reference Python engine (RES+CNG disabled) over the
whole file as ONE continuous stream before calling the model, which is the
correct single-utterance case for LinearAecEngine (a fresh engine per call
IS a cold start, matching the file's own beginning -- no cross-call state to
carry, unlike training's per-lane persistence across chunks).

Output is the common denoised, echo-free, early/dereverberated near-end speech
estimate used by every selected AIAEC candidate.

config.ini is not read for model shape: every shape-relevant setting is
recovered from the checkpoint's own contract, including the exact materialized
Python-PBFDKF frontend. Inference refuses a missing or drifted AEC contract.

``--input-is-linear-error`` is an evaluation-only bypass for published demos
that already provide KF residual Z. It does not make an external KF equivalent
to the frozen PBFDKF used to train this repository's checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, istft, stft
from AIAEC.inference_common import load_mic_far
from AIAEC.inference_common import load_linear_error_far
from AIAEC.training_common import (
    LinearAecEngine,
    auto_device,
    checkpoint_far_input_mode,
    require_checkpoint_model_identity,
    require_checkpoint_linear_aec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument(
        'mic_wav', metavar='mic_or_linear_error_wav',
        help='Microphone input, or precomputed error with '
             '--input-is-linear-error',
    )
    parser.add_argument('far_wav')
    parser.add_argument('out_wav')
    parser.add_argument('--device', default=None, help='cuda / cpu / mps (default: auto-detect)')
    parser.add_argument(
        '--input-is-linear-error', action='store_true',
        help='Evaluation only: first WAV is an existing KF/AEC error Z; '
             'bypass this project\'s PBFDKF and run only the neural post-filter. '
             'The supplied far WAV must already be aligned to that error.',
    )
    parser.add_argument(
        '--max-delay-frames', type=int, default=None,
        help='Deployment override for the alignment search depth D. The '
             'checkpoint contract stays the source of truth; when this '
             'differs, only the alignment depth is rebuilt (weights are '
             'D-agnostic, but the output is NOT numerically identical '
             'across D).',
    )
    parser.add_argument(
        '--stream', action='store_true',
        help='Run the model through create_stream_state()/forward_stream() '
             'one STFT frame at a time instead of the whole-utterance '
             'forward, so offline evaluation shares the exact computation '
             'graph with streaming deployment. STFT/ISTFT and the output '
             'path are unchanged.',
    )
    return parser


def load_model(checkpoint_path: str, device: str,
               max_delay_frames: int | None = None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = ckpt['contract']
    require_checkpoint_model_identity(contract, 'Align_ULCNet')
    # Missing field defaults to 'raw_far' (legacy checkpoints); an unknown
    # recorded mode is rejected here, before any weights load. streaming.py
    # shares this loader, so both CLIs print the mode at load time.
    print("checkpoint training far_input_mode: "
          f"{checkpoint_far_input_mode(contract)}; deployment: aligned_far")
    aec_grid = AecGrid(contract['sr'], contract['n_fft'], contract['win_len'], contract['hop_len'])
    linear_aec_contract = require_checkpoint_linear_aec(contract, aec_grid)
    model_grid = SignalGrid(aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len)
    model_kwargs = {
        k[len('ctor_'):]: v for k, v in contract.items() if k.startswith('ctor_')
    }
    # The checkpoint contract stays the source of truth for the model shape.
    # Build from the contract first so the checkpoint's own (possibly
    # grid-derived) alignment depth D is known before any override applies.
    model = AlignULCNet(model_grid, **model_kwargs).to(device)
    if (max_delay_frames is not None
            and int(max_delay_frames) != model.max_delay_frames):
        # Deployment-only override of D. It rebuilds the model with ONLY
        # max_delay_frames replaced; every other constructor argument still
        # comes from the contract, so nothing else can silently drift. D
        # never enters weight shapes (the delay attention projections and
        # score conv are D-agnostic), so the strict load below still
        # verifies every tensor.
        print(f"deployment override: max_delay_frames "
              f"{model.max_delay_frames} -> {int(max_delay_frames)} "
              f"(weights are D-agnostic; output is NOT numerically "
              f"identical across D)")
        override_kwargs = dict(model_kwargs,
                               max_delay_frames=int(max_delay_frames))
        model = AlignULCNet(model_grid, **override_kwargs).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model, aec_grid, linear_aec_contract


def stream_forward_spec(model, error_spec, far_spec):
    """Frame-by-frame replay of the model over precomputed STFT spectra.

    Takes the same complex ``[B,T,F]`` spectra the offline forward takes and
    drives ``create_stream_state()``/``forward_stream`` one frame at a time,
    so the returned enhanced spectrum comes from the exact computation graph
    streaming deployment runs -- bit-identical to it by construction. The
    STFT/ISTFT stay the caller's offline ones.
    """
    state = model.create_stream_state()
    enhanced = []
    with torch.no_grad():
        for t in range(error_spec.shape[1]):
            out = model.forward_stream(
                linear_error=error_spec[:, t:t + 1],
                far_end=far_spec[:, t:t + 1],
                state=state,
            )
            enhanced.append(out.enhanced)
    return torch.cat(enhanced, dim=1)


def main(args):
    device = auto_device(args.device)
    model, grid, linear_contract = load_model(
        args.checkpoint, device, max_delay_frames=args.max_delay_frames
    )

    if args.input_is_linear_error:
        error, far_t, source_rates = load_linear_error_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        print("using external linear-error input; PBFDKF bypassed; supplied "
              "far is assumed aligned")
    else:
        mic_t, far_t, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        linear_aec = LinearAecEngine(
            n_lanes=1, sample_rate=grid.sr, contract=linear_contract
        )
        error, _echo_estimate = linear_aec(
            mic_t, far_t, grid.sr
        )
        # Match the production C pipeline: feed the exact far hop PBFDKF
        # consumed (raw until acquisition, ring-aligned afterward).
        far_t = linear_aec.get_aligned_far()
    error = error.to(device)
    far_t = far_t.to(device)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled primary/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    length = error.shape[-1]

    error_spec = stft(error, grid).transpose(-2, -1)   # [B,T,F], the public model boundary
    far_spec = stft(far_t, grid).transpose(-2, -1)

    if args.stream:
        enhanced_spec = stream_forward_spec(model, error_spec, far_spec)
        print(f"streamed {error_spec.shape[1]} frames through forward_stream")
    else:
        with torch.no_grad():
            output = model(linear_error=error_spec, far_end=far_spec)
        enhanced_spec = output.enhanced

    enhanced = istft(enhanced_spec.transpose(-2, -1), grid, length=length)
    sf.write(args.out_wav, enhanced.squeeze(0).cpu().numpy(), grid.sr, subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz)")


if __name__ == '__main__':
    main(build_parser().parse_args())
