#!/usr/bin/env python3
"""Internal Align-ULCNet frame-by-frame deployment reference.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify
    python3 inference.py checkpoint.pth kf_error.wav far.wav out.wav \\
        --input-is-linear-error

Checkpoint loading and the audio frontend are identical to inference.py, far
branch included: the model is fed the linear AEC's own aligned-far seam, not
the raw far WAV.  PBFDKF, STFT, network, and ISTFT all advance one input hop at
a time.  The Python PBFDKF is only a reference for the production C seam, but
its state lifetime and hop cadence are the same here.

Everything after the linear AEC is strictly incremental: hop-sized sample
chunks feed StreamSTFT, every finished frame goes through forward_stream (one
complex [B,1,F] frame in, one out), and StreamISTFT reconstructs samples as
they become final.  On startup the per-invocation I/O table and the state
inventory (the deployment RAM contract) are printed.  ``--verify`` also runs
the offline whole-utterance forward and reports the waveform difference.  That
print is DIAGNOSTICS ONLY: no tolerance is applied and the exit code never
changes.  The streaming equivalence gate is
tests/test_streaming_align_ulcnet.py::test_stream_matches_offline, not this
flag.
"""

import os
import sys

import soundfile as sf
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT, state_report
from AIAEC.dataset_gen import istft, stft
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC._cli_common import print_io_table, require_matched_frames
from AIAEC.training_common import LinearAecEngine, auto_device


def main(args, load_model_fn=None):
    if load_model_fn is None:
        from AIAEC.Align_ULCNet.inference import load_model as load_model_fn
    device = auto_device(args.device)
    model, grid, linear_contract = load_model_fn(
        args.checkpoint, device, max_delay_frames=args.max_delay_frames
    )

    if args.input_is_linear_error:
        primary, input_far, source_rates = load_linear_error_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        print("using external linear-error input; PBFDKF bypassed")
        linear_aec = None
    else:
        primary, input_far, source_rates = load_mic_far(
            args.mic_wav, args.far_wav, grid.sr
        )
        linear_aec = LinearAecEngine(
            n_lanes=1, sample_rate=grid.sr, contract=linear_contract,
            delay_num_filters=getattr(args, 'delay_num_filters', None),
        )
        print('PBFDKF matched-filter bank: n=%d' %
              linear_aec.delay_num_filters)
    if source_rates != (grid.sr, grid.sr):
        print(f"resampled primary/far {source_rates[0]}/{source_rates[1]} -> "
              f"{grid.sr} Hz")
    if primary.shape != input_far.shape:
        raise RuntimeError(
            f"primary/far lengths differ "
            f"({primary.shape[-1]}/{input_far.shape[-1]})"
        )
    length = primary.shape[-1]

    window = grid.window(device=device)
    stft_error = StreamSTFT(grid.n_fft, grid.hop_len, window)
    stft_far = StreamSTFT(grid.n_fft, grid.hop_len, window)
    istft_out = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    pieces = []
    reference_error = []
    reference_far = []
    emitted = 0
    frames_done = 0

    def run_frames(error_frames, far_frames):
        nonlocal emitted, frames_done
        require_matched_frames(error_frames, far_frames, 'error', 'far')
        for error_frame, far_frame in zip(error_frames, far_frames):
            inputs = {
                "linear_error": error_frame.unsqueeze(1),   # complex [B,1,F]
                "far_end": far_frame.unsqueeze(1),
            }
            with torch.no_grad():
                out = model.forward_stream(state=state, **inputs)
            if frames_done == 0:
                print_io_table(inputs, out)
            samples = istft_out.push(out.enhanced[:, 0])
            emitted += samples.shape[-1]
            pieces.append(samples)
            frames_done += 1
            if frames_done == 1:
                print("streaming state after the first frame:")
                print(state_report({
                    **state,
                    "stft_error": stft_error,
                    "stft_far": stft_far,
                    "istft": istft_out,
                }))

    # Hop-sized sample chunks: PBFDKF and the neural frontend advance with the
    # same cadence.  Calling LinearAecEngine on each hop preserves its engine
    # state; it does not reconstruct a fresh AEC per call.
    for start in range(0, length, grid.hop_len):
        stop = min(start + grid.hop_len, length)
        primary_hop = primary[:, start:stop]
        input_far_hop = input_far[:, start:stop]
        if linear_aec is None:
            error_hop = primary_hop
            aligned_far_hop = input_far_hop
        else:
            error_hop, _ = linear_aec(primary_hop, input_far_hop, grid.sr)
            # Exact far hop PBFDKF consumed: raw until acquisition/ring fill,
            # aligned afterward.  This is the production model seam.
            aligned_far_hop = linear_aec.get_aligned_far()
        reference_error.append(error_hop)
        reference_far.append(aligned_far_hop)
        run_frames(stft_error.push(error_hop.to(device)),
                   stft_far.push(aligned_far_hop.to(device)))
    run_frames(stft_error.flush(), stft_far.flush())
    pieces.append(istft_out.flush(length=length, already_emitted=emitted))
    streamed = torch.cat(pieces, dim=-1)[:, :length]

    sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames_done} frames)")

    if args.verify:
        error = torch.cat(reference_error, dim=-1).to(device)
        far = torch.cat(reference_far, dim=-1).to(device)
        error_spec = stft(error, grid).transpose(-2, -1)
        far_spec = stft(far, grid).transpose(-2, -1)
        with torch.no_grad():
            reference = model(linear_error=error_spec, far_end=far_spec)
        offline = istft(reference.enhanced.transpose(-2, -1), grid,
                        length=length)
        difference = streamed - offline
        print(f"verify vs offline forward (diagnostics only): max-abs "
              f"{difference.abs().max().item():.3e}, RMS "
              f"{difference.square().mean().sqrt().item():.3e}")
