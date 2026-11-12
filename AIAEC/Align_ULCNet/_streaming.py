#!/usr/bin/env python3
"""Internal Align-ULCNet frame-by-frame deployment reference.

Everything after the linear AEC is strictly incremental: hop-sized sample chunks
feed StreamSTFT, every finished frame goes through forward_stream (one complex
[B,1,F] frame in, one out), and StreamISTFT reconstructs samples as they become
final. PBFDKF also advances one hop at a time on raw mic/far; the Python PBFDKF
is a reference for the production C seam with the same state lifetime and hop
cadence. ``--far-input-mode`` changes only the model's far branch, not PBFDKF or
its linear_error.

On startup the per-invocation I/O table and the state inventory (the deployment
RAM contract) are printed. ``--verify`` also runs the offline whole-utterance
forward and prints the waveform difference as diagnostics only (no tolerance;
the exit code never changes).

``--events PATH`` replays per-hop alignment events (alignment_events.py): an
event frame resets the model state and is a residual bypass, at every
``--frames-per-call`` (the C ``_freq`` flow; the analysis and synthesis keep
running). Without it no event ever occurs and the flow is unchanged.
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
from AIAEC.Align_ULCNet.alignment_events import AlignmentEvents
from AIAEC.dataset_gen import istft, stft
from AIAEC.inference_common import load_linear_error_far, load_mic_far
from AIAEC._cli_common import (
    print_io_table, require_frames_per_call, require_matched_frames)
from AIAEC.training_common import LinearAecEngine, auto_device


def main(args, load_model_fn=None):
    if load_model_fn is None:
        from AIAEC.Align_ULCNet.inference import load_model as load_model_fn
    device = auto_device(args.device)
    model, grid, linear_contract = load_model_fn(
        args.checkpoint, device, max_delay_frames=args.max_delay_frames
    )
    far_input_mode = getattr(args, 'far_input_mode', 'raw_far')
    print(f"inference model far_input_mode: {far_input_mode}")

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

    events = None
    if getattr(args, 'events', None):
        events = AlignmentEvents.load(args.events)
        events.require_hops(length, grid.hop_len)
        marked = events.event_frames()
        print("alignment events: %d hop records from %s, %d event frame(s)%s"
              % (events.hops, args.events, len(marked),
                 ' %s%s' % (marked[:16], ' ...' if len(marked) > 16 else '')
                 if marked else ''))

    window = grid.window(device=device)
    stft_error = StreamSTFT(grid.n_fft, grid.hop_len, window)
    stft_far = StreamSTFT(grid.n_fft, grid.hop_len, window)
    istft_out = StreamISTFT(grid.n_fft, grid.hop_len, window)
    state = model.create_stream_state()

    packet_size = require_frames_per_call(
        'Align_ULCNet', getattr(args, 'frames_per_call', None))
    packet_wrapper = packet_state = None
    if packet_size > 1:
        # N frames of this one stream per call through the packet graph's
        # PyTorch twin (host front end -> graph -> host back end, the C
        # pipeline's flow). Without --events every frame's event is 0 and the
        # packets continue the model state exactly as the one-frame path does
        # (N = 1 default: forward_stream).
        from AIAEC.Align_ULCNet import export_onnx as packet_export
        packet_wrapper = packet_export.AlignUlcnetPacketExport(
            model, packet_size).eval()
        packet_state = tuple(
            value.to(device) for value in packet_export.dummy_packet_inputs(
                packet_size, packet_wrapper.delay_depth,
                packet_wrapper.n_freqs, packet_wrapper.ta_bins
            )[packet_wrapper.signal_inputs:])

    pieces = []
    reference_error = []
    reference_far = []
    emitted = 0
    frames_done = 0
    calls = 0
    bypassed = 0
    io_table_done = False
    pending_error = []
    pending_far = []

    def run_packet(count):
        """One graph call over the first ``count`` queued frames (a short last
        packet is zero-completed; the zero frames' outputs are dropped and
        their frame_reset is 0)."""
        nonlocal emitted, frames_done, calls, packet_state, bypassed
        error_packet = pending_error[:count]
        far_packet = pending_far[:count]
        del pending_error[:count], pending_far[:count]
        real = len(error_packet)
        first = frames_done
        control = ([0] * packet_size if events is None
                   else events.packet_control(first, real, packet_size))
        for _missing in range(packet_size - real):
            error_packet.append(torch.zeros_like(error_packet[0]))
            far_packet.append(torch.zeros_like(far_packet[0]))
        with torch.no_grad():
            enhanced, packet_state = packet_export.run_packet(
                packet_wrapper, model,
                torch.stack(error_packet, dim=1), torch.stack(far_packet, dim=1),
                control, packet_state)
        if calls == 0:
            print("packet graph: %d frames per call, state carried across calls"
                  % packet_size)
        if events is not None and (calls == 0 or any(control)):
            print("packet %d frame_reset %s" % (
                calls, ''.join(str(int(flag)) for flag in control)))
        calls += 1
        for index in range(real):
            if control[index]:
                # The host fills an event hop with the error spectrum
                # unchanged (the graph's identity mask is only equal to
                # rounding), exactly as the N = 1 stream does.
                enhanced = enhanced.clone()
                enhanced[:, index] = error_packet[index]
                bypassed += 1
            samples = istft_out.push(enhanced[:, index])
            emitted += samples.shape[-1]
            pieces.append(samples)
            frames_done += 1

    def run_frames(error_frames, far_frames, final=False):
        nonlocal emitted, frames_done, state, bypassed, io_table_done
        require_matched_frames(error_frames, far_frames, 'error', 'far')
        if packet_size > 1:
            pending_error.extend(error_frames)
            pending_far.extend(far_frames)
            while len(pending_error) >= packet_size:
                run_packet(packet_size)
            if final and pending_error:
                run_packet(len(pending_error))
            return
        for error_frame, far_frame in zip(error_frames, far_frames):
            if events is not None and events.flag(frames_done):
                # Event hop: state back to zero before it, the hop is a
                # residual bypass and never reaches the model.
                state = model.create_stream_state()
                samples = istft_out.push(error_frame)
                emitted += samples.shape[-1]
                pieces.append(samples)
                frames_done += 1
                bypassed += 1
                continue
            inputs = {
                "linear_error": error_frame.unsqueeze(1),   # complex [B,1,F]
                "far_end": far_frame.unsqueeze(1),
            }
            with torch.no_grad():
                out = model.forward_stream(state=state, **inputs)
            first_model_frame = not io_table_done
            if first_model_frame:
                print_io_table(inputs, out)
                io_table_done = True
            samples = istft_out.push(out.enhanced[:, 0])
            emitted += samples.shape[-1]
            pieces.append(samples)
            frames_done += 1
            if first_model_frame:
                print("streaming state after the first frame:")
                print(state_report({
                    **state,
                    "stft_error": stft_error,
                    "stft_far": stft_far,
                    "istft": istft_out,
                }))

    # Hop-sized chunks: PBFDKF and the neural frontend advance with the same
    # cadence; LinearAecEngine keeps its state across calls. The model state is
    # reset only at the frames --events marks (the C stage's alignment edge,
    # alignment_events.EdgeTracker); the Python PBFDKF has no event accessor, so
    # without --events it runs continuously (README, "C/Python inference
    # parity"; function map D-44).
    for start in range(0, length, grid.hop_len):
        stop = min(start + grid.hop_len, length)
        primary_hop = primary[:, start:stop]
        input_far_hop = input_far[:, start:stop]
        if linear_aec is None:
            error_hop = primary_hop
            model_far_hop = input_far_hop
        else:
            error_hop, _ = linear_aec(primary_hop, input_far_hop, grid.sr)
            if far_input_mode == 'raw_far':
                # Keep the NN input on the checkpoint's training provenance;
                # PBFDKF and its linear_error are unaffected.
                model_far_hop = input_far_hop
            else:
                # Far hop PBFDKF consumed: raw until acquisition/ring fill,
                # aligned afterward. Retained for diagnostic comparison with the
                # raw-far contract.
                model_far_hop = linear_aec.get_aligned_far()
        reference_error.append(error_hop)
        reference_far.append(model_far_hop)
        run_frames(stft_error.push(error_hop.to(device)),
                   stft_far.push(model_far_hop.to(device)))
    run_frames(stft_error.flush(), stft_far.flush(), final=True)
    pieces.append(istft_out.flush(length=length, already_emitted=emitted))
    streamed = torch.cat(pieces, dim=-1)[:, :length]

    sf.write(args.out_wav, streamed.squeeze(0).cpu().numpy(), grid.sr,
             subtype='FLOAT')
    print(f"wrote {args.out_wav} ({length / grid.sr:.2f}s @ {grid.sr} Hz, "
          f"{frames_done} frames"
          f"{f', {calls} model calls of {packet_size}' if packet_size > 1 else ''}"
          f"{f', {bypassed} event frames bypassed' if events is not None else ''})")

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
        if events is not None:
            print("note: the offline reference has no alignment events")
        print(f"verify vs offline forward (diagnostics only): max-abs "
              f"{difference.abs().max().item():.3e}, RMS "
              f"{difference.square().mean().sqrt().item():.3e}")
