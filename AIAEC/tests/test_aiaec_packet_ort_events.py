"""The ``inference.py --events`` WAV flow against the C ``_freq`` family (ONNX Runtime as the accelerator).

Sibling of test_aiaec_packet_ort.py (same C packet driver, same CStream / ORT machinery) for ONE event record file:

  * the official CLI (``inference.py ... --events RECORD``, N = 1 and N = 8) writes its PCM and exposes the packet
    ``frame_reset`` it gave the graph;
  * the C ``_freq`` stream (N = 1 and N = 8, layout-18 / layout-19 graphs in ONNX Runtime) is fed the CLI's own analysis
    frames and the RAW ``changed`` / ``solid`` pairs of the same record (never the Python restart flags), so the C
    ``packet_events`` / ``apply_alignment_edge`` decide the events by themselves;
  * compared: the ``frame_reset`` tensor C produced for each packet against the CLI's control (the two edge-rule
    implementations), accelerator calls (one per packet at N = 8; hops minus events at N = 1), the state generation, and
    the synthesized PCM of C N = 8, C N = 1, CLI N = 8, CLI N = 1.

Random weights; see the tolerances in test_aiaec_packet_ort.py.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from test_aiaec_packet_ort import (   # noqa: F401  (fixtures and the C-stream helpers)
    CStream, GRID, F, HOP, N_FFT, NF, SR, driver, driver_path, model, sessions)
from test_align_ulcnet_event_replay import (   # noqa: F401
    CLIP, CLIP_ALIGNED, STREAMS, c_edge_rule, checkpoint, record, run_inference, write_record)

from AIAEC.dataset_gen import AecGrid
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT
from AIAEC.inference_common import load_linear_error_far

TOL_PCM = 1e-4
AEC_GRID = AecGrid(SR, N_FFT, N_FFT, HOP)


def analysis_frames(work, wav):
    """The complex [T, F] error / far spectra the CLI's own analysers produce."""
    error, far, _ = load_linear_error_far(str(work / ('err%s.wav' % wav)), str(work / ('far%s.wav' % wav)), SR)
    window = AEC_GRID.window()
    result = []
    for signal in (error, far):
        stft = StreamSTFT(N_FFT, HOP, window)
        frames = []
        for start in range(0, signal.shape[-1], HOP):
            frames += stft.push(signal[:, start:start + HOP])
        frames += stft.flush()
        result.append(torch.cat(frames, dim=0))
    return result[0], result[1], error.shape[-1]


def c_run(driver, model, session, n, error_c, far_c, changed, solid, length):
    """Every frame through one C stream; returns (PCM, stream, per-packet frame_reset tensors)."""
    stream = CStream(driver, session, model, n)
    controls = []
    real_pre = driver.pre

    def pre(slot, n_hops, err, far, ch, so):
        result = real_pre(slot, n_hops, err, far, ch, so)
        if slot == stream.slot and n > 1 and result[4]:
            controls.append(result[4][4].copy())          # the fifth input tensor of the packet: frame_reset
        return result

    driver.pre = pre
    try:
        frames = error_c.shape[0]
        padded = -(-frames // n) * n
        zero = torch.zeros(padded - frames, F, dtype=error_c.dtype)
        err_all, far_all = torch.cat((error_c, zero)), torch.cat((far_c, zero))
        # frames past the record (the analyser's tail frame) and padding hops carry no event: changed 0 and the
        # previous solid (no false edge, no tracker change)
        missing = padded - len(changed)
        changed_all = list(changed) + [0] * missing
        solid_all = list(solid) + [solid[-1]] * missing
        istft = StreamISTFT(N_FFT, HOP, AEC_GRID.window())
        pieces, emitted = [], 0
        for first in range(0, padded, n):
            sl = slice(first, first + n)
            enhanced = stream.run(err_all[sl], far_all[sl], changed_all[sl], solid_all[sl])
            for k in range(n):
                if first + k < frames:
                    samples = istft.push(torch.from_numpy(enhanced[k:k + 1]))
                    emitted += samples.shape[-1]
                    pieces.append(samples)
        pieces.append(istft.flush(length=length, already_emitted=emitted))
        return torch.cat(pieces, dim=-1)[:, :length].squeeze(0).numpy(), stream, controls
    finally:
        driver.pre = real_pre


def case(name):
    """(stream name, wav suffix, hops, (changed, solid))"""
    if name == 'aligned_length':
        return '_aligned', record(CLIP_ALIGNED // HOP, changed=[5, CLIP_ALIGNED // HOP - 1])
    return '', STREAMS[name]


CASES = ['hop3', 'adjacent', 'cross_packet', 'solid_edge_across_packets', 'last_real_hop', 'mixed', 'aligned_length']


@pytest.mark.parametrize('name', CASES)
def test_cli_and_c_freq_replay_the_same_record(driver, model, sessions, checkpoint, tmp_path, name):
    work, path = checkpoint
    wav, (changed, solid) = case(name)
    record_path = write_record(tmp_path / 'record.json', changed, solid)
    flags = c_edge_rule(changed, solid)

    cli1, _text1, _calls1 = run_inference(work, path, 'cli1.wav', events=record_path, wav=wav)
    cli8, _text8, calls8 = run_inference(work, path, 'cli8.wav', n=8, events=record_path, wav=wav)

    error_c, far_c, length = analysis_frames(work, wav)
    frames = error_c.shape[0]
    pcm8, c8, controls = c_run(driver, model, sessions[NF], NF, error_c, far_c, changed, solid, length)
    pcm1, c1, _ = c_run(driver, model, sessions[1], 1, error_c, far_c, changed, solid, length)

    packets = -(-frames // NF)
    # C's own frame_reset equals the control the CLI gave the graph, packet for packet (padding slots included)
    assert len(controls) == len(calls8) == packets
    for index, (from_c, call) in enumerate(zip(controls, calls8)):
        assert from_c.tolist() == [float(v) for v in call['events']], 'packet %d' % index
    # one accelerator call per packet at N = 8; hops minus events at N = 1; an event at N = 8 keeps the generation
    assert c8.calls == packets and len(c8.generations) == 1
    assert c1.calls == frames - sum(flags[:frames])
    assert len(c1.generations) == 1 + sum(flags[:frames])

    scale = max(np.abs(cli1).max(), 1e-6)
    assert pcm8.shape == pcm1.shape == cli8.shape == cli1.shape == (length,)
    worst = {'c8_cli8': np.abs(pcm8 - cli8).max() / scale, 'c1_cli1': np.abs(pcm1 - cli1).max() / scale,
             'c8_c1': np.abs(pcm8 - pcm1).max() / scale, 'cli8_cli1': np.abs(cli8 - cli1).max() / scale}
    print('%s: events %d, N=8 calls %d, N=1 calls %d; rel max PCM diff %s'
          % (name, sum(flags), c8.calls, c1.calls, {k: '%.2g' % v for k, v in worst.items()}))
    assert max(worst.values()) < TOL_PCM
    assert np.abs(pcm8[-300:] - cli8[-300:]).max() / scale < TOL_PCM            # the drained tail
