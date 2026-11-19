"""The ``inference.py --events`` WAV flow against the C ``_freq`` family (ONNX Runtime as the accelerator).

Sibling of test_aiaec_packet_ort.py (same C packet driver, same CStream / ORT machinery, 16 kHz / FFT 512) for ONE
event record file:

  * the official CLI (``inference.py ... --events RECORD``, N = 1 and N = 4) writes its PCM and exposes the packet
    ``frame_reset`` it gave the graph;
  * the C ``_freq`` stream (N = 1 and N = 4, layout-18 / layout-19 graphs in ONNX Runtime) is fed the CLI's own analysis
    frames and the RAW ``changed`` / ``solid`` pairs of the same record (never the Python restart flags), so the C
    ``packet_events`` / ``apply_alignment_edge`` decide the events by themselves;
  * compared: the ``frame_reset`` tensor C produced for each packet against the CLI's control (the two edge-rule
    implementations), accelerator calls (one per packet at N = 4; hops minus events at N = 1), the state generation, and
    the synthesized PCM of C N = 4, C N = 1, CLI N = 4, CLI N = 1.

Random weights; see the tolerances in test_aiaec_packet_ort.py.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from test_aiaec_packet_ort import (   # noqa: F401  (fixtures and the C-stream helpers)
    CStream, GRID, F, HOP, N_FFT, NF, SR, driver, driver_path, model, sessions)
from test_align_ulcnet_event_replay import c_edge_rule, record, run_inference, write_record
from aiaec_shipped_config import ctor_contract

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.dataset_gen import AecGrid, make_linear_aec_contract
from AIAEC.aiaec_streaming import StreamISTFT, StreamSTFT
from AIAEC.inference_common import load_linear_error_far

TOL_PCM = 1e-4
AEC_GRID = AecGrid(SR, N_FFT, N_FFT, HOP)
CLIP = SR + 111 + 5                        # 16116 samples: 63 real hops, the last one partial
HOPS = -(-CLIP // HOP)                     # 63
CLIP_ALIGNED = HOP * 60                    # 15360 samples: 60 hops, the analyser still emits a 61st frame

# Named records over HOPS = 63 hops (16 packets of 4, the last one holding 3 real hops).
STREAMS = {
    'hop3': record(HOPS, changed=[2]),
    'adjacent': record(HOPS, changed=[40, 41]),
    'cross_packet': record(HOPS, changed=[NF - 1, NF]),                 # last slot of packet 0, first of packet 1
    'solid_edge_across_packets': record(HOPS, solid_from=2 * NF),       # hop 7 (last of packet 1) low, hop 8 high
    'last_real_hop': record(HOPS, changed=[HOPS - 1]),                  # the partial hop, inside the short last packet
    'mixed': record(HOPS, changed=[3, 4, 30], solid_from=15, solid_until=45),
}


@pytest.fixture(scope='module')
def checkpoint(model, tmp_path_factory):
    """The packet test's model as a checkpoint the CLI rebuilds, and the error / far WAVs of the two clips."""
    work = tmp_path_factory.mktemp('al512_events')
    linear = make_linear_aec_contract(SR, frame_size=N_FFT)
    contract = {
        'model_name': 'Align_ULCNet', 'task': AlignULCNet.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        **ctor_contract('Align_ULCNet', AlignULCNet, (SR, N_FFT)),
    }
    path = work / 'al512.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    rng = np.random.RandomState(7)
    for name, length in (('', CLIP), ('_aligned', CLIP_ALIGNED)):
        error = 0.1 * rng.randn(length).astype('f')
        far = 0.1 * rng.randn(length).astype('f')
        sf.write(work / ('err%s.wav' % name), error, SR)
        sf.write(work / ('far%s.wav' % name), far, SR)
    return work, str(path)


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
    clin, _textn, callsn = run_inference(work, path, 'clin.wav', n=NF, events=record_path, wav=wav)

    error_c, far_c, length = analysis_frames(work, wav)
    frames = error_c.shape[0]
    pcmn, cn, controls = c_run(driver, model, sessions[NF], NF, error_c, far_c, changed, solid, length)
    pcm1, c1, _ = c_run(driver, model, sessions[1], 1, error_c, far_c, changed, solid, length)

    packets = -(-frames // NF)
    # C's own frame_reset equals the control the CLI gave the graph, packet for packet (padding slots included)
    assert len(controls) == len(callsn) == packets
    for index, (from_c, call) in enumerate(zip(controls, callsn)):
        assert from_c.tolist() == [float(v) for v in call['events']], 'packet %d' % index
    # one accelerator call per packet at N = 4; hops minus events at N = 1; an event at N = 4 keeps the generation
    assert cn.calls == packets and len(cn.generations) == 1
    assert c1.calls == frames - sum(flags[:frames])
    assert len(c1.generations) == 1 + sum(flags[:frames])

    scale = max(np.abs(cli1).max(), 1e-6)
    assert pcmn.shape == pcm1.shape == clin.shape == cli1.shape == (length,)
    worst = {'cn_clin': np.abs(pcmn - clin).max() / scale, 'c1_cli1': np.abs(pcm1 - cli1).max() / scale,
             'cn_c1': np.abs(pcmn - pcm1).max() / scale, 'clin_cli1': np.abs(clin - cli1).max() / scale}
    print('%s: events %d, N=4 calls %d, N=1 calls %d; rel max PCM diff %s'
          % (name, sum(flags), cn.calls, c1.calls, {k: '%.2g' % v for k, v in worst.items()}))
    assert max(worst.values()) < TOL_PCM
    assert np.abs(pcmn[-300:] - clin[-300:]).max() / scale < TOL_PCM            # the drained tail
