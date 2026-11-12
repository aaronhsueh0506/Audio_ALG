"""DeepVQE-S ``--frames-per-call`` in the general-WAV inference CLI and in calib,
on the new 16 kHz / FFT 256 / hop 128 grid.

All weights here are RANDOM (no trained checkpoint exists for this grid): the
tests prove frame order, state hand-off across packets, the zero-completed last
packet and output trimming, not audio quality.
"""

import contextlib
import io
import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.DeepVQE_S import DeepVQES
from AIAEC.DeepVQE_S import inference as dv_inference
from AIAEC.aiaec_common import SignalGrid
from AIAEC._streaming_export import _build, export_graph, host_signal_inputs

SR, N_FFT, HOP = 16000, 256, 128
GRID = SignalGrid(SR, N_FFT, N_FFT, HOP)
# 16000 + 116 samples: 126 STFT frames, not a multiple of 8 (126 = 15 * 8 + 6).
CLIP = 16000 + 37 * 3 + 5


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    work = tmp_path_factory.mktemp('dv256')
    torch.manual_seed(3)
    model = DeepVQES(GRID, max_delay_seconds=0.5).eval()
    contract = {
        'model_name': 'DeepVQE_S', 'task': DeepVQES.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'ctor_max_delay_seconds': 0.5,
        'ctor_similarity_channels': 4, 'ctor_gru_hidden': 192,
        'ctor_compression_exponent': 0.3,
    }
    path = work / 'dv256.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    rng = np.random.RandomState(1)
    sf.write(work / 'mic.wav', 0.1 * rng.randn(CLIP).astype('f'), SR)
    sf.write(work / 'far.wav', 0.1 * rng.randn(CLIP).astype('f'), SR)
    return work, str(path)


def _run_cli(work, checkpoint_path, name, frames_per_call=None):
    argv = [checkpoint_path, str(work / 'mic.wav'), str(work / 'far.wav'),
            str(work / name)]
    if frames_per_call is not None:
        argv += ['--frames-per-call', str(frames_per_call)]
    args = dv_inference.build_parser().parse_args(argv)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        dv_inference.main(args)
    audio, rate = sf.read(str(work / name), dtype='float32')
    assert rate == SR
    return audio, out.getvalue()


def test_wav_inference_n8_equals_n1_and_default(checkpoint):
    work, path = checkpoint
    default, _ = _run_cli(work, path, 'default.wav')
    one, text1 = _run_cli(work, path, 'n1.wav', 1)
    eight, text8 = _run_cli(work, path, 'n8.wav', 8)

    assert default.shape == one.shape == eight.shape == (CLIP,)
    assert np.array_equal(default, one)              # N=1 is the unchanged path
    assert np.abs(one - eight).max() < 2e-6
    # the last packet is partial (126 = 15 * 8 + 6): tail samples included
    assert np.abs(one[-300:] - eight[-300:]).max() < 2e-6
    assert '126 frames' in text1 and 'model calls' not in text1
    assert '126 frames, 16 model calls of 8' in text8   # ceil(126 / 8)


def test_packet_state_matches_frame_by_frame_at_every_boundary():
    torch.manual_seed(7)
    model = DeepVQES(GRID, max_delay_seconds=0.5).eval()
    gen = torch.Generator().manual_seed(2)
    shape = (1, 40, GRID.n_freqs)
    mic = torch.complex(torch.randn(shape, generator=gen), torch.randn(shape, generator=gen))
    far = torch.complex(torch.randn(shape, generator=gen), torch.randn(shape, generator=gen))
    single, packet = model.create_stream_state(), model.create_stream_state()
    with torch.no_grad():
        for start in range(0, 40, 8):
            for i in range(start, start + 8):
                model.forward_stream(mic[:, i:i + 1], far[:, i:i + 1], single)
            model.forward_stream(mic[:, start:start + 8], far[:, start:start + 8], packet)
            for name, cell in single.items():
                for key, value in cell.state_tensors().items():
                    other = packet[name].state_tensors()[key]
                    assert torch.allclose(value, other, atol=2e-5), (start, name, key)


def _frames():
    rng = np.random.RandomState(9)
    n = CLIP
    from AIAEC.dataset_gen import AecGrid, stft
    grid = AecGrid(SR, N_FFT, N_FFT, HOP)
    mic = torch.from_numpy(0.1 * rng.randn(1, n).astype('f'))
    far = torch.from_numpy(0.1 * rng.randn(1, n).astype('f'))
    return (stft(mic, grid).transpose(-2, -1), stft(far, grid).transpose(-2, -1))


def _graph_packets(session, wrapper, model, names, state, mic, far, n):
    """Taps of every real frame, one graph call per packet (zero-completed tail)."""
    frames = mic.shape[1]
    taps, calls = [], 0
    for start in range(0, frames, n):
        real = min(n, frames - start)
        packet = []
        for signal in (mic, far):
            chunk = signal[:, start:start + real]
            if real < n:
                chunk = torch.cat((chunk, torch.zeros_like(chunk[:, :1]).expand(-1, n - real, -1)), dim=1)
            packet.append(torch.view_as_real(chunk))
        mic_in, far_in = host_signal_inputs('DeepVQE_S', model, tuple(packet))
        feed = {'mic': mic_in.numpy(), 'far': far_in.numpy()}
        feed.update({k: v.numpy() for k, v in zip(names, state)})
        outputs = session.run(None, feed)
        taps.append(outputs[0][:, :real])
        state = tuple(torch.from_numpy(o) for o in outputs[1:])
        calls += 1
    return np.concatenate(taps, axis=1), calls


def test_onnx_graphs_n1_n8_agree_over_a_stream_with_a_partial_last_packet(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    torch.manual_seed(5)
    model = DeepVQES(GRID, max_delay_seconds=0.5).eval()
    ckpt = tmp_path / 'c.pth'
    ckpt.write_bytes(b'random-weights')
    mic, far = _frames()
    results = {}
    for n in (1, 8):
        built = _build('DeepVQE_S', model, n)
        wrapper, _inputs, input_names, _outputs, split = built
        graph = tmp_path / ('g%d.onnx' % n)
        from AIAEC.dataset_gen import AecGrid
        meta = export_graph(AecGrid(SR, N_FFT, N_FFT, HOP), built, str(ckpt), str(graph), 63, verify=True,
                            frames_per_call=n)
        assert meta['output_frames_per_invocation'] == n
        assert ('frames_per_call' in meta) == (n != 1)
        session = ort.InferenceSession(str(graph), providers=['CPUExecutionProvider'])
        state = wrapper.initial_state()
        results[n] = _graph_packets(session, wrapper, model,
                                    input_names[split.signal_inputs:], state, mic, far, n)
    taps1, calls1 = results[1]
    taps8, calls8 = results[8]
    frames = mic.shape[1]
    assert calls1 == frames and calls8 == -(-frames // 8)
    assert taps1.shape == taps8.shape
    print('ORT N=1 vs N=8 CCM taps max|diff| = %.3g over %d frames, calls %d vs %d'
          % (np.abs(taps1 - taps8).max(), frames, calls1, calls8))
    # random weights; N=8 vs N=1 differ only by batched-vs-stepped kernels
    assert np.abs(taps1 - taps8).max() < 5e-5


def test_calib_frames_per_call_captures_n_frame_tensors(checkpoint, tmp_path):
    from AIAEC import _streaming_calibration as calibration
    work, path = checkpoint
    mic_dir, far_dir = tmp_path / 'mic', tmp_path / 'far'
    mic_dir.mkdir(), far_dir.mkdir()
    rng = np.random.RandomState(4)
    for directory in (mic_dir, far_dir):
        sf.write(directory / 'a.wav', 0.1 * rng.randn(SR * 3 + 10).astype('f'), SR)
    shapes = {}
    for n in (1, 8):
        out = tmp_path / ('cal%d.npz' % n)
        import sys
        argv = ['calib', '--checkpoint', path, '--primary-dir', str(mic_dir),
                '--far-dir', str(far_dir), '--frames', '4', '--format', 'npz',
                '--frames-per-call', str(n), '--output', str(out)]
        old, sys.argv = sys.argv, argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                calibration.main('DeepVQE_S')
        finally:
            sys.argv = old
        data = np.load(out)
        assert data['mic'].shape == (4, 1, 2, n, GRID.n_freqs)     # 4 INVOCATIONS of N frames
        assert data['far'].shape == (4, 1, 2, n, GRID.n_freqs)
        report = json.loads(str(data['report'])) if 'report' in data.files else None
        shapes[n] = data['mic'].shape
        graph = json.load(open(str(out.with_suffix('.json')))) if out.with_suffix('.json').exists() else None
        onnx_meta = json.load(open(str(tmp_path / ('cal%d.json' % n)))) if (tmp_path / ('cal%d.json' % n)).exists() else None
        for meta in (report, graph, onnx_meta):
            if meta:
                assert meta.get('input_feature_frames', n) == n
    assert shapes[1][3] == 1 and shapes[8][3] == 8


def test_frames_per_call_is_refused_where_unverified_and_out_of_range(checkpoint):
    work, path = checkpoint
    from AIAEC._cli_common import require_frames_per_call
    assert require_frames_per_call('DeepVQE_S', None) == 1
    assert require_frames_per_call('DeepVQE_S', 8) == 8
    with pytest.raises(ValueError, match='not available'):
        require_frames_per_call('CAGCRN', 8)
    with pytest.raises(ValueError, match=r'\[1, 64\]'):
        require_frames_per_call('DeepVQE_S', 65)
    with pytest.raises(ValueError, match=r'\[1, 64\]'):
        require_frames_per_call('DeepVQE_S', 0)
