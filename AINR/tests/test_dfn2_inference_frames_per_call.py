"""``inference.py --frames-per-call N`` runs the general-WAV mode as the deployment graph does.

One stream, N consecutive feature frames per call (``export_onnx.StatelessDFN2Heads``
with ``frames_per_call=N``), the GRU and deep-filter pathway state carried from call
to call, the last call completed with zero frames and the output cut back to the
file's length. Without the option the whole file runs at once, as before.

  calls     the model is invoked ceil(frames / N) times, N = 1 and 8, on a file whose
            frame count is not a multiple of 8
  equality  N = 8, N = 1 and the whole-file run give the same output, normal and
            ``--mask-only`` files, sample for sample within float accumulation noise
  length    every output file has exactly the input's samples, tail included
  state     a stream whose state is dropped between calls differs from N = 1 by orders
            of magnitude more than the carried one
  lite      the mask-only graph (the lite product's) streamed at N = 8 gives the same
            ERB masks as N = 1 and as the full graph
  onnx      ``calib --frames-per-call 8 --frames 2`` exports and parity-checks the full
            and the mask-only graph; the exported N = 8 graphs streamed in
            onnxruntime reproduce ``stream_heads``
  cli       ``--frames-per-call`` outside [1, 64] stops at argument parsing
"""

import sys

import numpy as np
import pytest
import torch

import dfn_pairs

pytest.importorskip('soundfile')
pytest.importorskip('onnxruntime')

FAMILY = 'DeepFilterNet2'
PAIR = (1, 1)
_module = dfn_pairs.import_family_module


@pytest.fixture(scope='module')
def inference():
    return _module(FAMILY, 'inference')


@pytest.fixture(scope='module')
def export():
    return _module(FAMILY, 'export_onnx')


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    """(config, checkpoint, wav, samples): a random small checkpoint and a noise WAV
    whose STFT frame count is not a multiple of 8 (nor of any N used here)."""
    import soundfile
    root = tmp_path_factory.mktemp('fpc')
    config, checkpoint, _ = dfn_pairs.write_small_model(root, FAMILY, PAIR)
    hop = 512   # the small model's config.ini: 48 kHz / FFT 1024 / hop 512
    samples = hop * 37 + 101
    rng = np.random.default_rng(11)
    wave = (rng.normal(0, 0.08, samples)
            * (1 + 0.5 * np.sin(np.arange(samples) / 700.0))).astype(np.float32)
    wav = root / 'in.wav'
    soundfile.write(str(wav), wave, 48000)
    return config, checkpoint, wav, samples


@pytest.fixture(scope='module')
def loaded(inference, small):
    from types import SimpleNamespace
    config, checkpoint, _, _ = small
    return inference.load_model(SimpleNamespace(config=str(config),
                                                model=str(checkpoint)))


class _CallCounter:
    """Counts StatelessDFN2Heads invocations."""

    def __init__(self, export, monkeypatch):
        self.calls = 0
        original = export.StatelessDFN2Heads.forward
        counter = self

        def counted(module, *args, **kwargs):
            counter.calls += 1
            return original(module, *args, **kwargs)

        monkeypatch.setattr(export.StatelessDFN2Heads, 'forward', counted)


def _run(inference, loaded, small, out, frames_per_call, mask_only=True):
    import soundfile
    model, params = loaded
    _, _, wav, _ = small
    inference.process_file(str(wav), str(out), model, params,
                           mask_only=mask_only,
                           frames_per_call=frames_per_call)
    wave, _ = soundfile.read(str(out), dtype='float32')
    wave_mask, _ = soundfile.read(str(inference.mask_only_path(str(out))),
                                  dtype='float32')
    return wave, wave_mask


def _frames(inference, loaded, small):
    model, params = loaded
    _, _, wav, samples = small
    return samples // params['HOP_LEN'] + 1


def test_the_stream_is_not_a_multiple_of_the_packet(inference, loaded, small):
    frames = _frames(inference, loaded, small)
    assert frames % 8 != 0 and frames % 3 != 0


@pytest.mark.parametrize('frames_per_call', [1, 8])
def test_the_model_is_called_once_per_packet(
        inference, export, loaded, small, tmp_path, monkeypatch, frames_per_call):
    frames = _frames(inference, loaded, small)
    counter = _CallCounter(export, monkeypatch)
    _run(inference, loaded, small, tmp_path / 'o.wav', frames_per_call)
    assert counter.calls == -(-frames // frames_per_call)


def test_default_runs_the_whole_file_without_the_stream_graph(
        inference, export, loaded, small, tmp_path, monkeypatch):
    counter = _CallCounter(export, monkeypatch)
    _run(inference, loaded, small, tmp_path / 'o.wav', None)
    assert counter.calls == 0


def test_n8_equals_n1_equals_the_whole_file(inference, loaded, small, tmp_path):
    _, _, _, samples = small
    whole, whole_mask = _run(inference, loaded, small, tmp_path / 'w.wav', None)
    one, one_mask = _run(inference, loaded, small, tmp_path / 'a.wav', 1)
    eight, eight_mask = _run(inference, loaded, small, tmp_path / 'b.wav', 8)
    for wave in (whole, whole_mask, one, one_mask, eight, eight_mask):
        assert len(wave) == samples
    # same math in a different association order: float accumulation noise only
    tol = 5e-6
    assert np.abs(eight - one).max() < tol, np.abs(eight - one).max()
    assert np.abs(eight_mask - one_mask).max() < tol
    assert np.abs(one - whole).max() < tol, np.abs(one - whole).max()
    assert np.abs(eight - whole).max() < tol
    assert np.abs(eight_mask - whole_mask).max() < tol
    # the tail (the zero-completed packet) matches sample for sample
    assert np.abs(eight[-128:] - whole[-128:]).max() < tol
    # the output is a real enhancement, not silence or a copy of the input
    assert np.abs(eight).max() > 1e-3
    assert np.abs(eight - eight_mask).max() > 1e-6


def test_stream_heads_equals_the_models_heads_over_the_file(
        inference, loaded, small):
    import soundfile
    from AINR.DeepFilterNet2 import train  # noqa: F401  (qualified import)
    model, params = loaded
    wave, _ = soundfile.read(str(small[2]), dtype='float32')
    window = torch.hann_window(params['WIN_LEN']).pow(0.5)
    spec = torch.stft(torch.from_numpy(wave).unsqueeze(0), params['N_FFT'],
                      params['HOP_LEN'], params['WIN_LEN'], window=window,
                      return_complex=True, normalized=True)
    _, erb, feat, _ = inference.extract_dfn2_features(
        spec, model.erb_fb, params['DF_BINS'], feature_cfg=params['FEATURE_CFG'])
    with torch.no_grad():
        want = model.heads(erb, feat)
        for n in (1, 3, 8, 64):
            got = inference.stream_heads(model, erb, feat, n)
            assert got[3] == -(-erb.shape[2] // n)
            for a, b in zip(got[:3], want):
                assert a.shape == b.shape
                assert float((a - b).abs().max()) < 2e-6, (n, float((a - b).abs().max()))


def test_state_is_carried_across_packets(inference, export, loaded, small):
    """Dropping the state between calls is far from the one-frame stream; carrying
    it is not."""
    import soundfile
    model, params = loaded
    wave, _ = soundfile.read(str(small[2]), dtype='float32')
    window = torch.hann_window(params['WIN_LEN']).pow(0.5)
    spec = torch.stft(torch.from_numpy(wave).unsqueeze(0), params['N_FFT'],
                      params['HOP_LEN'], params['WIN_LEN'], window=window,
                      return_complex=True, normalized=True)
    _, erb, feat, _ = inference.extract_dfn2_features(
        spec, model.erb_fb, params['DF_BINS'], feature_cfg=params['FEATURE_CFG'])
    with torch.no_grad():
        carried = inference.stream_heads(model, erb, feat, 8)
        one = inference.stream_heads(model, erb, feat, 1)
        wrapper = export.StatelessDFN2Heads(model, frames_per_call=8).eval()
        dropped = []
        for e, s in zip(export.feature_chunk_windows(erb, model.mask_lookahead, 8),
                        export.feature_chunk_windows(feat, model.mask_lookahead, 8)):
            dropped.append(wrapper(e, s, *wrapper.initial_inputs()[2:])[0])
        dropped = torch.cat(dropped, dim=2)[:, :, :erb.shape[2]]
    good = float((carried[0] - one[0]).abs().max())
    bad = float((dropped - one[0]).abs().max())
    assert good < 2e-6
    # (random small weights make the recurrent state a small term; the gap is still
    # more than two orders above the float noise of the carried stream)
    assert bad > 1e-6 and bad > 100 * good, (good, bad)


def test_the_mask_only_graph_streams_the_same_masks(inference, export, loaded, small):
    """The lite product's graph (no deep-filter decoder) at N = 8 equals its own N = 1
    and the full graph's ERB masks."""
    import soundfile
    model, params = loaded
    wave, _ = soundfile.read(str(small[2]), dtype='float32')
    window = torch.hann_window(params['WIN_LEN']).pow(0.5)
    spec = torch.stft(torch.from_numpy(wave).unsqueeze(0), params['N_FFT'],
                      params['HOP_LEN'], params['WIN_LEN'], window=window,
                      return_complex=True, normalized=True)
    _, erb, feat, _ = inference.extract_dfn2_features(
        spec, model.erb_fb, params['DF_BINS'], feature_cfg=params['FEATURE_CFG'])

    def stream(n):
        wrapper = export.StatelessDFN2Heads(model, frames_per_call=n,
                                            mask_only=True).eval()
        state = wrapper.initial_inputs()[2:]
        masks, calls = [], 0
        for e, s in zip(export.feature_chunk_windows(erb, model.mask_lookahead, n),
                        export.feature_chunk_windows(feat, model.mask_lookahead, n)):
            out = wrapper(e, s, *state)
            masks.append(out[0])
            state = tuple(out[1:])
            calls += 1
        return torch.cat(masks, dim=2)[:, :, :erb.shape[2]], calls

    with torch.no_grad():
        m1, c1 = stream(1)
        m8, c8 = stream(8)
        full = inference.stream_heads(model, erb, feat, 8)[0]
    assert (c1, c8) == (erb.shape[2], -(-erb.shape[2] // 8))
    assert float((m8 - m1).abs().max()) < 2e-6
    assert float((m8 - full).abs().max()) < 2e-6


@pytest.mark.parametrize('mask_only', [False, True])
def test_calib_n8_graph_streamed_in_onnxruntime(
        inference, export, loaded, small, tmp_path, monkeypatch, capsys, mask_only):
    """calib --frames-per-call 8 --frames 2 exports + parity-checks the graph; the
    exported graph streamed in onnxruntime reproduces the torch stream."""
    import onnxruntime
    import soundfile
    config, checkpoint, wav, _ = small
    out = tmp_path / ('calib_lite.npz' if mask_only else 'calib_full.npz')
    wavs = tmp_path / 'wavs'
    wavs.mkdir()
    (wavs / 'a.wav').write_bytes(wav.read_bytes())
    argv = ['inference.py', '--config', str(config), '--model', str(checkpoint),
            '--wav-dir', str(wavs), '--output', str(out),
            '--frames-per-call', '8', '--frames', '2']
    if mask_only:
        argv.append('--mask-only')
    monkeypatch.setattr(sys, 'argv', argv)
    inference.calibration_main()
    printed = capsys.readouterr().out
    arrays = np.load(out)
    assert all(len(arrays[name]) == 2 for name in arrays.files)

    model, params = loaded
    wrapper = export.StatelessDFN2Heads(model, frames_per_call=8,
                                        mask_only=mask_only).eval()
    session = onnxruntime.InferenceSession(str(out.with_suffix('.onnx')),
                                           providers=['CPUExecutionProvider'])
    assert [i.name for i in session.get_inputs()] == list(wrapper.input_names)
    wave, _ = soundfile.read(str(wav), dtype='float32')
    window = torch.hann_window(params['WIN_LEN']).pow(0.5)
    spec = torch.stft(torch.from_numpy(wave).unsqueeze(0), params['N_FFT'],
                      params['HOP_LEN'], params['WIN_LEN'], window=window,
                      return_complex=True, normalized=True)
    _, erb, feat, _ = inference.extract_dfn2_features(
        spec, model.erb_fb, params['DF_BINS'], feature_cfg=params['FEATURE_CFG'])
    ort_state = [t.numpy() for t in wrapper.initial_inputs()[2:]]
    torch_state = wrapper.initial_inputs()[2:]
    worst = 0.0
    with torch.no_grad():
        for e, s in zip(export.feature_chunk_windows(erb, model.mask_lookahead, 8),
                        export.feature_chunk_windows(feat, model.mask_lookahead, 8)):
            feeds = dict(zip(wrapper.input_names, [e.numpy(), s.numpy()] + ort_state))
            got = session.run(None, feeds)
            want = wrapper(e, s, *torch_state)
            for g, w in zip(got, want):
                worst = max(worst, float(np.abs(g - w.numpy()).max()))
            n_heads = len(wrapper.head_output_names)
            ort_state = list(got[n_heads:])
            torch_state = tuple(want[n_heads:])
    print('ORT vs torch, %s N=8: worst abs %.3g over the file; calib output: %s'
          % ('mask-only' if mask_only else 'full', worst, printed.strip()[-300:]))
    assert worst < 1e-4


def test_cli_streams_and_checks_the_range(inference, small, tmp_path, monkeypatch):
    import soundfile
    config, checkpoint, wav, samples = small
    out = tmp_path / 'cli.wav'
    base = ['inference.py', '--config', str(config), '--model', str(checkpoint),
            '--input', str(wav), '--output', str(out), '--mask-only']
    monkeypatch.setattr(sys, 'argv', base + ['--frames-per-call', '8'])
    inference.cli()
    assert len(soundfile.read(str(out))[0]) == samples
    assert len(soundfile.read(str(inference.mask_only_path(str(out))))[0]) == samples
    for bad in ('0', '65', '-1'):
        monkeypatch.setattr(sys, 'argv', base + ['--frames-per-call', bad])
        with pytest.raises(SystemExit) as stop:
            inference.cli()
        assert stop.value.code == 2
