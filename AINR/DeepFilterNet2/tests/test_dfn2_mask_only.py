"""DFN2 stage 1 alone (``--mask-only``): same checkpoint, no deep-filter decoder.

The mask head of the full model and of the mask-only graph must be the same
numbers; the mask-only graph drops the deep-filter decoder (two GRU layers, the
pathway window) and says so in its metadata.
"""

import os
import sys

import pytest
import torch

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from export_onnx import (  # noqa: E402
    MASK_ONLY_STATE_LAYOUT_VERSION,
    STATE_LAYOUT_VERSION,
    StatelessDFN2Heads,
    export_graph,
    feature_chunk_windows,
)
from model import DeepFilterNet2  # noqa: E402

PARAMS = {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256}


def _small():
    torch.manual_seed(5)
    return DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64, enc_ch=8, emb_size=32,
        df_hidden=32, lin_groups=4, enc_lin_groups=4).eval()


def test_mask_only_compose_is_the_cascade_with_alpha_zero():
    model = _small()
    frames = 7
    spec = torch.randn(1, model.n_bins, frames, dtype=torch.complex64)
    erb_mask = torch.rand(1, 1, frames, model.n_erb)
    coefs = torch.randn(1, frames, model.df_bins, model.df_order * 2)
    zero = torch.zeros(1, frames, 1)
    assert torch.equal(model.compose_mask_only(spec, erb_mask),
                       model.compose(spec, erb_mask, coefs, zero))
    # the check can fail: with alpha = 1 the deep filter changes the low bins
    assert not torch.allclose(model.compose_mask_only(spec, erb_mask),
                              model.compose(spec, erb_mask, coefs,
                                            torch.ones(1, frames, 1)))


@pytest.mark.parametrize('frames_per_call', [1, 3, 8])
def test_mask_only_graph_reads_the_same_mask_as_the_full_graph(frames_per_call):
    model = _small()
    full = StatelessDFN2Heads(model, frames_per_call=frames_per_call).eval()
    mask = StatelessDFN2Heads(model, frames_per_call=frames_per_call,
                              mask_only=True).eval()
    erb = torch.randn(1, 1, 24, model.n_erb)
    spec = torch.randn(1, 2, 24, model.df_bins)
    state_full = full.initial_inputs()[2:]
    state_mask = mask.initial_inputs()[2:]
    assert len(state_mask) == 1 and state_mask[0].shape[1] == 3
    with torch.no_grad():
        for erb_w, spec_w in zip(
                feature_chunk_windows(erb, model.mask_lookahead,
                                      frames_per_call),
                feature_chunk_windows(spec, model.mask_lookahead,
                                      frames_per_call)):
            out_full = full(erb_w, spec_w, *state_full)
            out_mask = mask(erb_w, spec_w, *state_mask)
            assert len(out_mask) == 2
            assert torch.allclose(out_full[0], out_mask[0], atol=1e-6)
            # the encoder and ERB rows of h_gru are the same recurrence
            assert torch.allclose(out_full[3][:, :3], out_mask[1], atol=1e-6)
            state_full = tuple(out_full[3:])
            state_mask = (out_mask[1],)


def test_the_full_graph_still_demands_its_pathway_window():
    model = _small()
    full = StatelessDFN2Heads(model).eval()
    erb, spec, gru, _ = full.initial_inputs()
    with pytest.raises(ValueError, match='df_convp_history'):
        full(erb, spec, gru)


def _export(tmp_path, name, frames_per_call, mask_only):
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    wrapper = StatelessDFN2Heads(_small(), frames_per_call=frames_per_call,
                                 mask_only=mask_only).eval()
    path = tmp_path / ('%s.onnx' % name)
    metadata = export_graph(wrapper, PARAMS, str(checkpoint), str(path),
                            verify=True)
    return onnx.load(str(path)), metadata


def _gru_count(graph):
    return sum(node.op_type == 'GRU' for node in graph.graph.node)


def test_the_full_graph_keeps_its_deep_filter_decoder(tmp_path):
    graph, metadata = _export(tmp_path, 'full', 1, False)
    assert _gru_count(graph) == 5
    assert [i.name for i in graph.graph.input] == [
        'erb', 'spec', 'h_gru', 'df_convp_history']
    assert 'variant' not in metadata
    assert metadata['state_layout_version'] == STATE_LAYOUT_VERSION


@pytest.mark.parametrize('frames_per_call', [1, 8])
def test_exported_mask_only_graph_has_no_deep_filter_decoder(
        tmp_path, frames_per_call):
    graph, metadata = _export(tmp_path, 'mask', frames_per_call, True)
    assert _gru_count(graph) == 3
    assert [i.name for i in graph.graph.input] == ['erb', 'spec', 'h_gru']
    assert [o.name for o in graph.graph.output] == ['erb_mask', 'h_gru_out']
    assert metadata['variant'] == 'mask_only'
    assert metadata['state_layout_version'] == MASK_ONLY_STATE_LAYOUT_VERSION
    assert MASK_ONLY_STATE_LAYOUT_VERSION != STATE_LAYOUT_VERSION
    assert metadata['df_lookahead'] == 0
    assert metadata['output_schema'] == {
        'erb_mask': [1, 1, frames_per_call, 32], 'h_gru_out': [1, 3, 1, 32]}
    assert set(metadata['state_handoff']) == {'h_gru'}


def test_inference_writes_both_results_and_atten_lim_mixes_into_each(tmp_path):
    """``--mask-only`` writes the full result and the stage-1 result; the final
    ``--atten-lim`` mix (noisy * lim + enhanced * (1 - lim)) applies to both."""
    from types import SimpleNamespace

    import soundfile
    import torchaudio

    import dfn_pairs
    import inference as inf

    config, checkpoint, wavs = dfn_pairs.write_small_model(
        tmp_path, 'DeepFilterNet2', (1, 1))
    model, params = inf.load_model(
        SimpleNamespace(config=str(config), model=str(checkpoint)))
    source = str(wavs / 'noise.wav')
    noisy = torch.from_numpy(soundfile.read(source, dtype='float32')[0])[None]

    def run(name, atten_lim_db, mask_only):
        path = str(tmp_path / name)
        inf.process_file(source, path, model, params,
                         atten_lim_db=atten_lim_db, mask_only=mask_only)
        return path, torchaudio.load(path)[0]

    plain_path, _ = run('plain.wav', None, False)
    assert not os.path.exists(inf.mask_only_path(plain_path))

    path, full = run('a.wav', None, True)
    mask = torchaudio.load(inf.mask_only_path(path))[0]
    assert full.shape == mask.shape == noisy.shape
    assert not torch.allclose(full, mask)
    path, full_lim = run('b.wav', 12.0, True)
    mask_lim = torchaudio.load(inf.mask_only_path(path))[0]
    # mixing the noisy input back in moves both results toward it
    for limited, unlimited in ((full_lim, full), (mask_lim, mask)):
        assert (limited - noisy).abs().mean() < (unlimited - noisy).abs().mean()
