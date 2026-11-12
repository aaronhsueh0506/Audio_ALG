"""DeepVQE-S ``frames_per_call`` > 1: chunked forward_stream equals
frame-by-frame, and the exported graph for N > 1 passes the same in-place
state contract and ONNX-runtime parity check as N == 1.

Unlike DFN2, DeepVQE-S carried no frames_per_call concept before this: the
cells (DelayRingCell/FrameDelayAttentionCell/StreamConv2dCell in
AIAEC/aiaec_streaming.py) and the export wiring (AIAEC/_streaming_export.py)
were generalized to batch N frames per accelerator call. These tests are the
regression gate for that generalization, pinned at N == 1 byte-for-byte and
at N > 1 to small, bounded floating-point drift from the GRU's
batched-vs-stepped kernel path (the same divergence
test_streaming_deepvqe_s.py already documents for N == 1 chunk size).
"""

from types import SimpleNamespace

import pytest
import torch

from AIAEC.DeepVQE_S.model import DeepVQES
from AIAEC.aiaec_common import SignalGrid
from AIAEC._streaming_export import _build, export_graph, FRAMES_PER_CALL_READY_MODELS
from AIAEC._onnx_contract import validate_inplace_state_contract

GRID = SignalGrid(16000, 512, 512, 256)
FRAMES = 40

TOL_ENHANCED = 2e-5
TOL_AUX = 5e-6


def _model(seed: int = 0) -> DeepVQES:
    torch.manual_seed(seed)
    model = DeepVQES(GRID)
    model.eval()
    return model


def _random_pair(seed: int, frames: int = FRAMES, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, frames, GRID.n_freqs)

    def spec():
        return torch.complex(torch.randn(*shape, generator=g),
                             torch.randn(*shape, generator=g))
    return spec(), spec()


def _run_chunked(model, mic, far, chunk):
    state = model.create_stream_state()
    enhanced, taps = [], []
    total = mic.shape[1]
    i = 0
    with torch.no_grad():
        while i < total:
            step = min(chunk, total - i)
            out = model.forward_stream(mic[:, i:i + step], far[:, i:i + step],
                                       state)
            enhanced.append(out.enhanced)
            taps.append(out.auxiliary["ccm_taps"])
            i += step
    return torch.cat(enhanced, dim=1), torch.cat(taps, dim=1)


@pytest.mark.parametrize("chunk", (1, 2, 4, 5, 8))
def test_chunked_forward_stream_matches_frame_by_frame(chunk):
    model = _model()
    mic, far = _random_pair(1)
    enhanced_1, taps_1 = _run_chunked(model, mic, far, 1)
    enhanced_n, taps_n = _run_chunked(model, mic, far, chunk)
    assert (enhanced_n - enhanced_1).abs().max().item() < TOL_ENHANCED
    assert (taps_n - taps_1).abs().max().item() < TOL_AUX


def test_chunk_size_one_is_exactly_the_old_code_path():
    # chunk == 1 must still be the original single-frame arithmetic, not
    # merely "close": every op in that branch is untouched by the N > 1
    # generalization.
    model = _model()
    mic, far = _random_pair(2)
    enhanced, taps = _run_chunked(model, mic, far, 1)
    state = model.create_stream_state()
    enhanced_ref, taps_ref = [], []
    with torch.no_grad():
        for t in range(mic.shape[1]):
            out = model.forward_stream(mic[:, t:t + 1], far[:, t:t + 1], state)
            enhanced_ref.append(out.enhanced)
            taps_ref.append(out.auxiliary["ccm_taps"])
    enhanced_ref = torch.cat(enhanced_ref, dim=1)
    taps_ref = torch.cat(taps_ref, dim=1)
    assert torch.equal(enhanced, enhanced_ref)
    assert torch.equal(taps, taps_ref)


def _export(model_name, model, frames_per_call, tmp_path):
    built = _build(model_name, model, frames_per_call)
    wrapper, _inputs, input_names, output_names, split = built
    checkpoint = tmp_path / ("ckpt_%d.bin" % frames_per_call)
    checkpoint.write_bytes(b"hashed only")
    path = str(tmp_path / ("deepvqe_n%d.onnx" % frames_per_call))
    grid = SimpleNamespace(sr=16000, n_fft=512, win_len=512, hop_len=256)
    metadata = export_graph(grid, built, str(checkpoint), path, 0,
                            verify=True, frames_per_call=frames_per_call)
    return built, metadata, path


def test_frames_per_call_one_records_no_metadata_key(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(83)
    model = DeepVQES(GRID).eval()
    _built, metadata, _path = _export("DeepVQE_S", model, 1, tmp_path)
    assert "frames_per_call" not in metadata
    assert metadata["input_feature_frames"] == 1
    assert metadata["output_frames_per_invocation"] == 1


@pytest.mark.parametrize("n", (2, 4))
def test_frames_per_call_n_passes_in_place_contract_and_onnx_parity(n, tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(83)
    model = DeepVQES(GRID).eval()
    built, metadata, path = _export("DeepVQE_S", model, n, tmp_path)
    wrapper, _inputs, input_names, output_names, split = built

    assert metadata["frames_per_call"] == n
    assert metadata["input_feature_frames"] == n
    assert metadata["output_frames_per_invocation"] == n
    assert "frames_per_call" not in metadata["c_descriptor"]
    assert metadata["input_schema"]["mic"][2] == n
    assert metadata["output_schema"]["output"][1] == n

    loaded = onnx.load(path)
    count = validate_inplace_state_contract(
        loaded, input_names[split.signal_inputs:],
        output_names[split.head_outputs:],
        window_drops=wrapper.window_state_drops)
    assert count == len(wrapper.window_state_names)


def test_frames_per_call_rejects_out_of_range():
    model = DeepVQES(GRID).eval()
    with pytest.raises(ValueError):
        _build("DeepVQE_S", model, frames_per_call=0)
    with pytest.raises(ValueError):
        _build("DeepVQE_S", model, frames_per_call=65)


def test_other_models_reject_frames_per_call_greater_than_one():
    # Align_CRUSE/CAGCRN's attention/stream cells have not been generalized
    # or verified for N > 1 -- frames_per_call must fail loudly for them,
    # never silently trace an unverified graph.
    from AIAEC.Align_CRUSE import AlignCRUSE
    from AIAEC.CAGCRN import CAGCRN

    assert FRAMES_PER_CALL_READY_MODELS == ("DeepVQE_S",)
    for name, factory in (("Align_CRUSE", lambda: AlignCRUSE(GRID)),
                          ("CAGCRN", lambda: CAGCRN(GRID))):
        model = factory().eval()
        with pytest.raises(ValueError):
            _build(name, model, frames_per_call=4)
