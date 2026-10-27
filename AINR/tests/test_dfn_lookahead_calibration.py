"""The calibration captures of DeepFilterNet2 / DeepFilterNet3 replay the window
of their own (mask_lookahead, df_lookahead) pair and record the pair.

``inference.py calib`` and ``inference_batch.py calib`` slide the feature stream
with ``export_onnx.feature_windows(feature, model.mask_lookahead)``. Nothing
else exercises those two call sites, so a replay that used the wrong window
would hand the quantisation toolchain statistics from a window the host never
builds. Frame 0 enters the window at slot ``2 - L`` (``L`` = mask_lookahead),
so the first captured window has exactly ``2 - L`` all-zero leading slots: that
is what a wrong ``L`` changes, and what these tests read back. The report next
to the tensors names the pair, so a stray artifact is attributable.
"""

import json
import sys

import numpy as np
import pytest

import dfn_pairs

pytest.importorskip('soundfile')
pytest.importorskip('onnxruntime')
onnx = pytest.importorskip('onnx')

INPUT_FRAMES = 3
_BATCH_PAIRS = tuple(dict.fromkeys(
    pair for family in dfn_pairs.FAMILIES
    for pair in (dfn_pairs.shipped_pair(family), *dfn_pairs.SMOKE_PAIRS)))
_module = dfn_pairs.import_family_module
_prepare = dfn_pairs.write_small_model


def _first_window(captured):
    """The first captured window of the ERB input, whatever the leading axes."""
    return captured.reshape(-1, INPUT_FRAMES, captured.shape[-1])[0]


def _assert_window_of(window, mask_lookahead):
    zero_slots = [not window[slot].any() for slot in range(INPUT_FRAMES)]
    expected = [slot < INPUT_FRAMES - 1 - mask_lookahead
                for slot in range(INPUT_FRAMES)]
    assert zero_slots == expected, (zero_slots, expected)


def _check_artifact(output, pair):
    report = json.loads(output.with_suffix('.json').read_text())
    assert (report['mask_lookahead'], report['df_lookahead']) == pair
    assert report['input_feature_frames'] == INPUT_FRAMES
    props = {prop.key: prop.value for prop in
             onnx.load(str(output.with_suffix('.onnx'))).metadata_props}
    assert (int(props['mask_lookahead']), int(props['df_lookahead'])) == pair
    arrays = np.load(output)
    _assert_window_of(_first_window(arrays['erb']), pair[0])
    _assert_window_of(_first_window(arrays['spec']), pair[0])
    return arrays


@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
@pytest.mark.parametrize('pair', dfn_pairs.accepted_pairs('DeepFilterNet2'),
                         ids=dfn_pairs.pair_id)
def test_stream_calibration_replays_the_window_of_its_pair(
        tmp_path, monkeypatch, family, pair):
    assert pair in dfn_pairs.accepted_pairs(family)
    config, checkpoint, wavs = _prepare(tmp_path, family, pair)
    output = tmp_path / 'calib.npz'
    monkeypatch.setattr(sys, 'argv', [
        'inference.py', '--config', str(config), '--model', str(checkpoint),
        '--wav-dir', str(wavs), '--output', str(output), '--frames', '6'])
    _module(family, 'inference').calibration_main()
    arrays = _check_artifact(output, pair)
    # The windows slide one frame per capture: window n+1 drops the oldest
    # row of window n.
    erb = arrays['erb'].reshape(-1, INPUT_FRAMES, arrays['erb'].shape[-1])
    assert len(erb) == 6
    np.testing.assert_array_equal(erb[1:, :-1], erb[:-1, 1:])


@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
@pytest.mark.parametrize('pair', _BATCH_PAIRS, ids=dfn_pairs.pair_id)
def test_fixed_batch_calibration_replays_the_window_of_its_pair(
        tmp_path, monkeypatch, family, pair):
    config, checkpoint, wavs = _prepare(tmp_path, family, pair)
    output = tmp_path / 'batch.npz'
    monkeypatch.setattr(sys, 'argv', [
        'inference_batch.py', 'calib', '--config', str(config),
        '--model', str(checkpoint), '--batch-size', '2',
        '--wav-dir', str(wavs), '--output', str(output), '--batches', '2'])
    _module(family, 'inference_batch').cli()
    _check_artifact(output, pair)
