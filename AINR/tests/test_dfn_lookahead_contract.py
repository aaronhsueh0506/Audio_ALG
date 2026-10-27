"""DeepFilterNet2 and DeepFilterNet3 treat the lookahead pair the same way.

Both families share one definition of the window a ``mask_lookahead`` implies
(``onnx_streaming_contract``) and one instruction for a pair that does not match
a checkpoint. These tests hold the two exporters and the two trainers to it, and
run every legal pair through the Python entry points a user meets first:
construct, offline inference and the exporter's guards.
"""


import pytest
import torch

import dfn_pairs

FAMILIES = dfn_pairs.FAMILIES


def _exporter(family):
    return dfn_pairs.import_family_module(family, 'export_onnx')


def test_both_exporters_derive_the_same_window_and_word_the_refusal_alike():
    first, second = (_exporter(family) for family in FAMILIES)
    for mask in range(3):
        assert first.window_frames(mask) == second.window_frames(mask)
        assert first.window_alignment(mask) == second.window_alignment(mask)
    messages = []
    for exporter in (first, second):
        with pytest.raises(ValueError) as refused:
            exporter.window_frames(3)
        messages.append(str(refused.value))
    assert messages[0] == messages[1]
    assert 'model contract' in messages[0]
    assert 'fresh training run' in messages[0]
    assert 'C host built with the same pair' in messages[0]


@pytest.mark.parametrize('bad', [True, 1.0, None, '1', -1, 3])
@pytest.mark.parametrize('family', FAMILIES)
def test_the_exporter_refuses_anything_but_an_integer_in_range(family, bad):
    exporter = _exporter(family)
    with pytest.raises(ValueError, match='model contract'):
        exporter.window_alignment(bad)
    with pytest.raises(ValueError, match='model contract'):
        exporter.feature_windows(torch.zeros(1, 1, 4, 2), bad)


def test_both_trainers_word_a_pair_mismatch_alike():
    messages = {}
    for family in FAMILIES:
        train = dfn_pairs.import_family_module(family, 'train')

        def contract(pair):
            return {'mask_lookahead': pair[0], 'df_lookahead': pair[1]}

        checkpoint = {'model_version': train.MODEL_VERSION,
                      'feature_version': train.FEATURE_VERSION,
                      'loss_version': train.LOSS_VERSION,
                      'contract': contract((1, 1))}
        for key, pair in (('mask_lookahead', (0, 1)),
                          ('df_lookahead', (1, 3))):
            with pytest.raises(ValueError, match=key) as refused:
                train.require_checkpoint_contract(checkpoint, contract(pair))
            messages[(family, key)] = str(refused.value)
    for key in ('mask_lookahead', 'df_lookahead'):
        first, second = (messages[(family, key)] for family in FAMILIES)
        assert first == second
        assert 'model contract' in first and 'fresh training run' in first


@pytest.mark.parametrize('pair', dfn_pairs.accepted_pairs('DeepFilterNet2'),
                         ids=dfn_pairs.pair_id)
@pytest.mark.parametrize('family', FAMILIES)
def test_offline_inference_runs_for_every_legal_pair(
        tmp_path, family, pair):
    """load_model (checkpoint contract included) and the offline inference
    CLI path: a checkpoint trained for ``pair`` loads, and the enhanced file is
    finite and as long as the input."""
    pytest.importorskip('soundfile')
    from types import SimpleNamespace
    import soundfile
    import numpy as np

    assert pair in dfn_pairs.accepted_pairs(family)
    config, checkpoint, wavs = dfn_pairs.write_small_model(
        tmp_path, family, pair, seconds=0.3)
    inference = dfn_pairs.import_family_module(family, 'inference')
    output = tmp_path / 'enhanced.wav'
    inference.denoise_single(SimpleNamespace(
        config=str(config), model=str(checkpoint),
        input=str(wavs / 'noise.wav'), output=str(output), atten_lim=None))
    enhanced, _ = soundfile.read(str(output), dtype='float32')
    source, _ = soundfile.read(str(wavs / 'noise.wav'), dtype='float32')
    assert enhanced.shape == source.shape
    assert np.isfinite(enhanced).all()


@pytest.mark.parametrize('family', FAMILIES)
def test_a_checkpoint_of_another_pair_is_refused_at_load(tmp_path, family):
    from types import SimpleNamespace
    config, checkpoint, _ = dfn_pairs.write_small_model(
        tmp_path, family, (2, 4))
    inference = dfn_pairs.import_family_module(family, 'inference')
    other = tmp_path / 'other.ini'
    text = config.read_text().replace('df_lookahead = 4', 'df_lookahead = 3')
    other.write_text(text)
    with pytest.raises(ValueError, match='df_lookahead') as refused:
        inference.load_model(SimpleNamespace(
            config=str(other), model=str(checkpoint)))
    assert 'fresh training run' in str(refused.value)
