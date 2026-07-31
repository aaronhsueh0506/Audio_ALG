import configparser

import pytest
import torch

from AIAEC.Align_CRUSE import AlignCRUSE
from AIAEC.DeepFilterNet_AENR import DeepFilterNetAENR
from AIAEC.aiaec_common import SignalGrid
from AIAEC.training_common import (
    LinearAecEngine,
    compressed_spectral_loss,
    make_checkpoint_contract,
    read_grids,
    read_model_kwargs,
    require_checkpoint_contract,
)
from AINR.DeepFilterNet2.model import DeepFilterNet2


def _cfg(text: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


def test_read_grids_builds_matching_aec_and_signal_grids():
    aec_grid, model_grid = read_grids(_cfg('[signal]\nsr=16000\nn_fft=512\n'))
    assert (aec_grid.sr, aec_grid.n_fft, aec_grid.win_len, aec_grid.hop_len) == (16000, 512, 512, 256)
    assert (model_grid.sample_rate, model_grid.n_fft, model_grid.win_len, model_grid.hop_len) == \
        (16000, 512, 512, 256)


def test_read_model_kwargs_overlays_only_declared_keys():
    kwargs = read_model_kwargs(_cfg('[model]\ngru_hidden = 96\n'), AlignCRUSE)
    assert kwargs['gru_hidden'] == 96
    assert kwargs['alignment_mode'] == 'paper_global'   # untouched constructor default


def test_read_model_kwargs_rejects_unknown_key():
    with pytest.raises(ValueError, match='not a AlignCRUSE constructor argument'):
        read_model_kwargs(_cfg('[model]\nnot_a_real_arg = 1\n'), AlignCRUSE)


def test_read_model_kwargs_merges_extra_bases_for_kwargs_forwarding_subclass():
    kwargs = read_model_kwargs(
        _cfg('[model]\nenc_ch = 8\nn_erb = 32\n'), DeepFilterNetAENR,
        extra_bases=(DeepFilterNet2,),
        exclude={'n_fft', 'sr', 'n_erb', 'df_bins', 'df_order',
                'mask_lookahead', 'df_lookahead'},
    )
    # enc_ch only exists on the base class, forwarded via **kwargs.
    assert kwargs['enc_ch'] == 8
    # n_erb is DeepFilterNetAENR's own explicit param, not excluded from it.
    assert kwargs['n_erb'] == 32
    # Excluded base params must not leak back in as configurable.
    assert 'sr' not in kwargs and 'n_fft' not in kwargs


def test_checkpoint_contract_roundtrip_and_mismatch():
    grid = SignalGrid(16000, 512, 512, 256)
    kwargs = {'gru_hidden': 64}
    contract = make_checkpoint_contract(
        model_name='Align_CRUSE', task='direct_aec_preserve_noise', grid=grid,
        model_kwargs=kwargs, loss_version='v1',
    )
    require_checkpoint_contract({'contract': contract}, contract)   # must not raise
    with pytest.raises(ValueError, match='contract sr'):
        require_checkpoint_contract({'contract': contract}, {**contract, 'sr': 48000})


def test_checkpoint_contract_ctor_prefix_does_not_collide_with_model_name():
    grid = SignalGrid(16000, 512, 512, 256)
    contract = make_checkpoint_contract(
        model_name='Align_CRUSE', task='direct_aec_preserve_noise', grid=grid,
        model_kwargs={'gru_hidden': 64}, loss_version='v1',
    )
    assert contract['model_name'] == 'Align_CRUSE'
    assert contract['ctor_gru_hidden'] == 64


def test_compressed_spectral_loss_zero_for_identical_spectra():
    spec = torch.complex(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
    assert float(compressed_spectral_loss(spec, spec)) == pytest.approx(0.0, abs=1e-6)


def test_linear_aec_engine_full_length_output_and_reset():
    engine = LinearAecEngine(n_lanes=2, sample_rate=16000, preset='balanced')
    mic = torch.randn(2, 16000) * 0.05
    far = torch.randn(2, 16000) * 0.05
    error, echo_estimate = engine(mic, far, 16000)
    assert error.shape == mic.shape
    assert echo_estimate.shape == mic.shape
    assert torch.isfinite(error).all() and torch.isfinite(echo_estimate).all()
    torch.testing.assert_close(mic - error, echo_estimate)

    engine.arm_reset([True, False])
    error2, _ = engine(mic, far, 16000)
    assert error2.shape == mic.shape


def test_linear_aec_engine_rejects_sample_rate_mismatch():
    engine = LinearAecEngine(n_lanes=1, sample_rate=16000)
    mic = torch.randn(1, 16000) * 0.05
    with pytest.raises(ValueError, match='sample_rate'):
        engine(mic, mic, 48000)
