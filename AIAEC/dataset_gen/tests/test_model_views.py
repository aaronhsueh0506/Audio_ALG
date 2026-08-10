import configparser
import pathlib

import pytest
import torch

from AIAEC.dataset_gen import (
    MODEL_TASKS, AecGrid, AecStems, PACKED_STEM_ORDER, STEM_ORDER,
    build_model_view,
    build_spectral_model_view,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.DeepFilterNet_AENR import DeepFilterNetAENR
from AIAEC.DeepVQE_S import DeepVQES


def _stems():
    data = torch.zeros(len(PACKED_STEM_ORDER), 32)
    for i in range(len(PACKED_STEM_ORDER)):
        data[i].fill_(float(i + 1))
    return AecStems(data, PACKED_STEM_ORDER)


def test_dataset_gen_is_the_single_public_and_implementation_package():
    from AIAEC import dataset_gen

    assert dataset_gen.STEM_ORDER == STEM_ORDER
    assert dataset_gen.build_model_view is build_model_view
    assert build_model_view.__module__ == "AIAEC.dataset_gen.model_views"
    assert AecStems.__module__ == "AIAEC.dataset_gen.aec_features"


def test_all_six_candidates_have_one_dataset_contract():
    assert set(MODEL_TASKS) == {
        "Align_CRUSE", "Align_ULCNet", "GTCRN_AENR",
        "DeepFilterNet_AENR", "DeepVQE_S", "CAGCRN",
    }


def test_align_cruse_uses_early_near_target_for_joint_dereverb_task():
    stems = _stems()
    view = build_model_view(stems, "Align_CRUSE", 16000)
    assert torch.equal(view.target, stems.near_target)
    assert set(view.inputs) == {"microphone", "far_end"}


def test_deepvqe_uses_early_near_target_for_published_dereverb_task():
    stems = _stems()
    view = build_model_view(stems, "DeepVQE_S", 16000)
    assert torch.equal(view.target, stems.near_target)
    assert set(view.inputs) == {"microphone", "far_end"}


def test_cagcrn_uses_the_common_early_clean_target():
    stems = _stems()
    view = build_model_view(stems, "CAGCRN", 16000)
    assert torch.equal(view.target, stems.near_target)
    assert set(view.inputs) == {"microphone", "far_end"}


def test_residual_view_uses_materialized_linear_error_stem():
    stems = _stems()
    view = build_model_view(stems, "Align_ULCNet", 16000)
    assert torch.equal(view.inputs["linear_error"], stems.linear_error)
    assert torch.equal(view.echo_estimate, stems.mic_postclip - stems.linear_error)
    assert torch.equal(view.target, stems.near_target)


def test_grid_rejects_hidden_zero_padding():
    with pytest.raises(ValueError, match="forbids hidden FFT zero-padding"):
        AecGrid(sr=16000, n_fft=512, win_len=320, hop_len=160)


def test_spectral_view_is_directly_accepted_by_public_model_forward():
    stems = AecStems(
        torch.randn(len(PACKED_STEM_ORDER), 4096), PACKED_STEM_ORDER
    )
    waveform = build_model_view(stems, "DeepVQE_S", 16000)
    grid = AecGrid(16000, 512, 512, 256)
    spectral = build_spectral_model_view(waveform, grid)
    model = DeepVQES(SignalGrid(16000, 512, 512, 256)).eval()
    with torch.no_grad():
        output = model(**spectral.inputs)
    assert output.enhanced.shape == spectral.target.shape


def test_dfn_spectral_view_builds_independent_feature_states_and_runs():
    torch.manual_seed(3)
    stems = AecStems(
        torch.randn(len(PACKED_STEM_ORDER), 4096), PACKED_STEM_ORDER
    )

    waveform = build_model_view(stems, "DeepFilterNet_AENR", 16000)
    grid = AecGrid(16000, 512, 512, 256)
    model = DeepFilterNetAENR(
        SignalGrid(16000, 512, 512, 256), enc_ch=8, emb_size=32,
        df_hidden=32, lin_groups=4, enc_lin_groups=4,
    ).eval()
    cfg = configparser.ConfigParser()
    root = pathlib.Path(__file__).resolve().parents[3]
    cfg.read(root / "AINR" / "DeepFilterNet2" / "config.ini")
    from AINR.DeepFilterNet2.train import read_feature_config
    feature_cfg = read_feature_config(cfg, 16000, 256)
    spectral = build_spectral_model_view(
        waveform, grid, dfn_model=model, dfn_feature_config=feature_cfg,
    )
    assert set(spectral.inputs) == {
        "linear_error", "error_erb", "error_spec", "far_erb", "far_spec",
    }
    assert spectral.feature_state["error"] is not spectral.feature_state["far"]
    with torch.no_grad():
        output = model(**spectral.inputs)
    assert output.enhanced.shape == spectral.target.shape


def test_dfn_spectral_view_rejects_shared_ema_state():
    stems = _stems()
    view = build_model_view(stems, "DeepFilterNet_AENR", 16000)
    shared = {"erb": None, "spec": None}
    with pytest.raises(ValueError, match="must not be shared"):
        build_spectral_model_view(
            view, AecGrid(16000, 512, 512, 256),
            dfn_model=object(), dfn_feature_config={},
            dfn_feature_state={"error": shared, "far": shared},
        )
