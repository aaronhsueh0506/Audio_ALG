from types import SimpleNamespace

import numpy as np
import pytest

from lib.aec.python.modules.config import AecConfig
from lib.nr.core.frame_processor import FrameProcessor
from lib.nr.core.signal_grid import (
    _SIXTEEN_MS_HOP_SECONDS,
    resolve_signal_grid,
    retime_ema_alpha,
    retime_frame_count,
)
from lib.nr.denoisers import (
    MmseLsaDenoiser,
    PmmseDenoiser,
    SpectralSubtractionDenoiser,
    SppMmseDenoiser,
    WienerDenoiser,
)
from pipelines import aec_nr_pipeline as pipeline_module
from pipelines.aec_nr_pipeline import _build_denoiser, _project_grid


@pytest.mark.parametrize(
    "sample_rate,fft_size,expected",
    [
        # 16kHz default flipped 512/256 (16ms hop) -> 256/128 (8ms hop) on
        # 2026-08-02/03 (NR CHANGELOG [4.5.0]); 512/256 remains a supported,
        # explicit alternate grid (see the fft_size=512 case below).
        (16000, None, (256, 128, 256)),
        (16000, 512, (512, 256, 512)),
        (48000, None, (1024, 512, 1024)),
    ],
)
def test_nr_project_grid(sample_rate, fft_size, expected):
    assert resolve_signal_grid(sample_rate, fft_size) == expected


@pytest.mark.parametrize(
    "sample_rate,fft_size,expected",
    [
        # 16kHz default flipped 512/256 (16ms hop) -> 256/128 (8ms hop) on
        # 2026-08-02/03 (NR CHANGELOG [4.5.0]); 512/256 remains a supported,
        # explicit alternate grid (see the fft_size=512 case below).
        (16000, None, (256, 128, 256)),
        (16000, 512, (512, 256, 512)),
        (48000, None, (1024, 512, 1024)),
    ],
)
def test_integrated_pipeline_grid(sample_rate, fft_size, expected):
    assert _project_grid(sample_rate, fft_size) == expected


def test_integrated_pipeline_builds_explicit_low_latency_16k_grid():
    denoiser = _build_denoiser(16000, fft_size=256)
    assert (
        denoiser.processor.frame_size,
        denoiser.processor.frame_shift,
        denoiser.processor.fft_size,
    ) == (256, 128, 256)


@pytest.mark.parametrize(
    "sample_rate,fft_size",
    [(16000, 1024), (48000, 256), (48000, 512)],
)
def test_integrated_pipeline_rejects_cross_rate_grids(sample_rate, fft_size):
    with pytest.raises(ValueError, match="unsupported fft_size"):
        _project_grid(sample_rate, fft_size)


def test_python_aec_and_nr_reject_old_padded_grid():
    with pytest.raises(ValueError, match="no-padding"):
        AecConfig(sample_rate=16000, frame_size=320, hop_size=160)
    with pytest.raises(ValueError, match="no-padding grid"):
        FrameProcessor(
            sample_rate=16000,
            frame_size=320,
            frame_shift=160,
            fft_size=512,
        )


@pytest.mark.parametrize(
    "denoiser_type",
    [
        SpectralSubtractionDenoiser,
        WienerDenoiser,
        SppMmseDenoiser,
        MmseLsaDenoiser,
        PmmseDenoiser,
    ],
)
def test_every_traditional_nr_model_auto_selects_48k_no_padding_grid(denoiser_type):
    denoiser = denoiser_type(sample_rate=48000)
    assert denoiser.processor.frame_size == 1024
    assert denoiser.processor.frame_shift == 512
    assert denoiser.processor.fft_size == 1024


def test_low_latency_16k_frame_has_no_transform_padding():
    processor = FrameProcessor(
        sample_rate=16000,
        frame_size=256,
        frame_shift=128,
        fft_size=256,
    )
    magnitude, phase, spectrum = processor.process_frame(np.ones(256, np.float32))
    assert magnitude.shape == phase.shape == spectrum.shape == (129,)


def test_aec_only_time_output_matches_standalone_limiter_path(monkeypatch):
    class FakeAec:
        hop_size = 2

        def __init__(self, config):
            self.config = config

        def process(self, mic, ref):
            context = SimpleNamespace(
                formed_output=np.asarray(mic, dtype=np.float32) + 10.0
            )
            return np.asarray(mic, dtype=np.float32) + 1.0, context

        def get_erle(self):
            return 0.0

    monkeypatch.setattr(pipeline_module, "AEC", FakeAec)
    mic = np.arange(4, dtype=np.float32)
    ref = np.zeros_like(mic)

    formed, contexts = pipeline_module.run_aec_linear(
        mic, ref, SimpleNamespace(enable_res=True, return_res_context=False)
    )
    standalone, _ = pipeline_module.run_aec_linear(
        mic, ref, SimpleNamespace(enable_res=True, return_res_context=False),
        standalone_time_output=True,
    )

    np.testing.assert_array_equal(formed, mic + 10.0)
    np.testing.assert_array_equal(standalone, mic + 1.0)
    assert len(contexts) == 2


@pytest.mark.parametrize(
    "sample_rate,hop_size,expected_frames",
    [
        (8000, 128, 20),
        (16000, 128, 40),
        (16000, 256, 20),
        (48000, 512, 30),
    ],
)
def test_nr_temporal_constants_preserve_wallclock_duration(
    sample_rate, hop_size, expected_frames
):
    # 32 legacy frames represented 320 ms on the former 10-ms hop.
    assert retime_frame_count(32, sample_rate, hop_size) == expected_frames
    alpha = retime_ema_alpha(0.95, sample_rate, hop_size)
    updates_per_second = sample_rate / hop_size
    assert alpha ** updates_per_second == pytest.approx(0.95 ** 100, rel=1e-12)


@pytest.mark.parametrize(
    "sample_rate,fft_size",
    [(8000, 256), (16000, 256), (16000, 512), (48000, 1024)],
)
def test_mmse_lsa_outer_model_retimes_all_frame_domain_state(sample_rate, fft_size):
    denoiser = MmseLsaDenoiser(
        sample_rate=sample_rate,
        fft_size=fft_size,
        noise_method="mcra",
        alpha_s=0.95,
        alpha_d=0.7,
        alpha_p=0.2,
        alpha_xi=0.92,
        alpha_g=0.88,
        alpha_attack=0.3,
        L=32,
        num_init_frames=20,
        scene_change_min_frames=5,
    )
    hop = denoiser.processor.frame_shift
    # L=32 is documented in config/v3_2_config.yaml as authored directly against
    # a 16ms hop ("32 幀 x 16ms/hop = 512ms"); v3_2_mmse_lsa.py's L retime call
    # applies that basis unconditionally (NR CHANGELOG [4.5.0], 2026-08-03) --
    # unlike num_init_frames/scene_change_min_frames below, which stay on the
    # generic 10ms reference (no equivalent hop-basis evidence for them).
    assert denoiser.noise_estimator.L == retime_frame_count(
        32, sample_rate, hop, authored_hop_seconds=_SIXTEEN_MS_HOP_SECONDS
    )
    assert denoiser.noise_estimator.num_init_frames == retime_frame_count(
        20, sample_rate, hop
    )
    assert denoiser.noise_estimator.scene_change_min_frames == retime_frame_count(
        5, sample_rate, hop
    )
    assert denoiser.noise_estimator.alpha_s == pytest.approx(
        retime_ema_alpha(0.95, sample_rate, hop)
    )
    # alpha_xi is 16ms-native unconditionally, regardless of strength (see
    # v3_2_mmse_lsa.py's __init__ docstring; same NR CHANGELOG [4.5.0] fix as L
    # above) -- unlike alpha_g just below, which stays on the 10ms reference
    # here because this denoiser is constructed at the default strength='balanced'.
    assert denoiser.spp_estimator.alpha == pytest.approx(
        retime_ema_alpha(0.92, sample_rate, hop, authored_hop_seconds=_SIXTEEN_MS_HOP_SECONDS)
    )
    assert denoiser.gain_calculator.alpha_g == pytest.approx(
        retime_ema_alpha(0.88, sample_rate, hop)
    )
