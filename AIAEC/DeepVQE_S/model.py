"""DeepVQE-S: compact end-to-end AEC + NS + dereverberation model.

Reconstructed from Indenbom et al., INTERSPEECH 2023 / arXiv:2306.03177.
No official training implementation or checkpoint was published, so this file
keeps the paper's stated channel schedules and operators and exposes every
reconstruction choice in the sibling README.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AIAEC.aiaec_common import (
    AecOutput,
    CausalConv2d,
    CausalConvBlock,
    FrameDelayAttention,
    FreqUpsample,
    SignalGrid,
    apply_causal_tf_filter,
    compressed_ri_feature,
    require_complex_btf,
    three_vector_complex,
)


def _half(n: int) -> int:
    return (n + 1) // 2


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, (4, 3), bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.norm(self.conv(x)))


class DeepVQES(nn.Module):
    """The 0.59 M-class DeepVQE-S topology, adapted to project STFT grids."""

    paper_reference = "arXiv:2306.03177"
    task = "end_to_end_aec_res_nr_dereverb"

    def __init__(self, grid: SignalGrid, max_delay_seconds: float = 1.0,
                 similarity_channels: int = 4, gru_hidden: int = 192,
                 compression_exponent: float = 0.3):
        super().__init__()
        self.grid = grid
        self.compression_exponent = compression_exponent
        self.max_delay_frames = grid.delay_frames(max_delay_seconds)

        # DeepVQE-S schedules from the paper. All encoder residual blocks are
        # intentionally absent in the S variant.
        self.mic1 = CausalConvBlock(2, 16, (4, 3), (1, 2))
        self.mic2 = CausalConvBlock(16, 40, (4, 3), (1, 2))
        self.far1 = CausalConvBlock(2, 8, (4, 3), (1, 2))
        self.far2 = CausalConvBlock(8, 24, (4, 3), (1, 2))
        self.align = FrameDelayAttention(
            40, 24, 24, similarity_channels, self.max_delay_frames,
            score_kernel=(5, 3),
        )
        self.mic3 = CausalConvBlock(40 + 24, 56, (4, 3), (1, 2))
        self.mic4 = CausalConvBlock(56, 24, (4, 3), (1, 2))

        f1 = _half(grid.n_freqs)
        f2, f3, f4 = _half(f1), _half(_half(f1)), _half(_half(_half(f1)))
        bottleneck_size = 24 * f4
        self.gru = nn.GRU(bottleneck_size, gru_hidden, batch_first=True)
        self.gru_out = nn.Linear(gru_hidden, bottleneck_size)

        self.skip4 = nn.Conv2d(24, 24, 1)
        self.up3 = FreqUpsample(24, 40)  # first decoder block: no residual
        self.skip3 = nn.Conv2d(56, 40, 1)
        self.res3 = ResidualBlock(40)
        self.up2 = FreqUpsample(40, 32)
        self.skip2 = nn.Conv2d(40, 32, 1)
        self.res2 = ResidualBlock(32)
        self.up1 = FreqUpsample(32, 32)
        self.skip1 = nn.Conv2d(16, 32, 1)
        # Last S decoder block: no residual, BN or activation; 27 channels are
        # 3 time taps * 3 frequency taps * DeepVQE's 3-vector representation.
        self.ccm_up = FreqUpsample(
            32, 27, activation=False, normalization=False,
        )
        self.time_order = 3
        self.freq_radius = 1

    def forward(self, microphone: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(microphone, "microphone")
        require_complex_btf(far_end, "far_end")
        if microphone.shape != far_end.shape:
            raise ValueError("microphone and far_end STFT grids must match")
        if microphone.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        mic_feat = compressed_ri_feature(microphone, self.compression_exponent)
        far_feat = compressed_ri_feature(far_end, self.compression_exponent)
        m1 = self.mic1(mic_feat)
        m2 = self.mic2(m1)
        f1 = self.far1(far_feat)
        f2 = self.far2(f1)
        aligned, delay = self.align(m2, f2)
        m3 = self.mic3(torch.cat((m2, aligned), dim=1))
        m4 = self.mic4(m3)

        b, c, t, f = m4.shape
        x = m4.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.gru(x)
        x = self.gru_out(x).reshape(b, t, c, f).permute(0, 2, 1, 3)

        x = self.up3(x + self.skip4(m4), m3.shape[-1])
        x = self.up2(self.res3(x + self.skip3(m3)), m2.shape[-1])
        x = self.up1(self.res2(x + self.skip2(m2)), m1.shape[-1])
        raw = self.ccm_up(x + self.skip1(m1), self.grid.n_freqs)
        raw = raw.permute(0, 2, 3, 1).reshape(
            b, t, self.grid.n_freqs,
            self.time_order, 2 * self.freq_radius + 1, 3,
        )
        taps = three_vector_complex(raw)
        enhanced = apply_causal_tf_filter(
            microphone, taps, self.time_order, self.freq_radius,
        )
        return AecOutput(
            enhanced=enhanced,
            delay_distribution=delay,
            auxiliary={"ccm_taps": taps},
        )
