"""DeepVQE-S: compact end-to-end AEC + NS + dereverberation model.

Reconstructed from Indenbom et al., INTERSPEECH 2023 / arXiv:2306.03177.
No official training implementation or checkpoint was published, so this file
keeps the paper's stated channel schedules and operators and exposes every
reconstruction choice in the sibling README.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from AIAEC.aiaec_common import (
    AecOutput,
    CausalConv2d,
    CausalConvBlock,
    FrameDelayAttention,
    FreqUpsample,
    SignalGrid,
    apply_causal_tf_filter,
    compressed_ri_feature,
    fit_frequency,
    require_complex_btf,
    three_vector_complex,
)
from AIAEC.aiaec_streaming import (
    DelayRingCell,
    FrameDelayAttentionCell,
    StreamConv2dCell,
    StreamGRUCell,
    StreamModuleCell,
    assert_streaming_ready,
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


def _block_cell(block: CausalConvBlock) -> StreamModuleCell:
    return StreamModuleCell(
        StreamConv2dCell.from_causal(block.conv),
        [block.norm, block.activation],
    )


class _StreamFreqUpsampleCell:
    """Frame-by-frame twin of :class:`aiaec_common.FreqUpsample`.

    Only the inner causal conv carries time state; the sub-pixel shuffle,
    frequency fit, norm and activation are per-frame and must run in the
    block's offline order (norm/act come after the shuffle).
    """

    def __init__(self, block: FreqUpsample):
        self.block = block
        self.conv_cell = StreamConv2dCell.from_causal(block.conv)

    def reset(self) -> None:
        self.conv_cell.reset()

    def step(self, x: Tensor, target_freq: int) -> Tensor:
        block = self.block
        x = self.conv_cell.step(x)
        b, _, t, f = x.shape
        x = x.reshape(b, block.out_channels, 2, t, f)
        x = x.permute(0, 1, 3, 4, 2).reshape(b, block.out_channels, t, 2 * f)
        x = fit_frequency(x, target_freq)
        return block.act(block.norm(x))

    def state_tensors(self) -> Dict[str, Tensor]:
        return self.conv_cell.state_tensors()


class _StreamResidualCell:
    """Frame-by-frame twin of :class:`ResidualBlock` (same-frame skip)."""

    def __init__(self, block: ResidualBlock):
        self.block = block
        self.conv_cell = StreamConv2dCell.from_causal(block.conv)

    def reset(self) -> None:
        self.conv_cell.reset()

    def step(self, x: Tensor) -> Tensor:
        return x + self.block.act(self.block.norm(self.conv_cell.step(x)))

    def state_tensors(self) -> Dict[str, Tensor]:
        return self.conv_cell.state_tensors()


class DeepVQES(nn.Module):
    """The 0.59 M-class DeepVQE-S topology, adapted to project STFT grids."""

    paper_reference = "arXiv:2306.03177"
    task = "end_to_end_aec_res_nr_dereverb"
    # Every operator is causal with zero left context at reset, so streaming
    # emits each enhanced frame in the invocation that consumed its inputs.
    stream_output_delay = 0

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

    def create_stream_state(self) -> Dict[str, object]:
        assert_streaming_ready(self)
        return {
            "mic1": _block_cell(self.mic1),
            "mic2": _block_cell(self.mic2),
            "far1": _block_cell(self.far1),
            "far2": _block_cell(self.far2),
            "align": FrameDelayAttentionCell(self.align),
            "mic3": _block_cell(self.mic3),
            "mic4": _block_cell(self.mic4),
            "gru": StreamGRUCell(self.gru),
            "up3": _StreamFreqUpsampleCell(self.up3),
            "res3": _StreamResidualCell(self.res3),
            "up2": _StreamFreqUpsampleCell(self.up2),
            "res2": _StreamResidualCell(self.res2),
            "up1": _StreamFreqUpsampleCell(self.up1),
            "ccm_up": _StreamFreqUpsampleCell(self.ccm_up),
            # Raw complex mic spectrum ring for the CCM's (t, t-1, t-2) taps.
            "spec_ring": DelayRingCell(self.time_order),
        }

    def _apply_ccm_frame(self, spec_hist: Tensor, taps: Tensor) -> Tensor:
        """T=1 twin of :func:`aiaec_common.apply_causal_tf_filter`.

        ``spec_hist`` is complex ``[B, time_order, F]`` with slot ``d`` holding
        the raw mic spectrum ``d`` frames back; frequency shifts stay within
        the frame, so only the tap-summation order must match offline.
        """
        frame_taps = taps[:, 0]                       # [B,F,O,K]
        out = torch.zeros_like(spec_hist[:, 0])       # [B,F]
        for dt in range(self.time_order):
            shifted_t = spec_hist[:, dt]
            for j, df in enumerate(range(-self.freq_radius,
                                         self.freq_radius + 1)):
                if df < 0:
                    shifted = F.pad(shifted_t[..., :df], (-df, 0))
                elif df > 0:
                    shifted = F.pad(shifted_t[..., df:], (0, df))
                else:
                    shifted = shifted_t
                out = out + shifted * frame_taps[..., dt, j]
        return out.unsqueeze(1)

    def forward_stream(self, microphone: Tensor, far_end: Tensor,
                       state: Dict[str, object]) -> AecOutput:
        require_complex_btf(microphone, "microphone")
        require_complex_btf(far_end, "far_end")
        if microphone.shape[1] != 1 or far_end.shape[1] != 1:
            raise ValueError("forward_stream consumes exactly one frame (T=1)")
        if microphone.shape != far_end.shape:
            raise ValueError("microphone and far_end STFT grids must match")
        if microphone.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        mic_feat = compressed_ri_feature(microphone, self.compression_exponent)
        far_feat = compressed_ri_feature(far_end, self.compression_exponent)
        m1 = state["mic1"].step(mic_feat)
        m2 = state["mic2"].step(m1)
        f1 = state["far1"].step(far_feat)
        f2 = state["far2"].step(f1)
        aligned, delay = state["align"].step(m2, f2)
        m3 = state["mic3"].step(torch.cat((m2, aligned), dim=1))
        m4 = state["mic4"].step(m3)

        b, c, t, f = m4.shape
        x = m4.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = state["gru"].step(x)
        x = self.gru_out(x).reshape(b, t, c, f).permute(0, 2, 1, 3)

        x = state["up3"].step(x + self.skip4(m4), m3.shape[-1])
        x = state["up2"].step(state["res3"].step(x + self.skip3(m3)),
                              m2.shape[-1])
        x = state["up1"].step(state["res2"].step(x + self.skip2(m2)),
                              m1.shape[-1])
        raw = state["ccm_up"].step(x + self.skip1(m1), self.grid.n_freqs)
        raw = raw.permute(0, 2, 3, 1).reshape(
            b, t, self.grid.n_freqs,
            self.time_order, 2 * self.freq_radius + 1, 3,
        )
        taps = three_vector_complex(raw)
        spec_hist = state["spec_ring"].step(microphone)[:, 0, 0]
        enhanced = self._apply_ccm_frame(spec_hist, taps)
        return AecOutput(
            enhanced=enhanced,
            delay_distribution=delay,
            auxiliary={"ccm_taps": taps},
        )
