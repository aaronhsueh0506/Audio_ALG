"""Align-ULCNet hybrid residual-echo and noise suppressor.

This is the two-stage graph in Shetu et al., EUSIPCO 2025
(``arXiv:2410.13620``).  A frozen linear AEC/Kalman filter precedes the model;
the network consumes its error signal and the unaligned far-end reference.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import Tensor, nn

from AIAEC.aiaec_common import (
    AecOutput,
    CausalConv2d,
    FrameDelayAttention,
    SeparableCausalConv,
    SignalGrid,
    require_complex_btf,
)


def _signed_power(x: Tensor, exponent: float) -> Tensor:
    return x.sign() * x.abs().pow(exponent)


def _component_power_spectrum(spec: Tensor, exponent: float) -> Tuple[Tensor, Tensor, Tensor]:
    """ULCNet's modified power law, applied to real/imag independently."""
    real = _signed_power(spec.real, exponent)
    imag = _signed_power(spec.imag, exponent)
    magnitude = (real.square() + imag.square() + 1e-12).sqrt()
    phase = torch.atan2(imag, real)
    return real, imag, magnitude, phase


class ChannelSampledReorientation(nn.Module):
    """Paper C-SamFR: sample *subbands*, then frequency-stack each set.

    With ``K=257, K_B=2, gamma=5`` the input is padded to 130 two-bin
    subbands.  Channel 0 is ``[bands 0,5,10,...]`` (each band remains two
    contiguous bins), not ``[bins 0,5,10,...]``.
    """

    def __init__(self, n_freqs: int, gamma: int = 5,
                 subband_bins: int = 2):
        super().__init__()
        if min(n_freqs, gamma, subband_bins) <= 0:
            raise ValueError("C-SamFR dimensions must be positive")
        self.n_freqs = n_freqs
        self.gamma = gamma
        self.subband_bins = subband_bins
        groups = math.ceil(n_freqs / (gamma * subband_bins))
        self.subbands_per_set = groups
        self.n_subbands = groups * gamma
        self.padded_freqs = self.n_subbands * subband_bins
        self.width = groups * subband_bins

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] != self.n_freqs:
            raise ValueError(
                f"C-SamFR expects [B,1,T,{self.n_freqs}], got {tuple(x.shape)}"
            )
        if self.padded_freqs > self.n_freqs:
            x = torch.nn.functional.pad(x, (0, self.padded_freqs - self.n_freqs))
        b, _, t, _ = x.shape
        # [B,T,groups,gamma,K_B] -> [B,gamma,T,groups*K_B]
        x = x.reshape(
            b, 1, t, self.subbands_per_set, self.gamma, self.subband_bins,
        )
        return x[:, 0].permute(0, 3, 1, 2, 4).reshape(
            b, self.gamma, t, self.width,
        )

    def inverse(self, x: Tensor) -> Tensor:
        if (x.ndim != 4 or x.shape[1] != self.gamma
                or x.shape[-1] != self.width):
            raise ValueError("C-SamFR inverse shape mismatch")
        b, _, t, _ = x.shape
        x = x.reshape(
            b, self.gamma, t, self.subbands_per_set, self.subband_bins,
        )
        x = x.permute(0, 2, 3, 1, 4).reshape(b, t, self.padded_freqs)
        return x[..., :self.n_freqs]


class StreamEncoder(nn.Module):
    """The two 32-filter separable layers and 2x max-pool in each stream."""

    def __init__(self, gamma: int):
        super().__init__()
        self.conv1 = SeparableCausalConv(gamma, 32, (1, 5), activation="elu")
        self.conv2 = SeparableCausalConv(32, 32, (1, 3), activation="elu")
        self.pool = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.conv2(self.conv1(x)))


class JointConvBlock(nn.Module):
    """The paper's ordinary (not depthwise-separable) joint Conv layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = CausalConv2d(
            in_channels, out_channels, (1, 3), stride=(1, 2), bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class FrequencyGRU(nn.Module):
    """64-unit FGRU followed by the stated 64-filter point-wise Conv."""

    def __init__(self, input_channels: int = 96):
        super().__init__()
        # 32 units in each direction produce the paper's 64-unit output.
        self.gru = nn.GRU(
            input_channels, 32, batch_first=True, bidirectional=True,
        )
        self.pointwise = nn.Conv2d(64, 64, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, f = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        sequence, _ = self.gru(sequence)
        x = sequence.reshape(b, t, f, 64).permute(0, 3, 1, 2)
        return self.pointwise(x)


class SubbandTemporalGRU(nn.Module):
    """One temporal subband block: two causal GRU layers, 128 units."""

    def __init__(self, input_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, 128, num_layers=2, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # A complete frequency subband is flattened into each time step.
        b, c, t, f = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        return self.gru(sequence)[0]


class AlignULCNet(nn.Module):
    paper_reference = "arXiv:2410.13620"
    task = "linear_aec_postfilter_res_nr"

    def __init__(self, grid: SignalGrid, max_delay_seconds: float = 1.0,
                 gamma: int = 5, subband_bins: int = 2,
                 compression_exponent: float = 0.3,
                 max_delay_frames: int | None = None):
        super().__init__()
        if not 0.0 < compression_exponent <= 1.0:
            raise ValueError("compression_exponent must be in (0,1]")
        self.grid = grid
        self.compression_exponent = compression_exponent
        self.reorient = ChannelSampledReorientation(
            grid.n_freqs, gamma, subband_bins,
        )
        self.error_encoder = StreamEncoder(gamma)
        self.far_encoder = StreamEncoder(gamma)
        if max_delay_frames is None:
            # The paper fixes Dmax=64 at its 16 kHz/256-hop grid.
            max_delay_frames = (64 if (grid.sample_rate, grid.n_fft) == (16000, 512)
                                else grid.delay_frames(max_delay_seconds))
        self.max_delay_frames = int(max_delay_frames)
        self.align = FrameDelayAttention(
            32, 32, 32, 32, self.max_delay_frames, score_kernel=(5, 3),
        )
        self.joint1 = JointConvBlock(64, 64)
        self.joint2 = JointConvBlock(64, 96)
        self.fgru = FrequencyGRU(96)

        # Width after pool / two stride-2 joint convolutions.
        width = math.ceil(self.reorient.width / 2)
        width = math.ceil(width / 2)
        width = math.ceil(width / 2)
        split = (math.ceil(width / 2), width // 2)
        if split[1] == 0:
            raise ValueError("Align-ULCNet grid is too narrow for two subbands")
        self.subband_widths = split
        self.subband_grus = nn.ModuleList(
            SubbandTemporalGRU(64 * w) for w in split
        )
        self.mask_fc1 = nn.Linear(2 * 128, grid.n_freqs)
        self.mask_act = nn.PReLU()
        self.mask_fc2 = nn.Linear(grid.n_freqs, grid.n_freqs)

        # ULCNet stage 2: two 32-filter CNN layers and a 2-channel pointwise
        # complex-mask head.  It consumes the intermediate real/imag features.
        self.stage2_conv1 = CausalConv2d(2, 32, (1, 3), bias=False)
        self.stage2_norm1 = nn.BatchNorm2d(32)
        self.stage2_conv2 = CausalConv2d(32, 32, (1, 3), bias=False)
        self.stage2_norm2 = nn.BatchNorm2d(32)
        self.stage2_act = nn.PReLU(32)
        self.complex_mask = nn.Conv2d(32, 2, 1)

    def forward(self, linear_error: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(linear_error, "linear_error")
        require_complex_btf(far_end, "far_end")
        if linear_error.shape != far_end.shape:
            raise ValueError("linear_error and far_end STFT grids must match")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        zr, zi, zmag, zphase = _component_power_spectrum(
            linear_error, self.compression_exponent,
        )
        _yr, _yi, ymag, _yphase = _component_power_spectrum(
            far_end, self.compression_exponent,
        )
        e = self.error_encoder(self.reorient(zmag.unsqueeze(1)))
        f = self.far_encoder(self.reorient(ymag.unsqueeze(1)))
        aligned, delay = self.align(e, f)
        x = self.joint2(self.joint1(torch.cat((e, aligned), dim=1)))
        x = self.fgru(x)

        pieces: List[Tensor] = []
        at = 0
        for width, block in zip(self.subband_widths, self.subband_grus):
            pieces.append(block(x[..., at:at + width]))
            at += width
        if at != x.shape[-1]:
            raise RuntimeError("subband split no longer matches the encoder width")
        features = torch.cat(pieces, dim=-1)
        magnitude_mask = torch.sigmoid(
            self.mask_fc2(self.mask_act(self.mask_fc1(features))),
        )

        intermediate = torch.stack((
            magnitude_mask * torch.cos(zphase),
            magnitude_mask * torch.sin(zphase),
        ), dim=1)
        stage2 = self.stage2_act(self.stage2_norm1(self.stage2_conv1(intermediate)))
        stage2 = self.stage2_act(self.stage2_norm2(self.stage2_conv2(stage2)))
        mask_ri = self.complex_mask(stage2)
        mask = torch.complex(mask_ri[:, 0], mask_ri[:, 1])

        compressed_error = torch.complex(zr, zi)
        compressed_estimate = compressed_error * mask
        enhanced = torch.complex(
            _signed_power(compressed_estimate.real, 1.0 / self.compression_exponent),
            _signed_power(compressed_estimate.imag, 1.0 / self.compression_exponent),
        )
        return AecOutput(
            enhanced=enhanced, mask=mask, delay_distribution=delay,
            auxiliary={
                "magnitude_mask": magnitude_mask,
                "intermediate_ri": intermediate,
            },
        )
