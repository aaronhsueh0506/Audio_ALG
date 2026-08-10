"""CAGCRN backup model for joint AEC and noise suppression.

The graph follows Wang et al., INTERSPEECH 2025-608.  The publication has no
source release and leaves the ERB count, decoder channels, mask block and the
gradient path through ``floor(D)`` unspecified.  Those bounded reconstruction
choices are called out in the README; the published placement and data flow of
CATA, the two TF-GRUs, TFAG and the mirrored decoder are retained here.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from AIAEC.aiaec_common import (
    AecOutput,
    CausalConv2d,
    SeparableCausalConv,
    SignalGrid,
    causal_delay_stack,
    complex_mask,
    fit_frequency,
    mag_ri_feature,
    require_complex_btf,
)


class ErbBandCodec(nn.Module):
    """Retain linear bins through 2 kHz and ERB-compress 2--8 kHz."""

    def __init__(self, grid: SignalGrid, high_bands: int = 32,
                 split_hz: float = 2000.0):
        super().__init__()
        bin_hz = grid.sample_rate / grid.n_fft
        self.low_bins = min(grid.n_freqs, int(math.floor(split_hz / bin_hz)) + 1)
        high_bins = grid.n_freqs - self.low_bins
        if high_bins <= 0:
            raise ValueError("ERB high-frequency region is empty")
        high_bands = min(high_bands, high_bins)
        freqs = torch.linspace(split_hz, grid.sample_rate / 2, high_bins)
        erb = 21.4 * torch.log10(1.0 + 0.00437 * freqs)
        centers = torch.linspace(erb[0], erb[-1], high_bands)
        if high_bands == 1:
            weights = torch.ones(1, high_bins)
        else:
            step = (centers[-1] - centers[0]) / (high_bands - 1)
            weights = (1.0 - (erb[None] - centers[:, None]).abs()
                       / step.clamp_min(1e-6)).clamp_min(0.0)
        merge = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        split = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("merge_matrix", merge)
        self.register_buffer("split_matrix", split)
        self.compressed_bins = self.low_bins + high_bands

    def merge(self, x: Tensor) -> Tensor:
        low = x[..., :self.low_bins]
        high = x[..., self.low_bins:].matmul(self.merge_matrix.t())
        return torch.cat((low, high), dim=-1)

    def split(self, x: Tensor) -> Tensor:
        low = x[..., :self.low_bins]
        high = x[..., self.low_bins:].matmul(self.split_matrix)
        return torch.cat((low, high), dim=-1)


class EncoderBlock(nn.Module):
    """Conv2D -> P-Conv -> dilated group Conv -> P-Conv + residual."""

    def __init__(self, in_channels: int, out_channels: int, stride_f: int,
                 time_dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, (1, 5),
            stride=(1, stride_f), padding=(0, 2), bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.point1 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.group = CausalConv2d(
            out_channels, out_channels, (3, 3),
            dilation=(time_dilation, 1), groups=out_channels, bias=False,
        )
        self.point2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.residual = nn.Conv2d(
            in_channels, out_channels, 1, stride=(1, stride_f), bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        x = self.act(self.norm(self.conv(x)))
        x = self.point2(self.group(self.point1(x)))
        return x + residual


class CrossAttentionTemporalAlignment(nn.Module):
    """CATA after the first encoder block, as drawn in Figure 1(a,c).

    The paper's integer ``floor(D)`` window cannot receive ordinary gradients.
    A sigmoid boundary over a fixed maximum buffer preserves its intended
    learnable search span while keeping the graph differentiable.
    """

    def __init__(self, input_channels: int, hidden_channels: int,
                 max_delay_frames: int):
        super().__init__()
        self.max_delay_frames = max_delay_frames
        self.mic_query = SeparableCausalConv(
            input_channels, hidden_channels, (3, 3), activation="prelu",
        )
        self.mic_key = SeparableCausalConv(
            hidden_channels, hidden_channels, (3, 3), activation="prelu",
        )
        self.mic_value = nn.Conv2d(input_channels, hidden_channels, 1)
        self.ref_kv = SeparableCausalConv(
            input_channels, hidden_channels, (3, 3), activation="prelu",
        )
        self.fuse = nn.Conv2d(2 * hidden_channels, hidden_channels, 1)
        self.raw_window = nn.Parameter(torch.tensor(0.0))
        self.window_temperature = 1.0

    def forward(self, mic: Tensor, far: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if mic.shape != far.shape:
            raise ValueError("CATA shallow mic/reference features must match")
        q = self.mic_query(mic)
        k_mic = self.mic_key(q)
        # Figure 1(c)'s microphone self-attention branch.
        mic_distribution = torch.softmax(q * k_mic, dim=-1)
        y_mic = self.mic_value(mic) * mic_distribution

        kv = self.ref_kv(far)
        delayed = causal_delay_stack(kv, self.max_delay_frames)
        logits = q.unsqueeze(3) * delayed
        boundary = 1.0 + (self.max_delay_frames - 1.0) * torch.sigmoid(
            self.raw_window,
        )
        delay = torch.arange(
            self.max_delay_frames, device=mic.device, dtype=mic.dtype,
        )
        gate = torch.sigmoid(
            (boundary - delay) / self.window_temperature,
        ).clamp_min(1e-6)
        logits = logits + gate.log()[None, None, None, :, None]
        distribution = torch.softmax(logits, dim=3)
        y_ref = (distribution * delayed).sum(dim=3)
        fused = self.fuse(torch.cat((y_mic, y_ref), dim=1))
        # Public diagnostic is one delay distribution per frame.  The complete
        # channel/frequency map remains available to callers as an auxiliary.
        delay_distribution = distribution.mean(dim=(1, 4))
        return fused, delay_distribution, distribution


class TFAG(nn.Module):
    """Time-frequency adaptive gate from Figure 1(e)."""

    def __init__(self, channels: int = 24, hidden: int = 12):
        super().__init__()
        self.pre = nn.Conv2d(channels, hidden, 1)
        self.freq1 = CausalConv2d(hidden, hidden, (1, 3), groups=hidden)
        self.freq2 = CausalConv2d(
            hidden, hidden, (1, 3), dilation=(1, 2), groups=hidden,
        )
        self.time1 = CausalConv2d(hidden, hidden, (5, 1), groups=hidden)
        self.time2 = CausalConv2d(
            hidden, hidden, (5, 1), dilation=(2, 1), groups=hidden,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(4 * hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.PReLU(channels),
        )

    def forward(self, far_aligned: Tensor, mic: Tensor) -> Tensor:
        x = self.pre(far_aligned + mic)
        scales = torch.cat((
            self.freq1(x), self.freq2(x), self.time1(x), self.time2(x),
        ), dim=1)
        gate = self.gate(scales)
        return self.fuse(far_aligned * gate + mic * (1.0 - gate))


class TFGRU(nn.Module):
    """Frequency BiGRU then causal time GRU, with both residual additions."""

    def __init__(self, channels: int = 24):
        super().__init__()
        self.freq_gru = nn.GRU(
            channels, channels // 2, batch_first=True, bidirectional=True,
        )
        self.freq_fc = nn.Linear(channels, channels)
        self.freq_ln = nn.LayerNorm(channels)
        self.time_gru = nn.GRU(channels, channels, batch_first=True)
        self.time_fc = nn.Linear(channels, channels)
        self.time_ln = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, f = x.shape
        residual = x.permute(0, 2, 3, 1)
        z = residual.reshape(b * t, f, c)
        z, _ = self.freq_gru(z)
        z = self.freq_ln(self.freq_fc(z)).reshape(b, t, f, c)
        z = z + residual
        freq_residual = z
        z = z.permute(0, 2, 1, 3).reshape(b * f, t, c)
        z, _ = self.time_gru(z)
        z = self.time_ln(self.time_fc(z)).reshape(b, f, t, c)
        z = z + freq_residual.permute(0, 2, 1, 3)
        return z.permute(0, 3, 2, 1)


class DecoderBlock(nn.Module):
    """Mirrored encoder block; only the first Conv becomes frequency deconv."""

    def __init__(self, in_channels: int, out_channels: int, upsample: bool,
                 time_dilation: int):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, (1, 5),
                stride=(1, 2), padding=(0, 2), bias=False,
            )
            self.residual = nn.ConvTranspose2d(
                in_channels, out_channels, 1, stride=(1, 2), bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, (1, 5), padding=(0, 2), bias=False,
            )
            self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.point1 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.group = CausalConv2d(
            out_channels, out_channels, (3, 3),
            dilation=(time_dilation, 1), groups=out_channels, bias=False,
        )
        self.point2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def _first(self, module: nn.Module, x: Tensor, target_freq: int) -> Tensor:
        if not self.upsample:
            return module(x)
        return module(
            x, output_size=(x.shape[0], module.out_channels, x.shape[2], target_freq),
        )

    def forward(self, x: Tensor, target_freq: int) -> Tensor:
        residual = self._first(self.residual, x, target_freq)
        x = self.act(self.norm(self._first(self.conv, x, target_freq)))
        x = self.point2(self.group(self.point1(x)))
        return x + residual


class CAGCRN(nn.Module):
    paper_reference = "INTERSPEECH 2025-608"
    task = "end_to_end_aec_res_nr_dereverb"

    def __init__(self, grid: SignalGrid, high_erb_bands: int = 32,
                 max_delay_seconds: float = 1.0):
        super().__init__()
        self.grid = grid
        self.erb = ErbBandCodec(grid, high_erb_bands)
        self.max_delay_frames = grid.delay_frames(max_delay_seconds)

        mic_channels = (12, 12, 12, 24)
        far_channels = (12, 24, 24, 24)
        dilations = (1, 1, 2, 4)
        strides = (2, 2, 1, 1)
        self.mic_encoder = nn.ModuleList()
        self.far_encoder = nn.ModuleList()
        in_m = in_f = 3
        for cm, cf, dilation, stride in zip(
                mic_channels, far_channels, dilations, strides):
            self.mic_encoder.append(EncoderBlock(in_m, cm, stride, dilation))
            self.far_encoder.append(EncoderBlock(in_f, cf, stride, dilation))
            in_m, in_f = cm, cf
            # CATA changes the first far block's 12 channels to 24 before block2.
            if len(self.far_encoder) == 1:
                in_f = 24

        self.cata = CrossAttentionTemporalAlignment(
            12, 24, self.max_delay_frames,
        )
        self.mic_tfgru = TFGRU(24)
        self.far_tfgru = TFGRU(24)
        self.tfag = TFAG(24, 12)

        current_channels = (24, 24, 12, 12)
        output_channels = (24, 12, 12, 12)
        upsample = (False, False, True, True)
        decode_dilations = (4, 2, 1, 1)
        mic_skip_channels = tuple(reversed(mic_channels))
        # The shallow reference skip is the 24-channel CATA output, not the
        # 12-channel output of reference encoder block 1 before alignment.
        far_skip_channels = (24, 24, 24, 24)
        self.skip_mic = nn.ModuleList()
        self.skip_far = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for ci, co, up, dilation, cm, cf in zip(
                current_channels, output_channels, upsample,
                decode_dilations, mic_skip_channels, far_skip_channels):
            self.skip_mic.append(nn.Conv2d(cm, ci, 1, bias=False))
            self.skip_far.append(nn.Conv2d(cf, ci, 1, bias=False))
            self.decoder.append(DecoderBlock(ci, co, up, dilation))
        # The paper names but does not define Mask; bounded CRM is explicit.
        self.mask = nn.Sequential(nn.Conv2d(12, 2, 1), nn.Tanh())

    def forward(self, microphone: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(microphone, "microphone")
        require_complex_btf(far_end, "far_end")
        if microphone.shape != far_end.shape:
            raise ValueError("microphone and far_end STFT grids must match")
        if microphone.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        mic = self.erb.merge(mag_ri_feature(microphone))
        far = self.erb.merge(mag_ri_feature(far_end))
        mic_skips: List[Tensor] = []
        far_skips: List[Tensor] = []

        mic = self.mic_encoder[0](mic)
        far = self.far_encoder[0](far)
        mic_skips.append(mic)
        # Published placement: CATA is between reference encoder blocks 1/2.
        far, delay, full_attention = self.cata(mic, far)
        far_skips.append(far)
        for index in range(1, 4):
            mic = self.mic_encoder[index](mic)
            far = self.far_encoder[index](far)
            mic_skips.append(mic)
            far_skips.append(far)

        mic = self.mic_tfgru(mic)
        far = self.far_tfgru(far)
        x = self.tfag(far, mic)
        for index, block in enumerate(self.decoder):
            mic_skip = mic_skips[-1 - index]
            far_skip = far_skips[-1 - index]
            x = (x + fit_frequency(self.skip_mic[index](mic_skip), x.shape[-1])
                 + fit_frequency(self.skip_far[index](far_skip), x.shape[-1]))
            target = (mic_skips[-2 - index].shape[-1] if index < 3
                      else self.erb.compressed_bins)
            x = block(x, target)

        mask_erb = self.mask(x)
        mask_full = fit_frequency(self.erb.split(mask_erb), self.grid.n_freqs)
        enhanced = complex_mask(mask_full, microphone)
        return AecOutput(
            enhanced=enhanced, mask=mask_full,
            delay_distribution=delay,
            auxiliary={"erb_mask": mask_erb,
                       "cata_attention": full_attention},
        )
