"""Shared, model-independent tensor contracts for the AIAEC candidates.

The helpers here are deliberately small.  They implement shape handling,
causal padding, delay buffers and fixed reconstruction arithmetic; model
topologies remain in their paper-named directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


@dataclass(frozen=True)
class SignalGrid:
    sample_rate: int
    n_fft: int
    win_len: int
    hop_len: int

    def __post_init__(self) -> None:
        if self.sample_rate not in (16000, 48000):
            raise ValueError("AIAEC supports sample_rate 16000 or 48000")
        if min(self.n_fft, self.win_len, self.hop_len) <= 0:
            raise ValueError("FFT/window/hop must be positive")
        if self.n_fft & (self.n_fft - 1):
            raise ValueError("n_fft must be a power of two")
        if self.win_len != self.n_fft:
            raise ValueError("zero-padding is disabled: win_len must equal n_fft")
        if self.hop_len * 2 != self.win_len:
            raise ValueError("AIAEC model grids require 50% overlap")

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_len

    def delay_frames(self, seconds: float) -> int:
        # This is a maximum supported delay, not a nearest-duration estimate.
        # Rounding 62.5 frames down to 62 silently loses the last 8 ms of a
        # requested one-second search range on the 16 kHz/256-hop grid.
        return max(1, int(math.ceil(seconds * self.frame_rate)))


@dataclass
class AecOutput:
    enhanced: Tensor
    mask: Optional[Tensor] = None
    echo_estimate: Optional[Tensor] = None
    delay_distribution: Optional[Tensor] = None
    auxiliary: Dict[str, Tensor] = field(default_factory=dict)


def require_complex_btf(x: Tensor, name: str) -> None:
    if not torch.is_complex(x) or x.ndim != 3:
        raise ValueError(f"{name} must be complex [B,T,F], got {tuple(x.shape)} {x.dtype}")


def safe_abs(x: Tensor, eps: float = 1e-12) -> Tensor:
    return (x.real.square() + x.imag.square() + eps).sqrt()


def log_power_feature(x: Tensor, floor: float = 1e-12) -> Tensor:
    require_complex_btf(x, "spectrum")
    return (x.real.square() + x.imag.square()).clamp_min(floor).log().unsqueeze(1)


def compressed_ri_feature(x: Tensor, exponent: float = 0.3) -> Tensor:
    """Power-law complex feature as ``[B,2,T,F]``."""
    require_complex_btf(x, "spectrum")
    mag = safe_abs(x)
    scale = mag.clamp_min(1e-12).pow(exponent - 1.0)
    return torch.stack((x.real * scale, x.imag * scale), dim=1)


def mag_ri_feature(x: Tensor) -> Tensor:
    require_complex_btf(x, "spectrum")
    return torch.stack((safe_abs(x), x.real, x.imag), dim=1)


def complex_mask(mask_ri: Tensor, spec: Tensor) -> Tensor:
    if mask_ri.ndim != 4 or mask_ri.shape[1] != 2:
        raise ValueError("complex mask must be [B,2,T,F]")
    require_complex_btf(spec, "spec")
    mask = torch.complex(mask_ri[:, 0], mask_ri[:, 1])
    return mask * spec


def fit_frequency(x: Tensor, target: int) -> Tensor:
    if x.shape[-1] > target:
        return x[..., :target]
    if x.shape[-1] < target:
        return F.pad(x, (0, target - x.shape[-1]))
    return x


class CausalConv2d(nn.Module):
    """Conv2d with left-only time padding and symmetric frequency padding."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1),
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.kt, self.kf = kernel_size
        self.dt, self.df = dilation
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        time_left = (self.kt - 1) * self.dt
        freq_total = (self.kf - 1) * self.df
        freq_left = freq_total // 2
        freq_right = freq_total - freq_left
        return self.conv(F.pad(x, (freq_left, freq_right, time_left, 0)))


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(4, 3), stride=(1, 2),
                 dilation=(1, 1), groups: int = 1,
                 activation: str = "elu"):
        super().__init__()
        self.conv = CausalConv2d(
            in_channels, out_channels, kernel_size, stride, dilation, groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "prelu":
            self.activation = nn.PReLU(out_channels)
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class SeparableCausalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1),
                 activation: str = "prelu"):
        super().__init__()
        self.depth = CausalConv2d(
            in_channels, in_channels, kernel_size, stride, dilation,
            groups=in_channels, bias=False,
        )
        self.point = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = (nn.PReLU(out_channels) if activation == "prelu"
                    else nn.ELU() if activation == "elu" else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.point(self.depth(x))))


class FreqUpsample(nn.Module):
    """Causal convolution followed by a frequency-only sub-pixel shuffle."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(4, 3), activation: bool = True,
                 normalization: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.conv = CausalConv2d(
            in_channels, 2 * out_channels, kernel_size, bias=False,
        )
        self.norm = (nn.BatchNorm2d(out_channels) if normalization
                     else nn.Identity())
        self.act = nn.ELU() if activation else nn.Identity()

    def forward(self, x: Tensor, target_freq: Optional[int] = None) -> Tensor:
        x = self.conv(x)
        b, twice_c, t, f = x.shape
        x = x.reshape(b, self.out_channels, 2, t, f)
        x = x.permute(0, 1, 3, 4, 2).reshape(b, self.out_channels, t, 2 * f)
        if target_freq is not None:
            x = fit_frequency(x, target_freq)
        return self.act(self.norm(x))


def causal_delay_stack(x: Tensor, delays: int) -> Tensor:
    """Return ``[..., T, D, F]`` where slot d contains x[t-d]."""
    if x.ndim != 4:
        raise ValueError("delay-stack input must be [B,C,T,F]")
    if delays <= 0:
        raise ValueError("delays must be positive")
    padded = F.pad(x, (0, 0, delays - 1, 0))
    # unfold appends the window dimension: [B,C,T,F,D].  Reverse so d=0 is now.
    return padded.unfold(2, delays, 1).flip(-1).permute(0, 1, 2, 4, 3)


class FrameDelayAttention(nn.Module):
    """Per-frame causal cross-attention used by DeepVQE/Align-ULCNet."""

    def __init__(self, mic_channels: int, far_channels: int,
                 value_channels: int, similarity_channels: int,
                 max_delay_frames: int, score_kernel=(5, 3)):
        super().__init__()
        self.max_delay_frames = max_delay_frames
        self.query = nn.Conv2d(mic_channels, similarity_channels, 1, bias=False)
        self.key = nn.Conv2d(far_channels, similarity_channels, 1, bias=False)
        self.value = nn.Conv2d(far_channels, value_channels, 1, bias=False)
        self.score = CausalConv2d(
            similarity_channels, 1, score_kernel, bias=True,
        )

    def forward(self, mic: Tensor, far: Tensor):
        if mic.shape[2:] != far.shape[2:]:
            raise ValueError("attention feature grids must match")
        q = self.query(mic)
        k_delayed = causal_delay_stack(self.key(far), self.max_delay_frames)
        # [B,H,T,D], dot product over frequency.
        logits = (q.unsqueeze(3) * k_delayed).sum(dim=-1)
        logits = self.score(logits).squeeze(1)
        distribution = torch.softmax(logits, dim=-1)
        v_delayed = causal_delay_stack(self.value(far), self.max_delay_frames)
        aligned = (v_delayed * distribution[:, None, :, :, None]).sum(dim=3)
        return aligned, distribution


class GlobalDelayAttention(nn.Module):
    """The utterance-level delay distribution described by Align-CRUSE.

    The paper contracts the query/key product over both time and projection
    axes and therefore emits one ``D`` vector per utterance.  ``causal_running``
    is kept as an explicit deployment option; it must not be mistaken for the
    paper graph or used to load a paper-compatible checkpoint.
    """

    def __init__(self, mic_channels: int, far_channels: int,
                 mic_freq: int, far_freq: int, value_channels: int,
                 projection_size: int, max_delay_frames: int,
                 mode: str = "paper_global"):
        super().__init__()
        if mode not in ("paper_global", "causal_running"):
            raise ValueError("mode must be 'paper_global' or 'causal_running'")
        if value_channels != far_channels:
            raise ValueError(
                "Align-CRUSE aligns the far feature maps directly; "
                "value_channels must equal far_channels"
            )
        self.mode = mode
        self.max_delay_frames = max_delay_frames
        self.mic_pool = nn.MaxPool2d((1, 4), stride=(1, 4))
        self.far_pool = nn.MaxPool2d((1, 4), stride=(1, 4))
        mic_width = mic_freq // 4
        far_width = far_freq // 4
        if mic_width <= 0 or far_width <= 0:
            raise ValueError("alignment feature width must be at least four bins")
        self.query = nn.Linear(mic_channels * mic_width, projection_size)
        self.key = nn.Linear(far_channels * far_width, projection_size)

    def forward(self, mic: Tensor, far: Tensor):
        if mic.shape[0] != far.shape[0] or mic.shape[2:] != far.shape[2:]:
            raise ValueError("Align-CRUSE mic/far feature grids must match")
        b, _, t, _ = mic.shape
        q = self.mic_pool(mic).permute(0, 2, 1, 3).reshape(b, t, -1)
        k = self.far_pool(far).permute(0, 2, 1, 3).reshape(b, t, -1)
        q = self.query(q)
        k = self.key(k)
        k4 = k.transpose(1, 2).unsqueeze(-1)
        k_delayed = causal_delay_stack(k4, self.max_delay_frames)
        k_delayed = k_delayed.squeeze(-1).permute(0, 2, 3, 1)
        frame = torch.arange(t, device=mic.device)[None, :, None]
        delay = torch.arange(
            self.max_delay_frames, device=mic.device,
        )[None, None, :]
        valid = (frame >= delay).to(q.dtype)
        scores = (q.unsqueeze(2) * k_delayed).sum(-1)
        observable = torch.arange(
            self.max_delay_frames, device=mic.device,
        ) < t
        if self.mode == "paper_global":
            # Paper: time-axis dot product for every synthetic delay, followed
            # by one softmax vector D in R^dmax for the complete utterance.
            logits = (scores * valid).sum(dim=1)
            logits = logits.masked_fill(~observable[None], float("-inf"))
            distribution = torch.softmax(logits, dim=-1)
            weights = distribution[:, None, None, :, None]
        else:
            # Streaming counterpart: only past evidence contributes to D(t).
            score_sum = (scores * valid).cumsum(dim=1)
            logits = score_sum.masked_fill(
                ~observable[None, None], float("-inf"),
            )
            distribution = torch.softmax(logits, dim=-1)
            weights = distribution[:, None, :, :, None]
        v_delayed = causal_delay_stack(far, self.max_delay_frames)
        aligned = (v_delayed * weights).sum(dim=3)
        return aligned, distribution


def apply_causal_tf_filter(spec: Tensor, taps: Tensor,
                           time_order: int, freq_radius: int) -> Tensor:
    """Apply time/frequency-varying complex FIR taps to ``spec``.

    ``taps`` is complex ``[B,T,F,time_order,2*freq_radius+1]``.  Time taps are
    current and past only; frequency taps are symmetric.
    """
    require_complex_btf(spec, "spec")
    if not torch.is_complex(taps) or taps.ndim != 5:
        raise ValueError("taps must be complex [B,T,F,O,K]")
    if taps.shape[:3] != spec.shape:
        raise ValueError("tap and spectrum grids differ")
    out = torch.zeros_like(spec)
    for dt in range(time_order):
        shifted_t = F.pad(spec, (0, 0, dt, 0))[:, :spec.shape[1]]
        for j, df in enumerate(range(-freq_radius, freq_radius + 1)):
            if df < 0:
                shifted = F.pad(shifted_t[..., :df], (-df, 0))
            elif df > 0:
                shifted = F.pad(shifted_t[..., df:], (0, df))
            else:
                shifted = shifted_t
            out = out + shifted * taps[..., dt, j]
    return out


def three_vector_complex(weights: Tensor) -> Tensor:
    """DeepVQE's three 120-degree real weights -> one complex value."""
    if weights.shape[-1] != 3:
        raise ValueError("last dimension must contain three vector weights")
    root3_over_2 = 3.0 ** 0.5 / 2.0
    real = weights[..., 0] - 0.5 * weights[..., 1] - 0.5 * weights[..., 2]
    imag = root3_over_2 * (weights[..., 1] - weights[..., 2])
    return torch.complex(real, imag)
