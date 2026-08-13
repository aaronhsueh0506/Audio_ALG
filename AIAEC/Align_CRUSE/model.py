"""Align-CRUSE reproduction for direct neural AEC (including RES).

Paper: Indenbom et al., "Deep model with built-in cross-attention alignment
for acoustic echo cancellation", arXiv:2208.11308.

The model consumes unaligned microphone/reference complex spectra.  It emits a
real magnitude mask and preserves microphone phase, exactly as the paper's
prediction contract specifies.  The project target is now the joint
end-to-end AEC+RES+NR task (early/dereverberated near speech, denoised and
echo-cancelled) -- this candidate's original AEC-only, noise-preserving route
was retired and folded into that shared task (see
../dataset_gen/model_views.py's MODEL_TASKS).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from AIAEC.aiaec_common import (
    AecOutput,
    CausalConvBlock,
    GlobalDelayAttention,
    SignalGrid,
    fit_frequency,
    log_power_feature,
    require_complex_btf,
)
from AIAEC.aiaec_streaming import (
    GlobalDelayAttentionCell,
    StreamConv2dCell,
    StreamGRUCell,
    StreamModuleCell,
    assert_streaming_ready,
)


def _half(n: int) -> int:
    return (n + 1) // 2


class AlignCRUSE(nn.Module):
    """Paper-shaped CRUSE with configurable streaming/global alignment."""

    paper_reference = "arXiv:2208.11308"
    task = "end_to_end_aec_res_nr_dereverb"
    # Streaming emits the mask for the current frame immediately; no extra
    # algorithmic latency beyond the STFT hop itself.
    stream_output_delay = 0

    def __init__(self, grid: SignalGrid, max_delay_seconds: float = 1.0,
                 projection_size: int = 64, gru_hidden: int = 192,
                 alignment_mode: str = "causal_running"):
        super().__init__()
        self.grid = grid
        self.max_delay_frames = grid.delay_frames(max_delay_seconds)

        # Paper channel schedule: mic 16/40/72/32, far 8/24.
        self.mic1 = CausalConvBlock(1, 16, (4, 3), (1, 2))
        self.mic2 = CausalConvBlock(16, 40, (4, 3), (1, 2))
        self.far1 = CausalConvBlock(1, 8, (4, 3), (1, 2))
        self.far2 = CausalConvBlock(8, 24, (4, 3), (1, 2))

        f1 = _half(grid.n_freqs)
        f2 = _half(f1)
        self.align = GlobalDelayAttention(
            40, 24, f2, f2, 24, projection_size, self.max_delay_frames,
            mode=alignment_mode,
        )
        self.mic3 = CausalConvBlock(40 + 24, 72, (4, 3), (1, 2))
        self.mic4 = CausalConvBlock(72, 32, (4, 3), (1, 2))

        f3, f4 = _half(f2), _half(_half(f2))
        bottleneck_size = 32 * f4
        self.gru = nn.GRU(bottleneck_size, gru_hidden, batch_first=True)
        self.gru_out = nn.Linear(gru_hidden, bottleneck_size)

        # Three ConvT blocks (32/48/48) plus the paper's final mask block.
        # The mask block performs the fourth frequency restoration required by
        # four stride-2 encoder blocks; this is the only shape-complete reading
        # of the paper's text/figure at arbitrary odd RFFT widths.
        self.skip4 = nn.Conv2d(32, 32, 1)
        self.up3 = FrequencyTransposeBlock(32, 32)
        self.skip3 = nn.Conv2d(72, 32, 1)
        self.up2 = FrequencyTransposeBlock(32, 48)
        self.skip2 = nn.Conv2d(40, 48, 1)
        self.up1 = FrequencyTransposeBlock(48, 48)
        self.skip1 = nn.Conv2d(16, 48, 1)
        self.mask_up = FrequencyTransposeBlock(
            48, 1, activation=False, normalization=False,
        )
        self.mask_gain = nn.Parameter(torch.ones(()))

        self._encoded_freq = (f1, f2, f3, f4)

    def forward(self, microphone: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(microphone, "microphone")
        require_complex_btf(far_end, "far_end")
        if microphone.shape != far_end.shape:
            raise ValueError("microphone and far_end STFT grids must match")
        if microphone.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        m0 = log_power_feature(microphone)
        f0 = log_power_feature(far_end)
        m1 = self.mic1(m0)
        m2 = self.mic2(m1)
        f1 = self.far1(f0)
        f2 = self.far2(f1)
        aligned, delay = self.align(m2, f2)
        m3 = self.mic3(torch.cat((m2, aligned), dim=1))
        m4 = self.mic4(m3)

        b, c, t, f = m4.shape
        sequence = m4.permute(0, 2, 1, 3).reshape(b, t, c * f)
        sequence, _ = self.gru(sequence)
        x = self.gru_out(sequence).reshape(b, t, c, f).permute(0, 2, 1, 3)

        x = self.up3(x + self.skip4(m4), m3.shape[-1])
        x = self.up2(x + self.skip3(m3), m2.shape[-1])
        x = self.up1(x + self.skip2(m2), m1.shape[-1])
        logits = self.mask_up(x + self.skip1(m1), self.grid.n_freqs).squeeze(1)
        mask = torch.sigmoid(fit_frequency(logits, self.grid.n_freqs))
        mask = mask * self.mask_gain.clamp_min(0.0)
        enhanced = microphone * mask
        return AecOutput(enhanced=enhanced, mask=mask,
                         delay_distribution=delay)

    def create_stream_state(self) -> Dict[str, object]:
        """Named streaming cells for :meth:`forward_stream`.

        Only ``causal_running`` alignment can stream: ``paper_global`` reduces
        the query/key product over the whole utterance before its softmax.
        """
        assert_streaming_ready(self)
        if self.align.mode != "causal_running":
            raise ValueError(
                "Align-CRUSE streaming supports alignment_mode="
                "'causal_running' only; 'paper_global' emits one delay "
                "distribution per utterance and cannot run frame-by-frame"
            )

        def block(module: CausalConvBlock) -> StreamModuleCell:
            return StreamModuleCell(
                StreamConv2dCell.from_causal(module.conv),
                [module.norm, module.activation],
            )

        return {
            "mic1": block(self.mic1),
            "mic2": block(self.mic2),
            "far1": block(self.far1),
            "far2": block(self.far2),
            "align": GlobalDelayAttentionCell(self.align),
            "mic3": block(self.mic3),
            "mic4": block(self.mic4),
            "gru": StreamGRUCell(self.gru),
        }

    def forward_stream(self, microphone: Tensor, far_end: Tensor,
                       state: Dict[str, object]) -> AecOutput:
        """Streaming twin of :meth:`forward` for one new STFT frame.

        Inputs are complex ``[B, 1, F]`` (a T=1 slice of the offline input);
        the returned fields are the matching T=1 slices.  The decoder's
        transposed convolutions have time kernel 1, so they run per frame
        unchanged; only the encoder convs, the alignment integrator and the
        GRU carry state.

        Equivalence caveat (inherited from GlobalDelayAttentionCell): offline
        masks delays with ``arange(D) < T`` using the FINAL utterance length,
        which a stream cannot know.  For utterances of at least
        ``max_delay_frames`` hops the stream matches offline exactly; shorter
        utterances differ by design.
        """
        require_complex_btf(microphone, "microphone")
        require_complex_btf(far_end, "far_end")
        if microphone.shape != far_end.shape:
            raise ValueError("microphone and far_end STFT grids must match")
        if microphone.shape[1] != 1:
            raise ValueError("forward_stream consumes exactly one frame per call")
        if microphone.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        m0 = log_power_feature(microphone)
        f0 = log_power_feature(far_end)
        m1 = state["mic1"].step(m0)
        m2 = state["mic2"].step(m1)
        f1 = state["far1"].step(f0)
        f2 = state["far2"].step(f1)
        aligned, delay = state["align"].step(m2, f2)
        m3 = state["mic3"].step(torch.cat((m2, aligned), dim=1))
        m4 = state["mic4"].step(m3)

        b, c, _, f = m4.shape
        sequence = m4.permute(0, 2, 1, 3).reshape(b, 1, c * f)
        sequence = state["gru"].step(sequence)
        x = self.gru_out(sequence).reshape(b, 1, c, f).permute(0, 2, 1, 3)

        x = self.up3(x + self.skip4(m4), m3.shape[-1])
        x = self.up2(x + self.skip3(m3), m2.shape[-1])
        x = self.up1(x + self.skip2(m2), m1.shape[-1])
        logits = self.mask_up(x + self.skip1(m1), self.grid.n_freqs).squeeze(1)
        mask = torch.sigmoid(fit_frequency(logits, self.grid.n_freqs))
        mask = mask * self.mask_gain.clamp_min(0.0)
        enhanced = microphone * mask
        return AecOutput(enhanced=enhanced, mask=mask,
                         delay_distribution=delay.unsqueeze(1))


class FrequencyTransposeBlock(nn.Module):
    """Paper ConvT block: a frequency-only 1x3 transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 activation: bool = True, normalization: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(1, 3),
            stride=(1, 2), padding=(0, 1), bias=False,
        )
        self.norm = (nn.BatchNorm2d(out_channels) if normalization
                     else nn.Identity())
        self.act = nn.ELU() if activation else nn.Identity()

    def forward(self, x: Tensor, target_freq: int) -> Tensor:
        output_size = (x.shape[0], self.out_channels, x.shape[2], target_freq)
        x = self.conv(x, output_size=output_size)
        return self.act(self.norm(x))
