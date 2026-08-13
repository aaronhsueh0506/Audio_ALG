"""GTCRN conditioned on a linear-AEC error/reference pair.

This is a project AENR variant, not a model claimed by the GTCRN paper.  Every
GTCRN block after the first convolution is reused unchanged from the audited
standalone AINR implementation; only the input contract grows from one complex
spectrum to two.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from AINR.GTCRN.model import ConvBlock, GTCRN
from AIAEC.aiaec_common import AecOutput, SignalGrid, require_complex_btf
from AIAEC.aiaec_streaming import (
    StreamConv2dCell,
    StreamGRUCell,
    assert_streaming_ready,
)


def _deconv_as_causal_conv(deconv: nn.ConvTranspose2d) -> nn.Conv2d:
    """Equivalent causal Conv2d for a stride-1 depthwise ConvTranspose2d.

    The offline decoder left-pads ``(kt-1)*dt`` frames and lets the transposed
    conv's own time padding trim them back; for stride 1 that is a causal
    convolution with the time-and-frequency flipped kernel (same identity the
    upstream streaming reference applies at weight-conversion time), so the
    verified StreamConv2dCell applies unchanged.
    """
    kt, kf = deconv.kernel_size
    dt, df = deconv.dilation
    if deconv.stride != (1, 1):
        raise ValueError("streaming decoder expects stride-1 transposed convs")
    if not (deconv.groups == deconv.in_channels == deconv.out_channels):
        raise ValueError("kernel flip below assumes a depthwise transposed conv")
    if deconv.padding[0] != (kt - 1) * dt:
        raise ValueError("time padding must equal (kt-1)*dt for a causal trim")
    freq_pad = (kf - 1) * df - deconv.padding[1]
    if freq_pad < 0:
        raise ValueError("frequency padding exceeds the transposed kernel span")
    conv = nn.Conv2d(deconv.in_channels, deconv.out_channels, (kt, kf),
                     stride=(1, 1), padding=(0, freq_pad), dilation=(dt, df),
                     groups=deconv.groups, bias=deconv.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(deconv.weight.flip(-2, -1))
        if deconv.bias is not None:
            conv.bias.copy_(deconv.bias)
    conv = conv.to(device=deconv.weight.device, dtype=deconv.weight.dtype)
    conv.requires_grad_(False)
    return conv.eval()


def _gt_conv_block_step(block, x: Tensor, depth_cell: StreamConv2dCell,
                        tra_cell: StreamGRUCell) -> Tensor:
    """One-frame replay of GTConvBlock.forward.

    The depth cell owns the ``(kt-1)*dt`` left context that the offline
    ``F.pad`` provides; the TRA cell carries the attention GRU hidden state.
    """
    x1, x2 = torch.chunk(x, chunks=2, dim=1)
    x1 = block.sfe(x1)
    h1 = block.point_act(block.point_bn1(block.point_conv1(x1)))
    h1 = depth_cell.step(h1)
    h1 = block.depth_act(block.depth_bn(h1))
    h1 = block.point_bn2(block.point_conv2(h1))
    zt = h1.pow(2).mean(dim=-1)                      # [B,C,1]
    at = tra_cell.step(zt.transpose(1, 2))           # [B,1,2C]
    at = block.tra.att_act(block.tra.att_fc(at)).transpose(1, 2)
    return block.shuffle(h1 * at[..., None], x2)


def _dpgrnn_step(dp, x: Tensor, inter_a: StreamGRUCell,
                 inter_b: StreamGRUCell) -> Tensor:
    """One-frame replay of DPGRNN.forward.

    The intra pass is frequency-axis bidirectional, hence per-frame; only the
    two inter GRUs (batched over B*F) carry state.  GRNN's internal hidden
    default is float32 regardless of input dtype, so the intra hidden is passed
    explicitly in the input dtype.
    """
    b = x.shape[0]
    x = x.permute(0, 2, 3, 1)                        # [B,1,F,C]
    intra_x = x.reshape(b, dp.width, dp.input_size)
    h0 = intra_x.new_zeros(2, b, dp.intra_rnn.hidden_size)
    intra_x = dp.intra_rnn(intra_x, h0)[0]
    intra_x = dp.intra_fc(intra_x)
    intra_x = intra_x.reshape(b, 1, dp.width, dp.hidden_size)
    intra_x = dp.intra_ln(intra_x)
    intra_out = x + intra_x                          # [B,1,F,C]

    inter_in = intra_out.permute(0, 2, 1, 3).reshape(
        b * dp.width, 1, dp.hidden_size)
    i1, i2 = torch.chunk(inter_in, chunks=2, dim=-1)
    inter_x = torch.cat((inter_a.step(i1), inter_b.step(i2)), dim=-1)
    inter_x = dp.inter_fc(inter_x)
    inter_x = inter_x.reshape(b, dp.width, 1, dp.hidden_size).permute(0, 2, 1, 3)
    inter_x = dp.inter_ln(inter_x)
    return (intra_out + inter_x).permute(0, 3, 1, 2)  # [B,C,1,F]


class GTCRNAENR(GTCRN):
    base_reference = "Xiaobin Rong et al., GTCRN (INTERSPEECH 2024)"
    task = "linear_aec_postfilter_res_nr_dereverb"
    stream_output_delay = 0

    def __init__(self, grid: SignalGrid, erb_subband_1: int = 65,
                 erb_subband_2: int = 64):
        if (grid.sample_rate, grid.n_fft) != (16000, 512):
            raise ValueError(
                "GTCRN-AENR follows the upstream 16 kHz/512 grid; use the "
                "DeepFilterNet-AENR candidate for the 48 kHz route"
            )
        super().__init__(erb_subband_1, erb_subband_2,
                         nfft=grid.n_fft, fs=grid.sample_rate)
        self.grid = grid
        # One source produces [mag,re,im] then SFE(k=3) => 9 channels.
        # Error + far reference therefore produce 18. All later GTCRN layers,
        # DPGRNN widths, decoder and CRM arithmetic are unchanged.
        self.encoder.en_convs[0] = ConvBlock(
            18, 16, (1, 5), stride=(1, 2), padding=(0, 2),
        )

    @staticmethod
    def _three_features(spec: Tensor) -> Tensor:
        real = spec.real
        imag = spec.imag
        mag = (real.square() + imag.square() + 1e-12).sqrt()
        return torch.stack((mag, real, imag), dim=1)

    def forward(self, linear_error: Tensor, far_end: Tensor) -> AecOutput:
        require_complex_btf(linear_error, "linear_error")
        require_complex_btf(far_end, "far_end")
        if linear_error.shape != far_end.shape:
            raise ValueError("linear_error and far_end STFT grids must match")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        err = self.sfe(self.erb.bm(self._three_features(linear_error)))
        far = self.sfe(self.erb.bm(self._three_features(far_end)))
        feat, skips = self.encoder(torch.cat((err, far), dim=1))
        feat = self.dpgrnn2(self.dpgrnn1(feat))
        mask_erb = self.decoder(feat, skips)
        mask_full = self.erb.bs(mask_erb)
        enhanced_ri = self.mask(
            mask_full,
            torch.stack((linear_error.real, linear_error.imag), dim=1),
        )
        enhanced = torch.complex(enhanced_ri[:, 0], enhanced_ri[:, 1])
        mask = torch.complex(mask_full[:, 0], mask_full[:, 1])
        return AecOutput(enhanced=enhanced, mask=mask,
                         auxiliary={"erb_complex_mask": mask_erb})

    def create_stream_state(self) -> Dict[str, object]:
        """Named stateful cells for :meth:`forward_stream`.

        Only the six GT-block depthwise convs (2/4/10-frame input rings), the
        six TRA attention GRUs and the four DPGRNN inter GRUs carry state; ERB,
        SFE, the (1,5) frequency convs and the intra GRUs are per-frame.
        """
        assert_streaming_ready(self)
        state: Dict[str, object] = {}
        for i, block in enumerate(self.encoder.en_convs[2:], start=1):
            state[f"enc_gt{i}_depth"] = StreamConv2dCell(
                block.depth_conv, kt=block.depth_conv.kernel_size[0],
                dt=block.depth_conv.dilation[0], freq_left=0, freq_right=0)
            state[f"enc_gt{i}_tra"] = StreamGRUCell(block.tra.att_gru)
        for i, block in enumerate(self.decoder.de_convs[:3], start=1):
            conv = _deconv_as_causal_conv(block.depth_conv)
            state[f"dec_gt{i}_depth"] = StreamConv2dCell(
                conv, kt=conv.kernel_size[0], dt=conv.dilation[0],
                freq_left=0, freq_right=0)
            state[f"dec_gt{i}_tra"] = StreamGRUCell(block.tra.att_gru)
        for i, dp in enumerate((self.dpgrnn1, self.dpgrnn2), start=1):
            state[f"dpgrnn{i}_inter_a"] = StreamGRUCell(dp.inter_rnn.rnn1)
            state[f"dpgrnn{i}_inter_b"] = StreamGRUCell(dp.inter_rnn.rnn2)
        return state

    def forward_stream(self, linear_error: Tensor, far_end: Tensor,
                       state: Dict[str, object]) -> AecOutput:
        """One-frame streaming twin of :meth:`forward` (T = 1 slices in/out).

        Emits every :meth:`forward` field per frame: enhanced, mask and the
        ``erb_complex_mask`` auxiliary — nothing is omitted.
        """
        require_complex_btf(linear_error, "linear_error")
        require_complex_btf(far_end, "far_end")
        if linear_error.shape != far_end.shape:
            raise ValueError("linear_error and far_end STFT grids must match")
        if linear_error.shape[1] != 1:
            raise ValueError("forward_stream consumes exactly one STFT frame")
        if linear_error.shape[-1] != self.grid.n_freqs:
            raise ValueError("input frequency count does not match SignalGrid")

        err = self.sfe(self.erb.bm(self._three_features(linear_error)))
        far = self.sfe(self.erb.bm(self._three_features(far_end)))
        x = torch.cat((err, far), dim=1)

        skips = []
        x = self.encoder.en_convs[0](x)
        skips.append(x)
        x = self.encoder.en_convs[1](x)
        skips.append(x)
        for i, block in enumerate(self.encoder.en_convs[2:], start=1):
            x = _gt_conv_block_step(block, x, state[f"enc_gt{i}_depth"],
                                    state[f"enc_gt{i}_tra"])
            skips.append(x)

        x = _dpgrnn_step(self.dpgrnn1, x, state["dpgrnn1_inter_a"],
                         state["dpgrnn1_inter_b"])
        x = _dpgrnn_step(self.dpgrnn2, x, state["dpgrnn2_inter_a"],
                         state["dpgrnn2_inter_b"])

        for i, block in enumerate(self.decoder.de_convs[:3], start=1):
            x = _gt_conv_block_step(block, x + skips[5 - i],
                                    state[f"dec_gt{i}_depth"],
                                    state[f"dec_gt{i}_tra"])
        x = self.decoder.de_convs[3](x + skips[1])
        mask_erb = self.decoder.de_convs[4](x + skips[0])

        mask_full = self.erb.bs(mask_erb)
        enhanced_ri = self.mask(
            mask_full,
            torch.stack((linear_error.real, linear_error.imag), dim=1),
        )
        enhanced = torch.complex(enhanced_ri[:, 0], enhanced_ri[:, 1])
        mask = torch.complex(mask_full[:, 0], mask_full[:, 1])
        return AecOutput(enhanced=enhanced, mask=mask,
                         auxiliary={"erb_complex_mask": mask_erb})
