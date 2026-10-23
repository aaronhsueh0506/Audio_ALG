"""Explicit-state one-frame wrapper around the offline GTCRN model.

The learned modules are the objects from :mod:`model`; this file replays the
causal temporal convolutions and recurrent layers with tensor windows and stacked
recurrent rows so the graph can be exported.

State boundary: every state tensor is rank 4 with the same shape at the graph
input and at its ``*_out`` output, so a runtime binds both to one address.

* ``conv_enc0..2`` / ``conv_dec0..2`` ``(1, C, window, F)``: the full causal
  window of one depthwise convolution (the graph drops the oldest frame with a
  Slice, appends the new one, convolves, and returns the whole concatenation).
* ``h_tra`` ``(1, 6 * layers, 1, hidden)``: the six TRA attention GRUs.
* ``h_dpgrnn`` ``(1, 4 * layers, width, hidden)``: the four grouped DPGRNN inter
  GRUs, whose hidden batch is the frequency axis.

GRU stacks are cut with Slices on dim 1 of the input and joined with one Concat
on dim 1 on the way out.
"""

import torch
from torch import nn
import torch.nn.functional as F


class _FrequencyNeighborhood(nn.Module):
    """Kernel-3 SFE lowered to one grouped convolution.

    ``nn.Unfold`` exports a frequency-only neighbourhood as Pad/Gather/Gather/
    Transpose/Reshape, and seven copies of that pattern dominate the small GTCRN
    graph on accelerators with per-operator launch or DMA costs. A grouped one-hot
    convolution yields the same channel-interleaved ``[left, centre, right]``
    values as one Conv. The weights are fixed buffers, not checkpoint parameters.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size != 3:
            raise ValueError('streaming GTCRN requires SFE kernel_size=3')
        self.channels = channels
        weight = torch.zeros(channels * kernel_size, 1, 1, kernel_size)
        for channel in range(channels):
            for offset in range(kernel_size):
                weight[channel * kernel_size + offset, 0, 0, offset] = 1.0
        self.register_buffer('weight', weight)

    def forward(self, x):
        return F.conv2d(
            x, self.weight, padding=(0, 1), groups=self.channels
        )


def _encoder_pad_sizes(model):
    """Per-GT-block temporal history, (kernel-1)*dilation frames each."""
    return [block.pad_size for block in model.encoder.en_convs[2:]]


def _encoder_window_sizes(model):
    """Per-GT-block window: the history plus the frame being computed. It is the
    single source of the window depth, used by the allocation and the slice
    bounds."""
    return [pad + 1 for pad in _encoder_pad_sizes(model)]


def _decoder_window_sizes(model):
    return _encoder_window_sizes(model)[::-1]


def _gru_row_counts(model):
    """Layers per TRA GRU and per DPGRNN inter GRU."""
    gt_blocks = list(model.encoder.en_convs[2:])
    return (gt_blocks[0].tra.att_gru.num_layers,
            model.dpgrnn1.inter_rnn.num_layers)


def initial_inputs(model):
    """One random spectrum frame plus zero states, shaped for ``model``.

    Ordered like the graph I/O: three feature inputs, six block-local convolution
    windows, then the stacked ``h_tra`` and ``h_dpgrnn`` tensors. Every extent is
    read off the module tree, so the graph follows the grid the checkpoint was
    trained on.
    """
    erb = model.erb
    gt_blocks = list(model.encoder.en_convs[2:])
    channels = gt_blocks[0].depth_conv.in_channels
    width = model.dpgrnn1.width
    tra = gt_blocks[0].tra.att_gru
    inter = model.dpgrnn1.inter_rnn
    h_tra = torch.zeros(1, 2 * len(gt_blocks) * tra.num_layers, 1,
                        tra.hidden_size)
    # Each grouped inter-RNN owns two real GRUs. Frequency positions are the
    # GRU batch, hence (1, layers * 4, width, hidden/2).
    h_dpgrnn = torch.zeros(1, 4 * inter.num_layers, width,
                           inter.rnn1.hidden_size)
    encoder_windows = [
        torch.zeros(1, channels, size, width)
        for size in _encoder_window_sizes(model)
    ]
    decoder_windows = [
        torch.zeros(1, channels, size, width)
        for size in _decoder_window_sizes(model)
    ]
    bands = erb.erb_subband_1 + erb.erb_fc.out_features
    return (
        # Three separate ERB-domain feature frames (mag, real, imag), each
        # (1, E, 1). The magnitude and the frozen ERB forward matmul run on the
        # host in fp32 (C: gtcrn_model_input); the graph holds learned compute only
        # and each channel keeps its own quantization scale.
        torch.randn(1, bands, 1).abs(),
        torch.randn(1, bands, 1),
        torch.randn(1, bands, 1),
        *encoder_windows,
        *decoder_windows,
        h_tra,
        h_dpgrnn,
    )


def stream_features(model, spectrum_ri):
    """Host-side model input from RI spectra: (1, F, T, 2) -> 3x (1, E, T).

    The complete fixed front end, computed in fp32 outside the graph (C:
    gtcrn_model_input): [sqrt(re^2 + im^2 + 1e-12), re, im] followed by the frozen
    ERB forward matmul (model.erb.bm). E = erb_subband_1 + erb_subband_2 (129 on
    every grid).
    """
    real = spectrum_ri[..., 0]
    imag = spectrum_ri[..., 1]
    magnitude = (real.square() + imag.square() + 1e-12).sqrt()
    stacked = torch.stack((magnitude, real, imag), dim=-1)
    # (1, F, T, 3) -> (1, 3, T, F) for bm's (B, C, T, F) convention.
    banded = model.erb.bm(stacked.permute(0, 3, 2, 1))
    # Three SEPARATE tensors (1, E, T): independent quantization scales for
    # the positive magnitude and the signed real/imag channels.
    return (banded[:, 0].permute(0, 2, 1),
            banded[:, 1].permute(0, 2, 1),
            banded[:, 2].permute(0, 2, 1))


def host_synthesis(model, mask_erb, spectrum_ri):
    """Host-side fixed back end: ERB inverse + complex ratio mask in fp32.

    mask_erb (1, E, T, 2) is the graph output; spectrum_ri (1, F, T, 2) is the
    frame the features came from (C: gtcrn_model_output).
    """
    mask = model.erb.bs(mask_erb.permute(0, 3, 2, 1))     # (1, 2, T, F)
    spec = spectrum_ri.permute(0, 3, 2, 1)                # (1, 2, T, F)
    enhanced = model.mask(mask, spec)
    return enhanced.permute(0, 3, 2, 1)                   # (1, F, T, 2)


def _tra_step(tra, x, hidden):
    energy = (x * x).mean(dim=-1)   # x*x, not pow: exports as Mul
    attention, hidden = tra.att_gru(energy.transpose(1, 2), hidden)
    attention = tra.att_act(tra.att_fc(attention)).transpose(1, 2)
    return x * attention[..., None], hidden


def _conv_depth_step(conv, x, window, size):
    """Causal depthwise convolution over ``window`` (``size`` frames).

    The oldest frame is dropped by a Slice of the input with explicit integer
    bounds, the new frame is appended, and the whole concatenation is both the
    convolution input and the next window.
    """
    combined = torch.cat((window[:, :, 1:size], x), dim=2)
    return conv(combined), combined


def _deconv_depth_step(deconv, x, window, size, flipped_weight):
    """Causal stride-one depthwise ConvTranspose2d as an ordinary Conv2d.

    ``flipped_weight`` is ``deconv.weight.flip(-2, -1)`` precomputed at wrap time,
    so the traced forward carries no step=-1 Slice pair.
    """
    combined = torch.cat((window[:, :, 1:size], x), dim=2)
    next_window = combined
    kt, kf = deconv.kernel_size
    _, df = deconv.dilation
    frequency_pad = (kf - 1) * df - deconv.padding[1]
    if frequency_pad:
        combined = F.pad(combined, (frequency_pad, frequency_pad, 0, 0))
    output = F.conv2d(
        combined, flipped_weight, deconv.bias, stride=(1, 1), padding=0,
        dilation=deconv.dilation, groups=deconv.groups)
    return output, next_window


def _gt_step(block, x, conv_window, window_size, tra_hidden, frequency_sfe,
             flipped_weight=None):
    first, residual = torch.chunk(x, chunks=2, dim=1)
    first = frequency_sfe(first)
    first = block.point_act(block.point_bn1(block.point_conv1(first)))
    if flipped_weight is not None:
        first, conv_window = _deconv_depth_step(
            block.depth_conv, first, conv_window, window_size, flipped_weight)
    else:
        first, conv_window = _conv_depth_step(
            block.depth_conv, first, conv_window, window_size)
    first = block.depth_act(block.depth_bn(first))
    first = block.point_bn2(block.point_conv2(first))
    first, tra_hidden = _tra_step(block.tra, first, tra_hidden)
    return block.shuffle(first, residual), conv_window, tra_hidden


def _dp_step(dp, x, inter_cache_0, inter_cache_1):
    batch = x.shape[0]
    x = x.permute(0, 2, 3, 1)
    intra = x.reshape(batch, dp.width, dp.input_size)
    intra = dp.intra_rnn(intra)[0]
    intra = dp.intra_fc(intra).reshape(
        batch, 1, dp.width, dp.hidden_size)
    intra = dp.intra_ln(intra)
    intra_out = x + intra
    inter = intra_out.permute(0, 2, 1, 3).reshape(
        batch * dp.width, 1, dp.hidden_size)
    inter_0, inter_1 = torch.chunk(inter, chunks=2, dim=-1)
    inter_0, inter_cache_0 = dp.inter_rnn.rnn1(
        inter_0, inter_cache_0
    )
    inter_1, inter_cache_1 = dp.inter_rnn.rnn2(
        inter_1, inter_cache_1
    )
    inter = torch.cat((inter_0, inter_1), dim=-1)
    inter = dp.inter_fc(inter).reshape(
        batch, dp.width, 1, dp.hidden_size).permute(0, 2, 1, 3)
    return ((intra_out + dp.inter_ln(inter)).permute(0, 3, 1, 2),
            inter_cache_0, inter_cache_1)


class StreamGTCRN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_sfe = _FrequencyNeighborhood(
            3, model.sfe.kernel_size
        )
        gt_blocks = list(model.encoder.en_convs[2:])
        block_channels = (
            gt_blocks[0].point_conv1.in_channels //
            gt_blocks[0].sfe.kernel_size
        )
        if any(block.point_conv1.in_channels !=
               block_channels * block.sfe.kernel_size
               for block in gt_blocks):
            raise ValueError('GT blocks no longer share one SFE channel shape')
        self.block_sfe = _FrequencyNeighborhood(
            block_channels, gt_blocks[0].sfe.kernel_size
        )
        # Each GT block receives its own window tensor, so no state is packed or
        # unpacked in the graph.
        encoder_pads = _encoder_pad_sizes(model)
        decoder_pads = [block.pad_size
                        for block in model.decoder.de_convs[:len(encoder_pads)]]
        assert decoder_pads == encoder_pads[::-1]
        self.encoder_windows = _encoder_window_sizes(model)
        self.decoder_windows = _decoder_window_sizes(model)
        self.tra_rows, self.dpgrnn_rows = _gru_row_counts(model)
        for index, block in enumerate(
                model.decoder.de_convs[:len(encoder_pads)]):
            self.register_buffer(
                '_decoder_flip%d' % index,
                block.depth_conv.weight.detach().flip(-2, -1).clone(),
            )

    def forward(self, mag, real, imag,
                conv_enc0, conv_enc1, conv_enc2,
                conv_dec0, conv_dec1, conv_dec2,
                h_tra, h_dpgrnn):
        # Each GRU reads its rows through a Slice on dim 1 of the stacked input and
        # drops the singleton batch for nn.GRU's (layers, batch, hidden); slices use
        # Python integer bounds.
        tra_rows, dpgrnn_rows = self.tra_rows, self.dpgrnn_rows
        tra_in = [h_tra[:, i * tra_rows:(i + 1) * tra_rows].squeeze(0)
                  for i in range(6)]
        dp_in = [h_dpgrnn[:, i * dpgrnn_rows:(i + 1) * dpgrnn_rows].squeeze(0)
                 for i in range(4)]
        (h_tra_enc0, h_tra_enc1, h_tra_enc2,
         h_tra_dec0, h_tra_dec1, h_tra_dec2) = tra_in
        h_dpgrnn1_0, h_dpgrnn1_1, h_dpgrnn2_0, h_dpgrnn2_1 = dp_in
        model = self.model
        # mag/real/imag are the host-banded feature frames (1, E, 1); the graph
        # concatenates them into (B, 3, T, E).
        x = torch.stack((mag, real, imag), dim=1).permute(0, 1, 3, 2)
        x = self.input_sfe(x)

        skips = []
        for block in model.encoder.en_convs[:2]:
            x = block(x)
            skips.append(x)
        enc_conv = []
        enc_tra = []
        enc_hidden = (h_tra_enc0, h_tra_enc1, h_tra_enc2)
        enc_window = (conv_enc0, conv_enc1, conv_enc2)
        for index, block in enumerate(model.encoder.en_convs[2:]):
            x, cache, hidden = _gt_step(
                block, x, enc_window[index], self.encoder_windows[index],
                enc_hidden[index], self.block_sfe)
            enc_conv.append(cache)
            enc_tra.append(hidden)
            skips.append(x)

        x, inter10, inter11 = _dp_step(
            model.dpgrnn1, x, h_dpgrnn1_0, h_dpgrnn1_1
        )
        x, inter20, inter21 = _dp_step(
            model.dpgrnn2, x, h_dpgrnn2_0, h_dpgrnn2_1
        )

        dec_conv = []
        dec_tra = []
        dec_hidden = (h_tra_dec0, h_tra_dec1, h_tra_dec2)
        dec_window = (conv_dec0, conv_dec1, conv_dec2)
        for index, block in enumerate(model.decoder.de_convs[:3]):
            x, cache, hidden = _gt_step(
                block, x + skips[4 - index],
                dec_window[index], self.decoder_windows[index],
                dec_hidden[index], self.block_sfe,
                flipped_weight=getattr(self, '_decoder_flip%d' % index))
            dec_conv.append(cache)
            dec_tra.append(hidden)
        for index, block in enumerate(model.decoder.de_convs[3:], start=3):
            x = block(x + skips[4 - index])

        # x is the ERB-domain complex mask (B, 2, T, E); the fixed back end (ERB
        # inverse + CRM) runs on the host (host_synthesis).
        enhanced = x.permute(0, 3, 2, 1)
        # The stacks leave in the order they were sliced.
        h_tra_next = torch.cat(
            [hidden.unsqueeze(0) for hidden in (*enc_tra, *dec_tra)], dim=1)
        h_dpgrnn_next = torch.cat(
            [hidden.unsqueeze(0)
             for hidden in (inter10, inter11, inter20, inter21)], dim=1)
        return (enhanced, *enc_conv, *dec_conv,
                h_tra_next, h_dpgrnn_next)
