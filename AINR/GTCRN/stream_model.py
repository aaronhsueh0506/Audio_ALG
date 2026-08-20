"""Explicit-state one-frame wrapper around the tracked offline GTCRN model.

The learned modules remain the exact objects from :mod:`model`; this file only
replays causal temporal convolutions and recurrent layers with tensor caches.
It replaces the old export-time dependency on the ignored upstream reference
clone, so a clean checkout can export a deployable graph.
"""

import torch
from torch import nn
import torch.nn.functional as F


class _FrequencyNeighborhood(nn.Module):
    """Exact kernel-3 SFE lowering as one grouped convolution.

    ``nn.Unfold`` exports a frequency-only neighbourhood as Pad/Gather/Gather/
    Transpose/Reshape.  Seven copies of that pattern dominate the otherwise
    small GTCRN graph on accelerators with per-operator launch or DMA costs.
    A grouped one-hot convolution produces the same channel-interleaved
    ``[left, centre, right]`` values while lowering to one supported Conv.
    The weights are fixed buffers, not checkpoint parameters.
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
    """Per-GT-block temporal cache extent, (kernel-1)*dilation each.

    The single source for how much of the shared temporal cache each encoder
    block owns; both the cache allocation (`initial_inputs`) and the slice
    bounds that read it (`StreamGTCRN`) derive from this, so they cannot
    drift apart.
    """
    return [block.pad_size for block in model.encoder.en_convs[2:]]


def initial_inputs(model):
    """One random spectrum frame plus zero states, shaped for ``model``.

    Ordered exactly like the graph I/O: three feature inputs, six block-local
    convolution histories, then one ``h_*`` tensor per stateful GRU (the six
    TRA attention GRUs followed by the four grouped DPGRNN inter GRUs). Every
    extent is read off the
    module tree rather than written down, so the graph follows whatever grid
    the checkpoint was trained on: the spectrum width tracks n_fft through
    the ERB split, the cache depth tracks the GT blocks' temporal
    kernels/dilations, and the frequency width tracks the encoder strides
    through ``dpgrnn1.width``.
    """
    erb = model.erb
    bins = erb.erb_subband_1 + erb.erb_fc.in_features
    gt_blocks = list(model.encoder.en_convs[2:])
    channels = gt_blocks[0].depth_conv.in_channels
    width = model.dpgrnn1.width
    tra = gt_blocks[0].tra.att_gru
    inter = model.dpgrnn1.inter_rnn
    h_tra = [torch.zeros(tra.num_layers, 1, tra.hidden_size)
             for _ in range(2 * len(gt_blocks))]
    # Each grouped inter-RNN owns two real GRUs. Exposing one tensor per GRU
    # avoids slicing and re-concatenating recurrent state inside the graph.
    # Frequency positions are the GRU batch, hence (layers, width, hidden/2).
    h_dpgrnn = [
        torch.zeros(inter.num_layers, width, inter.rnn1.hidden_size)
        for _ in range(4)
    ]
    encoder_cache = [
        torch.zeros(1, channels, pad, width)
        for pad in _encoder_pad_sizes(model)
    ]
    decoder_cache = [
        torch.zeros(1, channels, block.pad_size, width)
        for block in model.decoder.de_convs[:len(gt_blocks)]
    ]
    bands = erb.erb_subband_1 + erb.erb_fc.out_features
    return (
        # Three separate ERB-domain feature frames (mag, real, imag), each
        # (1, E, 1): the magnitude AND the frozen ERB forward matmul run on
        # the HOST in fp32 (C: gtcrn_model_input), the graph concatenates
        # and holds learned compute only, and each channel keeps its own
        # quantization scale.
        torch.randn(1, bands, 1).abs(),
        torch.randn(1, bands, 1),
        torch.randn(1, bands, 1),
        *encoder_cache,
        *decoder_cache,
        *h_tra,
        *h_dpgrnn,
    )


def stream_features(model, spectrum_ri):
    """Host-side model input from RI spectra: (1, F, T, 2) -> 3x (1, E, T).

    The complete fixed front end, computed in fp32 OUTSIDE the graph
    (C: gtcrn_model_input): [sqrt(re^2 + im^2 + 1e-12), re, im] followed by
    the frozen ERB forward matmul (model.erb.bm), so neither the sqrt nor
    the filterbank ever enters the quantized domain. E = erb_subband_1 +
    erb_subband_2 (129 on every grid).
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
    """Host-side fixed back end: ERB inverse + complex ratio mask, in fp32.

    mask_erb (1, E, T, 2) is the graph output; spectrum_ri (1, F, T, 2) is
    the same frame the features came from. Mirrors model.erb.bs + Mask
    exactly (C: gtcrn_model_output).
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


def _conv_depth_step(conv, x, cache):
    combined = torch.cat((cache, x), dim=2)
    output = conv(combined)
    return output, combined[:, :, x.shape[2]:]


def _deconv_depth_step(deconv, x, cache, flipped_weight):
    """Causal stride-one depthwise ConvTranspose2d as an ordinary Conv2d.

    ``flipped_weight`` is ``deconv.weight.flip(-2, -1)`` precomputed at wrap
    time: flipping inside the traced forward would put a step=-1 Slice pair
    into the graph on every invocation (and defeat constant folding) for a
    tensor that never changes after load.
    """
    combined = torch.cat((cache, x), dim=2)
    next_cache = combined[:, :, x.shape[2]:]
    kt, kf = deconv.kernel_size
    _, df = deconv.dilation
    frequency_pad = (kf - 1) * df - deconv.padding[1]
    if frequency_pad:
        combined = F.pad(combined, (frequency_pad, frequency_pad, 0, 0))
    output = F.conv2d(
        combined, flipped_weight, deconv.bias, stride=(1, 1), padding=0,
        dilation=deconv.dilation, groups=deconv.groups)
    return output, next_cache


def _gt_step(block, x, conv_cache, tra_cache, frequency_sfe,
             flipped_weight=None):
    first, residual = torch.chunk(x, chunks=2, dim=1)
    first = frequency_sfe(first)
    first = block.point_act(block.point_bn1(block.point_conv1(first)))
    if flipped_weight is not None:
        first, conv_cache = _deconv_depth_step(
            block.depth_conv, first, conv_cache, flipped_weight)
    else:
        first, conv_cache = _conv_depth_step(
            block.depth_conv, first, conv_cache)
    first = block.depth_act(block.depth_bn(first))
    first = block.point_bn2(block.point_conv2(first))
    first, tra_cache = _tra_step(block.tra, first, tra_cache)
    return block.shuffle(first, residual), conv_cache, tra_cache


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
        # Each GT block receives its own graph state tensor. Keeping histories
        # separate eliminates the Gather/Slice/Concat packing graph that the
        # original shared cache required without changing total state bytes.
        encoder_pads = _encoder_pad_sizes(model)
        decoder_pads = [block.pad_size
                        for block in model.decoder.de_convs[:len(encoder_pads)]]
        assert decoder_pads == encoder_pads[::-1]
        for index, block in enumerate(
                model.decoder.de_convs[:len(encoder_pads)]):
            self.register_buffer(
                '_decoder_flip%d' % index,
                block.depth_conv.weight.detach().flip(-2, -1).clone(),
            )

    def forward(self, mag, real, imag,
                conv_enc0, conv_enc1, conv_enc2,
                conv_dec0, conv_dec1, conv_dec2,
                h_tra_enc0, h_tra_enc1, h_tra_enc2,
                h_tra_dec0, h_tra_dec1, h_tra_dec2,
                h_dpgrnn1_0, h_dpgrnn1_1,
                h_dpgrnn2_0, h_dpgrnn2_1):
        model = self.model
        # mag/real/imag are the HOST-banded feature frames (1, E, 1): the
        # fixed front end already ran outside; the graph concatenates them
        # into (B, 3, T, E) and holds learned compute only.
        x = torch.stack((mag, real, imag), dim=1).permute(0, 1, 3, 2)
        x = self.input_sfe(x)

        skips = []
        for block in model.encoder.en_convs[:2]:
            x = block(x)
            skips.append(x)
        enc_conv = []
        enc_tra = []
        enc_hidden = (h_tra_enc0, h_tra_enc1, h_tra_enc2)
        enc_cache = (conv_enc0, conv_enc1, conv_enc2)
        for index, block in enumerate(model.encoder.en_convs[2:]):
            x, cache, hidden = _gt_step(
                block, x, enc_cache[index],
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

        dec_conv_reverse = []
        dec_tra = []
        dec_hidden = (h_tra_dec0, h_tra_dec1, h_tra_dec2)
        dec_cache = (conv_dec0, conv_dec1, conv_dec2)
        for index, block in enumerate(model.decoder.de_convs[:3]):
            x, cache, hidden = _gt_step(
                block, x + skips[4 - index],
                dec_cache[index], dec_hidden[index],
                self.block_sfe,
                flipped_weight=getattr(self, '_decoder_flip%d' % index))
            dec_conv_reverse.append(cache)
            dec_tra.append(hidden)
        for index, block in enumerate(model.decoder.de_convs[3:], start=3):
            x = block(x + skips[4 - index])

        # x is the ERB-domain complex mask (B, 2, T, E); the fixed back end
        # (ERB inverse + CRM) runs on the host -- see host_synthesis.
        enhanced = x.permute(0, 3, 2, 1)
        return (enhanced, *enc_conv, *dec_conv_reverse,
                *enc_tra, *dec_tra,
                inter10, inter11, inter20, inter21)


def pack_state(model, state):
    """The sixteen per-slot state tensors as three, in graph order.

    The convolution histories become one ``(2, C, sum(pads), F)``: encoder row
    first, decoder row second, and within a row the blocks keep their order
    and join along the DEPTH axis, the only axis they differ on -- every
    history is ``(1, C, pad, F)`` with the same C and F. Every group joins
    along an axis it already has, the caches along that size-1 batch axis and
    the hiddens along their layer axis, so nothing gains a dimension and the
    four-dimensional ceiling holds.

    Shared by the combined wrapper's own forward and by whoever builds its
    initial inputs, so the packing rule exists once.
    """
    blocks = len(_encoder_pad_sizes(model))
    return (
        torch.cat((torch.cat(state[:blocks], dim=2),
                   torch.cat(state[blocks:2 * blocks], dim=2)), dim=0),
        torch.cat(state[2 * blocks:4 * blocks], dim=0),
        torch.cat(state[4 * blocks:], dim=0),
    )


class CombinedStateGTCRN(nn.Module):
    """Three-tensor state boundary over the per-slot streaming graph.

    Wraps ``StreamGTCRN`` instead of reimplementing it, so both boundaries run
    exactly the same compute and cannot drift; the difference is only where
    the sixteen state tensors are cut. The three groups are the ones whose
    members share a shape: the convolution histories (equal C and F, differing
    depth), the six TRA attention hiddens, and the four DPGRNN inter hiddens.
    """

    def __init__(self, model):
        super().__init__()
        self.stream = StreamGTCRN(model)
        self.encoder_pads = list(_encoder_pad_sizes(model))
        self.decoder_pads = self.encoder_pads[::-1]

    @property
    def model(self):
        return self.stream.model

    def forward(self, mag, real, imag, conv_cache, h_tra, h_dpgrnn):
        outputs = self.stream(
            mag, real, imag,
            *torch.split(conv_cache[0:1], self.encoder_pads, dim=2),
            *torch.split(conv_cache[1:2], self.decoder_pads, dim=2),
            *torch.split(h_tra, 1, dim=0),
            *torch.split(h_dpgrnn, 1, dim=0),
        )
        return (outputs[0],) + pack_state(self.model, outputs[1:])
