"""Explicit-state one-frame wrapper around the tracked offline GTCRN model.

The learned modules remain the exact objects from :mod:`model`; this file only
replays causal temporal convolutions and recurrent layers with tensor caches.
It replaces the old export-time dependency on the ignored upstream reference
clone, so a clean checkout can export a deployable graph.
"""

import torch
from torch import nn
import torch.nn.functional as F


def _encoder_pad_sizes(model):
    """Per-GT-block temporal cache extent, (kernel-1)*dilation each.

    The single source for how much of the shared temporal cache each encoder
    block owns; both the cache allocation (`initial_inputs`) and the slice
    bounds that read it (`StreamGTCRN`) derive from this, so they cannot
    drift apart.
    """
    return [block.pad_size for block in model.encoder.en_convs[2:]]


def initial_inputs(model):
    """One random spectrum frame plus zero caches, shaped for ``model``.

    Every extent is read off the module tree rather than written down, so the
    graph follows whatever grid the checkpoint was trained on: the spectrum
    width tracks n_fft through the ERB split, the cache depth tracks the GT
    blocks' temporal kernels/dilations, and the frequency width tracks the
    encoder strides through ``dpgrnn1.width``.
    """
    erb = model.erb
    bins = erb.erb_subband_1 + erb.erb_fc.in_features
    gt_blocks = list(model.encoder.en_convs[2:])
    channels = gt_blocks[0].depth_conv.in_channels
    depth = sum(_encoder_pad_sizes(model))
    width = model.dpgrnn1.width
    tra = gt_blocks[0].tra.att_gru
    inter = model.dpgrnn1.inter_rnn
    return (
        torch.randn(1, bins, 1, 2),
        torch.zeros(2, 1, channels, depth, width),
        torch.zeros(2, len(gt_blocks), tra.num_layers, 1, tra.hidden_size),
        torch.zeros(2, inter.num_layers, width, inter.hidden_size),
    )


def _tra_step(tra, x, hidden):
    energy = x.pow(2).mean(dim=-1)
    attention, hidden = tra.att_gru(energy.transpose(1, 2), hidden)
    attention = tra.att_act(tra.att_fc(attention)).transpose(1, 2)
    return x * attention[..., None], hidden


def _conv_depth_step(conv, x, cache):
    combined = torch.cat((cache, x), dim=2)
    output = conv(combined)
    return output, combined[:, :, x.shape[2]:]


def _deconv_depth_step(deconv, x, cache):
    """Causal stride-one depthwise ConvTranspose2d as an ordinary Conv2d."""
    combined = torch.cat((cache, x), dim=2)
    next_cache = combined[:, :, x.shape[2]:]
    kt, kf = deconv.kernel_size
    _, df = deconv.dilation
    frequency_pad = (kf - 1) * df - deconv.padding[1]
    if frequency_pad:
        combined = F.pad(combined, (frequency_pad, frequency_pad, 0, 0))
    weight = deconv.weight.flip(-2, -1)
    output = F.conv2d(
        combined, weight, deconv.bias, stride=(1, 1), padding=0,
        dilation=deconv.dilation, groups=deconv.groups)
    return output, next_cache


def _gt_step(block, x, conv_cache, tra_cache, decoder=False):
    first, residual = torch.chunk(x, chunks=2, dim=1)
    first = block.sfe(first)
    first = block.point_act(block.point_bn1(block.point_conv1(first)))
    if decoder:
        first, conv_cache = _deconv_depth_step(
            block.depth_conv, first, conv_cache)
    else:
        first, conv_cache = _conv_depth_step(
            block.depth_conv, first, conv_cache)
    first = block.depth_act(block.depth_bn(first))
    first = block.point_bn2(block.point_conv2(first))
    first, tra_cache = _tra_step(block.tra, first, tra_cache)
    return block.shuffle(first, residual), conv_cache, tra_cache


def _dp_step(dp, x, inter_cache):
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
    inter, inter_cache = dp.inter_rnn(inter, inter_cache)
    inter = dp.inter_fc(inter).reshape(
        batch, dp.width, 1, dp.hidden_size).permute(0, 2, 1, 3)
    return (intra_out + dp.inter_ln(inter)).permute(0, 3, 1, 2), inter_cache


class StreamGTCRN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Per-block slices of the shared temporal cache, from each GT block's
        # own (kernel-1)*dilation extent. The decoder mirrors the encoder's
        # dilation stack in reverse, so it reuses the same layout backwards;
        # the assert keeps that assumption loud if the architecture changes.
        encoder_pads = _encoder_pad_sizes(model)
        bounds, start = [], 0
        for pad in encoder_pads:
            bounds.append((start, start + pad))
            start += pad
        decoder_pads = [block.pad_size
                        for block in model.decoder.de_convs[:len(bounds)]]
        assert decoder_pads == encoder_pads[::-1]
        self._encoder_slices = tuple(bounds)
        self._decoder_slices = tuple(reversed(bounds))

    def forward(self, spectrum, conv_cache, tra_cache, inter_cache):
        model = self.model
        real = spectrum[..., 0].permute(0, 2, 1)
        imag = spectrum[..., 1].permute(0, 2, 1)
        magnitude = (real.square() + imag.square() + 1e-12).sqrt()
        x = torch.stack((magnitude, real, imag), dim=1)
        x = model.sfe(model.erb.bm(x))

        skips = []
        for block in model.encoder.en_convs[:2]:
            x = block(x)
            skips.append(x)
        enc_conv = []
        enc_tra = []
        for index, block in enumerate(model.encoder.en_convs[2:]):
            start, end = self._encoder_slices[index]
            x, cache, hidden = _gt_step(
                block, x, conv_cache[0, :, :, start:end],
                tra_cache[0, index])
            enc_conv.append(cache)
            enc_tra.append(hidden)
            skips.append(x)

        x, inter0 = _dp_step(model.dpgrnn1, x, inter_cache[0])
        x, inter1 = _dp_step(model.dpgrnn2, x, inter_cache[1])

        dec_conv_reverse = []
        dec_tra = []
        for index, block in enumerate(model.decoder.de_convs[:3]):
            start, end = self._decoder_slices[index]
            x, cache, hidden = _gt_step(
                block, x + skips[4 - index],
                conv_cache[1, :, :, start:end], tra_cache[1, index],
                decoder=True)
            dec_conv_reverse.append(cache)
            dec_tra.append(hidden)
        for index, block in enumerate(model.decoder.de_convs[3:], start=3):
            x = block(x + skips[4 - index])

        mask = model.erb.bs(x)
        enhanced = model.mask(mask, spectrum.permute(0, 3, 2, 1))
        enhanced = enhanced.permute(0, 3, 2, 1)
        conv_out = torch.stack((
            torch.cat(enc_conv, dim=2),
            torch.cat(tuple(reversed(dec_conv_reverse)), dim=2),
        ), dim=0)
        tra_out = torch.stack((torch.stack(enc_tra), torch.stack(dec_tra)))
        inter_out = torch.stack((inter0, inter1))
        return enhanced, conv_out, tra_out, inter_out
