"""Cross-layer equalization (CLE) of a checkpoint's weights before PTQ.

Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias
Correction" (ICCV 2019, arXiv 1906.04721): a channel one layer writes is
divided by ``s`` and every layer that reads it multiplies that input channel by
``s``. An activation in between must satisfy ``f(s * x) == s * f(x)`` for
``s > 0`` (ReLU, LeakyReLU, PReLU), so the FP32 function is unchanged while the
channels of each weight tensor span more similar ranges, which is what one
8-bit scale per tensor needs. ``s = sqrt(r_written / r_read)`` makes the two
ranges equal; the groups are swept until the scales stop moving.

The solver, the checks and the CLI are shared. A model takes part through a
table (``_MODELS``): its activations, its weights as the converter quantizes
them (BatchNorm folded into its conv, DeepVQE-S's three-vector map folded into
the CCM conv) and its channel groups -- the weights that write a set of
channels and the weights that read it, across skips, residual sums,
concatenations and the GRU input. The graph's state inputs carry the rescaled
channels, so its graph and calibration data must both come from the written
checkpoint (usage: ``DeepVQE_S/README.md``).
"""

import argparse
import copy
import math
import os
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from torch import Tensor, nn

from AIAEC._export_common import (
    alignment_depth,
    file_sha256,
    load_checkpoint_model,
)
from AIAEC._streaming_export import _fold_three_vector_conv, batch_norm_gain
from AIAEC.aiaec_common import three_vector_complex

#: Activations that commute with a positive per-channel scale.
HOMOGENEOUS_ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.PReLU)
#: Largest output change, relative to the output's peak, the CLI accepts.
OUTPUT_TOLERANCE = 1e-4
#: Largest factor a channel ends up scaled by, either way. Without it a
#: near-dead channel (tiny gain, real bias) is scaled up whole, and its
#: activation and the state that holds it grow without bound.
MAX_SCALE = 16.0


class Weight(NamedTuple):
    """One weight tensor as the converter quantizes it.

    ``get()`` returns it and ``put(value)`` writes such a value back into
    ``param``. A channel scale goes into ``param`` (rows with ``bias``, or
    columns), or into ``norm``'s affine for rows written through a
    BatchNorm.
    """

    name: str
    get: Callable[[], Tensor]
    put: Callable[[Tensor], None]
    param: nn.Parameter
    bias: Optional[nn.Parameter] = None
    norm: Optional[nn.BatchNorm2d] = None


class Side(NamedTuple):
    """Where a group's channels sit in one weight: channel ``c`` is the
    slices ``offset + [c * k, (c + 1) * k)`` along ``axis`` (0: rows the
    layer writes, 1: columns it reads)."""

    weight: Weight
    axis: int
    k: int = 1
    offset: int = 0

    def check(self, channels: int) -> None:
        weight = self.weight
        size = weight.get().shape[self.axis]
        end = self.offset + channels * self.k
        # A layer reads part of its input (a concatenation) but writes, or
        # rescales, every output row.
        if end > size or (self.axis == 0 and (self.offset, end) != (0, size)):
            raise ValueError('%s has %d slices on axis %d, the group covers '
                             '%d from %d' % (weight.name, size, self.axis,
                                             channels * self.k, self.offset))
        # scale() works on the parameter (or a norm's affine for rows),
        # ranges() on the quantized view: their slices must line up.
        through_norm = self.axis == 0 and weight.norm is not None
        if not through_norm and weight.param.shape[self.axis] != size:
            raise ValueError('%s: axis %d of the parameter is not the axis '
                             'the graph quantizes' % (weight.name, self.axis))

    def ranges(self, channels: int) -> Tensor:
        slices = self.weight.get().transpose(0, self.axis)[
            self.offset:self.offset + channels * self.k]
        return slices.reshape(channels, -1).abs().amax(1)

    def scale(self, factor: Tensor) -> None:
        """Multiply channel ``c`` by ``factor[c]``."""
        weight = self.weight
        if self.axis == 0 and weight.norm is not None:
            weight.norm.weight.data.mul_(factor)
            weight.norm.bias.data.mul_(factor)
            return
        per_slice = factor.repeat_interleave(self.k)
        shape = [1] * weight.param.dim()
        shape[self.axis] = -1
        if self.axis == 0:
            weight.param.data.mul_(per_slice.reshape(shape))
            if weight.bias is not None:
                weight.bias.data.mul_(per_slice)
        else:
            weight.param.data[:, self.offset:self.offset + per_slice.numel()
                              ].mul_(per_slice.reshape(shape))


class Group(NamedTuple):
    """Channels the ``writers`` produce (divided by s) and the ``readers``
    consume (multiplied by s)."""

    name: str
    channels: int
    writers: List[Side]
    readers: List[Side]

    def ranges(self):
        """``(written, read)``: each channel's largest weight on each side."""
        def largest(sides):
            return torch.stack(
                [side.ranges(self.channels) for side in sides]).amax(0)
        return largest(self.writers), largest(self.readers)


class Result(NamedTuple):
    sweeps: int
    last_change: float        # largest |log s| of the last sweep
    converged: bool
    scales: Dict[str, Tensor]  # each group's channel scale, writers / s


def equalize(groups: List[Group], max_sweeps: int = 100,
             tolerance: float = 1e-6,
             max_scale: float = MAX_SCALE) -> Result:
    """Sweep ``groups`` until no scale moves by more than ``tolerance`` (in
    log) or ``max_sweeps`` is reached. Each channel's total scale stays in
    ``[1 / max_scale, max_scale]``. The function stays exact whether or not
    it converges."""
    scales = {group.name: torch.ones(group.channels) for group in groups}
    sweep, change = 0, 0.0
    for sweep in range(1, max_sweeps + 1):
        change = 0.0
        for group in groups:
            total = scales[group.name]
            written, read = group.ranges()
            s = (written / read).sqrt()
            # A dead channel (no weight on one side) keeps its scale.
            s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s))
            s = (total * s).clamp(1.0 / max_scale, max_scale) / total
            total.mul_(s)
            for side in group.writers:
                side.scale(1.0 / s)
            for side in group.readers:
                side.scale(s)
            change = max(change, float(s.log().abs().max()))
        if change <= tolerance:
            break
    return Result(sweep, change, change <= tolerance, scales)


def at_bound(result: Result, max_scale: float = MAX_SCALE) -> int:
    """Channels whose scale stopped at ``max_scale`` either way."""
    limit = math.log(max_scale) * (1 - 1e-4)
    return sum(int((scale.log().abs() >= limit).sum())
               for scale in result.scales.values())


def int8_round(weight: Tensor, per_row: bool) -> Tensor:
    """Symmetric 8-bit rounding with one scale per tensor or per row
    (output channel)."""
    rows = weight.reshape(weight.shape[0], -1)
    peak = rows.abs().amax(1, keepdim=True) if per_row else rows.abs().max()
    step = torch.where(peak > 0, peak / 127.0, torch.ones_like(peak))
    return ((rows / step).round().clamp(-127, 127) * step).reshape(
        weight.shape)


# ---------------------------------------------------------------------------
# DeepVQE-S
# ---------------------------------------------------------------------------

def _plain(name, param, bias=None) -> Weight:
    return Weight(name, param.detach, lambda value: param.data.copy_(value),
                  param, bias)


def _through_norm(name, conv: nn.Conv2d, norm, phases=1) -> Weight:
    """A bias-free conv followed by ``norm`` (``phases`` conv rows per norm
    channel): the graph carries the norm folded into the conv."""
    def gain():
        return batch_norm_gain(norm).repeat_interleave(phases).reshape(
            -1, 1, 1, 1)

    def put(value):
        scale = gain()
        conv.weight.data.copy_(torch.where(scale != 0, value / scale,
                                           conv.weight.detach()))
    return Weight(name, lambda: conv.weight.detach() * gain(), put,
                  conv.weight, norm=norm)


def _three_vector_folded(name, conv: nn.Conv2d) -> Weight:
    """DeepVQE-S's CCM conv as the graph carries it, with the three-vector
    -> (re, im) map folded in (``_fold_three_vector_conv``: output channel
    ``(s * pairs + p) * 2 + r``). ``put`` goes back through the map's
    pseudo-inverse, which reproduces the value (the map has full row rank)."""
    pairs = conv.out_channels // 6
    unit = three_vector_complex(torch.eye(3, dtype=conv.weight.dtype))
    unfold = torch.linalg.pinv(torch.stack((unit.real, unit.imag)))  # [v, r]
    tail = conv.weight.shape[1:]

    def put(value):
        folded = value.reshape(2, pairs, 2, *tail)
        conv.weight.data.copy_(torch.einsum(
            'vr,spr...->pvs...', unfold, folded).reshape(6 * pairs, *tail))
    return Weight(name,
                  lambda: _fold_three_vector_conv(conv, pairs).weight.detach(),
                  put, conv.weight)


def _deepvqe_weights(model) -> List[Weight]:
    """Every weight of the DeepVQE-S graph (biases stay 32-bit)."""
    def block(name, phases=1):
        layer = getattr(model, name)
        return _through_norm(name, layer.conv.conv, layer.norm, phases)

    def conv(name, layer):
        return _plain(name, layer.weight, layer.bias)

    align = model.align
    return (
        [block(name) for name in ('mic1', 'mic2', 'far1', 'far2', 'mic3',
                                  'mic4', 'res3', 'res2')]
        + [block(name, 2) for name in ('up3', 'up2', 'up1')]
        + [conv('align.' + name, getattr(align, name))
           for name in ('query', 'key', 'value')]
        + [conv('align.score', align.score.conv),
           conv('gru_out', model.gru_out),
           _plain('gru.W_ih', model.gru.weight_ih_l0),
           _plain('gru.W_hh', model.gru.weight_hh_l0)]
        + [conv(name, getattr(model, name))
           for name in ('skip1', 'skip2', 'skip3', 'skip4')]
        + [_three_vector_folded('ccm_up', model.ccm_up.conv.conv)])


def _deepvqe_activations(model):
    return [layer.activation for layer in (
        model.mic1, model.mic2, model.far1, model.far2, model.mic3,
        model.mic4)] + [layer.act for layer in (
            model.up3, model.res3, model.up2, model.res2, model.up1)]


def _deepvqe_groups(model) -> List[Group]:
    weights = {weight.name: weight for weight in _deepvqe_weights(model)}

    def writes(name, k=1):
        return Side(weights[name], 0, k)

    def reads(name, offset=0, k=1):
        return Side(weights[name], 1, k, offset)

    def channels(name):
        return getattr(model, name).norm.num_features

    mic2 = channels('mic2')
    bottleneck = channels('mic4')
    width = model.gru.input_size // bottleneck
    queries = model.align.query.out_channels
    return [
        Group('mic1', channels('mic1'), [writes('mic1')],
              [reads('mic2'), reads('skip1')]),
        Group('mic2', mic2, [writes('mic2')],
              [reads('align.query'), reads('mic3'), reads('skip2')]),
        Group('far1', channels('far1'), [writes('far1')], [reads('far2')]),
        Group('far2', channels('far2'), [writes('far2')],
              [reads('align.key'), reads('align.value')]),
        # The aligned far feature is a softmax-weighted sum of value frames.
        Group('align_value', model.align.value.out_channels,
              [writes('align.value')], [reads('mic3', offset=mic2)]),
        # The logits are sum_f q_h * k_h: query row h down, key row h up.
        Group('align_query_key', queries, [writes('align.query')],
              [Side(weights['align.key'], 0)]),
        Group('mic3', channels('mic3'), [writes('mic3')],
              [reads('mic4'), reads('skip3')]),
        Group('mic4', bottleneck, [writes('mic4')],
              [reads('gru.W_ih', k=width), reads('skip4')]),
        Group('bottleneck', bottleneck,
              [writes('gru_out', width), writes('skip4')], [reads('up3')]),
        Group('decoder3', channels('up3'),
              [writes('up3', 2), writes('skip3'), writes('res3')],
              [reads('res3'), reads('up2')]),
        Group('decoder2', channels('up2'),
              [writes('up2', 2), writes('skip2'), writes('res2')],
              [reads('res2'), reads('up1')]),
        Group('decoder1', channels('up1'), [writes('up1', 2), writes('skip1')],
              [reads('ccm_up')]),
    ]


class _Model(NamedTuple):
    activations: Callable
    groups: Callable
    weights: Callable


_MODELS = {
    'DeepVQE_S': _Model(_deepvqe_activations, _deepvqe_groups,
                        _deepvqe_weights),
}


def model_groups(model_name: str, model) -> List[Group]:
    """``model``'s channel groups, each side checked against its weight;
    refuses an activation that does not commute with a positive scale."""
    if model_name not in _MODELS:
        raise ValueError('cross-layer equalization has no channel table for '
                         '%s (supported: %s)' % (model_name, ', '.join(_MODELS)))
    for activation in _MODELS[model_name].activations(model):
        if not isinstance(activation, HOMOGENEOUS_ACTIVATIONS):
            raise ValueError(
                'cross-layer equalization needs relu, leaky_relu or prelu; '
                'this checkpoint uses %s, which does not commute with a '
                'channel scale' % type(activation).__name__)
    groups = _MODELS[model_name].groups(model)
    for group in groups:
        for side in group.writers + group.readers:
            side.check(group.channels)
    return groups


def quantized_copy(model_name: str, model, per_row: bool):
    """A copy of ``model`` with every weight the graph quantizes rounded to
    8 bits (biases, norms and PReLU slopes kept)."""
    rounded = copy.deepcopy(model)
    with torch.no_grad():
        for weight in _MODELS[model_name].weights(rounded):
            weight.put(int8_round(weight.get(), per_row))
    return rounded


def stream_frames(model) -> int:
    """Frames of the random check stream: past the alignment window."""
    return alignment_depth(model) + 16


def _random_stream(model):
    generator = torch.Generator().manual_seed(0)
    bins = model.grid.n_freqs
    frames = stream_frames(model)

    def spectrum():
        return torch.complex(torch.randn(1, frames, bins, generator=generator),
                             torch.randn(1, frames, bins, generator=generator))
    return spectrum(), spectrum()


def output_difference(reference, model) -> float:
    """Largest difference of the enhanced spectrum and every auxiliary
    tensor, relative to its peak, on the random stream."""
    primary, far = _random_stream(reference)
    with torch.no_grad():
        want, got = reference(primary, far), model(primary, far)
    pairs = [(want.enhanced, got.enhanced)] + [
        (value, got.auxiliary[name]) for name, value in want.auxiliary.items()
        if torch.is_tensor(value)]
    worst = 0.0
    for expected, actual in pairs:
        peak = float(expected.abs().max())
        error = float((actual - expected).abs().max())
        worst = max(worst, error / peak if peak > 0 else error)
    return worst


def output_snr_db(reference, model) -> float:
    """SNR of ``model``'s enhanced spectrum against ``reference``'s on the
    random stream."""
    primary, far = _random_stream(reference)
    with torch.no_grad():
        want = reference(primary, far).enhanced
        noise = (model(primary, far).enhanced - want).abs().square().sum()
    if noise == 0:
        return math.inf
    return float(10.0 * torch.log10(want.abs().square().sum() / noise))


def weight_quantization_snr(model_name, reference, model):
    """Output SNR with 8-bit weights, one scale per tensor and per channel."""
    return {granularity: output_snr_db(
                reference, quantized_copy(model_name, model, per_row))
            for granularity, per_row in (('per_tensor', False),
                                         ('per_channel', True))}


def _module_peaks(model, primary, far):
    peaks = {}

    def hook(name):
        def record(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value) and not torch.is_complex(value):
                peaks[name] = max(peaks.get(name, 0.0),
                                  float(value.detach().abs().max()))
        return record

    handles = [module.register_forward_hook(hook(name))
               for name, module in model.named_modules() if name]
    try:
        with torch.no_grad():
            model(primary, far)
    finally:
        for handle in handles:
            handle.remove()
    return peaks


def peak_growth(reference, model):
    """``[(ratio, module)]``, largest first: how much each module output's
    peak grew on the random stream. A channel CLE scales up raises its
    activation and the state that holds it, and a tensor's quantization
    range follows its largest channel."""
    primary, far = _random_stream(reference)
    before = _module_peaks(reference, primary, far)
    after = _module_peaks(model, primary, far)
    return sorted(((after[name] / peak, name) for name, peak in before.items()
                   if peak > 0), reverse=True)


def main(model_name: str, argv=None) -> None:
    """Run one model's equalization CLI; each model's equalize.py names
    itself."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('-o', '--output', required=True,
                        help='the equalized checkpoint to write')
    args = parser.parse_args(argv)
    if os.path.realpath(args.output) == os.path.realpath(args.checkpoint):
        parser.error('--output must not be the input checkpoint')

    source_sha256 = file_sha256(args.checkpoint)
    reference, _grid = load_checkpoint_model(model_name, args.checkpoint)
    contract = torch.load(args.checkpoint, map_location='cpu',
                          weights_only=False)['contract']
    model = copy.deepcopy(reference)
    try:
        groups = model_groups(model_name, model)
    except ValueError as error:
        parser.error(str(error))
    result = equalize(groups)
    difference = output_difference(reference, model)
    if not difference <= OUTPUT_TOLERANCE:
        raise SystemExit('the equalized model changes the output by %.3g of '
                         'its peak (limit %g); nothing written'
                         % (difference, OUTPUT_TOLERANCE))
    before = weight_quantization_snr(model_name, reference, reference)
    after = weight_quantization_snr(model_name, reference, model)
    growth = peak_growth(reference, model)[:3]
    print('%d sweeps (last scale change %.2g in log); FP32 output difference '
          '%.2g of peak' % (result.sweeps, result.last_change, difference))
    if not result.converged:
        print('warning: the scales were still moving after %d sweeps; the '
              'ranges are only partly matched' % result.sweeps)
    bounded = at_bound(result)
    if bounded:
        print('%d channels held at the x%g scale bound (near-dead channels '
              'that would grow further)' % (bounded, MAX_SCALE))
    print('output SNR with 8-bit weights (random input; the converter rounds '
          'its own way):')
    for granularity in before:
        print('  one scale %-12s %6.1f dB -> %6.1f dB' % (
            granularity.replace('_', ' '), before[granularity],
            after[granularity]))
    print('largest growth of an internal peak (random input; the states '
          'holding it grow alike): ' + ', '.join(
              '%s x%.1f' % (name, ratio) for ratio, name in growth))

    if file_sha256(args.checkpoint) != source_sha256:
        raise SystemExit('%s changed while it was read; nothing written'
                         % args.checkpoint)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    partial = args.output + '.partial'
    torch.save({
        'contract': contract,
        'state_dict': model.state_dict(),
        'cross_layer_equalization': {
            'source_sha256': source_sha256,
            'sweeps': result.sweeps,
            'last_change': result.last_change,
            'converged': result.converged,
            'channels_at_scale_bound': bounded,
            'output_difference': difference,
            'weight_8bit_output_snr_db': {'before': before, 'after': after},
            'largest_peak_growth': [[name, ratio] for ratio, name in growth],
        },
    }, partial)
    os.replace(partial, args.output)
    print(args.output)
