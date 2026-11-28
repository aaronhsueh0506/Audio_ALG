#!/usr/bin/env python3
r"""Choose the calibration patterns a min/max PTQ converter sees.

A calibration artifact holds one array per graph input whose leading axis is
the captured invocation ("frame"): a ``.npz`` (+ ``<stem>.calib.json``), or a
BIN directory with one folder per graph input and one little-endian ``.bin``
file per frame (+ ``manifest.json``).  A BIN file records no shape, so its
frames are read against the input shapes and types of the graph: the one the
report names beside the artifact, else ``<dir>.onnx``, or ``--onnx PATH``;
without a graph, against the shapes ``manifest.json`` records.
The report is optional; without it the tool cannot check the model family
(the Align-ULCNet front-end stamp, a model that needs one contiguous stream),
the source files or which inputs stay unquantised, and says so.

The converter sets each tensor's range from the minimum and maximum it sees,
one range for the 8x8, 16x8 and 16x16 formats alike.  A frame holding an
outlier stretches the range for every other frame; a frame left out has its
values clipped on the device.  Consecutive audio frames are near copies of
each other, so what matters is how many recordings and situations the frames
come from, not how many frames are kept.

A capture run scans every recording itself and writes only the chosen frames
with ``--select`` (see each model's ``inference.py calib``): it holds a few
statistics per frame (a few kB with a graph of a hundred tensors) and the
frames a selection can pick, not every frame; ``--jobs N`` scans N recordings
at a time and gives the same result.  An artifact already captured is analysed
and reduced here:

    python3 AIAEC/DeepVQE_S/inference.py calib --checkpoint ckpt.pth \
        --primary-dir mic/ --far-dir far/ --select 1000 --output calib/sel.npz
    python3 PTQ/calib_analyze.py analyze calib/pool.npz --output calib/sel.npz
    # do the chosen frames cover recordings they have not seen?
    python3 PTQ/calib_analyze.py coverage calib/sel.npz --against calib/held.npz

Both write an HTML report.  The frames kept are every tensor's extreme frames,
then k-means over per-frame statistics for the rest, at most ``--count``
(default 1000).  The tensors inside the graph count too: each frame is
replayed through the float graph with onnxruntime (every frame carries its own
recorded state, as each converter pattern does), the frames holding the
extremes of every internal tensor are kept, and the report shows what a choice
from the inputs alone would have lost.  ``--inputs-only`` skips the replay,
as does a ``.npz`` without a graph beside it.

The report says whether the recordings are enough: the sources are split into
groups, each group in turn is left out, and the report lists how far each
tensor's values on the group left out go past the range the other groups set.
Tensors that reach a Softmax through no activation are marked: a clipped
attention score flattens the top the Softmax picks.  Every tensor is checked
for one channel or position far above the median one, on a typical frame (a
feature scale one per-tensor scale serves poorly) or only on a few frames.
These are the float ranges of the ONNX tensors: what a converter does to a
range afterwards (merged operators, table-based functions, tensors sharing one
scale) is not modelled, and the gates inside a recurrent operator are not
visible.  Recurrent-state inputs (named in the report, else a tensor whose
first frame is all zero) are scored on a symmetric grid under ``--scheme
asym``, since the device needs zero point 0 on them; ``--scheme pow2`` scores
every input on a symmetric grid whose bound is rounded up to a power of two,
the scale of a fixed-point input format.  Frames are streaming snapshots, so a
selection keeps each frame's own recorded state but is no longer one stream:
models that need one contiguous stream are refused, and so is an Align-ULCNet
artifact whose AEC front-end stamp is missing or retired.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AINR_DIR = os.path.join(os.path.dirname(_HERE), 'AINR')  # home of calibration_io
if _AINR_DIR not in sys.path:
    sys.path.insert(0, _AINR_DIR)

from calibration_io import (  # noqa: E402
    calibration_output,
    calibration_report_path,
    sibling_onnx_path,
    write_calibration_artifact,
)

SAMPLE_CAP = 2_000_000       # values per tensor behind the SQNR estimates
SAMPLE_FRAMES = 2000         # frames those values come from
FILL_PER_PATTERN = 2         # k-means candidates kept per pattern of --count
EPS = 1e-12
DEFAULT_COUNT = 1000         # the converter's recommended pattern cap
DEFAULT_BITS = (8, 16)       # activation widths of the 8x8 / 16x8 / 16x16 formats
RANDOM_DRAWS = 10
GROWTH_ORDERS = 20
FOLDS = 5                    # groups of sources, each left out in turn
FOLD_LINES = 8               # tensors a fold conclusion names
FIXED_BAND_RATIO = 8.0
FIXED_BAND_SHARE = 0.5       # frames whose largest index must be the same one
LAYER_SCHEME = 'asym'        # the grid scored for the tensors inside the graph
STEP_BITS = 16               # the width whose step the source table states
LAYER_BAND_LINES = 6         # fixed bands inside the graph written one by one
BLOCK = 256                  # frames whose statistics are taken together
CHART_POINTS = 2000          # bars of a per-frame chart in the report
PROGRESS_FRAMES = 5000


def fill_reserve(count):
    """Frames of smallest key kept as k-means candidates for ``count``."""
    return max(SAMPLE_FRAMES, FILL_PER_PATTERN * count)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _quantisable(specs, policy):
    """Names of ``specs`` the converter quantises, and the others with why."""
    names, skipped = [], {}
    for name, _, dtype in specs:
        if name in policy:
            skipped[name] = policy[name]
        elif not np.issubdtype(dtype, np.floating):
            skipped[name] = 'non-float dtype %s' % dtype
        else:
            names.append(name)
    return names, skipped


def _schemes(names, state, scheme):
    """Grid per tensor: ``scheme``, but state inputs need zero point 0, so
    they take the symmetric grid when ``scheme`` is asymmetric."""
    return {name: 'sym' if name in state and scheme == 'asym' else scheme
            for name in names}


def _contiguous(report):
    """Whether the model needs one uninterrupted stream (no selection)."""
    return report.get('contiguous_state_frames') is not None


def _starts_at_zero(low, high):
    """A tensor whose first frame is all zero: a recurrent state, fresh."""
    return low == 0 and high == 0


class Artifact:
    """Calibration tensors, the report that came with them and their format."""

    def __init__(self, path, fmt, arrays, report):
        self.path = path
        self.fmt = fmt
        self.arrays = arrays
        self.report = report
        self.frames = int(next(iter(arrays.values())).shape[0])
        self.specs = [(name, value.shape[1:], value.dtype)
                      for name, value in arrays.items()]
        self.names, self.skipped = _quantisable(
            self.specs, report.get('state_precision_policy') or {})
        self.contiguous = _contiguous(report)
        recorded = report.get('state_inputs')
        self.state = ([name for name in self.names if name in recorded]
                      if recorded is not None else
                      [name for name in self.names if _starts_at_zero(
                          arrays[name][0].min(), arrays[name][0].max())])

    def schemes(self, scheme):
        return _schemes(self.names, self.state, scheme)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def _npz_report(path):
    report = _read_json(calibration_report_path(path, 'npz'))
    if report:
        return report
    # An archive written before the report had its own name keeps it at
    # <stem>.json; that file is a report only when it says so, since the graph
    # exported with the archive shares the stem.
    legacy = _read_json(os.path.splitext(path)[0] + '.json')
    return legacy if legacy.get('artifact_format') == 'numpy_npz' else {}


def require_current_frontend(report, path):
    """Refuse an Align-ULCNet artifact captured behind another AEC front end.

    The graph inputs of an Align-ULCNet artifact are the unified AEC's analysis
    seam, so frames recorded behind a retired AEC build hold other statistics
    and must not drive range selection. ``inference.py calib`` stamps
    ``linear_aec_contract_version`` and ``aec_behavior_hash`` into the report.
    Other model families carry no AEC stamp and are not looked at. The AEC
    modules load only here, which keeps the tool light for DFN / AINR.
    """
    if report.get('model_family') != 'Align_ULCNet':
        return
    root = os.path.dirname(_HERE)
    if root not in sys.path:
        sys.path.insert(0, root)
    from AIAEC.dataset_gen.linear_aec import (
        LINEAR_AEC_CONTRACT_VERSION,
        RETIRED_BEHAVIOR_HASHES,
    )
    version = report.get('linear_aec_contract_version')
    behavior_hash = report.get('aec_behavior_hash')
    if not version or not behavior_hash:
        problem = ('carries no AEC front-end stamp (linear_aec_contract_version'
                   ' / aec_behavior_hash)')
    elif version != LINEAR_AEC_CONTRACT_VERSION:
        problem = ('was captured with front-end contract %r, this build uses '
                   '%r' % (version, LINEAR_AEC_CONTRACT_VERSION))
    elif behavior_hash in RETIRED_BEHAVIOR_HASHES:
        problem = 'was captured with retired AEC behaviour hash %s' % (
            behavior_hash)
    else:
        return
    raise ValueError('%s %s; regenerate it with inference.py calib' % (
        path, problem))


def _load_graph(path):
    import onnx

    try:
        return onnx.load(path)
    except Exception as error:      # protobuf raises its own decode error
        raise ValueError('cannot read the graph %s: %s' % (path, error))


def _graph_inputs(model):
    """(name, dims, dtype) of each graph input that is not an initializer."""
    from onnx import helper

    constant = {init.name for init in model.graph.initializer}
    out = []
    for value in model.graph.input:
        if value.name in constant:
            continue
        kind = value.type.tensor_type
        dims = [dim.dim_value if dim.HasField('dim_value') else None
                for dim in kind.shape.dim]
        out.append((value.name, dims,
                    np.dtype(helper.tensor_dtype_to_np_dtype(kind.elem_type))))
    return out


def _graph_file(path, report, override=None):
    """The graph an artifact at ``path`` binds to: ``override``, else the one
    its calibration run exported beside it (named in the report, or
    ``<stem>.onnx``). It may not exist."""
    if override:
        return os.path.abspath(override)
    named = report.get('graph')
    beside = os.path.dirname(path.rstrip(os.sep))
    return os.path.abspath(os.path.join(beside, os.path.basename(named))
                           if named else sibling_onnx_path(path))


def graph_path(art, override=None):
    path = _graph_file(art.path, art.report, override)
    if not os.path.isfile(path):
        raise ValueError('no graph at %s; pass --onnx PATH' % path)
    return path


def _natural_key(name):
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', name)]


def _read_bin(path, report, graph, threads=1):
    """The frames of a BIN directory, shaped by the inputs of ``graph`` or,
    without one, by what ``manifest.json`` records."""
    recorded = report.get('binary_tensors') or {}
    if os.path.isfile(graph):
        inputs = _graph_inputs(_load_graph(graph))
    elif recorded:
        inputs = [(name, list(spec['frame_shape']),
                   np.dtype(spec['dtype']).newbyteorder('='))
                  for name, spec in recorded.items()]
    else:
        raise ValueError('%s is a BIN directory, whose files record no shape, '
                         'and has no manifest.json; no graph to read the input '
                         'shapes from at %s: pass --onnx PATH' % (path, graph))
    folders = []
    for name, dims, dtype in inputs:
        # The name becomes a path component: reject absolute or ".."-containing
        # names, as the AINR/calibration_io.py writer does.
        if name != os.path.basename(name) or name in ('.', '..'):
            raise ValueError('graph input %r cannot name a folder' % name)
        if any(dim is None or dim <= 0 for dim in dims):
            raise ValueError('%s input %s has no fixed shape (%s)' % (
                graph, name, dims))
        shape = tuple(dims)
        spec = recorded.get(name)
        if spec is not None and (
                tuple(spec['frame_shape']) != shape
                or np.dtype(spec['dtype']).newbyteorder('=') != dtype):
            raise ValueError('manifest.json records %s as %s %s, the graph as '
                             '%s %s' % (name, spec['dtype'], spec['frame_shape'],
                                        dtype, list(shape)))
        folder = os.path.join(path, name)
        if not os.path.isdir(folder):
            raise ValueError('%s has no folder for graph input %s' % (
                path, name))
        files = sorted((f for f in os.listdir(folder) if f.endswith('.bin')),
                       key=_natural_key)
        folders.append((name, folder, files, shape, dtype))
    counts = {name: len(files) for name, _, files, _, _ in folders}
    if len(set(counts.values())) > 1:
        raise ValueError('the input folders of %s hold different frame counts: '
                         '%s' % (path, counts))
    frames = next(iter(counts.values()), 0)
    if not frames:
        raise ValueError('%s holds no frame' % path)
    if report.get('frames') is not None and int(report['frames']) != frames:
        raise ValueError('manifest.json records %s frames, the folders hold %d'
                         % (report['frames'], frames))
    arrays = {name: np.empty((frames,) + shape, dtype=dtype)
              for name, _, _, shape, dtype in folders}

    def read(item):
        (name, folder, files, shape, dtype), indices = item
        size = int(np.prod(shape)) * dtype.itemsize
        for index in indices:
            file = os.path.join(folder, files[index])
            if os.path.getsize(file) != size:
                raise ValueError('%s is %d bytes; %s %s %s needs %d' % (
                    file, os.path.getsize(file), name, dtype, list(shape),
                    size))
            arrays[name][index] = np.fromfile(
                file, dtype=dtype.newbyteorder('<')).reshape(shape)

    work = [(folder, range(a, min(a + BLOCK, frames)))
            for folder in folders for a in range(0, frames, BLOCK)]
    with concurrent.futures.ThreadPoolExecutor(max(1, threads)) as pool:
        list(pool.map(read, work))
    return arrays


def load_artifact(path, graph=None, threads=1):
    """An artifact; ``graph`` overrides the one beside it (BIN only reads its
    input shapes from it)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        report = _read_json(os.path.join(path, 'manifest.json'))
        require_current_frontend(report, path)
        arrays = _read_bin(path, report, _graph_file(path, report, graph),
                           threads)
        return Artifact(path, 'bin', arrays, report)
    if path.endswith('.npz') and os.path.isfile(path):
        report = _npz_report(path)
        require_current_frontend(report, path)
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        return Artifact(path, 'npz', arrays, report)
    raise ValueError('expected a .npz file or a BIN directory: %s' % path)


def detect_segments(art):
    """Frame indices where a new source file began, plus a reliability flag.

    A tensor whose first frame is exactly zero is treated as state; a boundary
    is a frame where every one of them is exactly zero again.  Trusted only if
    the count equals the number of recorded source files.
    """
    n = art.frames
    if not art.state:
        return [0], False, 'no tensor starts from an all-zero state'
    zero = np.ones(n, dtype=bool)
    for name in art.state:
        zero &= ~art.arrays[name].reshape(n, -1).any(axis=1)
    starts = [int(i) for i in np.flatnonzero(zero)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    sources = art.report.get('source_files') or []
    reliable = len(starts) == len(sources)
    note = ('%d state resets match %d source files' % (len(starts), len(sources))
            if reliable else
            '%d state resets but %d source files' % (len(starts), len(sources)))
    return starts, reliable, note


# --------------------------------------------------------------------------
# Per-index concentration
# --------------------------------------------------------------------------

def _loo_extreme(per_unit, take_max):
    """Global max (or min) with each unit removed, via the top-2 trick."""
    order = np.argsort(per_unit)
    if take_max:
        best, second = order[-1], order[-2] if len(order) > 1 else order[-1]
    else:
        best, second = order[0], order[1] if len(order) > 1 else order[0]
    out = np.full(per_unit.shape, per_unit[best], dtype=np.float64)
    out[best] = per_unit[second]
    return out


def _band_axes(shape):
    """The axes of a one-frame tensor ``shape`` whose indices are compared:
    the channel (axis 1 of a rank >= 3 tensor) and the last axis."""
    axes = {}
    if len(shape) >= 3 and shape[1] > 1:
        axes['channel'] = 1
    if len(shape) >= 1 and shape[-1] > 1:
        axes['last'] = len(shape) - 1
    return axes


def _index_peaks(x, axis):
    """Absolute maximum per index along ``axis``, the other axes reduced
    except the leading one: [frames, size] for a stack of frames, [size] for
    one frame read with a leading batch axis."""
    other = tuple(a for a in range(1, x.ndim) if a != axis % x.ndim)
    return np.maximum(x.max(axis=other), -x.min(axis=other)).astype(np.float64)


def _frame_peaks(peaks):
    """Per frame of ``peaks`` [frames, size]: the largest index's peak over
    the median non-zero index's (NaN on an all-zero frame, a fresh state) and
    which index is largest."""
    top = peaks.max(axis=1)
    ratio = np.full(top.shape, np.nan)
    live = top > 0
    if live.any():
        # Peaks are absolute values, so a sorted row starts with its zeros.
        rows = np.sort(peaks[live], axis=1)
        zeros = np.count_nonzero(rows <= 0, axis=1)
        count = rows.shape[1] - zeros
        at = np.arange(len(rows))
        median = 0.5 * (rows[at, zeros + (count - 1) // 2]
                        + rows[at, zeros + count // 2])
        ratio[live] = top[live] / (median + EPS)
    return ratio, peaks.argmax(axis=1)


def _concentration(peak, frame_ratio, frame_top):
    """How much a single index dominates. ``ratio`` compares the indices'
    peaks over every frame (``argmax`` is the largest), so one loud frame can
    set it; ``typical`` is the median over the frames of the same comparison
    inside one frame, ``frame_top`` the index most frames have largest and
    ``frame_share`` their share. The last three leave out all-zero frames.
    An index at zero needs no steps, so the medians leave zeros out: a
    rectifier with most channels off would otherwise divide by zero.
    ``top_shares`` lists the (at most three) indices most often largest on a
    frame, with their shares."""
    live = ~np.isnan(frame_ratio)
    tops = frame_top[live]
    shares = (np.bincount(tops, minlength=peak.size) / tops.size if tops.size
              else np.zeros(peak.size))
    mode = int(shares.argmax()) if tops.size else int(peak.argmax())
    leaders = np.argsort(-shares, kind='stable')[:3]
    return {
        'size': int(peak.size),
        'argmax': int(peak.argmax()),
        'ratio': (float(peak.max() / (np.median(peak[peak > 0]) + EPS))
                  if peak.max() > 0 else 0.0),
        'typical': float(np.median(frame_ratio[live])) if tops.size else 0.0,
        'frame_top': mode,
        'frame_share': float(shares[mode]) if tops.size else 0.0,
        'top_shares': [[int(i), float(shares[i])] for i in leaders
                       if shares[i] > 0],
    }


class _Bands:
    """``_concentration`` per axis of ``_band_axes(shape)``, gathered a block
    of frames at a time. Each frame keeps its ratio (float32) and largest
    index (the smallest integer type that holds it): a pool of a million
    frames keeps a few bytes per frame and axis."""

    def __init__(self, shape):
        self.axes = _band_axes(shape)
        self.peak = {key: np.zeros(shape[axis])
                     for key, axis in self.axes.items()}
        self.index = {key: np.min_scalar_type(shape[axis])
                      for key, axis in self.axes.items()}
        self.ratio = {key: [] for key in self.axes}
        self.top = {key: [] for key in self.axes}

    def add_peaks(self, key, peaks):
        np.maximum(self.peak[key], peaks.max(axis=0), out=self.peak[key])
        ratio, top = _frame_peaks(peaks)
        self.ratio[key].append(ratio.astype(np.float32))
        self.top[key].append(top.astype(self.index[key]))

    def add(self, block):
        for key, axis in self.axes.items():
            self.add_peaks(key, _index_peaks(block, axis + 1))

    def state(self):
        return {key: {'peak': self.peak[key],
                      'ratio': np.concatenate(self.ratio[key]),
                      'top': np.concatenate(self.top[key])}
                for key in self.axes}


def _merged_bands(states):
    """``_concentration`` per axis of per-part ``_Bands.state()`` lists (it
    does not depend on the frames' order)."""
    out = {}
    for key in states[0]:
        peak = np.max([s[key]['peak'] for s in states], axis=0)
        ratio = np.concatenate([s[key]['ratio'] for s in states])
        top = np.concatenate([s[key]['top'] for s in states]).astype(np.int64)
        out[key] = _concentration(peak, ratio, top)
    return out


# --------------------------------------------------------------------------
# Graph replay: the tensors inside the graph
# --------------------------------------------------------------------------

# Operators whose output is not followed: a layout-only one repeats its
# input, a Constant never changes.
UNFOLLOWED_OPS = frozenset(('Reshape', 'Transpose', 'Squeeze', 'Unsqueeze',
                            'Flatten', 'Identity', 'Constant'))
# Operators that bend their input: walking back from a Softmax stops here.
ACTIVATION_OPS = frozenset(('Relu', 'PRelu', 'LeakyRelu', 'Elu', 'Selu',
                            'Celu', 'Gelu', 'Sigmoid', 'HardSigmoid', 'Tanh',
                            'Softmax', 'LogSoftmax', 'Softplus', 'Softsign',
                            'HardSwish', 'Mish', 'Clip', 'Exp', 'Log'))

# Recorded by the exporter in the graph and by the capture in the report.
_GRAPH_PAIR_KEYS = ('checkpoint_sha256', 'state_layout_version')


def _check_graph(model, path, specs, report):
    """Refuse a graph whose inputs are not the frames' tensors (``specs``:
    name -> (one-frame shape, dtype)); return its input names and whether a
    checkpoint hash on both sides confirms they belong together."""
    inputs = _graph_inputs(model)
    names = [name for name, _, _ in inputs]
    if sorted(names) != sorted(specs):
        raise ValueError('%s takes inputs %s but the artifact holds %s' % (
            path, sorted(names), sorted(specs)))
    for name, dims, dtype in inputs:
        have, have_dtype = specs[name]
        if len(dims) != len(have) or any(
                want is not None and want != got
                for want, got in zip(dims, have)):
            raise ValueError('%s input %s is %s but the artifact holds %s per '
                             'frame' % (path, name, dims, list(have)))
        if dtype != have_dtype:
            raise ValueError('%s input %s is %s but the artifact holds %s' % (
                path, name, dtype, have_dtype))
    props = {prop.key: prop.value for prop in model.metadata_props}
    for key in _GRAPH_PAIR_KEYS:
        want, have = props.get(key), report.get(key)
        if want and have is not None and str(want) != str(have):
            raise ValueError('%s records %s %s, the artifact %s' % (
                path, key, str(want)[:12], str(have)[:12]))
    return names, bool(props.get('checkpoint_sha256')
                       and report.get('checkpoint_sha256'))


def softmax_path(graph):
    """Tensors (and graph inputs) from which a Softmax input is reached
    through no activation: a value clipped there flattens what the Softmax
    compares. The walk back stops at an activation's output."""
    producer = {name: node for node in graph.node for name in node.output}
    seen = set()
    stack = [node.input[0] for node in graph.node
             if node.op_type == 'Softmax' and node.input]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        node = producer.get(name)
        if node is not None and node.op_type not in ACTIVATION_OPS:
            stack.extend(item for item in node.input if item)
    return seen


def _check_at(index):
    """Frames compared against the unchanged graph: 0, 1, 2, 4, 8, ..."""
    return index == 0 or index & (index - 1) == 0


class _Replay:
    """Per-frame minimum and maximum of every tensor inside a graph.

    Each frame carries its own recorded state inputs, so one frame is one
    complete invocation and is replayed on its own, as the converter does with
    each pattern. A tensor is the float output of a node outside
    ``UNFOLLOWED_OPS``; its operands are the followed tensors or graph inputs
    its node reads, traced back through the layout-only operators. Frames 0,
    1, 2, 4, ... and the last are also run through the unchanged graph, whose
    outputs must match. ``threads`` sessions share a block's frames."""

    def __init__(self, path, specs, report, bands, threads=1):
        import onnx
        import onnxruntime as ort

        model = _load_graph(path)
        self.path = path
        self.feed, self.confirmed = _check_graph(
            model, path, {name: (shape, dtype) for name, shape, dtype in specs},
            report)
        self.graph = model.graph
        self.heads = [value.name for value in model.graph.output]
        self.ops = {name: node.op_type for node in model.graph.node
                    if node.op_type not in UNFOLLOWED_OPS
                    for name in node.output if name}
        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        probe.graph.output.extend(onnx.ValueInfoProto(name=name)
                                  for name in self.ops if name not in self.heads)
        options = ort.SessionOptions()
        options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._make = [lambda proto=proto: ort.InferenceSession(
            proto, options, providers=['CPUExecutionProvider'])
            for proto in (model.SerializeToString(), probe.SerializeToString())]
        self.threads = max(1, threads)
        self._plain = None          # the unchanged graph, for the comparison
        self._sessions = []         # one probe session per thread
        self.bands = bands
        self.tensors = None
        self.seen = 0
        self._last = None
        self._fmin, self._fmax = [], []

    def _probe(self, k):
        while len(self._sessions) <= k:
            self._sessions.append(self._make[1]())
        return self._sessions[k]

    def _start(self, feed):
        first = self._probe(0).run(list(self.ops), feed)
        self.tensors = [name for name, value in zip(self.ops, first)
                        if isinstance(value, np.ndarray) and value.size
                        and np.issubdtype(value.dtype, np.floating)]
        if not self.tensors:
            raise ValueError('%s has no float tensor to follow' % self.path)
        # Every head is fetched, followed or not, for the comparison.
        self.fetch = self.tensors + [name for name in self.heads
                                     if name not in self.tensors]
        self.head_at = [self.fetch.index(name) for name in self.heads]
        self._shape = {name: value.shape for name, value in zip(self.ops, first)}
        self._bands = {col: _Bands(self._shape[name])
                       for col, name in enumerate(self.tensors)} \
            if self.bands else {}
        self.band = [(col, key, axis) for col, bands in self._bands.items()
                     for key, axis in bands.axes.items()]
        # Per tensor: (band row, the axes its peaks reduce); the first gives
        # the tensor's minimum and maximum too.
        self._reduce = {}
        for i, (col, _, axis) in enumerate(self.band):
            rank = len(self._shape[self.tensors[col]])
            self._reduce.setdefault(col, []).append(
                (i, tuple(a for a in range(rank) if a != axis)))

    def _compare(self, feed, values, index):
        if self._plain is None:
            self._plain = self._make[0]()
        for at, want in zip(self.head_at, self._plain.run(self.heads, feed)):
            if not np.allclose(values[at], want, rtol=1e-5, atol=1e-6,
                               equal_nan=True):
                raise RuntimeError(
                    '%s: following the internal tensors changed output %s '
                    'at frame %d' % (self.path, self.fetch[at], index))

    def run(self, block, n):
        """Replay the ``n`` frames of ``block``; their [n, tensors] minima
        and maxima."""
        feeds = [{name: block[name][j] for name in self.feed}
                 for j in range(n)]
        if self.tensors is None:
            self._start(feeds[0])
        width = len(self.tensors)
        fmin = np.empty((n, width), dtype=np.float32)
        fmax = np.empty((n, width), dtype=np.float32)
        peaks = [np.empty((n, self._shape[self.tensors[col]][axis]))
                 for col, _, axis in self.band]
        last = [None]

        def work(k, rows):
            run = self._probe(k)
            for j in rows:
                values = run.run(self.fetch, feeds[j])
                for col in range(width):
                    value = values[col]
                    reduce = self._reduce.get(col)
                    if not reduce:
                        fmin[j, col] = value.min()
                        fmax[j, col] = value.max()
                        continue
                    for count, (i, other) in enumerate(reduce):
                        high, low = value.max(axis=other), value.min(axis=other)
                        peaks[i][j] = np.maximum(high, -low)
                        if not count:
                            fmin[j, col] = low.min()
                            fmax[j, col] = high.max()
                if _check_at(self.seen + j):
                    self._compare(feeds[j], values, self.seen + j)
                if j == n - 1:
                    last[0] = {at: values[at] for at in self.head_at}

        threads = min(self.threads, n)
        if threads == 1:
            work(0, range(n))
        else:
            for k in range(threads):
                self._probe(k)       # sessions are made here, not in threads
            bounds = np.linspace(0, n, threads + 1).astype(int)
            with concurrent.futures.ThreadPoolExecutor(threads) as pool:
                list(pool.map(lambda k: work(k, range(bounds[k],
                                                       bounds[k + 1])),
                              range(threads)))
        for i, (col, key, _) in enumerate(self.band):
            self._bands[col].add_peaks(key, peaks[i])
        self._last = ({name: np.array(value)
                       for name, value in feeds[-1].items()}, last[0],
                      self.seen + n - 1)
        self.seen += n
        self._fmin.append(fmin)
        self._fmax.append(fmax)
        return fmin, fmax

    def finish(self):
        feed, heads, index = self._last
        if not _check_at(index):
            self._compare(feed, heads, index)
        fmin, fmax = np.concatenate(self._fmin), np.concatenate(self._fmax)
        self._fmin = self._fmax = None
        bad = ~(np.isfinite(fmin).all(axis=0) & np.isfinite(fmax).all(axis=0))
        if bad.any():
            raise ValueError('%s: non-finite values in %s' % (
                self.path, ', '.join(np.array(self.tensors)[bad][:5])))
        producer = {name: node for node in self.graph.node
                    for name in node.output}
        known = set(self.tensors) | set(self.feed)

        def operand(name):
            while name in producer and \
                    producer[name].op_type in UNFOLLOWED_OPS \
                    and producer[name].input:
                name = producer[name].input[0]
            return name

        return {'graph': self.path, 'checkpoint_confirmed': self.confirmed,
                'names': list(self.tensors),
                'ops': [self.ops[name] for name in self.tensors],
                'operands': [[source for source in map(operand,
                                                       producer[name].input)
                              if source in known] for name in self.tensors],
                'softmax': sorted(softmax_path(self.graph) & known),
                'frame_min': fmin, 'frame_max': fmax,
                'bands': {name: self._bands[col].state()
                          for col, name in enumerate(self.tensors)}
                if self.bands else None}


# --------------------------------------------------------------------------
# Scan: the statistics of a stream of frames
# --------------------------------------------------------------------------

def _mix(x):
    """The splitmix64 finaliser, on a uint64 array."""
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _frame_keys(seed, columns):
    """A pseudo-random 64-bit key per frame from its content (``columns``:
    per-frame float64 statistics of its inputs). The frames of smallest key
    are the value sample and the k-means candidates: keyed by content, they
    are the same however the frames' sources were numbered and the frames
    split between processes. Equal frames get equal keys (``_distinct``)."""
    with np.errstate(over='ignore'):
        x = np.full(len(columns[0]),
                    _mix(np.array([seed], dtype=np.uint64))[0])
        for column in columns:
            x ^= np.ascontiguousarray(column, dtype=np.float64).view(np.uint64)
            x = _mix(x)
    return x


def _distinct(ranked, keys):
    """``ranked`` (frame indices, by key) without the frames whose key an
    earlier one has: equal frames, such as a stretch of digital silence, take
    one place among the frames of smallest key."""
    if ranked.size < 2:
        return ranked
    ordered = keys[ranked]
    return ranked[np.concatenate(([True], ordered[1:] != ordered[:-1]))]


_FRAME_FIELDS = ('fmin', 'fmax', 'mean', 'std', 'peak')


class Scan:
    """Statistics of calibration frames, gathered a block at a time.

    ``specs`` lists every graph input as (name, one-frame shape, dtype);
    ``policy`` names the inputs left unquantised. With ``graph`` (a path)
    each frame is also replayed through the float graph on ``threads``
    sessions. A frame's place is (source, frame within the source): frames
    are added in that order, and ``assemble`` sorts the parts of separate
    scans by it, so scans of separate sources merge into the scan of all.
    ``reserve`` > 0 keeps the frames a selection can pick (each tensor's
    extreme frame and the ``reserve`` frames of smallest key) for a caller
    whose frames do not stay addressable."""

    def __init__(self, specs, policy=None, report=None, graph=None,
                 bands=True, reserve=0, seed=0, threads=1, progress=None):
        self.specs = [(name, tuple(shape), np.dtype(dtype))
                      for name, shape, dtype in specs]
        self.names, self.skipped = _quantisable(self.specs, policy or {})
        if not self.names:
            raise ValueError('no float input to calibrate')
        self.bands = bands
        self.reserve = reserve
        self.seed = seed
        self.progress = progress
        self.replay = (_Replay(graph, self.specs, report or {}, bands, threads)
                       if graph else None)
        self.count = 0
        self._src, self._local, self._keys = [], [], []
        self._rows = {name: {field: [] for field in _FRAME_FIELDS}
                      for name in self.names}
        self._bands = {name: _Bands(shape) for name, shape, _ in self.specs
                       if name in self.names} if bands else {}
        # Every value while a tensor's total fits SAMPLE_CAP.
        self._values = {name: [] for name in self.names}
        self._block = None          # the frames ``add`` has not processed
        self._fill = 0
        self._block_at = np.empty((BLOCK, 2), dtype=np.int64)
        self._next = {}
        self._hold = None
        self._kept = {}
        self._reserve_keys = np.empty(0, dtype=np.uint64)
        self._reserve_at = np.empty((0, 2), dtype=np.int64)

    def add(self, frame, source=0):
        """One invocation's arrays (name -> one-frame array) of ``source``,
        copied into the current block."""
        if self._block is None:
            self._block = {name: np.empty((BLOCK,) + shape, dtype=dtype)
                           for name, shape, dtype in self.specs}
        local = self._next.get(source, 0)
        self._next[source] = local + 1
        for name, value in self._block.items():
            value[self._fill] = frame[name]
        self._block_at[self._fill] = source, local
        self._fill += 1
        if self._fill == BLOCK:
            self.flush()

    def flush(self):
        n, self._fill = self._fill, 0
        if n:
            at = self._block_at[:n].copy()
            self.process({name: value[:n] for name, value in self._block.items()},
                         at[:, 0], at[:, 1])

    def process(self, block, src, local):
        """The frames of ``block`` (name -> [frames, ...]), whose places are
        ``src`` and ``local``."""
        n = len(src)
        low, high, content = [], [], []
        for name in self.names:
            flat = block[name].reshape(n, -1)
            lo = flat.min(axis=1).astype(np.float64)
            hi = flat.max(axis=1).astype(np.float64)
            if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
                raise ValueError('non-finite calibration input in %s' % name)
            wide = flat.astype(np.float64)
            rows = self._rows[name]
            rows['fmin'].append(lo)
            rows['fmax'].append(hi)
            rows['mean'].append(wide.mean(axis=1))
            rows['std'].append(wide.std(axis=1))
            rows['peak'].append(np.maximum(-lo, hi))
            low.append(lo)
            high.append(hi)
            content += [lo, hi, rows['mean'][-1]]
            if self.bands:
                self._bands[name].add(block[name])
            values = self._values[name]
            if values is not None:
                if (self.count + n) * flat.shape[1] > SAMPLE_CAP:
                    self._values[name] = None
                else:
                    values.append(np.array(flat))
        if self.replay is not None:
            lmin, lmax = self.replay.run(block, n)
            low.extend(lmin.T.astype(np.float64))
            high.extend(lmax.T.astype(np.float64))
        keys = _frame_keys(self.seed, content)
        self._src.append(src)
        self._local.append(local)
        self._keys.append(keys)
        if self.reserve > 0:
            self._keep(np.stack(low, axis=1), np.stack(high, axis=1), src,
                       local, keys, block)
        done = self.count
        self.count += n
        if self.progress and (self.count // PROGRESS_FRAMES
                              > done // PROGRESS_FRAMES):
            self.progress(self.count)

    def _keep(self, low, high, src, local, keys, block):
        n, width = low.shape
        at = np.stack([src, local], axis=1)
        if self._hold is None:
            self._hold = [np.full(width, np.inf), np.full(width, -np.inf),
                          np.zeros((width, 2), np.int64),
                          np.zeros((width, 2), np.int64)]
        lo, hi, lo_at, hi_at = self._hold
        cols = np.arange(width)
        first = low.argmin(axis=0)          # the first frame on a tie
        better = low[first, cols] < lo
        lo[better] = low[first[better], cols[better]]
        lo_at[better] = at[first[better]]
        first = high.argmax(axis=0)
        better = high[first, cols] > hi
        hi[better] = high[first[better], cols[better]]
        hi_at[better] = at[first[better]]
        every = np.concatenate([self._reserve_keys, keys])
        places = np.concatenate([self._reserve_at, at])
        take = _distinct(np.argsort(every, kind='stable'),
                         every)[:self.reserve]
        self._reserve_keys, self._reserve_at = every[take], places[take]
        need = {tuple(p) for p in np.concatenate([lo_at, hi_at,
                                                  self._reserve_at]).tolist()}
        # While the reserve holds every frame seen, every frame is kept, as
        # ``assemble`` makes them all candidates then.
        whole = self.count + n <= self.reserve
        for j in range(n):
            place = (int(src[j]), int(local[j]))
            if (whole or place in need) and place not in self._kept:
                # Copied: the block's storage is reused.
                self._kept[place] = {name: np.array(block[name][j])
                                     for name in block}
        if not whole:
            for place in [p for p in self._kept if p not in need]:
                del self._kept[place]

    def part(self):
        """This scan's frames and statistics, for ``assemble``; the scan
        hands them over and keeps none."""
        self.flush()
        if not self.count:
            raise ValueError('no calibration frame was scanned')
        kept = sorted(self._kept)
        part = {
            'specs': self.specs,
            'src': np.concatenate(self._src),
            'local': np.concatenate(self._local),
            'keys': np.concatenate(self._keys),
            'rows': {name: {field: np.concatenate(values)
                            for field, values in rows.items()}
                     for name, rows in self._rows.items()},
            'bands': {name: bands.state()
                      for name, bands in self._bands.items()},
            'values': {name: np.concatenate(values) if values is not None
                       else None for name, values in self._values.items()},
            'layers': self.replay.finish() if self.replay else None,
            'kept_at': np.array(kept, dtype=np.int64).reshape(-1, 2),
            # Each input's frames leave the dict as they are stacked.
            'kept': {name: np.stack([self._kept[p].pop(name) for p in kept])
                     if kept else None for name, _, _ in self.specs},
        }
        self._kept = {}
        self._src = self._local = self._keys = self._rows = None
        self._values = self._block = None
        self._bands = {}
        return part


_PART_SCALARS = ('graph', 'checkpoint_confirmed', 'names', 'ops', 'operands',
                 'softmax')


def save_part(part, folder):
    """Write a ``Scan.part`` to ``folder`` (a process hands it back)."""
    os.makedirs(folder, exist_ok=True)
    arrays, meta = {}, {'rows': {}, 'bands': {}, 'kept': [], 'layers': None,
                        'specs': [[name, list(shape), np.dtype(dtype).str]
                                  for name, shape, dtype in part['specs']]}
    for key in ('src', 'local', 'keys', 'kept_at'):
        arrays[key] = part[key]
    for name, rows in part['rows'].items():
        meta['rows'][name] = list(rows)
        for field, value in rows.items():
            arrays['r/%s/%s' % (name, field)] = value
        if part['values'][name] is not None:
            arrays['v/%s' % name] = part['values'][name]

    def bands(prefix, states, into):
        for name, axes in states.items():
            into[name] = list(axes)
            for axis, state in axes.items():
                for field, value in state.items():
                    arrays['%s/%s/%s/%s' % (prefix, name, axis, field)] = value

    bands('b', part['bands'], meta['bands'])
    layers = part['layers']
    if layers is not None:
        meta['layers'] = {key: layers[key] for key in _PART_SCALARS}
        meta['layers']['bands'] = {} if layers['bands'] is not None else None
        arrays['l/frame_min'] = layers['frame_min']
        arrays['l/frame_max'] = layers['frame_max']
        if layers['bands'] is not None:
            bands('lb', layers['bands'], meta['layers']['bands'])
    for index, (name, value) in enumerate(part['kept'].items()):
        meta['kept'].append(name)
        if value is not None:
            np.save(os.path.join(folder, 'kept_%d.npy' % index), value)
    np.savez(os.path.join(folder, 'part.npz'), **arrays)
    with open(os.path.join(folder, 'part.json'), 'w', encoding='utf-8') as out:
        json.dump(meta, out)


def load_part(folder):
    """A part ``save_part`` wrote; kept frames are memory-mapped."""
    with open(os.path.join(folder, 'part.json'), encoding='utf-8') as stream:
        meta = json.load(stream)
    with np.load(os.path.join(folder, 'part.npz')) as data:
        arrays = {key: data[key] for key in data.files}

    def bands(prefix, names):
        return {name: {axis: {field: arrays['%s/%s/%s/%s' % (
            prefix, name, axis, field)] for field in ('peak', 'ratio', 'top')}
            for axis in axes} for name, axes in names.items()}

    part = {key: arrays[key] for key in ('src', 'local', 'keys', 'kept_at')}
    part['specs'] = [(name, tuple(shape), np.dtype(dtype))
                     for name, shape, dtype in meta['specs']]
    part['rows'] = {name: {field: arrays['r/%s/%s' % (name, field)]
                           for field in fields}
                    for name, fields in meta['rows'].items()}
    part['bands'] = bands('b', meta['bands'])
    part['values'] = {name: arrays.get('v/%s' % name) for name in meta['rows']}
    layers = meta['layers']
    if layers is not None:
        layers = dict(layers, frame_min=arrays['l/frame_min'],
                      frame_max=arrays['l/frame_max'],
                      bands=bands('lb', layers['bands'])
                      if layers['bands'] is not None else None)
    part['layers'] = layers
    part['kept'] = {}
    for index, name in enumerate(meta['kept']):
        file = os.path.join(folder, 'kept_%d.npy' % index)
        part['kept'][name] = (np.load(file, mmap_mode='r')
                              if os.path.exists(file) else None)
    return part


class Pool:
    """Everything the analysis reads about a set of frames: per-frame
    statistics of every input (and, with a graph, every internal tensor), the
    sources, the k-means candidates and ``frames(indices, name)`` to read a
    frame back."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def schemes(self, scheme):
        return _schemes(self.names, self.state, scheme)

    def frame_arrays(self, keep):
        return {name: np.asarray(self.frames(keep, name))
                for name, _, _ in self.specs}


def _tensor_stats(shape, rows, sample, total, concentration):
    fmin, fmax = rows['fmin'], rows['fmax']
    n = fmin.size
    gmin, gmax = float(fmin.min()), float(fmax.max())
    span = gmax - gmin
    loo_span = _loo_extreme(fmax, True) - _loo_extreme(fmin, False)
    shrink = (span - loo_span) / span if span > 0 else np.zeros(n)
    s64 = sample.astype(np.float64)
    p001, p999 = (float(v) for v in np.percentile(s64, [0.1, 99.9]))
    abs_p999 = float(np.percentile(np.abs(s64), 99.9))
    mean, std = float(s64.mean()), float(s64.std())
    kurt = (float(np.mean((s64 - mean) ** 4) / (std ** 4 + EPS) - 3.0)
            if std > 0 else 0.0)
    absmax = max(abs(gmin), abs(gmax))
    return {
        'shape': [n] + list(shape),
        'global': {
            'min': gmin, 'max': gmax, 'absmax': absmax,
            'mean': mean, 'std': std, 'excess_kurtosis': kurt,
            'p001': p001, 'p999': p999, 'abs_p999': abs_p999,
            'tail_ratio': absmax / (abs_p999 + EPS),
            # Share of the range the central 99.8 % of the values occupies.
            'core_frac': (p999 - p001) / span if span > 0 else 1.0,
            'zero_frac': float(np.mean(sample == 0)),
            'sampled': bool(sample.size < total),
        },
        'concentration': concentration,
        'frame_min': fmin, 'frame_max': fmax, 'frame_shrink': shrink,
        '_sample': sample,
    }


def _value_sample(frames, name, sample, size, keys):
    """Up to SAMPLE_CAP values of ``name`` drawn from each of the frames
    ``sample`` (those of smallest key), behind the SQNR estimates of a tensor
    whose values do not all fit."""
    per = max(1, SAMPLE_CAP // sample.size)
    out = []
    for a in range(0, sample.size, BLOCK):
        rows = sample[a:a + BLOCK]
        block = np.asarray(frames(rows, name)).reshape(len(rows), -1)
        if per >= size:
            out.append(block.reshape(-1))
            continue
        for value, frame in zip(block, rows):
            pick = np.random.default_rng(int(keys[frame])).integers(
                0, size, per)
            out.append(value[pick])
    return np.concatenate(out)


def assemble(parts, report=None, path='', fmt='npz', graph=None,
             source_names=None, source_note='', state=None, frames=None,
             reserve=SAMPLE_FRAMES):
    """The ``Pool`` of the frames ``parts`` (``Scan.part``) scanned, in
    (source, frame) order. ``report`` gives the inputs left unquantised and
    the model family; ``graph`` names the graph replayed. ``frames(indices,
    name)`` reads frames back by their index in that order; without it they
    come from the parts' kept frames. ``source_names`` names each source
    index (sources unknown without it); ``state`` names the state inputs
    (default: the inputs whose first frame is all zero)."""
    report = report or {}
    specs = parts[0]['specs']
    names, skipped = _quantisable(specs,
                                  report.get('state_precision_policy') or {})
    src = np.concatenate([p['src'] for p in parts])
    local = np.concatenate([p['local'] for p in parts])
    order = np.lexsort((local, src))
    n = order.size
    in_order = bool(np.all(order[1:] > order[:-1]))

    def merged(arrays):
        """``arrays`` (one per part) as one array in (source, frame) order;
        a single scan's is used as it is."""
        if len(arrays) == 1 and in_order:
            return arrays[0]
        joined = np.concatenate(arrays)
        return joined if in_order else joined[order]

    if not in_order:
        src, local = src[order], local[order]
    keys = merged([p['keys'] for p in parts])

    if frames is None:
        kept = [p['kept'] for p in parts]
        where = {}
        for k, p in enumerate(parts):
            for row, place in enumerate(p['kept_at'].tolist()):
                where[tuple(place)] = (k, row)

        def frames(indices, name):
            places = [where[(int(src[i]), int(local[i]))]
                      for i in np.atleast_1d(indices)]
            return np.stack([kept[k][name][row] for k, row in places])

    rows = {name: {field: merged([p['rows'][name][field] for p in parts])
                   for field in _FRAME_FIELDS} for name in names}
    ranked = _distinct(np.argsort(keys, kind='stable'), keys)
    sample = np.sort(ranked[:SAMPLE_FRAMES])
    fill = np.arange(n) if n <= reserve else np.sort(ranked[:reserve])
    shapes = {name: shape for name, shape, _ in specs}
    stats = {}
    for name in names:
        size = int(np.prod(shapes[name]))
        conc = (_merged_bands([p['bands'][name] for p in parts])
                if parts[0]['bands'] else {})
        every = [p['values'][name] for p in parts]
        if n * size <= SAMPLE_CAP and all(v is not None for v in every):
            values = merged(every).reshape(-1)
        else:
            values = _value_sample(frames, name, sample, size, keys)
        stats[name] = _tensor_stats(shapes[name], rows[name], values,
                                    n * size, conc)
    layers = None
    if parts[0]['layers'] is not None:
        first = parts[0]['layers']
        if any(p['layers']['names'] != first['names'] for p in parts):
            raise ValueError('the scans followed different tensors')
        layers = {key: first[key] for key in _PART_SCALARS}
        if graph:
            layers['graph'] = graph
        layers['checkpoint_confirmed'] = all(
            p['layers']['checkpoint_confirmed'] for p in parts)
        for key in ('frame_min', 'frame_max'):
            layers[key] = merged([p['layers'][key] for p in parts])
        layers['bands'] = ({name: _merged_bands(
            [p['layers']['bands'][name] for p in parts])
            for name in first['names']} if first['bands'] is not None else None)
    holders = []
    for name in names:
        holders += [int(stats[name]['frame_min'].argmin()),
                    int(stats[name]['frame_max'].argmax())]
    if layers is not None:
        holders += list(layers['frame_min'].argmin(axis=0))
        holders += list(layers['frame_max'].argmax(axis=0))
    candidates = np.union1d(np.asarray(holders, dtype=np.int64), fill)
    # The k-means features, of the candidates only (rows of ``candidates``).
    features = np.stack([rows[name][field][candidates] for name in names
                         for field in ('mean', 'std', 'peak')], axis=1)
    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    known = source_names is not None
    sources = {
        'reliable': known,
        'starts': [int(i) for i in starts],
        'files': ([source_names[int(s)] for s in src[starts]]
                  if known else None),
        'note': source_note,
    }
    if state is None:
        state = [name for name in names if _starts_at_zero(
            rows[name]['fmin'][0], rows[name]['fmax'][0])]
    return Pool(path=path, fmt=fmt, report=report, specs=specs, names=names,
                skipped=skipped, state=[name for name in names
                                        if name in state],
                contiguous=_contiguous(report), frames=frames, count=n,
                stats=stats, features=features, layers=layers,
                sources=sources, candidates=candidates)


def scan_artifact(art, graph=None, bands=True, threads=1, progress=None,
                  reserve=SAMPLE_FRAMES, seed=0):
    """The ``Pool`` of an artifact's frames, replayed through ``graph`` (a
    path) when given."""
    starts, reliable, note = detect_segments(art)
    src = np.zeros(art.frames, dtype=np.int64)
    if reliable:
        for start in starts[1:]:
            src[start:] += 1
    local = np.arange(art.frames) - np.asarray(starts)[src]
    scan = Scan(art.specs, art.report.get('state_precision_policy'),
                art.report, graph, bands, seed=seed, threads=threads,
                progress=progress)
    for a in range(0, art.frames, BLOCK):
        z = min(a + BLOCK, art.frames)
        scan.process({name: value[a:z] for name, value in art.arrays.items()},
                     src[a:z], local[a:z])
    arrays = art.arrays
    return assemble(
        [scan.part()], art.report, art.path, art.fmt, graph=graph,
        source_names=art.report.get('source_files') if reliable else None,
        source_note=note, state=art.state,
        frames=lambda indices, name: arrays[name][indices], reserve=reserve)


# --------------------------------------------------------------------------
# Min/max grids
# --------------------------------------------------------------------------

SCHEMES = ('asym', 'sym', 'pow2')


def pow2_bound(top):
    """2^ceil(log2(top)) for top > 0, else 0 (an all-zero tensor)."""
    top = np.asarray(top, dtype=np.float64)
    safe = np.where(top > 0, top, 1.0)
    bound = np.where(top > 0, np.exp2(np.ceil(np.log2(safe))), 0.0)
    return bound if bound.ndim else float(bound)


def grid_range(lo, hi, scheme):
    """What a min/max grid covers.

    Asymmetric: [lo, hi] widened to include 0, as min/max observers do so that
    0 is exact.  Symmetric: +-max(|lo|, |hi|).  Power of two: the symmetric
    bound rounded up to a power of two (a fixed-point scale).
    """
    if scheme == 'asym':
        return np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    top = np.maximum(np.abs(lo), np.abs(hi))
    if scheme == 'pow2':
        top = pow2_bound(top)
    elif scheme != 'sym':
        raise ValueError('unknown scheme %r' % scheme)
    return -top, top


def quantize(values, lo, hi, bits, scheme):
    """Quantise-dequantise ``values`` on the min/max grid over [lo, hi].

    Asymmetric: 2^bits levels, integer zero point (0 is exact).  Symmetric:
    levels -(2^(bits-1)-1) .. 2^(bits-1)-1, zero point 0.  Power of two:
    levels -2^(bits-1) .. 2^(bits-1)-1 with scale hi / 2^(bits-1), hi being a
    power of two.  [lo, hi] comes from grid_range, so it holds 0.
    """
    if scheme in ('sym', 'pow2'):
        top = 2 ** (bits - 1) - 1
        low = top if scheme == 'sym' else top + 1
        scale = hi / low
        if scale <= 0:
            return np.zeros_like(values)
        return np.clip(np.rint(values / scale), -low, top) * scale
    if scheme != 'asym':
        raise ValueError('unknown scheme %r' % scheme)
    top = 2 ** bits - 1
    scale = (hi - lo) / top
    if scale <= 0:
        return np.full_like(values, lo)
    zero = np.clip(np.rint(-lo / scale), 0, top)
    return (np.clip(np.rint(values / scale) + zero, 0, top) - zero) * scale


class GridError:
    """SQNR of a min/max grid, measured by quantising a value sample."""

    def __init__(self, values, scheme='asym'):
        self.scheme = scheme
        self.v = np.asarray(values, dtype=np.float64).ravel()
        self.sq = self.v * self.v
        self.signal = float(self.sq.sum())
        # Tables and random draws often share a range.
        self._memo = {}

    def _one(self, lo, hi, bits):
        key = (lo, hi, bits)
        if key not in self._memo:
            err = (quantize(self.v, lo, hi, bits, self.scheme) - self.v) ** 2
            inside = (self.v >= lo) & (self.v <= hi)
            self._memo[key] = (
                _db(self.signal, float(err.sum())),
                _db(float(self.sq[inside].sum()), float(err[inside].sum())))
        return self._memo[key]

    def evaluate(self, lo, hi, bits):
        """SQNR (dB) of the ``bits``-wide grid over [lo, hi] (from grid_range).

        ``sqnr_all`` counts clipped values at their full error;
        ``sqnr_typical`` looks only at the values inside the range.  ``lo`` and
        ``hi`` may be arrays of ranges.
        """
        if np.ndim(lo) == 0:
            full, typical = self._one(float(lo), float(hi), bits)
            return {'sqnr_all': full, 'sqnr_typical': typical}
        pairs = [self._one(float(a), float(b), bits) for a, b in zip(lo, hi)]
        return {'sqnr_all': [p[0] for p in pairs],
                'sqnr_typical': [p[1] for p in pairs]}


def _db(signal, error):
    return float(10 * np.log10((signal + EPS) / (error + EPS)))


# --------------------------------------------------------------------------
# Selection: pick the patterns
# --------------------------------------------------------------------------

def frame_score(stats):
    """How extreme each frame is: the worst ratio, over tensors, of the
    frame's abs-max to the median frame abs-max, so tensors of different
    scale compare.
    """
    score = None
    for st in stats.values():
        peak = np.maximum(np.abs(st['frame_min']), np.abs(st['frame_max']))
        ratio = peak / (np.median(peak) + EPS)
        score = ratio if score is None else np.maximum(score, ratio)
    return score


def _kmeans_pick(features, k, rng, iters=12):
    """Indices of k rows that stand for the rest (nearest to k-means centres)."""
    n = features.shape[0]
    if k >= n:
        return np.arange(n)
    z = (features - features.mean(0)) / (features.std(0) + EPS)
    zz = (z * z).sum(1)

    def nearest(centres):
        cc = (centres * centres).sum(1)
        arg = np.empty(n, dtype=np.int64)
        for a in range(0, n, 4096):
            d = zz[a:a + 4096, None] - 2 * z[a:a + 4096] @ centres.T + cc
            arg[a:a + 4096] = d.argmin(1)
        return arg

    # Seeding by uniform sampling keeps the picks proportional to density, so a
    # large regime is thinned rather than kept whole (k-means++ would favour
    # far-away points).
    centres = z[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        arg = nearest(centres)
        counts = np.bincount(arg, minlength=k)
        used = counts > 0
        for col in range(z.shape[1]):
            sums = np.bincount(arg, weights=z[:, col], minlength=k)
            centres[used, col] = sums[used] / counts[used]
    chosen = []
    taken = np.zeros(n, dtype=bool)
    for c in range(k):
        d = zz - 2 * z @ centres[c] + centres[c] @ centres[c]
        d[taken] = np.inf
        index = int(d.argmin())
        taken[index] = True
        chosen.append(index)
    return np.array(chosen)


def _rows(matrix, frames):
    """``matrix[frames]``, without the copy when ``frames`` is every row."""
    return matrix if len(frames) == matrix.shape[0] else matrix[frames]


def _share(part, span):
    """``part / span`` per tensor; 1 for a tensor whose span is 0."""
    return np.where(span > 0, part / np.where(span > 0, span, 1.0), 1.0)


def layer_holders(layers, pool, have):
    """Frames of ``pool``, beyond ``have``, that hold the internal extremes.

    Many tensors reach an end on most frames (a rectifier's 0, a saturated
    gate), so one frame per end would waste the pattern count.  The frames are
    chosen greedily instead: the frame reaching the most ends no chosen frame
    reaches yet, until every tensor's minimum and maximum over ``pool`` is
    held.  Returned most important first: by the largest share of a tensor's
    range that would go if that frame alone were left out.  ``pool`` is sorted
    and holds at least two frames.
    """
    fmin = _rows(layers['frame_min'], pool)
    fmax = _rows(layers['frame_max'], pool)
    low, high = fmin.min(axis=0), fmax.max(axis=0)
    reach = np.concatenate([fmin == low, fmax == high], axis=1)
    done = reach[np.isin(pool, have)].any(axis=0)
    gain = reach[:, ~done].sum(axis=1)
    rows = []
    while not done.all():
        row = int(gain.argmax())
        rows.append(row)
        new = reach[row] & ~done
        gain -= reach[:, new].sum(axis=1)
        done |= new
    rows = np.array(rows, dtype=np.int64)
    low2 = np.partition(fmin, 1, axis=0)[1]
    high2 = np.partition(fmax, -2, axis=0)[-2]
    left = (np.where(fmax[rows] == high, high2, high).astype(np.float64)
            - np.where(fmin[rows] == low, low2, low))
    loss = 1.0 - _share(left, (high - low).astype(np.float64))
    return pool[rows[np.argsort(-loss.max(axis=1), kind='stable')]]


def select_patterns(pool, count=DEFAULT_COUNT, seed=0, graph=True):
    """Choose which frames to calibrate with.

    ``count`` is a cap: a pool of at most ``count`` frames is kept whole.
    Otherwise the frames that hold each input's minimum and maximum are kept,
    then (with the graph replayed and ``graph``) those holding the minimum
    and maximum of every tensor inside the graph (``layer_holders``, most
    important first, so a short count cuts them first), and the rest are
    k-means picks among the candidates (``Pool.candidates``: those frames and
    the frames of smallest key) so the survivors still cover the
    distribution.  ``must`` counts the range-defining frames found,
    ``must_layers`` those found for the internal tensors only and
    ``kept_layers`` how many of the latter are kept.
    """
    if count < 1:
        raise ValueError('--count must be at least 1')
    rng = np.random.default_rng(seed)
    n = pool.count
    notes = []
    must = np.array([], dtype=np.int64)
    must_layers = kept_layers = 0
    layers = pool.layers if graph else None
    if count >= n:
        if pool.candidates.size < n:
            raise RuntimeError('only %d of %d frames were kept for a '
                               'selection of %d' % (pool.candidates.size, n,
                                                    count))
        keep = np.arange(n)
        if count > n:
            notes.append('only %d frames, fewer than --count %d; all are kept'
                         % (n, count))
    else:
        holders = set()
        for st in pool.stats.values():
            holders.add(int(np.argmin(st['frame_min'])))
            holders.add(int(np.argmax(st['frame_max'])))
        must = np.array(sorted(holders), dtype=np.int64)
        if layers is not None:
            inner = layer_holders(layers, pool.candidates, must)
            must_layers = int(inner.size)
            must = np.concatenate([must, inner])
            kept_layers = max(0, min(count, must.size) - (must.size - must_layers))
        if must.size >= count:
            keep = np.sort(must[:count])
            if must.size > count:
                notes.append('%d range-defining frames exceed --count %d; the '
                             'range will be narrower than the pool\'s'
                             % (must.size, count))
        else:
            rest = np.setdiff1d(pool.candidates, must)
            features = pool.features[np.searchsorted(pool.candidates, rest)]
            picked = rest[_kmeans_pick(features, count - must.size, rng)]
            keep = np.sort(np.concatenate([must, picked]))
    return {'keep': keep, 'must': int(must.size), 'notes': notes,
            'must_layers': must_layers, 'kept_layers': kept_layers}


# --------------------------------------------------------------------------
# Scoring a subset
# --------------------------------------------------------------------------

def frames_outside_pct(st, lo, hi, part=slice(None)):
    """Percent of the frames in ``part`` with a value outside [lo, hi]."""
    return 100.0 * float(np.mean((st['frame_min'][part] < lo)
                                 | (st['frame_max'][part] > hi)))


def score_row(name, st, model, lo, hi, bits):
    """One tensor's row for the range [lo, hi]: the percent of frames with a
    value outside it (counted exactly), then the SQNR per width."""
    row = {'tensor': name, 'lo': float(lo), 'hi': float(hi),
           'frames_clip_pct': frames_outside_pct(st, lo, hi)}
    for b in bits:
        m = model.evaluate(lo, hi, b)
        row['sqnr%d_all' % b] = float(m['sqnr_all'])
        row['sqnr%d_typ' % b] = float(m['sqnr_typical'])
    return row


def subset_table(stats, schemes, models, keep, label, bits):
    """Per tensor for ``keep``: grid range, frames clipped, SQNR per width."""
    rows = []
    for name, st in stats.items():
        lo, hi = grid_range(float(st['frame_min'][keep].min()),
                            float(st['frame_max'][keep].max()), schemes[name])
        rows.append(score_row(name, st, models[name], lo, hi, bits))
    return {'label': label, 'frames': int(len(keep)), 'rows': rows}


def random_baseline(stats, schemes, models, n, count, bits, draws=RANDOM_DRAWS,
                    seed=1):
    """Median (and worst) of ``draws`` uniformly random subsets of ``count``."""
    rng = np.random.default_rng(seed)
    tables = [subset_table(stats, schemes, models,
                           rng.choice(n, count, replace=False), '', bits)
              for _ in range(draws)]
    rows = []
    for index, first in enumerate(tables[0]['rows']):
        row = {'tensor': first['tensor'], 'worst': {}}
        for key in first:
            if key == 'tensor':
                continue
            values = np.array([t['rows'][index][key] for t in tables])
            row[key] = float(np.median(values))
            if key.startswith('sqnr'):
                row['worst'][key] = float(values.min())
            elif 'clip' in key:
                row['worst'][key] = float(values.max())
        rows.append(row)
    return {'label': 'random %d (median of %d)' % (count, draws),
            'frames': int(count), 'rows': rows}


def print_tables(tables, bits):
    head = '%-30s %-16s %-22s %8s' % ('subset', 'tensor', 'range', 'frames%')
    for b in bits:
        head += ' %9s %9s' % ('typ%d' % b, 'all%d' % b)
    print(head)
    for table in tables:
        for index, row in enumerate(table['rows']):
            label = ('%s (%d)' % (table['label'], table['frames'])
                     if index == 0 else '')
            line = '%-30s %-16s [%9.3g,%9.3g] %8.2f' % (
                label[:30], row['tensor'][:16], row['lo'], row['hi'],
                row['frames_clip_pct'])
            for b in bits:
                line += ' %9.2f %9.2f' % (row['sqnr%d_typ' % b],
                                          row['sqnr%d_all' % b])
            print(line)


# --------------------------------------------------------------------------
# Tensors inside the graph
# --------------------------------------------------------------------------

def overshoot(low, high, klow, khigh):
    """How far ``[low, high]`` reaches past ``[klow, khigh]``, as a share of
    the latter's width (inf past a zero width)."""
    past = np.maximum(np.maximum(high - khigh, klow - low), 0).astype(np.float64)
    width = np.broadcast_to(khigh.astype(np.float64) - klow, past.shape)
    share = np.full_like(past, np.inf)
    np.divide(past, width, out=share, where=width > 0)
    return np.where(past > 0, share, 0.0)


def layer_table(layers, keep, against=None):
    """Each internal tensor's grid (``LAYER_SCHEME``) over ``keep`` against
    every frame of ``against`` (default: ``layers``, the same replay): the
    share of the base grid kept, how far the base values reach past the kept
    grid (``overshoot``) and the percent of base frames holding a value
    outside it."""
    fmin, fmax = layers['frame_min'], layers['frame_max']
    against = layers if against is None else against
    bmin, bmax = against['frame_min'], against['frame_max']
    low, high = bmin.min(axis=0), bmax.max(axis=0)
    blow, bhigh = grid_range(low.astype(np.float64), high.astype(np.float64),
                             LAYER_SCHEME)
    klow, khigh = grid_range(fmin[keep].min(axis=0).astype(np.float64),
                             fmax[keep].max(axis=0).astype(np.float64),
                             LAYER_SCHEME)
    kept = _share(khigh - klow, bhigh - blow)
    over = overshoot(low, high, klow, khigh)
    outside = 100.0 * np.mean((bmin < klow) | (bmax > khigh), axis=0)
    softmax = set(layers.get('softmax') or ())
    rows = [{'tensor': name, 'op': op, 'lo': float(low[i]),
             'hi': float(high[i]), 'keep_lo': float(klow[i]),
             'keep_hi': float(khigh[i]), 'kept': float(kept[i]),
             'overshoot': float(over[i]),
             'frames_outside_pct': float(outside[i]),
             'softmax': name in softmax}
            for i, (name, op) in enumerate(zip(layers['names'],
                                               layers['ops']))]
    narrowed = [row for row in rows if row['frames_outside_pct'] > 0]
    return {'frames': int(len(keep)), 'base_frames': int(bmin.shape[0]),
            'tensors': len(rows), 'narrowed': len(narrowed),
            'worst': (min(narrowed, key=lambda row: row['kept'])
                      if narrowed else None),
            'frames_outside_max': max(
                (row['frames_outside_pct'] for row in rows), default=0.0),
            'rows': rows}


def print_layer_table(table, label, top=10):
    print('%s (%d of %d frames): %d of %d internal tensors narrower' % (
        label, table['frames'], table['base_frames'], table['narrowed'],
        table['tensors']))
    rows = sorted((row for row in table['rows']
                   if row['frames_outside_pct'] > 0),
                  key=lambda row: row['kept'])
    for row in rows[:top]:
        print('  %-40s %-10s [%9.3g,%9.3g] -> [%9.3g,%9.3g] keeps %5.1f%%, '
              '%.2f%% of frames outside' % (
                  row['tensor'][:40], row['op'][:10], row['lo'], row['hi'],
                  row['keep_lo'], row['keep_hi'], 100 * row['kept'],
                  row['frames_outside_pct']))
    if len(rows) > top:
        print('  ... %d more' % (len(rows) - top))


def layer_comparison(pool, sel, count, seed=0):
    """``sel`` (chosen with the graph) and the input-only choice, each
    against every frame."""
    plain = select_patterns(pool, count, seed, graph=False)
    layers = pool.layers
    return {'graph': layers['graph'],
            'checkpoint_confirmed': layers['checkpoint_confirmed'],
            'replayed': pool.count, 'must_layers': sel['must_layers'],
            'kept_layers': sel['kept_layers'],
            'selected': layer_table(layers, sel['keep']),
            'input_only': layer_table(layers, plain['keep']),
            'bands': layers.get('bands')}


def print_layer_comparison(inner):
    print_layer_table(inner['input_only'], 'selected from the inputs only')
    print_layer_table(inner['selected'], 'selected with the graph')


# --------------------------------------------------------------------------
# Sources: per-file contribution, range growth, leaving groups out
# --------------------------------------------------------------------------

def _source_bounds(pool):
    """Source starts and the end, when two or more sources are known; else
    None."""
    sources = pool.sources
    if not sources['reliable'] or len(sources['starts']) < 2:
        return None
    return np.array(sources['starts'] + [pool.count])


def _source_note(pool):
    sources = pool.sources
    if not sources['reliable']:
        return sources['note'] or 'no source files recorded'
    return 'a single source'


def _step(lo, hi, scheme, bits=STEP_BITS):
    """The quantisation step of the ``bits``-wide grid over [lo, hi]."""
    glo, ghi = grid_range(lo, hi, scheme)
    if scheme == 'asym':
        return (ghi - glo) / (2 ** bits - 1)
    return ghi / (2 ** (bits - 1) - (1 if scheme == 'sym' else 0))


def source_report(pool, keep, score, schemes, seed=0):
    """Per source: frames, kept, extremeness, and what leaving it out buys:
    the input whose grid (``schemes``) narrows most, by how much (the share
    the step shrinks), the ``STEP_BITS`` step before and after, and the
    frames of the source that hold the extremes it removes.

    Only when the sources are known; otherwise the section reports why they
    are unavailable instead of guessing.
    """
    bounds = _source_bounds(pool)
    if bounds is None:
        return {'available': False, 'note': _source_note(pool),
                'single': pool.sources['reliable']}
    starts = list(bounds[:-1])
    files = pool.sources['files']
    kept = (np.bincount(np.searchsorted(bounds, keep, side='right') - 1,
                        minlength=len(starts))
            if keep is not None else None)
    seg_score = np.maximum.reduceat(score, starts)
    shrink = np.zeros(len(starts))
    detail = [None] * len(starts)
    seg_min, seg_max = {}, {}
    for name, st in pool.stats.items():
        lo = np.minimum.reduceat(st['frame_min'], starts)
        hi = np.maximum.reduceat(st['frame_max'], starts)
        seg_min[name], seg_max[name] = lo, hi
        glo, ghi = grid_range(lo.min(), hi.max(), schemes[name])
        if ghi - glo <= 0:
            continue
        loo_lo, loo_hi = _loo_extreme(lo, False), _loo_extreme(hi, True)
        llo, lhi = grid_range(loo_lo, loo_hi, schemes[name])
        part = 1.0 - (lhi - llo) / (ghi - glo)
        for i in np.flatnonzero(part > shrink + 1e-12):
            a, z = bounds[i], bounds[i + 1]
            # The frames holding the ends that move without this source.
            ends = []
            if loo_lo[i] != lo.min():
                ends.append(int(st['frame_min'][a:z].argmin()))
            if loo_hi[i] != hi.max():
                ends.append(int(st['frame_max'][a:z].argmax()))
            shrink[i] = part[i]
            detail[i] = {'tensor': name,
                         'range': [float(lo.min()), float(hi.max())],
                         'range_without': [float(loo_lo[i]), float(loo_hi[i])],
                         'step': float(_step(lo.min(), hi.max(), schemes[name])),
                         'step_without': float(_step(loo_lo[i], loo_hi[i],
                                                     schemes[name])),
                         'frames': ends}
    rows = [{'source': files[i], 'start': int(starts[i]),
             'frames': int(bounds[i + 1] - bounds[i]),
             'kept': int(kept[i]) if kept is not None else None,
             'extremeness': float(seg_score[i]),
             'loso_shrink': float(shrink[i]),
             'loso_tensor': detail[i]['tensor'] if detail[i] else '',
             'loso': detail[i]}
            for i in range(len(starts))]

    # Range growth: sources added in random orders, the least-converged
    # tensor's share of its final span after each addition.
    rng = np.random.default_rng(seed)
    count = len(starts)
    curves = np.ones((GROWTH_ORDERS, count))
    for p in range(GROWTH_ORDERS):
        perm = rng.permutation(count)
        share = np.ones(count)
        for name in pool.stats:
            lo = np.minimum.accumulate(seg_min[name][perm])
            hi = np.maximum.accumulate(seg_max[name][perm])
            span = hi[-1] - lo[-1]
            if span > 0:
                share = np.minimum(share, (hi - lo) / span)
        curves[p] = share
    return {'available': True, 'note': pool.sources['note'], 'rows': rows,
            'growth': {'median': np.median(curves, 0),
                       'p10': np.percentile(curves, 10, axis=0)}}


def _grids(lo, hi, schemes):
    """``grid_range`` per column of ``lo``/``hi`` under ``schemes`` (one per
    column)."""
    out_lo, out_hi = np.empty(lo.shape), np.empty(hi.shape)
    for scheme in set(schemes):
        cols = np.array([s == scheme for s in schemes])
        out_lo[cols], out_hi[cols] = grid_range(lo[cols], hi[cols], scheme)
    return out_lo, out_hi


def source_folds(pool, schemes, folds=FOLDS, seed=0):
    """Each group of sources in turn left out: how far the values of the
    group go past the grid the other groups set (``schemes`` for the inputs,
    ``LAYER_SCHEME`` inside the graph), per tensor, the source that goes
    furthest, and the largest percent of a group's frames past it. The
    per-frame tables are read a source at a time, never copied whole."""
    bounds = _source_bounds(pool)
    if bounds is None:
        return {'available': False}
    count = bounds.size - 1
    k = min(folds, count)
    group = np.empty(count, dtype=np.int64)
    group[np.random.default_rng(seed).permutation(count)] = np.arange(count) % k
    files = pool.sources['files']
    names = list(pool.stats)
    ops = ['input'] * len(names)
    grids = [schemes[name] for name in names]
    tables = [(np.stack([pool.stats[name]['frame_min'] for name in names],
                        axis=1),
               np.stack([pool.stats[name]['frame_max'] for name in names],
                        axis=1))]
    layers = pool.layers
    if layers is not None:
        names += layers['names']
        ops += layers['ops']
        grids += [LAYER_SCHEME] * len(layers['names'])
        tables.append((layers['frame_min'], layers['frame_max']))
    softmax = set(layers.get('softmax') or ()) if layers else set()
    starts = bounds[:-1]
    # Per source and tensor, the extremes: [sources, tensors].
    src_lo = np.hstack([np.minimum.reduceat(low, starts, axis=0)
                        for low, _ in tables]).astype(np.float64)
    src_hi = np.hstack([np.maximum.reduceat(high, starts, axis=0)
                        for _, high in tables]).astype(np.float64)
    width = len(names)
    worst = np.zeros(width)
    worst_src = np.full(width, -1)
    past_pct = np.zeros(width)
    for g in range(k):
        mine = group == g
        olo, ohi = _grids(src_lo[~mine].min(axis=0), src_hi[~mine].max(axis=0),
                          grids)
        over = overshoot(src_lo[mine], src_hi[mine], olo, ohi)
        best = over.argmax(axis=0)
        value = over[best, np.arange(width)]
        up = value > worst
        worst[up] = value[up]
        worst_src[up] = np.flatnonzero(mine)[best[up]]
        past = np.zeros(width, dtype=np.int64)
        frames = 0
        for source in np.flatnonzero(mine):
            a, z = bounds[source], bounds[source + 1]
            column = 0
            for low, high in tables:
                cols = slice(column, column + low.shape[1])
                past[cols] += np.count_nonzero(
                    (low[a:z] < olo[cols]) | (high[a:z] > ohi[cols]), axis=0)
                column += low.shape[1]
            frames += z - a
        past_pct = np.maximum(past_pct, 100.0 * past / frames)
    rows = [{'tensor': name, 'op': op, 'overshoot': float(worst[i]),
             'frames_past_pct': float(past_pct[i]),
             'source': files[worst_src[i]] if worst_src[i] >= 0 else None,
             'softmax': name in softmax}
            for i, (name, op) in enumerate(zip(names, ops))]
    rows.sort(key=lambda row: -row['overshoot'])
    return {'available': True, 'groups': k, 'sources': count,
            'outside': sum(row['overshoot'] > 0 for row in rows),
            'tensors': len(rows), 'rows': rows}


# --------------------------------------------------------------------------
# Conclusions
# --------------------------------------------------------------------------

def _signed(value):
    """'+1.2' / '-0.4' / '+0.0' (never '-0.0')."""
    return '%+.1f' % (round(value, 1) + 0.0)


SCHEME_NAMES = {'asym': '非對稱', 'sym': '對稱', 'pow2': ' 2 的冪次'}


def _pow2_lines(stats, schemes, bits, n):
    """Each power-of-two tensor's bound, and what moving it down costs."""
    out, parts = [], []
    edge = 2.0 ** -(min(bits) - 1)
    for name, st in stats.items():
        if schemes[name] != 'pow2':
            continue
        peak = np.maximum(np.abs(st['frame_min']), np.abs(st['frame_max']))
        top = float(peak.max())
        bound = pow2_bound(top)
        if bound <= 0:
            parts.append('%s 全為 0' % name)
            continue
        over = int(np.count_nonzero(peak > bound / 2))
        parts.append('%s ±%g（降到 ±%g 要丟 %d 個 frame，%.2f%%）' % (
            name, bound, bound / 2, over, 100.0 * over / n))
        if top > bound * (1 - edge):
            out.append(('warn', '%s 的絕對值最大 %.6g，離 ±%g 不到 %d-bit 的一個 step：'
                        '若 converter 是把 scale（而不是範圍）取到 2 的冪次，上限會變成 '
                        '±%g，差 6 dB。請用 converter 實際輸出的 scale 確認。' % (
                            name, top, bound, min(bits), 2 * bound)))
    if parts:
        out.insert(0, ('info', '2 的冪次 grid 的上限（整池）：' + '；'.join(parts)
                       + '。range 要跨過下一個 2 的冪次，step 才會變細；沒跨過時，'
                       '丟極值或隨機挑選得到的輸入 grid 和保留極值完全相同。'))
    return out


def _narrowed_text(table):
    """How far ``table`` (``layer_table``) falls short of its base frames."""
    if not table['narrowed']:
        return '%d 個內部 tensor 的 range 都和基準相同' % table['tensors']
    worst = table['worst']
    return ('%d 個內部 tensor 中有 %d 個的 range 變窄；最窄的是 %s（%s），只剩 '
            '%.1f%%；有值超出的 frame 最多佔 %.2f%%' % (
                table['tensors'], table['narrowed'], worst['tensor'],
                worst['op'], 100 * worst['kept'], table['frames_outside_max']))


def _level(table):
    return 'warn' if table['narrowed'] else 'info'


def _levels(ratio, bits):
    """Steps a symmetric ``bits`` grid leaves the median index at its peak
    when one index is ``ratio`` times larger and sets the range."""
    steps = (2 ** (bits - 1) - 1) / ratio
    return '不到 1' if steps < 1 else '%d' % steps


def band_kind(c):
    """'fixed' (one index dominates a typical frame, and it is the same one
    on most frames), 'rotating' (a few indices take turns), 'loud' (only a
    few frames have one index far above the rest) or None."""
    if not c or c['ratio'] <= FIXED_BAND_RATIO:
        return None
    if c['typical'] <= FIXED_BAND_RATIO:
        return 'loud'
    return 'fixed' if c['frame_share'] >= FIXED_BAND_SHARE else 'rotating'


def _band_lines(bands, bits, ops=None, top=None):
    """Conclusions on one index dominating a tensor. ``bands`` maps each
    tensor to its ``_concentration`` per axis; ``ops`` names the operator of
    a tensor inside the graph; ``top`` caps the fixed ones written one by
    one. A peak only a few frames reach is told apart from one that is there
    on a typical frame (only the latter is a feature scale), and one index
    that is there on most frames from a few that take turns."""
    fixed, rotating, loud = [], [], []
    for name, conc in bands.items():
        for axis, label in (('channel', 'channel'), ('last', '最後一軸的位置')):
            c = conc.get(axis)
            kind = band_kind(c)
            if kind:
                {'fixed': fixed, 'rotating': rotating, 'loud': loud}[kind].append(
                    (name if not ops else '%s（%s）' % (name, ops[name]),
                     label, c))
    fixed.sort(key=lambda item: -item[2]['ratio'])
    rotating.sort(key=lambda item: -item[2]['ratio'])
    shown = fixed if top is None else fixed[:top]
    width = max(bits)
    out = [('warn', '%s 的極端值固定落在第 %d 個%s（從 0 數）：一般 frame 裡最大的是'
            '中位數的 %.1f 倍，%.0f%% 的 frame 最大值落在這裡；全部 frame 合起來，'
            '最大的峰值是中位數的 %.1f 倍。整個 tensor 共用一個 scale，%d-bit 時中位數'
            '的那一個在峰值只剩約 %s 階。這是特徵縮放的問題，刪 frame 解決不了。' % (
                name, c['frame_top'], label, c['typical'],
                100 * c['frame_share'], c['ratio'], width,
                _levels(c['ratio'], width)))
           for name, label, c in shown]
    rest = fixed[len(shown):]
    if rest:
        out.append(('warn', '另有 %d 處也固定落在同一個 channel 或位置：%s。' % (
            len(rest), '、'.join('%s 的第 %d 個%s（%.1f 倍）' % (
                name, c['frame_top'], label, c['ratio'])
                for name, label, c in rest))))
    for name, label, c in (rotating if top is None else rotating[:top]):
        out.append(('warn', '%s 的極端值集中在少數幾個%s，輪流出現（從 0 數，括號內是'
                    '該位置在各 frame 是最大值的比例）：%s。一般 frame 裡最大的是中位數的 '
                    '%.1f 倍；全部 frame 合起來，最大的峰值是中位數的 %.1f 倍。整個 tensor '
                    '共用一個 scale，%d-bit 時中位數的那一個在峰值只剩約 %s 階。這也是特徵'
                    '縮放的問題，刪 frame 解決不了。' % (
                        name, label, '、'.join('第 %d 個（%.0f%%）' % (i, 100 * share)
                                              for i, share in c['top_shares']),
                        c['typical'], c['ratio'], width,
                        _levels(c['ratio'], width))))
    if loud:
        out.append(('info', '%s 的峰值集中在單一 channel 或位置（超過中位數 %g 倍），'
                    '但一般 frame 裡沒有：是少數 frame 撐出來的，不是固定的特徵縮放。'
                    % ('、'.join(sorted({name for name, _, _ in loud})),
                       FIXED_BAND_RATIO)))
    return out


def _layer_lines(inner, must, bits):
    """Conclusions about the tensors inside the graph."""
    selected, plain = inner['selected'], inner['input_only']
    frames = selected['frames']
    out = [('info', '已用 %s 把 %d 個 frame 逐一重跑（FP32），記錄 graph 內部 %d 個 '
            'tensor 每個 frame 的最小值和最大值。%s' % (
                os.path.basename(inner['graph']), inner['replayed'],
                selected['tensors'],
                '' if inner['checkpoint_confirmed'] else
                'graph 和校正檔沒有都記錄 checkpoint，無法確認兩者出自同一個 checkpoint。'))]
    if inner.get('bands'):
        out += _band_lines(inner['bands'], bits,
                           {row['tensor']: row['op'] for row in selected['rows']},
                           LAYER_BAND_LINES)
    text = '挑出的 %d 個 frame：%s。' % (frames, _narrowed_text(selected))
    if must:
        held = min(must, frames)
        text += ('保住輸入和內部的極值共用了 %d 個 frame（為內部 tensor 多留的有 %d 個），'
                 '佔 %.0f%%；其餘才是依分布抽樣。佔比高時，這份資料會偏向極端值：'
                 '純 min/max 不受影響，但 converter 依直方圖的校正模式（只作用在 8-bit）'
                 '會受影響。' % (held, inner['kept_layers'],
                              100.0 * held / frames))
    out.append((_level(selected), text))
    out.append((_level(plain), '對照：只看輸入來挑（--inputs-only）的 %d 個 frame：%s。'
                % (plain['frames'], _narrowed_text(plain))))
    out.append(('info', '內部的 range 是 FP32 graph 每個 ONNX tensor 的最小值和最大值。'
                'converter 之後還會調整（合併 op、查表類函數夾範圍、多個 tensor 共用 scale），'
                '遞迴 op 內部的 gate 也看不到，所以這裡只保證「挑選後和整池看到的極值相同」，'
                '不代表量化後的精度。'))
    return out


def _width(share):
    return ('%.1f%% 的寬度' % (100 * share) if np.isfinite(share)
            else '寬度為 0 的 range')


def _fold_lines(folds):
    """Conclusions on leaving each group of sources out."""
    if not folds.get('available'):
        return []
    head = ('把 %d 個來源分成 %d 組，輪流拿掉一組，看那一組的值有沒有超出其他組'
            '定出的 grid：' % (folds['sources'], folds['groups']))
    outside = [row for row in folds['rows'] if row['overshoot'] > 0]
    if not outside:
        return [('info', head + '%d 個 tensor 都沒有超出。' % folds['tensors'])]
    worst = outside[0]
    others = '、'.join('%s %s' % (row['tensor'], _width(row['overshoot']))
                      for row in outside[1:FOLD_LINES])
    out = [('warn', head + '%d 個 tensor 中有 %d 個超出；最多的是 %s（%s）超出 %s，'
            '最遠的來源是「%s」，超出的那一組最多有 %.2f%% 的 frame 超出。數字越大，'
            '代表再多錄一些不同的資料，range 可能還會擴大；這個數字沒有驗證過的門檻。%s' % (
                folds['tensors'], len(outside), worst['tensor'],
                '輸入' if worst['op'] == 'input' else worst['op'],
                _width(worst['overshoot']), worst['source'],
                worst['frames_past_pct'],
                '其餘超出最多的：%s。' % others if others else ''))]
    soft = [row['tensor'] for row in outside if row['softmax']]
    if soft:
        out.append(('warn', '其中 %s 在 softmax 之前、中間沒有經過 activation：這類值'
                    '超出 range 被夾平時，最大的幾個會變成一樣大，softmax 就選不出原本'
                    '最大的那一個。要讓這些 tensor 的 range 蓋住，請補進會撐大它們的錄音'
                    '（上面列出的來源是一個起點）。' % '、'.join(soft[:FOLD_LINES])))
    return out


def conclusions(pool, schemes, recommend, sources, folds, bits, count):
    """What the numbers say, as (level, text) pairs in the report language."""
    out = []
    n = pool.count
    if not pool.report:
        out.append(('warn', '沒有校正報告（manifest.json／.calib.json）：無法檢查'
                    '模型種類（Align-ULCNet 的前端 stamp、是否需要連續串流）、來源檔，'
                    '以及哪些輸入不量化；這些檢查都略過了。'))
    if pool.contiguous:
        out.append(('warn', '%s 需要一段不中斷的串流，不能挑選或刪減 frame。'
                    % pool.report.get('model_family')))
    if sources['available']:
        rows = sources['rows']
        frames = np.array([r['frames'] for r in rows])
        out.append(('info', '%d 個 frame 來自 %d 個來源（每個來源 %d–%d 個 frame，'
                    '最大的一個佔 %.1f%%）。' % (
                        n, len(rows), frames.min(), frames.max(),
                        100.0 * frames.max() / n)))
        if len(rows) <= 3:
            out.append(('warn', '只有 %d 個來源。相鄰 frame 幾乎相同，來源太少時'
                        '涵蓋的情境很有限；建議從更多錄音擷取再挑選。' % len(rows)))
        top = sorted(rows, key=lambda r: -r['loso_shrink'])[:3]
        top = [r for r in top if r['loso_shrink'] > 0]
        if top:
            out.append(('info', 'range 由少數來源撐起：' + '；'.join(
                '拿掉「%s」，%s 的 grid 縮小 %.1f%%（%d-bit step 從 %.3g 變 %.3g；'
                '極值在該來源的第 %s 個 frame，從 0 數）' % (
                    r['source'], r['loso_tensor'], 100 * r['loso_shrink'],
                    STEP_BITS, r['loso']['step'], r['loso']['step_without'],
                    '、'.join(str(f) for f in r['loso']['frames']))
                for r in top) + '。縮小的比例就是 step 變細的比例；min 或 max 變了、'
                '但 grid 沒變的來源不列。值得去聽這些來源，確認是真實會遇到的情境'
                '還是壞資料。'))
        growth = sources['growth']['median']
        at90 = growth[max(0, int(np.ceil(0.9 * len(rows))) - 1)]
        out.append(('info', '依隨機順序加入來源（%d 種順序的中位數），加入最後 10%% '
                    '的來源時，最不穩定的 tensor 的 range 仍擴大了 %.1f%%。數字越大，'
                    '代表再多錄一些資料，可能還會出現新的極值；這個數字沒有驗證過的門檻。'
                    % (GROWTH_ORDERS, 100 * (1 - at90))))
    elif sources.get('single'):
        out.append(('warn', '所有 %d 個 frame 都來自同一個來源。相鄰 frame 幾乎相同，'
                    '涵蓋的情境很有限；建議從多個錄音擷取再挑選。' % n))
    else:
        out.append(('warn', '來源資訊不可用（%s）：無法看每個來源的貢獻，也無法判斷 '
                    'range 是否已經收斂。' % sources['note']))
    out += _fold_lines(folds)
    if recommend and n <= count:
        out.append(('warn', '整份校正資料只有 %d 個 frame，不超過要挑的 %d 個，等於沒有挑選。'
                    '只擷取目標數量只會涵蓋少數錄音；擷取時改用 --select，讓工具掃過'
                    '所有錄音再挑。' % (n, count)))
    out += _band_lines({name: st['concentration']
                        for name, st in pool.stats.items()}, bits)
    if pool.state:
        grid = schemes[pool.state[0]]
        out.append(('info', '%s 被視為 recurrent state，以%s grid 評分：板端 state '
                    'port 的 zero point 必須是 0。'
                    % ('、'.join(pool.state), SCHEME_NAMES[grid])))
    out += _pow2_lines(pool.stats, schemes, bits, n)
    if recommend:
        sel = recommend['selected']
        inner = recommend.get('layers')
        must_inputs = recommend['must'] - (inner['must_layers'] if inner else 0)
        if must_inputs > sel['frames']:
            out.append(('warn', '保住每個輸入 tensor 的最小值和最大值需要 %d 個 frame，'
                        '超過 --count %d：挑出的 frame 的輸入 range 比整池窄。'
                        % (must_inputs, count)))
        elif recommend['must'] > sel['frames']:
            out.append(('warn', '保住輸入和內部每個 tensor 的最小值和最大值需要 %d 個 '
                        'frame，超過 --count %d：輸入的 %d 個都保留，內部的 %d 個依影響'
                        '大小只保留了 %d 個。' % (
                            recommend['must'], count, must_inputs,
                            inner['must_layers'], inner['kept_layers'])))
        elif recommend['must']:
            text = ('挑出的 %d 個 frame 保有整池每個輸入 tensor 的最小值和'
                    '最大值，所以輸入 range 和整池相同。' % sel['frames'])
            if not inner:
                text += ('保住這些極值至少需要 %d 個 frame，但這只對輸入成立；'
                         '這次沒有重跑 graph，看不到 graph 內部各層的 range。'
                         % recommend['must'])
            out.append(('info', text))
        else:
            out.append(('info', '全部 %d 個 frame 都保留，輸入 range 和整池相同。'
                        % sel['frames']))
        if inner:
            out += _layer_lines(inner, recommend['must'], bits)
        rnd = recommend.get('random')
        if rnd:
            parts = []
            for b in bits:
                key = 'sqnr%d_all' % b
                med = min(r[key] - s[key] for r, s in zip(rnd['rows'], sel['rows']))
                worst = min(r['worst'][key] - s[key]
                            for r, s in zip(rnd['rows'], sel['rows']))
                parts.append('%d-bit 中位數 %s dB、最差 %s dB' % (
                    b, _signed(med), _signed(worst)))
            out.append(('info', '對照：隨機挑 %d 個（%d 次）相對於建議選法，含被夾值的 '
                        'SQNR 在最受影響的 tensor 上為 %s。這組對照用整池本身評分：整池的'
                        '極值就在評分資料裡，所以會高估丟掉極值的代價；不偏的數字請用 '
                        'coverage 對另一份沒看過的錄音評估。' % (
                            rnd['frames'], RANDOM_DRAWS, '；'.join(parts))))
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _chart(st):
    """A per-frame peak chart of at most CHART_POINTS bars (each the largest
    peak of its frames) and the frames holding the minimum and maximum."""
    peak = np.maximum(np.abs(st['frame_min']), np.abs(st['frame_max']))
    starts = np.unique(np.linspace(0, peak.size, CHART_POINTS,
                                   endpoint=False).astype(np.int64))
    return {'start': starts, 'peak': np.maximum.reduceat(peak, starts),
            'holders': [int(st['frame_min'].argmin()),
                        int(st['frame_max'].argmax())]}


def analyze(pool, scheme='asym', seed=0, count=DEFAULT_COUNT,
            bits=DEFAULT_BITS):
    schemes = pool.schemes(scheme)
    models = {name: GridError(pool.stats[name]['_sample'], schemes[name])
              for name in pool.names}
    score = frame_score(pool.stats)
    n = pool.count
    recommend = None
    keep = None
    if not pool.contiguous:
        sel = select_patterns(pool, count, seed)
        keep = sel['keep']
        recommend = {
            'keep_frames': keep, 'count': count, 'must': sel['must'],
            'notes': sel['notes'],
            'all': subset_table(pool.stats, schemes, models, np.arange(n),
                                'all frames', bits),
            'selected': subset_table(pool.stats, schemes, models, keep,
                                     'selected', bits),
        }
        if keep.size < n:
            recommend['random'] = random_baseline(pool.stats, schemes, models,
                                                  n, int(keep.size), bits)
        if pool.layers is not None:
            recommend['layers'] = layer_comparison(pool, sel, count, seed)
    sources = source_report(pool, keep, score, schemes, seed)
    folds = source_folds(pool, schemes, seed=seed)
    return {
        'meta': {
            'path': pool.path, 'format': pool.fmt, 'frames': n,
            'model_family': pool.report.get('model_family'),
            'graph': pool.layers['graph'] if pool.layers else None,
            'scheme': scheme, 'schemes': schemes, 'bits': list(bits),
            'contiguous': pool.contiguous, 'skipped': pool.skipped,
            'state': pool.state, 'sample_cap': SAMPLE_CAP,
            'sample_frames': SAMPLE_FRAMES,
            'fixed_band_ratio': FIXED_BAND_RATIO,
            'fixed_band_share': FIXED_BAND_SHARE,
        },
        'conclusions': [{'level': level, 'text': text} for level, text in
                        conclusions(pool, schemes, recommend, sources, folds,
                                    bits, count)],
        'recommend': recommend,
        'sources': sources,
        'folds': folds,
        'tensors': {
            name: {'shape': s['shape'], 'global': s['global'],
                   'concentration': s['concentration'], 'chart': _chart(s)}
            for name, s in pool.stats.items()
        },
    }


def _range_report(arrays, policy):
    out = {}
    for name, value in arrays.items():
        entry = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if name in policy:
            entry['precision'] = policy[name]
        elif np.issubdtype(value.dtype, np.floating):
            low, high = np.percentile(value, [0.1, 99.9])
            entry.update({'min': float(value.min()), 'max': float(value.max()),
                          'p001': float(low), 'p999': float(high)})
        out[name] = entry
    return out


def require_selectable(source):
    """Refuse a selection for a model that needs one contiguous stream
    (``source``: an artifact or a pool)."""
    if source.contiguous:
        raise SystemExit(
            '%s needs one uninterrupted stream; its cumulative state ranges '
            'are meaningless on a selected or pruned set. Regenerate with a '
            'different source instead.' % source.report.get('model_family'))


def write_selection(pool, keep, output, fmt, make_report=None, parent=None):
    """Write the frames ``keep`` of ``pool`` as a calibration artifact and
    return its report.

    The report is ``make_report(arrays)`` (a capture run's own) or the pool's
    with its input ranges redone; either way it records the state inputs, the
    sources that kept a frame and the selection."""
    keep = np.asarray(keep, dtype=np.int64)
    require_selectable(pool)
    arrays = pool.frame_arrays(keep)
    if make_report is None:
        report = dict(pool.report)
        report['inputs'] = _range_report(
            arrays, report.get('state_precision_policy') or {})
    else:
        report = dict(make_report(arrays))
    require_current_frontend(report, pool.path)
    if pool.sources['reliable']:
        bounds = np.array(pool.sources['starts'] + [pool.count])
        segment = np.searchsorted(bounds, keep, side='right') - 1
        files = pool.sources['files']
        alive = sorted(set(int(s) for s in segment))
        gone = [files[i] for i in range(len(files)) if i not in alive]
        if gone:
            print('note: %d source(s) have no frame left and leave '
                  'source_files: %s' % (len(gone), ', '.join(gone)))
        report['source_files'] = [files[i] for i in alive]
    report['frames'] = int(keep.size)
    report['state_inputs'] = list(pool.state)
    report['selection'] = {
        'tool': 'calib_analyze',
        'parent': parent or os.path.basename(pool.path),
        'parent_frames': pool.count,
        'kept_frames': int(keep.size),
        'kept_indices': [int(i) for i in keep],
    }
    for key in ('binary_tensors', 'artifact_format', 'binary_file_pattern',
                'binary_byte_order', 'binary_frame_index_base'):
        report.pop(key, None)
    write_calibration_artifact(output, arrays, report, fmt)
    return report


# --------------------------------------------------------------------------
# coverage: score a selection on a held-out artifact
# --------------------------------------------------------------------------

# Report fields two artifacts of one graph share; ``frames_per_call`` and
# ``variant`` are written only when they differ from 1 / the full model.
_PAIR_KEYS = ('model_family', 'checkpoint_sha256', 'state_layout_version',
              'sample_rate', 'n_fft', 'win_len', 'hop_len',
              'input_feature_frames', 'frames_per_call', 'mask_lookahead',
              'df_lookahead', 'variant', 'max_delay_frames',
              'linear_aec_contract_version', 'aec_behavior_hash')
_PAIR_DEFAULTS = {'frames_per_call': 1, 'variant': 'full'}


def check_pair(cal, held):
    """Refuse a held-out pool of another graph; return warnings."""
    if cal.names != held.names:
        raise ValueError('tensor names differ: %s vs %s' % (cal.names,
                                                            held.names))
    shapes = {name: shape for name, shape, _ in held.specs}
    for name, shape, _ in cal.specs:
        if name in shapes and shape != shapes[name]:
            raise ValueError('%s frame shape differs: %s vs %s' % (
                name, shape, shapes[name]))
    problems, unknown = [], []
    for key in _PAIR_KEYS:
        a = cal.report.get(key, _PAIR_DEFAULTS.get(key))
        b = held.report.get(key, _PAIR_DEFAULTS.get(key))
        if a is None and b is None:
            continue
        if a is None or b is None:
            unknown.append(key)
        elif a != b:
            problems.append('%s %r vs %r' % (key, a, b))
    if problems:
        raise ValueError('the held-out artifact comes from another graph or '
                         'capture setup: ' + '; '.join(problems))
    warnings = []
    if 'checkpoint_sha256' not in cal.report or 'checkpoint_sha256' not in \
            held.report:
        warnings.append('no checkpoint_sha256 on both reports: cannot confirm '
                        'the two come from one checkpoint (state inputs depend '
                        'on the weights)')
    if unknown:
        warnings.append('recorded on one report only, not compared: '
                        + ', '.join(unknown))
    names = {os.path.basename(p) for p in cal.report.get('source_files') or []}
    shared = names & {os.path.basename(p)
                      for p in held.report.get('source_files') or []}
    if shared:
        warnings.append('%d source file name(s) appear in both, so the '
                        'held-out set is not fully unseen (e.g. %s)'
                        % (len(shared), sorted(shared)[0]))
    return warnings


def coverage(cal, held, scheme='asym', bits=DEFAULT_BITS):
    """Ranges from every frame of ``cal`` scored on ``held`` (both pools)."""
    warnings = check_pair(cal, held)
    schemes = cal.schemes(scheme)
    bounds = _source_bounds(held)
    rows = []
    for name in cal.names:
        c = cal.stats[name]
        lo, hi = grid_range(float(c['frame_min'].min()),
                            float(c['frame_max'].max()), schemes[name])
        st = held.stats[name]
        row = score_row(name, st, GridError(st['_sample'], schemes[name]),
                        lo, hi, bits)
        if bounds is not None:
            # (value, source); stays (0, None) when no source has a clipped
            # frame.
            worst = (0.0, None)
            for i, source in enumerate(held.sources['files']):
                frac = frames_outside_pct(st, lo, hi,
                                          slice(bounds[i], bounds[i + 1]))
                if frac > worst[0]:
                    worst = (frac, source)
            row['worst_source'] = {'frames_clip_pct': worst}
        rows.append(row)
    return {'calibration': cal.path, 'held_out': held.path,
            'frames': cal.count, 'held_out_frames': held.count,
            'warnings': warnings,
            'segments': None if bounds is not None else _source_note(held),
            'rows': rows}


def print_coverage(result, bits):
    for warning in result['warnings']:
        print('warning: ' + warning)
    print('ranges from %d frames of %s, scored on %d held-out frames' % (
        result['frames'], result['calibration'], result['held_out_frames']))
    print_tables([{'label': 'held-out', 'frames': result['held_out_frames'],
                   'rows': result['rows']}], bits)
    if result['segments']:
        print('per-source worst case unavailable: ' + result['segments'])
        return
    for row in result['rows']:
        frac, src = row['worst_source']['frames_clip_pct']
        print('%-16s worst source: %s' % (
            row['tensor'], 'frames clipped %.2f%% (%s)' % (frac, src)
            if src else 'no clipped frame in any source'))


def layer_coverage(cal, held, scheme='asym'):
    """Each internal tensor's range over every frame of ``cal`` scored on
    every frame of ``held``, both replayed through one graph: how far past it
    and on what percent of frames, in graph order, and per source of
    ``held`` when its sources are known.  An origin goes past its range on a
    frame where every operand stays inside its own: the overshoot starts there
    instead of coming from upstream."""
    inner, outer = cal.layers, held.layers
    if inner['names'] != outer['names']:
        raise ValueError('the calibration and held-out replays follow '
                         'different tensors')
    names, operands = inner['names'], inner['operands']
    table = layer_table(inner, np.arange(cal.count), against=outer)
    rows = table['rows']
    for row, reads in zip(rows, operands):
        row['operands'] = reads
    klow, khigh = grid_range(inner['frame_min'].min(axis=0).astype(np.float64),
                             inner['frame_max'].max(axis=0).astype(np.float64),
                             LAYER_SCHEME)
    schemes = cal.schemes(scheme)
    # Per held-out frame: is the tensor (or graph input) past its grid?
    past = (outer['frame_min'] < klow) | (outer['frame_max'] > khigh)
    frame_past = {name: past[:, col] for col, name in enumerate(names)}
    for name in {name for reads in operands for name in reads} - set(names):
        if name in held.stats and name in cal.stats:
            c, h = cal.stats[name], held.stats[name]
            lo, hi = grid_range(c['frame_min'].min(), c['frame_max'].max(),
                                schemes[name])
            frame_past[name] = (h['frame_min'] < lo) | (h['frame_max'] > hi)
        else:       # an input left unquantised is never past
            frame_past[name] = np.zeros(held.count, dtype=bool)
    starts_here = np.stack(
        [frame_past[name] & ~np.any([frame_past[read] for read in reads],
                                    axis=0) if reads else frame_past[name]
         for name, reads in zip(names, operands)], axis=1)

    def origins(part):
        return [names[col] for col in
                np.flatnonzero(starts_here[part].any(axis=0))]

    over = np.array([row['overshoot'] for row in rows])
    result = {'graph': inner['graph'],
              'checkpoint_confirmed': (inner['checkpoint_confirmed']
                                       and outer['checkpoint_confirmed']),
              'frames': table['frames'],
              'held_out_frames': table['base_frames'],
              'tensors': table['tensors'], 'outside': int((over > 0).sum()),
              'origins': origins(slice(None)),
              'rows': rows, 'sources': None, 'segments': None}
    bounds = _source_bounds(held)
    if bounds is None:
        result['segments'] = _source_note(held)
        return result
    sources, per_source = [], []
    for i, source in enumerate(held.sources['files']):
        a, z = bounds[i], bounds[i + 1]
        over = overshoot(outer['frame_min'][a:z].min(axis=0),
                         outer['frame_max'][a:z].max(axis=0), klow, khigh)
        worst = int(over.argmax())
        sources.append({'source': source, 'frames': int(z - a),
                        'outside': int((over > 0).sum()),
                        'origins': origins(slice(a, z)),
                        'worst': names[worst] if over[worst] > 0 else None,
                        'overshoot': float(over[worst])})
        per_source.append(over)
    per_source = np.array(per_source)
    for col, row in enumerate(rows):
        if row['overshoot'] > 0:
            row['worst_source'] = sources[int(per_source[:, col].argmax())][
                'source']
    result['sources'] = sources
    return result


def _names(names, top=5):
    if not names:
        return ('none (each one has an operand past its range on the same '
                'frame)')
    more = ' and %d more' % (len(names) - top) if len(names) > top else ''
    return ', '.join(names[:top]) + more


def _past(share):
    return ('%.1f%% of its width' % (100 * share) if np.isfinite(share)
            else 'a zero-width range')


def print_layer_coverage(inner, top=20):
    print('inside the graph: %d calibration frames replayed, scored on %d '
          'held-out frames' % (inner['frames'], inner['held_out_frames']))
    if not inner['outside']:
        print('every one of the %d tensors stays within its calibration range '
              'on the held-out frames: the float graph shows no clipping '
              '(merged operators and the quantised state of the device are '
              'not modelled)' % inner['tensors'])
    else:
        print('%d of %d tensors go past their calibration range on the '
              'held-out frames' % (inner['outside'], inner['tensors']))
        print('origins (*), past on a frame where every operand stays inside: '
              + _names(inner['origins']))
        print('calibration range -> held-out range, furthest past first:')
        by_name = {row['tensor']: row for row in inner['rows']}
        origins = set(inner['origins'])
        rows = sorted((row for row in inner['rows'] if row['overshoot'] > 0),
                      key=lambda row: -row['overshoot'])
        for row in rows[:top]:
            print('%s %-40s %-8s [%9.3g,%9.3g] -> [%9.3g,%9.3g] past by %s, '
                  '%.2f%% of frames%s%s' % (
                      ' *' if row['tensor'] in origins else '  ',
                      row['tensor'][:40], row['op'][:8], row['keep_lo'],
                      row['keep_hi'], row['lo'], row['hi'],
                      _past(row['overshoot']), row['frames_outside_pct'],
                      ', most in %s' % row['worst_source']
                      if 'worst_source' in row else '',
                      ', reaches a Softmax through no activation'
                      if row['softmax'] else ''))
            if row['op'] in ('Add', 'Sub', 'Sum'):
                for name in row['operands']:
                    if name in by_name:
                        operand = by_name[name]
                        print('      operand %-32s [%9.3g,%9.3g] -> '
                              '[%9.3g,%9.3g]' % (
                                  name[:32], operand['keep_lo'],
                                  operand['keep_hi'], operand['lo'],
                                  operand['hi']))
        if len(rows) > top:
            print('  ... %d more' % (len(rows) - top))
    if inner['sources'] is None:
        print('per-source breakdown unavailable: ' + inner['segments'])
        return
    print('per source, the furthest past first:')
    for source in sorted(inner['sources'], key=lambda s: -s['overshoot'])[:top]:
        if source['outside']:
            print('  %s (%d frames): %d tensors past, furthest %s by %s; '
                  'origins: %s' % (source['source'], source['frames'],
                                   source['outside'], source['worst'],
                                   _past(source['overshoot']),
                                   _names(source['origins'])))
        else:
            print('  %s (%d frames): inside' % (source['source'],
                                                source['frames']))


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

def _round(value):
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_round(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return float('%.6g' % value) if np.isfinite(value) else None
    return value


def write_report(analysis, out_path):
    """The HTML page for ``analysis`` (already passed through ``_round``)."""
    template_path = os.path.join(_HERE, 'calib_analyze_report.html')
    with open(template_path, encoding='utf-8') as stream:
        template = stream.read()
    data = json.dumps(analysis, separators=(',', ':'), ensure_ascii=False)
    data = data.replace('</', '<\\/')
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as stream:
        stream.write(template.replace('/*__DATA__*/null', data))


def report_conclusions(analysis, out_path):
    """Write the page for ``analysis`` (rounded) and print its conclusions."""
    write_report(analysis, out_path)
    for item in analysis['conclusions']:
        print('[%s] %s' % (item['level'], item['text']))
    print('report: %s' % os.path.abspath(out_path))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _bits(text):
    try:
        values = tuple(int(v) for v in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('expected widths like 8,16')
    if not values or any(not 2 <= v <= 32 for v in values):
        raise argparse.ArgumentTypeError('widths must be within 2..32')
    return values


def _add_common(parser, count=True):
    parser.add_argument('--scheme', choices=SCHEMES, default='asym',
                        help='input grid: asym (state inputs then take sym), '
                             'sym, or pow2 (symmetric, power-of-two scale, '
                             'as a fixed-point input format uses)')
    parser.add_argument('--bits', type=_bits, default=DEFAULT_BITS,
                        help='activation widths to score, e.g. 8,16 '
                             '(default: %(default)s)')
    if count:
        parser.add_argument('--count', type=int, default=DEFAULT_COUNT,
                            help='keep at most this many frames (default '
                                 '%(default)s, the converter\'s pattern cap)')
    parser.add_argument('--onnx', default=None, metavar='PATH',
                        help='the graph the frames bind to (default: the one '
                             'the calibration run wrote beside the artifact)')
    parser.add_argument('--inputs-only', action='store_true',
                        help='do not replay the graph: the inputs only')
    parser.add_argument('--jobs', type=int, default=1, metavar='N',
                        help='threads reading BIN files and replaying the '
                             'graph (default 1; the result does not depend '
                             'on it)')


@contextlib.contextmanager
def _graph_usage(parser):
    """A missing package, graph or file is a usage error."""
    try:
        yield
    except ImportError as error:
        parser.error('reading a graph needs the onnx and onnxruntime packages '
                     '(%s); pass --inputs-only to skip the replay' % error)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def _dump_json(value, path):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False)


def progress_line(count):
    print('  %d frames' % count, flush=True)


def _replay_graph(art, args):
    """The graph to replay ``art`` through, or None (with the reason)."""
    if args.inputs_only:
        return None
    if art.contiguous:
        print('note: the graph is not replayed: no selection is made for %s'
              % art.report.get('model_family'))
        return None
    path = _graph_file(art.path, art.report, args.onnx)
    if not os.path.isfile(path):
        if args.onnx:
            raise ValueError('no graph at %s' % path)
        print('note: no graph beside the artifact (%s): the inputs only' % path)
        return None
    import onnxruntime  # noqa: F401  (a missing package is a usage error)
    return path


def print_graph_note(pool):
    if pool.layers is None:
        return
    print('%d internal tensors followed' % len(pool.layers['names']))
    if not pool.layers['checkpoint_confirmed']:
        print('warning: no checkpoint_sha256 on both the graph and the '
              'report: cannot confirm they come from one checkpoint')


def _scan(art, graph, jobs, bands, reserve=SAMPLE_FRAMES):
    if graph:
        print('replaying %d frames of %s through %s' % (art.frames, art.path,
                                                        graph))
    pool = scan_artifact(art, graph, bands, jobs, progress_line, reserve)
    print_graph_note(pool)
    return pool


def analysis_of(pool, count, scheme, bits):
    """Analyse ``pool`` and print the notes and the comparison of the tensors
    inside the graph; return the analysis rounded for the page."""
    analysis = analyze(pool, scheme, count=count, bits=bits)
    recommend = analysis['recommend']
    if recommend:
        for note in recommend['notes']:
            print('note: ' + note)
        if recommend.get('layers'):
            print_layer_comparison(recommend['layers'])
    return _round(analysis)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_an = sub.add_parser('analyze', help='report, choose the patterns and, '
                                          'with --output, write them')
    p_an.add_argument('artifact')
    p_an.add_argument('--out', default='calib_report.html')
    p_an.add_argument('--json', default=None, help='also dump the raw analysis')
    p_an.add_argument('--output', default=None,
                      help='write the chosen frames here (.npz or a BIN '
                           'directory)')
    p_an.add_argument('--format', choices=('bin', 'npz'), default=None,
                      help='of --output; defaults to the artifact\'s')
    _add_common(p_an)
    p_co = sub.add_parser('coverage',
                          help='score a calibration artifact on a held-out one')
    p_co.add_argument('artifact')
    p_co.add_argument('--against', required=True,
                      help='held-out artifact of the same graph')
    p_co.add_argument('--json', default=None)
    _add_common(p_co, count=False)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be at least 1')

    with _graph_usage(parser):
        art = load_artifact(args.artifact, args.onnx, args.jobs)
        print('%s: %d frames, %d quantisable tensors%s' % (
            art.path, art.frames, len(art.names),
            (', skipped: ' + ', '.join(art.skipped)) if art.skipped else ''))
        if getattr(args, 'output', None):
            # Refuse a bad output before the scan, which takes minutes.
            args.output, args.format = calibration_output(
                args.output, args.format or art.fmt, (args.out, args.json))
            require_selectable(art)
        graph = _replay_graph(art, args)
        if args.cmd == 'coverage':
            held = load_artifact(args.against, graph or args.onnx, args.jobs)
            check_pair(art, held)       # before the replays, which take minutes
            cal_pool = _scan(art, graph, args.jobs, False)
            held_pool = _scan(held, graph, args.jobs, False)
        else:
            pool = _scan(art, graph, args.jobs, True, fill_reserve(args.count))

    if args.cmd == 'coverage':
        result = coverage(cal_pool, held_pool, args.scheme, args.bits)
        print_coverage(result, args.bits)
        if graph:
            result['layers'] = layer_coverage(cal_pool, held_pool, args.scheme)
            print_layer_coverage(result['layers'])
        if args.json:
            _dump_json(_round(result), args.json)
        return 0

    analysis = analysis_of(pool, args.count, args.scheme, args.bits)
    report_conclusions(analysis, args.out)
    if args.json:
        _dump_json(analysis, args.json)
    if args.output and analysis['recommend']:
        keep = np.asarray(analysis['recommend']['keep_frames'], dtype=np.int64)
        write_selection(pool, keep, args.output, args.format)
        print('wrote %d/%d frames -> %s' % (
            keep.size, pool.count,
            calibration_report_path(args.output, args.format)))
        print('the paired .onnx is unchanged; copy/symlink it beside the new '
              'artifact if the quantiser expects them together')
    return 0


if __name__ == '__main__':
    sys.exit(main())
