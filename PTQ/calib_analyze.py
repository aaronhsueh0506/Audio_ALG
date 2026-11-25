#!/usr/bin/env python3
r"""Choose the calibration patterns a min/max PTQ converter sees.

A calibration artifact is what a model's ``inference.py calib`` writes: one
array per graph input whose leading axis is the captured invocation ("frame"),
either a ``.npz`` (+ sibling ``.calib.json``) or a per-frame ``.bin`` directory
(+ ``manifest.json``).  The converter sets each tensor's range from the
minimum and maximum it sees, one range for the 8x8, 16x8 and 16x16 formats
alike.  A frame holding an outlier stretches the range for every other frame;
a frame left out has its values clipped on the device.  The tool reads the
recorded input tensors and works for any model; with ``--onnx`` it also runs
the graph the tensors bind to, so the tensors inside the graph count too.

Consecutive audio frames are near copies of each other, so capturing exactly
the pattern count (``--frames 1000``) covers one or two recordings.  Capture a
large pool from many recordings and let ``select`` keep the count:

    python3 AINR/DeepFilterNet2/inference.py calib --model ckpt.pth \
        --wav-dir wavs/ --frames 30000 --output calib/pool.npz
    python3 PTQ/calib_analyze.py analyze calib/pool.npz --out report.html
    python3 PTQ/calib_analyze.py select calib/pool.npz --output calib/pool_1000.npz
    python3 PTQ/calib_analyze.py coverage calib/pool.npz --keep keep.json \
        --against calib/heldout.npz
    python3 PTQ/calib_analyze.py apply calib/pool.npz --keep keep.json \
        --output calib/pool_pruned.npz
    # with the graph: keep the extremes of the tensors inside it as well
    python3 PTQ/calib_analyze.py select calib/pool.npz --onnx \
        --output calib/pool_1000.npz
    python3 PTQ/calib_analyze.py layers calib/pool.npz --keep keep.json

``select`` keeps at most ``--count`` frames (default 1000): every tensor's
extreme frames, then k-means over per-frame statistics so the rest still
covers the pool.  With ``--max-clip-pct`` above 0 it first drops the most
extreme frames while at most that percent of each tensor's recorded values
falls outside the kept range (counted exactly).  SQNR is reported for every
``--bits`` width.  ``coverage`` scores a selection against a held-out artifact
of the same graph.

Without ``--onnx`` the statistics cover the graph INPUT tensors only: a frame
whose inputs look ordinary can still hold the largest value of a tensor inside
the graph, and leaving it out narrows that tensor's range.  ``--onnx [PATH]``
(default: the graph the calibration run wrote beside the artifact) replays
every frame through the float graph with onnxruntime and records each internal
tensor's minimum and maximum per frame; ``select`` then keeps the frames that
hold those extremes too and prints what a choice from the inputs alone would
have lost, ``analyze`` reports the same, and ``layers`` compares a given
selection with the pool, tensor by tensor.  These are the float ranges of the
ONNX tensors: what a converter does to a range afterwards (merged operators,
table-based functions, tensors sharing one scale) is not modelled, and the
gates inside a recurrent operator are not visible.  Recurrent-state inputs (a tensor whose first frame is all
zero) are scored on a symmetric grid under ``--scheme asym``, since the
device needs zero point 0 on them; ``--scheme pow2`` scores every input on a
symmetric grid whose bound is rounded up to a power of two, the scale of a
fixed-point input format.  Frames are streaming snapshots, so a pruned set keeps each frame's own
recorded state but is no longer one stream: models that need one contiguous
stream are refused, and so is an Align-ULCNet artifact whose AEC front-end
stamp is missing or retired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AINR_DIR = os.path.join(os.path.dirname(_HERE), 'AINR')  # home of calibration_io
if _AINR_DIR not in sys.path:
    sys.path.insert(0, _AINR_DIR)

from calibration_io import (  # noqa: E402
    calibration_report_path,
    resolve_calibration_format,
    sibling_onnx_path,
    write_calibration_artifact,
)

SAMPLE_CAP = 2_000_000       # values per tensor behind the SQNR estimates
EPS = 1e-12
DEFAULT_COUNT = 1000         # the converter's recommended pattern cap
DEFAULT_BITS = (8, 16)       # activation widths of the 8x8 / 16x8 / 16x16 formats
ALT_CLIP_PCT = 0.1           # budget the report prices next to "keep everything"
RANDOM_DRAWS = 10
GROWTH_ORDERS = 20
FIXED_BAND_RATIO = 8.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class Artifact:
    """Calibration tensors, the report that came with them and their format."""

    def __init__(self, path, fmt, arrays, report):
        self.path = path
        self.fmt = fmt
        self.arrays = arrays
        self.report = report
        self.frames = int(next(iter(arrays.values())).shape[0])
        policy = report.get('state_precision_policy') or {}
        self.skipped = {}
        self.names = []
        for name, value in arrays.items():
            if name in policy:
                self.skipped[name] = policy[name]
            elif not np.issubdtype(value.dtype, np.floating):
                self.skipped[name] = 'non-float dtype %s' % value.dtype
            else:
                self.names.append(name)
        self.contiguous = report.get('contiguous_state_frames') is not None
        # A fresh source starts from the all-zero initial state.
        self.state = [name for name in self.names
                      if not np.any(arrays[name][0])]
        self.features = None        # frame_features() fills it on first use

    def schemes(self, scheme):
        """Grid per tensor: ``scheme``, but state inputs need zero point 0, so
        they take the symmetric grid when ``scheme`` is asymmetric."""
        return {name: 'sym' if name in self.state and scheme == 'asym'
                else scheme for name in self.names}


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


def load_artifact(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        report = _read_json(os.path.join(path, 'manifest.json'))
        if 'binary_tensors' not in report:
            raise ValueError('%s has no manifest.json with binary_tensors'
                             % path)
        require_current_frontend(report, path)
        frames = int(report['frames'])
        width = max(4, len(str(frames - 1)))
        arrays = {}
        for name, spec in report['binary_tensors'].items():
            # The manifest is untrusted input and the name becomes a path
            # component: reject absolute or ".."-containing names, as the
            # AINR/calibration_io.py writer does.
            if name != os.path.basename(name) or name in ('.', '..'):
                raise ValueError('invalid calibration tensor name in '
                                 'manifest.json: %r' % name)
            dtype = np.dtype(spec['dtype']).newbyteorder('<')
            shape = tuple(spec['frame_shape'])
            out = np.empty((frames,) + shape, dtype=dtype.newbyteorder('='))
            for index in range(frames):
                file = os.path.join(path, name,
                                    '%s_%0*d.bin' % (name, width, index))
                out[index] = np.fromfile(file, dtype=dtype).reshape(shape)
            arrays[name] = out
        return Artifact(path, 'bin', arrays, report)
    if path.endswith('.npz') and os.path.isfile(path):
        report = _npz_report(path)
        require_current_frontend(report, path)
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        return Artifact(path, 'npz', arrays, report)
    raise ValueError('expected a .npz file or a BIN directory: %s' % path)


# --------------------------------------------------------------------------
# Per-tensor statistics
# --------------------------------------------------------------------------

def _sample(values, rng):
    """Up to SAMPLE_CAP of ``values`` (flattened), drawn uniformly with
    replacement (an unbiased SQNR estimate; only the sample is copied)."""
    values = values.ravel()
    if values.size <= SAMPLE_CAP:
        return values
    return values[np.sort(rng.integers(0, values.size, SAMPLE_CAP))]


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


def _axis_concentration(x, axis):
    """How much a single index along ``axis`` dominates the absolute maximum."""
    other = (0,) + tuple(a for a in range(1, x.ndim) if a != axis % x.ndim)
    peak = np.maximum(x.max(axis=other), -x.min(axis=other)).astype(np.float64)
    return {
        'size': int(peak.size),
        'argmax': int(peak.argmax()),
        'ratio': float(peak.max() / (np.median(peak) + EPS)),
        'profile': [float(v) for v in peak],
    }


def tensor_stats(x, rng):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    fmin = flat.min(axis=1).astype(np.float64)
    fmax = flat.max(axis=1).astype(np.float64)
    gmin, gmax = float(fmin.min()), float(fmax.max())
    span = gmax - gmin
    loo_span = _loo_extreme(fmax, True) - _loo_extreme(fmin, False)
    shrink = (span - loo_span) / span if span > 0 else np.zeros(n)

    sample = _sample(flat, rng)
    s64 = sample.astype(np.float64)
    p001, p999 = (float(v) for v in np.percentile(s64, [0.1, 99.9]))
    abs_p999 = float(np.percentile(np.abs(s64), 99.9))
    mean, std = float(s64.mean()), float(s64.std())
    kurt = (float(np.mean((s64 - mean) ** 4) / (std ** 4 + EPS) - 3.0)
            if std > 0 else 0.0)
    absmax = max(abs(gmin), abs(gmax))

    conc = {}
    if x.ndim >= 4 and x.shape[2] > 1:
        conc['channel'] = _axis_concentration(x, 2)
    if x.shape[-1] > 1:
        conc['last'] = _axis_concentration(x, -1)

    return {
        'shape': list(x.shape),
        'global': {
            'min': gmin, 'max': gmax, 'absmax': absmax,
            'mean': mean, 'std': std, 'excess_kurtosis': kurt,
            'p001': p001, 'p999': p999, 'abs_p999': abs_p999,
            'tail_ratio': absmax / (abs_p999 + EPS),
            # Share of the range the central 99.8 % of the values occupies.
            'core_frac': (p999 - p001) / span if span > 0 else 1.0,
            'zero_frac': float(np.mean(sample == 0)),
            'sampled': bool(sample.size < flat.size),
        },
        'concentration': conc,
        'frame_min': fmin, 'frame_max': fmax, 'frame_shrink': shrink,
        '_flat': flat, '_sample': sample,
    }


def all_stats(art, seed=0):
    rng = np.random.default_rng(seed)
    return {name: tensor_stats(art.arrays[name], rng) for name in art.names}


# --------------------------------------------------------------------------
# Segments (source files) from state resets
# --------------------------------------------------------------------------

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
# Graph replay: the tensors inside the graph
# --------------------------------------------------------------------------

# Operators whose output is not followed: a layout-only one repeats its
# input, a Constant never changes.
UNFOLLOWED_OPS = frozenset(('Reshape', 'Transpose', 'Squeeze', 'Unsqueeze',
                            'Flatten', 'Identity', 'Constant'))
REPLAY_CHECK_FRAMES = 3      # frames compared against the unchanged graph


def graph_path(art, override=None):
    """The graph ``art`` binds to: ``override``, else the one its calibration
    run exported beside it (named in the report, or ``<stem>.onnx``)."""
    if override:
        path = override
    else:
        named = art.report.get('graph')
        beside = os.path.dirname(art.path.rstrip(os.sep))
        path = (os.path.join(beside, os.path.basename(named)) if named
                else sibling_onnx_path(art.path))
    if not os.path.isfile(path):
        raise ValueError('no graph at %s; pass --onnx PATH' % path)
    return os.path.abspath(path)


# Recorded by the exporter in the graph and by the capture in the report.
_GRAPH_PAIR_KEYS = ('checkpoint_sha256', 'state_layout_version')


def _check_graph_inputs(art, model, path):
    """Refuse a graph whose inputs are not the artifact's tensors."""
    from onnx import helper

    graph = model.graph
    constant = {init.name for init in graph.initializer}
    inputs = [value for value in graph.input if value.name not in constant]
    names = [value.name for value in inputs]
    if sorted(names) != sorted(art.arrays):
        raise ValueError('%s takes inputs %s but the artifact holds %s' % (
            path, sorted(names), sorted(art.arrays)))
    for value in inputs:
        dims = [dim.dim_value if dim.HasField('dim_value') else None
                for dim in value.type.tensor_type.shape.dim]
        have = art.arrays[value.name].shape[1:]
        if len(dims) != len(have) or any(
                want is not None and want != got
                for want, got in zip(dims, have)):
            raise ValueError('%s input %s is %s but the artifact holds %s per '
                             'frame' % (path, value.name, dims, list(have)))
        dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        if dtype != art.arrays[value.name].dtype:
            raise ValueError('%s input %s is %s but the artifact holds %s' % (
                path, value.name, dtype, art.arrays[value.name].dtype))
    props = {prop.key: prop.value for prop in model.metadata_props}
    for key in _GRAPH_PAIR_KEYS:
        want, have = props.get(key), art.report.get(key)
        if want and have is not None and str(want) != str(have):
            raise ValueError('%s records %s %s, the artifact %s' % (
                path, key, str(want)[:12], str(have)[:12]))
    return names, bool(props.get('checkpoint_sha256')
                       and art.report.get('checkpoint_sha256'))


def layer_stats(art, path, progress=None):
    """Per-frame minimum and maximum of every tensor inside the graph.

    Each frame carries its own recorded state inputs, so one frame is one
    complete invocation and is replayed on its own, as the converter does with
    each pattern.  A tensor is the float output of a node outside
    ``UNFOLLOWED_OPS``.  The first, middle and last frame are also run through
    the unchanged graph, whose outputs must match.
    """
    import onnx
    import onnxruntime as ort

    try:
        model = onnx.load(path)
    except Exception as error:      # protobuf raises its own decode error
        raise ValueError('cannot read the graph %s: %s' % (path, error))
    names, confirmed = _check_graph_inputs(art, model, path)
    graph = model.graph
    heads = [value.name for value in graph.output]
    ops = {name: node.op_type for node in graph.node
           if node.op_type not in UNFOLLOWED_OPS
           for name in node.output if name}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.log_severity_level = 3

    def session(proto):
        return ort.InferenceSession(proto.SerializeToString(), options,
                                    providers=['CPUExecutionProvider'])

    def feed(index):
        return {name: art.arrays[name][index] for name in names}

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend(onnx.ValueInfoProto(name=name) for name in ops
                              if name not in heads)
    plain, run = session(model), session(probe)
    first = run.run(list(ops), feed(0))
    tensors = [name for name, value in zip(ops, first)
               if isinstance(value, np.ndarray) and value.size
               and np.issubdtype(value.dtype, np.floating)]
    if not tensors:
        raise ValueError('%s has no float tensor to follow' % path)
    # Every head is fetched, followed or not, for the comparison below.
    fetch = tensors + [name for name in heads if name not in tensors]
    head_at = [fetch.index(name) for name in heads]

    n = art.frames
    fmin = np.empty((n, len(tensors)), dtype=np.float32)
    fmax = np.empty((n, len(tensors)), dtype=np.float32)
    check = set(np.linspace(0, n - 1, min(n, REPLAY_CHECK_FRAMES)).astype(int))
    for index in range(n):
        values = run.run(fetch, feed(index))
        for col in range(len(tensors)):
            fmin[index, col] = values[col].min()
            fmax[index, col] = values[col].max()
        if index in check:
            for at, want in zip(head_at, plain.run(heads, feed(index))):
                if not np.allclose(values[at], want, rtol=1e-5, atol=1e-6,
                                   equal_nan=True):
                    raise RuntimeError(
                        '%s: following the internal tensors changed output %s '
                        'at frame %d' % (path, fetch[at], index))
        if progress and (index + 1) % 5000 == 0:
            progress(index + 1, n)
    bad = ~(np.isfinite(fmin).all(axis=0) & np.isfinite(fmax).all(axis=0))
    if bad.any():
        raise ValueError('%s: non-finite values in %s' % (
            path, ', '.join(np.array(tensors)[bad][:5])))
    return {'graph': path, 'checkpoint_confirmed': confirmed,
            'names': tensors, 'ops': [ops[name] for name in tensors],
            'frame_min': fmin, 'frame_max': fmax}


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


def layer_table(layers, keep, base=None):
    """Each internal tensor's range over ``keep`` against ``base`` (default:
    every frame): the share of the range kept and the percent of ``base``
    frames holding a value outside it."""
    fmin, fmax = layers['frame_min'], layers['frame_max']
    base = np.arange(fmin.shape[0]) if base is None else base
    bmin, bmax = _rows(fmin, base), _rows(fmax, base)
    low, high = bmin.min(axis=0), bmax.max(axis=0)
    klow, khigh = fmin[keep].min(axis=0), fmax[keep].max(axis=0)
    kept = _share(khigh.astype(np.float64) - klow,
                  (high - low).astype(np.float64))
    outside = 100.0 * np.mean((bmin < klow) | (bmax > khigh), axis=0)
    rows = [{'tensor': name, 'op': op, 'lo': float(low[i]),
             'hi': float(high[i]), 'keep_lo': float(klow[i]),
             'keep_hi': float(khigh[i]), 'kept': float(kept[i]),
             'frames_outside_pct': float(outside[i])}
            for i, (name, op) in enumerate(zip(layers['names'],
                                               layers['ops']))]
    narrowed = [row for row in rows if row['frames_outside_pct'] > 0]
    return {'frames': int(len(keep)), 'base_frames': int(len(base)),
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
        # Tables, random draws and curve points often share a range.
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


def outside_count(stats, frames, lo, hi, limit=None):
    """Values of ``frames`` outside [lo, hi], counted exactly.

    A frame whose own min and max lie inside is skipped without reading its
    data; with ``limit`` the count stops once it is exceeded.
    """
    fmin, fmax, flat = stats['frame_min'], stats['frame_max'], stats['_flat']
    frames = np.asarray(frames, dtype=np.int64)
    hot = np.sort(frames[(fmin[frames] < lo) | (fmax[frames] > hi)])
    total = 0
    for start in range(0, hot.size, 512):
        block = flat[hot[start:start + 512]]
        total += int(np.count_nonzero((block < lo) | (block > hi)))
        if limit is not None and total > limit:
            break
    return total


def frame_order(stats):
    """Frames most extreme first.

    Score = worst ratio, over tensors, of the frame's abs-max to the median
    frame abs-max, so the ranking is comparable between tensors of different
    scale.
    """
    score = None
    for st in stats.values():
        peak = np.maximum(np.abs(st['frame_min']), np.abs(st['frame_max']))
        ratio = peak / (np.median(peak) + EPS)
        score = ratio if score is None else np.maximum(score, ratio)
    return np.argsort(-score, kind='stable'), score


def _suffix_ranges(stats, schemes, order):
    """Per tensor: the grid range left after dropping order[:k], for every k."""
    out = {}
    for name, st in stats.items():
        lo = np.minimum.accumulate(st['frame_min'][order][::-1])[::-1]
        hi = np.maximum.accumulate(st['frame_max'][order][::-1])[::-1]
        out[name] = grid_range(lo, hi, schemes[name])
    return out


def drop_budget(stats, schemes, order, max_clip_pct):
    """How many of the most extreme frames may go.

    The largest k for which dropping order[:k] leaves every tensor with at
    most ``max_clip_pct`` percent of all its recorded values outside the kept
    range.  Only dropped frames can hold such values, and the count grows with
    k, so a binary search over exact counts finds it.  A budget of 0 drops
    nothing.
    """
    n = order.size
    if max_clip_pct <= 0 or n < 2:
        return 0
    suffix = _suffix_ranges(stats, schemes, order)

    def fits(k):
        for name, st in stats.items():
            limit = max_clip_pct / 100.0 * st['_flat'].size
            lo, hi = suffix[name][0][k], suffix[name][1][k]
            if outside_count(st, order[:k], lo, hi, limit) > limit:
                return False
        return True

    good, bad = 0, n            # k = 0 always fits; k = n keeps nothing
    while bad - good > 1:
        mid = (good + bad) // 2
        if fits(mid):
            good = mid
        else:
            bad = mid
    return good


def budget_pool(stats, schemes, order, max_clip_pct):
    """Frames left after the clip budget, and how many it drops.

    The ``drop_budget`` prefix of the ranking goes, except the frames that lie
    inside the grid the rest leave for every tensor: they clip nothing (a
    power-of-two bound can sit well above the kept range) and taking them back
    leaves every grid as it is.
    """
    k = drop_budget(stats, schemes, order, max_clip_pct)
    if k == 0:
        return np.sort(order), 0
    rest, dropped = order[k:], order[:k]
    inside = np.ones(k, dtype=bool)
    for name, st in stats.items():
        lo, hi = grid_range(float(st['frame_min'][rest].min()),
                            float(st['frame_max'][rest].max()), schemes[name])
        inside &= (st['frame_min'][dropped] >= lo) & (st['frame_max'][dropped] <= hi)
    return np.sort(np.concatenate([rest, dropped[inside]])), int(k - inside.sum())


# --------------------------------------------------------------------------
# Selection: pick the patterns
# --------------------------------------------------------------------------

def frame_features(art):
    """Per-frame mean, std and abs-max of every tensor, z-scored later.
    Computed once per artifact."""
    if art.features is not None:
        return art.features
    cols = []
    for name in art.names:
        flat = art.arrays[name].reshape(art.frames, -1)
        mean = np.empty(art.frames)
        std = np.empty(art.frames)
        peak = np.empty(art.frames)
        for a in range(0, art.frames, 256):
            block = flat[a:a + 256].astype(np.float64)
            mean[a:a + 256] = block.mean(axis=1)
            std[a:a + 256] = block.std(axis=1)
            peak[a:a + 256] = np.abs(block).max(axis=1)
        cols += [mean, std, peak]
    art.features = np.stack(cols, axis=1)
    return art.features


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


def select_patterns(art, stats, schemes, count=DEFAULT_COUNT, max_clip_pct=0.0,
                    seed=0, layers=None):
    """Choose which frames to calibrate with.

    1. Rank frames most extreme first (``frame_order``).
    2. Drop the first ``drop_budget`` of them (none at a budget of 0): those
       values get clipped in exchange for finer steps for the rest.  Dropped
       frames that clip nothing come back (``budget_pool``).
    3. If more than ``count`` frames remain, keep the frames that hold each
       tensor's minimum and maximum (otherwise the range would shrink past the
       budget) and thin the rest with k-means so the survivors still cover the
       distribution.  ``count`` is a cap: a smaller pool is kept whole.

    With ``layers`` (``layer_stats``) step 3 also keeps the frames holding the
    minimum and maximum of every tensor inside the graph (``layer_holders``),
    after the input ones and most important first when ``count`` cannot take
    them all.  ``must`` counts the range-defining frames found, ``must_layers``
    those found for the internal tensors only and ``kept_layers`` how many of
    the latter are kept.
    """
    if count < 1:
        raise ValueError('--count must be at least 1')
    rng = np.random.default_rng(seed)
    order, _ = frame_order(stats)
    pool, k_budget = budget_pool(stats, schemes, order, max_clip_pct)
    notes = []
    must = np.array([], dtype=np.int64)
    must_layers = kept_layers = 0
    if count >= pool.size:
        keep = pool
        if count > pool.size:
            notes.append('only %d frames remain, fewer than --count %d; all '
                         'are kept' % (pool.size, count))
    else:
        holders = set()
        for st in stats.values():
            holders.add(int(pool[np.argmin(st['frame_min'][pool])]))
            holders.add(int(pool[np.argmax(st['frame_max'][pool])]))
        must = np.array(sorted(holders), dtype=np.int64)
        if layers is not None:
            inner = layer_holders(layers, pool, must)
            must_layers = int(inner.size)
            must = np.concatenate([must, inner])
            # The internal holders come last, so a short count cuts them first.
            kept_layers = max(0, min(count, must.size) - (must.size - must_layers))
        if must.size >= count:
            keep = np.sort(must[:count])
            if must.size > count:
                notes.append('%d range-defining frames exceed --count %d; '
                             'the range will be narrower than the budget '
                             'allows' % (must.size, count))
        else:
            rest = np.setdiff1d(pool, must)
            features = frame_features(art)[rest]
            picked = rest[_kmeans_pick(features, count - must.size, rng)]
            keep = np.sort(np.concatenate([must, picked]))
    return {'keep': keep, 'k_budget': k_budget, 'must': int(must.size),
            'notes': notes, 'pool': pool, 'must_layers': must_layers,
            'kept_layers': kept_layers}


def frames_outside_pct(st, lo, hi, part=slice(None)):
    """Percent of the frames in ``part`` with a value outside [lo, hi]."""
    return 100.0 * float(np.mean((st['frame_min'][part] < lo)
                                 | (st['frame_max'][part] > hi)))


def score_row(name, st, model, lo, hi, frames, bits):
    """One tensor's row for the range [lo, hi].

    ``clip_pct`` is the percent of all recorded values outside the range
    (counted over ``frames``, the only ones that can hold such values),
    ``frames_clip_pct`` the percent of frames with at least one, then the
    SQNR per width.
    """
    row = {'tensor': name, 'lo': float(lo), 'hi': float(hi),
           'clip_pct': 100.0 * outside_count(st, frames, lo, hi)
           / st['_flat'].size,
           'frames_clip_pct': frames_outside_pct(st, lo, hi)}
    for b in bits:
        m = model.evaluate(lo, hi, b)
        row['sqnr%d_all' % b] = float(m['sqnr_all'])
        row['sqnr%d_typ' % b] = float(m['sqnr_typical'])
    return row


def subset_table(stats, schemes, models, keep, n, label, bits):
    """Per tensor for ``keep``: grid range, exact clip shares, SQNR per width."""
    kept = np.zeros(n, dtype=bool)
    kept[keep] = True
    others = np.flatnonzero(~kept)
    rows = []
    for name, st in stats.items():
        lo, hi = grid_range(float(st['frame_min'][kept].min()),
                            float(st['frame_max'][kept].max()), schemes[name])
        rows.append(score_row(name, st, models[name], lo, hi, others, bits))
    return {'label': label, 'frames': int(kept.sum()), 'rows': rows}


def random_baseline(stats, schemes, models, n, count, bits, draws=RANDOM_DRAWS,
                    seed=1):
    """Median (and worst) of ``draws`` uniformly random subsets of ``count``."""
    rng = np.random.default_rng(seed)
    tables = [subset_table(stats, schemes, models,
                           rng.choice(n, count, replace=False), n, '', bits)
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
    head = '%-30s %-16s %-22s %9s %8s' % ('subset', 'tensor', 'range',
                                         'clip%', 'frames%')
    for b in bits:
        head += ' %9s %9s' % ('typ%d' % b, 'all%d' % b)
    print(head)
    for table in tables:
        for index, row in enumerate(table['rows']):
            label = ('%s (%d)' % (table['label'], table['frames'])
                     if index == 0 else '')
            line = '%-30s %-16s [%9.3g,%9.3g] %9.4f %8.2f' % (
                label[:30], row['tensor'][:16], row['lo'], row['hi'],
                row['clip_pct'], row['frames_clip_pct'])
            for b in bits:
                line += ' %9.2f %9.2f' % (row['sqnr%d_typ' % b],
                                          row['sqnr%d_all' % b])
            print(line)


# --------------------------------------------------------------------------
# Sources: per-file contribution and range growth
# --------------------------------------------------------------------------

def source_report(art, stats, keep, score, seed=0):
    """Per source: frames, kept, extremeness, leave-one-source-out shrink.

    Only when the state resets match the recorded source files; otherwise the
    section reports why it is unavailable instead of guessing.
    """
    starts, reliable, _ = detect_segments(art)
    recorded = len(art.report.get('source_files') or [])
    if not art.state:
        note = '沒有任何 tensor 的第一個 frame 全為 0，無法用 state 重置切出來源'
    elif not reliable:
        note = '偵測到 %d 次 state 重置，但報告記錄了 %d 個來源檔' % (
            len(starts), recorded)
    else:
        note = '%d 次 state 重置對上 %d 個來源檔' % (len(starts), recorded)
    if not reliable or len(starts) < 2:
        return {'available': False, 'note': note, 'single': reliable}
    n = art.frames
    bounds = np.array(starts + [n])
    sources = art.report.get('source_files') or []
    seg_of_keep = (np.searchsorted(bounds, keep, side='right') - 1
                   if keep is not None else None)
    kept = (np.bincount(seg_of_keep, minlength=len(starts))
            if keep is not None else None)
    seg_score = np.maximum.reduceat(score, starts)
    shrink = np.zeros(len(starts))
    shrink_tensor = [''] * len(starts)
    seg_min, seg_max = {}, {}
    for name, st in stats.items():
        lo = np.minimum.reduceat(st['frame_min'], starts)
        hi = np.maximum.reduceat(st['frame_max'], starts)
        seg_min[name], seg_max[name] = lo, hi
        span = hi.max() - lo.min()
        if span <= 0:
            continue
        loo = _loo_extreme(hi, True) - _loo_extreme(lo, False)
        part = (span - loo) / span
        for i in np.flatnonzero(part > shrink):
            shrink[i], shrink_tensor[i] = part[i], name
    rows = [{'id': i, 'source': sources[i], 'start': int(starts[i]),
             'frames': int(bounds[i + 1] - bounds[i]),
             'kept': int(kept[i]) if kept is not None else None,
             'extremeness': float(seg_score[i]),
             'loso_shrink': float(shrink[i]), 'loso_tensor': shrink_tensor[i]}
            for i in range(len(starts))]

    # Range growth: sources added in random orders, the least-converged
    # tensor's share of its final span after each addition.
    rng = np.random.default_rng(seed)
    count = len(starts)
    curves = np.ones((GROWTH_ORDERS, count))
    for p in range(GROWTH_ORDERS):
        perm = rng.permutation(count)
        share = np.ones(count)
        for name in stats:
            lo = np.minimum.accumulate(seg_min[name][perm])
            hi = np.maximum.accumulate(seg_max[name][perm])
            span = hi[-1] - lo[-1]
            if span > 0:
                share = np.minimum(share, (hi - lo) / span)
        curves[p] = share
    return {'available': True, 'note': note, 'rows': rows,
            'growth': {'median': np.median(curves, 0),
                       'p10': np.percentile(curves, 10, axis=0)}}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _drop_grid(units):
    cap = units // 2
    grid = {0}
    value = 1.0
    while value <= cap:
        grid.add(int(round(value)))
        value *= 1.35
    grid.add(cap)
    return sorted(g for g in grid if g <= cap)


def build_curves(stats, schemes, models, order, bits):
    """SQNR and frames-with-clipping against frames dropped, most extreme first."""
    grid = _drop_grid(order.size)
    suffix = _suffix_ranges(stats, schemes, order)
    out = {'k': grid, 'tensors': {}}
    for name, st in stats.items():
        lo, hi = suffix[name][0][grid], suffix[name][1][grid]
        entry = {'frames_clip': [frames_outside_pct(st, a, b)
                                 for a, b in zip(lo, hi)]}
        for b in bits:
            m = models[name].evaluate(lo, hi, b)
            entry['b%d' % b] = {'all': m['sqnr_all'], 'typ': m['sqnr_typical']}
        out['tensors'][name] = entry
    return out


def _signed(value):
    """'+1.2' / '-0.4' / '+0.0' (never '-0.0')."""
    return '%+.1f' % (round(value, 1) + 0.0)


def _worst_delta(base, other, key):
    """Min over tensors of other - base for ``key`` (rows in the same order)."""
    return min(o[key] - b[key] for b, o in zip(base['rows'], other['rows']))


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


def _layer_lines(inner, must):
    """Conclusions about the tensors inside the graph."""
    selected, plain = inner['selected'], inner['input_only']
    frames = selected['frames']
    out = [('info', '已用 %s 把 %d 個 frame 逐一重跑（FP32），記錄 graph 內部 %d 個 '
            'tensor 每個 frame 的最小值和最大值。%s' % (
                os.path.basename(inner['graph']), inner['replayed'],
                selected['tensors'],
                '' if inner['checkpoint_confirmed'] else
                'graph 和校正檔沒有都記錄 checkpoint，無法確認兩者出自同一個 checkpoint。'))]
    text = '挑出的 %d 個 frame：%s。' % (frames, _narrowed_text(selected))
    if must:
        held = min(must, frames)
        text += ('保住輸入和內部的極值共用了 %d 個 frame（為內部 tensor 多留的有 %d 個），'
                 '佔 %.0f%%；其餘才是依分布抽樣。佔比高時，這份資料會偏向極端值：'
                 '純 min/max 不受影響，但 converter 依直方圖的校正模式（只作用在 8-bit）'
                 '會受影響。' % (held, inner['kept_layers'],
                              100.0 * held / frames))
    out.append((_level(selected), text))
    out.append((_level(plain), '對照：只看輸入來挑（不加 --onnx）的 %d 個 frame：%s。'
                % (plain['frames'], _narrowed_text(plain))))
    whole = inner.get('all_frames')
    if whole:
        out.append((_level(whole), '上面兩項的基準是套用 clip 預算後剩下的 %d 個 frame。'
                    '預算只依輸入丟 frame，被丟的 frame 也可能帶著內部 tensor 的極值；'
                    '以全部 %d 個 frame 為基準：%s。' % (
                        selected['base_frames'], whole['base_frames'],
                        _narrowed_text(whole))))
    out.append(('info', '內部的 range 是 FP32 graph 每個 ONNX tensor 的最小值和最大值。'
                'converter 之後還會調整（合併 op、查表類函數夾範圍、多個 tensor 共用 scale），'
                '遞迴 op 內部的 gate 也看不到，所以這裡只保證「挑選後和整池看到的極值相同」，'
                '不代表量化後的精度。'))
    return out


def conclusions(art, stats, schemes, recommend, sources, bits, count):
    """What the numbers say, as (level, text) pairs in the report language."""
    out = []
    n = art.frames
    if art.contiguous:
        out.append(('warn', '%s 需要一段不中斷的串流，不能挑選或刪減 frame。'
                    % art.report.get('model_family')))
    if sources['available']:
        rows = sources['rows']
        frames = np.array([r['frames'] for r in rows])
        out.append(('info', '%d 個 frame 來自 %d 個來源（每個來源 %d–%d 個 frame，'
                    '最大的一個佔 %.1f%%）。' % (
                        n, len(rows), frames.min(), frames.max(),
                        100.0 * frames.max() / n)))
        if len(rows) <= 3:
            out.append(('warn', '只有 %d 個來源。相鄰 frame 幾乎相同，來源太少時'
                        '涵蓋的情境很有限；建議從更多錄音擷取大池再挑選。'
                        % len(rows)))
        top = sorted(rows, key=lambda r: -r['loso_shrink'])[:3]
        top = [r for r in top if r['loso_shrink'] > 0]
        if top:
            out.append(('info', 'range 由少數來源撐起：' + '；'.join(
                '拿掉「%s」，%s 的 range 縮小 %.1f%%' % (
                    r['source'], r['loso_tensor'], 100 * r['loso_shrink'])
                for r in top) + '。值得去聽這些來源，確認是真實會遇到的情境'
                '還是壞資料。'))
        growth = sources['growth']['median']
        at90 = growth[max(0, int(np.ceil(0.9 * len(rows))) - 1)]
        out.append(('info', '依隨機順序加入來源（%d 種順序的中位數），加入最後 10%% '
                    '的來源時，最不穩定的 tensor 的 range 仍擴大了 %.1f%%。數字越大，'
                    '代表再多錄一些資料，可能還會出現新的極值；這個數字沒有驗證過的門檻。'
                    % (GROWTH_ORDERS, 100 * (1 - at90))))
    elif sources.get('single'):
        out.append(('warn', '所有 %d 個 frame 都來自同一個來源。相鄰 frame 幾乎相同，'
                    '涵蓋的情境很有限；建議從多個錄音擷取大池再挑選。' % n))
    else:
        out.append(('warn', '來源資訊不可用（%s）：無法看每個來源的貢獻，也無法判斷 '
                    'range 是否已經收斂。' % sources['note']))
    if recommend and n <= count:
        out.append(('warn', '整份校正檔只有 %d 個 frame，不超過要挑的 %d 個，等於沒有挑選。'
                    '直接擷取目標數量只會涵蓋少數錄音；請先擷取數萬個 frame 的大池，'
                    '再用 select 挑。' % (n, count)))
    for name, st in stats.items():
        for axis, label in (('channel', 'channel'), ('last', '最後一軸的位置')):
            c = st['concentration'].get(axis)
            if c and c['ratio'] > FIXED_BAND_RATIO:
                out.append(('warn', '%s 的極端值固定落在第 %d 個%s（%.1f 倍中位數）：'
                            '這是特徵縮放的問題，刪 frame 解決不了。' % (
                                name, c['argmax'], label, c['ratio'])))
    if art.state:
        grid = schemes[art.state[0]]
        out.append(('info', '%s 被視為 recurrent state（第一個 frame 全為 0），'
                    '以%s grid 評分：板端 state port 的 zero point 必須是 0。'
                    % ('、'.join(art.state), SCHEME_NAMES[grid])))
    out += _pow2_lines(stats, schemes, bits, n)
    if recommend:
        sel = recommend['selected']
        inner = recommend.get('layers')
        must_inputs = recommend['must'] - (inner['must_layers'] if inner else 0)
        if recommend['k_budget'] == 0 and must_inputs > sel['frames']:
            out.append(('warn', '保住每個輸入 tensor 的最小值和最大值需要 %d 個 frame，'
                        '超過 --count %d：挑出的 frame 的輸入 range 比整池窄。'
                        % (must_inputs, count)))
        elif recommend['k_budget'] == 0 and recommend['must'] > sel['frames']:
            out.append(('warn', '保住輸入和內部每個 tensor 的最小值和最大值需要 %d 個 '
                        'frame，超過 --count %d：輸入的 %d 個都保留，內部的 %d 個依影響'
                        '大小只保留了 %d 個。' % (
                            recommend['must'], count, must_inputs,
                            inner['must_layers'], inner['kept_layers'])))
        elif recommend['k_budget'] == 0 and recommend['must']:
            text = ('挑出的 %d 個 frame 保有整池每個輸入 tensor 的最小值和'
                    '最大值，所以輸入 range 和整池相同。' % sel['frames'])
            if not inner:
                text += ('保住這些極值至少需要 %d 個 frame，但這只對輸入成立；'
                         'graph 內部各層的 range，這個工具看不到。'
                         % recommend['must'])
            out.append(('info', text))
        elif recommend['k_budget'] == 0:
            out.append(('info', '全部 %d 個 frame 都保留，輸入 range 和整池相同。'
                        % sel['frames']))
        else:
            out.append(('info', '依 %.3g%% 預算丟了 %d 個最極端的 frame，再挑出 %d 個。'
                        % (recommend['max_clip_pct'], recommend['k_budget'],
                           sel['frames'])))
        if inner:
            out += _layer_lines(inner, recommend['must'])
        alt = recommend.get('alternative')
        if alt:
            base = recommend['all']
            deltas = '；'.join(
                '%d-bit 含被夾 %s～%s dB、不含被夾最多 %s dB' % (
                    b, _signed(_worst_delta(base, alt, 'sqnr%d_all' % b)),
                    _signed(-_worst_delta(alt, base, 'sqnr%d_all' % b)),
                    _signed(-_worst_delta(alt, base, 'sqnr%d_typ' % b)))
                for b in bits)
            worst_frames = max(r['frames_clip_pct'] for r in alt['rows'])
            out.append(('info', '若改成丟極端值（預算 %.3g%%）：會丟 %d 個 frame（%.1f%%）；'
                        'SQNR 變化（各 tensor 的範圍）%s；有值被夾的 frame 最多 %.2f%%。'
                        '三種格式共用同一份資料時，8-bit 的收益要和 16-bit 的損失一起看；'
                        '只有 8x8 要夾極值時，也可以考慮 converter 自己的校正模式（逐 tensor 夾，'
                        '包含 graph 內部各層；用前先確認它對偏斜特徵怎麼夾），不必從共用資料丟 frame。'
                        % (ALT_CLIP_PCT, alt['k_budget'],
                           100.0 * alt['k_budget'] / n, deltas, worst_frames)))
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
                        'SQNR 在最受影響的 tensor 上為 %s。' % (
                            rnd['frames'], RANDOM_DRAWS, '；'.join(parts))))
        if alt or rnd:
            out.append(('info', '上面兩組對照都用整池本身評分：整池的極值就在評分資料裡，'
                        '所以會高估丟掉極值的代價。不偏的數字請用 coverage 子命令，'
                        '對另一份沒看過的錄音評估。'))
    return out


def prepare(art, scheme, seed=0):
    """Per-tensor statistics, grid names and SQNR models of ``art``."""
    stats = all_stats(art, seed)
    schemes = art.schemes(scheme)
    models = {name: GridError(stats[name]['_sample'], schemes[name])
              for name in art.names}
    return stats, schemes, models


def layer_comparison(art, stats, schemes, layers, sel, count, max_clip_pct,
                     seed=0):
    """``sel`` (chosen with ``layers``) and the input-only choice, each against
    the frames left after the clip budget; when the budget dropped frames,
    ``sel`` against every frame too, since the budget ranks by the inputs."""
    plain = select_patterns(art, stats, schemes, count, max_clip_pct, seed)
    out = {'graph': layers['graph'],
           'checkpoint_confirmed': layers['checkpoint_confirmed'],
           'replayed': art.frames, 'must_layers': sel['must_layers'],
           'kept_layers': sel['kept_layers'],
           'selected': layer_table(layers, sel['keep'], sel['pool']),
           'input_only': layer_table(layers, plain['keep'], sel['pool'])}
    if sel['k_budget']:
        out['all_frames'] = layer_table(layers, sel['keep'])
    return out


def print_layer_comparison(inner):
    print_layer_table(inner['input_only'], 'selected from the inputs only')
    print_layer_table(inner['selected'], 'selected with the graph')
    if 'all_frames' in inner:
        print_layer_table(inner['all_frames'],
                          'selected with the graph, against every frame')


def analyze(art, scheme='asym', seed=0, count=DEFAULT_COUNT, max_clip_pct=0.0,
            bits=DEFAULT_BITS, layers=None):
    stats, schemes, models = prepare(art, scheme, seed)
    order, score = frame_order(stats)
    n = art.frames
    recommend = None
    keep = None
    if not art.contiguous:
        sel = select_patterns(art, stats, schemes, count, max_clip_pct, seed,
                              layers)
        keep = sel['keep']
        recommend = {
            'keep_frames': keep, 'count': count, 'max_clip_pct': max_clip_pct,
            'k_budget': sel['k_budget'], 'must': sel['must'],
            'notes': sel['notes'],
            'all': subset_table(stats, schemes, models, np.arange(n), n,
                                'all frames', bits),
            'selected': subset_table(stats, schemes, models, keep, n,
                                     'selected', bits),
        }
        if keep.size < n:
            recommend['random'] = random_baseline(stats, schemes, models, n,
                                                  int(keep.size), bits)
        if max_clip_pct <= 0:
            alt_pool, k_alt = budget_pool(stats, schemes, order, ALT_CLIP_PCT)
            alt = subset_table(stats, schemes, models, alt_pool, n,
                               'drop extremes %.3g%%' % ALT_CLIP_PCT, bits)
            alt['k_budget'] = k_alt
            alt['max_clip_pct'] = ALT_CLIP_PCT
            recommend['alternative'] = alt
        if layers is not None:
            recommend['layers'] = layer_comparison(
                art, stats, schemes, layers, sel, count, max_clip_pct, seed)
    sources = source_report(art, stats, keep, score, seed)
    return {
        'meta': {
            'path': art.path, 'format': art.fmt, 'frames': n,
            'model_family': art.report.get('model_family'),
            'scheme': scheme, 'schemes': schemes, 'bits': list(bits),
            'contiguous': art.contiguous, 'skipped': art.skipped,
            'state': art.state, 'sample_cap': SAMPLE_CAP,
            'fixed_band_ratio': FIXED_BAND_RATIO,
        },
        'conclusions': [{'level': level, 'text': text} for level, text in
                        conclusions(art, stats, schemes, recommend, sources,
                                    bits, count)],
        'recommend': recommend,
        'sources': sources,
        'frame_order': order,
        'frame_score': score,
        'curves': build_curves(stats, schemes, models, order, bits),
        'tensors': {
            name: {k: v for k, v in s.items() if not k.startswith('_')}
            for name, s in stats.items()
        },
    }


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
    """Refuse a held-out artifact of another graph; return warnings."""
    if cal.names != held.names:
        raise ValueError('tensor names differ: %s vs %s' % (cal.names,
                                                            held.names))
    for name in cal.names:
        if cal.arrays[name].shape[1:] != held.arrays[name].shape[1:]:
            raise ValueError('%s frame shape differs: %s vs %s' % (
                name, cal.arrays[name].shape[1:], held.arrays[name].shape[1:]))
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


def coverage(cal, held, keep=None, scheme='asym', bits=DEFAULT_BITS, seed=0):
    """Ranges from ``cal`` (all frames or ``keep``) scored on ``held``."""
    warnings = check_pair(cal, held)
    schemes = cal.schemes(scheme)
    frames = np.arange(cal.frames) if keep is None else keep
    hstats = all_stats(held, seed)
    starts, reliable, note = detect_segments(held)
    bounds = starts + [held.frames]
    sources = held.report.get('source_files') or []
    per_source = reliable and len(starts) > 1
    if reliable and not per_source:
        note = 'a single source'
    rows = []
    for name in cal.names:
        flat = cal.arrays[name].reshape(cal.frames, -1)
        if keep is not None:
            flat = flat[keep]
        lo, hi = grid_range(float(flat.min()), float(flat.max()),
                            schemes[name])
        st = hstats[name]
        row = score_row(name, st, GridError(st['_sample'], schemes[name]),
                        lo, hi, np.arange(held.frames), bits)
        if per_source:
            # (value, source); a frames-clipped entry stays (0, None) when no
            # source has a clipped frame.
            worst = {'frames_clip_pct': (0.0, None)}
            for i in range(len(starts)):
                a, z = bounds[i], bounds[i + 1]
                seg = st['_flat'][a:z]
                seg_model = GridError(_sample(seg, np.random.default_rng(i)),
                                      schemes[name])
                frac = frames_outside_pct(st, lo, hi, slice(a, z))
                if frac > worst['frames_clip_pct'][0]:
                    worst['frames_clip_pct'] = (frac, sources[i])
                for b in bits:
                    value = float(seg_model.evaluate(lo, hi, b)['sqnr_all'])
                    key = 'sqnr%d_all' % b
                    if value < worst.get(key, (np.inf, None))[0]:
                        worst[key] = (value, sources[i])
            row['worst_source'] = worst
        rows.append(row)
    return {'calibration': cal.path, 'held_out': held.path,
            'frames': int(len(frames)), 'held_out_frames': held.frames,
            'warnings': warnings,
            'segments': None if per_source else note, 'rows': rows}


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
        worst = row['worst_source']
        frac, src = worst['frames_clip_pct']
        line = '%-16s worst source: %s' % (
            row['tensor'], 'frames clipped %.2f%% (%s)' % (frac, src)
            if src else 'no clipped frame in any source')
        for b in bits:
            value, src = worst['sqnr%d_all' % b]
            line += '; SQNR%d all %.1f dB (%s)' % (b, value, src)
        print(line)


# --------------------------------------------------------------------------
# apply: write a reduced artifact
# --------------------------------------------------------------------------

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


def keep_indices(art, spec):
    n = art.frames
    if 'keep_frames' in spec:
        keep = np.unique(np.asarray(spec['keep_frames'], dtype=np.int64))
    elif 'drop_frames' in spec:
        drop = np.asarray(spec['drop_frames'], dtype=np.int64)
        if drop.size and (drop.min() < 0 or drop.max() >= n):
            raise ValueError('drop list is outside 0..%d' % (n - 1))
        keep = np.setdiff1d(np.arange(n), drop)
    else:
        raise ValueError('keep file needs keep_frames or drop_frames')
    if keep.size == 0 or keep.min() < 0 or keep.max() >= n:
        raise ValueError('keep list is empty or outside 0..%d' % (n - 1))
    return keep


def require_selectable(art):
    if art.contiguous:
        raise SystemExit(
            '%s needs one uninterrupted stream; its cumulative state ranges '
            'are meaningless on a selected or pruned set. Regenerate with a '
            'different source instead.' % art.report.get('model_family'))


def apply_selection(art, keep, output, fmt):
    require_current_frontend(art.report, art.path)
    require_selectable(art)
    arrays = {name: value[keep] for name, value in art.arrays.items()}
    report = dict(art.report)
    starts, reliable, _ = detect_segments(art)
    sources = report.get('source_files') or []
    if reliable and len(starts) > 1:
        bounds = starts + [art.frames]
        alive = [i for i in range(len(starts))
                 if ((keep >= bounds[i]) & (keep < bounds[i + 1])).any()]
        gone = [sources[i] for i in range(len(starts)) if i not in alive]
        if gone:
            print('note: %d source(s) have no frame left and leave '
                  'source_files: %s' % (len(gone), ', '.join(gone)))
        report['source_files'] = [sources[i] for i in alive]
    report['frames'] = int(keep.size)
    report['inputs'] = _range_report(
        arrays, report.get('state_precision_policy') or {})
    report['selection'] = {
        'tool': 'calib_analyze',
        'parent': os.path.basename(art.path),
        'parent_frames': art.frames,
        'kept_frames': int(keep.size),
        'kept_indices': [int(i) for i in keep],
    }
    for key in ('binary_tensors', 'artifact_format', 'binary_file_pattern',
                'binary_byte_order', 'binary_frame_index_base'):
        report.pop(key, None)
    return write_calibration_artifact(output, arrays, report, fmt)


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
    with open(out_path, 'w', encoding='utf-8') as stream:
        stream.write(template.replace('/*__DATA__*/null', data))


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


def _add_onnx(parser, default=None):
    parser.add_argument('--onnx', nargs='?', const='', default=default,
                        metavar='PATH',
                        help='also follow the tensors inside the graph '
                             '(needs onnx and onnxruntime); without PATH, the '
                             'graph the calibration run wrote beside the '
                             'artifact. Put it after the artifact, or write '
                             '--onnx=PATH')


def load_layers(art, override):
    """``layer_stats`` of the graph ``art`` binds to, with progress lines."""
    path = graph_path(art, override)
    print('replaying %d frames through %s' % (art.frames, path))
    layers = layer_stats(art, path, lambda done, total: print(
        '  %d/%d frames' % (done, total), flush=True))
    print('%d internal tensors followed' % len(layers['names']))
    if not layers['checkpoint_confirmed']:
        print('warning: no checkpoint_sha256 on both the graph and the '
              'report: cannot confirm they come from one checkpoint')
    return layers


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
        parser.add_argument('--max-clip-pct', type=float, default=0.0,
                            help='percent of each tensor\'s recorded values '
                                 'allowed outside the kept range; 0 (default) '
                                 'keeps every extreme frame')


def _load_keep(art, path):
    with open(path, encoding='utf-8') as stream:
        return keep_indices(art, json.load(stream))


def _dump_json(value, path):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_an = sub.add_parser('analyze', help='write an interactive HTML report')
    p_an.add_argument('artifact')
    p_an.add_argument('--out', default='calib_report.html')
    p_an.add_argument('--json', default=None, help='also dump the raw analysis')
    _add_common(p_an)
    _add_onnx(p_an)
    p_se = sub.add_parser('select', help='pick which patterns to keep')
    p_se.add_argument('artifact')
    _add_common(p_se)
    p_se.add_argument('--keep-out', default='keep.json',
                      help='where to write the selection (for apply)')
    p_se.add_argument('--output', default=None,
                      help='also write the reduced artifact here')
    p_se.add_argument('--format', choices=('bin', 'npz'), default=None)
    _add_onnx(p_se)
    p_la = sub.add_parser(
        'layers', help='compare a selection with the pool, tensor by tensor '
                       'inside the graph')
    p_la.add_argument('artifact')
    p_la.add_argument('--keep', required=True,
                      help='JSON with keep_frames or drop_frames')
    p_la.add_argument('--json', default=None)
    _add_onnx(p_la, default='')     # the graph is this command's subject
    p_co = sub.add_parser('coverage',
                          help='score a selection on a held-out artifact')
    p_co.add_argument('artifact')
    p_co.add_argument('--against', required=True,
                      help='held-out artifact of the same graph')
    p_co.add_argument('--keep', default=None,
                      help='JSON with keep_frames or drop_frames (default: all)')
    p_co.add_argument('--json', default=None)
    _add_common(p_co, count=False)
    p_ap = sub.add_parser('apply', help='write a reduced calibration artifact')
    p_ap.add_argument('artifact')
    p_ap.add_argument('--keep', required=True,
                      help='JSON with keep_frames or drop_frames')
    p_ap.add_argument('--output', required=True)
    p_ap.add_argument('--format', choices=('bin', 'npz'), default=None,
                      help='defaults to the input artifact format')
    _add_common(p_ap, count=False)
    args = parser.parse_args(argv)

    art = load_artifact(args.artifact)
    if getattr(args, 'output', None):
        # Refuse a bad output name before select writes the keep file.
        args.format = args.format or art.fmt
        try:
            resolve_calibration_format(args.output, args.format)
        except ValueError as error:
            parser.error(str(error))
    print('%s: %d frames, %d quantisable tensors%s' % (
        art.path, art.frames, len(art.names),
        (', skipped: ' + ', '.join(art.skipped)) if art.skipped else ''))

    layers = None
    if args.cmd in ('select', 'layers'):
        require_selectable(art)     # before the replay, which takes minutes
    if getattr(args, 'onnx', None) is not None and art.contiguous:
        print('note: --onnx is ignored: no selection is made for %s'
              % art.report.get('model_family'))
    elif getattr(args, 'onnx', None) is not None:
        try:
            if args.cmd == 'layers':
                keep = _load_keep(art, args.keep)
            layers = load_layers(art, args.onnx)
        except ImportError as error:
            parser.error('following the tensors inside the graph needs the '
                         'onnx and onnxruntime packages (%s)' % error)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    if args.cmd == 'layers':
        table = layer_table(layers, keep)
        print_layer_table(table, 'kept')
        if args.json:
            _dump_json(_round({'graph': layers['graph'], 'kept': table}),
                       args.json)
        return 0

    if args.cmd == 'analyze':
        analysis = _round(analyze(art, args.scheme, count=args.count,
                                  max_clip_pct=args.max_clip_pct,
                                  bits=args.bits, layers=layers))
        write_report(analysis, args.out)
        if args.json:
            _dump_json(analysis, args.json)
        for item in analysis['conclusions']:
            print('[%s] %s' % (item['level'], item['text']))
        print('report: %s' % os.path.abspath(args.out))
        return 0

    if args.cmd == 'coverage':
        held = load_artifact(args.against)
        keep = _load_keep(art, args.keep) if args.keep else None
        result = coverage(art, held, keep, args.scheme, args.bits)
        print_coverage(result, args.bits)
        if args.json:
            _dump_json(_round(result), args.json)
        return 0

    stats, schemes, models = prepare(art, args.scheme)
    n = art.frames

    if args.cmd == 'select':
        sel = select_patterns(art, stats, schemes, args.count,
                              args.max_clip_pct, layers=layers)
        keep = sel['keep']
        print('clip budget %.3g%%: %d of %d frames dropped; selected %d '
              '(%d hold a tensor extreme)' % (args.max_clip_pct, sel['k_budget'],
                                             n, keep.size, sel['must']))
        for note in sel['notes']:
            print('note: ' + note)
        tables = [subset_table(stats, schemes, models, np.arange(n), n,
                               'all frames', args.bits),
                  subset_table(stats, schemes, models, keep, n, 'selected',
                               args.bits)]
        if keep.size < n:
            tables.append(random_baseline(stats, schemes, models, n,
                                          int(keep.size), args.bits))
        print_tables(tables, args.bits)
        if layers is not None:
            inner = layer_comparison(art, stats, schemes, layers, sel,
                                     args.count, args.max_clip_pct)
            print('%d frames hold an extreme of a tensor inside the graph '
                  'only; %d of them are kept' % (sel['must_layers'],
                                                 sel['kept_layers']))
            print_layer_comparison(inner)
        _, score = frame_order(stats)
        sources = source_report(art, stats, keep, score)
        if sources['available']:
            empty = [r['source'] for r in sources['rows'] if not r['kept']]
            print('per source kept: ' + ', '.join(
                '%s %d/%d' % (r['source'], r['kept'], r['frames'])
                for r in sources['rows']))
            if empty:
                print('note: %d source(s) keep no frame' % len(empty))
        else:
            print('per-source counts unavailable: ' + sources['note'])
        record = {'artifact': art.path,
                  'keep_frames': [int(i) for i in keep],
                  'count': args.count,
                  'max_clip_pct': args.max_clip_pct}
        if layers is not None:
            record['graph'] = layers['graph']
        with open(args.keep_out, 'w', encoding='utf-8') as stream:
            json.dump(record, stream)
        print('selection: %s' % os.path.abspath(args.keep_out))
        if args.output:
            path = apply_selection(art, keep, args.output,
                                   args.format or art.fmt)
            print('wrote %d/%d frames -> %s' % (keep.size, n, path))
        return 0

    keep = _load_keep(art, args.keep)
    print_tables([subset_table(stats, schemes, models, np.arange(n), n,
                               'all frames', args.bits),
                  subset_table(stats, schemes, models, keep, n, 'kept',
                               args.bits)], args.bits)
    path = apply_selection(art, keep, args.output, args.format or art.fmt)
    print('wrote %d/%d frames -> %s' % (keep.size, n, path))
    print('the paired .onnx is unchanged; copy/symlink it beside the new '
          'artifact if the quantiser expects them together')
    return 0


if __name__ == '__main__':
    sys.exit(main())
