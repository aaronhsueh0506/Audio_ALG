#!/usr/bin/env python3
"""Analyse a PTQ calibration artifact and help choose which patterns to keep.

A calibration artifact is what ``inference.py calib`` writes: one array per
graph input whose leading axis is the captured invocation ("frame"), either a
``.npz`` (+ sibling ``.json``) or a per-frame ``.bin`` directory
(+ ``manifest.json``).  A range-based quantizer sets each tensor's scale from
the extremes it sees, so a handful of frames holding an outlier can stretch the
range and cost every other frame resolution.  This tool shows where that
happens and what dropping those frames would buy.

    # interactive HTML report (open it in a browser, no server needed)
    python3 PTQ/calib_analyze.py analyze calib/dfn2 --out calib_report.html

    # let the tool choose N patterns (clip budget + coverage)
    python3 PTQ/calib_analyze.py select calib/dfn2 --count 512 \
        --output calib/dfn2_512

    # write a reduced artifact from the keep list the report exports
    python3 PTQ/calib_analyze.py apply calib/dfn2 --keep keep.json \
        --output calib/dfn2_pruned

Per tensor the report gives min / max / percentiles / tail ratio, how fixed
the outlier position is along the channel and feature axes (a fixed band means
the problem is feature scaling, not pattern choice), each frame's leave-one-out
range shrink, and a simulated 8-bit quantisation error (SQNR) against how many
of the most extreme patterns are dropped.  Patterns are grouped into source
segments when the state resets (all-zero initial state) reveal them.

Selection caveats: this is a statistic over the recorded INPUT tensors, not
over the activations inside the graph; frames are streaming snapshots, so a
pruned set keeps each frame's own recorded state but is no longer one
continuous stream.  Models that need one contiguous stream are refused.
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

from calibration_io import write_calibration_artifact  # noqa: E402

SAMPLE_CAP = 2_000_000       # elements per tensor used for percentiles / SQNR
EPS = 1e-12


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


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def load_artifact(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        report = _read_json(os.path.join(path, 'manifest.json'))
        if 'binary_tensors' not in report:
            raise ValueError('%s has no manifest.json with binary_tensors'
                             % path)
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
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        report = _read_json(os.path.splitext(path)[0] + '.json')
        return Artifact(path, 'npz', arrays, report)
    raise ValueError('expected a .npz file or a BIN directory: %s' % path)


# --------------------------------------------------------------------------
# Per-tensor statistics
# --------------------------------------------------------------------------

def _sample(flat, per_frame, rng):
    """Up to SAMPLE_CAP elements of ``flat [N, per_frame]`` + their frame ids."""
    total = flat.size
    if total <= SAMPLE_CAP:
        values = flat.ravel()
        frame_ids = np.repeat(np.arange(flat.shape[0]), per_frame)
        return values, frame_ids
    index = np.sort(rng.choice(total, SAMPLE_CAP, replace=False))
    return flat.ravel()[index], index // per_frame


def _loo_extreme(per_frame, take_max):
    """Global max (or min) with each frame removed, via the top-2 trick."""
    order = np.argsort(per_frame)
    if take_max:
        best, second = order[-1], order[-2] if len(order) > 1 else order[-1]
    else:
        best, second = order[0], order[1] if len(order) > 1 else order[0]
    out = np.full(per_frame.shape, per_frame[best], dtype=np.float64)
    out[best] = per_frame[second]
    return out


def _axis_concentration(x, axis):
    """How much a single index along ``axis`` dominates the absolute maximum."""
    other = tuple(a for a in range(1, x.ndim) if a != axis % x.ndim)
    peak = np.abs(x).max(axis=(0,) + other).astype(np.float64)
    return {
        'size': int(peak.size),
        'argmax': int(peak.argmax()),
        'ratio': float(peak.max() / (np.median(peak) + EPS)),
        'profile': [float(v) for v in peak],
    }


def tensor_stats(x, rng):
    n = x.shape[0]
    flat = x.reshape(n, -1)
    per_frame = flat.shape[1]
    fmin = flat.min(axis=1).astype(np.float64)
    fmax = flat.max(axis=1).astype(np.float64)
    fabs = np.maximum(np.abs(fmin), np.abs(fmax))
    gmin, gmax = float(fmin.min()), float(fmax.max())
    span = gmax - gmin
    loo_span = _loo_extreme(fmax, True) - _loo_extreme(fmin, False)
    shrink = (span - loo_span) / span if span > 0 else np.zeros(n)

    sample, frame_ids = _sample(flat, per_frame, rng)
    sample64 = sample.astype(np.float64)
    pct = np.percentile(sample64, [0.1, 1, 50, 99, 99.9])
    abs_p999 = float(np.percentile(np.abs(sample64), 99.9))
    mean, std = float(sample64.mean()), float(sample64.std())
    kurt = (float(np.mean((sample64 - mean) ** 4) / (std ** 4 + EPS) - 3.0)
            if std > 0 else 0.0)
    absmax = max(abs(gmin), abs(gmax))

    # What a quantiser toolchain derives from the same data.
    scale_asym = (gmax - gmin) / 255.0
    p_lo, p_hi, p9999 = np.percentile(sample64, [0.1, 99.9, 99.99])
    mse_lo, mse_hi, mse_alpha = _mse_range(sample, gmin, gmax, 'asym')
    # EMA of per-frame extremes (PyTorch MovingAverageMinMax, constant 0.01).
    ema_lo, ema_hi = float(fmin[0]), float(fmax[0])
    for lo_f, hi_f in zip(fmin[1:], fmax[1:]):
        ema_lo += 0.01 * (lo_f - ema_lo)
        ema_hi += 0.01 * (hi_f - ema_hi)
    quant = {
        'scale_sym': absmax / 127.0,
        'scale_asym': scale_asym,
        'zero_point_asym': (int(np.clip(round(-gmin / scale_asym), 0, 255))
                            if scale_asym > 0 else 0),
        'levels_p998': float((p_hi - p_lo) / scale_asym) if scale_asym > 0
        else 0.0,
        'zero_frac': float(np.mean(sample == 0)),
        'p9999': float(p9999),
        'kl_threshold': kl_threshold(np.abs(sample64)),
        'mse_lo': float(mse_lo), 'mse_hi': float(mse_hi),
        'mse_alpha': mse_alpha,
        'ema_min': ema_lo, 'ema_max': ema_hi,
    }

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
            'p001': float(pct[0]), 'p01': float(pct[1]),
            'p50': float(pct[2]), 'p99': float(pct[3]), 'p999': float(pct[4]),
            'abs_p999': abs_p999,
            'tail_ratio': absmax / (abs_p999 + EPS),
            'sampled': bool(sample.size < flat.size),
            **quant,
        },
        'concentration': conc,
        'frame_min': fmin, 'frame_max': fmax, 'frame_absmax': fabs,
        'frame_shrink': shrink,
        '_sample': sample, '_sample_frames': frame_ids,
    }


# --------------------------------------------------------------------------
# Segments (source files) from state resets
# --------------------------------------------------------------------------

def detect_segments(art):
    """Frame indices where a new source file began, plus a reliability flag.

    A fresh source starts from the all-zero initial state.  Tensors whose first
    frame is exactly zero are treated as state; a boundary is a frame where
    every one of them is exactly zero again.  Trusted only if the count equals
    the number of recorded source files.
    """
    n = art.frames
    state = [name for name in art.names
             if not np.any(art.arrays[name][0])]
    if not state:
        return [0], False, 'no tensor starts from an all-zero state'
    zero = np.ones(n, dtype=bool)
    for name in state:
        zero &= ~art.arrays[name].reshape(n, -1).any(axis=1)
    starts = [int(i) for i in np.flatnonzero(zero)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    sources = art.report.get('source_files') or []
    reliable = len(starts) == len(sources)
    note = ('%d state resets match %d source files' % (len(starts), len(sources))
            if reliable else
            '%d state resets but %d source files: segments are a guess'
            % (len(starts), len(sources)))
    return starts, reliable, note


# --------------------------------------------------------------------------
# Simulated 8-bit quantisation
# --------------------------------------------------------------------------

def _qdq(values, lo, hi, scheme):
    values = values.astype(np.float32, copy=False)
    if scheme == 'sym':
        top = max(abs(lo), abs(hi))
        if top <= 0:
            return values.copy()
        scale = np.float32(top / 127.0)
        return np.clip(np.rint(values / scale), -127, 127) * scale
    if hi <= lo:
        return np.full_like(values, lo)
    scale = np.float32((hi - lo) / 255.0)
    return np.clip(np.rint((values - lo) / scale), 0, 255) * scale + lo


def _mse_range(values, lo, hi, scheme):
    """Clip [lo, hi] by the factor that minimises quantisation MSE.

    ``values`` is a (sub)sample; the search shrinks the observed range toward
    zero, the usual MSE-observer behaviour.
    """
    if values.size > 200_000:
        values = values[:: values.size // 200_000 + 1]
    best, best_err = (lo, hi, 1.0), np.inf
    for alpha in np.linspace(0.2, 1.0, 41):
        err = float(np.mean((_qdq(values, lo * alpha, hi * alpha, scheme)
                             - values).astype(np.float64) ** 2))
        if err < best_err:
            best, best_err = (lo * alpha, hi * alpha, float(alpha)), err
    return best


def kl_threshold(absvalues, bins=2048, target=128):
    """Symmetric clip threshold minimising KL(reference || 128-level quantised).

    The entropy (KL) calibration; ``absvalues`` is |x|.
    """
    top = float(absvalues.max())
    if top <= 0:
        return 0.0
    hist, edges = np.histogram(absvalues, bins=bins, range=(0.0, top))
    hist = hist.astype(np.float64)
    best_div, best_i = np.inf, bins
    for i in range(target, bins + 1, 8):
        ref = hist[:i].copy()
        ref[-1] += hist[i:].sum()
        if ref.sum() == 0:
            continue
        groups = np.array_split(hist[:i], target)
        quant = np.concatenate([
            np.where(g > 0, g.sum() / max((g > 0).sum(), 1), 0.0)
            for g in groups])
        mask = (ref > 0) & (quant > 0)
        p, q = ref[mask] / ref.sum(), quant[mask] / quant.sum()
        div = float(np.sum(p * np.log(p / q)))
        if div < best_div:
            best_div, best_i = div, i
    return float(edges[best_i])


def quant_eval(stats, kept, mode, scheme):
    """SQNR (dB) of an 8-bit grid fixed by the kept frames.

    ``full`` scores every recorded value (dropped frames get clipped, which is
    the honest cost); ``kept`` scores only what the grid was built from.
    """
    sample, frame_ids = stats['_sample'], stats['_sample_frames']
    keep_elem = kept[frame_ids]
    if not keep_elem.any():
        return None
    if mode in ('minmax', 'mse'):
        lo = float(stats['frame_min'][kept].min())
        hi = float(stats['frame_max'][kept].max())
        if mode == 'mse':
            lo, hi, _ = _mse_range(sample[keep_elem], lo, hi, scheme)
    else:
        lo, hi = (float(v) for v in
                  np.percentile(sample[keep_elem].astype(np.float64),
                                [0.1, 99.9]))
    deq = _qdq(sample, lo, hi, scheme)
    err = (deq - sample).astype(np.float64) ** 2
    sig = sample.astype(np.float64) ** 2

    def db(mask):
        s, e = sig[mask].sum(), err[mask].sum()
        return float(10 * np.log10((s + EPS) / (e + EPS)))

    clipped = float(((sample < lo) | (sample > hi))[~keep_elem].mean()
                    if (~keep_elem).any() else 0.0)
    return {'sqnr_full': db(np.ones_like(keep_elem)), 'sqnr_kept': db(keep_elem),
            'lo': lo, 'hi': hi, 'dropped_clip': clipped}


def _drop_grid(units):
    cap = max(1, units // 2)
    grid = {0}
    value = 1.0
    while value <= cap:
        grid.add(int(round(value)))
        value *= 1.35
    grid.add(cap)
    return sorted(g for g in grid if g <= cap)


def unit_order(stats_by_name, units):
    """Units (frame-index arrays) ordered most extreme first.

    Score = worst-case ratio of the unit's abs-max to the median unit abs-max
    across tensors, so the ranking is comparable between tensors of different
    scale.
    """
    score = np.zeros(len(units))
    for stats in stats_by_name.values():
        peak = np.array([stats['frame_absmax'][u].max() for u in units])
        score = np.maximum(score, peak / (np.median(peak) + EPS))
    return np.argsort(-score, kind='stable'), score


def build_curves(stats_by_name, units, order, n_frames, scheme):
    grid = _drop_grid(len(units))
    curves = {name: {'minmax': [], 'pct': [], 'mse': []}
              for name in stats_by_name}
    for k in grid:
        kept = np.ones(n_frames, dtype=bool)
        for u in order[:k]:
            kept[units[u]] = False
        for name, stats in stats_by_name.items():
            for mode in ('minmax', 'pct', 'mse'):
                curves[name][mode].append(quant_eval(stats, kept, mode, scheme))
    return {'k': grid, 'tensors': curves}



# --------------------------------------------------------------------------
# Selection: pick N patterns
# --------------------------------------------------------------------------

class RangeModel:
    """Analytic 8-bit error of a clip range, evaluated against every value.

    Built from the sorted value sample and its running sums, so scoring a
    range is a binary search instead of a pass over the data.  Error =
    rounding (step^2/12 inside the range) + clipping (squared distance of every
    value outside it).  Approximates ``quant_eval`` to a fraction of a dB.
    """

    def __init__(self, stats, scheme='asym'):
        self.scheme = scheme
        self.s = np.sort(stats['_sample'].astype(np.float64))
        self.c1 = np.concatenate([[0.0], np.cumsum(self.s)])
        self.c2 = np.concatenate([[0.0], np.cumsum(self.s * self.s)])
        self.n = self.s.size

    def evaluate(self, lo, hi):
        """Arrays of kept-set extremes -> clip fraction and two SQNRs (dB).

        ``sqnr_all`` counts clipped values at their full error; ``sqnr_typical``
        looks only at the values inside the range (the resolution the bulk of
        the data gets).
        """
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        if self.scheme == 'sym':
            top = np.maximum(np.abs(lo), np.abs(hi))
            lo, hi = -top, top
            step = top / 127.0
        else:
            step = (hi - lo) / 255.0
        s, c1, c2, n = self.s, self.c1, self.c2, self.n
        i = np.searchsorted(s, lo, side='left')
        j = np.searchsorted(s, hi, side='right')
        below = i * lo * lo - 2 * lo * c1[i] + c2[i]
        above = (c2[n] - c2[j]) - 2 * hi * (c1[n] - c1[j]) + (n - j) * hi * hi
        inner = (j - i) * step * step / 12.0
        sig_in = c2[j] - c2[i]
        db = lambda a, b: 10 * np.log10((a + EPS) / (b + EPS))
        return {
            'clip': (i + n - j) / n,
            'sqnr_all': db(c2[n], inner + np.maximum(below, 0)
                           + np.maximum(above, 0)),
            'sqnr_typical': db(sig_in, inner),
        }


def frame_features(art, names):
    """Per-frame mean, std and abs-max of every tensor, z-scored later."""
    cols = []
    for name in names:
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
    return np.stack(cols, axis=1)


def _kmeans_pick(features, k, rng, iters=12):
    """Indices of k rows that stand for the rest (nearest to k-means centres)."""
    n = features.shape[0]
    if k >= n:
        return np.arange(n)
    z = (features - features.mean(0)) / (features.std(0) + EPS)

    def nearest(centres):
        dist = np.empty(n)
        arg = np.empty(n, dtype=np.int64)
        for a in range(0, n, 2048):
            d = ((z[a:a + 2048, None, :] - centres[None, :, :]) ** 2).sum(-1)
            arg[a:a + 2048] = d.argmin(1)
            dist[a:a + 2048] = d.min(1)
        return arg, dist

    # Seeding by uniform sampling keeps the picks proportional to density, so a
    # large regime is thinned rather than kept whole (k-means++ would favour
    # far-away points).
    centres = z[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        arg, _ = nearest(centres)
        for c in range(k):
            members = z[arg == c]
            if members.size:
                centres[c] = members.mean(0)
    chosen = []
    taken = np.zeros(n, dtype=bool)
    for c in range(k):
        d = ((z - centres[c]) ** 2).sum(1)
        d[taken] = np.inf
        index = int(d.argmin())
        taken[index] = True
        chosen.append(index)
    return np.array(chosen)


def select_patterns(art, count=None, max_clip_pct=0.1, scheme='asym', seed=0):
    """Choose which frames to calibrate with.

    1. Rank frames most extreme first (same score the report uses).
    2. Drop extremes while every tensor would still have at most
       ``max_clip_pct`` percent of ALL recorded values outside the kept range.
       This is the explicit trade: those few values get clipped in exchange for
       finer resolution for the rest.
    3. If more than ``count`` frames remain, thin them with k-means so the
       survivors still cover the distribution, always keeping the frames that
       define each tensor's extremes (otherwise the range would shrink past
       the budget).  If fewer remain, the least extreme dropped frames return.
    """
    n = art.frames
    if count is not None and not 0 < count <= n:
        raise ValueError('--count must be within 1..%d' % n)
    rng = np.random.default_rng(seed)
    stats = {name: tensor_stats(art.arrays[name], rng) for name in art.names}
    models = {name: RangeModel(stats[name], scheme) for name in art.names}
    units = [np.array([i]) for i in range(n)]
    order, _ = unit_order(stats, units)

    ok = np.ones(n, dtype=bool)                 # budget holds after k drops
    for name, st in stats.items():
        fmin, fmax = st['frame_min'][order], st['frame_max'][order]
        lo = np.minimum.accumulate(fmin[::-1])[::-1]
        hi = np.maximum.accumulate(fmax[::-1])[::-1]
        ok &= models[name].evaluate(lo, hi)['clip'] <= max_clip_pct / 100.0
    ok[0] = True
    k_budget = int(np.flatnonzero(np.logical_and.accumulate(ok))[-1])
    pool = np.sort(order[k_budget:])
    dropped_by_extremeness = order[:k_budget][::-1]       # least extreme first

    notes = []
    if count is None or count == pool.size:
        keep = pool
    elif count > pool.size:
        keep = np.sort(np.concatenate(
            [pool, dropped_by_extremeness[:count - pool.size]]))
        notes.append('clip budget alone leaves %d frames, %d short of --count; '
                     'the least extreme dropped frames were added back'
                     % (pool.size, count - pool.size))
    else:
        must = set()
        for name, st in stats.items():
            must.add(int(pool[np.argmin(st['frame_min'][pool])]))
            must.add(int(pool[np.argmax(st['frame_max'][pool])]))
        must = np.array(sorted(must))
        if must.size > count:
            keep = np.sort(must[:count])
            notes.append('%d range-defining frames exceed --count %d; the '
                         'range will be narrower than the budget allows'
                         % (must.size, count))
        else:
            rest = np.setdiff1d(pool, must)
            features = frame_features(art, art.names)[rest]
            picked = rest[_kmeans_pick(features, count - must.size, rng)]
            keep = np.sort(np.concatenate([must, picked]))
    return keep, k_budget, stats, models, notes


def subset_table(art, stats, models, keep, label):
    """One row per tensor for ``keep``: range, clip %, SQNR, vs all frames."""
    kept = np.zeros(art.frames, dtype=bool)
    kept[keep] = True
    rows = []
    for name in art.names:
        st = stats[name]
        lo = st['frame_min'][kept].min()
        hi = st['frame_max'][kept].max()
        m = models[name].evaluate(lo, hi)
        rows.append({'tensor': name, 'lo': float(lo), 'hi': float(hi),
                     'clip_pct': 100 * float(m['clip']),
                     'sqnr_all': float(m['sqnr_all']),
                     'sqnr_typical': float(m['sqnr_typical'])})
    return {'label': label, 'frames': int(kept.sum()), 'rows': rows}


def random_baseline(art, stats, models, count, draws=30, seed=1):
    """Mean of ``draws`` uniformly random subsets of ``count`` frames."""
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(draws):
        keep = rng.choice(art.frames, count, replace=False)
        table = subset_table(art, stats, models, keep, 'random')
        if acc is None:
            acc = table
            continue
        for a, b in zip(acc['rows'], table['rows']):
            for key in ('lo', 'hi', 'clip_pct', 'sqnr_all', 'sqnr_typical'):
                a[key] += b[key]
    for row in acc['rows']:
        for key in ('lo', 'hi', 'clip_pct', 'sqnr_all', 'sqnr_typical'):
            row[key] /= draws
    acc['label'] = 'random (mean of %d)' % draws
    return acc


def print_tables(tables):
    print('%-22s %-16s %-20s %9s %12s %10s' % (
        'subset', 'tensor', 'range', 'clipped%', 'SQNR typ(dB)',
        'SQNR all(dB)'))
    for table in tables:
        for index, row in enumerate(table['rows']):
            head = ('%s (%d)' % (table['label'], table['frames'])
                    if index == 0 else '')
            print('%-22s %-16s [%8.3g,%8.3g] %9.3f %12.2f %10.2f' % (
                head, row['tensor'], row['lo'], row['hi'], row['clip_pct'],
                row['sqnr_typical'], row['sqnr_all']))


def segment_counts(art, keep):
    starts, reliable, _ = detect_segments(art)
    if len(starts) < 2:
        return None
    bounds = starts + [art.frames]
    sources = art.report.get('source_files') or []
    out = []
    for i in range(len(starts)):
        kept = int(((keep >= bounds[i]) & (keep < bounds[i + 1])).sum())
        out.append((sources[i] if reliable else 'segment %d' % i,
                    kept, bounds[i + 1] - bounds[i]))
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _round(value):
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_round(float(v)) for v in value]
    if isinstance(value, (float, np.floating)):
        return float('%.6g' % value)
    return value


def analyze(art, scheme='asym', seed=0, count=None, max_clip_pct=0.1):
    rng = np.random.default_rng(seed)
    stats = {name: tensor_stats(art.arrays[name], rng) for name in art.names}
    starts, reliable, note = detect_segments(art)
    bounds = starts + [art.frames]
    segments = [np.arange(bounds[i], bounds[i + 1])
                for i in range(len(starts))]
    use_segments = len(segments) > 1
    units = segments if use_segments else [np.array([i])
                                           for i in range(art.frames)]
    order, score = unit_order(stats, units)
    curves = build_curves(stats, units, order, art.frames, scheme)

    sources = art.report.get('source_files') or []
    seg_rows = []
    for i, seg in enumerate(segments):
        seg_rows.append({
            'id': i, 'start': int(seg[0]), 'frames': int(seg.size),
            'source': sources[i] if reliable and i < len(sources) else None,
        })

    recommend = None
    if not art.contiguous:
        keep, k_budget, sel_stats, models, notes = select_patterns(
            art, count, max_clip_pct, scheme, seed)
        recommend = {
            'keep_frames': [int(i) for i in keep], 'count': count,
            'max_clip_pct': max_clip_pct, 'notes': notes,
            'all': subset_table(art, sel_stats, models,
                                np.arange(art.frames), 'all frames'),
            'selected': subset_table(art, sel_stats, models, keep, 'selected'),
        }
    return {
        'recommend': recommend,
        'meta': {
            'path': art.path, 'format': art.fmt, 'frames': art.frames,
            'model_family': art.report.get('model_family'),
            'scheme': scheme, 'contiguous': art.contiguous,
            'skipped': art.skipped, 'segments_note': note,
            'segments_reliable': reliable,
            'unit': 'segment' if use_segments else 'frame',
        },
        'segments': seg_rows if use_segments else [],
        'unit_order': [int(u) for u in order],
        'unit_score': score,
        'curves': curves,
        'tensors': {
            name: {k: v for k, v in s.items() if not k.startswith('_')}
            for name, s in stats.items()
        },
    }


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


def keep_indices(art, spec, analysis_segments=None):
    n = art.frames
    if 'keep_frames' in spec:
        keep = np.unique(np.asarray(spec['keep_frames'], dtype=np.int64))
    elif 'drop_frames' in spec:
        keep = np.setdiff1d(np.arange(n), np.asarray(spec['drop_frames'],
                                                     dtype=np.int64))
    else:
        raise ValueError('keep file needs keep_frames or drop_frames')
    if keep.size == 0 or keep.min() < 0 or keep.max() >= n:
        raise ValueError('keep list is empty or outside 0..%d' % (n - 1))
    return keep


def apply_selection(art, keep, output, fmt):
    if art.contiguous:
        raise SystemExit(
            '%s needs one uninterrupted stream; its cumulative state ranges '
            'are meaningless on a pruned set. Regenerate with a different '
            'source instead.' % art.report.get('model_family'))
    arrays = {name: value[keep] for name, value in art.arrays.items()}
    report = dict(art.report)
    starts, reliable, _ = detect_segments(art)
    sources = report.get('source_files') or []
    if reliable and len(starts) > 1:
        bounds = starts + [art.frames]
        alive = [i for i in range(len(starts))
                 if ((keep >= bounds[i]) & (keep < bounds[i + 1])).any()]
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


def print_before_after(art, keep):
    kept = np.zeros(art.frames, dtype=bool)
    kept[keep] = True
    rng = np.random.default_rng(0)
    print('%-18s %-22s %-22s %7s %9s %8s %8s' %
          ('tensor', 'range before', 'range after', 'shrink', 'sqnr(dB)',
           'KL thr', 'KL thr'))
    print('%-18s %-22s %-22s %7s %9s %8s %8s' %
          ('', '', '', '', 'minmax', 'before', 'after'))
    for name in art.names:
        s = tensor_stats(art.arrays[name], rng)
        lo0, hi0 = s['frame_min'].min(), s['frame_max'].max()
        lo1, hi1 = s['frame_min'][kept].min(), s['frame_max'][kept].max()
        shrink = 1 - (hi1 - lo1) / (hi0 - lo0 + EPS)
        q = quant_eval(s, kept, 'minmax', 'asym')
        kept_abs = np.abs(art.arrays[name][kept]).ravel().astype(np.float64)
        if kept_abs.size > SAMPLE_CAP:
            kept_abs = kept_abs[:: kept_abs.size // SAMPLE_CAP + 1]
        print('%-18s [%8.3g,%8.3g]   [%8.3g,%8.3g]   %6.1f%% %9.2f %8.3g %8.3g' %
              (name, lo0, hi0, lo1, hi1, 100 * shrink, q['sqnr_full'],
               s['global']['kl_threshold'], kl_threshold(kept_abs)))


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------

def write_report(analysis, out_path):
    template_path = os.path.join(_HERE, 'calib_analyze_report.html')
    with open(template_path, encoding='utf-8') as stream:
        template = stream.read()
    data = json.dumps(_round(analysis), separators=(',', ':'))
    data = data.replace('</', '<\\/')
    with open(out_path, 'w', encoding='utf-8') as stream:
        stream.write(template.replace('/*__DATA__*/null', data))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_an = sub.add_parser('analyze', help='write an interactive HTML report')
    p_an.add_argument('artifact')
    p_an.add_argument('--out', default='calib_report.html')
    p_an.add_argument('--json', default=None, help='also dump the raw analysis')
    p_an.add_argument('--scheme', choices=('asym', 'sym'), default='asym',
                      help='8-bit grid used by the SQNR simulation')
    p_an.add_argument('--count', type=int, default=None,
                      help='also embed a recommended selection of this size')
    p_an.add_argument('--max-clip-pct', type=float, default=0.1,
                      help='clip budget used for the recommendation')
    p_se = sub.add_parser('select', help='pick which patterns to keep')
    p_se.add_argument('artifact')
    p_se.add_argument('--count', type=int, default=None,
                      help='number of frames to keep (default: as many as the '
                           'clip budget leaves)')
    p_se.add_argument('--max-clip-pct', type=float, default=0.1,
                      help='at most this percent of all recorded values may '
                           'fall outside any tensor\'s kept range')
    p_se.add_argument('--scheme', choices=('asym', 'sym'), default='asym')
    p_se.add_argument('--keep-out', default='keep.json',
                      help='where to write the selection (for apply)')
    p_se.add_argument('--output', default=None,
                      help='also write the reduced artifact here')
    p_se.add_argument('--format', choices=('bin', 'npz'), default=None)
    p_ap = sub.add_parser('apply', help='write a reduced calibration artifact')
    p_ap.add_argument('artifact')
    p_ap.add_argument('--keep', required=True,
                      help='JSON with keep_frames or drop_frames')
    p_ap.add_argument('--output', required=True)
    p_ap.add_argument('--format', choices=('bin', 'npz'), default=None,
                      help='defaults to the input artifact format')
    args = parser.parse_args(argv)

    art = load_artifact(args.artifact)
    print('%s: %d frames, %d quantisable tensors%s' % (
        art.path, art.frames, len(art.names),
        (', skipped: ' + ', '.join(art.skipped)) if art.skipped else ''))

    if args.cmd == 'analyze':
        analysis = analyze(art, args.scheme, count=args.count,
                           max_clip_pct=args.max_clip_pct)
        write_report(analysis, args.out)
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as stream:
                json.dump(_round(analysis), stream)
        print('segments: %s' % analysis['meta']['segments_note'])
        print('report: %s' % os.path.abspath(args.out))
        return 0

    if args.cmd == 'select':
        if art.contiguous:
            raise SystemExit('%s needs one uninterrupted stream; patterns '
                             'cannot be selected from it.'
                             % art.report.get('model_family'))
        keep, k_budget, stats, models, notes = select_patterns(
            art, args.count, args.max_clip_pct, args.scheme)
        print('clip budget %.3g%%: up to %d of %d frames can be dropped '
              '-> %d kept; selected %d' % (args.max_clip_pct, k_budget,
                                          art.frames, art.frames - k_budget,
                                          keep.size))
        for note in notes:
            print('note: ' + note)
        tables = [subset_table(art, stats, models, np.arange(art.frames),
                               'all frames'),
                  subset_table(art, stats, models, keep, 'selected')]
        if keep.size < art.frames:
            tables.append(random_baseline(art, stats, models, keep.size))
        print_tables(tables)
        counts = segment_counts(art, keep)
        if counts:
            print('per source: ' + ', '.join(
                '%s %d/%d' % c for c in counts))
        with open(args.keep_out, 'w', encoding='utf-8') as stream:
            json.dump({'artifact': art.path,
                       'keep_frames': [int(i) for i in keep],
                       'max_clip_pct': args.max_clip_pct}, stream)
        print('selection: %s' % os.path.abspath(args.keep_out))
        if args.output:
            path = apply_selection(art, keep, args.output,
                                   args.format or art.fmt)
            print('wrote %d/%d frames -> %s' % (keep.size, art.frames, path))
        return 0

    with open(args.keep, encoding='utf-8') as stream:
        keep = keep_indices(art, json.load(stream))
    print_before_after(art, keep)
    path = apply_selection(art, keep, args.output, args.format or art.fmt)
    print('wrote %d/%d frames -> %s' % (keep.size, art.frames, path))
    print('the paired .onnx is unchanged; copy/symlink it beside the new '
          'artifact if the quantiser expects them together')
    return 0


if __name__ == '__main__':
    sys.exit(main())
