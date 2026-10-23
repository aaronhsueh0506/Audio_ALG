#!/usr/bin/env python3
"""pyc_parity.py -- comparison half of pyc_parity_pipelines.sh (run it through the shell script).

The shell script builds the C code twice (Layer 1: std-math, Layer 2: shipped) and the two C drivers
(pyc_drv.c). This program runs the drivers, runs the Python references of the pipelines, compares, prints a
table and writes a JSON file. Every tolerance below is frozen: set from the measured max error over every
scenario in both layers, with a margin (see the comment above TOL1/TOL2). A tolerance prints next to the
number it judged and changes only by re-measuring, never silently.

Statuses: PASS / FAIL, EXPECTED_RED(S-xx) (a known gap exceeds its tolerance: allowed), UNEXPECTED_GREEN(S-xx)
(a known gap is within tolerance: FAIL, remove the tag with the fix), SKIP, NOT_RUN (could not be exercised:
the run is incomplete), INFO (reported, not judged).
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import types

import numpy as np

# --------------------------------------------------------------------------------------------------
# Scenario tables
# --------------------------------------------------------------------------------------------------
GRIDS = [('16k256', 16000, 256), ('16k512', 16000, 512), ('48k1024', 48000, 1024)]
GRID_LABEL = {'16k256': '16k/256', '16k512': '16k/512', '48k1024': '48k/1024'}
PRESETS = ['balanced', 'mild', 'aggressive']
# b.res4 only (not the shared PRESETS list, so PRESETS-driven scenarios such as l2.res4.*.replay
# are unaffected). aec=no_floor on the C side is a direct construction-time preset
# (se_derive()'s preset_valid() whitelists NO_FLOOR; see pyc_drv.c), matching the Python side's
# FourChannelAecConfig(aec_preset=NO_FLOOR).
PRESETS_B = PRESETS + ['no_floor']

# Known gaps: a row listed here may exceed its tolerance without failing; if it does not, the script
# fails so the tag is removed together with the fix. Empty while no gap is known.
EXPECTED_RED = {}

# Frozen tolerances. Each value is the max error measured over every scenario judged against it (a
# `pyc_parity_pipelines.sh --layers 1,2` run, EXPECTED_RED empty), rounded up to a clean figure with roughly
# a 1.8-2.7x margin in the linear constants (~1.9x in amplitude for the dB constant): enough that another
# toolchain's libm or NEON lowering should not flip a PASS to FAIL, tight enough that a regression that
# doubles the error is still caught. The measured max is in a comment next to each constant.
TOL1 = {
    'gain_abs': 2e-5,        # Layer 1: a gain in [0, 1], std-math C against Python; measured max 8.3e-6 (d.mono.16k256.nr_glue)
    'out_abs': 2e-7,         # Layer 1: float output samples (full scale 1.0); measured max 7.5e-8 (d.mono.48k1024.min_apply_ola)
    'cng_abs': 0.0,          # Layer 1: C aec_cng_fill against its Python twin: bit exact (measured: 18 of 18 cases bit exact)
}
TOL2 = {                     # Layer 2: shipped (fast-math, NEON) against std-math C
    'replay_gain_abs': 5e-2,     # open loop: same frames through the stage in both builds, max |dG|; measured max 0.0280 (l2.res4.48k1024.mild.replay)
    'replay_out_abs': 1e-3,      # open loop: float output samples, max |d|; measured max 4.25e-4 (l2.mono.16k256.replay.out)
    'chain_gain_p99_9': 5e-2,    # closed loop (adaptive filters diverge): p99.9 |dG| up to the first divergence hop; measured max 0.0286 (l2.chain4.48k1024.gres)
    'chain_out_p99_9': 4e-4,     # closed loop: p99.9 |d out| up to the first divergence hop; measured max 1.69e-4 (l2.chain4.48k1024.out)
    'chain_rms_db': -45.0,       # closed loop: rms(d)/rms(ref) in dB up to the first divergence hop; measured worst -50.6 dB (l2.chain4.48k1024.gres)
}

SOURCES = [   # hashed into the JSON so a result can be tied to the code it judged
    'Audio_ALG/pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c',
    'Audio_ALG/pipelines/4ch_aec_bf_nr_res/pipeline.py',
    'Audio_ALG/pipelines/4ch_aec_bf_nr_res/tests/conv_chain_4ch.h',
    'Audio_ALG/pipelines/aec_nr_pipeline.py',
    'Audio_ALG/pipelines/cng_aec3.py',
    'Audio_ALG/pipelines/common/se_anr_stage.c',
    'Audio_ALG/pipelines/mono_aec_nr_res/audio_pipeline.c',
    'Audio_ALG/lib/aec/c_impl/src/aec_res.c',
]

BUILD_NAME = {1: 'std-math SIMD=0', 2: 'shipped SIMD=1'}


def row_ids():
    ids = []
    for g, _, _ in GRIDS:
        ids += ['a.nr4.%s.%s' % (g, k) for k in ('glue_cflag', 'glue_fold', 'verdict_flag', 'gnr_e2e', 'legacy_amin')]
    for g, _, _ in GRIDS:
        ids += ['b.res4.%s.%s' % (g, p) for p in PRESETS_B]
    for g, _, _ in GRIDS:
        ids += ['c.realign4.%s' % g]
    for g, _, _ in GRIDS:
        ids += ['d.mono.%s.%s' % (g, k) for k in ('nr_glue', 'min_apply_ola', 'cng_glue', 'legacy_amin')]
    ids += ['d.mono.cng_twin', 'd.mono.cng_lookalike', 'd.mono.8k']
    for g, _, _ in GRIDS:
        ids += ['l2.nr4.%s.replay.%s' % (g, k) for k in ('gnr', 'out')]
        ids += ['l2.res4.%s.%s.replay' % (g, p) for p in PRESETS]
        ids += ['l2.mono.%s.replay.%s' % (g, k) for k in ('gnr', 'out')]
        ids += ['l2.chain4.%s.%s' % (g, k) for k in ('gnr', 'gres', 'out')]
        ids += ['l2.mono.%s.chain.%s' % (g, k) for k in ('gnr', 'out')]
        ids += ['l2.realign4.%s' % g]
    return ids


# --------------------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------------------
def _f(x):
    return None if x is None else float(x)


def stats(c, p, ref=None):
    """Differences of two arrays (complex allowed). rms_db = 20 log10(rms(c-p)/rms(ref)); null when identical."""
    c = np.asarray(c)
    p = np.asarray(p)
    n = min(c.shape[0], p.shape[0])
    c = c[:n]
    p = p[:n]
    d = np.abs(c.astype(np.complex128 if np.iscomplexobj(c) or np.iscomplexobj(p) else np.float64)
               - p.astype(np.complex128 if np.iscomplexobj(c) or np.iscomplexobj(p) else np.float64))
    r = np.asarray(c if ref is None else ref)
    rms_d = float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0
    rms_r = float(np.sqrt(np.mean(np.abs(r.astype(np.complex128)) ** 2))) if r.size else 0.0
    ident = bool(np.array_equal(c, p))
    if rms_d == 0.0:
        db = None
    else:
        db = 20.0 * np.log10(rms_d / max(rms_r, 1e-300))
    return {
        'max_abs': float(d.max()) if d.size else 0.0,
        'p99_9': float(np.percentile(d, 99.9)) if d.size else 0.0,
        'bit_identical': ident,
        'rms_db': _f(db),
        'n': int(d.size),
        'n_diff': int(np.count_nonzero(d)),
    }


def first_div_hop(a, b, thr=1e-2):
    """First hop where the L2 norm of (a - b) exceeds thr * L2 norm of b."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(a.shape[0], b.shape[0])
    num = np.sqrt((np.abs(a[:n] - b[:n]) ** 2).reshape(n, -1).sum(1))
    den = np.maximum(np.sqrt((np.abs(b[:n]) ** 2).reshape(n, -1).sum(1)), 1e-20)
    big = np.flatnonzero(num / den > thr)
    return int(big[0]) if big.size else None


# --------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------
def finalize(row):
    exp = row.get('expected_red')
    if row.get('skip'):
        st = 'SKIP'
    elif row.get('not_run'):
        st = 'NOT_RUN'
    elif row.get('gate') is None:
        st = 'INFO'
    elif exp:
        st = ('UNEXPECTED_GREEN(%s)' if row['gate'] else 'EXPECTED_RED(%s)') % exp
    else:
        st = 'PASS' if row['gate'] else 'FAIL'
    row['status'] = st
    return row


def auto_headline(m):
    """One-line digest for rows that have no array statistics."""
    if 'mismatch_hops' in m:
        return 'mismatch %d of %d hops (Python verdict true %d, C flag true %d)' % (
            m['mismatch_hops'], m['hops'], m['python_verdict_true_hops'], m['c_flag_true_hops'])
    if 'legacy0_vs_1_out_max_abs' in m:
        return 'legacy 0 vs 1: out max %.3g, G_nr max %.3g (identical bytes: %s); control enable_nr 0 vs 1: %.3g' % (
            m['legacy0_vs_1_out_max_abs'], m['legacy0_vs_1_gnr_max_abs'], m['legacy0_vs_1_out_bit_identical'],
            m['positive_control_enable_nr0_vs_1_out_max_abs'])
    if 'legacy0_vs_1_gnr_max_abs' in m:
        return 'legacy 0 vs 1: G_nr max %.3g' % m['legacy0_vs_1_gnr_max_abs']
    if 'python_raised' in m:
        return 'Python raised=%s at hop %s; C rc=%s, C mixed warm/soft hops %s' % (
            m['python_raised'], m['python_raise_hop'], m['c_fail_rc'], m['c_mixed_warm_soft_hops'])
    if 'cases' in m:
        return 'bit exact %d of %d cases' % (m['bit_exact_cases'], m['cases'])
    if 'rms_ratio_python_over_c' in m:
        return 'noise rms Python/C = %.4f' % m['rms_ratio_python_over_c']
    if 'c_accepts_8k' in m:
        return 'Python 8k: unsupported; C 8k: %s' % m['c_probe']
    if 'assertion' in m:
        return m['assertion']
    return None


def make_row(rid, layer, title, producer, gate=None, metrics=None, tol=None, note='', grid=None,
             skip=False, not_run=False, ref=None, htext=None):
    exp = None
    if grid is not None:
        key = rid.replace('.' + grid, '.%s') if ('.' + grid) in rid else rid
        exp = EXPECTED_RED.get(key) or EXPECTED_RED.get(rid)
    else:
        exp = EXPECTED_RED.get(rid)
    row = {
        'id': rid, 'layer': layer, 'build': BUILD_NAME.get(layer, 'n/a'), 'grid': GRID_LABEL.get(grid, grid),
        'title': title, 'reference': ref, 'producer': producer, 'tolerance': tol,
        'metrics': metrics or {}, 'gate': None if gate is None else bool(gate), 'expected_red': exp,
        'note': note, 'skip': skip, 'not_run': not_run, 'headline': htext if htext is not None else auto_headline(metrics or {}),
    }
    return finalize(row)


# --------------------------------------------------------------------------------------------------
# Context: tree, work dir, C drivers, Python references
# --------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.tree = os.path.abspath(args.tree)
        self.alg = os.path.join(self.tree, 'Audio_ALG')
        self.work = os.path.abspath(args.work)
        self.layers = [int(x) for x in args.layers.split(',')]
        self.info = dict(l.strip().split('=', 1) for l in open(args.build_info) if '=' in l)
        self.cache = {}
        self.runlog = []
        os.makedirs(os.path.join(self.work, 'run'), exist_ok=True)
        for p in (self.alg, os.path.join(self.alg, 'lib', 'aec', 'python'), os.path.join(self.alg, 'lib', 'nr', 'python')):
            sys.path.insert(0, p)
        self._mods = {}

    # ---- modules of the tree under test
    def mod(self, name):
        if name not in self._mods:
            self._mods[name] = importlib.import_module(name)
        return self._mods[name]

    @property
    def P(self):
        return self.mod('pipelines.aec_nr_pipeline')

    @property
    def M4(self):
        return self.mod('pipelines.4ch_aec_bf_nr_res.pipeline')

    # ---- C drivers
    def drv(self, layer, kind):
        return self.info['L%d.drv_%s' % (layer, kind)]

    def run(self, layer, kind, mode, name, **kv):
        """Runs a driver; the output file is cached by (layer, name). Returns the output path."""
        out = os.path.join(self.work, 'run', 'L%d' % layer, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        key = ('run', layer, name)
        if key in self.cache:
            return self.cache[key]
        cmd = [self.drv(layer, kind), mode] + ['%s=%s' % (k, v) for k, v in kv.items()] + ['out=' + out]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.work)
        if r.returncode != 0:
            raise RuntimeError('driver failed (%s): %s\n%s' % (' '.join(os.path.basename(c) if os.sep in c else c for c in cmd),
                                                              r.returncode, r.stderr.decode(errors='replace')))
        self.cache[key] = out
        return out

    def inp(self, name):
        return os.path.join(self.work, 'inputs', name)

    def tree_sha(self, rel):
        h = hashlib.sha256()
        with open(os.path.join(self.tree, rel), 'rb') as f:
            h.update(f.read())
        return h.hexdigest()


# --------------------------------------------------------------------------------------------------
# Seam files
# --------------------------------------------------------------------------------------------------
class Seam:
    """The dump format documented in pyc_drv.c."""

    def __init__(self, path):
        h = np.fromfile(path, np.int32, count=10)
        if h[0] != 0x53435950 or h[1] != 1:
            raise RuntimeError('bad seam header in %s' % path)
        self.kind, self.sr, self.fft, self.nf, self.hop, self.nh, self.fail_hop, self.fail_rc = [int(v) for v in h[2:]]
        nf, hop, nh = self.nf, self.hop, self.nh
        rec = 8 * nf + 2 * hop + 16
        d = np.fromfile(path, np.float32, offset=40)[:nh * rec].reshape(nh, rec)
        o = 0

        def take(k):
            nonlocal o
            v = d[:, o:o + k]
            o += k
            return v

        def cplx(v):
            v = np.ascontiguousarray(v).reshape(nh, nf, 2)
            return (v[..., 0] + 1j * v[..., 1]).astype(np.complex64)

        self.E = cplx(take(2 * nf))
        self.Y = cplx(take(2 * nf))
        self.R2 = np.ascontiguousarray(take(nf))
        self.CN = np.ascontiguousarray(take(nf))
        self.Gres = np.ascontiguousarray(take(nf))
        self.Gnr = np.ascontiguousarray(take(nf))
        self.far = np.ascontiguousarray(take(hop))
        self.out = np.ascontiguousarray(take(hop)).reshape(-1)
        self.sc = np.ascontiguousarray(take(16))

    @property
    def usable(self):
        return self.sc[:, 5] != 0

    @property
    def max_dt(self):
        return self.sc[:, 1]


class Replay:
    def __init__(self, path):
        h = np.fromfile(path, np.int32, count=8)
        _, _, self.kind, self.sr, self.fft, self.nf, self.hop, self.nh = [int(v) for v in h]
        rec = 2 * self.nf + self.hop
        d = np.fromfile(path, np.float32, offset=32)[:self.nh * rec].reshape(self.nh, rec)
        self.Gres = np.ascontiguousarray(d[:, :self.nf])
        self.Gnr = np.ascontiguousarray(d[:, self.nf:2 * self.nf])
        self.out = np.ascontiguousarray(d[:, 2 * self.nf:]).reshape(-1)


def grid_inputs(sr):
    t = '16k' if sr == 16000 else '48k'
    return dict(mic4='mic4_%s.f32' % t, mic1='mic1_%s.f32' % t, ref='ref_%s.f32' % t)


# --------------------------------------------------------------------------------------------------
# C runs (cached)
# --------------------------------------------------------------------------------------------------
def c_chain(cx, layer, g, sr, fft, tag='base', **opt):
    ins = grid_inputs(sr)
    name = 'chain_%s_%s.bin' % (g, tag)
    kv = dict(mic=cx.inp(ins['mic4']), ref=cx.inp(ins['ref']), sr=sr, fft=fft)
    kv.update(opt)
    key = ('seam', layer, name)
    if key not in cx.cache:
        cx.cache[key] = Seam(cx.run(layer, '4ch', 'chain', name, **kv))
    return cx.cache[key]


def c_mono(cx, layer, g, sr, fft, tag='base', **opt):
    ins = grid_inputs(sr)
    name = 'mono_%s_%s.bin' % (g, tag)
    kv = dict(mic=cx.inp(ins['mic1']), ref=cx.inp(ins['ref']), sr=sr, fft=fft)
    kv.update(opt)
    key = ('seam', layer, name)
    if key not in cx.cache:
        cx.cache[key] = Seam(cx.run(layer, 'mono', 'mono', name, **kv))
    return cx.cache[key]


def c_replay(cx, layer, kind, seam_file, name, usable=None, **opt):
    """Open-loop replay (built in `layer`) of the stages behind the AEC over a seam file written by a Layer-1 run."""
    kv = dict(seam=seam_file)
    if usable is not None:
        upath = os.path.join(cx.work, 'run', 'L%d' % layer, name + '.usable')
        os.makedirs(os.path.dirname(upath), exist_ok=True)
        np.asarray(usable, np.uint8).tofile(upath)
        kv['usable'] = upath
    kv.update(opt)
    key = ('replay', layer, name)
    if key not in cx.cache:
        cx.cache[key] = Replay(cx.run(layer, kind, 'replay', name, **kv))
    return cx.cache[key]


def seam_path(cx, layer, name):
    return os.path.join(cx.work, 'run', 'L%d' % layer, name)


# --------------------------------------------------------------------------------------------------
# Python references
# --------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def py_nr(cx, seam, usable, nr_preset='balanced'):
    """aec_nr_pipeline.run_nr_spectrum over a recorded seam, usable_linear from `usable`."""
    ctxs = [types.SimpleNamespace(error_spec=seam.E[h], r2=seam.R2[h], usable_linear=bool(usable[h]))
            for h in range(seam.nh)]
    with quiet():
        return cx.P.run_nr_spectrum(ctxs, seam.sr, nr_preset=nr_preset, inject_echo_psd=cx.P.PROD_INJECT_ECHO_PSD)


def py_res(cx, seam, preset):
    """PostBeamResidualSuppressor (+ set_preset) fed with the recorded fused frame."""
    from lib.aec.python.modules.dataclasses import AecResContext
    M4 = cx.M4
    sup = M4.PostBeamResidualSuppressor(seam.sr, seam.hop, seam.nf)
    if preset != 'balanced':
        sup.set_preset(preset)
    z = np.zeros(seam.hop, np.float32)
    zc = np.zeros(seam.nf, np.complex64)
    out = np.zeros((seam.nh, seam.nf), np.float32)
    for h in range(seam.nh):
        s = seam.sc[h]
        ctx = AecResContext(raw_output=z, formed_output=z, echo_spec=zc, far_power=float(s[0]), far_spec=zc,
                            near_spec=seam.Y[h], filter_converged=bool(s[3]), erle_factor=0.0,
                            dt_indicator=float(s[1]), divergence=0.0, over_sub=1.0, saturation_level=float(s[2]),
                            error_spec=seam.E[h], res_gain=np.ones(seam.nf, np.float32),
                            comfort_noise=seam.CN[h], r2=seam.R2[h], usable_linear=True)
        out[h] = sup.process(ctx, seam.far[h]).res_gain
    return out


def py4_stock(cx, g, sr, fft):
    """Stock Python 4-channel pipeline on the 4-mic input; stops at the first exception, which is the result."""
    M4 = cx.M4
    ins = grid_inputs(sr)
    mics = np.fromfile(cx.inp(ins['mic4']), np.float32).reshape(-1, 4)
    ref = np.fromfile(cx.inp(ins['ref']), np.float32)
    pipe = M4.FourChannelAecPipeline(M4.FourChannelAecConfig(sample_rate=sr, frame_size=fft))
    hop, nf = pipe.hop_size, pipe.n_freqs
    W = np.full((4, nf), 0.25, np.complex64)
    nh = min(len(mics), len(ref)) // hop
    res = dict(raised=False, hop=None, exc_type=None, message=None, lane_far_spread=None, hops_run=0)
    orig = M4._fuse_contexts
    seen = {}

    def spy(contexts, bf):
        try:
            return orig(contexts, bf)
        except ValueError:
            far = np.stack([np.asarray(c.far_spec, np.complex64) for c in contexts])
            seen['spread'] = float(np.max(np.abs(far - far[0:1])))
            seen['peak'] = float(np.max(np.abs(far)))
            raise

    M4._fuse_contexts = spy
    try:
        for h in range(nh):
            sl = slice(h * hop, (h + 1) * hop)
            front = pipe.front_process(mics[sl], ref[sl])
            bf = M4.BeamformerFrame(samples=front.linear_channels.mean(axis=1, dtype=np.float32), weights=W)
            pipe.fuse(front, bf)
            res['hops_run'] = h + 1
    except Exception as exc:                                     # noqa: BLE001 -- the exception is the result
        res.update(raised=True, hop=res['hops_run'], exc_type=type(exc).__name__, message=str(exc))
        res['lane_far_spread'] = seen.get('spread')
        res['lane_far_peak'] = seen.get('peak')
    finally:
        M4._fuse_contexts = orig
    res['nh'] = nh
    return res


def py4_run(cx, g, sr, fft):
    """Python 4-channel pipeline, whole input. A mixed warm/soft-realign hop fuses lane 0's far spectrum in
    the stock pipeline (_fuse_contexts() does not raise, as the C core); this helper counts such hops from
    the pipeline's own realign lane counters for the verdict_flag row below."""
    key = ('py4', g)
    if key in cx.cache:
        return cx.cache[key]
    M4 = cx.M4
    ins = grid_inputs(sr)
    mics = np.fromfile(cx.inp(ins['mic4']), np.float32).reshape(-1, 4)
    ref = np.fromfile(cx.inp(ins['ref']), np.float32)
    pipe = M4.FourChannelAecPipeline(M4.FourChannelAecConfig(sample_rate=sr, frame_size=fft))
    hop, nf = pipe.hop_size, pipe.n_freqs
    W = np.full((4, nf), 0.25, np.complex64)
    nh = min(len(mics), len(ref)) // hop
    r = dict(verdict=np.zeros(nh, np.uint8), conv=np.zeros(nh, np.uint8), d_samples=np.zeros(nh, np.int32),
             d_solid=np.zeros(nh, np.uint8), d_changed=np.zeros(nh, np.uint8), warm=np.zeros(nh, np.int32),
             soft=np.zeros(nh, np.int32))
    mixed_hops = 0
    prev_warm, prev_soft = pipe.realign_warm_lane_count, pipe.realign_soft_lane_count
    for h in range(nh):
        sl = slice(h * hop, (h + 1) * hop)
        front = pipe.front_process(mics[sl], ref[sl])
        bf = M4.BeamformerFrame(samples=front.linear_channels.mean(axis=1, dtype=np.float32), weights=W)
        fr = pipe.fuse(front, bf)
        r['verdict'][h] = 1 if fr.context.usable_linear else 0
        r['conv'][h] = 1 if fr.context.filter_converged else 0
        r['d_samples'][h] = front.delay.delay_samples
        r['d_solid'][h] = 1 if front.delay.solid else 0
        r['d_changed'][h] = 1 if front.delay.changed else 0
        warm_now, soft_now = pipe.realign_warm_lane_count, pipe.realign_soft_lane_count
        r['warm'][h] = warm_now
        r['soft'][h] = soft_now
        if warm_now > prev_warm and soft_now > prev_soft:
            mixed_hops += 1
        prev_warm, prev_soft = warm_now, soft_now
    r['borrowed_hops'] = mixed_hops
    r['nh'] = nh
    cx.cache[key] = r
    return r


# --------------------------------------------------------------------------------------------------
# Section (a): 4-channel NR gain
# --------------------------------------------------------------------------------------------------
def sec_a(cx, g, sr, fft, want):
    rows = []
    tol = TOL1['gain_abs']
    base = c_chain(cx, 1, g, sr, fft)
    pre = 'a.nr4.%s.' % g
    prod_c = 'pyc_drv_4ch chain sr=%d fft=%d (equal weights, CNG on)' % (sr, fft)
    if want(pre + 'glue_cflag'):
        g_cflag = py_nr(cx, base, base.usable)
        m = stats(base.Gnr, g_cflag)
        rows.append(make_row(pre + 'glue_cflag', 1, 'NR glue: Python run_nr_spectrum(C fused frame, usable = C frame flag) vs C G_nr',
                             prod_c + '; run_nr_spectrum(C E, C R2, C flag)', gate=m['max_abs'] <= tol, metrics=m,
                             tol={'max_abs': tol}, grid=g, ref='Python',
                             note='usable_linear hops in the C frame: %d of %d' % (int(base.usable.sum()), base.nh)))
    need_py = any(want(pre + k) for k in ('glue_fold', 'verdict_flag', 'gnr_e2e'))
    if need_py:
        py = py4_run(cx, g, sr, fft)
        n = min(base.nh, py['nh'])
        verdict = py['verdict'][:n]
        if want(pre + 'glue_fold'):
            g_pyv = py_nr(cx, base, np.concatenate([verdict, np.zeros(base.nh - n, np.uint8)]))
            rep = c_replay(cx, 1, '4ch', seam_path(cx, 1, 'chain_%s_base.bin' % g), 'replay_%s_fold.bin' % g,
                           usable=np.concatenate([verdict, np.zeros(base.nh - n, np.uint8)]))
            m = stats(rep.Gnr, g_pyv)
            rows.append(make_row(pre + 'glue_fold', 1,
                                 'NR glue with the R^2 fold active: Python NR vs C se_anr_stage over the same frame, usable = Python verdict',
                                 'pyc_drv_4ch replay (usable = Python verdict); run_nr_spectrum(C E, C R2, Python verdict)',
                                 gate=m['max_abs'] <= tol, metrics=m, tol={'max_abs': tol}, grid=g, ref='Python',
                                 note='verdict true on %d of %d hops (this row exercises se_anr_stage.c r2 fold)' % (int(verdict.sum()), n)))
        if want(pre + 'verdict_flag'):
            mism = int(np.count_nonzero(verdict.astype(bool) != base.usable[:n]))
            first = np.flatnonzero(verdict.astype(bool) != base.usable[:n])
            m = {'hops': int(n), 'python_verdict_true_hops': int(verdict.sum()), 'c_flag_true_hops': int(base.usable[:n].sum()),
                 'mismatch_hops': mism, 'first_mismatch_hop': int(first[0]) if first.size else None,
                 's27_borrowed_hops': int(py['borrowed_hops'])}
            rows.append(make_row(pre + 'verdict_flag', 1, 'fused usable_linear: Python verdict vs C frame flag, per hop',
                                 prod_c + '; pipeline.fuse() context.usable_linear (stock Python 4ch; lane-0 far-spec fusion on a mixed warm/soft-realign hop, S-27, no raise)',
                                 gate=mism == 0, metrics=m, tol={'mismatch_hops': 0}, grid=g, ref='Python',
                                 note='C flag = four_aec_fuse AND of usable_linear over contributing lanes'))
        if want(pre + 'gnr_e2e'):
            g_pyv = py_nr(cx, base, np.concatenate([verdict, np.zeros(base.nh - n, np.uint8)]))
            m = stats(base.Gnr, g_pyv)
            rows.append(make_row(pre + 'gnr_e2e', 1, 'NR gain the Python reference would apply (Python verdict) vs the C G_nr',
                                 prod_c + '; run_nr_spectrum(C E, C R2, Python verdict)', gate=m['max_abs'] <= tol,
                                 metrics=m, tol={'max_abs': tol}, grid=g, ref='Python'))
    if want(pre + 'legacy_amin'):
        leg = c_chain(cx, 1, g, sr, fft, tag='leg1', legacy_amin=1)
        nonr = c_chain(cx, 1, g, sr, fft, tag='nonr', enable_nr=0)
        d_gnr = stats(base.Gnr, leg.Gnr)
        d_out = stats(base.out, leg.out)
        ctl = stats(base.out, nonr.out)
        m = {'legacy0_vs_1_gnr_max_abs': d_gnr['max_abs'], 'legacy0_vs_1_out_max_abs': d_out['max_abs'],
             'legacy0_vs_1_out_bit_identical': d_out['bit_identical'], 'legacy0_vs_1_gnr_bit_identical': d_gnr['bit_identical'],
             'positive_control_enable_nr0_vs_1_out_max_abs': ctl['max_abs'],
             'positive_control_differs': ctl['max_abs'] > 0.0}
        gate = (d_out['max_abs'] > 0.0) and ctl['max_abs'] > 0.0
        note = 'legacy_amin must change the output (R^2 fold on/off); positive control (enable_nr 0 vs 1) differs: %s' % m['positive_control_differs']
        rows.append(make_row(pre + 'legacy_amin', 1, 'C internal: legacy_amin 0 vs 1 must give different output',
                             'pyc_drv_4ch chain legacy_amin=0 / 1 / enable_nr=0', gate=gate, metrics=m,
                             tol={'out_max_abs_must_exceed': 0.0}, grid=g, ref='C (self)', note=note))
    return rows


# --------------------------------------------------------------------------------------------------
# Section (b): 4-channel post-beam RES
# --------------------------------------------------------------------------------------------------
def sec_b(cx, g, sr, fft, want):
    rows = []
    tol = TOL1['gain_abs']
    for p in PRESETS_B:
        rid = 'b.res4.%s.%s' % (g, p)
        if not want(rid):
            continue
        seam = c_chain(cx, 1, g, sr, fft, tag='aec_' + p, aec=p)
        gres_py = py_res(cx, seam, p)
        m = stats(seam.Gres, gres_py)
        dt = seam.max_dt > 0.2
        m['dt_hops'] = int(dt.sum())
        m['hops'] = int(seam.nh)
        if dt.any():
            m['dt_hops_max_abs'] = stats(seam.Gres[dt], gres_py[dt])['max_abs']
        if (~dt).any():
            m['other_hops_max_abs'] = stats(seam.Gres[~dt], gres_py[~dt])['max_abs']
        vacuous = (p == 'aggressive') and not dt.any()
        note = ('no double-talk hop (max_dt > 0.2) in this input: the S-26 DT floor is not exercised' if vacuous
                else 'double-talk hops: %d of %d' % (m['dt_hops'], m['hops']))
        if p == 'no_floor':
            # Vacuity guard: the gate is only exercised if it actually moved the C gain away from
            # balanced's, on the SAME mic/ref input. Also serves as a construction-time identity
            # probe on the C side (see pyc_drv.c): a no_floor construction left inert (gate stuck
            # enabled) would show up here as a no_floor seam indistinguishable from balanced's.
            base = c_chain(cx, 1, g, sr, fft, tag='aec_balanced', aec='balanced')
            d_vs_balanced = stats(seam.Gres, base.Gres)
            m['c_gres_vs_balanced_max_abs'] = d_vs_balanced['max_abs']
            gate_moved = d_vs_balanced['max_abs'] > 0.0
            vacuous = not gate_moved
            note = ('C no_floor Gres is bit-identical to C balanced Gres on this input -- the '
                    'no_floor construction never disabled the gate' if vacuous else
                    'C no_floor vs C balanced Gres max_abs %.4g (gate moved the output): %s' %
                    (d_vs_balanced['max_abs'], note))
        rows.append(make_row(rid, 1, 'post-beam RES: PostBeamResidualSuppressor(set_preset(%s)) fed with the C fused frame vs C G_res' % p,
                             'pyc_drv_4ch chain aec=%s sr=%d fft=%d; PostBeamResidualSuppressor.process(C E, C Y, C R2, C CN, C far hop)' % (p, sr, fft),
                             gate=m['max_abs'] <= tol, metrics=m, tol={'max_abs': tol}, grid=g, ref='Python',
                             not_run=vacuous, note=note))
    return rows


# --------------------------------------------------------------------------------------------------
# Section (c): 4-channel shared realign
# --------------------------------------------------------------------------------------------------
def realign_events(seam):
    """Hops on which the realign totals moved: (hop, d_warm, d_soft)."""
    w = seam.sc[:, 9].astype(int)
    s = seam.sc[:, 10].astype(int)
    ev = []
    pw = ps = 0
    for h in range(seam.nh):
        if w[h] != pw or s[h] != ps:
            ev.append((h, int(w[h] - pw), int(s[h] - ps)))
        pw, ps = int(w[h]), int(s[h])
    return ev


def sec_c(cx, g, sr, fft, want):
    rows = []
    rid = 'c.realign4.%s' % g
    if not want(rid):
        return rows
    seam = c_chain(cx, 1, g, sr, fft)
    stock = py4_stock(cx, g, sr, fft)
    ev = realign_events(seam)
    mixed = [e for e in ev if 0 < e[1] < 4]
    c_rejects = seam.fail_hop >= 0
    py = py4_run(cx, g, sr, fft)
    n2 = min(seam.nh, py['nh'])
    traj_eq = dict(
        delay_samples=int(np.count_nonzero(seam.sc[:n2, 6].astype(int) == py['d_samples'][:n2])),
        solid=int(np.count_nonzero(seam.sc[:n2, 7].astype(int) == py['d_solid'][:n2])),
        changed=int(np.count_nonzero(seam.sc[:n2, 8].astype(int) == py['d_changed'][:n2])))
    m = {'c_realign_events': [list(e) for e in ev], 'c_mixed_warm_soft_hops': [e[0] for e in mixed],
         'c_rejects': bool(c_rejects), 'c_fail_hop': int(seam.fail_hop), 'c_fail_rc': int(seam.fail_rc),
         'python_raised': bool(stock['raised']), 'python_raise_hop': stock['hop'], 'python_exception': stock['exc_type'],
         'python_message': stock['message'], 'python_lane_far_spread': stock.get('lane_far_spread'),
         'python_lane_far_peak': stock.get('lane_far_peak'), 'hops_compared': int(n2),
         'trajectory_equal_hops': traj_eq}
    prod = 'pyc_drv_4ch chain sr=%d fft=%d; stock FourChannelAecPipeline.front_process/fuse until the first exception' % (sr, fft)
    if g == '48k1024':
        # The scenario is vacuous unless the C core really took a mixed warm/soft realign on this input.
        vac = not mixed
        agree = (stock['raised'] == c_rejects) and (not c_rejects or stock['hop'] == seam.fail_hop)
        rows.append(make_row(rid, 1, 'mixed warm/soft realign hop: stock Python 4ch vs C core (S-27: both sides accept the lanes\' legitimate far-spectrum disagreement -- fused X takes lane 0, neither side raises/resets)',
                             prod, gate=agree, metrics=m, tol=None, grid=g, ref='Python', not_run=vac,
                             note=('no mixed warm/soft realign hop occurred in C: input does not exercise S-27' if vac else
                                   'C mixed hop(s) %s; Python raised=%s at hop %s; C rc=%s' % (m['c_mixed_warm_soft_hops'], stock['raised'], stock['hop'], seam.fail_rc))))
    else:
        warm_soft_equal = bool(np.array_equal(seam.sc[:n2, 9].astype(int), py['warm'][:n2]) and
                               np.array_equal(seam.sc[:n2, 10].astype(int), py['soft'][:n2]))
        m['warm_soft_totals_equal_all_hops'] = warm_soft_equal
        agree = (stock['raised'] == c_rejects) and warm_soft_equal
        rows.append(make_row(rid, 1, 'realign control (no mixed hop expected): stock Python 4ch vs C core, lane counters and delay state',
                             prod, gate=agree, metrics=m, tol=None, grid=g, ref='Python',
                             note='C mixed hops: %s; Python raised: %s' % (m['c_mixed_warm_soft_hops'], stock['raised'])))
    return rows


# --------------------------------------------------------------------------------------------------
# Section (d): mono glue
# --------------------------------------------------------------------------------------------------
def mono_cng_twin_out(cx, seam, cn_floor):
    """Python min / apply / comfort noise (cng_aec3 twin, the amplitude recipe of aec_cng_fill) / OLA over a C seam."""
    from pipelines.cng_aec3 import Aec3ComfortNoise
    P = cx.P
    nf, hop = seam.nf, seam.hop
    bs = 2 * (nf - 1)
    win = P.synth_window(bs)
    ola = np.zeros(bs, np.float32)
    gen = Aec3ComfortNoise()
    psd = np.float32(32768.0 ** 2)
    out = np.zeros(seam.nh * hop, np.float32)
    for h in range(seam.nh):
        gt = P.total_gain(seam.Gnr[h], seam.Gres[h], True, True, 'min', nf)
        spec = P.apply_gain(types.SimpleNamespace(e=seam.E[h]), gt)
        if cn_floor is not None:
            n_amp = np.sqrt(np.maximum(seam.CN[h] / psd, 0.0)).astype(np.float32)
            ng = np.sqrt(np.maximum(1.0 - seam.Gres[h] * seam.Gres[h], 0.0)).astype(np.float32)
            nrf = np.maximum(seam.Gnr[h], np.float32(cn_floor)).astype(np.float32)
            gen.add(spec, ((n_amp * ng).astype(np.float32) * nrf).astype(np.float32))
        out[h * hop:(h + 1) * hop] = P.synthesize_hop(spec, ola, win, bs, bs, hop)
    return out


def sec_d(cx, g, sr, fft, want):
    rows = []
    pre = 'd.mono.%s.' % g
    tol, tol_out = TOL1['gain_abs'], TOL1['out_abs']
    prod = 'pyc_drv_mono mono sr=%d fft=%d' % (sr, fft)
    if want(pre + 'nr_glue') or want(pre + 'min_apply_ola'):
        nocng = c_mono(cx, 1, g, sr, fft, tag='nocng', enable_cng=0)
    if want(pre + 'nr_glue'):
        g_py = py_nr(cx, nocng, nocng.usable)
        m = stats(nocng.Gnr, g_py)
        rows.append(make_row(pre + 'nr_glue', 1, 'mono NR glue: Python run_nr_spectrum(C AEC seam, C usable flag) vs C G_nr',
                             prod + ' enable_cng=0; run_nr_spectrum(C E, C R2, C flag)', gate=m['max_abs'] <= tol, metrics=m,
                             tol={'max_abs': tol}, grid=g, ref='Python',
                             note='usable_linear hops: %d of %d (this row exercises the r2 fold)' % (int(nocng.usable.sum()), nocng.nh)))
    if want(pre + 'min_apply_ola'):
        out_py = mono_cng_twin_out(cx, nocng, None)
        m = stats(nocng.out, out_py)
        rows.append(make_row(pre + 'min_apply_ola', 1, 'mono min / apply / OLA: Python total_gain, apply_gain, synthesize_hop over the C seam and C gains vs C output (CNG off)',
                             prod + ' enable_cng=0; aec_nr_pipeline.total_gain/apply_gain/synthesize_hop', gate=m['max_abs'] <= tol_out,
                             metrics=m, tol={'max_abs': tol_out}, grid=g, ref='Python'))
    if want(pre + 'cng_glue'):
        cng = c_mono(cx, 1, g, sr, fft, tag='cng', enable_cng=1)
        out_py = mono_cng_twin_out(cx, cng, cx.P.CNG_NR_GAIN_FLOOR)
        m = stats(cng.out, out_py)
        rows.append(make_row(pre + 'cng_glue', 1, 'mono min / apply / comfort noise / OLA with CNG on: Python glue + cng_aec3 twin over the C seam vs C output',
                             prod + ' enable_cng=1; total_gain/apply_gain + cng_aec3.Aec3ComfortNoise.add/synthesize_hop', gate=m['max_abs'] <= tol_out,
                             metrics=m, tol={'max_abs': tol_out}, grid=g, ref='Python',
                             note='aec_nr_pipeline.comfort_noise_fill is a documented Gaussian look-alike and is not compared here (see d.mono.cng_lookalike)'))
    if want(pre + 'legacy_amin'):
        a = c_mono(cx, 1, g, sr, fft, tag='nocng', enable_cng=0)
        b = c_mono(cx, 1, g, sr, fft, tag='leg1', enable_cng=0, legacy_amin=1)
        d = stats(a.Gnr, b.Gnr)
        m = {'legacy0_vs_1_gnr_max_abs': d['max_abs'], 'legacy0_vs_1_gnr_bit_identical': d['bit_identical']}
        rows.append(make_row(pre + 'legacy_amin', 1, 'C internal (mono, control for S-25): legacy_amin 0 vs 1 changes the NR gain',
                             prod + ' legacy_amin=0 / 1', gate=d['max_abs'] > 0.0, metrics=m, tol={'gnr_max_abs_must_exceed': 0.0},
                             grid=g, ref='C (self)'))
    return rows


def sec_d_global(cx, want):
    rows = []
    if want('d.mono.cng_twin'):
        rows.append(cng_twin(cx))
    if want('d.mono.cng_lookalike'):
        rows.append(cng_lookalike(cx))
    if want('d.mono.8k'):
        P = cx.P
        try:
            P._project_grid(8000, None)
            raised, msg = False, None
        except Exception as exc:                                    # noqa: BLE001
            raised, msg = True, str(exc)
        r = subprocess.run([cx.drv(1, 'mono'), 'probe', 'sr=8000', 'fft=0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        c_line = r.stdout.decode(errors='replace').strip()
        m = {'python_supports_8k': not raised, 'python_message': msg, 'c_accepts_8k': r.returncode == 0, 'c_probe': c_line}
        rows.append(make_row('d.mono.8k', 1, 'mono at 8 kHz', 'aec_nr_pipeline._project_grid(8000); pyc_drv_mono probe sr=8000', skip=True,
                             metrics=m, grid=None, ref='Python',
                             note='SKIP: Python mono pipeline admits only 16000 and 48000 Hz (%s); C mono at 8 kHz: %s (C only, documented) -- no reference to compare' % (msg, c_line)))
    return rows


def cng_twin(cx):
    """C aec_cng_fill against cng_aec3.Aec3ComfortNoise, three hops with carried state, 18 cases."""
    from pipelines.cng_aec3 import Aec3ComfortNoise, SEED
    rs = np.random.RandomState(1)
    psd = np.float32(32768.0 ** 2)
    exe = cx.drv(1, 'mono')
    bad = []
    cases = 0
    max_abs = 0.0
    for n in (129, 257, 513):
        for floor in (0.31622777, 0.0, 2.5):
            for use_gnr in (True, False):
                spec = (rs.randn(n) + 1j * rs.randn(n)).astype(np.complex64)
                cn = (np.abs(rs.randn(n)) * 1e5).astype(np.float32)
                cn[3] = -5.0
                gr = rs.rand(n).astype(np.float32)
                gr[5] = 1.0
                gn = rs.rand(n).astype(np.float32)
                sc, py = spec.copy(), spec.copy()
                gen = Aec3ComfortNoise()
                rng_c = SEED
                for _ in range(3):
                    rng_c, sc = run_cng_c(cx, exe, n, rng_c, floor, sc, cn, gr, gn, use_gnr)
                    n_amp = np.sqrt(np.maximum(cn / psd, 0.0)).astype(np.float32)
                    ng = np.sqrt(np.maximum(1.0 - gr * gr, 0.0)).astype(np.float32)
                    nrf = (np.maximum(gn, np.float32(floor)) if use_gnr else
                           np.maximum(np.ones(n, np.float32), np.float32(floor))).astype(np.float32)
                    gen.add(py, ((n_amp * ng).astype(np.float32) * nrf).astype(np.float32))
                cases += 1
                eq = bool(np.array_equal(sc, py)) and rng_c == gen.state
                max_abs = max(max_abs, float(np.max(np.abs(sc - py))))
                if not eq:
                    bad.append('n%d_floor%s_gnr%d' % (n, floor, int(use_gnr)))
    m = {'cases': cases, 'bit_exact_cases': cases - len(bad), 'max_abs': max_abs, 'not_bit_exact': bad}
    return make_row('d.mono.cng_twin', 1, 'comfort noise: C aec_cng_fill vs cng_aec3.Aec3ComfortNoise (3 hops, carried generator state)',
                    'pyc_drv_mono cng; Aec3ComfortNoise.add', gate=(not bad), metrics=m, tol={'max_abs': TOL1['cng_abs']},
                    ref='Python', note='bit exact required (the generator is integer arithmetic plus identical float32 products)')


def run_cng_c(cx, exe, n, rng, floor, spec, cn, gr, gn, use_gnr):
    d = os.path.join(cx.work, 'run', 'L1')
    os.makedirs(d, exist_ok=True)
    fin, fout = os.path.join(d, 'cng_in.bin'), os.path.join(d, 'cng_out.bin')
    with open(fin, 'wb') as f:
        np.array([n], np.int32).tofile(f)
        np.array([rng], np.uint32).tofile(f)
        np.array([floor], np.float32).tofile(f)
        np.array([1 if use_gnr else 0], np.int32).tofile(f)
        np.stack([spec.real, spec.imag], -1).astype(np.float32).tofile(f)
        cn.astype(np.float32).tofile(f)
        gr.astype(np.float32).tofile(f)
        gn.astype(np.float32).tofile(f)
    r = subprocess.run([exe, 'cng', 'seam=' + fin, 'out=' + fout], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError('cng driver failed: ' + r.stderr.decode(errors='replace'))
    raw = open(fout, 'rb').read()
    rng_out = int(np.frombuffer(raw[:4], np.uint32)[0])
    s = np.frombuffer(raw[4:], np.float32).reshape(n, 2)
    return rng_out, (s[:, 0] + 1j * s[:, 1]).astype(np.complex64)


def cng_lookalike(cx):
    """aec_nr_pipeline.comfort_noise_fill (Gaussian look-alike) against the C fill: noise level only."""
    from pipelines.cng_aec3 import SEED
    from pipelines.se_stage import SeSpecFrame
    P = cx.P
    rs = np.random.RandomState(1)
    n = 513
    cn = (np.abs(rs.randn(n)) * 1e5 + 1e4).astype(np.float32)
    gr = (rs.rand(n) * 0.5).astype(np.float32)
    gn = np.ones(n, np.float32)
    spec0 = np.zeros(n, np.complex64)
    fr = SeSpecFrame(n_freqs=n, hop=512, e=spec0, cn=cn)
    exe = cx.drv(1, 'mono')
    tp, tc = [], []
    rng = np.random.RandomState(0)
    rng_c, sc = SEED, spec0.copy()
    for _ in range(400):
        tp.append(P.comfort_noise_fill(spec0, fr, gr, np.ones(n, np.float32), rng))
        rng_c, sc = run_cng_c(cx, exe, n, rng_c, 0.31622777, spec0, cn, gr, gn, True)
        tc.append(sc)
    tp, tc = np.array(tp), np.array(tc)
    rms = lambda x: float(np.sqrt(np.mean(x[:, 1:-1] ** 2)))             # noqa: E731
    m = {'rms_re_python': rms(tp.real), 'rms_re_c': rms(tc.real), 'rms_im_python': rms(tp.imag), 'rms_im_c': rms(tc.imag),
         'rms_ratio_python_over_c': rms(tp.real) / rms(tc.real), 'identical_samples': int(np.count_nonzero(tp == tc)), 'n': int(tp.size)}
    return make_row('d.mono.cng_lookalike', 1, 'mono comfort-noise fill of aec_nr_pipeline.py vs C (noise level; the source differs by design)',
                    'aec_nr_pipeline.comfort_noise_fill vs pyc_drv_mono cng, 400 hops, n=513', gate=None, metrics=m,
                    ref='Python', note='INFO: documented (D1) Gaussian look-alike of the AEC3 sin-LUT/LCG source; only the level recipe is shared. The exact twin is d.mono.cng_twin')


# --------------------------------------------------------------------------------------------------
# Layer 2
# --------------------------------------------------------------------------------------------------
def seg_hops(arr, k, hop):
    """First k hops of a per-hop array (nh, n) or of a flat sample stream."""
    arr = np.asarray(arr)
    return arr[:k] if arr.ndim == 2 else arr[:k * hop]


def chain_row(rid, title, producer, a, b, div, hop, tol_p99, grid, ref='C std-math'):
    """Closed-loop comparison: judged on the hops before the first divergence of the adaptive filters
    (E rel. 1e-2), because after it the two builds follow different trajectories and a pointwise error
    no longer measures the approximation. Whole-run numbers are reported next to it."""
    nh = a.shape[0] if a.ndim == 2 else a.shape[0] // hop
    k = nh if div is None else div
    whole = stats(a, b)
    pre = stats(seg_hops(a, k, hop), seg_hops(b, k, hop)) if k > 0 else None
    m = {'whole_run': whole, 'pre_divergence': pre, 'first_divergence_hop': div, 'pre_divergence_hops': int(k), 'hops': int(nh)}
    if pre is None or k < 50:
        return make_row(rid, 2, title, producer, not_run=True, metrics=m, grid=grid, ref=ref,
                        note='fewer than 50 hops before the first divergence: nothing to judge')
    ok = pre['p99_9'] <= tol_p99 and (pre['rms_db'] is None or pre['rms_db'] <= TOL2['chain_rms_db'])
    row = make_row(rid, 2, title, producer, gate=ok, metrics=m,
                   tol={'pre_divergence_p99_9': tol_p99, 'pre_divergence_rms_db': TOL2['chain_rms_db']}, grid=grid, ref=ref,
                   note='judged on hops 0..%d; first divergence hop: %s' % (k - 1, 'none' if div is None else div))
    row['judged'] = pre
    return row


def sec_l2(cx, g, sr, fft, want):
    rows = []
    if 2 not in cx.layers:
        return rows
    tg, to = TOL2['replay_gain_abs'], TOL2['replay_out_abs']
    base1 = c_chain(cx, 1, g, sr, fft)
    path1 = seam_path(cx, 1, 'chain_%s_base.bin' % g)
    pre = 'l2.nr4.%s.replay' % g
    if want(pre + '.gnr') or want(pre + '.out'):
        py = py4_run(cx, g, sr, fft)
        n = min(base1.nh, py['nh'])
        verdict = np.concatenate([py['verdict'][:n], np.zeros(base1.nh - n, np.uint8)])
        r1 = c_replay(cx, 1, '4ch', path1, 'replay2_%s_fold.bin' % g, usable=verdict)
        r2 = c_replay(cx, 2, '4ch', path1, 'replay2_%s_fold.bin' % g, usable=verdict)
        prod = 'pyc_drv_4ch replay seam=<L1 chain>, usable = Python verdict (R^2 fold active), built in both layers'
        for k, a, b, t in (('gnr', r1.Gnr, r2.Gnr, tg), ('out', r1.out, r2.out, to)):
            if want(pre + '.' + k):
                m = stats(a, b)
                rows.append(make_row(pre + '.' + k, 2, '4ch NR stage + min/apply/CNG/OLA over identical std-math frames, shipped vs std-math C: %s' % ('G_nr' if k == 'gnr' else 'output'),
                                     prod, gate=m['max_abs'] <= t, metrics=m, tol={'max_abs': t}, grid=g, ref='C std-math'))
    for p in PRESETS:
        rid = 'l2.res4.%s.%s.replay' % (g, p)
        if not want(rid):
            continue
        sp = c_chain(cx, 1, g, sr, fft, tag='aec_' + p, aec=p)
        sp_path = seam_path(cx, 1, 'chain_%s_aec_%s.bin' % (g, p))
        r1 = c_replay(cx, 1, '4ch', sp_path, 'replay2_%s_res_%s.bin' % (g, p), aec=p)
        r2 = c_replay(cx, 2, '4ch', sp_path, 'replay2_%s_res_%s.bin' % (g, p), aec=p)
        m = stats(r1.Gres, r2.Gres)
        m['dt_hops'] = int((sp.max_dt > 0.2).sum())
        rows.append(make_row(rid, 2, '4ch post-beam RES stage over identical std-math frames, shipped vs std-math C: G_res (%s)' % p,
                             'pyc_drv_4ch replay seam=<L1 chain aec=%s>, built in both layers' % p, gate=m['max_abs'] <= tg, metrics=m,
                             tol={'max_abs': tg}, grid=g, ref='C std-math'))
    pre = 'l2.mono.%s.replay' % g
    if want(pre + '.gnr') or want(pre + '.out'):
        mp = seam_path(cx, 1, 'mono_%s_cng.bin' % g)
        c_mono(cx, 1, g, sr, fft, tag='cng', enable_cng=1)
        r1 = c_replay(cx, 1, 'mono', mp, 'replay2_%s_mono.bin' % g, enable_cng=1)
        r2 = c_replay(cx, 2, 'mono', mp, 'replay2_%s_mono.bin' % g, enable_cng=1)
        for k, a, b, t in (('gnr', r1.Gnr, r2.Gnr, tg), ('out', r1.out, r2.out, to)):
            if want(pre + '.' + k):
                m = stats(a, b)
                rows.append(make_row(pre + '.' + k, 2, 'mono NR stage + min/apply/CNG/OLA over the identical std-math AEC seam, shipped vs std-math C: %s' % ('G_nr' if k == 'gnr' else 'output'),
                                     'pyc_drv_mono replay seam=<L1 mono>, built in both layers', gate=m['max_abs'] <= t, metrics=m,
                                     tol={'max_abs': t}, grid=g, ref='C std-math'))
    if any(want('l2.chain4.%s.%s' % (g, k)) for k in ('gnr', 'gres', 'out')):
        sh = c_chain(cx, 2, g, sr, fft)
        div = first_div_hop(sh.E, base1.E)
        for k, a, b, t in (('gnr', base1.Gnr, sh.Gnr, TOL2['chain_gain_p99_9']), ('gres', base1.Gres, sh.Gres, TOL2['chain_gain_p99_9']),
                           ('out', base1.out, sh.out, TOL2['chain_out_p99_9'])):
            rid = 'l2.chain4.%s.%s' % (g, k)
            if want(rid):
                rows.append(chain_row(rid, '4ch closed-loop chain, shipped vs std-math C: %s' % {'gnr': 'G_nr', 'gres': 'G_res', 'out': 'output'}[k],
                                      'pyc_drv_4ch chain built in both layers (the AEC lanes adapt: closed loop)', a, b, div, base1.hop, t, g))
    if want('l2.mono.%s.chain' % g + '.gnr') or want('l2.mono.%s.chain' % g + '.out'):
        a = c_mono(cx, 1, g, sr, fft, tag='cng', enable_cng=1)
        b = c_mono(cx, 2, g, sr, fft, tag='cng', enable_cng=1)
        div = first_div_hop(b.E, a.E)
        for k, x, y, t in (('gnr', a.Gnr, b.Gnr, TOL2['chain_gain_p99_9']), ('out', a.out, b.out, TOL2['chain_out_p99_9'])):
            rid = 'l2.mono.%s.chain.%s' % (g, k)
            if want(rid):
                rows.append(chain_row(rid, 'mono closed-loop pipeline, shipped vs std-math C: %s' % ('G_nr' if k == 'gnr' else 'output'),
                                      'pyc_drv_mono mono built in both layers (closed loop)', x, y, div, a.hop, t, g))
    if want('l2.realign4.%s' % g):
        a = c_chain(cx, 1, g, sr, fft)
        b = c_chain(cx, 2, g, sr, fft)
        ea, eb = realign_events(a), realign_events(b)
        m = {'std_math_events': [list(e) for e in ea], 'shipped_events': [list(e) for e in eb],
             'fail_hop_std': int(a.fail_hop), 'fail_hop_shipped': int(b.fail_hop)}
        rows.append(make_row('l2.realign4.%s' % g, 2, 'realign trajectory (hop, warm lanes, soft lanes) and rc: shipped vs std-math C',
                             'pyc_drv_4ch chain built in both layers', gate=(ea == eb and a.fail_hop == b.fail_hop), metrics=m, tol=None,
                             grid=g, ref='C std-math', htext='events %s' % (ea,)))
    return rows


# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------
def fmt(x, spec='%.3g'):
    return '-' if x is None else (spec % x)


def headline(row):
    """(max_abs, p99.9, bit identical, rms dB, text) shown in the table."""
    m = row['metrics']
    j = row.get('judged')
    if j is not None:
        return j['max_abs'], j['p99_9'], j['bit_identical'], j['rms_db'], row.get('headline')
    if 'max_abs' in m and 'p99_9' in m:
        return m['max_abs'], m['p99_9'], m['bit_identical'], m['rms_db'], row.get('headline')
    return None, None, None, None, row.get('headline')


def print_table(rows, out=sys.stdout):
    out.write('\n%-34s %-21s %-3s %-9s %-9s %-4s %-7s  %s\n' % ('scenario', 'status', 'lyr', 'max_abs', 'p99.9', 'bit', 'rms dB', 'tolerance (frozen) / note'))
    out.write('-' * 132 + '\n')
    for r in rows:
        ma, p99, bit, db, text = headline(r)
        tol = r['tolerance']
        tols = ' '.join('%s%s%s' % (k, '=' if k.endswith(('hops', 'must_exceed')) else '<=', v) for k, v in tol.items()) if tol else ''
        extra = text or ''
        out.write('%-34s %-21s %-3s %-9s %-9s %-4s %-7s  %s\n' % (
            r['id'], r['status'], r['layer'], fmt(ma), fmt(p99), '-' if bit is None else ('yes' if bit else 'no'),
            fmt(db, '%.1f'), (tols + ('  ' if tols and extra else '') + extra).strip()))
    out.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree')
    ap.add_argument('--work')
    ap.add_argument('--layers', default='1,2')
    ap.add_argument('--json')
    ap.add_argument('--build-info')
    ap.add_argument('--only', default='')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(row_ids()))
        return 0
    prefixes = [p for p in args.only.split(',') if p]

    def want(rid):
        return not prefixes or any(rid.startswith(p) or p.startswith(rid) for p in prefixes)

    cx = Ctx(args)
    rows = []
    errors = []
    for layer in (1, 2):
        key = 'L%d.flags' % layer
        if layer in cx.layers and key in cx.info:
            ok = cx.info[key].strip()
            rows.append(make_row('build.L%d.flags' % layer, layer,
                                 'build flags of the layer (%s)' % ('std-math, NR six USE_* off, scalar' if layer == 1 else 'Makefile defaults'),
                                 'make build log', gate=(ok == 'ok') if ok != 'not_asserted_reused_tree' else None,
                                 metrics={'assertion': ok}, note=ok))
    steps = []
    for g, sr, fft in GRIDS:
        if 1 in cx.layers:
            steps += [(sec_a, g, sr, fft), (sec_b, g, sr, fft), (sec_c, g, sr, fft), (sec_d, g, sr, fft)]
    for step in steps:
        fn, g, sr, fft = step
        sec = {'sec_a': 'a.nr4.', 'sec_b': 'b.res4.', 'sec_c': 'c.realign4.', 'sec_d': 'd.mono.'}[fn.__name__]
        if not want(sec + g):
            continue
        try:
            rows += fn(cx, g, sr, fft, want)
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            errors.append((fn.__name__, g, traceback.format_exc()))
            rows.append(make_row('%s%s' % (sec, g), 1, 'scenario group could not run', fn.__name__, not_run=True,
                                 note='%s: %s' % (type(exc).__name__, exc), grid=g))
    try:
        rows += sec_d_global(cx, want) if 1 in cx.layers else []
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        errors.append(('sec_d_global', '-', traceback.format_exc()))
        rows.append(make_row('d.mono.global', 1, 'scenario group could not run', 'sec_d_global', not_run=True, note=str(exc)))
    if 2 in cx.layers:
        for g, sr, fft in GRIDS:
            if not any(want(i) for i in row_ids() if i.startswith('l2.') and ('.%s.' % g in i or i.endswith('.' + g))):
                continue
            try:
                rows += sec_l2(cx, g, sr, fft, want)
            except Exception as exc:                                # noqa: BLE001
                import traceback
                errors.append(('sec_l2', g, traceback.format_exc()))
                rows.append(make_row('l2.%s' % g, 2, 'scenario group could not run', 'sec_l2', not_run=True, note=str(exc), grid=g))
    order = {rid: i for i, rid in enumerate(row_ids())}
    rows.sort(key=lambda r: (order.get(r['id'], -1), r['id']))
    print_table(rows)
    for fn, g, tb in errors:
        sys.stderr.write('--- %s %s\n%s\n' % (fn, g, tb))
    counts = {}
    for r in rows:
        k = r['status'].split('(')[0]
        counts[k] = counts.get(k, 0) + 1
    print('summary: ' + ', '.join('%s=%d' % (k, counts[k]) for k in sorted(counts)))
    print('tolerances are FROZEN (S-28, 2026-10-20); EXPECTED_RED rows carry the audit id of the gap they document.')
    inputs = {}
    mf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs.sha256')
    for line in open(mf):
        line = line.strip()
        if line and not line.startswith('#'):
            d, n = line.split(None, 1)
            inputs[n.strip()] = d
    result = {
        'schema': 1, 'tool': 'pyc_parity_pipelines', 'tolerances': 'frozen_2026_10_20_S28',
        'tolerance_table': {'layer1': TOL1, 'layer2': TOL2},
        'builds': {'1': BUILD_NAME[1], '2': BUILD_NAME[2]},
        'inputs_sha256': inputs,
        'sources_sha256': {s: cx.tree_sha(s) for s in SOURCES},
        'summary': counts,
        'scenarios': rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write('\n')
    bad = [r for r in rows if r['status'] == 'FAIL' or r['status'].startswith('UNEXPECTED_GREEN')]
    notrun = [r for r in rows if r['status'] == 'NOT_RUN']
    if bad:
        print('RESULT: FAIL (%s)' % ', '.join(r['id'] for r in bad))
        return 1
    if notrun:
        print('RESULT: INCOMPLETE (NOT_RUN: %s)' % ', '.join(r['id'] for r in notrun))
        return 3
    print('RESULT: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
