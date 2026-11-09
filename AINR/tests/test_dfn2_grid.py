"""DFN2Grid (dfn2_grid.h): a grid is a run-time value the application builds.

The library holds no grid. ``dfn2_grid_make()`` takes what a trained model fixes (sample rate,
fft size, ERB bands, deep-filter bins and order, the lookahead pair, the frames per call and the
seven feature-normaliser settings) and derives the rest; ``dfn2_grid_validate()`` re-derives and
compares; ``dfn2_norm_alpha()`` is train.py's ``make_norm_alpha`` in C. These tests pin:

  make / validate  every refusal, in the order the checks run, the grid left untouched by a
                   refusal; the derived fields of the 16, 32 and 48 kHz rows and of two
                   grids no preset knows; validate refusing a hand-edited derived field, an
                   alpha included
  no capacity      sizes no capacity bounds (n_fft 2048 and 8192, 36 bands, 97 and 128
                   deep-filter bins, a ring of 6 or 7) are accepted; the n_fft rule is
                   audiolib's
  mask-only        df_bins up to n_bins, no deep-filter lookahead, ring = mask + 1, any order
                   stored as 0
  make = validate  what make accepts, validate accepts, over a sweep of sample rates (1 to INT_MAX):
                   a rate whose norm alpha, as float32, is not inside (0, 1) is a normaliser
                   refusal
  extremes         n_erb of 1 << 30 and 0x7ffffffc (and the largest and smallest ints in every
                   field) are refused with no signed overflow, under -fsanitize=undefined; the
                   gate is shown to fail on the old `2 * n_erb > n_bins`
  alpha            ``dfn2_norm_alpha(sr, hop, tau)`` equals ``make_norm_alpha`` of train.py as
                   float32, bit for bit, over a sweep of sample rates, hops and taus (the float32
                   tau the library holds), including taus placed within an ulp of a rounding
                   boundary; the three shipped alphas are 0.989 / 0.992 / 0.992
  parity           the C features / compose / stream host equal the Python model on every
                   grid, from ONE binary that takes the grid from its arguments

The C programs are built a few at a time and memoised: the probe once per variant, the parity
wrapper and the stream driver once per SIMD mode.
"""

import concurrent.futures
import ctypes
import itertools
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import dfn_pairs

FAMILY = 'DeepFilterNet2'
DFN2 = dfn_pairs.family_dir(FAMILY)
# dfn2_*.c/.h moved to lib/ainr/dfn2/ in the single-model-release restructuring; DFN2 (still
# AINR/DeepFilterNet2, the Python training dir) now holds only config.ini and forwarding stubs.
DFN2_C_DIR = dfn_pairs.AINR.parent / 'lib' / 'ainr' / 'dfn2'

SIMD_MODES = (1, 0)

pytestmark = pytest.mark.skipif(
    dfn_pairs.require_toolchain() is not None,
    reason='C build tools unavailable')


def _f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _cc(args, **kwargs):
    return subprocess.run(['cc', *args], capture_output=True, text=True,
                          **kwargs)


@pytest.fixture(scope='module')
def dfn_py():
    """(model.py, train.py, export_onnx.py) of DeepFilterNet2."""
    return tuple(dfn_pairs.import_family_module(FAMILY, name)
                 for name in ('model', 'train', 'export_onnx'))


# ---------------------------------------------------------------------------
# the probe: dfn2_grid_make / _validate / _norm_alpha / _status_text, line by line
# ---------------------------------------------------------------------------

_PROBE = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dfn2_process.h"

static uint32_t bits_of(float v) { uint32_t u; memcpy(&u, &v, sizeof u); return u; }
static float float_of(uint32_t u) { float v; memcpy(&v, &u, sizeof v); return v; }

/* The model's order (config.ini), the spec's fields besides the six integers of a command. The
 * feature normaliser is a fixed constant inside dfn2_grid.h now, not a spec field. */
static DFN2GridSpec shipped_spec(void) {
    DFN2GridSpec spec;
    memset(&spec, 0, sizeof spec);
    spec.df_order = @ORDER@;
    spec.frames_per_call = 1;
    return spec;
}

/* "a b c d e f [key=value ...]": the six integers and optional spec fields, floats as hex bit
 * patterns (so a NaN is expressible). Returns the number of characters used, or -1. */
static int parse_spec(const char *text, DFN2GridSpec *spec) {
    int a, b, c, d, e, f, n = 0;
    char key[32], value[32];
    if (sscanf(text, "%d %d %d %d %d %d%n", &a, &b, &c, &d, &e, &f, &n) != 6) return -1;
    *spec = shipped_spec();
    spec->sample_rate = a;
    spec->n_fft = b;
    spec->n_erb = c;
    spec->df_bins = d;
    spec->mask_lookahead = e;
    spec->df_lookahead = f;
    for (;;) {
        int m = 0;
        if (sscanf(text + n, " %31[a-z_]=%31s%n", key, value, &m) != 2) break;
        n += m;
        if (!strcmp(key, "order")) spec->df_order = (int)strtol(value, NULL, 10);
        else if (!strcmp(key, "frames")) spec->frames_per_call = (int)strtol(value, NULL, 10);
        else return -1;
    }
    return n;
}

/* Set one field of a grid: integer fields from a decimal, float fields from a hex bit pattern. */
static int set_field(DFN2Grid *g, const char *name, const char *value) {
    long iv = strtol(value, NULL, 10);
    uint32_t fv = (uint32_t)strtoul(value, NULL, 16);
    if (!strcmp(name, "sample_rate")) g->sample_rate = (int)iv;
    else if (!strcmp(name, "n_fft")) g->n_fft = (int)iv;
    else if (!strcmp(name, "hop")) g->hop = (int)iv;
    else if (!strcmp(name, "n_bins")) g->n_bins = (int)iv;
    else if (!strcmp(name, "n_erb")) g->n_erb = (int)iv;
    else if (!strcmp(name, "df_bins")) g->df_bins = (int)iv;
    else if (!strcmp(name, "df_order")) g->df_order = (int)iv;
    else if (!strcmp(name, "mask_lookahead")) g->mask_lookahead = (int)iv;
    else if (!strcmp(name, "df_lookahead")) g->df_lookahead = (int)iv;
    else if (!strcmp(name, "df_history")) g->df_history = (int)iv;
    else if (!strcmp(name, "ring")) g->ring = (int)iv;
    else if (!strcmp(name, "frames_per_call")) g->frames_per_call = (int)iv;
    else if (!strcmp(name, "input_frames")) g->input_frames = (int)iv;
    else if (!strcmp(name, "erb_norm_alpha")) g->erb_norm_alpha = float_of(fv);
    else if (!strcmp(name, "spec_norm_alpha")) g->spec_norm_alpha = float_of(fv);
    else return -1;
    return 0;
}

int main(void) {
    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        char verb[16] = "";
        int a;
        DFN2GridSpec spec;
        int used;
        if (sscanf(line, "%15s", verb) != 1) continue;
        if (!strcmp(verb, "make")) {
            DFN2Grid g, before;
            int st, vst = 0, untouched = -1;
            if (parse_spec(line + 5, &spec) < 0) return 2;
            memset(&g, 0xA5, sizeof g);
            before = g;
            st = dfn2_grid_make(&g, &spec);
            if (st != 0) untouched = memcmp(&g, &before, sizeof g) == 0;
            else vst = dfn2_grid_validate(&g);
            if (st != 0) memset(&g, 0, sizeof g);
            printf("make %d %d %d %d %d %d %d %08x %08x %08x %08x %d %d %d\n", st, vst, g.hop,
                   g.n_bins, g.df_history, g.ring, g.n_erb == 0 ? 0 : 1,
                   bits_of(g.erb_norm_alpha), bits_of(g.spec_norm_alpha),
                   bits_of(dfn2_stft_scale(g.n_fft)), bits_of(dfn2_stft_unscale(g.n_fft)),
                   untouched, g.df_order, g.input_frames);
        } else if (!strcmp(verb, "edit")) {
            DFN2Grid g;
            char field[32], value[32];
            int st;
            used = parse_spec(line + 5, &spec);
            if (used < 0 || sscanf(line + 5 + used, "%31s %31s", field, value) != 2) return 2;
            st = dfn2_grid_make(&g, &spec);
            if (st != 0 || set_field(&g, field, value) != 0) {
                printf("edit %d -99\n", st);
            } else {
                printf("edit 0 %d\n", dfn2_grid_validate(&g));
            }
        } else if (!strcmp(verb, "null")) {
            DFN2Grid g;
            spec = shipped_spec();
            spec.sample_rate = 16000; spec.n_fft = 256; spec.n_erb = 24; spec.df_bins = 96;
            spec.mask_lookahead = 1; spec.df_lookahead = 1;
            printf("null %d %d %d\n", dfn2_grid_make(NULL, &spec), dfn2_grid_make(&g, NULL),
                   dfn2_grid_validate(NULL));
        } else if (!strcmp(verb, "text")) {
            const char *t;
            if (sscanf(line, "%*s %d", &a) != 1) return 2;
            t = dfn2_grid_status_text(a);
            printf("text %d %s\n", a, t ? t : "(null)");
        } else if (!strcmp(verb, "alpha")) {
            int sr, hop;
            double tau;
            if (sscanf(line, "%*s %d %d %lf", &sr, &hop, &tau) != 3) return 2;
            printf("alpha %08x\n", bits_of(dfn2_norm_alpha(sr, hop, (float)tau)));
        }
    }
    return 0;
}
'''


def _probe_source():
    values = dfn_pairs.dfn2_spec_values()
    return _PROBE.replace('@ORDER@', str(values['df_order']))


def _spec_options(options):
    """Spec fields of a probe command as ``key=value`` tokens: integers in decimal, floats as the
    hex bit pattern of the float32 (so a NaN or an exact tau is expressible)."""
    tokens = []
    for key, value in (options or {}).items():
        if key in ('order', 'frames'):
            tokens.append('%s=%d' % (key, value))
        else:
            tokens.append('%s=%08x' % (key, _f32_bits(value)))
    return ' '.join(tokens)


class _Probe:
    """One probe binary: its commands go in on stdin, one result line each."""

    def __init__(self, exe):
        self.exe = exe

    def run(self, lines):
        done = subprocess.run([str(self.exe)], input='\n'.join(lines) + '\n',
                              capture_output=True, text=True)
        assert done.returncode == 0, (done.returncode, done.stderr)
        out = done.stdout.splitlines()
        assert len(out) == len(lines), (len(out), len(lines))
        return out

    def make(self, grid, mask_lookahead, df_lookahead, **options):
        """dfn2_grid_make(sr, n_fft, n_erb, df_bins, mask, df, options) -> dict."""
        return self.make_all([(*grid, mask_lookahead, df_lookahead, options)])[0]

    def make_all(self, cases):
        """Each case: six ints, optionally followed by a dict of spec options (order, frames,
        erb_tau, erb_lo, erb_hi, erb_scale, spec_tau, spec_lo, spec_hi)."""
        rows = []
        lines = []
        for case in cases:
            options = case[6] if len(case) > 6 else None
            lines.append(('make %d %d %d %d %d %d ' % tuple(case[:6])) + _spec_options(options))
        for line in self.run(lines):
            fields = line.split()
            assert fields[0] == 'make', line
            (status, vstatus, hop, n_bins, df_history, ring, _built,
             erb_alpha, spec_alpha, scale, unscale, untouched, order,
             input_frames) = fields[1:]
            rows.append(dict(
                status=int(status), validate=int(vstatus), hop=int(hop),
                n_bins=int(n_bins), df_history=int(df_history), ring=int(ring),
                erb_alpha=int(erb_alpha, 16), spec_alpha=int(spec_alpha, 16),
                scale=int(scale, 16), unscale=int(unscale, 16),
                untouched=int(untouched), order=int(order),
                input_frames=int(input_frames)))
        return rows

    def edit_all(self, edits):
        """[(case, field, value)] -> [(make status, validate status)]."""
        out = []
        for line in self.run(['edit %d %d %d %d %d %d %s %s' % (*case[:6], field, value)
                              for case, field, value in edits]):
            _verb, make_status, validate_status = line.split()
            out.append((int(make_status), int(validate_status)))
        return out


class _Builds:
    """The programs this file compiles, memoised by (what, flags, SIMD mode)."""

    def __init__(self, work):
        self.work = work
        self._exes = {}
        self._counter = itertools.count(1)

    def _compile(self, label, source, flags, simd, units):
        stem = '%s_%d' % (label, next(self._counter))
        path = self.work / (stem + '.c')
        path.write_text(source, encoding='utf-8')
        exe = self.work / stem
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(DFN2),
                    str(path),
                    *[str(dfn_pairs.source_path(FAMILY, u)) for u in units],
                    dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(exe)])
        assert done.returncode == 0, done.stderr
        return exe

    def program(self, label, source, flags=(), simd=1,
                units=('process', 'model_io', 'prepost')):
        key = (label, tuple(flags), simd, tuple(units))
        if key not in self._exes:
            self._exes[key] = self._compile(label, source, flags, simd, units)
        return self._exes[key]

    def programs(self, specs):
        """Compile several (label, source, flags, simd, units) at once."""
        for simd in SIMD_MODES:     # resolved (and built) before the threads start
            dfn_pairs.house_cflags(simd)
            dfn_pairs.audio_common_lib(simd)
        todo = [s for s in specs
                if (s[0], tuple(s[2]), s[3], tuple(s[4])) not in self._exes]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for spec, exe in zip(todo, pool.map(lambda s: self._compile(*s), todo)):
                self._exes[(spec[0], tuple(spec[2]), spec[3], tuple(spec[4]))] = exe
        return [self._exes[(s[0], tuple(s[2]), s[3], tuple(s[4]))] for s in specs]


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    return _Builds(tmp_path_factory.mktemp('dfn2_grid_builds'))


# The two variants of the probe: the full model and the mask-only one.
_VARIANTS = {
    'full': [],
    'mask': ['-DDFN2_MASK_ONLY'],
}


@pytest.fixture(scope='module')
def probes(builds):
    specs = [('probe_' + name, _probe_source(), flags, 1, ('process',))
             for name, flags in _VARIANTS.items()]
    return dict(zip(_VARIANTS, map(_Probe, builds.programs(specs))))


# Status codes (dfn2_grid.h; -8 and -13 are unassigned).
(OK, ERR_NULL, ERR_SR, ERR_FFT, ERR_ERB, ERR_DF, ERR_MASK, ERR_DFLA, ERR_INC, ERR_ORDER,
 ERR_FRAMES, ERR_NORM_ALPHA) = (0, -1, -2, -3, -4, -5, -6, -7, -9, -10, -11, -12)

# name -> (sample rate, n_fft, n_erb, df_bins): the three shipped rows and two grids no
# preset knows (a 24 kHz model, a 16 kHz model with a narrower deep filter).
GRIDS = {
    '48k': (48000, 1024, 32, 96),
    '32k': (32000, 512, 28, 96),
    '16k': (16000, 256, 24, 96),
    'custom24k': (24000, 512, 24, 96),
    'custom16k_df64': (16000, 256, 24, 64),
}
DF_ORDER = 5


def _expect_f32(value):
    return _f32_bits(float(value))


def _scales(n_fft):
    """1/sqrt(n_fft) and sqrt(n_fft) in float32 arithmetic: what torch.stft(normalized=True)
    and its inverse apply."""
    root = np.sqrt(np.float32(n_fft))
    return (_f32_bits(float(np.float32(1.0) / root)), _f32_bits(float(root)))


# ---------------------------------------------------------------------------
# the derived fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pair', [(1, 1), (0, 0), (2, 0), (0, 1), (2, 2), (1, 3), (0, 4)])
@pytest.mark.parametrize('name', list(GRIDS))
def test_make_derives_the_grid(probes, dfn_py, name, pair):
    sr, n_fft, n_erb, df_bins = GRIDS[name]
    _model, train, _export = dfn_py
    got = probes['full'].make(GRIDS[name], *pair)
    hop = n_fft // 2
    alpha = _expect_f32(train.make_norm_alpha(sr, hop, 1.0))
    scale, unscale = _scales(n_fft)
    assert got['status'] == OK and got['validate'] == OK, got
    assert (got['hop'], got['n_bins']) == (hop, n_fft // 2 + 1)
    assert (got['order'], got['input_frames']) == (DF_ORDER, 3)
    assert got['df_history'] == DF_ORDER - pair[1] - 1
    assert got['ring'] == max(DF_ORDER, pair[0] + pair[1] + 1)
    assert got['erb_alpha'] == alpha and got['spec_alpha'] == alpha, (
        'alpha is exp(-(hop/sr)/tau) rounded to 3 decimals as the trainer derives it: %r'
        % (train.make_norm_alpha(sr, hop, 1.0),))
    assert (got['scale'], got['unscale']) == (scale, unscale)
    # train.py's analysis_scale is the same float32 (1 / sqrt(n_fft) at win_len == n_fft).
    assert got['scale'] == _expect_f32(train.analysis_scale(n_fft, n_fft, hop))


def test_the_shipped_rows_carry_the_alphas_of_their_models(probes):
    """The 48 / 32 / 16 kHz rows of the shipped normaliser (tau 1 s) are 0.989 / 0.992 / 0.992."""
    want = {'48k': 0.989, '32k': 0.992, '16k': 0.992}
    for name, literal in want.items():
        got = probes['full'].make(GRIDS[name], 1, 1)
        expected = _f32_bits(float(np.float32(literal)))
        assert got['erb_alpha'] == got['spec_alpha'] == expected, (name, got)
    # Not a vacuous literal: the neighbouring three-decimal values are other floats.
    assert _f32_bits(float(np.float32(0.989))) != _f32_bits(float(np.float32(0.99)))


def test_scales_are_exact_float32_for_every_legal_fft_size(probes):
    cases = [(16000, n, 4, 2, 0, 0) for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
    for case, got in zip(cases, probes['full'].make_all(cases)):
        assert got['status'] == OK, (case, got)
        assert (got['scale'], got['unscale']) == _scales(case[1]), case


# ---------------------------------------------------------------------------
# every refusal
# ---------------------------------------------------------------------------

# (case, expected status), full variant. n_fft is audiolib's rule: a power of two in [16, 8192].
_REFUSALS = [
    # sample rate
    ((0, 256, 24, 96, 1, 1), ERR_SR),
    ((-1, 256, 24, 96, 1, 1), ERR_SR),
    ((-16000, 256, 24, 96, 1, 1), ERR_SR),
    # n_fft: a power of two in [16, 8192]
    ((16000, 0, 24, 96, 1, 1), ERR_FFT),
    ((16000, -256, 24, 96, 1, 1), ERR_FFT),
    ((16000, 8, 24, 96, 1, 1), ERR_FFT),
    ((16000, 15, 24, 96, 1, 1), ERR_FFT),
    ((16000, 17, 24, 96, 1, 1), ERR_FFT),
    ((16000, 255, 24, 96, 1, 1), ERR_FFT),
    ((16000, 960, 24, 96, 1, 1), ERR_FFT),
    ((16000, 1000, 24, 96, 1, 1), ERR_FFT),
    ((16000, 16384, 24, 96, 1, 1), ERR_FFT),
    ((16000, 1 << 30, 24, 96, 1, 1), ERR_FFT),
    # n_erb: a positive multiple of 4, every band at least two bins (2 * n_erb <= n_bins)
    ((16000, 256, 0, 96, 1, 1), ERR_ERB),
    ((16000, 256, -4, 96, 1, 1), ERR_ERB),
    ((16000, 256, 2, 96, 1, 1), ERR_ERB),
    ((16000, 256, 3, 96, 1, 1), ERR_ERB),
    ((16000, 256, 30, 96, 1, 1), ERR_ERB),
    ((16000, 256, 22, 96, 1, 1), ERR_ERB),
    ((16000, 256, 68, 96, 1, 1), ERR_ERB),       # 2 * 68 > 129 bins
    ((16000, 64, 20, 4, 0, 0), ERR_ERB),         # 2 * 20 > 33 bins
    ((16000, 16, 8, 4, 0, 0), ERR_ERB),          # 2 * 8 > 9 bins
    ((48000, 1024, 33, 96, 1, 1), ERR_ERB),      # one band more than a multiple of four
    # so large that 2 * n_erb does not fit an int: a refusal, not undefined behaviour
    ((16000, 256, 1 << 30, 96, 1, 1), ERR_ERB),
    ((16000, 256, 0x7ffffffc, 96, 1, 1), ERR_ERB),    # the largest multiple of four
    ((16000, 256, 0x7fffffff, 96, 1, 1), ERR_ERB),
    ((16000, 256, -(1 << 31), 96, 1, 1), ERR_ERB),
    # df_bins in [2, n_bins) for a full build
    ((16000, 256, 24, 1, 1, 1), ERR_DF),
    ((16000, 256, 24, 0, 1, 1), ERR_DF),
    ((16000, 256, 24, -3, 1, 1), ERR_DF),
    ((16000, 256, 24, 129, 1, 1), ERR_DF),       # df_bins == n_bins: nothing left for the mask
    ((16000, 256, 24, 130, 1, 1), ERR_DF),
    ((16000, 64, 16, 33, 1, 1), ERR_DF),
    # lookahead pair (the order is 5: df_lookahead in [0, 4])
    ((16000, 256, 24, 96, -1, 1), ERR_MASK),
    ((16000, 256, 24, 96, 3, 1), ERR_MASK),
    ((16000, 256, 24, 96, 1, -1), ERR_DFLA),
    ((16000, 256, 24, 96, 1, 5), ERR_DFLA),
    # the order of the checks: the first failing one is reported
    ((0, 3, 3, 1, 9, 9), ERR_SR),
    ((16000, 3, 3, 1, 9, 9), ERR_FFT),
    ((16000, 256, 3, 1, 9, 9), ERR_ERB),
    ((16000, 256, 24, 1, 9, 9), ERR_DF),
    ((16000, 256, 24, 96, 9, 9), ERR_MASK),
    ((16000, 256, 24, 96, 1, 9), ERR_DFLA),
    ((48000, 4096, 32, 96, 1, 9), ERR_DFLA),
]

# (case with options, expected status): the fields the model config fixes besides the six
# integers. Everything is checked after the six, in the order order, mask, df lookahead, frames.
# The feature normaliser is a fixed constant now (dfn2_grid.h), not a spec field, so there is
# nothing left to corrupt at this level -- its own refusal (ERR_NORM_ALPHA) is exercised purely
# by sample rate elsewhere in this file (test_make_ok_implies_validate_ok_over_a_sweep_of_sample_rates).
_SPEC_REFUSALS = [
    ((16000, 256, 24, 96, 1, 1, {'order': 0}), ERR_ORDER),
    ((16000, 256, 24, 96, 1, 1, {'order': -1}), ERR_ORDER),
    ((16000, 256, 24, 96, 1, 1, {'order': 17}), ERR_ORDER),
    ((16000, 256, 24, 96, 1, 1, {'order': 2}), OK),           # df_lookahead 1 < 2
    ((16000, 256, 24, 96, 1, 2, {'order': 2}), ERR_DFLA),     # df_lookahead must stay below it
    ((16000, 256, 24, 96, 1, 1, {'order': 16}), OK),
    ((16000, 256, 24, 96, 1, 1, {'frames': 0}), ERR_FRAMES),
    ((16000, 256, 24, 96, 1, 1, {'frames': -1}), ERR_FRAMES),
    ((16000, 256, 24, 96, 1, 1, {'frames': 65}), ERR_FRAMES),
    ((16000, 256, 24, 96, 1, 1, {'frames': 64}), OK),
    # the order of the checks: order, then mask, then df lookahead, then frames
    ((16000, 256, 24, 96, 9, 9, {'order': 0, 'frames': 0}), ERR_ORDER),
    ((16000, 256, 24, 96, 9, 1, {'frames': 0}), ERR_MASK),
    ((16000, 256, 24, 96, 1, 9, {'frames': 0}), ERR_DFLA),
]


def test_every_refusal_reports_its_status_and_leaves_the_grid_alone(probes):
    cases = [case for case, _status in _REFUSALS + _SPEC_REFUSALS]
    for (case, want), got in zip(_REFUSALS + _SPEC_REFUSALS, probes['full'].make_all(cases)):
        assert got['status'] == want, (case, got)
        if want != OK:
            assert got['untouched'] == 1, ('a refused grid was written', case)
        else:
            assert got['validate'] == OK, ('make accepted a grid validate refuses', case, got)


def test_the_refusals_are_not_vacuous(probes):
    """One step inside each boundary, and the boundary itself, are accepted; so is every size a
    capacity bounds."""
    accepted = [
        (16000, 16, 4, 2, 0, 0),       # smallest legal frame, band count, df_bins
        (16000, 16, 4, 8, 0, 0),       # df_bins = n_bins - 1
        (16000, 64, 16, 32, 1, 1),     # 2 * n_erb = 32 <= 33; df_bins = n_bins - 1
        (16000, 256, 4, 2, 2, 2),      # lookahead 2 / 2
        (16000, 256, 24, 96, 0, 0),
        (16000, 256, 24, 96, 2, 0),
        (16000, 256, 24, 96, 0, 4),
        (48000, 1024, 32, 96, 2, 2),
        (16000, 256, 16, 96, 1, 1),
        # no capacity: larger frames, more bands, more deep-filter bins, deeper rings
        (48000, 2048, 32, 96, 1, 1),
        (48000, 8192, 32, 96, 1, 1),
        (48000, 1024, 36, 96, 1, 1),
        (48000, 1024, 32, 97, 1, 1),
        (48000, 1024, 32, 128, 1, 1),
        (48000, 1024, 32, 96, 2, 3),   # ring 6
        (48000, 1024, 32, 96, 2, 4),   # ring 7
        (48000, 1024, 32, 96, 1, 4),   # ring 6
    ]
    for case, got in zip(accepted, probes['full'].make_all(accepted)):
        assert got['status'] == OK and got['validate'] == OK, (case, got)
    assert [g['ring'] for g in probes['full'].make_all(accepted[-3:])] == [6, 7, 6]


def test_a_null_grid_is_refused(probes):
    (line,) = probes['full'].run(['null'])
    assert line == 'null %d %d %d' % (ERR_NULL, ERR_NULL, ERR_NULL)


# ---------------------------------------------------------------------------
# make and validate agree: what make accepts, validate accepts
# ---------------------------------------------------------------------------

# n_fft -> (n_erb, df_bins): a legal grid at that frame size, whatever the sample rate.
_RATE_SWEEP_GRIDS = {16: (4, 2), 256: (24, 96), 1024: (32, 96)}
INT_MAX = 2 ** 31 - 1


def _sweep_rates():
    """Sample rates 1 .. 2e6 coarse (every rate up to 130, then a ratio of 1.1), the edges the
    review found (1, 16, 67, INT_MAX) and a coarse climb to INT_MAX, where the exponential
    rounds to 1.0f."""
    rates = {*range(1, 131), 2_000_000, 44100, 22050, 11025, 88200, 192000}
    k = 0
    while round(1.1 ** k) <= 2_000_000:
        rates.add(round(1.1 ** k))
        k += 1
    rates |= {1, 16, 67, 10 ** 7, 10 ** 8, 10 ** 9, 2 ** 30, INT_MAX - 1, INT_MAX}
    return sorted(rates)


def _alpha_status(train, sample_rate, n_fft):
    """What dfn2_grid_make must return for an otherwise legal grid: the normaliser is refused
    exactly when the trainer's make_norm_alpha, as float32, is not strictly inside (0, 1)."""
    alpha = np.float32(train.make_norm_alpha(sample_rate, n_fft // 2, 1.0))
    return OK if 0.0 < alpha < 1.0 else ERR_NORM_ALPHA


def test_make_ok_implies_validate_ok_over_a_sweep_of_sample_rates(probes, dfn_py):
    """dfn2_grid_make must not return OK for a grid whose alpha is 0 or 1, which
    dfn2_grid_validate (and so every entry point) refuses. The two agree on every rate: OK from
    make is OK from validate, and a refusal is exactly an alpha outside (0, 1) as the trainer
    computes it, in float32."""
    _model, train, _export = dfn_py
    cases = [(sr, n_fft, *_RATE_SWEEP_GRIDS[n_fft], 1, 1)
             for sr in _sweep_rates() for n_fft in _RATE_SWEEP_GRIDS]
    assert len(cases) > 700
    outcomes = {OK: 0, ERR_NORM_ALPHA: 0}
    for case, got in zip(cases, probes['full'].make_all(cases)):
        want = _alpha_status(train, case[0], case[1])
        assert got['status'] == want, (case, got, want)
        if want == OK:
            assert got['validate'] == OK, ('make accepted a grid validate refuses', case, got)
        else:
            assert got['untouched'] == 1, ('a refused grid was written', case)
        outcomes[want] += 1
    # Not vacuous: both outcomes occur (low rates at a large hop and the top of the range).
    assert outcomes[OK] > 400 and outcomes[ERR_NORM_ALPHA] > 20, outcomes


def test_the_rates_the_review_named_are_decided_by_the_alpha(probes, dfn_py):
    """sr 1, 16, 67 and INT_MAX at n_fft 16 / 256 / 1024: refused (the normaliser) where the
    rounded alpha is 0 (a rate far below the hop rate) or reaches 1.0f (INT_MAX with a short
    hop), accepted otherwise. The expectation is written out, not only derived."""
    _model, train, _export = dfn_py
    want = {
        (1, 16): ERR_NORM_ALPHA, (1, 256): ERR_NORM_ALPHA, (1, 1024): ERR_NORM_ALPHA,   # exp(-hop) -> 0
        (16, 16): OK, (16, 256): ERR_NORM_ALPHA, (16, 1024): ERR_NORM_ALPHA,
        (67, 16): OK, (67, 256): OK, (67, 1024): ERR_NORM_ALPHA,          # exp(-512/67) < 5e-4
        (INT_MAX, 16): ERR_NORM_ALPHA,                                    # 0.99999999627 -> 1.0f
        (INT_MAX, 256): OK, (INT_MAX, 1024): OK,
    }
    cases = [(sr, n_fft, *_RATE_SWEEP_GRIDS[n_fft], 1, 1) for sr, n_fft in want]
    for (key, expected), case, got in zip(want.items(), cases, probes['full'].make_all(cases)):
        assert got['status'] == expected == _alpha_status(train, *key), (key, got)
        if expected == OK:
            assert got['validate'] == OK, (key, got)


# ---------------------------------------------------------------------------
# undefined behaviour: the arithmetic of make / validate on the largest ints
# ---------------------------------------------------------------------------

_UBSAN = ['-fsanitize=undefined', '-fno-sanitize-recover=undefined', '-fno-omit-frame-pointer',
          '-g']

_UBSAN_SELF_TEST = r'''
#include <limits.h>
int main(int argc, char **argv) {
    volatile int x = INT_MAX;
    (void)argv;
    x += argc;
    return x < 0;
}
'''


def _ubsan_unusable(work):
    """None when this host's toolchain builds with UBSan and the sanitizer traps (a sanitizer
    that is accepted as a flag but inert would turn the tests below into nothing); else why not."""
    source = work / 'ubsan_self_test.c'
    source.write_text(_UBSAN_SELF_TEST, encoding='utf-8')
    exe = work / 'ubsan_self_test'
    built = _cc([*_UBSAN, str(source), '-o', str(exe)])
    if built.returncode != 0:
        return 'cannot build with -fsanitize=undefined: %s' % built.stderr.strip()[-200:]
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    if ran.returncode == 0 or 'runtime error' not in ran.stderr:
        return 'the sanitizer did not report a signed overflow (rc %d)' % ran.returncode
    return None


@pytest.fixture(scope='module')
def ubsan_probe(builds):
    """The grid probe built with UBSan, aborting on the first finding; skipped (and the reason
    shown) on a host whose toolchain cannot sanitize."""
    why = _ubsan_unusable(builds.work)
    if why:
        pytest.skip('UBSan unusable on this host: ' + why)
    return _Probe(builds.program('probe_ubsan', _probe_source(), _UBSAN, 1, ('process',)))


def test_the_extremes_of_make_and_validate_run_clean_under_ubsan(ubsan_probe, probes, dfn_py):
    """n_erb = 1 << 30 and 0x7ffffffc (and every refusal of the table above, the sample-rate
    sweep and the largest and smallest ints in every field) give their status, with no signed
    overflow: `2 * n_erb > n_bins` overflowed for n_erb >= 2^30."""
    _model, train, _export = dfn_py
    cases = [case for case, _status in _REFUSALS + _SPEC_REFUSALS]
    wants = [status for _case, status in _REFUSALS + _SPEC_REFUSALS]
    for n_erb in (1 << 30, 0x7ffffffc, INT_MAX, -(1 << 31), 0x7ffffff8):
        for n_fft in (16, 256, 1024, 8192, INT_MAX):
            cases.append((16000, n_fft, n_erb, 96, 1, 1))
            wants.append(ERR_FFT if n_fft == INT_MAX else ERR_ERB)
    # every field at the largest, smallest and zero int: the answer is the unsanitised build's
    for field in range(6):
        for extreme in (INT_MAX, -(1 << 31), 0):
            case = [16000, 256, 24, 96, 1, 1]
            case[field] = extreme
            cases.append(tuple(case))
            wants.append(probes['full'].make(case[:4], *case[4:])['status'])
    for key in ('order', 'frames'):
        for extreme in (INT_MAX, -(1 << 31), 0):
            case = (16000, 256, 24, 96, 1, 1, {key: extreme})
            cases.append(case)
            wants.append(ERR_ORDER if key == 'order' else ERR_FRAMES)
    cases += [(sr, n_fft, *_RATE_SWEEP_GRIDS[n_fft], 1, 1)
              for sr in _sweep_rates() for n_fft in _RATE_SWEEP_GRIDS]
    wants += [_alpha_status(train, c[0], c[1]) for c in cases[len(wants):]]
    for case, want, got in zip(cases, wants, ubsan_probe.make_all(cases)):
        assert got['status'] == want, (case, got, want)
    # validate on a hand-edited struct holding the same extremes
    case = (*GRIDS['16k'], 1, 1)
    edits = [(case, field, value)
             for field in ('n_erb', 'df_bins', 'n_fft', 'sample_rate', 'mask_lookahead',
                           'df_lookahead')
             for value in (1 << 30, 0x7ffffffc, INT_MAX, -(1 << 31))]
    edits += [(case, field, value)
              for field in ('hop', 'n_bins', 'df_history', 'ring', 'input_frames', 'df_order',
                            'frames_per_call')
              for value in (INT_MAX, -(1 << 31))]
    for (_case, field, value), got, ref in zip(edits, ubsan_probe.edit_all(edits),
                                               probes['full'].edit_all(edits)):
        assert got == ref and got[0] == OK, (field, value, got, ref)
        assert got[1] != OK, (field, value, got)
    # the alpha helper on awkward arguments: zero, infinity, subnormal and huge taus
    lines = ubsan_probe.run(['alpha %d %d %r' % c for c in [
        (16000, 128, 1e-300), (16000, 128, 1e300), (16000, 128, float('inf')),
        (16000, 128, float('nan')), (INT_MAX, 8, 1.0), (1, 4096, 1.0), (1, INT_MAX, 1.0),
        (INT_MAX, INT_MAX, 1.0), (16000, -128, 1.0), (-16000, 128, 1.0)]])
    assert len(lines) == 10


def test_the_ubsan_gate_can_fail(builds, ubsan_probe, tmp_path):
    """The same cases against a copy of dfn2_process.c with the old `2 * n_erb > n_bins`
    abort under the sanitizer, naming the signed overflow; the shipped file does not."""
    source = (DFN2_C_DIR / 'dfn2_process.c').read_text(encoding='utf-8')
    fixed = 'spec->n_erb > n_bins / 2'
    assert source.count(fixed) == 1, 'the overflow-free form is gone: update this mutation'
    mutated = tmp_path / 'dfn2_process.c'
    mutated.write_text(source.replace(fixed, '2 * spec->n_erb > n_bins'), encoding='utf-8')
    probe_source = tmp_path / 'probe.c'
    probe_source.write_text(_probe_source(), encoding='utf-8')
    exe = tmp_path / 'probe_mutated'
    done = _cc([*dfn_pairs.house_cflags(1), *_UBSAN, '-I', str(DFN2), str(probe_source),
                str(mutated), dfn_pairs.audio_common_lib(1), '-lm', '-o', str(exe)])
    assert done.returncode == 0, done.stderr
    cases = ['make 16000 256 %d 96 1 1' % n_erb for n_erb in (1 << 30, 0x7ffffffc)]
    for case in cases:
        ran = subprocess.run([str(exe)], input=case + '\n', capture_output=True, text=True)
        assert ran.returncode != 0 and 'signed integer overflow' in ran.stderr, (case, ran)
        clean = subprocess.run([str(ubsan_probe.exe)], input=case + '\n', capture_output=True,
                               text=True)
        assert clean.returncode == 0 and clean.stdout.split()[1] == str(ERR_ERB), (case, clean)


# ---------------------------------------------------------------------------
# validate: the derived fields are re-derived, a hand edit is refused
# ---------------------------------------------------------------------------

def _hex(bits):
    return '%08x' % bits


def test_validate_refuses_a_hand_edited_derived_field(probes):
    edits = []
    for name in ('48k', '16k'):
        _sr, n_fft, _n_erb, _df_bins = GRIDS[name]
        case = (*GRIDS[name], 1, 1)
        hop, n_bins = n_fft // 2, n_fft // 2 + 1
        for field, value in [
                ('hop', hop + 1), ('hop', hop - 1), ('hop', n_fft), ('hop', 0),
                ('n_bins', n_bins + 1), ('n_bins', n_bins - 1), ('n_bins', 0),
                ('df_history', 0), ('df_history', 4), ('df_history', 2),
                ('ring', 4), ('ring', 6), ('ring', 7), ('ring', 0),
                ('input_frames', 2), ('input_frames', 4), ('input_frames', 0),
                ('df_order', 4), ('df_order', 6),
                # the alphas are derived from the taus: any other value is refused
                ('erb_norm_alpha', _hex(_f32_bits(float(np.float32(0.5))))),
                ('spec_norm_alpha', _hex(_f32_bits(float(np.float32(0.979))))),
                ('erb_norm_alpha', '3f7fffff'), ('spec_norm_alpha', '00000001'),
                ('erb_norm_alpha', '00000000'), ('spec_norm_alpha', '00000000'),
                ('erb_norm_alpha', _hex(_f32_bits(1.0))),
                ('spec_norm_alpha', _hex(_f32_bits(1.0))),
                ('erb_norm_alpha', _hex(_f32_bits(1.5))),
                ('spec_norm_alpha', _hex(_f32_bits(2.0))),
                ('erb_norm_alpha', _hex(_f32_bits(-0.5))),
                ('spec_norm_alpha', _hex(_f32_bits(-0.992))),
                ('erb_norm_alpha', '7fc00000'), ('spec_norm_alpha', '7fc00000'),
                ('erb_norm_alpha', '7f800000'), ('spec_norm_alpha', '7f800000'),
                ('erb_norm_alpha', '80000000')]:                 # -0.0
            edits.append((case, field, value))
    for (case, field, value), (make_status, status) in zip(
            edits, probes['full'].edit_all(edits)):
        assert make_status == OK and status == ERR_INC, (case, field, value, status)

    # The given fields, edited to a value make() itself refuses: that status comes back.
    case48, case16 = (*GRIDS['48k'], 1, 1), (*GRIDS['16k'], 1, 1)
    refused = [
        (case16, 'sample_rate', 0, ERR_SR), (case16, 'n_fft', 960, ERR_FFT),
        (case16, 'n_erb', 30, ERR_ERB), (case16, 'df_bins', 1, ERR_DF),
        (case16, 'df_bins', 129, ERR_DF), (case16, 'mask_lookahead', 3, ERR_MASK),
        (case16, 'df_lookahead', 5, ERR_DFLA), (case16, 'df_lookahead', -1, ERR_DFLA),
        (case16, 'frames_per_call', 0, ERR_FRAMES), (case16, 'frames_per_call', 65, ERR_FRAMES),
        # A legal fft size on a grid derived for another one: the derived fields are stale.
        (case48, 'n_fft', 2048, ERR_INC), (case48, 'n_fft', 512, ERR_INC),
        (case16, 'n_fft', 512, ERR_INC),
        (case16, 'n_fft', 128, ERR_DF),          # 96 bins do not fit below 65
    ]
    for (case, field, value, want), (make_status, status) in zip(
            refused, probes['full'].edit_all([r[:3] for r in refused])):
        assert (make_status, status) == (OK, want), (case, field, value, status)


def test_validate_accepts_the_given_fields_moved_to_other_legal_values(probes):
    """A given field moved to another legal value is consistent as long as the derived fields
    still follow: the bands, the deep-filter bins, the lookahead within the ring. The alphas
    are not the model's to edit (they follow the fixed tau and the sample rate), so a retuned
    rate on a made grid leaves a stale alpha."""
    case = (*GRIDS['48k'], 1, 1)
    edits = [(case, 'n_erb', 28), (case, 'n_erb', 36), (case, 'df_bins', 64),
             (case, 'df_bins', 97), (case, 'df_bins', 2), (case, 'mask_lookahead', 0),
             (case, 'mask_lookahead', 2)]
    for (_case, field, value), (make_status, status) in zip(
            edits, probes['full'].edit_all(edits)):
        assert (make_status, status) == (OK, OK), (field, value, status)
    stale = [(case, 'sample_rate', 44100)]
    for (_case, field, value), (make_status, status) in zip(
            stale, probes['full'].edit_all(stale)):
        assert (make_status, status) == (OK, ERR_INC), (field, value, status)
    # A lookahead edited on a grid whose derived fields were made for the other one.
    (_make, status), = probes['full'].edit_all([((*GRIDS['16k'], 0, 2), 'df_lookahead', 1)])
    assert status == ERR_INC             # df_history 2 was left behind (3 for this lookahead)
    # ... and one that deepens the ring.
    (_make, status), = probes['full'].edit_all([((*GRIDS['16k'], 2, 2), 'df_lookahead', 3)])
    assert status == ERR_INC             # ring 5 and df_history 2 were left behind


# ---------------------------------------------------------------------------
# mask-only: no deep filter, so no deep-filter lookahead and no deep-filter ring
# ---------------------------------------------------------------------------

def test_mask_only_grid_rules(probes):
    mask = probes['mask']
    cases = [
        ((16000, 256, 24, 96, 1, 0), OK, 2),            # ring = mask + 1
        ((16000, 256, 24, 96, 0, 0), OK, 1),
        ((48000, 1024, 32, 96, 1, 0), OK, 2),
        ((16000, 64, 16, 33, 1, 0), OK, 2),             # df_bins == n_bins is legal here
        ((16000, 64, 16, 34, 1, 0), ERR_DF, None),
        ((16000, 64, 16, 1, 1, 0), ERR_DF, None),
        ((16000, 256, 24, 96, 1, 1), ERR_DFLA, None),   # nothing to look ahead for
        ((16000, 256, 24, 96, 0, 4), ERR_DFLA, None),
        ((16000, 256, 24, 96, 3, 0), ERR_MASK, None),
        ((16000, 256, 24, 129, 1, 0), OK, 2),           # df_bins == n_bins = 129
        ((16000, 256, 24, 96, 2, 0), OK, 3),            # ring 3
        ((16000, 256, 22, 96, 1, 0), ERR_ERB, None),
    ]
    for (case, want, ring), got in zip(cases, mask.make_all([c for c, _s, _r in cases])):
        assert got['status'] == want, (case, got)
        if want == OK:
            assert got['validate'] == OK and got['ring'] == ring, (case, got)
            # No deep filter: the order is stored as 0 and there are no taps in the past.
            assert (got['order'], got['df_history']) == (0, 0), (case, got)
    # df_order is ignored: any value in [0, 16] gives the same grid, one beyond is refused.
    base = mask.make((16000, 256, 24, 96), 1, 0)
    for order in (0, 1, 5, 16):
        assert mask.make((16000, 256, 24, 96), 1, 0, order=order) == base, order
    for order in (-1, 17):
        assert mask.make((16000, 256, 24, 96), 1, 0, order=order)['status'] == ERR_ORDER
    # df_bins == n_bins is no grid for the full build (and the mask-only build accepts what
    # the full one refuses for that reason alone).
    assert probes['full'].make((16000, 64, 16, 33), 1, 0)['status'] == ERR_DF
    # A hand-edited ring is refused for mask-only as well.
    (_make, status), = mask.edit_all([((16000, 256, 24, 96, 1, 0), 'ring', 5)])
    assert status == ERR_INC


# ---------------------------------------------------------------------------
# status text
# ---------------------------------------------------------------------------

def test_status_text_names_each_status(probes):
    lines = probes['full'].run(['text %d' % s for s in range(1, -16, -1)])
    texts = {}
    for line in lines:
        _verb, status, text = line.split(' ', 2)
        texts[int(status)] = text
    assigned = [s for s in range(OK, ERR_NORM_ALPHA - 1, -1) if s != -8]
    for status in assigned:
        assert texts[status] and texts[status] != '(null)', status
    assert len({texts[s] for s in assigned}) == len(assigned) == 12, 'texts must differ'
    for status, word in [(OK, 'ok'), (ERR_NULL, 'null'), (ERR_SR, 'sample rate'),
                         (ERR_FFT, 'n_fft'), (ERR_ERB, 'n_erb'), (ERR_DF, 'df_bins'),
                         (ERR_ORDER, 'df_order'), (ERR_MASK, 'mask_lookahead'),
                         (ERR_DFLA, 'df_lookahead'), (ERR_FRAMES, 'frames_per_call'),
                         (ERR_NORM_ALPHA, 'normaliser'),
                         (ERR_INC, 'derived')]:
        assert word in texts[status].lower(), (status, texts[status])
    # an alpha outside (0, 1) is a normaliser refusal
    assert 'alpha' in texts[ERR_NORM_ALPHA].lower(), texts[ERR_NORM_ALPHA]
    for status in (1, -8, -13, -14, -15):        # unassigned: still a non-NULL string
        assert texts[status] and texts[status] != '(null)'
        assert texts[status] not in {texts[s] for s in assigned}


# ---------------------------------------------------------------------------
# dfn2_norm_alpha is train.py's make_norm_alpha as float32, bit for bit
# ---------------------------------------------------------------------------

_SAMPLE_RATES = sorted({*range(8000, 96001, 1000), 11025, 22050, 44100, 88200, 192000})
_HOPS = [*(1 << k for k in range(3, 13)), 100, 160, 240, 441, 480, 960, 1000, 1920]
_SHIPPED = [(48000, 512), (32000, 256), (16000, 128)]


def _tau_boundary_cases():
    """float32 taus placed so that exp(-(hop / sr) / tau) falls within three float32 ulp of a
    rounding boundary (k + 1/2) / 10**digits, for the three- and four-digit rounding the trainer
    uses: the half-to-even tie handling decides these. The library holds the tau as float32, so
    the cases are float32 values and the neighbours are float32 neighbours."""
    cases = []
    for sr, hop in _SHIPPED + [(96000, 8), (44100, 441)]:
        for digits, ks in ((3, range(900, 1000)), (4, range(9990, 10000)),
                           (5, range(99990, 100000, 2))):
            for k in ks:
                target = (k + 0.5) / 10 ** digits
                tau = np.float32(-(hop / sr) / math.log(target))
                lo = hi = tau
                cases.append((sr, hop, float(tau)))
                for _ in range(3):
                    lo, hi = np.nextafter(lo, np.float32(0.0)), np.nextafter(hi, np.float32(1e9))
                    cases += [(sr, hop, float(lo)), (sr, hop, float(hi))]
    return cases


def _alpha_cases():
    cases = [(sr, hop, 1.0) for sr in _SAMPLE_RATES for hop in _HOPS]
    taus = [0.25 + 0.0025 * i for i in range(1501)]            # 0.25 .. 4.0
    cases += [(sr, hop, tau) for sr, hop in _SHIPPED for tau in taus]
    cases += [(sr, hop, tau) for sr, hop in _SHIPPED
              for tau in (0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    cases += _tau_boundary_cases()
    return cases


def _tau32(tau):
    """The tau the library holds: the float32 nearest to the decimal."""
    return float(np.float32(tau))


def test_norm_alpha_equals_the_trainers_make_norm_alpha_bit_for_bit(probes, dfn_py):
    """Bit for bit against make_norm_alpha evaluated at the float32 tau the library holds (a
    decimal that is not a float32, 0.3, enters as its float32 value)."""
    _model, train, _export = dfn_py
    cases = _alpha_cases()
    assert len(cases) > 9000
    lines = probes['full'].run(['alpha %d %d %r' % case for case in cases])
    bad = []
    for (sr, hop, tau), line in zip(cases, lines):
        got = int(line.split()[1], 16)
        want = _f32_bits(train.make_norm_alpha(sr, hop, _tau32(tau)))
        if got != want:
            bad.append((sr, hop, tau, '%08x' % got, '%08x' % want,
                        train.make_norm_alpha(sr, hop, _tau32(tau))))
    assert not bad, '%d of %d differ, first: %r' % (len(bad), len(cases), bad[:8])


def _single_precision_make_norm_alpha(sr, hop, tau):
    """make_norm_alpha with the exponential evaluated in float32: the shortcut a C port is
    tempted by (the library works on a float32 tau but must round the double result)."""
    exact = float(np.exp(np.float32(-(hop / sr) / tau)))
    digits, alpha = 3, 1.0
    while alpha >= 1.0:
        alpha = round(exact, digits)
        digits += 1
    return alpha


def test_norm_alpha_sweep_is_not_vacuous(probes, dfn_py):
    """The sweep reaches the 4- and 5-digit branch, and its boundary taus are the cases where
    an exponential evaluated in single precision goes wrong: a port with that shortcut would be
    caught."""
    _model, train, _export = dfn_py
    values = {train.make_norm_alpha(sr, hop, 1.0) for sr in _SAMPLE_RATES for hop in _HOPS}
    digits = {len(repr(v).split('.')[1]) for v in values}
    assert max(digits) >= 5 and min(digits) <= 3, digits
    boundary = _tau_boundary_cases()
    assert len({train.make_norm_alpha(sr, hop, tau) for sr, hop, tau in boundary}) > 100
    shortcut_wrong = sum(
        _single_precision_make_norm_alpha(sr, hop, tau) != train.make_norm_alpha(sr, hop, tau)
        for sr, hop, tau in boundary)
    assert shortcut_wrong > 0, 'the boundary taus do not tell a single-precision exponential apart'
    got = probes['full'].run(['alpha 48000 512 1.0', 'alpha 0 512 1.0', 'alpha 48000 0 1.0',
                              'alpha 48000 512 0.0', 'alpha 48000 512 -1.0'])
    assert int(got[0].split()[1], 16) == _f32_bits(float(np.float32(0.989)))
    # Nonsense arguments give 0.0f, which no grid accepts (validate wants (0, 1)).
    assert [int(g.split()[1], 16) for g in got[1:]] == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# the C features / compose / stream host equal the Python model, one binary per SIMD mode
# ---------------------------------------------------------------------------

PAIR = dfn_pairs.shipped_pair(FAMILY)      # config.ini's lookahead pair

_WRAPPER = r'''
#include <stdlib.h>
#include "dfn2_process.h"

typedef struct { DFN2State state; void *block; } ParityDfn;

/* The grid comes from the caller: one library serves every grid. */
void *parity_dfn_create(int sr, int n_fft, int n_erb, int df_bins, int mask_lookahead,
                        int df_lookahead, int df_order, const float *fwd, const float *inv) {
    DFN2GridSpec spec = {
        .sample_rate = sr, .n_fft = n_fft, .n_erb = n_erb, .df_bins = df_bins,
        .df_order = df_order, .mask_lookahead = mask_lookahead,
        .df_lookahead = df_lookahead, .frames_per_call = 1,
    };
    DFN2Grid grid;
    ParityDfn *host;
    size_t bytes;
    if (dfn2_grid_make(&grid, &spec) != 0) return NULL;
    host = (ParityDfn *)calloc(1, sizeof(*host));
    if (!host) return NULL;
    bytes = dfn2_state_mem_size(&grid);
    host->block = aligned_alloc(16, bytes);
    if (!host->block || dfn2_state_init(&host->state, host->block, bytes, &grid) != 0) {
        free(host->block);
        free(host);
        return NULL;
    }
    dfn2_set_erb_matrices(&host->state, fwd, inv);
    return host;
}
void parity_dfn_destroy(void *host) {
    if (!host) return;
    free(((ParityDfn *)host)->block);
    free(host);
}
void parity_dfn_features(void *host, const float *re, const float *im,
                         float *erb, float *spec) {
    dfn2_compute_features(&((ParityDfn *)host)->state, re, im, erb, spec);
}
int parity_dfn_compose(void *host, const float *re, const float *im,
                       const float *mask, const float *coefs, float alpha,
                       float *out_re, float *out_im) {
    return dfn2_enhance_spectrum(&((ParityDfn *)host)->state, re, im, mask, coefs, alpha,
                                 out_re, out_im);
}
'''

# The host loop of a stream: the heads and windows of the exported-graph wrapper go in, the
# enhanced spectrum comes out. Every size is read from the grid the arguments name.
_STREAM_DRIVER = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_model_io.h"
#include "dfn2_process.h"

static float *take(const float **cursor, size_t count) {
    float *out = (float *)malloc(count * sizeof(float));
    memcpy(out, *cursor, count * sizeof(float));
    *cursor += count;
    return out;
}

int main(int argc, char **argv) {
    FILE *in, *out;
    long size;
    unsigned char *blob;
    const float *cursor;
    int32_t frames;
    float *erb_fwd, *erb_inv, *feat_erb, *feat_spec, *spec_re, *spec_im;
    float *head_mask, *head_coefs, *head_alpha, *win_erb, *win_spec;
    float *zero_erb, *zero_spec, *zero_bins, *unit_mask, *zero_coefs, *out_re, *out_im;
    static DFN2State st;
    static DFN2ModelIOState io;
    DFN2GridSpec spec;
    DFN2Grid grid;
    void *state_block, *io_block;
    size_t state_bytes, io_bytes;
    int total, c, ready_bad = 0, window_bad = 0, windows = 0, emitted = 0;
    int first_ready = -1, n_erb, df_bins, n_bins, coefs_n, mask_la, df_la, wf;
    float atten_lim_db;

    if (argc != 11) return 2;
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = atoi(argv[4]);
    spec.n_fft = atoi(argv[5]);
    spec.n_erb = atoi(argv[6]);
    spec.df_bins = atoi(argv[7]);
    spec.mask_lookahead = atoi(argv[8]);
    spec.df_lookahead = atoi(argv[9]);
    spec.df_order = atoi(argv[10]);
    spec.frames_per_call = 1;
    if (dfn2_grid_make(&grid, &spec) != 0) return 2;
    atten_lim_db = (float)atof(argv[3]);
    wf = grid.input_frames;
    n_erb = grid.n_erb;
    df_bins = grid.df_bins;
    n_bins = grid.n_bins;
    mask_la = grid.mask_lookahead;
    df_la = grid.df_lookahead;
    coefs_n = df_bins * grid.df_order * 2;

    in = fopen(argv[1], "rb");
    if (!in) return 2;
    fseek(in, 0, SEEK_END);
    size = ftell(in);
    fseek(in, 0, SEEK_SET);
    blob = (unsigned char *)malloc((size_t)size);
    if (fread(blob, 1, (size_t)size, in) != (size_t)size) return 2;
    fclose(in);
    memcpy(&frames, blob, sizeof frames);
    cursor = (const float *)(blob + sizeof frames);
    erb_fwd = take(&cursor, (size_t)n_bins * n_erb);
    erb_inv = take(&cursor, (size_t)n_erb * n_bins);
    feat_erb = take(&cursor, (size_t)frames * n_erb);
    feat_spec = take(&cursor, (size_t)frames * 2 * df_bins);
    spec_re = take(&cursor, (size_t)frames * n_bins);
    spec_im = take(&cursor, (size_t)frames * n_bins);
    head_mask = take(&cursor, (size_t)frames * n_erb);
    head_coefs = take(&cursor, (size_t)frames * coefs_n);
    head_alpha = take(&cursor, (size_t)frames);
    win_erb = take(&cursor, (size_t)frames * wf * n_erb);
    win_spec = take(&cursor, (size_t)frames * 2 * wf * df_bins);
    zero_erb = (float *)calloc((size_t)n_erb, sizeof(float));
    zero_spec = (float *)calloc((size_t)2 * df_bins, sizeof(float));
    zero_bins = (float *)calloc((size_t)n_bins, sizeof(float));
    zero_coefs = (float *)calloc((size_t)coefs_n, sizeof(float));
    unit_mask = (float *)malloc((size_t)n_erb * sizeof(float));
    out_re = (float *)calloc((size_t)n_bins, sizeof(float));
    out_im = (float *)calloc((size_t)n_bins, sizeof(float));
    for (c = 0; c < n_erb; ++c) unit_mask[c] = 1.0f;

    state_bytes = dfn2_state_mem_size(&grid);
    io_bytes = dfn2_model_io_mem_size(&grid);
    state_block = aligned_alloc(16, state_bytes);
    io_block = aligned_alloc(16, io_bytes);
    if (!state_block || !io_block) return 2;
    if (dfn2_state_init(&st, state_block, state_bytes, &grid) != 0) return 2;
    dfn2_set_erb_matrices(&st, erb_fwd, erb_inv);
    if (dfn2_model_io_init(&io, io_block, io_bytes, &grid, NULL, 0, NULL, 0) != 0) return 2;
    out = fopen(argv[2], "wb");
    if (!out) return 2;

    /* The caller keeps pushing zero frames after the last real one: that is the right-hand
     * zero padding of the offline model, and it flushes the last MASK + DF source frames. */
    total = frames + mask_la + df_la;
    for (c = 0; c < total; ++c) {
        const int real = c < frames;
        const float *erb = real ? feat_erb + (size_t)c * n_erb : zero_erb;
        const float *spec = real ? feat_spec + (size_t)c * 2 * df_bins : zero_spec;
        const float *re = real ? spec_re + (size_t)c * n_bins : zero_bins;
        const float *im = real ? spec_im + (size_t)c * n_bins : zero_bins;
        int ready = dfn2_model_io_push_features(&io, erb, spec);
        int valid;
        long long frame = -1;
        if (ready != (c >= mask_la ? 1 : 0)) ++ready_bad;
        if (ready == 1) {
            const int t = c - mask_la;
            const int has = t < frames;
            if (first_ready < 0) first_ready = c;
            if (has) {
                const size_t erb_n = (size_t)wf * n_erb;
                const size_t spec_n = (size_t)2 * wf * df_bins;
                ++windows;
                if (memcmp(win_erb + (size_t)t * erb_n, io.erb_window,
                           erb_n * sizeof(float)) != 0 ||
                    memcmp(win_spec + (size_t)t * spec_n, io.spec_window,
                           spec_n * sizeof(float)) != 0)
                    ++window_bad;
            }
            valid = dfn2_enhance_spectrum_stream(
                &st, re, im, 1, has ? head_mask + (size_t)t * n_erb : unit_mask,
                has ? head_coefs + (size_t)t * coefs_n : zero_coefs,
                has ? head_alpha[t] : 0.0f, atten_lim_db, out_re, out_im, &frame);
        } else {
            valid = dfn2_enhance_spectrum_stream(&st, re, im, 0, NULL, NULL, 0.0f, atten_lim_db,
                                        out_re, out_im, &frame);
        }
        if (valid < 0) return 3;
        if (valid == 1) {
            int32_t index = (int32_t)frame;
            fwrite(&index, sizeof index, 1, out);
            fwrite(out_re, sizeof(float), (size_t)n_bins, out);
            fwrite(out_im, sizeof(float), (size_t)n_bins, out);
            ++emitted;
        }
    }
    fclose(out);
    printf("first_ready=%d windows=%d window_bad=%d ready_bad=%d emitted=%d\n",
           first_ready, windows, window_bad, ready_bad, emitted);
    return (ready_bad || window_bad) ? 1 : 0;
}
'''


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _spec_tail(cfg):
    """df_order, the one per-model value left that the C side still takes from the model
    config (the feature normaliser is now a fixed constant inside dfn2_grid.h)."""
    return (dfn_pairs.dfn2_spec_values(cfg)['df_order'],)


@pytest.fixture(scope='module', params=SIMD_MODES, ids=['simd%d' % s for s in SIMD_MODES])
def grid_lib(request, tmp_path_factory):
    """ONE shared library per SIMD mode; every grid is created from its arguments."""
    simd = request.param
    work = tmp_path_factory.mktemp('dfn2_grid_lib_%d' % simd)
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER, encoding='utf-8')
    library = work / ('grid.dylib' if sys.platform == 'darwin' else 'grid.so')
    shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    built = _cc([shared, '-fPIC', *dfn_pairs.house_cflags(simd), '-I', str(DFN2),
                 str(wrapper), str(DFN2_C_DIR / 'dfn2_process.c'),
                 dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(library)])
    assert built.returncode == 0, built.stderr
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_dfn_create.argtypes = [ctypes.c_int] * 7 + [fp, fp]
    lib.parity_dfn_create.restype = ctypes.c_void_p
    lib.parity_dfn_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_dfn_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_dfn_compose.argtypes = [
        ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_float, fp, fp]
    lib.parity_dfn_compose.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize('name', list(GRIDS))
def test_c_features_and_compose_match_python(grid_lib, dfn_py, name):
    lib = grid_lib
    sr, n_fft, n_erb, df_bins = GRIDS[name]
    model_mod, train, _export = dfn_py
    hop, bins = n_fft // 2, n_fft // 2 + 1
    cfg = dfn_pairs.dfn2_grid_config(sr, n_fft, n_erb, df_bins)
    feature_cfg = train.read_feature_config(cfg, sr, hop)
    erb_fb, erb_inv = model_mod._build_erb_fb(n_fft, sr, n_erb)
    fwd = np.ascontiguousarray(erb_fb.T.numpy(), dtype=np.float32)
    inv = np.ascontiguousarray(erb_inv.numpy(), dtype=np.float32)

    rng = np.random.default_rng(41)
    frames = 40
    real = rng.normal(0, 0.15, (frames, bins)).astype(np.float32)
    imag = rng.normal(0, 0.15, (frames, bins)).astype(np.float32)
    spectrum = torch.complex(torch.from_numpy(real.T)[None],
                             torch.from_numpy(imag.T)[None])
    _, py_erb, py_spec, _ = train.extract_dfn2_features(
        spectrum, erb_fb, df_bins, feature_cfg)

    def create():
        state = lib.parity_dfn_create(sr, n_fft, n_erb, df_bins, *PAIR, *_spec_tail(cfg),
                                      _fp(fwd), _fp(inv))
        assert state
        return state

    state = create()
    try:
        c_erb = np.empty((frames, n_erb), np.float32)
        c_spec = np.empty((frames, 2 * df_bins), np.float32)
        for frame in range(frames):
            lib.parity_dfn_features(state, _fp(real[frame]), _fp(imag[frame]),
                                    _fp(c_erb[frame]), _fp(c_spec[frame]))
    finally:
        lib.parity_dfn_destroy(state)
    np.testing.assert_allclose(c_erb, py_erb[0, 0].numpy(), rtol=2e-5, atol=3e-6)
    expected_spec = py_spec[0].permute(1, 0, 2).reshape(frames, 2 * df_bins).numpy()
    np.testing.assert_allclose(c_spec, expected_spec, rtol=2e-5, atol=3e-6)

    model = model_mod.DeepFilterNet2(**train.read_model_config(cfg)).eval()
    mask = torch.sigmoid(torch.randn(1, 1, frames, n_erb))
    coefs = torch.randn(1, frames, df_bins, 10) * 0.1
    alpha = torch.sigmoid(torch.randn(1, frames, 1))
    with torch.no_grad():
        expected = model.compose(spectrum, mask, coefs, alpha)[0]

    state = create()
    c_out = []
    try:
        for frame in range(frames):
            out_re = np.empty(bins, np.float32)
            out_im = np.empty(bins, np.float32)
            valid = lib.parity_dfn_compose(
                state, _fp(real[frame]), _fp(imag[frame]),
                _fp(np.ascontiguousarray(mask[0, 0, frame].numpy())),
                _fp(np.ascontiguousarray(coefs[0, frame].numpy())),
                ctypes.c_float(alpha[0, frame, 0].item()),
                _fp(out_re), _fp(out_im))
            if valid:
                c_out.append(out_re + 1j * out_im)
    finally:
        lib.parity_dfn_destroy(state)
    got = np.stack(c_out, axis=1)
    np.testing.assert_allclose(
        got, expected[:, :frames - model.df_lookahead].numpy(), rtol=4e-5, atol=8e-6)


def test_the_wrapper_refuses_a_grid_the_library_refuses(grid_lib):
    """create() builds the grid from its arguments: a refused grid gives no state."""
    fp = ctypes.POINTER(ctypes.c_float)
    none = ctypes.cast(None, fp)
    tail = _spec_tail(dfn_pairs.dfn2_grid_config(16000, 256, 24, 96))
    assert grid_lib.parity_dfn_create(16000, 960, 24, 96, 1, 1, *tail, none, none) is None
    state = grid_lib.parity_dfn_create(16000, 256, 24, 96, 1, 1, *tail, none, none)
    assert state is not None
    grid_lib.parity_dfn_destroy(state)


@pytest.fixture(scope='module', params=SIMD_MODES, ids=['simd%d' % s for s in SIMD_MODES])
def stream_driver(request, builds):
    simd = request.param
    return builds.program('stream', _STREAM_DRIVER, simd=simd,
                          units=('process', 'model_io'))


@pytest.fixture(scope='module', params=list(GRIDS))
def stream_case(request, tmp_path_factory, dfn_py):
    """The offline model's heads, windows and reference spectrum for one grid; the stream
    input file is shared by every SIMD mode and attenuation limit."""
    name = request.param
    sr, n_fft, n_erb, df_bins = GRIDS[name]
    model_mod, train, export = dfn_py
    pair = PAIR
    cfg = dfn_pairs.dfn2_grid_config(sr, n_fft, n_erb, df_bins)
    torch.manual_seed(7)
    model = model_mod.DeepFilterNet2(**train.read_model_config(cfg)).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0.0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
    model.eval()
    wrapper = export.StatelessDFN2Heads(model).eval()
    frames = 12
    generator = torch.Generator().manual_seed(11)
    erb = torch.randn(1, 1, frames, model.n_erb, generator=generator)
    spec_feat = torch.randn(1, 2, frames, model.df_bins, generator=generator)
    spec = torch.complex(
        torch.randn(1, model.n_bins, frames, generator=generator),
        torch.randn(1, model.n_bins, frames, generator=generator))
    with torch.no_grad():
        reference, _ = model(spec, erb, spec_feat)
        erb_windows = list(export.feature_windows(erb, pair[0]))
        spec_windows = list(export.feature_windows(spec_feat, pair[0]))
        state = wrapper.initial_inputs()[2:]
        mask, coefs, alpha = [], [], []
        for erb_window, spec_window in zip(erb_windows, spec_windows):
            output = wrapper(erb_window, spec_window, *state)
            mask.append(output[0].numpy())
            coefs.append(output[1].numpy())
            alpha.append(output[2].numpy())
            state = output[3:]
    work = tmp_path_factory.mktemp('dfn2_stream_' + name)
    source = work / 'stream_in.bin'
    dfn_pairs.dfn2_write_stream_input(
        source, model, frames, erb, spec_feat, spec, (mask, coefs, alpha),
        (erb_windows, spec_windows))
    return dict(name=name, grid=GRIDS[name], pair=pair, frames=frames, n_bins=model.n_bins,
                source=source, spec=spec[0].numpy(), reference=reference[0].numpy(),
                work=work)


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
def test_c_host_stream_reproduces_the_offline_model(stream_driver, stream_case, atten_lim):
    case = stream_case
    sr, n_fft, n_erb, df_bins = case['grid']
    frames, n_bins = case['frames'], case['n_bins']
    result = case['work'] / ('stream_out_%s_%g.bin' % (Path(stream_driver).name,
                                                          atten_lim))
    tail = _spec_tail(dfn_pairs.dfn2_grid_config(sr, n_fft, n_erb, df_bins))
    done = subprocess.run(
        [str(stream_driver), str(case['source']), str(result), str(atten_lim),
         str(sr), str(n_fft), str(n_erb), str(df_bins), str(case['pair'][0]),
         str(case['pair'][1]), *map(repr, tail)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert 'first_ready=%d ' % case['pair'][0] in done.stdout, done.stdout
    assert 'window_bad=0 ready_bad=0 emitted=%d' % frames in done.stdout, done.stdout

    record = 4 + 2 * 4 * n_bins
    raw = result.read_bytes()
    assert len(raw) == frames * record
    got = np.zeros((n_bins, frames), dtype=np.complex64)
    for index in range(frames):
        chunk = raw[index * record:(index + 1) * record]
        assert int(np.frombuffer(chunk[:4], '<i4')[0]) == index
        values = np.frombuffer(chunk[4:], '<f4')
        got[:, index] = values[:n_bins] + 1j * values[n_bins:]
    want = case['reference']
    if atten_lim:
        mix = 10 ** (-abs(atten_lim) / 20)
        want = case['spec'] * mix + want * (1 - mix)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    assert float(np.abs(want - case['spec']).max()) > 1e-2


def test_the_stream_driver_refuses_a_grid_the_library_refuses(builds, stream_case):
    exe = builds.program('stream', _STREAM_DRIVER, simd=1, units=('process', 'model_io'))
    tail = _spec_tail(dfn_pairs.dfn2_grid_config(16000, 256, 24, 96))
    done = subprocess.run([str(exe), str(stream_case['source']),
                           str(stream_case['work'] / 'unused.bin'), '0', '16000', '960',
                           '24', '96', '1', '1', *map(repr, tail)],
                          capture_output=True, text=True)
    assert done.returncode == 2
