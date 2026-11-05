"""DFN2Grid (dfn2_grid.h): a grid is a run-time value the application builds.

The library holds no grid. ``dfn2_grid_make()`` takes the six values a trained model fixes
(sample rate, fft size, ERB bands, deep-filter bins, the lookahead pair) and derives the rest;
``dfn2_grid_validate()`` re-derives and compares; ``dfn2_norm_alpha()`` is train.py's
``make_norm_alpha`` in C. These tests pin:

  make / validate  every refusal, in the order the checks run, the grid left untouched by a
                   refusal; the derived fields of the 16, 32 and 48 kHz rows and of two
                   grids no preset knows; validate refusing a hand-edited derived field
  capacity         a grid above a build capacity is refused (ERR_CAPACITY) and the same
                   arguments are accepted once the capacity is raised: the refusal is a
                   capacity check, not a limit of the arithmetic
  mask-only        df_bins up to n_bins, no deep-filter lookahead, ring = mask + 1
  make = validate  what make accepts, validate accepts, over a sweep of sample rates (1 to INT_MAX):
                   a rate whose norm alpha, as float32, is not inside (0, 1) is ERR_SAMPLE_RATE
  extremes         n_erb of 1 << 30 and 0x7ffffffc (and the largest and smallest ints in every
                   field) are refused with no signed overflow, under -fsanitize=undefined; the
                   gate is shown to fail on the old `2 * n_erb > n_bins`
  framing          dfn2_framing_config leaves an all-zero config for a refused grid, and the
                   framing objects ask for 0 bytes for it
  tap count        -DDFN2_DF_ORDER=<anything> stops the compile of dfn2_grid.h (C11, gnu99)
  alpha            ``dfn2_norm_alpha(sr, hop, tau)`` equals ``make_norm_alpha`` of train.py as
                   float32, bit for bit, over a sweep of sample rates, hops and taus,
                   including taus placed one ulp either side of a rounding boundary; the
                   three shipped alphas are the literals the build-time grid carried
  parity           the C features / compose / stream host equal the Python model on every
                   grid, from ONE binary that takes the grid from its arguments

The C programs are built a few at a time and memoised: the probe once per capacity variant,
the parity wrapper and the stream driver once per SIMD mode.
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
    else if (!strcmp(name, "mask_lookahead")) g->mask_lookahead = (int)iv;
    else if (!strcmp(name, "df_lookahead")) g->df_lookahead = (int)iv;
    else if (!strcmp(name, "df_history")) g->df_history = (int)iv;
    else if (!strcmp(name, "ring")) g->ring = (int)iv;
    else if (!strcmp(name, "erb_norm_alpha")) g->erb_norm_alpha = float_of(fv);
    else if (!strcmp(name, "spec_norm_alpha")) g->spec_norm_alpha = float_of(fv);
    else if (!strcmp(name, "stft_scale")) g->stft_scale = float_of(fv);
    else if (!strcmp(name, "stft_unscale")) g->stft_unscale = float_of(fv);
    else return -1;
    return 0;
}

int main(void) {
    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        char verb[16] = "";
        int a, b, c, d, e, f;
        if (sscanf(line, "%15s", verb) != 1) continue;
        if (!strcmp(verb, "make")) {
            DFN2Grid g, before;
            int st, vst = 0, untouched = -1;
            if (sscanf(line, "%*s %d %d %d %d %d %d", &a, &b, &c, &d, &e, &f) != 6) return 2;
            memset(&g, 0xA5, sizeof g);
            before = g;
            st = dfn2_grid_make(&g, a, b, c, d, e, f);
            if (st != 0) untouched = memcmp(&g, &before, sizeof g) == 0;
            else vst = dfn2_grid_validate(&g);
            if (st != 0) memset(&g, 0, sizeof g);
            printf("make %d %d %d %d %d %d %d %08x %08x %08x %08x %d\n", st, vst, g.hop,
                   g.n_bins, g.df_history, g.ring, g.n_erb == 0 ? 0 : 1,
                   bits_of(g.erb_norm_alpha), bits_of(g.spec_norm_alpha),
                   bits_of(g.stft_scale), bits_of(g.stft_unscale), untouched);
        } else if (!strcmp(verb, "edit")) {
            DFN2Grid g;
            char field[32], value[32];
            int st;
            if (sscanf(line, "%*s %d %d %d %d %d %d %31s %31s", &a, &b, &c, &d, &e, &f, field,
                       value) != 8) return 2;
            st = dfn2_grid_make(&g, a, b, c, d, e, f);
            if (st != 0 || set_field(&g, field, value) != 0) {
                printf("edit %d -99\n", st);
            } else {
                printf("edit 0 %d\n", dfn2_grid_validate(&g));
            }
        } else if (!strcmp(verb, "null")) {
            printf("null %d %d\n", dfn2_grid_make(NULL, 16000, 256, 24, 96, 1, 1),
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
            printf("alpha %08x\n", bits_of(dfn2_norm_alpha(sr, hop, tau)));
        } else if (!strcmp(verb, "framing")) {
            /* framing <mode>: dfn2_framing_config() on a config that holds 0xA5 bytes, for a
             * 16 kHz grid that is valid (valid, borrowed) or invalid in one way. Prints: the
             * config is all zero bytes, the bytes ac_stft/ac_istft_get_mem_size ask for it
             * (-1: not asked), fft_size, the two scales, and whether fft / window were taken
             * over. The pointers handed in are never dereferenced: they stand for objects the
             * caller owns, and only a valid grid may store them. */
            char mode[32];
            DFN2Grid g, *grid = &g;
            AcStftConfig cfg;
            FftHandle *fft = (FftHandle *)(uintptr_t)64;
            const float *window = (const float *)(uintptr_t)64;
            const unsigned char *raw = (const unsigned char *)&cfg;
            size_t i, stft_bytes = (size_t)-1, istft_bytes = (size_t)-1;
            int zero = 1;
            if (sscanf(line, "%*s %31s", mode) != 1) return 2;
            if (dfn2_grid_make(&g, 16000, 256, 24, 96, 1, 1) != 0) return 2;
            if (!strcmp(mode, "null")) grid = NULL;
            else if (!strcmp(mode, "garbage")) memset(&g, 0xA5, sizeof g);
            else if (!strcmp(mode, "zeroed")) memset(&g, 0, sizeof g);
            else if (!strcmp(mode, "ring")) g.ring += 1;
            else if (!strcmp(mode, "hop")) g.hop += 1;
            else if (!strcmp(mode, "scale")) g.stft_scale *= 2.0f;
            else if (!strcmp(mode, "rate")) g.sample_rate = 0;
            else if (!strcmp(mode, "fft")) g.n_fft = 960;
            else if (!strcmp(mode, "alpha")) g.erb_norm_alpha = 1.0f;
            else if (!strcmp(mode, "valid")) { fft = NULL; window = NULL; }
            else if (strcmp(mode, "borrowed")) return 2;
            memset(&cfg, 0xA5, sizeof cfg);
            dfn2_framing_config(&cfg, grid, fft, window);
            for (i = 0; i < sizeof cfg; ++i) zero &= raw[i] == 0;
            if (strcmp(mode, "borrowed")) {
                stft_bytes = ac_stft_get_mem_size(&cfg);
                istft_bytes = ac_istft_get_mem_size(&cfg);
            }
            printf("framing %d %lld %lld %d %08x %08x %d %d\n", zero, (long long)stft_bytes,
                   (long long)istft_bytes, cfg.fft_size, bits_of(cfg.spec_scale),
                   bits_of(cfg.ispec_scale), cfg.fft == fft && fft != NULL,
                   cfg.window == window && window != NULL);
        }
    }
    return 0;
}
'''


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

    def make(self, grid, mask_lookahead, df_lookahead):
        """dfn2_grid_make(sr, n_fft, n_erb, df_bins, mask, df) -> dict."""
        return self.make_all([(*grid, mask_lookahead, df_lookahead)])[0]

    def make_all(self, cases):
        rows = []
        for line in self.run(['make %d %d %d %d %d %d' % tuple(c)
                              for c in cases]):
            fields = line.split()
            assert fields[0] == 'make', line
            (status, vstatus, hop, n_bins, df_history, ring, _built,
             erb_alpha, spec_alpha, scale, unscale, untouched) = fields[1:]
            rows.append(dict(
                status=int(status), validate=int(vstatus), hop=int(hop),
                n_bins=int(n_bins), df_history=int(df_history), ring=int(ring),
                erb_alpha=int(erb_alpha, 16), spec_alpha=int(spec_alpha, 16),
                scale=int(scale, 16), unscale=int(unscale, 16),
                untouched=int(untouched)))
        return rows

    def edit_all(self, edits):
        """[(case, field, value)] -> [(make status, validate status)]."""
        out = []
        for line in self.run(['edit %d %d %d %d %d %d %s %s' % (*case, field, value)
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


# Capacity variants of the probe. 'wide' raises every capacity the default grids can reach.
_WIDE = ['-DDFN2_MAX_N_FFT=8192', '-DDFN2_MAX_N_ERB=48', '-DDFN2_MAX_DF_BINS=129',
         '-DDFN2_MAX_DF_RING=7']
_VARIANTS = {
    'full': [],
    'full_wide': _WIDE,
    'mask': ['-DDFN2_MASK_ONLY'],
    'mask_wide': ['-DDFN2_MASK_ONLY', '-DDFN2_MAX_N_FFT=8192', '-DDFN2_MAX_N_ERB=48',
                  '-DDFN2_MAX_DF_BINS=129', '-DDFN2_MAX_DF_RING=3'],
}


@pytest.fixture(scope='module')
def probes(builds):
    specs = [('probe_' + name, _PROBE, flags, 1, ('process',))
             for name, flags in _VARIANTS.items()]
    return dict(zip(_VARIANTS, map(_Probe, builds.programs(specs))))


# Status codes (dfn2_grid.h).
OK, ERR_NULL, ERR_SR, ERR_FFT, ERR_ERB, ERR_DF, ERR_MASK, ERR_DFLA, ERR_CAP, ERR_INC = (
    0, -1, -2, -3, -4, -5, -6, -7, -8, -9)

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
    assert got['df_history'] == DF_ORDER - pair[1] - 1
    assert got['ring'] == max(DF_ORDER, pair[0] + pair[1] + 1)
    assert got['erb_alpha'] == alpha and got['spec_alpha'] == alpha, (
        'alpha is exp(-(hop/sr)/tau) rounded to 3 decimals as the trainer derives it: %r'
        % (train.make_norm_alpha(sr, hop, 1.0),))
    assert (got['scale'], got['unscale']) == (scale, unscale)
    # train.py's analysis_scale is the same float32 (1 / sqrt(n_fft) at win_len == n_fft).
    assert got['scale'] == _expect_f32(train.analysis_scale(n_fft, n_fft, hop))


def test_the_shipped_rows_carry_the_literals_of_the_build_time_grid(probes):
    """The 48 / 32 / 16 kHz rows used to be 0.989f / 0.992f / 0.992f in the header."""
    want = {'48k': 0.989, '32k': 0.992, '16k': 0.992}
    for name, literal in want.items():
        got = probes['full'].make(GRIDS[name], 1, 1)
        expected = _f32_bits(float(np.float32(literal)))
        assert got['erb_alpha'] == got['spec_alpha'] == expected, (name, got)
    # Not a vacuous literal: the neighbouring three-decimal values are other floats.
    assert _f32_bits(float(np.float32(0.989))) != _f32_bits(float(np.float32(0.99)))


def test_scales_are_exact_float32_for_every_legal_fft_size(probes):
    cases = [(16000, n, 4, 2, 0, 0) for n in (16, 32, 64, 128, 256, 512, 1024)]
    for case, got in zip(cases, probes['full'].make_all(cases)):
        assert got['status'] == OK, (case, got)
        assert (got['scale'], got['unscale']) == _scales(case[1]), case
    # Beyond the default capacity: the wide build covers 2048 .. 8192.
    cases = [(16000, n, 4, 2, 0, 0) for n in (2048, 4096, 8192)]
    for case, got in zip(cases, probes['full_wide'].make_all(cases)):
        assert got['status'] == OK, (case, got)
        assert (got['scale'], got['unscale']) == _scales(case[1]), case


# ---------------------------------------------------------------------------
# every refusal
# ---------------------------------------------------------------------------

# (case, expected status), full variant with the default capacities.
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
    # lookahead pair
    ((16000, 256, 24, 96, -1, 1), ERR_MASK),
    ((16000, 256, 24, 96, 3, 1), ERR_MASK),
    ((16000, 256, 24, 96, 1, -1), ERR_DFLA),
    ((16000, 256, 24, 96, 1, 5), ERR_DFLA),
    # capacity (the default build: 1024 / 32 / 96 / ring 5)
    ((48000, 2048, 32, 96, 1, 1), ERR_CAP),
    ((48000, 8192, 32, 96, 1, 1), ERR_CAP),
    ((48000, 1024, 36, 96, 1, 1), ERR_CAP),
    ((48000, 1024, 32, 97, 1, 1), ERR_CAP),
    ((48000, 1024, 32, 128, 1, 1), ERR_CAP),     # df_bins < n_bins but above the capacity
    ((48000, 1024, 32, 96, 2, 3), ERR_CAP),      # ring 6
    ((48000, 1024, 32, 96, 2, 4), ERR_CAP),      # ring 7
    ((48000, 1024, 32, 96, 1, 4), ERR_CAP),      # ring 6
    # the order of the checks: the first failing one is reported
    ((0, 3, 3, 1, 9, 9), ERR_SR),
    ((16000, 3, 3, 1, 9, 9), ERR_FFT),
    ((16000, 256, 3, 1, 9, 9), ERR_ERB),
    ((16000, 256, 24, 1, 9, 9), ERR_DF),
    ((16000, 256, 24, 96, 9, 9), ERR_MASK),
    ((16000, 256, 24, 96, 1, 9), ERR_DFLA),
    ((48000, 4096, 32, 96, 1, 9), ERR_DFLA),     # a bad lookahead before the capacity
    ((48000, 4096, 32, 96, 1, 1), ERR_CAP),
]


def test_every_refusal_reports_its_status_and_leaves_the_grid_alone(probes):
    cases = [case for case, _status in _REFUSALS]
    for (case, want), got in zip(_REFUSALS, probes['full'].make_all(cases)):
        assert got['status'] == want, (case, got)
        assert got['untouched'] == 1, ('a refused grid was written', case)


def test_the_refusals_are_not_vacuous(probes):
    """One step inside each boundary, and the boundary itself, are accepted."""
    accepted = [
        (16000, 16, 4, 2, 0, 0),       # smallest legal frame, band count, df_bins
        (16000, 16, 4, 8, 0, 0),       # df_bins = n_bins - 1
        (16000, 64, 16, 32, 1, 1),     # 2 * n_erb = 32 <= 33; df_bins = n_bins - 1
        (16000, 256, 4, 2, 2, 2),      # lookahead 2 / 2
        (16000, 256, 24, 96, 0, 0),
        (16000, 256, 24, 96, 2, 0),
        (16000, 256, 24, 96, 0, 4),
        (48000, 1024, 32, 96, 2, 2),   # ring 5, every capacity exactly met
        (16000, 256, 16, 96, 1, 1),
    ]
    for case, got in zip(accepted, probes['full'].make_all(accepted)):
        assert got['status'] == OK and got['validate'] == OK, (case, got)
    # The refused capacity cases are accepted once the capacity is raised: the grids are
    # fine, the build was not sized for them.
    raised = [case for case, status in _REFUSALS if status == ERR_CAP]
    assert len(raised) == 9
    for case, got in zip(raised, probes['full_wide'].make_all(raised)):
        assert got['status'] == OK and got['validate'] == OK, (case, got)
    # ... and the wide build still refuses what is not a capacity matter.
    for case, want in _REFUSALS:
        if want not in (ERR_CAP,):
            got = probes['full_wide'].make(case[:4], *case[4:])
            assert got['status'] == want, (case, got)


def test_a_null_grid_is_refused(probes):
    (line,) = probes['full'].run(['null'])
    assert line == 'null %d %d' % (ERR_NULL, ERR_NULL)


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
    """What dfn2_grid_make must return for an otherwise legal grid: the sample rate is refused
    exactly when the trainer's make_norm_alpha, as float32, is not strictly inside (0, 1)."""
    alpha = np.float32(train.make_norm_alpha(sample_rate, n_fft // 2, 1.0))
    return OK if 0.0 < alpha < 1.0 else ERR_SR


def test_make_ok_implies_validate_ok_over_a_sweep_of_sample_rates(probes, dfn_py):
    """The review finding: dfn2_grid_make derived an alpha of 0 or 1 and returned OK for a grid
    that dfn2_grid_validate (and so every entry point) then refuses. Now make refuses it with
    ERR_SAMPLE_RATE, and the two agree on every rate: OK from make is OK from validate, and a
    refusal is exactly an alpha outside (0, 1) as the trainer computes it, in float32."""
    _model, train, _export = dfn_py
    cases = [(sr, n_fft, *_RATE_SWEEP_GRIDS[n_fft], 1, 1)
             for sr in _sweep_rates() for n_fft in _RATE_SWEEP_GRIDS]
    assert len(cases) > 700
    outcomes = {OK: 0, ERR_SR: 0}
    for case, got in zip(cases, probes['full'].make_all(cases)):
        want = _alpha_status(train, case[0], case[1])
        assert got['status'] == want, (case, got, want)
        if want == OK:
            assert got['validate'] == OK, ('make accepted a grid validate refuses', case, got)
        else:
            assert got['untouched'] == 1, ('a refused grid was written', case)
        outcomes[want] += 1
    # Not vacuous: both outcomes occur (low rates at a large hop and the top of the range).
    assert outcomes[OK] > 400 and outcomes[ERR_SR] > 20, outcomes


def test_the_rates_the_review_named_are_decided_by_the_alpha(probes, dfn_py):
    """sr 1, 16, 67 and INT_MAX at n_fft 16 / 256 / 1024: refused (ERR_SAMPLE_RATE) where the
    rounded alpha is 0 (a rate far below the hop rate) or reaches 1.0f (INT_MAX with a short
    hop), accepted otherwise. The expectation is written out, not only derived."""
    _model, train, _export = dfn_py
    want = {
        (1, 16): ERR_SR, (1, 256): ERR_SR, (1, 1024): ERR_SR,           # exp(-hop) rounds to 0
        (16, 16): OK, (16, 256): ERR_SR, (16, 1024): ERR_SR,
        (67, 16): OK, (67, 256): OK, (67, 1024): ERR_SR,                # exp(-512/67) < 5e-4
        (INT_MAX, 16): ERR_SR,                                          # 0.99999999627 -> 1.0f
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
    return _Probe(builds.program('probe_ubsan', _PROBE, _UBSAN, 1, ('process',)))


def test_the_extremes_of_make_and_validate_run_clean_under_ubsan(ubsan_probe, probes, dfn_py):
    """n_erb = 1 << 30 and 0x7ffffffc (and every refusal of the table above, the sample-rate
    sweep and the largest and smallest ints in every field) give their status, with no signed
    overflow: `2 * n_erb > n_bins` overflowed for n_erb >= 2^30."""
    _model, train, _export = dfn_py
    cases = [case for case, _status in _REFUSALS]
    wants = [status for _case, status in _REFUSALS]
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
              for field in ('hop', 'n_bins', 'df_history', 'ring')
              for value in (INT_MAX, -(1 << 31))]
    for (_case, field, value), got, ref in zip(edits, ubsan_probe.edit_all(edits),
                                               probes['full'].edit_all(edits)):
        assert got == ref and got[0] == OK, (field, value, got, ref)
        if field != 'sample_rate':       # (a rate edited on a made grid can stay valid: the
            assert got[1] != OK, (field, value, got)         # alphas are the model's)
    # the alpha helper on awkward arguments: zero, infinity, subnormal and huge taus
    lines = ubsan_probe.run(['alpha %d %d %r' % c for c in [
        (16000, 128, 1e-300), (16000, 128, 1e300), (16000, 128, float('inf')),
        (16000, 128, float('nan')), (INT_MAX, 8, 1.0), (1, 4096, 1.0), (1, INT_MAX, 1.0),
        (INT_MAX, INT_MAX, 1.0), (16000, -128, 1.0), (-16000, 128, 1.0)]])
    assert len(lines) == 10


def test_the_ubsan_gate_can_fail(builds, ubsan_probe, tmp_path):
    """The same cases against a copy of dfn2_process.c with the old `2 * n_erb > n_bins`
    abort under the sanitizer, naming the signed overflow; the shipped file does not."""
    source = (DFN2 / 'dfn2_process.c').read_text(encoding='utf-8')
    fixed = 'n_erb > n_bins / 2'
    assert source.count(fixed) == 1, 'the overflow-free form is gone: update this mutation'
    mutated = tmp_path / 'dfn2_process.c'
    mutated.write_text(source.replace(fixed, '2 * n_erb > n_bins'), encoding='utf-8')
    probe_source = tmp_path / 'probe.c'
    probe_source.write_text(_PROBE, encoding='utf-8')
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
# dfn2_framing_config: a grid the library refuses gives no framing
# ---------------------------------------------------------------------------

_FRAMING_REFUSED = ['null', 'garbage', 'zeroed', 'ring', 'hop', 'scale', 'rate', 'fft', 'alpha']


def _framing(probe, modes):
    rows = {}
    for mode, line in zip(modes, probe.run(['framing ' + m for m in modes])):
        verb, *fields = line.split()
        assert verb == 'framing', line
        zero, stft, istft, fft_size, spec, ispec, fft_taken, window_taken = fields
        rows[mode] = dict(zero=int(zero), stft=int(stft), istft=int(istft),
                          fft_size=int(fft_size), spec=int(spec, 16), ispec=int(ispec, 16),
                          fft=int(fft_taken), window=int(window_taken))
    return rows


def test_framing_config_of_a_refused_grid_is_zero_and_sizes_nothing(probes):
    """A NULL grid, an uninitialised one and a made grid with one derived field edited
    (ring, hop, scale), one given field gone bad (rate, frame size) or an alpha outside (0, 1)
    all leave an all-zero AcStftConfig behind -- even over a config that held other bytes, and
    without taking over the FFT handle or window passed in -- and the framing objects ask for
    0 bytes for it, so a caller that ignores the missing status cannot build them."""
    rows = _framing(probes['full'], _FRAMING_REFUSED)
    for mode, row in rows.items():
        assert row == dict(zero=1, stft=0, istft=0, fft_size=0, spec=0, ispec=0, fft=0,
                           window=0), (mode, row)


def test_framing_config_of_a_valid_grid_describes_its_framing(probes):
    """The control of the test above: the 16 kHz grid gives a 256-point config with the grid's
    two scales, takes over the FFT handle and window it is given, and sizes both objects."""
    rows = _framing(probes['full'], ['valid', 'borrowed'])
    scale, unscale = _scales(256)
    valid, borrowed = rows['valid'], rows['borrowed']
    assert (valid['zero'], valid['fft_size'], valid['spec'], valid['ispec']) == (
        0, 256, scale, unscale), valid
    assert valid['stft'] > 0 and valid['istft'] > 0, valid
    assert (valid['fft'], valid['window']) == (0, 0)             # NULL: the objects carve their own
    assert (borrowed['zero'], borrowed['fft_size'], borrowed['spec'], borrowed['ispec']) == (
        0, 256, scale, unscale), borrowed
    assert (borrowed['fft'], borrowed['window']) == (1, 1)       # the caller's, taken over


# ---------------------------------------------------------------------------
# the tap count is not a grid choice
# ---------------------------------------------------------------------------

_ORDER_TU = '#include "dfn2_grid.h"\n'
_ORDER_MESSAGE = 'DFN2_DF_ORDER is fixed at 5'


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
@pytest.mark.parametrize('order', [3, 4, 5, 7])
def test_defining_the_deep_filter_order_stops_the_compile(tmp_path, std, order):
    """The graph's pathway window, the state layout and every ring depth are built for five
    taps, so dfn2_grid.h refuses any definition of DFN2_DF_ORDER -- another order, and the shipped
    value too, because the guard is `#ifdef`, not a comparison -- in C11 and in gnu99 (the
    integration build). A translation unit that only includes the header is enough."""
    source = tmp_path / 'order.c'
    source.write_text(_ORDER_TU, encoding='utf-8')
    done = _cc(['-std=' + std, '-fsyntax-only', '-I', str(DFN2), '-DDFN2_DF_ORDER=%d' % order,
                str(source)])
    assert done.returncode != 0, 'DFN2_DF_ORDER=%d was accepted' % order
    assert _ORDER_MESSAGE in done.stderr, done.stderr


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
def test_the_deep_filter_order_guard_is_not_vacuous(tmp_path, std):
    """The same translation unit without the definition compiles, under -Wall -Wextra
    -Wpedantic -Werror, and the library's order is five."""
    source = tmp_path / 'order.c'
    source.write_text(_ORDER_TU + 'typedef char order_is_five[DFN2_DF_ORDER == 5 ? 1 : -1];\n',
                      encoding='utf-8')
    done = _cc(['-std=' + std, '-fsyntax-only', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
                '-I', str(DFN2), str(source)])
    assert done.returncode == 0, done.stderr


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
        scale, unscale = _scales(n_fft)
        for field, value in [
                ('hop', hop + 1), ('hop', hop - 1), ('hop', n_fft), ('hop', 0),
                ('n_bins', n_bins + 1), ('n_bins', n_bins - 1), ('n_bins', 0),
                ('df_history', 0), ('df_history', 4), ('df_history', 2),
                ('ring', 4), ('ring', 6), ('ring', 7), ('ring', 0),
                ('stft_scale', _hex(scale + 1)), ('stft_scale', _hex(scale - 1)),
                ('stft_scale', _hex(_f32_bits(1.0))), ('stft_scale', _hex(unscale)),
                ('stft_scale', '00000000'), ('stft_scale', '7fc00000'),
                ('stft_unscale', _hex(unscale + 1)), ('stft_unscale', _hex(unscale - 1)),
                ('stft_unscale', _hex(scale)), ('stft_unscale', '7f800000'),
                ('stft_unscale', '00000000'),
                # the alphas: finite and strictly inside (0, 1)
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
        (case48, 'n_fft', 2048, ERR_CAP), (case48, 'n_erb', 36, ERR_CAP),
        # A legal fft size on a grid derived for another one: the derived fields are stale.
        (case48, 'n_fft', 512, ERR_INC), (case16, 'n_fft', 512, ERR_INC),
        (case16, 'n_fft', 128, ERR_DF),          # 96 bins no longer fit below 65
    ]
    for (case, field, value, want), (make_status, status) in zip(
            refused, probes['full'].edit_all([r[:3] for r in refused])):
        assert (make_status, status) == (OK, want), (case, field, value, status)


def test_validate_accepts_a_retuned_alpha_and_other_valid_edits(probes):
    """The alphas are the model's: a tau other than 1.0 is a legal grid. A given field moved
    to another legal value is consistent as long as the derived fields still follow."""
    case = (*GRIDS['48k'], 1, 1)
    ok_alphas = [_hex(_f32_bits(float(np.float32(a))))
                 for a in (0.5, 0.979, 0.989, 0.999, 0.9999999)]
    ok_alphas += ['3f7fffff', '00000001', '00800000']   # just below 1.0; denormal; smallest normal
    edits = []
    for alpha in ok_alphas:
        edits += [(case, 'erb_norm_alpha', alpha), (case, 'spec_norm_alpha', alpha)]
    edits += [(case, 'n_erb', 28), (case, 'df_bins', 64), (case, 'df_bins', 2),
              (case, 'mask_lookahead', 0), (case, 'mask_lookahead', 2),
              (case, 'sample_rate', 44100)]
    for (_case, field, value), (make_status, status) in zip(
            edits, probes['full'].edit_all(edits)):
        assert (make_status, status) == (OK, OK), (field, value, status)
    # A lookahead edited on a grid whose derived fields were made for the other one.
    (_make, status), = probes['full'].edit_all([((*GRIDS['16k'], 0, 2), 'df_lookahead', 1)])
    assert status == ERR_INC             # df_history 2 was left behind (3 now)
    # ... and one that deepens the ring (a build whose ring capacity allows it).
    (_make, status), = probes['full_wide'].edit_all(
        [((*GRIDS['16k'], 2, 2), 'df_lookahead', 3)])
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
        ((16000, 256, 24, 129, 1, 0), ERR_CAP, None),   # df_bins <= n_bins, above the capacity
        ((16000, 256, 24, 96, 2, 0), ERR_CAP, None),    # ring 3 over the mask-only capacity 2
        ((16000, 256, 22, 96, 1, 0), ERR_ERB, None),
    ]
    for (case, want, ring), got in zip(cases, mask.make_all([c for c, _s, _r in cases])):
        assert got['status'] == want, (case, got)
        if want == OK:
            assert got['validate'] == OK and got['ring'] == ring, (case, got)
            # df_history is still DF_ORDER - df_lookahead - 1: all five taps are past.
            assert got['df_history'] == DF_ORDER - 1
    # With the capacities raised the refused capacity cases are fine.
    for case in [(16000, 256, 24, 129, 1, 0), (16000, 256, 24, 96, 2, 0)]:
        got = probes['mask_wide'].make(case[:4], *case[4:])
        assert got['status'] == OK and got['validate'] == OK, (case, got)
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
    lines = probes['full'].run(['text %d' % s for s in range(1, -12, -1)])
    texts = {}
    for line in lines:
        _verb, status, text = line.split(' ', 2)
        texts[int(status)] = text
    for status in range(OK, ERR_INC - 1, -1):
        assert texts[status] and texts[status] != '(null)', status
    assert len({texts[s] for s in range(OK, ERR_INC - 1, -1)}) == 10, 'texts must differ'
    for status, word in [(OK, 'ok'), (ERR_NULL, 'null'), (ERR_SR, 'sample rate'),
                         (ERR_FFT, 'n_fft'), (ERR_ERB, 'n_erb'), (ERR_DF, 'df_bins'),
                         (ERR_MASK, 'mask_lookahead'), (ERR_DFLA, 'df_lookahead'),
                         (ERR_CAP, 'capacity'), (ERR_INC, 'derived')]:
        assert word in texts[status].lower(), (status, texts[status])
    # a rate is refused for its alpha as well as for being non-positive
    assert 'alpha' in texts[ERR_SR].lower(), texts[ERR_SR]
    for status in (1, -10, -11):                 # unknown: still a non-NULL string
        assert texts[status] and texts[status] != '(null)'
        assert texts[status] not in {texts[s] for s in range(OK, ERR_INC - 1, -1)}


# ---------------------------------------------------------------------------
# dfn2_norm_alpha is train.py's make_norm_alpha as float32, bit for bit
# ---------------------------------------------------------------------------

_SAMPLE_RATES = sorted({*range(8000, 96001, 1000), 11025, 22050, 44100, 88200, 192000})
_HOPS = [*(1 << k for k in range(3, 13)), 100, 160, 240, 441, 480, 960, 1000, 1920]
_SHIPPED = [(48000, 512), (32000, 256), (16000, 128)]


def _tau_boundary_cases():
    """taus placed so that exp(-(hop / sr) / tau) falls within three ulp of a rounding
    boundary (k + 1/2) / 10**digits, for the three- and four-digit rounding the trainer
    uses: the half-to-even tie handling decides these."""
    cases = []
    for sr, hop in _SHIPPED + [(96000, 8), (44100, 441)]:
        for digits, ks in ((3, range(900, 1000)), (4, range(9990, 10000)),
                           (5, range(99990, 100000, 2))):
            for k in ks:
                target = (k + 0.5) / 10 ** digits
                tau = -(hop / sr) / math.log(target)
                lo = hi = tau
                cases.append((sr, hop, tau))
                for _ in range(3):
                    lo, hi = math.nextafter(lo, 0.0), math.nextafter(hi, 1e9)
                    cases += [(sr, hop, lo), (sr, hop, hi)]
    return cases


def _alpha_cases():
    cases = [(sr, hop, 1.0) for sr in _SAMPLE_RATES for hop in _HOPS]
    taus = [0.25 + 0.0025 * i for i in range(1501)]            # 0.25 .. 4.0
    cases += [(sr, hop, tau) for sr, hop in _SHIPPED for tau in taus]
    cases += [(sr, hop, tau) for sr, hop in _SHIPPED
              for tau in (0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    cases += _tau_boundary_cases()
    return cases


def test_norm_alpha_equals_the_trainers_make_norm_alpha_bit_for_bit(probes, dfn_py):
    _model, train, _export = dfn_py
    cases = _alpha_cases()
    assert len(cases) > 9000
    lines = probes['full'].run(['alpha %d %d %r' % case for case in cases])
    bad = []
    for (sr, hop, tau), line in zip(cases, lines):
        got = int(line.split()[1], 16)
        want = _f32_bits(train.make_norm_alpha(sr, hop, tau))
        if got != want:
            bad.append((sr, hop, tau, '%08x' % got, '%08x' % want,
                        train.make_norm_alpha(sr, hop, tau)))
    assert not bad, '%d of %d differ, first: %r' % (len(bad), len(cases), bad[:8])


def _naive_make_norm_alpha(sr, hop, tau):
    """make_norm_alpha with the rounding done in double arithmetic (floor(x * 10**d + 1/2)):
    the shortcut a C port is tempted by."""
    exact = math.exp(-(hop / sr) / tau)
    digits, alpha = 3, 1.0
    while alpha >= 1.0:
        alpha = math.floor(exact * 10 ** digits + 0.5) / 10 ** digits
        digits += 1
    return alpha


def test_norm_alpha_sweep_is_not_vacuous(probes, dfn_py):
    """The sweep reaches the 4- and 5-digit branch, and its boundary taus are the cases where
    a rounding done in double arithmetic goes wrong: a naive port would be caught."""
    _model, train, _export = dfn_py
    values = {train.make_norm_alpha(sr, hop, 1.0) for sr in _SAMPLE_RATES for hop in _HOPS}
    digits = {len(repr(v).split('.')[1]) for v in values}
    assert max(digits) >= 5 and min(digits) <= 3, digits
    boundary = _tau_boundary_cases()
    assert len({train.make_norm_alpha(sr, hop, tau) for sr, hop, tau in boundary}) > 100
    naive_wrong = sum(_naive_make_norm_alpha(sr, hop, tau) != train.make_norm_alpha(sr, hop, tau)
                      for sr, hop, tau in boundary)
    assert naive_wrong > 0, 'the boundary taus do not tell a naive rounding apart'
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

/* The grid comes from the caller: one library serves every grid. */
void *parity_dfn_create(int sr, int n_fft, int n_erb, int df_bins, int mask_lookahead,
                        int df_lookahead, const float *fwd, const float *inv) {
    DFN2Grid grid;
    DFN2State *state;
    if (dfn2_grid_make(&grid, sr, n_fft, n_erb, df_bins, mask_lookahead, df_lookahead) != 0)
        return NULL;
    state = (DFN2State *)calloc(1, sizeof(*state));
    if (!state) return NULL;
    if (dfn2_state_init(state, &grid, NULL, NULL) != 0) {
        free(state);
        return NULL;
    }
    dfn2_set_erb_matrices(state, fwd, inv);
    return state;
}
void parity_dfn_destroy(void *state) { free(state); }
void parity_dfn_features(void *state, const float *re, const float *im,
                         float *erb, float *spec) {
    dfn2_compute_features((DFN2State *)state, re, im, erb, spec);
}
int parity_dfn_compose(void *state, const float *re, const float *im,
                       const float *mask, const float *coefs, float alpha,
                       float *out_re, float *out_im) {
    return dfn2_compose((DFN2State *)state, re, im, mask, coefs, alpha,
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
    DFN2Grid grid;
    int total, c, ready_bad = 0, window_bad = 0, pad_bad = 0, windows = 0, emitted = 0;
    int first_ready = -1, n_erb, df_bins, n_bins, coefs_n, mask_la, df_la;
    float atten_lim_db;

    if (argc != 10) return 2;
    if (dfn2_grid_make(&grid, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                       atoi(argv[8]), atoi(argv[9])) != 0)
        return 2;
    atten_lim_db = (float)atof(argv[3]);
    n_erb = grid.n_erb;
    df_bins = grid.df_bins;
    n_bins = grid.n_bins;
    mask_la = grid.mask_lookahead;
    df_la = grid.df_lookahead;
    coefs_n = df_bins * DFN2_DF_ORDER * 2;

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
    win_erb = take(&cursor, (size_t)frames * DFN2_MODEL_WINDOW_FRAMES * n_erb);
    win_spec = take(&cursor, (size_t)frames * 2 * DFN2_MODEL_WINDOW_FRAMES * df_bins);
    zero_erb = (float *)calloc((size_t)n_erb, sizeof(float));
    zero_spec = (float *)calloc((size_t)2 * df_bins, sizeof(float));
    zero_bins = (float *)calloc((size_t)n_bins, sizeof(float));
    zero_coefs = (float *)calloc((size_t)coefs_n, sizeof(float));
    unit_mask = (float *)malloc((size_t)n_erb * sizeof(float));
    out_re = (float *)calloc((size_t)n_bins, sizeof(float));
    out_im = (float *)calloc((size_t)n_bins, sizeof(float));
    for (c = 0; c < n_erb; ++c) unit_mask[c] = 1.0f;

    if (dfn2_state_init(&st, &grid, NULL, NULL) != 0) return 2;
    dfn2_set_erb_matrices(&st, erb_fwd, erb_inv);
    if (dfn2_model_io_init(&io, &grid) != 0) return 2;
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
                const size_t erb_n = (size_t)DFN2_MODEL_WINDOW_FRAMES * n_erb;
                const size_t spec_n = (size_t)2 * DFN2_MODEL_WINDOW_FRAMES * df_bins;
                size_t i;
                ++windows;
                if (memcmp(win_erb + (size_t)t * erb_n, io.erb_window,
                           erb_n * sizeof(float)) != 0 ||
                    memcmp(win_spec + (size_t)t * spec_n, io.spec_window,
                           spec_n * sizeof(float)) != 0)
                    ++window_bad;
                /* The packed run is all the state has: the rest of the storage stays zero. */
                for (i = erb_n; i < sizeof io.erb_window / sizeof(float); ++i)
                    if (io.erb_window[i] != 0.0f) ++pad_bad;
                for (i = spec_n; i < sizeof io.spec_window / sizeof(float); ++i)
                    if (io.spec_window[i] != 0.0f) ++pad_bad;
            }
            valid = dfn2_compose_stream(
                &st, re, im, 1, has ? head_mask + (size_t)t * n_erb : unit_mask,
                has ? head_coefs + (size_t)t * coefs_n : zero_coefs,
                has ? head_alpha[t] : 0.0f, atten_lim_db, out_re, out_im, &frame);
        } else {
            valid = dfn2_compose_stream(&st, re, im, 0, NULL, NULL, 0.0f, atten_lim_db,
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
    printf("first_ready=%d windows=%d window_bad=%d ready_bad=%d pad_bad=%d emitted=%d\n",
           first_ready, windows, window_bad, ready_bad, pad_bad, emitted);
    return (ready_bad || window_bad || pad_bad) ? 1 : 0;
}
'''


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


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
                 str(wrapper), str(DFN2 / 'dfn2_process.c'),
                 dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(library)])
    assert built.returncode == 0, built.stderr
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_dfn_create.argtypes = [ctypes.c_int] * 6 + [fp, fp]
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
        state = lib.parity_dfn_create(sr, n_fft, n_erb, df_bins, *PAIR, _fp(fwd), _fp(inv))
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
    assert grid_lib.parity_dfn_create(16000, 960, 24, 96, 1, 1, none, none) is None
    state = grid_lib.parity_dfn_create(16000, 256, 24, 96, 1, 1, none, none)
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
    done = subprocess.run(
        [str(stream_driver), str(case['source']), str(result), str(atten_lim),
         str(sr), str(n_fft), str(n_erb), str(df_bins), str(case['pair'][0]),
         str(case['pair'][1])], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert 'first_ready=%d ' % case['pair'][0] in done.stdout, done.stdout
    assert 'window_bad=0 ready_bad=0 pad_bad=0 emitted=%d' % frames in done.stdout, done.stdout

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
    done = subprocess.run([str(exe), str(stream_case['source']),
                           str(stream_case['work'] / 'unused.bin'), '0', '16000', '960',
                           '24', '96', '1', '1'], capture_output=True, text=True)
    assert done.returncode == 2
