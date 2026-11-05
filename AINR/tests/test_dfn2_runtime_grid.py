"""The DFN2 library holds no grid: the run-time grid, proved end to end.

One build of the library serves every grid whose size fits its capacities
(``DFN2_MAX_N_FFT`` / ``_N_ERB`` / ``_DF_BINS`` / ``_DF_RING``, dfn2_grid.h); the grid is an
argument. The tests here pin the property that makes that safe, and the identities that keep
two grids from being mixed up:

  capacity independence  the whole pre/post chain (``dfn2_runtime_grid_chain.c``: DFN2_IO_FREQ
                         and DFN2_IO_TIME, a stand-in accelerator, refused heads and state,
                         skipped and unwritten frames) prints the same digests whether the
                         state is sized for the 48 kHz capacity or tight for the grid under
                         test, NEON and scalar, full and mask-only. A stride that reads the
                         capacity instead of the grid breaks it; three such strides are put
                         into a copy of the library and the comparison goes red.
  pool identity          get_mem_size differs between grids, lookahead pairs, full and
                         mask-only, and either alpha; init_ex refuses a MemReq of another
                         grid or pair; a descriptor of one grid is refused by the host of
                         another. The pool size and hash of the shipped 48 kHz 1/1 grid are
                         pinned.
  normaliser starts      the pool hash reads dfn2_norm_starts[] of dfn2_process.c, the array
                         dfn2_state_init() starts the normalisers from: units built with other
                         starts link and cannot change the hash; process.c alone decides it
  no grid in the library no dfn2_*.c / .h defines or uses a grid macro (those are the consumer
                         headers' business); the test grid header stands alone and its band-border
                         generator, shared by the C suites, holds on every grid
  variant link symbol    objects built for mask-only and full, or with other capacities, do not
                         link; the grid and the pair are not in the symbol

The digests of the shipped grids are printed for the record; they are not pinned, because the
chain runs log10f / powf of the C library it is linked with.
"""

import concurrent.futures
import itertools
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
DFN2 = dfn_pairs.family_dir(FAMILY)
TESTS_DIR = Path(__file__).resolve().parent
CHAIN_SOURCE = TESTS_DIR / 'dfn2_runtime_grid_chain.c'
UNITS = ('process', 'model_io', 'prepost')
SIMD_MODES = (1, 0)

pytestmark = pytest.mark.skipif(
    dfn_pairs.require_toolchain() is not None,
    reason='C build tools unavailable')

_MASK = '-DDFN2_MASK_ONLY'


def _caps(fft, erb, df, ring):
    return ['-DDFN2_MAX_N_FFT=%d' % fft, '-DDFN2_MAX_N_ERB=%d' % erb,
            '-DDFN2_MAX_DF_BINS=%d' % df, '-DDFN2_MAX_DF_RING=%d' % ring]


# Library builds of the capacity matrix. 'W' is the default build: sized for the 48 kHz grid
# (1024 / 32 / 96, ring 5). The 'T' builds are tight for one grid; the 'R7' ones hold the
# lookahead pairs with a ring of 6 or 7.
FULL_CONFIGS = {
    'W': [],
    'W7': ['-DDFN2_MAX_DF_RING=7'],
    'T32': _caps(512, 28, 96, 5),
    'T16': _caps(256, 24, 96, 5),
    'T16R7': _caps(256, 24, 96, 7),
    'T16D': _caps(256, 24, 64, 5),
}
MASK_CONFIGS = {
    'MD': [_MASK],
    'MW': [_MASK, '-DDFN2_MAX_DF_RING=3'],
    'MT32': [_MASK, *_caps(512, 28, 96, 3)],
    'MT16': [_MASK, *_caps(256, 24, 96, 3)],
}
CONFIGS = {**FULL_CONFIGS, **MASK_CONFIGS}

# name -> (sample rate, n_fft, n_erb, df_bins)
GRIDS = {
    'g48': (48000, 1024, 32, 96),
    'g32': (32000, 512, 28, 96),
    'g24': (24000, 512, 24, 96),
    'g16': (16000, 256, 24, 96),
    'g16d64': (16000, 256, 24, 64),
}
FULL_PAIRS = [(1, 1), (0, 0), (2, 0), (0, 1), (2, 2), (1, 3), (0, 4), (2, 3), (2, 4)]
MASK_PAIRS = [(1, 0), (0, 0), (2, 0)]


def _capacity(flags):
    cap = {'N_FFT': 1024, 'N_ERB': 32, 'DF_BINS': 96,
           'DF_RING': 2 if _MASK in flags else 5}
    for flag in flags:
        match = re.fullmatch(r'-DDFN2_MAX_(N_FFT|N_ERB|DF_BINS|DF_RING)=(\d+)', flag)
        if match:
            cap[match.group(1)] = int(match.group(2))
    return cap


def _ring(pair, mask_only):
    noisy = pair[0] + pair[1] + 1
    return noisy if mask_only else max(5, noisy)


def _fits(flags, grid, pair):
    cap = _capacity(flags)
    mask_only = _MASK in flags
    return (grid[1] <= cap['N_FFT'] and grid[2] <= cap['N_ERB'] and grid[3] <= cap['DF_BINS']
            and _ring(pair, mask_only) <= cap['DF_RING'])


def _cc(args, **kwargs):
    return subprocess.run(['cc', *args], capture_output=True, text=True, **kwargs)


class _Builds:
    """Everything the tests compile, in one directory and memoised: the three library units per
    (library directory, flags, SIMD mode), and the programs linked against them."""

    def __init__(self, work):
        self.work = work
        self._counter = itertools.count(1)
        self._objects = {}
        self._exes = {}
        self._runs = {}

    def _key(self, lib, flags, simd):
        return (str(lib), tuple(flags), simd)

    def _compile_unit(self, lib, unit, flags, simd):
        stem = 'unit%d_%s' % (next(self._counter), unit)
        obj = self.work / (stem + '.o')
        source = Path(lib) / ('dfn2_%s.c' % unit)
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(lib), '-c', str(source),
                    '-o', str(obj)])
        assert done.returncode == 0, done.stderr
        return obj

    def prebuild(self, specs, chains=True):
        """Compile the units of every (lib, flags, simd) in `specs` and, with `chains`, link
        the chain driver against them, a few at a time."""
        dfn_pairs.house_cflags(1), dfn_pairs.house_cflags(0)   # resolved before the threads
        dfn_pairs.audio_common_lib(1), dfn_pairs.audio_common_lib(0)
        keys = []
        for lib, flags, simd in specs:
            key = self._key(lib, flags, simd)
            if key not in keys and key not in self._objects:
                keys.append(key)
        jobs = [(key, unit) for key in keys for unit in UNITS]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            built = list(pool.map(
                lambda job: self._compile_unit(job[0][0], job[1], job[0][1], job[0][2]), jobs))
        for index, key in enumerate(keys):
            self._objects[key] = built[index * len(UNITS):(index + 1) * len(UNITS)]
        if chains:
            todo = []
            for lib, flags, simd in specs:
                if ('chain', *self._key(lib, flags, simd)) not in self._exes:
                    todo.append((lib, flags, simd))
            todo = list(dict.fromkeys((str(l), tuple(f), s) for l, f, s in todo))
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                exes = list(pool.map(lambda t: self._link_chain(*t), todo))
            for spec, exe in zip(todo, exes):
                self._exes[('chain', *spec)] = exe

    def objects(self, flags, simd=1, lib=DFN2):
        key = self._key(lib, flags, simd)
        if key not in self._objects:
            self.prebuild([(lib, flags, simd)], chains=False)
        return list(self._objects[key])

    def _link_chain(self, lib, flags, simd):
        objs = self._objects[self._key(lib, flags, simd)]
        exe = self.work / ('chain%d' % next(self._counter))
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(lib), str(CHAIN_SOURCE),
                    *map(str, objs), dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(exe)])
        assert done.returncode == 0, done.stderr
        return exe

    def chain(self, flags, simd=1, lib=DFN2):
        key = ('chain', *self._key(lib, flags, simd))
        if key not in self._exes:
            self.prebuild([(lib, flags, simd)])
        return self._exes[key]

    def link(self, main_source, objs, label, flags=(), simd=1):
        """Compile `main_source` (under `flags`) and link it with `objs`; returns the finished
        process and the executable's path."""
        stem = '%s_%d' % (label, next(self._counter))
        source = self.work / (stem + '.c')
        source.write_text(main_source, encoding='utf-8')
        exe = self.work / stem
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(DFN2), str(source),
                    *map(str, objs), dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(exe)])
        return done, exe

    def run_chain(self, config, simd, grid, pair, lib=DFN2):
        """The output of the chain driver of library build `config` on a grid and a pair."""
        key = (str(lib), config, simd, grid, pair)
        if key not in self._runs:
            exe = self.chain(CONFIGS[config], simd, lib)
            done = subprocess.run([str(exe), *map(str, grid), *map(str, pair)],
                                  capture_output=True, text=True)
            assert done.returncode == 0, (config, grid, pair, done.stdout, done.stderr)
            self._runs[key] = done.stdout
        return self._runs[key]


def _digests(output):
    match = re.search(r'chain digest=([0-9a-f]{16}):([0-9a-f]{16})', output)
    assert match, output
    return match.groups()


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    return _Builds(tmp_path_factory.mktemp('dfn2_runtime_grid'))


@pytest.fixture(scope='module')
def matrix(builds):
    """Every library build of the capacity matrix, both SIMD modes, with the chain driver."""
    builds.prebuild([(DFN2, flags, simd) for flags in CONFIGS.values() for simd in SIMD_MODES])
    return builds


# ---------------------------------------------------------------------------
# capacity independence
# ---------------------------------------------------------------------------

def _compare_builds(builds, configs, grid_name, pairs, simd, lib=DFN2):
    """For every pair, the outputs of the builds that fit the grid; returns
    (number of comparisons, [mismatch descriptions])."""
    grid = GRIDS[grid_name]
    compared, bad = 0, []
    for pair in pairs:
        fit = [name for name, flags in configs.items() if _fits(flags, grid, pair)]
        if len(fit) < 2:
            continue
        outputs = {name: builds.run_chain(name, simd, grid, pair, lib) for name in fit}
        reference = outputs[fit[0]]
        compared += 1
        for name in fit[1:]:
            if outputs[name] != reference:
                bad.append('%s %s: %s %s vs %s %s' % (
                    grid_name, pair, fit[0], _digests(reference), name, _digests(outputs[name])))
    return compared, bad


@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
@pytest.mark.parametrize('grid_name', list(GRIDS))
def test_full_chain_is_the_same_whatever_the_capacity(matrix, grid_name, simd):
    """The state sized for the 48 kHz capacity and the state tight for the grid give the same
    digest (FREQ and TIME): every stride reads the grid, never the capacity."""
    compared, bad = _compare_builds(matrix, FULL_CONFIGS, grid_name, FULL_PAIRS, simd)
    assert compared >= 7, 'too few pairs fit two builds: %d' % compared
    assert not bad, bad


@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
@pytest.mark.parametrize('grid_name', list(GRIDS))
def test_mask_only_chain_is_the_same_whatever_the_capacity(matrix, grid_name, simd):
    compared, bad = _compare_builds(matrix, MASK_CONFIGS, grid_name, MASK_PAIRS, simd)
    assert compared >= 2, compared
    assert not bad, bad


@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
def test_the_chain_tells_grids_and_pairs_apart(matrix, simd):
    """Not a vacuous comparison: every grid and every lookahead pair gives its own digest, and
    the mask-only chain is not the full one."""
    seen = {}
    for grid_name, grid in GRIDS.items():
        for pair in FULL_PAIRS:
            if _fits(CONFIGS['W7'], grid, pair):
                digest = _digests(matrix.run_chain('W7', simd, grid, pair))
                assert digest not in seen, (grid_name, pair, seen[digest])
                seen[digest] = (grid_name, pair)
        for pair in MASK_PAIRS:
            if _fits(CONFIGS['MW'], grid, pair):
                digest = _digests(matrix.run_chain('MW', simd, grid, pair))
                assert digest not in seen, (grid_name, pair, seen[digest])
                seen[digest] = ('mask-only ' + grid_name, pair)
    assert len(seen) == len(GRIDS) * (len(FULL_PAIRS) + len(MASK_PAIRS))


@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
def test_a_build_too_small_for_the_grid_refuses_it(matrix, simd):
    """A grid above the capacity is not run in a smaller state: dfn2_grid_make() refuses it
    (the chain driver prints the status text and exits 3)."""
    for config, grid_name, pair in [('T16', 'g48', (1, 1)), ('T16', 'g32', (1, 1)),
                                    ('T32', 'g48', (1, 1)), ('T16D', 'g16', (1, 1)),
                                    ('T16', 'g16', (2, 4)), ('W', 'g16', (2, 4)),
                                    ('MD', 'g16', (2, 0))]:
        exe = matrix.chain(CONFIGS[config], simd)
        done = subprocess.run([str(exe), *map(str, GRIDS[grid_name]), *map(str, pair)],
                              capture_output=True, text=True)
        assert done.returncode == 3 and 'capacity' in done.stdout, (
            config, grid_name, pair, done.returncode, done.stdout)


def test_the_chain_driver_rejects_bad_arguments(matrix):
    exe = matrix.chain(CONFIGS['W'], 1)
    for args in ([], ['1', '2', '3'], ['16000', '256', '24', '96', '1', 'x'],
                 ['16000', '256', '24', '96', '1']):
        assert subprocess.run([str(exe), *args], capture_output=True,
                              text=True).returncode == 2, args
    done = subprocess.run([str(exe), '16000', '960', '24', '96', '1', '1'],
                          capture_output=True, text=True)
    assert done.returncode == 3 and 'power of two' in done.stdout


def test_the_digests_of_the_shipped_grids_are_printed_for_the_record(matrix, capsys):
    """Not pinned: the chain runs log10f / powf of the C library it is linked with. The lines
    are the ones to compare with a build of the compile-time grid."""
    rows = [('g48_1_1', 'W', 'g48', (1, 1)), ('g16_0_0', 'W', 'g16', (0, 0)),
            ('g32_0_0', 'W', 'g32', (0, 0)), ('m48_1_0', 'MD', 'g48', (1, 0))]
    with capsys.disabled():
        print()
        for label, config, grid_name, pair in rows:
            for simd in SIMD_MODES:
                out = matrix.run_chain(config, simd, GRIDS[grid_name], pair)
                freq, time = _digests(out)
                backend = re.search(r'backend=(\S+)', out).group(1)
                print('  chain digest %-8s %-6s freq %s time %s  (%s)' % (
                    label, 'neon' if simd else 'scalar', freq, time, backend))


# ---------------------------------------------------------------------------
# the gate can fail: a stride that reads the capacity is caught
# ---------------------------------------------------------------------------

# (name, file, text, replacement, grid, tight config, wide config): each puts the capacity where
# the grid's value belongs; the comparison is made between a build whose capacity equals the grid
# (the mutated code is then right) and the wide one (where it is wrong).
_MUTATIONS = [
    ('erb window shifted by the capacity', 'dfn2_model_io.c',
     'memmove(window, window + n_erb,', 'memmove(window, window + DFN2_MAX_N_ERB,',
     'g16', 'T16', 'W'),
    ('erb normaliser ramp over the capacity', 'dfn2_process.c',
     'float position = (float)b / (float)(grid->n_erb - 1);',
     'float position = (float)b / (float)(DFN2_MAX_N_ERB - 1);', 'g16', 'T16', 'W'),
    ('spec window shifted by the capacity', 'dfn2_model_io.c',
     'memmove(plane, plane + df_bins,', 'memmove(plane, plane + DFN2_MAX_DF_BINS,',
     'g16d64', 'T16D', 'W'),
]


def _mutated_library(work, name, filename, text, replacement):
    """A copy of the library sources with one text replaced; the text must be there exactly once
    (a mutation that no longer applies must not pass as a green gate)."""
    lib = work / ('lib_' + re.sub(r'\W+', '_', name))
    lib.mkdir()
    for path in DFN2.glob('dfn2_*.[ch]'):
        shutil.copy(path, lib / path.name)
    source = (lib / filename).read_text(encoding='utf-8')
    assert source.count(text) == 1, ('the mutation no longer applies', name, source.count(text))
    (lib / filename).write_text(source.replace(text, replacement), encoding='utf-8')
    return lib


def test_a_copy_of_the_library_passes_the_capacity_comparison(builds, tmp_path):
    """The control of the mutations below: the same copy, not mutated, compares equal."""
    lib = tmp_path / 'lib_copy'
    lib.mkdir()
    for path in DFN2.glob('dfn2_*.[ch]'):
        shutil.copy(path, lib / path.name)
    for grid_name, tight in (('g16', 'T16'), ('g16d64', 'T16D')):
        grid = GRIDS[grid_name]
        wide = builds.run_chain('W', 1, grid, (1, 1), lib)
        assert builds.run_chain(tight, 1, grid, (1, 1), lib) == wide
        assert wide == builds.run_chain('W', 1, grid, (1, 1)), 'a copy builds the same chain'


@pytest.mark.parametrize('mutation', _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_the_capacity_comparison_goes_red_for_a_stride_that_reads_the_capacity(
        builds, tmp_path, mutation):
    name, filename, text, replacement, grid_name, tight, wide = mutation
    lib = _mutated_library(tmp_path, name, filename, text, replacement)
    grid = GRIDS[grid_name]
    red = [builds.run_chain(tight, 1, grid, pair, lib) != builds.run_chain(wide, 1, grid, pair, lib)
           for pair in [(1, 1), (0, 0), (2, 0)]]
    assert any(red), 'the comparison did not notice: %s' % name


# ---------------------------------------------------------------------------
# the pool identity, the descriptor and the host's grid check
# ---------------------------------------------------------------------------

_POOL_MAIN = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dfn2_prepost.h"

#define MAX_BINS 513
#define MAX_ERB 32
static float fwd[MAX_BINS * MAX_ERB], inv[MAX_BINS * MAX_ERB];
static float zeros[MAX_BINS];
static int infer_calls;

static float float_of(uint32_t u) { float v; memcpy(&v, &u, sizeof v); return v; }

static int read_grid(const char *text, DFN2Grid *g, int *used) {
    int a, b, c, d, e, f, n = 0;
    if (sscanf(text, "%d %d %d %d %d %d%n", &a, &b, &c, &d, &e, &f, &n) != 6) return -2;
    *used = n;
    return dfn2_grid_make(g, a, b, c, d, e, f);
}

static void fill_tables(const DFN2Grid *g) {
    size_t i;
    for (i = 0; i < (size_t)g->n_bins * g->n_erb; ++i) fwd[i] = inv[i] = 0.01f;
}

static int config_for(const DFN2Grid *g, DFN2PrepostConfig *cfg) {
    if (dfn2_prepost_config_defaults(cfg, DFN2_IO_FREQ, g) != 0) return -1;
    fill_tables(g);
    cfg->erb_fwd = fwd;
    cfg->erb_inv = inv;
    return 0;
}

static int accepted(void *pool, size_t bytes, const DFN2PrepostConfig *cfg,
                    const DFN2PrepostMemReq *expected) {
    return dfn2_prepost_init_ex(pool, bytes, cfg, expected) != NULL;
}

static int infer_ok(void *user, const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    size_t i;
    (void)user;
    (void)in;
    ++infer_calls;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.5f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.0f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.5f;
#endif
    return 0;
}

int main(void) {
    char line[512], verb[16];
    while (fgets(line, sizeof line, stdin)) {
        DFN2Grid a, b;
        int used = 0, used2 = 0, st;
        memset(&a, 0, sizeof a);
        memset(&b, 0, sizeof b);
        if (sscanf(line, "%15s", verb) != 1) continue;
        st = read_grid(line + strlen(verb) + 1, &a, &used);
        if (st != 0) { printf("%s badgrid %d\n", verb, st); continue; }
        if (!strcmp(verb, "req")) {
            /* req <grid> [erb_alpha_bits spec_alpha_bits]: get_mem_size of the FREQ config */
            unsigned eb, sb;
            DFN2PrepostConfig cfg;
            DFN2PrepostMemReq r;
            memset(&r, 0, sizeof r);
            if (sscanf(line + strlen(verb) + 1 + used, "%x %x", &eb, &sb) == 2) {
                a.erb_norm_alpha = float_of(eb);
                a.spec_norm_alpha = float_of(sb);
            }
            if (config_for(&a, &cfg) != 0) { printf("req nocfg\n"); continue; }
            st = dfn2_prepost_get_mem_size(&cfg, &r);
            printf("req %d %llu %08x\n", st, (unsigned long long)r.bytes,
                   (unsigned)r.build_flags_hash);
        } else if (!strcmp(verb, "initex")) {
            /* initex <grid A> <grid B>: a MemReq made for A offered for a pool built for B */
            DFN2PrepostConfig ca, cb;
            DFN2PrepostMemReq ra, rb, bad;
            size_t bytes;
            void *pool;
            int own, cross, tampered = 0;
            if (read_grid(line + strlen(verb) + 1 + used, &b, &used2) != 0) {
                printf("initex badgrid2\n");
                continue;
            }
            if (config_for(&a, &ca) != 0 || config_for(&b, &cb) != 0 ||
                dfn2_prepost_get_mem_size(&ca, &ra) != 0 ||
                dfn2_prepost_get_mem_size(&cb, &rb) != 0) {
                printf("initex nosize\n");
                continue;
            }
            bytes = (size_t)rb.bytes;
            pool = aligned_alloc(16, bytes + 16);
            own = accepted(pool, bytes, &cb, &rb);
            cross = accepted(pool, bytes, &cb, &ra);
            /* every field of the MemReq is compared: a copy with one field changed is refused */
            bad = rb; bad.build_flags_hash ^= 1u; tampered += !accepted(pool, bytes, &cb, &bad);
            bad = rb; bad.bytes += 16u; tampered += !accepted(pool, bytes + 16u, &cb, &bad);
            bad = rb; bad.io_mode ^= 1u; tampered += !accepted(pool, bytes, &cb, &bad);
            bad = rb; bad.layout_version += 1u; tampered += !accepted(pool, bytes, &cb, &bad);
            bad = rb; bad.descriptor_version += 1u; tampered += !accepted(pool, bytes, &cb, &bad);
            bad = rb; bad.alignment *= 2u; tampered += !accepted(pool, bytes, &cb, &bad);
            printf("initex %d %d %d\n", own, cross, tampered);
            free(pool);
        } else if (!strcmp(verb, "runframe") || !strcmp(verb, "runnull")) {
            /* runframe <grid A> <grid B>: a host for B with one open frame is driven with the
             * descriptor of A (runnull: no descriptor) */
            DFN2PrepostConfig cb;
            DFN2ModelIoDescriptor desc;
            DFN2Model model;
            DFN2Prepost *p;
            int i, rc, ready = 0;
            if (read_grid(line + strlen(verb) + 1 + used, &b, &used2) != 0 ||
                config_for(&b, &cb) != 0) {
                printf("%s badgrid2\n", verb);
                continue;
            }
            p = dfn2_prepost_create(&cb);
            for (i = 0; i < 4 && !ready; ++i)
                ready = dfn2_prepost_pre_process_freq(p, zeros, zeros);
            if (dfn2_model_io_descriptor_default(&a, &desc) != 0) {
                printf("%s nodesc\n", verb);
                continue;
            }
            model.user = NULL;
            model.infer = infer_ok;
            model.reset = NULL;
            model.io_descriptor = strcmp(verb, "runnull") ? &desc : NULL;
            infer_calls = 0;
            rc = dfn2_model_run_frame(&model, p);
            printf("%s %d %d %d\n", verb, ready, rc, infer_calls);
            dfn2_prepost_destroy(p);
        } else if (!strcmp(verb, "desc")) {
            /* desc <grid A> <grid B>: the default descriptor of A against A and against B */
            DFN2ModelIoDescriptor d;
            int rc = dfn2_model_io_descriptor_default(&a, &d);
            if (read_grid(line + strlen(verb) + 1 + used, &b, &used2) != 0) {
                printf("desc badgrid2\n");
                continue;
            }
            printf("desc %d %d %d\n", rc, dfn2_model_io_descriptor_validate(&a, &d),
                   dfn2_model_io_descriptor_validate(&b, &d));
        } else if (!strcmp(verb, "descfield")) {
            /* descfield <grid>: every one of the 18 int fields changed by one is refused */
            DFN2ModelIoDescriptor d;
            int words[sizeof(DFN2ModelIoDescriptor) / sizeof(int)], k, refused = 0;
            dfn2_model_io_descriptor_default(&a, &d);
            for (k = 0; k < (int)(sizeof words / sizeof words[0]); ++k) {
                DFN2ModelIoDescriptor t = d;
                memcpy(words, &t, sizeof t);
                words[k] += 1;
                memcpy(&t, words, sizeof t);
                refused += dfn2_model_io_descriptor_validate(&a, &t) == -1;
            }
            printf("descfield %d %d %d\n", (int)(sizeof words / sizeof words[0]), refused,
                   dfn2_model_io_descriptor_validate(&a, &d));
        } else if (!strcmp(verb, "starts")) {
            /* starts <grid>: the normaliser starting points as compiled into dfn2_process.c
             * (dfn2_norm_starts) and what dfn2_state_init() puts into the two normalisers: the
             * first and last band of the ERB one, the first and last bin of the spectrum one */
            static DFN2State dsp;
            int i;
            if (dfn2_state_init(&dsp, &a, NULL, NULL) != 0) {
                printf("starts noinit\n");
                continue;
            }
            printf("starts");
            for (i = 0; i < DFN2_NORM_STARTS; ++i) printf(" %.9g", (double)dfn2_norm_starts[i]);
            printf(" | %.9g %.9g %.9g %.9g\n", (double)dsp.erb_norm_state[0],
                   (double)dsp.erb_norm_state[a.n_erb - 1], (double)dsp.spec_norm_state[0],
                   (double)dsp.spec_norm_state[a.df_bins - 1]);
        } else if (!strcmp(verb, "nullcalls")) {
            DFN2ModelIoDescriptor d;
            DFN2PrepostConfig cfg;
            printf("nullcalls %d %d %d %d\n", dfn2_model_io_descriptor_default(NULL, &d),
                   dfn2_model_io_descriptor_validate(NULL, &d),
                   dfn2_model_io_descriptor_validate(&a, NULL),
                   dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ, NULL));
        } else if (!strcmp(verb, "badgrid")) {
            /* a grid edited by hand is refused by every entry point that takes one */
            DFN2PrepostConfig cfg;
            DFN2ModelIoDescriptor d;
            DFN2ModelIOState io;
            DFN2State dsp;
            DFN2Grid edited = a;
            edited.ring += 1;
            printf("badgrid %d %d %d %d\n",
                   dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ, &edited),
                   dfn2_model_io_descriptor_default(&edited, &d),
                   dfn2_model_io_init(&io, &edited),
                   dfn2_state_init(&dsp, &edited, NULL, NULL));
        }
    }
    return 0;
}
'''

_PARSE_REQ = re.compile(r'req (-?\d+) (\d+) ([0-9a-f]{8})')

# The shipped grid: 48 kHz, 1024-point frames, 32 ERB bands, 96 deep-filter bins, lookahead 1/1.
_SHIPPED = (48000, 1024, 32, 96, 1, 1)

# The pool of the shipped grid in DFN2_IO_FREQ, as the pre/post unit sizes it for a 64-bit host
# (the state holds pointers, so a 32-bit host has another size; neither number depends on libm).
# CARVE_VERSION 9: DFN2State and DFN2Prepost each hold a DFN2Grid (2 x 56 bytes), 219536 -> 219648,
# and the hash folds the whole grid, the lookahead pair and both alphas.
_SHIPPED_POOL_BYTES = 219648
_SHIPPED_POOL_HASH = '5a69e185'


class _Pool:
    """The pool program of one library build: commands in on stdin, one result line each."""

    def __init__(self, exe):
        self.exe = exe

    def run(self, lines):
        done = subprocess.run([str(self.exe)], input='\n'.join(lines) + '\n',
                              capture_output=True, text=True)
        assert done.returncode == 0, (done.returncode, done.stderr)
        out = done.stdout.splitlines()
        assert len(out) == len(lines), (out, lines)
        return out

    def req(self, grid, alphas=None):
        line = 'req %d %d %d %d %d %d' % grid
        if alphas:
            line += ' %08x %08x' % alphas
        status, size, digest = _PARSE_REQ.fullmatch(self.run([line])[0]).groups()
        assert status == '0', (grid, status)
        return int(size), digest

    def starts(self, grid):
        """([the five dfn2_norm_starts], [erb first, erb last, spec first, spec last] as
        dfn2_state_init() set them), as float32 values."""
        verb, *fields = self.run(['starts %d %d %d %d %d %d' % grid])[0].split()
        assert verb == 'starts' and fields[5] == '|', fields
        values = [float(np.float32(f)) for f in fields[:5] + fields[6:]]
        return values[:5], values[5:]


@pytest.fixture(scope='module')
def pools(builds):
    full = builds.objects([])
    mask = builds.objects([_MASK])
    return {name: _Pool(builds.link(_POOL_MAIN, objs, 'pool_' + name, flags)[1])
            for name, objs, flags in (('full', full, []), ('mask', mask, [_MASK]))}


def _f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def test_pool_identity_follows_the_grid_the_pair_the_variant_and_the_alphas(pools):
    full, mask = pools['full'], pools['mask']
    shipped = full.req(_SHIPPED)
    assert full.req(_SHIPPED) == shipped, 'get_mem_size is a pure function of the config'
    # Another grid, another identity: bytes follow the frame size, the hash follows everything.
    grids = [(32000, 512, 28, 96, 1, 1), (16000, 256, 24, 96, 1, 1),
             (24000, 512, 24, 96, 1, 1), (16000, 256, 24, 64, 1, 1)]
    # One integer at a time from the 16 kHz grid. 16100 Hz rounds to the same alpha as 16000 Hz
    # (0.992), so the sample rate is in the hash on its own and not through the alpha.
    single = [(16100, 256, 24, 96, 1, 1), (16000, 512, 24, 96, 1, 1), (16000, 256, 28, 96, 1, 1),
              (16000, 256, 24, 95, 1, 1), (16000, 256, 24, 96, 1, 0), (16000, 256, 24, 96, 0, 1)]
    identities = [shipped] + [full.req(g) for g in grids + single]
    assert len({i[1] for i in identities}) == len(identities), identities
    assert len({i[0] for i in identities[:3]}) == 3, 'pool bytes follow the frame size'
    # Another lookahead pair on the same integers: another hash.
    pairs = [full.req((16000, 256, 24, 96, m, d)) for m, d in
             [(1, 1), (0, 0), (2, 0), (0, 1), (1, 0), (2, 2), (0, 4)]]
    assert len({p[1] for p in pairs}) == len(pairs), pairs
    # Mask-only is another build: smaller, and another hash.
    full_16, mask_16 = full.req((16000, 256, 24, 96, 1, 0)), mask.req((16000, 256, 24, 96, 1, 0))
    assert mask_16[0] < full_16[0] and mask_16[1] != full_16[1]
    full_48, mask_48 = full.req((48000, 1024, 32, 96, 1, 0)), mask.req((48000, 1024, 32, 96, 1, 0))
    assert mask_48[0] < full_48[0] and mask_48[1] != full_48[1]
    assert mask.req((16000, 256, 24, 96, 1, 0)) == mask_16 != mask.req((16000, 256, 24, 96, 0, 0))
    # Either alpha alone is in the hash (a model trained with another tau).
    base = full.req((16000, 256, 24, 96, 1, 1))
    a_def = _f32_bits(0.992)
    a_new = _f32_bits(0.979)
    assert full.req((16000, 256, 24, 96, 1, 1), (a_def, a_def)) == base, 'the default alpha'
    erb_only = full.req((16000, 256, 24, 96, 1, 1), (a_new, a_def))
    spec_only = full.req((16000, 256, 24, 96, 1, 1), (a_def, a_new))
    both = full.req((16000, 256, 24, 96, 1, 1), (a_new, a_new))
    assert len({base[1], erb_only[1], spec_only[1], both[1]}) == 4
    assert erb_only[0] == spec_only[0] == both[0] == base[0], 'an alpha changes the hash only'


# The normaliser starting points are libDF's own literals, the same for every grid; a model
# trained with others defines them on the build line (every translation unit). dfn2_process.c
# holds the five it was built with (dfn2_norm_starts[]) and dfn2_state_init() starts the
# normalisers from them; the pool hash reads the same array, so it describes the state that is
# actually initialised. Each entry: build flag, index in dfn2_norm_starts, the value it sets.
_NORMALISER_STARTS = [
    ('-DDFN2_ERB_NORM_INIT_LO_DB=-70.0f', 0, -70.0),
    ('-DDFN2_ERB_NORM_INIT_HI_DB=-80.0f', 1, -80.0),
    ('-DDFN2_ERB_NORM_SCALE_DB=30.0f', 2, 30.0),
    ('-DDFN2_SPEC_NORM_INIT_LO=0.002f', 3, 0.002),
    ('-DDFN2_SPEC_NORM_INIT_HI=0.0002f', 4, 0.0002),
]
# what dfn2_state_init() makes of each start: (index in the state read-out, shipped grid) -- the
# ERB scale is not a starting point of the state, it scales the feature
_STATE_SLOT = {0: 0, 1: 1, 3: 2, 4: 3}
_START_IDS = [flag[2:].split('=')[0] for flag, _i, _v in _NORMALISER_STARTS]
_START_DEFAULTS = [-60.0, -90.0, 40.0, 0.001, 0.0001]


def _f32(value):
    return float(np.float32(value))


def _link_pool(builds, objs, label):
    """The pool program linked from `objs` (process, model_io, prepost, in any build mix)."""
    done, exe = builds.link(_POOL_MAIN, objs, label)
    assert done.returncode == 0, ('the mix does not link', label, done.stderr)
    return _Pool(exe)


@pytest.mark.parametrize('flag,index,value', _NORMALISER_STARTS, ids=_START_IDS)
def test_the_pool_hash_reads_the_normaliser_starts_of_the_process_unit(
        builds, pools, flag, index, value):
    """Replaces the HEAD gate that objects built with different normaliser starts do not link:
    the link symbol no longer carries the starts (one build serves every grid), the starts have
    one owner instead. Units of mixed builds LINK, and the pool hash is dfn2_process.c's:
      prepost.c / model_io.c built with other starts than process.c   -> the hash of the default
      process.c built with another start                              -> another hash, the
        same bytes, the same hash as a build with every unit changed
    and the state that dfn2_state_init() starts is the one the hash describes."""
    default, flagged = builds.objects([]), builds.objects([flag])
    base = pools['full'].req(_SHIPPED)
    all_flagged = _link_pool(builds, flagged, 'starts_all')
    process_only = _link_pool(builds, [flagged[0], *default[1:]], 'starts_process')
    prepost_only = _link_pool(builds, [*default[:2], flagged[2]], 'starts_prepost')
    model_io_only = _link_pool(builds, [default[0], flagged[1], default[2]], 'starts_model_io')

    assert prepost_only.req(_SHIPPED) == base, 'prepost.c decided the hash'
    assert model_io_only.req(_SHIPPED) == base
    changed = process_only.req(_SHIPPED)
    assert changed[1] != base[1], 'the start is not in the hash'
    assert changed[0] == base[0], 'a normaliser start changes the hash only'
    assert all_flagged.req(_SHIPPED) == changed, 'only process.c decides'

    # The array and the state: process.c's values, whatever the other units were built with.
    expected = list(_START_DEFAULTS)
    expected[index] = _f32(value)
    default_starts = [_f32(v) for v in _START_DEFAULTS]
    assert process_only.starts(_SHIPPED)[0] == [_f32(v) for v in expected]
    assert prepost_only.starts(_SHIPPED)[0] == default_starts
    state = process_only.starts(_SHIPPED)[1]
    default_state = prepost_only.starts(_SHIPPED)[1]
    if index in _STATE_SLOT:
        # (the last band / bin is lo + 1.0 * (hi - lo) in float32: equal to hi within an ulp)
        assert state[_STATE_SLOT[index]] == pytest.approx(_f32(value), rel=1e-6), state
        assert state != default_state
    else:
        assert state == default_state, 'the scale is not part of the state'


def test_the_normaliser_starts_gate_can_fail(builds, pools, tmp_path):
    """The gate above is not vacuous: a prepost.c that took the starts from its own build line
    (the five macros) instead of dfn2_norm_starts[] puts the start into the hash of a mixed
    build, and the equality with the default hash is lost."""
    flag = _NORMALISER_STARTS[0][0]
    lib = _mutated_library(
        tmp_path, 'prepost_own_starts', 'dfn2_prepost.c',
        'for (i = 0; i < DFN2_NORM_STARTS; ++i) tuning[2 + i] = dfn2_norm_starts[i];',
        'tuning[2] = DFN2_ERB_NORM_INIT_LO_DB; tuning[3] = DFN2_ERB_NORM_INIT_HI_DB; '
        'tuning[4] = DFN2_ERB_NORM_SCALE_DB; tuning[5] = DFN2_SPEC_NORM_INIT_LO; '
        'tuning[6] = DFN2_SPEC_NORM_INIT_HI; (void)i;')
    default, flagged = builds.objects([], lib=lib), builds.objects([flag], lib=lib)
    base = _link_pool(builds, default, 'mut_starts_default').req(_SHIPPED)
    assert base == pools['full'].req(_SHIPPED), 'the control: the mutated copy hashes as shipped'
    mixed = _link_pool(builds, [*default[:2], flagged[2]], 'mut_starts_mixed').req(_SHIPPED)
    assert mixed[1] != base[1], 'the mix still hashes as the default: the gate cannot fail'


@pytest.mark.skipif(struct.calcsize('P') != 8, reason='pool bytes are pinned for 64-bit hosts')
def test_pool_of_the_shipped_grid_is_pinned(pools):
    """CARVE_VERSION 9. A change here is a change of the pool layout or of what its hash covers,
    and is made on purpose."""
    size, digest = pools['full'].req(_SHIPPED)
    assert (size, digest) == (_SHIPPED_POOL_BYTES, _SHIPPED_POOL_HASH), (size, digest)


@pytest.mark.parametrize('pool_name', ['full', 'mask'])
def test_init_ex_refuses_a_memreq_of_another_grid_or_pair(pools, pool_name):
    pool = pools[pool_name]
    mask = pool_name == 'mask'
    base = (16000, 256, 24, 96, 1, 0 if mask else 1)
    others = [(32000, 512, 28, 96, 1, base[5]), (16000, 256, 24, 64, 1, base[5]),
              (16000, 256, 28, 96, 1, base[5]), (24000, 256, 24, 96, 1, base[5]),
              (16100, 256, 24, 96, 1, base[5]),           # same alphas, another sample rate
              (16000, 256, 24, 96, 0, base[5]), (48000, 1024, 32, 96, 1, base[5])]
    if not mask:
        others += [(16000, 256, 24, 96, 1, 0), (16000, 256, 24, 96, 2, 2),
                   (16000, 256, 24, 96, 0, 1)]
    lines = ['initex %d %d %d %d %d %d  %d %d %d %d %d %d' % (*base, *base)]
    lines += ['initex %d %d %d %d %d %d  %d %d %d %d %d %d' % (*other, *base)
              for other in others]
    lines += ['initex %d %d %d %d %d %d  %d %d %d %d %d %d' % (*base, *other)
              for other in others]
    results = pool.run(lines)
    own, cross, tampered = map(int, results[0].split()[1:])
    assert (own, cross, tampered) == (1, 1, 6), ('the control: its own MemReq is accepted '
                                                  'and every changed field refused', results[0])
    for other, line in zip(others + others, results[1:]):
        own, cross, tampered = map(int, line.split()[1:])
        assert (own, cross, tampered) == (1, 0, 6), (other, line)


@pytest.mark.parametrize('pool_name', ['full', 'mask'])
def test_a_descriptor_of_one_grid_is_refused_by_the_host_of_another(pools, pool_name):
    pool = pools[pool_name]
    pair = (1, 0) if pool_name == 'mask' else (1, 1)
    base = (16000, 256, 24, 96, *pair)
    # One field at a time, then the shipped rows.
    others = [(24000, 256, 24, 96, *pair), (16100, 256, 24, 96, *pair),
              (16000, 512, 24, 96, *pair),
              (16000, 256, 28, 96, *pair), (16000, 256, 24, 64, *pair),
              (16000, 256, 24, 96, 0, pair[1]), (32000, 512, 28, 96, *pair),
              (48000, 1024, 32, 96, *pair)]
    if pool_name == 'full':
        others += [(16000, 256, 24, 96, 1, 0), (16000, 256, 24, 96, 2, 2)]
    lines = ['runframe %d %d %d %d %d %d  %d %d %d %d %d %d' % (*base, *base),
             'runnull %d %d %d %d %d %d  %d %d %d %d %d %d' % (*base, *base)]
    lines += ['runframe %d %d %d %d %d %d  %d %d %d %d %d %d' % (*other, *base)
              for other in others]
    results = pool.run(lines)
    # The control: the host's own descriptor drives the accelerator and the frame is committed.
    assert results[0].split() == ['runframe', '1', '1', '1'], results[0]
    # No descriptor at all is refused as well.
    assert results[1].split() == ['runnull', '1', '-1', '0'], results[1]
    for other, line in zip(others, results[2:]):
        assert line.split() == ['runframe', '1', '-1', '0'], (other, line)


@pytest.mark.parametrize('pool_name', ['full', 'mask'])
def test_the_default_descriptor_round_trips_through_its_own_grid_only(pools, pool_name):
    pool = pools[pool_name]
    pair = (1, 0) if pool_name == 'mask' else (1, 1)
    grids = [(48000, 1024, 32, 96, *pair), (32000, 512, 28, 96, *pair),
             (16000, 256, 24, 96, *pair), (24000, 512, 24, 96, *pair),
             (16100, 256, 24, 96, *pair),
             (16000, 256, 24, 64, *pair), (16000, 256, 24, 96, 0, pair[1])]
    if pool_name == 'full':
        grids += [(16000, 256, 24, 96, 2, 2), (16000, 256, 24, 96, 0, 0)]
    lines = ['desc %d %d %d %d %d %d  %d %d %d %d %d %d' % (*a, *b)
             for a in grids for b in grids]
    results = iter(pool.run(lines))
    for a in grids:
        for b in grids:
            rc, same, other = map(int, next(results).split()[1:])
            assert rc == 0 and same == 0, (a, rc, same)
            assert other == (0 if a == b else -1), (a, b, other)
    # Every one of the 18 fields is compared: a descriptor changed in one is refused.
    for line in pool.run(['descfield %d %d %d %d %d %d' % g for g in grids]):
        count, refused, plain = map(int, line.split()[1:])
        assert (count, refused, plain) == (18, 18, 0), line


@pytest.mark.parametrize('pool_name', ['full', 'mask'])
def test_entry_points_refuse_a_null_or_hand_edited_grid(pools, pool_name):
    pool = pools[pool_name]
    pair = (1, 0) if pool_name == 'mask' else (1, 1)
    (calls,) = pool.run(['nullcalls %d %d %d %d %d %d' % (16000, 256, 24, 96, *pair)])
    assert calls.split()[1:] == ['-1', '-1', '-1', '-1'], calls
    (edited,) = pool.run(['badgrid %d %d %d %d %d %d' % (16000, 256, 24, 96, *pair)])
    assert edited.split()[1:] == ['-1', '-1', '-1', '-1'], edited


# ---------------------------------------------------------------------------
# the library holds no grid
# ---------------------------------------------------------------------------

# The grid macros of the former build-time library. They are the test headers' alone
# (tests/dfn2_test_grid.h, pipelines/dfn/tests/dfn_test_grid.h: test scaffolding that sizes fixed
# arrays and sweeps grids), never the library's and never a product's.
GRID_MACROS = [
    'DFN2_GRID_KHZ', 'DFN2_SR', 'DFN2_N_FFT', 'DFN2_N_ERB', 'DFN2_DF_BINS',
    'DFN2_MASK_LOOKAHEAD', 'DFN2_DF_LOOKAHEAD',
    # the rest of what the build-time grid derived
    'DFN2_N_BINS', 'DFN2_WIN_LEN', 'DFN2_HOP_LEN', 'DFN2_OVL_LEN', 'DFN2_ERB_NORM_ALPHA',
    'DFN2_SPEC_NORM_ALPHA', 'DFN2_ANALYSIS_SCALE', 'DFN2_NORMALIZED_STFT_SCALE',
    'DFN2_NORMALIZED_STFT_UNSCALE', 'DFN2_DF_RING', 'DFN2_DF_RING_NOISY', 'DFN2_DF_HISTORY',
    'DFN2_GRID_IS_SHIPPED', 'DFN2_GRID_HEADER',
]
_MACRO_WORDS = re.compile(r'\b(?:%s)\b' % '|'.join(GRID_MACROS))


def _code_only(text):
    """C source with comments, string literals and character literals blanked, newlines kept
    (so a line number still points at the right line). An explanatory mention in a comment or
    an #error message is not a definition or a use."""
    out, i, n = [], 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == '/*':
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            out.append(re.sub(r'[^\n]', ' ', text[i:end]))
            i = end
        elif two == '//':
            end = text.find('\n', i)
            end = n if end < 0 else end
            out.append(' ' * (end - i))
            i = end
        elif text[i] in '"\'':
            quote, j = text[i], i + 1
            while j < n and text[j] != quote and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(quote + re.sub(r'[^\n]', ' ', text[i + 1:j - 1]) + quote)
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def _grid_macro_uses(directory):
    """[(file, line, macro)] of every grid macro in the code of dfn2_*.c / .h in `directory`."""
    hits = []
    for path in sorted(Path(directory).glob('dfn2_*.[ch]')):
        code = _code_only(path.read_text(encoding='utf-8'))
        for number, line in enumerate(code.splitlines(), 1):
            hits += [(path.name, number, m) for m in _MACRO_WORDS.findall(line)]
    return hits


def test_the_library_defines_and_uses_no_grid_macro():
    assert sorted(p.name for p in DFN2.glob('dfn2_*.[ch]')) == [
        'dfn2_grid.h', 'dfn2_model_io.c', 'dfn2_model_io.h', 'dfn2_prepost.c',
        'dfn2_prepost.h', 'dfn2_process.c', 'dfn2_process.h']
    assert _grid_macro_uses(DFN2) == []


def _product_sources():
    """The pipelines' DFN2 product code: the units, the shared headers and the two example mains
    (not their tests/ folders, which are the test scaffolding that picks a grid at build time)."""
    dfn = TESTS_DIR.parents[1] / 'pipelines' / 'dfn'
    paths = sorted([*dfn.glob('*.[ch]'), dfn / 'mono_aec_dfn_res' / 'main.c',
                    dfn / '4ch_aec_bf_dfn_res' / 'main.c'])
    return dfn, paths


_TEST_GRID_INCLUDE = re.compile(r'^\s*#\s*include\s+"(?:[^"]*/)?(?:dfn2?_test_grid)\.h"', re.M)


def _product_grid_uses(paths):
    """[(file, line, what)] of every build-time grid macro in the code of `paths` and of every
    include of a test grid header."""
    hits = []
    for path in paths:
        text = path.read_text(encoding='utf-8')
        for number, line in enumerate(_code_only(text).splitlines(), 1):
            hits += [(path.name, number, m) for m in _MACRO_WORDS.findall(line)]
        hits += [(path.name, text.count('\n', 0, m.start()) + 1, 'include ' + m.group(0).strip())
                 for m in _TEST_GRID_INCLUDE.finditer(text)]
    return hits


def test_the_pipelines_product_code_holds_no_build_time_grid():
    """One set of code: a unit or an example main takes its grid from the caller (a DFN2Grid in
    the config), never from a macro, and does not include the tests' grid headers."""
    dfn, paths = _product_sources()
    if not dfn.is_dir():
        pytest.skip('the pipelines tree is not next to AINR')
    names = [p.name for p in paths]
    for expected in ('dfn_res_stage.c', 'dfn_rate_bridge.c', 'dfn_hop.h', 'dfn_example_args.h'):
        assert expected in names, expected
    assert names.count('main.c') == 2
    assert _product_grid_uses(paths) == []


def test_the_product_scan_flags_a_macro_and_a_test_header_include(tmp_path):
    """The product scan is not vacuous: a grid macro in code and an include of either test grid
    header are found, a mention in a comment or a string is not."""
    probe = tmp_path / 'probe.h'
    for source, want in {
        'int x = DFN2_N_FFT;\n': ['DFN2_N_FFT'],
        '#include "dfn_test_grid.h"\n': ['include #include "dfn_test_grid.h"'],
        '  #  include "../../AINR/tests/dfn2_test_grid.h"\n':
            ['include #  include "../../AINR/tests/dfn2_test_grid.h"'],
        '/* DFN2_SR */ const char *s = "DFN2_N_ERB"; /* dfn_test_grid.h */\n': [],
        'int y = DFN2_MAX_N_FFT;\n': [],
    }.items():
        probe.write_text(source, encoding='utf-8')
        assert [w for _f, _l, w in _product_grid_uses([probe])] == want, source


def test_the_scan_sees_what_it_is_meant_to_see(tmp_path):
    """The scan is not vacuous: each shape of definition or use is found, a mention in a comment,
    a string, a longer identifier (DFN2_MAX_N_FFT) or the consumer macro of another family
    (DFN3_SR) is not, and a real library file with one macro added is flagged."""
    shapes = {
        '#define DFN2_N_FFT 1024\n': ['DFN2_N_FFT'],
        '#ifndef DFN2_GRID_KHZ\n#endif\n': ['DFN2_GRID_KHZ'],
        'int x = DFN2_SR * 2;\n': ['DFN2_SR'],
        'float a[DFN2_N_ERB];\n': ['DFN2_N_ERB'],
        '#if defined(DFN2_DF_BINS) && DFN2_MASK_LOOKAHEAD > 0\n#endif\n':
            ['DFN2_DF_BINS', 'DFN2_MASK_LOOKAHEAD'],
        'int y = DFN2_DF_LOOKAHEAD;\n': ['DFN2_DF_LOOKAHEAD'],
        '/* DFN2_SR */ int a;\n': [],
        '// DFN2_N_FFT is the consumer\'s\nint a;\n': [],
        '/* multi\n   line DFN2_N_ERB */\nint a;\n': [],
        'const char *s = "DFN2_DF_BINS";\n': [],
        '#error "DFN2_SR is the application\'s"\n': [],
        'int a = DFN2_MAX_N_FFT + DFN2_MAX_N_ERB + DFN2_MAX_DF_BINS + DFN3_SR;\n': [],
        "char c = '\"'; int z = DFN2_SR;\n": ['DFN2_SR'],
    }
    for source, want in shapes.items():
        (tmp_path / 'dfn2_probe.h').write_text(source, encoding='utf-8')
        assert [m for _f, _l, m in _grid_macro_uses(tmp_path)] == want, source
    mutated = tmp_path / 'lib'
    mutated.mkdir()
    for path in DFN2.glob('dfn2_*.[ch]'):
        shutil.copy(path, mutated / path.name)
    assert _grid_macro_uses(mutated) == []
    with (mutated / 'dfn2_process.h').open('a', encoding='utf-8') as stream:
        stream.write('#ifndef DFN2_N_ERB\n#define DFN2_N_ERB 32\n#endif\n')
    assert [(f, m) for f, _l, m in _grid_macro_uses(mutated)] == [
        ('dfn2_process.h', 'DFN2_N_ERB'), ('dfn2_process.h', 'DFN2_N_ERB')]


def test_the_consumer_header_is_where_the_grid_macros_live():
    """The macro list is what the tests' own header defines (so the scan above looks for the
    right names)."""
    code = _code_only((TESTS_DIR / 'dfn2_test_grid.h').read_text(encoding='utf-8'))
    for macro in GRID_MACROS[:7]:
        assert re.search(r'\b%s\b' % macro, code), macro


# The test header must stand alone: a program that includes it and nothing else gets the grid
# macros, the scales built on sqrtf, the band-border generator both C suites share and the grid.
_TEST_GRID_TU = r'''
#include "dfn2_test_grid.h"

float scales(void)
{
    return DFN2_NORMALIZED_STFT_SCALE * DFN2_NORMALIZED_STFT_UNSCALE * DFN2_ANALYSIS_SCALE;
}

int borders(int *b)
{
    dfn2_test_band_borders(b);
    return b[DFN2_N_ERB - 1] + dfn2_test_grid()->n_erb;
}
'''

_TEST_GRIDS = {
    'khz48': [], 'khz32': ['-DDFN2_GRID_KHZ=32'], 'khz16': ['-DDFN2_GRID_KHZ=16'],
    'custom24k': ['-DDFN2_SR=24000', '-DDFN2_N_FFT=512', '-DDFN2_N_ERB=24', '-DDFN2_DF_BINS=96'],
    'custom16k_df48': ['-DDFN2_SR=16000', '-DDFN2_N_FFT=512', '-DDFN2_N_ERB=20',
                       '-DDFN2_DF_BINS=48'],
    # every band exactly two bins wide but the last: 2 * n_erb = n_bins - 1
    'narrow_bands': ['-DDFN2_SR=16000', '-DDFN2_N_FFT=64', '-DDFN2_N_ERB=16', '-DDFN2_DF_BINS=32'],
    'mask_only': ['-DDFN2_MASK_ONLY'],
}


def _compile_test_grid_tu(work, header_dir, flags, name='tu'):
    source = work / (name + '.c')
    source.write_text(_TEST_GRID_TU, encoding='utf-8')
    return _cc([*dfn_pairs.house_cflags(1), '-fsyntax-only', '-I', str(header_dir), *flags,
                str(source)])


@pytest.mark.parametrize('grid', list(_TEST_GRIDS))
def test_the_test_grid_header_stands_alone(tmp_path, grid):
    """dfn2_test_grid.h, included alone, compiles under the house flags (-Wall -Wextra -Wpedantic
    -Werror) for every preset, a custom grid and mask-only. Its scale macros call sqrtf, so the
    header includes <math.h> itself instead of relying on what the library headers pull in."""
    done = _compile_test_grid_tu(tmp_path, TESTS_DIR, _TEST_GRIDS[grid])
    assert done.returncode == 0, done.stderr


def test_the_stand_alone_gate_can_fail(tmp_path):
    """The same translation unit against a copy of the header without its <math.h> include does
    not compile (sqrtf is undeclared)."""
    text = (TESTS_DIR / 'dfn2_test_grid.h').read_text(encoding='utf-8')
    assert text.count('#include <math.h>\n') == 1
    (tmp_path / 'dfn2_test_grid.h').write_text(text.replace('#include <math.h>\n', ''),
                                              encoding='utf-8')
    done = _compile_test_grid_tu(tmp_path, tmp_path, [])
    assert done.returncode != 0 and 'sqrtf' in done.stderr, done.stderr


_BORDERS_MAIN = r'''
#include <stdio.h>
#include "dfn2_test_grid.h"

int main(void)
{
    int borders[DFN2_N_ERB], b, narrowest = DFN2_N_BINS;
    dfn2_test_band_borders(borders);
    if (borders[0] != 0 || borders[DFN2_N_ERB - 1] != DFN2_N_BINS) return 1;
    for (b = 0; b + 1 < DFN2_N_ERB; ++b) {
        int width = borders[b + 1] - borders[b];
        if (width < 2) return 2;
        if (width < narrowest) narrowest = width;
    }
    printf("borders n_bins=%d n_erb=%d narrowest=%d\n", DFN2_N_BINS, DFN2_N_ERB, narrowest);
    return 0;
}
'''


@pytest.mark.parametrize('grid', [g for g in _TEST_GRIDS if g != 'mask_only'])
def test_the_band_borders_the_c_suites_share_cover_every_grid(builds, grid):
    """dfn2_test_band_borders() (the one generator test_c_prepost.c and test_nn_chains.c use):
    from bin 0 to the last bin, every band at least two bins wide, on the shipped table and on
    the generated one, down to a grid whose bands are all two bins wide."""
    source = builds.work / ('borders_%s.c' % grid)
    source.write_text(_BORDERS_MAIN, encoding='utf-8')
    exe = builds.work / ('borders_' + grid)
    done = _cc([*dfn_pairs.house_cflags(1), '-I', str(TESTS_DIR), *_TEST_GRIDS[grid], str(source),
                dfn_pairs.audio_common_lib(1), '-lm', '-o', str(exe)])
    assert done.returncode == 0, done.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, (ran.returncode, ran.stdout, ran.stderr)
    narrowest = int(re.search(r'narrowest=(\d+)', ran.stdout).group(1))
    assert narrowest >= 2
    if grid == 'narrow_bands':
        assert narrowest == 2, ran.stdout


# ---------------------------------------------------------------------------
# the variant link symbol
# ---------------------------------------------------------------------------

_LINK_MAIN = r'''
#include "dfn2_prepost.h"
int main(void) {
    DFN2Grid grid;
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_grid_make(&grid, 16000, 256, 24, 96, 1, 0) != 0) return 1;
    if (dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ, &grid) != 0) return 1;
    return dfn2_prepost_get_mem_size(&cfg, &req);
}
'''

_APP_MAIN = r'''
#include "dfn2_process.h"
int main(void) {
    DFN2_REQUIRE_VARIANT_LINK();   /* what an application calls in main() */
    return 0;
}
'''

# link-test variants: name -> (flags, symbol the units reference)
_VARIANTS = {
    'full': ([], 'dfn2_variant_full_n1024_e32_d96_r5'),
    'mask': ([_MASK], 'dfn2_variant_mask_only_n1024_e32_d96_r2'),
    'fft2048': (['-DDFN2_MAX_N_FFT=2048'], 'dfn2_variant_full_n2048_e32_d96_r5'),
    'erb36': (['-DDFN2_MAX_N_ERB=36'], 'dfn2_variant_full_n1024_e36_d96_r5'),
    'df128': (['-DDFN2_MAX_DF_BINS=128'], 'dfn2_variant_full_n1024_e32_d128_r5'),
    'ring7': (['-DDFN2_MAX_DF_RING=7'], 'dfn2_variant_full_n1024_e32_d96_r7'),
    'mask_ring3': ([_MASK, '-DDFN2_MAX_DF_RING=3'], 'dfn2_variant_mask_only_n1024_e32_d96_r3'),
    # Mask-only with every capacity equal to the full default: the variant alone tells them apart.
    'mask_ring5': ([_MASK, '-DDFN2_MAX_DF_RING=5'], 'dfn2_variant_mask_only_n1024_e32_d96_r5'),
}


def _assert_no_link(done, symbol):
    """The link failed and the linker names `symbol` as the one it cannot find."""
    assert done.returncode != 0, 'the objects linked'
    assert re.search(re.escape(symbol) + r'\b', done.stderr), done.stderr


@pytest.fixture(scope='module')
def variants(builds):
    builds.prebuild([(DFN2, flags, 1) for flags, _s in _VARIANTS.values()], chains=False)
    return {name: builds.objects(flags) for name, (flags, _s) in _VARIANTS.items()}


def test_units_of_one_variant_link(builds, variants):
    for name, objs in variants.items():
        done, _exe = builds.link(_LINK_MAIN, objs, 'same_' + name, _VARIANTS[name][0])
        assert done.returncode == 0, (name, done.stderr)


@pytest.mark.parametrize('other', [n for n in _VARIANTS if n != 'full'])
def test_units_of_different_variants_do_not_link(builds, variants, other):
    """A mask-only unit or one built with another capacity names a symbol the full default
    process unit does not define, and the other way round; two prepost units of different
    builds under one process unit fail too."""
    base, odd = variants['full'], variants[other]
    done, _exe = builds.link(_LINK_MAIN, [odd[0], *base[1:]], 'mix_a_' + other)
    _assert_no_link(done, _VARIANTS['full'][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], *odd[1:]], 'mix_b_' + other, _VARIANTS[other][0])
    _assert_no_link(done, _VARIANTS[other][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], base[1], odd[2]], 'mix_c_' + other,
                             _VARIANTS[other][0])
    _assert_no_link(done, _VARIANTS[other][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], odd[1], base[2]], 'mix_d_' + other,
                             _VARIANTS[other][0])
    _assert_no_link(done, _VARIANTS[other][1])


def test_the_symbol_carries_the_variant_and_the_capacities_but_not_the_grid(builds, variants):
    """Every variant defines exactly its own symbol; the grid and the lookahead pair are in no
    symbol (one build serves every grid)."""
    if shutil.which('nm') is None:
        pytest.skip('nm unavailable')
    for name, objs in variants.items():
        listing = subprocess.run(['nm', str(objs[0])], capture_output=True, text=True).stdout
        defined = re.findall(r'\b_?(dfn2_variant_\w+)', listing)
        assert defined == [_VARIANTS[name][1]], (name, defined)
    assert len({_VARIANTS[n][1] for n in _VARIANTS}) == len(_VARIANTS)


def test_an_application_built_for_another_variant_than_the_library_does_not_link(
        builds, variants):
    done, _exe = builds.link(_APP_MAIN, variants['full'], 'app_full')
    assert done.returncode == 0, done.stderr                            # the control
    for other in ('mask', 'fft2048', 'ring7'):
        flags, symbol = _VARIANTS[other]
        done, _exe = builds.link(_APP_MAIN, variants['full'], 'app_' + other, flags)
        _assert_no_link(done, symbol)
        done, _exe = builds.link(_APP_MAIN, variants[other], 'app_full_on_' + other)
        _assert_no_link(done, _VARIANTS['full'][1])
