"""The DFN2 library holds no grid: the run-time grid, proved end to end.

The grid is an argument of every object; one build of the library serves any grid its
checks accept. The tests here pin the identities that keep two grids from being mixed up
and the property that the library holds no grid of its own:

  chain                  the whole pre/post chain (``dfn2_runtime_grid_chain.c``: a float host
                         with registered recurrent state, a stand-in accelerator, refused heads
                         and state, skipped and unwritten frames) runs on every grid and
                         lookahead pair, NEON and scalar, full and mask-only, and every one
                         prints its own digest.
  pool identity          get_mem_size differs between grids, lookahead pairs, full and
                         mask-only and each normaliser field; init_ex refuses a MemReq of
                         another grid or pair; a descriptor of one grid is refused by the host
                         of another. Pool bytes and hash equal an independent Python mirror of
                         the layout rules (``_PoolMirror``), not a recorded constant.
  normaliser fields      the seven normaliser settings of the spec are in the pool hash alone
                         and reach the state ``dfn2_state_init()`` starts
  registered state       the all-or-none registration rule, the board form (nothing registered:
                         generation moves as documented, no foreign buffer touched) and the stale
                         pool gate (a refused init_ex leaves the pool and the state unchanged)
  no grid in the library no dfn2_*.c / .h defines or uses a grid macro (those are the consumer
                         headers' business); the test grid header stands alone and its band-border
                         generator, shared by the C suites, holds on every grid
  variant link symbol    objects built for mask-only and full do not link; the grid and the
                         pair are not in the symbol

The digests of the shipped grids are printed for the record; they are not pinned, because the
chain runs log10f / powf of the C library it is linked with.
"""

import concurrent.futures
import itertools
import math
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
# dfn2_*.c/.h moved to lib/ainr/dfn2/ in the single-model-release restructuring; DFN2 (still
# AINR/DeepFilterNet2, the Python training dir) now holds only config.ini and forwarding stubs.
DFN2_C_DIR = dfn_pairs.AINR.parent / 'lib' / 'ainr' / 'dfn2'
CHAIN_SOURCE = TESTS_DIR / 'dfn2_runtime_grid_chain.c'
UNITS = ('process', 'model_io', 'prepost')
SIMD_MODES = (1, 0)

pytestmark = pytest.mark.skipif(
    dfn_pairs.require_toolchain() is not None,
    reason='C build tools unavailable')

_MASK = '-DDFN2_MASK_ONLY'

# The two library builds: the full model and the mask-only one.
CONFIGS = {'full': [], 'mask': [_MASK]}

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


def _cc(args, **kwargs):
    return subprocess.run(['cc', *args], capture_output=True, text=True, **kwargs)


class _Builds:
    """Everything the tests compile, in one directory and memoised: the three library units per
    (flags, SIMD mode), and the programs linked against them."""

    def __init__(self, work):
        self.work = work
        self._counter = itertools.count(1)
        self._objects = {}
        self._exes = {}
        self._runs = {}

    def _compile_unit(self, unit, flags, simd):
        stem = 'unit%d_%s' % (next(self._counter), unit)
        obj = self.work / (stem + '.o')
        source = DFN2_C_DIR / ('dfn2_%s.c' % unit)
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(DFN2), '-c', str(source),
                    '-o', str(obj)])
        assert done.returncode == 0, done.stderr
        return obj

    def prebuild(self, specs, chains=True):
        """Compile the units of every (flags, simd) in `specs` and, with `chains`, link the
        chain driver against them, a few at a time."""
        dfn_pairs.house_cflags(1), dfn_pairs.house_cflags(0)   # resolved before the threads
        dfn_pairs.audio_common_lib(1), dfn_pairs.audio_common_lib(0)
        keys = []
        for flags, simd in specs:
            key = (tuple(flags), simd)
            if key not in keys and key not in self._objects:
                keys.append(key)
        jobs = [(key, unit) for key in keys for unit in UNITS]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            built = list(pool.map(
                lambda job: self._compile_unit(job[1], job[0][0], job[0][1]), jobs))
        for index, key in enumerate(keys):
            self._objects[key] = built[index * len(UNITS):(index + 1) * len(UNITS)]
        if chains:
            todo = list(dict.fromkeys((tuple(f), s) for f, s in specs
                                      if ('chain', tuple(f), s) not in self._exes))
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                exes = list(pool.map(lambda t: self._link_chain(*t), todo))
            for spec, exe in zip(todo, exes):
                self._exes[('chain', *spec)] = exe

    def objects(self, flags, simd=1):
        key = (tuple(flags), simd)
        if key not in self._objects:
            self.prebuild([(flags, simd)], chains=False)
        return list(self._objects[key])

    def _link_chain(self, flags, simd):
        objs = self._objects[(tuple(flags), simd)]
        exe = self.work / ('chain%d' % next(self._counter))
        done = _cc([*dfn_pairs.house_cflags(simd), *flags, '-I', str(DFN2), str(CHAIN_SOURCE),
                    *map(str, objs), dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(exe)])
        assert done.returncode == 0, done.stderr
        return exe

    def chain(self, flags, simd=1):
        key = ('chain', tuple(flags), simd)
        if key not in self._exes:
            self.prebuild([(flags, simd)])
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

    def run_chain(self, config, simd, grid, pair, *options):
        """The output of the chain driver of library build `config` on a grid and a pair."""
        key = (config, simd, grid, pair, options)
        if key not in self._runs:
            exe = self.chain(CONFIGS[config], simd)
            done = subprocess.run([str(exe), *map(str, grid), *map(str, pair), *options],
                                  capture_output=True, text=True)
            assert done.returncode == 0, (config, grid, pair, done.stdout, done.stderr)
            self._runs[key] = done.stdout
        return self._runs[key]


def _digest(output):
    match = re.search(r'chain digest=([0-9a-f]{16})', output)
    assert match, output
    return match.group(1)


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    return _Builds(tmp_path_factory.mktemp('dfn2_runtime_grid'))


@pytest.fixture(scope='module')
def matrix(builds):
    """Both library builds, both SIMD modes, with the chain driver."""
    builds.prebuild([(flags, simd) for flags in CONFIGS.values() for simd in SIMD_MODES])
    return builds


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
def test_the_chain_tells_grids_and_pairs_apart(matrix, simd):
    """Not a vacuous run: every grid and every lookahead pair gives its own digest, and the
    mask-only chain is not the full one."""
    seen = {}
    for grid_name, grid in GRIDS.items():
        for pair in FULL_PAIRS:
            digest = _digest(matrix.run_chain('full', simd, grid, pair))
            assert digest not in seen, (grid_name, pair, seen[digest])
            seen[digest] = (grid_name, pair)
        for pair in MASK_PAIRS:
            digest = _digest(matrix.run_chain('mask', simd, grid, pair))
            assert digest not in seen, (grid_name, pair, seen[digest])
            seen[digest] = ('mask-only ' + grid_name, pair)
    assert len(seen) == len(GRIDS) * (len(FULL_PAIRS) + len(MASK_PAIRS))


@pytest.mark.parametrize('simd', SIMD_MODES, ids=['neon', 'scalar'])
def test_the_chain_reads_every_spec_field(matrix, simd):
    """The optional spec fields of the driver (order, taus, starts, scale) reach the chain:
    each one alone changes the digest of the shipped grid."""
    grid, pair = GRIDS['g16'], (1, 1)
    base = _digest(matrix.run_chain('full', simd, grid, pair))
    changed = {}
    for option in ('order=4', 'order=7', 'erb_tau=0.5', 'erb_lo=-70', 'erb_hi=-80',
                   'erb_scale=30', 'spec_tau=0.5', 'spec_lo=0.002', 'spec_hi=0.0002'):
        digest = _digest(matrix.run_chain('full', simd, grid, pair, option))
        assert digest != base, option
        changed[option] = digest
    assert len(set(changed.values())) == len(changed), changed


def test_the_chain_driver_rejects_bad_arguments(matrix):
    exe = matrix.chain(CONFIGS['full'], 1)
    for args in ([], ['1', '2', '3'], ['16000', '256', '24', '96', '1', 'x'],
                 ['16000', '256', '24', '96', '1'],
                 ['16000', '256', '24', '96', '1', '1', 'bogus=1'],
                 ['16000', '256', '24', '96', '1', '1', 'order=x'],
                 ['16000', '256', '24', '96', '1', '1', 'frames=2']):
        assert subprocess.run([str(exe), *args], capture_output=True,
                              text=True).returncode == 2, args
    done = subprocess.run([str(exe), '16000', '960', '24', '96', '1', '1'],
                          capture_output=True, text=True)
    assert done.returncode == 3 and 'audiolib' in done.stdout, done.stdout
    done = subprocess.run([str(exe), '16000', '256', '24', '96', '1', '1', 'erb_tau=0'],
                          capture_output=True, text=True)
    assert done.returncode == 3 and 'grid refused' in done.stdout, done.stdout


def test_the_digests_of_the_shipped_grids_are_printed_for_the_record(matrix, capsys):
    """Not pinned: the chain runs log10f / powf of the C library it is linked with."""
    rows = [('g48_1_1', 'full', 'g48', (1, 1)), ('g16_0_0', 'full', 'g16', (0, 0)),
            ('g32_0_0', 'full', 'g32', (0, 0)), ('m48_1_0', 'mask', 'g48', (1, 0))]
    with capsys.disabled():
        print()
        for label, config, grid_name, pair in rows:
            for simd in SIMD_MODES:
                out = matrix.run_chain(config, simd, GRIDS[grid_name], pair)
                backend = re.search(r'backend=(\S+)', out).group(1)
                print('  chain digest %-8s %-6s %s  (%s)' % (
                    label, 'neon' if simd else 'scalar', _digest(out), backend))


# ---------------------------------------------------------------------------
# the pool identity, the descriptor and the host's grid check
# ---------------------------------------------------------------------------

_POOL_MAIN = r"""
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dfn2_prepost.h"

#define MAX_BINS 513
#define MAX_ERB 40
static float fwd[MAX_BINS * MAX_ERB], inv[MAX_BINS * MAX_ERB];
static float zeros[MAX_BINS];
static int infer_calls;

/* The seven normaliser settings and the order of the shipped model (config.ini), the spec's
 * other fields for every grid; "[order tau lo hi scale tau lo hi]" after a grid overrides them. */
static const float defaults[8] = { @ORDER@, @ERB_TAU@, @ERB_LO@, @ERB_HI@, @ERB_SCALE@,
                                   @SPEC_TAU@, @SPEC_LO@, @SPEC_HI@ };

static int read_grid(const char *text, DFN2Grid *g, int *used) {
    DFN2GridSpec spec;
    float v[8];
    int a, b, c, d, e, f, n = 0, m = 0, k;
    if (sscanf(text, "%d %d %d %d %d %d%n", &a, &b, &c, &d, &e, &f, &n) != 6) return -2;
    for (k = 0; k < 8; ++k) v[k] = defaults[k];
    while (text[n] == ' ') ++n;
    if (text[n] == '[') {
        float t[8];
        if (sscanf(text + n + 1, "%f %f %f %f %f %f %f %f%n", &t[0], &t[1], &t[2], &t[3], &t[4],
                   &t[5], &t[6], &t[7], &m) != 8) return -2;
        for (k = 0; k < 8; ++k) v[k] = t[k];
        n += 1 + m;
        while (text[n] == ' ') ++n;
        if (text[n] != ']') return -2;
        ++n;
    }
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = a;
    spec.n_fft = b;
    spec.n_erb = c;
    spec.df_bins = d;
    spec.df_order = (int)v[0];
    spec.mask_lookahead = e;
    spec.df_lookahead = f;
    spec.frames_per_call = 1;
    spec.erb_norm_tau_sec = v[1];
    spec.erb_norm_init_lo_db = v[2];
    spec.erb_norm_init_hi_db = v[3];
    spec.erb_norm_scale_db = v[4];
    spec.spec_norm_tau_sec = v[5];
    spec.spec_norm_init_lo = v[6];
    spec.spec_norm_init_hi = v[7];
    *used = n;
    return dfn2_grid_make(g, &spec);
}

static void fill_tables(const DFN2Grid *g) {
    size_t i;
    for (i = 0; i < (size_t)g->n_bins * g->n_erb; ++i) fwd[i] = inv[i] = 0.01f;
}

static int config_for(const DFN2Grid *g, DFN2PrepostConfig *cfg) {
    if (dfn2_prepost_config_defaults(cfg, g) != 0) return -1;
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


/* The recurrent state of the checks below: buffers of the program's own. The library takes their element counts
 * as given and only zeroes and scans them, so small counts stand for the graph's tensors. */
#define GRU_N 40u
#define CONVP_N 60u
static float s_gru[GRU_N], s_convp[CONVP_N];

static void state_fill(float v) {
    size_t i;
    for (i = 0; i < GRU_N; ++i) s_gru[i] = v;
    for (i = 0; i < CONVP_N; ++i) s_convp[i] = v;
}
static int gru_is(float v) {
    size_t i;
    for (i = 0; i < GRU_N; ++i) if (s_gru[i] != v) return 0;
    return 1;
}
static int convp_is(float v) {
    size_t i;
    for (i = 0; i < CONVP_N; ++i) if (s_convp[i] != v) return 0;
    return 1;
}
static int state_all(float v) { return gru_is(v) && convp_is(v); }
/* Zero in every tensor the build registers (the mask-only build has no pathway window). */
static int registered_zero(void) {
#ifndef DFN2_MASK_ONLY
    return gru_is(0.0f) && convp_is(0.0f);
#else
    return gru_is(0.0f);
#endif
}
/* The state a float host registers: h_gru, and with the full model the pathway window. */
static void register_state(DFN2PrepostConfig *cfg) {
    cfg->gru_hidden = s_gru;
    cfg->gru_hidden_elements = GRU_N;
#ifndef DFN2_MASK_ONLY
    cfg->df_convp_history = s_convp;
    cfg->df_convp_history_elements = CONVP_N;
#endif
}
static int infer_fail(void *user, const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    (void)user;
    (void)in;
    (void)out;
    return -1;
}
static int infer_nan(void *user, const DFN2PrepostInputs *in, DFN2PrepostOutputs *out) {
    infer_ok(user, in, out);
    out->erb_mask[0] = NAN;
    return 0;
}

/* One analysed frame of zeros; every hop is closed with post_process, as the call order says. */
static int hop_in(DFN2Prepost *p) { return dfn2_prepost_pre_process_freq(p, zeros, zeros); }
static int hop_out(DFN2Prepost *p) {
    float re[MAX_BINS], im[MAX_BINS];
    return dfn2_prepost_post_process_freq(p, re, im, NULL);
}
/* The state views of an open frame, every registered element set to `v` (NULL views: nothing to set). */
static void scribble_state(const DFN2PrepostInputs *in, const DFN2PrepostOutputs *out, float v) {
    size_t i;
    (void)in;
    for (i = 0; out->gru_hidden_next != NULL && i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = v;
#ifndef DFN2_MASK_ONLY
    for (i = 0; out->df_convp_window_next != NULL && i < out->df_convp_window_elements; ++i)
        out->df_convp_window_next[i] = v;
#endif
}
static int views_zero(const DFN2PrepostInputs *in) {
    size_t i;
    for (i = 0; in->gru_hidden != NULL && i < in->gru_hidden_elements; ++i) if (in->gru_hidden[i] != 0.0f) return 0;
#ifndef DFN2_MASK_ONLY
    for (i = 0; in->df_convp_window != NULL && i < in->df_convp_window_elements; ++i)
        if (in->df_convp_window[i] != 0.0f) return 0;
#endif
    return 1;
}
/* Hops of a cold or reset instance until the heads of a real frame are due: the first mask_lookahead hops are
 * warm-up frames (pre 1, open, the stand-in writes heads and scribbles on the registered state, finish discards
 * them and leaves that state zero, the hop closes); the last hop's pre_process result (1) is returned, else -1. */
static int to_ready(DFN2Prepost *p) {
    int i;
    for (i = 0; i < dfn2_prepost_grid(p)->mask_lookahead; ++i) {
        DFN2PrepostInputs in;
        DFN2PrepostOutputs out;
        if (hop_in(p) != 1 || dfn2_prepost_frame_open(p, &in, &out) != 0) return -1;
        infer_ok(NULL, &in, &out);
        scribble_state(&in, &out, 9.0f);
        if (dfn2_prepost_frame_finish(p) != 0 || !views_zero(&in) || hop_out(p) != 0) return -1;
    }
    return hop_in(p);
}
/* A ready frame through the accelerator stand-in: 1 when the views are the given buffers (NULL: none), the frame
 * commits and the hop closes. */
static int frame_commits(DFN2Prepost *p, const float *gru, const float *convp) {
    DFN2PrepostInputs in;
    DFN2PrepostOutputs out;
    if (to_ready(p) != 1 || dfn2_prepost_frame_open(p, &in, &out) != 0) return 0;
    if (in.gru_hidden != gru || out.gru_hidden_next != gru) return 0;
#ifndef DFN2_MASK_ONLY
    if (in.df_convp_window != convp || out.df_convp_window_next != convp) return 0;
#else
    (void)convp;
#endif
    infer_ok(NULL, &in, &out);
    return dfn2_prepost_frame_finish(p) == 0 && hop_out(p) == 0;
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
            /* req <grid>: get_mem_size of the config (no state registered) */
            DFN2PrepostConfig cfg;
            DFN2PrepostMemReq r;
            memset(&r, 0, sizeof r);
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
            bad = rb; bad.reserved0 ^= 1u; tampered += !accepted(pool, bytes, &cb, &bad);
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
            int rc, ready;
            if (read_grid(line + strlen(verb) + 1 + used, &b, &used2) != 0 ||
                config_for(&b, &cb) != 0) {
                printf("%s badgrid2\n", verb);
                continue;
            }
            p = dfn2_prepost_create(&cb);
            ready = to_ready(p);   /* past the warm-up frames: a real frame is open */
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
            /* descfield <grid>: every one of the 18 int fields changed by one; the 12 external
             * ones are refused, the 6 interior ones (caller-filled bookkeeping) are not read */
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
            /* starts <grid>: what dfn2_state_init() puts into the two normalisers: the first and
             * last band of the ERB one, the first and last bin of the spectrum one */
            static DFN2State dsp;
            size_t bytes = dfn2_state_mem_size(&a);
            void *block = aligned_alloc(16, bytes);
            if (!block || dfn2_state_init(&dsp, block, bytes, &a) != 0) {
                printf("starts noinit\n");
                free(block);
                continue;
            }
            printf("starts %.9g %.9g %.9g %.9g\n", (double)dsp.erb_norm_state[0],
                   (double)dsp.erb_norm_state[a.n_erb - 1], (double)dsp.spec_norm_state[0],
                   (double)dsp.spec_norm_state[a.df_bins - 1]);
            free(block);
        } else if (!strcmp(verb, "nullcalls")) {
            DFN2ModelIoDescriptor d;
            DFN2PrepostConfig cfg;
            printf("nullcalls %d %d %d %d\n", dfn2_model_io_descriptor_default(NULL, &d),
                   dfn2_model_io_descriptor_validate(NULL, &d),
                   dfn2_model_io_descriptor_validate(&a, NULL),
                   dfn2_prepost_config_defaults(&cfg, NULL));
        } else if (!strcmp(verb, "badgrid")) {
            /* a grid edited by hand is refused by every entry point that takes one; the blocks
             * are the right size for the unedited grid, so the grid is what is refused */
            DFN2PrepostConfig cfg;
            DFN2ModelIoDescriptor d;
            DFN2ModelIOState io;
            DFN2State dsp;
            DFN2Grid edited = a;
            size_t state_bytes = dfn2_state_mem_size(&a), io_bytes = dfn2_model_io_mem_size(&a);
            void *state_block = aligned_alloc(16, state_bytes);
            void *io_block = aligned_alloc(16, io_bytes);
            edited.ring += 1;
            printf("badgrid %d %d %d %d\n",
                   dfn2_prepost_config_defaults(&cfg, &edited),
                   dfn2_model_io_descriptor_default(&edited, &d),
                   dfn2_model_io_init(&io, io_block, io_bytes, &edited, NULL, 0, NULL, 0),
                   dfn2_state_init(&dsp, state_block, state_bytes, &edited));
            free(state_block);
            free(io_block);
        } else if (!strcmp(verb, "reg")) {
            /* reg <grid>  <bits>: bit 0 = h_gru pointer, 1 = h_gru count, 2 = pathway pointer, 3 = pathway count.
             * Prints: the get_mem_size status, init, create, clean (a refusal changed neither the request,
             * the pool nor the state; an acceptance kept the pool size and identity), works (generation 1, each
             * tensor zeroed exactly when registered, the views, a frame that commits). */
            DFN2PrepostConfig good, cfg;
            DFN2PrepostMemReq base, req, untouched;
            DFN2Prepost *p, *made;
            unsigned int bits = 0;
            int status, clean, works = 0;
            size_t bytes;
            unsigned char *pool, *copy;
            if (sscanf(line + strlen(verb) + 1 + used, "%u", &bits) != 1 || bits > 15u ||
                config_for(&a, &good) != 0 || dfn2_prepost_get_mem_size(&good, &base) != 0) {
                printf("reg badargs\n");
                continue;
            }
            bytes = (size_t)base.bytes;
            pool = aligned_alloc(16, (bytes + 15u) / 16u * 16u);
            copy = malloc(bytes);
            cfg = good;
            cfg.gru_hidden = (bits & 1u) ? s_gru : NULL;
            cfg.gru_hidden_elements = (bits & 2u) ? GRU_N : 0u;
            cfg.df_convp_history = (bits & 4u) ? s_convp : NULL;
            cfg.df_convp_history_elements = (bits & 8u) ? CONVP_N : 0u;
            memset(&untouched, 0xa5, sizeof untouched);
            req = untouched;
            state_fill(7.0f);
            memset(pool, 0xa5, bytes);
            memcpy(copy, pool, bytes);
            status = dfn2_prepost_get_mem_size(&cfg, &req);
            p = dfn2_prepost_init(pool, bytes, &cfg);
            if (p) {
                const int gru_in = (bits & 3u) == 3u, convp_in = (bits >> 2) == 3u;
                clean = status == 0 && req.bytes == base.bytes && req.build_flags_hash == base.build_flags_hash;
                works = dfn2_prepost_state_generation(p) == 1u && gru_is(gru_in ? 0.0f : 7.0f) &&
                       convp_is(convp_in ? 0.0f : 7.0f) &&
                       frame_commits(p, gru_in ? s_gru : NULL, convp_in ? s_convp : NULL);
            } else {
                clean = status == -1 && memcmp(&req, &untouched, sizeof req) == 0 &&
                        memcmp(pool, copy, bytes) == 0 && state_all(7.0f);
            }
            made = dfn2_prepost_create(&cfg);
            printf("reg %u %d %d %d %d %d\n", bits, status, p != NULL, made != NULL, clean, works);
            dfn2_prepost_destroy(made);
            free(pool);
            free(copy);
        } else if (!strcmp(verb, "board")) {
            /* board <grid>: nothing registered, on a pool that an earlier instance used with registered state.
             * Prints the generation after init, whether the views are empty, the generation after a committed
             * frame, a bypassed one, a refused one, a second finish, the bypass, a reset and then three
             * model_run_frame calls (infer fails, a NaN head, commits), whether every status was the documented
             * one, and whether every buffer the library was not handed (a guarded foreign one and the earlier
             * instance's state) is intact. */
            static struct {
                unsigned char head[32];
                float gru[GRU_N], convp[CONVP_N];
                unsigned char tail[32];
            } foreign, foreign_ref;
            DFN2PrepostConfig board, reg;
            DFN2PrepostMemReq req;
            DFN2PrepostInputs in;
            DFN2PrepostOutputs out;
            DFN2ModelIoDescriptor desc;
            DFN2Model model;
            DFN2Prepost *p;
            unsigned int g[10];
            int views, statuses, guard, commit_rc, bypass_rc, refuse_rc, again_rc, bypass2_rc, fail_rc, nan_rc, good_rc;
            void *pool;
            memset(&board, 0xa5, sizeof board);
            if (dfn2_prepost_config_defaults(&board, &a) != 0 ||
                dfn2_model_io_descriptor_default(&a, &desc) != 0) {
                printf("board nodefaults\n");
                continue;
            }
            fill_tables(&a);
            board.erb_fwd = fwd;
            board.erb_inv = inv;
            reg = board;
            register_state(&reg);
            if (dfn2_prepost_get_mem_size(&board, &req) != 0 || !(pool = aligned_alloc(16, (size_t)req.bytes))) {
                printf("board nosize\n");
                continue;
            }
            dfn2_prepost_init(pool, (size_t)req.bytes, &reg);
            state_fill(5.0f);
            memset(&foreign, 0x5a, sizeof foreign);
            memset(foreign.head, 0xa5, sizeof foreign.head);
            memset(foreign.tail, 0xa5, sizeof foreign.tail);
            memcpy(&foreign_ref, &foreign, sizeof foreign);
            p = dfn2_prepost_init(pool, (size_t)req.bytes, &board);
            g[0] = dfn2_prepost_state_generation(p);
            to_ready(p);
            dfn2_prepost_frame_open(p, &in, &out);
            views = in.gru_hidden == NULL && in.gru_hidden_elements == 0u && out.gru_hidden_next == NULL &&
                    out.gru_hidden_elements == 0u;
#ifndef DFN2_MASK_ONLY
            views = views && in.df_convp_window == NULL && in.df_convp_window_elements == 0u &&
                    out.df_convp_window_next == NULL && out.df_convp_window_elements == 0u;
#endif
            infer_ok(NULL, &in, &out);
            commit_rc = dfn2_prepost_frame_finish(p);
            hop_out(p);
            g[1] = dfn2_prepost_state_generation(p);
            hop_in(p);
            dfn2_prepost_frame_open(p, &in, &out);
            bypass_rc = dfn2_prepost_frame_bypass(p);
            hop_out(p);
            g[2] = dfn2_prepost_state_generation(p);
            hop_in(p);
            dfn2_prepost_frame_open(p, &in, &out);                   /* the heads stay NaN: refused */
            refuse_rc = dfn2_prepost_frame_finish(p);
            g[3] = dfn2_prepost_state_generation(p);
            again_rc = dfn2_prepost_frame_finish(p);
            g[4] = dfn2_prepost_state_generation(p);
            bypass2_rc = dfn2_prepost_frame_bypass(p);
            hop_out(p);
            g[5] = dfn2_prepost_state_generation(p);
            dfn2_prepost_reset(p);
            g[6] = dfn2_prepost_state_generation(p);
            model.user = NULL;
            model.reset = NULL;
            model.io_descriptor = &desc;
            model.infer = infer_fail;
            to_ready(p);
            fail_rc = dfn2_model_run_frame(&model, p);
            hop_out(p);
            g[7] = dfn2_prepost_state_generation(p);
            model.infer = infer_nan;
            hop_in(p);
            nan_rc = dfn2_model_run_frame(&model, p);
            hop_out(p);
            g[8] = dfn2_prepost_state_generation(p);
            model.infer = infer_ok;
            hop_in(p);
            good_rc = dfn2_model_run_frame(&model, p);
            hop_out(p);
            g[9] = dfn2_prepost_state_generation(p);
            statuses = commit_rc == 0 && bypass_rc == 0 && refuse_rc == -1 && again_rc == -1 && bypass2_rc == 0 &&
                       fail_rc == 0 && nan_rc == 0 && good_rc == 1;
            guard = memcmp(&foreign, &foreign_ref, sizeof foreign) == 0 && state_all(5.0f);
            printf("board %u %d %u %u %u %u %u %u %u %u %u %d %d\n", g[0], views, g[1], g[2], g[3], g[4], g[5],
                   g[6], g[7], g[8], g[9], statuses, guard);
            free(pool);
        } else if (!strcmp(verb, "stale")) {
            /* stale <grid A>  <grid B>: a live instance of B holds the pool; A's config is offered B's MemReq, a
             * forged copy of it (A's size and alignment, so the identity alone differs), B's config is offered A's,
             * and A's config is offered A's own MemReq with each compared field changed. Prints the refusals out
             * of 9, whether the pool and the registered state are byte-identical afterwards, whether the B
             * instance still reports its grid, and whether A's own MemReq then starts an instance that commits. */
            DFN2PrepostConfig ca, cb;
            DFN2PrepostMemReq ra, rb, bad;
            DFN2Prepost *live;
            size_t most;
            unsigned char *pool, *copy;
            int refused = 0, untouched, alive, field, control;
            if (read_grid(line + strlen(verb) + 1 + used, &b, &used2) != 0 || config_for(&a, &ca) != 0 ||
                config_for(&b, &cb) != 0 || dfn2_prepost_get_mem_size(&ca, &ra) != 0 ||
                dfn2_prepost_get_mem_size(&cb, &rb) != 0) {
                printf("stale nosize\n");
                continue;
            }
            register_state(&ca);
            register_state(&cb);
            most = (size_t)(ra.bytes > rb.bytes ? ra.bytes : rb.bytes);
            most = (most + 15u) / 16u * 16u;
            pool = aligned_alloc(16, most);
            copy = malloc(most);
            memset(pool, 0xa5, most);
            live = dfn2_prepost_init_ex(pool, most, &cb, &rb);
            state_fill(7.0f);
            memcpy(copy, pool, most);
            refused += !accepted(pool, most, &ca, &rb);
            bad = rb;
            bad.bytes = ra.bytes;
            bad.alignment = ra.alignment;
            refused += !accepted(pool, most, &ca, &bad);
            refused += !accepted(pool, most, &cb, &ra);
            for (field = 0; field < 6; ++field) {
                bad = ra;
                switch (field) {
                case 0: bad.descriptor_version += 1u; break;
                case 1: bad.layout_version += 1u; break;
                case 2: bad.reserved0 = 1u; break;
                case 3: bad.build_flags_hash ^= 1u; break;
                case 4: bad.alignment *= 2u; break;
                default: bad.bytes += 1u; break;
                }
                refused += !accepted(pool, most, &ca, &bad);
            }
            untouched = memcmp(pool, copy, most) == 0 && state_all(7.0f);
            alive = live != NULL && dfn2_prepost_hop_size(live) == b.hop && dfn2_prepost_num_bins(live) == b.n_bins;
            live = dfn2_prepost_init_ex(pool, most, &ca, &ra);
            control = live != NULL && registered_zero() && frame_commits(live, s_gru, s_convp);
            printf("stale %d %d %d %d\n", refused, untouched, alive, control);
            free(pool);
            free(copy);
        }
    }
    return 0;
}
"""


def _pool_main():
    values = dfn_pairs.dfn2_spec_values()
    text = _POOL_MAIN
    for token, key in (('ERB_TAU', 'erb_norm_tau_sec'), ('ERB_LO', 'erb_norm_init_lo_db'),
                       ('ERB_HI', 'erb_norm_init_hi_db'), ('ERB_SCALE', 'erb_norm_scale_db'),
                       ('SPEC_TAU', 'spec_norm_tau_sec'), ('SPEC_LO', 'spec_norm_init_lo'),
                       ('SPEC_HI', 'spec_norm_init_hi')):
        text = text.replace('@%s@' % token, repr(values[key]) + 'f')
    return text.replace('@ORDER@', '%d.0f' % values['df_order'])


_PARSE_REQ = re.compile(r'req (-?\d+) (\d+) ([0-9a-f]{8})')

# The shipped grid: 48 kHz, 1024-point frames, 32 ERB bands, 96 deep-filter bins, lookahead 1/1.
_SHIPPED = (48000, 1024, 32, 96, 1, 1)


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

    @staticmethod
    def grid_text(grid, overrides=None):
        """'sr fft erb df mask dfla' and, with `overrides` (a dict of the spec fields of
        dfn2_spec_values), the bracketed order and normaliser values."""
        text = '%d %d %d %d %d %d' % grid
        if overrides:
            values = {**dfn_pairs.dfn2_spec_values(), **overrides}
            text += ' [%d %s]' % (values['df_order'], ' '.join(repr(values[k]) for k in _NORM_KEYS))
        return text

    def req(self, grid, overrides=None):
        status, size, digest = _PARSE_REQ.fullmatch(
            self.run(['req ' + self.grid_text(grid, overrides)])[0]).groups()
        assert status == '0', (grid, status)
        return int(size), digest

    def starts(self, grid, overrides=None):
        """[erb first, erb last, spec first, spec last] as dfn2_state_init() set them, float32."""
        verb, *fields = self.run(['starts ' + self.grid_text(grid, overrides)])[0].split()
        assert verb == 'starts' and len(fields) == 4, fields
        return [float(np.float32(f)) for f in fields]


_NORM_KEYS = ('erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db', 'erb_norm_scale_db',
              'spec_norm_tau_sec', 'spec_norm_init_lo', 'spec_norm_init_hi')


@pytest.fixture(scope='module')
def pools(builds):
    main = _pool_main()
    full = builds.objects([])
    mask = builds.objects([_MASK])
    return {name: _Pool(builds.link(main, objs, 'pool_' + name, flags)[1])
            for name, objs, flags in (('full', full, []), ('mask', mask, [_MASK]))}


def test_pool_identity_follows_the_grid_the_pair_the_variant_and_the_taus(pools):
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
    # Either tau alone is in the hash (a model trained with another time constant), and so is
    # the order.
    base_grid = (16000, 256, 24, 96, 1, 1)
    base = full.req(base_grid)
    assert full.req(base_grid, {}) == base
    erb_only = full.req(base_grid, {'erb_norm_tau_sec': 0.5})
    spec_only = full.req(base_grid, {'spec_norm_tau_sec': 0.5})
    both = full.req(base_grid, {'erb_norm_tau_sec': 0.5, 'spec_norm_tau_sec': 0.5})
    order = full.req(base_grid, {'df_order': 6})
    assert len({base[1], erb_only[1], spec_only[1], both[1], order[1]}) == 5
    assert erb_only[0] == spec_only[0] == both[0] == base[0], 'a tau changes the hash only'
    assert order[0] > base[0], 'a longer filter needs a bigger pool'


# The normaliser settings of the spec besides the taus: each is in the pool hash and none of
# them changes the pool size. Each entry: the spec field, the slot of the state
# dfn2_state_init() fills with it (erb first/last, spec first/last; None for the scale, which
# scales the feature and starts nothing), the value tried.
_NORMALISER_FIELDS = [
    ('erb_norm_init_lo_db', 0, -70.0),
    ('erb_norm_init_hi_db', 1, -80.0),
    ('erb_norm_scale_db', None, 30.0),
    ('spec_norm_init_lo', 2, 0.002),
    ('spec_norm_init_hi', 3, 0.0002),
]


@pytest.mark.parametrize('field,slot,value', _NORMALISER_FIELDS,
                         ids=[f for f, _s, _v in _NORMALISER_FIELDS])
def test_the_pool_hash_reads_each_normaliser_field_and_the_state_starts_from_it(
        pools, field, slot, value):
    """A normaliser setting changes the pool hash and nothing else about the pool, and the state
    dfn2_state_init() starts is the one the hash describes."""
    full = pools['full']
    base = full.req(_SHIPPED)
    changed = full.req(_SHIPPED, {field: value})
    assert changed[1] != base[1], 'the setting is not in the hash'
    assert changed[0] == base[0], 'a normaliser setting changes the hash only'
    default_state = full.starts(_SHIPPED)
    state = full.starts(_SHIPPED, {field: value})
    if slot is None:
        assert state == default_state, 'the scale is not part of the state'
    else:
        # (the last band / bin is lo + 1.0 * (hi - lo) in float32: equal to hi within an ulp)
        assert state[slot] == pytest.approx(float(np.float32(value)), rel=1e-6), state
        assert state != default_state


class _PoolMirror:
    """An independent Python mirror of the pool of the pre/post unit (dfn2_prepost_get_mem_size):
    the byte count and the build-flags hash, derived from the layout rules alone -- the field
    lists of the four header structs under the platform's alignment rules, every array carved at
    a 16-byte offset, FNV-1a-32 over the fields the library's header comment names -- as a
    function of the grid, the normaliser settings and the variant. It reads nothing from the
    sources; the tests compare it with the C program on every row they use, so a change of the
    layout or of the hash walk is made on purpose, in both places."""

    LAYOUT_VERSION = {False: 8, True: 9}
    CARVE_VERSION = 11
    MODEL_INPUT_FRAMES = 3
    FEATURE_VERSION = 'dfn2_libdf_wnorm_upstream_init_no_eps_v5'

    def __init__(self, pointer_bytes=None):
        self.P = pointer_bytes or struct.calcsize('P')

    @staticmethod
    def _a16(x):
        return (x + 15) // 16 * 16

    def _struct_size(self, fields):
        sizes = {'ptr': (self.P, self.P), 'sizet': (self.P, self.P), 'int': (4, 4),
                 'float': (4, 4), 'uint': (4, 4), 'll': (8, 8), 'ull': (8, 8)}
        offset, align = 0, 1
        for field in fields:
            size, al = field if isinstance(field, tuple) else sizes[field]
            offset = (offset + al - 1) // al * al + size
            align = max(align, al)
        return (offset + align - 1) // align * align, align

    def _grid(self):
        # DFN2Grid: two float taus, 13 ints (sample_rate .. input_frames), 9 floats
        return self._struct_size(['float'] * 2 + ['int'] * 13 + ['float'] * 9)

    def _state_header(self, mask):
        fields = [self._grid()] + ['ptr'] * 2 + ['ptr'] * 2
        if not mask:
            fields += ['ptr'] * 7
        fields += ['ptr'] * 2 + ['ptr'] * 3 + ['ptr'] * 6 + ['ptr', 'sizet', 'll']
        if not mask:
            fields += ['int', 'int']
        return self._struct_size(fields)[0]

    def _io_header(self, mask):
        fields = ['ptr', 'ptr', 'ptr', 'sizet']
        if not mask:
            fields += ['ptr', 'sizet']
        fields += ['ptr', 'sizet', 'ull', 'uint'] + ['int'] * 5
        return self._struct_size(fields)[0]

    def _prepost_header(self, mask):
        fields = ['ptr', 'ptr', self._grid(), 'float', 'ptr', 'ptr'] + ['ptr'] * 4 + ['ptr'] * 2
        fields += ['ptr'] * 4 + ['ptr']
        if not mask:
            fields += ['ptr', 'ptr']
        fields += ['ptr']
        if not mask:
            fields += ['ptr']
        fields += ['int', 'ptr', 'ptr', 'll', 'int', 'int', 'int', 'ptr']
        return self._struct_size(fields)[0]

    def _carve(self, sizes):
        cursor = 0
        for nbytes in sizes:
            cursor = self._a16(cursor) + nbytes
        return self._a16(cursor)

    @staticmethod
    def derive(sample_rate, n_fft, n_erb, df_bins, mask_la, df_la, mask, order=5, frames=1):
        n_bins = n_fft // 2 + 1
        if mask:
            order, df_la = 0, 0
        noisy = mask_la + df_la + 1
        return dict(sample_rate=sample_rate, n_fft=n_fft, hop=n_fft // 2, n_bins=n_bins,
                    n_erb=n_erb, df_bins=df_bins, df_order=order, mask_lookahead=mask_la,
                    df_lookahead=df_la, frames=frames, input_frames=frames + 2,
                    ring=noisy if mask else max(order, noisy), mask=mask)

    def pool_bytes(self, g):
        P, mask = self.P, g['mask']
        n, nb, ne, db, order, ring = (g['frames'], g['n_bins'], g['n_erb'], g['df_bins'],
                                      g['df_order'], g['ring'])
        state = [4 * ne, 4 * db]
        if not mask:
            state += [4 * ring * nb, 4 * ring * nb, 4 * ring * db * order * 2, 4 * ring, 4 * db,
                      P * order, P * order]
        state += [4 * ring * nb, 4 * ring * nb, 4 * nb, 4 * ne, 4 * nb]
        state += [2 * x for x in (nb, nb, ne, ne, ne, ne)]
        io = [4 * g['input_frames'] * ne, 4 * 2 * g['input_frames'] * db]
        sizes = [self._prepost_header(mask), self._state_header(mask), self._carve(state),
                 self._io_header(mask), self._carve(io),
                 4 * n * nb, 4 * n * nb, 4 * ne, 4 * 2 * db, 4 * n * nb, 4 * n * nb, 4 * n * ne]
        if not mask:
            sizes += [4 * n * db * order * 2, 4 * n]
        sizes += [4 * ne]
        if not mask:
            sizes += [4 * db * order * 2]
        sizes += [P * n, P * n, 4 * n, 8 * n, 4 * n * nb, 4 * n * nb]
        return self._carve(sizes)

    @staticmethod
    def _f32(value):
        return np.float32(value)

    @classmethod
    def _bits(cls, value):
        return struct.unpack('<I', struct.pack('<f', float(cls._f32(value))))[0]

    @staticmethod
    def _fnv(h, value):
        for i in range(4):
            h ^= (value >> (8 * i)) & 0xFF
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    @staticmethod
    def norm_alpha(sample_rate, hop, tau):
        """train.py make_norm_alpha: three decimals, one more digit while the value reaches 1.0
        (Python's round() on the exact double), as float32; the tau is the float32 the C library
        holds."""
        exact = math.exp(-(hop / sample_rate) / float(np.float32(tau)))
        precision, alpha = 3, 1.0
        while alpha >= 1.0:
            alpha = round(exact, precision)
            precision += 1
        return np.float32(alpha)

    def build_hash(self, g, norm):
        """`norm` = (erb tau, lo, hi, scale, spec tau, lo, hi)."""
        mask = g['mask']
        et, elo, ehi, esc, st, slo, shi = norm
        erb_a = self.norm_alpha(g['sample_rate'], g['hop'], et)
        spec_a = self.norm_alpha(g['sample_rate'], g['hop'], st)
        scale = np.float32(1.0) / np.sqrt(np.float32(g['n_fft']))
        unscale = np.sqrt(np.float32(g['n_fft']))
        history = 0 if mask else g['df_order'] - g['df_lookahead'] - 1
        h = 2166136261
        for v in (g['sample_rate'], g['n_fft'], g['hop'], g['n_bins'], g['n_erb'], g['df_bins'],
                  g['df_order'], g['mask_lookahead'], g['df_lookahead'], history, g['ring'],
                  g['frames'], g['input_frames'], 1 if mask else 0, self.LAYOUT_VERSION[mask],
                  self.CARVE_VERSION, self.MODEL_INPUT_FRAMES):
            h = self._fnv(h, v & 0xFFFFFFFF)
        for ch in self.FEATURE_VERSION.encode():
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        for v in (et, st, erb_a, spec_a, scale, unscale, elo, ehi, esc, slo, shi):
            h = self._fnv(h, self._bits(v))
        if self.P == 4:                       # audiolib's ABI token on an ILP32 build
            for ch in b'|abi:ilp32':
                h ^= ch
                h = (h * 16777619) & 0xFFFFFFFF
        return h


_MIRROR_ROWS = [
    ((48000, 1024, 32, 96, 1, 1), {}), ((16000, 256, 24, 96, 0, 0), {}),
    ((16000, 256, 24, 96, 1, 1), {}), ((32000, 512, 28, 96, 2, 4), {}),
    ((24000, 512, 24, 64, 1, 2), {}), ((16000, 256, 24, 96, 1, 1), {'df_order': 7}),
    ((16000, 256, 24, 96, 1, 1), {'erb_norm_tau_sec': 0.3, 'spec_norm_tau_sec': 2.0}),
    ((16000, 256, 24, 96, 1, 1), {'erb_norm_init_lo_db': -70.0, 'erb_norm_scale_db': 30.0,
                                  'spec_norm_init_hi': 0.0005}),
    ((8000, 128, 16, 40, 1, 1), {'df_order': 3}),
]


@pytest.mark.skipif(struct.calcsize('P') != 8, reason='the mirror is compared on 64-bit hosts')
@pytest.mark.parametrize('pool_name', ['full', 'mask'])
def test_pool_equals_the_independent_mirror(pools, pool_name):
    """The pool bytes and the build-flags hash of the C library equal the Python mirror of the
    layout rules, for the shipped grid and others, orders and normaliser settings. The mirror
    holds the formula, so a deliberate layout change is made here as well as in the library."""
    mask = pool_name == 'mask'
    pool = pools[pool_name]
    mirror = _PoolMirror(8)
    for grid, overrides in _MIRROR_ROWS:
        grid = (*grid[:5], 0) if mask else grid
        values = {**dfn_pairs.dfn2_spec_values(), **overrides}
        g = _PoolMirror.derive(*grid, mask, order=values['df_order'])
        norm = tuple(values[k] for k in _NORM_KEYS)
        size, digest = pool.req(grid, overrides)
        assert (size, digest) == (mirror.pool_bytes(g), '%08x' % mirror.build_hash(g, norm)), (
            pool_name, grid, overrides, size, digest)


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
    # A MemReq of another tau or order is refused as well.
    tau = ['initex %s  %s' % (_Pool.grid_text(base, {'erb_norm_tau_sec': 0.5}),
                              _Pool.grid_text(base)),
           'initex %s  %s' % (_Pool.grid_text(base, {'df_order': 6}), _Pool.grid_text(base))]
    if mask:
        tau = tau[:1]   # the order is not part of the mask-only model
    for line in pool.run(tau):
        own, cross, tampered = map(int, line.split()[1:])
        assert (own, cross, tampered) == (1, 0, 6), line


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
    # The control: past the warm-up frames the host's own descriptor drives the accelerator and the frame is
    # committed.
    assert results[0].split() == ['runframe', '1', '1', '1'], results[0]
    # No descriptor at all is refused as well.
    assert results[1].split() == ['runnull', '1', '-1', '0'], results[1]
    for other, line in zip(others, results[2:]):
        assert line.split() == ['runframe', '1', '-1', '0'], (other, line)


# ---------------------------------------------------------------------------
# the registered state, the board form and the stale-pool gate (full and mask-only, NEON and scalar)
# ---------------------------------------------------------------------------

_STATE_KEYS = [(name, simd) for name in ('full', 'mask') for simd in SIMD_MODES]


@pytest.fixture(scope='module')
def state_pools(builds):
    """The pool program of both library builds in both SIMD modes."""
    main = _pool_main()
    programs = {}
    for name, flags in (('full', []), ('mask', [_MASK])):
        for simd in SIMD_MODES:
            done, exe = builds.link(main, builds.objects(flags, simd), 'state_%s_%d' % (name, simd),
                                    flags, simd)
            assert done.returncode == 0, done.stderr
            programs[(name, simd)] = _Pool(exe)
    return programs


@pytest.fixture(params=_STATE_KEYS, ids=['%s-%s' % (n, 'neon' if s else 'scalar') for n, s in _STATE_KEYS])
def state_pool(request, state_pools):
    return request.param[0], state_pools[request.param]


def _registration_is_valid(variant, bits):
    """The registration rule, restated from the header: bit 0 = h_gru pointer, 1 = h_gru count, 2 = pathway
    pointer, 3 = pathway count. Both pairs set or all four NULL/0; with mask-only the pathway pair must be absent."""
    gru, convp = bits & 3, bits >> 2
    if variant == 'mask':
        return convp == 0 and gru in (0, 3)
    return (gru, convp) in ((0, 0), (3, 3))


def test_registration_is_all_or_none(state_pool):
    """Of the 16 combinations of the four registration values only the documented ones are accepted: a
    refusal leaves the request, the pool and the registered buffers as they were (and create refuses too); an
    acceptance keeps the pool size and identity, zeroes the registered tensors, starts the generation at 1,
    hands the registered buffers to the accelerator and runs a frame."""
    variant, pool = state_pool
    grid = _Pool.grid_text((16000, 256, 24, 96, 1, 0 if variant == 'mask' else 1))
    results = pool.run(['reg %s  %d' % (grid, bits) for bits in range(16)])
    accepted = []
    for bits, line in enumerate(results):
        verb, index, status, init, create, clean, works = line.split()
        want = int(_registration_is_valid(variant, bits))
        assert (verb, int(index)) == ('reg', bits), line
        assert (int(status), int(init), int(create)) == (0 if want else -1, want, want), (variant, bits, line)
        assert (int(clean), int(works)) == (1, want), (variant, bits, line)
        if want:
            accepted.append(bits)
    assert accepted == ([0, 3] if variant == 'mask' else [0, 15])


def test_board_form_moves_the_generation_as_documented(state_pool):
    """Nothing registered, on a pool an earlier instance used with registered state: the views hold NULL and
    0, the generation is 1 after init and moves by one per reset, per warm-up frame and per refused finish only
    (not on a commit, a bypass, a second finish of a refused frame or a failing model), every status is the
    documented one, and no buffer the library was not handed is touched. Every lookahead pair of the build, and
    the shipped grid."""
    variant, pool = state_pool
    pairs = MASK_PAIRS if variant == 'mask' else FULL_PAIRS
    grids = [(16000, 256, 24, 96, *pair) for pair in pairs] + [(48000, 1024, 32, 96, *pairs[0])]
    # generation after init, empty views, then after: commit, bypass, refusal, second finish, bypass, reset,
    # failing model, NaN head, committing model; all statuses right; no foreign byte changed. With
    # m = mask_lookahead, each warm-up frame is one step (also with nothing registered), twice: after init
    # and after the reset; the real frames move it as before:
    #   init 1 | warm-up m -> 1+m | commit 1+m | bypass 1+m | refused finish 2+m | second finish 2+m |
    #   bypass 2+m | reset 3+m | warm-up m + failing model 3+2m | NaN head 4+2m | commit 4+2m
    for grid, line in zip(grids, pool.run(['board %s' % _Pool.grid_text(g) for g in grids])):
        m = grid[4]
        steps = (1, 1, 1 + m, 1 + m, 2 + m, 2 + m, 2 + m, 3 + m, 3 + 2 * m, 4 + 2 * m, 4 + 2 * m, 1, 1)
        want = ['board'] + [str(n) for n in steps]
        assert line.split() == want, (variant, grid, line)


def _stale_neighbours(mask):
    """Grids that differ from the 16 kHz base in one value (and the mask-only / full pairs), as spec text."""
    pair = (1, 0) if mask else (1, 1)
    base = (16000, 256, 24, 96, *pair)
    others = [(32000, 512, 28, 96, *pair), (16000, 256, 24, 64, *pair), (16000, 256, 28, 96, *pair),
              (24000, 256, 24, 96, *pair), (16100, 256, 24, 96, *pair), (16000, 256, 24, 96, 0, pair[1]),
              (48000, 1024, 32, 96, *pair)]
    if not mask:
        others += [(16000, 256, 24, 96, 1, 0), (16000, 256, 24, 96, 2, 2), (16000, 256, 24, 96, 0, 1)]
    texts = [_Pool.grid_text(other) for other in others]
    texts.append(_Pool.grid_text(base, {'erb_norm_tau_sec': 0.5}))
    if not mask:
        texts.append(_Pool.grid_text(base, {'df_order': 6}))
    return _Pool.grid_text(base), texts


def test_a_refused_stale_pool_changes_nothing(state_pool):
    """init_ex() refuses a pool made for another grid, pair or order, or described by a MemReq that differs in
    any compared field, before it writes anything: the pool (a live instance of the other grid) and the
    registered buffers are byte-identical afterwards, the other instance still reports its grid, and the
    matching MemReq then starts an instance that runs a frame."""
    variant, pool = state_pool
    base, others = _stale_neighbours(variant == 'mask')
    results = pool.run(['stale %s  %s' % (base, other) for other in others])
    for other, line in zip(others, results):
        assert line.split() == ['stale', '9', '1', '1', '1'], (variant, other, line)


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
    # The 12 external fields of the 18 are compared: a descriptor changed in one is refused. The
    # other 6 (GRU layers and width, encoder channels, pathway window) are caller-filled
    # bookkeeping the library does not read.
    for line in pool.run(['descfield %d %d %d %d %d %d' % g for g in grids]):
        count, refused, plain = map(int, line.split()[1:])
        assert (count, refused, plain) == (18, 12, 0), line


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

# The grid macros of a build-time grid. They are the test headers' alone
# (tests/dfn2_test_grid.h, pipelines/dfn/tests/dfn_test_grid.h: test scaffolding that sizes fixed
# arrays and sweeps grids), never the library's and never a product's.
GRID_MACROS = [
    'DFN2_GRID_KHZ', 'DFN2_SR', 'DFN2_N_FFT', 'DFN2_N_ERB', 'DFN2_DF_BINS',
    'DFN2_MASK_LOOKAHEAD', 'DFN2_DF_LOOKAHEAD', 'DFN2_DF_ORDER',
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
    assert sorted(p.name for p in DFN2_C_DIR.glob('dfn2_*.[ch]')) == [
        'dfn2_grid.h', 'dfn2_model_io.c', 'dfn2_model_io.h', 'dfn2_model_io_priv.h',
        'dfn2_prepost.c', 'dfn2_prepost.h', 'dfn2_process.c', 'dfn2_process.h']
    assert _grid_macro_uses(DFN2_C_DIR) == []


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
        'int y = DFN2_N_FFT_EXTRA + DFN2_TEST_GRU_ELEMENTS;\n': [],
    }.items():
        probe.write_text(source, encoding='utf-8')
        assert [w for _f, _l, w in _product_grid_uses([probe])] == want, source


def test_the_scan_sees_what_it_is_meant_to_see(tmp_path):
    """The scan is not vacuous: each shape of definition or use is found, a mention in a comment,
    a string, a longer identifier (DFN2_N_FFT_EXTRA) or the consumer macro of another family
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
        'int a = DFN2_N_FFT_EXTRA + DFN2_TEST_ERB_NORM_TAU_SEC + DFN3_SR;\n': [],
        "char c = '\"'; int z = DFN2_SR;\n": ['DFN2_SR'],
    }
    for source, want in shapes.items():
        (tmp_path / 'dfn2_probe.h').write_text(source, encoding='utf-8')
        assert [m for _f, _l, m in _grid_macro_uses(tmp_path)] == want, source
    mutated = tmp_path / 'lib'
    mutated.mkdir()
    for path in DFN2_C_DIR.glob('dfn2_*.[ch]'):
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
    for macro in GRID_MACROS[:8]:
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

# ---------------------------------------------------------------------------
# the variant link symbol
# ---------------------------------------------------------------------------

_LINK_MAIN = r"""
#include "dfn2_prepost.h"
int main(void) {
    DFN2GridSpec spec = {
        .sample_rate = 16000, .n_fft = 256, .n_erb = 24, .df_bins = 96, .df_order = 5,
        .mask_lookahead = 1, .df_lookahead = 0, .frames_per_call = 1,
        .erb_norm_tau_sec = 1.0f, .erb_norm_init_lo_db = -60.0f, .erb_norm_init_hi_db = -90.0f,
        .erb_norm_scale_db = 40.0f, .spec_norm_tau_sec = 1.0f, .spec_norm_init_lo = 0.001f,
        .spec_norm_init_hi = 0.0001f,
    };
    DFN2Grid grid;
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_grid_make(&grid, &spec) != 0) return 1;
    if (dfn2_prepost_config_defaults(&cfg, &grid) != 0) return 1;
    return dfn2_prepost_get_mem_size(&cfg, &req);
}
"""

_APP_MAIN = r"""
#include "dfn2_process.h"
int main(void) {
    DFN2_REQUIRE_VARIANT_LINK();   /* what an application calls in main() */
    return 0;
}
"""

# link-test variants: name -> (flags, symbol the units reference)
_VARIANTS = {
    'full': ([], 'dfn2_variant_full'),
    'mask': ([_MASK], 'dfn2_variant_mask_only'),
}


def _assert_no_link(done, symbol):
    """The link failed and the linker names `symbol` as the one it cannot find."""
    assert done.returncode != 0, 'the objects linked'
    assert re.search(re.escape(symbol) + r'\b', done.stderr), done.stderr


@pytest.fixture(scope='module')
def variants(builds):
    builds.prebuild([(flags, 1) for flags, _s in _VARIANTS.values()], chains=False)
    return {name: builds.objects(flags) for name, (flags, _s) in _VARIANTS.items()}


def test_units_of_one_variant_link(builds, variants):
    for name, objs in variants.items():
        done, _exe = builds.link(_LINK_MAIN, objs, 'same_' + name, _VARIANTS[name][0])
        assert done.returncode == 0, (name, done.stderr)


def test_units_of_different_variants_do_not_link(builds, variants):
    """A mask-only unit names a symbol the full process unit does not define, and the other way
    round; two prepost units of different builds under one process unit fail too."""
    base, odd = variants['full'], variants['mask']
    done, _exe = builds.link(_LINK_MAIN, [odd[0], *base[1:]], 'mix_a')
    _assert_no_link(done, _VARIANTS['full'][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], *odd[1:]], 'mix_b', _VARIANTS['mask'][0])
    _assert_no_link(done, _VARIANTS['mask'][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], base[1], odd[2]], 'mix_c', _VARIANTS['mask'][0])
    _assert_no_link(done, _VARIANTS['mask'][1])
    done, _exe = builds.link(_LINK_MAIN, [base[0], odd[1], base[2]], 'mix_d', _VARIANTS['mask'][0])
    _assert_no_link(done, _VARIANTS['mask'][1])


def test_the_symbol_carries_the_variant_but_not_the_grid(builds, variants):
    """Every variant defines exactly its own symbol; the grid, the lookahead pair and any size
    are in no symbol (one build serves every grid)."""
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
    flags, symbol = _VARIANTS['mask']
    done, _exe = builds.link(_APP_MAIN, variants['full'], 'app_mask', flags)
    _assert_no_link(done, symbol)
    done, _exe = builds.link(_APP_MAIN, variants['mask'], 'app_full_on_mask')
    _assert_no_link(done, _VARIANTS['full'][1])
