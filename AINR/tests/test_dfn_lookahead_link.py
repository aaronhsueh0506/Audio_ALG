"""The lookahead pair is guarded where it is chosen, in both C families.

DeepFilterNet3: the pair is a build constant.

* Link time. ``(mask_lookahead, df_lookahead)`` is a ``-D`` constant of every
  translation unit, so one stray flag would build, say, ``dfn3_prepost.o`` for
  another pair than ``dfn3_process.o`` and run a window, ring or pool of one
  pair against another (DFN3State's size even depends on the ring). The units
  that depend on the pair reference a symbol named after their pair and
  ``*_process.c`` defines the one of its own, so any mix fails to link and a
  matched set links. This must also hold when the linker drops unreferenced
  sections.
* Compile time. A pair outside what the model constructor accepts does not
  compile, in C11, gnu99 (the integration build) and C++; and the set the
  headers accept is the set the constructor accepts.
* The C header defaults are config.ini's pair, so the build with no override is
  the shipped model's.

DeepFilterNet2: the pair is a field of the ``DFN2Grid`` the application builds
(``dfn2_grid_make``), so the library holds no pair and none of the three guards
above can exist. What replaces them:

* Link time. The state layout is a build property, so the units reference one
  symbol, ``dfn2_variant_{full|mask_only}_n<N_FFT>_e<N_ERB>_d<DF_BINS>_r<RING>``
  (the deep-filter variant and the four capacities), and ``dfn2_process.c``
  defines it: objects of another variant or capacity do not link, objects
  built under another pair flag are the same objects and do.
* Run time. ``dfn2_grid_make`` accepts exactly the pairs the model constructor
  accepts and refuses a ring above ``DFN2_MAX_DF_RING`` (``ERR_CAPACITY``); a
  pool sized for another pair is refused by ``dfn2_prepost_init_ex`` (the pool
  hash folds the whole grid), a model-I/O descriptor of another pair by
  ``dfn2_model_run_frame``, and a grid whose pair fields were edited by hand by
  every entry point that takes one.
* The pair that ``tests/dfn2_test_grid.h`` and ``pipelines/dfn/tests/dfn_test_grid.h``
  (the test programs' own grid headers: the library and the pipelines read
  neither) use when no flag is given is config.ini's, so the build with no
  override is the shipped model's.
"""

import configparser
import itertools
import re
import subprocess
import sys

import pytest

import dfn_pairs

DFN2 = 'DeepFilterNet2'
# The families whose pair is a build constant, hence a link and a compile-time
# property. DeepFilterNet2 takes its pair from the run-time grid.
PAIR_BUILD_FAMILIES = ('DeepFilterNet3',)

_KINDS = ('process', 'model_io', 'prepost')
_MAIN = r'''
#include "DeepFilterNet{n}/dfn{n}_prepost.h"
int main(void) {{
    DFN{n}PrepostConfig config;
    DFN{n}PrepostMemReq request;
    static DFN{n}ModelIOState io;
    static const float erb[DFN{n}_N_ERB];
    static const float spec[2][DFN{n}_DF_BINS];
    static const float fwd[DFN{n}_N_BINS * DFN{n}_N_ERB];
    static const float inv[DFN{n}_N_ERB * DFN{n}_N_BINS];
    dfn{n}_model_io_reset(&io);
    (void)dfn{n}_model_io_push_features(&io, erb, spec);
    if (dfn{n}_prepost_config_defaults(&config, DFN{n}_IO_FREQ) != 0) return 1;
    config.erb_fwd = fwd;
    config.erb_inv = inv;
    return dfn{n}_prepost_get_mem_size(&config, &request) == 0 ? 0 : 1;
}}
'''
_GC_LINK = '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections'


def _other_pairs(family):
    """A pair of the other mask and one of the same mask, against the shipped."""
    shipped = dfn_pairs.shipped_pair(family)
    accepted = dfn_pairs.accepted_pairs(family)
    other_mask = next(p for p in reversed(accepted) if p[0] != shipped[0])
    other_df = next(p for p in reversed(accepted)
                    if p[0] == shipped[0] and p[1] != shipped[1])
    return (other_mask, other_df)


@pytest.fixture(scope='module')
def objects(tmp_path_factory):
    """Every unit of the pair-build families compiled for the shipped pair and
    for two others, with sections split so a garbage-collecting link is
    exercised."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    work = tmp_path_factory.mktemp('pair_link')
    built = {}
    for family in PAIR_BUILD_FAMILIES:
        prefix = family[-1]
        main = work / ('main%s.c' % prefix)
        main.write_text(_MAIN.format(n=prefix))
        pairs = {dfn_pairs.shipped_pair(family), *_other_pairs(family)}
        for pair, kind in itertools.product(sorted(pairs), _KINDS + ('main',)):
            source = main if kind == 'main' else dfn_pairs.source_path(
                family, kind)
            target = work / ('%s_%s_%s.o' % (family, kind, dfn_pairs.pair_id(pair)))
            # `main` is compiled for the shipped pair: it reads no pair macro.
            subprocess.run(
                ['cc', *dfn_pairs.house_cflags(), '-ffunction-sections',
                 '-fdata-sections', *dfn_pairs.defines(family, pair), '-c',
                 str(source), '-o', str(target)],
                check=True, capture_output=True)
            built[(family, kind, pair)] = target
    return work, built


def _link(work, built, family, assignment, tag, gc=False):
    """Link main + the three units built for ``assignment`` (one pair each)."""
    shipped = dfn_pairs.shipped_pair(family)
    exe = work / ('link_%s_%s' % (family, tag))
    command = ['cc', str(built[(family, 'main', shipped)]),
               *[str(built[(family, kind, pair)])
                 for kind, pair in zip(_KINDS, assignment)],
               dfn_pairs.audio_common_lib(), '-lm', '-o', str(exe)]
    if gc:
        command[1:1] = [_GC_LINK]
    return subprocess.run(command, capture_output=True, text=True), exe


@pytest.mark.parametrize('gc', [False, True], ids=['plain', 'gc_sections'])
@pytest.mark.parametrize('family', PAIR_BUILD_FAMILIES)
def test_units_built_for_different_pairs_do_not_link(objects, family, gc):
    work, built = objects
    shipped = dfn_pairs.shipped_pair(family)
    for other in _other_pairs(family):
        for assignment in itertools.product((shipped, other), repeat=3):
            tag = '%s_%s%s' % (dfn_pairs.pair_id(other), ''.join(
                'a' if pair == shipped else 'b' for pair in assignment),
                '_gc' if gc else '')
            done, exe = _link(work, built, family, assignment, tag, gc)
            if len(set(assignment)) == 1:
                assert done.returncode == 0, (assignment, done.stderr)
                assert subprocess.run([str(exe)]).returncode == 0
            else:
                assert done.returncode != 0, assignment
                assert re.search(r'%s_lookahead_mask\d_df\d' %
                                 dfn_pairs.macro_prefix(family).lower(),
                                 done.stderr), done.stderr


def test_every_unit_that_depends_on_the_pair_references_its_symbol(objects):
    """What makes the mix fail: process defines the symbol of its pair, model_io
    and prepost leave the symbol of theirs undefined (``nm``)."""
    work, built = objects
    for family in PAIR_BUILD_FAMILIES:
        prefix = dfn_pairs.macro_prefix(family).lower()
        for pair in {dfn_pairs.shipped_pair(family), *_other_pairs(family)}:
            symbol = '%s_lookahead_mask%d_df%d' % (prefix, *pair)
            for kind in _KINDS:
                listing = subprocess.run(
                    ['nm', str(built[(family, kind, pair)])],
                    capture_output=True, text=True, check=True).stdout
                defined = re.search(r'\s[TDRS]\s_?%s$' % symbol, listing, re.M)
                undefined = re.search(r'\sU\s_?%s$' % symbol, listing, re.M)
                assert bool(defined) == (kind == 'process'), (family, kind)
                assert bool(undefined) == (kind != 'process'), (family, kind)


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
@pytest.mark.parametrize('family', PAIR_BUILD_FAMILIES)
def test_the_headers_accept_exactly_the_pairs_the_model_accepts(
        tmp_path, family, std):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    accepted = set(dfn_pairs.accepted_pairs(family))
    grid = itertools.product(range(-1, 4), range(-1, 6))
    for pair in grid:
        done = dfn_pairs.header_compiles(family, pair, std, tmp_path)
        assert (done.returncode == 0) == (pair in accepted), (pair, done.stderr)


@pytest.mark.parametrize('family', PAIR_BUILD_FAMILIES)
def test_the_range_check_also_holds_in_cpp(tmp_path, family):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    shipped = dfn_pairs.shipped_pair(family)
    assert dfn_pairs.header_compiles(
        family, shipped, 'c++11', tmp_path).returncode == 0
    for pair in ((3, 0), (0, 5), (-1, 0)):
        assert dfn_pairs.header_compiles(
            family, pair, 'c++11', tmp_path).returncode != 0, pair


@pytest.mark.parametrize('family', PAIR_BUILD_FAMILIES)
def test_c_header_defaults_are_the_pair_of_config_ini(family):
    """The build with no override is the shipped model's: a one-sided edit of
    config.ini or of the header would change latency and tap alignment on the
    device only."""
    header = dfn_pairs.source_path(family, 'process').with_suffix(
        '.h').read_text()
    prefix = dfn_pairs.macro_prefix(family)

    def macro(name):
        found = re.search(r'^#define\s+%s_%s\s+(\d+)\b' % (prefix, name),
                          header, re.M)
        assert found, name
        return int(found.group(1))

    assert (macro('MASK_LOOKAHEAD'), macro('DF_LOOKAHEAD')) == (
        dfn_pairs.shipped_pair(family))


# ---------------------------------------------------------------------------
# DeepFilterNet2: the pair is a run-time grid field
# ---------------------------------------------------------------------------

def _dfn2_symbol(flags):
    """The variant symbol a DFN2 build with ``flags`` must define, from the
    capacities the preprocessor resolves (dfn2_process.h names it)."""
    cap = dfn_pairs.dfn2_capacities(flags)
    return 'dfn2_variant_%s_n%d_e%d_d%d_r%d' % (
        'mask_only' if cap['mask_only'] else 'full', cap['n_fft'],
        cap['n_erb'], cap['df_bins'], cap['ring'])


# Builds the link test mixes: the variant of the shipped build, two other
# lookahead pairs (the library never reads the macros, so they are the same
# objects), and one build per ingredient of the symbol.
_DFN2_VARIANTS = {
    'base': [],
    'pair_2_0': dfn_pairs.defines(DFN2, (2, 0)),
    'pair_0_0': dfn_pairs.defines(DFN2, (0, 0)),
    'n_fft': ['-DDFN2_MAX_N_FFT=2048'],
    'n_erb': ['-DDFN2_MAX_N_ERB=48'],
    'df_bins': ['-DDFN2_MAX_DF_BINS=128'],
    'ring': ['-DDFN2_MAX_DF_RING=7'],
    'mask_only': ['-DDFN2_MASK_ONLY'],
}

_DFN2_MAIN = r'''
#include "DeepFilterNet2/dfn2_prepost.h"

#ifdef DFN2_MASK_ONLY
#define MAIN_DF_LOOKAHEAD 0
#else
#define MAIN_DF_LOOKAHEAD 1
#endif

int main(void) {
    DFN2Grid grid;
    DFN2PrepostConfig config;
    DFN2PrepostMemReq request;
    static DFN2ModelIOState io;
    static const float erb[24];
    static const float spec[2 * 96];
    static const float fwd[129 * 24];
    static const float inv[24 * 129];
    if (dfn2_grid_make(&grid, 16000, 256, 24, 96, 1, MAIN_DF_LOOKAHEAD) != 0)
        return 1;
    if (dfn2_model_io_init(&io, &grid) != 0) return 1;
    (void)dfn2_model_io_push_features(&io, erb, spec);
    if (dfn2_prepost_config_defaults(&config, DFN2_IO_FREQ, &grid) != 0)
        return 1;
    config.erb_fwd = fwd;
    config.erb_inv = inv;
    return dfn2_prepost_get_mem_size(&config, &request) == 0 ? 0 : 1;
}
'''


@pytest.fixture(scope='module')
def dfn2_objects(tmp_path_factory):
    """Every DFN2 unit compiled for each build above, with sections split so a
    garbage-collecting link is exercised."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    work = tmp_path_factory.mktemp('dfn2_variant_link')
    main = work / 'main2.c'
    main.write_text(_DFN2_MAIN)
    built = {}
    for variant, flags in _DFN2_VARIANTS.items():
        for kind in _KINDS + ('main',):
            source = main if kind == 'main' else dfn_pairs.source_path(
                DFN2, kind)
            target = work / ('%s_%s.o' % (kind, variant))
            subprocess.run(
                ['cc', *dfn_pairs.house_cflags(), '-ffunction-sections',
                 '-fdata-sections', *flags, '-c', str(source), '-o',
                 str(target)],
                check=True, capture_output=True)
            built[(variant, kind)] = target
    return work, built


def _link_dfn2(work, built, assignment, tag, gc):
    """Link main + the three units built for ``assignment`` (one build each);
    main is the one of the matched build, or the shipped one in a mix."""
    main = assignment[0] if len(set(assignment)) == 1 else 'base'
    exe = work / ('link_%s' % tag)
    command = ['cc', str(built[(main, 'main')]),
               *[str(built[(variant, kind)])
                 for kind, variant in zip(_KINDS, assignment)],
               dfn_pairs.audio_common_lib(), '-lm', '-o', str(exe)]
    if gc:
        command[1:1] = [_GC_LINK]
    return subprocess.run(command, capture_output=True, text=True), exe


@pytest.mark.parametrize('gc', [False, True], ids=['plain', 'gc_sections'])
def test_dfn2_units_built_for_another_variant_do_not_link(dfn2_objects, gc):
    """Mask-only against full, and each of the four capacities, are link
    properties: any mix of units of different symbols fails and names the
    symbol the odd unit wants; a matched set links and runs. The lookahead pair
    is not one: units built under another pair flag carry the same symbol, so
    they link with the shipped ones (the pair is refused at run time, below)."""
    work, built = dfn2_objects
    symbols = {name: _dfn2_symbol(flags)
               for name, flags in _DFN2_VARIANTS.items()}
    assert symbols['pair_2_0'] == symbols['pair_0_0'] == symbols['base']
    assert len({symbols[name] for name in ('base', 'n_fft', 'n_erb', 'df_bins',
                                           'ring', 'mask_only')}) == 6
    for other in _DFN2_VARIANTS:
        if other == 'base':
            continue
        for assignment in itertools.product(('base', other), repeat=3):
            tag = '%s_%s%s' % (other, ''.join(
                'a' if name == 'base' else 'b' for name in assignment),
                '_gc' if gc else '')
            done, exe = _link_dfn2(work, built, assignment, tag, gc)
            wanted = [symbols[name] for name in assignment]
            if len(set(wanted)) == 1:
                assert done.returncode == 0, (other, assignment, done.stderr)
                assert subprocess.run([str(exe)]).returncode == 0, assignment
            else:
                assert done.returncode != 0, (other, assignment)
                # model_io and prepost want the symbol of their build; the
                # linker names every one that process does not define.
                for kind, symbol in zip(_KINDS[1:], wanted[1:]):
                    if symbol != wanted[0]:
                        assert symbol in done.stderr, (kind, assignment,
                                                       done.stderr)


def test_every_dfn2_unit_that_depends_on_the_variant_references_its_symbol(
        dfn2_objects):
    """What makes the mix fail: process defines the variant symbol of its build,
    model_io and prepost leave the symbol of theirs undefined (``nm``). The
    lookahead pair is in no symbol."""
    work, built = dfn2_objects
    for variant, flags in _DFN2_VARIANTS.items():
        symbol = _dfn2_symbol(flags)
        for kind in _KINDS:
            listing = subprocess.run(
                ['nm', str(built[(variant, kind)])],
                capture_output=True, text=True, check=True).stdout
            defined = re.findall(r'\s[TDRSsrdb]\s_?(dfn2_variant_\w+)$',
                                 listing, re.M)
            undefined = re.findall(r'\sU\s_?(dfn2_variant_\w+)$', listing,
                                   re.M)
            assert defined == ([symbol] if kind == 'process' else []), (
                variant, kind, defined)
            assert undefined == ([] if kind == 'process' else [symbol]), (
                variant, kind, undefined)
            assert 'lookahead_mask' not in listing, (variant, kind)


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    """``probe(std, flags, cxx)``: the grid probe of ``dfn_pairs``, built once
    per language mode and build."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    cache = {}

    def build(std='c11', flags=(), cxx=False):
        key = (std, tuple(flags), cxx)
        if key not in cache:
            cache[key] = dfn_pairs.build_dfn2_grid_probe(
                tmp_path_factory.mktemp('grid_probe'), std, flags, cxx)
        return cache[key]

    return build


# The signal grids the pair check runs on: the pair rules do not depend on
# them, and the three presets of the test header are the grids that exist.
_GRIDS = ((48000, 1024, 32, 96), (32000, 512, 28, 96), (16000, 256, 24, 96))
_ALL_PAIR_CAPACITY = ['-DDFN2_MAX_DF_RING=7']


def _expected_status(codes, pair, cap_ring, mask_only=False):
    """The refusal the pair rules give: mask 0..2 first, then df 0..4 (0 when
    mask-only), then the ring against the capacity."""
    if not 0 <= pair[0] <= 2:
        return codes['ERR_MASK_LOOKAHEAD']
    if not 0 <= pair[1] <= (0 if mask_only else 4):
        return codes['ERR_DF_LOOKAHEAD']
    if dfn_pairs.dfn2_ring(pair, mask_only) > cap_ring:
        return codes['ERR_CAPACITY']
    return codes['OK']


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
def test_dfn2_grid_make_accepts_exactly_the_pairs_the_model_accepts(probe, std):
    """The run-time replacement of the header ``#error``: at a capacity that
    holds every ring, ``dfn2_grid_make`` accepts exactly the pairs the model
    constructor accepts, on every signal grid, refuses the rest by the range
    that fails, derives the ring ``max(DF_ORDER, mask + df + 1)`` and leaves the
    grid alone on a refusal."""
    codes = dfn_pairs.dfn2_status_codes()
    accepted = set(dfn_pairs.accepted_pairs(DFN2))
    exe = probe(std, _ALL_PAIR_CAPACITY)
    for grid in _GRIDS:
        result = dfn_pairs.run_dfn2_grid_probe(exe, grid)
        ok = {pair for pair, row in result.items() if row[0] == codes['OK']}
        assert ok == accepted, (grid, sorted(ok ^ accepted))
        for pair, (status, ring, valid, untouched, text) in result.items():
            assert untouched, (grid, pair)
            assert status == _expected_status(codes, pair, 7), (grid, pair,
                                                                text)
            if status == codes['OK']:
                assert ring == dfn_pairs.dfn2_ring(pair), (grid, pair)
                assert valid == codes['OK'], (grid, pair)


def test_dfn2_grid_make_refuses_a_ring_above_the_capacity(probe):
    """A pair whose ring is deeper than ``DFN2_MAX_DF_RING`` is refused with
    ERR_CAPACITY, not accepted into a state that cannot hold it: at the default
    capacity of five frames that is 1/4, 2/3 and 2/4; raising the capacity on
    the build admits them, and a pair outside the range is still named by its
    range, not by the capacity."""
    codes = dfn_pairs.dfn2_status_codes()
    default_ring = dfn_pairs.dfn2_capacities()['ring']
    assert default_ring == 5
    deep = {pair for pair in dfn_pairs.accepted_pairs(DFN2)
            if dfn_pairs.dfn2_ring(pair) > default_ring}
    assert deep == {(1, 4), (2, 3), (2, 4)}
    for flags, cap in (([], default_ring), (_ALL_PAIR_CAPACITY, 7),
                       (['-DDFN2_MAX_DF_RING=6'], 6)):
        assert dfn_pairs.dfn2_capacities(flags)['ring'] == cap
        result = dfn_pairs.run_dfn2_grid_probe(probe('c11', flags))
        for pair, row in result.items():
            assert row[0] == _expected_status(codes, pair, cap), (cap, pair)
        refused = {pair for pair, row in result.items()
                   if row[0] == codes['ERR_CAPACITY']}
        assert refused == {pair for pair in deep
                           if dfn_pairs.dfn2_ring(pair) > cap}, cap
    assert {pair for pair, row in dfn_pairs.run_dfn2_grid_probe(
        probe('c11', [])).items() if row[0] == codes['ERR_CAPACITY']} == deep


def test_dfn2_mask_only_grid_has_no_deep_filter_lookahead(probe):
    """With ``-DDFN2_MASK_ONLY`` there is no deep filter to look ahead for: a
    nonzero df_lookahead is ERR_DF_LOOKAHEAD, the ring is the noisy ring
    ``mask + 1`` and the default capacity of two frames refuses mask 2 until it
    is raised to three."""
    codes = dfn_pairs.dfn2_status_codes()
    for flags, cap in ((['-DDFN2_MASK_ONLY'], 2),
                       (['-DDFN2_MASK_ONLY', '-DDFN2_MAX_DF_RING=3'], 3)):
        assert dfn_pairs.dfn2_capacities(flags)['ring'] == cap
        result = dfn_pairs.run_dfn2_grid_probe(probe('c11', flags))
        for pair, (status, ring, valid, untouched, text) in result.items():
            assert untouched, pair
            assert status == _expected_status(codes, pair, cap, True), (
                cap, pair, text)
            if status == codes['OK']:
                assert pair[1] == 0 and ring == pair[0] + 1
                assert valid == codes['OK']
        assert {pair for pair, row in result.items()
                if row[0] == codes['OK']} == (
            {(0, 0), (1, 0)} | ({(2, 0)} if cap == 3 else set()))


def test_dfn2_grid_check_also_holds_when_called_from_cpp(probe):
    """The same sweep from a C++ translation unit (the headers' linkage) gives
    the same answers as the C one."""
    flags = _ALL_PAIR_CAPACITY
    from_c = dfn_pairs.run_dfn2_grid_probe(probe('c11', flags))
    from_cpp = dfn_pairs.run_dfn2_grid_probe(probe('c++11', flags, cxx=True))
    assert from_cpp == from_c
    assert {pair for pair, row in from_cpp.items() if row[0] == 0} == set(
        dfn_pairs.accepted_pairs(DFN2))


# The run-time guards that stand in for the link error: one program, built for
# the whole accepted set (a capacity that holds every ring), drives every pair
# against every pair.
_GUARD_DRIVER = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"

#define SR 16000
#define N_FFT 256
#define N_ERB 24
#define DF_BINS 96
#define N_BINS (N_FFT / 2 + 1)
#define MAX_PAIRS 32

typedef struct {
    DFN2Grid grid;
} Pair;

static float erb_fwd[N_BINS * N_ERB];
static float erb_inv[N_ERB * N_BINS];
static float zero_bins[N_BINS];
static DFN2State state_scratch;
static DFN2ModelIOState io_scratch;
static int infer_calls;

static int infer(void *user, const DFN2PrepostInputs *in,
                 DFN2PrepostOutputs *out)
{
    size_t i;
    (void)user;
    (void)in;
    ++infer_calls;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 1.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.0f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.0f;
#endif
    return 0;
}

static int make_config(DFN2PrepostConfig *cfg, const DFN2Grid *grid)
{
    if (dfn2_prepost_config_defaults(cfg, DFN2_IO_FREQ, grid) != 0) return -1;
    cfg->erb_fwd = erb_fwd;
    cfg->erb_inv = erb_inv;
    return 0;
}

static void *aligned_pool(size_t bytes, void **raw)
{
    unsigned char *p = (unsigned char *)malloc(bytes + DFN2_PREPOST_ALIGNMENT);
    *raw = p;
    if (!p) return NULL;
    return p + (DFN2_PREPOST_ALIGNMENT -
                (size_t)((uintptr_t)p % DFN2_PREPOST_ALIGNMENT)) %
                   DFN2_PREPOST_ALIGNMENT;
}

/* A pool sized by one pair offered to another: the descriptor has the right
 * size and only the identity (the hash) can tell them apart. */
static void stale_pool(const Pair *a, const Pair *b)
{
    DFN2PrepostConfig ca, cb;
    DFN2PrepostMemReq ra, rb, stale;
    void *raw = NULL, *pool;
    int by_stale, by_exact, by_plain;
    if (make_config(&ca, &a->grid) || make_config(&cb, &b->grid) ||
        dfn2_prepost_get_mem_size(&ca, &ra) ||
        dfn2_prepost_get_mem_size(&cb, &rb)) {
        printf("stale error\n");
        return;
    }
    stale = ra;
    stale.bytes = rb.bytes;
    pool = aligned_pool((size_t)rb.bytes, &raw);
    by_stale = dfn2_prepost_init_ex(pool, (size_t)rb.bytes, &cb, &stale) != NULL;
    by_exact = dfn2_prepost_init_ex(pool, (size_t)rb.bytes, &cb, &rb) != NULL;
    by_plain = dfn2_prepost_init_ex(pool, (size_t)rb.bytes, &cb, NULL) != NULL;
    printf("stale a=%d/%d b=%d/%d same_hash=%d same_bytes=%d stale=%d exact=%d "
           "plain=%d\n",
           a->grid.mask_lookahead, a->grid.df_lookahead,
           b->grid.mask_lookahead, b->grid.df_lookahead,
           ra.build_flags_hash == rb.build_flags_hash, ra.bytes == rb.bytes,
           by_stale, by_exact, by_plain);
    free(raw);
}

/* A descriptor made for one pair against an instance of another, through the
 * accelerator callback boundary. */
static void descriptor(const Pair *a, const Pair *b)
{
    DFN2ModelIoDescriptor da, db;
    DFN2PrepostConfig cb;
    DFN2Model model;
    DFN2Prepost *p;
    int valid, run1, run2 = -9, calls1, calls2, i, ready = 0;
    if (dfn2_model_io_descriptor_default(&a->grid, &da) ||
        dfn2_model_io_descriptor_default(&b->grid, &db) ||
        make_config(&cb, &b->grid) || !(p = dfn2_prepost_create(&cb))) {
        printf("descriptor error\n");
        return;
    }
    valid = dfn2_model_io_descriptor_validate(&b->grid, &da);
    for (i = 0; i <= b->grid.mask_lookahead; ++i)
        ready = dfn2_prepost_pre_process_freq(p, zero_bins, zero_bins);
    memset(&model, 0, sizeof model);
    model.infer = infer;
    model.io_descriptor = &da;
    infer_calls = 0;
    run1 = dfn2_model_run_frame(&model, p);
    calls1 = infer_calls;
    if (run1 == -1) {
        /* The frame is still open: the matching descriptor runs it. */
        model.io_descriptor = &db;
        run2 = dfn2_model_run_frame(&model, p);
    }
    calls2 = infer_calls;
    printf("descriptor a=%d/%d b=%d/%d valid=%d ready=%d run1=%d calls1=%d "
           "run2=%d calls2=%d\n",
           a->grid.mask_lookahead, a->grid.df_lookahead,
           b->grid.mask_lookahead, b->grid.df_lookahead, valid, ready, run1,
           calls1, run2, calls2);
    dfn2_prepost_destroy(p);
}

static void edited(const Pair *a, const char *what, const DFN2Grid *g)
{
    DFN2PrepostConfig cfg;
    DFN2ModelIoDescriptor d;
    printf("edit a=%d/%d what=%s validate=%d config=%d state=%d io=%d "
           "descriptor=%d\n",
           a->grid.mask_lookahead, a->grid.df_lookahead, what,
           dfn2_grid_validate(g),
           dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ, g),
           dfn2_state_init(&state_scratch, g, NULL, NULL),
           dfn2_model_io_init(&io_scratch, g),
           dfn2_model_io_descriptor_default(g, &d));
}

/* A grid of one pair with a field of another written over it by hand: the
 * derived fields no longer follow from the given ones. */
static void hand_edited(const Pair *a)
{
    DFN2Grid g;
    char what[32];
    int k;
    edited(a, "none", &a->grid);
    for (k = 0; k <= 4; ++k) {
        if (k == a->grid.df_lookahead) continue;
        g = a->grid;
        g.df_lookahead = k;
        snprintf(what, sizeof what, "df:%d", k);
        edited(a, what, &g);
    }
    for (k = -1; k <= 3; k += 4) {
        g = a->grid;
        g.mask_lookahead = k;
        snprintf(what, sizeof what, "mask:%d", k);
        edited(a, what, &g);
    }
    for (k = -1; k <= 5; k += 6) {
        g = a->grid;
        g.df_lookahead = k;
        snprintf(what, sizeof what, "df:%d", k);
        edited(a, what, &g);
    }
    for (k = -1; k <= 1; k += 2) {
        g = a->grid;
        g.ring += k;
        snprintf(what, sizeof what, "ring:%+d", k);
        edited(a, what, &g);
    }
    g = a->grid;
    g.df_history += 1;
    edited(a, "history+1", &g);
}

int main(void)
{
    Pair pairs[MAX_PAIRS];
    int count = 0, mask, df, a, b;
    for (mask = 0; mask <= 2; ++mask) {
        for (df = 0; df <= 4; ++df) {
            int status = dfn2_grid_make(&pairs[count].grid, SR, N_FFT, N_ERB,
                                        DF_BINS, mask, df);
            if (status != DFN2_GRID_OK) {
                printf("refused %d/%d %d\n", mask, df, status);
                continue;
            }
            ++count;
        }
    }
    printf("pairs %d\n", count);
    for (a = 0; a < count; ++a) {
        hand_edited(&pairs[a]);
        for (b = 0; b < count; ++b) {
            stale_pool(&pairs[a], &pairs[b]);
            descriptor(&pairs[a], &pairs[b]);
        }
    }
    return 0;
}
'''

# The accepted set by build, from the rule the grid states: mask 0..2, df 0..4
# (0 when mask-only), the ring within the capacity this build raises it to.
_GUARD_BUILDS = {
    'full': (['-DDFN2_MAX_DF_RING=7'], False),
    'mask_only': (['-DDFN2_MASK_ONLY', '-DDFN2_MAX_DF_RING=3'], True),
}


@pytest.fixture(scope='module')
def guard_output(tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    lib = dfn_pairs.audio_common_lib()
    outputs = {}
    for build, (flags, _mask_only) in _GUARD_BUILDS.items():
        work = tmp_path_factory.mktemp('dfn2_guard_%s' % build)
        source = work / 'guard.c'
        source.write_text(_GUARD_DRIVER)
        exe = work / 'guard'
        subprocess.run(
            ['cc', *dfn_pairs.house_cflags(), *flags, str(source),
             *[str(dfn_pairs.source_path(DFN2, kind)) for kind in _KINDS],
             lib, '-lm', '-o', str(exe)],
            check=True, capture_output=True)
        done = subprocess.run([str(exe)], capture_output=True, text=True)
        assert done.returncode == 0, (build, done.stdout, done.stderr)
        outputs[build] = done.stdout.splitlines()
    return outputs


def _rows(lines, kind):
    """``[{key: value}]`` of the lines that start with ``kind``; a pair is
    ``(mask, df)`` and a value that is a number is an int."""
    rows = []
    for line in lines:
        if not line.startswith(kind + ' '):
            continue
        row = {}
        for token in line.split()[1:]:
            if '=' not in token:
                continue
            key, value = token.split('=', 1)
            if '/' in value:
                row[key] = tuple(int(part) for part in value.split('/'))
            else:
                try:
                    row[key] = int(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _guard_pairs(build):
    """The pairs the guard driver accepts for ``build``, from the rules."""
    flags, mask_only = _GUARD_BUILDS[build]
    cap = dfn_pairs.dfn2_capacities(flags)['ring']
    return sorted(
        (mask, df) for mask in range(3) for df in range(5)
        if _expected_status(dfn_pairs.dfn2_status_codes(), (mask, df), cap,
                            mask_only) == 0)


@pytest.mark.parametrize('build', sorted(_GUARD_BUILDS))
def test_dfn2_pool_of_another_pair_is_refused(guard_output, build):
    """``dfn2_prepost_init_ex`` with the MemReq of another pair is refused: the
    pool hash folds the whole grid, lookahead pair included. The stale request
    here has the right size and every field but the hash equal to the real one,
    so only the hash can refuse it; the real request and no request at all still
    build. A pool of the same pair is accepted."""
    lines = guard_output[build]
    pairs = _guard_pairs(build)
    assert 'pairs %d' % len(pairs) in lines
    rows = _rows(lines, 'stale')
    assert len(rows) == len(pairs) ** 2
    for row in rows:
        same = row['a'] == row['b']
        assert row['same_hash'] == int(same), row
        assert row['stale'] == int(same), row
        assert row['exact'] == 1 and row['plain'] == 1, row
    refused = [row for row in rows if row['stale'] == 0]
    assert len(refused) == len(pairs) * (len(pairs) - 1) > 0


@pytest.mark.parametrize('build', sorted(_GUARD_BUILDS))
def test_dfn2_descriptor_of_another_pair_is_refused(guard_output, build):
    """A model-I/O descriptor made for another pair is refused by
    ``dfn2_model_io_descriptor_validate`` and so by ``dfn2_model_run_frame``
    before the accelerator callback is touched (-1, the frame stays open and
    the matching descriptor runs it); the descriptor of the instance's own pair
    runs the frame."""
    lines = guard_output[build]
    pairs = _guard_pairs(build)
    rows = _rows(lines, 'descriptor')
    assert len(rows) == len(pairs) ** 2
    for row in rows:
        same = row['a'] == row['b']
        assert row['ready'] == 1, row
        assert (row['valid'] == 0) == same, row
        assert row['run1'] == (1 if same else -1), row
        assert row['calls1'] == (1 if same else 0), row
        assert row['run2'] == (-9 if same else 1), row
        assert row['calls2'] == 1, row


@pytest.mark.parametrize('build', sorted(_GUARD_BUILDS))
def test_dfn2_grid_with_hand_edited_pair_fields_is_refused(
        guard_output, build):
    """A grid of one pair with a lookahead, the ring or the history written over
    by hand is refused by ``dfn2_grid_validate`` and by every entry point that
    takes a grid (config defaults, state init, model-I/O init, descriptor); the
    unedited grid of each pair is accepted by all of them."""
    lines = guard_output[build]
    pairs = _guard_pairs(build)
    rows = _rows(lines, 'edit')
    for pair in pairs:
        edits = [row['what'] for row in rows if row['a'] == pair]
        assert 'none' in edits and len(edits) >= 8, (pair, edits)
    for row in rows:
        if row['what'] == 'none':
            assert [row[key] for key in
                    ('validate', 'config', 'state', 'io', 'descriptor')] == [
                0, 0, 0, 0, 0], row
        else:
            assert row['validate'] != 0, row
            assert [row[key] for key in
                    ('config', 'state', 'io', 'descriptor')] == [-1] * 4, row


@pytest.mark.parametrize('header', [
    pytest.param('AINR/tests/dfn2_test_grid.h', id='test_grid_header'),
    pytest.param('pipelines/dfn/tests/dfn_test_grid.h', id='pipeline_test_grid_header')])
@pytest.mark.parametrize('mask_only', [False, True],
                         ids=['full', 'mask_only'])
def test_dfn2_grid_header_defaults_are_the_pair_of_config_ini(
        header, mask_only):
    """The build with no override is the shipped model's: a one-sided edit of
    config.ini or of the header a program takes its grid from would change
    latency and tap alignment on the device only. The headers that give a
    test program its DFN2Grid (the AINR one and the pipelines' one, which is
    for the pipeline test programs only: pipelines and board samples build
    their grid with dfn2_grid_make()) run the shipped 48 kHz grid and the pair
    of config.ini when no flag is given; mask-only keeps the mask lookahead and
    has no deep filter (0)."""
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    out = subprocess.run(
        ['cc', '-E', '-dM', '-x', 'c', '-I', str(dfn_pairs.AINR),
         '-I', str(dfn_pairs.family_dir(DFN2)),
         '-I', str(dfn_pairs.AUDIO_COMMON / 'include'),
         '-I', str(dfn_pairs.AUDIO_COMMON),
         *(['-DDFN2_MASK_ONLY'] if mask_only else []),
         str(dfn_pairs.AINR.parent / header)],
        check=True, capture_output=True, text=True).stdout
    macros = dict(re.findall(r'^#define\s+(DFN2_\w+)\s+(\d+)\s*$', out, re.M))
    mask, df = dfn_pairs.shipped_pair(DFN2)
    assert (int(macros['DFN2_MASK_LOOKAHEAD']),
            int(macros['DFN2_DF_LOOKAHEAD'])) == (mask, 0 if mask_only else df)

    cfg = configparser.ConfigParser()
    cfg.read(dfn_pairs.family_dir(DFN2) / 'config.ini')
    assert (int(macros['DFN2_SR']), int(macros['DFN2_N_FFT']),
            int(macros['DFN2_N_ERB']), int(macros['DFN2_DF_BINS'])) == (
        cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft'),
        cfg.getint('model', 'n_erb'), cfg.getint('model', 'df_bins'))
