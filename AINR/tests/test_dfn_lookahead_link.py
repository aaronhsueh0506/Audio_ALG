"""The lookahead pair is guarded where it is chosen, in both C families.

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
"""

import itertools
import re
import subprocess
import sys

import pytest

import dfn_pairs

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
    """Every unit of both families compiled for the shipped pair and for two
    others, with sections split so a garbage-collecting link is exercised."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    work = tmp_path_factory.mktemp('pair_link')
    built = {}
    for family in dfn_pairs.FAMILIES:
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
@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
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
    for family in dfn_pairs.FAMILIES:
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
@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
def test_the_headers_accept_exactly_the_pairs_the_model_accepts(
        tmp_path, family, std):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    accepted = set(dfn_pairs.accepted_pairs(family))
    grid = itertools.product(range(-1, 4), range(-1, 6))
    for pair in grid:
        done = dfn_pairs.header_compiles(family, pair, std, tmp_path)
        assert (done.returncode == 0) == (pair in accepted), (pair, done.stderr)


@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
def test_the_range_check_also_holds_in_cpp(tmp_path, family):
    if dfn_pairs.require_toolchain():
        pytest.skip('C compiler unavailable')
    shipped = dfn_pairs.shipped_pair(family)
    assert dfn_pairs.header_compiles(
        family, shipped, 'c++11', tmp_path).returncode == 0
    for pair in ((3, 0), (0, 5), (-1, 0)):
        assert dfn_pairs.header_compiles(
            family, pair, 'c++11', tmp_path).returncode != 0, pair


@pytest.mark.parametrize('family', dfn_pairs.FAMILIES)
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
