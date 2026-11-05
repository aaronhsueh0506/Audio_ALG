"""Lookahead pairs the DFN2 stage tests run (a helper, not a test module).

The accepted set is the model's (``AINR/DeepFilterNet2/model.py``); this file
only picks a light subset: ``PAIRS`` for the Python reference, ``C_PAIRS`` for
the C-parity shims, which are rebuilt with the pair's ``-D`` overrides.  The
other legal pairs are not run here (``AINR/tests`` covers them)."""
import configparser
import pathlib

from pipelines.dfn import dfn2_stage

SHIPPED = (dfn2_stage.MASK_LOOKAHEAD, dfn2_stage.DF_LOOKAHEAD)

#: The shipped pair first, then (0, 0), a deep filter looking further ahead than
#: the mask (0, 1), the longest mask lookahead with none in the filter (2, 0) and
#: the longest cascade (2, 4), whose ring outgrows DF_ORDER.
PAIRS = tuple(dict.fromkeys((SHIPPED, (0, 0), (0, 1), (2, 0), (2, 4))))

#: The pairs the C parity shims are built for.
C_PAIRS = tuple(dict.fromkeys((SHIPPED, (0, 1), (2, 4))))


def pair_id(pair):
    return 'mask%d_df%d' % tuple(pair)


def pair_config(directory, pair):
    """A copy of the shipped config.ini with [model] set to `pair`."""
    cfg = configparser.ConfigParser()
    cfg.read(pathlib.Path(dfn2_stage._DFN2_DIR) / 'config.ini')
    cfg.set('model', 'mask_lookahead', str(pair[0]))
    cfg.set('model', 'df_lookahead', str(pair[1]))
    path = pathlib.Path(directory) / ('config_%s.ini' % pair_id(pair))
    with open(path, 'w', encoding='utf-8') as handle:
        cfg.write(handle)
    return str(path)


#: State-ring depth the DFN2 library holds by default (``DFN2_MAX_DF_RING``,
#: ``dfn2_grid.h``); a pair whose ring is deeper raises it on every unit.
DEFAULT_RING = 5


def pair_defines(pair):
    """``-D`` overrides selecting `pair`; none for the shipped pair, so its
    build is the default one.  The shims read the pair from the test header
    ``dfn/tests/dfn_test_grid.h`` and hand it to the library as a run-time grid;
    a pair whose state ring (``pair_geometry``) is deeper than the library's
    default capacity also raises ``DFN2_MAX_DF_RING``."""
    if tuple(pair) == SHIPPED:
        return []
    defines = ['DFN2_MASK_LOOKAHEAD=%d' % pair[0], 'DFN2_DF_LOOKAHEAD=%d' % pair[1]]
    ring = dfn2_stage.pair_geometry(*pair)[1]
    if ring > DEFAULT_RING:
        defines.append('DFN2_MAX_DF_RING=%d' % ring)
    return defines
