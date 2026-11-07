"""The model numbers of the pipeline test grid against the checkpoint config they stand for.

``dfn_test_grid.h`` carries df_order, the frames per call and the seven normaliser values as
literals (a test program has no config.ini); every table row of the DFN glue tests builds its
spec from them.  This compares them with ``AINR/DeepFilterNet2/config.ini`` so a retrained config
cannot leave the tests running another model.  A macro the header no longer defines fails loudly.
"""
import configparser
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
HEADER = ROOT / 'pipelines' / 'dfn' / 'tests' / 'dfn_test_grid.h'
CONFIG = ROOT / 'AINR' / 'DeepFilterNet2' / 'config.ini'

#: header macro -> (config section, key)
FEATURE = {
    'DFN_TEST_ERB_NORM_TAU_SEC': ('feature', 'erb_norm_tau_sec'),
    'DFN_TEST_ERB_NORM_INIT_LO_DB': ('feature', 'erb_norm_init_lo_db'),
    'DFN_TEST_ERB_NORM_INIT_HI_DB': ('feature', 'erb_norm_init_hi_db'),
    'DFN_TEST_ERB_NORM_SCALE_DB': ('feature', 'erb_norm_scale_db'),
    'DFN_TEST_SPEC_NORM_TAU_SEC': ('feature', 'spec_norm_tau_sec'),
    'DFN_TEST_SPEC_NORM_INIT_LO': ('feature', 'spec_norm_init_lo'),
    'DFN_TEST_SPEC_NORM_INIT_HI': ('feature', 'spec_norm_init_hi'),
}


def _config():
    cfg = configparser.ConfigParser()
    assert cfg.read(CONFIG), CONFIG
    return cfg


def _macro(name):
    """The numeric literal `#define name <literal>` (optionally parenthesised, float suffix f)."""
    pattern = r'^#define\s+%s\s+\(?\s*(-?[0-9][0-9.]*(?:e-?[0-9]+)?)f?\s*\)?\s*(?:/\*.*)?$' % name
    found = re.findall(pattern, HEADER.read_text(encoding='utf-8'), re.M)
    assert len(found) == 1, f'{name} is not defined exactly once as a number in {HEADER.name}'
    return found[0]


def test_the_normaliser_numbers_are_the_config_numbers():
    cfg = _config()
    for macro, (section, key) in FEATURE.items():
        # The C literal is a float: compare as float32, the way the grid stores it.
        assert np.float32(float(_macro(macro))) == np.float32(cfg.getfloat(section, key)), macro


def test_the_order_and_the_table_ceiling_are_the_config_numbers():
    cfg = _config()
    assert int(_macro('DFN_TEST_DF_ORDER')) == cfg.getint('model', 'df_order')
    n_bins = cfg.getint('signal', 'n_fft') // 2 + 1
    assert int(_macro('DFN_TEST_TABLE_MAX_BINS')) == n_bins
    assert int(_macro('DFN_TEST_TABLE_MAX_ERB')) == cfg.getint('model', 'n_erb')
    # The graph the tests describe runs one frame per call (export_onnx.py --frames-per-call 1).
    assert int(_macro('DFN_TEST_FRAMES_PER_CALL')) == 1
