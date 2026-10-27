"""Shared support for the DFN2 / DFN3 lookahead-pair tests.

The ``(mask_lookahead, df_lookahead)`` pair is a model contract, so every test
that depends on it asks this module instead of keeping its own list:

* ``accepted_pairs(family)`` is what the model constructor accepts, probed, not
  restated; a widened or narrowed rule changes every parametrisation at once.
* ``c_build_pairs(family)`` is the subset whose C hosts are rebuilt and run:
  the cheap Python and compile-time checks cover every accepted pair, the C
  builds the six pairs with ``df_lookahead <= mask_lookahead`` plus
  ``SMOKE_PAIRS``, three pairs whose deep filter looks further ahead than the
  mask path. The remaining legal pairs are not built as C hosts here.
* ``shipped_pair(family)`` is ``config.ini [model]``, the pair a retrained
  default would change. ``tests/test_dfn_lookahead_link.py`` pins the C header
  defaults to it, so the build that takes no ``-D`` override is that pair.
* ``house_cflags`` and ``audio_common_lib`` are the Makefile's own flags and
  archive, so a pytest compile cannot drift from ``make``.

This is a helper, not a test module; ``conftest.py`` puts this directory on the
import path.
"""

import configparser
import functools
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

AINR = Path(__file__).resolve().parents[1]
ROOT = AINR.parent.parent
AUDIO_COMMON = ROOT / 'audiolib' / 'common'

FAMILIES = ('DeepFilterNet2', 'DeepFilterNet3')
_MACRO_PREFIX = {'DeepFilterNet2': 'DFN2', 'DeepFilterNet3': 'DFN3'}
_SOURCE_PREFIX = {'DeepFilterNet2': 'dfn2', 'DeepFilterNet3': 'dfn3'}

# Pairs with df_lookahead > mask_lookahead that the C builds cover: a head
# queue in DFN3, a ring of 6 or 7 slots in DFN2.
SMOKE_PAIRS = ((0, 1), (1, 3), (2, 4))

# A grid wider than any rule the constructor could plausibly have, used to
# probe it.
_PROBE_MASK = range(-1, 4)
_PROBE_DF = range(-1, 6)


def family_dir(family):
    return AINR / family


def macro_prefix(family):
    """``'DFN2'`` / ``'DFN3'``: the prefix of the family's C macros."""
    return _MACRO_PREFIX[family]


def source_path(family, kind):
    """``<family>/dfn2_<kind>.c`` for kind in process, model_io, prepost."""
    return family_dir(family) / ('%s_%s.c' % (_SOURCE_PREFIX[family], kind))


def pair_id(pair):
    return 'mask%d_df%d' % tuple(pair)


def shipped_pair(family):
    """The pair config.ini names, i.e. what the default C build must equal."""
    cfg = configparser.ConfigParser()
    cfg.read(family_dir(family) / 'config.ini')
    return (cfg.getint('model', 'mask_lookahead'),
            cfg.getint('model', 'df_lookahead'))


def import_family_module(family, name):
    """``AINR.<family>.<name>``, imported qualified so the two families' same
    named modules (train, model, inference, ...) never shadow each other."""
    parent = str(AINR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module('AINR.%s.%s' % (family, name))


@functools.lru_cache(maxsize=None)
def accepted_pairs(family):
    """Every pair the family's model constructor accepts, in (mask, df) order."""
    model_class = getattr(import_family_module(family, 'model'), family)
    accepted = []
    for mask in _PROBE_MASK:
        for df in _PROBE_DF:
            try:
                model_class(
                    n_fft=512, sr=16000, n_erb=32, df_bins=64, enc_ch=8,
                    emb_size=32, df_hidden=32, lin_groups=4,
                    enc_lin_groups=4, mask_lookahead=mask, df_lookahead=df,
                )
            except ValueError:
                continue
            accepted.append((mask, df))
    return tuple(accepted)


def c_build_pairs(family):
    """The accepted pairs the C hosts are built and run for (see above)."""
    return tuple(pair for pair in accepted_pairs(family)
                 if pair[1] <= pair[0] or pair in SMOKE_PAIRS)


def defines(family, pair):
    """``-D`` overrides selecting ``pair``; none for the shipped pair, which is
    the build with no override."""
    if tuple(pair) == shipped_pair(family):
        return []
    prefix = macro_prefix(family)
    return ['-D%s_MASK_LOOKAHEAD=%d' % (prefix, pair[0]),
            '-D%s_DF_LOOKAHEAD=%d' % (prefix, pair[1])]


def header_compiles(family, pair, std, workdir):
    """Whether ``<family>_model_io.h`` compiles (syntax only) for ``pair`` under
    ``-std=<std>`` (``c++11`` selects the C++ compiler); returns the finished
    process so a caller can read the diagnostic."""
    prefix = macro_prefix(family)
    cxx = std.startswith('c++')
    source = Path(workdir) / ('header_%s.%s' % (_SOURCE_PREFIX[family],
                                                'cpp' if cxx else 'c'))
    source.write_text('#include "%s_model_io.h"\nint main(void) { return 0; }\n'
                      % _SOURCE_PREFIX[family])
    return subprocess.run(
        ['c++' if cxx else 'cc', '-std=' + std, '-fsyntax-only',
         '-I', str(family_dir(family)),
         '-I', str(AUDIO_COMMON / 'include'), '-I', str(AUDIO_COMMON),
         '-D%s_MASK_LOOKAHEAD=%d' % (prefix, pair[0]),
         '-D%s_DF_LOOKAHEAD=%d' % (prefix, pair[1]), str(source)],
        capture_output=True, text=True)


def require_toolchain():
    """Skip text for a host without the C build tools, else None."""
    if shutil.which('cc') is None or shutil.which('make') is None:
        return 'C build tools unavailable'
    return None


@functools.lru_cache(maxsize=None)
def house_cflags(simd=1):
    """The flags ``make`` compiles AINR with (``make print-cflags``) with -Werror
    on and the include of AINR itself made absolute."""
    out = subprocess.run(
        ['make', '-s', '-C', str(AINR), 'WERROR=1', 'SIMD=%d' % simd,
         'print-cflags'],
        check=True, capture_output=True, text=True).stdout
    flags = out.strip().splitlines()[-1].split()
    return tuple('-I' + str(AINR) if flag == '-I.' else flag
                 for flag in flags)


@functools.lru_cache(maxsize=None)
def suite_sources(driver):
    """The sources of ``make test``'s contract binary, with ``tests/<driver>``
    in place of its test driver (``make print-sources``), as absolute paths."""
    out = subprocess.run(['make', '-s', '-C', str(AINR), 'print-sources'],
                         check=True, capture_output=True, text=True).stdout
    sources = out.strip().splitlines()[-1].split()
    assert sources[0].startswith('tests/'), sources
    return tuple(str(AINR / name) for name in
                 ['tests/' + driver] + sources[1:])


@functools.lru_cache(maxsize=None)
def audio_common_lib(simd=1):
    """audiolib's NE10 archive for the SIMD setting, built if needed."""
    make = ['make', '-s', '-C', str(AUDIO_COMMON), 'BACKEND=ne10',
            'SIMD=%d' % simd]
    subprocess.run(make + ['lib'], check=True, capture_output=True)
    lib = subprocess.run(make + ['print-lib-path'], check=True,
                         capture_output=True,
                         text=True).stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


# Small enough to build in a moment; the signal grid and the pair are the
# family's own.
_SMALL_MODEL = {'enc_channels': '8', 'emb_size': '32', 'df_hidden': '32',
                'lin_groups': '4', 'enc_lin_groups': '4'}


def write_small_model(tmp_path, family, pair, seconds=0.5):
    """A config.ini for ``pair``, a checkpoint trained for it (random weights,
    the real checkpoint contract) and a directory with one noise WAV.
    Returns ``(config, checkpoint, wav_dir)``."""
    import numpy as np
    import soundfile
    import torch

    cfg = configparser.ConfigParser()
    cfg.read(family_dir(family) / 'config.ini')
    for key, value in _SMALL_MODEL.items():
        cfg.set('model', key, value)
    cfg.set('model', 'mask_lookahead', str(pair[0]))
    cfg.set('model', 'df_lookahead', str(pair[1]))
    config = Path(tmp_path) / 'config.ini'
    with config.open('w') as stream:
        cfg.write(stream)

    train = import_family_module(family, 'train')
    model_cfg = train.read_model_config(cfg)
    sr = cfg.getint('signal', 'sr')
    hop = cfg.getint('signal', 'hop_len')
    torch.manual_seed(5)
    model = getattr(import_family_module(family, 'model'), family)(
        **model_cfg).eval()
    contract = train.make_checkpoint_contract(
        sr=sr, n_fft=cfg.getint('signal', 'n_fft'),
        win_len=cfg.getint('signal', 'win_len'), hop_len=hop,
        **{key: model_cfg[key] for key in (
            'n_erb', 'df_bins', 'df_order', 'mask_lookahead',
            'df_lookahead', 'mask_pf', 'pf_beta')},
        feature_cfg=train.read_feature_config(cfg, sr, hop),
        loss_cfg=train.read_loss_config(cfg))
    checkpoint = Path(tmp_path) / 'checkpoint.pth'
    torch.save({'model_version': train.MODEL_VERSION,
                'feature_version': train.FEATURE_VERSION,
                'loss_version': train.LOSS_VERSION, 'contract': contract,
                'state_dict': model.state_dict()}, checkpoint)

    wavs = Path(tmp_path) / 'wavs'
    wavs.mkdir()
    noise = np.random.default_rng(3).normal(0, 0.1, int(sr * seconds))
    soundfile.write(str(wavs / 'noise.wav'), noise.astype(np.float32), sr)
    return config, checkpoint, wavs
