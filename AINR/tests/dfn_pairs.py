"""Shared support for the DFN2 / DFN3 lookahead-pair tests.

The ``(mask_lookahead, df_lookahead)`` pair is a model contract, so every test
that depends on it asks this module instead of keeping its own list:

* ``accepted_pairs(family)`` is what the model constructor accepts, probed, not
  restated; a widened or narrowed rule changes every parametrisation at once.
* ``c_build_pairs(family)`` is the subset whose C hosts are rebuilt and run:
  the cheap Python and grid/header checks cover every accepted pair, the C
  builds the six pairs with ``df_lookahead <= mask_lookahead`` plus
  ``SMOKE_PAIRS``, three pairs whose deep filter looks further ahead than the
  mask path. The remaining legal pairs are not built as C hosts here.
* ``shipped_pair(family)`` is ``config.ini [model]``, the pair a retrained
  default would change. ``tests/test_dfn_lookahead_link.py`` pins the pair that
  ``tests/dfn2_test_grid.h`` and ``pipelines/dfn/tests/dfn_test_grid.h`` -- the
  test programs' own grid headers, never read by the library -- use when no flag
  is given to it, so the build that takes no ``-D`` override is that pair.
* Neither library holds a pair: it is a field of the ``DFN2Grid`` /
  ``DFN3Grid`` an application builds with ``dfn2_grid_make`` /
  ``dfn3_grid_make``. ``build_dfn2_grid_probe`` and ``run_dfn2_grid_probe``
  build and run a small program that sweeps ``(mask, df)`` through the DFN2
  maker, so the tests can check its refusals from C, in C11, gnu99 and C++;
  ``dfn2_status_codes`` and ``dfn2_ring`` give them the library's own enum and
  ring rule. ``defines`` selects a pair in the DFN2 test programs, which read
  it from the build; DFN3 test programs pass it in the spec.
* ``dfn2_grid_config``, ``dfn2_spec_values`` and ``dfn2_write_stream_input``
  are the helpers the DFN2 grid tests share: config.ini with another signal
  grid (so the Python reference is built for it as the trainer would), the
  order and normaliser fields of the spec a C host builds from such a config,
  and the file the stream drivers read (matrices, features, heads and windows
  of the exported-graph wrapper).
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
import re
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
# queue in DFN3, a ring deeper than the order in DFN2.
SMOKE_PAIRS = ((0, 1), (1, 3), (2, 4))

# A grid wider than any rule the constructor could plausibly have, used to
# probe it.
_PROBE_MASK = range(-1, 4)
_PROBE_DF = range(-1, 6)


def family_dir(family):
    return AINR / family


def c_dir(family):
    """``lib/ainr/<dfn2|dfn3>/``: the family's C core, moved out of ``family_dir()``
    (now Python training/config.ini only) by the single-model-release restructuring."""
    return AINR.parent / 'lib' / 'ainr' / _SOURCE_PREFIX[family]


def macro_prefix(family):
    """``'DFN2'`` / ``'DFN3'``: the prefix of the family's C macros."""
    return _MACRO_PREFIX[family]


def source_path(family, kind):
    """``lib/ainr/<dfn2|dfn3>/dfn2_<kind>.c`` for kind in process, model_io, prepost."""
    return c_dir(family) / ('%s_%s.c' % (_SOURCE_PREFIX[family], kind))


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
    """``-D`` overrides selecting ``pair`` in a DFN2 test program; none for the
    shipped pair, which is the build with no override. DFN3 reads its pair from
    the grid spec, so it has no flag."""
    if family != 'DeepFilterNet2' or tuple(pair) == shipped_pair(family):
        return []
    return ['-DDFN2_MASK_LOOKAHEAD=%d' % pair[0],
            '-DDFN2_DF_LOOKAHEAD=%d' % pair[1]]


DFN2_DF_ORDER = 5


def dfn2_ring(pair, mask_only=False, df_order=DFN2_DF_ORDER):
    """Frames the DFN2 state rings need for ``pair`` (dfn2_process.h): the noisy
    ring holds mask + df + 1 frames and, with a deep filter, the masked ring
    holds ``df_order``, so the ring is the larger of the two. A mask-only build
    keeps the noisy ring alone."""
    noisy = pair[0] + pair[1] + 1
    return noisy if mask_only else max(df_order, noisy)


_GRID_HEADER = {'DeepFilterNet2': 'dfn2_grid.h', 'DeepFilterNet3': 'dfn3_process.h'}


def grid_status_codes(family):
    """``{'OK': 0, 'ERR_MASK_LOOKAHEAD': ..., ...}`` read from the family's own
    ``DFN<n>_GRID_*`` enum, so a test names a refusal instead of a number."""
    text = (c_dir(family) / _GRID_HEADER[family]).read_text()
    codes = {name: int(value) for name, value in re.findall(
        r'%s_GRID_(\w+)\s*=\s*(-?\d+)' % macro_prefix(family), text)}
    assert 'OK' in codes and 'ERR_MASK_LOOKAHEAD' in codes, codes
    return codes


def dfn2_status_codes():
    codes = grid_status_codes('DeepFilterNet2')
    assert 'ERR_FRAMES_PER_CALL' in codes, codes
    return codes


def dfn2_spec_values(cfg=None, family='DeepFilterNet2'):
    """The spec fields a C host takes from the model config besides the signal
    grid and the pair: ``df_order``, DFN2's ``frames_per_call`` (1: the one-frame
    graph) and the seven ``[feature]`` normaliser settings, as the trainer reads
    them. ``cfg`` is a ``dfn2_grid_config`` result; default is the shipped file of
    ``family``."""
    if cfg is None:
        cfg = configparser.ConfigParser()
        cfg.read(family_dir(family) / 'config.ini')
    values = {'df_order': cfg.getint('model', 'df_order')}
    if family == 'DeepFilterNet2':
        values['frames_per_call'] = 1
    for key in ('erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
                'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
                'spec_norm_init_hi'):
        values[key] = cfg.getfloat('feature', key)
    return values


# A program valid as C and as C++ that sweeps dfn<n>_grid_make over every
# (mask_lookahead, df_lookahead) of the probe range at one signal grid. The spec
# is filled field by field (a designated initialiser is not C++11); every field
# is set, so a zero frames_per_call cannot slip through.
_GRID_PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet@N@/dfn@N@_prepost.h"

int main(int argc, char **argv)
{
    DFN@N@GridSpec spec;
    int mask, df;
    if (argc != 13) return 2;
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = atoi(argv[1]);
    spec.n_fft = atoi(argv[2]);
    spec.n_erb = atoi(argv[3]);
    spec.df_bins = atoi(argv[4]);
    spec.df_order = atoi(argv[5]);
@FRAMES@    spec.erb_norm_tau_sec = (float)atof(argv[6]);
    spec.erb_norm_init_lo_db = (float)atof(argv[7]);
    spec.erb_norm_init_hi_db = (float)atof(argv[8]);
    spec.erb_norm_scale_db = (float)atof(argv[9]);
    spec.spec_norm_tau_sec = (float)atof(argv[10]);
    spec.spec_norm_init_lo = (float)atof(argv[11]);
    spec.spec_norm_init_hi = (float)atof(argv[12]);
    for (mask = -1; mask <= 3; ++mask) {
        for (df = -1; df <= 5; ++df) {
            DFN@N@Grid grid, before;
            int status, ring = 0, valid = -99, untouched;
            spec.mask_lookahead = mask;
            spec.df_lookahead = df;
            memset(&grid, 0x5a, sizeof grid);
            before = grid;
            status = dfn@N@_grid_make(&grid, &spec);
            untouched = memcmp(&grid, &before, sizeof grid) == 0;
            if (status == DFN@N@_GRID_OK) {
                ring = grid.ring;
                valid = dfn@N@_grid_validate(&grid);
                untouched = 1;
            }
            printf("%d %d %d %d %d %d %s\n", mask, df, status, ring, valid,
                   untouched, dfn@N@_grid_status_text(status));
        }
    }
    return 0;
}
"""


def _grid_probe_source(family):
    number = family[-1]
    frames = '    spec.frames_per_call = 1;\n' if family == 'DeepFilterNet2' else ''
    return (_GRID_PROBE.replace('@N@', number).replace('@FRAMES@', frames))


def build_grid_probe(workdir, std='c11', flags=(), cxx=False, family='DeepFilterNet2'):
    """Compile the probe against ``dfn<n>_process.c`` (where ``dfn<n>_grid_make``
    lives) under ``-std=<std>`` (or C++11 for ``cxx``), with ``flags`` after the
    house flags. Returns the executable."""
    workdir = Path(workdir)
    source = workdir / 'grid_probe.c'
    source.write_text(_grid_probe_source(family))
    base = [flag for flag in house_cflags() if not flag.startswith('-std=')]
    flags = base + ['-std=' + ('c11' if cxx else std), *flags]
    library = workdir / 'grid_probe_process.o'
    subprocess.run(['cc', *flags, '-c', str(source_path(family, 'process')),
                    '-o', str(library)], check=True, capture_output=True)
    exe = workdir / 'grid_probe'
    if cxx:
        # The C++ driver sees the headers' linkage: -x c++ compiles the same text.
        cxx_flags = [flag for flag in flags
                     if not flag.startswith(('-std=', '-Wpedantic'))]
        command = ['c++', '-std=' + std, *cxx_flags, '-x', 'c++', str(source),
                   '-x', 'none', str(library), audio_common_lib(), '-lm',
                   '-o', str(exe)]
    else:
        command = ['cc', *flags, str(source), str(library),
                   audio_common_lib(), '-lm', '-o', str(exe)]
    subprocess.run(command, check=True, capture_output=True)
    return exe


def build_dfn2_grid_probe(workdir, std='c11', flags=(), cxx=False):
    return build_grid_probe(workdir, std, flags, cxx, 'DeepFilterNet2')


_SHIPPED_GRID = (48000, 1024, 32, 96)


def run_grid_probe(exe, grid=_SHIPPED_GRID, cfg=None, family='DeepFilterNet2'):
    """``{(mask, df): (status, ring, validate, untouched, text)}`` over the
    probe range for the signal ``grid`` (sample rate, n_fft, n_erb, df_bins),
    with the order and normalisers of ``cfg`` (default: the family's shipped
    config)."""
    values = dfn2_spec_values(cfg, family)
    spec = [values['df_order'], *[repr(values[key]) for key in (
        'erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
        'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
        'spec_norm_init_hi')]]
    done = subprocess.run([str(exe), *map(str, grid), *map(str, spec)],
                          capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    result = {}
    for line in done.stdout.splitlines():
        mask, df, status, ring, valid, untouched, text = line.split(' ', 6)
        result[(int(mask), int(df))] = (int(status), int(ring), int(valid),
                                        bool(int(untouched)), text)
    assert len(result) == len(_PROBE_MASK) * len(_PROBE_DF), result
    return result


def run_dfn2_grid_probe(exe, grid=_SHIPPED_GRID, cfg=None):
    return run_grid_probe(exe, grid, cfg, 'DeepFilterNet2')


def dfn2_grid_config(sr, n_fft, n_erb, df_bins, pair=None):
    """DeepFilterNet2's config.ini with another signal grid (and, for ``pair``,
    another lookahead pair), so the Python reference -- feature config, model,
    ERB matrices -- is built for that grid exactly as the trainer would. Order,
    the feature taus and, without ``pair``, the lookahead pair are the shipped
    ones."""
    cfg = configparser.ConfigParser()
    cfg.read(family_dir('DeepFilterNet2') / 'config.ini')
    cfg.set('signal', 'sr', str(sr))
    cfg.set('signal', 'n_fft', str(n_fft))
    cfg.set('signal', 'win_len', str(n_fft))
    cfg.set('signal', 'hop_len', str(n_fft // 2))
    cfg.set('model', 'n_erb', str(n_erb))
    cfg.set('model', 'df_bins', str(df_bins))
    if pair is not None:
        cfg.set('model', 'mask_lookahead', str(pair[0]))
        cfg.set('model', 'df_lookahead', str(pair[1]))
    return cfg


def dfn2_write_stream_input(path, model, frames, erb, spec_feat, spec, heads,
                            windows):
    """The file the DFN2 stream drivers read: the frame count, the model's ERB
    matrices (forward bin-major, inverse band-major), the features, the noisy
    spectra, the heads of the exported-graph wrapper and the windows that
    produced them, all little-endian."""
    import numpy as np

    chunks = [
        np.array([frames], dtype='<i4').tobytes(),
        # erb_fwd is bin-major in C, the model keeps (n_erb, n_bins).
        model.erb_fb.detach().numpy().astype('<f4').T.copy().tobytes(),
        model.erb_inv.detach().numpy().astype('<f4').tobytes(),
        erb[0, 0].numpy().astype('<f4').tobytes(),
        spec_feat[0].permute(1, 0, 2).contiguous().numpy()
        .astype('<f4').tobytes(),
        spec[0].real.T.contiguous().numpy().astype('<f4').tobytes(),
        spec[0].imag.T.contiguous().numpy().astype('<f4').tobytes(),
    ]
    mask, coefs, alpha = heads
    chunks += [
        np.concatenate(mask, axis=2)[0, 0, :, :].astype('<f4').tobytes(),
        np.concatenate(coefs, axis=1)[0].astype('<f4').tobytes(),
        np.concatenate(alpha, axis=1)[0, :, 0].astype('<f4').tobytes(),
    ]
    erb_windows, spec_windows = windows
    chunks += [
        np.stack([w[0, 0].numpy() for w in erb_windows])
        .astype('<f4').tobytes(),
        np.stack([w[0].numpy() for w in spec_windows])
        .astype('<f4').tobytes(),
    ]
    Path(path).write_bytes(b''.join(chunks))


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
