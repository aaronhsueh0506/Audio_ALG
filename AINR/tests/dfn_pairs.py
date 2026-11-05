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
  default would change. ``tests/test_dfn_lookahead_link.py`` pins the C header
  defaults to it (DFN3: ``dfn3_process.h``; DFN2: the pair that
  ``tests/dfn2_test_grid.h`` and ``pipelines/dfn/tests/dfn_test_grid.h`` -- the test
  programs' own grid headers, never read by the library -- use when no flag is
  given), so the build that takes no ``-D`` override is that pair.
* DFN2 has no pair in its library: the pair is a field of the ``DFN2Grid`` an
  application builds with ``dfn2_grid_make``. ``build_dfn2_grid_probe`` and
  ``run_dfn2_grid_probe`` build and run a small program that sweeps
  ``(mask, df)`` through that function, so the tests can check its refusals
  from C, in C11, gnu99 and C++; ``dfn2_status_codes``, ``dfn2_ring`` and
  ``dfn2_capacities`` give them the library's own enum, ring rule and
  capacities.
* ``dfn2_grid_config`` and ``dfn2_write_stream_input`` are the two helpers the
  DFN2 grid tests share: config.ini with another signal grid (so the Python
  reference is built for it as the trainer would), and the file the stream
  drivers read (matrices, features, heads and windows of the exported-graph
  wrapper).
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


# DFN2's state rings hold DFN2_MAX_DF_RING frames (dfn2_grid.h): ring =
# max(DF_ORDER, mask + df + 1) for a pair, so a pair deeper than this needs the
# capacity raised on every translation unit of the build.
_DFN2_DEFAULT_RING = 5


def defines(family, pair):
    """``-D`` overrides selecting ``pair``; none for the shipped pair, which is
    the build with no override. For DFN2 a pair that needs a deeper ring than
    the library's default capacity also raises it (the test programs read the
    pair, the library takes it from the grid they build)."""
    if tuple(pair) == shipped_pair(family):
        return []
    prefix = macro_prefix(family)
    flags = ['-D%s_MASK_LOOKAHEAD=%d' % (prefix, pair[0]),
             '-D%s_DF_LOOKAHEAD=%d' % (prefix, pair[1])]
    if family == 'DeepFilterNet2':
        ring = max(_DFN2_DEFAULT_RING, pair[0] + pair[1] + 1)
        if ring > _DFN2_DEFAULT_RING:
            flags.append('-DDFN2_MAX_DF_RING=%d' % ring)
    return flags


DFN2_DF_ORDER = 5


def dfn2_ring(pair, mask_only=False):
    """Frames the DFN2 state rings need for ``pair`` (dfn2_grid.h): the noisy
    ring holds mask + df + 1 frames and, with a deep filter, the masked ring
    holds DF_ORDER, so the ring is the larger of the two. A mask-only build
    keeps the noisy ring alone."""
    noisy = pair[0] + pair[1] + 1
    return noisy if mask_only else max(DFN2_DF_ORDER, noisy)


def dfn2_status_codes():
    """``{'OK': 0, 'ERR_MASK_LOOKAHEAD': -6, ...}`` read from the library's
    own ``DFN2_GRID_*`` enum, so a test names a refusal instead of a number."""
    text = (family_dir('DeepFilterNet2') / 'dfn2_grid.h').read_text()
    codes = {name: int(value) for name, value in re.findall(
        r'DFN2_GRID_(\w+)\s*=\s*(-?\d+)', text)}
    assert 'OK' in codes and 'ERR_CAPACITY' in codes, codes
    return codes


def dfn2_capacities(flags=()):
    """The ``DFN2_MAX_*`` capacities (and ``DFN2_MASK_ONLY``) a DFN2 build with
    ``flags`` sees, as the preprocessor resolves them from ``dfn2_grid.h``."""
    out = subprocess.run(
        ['cc', '-E', '-dM', '-x', 'c', '-I', str(family_dir('DeepFilterNet2')),
         *flags, '-'],
        input='#include "dfn2_grid.h"\n', check=True, capture_output=True,
        text=True).stdout
    macros = dict(re.findall(r'^#define\s+(\w+)\s+(\S+)\s*$', out, re.M))
    return {'n_fft': int(macros['DFN2_MAX_N_FFT']),
            'n_erb': int(macros['DFN2_MAX_N_ERB']),
            'df_bins': int(macros['DFN2_MAX_DF_BINS']),
            'ring': int(macros['DFN2_MAX_DF_RING']),
            'mask_only': 'DFN2_MASK_ONLY' in macros}


# A program valid as C and as C++ that sweeps dfn2_grid_make over every
# (mask_lookahead, df_lookahead) of the probe range at one signal grid.
_DFN2_GRID_PROBE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"

int main(int argc, char **argv)
{
    int sr, n_fft, n_erb, df_bins, mask, df;
    if (argc != 5) return 2;
    sr = atoi(argv[1]);
    n_fft = atoi(argv[2]);
    n_erb = atoi(argv[3]);
    df_bins = atoi(argv[4]);
    for (mask = -1; mask <= 3; ++mask) {
        for (df = -1; df <= 5; ++df) {
            DFN2Grid grid, before;
            int status, ring = 0, valid = -99, untouched;
            memset(&grid, 0x5a, sizeof grid);
            before = grid;
            status = dfn2_grid_make(&grid, sr, n_fft, n_erb, df_bins, mask, df);
            untouched = memcmp(&grid, &before, sizeof grid) == 0;
            if (status == DFN2_GRID_OK) {
                ring = grid.ring;
                valid = dfn2_grid_validate(&grid);
                untouched = 1;
            }
            printf("%d %d %d %d %d %d %s\n", mask, df, status, ring, valid,
                   untouched, dfn2_grid_status_text(status));
        }
    }
    return 0;
}
'''


def build_dfn2_grid_probe(workdir, std='c11', flags=(), cxx=False):
    """Compile the probe against ``dfn2_process.c`` (where ``dfn2_grid_make``
    lives) under ``-std=<std>`` (or C++11 for ``cxx``), with ``flags`` after the
    house flags. Returns the executable."""
    workdir = Path(workdir)
    source = workdir / 'grid_probe.c'
    source.write_text(_DFN2_GRID_PROBE)
    base = [flag for flag in house_cflags() if not flag.startswith('-std=')]
    flags = base + ['-std=' + ('c11' if cxx else std), *flags]
    library = workdir / 'grid_probe_process.o'
    subprocess.run(['cc', *flags, '-c', str(source_path('DeepFilterNet2',
                                                        'process')),
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


def run_dfn2_grid_probe(exe, grid=(48000, 1024, 32, 96)):
    """``{(mask, df): (status, ring, validate, untouched, text)}`` over the
    probe range for the signal ``grid`` (sample rate, n_fft, n_erb, df_bins)."""
    done = subprocess.run([str(exe), *map(str, grid)], capture_output=True,
                          text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    result = {}
    for line in done.stdout.splitlines():
        mask, df, status, ring, valid, untouched, text = line.split(' ', 6)
        result[(int(mask), int(df))] = (int(status), int(ring), int(valid),
                                        bool(int(untouched)), text)
    assert len(result) == len(_PROBE_MASK) * len(_PROBE_DF), result
    return result


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


def header_compiles(family, pair, std, workdir, extra=()):
    """Whether ``<family>_model_io.h`` compiles (syntax only) for ``pair`` under
    ``-std=<std>`` (``c++11`` selects the C++ compiler), with ``extra`` flags
    after the pair's; returns the finished process so a caller can read the
    diagnostic."""
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
         '-D%s_DF_LOOKAHEAD=%d' % (prefix, pair[1]), *extra, str(source)],
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
