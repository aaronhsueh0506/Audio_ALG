"""DFN2 grid selection (dfn2_grid.h): the three built-in presets and a grid defined
outside the library.

The signal grid and the model geometry are build-time constants, so the grid is a
contract between the C library, the trained model and its ERB matrices. These tests
pin, per grid:

  constants       sr, n_fft, hop, bins, n_erb, df_bins and both norm alphas and
                  the analysis scale equal what train.py derives for that grid
  features        C ERB / complex features and compose equal the Python model
  stream          the C host stream equals the offline model, frame for frame
  refusals        a grid the C path cannot build stops at compile time, naming the
                  macro to fix
  link / identity objects built for different grids do not link, and the pool
                  identity changes with the grid, with each alpha and with each
                  normaliser starting point

48 kHz is the shipped grid, and only while the build line leaves the normaliser
starting points alone. 32 kHz is provisional (no model is trained for it).
'custom24k' is a grid nothing in the library knows: it is defined entirely from
the build line, which is how a later grid is added.
"""

import concurrent.futures
import configparser
import ctypes
import re
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

import dfn_pairs
import test_dfn2_lookahead_pairs as stream_tests

FAMILY = 'DeepFilterNet2'
DFN2 = dfn_pairs.family_dir(FAMILY)
AC = dfn_pairs.AUDIO_COMMON

pytestmark = pytest.mark.skipif(
    dfn_pairs.require_toolchain() is not None,
    reason='C build tools unavailable')


def _external(sr=24000, n_fft=512, n_erb=24, df_bins=96, alpha=0.989,
              erb_alpha=None, spec_alpha=None):
    """Build-line flags of a grid defined outside the library, in the order
    sr, n_fft, n_erb, df_bins, ERB alpha, SPEC alpha (a missing-macro case cuts
    the list). Each alpha not given takes `alpha`."""
    erb = alpha if erb_alpha is None else erb_alpha
    spec = alpha if spec_alpha is None else spec_alpha
    return ['-DDFN2_SR=%d' % sr, '-DDFN2_N_FFT=%d' % n_fft,
            '-DDFN2_N_ERB=%d' % n_erb, '-DDFN2_DF_BINS=%d' % df_bins,
            '-DDFN2_ERB_NORM_ALPHA=%.3ff' % erb,
            '-DDFN2_SPEC_NORM_ALPHA=%.3ff' % spec]


_CUSTOM_24K = _external()
_GRID_16K = dict(sr=16000, n_fft=256, alpha=0.992)

# The shipped integers defined on the build line instead of chosen as the preset.
_SHIPPED_INTEGERS = _external(sr=48000, n_fft=1024, n_erb=32)

# The link symbol of a 48 kHz build that is not the shipped grid (an external
# definition, or the preset with a normaliser starting point overridden).
_OTHER_48K_SYMBOL = 'dfn2_variant_full_s48000_f1024_e32_d96'

# name -> (build flags, sample rate, n_fft, n_erb, df_bins); order 5, pair 1/1.
GRIDS = {
    '48k': ([], 48000, 1024, 32, 96),
    '32k': (['-DDFN2_GRID_KHZ=32'], 32000, 512, 28, 96),
    '16k': (['-DDFN2_GRID_KHZ=16'], 16000, 256, 24, 96),
    'custom24k': (_CUSTOM_24K, 24000, 512, 24, 96),
    # Another deep-filter width than every preset's 96.
    'custom16k_df64': (_external(**_GRID_16K, df_bins=64), 16000, 256, 24, 64),
}
SIMD_MODES = (1, 0)
_GRID_SIMD = [(name, simd) for name in GRIDS for simd in SIMD_MODES]
_GRID_SIMD_IDS = ['%s-simd%d' % pair for pair in _GRID_SIMD]

# The normaliser starting points: build-line overrides of dfn2_process.h's own
# defaults, each a float in the pool hash.
_NORMALISER_STARTS = [
    '-DDFN2_ERB_NORM_INIT_LO_DB=-70.0f',
    '-DDFN2_ERB_NORM_INIT_HI_DB=-80.0f',
    '-DDFN2_ERB_NORM_SCALE_DB=30.0f',
    '-DDFN2_SPEC_NORM_INIT_LO=0.002f',
    '-DDFN2_SPEC_NORM_INIT_HI=0.0002f',
]


def _macro_name(flag):
    return flag[2:].split('=')[0]


def _f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _grid_config(sr, n_fft, n_erb, df_bins):
    """config.ini with the signal grid and band count of one grid; df_bins,
    order, lookahead and the feature taus are the shipped ones."""
    cfg = configparser.ConfigParser()
    cfg.read(DFN2 / 'config.ini')
    cfg.set('signal', 'sr', str(sr))
    cfg.set('signal', 'n_fft', str(n_fft))
    cfg.set('signal', 'win_len', str(n_fft))
    cfg.set('signal', 'hop_len', str(n_fft // 2))
    cfg.set('model', 'n_erb', str(n_erb))
    cfg.set('model', 'df_bins', str(df_bins))
    return cfg


@pytest.fixture(scope='module')
def dfn_py():
    """(model.py, train.py, export_onnx.py) of DeepFilterNet2."""
    return tuple(dfn_pairs.import_family_module(FAMILY, name)
                 for name in ('model', 'train', 'export_onnx'))


def _cc(args, **kwargs):
    return subprocess.run(['cc', *args], capture_output=True, text=True,
                          **kwargs)


_INCLUDE_FLAGS = ['-I', str(DFN2), '-I', str(AC / 'include'), '-I', str(AC)]

_PROBE = r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dfn2_process.h"
static uint32_t bits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
int main(void) {
    printf("%d %d %d %d %d %d %d %u %u %u %d\n", DFN2_SR, DFN2_N_FFT,
           DFN2_HOP_LEN, DFN2_N_BINS, DFN2_N_ERB, DFN2_DF_BINS, DFN2_DF_ORDER,
           bits(DFN2_ERB_NORM_ALPHA), bits(DFN2_SPEC_NORM_ALPHA),
           bits(DFN2_ANALYSIS_SCALE), DFN2_GRID_IS_SHIPPED);
    return 0;
}
'''

_HEADER_MAIN = '#include "dfn2_process.h"\nint main(void){return 0;}\n'

_LINK_MAIN = r'''
#include "dfn2_prepost.h"
int main(void) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    return dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) +
           dfn2_prepost_get_mem_size(&cfg, &req);
}
'''

_IDENTITY_MAIN = r'''
#include <stdio.h>
#include "dfn2_prepost.h"
static float fwd[DFN2_N_BINS * DFN2_N_ERB], inv[DFN2_N_ERB * DFN2_N_BINS];
int main(void) {
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) != 0) return 1;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    if (dfn2_prepost_get_mem_size(&cfg, &req) != 0) return 1;
    printf("%llu %08x\n", (unsigned long long)req.bytes,
           (unsigned)req.build_flags_hash);
    return 0;
}
'''

_APP_MAIN = r'''
#include "dfn2_process.h"
int main(void) {
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();   /* what the board samples call in main() */
    return 0;
}
'''


class _Builds:
    """Everything the tests compile, in one directory and memoised by build line:
    the same flags give the same probe output, objects and pool identity."""

    def __init__(self, work):
        self.work = work
        self._count = 0
        self._probes = {}
        self._units = {}
        self._identities = {}

    def _stem(self, label):
        self._count += 1
        return '%s_%d' % (label, self._count)

    def probe(self, flags):
        """The values _PROBE prints for a build line, as ints."""
        key = tuple(flags)
        if key not in self._probes:
            stem = self._stem('probe')
            source = self.work / (stem + '.c')
            source.write_text(_PROBE, encoding='utf-8')
            exe = self.work / stem
            built = _cc(['-std=c11', *flags, *_INCLUDE_FLAGS, str(source),
                         '-o', str(exe)])
            assert built.returncode == 0, built.stderr
            out = subprocess.run([str(exe)], capture_output=True, text=True,
                                 check=True).stdout.split()
            self._probes[key] = [int(token) for token in out]
        return list(self._probes[key])

    def objects(self, flags):
        """The three units compiled for a build line."""
        key = tuple(flags)
        if key not in self._units:
            stem = self._stem('unit')
            cflags = dfn_pairs.house_cflags(1)    # resolved before the threads start

            def build(unit):
                obj = self.work / ('%s_%s.o' % (stem, unit))
                source = dfn_pairs.source_path(FAMILY, unit)
                built = _cc([*cflags, *flags, '-I', str(DFN2), '-c',
                             str(source), '-o', str(obj)])
                assert built.returncode == 0, built.stderr
                return obj

            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                self._units[key] = list(
                    pool.map(build, ('process', 'model_io', 'prepost')))
        return list(self._units[key])

    def link(self, main_source, objs, label, flags=()):
        """Compile main_source (with flags) and link it with objs; returns the
        finished process and the executable's path."""
        stem = self._stem(label)
        source = self.work / (stem + '.c')
        source.write_text(main_source, encoding='utf-8')
        exe = self.work / stem
        done = _cc([*dfn_pairs.house_cflags(1), *flags, '-I', str(DFN2),
                    str(source), *map(str, objs), dfn_pairs.audio_common_lib(1),
                    '-lm', '-o', str(exe)])
        return done, exe

    def identity(self, flags):
        """[pool bytes, pool hash] of the build line, as the pre/post unit reports."""
        key = tuple(flags)
        if key not in self._identities:
            done, exe = self.link(_IDENTITY_MAIN, self.objects(flags), 'identity',
                                  flags)
            assert done.returncode == 0, done.stderr
            self._identities[key] = subprocess.run(
                [str(exe)], capture_output=True, text=True,
                check=True).stdout.split()
        return list(self._identities[key])


@pytest.fixture(scope='module')
def builds(tmp_path_factory):
    return _Builds(tmp_path_factory.mktemp('dfn2_grid_builds'))


def _compile_header(tmp_path, flags):
    """Syntax-check dfn2_process.h (which includes dfn2_grid.h) under flags."""
    source = tmp_path / 'header.c'
    source.write_text(_HEADER_MAIN, encoding='utf-8')
    return _cc(['-std=c11', '-fsyntax-only', *flags, *_INCLUDE_FLAGS,
                str(source)])


def _assert_no_link(done, symbol):
    """The link failed and the linker names `symbol` as the one it cannot find."""
    assert done.returncode != 0, 'the objects linked'
    assert re.search(re.escape(symbol) + r'\b', done.stderr), done.stderr


def _variant_symbol(name, variant='full'):
    """The link symbol the units of GRIDS[name] define, for the 'full' or the
    'mask_only' variant."""
    _flags, sr, n_fft, n_erb, df_bins = GRIDS[name]
    suffix = ('' if name == '48k'
              else '_s%d_f%d_e%d_d%d' % (sr, n_fft, n_erb, df_bins))
    return 'dfn2_variant_' + variant + suffix


# ---------------------------------------------------------------------------
# the constants of each grid are the trainer's
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', list(GRIDS))
def test_grid_constants_equal_what_the_trainer_derives(builds, dfn_py, name):
    flags, sr, n_fft, n_erb, df_bins = GRIDS[name]
    _model, train, _export = dfn_py
    hop = n_fft // 2
    (got_sr, got_fft, got_hop, got_bins, got_erb, got_df, got_order,
     erb_alpha, spec_alpha, scale, shipped) = builds.probe(flags)
    assert (got_sr, got_fft, got_hop, got_bins, got_erb, got_df, got_order) == (
        sr, n_fft, hop, n_fft // 2 + 1, n_erb, df_bins, 5)
    alpha = _f32_bits(train.make_norm_alpha(sr, hop, 1.0))
    assert erb_alpha == alpha and spec_alpha == alpha, (
        'alpha is exp(-(hop/sr)/tau) rounded to 3 decimals; the grid header '
        'must carry the float the trainer derives: %r' % (
            train.make_norm_alpha(sr, hop, 1.0),))
    assert scale == _f32_bits(train.analysis_scale(n_fft, n_fft, hop))
    assert shipped == (1 if name == '48k' else 0)


def test_an_unchosen_grid_is_the_shipped_one(builds):
    assert builds.probe([]) == builds.probe(['-DDFN2_GRID_KHZ=48'])


def test_a_grid_can_be_defined_in_a_header(builds, tmp_path):
    header = tmp_path / 'my_grid.h'
    header.write_text(
        ''.join('#define %s\n' % flag[2:].replace('=', ' ', 1)
                for flag in _CUSTOM_24K), encoding='utf-8')
    assert builds.probe(['-DDFN2_GRID_HEADER="%s"' % header]) == builds.probe(
        _CUSTOM_24K)


# ---------------------------------------------------------------------------
# a grid the C path cannot build stops at compile time
# ---------------------------------------------------------------------------

_BAD_GRIDS = [
    (['-DDFN2_GRID_KHZ=24'], 'must be 16, 32 or 48'),
    (['-DDFN2_GRID_KHZ=16', '-DDFN2_SR=16000'], 'not both'),
    (['-DDFN2_GRID_KHZ=16', '-DDFN2_N_ERB=20'], 'preset is complete'),
    # A macro left out: the flags cut after sr, after df_bins, after the ERB alpha.
    (_external()[:1], 'DFN2_N_FFT is not defined'),
    (_external()[:4], 'DFN2_ERB_NORM_ALPHA is not defined'),
    (_external()[:5], 'DFN2_SPEC_NORM_ALPHA is not defined'),
    (_external(n_fft=960), 'power of two'),
    (_external(n_fft=8), 'power of two'),
    (_external(n_fft=16384), 'power of two'),
    (_external(n_erb=30), 'multiple of 4'),
    (_external(**_GRID_16K, n_erb=100), 'at least two bins'),
    (_external(**_GRID_16K, df_bins=129), 'DFN2_DF_BINS must be below'),
    ([*_external(**_GRID_16K, df_bins=200), '-DDFN2_MASK_ONLY'],
     'DFN2_DF_BINS must be in'),
    # One bin would divide the normaliser's start values by zero.
    (_external(**_GRID_16K, df_bins=1), 'DFN2_DF_BINS must be in [2'),
    (_external(sr=0), 'DFN2_SR must be positive'),
    (_external(sr=-16000), 'DFN2_SR must be positive'),
    # The tap count is in no link identity, so it is not a grid choice.
    ([*_CUSTOM_24K, '-DDFN2_DF_ORDER=3'], 'DFN2_DF_ORDER is fixed at 5'),
]


@pytest.mark.parametrize('flags,message', _BAD_GRIDS,
                         ids=[m for _f, m in _BAD_GRIDS])
def test_header_refuses_a_grid_it_cannot_build(tmp_path, flags, message):
    done = _compile_header(tmp_path, flags)
    assert done.returncode != 0, 'the grid was accepted'
    assert message in done.stderr, done.stderr


def test_the_refusals_are_not_vacuous(tmp_path):
    """The same compile line with a valid grid passes."""
    done = _compile_header(tmp_path, _CUSTOM_24K)
    assert done.returncode == 0, done.stderr
    # df_bins may equal n_bins only in the mask-only build.
    done = _compile_header(
        tmp_path, [*_external(**_GRID_16K, df_bins=129), '-DDFN2_MASK_ONLY'])
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------------------
# C features and compose vs the Python model, per grid
# ---------------------------------------------------------------------------

_WRAPPER = r'''
#include <stdlib.h>
#include "dfn2_process.h"

void *parity_dfn_create(const float *fwd, const float *inv) {
    DFN2State *state = (DFN2State *)calloc(1, sizeof(*state));
    if (!state) return NULL;
    dfn2_state_init(state, NULL, NULL);
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


def _fp(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


@pytest.fixture(scope='module', params=_GRID_SIMD, ids=_GRID_SIMD_IDS)
def grid_lib(request, tmp_path_factory):
    name, simd = request.param
    flags = GRIDS[name][0]
    work = tmp_path_factory.mktemp('dfn2_grid_%s_%d' % (name, simd))
    wrapper = work / 'wrapper.c'
    wrapper.write_text(_WRAPPER, encoding='utf-8')
    library = work / ('grid.dylib' if sys.platform == 'darwin' else 'grid.so')
    shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
    built = _cc([shared, '-fPIC', *dfn_pairs.house_cflags(simd), *flags,
                 '-I', str(DFN2), str(wrapper), str(DFN2 / 'dfn2_process.c'),
                 dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(library)])
    assert built.returncode == 0, built.stderr
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.parity_dfn_create.argtypes = [fp, fp]
    lib.parity_dfn_create.restype = ctypes.c_void_p
    lib.parity_dfn_destroy.argtypes = [ctypes.c_void_p]
    lib.parity_dfn_features.argtypes = [ctypes.c_void_p, fp, fp, fp, fp]
    lib.parity_dfn_compose.argtypes = [
        ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_float, fp, fp]
    lib.parity_dfn_compose.restype = ctypes.c_int
    return name, lib


def test_c_features_and_compose_match_python(grid_lib, dfn_py):
    name, lib = grid_lib
    _flags, sr, n_fft, n_erb, df_bins = GRIDS[name]
    model_mod, train, _export = dfn_py
    hop, bins = n_fft // 2, n_fft // 2 + 1
    cfg = _grid_config(sr, n_fft, n_erb, df_bins)
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

    state = lib.parity_dfn_create(_fp(fwd), _fp(inv))
    assert state
    try:
        c_erb = np.empty((frames, n_erb), np.float32)
        c_spec = np.empty((frames, 2 * df_bins), np.float32)
        for frame in range(frames):
            lib.parity_dfn_features(state, _fp(real[frame]), _fp(imag[frame]),
                                    _fp(c_erb[frame]), _fp(c_spec[frame]))
    finally:
        lib.parity_dfn_destroy(state)
    np.testing.assert_allclose(c_erb, py_erb[0, 0].numpy(),
                               rtol=2e-5, atol=3e-6)
    expected_spec = py_spec[0].permute(1, 0, 2).reshape(frames, 2 * df_bins).numpy()
    np.testing.assert_allclose(c_spec, expected_spec, rtol=2e-5, atol=3e-6)

    model = model_mod.DeepFilterNet2(**train.read_model_config(cfg)).eval()
    mask = torch.sigmoid(torch.randn(1, 1, frames, n_erb))
    coefs = torch.randn(1, frames, df_bins, 10) * 0.1
    alpha = torch.sigmoid(torch.randn(1, frames, 1))
    with torch.no_grad():
        expected = model.compose(spectrum, mask, coefs, alpha)[0]

    state = lib.parity_dfn_create(_fp(fwd), _fp(inv))
    assert state
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
        got, expected[:, :frames - model.df_lookahead].numpy(),
        rtol=4e-5, atol=8e-6)


# ---------------------------------------------------------------------------
# C host stream vs the offline model, per grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', params=_GRID_SIMD, ids=_GRID_SIMD_IDS)
def stream_driver(request, tmp_path_factory):
    name, simd = request.param
    work = tmp_path_factory.mktemp('dfn2_grid_stream_%s_%d' % (name, simd))
    source = work / 'driver.c'
    source.write_text(stream_tests._STREAM_DRIVER, encoding='utf-8')
    exe = work / 'driver'
    built = _cc([*dfn_pairs.house_cflags(simd), *GRIDS[name][0], str(source),
                 str(dfn_pairs.source_path(FAMILY, 'process')),
                 str(dfn_pairs.source_path(FAMILY, 'model_io')),
                 dfn_pairs.audio_common_lib(simd), '-lm', '-o', str(exe)])
    assert built.returncode == 0, built.stderr
    return name, exe


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
def test_c_host_stream_reproduces_the_offline_model(
        tmp_path, dfn_py, stream_driver, atten_lim):
    name, exe = stream_driver
    _flags, sr, n_fft, n_erb, df_bins = GRIDS[name]
    model_mod, train, export = dfn_py
    pair = (1, 1)
    cfg = _grid_config(sr, n_fft, n_erb, df_bins)
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

    source = tmp_path / 'stream_in.bin'
    result = tmp_path / 'stream_out.bin'
    stream_tests._write_stream_input(
        source, model, frames, erb, spec_feat, spec, (mask, coefs, alpha),
        (erb_windows, spec_windows))
    done = subprocess.run([str(exe), str(source), str(result), str(atten_lim)],
                          capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert 'window_bad=0 ready_bad=0 emitted=%d' % frames in done.stdout, (
        done.stdout)

    record = 4 + 2 * 4 * model.n_bins
    raw = result.read_bytes()
    assert len(raw) == frames * record
    got = np.zeros((model.n_bins, frames), dtype=np.complex64)
    for index in range(frames):
        chunk = raw[index * record:(index + 1) * record]
        assert int(np.frombuffer(chunk[:4], '<i4')[0]) == index
        values = np.frombuffer(chunk[4:], '<f4')
        got[:, index] = values[:model.n_bins] + 1j * values[model.n_bins:]
    want = reference[0].numpy()
    if atten_lim:
        mix = 10 ** (-abs(atten_lim) / 20)
        want = spec[0].numpy() * mix + want * (1 - mix)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    assert float(np.abs(want - spec[0].numpy()).max()) > 1e-2


# ---------------------------------------------------------------------------
# objects of different grids do not link; the pool identity follows the grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('first,second', [('16k', '48k'),
                                          ('16k', 'custom16k_df64')],
                         ids=['16k-48k', 'df_bins-96-64'])
def test_units_of_different_grids_do_not_link(builds, first, second):
    objs = {name: builds.objects(GRIDS[name][0]) for name in (first, second)}
    for name in (first, second):                         # the controls link
        done, _exe = builds.link(_LINK_MAIN, objs[name], 'same_' + name)
        assert done.returncode == 0, done.stderr
    # dfn2_process.c of one grid under the other units of another: the units name
    # the variant symbol of their own grid and the process unit does not define it.
    for process, others in ((first, second), (second, first)):
        mixed = [objs[process][0], *objs[others][1:]]
        done, _exe = builds.link(_LINK_MAIN, mixed, 'mix_%s_process' % process)
        _assert_no_link(done, _variant_symbol(others))


def test_an_application_built_for_another_grid_than_the_library_does_not_link(
        builds):
    """A grid chosen in the application's own file alone does not reach the
    library objects; the link check turns that into a build error."""
    lib = builds.objects([])
    done, _exe = builds.link(_APP_MAIN, lib, 'app48')
    assert done.returncode == 0, done.stderr                    # the control
    done, _exe = builds.link(_APP_MAIN, lib, 'app16', GRIDS['16k'][0])
    _assert_no_link(done, _variant_symbol('16k'))


def test_pool_identity_follows_the_grid(builds):
    shipped = builds.identity([])
    # The shipped pool identity is the historical one: a change here is a change of
    # the shipped pool layout or of what its hash covers, and is made on purpose.
    assert shipped == ['219536', 'd2d3bc45'], shipped
    assert builds.identity(['-DDFN2_GRID_KHZ=48']) == shipped
    others = [builds.identity(GRIDS[name][0])
              for name in ('32k', '16k', 'custom24k', 'custom16k_df64')]
    assert len({tuple(v) for v in [shipped, *others]}) == 5


@pytest.mark.parametrize('which', ['erb_alpha', 'spec_alpha'])
def test_each_alpha_is_in_the_pool_identity(builds, which):
    """Same integers, another tau for one of the two EMAs: another pool identity,
    so neither alpha can drop out of the hash unseen."""
    retuned = builds.identity(_external(**{which: 0.979}))
    assert retuned[1] != builds.identity(_CUSTOM_24K)[1]


@pytest.mark.parametrize('flag', _NORMALISER_STARTS, ids=_macro_name)
def test_every_normaliser_start_is_in_the_pool_identity(builds, flag):
    base = builds.identity(_CUSTOM_24K)
    assert builds.identity([*_CUSTOM_24K, flag])[1] != base[1], flag


@pytest.mark.parametrize('flag', _NORMALISER_STARTS, ids=_macro_name)
def test_a_normaliser_start_on_the_48k_preset_makes_it_another_grid(builds, flag):
    """The 48 kHz preset is the shipped grid only while the build line leaves the
    normaliser starting points alone: with one overridden it has the suffixed link
    symbol and its starting points in the pool hash."""
    assert builds.probe([])[-1] == 1                              # the control
    assert builds.probe([flag])[-1] == 0
    overridden = builds.identity([flag])
    assert overridden[1] != builds.identity([])[1]
    # Against a build of the same integers and alphas with every starting point at
    # its default, this one starting point is what the hash tells apart.
    assert overridden[1] != builds.identity(_SHIPPED_INTEGERS)[1]
    shipped_objs, overridden_objs = builds.objects([]), builds.objects([flag])
    mixed = [shipped_objs[0], *overridden_objs[1:]]
    done, _exe = builds.link(_LINK_MAIN, mixed, 'mix_shipped_process')
    _assert_no_link(done, _OTHER_48K_SYMBOL)
    mixed = [overridden_objs[0], *shipped_objs[1:]]
    done, _exe = builds.link(_LINK_MAIN, mixed, 'mix_overridden_process')
    _assert_no_link(done, _variant_symbol('48k'))


def test_a_grid_with_the_shipped_integers_is_still_another_grid(builds):
    """Defined on the build line, the shipped integers do not make the shipped
    grid: its alphas are the definer's, so it has its own link symbol and pool
    hash."""
    assert builds.probe(_SHIPPED_INTEGERS)[-1] == 0
    assert builds.identity(_SHIPPED_INTEGERS)[1] != builds.identity([])[1]
    # And its objects do not link under the preset's, nor the preset's under it.
    done, _exe = builds.link(
        _APP_MAIN, builds.objects(_SHIPPED_INTEGERS), 'app_preset')
    _assert_no_link(done, _variant_symbol('48k'))
    done, _exe = builds.link(
        _APP_MAIN, builds.objects([]), 'app_external', _SHIPPED_INTEGERS)
    _assert_no_link(done, _OTHER_48K_SYMBOL)


def test_the_mask_only_variant_follows_the_grid_too(builds):
    for name in ('16k', 'custom24k', 'custom16k_df64'):
        flags = GRIDS[name][0]
        full = builds.identity(flags)
        mask = builds.identity([*flags, '-DDFN2_MASK_ONLY'])
        assert int(mask[0]) < int(full[0]), (name, full, mask)
        assert mask[1] != full[1], (name, full, mask)


@pytest.mark.parametrize('name', ['48k', '16k'])
def test_mask_only_and_full_units_of_one_grid_do_not_link(builds, name):
    """The variant is part of the link next to the grid: a mask-only unit and a full
    unit of the same grid name different symbols, whether the mix is inside the
    library or between an application and the library (the application of the
    board sample picks its variant with -DDFN2_MASK_ONLY, as the library does)."""
    flags = GRIDS[name][0]
    mask = builds.objects([*flags, '-DDFN2_MASK_ONLY'])
    full = builds.objects(flags)
    full_symbol = _variant_symbol(name)
    mask_symbol = _variant_symbol(name, 'mask_only')
    done, _exe = builds.link(_LINK_MAIN, [mask[0], *full[1:]], 'mix_mask_process')
    _assert_no_link(done, full_symbol)
    done, _exe = builds.link(_LINK_MAIN, [full[0], *mask[1:]], 'mix_full_process')
    _assert_no_link(done, mask_symbol)
    done, _exe = builds.link(_APP_MAIN, full, 'app_full_on_full', flags)
    assert done.returncode == 0, done.stderr                    # the control
    done, _exe = builds.link(_APP_MAIN, full, 'app_mask_on_full',
                             [*flags, '-DDFN2_MASK_ONLY'])
    _assert_no_link(done, mask_symbol)
    done, _exe = builds.link(_APP_MAIN, mask, 'app_full_on_mask', flags)
    _assert_no_link(done, full_symbol)
