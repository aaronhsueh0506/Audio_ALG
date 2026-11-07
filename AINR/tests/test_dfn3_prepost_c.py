"""DFN3 pre/post class: accessor and ERB-matrix swap contract through ctypes.

``dfn3_prepost.c`` is built once per SIMD setting as a shared library with the two
translation units it composes and the audiolib archive, then driven through its
public C API only. The lookahead pair is a field of the ``DFN3Grid`` each instance is
created for, so every pair the trainer accepts runs against the same library. Two
groups of contract:

* the accessors (``_hop_size``, ``_num_bins``) report the fixed 48 kHz / 1024 / 512
  grid and answer -1 for a NULL instance;
* ``_set_erb_matrices`` ("BETWEEN HOPS ONLY") is atomic per hop: a swap before
  the first hop is indistinguishable from constructing with the new pair, a
  mid-stream swap leaves the rows already in the graph window untouched, and a
  refused call (NULL argument, or a frame still open) changes nothing.

Every hop opens a frame, the first ``mask_lookahead`` of them being warm-up frames
whose heads are discarded; the first output comes ``max(mask, df)`` hops in.

Comparisons between instances are ``tobytes()`` equality, never a tolerance, on
both halves of the pair: the forward matrix through the ERB feature window and
the inverse matrix through the enhanced spectrum.
"""
import configparser
import ctypes
import hashlib
import subprocess
import sys

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet3'
DFN = dfn_pairs.family_dir(FAMILY)

LOOKAHEAD_PAIRS = dfn_pairs.accepted_pairs(FAMILY)
SHIPPED_PAIR = dfn_pairs.shipped_pair(FAMILY)

HOP = 512
BINS = 513
N_ERB = 32
FRAMES_IN_WINDOW = 3
GRU_ELEMENTS = 5 * 256               # h_gru (1, 5, 1, 256)
CONVP_ELEMENTS = 64 * 5 * 96         # df_convp_history (1, 64, 5, 96)
SOURCES = ('prepost', 'process', 'model_io')

# The shipped 48 kHz / 1024-point ERB band borders (N_ERB + 0 entries; the
# last border is the bin count).
_BORDERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
            50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
            325, 378, 440, 513)

_FP = ctypes.POINTER(ctypes.c_float)

_PROBE = r'''
#include <stddef.h>
#include "DeepFilterNet3/dfn3_prepost.h"
size_t probe_sizeof_spec(void) { return sizeof(DFN3GridSpec); }
size_t probe_sizeof_grid(void) { return sizeof(DFN3Grid); }
size_t probe_sizeof_config(void) { return sizeof(DFN3PrepostConfig); }
size_t probe_offsetof_erb_fwd(void) { return offsetof(DFN3PrepostConfig, erb_fwd); }
'''


def _fp(array):
    return array.ctypes.data_as(_FP)


class _GridSpec(ctypes.Structure):
    """DFN3GridSpec, field for field."""
    _fields_ = [('sample_rate', ctypes.c_int), ('n_fft', ctypes.c_int),
                ('n_erb', ctypes.c_int), ('df_bins', ctypes.c_int),
                ('df_order', ctypes.c_int), ('mask_lookahead', ctypes.c_int),
                ('df_lookahead', ctypes.c_int),
                ('erb_norm_tau_sec', ctypes.c_float),
                ('erb_norm_init_lo_db', ctypes.c_float),
                ('erb_norm_init_hi_db', ctypes.c_float),
                ('erb_norm_scale_db', ctypes.c_float),
                ('spec_norm_tau_sec', ctypes.c_float),
                ('spec_norm_init_lo', ctypes.c_float),
                ('spec_norm_init_hi', ctypes.c_float)]


class _Grid(ctypes.Structure):
    """DFN3Grid, field for field."""
    _fields_ = [(name, ctypes.c_int) for name in (
        'sample_rate', 'n_fft', 'hop', 'n_bins', 'n_erb', 'df_bins',
        'df_order', 'mask_lookahead', 'df_lookahead', 'output_delay',
        'df_history', 'ring')] + [(name, ctypes.c_float) for name in (
            'erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
            'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
            'spec_norm_init_hi', 'erb_norm_alpha', 'spec_norm_alpha',
            'stft_scale', 'stft_unscale')]


class _Config(ctypes.Structure):
    """DFN3PrepostConfig, field for field."""
    _fields_ = [('grid', _Grid),
                ('erb_fwd', ctypes.c_void_p),
                ('erb_inv', ctypes.c_void_p),
                ('atten_lim_db', ctypes.c_float),
                ('gru_hidden', ctypes.c_void_p),
                ('gru_hidden_elements', ctypes.c_size_t),
                ('df_convp_history', ctypes.c_void_p),
                ('df_convp_history_elements', ctypes.c_size_t)]


class _Inputs(ctypes.Structure):
    _fields_ = [('erb_window', ctypes.c_void_p),
                ('spec_window', ctypes.c_void_p),
                ('gru_hidden', ctypes.c_void_p),
                ('df_convp_window', ctypes.c_void_p),
                ('erb_window_elements', ctypes.c_size_t),
                ('spec_window_elements', ctypes.c_size_t),
                ('gru_hidden_elements', ctypes.c_size_t),
                ('df_convp_window_elements', ctypes.c_size_t)]


class _Outputs(ctypes.Structure):
    _fields_ = [('erb_mask', ctypes.c_void_p),
                ('coefs', ctypes.c_void_p),
                ('gru_hidden_next', ctypes.c_void_p),
                ('df_convp_window_next', ctypes.c_void_p),
                ('erb_mask_elements', ctypes.c_size_t),
                ('coefs_elements', ctypes.c_size_t),
                ('gru_hidden_elements', ctypes.c_size_t),
                ('df_convp_window_elements', ctypes.c_size_t)]


def _grid_spec(pair):
    """The shipped model's grid straight from config.ini, with ``pair`` as its lookahead."""
    cfg = configparser.ConfigParser()
    cfg.read(DFN / 'config.ini')
    # The tests below size their arrays for this grid.
    assert (cfg.getint('model', 'n_erb'), cfg.getint('model', 'df_bins'),
            cfg.getint('model', 'df_order')) == (N_ERB, 96, 5)
    return _GridSpec(
        sample_rate=cfg.getint('signal', 'sr'),
        n_fft=cfg.getint('signal', 'n_fft'),
        n_erb=cfg.getint('model', 'n_erb'),
        df_bins=cfg.getint('model', 'df_bins'),
        df_order=cfg.getint('model', 'df_order'),
        mask_lookahead=pair[0], df_lookahead=pair[1],
        **{key: cfg.getfloat('feature', key) for key in (
            'erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
            'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
            'spec_norm_init_hi')})


def _erb_pair(reverse_bands=False):
    """Bin-major forward and band-major inverse matrices, float32.

    ``reverse_bands`` mirrors the band axis: a valid pair with exactly the same
    structure but different values, so a swap is visible in the features and in
    the mask expansion without any tolerance judgement.
    """
    fwd = np.zeros((BINS, N_ERB), np.float32)
    inv = np.zeros((N_ERB, BINS), np.float32)
    segment = 0
    for k in range(BINS):
        while segment + 1 < N_ERB - 1 and k >= _BORDERS[segment + 1]:
            segment += 1
        lo, hi = _BORDERS[segment], _BORDERS[segment + 1]
        right = np.float32(k - lo) / np.float32(hi - lo)
        left = np.float32(1.0) - right
        fleft, fright = left, right
        if segment == 0:
            fleft = fleft * np.float32(2.0)
        if segment + 1 == N_ERB - 1:
            fright = fright * np.float32(2.0)
        fwd[k, segment] = fleft
        fwd[k, segment + 1] = fright
        inv[segment, k] = left
        inv[segment + 1, k] = right
    if reverse_bands:
        fwd = fwd[:, ::-1]
        inv = inv[::-1, :]
    return np.ascontiguousarray(fwd), np.ascontiguousarray(inv)


@pytest.fixture(scope='module', params=(0, 1),
                ids=lambda simd: 'simd' if simd else 'scalar')
def dll(request, tmp_path_factory):
    """The class built once for a SIMD setting; no lookahead pair is compiled in."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    simd = request.param
    archive = dfn_pairs.audio_common_lib(simd)
    work = tmp_path_factory.mktemp(f'dfn3_prepost_{simd}')
    probe = work / 'probe.c'
    probe.write_text(_PROBE)
    library = work / ('dfn3_prepost.dylib' if sys.platform == 'darwin'
                      else 'dfn3_prepost.so')
    subprocess.run(['cc', *dfn_pairs.house_cflags(simd),
                    '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                    '-fPIC', str(probe)]
                   + [str(dfn_pairs.source_path(FAMILY, kind))
                      for kind in SOURCES]
                   + [archive, '-lm', '-o', str(library)],
                   check=True, capture_output=True)
    dll = ctypes.CDLL(str(library))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    for probe in (dll.probe_sizeof_spec, dll.probe_sizeof_grid,
                  dll.probe_sizeof_config, dll.probe_offsetof_erb_fwd):
        probe.restype = ctypes.c_size_t
    # A field-order slip in a ctypes mirror would be silent undefined behaviour.
    assert ctypes.sizeof(_GridSpec) == dll.probe_sizeof_spec()
    assert ctypes.sizeof(_Grid) == dll.probe_sizeof_grid()
    assert ctypes.sizeof(_Config) == dll.probe_sizeof_config()
    assert _Config.erb_fwd.offset == dll.probe_offsetof_erb_fwd()
    dll.dfn3_grid_make.argtypes = [ctypes.POINTER(_Grid),
                                   ctypes.POINTER(_GridSpec)]
    dll.dfn3_grid_make.restype = ci
    dll.dfn3_grid_validate.argtypes = [ctypes.POINTER(_Grid)]
    dll.dfn3_grid_validate.restype = ci
    dll.dfn3_prepost_config_defaults.argtypes = [ctypes.POINTER(_Config),
                                                 ctypes.POINTER(_Grid)]
    dll.dfn3_prepost_config_defaults.restype = ci
    dll.dfn3_prepost_create.argtypes = [ctypes.POINTER(_Config)]
    dll.dfn3_prepost_create.restype = vp
    dll.dfn3_prepost_destroy.argtypes = [vp]
    dll.dfn3_prepost_reset.argtypes = [vp]
    dll.dfn3_prepost_state_generation.argtypes = [vp]
    dll.dfn3_prepost_state_generation.restype = ctypes.c_uint
    for name in ('hop_size', 'num_bins'):
        getter = getattr(dll, f'dfn3_prepost_{name}')
        getter.argtypes = [vp]
        getter.restype = ci
    dll.dfn3_prepost_set_erb_matrices.argtypes = [vp, vp, vp]
    dll.dfn3_prepost_set_erb_matrices.restype = ci
    dll.dfn3_prepost_pre_process_freq.argtypes = [vp, _FP, _FP]
    dll.dfn3_prepost_pre_process_freq.restype = ci
    dll.dfn3_prepost_frame_open.argtypes = [
        vp, ctypes.POINTER(_Inputs), ctypes.POINTER(_Outputs)]
    dll.dfn3_prepost_frame_open.restype = ci
    dll.dfn3_prepost_frame_finish.argtypes = [vp]
    dll.dfn3_prepost_frame_finish.restype = ci
    dll.dfn3_prepost_frame_bypass.argtypes = [vp]
    dll.dfn3_prepost_frame_bypass.restype = ci
    dll.dfn3_prepost_post_process_freq.argtypes = [
        vp, _FP, _FP, ctypes.POINTER(ci)]
    dll.dfn3_prepost_post_process_freq.restype = ci
    dll.dfn3_prepost_output_frame_index.argtypes = [
        vp, ctypes.POINTER(ctypes.c_longlong)]
    dll.dfn3_prepost_output_frame_index.restype = ci
    return dll


class _Lib:
    """The shared library plus the lookahead pair under test and its grid."""

    def __init__(self, dll, pair):
        self._dll = dll
        self.mask_lookahead, self.df_lookahead = pair
        self.output_delay = max(pair)
        self.grid = _Grid()
        spec = _grid_spec(pair)
        assert dll.dfn3_grid_make(ctypes.byref(self.grid),
                                  ctypes.byref(spec)) == 0
        assert dll.dfn3_grid_validate(ctypes.byref(self.grid)) == 0
        assert (self.grid.hop, self.grid.n_bins) == (HOP, BINS)
        assert self.grid.output_delay == self.output_delay

    def __getattr__(self, name):
        return getattr(self._dll, name)


@pytest.fixture(scope='module', params=LOOKAHEAD_PAIRS,
                ids=dfn_pairs.pair_id)
def lib(dll, request):
    return _Lib(dll, request.param)


class _Instance:
    """One DFN3Prepost plus the arrays it borrows."""

    def __init__(self, dll, fwd, inv, registered=False):
        self.dll = dll
        self.fwd, self.inv = fwd, inv     # borrowed by the C object: keep alive
        cfg = _Config()
        assert dll.dfn3_prepost_config_defaults(
            ctypes.byref(cfg), ctypes.byref(dll.grid)) == 0
        cfg.erb_fwd = ctypes.cast(_fp(fwd), ctypes.c_void_p)
        cfg.erb_inv = ctypes.cast(_fp(inv), ctypes.c_void_p)
        # The recurrent state, when registered, is the caller's (borrowed).
        self.gru = np.zeros(GRU_ELEMENTS if registered else 0, np.float32)
        self.convp = np.zeros(CONVP_ELEMENTS if registered else 0, np.float32)
        if registered:
            cfg.gru_hidden = ctypes.cast(_fp(self.gru), ctypes.c_void_p)
            cfg.gru_hidden_elements = self.gru.size
            cfg.df_convp_history = ctypes.cast(_fp(self.convp),
                                               ctypes.c_void_p)
            cfg.df_convp_history_elements = self.convp.size
        self.handle = dll.dfn3_prepost_create(ctypes.byref(cfg))
        assert self.handle
        rng = np.random.default_rng(4101)
        # Fixed heads, identical for every instance: a ramp over the bands and
        # small deep-filter taps.  What differs between instances is then only
        # the matrix pair.
        self.mask = np.linspace(0.1, 1.0, N_ERB).astype(np.float32)
        self.coefs = rng.normal(0.0, 0.05, 96 * 5 * 2).astype(np.float32)

    def close(self):
        self.dll.dfn3_prepost_destroy(self.handle)
        self.handle = None

    def generation(self):
        return self.dll.dfn3_prepost_state_generation(self.handle)

    def state_is_zero(self):
        return not self.gru.any() and not self.convp.any()

    def pre(self, re, im):
        return self.dll.dfn3_prepost_pre_process_freq(
            self.handle, _fp(re), _fp(im))

    def erb_window(self):
        inputs, outputs = _Inputs(), _Outputs()
        assert self.dll.dfn3_prepost_frame_open(
            self.handle, ctypes.byref(inputs), ctypes.byref(outputs)) == 0
        count = inputs.erb_window_elements
        assert count == FRAMES_IN_WINDOW * N_ERB
        view = (ctypes.c_float * count).from_address(inputs.erb_window)
        return np.frombuffer(view, np.float32).copy(), outputs

    def commit_fixed_heads(self, outputs):
        assert outputs.erb_mask_elements == N_ERB
        assert outputs.coefs_elements == self.coefs.size
        ctypes.memmove(outputs.erb_mask, self.mask.ctypes.data,
                       self.mask.nbytes)
        ctypes.memmove(outputs.coefs, self.coefs.ctypes.data,
                       self.coefs.nbytes)
        return self.dll.dfn3_prepost_frame_finish(self.handle)

    def post(self):
        re = np.empty(BINS, np.float32)
        im = np.empty(BINS, np.float32)
        valid = ctypes.c_int(-9)
        assert self.dll.dfn3_prepost_post_process_freq(
            self.handle, _fp(re), _fp(im), ctypes.byref(valid)) == 0
        return valid.value, re, im

    def hop(self, re, im):
        """One full frame; returns (erb_window, valid, out_re, out_im)."""
        assert self.pre(re, im) == 1
        window, outputs = self.erb_window()
        assert self.commit_fixed_heads(outputs) == 0
        valid, out_re, out_im = self.post()
        return window, valid, out_re, out_im


def _spectra(count, seed=937):
    rng = np.random.default_rng(seed)
    re = rng.normal(0, 0.15, (count, BINS)).astype(np.float32)
    im = rng.normal(0, 0.15, (count, BINS)).astype(np.float32)
    return re, im


def _run(instance, re, im, first=0, last=None, swap_at=None, swap_to=None):
    """Drive frames [first, last); optionally swap the pair before ``swap_at``."""
    last = len(re) if last is None else last
    trace = []
    for t in range(first, last):
        if swap_at == t:
            assert instance.dll.dfn3_prepost_set_erb_matrices(
                instance.handle, _fp(swap_to[0]), _fp(swap_to[1])) == 0
            instance.fwd, instance.inv = swap_to      # keep the new pair alive
        window, valid, out_re, out_im = instance.hop(re[t], im[t])
        trace.append((window.tobytes(), valid, out_re.tobytes(),
                      out_im.tobytes()))
    return trace


@pytest.fixture
def pair_a():
    return _erb_pair(False)


@pytest.fixture
def pair_b():
    return _erb_pair(True)


def test_accessors_report_the_fixed_grid(lib, pair_a):
    inst = _Instance(lib, *pair_a)
    try:
        assert lib.dfn3_prepost_hop_size(inst.handle) == HOP
        assert lib.dfn3_prepost_num_bins(inst.handle) == BINS
    finally:
        inst.close()


def test_accessors_do_not_change_with_stream_state(lib, pair_a):
    inst = _Instance(lib, *pair_a)
    try:
        re, im = _spectra(4)
        _run(inst, re, im)
        assert lib.dfn3_prepost_hop_size(inst.handle) == HOP
        assert lib.dfn3_prepost_num_bins(inst.handle) == BINS
    finally:
        inst.close()


def test_warm_up_is_the_lookahead_and_the_window_is_zero_padded(lib, pair_a):
    """Every hop opens a frame, the warm-up frames (t < mask_lookahead)
    included; hop t's window holds frames [t-2 .. t], the frames before the
    stream start being zero rows; the first output is source frame 0, so the
    delay is max(mask, df) hops."""
    mask, delay = lib.mask_lookahead, lib.output_delay
    inst = _Instance(lib, *pair_a)
    try:
        re, im = _spectra(delay + 5)
        trace = _run(inst, re, im)
        windows = []
        for t, (window, valid, _re, _im) in enumerate(trace):
            assert valid == int(t >= delay), t
            rows = np.frombuffer(window, np.float32).reshape(
                FRAMES_IN_WINDOW, N_ERB)
            pad = max(FRAMES_IN_WINDOW - 1 - t, 0)
            assert not rows[:pad].any(), t
            assert rows[pad:].any(axis=1).all(), t
            windows.append(rows)
        # The newest row moves one frame per hop: the next window drops its
        # oldest row and keeps the other two, warm-up frames included.
        for t in range(len(windows) - 1):
            assert windows[t + 1][:-1].tobytes() == windows[t][1:].tobytes(), t
        # The first real frame (hop mask) holds frames [-(2-mask) .. mask].
        pad = FRAMES_IN_WINDOW - 1 - mask
        assert not windows[mask][:pad].any()
        assert windows[mask][pad:].any(axis=1).all()
        frame = ctypes.c_longlong(-1)
        assert lib.dfn3_prepost_output_frame_index(
            inst.handle, ctypes.byref(frame)) == 0
        assert frame.value == len(trace) - 1 - delay
    finally:
        inst.close()


# Digests of a fixed 12-hop stream (fixed heads, _spectra(12)) recorded with the warm-up hops not opened
# (pre_process_freq returned 0 there): the real frames' ERB windows and every hop's valid flag and enhanced
# spectrum, per pair (the SIMD and the scalar build agree).
_RECORDED_STREAM_DIGESTS = {
    (0, 0): '6a0c1669bd6f53f2',
    (0, 1): '0d38db3108381ef7',
    (0, 2): '7d404445cde0e1c3',
    (0, 3): '7fe3190fe31ef185',
    (0, 4): '87accc05d84a562d',
    (1, 0): 'b41b51da7b001117',
    (1, 1): '67ed770129d8aef3',
    (1, 2): 'e0bd72213e6cbb5c',
    (1, 3): '57eff16f471323e6',
    (1, 4): '006cc0208aef35f0',
    (2, 0): '525e4274c0d97861',
    (2, 1): '6496e2d2ad2527ec',
    (2, 2): '26d1b18180527f22',
    (2, 3): 'b7f03161b2d54ef0',
    (2, 4): '83d4b2dd8610e1d3',
}


def _stream_digest(lib, pair_a):
    inst = _Instance(lib, *pair_a)
    try:
        re, im = _spectra(12)
        digest = hashlib.sha256()
        for t, (window, valid, out_re, out_im) in enumerate(
                _run(inst, re, im)):
            if t >= lib.mask_lookahead:
                digest.update(window)
            digest.update(bytes([valid]) + out_re + out_im)
        return digest.hexdigest()[:16]
    finally:
        inst.close()


def test_real_frame_windows_and_audio_equal_the_recorded_stream(
        lib, pair_a):
    """The warm-up frames change return codes and counters only: the windows
    of the real frames and the enhanced spectra are bit-identical to the
    recorded stream (the digests above)."""
    want = _RECORDED_STREAM_DIGESTS[(lib.mask_lookahead, lib.df_lookahead)]
    assert _stream_digest(lib, pair_a) == want


@pytest.mark.parametrize('how', ['finish_nan_heads', 'finish_nan_state',
                                 'bypass'])
def test_warm_up_frames_discard_heads_and_zero_the_registered_state(
        lib, pair_a, how):
    """A warm-up frame is closed whatever its heads hold (they are never
    read, so NaN is fine) and whatever the accelerator wrote into the
    registered state: the state is zero afterwards, the generation moved by
    exactly one (also with a non-finite state) and nothing is emitted. The
    first real frame starts from zero state and its heads are read."""
    mask = lib.mask_lookahead
    inst = _Instance(lib, *pair_a, registered=True)
    try:
        re, im = _spectra(mask + 2)
        gen0 = inst.generation()
        assert gen0 == 1
        for t in range(mask):
            assert inst.pre(re[t], im[t]) == 1
            _window, outputs = inst.erb_window()
            assert np.isnan(
                (ctypes.c_float * N_ERB).from_address(outputs.erb_mask)[0])
            inst.gru[:] = 7.0
            inst.convp[:] = np.nan if how == 'finish_nan_state' else 7.0
            if how == 'bypass':
                assert lib.dfn3_prepost_frame_bypass(inst.handle) == 0
            else:
                assert lib.dfn3_prepost_frame_finish(inst.handle) == 0
            assert inst.state_is_zero()
            assert inst.generation() == gen0 + t + 1
            # closed: a second close is refused and changes nothing
            assert lib.dfn3_prepost_frame_finish(inst.handle) == -1
            assert lib.dfn3_prepost_frame_bypass(inst.handle) == -1
            assert inst.generation() == gen0 + t + 1
            valid, out_re, out_im = inst.post()
            assert valid == 0 and not out_re.any() and not out_im.any()
            frame = ctypes.c_longlong(-1)
            assert lib.dfn3_prepost_output_frame_index(
                inst.handle, ctypes.byref(frame)) == -1
        # the first real frame: zero state in, its NaN heads are refused
        assert inst.pre(re[mask], im[mask]) == 1
        inst.erb_window()
        assert inst.state_is_zero()
        assert lib.dfn3_prepost_frame_finish(inst.handle) == -1
        assert inst.generation() == gen0 + mask + 1
    finally:
        inst.close()


def test_reset_restarts_the_warm_up_also_inside_a_warm_up_frame(lib, pair_a):
    mask = lib.mask_lookahead
    if mask == 0:
        pytest.skip('no warm-up frame with mask_lookahead 0')
    inst = _Instance(lib, *pair_a, registered=True)
    try:
        re, im = _spectra(mask + 1)
        assert inst.pre(re[0], im[0]) == 1          # a warm-up frame, left open
        gen = inst.generation()
        lib.dfn3_prepost_reset(inst.handle)
        assert inst.generation() == gen + 1
        assert lib.dfn3_prepost_frame_finish(inst.handle) == -1
        assert lib.dfn3_prepost_frame_bypass(inst.handle) == -1
        for t in range(mask):                       # due again, each +1
            assert inst.pre(re[t], im[t]) == 1
            assert lib.dfn3_prepost_frame_bypass(inst.handle) == 0
            assert inst.generation() == gen + 2 + t
        assert inst.pre(re[mask], im[mask]) == 1    # the real frame
        assert lib.dfn3_prepost_frame_bypass(inst.handle) == 0
        assert inst.generation() == gen + 1 + mask
    finally:
        inst.close()


def test_accessors_answer_minus_one_for_a_null_instance(lib):
    assert lib.dfn3_prepost_hop_size(None) == -1
    assert lib.dfn3_prepost_num_bins(None) == -1


def test_swap_before_the_first_hop_equals_constructing_with_the_new_pair(
        lib, pair_a, pair_b):
    re, im = _spectra(8)
    reference = _Instance(lib, *pair_b)
    swapped = _Instance(lib, *pair_a)
    untouched = _Instance(lib, *pair_a)
    try:
        expected = _run(reference, re, im)
        got = _run(swapped, re, im, swap_at=0, swap_to=pair_b)
        baseline = _run(untouched, re, im)
        assert got == expected
        # Not vacuous: the two pairs give different windows and spectra.
        assert baseline != expected
        windows_a = [w for w, *_ in baseline]
        windows_b = [w for w, *_ in expected]
        assert windows_a != windows_b
        enhanced_a = [(r, i) for _, v, r, i in baseline if v]
        enhanced_b = [(r, i) for _, v, r, i in expected if v]
        assert enhanced_a and enhanced_a != enhanced_b
    finally:
        reference.close()
        swapped.close()
        untouched.close()


def test_midstream_swap_keeps_the_rows_already_in_the_window(
        lib, pair_a, pair_b):
    re, im = _spectra(8)
    split = 4
    control = _Instance(lib, *pair_a)
    swapped = _Instance(lib, *pair_a)
    try:
        before = _run(control, re, im)
        got = _run(swapped, re, im, swap_at=split, swap_to=pair_b)
        # Everything before the swap point is bitwise the unswapped stream.
        assert got[:split] == before[:split]
        # The first hop after it slides the window: the two older rows were
        # computed under the old pair and stay, the newest row is new.
        old_window = np.frombuffer(before[split][0], np.float32)
        new_window = np.frombuffer(got[split][0], np.float32)
        rows_old = old_window.reshape(FRAMES_IN_WINDOW, N_ERB)
        rows_new = new_window.reshape(FRAMES_IN_WINDOW, N_ERB)
        assert rows_new[:-1].tobytes() == rows_old[:-1].tobytes()
        assert rows_new[-1].tobytes() != rows_old[-1].tobytes()
        assert got[split:] != before[split:]
    finally:
        control.close()
        swapped.close()


def test_swap_refuses_null_arguments_and_changes_nothing(lib, pair_a, pair_b):
    re, im = _spectra(8)
    control = _Instance(lib, *pair_a)
    probed = _Instance(lib, *pair_a)
    try:
        expected = _run(control, re, im)
        fwd_b, inv_b = pair_b
        assert lib.dfn3_prepost_set_erb_matrices(
            None, _fp(fwd_b), _fp(inv_b)) == -1
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, None, _fp(inv_b)) == -1
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, _fp(fwd_b), None) == -1
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, None, None) == -1
        assert _run(probed, re, im) == expected
    finally:
        control.close()
        probed.close()


@pytest.mark.parametrize('close_with', ['commit', 'skip'])
@pytest.mark.parametrize('open_frame', ['first_real', 'first_warm_up'])
def test_swap_is_refused_while_a_frame_is_open_and_accepted_after(
        lib, pair_a, pair_b, close_with, open_frame):
    if open_frame == 'first_warm_up' and lib.mask_lookahead == 0:
        pytest.skip('no warm-up frame with mask_lookahead 0')
    re, im = _spectra(8)
    control = _Instance(lib, *pair_a)
    probed = _Instance(lib, *pair_a)
    try:
        # hops run before the frame that is held open
        first = lib.mask_lookahead if open_frame == 'first_real' else 0
        expected = _run(control, re, im, last=first + 2)
        fwd_b, inv_b = pair_b

        for t in range(first):              # the warm-up frames are frames
            probed.hop(re[t], im[t])
        assert probed.pre(re[first], im[first]) == 1
        window, outputs = probed.erb_window()          # the frame is open
        assert window.tobytes() == expected[first][0]
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, _fp(fwd_b), _fp(inv_b)) == -1
        if close_with == 'commit':
            assert probed.commit_fixed_heads(outputs) == 0
        else:
            assert lib.dfn3_prepost_frame_bypass(probed.handle) == 0
        probed.post()

        # With the frame closed the same call is accepted ...
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, _fp(fwd_b), _fp(inv_b)) == 0
        probed.fwd, probed.inv = pair_b
        # ... and the next hop's newest feature row comes from the new pair,
        # while the two older rows were already computed and stay.
        assert probed.pre(re[first + 1], im[first + 1]) == 1
        window2, _ = probed.erb_window()
        rows_new = np.frombuffer(window2, np.float32).reshape(
            FRAMES_IN_WINDOW, N_ERB)
        rows_old = np.frombuffer(expected[first + 1][0], np.float32).reshape(
            FRAMES_IN_WINDOW, N_ERB)
        assert rows_new[:-1].tobytes() == rows_old[:-1].tobytes()
        assert rows_new[-1].tobytes() != rows_old[-1].tobytes()
    finally:
        control.close()
        probed.close()
