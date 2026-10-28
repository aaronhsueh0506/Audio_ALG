"""DFN3 pre/post class: accessor and ERB-matrix swap contract through ctypes.

``dfn3_prepost.c`` is built as a shared library with the two translation units
it composes and the audiolib archive, then driven through its public C API
only. Two groups of contract:

* the accessors (``_hop_size``, ``_num_bins``, ``_io_mode``) report the fixed
  48 kHz / 1024 / 512 grid and the mode fixed at init, and answer -1 for a NULL
  instance;
* ``_set_erb_matrices`` ("BETWEEN HOPS ONLY") is atomic per hop: a swap before
  the first hop is indistinguishable from constructing with the new pair, a
  mid-stream swap leaves the rows already in the graph window untouched, and a
  refused call (NULL argument, or a frame still open) changes nothing.

Everything runs for every pair in ``dfn_pairs.c_build_pairs``, each its own
build of the C host (-DDFN3_MASK_LOOKAHEAD, -DDFN3_DF_LOOKAHEAD; the shipped
pair is the build with no override): the first ``mask_lookahead`` hops only
fill the graph window, and the first output comes ``max(mask, df)`` hops in.

Comparisons between instances are ``tobytes()`` equality, never a tolerance, on
both halves of the pair: the forward matrix through the ERB feature window and
the inverse matrix through the enhanced spectrum.
"""
import ctypes
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet3'
DFN = dfn_pairs.family_dir(FAMILY)

LOOKAHEAD_PAIRS = dfn_pairs.c_build_pairs(FAMILY)
SHIPPED_PAIR = dfn_pairs.shipped_pair(FAMILY)

HOP = 512
BINS = 513
N_ERB = 32
FRAMES_IN_WINDOW = 3
IO_FREQ = 0
IO_TIME = 1
SOURCES = ('prepost', 'process', 'model_io')

# The shipped 48 kHz / 1024-point ERB band borders (N_ERB + 0 entries; the
# last border is the bin count).
_BORDERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
            50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
            325, 378, 440, 513)

_FP = ctypes.POINTER(ctypes.c_float)


def _fp(array):
    return array.ctypes.data_as(_FP)


class _Config(ctypes.Structure):
    _fields_ = [('io_mode', ctypes.c_int),
                ('fft', ctypes.c_void_p),
                ('window', ctypes.c_void_p),
                ('istft', ctypes.c_void_p),
                ('erb_fwd', ctypes.c_void_p),
                ('erb_inv', ctypes.c_void_p),
                ('atten_lim_db', ctypes.c_float)]


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


def _make_istft(dll, fft, ola=None):
    """The caller-owned synthesis object of DFN3_IO_TIME: audiolib's AcIstft
    built from dfn3_framing_config() over the shared FFT handle.  ``ola``
    overrides the overlap-add mode (AcStftConfig field 3), to build an object
    that is not the model's.  Returns the handle and the buffer that backs it
    (keep both alive)."""
    config = ctypes.create_string_buffer(128)
    dll.dfn3_framing_config(config, fft, None)
    if ola is not None:
        struct.pack_into('i', config, 8, ola)
    size = dll.ac_istft_get_mem_size(config)
    assert size
    raw = ctypes.create_string_buffer(size + 16)
    base = ctypes.addressof(raw)
    block = base + (-base) % 16
    handle = dll.ac_istft_init(block, size, config)
    assert handle
    return handle, raw


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


@pytest.fixture(scope='module', params=[
    (simd, pair) for pair in LOOKAHEAD_PAIRS for simd in (0, 1)],
    ids=lambda case: ('simd' if case[0] else 'scalar') + (
        '' if case[1] == SHIPPED_PAIR else '-' + dfn_pairs.pair_id(case[1])))
def lib(request, tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    simd, pair = request.param
    archive = dfn_pairs.audio_common_lib(simd)
    work = tmp_path_factory.mktemp(f'dfn3_prepost_{simd}_{pair[0]}{pair[1]}')
    flags = ['cc', *dfn_pairs.house_cflags(simd),
             *dfn_pairs.defines(FAMILY, pair)]
    library = work / ('dfn3_prepost.dylib' if sys.platform == 'darwin'
                      else 'dfn3_prepost.so')
    subprocess.run(flags + ['-dynamiclib' if sys.platform == 'darwin'
                            else '-shared', '-fPIC']
                   + [str(dfn_pairs.source_path(FAMILY, kind))
                      for kind in SOURCES]
                   + [archive, '-lm', '-o', str(library)],
                   check=True, capture_output=True)
    dll = ctypes.CDLL(str(library))
    vp, ci = ctypes.c_void_p, ctypes.c_int
    dll.fft_create.argtypes = [ci]
    dll.fft_create.restype = vp
    dll.fft_destroy.argtypes = [vp]
    dll.dfn3_framing_config.argtypes = [vp, vp, vp]
    dll.dfn3_framing_config.restype = None
    dll.ac_istft_get_mem_size.argtypes = [vp]
    dll.ac_istft_get_mem_size.restype = ctypes.c_size_t
    dll.ac_istft_init.argtypes = [vp, ctypes.c_size_t, vp]
    dll.ac_istft_init.restype = vp
    dll.dfn3_prepost_config_defaults.argtypes = [ctypes.POINTER(_Config), ci]
    dll.dfn3_prepost_config_defaults.restype = ci
    dll.dfn3_prepost_create.argtypes = [ctypes.POINTER(_Config)]
    dll.dfn3_prepost_create.restype = vp
    dll.dfn3_prepost_destroy.argtypes = [vp]
    for name in ('hop_size', 'num_bins', 'io_mode'):
        getter = getattr(dll, f'dfn3_prepost_{name}')
        getter.argtypes = [vp]
        getter.restype = ci
    dll.dfn3_prepost_set_erb_matrices.argtypes = [vp, vp, vp]
    dll.dfn3_prepost_set_erb_matrices.restype = ci
    dll.dfn3_prepost_pre_process_freq.argtypes = [vp, _FP, _FP]
    dll.dfn3_prepost_pre_process_freq.restype = ci
    dll.dfn3_prepost_frame_inputs.argtypes = [
        vp, ctypes.POINTER(_Inputs), ctypes.POINTER(_Outputs)]
    dll.dfn3_prepost_frame_inputs.restype = ci
    dll.dfn3_prepost_frame_commit.argtypes = [vp]
    dll.dfn3_prepost_frame_commit.restype = ci
    dll.dfn3_prepost_frame_skip.argtypes = [vp]
    dll.dfn3_prepost_frame_skip.restype = ci
    dll.dfn3_prepost_post_process_freq.argtypes = [
        vp, _FP, _FP, ctypes.POINTER(ci)]
    dll.dfn3_prepost_post_process_freq.restype = ci
    dll.dfn3_prepost_output_frame_index.argtypes = [
        vp, ctypes.POINTER(ctypes.c_longlong)]
    dll.dfn3_prepost_output_frame_index.restype = ci
    dll.mask_lookahead, dll.df_lookahead = pair
    dll.output_delay = max(pair)
    return dll


class _Instance:
    """One FREQ-mode DFN3Prepost plus the arrays it borrows."""

    def __init__(self, dll, fwd, inv):
        self.dll = dll
        self.fwd, self.inv = fwd, inv     # borrowed by the C object: keep alive
        cfg = _Config()
        assert dll.dfn3_prepost_config_defaults(ctypes.byref(cfg), IO_FREQ) == 0
        cfg.erb_fwd = ctypes.cast(_fp(fwd), ctypes.c_void_p)
        cfg.erb_inv = ctypes.cast(_fp(inv), ctypes.c_void_p)
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

    def pre(self, re, im):
        return self.dll.dfn3_prepost_pre_process_freq(
            self.handle, _fp(re), _fp(im))

    def erb_window(self):
        inputs, outputs = _Inputs(), _Outputs()
        assert self.dll.dfn3_prepost_frame_inputs(
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
        return self.dll.dfn3_prepost_frame_commit(self.handle)

    def post(self):
        re = np.empty(BINS, np.float32)
        im = np.empty(BINS, np.float32)
        valid = ctypes.c_int(-9)
        assert self.dll.dfn3_prepost_post_process_freq(
            self.handle, _fp(re), _fp(im), ctypes.byref(valid)) == 0
        return valid.value, re, im

    def hop(self, re, im):
        """One full frame; returns (erb_window or None, valid, out_re, out_im)."""
        started = self.pre(re, im)
        assert started in (0, 1)
        window = None
        if started:
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
        trace.append((None if window is None else window.tobytes(), valid,
                      out_re.tobytes(), out_im.tobytes()))
    return trace


@pytest.fixture
def pair_a():
    return _erb_pair(False)


@pytest.fixture
def pair_b():
    return _erb_pair(True)


def test_accessors_report_the_fixed_grid_and_the_configured_mode(lib, pair_a):
    fwd, inv = pair_a
    fft = lib.fft_create(1024)
    assert fft
    istft, istft_backing = _make_istft(lib, fft)
    try:
        for mode in (IO_FREQ, IO_TIME):
            cfg = _Config()
            assert lib.dfn3_prepost_config_defaults(ctypes.byref(cfg), mode) == 0
            cfg.erb_fwd = ctypes.cast(_fp(fwd), ctypes.c_void_p)
            cfg.erb_inv = ctypes.cast(_fp(inv), ctypes.c_void_p)
            if mode == IO_TIME:
                cfg.fft = fft
                cfg.istft = istft
            handle = lib.dfn3_prepost_create(ctypes.byref(cfg))
            assert handle
            try:
                assert lib.dfn3_prepost_hop_size(handle) == HOP
                assert lib.dfn3_prepost_num_bins(handle) == BINS
                assert lib.dfn3_prepost_io_mode(handle) == mode
            finally:
                lib.dfn3_prepost_destroy(handle)
    finally:
        lib.fft_destroy(fft)


def test_time_mode_requires_the_callers_synthesis_object(lib, pair_a):
    """DFN3_IO_TIME borrows the caller's AcIstft: without one the class is
    refused, and a synthesis object in the other overlap-add mode is refused
    too."""
    fwd, inv = pair_a
    fft = lib.fft_create(1024)
    assert fft
    istft, backing = _make_istft(lib, fft)
    wrong, wrong_backing = _make_istft(lib, fft, ola=1)
    try:
        cfg = _Config()
        assert lib.dfn3_prepost_config_defaults(
            ctypes.byref(cfg), IO_TIME) == 0
        cfg.erb_fwd = ctypes.cast(_fp(fwd), ctypes.c_void_p)
        cfg.erb_inv = ctypes.cast(_fp(inv), ctypes.c_void_p)
        cfg.fft = fft
        assert not lib.dfn3_prepost_create(ctypes.byref(cfg))
        cfg.istft = wrong
        assert not lib.dfn3_prepost_create(ctypes.byref(cfg))
        cfg.istft = istft
        handle = lib.dfn3_prepost_create(ctypes.byref(cfg))
        assert handle
        lib.dfn3_prepost_destroy(handle)
    finally:
        lib.fft_destroy(fft)


def test_accessors_do_not_change_with_stream_state(lib, pair_a):
    inst = _Instance(lib, *pair_a)
    try:
        re, im = _spectra(4)
        _run(inst, re, im)
        assert lib.dfn3_prepost_hop_size(inst.handle) == HOP
        assert lib.dfn3_prepost_num_bins(inst.handle) == BINS
        assert lib.dfn3_prepost_io_mode(inst.handle) == IO_FREQ
    finally:
        inst.close()


def test_warm_up_is_the_lookahead_and_the_window_is_zero_padded(lib, pair_a):
    """Hop t opens a frame iff t >= mask_lookahead; the first window holds
    frames [-(2-mask) .. mask], the frames before the stream start being zero
    rows; the first output is source frame 0, so the delay is
    max(mask, df) hops."""
    mask, delay = lib.mask_lookahead, lib.output_delay
    inst = _Instance(lib, *pair_a)
    try:
        re, im = _spectra(delay + 5)
        trace = _run(inst, re, im)
        for t, (window, valid, _re, _im) in enumerate(trace):
            assert (window is not None) == (t >= mask), t
            assert valid == int(t >= delay), t
        rows = np.frombuffer(trace[mask][0], np.float32).reshape(
            FRAMES_IN_WINDOW, N_ERB)
        pad = FRAMES_IN_WINDOW - 1 - mask
        assert not rows[:pad].any()
        assert rows[pad:].any(axis=1).all()
        # The newest row moves one frame per hop: the window of the next head
        # drops its oldest row and keeps the other two.
        later = np.frombuffer(trace[mask + 1][0], np.float32).reshape(
            FRAMES_IN_WINDOW, N_ERB)
        assert later[:-1].tobytes() == rows[1:].tobytes()
        frame = ctypes.c_longlong(-1)
        assert lib.dfn3_prepost_output_frame_index(
            inst.handle, ctypes.byref(frame)) == 0
        assert frame.value == len(trace) - 1 - delay
    finally:
        inst.close()


def test_accessors_answer_minus_one_for_a_null_instance(lib):
    assert lib.dfn3_prepost_hop_size(None) == -1
    assert lib.dfn3_prepost_num_bins(None) == -1
    assert lib.dfn3_prepost_io_mode(None) == -1


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
        windows_a = [w for w, *_ in baseline if w is not None]
        windows_b = [w for w, *_ in expected if w is not None]
        assert windows_a and windows_a != windows_b
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
def test_swap_is_refused_while_a_frame_is_open_and_accepted_after(
        lib, pair_a, pair_b, close_with):
    re, im = _spectra(8)
    control = _Instance(lib, *pair_a)
    probed = _Instance(lib, *pair_a)
    try:
        first = lib.mask_lookahead          # hops before the first open frame
        expected = _run(control, re, im, last=first + 2)
        fwd_b, inv_b = pair_b

        for t in range(first):              # left warm-up: no frame is opened
            probed.hop(re[t], im[t])
        assert probed.pre(re[first], im[first]) == 1
        window, outputs = probed.erb_window()          # first frame is open
        assert window.tobytes() == expected[first][0]
        assert lib.dfn3_prepost_set_erb_matrices(
            probed.handle, _fp(fwd_b), _fp(inv_b)) == -1
        if close_with == 'commit':
            assert probed.commit_fixed_heads(outputs) == 0
        else:
            assert lib.dfn3_prepost_frame_skip(probed.handle) == 0
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
