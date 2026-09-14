"""Python twin of ``audio_common/src/audio_resampler.c``.

The house resampler is a stateful rational polyphase FIR: Blackman-windowed
sinc, ``AUDIO_RESAMPLER_HALF_TAPS_PER_RATIO = 16`` taps per unit of ratio,
cutoff ``0.47 / max(up, down)`` on the up-sampled grid, unity DC pinned per
phase, and the ``next_output_tick`` machine that decides how many outputs each
input frame produces.  This port keeps the same design, the same float32
arithmetic and the same tap order, so it is the reference the non-48 kHz DFN2
path (and Experiment 0's resampler round trip) is measured against; it is not
a generic resampler.

Group delay is ``(filter_length - 1) / (2 * up)`` input frames -- 16 frames at
every 8 k / 16 k / 48 k pairing -- so a native -> 48 k -> native round trip is
exactly 32 native samples late.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

HALF_TAPS_PER_RATIO = 16
SUPPORTED_RATES = (8000, 16000, 24000, 32000, 48000)


def _libm_float_trig():
    """``cosf``/``sinf`` from the process's libm.

    numpy's float32 ``cos``/``sin`` are vectorised approximations that are not
    correctly rounded on every platform, and the C designs its taps with libm's
    ``cosf``/``sinf``; reproducing the coefficient table bit for bit means
    calling the same functions."""
    import ctypes
    libm = ctypes.CDLL(None)
    cosf = libm.cosf
    sinf = libm.sinf
    cosf.argtypes = [ctypes.c_float]
    cosf.restype = ctypes.c_float
    sinf.argtypes = [ctypes.c_float]
    sinf.restype = ctypes.c_float
    return (lambda x: np.float32(cosf(ctypes.c_float(float(x)))),
            lambda x: np.float32(sinf(ctypes.c_float(float(x)))))


_COSF, _SINF = _libm_float_trig()


class AudioResampler:
    """One channel; ``process`` consumes whole input frames like the C."""

    def __init__(self, input_rate: int, output_rate: int):
        if input_rate not in SUPPORTED_RATES or output_rate not in SUPPORTED_RATES:
            raise ValueError('unsupported rate pair')
        divisor = math.gcd(input_rate, output_rate)
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.up = output_rate // divisor
        self.down = input_rate // divisor
        ratio_max = max(self.up, self.down)
        half = HALF_TAPS_PER_RATIO * ratio_max
        self.identity = self.up == self.down
        self.filter_length = 1 if self.identity else 2 * half + 1
        self.taps_per_phase = (self.filter_length + self.up - 1) // self.up
        self.coefficients = self._design()
        self.reset()

    def _design(self) -> np.ndarray:
        up = self.up
        taps = self.taps_per_phase
        coefficients = np.zeros((up, taps), dtype=np.float32)
        if self.identity:
            coefficients[0, 0] = np.float32(1.0)
            return coefficients
        half = (self.filter_length - 1) // 2
        ratio_max = max(self.up, self.down)
        cutoff = np.float32(0.47) / np.float32(ratio_max)
        pi = np.float32(math.pi)
        for n in range(self.filter_length):
            phase = n % up
            tap = n // up
            offset = np.float32(n - half)
            position = np.float32(n) / np.float32(self.filter_length - 1)
            window = (np.float32(0.42) - np.float32(0.5) * _COSF(np.float32(2.0) * pi * position)
                      + np.float32(0.08) * _COSF(np.float32(4.0) * pi * position))
            if n == half:
                ideal = np.float32(2.0) * cutoff
            else:
                ideal = (_SINF(np.float32(2.0) * pi * cutoff * offset)
                         / (pi * offset))
            coefficients[phase, tap] = np.float32(up) * np.float32(ideal) * np.float32(window)
        for phase in range(up):
            total = np.float32(0.0)
            for tap in range(taps):
                total = total + coefficients[phase, tap]
            if abs(float(total)) > 1e-12:
                inverse = np.float32(1.0) / total
                for tap in range(taps):
                    coefficients[phase, tap] = coefficients[phase, tap] * inverse
        return coefficients

    def reset(self) -> None:
        self.history = np.zeros(self.taps_per_phase, dtype=np.float32)
        self.history_head = self.taps_per_phase - 1
        self._tap_offsets = np.arange(self.taps_per_phase)
        self.input_index = 0
        self.next_output_tick = 0

    @property
    def latency_input_frames(self) -> int:
        up = 2 * self.up
        return (self.filter_length - 1 + up // 2) // up

    def output_bound(self, input_frames: int) -> int:
        if self.identity:
            return input_frames
        return (input_frames * self.up + self.down - 1) // self.down + 1

    def _dot(self) -> np.float32:
        """Scalar tap-order accumulation, newest to oldest, as the C scalar
        kernel does it (``audio_resampler_dot`` without NEON)."""
        taps = self.taps_per_phase
        phase = int(self.next_output_tick % self.up)
        coefficients = self.coefficients[phase]
        # Circular history read descending from history_head.
        idx = (self.history_head - self._tap_offsets) % taps
        products = coefficients * self.history[idx]
        total = np.float32(0.0)
        for value in products:
            total = total + value
        return total

    def process(self, frames: np.ndarray) -> np.ndarray:
        """Consume every input frame; return the produced output frames."""
        frames = np.asarray(frames, dtype=np.float32)
        if self.identity:
            self.input_index += len(frames)
            self.next_output_tick += len(frames)
            return frames.copy()
        out = []
        taps = self.taps_per_phase
        for sample in frames:
            self.history_head += 1
            if self.history_head == taps:
                self.history_head = 0
            self.history[self.history_head] = sample
            while self.next_output_tick // self.up == self.input_index:
                out.append(self._dot())
                self.next_output_tick += self.down
            self.input_index += 1
        return np.array(out, dtype=np.float32)


class CResampler:
    """The C ``audio_resampler`` itself, through ctypes: same result as
    :class:`AudioResampler` on the scalar kernel, orders of magnitude faster.
    Used by the experiment drivers; the pure-Python twin stays the readable
    reference the tests compare against."""

    _lib = None

    @classmethod
    def library(cls):
        if cls._lib is None:
            import ctypes
            import hashlib
            import os
            import shutil
            import subprocess
            import sys
            import tempfile
            here = os.path.dirname(os.path.abspath(__file__))
            ac = os.path.join(os.path.dirname(os.path.dirname(here)), 'audio_common')
            source = os.path.join(ac, 'src', 'audio_resampler.c')
            tag = hashlib.sha1(open(source, 'rb').read()).hexdigest()[:12]
            work = os.path.join(tempfile.gettempdir(), 'audio_alg_resampler_' + tag)
            os.makedirs(work, exist_ok=True)
            library = os.path.join(work, 'rs.dylib' if sys.platform == 'darwin' else 'rs.so')
            if not os.path.exists(library):
                cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
                shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
                subprocess.run(
                    [cc, shared, '-fPIC', '-O2', '-std=c11', '-ffp-contract=off',
                     '-fno-math-errno', '-DSIMD_KERNELS_FORCE_SCALAR',
                     '-I', os.path.join(ac, 'include'), source, '-lm', '-o', library],
                    check=True, capture_output=True)
            lib = ctypes.CDLL(library)
            fp = ctypes.POINTER(ctypes.c_float)
            ip = ctypes.POINTER(ctypes.c_int)
            lib.audio_resampler_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.audio_resampler_create.restype = ctypes.c_void_p
            lib.audio_resampler_destroy.argtypes = [ctypes.c_void_p]
            lib.audio_resampler_process.argtypes = [ctypes.c_void_p, fp, ctypes.c_int, fp,
                                                    ctypes.c_int, ip, ip]
            lib.audio_resampler_process.restype = ctypes.c_int
            lib.audio_resampler_output_bound.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.audio_resampler_output_bound.restype = ctypes.c_int
            lib.audio_resampler_latency_input_frames.argtypes = [ctypes.c_void_p]
            lib.audio_resampler_latency_input_frames.restype = ctypes.c_int
            lib.audio_resampler_reset.argtypes = [ctypes.c_void_p]
            cls._lib = lib
        return cls._lib

    def __init__(self, input_rate: int, output_rate: int):
        import ctypes
        self._ctypes = ctypes
        self.lib = self.library()
        self.handle = self.lib.audio_resampler_create(int(input_rate), int(output_rate), 1)
        if not self.handle:
            raise ValueError('unsupported rate pair')
        self.latency_input_frames = self.lib.audio_resampler_latency_input_frames(self.handle)

    def __del__(self):
        handle = getattr(self, 'handle', None)
        if handle:
            self.lib.audio_resampler_destroy(handle)
            self.handle = None

    def reset(self):
        self.lib.audio_resampler_reset(self.handle)

    def process(self, frames: np.ndarray) -> np.ndarray:
        ctypes = self._ctypes
        block = np.ascontiguousarray(frames, np.float32)
        bound = self.lib.audio_resampler_output_bound(self.handle, len(block))
        out = np.zeros(max(bound, 1), np.float32)
        consumed = ctypes.c_int(0)
        produced = ctypes.c_int(0)
        fp = ctypes.POINTER(ctypes.c_float)
        rc = self.lib.audio_resampler_process(
            self.handle, block.ctypes.data_as(fp), len(block),
            out.ctypes.data_as(fp), len(out), ctypes.byref(consumed), ctypes.byref(produced))
        if rc != 0 or consumed.value != len(block):
            raise RuntimeError('audio_resampler_process failed')
        return out[:produced.value].copy()


def round_trip(signal: np.ndarray, native_rate: int, via_rate: int = 48000
               ) -> Tuple[np.ndarray, int]:
    """native -> via -> native through two fresh resamplers.

    Returns the resampled signal and the round-trip delay in native samples
    (up latency + down latency, both in their own input frames, which for the
    supported pairs is 16 + 16 native samples)."""
    up = AudioResampler(native_rate, via_rate)
    down = AudioResampler(via_rate, native_rate)
    high = up.process(signal)
    back = down.process(high)
    delay = up.latency_input_frames + (down.latency_input_frames * native_rate) // via_rate
    return back, delay


class Dfn2RateBridge:
    """Twin of ``pipelines/dfn_rate_bridge.c``: run a 48 kHz :class:`Dfn2Stage`
    inside an 8 or 16 kHz pipeline.

    Per native hop: the two native spectra (E, P) are synthesised with the
    host's sqrt-Hann WOLA, upsampled through the C ``audio_resampler``
    (:class:`CResampler`, bit-exact with the product kernel), analysed on
    the 48 kHz 1024/512 grid every 512 new samples, pushed through the stage
    (silence for its two warm-up frames, as in C), synthesised at 48 kHz,
    downsampled, and emitted one native hop at a time from an output FIFO
    prefilled with the schedule's largest deficit.  ``added_delay`` is the
    native-sample delay this adds to the host's own output; the C header
    tabulates it (16 kHz/128: 672, 16 kHz/256: 629, 8 kHz/128: 330).
    """

    N48_FFT = 1024
    N48_HOP = 512

    def __init__(self, stage, sample_rate: int, fft_size: int = 0):
        fft = 256 if fft_size == 0 else int(fft_size)
        if sample_rate == 8000:
            if fft != 256:
                raise ValueError('8 kHz offers fft 256 only')
        elif sample_rate == 16000:
            if fft not in (256, 512):
                raise ValueError('16 kHz offers fft 256 or 512')
        else:
            raise ValueError('the rate bridge serves 8 and 16 kHz (48 kHz drives the stage directly)')
        self.stage = stage
        self.sample_rate = int(sample_rate)
        self.fft = fft
        self.hop = fft // 2
        self.n_freqs = fft // 2 + 1
        self.win_n = _root_hann(fft)
        self.win48 = _root_hann(self.N48_FFT)
        self.up_e = CResampler(sample_rate, 48000)
        self.up_p = CResampler(sample_rate, 48000)
        self.down = CResampler(48000, sample_rate)
        self.prefill = self._calibrate_prefill()
        lat_up = self.up_e.latency_input_frames
        lat_down = self.down.latency_input_frames
        self.added_delay = (self.prefill + lat_up + 3 * self.N48_HOP * sample_rate // 48000
                            + lat_down * sample_rate // 48000)
        self.reset()

    def _calibrate_prefill(self) -> int:
        """Count schedule on silence with the bridge's own resamplers (reset
        afterwards, as in C): the largest emitted-minus-produced deficit over
        eight hops, two periods of every supported schedule."""
        fill = produced = emitted = deficit = 0
        zeros_hop = np.zeros(self.hop, np.float32)
        zeros_frame = np.zeros(self.N48_HOP, np.float32)
        for _ in range(8):
            fill += len(self.up_e.process(zeros_hop))
            while fill >= self.N48_HOP:
                fill -= self.N48_HOP
                produced += len(self.down.process(zeros_frame))
            emitted += self.hop
            deficit = max(deficit, emitted - produced)
        self.up_e.reset()
        self.down.reset()
        return deficit

    def reset(self):
        self.stage.reset()
        for resampler in (self.up_e, self.up_p, self.down):
            resampler.reset()
        self.ola_e = np.zeros(self.fft, np.float32)
        self.ola_p = np.zeros(self.fft, np.float32)
        self.ola48 = np.zeros(self.N48_FFT, np.float32)
        # [512 of history | pending 48 kHz samples], as in C.
        self.fifo_e = np.zeros(self.N48_HOP, np.float32)
        self.fifo_p = np.zeros(self.N48_HOP, np.float32)
        self.out_fifo = np.zeros(self.prefill, np.float32)
        self.frames48 = 0
        self.last_hop_frames = 0

    def _synth_native(self, spec: np.ndarray, ola: np.ndarray) -> np.ndarray:
        full = np.fft.irfft(np.asarray(spec, np.complex64), n=self.fft).astype(np.float32)
        ola += full * self.win_n
        hop = ola[:self.hop].copy()
        ola[:-self.hop] = ola[self.hop:]
        ola[-self.hop:] = 0.0
        return hop

    def _analyze48(self, fifo: np.ndarray) -> np.ndarray:
        frame = (fifo[:self.N48_FFT] * self.win48).astype(np.float32)
        return np.fft.rfft(frame).astype(np.complex64)

    def push(self, estimate_spec: np.ndarray, apply_spec: np.ndarray) -> np.ndarray:
        e_hop = self._synth_native(estimate_spec, self.ola_e)
        p_hop = self._synth_native(apply_spec, self.ola_p)
        self.fifo_e = np.concatenate([self.fifo_e, self.up_e.process(e_hop)])
        self.fifo_p = np.concatenate([self.fifo_p, self.up_p.process(p_hop)])
        if len(self.fifo_e) != len(self.fifo_p):
            raise RuntimeError('the two lanes share one clock')
        self.last_hop_frames = 0
        while len(self.fifo_e) >= self.N48_FFT:
            e48 = self._analyze48(self.fifo_e)
            p48 = self._analyze48(self.fifo_p)
            # The consumed 512 become the next frame's history.
            self.fifo_e = self.fifo_e[self.N48_HOP:]
            self.fifo_p = self.fifo_p[self.N48_HOP:]
            result = self.stage.push(e48, p48)
            out48 = (np.asarray(result.spectrum, np.complex64) if result is not None
                     else np.zeros(self.N48_FFT // 2 + 1, np.complex64))
            full = np.fft.irfft(out48, n=self.N48_FFT).astype(np.float32)
            self.ola48 += full * self.win48
            chunk = self.ola48[:self.N48_HOP].copy()
            self.ola48[:-self.N48_HOP] = self.ola48[self.N48_HOP:]
            self.ola48[-self.N48_HOP:] = 0.0
            self.out_fifo = np.concatenate([self.out_fifo, self.down.process(chunk)])
            self.frames48 += 1
            self.last_hop_frames += 1
        if len(self.out_fifo) < self.hop:
            raise RuntimeError('output FIFO underrun: the prefill is wrong')
        out = self.out_fifo[:self.hop].copy()
        self.out_fifo = self.out_fifo[self.hop:]
        return out


def _root_hann(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    return np.sqrt(np.float32(0.5) - np.float32(0.5) * np.cos(
        np.float32(2.0 * np.pi) * idx / np.float32(n))).astype(np.float32)
