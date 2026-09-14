"""AEC3 comfort-noise generator, ported bit-for-bit from the C pipelines.

Both C pipelines (``mono_aec_nr_res/audio_pipeline.c`` ``cng_lut_index`` and
``4ch_aec_bf_nr_res/4aec_nr_res.c``) fill the residual-suppressed bins with
comfort noise drawn from one multiplicative-congruential step per bin whose top
five bits index ``AEC3B_SQRT2_SIN_LUT`` (``lib/aec/c_impl/include/
aec3_balanced_config.h``), the imaginary part reading a quarter turn (+8 of 32)
ahead of the real one.  This module reproduces that generator exactly so a
Python reference can be compared with the C output with comfort noise ON.

The generator is advanced only when a caller adds noise to a frame; a stage
that emits nothing on a hop must not draw, so the draw count stays paired with
the emitted-frame count on both sides.
"""

from __future__ import annotations

import numpy as np

#: ``AUDIO_PIPELINE_RNG_SEED`` in both C pipelines.
SEED = 0x9E3779B9

#: The 32 float32 literals of ``AEC3B_SQRT2_SIN_LUT``: sqrt(2) * sin(2*pi*i/32).
#: Copied as literals (index 16 is 1.73191212e-16, not 0) so the table is the
#: same bits the C reads.
LUT = np.array(
    [0.0, 0.27589938, 0.541196108, 0.785694957, 1.0, 1.17587554,
     1.30656302, 1.3870399, 1.41421354, 1.3870399, 1.30656302, 1.17587554,
     1.0, 0.785694957, 0.541196108, 0.27589938, 1.73191212e-16,
     -0.27589938, -0.541196108, -0.785694957, -1.0, -1.17587554,
     -1.30656302, -1.3870399, -1.41421354, -1.3870399, -1.30656302,
     -1.17587554, -1.0, -0.785694957, -0.541196108, -0.27589938],
    dtype=np.float32,
)

_MULT = 69069
_INC = 1
_MASK = 0x7FFFFFFF


def lcg_step(state: int) -> int:
    """One ``rng = (rng*69069+1) & 0x7FFFFFFF`` step."""
    return (state * _MULT + _INC) & _MASK


class Aec3ComfortNoise:
    """Stateful twin of the C generator.

    ``draw(count)`` returns ``count`` LUT indices (0..31) advancing the state
    exactly ``count`` times, in one vectorised affine map (the LCG's n-step
    composition is affine mod 2^31), so a frame's 511 draws cost one numpy
    expression rather than a Python loop.
    """

    def __init__(self, seed: int = SEED):
        self.seed = int(seed)
        self.state = int(seed)
        self._powers: dict = {}

    def reset(self) -> None:
        self.state = self.seed

    def _affine(self, count: int):
        table = self._powers.get(count)
        if table is None:
            mult = np.empty(count, dtype=np.uint64)
            offset = np.empty(count, dtype=np.uint64)
            a_pow = 1
            c_sum = 0
            for i in range(count):
                # x_{i+1} = a*x_i + c  ->  x_n = a^n x_0 + c(1 + a + ... + a^{n-1})
                c_sum = (c_sum * _MULT + _INC) & _MASK
                a_pow = (a_pow * _MULT) & _MASK
                mult[i] = a_pow
                offset[i] = c_sum
            table = (mult, offset)
            self._powers[count] = table
        return table

    def draw(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.uint32)
        mult, offset = self._affine(count)
        states = (np.uint64(self.state) * mult + offset) & np.uint64(_MASK)
        self.state = int(states[-1])
        return (states >> np.uint64(26)).astype(np.uint32)

    def add(self, spec: np.ndarray, amplitude: np.ndarray) -> None:
        """Add comfort noise to ``spec`` (complex64, in place) on bins 1..n-2.

        ``amplitude[k]`` is the per-bin amplitude already folded from the CNG
        PSD, the residual gain and the NR factor, exactly as the C computes
        ``a`` before the LUT lookup.  Bins 0 and n-1 are never touched.
        """
        n = spec.shape[0]
        ix = self.draw(n - 2)
        re = LUT[ix]
        im = LUT[(ix + np.uint32(8)) & np.uint32(31)]
        a = amplitude[1:n - 1].astype(np.float32, copy=False)
        spec[1:n - 1] += (a * re).astype(np.float32) + 1j * (a * im).astype(np.float32)
