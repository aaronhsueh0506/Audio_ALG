"""How much a rendered echo path moves, measured back out of the audio.

The corpus's movement axis is calibrated against real captures, so the
calibration has to be checkable from the rendered pair alone -- not from the
weight schedule the renderer happens to have written down.  Everything here
takes only ``(far, echo)``, and it is also the estimator ``CALIBRATION_TARGETS``
below was measured with, so a render and a real capture are compared through
one implementation:

    1. compensate the bulk delay (GCC-PHAT over the whole clip);
    2. per 64 ms frame, take the frequency-domain Wiener path estimate
           H[b] = sum_t Y[t,b] X*[t,b] / sum_t |X[t,b]|^2
       over a 2 s window, restricted to 300 <= f < 4000 Hz, stepped by 1 s;
    3. report how that per-second path track changes.

The same band and the same 64 ms frame are where a room's positions are
compared with each other: ``position_gram`` / ``mixture_correlation`` answer
"how correlated would two mixtures of these positions measure", which is what
the generator solves its trajectory's mixture depth from. One band, one frame
and one statistic for both, so the number a corpus is tuned to is the number it
is measured by.

⚠ The correlation is ``|<H1,H2>| / (||H1|| ||H2||)`` -- invariant to a complex
scalar, so it measures the path's SHAPE and nothing else.  Level change is
reported separately as the least-squares complex scalar that maps one window's
path onto the next, in dB, because a real path's gain and its shape do not move
together and a single number that mixed them could not say which one a corpus
was missing.

⚠ A window whose far end is nearly silent has no path to estimate.  Those
windows are dropped rather than contributing a near-zero H that would read as a
large, fictional path change.

⚠ ONE bulk delay is assumed for the whole clip: the estimator aligns the pair
once and then treats every window as a shape/level comparison at that
alignment.  A clip carrying the ``delay_step`` impairment breaks that
assumption -- a mid-clip re-timing reads here as violent movement (measured:
0.81/0.57/0.21/0.15 for a 150 ms step on an otherwise frozen path) -- so such
sequences are outside this helper's scope and must be excluded from any
aggregate rather than explained.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    'ACTIVITY_GATE_DB',
    'ANALYSIS_FRAME_SEC',
    'BAND_HZ',
    'CALIBRATION_TARGETS',
    'CALIBRATION_TARGET_OF_MODE',
    'DEFAULT_LAGS_SEC',
    'estimate_path_track',
    'mixture_correlation',
    'path_drift_metrics',
    'position_gram',
]


# Lags the reference measurement reports, in seconds.
DEFAULT_LAGS_SEC = (1.0, 2.0, 4.0, 8.0)

# The band every statement about path SHAPE is made in, low inclusive, high
# exclusive.  One constant because the per-window estimate below and the
# per-room Gram further down have to describe the same thing: a generator that
# solved its mixture in one band and was measured in another would be tuned
# against a number nobody reads.
BAND_HZ = (300.0, 4000.0)

# The frame every statement about path SHAPE is made through, in seconds.  A
# path longer than this cannot be written as one frequency response, so what
# the per-window estimate below reads is the response of the first
# ANALYSIS_FRAME_SEC of the path plus the un-modelled tail acting as
# state-dependent noise.  The per-room Gram takes the same view of the same
# responses, for the same reason the band is shared.
ANALYSIS_FRAME_SEC = 0.064

# A window counts as carrying a reference when its in-band energy is within
# this many dB of the loudest window in the clip.  An absolute anchor, so a
# clip that is mostly silent cannot make its own silence the reference level.
ACTIVITY_GATE_DB = 25.0

# What 46 paired static and 46 paired movement far-end captures from the
# AEC-challenge blind set measure THROUGH THE HELPER BELOW, at its own
# defaults: bulk-delay search over 0..2 s, 64 ms frames, a hop that divides the
# sample rate into whole-second steps, 2 s windows stepped 1 s, band
# 300 <= f < 4000 Hz, 25 dB per-window activity gate, 1e-9 ridge, and the
# least-squares complex scalar as the level step. Every paired id is kept: no
# quality gate, because gating on measurable echo moves 4 s and 8 s by up to
# 0.12 and would calibrate the corpus against the clips that happen to have a
# strong linear path. `correlation` is keyed by lag in seconds; `gain_step_db`
# and `relative_change` are per-second medians.
# ⚠ Two decimals is all that is real: the bootstrap 95% interval of the 8 s
# movement median at n = 46 is [0.35, 0.79].
# ⚠ THIS DICT IS THE SINGLE SOURCE OF THESE NUMBERS. The configs, the
# generator and the README point at it instead of restating them, because a
# re-calibration that updates one prose copy and not the other five is
# invisible until someone re-derives the axis from a stale table.
# ⚠ 'static_device' is a real device nobody touched, NOT a synthetic frozen
# path: its correlation is well below 1.0 at 8 s. A generator whose ordinary
# sequences sit at 1.0 is the thing these numbers exist to detect.
CALIBRATION_TARGETS = {
    'movement': {
        'correlation': {1.0: 0.90, 2.0: 0.78, 4.0: 0.75, 8.0: 0.67},
        'relative_change': 0.29,
        'gain_step_db': 1.4,
    },
    'static_device': {
        'correlation': {1.0: 0.95, 2.0: 0.92, 4.0: 0.85, 8.0: 0.80},
        'relative_change': 0.15,
        'gain_step_db': 0.7,
    },
}

# Which measured population each rendered motion mode is calibrated against.
# The generator's 'slow_drift' mode exists to reproduce the path of a device
# NOBODY TOUCHED, so its target is the static-device curve; naming the target
# after the mode would suggest someone measured a slowly drifting device.
CALIBRATION_TARGET_OF_MODE = {
    'movement': 'movement',
    'slow_drift': 'static_device',
}


def _as_float_array(signal) -> np.ndarray:
    array = np.asarray(
        signal.detach().cpu().numpy() if hasattr(signal, 'detach') else signal,
        dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected one channel, got shape {array.shape}")
    return array


def _bulk_delay(far: np.ndarray, echo: np.ndarray, max_lag: int) -> int:
    """GCC-PHAT lag of ``echo`` behind ``far``, in samples.

    Phase transform rather than a plain cross-correlation: the broadband peak
    of a reverberant echo sits on the path's own spectral tilt and can land a
    frame away from the direct path, which then shows up as a fixed shape
    error in every window's estimate.

    ⚠ Non-negative lags only. An echo cannot precede the reference that caused
    it, and this corpus always delays the played signal by a positive bulk
    delay; a captured pair whose reference is stored BEHIND the microphone
    would need the search widened.
    """
    size = 1
    while size < far.size + echo.size:
        size *= 2
    cross = np.fft.rfft(echo, size) * np.conj(np.fft.rfft(far, size))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.irfft(cross, size)
    reach = max(1, min(max_lag, far.size - 1))
    return int(np.argmax(correlation[:reach + 1]))


def _stft(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if signal.size < frame:
        return np.zeros((0, frame // 2 + 1), dtype=np.complex128)
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop]
    return np.fft.rfft(frames * np.hanning(frame), axis=-1)


def _exact_hop(sr: int, frame: int) -> int:
    """The largest hop no longer than ``frame // 2`` that divides ``sr``.

    A lag reported as "8 s" has to BE 8 s.  With the natural ``frame // 2``
    hop, one second is 62.5 hops at 16 kHz, the window and step lengths round,
    and every reported lag comes out ~0.8% short -- small enough never to be
    noticed and exactly the kind of quiet offset that makes two measurements
    of the same corpus disagree.
    """
    target = max(1, frame // 2)
    for hop in range(target, 0, -1):
        if sr % hop == 0:
            return hop
    return target


def _band_mask(size: int, sr: int,
               band_hz: Tuple[float, float]) -> np.ndarray:
    """Which bins of a ``size``-point transform at ``sr`` fall inside the band."""
    frequencies = np.fft.rfftfreq(size, 1.0 / sr)
    return (frequencies >= band_hz[0]) & (frequencies < band_hz[1])


def position_gram(responses: Sequence, sr: int,
                  band_hz: Tuple[float, float] = BAND_HZ,
                  frame_sec: float = ANALYSIS_FRAME_SEC) -> np.ndarray:
    """Band-limited Gram matrix ``G_jk = <H_j, H_k>`` of a room's paths.

    ``responses`` are the impulse responses of the positions AS RENDERED --
    prepared, and carrying their relative levels, because a mixture weights
    the positions and a position twice as loud contributes four times the
    energy to the mixture's own response.

    ⚠ Each response is taken over the estimator's own analysis frame, which is
    what makes this the same statistic rather than a related one: the estimator
    reads a path through a ``frame_sec`` frame, so a longer response is not the
    ``H`` it reports.  Taking the whole response instead over-predicts the
    correlation the estimator will read by up to 0.08 on a 1 s pool, and the
    depth solved from it then under-drives the movement axis in exactly the
    reverberant rooms the corpus is aimed at.  What is left after this is the
    tail acting as state-dependent noise on the estimate, which pushes the
    reading the other way and which real rooms carry too.

    This is the ONE thing that decides how far a mixture of those positions
    can move: the correlation of the estimator above is a statement about the
    mixture's frequency response, and every such response is a fixed linear
    combination of these ``H_k``.  A pool whose positions share more of their
    energy has a correlation floor no weight schedule can reach below, and a
    pool whose tails differ decorrelates further per unit of weight change --
    which is why the trajectory's mixture depth is SOLVED from this matrix per
    room rather than configured as a constant (see
    ``aec_dataset.solve_mixture_depth``).

    ⚠ Hermitian, not symmetric, and ``w^T G v`` is complex for real mixtures.
    Only ``|w^T G v|`` and the real diagonal forms ``w^T G w`` are used, which
    is the same magnitude statistic ``path_drift_metrics`` reports.
    """
    if not len(responses):
        raise ValueError("a Gram matrix needs at least one response")
    size = max(1, int(round(frame_sec * sr)))
    spectra = np.asarray([
        np.fft.rfft(_as_float_array(response)[:size], size)
        for response in responses])
    band = _band_mask(size, sr, band_hz)
    if not band.any():
        raise ValueError(
            f"no analysis bin falls inside {band_hz[0]:g}..{band_hz[1]:g} Hz "
            f"at sr={sr} with a {size}-point transform")
    spectra = spectra[:, band]
    return spectra @ spectra.conj().T


def mixture_correlation(gram: np.ndarray, first: Sequence[float],
                        second: Sequence[float]) -> float:
    """Correlation of two mixture STATES of the same positions.

    ``|w^T G v| / sqrt((w^T G w)(v^T G v))`` -- the correlation
    ``path_drift_metrics`` would report between two windows in which the path
    held those two mixtures, computed from the positions alone.
    """
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    energy = float(abs(left @ gram @ left) * abs(right @ gram @ right))
    if energy <= 0.0:
        return 1.0
    return float(abs(left @ gram @ right) / math.sqrt(energy))


def estimate_path_track(far, echo, sr: int = 16000, *,
                        frame_sec: float = ANALYSIS_FRAME_SEC,
                        window_sec: float = 2.0,
                        step_sec: float = 1.0,
                        band_hz: Tuple[float, float] = BAND_HZ,
                        max_delay_sec: float = 2.0,
                        ridge: float = 1e-9
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-window echo-path estimate.

    Returns ``(H, valid, step_sec)``: ``H`` is (n_windows, n_bins) complex over
    the requested band, ``valid`` marks the windows whose far end carried
    enough energy to define a path, and ``step_sec`` is the ACTUAL spacing
    between windows -- exact whole seconds whenever the hop divides the sample
    rate, which ``_exact_hop`` arranges for every rate this corpus uses.

    ⚠ ``ridge`` is an ABSOLUTE floor on the Wiener denominator, matching the
    reference measurement.  A floor proportional to the window's own mean band
    power looks safer and is not: it shrinks each window's weak bins by an
    amount that depends on that window's spectrum, which is a shape change the
    correlation then reports as path movement.

    ⚠ ``max_delay_sec`` has to cover the WHOLE delay the pair can carry, with
    headroom.  Outside its true delay the PHAT peak is noise, and locking onto
    noise does not fail loudly: the path track becomes garbage that explains
    ~0.05 dB of the capture instead of ~10 dB, and its correlation collapses
    toward 0 -- which reads as violent movement and drags any population median
    down with it.  The default covers a captured delay of up to 2 s, i.e. every
    delay this corpus can render (bulk delay plus the long-delay tail, the
    jitter and a delay step) and every capture the targets above were measured
    on, whose largest genuine delay is 0.93 s.
    """
    far = _as_float_array(far)
    echo = _as_float_array(echo)
    length = min(far.size, echo.size)
    far, echo = far[:length], echo[:length]

    delay = _bulk_delay(far, echo, int(sr * max_delay_sec))
    if delay:
        # Both signals are TRIMMED, not zero-padded: pairing an active
        # reference with a zeroed echo tail would make the last window's path
        # collapse toward zero and read as a level step.
        echo = echo[delay:]
        far = far[:echo.size]

    frame = int(round(frame_sec * sr))
    hop = _exact_hop(sr, frame)
    reference = _stft(far, frame, hop)
    observed = _stft(echo, frame, hop)
    n_frames = min(reference.shape[0], observed.shape[0])

    band = _band_mask(frame, sr, band_hz)
    reference = reference[:n_frames, band]
    observed = observed[:n_frames, band]

    window_frames = max(2, int(round(window_sec * sr / hop)))
    step_frames = max(1, int(round(step_sec * sr / hop)))
    starts = range(0, max(0, n_frames - window_frames + 1), step_frames)

    tracks = []
    energies = []
    for start in starts:
        span = slice(start, start + window_frames)
        denominator = (np.abs(reference[span]) ** 2).sum(axis=0)
        energies.append(denominator.sum())
        tracks.append((observed[span] * np.conj(reference[span])).sum(axis=0)
                      / (denominator + ridge))

    if not tracks:
        return (np.zeros((0, int(band.sum())), dtype=np.complex128),
                np.zeros(0, dtype=bool), step_frames * hop / sr)

    energy = np.asarray(energies)
    loudest = float(energy.max()) if energy.size else 0.0
    valid = energy >= loudest * 10.0 ** (-ACTIVITY_GATE_DB / 10.0)
    return np.asarray(tracks), valid, step_frames * hop / sr


def _inner(paths: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Row-wise band inner product ``<H_i, G_i> = sum_b conj(H) G``."""
    return (np.conj(paths) * others).sum(axis=1)


def path_drift_metrics(far, echo, sr: int = 16000, *,
                       lags_sec: Sequence[float] = DEFAULT_LAGS_SEC,
                       **track_kwargs) -> Dict[str, object]:
    """Correlation vs lag, per-second relative change, per-second level step.

    ``correlation`` is keyed by the requested lag in seconds; a lag longer than
    the clip -- or one whose window pairs are all far-silent -- reports
    ``None`` rather than a number computed from nothing.

    ⚠ ``gain_step_db`` is the least-squares complex scalar ``a`` that maps one
    window's path onto the next, ``a = <H1, H2> / <H1, H1>``, in dB.  A ratio
    of norms would be a different statistic: two paths of equal norm and
    different shape have a norm ratio of exactly 1 while ``|a|`` falls with
    their correlation, so a corpus tuned to close the gap on the norm ratio
    would over-drive its level walk by whatever the shape change contributes.
    """
    track, valid, step_sec = estimate_path_track(far, echo, sr, **track_kwargs)
    norms = np.linalg.norm(track, axis=1) if track.size else np.zeros(0)
    usable = valid & (norms > 0)

    def _pairs(lag_windows: int):
        first = np.arange(0, max(0, len(track) - lag_windows))
        second = first + lag_windows
        keep = usable[first] & usable[second] if first.size else first
        return first[keep], second[keep]

    correlation: Dict[float, Optional[float]] = {}
    for lag in lags_sec:
        lag_windows = max(1, int(round(lag / step_sec)))
        first, second = _pairs(lag_windows)
        if first.size == 0:
            correlation[float(lag)] = None
            continue
        inner = np.abs(_inner(track[first], track[second]))
        correlation[float(lag)] = float(
            np.median(inner / (norms[first] * norms[second])))

    first, second = _pairs(max(1, int(round(1.0 / step_sec))))
    if first.size:
        change = float(np.median(
            (np.abs(track[second] - track[first]) ** 2).sum(axis=1)
            / (norms[first] ** 2)))
        projection = _inner(track[first], track[second]) / (norms[first] ** 2)
        gain_step = float(np.median(np.abs(
            20.0 * np.log10(np.maximum(np.abs(projection), 1e-12)))))
    else:
        change = gain_step = math.nan

    return {
        'correlation': correlation,
        'relative_change': change,
        'gain_step_db': gain_step,
        'step_sec': step_sec,
    }
