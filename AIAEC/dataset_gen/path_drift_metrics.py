"""How much a rendered echo path moves, measured back out of the audio.

The corpus's movement axis is calibrated against real captures, so the
calibration has to be checkable from the rendered pair alone, not from the
weight schedule the renderer wrote down. Everything here takes only
``(far, echo)`` and is the estimator ``CALIBRATION_TARGETS`` was measured with:

    1. compensate the bulk delay (GCC-PHAT over the whole clip);
    2. per 64 ms frame, take the frequency-domain Wiener path estimate
           H[b] = sum_t Y[t,b] X*[t,b] / sum_t |X[t,b]|^2
       over a 2 s window, restricted to 300 <= f < 4000 Hz, stepped by 1 s;
    3. report how that per-second path track changes.

``position_gram`` / ``mixture_correlation`` use the same band, frame and
statistic to compare a room's positions, so the number a corpus is tuned to is
the number it is measured by.

⚠ The correlation ``|<H1,H2>| / (||H1|| ||H2||)`` is invariant to a complex
scalar, so it measures path SHAPE only; level change is reported separately as
the least-squares complex scalar mapping one window's path onto the next (dB),
because a real path's gain and shape do not move together.

⚠ Windows with a nearly silent far end are dropped (no path to estimate).
⚠ ONE bulk delay is assumed for the whole clip, so a clip carrying the
``delay_step`` impairment reads as violent movement and must be excluded from
any aggregate.
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
# AEC-challenge blind set measure THROUGH THE HELPER BELOW at its own defaults
# (bulk-delay search 0..2 s, 64 ms frames, 2 s windows stepped 1 s, 300 <= f <
# 4000 Hz, 25 dB activity gate, 1e-9 ridge); every paired id is kept, with no
# quality gate. `correlation` is keyed by lag in seconds; `gain_step_db` and
# `relative_change` are per-second medians. ⚠ Two decimals is all that is real.
# ⚠ THIS DICT IS THE SINGLE SOURCE of these numbers; others point at it. ⚠
# 'static_device' is a real untouched device, NOT a synthetic frozen path.
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

    Phase transform rather than plain cross-correlation: the broadband peak of
    a reverberant echo sits on the path's spectral tilt and can land a frame
    from the direct path. ⚠ Non-negative lags only: an echo cannot precede its
    reference and this corpus always delays the played signal.
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

    ``responses`` are the impulse responses AS RENDERED, with their relative
    levels. ⚠ Each is taken over the estimator's own ``frame_sec`` frame; the
    whole response would over-predict the correlation it reads. The mixture
    depth is SOLVED from this Hermitian matrix per room
    (``aec_dataset.solve_mixture_depth``).
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
    """Per-window echo-path estimate ``(H, valid, step_sec)``.

    ``H`` is (n_windows, n_bins) complex over the band, ``valid`` marks windows
    whose far end defines a path, ``step_sec`` is the ACTUAL window spacing. ⚠
    ``ridge`` is an ABSOLUTE floor on the Wiener denominator (a
    band-power-proportional one would be read as movement). ⚠ ``max_delay_sec``
    must cover the WHOLE delay the pair can carry, else the PHAT peak is noise.
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

    ``correlation`` is keyed by the requested lag in seconds; ``None`` when the
    clip is shorter or all window pairs are far-silent. ⚠ ``gain_step_db`` is
    the least-squares complex scalar ``a = <H1, H2> / <H1, H1>`` in dB, not a
    norm ratio (equal norms give 1 whatever the shape, while ``|a|`` falls with
    correlation).
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
