"""Shared-delay, four-linear-filter AEC front end for microphone arrays.

This Python reference deliberately does not implement SRP-PHAT or GSC.  Its
production API is split at that ownership boundary::

    pre = pipeline.process_pre_beamformer(mics, render)
    bf = external_srp_gsc(pre.linear_channels)
    post = pipeline.process_post_beamformer(pre, bf)

The optional ``Beamformer`` object and ``EqualWeightBeamformer`` exist only for
integration tests and offline fixtures.  Requiring the external stage to return
its complex per-bin effective weights lets the AEC error/echo spectra cross the
boundary coherently before the existing mono NR + RES stages, without a second
beamformer or a fifth AEC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, Sequence

import numpy as np

from lib.aec.python.modules.config import AecConfig
from lib.aec.python.modules.dataclasses import AecResContext
from lib.aec.python.modules.delay.legacy_compat import LegacyDelayShim
from lib.aec.python.modules.enums import AecMode
from lib.aec.python.modules.orchestrator import AEC
from lib.aec.python.modules.residual.suppression_gain import (
    EchoAudibilityConfig,
    SuppressionGain,
    SuppressorConfig,
)
from lib.aec.python.modules.aec3_scale import fft_density_scale


_SUPPORTED_SAMPLE_RATES = (16000, 48000)
_N_CHANNELS = 4


# ---------------------------------------------------------------------------
# Shared delay and beamformer handoff types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharedDelayState:
    """Observable state of the one shared matched-filter instance."""

    delay_samples: int
    confidence: float
    solid: bool
    changed: bool
    estimator_calls: int
    estimator_updates: int


class SharedMatchedDelayEstimator:
    """One matched filter driven by one configurable array capture proxy.

    Anti-alias status (fixed 2026-08-03): this wrapper used to do its OWN
    external 48 kHz decimation -- a bare stride-pick with NO anti-alias
    filter, real aliasing at 48 kHz -- ahead of an inner ``LegacyDelayShim``
    that it hardcoded to ``sample_rate=16000`` regardless of this wrapper's
    real rate, then manually rescaled the returned delay by
    ``_rate_factor``. That duplicated (and, at 48 kHz, aliased) a decimation
    path the estimator now owns correctly, and diverged from the same class
    of fix already applied to the mono AEC's C port
    (``delay_aec3.c``'s ``DaResample48`` sidechain, 2026-08-02) and to
    Python's own ``EchoPathDelayEstimator`` (its ``_Resample48`` sidechain,
    2026-08-03).

    This wrapper now constructs its inner ``LegacyDelayShim`` with this
    wrapper's TRUE native ``sample_rate`` (8000/16000/48000) and feeds it
    every hop RAW and un-decimated. At 48 kHz, ``EchoPathDelayEstimator``
    internally anti-alias-filters (order-7 elliptic, 4 SOS sections) and
    decimates by 3 ahead of its inner AEC3-rate matched-filter chain, and it
    already rescales the delay it returns back to the true native sample
    domain (see its ``_process_inner_block``'s
    ``* _DOWN_SAMPLING_FACTOR * self._rate_factor`` line) -- so this wrapper
    must NOT decimate or rescale a second time. No external decimation, no
    decimation-phase bookkeeping, and no manual rate-factor rescale live
    here anymore.
    """

    def __init__(self, sample_rate: int, hop_size: int) -> None:
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"shared delay supports {_SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        # LegacyDelayShim -> EchoPathDelayEstimator now owns 48kHz anti-alias
        # + decimation internally (and rescales its returned delay back to
        # native samples), so it is constructed with the TRUE native rate --
        # never a hardcoded 16000 -- and fed raw hops directly (see
        # accumulate() below).
        self._estimator = LegacyDelayShim(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
        )
        self._calls = 0
        self._accepted_delay = 0

    @property
    def instance_count(self) -> int:
        """Structural audit hook: this wrapper owns exactly one estimator."""
        return 1

    def reset(self) -> None:
        self._estimator.reset()
        self._calls = 0
        self._accepted_delay = 0

    def accumulate(self, capture_proxy: np.ndarray, render: np.ndarray) -> SharedDelayState:
        capture_proxy = np.asarray(capture_proxy, dtype=np.float32)
        render = np.asarray(render, dtype=np.float32)
        if (
            capture_proxy.ndim != 1
            or render.ndim != 1
            or capture_proxy.shape != render.shape
        ):
            raise ValueError("capture proxy and render must be equal-length 1-D hops")
        self._calls += 1
        # Raw, un-decimated native-rate hop straight into the estimator --
        # it anti-alias filters + decimates internally at 48kHz and returns
        # the delay already rescaled back to native samples.
        self._estimator.accumulate(capture_proxy, render)

        estimated = int(self._estimator.estimated_delay)
        # Match the production acquisition rule: a delay is not allowed to
        # disturb four learned filters until the shared estimate is solid and
        # has produced at least three aggregate updates.
        eligible = (
            estimated >= 0
            and self._estimator.is_solid
            and self._estimator._n_updates >= 3
        )
        changed = bool(eligible and estimated != self._accepted_delay)
        if eligible:
            self._accepted_delay = estimated
        return SharedDelayState(
            delay_samples=int(self._accepted_delay),
            confidence=float(self._estimator.confidence),
            solid=bool(self._estimator.is_solid),
            changed=changed,
            estimator_calls=self._calls,
            estimator_updates=int(self._estimator._n_updates),
        )


class _SharedReferenceDelayLine:
    """Streaming raw-reference delay line shared by all four AEC lanes."""

    def __init__(self, max_delay_samples: int, max_hop_size: int) -> None:
        if max_delay_samples < 0 or max_hop_size <= 0:
            raise ValueError("invalid delay-line dimensions")
        self._capacity = int(max_delay_samples + 2 * max_hop_size + 1)
        self._ring = np.zeros(self._capacity, dtype=np.float32)
        self._samples_seen = 0

    def reset(self) -> None:
        self._ring.fill(0.0)
        self._samples_seen = 0

    def process(self, render: np.ndarray, delay_samples: int) -> np.ndarray:
        render = np.asarray(render, dtype=np.float32)
        if render.ndim != 1:
            raise ValueError("render must be a 1-D hop")
        if delay_samples < 0 or delay_samples >= self._capacity - render.size:
            raise ValueError("delay is outside the shared delay-line capacity")

        start = self._samples_seen
        absolute = start + np.arange(render.size, dtype=np.int64)
        self._ring[absolute % self._capacity] = render
        source = absolute - int(delay_samples)
        out = np.zeros(render.size, dtype=np.float32)
        valid = source >= 0
        out[valid] = self._ring[source[valid] % self._capacity]
        self._samples_seen += render.size
        return out


@dataclass(frozen=True)
class BeamformerFrame:
    """One external-beamformer result.

    ``weights[ch, bin]`` uses the convention
    ``output_spec[bin] = sum(weights[ch, bin] * input_spec[ch, bin])``.
    """

    samples: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PreBeamformerFrame:
    """Owned handoff from four linear AEC lanes to external SRP-PHAT/GSC.

    ``linear_channels`` is ``[hop, 4]``.  ``contexts`` and ``aligned_render``
    are snapshots: the caller may queue frames while the four adaptive filters
    continue processing, provided external results return to the post stage in
    frame order.  ``generation`` invalidates queued frames after ``reset()``.
    """

    frame_index: int
    generation: int
    _owner_token: object = field(repr=False, compare=False)
    linear_channels: np.ndarray
    contexts: tuple[AecResContext, ...]
    aligned_render: np.ndarray
    delay: SharedDelayState

    @property
    def linear_hops(self) -> np.ndarray:
        """Channel-first ``[4, hop]`` view used by the test adapter contract."""
        return self.linear_channels.T


class Beamformer(Protocol):
    def process(
        self, linear_hops: np.ndarray, contexts: Sequence[AecResContext]
    ) -> BeamformerFrame:
        ...


class EqualWeightBeamformer:
    """Deterministic test adapter; replace with the external beamformer."""

    def process(
        self, linear_hops: np.ndarray, contexts: Sequence[AecResContext]
    ) -> BeamformerFrame:
        linear_hops = np.asarray(linear_hops, dtype=np.float32)
        if linear_hops.ndim != 2 or linear_hops.shape[0] != _N_CHANNELS:
            raise ValueError("linear_hops must have shape [4, hop]")
        if len(contexts) != _N_CHANNELS:
            raise ValueError("exactly four AEC contexts are required")
        n_freqs = int(np.asarray(contexts[0].error_spec).size)
        weights = np.full((_N_CHANNELS, n_freqs), 0.25, dtype=np.complex64)
        return BeamformerFrame(
            samples=np.mean(linear_hops, axis=0, dtype=np.float32),
            weights=weights,
        )


@dataclass(frozen=True)
class FourChannelAecConfig:
    sample_rate: int = 16000
    frame_size: Optional[int] = None
    hop_size: Optional[int] = None
    filter_length: Optional[int] = None
    capture_proxy_channel: int = 0
    max_delay_ms: float = 1024.0
    aec_mode: AecMode = AecMode.PBFDKF

    def resolved_grid(self) -> tuple[int, int]:
        if self.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"four-channel path supports {_SUPPORTED_SAMPLE_RATES}, got {self.sample_rate}"
            )
        # 16 kHz default is 256/128 (8ms hop), matching the C core's own
        # derive_dims_and_configs default (4aec_nr_res.c) as of 2026-08-02/03.
        # 512/256 remains a supported, explicit alternate (see whitelist below).
        default_frame = 256 if self.sample_rate == 16000 else 1024
        frame = default_frame if self.frame_size is None else int(self.frame_size)
        hop = frame // 2 if self.hop_size is None else int(self.hop_size)
        # Mirrors the C core's exact whitelist (4aec_nr_res.c derive_dims_and_configs):
        # 16 kHz supports frame/FFT in {256, 512}; 48 kHz supports only 1024. A
        # frame value the C side would reject must be rejected here too, so an
        # explicit override can't silently diverge between the Python reference
        # and the C acceptance gate.
        if self.sample_rate == 16000:
            if frame not in (256, 512):
                raise ValueError(
                    "16 kHz four-channel grid supports frame/FFT in "
                    f"(256, 512), got {frame}"
                )
        elif frame != 1024:
            raise ValueError(
                f"48 kHz four-channel grid supports only frame/FFT 1024, got {frame}"
            )
        if frame != 2 * hop:
            raise ValueError("frame_size must equal 2 * hop_size")
        if not 0 <= self.capture_proxy_channel < _N_CHANNELS:
            raise ValueError("capture_proxy_channel must be in [0, 3]")
        return frame, hop


@dataclass(frozen=True)
class FourChannelFrame:
    frame_index: int
    linear_channels: np.ndarray
    beamformed: np.ndarray
    context: AecResContext
    delay: SharedDelayState


# ---------------------------------------------------------------------------
# Context projection
# ---------------------------------------------------------------------------

def _snapshot_context(context: AecResContext) -> AecResContext:
    """Detach a context from mutable per-lane AEC work buffers."""

    def copied(value):
        return None if value is None else np.asarray(value).copy()

    return replace(
        context,
        raw_output=copied(context.raw_output),
        echo_spec=copied(context.echo_spec),
        far_spec=copied(context.far_spec),
        near_spec=copied(context.near_spec),
        error_spec=copied(context.error_spec),
        res_gain=copied(context.res_gain),
        comfort_noise=copied(context.comfort_noise),
        r2=copied(context.r2),
    )


def _require_context_arrays(contexts: Sequence[AecResContext]) -> int:
    if len(contexts) != _N_CHANNELS:
        raise ValueError("exactly four contexts are required")
    n_freqs = -1
    for context in contexts:
        for name in ("error_spec", "res_gain", "r2", "comfort_noise"):
            value = getattr(context, name)
            if value is None:
                raise ValueError(f"AEC context is missing {name}")
            size = int(np.asarray(value).size)
            if n_freqs < 0:
                n_freqs = size
            elif size != n_freqs:
                raise ValueError("AEC context frequency dimensions do not match")
    return n_freqs


def _fuse_contexts(
    contexts: Sequence[AecResContext], beamformed: BeamformerFrame
) -> AecResContext:
    """Cross the beamformer boundary without creating a fifth AEC.

    Error/near/echo spectra are exactly coherent for the supplied weights.
    R2 is coherently combined using each channel's echo-estimate phase.
    This function fuses measurements only. A single stateful post-beam RES
    computes ``res_gain`` afterward; per-lane gains are intentionally ignored.
    """

    n_freqs = _require_context_arrays(contexts)
    samples = np.asarray(beamformed.samples, dtype=np.float32)
    weights = np.asarray(beamformed.weights, dtype=np.complex64)
    if samples.ndim != 1:
        raise ValueError("beamformer samples must be 1-D")
    if weights.shape != (_N_CHANNELS, n_freqs):
        raise ValueError(
            f"beamformer weights must have shape (4, {n_freqs}), got {weights.shape}"
        )
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)):
        raise ValueError("beamformer returned non-finite values")

    def coherent(name: str) -> np.ndarray:
        values = np.stack(
            [np.asarray(getattr(c, name), dtype=np.complex64) for c in contexts]
        )
        return np.sum(weights * values, axis=0, dtype=np.complex64)

    error_spec = coherent("error_spec")
    echo_spec = coherent("echo_spec")
    near_spec = coherent("near_spec")

    # X is one shared digital render reference, not four spatial microphone
    # observations. Applying the beamformer weights to it would be physically
    # wrong: a valid zero-sum spatial weight vector would erase an active X.
    far_specs = np.stack(
        [np.asarray(c.far_spec, dtype=np.complex64) for c in contexts]
    )
    if not np.allclose(far_specs, far_specs[0:1], rtol=1e-6, atol=1e-7):
        raise ValueError("linear AEC lanes do not share one far-end spectrum")
    far_spec = far_specs[0]

    r2 = np.stack([np.asarray(c.r2, dtype=np.float32) for c in contexts])
    echo_phase = np.stack(
        [np.angle(np.asarray(c.echo_spec, dtype=np.complex64)) for c in contexts]
    )
    residual_phasor = np.sqrt(np.maximum(r2, 0.0)) * np.exp(1j * echo_phase)
    fused_r2 = np.abs(np.sum(weights * residual_phasor, axis=0)) ** 2

    comfort = np.stack(
        [np.asarray(c.comfort_noise, dtype=np.float32) for c in contexts]
    )
    fused_comfort = np.sum(np.abs(weights) ** 2 * comfort, axis=0)

    channel_weight = np.mean(np.abs(weights), axis=1)
    weight_sum = float(np.sum(channel_weight))
    if weight_sum <= 1e-12:
        raise ValueError("beamformer has zero weight on all channels")
    norm_weight = channel_weight / weight_sum
    far_powers = np.asarray([c.far_power for c in contexts], dtype=np.float64)
    if not np.allclose(far_powers, far_powers[0], rtol=1e-6, atol=1e-9):
        raise ValueError("linear AEC lanes do not share one far-end power")
    shared_far_power = float(far_powers[0])

    return AecResContext(
        raw_output=samples.copy(),
        echo_spec=echo_spec.astype(np.complex64),
        far_power=shared_far_power,
        far_spec=far_spec.astype(np.complex64),
        near_spec=near_spec.astype(np.complex64),
        filter_converged=all(c.filter_converged for c in contexts),
        erle_factor=float(min(c.erle_factor for c in contexts)),
        dt_indicator=float(max(c.dt_indicator for c in contexts)),
        divergence=float(max(c.divergence for c in contexts)),
        over_sub=float(max(c.over_sub for c in contexts)),
        saturation_level=float(max(c.saturation_level for c in contexts)),
        erl_estimate=float(
            sum(float(w) * float(c.erl_estimate) for w, c in zip(norm_weight, contexts))
        ),
        error_spec=error_spec.astype(np.complex64),
        # Filled by PostBeamResidualSuppressor. Never select/min a lane gain.
        res_gain=np.ones(n_freqs, dtype=np.float32),
        comfort_noise=np.maximum(fused_comfort, 0.0).astype(np.float32),
        r2=np.maximum(fused_r2, 0.0).astype(np.float32),
    )


class _PostBeamAecState:
    """Small SuppressionGain-facing view of the fused AEC state."""

    def __init__(self) -> None:
        self.saturated = False

    def saturated_echo(self) -> bool:
        return self.saturated


class PostBeamResidualSuppressor:
    """Exactly one stateful RES gain calculator after spatial filtering.

    The four adaptive filters still own the residual-echo estimators needed to
    produce R2. Those power estimates are spatially combined first; this
    object then owns the one temporal SuppressionGain state consumed by the
    mono NR -> RES output path.

    AecResContext currently does not export unbounded R2 or the full
    stationarity/AecState surface. Until that seam is extended, bounded R2 is
    used for both gain inputs and no stationary mask is supplied. This is an
    explicit parity limitation, not a reason to reuse one microphone's gain.
    """

    def __init__(self, sample_rate: int, hop_size: int, n_freqs: int) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.n_freqs = int(n_freqs)
        self.instance_count = 1
        self._state = _PostBeamAecState()
        self._initial_state = True
        self._build_gain()

    def _build_gain(self) -> None:
        config = AecConfig(
            sample_rate=self.sample_rate,
            frame_size=2 * self.hop_size,
            hop_size=self.hop_size,
        )
        default_audibility = EchoAudibilityConfig()
        suppressor_config = SuppressorConfig()
        suppressor_config.echo_audibility = replace(
            suppressor_config.echo_audibility,
            use_stationarity_properties=True,
            floor_power=fft_density_scale(
                default_audibility.floor_power, 2 * self.hop_size
            ),
            low_render_limit=fft_density_scale(
                default_audibility.low_render_limit, 2 * self.hop_size
            ),
            normal_render_limit=fft_density_scale(
                default_audibility.normal_render_limit, 2 * self.hop_size
            ),
        )
        suppressor_config.dominant_nearend_detection = replace(
            suppressor_config.dominant_nearend_detection,
            use_wallclock_trigger_threshold=True,
        )
        self._gain = SuppressionGain(
            n_bins=self.n_freqs,
            config=suppressor_config,
            sr=self.sample_rate,
            hop_size=self.hop_size,
            use_wallclock_block_energy_threshold=True,
            use_wallclock_gain_ratchet=True,
            soft_nearend_blend_enabled=config.soft_nearend_blend_enabled,
            soft_nearend_blend_enr_threshold=config.soft_nearend_blend_enr_threshold,
            soft_nearend_blend_softness=config.soft_nearend_blend_softness,
            soft_nearend_blend_per_bin=config.soft_nearend_blend_per_bin,
            split_floor_enabled=config.min_gain_split_floor_enabled,
            split_floor_far_active_db=config.min_gain_floor_far_active_db,
            split_floor_far_silent_db=config.min_gain_floor_far_silent_db,
            split_floor_dt_db=config.min_gain_floor_dt_db,
            split_floor_latch_power=config.min_gain_far_latch_power,
        )

    def reset(self) -> None:
        self._state = _PostBeamAecState()
        self._initial_state = True
        self._build_gain()

    def process(self, context: AecResContext, render: np.ndarray) -> AecResContext:
        error = np.asarray(context.error_spec, dtype=np.complex64)
        near = np.asarray(context.near_spec, dtype=np.complex64)
        r2 = np.maximum(np.asarray(context.r2, dtype=np.float32), 0.0)
        comfort = np.maximum(
            np.asarray(context.comfort_noise, dtype=np.float32), 0.0
        )
        if not (
            error.size == near.size == r2.size == comfort.size == self.n_freqs
        ):
            raise ValueError("post-beam RES context has inconsistent dimensions")

        scale = np.float32(32768.0 * 32768.0)
        error_power = (np.abs(error) ** 2 * scale).astype(np.float32)
        near_power = (np.abs(near) ** 2 * scale).astype(np.float32)
        suppressor_input = (
            np.minimum(error_power, near_power)
            if context.filter_converged
            else near_power
        )
        if self._initial_state and context.filter_converged:
            self._gain.set_initial_state(False)
            self._initial_state = False
        self._state.saturated = bool(context.saturation_level > 0.5)
        self._gain._dt_protect_active = bool(context.dt_indicator > 0.2)
        gain = self._gain.get_gain(
            aec_state=self._state,
            nearend_spectrum=suppressor_input,
            residual_echo_spectrum=r2,
            # Not exported by AecResContext yet; bounded R2 is the explicit
            # conservative placeholder for the DNE-only unbounded input.
            residual_echo_spectrum_unbounded=r2,
            comfort_noise_spectrum=comfort,
            render_block=np.asarray(render, dtype=np.float32) * 32768.0,
            clock_drift=False,
            stationary_mask=None,
        )
        return replace(context, res_gain=np.asarray(gain, dtype=np.float32).copy())


# ---------------------------------------------------------------------------
# Four-channel pipeline
# ---------------------------------------------------------------------------

class FourChannelAecPipeline:
    """Pre/post shell around an externally owned SRP-PHAT/GSC stage."""

    def __init__(
        self,
        config: Optional[FourChannelAecConfig] = None,
        beamformer: Optional[Beamformer] = None,
    ) -> None:
        self.config = config or FourChannelAecConfig()
        frame, hop = self.config.resolved_grid()
        self.frame_size = frame
        self.hop_size = hop
        self.n_freqs = frame // 2 + 1
        filter_length = (
            self.config.filter_length
            if self.config.filter_length is not None
            else (
                self.config.sample_rate * (64 if self.config.sample_rate == 48000 else 52)
                // 1000
            )
        )
        lane_config = AecConfig(
            sample_rate=self.config.sample_rate,
            frame_size=frame,
            hop_size=hop,
            filter_length=int(filter_length),
            mode=self.config.aec_mode,
            enable_delay_est=False,
            fixed_delay_samples=-1,
            enable_res=False,
            return_res_context=True,
        )
        # Four independent adaptive-filter states.  ``replace`` prevents one
        # AEC instance from mutating config state observed by another lane.
        self._lanes = [AEC(replace(lane_config)) for _ in range(_N_CHANNELS)]
        self._shared_delay = SharedMatchedDelayEstimator(
            sample_rate=self.config.sample_rate, hop_size=hop
        )
        max_delay = int(self.config.max_delay_ms * self.config.sample_rate / 1000.0)
        self._delay_line = _SharedReferenceDelayLine(max_delay, hop)
        # Production leaves this unset and calls the explicit pre/post methods.
        # A configured object enables only the backwards-compatible one-call
        # process/process_signal helpers used by tests and offline fixtures.
        self._beamformer = beamformer
        self._post_beam_res = PostBeamResidualSuppressor(
            self.config.sample_rate, hop, self.n_freqs
        )
        self._last_delay = 0
        self._generation = 0
        self._owner_token = object()
        self._next_pre_frame = 0
        self._next_post_frame = 0

    @property
    def matched_filter_instance_count(self) -> int:
        return self._shared_delay.instance_count

    @property
    def linear_filter_instance_count(self) -> int:
        return len(self._lanes)

    @property
    def residual_suppressor_instance_count(self) -> int:
        return self._post_beam_res.instance_count

    @property
    def beamformer_configured(self) -> bool:
        return self._beamformer is not None

    def reset(self) -> None:
        self._shared_delay.reset()
        self._delay_line.reset()
        for lane in self._lanes:
            lane.reset()
        self._post_beam_res.reset()
        self._last_delay = 0
        self._generation += 1
        self._next_pre_frame = 0
        self._next_post_frame = 0

    def process_pre_beamformer(
        self, microphones: np.ndarray, render: np.ndarray
    ) -> PreBeamformerFrame:
        """Run only shared alignment and four independent linear AEC lanes."""
        microphones = np.asarray(microphones, dtype=np.float32)
        render = np.asarray(render, dtype=np.float32)
        if microphones.shape != (self.hop_size, _N_CHANNELS):
            raise ValueError(
                f"microphones must have shape ({self.hop_size}, 4), got {microphones.shape}"
            )
        if render.shape != (self.hop_size,):
            raise ValueError(
                f"render must have shape ({self.hop_size},), got {render.shape}"
            )
        if not np.all(np.isfinite(microphones)) or not np.all(np.isfinite(render)):
            raise ValueError("input contains non-finite samples")

        proxy = microphones[:, self.config.capture_proxy_channel]
        delay = self._shared_delay.accumulate(proxy, render)
        aligned_render = self._delay_line.process(render, delay.delay_samples)
        if delay.changed:
            # All filters learned against the former shared alignment.  Reset
            # all four together; never let lanes drift to different delays.
            for lane in self._lanes:
                lane.reset()
            self._last_delay = delay.delay_samples

        lane_outputs = np.empty((_N_CHANNELS, self.hop_size), dtype=np.float32)
        contexts: list[AecResContext] = []
        for channel, lane in enumerate(self._lanes):
            result = lane.process(microphones[:, channel], aligned_render)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("linear AEC lane did not return its RES context")
            lane_output, context = result
            lane_outputs[channel] = np.asarray(lane_output, dtype=np.float32)
            contexts.append(_snapshot_context(context))

        frame = PreBeamformerFrame(
            frame_index=self._next_pre_frame,
            generation=self._generation,
            _owner_token=self._owner_token,
            linear_channels=lane_outputs.T.copy(),
            contexts=tuple(contexts),
            aligned_render=np.asarray(aligned_render, dtype=np.float32).copy(),
            delay=delay,
        )
        self._next_pre_frame += 1
        return frame

    def process_post_beamformer(
        self, pre: PreBeamformerFrame, beamformed: BeamformerFrame
    ) -> FourChannelFrame:
        """Resume with one external mono/weight result and calculate mono RES.

        This method does not run SRP-PHAT or update GSC coefficients.  It only
        reuses the supplied effective weights to project AEC context, then runs
        the one stateful post-beam residual-gain calculator.  Results must be
        returned in order because that RES state is temporal.
        """
        if not isinstance(pre, PreBeamformerFrame):
            raise TypeError("pre must be a PreBeamformerFrame from this pipeline")
        if pre._owner_token is not self._owner_token:
            raise ValueError("pre-beamformer frame belongs to a different pipeline")
        if pre.generation != self._generation:
            raise ValueError("pre-beamformer frame was invalidated by reset()")
        if pre.frame_index != self._next_post_frame:
            raise ValueError(
                f"external beamformer result is out of order: got frame "
                f"{pre.frame_index}, expected {self._next_post_frame}"
            )
        samples = np.asarray(beamformed.samples, dtype=np.float32)
        if samples.shape != (self.hop_size,):
            raise ValueError(
                f"external beamformer samples must have shape "
                f"({self.hop_size},), got {samples.shape}"
            )

        fused = _fuse_contexts(pre.contexts, beamformed)
        fused = self._post_beam_res.process(fused, pre.aligned_render)
        result = FourChannelFrame(
            frame_index=pre.frame_index,
            linear_channels=pre.linear_channels.copy(),
            beamformed=samples.copy(),
            context=fused,
            delay=pre.delay,
        )
        self._next_post_frame += 1
        return result

    def process(self, microphones: np.ndarray, render: np.ndarray) -> FourChannelFrame:
        """Test/offline convenience path using an explicitly configured adapter."""
        if self._beamformer is None:
            raise RuntimeError(
                "no beamformer is configured: call process_pre_beamformer(), "
                "run the external SRP-PHAT/GSC, then call "
                "process_post_beamformer()"
            )
        pre = self.process_pre_beamformer(microphones, render)
        beamformed = self._beamformer.process(pre.linear_hops, pre.contexts)
        return self.process_post_beamformer(pre, beamformed)

    def process_signal(
        self, microphones: np.ndarray, render: np.ndarray
    ) -> tuple[np.ndarray, list[AecResContext], list[SharedDelayState]]:
        """Offline convenience path; production should use the explicit seam."""
        if self._beamformer is None:
            raise RuntimeError(
                "process_signal requires an explicit test/offline beamformer; "
                "production must drive the pre/post beamformer methods"
            )
        microphones = np.asarray(microphones, dtype=np.float32)
        render = np.asarray(render, dtype=np.float32)
        if microphones.ndim != 2 or microphones.shape[1] != _N_CHANNELS:
            raise ValueError("microphones must have shape [samples, 4]")
        n_samples = min(microphones.shape[0], render.size)
        n_complete = n_samples // self.hop_size
        output = np.zeros(n_complete * self.hop_size, dtype=np.float32)
        contexts: list[AecResContext] = []
        delays: list[SharedDelayState] = []
        for frame in range(n_complete):
            start = frame * self.hop_size
            stop = start + self.hop_size
            result = self.process(microphones[start:stop], render[start:stop])
            output[start:stop] = result.beamformed
            contexts.append(result.context)
            delays.append(result.delay)
        return output, contexts, delays
