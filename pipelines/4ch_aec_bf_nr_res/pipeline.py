"""Shared-delay, four-linear-filter AEC front end for microphone arrays.

This Python reference deliberately does not implement SRP-PHAT or the
beamformer (bf).  Its production API is split at that ownership boundary, with
the stage names of the C core (mcab_process_front / mcab_fuse)::

    front = pipeline.process_front(mics, render)
    bf = external_srp_bf(front.linear_channels)
    post = pipeline.fuse(front, bf)     # context projection only; res_gain is None

The post-beam residual gain is a separate stage, owned by the caller::

    suppressor = PostBeamResidualSuppressor(sr, pipeline.hop_size, pipeline.n_freqs)
    context = suppressor.process(post.context, post.aligned_render)   # res_gain set

The remaining stages (NR gain, min, apply, comfort noise, synthesis) are the
functions of pipelines/mono_aec_nr_res/pipeline.py over that context.

The optional ``Beamformer`` object and ``EqualWeightBeamformer`` exist only for
integration tests and offline fixtures.  Requiring the external stage to return
its complex per-bin effective weights lets the AEC error/echo spectra cross the
boundary coherently before the mono NR + RES stages, without a second
beamformer or a fifth AEC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Protocol, Sequence

import numpy as np

from lib.aec.python.modules.config import AecConfig
from lib.aec.python.modules.dataclasses import AecResContext
from lib.aec.python.modules.delay.legacy_compat import LegacyDelayShim
from lib.aec.python.modules.enums import AecMode, AecPreset
from lib.aec.python.modules.orchestrator import AEC
from lib.aec.python.modules.residual.suppression_gain import (
    EchoAudibilityConfig,
    SuppressionGain,
    SuppressorConfig,
)
from lib.aec.python.modules.aec3_scale import fft_density_scale


_SUPPORTED_SAMPLE_RATES = (16000, 48000)
_N_CHANNELS = 4

# Admission of a shared-delay CHANGE (lib/aec's Path B; same values as the C
# core's MCAB_DELAY_CHANGE_MIN_SAMPLES / _CONFIRM_SAMPLES / _CANDIDATE_TTL in
# mcab_internal.h): the movement must exceed MIN samples and be offered
# again within CONFIRM samples before the held candidate's TTL hops run out.
_DELAY_CHANGE_MIN_SAMPLES = 32
_DELAY_CHANGE_CONFIRM_SAMPLES = 16
_DELAY_CHANGE_CANDIDATE_TTL = 3


# ---------------------------------------------------------------------------
# Shared delay and beamformer handoff types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class McabDelayState:
    """Observable state of the one shared matched-filter instance.

    ``solid``: a usable accepted alignment generation exists; it never leads
    ``delay_samples`` and only ``reset()`` clears it.  ``changed``: a NEW usable
    generation begins on this hop (first acquisition included, even at applied
    delay 0); consumers flush history derived from the aligned reference.  A
    change while locked must pass the admission rule above.  Same contract as
    the C core's ``McabDelayState`` (mcab.h).
    """

    delay_samples: int
    confidence: float
    solid: bool
    changed: bool
    estimator_calls: int
    estimator_updates: int


class SharedMatchedDelayEstimator:
    """One matched filter driven by one configurable array capture proxy.

    No decimation here: the inner ``LegacyDelayShim`` is built with the true
    native ``sample_rate`` and fed raw hops.  ``EchoPathDelayEstimator`` owns
    the 48 kHz anti-alias + decimate-by-3 sidechain (as ``delay_aec.c`` does)
    and returns the delay already in native samples, so decimating or
    rescaling again here would apply it twice.
    """

    def __init__(self, sample_rate: int, hop_size: int,
                 quarantine_enabled: bool = False,
                 quarantine_s: float = 1.0) -> None:
        if sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"shared delay supports {_SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.quarantine_enabled = bool(quarantine_enabled)
        # Seconds -> hops once, with the same floor of 1 as the C core and lib/aec.
        self.quarantine_hops = max(
            1, int(round(float(quarantine_s) / (self.hop_size / self.sample_rate))))
        self._quarantine_left = -1   # -1 = disarmed, the only re-armable state
        # True native rate, never a hardcoded 16000; fed raw hops (class docstring).
        self._estimator = LegacyDelayShim(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
        )
        self._calls = 0
        self._accepted_delay = 0
        self._usable = False
        # Two-step admission for a CHANGE (see accumulate()): the held candidate
        # and the hops of life left on it; ``_pending_ttl > 0`` means held.
        self._pending_delay = 0
        self._pending_ttl = 0

    @property
    def instance_count(self) -> int:
        """Structural audit hook: this wrapper owns exactly one estimator."""
        return 1

    def reset(self) -> None:
        self._estimator.reset()
        self._calls = 0
        self._accepted_delay = 0
        self._usable = False
        self._quarantine_left = -1
        self._pending_delay = 0
        self._pending_ttl = 0

    def accumulate(
        self,
        capture_proxy: np.ndarray,
        render: np.ndarray,
        proxy_lane_cancelling: bool = False,
    ) -> McabDelayState:
        """``proxy_lane_cancelling`` mirrors the C core's read of
        ``p->lanes[cfg.capture_proxy_channel]`` in ``update_shared_delay()``:
        True when the lane the estimator is fed from still cancels at the
        applied alignment (that lane only: its cancellation is the evidence
        about the estimate).  Passed in because this object does not own the
        lanes.  Ignored unless ``delay_backward_quarantine_enabled`` is on."""
        capture_proxy = np.asarray(capture_proxy, dtype=np.float32)
        render = np.asarray(render, dtype=np.float32)
        if (
            capture_proxy.ndim != 1
            or render.ndim != 1
            or capture_proxy.shape != render.shape
        ):
            raise ValueError("capture proxy and render must be equal-length 1-D hops")
        self._calls += 1
        # Raw native-rate hop; the estimator decimates internally.
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
        # One hop of the held candidate's life is spent here, ahead of the
        # quarantine and admission, where lib/aec ages its pending delay.
        if self._pending_ttl > 0:
            self._pending_ttl -= 1
            if self._pending_ttl <= 0:
                self._pending_ttl = 0
                self._pending_delay = 0
        # Backward-jump quarantine (McabConfig
        # .delay_backward_quarantine_enabled): only once a usable generation
        # exists and only for an estimate strictly EARLIER than the accepted
        # one.  Armed once per continuous backward episode, then one tick per
        # hop; a forward/ineligible estimate or lost cancellation disarms it.
        # Mirrors the C core's update_shared_delay().
        if (
            eligible
            and self.quarantine_enabled
            and self._usable
            and estimated < self._accepted_delay
            and proxy_lane_cancelling
        ):
            if self._quarantine_left < 0:
                self._quarantine_left = self.quarantine_hops
            if self._quarantine_left > 0:
                self._quarantine_left -= 1
                eligible = False
        else:
            self._quarantine_left = -1
        # ``solid`` comes from the same acceptance test that writes the accepted
        # delay (not the estimator's raw confidence, which could lead the
        # applied delay) and is sticky once a generation exists; ``changed``
        # also fires on an acquisition that lands on delay 0.  A CHANGE must
        # additionally pass the admission rule above (a movement seen once and
        # gone must not shift four converged filters); first acquisition is
        # exempt, and a too-small movement does not clear the candidate.
        was_usable = self._usable
        now_usable = was_usable or eligible
        changed = False
        if eligible:
            if not was_usable:
                changed = True
                self._pending_delay = 0
                self._pending_ttl = 0
            elif abs(estimated - self._accepted_delay) > _DELAY_CHANGE_MIN_SAMPLES:
                if (
                    self._pending_ttl > 0
                    and abs(estimated - self._pending_delay)
                    < _DELAY_CHANGE_CONFIRM_SAMPLES
                ):
                    changed = True
                    self._pending_delay = 0
                    self._pending_ttl = 0
                else:
                    self._pending_delay = estimated
                    self._pending_ttl = _DELAY_CHANGE_CANDIDATE_TTL
        if changed:
            self._accepted_delay = estimated
        self._usable = now_usable
        return McabDelayState(
            delay_samples=int(self._accepted_delay),
            confidence=float(self._estimator.confidence),
            solid=bool(now_usable),
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
        # Whole-hop decision, as the C core's align_render(): the ring serves the
        # offset only once the samples seen cover ``delay_samples``; before that
        # the far content is the raw render hop (never silence).  Per hop, not
        # per sample, so raw and shifted audio are never spliced in one frame.
        if start >= int(delay_samples):
            source = absolute - int(delay_samples)
            out = self._ring[source % self._capacity]
        else:
            out = render.copy()
        self._samples_seen += render.size
        return out


@dataclass(frozen=True)
class McabBeam:
    """One external-beamformer result.

    ``weights[ch, bin]`` uses the convention
    ``output_spec[bin] = sum(weights[ch, bin] * input_spec[ch, bin])``.
    """

    samples: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class McabFront:
    """Owned handoff from four linear AEC lanes to external SRP-PHAT/BF.

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
    delay: McabDelayState

    @property
    def linear_hops(self) -> np.ndarray:
        """Channel-first ``[4, hop]`` view used by the test adapter contract."""
        return self.linear_channels.T


class Beamformer(Protocol):
    def process(
        self, linear_hops: np.ndarray, contexts: Sequence[AecResContext]
    ) -> McabBeam:
        ...


class EqualWeightBeamformer:
    """Deterministic test adapter; replace with the external beamformer."""

    def process(
        self, linear_hops: np.ndarray, contexts: Sequence[AecResContext]
    ) -> McabBeam:
        linear_hops = np.asarray(linear_hops, dtype=np.float32)
        if linear_hops.ndim != 2 or linear_hops.shape[0] != _N_CHANNELS:
            raise ValueError("linear_hops must have shape [4, hop]")
        if len(contexts) != _N_CHANNELS:
            raise ValueError("exactly four AEC contexts are required")
        n_freqs = int(np.asarray(contexts[0].error_spec).size)
        weights = np.full((_N_CHANNELS, n_freqs), 0.25, dtype=np.complex64)
        return McabBeam(
            samples=np.mean(linear_hops, axis=0, dtype=np.float32),
            weights=weights,
        )


@dataclass(frozen=True)
class McabConfig:
    sample_rate: int = 16000
    frame_size: Optional[int] = None
    hop_size: Optional[int] = None
    capture_proxy_channel: int = 0
    max_delay_ms: float = 1024.0
    # Backward-jump quarantine on a shared-delay CHANGE (default off; mirrors
    # McabConfig::delay_backward_quarantine_enabled / _s): an estimate
    # strictly EARLIER than the accepted delay is held while the estimator's own
    # capture lane (capture_proxy_channel) still cancels at the applied
    # alignment, and is adopted at the window's expiry or when cancellation
    # collapses.  A larger estimate and first acquisition are never held.
    delay_backward_quarantine_enabled: bool = False
    # Window in SECONDS, converted to hops once at construction (minimum 1).
    # Inert while delay_backward_quarantine_enabled is False.
    delay_backward_quarantine_s: float = 1.0

    def resolved_grid(self) -> tuple[int, int]:
        if self.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"four-channel path supports {_SUPPORTED_SAMPLE_RATES}, got {self.sample_rate}"
            )
        # 16 kHz default is 256/128 (8 ms hop), as the C core's
        # derive_flow (mcab.c); 512/256 is an explicit alternate.
        default_frame = 256 if self.sample_rate == 16000 else 1024
        frame = default_frame if self.frame_size is None else int(self.frame_size)
        hop = frame // 2 if self.hop_size is None else int(self.hop_size)
        # Same whitelist as the C core (mcab.c derive_flow):
        # 16 kHz frame/FFT in {256, 512}, 48 kHz only 1024.
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


def mcab_apply_lane_contract(config: McabConfig, lane: AecConfig) -> AecConfig:
    """Twin of C ``mcab_apply_lane_contract``: the flow's contract stamped onto an
    application-composed lane config (the flow's grid, no own delay estimation, the
    linear context returned without a gain of its own).  Every other field is the
    application's.  Raises ValueError for a lane built for another sample rate."""
    frame, hop = config.resolved_grid()
    if lane.sample_rate != config.sample_rate:
        raise ValueError("the lane config is for another sample rate")
    return replace(
        lane,
        frame_size=frame,
        hop_size=hop,
        enable_delay_est=False,
        fixed_delay_samples=-1,
        enable_res=False,
        return_res_context=True,
    )


def mcab_lane_config(config: McabConfig) -> AecConfig:
    """Twin of C ``mcab_lane_config``: the default lane (PBFDKF, 52 ms filter at 16 kHz,
    64 ms at 48 kHz) with the contract stamped."""
    filter_length = config.sample_rate * (64 if config.sample_rate == 48000 else 52) // 1000
    return mcab_apply_lane_contract(
        config,
        AecConfig(sample_rate=config.sample_rate, filter_length=int(filter_length),
                  mode=AecMode.PBFDKF),
    )


@dataclass(frozen=True)
class McabFrame:
    frame_index: int
    linear_channels: np.ndarray
    beamformed: np.ndarray
    context: AecResContext
    delay: McabDelayState
    aligned_render: np.ndarray


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
        formed_output=copied(context.formed_output),
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
    contexts: Sequence[AecResContext], beamformed: McabBeam
) -> AecResContext:
    """Cross the beamformer boundary without creating a fifth AEC.

    Error/near/echo spectra are exactly coherent for the supplied weights.
    R2 is coherently combined using each channel's echo-estimate phase.
    This function fuses measurements only. ``res_gain`` is left None: the
    caller's one stateful ``PostBeamResidualSuppressor`` computes it afterward,
    and per-lane gains are intentionally ignored.
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
    # The lanes normally agree bit for bit, but on a mixed warm/soft-realign
    # hop a soft-reset lane has zeroed its far_buffer OLA history
    # (aec_reset_filter_for_realign) while a warm lane keeps it, so their far
    # FFTs differ for that hop.  The single post-beam RES needs one X: lane 0's.
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
    # Match fuse_contexts() in the C core: a lane that contributes no
    # spectrum must not steer the post-beam suppressor's three control
    # scalars. Test coefficients directly, not normalized magnitudes, so a
    # genuinely nonzero coefficient cannot disappear through underflow.
    contributes = np.any((weights.real != 0.0) | (weights.imag != 0.0), axis=1)
    contributors = [c for c, active in zip(contexts, contributes) if active]
    far_powers = np.asarray([c.far_power for c in contexts], dtype=np.float64)
    if not np.allclose(far_powers, far_powers[0], rtol=1e-6, atol=1e-9):
        raise ValueError("linear AEC lanes do not share one far-end power")
    shared_far_power = float(far_powers[0])

    return AecResContext(
        raw_output=samples.copy(),
        formed_output=samples.copy(),
        echo_spec=echo_spec.astype(np.complex64),
        far_power=shared_far_power,
        far_spec=far_spec.astype(np.complex64),
        near_spec=near_spec.astype(np.complex64),
        filter_converged=all(c.filter_converged for c in contributors),
        # Contributors only, as filter_converged: a fused context must carry a
        # real verdict (None reads as "not usable" in run_nr_spectrum's gate).
        usable_linear=all(bool(c.usable_linear) for c in contributors),
        erle_factor=float(min(c.erle_factor for c in contexts)),
        dt_indicator=float(max(c.dt_indicator for c in contributors)),
        divergence=float(max(c.divergence for c in contexts)),
        over_sub=float(max(c.over_sub for c in contexts)),
        saturation_level=float(max(c.saturation_level for c in contributors)),
        erl_estimate=float(
            sum(float(w) * float(c.erl_estimate) for w, c in zip(norm_weight, contexts))
        ),
        error_spec=error_spec.astype(np.complex64),
        # None, so a reader that skips the suppressor fails instead of reading
        # a silent unity gain. Never select/min a lane gain.
        res_gain=None,
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

    AecResContext exports no unbounded R2 or full stationarity/AecState
    surface, so bounded R2 feeds both gain inputs and no stationary mask is
    supplied (an explicit parity limitation).
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        n_freqs: int,
        preset: Optional[AecPreset] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self.n_freqs = int(n_freqs)
        self._state = _PostBeamAecState()
        self._initial_state = True
        # None means "take AecConfig's own dataclass default".  With a preset,
        # all three (far-active floor, DT floor, split-floor gate) are seeded
        # from one AecConfig.from_preset(preset), as aec_config_from_preset()
        # does.  They are stored because _build_gain() (re-run by reset())
        # builds a fresh AecConfig each time; set_preset() retargets them live.
        if preset is not None:
            preset_cfg = AecConfig.from_preset(preset)
            self._split_floor_far_active_db = preset_cfg.min_gain_floor_far_active_db
            self._split_floor_dt_db = preset_cfg.min_gain_floor_dt_db
            self._split_floor_enabled = preset_cfg.min_gain_split_floor_enabled
        else:
            self._split_floor_far_active_db = None
            self._split_floor_dt_db = None
            self._split_floor_enabled = None
        self._build_gain()

    @classmethod
    def from_aec_config(cls, config: AecConfig) -> "PostBeamResidualSuppressor":
        """Build from an AecConfig: grid and the three strength fields come
        from it, so the strength survives ``reset()``."""
        sup = cls(config.sample_rate, config.hop_size, config.frame_size // 2 + 1)
        sup._split_floor_far_active_db = config.min_gain_floor_far_active_db
        sup._split_floor_dt_db = config.min_gain_floor_dt_db
        sup._split_floor_enabled = config.min_gain_split_floor_enabled
        sup._build_gain()
        return sup

    def _build_gain(self) -> None:
        config = AecConfig(
            sample_rate=self.sample_rate,
            frame_size=2 * self.hop_size,
            hop_size=self.hop_size,
        )
        if self._split_floor_far_active_db is not None:
            config = replace(
                config,
                min_gain_floor_far_active_db=self._split_floor_far_active_db,
            )
        if self._split_floor_dt_db is not None:
            config = replace(
                config,
                min_gain_floor_dt_db=self._split_floor_dt_db,
            )
        if self._split_floor_enabled is not None:
            config = replace(
                config,
                min_gain_split_floor_enabled=self._split_floor_enabled,
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

    def set_preset(self, preset, ramp_ms: float = 0.0) -> None:
        """Retarget the strength this stage applies, on a running instance.

        This, not the four lanes, shapes the output (the lanes compute no gain
        of their own).  Mirrors the C aec_res_set_preset(): a
        three-field retarget (far-active floor, DT floor, split-floor gate);
        all four presets, NO_FLOOR included, are switchable.  The stored value
        is what a later reset() rebuilds from.
        """
        # Coerce first: from_preset() falls back to balanced for unknown values;
        # a setter must reject them.
        try:
            preset = AecPreset(preset)
        except (TypeError, ValueError):
            raise ValueError(f"unknown AEC preset: {preset!r}")
        preset_cfg = AecConfig.from_preset(preset)
        db = preset_cfg.min_gain_floor_far_active_db
        dt_db = preset_cfg.min_gain_floor_dt_db
        enabled = preset_cfg.min_gain_split_floor_enabled
        # The dB setters validate and raise without writing, so they run first;
        # the gate setter cannot fail and runs last (as orchestrator.py's
        # set_preset()), keeping the stored values and the live floors in step.
        self._gain.set_split_floor_far_active_db(db, ramp_ms)
        self._split_floor_far_active_db = db
        self._gain.set_split_floor_dt_db(dt_db)
        self._split_floor_dt_db = dt_db
        self._gain.set_split_floor_enabled(enabled)
        self._split_floor_enabled = enabled

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
            # AecResContext has no unbounded R2; bounded R2 stands in.
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

class McabPipeline:
    """Pre/post shell around an externally owned SRP-PHAT/BF stage."""

    def __init__(
        self,
        config: Optional[McabConfig] = None,
        beamformer: Optional[Beamformer] = None,
        lane_config: Optional[AecConfig] = None,
    ) -> None:
        """``lane_config``: the application's lane parameters (preset, filter length,
        ...); the flow stamps its contract onto them.  None selects mcab_lane_config()."""
        self.config = config or McabConfig()
        frame, hop = self.config.resolved_grid()
        self.frame_size = frame
        self.hop_size = hop
        self.n_freqs = frame // 2 + 1
        lane_config = (mcab_lane_config(self.config) if lane_config is None
                       else mcab_apply_lane_contract(self.config, lane_config))
        # ``replace`` gives each lane its own config object.
        self._lanes = [AEC(replace(lane_config)) for _ in range(_N_CHANNELS)]
        self._shared_delay = SharedMatchedDelayEstimator(
            sample_rate=self.config.sample_rate, hop_size=hop,
            quarantine_enabled=self.config.delay_backward_quarantine_enabled,
            quarantine_s=self.config.delay_backward_quarantine_s,
        )
        max_delay = int(self.config.max_delay_ms * self.config.sample_rate / 1000.0)
        self._delay_line = _SharedReferenceDelayLine(max_delay, hop)
        # Production leaves this unset and calls the explicit pre/post methods;
        # a configured object enables only the one-call process/process_signal
        # helpers used by tests and offline fixtures.
        self._beamformer = beamformer
        self._last_delay = 0
        self._generation = 0
        self._owner_token = object()
        self._next_pre_frame = 0
        self._next_post_frame = 0
        self._realign_warm_lanes = 0
        self._realign_soft_lanes = 0

    @property
    def matched_filter_instance_count(self) -> int:
        return self._shared_delay.instance_count

    @property
    def linear_filter_instance_count(self) -> int:
        return len(self._lanes)

    @property
    def beamformer_configured(self) -> bool:
        return self._beamformer is not None

    @property
    def realign_warm_lane_count(self) -> int:
        """How many lane realigns took the warm tap-transfer path.

        Cumulative since construction or the last ``reset()``.  Every hop that
        publishes ``changed`` sweeps all four lanes, so this and
        ``realign_soft_lane_count`` grow by 4 between them on such a hop.  Warm:
        the learned IR was shifted and the cancellation survived.  Soft: the
        lane's filter was restarted, or the delta was 0 and nothing happened
        (``apply_external_realign`` returns 0 for both), so read the pair
        against the delay trajectory.  The rejection outcome (-1) cannot occur
        here (the lanes resolve to EXTERNAL_ALIGNED).  Mirrors the C core's
        ``mcab_realign_warm_lane_count()`` pair.
        """
        return self._realign_warm_lanes

    @property
    def realign_soft_lane_count(self) -> int:
        return self._realign_soft_lanes

    def reset(self) -> None:
        self._shared_delay.reset()
        self._delay_line.reset()
        for lane in self._lanes:
            lane.reset()
        self._last_delay = 0
        # Zeroed with the lanes' state (as mcab_reset()).
        self._realign_warm_lanes = 0
        self._realign_soft_lanes = 0
        self._generation += 1
        self._next_pre_frame = 0
        self._next_post_frame = 0

    def process_front(
        self, microphones: np.ndarray, render: np.ndarray
    ) -> McabFront:
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
        # Read before the lanes process this hop, so it reports last hop's
        # cancellation (as lib/aec's Path-B guard and the C core, which calls
        # update_shared_delay() ahead of its lane loop).  Capture-proxy lane
        # only, and only when the quarantine is on.
        proxy_lane_cancelling = bool(
            self.config.delay_backward_quarantine_enabled
            and self._lanes[
                self.config.capture_proxy_channel].linear_is_cancelling()
        )
        delay = self._shared_delay.accumulate(proxy, render,
                                              proxy_lane_cancelling)
        aligned_render = self._delay_line.process(render, delay.delay_samples)
        if delay.changed:
            # All four filters move together on the same hop as the reference.
            # Realign, not reset: ``apply_external_realign`` shifts each filter
            # by the delta (warm tap-transfer, else a filter-only restart), so
            # cancellation survives and the WOLA sequences continue.
            # ``_last_delay`` is the alignment served before this hop's decision
            # (0 until the estimate first turns solid).  Same sweep as the C
            # flow's mcab_process_front (mcab.c).
            delta = int(delay.delay_samples) - int(self._last_delay)
            for lane in self._lanes:
                outcome = lane.apply_external_realign(delta)
                if outcome == 1:
                    self._realign_warm_lanes += 1
                elif outcome == 0:
                    self._realign_soft_lanes += 1
            self._last_delay = delay.delay_samples

        lane_outputs = np.empty((_N_CHANNELS, self.hop_size), dtype=np.float32)
        contexts: list[AecResContext] = []
        for channel, lane in enumerate(self._lanes):
            result = lane.process(microphones[:, channel], aligned_render)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("linear AEC lane did not return its RES context")
            _lane_output, context = result
            # Same signal as context.error_spec (selection, crossfade, WOLA).
            lane_outputs[channel] = np.asarray(
                context.formed_output, dtype=np.float32
            )
            contexts.append(_snapshot_context(context))

        frame = McabFront(
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

    def fuse(
        self, pre: McabFront, beamformed: McabBeam
    ) -> McabFrame:
        """Resume with one external mono/weight result and project the context.

        This method does not run SRP-PHAT or update the beamformer's
        coefficients.  It only reuses the supplied effective weights to project
        the AEC context; ``context.res_gain`` stays None for the caller's
        ``PostBeamResidualSuppressor``.  Results must be returned in order
        because the shared delay generation is temporal.
        """
        if not isinstance(pre, McabFront):
            raise TypeError("pre must be a McabFront from this pipeline")
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
        result = McabFrame(
            frame_index=pre.frame_index,
            linear_channels=pre.linear_channels.copy(),
            beamformed=samples.copy(),
            context=fused,
            delay=pre.delay,
            aligned_render=pre.aligned_render.copy(),
        )
        self._next_post_frame += 1
        return result

    def process(self, microphones: np.ndarray, render: np.ndarray) -> McabFrame:
        """Test/offline convenience path using an explicitly configured adapter.
        ``result.context.res_gain`` is None."""
        if self._beamformer is None:
            raise RuntimeError(
                "no beamformer is configured: call process_front(), "
                "run the external SRP-PHAT/BF, then call "
                "fuse()"
            )
        pre = self.process_front(microphones, render)
        beamformed = self._beamformer.process(pre.linear_hops, pre.contexts)
        return self.fuse(pre, beamformed)

    def process_signal(
        self, microphones: np.ndarray, render: np.ndarray
    ) -> tuple[np.ndarray, list[AecResContext], list[McabDelayState]]:
        """Offline convenience path; production should use the explicit seam.
        The returned contexts carry ``res_gain=None``."""
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
        delays: list[McabDelayState] = []
        for frame in range(n_complete):
            start = frame * self.hop_size
            stop = start + self.hop_size
            result = self.process(microphones[start:stop], render[start:stop])
            output[start:stop] = result.beamformed
            contexts.append(result.context)
            delays.append(result.delay)
        return output, contexts, delays
