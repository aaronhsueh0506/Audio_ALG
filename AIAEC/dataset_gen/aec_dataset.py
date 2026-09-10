"""AEC scenario simulator: renders parent sequences of 5 aligned stems.

THE SIGNAL MODEL THIS FILE EXISTS TO PRODUCE
--------------------------------------------
    Y = S + N + D        microphone   (S near speech, N local noise, D echo)
    X                    far-end reference
    D_hat                frozen-linear echo estimate
    E     = Y - D_hat    materialized linear error          <-- RES+NR input
    R     = D - D_hat    residual echo -- audit only, not target

The four acoustic stems stay separated and the fifth persisted channel is the
real Python-PBFDKF linear error. The filter runs once over the complete parent
sequence before it is split into chunks, so its adaptation state remains
continuous while every trainer can still randomize chunks freely.

Everything a model may need is derivable from the persisted stems:
    Y = mic_postclip     X = far_render
    S = near_speech
    S_early = near_target (DeepVQE-S and Align-CRUSE's dereverb target)

``D`` (echo), ``N`` (local noise) and the pre-clip/AGC ``mic_preclip`` are NOT
persisted -- no model task targets echo cancellation without denoising any
more, so none of the three needs to reach a trainer as its own channel, and no
candidate is meant to see an oracle residual. All three are
still COMPUTED on every render and returned as ``RenderedSequence.audit``, so
the corpus's central invariants (``mic_preclip == S+N+D``, "echo really is a
delayed copy of X") stay checked at generation time -- see
``tests/test_aec_dataset.py``, which verifies them directly against the
renderer rather than from a packed shard.

REUSE
-----
The DSP primitives come from ``AINR/dataset_gen/dataset.py`` -- the NR generator and
this one share one ``prepare_rir``, one ``fftconvolve``, one ``active_rms``, and
the same discrete SNR convention.  Only genuinely new behaviour lives here:
loudspeaker nonlinearity, echo-path switching, sample-rate offset, reference
dropout and the mic AGC.
"""

import configparser
import dataclasses
import hashlib
import itertools
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from AINR.dataset_gen.dataset import (
    BIQUAD_TYPES,
    _biquad_coeffs,
    active_rms,
    apply_biquad,
    apply_clipping,
    delay_signal,
    fftconvolve,
    parse_snr_values,
    prepare_rir,
    prevent_clipping,
    rand_biquad_filter,
    sample_snr,
    simulate_upsampled_source,
)
from .aec_features import BASE_STEM_ORDER, STEM_ORDER, alpha_from_tau
from .linear_aec import (
    MATCHED_REACH_MS,
    LinearAecContract,
    linear_aec_contract_from_config,
    materialize_linear_error,
)
from .manifest import SourcePools, config_hash
from .path_drift_metrics import mixture_correlation, position_gram


__all__ = [
    'ACOUSTIC_TAILS',
    'ACOUSTIC_TAIL_RANGE_KEYS',
    'CHUNK_EVENT_LABELS',
    'DELAY_STEP_EDGE_MARGIN_SEC',
    'DRIFT_MODES',
    'ECHO_MODES',
    'IMPAIRMENTS',
    'NONLINEAR_MODELS',
    'PATH_MOTION_MODES',
    'SCENARIOS',
    'STATIC_PATH',
    'TALK_MODES',
    'AecSequenceRenderer',
    'DeviceModel',
    'PathMotion',
    'RenderedSequence',
    'SequencePlan',
    'apply_agc',
    'apply_loudspeaker_nonlinearity',
    'apply_weight_schedule',
    'chunk_samples_from_config',
    'configured_device_ids',
    'configured_nonlinear_models',
    'device_for_id',
    'drift_corners',
    'drift_keyframes',
    'drift_weights',
    'gain_walk_db',
    'moving_chunks',
    'nonlinearity_by_id',
    'path_motion_mode',
    'plan_sequences',
    'position_correlation_target',
    'resolve_acoustic_tails',
    'resolve_sequence_plan',
    'resample_by_ratio',
    'simulate_codec',
    'solve_mixture_depth',
    'stable_seed',
]


# The scenario vocabulary.  A chunk's `scenario` metadata is always one of
# these, so a downstream filter such as `meta['scenario'] == 'double_talk'`
# never silently matches nothing because of a typo.
# New corpora plan speech activity, echo availability, and physical
# impairments independently.  ``SCENARIOS`` below remains the public label
# vocabulary and the compatibility vocabulary for old hand-built
# ``SequencePlan(scenario=...)`` callers.
TALK_MODES = ('far_only', 'near_only', 'double_talk', 'duplex_random')
ECHO_MODES = ('normal', 'ref_dropout', 'far_active_no_echo')
IMPAIRMENTS = (
    'echo_path_change',
    'slow_drift',
    'movement',
    'nonlinear_spk',
    'clipping_agc',
    'delay_jitter',
    'delay_step',
    'sro',
    'codec_mismatch',
)

# How the echo path moves during one sequence, drawn ONCE as a categorical
# choice rather than as three independent Bernoullis.  A path that both drifts
# continuously and jumps once between two fixed positions describes no device,
# and two of them at once would make the per-chunk movement label ambiguous.
# Order is load-bearing: the categorical draw walks this tuple.
PATH_MOTION_MODES = ('slow_drift', 'movement', 'echo_path_change')
# The remaining outcome of that draw: the path holds still for the whole
# sequence.  Named because it is a value the metadata carries, not an absence.
STATIC_PATH = 'static'


@dataclasses.dataclass(frozen=True)
class _MotionPolicy:
    """What one motion mode asks of the renderer.

    ``needs_waypoints`` is the fewest positions in one room the mode can be
    rendered with, or ``None`` for "whatever [path_motion] waypoints_min says".
    """

    needs_waypoints: Optional[int]
    solves_mixture: bool
    near_gets_trajectory: bool
    renders_crossfade: bool


# The trajectory modes rendered as a continuous weight schedule over K
# positions, and the one rendered as a single crossfade between two pure ones.
# Every per-mode decision reads this table instead of a membership test, so a
# mode added to PATH_MOTION_MODES is described in one place and a mode nothing
# describes is refused rather than quietly rendered still.
_DRIFT_POLICY = _MotionPolicy(needs_waypoints=None, solves_mixture=True,
                              near_gets_trajectory=True,
                              renders_crossfade=False)
MOTION_POLICY = {
    STATIC_PATH: _MotionPolicy(needs_waypoints=1, solves_mixture=False,
                               near_gets_trajectory=False,
                               renders_crossfade=False),
    'slow_drift': _DRIFT_POLICY,
    'movement': _DRIFT_POLICY,
    'echo_path_change': _MotionPolicy(needs_waypoints=2, solves_mixture=False,
                                      near_gets_trajectory=False,
                                      renders_crossfade=True),
}
DRIFT_MODES = frozenset(mode for mode, policy in MOTION_POLICY.items()
                        if policy.solves_mixture)


def motion_policy(mode: str) -> _MotionPolicy:
    """What ``mode`` asks of the renderer."""
    policy = MOTION_POLICY.get(mode)
    if policy is None:
        raise ValueError(
            f"unknown path motion mode {mode!r}; the modes that can be "
            f"rendered are {[STATIC_PATH] + list(PATH_MOTION_MODES)}")
    return policy

@dataclasses.dataclass(frozen=True)
class _OptionSpec:
    """What one config key means to the three checks that read it.

    ``check`` names what the value may BE, ``pair`` is the ``..._max`` key a
    floor is bounded against, and ``retired`` is the key this one replaced.
    ``check=None`` is a key the renderer reads but holds to no range.
    """

    check: Optional[str] = None
    pair: Optional[str] = None
    retired: Optional[str] = None


# Per-drift-mode [path_motion] keys, and the ones shared by both modes.  ONE
# table per section, because three checks read the same list and a key added to
# two of them is a hole nobody sees: the PRESENCE contract the planner enforces
# (a key listed here cannot reach a render worker missing from a config), the
# refusal of a retired name, and the range validator all derive from it.
#   'positive_range'  the floor of a finite min <= max pair, itself > 0
#   'paired'          the ceiling of such a pair, checked with its floor
#   'whole_range'     a min <= max pair of counts, floor >= 0
#   'waypoints'       the trajectory length, bounded by _waypoint_range
#   the rest name what the value may be on its own
DRIFT_MODE_OPTION_SPECS = {
    'segment_sec_min': _OptionSpec('positive_range', pair='segment_sec_max'),
    'segment_sec_max': _OptionSpec('paired'),
    'dwell_p': _OptionSpec('probability'),
    'dwell_sec_min': _OptionSpec('positive_range', pair='dwell_sec_max'),
    'dwell_sec_max': _OptionSpec('paired'),
    # Retired: a mixture depth is not a path correlation and reads as a
    # plausible number in the same range, so a config carrying the old name
    # describes a DIFFERENT model and is named rather than ignored.
    'position_correlation': _OptionSpec('open_unit', retired='depth'),
    'gain_sigma_db_per_sec': _OptionSpec('non_negative'),
    'gain_clamp_db': _OptionSpec('positive'),
}
SHARED_PATH_MOTION_SPECS = {
    'waypoints_min': _OptionSpec('waypoints'),
    'waypoints_max': _OptionSpec('waypoints'),
    'near_slowdown': _OptionSpec('positive'),
    'near_gain_scale': _OptionSpec('non_negative'),
    'gain_update_sec': _OptionSpec('positive'),
    'gain_recentre_sec': _OptionSpec('positive'),
    'moving_label_weight_delta': _OptionSpec('open_closed_unit'),
    'far_active_after_event_sec': _OptionSpec('non_negative'),
}
DRIFT_MODE_OPTION_STEMS = tuple(DRIFT_MODE_OPTION_SPECS)
RETIRED_DRIFT_MODE_OPTION_STEMS = {
    spec.retired: stem
    for stem, spec in DRIFT_MODE_OPTION_SPECS.items() if spec.retired
}
SHARED_PATH_MOTION_OPTIONS = tuple(SHARED_PATH_MOTION_SPECS)
PATH_MOTION_OPTIONS = tuple(
    f'{mode}_{stem}'
    for mode in sorted(DRIFT_MODES) for stem in DRIFT_MODE_OPTION_STEMS
) + SHARED_PATH_MOTION_OPTIONS

# The same one-table contract for [echo_path]: every key a render worker reads
# out of that section, whether it describes the bulk delay, the one-shot
# crossfade, the delay step or the jitter walk. The section carries no planner
# probability, so nothing else would notice a missing key until a worker drew
# the impairment that reads it.
ECHO_PATH_OPTION_SPECS = {
    'bulk_delay_ms_min': _OptionSpec('range', pair='bulk_delay_ms_max'),
    'bulk_delay_ms_max': _OptionSpec('paired'),
    'delay_step_ms_min': _OptionSpec('range', pair='delay_step_ms_max'),
    'delay_step_ms_max': _OptionSpec('paired'),
    'jitter_steps_min': _OptionSpec('whole_range', pair='jitter_steps_max'),
    'jitter_steps_max': _OptionSpec('paired'),
    'jitter_ms_min': _OptionSpec('range', pair='jitter_ms_max'),
    'jitter_ms_max': _OptionSpec('paired'),
    'jitter_fade_sec': _OptionSpec('positive'),
    'path_change_fade_sec': _OptionSpec('positive'),
}
ECHO_PATH_OPTIONS = tuple(ECHO_PATH_OPTION_SPECS)

# Per-chunk labels that no sequence-level axis can produce, because they
# describe WHEN inside the sequence something happened.  Kept in SCENARIOS so
# a chunk's `scenario` stays inside one declared vocabulary.
CHUNK_EVENT_LABELS = ('echo_path_moving',)

# Low-probability operating points that widen a scalar acoustic distribution
# without moving the ordinary range.  These are deliberately a separate axis
# from IMPAIRMENTS: a quiet far end and a strong echo are valid physical
# operating points, not corruptions of the signal.
ACOUSTIC_TAILS = (
    'long_delay',
    'quiet_far',
    'strong_echo',
)
# Config key stem of each tail's range: `<stem>_min` / `<stem>_max` under
# [acoustic_tails]. The planner's required-option list, the range validator
# and the renderer's range pick all read this one mapping.
ACOUSTIC_TAIL_RANGE_KEYS = {
    'long_delay': 'long_delay_ms',
    'quiet_far': 'quiet_far_dbfs',
    'strong_echo': 'strong_echo_erl_db',
}
# A quiet reference with an ordinary 0--30 dB ERL puts the echo at the mic
# down to -70 dBFS, below the local-noise floor: the sequence teaches nothing
# and its measured per-chunk labels stop describing the audio. quiet_far
# therefore caps the ERL draw unless strong_echo supplies its own range.
QUIET_FAR_ERL_CAP_KEY = 'quiet_far_erl_db_max'

# Derived, not restated: the label vocabulary IS the three axes projected onto
# one string, and spelling it out again is how the two drift apart.  'normal'
# is not a label -- it is the absence of an echo-mode event.
# ⚠ A chunk's `scenario` is one of these, and so is a sequence's
# `sequence_scenario`, but the two do not draw from the same subset:
# 'duplex_random' is a sequence-level talk mode that no chunk is ever labelled.
SCENARIOS = (TALK_MODES
             + tuple(mode for mode in ECHO_MODES if mode != 'normal')
             + IMPAIRMENTS
             + CHUNK_EVENT_LABELS)

# Impairments that act on the echo path, so they have nothing to act on when
# there is no far end or no acoustic return.  Named once because two places
# depend on agreeing about it: plan_sequences() STRIPS these from such plans
# and resolve_sequence_plan() REJECTS them, and a new echo-path impairment
# added to only one of the two is either never generated or always fatal.
ECHO_PATH_IMPAIRMENTS = frozenset({
    'echo_path_change', 'slow_drift', 'movement', 'nonlinear_spk',
    'delay_jitter', 'delay_step', 'sro', 'codec_mismatch',
})

# The impairments [complex_cases] p_dt_stress_combo forces together.  Named
# because three places must agree on it: the planner that sets it, the
# renderer's forced-edge-overlap gate that detects it, and the CLI census that
# counts it.  Two of those three fail SILENTLY when the definition moves.
# ⚠ Path motion is deliberately NOT in this bundle. It is drawn on its own
# axis for 67% of compatible echo paths at the shipped defaults, so bundling
# it here would make
# "DT while the path moves" indistinguishable from "DT while the loudspeaker
# distorts and the capture clips": the two would only ever appear together and
# neither could be attributed.
DT_STRESS_IMPAIRMENTS = frozenset({
    'nonlinear_spk', 'clipping_agc',
})

# The hard acoustic DT intersection.  In addition to independently sampled
# tails, [complex_cases] can force this set so a finite campaign contains
# examples at both the cold-start and mature-PBFDKF DT boundaries. Trainers
# currently reset neural state per chunk, so this does not claim to exercise a
# 20--30 s continuously carried GRU state.
DT_ACOUSTIC_TAILS = frozenset(ACOUSTIC_TAILS)

# Scenarios whose defining event occupies the WHOLE sequence.  The others are
# localised in time and get a per-chunk label instead -- see _chunk_scenario.
# ⚠ 'far_active_no_echo' is deliberately NOT here: its label is measured per
# chunk from the reference's actual activity, never asserted for the sequence.
WHOLE_SEQUENCE_SCENARIOS = frozenset({
    'nonlinear_spk', 'clipping_agc', 'delay_jitter', 'sro', 'codec_mismatch',
})

# Memoryless loudspeaker distortion models.  A device is pinned to exactly one
# of these for the whole corpus, which is what makes device-disjoint validation
# mean "an unheard loudspeaker" rather than "the same loudspeaker again".
NONLINEAR_MODELS = (
    'linear',
    'softclip_tanh',
    'arctan',
    'hardclip',
    'sef',
    'poly_odd',
    'diode',
)

# A chunk counts as containing speech above this active level.  Only used to
# label chunks honestly (far_only / near_only / double_talk); every stem is
# scaled to at least -40 dBFS, so this sits well below anything intentional.
ACTIVITY_LABEL_DBFS = -55.0

# How close to either end of a sequence a one-shot delay step may land.  The
# step has to be measurable, and a re-timing in the first or last few frames
# leaves no interval on one side of it to measure the delay from; a sequence
# shorter than twice this margin gets no step at all.  Named because the test
# that verifies the recorded instant has to use the same number.
DELAY_STEP_EDGE_MARGIN_SEC = 0.5


# ============================================================
# Deterministic identity
# ============================================================

def stable_seed(*parts) -> int:
    """Reproducible 63-bit seed from arbitrary parts.

    ⚠ Not ``hash()``: CPython salts string hashing per process, so a corpus
    seeded from it would differ between two runs of the same command and
    ``--resume`` would stitch together two different datasets.
    """
    digest = hashlib.sha256('\x1f'.join(str(p) for p in parts).encode('utf-8'))
    return int.from_bytes(digest.digest()[:8], 'big') >> 1


def _draw(sequence_seed: int, *parts) -> float:
    """One per-sequence field's uniform draw, out of its own named stream."""
    return random.Random(stable_seed(sequence_seed, *parts)).random()


class _seeded_global_rng:
    """Seed the process-wide RNGs, then put them back.

    The reused helpers in ``AINR/dataset_gen/dataset.py`` draw from the ``random``
    module and torch's global generator.  Reusing them (rather than forking a
    generator-aware copy) means seeding those globals -- but doing so without
    restoring would make every sequence's randomness depend on render ORDER,
    which destroys ``--resume`` and any multi-worker reproducibility.
    """

    def __init__(self, seed: int):
        self.seed = int(seed) % (2 ** 63)
        self._python = None
        self._numpy = None
        self._torch = None

    def __enter__(self):
        self._python = random.getstate()
        self._numpy = np.random.get_state()
        self._torch = torch.random.get_rng_state()
        random.seed(self.seed)
        np.random.seed(self.seed % (2 ** 32))
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *_exc):
        random.setstate(self._python)
        np.random.set_state(self._numpy)
        torch.random.set_rng_state(self._torch)
        return False


# ============================================================
# Device (loudspeaker + mic) model
# ============================================================

@dataclasses.dataclass(frozen=True)
class DeviceModel:
    """One physical playback/capture device.

    Its identity is fully derived from ``device_id`` plus the corpus seed, so
    the same device id always means the same nonlinearity and the same
    frequency response -- across splits, across runs, across machines.  If it
    were drawn per clip, "device-disjoint validation" would be a phrase with no
    referent.
    """

    device_id: str
    nonlinear: str
    drive: float
    speaker_eq_seed: int
    mic_eq_seed: int
    speaker_hp_hz: float
    speaker_lp_hz: float


def configured_device_ids(cfg: configparser.ConfigParser) -> List[str]:
    """The corpus's whole device population, in configured order."""
    return [d.strip() for d in cfg.get('devices', 'device_ids').split(',')
            if d.strip()]


def configured_nonlinear_models(cfg: configparser.ConfigParser) -> List[str]:
    """The loudspeaker models the population is drawn from, checked."""
    models = [m.strip() for m in cfg.get('devices', 'nonlinear_models').split(',')
              if m.strip()]
    unknown = sorted(set(models) - set(NONLINEAR_MODELS))
    if unknown:
        raise ValueError(f"[devices] unknown nonlinear_models {unknown}; "
                         f"choose from {list(NONLINEAR_MODELS)}")
    if not models:
        raise ValueError("[devices] nonlinear_models is empty")
    return models


def nonlinearity_by_id(cfg: configparser.ConfigParser,
                       corpus_seed: int) -> Dict[str, str]:
    """Which loudspeaker model each configured id gets, STRATIFIED.

    An independent draw per id makes the realised loudspeaker population a
    lottery: with 8 ids over 7 models a third of the corpus seeds contain no
    linear device at all and the rarer models are missing from most of them,
    which moves what the corpus is -- the calibration of the movement axis is
    measured THROUGH this population, so a seed that drew no linear device
    renders a different curve from one that drew three.

    A seeded permutation of the model list therefore fills the ids first, and
    only the remainder is drawn uniformly: every configured model appears at
    least once whenever there are at least as many ids as models, at every
    corpus seed. The per-id determinism is unchanged -- the same (seed, id
    list) always yields the same map -- and every other characteristic still
    comes from the id's own stream.
    """
    models = configured_nonlinear_models(cfg)
    ids = configured_device_ids(cfg)
    rng = random.Random(stable_seed(corpus_seed, 'device-models'))
    assigned = list(models)
    rng.shuffle(assigned)
    assigned = assigned[:len(ids)]
    assigned += [models[rng.randrange(len(models))]
                 for _ in range(len(ids) - len(assigned))]
    return dict(zip(ids, assigned))


def device_for_id(device_id: str, cfg: configparser.ConfigParser,
                  corpus_seed: int, sr: int, *,
                  nonlinear_of: Optional[Dict[str, str]] = None) -> DeviceModel:
    """Derive a device's fixed characteristics from its id.

    ``nonlinear_of`` is the whole configured population's model map, which is a
    property of ``(cfg, corpus_seed)`` and not of the id: a caller deriving
    several devices passes it in rather than re-drawing the same permutation
    once per id.
    """
    if nonlinear_of is None:
        nonlinear_of = nonlinearity_by_id(cfg, corpus_seed)
    if device_id not in nonlinear_of:
        raise ValueError(
            f"device {device_id!r} is not in [devices] device_ids; its "
            f"loudspeaker model is drawn as part of the whole configured "
            f"population, so an id from outside it has none")

    rng = random.Random(stable_seed(corpus_seed, 'device', device_id))
    nyquist = sr / 2.0
    lp_frac = rng.uniform(cfg.getfloat('devices', 'speaker_lp_nyquist_frac_min'),
                          cfg.getfloat('devices', 'speaker_lp_nyquist_frac_max'))
    return DeviceModel(
        device_id=device_id,
        nonlinear=nonlinear_of[device_id],
        drive=rng.uniform(cfg.getfloat('devices', 'drive_min'),
                          cfg.getfloat('devices', 'drive_max')),
        speaker_eq_seed=rng.getrandbits(40),
        mic_eq_seed=rng.getrandbits(40),
        speaker_hp_hz=rng.uniform(cfg.getfloat('devices', 'speaker_hp_hz_min'),
                                  cfg.getfloat('devices', 'speaker_hp_hz_max')),
        # A small loudspeaker rolls off well below Nyquist. ⚠ Expressed as a
        # FRACTION of Nyquist, but a real driver's rolloff sits at an absolute
        # frequency, so the fraction does not carry across rates: left alone
        # at 48 kHz it lands at 13-23 kHz, where the device population's
        # spread at 6 kHz is 0.4 dB instead of 12.5 dB -- i.e. gone. The
        # per-rate values are in config.example.ini's recipe, and the
        # generator refuses a rate whose fractions were never rescaled.
        speaker_lp_hz=min(nyquist * lp_frac, nyquist * 0.98),
    )


def apply_loudspeaker_nonlinearity(x: torch.Tensor, model: str,
                                   drive: float) -> torch.Tensor:
    """Memoryless driver distortion.

    Operates on a peak-normalised copy so ``drive`` means the same amount of
    distortion regardless of how loud the far end happens to be; the output is
    scaled back to the input peak because the echo level is set afterwards by
    the ERL draw, not here.
    """
    peak = x.abs().max()
    if float(peak) < 1e-9 or model == 'linear':
        return x.clone()
    u = (x / peak) * drive

    if model == 'softclip_tanh':
        y = torch.tanh(u)
    elif model == 'arctan':
        y = torch.atan(u * (math.pi / 2.0)) * (2.0 / math.pi)
    elif model == 'hardclip':
        y = u.clamp(-1.0, 1.0)
    elif model == 'sef':
        # Sigmoidal expansion function, the standard smooth saturating model in
        # the nonlinear-AEC literature; eta = 1 after the drive normalisation.
        y = torch.erf(u / math.sqrt(2.0))
    elif model == 'poly_odd':
        # ⚠ Odd memoryless polynomials are monotone only over a bounded range.
        # Clamping first is what keeps this a saturating distortion instead of
        # an expander that inverts slope and produces a nonsense echo path.
        v = u.clamp(-1.0, 1.0)
        y = v - 0.3 * v.pow(3) + 0.1 * v.pow(5)
    elif model == 'diode':
        # Asymmetric saturation (the two rails behave differently), which unlike
        # the odd models generates even-order harmonics.
        y = torch.where(
            u >= 0,
            1.0 - torch.exp(-u.clamp(min=0.0)),
            -(1.0 - torch.exp(2.0 * u.clamp(max=0.0))) / 2.0,
        )
    else:
        raise ValueError(f"unknown nonlinearity model {model!r}")

    return y * (peak / y.abs().max().clamp_min(1e-9))


def _biquad_chain(sr: int, seed: int, n_filters: int, gain_db: float,
                  q_min: float, q_max: float
                  ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """A fixed, level-preserving biquad cascade.

    ``rand_biquad_filter`` is reused for the loudspeaker path, but it cannot be
    used here: it RMS-normalises its own output, so applying it to S, N and D
    separately would give each stem a different gain and break the
    ``mic_preclip == S + N + D`` identity the whole corpus rests on.  This
    builds coefficients once, from the same ``_biquad_coeffs`` cookbook, so one
    identical filter can be applied to every stem.
    """
    rng = random.Random(seed)
    chain = []
    for _ in range(n_filters):
        ftype, freq_lo, freq_hi = BIQUAD_TYPES[rng.randrange(len(BIQUAD_TYPES))]
        chain.append(_biquad_coeffs(
            ftype,
            rng.uniform(freq_lo, min(freq_hi, sr / 2 - 1)),
            sr,
            rng.uniform(q_min, q_max),
            rng.uniform(-gain_db, gain_db),
        ))
    return chain


def _apply_chain(x: torch.Tensor,
                 chain: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    for b, a in chain:
        x = apply_biquad(x, b, a)
    return x


# ============================================================
# Sample-rate offset and codec simulation
# ============================================================

def resample_by_ratio(x: torch.Tensor, ratio: float,
                      n_out: Optional[int] = None) -> torch.Tensor:
    """Read ``x`` at a slightly wrong clock: ``y[n] = x[n * ratio]``.

    ⚠ Catmull-Rom fractional interpolation, NOT a bandlimited resampler, and
    that is deliberate.  A few ppm of drift is a slowly accumulating fractional
    delay -- 5 ppm over 60 s at 16 kHz is 4.8 samples -- and no integer-rate
    resampler can express it: ``resample(16000, 16001)`` is 62.5 ppm, an order
    of magnitude too coarse.  The interpolation error sits far below the
    misalignment being modelled, which is the thing under test.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"ratio must be positive and finite, got {ratio}")
    n_in = x.shape[-1]
    n_out = n_in if n_out is None else int(n_out)
    # float64 positions: at 48 kHz x 60 s, float32 cannot resolve consecutive
    # sample indices, let alone a ppm-scale offset between them.
    pos = torch.arange(n_out, dtype=torch.float64) * float(ratio)
    base = torch.floor(pos).long()
    frac = (pos - base).to(x.dtype)

    def tap(offset: int) -> torch.Tensor:
        return x[(base + offset).clamp(0, n_in - 1)]

    p0, p1, p2, p3 = tap(-1), tap(0), tap(1), tap(2)
    t2 = frac * frac
    t3 = t2 * frac
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * frac
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def simulate_codec(x: torch.Tensor, sr: int, source_sr: int, bits: int) -> torch.Tensor:
    """Band limiting plus mu-law requantisation.

    ⚠ An approximation, on purpose.  A real Opus/AAC round trip would add a
    dependency this package does not have, and the property that matters for
    AEC is that the played signal carries a nonlinear, non-invertible difference
    from the reference the canceller was handed.  Coarse companded quantisation
    produces exactly that.  It does not reproduce any particular codec's
    artefacts, so no result may be reported as "robust to <codec>".
    """
    y = simulate_upsampled_source(x, sr, source_sr)
    peak = y.abs().max()
    if float(peak) < 1e-9:
        return y
    mu = 255.0
    z = y / peak
    compressed = torch.sign(z) * torch.log1p(mu * z.abs()) / math.log1p(mu)
    levels = float(2 ** int(bits)) / 2.0 - 1.0
    compressed = torch.round(compressed * levels) / levels
    expanded = torch.sign(compressed) * ((1.0 + mu) ** compressed.abs() - 1.0) / mu
    return expanded * peak


# ============================================================
# Mic AGC
# ============================================================

def apply_agc(x: torch.Tensor, sr: int, target_dbfs: float,
              attack_sec: float, release_sec: float,
              max_gain_db: float) -> torch.Tensor:
    """Slow envelope-following gain, the kind a capture chain applies.

    ⚠ Both time constants arrive in SECONDS and go through
    ``alpha_from_tau(..., hop_len=1, sr)``.  Writing the coefficient directly
    would silently make the AGC three times slower on the 48 kHz grid, and the
    corpus would stop matching the one it is supposed to be a variant of.
    """
    a_attack = alpha_from_tau(attack_sec, 1, sr)
    a_release = alpha_from_tau(release_sec, 1, sr)

    # Frame-wise, so the smoother is not a million-iteration Python loop.  The
    # frame is 1 ms, far shorter than either time constant, so the envelope is
    # indistinguishable from the per-sample recursion at these taus.
    frame = max(1, int(sr * 0.001))
    n_frames = x.shape[-1] // frame
    if n_frames < 2:
        return x.clone()
    peaks = x[:n_frames * frame].abs().reshape(n_frames, frame).max(dim=1).values.tolist()

    a_att_f = a_attack ** frame
    a_rel_f = a_release ** frame
    state = peaks[0]
    smoothed = []
    for value in peaks:
        alpha = a_att_f if value > state else a_rel_f
        state = alpha * state + (1.0 - alpha) * value
        smoothed.append(state)

    target = 10.0 ** (target_dbfs / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = (target / torch.tensor(smoothed, dtype=x.dtype).clamp_min(1e-6))
    gain = gain.clamp(1.0 / max_gain, max_gain).repeat_interleave(frame)
    if gain.shape[-1] < x.shape[-1]:
        gain = F.pad(gain, (0, x.shape[-1] - gain.shape[-1]), value=float(gain[-1]))
    return x * gain[:x.shape[-1]]


# ============================================================
# Talker activity
# ============================================================

def activity_runs(n_samples: int, sr: int, talk_sec: float, gap_sec: float,
                  rng: random.Random, start_active: Optional[bool] = None
                  ) -> List[Tuple[int, int]]:
    """Alternating talk/silence, run lengths drawn from exponentials.

    Means are given in SECONDS, so one config produces the same conversational
    rhythm at 16 and 48 kHz.
    """
    if talk_sec <= 0 or gap_sec <= 0:
        raise ValueError("talk_sec and gap_sec must be positive")
    runs = []
    position = 0
    active = rng.random() < 0.5 if start_active is None else start_active
    while position < n_samples:
        end = _draw_run_end(rng, position, talk_sec if active else gap_sec,
                            sr, n_samples)
        if active:
            runs.append((position, end))
        position = end
        active = not active
    return runs


def _draw_run_end(rng: random.Random, position: int, mean_sec: float,
                  sr: int, n_samples: int) -> int:
    """Where a run starting at ``position`` ends, drawn from an exponential.

    Clamped so one unlucky draw cannot swallow a whole 60 s sequence in a
    single run, and floored so no run is too short to carry an utterance.
    Shared by both schedulers so the two cannot drift into different rhythms.
    """
    seconds = min(max(rng.expovariate(1.0 / mean_sec), 0.15), mean_sec * 4.0)
    return min(position + max(int(sr * seconds), int(sr * 0.15)), n_samples)


def _contiguous_runs(n_samples: int, sr: int, talk_sec: float,
                     rng: random.Random) -> List[Tuple[int, int]]:
    """Back-to-back talk runs covering the whole span, with no silent gap.

    The degenerate case of ``activity_runs`` where the talker never stops.
    Split into utterance-sized runs rather than returned as ONE long run so
    that ``_render_talker`` draws a fresh pool file per run: one whole-sequence
    run would be a single speaker repeating for 20-30 s.  Keeping the far end
    ACTIVE across a run that outlasts its file is the other half of the job and
    belongs to that caller's ``loop=True``.
    """
    if talk_sec <= 0:
        raise ValueError("talk_sec must be positive")
    runs = []
    position = 0
    while position < n_samples:
        end = _draw_run_end(rng, position, talk_sec, sr, n_samples)
        runs.append((position, end))
        position = end
    return runs


# ============================================================
# Planning
# ============================================================

@dataclasses.dataclass(frozen=True)
class SequencePlan:
    """What a sequence will be, decided before any audio is touched.

    Drawn from a dedicated planning RNG so ``--hours`` resolves to an exact
    sequence list up front.  That is what makes ``--resume`` exact and lets N
    workers render out of order without changing the corpus.
    """

    sequence_id: int
    n_chunks: int
    scenario: str
    seed: int
    # Layered planner fields. ``None`` keeps old direct callers source- and
    # behaviour-compatible: their single scenario is resolved below into one
    # talk mode / echo mode / impairment tuple.
    talk_mode: Optional[str] = None
    echo_mode: str = 'normal'
    impairments: Tuple[str, ...] = ()
    acoustic_tails: Tuple[str, ...] = ()


@dataclasses.dataclass
class RenderedSequence:
    stems: torch.Tensor              # (5, T) float32, channel order = STEM_ORDER; PERSISTED
    chunk_meta: List[dict]
    chunk_samples: int
    linear_aec_contract: Dict = dataclasses.field(default_factory=dict)
    audit: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    # 'echo', 'noise' and 'mic_preclip' -- computed on every render (see this
    # module's docstring), NEVER written to WAV/shard. gen_aec_dataset.py's
    # WAV writer only ever touches ``.stems``; this field exists purely so
    # tests/test_aec_dataset.py can still verify the corpus's central
    # invariants against a full, un-trimmed render.


def plan_sequences(cfg: configparser.ConfigParser, hours: float, seed: int,
                   split: str, start_id: int = 0) -> List[SequencePlan]:
    """Resolve ``--hours`` into a fixed list of sequences."""
    if hours <= 0 or not math.isfinite(hours):
        raise ValueError(f"--hours must be positive and finite, got {hours}")

    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    seq_min = cfg.getfloat('sequence', 'seq_sec_min')
    seq_max = cfg.getfloat('sequence', 'seq_sec_max')
    if not 0 < chunk_sec <= seq_min <= seq_max:
        raise ValueError(
            f"[sequence] requires 0 < chunk_sec <= seq_sec_min <= seq_sec_max, "
            f"got chunk={chunk_sec}, min={seq_min}, max={seq_max}"
        )

    # A worker renders every sequence out of [echo_path] and [path_motion],
    # and a config written against an older trajectory model carries neither
    # section. Their PRESENCE is therefore established before the planner
    # options below: a config missing [path_motion] outright would otherwise
    # be told about three missing [impairments] probabilities and only learn
    # what it really lacks on the next run. Their RANGES are checked further
    # down, where the planner branch is known.
    _refuse_retired_path_motion_options(cfg)
    _validate_renderer_sections(cfg)

    layer_sections = ('talk_modes', 'echo_modes', 'impairments',
                      'acoustic_tails', 'complex_cases')
    present_layers = tuple(name for name in layer_sections
                           if cfg.has_section(name))
    layered = bool(present_layers)
    if layered:
        # Every probability below reads through a fallback, and configparser
        # returns that fallback for an ABSENT SECTION just as readily as for an
        # absent key. A layered config missing [impairments] would therefore
        # generate a full, plausible, impairment-free corpus and say nothing.
        absent = [name for name in layer_sections
                  if not cfg.has_section(name)]
        if absent:
            raise ValueError(
                "a layered config must carry all five planner sections; "
                f"found {', '.join('[' + name + ']' for name in present_layers)} "
                "but not "
                f"{', '.join('[' + name + ']' for name in absent)}; without "
                "the section every probability in it silently reads as 0")
        required_options = {
            'talk_modes': tuple(f'p_{name}' for name in TALK_MODES),
            'echo_modes': ('p_ref_dropout', 'p_far_active_no_echo'),
            'impairments': tuple(f'p_{name}' for name in IMPAIRMENTS),
            'acoustic_tails': tuple(
                key for name in ACOUSTIC_TAILS
                for key in (f'p_{name}',
                            f'{ACOUSTIC_TAIL_RANGE_KEYS[name]}_min',
                            f'{ACOUSTIC_TAIL_RANGE_KEYS[name]}_max')
            ) + (QUIET_FAR_ERL_CAP_KEY,),
            'complex_cases': ('p_dt_stress_combo', 'p_dt_acoustic_combo'),
        }
        _require_options(
            cfg, required_options,
            "a layered config must explicitly set every planner option")
        talk_weights = _named_weights(cfg, 'talk_modes', TALK_MODES)
        talk_names = list(talk_weights)
        talk_probabilities = [talk_weights[name] for name in talk_names]
        p_ref_dropout = _probability(cfg, 'echo_modes', 'p_ref_dropout')
        p_far_active_no_echo = _probability(
            cfg, 'echo_modes', 'p_far_active_no_echo')
        if p_ref_dropout + p_far_active_no_echo > 1.0:
            raise ValueError(
                "[echo_modes] p_ref_dropout + p_far_active_no_echo must be "
                f"<= 1, got {p_ref_dropout + p_far_active_no_echo:g}")
        impairment_p = {
            name: _probability(cfg, 'impairments', f'p_{name}')
            for name in IMPAIRMENTS
        }
        motion_total = sum(impairment_p[name] for name in PATH_MOTION_MODES)
        if motion_total > 1.0:
            raise ValueError(
                "[impairments] " + " + ".join(f'p_{n}' for n in PATH_MOTION_MODES)
                + " must be <= 1 (they are one mutually exclusive draw), got "
                f"{motion_total:g}")
        acoustic_tail_p = {
            name: _probability(cfg, 'acoustic_tails', f'p_{name}')
            for name in ACOUSTIC_TAILS
        }
        p_dt_stress = _probability(
            cfg, 'complex_cases', 'p_dt_stress_combo')
        p_dt_acoustic = _probability(
            cfg, 'complex_cases', 'p_dt_acoustic_combo')
    else:
        # Compatibility for an older copied config or a focused test config.
        # It deliberately preserves the old mutually-exclusive semantics;
        # shipped configs use the layered sections above.
        # ⚠ Without CHUNK_EVENT_LABELS: those describe WHEN inside a sequence
        # something happened, so they are not sequences anybody can plan. A
        # weight on one would resolve to a motionless duplex_random sequence
        # that still labels every chunk with the event.
        weights = _named_weights(
            cfg, 'scenarios',
            tuple(name for name in SCENARIOS if name not in CHUNK_EVENT_LABELS))
        names = list(weights)
        probabilities = [weights[name] for name in names]

    # Both branches plan sequences a worker renders out of [echo_path] and
    # [path_motion], so both branches check those ranges here. Validating only
    # the layered branch leaves the compatibility path with the failure mode
    # the checks exist to remove -- a whole corpus planned, then a worker dying
    # on the first sequence that draws motion or a delay step -- and that path
    # is exactly where an older copied config lands. The tail ranges are the
    # exception: they live in a layer section, so a compatibility config has no
    # tails to check and cannot draw one.
    _validate_echo_path_ranges(cfg)
    if layered:
        _validate_acoustic_tail_ranges(cfg)
    _validate_path_motion_ranges(cfg)

    # Depends on (seed, split) only -- not on --hours -- so extending a corpus
    # keeps every sequence it already had, byte for byte.
    rng = random.Random(stable_seed(seed, 'plan', split))

    chunks_min = max(1, int(seq_min / chunk_sec))
    chunks_max = max(chunks_min, int(seq_max / chunk_sec))

    plans: List[SequencePlan] = []
    total_sec = 0.0
    target_sec = hours * 3600.0
    sequence_id = start_id
    while total_sec < target_sec:
        n_chunks = rng.randint(chunks_min, chunks_max)
        sequence_seed = stable_seed(seed, 'sequence', split, sequence_id)
        if layered:
            # Each field owns a seed. Changing one impairment probability no
            # longer silently reshuffles talk modes or every impairment after
            # it, and worker/render order remains irrelevant.
            talk_mode = random.Random(stable_seed(
                sequence_seed, 'talk_mode')).choices(
                    talk_names, weights=talk_probabilities, k=1)[0]
            has_far = talk_mode != 'near_only'
            echo_mode = 'normal'
            if has_far:
                draw = _draw(sequence_seed, 'echo_mode')
                if draw < p_far_active_no_echo:
                    echo_mode = 'far_active_no_echo'
                elif draw < p_far_active_no_echo + p_ref_dropout:
                    echo_mode = 'ref_dropout'

            impairments = set()
            for name, probability in impairment_p.items():
                if name in PATH_MOTION_MODES:
                    continue
                if _draw(sequence_seed, 'impairment', name) < probability:
                    impairments.add(name)
            # One draw for the whole motion axis, from its own seed: see
            # PATH_MOTION_MODES for why the three cannot co-occur.
            motion = _draw_path_motion(sequence_seed, impairment_p)
            if motion != STATIC_PATH:
                impairments.add(motion)

            # Acoustic tails own independent seeds just like impairments. They
            # are only meaningful when a real echo path exists: no-echo is a
            # separate hard negative, and near-only has no far-end operating
            # point to widen.
            acoustic_tails = set()
            if has_far and echo_mode == 'normal':
                for name, probability in acoustic_tail_p.items():
                    if _draw(sequence_seed, 'acoustic_tail', name) < probability:
                        acoustic_tails.add(name)

            # Echo-path impairments have no signal to act on in near-only or
            # far-active/no-echo sequences. Capture clipping/AGC remains valid
            # in both and is therefore intentionally retained.
            if not has_far or echo_mode == 'far_active_no_echo':
                impairments.difference_update(ECHO_PATH_IMPAIRMENTS)

            # Pure independent draws make the exact DT + nonlinear/clipped
            # intersection too rare to teach in a 200 h campaign.
            # This conditional draw creates a measurable tail without turning
            # every ordinary DT example into an adversarial one.
            if (talk_mode == 'double_talk'
                    and echo_mode == 'normal'
                    and _draw(sequence_seed, 'dt_stress_combo') < p_dt_stress):
                impairments.update(DT_STRESS_IMPAIRMENTS)

            if (talk_mode == 'double_talk'
                    and echo_mode == 'normal'
                    and _draw(sequence_seed,
                              'dt_acoustic_combo') < p_dt_acoustic):
                acoustic_tails.update(DT_ACOUSTIC_TAILS)

            plans.append(SequencePlan(
                sequence_id=sequence_id,
                n_chunks=n_chunks,
                scenario=talk_mode,
                seed=sequence_seed,
                talk_mode=talk_mode,
                echo_mode=echo_mode,
                impairments=tuple(sorted(impairments)),
                acoustic_tails=tuple(sorted(acoustic_tails)),
            ))
        else:
            plans.append(SequencePlan(
                sequence_id=sequence_id,
                n_chunks=n_chunks,
                scenario=rng.choices(names, weights=probabilities, k=1)[0],
                seed=sequence_seed,
            ))
        total_sec += n_chunks * chunk_sec
        sequence_id += 1
    return plans


def _draw_path_motion(sequence_seed: int,
                      impairment_p: Dict[str, float]) -> str:
    """Which of the mutually exclusive motion modes this sequence gets."""
    draw = _draw(sequence_seed, 'path_motion')
    cumulative = 0.0
    for name in PATH_MOTION_MODES:
        cumulative += impairment_p[name]
        if draw < cumulative:
            return name
    return STATIC_PATH


def path_motion_mode(impairments: Sequence[str]) -> str:
    """The motion mode an impairment set carries, or ``STATIC_PATH``."""
    for name in PATH_MOTION_MODES:
        if name in impairments:
            return name
    return STATIC_PATH


def _named_weights(cfg: configparser.ConfigParser, section: str,
                   names: Sequence[str]) -> Dict[str, float]:
    weights = {}
    for name in names:
        value = cfg.getfloat(section, f'p_{name}', fallback=0.0)
        if value < 0 or not math.isfinite(value):
            raise ValueError(
                f"[{section}] p_{name} must be finite and >= 0, got {value}")
        if value > 0:
            weights[name] = value
    if not weights:
        raise ValueError(f"[{section}] every weight is zero; nothing to generate")
    missing = sorted(set(names) - set(weights))
    if missing:
        # Not fatal -- an ablation corpus is a legitimate thing to want -- but a
        # silently absent mode is a hole nobody finds until evaluation.
        print(f"  ⚠ zero-probability [{section}] entries, absent from this "
              f"corpus: {missing}")
    return weights


def _probability(cfg: configparser.ConfigParser, section: str, key: str) -> float:
    # No fallback: plan_sequences has already refused a layered config that
    # omits any of these keys, so a default here could only mask that check.
    value = cfg.getfloat(section, key)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"[{section}] {key} must be in [0, 1], got {value}")
    return value


def _finite_range(cfg: configparser.ConfigParser, section: str,
                  low_key: str, high_key: str) -> Tuple[float, float]:
    low = cfg.getfloat(section, low_key)
    high = cfg.getfloat(section, high_key)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(
            f"[{section}] requires finite {low_key} <= {high_key}, got "
            f"{low:g} .. {high:g}")
    return low, high


def _acoustic_tail_range(cfg: configparser.ConfigParser,
                         name: str) -> Tuple[float, float]:
    stem = ACOUSTIC_TAIL_RANGE_KEYS[name]
    return _finite_range(cfg, 'acoustic_tails', f'{stem}_min', f'{stem}_max')


def _validate_echo_path_ranges(cfg: configparser.ConfigParser) -> None:
    """Refuse a delay model the renderer cannot place an echo inside.

    Separate from the acoustic-tail checks below because these keys are read
    for EVERY far-capable sequence, including one planned by the compatibility
    branch, which has no [acoustic_tails] section to check at all.
    """
    floor = {}
    for key, spec in ECHO_PATH_OPTION_SPECS.items():
        if spec.check == 'range':
            floor[key] = _finite_range(cfg, 'echo_path', key, spec.pair)[0]
        elif spec.check == 'whole_range':
            low = _whole(cfg, 'echo_path', key)
            high = _whole(cfg, 'echo_path', spec.pair)
            floor[key] = low
            if not 0 <= low <= high:
                raise ValueError(
                    f"[echo_path] requires 0 <= {key} <= {spec.pair}, "
                    f"got {low} .. {high}")
        else:
            _check_option(cfg, 'echo_path', spec, key, spec.pair)
    # A negative delay step is drawn from the headroom above this floor, so it
    # decides where the echo may be re-timed TO; a config with a zero or
    # inverted bulk-delay range would put it at 0 ms, inside the frozen matched
    # filter's unresolvable first bin.
    if floor['bulk_delay_ms_min'] <= 0:
        raise ValueError(
            "[echo_path] bulk_delay_ms_min must be > 0, got "
            f"{floor['bulk_delay_ms_min']:g}")
    if floor['delay_step_ms_min'] <= 0:
        raise ValueError(
            "[echo_path] delay_step_ms_min must be > 0, got "
            f"{floor['delay_step_ms_min']:g}")
    for key in ('jitter_steps_min', 'jitter_ms_min'):
        if floor[key] <= 0:
            raise ValueError(
                f"[echo_path] {key} must be > 0 so a delay_jitter label "
                f"always carries a real timing change, got {floor[key]:g}")
    jitter_max = cfg.getfloat('echo_path', 'jitter_ms_max')
    if floor['delay_step_ms_min'] <= jitter_max:
        raise ValueError(
            "[echo_path] delay_step_ms_min must be > jitter_ms_max so a "
            "delay_step cannot be labelled as delay_jitter, got "
            f"{floor['delay_step_ms_min']:g} <= {jitter_max:g}")


def _validate_acoustic_tail_ranges(cfg: configparser.ConfigParser) -> None:
    """Keep each tail disjoint from the ordinary range and in AEC reach."""
    long_min, long_max = _acoustic_tail_range(cfg, 'long_delay')
    base_delay_max = cfg.getfloat('echo_path', 'bulk_delay_ms_max')
    if long_min < base_delay_max:
        raise ValueError(
            "[acoustic_tails] long_delay_ms_min must be >= "
            f"[echo_path] bulk_delay_ms_max ({base_delay_max:g}), got "
            f"{long_min:g}")
    reliable_reach = MATCHED_REACH_MS[max(MATCHED_REACH_MS)]
    jitter_headroom = (_whole(cfg, 'echo_path', 'jitter_steps_max')
                       * cfg.getfloat('echo_path', 'jitter_ms_max'))
    rir_headroom = cfg.getfloat('rir', 'pre_delay_keep_ms')
    # The delay step is an independent impairment, so it stacks on top of the
    # long-delay tail and the jitter in the same sequence.
    _step_min, step_headroom = _finite_range(
        cfg, 'echo_path', 'delay_step_ms_min', 'delay_step_ms_max')
    worst_case_delay = (long_max + jitter_headroom + step_headroom
                        + rir_headroom)
    if worst_case_delay > reliable_reach:
        raise ValueError(
            f"[acoustic_tails] worst-case long delay {worst_case_delay:g} ms "
            f"(tail {long_max:g} + jitter {jitter_headroom:g} + delay step "
            f"{step_headroom:g} + RIR {rir_headroom:g}) exceeds the frozen "
            f"n=5 matched-filter reliable reach {reliable_reach:g} ms")

    _quiet_min, quiet_max = _acoustic_tail_range(cfg, 'quiet_far')
    base_far_min = cfg.getfloat('levels', 'far_level_dbfs_min')
    if quiet_max > base_far_min:
        raise ValueError(
            "[acoustic_tails] quiet_far_dbfs_max must be <= "
            f"[levels] far_level_dbfs_min ({base_far_min:g}), got "
            f"{quiet_max:g}")
    base_erl_min = cfg.getfloat('levels', 'erl_db_min')
    base_erl_max = cfg.getfloat('levels', 'erl_db_max')
    quiet_erl_cap = cfg.getfloat('acoustic_tails', QUIET_FAR_ERL_CAP_KEY)
    if not (math.isfinite(quiet_erl_cap)
            and base_erl_min <= quiet_erl_cap <= base_erl_max):
        raise ValueError(
            f"[acoustic_tails] {QUIET_FAR_ERL_CAP_KEY} must lie within "
            f"[levels] erl_db_min..erl_db_max ({base_erl_min:g}.."
            f"{base_erl_max:g}), got {quiet_erl_cap:g}")

    _strong_min, strong_max = _acoustic_tail_range(cfg, 'strong_echo')
    if strong_max > base_erl_min:
        raise ValueError(
            "[acoustic_tails] strong_echo_erl_db_max must be <= "
            f"[levels] erl_db_min ({base_erl_min:g}), got {strong_max:g}")


def _positive(cfg: configparser.ConfigParser, section: str, key: str,
              *, allow_zero: bool = False) -> float:
    value = cfg.getfloat(section, key)
    floor_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not floor_ok:
        raise ValueError(
            f"[{section}] {key} must be finite and "
            f"{'>= 0' if allow_zero else '> 0'}, got {value:g}")
    return value


def _positive_range(cfg: configparser.ConfigParser, section: str,
                    min_key: str, max_key: str, *,
                    allow_zero: bool = False) -> float:
    """A finite ``min_key <= max_key`` pair whose floor clears zero."""
    _finite_range(cfg, section, min_key, max_key)
    return _positive(cfg, section, min_key, allow_zero=allow_zero)


def _whole(cfg: configparser.ConfigParser, section: str, key: str) -> int:
    """A count, refused with the same wording as every other bad value.

    ``cfg.getint`` raises ``int()``'s own message, which names neither the
    section nor the key -- so the one non-integer key in a section reports
    differently from every other kind of mistake in it.
    """
    try:
        return cfg.getint(section, key)
    except ValueError:
        raise ValueError(
            f"[{section}] {key} must be a whole number, got "
            f"{cfg.get(section, key)!r}") from None


def _waypoint_range(cfg: configparser.ConfigParser) -> Tuple[int, int]:
    """How many positions a trajectory may visit, (fewest, most).

    Read by the plan-time validator and by the renderer, so a config that
    cannot describe a trajectory is refused with the same words wherever it is
    first noticed.
    """
    low = _whole(cfg, 'path_motion', 'waypoints_min')
    high = _whole(cfg, 'path_motion', 'waypoints_max')
    if not 2 <= low <= high:
        raise ValueError(
            "[path_motion] requires 2 <= waypoints_min <= waypoints_max, got "
            f"{low} .. {high}")
    return low, high


def _require_options(cfg: configparser.ConfigParser,
                     required: Dict[str, Sequence[str]], prefix: str) -> None:
    """Refuse a config missing any of ``required``, naming every one of them.

    All of them, not the first: a config written against an older tree is
    usually missing several, and one name per run is one run per name.
    """
    missing = [f'[{section}] {option}'
               for section, options in required.items()
               for option in options
               if not cfg.has_option(section, option)]
    if missing:
        raise ValueError(f"{prefix}; missing " + ', '.join(missing))


def _validate_renderer_sections(cfg: configparser.ConfigParser) -> None:
    """Refuse a config missing anything the RENDER workers read.

    [echo_path] and [path_motion] carry no planner probability, so nothing
    else in the plan would notice: a missing key survives the plan and the
    manifest and kills the first worker that draws the impairment reading it,
    hours into a run. Every key each section is read for is listed, and the
    SECTION itself is named first -- an older copied config that has neither is
    the likeliest way to arrive here, and configparser's own NoSectionError
    says nothing about what to add.
    """
    required = {'echo_path': ECHO_PATH_OPTIONS,
                'path_motion': PATH_MOTION_OPTIONS}
    read_for = {
        'echo_path': 'the bulk delay, the one-shot crossfade, the delay step '
                     'and the jitter walk',
        'path_motion': 'the trajectory model',
    }
    for section in sorted(required):
        if not cfg.has_section(section):
            raise ValueError(
                f"section [{section}] is required: every render worker reads "
                f"it for {read_for[section]}. Copy it from "
                f"config.example.ini")
    _require_options(cfg, required,
                     "every option the render workers read must be set")


def _refuse_retired_path_motion_options(cfg: configparser.ConfigParser) -> None:
    """Name a retired trajectory knob before anything else is checked.

    A config carrying one is a config written against an older trajectory
    model, and it is missing the key that replaced it -- so the presence check
    would refuse it first, for the missing key, and never mention the knob the
    author actually has to change. Refusing by name first is what makes the
    message the one that fixes the file.
    """
    for mode in sorted(DRIFT_MODES):
        for replacement, spec in DRIFT_MODE_OPTION_SPECS.items():
            if spec.retired is None:
                continue
            if cfg.has_option('path_motion', f'{mode}_{spec.retired}'):
                raise ValueError(
                    f"[path_motion] {mode}_{spec.retired} is not part of the "
                    f"trajectory model; the calibrated knob is "
                    f"{mode}_{replacement} and the mixture depth is solved "
                    f"per room from the positions it mixes")


def _check_option(cfg: configparser.ConfigParser, section: str,
                  spec: _OptionSpec, key: str,
                  pair: Optional[str]) -> None:
    """Hold one option to what its spec says its value may be."""
    if spec.check == 'positive_range':
        _positive_range(cfg, section, key, pair)
    elif spec.check == 'probability':
        _probability(cfg, section, key)
    elif spec.check == 'open_unit':
        # The trajectory's calibrated quantity: how correlated the path stays
        # with itself between the anchor mixture and a corner. 1.0 is a path
        # that never moves and 0 is unreachable by any mixture of one room's
        # positions, so both ends are refused rather than solved for.
        value = _probability(cfg, section, key)
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"[{section}] {key} must be inside (0, 1), got {value:g}: 1 "
                f"is a path that never moves and 0 is below what any mixture "
                f"of one room's positions can reach")
    elif spec.check == 'open_closed_unit':
        value = _probability(cfg, section, key)
        if value <= 0.0:
            raise ValueError(
                f"[{section}] {key} must be inside (0, 1], got {value:g}")
    elif spec.check == 'positive':
        _positive(cfg, section, key)
    elif spec.check == 'non_negative':
        _positive(cfg, section, key, allow_zero=True)


def _validate_path_motion_ranges(cfg: configparser.ConfigParser) -> None:
    """Refuse a trajectory model that cannot be rendered, at plan time.

    Every value here is read inside a render worker, i.e. minutes to hours
    after the plan and the manifest were written. Checking them beside the
    acoustic-tail ranges is what keeps "a config error fails in under a
    second" true for the motion axis as well.
    """
    _waypoint_range(cfg)
    for mode in sorted(DRIFT_MODES):
        for stem, spec in DRIFT_MODE_OPTION_SPECS.items():
            _check_option(cfg, 'path_motion', spec, f'{mode}_{stem}',
                          f'{mode}_{spec.pair}' if spec.pair else None)
    for stem, spec in SHARED_PATH_MOTION_SPECS.items():
        _check_option(cfg, 'path_motion', spec, stem, spec.pair)


def resolve_sequence_plan(plan: SequencePlan) -> Tuple[str, str, Tuple[str, ...]]:
    """Return ``(talk_mode, echo_mode, impairments)`` for new and old plans."""
    if plan.talk_mode is not None:
        talk_mode = plan.talk_mode
        echo_mode = plan.echo_mode
        impairments = tuple(plan.impairments)
    else:
        legacy = plan.scenario
        talk_mode = legacy if legacy in TALK_MODES else 'duplex_random'
        echo_mode = (legacy if legacy in ('ref_dropout', 'far_active_no_echo')
                     else 'normal')
        impairments = (legacy,) if legacy in IMPAIRMENTS else ()

    if talk_mode not in TALK_MODES:
        raise ValueError(f"unknown talk_mode {talk_mode!r}")
    if echo_mode not in ECHO_MODES:
        raise ValueError(f"unknown echo_mode {echo_mode!r}")
    unknown = sorted(set(impairments) - set(IMPAIRMENTS))
    if unknown:
        raise ValueError(f"unknown impairments: {unknown}")
    motion = sorted(set(impairments) & set(PATH_MOTION_MODES))
    if len(motion) > 1:
        raise ValueError(
            f"a sequence may carry at most one path-motion mode, got {motion}")
    if talk_mode == 'near_only' and echo_mode != 'normal':
        raise ValueError(f"near_only cannot use echo_mode={echo_mode}")
    # A near-only sequence radiates nothing, so there is no echo path for these
    # to act on: accepting them would render a sequence whose metadata claims a
    # moving/re-timed echo path over a microphone that never carried one.
    if talk_mode == 'near_only' and (set(impairments) & ECHO_PATH_IMPAIRMENTS):
        raise ValueError("near_only cannot carry echo-path impairments")
    if echo_mode == 'far_active_no_echo' and (
            set(impairments) & ECHO_PATH_IMPAIRMENTS):
        raise ValueError(
            "far_active_no_echo cannot carry echo-path impairments")
    return talk_mode, echo_mode, tuple(sorted(set(impairments)))


def resolve_acoustic_tails(plan: SequencePlan) -> Tuple[str, ...]:
    """Validate and return the independent acoustic-tail axis."""
    tails = tuple(plan.acoustic_tails)
    unknown = sorted(set(tails) - set(ACOUSTIC_TAILS))
    if unknown:
        raise ValueError(f"unknown acoustic tails: {unknown}")
    talk_mode, echo_mode, _ = resolve_sequence_plan(plan)
    if tails and (talk_mode == 'near_only' or echo_mode != 'normal'):
        raise ValueError(
            f"acoustic tails require a real far/echo path, got "
            f"talk_mode={talk_mode}, echo_mode={echo_mode}")
    return tuple(sorted(set(tails)))


# ============================================================
# Chunk geometry
# ============================================================

# Keys that are correct at one product rate and wrong at the other, but whose
# right value is editorial -- the generator cannot pick a device population or
# a codec ladder for you. What it CAN see is that a key was never revisited:
# "still exactly the OTHER rate's shipped default" is not a choice anyone
# makes on purpose. Everything else is accepted silently, so deliberate tuning
# is never blocked.
#
# This exists because both of these degrade SILENTLY. A 48 kHz run that kept
# the 16 kHz loudspeaker fractions band-limits at 13-23 kHz, i.e. not at all
# (the device population's spread at 6 kHz collapses from -6.5..-1.4 dB to
# -0.18..-0.02 dB, and device-disjoint validation loses an axis); one that
# kept the 16 kHz codec ladder turns a 1.3-2x resample into a 4-6x one. Both
# produce a full, finite, plausible corpus.
RATE_DEPENDENT_KEYS = {
    ('devices', 'speaker_lp_nyquist_frac_min'): {16000: '0.55', 48000: '0.1833'},
    ('devices', 'speaker_lp_nyquist_frac_max'): {16000: '0.95', 48000: '0.3167'},
    ('codec', 'source_sr_values'): {
        16000: '8000, 12000',
        48000: '8000, 12000, 16000, 24000, 32000',
    },
}


def _normalised(value: str) -> str:
    return ','.join(part.strip() for part in value.split(','))


def check_rate_dependent_values(cfg: configparser.ConfigParser) -> None:
    """Refuse a config whose rate-dependent keys were never rescaled.

    Only fires on an exact match with another rate's shipped default, so it
    cannot stand in the way of a deliberately different device population.
    """
    sample_rate = cfg.getint('signal', 'sr')
    stale = []
    for (section, key), by_rate in sorted(RATE_DEPENDENT_KEYS.items()):
        if sample_rate not in by_rate or not cfg.has_option(section, key):
            continue
        current = _normalised(cfg.get(section, key))
        if current == _normalised(by_rate[sample_rate]):
            continue
        for other_rate, other_value in by_rate.items():
            if other_rate != sample_rate and current == _normalised(other_value):
                stale.append(
                    '[%s] %s = %s is the %d Hz value; at %d Hz it should be %s'
                    % (section, key, current, other_rate, sample_rate,
                       by_rate[sample_rate])
                )
    if stale:
        raise ValueError(
            'this config is set to sr=%d but still carries another rate\'s '
            'values:\n  %s\nThese do not fail loudly during generation -- '
            'they quietly change what the corpus contains -- so they are '
            'refused here. See the recipe at the top of config.example.ini. '
            'Any OTHER value is accepted; only an exact match with the other '
            'rate\'s shipped default is treated as "never revisited".'
            % (sample_rate, '\n  '.join(stale))
        )


def chunk_samples_from_config(cfg: configparser.ConfigParser,
                              hop_size: int) -> int:
    """`[sequence] chunk_sec` in samples, or raise naming what has to change.

    The linear-AEC frontend consumes whole hops, so a chunk that is not an
    integer number of them cannot be materialized. The hop is frozen per
    sample rate (linear_aec.FROZEN_FRAME_HOP_BY_SR), which makes `chunk_sec`
    the only adjustable side of

        round(chunk_sec * sr) % hop == 0

    and makes the constraint rate-dependent: the same `chunk_sec` that is
    exact at one rate need not be at another. Called both by the renderer
    (below) and by gen_aec_dataset.py's config preflight, so a CLI run and an
    in-process one fail identically.
    """
    sample_rate = cfg.getint('signal', 'sr')
    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    chunk_samples = int(round(chunk_sec * sample_rate))
    if chunk_samples <= 0:
        raise ValueError(
            f"[sequence] chunk_sec = {chunk_sec:g} is too small for "
            f"sr={sample_rate}: it rounds to {chunk_samples} samples"
        )
    if chunk_samples % hop_size:
        # Whole seconds are exact only in multiples of this many, from
        # n * sr = 0 (mod hop)  <=>  n = 0 (mod hop / gcd(sr, hop)).
        second_step = hop_size // math.gcd(sample_rate, hop_size)
        suggestion = max(second_step,
                         round(chunk_sec / second_step) * second_step)
        raise ValueError(
            f"training chunk geometry must be divisible by the linear AEC "
            f"hop: [sequence] chunk_sec = {chunk_sec:g} at sr={sample_rate} "
            f"is {chunk_samples} samples, which is not a whole number of "
            f"hop={hop_size} hops. The hop is frozen per sample rate, so "
            f"chunk_sec is the side that has to change: it must satisfy "
            f"round(chunk_sec * sr) % hop == 0. Among whole seconds this "
            f"rate admits only multiples of {second_step} s -- "
            f"e.g. chunk_sec = {suggestion:g}."
        )
    return chunk_samples


# ============================================================
# Continuous echo-path motion
# ============================================================
#
# WHAT THIS MODELS AND WHY IT IS NOT A CROSSFADE
# ----------------------------------------------
# A device's echo path is never frozen.  Measured on paired static/movement
# far-end captures, its correlation with itself seconds earlier falls away even
# on a device nobody touches, and a moving one has largely forgotten its own
# path after 8 s.  The numbers live in ONE place --
# ``path_drift_metrics.CALIBRATION_TARGETS``, which is also what the
# calibration test asserts against -- so that a re-calibration cannot leave a
# stale copy behind here.  A single 20 ms crossfade between two positions
# reproduces none of that curve: it leaves the path EXACTLY constant on both
# sides of one instant, which is what makes a filter that only converges once
# look adequate.
#
# The trajectory is a weight schedule over K positions in ONE room,
#
#     H(t) = sum_k w_k(t) * h_k,      sum_k w_k(t) = 1,
#
# whose corners (`keyframes`) walk between per-position target mixtures and
# whose transitions are raised cosines, so neither the weights nor their first
# derivative step.  Two knobs set the correlation curve: how far a keyframe
# pulls the mixture toward a single position
# (`*_position_correlation`, solved into a per-room mixture depth by
# `solve_mixture_depth`) and how long a transition takes
# (`*_segment_sec_*`).  Level is a separate dB-domain random walk, because a
# real path's gain and its shape do not change together.
DriftKeyframes = Tuple[Tuple[int, int], ...]


def drift_keyframes(n_samples: int, sr: int, n_waypoints: int,
                    rng: random.Random, segment_sec: Tuple[float, float],
                    dwell_p: float, dwell_sec: Tuple[float, float]
                    ) -> DriftKeyframes:
    """Corner times and waypoint indices of one trajectory.

    A dwell is expressed as a keyframe repeating the previous waypoint, so the
    interpolator below needs no separate "holding" state: interpolating
    between two identical mixtures is a constant.

    ⚠ Both durations are drawn LOG-uniformly over their range.  Movement
    durations are naturally log-distributed -- a gesture lasts a fraction of a
    second, crossing a room takes several -- and a uniform draw concentrates
    the trajectory on one timescale, which shows up directly as a
    correlation-vs-lag curve with a sharp knee where the measured one has none.
    """
    if n_waypoints < 2:
        raise ValueError(f"a trajectory needs >= 2 waypoints, got {n_waypoints}")
    keyframes = [(0, 0)]
    position = 0
    current = 0
    while position < n_samples:
        if rng.random() < dwell_p:
            position += _log_uniform_samples(rng, dwell_sec, sr)
            keyframes.append((position, current))
        drawn = rng.randrange(n_waypoints - 1)
        current = drawn if drawn < current else drawn + 1  # never the same twice
        position += _log_uniform_samples(rng, segment_sec, sr)
        keyframes.append((position, current))
    return tuple(keyframes)


def _log_uniform_samples(rng: random.Random, seconds: Tuple[float, float],
                         sr: int) -> int:
    low, high = seconds
    if not 0 < low <= high:
        raise ValueError(f"requires 0 < min <= max, got {low} .. {high}")
    return max(1, int(sr * math.exp(
        rng.uniform(math.log(low), math.log(high)))))


def drift_corners(n_waypoints: int, depth: float) -> torch.Tensor:
    """The (K, K) mixtures the keyframes walk between; row k is corner k.

    ``depth`` is how far one keyframe pulls the mixture toward its own
    position: 1.0 makes each corner a pure single-position path (and a lag long
    enough to reach a different corner decorrelates as far as the two positions
    themselves differ), 0.0 makes every corner the same uniform blend (nothing
    ever changes).  Every corner is therefore ``(1 - depth) * anchor + depth *
    e_k`` around the uniform ANCHOR mixture, which is what
    ``solve_mixture_depth`` measures the corners against.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"drift depth must be in [0, 1], got {depth}")
    floor = (1.0 - depth) / n_waypoints
    corners = torch.full((n_waypoints, n_waypoints), floor)
    corners.fill_diagonal_(floor + depth)
    return corners


# How far a rendered reach may sit above the configured target before the
# renderer calls the room unreachable. The bisection below stops at a depth
# resolution of `tolerance`, so the correlation it lands on is that resolution
# times the local slope away from the target -- measured over the calibration
# pools at 1.8e-05 worst, i.e. two orders of magnitude inside this.
POSITION_REACH_TOLERANCE = 1e-3


def _reaches(reach: float, target: float) -> bool:
    """Does a set of positions move as far as the config asked for?

    Lower IS further: the number is a path correlation, so a set reaches its
    target by sitting at or below it.
    """
    return reach <= target + POSITION_REACH_TOLERANCE


def position_correlation_target(cfg: configparser.ConfigParser,
                                mode: str) -> float:
    """The path correlation ``mode``'s trajectory is configured to reach."""
    return cfg.getfloat('path_motion', f'{mode}_position_correlation')


# How many position sets a room's eligibility is certified by ENUMERATION,
# per waypoint count. A room offering more of them is certified
# constructively instead, over its most distinct positions
# (``certified_positions``): what has to be shown is that SOME drawable set
# reaches the target, and a set that does is a complete proof of that.
REACH_SUBSET_LIMIT = 20000

# How many position sets one render may draw and reject before it falls back
# to the room's certified set. Eligibility says a reaching set EXISTS, so the
# set a sequence happens to draw still has to be checked -- a room can hold
# near-duplicate positions next to distinctive ones. A rejected draw costs one
# Gram lookup and no audio, and the fallback always reaches, so this trades
# only how much of the room a saturating draw keeps available.
POSITION_DRAW_ATTEMPTS = 8


def mixture_reach(gram: np.ndarray, n_waypoints: int, depth: float) -> float:
    """Mean corner-vs-anchor path correlation of a trajectory at ``depth``.

    Averaged over the corners: the trajectory visits them all, so the room's
    path correlation is what a corner costs on average, not what the most or
    least distinctive position happens to cost.
    """
    anchor = np.full(n_waypoints, 1.0 / n_waypoints)
    return float(np.mean([
        mixture_correlation(gram, anchor, corner)
        for corner in drift_corners(n_waypoints, depth).numpy()]))


def positions_reach(gram: np.ndarray, positions: Sequence[int]) -> float:
    """How far a trajectory over ``positions`` can decorrelate the path.

    A trajectory can only decorrelate as far as the positions it mixes differ,
    and its deepest mixture -- pure single-position corners -- is that limit.
    A set whose limit sits ABOVE the configured target cannot render the
    corpus's movement axis: the solve saturates at depth 1.0 and the audio
    carries a shallower path than the corpus asked for. The renderer therefore
    asks this of the set it has DRAWN, and draws again when the answer is no.
    """
    index = np.asarray(positions, dtype=np.intp)
    return mixture_reach(gram[index[:, None], index[None, :]], len(index), 1.0)


def most_distinct_positions(gram: np.ndarray,
                            n_waypoints: int) -> Tuple[int, ...]:
    """``n_waypoints`` positions of a room, farthest-first on the framed Gram.

    Greedy from the least alike PAIR, then repeatedly the position whose
    strongest band-limited correlation with the set so far is weakest.
    Saturation is a property of positions that share their energy, so this is
    the room's best candidate for a set that reaches -- and it is available to
    the renderer as a fallback, which is what lets a set that reaches CERTIFY
    the room without enumerating every other set.
    """
    count = gram.shape[0]
    if n_waypoints >= count or n_waypoints < 2:
        return tuple(range(min(count, max(0, n_waypoints))))
    norms = np.sqrt(np.abs(np.diag(gram)))
    similarity = np.abs(gram) / np.maximum(np.outer(norms, norms), 1e-30)
    np.fill_diagonal(similarity, np.inf)
    first, second = np.unravel_index(int(np.argmin(similarity)),
                                     similarity.shape)
    chosen = [int(first), int(second)]
    while len(chosen) < n_waypoints:
        strongest = similarity[:, chosen].max(axis=1)
        strongest[chosen] = np.inf
        chosen.append(int(np.argmin(strongest)))
    return tuple(sorted(chosen))


def certified_positions(gram: np.ndarray, n_waypoints: int,
                        subset_limit: int = REACH_SUBSET_LIMIT
                        ) -> Tuple[Tuple[int, ...], float]:
    """A set of ``n_waypoints`` positions that reaches as far as the room can,
    and how far that is.

    This is what room ELIGIBILITY is decided on, and eligibility is an
    EXISTENCE statement: a room can host a drift mode when at least one
    drawable set of positions reaches the configured path correlation
    (``AecSequenceRenderer.rooms_that_can_host``). Exact by enumeration while
    the room offers few enough sets; beyond that the most distinct positions,
    which certifies the room CONSTRUCTIVELY -- the renderer can always fall
    back to this very set, so a set that reaches proves the claim whatever the
    sets nobody enumerated would have shown.

    ⚠ Ranking sets by their most alike PAIR is not a substitute. Pairwise
    similarity does not order set reach, so a room can be certified over a set
    it would never draw, or refused over one that does not exist.
    """
    count = gram.shape[0]
    if n_waypoints < 1 or count < n_waypoints:
        return (), 1.0
    if math.comb(count, n_waypoints) > subset_limit:
        chosen = most_distinct_positions(gram, n_waypoints)
        return chosen, positions_reach(gram, chosen)
    sets = np.asarray(
        list(itertools.combinations(range(count), n_waypoints)), dtype=np.intp)
    # Initialised past the top of the statistic's range rather than at 1.0:
    # every set of a room whose responses are identical reaches exactly 1.0,
    # and a strict comparison against 1.0 would then keep no set at all.
    best_set: Tuple[int, ...] = ()
    best_reach = math.inf
    # Closed form rather than the bisection below: at depth 1.0 every corner IS
    # a single position, so the corner-vs-anchor correlation is
    # |sum_j G_jk| / sqrt(|sum_ij G_ij| * G_kk) -- the same statistic
    # `mixture_correlation` computes, evaluated for every set at once.
    # Chunked because the gather is (sets, K, K) complex.
    for start in range(0, len(sets), 4096):
        index = sets[start:start + 4096]
        block = gram[index[:, :, None], index[:, None, :]]
        columns = np.abs(block.sum(axis=1))
        total = np.abs(block.sum(axis=(1, 2)))[:, None]
        diagonal = np.abs(np.diagonal(block, axis1=1, axis2=2))
        energy = np.maximum(total * diagonal, 1e-300)
        reach = (columns / np.sqrt(energy)).mean(axis=1)
        at = int(np.argmin(reach))
        if float(reach[at]) < best_reach:
            best_set = tuple(int(position) for position in index[at])
            best_reach = float(reach[at])
    return best_set, best_reach


def solve_mixture_depth(gram: np.ndarray, n_waypoints: int,
                        target_correlation: float,
                        tolerance: float = 1e-4) -> Tuple[float, float]:
    """The mixture depth that puts the corners ``target_correlation`` from the
    anchor, and the correlation actually reached.

    ⚠ Depth is NOT the quantity a corpus can be calibrated in.  What the path
    estimator sees is the correlation between mixture STATES, and that depends
    on how much the room's positions share: the same depth measures a corner
    correlation of 0.79 on a pool of 60 ms paths and 0.65 on one of 350 ms
    paths, because longer, more different tails decorrelate further per unit of
    weight change.  A depth fitted on one pool therefore over- or under-drives
    the movement axis on every other one, INCLUDING the same pool at another
    sample rate.  Solving per room from the room's own Gram matrix makes the
    configured number a property of the corpus instead: the target correlation
    is what the calibration is expressed in, and the depth is whatever that
    room needs to reach it.

    ⚠ The reached correlation is returned rather than assumed.  A set of
    positions that share most of their energy cannot decorrelate to an
    arbitrary target even at depth 1.0 (pure single-position corners).
    Rendering that silently is the failure this solve exists to remove, and it
    is removed UPSTREAM: the room is certified to offer a set that reaches
    (``certified_positions``) and the set actually drawn is checked before any
    audio is touched (``AecSequenceRenderer._draw_positions``), so a saturated
    return here means the caller passed a set neither of those two approved --
    which the renderer refuses.  Both numbers are recorded per chunk as
    diagnostics, and the CLI reports the reached median against the configured
    target.
    """
    def reached(depth: float) -> float:
        return mixture_reach(gram, n_waypoints, depth)

    deepest = reached(1.0)
    if deepest >= target_correlation:
        return 1.0, deepest
    low, high = 0.0, 1.0
    # Monotone in depth (every corner walks away from the anchor along a
    # straight line in mixture space), so bisection converges on the one root.
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if reached(middle) > target_correlation:
            low = middle
        else:
            high = middle
    depth = 0.5 * (low + high)
    return depth, reached(depth)


def drift_weights(keyframes: DriftKeyframes, n_waypoints: int, depth: float,
                  n_samples: int) -> torch.Tensor:
    """The (K, n_samples) weight schedule the keyframes describe."""
    corners = drift_corners(n_waypoints, depth)
    weights = torch.zeros(n_waypoints, n_samples)
    for (start, from_index), (end, to_index) in zip(keyframes, keyframes[1:]):
        if start >= n_samples:
            break
        stop = min(end, n_samples)
        span = torch.arange(stop - start, dtype=torch.float32) / max(1, end - start)
        # Raised cosine: zero slope at both corners, so a trajectory that
        # reverses direction has no velocity step for a model to key on.
        blend = 0.5 - 0.5 * torch.cos(math.pi * span)
        weights[:, start:stop] = (
            corners[from_index].unsqueeze(1) * (1.0 - blend)
            + corners[to_index].unsqueeze(1) * blend)
    return weights


def gain_walk_db(n_samples: int, sr: int, rng: random.Random,
                 sigma_db_per_sec: float, clamp_db: float, update_sec: float,
                 recentre_sec: float) -> torch.Tensor:
    """A bounded dB-domain random walk, linearly interpolated to per sample.

    Per-sample rather than per-block because a block-constant gain is a step,
    and a step in the echo's level is a broadband click the model can use as a
    free movement detector.

    ⚠ ``recentre_sec`` (a pull back toward the sequence's own mean level) is
    not decoration.  A free walk that is merely clipped at +-clamp spends most
    of a 30 s sequence pinned to one rail: at the sigma needed for a realistic
    per-second step its level spread comes out 10 dB wide against a measured
    ~6 dB, and the level distribution turns bimodal.  Reverting instead bounds
    the spread by the process itself, so sigma and clamp keep meaning what
    they say.
    """
    if update_sec <= 0:
        raise ValueError(f"gain update_sec must be positive, got {update_sec}")
    if recentre_sec <= 0:
        raise ValueError(
            f"gain recentre_sec must be positive, got {recentre_sec}")
    hop = max(1, int(sr * update_sec))
    n_steps = n_samples // hop + 2
    step_sigma = sigma_db_per_sec * math.sqrt(hop / sr)
    retain = math.exp(-(hop / sr) / recentre_sec)
    value = 0.0
    track = [0.0]
    for _ in range(n_steps - 1):
        value = min(max(retain * value + rng.gauss(0.0, step_sigma),
                        -clamp_db), clamp_db)
        track.append(value)

    knots = torch.tensor(track, dtype=torch.float32)
    position = torch.arange(n_samples, dtype=torch.float32) / hop
    low = position.floor().long().clamp(max=n_steps - 2)
    frac = position - low
    return knots[low] * (1.0 - frac) + knots[low + 1] * frac


def apply_weight_schedule(convolved: Sequence[torch.Tensor],
                          weights: torch.Tensor) -> torch.Tensor:
    """sum_k w_k(t) * (signal convolved with RIR k)."""
    out = convolved[0] * weights[0]
    for signal, weight in zip(convolved[1:], weights[1:]):
        out = out + signal * weight
    return out


def moving_chunks(weights: Optional[torch.Tensor], chunk_samples: int,
                  n_chunks: int, weight_delta: float) -> frozenset:
    """Which chunks the path actually travels during.

    "Moving" is the chunk's own weight EXCURSION -- the largest within-chunk
    swing of any single weight -- not the difference between its endpoints.  A
    chunk in which the path leaves a position and comes back is moving, and
    endpoint differencing would call it still.

    ⚠ The level walk is deliberately NOT part of this.  A device whose level
    drifts while it stays where it is has not moved, and the walk never stops:
    its per-chunk swing clears any useful threshold on every chunk of every
    drifting sequence, so folding it in either labels everything (as a
    disjunction) or makes the label silently depend on a knob that has nothing
    to do with the trajectory (as a conjunction).  The weight excursion is the
    whole discriminating quantity: a dwell keyframe repeats its predecessor, so
    the weights are EXACTLY constant across a dwell.

    ⚠ How much this discriminates depends on the CHUNK LENGTH.  A chunk longer
    than the longest dwell the mode can draw always contains part of a
    transition, so at ``[sequence] chunk_sec`` above ``*_dwell_sec_max`` every
    far-active chunk of a drifting sequence is labelled and the label coincides
    with ``path_motion in DRIFT_MODES and far_active``.  It separates
    travelling chunks from dwelling ones only when chunks are shorter than a
    dwell.

    ⚠ This answers "did the SCHEDULE move", which is not the whole label the
    corpus records: ``_build_meta`` also requires the chunk's own measured far
    activity, because a moving path over a silent reference puts no moving echo
    in the microphone. A caller reproducing the label has to apply that too.
    """
    if weights is None:
        return frozenset()
    moving = set()
    for index in range(n_chunks):
        window = slice(index * chunk_samples, (index + 1) * chunk_samples)
        span = weights[:, window]
        if not span.numel():
            continue
        if float((span.amax(dim=1) - span.amin(dim=1)).max()) >= weight_delta:
            moving.add(index)
    return frozenset(moving)


@dataclasses.dataclass(frozen=True)
class PathMotion:
    """One sequence's echo-path trajectory, decided before any convolution."""

    mode: str = STATIC_PATH
    keyframes: DriftKeyframes = ()
    weights: Optional[torch.Tensor] = None      # (K, T), columns sum to 1
    gain_db: Optional[torch.Tensor] = None      # (T,)
    switch_sample: int = -1                     # one-shot crossfade only
    # What the mixture solve did with THIS room's positions: the depth it
    # needed and the corner-to-anchor path correlation it reached. -1.0 for a
    # path with no solved mixture -- a still one, or the one-shot switch, whose
    # two positions are rendered pure.
    mixture_depth: float = -1.0
    position_correlation: float = -1.0
    # The sample after which the renderer guarantees far-end activity: the
    # switch instant, or the start of the last transition. An echo-path event
    # with no echo behind it teaches nothing.
    event_sample: int = -1

    def render(self, convolved: Sequence[torch.Tensor],
               fade: int = 1) -> torch.Tensor:
        """Travel this trajectory through the responses it was planned over.

        ``convolved`` is the signal already convolved with each of the
        trajectory's positions, in waypoint order.  One implementation for the
        echo path and for the near talker's own path, so a change to how a
        trajectory sounds cannot reach one of them and not the other.

        The level walk is applied here too: it is part of the trajectory, and
        applying it at some call sites and not others is what would make the
        movement axis and the level axis drift apart.
        """
        if self.mode == 'echo_path_change':
            out = _crossfade(convolved[0], convolved[1],
                             self.switch_sample, fade)
        elif self.weights is not None:
            out = apply_weight_schedule(convolved, self.weights)
        else:
            out = convolved[0]
        if self.gain_db is not None:
            out = out * torch.pow(10.0, self.gain_db / 20.0)
        return out


@dataclasses.dataclass(frozen=True)
class _EchoPathRender:
    """What one render's echo path actually did.

    Everything in here exists only AFTER the draws: the trajectory, the near
    talker's own trajectory, which chunks moved, which RIR files the positions
    came from, and the one-shot events.  ``_build_meta`` re-derives whatever it
    can from the plan and takes the rest through this one record, so adding a
    field to the motion model does not add a positional argument to a call site
    nobody can read.
    """

    motion: PathMotion
    near_motion: PathMotion
    moving: frozenset
    waypoint_ids: Tuple[str, ...]
    position_redraws: int
    position_fallbacks: int
    near_shared_positions: int
    delay_step: int
    delay_step_at: int


# ============================================================
# Renderer
# ============================================================

class AecSequenceRenderer:
    """Renders one parent sequence of aligned stems from one split's sources."""

    def __init__(self, cfg: configparser.ConfigParser, pools: SourcePools,
                 corpus_seed: int):
        self.cfg = cfg
        self.pools = pools
        self.corpus_seed = int(corpus_seed)

        self.sr = cfg.getint('signal', 'sr')

        # Kept in the in-process audit metadata. The WAV-only corpus does not
        # persist this value, so gen_aec_dataset.py's --resume can validate
        # shape/encoding but cannot use it to identify an earlier render.
        self.config_hash = config_hash(cfg)

        self.linear_aec_contract: LinearAecContract = (
            linear_aec_contract_from_config(cfg)
        )
        self.chunk_samples = chunk_samples_from_config(
            cfg, self.linear_aec_contract.hop_size)

        self.snr_values = parse_snr_values(cfg.get('levels', 'snr_values'))
        # One map for the whole population: which model an id gets is a
        # property of (config, corpus seed), not of the id, so it is drawn
        # once here rather than re-permuted per device.
        nonlinear_of = nonlinearity_by_id(cfg, self.corpus_seed)
        self.devices = {
            device_id: device_for_id(device_id, cfg, self.corpus_seed, self.sr,
                                     nonlinear_of=nonlinear_of)
            for device_id in pools.devices
        }
        # Per-device mic cascades, built once: identical for every sequence that
        # uses the device, which is the entire point of a device identity.
        self._mic_chains: Dict[str, list] = {}
        # The largest room in the split, i.e. the longest trajectory this RIR
        # pool could hold at all. A fact about the manifest, never affected by
        # which sequence is rendering, so it is read once here.
        self._largest_room = max(
            (len(pools.rirs_by_room[room]) for room in pools.rooms), default=0)
        # Room eligibility is a property of the manifest and the config, so it
        # is remembered rather than re-decided per sequence: the Gram of a
        # room's positions costs one load of every RIR in it, and a corpus
        # draws each room hundreds of times.
        self._room_grams: Dict[str, np.ndarray] = {}
        self._room_certificates: Dict[Tuple[str, int, Tuple[int, ...]],
                                      Tuple[Tuple[int, ...], float]] = {}
        self._room_hosts: Dict[Tuple[str, str], bool] = {}
        self._hosting_rooms: Dict[str, Tuple[str, ...]] = {}

    # ---------------- source loading ----------------

    def _load_mono(self, path: str) -> torch.Tensor:
        """One channel of ``path``, at the corpus rate."""
        audio, file_sr = torchaudio.load(path)
        audio = audio[0].float()
        if audio.numel() == 0:
            raise RuntimeError(f"empty audio: {path}")
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)
        return audio

    def _load_audio(self, path: str, rng: random.Random,
                    n_samples: int, loop: bool) -> torch.Tensor:
        audio = self._load_mono(path)
        if audio.shape[-1] < n_samples:
            if loop:
                audio = audio.repeat(n_samples // audio.shape[-1] + 1)
            else:
                audio = F.pad(audio, (0, n_samples - audio.shape[-1]))
        start = rng.randint(0, audio.shape[-1] - n_samples)
        return audio[start:start + n_samples].clone()

    def _load_rir_set(self, paths: Sequence[str]
                      ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepared (early, full) RIR pairs sharing ONE normalisation factor.

        ``prepare_rir`` L2-normalises whatever it is handed, which is right for
        a single path and wrong for a trajectory: it erases precisely the level
        difference between two loudspeaker positions that a moving path exists
        to carry (measured p5--p95 path level spread: ~6 dB).  Every pair is
        therefore rescaled to the first path's normalisation, which leaves a
        one-element set -- every static sequence -- bit-identical to the plain
        ``prepare_rir`` result.

        The factor is recovered without re-implementing the pre-delay trim:
        that trim always keeps the direct-path peak, so ``max|raw| /
        max|prepared|`` IS the L2 norm ``prepare_rir`` divided out.  The trim
        itself stays in force, so the paths differ in level and shape but not
        in position-dependent propagation delay. This is intentional: the RIR
        files do not share a guaranteed recording time origin, so their raw
        peak offsets cannot safely be interpreted as distance. Timing changes
        remain separate, countable ``delay_jitter`` / ``delay_step`` / ``sro``
        axes rather than an untraceable property of the chosen RIR file.
        """
        prepared: List[Tuple[torch.Tensor, torch.Tensor]] = []
        norms: List[float] = []
        for path in paths:
            audio = self._load_mono(path)
            target_rir, full_rir = prepare_rir(
                audio, self.sr,
                late_offset_ms=self.cfg.getfloat('rir', 'early_rir_ms'),
                pre_delay_keep_ms=self.cfg.getfloat('rir', 'pre_delay_keep_ms'),
                rt60=self.pools.rt60.get(path, 0.4),
            )
            peak = float(full_rir.abs().max())
            norms.append(float(audio.abs().max()) / peak if peak > 0 else 1.0)
            prepared.append((target_rir, full_rir))

        anchor = norms[0] if norms[0] > 0 else 1.0
        return [(target * (norm / anchor), full * (norm / anchor))
                for (target, full), norm in zip(prepared, norms)]

    def _render_talker(self, runs: Sequence[Tuple[int, int]], n_samples: int,
                       rng: random.Random, pool: Sequence[str],
                       *, loop: bool = False
                       ) -> Tuple[torch.Tensor, List[str]]:
        """Place whole utterances inside the active runs, drawn from ``pool``.

        Gating a continuous stream with a mask would cut words in half and leave
        a step discontinuity at every boundary; the model would then learn to
        treat that click as the cue for talk onset.

        ``pool`` is which speech corpus to draw from -- ``self.pools.far_speech_files``
        for the far-end talker, ``self.pools.speech_files`` for the near-end one.
        They are the same list unless ``[paths] far_speech_dir`` is configured.

        ``loop`` repeats a drawn file that is shorter than its run instead of
        zero-padding it.  Off for conversational talkers, where a run that
        outlasts its utterance SHOULD fall silent; on only where the run's
        whole purpose is that the signal never stops (far_active_no_echo).
        """
        out = torch.zeros(n_samples)
        used: List[str] = []
        fade = self._talk_fade()
        for start, end in runs:
            length = end - start
            if length <= 2 * fade:
                continue
            path = pool[rng.randrange(len(pool))]
            try:
                segment = self._load_audio(path, rng, length, loop=loop)
            except Exception:
                continue
            ramp = torch.linspace(0.0, 1.0, fade)
            segment[:fade] *= ramp
            segment[-fade:] *= ramp.flip(0)
            out[start:end] = segment
            used.append(path)
        if runs and not used:
            # Every planned run failed to load (or was too short to use) --
            # the chunk would otherwise render as silent while still carrying
            # its planned scenario/level labels, an unlabelled-but-empty clip
            # a consumer has no way to detect from the metadata alone.
            raise RuntimeError(
                f"{len(runs)} talker run(s) were planned but none produced "
                f"usable audio; refusing to emit a silently-empty chunk"
            )
        return out, used

    def _talk_fade(self) -> int:
        """The fade every placed utterance is given, in samples.

        ``_render_talker`` skips a run of ``2 * fade`` or shorter, so a caller
        asking for a run it intends to RELY on has to know the same number --
        it is what separates a guaranteed stretch of far end from one that is
        silently dropped.
        """
        return max(1, int(self.sr * self.cfg.getfloat('activity',
                                                      'talk_fade_sec')))

    def _mic_chain(self, device: DeviceModel):
        if device.device_id not in self._mic_chains:
            self._mic_chains[device.device_id] = _biquad_chain(
                self.sr, device.mic_eq_seed,
                n_filters=self.cfg.getint('devices', 'n_biquad_filters'),
                gain_db=self.cfg.getfloat('devices', 'mic_eq_gain_db'),
                q_min=self.cfg.getfloat('devices', 'biquad_q_min'),
                q_max=self.cfg.getfloat('devices', 'biquad_q_max'),
            )
        return self._mic_chains[device.device_id]

    def _render_noise(self, n_samples: int,
                      rng: random.Random) -> Tuple[torch.Tensor, List[str]]:
        count = rng.randint(1, max(1, self.cfg.getint('noise', 'max_noise_mix')))
        noise = torch.zeros(n_samples)
        ids: List[str] = []
        for _ in range(count):
            path = self.pools.noise_files[rng.randrange(len(self.pools.noise_files))]
            try:
                noise = noise + self._load_audio(path, rng, n_samples, loop=True)
            except Exception:
                continue
            ids.append(self.pools.noise_of.get(path, os.path.basename(path)))
        if not ids:
            # count >= 1 always: noise is never optional the way far/near
            # speech is. Every attempt failing would otherwise emit a chunk
            # whose SNR was drawn against silence while still labelled with a
            # real noise_id.
            raise RuntimeError(
                f"{count} noise file(s) were planned but every audio file "
                f"load failed; refusing to emit a silently-empty chunk"
            )
        return noise, ids

    def _event_window(self, n_samples: int) -> Tuple[int, int, int]:
        """The far-active guarantee behind an echo-path event, measured once.

        ``(horizon, min_run, latest)``: how long the window is, the shortest
        run ``_render_talker`` will place an utterance into -- below that the
        repair is dropped there and the guarantee would be nominal -- and the
        last sample an event may land on with both still inside the sequence.
        """
        horizon = int(self.sr * self.cfg.getfloat(
            'path_motion', 'far_active_after_event_sec'))
        min_run = 2 * self._talk_fade() + 1
        return horizon, min_run, n_samples - horizon - min_run

    def _switch_point(self, n_chunks: int, rng: random.Random) -> int:
        """When the one-shot crossfade happens, or -1 if it cannot happen.

        Away from the very edges, so the chunk labelled 'echo_path_change'
        really does contain audio from both paths -- and far enough from the
        end that the guaranteed far-active window behind the switch fits
        inside the sequence, the same preference the drift modes apply to
        their last transition.  Without it the switch lands in the final chunk
        whenever the sequence has two of them, and the guarantee is truncated
        to whatever is left: a window of a few milliseconds is an echo-path
        event nothing can be measured against.

        A sequence too short to hold both gets no switch at all: the caller
        renders it with a still path rather than labelling an event whose
        window it cannot supply.
        """
        n_samples = n_chunks * self.chunk_samples
        _horizon, _min_run, latest = self._event_window(n_samples)
        lo = max(1, n_chunks // 4)
        hi = min(max(lo, (3 * n_chunks) // 4), latest // self.chunk_samples)
        if hi < lo:
            return -1
        chunk = rng.randint(lo, hi)
        start = chunk * self.chunk_samples
        return start + rng.randrange(min(self.chunk_samples,
                                         latest - start + 1))

    def _room_gram(self, room: str) -> np.ndarray:
        """The band-limited Gram of every position in ``room``.

        Over the WHOLE room rather than one trajectory's positions: which
        positions a sequence draws is a per-sequence draw, and the question
        here is whether the room can host the mode at all.  A trajectory's own
        Gram is the sub-matrix of this one -- ``_load_rir_set`` normalises a
        set to its first response, and a common scalar cancels in every
        correlation -- so the two are the same statement.
        """
        gram = self._room_grams.get(room)
        if gram is None:
            gram = position_gram(
                [full for _target, full
                 in self._load_rir_set(self.pools.rirs_by_room[room])],
                self.sr)
            self._room_grams[room] = gram
        return gram

    def _waypoint_counts(self, mode: str, room: str) -> range:
        """Every trajectory length ``mode`` can draw inside ``room``."""
        positions = len(self.pools.rirs_by_room[room])
        longest = min(_waypoint_range(self.cfg)[1], positions)
        return range(self.waypoints_needed(mode), longest + 1)

    def _certificate(self, room: str, n_waypoints: int,
                     avoid: Tuple[int, ...] = ()
                     ) -> Tuple[Tuple[int, ...], float]:
        """The set of ``room``'s positions that reaches furthest, cached.

        Both an eligibility certificate and the set a render falls back to
        when its own draws saturate, which is why the two read the same call.

        ``avoid`` restricts it to the positions outside that set, which is what
        lets a fallback keep the preference the draws it replaces were
        expressing.  The restricted answer is a certificate of the same kind --
        a drawable set with its reach computed the same way -- so the caller
        holds it to the same target and asks again without ``avoid`` only when
        it does not reach.  Cached per restriction, because a room drawn
        hundreds of times is drawn against the same few restrictions.
        """
        key = (room, n_waypoints, avoid)
        if key not in self._room_certificates:
            gram = self._room_gram(room)
            if avoid:
                keep = [index for index in range(gram.shape[0])
                        if index not in avoid]
                chosen, reach = certified_positions(
                    gram[np.ix_(keep, keep)], n_waypoints)
                answer = (tuple(keep[index] for index in chosen), reach)
            else:
                answer = certified_positions(gram, n_waypoints)
            self._room_certificates[key] = answer
        return self._room_certificates[key]

    def can_host(self, room: str, mode: str) -> Optional[str]:
        """Why ``room`` cannot carry ``mode``'s trajectory, or ``None``.

        The reason rather than a boolean because the census reports it and the
        renderer only asks whether there is one.

        Two requirements, and the second is why this is decided before a room
        is drawn rather than after: the room needs enough positions, and it has
        to OFFER a set of them whose responses differ enough that the
        configured path correlation is reachable at a mixture depth of at most
        1.0.  A trajectory over positions that share most of their energy
        renders a shallower movement axis than the corpus asked for and nothing
        in the audio says so, so a room offering no such set does not host the
        mode -- for every waypoint count the mode could draw here, because the
        count is drawn per sequence.

        ⚠ EXISTENCE, not universality: which of the room's positions a
        sequence draws is a draw of its own, so the render checks its own set
        and draws again (``_draw_positions``) rather than the room being
        refused for the sets that saturate.  A room where most sets saturate
        stays usable and says so in the census.

        The one-shot switch has no mixture to solve: it renders its two
        positions pure, and how far apart they are is what it is.
        """
        key = (room, mode)
        if key not in self._room_hosts:
            if len(self.pools.rirs_by_room[room]) < self.waypoints_needed(mode):
                verdict = 'too_few_positions'
            elif not motion_policy(mode).solves_mixture:
                verdict = None
            else:
                target = position_correlation_target(self.cfg, mode)
                verdict = None if all(
                    _reaches(self._certificate(room, count)[1], target)
                    for count in self._waypoint_counts(mode, room)
                ) else 'positions_too_alike'
            self._room_hosts[key] = verdict
        return self._room_hosts[key]

    def _draw_positions(self, room: str, mode: str, count: int,
                        rng: random.Random,
                        avoid=frozenset()) -> Tuple[List[str], int, bool]:
        """``count`` of ``room``'s positions whose trajectory REACHES ``mode``'s
        configured path correlation, how many draws were rejected first, and
        whether the set is a certified one rather than a drawn one.

        ``can_host`` says a reaching set exists; this is where the set the
        sequence actually renders is held to it.  A rejected draw costs one
        Gram lookup and no audio.  After ``POSITION_DRAW_ATTEMPTS`` the room's
        certified set is rendered instead -- that set is the certificate, so
        the fallback cannot fail -- and both counts are recorded per sequence,
        because they say different things: re-draws mean the room holds
        near-duplicate positions next to distinctive ones, while a fallback
        means this sequence renders the room's ONE certified trajectory, which
        every other saturating draw in that room renders too.

        ⚠ The fallback keeps ``avoid``.  Certifying over the room's positions
        minus that set answers the same existence question on a smaller room,
        so a near talker whose draws saturate still leaves the loudspeaker's
        positions where a reaching set exists beside them; only a restriction
        that leaves nothing reaching gives the shared certificate back.
        """
        room_rirs = self.pools.rirs_by_room[room]
        if not motion_policy(mode).solves_mixture or count < 2:
            return _sample_waypoints(room_rirs, count, rng, avoid=avoid), 0, False
        target = position_correlation_target(self.cfg, mode)
        gram = self._room_gram(room)
        index_of = {path: index for index, path in enumerate(room_rirs)}
        for rejected in range(POSITION_DRAW_ATTEMPTS):
            drawn = _sample_waypoints(room_rirs, count, rng, avoid=avoid)
            reach = positions_reach(gram, [index_of[path] for path in drawn])
            if _reaches(reach, target):
                return drawn, rejected, False
        avoided = tuple(sorted(index_of[path] for path in avoid
                               if path in index_of))
        certified, reach = self._certificate(room, count, avoided)
        if len(certified) < count or not _reaches(reach, target):
            certified = self._certificate(room, count)[0]
        return ([room_rirs[index] for index in certified],
                POSITION_DRAW_ATTEMPTS, True)

    def rooms_that_can_host(self, mode: str) -> Tuple[str, ...]:
        """Every room of this split ``mode`` may be drawn from."""
        if mode not in self._hosting_rooms:
            self._hosting_rooms[mode] = tuple(
                room for room in self.pools.rooms
                if self.can_host(room, mode) is None)
        return self._hosting_rooms[mode]

    def motion_room_census(self, mode: str) -> Dict[str, List[str]]:
        """Which rooms can host ``mode``, and why the others cannot.

        Keyed 'eligible' / 'too_few_positions' / 'positions_too_alike', the
        last meaning the room's CERTIFICATE does not reach the configured path
        correlation at some waypoint count. On the constructive branch that is
        sufficient and not necessary: the certificate is one set the renderer
        can always draw, so a room it clears is renderable, while a room it
        does not clear may still hold a reaching set nobody enumerated. The
        refusal is that way round on purpose -- what a render needs is a set it
        can be given, not a proof that one exists somewhere.

        The preflight prints it and refuses on it; it costs one load of every
        RIR in the split's rooms, which is why the CLI does it once up front
        instead of every worker discovering it.
        """
        census: Dict[str, List[str]] = {
            'eligible': [], 'too_few_positions': [], 'positions_too_alike': []}
        for room in self.pools.rooms:
            census[self.can_host(room, mode) or 'eligible'].append(room)
        return census

    def waypoints_needed(self, mode: str) -> int:
        """The fewest positions ``mode`` can be rendered with at all."""
        needed = motion_policy(mode).needs_waypoints
        return _waypoint_range(self.cfg)[0] if needed is None else needed

    def _waypoint_count(self, mode: str, rng: random.Random) -> int:
        """How many positions this sequence's trajectory visits.

        Capped by the largest room in the split rather than failing: the count
        is a modelling preference, the RIR inventory is a fact.  The drawn
        room caps it again (see ``_render_impl``), and a pool too sparse for
        even the minimum is refused there, because silently rendering a
        two-position "trajectory" would put the corpus's movement share where
        the metadata says continuous drift is.
        """
        needed = self.waypoints_needed(mode)
        if not motion_policy(mode).solves_mixture:
            return needed          # the one-shot switch visits exactly two
        return rng.randint(needed, min(_waypoint_range(self.cfg)[1],
                                       max(2, self._largest_room)))

    def _plan_path_motion(self, mode: str, n_waypoints: int, n_chunks: int,
                          n_samples: int, rng: random.Random,
                          responses: Sequence[torch.Tensor] = (),
                          *, room: str = '', slowdown: float = 1.0,
                          gain_scale: float = 1.0) -> PathMotion:
        """Draw the trajectory before any audio is touched.

        ``responses`` are the prepared impulse responses of the positions this
        trajectory mixes, in waypoint order: the configured knob is a target
        path correlation and the mixture depth that reaches it is a property of
        those responses, so a drift mode cannot be planned without them.

        ⚠ Reaching the configured correlation is an INVARIANT here, not an
        outcome: ``can_host`` certified before the room was drawn that a
        reaching set exists for every trajectory length this mode can draw in
        it, and ``_draw_positions`` handed over a set that reaches -- its own
        draw or, failing that, the certified one. A solve that saturates means
        those two disagree, which is worth a stopped run: the alternative is a
        corpus whose movement axis is quietly shallower than its own metadata
        claims.

        ``slowdown`` / ``gain_scale`` are how the near talker's own path is
        derived from the same draw: a person shifting in a chair moves the same
        way a loudspeaker does, only slower and with less level change.
        """
        cfg = self.cfg
        policy = motion_policy(mode)
        if policy.renders_crossfade:
            switch = self._switch_point(n_chunks, rng)
            if switch < 0:
                # No room for the event's own far-active window: render the
                # sequence still rather than record an event the audio cannot
                # support. `path_motion` in the metadata is the rendered mode,
                # so the downgrade is countable against `impairments`.
                return PathMotion()
            return PathMotion(mode=mode, switch_sample=switch,
                              event_sample=switch)
        if not policy.solves_mixture:
            return PathMotion()

        def value(key: str) -> float:
            return cfg.getfloat('path_motion', f'{mode}_{key}')

        if len(responses) != n_waypoints:
            raise ValueError(
                f"a {mode!r} trajectory over {n_waypoints} positions needs the "
                f"same number of impulse responses to solve its mixture from, "
                f"got {len(responses)}")
        keyframes = drift_keyframes(
            n_samples, self.sr, n_waypoints, rng,
            segment_sec=(value('segment_sec_min') * slowdown,
                         value('segment_sec_max') * slowdown),
            dwell_p=value('dwell_p'),
            dwell_sec=(value('dwell_sec_min') * slowdown,
                       value('dwell_sec_max') * slowdown),
        )
        target = position_correlation_target(cfg, mode)
        depth, reached = solve_mixture_depth(
            position_gram(responses, self.sr), n_waypoints, target)
        if not _reaches(reached, target):
            raise RuntimeError(
                f"{mode} over {n_waypoints} positions of room {room!r} reaches "
                f"a path correlation of {reached:.4f} at mixture depth "
                f"{depth:.4f}, short of the configured {target:g}: these "
                f"positions share too much of their energy to move that far. "
                f"A trajectory renders a set of positions that reaches the "
                f"target -- its own draw or the room's certified set -- so "
                f"this render was handed a set neither of those two approved")
        weights = drift_weights(keyframes, n_waypoints, depth, n_samples)
        gain = gain_walk_db(
            n_samples, self.sr, rng,
            value('gain_sigma_db_per_sec') * gain_scale,
            value('gain_clamp_db') * gain_scale,
            cfg.getfloat('path_motion', 'gain_update_sec'),
            cfg.getfloat('path_motion', 'gain_recentre_sec'))
        # The last corner the path actually travels to: a dwell repeats its
        # predecessor, so scanning back for a changed waypoint index is what
        # finds the final transition rather than the final keyframe.
        # ⚠ Preferring the last transition that still has room for the
        # guaranteed far-active stretch behind it: a transition starting in the
        # final fraction of a second is an event nothing can be scheduled after,
        # and reporting it would spend the guarantee on a stretch too short to
        # hold an utterance while an earlier, verifiable transition exists.
        horizon, _min_run, _latest = self._event_window(n_samples)
        last_move = 0
        last_move_with_room = -1
        for (start, from_index), (_end, to_index) in zip(keyframes,
                                                         keyframes[1:]):
            if to_index != from_index and start < n_samples:
                last_move = start
                if start + horizon <= n_samples:
                    last_move_with_room = start
        return PathMotion(
            mode=mode, keyframes=keyframes, weights=weights, gain_db=gain,
            mixture_depth=depth, position_correlation=reached,
            event_sample=(last_move_with_room if last_move_with_room >= 0
                          else last_move))

    # ---------------- the render ----------------

    def render(self, plan: SequencePlan) -> RenderedSequence:
        # One seed for the whole sequence: sequence 412 renders identically
        # whether it goes first, last, or through worker 3.
        with _seeded_global_rng(stable_seed(plan.seed, 'globals')):
            return self._render_impl(plan)

    def _render_impl(self, plan: SequencePlan) -> RenderedSequence:
        cfg = self.cfg
        sr = self.sr
        rng = random.Random(plan.seed)
        n_chunks = plan.n_chunks
        n_samples = n_chunks * self.chunk_samples
        talk_mode, echo_mode, impairments_tuple = resolve_sequence_plan(plan)
        impairments = frozenset(impairments_tuple)
        acoustic_tails = frozenset(resolve_acoustic_tails(plan))

        # --- sources -------------------------------------------------------
        device = self.devices[self.pools.devices[rng.randrange(len(self.pools.devices))]]
        if 'nonlinear_spk' in impairments:
            # This impairment IS strong distortion, so drawing a linear
            # device would make the plan a lie a consumer cannot detect.
            distorting = sorted(
                (d for d in self.devices.values() if d.nonlinear != 'linear'),
                key=lambda d: d.device_id,
            )
            if distorting:
                device = distorting[rng.randrange(len(distorting))]

        motion_mode = path_motion_mode(impairments)
        policy = motion_policy(motion_mode)
        needed = self.waypoints_needed(motion_mode)
        if motion_mode != STATIC_PATH and self._largest_room < needed:
            # A split whose largest room cannot hold the shortest trajectory
            # this mode admits is a configuration error, not an inventory one:
            # every sequence of that mode would silently render still.
            # gen_aec_dataset.py refuses it once, up front, before any worker
            # starts; this is the belt for direct AecSequenceRenderer callers
            # (tests, rematerialize_linear_aec.py) that skip that preflight.
            raise RuntimeError(
                f"path motion {motion_mode!r} requires a room with >= "
                f"{needed} RIR files; none of the "
                f"{len(self.pools.rooms)} available room(s) qualify"
            )
        n_waypoints = self._waypoint_count(motion_mode, rng)
        # The room is drawn from the WHOLE pool; a moving sequence that draws a
        # room which cannot host its trajectory draws again from the rooms that
        # can, so the planned motion share is the rendered one.
        # ⚠ That leaves an INHERENT cue: a trajectory needs >= waypoints_min
        # positions in one room (a waypoint in another room would leave the
        # near talker behind) whose responses differ enough to reach the
        # configured path correlation, so moving sequences can only ever come
        # from rooms that qualify, and a single-position room only ever renders
        # still. Drawing every sequence from the whole pool does not remove it
        # -- it only adds still sequences to the rich rooms while downgrading
        # most of the motion axis. gen_aec_dataset.py prints the eligible-room
        # share per mode and refuses a split with fewer than two.
        room = self.pools.rooms[rng.randrange(len(self.pools.rooms))]
        if (motion_mode != STATIC_PATH
                and self.can_host(room, motion_mode) is not None):
            eligible = self.rooms_that_can_host(motion_mode)
            if not eligible:
                raise RuntimeError(
                    f"path motion {motion_mode!r} has no room to render in: "
                    f"of the {len(self.pools.rooms)} room(s) in this split, "
                    f"none holds >= {needed} positions whose responses differ "
                    f"enough to reach the configured path correlation")
            room = eligible[rng.randrange(len(eligible))]
        room_rirs = self.pools.rirs_by_room[room]
        # The count is a modelling preference and the inventory is a fact, so a
        # room holding fewer files than the drawn count shortens the trajectory
        # -- it can no longer be below the minimum the mode needs.
        n_waypoints = min(n_waypoints, len(room_rirs))
        # ⚠ The loudspeaker and the near talker are ALWAYS in the SAME room --
        # RIRs from different rooms would hand the model an acoustic "this is
        # echo" cue that no real device ever has. This holds for every
        # trajectory too: the waypoint count above is capped by this room's own
        # inventory, so no waypoint ever has to leave it.
        # ⚠ The set is drawn against its own reach, not just the room's: the
        # room is eligible because a reaching set EXISTS in it, and a drift
        # mode has to render one of those.
        echo_waypoints, echo_redraws, echo_fell_back = self._draw_positions(
            room, motion_mode, n_waypoints, rng)

        has_far = talk_mode != 'near_only'
        has_near = talk_mode != 'far_only'

        # The near talker prefers positions the loudspeaker is not using, and
        # falls back to sharing them only in a room too sparse to keep them
        # apart -- the preference holds in the draw and in the certified set
        # that replaces the draw, so a room whose sets mostly saturate does not
        # quietly put both paths on the same positions.
        # Only a DRIFTING sequence gives the near talker a trajectory, so the
        # one-shot switch draws a single near position: loading a second RIR
        # there would spend I/O and rng entropy on a path nothing travels.
        near_waypoints, near_redraws, near_fell_back = self._draw_positions(
            room, motion_mode,
            n_waypoints if policy.near_gets_trajectory else 1, rng,
            avoid=frozenset(echo_waypoints)) if has_near else ([], 0, False)

        # Loaded before the trajectory is planned, not after: how deep a
        # mixture has to be to reach the configured path correlation is a
        # property of THESE responses (see solve_mixture_depth), so the
        # positions have to be on hand before their weight schedule exists.
        # The echo must contain the COMPLETE acoustic path; the paired early
        # RIR is only the near talker's dereverberation target.
        echo_rirs = [full for _target, full in self._load_rir_set(echo_waypoints)]
        near_rirs = self._load_rir_set(near_waypoints) if has_near else []

        motion = self._plan_path_motion(
            motion_mode, n_waypoints, n_chunks, n_samples, rng,
            echo_rirs if policy.solves_mixture else (), room=room)
        # DT movement: the same trajectory shape on the near talker's own path,
        # slower and with less level swing. Drawn from the same RNG stream and
        # only when there IS a near talker, so a far-only sequence spends no
        # draws on a path nothing travels. Its own positions solve its own
        # mixture depth: the near talker stands somewhere else in the room, and
        # a depth borrowed from the loudspeaker's positions would move that
        # path by however much the two sets of responses happen to differ.
        near_motion = (
            self._plan_path_motion(
                motion_mode, n_waypoints, n_chunks, n_samples, rng,
                [full for _target, full in near_rirs], room=room,
                slowdown=cfg.getfloat('path_motion', 'near_slowdown'),
                gain_scale=cfg.getfloat('path_motion', 'near_gain_scale'))
            if has_near and policy.near_gets_trajectory else PathMotion())
        if motion.mode == STATIC_PATH and motion_mode != STATIC_PATH:
            # The trajectory could not be placed inside this sequence, so the
            # positions drawn for it are not rendered either and must not be
            # reported as the path's waypoints.
            echo_waypoints = echo_waypoints[:1]
            echo_rirs = echo_rirs[:1]

        # --- talker activity -----------------------------------------------
        if echo_mode == 'far_active_no_echo':
            # This is a hard negative for the adaptive filter/model, not an
            # ordinary conversation label. Keep the reference scheduled over
            # the whole parent sequence so every future chunk actually tests
            # "far present, echo absent" instead of spending part of this
            # scarce class on an ordinary silent-reference interval.
            far_runs = _contiguous_runs(
                n_samples, sr,
                cfg.getfloat('activity', 'far_talk_sec_mean'), rng)
        else:
            far_runs = activity_runs(
                n_samples, sr,
                cfg.getfloat('activity', 'far_talk_sec_mean'),
                cfg.getfloat('activity', 'far_gap_sec_mean'),
                rng, start_active=True,
            ) if has_far else []
        guaranteed_far = None
        if has_far and motion.event_sample >= 0:
            # An echo-path event with no far end behind it is invisible: the
            # mic carries no echo to have moved. Two independent exponential
            # chains leave that hole often enough to matter, so the far
            # schedule is repaired rather than hoped for.
            horizon, min_run, _latest = self._event_window(n_samples)
            far_runs, guaranteed_far = _ensure_far_activity_after(
                far_runs, motion.event_sample, n_samples, horizon, min_run)
        near_runs = activity_runs(
            n_samples, sr,
            cfg.getfloat('activity', 'near_talk_sec_mean'),
            cfg.getfloat('activity', 'near_gap_sec_mean'),
            rng,
        ) if has_near else []
        if talk_mode == 'double_talk' and far_runs:
            # Set test first: it rejects ~98% of double_talk sequences for a
            # fraction of what parsing the config value costs.
            force_edges = (
                (DT_STRESS_IMPAIRMENTS <= impairments
                 or DT_ACOUSTIC_TAILS <= acoustic_tails)
                and cfg.getboolean(
                    'activity', 'dt_force_edge_overlap', fallback=False)
            )
            # The forced leading edge must still overlap actual ECHO, which
            # arrives one bulk delay after the reference: widen its floor by
            # the tail's largest delay when this sequence draws the tail.
            edge_floor_extra_s = (
                cfg.getfloat('acoustic_tails', 'long_delay_ms_max') / 1000.0
                if 'long_delay' in acoustic_tails else 0.0)
            near_runs = _force_overlap(
                far_runs, near_runs, rng, sr, cfg, force_edges=force_edges,
                edge_floor_extra_s=edge_floor_extra_s)

        far_speech, far_paths = (
            self._render_talker(far_runs, n_samples, rng,
                                self.pools.far_speech_files,
                                loop=echo_mode == 'far_active_no_echo')
            if has_far else (torch.zeros(n_samples), []))
        near_dry, near_paths = (
            self._render_talker(near_runs, n_samples, rng, self.pools.speech_files)
            if has_near else (torch.zeros(n_samples), []))

        def draw_range(tail: str, base_section: str, base_stem: str):
            """(min, max) of a scalar draw: the tail's range when that tail
            was drawn for this sequence, otherwise the ordinary range."""
            if tail in acoustic_tails:
                section, stem = 'acoustic_tails', ACOUSTIC_TAIL_RANGE_KEYS[tail]
            else:
                section, stem = base_section, base_stem
            return (cfg.getfloat(section, f'{stem}_min'),
                    cfg.getfloat(section, f'{stem}_max'))

        # --- far-end reference X -------------------------------------------
        far_render = _scale_to_active_dbfs(
            far_speech, sr,
            rng.uniform(*draw_range('quiet_far', 'levels', 'far_level_dbfs')))

        # Reference dropout, chosen as WHOLE chunks so that a chunk labelled
        # 'ref_dropout' is unambiguously an idle chunk.
        dropout_chunks = set()
        if echo_mode == 'ref_dropout' and has_far and n_chunks > 1:
            hi = min(cfg.getint('dropout', 'ref_dropout_chunks_max'), n_chunks - 1)
            lo = min(cfg.getint('dropout', 'ref_dropout_chunks_min'), hi)
            count = rng.randint(max(1, lo), max(1, hi))
            # A dropout zeroes whole chunks of the reference, so one landing on
            # the stretch that has to carry the echo-path event would silently
            # undo the guarantee above -- whether that stretch was repaired or
            # was already scheduled.
            starts, count = _dropout_placement(
                n_chunks, count, self.chunk_samples, guaranteed_far)
            first = starts[rng.randrange(len(starts))]
            dropout_chunks = set(range(first, first + count))
            for chunk in dropout_chunks:
                at = chunk * self.chunk_samples
                far_render[at:at + self.chunk_samples] = 0.0

        # --- the played signal (what the loudspeaker actually radiates) -----
        played = far_render
        sro_ppm = 0.0
        if 'codec_mismatch' in impairments:
            candidates = [int(v) for v in cfg.get('codec', 'source_sr_values').split(',')
                          if v.strip() and int(v) < sr]
            if candidates:
                # ⚠ Applied to the PLAYED path only.  The stored reference stays
                # clean, so the corpus contains a reference/played mismatch that
                # a linear filter cannot fully explain -- which is the point.
                played = simulate_codec(
                    played, sr, candidates[rng.randrange(len(candidates))],
                    rng.randint(cfg.getint('codec', 'bits_min'),
                                cfg.getint('codec', 'bits_max')))

        drive = device.drive
        if 'nonlinear_spk' in impairments:
            drive *= cfg.getfloat('devices', 'nonlinear_spk_drive_boost')
        played = apply_loudspeaker_nonlinearity(played, device.nonlinear, drive)

        # Radiated response: band limit, then the device's fixed EQ.  Order is
        # physical -- the driver distorts, the enclosure colours what escapes.
        played = torchaudio.functional.highpass_biquad(played, sr, device.speaker_hp_hz)
        played = torchaudio.functional.lowpass_biquad(played, sr, device.speaker_lp_hz)
        with _seeded_global_rng(device.speaker_eq_seed):
            # Reused as-is; its internal RMS normalisation is harmless here
            # because the echo level is set by the ERL draw further down.
            played = rand_biquad_filter(
                played, sr,
                n_filters=cfg.getint('devices', 'n_biquad_filters'),
                gain_db=cfg.getfloat('devices', 'biquad_gain_db'),
                q_min=cfg.getfloat('devices', 'biquad_q_min'),
                q_max=cfg.getfloat('devices', 'biquad_q_max'))

        if 'sro' in impairments:
            sro_ppm = rng.uniform(cfg.getfloat('sro', 'ppm_min'),
                                  cfg.getfloat('sro', 'ppm_max'))
            if rng.random() < 0.5:
                sro_ppm = -sro_ppm
            # On the PLAYED path, never on the stored reference: the drift has
            # to appear as X and D pulling apart over the sequence.
            played = resample_by_ratio(played, 1.0 + sro_ppm * 1e-6, n_samples)

        # --- bulk delay (and jitter) ---------------------------------------
        delay_ms_min, delay_ms_max = draw_range(
            'long_delay', 'echo_path', 'bulk_delay_ms')
        bulk_delay = rng.randint(int(sr * delay_ms_min / 1000),
                                 int(sr * delay_ms_max / 1000))
        # A device/buffer event, not a movement one: delay jumps > 20 ms were
        # measured in 53% of moving and 63% of static captures, so the two are
        # drawn independently and one must never stand in for the other.
        delay_step, delay_step_at = (
            _draw_delay_step(n_samples, bulk_delay, sr, rng, cfg)
            if 'delay_step' in impairments else (0, -1))
        if delay_step:
            played = _crossfade(
                played, _shift_signal(played, delay_step), delay_step_at,
                max(1, int(sr * cfg.getfloat('echo_path', 'jitter_fade_sec'))))
        delay_jitter = 'delay_jitter' in impairments
        played = (_apply_jittered_delay(played, bulk_delay, sr, rng, cfg,
                                        applied_step=delay_step,
                                        step_at=delay_step_at)
                  if delay_jitter else delay_signal(played, bulk_delay))

        # --- echo path -----------------------------------------------------
        played_through = _convolver(played)
        echo_raw = motion.render(
            [played_through(rir) for rir in echo_rirs],
            max(1, int(sr * cfg.getfloat('echo_path',
                                         'path_change_fade_sec'))))

        if has_near:
            near_through = _convolver(near_dry)
            near_raw = near_motion.render(
                [near_through(full) for _target, full in near_rirs])
            near_target_raw = near_motion.render(
                [near_through(target) for target, _full in near_rirs])
        else:
            near_raw = torch.zeros(n_samples)
            near_target_raw = torch.zeros(n_samples)
        noise_raw, noise_ids = self._render_noise(n_samples, rng)

        # --- microphone -----------------------------------------------------
        # ⚠ The mic response is applied BEFORE the level draws, not after.  The
        # cascade has its own frequency-dependent gain, so filtering afterwards
        # would move every stem by a few dB and the recorded erl/ser/snr would
        # no longer describe the audio a consumer measures.
        chain = self._mic_chain(device)
        echo = _apply_chain(echo_raw, chain)
        near_speech = _apply_chain(near_raw, chain)
        near_target = _apply_chain(near_target_raw, chain)
        noise = _apply_chain(noise_raw, chain)

        # Echo return loss against the STORED reference, so erl_db is a property
        # a consumer can verify directly from the shard.
        erl_db_min, erl_db_max = draw_range('strong_echo', 'levels', 'erl_db')
        if 'quiet_far' in acoustic_tails and 'strong_echo' not in acoustic_tails:
            # Keep the echo above the local-noise floor (see
            # QUIET_FAR_ERL_CAP_KEY); strong_echo already draws a low ERL.
            erl_db_max = min(
                erl_db_max, cfg.getfloat('acoustic_tails', QUIET_FAR_ERL_CAP_KEY))
        erl_db = rng.uniform(erl_db_min, erl_db_max)
        echo = _scale_to_ratio(echo, far_render, sr, -erl_db)

        if echo_mode == 'far_active_no_echo':
            # A reference at full level and no acoustic path back to this
            # microphone; near-end speech may independently be present. Every
            # normal-echo plan ties echo to far_render through erl_db, whose
            # range stops at 30 dB, so the quietest echo the corpus could
            # otherwise produce still sits only ser_db_max below the near
            # speech -- around 25 dB short of what a real headset or a muted
            # loudspeaker looks like.  Without this case nothing teaches that a
            # loud reference can be irrelevant, and a linear filter is free to
            # fit reference-shaped noise onto near speech and subtract
            # something that was never there.
            #
            # Distinct from near_only (far_render itself is silent, so there is
            # no reference to be misled by) and from ref_dropout (which zeros
            # the reference and the echo together, teaching the converse).
            echo = torch.zeros_like(echo)
            erl_db = float('inf')

        if dropout_chunks and rng.random() >= cfg.getfloat(
                'dropout', 'ref_dropout_echo_continues_p'):
            # Default: the far end is genuinely silent, so there is no echo
            # either.  ⚠ This is what makes the hard gate "ref == 0 implies
            # output ~= mic" supervisable -- with X and D both zero the correct
            # D_hat is zero and E == Y exactly.  The alternative (echo keeps
            # playing while the reference is lost) is reachable through
            # ref_dropout_echo_continues_p, but it asks the model to predict an
            # echo from nothing, so it is off by default: raising it trains
            # hallucination.
            for chunk in dropout_chunks:
                at = chunk * self.chunk_samples
                echo[at:at + self.chunk_samples] = 0.0

        if has_near and float(echo.abs().max()) > 0:
            ser_db = rng.uniform(cfg.getfloat('levels', 'ser_db_min'),
                                 cfg.getfloat('levels', 'ser_db_max'))
            near_speech, near_target = _scale_pair_to_ratio(
                near_speech, near_target, echo, sr, ser_db,
            )
        elif has_near:
            ser_db = float('inf')          # no echo: signal-to-echo is unbounded
            near_speech, near_target = _scale_pair_to_active_dbfs(
                near_speech, near_target, sr,
                rng.uniform(cfg.getfloat('levels', 'near_level_dbfs_min'),
                            cfg.getfloat('levels', 'near_level_dbfs_max')),
            )
        else:
            ser_db = float('-inf')         # no near talker at all

        if has_near and float(near_speech.abs().max()) > 0:
            # Same discrete SNR set as the NR generator, drawn with the same
            # helpers, so the two corpora are comparable on the noise axis.
            snr_db = sample_snr(self.snr_values)
            noise = _scale_to_ratio(noise, near_speech, sr, -snr_db)
        else:
            # Nothing to define an SNR against; the noise gets an absolute level
            # and the metadata says so instead of recording a meaningless number.
            snr_db = float('-inf')
            noise = _scale_to_active_dbfs(
                noise, sr,
                rng.uniform(cfg.getfloat('levels', 'noise_level_dbfs_min'),
                            cfg.getfloat('levels', 'noise_level_dbfs_max')))

        mic_preclip = near_speech + noise + echo

        # ⚠ ONE common scale across all acoustic stems. Scaling only the mic would
        # break mic_preclip == S + N + D; scaling everything except the
        # reference would silently change the ERL the metadata claims.
        (far_render, echo, near_speech, near_target,
         noise, mic_preclip) = prevent_clipping(
            far_render, echo, near_speech, near_target, noise, mic_preclip,
            threshold=cfg.getfloat('mic', 'peak_guard'))

        # ⚠ `clipped` and `agc` are recorded separately.  They are two different
        # distortions -- one memoryless and instantaneous, one a slow gain with
        # memory -- and a model that confuses them fixes the wrong one.  Storing
        # a single "the mic was altered" flag would also make it impossible to
        # tell, from the metadata alone, whether mic_postclip differs from
        # mic_preclip at all.
        clipped = False
        agc = False
        mic_postclip = mic_preclip.clone()
        if ('clipping_agc' in impairments
                or rng.random() < cfg.getfloat('mic', 'p_clipping')):
            # apply_clipping() also returns the sampled clip_snr (added for
            # AINR's own per-sample metadata) -- this caller already tracks
            # a separate `clipped` boolean, not the exact sampled value.
            mic_postclip, _clip_snr = apply_clipping(
                mic_postclip,
                cfg.getfloat('mic', 'clip_snr_min'),
                cfg.getfloat('mic', 'clip_snr_max'))
            clipped = True
        if ('clipping_agc' in impairments
                or rng.random() < cfg.getfloat('mic', 'p_agc')):
            mic_postclip = apply_agc(
                mic_postclip, sr,
                cfg.getfloat('mic', 'agc_target_dbfs'),
                cfg.getfloat('mic', 'agc_attack_sec'),
                cfg.getfloat('mic', 'agc_release_sec'),
                cfg.getfloat('mic', 'agc_max_gain_db'))
            agc = True
        # ⚠ apply_clipping renormalises back to the input RMS, which can push
        # the peak above 1.0 again; the AGC can too.  Only mic_postclip is
        # touched here, so the pre-clip sum identity survives untouched.
        mic_postclip = mic_postclip.clamp(-0.999, 0.999)

        # Materialize the frozen linear error over the COMPLETE sequence before
        # chunking. A fresh processor here means cold start at sequence start;
        # the one call preserves PBFDKF adaptation across every future chunk.
        linear_error, echo_estimate = materialize_linear_error(
            mic_postclip.to(torch.float32).contiguous(),
            far_render.to(torch.float32).contiguous(),
            self.linear_aec_contract,
        )

        base_stems = torch.stack([
            far_render, near_speech, near_target, mic_postclip,
        ]).to(torch.float32).contiguous()
        if base_stems.shape[0] != len(BASE_STEM_ORDER):
            raise AssertionError("acoustic stem stack does not match BASE_STEM_ORDER")
        stems = torch.cat([base_stems, linear_error.unsqueeze(0)], dim=0).contiguous()
        if stems.shape[0] != len(STEM_ORDER):
            raise AssertionError("stem stack does not match STEM_ORDER")

        echo_path = _EchoPathRender(
            motion=motion, near_motion=near_motion,
            moving=moving_chunks(
                motion.weights, self.chunk_samples, n_chunks,
                cfg.getfloat('path_motion', 'moving_label_weight_delta')),
            waypoint_ids=tuple(self.pools.rir_id_of.get(path, path)
                               for path in echo_waypoints),
            position_redraws=echo_redraws + near_redraws,
            position_fallbacks=int(echo_fell_back) + int(near_fell_back),
            near_shared_positions=len(
                set(near_waypoints) & set(echo_waypoints)),
            delay_step=delay_step, delay_step_at=delay_step_at,
        )

        chunk_meta = self._build_meta(
            plan, stems, device, room, erl_db, ser_db, snr_db,
            bulk_delay, delay_jitter, sro_ppm, clipped, agc, dropout_chunks,
            noise_ids, echo_path,
            near_speaker=self.pools.speaker_of.get(near_paths[0], '') if near_paths else '',
            far_speaker=self.pools.far_speaker_of.get(far_paths[0], '') if far_paths else '',
        )
        audit = {'echo': echo.to(torch.float32).contiguous(),
                 'noise': noise.to(torch.float32).contiguous(),
                 'mic_preclip': mic_preclip.to(torch.float32).contiguous(),
                 'echo_estimate': echo_estimate.contiguous()}
        # The trajectory itself, so a test can recompute the movement labels
        # from the schedule the renderer actually used instead of from the
        # summary the renderer also wrote.
        if motion.weights is not None:
            audit['echo_path_weights'] = motion.weights.contiguous()
        if motion.gain_db is not None:
            audit['echo_path_gain_db'] = motion.gain_db.contiguous()
        return RenderedSequence(
            stems=stems, chunk_meta=chunk_meta, chunk_samples=self.chunk_samples,
            linear_aec_contract=self.linear_aec_contract.as_dict(),
            audit=audit,
        )

    def _build_meta(self, plan, stems, device, room, erl_db, ser_db,
                    snr_db, bulk_delay, delay_jitter, sro_ppm, clipped, agc,
                    dropout_chunks, noise_ids, echo_path: '_EchoPathRender',
                    near_speaker, far_speaker) -> List[dict]:
        # Re-derived from `plan` rather than threaded down as four more
        # positional arguments: resolve_sequence_plan is pure and O(1), and
        # this call already carries enough of them to align by eye.  Everything
        # that only EXISTS after the draw travels in one `_EchoPathRender`
        # instead, for the same reason.
        talk_mode, echo_mode, impairments = resolve_sequence_plan(plan)
        acoustic_tails = resolve_acoustic_tails(plan)
        legacy_scenario = plan.scenario if plan.talk_mode is None else None
        motion, near_motion = echo_path.motion, echo_path.near_motion
        delay_step, delay_step_at = echo_path.delay_step, echo_path.delay_step_at
        # ⚠ ser_db / snr_db / erl_db are SEQUENCE-level: they describe how the
        # parent sequence was set up, measured over its whole duration.  A
        # single 4 s chunk can depart from them by several dB (ERL) or by
        # anything at all (SER/SNR), because a chunk in which the near talker
        # happens to be silent has no signal to define a ratio against.  Do NOT
        # build a per-chunk curriculum by filtering on these; filter on
        # `scenario`, or measure the chunk yourself from the stems -- which is
        # possible precisely because the stems are stored separately.
        far = stems[STEM_ORDER.index('far_render')]
        near = stems[STEM_ORDER.index('near_speech')]
        threshold = 10.0 ** (ACTIVITY_LABEL_DBFS / 20.0)
        # Clipped to the sequence, KEEPING the corner the last rendered ramp is
        # travelling toward: the keyframe walk stops one corner past the end,
        # and the renderer interpolates the tail toward that corner, so
        # dropping it would describe a ramp the audio really contains as a
        # hold. Everything after it never happened and is dropped.
        # ⚠ The last keyframe's `t_sec` may therefore exceed the sequence
        # length; it is the destination of the final transition, not a corner
        # the path reaches inside the audio.
        n_samples = stems.shape[-1]
        keyframes = []
        for start, index in motion.keyframes:
            keyframes.append({'t_sec': start / self.sr, 'waypoint': index})
            if start >= n_samples:
                break
        gain_track = _gain_track_summary(motion.gain_db)
        switch_chunk = motion.switch_sample // self.chunk_samples
        # What the SEQUENCE decided, assembled once: only the five entries the
        # loop adds below depend on the chunk, and re-deriving the rest per
        # chunk is how two chunks of one sequence come to disagree about the
        # sequence they belong to.
        common = {
            'sequence_id': int(plan.sequence_id),
            # 'speaker_id' is the NEAR talker -- the signal that must
            # survive.  '' means this sequence has no near talker.
            'speaker_id': near_speaker,
            'far_speaker_id': far_speaker,
            'noise_id': '+'.join(noise_ids),
            'rir_id': '|'.join(echo_path.waypoint_ids),
            'room_id': room,
            # How many positions this room holds. A trajectory needs
            # `waypoints_min` of them in one room, so this is the audit of
            # the one motion cue the same-room invariant cannot remove:
            # count the rendered motion modes against it and the cue is a
            # number rather than an assumption.
            'room_rir_count': len(self.pools.rirs_by_room[room]),
            # How many position sets this sequence drew and rejected for
            # not reaching the configured path correlation, over both
            # trajectories. A room can hold near-duplicate positions next
            # to distinctive ones, so this is the audit of how much of an
            # eligible room a drift mode can actually be rendered from.
            'path_position_redraws': int(echo_path.position_redraws),
            # How many of this sequence's two trajectories exhausted their
            # draws and rendered the room's certified set instead. Counted
            # apart from the re-draws above because it is a different fact:
            # every saturating draw in one room falls back to the SAME set,
            # so a corpus with many fallbacks repeats a handful of
            # trajectories however many sequences it holds.
            'path_position_fallbacks': int(echo_path.position_fallbacks),
            # How many positions the near talker's trajectory shares with
            # the loudspeaker's. Preferred apart, but a room can be too
            # sparse to keep them apart, and then the near path and the
            # echo path carry the same room response.
            'near_path_shared_positions':
                int(echo_path.near_shared_positions),
            'device_id': device.device_id,
            'ser_db': float(ser_db),
            'snr_db': float(snr_db),
            'erl_db': float(erl_db),
            'bulk_delay_samples': int(bulk_delay),
            'delay_jitter': bool(delay_jitter),
            # Signed: a dropped playout buffer moves the echo earlier just
            # as readily as a duplicated one moves it later. The instant is
            # given in the MIC timeline, i.e. the reference-side step time
            # plus the bulk delay it travels through.
            # ⚠ That conversion uses the sequence's NOMINAL bulk delay, so
            # on a sequence that also carries delay_jitter the true instant
            # moves with the jitter state (up to jitter_steps_max *
            # jitter_ms_max), and under sro by the drift accumulated at
            # that point. Both are bounded and small next to a chunk; a
            # consumer aligning an evaluation window tightly around the
            # step should widen it by that much.
            'delay_step_samples': int(delay_step),
            'delay_step_at': int(delay_step_at + bulk_delay
                                 if delay_step else -1),
            'sro_ppm': float(sro_ppm),
            'nonlinear': device.nonlinear,
            'clipped': bool(clipped),
            'agc': bool(agc),
            'talk_mode': talk_mode,
            'echo_mode': echo_mode,
            'impairments': list(impairments),
            'acoustic_tails': list(acoustic_tails),
            # Rendered, not planned: a sequence whose drawn room could not
            # supply a second position renders a path that never changes,
            # and this field has to describe the audio. Compare it with
            # `impairments` to count those.
            'echo_path_change': motion.mode == 'echo_path_change',
            'codec_mismatch': 'codec_mismatch' in impairments,
            # The motion axis. `path_motion` is the sequence's rendered
            # mode; `echo_path_moving` is per chunk and MEASURED -- from
            # the trajectory's own within-chunk weight excursion and from
            # the chunk's reference activity -- so a chunk sitting inside a
            # dwell is not labelled, and neither is one whose reference is
            # silent: with no echo in the microphone there is no moving
            # path to hear, and the label would be indistinguishable from
            # a near-only chunk.
            # ⚠ How much the label DISCRIMINATES depends on the chunk
            # length; moving_chunks owns that rule.
            'path_motion': motion.mode,
            'echo_path_waypoints': list(echo_path.waypoint_ids),
            'echo_path_keyframes': keyframes,
            # The switch instant, or the start of the trajectory's last
            # transition that still has room behind it: the renderer
            # guarantees far-end activity inside the
            # far_active_after_event_sec window that follows this sample,
            # and steers a reference dropout away from it. -1 when the
            # path never moves -- including a sequence whose drawn motion
            # could not be given that window, which renders still and says
            # so in `path_motion` rather than recording an event with
            # nothing behind it.
            'echo_path_event_at': int(motion.event_sample),
            # What the mixture solve made of this room: the depth the
            # configured path correlation needed here, and the correlation
            # it actually reached. They differ when the room's positions
            # share too much of their energy to decorrelate that far even
            # at depth 1.0 -- the corpus's movement axis is then shallower
            # than it asked for, and this is where that is visible. -1.0
            # for a sequence with no solved mixture.
            'echo_path_mixture_depth': float(motion.mixture_depth),
            'echo_path_position_correlation': float(
                motion.position_correlation),
            'echo_path_gain_walk_db': gain_track,
            'near_path_motion': near_motion.mode,
            # The near talker's own solve, from the near talker's own
            # positions: they are somewhere else in the room, so their
            # mixture needs its own depth to reach the same configured
            # correlation. Recorded separately for the same reason the
            # echo path's is -- these two numbers are the only place the
            # near trajectory's geometry is visible. -1.0 when the near
            # path has no solved mixture.
            'near_path_mixture_depth': float(near_motion.mixture_depth),
            'near_path_position_correlation': float(
                near_motion.position_correlation),
            'manifest_version': self.pools.manifest_version,
            'linear_aec_contract_hash': self.linear_aec_contract.fingerprint(),
            'config_hash': self.config_hash,
            # config_hash alone would not identify a render: --seed lives
            # outside config.ini, so sequence_seed is recorded beside it.
            # Both fields are in-process audit data only; neither is
            # persisted in the WAV-only corpus or checked by --resume.
            'sequence_seed': int(plan.seed),
            # Compatibility summary. New consumers must use the
            # orthogonal fields above rather than infer combinations from
            # this one string.
            'sequence_scenario': plan.scenario,
            'split': self.pools.split,
        }
        meta = []
        for chunk_index in range(plan.n_chunks):
            at = chunk_index * self.chunk_samples
            window = slice(at, at + self.chunk_samples)
            far_active = float(far[window].pow(2).mean().sqrt()) > threshold
            near_active = float(near[window].pow(2).mean().sqrt()) > threshold
            moving = bool(chunk_index in echo_path.moving and far_active)
            meta.append({
                **common,
                'chunk_index': int(chunk_index),
                'echo_path_moving': moving,
                # Measured activity, kept beside `scenario` because an event
                # label displaces the activity one: without these two a moving
                # double-talk chunk would carry no record that it is DT.
                'far_active': bool(far_active),
                'near_active': bool(near_active),
                'scenario': _chunk_scenario(
                    echo_mode, legacy_scenario, chunk_index, dropout_chunks,
                    switch_chunk, far_active=far_active,
                    near_active=near_active, moving=moving,
                ),
            })
        return meta


def _chunk_scenario(echo_mode: str,
                    legacy_scenario: Optional[str], chunk_index: int,
                    dropout_chunks, switch_chunk: int, far_active: bool,
                    near_active: bool, moving: bool = False) -> str:
    """Per-chunk label, which is not always the sequence's label.

    ⚠ A 'ref_dropout' parent sequence is mostly NOT a dropout, and an
    'echo_path_change' sequence contains exactly one chunk where the path
    changes.  Labelling every chunk with the sequence's intent would make the
    honest test "every ref_dropout clip has a silent reference" fail, and would
    let a dropout-conditioned loss term train on chunks whose reference is fully
    active.  So the label marks the chunks that really are the event, and the
    rest are labelled by what they actually contain.  ``sequence_scenario``
    keeps the sequence-level intent for anyone who needs it.

    ⚠ An event label DISPLACES the activity label, and a drifting sequence's
    event covers most of its chunks.  The chunk's own ``far_active`` /
    ``near_active`` are recorded separately for that reason -- filtering a
    double-talk curriculum on this string alone would silently drop the moving
    share of the corpus.
    """
    if chunk_index in dropout_chunks:
        return 'ref_dropout'
    # ⚠ Both echo-path event labels require the chunk's OWN measured far
    # activity, for the same reason 'far_active_no_echo' does below: a label
    # claiming the path moved or changed cannot be attached to a chunk carrying
    # no echo, where it would be signal-identical to a near-only chunk. The
    # caller has already folded that measurement into `moving`.
    if far_active and chunk_index == switch_chunk:
        return 'echo_path_change'
    if moving:
        return 'echo_path_moving'
    # Measured, not asserted: this label claims the reference IS playing, so a
    # chunk whose far end is actually silent must fall through and be labelled
    # by what it contains.  Otherwise a scheduling regression upstream would
    # keep emitting the label over a silent reference -- signal-identical to a
    # 'ref_dropout' chunk, and contradicting it.
    if echo_mode == 'far_active_no_echo' and far_active:
        return 'far_active_no_echo'
    # Preserve old direct-plan diagnostics exactly. In a layered corpus the
    # physical conditions live in ``impairments`` and this label is reserved
    # for the actual speech activity of the chunk.
    if legacy_scenario in WHOLE_SEQUENCE_SCENARIOS:
        return legacy_scenario
    if far_active and near_active:
        return 'double_talk'
    if far_active:
        return 'far_only'
    return 'near_only'


def _force_overlap(far_runs, near_runs, rng, sr, cfg, *, force_edges=False,
                   edge_floor_extra_s=0.0):
    """Guarantee genuine double talk instead of hoping two chains collide.

    The first and last far bursts are optionally load-bearing. They expose the
    model to DT while the frozen linear AEC is cold and again after its state
    has matured, matching the two failure positions that random middle-only
    overlap used to miss.
    """
    overlap_p = cfg.getfloat('activity', 'dt_overlap_p')
    frac_min = cfg.getfloat('activity', 'dt_overlap_frac_min')
    frac_max = cfg.getfloat('activity', 'dt_overlap_frac_max')
    added = list(near_runs)
    floor = int(sr * 0.2)
    # A forced edge has to cover the echo's arrival too, so its floor grows by
    # the caller's bulk-delay allowance; random middle overlaps keep the base.
    edge_floor = int(sr * (0.2 + edge_floor_extra_s))
    last_index = len(far_runs) - 1
    for index, (start, end) in enumerate(far_runs):
        edge = force_edges and index in (0, last_index)
        if not edge and rng.random() >= overlap_p:
            continue
        length = end - start
        window = int(length * rng.uniform(frac_min, frac_max))
        if edge:
            # A forced edge is a guarantee, so a short draw is widened rather
            # than dropped, and a burst too short to reach the floor is covered
            # whole instead of skipped. Applying the floor here as a `continue`
            # silently cost ~21% of stress-combo sequences at least one of
            # their two edges, while the config and README promised both.
            window = max(window, min(length, edge_floor))
        elif window < floor:
            continue
        if edge and index == 0:
            # The far activity chain starts at sample zero. Pinning this
            # overlap to its leading edge creates a real cold-start DT case.
            offset = 0
        elif edge and index == last_index:
            offset = length - window
        else:
            offset = rng.randint(0, length - window)
        added.append((start + offset, start + offset + window))
    return _merge_runs(added)


def _merge_runs(runs):
    """Sorted, non-overlapping runs.

    ⚠ The accumulator is seeded from the SORTED list, not from ``runs[0]``.
    Seeding from the caller's first run silently drops every run that starts
    before it, and both callers append their new run to the end of a list they
    did not sort.
    """
    if not runs:
        return []
    ordered = sorted(runs)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _apply_jittered_delay(x, base_delay, sr, rng, cfg, applied_step: int = 0,
                          step_at: int = -1):
    """Piecewise bulk delay with crossfades, i.e. playout buffer glitches.

    Discrete steps rather than a slow ramp: a dropped or duplicated playout
    buffer moves the echo by a whole block at once, which is a completely
    different disturbance from an SRO's continuous drift.  Conflating the two
    would let one scenario stand in for the other, and it cannot.

    ⚠ The walk is floored at ``bulk_delay_ms_min``, not at zero, and a
    NEGATIVE ``applied_step`` (a ``delay_step`` already baked into ``x`` at
    ``step_at``) raises that floor by its own size.  Both halves are the same
    rule -- the rendered echo is never closer to its reference than the
    configured floor -- and both are reachable: the walk alone can take a 10 ms
    bulk delay to 0 in two downward steps, and after a negative step the NET
    delay is the walk minus the step, which goes negative outright, an echo
    arriving before the reference that caused it. Inside the frozen matched
    filter's first bin the delay estimate simply stops moving, so either one
    costs the rest of the sequence.

    ⚠ The raised floor applies only where the step is actually in the signal.
    A walk value holds until the next boundary, so it is raised exactly when
    the stretch it governs reaches the step; before that the net delay is the
    walk alone and flooring it at ``bulk_delay_ms_min - step`` would make the
    whole first part of the walk one-sided, by up to the largest step the
    config admits.  ``step_at`` indexes the INPUT and the boundaries index the
    delayed output, so the instant compared against them is
    ``step_at + current``: the stretch's own delay is how much later the step
    reaches the output, and comparing without it raises the floor up to one
    bulk delay early.
    """
    steps = rng.randint(cfg.getint('echo_path', 'jitter_steps_min'),
                        cfg.getint('echo_path', 'jitter_steps_max'))
    span_min = int(sr * cfg.getfloat('echo_path', 'jitter_ms_min') / 1000)
    span_max = int(sr * cfg.getfloat('echo_path', 'jitter_ms_max') / 1000)
    fade = max(1, int(sr * cfg.getfloat('echo_path', 'jitter_fade_sec')))
    floor = int(sr * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    raised = floor - min(0, int(applied_step))
    n_samples = x.shape[-1]

    out = delay_signal(x, base_delay)
    current = base_delay
    boundaries = sorted(rng.randint(fade, max(fade + 1, n_samples - fade))
                        for _ in range(steps))
    for index, boundary in enumerate(boundaries):
        held_until = (boundaries[index + 1] if index + 1 < len(boundaries)
                      else n_samples)
        shift = rng.randint(span_min, max(span_min, span_max))
        current = current + (-shift if rng.random() < 0.5 else shift)
        arrives = step_at + current
        current = max(raised if 0 <= arrives < held_until else floor, current)
        out = _crossfade(out, delay_signal(x, current), boundary, fade)
    return out


def _convolver(signal: torch.Tensor):
    """``fftconvolve(signal, kernel)``, with the signal transformed once.

    A trajectory convolves ONE signal with K responses, and the shared operand
    is the long one: re-transforming it per position is most of the work.  The
    arithmetic is ``fftconvolve``'s, term for term -- the same power-of-two
    length, the same product, the same inverse and the same trim -- so the
    result is bit-identical; only the transform of ``signal`` is remembered,
    per length, because two responses of different lengths do not share one.
    """
    spectra: Dict[int, torch.Tensor] = {}

    def convolve(kernel: torch.Tensor) -> torch.Tensor:
        n = signal.shape[-1] + kernel.shape[-1] - 1
        fft_size = 1
        while fft_size < n:
            fft_size *= 2
        if fft_size not in spectra:
            spectra[fft_size] = torch.fft.rfft(signal, n=fft_size)
        out = torch.fft.irfft(
            spectra[fft_size] * torch.fft.rfft(kernel, n=fft_size), n=fft_size)
        return out[:signal.shape[-1]]

    return convolve


def _crossfade(a: torch.Tensor, b: torch.Tensor, at: int, fade: int) -> torch.Tensor:
    """Blend from ``a`` to ``b`` over ``fade`` samples starting at ``at``.

    A hard switch would leave a step discontinuity in the echo -- a broadband
    click the model can key on to detect the event, which is not a cue any real
    echo-path change provides.
    """
    n = a.shape[-1]
    at = int(min(max(at, 0), n))
    fade = int(min(max(fade, 0), n - at))
    weight = torch.ones(n)
    if fade > 0:
        weight[at:at + fade] = torch.linspace(1.0, 0.0, fade)
    weight[at + fade:] = 0.0
    return a * weight + b * (1.0 - weight)


def _sample_waypoints(room_rirs: Sequence[str], count: int,
                      rng: random.Random, avoid=frozenset()) -> List[str]:
    """``count`` distinct RIR files from ONE room.

    ``avoid`` is a preference, not a constraint: a room with just enough files
    for the loudspeaker's trajectory must still be able to place the near
    talker, and sharing a position with the loudspeaker is a physically real
    arrangement whereas leaving the room is not.
    """
    preferred = [path for path in room_rirs if path not in avoid]
    chosen = rng.sample(preferred, min(count, len(preferred)))
    if len(chosen) < count:
        remaining = [path for path in room_rirs if path not in chosen]
        chosen += rng.sample(remaining, min(count - len(chosen), len(remaining)))
    return chosen


def _ensure_far_activity_after(far_runs, at: int, n_samples: int,
                               horizon: int, min_run: int):
    """Guarantee far-end activity INSIDE ``[at, at + horizon)``.

    Returns the (possibly extended) run list and the window that must stay
    active, or ``None`` when the sequence has too little left after ``at`` to
    carry an utterance at all.

    ⚠ The window is returned WHETHER OR NOT a run had to be added.  The caller
    steers a reference dropout away from it, and a dropout zeroes the reference
    just as thoroughly over a naturally scheduled run as over a repaired one --
    "the schedule already covered it" is not protection, it is the thing that
    needs protecting.

    ⚠ Coverage is counted inside the window, not anywhere after ``at``.  A
    schedule that falls silent at the event and resumes ten seconds later
    satisfies "some far activity later in the sequence" while leaving the event
    itself invisible, which is the case this exists to remove.

    ⚠ The demand is capped at what is left of the sequence, and abandoned when
    that remainder is shorter than ``min_run`` -- the shortest run
    ``_render_talker`` can place an utterance into.  Buying the full amount at
    the very end would mean starting the run BEFORE the event, and a run below
    that floor is dropped by the talker renderer while the caller would still
    be treating it as guaranteed.
    """
    at = max(0, at)
    need = min(horizon, n_samples - at)
    if need < max(1, min_run):
        return far_runs, None
    window = (at, at + need)
    covered = sum(max(0, min(end, window[1]) - max(start, window[0]))
                  for start, end in far_runs)
    if covered >= need:
        return far_runs, window
    return _merge_runs(list(far_runs) + [window]), window


def _dropout_placement(n_chunks: int, count: int, chunk_samples: int,
                       protect=None):
    """Where a ``count``-chunk reference dropout may start.

    ``protect`` is the ``(start, stop)`` sample window that must keep its
    reference (the stretch behind an echo-path event), or ``None``.  A dropout
    zeroes WHOLE chunks, so the window is protected by the chunks it touches;
    the dropout SHRINKS to fit beside them rather than being placed on top of
    them, because a shorter idle stretch is still an idle stretch while an
    event with its echo zeroed is a label describing audio that is not there.

    ⚠ Only a sequence with no room at all for both -- the protected chunks are
    every chunk -- keeps its drawn length, and there it is placed where it
    zeroes the FEWEST protected samples.  A window that straddles the only
    interior chunk boundary of a two-chunk sequence protects both chunks while
    lying almost entirely in one of them, so this is the difference between
    losing a few milliseconds of the window and losing all of it.  The measured
    per-chunk ``far_active`` records the outcome either way.
    """
    count = max(1, count)
    keep_out = None
    if protect is not None:
        keep_out = range(protect[0] // chunk_samples,
                         (protect[1] - 1) // chunk_samples + 1)
    for length in range(count, 0, -1):
        starts = [start for start in range(0, n_chunks - length + 1)
                  if keep_out is None
                  or start + length <= keep_out[0] or start > keep_out[-1]]
        if starts:
            return starts, length
    candidates = list(range(0, max(1, n_chunks - count + 1)))
    zeroed = [max(0, min((start + count) * chunk_samples, protect[1])
                  - max(start * chunk_samples, protect[0]))
              for start in candidates]
    least = min(zeroed)
    return [start for start, samples in zip(candidates, zeroed)
            if samples == least], count


def _draw_delay_step(n_samples: int, bulk_delay: int, sr: int,
                     rng: random.Random, cfg) -> Tuple[int, int]:
    """One signed playout-buffer step: its size in samples and when it lands.

    ⚠ A negative step must leave the echo at least ``bulk_delay_ms_min`` behind
    the reference, and the magnitude is redrawn inside that headroom rather
    than clipped to it.  Both halves matter:

    * clipping to the bulk delay puts the post-step echo at 0 ms, where the
      frozen matched filter has no bin to lock onto -- the delay estimate then
      stays at the pre-step value and the persisted linear error is the
      uncancelled capture for the whole rest of the sequence;
    * clipping also produces steps smaller than ``delay_step_ms_min`` while the
      metadata still says ``delay_step``, which makes the impairment
      indistinguishable from ``delay_jitter``.

    When the sequence's own bulk delay leaves less headroom than
    ``delay_step_ms_min``, the step is positive instead: the echo can always be
    pushed later, and a sequence that keeps the impairment label has to carry a
    step inside the configured range.

    ⚠ The floor holds END TO END, not just here: ``delay_jitter`` is drawn
    independently and walks the same delay afterwards, so it takes this step
    and raises its own lower clamp by it, so the rendered echo is at least
    ``bulk_delay_ms_min`` behind the reference.

    ⚠ With ONE exception, and it is a physical one: ``sro`` resamples the
    played path before the delay is applied, so a positive offset advances the
    echo by ppm x elapsed time -- 1.8 ms at the largest configured offset over
    a 30 s sequence. Two clocks running apart do move the echo, which is what
    that impairment is; the floor bounds everything the delay draws do.
    """
    low = int(sr * cfg.getfloat('echo_path', 'delay_step_ms_min') / 1000)
    high = max(low, int(sr * cfg.getfloat('echo_path', 'delay_step_ms_max') / 1000))
    headroom = bulk_delay - int(
        sr * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    negative = rng.random() < 0.5 and headroom >= low
    magnitude = rng.randint(low, min(high, headroom) if negative else high)
    margin = int(sr * DELAY_STEP_EDGE_MARGIN_SEC)
    if n_samples <= 2 * margin:
        return 0, -1
    return (-magnitude if negative else magnitude,
            rng.randrange(margin, n_samples - margin))


def _shift_signal(x: torch.Tensor, samples: int) -> torch.Tensor:
    """Delay (``samples`` > 0) or advance (< 0), keeping the length."""
    if samples >= 0:
        return delay_signal(x, samples)
    advance = min(-samples, x.shape[-1])
    return F.pad(x[advance:], (0, advance))


def _gain_track_summary(gain_db: Optional[torch.Tensor]) -> Optional[dict]:
    """A compact description of the level walk, not the walk itself.

    Per-chunk metadata is copied once per chunk, so the full per-sample track
    would be the largest field in the corpus's description by three orders of
    magnitude while answering no question the spread does not.
    """
    if gain_db is None:
        return None
    quantiles = torch.quantile(
        gain_db, torch.tensor([0.05, 0.5, 0.95], dtype=gain_db.dtype))
    return {'p5_db': float(quantiles[0]), 'median_db': float(quantiles[1]),
            'p95_db': float(quantiles[2]),
            'span_db': float(gain_db.max() - gain_db.min())}


def _scale_to_active_dbfs(x: torch.Tensor, sr: int, dbfs: float) -> torch.Tensor:
    if float(x.abs().max()) < 1e-9:
        return x
    return x * ((10.0 ** (dbfs / 20.0)) / max(active_rms(x, sr), 1e-10))


def _scale_to_ratio(x: torch.Tensor, reference: torch.Tensor, sr: int,
                    ratio_db: float) -> torch.Tensor:
    """Scale ``x`` so ``active_rms(x) / active_rms(reference)`` is ``ratio_db``.

    Active RMS, not plain RMS, because these signals are mostly silence: a
    plain-RMS SER would make a talker who pauses more sound quieter, so the
    recorded ser_db would stop describing what is audible during speech.  This
    is the same definition ``AINR/dataset_gen/dataset.py`` uses for SNR.
    """
    if float(x.abs().max()) < 1e-9 or float(reference.abs().max()) < 1e-9:
        return x
    target = active_rms(reference, sr) * (10.0 ** (ratio_db / 20.0))
    return x * (target / max(active_rms(x, sr), 1e-10))


def _scale_pair_to_ratio(primary: torch.Tensor, paired: torch.Tensor,
                         reference: torch.Tensor, sr: int,
                         ratio_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one gain to full/early near-speech versions."""
    if (float(primary.abs().max()) < 1e-9
            or float(reference.abs().max()) < 1e-9):
        return primary, paired
    target = active_rms(reference, sr) * (10.0 ** (ratio_db / 20.0))
    gain = target / max(active_rms(primary, sr), 1e-10)
    return primary * gain, paired * gain


def _scale_pair_to_active_dbfs(primary: torch.Tensor, paired: torch.Tensor,
                               sr: int, dbfs: float
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    if float(primary.abs().max()) < 1e-9:
        return primary, paired
    gain = 10.0 ** (dbfs / 20.0) / max(active_rms(primary, sr), 1e-10)
    return primary * gain, paired * gain
