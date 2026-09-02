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
    LinearAecContract,
    linear_aec_contract_from_config,
    materialize_linear_error,
)
from .manifest import SourcePools, config_hash


__all__ = [
    'ECHO_MODES',
    'IMPAIRMENTS',
    'NONLINEAR_MODELS',
    'SCENARIOS',
    'TALK_MODES',
    'AecSequenceRenderer',
    'DeviceModel',
    'RenderedSequence',
    'SequencePlan',
    'apply_agc',
    'apply_loudspeaker_nonlinearity',
    'chunk_samples_from_config',
    'device_for_id',
    'plan_sequences',
    'resolve_sequence_plan',
    'resample_by_ratio',
    'simulate_codec',
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
    'nonlinear_spk',
    'clipping_agc',
    'delay_jitter',
    'sro',
    'codec_mismatch',
)

# Derived, not restated: the label vocabulary IS the three axes projected onto
# one string, and spelling it out again is how the two drift apart.  'normal'
# is not a label -- it is the absence of an echo-mode event.
# ⚠ A chunk's `scenario` is one of these, and so is a sequence's
# `sequence_scenario`, but the two do not draw from the same subset:
# 'duplex_random' is a sequence-level talk mode that no chunk is ever labelled.
SCENARIOS = (TALK_MODES
             + tuple(mode for mode in ECHO_MODES if mode != 'normal')
             + IMPAIRMENTS)

# Impairments that act on the echo path, so they have nothing to act on when
# there is no far end or no acoustic return.  Named once because two places
# depend on agreeing about it: plan_sequences() STRIPS these from such plans
# and resolve_sequence_plan() REJECTS them, and a new echo-path impairment
# added to only one of the two is either never generated or always fatal.
ECHO_PATH_IMPAIRMENTS = frozenset({
    'echo_path_change', 'nonlinear_spk', 'delay_jitter', 'sro',
    'codec_mismatch',
})

# The impairments [complex_cases] p_dt_stress_combo forces together.  Named
# because three places must agree on it: the planner that sets it, the
# renderer's forced-edge-overlap gate that detects it, and the CLI census that
# counts it.  Two of those three fail SILENTLY when the definition moves.
DT_STRESS_IMPAIRMENTS = frozenset({
    'echo_path_change', 'nonlinear_spk', 'clipping_agc',
})

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


def device_for_id(device_id: str, cfg: configparser.ConfigParser,
                  corpus_seed: int, sr: int) -> DeviceModel:
    """Derive a device's fixed characteristics from its id."""
    models = [m.strip() for m in cfg.get('devices', 'nonlinear_models').split(',')
              if m.strip()]
    unknown = sorted(set(models) - set(NONLINEAR_MODELS))
    if unknown:
        raise ValueError(f"[devices] unknown nonlinear_models {unknown}; "
                         f"choose from {list(NONLINEAR_MODELS)}")
    if not models:
        raise ValueError("[devices] nonlinear_models is empty")

    rng = random.Random(stable_seed(corpus_seed, 'device', device_id))
    nyquist = sr / 2.0
    lp_frac = rng.uniform(cfg.getfloat('devices', 'speaker_lp_nyquist_frac_min'),
                          cfg.getfloat('devices', 'speaker_lp_nyquist_frac_max'))
    return DeviceModel(
        device_id=device_id,
        nonlinear=models[rng.randrange(len(models))],
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

    layer_sections = ('talk_modes', 'echo_modes', 'impairments',
                      'complex_cases')
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
                "a layered config must carry all four planner sections; "
                f"found {', '.join('[' + name + ']' for name in present_layers)} "
                "but not "
                f"{', '.join('[' + name + ']' for name in absent)}; without "
                "the section every probability in it silently reads as 0")
        required_options = {
            'talk_modes': tuple(f'p_{name}' for name in TALK_MODES),
            'echo_modes': ('p_ref_dropout', 'p_far_active_no_echo'),
            'impairments': tuple(f'p_{name}' for name in IMPAIRMENTS),
            'complex_cases': ('p_dt_stress_combo',),
        }
        missing_options = [
            f'[{section}] {option}'
            for section, options in required_options.items()
            for option in options
            if not cfg.has_option(section, option)
        ]
        if missing_options:
            raise ValueError(
                "a layered config must explicitly set every planner option; "
                "missing " + ', '.join(missing_options))
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
        p_dt_stress = _probability(
            cfg, 'complex_cases', 'p_dt_stress_combo')
    else:
        # Compatibility for an older copied config or a focused test config.
        # It deliberately preserves the old mutually-exclusive semantics;
        # shipped configs use the layered sections above.
        weights = _named_weights(cfg, 'scenarios', SCENARIOS)
        names = list(weights)
        probabilities = [weights[name] for name in names]

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
                draw = random.Random(stable_seed(
                    sequence_seed, 'echo_mode')).random()
                if draw < p_far_active_no_echo:
                    echo_mode = 'far_active_no_echo'
                elif draw < p_far_active_no_echo + p_ref_dropout:
                    echo_mode = 'ref_dropout'

            impairments = set()
            for name, probability in impairment_p.items():
                if random.Random(stable_seed(
                        sequence_seed, 'impairment', name)).random() < probability:
                    impairments.add(name)

            # Echo-path impairments have no signal to act on in near-only or
            # far-active/no-echo sequences. Capture clipping/AGC remains valid
            # in both and is therefore intentionally retained.
            if not has_far or echo_mode == 'far_active_no_echo':
                impairments.difference_update(ECHO_PATH_IMPAIRMENTS)

            # Pure independent draws make the exact DT + moving/nonlinear/
            # clipped intersection too rare to teach in a 200 h campaign.
            # This conditional draw creates a measurable tail without turning
            # every ordinary DT example into an adversarial one.
            if (talk_mode == 'double_talk'
                    and echo_mode == 'normal'
                    and random.Random(stable_seed(
                        sequence_seed, 'dt_stress_combo')).random() < p_dt_stress):
                impairments.update(DT_STRESS_IMPAIRMENTS)

            plans.append(SequencePlan(
                sequence_id=sequence_id,
                n_chunks=n_chunks,
                scenario=talk_mode,
                seed=sequence_seed,
                talk_mode=talk_mode,
                echo_mode=echo_mode,
                impairments=tuple(sorted(impairments)),
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
    if talk_mode == 'near_only' and echo_mode != 'normal':
        raise ValueError(f"near_only cannot use echo_mode={echo_mode}")
    if echo_mode == 'far_active_no_echo' and (
            set(impairments) & ECHO_PATH_IMPAIRMENTS):
        raise ValueError(
            "far_active_no_echo cannot carry echo-path impairments")
    return talk_mode, echo_mode, tuple(sorted(set(impairments)))


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
        self.devices = {
            device_id: device_for_id(device_id, cfg, self.corpus_seed, self.sr)
            for device_id in pools.devices
        }
        # Per-device mic cascades, built once: identical for every sequence that
        # uses the device, which is the entire point of a device identity.
        self._mic_chains: Dict[str, list] = {}
        # Rooms eligible for 'echo_path_change' (>= 2 RIR files), fixed by the
        # manifest's RIR pool and never affected by which sequence is
        # rendering -- computed once here rather than re-filtered on every
        # 'echo_path_change' sequence.
        self._path_change_rooms = rooms_eligible_for_path_change(pools)

    # ---------------- source loading ----------------

    def _load_audio(self, path: str, rng: random.Random,
                    n_samples: int, loop: bool) -> torch.Tensor:
        audio, file_sr = torchaudio.load(path)
        audio = audio[0].float()
        if audio.numel() == 0:
            raise RuntimeError(f"empty audio: {path}")
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)
        if audio.shape[-1] < n_samples:
            if loop:
                audio = audio.repeat(n_samples // audio.shape[-1] + 1)
            else:
                audio = F.pad(audio, (0, n_samples - audio.shape[-1]))
        start = rng.randint(0, audio.shape[-1] - n_samples)
        return audio[start:start + n_samples].clone()

    def _load_rir_pair(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, file_sr = torchaudio.load(path)
        audio = audio[0].float()
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)
        target_rir, full_rir = prepare_rir(
            audio, self.sr,
            late_offset_ms=self.cfg.getfloat('rir', 'early_rir_ms'),
            pre_delay_keep_ms=self.cfg.getfloat('rir', 'pre_delay_keep_ms'),
            rt60=self.pools.rt60.get(path, 0.4),
        )
        return target_rir, full_rir

    def _load_rir(self, path: str) -> torch.Tensor:
        # Echo must contain the complete acoustic path.  The paired early RIR
        # is used only for DeepVQE's dereverberation target on the near talker.
        return self._load_rir_pair(path)[1]

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
        fade = max(1, int(self.sr * self.cfg.getfloat('activity', 'talk_fade_sec')))
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

    def _switch_point(self, n_chunks: int, rng: random.Random) -> int:
        # Away from the very edges, so the chunk labelled 'echo_path_change'
        # really does contain audio from both paths.
        lo = max(1, n_chunks // 4)
        hi = max(lo, (3 * n_chunks) // 4)
        chunk = rng.randint(lo, hi)
        return min(chunk * self.chunk_samples + rng.randrange(self.chunk_samples),
                   n_chunks * self.chunk_samples - 1)

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
        legacy_scenario = plan.scenario if plan.talk_mode is None else None

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

        room_pool = self.pools.rooms
        if 'echo_path_change' in impairments:
            # The post-change RIR (_pick_path_change_rir below) must stay in
            # this same room -- see the invariant note just below -- so this
            # scenario may only draw a room that actually has a second RIR
            # file to switch to. gen_aec_dataset.py already validates this
            # eligibility once, up front, before any sequence is rendered;
            # this is the belt for direct AecSequenceRenderer callers (tests,
            # rematerialize_linear_aec.py) that skip that preflight check.
            room_pool = self._path_change_rooms
            if not room_pool:
                raise RuntimeError(
                    "scenario 'echo_path_change' requires a room with >= 2 "
                    f"RIR files; none of the {len(self.pools.rooms)} "
                    "available room(s) qualify"
                )
        room = room_pool[rng.randrange(len(room_pool))]
        room_rirs = self.pools.rirs_by_room[room]
        # ⚠ The loudspeaker and the near talker are ALWAYS in the SAME room --
        # RIRs from different rooms would hand the model an acoustic "this is
        # echo" cue that no real device ever has. This holds for
        # echo_path_change too: the room-eligibility filter above guarantees
        # _pick_path_change_rir's post-change RIR never has to leave it.
        echo_rir_path = room_rirs[rng.randrange(len(room_rirs))]
        near_pool = [p for p in room_rirs if p != echo_rir_path] or room_rirs
        near_rir_path = near_pool[rng.randrange(len(near_pool))]

        has_far = talk_mode != 'near_only'
        has_near = talk_mode != 'far_only'

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
                DT_STRESS_IMPAIRMENTS <= impairments
                and cfg.getboolean(
                    'activity', 'dt_force_edge_overlap', fallback=False)
            )
            near_runs = _force_overlap(
                far_runs, near_runs, rng, sr, cfg, force_edges=force_edges)

        far_speech, far_paths = (
            self._render_talker(far_runs, n_samples, rng,
                                self.pools.far_speech_files,
                                loop=echo_mode == 'far_active_no_echo')
            if has_far else (torch.zeros(n_samples), []))
        near_dry, near_paths = (
            self._render_talker(near_runs, n_samples, rng, self.pools.speech_files)
            if has_near else (torch.zeros(n_samples), []))

        # --- far-end reference X -------------------------------------------
        far_render = _scale_to_active_dbfs(
            far_speech, sr,
            rng.uniform(cfg.getfloat('levels', 'far_level_dbfs_min'),
                        cfg.getfloat('levels', 'far_level_dbfs_max')))

        # Reference dropout, chosen as WHOLE chunks so that a chunk labelled
        # 'ref_dropout' is unambiguously an idle chunk.
        dropout_chunks = set()
        if echo_mode == 'ref_dropout' and has_far and n_chunks > 1:
            hi = min(cfg.getint('dropout', 'ref_dropout_chunks_max'), n_chunks - 1)
            lo = min(cfg.getint('dropout', 'ref_dropout_chunks_min'), hi)
            count = rng.randint(max(1, lo), max(1, hi))
            first = rng.randint(0, n_chunks - count)
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
        bulk_delay = rng.randint(
            int(sr * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000),
            int(sr * cfg.getfloat('echo_path', 'bulk_delay_ms_max') / 1000))
        delay_jitter = 'delay_jitter' in impairments
        played = (_apply_jittered_delay(played, bulk_delay, sr, rng, cfg)
                  if delay_jitter else delay_signal(played, bulk_delay))

        # --- echo path -----------------------------------------------------
        switch_chunk = -1
        if 'echo_path_change' in impairments:
            second_path = _pick_path_change_rir(room_rirs, echo_rir_path, rng)
            switch = self._switch_point(n_chunks, rng)
            echo_raw = _crossfade(
                fftconvolve(played, self._load_rir(echo_rir_path)),
                fftconvolve(played, self._load_rir(second_path)),
                switch,
                max(1, int(sr * cfg.getfloat('echo_path', 'path_change_fade_sec'))))
            rir_id = (f"{self.pools.rir_id_of.get(echo_rir_path, echo_rir_path)}|"
                      f"{self.pools.rir_id_of.get(second_path, second_path)}")
            switch_chunk = switch // self.chunk_samples
        else:
            echo_raw = fftconvolve(played, self._load_rir(echo_rir_path))
            rir_id = self.pools.rir_id_of.get(echo_rir_path, echo_rir_path)

        if has_near:
            near_target_rir, near_full_rir = self._load_rir_pair(near_rir_path)
            near_raw = fftconvolve(near_dry, near_full_rir)
            near_target_raw = fftconvolve(near_dry, near_target_rir)
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
        erl_db = rng.uniform(cfg.getfloat('levels', 'erl_db_min'),
                             cfg.getfloat('levels', 'erl_db_max'))
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

        chunk_meta = self._build_meta(
            plan, stems, device, room, rir_id, erl_db, ser_db, snr_db,
            bulk_delay, delay_jitter, sro_ppm, clipped, agc, dropout_chunks,
            switch_chunk, noise_ids,
            near_speaker=self.pools.speaker_of.get(near_paths[0], '') if near_paths else '',
            far_speaker=self.pools.far_speaker_of.get(far_paths[0], '') if far_paths else '',
        )
        return RenderedSequence(
            stems=stems, chunk_meta=chunk_meta, chunk_samples=self.chunk_samples,
            linear_aec_contract=self.linear_aec_contract.as_dict(),
            audit={'echo': echo.to(torch.float32).contiguous(),
                  'noise': noise.to(torch.float32).contiguous(),
                  'mic_preclip': mic_preclip.to(torch.float32).contiguous(),
                  'echo_estimate': echo_estimate.contiguous()},
        )

    def _build_meta(self, plan, stems, device, room, rir_id, erl_db, ser_db,
                    snr_db, bulk_delay, delay_jitter, sro_ppm, clipped, agc,
                    dropout_chunks, switch_chunk, noise_ids,
                    near_speaker, far_speaker) -> List[dict]:
        # Re-derived from `plan` rather than threaded down as four more
        # positional arguments: resolve_sequence_plan is pure and O(1), and
        # this call already carries enough of them to align by eye.
        talk_mode, echo_mode, impairments = resolve_sequence_plan(plan)
        legacy_scenario = plan.scenario if plan.talk_mode is None else None
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
        meta = []
        for chunk_index in range(plan.n_chunks):
            at = chunk_index * self.chunk_samples
            window = slice(at, at + self.chunk_samples)
            meta.append({
                'sequence_id': int(plan.sequence_id),
                'chunk_index': int(chunk_index),
                # 'speaker_id' is the NEAR talker -- the signal that must
                # survive.  '' means this sequence has no near talker.
                'speaker_id': near_speaker,
                'far_speaker_id': far_speaker,
                'noise_id': '+'.join(noise_ids),
                'rir_id': rir_id,
                'room_id': room,
                'device_id': device.device_id,
                'ser_db': float(ser_db),
                'snr_db': float(snr_db),
                'erl_db': float(erl_db),
                'bulk_delay_samples': int(bulk_delay),
                'delay_jitter': bool(delay_jitter),
                'sro_ppm': float(sro_ppm),
                'nonlinear': device.nonlinear,
                'clipped': bool(clipped),
                'agc': bool(agc),
                'talk_mode': talk_mode,
                'echo_mode': echo_mode,
                'impairments': list(impairments),
                'echo_path_change': 'echo_path_change' in impairments,
                'codec_mismatch': 'codec_mismatch' in impairments,
                'manifest_version': self.pools.manifest_version,
                'linear_aec_contract_hash': self.linear_aec_contract.fingerprint(),
                'config_hash': self.config_hash,
                # config_hash alone would not identify a render: --seed lives
                # outside config.ini, so sequence_seed is recorded beside it.
                # Both fields are in-process audit data only; neither is
                # persisted in the WAV-only corpus or checked by --resume.
                'sequence_seed': int(plan.seed),
                'scenario': _chunk_scenario(
                    echo_mode, legacy_scenario,
                    chunk_index, dropout_chunks, switch_chunk,
                    far_active=float(far[window].pow(2).mean().sqrt()) > threshold,
                    near_active=float(near[window].pow(2).mean().sqrt()) > threshold,
                ),
                # Compatibility summary. New consumers must use the
                # orthogonal fields above rather than infer combinations from
                # this one string.
                'sequence_scenario': plan.scenario,
                'split': self.pools.split,
            })
        return meta


def _chunk_scenario(echo_mode: str,
                    legacy_scenario: Optional[str], chunk_index: int,
                    dropout_chunks, switch_chunk: int, far_active: bool,
                    near_active: bool) -> str:
    """Per-chunk label, which is not always the sequence's label.

    ⚠ A 'ref_dropout' parent sequence is mostly NOT a dropout, and an
    'echo_path_change' sequence contains exactly one chunk where the path
    changes.  Labelling every chunk with the sequence's intent would make the
    honest test "every ref_dropout clip has a silent reference" fail, and would
    let a dropout-conditioned loss term train on chunks whose reference is fully
    active.  So the label marks the chunks that really are the event, and the
    rest are labelled by what they actually contain.  ``sequence_scenario``
    keeps the sequence-level intent for anyone who needs it.
    """
    if chunk_index in dropout_chunks:
        return 'ref_dropout'
    if chunk_index == switch_chunk:
        return 'echo_path_change'
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


def _force_overlap(far_runs, near_runs, rng, sr, cfg, *, force_edges=False):
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
            window = max(window, min(length, floor))
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
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in sorted(runs)[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _apply_jittered_delay(x, base_delay, sr, rng, cfg):
    """Piecewise bulk delay with crossfades, i.e. playout buffer glitches.

    Discrete steps rather than a slow ramp: a dropped or duplicated playout
    buffer moves the echo by a whole block at once, which is a completely
    different disturbance from an SRO's continuous drift.  Conflating the two
    would let one scenario stand in for the other, and it cannot.
    """
    steps = rng.randint(cfg.getint('echo_path', 'jitter_steps_min'),
                        cfg.getint('echo_path', 'jitter_steps_max'))
    span_min = int(sr * cfg.getfloat('echo_path', 'jitter_ms_min') / 1000)
    span_max = int(sr * cfg.getfloat('echo_path', 'jitter_ms_max') / 1000)
    fade = max(1, int(sr * cfg.getfloat('echo_path', 'jitter_fade_sec')))
    n_samples = x.shape[-1]

    out = delay_signal(x, base_delay)
    current = base_delay
    boundaries = sorted(rng.randint(fade, max(fade + 1, n_samples - fade))
                        for _ in range(steps))
    for boundary in boundaries:
        shift = rng.randint(span_min, max(span_min, span_max))
        current = max(0, current + (-shift if rng.random() < 0.5 else shift))
        out = _crossfade(out, delay_signal(x, current), boundary, fade)
    return out


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


def rooms_eligible_for_path_change(pools: SourcePools) -> List[str]:
    """Rooms with >= 2 RIR files -- the only ones 'echo_path_change' may draw.

    A pure function of the manifest's RIR pool, so it is safe to call once
    per renderer (``AecSequenceRenderer.__init__``) and once more, before any
    rendering starts, as a preflight check in ``gen_aec_dataset.py`` -- a
    corpus too RIR-sparse to support the scenario then fails in under a
    second instead of partway through a multi-hour run, whenever a worker
    happens to draw the first 'echo_path_change' sequence.
    """
    return [r for r in pools.rooms if len(pools.rirs_by_room[r]) >= 2]


def _pick_path_change_rir(room_rirs, current, rng):
    """The post-change RIR: a different file in the SAME room.

    A same-room switch (moved loudspeaker, opened door) is the only kind
    modelled -- a cross-room switch would leave the near talker's RIR behind
    in the old room, which is exactly the acoustic "this is echo" leak the
    same-room invariant exists to prevent. The caller only reaches this
    scenario with a room that has >= 2 RIR files, so ``others`` is guaranteed
    non-empty; there is no fallback to fall back to.
    """
    others = [p for p in room_rirs if p != current]
    return others[rng.randrange(len(others))]


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
