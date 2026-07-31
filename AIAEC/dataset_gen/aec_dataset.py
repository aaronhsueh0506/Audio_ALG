"""AEC scenario simulator: renders parent sequences of 7 aligned stems.

THE SIGNAL MODEL THIS FILE EXISTS TO PRODUCE
--------------------------------------------
    Y = S + N + D        microphone   (S near speech, N local noise, D echo)
    X                    far-end reference
    D_hat                optional frozen-linear echo estimate
    E     = Y - D_hat    linear error                       <-- RES+NR input
    R     = D - D_hat    residual echo -- audit only, not target

The seven stems stay separated so direct AEC, end-to-end AEC+NR, and
frozen-linear RES+NR routes can share one source corpus without changing task
targets. ``model_views.py`` owns the authoritative per-model mapping.

Everything a model may need is derivable from the stems:
    Y = mic_postclip     X = far_render     D = echo
    S = near_speech      N = local_noise
    S_early = near_target (DeepVQE dereverberation target only)

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
from .aec_features import STEM_ORDER, alpha_from_tau
from .manifest import SourcePools


__all__ = [
    'NONLINEAR_MODELS',
    'SCENARIOS',
    'AecSequenceRenderer',
    'DeviceModel',
    'RenderedSequence',
    'SequencePlan',
    'apply_agc',
    'apply_loudspeaker_nonlinearity',
    'device_for_id',
    'plan_sequences',
    'resample_by_ratio',
    'simulate_codec',
    'stable_seed',
]


# The scenario vocabulary.  A chunk's `scenario` metadata is always one of
# these, so a downstream filter such as `meta['scenario'] == 'double_talk'`
# never silently matches nothing because of a typo.
SCENARIOS = (
    'far_only',
    'near_only',
    'double_talk',
    'ref_dropout',
    'echo_path_change',
    'nonlinear_spk',
    'clipping_agc',
    'delay_jitter',
    'sro',
    'codec_mismatch',
)

# Scenarios whose defining event occupies the WHOLE sequence.  The others are
# localised in time and get a per-chunk label instead -- see _chunk_scenario.
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
        # A small loudspeaker rolls off well below Nyquist; expressed as a
        # fraction of Nyquist so the 48 kHz variant needs no new number.
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
        mean = talk_sec if active else gap_sec
        # Clamped so one unlucky exponential draw cannot swallow a whole 60 s
        # sequence in a single silence.
        seconds = min(max(rng.expovariate(1.0 / mean), 0.15), mean * 4.0)
        end = min(position + max(int(sr * seconds), int(sr * 0.15)), n_samples)
        if active:
            runs.append((position, end))
        position = end
        active = not active
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


@dataclasses.dataclass
class RenderedSequence:
    stems: torch.Tensor              # (7, T) float32, channel order = STEM_ORDER
    chunk_meta: List[dict]
    chunk_samples: int


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

    weights = _scenario_weights(cfg)
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
        plans.append(SequencePlan(
            sequence_id=sequence_id,
            n_chunks=n_chunks,
            scenario=rng.choices(names, weights=probabilities, k=1)[0],
            seed=stable_seed(seed, 'sequence', split, sequence_id),
        ))
        total_sec += n_chunks * chunk_sec
        sequence_id += 1
    return plans


def _scenario_weights(cfg: configparser.ConfigParser) -> Dict[str, float]:
    weights = {}
    for name in SCENARIOS:
        value = cfg.getfloat('scenarios', f'p_{name}', fallback=0.0)
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"[scenarios] p_{name} must be finite and >= 0, got {value}")
        if value > 0:
            weights[name] = value
    if not weights:
        raise ValueError("[scenarios] every probability is zero; nothing to generate")
    missing = sorted(set(SCENARIOS) - set(weights))
    if missing:
        # Not fatal -- an ablation corpus is a legitimate thing to want -- but a
        # silently absent scenario is a hole nobody finds until evaluation.
        print(f"  ⚠ zero-probability scenarios, absent from this corpus: {missing}")
    return weights


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
        self.chunk_samples = int(round(cfg.getfloat('sequence', 'chunk_sec') * self.sr))
        if self.chunk_samples <= 0:
            raise ValueError("[sequence] chunk_sec is too small for this sample rate")

        self.snr_values = parse_snr_values(cfg.get('levels', 'snr_values'))
        self.devices = {
            device_id: device_for_id(device_id, cfg, self.corpus_seed, self.sr)
            for device_id in pools.devices
        }
        # Per-device mic cascades, built once: identical for every sequence that
        # uses the device, which is the entire point of a device identity.
        self._mic_chains: Dict[str, list] = {}

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
                       rng: random.Random) -> Tuple[torch.Tensor, List[str]]:
        """Place whole utterances inside the active runs.

        Gating a continuous stream with a mask would cut words in half and leave
        a step discontinuity at every boundary; the model would then learn to
        treat that click as the cue for talk onset.
        """
        pool = self.pools.speech_files
        out = torch.zeros(n_samples)
        used: List[str] = []
        fade = max(1, int(self.sr * self.cfg.getfloat('activity', 'talk_fade_sec')))
        for start, end in runs:
            length = end - start
            if length <= 2 * fade:
                continue
            path = pool[rng.randrange(len(pool))]
            try:
                segment = self._load_audio(path, rng, length, loop=False)
            except Exception:
                continue
            ramp = torch.linspace(0.0, 1.0, fade)
            segment[:fade] *= ramp
            segment[-fade:] *= ramp.flip(0)
            out[start:end] = segment
            used.append(path)
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
        return noise, ids or ['none']

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
        scenario = plan.scenario

        # --- sources -------------------------------------------------------
        device = self.devices[self.pools.devices[rng.randrange(len(self.pools.devices))]]
        if scenario == 'nonlinear_spk':
            # The scenario IS strong distortion, so drawing a linear device
            # would make the label a lie a consumer cannot detect.
            distorting = sorted(
                (d for d in self.devices.values() if d.nonlinear != 'linear'),
                key=lambda d: d.device_id,
            )
            if distorting:
                device = distorting[rng.randrange(len(distorting))]

        room = self.pools.rooms[rng.randrange(len(self.pools.rooms))]
        room_rirs = self.pools.rirs_by_room[room]
        # ⚠ The loudspeaker and the near talker are in the SAME room.  RIRs from
        # different rooms would hand the model an acoustic "this is echo" cue
        # that no real device ever has.
        echo_rir_path = room_rirs[rng.randrange(len(room_rirs))]
        near_pool = [p for p in room_rirs if p != echo_rir_path] or room_rirs
        near_rir_path = near_pool[rng.randrange(len(near_pool))]

        has_far = scenario != 'near_only'
        has_near = scenario != 'far_only'

        # --- talker activity -----------------------------------------------
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
        if scenario == 'double_talk' and far_runs:
            near_runs = _force_overlap(far_runs, near_runs, rng, sr, cfg)

        far_speech, far_paths = (self._render_talker(far_runs, n_samples, rng)
                                 if has_far else (torch.zeros(n_samples), []))
        near_dry, near_paths = (self._render_talker(near_runs, n_samples, rng)
                                if has_near else (torch.zeros(n_samples), []))

        # --- far-end reference X -------------------------------------------
        far_render = _scale_to_active_dbfs(
            far_speech, sr,
            rng.uniform(cfg.getfloat('levels', 'far_level_dbfs_min'),
                        cfg.getfloat('levels', 'far_level_dbfs_max')))

        # Reference dropout, chosen as WHOLE chunks so that a chunk labelled
        # 'ref_dropout' is unambiguously an idle chunk.
        dropout_chunks = set()
        if scenario == 'ref_dropout' and has_far and n_chunks > 1:
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
        if scenario == 'codec_mismatch':
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
        if scenario == 'nonlinear_spk':
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

        if scenario == 'sro':
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
        delay_jitter = scenario == 'delay_jitter'
        played = (_apply_jittered_delay(played, bulk_delay, sr, rng, cfg)
                  if delay_jitter else delay_signal(played, bulk_delay))

        # --- echo path -----------------------------------------------------
        switch_chunk = -1
        if scenario == 'echo_path_change':
            second_path = _pick_path_change_rir(room_rirs, echo_rir_path,
                                                self.pools, rng)
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

        # ⚠ ONE common scale across ALL SEVEN stems. Scaling only the mic would
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
        if scenario == 'clipping_agc' or rng.random() < cfg.getfloat('mic', 'p_clipping'):
            mic_postclip = apply_clipping(mic_postclip,
                                          cfg.getfloat('mic', 'clip_snr_min'),
                                          cfg.getfloat('mic', 'clip_snr_max'))
            clipped = True
        if scenario == 'clipping_agc' or rng.random() < cfg.getfloat('mic', 'p_agc'):
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

        stems = torch.stack([far_render, echo, near_speech, near_target, noise,
                             mic_preclip, mic_postclip]).to(torch.float32).contiguous()
        if stems.shape[0] != len(STEM_ORDER):
            raise AssertionError("stem stack does not match STEM_ORDER")

        chunk_meta = self._build_meta(
            plan, stems, device, room, rir_id, erl_db, ser_db, snr_db,
            bulk_delay, delay_jitter, sro_ppm, clipped, agc, dropout_chunks,
            switch_chunk, noise_ids,
            near_speaker=self.pools.speaker_of.get(near_paths[0], '') if near_paths else '',
            far_speaker=self.pools.speaker_of.get(far_paths[0], '') if far_paths else '',
        )
        return RenderedSequence(stems=stems, chunk_meta=chunk_meta,
                                chunk_samples=self.chunk_samples)

    def _build_meta(self, plan, stems, device, room, rir_id, erl_db, ser_db,
                    snr_db, bulk_delay, delay_jitter, sro_ppm, clipped, agc,
                    dropout_chunks, switch_chunk, noise_ids,
                    near_speaker, far_speaker) -> List[dict]:
        # ⚠ ser_db / snr_db / erl_db are SEQUENCE-level: they describe how the
        # 20-60 s sequence was set up, measured over its whole duration.  A
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
                'scenario': _chunk_scenario(
                    plan.scenario, chunk_index, dropout_chunks, switch_chunk,
                    far_active=float(far[window].pow(2).mean().sqrt()) > threshold,
                    near_active=float(near[window].pow(2).mean().sqrt()) > threshold,
                ),
                'sequence_scenario': plan.scenario,
                'split': self.pools.split,
            })
        return meta


def _chunk_scenario(sequence_scenario: str, chunk_index: int, dropout_chunks,
                    switch_chunk: int, far_active: bool, near_active: bool) -> str:
    """Per-chunk label, which is not always the sequence's label.

    ⚠ A 40 s 'ref_dropout' sequence is mostly NOT a dropout, and an
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
    if sequence_scenario in WHOLE_SEQUENCE_SCENARIOS:
        return sequence_scenario
    if far_active and near_active:
        return 'double_talk'
    if far_active:
        return 'far_only'
    return 'near_only'


def _force_overlap(far_runs, near_runs, rng, sr, cfg):
    """Guarantee genuine double talk instead of hoping two chains collide."""
    overlap_p = cfg.getfloat('activity', 'dt_overlap_p')
    frac_min = cfg.getfloat('activity', 'dt_overlap_frac_min')
    frac_max = cfg.getfloat('activity', 'dt_overlap_frac_max')
    added = list(near_runs)
    for start, end in far_runs:
        if rng.random() >= overlap_p:
            continue
        length = end - start
        window = int(length * rng.uniform(frac_min, frac_max))
        if window < int(sr * 0.2):
            continue
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


def _pick_path_change_rir(room_rirs, current, pools, rng):
    """The post-change RIR: same room if it has another, otherwise any room.

    A same-room switch is a moved loudspeaker or an opened door; a cross-room
    switch is what a device sees when it is picked up and carried.  Both are
    real, so the fallback is not a compromise.
    """
    others = [p for p in room_rirs if p != current]
    if others:
        return others[rng.randrange(len(others))]
    every = sorted(p for paths in pools.rirs_by_room.values()
                   for p in paths if p != current)
    return every[rng.randrange(len(every))] if every else current


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
