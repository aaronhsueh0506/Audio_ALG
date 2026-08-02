"""Shared signal-grid and stem primitives for every AEC model project.

WHY THIS LIVES HERE
-------------------
Three model projects will consume the same packed AEC corpus and be compared
against each other.  Anything that decides *what the model sees* -- the STFT
grid, the window convention, the channel order of the stem tensor, and how
consecutive chunks of one sequence reach a batch -- is therefore part of the
comparison protocol, not an implementation detail of any one trainer.

``AINR/dataset_gen/loader.py`` already records what happens when such things
are copied into each ``train.py``: the held-out fraction drifted from 5% to
10% and the two models being compared were silently trained on different
corpora.  The same failure mode is available here and is worse, because a
channel-order or window mismatch does not look like a bug -- it looks like a
model that is merely worse.

So: one ``AecGrid``, one ``stft``/``istft``, one ``STEM_ORDER``.  A project
that needs something different is opting out of the comparison and should say
so in its own README.

⚠ Nothing in this module derives a time constant from a fixed hop count.  Time
constants arrive in SECONDS and go through ``alpha_from_tau``; a 48 kHz grid
with hop 512 then produces the same physical smoothing as 16 kHz with hop 256
without a single edit.
"""

import configparser
import dataclasses
import functools
import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler


__all__ = [
    'BASE_STEM_ORDER',
    'STEM_ORDER',
    'AecGrid',
    'AecStems',
    'SequenceChunkSampler',
    'alpha_from_tau',
    'istft',
    'lane_reset_mask',
    'sqrt_hann_window',
    'stft',
]


# ============================================================
# Stem channel order
# ============================================================
#
# Fixed, and duplicated nowhere else.  The packed shards record this same tuple
# under 'stems' so a shard that disagrees can be rejected instead of silently
# feeding a stem into the wrong slot.
#
# ``echo`` (D), ``local_noise`` (N) and ``mic_preclip`` (S+N+D before
# clipping/AGC) remain audit signals only -- no model task targets echo
# cancellation without denoising any more, so N never needs to reach a
# trainer as its own channel (see ``RenderedSequence.audit`` in
# aec_dataset.py). ``linear_error`` is different: it is the real frozen
# PBFDKF output E=Y-D_hat consumed by the three RES+NR candidates. It is
# materialized over the COMPLETE parent sequence before chunking, never
# manufactured from oracle D and never recomputed inside a training epoch.
BASE_STEM_ORDER = (
    'far_render',
    'near_speech',
    'near_target',
    'mic_postclip',
)
STEM_ORDER = BASE_STEM_ORDER + ('linear_error',)


# ============================================================
# Signal grid
# ============================================================

@dataclasses.dataclass(frozen=True)
class AecGrid:
    """The time-frequency grid, derived once from config.

    Frozen because a grid that can be mutated after a model is built is a grid
    that will disagree with the weights it trained.  ``n_freqs`` and
    ``frame_rate`` are real fields rather than properties so that
    ``dataclasses.asdict(grid)`` produces a complete record for a checkpoint
    contract -- a contract that stored only ``n_fft`` would happily accept a
    checkpoint trained at a different hop.
    """

    sr: int
    n_fft: int
    win_len: int
    hop_len: int
    n_freqs: int = dataclasses.field(init=False)
    frame_rate: float = dataclasses.field(init=False)

    def __post_init__(self):
        for name in ('sr', 'n_fft', 'win_len', 'hop_len'):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"AecGrid.{name} must be a positive int, got {value!r}")
        if self.win_len != self.n_fft:
            raise ValueError(
                f"win_len ({self.win_len}) must equal n_fft ({self.n_fft}): "
                "the AEC dataset contract forbids hidden FFT zero-padding"
            )
        # ⚠ 50% overlap is not decoration.  sqrt-Hann analysis + sqrt-Hann
        # synthesis sums to unity only at hop == win_len/2; at any other hop the
        # WOLA reconstruction acquires a periodic amplitude ripple that the model
        # then has to learn to undo.  Change this only together with the window.
        if self.win_len % 2 != 0 or self.hop_len != self.win_len // 2:
            raise ValueError(
                f"hop_len ({self.hop_len}) must be win_len/2 ({self.win_len // 2}) "
                f"for the sqrt-Hann COLA convention"
            )
        object.__setattr__(self, 'n_freqs', self.n_fft // 2 + 1)
        object.__setattr__(self, 'frame_rate', self.sr / self.hop_len)

    @classmethod
    def from_config(cls, cfg: configparser.ConfigParser,
                    section: str = 'signal') -> 'AecGrid':
        """Build the grid from a ``[signal]`` section.

        ``win_len``/``hop_len`` fall back off ``n_fft`` exactly as the existing
        model configs do, so a 48 kHz variant is ``n_fft = 1024`` and nothing
        else.
        """
        n_fft = cfg.getint(section, 'n_fft')
        win_len = cfg.getint(section, 'win_len', fallback=n_fft)
        return cls(
            sr=cfg.getint(section, 'sr'),
            n_fft=n_fft,
            win_len=win_len,
            hop_len=cfg.getint(section, 'hop_len', fallback=win_len // 2),
        )

    def n_frames(self, n_samples: int, center: bool = True) -> int:
        """Frames a signal of ``n_samples`` produces on this grid.

        ⚠ Call this instead of writing a frame count into a config or a test.
        A hardcoded 188 is correct for 3 s at 16 kHz/hop 256 and wrong for
        every other grid, including the 48 kHz variant this project must
        support by config change alone.
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {n_samples}")
        if center:
            return n_samples // self.hop_len + 1
        if n_samples < self.win_len:
            return 0
        return (n_samples - self.win_len) // self.hop_len + 1

    def window(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """The shared analysis/synthesis window.  ⚠ Do not mutate the result."""
        return sqrt_hann_window(self.win_len, device=device, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _cached_window(win_len: int, device_key: str, dtype: torch.dtype) -> torch.Tensor:
    # periodic=True is torch's default and is the DFT-correct choice; a
    # symmetric Hann breaks COLA at 50% overlap by one sample's worth of
    # amplitude, which shows up as a low-level buzz in the reconstruction.
    return torch.hann_window(
        win_len, periodic=True, device=torch.device(device_key), dtype=dtype
    ).sqrt()


def sqrt_hann_window(win_len: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """sqrt-Hann (periodic).  Cached, so ⚠ callers must treat it as read-only."""
    key = 'cpu' if device is None else str(device)
    return _cached_window(int(win_len), key, dtype)


def stft(x: torch.Tensor, grid: AecGrid, center: bool = True) -> torch.Tensor:
    """``(..., T)`` real waveform -> ``(..., n_freqs, n_frames)`` complex.

    Leading dimensions are preserved, so a ``(B, 7, T)`` stem tensor transforms
    in one call and keeps its stem axis.
    """
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    spec = torch.stft(
        flat, n_fft=grid.n_fft, hop_length=grid.hop_len, win_length=grid.win_len,
        window=grid.window(device=x.device, dtype=x.dtype),
        center=center, return_complex=True,
    )
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft(spec: torch.Tensor, grid: AecGrid, length: Optional[int] = None,
          center: bool = True) -> torch.Tensor:
    """Inverse of :func:`stft`.  ``length`` recovers the exact input length."""
    if not torch.is_complex(spec):
        raise TypeError("istft expects a complex spectrum; use torch.view_as_complex")
    lead = spec.shape[:-2]
    flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1])
    wav = torch.istft(
        flat, n_fft=grid.n_fft, hop_length=grid.hop_len, win_length=grid.win_len,
        window=grid.window(device=spec.device, dtype=spec.real.dtype),
        center=center, length=length,
    )
    return wav.reshape(*lead, wav.shape[-1])


def alpha_from_tau(tau_sec: float, hop_len: int, sr: int) -> float:
    """One-pole smoothing coefficient for a time constant given in SECONDS.

    ``alpha = exp(-hop_len / (sr * tau_sec))``

    ⚠ This is the only place an EMA coefficient may be produced.  A literal
    ``0.92`` in a model or a config is a frame-rate-dependent constant in
    disguise: it means 220 ms at 16 kHz/hop 256 and 73 ms at 48 kHz/hop 512,
    so the 48 kHz variant would quietly become a different algorithm.

    Pass ``hop_len=1`` for a per-sample smoother (mic AGC, envelope followers).
    """
    if sr <= 0 or hop_len <= 0:
        raise ValueError(f"sr and hop_len must be positive, got sr={sr}, hop={hop_len}")
    if not math.isfinite(tau_sec) or tau_sec < 0.0:
        raise ValueError(f"tau_sec must be finite and non-negative, got {tau_sec}")
    if tau_sec == 0.0:
        return 0.0     # no memory: alpha 0 means "take the new value"
    return math.exp(-hop_len / (sr * tau_sec))


def frames_from_seconds(seconds: float, frame_rate: float,
                        minimum: int = 0) -> int:
    """Duration in SECONDS -> whole frames on this grid.

    ⚠ The counterpart of :func:`alpha_from_tau` for durations that are counted
    in frames rather than smoothed: lookahead, filter spans, gate memories.
    Same reasoning, same hazard.  A config that says ``taps = 16`` has baked in
    a frame rate: 16 frames is 256 ms at 16 kHz/hop 256 but only 171 ms at
    48 kHz/hop 512, so the 48 kHz variant silently covers a shorter echo tail
    than the one it was tuned for.  Every duration in a config.ini is therefore
    written in seconds and converted here.

    ``minimum`` is for spans where zero is structurally meaningless -- a filter
    with 0 taps or a gate with 0 frames of memory is not a degenerate setting,
    it is an invalid one -- so the caller asks for at least 1.
    """
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"duration must be finite and non-negative, got {seconds}")
    if frame_rate <= 0.0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    return max(minimum, int(round(seconds * frame_rate)))


# ============================================================
# Stem accessors
# ============================================================

class AecStems:
    """Named access to the ``(..., 5, T)`` stem tensor.

    Channel 0 is far_render and channel 1 is near_speech.  Getting those two
    the wrong way round produces a model that trains, converges, and cancels
    the talker -- so no project indexes this tensor by number.
    """

    __slots__ = ('_data', '_order', '_index')

    def __init__(self, data: torch.Tensor,
                 order: Sequence[str] = STEM_ORDER):
        order = tuple(order)
        if data.ndim < 2:
            raise ValueError(f"stem tensor must be (..., n_stems, T), got {tuple(data.shape)}")
        if data.shape[-2] != len(order):
            raise ValueError(
                f"stem tensor has {data.shape[-2]} channels but the declared "
                f"order has {len(order)}: {order}"
            )
        if set(order) != set(STEM_ORDER):
            raise ValueError(f"unknown stem order {order}; expected {STEM_ORDER}")
        self._data = data
        self._order = order
        self._index = {name: i for i, name in enumerate(order)}

    # -- raw stems, exactly as stored -----------------------------------
    def stem(self, name: str) -> torch.Tensor:
        try:
            return self._data[..., self._index[name], :]
        except KeyError:
            raise KeyError(f"no stem {name!r}; have {self._order}") from None

    @property
    def far_render(self) -> torch.Tensor:
        """The far-end signal as the device rendered it -- the AEC reference."""
        return self.stem('far_render')

    @property
    def near_speech(self) -> torch.Tensor:
        """S at the mic, i.e. already through the near-talker's room RIR."""
        return self.stem('near_speech')

    @property
    def near_target(self) -> torch.Tensor:
        """Early/direct S target for models that also dereverberate."""
        return self.stem('near_target')

    @property
    def mic_postclip(self) -> torch.Tensor:
        """The mic signal a model actually receives."""
        return self.stem('mic_postclip')

    @property
    def linear_error(self) -> torch.Tensor:
        """Frozen PBFDKF output E=Y-D_hat, materialized before chunking."""
        return self.stem('linear_error')

    # -- signal-model aliases -------------------------------------------
    #
    #   Y = S + N + D        (mic)
    #   X                    (far-end reference)
    #   D_hat                a linear/model echo estimate when available
    #   E = Y - D_hat        by subtraction, never a mask on Y
    #   R = D - D_hat        residual echo; emerges, never a target
    #
    # ``D`` (true echo), ``N`` (local noise) and ``R`` (oracle residual echo)
    # deliberately have no accessor: no model task targets echo cancellation
    # without denoising any more, so N is audit-only (see
    # ``RenderedSequence.audit`` in aec_dataset.py) like D and R always were.
    # E is the materialized *linear error*, not R. D_hat is derived exactly
    # from the two persisted waveforms rather than stored twice.
    @property
    def Y(self) -> torch.Tensor:
        return self.mic_postclip

    @property
    def X(self) -> torch.Tensor:
        return self.far_render

    @property
    def S(self) -> torch.Tensor:
        return self.near_speech

    @property
    def E(self) -> torch.Tensor:
        return self.linear_error

    @property
    def D_hat(self) -> torch.Tensor:
        return self.mic_postclip - self.linear_error

    def as_tensor(self) -> torch.Tensor:
        return self._data

    @property
    def order(self) -> Tuple[str, ...]:
        return self._order

    def to(self, *args, **kwargs) -> 'AecStems':
        return AecStems(self._data.to(*args, **kwargs), self._order)

    def __repr__(self):
        return f"AecStems(shape={tuple(self._data.shape)}, order={self._order})"


# ============================================================
# Sequence-aware batching
# ============================================================

def lane_reset_mask(chunk_indices) -> torch.Tensor:
    """``True`` where a lane starts a new sequence and must reset its state.

    ``SequenceChunkSampler`` guarantees that lane *k* walks one sequence in
    order, so ``chunk_index == 0`` is exactly the moment its recurrent state
    stops being valid.
    """
    idx = torch.as_tensor(chunk_indices)
    return idx == 0


class SequenceChunkSampler(Sampler):
    """Batch sampler whose lanes carry consecutive chunks of one sequence.

    Batch *b* holds one chunk per lane; batch *b+1* holds the NEXT chunk of the
    same sequence in each lane.  A trainer that keeps a per-lane recurrent state
    across batches therefore sees one whole configured parent sequence unbroken.

    ⚠ Resetting recurrent state every chunk is the default a naive DataLoader
    gives you, and it hides precisely the behaviours this corpus was built to
    expose: convergence from cold, recovery after an echo-path change, and
    long-term drift.  A model that is never asked to remember anything past one
    chunk cannot be shown to fail at any of them.

    Lane boundaries: when a lane finishes a sequence it starts the next one, and
    the first chunk of that sequence has ``chunk_index == 0``.  Use
    :func:`lane_reset_mask` on the batch's ``chunk_index`` metadata to zero the
    state for those lanes.
    """

    def __init__(self, sequence_ids: Sequence[int], chunk_indices: Sequence[int],
                 n_lanes: int, shuffle: bool = True, seed: int = 42,
                 drop_last: bool = True):
        if len(sequence_ids) != len(chunk_indices):
            raise ValueError(
                f"sequence_ids ({len(sequence_ids)}) and chunk_indices "
                f"({len(chunk_indices)}) must be the same length"
            )
        if n_lanes <= 0:
            raise ValueError(f"n_lanes must be positive, got {n_lanes}")

        by_sequence: Dict[int, List[Tuple[int, int]]] = {}
        for dataset_index, (sid, cidx) in enumerate(zip(sequence_ids, chunk_indices)):
            by_sequence.setdefault(int(sid), []).append((int(cidx), dataset_index))

        # Sort by chunk_index rather than trusting shard order.  A shard that
        # was packed out of order would otherwise feed the model a sequence
        # backwards, which looks like a convergence failure, not a data bug.
        self._sequences: Dict[int, List[int]] = {}
        for sid, pairs in by_sequence.items():
            pairs.sort()
            expected = list(range(len(pairs)))
            if [c for c, _ in pairs] != expected:
                raise ValueError(
                    f"sequence {sid} has chunk_index {[c for c, _ in pairs]}, "
                    f"expected {expected}: the corpus is missing chunks or packs "
                    f"one sequence across a filtered subset"
                )
            self._sequences[sid] = [i for _, i in pairs]

        if n_lanes > len(self._sequences):
            raise ValueError(
                f"n_lanes ({n_lanes}) exceeds the number of sequences "
                f"({len(self._sequences)}); every lane needs its own sequence "
                f"or two lanes would replay the same state"
            )

        self.n_lanes = int(n_lanes)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._schedule = self._build_schedule(self.epoch)

    @classmethod
    def from_dataset(cls, dataset, n_lanes: int, **kwargs) -> 'SequenceChunkSampler':
        """Build from anything exposing ``sequence_ids()``/``chunk_indices()``."""
        return cls(dataset.sequence_ids(), dataset.chunk_indices(), n_lanes, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle which sequences land in which lane.  Call once per epoch."""
        self.epoch = int(epoch)
        self._schedule = self._build_schedule(self.epoch)

    def _build_schedule(self, epoch: int) -> List[List[int]]:
        order = list(self._sequences.keys())
        if self.shuffle:
            # A dedicated generator, so the lane layout depends only on
            # (seed, epoch) and not on how much randomness the rest of setup
            # happened to draw first -- the same reasoning as
            # locality_preserving_random_split in AINR/dataset_gen/loader.py.
            generator = torch.Generator().manual_seed(self.seed * 1000003 + epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]

        lanes: List[List[int]] = [[] for _ in range(self.n_lanes)]
        for position, sid in enumerate(order):
            lanes[position % self.n_lanes].extend(self._sequences[sid])

        if self.drop_last:
            # Every batch is exactly n_lanes wide, so a trainer can allocate its
            # per-lane state once.  A short final batch would silently reshape
            # the state tensor instead.
            steps = min(len(lane) for lane in lanes)
            return [[lane[step] for lane in lanes] for step in range(steps)]

        steps = max(len(lane) for lane in lanes)
        return [
            [lane[step] for lane in lanes if step < len(lane)]
            for step in range(steps)
        ]

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._schedule)

    def __len__(self) -> int:
        return len(self._schedule)
