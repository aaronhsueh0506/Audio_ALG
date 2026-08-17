"""Shared stateful cells for frame-by-frame AIAEC inference.

Every cell reuses the weights of an existing offline module and replays its
exact arithmetic on one (or a few) new time frames, carrying the time context
in explicit state tensors.  Zero-initialised history equals the offline
left-only zero padding, so a stream started from reset produces bit-for-bit
the same causal computation as one whole-utterance ``forward`` -- the only
intended exceptions are documented on the model's own ``forward_stream``
(Align-CRUSE warm-up, DeepFilterNet's two-hop output delay).

These cells are a Python deployment *reference*: they define the state
inventory and per-invocation I/O contract that an NPU/C port must reproduce.
They are not optimised.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from AIAEC.aiaec_common import CausalConv2d, FrameDelayAttention, GlobalDelayAttention


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def assert_streaming_ready(model: nn.Module) -> None:
    """Refuse to stream a model whose normalisation would be utterance-level.

    BatchNorm in training mode takes statistics over the (B, T, F) extent --
    that is a whole-utterance operation and silently breaks frame-by-frame
    equivalence.  Everything downstream assumes eval-mode running stats.
    """
    if model.training:
        raise RuntimeError(
            "streaming requires model.eval(); call it before create_stream_state"
        )
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.training:
                raise RuntimeError(
                    f"BatchNorm module {name!r} is in training mode; "
                    "streaming output would not match offline inference"
                )


# ---------------------------------------------------------------------------
# convolution cells
# ---------------------------------------------------------------------------

class StreamConv2dCell:
    """Streaming context for one time-extended Conv2d.

    Wraps a conv whose offline call pads ``time_left = (kt - 1) * dt`` zeros on
    the left (and, when ``lookahead > 0``, ``lookahead`` frames on the right).
    The cell keeps the ``time_left`` most recent *input* frames; ``step``
    accepts ``[B, C, T_new, F]`` and returns the outputs that become
    computable.  With ``lookahead == 0`` that is exactly ``T_new`` frames; with
    ``lookahead == k`` the first ``k`` pushed frames produce nothing and every
    later push yields the output for ``k`` frames earlier.
    """

    def __init__(self, conv: nn.Conv2d, kt: int, dt: int,
                 freq_left: int, freq_right: int, lookahead: int = 0):
        if conv.stride[0] != 1:
            raise ValueError("streaming cells support time stride 1 only")
        if lookahead < 0 or lookahead > kt - 1:
            raise ValueError("lookahead must lie in [0, kt-1]")
        self.conv = conv
        self.kt = kt
        self.dt = dt
        self.freq_left = freq_left
        self.freq_right = freq_right
        self.lookahead = lookahead
        # Offline pads `time_left` zeros before the first frame -- that is the
        # initial history.  Between steps the cell must retain the full kernel
        # span minus one, which is larger than `time_left` when lookahead > 0.
        self.time_left = (kt - 1) * dt - lookahead
        self.retain = (kt - 1) * dt
        self._history: Optional[Tensor] = None

    @classmethod
    def from_causal(cls, module: CausalConv2d) -> "StreamConv2dCell":
        freq_total = (module.kf - 1) * module.df
        freq_left = freq_total // 2
        return cls(module.conv, module.kt, module.dt,
                   freq_left, freq_total - freq_left)

    def reset(self) -> None:
        self._history = None

    def step(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("StreamConv2dCell expects [B,C,T,F]")
        if self._history is None:
            b, c, _, f = x.shape
            self._history = x.new_zeros(b, c, self.time_left, f)
        window = torch.cat((self._history, x), dim=2)
        if self.retain > 0:
            self._history = window[:, :, -self.retain:]
        else:
            self._history = window[:, :, :0]
        span = (self.kt - 1) * self.dt
        if window.shape[2] <= span:
            # Not enough context yet (possible only while lookahead frames
            # are still pending at stream start).
            b = x.shape[0]
            return x.new_zeros(b, self.conv.out_channels, 0,
                               self._out_freq(x.shape[-1]))
        padded = F.pad(window, (self.freq_left, self.freq_right, 0, 0))
        return self.conv(padded)

    def _out_freq(self, f: int) -> int:
        eff = f + self.freq_left + self.freq_right
        kf = self.conv.kernel_size[1]
        df = self.conv.dilation[1]
        return (eff - (kf - 1) * df - 1) // self.conv.stride[1] + 1

    def state_tensors(self) -> Dict[str, Tensor]:
        return {} if self._history is None else {"history": self._history}


class StreamModuleCell:
    """Run a whole offline block frame-by-frame when its time kernel is 1.

    Blocks whose only time-extended member is a leading CausalConv2d (the
    common ``conv -> norm -> act`` pattern) are handled by pairing a
    :class:`StreamConv2dCell` for the conv with the block's remaining
    per-frame modules.
    """

    def __init__(self, conv_cell: StreamConv2dCell,
                 per_frame: Iterable[nn.Module]):
        self.conv_cell = conv_cell
        self.per_frame = list(per_frame)

    def reset(self) -> None:
        self.conv_cell.reset()

    def step(self, x: Tensor) -> Tensor:
        y = self.conv_cell.step(x)
        for module in self.per_frame:
            y = module(y)
        return y

    def state_tensors(self) -> Dict[str, Tensor]:
        return self.conv_cell.state_tensors()


# ---------------------------------------------------------------------------
# recurrence / ring cells
# ---------------------------------------------------------------------------

class StreamGRUCell:
    """Hidden-state carrier for an ``nn.GRU`` running along the time axis."""

    def __init__(self, gru: nn.GRU):
        if not gru.batch_first:
            raise ValueError("streaming GRU cells expect batch_first=True")
        if gru.bidirectional:
            raise ValueError(
                "a bidirectional GRU along time cannot stream; "
                "frequency-axis bidirectional GRUs are per-frame and need no cell"
            )
        self.gru = gru
        self._hidden: Optional[Tensor] = None

    def reset(self) -> None:
        self._hidden = None

    def step(self, x: Tensor) -> Tensor:
        y, self._hidden = self.gru(x, self._hidden)
        return y

    def state_tensors(self) -> Dict[str, Tensor]:
        return {} if self._hidden is None else {"hidden": self._hidden}


class DelayRingCell:
    """Streaming twin of :func:`aiaec_common.causal_delay_stack`.

    Keeps the ``delays`` most recent frames of a ``[B, C, F]`` feature; slot
    ``d`` of the returned ``[B, C, 1, delays, F]`` stack holds ``x[t - d]``
    (zero before stream start, matching the offline left zero pad).
    """

    def __init__(self, delays: int):
        if delays <= 0:
            raise ValueError("delays must be positive")
        self.delays = delays
        self._ring: Optional[Tensor] = None

    def reset(self) -> None:
        self._ring = None

    def step(self, frame: Tensor) -> Tensor:
        if frame.ndim == 4:
            if frame.shape[2] != 1:
                raise ValueError("DelayRingCell takes one frame per step")
            frame = frame[:, :, 0]
        if self._ring is None:
            b, c, f = frame.shape
            self._ring = frame.new_zeros(b, c, self.delays, f)
        self._ring = torch.cat(
            (frame.unsqueeze(2), self._ring[:, :, : self.delays - 1]), dim=2,
        )
        return self._ring.unsqueeze(2)

    def state_tensors(self) -> Dict[str, Tensor]:
        return {} if self._ring is None else {"ring": self._ring}


class FrameDelayAttentionCell:
    """Streaming twin of :class:`aiaec_common.FrameDelayAttention`.

    State: key ring, value ring (``max_delay_frames`` frames each) and the
    score conv's four-frame logit history.  The per-frame arithmetic keeps the
    offline multiply-then-sum order, so outputs match ``forward`` exactly.
    """

    def __init__(self, attention: FrameDelayAttention):
        self.attention = attention
        self.key_ring = DelayRingCell(attention.max_delay_frames)
        self.value_ring = DelayRingCell(attention.max_delay_frames)
        self.score_cell = StreamConv2dCell.from_causal(attention.score)

    def reset(self) -> None:
        self.key_ring.reset()
        self.value_ring.reset()
        self.score_cell.reset()

    def step(self, mic: Tensor, far: Tensor) -> Tuple[Tensor, Tensor]:
        att = self.attention
        q = att.query(mic)                            # [B,H,1,F]
        k_delayed = self.key_ring.step(att.key(far))  # [B,H,1,D,F]
        logits = (q.unsqueeze(3) * k_delayed).sum(dim=-1)   # [B,H,1,D]
        logits = self.score_cell.step(logits).squeeze(1)    # [B,1,D]
        distribution = torch.softmax(logits, dim=-1)
        v_delayed = self.value_ring.step(att.value(far))
        aligned = (v_delayed * distribution[:, None, :, :, None]).sum(dim=3)
        return aligned, distribution

    def state_tensors(self) -> Dict[str, Tensor]:
        out = {}
        for prefix, cell in (("key", self.key_ring), ("value", self.value_ring),
                             ("score", self.score_cell)):
            for name, tensor in cell.state_tensors().items():
                out[f"{prefix}_{name}"] = tensor
        return out


class GlobalDelayAttentionCell:
    """Streaming twin of Align-CRUSE's ``causal_running`` delay attention.

    State: projected-key ring, far-feature value ring, the *undecayed* score
    accumulator (``cumsum`` integrator) and an absolute frame counter.

    Equivalence caveat, by construction: the offline forward masks delays with
    ``observable = arange(D) < T`` using the FINAL utterance length ``T``.  A
    stream cannot know ``T``, so this cell assumes ``T >= D`` (any utterance of
    at least ``D`` hops).  For such inputs the stream matches offline exactly;
    for shorter utterances the offline mask differs and so does the output.
    ``paper_global`` mode reduces over the whole utterance and is rejected.
    """

    def __init__(self, attention: GlobalDelayAttention):
        if attention.mode != "causal_running":
            raise ValueError(
                "GlobalDelayAttention paper_global mode is utterance-level "
                "and cannot stream; retrain or switch the checkpoint mode"
            )
        self.attention = attention
        self.key_ring = DelayRingCell(attention.max_delay_frames)
        self.value_ring = DelayRingCell(attention.max_delay_frames)
        self._score_sum: Optional[Tensor] = None
        self.frame_index: Optional[Tensor] = None

    def reset(self) -> None:
        self.key_ring.reset()
        self.value_ring.reset()
        self._score_sum = None
        self.frame_index = None

    def step(self, mic: Tensor, far: Tensor) -> Tuple[Tensor, Tensor]:
        att = self.attention
        b = mic.shape[0]
        q = att.mic_pool(mic).permute(0, 2, 1, 3).reshape(b, 1, -1)
        k = att.far_pool(far).permute(0, 2, 1, 3).reshape(b, 1, -1)
        q = att.query(q)[:, 0]                      # [B,P]
        k = att.key(k)[:, 0]                        # [B,P]
        k_delayed = self.key_ring.step(k.unsqueeze(-1))[:, :, 0, :, 0]  # [B,P,D]
        scores = (q.unsqueeze(-1) * k_delayed).sum(dim=1)               # [B,D]
        delay = torch.arange(att.max_delay_frames, device=mic.device)
        if self.frame_index is None:
            self.frame_index = torch.zeros((), dtype=torch.long,
                                           device=mic.device)
        valid = (self.frame_index >= delay).to(scores.dtype)
        if self._score_sum is None:
            self._score_sum = torch.zeros_like(scores)
        self._score_sum = self._score_sum + scores * valid
        distribution = torch.softmax(self._score_sum, dim=-1)
        v_delayed = self.value_ring.step(far[:, :, 0])                  # [B,C,1,D,F]
        aligned = (v_delayed * distribution[:, None, None, :, None]).sum(dim=3)
        self.frame_index = self.frame_index + 1
        return aligned, distribution

    def state_tensors(self) -> Dict[str, Tensor]:
        out = {}
        for prefix, cell in (("key", self.key_ring), ("value", self.value_ring)):
            for name, tensor in cell.state_tensors().items():
                out[f"{prefix}_{name}"] = tensor
        if self._score_sum is not None:
            out["score_sum"] = self._score_sum
        if self.frame_index is not None:
            out["frame_index"] = self.frame_index
        return out


# ---------------------------------------------------------------------------
# stateful STFT / WOLA replicating the offline center=True contract
# ---------------------------------------------------------------------------

class StreamSTFT:
    """Incremental ``torch.stft(center=True, pad_mode='reflect')``.

    Deployment decision (design doc section 7): the first version reproduces
    the offline centered timing, paying ``n_fft/2`` samples of lookahead, so a
    frame is emitted only once its trailing half-window has arrived.  Feed
    arbitrary-sized sample chunks; ``push`` returns zero or more complex
    ``[B, n_freqs]`` frames; ``flush`` applies the trailing reflect pad and
    emits the remaining frames (total ``L // hop + 1``).
    """

    def __init__(self, n_fft: int, hop: int, window: Tensor):
        if window.numel() != n_fft:
            raise ValueError("window length must equal n_fft")
        self.n_fft = n_fft
        self.hop = hop
        self.half = n_fft // 2
        self.window = window
        self._raw: Optional[Tensor] = None       # pending raw samples pre-pad
        self._buf: Optional[Tensor] = None       # padded-domain samples
        self._tail: Optional[Tensor] = None      # last half+1 raw samples
        self._started = False
        self._flushed = False
        self.samples_seen = 0

    def reset(self) -> None:
        self._raw = None
        self._buf = None
        self._tail = None
        self._started = False
        self._flushed = False
        self.samples_seen = 0

    def _emit(self) -> List[Tensor]:
        frames = []
        while self._buf is not None and self._buf.shape[-1] >= self.n_fft:
            segment = self._buf[:, : self.n_fft]
            frames.append(torch.fft.rfft(segment * self.window, dim=-1))
            self._buf = self._buf[:, self.hop:]
        return frames

    def push(self, samples: Tensor) -> List[Tensor]:
        if self._flushed:
            raise RuntimeError(
                "StreamSTFT.push after flush: the trailing reflect pad is "
                "already emitted; reset() before starting a new stream"
            )
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        self.samples_seen += samples.shape[-1]
        tail = samples if self._tail is None else torch.cat(
            (self._tail, samples), dim=-1)
        self._tail = tail[:, -(self.half + 1):]
        if not self._started:
            self._raw = samples if self._raw is None else torch.cat(
                (self._raw, samples), dim=-1)
            # The reflect prefix mirrors samples 1..half, so it exists only
            # once half+1 raw samples have arrived.
            if self._raw.shape[-1] < self.half + 1:
                return []
            prefix = self._raw[:, 1: self.half + 1].flip(-1)
            self._buf = torch.cat((prefix, self._raw), dim=-1)
            self._raw = None
            self._started = True
        else:
            self._buf = torch.cat((self._buf, samples), dim=-1)
        return self._emit()

    def flush(self) -> List[Tensor]:
        if self._flushed:
            raise RuntimeError("StreamSTFT.flush called twice; reset() first")
        self._flushed = True
        if not self._started:
            if self._raw is None or self._raw.shape[-1] == 0:
                return []
            raise ValueError(
                f"stream shorter than half a window ({self.half + 1} samples); "
                "the centered contract needs at least that much audio"
            )
        # Trailing reflect pad mirrors the half samples before the last one.
        # Total frames come out to L // hop + 1 exactly, because the emit
        # loop slides while a full window exists over L + 2*half samples.
        suffix = self._tail[:, :-1].flip(-1)
        self._buf = torch.cat((self._buf, suffix), dim=-1)
        return self._emit()

    def state_tensors(self) -> Dict[str, Tensor]:
        out = {}
        if self._raw is not None:
            out["raw"] = self._raw
        if self._buf is not None:
            out["buffer"] = self._buf
        if self._tail is not None:
            out["tail"] = self._tail
        return out


class StreamISTFT:
    """Incremental WOLA inverse of :class:`StreamSTFT` (center=True).

    Matches ``torch.istft``: synthesis window multiply, overlap-add, division
    by the accumulated squared window, then the leading ``n_fft/2`` samples are
    trimmed.  A sample is emitted once no future frame can overlap it, which
    for 50% overlap is one hop after its frame lands.
    """

    def __init__(self, n_fft: int, hop: int, window: Tensor):
        self.n_fft = n_fft
        self.hop = hop
        self.half = n_fft // 2
        self.window = window
        self._acc: Optional[Tensor] = None
        self._env: Optional[Tensor] = None
        self._acc_origin = 0        # untrimmed position of _acc[:, 0]
        self._flushed = False
        self.frames_seen = 0

    def reset(self) -> None:
        self._acc = None
        self._env = None
        self._acc_origin = 0
        self._flushed = False
        self.frames_seen = 0

    def push(self, frame: Tensor) -> Tensor:
        if self._flushed:
            raise RuntimeError(
                "StreamISTFT.push after flush: the tail is already finalized; "
                "reset() before starting a new stream"
            )
        segment = torch.fft.irfft(frame, n=self.n_fft, dim=-1) * self.window
        wsq = (self.window * self.window).unsqueeze(0)
        if self._acc is None:
            self._acc = segment.clone()
            self._env = wsq.clone()
        else:
            local = self.frames_seen * self.hop - self._acc_origin
            need = local + self.n_fft - self._acc.shape[-1]
            if need > 0:
                self._acc = F.pad(self._acc, (0, need))
                self._env = F.pad(self._env, (0, need))
            self._acc[:, local:local + self.n_fft] += segment
            self._env[:, local:local + self.n_fft] += wsq
        self.frames_seen += 1
        # No later frame reaches below the next frame's start position.
        final_upto = self.frames_seen * self.hop
        n_ready = final_upto - self._acc_origin
        ready = self._acc[:, :n_ready] / self._env[:, :n_ready].clamp_min(1e-11)
        emit_lo = max(self.half, self._acc_origin)
        out = ready[:, emit_lo - self._acc_origin:]
        self._acc = self._acc[:, n_ready:]
        self._env = self._env[:, n_ready:]
        self._acc_origin = final_upto
        return out

    def flush(self, length: Optional[int] = None,
              already_emitted: int = 0) -> Tensor:
        if self._flushed:
            raise RuntimeError("StreamISTFT.flush called twice; reset() first")
        self._flushed = True
        if self._acc is None or self._acc.shape[-1] == 0:
            return torch.zeros(1, 0)
        tail = self._acc / self._env.clamp_min(1e-11)
        emit_lo = max(self.half, self._acc_origin)
        tail = tail[:, emit_lo - self._acc_origin:]
        if length is not None:
            tail = tail[:, : max(length - already_emitted, 0)]
        return tail

    def state_tensors(self) -> Dict[str, Tensor]:
        out = {}
        if self._acc is not None:
            out["overlap"] = self._acc
            out["envelope"] = self._env
        return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def state_report(cells: Dict[str, object]) -> str:
    """Human-readable inventory of every persistent tensor in a state dict.

    ``cells`` maps a name to anything exposing ``state_tensors()`` (the cells
    above) or to a raw tensor.  The report is the deployment RAM contract: an
    NPU/C port must persist exactly these tensors between invocations.
    """
    rows = []
    total = 0
    for name, cell in sorted(cells.items()):
        tensors = (cell.state_tensors() if hasattr(cell, "state_tensors")
                   else {"": cell} if isinstance(cell, Tensor) else {})
        for sub, tensor in tensors.items():
            label = f"{name}.{sub}" if sub else name
            nbytes = tensor.numel() * tensor.element_size()
            total += nbytes
            rows.append(f"  {label:<44s} {str(tuple(tensor.shape)):<24s} "
                        f"{str(tensor.dtype).replace('torch.', ''):<10s} "
                        f"{nbytes / 1024:8.1f} KB")
    rows.append(f"  {'TOTAL':<44s} {'':<24s} {'':<10s} {total / 1024:8.1f} KB")
    return "\n".join(rows)
