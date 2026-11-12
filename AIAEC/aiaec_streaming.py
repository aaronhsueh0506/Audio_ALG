"""Shared stateful cells for frame-by-frame AIAEC inference.

Every cell reuses the weights of an offline module and replays its arithmetic
on one (or a few) new time frames, carrying the time context in explicit state
tensors. Zero-initialised history equals the offline left-only zero padding, so
a stream started from reset matches one whole-utterance ``forward`` (exceptions
are documented on each model's ``forward_stream``). The cells are a Python
deployment reference defining the state inventory and per-invocation I/O
contract an NPU/C port must reproduce.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from AIAEC.aiaec_common import CausalConv2d, FrameDelayAttention, GlobalDelayAttention


# ---- guards ----

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


# ---- convolution cells ----

class StreamConv2dCell:
    """Streaming context for one time-extended Conv2d.

    Wraps a conv whose offline call pads ``(kt - 1) * dt`` zeros on the left.
    The state is the full input window of ``retain + 1`` frames, oldest first,
    ``retain = (kt - 1) * dt``. ``step`` takes ``[B, C, T_new, F]`` and returns
    ``T_new`` frames; the next state is the last ``retain + 1`` frames of the
    conv input, so an exported graph is a Slice plus a Concat.
    """

    def __init__(self, conv: nn.Conv2d, kt: int, dt: int,
                 freq_left: int, freq_right: int, lookahead: int = 0):
        if conv.stride[0] != 1:
            raise ValueError("streaming cells support time stride 1 only")
        if lookahead != 0:
            raise ValueError("streaming cells support causal convs only "
                             "(lookahead must be 0)")
        if (kt - 1) * dt < 1:
            raise ValueError("a conv without temporal context needs no cell")
        self.conv = conv
        self.kt = kt
        self.dt = dt
        self.freq_left = freq_left
        self.freq_right = freq_right
        self.retain = (kt - 1) * dt
        self.window = self.retain + 1
        self._window: Optional[Tensor] = None

    @classmethod
    def from_causal(cls, module: CausalConv2d) -> "StreamConv2dCell":
        freq_total = (module.kf - 1) * module.df
        freq_left = freq_total // 2
        return cls(module.conv, module.kt, module.dt,
                   freq_left, freq_total - freq_left)

    def reset(self) -> None:
        self._window = None

    def step(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("StreamConv2dCell expects [B,C,T,F]")
        if self._window is None:
            b, c, _, f = x.shape
            self._window = x.new_zeros(b, c, self.window, f)
        # The window's one allowed Slice consumer (checked by the in-place
        # state contract) -- named once so the T_new > 1 branch below reads
        # it again instead of slicing self._window a second time.
        window_tail = self._window[:, :, 1:self.window]
        conv_input = torch.cat((window_tail, x), dim=2)
        if conv_input.shape[2] == self.window:
            self._window = conv_input
        else:
            # T_new > 1: conv_input is longer than the window, and its
            # trimmed tail must still be produced by a Concat, not a bare
            # Slice (the in-place state-output contract) -- split at the
            # carried-old/all-new boundary, the same old-tail + new-frame
            # shape the T_new == 1 branch above already produces.
            new_keep = min(self.window, x.shape[2])
            old_keep = self.window - new_keep
            old_tail = window_tail[:, :, window_tail.shape[2] - old_keep :]
            new_tail = x[:, :, x.shape[2] - new_keep :]
            self._window = torch.cat((old_tail, new_tail), dim=2)
        padded = F.pad(conv_input, (self.freq_left, self.freq_right, 0, 0))
        return self.conv(padded)

    def state_tensors(self) -> Dict[str, Tensor]:
        return {} if self._window is None else {"window": self._window}


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


# ---- recurrence / ring cells ----

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

    Keeps the ``delays`` most recent frames of a ``[B, C, F]`` feature.
    ``step`` takes one frame (``[B, C, F]`` or ``[B, C, 1, F]``) and returns
    ``[B, C, 1, delays, F]`` with slot ``d`` holding ``x[t - d]`` (zero before
    stream start, matching the offline left zero pad) -- this one-frame path
    is byte-for-byte the original code, unchanged, since every existing
    caller still uses it. Given ``[B, C, T_new, F]`` with ``T_new > 1`` (a
    multi-frame accelerator call), it instead returns ``[B, C, T_new, delays,
    F]``: slot ``i`` of the new axis is the ring exactly as it stood after
    frame ``t + i`` was pushed, i.e. the same per-frame output as calling
    this method ``T_new`` times with one frame each.
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
            t_new = frame.shape[2]
            if t_new == 1:
                frame = frame[:, :, 0]
        elif frame.ndim == 3:
            t_new = 1
        else:
            raise ValueError("DelayRingCell expects [B,C,F] or [B,C,T,F]")
        if self._ring is None:
            if t_new == 1:
                b, c, f = frame.shape
            else:
                b, c, _, f = frame.shape
            self._ring = frame.new_zeros(b, c, self.delays, f)
        if t_new == 1:
            self._ring = torch.cat(
                (frame.unsqueeze(2), self._ring[:, :, : self.delays - 1]), dim=2,
            )
            return self._ring.unsqueeze(2)
        # history: the ring's own single Slice consumer, unchanged from the
        # one-frame path above regardless of T_new -- everything below reads
        # its output, never the ring state tensor itself again.
        history = self._ring[:, :, : self.delays - 1]
        full = torch.cat((history.flip(2), frame), dim=2)   # chronological
        # Tensor.unfold has no ONNX exporter support; the same sliding
        # window, one Slice+Flip per output position, stacked on a new axis,
        # traces to Slice/Concat like every other cell here.
        windows = torch.stack(
            [full[:, :, i : i + self.delays].flip(2) for i in range(t_new)],
            dim=2,
        )
        # The new ring state must be produced by a Concat, not a bare Slice
        # of `windows` (the in-place state-output contract): same newest-
        # kept-new + oldest-kept-old split as the T_new == 1 branch, and
        # `history` (not `self._ring` again) is the sole second operand, so
        # the ring input keeps its one required Slice consumer.
        new_keep = min(self.delays, t_new)
        old_keep = self.delays - new_keep
        new_part = frame[:, :, frame.shape[2] - new_keep :].flip(2)
        old_part = history[:, :, :old_keep]
        self._ring = torch.cat((new_part, old_part), dim=2)
        return windows

    def state_tensors(self) -> Dict[str, Tensor]:
        return {} if self._ring is None else {"ring": self._ring}


class FrameDelayAttentionCell:
    """Streaming twin of :class:`aiaec_common.FrameDelayAttention`.

    State: key ring, value ring (``max_delay_frames`` frames each, newest
    first) and the score conv's five-frame logit window (oldest first).  The
    per-frame arithmetic keeps the offline multiply-then-sum order, so
    outputs match ``forward`` exactly. ``step`` takes one frame (``mic``/
    ``far`` time axis == 1, the original contract) or ``T_new > 1`` frames at
    once; every op below is already batched over that axis (the rings via
    :class:`DelayRingCell`, the score conv via :class:`StreamConv2dCell`,
    softmax/the weighted sums elementwise per output frame), so the T_new > 1
    result is the same per-frame output as T_new calls of one frame each.
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
        q = att.query(mic)                            # [B,H,T,F]
        k_delayed = self.key_ring.step(att.key(far))  # [B,H,T,D,F]
        logits = (q.unsqueeze(3) * k_delayed).sum(dim=-1)   # [B,H,T,D]
        logits = self.score_cell.step(logits).squeeze(1)    # [B,T,D]
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
    accumulator and an absolute frame counter. The offline forward masks delays
    with ``arange(D) < T`` using the final utterance length ``T``; a stream
    cannot know ``T`` and assumes ``T >= D``. ``paper_global`` mode is
    rejected.
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


# ---- stateful STFT / WOLA (offline center=True contract) ----

class StreamSTFT:
    """Incremental ``torch.stft(center=True, pad_mode='reflect')``.

    Reproduces the offline centered timing at the cost of ``n_fft/2`` samples of
    lookahead: a frame is emitted once its trailing half-window has arrived. Feed
    arbitrary-sized sample chunks; ``push`` returns zero or more complex
    ``[B, n_freqs]`` frames; ``flush`` applies the trailing reflect pad and emits
    the remaining frames (total ``L // hop + 1``).
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
        # Trailing reflect pad mirrors the half samples before the last one; total
        # frames are L // hop + 1.
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


# ---- reporting ----

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
