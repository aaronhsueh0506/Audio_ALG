"""AECNet: a standalone neural acoustic echo canceller that regresses the ECHO.

    in : Y (microphone) and X (far-end reference), complex spectra
    out: D_hat, an ESTIMATE OF THE ECHO D
    then  E = Y - D_hat   by SUBTRACTION, computed OUTSIDE this module.

⚠ THE SINGLE MOST IMPORTANT PROPERTY OF THIS FILE.
The network's target is D, the echo itself.  It is NOT trained to output the
enhanced signal S + N, and its output is NOT a mask applied to Y.  A mask on Y
would let one network silently perform cancellation, residual suppression and
noise reduction at once, and no measurement afterwards could attribute a
failure to any of the three.  ``tests/test_aecnet_model.py`` pins this
behaviourally: with Y == 0 and X active the model must still be able to emit a
non-zero D_hat, which no mask-on-Y architecture can do.

ARCHITECTURE
------------
FCRN / CRUSE-DAEC family:

    (B, 4, T, F)  [Re(Y), Im(Y), Re(X), Im(X)]
        -> strided conv encoder down the FREQUENCY axis, causal in time
        -> grouped-GRU bottleneck (channel-grouped, shuffled between layers)
        -> mirrored transposed-conv decoder with encoder skips
    (B, 2, T, F)  [Re(D_hat), Im(D_hat)]

Everything is causal on the time axis.  ``lookahead`` is a config knob and
defaults to 0; see :meth:`AecNet.forward` for the delay convention it imposes.

STREAMING STATE
---------------
Every time-dependent module carries explicit state, so a 4 s chunk processed
after its predecessor is bit-comparable with the same frames processed inside
one long call (``tests/test_aecnet_model.py::test_chunked_state_matches_one_shot``).
That is what lets the trainer walk a whole 20-60 s sequence with the recurrent
state intact -- ⚠ resetting state every chunk hides convergence from cold,
echo-path-change recovery and long-term drift, which are the behaviours this
corpus exists to expose.
"""

import dataclasses
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen_aec import frames_from_seconds  # noqa: E402


__all__ = [
    'MODEL_KEYS',
    'AecNet',
    'AecNetConfig',
    'assert_zero_reference_gate',
    'compress_spec',
    'expand_spec',
    'safe_mag',
    'zero_reference_leak_db',
]


# ⚠ Not a config knob.  This only exists to give ``sqrt`` and a negative power
# a finite value and a finite gradient at exactly zero; it is not a tuning
# parameter and nothing about the model's behaviour should depend on it.
#
# 1e-12 is chosen so the region where it distorts anything sits at -240 dBFS in
# the compressed domain, i.e. below any signal this corpus can produce.  A
# larger value (1e-8 is the usual reflex) makes ``compress_spec`` gradient-flat
# for outputs below roughly -190 dBFS, which sounds harmless until the idle loss
# has driven D_hat to zero and the model can no longer climb back out.
_MAG_EPS = 1e-12


# ============================================================
# Spectral compression
# ============================================================
#
# The model reads and writes the COMPRESSED domain: |Z|^c with the phase
# untouched.  Two reasons, and the second is the load-bearing one:
#
#   1. Speech spectra span ~80 dB.  An uncompressed MSE is dominated by a
#      handful of loud low-frequency bins and never learns the high band.
#   2. Because the network predicts in the compressed domain and the loss is
#      taken there too, ``L_echo`` is a PLAIN MSE ON THE NETWORK OUTPUT --
#      compress(expand(u)) == u.  Predicting linearly and compressing only in
#      the loss would put a c-th root between the parameters and the objective,
#      which reintroduces exactly the dynamic-range conditioning problem the
#      compression exists to remove.
#
# c = 0.3 is the DeepFilterNet/GTCRN value and is what the sibling projects in
# this repo already use, so the loss curves are on a comparable scale.

def safe_mag(spec: torch.Tensor, eps: float = _MAG_EPS) -> torch.Tensor:
    """|z| with a finite gradient at z == 0.

    ⚠ Do not replace with ``spec.abs()``.  torch's complex ``abs`` has gradient
    ``conj(z)/|z|``, which is NaN at the origin -- and the origin genuinely
    occurs here: during a reference dropout X is exactly 0 and so is D.
    """
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + eps * eps)


def compress_spec(spec: torch.Tensor, c: float, eps: float = _MAG_EPS) -> torch.Tensor:
    """``z -> z * |z|^(c-1)``, i.e. magnitude^c with the phase preserved."""
    if c == 1.0:
        return spec
    return spec * safe_mag(spec, eps).pow(c - 1.0)


def expand_spec(spec: torch.Tensor, c: float, eps: float = _MAG_EPS) -> torch.Tensor:
    """Inverse of :func:`compress_spec`."""
    if c == 1.0:
        return spec
    return spec * safe_mag(spec, eps).pow(1.0 / c - 1.0)


# ============================================================
# Config
# ============================================================

# Every key the ``[model]`` section may contain.  ⚠ The round-trip test asserts
# config.ini's [model] section is EXACTLY this set: a key nobody reads is a
# knob that silently does nothing, and a key the code reads but the file omits
# is a default that no one reviewed.
MODEL_KEYS = (
    'channels',
    'kernel_t',
    'kernel_f',
    'stride_f',
    'gru_layers',
    'gru_groups',
    'lookahead_sec',
    'compress_exponent',
)


@dataclasses.dataclass(frozen=True)
class AecNetConfig:
    """Resolved ``[model]`` section.  Frozen -- see ``AecGrid`` for the reasoning."""

    channels: Tuple[int, ...] = (16, 32, 48, 64)
    kernel_t: int = 2
    kernel_f: int = 5
    stride_f: int = 2
    gru_layers: int = 2
    gru_groups: int = 8
    # ⚠ RESOLVED frames, not the seconds the config carries.  The config knob is
    # ``lookahead_sec``; this field is what it became on this grid, and it is
    # what goes in the checkpoint contract -- a checkpoint trained at 16 kHz
    # with 0.032 s of lookahead must not silently load at 48 kHz, where the same
    # seconds are a different number of frames and therefore different weights.
    lookahead: int = 0
    compress_exponent: float = 0.3

    @classmethod
    def from_config(cls, cfg, frame_rate: float,
                    section: str = 'model') -> 'AecNetConfig':
        if not cfg.has_section(section):
            raise ValueError(f"config has no [{section}] section")
        present = set(cfg[section])
        unknown = present - set(MODEL_KEYS)
        if unknown:
            raise ValueError(
                f"[{section}] has key(s) {sorted(unknown)} that this model does "
                f"not read; known keys are {list(MODEL_KEYS)}. A knob nobody "
                f"consumes is worse than no knob at all."
            )
        raw = cfg.get(section, 'channels', fallback=None)
        channels = cls.__dataclass_fields__['channels'].default
        if raw is not None:
            channels = tuple(int(part) for part in raw.replace(',', ' ').split())
        return cls(
            channels=channels,
            kernel_t=cfg.getint(section, 'kernel_t', fallback=2),
            kernel_f=cfg.getint(section, 'kernel_f', fallback=5),
            stride_f=cfg.getint(section, 'stride_f', fallback=2),
            gru_layers=cfg.getint(section, 'gru_layers', fallback=2),
            gru_groups=cfg.getint(section, 'gru_groups', fallback=8),
            lookahead=frames_from_seconds(
                cfg.getfloat(section, 'lookahead_sec', fallback=0.0), frame_rate),
            compress_exponent=cfg.getfloat(section, 'compress_exponent', fallback=0.3),
        )

    def as_contract(self) -> Dict[str, object]:
        """The subset that changes the meaning of trained weights."""
        return {
            'channels': ','.join(str(c) for c in self.channels),
            'kernel_t': self.kernel_t,
            'kernel_f': self.kernel_f,
            'stride_f': self.stride_f,
            'gru_layers': self.gru_layers,
            'gru_groups': self.gru_groups,
            'lookahead': self.lookahead,
            'compress_exponent': self.compress_exponent,
        }


# ============================================================
# Building blocks
# ============================================================

class FreqChannelNorm(nn.Module):
    """Normalise over (channel, frequency) for each (batch, frame), then scale.

    ⚠ Deliberately NOT BatchNorm2d, which every published CRUSE implementation
    uses.  BatchNorm2d over ``(B, C, T, F)`` normalises across the TIME axis at
    training time, so the statistics used for frame t include frames t+1..T-1.
    The model then trains against a normaliser that peeks at the future and runs
    at inference against running averages that do not -- a train/eval mismatch
    that is invisible in the loss curve and fatal in a streaming product.  It
    also makes the causality test pass only in ``eval()`` mode, which is exactly
    the kind of "the test proves less than it looks like it does" that this repo
    has been bitten by before.

    The cost is a per-frame mean/variance instead of a batched one; the benefit
    is that train and eval are the same function and batch size 1 works.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class _CausalFreqConv(nn.Module):
    """Conv (or transposed conv) that is CAUSAL in time and strided in frequency.

    Both directions use the same convention: the caller's state holds the last
    ``kernel_t - 1`` input frames, they are prepended, and the output is sliced
    back to ``T`` frames.  Writing the decoder this way -- rather than the usual
    "transposed conv in time and hope" -- means the upsampling path has the same
    state discipline as the encoder, with no overlap-add tail and no bias
    double-counting to get wrong.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_t: int, kernel_f: int,
                 stride_f: int, transposed: bool, bias: bool):
        super().__init__()
        self.pad_t = kernel_t - 1
        self.transposed = transposed
        self.in_ch = in_ch
        pad_f = kernel_f // 2
        if transposed:
            self.conv = nn.ConvTranspose2d(
                in_ch, out_ch, (kernel_t, kernel_f),
                stride=(1, stride_f), padding=(0, pad_f), bias=bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_ch, out_ch, (kernel_t, kernel_f),
                stride=(1, stride_f), padding=(0, pad_f), bias=bias,
            )

    def state_shape(self, batch: int, n_freqs: int) -> Tuple[int, int, int, int]:
        return (batch, self.in_ch, self.pad_t, n_freqs)

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T, F)
        n_frames = x.shape[2]
        if self.pad_t == 0:
            y = self.conv(x)
            return y, x.new_zeros(self.state_shape(x.shape[0], x.shape[3]))
        if state is None:
            state = x.new_zeros(self.state_shape(x.shape[0], x.shape[3]))
        padded = torch.cat([state, x], dim=2)
        new_state = padded[:, :, -self.pad_t:, :]
        y = self.conv(padded)
        if self.transposed:
            # ConvTranspose spreads each input frame forward by kernel_t-1, so
            # the frame that corresponds to input t sits at index t + pad_t.
            y = y[:, :, self.pad_t:self.pad_t + n_frames, :]
        return y, new_state


def _fit_freq(x: torch.Tensor, target: int) -> torch.Tensor:
    """Crop or zero-pad the frequency axis to ``target``.

    For an odd ``kernel_f`` and an ``n_freqs`` of the form ``k*stride^n + 1``
    -- which every grid in ``AecGrid`` produces, 257 and 513 included -- the
    transposed conv already lands exactly on the encoder's input width and this
    is a no-op.  It exists so an exotic config degrades into a one-bin crop
    instead of a shape error three layers later.
    """
    got = x.shape[-1]
    if got == target:
        return x
    if got > target:
        return x[..., :target]
    return F.pad(x, (0, target - got))


class GroupedGRU(nn.Module):
    """CRUSE-style grouped GRU with a channel shuffle between layers.

    The bottleneck feature vector is ``channels[-1] * n_freqs_enc`` wide -- 1088
    at the default 16 kHz config -- and a full GRU on it costs ~14 M parameters
    on its own.  Splitting it into ``n_groups`` independent GRUs divides that by
    ``n_groups`` and is what keeps the whole model inside its 1-3 M budget.

    ⚠ The shuffle between layers is not decoration.  Without it the groups are
    ``n_groups`` completely independent recurrent networks stacked in parallel,
    and information in one frequency/channel group can never reach another --
    which for echo cancellation is fatal, since the cue that a bin contains echo
    lives in the reference's activity across the whole spectrum.

    Groups are cut CHANNEL-major: group g owns ``channels[-1]/n_groups`` feature
    maps across all encoder frequencies.  Frequency-major grouping (each group a
    contiguous band) would be the more intuitive split, but it needs
    ``n_freqs_enc`` divisible by ``n_groups`` and no config guarantees that --
    17 encoder bins at 16 kHz do not divide by 8.
    """

    def __init__(self, feature_dim: int, n_groups: int, n_layers: int):
        super().__init__()
        if feature_dim % n_groups != 0:
            raise ValueError(
                f"bottleneck width {feature_dim} is not divisible by gru_groups "
                f"{n_groups}")
        self.feature_dim = feature_dim
        self.n_groups = n_groups
        self.n_layers = n_layers
        self.width = feature_dim // n_groups
        self.grus = nn.ModuleList([
            nn.GRU(self.width, self.width, num_layers=1, batch_first=True)
            for _ in range(n_layers * n_groups)
        ])

    def state_shape(self, batch: int) -> Tuple[int, int, int]:
        return (self.n_layers * self.n_groups, batch, self.width)

    def _shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return (x.reshape(b, t, self.n_groups, self.width)
                 .transpose(2, 3)
                 .reshape(b, t, self.feature_dim))

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, feature_dim)
        if state is None:
            state = x.new_zeros(self.state_shape(x.shape[0]))
        out = x
        new_states: List[torch.Tensor] = []
        for layer in range(self.n_layers):
            parts = out.chunk(self.n_groups, dim=-1)
            outs = []
            for group in range(self.n_groups):
                index = layer * self.n_groups + group
                h0 = state[index:index + 1].contiguous()
                y, h1 = self.grus[index](parts[group].contiguous(), h0)
                outs.append(y)
                new_states.append(h1)
            out = torch.cat(outs, dim=-1)
            if layer + 1 < self.n_layers:
                out = self._shuffle(out)
        return out, torch.cat(new_states, dim=0)


# ============================================================
# The model
# ============================================================

class AecNet(nn.Module):
    """Echo estimator.  ``forward`` returns D_hat, never enhanced speech."""

    def __init__(self, n_freqs: int, config: Optional[AecNetConfig] = None,
                 in_channels: int = 4, out_channels: int = 2):
        super().__init__()
        cfg = config or AecNetConfig()
        if cfg.kernel_f % 2 != 1:
            raise ValueError(
                f"kernel_f must be odd (got {cfg.kernel_f}); an even frequency "
                f"kernel makes the transposed conv miss the encoder's width and "
                f"the skip connections would be silently cropped")
        if cfg.kernel_t < 1:
            raise ValueError(f"kernel_t must be >= 1, got {cfg.kernel_t}")
        if cfg.lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {cfg.lookahead}")
        if not 0.0 < cfg.compress_exponent <= 1.0:
            raise ValueError(
                f"compress_exponent must be in (0, 1], got {cfg.compress_exponent}")
        if len(cfg.channels) == 0:
            raise ValueError("channels must list at least one encoder stage")

        self.cfg = cfg
        self.n_freqs = int(n_freqs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lookahead = cfg.lookahead
        self.compress_exponent = cfg.compress_exponent

        pad_f = cfg.kernel_f // 2
        freqs = [self.n_freqs]
        for _ in cfg.channels:
            nxt = (freqs[-1] + 2 * pad_f - cfg.kernel_f) // cfg.stride_f + 1
            if nxt < 1:
                raise ValueError(
                    f"encoder collapses the frequency axis to {nxt} bins; "
                    f"reduce len(channels) or stride_f for n_freqs={n_freqs}")
            freqs.append(nxt)
        self.freq_sizes = tuple(freqs)

        # -- encoder --
        self.encoder = nn.ModuleList()
        self.enc_norm = nn.ModuleList()
        self.enc_act = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(cfg.channels):
            self.encoder.append(_CausalFreqConv(
                prev, ch, cfg.kernel_t, cfg.kernel_f, cfg.stride_f,
                transposed=False, bias=False,   # the norm below carries the bias
            ))
            self.enc_norm.append(FreqChannelNorm(ch))
            self.enc_act.append(nn.PReLU(ch))
            prev = ch

        # -- bottleneck --
        feature_dim = cfg.channels[-1] * self.freq_sizes[-1]
        if cfg.channels[-1] % cfg.gru_groups != 0:
            raise ValueError(
                f"channels[-1] ({cfg.channels[-1]}) must be divisible by "
                f"gru_groups ({cfg.gru_groups}) so a group boundary falls on a "
                f"channel boundary")
        self.bottleneck = GroupedGRU(feature_dim, cfg.gru_groups, cfg.gru_layers)

        # -- decoder (mirrored, skip-concatenated) --
        self.decoder = nn.ModuleList()
        self.dec_norm = nn.ModuleList()
        self.dec_act = nn.ModuleList()
        n_stages = len(cfg.channels)
        for k in range(n_stages):
            i = n_stages - 1 - k
            in_ch = 2 * cfg.channels[i]            # decoder input + encoder skip
            last = (i == 0)
            out_ch = out_channels if last else cfg.channels[i - 1]
            self.decoder.append(_CausalFreqConv(
                in_ch, out_ch, cfg.kernel_t, cfg.kernel_f, cfg.stride_f,
                transposed=True, bias=last,        # last stage has no norm
            ))
            self.dec_norm.append(nn.Identity() if last else FreqChannelNorm(out_ch))
            self.dec_act.append(nn.Identity() if last else nn.PReLU(out_ch))

    # ---------------- state ----------------

    def zero_state(self, batch: int, device=None, dtype=torch.float32) -> Dict:
        """Fresh state for ``batch`` independent lanes."""
        kw = {'device': device, 'dtype': dtype}
        n_stages = len(self.cfg.channels)
        enc = [torch.zeros(*self.encoder[i].state_shape(batch, self.freq_sizes[i]), **kw)
               for i in range(n_stages)]
        dec = [torch.zeros(
            *self.decoder[k].state_shape(batch, self.freq_sizes[n_stages - k]), **kw)
            for k in range(n_stages)]
        return {
            'enc': enc,
            'gru': torch.zeros(*self.bottleneck.state_shape(batch), **kw),
            'dec': dec,
        }

    @staticmethod
    def detach_state(state: Optional[Dict]) -> Optional[Dict]:
        """⚠ Call once per chunk.  Without it the autograd graph grows for the
        whole 60 s sequence and the first backward pass exhausts memory."""
        if state is None:
            return None
        return {
            'enc': [t.detach() for t in state['enc']],
            'gru': state['gru'].detach(),
            'dec': [t.detach() for t in state['dec']],
        }

    @staticmethod
    def reset_state(state: Optional[Dict], reset: Optional[torch.Tensor]) -> Optional[Dict]:
        """Zero the lanes flagged in ``reset`` (a bool tensor of shape ``(B,)``).

        Pair with ``dataset_gen_aec.lane_reset_mask`` on the batch's
        ``chunk_index``: a lane whose chunk_index is 0 has just started a new
        sequence and its state belongs to a different room, device and talker.
        """
        if state is None or reset is None:
            return state
        reset = reset.to(dtype=torch.bool)
        if not bool(reset.any()):
            return state
        conv_mask = reset.view(-1, 1, 1, 1)
        gru_mask = reset.view(1, -1, 1)
        return {
            'enc': [t.masked_fill(conv_mask, 0.0) for t in state['enc']],
            'gru': state['gru'].masked_fill(gru_mask, 0.0),
            'dec': [t.masked_fill(conv_mask, 0.0) for t in state['dec']],
        }

    def align_to_input(self, out: torch.Tensor) -> torch.Tensor:
        """Drop the output delay, so index t is the estimate FOR input frame t.

        Offline convenience.  The last ``lookahead`` input frames have no
        estimate at all -- theirs would need input that has not arrived -- so
        the aligned result is ``lookahead`` frames shorter.  ``denoise.py``
        does the same thing but keeps the length by passing the tail through
        uncancelled, which is what a streaming implementation emits at
        end-of-stream.
        """
        if not self.lookahead:
            return out
        return out[:, :, self.lookahead:, :]

    # ---------------- forward ----------------

    def forward(self, x: torch.Tensor, state: Optional[Dict] = None
                ) -> Tuple[torch.Tensor, Dict]:
        """``(B, 4, T, F)`` -> ``(B, 2, T, F)``.

        Input channels are ``[Re(Y), Im(Y), Re(X), Im(X)]`` in the LINEAR
        domain; output channels are ``[Re(D_hat), Im(D_hat)]``, also linear.
        The compression the network actually works in is applied and undone
        inside this method, so ``compress_exponent`` is a property of the
        weights rather than of the caller.

        ⚠ LOOKAHEAD IS A LABEL, NOT A PADDING -- and that is not a shortcut, it
        is forced.  Any model that can be fed in chunks satisfies
        ``out[i] = f(x[0..i])``: at the moment output index i must be produced,
        input frame i+1 of that chunk does not exist yet.  A right-padded
        "non-causal" convolution does not escape this; a streaming
        implementation of one buffers, and the buffering IS the delay.  The only
        freedom left is which input frame ``out[i]`` is an estimate FOR.

        ``lookahead = L`` declares that ``out[i]`` is the estimate for INPUT
        FRAME ``i - L``.  That estimate has therefore seen inputs up to frame
        ``(i-L) + L`` -- exactly L frames of future context.  It costs L frames
        of algorithmic latency at inference (16 ms per frame on the 16 kHz
        grid), and it genuinely changes what the model learns, because the
        target it is trained against is shifted.  What it does NOT change is the
        arithmetic of this method, which is why it lives in the checkpoint
        contract and why ``EchoEstimationLoss`` has to be told the same value:
        a loss that compares index-for-index scores a target shifted by L
        frames, and that presents as a model which simply will not converge.

        Use :meth:`align_to_input` to undo the delay offline.
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, T, F), got {tuple(x.shape)}")
        if x.shape[3] != self.n_freqs:
            raise ValueError(
                f"input has {x.shape[3]} frequency bins, model was built for "
                f"{self.n_freqs}")
        if state is None:
            state = self.zero_state(x.shape[0], device=x.device, dtype=x.dtype)
        core_in = x

        # Compression is pointwise in time-frequency, so it does not touch
        # causality.  Y and X are compressed independently: their levels differ
        # by the echo return loss (0-30 dB in this corpus) and a shared scaling
        # would hand the loud one the whole dynamic range.
        y_c = compress_spec(torch.complex(core_in[:, 0], core_in[:, 1]),
                            self.compress_exponent)
        x_c = compress_spec(torch.complex(core_in[:, 2], core_in[:, 3]),
                            self.compress_exponent)
        h = torch.stack([y_c.real, y_c.imag, x_c.real, x_c.imag], dim=1)

        skips: List[torch.Tensor] = []
        enc_state: List[torch.Tensor] = []
        for i, conv in enumerate(self.encoder):
            h, s = conv(h, state['enc'][i])
            enc_state.append(s)
            h = self.enc_act[i](self.enc_norm[i](h))
            skips.append(h)

        b, c, t, f = h.shape
        flat = h.permute(0, 2, 1, 3).reshape(b, t, c * f)
        flat, gru_state = self.bottleneck(flat, state['gru'])
        h = flat.reshape(b, t, c, f).permute(0, 2, 1, 3)

        n_stages = len(self.encoder)
        dec_state: List[torch.Tensor] = []
        for k, conv in enumerate(self.decoder):
            i = n_stages - 1 - k
            h = torch.cat([h, skips[i]], dim=1)
            h, s = conv(h, state['dec'][k])
            dec_state.append(s)
            h = _fit_freq(h, self.freq_sizes[i])
            h = self.dec_act[k](self.dec_norm[k](h))

        d_hat_c = torch.complex(h[:, 0], h[:, 1])
        d_hat = expand_spec(d_hat_c, self.compress_exponent)
        out = torch.stack([d_hat.real, d_hat.imag], dim=1)
        return out, {'enc': enc_state, 'gru': gru_state, 'dec': dec_state}

    def forward_spec(self, y_spec: torch.Tensor, x_spec: torch.Tensor,
                     state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Complex ``(B, F, T)`` in, complex ``(B, F, T)`` D_hat out.

        The convenience wrapper the trainer and ``denoise.py`` use, so the
        ``(F, T)`` layout of ``dataset_gen_aec.stft`` never has to be transposed
        at three different call sites.
        """
        stacked = torch.stack(
            [y_spec.real, y_spec.imag, x_spec.real, x_spec.imag], dim=1
        ).transpose(-2, -1)                      # (B, 4, T, F)
        out, state = self.forward(stacked, state)
        d_hat = torch.complex(out[:, 0], out[:, 1]).transpose(-2, -1)
        return d_hat, state

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# The zero-reference hard gate
# ============================================================
#
# ⚠ This is the gate the idle loss term exists to satisfy: a silent reference
# must produce no echo estimate, so that E = Y - D_hat degenerates to Y and the
# canceller is provably transparent when there is nothing to cancel.  The
# corpus supplies the supervision through the 'ref_dropout' scenario, where X
# and D are both exactly zero.
#
# It is stated as a MEASUREMENT plus a threshold rather than an exact zero,
# because an exact zero is not achievable by a network with biases and is not
# what matters -- what matters is that the leak sits below anything audible.

def zero_reference_leak_db(model: AecNet, y_spec: torch.Tensor,
                           state: Optional[Dict] = None) -> float:
    """Echo estimate the model invents when the reference is silent, in dB.

    ``20*log10(||D_hat|| / ||Y||)`` with X forced to zero.  Since
    ``E = Y - D_hat``, this is exactly the level at which the canceller damages
    a signal it should be passing through untouched.
    """
    x_spec = torch.zeros_like(y_spec)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            d_hat, _ = model.forward_spec(y_spec, x_spec, state)
    finally:
        model.train(was_training)
    leaked = float(d_hat.abs().pow(2).sum())
    reference = float(y_spec.abs().pow(2).sum())
    if reference <= 0.0:
        raise ValueError("y_spec is all zeros; the leak ratio is undefined")
    if leaked <= 0.0:
        return -math.inf
    return 10.0 * math.log10(leaked / reference)


def assert_zero_reference_gate(model: AecNet, y_spec: torch.Tensor,
                               max_leak_db: float = -40.0,
                               state: Optional[Dict] = None) -> float:
    """Raise unless a silent reference produces a negligible echo estimate.

    ⚠ Only meaningful for a TRAINED model.  An untrained network fails this by
    construction, so the unit tests check shape and finiteness instead and this
    helper is what the trained-model evaluation calls.
    """
    leak_db = zero_reference_leak_db(model, y_spec, state)
    if not math.isfinite(leak_db) and leak_db != -math.inf:
        raise AssertionError(
            "zero-reference response is not finite; the model produced NaN or "
            "inf from a silent reference")
    if leak_db > max_leak_db:
        raise AssertionError(
            f"zero-reference gate FAILED: with X == 0 the model emits an echo "
            f"estimate at {leak_db:.1f} dB relative to the microphone, above the "
            f"{max_leak_db:.1f} dB limit. E = Y - D_hat is therefore not "
            f"transparent when there is no echo to cancel; raise lambda_idle or "
            f"the share of 'ref_dropout' chunks in the corpus.")
    return leak_db


def build_model(cfg, grid) -> AecNet:
    """The one place a model is constructed from a config + an ``AecGrid``."""
    return AecNet(n_freqs=grid.n_freqs,
                  config=AecNetConfig.from_config(cfg, grid.frame_rate))


def describe(model: AecNet, frame_rate: Optional[float] = None) -> Sequence[str]:
    """Lines for the startup banner.

    ``frame_rate`` is only needed to render the lookahead in milliseconds; the
    resolved frame count is the truth either way.
    """
    cfg = model.cfg
    look_ms = ("" if frame_rate is None
               else f" = {1000.0 * cfg.lookahead / frame_rate:.1f} ms")
    return [
        f"channels      : {list(cfg.channels)}  (kernel {cfg.kernel_t}x{cfg.kernel_f}, "
        f"freq stride {cfg.stride_f})",
        f"freq sizes    : {list(model.freq_sizes)}",
        f"bottleneck    : {cfg.gru_layers} x grouped GRU, {cfg.gru_groups} groups "
        f"of {model.bottleneck.width} (feature {model.bottleneck.feature_dim})",
        f"lookahead     : {cfg.lookahead} frame(s){look_ms}"
        + ("" if cfg.lookahead == 0
           else "  ⚠ output index i is the estimate for input frame i-lookahead"),
        f"compression   : |Z|^{cfg.compress_exponent} (input, output and loss domain)",
    ]
