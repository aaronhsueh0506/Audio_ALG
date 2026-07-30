"""JointAECNR: one network that does cancellation, residual suppression and
noise reduction in a single step.

    in  : (Y, X)      microphone spectrum, far-end reference spectrum
    out : S_hat       the finished near-end speech
    target: S

⚠ THIS IS THE OPAQUE ENTRY IN THE BAKE-OFF, BY CONSTRUCTION.
There is no D_hat handoff to a later stage, no min() fusion of two gains, and
no classical comfort-noise generator driven by g_res -- this model destroys all
three by doing their jobs at once.  That is the point of the experiment, but it
means a failure here says only "worse", never "worse at what".  The two
auxiliary heads below exist solely to buy that attribution back, and are the
only reason this project can be compared against a cascade at all:

  * ``aux_echo_head``      -> D_hat, supervised by the independent ``echo``
                              stem.  Separates "failed to cancel" from
                              "failed to suppress what it did not cancel".
  * ``aux_noise_psd_head`` -> the local-noise PSD, so a downstream comfort-noise
                              generator has the reference the classical chain
                              used to get from g_res.

Both are optional (config switches).  Turning them off is legitimate for a
shipping build and is exactly what makes the model unattributable, so the
banner prints their state on every run.

THE ZERO-REFERENCE GUARANTEE
----------------------------
The whole reference pathway -- ref encoder, the temporal-context conv, and the
1x1 fusion projections -- is bias-free and uses only positively homogeneous
activations, with no normalisation layer that could inject an offset.  So:

    X == 0 over the reference receptive field  ==>  ref pathway output == 0
                                               ==>  the network is EXACTLY a
                                                   mic-only NR model.

This is a property of the architecture, not of the training, and
``tests/test_joint_aecnr.py`` asserts it by randomising the reference-branch
weights and checking the output does not move.  It is what gives the hard gate
("ref == 0 must leave the microphone essentially unmodified") something to
stand on before a single gradient step.

⚠ The guarantee is NOT "output == mic".  This model also removes noise, so with
X == 0 and noise present the correct output differs from the mic on purpose.
The scoped, honest statement is the one above: with no reference there is no
reference-driven behaviour.  ``idle_gate_report`` measures the trained-model
version of the same question.
"""

import dataclasses
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.aec import AecGrid, frames_from_seconds  # noqa: E402


__all__ = [
    'JointAECNR',
    'JointOutputs',
    'bins_from_hz',
    'compress_complex',
    'deep_filter_apply',
    'detach_state',
    'frames_from_seconds',
    'idle_gate_report',
    'reference_activity_gate',
    'reset_state',
]


# ============================================================
# Grid-derived quantities
# ============================================================
#
# ⚠ Every knob that is physically a DURATION arrives in SECONDS and is turned
# into frames by ``dataset_gen.aec.frames_from_seconds``, which all three AEC
# models import rather than redefine.  A literal frame count in a config means
# one thing at 16 kHz/hop 256 and something else at 48 kHz/hop 512, so the
# 48 kHz variant would quietly become a different model rather than the same
# model on a different grid.  ``frames_from_seconds`` is re-exported here only
# so this module's own callers keep one import site.


def bins_from_hz(hz: float, grid: AecGrid) -> int:
    """Upper edge in Hz -> number of FFT bins covering [0, hz], inclusive.

    Used for the deep-filter band.  Expressed in Hz rather than bins so the
    48 kHz variant covers the same ACOUSTIC band without a config edit.
    """
    if hz <= 0.0:
        return 0
    bins = int(round(hz * grid.n_fft / grid.sr)) + 1
    return max(1, min(bins, grid.n_freqs))


# ============================================================
# Input features
# ============================================================

def compress_complex(spec: torch.Tensor, exponent: float,
                     eps: float = 1e-12) -> torch.Tensor:
    """``(B, F, T)`` complex -> ``(B, 2, T, F)`` real, power-law compressed.

    ``spec * |spec|^(exponent-1)``, i.e. magnitude is raised to ``exponent``
    while phase is untouched.

    ⚠ The reference branch MUST NOT use a log-magnitude feature.  ``log|X|`` of
    a silent reference is ``-inf``, which any implementation clamps to some
    large negative floor -- a non-zero constant.  That single choice would
    destroy the zero-reference guarantee this whole model is built around,
    because a bias-free network fed a constant no longer outputs zero.  This
    compression maps exactly 0 to exactly 0.
    """
    mag = spec.abs()
    scale = mag.clamp_min(eps).pow(exponent - 1.0)
    out = spec * scale
    return torch.stack([out.real, out.imag], dim=1).transpose(2, 3)


# ============================================================
# Causal building blocks
# ============================================================
#
# Tensor layout inside the network is (B, C, T, F): time is the conv "height"
# so a causal time kernel is a manual left-pad, and frequency is the "width"
# where striding happens.
#
# ⚠ There is deliberately no BatchNorm anywhere.  BatchNorm2d normalises over
# (N, H, W) and H is TIME here, so it would compute frame t's statistics from
# the whole chunk including frames after t -- acausal, and invisible in any
# offline test because offline the future is always there.  A frame-wise
# LayerNorm over (C, F) would be causal and is the acceptable alternative; it
# is left out so the reference branch and the mic branch stay structurally
# identical, which is what keeps the zero-reference guarantee easy to verify.

class CausalConv2d(nn.Module):
    """Conv over (T, F) that is strictly causal in T and carries its history.

    ``forward`` takes and returns the ``kernel_t - 1`` trailing input frames, so
    a chunked caller gets bit-identical results to processing the whole
    sequence at once.  ⚠ Without that, every chunk boundary is a zero-padded
    cold start and the model learns a "chunk began" cue no real stream has.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_t: int = 2,
                 kernel_f: int = 3, stride_f: int = 1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        if kernel_t < 1:
            raise ValueError(f"kernel_t must be >= 1, got {kernel_t}")
        self.history = kernel_t - 1
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_t, kernel_f),
            stride=(1, stride_f), padding=(0, (kernel_f - 1) // 2),
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.history == 0:
            return self.conv(x), None
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[1], self.history, x.shape[3])
        padded = torch.cat([state, x], dim=2)
        # ⚠ Not detached here.  Truncated BPTT is the trainer's decision; a
        # detach baked into the model would make full-sequence gradients
        # impossible and would be invisible from outside.
        return self.conv(padded), padded[:, :, -self.history:]


class FreqUpsample(nn.Module):
    """Transposed conv that undoes one encoder stage's frequency stride.

    ⚠ Time kernel is 1 on purpose: a decoder with time context would need its
    own history buffer in every streaming state, for no measured benefit --
    the recurrent bottleneck already carries the temporal memory.  Released
    DeepFilterNet3 makes the same choice (``convt_kernel = 1,3``).
    """

    def __init__(self, in_ch: int, out_ch: int, in_freqs: int, out_freqs: int,
                 kernel_f: int = 3, stride_f: int = 2, bias: bool = True):
        super().__init__()
        pad = (kernel_f - 1) // 2
        natural = (in_freqs - 1) * stride_f - 2 * pad + kernel_f
        output_padding = out_freqs - natural
        if not 0 <= output_padding < stride_f:
            raise ValueError(
                f"cannot upsample {in_freqs} -> {out_freqs} with stride "
                f"{stride_f} and kernel {kernel_f} (needs output_padding "
                f"{output_padding})"
            )
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(1, kernel_f), stride=(1, stride_f),
            padding=(0, pad), output_padding=(0, output_padding), bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# Reference activity
# ============================================================

def reference_activity_gate(x_spec: torch.Tensor, memory_frames: int,
                            floor_power: float,
                            state: Optional[torch.Tensor] = None
                            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-frame ``[0, 1)`` gate: "has the far end played recently?"

    ``p[t] = max over the last memory_frames of mean_F |X|^2`` and
    ``gate = p / (p + floor_power)``.  Exactly 0 iff the reference has been
    exactly silent for the whole window; smoothly ~1 once it is well above the
    silence floor.

    ⚠ ``memory_frames`` is a PHYSICS parameter, not a smoothing preference.  The
    echo of the last sample the loudspeaker played arrives one bulk delay later
    (up to 120 ms in this corpus) and then decays over the room's RT60 (up to
    1.2 s).  A window shorter than that structurally forbids the echo head from
    cancelling the tail of a burst that has just ended -- it would be forced to
    zero while real echo is still arriving.  A window much longer than that
    merely delays the idle guarantee, which is the cheap direction to err in.

    Returns ``(gate, new_state)``; ``state`` is the previous
    ``memory_frames - 1`` power values, so a chunked caller matches the
    whole-sequence result exactly.
    """
    if memory_frames < 1:
        raise ValueError(f"memory_frames must be >= 1, got {memory_frames}")
    power = x_spec.abs().pow(2).mean(dim=1)          # (B, T)
    history = memory_frames - 1
    if history > 0:
        if state is None:
            state = power.new_zeros(power.shape[0], history)
        padded = torch.cat([state, power], dim=1)
        new_state = padded[:, -history:]
    else:
        padded, new_state = power, None
    held = F.max_pool1d(padded.unsqueeze(1), kernel_size=memory_frames,
                        stride=1).squeeze(1)          # (B, T)
    return held / (held + floor_power), new_state


# ============================================================
# Multi-frame deep filter
# ============================================================

def deep_filter_apply(spec: torch.Tensor, coefs: torch.Tensor, df_bins: int,
                      df_order: int, df_lookahead: int = 0,
                      state: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-bin FIR across frames on the lowest ``df_bins`` of ``spec``.

    A purely multiplicative mask cannot undo an echo whose energy has been
    smeared across frames by the room; the low bins are where that smearing
    costs the most speech.  Same construction as this repo's
    ``DeepFilterNet2/model.py``, extended with a streaming history so it obeys
    the same chunk-carry discipline as the rest of the model.

    Args:
        spec  : (B, F, T) complex -- the ALREADY masked spectrum
        coefs : (B, T, df_bins, df_order * 2)
        state : (B, df_bins, 2, df_order - 1 - df_lookahead) previous frames

    ⚠ ``df_lookahead`` is a SECOND source of algorithmic delay, independent of
    the model's ``lookahead_sec``.  Both are printed by the trainer banner in
    milliseconds; do not add one to a latency budget and forget the other.
    """
    if not 0 <= df_lookahead < df_order:
        raise ValueError(
            f"df_lookahead must be in [0, {df_order - 1}], got {df_lookahead}")

    coefs = coefs.view(coefs.shape[0], coefs.shape[1], df_bins, df_order, 2)
    spec_ri = torch.view_as_real(spec[:, :df_bins])   # (B, df_bins, T, 2)
    spec_ri = spec_ri.permute(0, 1, 3, 2)             # (B, df_bins, 2, T)

    history = df_order - df_lookahead - 1
    if history > 0:
        if state is None:
            state = spec_ri.new_zeros(spec_ri.shape[0], df_bins, 2, history)
        padded = torch.cat([state, spec_ri], dim=-1)
        new_state = padded[..., -history:]
    else:
        padded, new_state = spec_ri, None
    if df_lookahead > 0:
        # ⚠ Chunk-boundary zeros, exactly as for the model-level lookahead: the
        # future frames a real stream would supply are not in this chunk.
        padded = F.pad(padded, (0, df_lookahead))

    unfolded = padded.unfold(-1, df_order, 1)          # (B, df_bins, 2, T, order)
    coefs_p = coefs.permute(0, 2, 4, 1, 3)             # (B, df_bins, 2, T, order)

    df_re = (unfolded[:, :, 0] * coefs_p[:, :, 0]
             - unfolded[:, :, 1] * coefs_p[:, :, 1]).sum(-1)
    df_im = (unfolded[:, :, 1] * coefs_p[:, :, 0]
             + unfolded[:, :, 0] * coefs_p[:, :, 1]).sum(-1)

    out = spec.clone()
    out[:, :df_bins] = torch.view_as_complex(
        torch.stack([df_re, df_im], dim=-1).contiguous())
    return out, new_state


# ============================================================
# Outputs
# ============================================================

@dataclasses.dataclass
class JointOutputs:
    """What one forward pass produced.

    A dataclass rather than a tuple so that switching an auxiliary head off
    turns its field into ``None`` instead of silently shifting the position of
    everything after it.
    """

    speech_spec: torch.Tensor                    # S_hat, (B, F, T) complex
    mask: torch.Tensor                           # the complex mask on Y
    echo_spec: Optional[torch.Tensor] = None     # D_hat, aux_echo_head only
    noise_log_psd: Optional[torch.Tensor] = None  # log10 PSD, aux head only
    ref_gate: Optional[torch.Tensor] = None      # (B, T) reference activity


# ============================================================
# The model
# ============================================================

class JointAECNR(nn.Module):
    """Dual-input encoder, cross-branch fusion, recurrent bottleneck, complex
    mask decoder, optional deep-filter stage and optional auxiliary heads.

    ``forward(y_spec, x_spec, state)`` -> ``(JointOutputs, state)``.  The state
    is a plain dict of tensors so a caller can inspect, reset per lane, or
    detach it without the model knowing; see :func:`reset_state` and
    :func:`detach_state`.
    """

    def __init__(self, grid: AecGrid, enc_channels: int = 48,
                 enc_stages: int = 4, kernel_t: int = 2, kernel_f: int = 3,
                 rnn_hidden: int = 256, rnn_layers: int = 2,
                 lookahead_frames: int = 0, ref_context_frames: int = 16,
                 echo_gate_memory_frames: int = 32,
                 ref_active_floor_dbfs: float = -70.0,
                 compress_exponent: float = 0.5, mask_max: float = 2.0,
                 aux_echo_head: bool = True, echo_head_ref_gated: bool = True,
                 aux_noise_psd_head: bool = True,
                 use_deep_filter: bool = True, df_bins: int = 48,
                 df_order: int = 3, df_lookahead: int = 0,
                 noise_psd_eps: float = 1e-12):
        super().__init__()
        self.grid = grid
        self.enc_channels = enc_channels
        self.enc_stages = enc_stages
        self.lookahead_frames = int(lookahead_frames)
        self.ref_context_frames = max(1, int(ref_context_frames))
        self.echo_gate_memory_frames = max(1, int(echo_gate_memory_frames))
        self.ref_active_floor_power = 10.0 ** (ref_active_floor_dbfs / 10.0)
        self.compress_exponent = compress_exponent
        self.mask_max = mask_max
        self.aux_echo_head = bool(aux_echo_head)
        self.echo_head_ref_gated = bool(echo_head_ref_gated)
        self.aux_noise_psd_head = bool(aux_noise_psd_head)
        self.use_deep_filter = bool(use_deep_filter)
        self.df_order = int(df_order)
        self.df_lookahead = int(df_lookahead)
        self.noise_psd_eps = noise_psd_eps

        # -- frequency resolution at each depth -------------------------
        freqs = [grid.n_freqs]
        for _ in range(enc_stages):
            freqs.append((freqs[-1] - 1) // 2 + 1)
        if freqs[-1] < 4:
            raise ValueError(
                f"enc_stages={enc_stages} reduces {grid.n_freqs} bins to "
                f"{freqs[-1]}; the bottleneck would see almost no frequency "
                f"structure.  Lower enc_stages or raise n_fft.")
        self.freqs = freqs

        self.df_bins = min(int(df_bins), grid.n_freqs) if self.use_deep_filter else 0
        if self.use_deep_filter and not 0 < self.df_bins <= grid.n_freqs:
            raise ValueError(f"df_bins must be in (0, {grid.n_freqs}], got {df_bins}")

        act = nn.LeakyReLU(0.01)
        self.act = act

        # -- encoders ---------------------------------------------------
        # ⚠ Reference branch: bias=False everywhere, LeakyReLU only, no norm.
        # That is what makes ref_pathway(0) == 0 exactly.  Adding a bias to any
        # module reachable from x_spec silently deletes the zero-reference
        # guarantee AND the structural D_hat == 0 property, and nothing except
        # tests/test_joint_aecnr.py would notice.
        self.mic_encoder = nn.ModuleList()
        self.ref_encoder = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for stage in range(enc_stages):
            in_ch = 2 if stage == 0 else enc_channels
            self.mic_encoder.append(CausalConv2d(
                in_ch, enc_channels, kernel_t, kernel_f, stride_f=2, bias=True))
            self.ref_encoder.append(CausalConv2d(
                in_ch, enc_channels, kernel_t, kernel_f, stride_f=2, bias=False))
            self.fusion.append(
                nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=False))

        # Delay search.  A 1x1 fusion can only align the two branches at zero
        # lag, but the reference leads the echo by 10-120 ms of playout and
        # capture buffering.  This depthwise time-only conv gives the fusion a
        # window of past reference frames to pick the lag out of; it sits at the
        # deepest (cheapest, most abstract) stage, so its cost is
        # enc_channels * taps parameters and nothing per frequency bin.
        self.ref_context = CausalConv2d(
            enc_channels, enc_channels, kernel_t=self.ref_context_frames,
            kernel_f=1, stride_f=1, groups=enc_channels, bias=False)

        # -- recurrent bottleneck --------------------------------------
        bottleneck_width = enc_channels * freqs[-1]
        self.bottleneck_in = nn.Linear(bottleneck_width, rnn_hidden)
        self.rnn = nn.GRU(rnn_hidden, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True)
        self.bottleneck_out = nn.Linear(rnn_hidden, bottleneck_width)
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers

        # -- decoder ----------------------------------------------------
        self.skip_path = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for stage in range(enc_stages):
            self.skip_path.append(
                nn.Conv2d(enc_channels, enc_channels, kernel_size=1, bias=True))
            self.decoder.append(FreqUpsample(
                enc_channels, enc_channels, in_freqs=freqs[stage + 1],
                out_freqs=freqs[stage], kernel_f=kernel_f, stride_f=2))

        # -- heads ------------------------------------------------------
        pad_f = (kernel_f - 1) // 2
        self.mask_head = nn.Conv2d(enc_channels, 2, (1, kernel_f), padding=(0, pad_f))
        self.echo_head = (
            nn.Conv2d(enc_channels, 2, (1, kernel_f), padding=(0, pad_f))
            if self.aux_echo_head else None)
        self.noise_psd_head = (
            nn.Conv2d(enc_channels, 1, (1, kernel_f), padding=(0, pad_f))
            if self.aux_noise_psd_head else None)
        self.df_head = (
            nn.Conv2d(enc_channels, 2 * self.df_order, (1, kernel_f),
                      padding=(0, pad_f))
            if self.use_deep_filter else None)

    # ---------------- construction from config ----------------

    @classmethod
    def from_config(cls, cfg, grid: AecGrid) -> 'JointAECNR':
        """Build from a parsed ``config.ini`` and the resolved grid.

        Every duration is converted here and nowhere else, so there is exactly
        one place where "0.25 s" becomes "16 frames".
        """
        rate = grid.frame_rate
        return cls(
            grid=grid,
            enc_channels=cfg.getint('model', 'enc_channels'),
            enc_stages=cfg.getint('model', 'enc_stages'),
            kernel_t=cfg.getint('model', 'kernel_t', fallback=2),
            kernel_f=cfg.getint('model', 'kernel_f', fallback=3),
            rnn_hidden=cfg.getint('model', 'rnn_hidden'),
            rnn_layers=cfg.getint('model', 'rnn_layers'),
            lookahead_frames=frames_from_seconds(
                cfg.getfloat('model', 'lookahead_sec'), rate),
            ref_context_frames=frames_from_seconds(
                cfg.getfloat('model', 'ref_context_sec'), rate, minimum=1),
            echo_gate_memory_frames=frames_from_seconds(
                cfg.getfloat('model', 'echo_gate_memory_sec'), rate, minimum=1),
            ref_active_floor_dbfs=cfg.getfloat('model', 'ref_active_floor_dbfs'),
            compress_exponent=cfg.getfloat('model', 'compress_exponent'),
            mask_max=cfg.getfloat('model', 'mask_max'),
            aux_echo_head=cfg.getboolean('model', 'aux_echo_head'),
            echo_head_ref_gated=cfg.getboolean('model', 'echo_head_ref_gated'),
            aux_noise_psd_head=cfg.getboolean('model', 'aux_noise_psd_head'),
            use_deep_filter=cfg.getboolean('model', 'use_deep_filter'),
            df_bins=bins_from_hz(cfg.getfloat('model', 'df_band_hz'), grid),
            df_order=frames_from_seconds(
                cfg.getfloat('model', 'df_span_sec'), rate, minimum=1),
            df_lookahead=frames_from_seconds(
                cfg.getfloat('model', 'df_lookahead_sec'), rate),
            noise_psd_eps=cfg.getfloat('model', 'noise_psd_eps', fallback=1e-12),
        )

    # ---------------- introspection ----------------

    @property
    def reference_receptive_field_frames(self) -> int:
        """How long X must have been exactly zero for the ref pathway to be 0.

        ⚠ Report this next to any idle-behaviour claim.  "The model ignores a
        silent reference" is true only after this many frames of silence; in
        the transition immediately after the far end drops out, the pathway is
        still carrying real reference history -- and it must be, because the
        echo of what was just played is still arriving.
        """
        per_stage = sum(conv.history for conv in self.ref_encoder)
        return per_stage + self.ref_context.history + 1

    def reference_pathway_parameters(self):
        """Every parameter reachable from ``x_spec`` and from nothing else.

        The hard-gate test randomises exactly these and asserts the output does
        not move; keeping the list next to the modules stops it drifting when a
        module is added.
        """
        modules = list(self.ref_encoder) + list(self.fusion) + [self.ref_context]
        for module in modules:
            yield from module.parameters()

    # ---------------- streaming state ----------------

    def init_state(self, batch_size: int, device=None,
                   dtype=torch.float32) -> Dict[str, torch.Tensor]:
        """Zeroed state for ``batch_size`` independent lanes."""
        kwargs = {'device': device, 'dtype': dtype}
        state: Dict[str, torch.Tensor] = {}
        for stage, (mic, ref) in enumerate(zip(self.mic_encoder, self.ref_encoder)):
            in_ch = 2 if stage == 0 else self.enc_channels
            width = self.freqs[stage]
            if mic.history:
                state[f'mic{stage}'] = torch.zeros(
                    batch_size, in_ch, mic.history, width, **kwargs)
            if ref.history:
                state[f'ref{stage}'] = torch.zeros(
                    batch_size, in_ch, ref.history, width, **kwargs)
        if self.ref_context.history:
            state['ref_ctx'] = torch.zeros(
                batch_size, self.enc_channels, self.ref_context.history,
                self.freqs[-1], **kwargs)
        if self.echo_gate_memory_frames > 1:
            state['ref_power'] = torch.zeros(
                batch_size, self.echo_gate_memory_frames - 1, **kwargs)
        state['rnn'] = torch.zeros(
            self.rnn_layers, batch_size, self.rnn_hidden, **kwargs)
        df_history = self.df_order - self.df_lookahead - 1
        if self.use_deep_filter and df_history > 0:
            state['df'] = torch.zeros(
                batch_size, self.df_bins, 2, df_history, **kwargs)
        return state

    # ---------------- forward ----------------

    def forward(self, y_spec: torch.Tensor, x_spec: torch.Tensor,
                state: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[JointOutputs, Dict[str, torch.Tensor]]:
        """
        Args:
            y_spec: (B, F, T) complex microphone spectrum
            x_spec: (B, F, T) complex far-end reference spectrum
            state:  previous streaming state, or None for a cold start
        """
        if y_spec.shape != x_spec.shape:
            raise ValueError(
                f"Y {tuple(y_spec.shape)} and X {tuple(x_spec.shape)} must have "
                f"the same shape; X is a REFERENCE aligned to the same frames, "
                f"not an independently framed signal")
        if y_spec.shape[1] != self.grid.n_freqs:
            raise ValueError(
                f"got {y_spec.shape[1]} frequency bins, model was built for "
                f"{self.grid.n_freqs} (n_fft={self.grid.n_fft})")
        state = {} if state is None else dict(state)
        new_state: Dict[str, torch.Tensor] = {}

        y_feat = compress_complex(y_spec, self.compress_exponent)
        x_feat = compress_complex(x_spec, self.compress_exponent)
        if self.lookahead_frames:
            # ⚠ Chunk-boundary artefact, and the reason lookahead_sec defaults
            # to 0: the last `lookahead_frames` of every chunk see zeros where
            # a real stream would supply the next chunk's frames.  The
            # alternative -- delaying the output by the same amount -- would
            # misalign S_hat against the target S and is the trainer's problem
            # to solve, not the model's to hide.
            y_feat = self._advance(y_feat)
            x_feat = self._advance(x_feat)

        # -- dual encoder with cross-branch fusion ---------------------
        mic, ref = y_feat, x_feat
        skips = []
        for stage in range(self.enc_stages):
            mic, tail = self.mic_encoder[stage](mic, state.get(f'mic{stage}'))
            if tail is not None:
                new_state[f'mic{stage}'] = tail
            mic = self.act(mic)

            ref, tail = self.ref_encoder[stage](ref, state.get(f'ref{stage}'))
            if tail is not None:
                new_state[f'ref{stage}'] = tail
            ref = self.act(ref)

            if stage == self.enc_stages - 1:
                ref, tail = self.ref_context(ref, state.get('ref_ctx'))
                if tail is not None:
                    new_state['ref_ctx'] = tail

            mic = mic + self.fusion[stage](ref)
            skips.append(mic)

        # -- recurrent bottleneck --------------------------------------
        batch, channels, frames, width = mic.shape
        flat = mic.permute(0, 2, 1, 3).reshape(batch, frames, channels * width)
        hidden, rnn_state = self.rnn(self.bottleneck_in(flat), state.get('rnn'))
        new_state['rnn'] = rnn_state
        decoded = self.bottleneck_out(hidden)
        mic = decoded.reshape(batch, frames, channels, width).permute(0, 2, 1, 3)

        # -- decoder ---------------------------------------------------
        for stage in reversed(range(self.enc_stages)):
            mic = self.act(mic + self.skip_path[stage](skips[stage]))
            mic = self.decoder[stage](mic)

        # -- reference activity ----------------------------------------
        ref_gate, tail = reference_activity_gate(
            x_spec, self.echo_gate_memory_frames, self.ref_active_floor_power,
            state.get('ref_power'))
        if tail is not None:
            new_state['ref_power'] = tail

        # -- heads -----------------------------------------------------
        mask = self._bounded_mask(self.mask_head(mic))
        speech = mask * y_spec

        if self.use_deep_filter:
            coefs = self.df_head(mic)[..., :self.df_bins]          # (B,2K,T,Fdf)
            coefs = coefs.permute(0, 2, 3, 1)                       # (B,T,Fdf,2K)
            speech, tail = deep_filter_apply(
                speech, coefs, self.df_bins, self.df_order,
                self.df_lookahead, state.get('df'))
            if tail is not None:
                new_state['df'] = tail

        echo = None
        if self.echo_head is not None:
            echo = self._bounded_mask(self.echo_head(mic)) * y_spec
            if self.echo_head_ref_gated:
                # ⚠ This multiplication is what makes "silent reference implies
                # D_hat == 0" structural rather than hoped-for.  The gate has
                # echo_gate_memory_sec of memory precisely so the echo TAIL of a
                # burst that has just ended can still be cancelled; a gate on
                # the current frame alone would forbid that and look like a
                # model that cannot handle reverberant rooms.
                echo = echo * ref_gate.unsqueeze(1)

        noise_log_psd = None
        if self.noise_psd_head is not None:
            noise_log_psd = self.noise_psd_head(mic).squeeze(1).transpose(1, 2)

        return (
            JointOutputs(speech_spec=speech, mask=mask, echo_spec=echo,
                         noise_log_psd=noise_log_psd, ref_gate=ref_gate),
            new_state,
        )

    # ---------------- internals ----------------

    def _advance(self, feat: torch.Tensor) -> torch.Tensor:
        """Shift a (B, C, T, F) feature stream `lookahead_frames` into the future."""
        if self.lookahead_frames >= feat.shape[2]:
            raise ValueError(
                f"lookahead is {self.lookahead_frames} frames but the chunk is "
                f"only {feat.shape[2]}; every output frame would be looking at "
                f"padding")
        shifted = feat[:, :, self.lookahead_frames:, :]
        return F.pad(shifted, (0, 0, 0, self.lookahead_frames))

    def _bounded_mask(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, 2, T, F) real -> (B, F, T) complex mask with |M| <= mask_max.

        ⚠ ``mask_max`` must be comfortably ABOVE 1.  ``|M| = mask_max *
        tanh(|m|)`` reaches ``mask_max`` only asymptotically, so at
        ``mask_max = 1`` the identity mask -- the one the idle case needs -- is
        unreachable and the model would always attenuate slightly.
        """
        raw = raw.permute(0, 3, 2, 1).contiguous()      # (B, F, T, 2)
        complex_raw = torch.view_as_complex(raw)
        # ⚠ Floor inside the square root rather than a clamp after a norm.  Both
        # give a mask of exactly zero when the head sits at zero, and both have
        # a finite gradient -- but the clamped form divides by the clamp, so its
        # gradient near zero is scaled by 1/1e-12, i.e. 1e12.  Flooring inside
        # gives a magnitude of 1e-6 there and a gradient six orders of magnitude
        # smaller.  A head CAN sit at exactly zero for a whole frame, so this is
        # reachable, not theoretical.
        magnitude = torch.sqrt(raw.pow(2).sum(dim=-1) + 1e-12)
        bounded = self.mask_max * torch.tanh(magnitude)
        return complex_raw * (bounded / magnitude)


# ============================================================
# State helpers (the trainer's, not the model's)
# ============================================================

def detach_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Cut the graph between chunks -- truncated BPTT."""
    return {key: value.detach() for key, value in state.items()}


def reset_state(state: Dict[str, torch.Tensor],
                reset: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Zero the lanes flagged by ``reset`` (a ``(B,)`` bool tensor).

    Used with ``dataset_gen.aec.lane_reset_mask`` when a lane moves on to a new
    sequence.  ⚠ Skipping this carries one talker's adaptation state into the
    next sequence, which makes cold-start convergence look better than it is --
    the single most flattering bug available in this whole project.
    """
    out = {}
    for key, value in state.items():
        keep = (~reset).to(value.dtype)
        # 'rnn' is (layers, B, hidden); everything else is (B, ...).
        shape = ([1, -1] + [1] * (value.dim() - 2) if key == 'rnn'
                 else [-1] + [1] * (value.dim() - 1))
        out[key] = value * keep.reshape(shape)
    return out


# ============================================================
# Hard gate
# ============================================================

def idle_gate_report(model: JointAECNR, y_spec: torch.Tensor,
                     state: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """⚠ HARD GATE: with X == 0, how much does the model touch the microphone?

    Runs the model with an all-zero reference and reports, in dB:

      ``mic_delta_db``   20*log10(||S_hat - Y|| / ||Y||).  This is the number
                         the gate is about.  It is NOT expected to be -inf for
                         this model: JointAECNR also removes noise, so on a
                         noisy mic a large delta is correct behaviour.  Feed it
                         a CLEAN near-only mic and it must be deeply negative.
      ``echo_energy_db`` 20*log10(||D_hat|| / ||Y||).  With the reference
                         silent for longer than ``echo_gate_memory_frames``
                         this is structurally -inf; anything finite means the
                         gate has been removed or the state was not cold.

    Returns a dict rather than asserting, because the pass threshold depends on
    what mic you fed it, and a helper that hides that would let a caller claim
    the gate passed on noisy input.
    """
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs, _ = model(y_spec, torch.zeros_like(y_spec), state)
    if model_was_training:
        model.train()

    reference = y_spec.abs().pow(2).sum().sqrt()
    delta = (outputs.speech_spec - y_spec).abs().pow(2).sum().sqrt()
    report = {
        'mic_delta_db': _safe_db(delta, reference),
        'reference_receptive_field_frames': model.reference_receptive_field_frames,
        'echo_gate_memory_frames': model.echo_gate_memory_frames,
    }
    if outputs.echo_spec is not None:
        echo = outputs.echo_spec.abs().pow(2).sum().sqrt()
        report['echo_energy_db'] = _safe_db(echo, reference)
    return report


def _safe_db(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if float(numerator) == 0.0:
        return float('-inf')
    return float(20.0 * torch.log10(numerator / denominator.clamp_min(1e-20)))
