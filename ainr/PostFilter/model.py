"""PostFilter: the joint residual-echo + noise suppressor.

    in : (E, D_hat[, X])  E = Y - D_hat (AEC output), D_hat = echo estimate
    out: G in [0,1] per ERB band (expanded to bins) or per bin
    target: S, the near-end speech including its own room reverberation

WHY THE FEATURES LOOK LIKE THIS
-------------------------------
(E, D_hat) arrive from a FROZEN upstream front-end which may be a classical
linear canceller or a learned one, and the two leave behind visibly different
residuals.  What must NOT differ is the input distribution's *scale*: an
upstream gain retune, a different normalisation convention, or simply a
front-end that emits half-scale output would otherwise shift every feature and
silently invalidate the weights.

So every channel this model sees is a ratio or a log-domain difference,
computed after E, D_hat and Y have been divided by ``sqrt(mean(|E|^2 +
|D_hat|^2))`` of the same frame.  Multiplying the front-end's whole output by a
constant leaves the features numerically unchanged.  The one exception,
``include_absolute_level``, is ON by default -- see PostFilterFeatures for
the evidence that forced that default.

⚠ The suppression decision ends here.  Downstream there is no residual-echo
suppressor and no min(g_nr, g_res) fusion, so the preset floor and the
attenuation cap in postproc.py are the caller's, not the network's -- they are
never applied inside ``forward``.

Convention: complex spectra are ``(B, F, T)``; feature and conv tensors are
``(B, C, T, K)`` with K the mask axis (bands or bins), matching the sibling
projects' ``(B, C, T, F)`` layout.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.aec import AecGrid, alpha_from_tau, frames_from_seconds  # noqa: E402


__all__ = [
    'PostFilterFeatures',
    'PostFilterNet',
    'build_band_matrices',
    'build_model',
    'compute_erb_matrix',
    'erb2freq',
    'erb_bandborder',
    'freq2erb',
    'mask_magnitude',
    'resolve_downsamples',
]


# Numeric floors.  Both act on quantities that have already been normalised to
# O(1) by the frame reference, so they only ever bite on digital silence.
_EPS_POWER = 1e-10      # inside log10(.) of a normalised band power
_EPS_REF = 1e-20        # guards the frame reference itself against 0/0
_LOG_CLAMP = 6.0        # +-120 dB, i.e. the floors above, expressed as log10


# ============================================================
# ERB filterbank
# ============================================================
#
# ⚠ This is a deliberate copy of DeepFilterNet2/model.py's implementation, which
# is itself the construction RNNoise-ERB/train.py uses.  It is copied rather
# than imported because importing across model projects would make PostFilter's
# feature definition depend on a 48 kHz project's file, and a change made there
# for its own reasons would silently alter this model's input.
# tests/test_model.py asserts the copy is still bit-identical to DeepFilterNet2's,
# so the duplication is guarded rather than trusted.

def freq2erb(freq_hz):
    """Hz -> ERB-number (Glasberg-Moore, the 9.265/24.7 constants)."""
    return 9.265 * np.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """ERB-number -> Hz (inverse of freq2erb)."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft, min_bins_per_band=2):
    """FFT-bin band borders on the ERB scale; ``nfftborder[0] = 0``, last = n_bins.

    Band widths are enforced by a strict greedy-forward minimum (every
    consecutive border pair is >= ``min_bins_per_band`` apart, only ever
    borrowing forward).  ⚠ Without this, the top of the range at 32 bands
    produces zero-width bands whose column of the matrix is all zeros -- a band
    the model can never influence and whose gradient is exactly 0.
    """
    n_bins = n_fft // 2 + 1
    high_lim = sr / 2.0
    bw = high_lim / (n_fft / 2.0)
    erb_lims = np.linspace(freq2erb(0.0), freq2erb(high_lim), n_bands)
    cutoffs = erb2freq(erb_lims)
    ideal = np.round((cutoffs + bw / 2.0) / bw).astype(int)
    nb = [0]
    for i in range(1, n_bands):
        nxt = max(int(ideal[i]), nb[-1] + min_bins_per_band)
        nxt = min(nxt, n_bins)
        nb.append(nxt)
    nb = np.array(nb, dtype=int)
    nb[-1] = n_bins
    return nb


def compute_erb_matrix(nfftborder, n_fft, mode=0):
    """Triangular ERB filterbank, shape ``(n_bins, n_bands)``.

    ``mode=0`` (forward / analysis): the two one-sided edge columns are doubled
    so their band energy is comparable to the interior triangles.
    ``mode=1`` (inverse / gain expansion): NOT doubled, so the columns form a
    partition of unity.  ⚠ That is what makes ``gain = 1`` expand to
    ``bin_gain = 1`` exactly and keeps an expanded gain inside [0,1] -- the
    expansion is then a convex combination, which the range test relies on.
    """
    n_bins = n_fft // 2 + 1
    n_cols = len(nfftborder)
    matrix = np.zeros((n_bins, n_cols), dtype=np.float32)
    for i in range(n_cols - 1):
        lo, hi = int(nfftborder[i]), int(nfftborder[i + 1])
        width = hi - lo
        for j in range(width):
            matrix[lo + j, i] = 1.0 - j / width
            matrix[lo + j, i + 1] = j / width
    if mode == 0:
        matrix[:, 0] *= 2.0
        matrix[:, n_cols - 1] *= 2.0
    return matrix


def build_band_matrices(grid: AecGrid, n_bands: int, resolution: str):
    """``(analysis, synthesis)`` matrices, both ``(K, n_freqs)``.

    At ``resolution='full'`` both are the identity and ``K == n_freqs``: the
    model then reads and writes per bin, and every downstream shape expression
    stays the same instead of forking into two code paths.
    """
    if resolution == 'full':
        eye = torch.eye(grid.n_freqs, dtype=torch.float32)
        return eye, eye.clone()
    if resolution != 'band':
        raise ValueError(f"mask_resolution must be 'band' or 'full', got {resolution!r}")
    if n_bands < 4 or n_bands > grid.n_freqs:
        raise ValueError(
            f"n_bands={n_bands} is outside [4, n_freqs={grid.n_freqs}]")
    borders = erb_bandborder(n_bands, grid.sr, grid.n_fft)
    analysis = compute_erb_matrix(borders, grid.n_fft, mode=0).T.copy()
    synthesis = compute_erb_matrix(borders, grid.n_fft, mode=1).T.copy()
    # ⚠ The ERB helpers are standalone numpy ports and re-derive the bin count
    # from n_fft internally.  That agrees with AecGrid today; this pins it so a
    # future change to either definition fails here rather than producing a
    # filterbank silently misaligned with the spectra it will be applied to.
    for name, matrix in (('analysis', analysis), ('synthesis', synthesis)):
        assert matrix.shape[1] == grid.n_freqs, (
            f"{name} filterbank has {matrix.shape[1]} bins but the grid has "
            f"{grid.n_freqs}; the ERB helpers and AecGrid disagree")
    return torch.from_numpy(analysis), torch.from_numpy(synthesis)


# ============================================================
# Features
# ============================================================

class PostFilterFeatures(nn.Module):
    """Scale-invariant band (or bin) features from ``(E, D_hat, X)``.

    Channels, in order:

    ==  ==================================================================
    0   ``log10(band|E|^2)`` of the reference-normalised E -- spectral shape
    1   ``log10(band|D_hat|^2 / band|E|^2)`` -- the echo-estimate-to-output
        ratio.  The single most informative residual-echo cue, and the one
        the spec names first.
    2   ``log10(band|E|^2 / band|Y|^2)`` -- how much the linear stage already
        removed.  Distinguishes "nothing to cancel" from "cancelled well".
    3   banded magnitude-squared coherence between E and D_hat, in [0,1].
        Residual echo is the part of E still correlated with the echo
        estimate; near-end speech and local noise are not.
    4   ``channel 0`` minus its own causal EMA -- temporal contrast, which
        separates stationary noise from speech WITHOUT an absolute level.
    5   ``log10(band|X|^2)``, X normalised by its OWN frame energy   (optional)
    6   banded coherence between E and X                             (optional)
    7   ``log10(frame reference energy)``                            (optional, ⚠)
    ==  ==================================================================

    Channels for X exist because a LINEAR front-end's D_hat cannot contain the
    loudspeaker's nonlinear products; X is the only place they are visible.

    ⚠ WHICH SIDE-INPUTS TO USE IS AN OPEN QUESTION IN THE LITERATURE, not a
    settled recipe.  Two exhaustive-enough ablations reach opposite conclusions:

      - Franzen & Fingscheidt (ICASSP 2022) swap X for D_hat and gain
        +0.23 PESQ / +4.15 dB ERLE_BB; their winner is {Y, D_hat, E} and they
        write "the often used loudspeaker reference signal should not be used
        for this task".
      - Align-ULCNet (EUSIPCO 2025) is the only paper with a quantitative input
        table and concludes the reverse, DELETING the echo estimate its own
        predecessor used: {E, X} scores DT/FST 4.20/4.69, and adding D_hat drops
        it to 4.04/4.18.  Valin (ICASSP 2021) agrees on X, arguing that X "does
        not depend on the AEC behaviour, [so] convergence problems with the echo
        canceller are less likely to affect the RES performance".

    A plausible reconciliation -- ours, not either paper's -- is that D_hat wins
    when the front-end is healthy (Franzen: synthetic, well-behaved) and X wins
    when it is not (Align-ULCNet deliberately measured on clips where the Kalman
    filter FAILED).  Both are on by default here so the ablation can be run;
    neither should be assumed.

    ⚠ ONE THING IS SOLID ACROSS BOTH CAMPS: the microphone Y helps, and it helps
    the NOISE half.  It is on by default for that reason.

    ⚠ SCALE INVARIANCE IS THIS PROJECT'S HYPOTHESIS, NOT ESTABLISHED PRACTICE.
    A survey of twelve published residual-echo suppressors found the dominant
    input convention is RAW compressed/log spectra with either no normalisation
    or a FIXED ABSOLUTE scale.  The only published precedent for ratio inputs
    (Pfeifenberger & Pernkopf, Interspeech 2020) never justifies the choice and
    never tests front-end portability.  The strongest counter-evidence is that
    the one shipped production neural residual-echo estimator hard-codes an
    absolute scale (a 1/32768 constant with the comment "Trained model expects
    [-1,1]-scaled signals") and feeds raw power spectra -- no ratio, no per-frame
    normalisation.  Where ratios genuinely ARE universal is the OUTPUT side: the
    mask multiplies E, and the usual targets (IRM, IAM, PSM) are themselves
    ratios, which already buys most of the front-end-scale robustness for free.

    So ``include_absolute_level`` now defaults to TRUE: ratio channels ALONGSIDE
    an absolute one, not instead of it.  Set it false to run the pure-ratio
    variant, which is the interesting experiment -- nobody has published it.

    ⚠ State (three EMAs) is carried across chunks of a sequence by the trainer.
    Reinitialising it every chunk resets the coherence estimate to 0 every four
    seconds, which reads as "no residual echo" for the first ~60 ms of every
    chunk -- exactly the window in which an echo-path change happens.
    """

    def __init__(self, grid: AecGrid, n_bands: int, resolution: str = 'band',
                 use_reference: bool = True, coherence_tau_sec: float = 0.06,
                 level_tau_sec: float = 1.0,
                 include_absolute_level: bool = True,
                 use_mic: bool = True):
        super().__init__()
        analysis, synthesis = build_band_matrices(grid, n_bands, resolution)
        self.register_buffer('analysis', analysis)      # (K, n_freqs)
        self.register_buffer('synthesis', synthesis)    # (K, n_freqs)
        self.grid = grid
        self.resolution = resolution
        self.n_out = analysis.shape[0]
        self.use_reference = bool(use_reference)
        self.include_absolute_level = bool(include_absolute_level)
        self.use_mic = bool(use_mic)
        # ⚠ Both alphas come from a time constant in SECONDS.  A literal 0.92
        # here would mean 220 ms at 16 kHz/hop 256 and 73 ms at 48 kHz/hop 512,
        # so the 48 kHz variant would quietly become a different algorithm.
        self.alpha_coh = alpha_from_tau(coherence_tau_sec, grid.hop_len, grid.sr)
        self.alpha_level = alpha_from_tau(level_tau_sec, grid.hop_len, grid.sr)
        self.coherence_tau_sec = float(coherence_tau_sec)
        self.level_tau_sec = float(level_tau_sec)

    @property
    def n_channels(self) -> int:
        return (5
                + (1 if self.use_mic else 0)
                + (2 if self.use_reference else 0)
                + (1 if self.include_absolute_level else 0))

    # ---------------- state ----------------

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> dict:
        k = self.n_out
        zeros = lambda: torch.zeros(batch, k, device=device, dtype=dtype)  # noqa: E731
        state = {
            'coh_ed': torch.zeros(batch, k, device=device, dtype=torch.complex64),
            'coh_ee': zeros(),
            'coh_dd': zeros(),
            'level': zeros(),
            'level_init': torch.zeros(batch, device=device, dtype=torch.bool),
        }
        if self.use_reference:
            state['coh_ex'] = torch.zeros(batch, k, device=device, dtype=torch.complex64)
            state['coh_xx'] = zeros()
        return state

    # ---------------- helpers ----------------

    def _band(self, per_bin: torch.Tensor) -> torch.Tensor:
        """``(B, n_freqs, T)`` -> ``(B, K, T)``.  Works for real and complex."""
        if self.resolution == 'full':
            return per_bin            # the matrix is the identity; skip the matmul
        matrix = self.analysis
        if torch.is_complex(per_bin):
            matrix = matrix.to(per_bin.dtype)
        return torch.matmul(matrix, per_bin)

    def expand(self, band_values: torch.Tensor) -> torch.Tensor:
        """``(B, K, T)`` band gains -> ``(B, n_freqs, T)`` bin gains.

        The synthesis matrix is a partition of unity, so this is a convex
        combination: a band gain in [0,1] cannot expand outside [0,1].
        """
        if self.resolution == 'full':
            return band_values
        return torch.matmul(self.synthesis.transpose(0, 1), band_values)

    # ---------------- forward ----------------

    def forward(self, E: torch.Tensor, D_hat: torch.Tensor,
                X: torch.Tensor = None, state: dict = None):
        """``(B, F, T)`` complex spectra -> ``((B, C, T, K), state)``."""
        if E.shape != D_hat.shape:
            raise ValueError(f"E {tuple(E.shape)} and D_hat {tuple(D_hat.shape)} must match")
        batch, _, n_frames = E.shape
        if state is None:
            state = self.init_state(batch, device=E.device)

        Y = E + D_hat

        # --- the frame reference.  Everything below is measured against it, so
        # scaling (E, D_hat) by any constant leaves every channel unchanged. ---
        pow_e = E.real.square() + E.imag.square()
        pow_d = D_hat.real.square() + D_hat.imag.square()
        ref = (pow_e + pow_d).mean(dim=1, keepdim=True) + _EPS_REF   # (B, 1, T)
        scale = ref.sqrt()
        e_n, d_n, y_n = E / scale, D_hat / scale, Y / scale

        band_e = self._band(e_n.real.square() + e_n.imag.square())
        band_d = self._band(d_n.real.square() + d_n.imag.square())
        band_y = self._band(y_n.real.square() + y_n.imag.square())

        log_e = torch.log10(band_e + _EPS_POWER)
        # ⚠ The clamp is what makes D_hat == 0 a well-defined input rather than
        # -inf: the ratio saturates at -120 dB and the model reads "no echo
        # estimate at all", which is precisely the ref_dropout / none-front-end
        # case it must handle.
        ratio_de = torch.clamp(
            torch.log10(band_d + _EPS_POWER) - log_e, -_LOG_CLAMP, _LOG_CLAMP)
        ratio_ey = torch.clamp(
            log_e - torch.log10(band_y + _EPS_POWER), -_LOG_CLAMP, _LOG_CLAMP)

        # Instantaneous cross-spectra.  Everything recursive below happens in ONE
        # frame loop: running the E auto-power through two separate loops (once
        # per coherence pair) would smooth it twice and quietly halve its
        # effective time constant.
        cross_ed = self._band(e_n * d_n.conj())            # (B, K, T) complex
        if self.use_reference:
            if X is None:
                raise ValueError("use_reference=True but no X was passed")
            # X gets its OWN reference: the far-end playback gain is set
            # independently of the mic path, so normalising it by the mic-side
            # reference would reintroduce exactly the cross-stage scale coupling
            # this whole design removes.
            pow_x = X.real.square() + X.imag.square()
            x_n = X / (pow_x.mean(dim=1, keepdim=True) + _EPS_REF).sqrt()
            band_x = self._band(x_n.real.square() + x_n.imag.square())
            cross_ex = self._band(e_n * x_n.conj())
        else:
            band_x = cross_ex = None

        coh_ed, coh_ex, level, state = self._recursive_channels(
            log_e, band_e, band_d, band_x, cross_ed, cross_ex, state)

        channels = [log_e, ratio_de, ratio_ey, coh_ed, level]
        if self.use_mic:
            # The microphone's OWN spectral shape, not just its role as the
            # denominator of ratio_ey.  Franzen & Fingscheidt (ICASSP 2022) ran
            # the only exhaustive side-input sweep for this block and found Y is
            # the one input that consistently helps the NOISE half: adding it to
            # E lifts noise-only dSNR from 14.74 to 22.38 dB.  Their stated
            # reason is that Y is UNPROCESSED BY THE AEC -- its noise has not yet
            # been distorted by the adaptive filter -- which a ratio against Y
            # throws away, because a ratio keeps only the relationship and
            # discards the shape.
            channels.append(torch.log10(band_y + _EPS_POWER))
        if self.use_reference:
            channels.append(torch.log10(band_x + _EPS_POWER))
            channels.append(coh_ex)

        if self.include_absolute_level:
            # The one channel that is NOT scale-invariant, and it is ON by
            # default -- see the class docstring for why the literature forced
            # that default.
            absolute = torch.log10(ref) * torch.ones_like(log_e)
            channels.append(torch.clamp(absolute, -_LOG_CLAMP, _LOG_CLAMP))

        # /3 puts the log channels roughly in [-2, 2].  A fixed input scaling,
        # part of FEATURE_VERSION -- changing it changes what the weights mean.
        feats = torch.stack(channels, dim=1) / 3.0          # (B, C, K, T)
        feats = feats.permute(0, 1, 3, 2).contiguous()      # (B, C, T, K)
        return feats, state

    def _recursive_channels(self, log_e, band_e, band_d, band_x,
                            cross_ed, cross_ex, state):
        """The EMA-driven channels: two coherences and the temporal contrast.

        Coherence is bounded in [0,1] by Cauchy-Schwarz with the non-negative
        band weights, so it needs no clamp and cannot poison the sigmoid.

        ⚠ The temporal-contrast EMA is seeded from the FIRST frame a lane ever
        sees, not from a calibrated constant.  A constant would be an absolute
        level in disguise and would make the first second of every stream
        scale-dependent; seeding from the first frame makes the channel exactly
        invariant instead, at the cost of reading 0 on frame one -- which is
        honest, there is no history to contrast against yet.
        """
        a_coh, a_lvl = self.alpha_coh, self.alpha_level
        s_ed, s_ee, s_dd = state['coh_ed'], state['coh_ee'], state['coh_dd']
        s_ex = state.get('coh_ex')
        s_xx = state.get('coh_xx')
        mean, initialised = state['level'], state['level_init']

        if not bool(initialised.all()):
            fresh = (~initialised).unsqueeze(-1).to(log_e.dtype)
            mean = mean * (1.0 - fresh) + log_e[..., 0] * fresh
            initialised = torch.ones_like(initialised)

        out_ed, out_ex, out_lvl = [], [], []
        for t in range(log_e.shape[-1]):
            s_ed = a_coh * s_ed + (1.0 - a_coh) * cross_ed[..., t]
            s_ee = a_coh * s_ee + (1.0 - a_coh) * band_e[..., t]
            s_dd = a_coh * s_dd + (1.0 - a_coh) * band_d[..., t]
            out_ed.append(s_ed.abs() / (s_ee * s_dd).clamp_min(_EPS_POWER).sqrt())
            if cross_ex is not None:
                s_ex = a_coh * s_ex + (1.0 - a_coh) * cross_ex[..., t]
                s_xx = a_coh * s_xx + (1.0 - a_coh) * band_x[..., t]
                out_ex.append(s_ex.abs() / (s_ee * s_xx).clamp_min(_EPS_POWER).sqrt())
            mean = a_lvl * mean + (1.0 - a_lvl) * log_e[..., t]
            out_lvl.append(torch.clamp(log_e[..., t] - mean, -_LOG_CLAMP, _LOG_CLAMP))

        state = dict(state)
        state.update(coh_ed=s_ed.detach(), coh_ee=s_ee.detach(), coh_dd=s_dd.detach(),
                     level=mean.detach(), level_init=initialised)
        if cross_ex is not None:
            state.update(coh_ex=s_ex.detach(), coh_xx=s_xx.detach())

        coh_ed = torch.stack(out_ed, dim=-1).clamp(0.0, 1.0)
        coh_ex = (torch.stack(out_ex, dim=-1).clamp(0.0, 1.0)
                  if cross_ex is not None else None)
        return coh_ed, coh_ex, torch.stack(out_lvl, dim=-1), state


# ============================================================
# Convolution blocks
# ============================================================

class CausalConv2d(nn.Module):
    """Conv over ``(B, C, T, K)``, causal in T with an explicit lookahead.

    ⚠ ``lookahead`` frames of right padding are what the model is ALLOWED to
    see; the caller pays hop_len/sr per frame in algorithmic delay for them.  It
    is a config knob and not a free accuracy win.

    ⚠ The left context is STATE, returned so it can be carried into the next
    chunk.  Zero-padding it per chunk instead makes the first ``kernel_t - 1``
    frames of every chunk see a fabricated silence -- a 32 ms glitch every four
    seconds, right where an echo-path change would show up.
    """

    def __init__(self, in_ch, out_ch, kernel_t, kernel_f, lookahead=0,
                 stride_f=1, groups=1):
        super().__init__()
        if not 0 <= lookahead <= kernel_t - 1:
            raise ValueError(
                f"lookahead {lookahead} must be in [0, kernel_t-1={kernel_t - 1}]")
        self.in_ch = in_ch
        self.pad_left = kernel_t - 1 - lookahead
        self.pad_right = lookahead
        self.pad_f = kernel_f // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (kernel_t, kernel_f),
                              stride=(1, stride_f), groups=groups)
        self.stride_f = stride_f
        self.kernel_f = kernel_f

    def context_shape(self, batch, width):
        return (batch, self.in_ch, self.pad_left, width)

    def forward(self, x, left_context=None):
        """``(x, left_context) -> (y, new_left_context)``."""
        if self.pad_left:
            if left_context is None:
                left_context = x.new_zeros(
                    *self.context_shape(x.shape[0], x.shape[3]))
            x = torch.cat([left_context, x], dim=2)
        new_context = x[:, :, x.shape[2] - self.pad_left:] if self.pad_left else None
        # ⚠ The right (lookahead) padding stays zero at a chunk boundary: the
        # future frames simply are not in this chunk.  With lookahead = 0 there
        # is no right padding and chunked processing is bit-exact against the
        # whole-sequence result (tests/test_model.py asserts it).
        x = F.pad(x, (self.pad_f, self.pad_f, 0, self.pad_right))
        return self.conv(x), new_context


class SeparableFreqConv(nn.Module):
    """Depthwise-over-frequency + pointwise, optionally halving the freq axis.

    No temporal kernel: the encoder's temporal context is bought once in
    ``CausalConv2d`` and everything after it gets its memory from the GRU.  That
    keeps the causal padding bookkeeping in exactly one place, which is where it
    is checkable.
    """

    def __init__(self, in_ch, out_ch, kernel_f=3, stride_f=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, (1, kernel_f),
                                   stride=(1, stride_f),
                                   padding=(0, kernel_f // 2), groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


def _downsampled_width(width, n_down):
    """Frequency width after ``n_down`` stride-2, kernel-3, pad-1 convolutions."""
    for _ in range(n_down):
        width = (width + 2 * 1 - 3) // 2 + 1
    return width


def resolve_downsamples(width, spec, target=33, minimum=2):
    """``'auto'`` -> the fewest halvings that bring ``width`` to <= ``target``.

    32 bands -> 2 (floor), 257 bins -> 3.  ⚠ The floor of 2 exists so that the
    band model still has a frequency-mixing encoder; without it a 32-band input
    would go straight to the GRU and the convolutional stage would do nothing.
    """
    if spec is None or str(spec).strip().lower() == 'auto':
        n_down = 0
        while _downsampled_width(width, n_down) > target or n_down < minimum:
            n_down += 1
            if _downsampled_width(width, n_down) < 2:
                raise ValueError(
                    f"cannot downsample width {width} to <= {target}: "
                    f"the frequency axis collapses first")
        return n_down
    n_down = int(spec)
    if n_down < 0 or _downsampled_width(width, n_down) < 2:
        raise ValueError(f"enc_downsamples={n_down} collapses width {width}")
    return n_down


# ============================================================
# The network
# ============================================================

def mask_magnitude(mask: torch.Tensor) -> torch.Tensor:
    """|mask|, for both output types.  This is the quantity postproc bounds."""
    return mask.abs() if torch.is_complex(mask) else mask


class PostFilterNet(nn.Module):
    """Convolutional encoder + GRU bottleneck -> per-band (or per-bin) gain.

    ``forward(E, D_hat, X, state) -> (mask, state)``.  ``mask`` is real and in
    [0,1] for ``output_type='gain'``; for ``'complex'`` it is ``G * e^{j theta}``
    with ``|mask| = G`` still in [0,1], so every caller-side bound in postproc.py
    applies unchanged.

    ⚠ ``forward`` returns the NETWORK's decision only.  The preset floor and the
    attenuation cap are not applied here -- see postproc.py.
    """

    def __init__(self, grid: AecGrid, n_bands: int = 32,
                 mask_resolution: str = 'band', output_type: str = 'gain',
                 use_reference: bool = True, coherence_tau_sec: float = 0.06,
                 level_tau_sec: float = 1.0, include_absolute_level: bool = True,
                 use_mic: bool = True,
                 enc_channels: int = 48, enc_kernel_t: int = 3,
                 enc_kernel_f: int = 3, enc_downsamples='auto',
                 gru_hidden: int = 352, gru_layers: int = 2,
                 dec_hidden: int = 128, lookahead_frames: int = 0):
        super().__init__()
        if output_type not in ('gain', 'complex'):
            raise ValueError(f"output_type must be 'gain' or 'complex', got {output_type!r}")
        if output_type == 'complex' and mask_resolution != 'full':
            # ⚠ One phase rotation per ERB band is meaningless: phase decorrelates
            # within a few bins, so a band-wide rotation is an average of
            # essentially random angles.  Refusing the combination is better than
            # shipping a variant that merely underperforms for an invisible reason.
            raise ValueError(
                "output_type='complex' requires mask_resolution='full'; a single "
                "phase rotation per ERB band has no physical meaning")

        self.features = PostFilterFeatures(
            grid, n_bands, resolution=mask_resolution, use_reference=use_reference,
            coherence_tau_sec=coherence_tau_sec, level_tau_sec=level_tau_sec,
            include_absolute_level=include_absolute_level, use_mic=use_mic)
        self.grid = grid
        self.mask_resolution = mask_resolution
        self.output_type = output_type
        self.lookahead_frames = int(lookahead_frames)
        self.n_out = self.features.n_out

        in_ch = self.features.n_channels
        self.n_down = resolve_downsamples(self.n_out, enc_downsamples)
        self.enc_channels = enc_channels
        self.enc_kernel_t = enc_kernel_t
        self.enc_kernel_f = enc_kernel_f

        self.encoder_in = CausalConv2d(in_ch, enc_channels, enc_kernel_t,
                                       enc_kernel_f, lookahead=lookahead_frames)
        self.encoder_in_norm = nn.BatchNorm2d(enc_channels)
        self.encoder_in_act = nn.ReLU(inplace=True)
        self.encoder = nn.ModuleList([
            SeparableFreqConv(enc_channels, enc_channels, enc_kernel_f, stride_f=2)
            for _ in range(self.n_down)
        ])
        self.enc_width = _downsampled_width(self.n_out, self.n_down)

        self.gru_in_dim = enc_channels * self.enc_width
        self.gru = nn.GRU(self.gru_in_dim, gru_hidden, num_layers=gru_layers,
                          batch_first=True)
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.dec_hidden = dec_hidden
        self.dec = nn.Sequential(
            nn.Linear(gru_hidden, dec_hidden),
            nn.ReLU(inplace=True),
        )
        self.gain_head = nn.Linear(dec_hidden, self.n_out)
        self.phase_head = (nn.Linear(dec_hidden, 2 * self.n_out)
                           if output_type == 'complex' else None)

    # ---------------- state ----------------

    def init_state(self, batch: int, device=None) -> dict:
        state = self.features.init_state(batch, device=device)
        state['gru'] = torch.zeros(self.gru_layers, batch, self.gru_hidden,
                                   device=device)
        if self.encoder_in.pad_left:
            state['conv_tail'] = torch.zeros(
                *self.encoder_in.context_shape(batch, self.n_out), device=device)
        return state

    def reset_lanes(self, state: dict, reset: torch.Tensor) -> dict:
        """Zero the lanes flagged by ``reset`` (``chunk_index == 0``).

        ⚠ Every state a lane owns must be listed here.  A forgotten EMA is
        invisible: the model still trains, it just carries the previous
        sequence's echo path into the first frames of the next one, which reads
        as a slow-converging model rather than a bug.
        """
        if state is None:
            return None
        reset = reset.to(dtype=torch.bool)
        out = {}
        for key, value in state.items():
            if key == 'gru':
                keep = (~reset).to(value.dtype).view(1, -1, 1)
                out[key] = value * keep
            elif key == 'level_init':
                out[key] = value & (~reset)
            else:
                keep = (~reset).to(value.dtype).view(-1, *([1] * (value.dim() - 1)))
                out[key] = value * keep
        return out

    @staticmethod
    def detach_state(state: dict) -> dict:
        """⚠ Truncated BPTT boundary.  Without it the autograd graph grows for
        the whole 60 s sequence and the first backward pass exhausts memory."""
        if state is None:
            return None
        return {k: v.detach() for k, v in state.items()}

    # ---------------- forward ----------------

    def forward(self, E, D_hat, X=None, state=None):
        feats, state = self.features(E, D_hat, X, state)      # (B, C, T, K)
        batch, _, n_frames, _ = feats.shape

        h, conv_tail = self.encoder_in(feats, state.get('conv_tail'))
        h = self.encoder_in_act(self.encoder_in_norm(h))
        for block in self.encoder:
            h = block(h)                                      # (B, ch, T, K')

        h = h.permute(0, 2, 1, 3).reshape(batch, n_frames, -1)
        gru_state = state.get('gru')
        if gru_state is None:
            gru_state = torch.zeros(self.gru_layers, batch, self.gru_hidden,
                                    device=feats.device, dtype=feats.dtype)
        h, gru_state = self.gru(h, gru_state.contiguous())
        state = dict(state)
        state['gru'] = gru_state
        if conv_tail is not None:
            state['conv_tail'] = conv_tail

        h = self.dec(h)                                       # (B, T, dec_hidden)
        gain = torch.sigmoid(self.gain_head(h))               # (B, T, K) in [0,1]
        gain = gain.transpose(1, 2)                           # (B, K, T)

        if self.phase_head is None:
            return gain, state

        phase = self.phase_head(h).transpose(1, 2)            # (B, 2K, T)
        cos, sin = phase[:, :self.n_out], phase[:, self.n_out:]
        # Unit-norm the rotation so |mask| == gain exactly; a free-magnitude
        # complex mask would break every bound postproc.py enforces.
        norm = torch.sqrt(cos * cos + sin * sin).clamp_min(1e-8)
        return torch.complex(gain * cos / norm, gain * sin / norm), state

    # ---------------- applying the mask ----------------

    def apply_mask(self, mask, E):
        """``mask`` (band or bin resolution) applied to ``E`` -> ``(B, F, T)``.

        Band gains are expanded through the ERB partition of unity first, so a
        band gain of 1 leaves E untouched bit for bit.
        """
        if self.mask_resolution == 'band':
            mask = self.features.expand(mask)
        if torch.is_complex(mask):
            return mask * E
        return mask.to(E.real.dtype) * E

    def expand_to_bins(self, mask):
        """Bin-resolution view of a mask, for callers that want G(f) itself."""
        if self.mask_resolution == 'band':
            return self.features.expand(mask)
        return mask

    # ---------------- complexity ----------------

    def macs_per_frame(self) -> int:
        """Multiply-accumulates for ONE frame of steady-state streaming.

        Counted the way a DSP budget counts them: multiplies in convolutions,
        GRU gates and linear layers.  BatchNorm folds into the preceding
        pointwise convolution at inference and is not counted; the feature
        front-end's band matmuls ARE counted, because they are real work on the
        target platform.
        """
        total = 0
        k = self.features.n_out
        n_freqs = self.grid.n_freqs
        if self.mask_resolution == 'band':
            # analysis of |E|^2,|D|^2,|Y|^2 and the coherence cross-terms, plus
            # the synthesis expansion of the output gain.
            n_analysis = 5 + (2 if self.features.use_reference else 0)
            total += n_analysis * k * n_freqs + k * n_freqs

        width = k
        total += (self.enc_kernel_t * self.enc_kernel_f
                  * self.features.n_channels * self.enc_channels * width)
        for _ in range(self.n_down):
            width = (width + 2 - 3) // 2 + 1
            total += self.enc_kernel_f * self.enc_channels * width          # depthwise
            total += self.enc_channels * self.enc_channels * width          # pointwise

        for layer in range(self.gru_layers):
            in_dim = self.gru_in_dim if layer == 0 else self.gru_hidden
            total += 3 * (in_dim * self.gru_hidden + self.gru_hidden * self.gru_hidden)

        total += self.gru_hidden * self.dec_hidden
        total += self.dec_hidden * self.n_out
        if self.phase_head is not None:
            total += self.dec_hidden * 2 * self.n_out
        return int(total)

    def macs_per_second(self) -> float:
        return self.macs_per_frame() * self.grid.frame_rate

    def describe(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return (
            f"PostFilterNet: {params:,} parameters, "
            f"{self.macs_per_frame():,} MACs/frame, "
            f"{self.macs_per_second() / 1e6:.1f} M MACs/s @ "
            f"{self.grid.frame_rate:.4g} fps\n"
            f"  resolution={self.mask_resolution} (K={self.n_out}), "
            f"output={self.output_type}, in_channels={self.features.n_channels}\n"
            f"  encoder={self.enc_channels}ch x{self.n_down} downsamples -> "
            f"K'={self.enc_width}, gru={self.gru_layers}x{self.gru_hidden}, "
            f"dec={self.dec_hidden}\n"
            f"  lookahead={self.lookahead_frames} frame(s) = "
            f"{1000.0 * self.lookahead_frames * self.grid.hop_len / self.grid.sr:.1f} ms"
        )


def build_model(cfg, grid: AecGrid) -> PostFilterNet:
    """Construct from a parsed config.ini.  Used by train.py and denoise.py so
    the two can never build subtly different models from the same file."""
    return PostFilterNet(
        grid,
        n_bands=cfg.getint('feature', 'n_bands', fallback=32),
        mask_resolution=cfg.get('model', 'mask_resolution', fallback='band'),
        output_type=cfg.get('model', 'output_type', fallback='gain'),
        use_reference=cfg.getboolean('feature', 'use_reference', fallback=True),
        coherence_tau_sec=cfg.getfloat('feature', 'coherence_tau_sec', fallback=0.06),
        level_tau_sec=cfg.getfloat('feature', 'level_tau_sec', fallback=1.0),
        include_absolute_level=cfg.getboolean(
            'feature', 'include_absolute_level', fallback=True),
        use_mic=cfg.getboolean('feature', 'use_mic', fallback=True),
        enc_channels=cfg.getint('model', 'enc_channels', fallback=48),
        enc_kernel_t=cfg.getint('model', 'enc_kernel_t', fallback=3),
        enc_kernel_f=cfg.getint('model', 'enc_kernel_f', fallback=3),
        enc_downsamples=cfg.get('model', 'enc_downsamples', fallback='auto'),
        gru_hidden=cfg.getint('model', 'gru_hidden', fallback=352),
        gru_layers=cfg.getint('model', 'gru_layers', fallback=2),
        dec_hidden=cfg.getint('model', 'dec_hidden', fallback=128),
        lookahead_frames=frames_from_seconds(
            cfg.getfloat('model', 'lookahead_sec', fallback=0.0),
            grid.frame_rate),
    )


if __name__ == '__main__':   # quick shape/complexity probe
    _grid = AecGrid(sr=16000, n_fft=512, win_len=512, hop_len=256)
    for _res, _out in (('band', 'gain'), ('full', 'gain'), ('full', 'complex')):
        _net = PostFilterNet(_grid, mask_resolution=_res, output_type=_out)
        print(_net.describe())
        _e = torch.randn(2, _grid.n_freqs, 20, dtype=torch.complex64)
        _m, _ = _net(_e, _e * 0.1, _e * 2)
        print(f"  mask {tuple(_m.shape)} {_m.dtype}, "
              f"|mask| in [{mask_magnitude(_m).min():.3f}, "
              f"{mask_magnitude(_m).max():.3f}]\n")
