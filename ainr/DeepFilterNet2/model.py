"""
DeepFilterNet2 adaptation with config-driven sample rate and FFT size.

Architecture overview (aligned with DeepFilterNet2 paper):
  - Encoder: ERB path (4 levels, freq stride-2 ×2) + DF path (2 levels)
             → joint embedding GRU
  - ERBDecoder: U-Net transposed conv × 4 → sigmoid ERB mask
  - DFDecoder: GRU → per-bin FIR coefficients
  - deep_filter_apply: causal real/imag FIR filtering on low-freq bins

Convention: (B, C, T, F) throughout (time = H dim 2, freq = W dim 3).
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ERB filterbank helpers
# ============================================================

def freq2erb(freq_hz):
    """Hz -> ERB-number (Glasberg-Moore). Verified bit-for-bit against
    upstream Rikorose/DeepFilterNet's libDF/src/lib.rs freq2erb (9.265/24.7
    constants) and this project's own aaronhsueh0506/DeepFilterNet-Keras
    bandERB.ipynb -- also the exact formula RNNoise-ERB's train.py uses."""
    return 9.265 * np.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """ERB-number -> Hz (inverse of freq2erb)."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft, min_bins_per_band=2):
    """Port of aaronhsueh0506/DeepFilterNet-Keras bandERB.ipynb's ERBBand():
    returns nfftborder, a length-N (N = n_bands) int array of FFT-bin band
    borders on the Glasberg-Moore ERB scale. nfftborder[0]=0 (DC),
    nfftborder[-1]=n_fft//2+1 (Nyquist+1). N borders -> N ERB bands (one
    matrix column each, see compute_erb_matrix):
        cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N))
        ideal border = round((cutoff + bw/2) / bw), bw = sr/n_fft

    Band-width enforcement: strict greedy-forward minimum (every consecutive
    border pair is >= min_bins_per_band bins apart, only ever borrowing
    forward). This is RNNoise-ERB train.py's own v5 fix, applied here from
    the start: the original notebook's "every-OTHER-band-pair >= 2" rule
    (checked only i, i+2) does not actually guarantee every individual band
    is >= 2 bins wide.
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
    """Port of bandERB.ipynb's ERB_pro_matrix(nfftborder, NFFT, mode) --
    identical to RNNoise-ERB train.py's compute_erb_matrix(). Triangular ERB
    filterbank, shape (n_bins, N) with N = len(nfftborder) bands. The
    range(N-1) blocks lie BETWEEN consecutive borders, so column 0 (falling
    ramp only) and column N-1 (rising ramp only) are ONE-SIDED; interior
    columns are full triangles.
        mode=0 (forward / feature ERBB): x2 on the two one-sided edge
               columns so their band energy is comparable to interior bands.
        mode=1 (inverse / mask expansion): no x2 -- a clean partition of
               unity, so gain=1 maps to bin_gain=1 with NO row normalisation.
    """
    n_bins = n_fft // 2 + 1
    N = len(nfftborder)
    W = np.zeros((n_bins, N), dtype=np.float32)
    for i in range(N - 1):
        lo, hi = int(nfftborder[i]), int(nfftborder[i + 1])
        bs = hi - lo
        for j in range(bs):
            W[lo + j, i] = 1.0 - j / bs
            W[lo + j, i + 1] = j / bs
    if mode == 0:
        W[:, 0] *= 2.0
        W[:, N - 1] *= 2.0
    return W


def _build_erb_fb(n_fft: int, sr: int, n_erb: int):
    """
    Build ERB band-assignment matrices for the configured sample rate and FFT.
    Ported from this project's own aaronhsueh0506/DeepFilterNet-Keras
    bandERB.ipynb (ERBBand()/ERB_pro_matrix()) -- the same triangular
    construction RNNoise-ERB's train.py uses (erb_bandborder()/
    compute_erb_matrix()), including the v5 strict-minimum-band-width fix.

    Returns:
        erb_fb   : (n_erb, n_bins) — forward transform (mode=0: edge columns
                   doubled so their band energy is comparable to interior
                   bands, matching bandERB.ipynb -- NOT normalised to sum 1)
        erb_inv  : (n_erb, n_bins) — inverse transform (mode=1: clean
                   partition-of-unity by construction, no row normalisation)
    """
    borders = erb_bandborder(n_erb, sr, n_fft)
    fb = compute_erb_matrix(borders, n_fft, mode=0).T.copy()   # (n_erb, n_bins)
    inv = compute_erb_matrix(borders, n_fft, mode=1).T.copy()  # (n_erb, n_bins)
    return torch.from_numpy(fb), torch.from_numpy(inv)


# ============================================================
# Building blocks
# ============================================================

class SeparableConv2d(nn.Module):
    """Separable Conv2d + BN + ReLU, causal on the time axis.

    Matches upstream ``Conv2dNormAct`` (df/modules.py):
      * grouping is ``gcd(in_ch, out_ch)`` and the *first* conv already changes
        the channel count, followed by a 1x1 pointwise.  When in_ch == out_ch
        this is the usual depthwise+pointwise pair; when they differ (the input
        convs) it is NOT, which is why this used to diverge.
      * activation is ReLU, not PReLU.
      * the time axis is padded ``kernel[0] - 1`` on the LEFT ONLY, so the conv
        is strictly causal.  The previous version always passed
        ``padding=(0, kf//2)``, i.e. zero time padding — correct only for
        kt == 1, and silently wrong (output shrinks in time, no causal pad) for
        any larger temporal kernel such as DeepFilterNet3-ll's (2, 3).
    """

    def __init__(self, in_ch, out_ch, kernel=(1, 3), stride=(1, 1), padding=None,
                 separable=True, act_layer=partial(nn.ReLU, inplace=True)):
        super().__init__()
        kt, kf = kernel
        fpad = kf // 2 if padding is None else padding[1]
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1 or max(kernel) == 1:
            separable = False

        self.pad = nn.ConstantPad2d((0, 0, kt - 1, 0), 0.0) if kt > 1 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, (0, fpad),
                              groups=groups, bias=False)
        self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False) if separable else nn.Identity()
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.conv(self.pad(x)))))


class SeparableConvTranspose2d(nn.Module):
    """Separable ConvTranspose2d + BN + ReLU, causal on the time axis.

    Matches upstream ``ConvTranspose2dNormAct``: the *transpose* runs first
    (with ``groups = gcd``), then the 1x1 pointwise.  The previous version had
    the order reversed (pointwise first, then a depthwise transpose).
    """

    def __init__(self, in_ch, out_ch, kernel=(1, 3), stride=(1, 1), padding=None,
                 output_padding=None, act=True, separable=True):
        super().__init__()
        kt, kf = kernel
        fpad = kf // 2 if padding is None else padding[1]
        # Upstream ConvTranspose2dNormAct hardcodes output_padding=(0, kf//2);
        # with padding=(kt-1, kf//2) that makes F_out exactly stride*F_in, so
        # the shape-matching heuristic the decoder used to carry is unnecessary.
        if output_padding is None:
            output_padding = (0, fpad)
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False

        self.pad = nn.ConstantPad2d((0, 0, kt - 1, 0), 0.0) if kt > 1 else nn.Identity()
        self.convt = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride,
                                        (kt - 1, fpad),
                                        output_padding=output_padding,
                                        groups=groups, bias=False)
        self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False) if separable else nn.Identity()
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.convt(self.pad(x)))))


class LookaheadConv2d(nn.Sequential):
    """Conv2d with an explicit, streaming-compatible temporal lookahead.

    For a temporal kernel of size ``K`` and lookahead ``L``, output frame ``t``
    consumes input frames ``[t-(K-L-1), ..., t+L]``.  Frequency padding remains
    symmetric.  This makes the frame alignment explicit instead of relying on
    Conv2d's symmetric time padding.

    Equivalent to upstream's scheme (strictly causal convs plus a whole-stream
    feature shift ``ConstantPad2d((0, 0, -L, L))``): at L = 1 both give the
    receptive field ``[t-1, t, t+1]``.  Doing it inside the conv keeps the
    alignment explicit for the streaming port.

    Separability follows upstream: ``groups = gcd(in_ch, out_ch)``, so the ERB
    input conv (1 -> C, gcd = 1) is dense while the DF input conv (2 -> C,
    gcd = 2) is grouped + pointwise.
    """

    def __init__(self, in_ch, out_ch, kernel=(3, 3), lookahead=0, separable=True):
        kt, kf = kernel
        if not 0 <= lookahead < kt:
            raise ValueError(
                f"lookahead must be in [0, {kt - 1}] for kernel={kernel}, "
                f"got {lookahead}"
            )
        time_left = kt - lookahead - 1
        freq_pad = kf // 2
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1 or max(kernel) == 1:
            separable = False

        layers = [
            nn.ConstantPad2d((freq_pad, freq_pad, time_left, lookahead), 0.0),
            nn.Conv2d(in_ch, out_ch, kernel, padding=0, groups=groups, bias=False),
        ]
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, 1, bias=False))
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    """Grouped linear projection, no bias — upstream df/modules.py.

    Weight is (G, I/G, H/G) and the input is reshaped to (B, T, G, I/G) before
    an einsum, so each group only sees its own slice of the feature axis.  With
    G groups this is G x fewer parameters than a dense Linear of the same
    shape; upstream uses G=16 everywhere except the encoder's DF projection,
    which uses G=32.

    Note there is deliberately NO bias (upstream registers only 'weight'), and
    kaiming init on the 3-D weight makes torch use fan_in = (I/G)*(H/G).
    """

    def __init__(self, input_size, hidden_size, groups=1):
        super().__init__()
        assert input_size % groups == 0, f"{input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"{hidden_size} not divisible by {groups}"
        self.input_size, self.hidden_size, self.groups = input_size, hidden_size, groups
        self.ws = input_size // groups
        self.weight = nn.Parameter(
            torch.zeros(groups, input_size // groups, hidden_size // groups)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        b, t, _ = x.shape
        x = x.view(b, t, self.groups, self.ws)
        x = torch.einsum("btgi,gih->btgh", x, self.weight)
        return x.flatten(2, 3)

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, groups={self.groups})")


class SqueezedGRU_S(nn.Module):
    """Grouped-linear squeeze -> GRU -> optional grouped-linear expand.

    Upstream df/modules.py.  The GRU itself is always hidden->hidden; the
    512<->256 width change is done by the bias-free GroupedLinearEinsum pair,
    which is why upstream's GRUs are all 256->256 while its embedding bus is
    512 wide.  ``_S`` places the optional skip on the raw input AFTER
    linear_out (DFN2's plain SqueezedGRU added it before).
    """

    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1,
                 linear_groups=8, batch_first=True, gru_skip_op=None,
                 linear_act_layer=nn.Identity):
        super().__init__()
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer(),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, x, h=None):
        x = self.linear_in(x)
        y, h = self.gru(x, h)
        y = self.linear_out(y)
        if self.gru_skip is not None:
            y = y + self.gru_skip(x)
        return y, h


# ============================================================
# Encoder
# ============================================================

class DFN2Encoder(nn.Module):
    def __init__(self, n_erb, df_bins, enc_ch=64, emb_size=256,
                 conv_kernel=(1, 3), conv_kernel_inp=(3, 3),
                 mask_lookahead=1, lin_groups=16, enc_lin_groups=32,
                 enc_concat=False):
        super().__init__()
        # ERB path.  Only this input convolution uses temporal context.
        self.erb_conv0 = LookaheadConv2d(
            1, enc_ch, conv_kernel_inp, lookahead=mask_lookahead,
        )
        self.erb_conv1 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2))
        self.erb_conv2 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2))
        self.erb_conv3 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 1))

        # DF path
        self.df_conv0 = LookaheadConv2d(
            2, enc_ch, conv_kernel_inp, lookahead=mask_lookahead,
        )
        self.df_conv1 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2))

        # Embedding bus is enc_ch * n_erb//4 (n_erb//4 because erb_conv1 and
        # erb_conv2 each halve the frequency axis).  At the upstream conv_ch=64
        # this is 512, NOT emb_size.
        self.emb_in_dim = enc_ch * (n_erb // 4)
        df_feat_dim = enc_ch * (df_bins // 2)

        # Upstream has NO projection on the ERB branch: e3 is flattened and fed
        # straight into `combine`.  Only the DF branch is projected, and with
        # enc_lin_groups (32) rather than lin_groups.
        self.df_fc_emb = nn.Sequential(
            GroupedLinearEinsum(df_feat_dim, self.emb_in_dim, groups=enc_lin_groups),
            nn.ReLU(inplace=True),
        )
        # enc_concat=False upstream -> the two branches are ADDED, so the bus
        # stays emb_in_dim wide and there is no post-combine projection either.
        self.enc_concat = enc_concat
        if enc_concat:
            self.emb_in_dim *= 2

        # One GRU layer here; the remaining emb_num_layers-1 live in the ERB
        # decoder.  Upstream hardcodes num_layers=1 at this site even though
        # emb_num_layers defaults to 2 -- reading that config value as the
        # encoder's depth is the trap this port previously fell into.
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim, emb_size, output_size=self.emb_in_dim,
            num_layers=1, linear_groups=lin_groups, gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        # No lsnr head: upstream trains one with [localsnrloss] factor = 1e-3,
        # but this port deliberately omits that loss term, so the head would be
        # untrained dead weight.  (Verified upstream: enc.lsnr_fc weights span
        # -3.72..+1.35, far outside their +-0.0442 init bound, i.e. trained.)

    def forward(self, feat_erb, feat_spec):
        """
        feat_erb  : (B, 1, T, n_erb)
        feat_spec : (B, 2, T, df_bins)
        """
        # ERB path
        e0 = self.erb_conv0(feat_erb)    # (B, enc_ch, T, n_erb)
        e1 = self.erb_conv1(e0)          # (B, enc_ch, T, n_erb//2)
        e2 = self.erb_conv2(e1)          # (B, enc_ch, T, n_erb//4)
        e3 = self.erb_conv3(e2)          # (B, enc_ch, T, n_erb//4)

        # DF path
        c0 = self.df_conv0(feat_spec)    # (B, enc_ch, T, df_bins)
        c1 = self.df_conv1(c0)           # (B, enc_ch, T, df_bins//2)

        # Flatten.  The ERB branch goes into `combine` UNPROJECTED (upstream
        # has no counterpart to the old erb_fc); only the DF branch is
        # projected, down to the same width.
        # Frequency-major: (B, C, T, F) -> (B, T, F, C) -> flatten.  Upstream
        # does exactly this (`e3.permute(0, 2, 3, 1).flatten(2)`), and the ERB
        # decoder's `emb.view(b, t, f8, -1)` only reconstitutes the spatial map
        # correctly under this layout.  Channel-major flattening has the same
        # parameter count but hands GroupedLinearEinsum a different partition:
        # F-major makes each group a contiguous frequency span, C-major makes
        # it a set of channels spanning all frequencies.
        B, _, T, _ = e3.shape
        e3_flat = e3.permute(0, 2, 3, 1).reshape(B, T, -1)   # (B, T, n_erb//4 * enc_ch)
        c1_flat = c1.permute(0, 2, 3, 1).reshape(B, T, -1)   # (B, T, df_bins//2 * enc_ch)

        df_emb = self.df_fc_emb(c1_flat)                     # (B, T, emb_in_dim)
        if self.enc_concat:
            emb = torch.cat([e3_flat, df_emb], dim=-1)
        else:
            emb = e3_flat + df_emb                           # upstream default

        emb, _ = self.emb_gru(emb)                           # (B, T, emb_in_dim)

        return e0, e1, e2, e3, emb, c0


# ============================================================
# ERB Decoder
# ============================================================

class ERBDecoder(nn.Module):
    """4-level U-Net decoder: emb -> (B, 1, T, n_erb) sigmoid mask.

    Encoder features enter through 1x1 *pathway* convolutions and are ADDED to
    the running decoder state, which is upstream's structure (``conv3p`` ..
    ``conv0p`` in ErbDecoder).  This port previously concatenated the skips and
    fed 2*enc_ch into each transposed conv.  Concat and pathway+add can be made
    to agree on parameter count, so nothing about the totals reveals which one
    is in use -- but they are different functions.

    Note ``convt3`` and ``conv0_out`` are ordinary convolutions, not transposed
    ones: only the two stride-2 stages upsample the frequency axis.
    """

    def __init__(self, n_erb, enc_ch=64, emb_size=256, conv_kernel=(1, 3),
                 emb_num_layers=3, lin_groups=16, convt_kernel=None):
        super().__init__()
        emb_dim = enc_ch * (n_erb // 4)

        # THE decoder recurrence.  Upstream splits emb_num_layers between the
        # encoder (a hardcoded 1) and here (emb_num_layers - 1); this port
        # previously put both layers in the encoder and left the ERB decoder
        # with no recurrence at all.
        self.emb_gru = SqueezedGRU_S(
            emb_dim, emb_size, output_size=emb_dim,
            num_layers=max(1, emb_num_layers - 1), linear_groups=lin_groups,
            gru_skip_op=None, linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        # DFN3 uses a separate convt_kernel; DFN2 reused conv_kernel.
        convt_kernel = convt_kernel or conv_kernel

        # Pathway convs are 1x1, so `max(kernel) == 1` disables the pointwise
        # stage while groups stays gcd(C, C) = C: a depthwise per-channel
        # scale, exactly as upstream builds it.
        self.conv3p = SeparableConv2d(enc_ch, enc_ch, (1, 1))
        self.convt3 = SeparableConv2d(enc_ch, enc_ch, conv_kernel)
        self.conv2p = SeparableConv2d(enc_ch, enc_ch, (1, 1))
        self.convt2 = SeparableConvTranspose2d(enc_ch, enc_ch, convt_kernel,
                                               stride=(1, 2))
        self.conv1p = SeparableConv2d(enc_ch, enc_ch, (1, 1))
        self.convt1 = SeparableConvTranspose2d(enc_ch, enc_ch, convt_kernel,
                                               stride=(1, 2))
        self.conv0p = SeparableConv2d(enc_ch, enc_ch, (1, 1))
        self.conv0_out = SeparableConv2d(enc_ch, 1, conv_kernel,
                                         act_layer=nn.Sigmoid)

        self.enc_ch = enc_ch
        self.n_erb_4 = n_erb // 4

    def forward(self, emb, e3, e2, e1, e0):
        """emb: (B, T, enc_ch*n_erb//4); e0..e3: (B, enc_ch, T, *)"""
        B, T, _ = emb.shape
        x, _ = self.emb_gru(emb)                       # (B, T, n_erb//4 * enc_ch)
        # Frequency-major, matching the encoder's flatten and upstream's
        # `emb.view(b, t, f8, -1).permute(0, 3, 1, 2)`.
        x = x.reshape(B, T, self.n_erb_4, self.enc_ch)
        x = x.permute(0, 3, 1, 2)                      # (B, enc_ch, T, n_erb//4)

        x = self.convt3(self.conv3p(e3) + x)           # (B, enc_ch, T, n_erb//4)
        x = self.convt2(self.conv2p(e2) + x)           # (B, enc_ch, T, n_erb//2)
        x = self.convt1(self.conv1p(e1) + x)           # (B, enc_ch, T, n_erb)
        return self.conv0_out(self.conv0p(e0) + x)     # (B, 1, T, n_erb)


# ============================================================
# DF Decoder
# ============================================================

class DFDecoder(nn.Module):
    """
    Per-frame, per-bin FIR filter coefficients.  DFN2's alpha blend weight is
    gone: DFN3 splits the spectrum by band instead of blending, so there is
    nothing for an alpha to weigh.
    """
    def __init__(self, df_bins, df_order, df_hidden=256, emb_in_dim=512,
                 enc_ch=64, conv_kernel=(1, 3), num_layers=2, lin_groups=16,
                 df_gru_skip='groupedlinear', pathway_kernel_size_t=5):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        df_out_ch = df_order * 2   # real + imag

        # ⚠ linear_groups is NOT passed here: upstream relies on
        # SqueezedGRU_S's own default of 8 at this one site, while the encoder
        # and ERB decoder use 16.  Checkpoint-confirmed
        # (df_dec.df_gru.linear_in.0.weight has shape (8, 64, 32)).
        # Do NOT "fix" this to lin_groups.
        self.df_gru = SqueezedGRU_S(
            emb_in_dim, df_hidden, num_layers=num_layers, gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )

        # Residual path from the embedding into the DF decoder.
        assert df_gru_skip in ('none', 'identity', 'groupedlinear')
        if df_gru_skip == 'none':
            self.df_skip = None
        elif df_gru_skip == 'identity':
            assert emb_in_dim == df_hidden, "identity skip needs matching dims"
            self.df_skip = nn.Identity()
        else:
            self.df_skip = GroupedLinearEinsum(emb_in_dim, df_hidden, groups=lin_groups)

        # Pathway conv over the DF encoder output, kernel (kt, 1) with causal pad.
        kt = pathway_kernel_size_t
        self.df_convp = SeparableConv2d(enc_ch, df_out_ch, (kt, 1))
        self.df_out = nn.Sequential(
            GroupedLinearEinsum(df_hidden, df_bins * df_out_ch, groups=lin_groups),
            nn.Tanh(),
        )
        # Not used by forward -- but released DeepFilterNet3 still carries it,
        # and its 257 parameters are inside the 2,135,484 this port reconciles
        # against (ours = 2,135,484 - 513 for the removed lsnr head).  Deleting
        # it would silently break that reconciliation, which is the only check
        # that the architecture realignment is complete.
        self.df_fc_a = nn.Sequential(nn.Linear(df_hidden, 1), nn.Sigmoid())

    def forward(self, emb, c0):
        """
        emb : (B, T, emb_in_dim)
        c0  : (B, enc_ch, T, df_bins)
        Returns:
            coefs : (B, T, df_bins, df_order*2)
        """
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)                             # (B, T, df_hidden)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        # c0 residual: conv → (B, df_out_ch, T, df_bins) → permute → (B, T, df_bins, df_out_ch)
        c0_res = self.df_convp(c0).permute(0, 2, 3, 1)     # (B, T, df_bins, df_order*2)

        c_out = self.df_out(c)                               # (B, T, df_bins * df_order*2)
        coefs = c_out.view(b, t, self.df_bins, self.df_order * 2) + c0_res
        return coefs


# ============================================================
# Deep Filter Apply
# ============================================================

def deep_filter_apply(spec, coefs, df_bins, df_order, df_lookahead=0):
    """
    Apply a per-bin FIR filter to the lowest df_bins of spec.

    The deployed configuration uses ``df_lookahead=0``: order 5 therefore
    consumes masked spectra ``[t-4, ..., t]`` and only requires four history
    frames in a streaming ring buffer.  Mask lookahead is an independent model
    delay and must not be folded into this FIR window.

    Args:
        spec   : (B, n_bins, T) complex
        coefs  : (B, T, df_bins, df_order*2)
        df_bins: int
        df_order: int
        df_lookahead: number of future masked-spectrum frames used by the FIR
    Returns:
        out    : (B, n_bins, T) complex (only first df_bins modified)
    """
    if not 0 <= df_lookahead < df_order:
        raise ValueError(
            f"df_lookahead must be in [0, {df_order - 1}], got {df_lookahead}"
        )

    # Reshape coefs: (B, T, df_bins, df_order, 2)
    coefs = coefs.view(coefs.shape[0], coefs.shape[1], df_bins, df_order, 2)

    # Extract real/imag of low-freq bins and prepare for unfold
    spec_ri = torch.view_as_real(spec[:, :df_bins])   # (B, df_bins, T, 2)
    spec_ri = spec_ri.permute(0, 1, 3, 2)             # (B, df_bins, 2, T)

    history = df_order - df_lookahead - 1
    spec_padded = F.pad(spec_ri, (history, df_lookahead))

    # Sliding window: (B, df_bins, 2, T, df_order)
    spec_unfolded = spec_padded.unfold(-1, df_order, 1)

    # Permute coefs to match: (B, df_bins, 2, T, df_order)
    coefs_p = coefs.permute(0, 2, 4, 1, 3)

    # Complex multiply-accumulate: (re*re - im*im, im*re + re*im)
    df_re = (spec_unfolded[:, :, 0] * coefs_p[:, :, 0]
             - spec_unfolded[:, :, 1] * coefs_p[:, :, 1]).sum(-1)  # (B, df_bins, T)
    df_im = (spec_unfolded[:, :, 1] * coefs_p[:, :, 0]
             + spec_unfolded[:, :, 0] * coefs_p[:, :, 1]).sum(-1)

    df_out = torch.view_as_complex(torch.stack([df_re, df_im], dim=-1))  # (B, df_bins, T)

    out = spec.clone()
    out[:, :df_bins] = df_out
    return out


# ============================================================
# Main model
# ============================================================

class DeepFilterNet2(nn.Module):
    """
    DeepFilterNet2 with config-driven sample rate and FFT size.

    Inputs (from extract_dfn2_features):
        spec      : (B, n_bins, T) complex
        feat_erb  : (B, 1, T, n_erb)
        feat_spec : (B, 2, T, df_bins)

    Returns:
        enhanced_spec : (B, n_bins, T) complex
        erb_mask      : (B, 1, T, n_erb)
    """
    def __init__(self, n_fft=512, sr=16000, n_erb=32, df_bins=64, df_order=5,
                 enc_ch=64, emb_size=256, df_hidden=256, df_num_layers=2,
                 mask_lookahead=1, df_lookahead=0,
                 emb_num_layers=3, lin_groups=16, enc_lin_groups=32,
                 enc_concat=False, df_gru_skip='groupedlinear',
                 df_pathway_kernel_size_t=5, conv_kernel=(1, 3),
                 conv_kernel_inp=(3, 3), convt_kernel=(1, 3)):
        super().__init__()
        n_bins = n_fft // 2 + 1

        self.n_erb     = n_erb
        self.df_bins   = df_bins
        self.df_order  = df_order
        self.mask_lookahead = mask_lookahead
        self.df_lookahead = df_lookahead
        self.n_bins    = n_bins

        if not 0 <= mask_lookahead <= 2:
            raise ValueError(
                f"mask_lookahead must be in [0, 2], got {mask_lookahead}"
            )
        if not 0 <= df_lookahead < df_order:
            raise ValueError(
                f"df_lookahead must be in [0, {df_order - 1}], "
                f"got {df_lookahead}"
            )

        # ERB filterbank — non-trainable buffers
        erb_fb, erb_inv = _build_erb_fb(n_fft, sr, n_erb)
        self.register_buffer('erb_fb',  erb_fb)   # (n_erb, n_bins)
        self.register_buffer('erb_inv', erb_inv)  # (n_erb, n_bins) indicator

        self.encoder = DFN2Encoder(
            n_erb, df_bins, enc_ch, emb_size,
            conv_kernel=conv_kernel, conv_kernel_inp=conv_kernel_inp,
            mask_lookahead=mask_lookahead, lin_groups=lin_groups,
            enc_lin_groups=enc_lin_groups, enc_concat=enc_concat,
        )
        # The embedding bus is enc_ch * n_erb//4 wide (512 at the upstream
        # conv_ch=64), NOT emb_size -- emb_size is only the GRU hidden width.
        emb_in_dim = self.encoder.emb_in_dim
        self.erb_dec = ERBDecoder(n_erb, enc_ch, emb_size, conv_kernel,
                                  emb_num_layers=emb_num_layers,
                                  lin_groups=lin_groups,
                                  convt_kernel=convt_kernel)
        self.df_dec  = DFDecoder(df_bins, df_order, df_hidden, emb_in_dim,
                                 enc_ch, conv_kernel,
                                 num_layers=df_num_layers,
                                 lin_groups=lin_groups,
                                 df_gru_skip=df_gru_skip,
                                 pathway_kernel_size_t=df_pathway_kernel_size_t)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DeepFilterNet2: {n_params:,} trainable parameters")

    def forward(self, spec, feat_erb, feat_spec):
        """
        spec      : (B, n_bins, T) complex
        feat_erb  : (B, 1, T, n_erb)
        feat_spec : (B, 2, T, df_bins)
        """
        e0, e1, e2, e3, emb, c0 = self.encoder(feat_erb, feat_spec)

        # ERB mask → expand to per-bin.  Only the bins at or above df_bins
        # survive the band split below, so the mask is expanded over that slice
        # alone; the lower bins would be computed and then overwritten.
        erb_mask = self.erb_dec(emb, e3, e2, e1, e0)   # (B, 1, T, n_erb)
        bin_mask = erb_mask.squeeze(1).matmul(self.erb_inv[:, self.df_bins:])
        bin_mask = bin_mask.permute(0, 2, 1)           # (B, n_bins-df_bins, T) — NOT .T
        spec_m = spec[:, self.df_bins:] * bin_mask

        # DFN3 composes the two stages as a PARALLEL BAND SPLIT, not a cascade:
        # the deep filter runs on the *unmasked* spectrum and owns the lowest
        # df_bins outright, while everything above comes from the ERB mask.
        # DFN2 instead ran DF on the already-masked spectrum and blended with a
        # learned alpha.  This costs zero parameters, so no parameter count can
        # tell you which one you have -- and it is the most behaviourally
        # significant DFN2->DFN3 change.
        coefs = self.df_dec(emb, c0)
        spec_e = deep_filter_apply(
            spec, coefs, self.df_bins, self.df_order, self.df_lookahead,
        )
        spec_e[:, self.df_bins:] = spec_m

        return spec_e, erb_mask
