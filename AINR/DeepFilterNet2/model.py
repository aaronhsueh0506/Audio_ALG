"""DeepFilterNet2 with config-driven sample rate and FFT size.

Architecture:
  - Encoder: ERB path (4 levels, freq stride-2 x2) + DF path (2 levels)
             -> joint embedding GRU
  - ERBDecoder: U-Net transposed conv x 4 -> sigmoid ERB mask
  - DFDecoder: GRU -> per-bin FIR coefficients
  - deep_filter_apply: causal real/imag FIR filtering on low-frequency bins

Convention: (B, C, T, F) throughout (time = dim 2, freq = dim 3).
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- ERB filterbank helpers ----

def freq2erb(freq_hz):
    """Hz -> ERB-number (Glasberg-Moore; 9.265/24.7 constants as in libDF freq2erb
    and RNNoise-ERB train.py)."""
    return 9.265 * np.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """ERB-number -> Hz (inverse of freq2erb)."""
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft, min_bins_per_band=2):
    """ERB band borders as FFT-bin indices on the Glasberg-Moore scale (after
    bandERB.ipynb ERBBand()).

    Returns a length-N (N = n_bands) int array with nfftborder[0] = 0 (DC) and
    nfftborder[-1] = n_fft//2 + 1; N borders give N ERB bands.
        cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N))
        ideal border = round((cutoff + bw/2) / bw), bw = sr/n_fft
    Band width is enforced greedily forward, so every consecutive border pair is
    at least min_bins_per_band bins apart (as in RNNoise-ERB train.py).
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
    """Triangular ERB filterbank (after bandERB.ipynb ERB_pro_matrix), shape
    (n_bins, N) with N = len(nfftborder) bands.

    The range(N-1) blocks lie between consecutive borders, so column 0 and column
    N-1 are one-sided and interior columns are full triangles.
        mode=0 (forward / feature): x2 on the two edge columns so their band
               energy is comparable to interior bands.
        mode=1 (inverse / mask expansion): no x2; a partition of unity, so gain=1
               maps to bin_gain=1 without row normalisation.
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
    """Build ERB band-assignment matrices for the configured sample rate and FFT.

    Returns:
        erb_fb   : (n_erb, n_bins) forward transform (mode=0, edge columns
                   doubled; not normalised to sum 1)
        erb_inv  : (n_erb, n_bins) inverse transform (mode=1, partition of unity)
    """
    borders = erb_bandborder(n_erb, sr, n_fft)
    fb = compute_erb_matrix(borders, n_fft, mode=0).T.copy()   # (n_erb, n_bins)
    inv = compute_erb_matrix(borders, n_fft, mode=1).T.copy()  # (n_erb, n_bins)
    return torch.from_numpy(fb), torch.from_numpy(inv)


# ---- Building blocks ----

class SeparableConv2d(nn.Module):
    """Separable Conv2d + BN + ReLU, causal on the time axis.

    Matches upstream ``Conv2dNormAct``: grouping is ``gcd(in_ch, out_ch)`` and the
    first conv already changes the channel count, followed by a 1x1 pointwise (the
    usual depthwise+pointwise pair only when in_ch == out_ch); activation is ReLU;
    the time axis is padded ``kernel[0] - 1`` on the left only, so the conv is
    strictly causal for any temporal kernel.
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
    (with ``groups = gcd``), then the 1x1 pointwise.
    """

    def __init__(self, in_ch, out_ch, kernel=(1, 3), stride=(1, 1), padding=None,
                 output_padding=None, act=True, separable=True):
        super().__init__()
        kt, kf = kernel
        fpad = kf // 2 if padding is None else padding[1]
        # Upstream hardcodes output_padding=(0, kf//2); with padding=(kt-1, kf//2)
        # this makes F_out exactly stride*F_in.
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
    consumes input frames ``[t-(K-L-1), ..., t+L]``; frequency padding stays
    symmetric. This equals upstream's strictly causal convs plus a whole-stream
    feature shift (at L = 1 both give ``[t-1, t, t+1]``) but keeps the alignment
    explicit for the streaming port.

    Separability follows upstream: ``groups = gcd(in_ch, out_ch)``, so the ERB
    input conv (1 -> C) is dense while the DF input conv (2 -> C) is grouped +
    pointwise.
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
    """Grouped linear projection, no bias (upstream df/modules.py).

    Weight is (G, I/G, H/G) and the input is reshaped to (B, T, G, I/G) before an
    einsum, so each group sees only its slice of the feature axis (G x fewer
    parameters than a dense Linear). There is deliberately no bias (upstream
    registers only 'weight').

    Three group counts are in play: upstream's shipped linear_groups=16 and
    enc_linear_groups=32, while df_dec.df_gru never receives linear_groups and
    uses SqueezedGRU_S's module default of 8 (the released checkpoint has
    df_dec.df_gru.linear_in.0.weight of shape (8, 64, 32)). Changing df_gru to 16
    breaks checkpoint compatibility.
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
        # Grouped matmul as a broadcast matmul instead of einsum: the ONNX Einsum
        # op is unsupported by many accelerator toolchains, and this form exports
        # as a plain MatMul with the same contraction order.
        x = (x.unsqueeze(-2) @ self.weight).squeeze(-2)
        return x.flatten(2, 3)

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, groups={self.groups})")


class SqueezedGRU_S(nn.Module):
    """Grouped-linear squeeze -> GRU -> optional grouped-linear expand (upstream
    df/modules.py).

    The GRU is always hidden->hidden; the 512<->256 width change is done by the
    bias-free GroupedLinearEinsum pair. ``_S`` places the optional skip on the raw
    input after linear_out.
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
        # The skip is fed the raw input, not linear_in's output (as upstream). It
        # is dormant here: every call site passes gru_skip_op=None.
        skip_in = x
        x = self.linear_in(x)
        y, h = self.gru(x, h)
        y = self.linear_out(y)
        if self.gru_skip is not None:
            y = y + self.gru_skip(skip_in)
        return y, h


# ---- Encoder ----

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

        # Embedding bus is enc_ch * n_erb//4 (erb_conv1 and erb_conv2 each halve
        # the frequency axis): 512 at conv_ch=64, not emb_size.
        self.emb_dim = enc_ch * (n_erb // 4)          # the bus, in and out
        df_feat_dim = enc_ch * (df_bins // 2)

        # As upstream, the ERB branch is flattened straight into `combine`; only
        # the DF branch is projected, with enc_lin_groups (32).
        self.df_fc_emb = nn.Sequential(
            GroupedLinearEinsum(df_feat_dim, self.emb_dim, groups=enc_lin_groups),
            nn.ReLU(inplace=True),
        )
        # enc_concat=False (upstream): the branches are added, so the bus stays
        # one width.
        self.enc_concat = enc_concat
        gru_in_dim = self.emb_dim * 2 if enc_concat else self.emb_dim

        # One GRU layer here; the other emb_num_layers-1 live in the ERB decoder
        # (upstream hardcodes 1 at this site).
        self.emb_gru = SqueezedGRU_S(
            gru_in_dim, emb_size, output_size=self.emb_dim,
            num_layers=1, linear_groups=lin_groups, gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        # No lsnr head: its loss term is not trained here. Upstream's runtime uses
        # lsnr to skip the ERB/DF stages on clean or lightly noisy frames, so a
        # streaming port of this model runs both stages on every frame.

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

        # Flatten frequency-major, (B, C, T, F) -> (B, T, F, C) -> flatten, as
        # upstream: the ERB decoder's emb.view(b, t, f8, -1) only reconstitutes the
        # spatial map under this layout, and each GroupedLinear group is then a
        # contiguous frequency span. The ERB branch enters `combine` unprojected;
        # only the DF branch is projected.
        B, _, T, _ = e3.shape
        e3_flat = e3.permute(0, 2, 3, 1).reshape(B, T, -1)   # (B, T, n_erb//4 * enc_ch)
        c1_flat = c1.permute(0, 2, 3, 1).reshape(B, T, -1)   # (B, T, df_bins//2 * enc_ch)

        df_emb = self.df_fc_emb(c1_flat)                     # (B, T, emb_dim)
        if self.enc_concat:
            emb = torch.cat([e3_flat, df_emb], dim=-1)
        else:
            emb = e3_flat + df_emb                           # upstream default

        emb, _ = self.emb_gru(emb)                           # (B, T, emb_dim)

        return e0, e1, e2, e3, emb, c0


# ---- ERB Decoder ----

class ERBDecoder(nn.Module):
    """4-level U-Net decoder: emb -> (B, 1, T, n_erb) sigmoid mask.

    Encoder features enter through 1x1 pathway convolutions and are added to the
    running decoder state (upstream conv3p .. conv0p); concatenating the skips
    would be a different function. ``convt3`` and ``conv0_out`` are ordinary
    convolutions: only the two stride-2 stages upsample the frequency axis.
    """

    def __init__(self, n_erb, enc_ch=64, emb_size=256, conv_kernel=(1, 3),
                 emb_num_layers=3, lin_groups=16, convt_kernel=None):
        super().__init__()
        emb_dim = enc_ch * (n_erb // 4)

        # Decoder recurrence: upstream splits emb_num_layers between the encoder
        # (1) and here (emb_num_layers - 1). There is no max(1, ...) clamp: a
        # config that cannot be honoured must fail (nn.GRU rejects num_layers=0)
        # instead of silently changing the total depth.
        self.emb_gru = SqueezedGRU_S(
            emb_dim, emb_size, output_size=emb_dim,
            num_layers=emb_num_layers - 1, linear_groups=lin_groups,
            gru_skip_op=None, linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        # DFN3 uses a separate convt_kernel; DFN2 reused conv_kernel.
        convt_kernel = convt_kernel or conv_kernel

        # Pathway convs are 1x1, so max(kernel) == 1 disables the pointwise stage
        # while groups stays gcd(C, C) = C: a depthwise per-channel scale, as
        # upstream.
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
        # emb.view(b, t, f8, -1).permute(0, 3, 1, 2).
        x = x.reshape(B, T, self.n_erb_4, self.enc_ch)
        x = x.permute(0, 3, 1, 2)                      # (B, enc_ch, T, n_erb//4)

        x = self.convt3(self.conv3p(e3) + x)           # (B, enc_ch, T, n_erb//4)
        x = self.convt2(self.conv2p(e2) + x)           # (B, enc_ch, T, n_erb//2)
        x = self.convt1(self.conv1p(e1) + x)           # (B, enc_ch, T, n_erb)
        return self.conv0_out(self.conv0p(e0) + x)     # (B, 1, T, n_erb)


# ---- DF Decoder ----

class DFDecoder(nn.Module):
    """
    Per-frame, per-bin FIR coefficients plus DFN2's learned residual blend.

    Returns coefs: (B, T, df_bins, df_order*2) and alpha: (B, T, 1).
    """
    def __init__(self, df_bins, df_order, df_hidden=256, emb_in_dim=512,
                 enc_ch=64, conv_kernel=(1, 3), num_layers=2, lin_groups=16,
                 df_gru_skip='groupedlinear', pathway_kernel_size_t=5):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        df_out_ch = df_order * 2   # real + imag

        # linear_groups is not passed: upstream relies on SqueezedGRU_S's default
        # of 8 at this site (checkpoint df_dec.df_gru.linear_in.0.weight is
        # (8, 64, 32)). Do not change it to lin_groups.
        self.df_gru = SqueezedGRU_S(
            emb_in_dim, df_hidden, num_layers=num_layers, gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )

        # Residual path from the embedding into the DF decoder.
        assert df_gru_skip in ('none', 'identity', 'groupedlinear')
        if df_gru_skip == 'none':
            self.df_skip = None
        elif df_gru_skip == 'identity':
            # Intentionally not upstream's guard (emb_hidden_dim == df_hidden_dim,
            # which passes and then fails to broadcast): the tensor added here is
            # the emb_in_dim-wide bus, so emb_in_dim == df_hidden is the condition
            # that matters.
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
        # DFN2 uses this head to decide how much of the deep-filter residual to
        # trust per frame. DFN3 keeps the parameters but ignores the output (hard
        # low/high band split).
        self.df_fc_a = nn.Sequential(nn.Linear(df_hidden, 1), nn.Sigmoid())
        # Released DFN3's convt_depthwise key is never read upstream (separable
        # is hardcoded and the transposed convs are depthwise), so this port has
        # no such key.

    def forward(self, emb, c0):
        """
        emb : (B, T, emb_in_dim)
        c0  : (B, enc_ch, T, df_bins)
        Returns:
            coefs : (B, T, df_bins, df_order*2)
            alpha : (B, T, 1)
        """
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)                             # (B, T, df_hidden)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        # c0 residual: conv → (B, df_out_ch, T, df_bins) → permute → (B, T, df_bins, df_out_ch)
        c0_res = self.df_convp(c0).permute(0, 2, 3, 1)     # (B, T, df_bins, df_order*2)

        alpha = self.df_fc_a(c)                              # (B, T, 1)
        c_out = self.df_out(c)                               # (B, T, df_bins * df_order*2)
        coefs = c_out.view(b, t, self.df_bins, self.df_order * 2) + c0_res
        return coefs, alpha


# ---- Deep Filter Apply ----

def deep_filter_apply(spec, coefs, df_bins, df_order, df_lookahead):
    """Apply a per-bin FIR filter to the lowest df_bins of spec.

    The caller picks the spectrum this parameter-free FIR sees: DFN2 passes the
    ERB-masked spectrum (cascade), DFN3 the raw spectrum (parallel band split).
    ``df_lookahead`` has no default on purpose: it is a latency decision that must
    agree with ``[model] df_lookahead`` in config.ini.

    Streaming ring at the shipped ``df_order=5, df_lookahead=1``: history = 3,
    window ``[t-3, ..., t+1]``, one frame of delay for this FIR. In DFN2's cascade
    the future source is already ERB-masked, so mask lookahead adds a second
    serial dependency (shipped 1/1 = two hops at the accelerator boundary); DFN3
    reads raw future spectra and stays at max(1,1).

    Args:
        spec   : (B, n_bins, T) complex
        coefs  : (B, T, df_bins, df_order*2)
        df_bins, df_order: int
        df_lookahead: number of future input-spectrum frames used by the FIR
    Returns:
        out    : (B, n_bins, T) complex (only the first df_bins modified)
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


# ---- Main model ----

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
                 mask_lookahead=1, df_lookahead=1,
                 # 1, not 0: read_model_config takes its defaults from this
                 # signature, and an omitted [model] df_lookahead would otherwise
                 # build the 1/0 pairing config.ini documents as buying nothing.

                 emb_num_layers=3, lin_groups=16, enc_lin_groups=32,
                 enc_concat=False, df_gru_skip='groupedlinear',
                 df_pathway_kernel_size_t=5, conv_kernel=(1, 3),
                 conv_kernel_inp=(3, 3), convt_kernel=(1, 3),
                 mask_pf=False, pf_beta=0.02):
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
        # The embedding bus is enc_ch * n_erb//4 wide (512 at conv_ch=64), not
        # emb_size (the GRU hidden width); the DF decoder consumes what the encoder
        # emits and ERBDecoder recomputes the same width.
        emb_out_dim = self.encoder.emb_dim
        self.erb_dec = ERBDecoder(n_erb, enc_ch, emb_size, conv_kernel,
                                  emb_num_layers=emb_num_layers,
                                  lin_groups=lin_groups,
                                  convt_kernel=convt_kernel)
        self.df_dec  = DFDecoder(df_bins, df_order, df_hidden, emb_out_dim,
                                 enc_ch, conv_kernel,
                                 num_layers=df_num_layers,
                                 lin_groups=lin_groups,
                                 df_gru_skip=df_gru_skip,
                                 pathway_kernel_size_t=df_pathway_kernel_size_t)

        # Valin post-filter, DFN3 form (see self.post_filter). Both mask_pf and
        # PF_BETA (upstream default 0.02, exposed as --pf) are config-driven.
        self.mask_pf = mask_pf
        self.pf_beta = pf_beta

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DeepFilterNet2: {n_params:,} trainable parameters")

    def heads(self, feat_erb, feat_spec):
        """The network's predictions only, with no composition.

        Returns:
            erb_mask : (B, 1, T, n_erb)              real, sigmoid, in [0, 1]
            coefs    : (B, T, df_bins, df_order*2)   deep-filter taps, re/im
            alpha    : (B, T, 1)                     learned DF residual blend

        This is the export boundary: everything after it (ERB mask expansion
        through ``erb_inv``, the deep-filter FIR, band split, post-filter, ISTFT)
        is fixed arithmetic with no learned parameters and runs in the C or
        accelerator host. ``forward`` calls this and then composes, so the exported
        half cannot drift from the trained one.
        """
        e0, e1, e2, e3, emb, c0 = self.encoder(feat_erb, feat_spec)
        coefs, alpha = self.df_dec(emb, c0)
        return self.erb_dec(emb, e3, e2, e1, e0), coefs, alpha

    def compose(self, spec, erb_mask, coefs, alpha):
        """Turn the three predicted heads into an enhanced spectrum.

        Parameter-free by construction: this is the arithmetic the C port
        reproduces. Anything learned belongs in ``heads``.
        """
        # DFN2 cascade, stage 1: expand the ERB mask over all bins and mask the
        # full spectrum; the deep filter consumes this masked result.
        bin_mask = erb_mask.squeeze(1).matmul(self.erb_inv)
        bin_mask = bin_mask.permute(0, 2, 1)           # (B, n_bins, T) — NOT .T
        spec_m = spec * bin_mask

        # DFN2 cascade, stage 2: deep-filter the masked low bins, then residual-mix
        # with the masked input by the learned alpha; high bins keep the stage-1
        # output.
        spec_df = deep_filter_apply(
            spec_m, coefs, self.df_bins, self.df_order, self.df_lookahead,
        )
        alpha_t = alpha.permute(0, 2, 1)               # (B, 1, T)
        spec_e = spec_m.clone()
        spec_e[:, :self.df_bins] = (
            alpha_t * spec_df[:, :self.df_bins]
            + (1 - alpha_t) * spec_m[:, :self.df_bins]
        )
        return self.post_filter(spec, spec_e)

    def forward(self, spec, feat_erb, feat_spec):
        """
        spec      : (B, n_bins, T) complex
        feat_erb  : (B, 1, T, n_erb)
        feat_spec : (B, 2, T, df_bins)
        """
        erb_mask, coefs, alpha = self.heads(feat_erb, feat_spec)
        return self.compose(spec, erb_mask, coefs, alpha), erb_mask

    def post_filter(self, spec, spec_e):
        """Valin post-filter, DeepFilterNet3 form (deepfilternet3.py).

        Differs from DFN2's ``Mask.pf``: it works on the final complex spectrum
        (after the ERB mask and the deep filter), derives its mask as
        ``|spec_e| / |spec|`` (the realised gain), has no ``mask`` factor in the
        numerator (a multiplicative correction), does not gate on ``self.training``,
        and takes ``beta`` from config (upstream ``PF_BETA`` default 0.02, ``--pf``).
        """
        if not self.mask_pf:
            return spec_e
        eps = 1e-12
        mask = (spec_e.abs() / spec.abs().add(eps)).clamp(eps, 1)
        mask_sin = mask * torch.sin(math.pi * mask / 2).clamp_min(eps)
        pf = (1 + self.pf_beta) / (
            1 + self.pf_beta * mask.div(mask_sin).pow(2)
        )
        return spec_e * pf
