"""
DeepFilterNet2 adaptation with config-driven sample rate and FFT size.

Architecture overview (aligned with DeepFilterNet2 paper):
  - Encoder: ERB path (4 levels, freq stride-2 ×2) + DF path (2 levels)
             → joint embedding GRU → lsnr
  - ERBDecoder: U-Net transposed conv × 4 → sigmoid ERB mask
  - DFDecoder: GRU → per-bin FIR coefficients + alpha blend
  - deep_filter_apply: causal real/imag FIR filtering on low-freq bins

Convention: (B, C, T, F) throughout (time = H dim 2, freq = W dim 3).
"""

import math
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
    """Depthwise-separable Conv2d with BN + PReLU."""
    def __init__(self, in_ch, out_ch, kernel=(1, 3), stride=(1, 1), padding=(0, 1)):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class SeparableConvTranspose2d(nn.Module):
    """Pointwise → Depthwise-transpose Conv2d with BN + PReLU."""
    def __init__(self, in_ch, out_ch, kernel=(1, 3), stride=(1, 1), padding=(0, 1),
                 output_padding=(0, 0), act=True):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw = nn.ConvTranspose2d(out_ch, out_ch, kernel, stride, padding,
                                     output_padding=output_padding,
                                     groups=out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.dw(self.pw(x))))


class LookaheadConv2d(nn.Sequential):
    """Conv2d with an explicit, streaming-compatible temporal lookahead.

    For a temporal kernel of size ``K`` and lookahead ``L``, output frame ``t``
    consumes input frames ``[t-(K-L-1), ..., t+L]``.  Frequency padding remains
    symmetric.  This makes the frame alignment explicit instead of relying on
    Conv2d's symmetric time padding.
    """

    def __init__(self, in_ch, out_ch, kernel=(3, 3), lookahead=0):
        kt, kf = kernel
        if not 0 <= lookahead < kt:
            raise ValueError(
                f"lookahead must be in [0, {kt - 1}] for kernel={kernel}, "
                f"got {lookahead}"
            )
        time_left = kt - lookahead - 1
        freq_pad = kf // 2
        super().__init__(
            nn.ConstantPad2d((freq_pad, freq_pad, time_left, lookahead), 0.0),
            nn.Conv2d(in_ch, out_ch, kernel, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )


class GroupedGRU(nn.Module):
    """
    Simple GRU wrapper that supports groups=1 (standard) or groups>1 (split-concat).
    Default groups=1 matches DFN2's default gru_groups=1.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, groups=1, batch_first=True):
        super().__init__()
        assert input_size % groups == 0 and hidden_size % groups == 0
        self.groups = groups
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        chunk_in  = input_size  // groups
        chunk_hid = hidden_size // groups
        self.grus = nn.ModuleList([
            nn.GRU(chunk_in, chunk_hid, num_layers, batch_first=batch_first)
            for _ in range(groups)
        ])

    def forward(self, x, h=None):
        """x: (B, T, input_size) → (B, T, hidden_size)"""
        chunks = x.chunk(self.groups, dim=-1)
        if h is None:
            hs = [None] * self.groups
        else:
            hs = h.chunk(self.groups, dim=-1)

        outs, new_hs = [], []
        for i, (gru, xi) in enumerate(zip(self.grus, chunks)):
            hi = hs[i].contiguous() if hs[i] is not None else None
            yi, new_hi = gru(xi, hi)
            outs.append(yi)
            new_hs.append(new_hi)

        return torch.cat(outs, dim=-1), torch.cat(new_hs, dim=-1)


# ============================================================
# Encoder
# ============================================================

class DFN2Encoder(nn.Module):
    def __init__(self, n_erb, df_bins, enc_ch=16, emb_size=256,
                 gru_groups=1, conv_kernel=(1, 3), conv_kernel_inp=(3, 3),
                 mask_lookahead=1):
        super().__init__()
        # ERB path.  Only this input convolution uses temporal context.
        self.erb_conv0 = LookaheadConv2d(
            1, enc_ch, conv_kernel_inp, lookahead=mask_lookahead,
        )
        self.erb_conv1 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2),
                                          padding=(0, conv_kernel[1] // 2))
        self.erb_conv2 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2),
                                          padding=(0, conv_kernel[1] // 2))
        self.erb_conv3 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 1),
                                          padding=(0, conv_kernel[1] // 2))

        # DF path
        self.df_conv0 = LookaheadConv2d(
            2, enc_ch, conv_kernel_inp, lookahead=mask_lookahead,
        )
        self.df_conv1 = SeparableConv2d(enc_ch, enc_ch, conv_kernel, stride=(1, 2),
                                         padding=(0, conv_kernel[1] // 2))

        # Embedding dimensions
        # n_erb//4 because erb_conv1 and erb_conv2 each halve freq
        erb_feat_dim = enc_ch * (n_erb // 4)
        df_feat_dim  = enc_ch * (df_bins // 2)

        self.erb_fc = nn.Linear(erb_feat_dim, emb_size, bias=False)
        self.df_fc  = nn.Linear(df_feat_dim,  emb_size, bias=False)
        self.emb_fc = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size, bias=False), nn.ReLU()
        )
        self.gru_emb = GroupedGRU(emb_size, emb_size, num_layers=2,
                                   groups=gru_groups, batch_first=True)
        self.lsnr_fc = nn.Sequential(nn.Linear(emb_size, 1), nn.Sigmoid())

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

        # Flatten and project to embedding
        B, _, T, _ = e3.shape
        e3_flat = e3.permute(0, 2, 1, 3).reshape(B, T, -1)   # (B, T, enc_ch * n_erb//4)
        c1_flat = c1.permute(0, 2, 1, 3).reshape(B, T, -1)   # (B, T, enc_ch * df_bins//2)

        erb_emb = self.erb_fc(e3_flat)   # (B, T, emb_size)
        df_emb  = self.df_fc(c1_flat)    # (B, T, emb_size)

        emb = torch.cat([erb_emb, df_emb], dim=-1)   # (B, T, 2*emb_size)
        emb = self.emb_fc(emb)                        # (B, T, emb_size)
        emb, _ = self.gru_emb(emb)                   # (B, T, emb_size)
        lsnr = self.lsnr_fc(emb)                     # (B, T, 1)

        return e0, e1, e2, e3, emb, c0, lsnr


# ============================================================
# ERB Decoder
# ============================================================

class ERBDecoder(nn.Module):
    """
    4-level U-Net decoder: emb → (B,1,T,n_erb) sigmoid mask.
    Skip connections from encoder (e3, e2, e1, e0) concatenated at each level.
    """
    def __init__(self, n_erb, enc_ch=16, emb_size=256, conv_kernel=(1, 3)):
        super().__init__()
        # Project embedding back to spatial representation
        self.emb_fc = nn.Linear(emb_size, enc_ch * (n_erb // 4), bias=False)

        # Derive output_padding for each stride-2 transposed conv so shapes match
        # the skip connection tensors from the encoder.
        # ConvTranspose2d(kernel=1,3; stride=1,2; padding=0,1) gives:
        #   F_out = (F_in - 1)*2 - 2*1 + 3 + op = 2*F_in - 1 + op
        # We need F_out to equal the corresponding encoder skip size.
        n_erb_4   = n_erb // 4
        n_erb_2   = n_erb // 2
        # dc2: input F = n_erb_4 = 8, skip target = n_erb_2 = 16
        op  = 1 if (2 * n_erb_4 - 1) < n_erb_2 else 0
        # dc1: input F = n_erb_2 = 16, skip target = n_erb = 32
        op2 = 1 if (2 * n_erb_2 - 1) < n_erb else 0

        p = conv_kernel[1] // 2

        # dc3: no freq upsampling, skip from e3
        self.dc3 = SeparableConvTranspose2d(enc_ch * 2, enc_ch, conv_kernel,
                                             padding=(0, p))
        # dc2: freq × 2, skip from e2
        self.dc2 = SeparableConvTranspose2d(enc_ch * 2, enc_ch, conv_kernel,
                                             stride=(1, 2), padding=(0, p),
                                             output_padding=(0, op))
        # dc1: freq × 2, skip from e1
        self.dc1 = SeparableConvTranspose2d(enc_ch * 2, enc_ch, conv_kernel,
                                             stride=(1, 2), padding=(0, p),
                                             output_padding=(0, op2))
        # dc0: no freq upsampling, skip from e0 → single channel mask
        self.dc0 = nn.Sequential(
            nn.ConvTranspose2d(enc_ch * 2, 1, conv_kernel,
                               padding=(0, p), bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.enc_ch = enc_ch
        self.n_erb_4 = n_erb_4

    def forward(self, emb, e3, e2, e1, e0):
        """emb: (B, T, emb_size); e0..e3: (B, enc_ch, T, *)"""
        B, T, _ = emb.shape
        # Reshape emb to spatial
        x = self.emb_fc(emb)                           # (B, T, enc_ch*n_erb//4)
        x = x.reshape(B, T, self.enc_ch, self.n_erb_4)
        x = x.permute(0, 2, 1, 3)                      # (B, enc_ch, T, n_erb//4)

        x = self.dc3(torch.cat([x, e3], dim=1))        # (B, enc_ch, T, n_erb//4)
        x = self.dc2(torch.cat([x, e2], dim=1))        # (B, enc_ch, T, n_erb//2)
        x = self.dc1(torch.cat([x, e1], dim=1))        # (B, enc_ch, T, n_erb)
        x = self.dc0(torch.cat([x, e0], dim=1))        # (B, 1, T, n_erb)
        return x


# ============================================================
# DF Decoder
# ============================================================

class DFDecoder(nn.Module):
    """
    Per-frame, per-bin FIR filter coefficients + alpha blend weight.
    Returns coefs: (B, T, df_bins, df_order*2) and alpha: (B, T, 1).
    """
    def __init__(self, df_bins, df_order, df_hidden=256, emb_size=256,
                 enc_ch=16, conv_kernel=(1, 3), num_layers=3, gru_groups=1):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        df_out_ch = df_order * 2   # real + imag

        self.df_gru = GroupedGRU(emb_size, df_hidden, num_layers=num_layers,
                                  groups=gru_groups, batch_first=True)
        # Residual c0 path: (B, enc_ch, T, df_bins) → permute → (B, T, df_bins, enc_ch) → fc
        self.df_convp = SeparableConv2d(enc_ch, df_out_ch, conv_kernel,
                                         padding=(0, conv_kernel[1] // 2))
        self.df_out  = nn.Sequential(
            nn.Linear(df_hidden, df_bins * df_out_ch), nn.Tanh()
        )
        self.df_fc_a = nn.Sequential(nn.Linear(df_hidden, 1), nn.Sigmoid())

    def forward(self, emb, c0):
        """
        emb : (B, T, emb_size)
        c0  : (B, enc_ch, T, df_bins)
        Returns:
            coefs : (B, T, df_bins, df_order*2)
            alpha : (B, T, 1)
        """
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)                             # (B, T, df_hidden)

        # c0 residual: conv → (B, df_out_ch, T, df_bins) → permute → (B, T, df_bins, df_out_ch)
        c0_res = self.df_convp(c0).permute(0, 2, 3, 1)     # (B, T, df_bins, df_order*2)

        alpha = self.df_fc_a(c)                              # (B, T, 1)
        c_out = self.df_out(c)                               # (B, T, df_bins * df_order*2)
        coefs = c_out.view(b, t, self.df_bins, self.df_order * 2) + c0_res
        return coefs, alpha


# ============================================================
# Deep Filter Apply
# ============================================================

def deep_filter_apply(spec, coefs, alpha, df_bins, df_order, df_lookahead=0):
    """
    Apply a per-bin FIR filter to the lowest df_bins of spec.

    The deployed configuration uses ``df_lookahead=0``: order 5 therefore
    consumes masked spectra ``[t-4, ..., t]`` and only requires four history
    frames in a streaming ring buffer.  Mask lookahead is an independent model
    delay and must not be folded into this FIR window.

    Args:
        spec   : (B, n_bins, T) complex
        coefs  : (B, T, df_bins, df_order*2)
        alpha  : (B, T, 1) blend weight
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

    # Alpha blend: alpha is (B, T, 1) → permute to (B, 1, T) for broadcast over df_bins
    alpha_t = alpha.permute(0, 2, 1)   # (B, 1, T)
    out = spec.clone()
    out[:, :df_bins] = alpha_t * df_out + (1 - alpha_t) * spec[:, :df_bins]
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
                 enc_ch=16, emb_size=256, df_hidden=256, df_num_layers=3,
                 gru_groups=1, mask_lookahead=1, df_lookahead=0):
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
            n_erb, df_bins, enc_ch, emb_size, gru_groups,
            mask_lookahead=mask_lookahead,
        )
        self.erb_dec  = ERBDecoder(n_erb, enc_ch, emb_size)
        self.df_dec   = DFDecoder(df_bins, df_order, df_hidden, emb_size,
                                   enc_ch, num_layers=df_num_layers,
                                   gru_groups=gru_groups)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DeepFilterNet2: {n_params:,} trainable parameters")

    def forward(self, spec, feat_erb, feat_spec):
        """
        spec      : (B, n_bins, T) complex
        feat_erb  : (B, 1, T, n_erb)
        feat_spec : (B, 2, T, df_bins)
        """
        e0, e1, e2, e3, emb, c0, lsnr = self.encoder(feat_erb, feat_spec)

        # ERB mask → expand to per-bin
        erb_mask = self.erb_dec(emb, e3, e2, e1, e0)   # (B, 1, T, n_erb)
        bin_mask = erb_mask.squeeze(1).matmul(self.erb_inv)   # (B, T, n_bins)
        bin_mask = bin_mask.permute(0, 2, 1)                  # (B, n_bins, T) — NOT .T
        spec = spec * bin_mask

        # DF filter on low-freq bins
        coefs, alpha = self.df_dec(emb, c0)
        spec = deep_filter_apply(
            spec, coefs, alpha, self.df_bins, self.df_order,
            self.df_lookahead,
        )

        return spec, erb_mask
