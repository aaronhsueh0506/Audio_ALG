"""Dual-input streaming twin of DeepFilterNet2's ``dfn2_prepost`` class.

One frame in, at most one head evaluation, one (delayed) frame out -- the
FREQ-mode contract of ``AINR/DeepFilterNet2/dfn2_prepost.c`` with the single
staging pair split in two:

* the ESTIMATION spectrum ``A`` feeds ``dfn2_compute_features`` (the ERB and
  complex-feature EMA normalisers, the ``[t-1, t, t+1]`` graph window);
* the APPLICATION spectrum ``B`` feeds ``dfn2_compose_stream`` (the noisy ring,
  the ERB mask, the deep-filter FIR, the alpha blend, the attenuation limit).

The two touch disjoint state, exactly as the C separates them.  Both spectra
arrive on the pipeline's grid in AUDIO scale (the unnormalised sqrt-Hann STFT
lib/aec exposes as ``AecResContext.error_spec``); DFN2 consumes
``torch.stft(normalized=True)`` scale, so ``SCALE_IN = 2**-5`` is applied on
the way in and ``SCALE_OUT = 2**5`` on the way out.  Both are exact powers of
two, so the round trip is bit-exact in float32.

Timing: the output emitted on frame ``t`` belongs to source frame ``t - 2``
(``MASK_LOOKAHEAD + DF_LOOKAHEAD``, a cascade, not a maximum).  Frames 0 and 1
emit nothing; after the last real frame the caller pushes two all-zero frames
(``flush``) to drain the last two source frames, with the heads still being
evaluated on them -- the same tail discipline the C class documents.

Fail-open: a heads callable that raises, or returns a non-finite value in any
head or next-state tensor, is treated exactly like ``dfn2_prepost_frame_skip``:
unit ERB mask, zero taps, alpha 0 (an exact identity through the cascade,
because ``erb_inv`` is a partition of unity), recurrent state NOT stepped,
framing and compose clocks advanced.

The compose arithmetic keeps real and imaginary parts as separate float32
arrays and performs every multiply and add as its own numpy operation, so the
rounding sequence is the C's ``-ffp-contract=off`` sequence and not a fused or
reassociated one.
"""

from __future__ import annotations

import configparser
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AINR_DIR = os.path.join(_ROOT, 'AINR')
_DFN2_DIR = os.path.join(_AINR_DIR, 'DeepFilterNet2')

# Grid and geometry pinned by the model contract (config.ini / dfn2_process.h).
SAMPLE_RATE = 48000
N_FFT = 1024
HOP = 512
N_BINS = N_FFT // 2 + 1
N_ERB = 32
DF_BINS = 96
DF_ORDER = 5
MASK_LOOKAHEAD = 1
DF_LOOKAHEAD = 1
DF_HISTORY = DF_ORDER - DF_LOOKAHEAD - 1
DF_RING = 5
MODEL_LOOKAHEAD = MASK_LOOKAHEAD + DF_LOOKAHEAD
INPUT_FRAMES = 3

#: Pipeline (audio-scale, unnormalised rfft) -> DFN2 (normalized=True) and back.
SCALE_IN = np.float32(2.0 ** -5)
SCALE_OUT = np.float32(2.0 ** 5)

_DFN2_MODULES = None


#: Bare module names every AINR project defines for itself.
_AINR_BARE_NAMES = ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx')


def dfn2_modules():
    """Import the DFN2 training/export modules the way they import each other
    (bare names with their directory first on ``sys.path``).

    Each AINR project has its own top-level ``train.py``/``model.py``/
    ``export_onnx.py``; in one Python session the first project imported
    owns those names in ``sys.modules``, so a sibling's module would be
    handed back silently. The cached names are dropped and the DFN2
    directory moved to the front of ``sys.path`` before importing, as the
    projects' own tests do."""
    global _DFN2_MODULES
    if _DFN2_MODULES is None:
        for path in (_AINR_DIR, _DFN2_DIR):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        for stale in _AINR_BARE_NAMES:
            sys.modules.pop(stale, None)
        import train as dfn2_train  # noqa: E402
        import model as dfn2_model  # noqa: E402
        import export_onnx as dfn2_export  # noqa: E402
        _DFN2_MODULES = (dfn2_train, dfn2_model, dfn2_export)
    return _DFN2_MODULES


@dataclass
class Dfn2Assets:
    model: torch.nn.Module
    feature_cfg: dict
    erb_fb: torch.Tensor          # (n_erb, n_bins), the trained forward bank
    erb_inv: np.ndarray           # (n_erb, n_bins) float32, band-major


def load_dfn2(checkpoint: Optional[str] = None,
              config_path: Optional[str] = None,
              seed: Optional[int] = None) -> Dfn2Assets:
    """Build the DFN2 model from ``config.ini`` and optionally a checkpoint.

    ``checkpoint=None`` gives a randomly initialised model (``seed`` pins it),
    which is what the contract tests use; the experiments load
    ``AINR/DeepFilterNet2/output/dfn2_best.pth``.
    """
    train, model_mod, _export = dfn2_modules()
    cfg = configparser.ConfigParser()
    cfg.read(config_path or os.path.join(_DFN2_DIR, 'config.ini'))
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    hop = cfg.getint('signal', 'hop_len', fallback=n_fft // 2)
    if (sr, n_fft, hop) != (SAMPLE_RATE, N_FFT, HOP):
        raise ValueError('this stage is pinned to 48000/1024/512, config says '
                         f'{sr}/{n_fft}/{hop}')
    model_cfg = train.read_model_config(cfg)
    if (model_cfg['n_erb'], model_cfg['df_bins'], model_cfg['df_order'],
            model_cfg['mask_lookahead'], model_cfg['df_lookahead']) != (
            N_ERB, DF_BINS, DF_ORDER, MASK_LOOKAHEAD, DF_LOOKAHEAD):
        raise ValueError('model geometry differs from the pinned contract')
    feature_cfg = train.read_feature_config(cfg, sr, hop)
    if seed is not None:
        torch.manual_seed(seed)
    net = model_mod.DeepFilterNet2(**model_cfg)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        # The same contract gate inference.py applies: a checkpoint trained
        # under another model/feature/loss version has compatible tensor
        # shapes but is a different network, and loading it silently would
        # make every downstream number meaningless.
        contract = train.make_checkpoint_contract(
            sr, n_fft, cfg.getint('signal', 'win_len', fallback=n_fft), hop,
            model_cfg['n_erb'], model_cfg['df_bins'], model_cfg['df_order'],
            model_cfg['mask_lookahead'], model_cfg['df_lookahead'],
            model_cfg['mask_pf'], model_cfg['pf_beta'], feature_cfg,
            train.read_loss_config(cfg))
        train.require_checkpoint_contract(state, contract, context=checkpoint,
                                          for_training=False)
        net.load_state_dict(state['state_dict'])
    net.eval()
    erb_inv = net.erb_inv.detach().cpu().numpy().astype(np.float32)
    return Dfn2Assets(model=net, feature_cfg=feature_cfg,
                      erb_fb=net.erb_fb, erb_inv=erb_inv)


def expand_erb_mask(band_gain: np.ndarray, erb_inv: np.ndarray) -> np.ndarray:
    """``df_common_expand_mask``: accumulate band by band, in band order."""
    bin_gain = np.zeros(erb_inv.shape[1], dtype=np.float32)
    for b in range(erb_inv.shape[0]):
        bin_gain += erb_inv[b] * np.float32(band_gain[b])
    return bin_gain


def erb_inv_is_partition_of_unity(erb_inv: np.ndarray) -> bool:
    """A unit band mask must expand to exactly 1.0f per bin, in the C order."""
    return bool(np.all(expand_erb_mask(np.ones(erb_inv.shape[0], np.float32),
                                       erb_inv) == np.float32(1.0)))


HeadsResult = Tuple[np.ndarray, np.ndarray, float]


class IdentityHeads:
    """``frame_skip``'s heads: unit mask, zero taps, alpha 0.  No state."""

    def __call__(self, erb_window: np.ndarray, spec_window: np.ndarray) -> HeadsResult:
        return (np.ones(N_ERB, np.float32),
                np.zeros((DF_BINS, DF_ORDER, 2), np.float32), 0.0)

    def reset(self) -> None:
        pass


class Dfn2Heads:
    """The exported graph, stepped one frame at a time with explicit state.

    Wraps ``export_onnx.StatelessDFN2Heads`` (the split, batch-one layout the
    C binds) and owns the four state tensors the accelerator would carry.
    A call commits the next state only when every head and every next-state
    tensor is finite; otherwise it raises and leaves the state untouched, so
    the stage's fail-open path sees exactly what ``frame_commit`` refuses.
    """

    def __init__(self, model: torch.nn.Module):
        _train, _model, export = dfn2_modules()
        self.wrapper = export.StatelessDFN2Heads(model).eval()
        self.reset()

    def reset(self) -> None:
        self.state = tuple(self.wrapper.initial_inputs()[2:])

    @torch.no_grad()
    def __call__(self, erb_window: np.ndarray, spec_window: np.ndarray) -> HeadsResult:
        erb = torch.from_numpy(np.ascontiguousarray(erb_window, np.float32)).view(
            1, 1, INPUT_FRAMES, N_ERB)
        spec = torch.from_numpy(np.ascontiguousarray(spec_window, np.float32)).view(
            1, 2, INPUT_FRAMES, DF_BINS)
        outputs = self.wrapper(erb, spec, *self.state)
        erb_mask, coefs, alpha = outputs[:3]
        next_state = tuple(outputs[3:])
        tensors = (erb_mask, coefs, alpha) + next_state
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise FloatingPointError('non-finite head or next-state tensor')
        self.state = next_state
        mask = erb_mask.reshape(-1).numpy().astype(np.float32)
        taps = coefs.reshape(DF_BINS, DF_ORDER, 2).numpy().astype(np.float32)
        return mask, taps, float(alpha.reshape(-1)[0])


@dataclass
class StageOutput:
    spectrum: np.ndarray      # complex64 (n_bins,), audio scale
    frame_index: int          # source frame this output belongs to
    bin_gain: np.ndarray      # float32 (n_bins,), the expanded ERB gain that
                              # frame got. A test seam only (the C stage keeps
                              # no such history): it lets the tests witness
                              # which input the heads were estimated from.


class Dfn2Stage:
    """Estimate on ``A``, apply on ``B``; see the module docstring."""

    def __init__(self, assets: Dfn2Assets,
                 heads: Optional[Callable[[np.ndarray, np.ndarray], HeadsResult]] = None,
                 atten_lim_db: float = 0.0):
        if not erb_inv_is_partition_of_unity(assets.erb_inv):
            raise ValueError('erb_inv is not a partition of unity; the identity '
                             'frame_skip would not be exact')
        self.assets = assets
        self.heads = heads if heads is not None else Dfn2Heads(assets.model)
        self.atten_lim_db = float(atten_lim_db)
        self.reset()

    # ---- lifecycle -----------------------------------------------------
    def reset(self) -> None:
        self.ema_state = None
        self.erb_window = np.zeros((INPUT_FRAMES, N_ERB), np.float32)
        self.spec_window = np.zeros((2, INPUT_FRAMES, DF_BINS), np.float32)
        self.feature_frames_seen = 0
        self.frames_in = 0                       # dfn2_prepost's analysis clock
        self.stream_frame_index = 0              # DFN2State.stream_frame_index
        self.noisy_re = np.zeros((DF_RING, N_BINS), np.float32)
        self.noisy_im = np.zeros((DF_RING, N_BINS), np.float32)
        self.df_re = np.zeros((DF_RING, DF_BINS), np.float32)
        self.df_im = np.zeros((DF_RING, DF_BINS), np.float32)
        self.hi_re = np.zeros((DF_RING, N_BINS - DF_BINS), np.float32)
        self.hi_im = np.zeros((DF_RING, N_BINS - DF_BINS), np.float32)
        self.coef_ring = np.zeros((DF_RING, DF_BINS, DF_ORDER, 2), np.float32)
        self.alpha_ring = np.zeros(DF_RING, np.float32)
        self.gain_ring = np.zeros((DF_RING, N_BINS), np.float32)
        self.heads_calls = 0
        self.skips = 0
        if hasattr(self.heads, 'reset'):
            self.heads.reset()

    def set_atten_lim(self, atten_lim_db: float) -> None:
        if not np.isfinite(atten_lim_db):
            raise ValueError('atten_lim_db must be finite')
        self.atten_lim_db = float(atten_lim_db)

    # ---- estimation branch ----------------------------------------------
    def _features(self, a_re: np.ndarray, a_im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        train, _model, _export = dfn2_modules()
        spec = torch.from_numpy(np.ascontiguousarray(a_re + 1j * a_im, np.complex64))
        spec = spec.view(1, N_BINS, 1)
        with torch.no_grad():
            _spec, feat_erb, feat_spec, self.ema_state = train.extract_dfn2_features(
                spec, self.assets.erb_fb, DF_BINS, self.assets.feature_cfg,
                self.ema_state)
        erb = feat_erb.reshape(N_ERB).numpy().astype(np.float32)
        cplx = feat_spec.reshape(2, DF_BINS).numpy().astype(np.float32)
        return erb, cplx

    def _push_window(self, erb: np.ndarray, cplx: np.ndarray) -> bool:
        """``dfn2_model_io_push_features``: slide, append, report readiness."""
        self.erb_window[:-1] = self.erb_window[1:]
        self.erb_window[-1] = erb
        self.spec_window[:, :-1] = self.spec_window[:, 1:]
        self.spec_window[:, -1] = cplx
        if self.feature_frames_seen < 2:
            self.feature_frames_seen += 1
        return self.feature_frames_seen == 2

    # ---- application branch ---------------------------------------------
    def _compose(self, b_re: np.ndarray, b_im: np.ndarray, heads_valid: bool,
                 mask: Optional[np.ndarray], coefs: Optional[np.ndarray],
                 alpha: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], int]:
        """``dfn2_compose_stream`` on planar float32 arrays."""
        current = self.stream_frame_index
        if current < MASK_LOOKAHEAD:
            if heads_valid:
                raise RuntimeError('heads offered before the mask lookahead')
        elif not heads_valid:
            raise RuntimeError('heads missing after the mask lookahead')
        self.stream_frame_index += 1
        current_slot = current % DF_RING
        self.noisy_re[current_slot] = b_re
        self.noisy_im[current_slot] = b_im
        if current < MASK_LOOKAHEAD:
            return False, None, None, -1

        head_frame = current - MASK_LOOKAHEAD
        head_slot = head_frame % DF_RING
        bin_gain = expand_erb_mask(mask, self.assets.erb_inv)
        self.gain_ring[head_slot] = bin_gain
        self.df_re[head_slot] = self.noisy_re[head_slot, :DF_BINS] * bin_gain[:DF_BINS]
        self.df_im[head_slot] = self.noisy_im[head_slot, :DF_BINS] * bin_gain[:DF_BINS]
        self.hi_re[head_slot] = self.noisy_re[head_slot, DF_BINS:] * bin_gain[DF_BINS:]
        self.hi_im[head_slot] = self.noisy_im[head_slot, DF_BINS:] * bin_gain[DF_BINS:]
        self.coef_ring[head_slot] = coefs
        self.alpha_ring[head_slot] = np.float32(alpha)

        target_frame = head_frame - DF_LOOKAHEAD
        if target_frame < 0:
            return False, None, None, -1
        target_slot = target_frame % DF_RING
        a = float(self.alpha_ring[target_slot])
        a = min(max(a, 0.0), 1.0)
        a32 = np.float32(a)
        one_minus = np.float32(1.0) - a32

        filtered_re = np.zeros(DF_BINS, np.float32)
        filtered_im = np.zeros(DF_BINS, np.float32)
        for tap in range(DF_ORDER):
            source_frame = target_frame - DF_HISTORY + tap
            if source_frame >= 0:
                s = source_frame % DF_RING
                xr = self.df_re[s]
                xi = self.df_im[s]
            else:
                xr = np.zeros(DF_BINS, np.float32)
                xi = xr
            cr = self.coef_ring[target_slot, :, tap, 0]
            ci = self.coef_ring[target_slot, :, tap, 1]
            filtered_re += (xr * cr) - (xi * ci)
            filtered_im += (xi * cr) + (xr * ci)
        out_re = np.empty(N_BINS, np.float32)
        out_im = np.empty(N_BINS, np.float32)
        out_re[:DF_BINS] = (a32 * filtered_re) + (one_minus * self.df_re[target_slot])
        out_im[:DF_BINS] = (a32 * filtered_im) + (one_minus * self.df_im[target_slot])
        out_re[DF_BINS:] = self.hi_re[target_slot]
        out_im[DF_BINS:] = self.hi_im[target_slot]
        if self.atten_lim_db != 0.0:
            lim = np.float32(10.0 ** (-abs(self.atten_lim_db) / 20.0))
            mix = np.float32(1.0) - lim
            out_re = (self.noisy_re[target_slot] * lim) + (out_re * mix)
            out_im = (self.noisy_im[target_slot] * lim) + (out_im * mix)
        return True, out_re, out_im, target_frame

    # ---- per-frame entry --------------------------------------------------
    def push(self, estimate: np.ndarray, apply: np.ndarray) -> Optional[StageOutput]:
        """One frame of ``A`` (estimate) and ``B`` (apply), both complex64
        (n_bins,) at audio scale.  Returns the emitted source frame or None
        during the two warm-up frames."""
        a = np.asarray(estimate, np.complex64)
        b = np.asarray(apply, np.complex64)
        if a.shape != (N_BINS,) or b.shape != (N_BINS,):
            raise ValueError('expected two (%d,) complex64 spectra' % N_BINS)
        a_re = (a.real * SCALE_IN).astype(np.float32)
        a_im = (a.imag * SCALE_IN).astype(np.float32)
        b_re = (b.real * SCALE_IN).astype(np.float32)
        b_im = (b.imag * SCALE_IN).astype(np.float32)

        erb, cplx = self._features(a_re, a_im)
        heads_needed = self._push_window(erb, cplx)
        mask = coefs = None
        alpha = 0.0
        if heads_needed:
            try:
                mask, coefs, alpha = self.heads(self.erb_window, self.spec_window)
                self.heads_calls += 1
                if (mask.shape != (N_ERB,) or coefs.shape != (DF_BINS, DF_ORDER, 2)
                        or not np.isfinite(alpha)
                        or not np.all(np.isfinite(mask))
                        or not np.all(np.isfinite(coefs))):
                    raise FloatingPointError('malformed heads')
            except Exception:
                # frame_skip: exact identity for this frame, state untouched.
                self.skips += 1
                mask, coefs, alpha = IdentityHeads()(self.erb_window, self.spec_window)
        emitted, out_re, out_im, target = self._compose(
            b_re, b_im, heads_needed, mask, coefs, alpha)
        self.frames_in += 1
        if not emitted:
            return None
        spectrum = ((out_re * SCALE_OUT).astype(np.float32)
                    + 1j * (out_im * SCALE_OUT).astype(np.float32)).astype(np.complex64)
        return StageOutput(spectrum=spectrum, frame_index=target,
                           bin_gain=self.gain_ring[target % DF_RING].copy())

    def flush(self):
        """Drain the last two source frames by pushing two all-zero frames."""
        zero = np.zeros(N_BINS, np.complex64)
        outputs = []
        for _ in range(MODEL_LOOKAHEAD):
            out = self.push(zero, zero)
            if out is not None:
                outputs.append(out)
        return outputs

    @property
    def output_delay_frames(self) -> int:
        return MODEL_LOOKAHEAD
