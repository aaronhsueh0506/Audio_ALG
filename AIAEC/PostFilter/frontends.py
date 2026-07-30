"""The FROZEN stage-1 front-ends the PostFilter is trained against.

A front-end turns ``(Y, X)`` into ``(E, D_hat)`` on the shared STFT grid:

    D_hat = AEC(Y, X)      an ECHO ESTIMATE
    E     = Y - D_hat      by SUBTRACTION, never a mask on Y
    R     = D - D_hat      the residual; it emerges, it is never a target

⚠ WHY THE IDENTITY OF THE FRONT-END IS PART OF THE CHECKPOINT
Every front-end here leaves a different residual behind -- different convergence
speed, different musical noise, different nonlinear echo survival, different
near-end leakage, different path-change transients.  A PostFilter trained on one
and attached to another is a legitimate out-of-distribution experiment and NOT a
comparable result.  ``frontend_id`` is therefore recorded in the checkpoint,
gated on resume, and appended to a permanent ``frontend_history`` when the gate
is overridden, so the two can never be confused after the fact.

⚠ Nothing in here is trained.  The trainer runs it under ``torch.no_grad()``
and detaches its output; if a gradient ever reaches a front-end parameter, the
"frozen" in the title has stopped being true.
"""

import hashlib
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen_aec import AecGrid, alpha_from_tau, frames_from_seconds  # noqa: E402


__all__ = [
    'FrozenFrontEnd',
    'NullFrontEnd',
    'OracleFrontEnd',
    'PluginFrontEnd',
    'StftNlmsFrontEnd',
    'build_frontend',
]


class FrozenFrontEnd:
    """Interface every front-end implements.

    ``process(Y, X, state, D=None) -> (E, D_hat, state)`` with all spectra
    ``(B, n_freqs, n_frames)`` complex.  ``D`` is passed only so the oracle can
    use it; a real front-end must ignore it, and one that does not is not a
    front-end.
    """

    frontend_id = 'undefined'
    stateful = False

    def init_state(self, batch, device=None):
        return None

    def reset_lanes(self, state, reset):
        """Zero the lanes flagged by ``reset`` (their sequence just started)."""
        if state is None:
            return None
        reset = reset.to(dtype=torch.bool)
        out = {}
        for key, value in state.items():
            keep = (~reset).to(value.dtype).view(-1, *([1] * (value.dim() - 1)))
            out[key] = value * keep
        return out

    def process(self, Y, X, state=None, D=None):
        raise NotImplementedError


class NullFrontEnd(FrozenFrontEnd):
    """``D_hat = 0``, ``E = Y``.

    The pure noise-suppression ablation, and the only configuration in which
    this project answers the same question as the NR bake-off models.  ⚠ Also
    the input the PostFilter sees during a reference dropout regardless of which
    front-end is configured, which is why the model must not depend on D_hat
    being non-zero (tests/test_model.py asserts it).
    """

    frontend_id = 'none_v1'

    def process(self, Y, X, state=None, D=None):
        return Y, torch.zeros_like(Y), state


class OracleFrontEnd(FrozenFrontEnd):
    """``D_hat = D``, so ``R = 0`` and the PostFilter faces noise only.

    ⚠ A DIAGNOSTIC UPPER BOUND, not a result.  It answers "how much of the
    remaining damage is noise rather than residual echo"; any number derived
    from it must be reported as "oracle AEC" or it is a claim about a system
    nobody can build.
    """

    frontend_id = 'oracle_v1'

    def process(self, Y, X, state=None, D=None):
        if D is None:
            raise ValueError(
                "OracleFrontEnd needs the true echo D; it is only usable in "
                "training/evaluation over the packed corpus, never at inference")
        return Y - D, D, state


class StftNlmsFrontEnd(FrozenFrontEnd):
    """Per-bin multi-tap NLMS echo canceller on the shared STFT grid.

    ``D_hat(f,t) = sum_k W_k(f) X(f, t-k)``, ``E = Y - D_hat``, with the usual
    normalised complex LMS update ``W_k += mu * E * conj(X(f,t-k)) / sum|X|^2``.

    This is the multi-delay (partitioned frequency-domain) structure a classical
    canceller uses, and it behaves like one: it converges over seconds,
    mis-converges during double talk, and has to re-converge after an echo-path
    change.  That behaviour is the POINT -- a front-end that produced a clean
    residual would train a PostFilter that has never seen the residual it will
    actually be given.

    ⚠ Per-bin filtering ignores cross-band leakage between neighbouring FFT
    bins, so this cancels a little less than a time-domain filter of the same
    length.  It is an approximation to a classical AEC, not a port of one; do
    not quote its ERLE as "the classical AEC's ERLE".

    ⚠ ``taps`` is the RESOLVED filter length in frames, not a config knob.  The
    config states the coverage as ``filter_span_sec`` and ``build_frontend``
    converts it, because the requirement is acoustic: the span must cover the
    corpus's bulk delay plus the room RIR.  Too short and every residual in the
    corpus is "the filter could not reach the echo", which is a different
    distribution from "the filter reached it and did its best" -- and a taps
    count written straight into a config quietly becomes too short at 48 kHz.
    """

    stateful = True

    def __init__(self, grid: AecGrid, taps=16, mu=0.35, leak=0.9999,
                 far_active_db=-60.0, peak_tau_sec=10.0,
                 randomize=False, taps_choices=(4, 8, 16, 32),
                 mu_range=(0.1, 0.7), seed=0):
        if taps < 1:
            raise ValueError(f"taps must be >= 1, got {taps}")
        if not 0.0 < mu <= 2.0:
            raise ValueError(f"mu must be in (0, 2], got {mu}")
        self.grid = grid
        self.randomize = bool(randomize)
        self.taps_choices = tuple(int(t) for t in taps_choices)
        if self.randomize and min(self.taps_choices) < 1:
            raise ValueError("taps_choices must all be >= 1")
        self.mu_range = (float(mu_range[0]), float(mu_range[1]))
        # ⚠ With randomisation on, the filter is ALLOCATED at the longest choice
        # and each lane masks itself down.  A per-lane allocation would need a
        # ragged tensor; masking costs one multiply and keeps the shape static.
        self.taps = max(self.taps_choices) if self.randomize else int(taps)
        self.mu = float(mu)
        self.leak = float(leak)
        self.far_active_db = float(far_active_db)
        self._gen = torch.Generator().manual_seed(int(seed))
        # The reference-activity gate compares the frame's reference power to a
        # slowly decaying peak.  ⚠ In SECONDS: a frame-count decay would mean
        # 10 s at 16 kHz and 3.3 s at 48 kHz.
        self.peak_decay = alpha_from_tau(peak_tau_sec, grid.hop_len, grid.sr)
        self.peak_tau_sec = float(peak_tau_sec)
        if self.randomize:
            # ⚠ The id names the DISTRIBUTION, not a point.  That is the whole
            # change: a postfilter trained under randomisation is not tied to one
            # front-end configuration, so pinning it to one would be a lie.
            choices = '-'.join(str(t) for t in self.taps_choices)
            self.frontend_id = (
                f"stft_nlms_rand_p{{{choices}}}"
                f"_mu[{self.mu_range[0]:g},{self.mu_range[1]:g}]"
                f"_leak{self.leak:g}_gate{self.far_active_db:g}_seed{seed}_v1")
        else:
            self.frontend_id = (
                f"stft_nlms_p{self.taps}_mu{self.mu:g}_leak{self.leak:g}"
                f"_gate{self.far_active_db:g}_v1")

    def _draw(self, n):
        """Per-lane (taps, mu) draws.  CPU generator: reproducible under --seed
        and independent of how many CUDA streams happen to exist."""
        idx = torch.randint(len(self.taps_choices), (n,), generator=self._gen)
        taps = torch.tensor([self.taps_choices[i] for i in idx.tolist()])
        lo, hi = self.mu_range
        mu = lo + (hi - lo) * torch.rand(n, generator=self._gen)
        return taps, mu

    def _tap_mask(self, taps_per_lane, device):
        """(B, taps, 1) float mask zeroing each lane's unused tail."""
        ramp = torch.arange(self.taps, device=device).view(1, -1)
        return (ramp < taps_per_lane.to(device).view(-1, 1)).to(
            torch.float32).unsqueeze(-1)

    def init_state(self, batch, device=None):
        n_freqs, taps = self.grid.n_freqs, self.taps
        state = {
            'w': torch.zeros(batch, taps, n_freqs, dtype=torch.complex64,
                             device=device),
            'xbuf': torch.zeros(batch, taps, n_freqs, dtype=torch.complex64,
                                device=device),
            'peak': torch.zeros(batch, dtype=torch.float32, device=device),
        }
        if self.randomize:
            taps_per_lane, mu = self._draw(batch)
            state['tap_mask'] = self._tap_mask(taps_per_lane, device)
            state['mu'] = mu.to(device=device, dtype=torch.float32)
        return state

    def reset_lanes(self, state, reset):
        """Zero the reset lanes AND redraw their front-end configuration.

        ⚠ Redrawing at the sequence boundary is the point of randomisation: the
        postfilter must see many front-end behaviours, not one. Redrawing MID
        sequence would instead look like the canceller being reconfigured
        underneath it, which no real system does.
        """
        state = super().reset_lanes(state, reset)
        if state is None or not self.randomize:
            return state
        reset = reset.to(dtype=torch.bool)
        n = int(reset.sum())
        if n == 0:
            return state
        device = state['w'].device
        taps_per_lane, mu = self._draw(n)
        new_mask = self._tap_mask(taps_per_lane, device)
        # super() already zeroed tap_mask/mu for the reset lanes; write the fresh
        # draws into exactly those rows.
        state['tap_mask'] = state['tap_mask'].clone()
        state['mu'] = state['mu'].clone()
        state['tap_mask'][reset] = new_mask
        state['mu'][reset] = mu.to(device=device, dtype=torch.float32)
        return state

    def process(self, Y, X, state=None, D=None):
        batch, n_freqs, n_frames = Y.shape
        if n_freqs != self.grid.n_freqs:
            raise ValueError(
                f"spectrum has {n_freqs} bins, grid says {self.grid.n_freqs}")
        if state is None:
            state = self.init_state(batch, device=Y.device)
        w, xbuf, peak = state['w'], state['xbuf'], state['peak']
        w = w.to(Y.device)
        xbuf = xbuf.to(Y.device)
        peak = peak.to(Y.device)
        if self.randomize:
            tap_mask = state['tap_mask'].to(Y.device)          # (B, taps, 1)
            # ⚠ (B,), NOT (B,1): it multiplies `active`, which is (B,).  A (B,1)
            # here broadcasts to (B,B) and every lane gets every other lane's
            # step size -- a shape bug that still runs when B == 1.
            mu = state['mu'].to(Y.device)                      # (B,)
            w = w * tap_mask                                   # enforce on resume
        else:
            tap_mask = None
            mu = self.mu

        gate = 10.0 ** (self.far_active_db / 10.0)
        estimates, outputs = [], []
        for t in range(n_frames):
            x_t = X[:, :, t]
            xbuf = torch.cat([x_t.unsqueeze(1), xbuf[:, :-1]], dim=1)

            d_hat = (w * xbuf).sum(dim=1)                     # (B, F)
            e = Y[:, :, t] - d_hat
            estimates.append(d_hat)
            outputs.append(e)

            frame_power = (x_t.real.square() + x_t.imag.square()).mean(dim=1)
            peak = torch.maximum(peak * self.peak_decay, frame_power)
            active = (frame_power > peak * gate).to(Y.real.dtype)

            # ⚠ The leak is applied only while adapting.  Applying it always
            # would decay a converged filter to zero through a long near-only
            # stretch, and the model would then see a bogus "the AEC forgot the
            # path" transient that no real canceller produces.
            norm = (xbuf.real.square() + xbuf.imag.square()).sum(dim=1)
            step = (mu * active).unsqueeze(-1) / norm.clamp_min(1e-12)
            leak = 1.0 - active.view(-1, 1, 1) * (1.0 - self.leak)
            w = w * leak + (step * e).unsqueeze(1).to(w.dtype) * xbuf.conj()
            if tap_mask is not None:
                # ⚠ Re-applied every frame, not just at init: the update writes
                # into every tap, so a mask applied once would be undone on the
                # first adapting frame and every lane would silently run at the
                # longest filter.
                w = w * tap_mask

        state = dict(state)
        state.update(w=w.detach(), xbuf=xbuf.detach(), peak=peak.detach())
        return (torch.stack(outputs, dim=-1), torch.stack(estimates, dim=-1),
                state)


class PluginFrontEnd(FrozenFrontEnd):
    """A stage-1 model (e.g. the AECNet sibling project) loaded by import path.

    ``plugin`` is ``'package.module:factory'``.  The factory is called as
    ``factory(checkpoint=<path>, grid=<AecGrid>, device=<torch.device>)`` and
    must return an object with ``frontend_id``, ``init_state``, ``reset_lanes``
    and ``process``.

    ⚠ The checkpoint's content hash is folded into ``frontend_id``, so the same
    architecture retrained is a DIFFERENT front-end as far as the resume gate is
    concerned.  That is deliberate: retraining stage 1 changes the residual
    distribution just as surely as replacing it does.
    """

    stateful = True

    def __init__(self, spec, checkpoint, grid: AecGrid, device=None):
        if ':' not in spec:
            raise ValueError(
                f"frontend plugin must be 'module:factory', got {spec!r}")
        module_name, factory_name = spec.rsplit(':', 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        self.inner = factory(checkpoint=checkpoint, grid=grid, device=device)
        for required in ('frontend_id', 'init_state', 'reset_lanes', 'process'):
            if not hasattr(self.inner, required):
                raise TypeError(
                    f"{spec} returned an object without {required!r}; it does "
                    f"not satisfy the FrozenFrontEnd interface")
        self.frontend_id = f"{self.inner.frontend_id}@{_file_digest(checkpoint)}"

    def init_state(self, batch, device=None):
        return self.inner.init_state(batch, device=device)

    def reset_lanes(self, state, reset):
        return self.inner.reset_lanes(state, reset)

    def process(self, Y, X, state=None, D=None):
        return self.inner.process(Y, X, state)


def _file_digest(path, length=12):
    if not path:
        raise ValueError("frontend kind='plugin' needs [frontend] checkpoint")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"frontend checkpoint not found: {path}")
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()[:length]


def build_frontend(cfg, grid: AecGrid, device=None) -> FrozenFrontEnd:
    """Construct the configured front-end.  One definition, so train.py and
    denoise.py cannot disagree about what ``frontend_id`` means."""
    kind = cfg.get('frontend', 'kind', fallback='stft_nlms').strip().lower()
    if kind == 'none':
        return NullFrontEnd()
    if kind == 'oracle':
        return OracleFrontEnd()
    if kind == 'stft_nlms':
        spans = [float(s) for s in cfg.get(
            'frontend', 'randomize_span_sec',
            fallback='0.064,0.128,0.256,0.512').split(',') if s.strip()]
        mu_lo = cfg.getfloat('frontend', 'randomize_mu_min', fallback=0.1)
        mu_hi = cfg.getfloat('frontend', 'randomize_mu_max', fallback=0.7)
        return StftNlmsFrontEnd(
            grid,
            taps=frames_from_seconds(
                cfg.getfloat('frontend', 'filter_span_sec', fallback=0.256),
                grid.frame_rate, minimum=1),
            mu=cfg.getfloat('frontend', 'mu', fallback=0.35),
            leak=cfg.getfloat('frontend', 'leak', fallback=0.9999),
            far_active_db=cfg.getfloat('frontend', 'far_active_db', fallback=-60.0),
            randomize=cfg.getboolean('frontend', 'randomize', fallback=True),
            # ⚠ Spans in SECONDS, converted here, for the same reason
            # filter_span_sec exists: "16 taps" is 256 ms at 16 kHz and 171 ms at
            # 48 kHz, and the short one lands under the corpus's bulk delay.
            taps_choices=tuple(
                frames_from_seconds(s, grid.frame_rate, minimum=1) for s in spans),
            mu_range=(mu_lo, mu_hi),
            seed=cfg.getint('training', 'seed', fallback=42),
        )
    if kind == 'plugin':
        return PluginFrontEnd(
            cfg.get('frontend', 'plugin', fallback='').strip(),
            cfg.get('frontend', 'checkpoint', fallback='').strip(),
            grid, device=device)
    raise ValueError(
        f"unknown [frontend] kind={kind!r}; expected one of "
        f"none / oracle / stft_nlms / plugin")
