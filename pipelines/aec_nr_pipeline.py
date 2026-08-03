"""
AEC + NR + RES Processing Pipeline
聲學回聲消除 + 降噪 + 殘留回聲抑制串接管線

Pipeline: AEC(linear) → echo-aware NR(E) → RES  (freq A_min_pl, production).
The NR folds the AEC residual-echo PSD R²(f) into its noise floor (ξ=S²/(N²+R²))
and the near-end floor is far-activity-gated (2026-06-23 re-tune); pass
--legacy-amin for the prior min-only A_min_pl. AEC and NR are each selected by
their own preset; the CLI exposes only presets + switches.

Usage:
    cd Audio_ALG
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-preset balanced --nr-preset balanced
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --nr-preset aggressive
    python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-only
"""

import sys
import os
import argparse
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Add parent directory to path for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                       # lib.* + pipelines.*
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python')) # aec + modules

from lib.aec.python.aec import (
    AEC, AecConfig, AecMode, AecPreset, AecResContext,
)
from lib.nr.denoisers import MmseLsaDenoiser
from lib.nr.core.signal_grid import retime_frame_count
from lib.nr.core.nr_strength import apply_strength
from lib.nr.process_audio import build_v3_2_base_params, load_config

# The standalone ResFilter was retired in AEC v3.21; the separated
# `--pipeline-mode linear` (AEC-linear -> NR -> RES) is now rebuilt on the
# AEC3 freq-domain seam (AecConfig.return_res_context exposes the linear
# error spectrum + the per-frame AEC3 SuppressionGain), so the post-NR RES is
# a pure freq multiply — see run_res() + docs/freq_domain_pipeline_design.md.


def parse_aec_mode(mode_str: str) -> AecMode:
    modes = {
        'fdaf': AecMode.FDAF, 'pbfdaf': AecMode.PBFDAF,
        'pbfdkf': AecMode.PBFDKF,
        'subband': AecMode.SUBBAND,  # backward compat alias (= PBFDKF)
    }
    return modes.get(mode_str, AecMode.PBFDKF)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_aec_classic(mic_signal: np.ndarray, ref_signal: np.ndarray,
                    config: AecConfig) -> np.ndarray:
    """Run AEC with internal RES (classic mode)."""
    aec = AEC(config)
    hop = aec.hop_size
    min_len = min(len(mic_signal), len(ref_signal))
    output = np.zeros(min_len, dtype=np.float32)

    for i in range(0, min_len - hop + 1, hop):
        output[i:i + hop] = aec.process(
            mic_signal[i:i + hop], ref_signal[i:i + hop])

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB")
    return output


def run_aec_linear(mic_signal: np.ndarray, ref_signal: np.ndarray,
                   config: AecConfig, *,
                   standalone_time_output: bool = False,
                   ) -> Tuple[np.ndarray, List[AecResContext]]:
    """Run AEC without RES, returning per-frame context for external RES.

    By default the time output is the reconstructing WOLA hop underlying
    ``ctx.error_spec``.  ``standalone_time_output=True`` instead returns the
    limiter-processed linear AEC hop, matching the C pipeline's ``aec_only``
    path; the frequency-domain context is unchanged.
    """
    config.enable_res = False
    config.return_res_context = True
    aec = AEC(config)
    hop = aec.hop_size
    min_len = min(len(mic_signal), len(ref_signal))
    output = np.zeros(min_len, dtype=np.float32)
    contexts: List[AecResContext] = []

    for i in range(0, min_len - hop + 1, hop):
        result = aec.process(mic_signal[i:i + hop], ref_signal[i:i + hop])
        _out, ctx = result
        # The external WOLA seam may select/crossfade the shadow output.  The
        # full pipeline needs the exact formed hop underlying ctx.error_spec;
        # AEC-only instead mirrors C and returns the standalone limiter output.
        output[i:i + hop] = _out if standalone_time_output else ctx.formed_output
        contexts.append(ctx)

    print(f"  AEC ERLE: {aec.get_erle():.1f} dB  ({len(contexts)} frames)")
    return output, contexts


# 2026-08-03: the standalone hand-rolled NR_PRESETS dict (mild/balanced/aggressive
# {g_min,q,xi_min,alpha_g}) that used to live here was removed. It duplicated and
# diverged from NR's own single source of truth (NR/config/v3_2_config.yaml +
# NR/core/nr_strength.py's apply_strength(), the same composition
# NR/process_audio.py's create_denoiser_from_config() uses) -- most visibly,
# 'balanced'.g_min_db was -15.0 here vs the real NR balanced's -30.0
# (lib/nr/c_impl/include/mmse_lsa_types.h / lib/nr/config/v3_2_config.yaml), an
# amplitude-dB(/20) floor mismatch, not just a naming coincidence. It also never
# wired `strength=` into MmseLsaDenoiser, so mild/aggressive never touched the
# real temporal-smoothing knobs (alpha_noise/alpha_g/alpha_attack/alpha_decay) --
# only g_min_db (via this dict) changed. See _build_denoiser() for the fix.
_NR_YAML_CONFIG = os.path.join(_ROOT, 'lib', 'nr', 'config', 'v3_2_config.yaml')

# MCRA minima-tracking window (MmseLsaDenoiser's `L`): this pipeline's own ~1.5s
# wall-clock design, independent of NR's own L=32 (512ms, tuned for standalone
# noise) -- see _build_denoiser()'s docstring for the full derivation of 94.
_NR_L_MINIMA_WINDOW = 94

# Production recipe. On top of A_min_pl's
# min(G_nr, G_res), two changes — validated 800-case (echo FS +0.12~0.15,
# DT BAK +0.06~0.07, NE protected, all ship bars pass, default-OFF byte-equal):
#   1. UNIFIED gain: fold the AEC residual-echo PSD R²(f) into the OM-LSA noise
#      floor (ξ = S²/(N²+R²), the Speex/Habets canonical) so the NR ALSO
#      suppresses residual echo. g_res stays in the min (dropping it kills echo).
#   2. FAR-ACTIVITY-GATED near-end floor: 0.4 only when the far end is SILENT
#      AND near-end SPEECH is present (true NE → protect); drop to 0.2 when far
#      is active OR near is silent (FS/DT echo + noise gaps → clean background).
# Set LEGACY_AMIN=1 (env / arg) to restore the prior min-only A_min_pl.
PROD_INJECT_ECHO_PSD = True
PROD_NE_FLOOR = 0.4
PROD_NE_FLOOR_FAR_ACTIVE = 0.2
PROD_NEAR_GATE_THRESH = 1e-3
PROD_NEAR_HANGOVER = 8


def _project_grid(sample_rate: int, fft_size: int = None) -> Tuple[int, int, int]:
    """Resolve an integrated-pipeline no-padding ``(frame, hop, fft)`` grid."""
    allowed = {16000: (256, 512), 48000: (1024,)}
    # 16 kHz default is 256/128 (8ms hop) as of 2026-08-02/03, matching
    # AEC/NR/the 4-channel pipeline's own defaults; 512 remains a
    # supported, explicit alternate.
    defaults = {16000: 256, 48000: 1024}
    if sample_rate not in allowed:
        raise ValueError(f"supported sample rates are 16000 and 48000, got {sample_rate}")
    chosen = defaults[sample_rate] if fft_size is None else int(fft_size)
    if chosen not in allowed[sample_rate]:
        choices = "/".join(str(v) for v in allowed[sample_rate])
        raise ValueError(
            f"unsupported fft_size={chosen} at {sample_rate} Hz; expected {choices}"
        )
    return chosen, chosen // 2, chosen


def _build_denoiser(sample_rate: int,
                    nr_preset: str = 'balanced',
                    frame_size: int = None,
                    frame_shift: int = None,
                    fft_size: int = None) -> MmseLsaDenoiser:
    """MMSE-LSA denoiser on a power-of-two, no-padding 50%-overlap grid.

    Routed through NR's own single source of truth: NR/config/v3_2_config.yaml's
    base params (via NR/process_audio.py's build_v3_2_base_params(), the same
    helper create_denoiser_from_config() uses) with NR/core/nr_strength.py's
    apply_strength(nr_preset) overlaid -- i.e. exactly the base-YAML -> strength
    composition create_denoiser_from_config() does, minus the mode overlay (this
    pipeline has no content-mode selection, so `mode='full'`, itself an empty
    overlay). ``strength`` is also recorded on the params dict so the
    MmseLsaDenoiser constructor can correctly disambiguate the pre-/post-16ms-grid
    provenance of alpha_noise/alpha_g/alpha_attack/alpha_decay when retiming them
    (see v3_2_mmse_lsa.py's __init__ docstring) -- omitting it would silently
    retime mild/aggressive's post-16ms-grid overlay values as if they were
    pre-16ms-grid, corrupting them at any non-16ms-hop grid.

    2026-08-03 fix: this replaces a standalone, hand-rolled NR_PRESETS dict that
    only touched {g_min, q, xi_min, alpha_g} (so mild/aggressive never affected
    alpha_noise/alpha_attack/alpha_decay -- a real functional gap, not just a
    naming coincidence) and whose 'balanced' used g_min_db=-15.0 against the real
    NR balanced's -30.0 (both are the amplitude-dB/20 convention -- an actual
    2x-in-dB floor mismatch, confirmed against both
    lib/nr/c_impl/include/mmse_lsa_types.h and lib/nr/config/v3_2_config.yaml,
    which already agree with each other). Fixing that wiring gap also surfaces
    two more stale hardcoded values this file was carrying that have no
    AEC-specific rationale anywhere in its history (unlike broadband_threshold/L
    below, which do): alpha_xi 0.88->0.92 (this pipeline predates and never
    picked up the 2026-07-10 musical-noise fix, shared across ALL strength
    presets in the real system -- this also closes the project's own
    long-flagged "AEC-YAML alpha_xi coupling untested" open question) and
    alpha_noise/alpha_d 0.95->0.7 for the `balanced` preset specifically (0.95
    was only ever the side effect of never passing alpha_d at all, never a
    validated choice -- mild/aggressive already need real alpha_noise deltas per
    the fix above, so there is no consistent way to keep 'balanced' pinned at
    0.95 without that pin also silently damping mild/aggressive's own tuning).
    This is a real, intended change to the pipeline's default numerical output --
    per this project's convention for NR numerical changes, treat this as
    needing a fresh 800-case bench pass before shipping.

    Two -- and only two -- genuinely pipeline-specific structural overlays
    remain on top of the canonical params, applied after apply_strength() so
    they win regardless of preset (neither is part of NR's own strength/mode
    axes, so there is nothing in the standalone system for them to diverge
    from):
      - broadband_threshold=0.8 (yaml=1.0, disabled): the broadband scene-reset
        path is active here; on AEC residual signals this gives faster
        adaptation after echo bursts.
      - L=94 (see _NR_L_MINIMA_WINDOW): the MCRA minima-tracking window,
        investigated 2026-08-03. This pipeline's own L was authored as
        "150 x 10ms = 1.5s" (commit de16bce, back when NR ran a literal 10ms
        hop with no retiming abstraction at all -- an independent AEC-residual
        stationarity design, unrelated to NR's own L=32/512ms tuning). NR's own
        L retiming call (lib/nr/denoisers/v3_2_mmse_lsa.py) was just changed
        (NR CHANGELOG [4.5.0]) from a generic 10ms-authored assumption to an
        UNCONDITIONAL 16ms-authored one, because NR's own L=32 is genuinely
        16ms-authored and the shared retime call has no way to tell this
        pipeline's L apart from NR's. Left as the literal 150, this pipeline's
        window would have silently grown from 1.5s to 2.4s (150 x 16ms instead
        of 150 x 10ms) the moment lib/nr picked up that fix -- with no code
        change or comment update on this side to explain why. 94 (x16ms =
        1.504s) is the 16ms-authored count that reproduces the *exact* retimed
        frame counts (188 / 94 / 141 at the 16k-8ms-hop, 16k-16ms-hop, and
        48k-10.67ms-hop grids respectively) the old literal 150 gave under the
        pre-fix 10ms-authored retiming -- i.e. this is a compensating fix that
        restores the original 1.5s behavior, not a new tuning choice.
        Caveat found in the same investigation, NOT acted on here (out of
        scope, flagged for follow-up): per NR CHANGELOG [4.5.0]'s own 824-case
        measurement, L only affects noise_psd when
        `mcra_accept_external_spp=False`; this pipeline never sets that flag
        (stays at the MmseLsaDenoiser default `True`), so today L's value --
        1.5s, 2.4s, or NR's own 512ms -- has NO effect on this pipeline's
        output either way. The "longer window improves stationarity
        estimation" rationale in the original commit only holds if/when
        `mcra_accept_external_spp=False` is also wired in.
    """
    # When fft_size is explicitly selected (notably 512 at 16 kHz), infer the
    # matching frame/hop defaults from that grid instead of first resolving the
    # sample-rate default (256 at 16 kHz).
    default_frame, default_hop, default_fft = _project_grid(sample_rate, fft_size)
    frame_size = default_frame if frame_size is None else int(frame_size)
    frame_shift = default_hop if frame_shift is None else int(frame_shift)
    fft_size = default_fft if fft_size is None else int(fft_size)
    if frame_size != fft_size:
        raise ValueError(
            f"no-padding invariant violated: frame_size={frame_size}, fft_size={fft_size}"
        )
    if frame_size != 2 * frame_shift:
        raise ValueError(
            f"50% overlap invariant violated: frame_size={frame_size}, "
            f"frame_shift={frame_shift}"
        )
    if fft_size <= 0 or fft_size & (fft_size - 1):
        raise ValueError("fft_size must be a positive power of two")
    # Enforce the project whitelist as well as the generic DSP invariants.
    _project_grid(sample_rate, fft_size)

    config = load_config(_NR_YAML_CONFIG)
    if not config:
        raise RuntimeError(
            f"failed to load NR config at {_NR_YAML_CONFIG!r} (missing file, or "
            "PyYAML not installed) -- refusing to silently fall back to "
            "build_v3_2_base_params()'s own built-in defaults, which do NOT "
            "match this project's tuned v3_2_config.yaml (e.g. g_min_db "
            "-40.0 vs the real -30.0) and would reintroduce exactly the kind "
            "of silent divergent-duplicate bug this function was fixed to avoid"
        )
    params = build_v3_2_base_params(config, sample_rate, frame_size, frame_shift, fft_size)
    params = apply_strength(params, nr_preset)
    params['strength'] = nr_preset  # retiming-provenance disambiguator; see docstring
    params['mode'] = 'full'         # no content-mode selection in this pipeline (empty overlay)

    # The only two genuinely pipeline-specific overlays -- see docstring.
    params['broadband_threshold'] = 0.8
    params['L'] = _NR_L_MINIMA_WINDOW

    return MmseLsaDenoiser(**params)


def run_nr(signal: np.ndarray, sample_rate: int,
           return_gain: bool = False, nr_preset: str = 'balanced',
           ) -> np.ndarray:
    """Run NR on the sample-rate-specific no-padding project grid."""
    denoiser = _build_denoiser(sample_rate, nr_preset)
    result = denoiser.denoise(signal, return_gain=return_gain)
    if return_gain:
        enhanced, gains = result
        print(f"  NR: {gains.shape[0]} frames, gain shape {gains.shape}")
        return enhanced, gains
    return result


def run_nr_spectrum(aec_contexts: List[AecResContext], sample_rate: int,
                    nr_preset: str = 'balanced',
                    inject_echo_psd: bool = False) -> np.ndarray:
    """NR directly on the AEC linear error spectra E(f) — no re-FFT.

    Consumes ``ctx.error_spec`` (the AEC's sqrt-Hann-windowed linear error
    spectrum) for every hop, runs MMSE-LSA in the frequency domain via
    ``denoise_spectrum``, and returns the per-frame gain G_nr(f)
    (``n_frames, n_freqs``)
    for the freq-domain RES. This is the FFT-deduplicated path: the spectra the
    time-domain run_nr would re-derive are already in the context.

    ``inject_echo_psd`` (default False = unchanged): fold the AEC residual-echo
    PSD R²(f) into the OM-LSA noise floor (a priori SNR ξ = S²/(N²+R²), the
    canonical Speex/Habets "echo-as-extra-noise" unified gain), so the single
    NR gain ALSO suppresses residual echo. R²=ctx.r2/psd_scale is already on the
    |E|² scale the denoiser's noise floor uses (β_r=1). Combine downstream with
    ``min(g_nr, g_res)`` — g_res stays load-bearing (dropping it kills echo).
    """
    if not aec_contexts:
        raise ValueError("aec_contexts must not be empty")
    spectra = np.stack([np.asarray(c.error_spec, dtype=np.complex64)
                        for c in aec_contexts])          # (n_frames, n_freqs)
    magnitude = np.abs(spectra).astype(np.float64)
    phase = np.angle(spectra).astype(np.float64)
    extra = None
    if inject_echo_psd:
        psd_scale = 32768.0 ** 2   # ctx.r2 is int16²-scaled like comfort_noise
        n_freqs = magnitude.shape[1]
        extra = np.stack([
            (np.asarray(c.r2, dtype=np.float64) / psd_scale) if c.r2 is not None
            else np.zeros(n_freqs) for c in aec_contexts])
    fft_size = 2 * (spectra.shape[1] - 1)
    denoiser = _build_denoiser(
        sample_rate, nr_preset,
        frame_size=fft_size, frame_shift=fft_size // 2, fft_size=fft_size,
    )
    _, _, gains = denoiser.denoise_spectrum(
        magnitude, phase, return_gain=True, extra_noise_psd=extra)
    print(f"  NR(freq): {gains.shape[0]} frames on E(f), gain shape {gains.shape}"
          f"{' [echo-aware ξ=S²/(N²+R²)]' if inject_echo_psd else ''}")
    return gains.astype(np.float32)


def run_res(nr_output: np.ndarray, nr_gains: np.ndarray,
            aec_contexts: List[AecResContext],
            config: AecConfig,
            use_nr: bool = True, use_res: bool = True,
            dt_relax: float = 0.0, ne_floor: float = 0.0,
            ne_gate: str = 'r2', combine: str = 'product',
            ne_floor_far_active: float = None,
            far_gate_thresh: float = 1e-4,
            near_gate_thresh: float = None,
            near_hangover_frames: int = 8) -> np.ndarray:
    """Residual-echo suppression AFTER NR, in the frequency domain.

    Reuses the linear AEC's own per-frame AEC3 SuppressionGain (``ctx.res_gain``)
    and its windowed error spectrum (``ctx.error_spec``), exposed via
    ``AecConfig.return_res_context``. The post-NR residual is a single freq
    multiply per hop::

        S(f) = E(f) · G_nr(f) · G_res(f)   (+ comfort noise on the cut bins)

    then one sqrt-Hann OLA — no standalone ResFilter, no extra FFT round-trip.
    The sqrt-Hann synthesis window matches the AEC's ``_aec3_synth_window`` exactly
    (periodic Hann, denom = block_size) for perfect reconstruction.

    Frame alignment: ``ctx[i].error_spec`` spans [(i-1)*hop, (i+1)*hop) (AEC
    analysis window ends at the current hop). In ``--pipeline-mode freq`` NR
    is also computed from ``ctx.error_spec`` so frame i = frame i. In
    ``--pipeline-mode linear`` NR is computed from the AEC time-domain output
    whose frame i covers [i*hop, (i+2)*hop), so ``nr_gains[i]`` is one hop
    AHEAD of ``ctx[i].error_spec`` — callers should pass ``nr_gains`` with a
    one-frame shift (``np.concatenate([[nr_gains[0]], nr_gains[:-1]])``) or
    use ``--pipeline-mode freq`` (the default).
    """
    bs = int(config.frame_size)
    hop = int(config.hop_size)
    fft = int(config.fft_size)
    n_freqs = fft // 2 + 1
    if bs != fft:
        raise ValueError(
            f"no-padding invariant violated: frame_size={bs}, fft_size={fft}"
        )
    if bs != 2 * hop:
        raise ValueError(
            f"50% overlap invariant violated: frame_size={bs}, hop_size={hop}"
        )
    psd_scale = 32768.0 ** 2         # int16² scale of ctx.comfort_noise (AEC3)

    idx = np.arange(bs, dtype=np.float64)
    synth_win = np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * idx / float(bs)))).astype(np.float32)
    ola = np.zeros(bs, dtype=np.float32)
    rng = np.random.RandomState(0)   # deterministic comfort noise

    n_frames = min(len(aec_contexts), nr_gains.shape[0])
    output = np.zeros(len(nr_output), dtype=np.float32)
    near_hang = 0                    # near-activity hangover counter (gated floor)

    for i in range(n_frames):
        ctx = aec_contexts[i]
        if ctx.error_spec is None or ctx.res_gain is None:
            n_frames = i        # AEC built without the freq seam → stop here
            break
        start = i * hop
        end = start + hop
        if end > len(output):
            n_frames = i
            break

        g_nr = nr_gains[i].astype(np.float32) if use_nr else 1.0
        g_res = ctx.res_gain.astype(np.float32) if use_res else 1.0
        # Save AEC-only gain before combining with G_nr, for CNG noise level.
        # CNG must reflect AEC suppression only — using g_total would re-inject
        # noise into NR-suppressed bins (BAK ceiling).
        g_aec = g_res if isinstance(g_res, np.ndarray) else np.full(n_freqs, g_res, dtype=np.float32)
        if use_nr and use_res and combine == 'min':
            # A_min_pl: per-bin min recovers the AEC3 echo gain that v0 discarded
            # — near-end bins (g_res≈1) keep g_nr; echo bins (g_res<g_nr) cut the
            # leak — WITHOUT the product's double-cut on double-talk near-end.
            g_total = np.minimum(g_nr, g_res).astype(np.float32)
        else:
            g_total = (g_nr * g_res).astype(np.float32) if (use_nr or use_res) \
                else np.ones(n_freqs, dtype=np.float32)
        # Near-end-aware relax: in double-talk (ctx.dt_indicator high), the
        # G_nr·G_res product over-cuts near speech. Blend the total gain toward
        # 1.0 by dt_relax·dt_indicator so DT near-end is preserved (FS, where
        # dt_indicator≈0, is untouched). dt_relax=0 → off (plain product).
        if dt_relax > 0.0:
            w = float(dt_relax) * float(ctx.dt_indicator)   # 0..(dt_relax·0.8)
            if w > 0.0:
                g_total = (1.0 - w) * g_total + w * 1.0
        # Lightweight near-end preservation floor (per-bin, echo-aware): lift the
        # gain toward 1.0 ONLY where there is little residual echo. echo_frac =
        # R²/|E|² ≈ 0 in clean near-end (→ full lift, preserve speech) and ≈ 1 in
        # echo-dominated bins (→ no lift, stay suppressed). Unlike dt_relax this
        # keys off per-bin echo, not a frame scalar, so it never lifts in far-end.
        # Far-activity-gated floor strength (default off → scalar ne_floor):
        # keep the high floor when the far end is SILENT (pure near-end — the only
        # place lowering it just damages near speech, e.g. NE) and drop to
        # ne_floor_far_active when the far end is ACTIVE (FS/DT — there the floor
        # was over-protecting noise; less echo to guard now that the linear filter
        # improved). ctx.far_power separates the two cleanly (NE≈0, FS/DT bursts).
        nf_eff = ne_floor
        if ne_floor_far_active is not None:
            fp = float(ctx.far_power) if ctx.far_power is not None else 0.0
            far_active = fp > far_gate_thresh
            if near_gate_thresh is not None:
                # Near-activity gate: protect the high floor ONLY when the far end
                # is silent AND there is genuine near-end SPEECH (true NE). Far-
                # silent + near-silent (FS/DT noise gaps) has no speech to damage →
                # drop the floor and clean the background. near_energy = |E(f)|²
                # (≈ near+noise when far-silent); hangover protects speech offsets.
                near_energy = float(np.mean(np.abs(ctx.error_spec) ** 2))
                if near_energy > near_gate_thresh:
                    near_hang = int(near_hangover_frames)
                near_active = near_hang > 0
                if near_hang > 0:
                    near_hang -= 1
                protect = (not far_active) and near_active
            else:
                protect = not far_active
            nf_eff = ne_floor if protect else ne_floor_far_active
        if nf_eff > 0.0 and ctx.r2 is not None:
            r2_nr = np.asarray(ctx.r2, dtype=np.float32) / psd_scale
            e2 = np.abs(ctx.error_spec).astype(np.float32) ** 2
            echo_frac = np.clip(r2_nr / (e2 + 1e-12), 0.0, 1.0)
            no_echo_r2 = 1.0 - echo_frac
            g_res = ctx.res_gain.astype(np.float32)  # AEC3 gain ≈1 no-echo, low=echo
            if ne_gate == 'r2':
                no_echo = no_echo_r2
            elif ne_gate == 'resgain':
                no_echo = g_res
            elif ne_gate == 'both_sharp':  # res_gain² → lift drops faster in echo bins
                no_echo = (g_res ** 2) * no_echo_r2
            else:  # 'both' — product gate: lift only where BOTH say no echo
                no_echo = g_res * no_echo_r2
            lift = float(nf_eff) * no_echo
            g_total = (1.0 - lift) * g_total + lift * 1.0
        spec = ctx.error_spec.astype(np.complex64) * g_total

        # Comfort noise fills the suppressed bins (port of AEC3 RES CNG; the
        # AEC3 sin-LUT/LCG source is approximated with deterministic Gaussian
        # noise at the same per-bin level — this is a separated-pipeline
        # baseline, not bit-identical to the internal RES).
        if config.enable_cng and ctx.comfort_noise is not None:
            n_amp = np.sqrt(np.maximum(ctx.comfort_noise / psd_scale, 0.0)).astype(np.float32)
            noise_gain = np.sqrt(np.maximum(1.0 - g_aec ** 2, 0.0)).astype(np.float32)
            cn = np.zeros(n_freqs, dtype=np.complex64)
            cn[1:-1] = (n_amp[1:-1]
                        * (rng.randn(n_freqs - 2) + 1j * rng.randn(n_freqs - 2)))
            spec = spec + noise_gain * cn

        e_full = np.fft.irfft(spec, n=fft).astype(np.float32)
        ola += e_full[:bs] * synth_win
        output[start:end] = ola[:hop]
        ola[:-hop] = ola[hop:]
        ola[-hop:] = 0.0

    # Copy tail (frames beyond AEC/NR context)
    tail_start = n_frames * hop
    if tail_start < len(nr_output):
        output[tail_start:] = nr_output[tail_start:]

    print(f"  RES: {n_frames} frames (freq-domain E·G_nr·G_res)")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AEC + NR + RES Pipeline (freq A_min_pl)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""AEC(linear) -> noise-only NR(E) -> RES   (g_total = min(G_nr, G_res)).
AEC and NR are each chosen by their own preset; the CLI exposes only
presets + switches.

Presets:
  --aec-preset  mild | balanced | aggressive   (echo-vs-near strength)
  --nr-preset   mild   | balanced | aggressive   (noise-suppression strength)
Switches:
  --aec-only    run AEC only, skip NR/RES
""")
    parser.add_argument('--mic', required=True, help='Microphone input WAV')
    parser.add_argument('--ref', required=True, help='Reference/loudspeaker WAV')
    parser.add_argument('--output', required=True, help='Output WAV')
    parser.add_argument(
        '--fft-size', type=int, choices=[256, 512, 1024], default=None,
        help=('No-padding FFT/frame size. Defaults: 256 at 16 kHz and 1024 at '
              '48 kHz; 16 kHz also supports 512. Hop is always FFT/2.'))
    parser.add_argument('--aec-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='AEC preset (default: balanced)')
    parser.add_argument('--nr-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='NR preset (default: balanced)')
    parser.add_argument('--aec-only', action='store_true',
                        help='Run AEC only, skip NR/RES')
    parser.add_argument('--legacy-amin', action='store_true',
                        help='Restore the prior min-only A_min_pl (no R² injection, '
                             'scalar near-end floor) — disables the 2026-06-23 re-tune')
    args = parser.parse_args()

    # Load audio
    mic_signal, sr_mic = sf.read(args.mic, dtype='float32')
    ref_signal, sr_ref = sf.read(args.ref, dtype='float32')

    if sr_mic != sr_ref:
        print(f"Error: sample rate mismatch ({sr_mic} vs {sr_ref})", file=sys.stderr)
        sys.exit(1)

    sample_rate = sr_mic
    try:
        frame_size, hop_size, fft_size = _project_grid(sample_rate, args.fft_size)
    except ValueError as exc:
        parser.error(str(exc))
    duration = len(mic_signal) / sample_rate

    preset_map = {
        'mild': AecPreset.MILD, 'balanced': AecPreset.BALANCED,
        'aggressive': AecPreset.AGGRESSIVE,
    }
    preset = preset_map[args.aec_preset]

    print("AEC + NR + RES Pipeline")
    print("=======================")
    print(f"Input:    {args.mic} ({len(mic_signal)} samples, {duration:.2f}s)")
    print(f"Ref:      {args.ref}")
    print(f"Output:   {args.output}")
    print(f"Rate:     {sample_rate} Hz")
    print(f"Grid:     frame={frame_size}, hop={hop_size}, fft={fft_size} (no padding)")
    print(f"AEC:      preset={args.aec_preset}")
    if not args.aec_only:
        print(f"NR:       preset={args.nr_preset}")
    else:
        print("NR/RES:   disabled (--aec-only)")
    print()

    # AEC config: production filter (PBFDKF); the preset sets the echo-vs-near
    # min-gain floor (the only inter-preset difference).
    aec_config = AecConfig.from_preset(
        preset,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        mode=AecMode.PBFDKF,
        mu=0.3,
        enable_res=True,
    )

    # freq A_min_pl pipeline (production): AEC(linear) -> noise-only NR(E) ->
    # g_total = min(G_nr, G_res) + per-bin echo-gated near-end floor. NR handles
    # noise; the AEC's own near-end-aware AEC3 echo gain G_res handles echo; the
    # per-bin min keeps both without the product's double-talk double-cut.
    print("Stage 1: AEC (linear, no RES)...")
    aec_output, contexts = run_aec_linear(
        mic_signal, ref_signal, aec_config,
        standalone_time_output=args.aec_only,
    )

    if args.aec_only:
        final_output = aec_output
    else:
        _legacy = getattr(args, 'legacy_amin', False)
        print("Stage 2: NR on E(f)"
              + ("" if _legacy else " [echo-aware ξ=S²/(N²+R²)]") + "...")
        nr_gains = run_nr_spectrum(contexts, sample_rate, nr_preset=args.nr_preset,
                                   inject_echo_psd=(not _legacy) and PROD_INJECT_ECHO_PSD)
        print("Stage 3: g_total=min(G_nr,G_res)"
              + (" + near-end floor" if _legacy else " + far-gated near-end floor") + "...")
        # Pass aec_output (zero-padded to mic length) as the tail source so
        # samples beyond the last full AEC frame are not silenced (mic>ref case).
        _tail_src = np.zeros(len(mic_signal), dtype=np.float32)
        _tail_src[:len(aec_output)] = aec_output
        final_output = run_res(
            _tail_src, nr_gains, contexts, aec_config,
            use_res=True, combine='min', ne_floor=PROD_NE_FLOOR, ne_gate='both',
            ne_floor_far_active=None if _legacy else PROD_NE_FLOOR_FAR_ACTIVE,
            near_gate_thresh=None if _legacy else PROD_NEAR_GATE_THRESH,
            near_hangover_frames=retime_frame_count(
                PROD_NEAR_HANGOVER, sample_rate, hop_size))

    # Save
    sf.write(args.output, final_output, sample_rate)
    print(f"\nDone! Output: {args.output} ({len(final_output)} samples)")


if __name__ == '__main__':
    main()
