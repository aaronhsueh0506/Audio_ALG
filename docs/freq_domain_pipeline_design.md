# Frequency-domain pipeline: AEC(linear) → NR → AEC(residual)
Design review + FFT-deduplication plan + NN integration seams · 2026-06-07 (AEC v3.22.5)

## Goal (user request, item 12/13)
1. Architecture = **AEC(linear) + NR + AEC(residual)** as three separable stages, so an NN can later
   replace any one of them (NN-residual, NN-NR, or NN-residual+NR) by swapping a single freq-domain block.
2. Reduce **fft/ifft duplication**: review whether all three stages can run in one frequency domain, with
   windowing+FFT done once at the front and IFFT+OLA once at the back.
3. Confirm whether changing **hop / sample-rate / fft** breaks the algorithms (item 13).

## Verdict — FEASIBLE, and most pieces already exist
The "separated" pipeline (`pipelines/aec_nr_pipeline.py --pipeline-mode linear`) already chains
`AEC(linear) → NR → RES` and already aligns all three at **hop=160 (10 ms)** ("frame i of AEC = frame i
of NR", `run_res` docstring). The three freq-domain seams needed for a single-transform pipeline already
exist; the current code merely wires them through **redundant time-domain hops** and re-FFTs at each stage.

| Stage | Freq-domain seam that already exists |
|---|---|
| AEC(linear) | `AecConfig.return_res_context=True` → per-frame `AecResContext` carrying `near_spec` (error spectrum E), `far_spec`, `echo_spec` (Ŷ), Kalman/ERLE state — `run_aec_linear()` |
| NR | every denoiser has **`denoise_spectrum()`** (base_denoiser.py:36) — a freq-in/freq-out entry separate from the STFT-wrapping `denoise()` |
| AEC(residual) | `run_res()` already consumes the freq contexts + NR gains: `corrected_echo = ctx.echo_spec * nr_gain`, then `res.process(echo_spec=…, far_spec=…, near_spec=…)` |

Both AEC and NR already use **fft=512, frame=512, sqrt-Hann periodic (COLA), 16 kHz** — identical analysis
grids. The only divergence is NR's *standalone* hop (256); the *integrated* pipeline already overrides it
to 160.

## Current data flow (the redundancy)
```
mic,ref ─► AEC.process ──(ifft)──► aec_out[time] ──(fft)──► NR ──(ifft)──► nr_out[time] ──(fft)──► RES ──(ifft+OLA)──► out
              │ exposes E(f),Ŷ(f),X(f) via ctx        ▲ re-derives E(f)        ▲ re-derives spectrum
              └───────────────── freq spectra thrown away then recomputed ─────┘
```
Per hop this pays **2 redundant FFTs + 2 redundant IFFTs** (AEC→NR and NR→RES round-trips). The spectra
NR and RES recompute are already sitting in `AecResContext`.

## Target data flow (single transform)
```
mic,ref ─►[window+rFFT once]─► X(f),Y(f) ─► AEC-linear ─► E(f) ─► NR.denoise_spectrum ─► G_nr·E(f) ─► AEC-residual ─► S(f) ─►[irFFT+OLA once]─► out
                                              (PBFDKF, all freq)       (freq gain)              (freq, ENR/EMR)
```
- AEC-linear already computes E(f)=Y−ŴX internally; **stop ifft-ing it** — hand E(f) to NR.
- NR consumes E(f) via `denoise_spectrum()` → returns gain G_nr(f) (and/or enhanced spectrum). Already exists.
- AEC-residual consumes `G_nr·Ŷ`-corrected echo + E(f) (post-NR) and applies the ENR/EMR `GainToNoAudibleEcho`
  → final S(f). `run_res` already does the gain math in freq; only its `error_hop` time input changes to E(f).
- One sqrt-Hann analysis FFT and one IFFT+OLA for the whole chain.

**Net win:** 1 analysis FFT + 1 synthesis IFFT per hop instead of 3 FFT + 3 IFFT — a ~3× reduction in
transform cost, and the spectra are passed by reference (no re-derivation error).

## Item 13 — hop / SR / FFT sensitivity (what may NOT change)
Measured from the algorithm-structural code (filters.py, modules/delay/, freq_utils.py):

| Knob | Sensitivity | Why |
|---|---|---|
| **fft_size** | **Moderate** — changeable with recalibration | `freq_utils.hz_to_bin/bin_to_hz` derive fft from `n_bins`; filter `n_freqs=fft//2+1` is computed. BUT AEC3 echo-model / noise-gate constants are int16²-PSD-**sum** scale calibrated at fft=512 (31.25 Hz/bin) → must be re-derived if fft changes. |
| **hop_size** | **High** | PBFDKF partition→time mapping, `ms_to_hops`, clock-drift cadence all key off hop; the 10 ms hop is structurally assumed. The integrated pipeline already standardizes **all three stages on hop=160** — keep it. |
| **sample_rate** | **Highest** | Delay subsystem hardcodes `_K_NUM_BLOCKS_PER_SECOND = 250 = 16000/64` and 4 kHz decimation (16000/4); PSD constants calibrated at 16 k. Changing SR needs delay block-rate + decimation + PSD recalibration. |

**Design rule:** fix the shared grid at **16 kHz / fft=512 / hop=160 / sqrt-Hann**. Bring NR onto hop=160
(it was A/B-validated at 160 = its "OLD" setting). Do **not** make AEC follow NR's 256 — SR/hop are most
baked on the AEC side. SR/fft changes are out-of-scope refactors (would require delay re-derivation + PSD
recalibration), not free parameters.

## NN integration seams (item 10 — interface, doc-only)
The single-transform freq pipeline makes each stage a swappable freq-in/freq-out block. An NN replaces a
stage by matching that block's I/O contract on the shared 257-bin / hop-160 grid:

| Swap | Replaces | Freq-domain contract (in → out) |
|---|---|---|
| **NN-residual** | AEC(residual) `ResFilter` | in: E(f) post-NR + `AecResContext`(Ŷ(f),X(f),ERLE,DT) ; out: S(f) or gain G_res(f)∈[0,1] |
| **NN-NR** | `denoise_spectrum()` | in: E(f) (+ optional noise estimate) ; out: enhanced spectrum or gain G_nr(f)∈[0,1] |
| **NN-residual+NR** | NR + RES jointly | in: E(f) + `AecResContext` ; out: S(f) — one network does denoise+residual in a single mask |

All three consume the **same `AecResContext`** already exposed today (echo_spec, far_spec, near_spec,
filter_converged, erle_factor, dt_indicator, over_sub, divergence). That context IS the NN feature vector.
The DSP stages stay as the default/fallback; an NN is enabled by routing the freq block through the model
instead of the DSP function — no change to the front-end transform or back-end OLA.

## Implementation status — IMPLEMENTED (2026-06-07, AEC v3.22.5)
The separated RES-after-NR topology is rebuilt on the AEC3 freq seam (the standalone `ResFilter` retired in
v3.21 is **not** revived). What shipped:

- **AEC seam (`AecResContext`)**: 3 additive fields — `error_spec` (windowed linear E(f)), `res_gain`
  (the per-frame AEC3 `SuppressionGain` G_res), `comfort_noise` (N²). `_aec3_post` stashes them when
  `return_res_context=True`, and now returns the **true linear** residual when `enable_res=False` (it used to
  apply suppression even in context mode → the old separated pipeline was silently double-suppressing). The
  default production path (`enable_res=True`, no context) is **byte-equal** — verified peak|Δ|=0.0.
- **Pipeline (`aec_nr_pipeline.py`)**: `run_res` rewritten as one freq multiply `S(f)=E(f)·G_nr(f)·G_res(f)`
  (+CNG) → single sqrt-Hann OLA, reusing the AEC's tuned gain (no ResFilter). Two new modes:
  - `linear` — NR on the time-domain linear output, then freq RES (reference for A/B).
  - `freq` (**default**) — `run_nr_spectrum` feeds **E(f)** straight to `NR.denoise_spectrum` (no re-FFT),
    then freq RES. This is the single-transform chain: AEC analysis FFT (internal) → NR + RES in freq →
    one IFFT+OLA. No inter-stage round-trips.
- **A/B (3-case smoke)**: `freq ≈ linear` (RMS within 0.1 dB, waveform corr 0.985–0.994 — pure transform
  rounding, as predicted). Separated vs classic: echo −0.6 dB deeper (farend), near-end preservation
  identical (0.07 dB), doubletalk corr 0.991. 800-case AECMOS A/B: `pipelines/rebench_sep_vs_classic.py`.

### Resolved risks (were open in the original design)
- NR's `denoise_spectrum` is a **batch freq-in API** (magnitude/phase arrays) that does no windowing itself,
  so feeding the AEC's sqrt-Hann `|E(f)|` is self-consistent: sqrt-Hann analysis (AEC) → freq gains →
  sqrt-Hann synth OLA (run_res) = COLA. NR's internal Hann is only used by its time-domain `denoise()`.
- NR already runs at frame_shift=160 in the integrated pipeline, so the MCRA framing concern is moot.

## Risks / open items
- NR's `denoise_spectrum` must accept the **AEC error spectrum** E(f) as its "noisy" input (today the
  integrated NR re-ffts aec_out, which equals OLA(E); passing E(f) directly removes the OLA round-trip but
  must use the same window convention — verify the sqrt-Hann analysis matches on both sides).
- NR noise tracking (MCRA/IMCRA minima) assumes its own framing; at hop=160 the minima windows must be
  re-tuned in *frames* (they were tuned at 256). Low risk (config), but bench it.
- This is a **wiring/efficiency refactor**, not an algorithm change — the DSP outputs should match the
  current `linear` pipeline within transform-rounding. The NN swaps are a *later* arc; this design only
  guarantees the seams exist and the transform is shared.
