# AINR dataset_gen

Model-independent generator for (noisy, clean) speech-enhancement training
pairs, living at `ainr/dataset_gen/`. Extracted from the RNNoise-ERB
dataset-generation chain so the same augmented corpus can be reused across
multiple models (RNNoise-ERB today, GTCRN next, others later) without
re-running the whole augmentation pipeline per model.

## Purpose

This package does exactly one job: take a raw speech / noise / RIR corpus
(e.g. DNS Challenge 4) and produce augmented `(noisy, clean)` WAV pairs —
biquad EQ, RIR/reverb, SNR mixing, gain randomization, optional bandwidth
limiting and clipping distortion. It does **not** know anything about any
particular model's architecture, feature extraction, or training loop. Model
training scripts (e.g. RNNoise-ERB's `train.py`) live in their own model repo
and consume the output of this package.

## Design: 48k master + model-side resample

**Decision:** generate the augmented dataset **once**, at a canonical high
sample rate (48 kHz "master"), independent of any model's target rate. Each
model then resamples its own copy at pack-time (`resample_dataset.py`) to
whatever rate it actually trains at (e.g. 16 kHz for RNNoise-ERB). This
makes dataset generation model-independent and reusable: regenerating the
whole augmented corpus (RIR convolution, SNR mixing, etc. — the expensive
part) is no longer needed every time a new model with a different input rate
comes along; only a cheap resample pass is.

### Where the working rate actually comes from (honest finding)

Before this extraction, there was **no explicit "generate at X Hz" contract**
— the working rate was an implicit side effect of `config.ini`'s
`[signal] sr` key, read once in `DNS4Dataset.__init__` (`dataset.py`):

```python
self.sr = cfg.getint('signal', 'sr')
```

`self.sr` then drives *everything*: segment length in samples
(`segment_samples = segment_sec * sr`), the target rate every speech / noise
/ RIR file is resampled to on load (`_load_and_crop`, `_load_noise`,
`_load_rir` all call `torchaudio.functional.resample(audio, orig_sr, self.sr)`
whenever a source file's native rate differs), STFT parameters, and ERB band
edges. The **source corpus itself has no fixed rate** — DNS4 speech/noise/RIR
files are typically 48 kHz already but are not required to be; whatever rate
they're at gets resampled to `self.sr` on load, so `self.sr` is the *sole*
place the working rate is pinned. The pre-extraction default in
`config.ini` was `sr = 16000` — i.e. the generator was implicitly
RNNoise-ERB-shaped (16 kHz) even though nothing about the augmentation logic
requires that.

This extraction makes that contract **explicit** instead of implicit:
- `gen_dataset.py` gained a `--sample-rate` CLI flag (default **48000**) that
  overrides `config.ini`'s `[signal] sr` at runtime, so the canonical rate is
  a first-class, visible generation parameter instead of a config file value
  you have to go read `dataset.py` to understand the significance of.
- `config.example.ini`'s `[signal] sr` default was changed to `48000` to
  match (it's overridden by `--sample-rate` regardless, but should not lie
  about the intended default when read standalone).

One additional coupling was found and removed: `DNS4Dataset._compute_erb_bands`
used to lazily `from train import compute_hybrid_bands, compute_erb_bands` —
a hard dependency on the RNNoise-ERB model repo's `train.py`, pulled in
**unconditionally** at `DNS4Dataset.__init__` time (even in
`return_raw=True` / WAV-pair-generation mode, where ERB bands are never
actually used). Those two functions plus their `erb_rate`/`erb_inv` helpers
are pure NumPy math with no model-specific dependencies, so they were
inlined directly into `dataset.py` here. This is the only code change beyond
path/CLI wiring — everything else in `dataset.py` was already fully
sample-rate-parameterized (no hardcoded 16000/48000 anywhere in the
augmentation logic itself).

## Files

| File | Role |
|---|---|
| `gen_dataset.py` | CLI entry point — offline pre-generation of `(noisy, clean)` WAV pairs |
| `dataset.py` | `DNS4Dataset` — the augmentation engine (biquad filters, RIR/RT60, SNR mixing, bandwidth limiting, clipping) |
| `pack_dataset.py` | Packs a WAV-pair directory into a single `.pt` tensor file (removes per-file I/O overhead for small/medium datasets) |
| `resample_dataset.py` | **New.** Model-side pack-stage resample: 48k master → any target rate, once, offline |
| `config.example.ini` | Example config (copy to `config.ini` and edit `[paths]` for your corpus) |

`train.py` is **not** part of this package — it stays in the model repo
(RNNoise-ERB) since it's model-specific (architecture, loss, optimizer). It
currently reads the same `config.ini` format and consumes WAV pairs / packed
`.pt` files produced here.

## CLI usage

### 1. Generate the 48k master dataset

```bash
cp config.example.ini config.ini
# edit [paths] speech_dir / noise_dir / rir_dir to point at your corpus (e.g. DNS4)

python3 gen_dataset.py --config config.ini --output data/ --hours 25
python3 gen_dataset.py --config config.ini --output data/ --hours 50 --workers 4
python3 gen_dataset.py --config config.ini --output data/ --hours 25 --sample-rate 48000
```

Output layout:
```
data/
  pairs/000000.wav, 000001.wav, ...   # 2-channel WAV: ch0=noisy, ch1=clean
  meta.json                            # n_samples, sr, segment_sec, ...
```

Useful flags: `--resume` (continue an interrupted batch), `--start-idx` /
`[gen] start_idx` (extend an existing dataset without overwriting or
re-sampling old data — `effective_seed = --seed + start_idx`), `--seed`
(default 42, `-1` to disable).

### 2. Resample for a specific model (pack-stage, once)

```bash
python3 resample_dataset.py --input data/ --output data_16k/ --target-sr 16000
python3 resample_dataset.py --input data/ --output data_16k/ --target-sr 16000 --workers 4
```

This produces a byte-for-byte independent copy of the dataset at the target
rate, preserving directory structure, filenames, and the 2-channel
noisy/clean pair layout, plus an updated `meta.json` (`sr` set to the target
rate, original rate recorded as `source_sr`). Point the model's training
loader at `data_16k/` instead of `data/` — **zero resample cost at train
time**, for every epoch, forever.

### Resample guidance (anti-aliasing parameters)

`resample_dataset.py` uses `torchaudio.functional.resample` with the
Kaiser-windowed sinc-interpolation kernel (`resampling_method
= "sinc_interp_kaiser"`), which is explicitly anti-aliased — a lowpass is
applied before decimation so no spectral energy folds back below Nyquist.
Default preset (`--quality best`, torchaudio's own "kaiser_best" tutorial
preset):

| Parameter | Value | Meaning |
|---|---|---|
| `lowpass_filter_width` | 64 | Sinc kernel half-width in zero-crossings — wider = sharper transition band, more compute |
| `rolloff` | 0.9475937167399596 | Lowpass cutoff as a fraction of Nyquist — lower leaves more anti-aliasing headroom at the cost of some top-octave attenuation |
| `beta` | 14.769656459379492 | Kaiser window shape parameter — controls stopband attenuation vs. transition sharpness tradeoff |

A cheaper preset (`--quality fast`, torchaudio's "kaiser_fast":
`lowpass_filter_width=16, rolloff=0.85, beta=8.555504641634386`) is available
for quick smoke tests only. Since resampling happens once offline (not per
training epoch), the extra compute of `best` over `fast` is irrelevant —
quality was prioritized deliberately.

**Clipping guard:** Kaiser-sinc resampling can slightly overshoot the
original peak amplitude (Gibbs-like ringing near sharp transients or
near-full-scale content). If a resampled pair's peak exceeds `0.999`,
`resample_dataset.py` scales the whole `(noisy, clean)` pair down together
(preserving their relative level / SNR) before writing, rather than letting
`torchaudio.save`'s int16 write clip silently. The final run summary reports
how many files needed this and the overall peak level observed.

### 3. (Optional) Pack into a single tensor file

```bash
python3 pack_dataset.py --input data_16k/pairs/ --output data_16k/packed.pt
python3 pack_dataset.py --input data_16k/pairs/ --output data_16k/packed.pt --dtype float16
```

## Requirements

```
torch>=1.13
torchaudio>=0.13
numpy
tqdm
```

## Status / later arc

The sibling model repo `../RNNoise-ERB/` still has its own copies of
`gen_dataset.py`, `dataset.py`, `pack_dataset.py`, `config.ini` — this
extraction copied them out without touching or deleting the originals. A
later arc will switch `../RNNoise-ERB/train.py` (and any operator docs /
scripts) to point at this package instead of its local copies; that
switchover is out of scope here.
