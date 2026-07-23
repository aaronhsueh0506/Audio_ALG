# AINR dataset_gen

Model-independent generator for (noisy, clean) speech-enhancement training
pairs, living at `ainr/dataset_gen/`. Extracted from the RNNoise-ERB
dataset-generation chain so one generator can serve multiple models:
RNNoise-ERB/GTCRN at 16 kHz and DeepFilterNet2 at 48 kHz.

## Purpose

This package does exactly one job: take a raw speech / noise / RIR corpus
(e.g. DNS Challenge 4) and produce augmented `(noisy, clean)` WAV pairs —
biquad EQ, RIR/reverb, SNR mixing, gain randomization, optional bandwidth
simulation for upsampled lower-rate sources, and clipping distortion. It does
**not** know anything about any particular model's architecture, feature
extraction, or training loop. Model
training scripts (e.g. RNNoise-ERB's `train.py`) live in their own model repo
and consume the output of this package.

### Config scope

`config.example.ini` contains generation settings only: working sample rate,
source paths, segment length, generation/resume controls, mixing, RIR, noise,
and waveform augmentations. Model architecture, FFT/ERB feature parameters,
optimizer, training schedule, and loss settings stay in the RNNoise-ERB,
GTCRN, or DeepFilterNet2 model config that consumes the generated pairs.

The optional `[gen] pass_size` limits how many shuffled speech files form one
generation pass; it is not a training epoch. Omitting it or setting it to `0`
uses all speech files.

## Design: one selected sample rate per run

Each invocation generates exactly one dataset at the requested working rate.
Use config.ini's `[signal] sr` or override it with `--sample-rate`:

```text
16000 Hz → RNNoise-ERB / GTCRN
48000 Hz → DeepFilterNet2
```

Run the command separately with different output directories when both are
needed. These are independent augmentation runs, not synchronized copies.
`resample_dataset.py` remains available when a downsampled copy of an existing
dataset is explicitly preferred.

### Upsampled lower-rate source simulation

Some deployments run a 16 or 48 kHz enhancement algorithm on audio that was
captured at a lower sample rate and upsampled only to satisfy the algorithm's
input contract. The generator models that path with discrete, realistic source
rates:

```ini
[augmentation]
p_resample = 0.2
source_sr_values = 8000, 12000, 16000, 22050, 24000, 32000, 44100
```

For each selected pair, both `noisy` and `target` follow the identical
`algorithm_sr -> source_sr -> algorithm_sr` resampling path. This keeps the
task as denoising of upsampled audio; unlike DeepFilterNet bandwidth-extension
training, the target does not retain frequencies that the simulated source
could not capture.

Rates greater than or equal to the selected algorithm rate are removed
automatically. A 48 kHz run can use the complete list, while a 16 kHz run uses
only 8 and 12 kHz. Samples outside `p_resample` remain at the native algorithm
rate. The default probability is 20% and should be adjusted to the expected
deployment mix.

### RIR / DRR contract

The RIR path follows DeepFilterNet's `RandReverbSim` conventions: full and
late-suppressed target RIRs are independently L2-normalized, and SNR is
measured against the clean target while the mixture contains the full-RIR
speech. `drr = 0.3` is a linear dry/reverberant blend factor in `[0, 1]`, not
a dB value. The dry target is delayed to the trimmed RIR's direct-path peak so
target and reverberant mixture do not acquire a pre-delay mismatch.

### SNR and clean/noise edge cases

Mixed samples uniformly draw from DeepFilterNet's discrete SNR set:
`-5, 0, 5, 10, 20, 40 dB`. DeepFilterNet3 puts `-100` in that list as a
sentinel for a noise-only sample; this generator represents the edge case
explicitly instead:

```ini
[mixing]
snr_values = -5, 0, 5, 10, 20, 40

[noise]
noise_only_p = 0.05
speech_only_p = 0.05
```

The two special modes are mutually exclusive. With the defaults, 5% of pairs
are pure noise (`target = silence`), 5% are exact speech identity pairs
(`noisy = target`), and the remaining 90% are noisy mixtures. Speech-only
pairs deliberately skip input-only clipping so they stay exact identity
examples. Keeping a small speech-only share teaches the model not to alter an
already-clean input; use a genuinely clean speech corpus, since any noise in a
speech source file is necessarily treated as desired target content.

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
- `gen_dataset.py --sample-rate` overrides config.ini's `[signal] sr`.
- Omitting the CLI flag genuinely uses the config value.
- `config.example.ini` uses 48000 as its example; change it to 16000 or pass
  `--sample-rate 16000` for RNNoise-ERB/GTCRN.

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
| `packed_dataset.py` | Shared mmap-capable loader for packed `(N, 2, T)` tensors |
| `resample_dataset.py` | Optional standalone resample of an existing dataset |
| `config.example.ini` | Example config (copy to `config.ini` and edit `[paths]` for your corpus) |
| `tests/` | Hours, worker seed, RIR delay and resample-length regression tests |

`train.py` is **not** part of this package — it stays in the model repo
(RNNoise-ERB) since it's model-specific (architecture, loss, optimizer). It
currently reads the same `config.ini` format and consumes WAV pairs / packed
`.pt` files produced here.

## CLI usage

### 1. Generate the selected model dataset

```bash
cp config.example.ini config.ini
# edit [paths] speech_dir / noise_dir / rir_dir to point at your corpus (e.g. DNS4)

# RNNoise-ERB / GTCRN
python3 gen_dataset.py --config config.ini --output data_16k --hours 25 \
    --sample-rate 16000

# DeepFilterNet2
python3 gen_dataset.py --config config.ini --output data_48k --hours 25 \
    --sample-rate 48000
```

Each command creates:
```
data_16k/ or data_48k/
  meta.json
  pairs/000000.wav, 000001.wav, ...
```

Each WAV is 2-channel: ch0=noisy, ch1=clean.

Useful flags: `--resume` (continue an interrupted batch), `--start-idx` /
`[gen] start_idx` (extend an existing dataset without overwriting or
re-sampling old data — `effective_seed = seed + start_idx`), and `--seed`.
The default `[gen] seed = -1` requests a fresh OS-generated seed every run;
the actual seed is printed and stored in `meta.json`. Pass a non-negative
seed such as `--seed 42` for reproducibility. DataLoader workers are seeded
independently from that run seed, so workers do not repeat the same Python or
NumPy random stream.

`--hours` is converted directly to a whole-segment count:
`ceil(hours × 3600 / segment_sec)`. It therefore exceeds the requested
duration by less than one segment. It is not rounded to a complete
`DNS4Dataset` epoch; when necessary, the final dataset pass is partial.

### 2. Optional: resample an existing dataset

```bash
python3 resample_dataset.py \
    --input data_48k --output data_16k --target-sr 16000 --workers 4
```

This produces an independent copy of the dataset at the target
rate, preserving directory structure, filenames, and the 2-channel
noisy/clean pair layout, plus an updated `meta.json` (`sr` set to the target
rate and records the original rate as `source_sr`.

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

### 3. Pack the generated datasets

```bash
python3 pack_dataset.py \
    --input data_16k/pairs --output data_16k/packed.pt --dtype float16
python3 pack_dataset.py \
    --input data_48k/pairs --output data_48k/packed.pt --dtype float16
```

Use `data_16k/packed.pt` for RNNoise-ERB and `data_48k/packed.pt` for
DeepFilterNet2.

### Tests

```bash
make test
```

## Requirements

```
torch>=1.13
torchaudio>=0.13
numpy
tqdm
```

## Consumers

`../RNNoise-ERB/train.py` imports the shared `PackedDataset` from this package;
DeepFilterNet2 consumes the corresponding 48 kHz packed copy. Model
directories do not keep private copies of the augmentation engine.
