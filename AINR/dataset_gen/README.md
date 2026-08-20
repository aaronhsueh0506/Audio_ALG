# AINR dataset_gen

Model-independent generator for (noisy, clean) speech-enhancement training
pairs, living at `AINR/dataset_gen/`. Extracted from the RNNoise-ERB
dataset-generation chain so one generator can serve multiple models:
RNNoise-ERB/GTCRN at 16 kHz and DeepFilterNet2 at 48 kHz.

## Purpose

This package does exactly one job: take a raw speech / noise / RIR corpus
(e.g. DNS Challenge 4) and produce augmented `(noisy, clean)` WAV pairs —
biquad EQ, RIR/reverb, SNR mixing, optional bandwidth simulation for
upsampled lower-rate sources, and clipping distortion. It does
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

## Design: generate natively per rate; downsample-at-pack-time is a convenience path, not the standard one

Each `gen_dataset.py` invocation generates exactly one dataset at the
requested working rate (config.ini's `[signal] sr`, overridable with
`--sample-rate`). **Native per-rate generation is the recommended standard
flow**: run `gen_dataset.py --sample-rate 16000` for RNNoise-ERB/GTCRN and
`gen_dataset.py --sample-rate 48000` for DeepFilterNet2, each
producing its own WAV pairs measured and mixed at the rate it will actually
train on:

```bash
python3 gen_dataset.py --config config.ini --output data_16k --hours 25 \
    --sample-rate 16000
python3 gen_dataset.py --config config.ini --output data_48k --hours 25 \
    --sample-rate 48000
```

⚠ **`pack_dataset.py --target-sr` does NOT exactly preserve level/SNR** —
do not treat a 48 kHz→16 kHz downsample-at-pack-time as a standard way to
produce a second rate's training data when SNR/level fidelity matters. The
level normalization (and the requested SNR mix) happens at generation time,
against the SOURCE rate's spectrum; a later resample only re-shapes the
already-mixed signal, it does not re-mix to the new rate's requested SNR.
For narrowband content this can drift by tens of dB — an extreme but legal
config (1 kHz speech mixed against 12 kHz noise at a requested ~0 dB SNR)
measures ~48.6 dB after a 48 k→16 k downsample, because the noise energy
above the new 8 kHz Nyquist is simply gone. Real corpus content typically
drifts far less (most speech/noise energy sits well below either Nyquist),
but "typically less" is not a contract `pack_dataset.py --target-sr` can
promise — only re-mixing speech/noise stems natively at the target rate can
guarantee the requested SNR exactly. `pack_dataset.py`'s packed payload now
records `effective_rms_dbfs` (measured AFTER any `--target-sr` resample AND
after casting to the packed `dtype` — i.e. on the exact bytes stored in
`data`, not the higher-precision tensor before either step — per sample,
per channel) precisely so a consumer can audit this drift on its own
packed data rather than trusting generation-time metadata that describes a
different rate.

Downsampling at pack time remains a supported, useful SHORTCUT when you
specifically do not need exact fidelity — e.g. a quick 16 kHz smoke-test
packed file derived from an already-generated 48 kHz batch, or extending
`source_sr_values` coverage cheaply (see below) — not the default path to
a second rate's real training data:

```bash
python3 pack_dataset.py --input data_48k/pairs --output data_48k/packed.pt \
    --dtype float16
python3 pack_dataset.py --input data_48k/pairs --output data_16k/packed.pt \
    --target-sr 16000 --dtype float16
```

`pack_dataset.py --target-sr` shares the exact anti-aliasing constants and
clip guard with `resample_dataset.py` (imported, not duplicated), so the two
are numerically equivalent up to the 16-bit quantization `resample_dataset.py`
adds by writing an intermediate WAV — `pack_dataset.py` skips that write
entirely. Keep using `resample_dataset.py` when you actually want a listenable
16 kHz WAV copy (spot-checks, external tools).

Generating independently per rate is also the only way to get the full,
undiluted `source_sr_values` sweep for a 16 kHz consumer — see the accepted
trade-off noted below.

### Upsampled lower-rate source simulation

Some deployments run a 16 or 48 kHz enhancement algorithm on audio that was
captured at a lower sample rate and upsampled only to satisfy the algorithm's
input contract. The generator models that path with discrete, realistic source
rates:

```ini
[augmentation]
p_resample = 0.1
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
rate. The default probability is 10% and should be adjusted to the expected
deployment mix.

⚠ **Accepted trade-off when reusing a 48 kHz set for a 16 kHz consumer**: this
filtering happens once, at generation time, against the 48 kHz algorithm rate.
`resample_dataset.py`'s later 48→16 kHz pass does not re-evaluate it — any
draw of `source_sr_values` at or above 16000 has a post-resample bandwidth
cap (≥8 kHz Nyquist) indistinguishable from an unaugmented sample, so the
*effective* `p_resample` seen by the 16 kHz copy is diluted to roughly
`p_resample × (rates below 16000) / (rates configured)` — with the example
list above, about 2/7 of the configured probability. This is a known,
accepted cost of sharing one 48 kHz source across both rates rather than
generating natively per rate; restrict `source_sr_values` to `8000, 12000`
before generating if a 16 kHz consumer needs the full, undiluted augmentation
rate instead.

### RIR / DRR contract

The RIR path follows DeepFilterNet's `RandReverbSim` conventions: full and
late-suppressed target RIRs are independently L2-normalized, and SNR is
measured against the clean target while the mixture contains the full-RIR
speech. `drr = 0.3` is a linear dry/reverberant blend factor in `[0, 1]`, not
a dB value. The dry target is delayed to the trimmed RIR's direct-path peak so
target and reverberant mixture do not acquire a pre-delay mismatch.

### SNR and clean/noise edge cases

Mixed samples uniformly draw from a discrete SNR set (extended 2026-08 with
two lower, harsher values beyond DeepFilterNet's original
`-5, 0, 5, 10, 20, 40 dB`):
`-15, -10, -5, 0, 5, 10, 20, 40 dB`. DeepFilterNet3 puts `-100` in that list
as a sentinel for a noise-only sample; this generator represents the edge
case explicitly instead:

```ini
[mixing]
snr_values = -15, -10, -5, 0, 5, 10, 20, 40

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

### Target-level normalization (`level_mode`)

Without this, a sample's absolute level is purely inherited from whatever
level the source speech/noise files happened to be recorded at — a real
deployment sees far more level variation than that (near vs. far mic
placement, different recording gain, etc.). DNS-style target-level
normalization closes that gap:

```ini
[mixing]
level_mode = dns_target_level
target_level_min_db = -40
target_level_max_db = -10
```

Per sample, a `requested_level_dbfs` is drawn uniformly from
`[target_level_min_db, target_level_max_db]`, and ONE scale factor — derived
from the noisy mixture's whole-signal RMS — is applied identically to both
`noisy` and `target`, so the requested SNR (and, for `speech_only`, the exact
`noisy == target` identity) survives untouched. This runs **after** the
lower-rate-source-simulation step (`p_resample`), not before: measuring
against whatever that step's resample left behind means the level a
downstream consumer actually reads matches what was requested — it doesn't
drift across a resample the way measuring first would. `level_mode` is
validated at construction time (`'dns_target_level'` is the only implemented
value today; anything else raises rather than silently no-op'ing on a typo).

Per-sample `requested_level_dbfs` / `effective_level_dbfs` (the latter
measured last, after the final peak guard) are available via the
`return_metadata` API below — `effective_level_dbfs` can differ slightly from
`requested_level_dbfs` when the peak guard or mixture clipping distortion
also touched the signal after the level scale was applied.

### Clipping distortion: two independent knobs

The single `p_clipping` knob was split into a pre-mix and a post-mix
probability, aligned with how DeepFilterNet3 actually behaves (not just its
config surface — see the comment in `config.example.ini`):

```ini
[augmentation]
p_noise_clipping = 0.10     ; pre-mix, on the noise chain itself
p_mixture_clipping = 0.0    ; post-mix, post-level-normalization (old p_clipping)
clip_snr_min = 0
clip_snr_max = 20
```

`p_noise_clipping` distorts the noise signal before it is scaled into any
mixture — including `noise_only` samples, where `noisy` *is* that noise.
`p_mixture_clipping` is the old `p_clipping`'s exact behavior/semantics
(post-mix, `target` unaffected), just renamed for symmetry. Both are skipped
for `speech_only` pairs, which stay exact identity pairs.

By default this project has no train/val split concept at generation time
(splitting happens downstream, in each model's own `train.py`), so unlike
DFN3's train-only clipping, both knobs apply unconditionally to every
generated sample — a deliberate distribution choice for a generator whose
output IS the val set too, not an oversight. For a run whose entire output
IS a held-out validation batch, set `[gen] generation_split = validation`
to force-zero both clipping knobs for that run, matching DFN3 exactly:

```ini
[gen]
generation_split = validation   ; omit (or 'train') for the unconditional default above
```

`generation_split` participates in the `--resume` config-hash check (see
"Resuming" below) like every other `config.ini` key, so switching it and
resuming into the same `--output` directory is refused rather than silently
mixing a clipped and an unclipped batch together.

⚠ **`generation_split = validation` controls ONLY the clipping
augmentation.** It does not partition `[paths] speech_dir`/`noise_dir`
into disjoint train/validation source files, and this generator has no
mechanism that does. If you need a genuinely held-out validation set (no
speech/noise source file used in both splits), you must supply separate
`speech_dir`/`noise_dir` (or otherwise pre-partitioned source lists)
yourself — e.g. two `config.ini` files pointing at two non-overlapping
corpus directories, one per `gen_dataset.py --output` batch. Setting
`generation_split = validation` against the SAME source directories as a
`train` batch only changes clipping; the two batches can still share
underlying speech/noise recordings.

### Per-sample metadata (`return_metadata`)

`DNS4Dataset(cfg, return_raw=True, return_metadata=True)` changes
`__getitem__`'s return from `(noisy, target)` to `(noisy, target, metadata)`,
where `metadata` is a plain per-call dict — not a shared/stateful
side-channel, so it is safe to read under multi-worker `DataLoader` use.
Fields cover mix mode, source file provenance (speech/noise/RIR paths),
which augmentations fired this sample (biquad, resample simulation, both
clipping knobs) and their sampled parameters, and the level-normalization
pair (`requested_level_dbfs` / `effective_level_dbfs`). `return_metadata=True`
requires `return_raw=True` (raises otherwise) — it has no meaning against the
STFT feature/gain-target return shape.

⚠ **`gen_dataset.py` does not persist this.** It used to write a
`NNNNNN.json` sidecar beside every `NNNNNN.wav`; the on-disk corpus is now
WAV only, so this dict exists solely for a caller that constructs
`DNS4Dataset` itself and reads it in-process. The field that describes the
audio a consumer actually trains on is `pack_dataset.py`'s own
`effective_rms_dbfs` (see the design section above), measured after any
`--target-sr` resample and after the cast to the packed dtype.

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
| `dataset.py` | `DNS4Dataset` — the augmentation engine (biquad filters, RIR/RT60, SNR mixing, bandwidth limiting, DNS-style target-level normalization, split pre/post-mix clipping, opt-in per-sample metadata) |
| `pack_dataset.py` | Packs a WAV-pair directory into a single `.pt` tensor file, optionally resampling (`--target-sr`) while packing (removes per-file I/O overhead for small/medium datasets) |
| `packed_dataset.py` | Shared mmap-capable loader for packed `(N, 2, T)` tensors |
| `resample_dataset.py` | Optional standalone resample of an existing dataset to a WAV copy (listening/QC; `pack_dataset.py --target-sr` covers the training path) |
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
  pairs/000000.wav, 000001.wav, 000002.wav, ...
```

Each WAV is 2-channel: ch0=noisy, ch1=clean, written atomically (temp-file +
rename). **That is the whole on-disk format** — no `meta.json`, no per-sample
JSON sidecar, no contract version. A directory of `NNNNNN.wav` is the corpus,
so one stays packable after being copied, resampled or trimmed by hand, and a
corpus generated by the older sidecar-writing version packs unchanged (its
leftover JSON is simply ignored, not read and not an error).

Run BOTH commands above — one native generation per target rate is the
recommended path (see the design section above for why downsampling at
pack time is a shortcut, not a substitute, when SNR/level fidelity
matters). Skipping the 16 kHz command and relying solely on
`pack_dataset.py --target-sr` against the 48 kHz batch is only appropriate
when exact fidelity does not matter for your use case.

Useful flags: `--resume` (continue an interrupted batch, or extend a
finished one to a larger `--hours` total), `--start-idx` / `[gen] start_idx`
(numbers a brand-new, EMPTY `--output` directory, e.g. a separate shard —
`effective_seed = seed + start_idx`), and `--seed`. Without `--resume`,
`gen_dataset.py` refuses to run at all against an `--output` directory that
already contains any sample file, regardless of `--start-idx` — see
"Resuming / dataset contract" below for why.
The default `[gen] seed = -1` requests a fresh OS-generated seed every run;
the actual seed is **printed and not stored anywhere**, so a batch generated
with the default seed cannot be reproduced after the fact. Pass a non-negative
seed such as `--seed 42` when reproducibility matters. DataLoader workers are seeded
independently from that run seed, so workers do not repeat the same Python or
NumPy random stream.

`--hours` is converted directly to a whole-segment count:
`ceil(hours × 3600 / segment_sec)`. It therefore exceeds the requested
duration by less than one segment. It is not rounded to a complete
`DNS4Dataset` epoch; when necessary, the final dataset pass is partial.

#### Resuming: what the directory can and cannot tell you

`--resume` continues from the highest `NNNNNN.wav` in `pairs/`. Every WAV is
written to `tmp.NNNNNN.wav` and renamed into place, so **a visible file is a
finished file** — a process killed mid-write leaves a `tmp.` file that no scan
here or in `pack_dataset.py` ever matches. That rename is now the only
completion marker a sample has, which is why nothing may ever write directly
to the final path.

The scan runs FIRST, before `DNS4Dataset` is constructed (before the RIR
directory glob/cache load and the one-sample profiling pass), so a refusal
exits immediately rather than after paying for both. What it checks:

- **The numbering must have no hole.** Resuming continues from
  `max(index) + 1` and only ever walks forward, so a gap below the highest
  index can never be backfilled. `--resume` refuses instead of leaving it.
- **The batch's start is its own lowest index.** A shard first created with
  `--start-idx 500` therefore resumes at the end of the `000500...` range
  without the flag being repeated, and passing a conflicting `--start-idx` is
  refused rather than silently renumbering the batch.

**⚠ What is no longer checked, because nothing records it.** There is no
`meta.json`, so a `config.ini` edit (`snr_values`, `p_rir`, `level_mode`, the
clipping knobs, `generation_split`, …) or a `--sample-rate` change between two
`--resume` runs is **not detected**: both distributions land in one directory
with no error and no record. Resume into a directory only with the config that
started it; use a fresh `--output` otherwise. Deleting a batch's *leading*
sample is likewise absorbed — the next-lowest index simply becomes the start.

**Without `--resume`, any sample already present in `pairs/` is a hard
refusal — regardless of `--start-idx`.** A plain re-run into a non-empty
`--output` used to be silently accepted: for `--start-idx 0` it validated
nothing at all, and for `--start-idx N > 0` it only warned about a single
exact-index collision. Both let a config change get generated over part of an
existing batch while the rest stayed behind. Extending an existing batch goes
through `--resume --hours <new TOTAL, not an increment>`; `--start-idx` is only
for numbering a genuinely fresh, empty `--output` directory (e.g. a separate
shard).

`--repair-resume` now does one thing: delete leftover `tmp.*.wav` files from an
interrupted run. They are invisible to every scan either way, so this is disk
hygiene, not a repair.

### 2. Pack the generated dataset, resampling per consumer as needed

```bash
python3 pack_dataset.py \
    --input data_48k/pairs --output data_48k/packed.pt --dtype float16
python3 pack_dataset.py \
    --input data_48k/pairs --output data_16k/packed.pt \
    --target-sr 16000 --dtype float16
```

Use `data_48k/packed.pt` for DeepFilterNet2 and
`data_16k/packed.pt` for RNNoise-ERB/GTCRN. Omitting `--target-sr` packs at
the source WAVs' own rate and requires every input file to already share one
native rate. `--target-sr` resamples everything to one output rate, so mixed
native rates are accepted in that mode.

`--input` is a directory of 2-channel `NNNNNN.wav` files — nothing else is
required, and nothing else is read. A stray `tmp.NNNNNN.wav` (an
in-progress/crashed write, see "Resuming" above) and any WAV whose name is not
all digits are both ignored, not errors. Leftover `NNNNNN.json` sidecars or a
`meta.json` from an older corpus are ignored too, which is what lets existing
data be packed without regenerating it.

What still stops a pack, because it is checkable from the audio itself:

- **A gap in the numbering** (e.g. 0, 1, 3 with 2 missing) — usually a sample
  deleted and never regenerated. Pass `--allow-index-gaps` if the subset is
  deliberate.
- **A file that is not 2-channel**, whose length differs from the first file's,
  which contains NaN/Inf, or (without `--target-sr`) whose native rate differs
  from the first file's. Any one of these stops the whole pack rather than
  being silently excluded — dropping it would shrink the corpus without saying
  so and reopen an index gap the scan just checked. `--target-sr` resamples
  every file to one output rate, so mixed native rates are legitimate then.

⚠ **What the packer cannot check any more.** Nothing records which config, seed
or generation run produced a directory, so two runs' output merged into one
directory packs cleanly, and the packed payload carries no
`contract_version`/`config_hash` — it does not claim a provenance it cannot
see. The payload does carry `sample_indices` (the original `NNNNNN` for each
row of `data`, same order), so a packed row still traces back to its source
WAV. The `.pt` file itself is written atomically (temp-file + `os.replace`) —
a crash partway through `torch.save()` (payloads can be many GB) can never
leave a truncated file that looks like a complete packed dataset.

### Resample guidance (anti-aliasing parameters)

`pack_dataset.py --target-sr` (and the standalone `resample_dataset.py`,
below) use `torchaudio.functional.resample` with the Kaiser-windowed
sinc-interpolation kernel (`resampling_method = "sinc_interp_kaiser"`), which
is explicitly anti-aliased — a lowpass is applied before decimation so no
spectral energy folds back below Nyquist. Default preset (`--quality best`,
torchaudio's own "kaiser_best" tutorial preset):

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
near-full-scale content). If a resampled pair's peak exceeds `0.999`, both
`pack_dataset.py --target-sr` and `resample_dataset.py` scale the whole
`(noisy, clean)` pair down together (preserving their relative level / SNR)
before writing/packing, rather than letting a downstream int16 write clip
silently. `pack_dataset.py` reports how many pairs needed this; the
standalone `resample_dataset.py` also reports the overall peak level
observed.

### 3. Optional: materialize a resampled WAV copy

```bash
python3 resample_dataset.py \
    --input data_48k --output data_16k --target-sr 16000 --workers 4
```

Same anti-aliasing/clip-guard as `pack_dataset.py --target-sr` (the constants
are imported from this module, not duplicated), but writes an actual 16 kHz
WAV directory — useful for listening spot-checks or feeding external tools.
The output is WAV only, like its input, so it is packable exactly like the
original. Not needed for training: `pack_dataset.py --target-sr`
resamples straight into the packed tensor and skips this WAV write (and the
16-bit quantization it would otherwise add) entirely.

### Tests

```bash
make test
```

## Requirements

```
torch>=1.13,<2.9
torchaudio>=0.13,<2.9
soundfile>=0.12
numpy
tqdm
```

## Consumers

`../RNNoise-ERB/train.py` imports the shared `PackedDataset` from this package;
DeepFilterNet2 consumes the corresponding 48 kHz packed dataset. Model
directories do not keep private copies of the augmentation engine. A model's
own checkpoint contract does not require different waveform pairs merely
because its output composition differs.
