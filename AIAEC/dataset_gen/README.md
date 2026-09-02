# AEC dataset generation

Renders acoustic-echo scenarios as five separated WAV stems, then projects the
four signals required for training into `.pt` shards. This is the only AIAEC
dataset package. It reuses shared DSP from the separate `AINR/dataset_gen/` NR
generator rather than forking that DSP.

## The five stems

Every generated WAV is a `(5, T)` tensor in `STEM_ORDER`:

| # | stem | what it is |
|---|---|---|
| 0 | `far_render` | **X** — the far-end signal as the device rendered it, i.e. the AEC reference. Digital and clean: the loudspeaker's distortion happens *downstream* of this tap. |
| 1 | `near_speech` | **S** — the near talker through the full room RIR; retained in WAVs for mixing/audit, not copied into packed shards. |
| 2 | `near_target` | **S_early** — the same near talker and gain through the early/late-suppressed RIR; the common training target. |
| 3 | `mic_postclip` | **Y** — what a model actually receives, after capture clipping/AGC. |
| 4 | `linear_error` | **E** — frozen PBFDKF output, `Y - D_hat`. This is not oracle residual echo, and it is not unconditionally the filter's residual: the frontend runs the AEC3 post chain on the context seam, so a hop whose linear estimate the quality analyzer has not declared usable and whose residual carries more energy than the capture publishes the capture instead (`E == Y` there, crossfaded over 30 samples). |

**N** (`local_noise`, ambient noise at the mic) is no longer a persisted
stem: no current model task targets echo cancellation without denoising, so
it is audit-only now, like `echo` and `mic_preclip` always were (see below).

The signal model the corpus exists to serve:

```
Y = S + N + D          microphone
X                      far-end reference
D_hat = Y - E          frozen PBFDKF echo estimate, derived when needed
E     = Y - D_hat      stored linear error
R     = D - D_hat      residual echo — emerges, never a target
```

The default config renders each complete 20–30 second parent sequence first,
runs one stateful Python PBFDKF instance over `mic_postclip + far_render`, and
only then cuts all five WAV stems into 10-second chunks. The PBFDKF resets
between parent sequences and never at a chunk boundary.

The packer writes `(4,T)` tensors in `PACKED_STEM_ORDER`:
`far_render`, `mic_postclip`, `linear_error`, `near_target`. RES+NR candidates
read `E + X`; end-to-end candidates read `Y + X`; all target `S_early`
(denoised, dereverberated and echo-free). `D_hat` is derived as
`mic_postclip - linear_error` when required and is never stored separately.

**⚠ `echo` (D), `local_noise` (N) and `mic_preclip` (S+N+D, pre-clip/AGC) are
NOT stored.** No model task targets echo cancellation without denoising any
more, and no candidate sees oracle residual `R`. All three are still
**computed on every render** — `aec_dataset.AecSequenceRenderer.render()`
returns them under `RenderedSequence.audit` — so the corpus's central
invariants (`mic_preclip == S+N+D`, "echo really is a delayed copy of X") stay
verified at generation time; see `tests/test_aec_dataset.py`, which checks them
directly against the renderer rather than a packed shard. If you need `echo`/
`local_noise`/`mic_preclip` for a one-off analysis, call
`AecSequenceRenderer.render()` yourself — do not add them back to
`STEM_ORDER` for that.

Old six-channel WAVs/shards (with a separate `local_noise` stem) are rejected.
To upgrade an existing four-channel render without repeating speech/noise/RIR
mixing, run `rematerialize_linear_aec.py`; it reconstructs complete sequences
in `(sequence_id, chunk_index)` order from the filenames and rewrites the last
channel (`linear_error`) in place.

Existing five-channel WAV corpora need no regeneration for the four-channel
training contract: run `pack_aec_dataset.py` again. Old five-channel `.pt`
shards are intentionally rejected because their target semantics differ;
repacking reads the existing WAVs and drops only `near_speech`.

**⚠ That covers the channel layout, not the fifth channel's contents.** A corpus
rendered before the frozen frontend moved onto the context seam carries a
`linear_error` produced without the over-output capture guard, and the packer
stamps whatever contract `--config` builds today onto whatever audio it finds.
Rematerialize that channel first — see "Recomputing `linear_error` after a
frontend change".

`AecStems` gives these names; nothing indexes the channel axis by number.

```python
from AIAEC.dataset_gen import PackedAecDataset, build_model_view
ds = PackedAecDataset('data_aec/packed/train')
v = ds.stems_of(0)
view = build_model_view(v, 'DeepVQE_S', sample_rate=ds.sr)
mic, far, target = view.inputs['microphone'], view.inputs['far_end'], view.target
```

`build_spectral_model_view(view, grid)` converts that waveform contract into
the exact `[B,T,F]` keyword tensors accepted by each retained model.

## Metadata

**The rendered corpus is WAV and nothing else.** `<split>/seqs/SSSSSS_CCC.wav`
is the entire on-disk contract: no run `meta.json`, no per-sequence sidecar. A
packed clip therefore carries only where it came from:

| field | meaning |
|---|---|
| `sequence_id`, `chunk_index` | which parent sequence, and where in it — both read back out of the filename |

Everything else the shard needs is either fixed (`stems`, the channel order),
read from the audio (`sr`, chunk length) or rebuilt from `config.ini` at pack
time (`linear_aec`, `linear_aec_contract_hash` — the frozen PBFDKF contract,
which cannot be recovered from a WAV and which inference must reconstruct
exactly).

⚠ **What that costs.** The renderer still computes a full per-chunk
description — `speaker_id`, `far_speaker_id`, `noise_id`, `rir_id`, `room_id`,
`device_id`, `ser_db`, `snr_db`, `erl_db`, `bulk_delay_samples`,
`delay_jitter`, `sro_ppm`, `nonlinear`, `clipped`, `agc`, `talk_mode`,
`echo_mode`, `impairments`, `scenario`, `sequence_scenario`, `sequence_seed`,
`split` — but it is now visible only
in-process, on the `RenderedSequence` a worker hands back
(`tests/test_aec_dataset.py` reads it there). None of it reaches disk, so:

- a curriculum keyed on `scenario` has to measure the chunk instead, which the
  separated stems make possible;
- source-disjointness is audited at the renderer, not on the packed corpus;
- the packer can no longer tell WHICH config, seed or manifest
  produced a chunk. They check shape — chunk count, rate, length, channels,
  encoding — and nothing else. Resume into a directory only with the run that
  started it.

The clip-level facts below still hold of the audio itself, they are simply no
longer written down:

`clipped` and `agc` are separate distortions — one memoryless and
instantaneous, one a slow gain with memory — and a model that confuses them
fixes the wrong one. Together they are exact: `mic_postclip` differs from
`mic_preclip` **if and only if** one of the two is set (`mic_preclip` itself is
audit-only, see above — `tests/test_aec_dataset.py` checks this identity
against the renderer directly).

**⚠ `ser_db` / `snr_db` / `erl_db` are sequence-level.** They describe how the
whole configured parent sequence was set up. A single chunk departs from them
by a few dB (ERL) or by anything at all (SER/SNR), because a chunk in which the
near talker happens to be silent has no signal to define a ratio against.

**⚠ `±inf` is deliberate.** It marks a ratio that is undefined because one of
its two signals is absent, rather than a fabricated number that would silently
pass a threshold filter.

### Layered scenes

New corpora no longer choose one mutually-exclusive scenario. They draw three
orthogonal layers:

- `[talk_modes]`: `far_only`, `near_only`, `double_talk`, or
  `duplex_random` (relative categorical weights);
- `[echo_modes]`: normal echo, `ref_dropout`, or `far_active_no_echo`
  (conditional probabilities for far-capable sequences);
- `[impairments]`: independent path/capture events such as
  `echo_path_change`, `nonlinear_spk`, `clipping_agc`, `delay_jitter`, `sro`
  and `codec_mismatch`.

This separation is intentional. In the former categorical planner a sequence
could be `double_talk` **or** `echo_path_change` **or** nonlinear/clipped, so
the difficult intersection never existed. `[complex_cases]
p_dt_stress_combo` now guarantees a small measurable tail containing DT +
path movement + nonlinear loudspeaker + clipping/AGC. Within that tail,
`[activity] dt_force_edge_overlap` pins DT to the leading edge of the first far
burst and trailing edge of the last, covering both a cold PBFDKF/model state
and a mature state late in the sequence without making ordinary DT scripted.
`far_active_no_echo` is a hard negative: its far scheduler covers the complete
parent sequence — as back-to-back utterances, each drawing its own file, since
one whole-sequence run would be zero-padded to length and go silent after the
first utterance — while acoustic echo is exactly zero. The per-chunk label is
then measured rather than asserted, so no emitted chunk can silently degenerate
into the already-covered silent-reference case and keep the label anyway.

`scenario` remains a compatibility, **per-chunk** summary and cannot describe
all simultaneous conditions. A `ref_dropout` parent is mostly *not* in
dropout, and a path-changing parent has exactly one transition chunk. New
diagnostics must use `talk_mode`, `echo_mode` and `impairments` in-process (or
measure the persisted separated stems). `sequence_scenario` remains only a
legacy summary.

**`ref_dropout` is load-bearing.** During a dropout the far end is genuinely
silent — `X == 0` **and** `D == 0` — so no model may hallucinate echo removal.
Every current candidate is a joint AEC+NR route and may still suppress `N`, so
`ref == 0 -> output == mic` is **not** a universal gate (that expectation
belonged to the now-retired AEC-only Align-CRUSE route, which targeted
`S+N`). `[dropout] ref_dropout_echo_continues_p` can make
the loudspeaker keep playing while the reference is lost, but that asks the
model to predict an echo from nothing, so it is **0 by default**: raising it
trains hallucination.

Note that `p_ref_dropout` is conditional per far-capable *sequence*; the
per-*chunk* share is smaller. If the idle term needs more, lengthen the dropouts
(`ref_dropout_chunks_max`) rather than adding sequences that are mostly not
dropouts. `near_only` supplies idle chunks too, but only `ref_dropout`
contains the *transition* into and out of idle, which is the hard part.

## Train/validation split

The selected training protocol generates one unified pool (`--split all`) and
uses `training_common.split_dataset_by_sample` after packing. A dedicated
seeded generator randomly assigns individual 10-second chunk indices, so the
split is reproducible, disjoint, and covers the whole corpus. Different chunks
from the same sequence, speaker, RIR, or device may intentionally straddle
train and validation. The train loader reshuffles every epoch; validation does
not shuffle.

This validation score is useful for optimization progress, not a
source-generalisation claim. Use a separately generated source-disjoint corpus
(`--split train`/`val`) or held-out real recordings for that measurement.
`manifest.py` retains the optional source-disjoint generator for this purpose.
The split is built deterministically in memory by default, so generation emits
only WAVs. Pass an explicit `--manifest PATH` only when the source directories
may change between the separate train/val runs and the split must be frozen on
disk.
Every checkpoint stores the dataset fingerprint, split seed/fraction, complete
train/validation indices, and PBFDKF contract so resume cannot silently change
the comparison.

## Why sequences are long

Parent sequences are 20–30 s by default, cut into consecutive fixed-length chunks that
share a `sequence_id` and carry an increasing `chunk_index`.

Long sequences are still required because PBFDKF adaptation from cold,
echo-path changes, and drift must happen before the last channel is cut. The packer
keeps `(sequence_id, chunk_index)` order for deterministic reconstruction and
streaming evaluation. Training itself treats chunks as independent shuffled
samples; `SequenceChunkSampler` remains only as an evaluation utility and is
not used by any trainer.

### Migrating an existing corpus

Changing only `linear_error` cannot create the newly composed acoustic scenes:
the four source stems already encode talk activity, path movement,
nonlinearity and capture clipping. Generate a new WAV corpus in a new output
directory, pack it, and train a new checkpoint. Do **not** use `--resume` into
an old corpus: WAV filenames carry no config fingerprint, so shape-compatible
old chunks would be accepted even though they were rendered by the categorical
planner.

## Files

| File | Role |
|---|---|
| `aec_dataset.py` | the scenario simulator: nonlinearity, echo path, delay/jitter, SRO, dropout, AGC |
| `manifest.py` | unified/source-disjoint source manifest |
| `seq_layout.py` | the on-disk naming rules (`SSSSSS_CCC.wav`, temp names, scans) that the generator, packer and re-materializer all agree on |
| `gen_aec_dataset.py` | CLI — renders complete sequences to 5-channel WAV chunks |
| `linear_aec.py` | frozen PBFDKF contract and full-sequence materializer |
| `rematerialize_linear_aec.py` | rebuilds the last channel from existing four/five-channel WAVs |
| `materialize_pair.py` | **diagnostic**: one loose mic/far pair to a `linear_error` WAV, contract taken from the runtime |
| `pack_aec_dataset.py` | projects the five-channel WAVs into four-channel `.pt` shards (`--dtype float16` halves shard size and is safe to train on: every trainer widens to float32 at the device move, so only the stored dtype changes) |
| `packed_aec_dataset.py` | `PackedAecDataset`, returning `(stems, meta)` |
| `aec_features.py` | **the shared module the model projects import** |
| `config.example.ini` | every knob, documented (16 kHz) |
| `config.example.48k.ini` | the same file with the 48 kHz recipe already applied; the tests derive it from the recipe above and compare key by key, so the two cannot drift apart |
| `tests/` | the invariants a consumer cannot detect being broken |

`aec_features.py` owns `AecGrid`, `stft`/`istft`, `alpha_from_tau`, `AecStems`,
`STEM_ORDER`, `PACKED_STEM_ORDER` and `SequenceChunkSampler`. **⚠ A model project that re-declares
any of these is opting out of the comparison** — the same failure that
`AINR/tests/test_bakeoff_protocol.py` already guards for the NR split, where a
5%-vs-10% divergence meant two models were compared on different corpora.

### The signal grid

16 kHz first target: `n_fft = 512`, `win_len = 512`, `hop_len = 256`,
`n_freqs = 257`, sqrt-Hann (periodic), 50 % overlap (COLA), 62.5 fps. The
48 kHz grid is `sr = 48000, n_fft = 1024, win_len = 1024, hop_len = 512` —
`AecGrid` derives `n_freqs` and `frame_rate`, and rejects a hop that is not
`win_len/2` or a `win_len` different from `n_fft`.

**⚠ `[signal]` is not the whole rate change.** The linear-AEC frontend's
`(frame, hop)` is frozen per rate in `linear_aec.py`'s
`FROZEN_FRAME_HOP_BY_SR`, so `[sequence] chunk_sec` must satisfy
`round(chunk_sec * sr) % hop == 0` — the shipped `10.0` is exact at 16 kHz and
not at 48 kHz, where whole seconds must be multiples of 4 (`8.0` is exact on
both grids and keeps the same 2–3 chunks per sequence). `[codec]
source_sr_values` is rate-dependent as well. The complete 48 kHz recipe
is at the top of `config.example.ini`, and `config.example.48k.ini` is that
recipe already applied — copy it instead of re-deriving the values by hand.
Generation refuses an inexact `chunk_sec` in its config preflight, before the
source/RIR scan.

**⚠ No frame counts, EMA coefficients or window lengths are hardcoded
anywhere.** Time constants are given in seconds and converted with
`alpha_from_tau(tau_sec, hop_len, sr) = exp(-hop_len / (sr · tau_sec))`. A
literal `0.92` in a config would be 191 ms on one grid and 128 ms on the other,
so the 48 kHz variant would quietly become a different algorithm. Use
`grid.n_frames(n_samples)` instead of writing a frame count.

## Usage

**Selected training protocol — unified pool and random chunk split:**

```bash
cp AIAEC/dataset_gen/config.example.ini AIAEC/dataset_gen/config.ini
# edit [paths] speech_dir / noise_dir / rir_dir, and [devices] device_ids
# optionally also [paths] far_speech_dir, for an independent far-end
# reference corpus that never shares a file/speaker with speech_dir

python3 -m AIAEC.dataset_gen.gen_aec_dataset \
    --config AIAEC/dataset_gen/config.ini --output data_aec \
    --hours 100 --split all --workers 4 --seed 42

python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all
```

⚠ The packer needs `--config` because the frozen linear-AEC contract that
produced the `linear_error` stem cannot be recovered from a WAV, and inference
has to construct the same one. Pass the config the corpus was generated with;
nothing cross-checks that claim any more.

⚠ `--output` must not already contain `shard_*.pt`. Loading a packed directory
takes every `shard_*.pt` in it and there is no index file naming this pack's own
shards, so a leftover from an earlier pack would silently join the corpus. Use
`--overwrite` to replace them deliberately. New shards are staged under
`.pt.tmp` names and published only after every WAV passes validation; a
validation/serialization failure keeps the previous pack intact.

⚠ Every current trainer runs at 16 kHz/512. A trainer on a different grid
would need a SEPARATE `config.ini` carrying the whole recipe for that rate
(`[signal]` alone is not enough — start from `config.example.48k.ini`, which
carries the whole 48 kHz recipe) and its OWN
`--output` (e.g. `data_aec_16k` / `data_aec_48k`, matching that trainer's
`packed_dir`): generating a second rate into the same `--output` as the first
is refused once chunk WAVs exist, since the directory is not namespaced by
sample rate. The rate is a config property with no CLI override, precisely so
that the config the packer is later handed cannot disagree with the audio.

Layout:

```
data_aec/
  all/seqs/000000_000.wav       5-channel chunk, channels = STEM_ORDER
  all/seqs/000000_001.wav
  packed/all/shard_00000.pt     4-channel tensor, channels = PACKED_STEM_ORDER
```

No JSON is created in the default flow. An explicit `--manifest PATH` is the
only opt-in exception, used to freeze a source-disjoint split; packing and
training never read it.

The four trainers read `packed/all` and create the deterministic random chunk
split from `[data] val_fraction` and the training seed.

The contract records the AEC's identity three ways, and they are used for
different things:

| Field | Scope | Used by |
|---|---|---|
| `aec_commit`, `aec_source_hash` | raw-text **provenance** | `fingerprint()` → `--resume`, packing, integrity |
| `aec_behavior_hash` | normalized-AST **behaviour** | `require_linear_aec_contract` → materialization + inference |
| `behavior_hash_schema` | which canonicalizer produced the hash | compared alongside it, so a serializer change reports itself by name |

`aec_behavior_hash` hashes the parsed AST with docstrings stripped, so a comment
reflow, docstring reword or reindent under `lib/aec/python` does **not**
invalidate existing shards or checkpoints, while any change to an expression,
constant or control-flow path does — and fails closed.

**It must not depend on the interpreter, and getting that right needed a custom
serializer.** The first implementation used `ast.dump()`. Python 3.13 changed
`ast.dump` to omit fields equal to their default, so the same 48 files digest to
`89b866cd` under 3.9 and `402acc1a` under 3.14 with no code difference at all —
a dataset generated under one interpreter would be refused by training under
another, and a checkpoint would become unloadable on a Python upgrade alone.
`aec_behavior_hash.py` therefore canonicalizes the tree itself (`_canon_ast`),
applying that same "drop empty fields" rule uniformly on every version. The rule
also absorbs fields that simply do not exist on older versions — `type_params`
(PEP 695) is absent on 3.9 and `[]` on 3.14, so both emit nothing — while a
field that is genuinely *used* is non-empty, is emitted, and does change the
hash. Nothing is dropped silently.

That module is deliberately free of third-party imports, so the parity test can
run it under every CPython on the machine.
`tests/test_linear_aec_behavior_hash.py` asserts they all agree, and runs a
control that reproduces the old `ast.dump` path and asserts it *disagrees* — the
stability claim would otherwise pass vacuously on a single-interpreter machine.

It is a hash rather than a hand-maintained `behavior_version` on purpose: every
other compared field is either a `__post_init__` literal or echoed out of the
recorded contract by both call sites, so a version integer would be the same
constant on both sides and could never differ. That tautology shipped once
(2026-08-06) and `test_contract_comparison_is_not_vacuous` now guards it.

Scope: `aec_behavior_hash` covers `aec.py` plus everything under `modules/` —
the code `LinearAecProcessor` can actually reach. `diag/`, `tests/` and the
bench/eval tooling are excluded, and the module-level `__version__` assignment
is stripped before hashing, so editing a test, a golden generator or the release
version cannot strand a checkpoint. `aec_source_hash` still covers every
Python file under `lib/aec/python`; unlike the 48-file signal-path scope, that
provenance count intentionally grows when tests or diagnostics are added.

**Recomputing `linear_error` after a frontend change.** The source WAVs do not
need re-rendering — only the fifth channel does, then the packed shards:

```bash
python3 -m AIAEC.dataset_gen.rematerialize_linear_aec \
    --input data_aec/all --config AIAEC/dataset_gen/config.ini --jobs 8

python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all --overwrite
```

Far, mic, near-target and the sequence boundaries are preserved; only
`linear_error` is recomputed. Any checkpoint trained on the old distribution
must be retrained.

**Before you retrain, you can measure what the change cost.** A checkpoint
carries the contract it was trained under, so every path that loads one refuses
an engine whose `aec_behavior_hash` has moved — that refusal is the guard doing
its job, not a defect. To see how the existing checkpoint behaves on the new
frontend's signal, materialize one pair and hand it straight to the neural
post-filter:

```bash
python3 -m AIAEC.dataset_gen.materialize_pair mic.wav far.wav error.wav

python3 -m AIAEC.Align_ULCNet.inference ckpt.pt error.wav far.wav out.wav \
    --input-is-linear-error
```

`materialize_pair.py` builds its contract from the installed library rather
than from a checkpoint, so it has no recorded hash to disagree with;
`--input-is-linear-error` bypasses the frontend entirely, so nothing on that
side carries a contract either. Pass the **raw** far to inference — the
training contract pairs `linear_error` with raw far and the model's alignment
consumes it.

⚠ Both halves are evaluation-only. `materialize_pair.py` pads one waveform to a
hop boundary and trims the result back, so its tail is not what the corpus path
produces; its output must never be written into a corpus, and no contract
comparison can tell the two apart. Run the same pair through the old library
too if you want the comparison to have a baseline.

**`--jobs` is the only lever that matters.** Measured on a 16 kHz corpus, the
Python PBFDKF is **99.8%** of the run and file I/O is 0.1%, so nothing about
the WAV handling is worth optimizing. One process sustains roughly 3.3x
realtime, which puts a 200-hour corpus near 62 hours; sequences are
independent, so `--jobs N` scales approximately with N until CPU or memory
bandwidth saturates. Do not expect exactly N×: process startup, memory
bandwidth and storage all take a cut, and the useful ceiling is usually below
the logical-core count.

`--jobs` cannot change the corpus. Each sequence gets a fresh PBFDKF, writes
only its own chunks, and has no random source, so the only thing N changes is
the order sequences finish in. Pinned by
`tests/test_rematerialize_linear_aec.py`, which compares every sample of
`--jobs 1` against `--jobs 3`.

> ⚠ Compare AUDIO, not file bytes, if you check this yourself. libsndfile
> stamps a `PEAK` chunk with the wall-clock time when it writes a float WAV,
> so any two runs seconds apart differ in one byte at offset 61 regardless of
> `--jobs`.

**`--resume` is safe for this.** It skips a sequence only when THIS contract
already wrote it AND that sequence's chunks still match on disk — the ledger
records what was written, not what survived, so every claim is re-checked
before it is honoured. A ledger written by a different contract is discarded
whole rather than partially trusted. The ledger records a sequence only after
all of its chunks are on disk, so an interrupted run redoes at most the ones
that were in flight (up to `--jobs` of them). A run WITHOUT `--resume` starts
by emptying the ledger, so it can never inherit a claim it did not make.

**Between the two commands.** The packer is the only thing that checks the
corpus is whole, so the order matters and there is no step to skip:

1. `rematerialize_linear_aec` must exit **0**. A non-zero exit means some
   sequences still carry the old fifth channel; the ledger will say so and the
   packer will refuse, but do not go looking for a way around that.
2. Do not run a second rematerializer against the same split concurrently.
   Both would write the same ledger and each would overwrite the other's
   claims.
3. Then pack. If it refuses, finish the rematerialization -- re-run it, this
   time WITH `--resume`, which now re-checks each claim against the chunks on
   disk before honouring it.

If the run was interrupted, re-running with `--resume` is the cheap path and
is safe: it redoes only what is not already recorded and verified. Re-running
without `--resume` is also correct, just slower -- it empties the ledger and
starts over.

**The packer refuses a half-rematerialized corpus.** If
`linear_error.done.json` exists but does not name every sequence, the split is
part new frontend and part old, and `pack_aec_dataset.py` stops instead of
labelling the shards with one contract while they hold two — the one failure
here with no downstream detector, since the fifth channel is just samples once
packed. A corpus with no ledger at all is fine: a one-pass generation never
had one, and the guard is against a PARTIAL claim, not an absent one.

**v2 → v3 has no automatic migration, deliberately.** A v2 contract records only
a raw-text source hash, so once `lib/aec` has moved on there is no way to
recover what the producing build's *behaviour* hash was — stamping the current
one would assert a compatibility nobody verified. Re-stamp a dataset by
re-running `rematerialize_linear_aec.py` (it re-runs PBFDKF and rewrites the
`linear_error` channel, which is the honest thing to do). A v2 **checkpoint**
cannot be repaired and must be retrained against a v3 corpus.

`behavior_hash_schema` is folded into the digest as well as compared, so the two
can never disagree. Bump it (`canon-ast-1` → `canon-ast-2`) whenever `_canon_ast`
changes what it emits for unchanged input; a mismatch then names the serializer
instead of looking like an AEC code change. No stamped artifact has ever carried
a v3 contract without this field, so there is no migration path for one.

The check is conservative: a pure refactor (renaming a local, reordering
independent statements) also changes the behaviour hash. Refusing to load is the
safe direction — rematerialize rather than loosening the check. Refresh the
channel with the command below, which avoids repeating acoustic mixing:

#### The one exception: verified frontend-equivalent migrations

> **⚠ The table holds exactly one pair.** `aec_reset()` was made to return a
> fresh instance (it had been carrying a converged filter's AEC3 startup gates,
> its Kalman `H_error` and the near-end recency pair across the call), and the
> same revision removed the dead far-end power EMA from the Python filter. Both
> edits are under the signal scope, so the hash moved — but materialization
> builds one engine per parent sequence and never calls `reset()`, and nothing
> read the removed fields, so `linear_error` is byte-identical across the pair.
> A corpus already on disk travels forward; nothing needs rematerializing.
>
> The identities that predate the dominant-matched-filter-peak change are a
> different matter and are **retired, not retargeted**: that change **moves
> `linear_error`**, so their byte-identity evidence describes no frontend after
> it, and pointing one at a live hash would declare an old waveform compatible
> with a build that does not produce it. They are listed in
> `RETIRED_BEHAVIOR_HASHES` and refused with an instruction to rematerialize
> (`rematerialize_linear_aec.py`, WITHOUT `--resume`, then repack, then
> retrain), not with a bare hash mismatch.
> `behavior_hash_schema` stays `canon-ast-1`: these are behaviour changes, not
> canonicalizer changes. Pinned by
> `tests/test_linear_aec_behavior_migration.py`.

`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` in `linear_aec.py` is an explicit table of
`recorded → current` behaviour-hash pairs that are known to produce the *same*
`linear_error`. `require_linear_aec_contract` accepts exactly those pairs, with
a `RuntimeWarning` naming the migration; the existing shards and the trained
checkpoint stay valid, so **no regeneration, no re-stamping, no retraining**.
`behavior_hash_schema` is *not* bumped — the canonicalizer is unchanged, only
the sources it is applied to.

This is not a loosening of the guard, and specifically is not a way to accept
"old hashes" in general:

- an entry is a **single explicit pair**, never a wildcard or a version floor;
- it applies **only when `aec_behavior_hash` is the sole differing field**, so a
  real frontend change cannot ride along with an accepted pair;
- it is **one-way**. A checkpoint recorded under the newer hash run against the
  older build is a downgrade and stays refused;
- it is **single-hop**: the table is not applied transitively, so two stacked
  migrations need the composed pair, re-verified end to end;
- an unlisted hash is refused exactly as before.

Admission requires *measured* evidence, not an argument that the change looks
inert: render the frozen frontend (`LinearAecProcessor`, formed_output seam)
before and after over a scene that actually reaches the changed code, and show
the bytes are identical — plus a control proving the same harness *can* fail
(render with the new mechanism enabled and confirm the bytes move). The
byte-equality of a dead harness is worth nothing. The rationale and the numbers
for each shipped entry live in the comment on the entry itself.

The one shipped entry covers the frontend the 200-hour corpus was materialized
under. Its evidence is three complete 30 s sequences whose echo path changes
delay mid-sequence — so each one acquires, loses and re-acquires a lock, which
is the shared filter-restart path the same revision touches — rendered through
`LinearAecProcessor` on both sides of the pair and compared byte for byte, with
the run asserting `AEC.reset` is never called while materializing. The control
splices one `engine.reset()` into the hop loop and confirms the bytes move.
The exact scene and digests are recorded beside the table entry. Anything that
retunes a *live* mechanism does not qualify — rematerialize instead:

```bash
python3 -m AIAEC.dataset_gen.rematerialize_linear_aec \
    --input data_aec/all \
    --config AIAEC/dataset_gen/config.ini
python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all --overwrite
```

`rematerialize_linear_aec.py --resume` skips a sequence only when the CURRENT
contract already wrote it, per the `linear_error.done.json` ledger beside the
corpus; a ledger from another contract is ignored whole. Re-running after a
`[linear_aec]` config edit therefore redoes every sequence on its own, with or
without `--resume`. That holds across an accepted migration too: the ledger is
keyed on `fingerprint()`, which folds in the raw-text `aec_source_hash` and
`aec_commit`, so an INTERRUPTED rematerialization restarts even where the
migration applies. A COMPLETED one does not — the packer honours that ledger
with a `RuntimeWarning`, because refusing there would strand a finished corpus
behind advice that cannot be followed: the config is identical, and what moved
is `lib/aec`.

**The ledger records the frontend that wrote it,** not only the fingerprint.
The fingerprint is one-way, so a ledger carrying nothing else can say a corpus
was written by a different contract but never *which* — which leaves an
operator comparing two opaque hashes. `linear_error.done.json` therefore also
carries the producing contract under `linear_aec`, and the packer bridges on
that: an exact, config-independent comparison through
`require_linear_aec_contract`, which reports the field that disagrees and
names both builds. A ledger written before that field existed (only `contract`
and `sequences`) still reads normally everywhere; for those the packer
reconstructs the migrated-from build's fingerprint from its recorded
provenance instead (`MIGRATED_SOURCE_PROVENANCE`, one candidate per `lib/aec`
revision that carried the migrated-from behaviour hash — a behaviour hash is
comment-insensitive, so a RANGE of revisions carries it and each one wrote a
different fingerprint). If that refuses, the message says the ledger records
no identity and asks for `git -C lib/aec rev-parse HEAD` plus the `[signal]`
and `[linear_aec]` config sections from the machine that ran the
rematerialization — the two things needed to identify the build. Add `--jobs N` to spread independent sequences across
cores — it does not change a single sample.

Generation is deterministic given `--seed`: each sequence is seeded from
`(seed, split, sequence_id)`, so it renders identically regardless of worker
count or ordering, and `--resume` continues exactly. `--hours` resolves to a
fixed sequence list up front, so extending a corpus keeps every sequence it
already had.

**⚠ `--wav-encoding` defaults to `float32`** because the corpus's central
invariant, `mic_preclip == near_speech + local_noise + echo`, is checked at
generation time against the renderer's un-quantised audit tensors (`echo`,
`local_noise` and `mic_preclip` are not among the generated WAV stems — see
"The five stems" above). Quantising the WAV stems to `int16` would still
degrade any downstream arithmetic that combines them (e.g.
`D_hat = mic_postclip - linear_error`) by ~1e-4. `int16` halves the disk cost
and is fine for listening, not for arithmetic.

**⚠ `--workers > 0` on macOS uses spawn.** The shipped CLI has the
`if __name__ == '__main__'` guard it needs; a script that calls
`gen_aec_dataset()` at module level will fail without one.

## Tests

```bash
python3 -m pytest AIAEC/dataset_gen/tests/ -q      # from Audio_ALG/
```

They render a small synthetic corpus through the real pipeline and check the
things a consumer cannot notice being wrong: stem channel order, the stem-sum
identity, that `ref_dropout` chunks really have a silent reference, that the
echo really is a delayed copy of the reference at the recorded delay, that the
split is disjoint in the *generated data* and not only in the manifest, that a
sequence's chunks are contiguous and ordered, and that the STFT round-trips on
both the 16 and 48 kHz grids.

## Reuse and approximations

`prepare_rir`, `estimate_rt60`, `fftconvolve`, `delay_signal`, `active_rms`,
`apply_clipping`, `prevent_clipping`, `rand_biquad_filter`, `sample_snr`,
`parse_snr_values` and `simulate_upsampled_source` are imported from
`AINR/dataset_gen/dataset.py`. Local noise uses the **same discrete SNR set** as the
NR generator, drawn with the same helpers, so the two corpora are comparable on
the noise axis instead of being two definitions of "10 dB SNR".

Two deliberate approximations, both flagged in the code:

- **SRO** is Catmull-Rom fractional interpolation, not a bandlimited
  resampler. A few ppm is a slowly accumulating fractional delay and no
  integer-rate resampler can express it — `resample(16000, 16001)` is 62.5 ppm,
  an order of magnitude too coarse.
- **`codec_mismatch`** is band limiting plus µ-law requantisation, not a real
  codec. It produces the property that matters (a nonlinear, non-invertible
  difference between the reference and what was played) without adding a
  dependency. **⚠ No result may be reported as "robust to \<codec\>".**
