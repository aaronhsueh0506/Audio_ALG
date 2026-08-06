# AEC dataset generation

Renders acoustic-echo scenarios as **five separated stems** and packs them into
`.pt` shards. This is the only AIAEC dataset package. It reuses shared DSP from
the separate `AINR/dataset_gen/` NR generator rather than forking that DSP.

## The five stems

Every clip is a `(5, T)` tensor whose channel order is fixed and declared in
each shard:

| # | stem | what it is |
|---|---|---|
| 0 | `far_render` | **X** — the far-end signal as the device rendered it, i.e. the AEC reference. Digital and clean: the loudspeaker's distortion happens *downstream* of this tap. |
| 1 | `near_speech` | **S** — the near talker at the mic, already through the room RIR. Reverberant on purpose; that reverberation is desired signal. |
| 2 | `near_target` | **S_early** — the same near talker and gain through the early/late-suppressed RIR; the dereverberation target for DeepVQE-S and Align-CRUSE. |
| 3 | `mic_postclip` | **Y** — what a model actually receives, after capture clipping/AGC. |
| 4 | `linear_error` | **E** — frozen PBFDKF output, `Y - D_hat`. This is not oracle residual echo. |

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

The default config renders each complete 20–30 second parent sequence first, then
runs one stateful Python PBFDKF instance over `mic_postclip + far_render`, and
only then cuts all five stems into 10-second chunks. The PBFDKF resets between
parent sequences and never at a chunk boundary. Its full engine/source/grid
contract is stored in run, chunk, shard, and checkpoint metadata.

`model_views.py` maps the five stems to the candidate contracts. Align-ULCNet
and the two AENR variants read stored `E + X` and target `S`; CAGCRN targets
`S`; DeepVQE-S and Align-CRUSE target `S_early` (the joint end-to-end
AEC+RES+NR task, denoised + dereverberated + echo-cancelled). `D_hat` is
derived as `mic_postclip - linear_error` when required and is never stored
separately.

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
in `(sequence_id, chunk_index)` order, rewrites the last channel
(`linear_error`), and updates metadata.

`AecStems` gives these names; nothing indexes the channel axis by number.

```python
from AIAEC.dataset_gen import PackedAecDataset, build_model_view
ds = PackedAecDataset('data_aec/packed/train')
v = ds.stems_of(0)
view = build_model_view(v, 'DeepVQE_S', sample_rate=ds.sr)
mic, far, target = view.inputs['microphone'], view.inputs['far_end'], view.target
```

`build_spectral_model_view(view, grid)` converts that waveform contract into
the exact `[B,T,F]` keyword tensors accepted by each model. For
`DeepFilterNet_AENR`, also pass the model and its exact `read_feature_config()`
result; the adapter uses normalized STFT coefficients and returns separate
causal EMA states for the error and far feature streams. It rejects one shared
state because that silently changes both feature distributions.

## Metadata

Each clip carries a dict. The contract fields:

| field | meaning |
|---|---|
| `sequence_id`, `chunk_index` | which parent sequence, and where in it |
| `speaker_id` | the **near** talker (`''` = none). `far_speaker_id` is the other one -- from an independent pool if `[paths] far_speech_dir` is set, otherwise the same pool |
| `noise_id`, `rir_id` | source ids; `rir_id` is `"a\|b"` for an echo-path change (both `a` and `b` are always in the same room as `room_id`) |
| `manifest_version` | source-manifest schema/identity; prevents resume or packing from mixing sequences built under different source-mapping contracts |
| `config_hash` | fingerprint of the config this chunk was rendered under; what `--resume` compares to reject a sequence rendered under different settings |
| `sequence_seed` | the per-sequence RNG seed `plan_sequences()` derived from `--seed`; `--resume` also compares this (and `sequence_scenario`) since `--seed` lives outside config.ini and would otherwise not be seen at all |
| `ser_db` | near-speech-to-echo ratio (`+inf` = no echo, `-inf` = no near talker) |
| `snr_db` | near-speech-to-noise ratio (`-inf` = no near speech to define it against) |
| `erl_db` | echo return loss, echo vs the stored reference |
| `bulk_delay_samples` | reference-to-echo delay |
| `delay_jitter`, `sro_ppm`, `nonlinear`, `clipped`, `agc` | which impairments are present |
| `scenario` | see below |

Plus `sequence_scenario`, `far_speaker_id`, `room_id`, `device_id` and `split`.
The last three exist so that disjointness can be audited on the **generated
data** and not only on the manifest.

`clipped` and `agc` are separate flags because they are separate distortions —
one memoryless and instantaneous, one a slow gain with memory — and a model
that confuses them fixes the wrong one. Together they are exact: `mic_postclip`
differs from `mic_preclip` **if and only if** one of the two is set (`mic_preclip`
itself is audit-only, see above — `tests/test_aec_dataset.py` checks this
identity against the renderer directly).

**⚠ `ser_db` / `snr_db` / `erl_db` are sequence-level.** They describe how the
whole configured parent sequence was set up. A single chunk departs from them by a few dB
(ERL) or by anything at all (SER/SNR), because a chunk in which the near talker
happens to be silent has no signal to define a ratio against. Build curricula
on `scenario`, or measure the chunk yourself — which the separated stems make
possible.

**⚠ `±inf` is deliberate.** It marks a ratio that is undefined because one of
its two signals is absent, rather than a fabricated number that would silently
pass a threshold filter.

### Scenarios

`far_only`, `near_only`, `double_talk`, `ref_dropout`, `echo_path_change`,
`nonlinear_spk`, `clipping_agc`, `delay_jitter`, `sro`, `codec_mismatch`.

`scenario` is **per chunk** and is not always the sequence's intent. A 40 s
`ref_dropout` sequence is mostly *not* a dropout, and an `echo_path_change`
sequence contains exactly one chunk where the path changes. Labelling every
chunk with the sequence's intent would train a dropout-conditioned loss on
chunks whose reference is fully active. So localised events label only the
chunks that really are the event, and the rest are labelled by what they
contain. `sequence_scenario` keeps the sequence-level intent.

**`ref_dropout` is load-bearing.** During a dropout the far end is genuinely
silent — `X == 0` **and** `D == 0` — so no model may hallucinate echo removal.
Every current candidate is a joint AEC+NR route and may still suppress `N`, so
`ref == 0 -> output == mic` is **not** a universal gate (that expectation
belonged to the now-retired AEC-only Align-CRUSE route, which targeted
`S+N`). `[dropout] ref_dropout_echo_continues_p` can make
the loudspeaker keep playing while the reference is lost, but that asks the
model to predict an echo from nothing, so it is **0 by default**: raising it
trains hallucination.

Note that `p_ref_dropout` is a per-*sequence* probability; the per-*chunk*
share is far smaller. If the idle term needs more, lengthen the dropouts
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

## Files

| File | Role |
|---|---|
| `aec_dataset.py` | the scenario simulator: nonlinearity, echo path, delay/jitter, SRO, dropout, AGC |
| `manifest.py` | unified/source-disjoint source manifest |
| `gen_aec_dataset.py` | CLI — renders complete sequences to 5-channel WAV chunks |
| `linear_aec.py` | frozen PBFDKF contract and full-sequence materializer |
| `rematerialize_linear_aec.py` | rebuilds the last channel from existing four/five-channel WAVs |
| `pack_aec_dataset.py` | packs those WAVs into `.pt` shards |
| `packed_aec_dataset.py` | `PackedAecDataset`, returning `(stems, meta)` |
| `aec_features.py` | **the shared module the model projects import** |
| `config.example.ini` | every knob, documented |
| `tests/` | the invariants a consumer cannot detect being broken |

`aec_features.py` owns `AecGrid`, `stft`/`istft`, `alpha_from_tau`, `AecStems`,
`STEM_ORDER` and `SequenceChunkSampler`. **⚠ A model project that re-declares
any of these is opting out of the comparison** — the same failure that
`AINR/tests/test_bakeoff_protocol.py` already guards for the NR split, where a
5%-vs-10% divergence meant two models were compared on different corpora.

### The signal grid

16 kHz first target: `n_fft = 512`, `win_len = 512`, `hop_len = 256`,
`n_freqs = 257`, sqrt-Hann (periodic), 50 % overlap (COLA), 62.5 fps. The
48 kHz variant is `sr = 48000, n_fft = 1024, win_len = 1024, hop_len = 512` and
**nothing else** — `AecGrid` derives `n_freqs` and `frame_rate`, and rejects a
hop that is not `win_len/2` or a `win_len` different from `n_fft`.

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
    --input data_aec/all --output data_aec/packed/all
```

⚠ Five of the six trainers run at 16 kHz/512 and one (DeepFilterNet-AENR) runs
at 48 kHz/1024 (`[signal] sr = 48000, n_fft = win_len = 1024, hop_len = 512`
in a SEPARATE `config.ini` -- see config.example.ini's top comment). The two
rates need their OWN `--output` (e.g. `data_aec_16k` / `data_aec_48k`, matching
each trainer's `packed_dir`): generating the second rate into the same
`--output` as the first silently overwrites its `seqs/`/`packed/` content,
since neither directory is namespaced by sample rate.

Layout:

```
data_aec/
  manifest.json                 source-list provenance
  all/meta.json                 run summary + PBFDKF contract
  all/seqs/000000.json          chunk metadata for one parent sequence
  all/seqs/000000_000.wav       5-channel chunk, channels = STEM_ORDER
  packed/all/shard_00000.pt
```

The six trainers read `packed/all` and create the deterministic random chunk
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
version cannot strand a checkpoint. `aec_source_hash` still covers all 87 files.

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

```bash
python3 -m AIAEC.dataset_gen.rematerialize_linear_aec \
    --input data_aec/all \
    --config AIAEC/dataset_gen/config.ini \
    --resume
python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --input data_aec/all --output data_aec/packed/all
```

`--resume` marks a sequence complete only after all five-channel WAVs and its
matching config, manifest and linear-AEC metadata contracts exist. Manifest v1
files do not contain the exact file-to-source-id maps required by v2; rebuild
the manifest and re-render/repack rather than mixing old sidecars or shards.

Generation is deterministic given `--seed`: each sequence is seeded from
`(seed, split, sequence_id)`, so it renders identically regardless of worker
count or ordering, and `--resume` continues exactly. `--hours` resolves to a
fixed sequence list up front, so extending a corpus keeps every sequence it
already had.

**⚠ `--wav-encoding` defaults to `float32`** because the corpus's central
invariant, `mic_preclip == near_speech + local_noise + echo`, is checked at
generation time against the renderer's un-quantised audit tensors (`echo`,
`local_noise` and `mic_preclip` are not among the persisted stems — see "The
five stems" above). Quantising the PERSISTED stems to `int16` would still
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
