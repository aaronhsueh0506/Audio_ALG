# AEC dataset generation

Renders acoustic-echo scenarios as **seven separated stems** and packs them into
`.pt` shards. This is the only AIAEC dataset package. It reuses shared DSP from
the separate `AINR/dataset_gen/` NR generator rather than forking that DSP.

## The seven stems

Every clip is a `(7, T)` tensor whose channel order is fixed and declared in
each shard:

| # | stem | what it is |
|---|---|---|
| 0 | `far_render` | **X** — the far-end signal as the device rendered it, i.e. the AEC reference. Digital and clean: the loudspeaker's distortion happens *downstream* of this tap. |
| 1 | `echo` | **D** — the echo actually present at the mic: X through the loudspeaker nonlinearity, the radiated response, the bulk delay and the room. |
| 2 | `near_speech` | **S** — the near talker at the mic, already through the room RIR. Reverberant on purpose; that reverberation is desired signal. |
| 3 | `near_target` | **S_early** — the same near talker and gain through the early/late-suppressed RIR; used only for DeepVQE's published dereverberation target. |
| 4 | `local_noise` | **N** — ambient noise at the mic. |
| 5 | `mic_preclip` | `S + N + D`, exactly, before any clipping or AGC. |
| 6 | `mic_postclip` | **Y** — what a model actually receives, after capture clipping/AGC. |

The signal model the corpus exists to serve:

```
Y = S + N + D          microphone
X                      far-end reference
D_hat                  optional linear/model echo estimate
E     = Y - D_hat      linear error, by subtraction
R     = D - D_hat      residual echo — emerges, never a target
```

`D_hat` exists only when a frozen linear front-end supplies it. The corpus does
not bake a residual into storage: `model_views.py` maps the stems to the six
candidate contracts. Align-CRUSE targets `S+N`; Align-ULCNet and the two AENR
variants require the real frozen-linear error and target `S`; CAGCRN targets
`S`; DeepVQE-S targets `S_early` because its published task includes
dereverberation. The separated stems keep those task
boundaries auditable instead of silently training every model on one target.

`mic_preclip` and `mic_postclip` are both stored so that clipping/AGC effects
can be separated from echo-path nonlinearity — they are two different
distortions and a model that confuses them will fix the wrong one.

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
| `speaker_id` | the **near** talker (`''` = none). `far_speaker_id` is the other one |
| `noise_id`, `rir_id` | source ids; `rir_id` is `"a\|b"` for an echo-path change |
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
differs from `mic_preclip` **if and only if** one of the two is set.

**⚠ `ser_db` / `snr_db` / `erl_db` are sequence-level.** They describe how the
whole 20–60 s sequence was set up. A single chunk departs from them by a few dB
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
For AEC-only Align-CRUSE the desired signal is `S+N`; joint-NR models may still
suppress `N`, so `ref == 0 -> output == mic` is **not** a universal gate.
`[dropout] ref_dropout_echo_continues_p` can make
the loudspeaker keep playing while the reference is lost, but that asks the
model to predict an echo from nothing, so it is **0 by default**: raising it
trains hallucination.

Note that `p_ref_dropout` is a per-*sequence* probability; the per-*chunk*
share is far smaller. If the idle term needs more, lengthen the dropouts
(`ref_dropout_chunks_max`) rather than adding sequences that are mostly not
dropouts. `near_only` supplies idle chunks too, but only `ref_dropout`
contains the *transition* into and out of idle, which is the hard part.

## Why the split is source-disjoint

`AINR/dataset_gen/loader.py` splits the *generated* corpus — draw a permutation over
finished clips, hold 5% out. That is correct when every clip is an independent
draw, and wrong here.

Two AEC clips rendered from the same speaker, the same echo RIR and the same
loudspeaker are not independent: they share the exact voice, the exact room
response and the exact nonlinearity. Split after generation and both halves of
that pair land on opposite sides of the fence, so validation measures how well
the model memorised a room. The resulting number is high, stable, reproducible
and meaningless.

So `manifest.py` decides the split **before** anything is rendered, over the
source lists: **speaker**-disjoint (which makes speech files disjoint too),
**noise**-disjoint, **room/RIR**-disjoint, **device**-disjoint. The decision is
written to `manifest.json` so that the train run and the val run — separate
invocations — provably use the same one, and `assert_source_disjoint` re-checks
every axis on load.

**⚠ Device disjointness is the aggressive one.** Validation is scored on
loudspeaker nonlinearities the model has never heard, which is the honest
question for a shipped product and materially harder than the usual AEC
benchmark. `[split] device_split = shared` relaxes it — and then the val score
answers a different, easier question, so say so wherever it is reported. The
manifest records which was used.

## Why sequences are long

Parent sequences are 20–60 s, cut into consecutive fixed-length chunks that
share a `sequence_id` and carry an increasing `chunk_index`.

Adaptation from cold, recovery after an echo-path change, and long-term drift
are all invisible inside a 3 s clip. **⚠ A trainer that resets recurrent state
every chunk cannot be shown to fail at any of them** — it will look fine and
ship broken. `SequenceChunkSampler` supplies the ordering needed to prevent
that: each lane of a batch walks one sequence in order, so batch *b+1* holds
the next chunk of the same sequence that batch *b* held. The sampler does
**not** make a stateless model API stateful: the external trainer must either
carry the model's per-lane recurrent/cache state or concatenate consecutive
chunks before `forward`. When a lane starts a new sequence its `chunk_index`
is 0 — that is the reset signal, and `lane_reset_mask` reports it.

```python
from torch.utils.data import DataLoader
from AIAEC.dataset_gen import PackedAecDataset, SequenceChunkSampler, aec_collate

ds = PackedAecDataset('data_aec/packed/train')
sampler = SequenceChunkSampler.from_dataset(ds, n_lanes=8, seed=42)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=aec_collate)
for epoch in range(epochs):
    sampler.set_epoch(epoch)      # reshuffles which sequence sits in which lane
    for stems, meta in loader:
        reset = [m['chunk_index'] == 0 for m in meta]
        ...
```

The packer keeps a sequence's chunks adjacent and in order, and never splits a
sequence across shards.

## Files

| File | Role |
|---|---|
| `aec_dataset.py` | the scenario simulator: nonlinearity, echo path, delay/jitter, SRO, dropout, AGC |
| `manifest.py` | the source-disjoint split, decided before generation |
| `gen_aec_dataset.py` | CLI — renders sequences to 7-channel WAV chunks |
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

```bash
cp AIAEC/dataset_gen/config.example.ini AIAEC/dataset_gen/config.ini
# edit [paths] speech_dir / noise_dir / rir_dir, and [devices] device_ids

# Both runs MUST share --seed and the manifest.
python3 -m AIAEC.dataset_gen.gen_aec_dataset \
    --config AIAEC/dataset_gen/config.ini --output data_aec \
    --hours 40 --split train --workers 4 --seed 42
python3 -m AIAEC.dataset_gen.gen_aec_dataset \
    --config AIAEC/dataset_gen/config.ini --output data_aec \
    --hours 4  --split val   --workers 4 --seed 42

python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --input data_aec/train --output data_aec/packed/train
python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --input data_aec/val --output data_aec/packed/val
```

Layout:

```
data_aec/
  manifest.json                 the split decision, shared by both runs
  train/meta.json               run summary + provenance
  train/seqs/000000.json        chunk metadata for one parent sequence
  train/seqs/000000_000.wav     7-channel chunk, channels = STEM_ORDER
  packed/train/shard_00000.pt
```

Generation is deterministic given `--seed`: each sequence is seeded from
`(seed, split, sequence_id)`, so it renders identically regardless of worker
count or ordering, and `--resume` continues exactly. `--hours` resolves to a
fixed sequence list up front, so extending a corpus keeps every sequence it
already had.

**⚠ `--wav-encoding` defaults to `float32`** because the corpus's central
invariant is `mic_preclip == near_speech + local_noise + echo`. Quantising seven
stems independently to `int16` makes that identity hold only to ~1e-4, and
anything that recomputes one stem from the others inherits the error. `int16`
halves the disk cost and is fine for listening, not for arithmetic.

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
