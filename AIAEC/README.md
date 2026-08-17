# AIAEC model candidates

Neural AEC/AENR research surface. Every directory names the paper/base model
it represents; the removed generic `AECNet`, `PostFilter`, and `JointAECNR`
prototype paths are not public APIs.

Install `AIAEC/requirements.txt` before generating data or training.  Its
TorchAudio `<2.9` upper bound is part of the dataset contract: TorchAudio 2.9
moved WAV I/O to TorchCodec and ignores the requested `PCM_F`/32-bit save
arguments.

| Route | Model | Public inputs | Target | Status |
|---|---|---|---|---|
| end-to-end AEC+RES+NR | `Align_CRUSE` | mic + unaligned far | early near (dereverb) | selected |
| linear AEC -> RES+NR | `Align_ULCNet` | linear error + far | early near (dereverb) | paper reference |
| linear AEC -> RES+NR | `GTCRN_AENR` | linear error + far | early near (dereverb) | project variant |
| linear AEC -> RES+NR | `DeepFilterNet_AENR` | conditioned DFN features | early near (dereverb) | project variant |
| end-to-end AEC+RES+NR | `DeepVQE_S` | mic + unaligned far | early near (dereverb) | primary |
| end-to-end AEC+RES+NR | `CAGCRN` | mic + unaligned far | early near (dereverb) | backup |

Align-CRUSE previously ran its own AEC-only, noise-preserving route (target
`near_speech + local_noise`); that route was retired and folded into the
end-to-end AEC+RES+NR task above, so there is no more AEC-only candidate.

All public complex spectra use `[batch,time,frequency]`. The project signal grids
are zero-padding-free, 50%-overlap `FFT/window/hop = 512/512/256 @ 16 kHz` and
`1024/1024/512 @ 48 kHz`. GTCRN-AENR remains locked to its original 16 kHz
grid; the other reconstructed/project variants accept both grids. A model's
README distinguishes published facts from reconstruction choices.

These are AIAEC model grids, not an implicit alias of a conventional
`pipelines/` instance. Both surfaces now use zero-padding-free 50%-overlap
grids, but a model boundary must still validate rate/frame/hop and checkpoint
contract explicitly; a shared FFT size alone does not establish compatibility.

`dataset_gen/` is the one AIAEC dataset implementation and public import/CLI
path. It renders complete stateful scenarios, runs the frozen Python PBFDKF
once over each complete sequence, and only then cuts the result into
10-second WAV chunks. Generated WAVs retain five lossless stems:
`far_render`, `near_speech`, `near_target`, `mic_postclip`, and
`linear_error`. Packing projects them to the four-channel training contract
`far_render`, `mic_postclip`, `linear_error`, `near_target`; the reverberant
`near_speech` audit stem is not copied into `.pt`. `linear_error` is
`E = mic_postclip - D_hat`, not oracle residual echo. Every candidate targets
the same denoised, echo-free early near speech; `model_views.build_model_view`
is the single mapping from the four packed stems to each model.

All trainers use a deterministic random split over chunks. Train batches
shuffle every epoch and validation batches do not. Chunks from one parent
sequence, speaker, or RIR are intentionally allowed on both sides of this
split; use a separately generated source-disjoint or real-recording test set
when measuring generalisation.

## DeepFilterNet-AENR note

`DeepFilterNet_AENR` now inherits the local
[`AINR/DeepFilterNet2`](../AINR/DeepFilterNet2/README.md) cascade/alpha graph:
full-band ERB mask, low-band deep filter on the masked spectrum, then a learned
alpha residual mix. Its two conditioners start as an exact error-only
pass-through.

A standalone initialization is valid only when the DFN2 v6 model contract,
feature contract, grid, and shapes match. The preserved DFN3 v5 band-split
checkpoint is not compatible.

## Training, config and inference

Every candidate directory owns a `train.py` / `config.ini` / `denoise.py`
trio, mirroring `AINR/`'s per-project layout -- each file's own top-of-file
comment documents its exact usage, config sections and CLI. What they do NOT
each keep a private copy of (seeding, checkpoint contracts, the NaN-halt
guards, the shared loss, the inference-only frozen linear-AEC frontend, and
the train/val split itself) lives
in [`training_common.py`](training_common.py) -- see its own top-of-file
docstring before adding a seventh copy of any of it into a candidate.

Every candidate's `[data]` points at one packed corpus (`--split all`) and
holds out `val_fraction` at load time with
`training_common.split_dataset_by_sample`. The checkpoint stores the dataset
fingerprint, complete train/validation indices, split seed/fraction, and PBFDKF
contract; resume refuses a different dataset, split, frontend, or signal grid.
"Different frontend" means a different `aec_behavior_hash` (the AEC's code
identity); a comment-only edit under `lib/aec/python` moves the recorded
provenance but not that hash, so it does not strand a checkpoint.

The one exception is `ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` in
`dataset_gen/linear_aec.py`: today two explicit `recorded -> current` hash
pairs, each admitted only on measured byte-identical `linear_error` evidence
plus a control proving the same harness can fail. It applies only when the
behaviour hash is the sole differing field, is one-way (a newer-hash artifact
against an older build stays refused) and single-hop (the table is never read
transitively). Hitting an entry warns and proceeds: existing shards need no
regeneration and trained checkpoints need no retraining. `dataset_gen/README.md`
documents the admission evidence; an unlisted hash is refused exactly as before.

```bash
cd AIAEC/Align_CRUSE   # or any of the other five directories
python3 train.py --config config.ini \
  --packed-dir /path/to/packed/all --gpu 0 --mmap
python3 denoise.py output/align_cruse_best.pth mic.wav far.wav out.wav
```

Inference accepts mono microphone/reference files at rates different from the
checkpoint. Both inputs are converted together with a band-limited Kaiser-sinc
resampler **before** PBFDKF/STFT/model processing, so a 48 kHz capture sent to a
16 kHz checkpoint follows the same 16 kHz frontend used for training. The
output WAV remains at the checkpoint/model sample rate (16 kHz in that case),
not the capture rate. The two inputs must share a source rate and start time.
The microphone owns the output timeline: when only the tails differ, the far-end
reference is zero-padded or cropped to the microphone length before resampling,
with a warning. This matches AEC evaluation sets whose loopback/reference ends
before the microphone recording; it does not estimate or correct a start offset.

`--packed-dir` overrides `[data] packed_dir`; omit it to use the config value.
`--gpu N` selects `cuda:N` and takes precedence over `--device`. `--mmap`
keeps packed tensors disk-backed to reduce host RAM use.

## ONNX and embedded pre/post-processing

Install `requirements-export.txt` together with the normal requirements.
Production exports are stateless accelerator graphs: all recurrent,
convolution, attention and delay-ring state appears as ordinary graph I/O.
Real/imag pairs are used instead of ONNX complex tensors.

```bash
python3 AIAEC/export_streaming_onnx.py DeepVQE_S \
  --checkpoint /path/to/checkpoint.pth --max-delay-frames 8 \
  --output output/deepvqe_stream.onnx --verify
python3 AIAEC/export_streaming_calibration.py DeepVQE_S \
  --checkpoint /path/to/checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 256 --max-delay-frames 8 \
  --output output/calibration.npz
```

For end-to-end candidates, `--primary-dir` contains microphone WAVs; for
RES+NR candidates it contains materialized linear-error WAVs. Relative WAV
paths in the primary and far directories must match. `--frames` on the
calibration tool means the number of one-frame invocations to capture; it is
not an ONNX block dimension. For alignment models, `--max-delay-frames` can
reduce D at export without changing weight shapes or retraining; use the same
D for calibration and the deployed CPU-side state allocation.

Align-ULCNet calibration intentionally accepts the training-domain
`linear_error + raw_far` pair. The report records
`calibration_far_input_mode=raw_far` separately from the fixed production
`deployment_far_input_mode=aligned_far`; calibration does not claim that the
two signal seams are identical. D still has to match the exported graph
because it changes K/V-history tensor shapes and host state RAM, even though D
does not change learned weight shapes.

Align-CRUSE's `state_align_score_sum` is an undecayed cumulative state. Its
calibration frames must therefore come from one uninterrupted recording (the
tool rejects a set that can only reach `--frames` by joining reset clips), and
the generated metadata marks `score_sum` as `float32_no_ptq` and its frame
counter as `int64_no_ptq`. If the target accelerator cannot preserve those
state dtypes, this Align-CRUSE graph is not a valid integer-only deployment
boundary; collecting more short clips cannot fix the unbounded-state issue.

Tensors named in `state_precision_policy` are recorded WITHOUT `min`/`max`/
`p001`/`p999`; their `inputs` entry carries a `precision` marker instead. A
range for an undecayed accumulator only describes how long the capture ran,
and float percentiles over the int64 frame counter describe nothing at all —
publishing either would invite a quantizer to use them. The streaming report
also records `state_layout_version` beside `max_delay_frames`, so a
calibration set can be checked against the graph it was recorded on before
either is trusted. Only Align-ULCNet's state layout is versioned today (its
exporter writes `3`, pinned to `ULCNET_MODEL_IO_LAYOUT_VERSION`); the shared
exporter does not write that key, so the report carries `null` for the other
five and only `max_delay_frames` cross-checks. D itself must agree across
three artifacts — the exported graph, the calibration report, and the C
descriptor the board initialises from.

Align-ULCNet additionally has a true stateless-accelerator, one-frame export:

```bash
python3 AIAEC/Align_ULCNet/export_streaming_onnx.py \
  --checkpoint /path/to/checkpoint.pth --max-delay-frames 8 \
  --output output/align_ulcnet_d8_stream.onnx --verify
```

Its ONNX inputs include CPU-owned K/V history, four score-history frames and
two temporal-GRU hidden tensors. Outputs contain enhanced RI spectrum and
only the new K/V/logit entries plus next GRU hidden tensors. CPU storage and
ring updates are provided by `Align_ULCNet/ulcnet_model_io.c/.h`; no state is
retained inside the accelerator. See `Align_ULCNet/README.md` for the complete
CPU/model flowchart and tensor shapes.

Align-CRUSE, DeepVQE-S, CAGCRN and GTCRN-AENR consume one new STFT frame per
call. DeepFilterNet-AENR consumes four three-frame feature windows
(`[t-1,t,t+1]` for error/far ERB/complex features) and returns one set of DFN
heads. Its additional four-frame DF-path cache and every model's recurrent or
attention state are explicit input/output tensors. The export JSON records
the exact `state_handoff` mapping and dtype; the host must return each state
output as its paired input on the next invocation.

`AIAEC/export_onnx.py` and `export_calibration.py` are retained only as legacy
fixed-block research tools. They reset temporal state at each block and are
not the production accelerator boundary. Their report records the same
`calibration_far_input_mode`/`deployment_far_input_mode` pair as the streaming
one: the far seam a set was recorded on is a property of the recording, not of
which exporter produced it.

Only three candidates contain ERB maps. Export their checkpoint-exact tables
with:

```bash
python3 AIAEC/export_erb_matrix.py CAGCRN \
  --checkpoint /path/to/checkpoint.pth --output-dir output/erb --format all
```

The C host boundary is built with `make -C AIAEC` and produces
`AIAEC/build/libaiaec_prepost.a`. `SIMD=0` selects the shared scalar reference.
The current mapping is:

| Model | Accelerator output | Host composition |
|---|---|---|
| Align-CRUSE | real mask | `aiaec_apply_real_mask` |
| Align-ULCNet fixed-block/debug | compressed-domain complex mask | `aiaec_apply_ulcnet_compressed_mask` |
| Align-ULCNet one-frame streaming | enhanced RI + delta state | WOLA; `ulcnet_model_io_commit` updates CPU state |
| CAGCRN / GTCRN-AENR | complex mask | `aiaec_apply_complex_mask` |
| DeepVQE-S | 3x3 complex CCM taps | `deepvqe_ccm_process` |
| DeepFilterNet-AENR | ERB mask, DF coefficients, alpha | `dfn_aenr_compose_stream` |

`aiaec_process.c/.h` and `DeepVQE_S/deepvqe_process.c/.h` implement the
current 16 kHz `512/512/256` centered STFT/WOLA path. The DFN-AENR wrapper is
separate and uses the current 48 kHz `1024/1024/512` grid. Its centered,
normalized STFT/WOLA matches AIAEC training and `forward_stream()`; it reuses
DFN2's feature normalization and head composition, but intentionally does not
reuse AINR DFN2's zero-padded `center=False` analysis. Generated ERB matrices
must match the checkpoint and replace, not supplement, any compiled default
table.

## Navigation and tests

- [`dataset_gen/README.md`](dataset_gen/README.md): five-channel WAV generation,
  four-channel packed training data, and
  per-model views.
- [`../docs/ai_aec_candidate_matrix.md`](../docs/ai_aec_candidate_matrix.md):
  current selection and deployment rules.
- [`../pipelines/4ch_aec_bf_nr_res/README.md`](../pipelines/4ch_aec_bf_nr_res/README.md): separate
  conventional four-channel integration boundary.

From the repository root:

```bash
python3 -m pytest AIAEC/tests
```

## Local test assets (not in version control)

`AIAEC/wav_testset/` is a local clone of the Align-ULCNet paper's listening
demo page (`github.com/fhgainr/alignulcnet-aenr`), used as real-recording
test input (e.g. the FST far-end clips). It is gitignored: it carries its
own nested `.git` and the sample licensing is not established for
redistribution. Re-clone it locally when a machine needs these inputs.
