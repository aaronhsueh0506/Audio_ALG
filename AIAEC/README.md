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
not the capture rate. The two inputs must cover the same duration; the scripts
reject a mismatch instead of silently truncating an unaligned echo reference.

`--packed-dir` overrides `[data] packed_dir`; omit it to use the config value.
`--gpu N` selects `cuda:N` and takes precedence over `--device`. `--mmap`
keeps packed tensors disk-backed to reduce host RAM use.

## Navigation and tests

- [`dataset_gen/README.md`](dataset_gen/README.md): five-channel WAV generation,
  four-channel packed training data, and
  per-model views.
- [`../docs/ai_aec_candidate_matrix.md`](../docs/ai_aec_candidate_matrix.md):
  current selection and deployment rules.
- [`../pipelines/4ch_pipelines/README.md`](../pipelines/4ch_pipelines/README.md): separate
  conventional four-channel integration boundary.

From the repository root:

```bash
python3 -m pytest AIAEC/tests
```
