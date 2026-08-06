# AIAEC current candidate matrix

This is the current implementation map. Superseded research notes are retained
in Git history rather than treated as implementation documentation.

## Decision matrix

| Route | Candidate | Public input | Supervised target | Current role |
|---|---|---|---|---|
| end-to-end AEC+RES+NR | `Align_CRUSE` | mic + unaligned far | early near target | selected |
| linear AEC → RES+NR | `Align_ULCNet` | production linear error + far | clean reverberant near | paper reference |
| linear AEC → RES+NR | `GTCRN_AENR` | production linear error + far | clean reverberant near | project variant |
| linear AEC → RES+NR | `DeepFilterNet_AENR` | independently normalized error/far DFN features | clean reverberant near | project variant |
| end-to-end AEC+RES+NR | `DeepVQE_S` | mic + unaligned far | early near target | primary |
| end-to-end AEC+RES+NR | `CAGCRN` | mic + unaligned far | clean reverberant near | backup |

`Align_CRUSE` previously ran its own "direct AEC/RES, preserve local noise"
route (target: near speech + local noise, untouched). That route was retired
and folded into the end-to-end AEC+RES+NR task above -- there is no more
standalone AEC-only candidate in this project.

No current model lives under `AINR/AECNet`, `AINR/PostFilter`, or
`AINR/JointAECNR`. Those prototype names occur only in the archived design
record.

## Selection rules

- Use a RES+NR candidate only after the same production linear AEC that will be
  frozen at deployment. An oracle residual is rejected by the dataset view.
- Treat `GTCRN_AENR` and `DeepFilterNet_AENR` as project variants, not published
  author AEC models.
- Use `DeepVQE_S` when dereverberation is part of the target. `CAGCRN` is the
  smaller backup E2E route.

## Common contracts

- Public complex spectra have shape `[batch, time, frequency]`.
- Model grids are `512/512/256 @ 16 kHz` and
  `1024/1024/512 @ 48 kHz`.
- `GTCRN_AENR` stays locked to its original 16 kHz grid.
- The public forwards are clip-level; explicit per-model cache/state I/O is a
  separate streaming deployment concern.
- AIAEC data is generated only through `AIAEC/dataset_gen/`. Five stored stems
  are mapped to model inputs and targets by
  `dataset_gen.model_views.build_model_view`.

## DeepFilterNet-AENR baseline

`DeepFilterNet_AENR` follows the current local DFN2 output graph:

```text
full-band ERB mask
    → deep filter only the masked low bins
    → learned sigmoid-alpha residual blend in those low bins
```

The AENR conditioners are initialized as an error-only pass-through. A
standalone checkpoint is a valid initialization only when its DFN2
`MODEL_VERSION`, feature contract, grid, and tensor shapes all match. The
preserved DFN3 band-split checkpoint is not compatible.

## Dataset and deployment boundary

The AIAEC generator renders complete stateful scenario sequences, executes one
frozen production Python PBFDKF over each sequence, appends its
`linear_error = mic_postclip - D_hat` as the dataset's last channel, and then
cuts 10-second chunks. RES+NR trainers read this stored channel directly. File/stream
inference restores the same PBFDKF contract from the checkpoint and runs the
frontend continuously before the neural model.

The selected train/validation protocol is a deterministic random split over
individual packed chunks. Chunks from the same sequence, speaker, or RIR may
straddle; source-generalisation must be evaluated on a separate held-out set.

The conventional 4-channel path remains:

```text
one shared delay matcher
    → four linear adaptive filters
    → externally owned SRP-PHAT/GSC
    → one mono NR/RES path
```

It is not a six-candidate AIAEC bake-off surface. See
[`../pipelines/4ch_pipelines/README.md`](../pipelines/4ch_pipelines/README.md).
