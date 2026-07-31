# AIAEC model candidates

Neural AEC/AENR research surface. Every directory names the paper/base model
it represents; the removed generic `AECNet`, `PostFilter`, and `JointAECNR`
prototype paths are not public APIs.

| Route | Model | Public inputs | Target | Status |
|---|---|---|---|---|
| direct AEC/RES, preserve noise | `Align_CRUSE` | mic + unaligned far | near + local noise | selected |
| linear AEC -> RES+NR | `Align_ULCNet` | linear error + far | clean near | paper reference |
| linear AEC -> RES+NR | `GTCRN_AENR` | linear error + far | clean near | project variant |
| linear AEC -> RES+NR | `DeepFilterNet_AENR` | conditioned DFN features | clean near | project variant |
| end-to-end AEC+RES+NR | `DeepVQE_S` | mic + unaligned far | early near (dereverb) | primary |
| end-to-end AEC+RES+NR | `CAGCRN` | mic + unaligned far | clean near | backup |

All public complex spectra use `[batch,time,frequency]`. The project signal grids
are zero-padding-free, 50%-overlap `FFT/window/hop = 512/512/256 @ 16 kHz` and
`1024/1024/512 @ 48 kHz`. GTCRN-AENR remains locked to its original 16 kHz
grid; the other reconstructed/project variants accept both grids. A model's
README distinguishes published facts from reconstruction choices.

These are AIAEC model grids, not the conventional `pipelines/` 20 ms / 10 ms
grid. A model boundary must resample/reframe explicitly; no caller should infer
compatibility from a shared FFT size alone.

`dataset_gen/` is the one AIAEC dataset implementation and public import/CLI
path. It renders 3-second chunks inside long stateful scenario sequences and
stores seven lossless stems, including separate full-RIR and
early-RIR near-speech targets. `model_views.build_model_view` is the
single mapping from those stems to each candidate. RES+NR views require the
actual frozen production linear AEC and deliberately reject an oracle residual.

The common public forwards are clip-level APIs. `SequenceChunkSampler` preserves
long-sequence ordering, but does not manufacture streaming state: a trainer that
wants 20–60 s continuity must concatenate adjacent 3-second chunks before the
forward call, or add model-specific recurrent/convolution-cache state I/O for
the eventual frame-streaming deployment.

## DeepFilterNet-AENR note

`DeepFilterNet_AENR` now inherits the local
[`AINR/DeepFilterNet2`](../AINR/DeepFilterNet2/README.md) cascade/alpha graph:
full-band ERB mask, low-band deep filter on the masked spectrum, then a learned
alpha residual mix. Its two conditioners start as an exact error-only
pass-through.

A standalone initialization is valid only when the DFN2 v6 model contract,
feature contract, grid, and shapes match. The preserved DFN3 v5 band-split
checkpoint is not compatible.

## Navigation and tests

- [`dataset_gen/README.md`](dataset_gen/README.md): seven-stem scenario data and
  per-model views.
- [`../docs/ai_aec_candidate_matrix.md`](../docs/ai_aec_candidate_matrix.md):
  current selection and deployment rules.
- [`../pipelines/aec_4ch/README.md`](../pipelines/aec_4ch/README.md): separate
  conventional four-channel integration boundary.

From the repository root:

```bash
python3 -m pytest AIAEC/tests
```
