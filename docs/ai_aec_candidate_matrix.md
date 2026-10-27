# AIAEC current candidate matrix

This is the current implementation map.

## Decision matrix

| Route | Candidate | Public input | Supervised target | Current role |
|---|---|---|---|---|
| end-to-end AEC+RES+NR | `Align_CRUSE` | mic + unaligned far | early near target | selected |
| linear AEC → RES+NR | `Align_ULCNet` | production linear error + far | early near target | selected |
| end-to-end AEC+RES+NR | `DeepVQE_S` | mic + unaligned far | early near target | primary |
| end-to-end AEC+RES+NR | `CAGCRN` | mic + unaligned far | early near target | backup |

⚠ **Every candidate targets the same early/dereverberated `near_target`**
(`MODEL_TASKS` values all end in `_dereverb`). For `Align_ULCNet` and
`CAGCRN` that is a project extension of the published task, not the published
task itself. A checkpoint trained for another target is refused at inference by
`require_checkpoint_model_identity`.

There is no standalone AEC-only candidate in this project.

## Selection rules

- Use a RES+NR candidate only after the same production linear AEC that will be
  frozen at deployment. An oracle residual is rejected by the dataset view.
- `DeepVQE_S` is the primary E2E route and `CAGCRN` the smaller backup.
  Dereverberation does not separate them, since every candidate targets
  `near_target`.

## Common contracts

- Public complex spectra have shape `[batch, time, frequency]`.
- Model grids are `512/512/256 @ 16 kHz` and
  `1024/1024/512 @ 48 kHz`.
- The public forwards are clip-level; explicit per-model cache/state I/O is a
  separate streaming deployment concern -- see
  [`aiaec_streaming_readiness_zh_TW.md`](aiaec_streaming_readiness_zh_TW.md)
  for the per-candidate streaming audit (lookahead, states, blockers).
- AIAEC data is generated only through `AIAEC/dataset_gen/`. Generated WAVs
  carry five stems; packing projects them to the four-channel training contract
  `PACKED_STEM_ORDER` (`far_render`, `mic_postclip`, `linear_error`,
  `near_target`), which `dataset_gen.model_views.build_model_view` maps to each
  model's inputs and target. The reverberant `near_speech` stem stays in the
  WAVs for audit and is not packed.

## Dataset and deployment boundary

The AIAEC generator renders complete stateful scenario sequences, executes one
frozen production Python PBFDKF over each sequence on the context seam
(`enable_res=False`, `return_res_context=True`), appends its `linear_error` as
the dataset's last channel — usually `mic_postclip - D_hat`, but a hop the
filtering-quality analyzer has not cleared and whose residual outweighs the
capture publishes the capture instead — and then
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
    → SRP-PHAT + BF (the sample beam helper `mcab_sample_beam_*`, or the
      application's own beamformer between `mcab_process_front()` and `mcab_fuse()`)
    → one post-beam RES, one mono NR path (separate libraries chained by the integrator)
```

It is not an AIAEC model bake-off surface. See
[`../pipelines/4ch_aec_bf_nr_res/README.md`](../pipelines/4ch_aec_bf_nr_res/README.md).
