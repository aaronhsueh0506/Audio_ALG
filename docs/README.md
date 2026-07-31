# Documentation index

Use this page to distinguish current integration contracts from dated audit
records. Source code and tests win if a dated report disagrees with a current
README.

## Current sources of truth

| Topic | Document |
|---|---|
| Repository scope and signal-grid boundary | [`../README.md`](../README.md) |
| Conventional mono AEC + NR/RES | [`../pipelines/README.md`](../pipelines/README.md) |
| C and caller-owned-pool integration | [`c_user_manual_zh_TW.md`](c_user_manual_zh_TW.md) |
| Frequency-domain gain fusion | [`freq_domain_pipeline_design.md`](freq_domain_pipeline_design.md) |
| Four-channel C/Python AEC / external beamformer seam | [`../pipelines/aec_4ch/README.md`](../pipelines/aec_4ch/README.md) |
| Standalone AINR models | [`../AINR/README.md`](../AINR/README.md) |
| DFN2 cascade/alpha contract | [`../AINR/DeepFilterNet2/README.md`](../AINR/DeepFilterNet2/README.md) |
| DFN3 band-split contract | [`../AINR/DeepFilterNet3/README.md`](../AINR/DeepFilterNet3/README.md) |
| Neural AEC candidates | [`../AIAEC/README.md`](../AIAEC/README.md) |
| AIAEC candidate decision matrix | [`ai_aec_candidate_matrix.md`](ai_aec_candidate_matrix.md) |
| Development / submodules | [`development.md`](development.md) |

`architecture.html` is a visual explanation of the conventional mono path. It
does not describe AINR or AIAEC model internals.

## Dated audits

- [`aiaec_4ch_signal_grid_review_2026_07_30.md`](aiaec_4ch_signal_grid_review_2026_07_30.md)
  records the structural sign-off that introduced the six AIAEC candidates and
  the 4-channel boundary.
- [`archive/ai_aec_candidate_matrix_2026_07_30.md`](archive/ai_aec_candidate_matrix_2026_07_30.md)
  preserves the long pre-refactor AI-AEC research trail. It mentions removed
  prototypes and is not an implementation map.
- [`../pipelines/PLAN_audio_pipeline_api.md`](../pipelines/PLAN_audio_pipeline_api.md)
  is the historical proposal that preceded the implemented
  `audio_pipeline.h`; use it for rationale only.

## Maintenance rule

When code changes:

1. update the nearest component README and its checkpoint/signal-grid contract;
2. update the root summary only if repository boundaries changed;
3. keep dated audits immutable except for a short superseded notice;
4. move obsolete design narratives to `docs/archive/` instead of leaving them
   mixed with current setup instructions.
