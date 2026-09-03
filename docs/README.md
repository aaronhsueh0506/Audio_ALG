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
| AEC / NR / audio_common public API quick reference | [`audio_libraries_c_api_zh_TW.md`](audio_libraries_c_api_zh_TW.md) |
| DEBUG observability: current inventory and proposals | [`debug_observability_zh_TW.md`](debug_observability_zh_TW.md) |
| Board integration of `libaudio_pipeline.a` | [`pipeline_board_integration.md`](pipeline_board_integration.md) |
| Frequency-domain gain fusion | [`freq_domain_pipeline_design.md`](freq_domain_pipeline_design.md) |
| Four-channel C/Python AEC / external beamformer seam | [`../pipelines/4ch_aec_bf_nr_res/README.md`](../pipelines/4ch_aec_bf_nr_res/README.md) |
| Four-channel Align-ULCNet application (direct GSC-spectrum path) | [`../pipelines/4ch_alignulcnet/README.md`](../pipelines/4ch_alignulcnet/README.md) |
| Standalone AINR models | [`../AINR/README.md`](../AINR/README.md) |
| DFN2 cascade/alpha contract | [`../AINR/DeepFilterNet2/README.md`](../AINR/DeepFilterNet2/README.md) |
| Neural AEC candidates | [`../AIAEC/README.md`](../AIAEC/README.md) |
| AIAEC candidate decision matrix | [`ai_aec_candidate_matrix.md`](ai_aec_candidate_matrix.md) |
| PBFDKF + Align-ULCNet embedded streaming proposal | [`align_ulcnet_embedded_streaming_design_zh_TW.md`](align_ulcnet_embedded_streaming_design_zh_TW.md) |
| Align-ULCNet delay/`n`/`D` profile plan | [`align_ulcnet_delay_profile_plan_zh_TW.md`](align_ulcnet_delay_profile_plan_zh_TW.md) |
| AIAEC four-candidate streaming readiness audit | [`aiaec_streaming_readiness_zh_TW.md`](aiaec_streaming_readiness_zh_TW.md) |
| Mono pipeline integration (config/pool/errors) | [`integration_mono_zh_TW.md`](integration_mono_zh_TW.md) |
| 4ch core integration (`4aec_nr_res`) | [`integration_4ch_core_zh_TW.md`](integration_4ch_core_zh_TW.md) |
| 4ch spatial wrapper integration (SRP/GSC) | [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md) |
| Release gate and reference test counts | [`release_checklist.md`](release_checklist.md) |
| Development / submodules | [`development.md`](development.md) |

The offline HTML documentation site lives in [`html/`](html/index.html): one
page per module (overview / API / block diagram / signal swimlane / I-O table /
state table / latency / file paths), plus `conventions.html` for the chain-wide
grid and framing table and `onnx_prepost.html` for the ONNX-boundary,
state-layout-version and PTQ-calibration status. Component pages for AEC, NR
and audio_common live in those repositories' own `docs/html/` and are reached
through sibling relative links.

For integrator-facing C API reference, the site also carries one page per
library -- `aec_c_api.html`, `nr_c_api.html` and `audio_common_c_api.html` in
those repositories' own `docs/html/` -- plus `integration_example.html` here,
which uses the four pipelines as the worked wiring example and records what
remains observable in a release build.

## Maintenance rule

When code changes:

1. update the nearest component README and its checkpoint/signal-grid contract;
2. update the root summary only if repository boundaries changed;
3. keep dated audits immutable except for a short superseded notice;
4. move obsolete design narratives to `docs/archive/` instead of leaving them
   mixed with current setup instructions.
