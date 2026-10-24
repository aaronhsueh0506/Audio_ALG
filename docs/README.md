# Documentation index

Index of the current integration contracts. Source code and tests win if a
document disagrees with them.

## Current sources of truth

| Topic | Document |
|---|---|
| Repository scope and signal-grid boundary | [`../README.md`](../README.md) |
| Conventional mono AEC + NR/RES | [`../pipelines/README.md`](../pipelines/README.md) |
| C and caller-owned-pool integration | [`c_user_manual_zh_TW.md`](c_user_manual_zh_TW.md) |
| AEC / NR / audiolib public API quick reference | [`audio_libraries_c_api_zh_TW.md`](audio_libraries_c_api_zh_TW.md) |
| DEBUG observability: current inventory and proposals | [`debug_observability_zh_TW.md`](debug_observability_zh_TW.md) |
| Board integration (separate libraries, the one 4-channel unit's descriptor) | [`pipeline_board_integration.md`](pipeline_board_integration.md) |
| Frequency-domain gain fusion | [`freq_domain_pipeline_design.md`](freq_domain_pipeline_design.md) |
| Four-channel front end (external beamformer seam), the linear AEC + BF unit and the Python reference | [`../pipelines/4ch_aec_bf_nr_res/README.md`](../pipelines/4ch_aec_bf_nr_res/README.md) |
| Four-channel Align-ULCNet application (the unit's BF spectrum goes to the model directly) | [`../pipelines/4ch_alignulcnet/README.md`](../pipelines/4ch_alignulcnet/README.md) |
| Mono AEC + RES + DeepFilterNet2 application | [`../pipelines/mono_aec_dfn_res/README.md`](../pipelines/mono_aec_dfn_res/README.md) |
| Four-channel unit + application-owned libaec `AecRes` + DeepFilterNet2 application | [`../pipelines/4ch_aec_bf_dfn_res/README.md`](../pipelines/4ch_aec_bf_dfn_res/README.md) |
| Standalone AINR models | [`../AINR/README.md`](../AINR/README.md) |
| DFN2 cascade/alpha contract | [`../AINR/DeepFilterNet2/README.md`](../AINR/DeepFilterNet2/README.md) |
| DFN3 raw-low / masked-high comparison contract | [`../AINR/DeepFilterNet3/README.md`](../AINR/DeepFilterNet3/README.md) |
| Neural AEC candidates | [`../AIAEC/README.md`](../AIAEC/README.md) |
| AIAEC candidate decision matrix | [`ai_aec_candidate_matrix.md`](ai_aec_candidate_matrix.md) |
| PBFDKF + Align-ULCNet embedded streaming proposal | [`align_ulcnet_embedded_streaming_design_zh_TW.md`](align_ulcnet_embedded_streaming_design_zh_TW.md) |
| AIAEC four-candidate streaming readiness audit | [`aiaec_streaming_readiness_zh_TW.md`](aiaec_streaming_readiness_zh_TW.md) |
| Mono integration (sequence diagram, calls, external/internal parameters, config/pool/errors) | [`integration_mono_zh_TW.md`](integration_mono_zh_TW.md) |
| 4ch core integration (`4aec_nr_res`; own beamformer; sequence, calls, parameters) | [`integration_4ch_core_zh_TW.md`](integration_4ch_core_zh_TW.md) |
| 4ch linear AEC + BF unit integration (`AudioPipeline4Ch`, up to `fuse()`; the application creates its own libaec `AecRes`; sequence, calls, parameters) | [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md) |
| Release gate and reference test targets | [`release_checklist.md`](release_checklist.md) |
| Development / submodules | [`development.md`](development.md) |

The offline HTML documentation site lives in [`html/`](html/index.html): one
page per module (overview / API / block diagram / signal swimlane / I-O table /
state table / latency / file paths), plus `conventions.html` for the chain-wide
grid, framing and layout/carve-version tables (the per-stage latency budget is split out into
`conventions_latency.html`) and `onnx_prepost.html` for the ONNX-boundary,
state-layout-version and PTQ-calibration status. Component pages live in the
component repositories' own `docs/html/` and are reached through sibling relative
links (`SE/{AEC,NR,audiolib,Audio_ALG}` side by side):

| Repository | Directory | Pages |
|---|---|---|
| AEC | `AEC/docs/html/` | `aec.html`, `aec_detail.html`, `aec_usage.html`, `aec_sequence.html`, `aec_params.html`, `aec_params_internal.html`, `aec_c_api.html`, `aec_c_api_params.html`, `aec_res.html` |
| NR | `NR/docs/html/` | `nr.html`, `nr_c_api.html`, `nr_c_api_params.html` |
| audiolib common | `audiolib/common/docs/html/` | `audio_common.html`, `audio_common_detail.html`, `audio_common_params.html`, `audio_common_c_api.html` |
| audiolib spatial chain | `audiolib/docs/html/` | `spatial_chain.html`, `doa_srp.html`, `bf.html`, `vad.html` |

Pipeline pages in `html/`: `pipeline_mono.html` (with `pipeline_mono_params.html` and
`pipeline_mono_detail.html`), `pipeline_4ch.html` (with `pipeline_4ch_params.html`; runtime-strength and
per-stage timing detail in `pipeline_4ch_detail.html`), `pipeline_ulcnet_mono.html`,
`pipeline_ulcnet_4ch.html` (shared adapter and identity-reprime detail in
`pipeline_ulcnet_detail.html`, call order / sequence / parameters in `pipeline_ulcnet_usage.html`,
`pipeline_ulcnet_sequence.html` and `pipeline_ulcnet_params.html`), plus
`pipeline_dfn_mono.html` and `pipeline_dfn_4ch.html` for the two DeepFilterNet2
variants (shared entry-point, rate-bridge and test detail in
`pipeline_dfn_detail.html`, with `pipeline_dfn_usage.html`, `pipeline_dfn_sequence.html` and
`pipeline_dfn_params.html`; the host runs at the product rate -- mono 8/16/48 kHz, 4ch 16/48 kHz,
16 kHz by default -- and only the DFN2 stage runs at 48 kHz / 1024 / 512,
reached through `pipelines/dfn_rate_bridge.{h,c}`; the shared post-RES stage
`pipelines/dfn_res_stage.{h,c}` estimates the model from the pre-RES spectrum
and applies it to the post-RES one). Model pages: `ainr_dfn2.html` (pool, callback and build detail in
`ainr_dfn2_detail.html`), `ainr_dfn3.html`, `ainr_gtcrn.html`, `ainr_rnnoise_erb.html`, each with a
usage page (`ainr_dfn2_usage.html`, `ainr_dfn3_usage.html`, `ainr_gtcrn_usage.html`,
`ainr_rnnoise_erb_usage.html`), and the four AIAEC pages `aiaec_align_ulcnet.html`,
`aiaec_align_cruse.html`, `aiaec_deepvqe_s.html` and `aiaec_cagcrn.html`, each with a usage page
(`aiaec_align_ulcnet_usage.html`, `aiaec_align_cruse_usage.html`, `aiaec_deepvqe_s_usage.html`,
`aiaec_cagcrn_usage.html`; Align-ULCNet far-branch, PTQ and layout detail in
`aiaec_align_ulcnet_detail.html`).

Who owns the residual-echo suppressor (RES):

- **mono**: the AEC computes the RES gain inside `aec_process_context()` and hands it over as
  `frame.g_res_ctx` through `se_frame_from_aec()`; its strength is set with `aec_set_preset()`.
  The sample only decides whether to apply it. A pure-AEC sample may call the full `aec_process()`
  (RES and the final synthesis in one call).
- **4ch**: the unit `AudioPipeline4Ch` (shared delay aligner, four AEC lanes, SRP-PHAT, BF, `fuse()`)
  has no RES; `fuse()` leaves `frame->g_res_ctx` NULL, like the core. An application that wants RES creates
  it in its own init with `aec_res_get_mem_size()` / `aec_res_init()` (config from
  `aec_config_from_preset()`), calls `se_res_stage_gain()` (`pipelines/common/se_stage.h`) once per hop,
  sets strength with `aec_res_set_preset()` and clears it with `aec_res_reset()`. The 4ch Align-ULCNet
  application runs the unit with `core.enable_post = 0` and has no RES.

Current layout versions: `AUDIO_PIPELINE_4CH_LAYOUT_VERSION = 21` (unit) and
`FOUR_AEC_NR_RES_LAYOUT_VERSION = 24` (core); the full table is in `html/conventions.html`.

For integrator-facing C API reference, the site also carries one page per
library -- `aec_c_api.html`, `nr_c_api.html` and `audio_common_c_api.html` in
those repositories' own `docs/html/` -- plus `integration_example.html` here,
which uses the pipelines as the worked wiring example (the release-build
observability detail is in `integration_example_detail.html`).

For a reader who wants the call order, the sequence diagram and the external/internal parameter
tables without the full Markdown manuals, components and pipelines carry up to three kinds of
pages next to their overview page (where they exist): `*_usage.html` (call order and a minimal C example),
`*_sequence.html` (init, steady-state hop, reset and refused-call lifelines, plus the hop swimlane
where the component has latency) and `*_params.html` (external parameters a third party may tune
versus internal ones that are not recommended to open, and the public versus internal functions).
Here they are `pipeline_*_params.html`, `pipeline_dfn_usage.html` / `pipeline_dfn_sequence.html`,
`pipeline_ulcnet_usage.html` / `pipeline_ulcnet_sequence.html`, `integration_sequence.html`
(init, hop and reset lifelines of the board path) and `ainr_*_usage.html` / `aiaec_*_usage.html`;
in the component repositories they are `aec_usage.html`, `aec_sequence.html`, `aec_params.html`,
`aec_params_internal.html`, `nr_c_api_params.html`, `audio_common_params.html` and the per-module
spatial pages above. The whole-repository overview pages are `architecture.html`,
`architecture_aec.html` and `architecture_detail.html` (in this directory, next to `html/`).
The Markdown manuals above remain the field-by-field reference; section 0 of each
integration manual is the one-screen version of the same three topics.

## Maintenance rule

When code changes:

1. update the nearest component README and its checkpoint/signal-grid contract;
2. update the root summary only if repository boundaries changed;
3. move obsolete design narratives to `docs/archive/` instead of leaving them
   mixed with current setup instructions.
