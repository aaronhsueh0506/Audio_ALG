# C Pipeline: linear AEC + echo-aware NR/RES gain fusion

## Contents <!-- split-toc -->
- [C Pipeline: linear AEC + echo-aware NR/RES gain fusion](README.md#c-pipeline-linear-aec--echo-aware-nrres-gain-fusion) <!-- split-toc -->
  - [ARM target selection](README.md#arm-target-selection) <!-- split-toc -->
  - [Architecture](README.md#architecture) <!-- split-toc -->
  - [Modules](README.md#modules) <!-- split-toc -->
  - [Application layout](README.md#application-layout) <!-- split-toc -->
  - [Call sequence, public/internal functions, parameters at a glance](README.md#call-sequence-publicinternal-functions-parameters-at-a-glance) <!-- split-toc -->
  - [Parameter Alignment](README.md#parameter-alignment) <!-- split-toc -->
  - [Four-channel sample driver](README.md#four-channel-sample-driver) <!-- split-toc -->
  - [Latency & Performance](README_01_latency_call_flow.md#latency--performance) <!-- split-toc -->
    - [Memory Budget](README_01_latency_call_flow.md#memory-budget) <!-- split-toc -->
  - [Call flow](README_01_latency_call_flow.md#call-flow) <!-- split-toc -->
    - [How do I chain a new frequency-domain stage?](README_01_latency_call_flow.md#how-do-i-chain-a-new-frequency-domain-stage) <!-- split-toc -->
    - [Where do I call the inverse transform?](README_01_latency_call_flow.md#where-do-i-call-the-inverse-transform) <!-- split-toc -->
    - [What memory do I provide?](README_01_latency_call_flow.md#what-memory-do-i-provide) <!-- split-toc -->
    - [How do I run the NN stage on the NPU?](README_01_latency_call_flow.md#how-do-i-run-the-nn-stage-on-the-npu) <!-- split-toc -->
    - [How do the 4-channel applications split the hop?](README_01_latency_call_flow.md#how-do-the-4-channel-applications-split-the-hop) <!-- split-toc -->
    - [What is in the frame?](README_01_latency_call_flow.md#what-is-in-the-frame) <!-- split-toc -->
  - [NR OLA Delay](README_01_latency_call_flow.md#nr-ola-delay) <!-- split-toc -->
  - [Build](README_02_build_debug_board.md#build) <!-- split-toc -->
  - [Debugging & Performance Flags](README_02_build_debug_board.md#debugging--performance-flags) <!-- split-toc -->
    - [`--debug`](README_02_build_debug_board.md#--debug) <!-- split-toc -->
    - [Delay-estimator duty-cycling (opt-in)](README_02_build_debug_board.md#delay-estimator-duty-cycling-opt-in) <!-- split-toc -->
  - [Two Versions](README_02_build_debug_board.md#two-versions) <!-- split-toc -->
    - [Version A: malloc (`mono_aec_nr_res/main.c`)](README_02_build_debug_board.md#version-a-malloc-mono_aec_nr_resmainc) <!-- split-toc -->
    - [Version B: static memory (`mono_aec_nr_res/static_main.c`)](README_02_build_debug_board.md#version-b-static-memory-mono_aec_nr_resstatic_mainc) <!-- split-toc -->
  - [Board Integration](README_02_build_debug_board.md#board-integration) <!-- split-toc -->
  - [Tunable Parameters](README_03_tuning_troubleshooting.md#tunable-parameters) <!-- split-toc -->
    - [AEC (`AecConfig`, see `aec.h`)](README_03_tuning_troubleshooting.md#aec-aecconfig-see-aech) <!-- split-toc -->
    - [Runtime strength control（不重建、不動 pool）](README_03_tuning_troubleshooting.md#runtime-strength-control不重建不動-pool) <!-- split-toc -->
    - [NR (`AnrConfig`, see `anr_types.h`)](README_03_tuning_troubleshooting.md#nr-anrconfig-see-anr_typesh) <!-- split-toc -->
  - [Troubleshooting & Tuning Guide](README_03_tuning_troubleshooting.md#troubleshooting--tuning-guide) <!-- split-toc -->
    - [External remedies (AEC, NR, pipeline)](README_03_tuning_troubleshooting.md#external-remedies-aec-nr-pipeline) <!-- split-toc -->
    - [Internal remedies (not recommended to open)](README_03_tuning_troubleshooting.md#internal-remedies-not-recommended-to-open) <!-- split-toc -->
    - [Verification](README_03_tuning_troubleshooting.md#verification) <!-- split-toc -->
## ARM target selection

C deployment supports both ARM EABI32 and AArch64 builds. The target compiler's
predefined macros select the path: AArch64 uses the guarded NEON model kernels;
ARM32 uses their existing scalar implementations. `SIMD=0` still forces scalar.
The shared FFT independently selects its available target implementation; select
the board compiler and ABI flags rather than editing source guards. Signal grids,
model state tensors and latency do not change with pointer width.

For ARM32 NEON use `EXTRA_CFLAGS=-mfpu=neon-vfpv4` (`-mfpu=neon-fp-armv8` on
AArch32 ARMv8 cores), not plain `-mfpu=neon`: without hardware FMA each explicit
`fmaf()` in the AEC, NR and shared kernels becomes a libm call (same bits, much
slower). ARMv7 Advanced SIMD flushes subnormals to zero and returns the default
NaN, so `SIMD=1` and `SIMD=0` can differ on ARM32 for subnormal or NaN-payload
values; only the NE10 NEON FFT and, on cores with fused FMA
(`-mfpu=neon-vfpv4`), the AEC delay matched-filter kernels are affected.

Public memory descriptors remain 32 bytes. Their build hashes add an ILP32 token
on 32-bit targets and preserve the existing hashes on 64-bit targets. Re-query
requirements for the target build; `_init_ex` rejects a descriptor from the other
ABI before touching the supplied pool.

`make target-abi-test` builds and runs the 8-interface descriptor test with the
same target compiler/producer settings; `make test_target_abi` builds it without
running. The keyed `test_target_abi` binary accepts `--write descriptor.bin` and
`--reject descriptor.bin`. Exchange files between 32/64-bit builds with the same
model/grid settings: rejection must leave caller memory unchanged even after the
foreign byte budget is normalized to the current requirement.


## Architecture

```
mic/ref → linear AEC (aec_process_context(): context-only, no emit)
              ├─ E(f) ────────────────→ echo-aware NR ─→ G_nr ─┐
              └─ AecResContext {R², G_res, CNG, far power} ─────┤
                                                                ↓
                                  G_total = min(G_nr, G_res) + CNG
                                                                ↓
                                                       one iFFT/OLA

aec_only mode (no NR/RES stage at all):
mic/ref → linear AEC (aec_process(): emits into out) → out
```

There is no pipeline library behind this diagram: the AEC, the denoiser, the RES object and the
synthesis are separate libraries that the application initialises (each with its own
`mem_size` and `init`) and chains; each example `main` writes the per-hop chain itself. There is
no independent time-domain RES filter after NR. In the mono flow `G_res` is calculated
inside the AEC3 suppression path (`ctx.res_gain`), exported with the linear residual, fused with
`G_nr`, and applied once to `E(f)`.

The non-`aec_only` path never reads `aec_process()`'s own returned audio —
only `AecResContext`'s linear residual matters downstream — so it drives AEC
via `aec_process_context()` instead: same linear filter and AEC3
post/RES-context computation, minus the copy into an `out` buffer neither the
example nor the caller ever reads on that path. The mono examples pick one entry
point at start-up from `--aec-only`; the two entry points advance identical state and
may be mixed freely on one instance.

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libanr.a | anr.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a | aec_res.h (`AecRes`) / aec.h (`AecResContext`) | Mono: inside the AEC (`ctx.res_gain`, strength `aec_set_preset()`). 4-ch: managed by libaec, created by the application after the unit (or the core) with `aec_res_get_mem_size()` / `aec_res_init()`; neither holds one |
| Mono | no unit: `Aec` itself | aec.h, common/se_stage.h (`se_frame_from_aec()`) | One-mic AEC with its own RES gain; the application calls `aec_process_context()` and chains RES gain / NR / comfort noise / synthesis (the example mains do) |
| 4-ch front end | libaudio_pipeline_4ch.a | 4ch_aec_bf_nr_res/4aec_nr_res.h | One shared aligner + four linear AECs + the fuse (`four_aec_front_process()`, `four_aec_fuse()`); no RES in the core, the application builds its own `AecRes` |
| 4-ch linear AEC + BF unit | libaudio_pipeline_4ch.a | 4ch_aec_bf_nr_res/audio_pipeline_4ch.h | The one wrapped unit: front end (shared aligner + four lanes) + reusable SRP-PHAT/BF libraries; per hop `process_front()` / `process_beam()` (or `process_with_activity()`) give the BF spectrum and weights, `audio_pipeline_4ch_fuse()` is the optional step that fills a `SeSpecFrame` (`g_res_ctx` stays NULL). The post-beam RES, NR, comfort noise, synthesis and neural post-filters are not in it |
| DFN2 stage / rate bridge | libaudio_pipeline_dfn.a (`pipelines/` build), libdfn_res_stage.a (4-ch build) | dfn_res_stage.h, dfn_rate_bridge.h | The post-RES dual-input DFN2 stage (48 kHz directly, below 48 kHz through the bridge); the hop driver is `samples/dfn_hop.h` |
| Example code | not a library: headers and objects of the examples | samples/ | `se_anr_stage.h`, `dfn_hop.h`, `ulcnet_nn_stage.{h,c}`: no ABI, no layout version, in no archive |

RES is available as an independent frequency-domain `AecRes` object in
`aec_res.h`, shipped in `libaec.a`. A four-channel application, whether it runs the unit or drives the core
directly, creates one such object in its own init after the unit
(`aec_config_from_preset(preset, sample_rate)` with `fft_size` set to the unit's grid, then
`aec_res_get_mem_size()` / `aec_res_init()`; there is no `aec_res_destroy`) and calls `se_res_stage_gain()` once per hop
over the fused frame; neither `audio_pipeline_4ch_fuse()` nor `four_aec_fuse()` runs it
(`frame.g_res_ctx` stays NULL). NR independently calls `anr_process_gain()` on the
same fused `frame.e`; the application merges the two gains, applies them once,
and performs one final inverse transform/OLA. This is gain fusion, not a
RES-applied spectrum feeding NR.

The conventional mono application instead takes the AEC-owned `res_gain` from
`AecResContext` (`g_res_ctx` in `SeSpecFrame`). With `return_res_context=1` and
`enable_res=0`, `aec_process_context()` advances the linear filter and context
calculation without copying returned time audio. The `aec_process()`
convenience API also emits time audio; with RES enabled it applies its own
suppression/synthesis internally. That full convenience path is distinct from
the four-channel linear front → SRP/BF → fuse → gain stages flow.
See `lib/aec/c_impl/include/aec.h` and `aec_res.h` for ownership and inputs.

**`error_spec` does not always mean "error."** That configuration —
`enable_res=0` with `return_res_context=1` — is the one mode where `lib/aec`'s
formed selector opens a third candidate: on a hop the filtering-quality
analyzer has not yet cleared, whose selected residual carries more energy than
the mic capture, `formed_hop` and `error_spec` both become the capture itself,
faded in over the existing 30-sample transition. NR/RES still run on it as
usual — this is the steady-state fallback for a filter that is adding signal
rather than cancelling. `aec_process()`'s own emitted audio is unaffected.

The four-channel API is a separate zero-padding-free grid. See [the 4-channel contract](4ch_aec_bf_nr_res/README.md) and
[`4ch_aec_bf_nr_res/4aec_nr_res.h`](4ch_aec_bf_nr_res/4aec_nr_res.h) for its two
front calls (`four_aec_front_process()`, `four_aec_fuse()`) around the beamformer, and
[`4ch_aec_bf_nr_res/audio_pipeline_4ch.h`](4ch_aec_bf_nr_res/audio_pipeline_4ch.h) for the unit that owns the SRP-PHAT and the BF (it holds no RES).

The directory root contains only the aggregate build, documentation,
requirements and the public Python reference module. C sources live with
their applications. Shared ULCNet model-I/O support is owned by
`../AIAEC/Align_ULCNet/`; offline scoring and benchmark drivers remain under
`tools/`.

## Application layout

Only the multi-channel linear AEC + beamformer is wrapped as one unit (`AudioPipeline4Ch`); the post-beam RES is the application's own `AecRes`.
The product flows are compositions: each directory's `main.c` (and the static, raw and mcab
drivers) initialises the libraries one by one and chains them as an example; the chaining headers
live in [`samples/`](samples/README.md):

| Application | Entry directory | Composition |
|---|---|---|
| Mono AEC + NR/RES | `mono_aec_nr_res/` | `Aec` (its own RES gain) + `se_frame_from_aec()` + `AnrDenoiser` + comfort noise + `AcIstft` synthesis, chained in the example's `mono_form()` |
| 4-ch AEC + BF + NR/RES | `4ch_aec_bf_nr_res/` | the unit (4-lane front end + DOA/BF) + `audio_pipeline_4ch_fuse()` + the application's `AecRes` (`se_res_stage_gain()`) + `AnrDenoiser` + comfort noise + `AcIstft` synthesis, chained in the example's `raw_chain()` / `static_chain()` |
| Mono Align-ULCNet | `mono_alignulcnet/` | `Aec` + `ulcnet_nn_stage` + AIAEC pre/post |
| 4-ch Align-ULCNet | `4ch_alignulcnet/` | the unit with `enable_post = 0` + `ulcnet_nn_stage` + AIAEC pre/post |
| Mono DFN2 | `mono_aec_dfn_res/` | `Aec` (its own RES gain; P formed in `rig_process()`, no NR, no synthesis of its own) + `dfn_hop` + the dual-input DFN2 stage |
| 4-ch DFN2 | `4ch_aec_bf_dfn_res/` | the unit (real SRP-PHAT/BF) + `audio_pipeline_4ch_fuse()` + the application's `AecRes` (P formed in `dfn_form()`) + `dfn_hop` + the dual-input DFN2 stage |

The mono, 4-channel and mono ULCNet examples take their signal grid, AEC
configuration and delay-request rules from `common/se_config.{h,c}`
(`se_derive()`; each application passes its own `SeRules`: rate set, strict
delay check, lane alignment, seam context). The NR examples use the NR
library's own strength preset on the AEC grid (`anr_config_for_mode_grid()`,
no overrides), so `anr_set_mode()` changes the strength of a running denoiser;
`samples/se_anr_stage.h` holds the per-hop gain call that folds the AEC
residual-echo powers into the noise floor.

The ULCNet applications use `../AIAEC/Align_ULCNet/ulcnet_accelerator_adapter.*`: CPU memory owns
K/V, logit and GRU state, while the board implements one stateless tensor
invocation callback. An absent/failing accelerator is fail-open. The adapter
drives an `ULCNET_IO_FREQ` instance of the `ulcnet_prepost` pre/post class
(spectrum in, spectrum out, no transform of its own), so both application
Makefiles link `ulcnet_prepost.o` wherever the adapter is linked — see
`ULCNET_ADAPTER_OBJS` in [`Makefile`](Makefile) and
[`4ch_aec_bf_nr_res/Makefile`](4ch_aec_bf_nr_res/Makefile).

## Call sequence, public/internal functions, parameters at a glance

The per-flow diagrams (return values, refusal rules, the swimlane over hops) are section 0 of
[`../docs/integration_mono_zh_TW.md`](../docs/integration_mono_zh_TW_01_overview_link_quickstart.md#0-速覽流程時序圖公開內部函式外部內部參數),
[`../docs/integration_4ch_core_zh_TW.md`](../docs/integration_4ch_core_zh_TW.md) and
[`../docs/integration_4ch_spatial_zh_TW.md`](../docs/integration_4ch_spatial_zh_TW.md); the board
sequence (query, allocate, init, teardown) is in
[`../docs/pipeline_board_integration.md`](../docs/pipeline_board_integration.md). This is the one-screen
version.

```
Mono                App        Aec                 AnrDenoiser      AcIstft
 init (once)         |-- se_derive; aec_get_mem_size/aec_init -->|
                     |-- anr_config_for_mode_grid; anr_get_mem_size/anr_init ->|
                     |-- ac_stft_config_default; ac_istft_get_mem_size/ac_istft_init ------->|
 per hop             |-- aec_process_context(mic,ref) -->|
                     |-- se_frame_from_aec --> frame{e,r2,cn,g_res_ctx}
                     |-- se_anr_stage_gain(frame) ------------>| G_nr
                     |-- sk_min_f32, sk_capply_gain_f32, aec_cng_fill
                     |-- ac_istft_push(spec) ------------------------------------>| out (1 hop later)
 refused / non-finite|-- aec_reset, anr_reset, ac_istft_reset, rng = SE_CNG_RNG_SEED  (all together)

4-channel           App        AudioPipeline4Ch      libaec AecRes   AnrDenoiser      AcIstft
 init (once)         |-- audio_pipeline_4ch_default_config; _get_mem_requirements; _init_ex -->|
                     |-- aec_config_from_preset (fft_size = the unit's); aec_res_get_mem_size/aec_res_init ->|
                     |-- anr_* and ac_istft_* as above (separate blocks) ---------------------->|----->|
 per hop             |-- _process_front --> (mask / VAD) --> _process_beam   (or _process_with_activity / _process)
                     |-- _fuse --> frame{e,r2,cn,g_res_ctx = NULL}
                     |-- se_res_stage_gain(res, &frame) ---------->| G_res
                     |-- se_anr_stage_gain; sk_min_f32; sk_capply_gain_f32; aec_cng_fill ->| ... ->| out
 core.enable_post=0  |-- no _fuse; beam.spectrum goes to the NN stage (ulcnet_nn_stage_*); no RES
 refused / non-finite|-- audio_pipeline_4ch_reset, aec_res_reset, anr_reset, ac_istft_reset, rng = SE_CNG_RNG_SEED
```

The 4-channel core (`FourAecNrRes`) replaces `_process_front/_process_beam/_fuse` with
`four_aec_front_process()`, your beamformer and `four_aec_fuse()`. Neither the core nor the unit has a RES:
build an `AecRes` yourself and call `se_res_stage_gain()`.

| | Public (an integrator calls these) | Internal (do not include or depend on) |
|---|---|---|
| Libraries | `aec.h`, `aec_res.h`, `anr.h`, `audio_stft.h`, `common/se_config.h`, `common/se_stage.h`, `utility/simd_kernels.h` (`sk_*`), `utility/mem_align.h` (`ALIGN16`) | AEC `pbfdkf.h`, `delay_aec.h`, `aec_post.h`, `suppression_gain.h`, `aec_config_build.h`, `aec_balanced_config.h`; NR `noise_estimator.h`, `spp_estimator.h`, `anr_internal.h` |
| 4-channel | `audio_pipeline_4ch.h` (unit), `4aec_nr_res.h` (core) | `4aec_nr_res_internal.h`, `4aec_projection_kernels.h`, `bf.h`, `srp.h` |
| Example code (copy and adapt) | `samples/se_anr_stage.h`, `samples/dfn_hop.h`, `samples/ulcnet_nn_stage.{h,c}` | - |

| Parameters | External (a third party may tune) | Internal (exists, tunable in code, not recommended to open) |
|---|---|---|
| Deployment | sample rate and FFT size (the grids below), `filter_length`, delay mode with `delay_num_filters` / `fixed_delay_samples`, array geometry and `num_angles` (4-channel) | `n_partitions` and `delay_buffer_ms` of the lane `AecConfig` |
| Strength | `AecPreset` (`aec_set_preset()`, `aec_res_set_preset()`), `AnrMode` (`anr_set_mode()`) | the raw `min_gain_floor_*_db` values, the NR `g_min_db` / `q` / `xi_min_db` / `noise_over_subtraction` set one by one |
| Switches | the examples' RES / NR / CNG switches; `core.enable_post` | `spatial_linear_context`, `return_res_context`, the lanes' `enable_res = 0` (set by the pipeline code) |
| Tracking and adaptation | - | AEC step sizes, Kalman Q, detector thresholds, 10 ms-authored time constants; NR trackers, smoothing, scene-change, speech-protection; DOA/BF recursion constants (`bf_lambda`, `bf_mu`, `doa_*`) |

Field-by-field classification with the header and line of every field: section 0.3 of the three
manuals. `L` and `num_init_frames` of the denoiser are pool-sizing fields: `anr_set_mode()` and
`anr_reconfigure()` refuse to change them, so a different value means a new `anr_get_mem_size()`
and `anr_init()`.

## Parameter Alignment

All modules use an explicit, zero-padding-free 50%-overlap grid (frame ==
fft_size, hop == fft_size/2), auto-configured by sample rate. 16 kHz has two
selectable grids; the others have exactly one:

| Parameter | 8 kHz | 16 kHz (default) | 16 kHz (alt) | 48 kHz | Formula |
|-----------|-------|-------------------|--------------|--------|---------|
| frame_size / fft_size | 256 | 256 | 512 | 1024 | frame == fft_size, no padding |
| hop_size | 128 | 128 | 256 | 512 | frame / 2 |
| n_freqs | 129 | 129 | 257 | 513 | fft/2 + 1 |
| filter_length | 416 | 832 | 832 | 3072 | ms-derived: sr × 52ms (64ms ≥44.1 kHz) |
| n_partitions | 4 | 7 | 4 | 6 | ceil(filter_length / hop) |

## Four-channel sample driver

The four 4-channel sample mains use `4ch_aec_bf_nr_res/mcab_sample_driver.{c,h}`
for option scanning and the setup → read → process → finite check → consume →
finish → cleanup sequence. Each main supplies typed pipeline/model adapters;
the driver is linked into these executables and is excluded from delivered
algorithm archives.

| Executable | Input and lifecycle |
|---|---|
| `audio_pipeline_4ch_raw` | Complete interleaved float32 raw hops, optional per-hop VAD, heap pipeline; trailing incomplete hops are ignored. |
| `audio_pipeline_4ch_static` | 48 generated tone hops and an explicit VAD schedule; query/allocate/init_ex with caller-owned pool. |
| `4ch_aec_bf_dfn_res` | Eight silent fail-open model hops; DFN2 matrix, delay-profile and descriptor setup stays local. |
| `4ch_alignulcnet` | Four silent fail-open model hops; descriptor/accelerator pool and model installation stay local. |

Model samples report success after resource destruction. The shared hop loop
performs no allocation. Run `make -C 4ch_aec_bf_nr_res test-mcab-sample-driver`
for CLI boundary, short-circuit, finite-output and cleanup-order regressions;
this gate also runs through that directory's `make test`.

