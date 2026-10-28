# C Pipeline: linear AEC + echo-aware NR/RES gain fusion

## Contents <!-- split-toc -->
- [C Pipeline: linear AEC + echo-aware NR/RES gain fusion](README.md#c-pipeline-linear-aec--echo-aware-nrres-gain-fusion) <!-- split-toc -->
  - [ARM target selection](README.md#arm-target-selection) <!-- split-toc -->
  - [Architecture](README.md#architecture) <!-- split-toc -->
  - [Modules](README.md#modules) <!-- split-toc -->
  - [Application layout](README.md#application-layout) <!-- split-toc -->
  - [Call sequence, public/internal functions, parameters at a glance](README.md#call-sequence-publicinternal-functions-parameters-at-a-glance) <!-- split-toc -->
  - [Parameter Alignment](README.md#parameter-alignment) <!-- split-toc -->
  - [Example code (copy and adapt)](README.md#example-code-copy-and-adapt) <!-- split-toc -->
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
    - [Version A: split pool (`aec_nr_pipeline`)](README_02_build_debug_board.md#version-a-split-pool-aec_nr_pipeline) <!-- split-toc -->
    - [Version B: one caller pool (`aec_nr_pipeline_static`)](README_02_build_debug_board.md#version-b-one-caller-pool-aec_nr_pipeline_static) <!-- split-toc -->
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

`make target-abi-test` builds and runs the descriptor test (seven interfaces, `Mcab` in two
carves, eight cases) with the same target compiler/producer settings; `make test_target_abi`
builds it without running. The keyed `test_target_abi` binary accepts `--write descriptor.bin` and
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

Without `--aec-only` the chain never reads `aec_process()`'s returned audio, so it calls
`aec_process_context()`: the same linear filter and RES-context computation, minus the copy into an
`out` buffer. The mono examples pick one entry point at start-up from `--aec-only`; the two advance
identical state and may be mixed freely on one instance.

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libanr.a | anr.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a | aec_res.h (`AecRes`) / aec.h (`AecResContext`) | Mono: inside the AEC (`ctx.res_gain`, strength `aec_set_level()`). 4-ch: managed by libaec, created by the application with `aec_res_get_mem_size()` / `aec_res_init()`; `Mcab` does not hold one |
| Mono | no wrapper module: `Aec` itself | aec.h (`aec_process_context()`, `aec_get_res_context()`) | One-mic AEC with its own RES gain; the application calls `aec_process_context()`, reads the `AecResContext` and chains RES gain / NR / comfort noise / synthesis (the example mains do) |
| 4-ch flow (`Mcab`) | libaudio_pipeline_mcab.a (`mcab.o` only) | 4ch_aec_bf_nr_res/mcab.h | A flow layer, not a library that owns lib objects: the application initialises four `Aec` lanes (and, MATCHED only, one shared `DelayAec`) and passes pointers; `Mcab` owns only the flow state in its own pool (delay admission state machine, reference ring, aligned far, fuse buffers). Per hop `mcab_process_front()` and `mcab_fuse()`; no beamformer, no RES, no NR inside |
| DFN2 stage / rate bridge | libaudio_pipeline_dfn.a (`pipelines/` build), libdfn_res_stage.a (4-ch build) | dfn/dfn_res_stage.h, dfn/dfn_rate_bridge.h | The post-RES dual-input DFN2 stage (48 kHz directly, below 48 kHz through the bridge); the hop driver is `dfn/dfn_hop.h`. The model lookahead pair is a build-time choice (default 1/1, see [Build](README_02_build_debug_board.md#build)) |
| Example code | not a library: headers and objects of the examples | common/example_cli.h, dfn/dfn_hop.h, ulcnet/ulcnet_nn_stage.{h,c}, 4ch_aec_bf_nr_res/mcab_sample_* | No ABI, no layout version, in no archive (see [Example code](#example-code-copy-and-adapt)) |

RES is available as an independent frequency-domain `AecRes` object in
`aec_res.h`, shipped in `libaec.a`. A four-channel application creates one in its own
init (`aec_config_from_level(level, sample_rate)` with `fft_size` set to the flow's grid
(`mcab_fft_size()`), then `aec_res_get_mem_size()` / `aec_res_init()`; there is no `aec_res_destroy`)
and calls `aec_res_process()` once per hop on the input `mcab_sample_res_input()` makes from the fused frame; `mcab_fuse()` does not run it
(`frame.g_res_ctx` stays NULL). NR independently calls `anr_process_gain()` on the
same fused `frame.e`; the application merges the two gains, applies them once,
and performs one final inverse transform/OLA. This is gain fusion, not a
RES-applied spectrum feeding NR.

The mono application instead takes the AEC-owned `res_gain` from `AecResContext`
(`aec_get_res_context()`). With `return_res_context=1` and `enable_res=0`,
`aec_process_context()` advances the linear filter and context calculation without copying
returned time audio. The `aec_process()` convenience API also emits time audio (with RES enabled it
applies its own suppression/synthesis internally); that full path is distinct from the
four-channel `mcab_process_front()` → beamformer → `mcab_fuse()` → gain stages flow.
See `lib/aec/c_impl/include/aec.h` and `aec_res.h` for ownership and inputs.

**`error_spec` does not always mean "error."** That configuration —
`enable_res=0` with `return_res_context=1` — is the one mode where `lib/aec`'s
formed selector opens a third candidate: on a hop the filtering-quality
analyzer has not yet cleared, whose selected residual carries more energy than
the mic capture, `formed_hop` and `error_spec` both become the capture itself,
faded in over the existing 30-sample transition. NR/RES still run on it as
usual — this is the steady-state fallback for a filter that is adding signal
rather than cancelling. `aec_process()`'s own emitted audio is unaffected.

The four-channel flow is a separate zero-padding-free grid. See [the 4-channel contract](4ch_aec_bf_nr_res/README.md) and
[`4ch_aec_bf_nr_res/mcab.h`](4ch_aec_bf_nr_res/mcab.h) for its two per-hop calls
(`mcab_process_front()`, `mcab_fuse()`) around the application's beamformer; the SRP-PHAT/BF glue of the
examples is the sample helper [`4ch_aec_bf_nr_res/mcab_sample_beam.h`](4ch_aec_bf_nr_res/mcab_sample_beam.h)
(not part of the archive).

The directory root holds the aggregate build, the documentation, `requirements.txt`, `scripts/`,
`tools/` (two offline drivers, `eval_dfn2_4ch_inputs.py` and `profile_dfn2_ops.py`; the other research drivers
are archived outside the repository, see the cleanup record) and `tests/` (the cross-cutting tests:
`test_c_python_map.py`, `test_doc_layout_versions.py`, `test_readme_flow.c`, `test_target_abi.c`, `pyc_parity*`).
C sources and the Python twins live with their applications: `common/` (`example_cli.h`, the strict
command-line parser of the examples; `cng_aec3.py`; `tests/`), `mono_aec_nr_res/` (`main.c`, `pipeline.py`),
`4ch_aec_bf_nr_res/` (mcab and its samples, `pipeline.py`), `dfn/` (the DFN2 stage, rate bridge,
`dfn2_stage.py`, `dfn2_rate_adapter.py` and the two DFN applications) and `ulcnet/` (the ULCNet stage and
the two ULCNet applications). Shared ULCNet model-I/O support is owned by `../AIAEC/Align_ULCNet/`.

## Application layout

Only the shared delay, the four-lane flow and the fuse projection of the multi-channel linear AEC
are one module (`Mcab`, a flow layer over objects the application initialises); the post-beam RES
is the application's own `AecRes`. The product flows are compositions: each directory's `main.c` (and
the `mcab_*` drivers) initialises the libraries one by one and chains them as an example; the chaining
headers are listed under [Example code](#example-code-copy-and-adapt):

| Application | Entry directory | Composition |
|---|---|---|
| Mono AEC + NR/RES | `mono_aec_nr_res/` | `Aec` (its own RES gain, read with `aec_get_res_context()`) + `AnrDenoiser` + comfort noise + `AcIstft` synthesis, chained in the example's `mono_form()` |
| 4-ch AEC + BF + NR/RES | `4ch_aec_bf_nr_res/` | four `Aec` lanes + one shared `DelayAec` (MATCHED), initialised by the application + `Mcab` + a beamformer (the sample helper `mcab_sample_beam` runs SRP-PHAT/BF, or the application's own) + `mcab_fuse()` + the application's `AecRes` (`aec_res_process()`) + `AnrDenoiser` + comfort noise + `AcIstft` synthesis, chained in the example's `raw_chain()` (`mcab_raw`), `static_chain()` (`mcab_static`) or `post_hop()` (`mcab_external_static`) |
| Mono Align-ULCNet | `ulcnet/mono_alignulcnet/` | `Aec` + `ulcnet_nn_stage` + AIAEC pre/post |
| 4-ch Align-ULCNet | `ulcnet/4ch_alignulcnet/` | the flow with `enable_post = 0` + the sample beam helper + `ulcnet_nn_stage` + AIAEC pre/post |
| Mono DFN2 | `dfn/mono_aec_dfn_res/` | `Aec` (its own RES gain; P formed in `rig_process()`, no NR, no synthesis of its own) + `dfn_hop` + the dual-input DFN2 stage |
| 4-ch DFN2 | `dfn/4ch_aec_bf_dfn_res/` | the flow + the sample beam helper (real SRP-PHAT/BF) + `mcab_sample_beam_fuse()` + the application's `AecRes` (P formed in `dfn_form()`) + `dfn_hop` + the dual-input DFN2 stage |

The mono, mono DFN2 and mono ULCNet examples each derive their AEC configuration in their own
`main.c` (`derive_lane()`: `aec_config_from_level()`, the grid from `aec_resolve_signal_grid()`
(ULCNet: its compiled grid), then the lane's seam fields; a delay profile is parsed with `common/example_cli.h`). The 4-channel flow resolves its own grid from
`McabConfig` (`sample_rate`, `fft_size`) and stamps the lane contract with
`mcab_apply_lane_contract()`. The NR examples use the NR
library's own strength level on the AEC grid (`anr_config_for_level_grid()`,
no overrides), so `anr_set_level()` changes the strength of a running denoiser;
the per-hop gain call that folds the AEC residual-echo powers into the noise floor is written in
each example (`mono_nr_gain()` in `mono_aec_nr_res/main.c`) and, shared by the 4-channel programs,
`mcab_sample_anr_gain()` in `4ch_aec_bf_nr_res/mcab_sample_post.h`. The numeric options of the DFN2 and ULCNet
examples and of `mcab_raw`, `mcab_static` and `mcab_external_static` go through the strict parser of
`common/example_cli.h` (`example_cli_parse_int()`, `example_cli_parse_float()`): an empty string, leading space,
trailing characters, an out-of-range value, `inf` or `nan` exit with code 2 (for example
`--delay-num-filters 3x`, or an empty `--fixed-doa-deg` of `mcab_raw`). `mcab_static` and `mcab_external_static`
also exit 2 for an option without its value (a token starting with `--` where a value belongs is the next option, not a value) and for an unknown option; `mcab_external_static` also exits 2 for an unknown `--nr-level` name (earlier spelling `--nr-preset`), an unknown AEC level (positional or `--aec-level`) and a second AEC level (the mono programs exit 2 for a second AEC level too); a number that parses but names no grid
or channel (`--sample-rate 44100`) is refused when the config is built and still exits 1. Still lenient:
`mono_aec_nr_res/main.c` (`aec_nr_pipeline`, `aec_nr_pipeline_static`) reads `--sample-rate` and `--fft-size` with
`atoi()`, ignores an unknown option and falls back to `balanced` for an unknown level name (it does exit 2 for a missing value, as above, and for a second AEC level); the test and smoke helpers `tests/pyc_parity/pyc_drv.c` and
`4ch_aec_bf_nr_res/tests/dump_delay_parity.c` also read numbers with `atoi()` / `atof()` / `strtoul()`.

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
 init (once)         |-- aec_resolve_signal_grid; aec_config_from_level; aec_get_mem_size/aec_init -->|  (derive_lane())
                     |-- anr_config_for_level_grid; anr_get_mem_size/anr_init ->|
                     |-- ac_stft_config_default; ac_istft_get_mem_size/ac_istft_init ------->|
 per hop             |-- aec_process_context(mic,ref) -->|
                     |-- aec_get_res_context --> ctx{error_spec,r2,comfort_noise,res_gain}
                     |-- anr_process_gain / anr_get_gain ------>| G_nr
                     |-- sk_min_f32, sk_capply_gain_f32, aec_cng_fill
                     |-- ac_istft_push(spec) ------------------------------------>| out (1 hop later)
 refused hop         |-- aec_reset, anr_reset, ac_istft_reset, rng = CNG_RNG_SEED  (all together)

4-channel           App        Aec x4 + DelayAec    Mcab                 AecRes   AnrDenoiser   AcIstft
 init (once)         |-- mcab_default_config; mcab_lane_config (or aec_config_from_level + mcab_apply_lane_contract)
                     |-- aec_get_mem_size/aec_init x4; mcab_delay_estimator_config; delay_aec_get_mem_size/delay_aec_init (MATCHED) -->|
                     |-- mcab_get_mem_requirements; mcab_init_ex(mem, bytes, &cfg, &libs, &req) ----------------->|
                     |-- aec_config_from_level (fft_size = mcab_fft_size(p)); aec_res_*, anr_*, ac_istft_* : each its own mem_size + init
 per hop             |-- mcab_process_front(mics, ref, &front) --> front.linear_spectra[4]
                     |-- your beamformer (or the sample helper's mcab_sample_beam_process_beam) --> W[4][n_freqs] (+ E)
                     |-- mcab_fuse(p, W, E, &frame) --> frame{e,r2,cn,g_res_ctx = NULL}
                     |-- mcab_sample_res_input + aec_res_process(res, &in) --> G_res ; mcab_sample_anr_gain --> G_nr ; sk_min_f32 ; sk_capply_gain_f32
                     |-- aec_cng_fill ; ac_istft_push --> out (1 hop behind the input)
 cfg.enable_post = 0 |-- no mcab_fuse (MCAB_INVALID_ARGUMENT); the beamformed spectrum goes to ulcnet_nn_stage_*; no RES
 refused / non-finite|-- mcab_reset (also resets the lanes and the estimator), aec_res_reset, anr_reset, ac_istft_reset, rng = MCAB_SAMPLE_CNG_RNG_SEED
 teardown            |-- mcab_destroy; then the application tears the lanes down and frees its memory
```

`Mcab` takes the beamformer's weights in `mcab_fuse()`; nothing in `mcab.h` runs SRP-PHAT or a
beamformer. The sample helper `mcab_sample_beam` (SRP-PHAT, BF and a fallback energy VAD between
`mcab_process_front()` and `mcab_fuse()`) is sample code; while it is attached the flow is driven only
through it. Neither `Mcab` nor the helper has a RES: build an `AecRes` yourself and call
`aec_res_process()`.

| | Public (an integrator calls these) | Internal (do not include or depend on) |
|---|---|---|
| Libraries | `aec.h`, `aec_res.h`, `anr.h`, `audio_stft.h`, `utility/simd_kernels.h` (`sk_*`), `utility/mem_align.h` (`ALIGN16`) | AEC `pbfdkf.h`, `delay_aec.h` (other than the estimator init above), `aec_post.h`, `suppression_gain.h`, `aec_config_build.h`, `aec_balanced_config.h`; NR `noise_estimator.h`, `spp_estimator.h`, `anr_internal.h` |
| 4-channel | `mcab.h` (the flow); `delay_aec.h` is public only for the estimator init the application performs (`delay_aec_get_mem_size()`, `delay_aec_init()`) | `mcab_internal.h`; `bf.h` / `srp.h` and `audiolib/{doa,bf}` (used by the sample helper only) |
| Example code (copy and adapt) | the files of [Example code](#example-code-copy-and-adapt) | - |

| Parameters | External (a third party may tune) | Internal (exists, tunable in code, not recommended to open) |
|---|---|---|
| Deployment | sample rate and FFT size (the grids below), `filter_length` (a field of each 4-channel lane's `AecConfig`, set before `aec_get_mem_size()`), delay mode with `delay_num_filters` / `fixed_delay_samples` (`McabConfig` for 4-channel), array geometry and `num_angles` (`McabSampleBeamParams`, sample helper) | `n_partitions` and `delay_buffer_ms` of the lane `AecConfig` |
| Strength | `AecLevel` (`aec_set_level()`, `aec_res_set_level()`), `AnrLevel` (`anr_set_level()`) | the raw `min_gain_floor_*_db` values, the NR `g_min_db` / `q` / `xi_min_db` / `noise_over_subtraction` set one by one |
| Switches | the examples' RES / NR / CNG switches; `McabConfig.enable_post` | `spatial_linear_context`, `return_res_context`, the lanes' `enable_res = 0` (stamped by `mcab_apply_lane_contract()`) |
| Tracking and adaptation | - | AEC step sizes, Kalman Q, detector thresholds, 10 ms-authored time constants; NR trackers, smoothing, scene-change, speech-protection; DOA/BF recursion constants (`bf_lambda`, `bf_mu`, `doa_*` of `McabSampleBeamParams`) |

Field-by-field classification with the header and line of every field: section 0.3 of the three
manuals. `L` and `num_init_frames` of the denoiser are pool-sizing fields: `anr_set_level()` and
`anr_reconfigure()` refuse to change them, so a different value means a new `anr_get_mem_size()`
and `anr_init()`.

C level == Python preset. The C code, its tests and its command lines say *level* (`AecLevel`,
`AnrLevel`, `--nr-level`); the Python twins in this tree (`pipeline.py`, `set_preset()`, the Python
`--nr-preset`) keep *preset*. The libraries keep the legacy C names `AecPreset`, `AEC_PRESET_*`,
`AnrMode` and `ANR_MODE_*` (same values), and the C programs still accept `--nr-preset` as another
spelling of `--nr-level`.

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

## Example code (copy and adapt)

Examples, not library API: no ABI, no layout version, no descriptor, in no archive. Copy a header
into your project and change it. There is no shared chain header: each example `main` initialises
the libraries it uses and writes the per-hop sequence itself, for instance `mono_form()` in
`mono_aec_nr_res/main.c` and `raw_chain()` in `4ch_aec_bf_nr_res/mcab_raw.c`.

| File | What it shows | Sequence (app-side order) | Return values |
|---|---|---|---|
| `4ch_aec_bf_nr_res/mcab_sample_post.h` | The glue between `mcab_fuse()` and the libraries behind it: the libaec RES input of a fused frame (`mcab_sample_res_input`), and the NR gain call that folds the residual-echo powers into the denoiser's noise floor (`mcab_sample_anr_gain`). | per hop `mcab_sample_res_input(&frame, &in)` -> `aec_res_process(res, &in)`; `mcab_sample_anr_gain(nr, &frame, scratch, use_r2)`; on a refusal `anr_reset()` | gains owned by the library objects, valid until their next call; `NULL` = a NULL argument or a refused hop |
| `dfn/dfn_hop.h` | One DeepFilterNet2 hop: the 48 kHz stage with a caller-owned synthesis, or the rate bridge below 48 kHz; the phased begin / model / end calls. | `dfn_hop_init_stage()` (48 kHz, caller-owned `AcIstft`) or `dfn_hop_init_bridge()`; per hop `dfn_hop_begin(E, P)` -> `while (dfn_hop_nn_begin() == 1) dfn_hop_nn_end(rc)` -> `dfn_hop_end(out)`; `dfn_hop_reset()` on a refusal | `DFN_HOP_INVALID` (-1), `DFN_HOP_SEQUENCE_ERROR` (-2) |
| `ulcnet/ulcnet_nn_stage.{h,c}` | The Align-ULCNet post-filter stage behind a linear AEC: far-end STFT, model call policy (reprime, alignment events, fail-open), WOLA. | `ulcnet_nn_stage_mem_size()` -> `_init()` -> `_set_model()`; per hop `_begin()` (re/im arrays), `_begin_time()` or `_begin_complex()` -> `_nn_begin()` -> `_nn_end(rc)` -> `_end(out)`; `_reset()` on a refusal | `nn_begin` 1 = a frame to run, 0 = none (warm-up or reprime) |
| `4ch_aec_bf_nr_res/mcab_sample_flow.{h,c}` | The application-side init of one mcab flow in one pool: four lanes (`mcab_lane_config()` + `aec_init()`), the shared estimator (MATCHED only) and `mcab_init_ex()`. | `mcab_sample_flow_get_mem_size()` -> `mcab_sample_flow_init()` (or `_create()`); `mcab_sample_flow_destroy()` | 0 / -1 |
| `4ch_aec_bf_nr_res/mcab_sample_beam.{h,c}` | SRP-PHAT, BF and the fallback VAD glue between `mcab_process_front()` and `mcab_fuse()`. | `mcab_sample_beam_get_mem_size()` -> `mcab_sample_beam_init()`; per hop `_process_front()` -> `_process_beam()` (or `_process_with_activity()` / `_process()`) -> `_fuse()`; `mcab_sample_beam_destroy()` first, then `mcab_destroy()` and the lanes | `MCAB_OK`, `MCAB_INVALID_ARGUMENT`, `MCAB_SEQUENCE_ERROR`, `MCAB_DSP_ERROR` |

The stage constants (`MCAB_SAMPLE_PSD_SCALE_INV`, `ULCNET_NN_STAGE_REPRIME_FRAMES`, the DFN scale powers of two) are internal.
`make test_mcab_sample_post` builds the test of `mcab_sample_post.h`.

`ulcnet_nn_stage_mem_size(1)` sizes a stage that also owns an STFT for `_begin_time()`. A 4-channel caller passes `{ front.delay.changed, front.delay.solid }` to `_begin_complex()`; a mono lane maps `AecLinearContext.delay_state` to `{ state == CHANGED, state != UNLOCKED }` and may pass `formed_hop` to `ulcnet_nn_stage_begin_time()` on a stage built with an error
analysis. One build serves one grid: `ULCNET_GRID=16k` (16 kHz / fft 512, default) or `48k`
(48 kHz / fft 1024). The tests of the example code are `dfn/tests/test_dfn_hop.c` and
`ulcnet/tests/test_ulcnet_nn_stage.c`: `make test_dfn_hop` and `make test_ulcnet_nn_stage` build
them, `make test` builds and runs both.

## Four-channel sample driver

`mcab_raw`, `mcab_static`, `4ch_aec_bf_dfn_res` and `4ch_alignulcnet` use
`4ch_aec_bf_nr_res/mcab_sample_driver.{c,h}` for option scanning and the setup -> read -> process ->
finite check -> consume -> finish -> cleanup sequence. Each main supplies typed adapters; the driver
is linked into these executables, not into any delivered archive. `mcab_external_static` has its own
`main` (fixed equal-weight smoke beamformer) and does not link it.

| Executable | Input and lifecycle |
|---|---|
| `mcab_raw` | Complete interleaved float32 raw hops, optional per-hop VAD; the flow and sample beam helper in one aligned pool, the post-beam stages in a second one; trailing incomplete hops are ignored. |
| `mcab_static` | 48 generated tone hops and an explicit VAD schedule; the same two-pool layout, lanes and estimator initialised by the sample, then `mcab_get_mem_requirements()` and `mcab_init_ex()`. |
| `4ch_aec_bf_dfn_res` | Eight silent fail-open model hops; DFN2 matrix, delay-profile and descriptor setup stays local. |
| `4ch_alignulcnet` | Four silent fail-open model hops; descriptor/accelerator pool and model installation stay local. |

Model samples report success after resource destruction; the shared hop loop allocates nothing.
`make -C 4ch_aec_bf_nr_res test-mcab-sample-driver` runs the CLI boundary, short-circuit,
finite-output and cleanup-order regressions (also part of that directory's `make test`).
