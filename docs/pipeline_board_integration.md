# Board integration: separate libraries, one flow module

## Scope and prerequisites

This document is the firmware-side contract for the delivered pipeline code: pool
sizing, the query -> init -> process -> destroy sequence, the memory descriptor of the one
flow module, the board build, and the on-target verification checklist.

What is delivered, and how it is initialised:

- Only the multi-channel linear AEC flow is a module with a descriptor: `Mcab`
  (`../pipelines/4ch_aec_bf_nr_res/mcab.h`: the shared-delay admission state machine, the
  reference ring, the aligned far and the `mcab_fuse()` projection). It is a flow layer, not a
  library that owns lib objects. The application creates the four `Aec` lanes
  (`mcab_lane_config()` or `mcab_apply_lane_contract()`, then `aec_get_mem_size()` /
  `aec_init()`) and, MATCHED only, one shared `DelayAec` (`mcab_delay_estimator_config()`, then
  `delay_aec_get_mem_size()` / `delay_aec_init()`), and passes them by pointer (`McabLibs`) to
  `mcab_init_ex()`. `mcab` never allocates, resets-by-destroying or frees one of them; its own
  pool holds only flow state. The descriptor (`McabMemReq`, fixed 32 bytes) and the
  `mcab_init_ex()` gate cover that pool only. `mcab` has no beamformer and no RES: `mcab_fuse()` takes
  the caller's weights and leaves `frame->g_res_ctx` NULL.
- Every other stage is a separate library that the integrator initialises with its own size
  query and `init`, and chains: the mono `Aec` (lib/aec, whose own RES gain is `ctx.res_gain`),
  `AecRes` (lib/aec; the 4-channel post-beam RES, which the application creates itself with
  `aec_res_get_mem_size()`/`aec_res_init()`, as `mcab_static.c` and
  `mcab_external_static.c` do; the mono RES is not a separate object, it is inside the `Aec` call),
  `AnrDenoiser` (lib/nr), `AcIstft` (audiolib), the DFN2 stage and rate bridge (`dfn_res_stage.h`,
  `dfn_rate_bridge.h`, which carry their own descriptors). There is no mono wrapper module and no "complete"
  wrapper object; the mono flow is `Aec` plus the libraries it chains per hop.
- SRP-PHAT, BF and the fallback VAD are not in `mcab`: the examples use the sample helper
  `mcab_sample_beam.{h,c}` (builds on audiolib's `libdoa.a`, `libbf.a`, `libspatial_common.a`; not in
  `libaudio_pipeline_mcab.a`), and `mcab_sample_flow.{h,c}` shows the application-side init of lanes,
  estimator and `Mcab` in one pool. A product with its own beamformer calls `mcab_process_front()`
  and `mcab_fuse()` directly.
- Each example `main` initialises the libraries it uses and writes the per-hop chaining itself
  (res gain, `anr_process_gain()`, `sk_min_f32()`, `sk_capply_gain_f32()`, `aec_cng_fill()`,
  `ac_istft_push()`). The shared example code is `../pipelines/4ch_aec_bf_nr_res/mcab_sample_post.h`,
  `../pipelines/dfn/dfn_hop.h` and `../pipelines/ulcnet/ulcnet_nn_stage.{h,c}`. It has no ABI, no
  layout version and no descriptor; copy and adapt it. The example mains (`main.c` of
  `mono_aec_nr_res`, `mcab_static.c`, `mcab_external_static.c`) and `example_board_adapter.c`
  show the composition.

It assumes [`../pipelines/README.md`](../pipelines/README.md) has been read
first — that file owns the algorithm description, the parameter/grid
alignment, the measured memory budget ("Memory Budget"), the heap-versus-pool
statement ("Verification"), the heap-vs-pool overview ("Two Versions"), and
the tunable-parameter tables. References below of the form
`see "X" in ../pipelines/README.md` point back there.

Every relative path in this file is written from `Audio_ALG/docs/`; every
`make` invocation shown is run from `Audio_ALG/pipelines/`.

- [Board integration: separate libraries, one flow module](#board-integration-separate-libraries-one-flow-module) <!-- split-toc -->
  - [Scope and prerequisites](#scope-and-prerequisites) <!-- split-toc -->
  - [At a glance: sequence, public/internal functions, external/internal parameters](pipeline_board_integration_01_sequence.md#at-a-glance-sequence-publicinternal-functions-externalinternal-parameters) <!-- split-toc -->
  - [Static APIs and pool sizing](pipeline_board_integration_01_sequence.md#static-apis-and-pool-sizing) <!-- split-toc -->
  - [Sequence (mono: `Aec` plus the libraries it chains)](pipeline_board_integration_01_sequence.md#sequence-mono-aec-plus-the-libraries-it-chains) <!-- split-toc -->
  - [Sequence (4-channel: the `mcab` flow plus the libraries it feeds)](pipeline_board_integration_01_sequence.md#sequence-4-channel-the-mcab-flow-plus-the-libraries-it-feeds) <!-- split-toc -->
    - [Running the model on the NPU (NN applications)](pipeline_board_integration_01_sequence.md#running-the-model-on-the-npu-nn-applications) <!-- split-toc -->
  - [Warnings](pipeline_board_integration_02_contract.md#warnings) <!-- split-toc -->
  - [Descriptor semantics (`McabMemReq`)](pipeline_board_integration_02_contract.md#descriptor-semantics-mcabmemreq) <!-- split-toc -->
  - [Serializing the descriptor](pipeline_board_integration_02_contract.md#serializing-the-descriptor) <!-- split-toc -->
  - [Dirty-pool contract](pipeline_board_integration_02_contract.md#dirty-pool-contract) <!-- split-toc -->
  - [`NO_STDIO=1` — building without libc stdio](pipeline_board_integration_02_contract.md#no_stdio1--building-without-libc-stdio) <!-- split-toc -->
  - [Teardown order](pipeline_board_integration_02_contract.md#teardown-order) <!-- split-toc -->
  - [Board-side verification checklist](pipeline_board_integration_03_verify_build.md#board-side-verification-checklist) <!-- split-toc -->
  - [Cross-compile / board build](pipeline_board_integration_03_verify_build.md#cross-compile--board-build) <!-- split-toc -->
  - [Compile flags (`EXTRA_CFLAGS`)](pipeline_board_integration_03_verify_build.md#compile-flags-extra_cflags) <!-- split-toc -->
  - [Unified FP-contraction policy](pipeline_board_integration_03_verify_build.md#unified-fp-contraction-policy) <!-- split-toc -->
