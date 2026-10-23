# Changelog

`Audio_ALG/pipelines`（含它建置的 `AINR`、`AIAEC` C 程式）的重要改動記錄在此，格式依
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。README 只寫現況，改動經過寫在這裡。

## [Unreleased] — 2026-10-24 — NR minimum window: library default

### Changed
- The pipelines build their denoiser from the NR library preset alone,
  `anr_config_for_mode_grid(sample_rate, fft_size, nr_mode)`. The override of the minimum-tracking
  window `L` (`anr_retime_frames(150, sample_rate, hop)`: 188 / 94 / 141 hops at 16 kHz 256/128,
  16 kHz 512/256, 48 kHz 1024/512, about 1.5 s) is gone; `L` is the library default, 32 frames of
  16 ms (64 / 32 / 48 hops, about 0.512 s; 32 hops at 8 kHz). The `alpha_decay = alpha_g` line went with
  it: the preset already sets both from the same expression. Python mirrors it: `_build_denoiser()`
  no longer sets `L = 94` (`_NR_L_MINIMA_WINDOW`), so its `L` comes from `v3_2_config.yaml`.
- Output is bit-identical. With the scene-reset gate off (`broadband_threshold = 1.0`) the window
  length does not reach the gain: the golden matrix (audio, seam and driver rows) and the
  `pyc_parity` rows are unchanged at `SIMD=1` and `SIMD=0`; a control with the gate on
  (`broadband_threshold = 0.8`) does tell the two windows apart.
- `anr_set_mode()` now accepts the chain's denoiser: `anr_set_mode(se_sample_chain_nr(chain), mode)`
  changes its strength while it runs.
- The tests' oracles (`conv_chain_mono.h`, `conv_chain_4ch.h`, `test_se_common.c`) write the 512 ms
  window out in whole hops; `test_se_common` checks the window per grid as literals, that the chain
  sizes its denoiser like the oracle, that `anr_set_mode()` before the first hop equals a denoiser
  built in that mode on every grid, and that a target one hop longer is refused.

### Removed (example code, no ABI)
- `se_anr_compose_config()`, `se_anr_stage_set_mode()` and the pass-throughs
  `se_anr_stage_{mem_size,init,reset,destroy}()` (`samples/se_anr_stage.h`; call
  `anr_config_for_mode_grid()`, `anr_get_mem_size()`, `anr_init()`, `anr_set_mode()`, `anr_reset()`,
  `anr_destroy()`). `samples/se_anr_stage.h` keeps `se_anr_stage_gain()`, `se_psd_scale()` and
  `SE_PSD_SCALE` / `SE_PSD_SCALE_INV`.
- `se_sample_chain_set_nr_mode()` (`samples/se_sample_chain.h`): use `anr_set_mode()` on
  `se_sample_chain_nr(chain)`; a chain without NR returns `NULL` there and `anr_set_mode()` returns -1.

### Pool sizes (NE10, `SIMD=1`; bytes, after the split entry below -> now)
| Item | 8 kHz 256/128 | 16 kHz 256/128 | 16 kHz 512/256 | 48 kHz 1024/512 |
|---|---|---|---|---|
| `AnrDenoiser` (`anr_get_mem_size()`; `L` in hops) | 60,784 -> 28,784 (94 -> 32) | 109,328 -> 45,344 (188 -> 64) | 120,176 -> 56,432 (94 -> 32) | 335,424 -> 144,576 (141 -> 48) |
| Mono AEC + chain (`aec_nr_pipeline_static --print-mem-size`) | 359,232 -> 327,232 | 521,168 -> 457,184 | 672,768 -> 609,024 | 1,612,864 -> 1,422,016 |
| 4-channel front end + chain (`4aec_nr_res_static`) | | 1,171,200 -> 1,107,216 | 1,719,200 -> 1,655,456 | 3,863,568 -> 3,672,720 |
| 4-channel unit + chain (`audio_pipeline_4ch_static`; chain alone) | | 1,963,504 -> 1,899,520 (131,328 -> 67,344) | 3,289,232 -> 3,225,488 (161,376 -> 97,632) | 6,989,056 -> 6,798,208 (418,096 -> 227,248) |

Layout versions and `build_flags_hash` are unchanged: the denoiser lives in the header-only sample
chain, not in the unit or the core.


## [Unreleased] — 2026-10-24 — Pipeline split: one wrapped unit, separate libraries

### Changed
- Only the multi-channel linear AEC + beamformer is wrapped as one unit: `AudioPipeline4Ch`
  (`4ch_aec_bf_nr_res/audio_pipeline_4ch.{h,c}`), now `process_front()` / `process_beam()` /
  `process_with_activity()` / `process()` returning the beamformed spectrum and weights
  (`AudioPipeline4ChBeam`), plus an independent `audio_pipeline_4ch_fuse()` that fills a `SeSpecFrame`.
  RES, NR, comfort noise, synthesis and the neural post-filters are separate libraries that the
  integrator initialises (each with its own `mem_size` and `init`) and chains. Outputs of the
  example programs are bit-identical to the previous revision at `SIMD=1` and `SIMD=0` (a golden
  matrix of audio, seam and driver outputs, plus the `pyc_parity` seam rows); only pool sizes, titles and
  memory lines changed.
- The gain sequence (RES gain, NR gain, `min`, apply, comfort noise, synthesis) is example code:
  `pipelines/samples/se_sample_chain.h`. The NR recipe moved to `samples/se_anr_stage.h`, the DFN2
  hop driver to `samples/dfn_hop.h`, the Align-ULCNet stage to `samples/ulcnet_nn_stage.{h,c}`.
  The samples have no ABI, no layout version and no descriptor, and are in no archive.
- `se_frame_from_aec()` (`common/se_stage.h`) fills the `SeSpecFrame` of a mono `Aec`; mono has no
  unit at all (`Aec` + `se_frame_from_aec` + the chain).
- `pipelines/mono_*`, `4ch_alignulcnet` and `4ch_aec_bf_dfn_res` mains, the static mains, the raw
  runner, the mcab driver and `pyc_drv.c` are compositions of the above and stay as examples; CLI
  flags, positional arguments, output formats and the raw runner JSON keys are unchanged.
  `pyc_drv` now runs the shipped sample chain instead of the test oracle.
- Python: `FourChannelAecPipeline.fuse()` no longer runs the post-beam suppressor; its
  `context.res_gain` is `None`, and the caller applies `PostBeamResidualSuppressor` (exported from
  `4ch_aec_bf_nr_res/__init__.py`) to `FourChannelFrame.aligned_render`.
  `PostBeamResidualSuppressor.from_aec_config()` builds one from an `AecConfig`.
  `run_front_end` (DFN2 4-channel), `evaluate_recordings.py` and `tools/eval_dfn2_4ch_inputs.py`
  own their suppressor. Output is unchanged. `FourChannelAecPipeline.reset()` no longer resets the
  suppressor: the caller calls `suppressor.reset()` together with `pipeline.reset()`.
- Tests: `test_audio_pipeline*.c` and `test_4aec_dfn_res.c` are replaced by composition-level tests
  against hand-written oracles (`conv_chain_mono.h`, `conv_chain_4ch.h`): `test_mono_chain`,
  `test_mono_dfn`, `test_mono_ulcnet`, `test_4ch_dfn`, `test_4ch_ulcnet`, and
  `samples/tests/test_ulcnet_nn_stage.c`, `test_dfn_hop.c`. The cng-floor witness line lives in
  `mono_aec_nr_res/tests/test_mono_chain.c`.
- Fixed in the mono examples: `--no-nr` no longer dereferences a NULL denoiser under `DUMP_CTX`
  (the NR gain reads as the unity vector), and `static_main.c --print-mem-size` honours
  `--no-nr`, `--no-res` and `--no-cng`.
- A refused hop is recovered by the integrator, not by a library: reset the unit (or `Aec`), the
  chain and any neural stage together (`audio_pipeline_4ch_reset()` / `aec_reset()`,
  `se_sample_chain_reset()` — which also restarts the comfort-noise generator —
  `ulcnet_nn_stage_reset()` / `dfn_hop_reset()`). The DFN stage is still not reset by an AEC delay change.
- `audio_pipeline_4ch_process_front()` (and the calls built on it) leaves no front or beam pending
  after any refusal, so a following `process_beam()` / `audio_pipeline_4ch_fuse()` returns
  `SEQUENCE_ERROR`; `audio_pipeline_4ch_fuse()` no longer resets the unit on `SEQUENCE_ERROR`.
  A caller that skips a hop refused for a non-finite sample keeps all adaptive state.

### Removed (hard break, no shims)
- Mono: `audio_pipeline_{default_config,get_mem_requirements,init,init_ex,create,process,process_spec,synthesize,set_aec_preset,set_nr_mode,reset,destroy,hop_size,n_freqs,sample_rate,get_aec,get_nr,get_mem_breakdown,get_last_timing}`,
  `AudioPipelineConfig` / `AudioPipelineMemReq` / `AudioPipelineMemBreakdown` / `AudioPipelineLastTiming`,
  `mono_aec_nr_res/audio_pipeline.{h,c}`, `libaudio_pipeline.a`.
- `audio_pipeline_ulcnet_*`, `audio_pipeline_4ch_ulcnet_*`, `mono_aec_dfn_res_*`, `four_aec_dfn_res_*`
  and their config / memory-requirement types, headers and sources.
- `FourAecNrResConfig.{aec_preset,nr_mode,enable_res,enable_nr,enable_cng,legacy_amin}` and
  `four_aec_nr_res_stage_configs()`: the lanes are always derived with `AEC_PRESET_BALANCED` and
  compute only the linear residual; the settings belong to the chain (`SeSampleChainConfig`).
- `AudioPipeline4Ch`: the time-domain `output` of `process()` / `process_with_activity()`,
  `audio_pipeline_4ch_{set_aec_preset,post_split_floor,set_nr_mode,nr_count,post_res_count}`.
  The unit's config validation no longer requires `core.enable_post == 1`; with `enable_post = 0`
  `audio_pipeline_4ch_fuse()` returns `INVALID_ARGUMENT`.
- Build: `-DAUDIO_PIPELINE_STAGE_TIMING`, `-DAUDIO_PIPELINE_NO_STDIO` and `-DAUDIO_PIPELINE_NOW_US`
  (`PROFILE=1` now sets `-DAEC_STAGE_TIMING=1` only; the example programs read their own clock).
  `libaudio_pipeline_4ch.a` now holds `4aec_nr_res.o`, `audio_pipeline_4ch.o` and `se_config.o`.
- `common/se_anr_stage.{c,h}` (the recipe lives in `samples/se_anr_stage.h`), `SeRules.with_nr` and
  `SeRequest.nr_mode` (`common/se_config.h`; `se_derive()` no longer looks at the NR mode, and
  `se_config.h` no longer includes `anr_types.h`).
- Python: `FourChannelAecConfig.aec_preset`, `FourChannelAecPipeline.set_aec_preset()`,
  `FourChannelAecPipeline.residual_suppressor_instance_count`.

### Layout versions
| Macro | Old | New |
|---|---|---|
| `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` | 17 | **18** (pool no longer carries RES / NR / synthesis; `build_flags_hash` carve token changed) |
| `FOUR_AEC_NR_RES_LAYOUT_VERSION` | 22 | **23** (the config embedded in the control block shrank) |
| `AUDIO_PIPELINE_LAYOUT_VERSION` (+ descriptor version 2) | 15 | removed |
| `AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION` | 15 | removed |
| `AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION` | 23 | removed |
| `MONO_AEC_DFN_RES_LAYOUT_VERSION` | 5 | removed |
| `FOUR_AEC_DFN_RES_LAYOUT_VERSION` | 6 | removed |

`DFN_RES_STAGE_LAYOUT_VERSION` (3), `DFN_RATE_BRIDGE_LAYOUT_VERSION` (3) and the AINR / AIAEC layouts are
unchanged. The unit's `build_flags_hash` is `0x0ac1063e` -> `0x8aabdea5`; the core's is `0x5812ead9`.
`tests/test_doc_layout_versions.py` no longer lists the removed macros.

### Pool sizes (NE10, `SIMD=1`, matched n=5; bytes, old -> new)
| Item | 16 kHz | 48 kHz |
|---|---|---|
| Mono AEC + NR chain, `--print-mem-size` (16 kHz default 256/128; 48 kHz 1024/512) | 520,672 -> 521,168 | 1,610,832 -> 1,612,864 |
| Mono chain at 16 kHz 512/256; 8 kHz | 671,760 -> 672,768; 358,736 -> 359,232 | |
| 4-channel front end (`FourAecNrRes`) | 1,039,904 -> 1,039,872 | 3,445,504 -> 3,445,472 |
| 4-channel front end + chain (`4aec_nr_res_static`, 16 kHz 512/256 in the second row) | 1,170,576 -> 1,171,200; 1,718,064 -> 1,719,200 | 3,861,408 -> 3,863,568 |
| 4-channel unit + chain (`audio_pipeline_4ch_static`; the unit alone is 1,832,176 / 6,570,960) | 1,962,960 -> 1,963,504 | 6,986,976 -> 6,989,056 |
| Mono DFN2 | 764,096 -> 750,384 | 1,522,048 -> 1,513,536 |
| 4-channel DFN2 | 1,400,048 -> 1,400,416 | 3,762,336 -> 3,764,240 |
| Mono Align-ULCNet (n=5) | 568,128 -> 568,080 | 1,308,336 -> 1,308,288 |
| 4-channel Align-ULCNet (unit + stage) | 3,153,936 -> 3,117,520 + 36,400 | 6,622,128 -> 6,550,384 + 71,728 |

The READMEs, the integration manuals and the HTML pages carry the re-measured numbers; the
8 kHz mono DFN2 host is 652,336 -> 638,624.

### Migration notes for board code
- Recompile. A persisted unit or core descriptor from the previous layout is refused by `init_ex()`
  (layout 17 -> 18, 22 -> 23): query a fresh one.
- Mono: replace the `audio_pipeline_*` calls by `se_derive()` -> `aec_get_mem_size()` /
  `se_sample_chain_mem_size()` -> `aec_init()` / `se_sample_chain_init()`; per hop
  `aec_process_context()` -> `se_frame_from_aec()` -> `se_sample_chain_form()` ->
  `se_sample_chain_synthesize()`; `--aec-only` is `aec_process()`. Runtime strength moves to
  `aec_set_preset()` and `anr_set_mode(se_sample_chain_nr(chain), mode)` (see the NR entry above).
- 4-channel: `audio_pipeline_4ch_process*()` now returns the beam; call `audio_pipeline_4ch_fuse()`
  and the chain for the frequency-domain stages (`aec_res_set_preset(se_sample_chain_res(chain), ...)`
  replaces `audio_pipeline_4ch_set_aec_preset()`). Align-ULCNet code builds the unit with
  `core.enable_post = 0` and feeds `beam.spectrum`, the raw far hop and the unit's
  `delay.changed` / `delay.solid` to `ulcnet_nn_stage_begin_complex()`.
- DFN2: `dfn_hop_*` over the stage or the rate bridge replaces `*_process_begin` / `nn_begin` /
  `nn_end` / `process_end`; P must be formed before `dfn_hop_begin()`.
- The pool of each library is sized and carved separately; the 4-channel unit and the chain no longer
  share one block.
- Everything an application did with a failed hop (`reset` of the whole wrapper) is now the integrator's
  paired reset (see Changed).

### NR configuration
This entry changes no NR numbers. The NR window override it carried over into `samples/se_anr_stage.h`
is removed by the entry above.


## [Unreleased] — 2026-10-23 — Sample driver tidy-up; sources and docs condensed

### Changed
- `mcab_sample_driver`: a NULL `finish` hook counts as success; the raw and static drivers and the 4-channel
  DFN / ULCNet `main.c` use it, with a test for the NULL case.
- `dfn3_process.c` drops duplicate includes. Docs: the quick-start clone URL, archive paths (`print-lib-path`),
  layout-version numbers and the Align-ULCNet far-input description (raw far) match the code; the long design and
  upstream-alignment notes are cut to present-state tables, with history left in git.


## [Unreleased] — 2026-10-23 — Smaller AEC pool; DFN3 pre/post accessor and ERB-swap tests

### Changed
- The AEC pool shrank (`PBFDAF.scr_e2` and the filters' `is_static` fields were removed; the FFT handle
  stages its preserving transforms in one block), and the denoiser's start-up rows share the minimum
  ring, so every pipeline that composes them carries a new carve layout version, as the version
  comments require: `AUDIO_PIPELINE_LAYOUT_VERSION` 15, `AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION` 15,
  `MONO_AEC_DFN_RES_LAYOUT_VERSION` 5, `FOUR_AEC_NR_RES_LAYOUT_VERSION` 22,
  `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` 17, `AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION` 23,
  `FOUR_AEC_DFN_RES_LAYOUT_VERSION` 6 and `DFN_RATE_BRIDGE_LAYOUT_VERSION` 3. A persisted descriptor
  from the previous layout is refused by `init_ex()`; query a fresh one. The DFN stage and the
  AINR/AIAEC pre/post and model layouts are unchanged (their pools borrow the FFT handle).
- Pool totals in the memory tables and CLI examples are re-measured (for example mono 16 kHz/256
  520,672 B, 4-channel front end 16 kHz/256 1,039,904 B).

### Added
- `AINR/tests/test_dfn3_prepost_c.py`: ctypes tests of `dfn3_prepost_hop_size/num_bins/io_mode` and of
  the between-hops `dfn3_prepost_set_erb_matrices()` contract (a swap before the first hop equals
  constructing with the new pair, a mid-stream swap keeps the rows already in the window, NULL
  arguments and an open frame are refused and change nothing).

### Removed
- `dfn_rate_bridge_set_atten_lim()`, a pure forwarder; call `dfn_res_stage_set_atten_lim()` on
  `dfn_rate_bridge_stage()`, or set `atten_lim_db` at creation.

### Source comments condensed
- Release comments in the pipeline sources are cut to contracts and one-line rationale; the design
  decisions that lived only in comments are recorded here.
- Shared-delay duty cycle (`4aec_nr_res.c`): the period floor is 1 hop (a floor of 2 capped the
  `delay_est_period_s = 0.0` default at 50 % engagement); the watchdog leak is lib/aec's 0.1 dB per
  second and scales linearly with the hop period (0.001 dB per 10 ms hop), not as a retention
  power law. The opt-in throttle saved about 4x of the shared delay stage (17.0 -> 4.3 us/hop
  measured) at the price of a possibly different settled lag; with no lane clearing the ERLE arming
  threshold, re-acquisition costs up to about 2.5x on 16 kHz/256 (bounded at K).
- Shared-delay change admission uses lib/aec's Path-B numbers (movement above 32 native samples,
  offered again within 16, candidate life 3 hops). DelayAec publishes on a 16-downsampled grid
  (64 native samples at 16 kHz, 192 at 48 kHz), so the 32-sample term is a floor under a finer
  source. The TTL is not reachable from a stream (DelayAec re-offers every hop), hence the
  non-static state machine.
- An alignment change realigns the four lanes (`aec_apply_external_realign()`: warm tap transfer or
  a filter-only reset) instead of resetting them; a full reset costs one near-zero hop plus dozens
  of hops of re-exposed echo. A mixed warm/soft sweep makes the lanes' far spectra diverge (seen on a
  real 48 kHz 4-microphone fixture), so that hop skips lane-0 far-FFT sharing and the agreement
  predicate; on all other hops four far-FFT counter reads replace 4*n_freqs compares.
- Identity reprime after an alignment boundary is 1 frame in both ULCNet pipelines: both branches are
  framed from the current hop and each 50 %-overlap frame spans two hops, so only the boundary hop's
  frame straddles. The count was derived from a marker run containing no boundary.
- `delay_backward_quarantine_enabled` stays off in the 4-channel ULCNet defaults: the guard only
  delays a mis-lock by its window, so enabling it is a policy decision pending a spot check with the
  deployed checkpoint.
- DFN stage: the comfort-noise fill stays off by default because additive noise perturbs P's phase in
  the RES-cut bins, where the deep filter's estimate/apply consistency (fitted on E, applied on P)
  no longer holds; quality of the dual-input topology against a mask-only compose is to be settled
  with the Python evaluators once a checkpoint of the current contract exists.
- `post_gain`: the angle-to-bin mapping rounds in radians (SRP's rounds through degrees and differs
  on 208 of 2.4M sampled angles); the frequency smoother splits edge/interior loops (247.2 -> 34.6
  ns/frame at F=257; a running accumulator is rejected because it reassociates the sum, hand NEON
  38.0 ns); the target loop uses `&` instead of `&&` (103.0 -> 18.7 ns/frame); hoisting the time
  smoother's coefficients took 96.1 -> 26.7 ns/frame; a two-pass dilation was slower (132.5 -> 201.1);
  `sk_asym_ema_f32` is not used because it breaks ties on the other side (17.0 % of values differ).
- `fix_gain`/`nr_gain`: a `global_gain` of exactly 0 means unset (1.0) so existing tuning files keep
  their meaning; `fix_gain` clips in a separate pass through the shared kernels.
- Stage timing stays off by default: nine clock reads per 4-channel hop plus five per lane (29 in
  all), four per mono hop; the profile build sets both halves with `PROFILE=1`.

## [Unreleased] — 2026-10-23 — Frozen-digest regression checks are no longer shipped

### Removed
- The recorded `kExpect*` digests and libm probes of the AINR (DFN2/GTCRN) and AIAEC
  (Align-ULCNet/DeepVQE-S) pre/post chain tests, and the unreferenced reference-trace manifests
  `AINR/tests/state_layout_traces.sha256` and `AIAEC/tests/state_window_reference.sha256`.
  `AINR/tests/test_nn_chains.c` and `AIAEC/tests/test_prepost_chains.{c,py}` keep the whole-chain
  drive (non-finite heads and state, skipped and unwritten frames); `make -C AINR test-simd` and the
  pytest module require the NEON and forced-scalar builds to print identical digests, and each chain
  must reproduce its own digest within a process.

## [Unreleased] — 2026-10-22 — ARM32 memory descriptor compatibility

### Changed
- All 13 pipeline and neural pre/post memory-descriptor producers include the
  shared ILP32 ABI token on 32-bit targets; AArch64 hashes and the fixed 32-byte
  descriptor ABI stay unchanged. Target selection follows compiler macros.
- Initialize the DFN2/DFN3 pool-walk output pointer explicitly for GCC's strict
  ARM32 build. C11 test calls use explicit const-array views and portable
  declarations; algorithms, presets, state geometry and latency are unchanged.
- Add `tests/test_target_abi.c` to exchange real descriptors between 32/64-bit
  builds and verify rejection without modifying caller memory, independently
  of differing pool byte budgets.
- Static mono/4-channel host examples request their descriptor's alignment
  explicitly; ARM32's ordinary malloc alignment is insufficient for a 16-byte
  caller pool. RNNoise C test framing fixtures follow the same rule.
- Four-lane diagnostic sums and realign totals saturate at `LONG_MAX`.
  Saturated/rewound per-lane counts cannot establish far-spectrum provenance;
  the existing full comparison is used. Borrow/history calls and normal-range
  values are unchanged. Boundary regressions cover both long widths.

## [Unreleased] — 2026-10-21 — Build isolation with uncommitted new sources

### Fixed
- Build-isolation assertions compare the pipeline attestation's untracked flag with independently enumerated current files. New files can remain untracked during review; publish refusal policies and explicit escape hatches are unchanged.
- Added scratch-only fixtures for whitelist-only (`0`) and ordinary new-source (`1`) states. No staging or commits occur in the real checkout.

## [Unreleased] — 2026-10-21 — Parity driver include roots

### Fixed
- The standalone pipeline parity driver includes the common root for `utility/` headers, matching the production build after header relocation. Numeric drivers, frozen tolerances and expected-red registries are unchanged.

## [Unreleased] — 2026-10-21 — Common lifecycle for all four 4-channel samples

### Changed
- All four 4-channel mains now share actual CLI scanning, delay-profile
  validation and the setup/read/process/finite-check/consume/finish/cleanup
  driver. Typed adapters retain each pipeline's config, model and pool APIs.
- Preserve per-target CLI conversion/default/error policies, complete-hop
  behavior, VAD handling, finite checks and destruction-before-success order.
  The driver links into sample executables only; algorithm archives, pool
  layouts and hot-path allocation behavior are unchanged.
- Add a permanent control-flow/CLI regression gate to the 4-channel Makefile.

### Validation
- Scalar/SIMD builds for both ULCNet deployment grids pass. Across 328 paired
  before/after cases, exits, stdout/stderr, raw output files and 36 observed
  real-API sample output traces are byte-identical, including all three raw
  signal grids, explicit/automatic VAD and incomplete/empty/error inputs.
- The shared runner tests setup/read/process/consume failures, early help,
  finite-output policies and cleanup before model-sample success reporting.
  No dataset evaluation or performance conclusion is part of this change.

## [Unreleased] — 2026-10-21 — Numeric AEC presets in conventional samples

### Changed
- The mono heap/static WAV CLIs and `4aec_nr_res_static` accept AEC preset
  digits `0/1/2/3` as aliases for `mild/balanced/aggressive/no_floor`.
  Successful output labels include `no_floor`; usage text documents the mapping.
- Existing named mild/balanced/aggressive behavior, default balanced preset and
  the historical unknown-string fallback to balanced remain unchanged. This is
  CLI exposure only: no DSP equations, defaults, configs, pool or ABI changed.

## [Unreleased] — 2026-10-21 — Shared periodic sqrt-Hann window

### Changed
- Pipeline test fixtures and the RNNoise host table generator use audiolib's
  standalone `audio_window.h`. All generated RNNoise table values stay byte-identical;
  the header comment now gives the complete standalone regeneration command;
  its independent table recomputation and audiolib's frozen STFT references are
  retained. RNNoise's runtime FFT and framing are unchanged by this extraction.
- Updated the C/Python function map to locate `ac_window_build` in
  `audiolib/common/src/audio_window.c`.

## [Unreleased] · 2026-10-20/21 · Part A：`audio_common` → `audiolib` 改名＋third_party 合併

純識別字/路徑改名＋目錄合併（跨 repo C/Python 整理計畫的一部分），不含任何行為改動；
binary-golden／pyc_parity 驗證全過（細節見 commit 說明／執行報告）。

### 變更 (Changed)

- 同層 `audio_common` 檢出改名為 `audiolib`：`git mv` 原內容（`include/`、`src/`、`lib/`、`test/`、
  `scripts/`、`docs/`、`Makefile`、`README.md`、`CHANGELOG.md`、`.gitignore`）進新的 `common/` 子目錄
  （保留 git 歷史），`audio_common`/`AC_DIR` 的 Wave-S kiss 移除等未提交改動隨 `git mv` 一併帶過去。
- `pipelines/third_party/{doa,bf,VAD}` 與 `third_party/utility/{complex,spatial_simd}.*` 合併進
  `audiolib`（`doa/`、`bf/`、`VAD/`、新的 `common/utility/`）；`complex.{c,h}` 改名
  `spatial_complex.{c,h}`（guard 改名、符號已是 `spatial_complex_*` 不變）。`third_party/gain/` 留在原地
  （不進 `pipelines/` 的連結範圍，不在此次合併內）；`third_party/tests/test_spatial_third_party.c`
  依此拆成兩半：PHAT/BF/SRP/VAD 移到 `audiolib/tests/`，gain 五項測試（連同各自一份
  `check_pool_contract`）移到新的 `pipelines/third_party/gain/tests/test_gain.c`（目前手動建置驗證過，
  尚無自己的 Makefile target — 待裁定）。`third_party/Makefile` 隨之退場；`third_party/README.md` 改寫為
  只涵蓋留下的 `gain/`。
- 本檔（`pipelines/Makefile`）、`4ch_aec_bf_nr_res/Makefile`、`AEC/c_impl/Makefile`、
  `NR/c_impl/Makefile`（`lib/aec`、`lib/nr` 兩個 submodule 工作樹各自同步鏡像，不經由 submodule
  commit/push）、`AINR/Makefile`、`AIAEC/Makefile` 的 `AC_DIR`/`AC_LIB`/`AC_FWD` 改名為
  `COMMON_DIR`/`COMMON_LIB`/`COMMON_FWD`；`4ch_aec_bf_nr_res/Makefile` 的 `THIRD_PARTY_DIR` 改名
  `AUDIOLIB_DIR`。所有硬寫的 `../../audio_common` 類相對路徑改成多一層的 `../../audiolib/common`。
- `audio_common.pin` 改名 `audiolib.pin`，`AUDIO_COMMON_COMMIT` 改名 `AUDIOLIB_COMMIT`；`check-pins`
  改查整個 `audiolib` 檢出的 HEAD（新 `AUDIOLIB_DIR` 變數），不只 `common/` 子目錄；pin 的 commit 值
  不變（改名是未提交的工作樹搬移，不是新 commit）。
- `docs/c_python_function_map.md`：`pipelines/third_party/{VAD,doa,utility}` 的路徑列改成
  `audiolib/{VAD,doa,common/utility}`；`tests/test_c_python_map.py` 的路徑解析一併擴充（原本只認
  `audiolib/common/` 前綴，現在整個 `audiolib/` 檢出都能解析）。**留著不動、本輪範圍外**：表內仍有
  Part E（GSC→BF 改名）遺留的 `GSC`/`gsc_*`/`audio_pipeline_4ch_gsc_*` 等舊識別字與
  `pipelines/third_party/GSC/gsc.c` 路徑尚未同步（`test_c_python_map.py` 這兩項測試在本輪之前就已經會
  失敗，與本次 audiolib 改名無關，留給之後專門的 Part E 收尾處理）。
- 歸檔檔名不變（`libaudio_common.a` 仍是這個名字）——只有產生它的目錄路徑搬了家。

## [Unreleased] · 2026-10-20 · Part E：`third_party/GSC` 與相關 4ch 識別字全面改名為 `bf`/`BF`

純識別字/路徑改名（跨 repo C/Python 整理計畫的一部分），不含任何行為改動；`pipelines/` only。

### 變更 (Changed)

- 目錄/檔案（以 `git mv` 追蹤為搬移）：`pipelines/third_party/GSC/` → `pipelines/third_party/bf/`；
  `gsc.c`→`bf.c`、`gsc.h`→`bf.h`、`gsc_test_hooks.h`→`bf_test_hooks.h`。
- C 識別字（函式/型別/巨集，大小寫精確）：`gsc_create`/`gsc_init`/`gsc_destroy`/`gsc_get_mem_size`/
  `gsc_reset`/`gsc_reset_bin`/`gsc_process`/`gsc_process_with_weights`(`_impl`)/`gsc_get_bin_resets`/
  `gsc_get_doa_used`/`gsc_get_adaptive`/`gsc_effective_adapt_interval`/`gsc_wa_leak_factor`/
  `gsc_p_diag_floor`/`gsc_p_diag_ceil`/`gsc_rls_update_bin_scalar`/`gsc_rls_update_bins`/
  `gsc_rls_update_four_neon`/`gsc_conj` → 同動詞 `bf_` 前綴；型別 `GSC`→`BF`、`GSC_Config`→`BF_Config`；
  巨集 `GSC_H`/`GSC_WA_LEAK`/`GSC_P_DIAG_FLOOR`/`GSC_P_DIAG_CEIL`/`GSC_USE_PROJECTION_BLOCKING`/
  `GSC_TESTING`/`GSC_TEST_HOOKS_H` → `BF_` 對應；`spatial_simd` 的 `spatial_gsc_projection*`/
  `spatial_gsc_effective_weights*` → `spatial_bf_*`。
- 4ch pipeline 本地識別字（非公開 API，但仍是 C 程式碼，依既有命名清理慣例一併改）：Config 欄位
  `gsc_fixed_mode`/`gsc_fixed_doa_rad`/`gsc_fixed_align_notebook`/`gsc_adaptive`/`gsc_mu`/`gsc_lambda`/
  `gsc_adapt_interval`/`gsc_enable` → `bf_` 前綴；本地變數/欄位 `gsc_cfg`/`gsc_spectrum`/`gsc_weights`/
  `gsc_out`/`gsc_region`/`gsc_steering`/`gsc_bytes`/`saw_gsc_adaptive`/`fixed_gsc` → `bf_` 前綴；
  `audio_pipeline_4ch` 的 getter `audio_pipeline_4ch_gsc_*` → `audio_pipeline_4ch_bf_*`；測試識別字
  `test_gsc_*`/`run_gsc_*`/`_c_srp_gsc_nr_res` → `test_bf_*`/`run_bf_*`/`_c_srp_bf_nr_res`。
- Makefile/build（`third_party/Makefile`、`4ch_aec_bf_nr_res/Makefile`、`pipelines/Makefile` 三處）：
  `GSC_LIB`→`BF_LIB`、`libgsc.a`→`libbf.a`、`print-gsc-lib-path`→`print-bf-lib-path`、
  `audit-no-gsc-test-symbols`→`audit-no-bf-test-symbols`、`-DGSC_TESTING`→`-DBF_TESTING`、
  `-IGSC`→`-Ibf`。
- Python 端跟隨改名（功能性，非僅風格）：`evaluate_external_recordings.py`／`test_external_evaluator.py`
  的 JSON 契約鍵 `gsc_sample_rate`/`gsc_frame_size`/`gsc_hop_size`/`gsc_fft_size`/`gsc_adaptive_frames`
  → `bf_` 對應（這些鍵直接鏡射 C 端 `printf` 出的 JSON；C 端改名後若 Python 端不跟著改，
  `validate()` 契約檢查會全部失敗，這不是風格選擇而是維持契約可用的必要改動）；`pipeline.py`
  （`4ch_aec_bf_nr_res`、`4ch_aec_bf_dfn_res` 兩份）、`evaluate_recordings.py`、
  `tools/eval_dfn2_4ch_inputs.py` 的描述性 docstring／註解同步改字，避免指向已不存在的符號。
- 註解/說明文字：C 檔案中「GSC beamformer」一類描述性文字改寫為「beamformer (bf)」或視語境改寫
  整句，不做盲目字串替換（例如避免產生「the bf GSC」這類無意義文字；原本就是 `BF/GSC` 複合占位用語
  的句子，改寫為單一的「beamformer (bf)」）；`.md` 檔案視情況同步（非強制，但修掉會指向已不存在
  檔案/符號的失效引用，例如 `third_party/README.md` 的 `` `GSC/gsc.h` `` → `` `bf/bf.h` ``、
  `` `libgsc.a` `` → `` `libbf.a` ``）。

### 範圍確認 (Scope)

- 只動 `pipelines/` 底下的檔案；不動 `docs/*.md`、`docs/html/*.html`（這些檔案仍含歷史性 `GSC`
  字串引用，不在這次「`pipelines/` only」範圍內，已列在複核報告供之後的文件同步批次處理，故意
  延後、不是遺漏）。

### 證據 (Evidence)

- `grep -rli "gsc" pipelines/ --include=*.c --include=*.h`：改動後回零筆（C-code-only 政策要求的
  確認指令）。
- `third_party` 測試套件（SIMD=0／SIMD=1）：改動前後各 **10/10 PASS**，逐行訊息相同
  （`All third-party spatial tests passed`）。
- 4ch 測試套件（`4ch_aec_bf_nr_res/` 的 `make test`，涵蓋 third-party／`test_4aec_nr_res`／
  `test_4aec_dfn_res`／`test_audio_pipeline_4ch`／`test_audio_pipeline_4ch_ulcnet`／
  `4ch_alignulcnet`／`4ch_aec_bf_dfn_res`）：改動前後各 SIMD=0、SIMD=1 皆 **582/582 PASS**、無
  `FAIL`。
- 行為不變（binary-golden）：改動前後各自重新建置的 `audio_pipeline_4ch_raw`，對兩筆真實錄音
  （`datasets/aec_take_turn`、`datasets/aec_together`）跑 `evaluate_external_recordings.py`：
  輸出 wav 的 **sha256 逐位元相同**；JSON 量化指標（`finite`／`rms`／delay／cohort
  `attenuation_db` 等）逐值相同；raw C `summary` dict 除鍵名 `gsc_*`→`bf_*` 外其餘完全相同。
  改動後的 `validate()` 契約檢查（使用新鍵名）對改動後的二進位輸出 **PASS**。
- `audit-no-bf-test-symbols` 閂門驗證「會失敗」（非恆真閂）：對真正的 `libbf.a` 跑 **PASS**（0 個
  `bf_test_` 符號洩漏到可部署 archive）；人工把 `bf_test.o` 注入同一份 archive 副本後重新
  `nm`/`grep`，閂門正確偵測到並回報 **FAIL**。
- `pytest pipelines/4ch_aec_bf_nr_res/tests/test_external_evaluator.py`：**4 passed**。

### 代價與取捨 (Cost / tradeoffs)

- 無已知代價：純識別字/路徑改名，未觸及任何演算法邏輯、常數或控制流；第三方 GSC/VAD/doa/gain 的
  實際數學完全未動（僅改函式/型別/巨集名稱與其內部字面字串）。
- `pipelines/third_party/GSC/` 目錄已不存在；`docs/` 底下仍有歷史引用（見下）留待之後處理。

### 已知限制 (Known limitation)

- `docs/*.md`、`docs/html/*.html` 底下仍有對 `GSC`／`libgsc.a`／`third_party/GSC` 的歷史引用
  （不在這次「`pipelines/` only」範圍內），已在複核報告中列出，留給之後的文件同步批次處理。
- `third_party/Makefile` 是這次唯一完全手動改字（未經腳本）的檔案，因為它在這次工作開始前就已經
  被另一條並行工作串改動過（`BACKEND=kiss` 移除）；人工改字時已逐行核對避免衝突，驗證方式見上述
  「行為不變」證據（third_party 測試套件 10/10 PASS）。

## [Unreleased] · 2026-10-20 · S-30：4ch NO_FLOOR 的 construction-time 缺口補齊（`se_derive()`/`preset_valid()`，opus review C3）

### 變更 (Changed)

- `common/se_config.c`：`preset_valid()` 白名單加入 `AEC_PRESET_NO_FLOOR`（原本只列
  mild/balanced/aggressive，是 S-29 落地 runtime 半邊時就已標記、刻意延後的缺口——`git blame`
  確認這個函式自建檔以來沒再改過，純粹是 NO_FLOOR 加入時沒回頭更新，不是有意的限制）。檔內只有
  `se_derive()` 這一處呼叫 `preset_valid()`，沒有其他 keyed-by-preset 的 switch/array 要同步改。
- `tests/pyc_parity/pyc_drv.c`：拿掉 `run_chain()` 的「BALANCED 建構 + `aec_res_set_preset()`
  runtime retarget」workaround，`aec=no_floor` 現在直接走 `cfg.aec_preset = a->aec` 建構時路徑；
  同步更新檔頭與行內註解、移除不再使用的 `aec_res_set_preset()` include 註解。
- `tests/pyc_parity/pyc_parity.py`：`PRESETS_B` 上方與 `sec_b()` 內 no_floor 分支的註解改成
  「construction-time 直接支援」，不再描述已不存在的 retarget workaround。
- `tests/test_se_common.c`：`test_derive_rejects()` 原本的 `r.aec_preset = (AecPreset)3` 斷言
  `se_derive() == -1`（值 3 就是 `AEC_PRESET_NO_FLOOR`，修完白名單後這個斷言方向是錯的）改成
  斷言 NO_FLOOR 在建構時被接受、且 `min_gain_split_floor_enabled == 0`；真正的「超出範圍」案例
  換成 `(AecPreset)4`。`test_derive_fields()` 加一段：NO_FLOOR 建構出的 `AecConfig` 三個欄位
  （`min_gain_floor_far_active_db`/`min_gain_floor_dt_db`/`min_gain_split_floor_enabled`）與
  `aec_config_from_preset(NO_FLOOR)` 直接呼叫逐欄位相同。

### 證據 (Evidence)

- **can-fail，單元層（`preset_valid()` 修法本身）**：在 `SE_verify_20261019/scratch/
  C3_nofloor_construct/` 下的整樹拷貝裡把 `preset_valid()` 還原成修法前的三項白名單、保留新測試，
  `make SIMD=0 test_se_common` 建置後執行：4 個新斷言全部 `FAIL`（`mono derive no_floor`、欄位
  相符、`preset NO_FLOOR accepted at construction`、`NO_FLOOR construction disables the split
  floor gate`）；還原回真樹後 `./bin/.../test_se_common` 重跑 → `ALL PASS`。
- **can-fail，construction path 本身（`pyc_drv_4ch` → `conv_create()` → `se_derive()`）**：另一份
  整樹拷貝（含 `../../audio_common` 的 symlink，`PYC_WORK` 指到 `.../pyc_work_red/`），同樣還原
  `preset_valid()`、但用**修完的** `pyc_drv.c`（已拿掉 workaround、走 `cfg.aec_preset = a->aec`
  直接建構）執行 `pyc_parity_pipelines.sh --layers 1 --only b.res4.16k256.no_floor`：
  `pyc_drv_4ch` 印出 `conv_create failed`、exit code 1，Python 側 `cx.run()` 拋
  `RuntimeError('driver failed (...): 1\nconv_create failed\n')`，該 scenario 判定
  `NOT_RUN`，腳本整體 `RESULT: INCOMPLETE`、`exit=3`。還原回真樹（`preset_valid()` 含
  NO_FLOOR）後同一個 scenario 重跑：`PASS`、`max_abs=6.5e-06`、`RESULT: OK`、`exit=0`。兩份
  scratch 樹驗證完都已刪除。
- **pipelines 自測**：真樹上 `make SIMD=0 test_se_common` 與 `make SIMD=1 test_se_common` 皆
  `ALL PASS`（`exit=0`）。
- **construction-time 與舊 retarget workaround 行為等價（byte-identical）**：用
  `tests/pyc_parity_pipelines.sh --layers 1 --only b.res4.16k256.no_floor`（`PYC_WORK` 指到
  `SE_verify_20261019/scratch/C3_nofloor_construct/pyc_work_before/`）在改 `pyc_drv.c` **之前**
  先跑一次，存一份 `chain_16k256_aec_no_floor.bin`（當時仍是 BALANCED 建構 + runtime retarget
  的舊路徑，`se_config.c` 的白名單修法對這條路徑沒有影響）；改完 `pyc_drv.c`（拿掉 workaround、
  直接用 NO_FLOOR 建構）後在同一個 `PYC_WORK` 重跑同一個 scenario，兩份 `.bin` 用 `cmp` 比對
  **逐位元相同**（sha256 兩邊都是 `158dbfe0...4cbd17`）：新的 construction-time 路徑與舊的
  runtime-retarget workaround 在同一筆真實錄音輸入上輸出完全一致。
- **三個 grid 的 `b.res4.*.no_floor` 數字與 S-29 落地時記錄的完全一致**（6.50e-06 / 4.47e-06 /
  3.93e-06，皆 ≤ 2e-5 容差），`--layers 1 --only b.res4` 全 PASS（13 列）。vacuity guard
  （`c_gres_vs_balanced_max_abs`：直接建構的 C no_floor 輸出與 C balanced 輸出差多少）在新的
  construction-time 路徑下量到 0.9747 / 0.9622 / 0.9799——跟 S-29 記錄的數字一致，證明 gate
  在「直接建構」下一樣真的生效，不是退化成跟 balanced 沒差的建構。
- **`pyc_parity_pipelines.sh --layers 1,2`（全量）**：84 PASS、1 SKIP、1 INFO，與改動前的基準
  （S-29 記錄的 84/1/1）完全一致，逐列無退步，`RESULT: OK`。
- **4ch C 測試套件**：`4ch_aec_bf_nr_res/` 下 `make SIMD=0 test` 與 `make SIMD=1 test` 均
  `exit=0`，`third_party`/`4aec_nr_res`/`audio_pipeline_4ch`/`audio_pipeline_4ch_ulcnet` 四個
  套件都印出「All ... tests passed」，無 `FAIL`；`pipelines/` 上層 `make SIMD=0/1 test`
  （mono 系列）亦均 `exit=0`、無 `FAIL`。
- **4ch Python 測試**（未改動，確認不受影響）：`pytest
  pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py` 49 passed。

### 代價與取捨 (Cost / tradeoffs)

- 無已知代價：MILD/BALANCED/AGGRESSIVE 既有三個 preset 的 construction-time 行為完全不變
  （`preset_valid()` 只新增一個 `||` 分支）；S-29 遺留的 `pyc_drv.c` 技術債已清償，
  `tests/pyc_parity/pyc_drv.c` 不再需要為 NO_FLOOR 維護一條平行的 retarget 路徑。
- `se_derive()` 是 mono（`mono_aec_nr_res/audio_pipeline.c`）、ulcnet
  （`mono_alignulcnet/audio_pipeline_ulcnet.c`）、4ch 三條應用共用的同一個函式，這次修法連帶把
  construction-time NO_FLOOR 對 mono／ulcnet 也打開了，不只 4ch——頂層 `pipelines/` 的
  `make SIMD=0/1 test`（含 mono／ulcnet 套件）已跑過確認無迴歸，但本輪沒有另外替 mono／ulcnet
  加 NO_FLOOR 專屬斷言，範圍就 opus review C3 點名的 4ch 缺口。
- 觀察到但不在本輪範圍內的後續缺口：`4ch_aec_bf_nr_res/4aec_nr_res_static.c`、
  `mono_aec_nr_res/main.c`、`mono_aec_nr_res/static_main.c` 的命令列字串解析（`"mild"`/
  `"balanced"`/`"aggressive"`）都沒有 `"no_floor"` 分支，所以這幾個執行檔的 CLI 仍然碰不到
  NO_FLOOR（會落到預設的 BALANCED）；留給後續輪次，這次沒動。

## [Unreleased] · 2026-10-20 · S-29：4ch NO_FLOOR 的 runtime 半邊補齊（`PostBeamResidualSuppressor.set_preset()`）

### 變更 (Changed)

- `4ch_aec_bf_nr_res/pipeline.py`：`PostBeamResidualSuppressor.set_preset()` 新增第三個欄位的
  retarget——`min_gain_split_floor_enabled`（經 `SuppressionGain.set_split_floor_enabled()`，不帶
  ramp，跟 DT 地板一樣立即生效）。之前這個 setter 只動兩個 dB 欄位（far-active／DT 地板），把一條
  已經在跑的 4ch suppressor 切到 `AEC_PRESET_NO_FLOOR` 不會真的關掉地板機制——gate 停在建構時的值
  （預設 `True`，等於 BALANCED），只有建構時傳 `aec_preset=NO_FLOOR` 才有效（這一半是先前 S-26+
  NO_FLOOR construction 那輪已經落地的）。C 側（`aec_res_set_preset()`）已經是三欄位一起 retarget；
  這次把 Python 的 runtime 半邊補齊，兩邊重新一致。
- `audio_pipeline_4ch.h`：`audio_pipeline_4ch_set_aec_preset()` 的文件註解改成三欄位（原文只提兩個
  gain 地板），加一句 NO_FLOOR 可在 runtime 切換（gate 跟兩個 dB 欄位同一次呼叫一起 retarget，沒有
  中間態）。
- `README.md`：**Presets** 清單加 `AEC_PRESET_NO_FLOOR`；preset 強度表從「兩欄位」改成「三欄位」
  （新增 `min_gain_split_floor_enabled` 一列）；Runtime strength control 段落加一段說明 NO_FLOOR
  可在運轉中的 instance 上切換，mono／4ch、Python／C 四邊都支援。
- `4ch_aec_bf_nr_res/tests/test_pipeline.py`：新增 4 個測試——
  `test_preset_retarget_moves_split_floor_enabled_too`、
  `test_preset_retarget_split_floor_enabled_survives_reset`、
  `test_construction_time_no_floor_matches_runtime_retarget`、
  `test_runtime_no_floor_switch_changes_suppressor_gain_output`（最後一個直接比較
  `PostBeamResidualSuppressor.process()` 在一個地板真的會生效的場景——converged、far-active latch
  一打就中、r2 遠大於 near_spec——上，BALANCED 跟「runtime 切到 NO_FLOOR」兩個 instance 算出來的
  `res_gain` 確實不同，不只是內部旗標翻轉）。順便修正兩個失效的舊註解：`__init__`／
  `FourChannelAecConfig.aec_preset` 原本寫「NO_FLOOR 沒有 runtime setter」，`test_pipeline.py` 一條
  舊測試的 docstring 也這樣寫，三處都已經過期（construction-time 半邊落地時就已經過期，不是這次新
  造成的，只是一直沒人回頭改）。
- `tests/pyc_parity/pyc_drv.c` / `pyc_parity.py`：新增 `b.res4.<grid>.no_floor` parity scenario
  （3 個，一個 grid 一個）。C 側 `se_derive()` 的 `preset_valid()`（`pipelines/common/se_config.c`）
  還沒把 `AEC_PRESET_NO_FLOOR` 列入白名單（獨立、已另外標記的缺口，這次不修），所以 `pyc_drv.c`
  的 `run_chain()` 用 BALANCED 建構、再呼叫 `aec_res_set_preset(c->res, AEC_PRESET_NO_FLOOR, 0)`
  retarget——跟 Python 側 `py_res()` 本來就有的「balanced 建構 + `set_preset()`」配對完全對應。這
  三列**沒有**動 `PRESETS`（影響 `l2.res4.*.replay` 等其他 scenario 的共用清單），只加進 `sec_b`
  自己用的 `PRESETS_B = PRESETS + ['no_floor']`。

### 證據 (Evidence)

- **pytest**：`PYTHONPATH=Audio_ALG python3 -m pytest
  Audio_ALG/pipelines/4ch_aec_bf_nr_res/tests/test_pipeline.py -q` → **49 passed**（45 條既有 + 4
  條新增）。can-fail：把 `set_preset()` 裡新加的
  `self._gain.set_split_floor_enabled(enabled)` / `self._split_floor_enabled = enabled` 兩行直接在
  真實樹上註解掉、跑完測試馬上還原，4 個新測試全部 RED（
  `test_construction_time_no_floor_matches_runtime_retarget` 斷言 `False == True`；
  `test_runtime_no_floor_switch_changes_suppressor_gain_output` 斷言 `_gate(pipeline2) is False` 失敗）；
  還原後重跑 49 passed。
- **pyc_parity**（`pyc_parity_pipelines.sh --layers 1,2`，`PYC_WORK` 指到
  `SE_verify_20261019/scratch/nofloor-4ch-runtime-python/`）：**84 PASS、1 SKIP、1 INFO**（改動前
  跑一次基準是 81 PASS／1 SKIP／1 INFO；新增的 3 列剛好補上差額，沒有任何既有列的狀態改變）。三個
  新列的實測數字：

  | scenario | Python vs C G_res max_abs | tolerance | C no_floor vs C balanced max_abs（gate 有沒有真的動） | double-talk hops |
  |---|---|---|---|---|
  | `b.res4.16k256.no_floor` | 6.50e-06 | ≤2e-05 | 0.9747 | 883/1250 |
  | `b.res4.16k512.no_floor` | 4.47e-06 | ≤2e-05 | 0.9622 | 436/625 |
  | `b.res4.48k1024.no_floor` | 3.93e-06 | ≤2e-05 | 0.9799 | 655/937 |

  「C no_floor vs C balanced」那欄是這次新加的 vacuity guard：三列都遠大於 0，證明 NO_FLOOR 的
  runtime retarget真的把 C 的輸出推離了 balanced（不是建構出一個跟 balanced 沒差的 instance），
  guard 本身若量到 0 會把該列判成 `NOT_RUN` 而不是虛假的 PASS。
- **既有 83 列（balanced／mild／aggressive／NR／mono／realign／Layer 2 全部）完全未受影響**：改動
  前後兩次 `--layers 1,2` 全跑的 `results.json` 逐列比對（`status`／`gate`／`metrics`／
  `tolerance`），83 列 diff=0；9 個 `chain_<grid>_aec_<preset>.bin`（balanced/mild/aggressive ×
  3 grid）seam 檔 `cmp` 逐位元相同。
- **can-fail，pyc_parity 側**：同一個 Python mutation（註解掉 gate retarget 兩行）下用
  `--layers 1 --only b.res4 --reuse-build` 重跑：3 個 `no_floor` 列全部 FAIL（max_abs 0.975／
  0.962／0.98，等於「Python 停在 balanced，C 已經是 no_floor」兩邊的真實差距），`balanced`／
  `mild`／`aggressive` 9 列不受影響仍 PASS，整個呼叫 `exit=1`。還原後重跑：12 列全 PASS，`exit=0`。
- **C 側 4ch 測試套件（`4ch_aec_bf_nr_res/`，這次只動了 `.h` 裡的文件註解，純驗證無迴歸）**：
  `make SIMD=0 test` 與 `make SIMD=1 test` 皆 `exit=0`；兩次各自的
  `test_4aec_nr_res`／`test_audio_pipeline_4ch`／`test_audio_pipeline_4ch_ulcnet`／third-party
  spatial 測試都印出「All ... tests passed」且沒有任何 `FAIL`；兩次各自 grep 到的 `PASS`/`PASS:`
  行數都是 572（SIMD=0 backend=scalar／SIMD=1 backend=aarch64-neon，兩次行數相同本身就是一個
  一致性訊號）。

### 代價與取捨 (Cost / tradeoffs)

- `set_preset()` 新增一個布林欄位的寫入路徑；因為沒有驗證需求（任何真值都合法，跟
  `set_initial_state()` 同一個契約），放在兩個會拋例外的 dB setter **之後**寫，順序本身就是防呆
  （拒絕的呼叫不會留下「dB 改了但 gate 沒改」的半套狀態），不是額外成本，但是後續有人要再加第四個
  retarget 欄位時要記得遵守同一個順序規則。
- `pyc_drv.c` 的「BALANCED 建構 + runtime retarget」是**測試 harness 的 workaround**，不是產品路徑
  ——真正的缺口是 `se_derive()`／`preset_valid()` 還沒把 `AEC_PRESET_NO_FLOOR` 納入白名單（C 側
  construction-time 的 NO_FLOOR 仍然是一個 Python→C 不對稱：Python 的
  `FourChannelAecConfig(aec_preset=NO_FLOOR)` 建構沒問題，C 的 `se_derive()` 會拒絕）。等那個缺口
  真正修掉之後，這個 workaround 要跟著拿掉，`pyc_drv.c` 可以直接讓 `aec=no_floor` 走construction-time
  路徑。
- 本益：好處是「運轉中切 NO_FLOOR」現在 Python／C、mono／4ch 四邊語義一致，不會有切換後 gate 沒跟上
  的隱性退化；量到的既有 preset 數值成本是零（83 列既有 parity scenario 與 49 條既有 pytest 全部
  逐位元/逐值不變），但不是沒有成本：上一條提到的 `pyc_drv.c` workaround 本身就是一筆要還的技術債，
  `se_derive()` 補上 NO_FLOOR 白名單之前都要留著。

## [Unreleased] · 2026-10-20 · S-28：凍結 `pyc_parity_pipelines.sh` 容差（原卡在 S-27）

### 變更 (Changed)

- `tests/pyc_parity/pyc_parity.py` 的 `TOL1`／`TOL2` 從「第一次量測的暫定值」改為凍結值。凍結前先確認
  `EXPECTED_RED = {}`（S-25／S-26／S-27 都已真正落地，沒有殘留的已知缺口標記），再用
  `pyc_parity_pipelines.sh --layers 1,2` 對現在的真實樹重跑一次全部 scenario：81 PASS、1 SKIP
  （`d.mono.8k`，Python mono pipeline 本就不支援 8 kHz）、1 INFO（`d.mono.cng_lookalike`，本就不判
  gate），沒有 FAIL／UNEXPECTED_GREEN／NOT_RUN，跟 S-27 條目自己重跑的數字一致。
- 凍結值 = 對每個常數「所有引用它的 scenario、兩個 layer」量到的實際最大誤差，取約 1.8–2.7×
  margin（dB 常數取線性域約 1.9× 的 margin）後取一個乾淨數字：

  | 常數 | 舊（暫定） | 新（凍結） | 這次量到的最大值（scenario） | margin |
  |---|---|---|---|---|
  | `TOL1.gain_abs` | 1e-4 | **2e-5** | 8.3e-6（`d.mono.16k256.nr_glue`） | 2.4× |
  | `TOL1.out_abs` | 1e-6 | **2e-7** | 7.5e-8（`d.mono.48k1024.min_apply_ola`） | 2.7× |
  | `TOL1.cng_abs` | 0.0 | 0.0（不變） | 0.0（18/18 case bit-exact） | — |
  | `TOL2.replay_gain_abs` | 5e-2 | 5e-2（不變） | 0.0280（`l2.res4.48k1024.mild.replay`） | 1.8× |
  | `TOL2.replay_out_abs` | 2e-3 | **1e-3** | 4.25e-4（`l2.mono.16k256.replay.out`） | 2.4× |
  | `TOL2.chain_gain_p99_9` | 5e-2 | 5e-2（不變） | 0.0286（`l2.chain4.48k1024.gres`） | 1.7× |
  | `TOL2.chain_out_p99_9` | 1e-3 | **4e-4** | 1.69e-4（`l2.chain4.48k1024.out`） | 2.4× |
  | `TOL2.chain_rms_db` | -40.0 dB | **-45.0 dB** | -50.6 dB（`l2.chain4.48k1024.gres`，最差／最接近門檻） | 5.6 dB（~1.9×） |

  三個 Layer 1 常數（std-math C vs Python）都明顯收緊（13× margin 收到 2.4–2.7×）；Layer 2
  （fast-math/NEON shipped vs std-math C）的 `replay_out_abs`／`chain_out_p99_9` 也收緊到同一量級。
  `replay_gain_abs` 與 `chain_gain_p99_9` 維持 5e-2 不變——**不是因為這兩個本來就夠緊**：這次重跑量到
  的最大值（0.0280／0.0286）比 S-25/26/27 落地前的量測（同一台機器上的 `det1.out` 對照：
  `l2.res4.*.replay` 最大 0.0155、`l2.chain4.48k1024.gres` p99.9 0.0173）大了接近 2×，因為
  S-25/26/27 新接上／新啟用的路徑（R² fold、DT 地板、mixed realign 的 lane-0 far_spec 共用）讓
  Layer 2（fast-math/NEON）跟 Layer 1 之間新多出可觀的分歧量。維持 5e-2 是「這次量到的最大值
  （0.0286）乘以約 1.75× margin」算出來的結果，剛好跟舊的暫定值一樣，是巧合，不是「這兩個不用緊」。
- 套用凍結值後重跑：verdict 與 before 完全一致（81 PASS／1 SKIP／1 INFO，沒有任何列從 PASS 變
  FAIL），`--layers 1,2` 連跑兩次的 `table.out` 與 `results.json` 逐位元相同
  （sha256 都是 `973361a2...`／`3bcc4818...`）。這次凍結所量測的 8 個 `SOURCES` 檔案的 sha256
  （`results.json` 的 `sources_sha256`，`run1`／`run2_frozen`／`run3_frozen` 三次重跑逐檔相同）：
  `4aec_nr_res.c`=`0c8d9ec3f7db...`、`pipeline.py`=`bb0f3e09e223...`、`se_anr_stage.c`=`b06484c9c809...`、
  `aec_res.c`=`c494f9b3f054...`、`audio_pipeline.c`（mono）=`afae24e47412...`、
  `aec_nr_pipeline.py`=`9e701d972d80...`、`cng_aec3.py`=`f8040b813f49...`、
  `conv_chain_4ch.h`=`89cdfae37710...`（皆取前 12 hex 字）——凍結值可以追溯到跑出這些數字的確切程式碼。
- 連帶把輸出介面上宣告「還沒 freeze」的文字改掉：每列的 `tolerance_provisional` 欄位改名
  `tolerance`（repo 內沒有別的程式讀這個 JSON 欄位，純粹是這個檔案自己的輸出，改名不影響任何消費
  端）；JSON 的 `tolerances` 欄位值從 `"provisional"` 改成 `"frozen_2026_10_20_S28"`；終端輸出的
  `tolerances are PROVISIONAL (not frozen)` 改成 `tolerances are FROZEN (S-28, 2026-10-20)`；
  `pyc_parity_pipelines.sh` 開頭的說明段落同步改掉。

### 證據 (Evidence) — mutation-proof 證明凍結後的容差仍然「會失敗」

- `pyc_parity_pipelines.sh --mutation-proof`（內建機制：在一份 scratch 副本裡一次注入一個真實的小
  偏差，確認對應列變 FAIL，再還原確認變回 PASS；詳見 `pyc_parity/mutation_proof.sh` 開頭的機制說明）
  在凍結後的樹上重跑，所有既有、跟容差數值相關的 mutation 仍被偵測到，而且因為容差變緊，偵測門檻比
  凍結前更靈敏：
  - `fold_coef_ulp`（`se_anr_stage.c` 的 R² fold 係數偏移 K 個 float32 ULP）：凍結前（`gain_abs`=1e-4）
    最小偵測到的 K 是 256（見本輪量測前、同一台機器上跑的對照：`SE_verify_20261019/scratch/PARITY_PIPE/mutation_full.log`）；
    凍結後（`gain_abs`=2e-5）K=16 就被 `d.mono.16k256.nr_glue` 抓到、K=256 開始連
    `a.nr4.16k256.glue_cflag`／`glue_fold` 一起 FAIL——偵測門檻變靈敏了約 16×，K=1 兩邊都還是「no」
    （1 ULP ≈ 1.2e-7 相對值，遠低於 Python/C 本來就有的 ~1e-5 量級一致度，兩個容差都量不到，這是
    文件裡本來就說明過的已知限制，不是這次改動造成的）。
  - `xi_min_db_div10_5`（NR `xi_min_db / 10.0f` 改成 `/ 10.5f`）、`cng_seed_1bit`（CNG RNG seed 改 1
    bit）、`aec_floor_balanced_ulp`（RES balanced 地板偏移 K ULP，K=4096 起偵測到，跟凍結前一樣，
    因為這條走的是 RES 路徑、數值鏈路跟 NR 路徑不同，門檻沒有變）、`usable_gate_dropped`（R² fold 的
    `usable_linear` 閂拿掉，跟這次改的 `gain_abs` 直接相關）：全部仍被偵測到（RED，該次子跑
    `exit=1`），對照組（不注入任何 mutation）本身是 `exit=0`，證明「沒有 FAIL 不是因為 harness 本身
    壞了」。還原 mutation 後（腳本自動 `cp` 回原檔）下一個 mutation 的對照同樣 `exit=0`，證明每次都
    確實變回 GREEN。整個 `--mutation-proof` 呼叫本身最終是 `exit=2`（見下面「已知限制」的 flip_s27
    段落），不是 0——這裡回報的是個別 mutation 子跑的 exit code，不是整個指令的 exit code。
  - `mutation_proof.sh` 用 `--layers 1`（只跑 Layer 1），不會碰到 `TOL2` 的
    `replay_out_abs`／`chain_out_p99_9`／`chain_rms_db`。這三個改過的 Layer 2 常數另外手動證明
    「會失敗」：在真實樹的 `pyc_parity.py` 把三個值暫時改到比這次量到的最大值更緊（`replay_out_abs`
    1e-3→2e-4、`chain_out_p99_9` 4e-4→1e-4、`chain_rms_db` -45.0→-51.0dB），用
    `--reuse-build`（沿用同一個 build，不用重編譯）跑
    `--only l2.mono.16k256.replay.out,l2.chain4.48k1024.out,l2.chain4.48k1024.gres`：三列全部
    FAIL（`exit=1`，`RESULT: FAIL`）。改回真正的凍結值（1e-3／4e-4／-45.0）用同樣的 `--only` 重跑：
    三列全部回到 PASS（`exit=0`，`RESULT: OK`），數字跟之前的量測（`max_abs=0.000425`／
    `p99_9=0.000169`／`rms_db=-50.6`）完全一致。
  - 完整表格見 `scratch/S28_freeze_pyc_tol/mutproof.log`
    （Layer 1 的 5 個 mutation）；TOL2 的 RED/GREEN 紀錄見同目錄下
    `run3_frozen/canfail.out`（RED）與 `run3_frozen/canfail_restored.out`（GREEN）。

### 已知限制 (Known limitation)

- 這次凍結的量測只在這台機器、這個編譯器（Apple clang 17、arm64 macOS）上做，沒有跨機器／跨
  toolchain（例如目標平台的交叉編譯器）重跑過；凍結值的 margin（1.8–2.7×／~1.9× in amplitude）是
  根據「多一次 libm／NEON lowering 版本差異，不應該把 PASS 翻成 FAIL，但誤差加倍應該被抓到」這個
  經驗法則訂的，不是從多台機器的實測分佈反推出來的統計門檻；如果之後真的在另一個 toolchain 上出現
  誤報，margin 需要重新量測調整。
- `TOL2.chain_rms_db` 用 dB 表示、margin 只有 5.6 dB（~1.9× in amplitude）：如果真實回歸把誤差放大
  不到 1.9 倍（例如放大 1.5 倍，從 -50.6 dB 變成約 -46.9 dB），仍然會落在新門檻 -45.0 dB 之內、不會
  被這個常數單獨抓到——跟其他線性容差「加倍一定抓到」的保證不完全一樣，這是 log 尺度本身的取捨，
  已誠實記錄不是「無成本」。
- **附帶發現、跟這次凍結容差無關、不在這次範圍內修，但重要（澄清：是測試選錯列，不是 harness 量
  不到這個回歸）**：`mutation_proof.sh` 的 `--mutation-proof` 這次整個呼叫是 `exit=2`（不是全綠），
  因為跑到 `flip_s27_python_borrow` 時直接中止：它要比對的
  `raise ValueError("linear AEC lanes do not share one far-end spectrum")` 字串在現在的
  `pipeline.py` 裡已經找不到（S-27 真正的修法就是刪掉這一行），`perl` 替換沒有任何效果，
  `mutation_proof.sh` 自己偵測到「檔案沒被改到」就回報 `pattern drifted?` 並 `return 2`。
  在它之前的 `flip_s25_c_flag_forced`（讓 `four_aec_fuse()` 強制 `frame->usable_linear = 1`）、
  `flip_s26_python_dt_floor`（讓 `PostBeamResidualSuppressor.set_preset()` 再多做一次 DT 地板
  retarget）兩個子跑都回報「沒偵測到」（`a.nr4.16k256.legacy_amin`／`b.res4.16k256.aggressive`
  都還是 PASS）。**查證後確認：這不是 harness 量不到這個回歸，是這兩個 flip 測試自己檢查的列選錯
  了**。手動把 `flip_s25` 的 mutation（強制 `usable_linear=1`）重新套到 scratch 副本、把 `--only`
  擴大到也包含 `a.nr4.16k256.verdict_flag`（原本的 `M_ONLY` 沒有這一列）：這一列立刻變 FAIL
  （`mismatch 288 of 1250 hops, Python verdict true 962, C flag true 1250`，`exit=1`）；還原
  mutation 後重跑同一組 `--only`：三列都回到 PASS（`exit=0`）。也就是說，`usable_linear` 被強制設
  1 這種回歸，`a.nr4.*.verdict_flag`（逐 hop 比對 Python 判定 vs C 旗標）本來就抓得到，`flip_s25`
  只是檢查了 `legacy_amin`／`glue_cflag` 這兩個已經不會因為這個特定 mutation 而改變的列
  （`legacy_amin` 測的是另一個獨立開關；`glue_cflag` 在 16k256 這個 grid 碰巧量到的差異本來就低於
  容差，forcing usable_linear=1 沒有把它推過門檻）。`flip_s26` 大概同理（疑似把「S-26 真正的修法」
  又套了一次、變成冗餘動作），但沒有像 `flip_s25` 一樣逐列重新驗證過，這點保留、不下定論。
  這三個 flip case 本質上是「S-25/26/27 還沒真正落地、`EXPECTED_RED` 還非空」那個時期寫的，現在
  `EXPECTED_RED = {}` 之後，它們檢查的列組合已經過期，需要重新設計（例如 `flip_s25` 應該把
  `a.nr4.*.verdict_flag` 也納入 `M_MUST`），不是這次凍結容差的範圍，留給之後整理 `mutation_proof.sh`
  的人。
- **附帶發現、已查明是良性、不影響這次的量測**：凍結容差的過程中，`4ch_aec_bf_nr_res/4aec_nr_res.c`
  在工作樹上被改動過一次（本次工作期間；`git status` 顯示它跟 `pipeline.py` 一樣是既有的
  uncommitted 修改）——sha256 從量測時的 `0c8d9ec3...`（`run1`／`run2_frozen`／`run3_frozen` 三次都
  是這個值）變成之後的 `97bbba58...`。逐行 diff 確認只有一段註解被精簡（S-27 那段「fused X 還是取
  lane 0 的 far_spec」的說明文字去重），沒有任何數字、邏輯或空白敏感的程式碼改動；`pipeline.py` 全程
  sha256 不變（`bb0f3e09...`）。拿這之後的狀態重跑一次完整 `pyc_parity_pipelines.sh --layers 1,2`
  （`run4_final`）：81 PASS／1 SKIP／1 INFO，跟 `run2_frozen` 的 `table.out` 逐位元相同、83 列的
  status 逐列相同——確認這次凍結的容差值對這個註解級別的改動不受影響，不需要重新量測。

## [Unreleased] · 2026-10-20 · S-27：4ch 混合 warm/soft-realign hop 的 far-spectrum 處理

### 修復 (Fixed)

- **4ch 前端在混合 warm/soft realign 的那一個 hop 上，原本仍無條件把 lane 0 的 far FFT 共享給
  其他三個 lane**（`four_aec_front_process()` 的 per-lane loop），即使 soft lane 在同一 hop 已被
  `aec_apply_external_realign()` 清空 far_buffer OLA history、warm lane 沒有——兩者的真實 far FFT
  本該不同，卻被當成相同處理；`fuse_contexts()` 的 provenance 快路徑（`far_spec_shared_hop`）因為
  計數器證據（lane 0 跑 1 次 transform、lane 1-3 跑 0 次）仍然成立而直接 accept，從未察覺這個差異，
  soft lane 靜默吃到 lane 0 未重置的 far_spec（錯誤但不報錯）。

  修法是「路由」層級的改動，沒有新增任何 DSP/FFT 邏輯：`aec_process_context()`（每個 lane 原本就有、
  非共享情境下每個 lane 本來就會用的真正 far FFT）早就存在，只是這一個 hop 從沒被路由過去。新增
  `FourAecNrRes.far_spec_mixed_hop`（跟既有 `far_spec_shared_hop` 同 precedent），在
  `delay.changed` 的 realign 迴圈裡用「這一 hop」的 warm/soft outcome 計數（不是累計總數）判定
  `hop_warm>0 && hop_soft>0`；per-lane loop 的分支條件從 `ch==0` 改成
  `ch==0 || p->far_spec_mixed_hop`，讓每個 lane 在混合 hop 上各自跑自己的真實 far FFT（這一 hop
  最多多付 3 次 FFT，符合先前估算）。既有的 provenance 計數器邏輯本身沒動——lane 1-3 這時
  `ran==1`（不再是 0），自然讓 `lanes_borrowed=0`、`far_spec_shared_hop=0`，如實反映「這 hop 沒有
  共享」。

  只做上面這一半、不放寬 `fuse_contexts()` 的 far_spec 一致性檢查，會讓每個混合 hop 變成全部
  reset（`four_aec_nr_res_far_spec_agrees()` 的全比較分支會如實偵測到四個 lane 的 far_spec 現在真的
  不同而 reject，`four_aec_fuse()` 接著呼叫 `four_aec_nr_res_reset()`）——比現有的靜默錯誤還糟，等於
  把每次混合 realign 都變成一次全重置。因此兩半一起做：`fuse_contexts()` 在呼叫
  `four_aec_nr_res_far_spec_agrees()` 之前先檢查 `p->far_spec_mixed_hop`，是的話整段比較都跳過——
  不是放寬斷言本身，`four_aec_nr_res_far_spec_agrees()` 這個 predicate 維持原語意不變，
  `test_far_spec_provenance_guard` 仍直接驅動它；融合後交給下游的 `X`（`four_aec_fuse()` 裡的
  `frame->x`）不管哪種情況都還是取 lane 0 的 far_spec，跟 `pipeline.py` 的語意一致。

  Python 側對應：`pipeline.py` 的 `_fuse_contexts()` 本來就沒有任何 FFT 共享優化（每個 lane 本來就
  各自算自己的 context），差異只在它原本對跨 lane far_spec 不一致會 `raise ValueError`。移除這個
  raise，改成接受並直接取 `far_specs[0]`（跟 C 核心融合後的 `X` 同一語意），其他三個 lane 保留自己
  未融合的 far_spec。

  連帶更新了三處描述這個機制的註解（`4aec_nr_res.c` 的 `four_aec_nr_res_far_spec_agrees()` 上方、
  同檔案 per-lane loop 裡）、`4aec_nr_res_internal.h` 裡宣稱「lane 永遠共享同一個 far spectrum，
  所以這個 predicate 的拒絕分支測試不到」的過期說法、以及 `README.md`「Far-end FFT sharing」整段
  與開頭的流程圖，全部改成反映修法後的實際行為（而不是像草稿版本那樣停在「這裡還沒修」）。

- `tests/pyc_parity/pyc_parity.py`：確認 `pyc_parity_pipelines.sh --only c.realign4` 重跑後
  `c.realign4.48k1024` 變成 `PASS`，才移除 `EXPECTED_RED['c.realign4.48k1024'] = 'S-27'` 標記；
  同步修正兩處描述舊「raise」語意的過期文字（該 scenario 的標題、`verdict_flag` 行的 producer
  字串），並把 `py4_run()` 裡原本靠 monkeypatch `_fuse_contexts()`、`except ValueError` 來模擬
  「lane 0 far_spec 融合」的輔助邏輯整個拿掉——那段邏輯現在是死碼（真正的 `pipeline.py` 已經直接
  做這件事，不會再 raise），改成直接用 pipeline 自己的 `realign_warm_lane_count`／
  `realign_soft_lane_count` 計數混合 hop 數，供既有的 `s27_borrowed_hops` 指標使用。

### 證據 (Evidence)

- `pyc_parity_pipelines.sh --only c.realign4 --layers 1`：修法前 `c.realign4.48k1024` =
  `EXPECTED_RED(S-27)`（`python_raised=True` at hop 190、
  `python_message="linear AEC lanes do not share one far-end spectrum"`；C 側當時沒被影響，
  `c_fail_hop=-1`，因為舊碼是強制共享而非報錯——這條本身是「靜默吃錯值」而非「崩潰」）。修法後
  重跑同一 scenario：`PASS`（`python_raised=False`，C 側 `c_fail_hop=-1` 不變、
  `c_mixed_warm_soft_hops=[190]` 確認那一 hop 真的是混合 hop、且沒有觸發
  `four_aec_nr_res_reset()`）。`pyc_parity_pipelines.sh --layers 1,2`（全部 scenario）重跑：
  81 PASS、1 SKIP（8k，既有、跟這次修法無關）、1 INFO，`RESULT: OK`，沒有任何 `FAIL` 或
  `UNEXPECTED_GREEN`；Layer 2（SIMD=1 shipped build）的 `l2.realign4.48k1024` 也 PASS（events
  `[(190, 2, 2)]`）。
- `measure_mixed_realign.py`（唯讀，驅動 Python 參考 pipeline 的 `front_process()`，不呼叫
  `fuse()`，所以不直接測到這次修法動到的程式碼，但確認 realign 計數本身沒被動到）：修法前後對
  `datasets/aec_together`、`datasets/aec_take_turn` 兩個真實 4-mic 錄音跑出**完全相同**的結果——
  共 12620 個 hop，3 次 realign，其中 1 次是混合 hop（`aec_together` hop 976：3 個 lane warm、
  1 個 lane soft，`delta=60`），0 次混合且 `delta<0`（retard）。
- 4ch 測試套件（`4ch_aec_bf_nr_res/` 的 `make SIMD=0 test`、`make SIMD=1 test`）：572/572 全部
  PASS，兩個 SIMD 設定都是。pipelines 頂層（mono）測試套件（`make SIMD=0 test`、
  `make SIMD=1 test`）：全部 PASS，exit code 0，沒有 `FAIL`——mono pipeline 完全不含 multi-lane
  realign 的概念，架構上就沒有路徑會碰到這次改動的任何檔案（`4aec_nr_res.c`、
  `4aec_nr_res_internal.h`、4ch 的 `pipeline.py`），這次重跑是確認性的，不是「可能受影響」。

### 已知限制 (Known limitation)

- 目前唯一能取得的真實錄音證據只有 **3 次 realign、其中 1 次混合 hop、0 次混合且 `delta<0`**
  （來自 `aec_together`／`aec_take_turn` 兩個既有 4-mic 錄音）。這是很小的樣本，尤其
  「混合 hop 又正好是 delay 後退（retard）」的組合完全沒有實例可驗證。使用者已核准在這個樣本量
  下先行落地此修法，重新驗證（在更大樣本或新的多 realign 語料上）延後到有這樣的語料時再做。

## [Unreleased] · 2026-10-20 · S-25：4ch NR 重新接上 `usable_linear`（遺失更新回歸）

### 修復 (Fixed)

- **`4ch_aec_bf_nr_res/4aec_nr_res.c` 的 `four_aec_fuse()` 從未寫入 `frame->usable_linear`**，
  一律是 `memset` 留下的 0。這是兩個 commit 之間的遺失更新：`49f73b2` 加進的那段邏輯，被
  `af914d4` 意外刪掉的 9 行帶走，之後沒人補回來。後果：NR glue 永遠把已收斂的線性 AEC 估計
  當成「不可信」，R² fold 對 4ch 一直是死碼，double-talk／far-only 時少做了該有的回音／噪聲
  抑制；near-end-only 的行為不受影響（fold 只在 `usable_linear` 為真時啟動，一直是假，所以
  一直不啟動，近端這段時間本就沒有 fold 的差異可言）。
  修法：補回 8 個插入點——`FourAecLaneSnapshot` 加 `usable_linear` 欄位、`bind_lane_view()` 從
  `AecResContext` 複製該欄位（`AecResContext.usable_linear` 本身一直都有，只是沒傳到這個快照
  裡）、`fuse_contexts()` 的簽名／seed／AND-fold（只 AND 過 `lane_contributes` 為真的 lane，
  和 `all_converged` 的 fold 同一個 gate，不是對全部四個 lane 做 AND）、`four_aec_fuse()` 裡的
  區域變數與呼叫處、最後寫回 `frame->usable_linear`。mono pipeline（`mono_aec_nr_res/audio_pipeline.c`
  第 655 行附近的 `f->usable_linear = ctx.usable_linear;`）本來就是對的，沒受這個回歸影響，這次
  也完全沒改動。
- `tests/test_4aec_nr_res.c` 的 `test_fuse_frame_contract()` 加兩條斷言：`usable_linear` 每個 hop
  都是 0/1、且在四個等權重、同步收斂的 lane 下於 60-hop 視窗內確實變成過 1（這條在修法前的程式碼
  上必定 FAIL：驗證過，暫時把 `frame->usable_linear = usable_linear;` 那行拿掉重跑會讓它 FAIL，
  補回來後 PASS）。另加一個新測試
  `test_fuse_usable_linear_and_over_contributors()`：一個 lane 權重設 0（不貢獻）且刻意用跟
  far-end 不相關的雜訊餵它，讓它自己的 `usable_linear` 全程為假，其餘三個 lane 正常收斂——同一個
  front frame 分別用「該 lane 權重 0」與「該 lane 權重 0.25（變成貢獻者）」兩組權重各 fuse 一次：
  前者必須讀到 `usable_linear=true`（否則是誤把 AND 做在全部四個 lane 而非只做在貢獻的 lane
  上），後者必須讀到 `false`（否則是「永遠強制 1」或「漏掉 AND-fold」這類修法，第一個既有測試
  等權重的設計看不出這種差異，這個新測試可以分辨）。兩個 mutation 都已手動注入驗證過會讓對應
  斷言 FAIL。
- `tests/pyc_parity/pyc_parity.py` 移除 `a.nr4.*.verdict_flag`／`a.nr4.*.gnr_e2e`／`a.nr4.*.legacy_amin`
  三族的 `EXPECTED_RED(S-25)` 標記（16k/256、16k/512、48k/1024 三個 grid）；`pyc_parity_pipelines.sh`
  重跑後這三族在三個 grid 上全部變成 PASS（`mismatch_hops=0`；`legacy_amin` 的 R² fold on/off
  差異在 C 內部也量到了）。**附帶發現、與本次修法本身無關、但已查清並一併收尾**：這次重跑時
  `b.res4.*.aggressive`（另一個已知缺口 S-26，RES 的 double-talk floor）也是
  `UNEXPECTED_GREEN(S-26)` 而非預期的 `EXPECTED_RED(S-26)`。最初以為是跟這次 S-25 修法無關的
  既有未解之謎（已查過 `se_res_stage_gain()` 根本沒把 `usable_linear` 傳進 `AecResInput`，把
  S-25 的修法暫時還原重跑 `--only b.res4` 數值完全不變，確認跟 S-25 無關）。後來核對同一時間
  平行進行的另一批修法才查明：S-26 本身（`pipeline.py` 的 `PostBeamResidualSuppressor` 補上
  DT 地板傳播、`FourChannelAecConfig` 加 `aec_preset`）剛好在同一個工作樹上落地，這個
  `UNEXPECTED_GREEN(S-26)` 正是 S-26 真的修好了的結果，不是謎——已重跑 `--only b.res4` 確認三個
  grid 的 `aggressive` 全部 PASS（`max_abs` 7.69e-6～9.78e-6，容差 1e-4），並把
  `EXPECTED_RED(S-26)` 標記移除。
- 用真實重建的 `audio_pipeline_4ch_raw` 跑 `evaluate_external_recordings.py`（`aec_take_turn`、
  `aec_together`，16 kHz/256 與 16 kHz/512）：near-end-only attenuation 基本不變（例如
  `aec_take_turn` @256 12.4164→12.4165 dB），double-talk／far-only attenuation 上升——
  `aec_together` double_talk @256 15.51→16.49 dB、@512 13.83→15.24 dB；far_only @256
  18.69→18.85 dB、@512 19.63→20.07 dB；`aec_take_turn` far_only @256 20.04→20.13 dB、@512
  21.06→21.15 dB。silence 與 delay 相關欄位（`final_delay_error_samples` 等）不變。

### 證據 (Evidence) — AECMOS 近端品質量測

- 用官方未修改的 `FastAECMOS`（`AEC/python/bench_aecmos.py`）對修前（單點還原
  `frame->usable_linear = usable_linear;` 這一行寫回，其餘 7 個插入點不動）／修後兩個重建的
  `audio_pipeline_4ch_raw` 跑分：`datasets/aec_take_turn`、`datasets/aec_together`（各 16 kHz
  @256／@512）＋ 4 組 `~/Desktop/exp` 真實錄音（`ref.wav` 已核對是原始遠端回授訊號，非 AEC
  內部估計值），10 秒窗分話務型態（dt／nst／st），排除收斂期第 0 窗。雙講（dt）19 個窗：
  `echo_mos` 平均 +0.071（回音消除變好，符合預期）、`deg_mos` 平均 **−0.073**（74% 的窗變差，
  6 個窗掉超過 0.1，最差窗 `0827_doub_tog` w4 掉 0.333）；近端單講（nst）2 個窗持平；遠端單講
  （st）8 個窗因該情境本來就無近端語音，`deg_mos` 兩邊都頂在 ≈4.998～4.999，量不出東西。
- **根因核對（opus 二次確認）**：`frame->usable_linear` 在整條 4ch pipeline 裡只有一處讀取——
  `pipelines/common/se_anr_stage.c:51` 的 `if (use_r2 && frame->r2 && scratch &&
  frame->usable_linear)`，是 RES→NR 的 R² 注入閘。`four_aec_fuse()` 進來時先把 `frame`
  `memset` 成 0，所以 `af914d4` 刪掉這行寫回之後，`usable_linear` 永遠是 0——不是只關掉
  cold-start 保護，是把整條 R² 注入路徑整個斷開。因此這次 AECMOS 的「修前」其實量的是「R² 注入
  全關」，「修後」量的是「R² 注入（有 cold-start gate）重新接上」，兩者的差異正是**R1 設計本來就
  接受的雙講代價**（`49f73b2` 之前的長期上線行為是無條件注入，由 `legacy_amin` 這個 static flag
  控制；`49f73b2`→`af914d4` 中間那段「有 gate」的版本只上線過約 1.5 小時，是意外被刪除的那個片段，
  不是新增邏輯），不是這次修法引入的新過度抑制 bug。
- **裁定：維持現狀出貨，不需要為此延後或另修**。雙講 deg_mos 的下降是重新接上既有、曾經上線過的
  RES→NR 耦合之後的既有代價，衡量基準應該是「R² 注入開」的 R1 行為，不是「注入全關」這個意外狀態。
  若未來認為這個雙講代價太高，屬於既有的 DT over-suppression／RES 操作點調校 arc 的範疇（調多重的
  R² 權重），不應該用「讓這條路徑維持斷開」的方式處理。

### 已知限制 (Known limitation)

- 這批 AECMOS 證據只有 16 kHz（這個 checkout 沒有 48 kHz 的 AECMOS 模型）、只建置
  SIMD=1/NE10/arm64（未測 SIMD=0 或目標平台工具鏈）；近端單講只有 2 個窗，證據薄；
  `~/Desktop/exp` 的話務型態是依檔名指派、不是量測到的近端活動量；是 36 個窗的配對證據，
  不是有統計檢定力的語料級結論。

## [Unreleased] · 2026-10-19 · 跟上 AEC／NR 的 C 符號改名、移除 `BACKEND=kiss`、`four_aec_nr_res_stage_configs()` 在 `enable_nr = 0` 時不組 NR 設定

所有 pipeline 的輸出與改動前逐位元相同：以 `harness/pipelines_suite.sh` 對改動前的樹與改動後的樹各跑一次
（SIMD=1 與 SIMD=0，含 mono／4ch／ULCNet／DFN 的 CLI 輸出檔、`--print-mem-size` 表、`make test` 變體、
AINR／AIAEC 的 C 測試與 C/Python parity），manifest 逐行相同。唯一的文字差異是 README、註解、
`make` 的 banner 與 `make help` 類說明；`build_flags_hash`、`layout_version`、`backend_id` 的數值都沒變。

### 變更 (Changed)

- **公開 API 的行為改變：`four_aec_nr_res_stage_configs()`。** `cfg->enable_nr == 0` 時不再呼叫
  `se_anr_compose_config()`，`*nr_cfg` 回傳全零的 `AnrConfig`（`se_anr_stage_mem_size()` 對它回 0，不可拿去
  初始化 NR）；`enable_nr != 0` 時與先前逐位元相同。函式簽章、回傳值規則（NULL 引數或核心會拒絕的設定
  回 -1）、`res_cfg` 都不變。五個呼叫端：
  - `4ch_aec_bf_nr_res/audio_pipeline_4ch.c`：本來就只在 `enable_nr` 時才計算 `nr_bytes`，不需改。
  - `4ch_aec_bf_nr_res/4aec_nr_res_static.c`：`plan_stages()` 只在 `enable_nr` 時才要求 `nr_bytes != 0`
    （先前無條件要求，`nr_bytes == 0` 會讓整個規劃失敗）。
  - `4ch_aec_bf_nr_res/tests/conv_chain_4ch.h`：NR 區塊大小只在 `enable_nr` 時計算，`enable_nr = 0`
    的測試鏈不再多配一塊用不到的 NR 記憶體。
  - `4ch_aec_bf_dfn_res/4aec_dfn_res.c`：要求 `enable_nr == 0`，丟棄本地 `nr_cfg`，加註解說明。
  - `4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c`：新增檢查——`enable_nr = 0` 時 `nr_cfg` 為全零、
    `res_cfg` 與 NR 開啟時推導的一致。已做變異測試：把閘門拿掉，這項檢查與整個
    `test_stage_sizing_and_switches` 會失敗。
  - 這個修法**不會**讓任何 Makefile 的連結依賴變小：`4aec_nr_res.o` 仍參照 `se_anr_compose_config`，
    `audio_pipeline_4ch.o` 仍直接參照 `se_anr_stage_*`（`4ch_alignulcnet` 照舊連結 `se_anr_stage.o`）。
- **跟上 AEC 的 C 符號改名**（AEC 的改名表見 `lib/aec` 的 CHANGELOG）：`DelayAec3` → `DelayAec`、
  `delay_aec3.h` → `delay_aec.h`、`delay_aec3_*` → `delay_aec_*`、`AEC3B_*` → `AECB_*`、
  `aec3_balanced_config.h` → `aec_balanced_config.h`、`pbfdkf_update_weights_aec3` → `pbfdkf_update_weights`、
  `AEC3_FILTER_NOISE_GATE_POWER_FLOAT` → `AEC_FILTER_NOISE_GATE_POWER_FLOAT`。
  消費端：`4aec_nr_res.{c,h}`、`4aec_nr_res_internal.h`、`tests/test_4aec_nr_res.c`、
  `4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c`、`mono_alignulcnet/tests/test_audio_pipeline_ulcnet.c`，
  以及未追蹤的診斷程式 `experimental/linear_aec_probe.c`（`DelayAec`、`delay_aec_estimated_delay`）、
  `experimental/algorithm_completeness/check_aec_contract.c`（`pbfdkf_update_weights`、
  `AEC_FILTER_NOISE_GATE_POWER_FLOAT`）、`tools/joint_postfilter_cli.c`（`aec_balanced_config.h`、`AECB_SQRT2_SIN_LUT`）；
  Python 側只跟著改讀 C 標頭的地方（`tests/test_cng_aec3.py`、`cng_aec3.py` 的說明文字），
  另有只改註解裡舊名的 `4ch_aec_bf_nr_res/pipeline.py`、`tests/_scripted_delay.py`、`tests/test_pipeline.py`。
- **跟上 NR 的 C 符號改名**：`McraNoiseEstimator` → `NoiseEstimator`、`mcra_noise_estimator.{h,c}` →
  `noise_estimator.{h,c}`、`mcra_{get_noise_psd,update_ex,accumulate_init_power,init_noise}` →
  `noise_estimator_*`。使用者是診斷探測程式：`tools/res_nr_{coupling,dd,near,external_noise,tracker}_probe.c`、
  `experimental/nr_soft_scene_probe.c`、`experimental/algorithm_completeness/nr_probe.c`；
  指到 `mcra_noise_estimator.c` 路徑的建置腳本與 Python 一併改（`tools/build_*.sh`、
  `tools/evaluate_res_nr_coupling.py`、`tools/finalize_res_nr_repair.py`、`experimental/algorithm_completeness/build.sh`）。
- `4aec_nr_res.c` 的 `build_flags_hash` 輸入字串原本含 `aec3`（第 4 條 lane 的標籤）。改用巨集
  `LANE_TAG(i)` 組出同一個字串，hash 值不變（已用 `--print-mem-size` 的 `build_flags_hash` 比對）。
- 精簡經手檔案裡帶日期或 campaign 代號的註解（`4aec_nr_res.{c,h}`、`tests/test_4aec_nr_res.c`、`Makefile`），
  保留技術理由。
- `README.md` 的 Memory Budget 表改為只列 NE10 的現行實測值（8 kHz 368,608 B；16 kHz 536,736 B（256/128）／
  691,360 B（512/256）；48 kHz 1,662,192 B），控制區塊 160 B 計入總量；`4ch_alignulcnet/README.md` 的 pool
  大小改為 3,172,656 B（16 kHz）／6,659,280 B（48 kHz）。`mono_aec_nr_res/example_board_adapter.c` 的 arena
  大小說明同步（2 MiB 對 48 kHz 的 1,662,192 B 約有 425 KiB 餘量）。

### 移除 (Removed)

- `BACKEND=kiss`：`pipelines/Makefile`、`4ch_aec_bf_nr_res/Makefile`、`third_party/Makefile`、`AINR/Makefile`
  只接受 `ne10`，其他值在解析階段以 `BACKEND must be ne10` 報錯。`BACKEND` 仍是變數，因為它是 `obj/`、
  `bin/`、`dist/` 目錄名的一部分，也轉送給 audio_common／lib/aec／lib/nr。連帶移除：
  `-I$(AC_DIR)/lib/kiss_fft`、`LINK` 依 backend 切換（改為恆用 `$(CXX)`）、`_cfg_guard` 的 ne10-only 條件、
  banner 與用法說明裡的 `make BACKEND=kiss` 範例、README 與測試檔頭的 kiss 建置指令。
  `AIAEC/Makefile` 本來就沒有 BACKEND 分支，不需改。
- **公開巨集移除（API 變更）：四個 kiss 後端 ID，值都是 1。** 各在一個公開標頭裡：
  `FOUR_AEC_NR_RES_BACKEND_KISS`（`4ch_aec_bf_nr_res/4aec_nr_res.h`）、
  `AUDIO_PIPELINE_BACKEND_KISS`（`mono_aec_nr_res/audio_pipeline.h`）、
  `AUDIO_PIPELINE_ULCNET_BACKEND_KISS`（`mono_alignulcnet/audio_pipeline_ulcnet.h`）、
  `DFN_RES_STAGE_BACKEND_KISS`（`dfn_res_stage.h`）。程式若拿它們跟 descriptor 的 `backend_id` 比對，
  升級後會編譯失敗。對應的 `"kiss"` 字串比對與 `backend_id_name()`／`static_main.c` 的 `case` 一併移除
  （`4aec_nr_res.c`、`4aec_nr_res_static.c`、`audio_pipeline_4ch_raw.c`、`audio_pipeline_4ch_static.c`、
  `mono_aec_nr_res/audio_pipeline.c`、`mono_aec_nr_res/static_main.c`、`mono_alignulcnet/audio_pipeline_ulcnet.c`、
  `dfn_res_stage.c`）。四個 `*_BACKEND_NE10`（值 2）不變，descriptor 裡的 `backend_id` 仍是 2；值 1 保留不再回傳。
- `audit-no-stdio-stack`（pipelines）與 `audit-no-stdio`（4ch）原本對 kiss 只查較小的 stdio 符號集合，
  現在只剩完整集合（`stderr`／`__stderrp`／`fputc`／`fwrite` 也算），與先前 ne10 的檢查相同。
- `scripts/audit_fp_contract.sh` 的預設 backend 清單由 `ne10 kiss` 改為 `ne10`。
- `scripts/test_build_isolation.sh`：所有 `BACKEND=kiss` 改為 `ne10`（dist 路徑、MANIFEST 的 producer
  簽章、`[ne10]` 日誌比對、`aec_post.o`／`noise_estimator.o` 檔名一併跟上）。其中兩組測試**語意改寫**，
  不只是改字：
  - SP1 原本是「kiss → ne10 → kiss」的 A→B→A（以後端當作 A／B 兩個建置），後端只剩一種後，改為
    「SIMD=1 → SIMD=0 → SIMD=1」的 A→B→A：斷言變成「SIMD=1 與 SIMD=0 落在不同 bin 目錄」「SIMD=1 的 bin 目錄在
    A→B→A 後不變」「第三次不重連 `aec_nr_pipeline`」「SIMD=1 連得到 ne10 FFT 符號」（原本的 ne10 斷言移到
    SIMD=1 的建置上）「SIMD=0 只連標量 ne10 FFT、不含 NEON FFT 符號」（新增），取代原本的 kiss 對應斷言
    （kiss 的 bin 目錄穩定、kiss 第三次不重連、kiss 不連 ne10 FFT 符號）。
  - SP-S12 原本驗證「`BACKEND=kiss` 時，CXX 與 CC 目標不一致的守衛不生效、建置仍成功」，改為驗證
    `TOOLCHAIN_CHECK=0` 同樣略過該守衛、建置成功（同一個目標不一致的 CXX shim）。
  - `resolve_producers` 新增可選的 SIMD 引數。
  這兩組測試的斷言名稱與條數和改動前不同，因此這個腳本的 PASS 清單不能與改動前逐行比對。

### AINR／AIAEC

- `AINR/Makefile`：加上與其他 Makefile 相同的 `BACKEND must be ne10` 檢查（它把 `BACKEND` 轉送給
  audio_common 以取得封存檔路徑，audio_common 現在拒絕其他值）。`CFG_SIG` 不變。
- `AIAEC/Align_ULCNet/ulcnet_process.h` 與 `AIAEC/tests/test_ulcnet_process_c.py`：移除說明文字中的
  kiss／backend 切換敘述。AINR 與 AIAEC 的 C 原始碼沒有其他改動。

### 已知、未動

- `Audio_ALG/docs/`（Markdown 與 `docs/html`）仍有引用舊 C 名稱與 `BACKEND=kiss` 的敘述
  （`docs/c_user_manual_zh_TW.md`、`docs/audio_libraries_c_api_zh_TW.md`、`docs/integration_*`、
  `docs/pipeline_board_integration.md`、`docs/release_checklist.md`、`docs/html/*` 等），不在這一輪範圍；
  只有 `docs/integration_4ch_core_zh_TW.md` 裡三處 `four_aec_nr_res_stage_configs()` 的敘述同步了
  `enable_nr = 0` 的新行為。
- `pipelines/tools/` 與 `pipelines/experimental/` 曾是未追蹤的研究用程式，其中多個檔案（`nr_overlay.h`／
  `anr_echo_internal.h`／`run_frame_gain_stage` 參數個數／`NoiseEstimator::scene_change_tonal_veto`／
  `AudioPipelinePostView` 等）在改名之前就已經編不過（逐一驗證過，錯誤相同）。結構性整理這一輪
  （2026-10-20）確認這些檔案在真正的建置系統（所有 Makefile）裡零引用，使用者核准整批刪除：
  `pipelines/tools/` 下 12 個孤兒 `.c`、4 個獨立 `build_*.sh`、135 個未追蹤的 `.py` 診斷腳本（保留 11 個
  確認被 Makefile 使用的 `.py` 工具不動），以及整個 `pipelines/experimental/`（17 個檔案，同樣未追蹤、
  同樣零建置引用，且其中 5 個檔案 import 了已刪除的 `pipelines/tools/` 模組，刪除前就已經壞掉）。
