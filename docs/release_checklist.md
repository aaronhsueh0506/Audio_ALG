# Audio Stack Release Checklist

This document is the release gate for the repositories below:

- `audiolib`
- `AEC`
- `NR`
- `Audio_ALG`
- `Audio_ALG/lib/aec`
- `Audio_ALG/lib/nr`

It covers production traditional-DSP libraries and the AINR/AIAEC development
frameworks. AINR/AIAEC release scope ends at dataset generation, training
architecture, and reference inference code. Full dataset generation, long
training runs, final checkpoints, and trained-model quality scores are done on
another machine and are not blockers for this source/framework release.

The release owner must record every command, commit, result, failure, and accepted
exception in a dated release report. Do not convert an unexecuted item into a
PASS based only on source inspection.

**Contents** <!-- split-toc -->
- [Audio Stack Release Checklist](#audio-stack-release-checklist) <!-- split-toc -->
  - [1. Product contract](#1-product-contract) <!-- split-toc -->
    - [1.1 Traps an integrator will hit](#11-traps-an-integrator-will-hit) <!-- split-toc -->
  - [2. Repository freeze and hygiene](#2-repository-freeze-and-hygiene) <!-- split-toc -->
  - [3. `audiolib` gate](#3-audiolib-gate) <!-- split-toc -->
  - [4. AEC automated gate](#4-aec-automated-gate) <!-- split-toc -->
  - [5. AEC quality blockers](#5-aec-quality-blockers) <!-- split-toc -->
    - [5.1 Current-code 800-case benchmark](#51-current-code-800-case-benchmark) <!-- split-toc -->
    - [5.2 Native 48 kHz validation](#52-native-48-khz-validation) <!-- split-toc -->
  - [6. NR automated gate](#6-nr-automated-gate) <!-- split-toc -->
  - [7. NR quality decision](#7-nr-quality-decision) <!-- split-toc -->
  - [8. Mono pipeline gate](release_checklist_01_pipeline_gates.md#8-mono-pipeline-gate) <!-- split-toc -->
    - [8.1 Mono DeepFilterNet2 variant (`mono_aec_dfn_res`)](release_checklist_01_pipeline_gates.md#81-mono-deepfilternet2-variant-mono_aec_dfn_res) <!-- split-toc -->
  - [9. Four-channel pipeline gate](release_checklist_01_pipeline_gates.md#9-four-channel-pipeline-gate) <!-- split-toc -->
    - [9.0 Four-channel DeepFilterNet2 variant (`4ch_aec_bf_dfn_res`)](release_checklist_01_pipeline_gates.md#90-four-channel-deepfilternet2-variant-4ch_aec_bf_dfn_res) <!-- split-toc -->
    - [9.1 Real four-microphone validation](release_checklist_01_pipeline_gates.md#91-real-four-microphone-validation) <!-- split-toc -->
    - [9.2 Post-beam residual suppressor](release_checklist_01_pipeline_gates.md#92-post-beam-residual-suppressor) <!-- split-toc -->
  - [10. Target-board acceptance](release_checklist_01_pipeline_gates.md#10-target-board-acceptance) <!-- split-toc -->
  - [11. AINR framework gate](release_checklist_02_ainr_aiaec_gates.md#11-ainr-framework-gate) <!-- split-toc -->
    - [11.1 Dataset generator](release_checklist_02_ainr_aiaec_gates.md#111-dataset-generator) <!-- split-toc -->
    - [11.2 Training-framework smoke](release_checklist_02_ainr_aiaec_gates.md#112-training-framework-smoke) <!-- split-toc -->
    - [11.3 Reference inference](release_checklist_02_ainr_aiaec_gates.md#113-reference-inference) <!-- split-toc -->
    - [11.4 C pre/post archive](release_checklist_02_ainr_aiaec_gates.md#114-c-prepost-archive) <!-- split-toc -->
  - [12. AIAEC framework gate](release_checklist_02_ainr_aiaec_gates.md#12-aiaec-framework-gate) <!-- split-toc -->
    - [12.1 Dataset generator](release_checklist_02_ainr_aiaec_gates.md#121-dataset-generator) <!-- split-toc -->
    - [12.2 Training-architecture smoke](release_checklist_02_ainr_aiaec_gates.md#122-training-architecture-smoke) <!-- split-toc -->
    - [12.3 Reference inference](release_checklist_02_ainr_aiaec_gates.md#123-reference-inference) <!-- split-toc -->
    - [12.4 C pre/post archive](release_checklist_02_ainr_aiaec_gates.md#124-c-prepost-archive) <!-- split-toc -->
  - [13. Documentation gate](release_checklist_03_docs_publish_appendices.md#13-documentation-gate) <!-- split-toc -->
  - [14. Publish and tag](release_checklist_03_docs_publish_appendices.md#14-publish-and-tag) <!-- split-toc -->
  - [15. Final GO/NO-GO](release_checklist_03_docs_publish_appendices.md#15-final-gono-go) <!-- split-toc -->
  - [Appendix A. Reference test targets](release_checklist_03_docs_publish_appendices.md#appendix-a-reference-test-targets) <!-- split-toc -->
  - [Appendix B. Benchmark protocol](release_checklist_03_docs_publish_appendices.md#appendix-b-benchmark-protocol) <!-- split-toc -->
  - [Appendix C. Cross-repo mechanics](release_checklist_03_docs_publish_appendices.md#appendix-c-cross-repo-mechanics) <!-- split-toc -->
## 1. Product contract

The traditional mono and four-channel product grids are:

| Sample rate | FFT/frame | Hop | Status |
|---:|---:|---:|---|
| 16 kHz | 256 | 128 | Default |
| 16 kHz | 512 | 256 | Supported alternate |
| 48 kHz | 1024 | 512 | Supported |

The 8 kHz standalone grid is supported by AEC and NR but is not a main product
grid for this release. Either keep it classified that way or give it an
independent quality verdict before promoting it to a product grid.

The traditional signal contract is:

- `frame == FFT`;
- `hop == frame / 2`;
- no transform zero-padding;
- periodic sqrt-Hann and 50% overlap at downstream STFT/WOLA boundaries;
- SIMD enabled by default and disabled everywhere through `SIMD=0`;
- NE10 is the only FFT backend (any other `BACKEND` value is rejected at parse time);
- heap/static and SIMD/scalar paths must satisfy their documented parity
  contract within one backend;
- static APIs use caller-owned pools and allocate no heap after init;
- Python/C AEC parity is tolerance-based, not bit-exact (frozen tolerances: `AEC/CHANGELOG.md`,
  `pipelines/tests/pyc_parity/pyc_parity.py`).

AINR and AIAEC retain model-specific grids. A model must validate its own
sample-rate/frame/hop/checkpoint contract and must not silently inherit the
traditional pipeline default.

### 1.1 Traps an integrator will hit

Three things are true, deliberate, and easy to get wrong. State each of them in
the handover documentation, and check them off here.

- [ ] **AIAEC is on a different grid from the pipeline, on purpose.** The
      traditional pipeline defaults to 16 kHz **256/128**; AIAEC's frozen
      dataset contract is 16 kHz **512/256** (`AIAEC/dataset_gen/config.ini`,
      `FROZEN_FRAME_HOP_BY_SR` in `linear_aec.py`). These are two independent
      contracts, not a drift to be reconciled. A dataset must never be
      regenerated "to match the pipeline default" without a contract-version
      bump and retraining, and the pipeline default must never be changed to
      match AIAEC.

- [ ] **Static pools are fully supported but are NOT what `*_create()` gives
      you.** `aec_create()` / `mcab_create()` / the other `*_create()` constructors use
      the heap. Zero-heap operation requires the caller to explicitly use
      `*_get_mem_requirements()` (or `aec_get_mem_size()`) followed by
      `*_init_ex()` / `aec_init()`. A board integration that calls `*_create()`
      and then measures heap will conclude, wrongly, that the static path does
      not exist. The board-facing documents must show the query+init pair, not
      the convenience constructor.

- [ ] **`torchaudio < 2.9` is pinned deliberately.** The AI framework code
      calls `torchaudio.info`/decode APIs that torchaudio 2.9 removes, and
      emits a deprecation warning today. All three shipping requirements files
      carry `torchaudio>=0.13,<2.9`:
      `AIAEC/requirements.txt`, `AINR/dataset_gen/requirements.txt`,
      `AINR/RNNoise-ERB/requirements.txt`. This is a recorded constraint, not a
      blocker. Unpinning without first porting off the deprecated APIs breaks
      dataset generation. Verify all three still carry the bound:
      `grep -rn torchaudio --include=requirements*.txt Audio_ALG/`

## 2. Repository freeze and hygiene

- [ ] All six working trees are clean.
- [ ] `Audio_ALG/lib/aec` equals the selected AEC release commit.
- [ ] `Audio_ALG/lib/nr` equals the selected NR release commit.
- [ ] Standalone and submodule tracked source trees match.
- [ ] `git diff --check` passes in every repository.
- [ ] No untracked checkpoints, datasets, ZIP files, temporary WAV files, or
      build products are present.
- [ ] No absolute local paths or credentials are present.
- [ ] No `review_*.md`, review-round wording, or other development-process
      messages remain in release source, CLI output, or current user
      documentation.
- [ ] Public headers compile from a clean external C99 consumer.
- [ ] Every public API/ABI/layout break has a version bump, migration note,
      and negative compatibility test.
- [ ] Third-party NE10 and other redistributed code has
      correct license/source attribution.

## 3. `audiolib` gate

Every repository's Makefile also has the aggregate entry points `make test` (core tier) and `make test-full`
(extended tier); the lists in sections [3](#3-audiolib-gate) to [9](release_checklist_01_pipeline_gates.md#9-four-channel-pipeline-gate) name the individual targets that a release report must
record one by one.

Run from `audiolib/common/`:

```bash
make clean
make BACKEND=ne10 SIMD=0 WERROR=1
make BACKEND=ne10 SIMD=0 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test-wav-ubsan test_zero_heap

make BACKEND=ne10 SIMD=1 WERROR=1
make BACKEND=ne10 SIMD=1 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test-wav-ubsan test_zero_heap

make test_ne10_force_c
```

Run from `audiolib/` (the DOA/BF/VAD spatial chain), with `SIMD=0` and `SIMD=1`:

```bash
make SIMD=<0-or-1> WERROR=1 test         # includes audit-no-bf-test-symbols
make SIMD=<0-or-1> WERROR=1 test-ubsan   # the same test under -fsanitize=undefined
```

Pass conditions:

- [ ] No compiler warnings.
- [ ] `test_spatial_third_party` passes plain and under UBSan (`test-ubsan`).
- [ ] SIMD/scalar kernel parity passes.
- [ ] FFT heap/static parity passes.
- [ ] Resampler, WAV, invalid-input, and pool tests pass.
- [ ] Invalid input does not modify a caller-owned pool.
- [ ] Static processing allocates no heap after init.
- [ ] The unified `-ffp-contract=off -fno-math-errno` policy and FP/FSQRT audits pass.

## 4. AEC automated gate

Run each target with both SIMD settings:

- `BACKEND=ne10 SIMD=0`
- `BACKEND=ne10 SIMD=1`

Use `WERROR=1` throughout. From `AEC/c_impl/`:

```bash
make clean
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 selftest
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-rate-structural
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-counter-saturation
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-delay-reset
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-config-validation
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-process-context
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-shared-far-spec
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-shared-fft-handle
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-static-aec
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-zero-heap
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-detectors-parity-all
```

Run the supported Python test suites from `AEC/` using the release Python
environment.

Pass conditions:

- [ ] All product grids resolve to the required FFT/frame/hop.
- [ ] Runtime effective-value tests cover all wall-clock-retimed parameters.
- [ ] **An inventory of every hop-authored constant exists, and each one has a
      verdict: retimed (with an effective-value test at all four grids) or
      deliberately not retimed (with a recorded reason).** "The retimed ones
      are tested" is not the same claim and does not satisfy this. Build the
      list mechanically: grep `c_impl/src/*.c` and `python/modules/*.py` for bare
      float EMA retentions and integer frame counts, subtract everything
      routed through `aec_growth_rehop` / `aec_ms_to_hops` /
      `growth_rehop` / `ms_to_hops`, and adjudicate the remainder one by one.
- [ ] Effective-value tests are mutation-tested: reverting any single constant
      to its authored literal, and retiming any constant off the wrong
      reference grid, must both fail the suite. Note that this subsystem has
      **more than one authoring grid** (10 ms and 16 ms), so a constant that
      correctly does not move on a 16 ms grid is not evidence of a missing
      retime.
- [ ] Same-backend heap/static output is byte-equal.
- [ ] Context-only and full processing may be interleaved without state drift.
- [ ] Four-channel far-reference FFT sharing performs at most one real far FFT
      per hop.
- [ ] Shared FFT state remains synchronized through resets, delay changes,
      saturation, and error paths.
- [ ] NULL, invalid grid, undersized pool, and misaligned pool inputs fail
      before modifying instance or pool memory.
- [ ] No output or internal state becomes NaN/Inf.
- [ ] Static processing allocates no heap after init.
- [ ] The removed custom output limiter is not present in the linear seam.
- [ ] Float-to-PCM boundary saturation remains intact.
- [ ] Linear output is the selected/crossfaded/WOLA-formed linear error and is
      not output-limiter processed.

## 5. AEC quality blockers

### 5.1 Current-code 800-case benchmark

Run the full AEC Challenge benchmark against a frozen pre-release baseline on
the deployed C path:

- [ ] 16 kHz / 256 / 128.
- [ ] 16 kHz / 512 / 256.
- [ ] Far-end static.
- [ ] Far-end movement.
- [ ] Double-talk static.
- [ ] Double-talk movement.
- [ ] Near-end single-talk.

The report must include:

- AECMOS echo/deg;
- ERLE;
- SDR or SI-SDR proxy;
- near-end preservation;
- clipping/peak and non-finite counts;
- per-case deltas;
- bucket mean, median, p10, and worst-N;
- audio/energy investigation for material outliers.

Pass conditions:

- [ ] Baseline, candidate, corpus, grid, and gates are frozen before scoring.
- [ ] Historical production bars remain satisfied: FS echo > 3.5, DT echo >
      4.0, DT deg > 2.0, and NE deg >= 4.0.
- [ ] No new severe regression cohort is hidden by a bucket average.
- [ ] Every material score regression is checked against waveform energy and
      human listening before being accepted as a metric artifact.
- [ ] Both 16 kHz grids receive an explicit verdict.

### 5.2 Native 48 kHz validation

Upsampled 16 kHz material is not valid evidence. Use native 48 kHz recordings
with real energy above 8 kHz and cover:

- [ ] far-end only;
- [ ] double-talk;
- [ ] near-end only;
- [ ] echo-path movement;
- [ ] delay changes;
- [ ] mid-stream reset;
- [ ] near-full-scale capture;
- [ ] representative device paths.

Report full-band ERLE, AECMOS, SI-SDR/STOI, 8--24 kHz artifact/energy analysis,
clipping/non-finite counts, NE10 behavior, and human listening. Do not
remove the AEC RC marker without this verdict.

## 6. NR automated gate

From `NR/`:

```bash
python3 -m pytest -q tests
```

From `NR/c_impl/`, run both SIMD 0/1 combinations:

```bash
make clean
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 \
  test test-config test-config-parity parity
```

Pass conditions:

- [ ] Python/C effective config parity passes for 3 grids x 4 strengths.
- [ ] Defaults are 16k/256/128 and 48k/1024/512.
- [ ] 16k/512/256 remains a supported explicit alternate.
- [ ] `FrameProcessor(sample_rate=...)` selects the project default grid.
- [ ] A partial explicit grid fails closed.
- [ ] Optional gain output does not change the valid processing result.
- [ ] Heap/static construction parity passes.
- [ ] Static sizing and carve positions remain in lockstep.
- [ ] Static processing allocates no heap after init.
- [ ] No output or internal state becomes NaN/Inf.

## 7. NR quality decision

Use the current checked-in runner, not an old scratch script:

```bash
python tools/run_vctk_benchmark.py \
  --mode full --strength balanced \
  --output results/vctk_candidate.json

python tools/compare_vctk_benchmark.py \
  results/vctk_baseline.json \
  results/vctk_candidate.json
```

Run the full 824-case VCTK+DEMAND set for:

- [ ] 16k/256/128 full mode;
- [ ] 16k/512/256 full mode;
- [ ] stationary mode on both supported 16 kHz grids.

Make one explicit product decision:

1. If standalone NR is a production product, the checked-in comparator must
   pass, or a new baseline must be explicitly approved with listening evidence
   and a CHANGELOG decision.
2. If NR is released only for the AEC-integrated pipeline, narrow the README
   and version contract and do not claim that the standalone VCTK gate passed.

Also benchmark the AEC-integrated `accept_external_spp=False` path so the
retimed `L` and related state are exercised. Listen to stationary noise,
non-stationary noise, music, and competing speech cases.

