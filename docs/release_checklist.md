# Audio Stack Release Checklist

This document is the release gate for the repositories below:

- `audio_common`
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

## 0. Closed audit findings and live release gates

This section preserves the failure modes found during the release audit. Every
item below is now fixed, regression-tested, or explicitly classified as a
documented historical limitation. Remaining GO/NO-GO work is tracked in the
numbered gates below, not by stale source-audit notes.

As of 2026-08-06, pins `AEC a432523` / `NR 5708f49` / `audio_common 1b359d3` /
`Audio_ALG 992dc56` (all pushed):

- [x] **F2.4's mu-holdoff no-reset guard was silently reverted and shipped
      broken for ten weeks.** FIXED 2026-08-06 (Python + C + regression tests at
      all four grids, mutation-verified). Recorded here because the failure mode
      is the reusable lesson, not the line: commit `2f3699f` ("remove dead flag
      branches") collapsed
      `if not cfg.mu_holdoff_no_reset or holdoff == 0:` to an **unconditional**
      arm — keeping the arm the flag had *disabled*, because
      `mu_holdoff_no_reset` was in `AecConfig._LEGACY_HARDCODE_TRUE`. The
      explanatory comment above the line survived intact and went on describing
      the guard, so the source read correctly; only the executed branch was
      wrong, and the C port mirrored it. Nothing caught it for ten weeks.
      **When retiring a feature flag, evaluate which arm was live — a
      hardcoded-TRUE flag means the `if not flag` branch is the dead one.**

- [x] **The wall-clock timing audit was incomplete while shipped documentation
      said it was complete.** `AEC/README.md` presented a closed three-item blocker list
      (800-case, native 48 kHz, tuning-not-timing) and `AEC/CHANGELOG.md:965`
      states "Every constant now covers the same wall-clock span at every
      grid". Both are false. Known un-retimed constants on the default-ON path,
      all of which mistime on this release's own new 16 kHz default grid
      (hop 128 = 8 ms) — **this list is not yet closed; a full inventory is the
      first task**, see §4:

      | constant | site | validated anchor | 16k/256 | 16k/512 | 48k/1024 |
      |---|---|---:|---:|---:|---:|
      | `simple_mu_holdoff = 20` | `aec.c:448`, `orchestrator.py:1332` | 200 ms | 160 ms | 320 ms | 213 ms |
      | coherence `alpha = 0.05` TC | `aec3_post.c:56`, `config.py:63` | 195 ms | 156 ms | 312 ms | 208 ms |
      | misadjustment `n_hops_target = 2` | `aec.c:1518`, `orchestrator.py:130` | ~16 ms | 16 ms | 32 ms | 21 ms |
      | ERLE window `decay = 0.999` TC | `aec.c:2213`, `orchestrator.py:2137` | ~10 s | 8 s | 16 s | 10.7 s |
      | recent ERLE peak `last 15 hops` | — | 150 ms | 120 ms | 240 ms | 160 ms |
      | stationary-DT baseline `0.999/0.95/0.995` | `aec.c:2286`, `orchestrator.py:2229` | tbd | | | |
      | instant-ERLE `0.7` | `aec.c:2300`, `orchestrator.py:2249` | tbd | | | |
      | `simple_mu` alphas `0.3/0.99/0.95` | `aec.c:448-450` | tbd | | | |

      **The anchor is the last commit that empirically VALIDATED the constant,
      not the commit that introduced it, and not the in-code comment.** Worked
      example: `simple_mu_holdoff = 20` was introduced at `d774771`
      (2026-03-20) when the default was frame 512 / hop 256 = 16 ms, so its
      comment says "~320ms" and was correct *then*. It was validated at
      `7b2cf04` (2026-05-12, the F2.4 800-case ablation) when `AecConfig` had
      flipped to `frame_size = -1  # Auto: sample_rate * 20ms` = hop 160 =
      10 ms. The validated span is therefore **200 ms**, and the comment has
      been stale since 2026-05-12. Anchoring on the introduction commit instead
      gives a 1.6x error — and did, in an earlier round of this very campaign.

      Each constant needs a verdict: retime (and extend `test_rate_structural`
      check (d2) + the Python effective-value suite), or classify as a genuine
      event count with a recorded rationale (as `CONV_FRAMES` and
      `ne_recent_sustain` were). Then restore an accurate disclosure. Do not
      close §5 or §15 while the README asserts a completeness the tree lacks.
      **Resolved:** the full inventory now lives in
      `AEC/docs/timing_constant_inventory.md`; every candidate has an authored
      grid and a retime/event-count/dead verdict. Effective-value and mutation
      tests cover the retimed values. The `simple_mu` batch is deliberately
      not retimed because the measured full-batch change was not Pareto-safe.

- [x] **AIAEC was the most-deviated consumer of the timing gap.** AIAEC pins
      16 kHz **512/256**, the
      grid furthest from every anchor in the table: holdoff 320 ms vs 200 ms,
      ERLE decay ~16 s vs ~10 s, coherence TC 312 ms vs 195 ms, misadjustment
      window 32 ms vs ~16 ms. Retiming will change `aec_behavior_hash`, so any
      dataset generated first must be rematerialized or its contract bumped,
      and any checkpoint trained on it retrained. Do not start bulk AIAEC
      dataset generation before the retime lands. AINR is unaffected.
      **Resolved:** the accepted timing changes are mirrored into `lib/aec`;
      existing `8e5d05708`-era data is admitted only through an explicit
      behavior-hash migration whose old/new linear-error and echo-estimate
      WAVs were byte-identical. Unknown hashes still fail closed.

- [x] **`FrameProcessor`'s docstring contradicted the code shipped with it.**
      `NR/core/frame_processor.py:19` documents per-parameter defaults
      (`frame_shift` "omitted -> frame_size / 2"), but the constructor raises
      `ValueError` unless all three dimensions are omitted or all three given.
      A caller following the docstring to reach the legacy 16 kHz 512/256 grid
      gets a hard construction failure. The CHANGELOG is correct; the API
      documentation an integrator actually reads is not. **Resolved:** the
      public docstring now requires all three dimensions together, or all
      omitted for the project default.

- [x] **The `hop <= 0` guards in `pbfdkf_init` / `pbfdkf_init_static` lacked
      regression coverage.** Deleting both left the old suite green,
      because the added CHECKs assert only the return value, which the nested
      `pbfdaf_*` guard supplies independently. The guards are load-bearing:
      without them `p->is_static = 0` and `memset(p, 0, sizeof(*p))` run before
      rejection, breaking the no-write-on-reject contract that the same commit
      published in `pbfdkf.h`. Add a sentinel + memcmp assertion on the
      instance, matching the one that already covers the `sample_rate` case.
      (The pool half of the contract is genuinely protected; only the instance
      half is exposed.) **Resolved:** sentinel/no-write assertions now cover
      both instance and pool entry points and are mutation-tested.

- [x] **`AEC/python/modules/aec3_scale.py` claimed "default 160 samples
      @ 16 kHz".** The C twin's identical sentence was corrected; the Python
      source-of-truth it is a port of was not, so the two now contradict each
      other. hop=160 is not merely non-default — `config.py`'s grid whitelist
      makes it unreachable. This is the exact misreading that produced the 1.6x
      mistiming class. **Resolved:** Python and C documentation now identify
      16 kHz/256/128 as product default and 16 kHz/512/256 as the alternate.

- [x] **`AEC/CHANGELOG.md` contains retired historical documentation paths.**
      Of the original 73
      prose `docs/*.md` / `docs/*.html` mentions, 71 point at verdict and
      design documents deleted during the release cleanup
      (`docs/f2_4_verdict.md`, `docs/v3_14_plan.md`, ...). Verify with:
      `grep -oE '\bdocs/[a-zA-Z0-9_/-]+\.(md|html)' CHANGELOG.md | while read p; do [ -f "$p" ] || echo "$p"; done`
      An integrator following any of them hits nothing. Either restore the
      archive, rewrite them as pinned-commit history links, or drop the
      reference and inline the one-line conclusion. Do not leave them as-is —
      the CHANGELOG is a release surface, and §13 requires that every
      documented file exist and work from a clean clone. **Accepted historical
      limitation:** the CHANGELOG now labels those paths as retired evidence
      and gives `git show <commit>:<path>` recovery instructions; current
      README/manual pages do not link them as live specifications.

- [x] **Five orchestrator state fields were written but never read.**
      `_shadow_error_psd`, `_shadow_R`, `_shadow_mu_holdoff` (init + two reset
      sites each), `_prev_filter_state` (init, reset, one write at
      `orchestrator.py:2614`), and `_hb_mic_pwr_ring` (a 32-float array
      allocated at `orchestrator.py:718` and never touched again). Confirmed
      by grep across `python/` — every reference is an assignment. Dead state
      in a 4500-line file with tightly coupled state is an active hazard: it
      reads as live invariant maintenance and invites "fixes" to code paths
      that do not exist. **Resolved:** the dead assignments were removed.

## 1. Product contract

The traditional mono and four-channel product grids are:

| Sample rate | FFT/frame | Hop | Status |
|---:|---:|---:|---|
| 16 kHz | 256 | 128 | Default |
| 16 kHz | 512 | 256 | Supported alternate |
| 48 kHz | 1024 | 512 | Supported |

The 8 kHz standalone grid is legacy support and is not a main product grid for
this release. Either retain that label or give it an independent quality
verdict before promoting it to a product grid.

The traditional signal contract is:

- `frame == FFT`;
- `hop == frame / 2`;
- no transform zero-padding;
- periodic sqrt-Hann and 50% overlap at downstream STFT/WOLA boundaries;
- SIMD enabled by default and disabled everywhere through `SIMD=0`;
- KISS is the reference backend and NE10 is the embedded backend;
- KISS and NE10 are not required to be bit-exact to each other;
- heap/static and SIMD/scalar paths must satisfy their documented parity
  contract within one backend;
- static APIs use caller-owned pools and allocate no heap after init;
- Python/C AEC parity is tolerance-based, not bit-exact.

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
      you.** `aec_create()` / `audio_pipeline_init()` / the 4ch equivalents use
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
- [ ] No `review_*.md`, Claude/Codex/simplify/round-N wording, or other
      development-process messages remain in release source, CLI output, or
      current user documentation.
- [ ] Public headers compile from a clean external C99 consumer.
- [ ] Every public API/ABI/layout break has a version bump, migration note,
      and negative compatibility test.
- [ ] Third-party KISS, NE10, SRP, GSC, and other redistributed code has
      correct license/source attribution.

## 3. `audio_common` gate

Run from `audio_common/`:

```bash
make clean
make BACKEND=kiss SIMD=0 WERROR=1
make BACKEND=kiss SIMD=0 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test-wav-ubsan test_zero_heap

make BACKEND=kiss SIMD=1 WERROR=1
make BACKEND=kiss SIMD=1 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test-wav-ubsan test_zero_heap

make BACKEND=ne10 SIMD=0 WERROR=1
make BACKEND=ne10 SIMD=0 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test_zero_heap

make BACKEND=ne10 SIMD=1 WERROR=1
make BACKEND=ne10 SIMD=1 selftest test_audio_utils test_pool \
  test_wav test_wav_nr_style test_zero_heap

make test_ne10_force_c
```

Pass conditions:

- [ ] No compiler warnings.
- [ ] SIMD/scalar kernel parity passes.
- [ ] FFT heap/static parity passes.
- [ ] Pre-gain, resampler, WAV, invalid-input, and pool tests pass.
- [ ] Invalid input does not modify a caller-owned pool.
- [ ] Static processing allocates no heap after init.
- [ ] The unified `-ffp-contract=off` policy and audit pass.

## 4. AEC automated gate

Run each target with all four build combinations:

- `BACKEND=kiss SIMD=0`
- `BACKEND=kiss SIMD=1`
- `BACKEND=ne10 SIMD=0`
- `BACKEND=ne10 SIMD=1`

Use `WERROR=1` throughout. From `AEC/c_impl/`:

```bash
make clean
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 selftest
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-rate-structural
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-counter-saturation
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-delay-reset
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-config-validation
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-process-context
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-shared-far-spec
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-shared-fft-handle
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-static-aec
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-zero-heap
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test-detectors-parity-all
```

Run the supported Python test suites from `AEC/` using the release Python
environment.

Pass conditions:

- [ ] All product grids resolve to the required FFT/frame/hop.
- [ ] Runtime effective-value tests cover all wall-clock-retimed parameters.
- [ ] **An inventory of every hop-authored constant exists, and each one has a
      verdict: retimed (with an effective-value test at all four grids) or
      deliberately not retimed (with a recorded reason).** "The retimed ones
      are tested" is not the same claim and does not satisfy this — the gap
      recorded in §0 passed every effective-value test precisely because the
      un-retimed constants were never in the inventory. Build the list
      mechanically: grep `c_impl/src/*.c` and `python/modules/*.py` for bare
      float EMA retentions and integer frame counts, subtract everything
      routed through `aec3_growth_rehop` / `aec3_ms_to_hops` /
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
clipping/non-finite counts, KISS/NE10 behavior, and human listening. Do not
remove the AEC RC marker without this verdict.

## 6. NR automated gate

From `NR/`:

```bash
python3 -m pytest -q tests
```

From `NR/c_impl/`, run all four KISS/NE10 and SIMD 0/1 combinations:

```bash
make clean
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 \
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

## 8. Mono pipeline gate

From `Audio_ALG/pipelines/`, run all KISS/NE10 and SIMD 0/1 combinations:

```bash
make clean-all
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test
```

From `Audio_ALG/`:

```bash
python3 -m pytest -q pipelines/tests
```

Pass conditions:

- [ ] AEC -> NR -> RES ordering is correct.
- [ ] Context-only AEC avoids unused output work.
- [ ] NR/RES share the selected STFT/WOLA grid.
- [ ] All three product grids pass.
- [ ] Same-backend heap/static output is byte-equal.
- [ ] Static processing allocates no heap after init.
- [ ] Real recordings complete without crash, NaN, or Inf.
- [ ] AEC-only and AEC+NR+RES examples use only public APIs and are documented.

## 9. Four-channel pipeline gate

From `Audio_ALG/pipelines/4ch_aec_bf_nr_res/`, run all KISS/NE10 and SIMD 0/1
combinations:

```bash
make clean-all
make BACKEND=<backend> SIMD=<0-or-1> WERROR=1 test
```

From `Audio_ALG/`:

```bash
python3 -m pytest -q pipelines/4ch_aec_bf_nr_res/tests
```

Pass conditions:

- [ ] One matched-filter/delay-estimator state is shared.
- [ ] Four independent linear AEC filters are used.
- [ ] Far-reference analysis FFT is computed once per hop.
- [ ] Four linear spectra cross the beamformer boundary and only one mono
      NR+RES path follows it.
- [ ] SRP/GSC and AEC/NR use the same selected grid.
- [ ] Pre/post token, reset, replay, and cross-instance misuse fail closed.
- [ ] The complete wrapper supports heap and caller-owned static-pool init.
- [ ] SRP/GSC and the complete wrapper allocate no heap after static init.
- [ ] SIMD/scalar parity passes.
- [ ] All three product grids pass.

### 9.1 Real four-microphone validation

Use the final microphone geometry, synchronized four-channel recordings, and
the production external beamformer or production SRP-PHAT/GSC implementation.
Cover near-end angles, far-end only, double-talk, moving speakers, moving echo
paths, unstable/wrong DOA, silence, low-level input, channel imbalance, failed
channels, and clipping.

Report DOA error, pre/post beamformer SNR/SDR, echo suppression, near-end
preservation, four-channel versus best-microphone behavior, reacquisition time,
clipping/non-finite counts, and listening results.

Beamforming output is not required to be bit-exact. It must be finite,
reproducible, structurally correct, and quality-approved.

### 9.2 Post-beam residual suppressor

The current bounded-R2/no-complete-stationarity approximation requires a
cohort verdict before production sign-off:

- [ ] Compare NR-only, RES-only, and NR+RES.
- [ ] Check near-end over-suppression.
- [ ] Tune using the real four-channel cohort if needed.
- [ ] Record the accepted limitation and result if the approximation remains.

## 10. Target-board acceptance

Measure all three product grids on the target toolchain and board:

- [ ] cycles per hop;
- [ ] real-time factor;
- [ ] peak CPU;
- [ ] static-pool bytes;
- [ ] stack high-water mark;
- [ ] heap allocation count after init;
- [ ] worst-case latency;
- [ ] path-change/reset peak cost;
- [ ] long-duration soak stability;
- [ ] NEON-on/off parity contract.

The owner must supply or approve CPU, memory, and latency budgets before this
section can receive a release PASS. A measurement report without an approved
budget is not an efficiency verdict.

## 11. AINR framework gate

Public AINR models are RNNoise-ERB, GTCRN, and DeepFilterNet2. Full training
and trained-model quality are out of scope.

### 11.1 Dataset generator

Run small real end-to-end jobs separately at 16 kHz and 48 kHz:

- [ ] generate;
- [ ] interrupted-run resume;
- [ ] refusal of a plain re-run into a non-empty output;
- [ ] index-gap rejection and temporary-WAV cleanup;
- [ ] pack;
- [ ] mmap load;
- [ ] WAV-only layout (`NNNNNN.wav`, legacy JSON ignored);
- [ ] post-resample `effective_rms_dbfs` validation;
- [ ] train/validation clipping contract;
- [ ] rejection of mixed sample-rate directories;
- [ ] packed sample-index validation.

The generator intentionally does not persist config, seed, or per-sample JSON.
Resume therefore cannot detect a changed generation distribution; that is a
documented WAV-only trade-off, not a contract-mismatch gate. Use a fresh output
directory when config or sample rate changes.

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AINR/dataset_gen/tests AINR/tests
```

### 11.2 Training-framework smoke

For every public model, use a tiny packed dataset and run at least 10--20
optimizer steps. Validate:

- [ ] CPU smoke;
- [ ] `--packed-dir`;
- [ ] `--gpu` argument and device selection;
- [ ] `--mmap`;
- [ ] progress reporting;
- [ ] checkpoint save/resume;
- [ ] dataset/grid/model-contract mismatch rejection;
- [ ] EMA and scheduler resume;
- [ ] NaN/Inf halt;
- [ ] validation does not update model state;
- [ ] deterministic seed and split behavior.

A short CUDA smoke may be executed on the training machine. A long training
run is not required.

### 11.3 Reference inference

A legal checkpoint created from a randomly initialized model is sufficient.
For each model verify:

- [ ] checkpoint load;
- [ ] WAV input/output;
- [ ] finite output;
- [ ] sample-rate/grid mismatch rejection;
- [ ] streaming state and reset;
- [ ] lookahead/warm-up frame indexing;
- [ ] every advertised export path;
- [ ] Python/C feature parity;
- [ ] C pre/post SIMD/scalar parity;
- [ ] accelerator-boundary tensor shape, layout, normalization, and latency
      documentation.

Do not claim trained-model quality without a trained checkpoint.

## 12. AIAEC framework gate

The current public candidates are Align-CRUSE, Align-ULCNet, DeepVQE-S, and
CAGCRN. A candidate that is not intended to
be runnable must be clearly marked experimental or removed from the public
surface.

### 12.1 Dataset generator

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AIAEC/dataset_gen/tests
```

Pass conditions:

- [ ] Five-stem contract passes.
- [ ] `linear_error` is not output-limiter processed.
- [ ] Complete stateful PBFDKF scenarios are rendered before chunking.
- [ ] Behavior hash is stable across supported Python versions.
- [ ] Source hash and behavior hash retain their separate meanings.
- [ ] Final WAV writes are atomic; legacy JSON is neither required nor read.
- [ ] Plain re-run, index gaps, and conflicting resume start indices fail
      closed.
- [ ] Temporary-WAV cleanup, pack, and mmap load pass.
- [ ] Every public model view maps stems consistently.
- [ ] 16/48 kHz model-grid validation passes.
- [ ] Packed four-stem version and PBFDKF behavior-hash compatibility are
      rejected without silent migration.

### 12.2 Training-architecture smoke

Run:

```bash
cd Audio_ALG
python3 -m pytest -q AIAEC/tests
```

For every public candidate, run a tiny packed dataset for 10--20 optimizer
steps and verify:

- [ ] `train.py --packed-dir ... --gpu ... --mmap`;
- [ ] finite loss;
- [ ] checkpoint save/resume;
- [ ] dataset fingerprint and split indices are stored;
- [ ] PBFDKF behavior contract is stored;
- [ ] contract mismatch fails closed;
- [ ] EMA, scheduler, and non-finite halt behavior;
- [ ] validation does not update training state.

### 12.3 Reference inference

For every public candidate verify:

- [ ] `inference.py` loads a legal random checkpoint.
- [ ] Mic/far or linear-error/far input mapping is correct.
- [ ] Feature and waveform shapes are correct.
- [ ] Supported 16/48 kHz grids match the model README.
- [ ] Offline/streaming behavior, state, reset, cache, and lookahead are
      explicit.
- [ ] Hardware inference tensor layout, normalization, output meaning, and
      pre/post-processing are documented.

No production quality claim is required or permitted without trained models.

## 13. Documentation gate

Every release surface must document:

- [ ] quick start;
- [ ] supported grids;
- [ ] heap/static construction;
- [ ] SIMD/backend switches;
- [ ] public ownership, lifetime, and reset rules;
- [ ] latency/lookahead;
- [ ] memory query and allocation example;
- [ ] errors and fail-closed behavior;
- [ ] thread-safety/reentrancy;
- [ ] known limitations;
- [ ] build/test/publish commands;
- [ ] API, checkpoint, and dataset migration;
- [ ] third-party licenses.

All documented symbols, flags, files, and commands must exist and work from a
clean clone. Historical/archive documents must not be linked as current
specifications.

## 14. Publish and tag

After all required gates pass:

1. Freeze final commits.
2. Change AEC from `4.0.0rc1` to the approved final version.
3. Consolidate NR's applicable `Unreleased` entries into a final version.
4. Record Audio_ALG compatibility and exact AEC/NR pins.
5. Add benchmark, real-array, board, and framework-smoke verdicts to the
   relevant CHANGELOG/README.
6. Publish in dependency order:

   ```text
   audio_common -> AEC -> NR -> Audio_ALG
   ```

7. Run `make publish` for the applicable KISS and NE10 deliverables.
8. Verify immutable artifacts, manifest, attestation, backend symbols,
   republish byte verification, and the `current` symlink.
9. Confirm Audio_ALG pins the exact published AEC and NR commits.
10. Create annotated/signed tags only after publish validation.
11. From a new recursive clone, rebuild the public consumer examples and run
    final smoke tests.

## 15. Final GO/NO-GO

Release is GO only when all of the following are true:

- [ ] Current-code AEC 800-case verdict passes.
- [ ] Native 48 kHz AEC verdict passes.
- [ ] The standalone-versus-pipeline NR product decision is closed.
- [ ] Real four-channel and post-beam RES verdicts pass.
- [ ] Target-board CPU, memory, latency, and zero-heap acceptance passes.
- [ ] AINR dataset/training/inference framework smoke passes.
- [ ] AIAEC dataset/training/inference framework smoke passes.
- [ ] Repositories are clean, versions are final, and submodule pins match.
- [ ] Publish manifests/attestations and final tags are complete.

Full AI dataset generation, long training, final checkpoints, and AI model
quality scores are explicitly outside this source/framework release gate.

## Appendix A. Known-good automated counts

Reference values re-measured on the current four-repo working tree, KISS
backend, SIMD=1, macOS, `SE/.venv` Python. Re-measure and re-record these
whenever the tree moves — a stale reference count silently converts a real
regression into a "matches the doc" pass. A count that **drops** is a
regression; a count that
**rises** without a corresponding new test in the diff means someone
parameterized an existing test rather than adding coverage. Always `make clean`
first — this tree has a stale-`.o` hazard that produces spurious segfaults and
is routinely misattributed to a real bug.

| Suite | Count |
|---|---:|
| `AEC` `make test-rate-structural` | 360 |
| `AEC` `make test-config-validation` | 388 |
| `AEC` `make test-delay-reset` | 16 |
| `AEC` `make test-delay-backward-quarantine` | 31 |
| `AEC` `python3 -m pytest python/tests` | 237 |
| `NR` `python3 -m pytest tests` | 56 |
| `Audio_ALG/lib/nr` `python3 -m pytest tests` | 56 |
| `Audio_ALG` `python3 -m pytest pipelines` | 58 |
| `Audio_ALG` `python3 -m pytest pipelines/tests` | 26 |
| `Audio_ALG` `python3 -m pytest pipelines/4ch_aec_bf_nr_res/tests` | 32 |
| `Audio_ALG` `python3 -m pytest AIAEC` | 336 |
| `Audio_ALG` `python3 -m pytest AINR` | 201 |
| `Audio_ALG` `python3 -m pytest AIAEC AINR pipelines` | 595 |

C-side `make test` targets that print `PASS:` markers rather than a count
(re-measured on the current working tree, KISS backend, `make clean` first):
`pipelines/mono_aec_nr_res` 75, `pipelines/mono_alignulcnet` 149,
`pipelines/4ch_aec_bf_nr_res` 316 (one-stop gate; also builds and runs the
`4ch_alignulcnet` binaries). Each additionally prints a small number of
non-`PASS:` pass lines (static smokes, board-skeleton profiles, adapter).

Pass/fail-only targets (no count): `AEC` `selftest`, `test-counter-saturation`,
`test-process-context`, `test-shared-far-spec`, `test-shared-fft-handle`,
`test-static-aec`, `test-zero-heap`, `test-detectors-parity-all`;
`audio_common` `selftest`, `test_audio_utils`, `test_pool`, `test_wav`,
`test_wav_nr_style`, `test_zero_heap`, `test_ne10_force_c`; `NR/c_impl` `test`,
`test-config`; both pipeline `make test` targets.

## Appendix B. Benchmark protocol

Rules that invalidate a benchmark result if broken. They exist because each has
already produced a wrong verdict in this project at least once.

- **`NO_PREALIGN=1` is mandatory for every AEC benchmark.** The eval driver's
  *default* is an offline GCC-PHAT pre-align crutch that hides exactly the
  class of timing/delay error being measured. A run without it is not weaker
  evidence, it is no evidence. Note `eval_aec_challenge.py` hardcodes
  pre-align in some paths — check, do not assume.
- **Verify the baseline before trusting any delta.** A stale or
  wrong-configuration baseline makes every Δ a mirage. Re-render the baseline
  from the recorded commit rather than reusing an old scores file.
- **AECMOS is only valid at 16 kHz** (16 kHz corpus, 16 kHz model). 8 kHz and
  48 kHz get parity/structural verdicts, never an AECMOS number.
- **Compare on the Pareto front, not on one axis.** Beating on `deg` alone is
  not a win — less cancellation automatically raises `deg`. Compare at matched
  echo, or at matched deg.
- **A neutral blind-test result is "no harm", not "no effect".** If the changed
  quantity feeds a gate that rarely fires on the test corpus, the corpus did
  not exercise the change. Say which of the two the result is.
- **Archive per-case scores, the exact baseline→candidate diff, and the
  harness** for any A/B that gates a release decision — a summary line is not
  reproducible. See `AEC/eval/ab_evidence/` for the expected shape.

## Appendix C. Cross-repo mechanics

- Submodule pins are updated by committing the standalone repo first, then
  fetching that commit into `Audio_ALG/lib/<name>` and committing the new
  gitlink. Publish order is `audio_common -> AEC -> NR -> Audio_ALG` (§14).
- After any file moves, stage with `git add -A` and then **review the
  rename detection** (`git diff --cached --name-status -M`). Before committing,
  confirm every deletion either has a counterpart at a new path or is
  recoverable from git history. Do not run `git checkout .`, `git reset`, or
  `git stash` with `mv`-based moves pending — untracked new paths are not
  covered by a plain stash.
- Confirm `git rev-list --count @{u}..HEAD == 0` and
  `git rev-parse origin/main == git rev-parse HEAD` per repo before claiming
  anything is pushed.
