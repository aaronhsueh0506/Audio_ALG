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
      you.** `aec_create()` / `audio_pipeline_4ch_create()` / the other `*_create()` constructors use
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
- [ ] Third-party NE10, SRP, GSC, and other redistributed code has
      correct license/source attribution.

## 3. `audiolib` gate

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

Pass conditions:

- [ ] No compiler warnings.
- [ ] SIMD/scalar kernel parity passes.
- [ ] FFT heap/static parity passes.
- [ ] Pre-gain, resampler, WAV, invalid-input, and pool tests pass.
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

## 8. Mono pipeline gate

From `Audio_ALG/pipelines/`, run both SIMD 0/1 combinations:

```bash
make clean-all
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
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

### 8.1 Mono DeepFilterNet2 variant (`mono_aec_dfn_res`)

`make -C pipelines test` also builds and runs the shared post-RES DFN2
stage, the rate bridge that takes that stage to its 48 kHz grid from a
product-rate host, and the mono example that composes lib/aec's `Aec`, the
sample chain with the RES gain only (`enable_nr=0`, no synthesis) and
`samples/dfn_hop.h`:

```bash
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_dfn_res_stage, test_dfn_rate_bridge, test_mono_dfn, test_se_common,
#       mono_aec_dfn_res (board skeleton at its 16 kHz default, also run with
#       --delay-num-filters 1 --cng --atten-lim 12, and with
#       --sample-rate 48000)
make -C pipelines/mono_aec_dfn_res test   # scoped: test_dfn_res_stage,
#       test_mono_dfn and the skeleton only -- it does NOT cover
#       test_dfn_rate_bridge or the --sample-rate 48000 run, so the top-level
#       target above is the one that gates a release.
```

Pass conditions:

- [ ] At a 48 kHz host, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`mono_aec_nr_res/tests/conv_chain_mono.h`) with the denoiser off,
      delayed by exactly two hops, with comfort noise OFF and ON, across an AEC
      delay change.
- [ ] The stimulus actually engages RES (`min G_res < 0.9`), so an
      estimate/apply swap cannot pass unnoticed.
- [ ] Stage counters show no mid-stream reset (`frames == hops`,
      `commits == 0`, `skips == hops - 1` on the identity path).
- [ ] `erb_inv` that is not a bitwise partition of unity is refused at
      `get_mem_requirements()`.
- [ ] Native-grid gates pass for 16 kHz/fft 256, 16 kHz/fft 512 and
      8 kHz/fft 256: the documented per-hop 48 kHz frame schedule (0,1,1,1 at
      16 kHz/hop 128; 1,2 on the other two), the cumulative frame count, the
      impulse-pinned added delay (672 / 629 / 330 native samples), and a
      pass-band tolerance against the conventional NR-off output through the
      same 0.35 fs low-pass. The native path is a tolerance, NOT a memcmp:
      the signal crosses two FIR resamplers.
- [ ] `test_dfn_rate_bridge` passes on all three native grids standalone
      (schedule, prefill/added-delay table, impulse, tolerance identity,
      E-only silence, reset parity, rejection row).
- [ ] The board skeleton runs at `--sample-rate 16000` (default),
      `8000` and `48000`, and prints a pool size, hop and algorithmic latency
      for each.
- [ ] Publish artifacts include `libaudio_pipeline_dfn.a` (carrying
      `dfn_rate_bridge.o`) and the `mono_aec_dfn_res` board skeleton alongside
      the other artifacts.

## 9. Four-channel pipeline gate

From `Audio_ALG/pipelines/4ch_aec_bf_nr_res/`, run both SIMD 0/1
combinations:

```bash
make clean-all
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
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
- [ ] The unit (`AudioPipeline4Ch`) supports heap and caller-owned static-pool init.
- [ ] SRP/GSC and the unit allocate no heap after static init.
- [ ] The unit contains no RES, NR, comfort noise or synthesis; those stages are
      separate libraries that the example programs initialise and chain, and
      `unit + fuse + sample chain` equals the hand-written reference chain bit
      for bit (`test_audio_pipeline_4ch`).
- [ ] SIMD/scalar parity passes.
- [ ] All three product grids pass.

### 9.0 Four-channel DeepFilterNet2 variant (`4ch_aec_bf_dfn_res`)

`make -C pipelines/4ch_aec_bf_nr_res test` also builds and runs the DFN2
example over the core (`enable_post=1`; the sample chain with the RES gain only,
no denoiser, no synthesis; `samples/dfn_hop.h`):

```bash
make -C pipelines/4ch_aec_bf_nr_res BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_4ch_dfn, 4ch_aec_bf_dfn_res (board skeleton at its 16 kHz
#       default, also run with --delay-num-filters 1 --cng --atten-lim 12,
#       and with --sample-rate 48000)
make -C pipelines/4ch_aec_bf_dfn_res test   # scoped: test_4ch_dfn and the
#       skeleton only -- it does NOT cover the --sample-rate 48000 run, and the
#       rate bridge's own unit test lives under the mono target
#       (`make -C pipelines test`, section 8.1). Run both before a release.
```

Pass conditions:

- [ ] At a 48 kHz core, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`4ch_aec_bf_nr_res/tests/conv_chain_4ch.h`) with the denoiser off,
      delayed by exactly two hops, with both fuse forms (reconstructed weights
      and trusted spectrum) and with comfort noise OFF and ON, across an AEC
      delay change.
- [ ] The stimulus actually engages the post-beam RES (a RES-off core differs).
- [ ] Stage counters show no mid-stream reset (the stage is not reset by a
      delay change of the shared aligner).
- [ ] A rate the core does not offer is refused at
      `four_aec_nr_res_get_mem_requirements()`, and a non-partition-of-unity
      `erb_inv` at `dfn_res_stage_get_mem_requirements()` (`test_dfn_res_stage`).
      8 kHz in particular must be refused: the core has no 8 kHz grid, so the
      4-channel DFN2 example cannot offer one either.
- [ ] Native-grid gates pass for 16 kHz/fft 256 and 16 kHz/fft 512: the
      per-hop 48 kHz frame schedule (0,1,1,1 at hop 128; 1,2 at hop 256), the
      cumulative frame count, the impulse-pinned added delay (672 / 629 native
      samples) and a pass-band tolerance against the conventional NR-off
      output. The native path is a tolerance, NOT a memcmp.
- [ ] The build produces `libdfn_res_stage.a` (carrying
      `dfn_rate_bridge.o`); `FOUR_AEC_NR_RES_LAYOUT_VERSION` is 23 and
      `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` is 18.

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

Public AINR models are RNNoise-ERB, GTCRN, DeepFilterNet2, and DeepFilterNet3. Full training
and trained-model quality are out of scope.

### 11.1 Dataset generator

Run small real end-to-end jobs separately at 16 kHz and 48 kHz:

- [ ] generate;
- [ ] interrupted-run resume;
- [ ] refusal of a plain re-run into a non-empty output;
- [ ] index-gap rejection and temporary-WAV cleanup;
- [ ] pack;
- [ ] mmap load;
- [ ] WAV-only layout (`NNNNNN.wav`, leftover JSON ignored);
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

### 11.4 C pre/post archive

`AINR/Makefile` builds `libainr_prepost.a` — `dfn2_process`, `dfn2_model_io`,
`dfn2_prepost` and `gtcrn_process`. RNNoise-ERB is deliberately NOT a member:
it has its own Makefile and no archive.

```bash
make -C AINR SIMD=1 WERROR=1 lib
make -C AINR SIMD=1 WERROR=1 print-lib-path   # absolute path of THIS config
make -C AINR SIMD=0 WERROR=1 test
```

- [ ] Objects, test binaries and the archive are configuration-keyed
      (`build/simd<N>-<sig>/`), so a `SIMD=0` build never hands back the
      `SIMD=1` archive; resolve the archive with `print-lib-path`, never by
      spelling the directory.
- [ ] `WERROR=1` actually reaches the compiler.
- [ ] The SIMD knob changes only `dfn2_process.o` and `gtcrn_process.o`; the
      other members are byte-identical between `SIMD=0` and `SIMD=1`.

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
- [ ] Final WAV writes are atomic; leftover JSON is neither required nor read.
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

### 12.4 C pre/post archive

`AIAEC/Makefile` builds `libaiaec_prepost.a` — `ulcnet_process`,
`ulcnet_model_io`, `ulcnet_prepost`, `ulcnet_accelerator_adapter`,
`aiaec_process`, `deepvqe_process`, `deepvqe_prepost`.

```bash
make -C AIAEC SIMD=1 WERROR=1 lib
make -C AIAEC SIMD=1 WERROR=1 print-lib-path   # absolute path of THIS config
make -C AIAEC SIMD=0 WERROR=1 test             # lib + pytest
```

- [ ] The archive is configuration-keyed (`build/simd<N>-<sig>/`, keyed on
      `CC`/`AR`/`CFLAGS`/`CPPFLAGS`/`SIMD`/`WERROR`) and a signature collision
      is refused rather than reinterpreted; resolve it with `print-lib-path`.
- [ ] No AIAEC translation unit includes `simd_kernels.h`, so the SIMD knob
      must leave every AIAEC object byte-identical — a difference here means
      something started depending on the kernels and the claim needs redoing.
- [ ] Each pre/post class (`ulcnet_prepost`, `deepvqe_prepost`) is a THIRD
      translation unit composing two files that stay independently linkable
      for their own parity tests, and its byte-identity tests against the
      hand-composed path pass.

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
   audiolib -> AEC -> NR -> Audio_ALG
   ```

7. Run `make publish` for the applicable NE10 deliverables.
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

## Appendix A. Reference test targets

Re-measure counts on the release tree and record them in the dated release
report. A count that **drops** is a regression; a count that **rises** without
a corresponding new test in the diff means someone parameterized an existing
test rather than adding coverage. Always `make clean` first — the tree has a
stale-`.o` hazard that produces spurious segfaults and is routinely
misattributed to a real bug.

Count-bearing suites: `AEC` `make test-rate-structural`,
`test-config-validation`, `test-delay-reset`, `test-aec-res`,
`test-linear-context` and `python3 -m pytest python/tests`; `NR` /
`Audio_ALG/lib/nr` `python3 -m pytest python/tests`; `Audio_ALG`
`python3 -m pytest pipelines`, `AIAEC`, `AINR`, and the combined
`python3 -m pytest AIAEC AINR pipelines`. Per-package collection counts
(`--collect-only`) must sum exactly to the combined row, which checks that the
combined invocation collects everything.

The combined row is the one that matters: pytest gives every `conftest.py` the
same top-level module name, so a test doing `from conftest import ...` picks up
whichever package was imported first and the whole run dies at collection while
each package on its own stays green.
`AIAEC/tests/test_cross_package_collection.py` drives the combined invocation
in a subprocess, in both argument orders. Collection is not the only place this
bites: every AINR project also defines a top-level `train`, `model` and
`export_onnx`, and a bare import made after a sibling's is cached in
`sys.modules` is handed the sibling's module at RUN time. Consumers of those
modules drop the cached names and front their own directory before importing,
as the projects' tests do; `pipelines/tests/test_dfn2_stage.py` pins that for
the DFN2 stage. Re-run the combined row, not just the per-package rows.

One combined-run test depends on the host rather than the tree:
`AIAEC/tests/test_training_common.py::test_the_mps_stft_backward_bound_is_where_the_backend_actually_breaks`
probes the host's MPS backend and fails where the backend agrees with the CPU.

C-side `make test` targets that print `PASS:` markers rather than a count
(`make clean` first, `^PASS:` lines counted): `pipelines` (all applications),
`pipelines/mono_aec_nr_res`, `pipelines/mono_alignulcnet` (the same count on the
16 kHz and the `ULCNET_GRID=48k` build) and `pipelines/4ch_aec_bf_nr_res`
(one-stop gate; also builds and runs the `4ch_alignulcnet` binaries, whose test
prints a single `All 4ch_ulcnet tests passed` line and no `PASS:`
markers, and the `4ch_aec_bf_dfn_res` command-line runs; the same count on the
`ULCNET_GRID=48k` build). Each additionally prints a small number of non-`PASS:`
pass lines (static smokes, board-skeleton profiles, adapter).

Pass/fail-only targets (no count): `AEC` `selftest`, `test-counter-saturation`,
`test-process-context`, `test-shared-far-spec`, `test-shared-fft-handle`,
`test-static-aec` (needs the doubletalk blind-test wav pair, overridable with
`TEST_STATIC_MIC` / `TEST_STATIC_REF`), `test-zero-heap`,
`test-detectors-parity-all`; `audiolib` `selftest` (includes `test_stft` and
`test_stft_guard`), `test_audio_utils`, `test_pool`, `test_wav`,
`test_wav_nr_style`, `test_zero_heap`, `test_ne10_force_c`; `NR/c_impl`
`test-config`, `test-reconfigure`, `test-noise-restart`, `test-scene-change`,
`test-config-parity`, `test-bit-exact` (plain `make test` there only runs the
denoiser when a `test.wav` sits next to it, so it is not a gate).

## Appendix B. Benchmark protocol

Rules that invalidate a benchmark result if broken.

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
  gitlink. Publish order is `audiolib -> AEC -> NR -> Audio_ALG` (§14).
- After any file moves, stage with `git add -A` and then **review the
  rename detection** (`git diff --cached --name-status -M`). Before committing,
  confirm every deletion either has a counterpart at a new path or is
  recoverable from git history. Do not run `git checkout .`, `git reset`, or
  `git stash` with `mv`-based moves pending — untracked new paths are not
  covered by a plain stash.
- Confirm `git rev-list --count @{u}..HEAD == 0` and
  `git rev-parse origin/main == git rev-parse HEAD` per repo before claiming
  anything is pushed.
