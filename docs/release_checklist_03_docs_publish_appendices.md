[Index](release_checklist.md) | [Previous](release_checklist_02_ainr_aiaec_gates.md) <!-- split-nav -->
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
2. Change AEC from `5.0.0rc1` to the approved final version.
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
  gitlink. Publish order is `audiolib -> AEC -> NR -> Audio_ALG` ([§14](#14-publish-and-tag)).
- After any file moves, stage with `git add -A` and then **review the
  rename detection** (`git diff --cached --name-status -M`). Before committing,
  confirm every deletion either has a counterpart at a new path or is
  recoverable from git history. Do not run `git checkout .`, `git reset`, or
  `git stash` with `mv`-based moves pending — untracked new paths are not
  covered by a plain stash.
- Confirm `git rev-list --count @{u}..HEAD == 0` and
  `git rev-parse origin/main == git rev-parse HEAD` per repo before claiming
  anything is pushed.

[Index](release_checklist.md) | [Previous](release_checklist_02_ainr_aiaec_gates.md) <!-- split-nav -->
