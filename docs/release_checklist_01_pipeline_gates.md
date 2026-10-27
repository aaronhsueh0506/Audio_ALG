[Index](release_checklist.md) | [Previous](release_checklist.md) | [Next](release_checklist_02_ainr_aiaec_gates.md) <!-- split-nav -->
## 8. Mono pipeline gate

From `Audio_ALG/pipelines/`, run both SIMD 0/1 combinations:

```bash
make clean-all
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
```

From `Audio_ALG/` (the Python tests of every pipeline folder live next to their
code, so the whole `pipelines/` tree is collected, not only `pipelines/tests`):

```bash
python3 -m pytest -q pipelines
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
product-rate host, and the mono example that composes lib/aec's `Aec` (its own RES
gain, formed into P in the example's `rig_process`, no denoiser, no synthesis) and
`dfn/dfn_hop.h`:

```bash
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_dfn_res_stage, test_dfn_rate_bridge, test_mono_dfn, test_dfn_hop,
#       test_mcab_sample_post, mono_aec_dfn_res (board skeleton at its 16 kHz default,
#       also run with --delay-num-filters 1 --cng --atten-lim 12, and with
#       --sample-rate 48000 and 8000)
make -C pipelines/dfn/mono_aec_dfn_res test   # scoped: test_dfn_res_stage,
#       test_dfn_rate_bridge, test_mono_dfn and the skeleton runs (16 kHz,
#       48 kHz, 8 kHz, the malformed-number refusals); the top-level target
#       above is the one that gates a release (it also runs test_dfn_hop,
#       test_mcab_sample_post and the rest of the pipelines/ set).
```

The stage is built for one DeepFilterNet2 lookahead pair (`DFN2_MASK_LOOKAHEAD`
/ `DFN2_DF_LOOKAHEAD` make variables, default 1/1, 15 legal pairs); the delays
quoted in the pass conditions below are for the shipped 1/1 pair. Other pairs:

```bash
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-pairs
#       default DFN_TEST_PAIRS="0/0 0/1 2/0 2/4"; override with
#       DFN_TEST_PAIRS="0/1 2/4" (mask/df pairs from 0..2/0..4); an empty list is an
#       error, and make -n / -q / -t only print the plan
```

`test-pairs` rebuilds and runs `test_dfn_res_stage`, `test_dfn_rate_bridge`,
`test_dfn_hop`, `test_mono_dfn`, `test_readme_flow` and `mono_aec_dfn_res` for
each listed pair only; the other pairs are not built or run by this target
(the AINR C hosts have their own pair tests, section 11.4), and no checkpoint
trained for a pair other than 1/1 exists here, so their audio quality is
unmeasured.

Pass conditions:

- [ ] At a 48 kHz host, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`mono_aec_nr_res/tests/conv_chain_mono.h`) with the denoiser off,
      delayed by exactly two hops (`M + D` for the build's pair), with comfort
      noise OFF and ON, across an AEC delay change.
- [ ] The stimulus actually engages RES (`min G_res < 0.9`), so an
      estimate/apply swap cannot pass unnoticed.
- [ ] Stage counters show no mid-stream reset (`frames == hops`,
      `commits == 0`, `skips == hops - M` on the identity path, i.e. `hops - 1`
      for the shipped 1/1 pair).
- [ ] `erb_inv` that is not a bitwise partition of unity is refused at
      `get_mem_requirements()`.
- [ ] Native-grid gates pass for 16 kHz/fft 256, 16 kHz/fft 512 and
      8 kHz/fft 256: the documented per-hop 48 kHz frame schedule (0,1,1,1 at
      16 kHz/hop 128; 1,2 on the other two), the cumulative frame count, the
      impulse-pinned added delay (shipped 1/1 pair: 672 / 629 / 330 native samples), and a
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
      `dfn_rate_bridge.o`, built from `dfn/dfn_rate_bridge.c`) and the `mono_aec_dfn_res` board skeleton alongside
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
- [ ] SRP-PHAT/BF and AEC/NR use the same selected grid.
- [ ] Sequence rules (front before fuse; with the sample helper, front before beam before fuse), reset and refused calls fail closed: a call out of order returns `MCAB_SEQUENCE_ERROR` and changes nothing, a fuse refused with `MCAB_DSP_ERROR` resets the flow (the sample helper resets everything it owns).
- [ ] `mcab` is a flow layer, not a library: the application creates the four lanes (`aec_init()` over `mcab_lane_config()`, or a config stamped with `mcab_apply_lane_contract()`) and, in MATCHED mode, the shared `DelayAec`, and passes them to `mcab_init_ex()`; `mcab` supports caller-owned pool init (and `mcab_create()` for its own small pool), never creates, resets by destroying or frees an injected object, and refuses a flow built from the wrong objects.
- [ ] `mcab` and the sample beam helper (SRP-PHAT/BF) allocate no heap after static init.
- [ ] `mcab` contains no beamformer and no post-beam RES, NR, comfort noise or synthesis
      (`mcab_fuse()` leaves `frame.g_res_ctx` NULL); those stages are separate libraries that the
      example programs initialise and chain (the RES is an `AecRes` created with the
      `aec_res_*` functions of lib/aec), and the flow plus the examples' per-hop chain equals
      the hand-written reference chain bit for bit (`test_mcab_sample_beam`, oracle
      `4ch_aec_bf_nr_res/tests/conv_chain_mcab.h`).
- [ ] `libaudio_pipeline_mcab.a` holds `mcab.o` only and references no SRP-PHAT/BF symbol; the
      sample helpers (`mcab_sample_flow.o`, `mcab_sample_beam.o`) are not in it.
- [ ] `make libaudio_pipeline_mcab.a` builds with `NR_DIR` absent (the archive needs only audiolib common and
      lib/aec; check SP-S21 in `pipelines/scripts/test_build_isolation.sh`).
- [ ] SIMD/scalar parity passes.
- [ ] All three product grids pass.

### 9.0 Four-channel DeepFilterNet2 variant (`4ch_aec_bf_dfn_res`)

`make -C pipelines/4ch_aec_bf_nr_res test` also builds and runs the DFN2
example over `mcab` plus the sample beam helper (real SRP-PHAT/BF; the post-beam RES is an `AecRes`
the example creates itself; no denoiser, no synthesis; `dfn/dfn_hop.h`):

```bash
make -C pipelines/4ch_aec_bf_nr_res BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_4ch_dfn, 4ch_aec_bf_dfn_res (board skeleton at its 16 kHz
#       default, also run with --delay-num-filters 1 --cng --atten-lim 12,
#       and with --sample-rate 48000)
make -C pipelines/dfn/4ch_aec_bf_dfn_res test   # scoped: test_4ch_dfn and the
#       skeleton runs (16 kHz, 48 kHz, the malformed-number and 8 kHz
#       refusals); the rate bridge's own unit test lives under the mono target
#       (`make -C pipelines test`, section 8.1). Run both before a release.
#       The 4-channel Makefile takes the same DFN2_MASK_LOOKAHEAD /
#       DFN2_DF_LOOKAHEAD variables and has no test-pairs target.
```

Pass conditions:

- [ ] At a 48 kHz host, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`4ch_aec_bf_nr_res/tests/conv_chain_mcab.h`, fed the sample helper's beam weights
      and beamformed spectrum) with the denoiser off,
      delayed by exactly two hops (`M + D` for the build's pair), with comfort
      noise OFF and ON, across an AEC delay change.
- [ ] The stimulus actually engages the post-beam RES (a run without the example's `AecRes` differs).
- [ ] Stage counters show no mid-stream reset (the stage is not reset by a
      delay change of the shared aligner).
- [ ] A rate `mcab` does not offer is refused at
      `mcab_get_mem_requirements()`, and a non-partition-of-unity
      `erb_inv` at `dfn_res_stage_get_mem_requirements()` (`test_dfn_res_stage`).
      8 kHz in particular must be refused: `mcab` has no 8 kHz grid, so the
      4-channel DFN2 example cannot offer one either.
- [ ] Native-grid gates pass for 16 kHz/fft 256 and 16 kHz/fft 512: the
      per-hop 48 kHz frame schedule (0,1,1,1 at hop 128; 1,2 at hop 256), the
      cumulative frame count, the impulse-pinned added delay (shipped 1/1 pair: 672 / 629 native
      samples) and a pass-band tolerance against the conventional NR-off
      output. The native path is a tolerance, NOT a memcmp.
- [ ] The build produces `libdfn_res_stage.a` (carrying
      `dfn_rate_bridge.o`); `MCAB_LAYOUT_VERSION` is 25 (the pool of `mcab` alone).

### 9.1 Real four-microphone validation

Use the final microphone geometry, synchronized four-channel recordings, and
the production external beamformer or production SRP-PHAT/BF implementation.
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

[Index](release_checklist.md) | [Previous](release_checklist.md) | [Next](release_checklist_02_ainr_aiaec_gates.md) <!-- split-nav -->
