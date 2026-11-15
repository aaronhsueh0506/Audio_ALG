# 管線驗收

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

The source-text guards in that tree read the AEC library's public headers (`aec.h`,
`aec_balanced_config.h`) and, for the parity hash, its `include/internal/aec_impl.h` from the same place `lib/deps.mk` points
the build at: `AEC_DIR` from the environment (absolute path), else the sibling
`../AEC/c_impl`. They pin `AEC_LAYOUT_VERSION` against the docs
(`pipelines/tests/test_doc_layout_versions.py`), the comfort-noise floor and
seed against the oracle chains, the Python twin and the C witnesses
(`pipelines/mono/tests/test_cng_floor_constant.py`, `test_cng_aec3.py`), and
the file list `pyc_parity.py` hashes into its result against the tree
(`pipelines/tests/test_pyc_parity_sources.py`).

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
product-rate host (`dfn/dfn_rate_bridge.c`, a thin adapter over `stream/ainr/bridge`'s
model-free `AinrBridge`; `make -C stream/ainr/bridge test` is the bridge's own gate), and the mono example that composes lib/aec's `Aec` (its own RES
gain, formed into P in the example's `rig_process`; no denoiser; the host synthesis is
the example's `AcIstft` on the direct path and the bridge's own otherwise) and
`dfn/dfn_hop.h`. The example and `dfn_hop.h` run the model through the model stream
(`dfn2_stream_*_freq` of `stream/ainr/dfn2`, memory mode) and `AinrBridge`, not through the stage;
`test_dfn_hop` holds the hop to the stage and the rate bridge bit for bit, `test_dfn_stream_equivalence` the stream to the stage:

```bash
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_dfn_res_stage, test_dfn_rate_bridge, test_dfn_stream_equivalence, test_mono_dfn, test_dfn_hop,
#       test_array_sample_post, mono_aec_dfn_res (board skeleton at its 16 kHz default,
#       also run with --delay-num-filters 1 --cng --atten-lim 12, and with
#       --sample-rate 48000 and 8000, and with --ai-sample-rate 16000
#       --ai-fft-size 256, the model grid the model stream runs directly on the host grid)
make -C pipelines/mono/aec_dfn_res test   # scoped: test_dfn_res_stage,
#       test_dfn_rate_bridge, test_mono_dfn and the skeleton runs (16 kHz,
#       48 kHz, 8 kHz, the model-grid flags, the malformed-number and
#       model-grid refusals); the top-level target
#       above is the one that gates a release (it also runs test_dfn_hop,
#       test_array_sample_post and the rest of the pipelines/ set).
```

The stage runs the DeepFilterNet2 grid its config carries, lookahead pair included (a row of
the example's table, `--ai-sample-rate` / `--ai-fft-size`, default 48 kHz / 1024 / 1/1; 15 legal
pairs); the `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` make variables (default 1/1) choose only
the pair the test programs run. The delays quoted in the pass conditions below are for the
shipped 1/1 pair. Other pairs, and the stage tests at the 16 and 32 kHz grids:

```bash
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-pairs
#       default DFN_TEST_PAIRS="0/0 0/1 2/0 2/4"; override with
#       DFN_TEST_PAIRS="0/1 2/4" (mask/df pairs from 0..2/0..4); an empty list is an
#       error, and make -n / -q / -t only print the plan
make -C pipelines BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test-dfn-grids
#       test_dfn_res_stage, test_dfn_rate_bridge and test_dfn_hop built for the
#       16 and 32 kHz test grids (DFN2_TEST_GRID_KHZ), with the test header's
#       pair 1/1 -- not the 0/0 of those rows in the example table
```

`test-pairs` rebuilds and runs `test_dfn_res_stage`, `test_dfn_rate_bridge`,
`test_dfn_hop`, `test_mono_dfn`, `test_readme_flow` and `mono_aec_dfn_res` for
each listed pair only. Every one of them prints the pair it was built with and, for the stage test and
the mono composition, the stage latency it measured, and `dfn/tests/check_pair_report.sh` fails the
target when the pair or the latency (`mask + df` hops, `(mask + df + 1) * hop` samples) is not the one
asked for -- a pair flag that never reached the test build prints 1/1 and is red. The other pairs are not built or run by this target
(the AINR C hosts have their own pair tests, section 11.4), and no checkpoint
trained for a pair other than 1/1 exists here, so their audio quality is
unmeasured.

Pass conditions:

- [ ] At a 48 kHz host, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`mono/aec_nr_res/tests/conv_chain_mono.h`) with the denoiser off,
      delayed by exactly two hops (`M + D` for the build's pair), with comfort
      noise OFF and ON, across an AEC delay change.
- [ ] The stimulus actually engages RES (`min G_res < 0.9`), so an
      estimate/apply swap cannot pass unnoticed.
- [ ] The model stream's state generation shows no mid-stream reset
      (`dfn_hop_state_generation()` is `1 + M` at the end of the identity run: 1 plus one per
      warm-up frame, i.e. 2 for the shipped 1/1 pair; a reset or a refused commit would add to it).
- [ ] `erb_inv` that is not a bitwise partition of unity is refused at
      `get_mem_requirements()` (stage) and at `init` (`dfn2_stream_init*`, `ainr_init*`); the two
      example mains run the model stream, so they report it as `pipeline init failed` (exit 1),
      from `dfn_hop_pool_build()`.
- [ ] `--erb-fwd FILE` and `--erb-inv FILE` take the two raw float32 ERB matrices of the
      model grid (exactly `n_bins * n_erb` floats each): a missing value or only one of the
      two exits 2, an unreadable file or a wrong float count exits 1; with neither option a
      unit partition serves the fail-open smoke run only.
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
      `dfn_rate_bridge.o`, built from `dfn/dfn_rate_bridge.c`, `ainr_bridge.o`, built from
      `stream/ainr/bridge/ainr_bridge.c`, and `dfn2_stream.o`, built from `stream/ainr/dfn2/dfn2_stream.c`) and the `mono_aec_dfn_res` board skeleton alongside
      the other artifacts.

## 9. Four-channel pipeline gate

From `Audio_ALG/pipelines/4ch/`, run both SIMD 0/1
combinations:

```bash
make clean-all
make BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
```

From `Audio_ALG/`:

```bash
python3 -m pytest -q pipelines/4ch/tests
```

Pass conditions:

- [ ] One matched-filter/delay-estimator state is shared.
- [ ] Four independent linear AEC filters are used.
- [ ] Far-reference analysis FFT is computed once per hop.
- [ ] Four linear spectra cross the beamformer boundary and only one mono
      NR+RES path follows it.
- [ ] SRP-PHAT/BF and AEC/NR use the same selected grid.
- [ ] Sequence rules (linear before residual; with the sample helper, front before beam before fuse), reset and refused calls fail closed: a call out of order returns `AEC_ERR_SEQUENCE` and changes nothing, a residual stage refused with `AEC_ERR_DSP` has already reset the array (the sample helper resets everything it owns).
- [ ] The 4-channel flow is one `Aec` instance of lib/aec (`AecConfig.n_mics = AEC_ARRAY_MICS`), not a pipeline module: `aec_get_mem_requirements()` + `aec_init_ex()` (or `aec_create()`) build the shared `DelayAec`, the four lanes, the alignment ring, the fusion buffers and the gain stage in one pool; the application supplies no lane or estimator, and the lanes have no public handle.
- [ ] The array and the sample beam helper (SRP-PHAT/BF) allocate no heap after static init.
- [ ] The array contains no beamformer and no NR, comfort-noise rendering or synthesis (`aec_process_residual()` fuses with the caller's weights; the gain comes from the array's gain stage under `AEC_RESIDUAL_FULL`, rendered by `aec_apply_gain()` / `aec_apply_cng()`); the denoiser and the synthesis are separate libraries that the
      example programs initialise and chain, and the array plus the examples' per-hop chain equals
      the hand-written reference chain bit for bit (`test_array_sample_beam`, oracle
      `4ch/tests/conv_chain_4ch.h`, a second array driven through explicit arrays); the array equalled the former flow bit for bit
      (one-off comparison against the pre-fold flow, retired on 2026-11-13; the AEC repo's `test_aec_array` keeps the contract checks).
- [ ] There is no array archive: `libaec.a` holds the array and references no SRP-PHAT/BF symbol; the
      sample helper (`array_sample_beam.o`) is in no archive.
- [ ] `libaec.a` and the 4ch `test_4ch_chain` build with the spatial archives absent (`make test_4ch_chain` needs common, lib/aec and lib/nr only; check SP-S21..SP-S25 in `pipelines/scripts/test_build_isolation.sh`).
- [ ] SIMD/scalar parity passes.
- [ ] All three product grids pass.

### 9.0 Four-channel DeepFilterNet2 variant (`4ch_aec_bf_dfn_res`)

`make -C pipelines/4ch test` also builds and runs the DFN2
example over the array plus the sample beam helper (real SRP-PHAT/BF; the post-beam RES is the array's gain stage; no denoiser; the host synthesis is the example's `AcIstft` on the
direct path and the bridge's own otherwise; `dfn/dfn_hop.h` over the model stream `stream/ainr/dfn2`):

```bash
make -C pipelines/4ch BACKEND=ne10 SIMD=<0-or-1> WERROR=1 test
# adds: test_4ch_dfn, 4ch_aec_bf_dfn_res (board skeleton at its 16 kHz
#       default, also run with --delay-num-filters 1 --cng --atten-lim 12,
#       with --sample-rate 48000 and with --ai-sample-rate 16000 --ai-fft-size 256)
make -C pipelines/4ch/aec_bf_dfn_res test   # scoped: test_4ch_dfn and the
#       skeleton runs (16 kHz, 48 kHz, the model-grid flags, the malformed-number,
#       8 kHz and model-grid refusals); the rate bridge's own unit test lives under the mono target
#       (`make -C pipelines test`, section 8.1). Run both before a release.
#       The 4-channel Makefile takes the same DFN2_MASK_LOOKAHEAD /
#       DFN2_DF_LOOKAHEAD variables (for test_4ch_dfn alone) and has no test-pairs target.
```

Pass conditions:

- [ ] At a 48 kHz host, with no model (or a failing one), the example output is
      byte-equal to the hand-written conventional chain
      (`4ch/tests/conv_chain_4ch.h`, fed the sample helper's beam weights
      and beamformed spectrum) with the denoiser off,
      delayed by exactly two hops (`M + D` for the build's pair), with comfort
      noise OFF and ON, across an AEC delay change.
- [ ] The stimulus actually engages the post-beam RES (a run without the array's gain stage differs).
- [ ] The model stream's state generation shows no mid-stream reset (1 plus one per warm-up
      frame at the end of the run: the hop is not reset by a delay change of the shared aligner).
      The mono test asserts the same.
- [ ] A rate the array does not offer is refused at
      `aec_get_mem_requirements()`, and a non-partition-of-unity
      `erb_inv` at `dfn_res_stage_get_mem_requirements()` (`test_dfn_res_stage`) and by the model
      stream's init, that is by `dfn_hop_pool_build()` (`test_dfn_hop`; `make -C stream/ainr/dfn2 erb-partition`).
      8 kHz in particular must be refused: the array has no 8 kHz grid, so the
      4-channel DFN2 example cannot offer one either.
- [ ] Native-grid gates pass for 16 kHz/fft 256 and 16 kHz/fft 512: the
      per-hop 48 kHz frame schedule (0,1,1,1 at hop 128; 1,2 at hop 256), the
      cumulative frame count, the impulse-pinned added delay (shipped 1/1 pair: 672 / 629 native
      samples) and a pass-band tolerance against the conventional NR-off
      output. The native path is a tolerance, NOT a memcmp.
- [ ] The build produces `libdfn_res_stage.a` (carrying
      `dfn_rate_bridge.o`, `ainr_bridge.o` and `dfn2_stream.o`); `AEC_LAYOUT_VERSION` is 3 (the `Aec` pool, array included).

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

[返回本主題](README.md)
