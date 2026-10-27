# pipelines/dfn

The DeepFilterNet2 (DFN2) post-RES stage for the AEC pipelines and the two example programs that
compose it. Two modules carry a layout version and a 32-byte descriptor: `dfn_res_stage` and
`dfn_rate_bridge`. Everything else here is example code (header-only or `main.c`, no ABI, copy and
adapt).

| File | What it is |
|---|---|
| `dfn_res_stage.{h,c}` | The 48 kHz post-RES DFN2 stage. The network estimates from E (the linear-AEC error) and is applied to P (E after the RES gain, plus comfort noise when enabled). `DFN_RES_STAGE_LAYOUT_VERSION`. |
| `dfn_rate_bridge.{h,c}` | The rate bridge for hosts below 48 kHz: native WOLA, resampling to the stage's 48 kHz 1024/512 grid and back; adds a fixed delay. `DFN_RATE_BRIDGE_LAYOUT_VERSION`. |
| `dfn_hop.h` | Example: one DFN2 hop (`DfnHop`) over the stage, with a caller-owned synthesis, or over the bridge; the phased begin / model / end calls. |
| `dfn_hop_pool.h` | Example: sizes and carves the DFN tail of an example's pool (the stage or the bridge, its synthesis and the output spectrum). |
| `dfn_example_args.h` | Example: the command line shared by the two board skeletons (`--sample-rate`, `--erb-fwd`, `--erb-inv`, `--atten-lim`, `--cng`); the delay options come from `common/example_cli.h`. |
| `dfn_unit_partition.h` | The unit ERB partition used when no ERB matrices are given (fail-open smoke runs only). |
| `dfn2_stage.py`, `dfn2_rate_adapter.py` | Python twins of the stage and the rate bridge (`pipelines.dfn.dfn2_stage`, `pipelines.dfn.dfn2_rate_adapter`). The stage takes its lookahead pair from the model it is built with. |
| `tests/` | C: `test_dfn_res_stage.c`, `test_dfn_rate_bridge.c`, `test_dfn_hop.c` and the shared `dfn_test_fixture.h`. Python: `test_dfn2_stage.py`, `test_dfn2_rate_adapter.py` and the C-parity tests `test_dfn2_stage_c_parity.py`, `test_dfn2_rate_bridge_c_parity.py`; `dfn2_pairs.py` lists the lookahead pairs they run. |
| `mono_aec_dfn_res/` | Mono example: lib/aec's `Aec` (linear AEC and its RES gain) -> DFN2. `pipeline.py` is its Python reference (`python -m pipelines.dfn.mono_aec_dfn_res.pipeline`); tests: `tests/test_mono_dfn.c`, `tests/test_dfn2_pipeline_contract.py`. |
| `4ch_aec_bf_dfn_res/` | 4-channel example: the mcab flow and the sample SRP-PHAT/BF helper, a post-beam `AecRes`, then DFN2. `pipeline.py` is its Python reference; tests: `tests/test_4ch_dfn.c`, `tests/test_dfn2_4ch_contract.py`. |

## How the examples are called

| Header | Sequence (app-side order) | Return values |
|---|---|---|
| `dfn_hop.h` | `dfn_hop_init_stage()` (48 kHz, caller-owned `AcIstft`) or `dfn_hop_init_bridge()`; per hop `dfn_hop_begin(E, P)` -> `while (dfn_hop_nn_begin() == 1) dfn_hop_nn_end(rc)` -> `dfn_hop_end(out)`; `dfn_hop_reset()` on a refusal | `DFN_HOP_INVALID` (-1), `DFN_HOP_SEQUENCE_ERROR` (-2) |
| `dfn_hop_pool.h` | `dfn_hop_pool_plan()` -> `dfn_hop_pool_bytes()` -> `dfn_hop_pool_build()` | 0, or -1 for a configuration no stage accepts |

The host forms E and P itself before the hop begins; the two example mains show it (`rig_process()`
in `mono_aec_dfn_res/main.c`, `dfn_form()` in `4ch_aec_bf_dfn_res/main.c`). External parameters are
the arguments shown (`atten_lim_db`, the ERB matrices, the model callback and its I/O descriptor,
the host rate); the stage constants (`DFN_RES_STAGE_IN_SCALE` / `_OUT_SCALE`, the bridge FIFO sizes)
are internal.

## Build and test

The two programs are built by different Makefiles, because the 4-channel one links the mcab flow:

```sh
make -C pipelines WERROR=1 libaudio_pipeline_dfn.a mono_aec_dfn_res       # mono example
make -C pipelines/4ch_aec_bf_nr_res WERROR=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res   # 4-channel example
make -C pipelines test                                                    # stage, bridge, DfnHop, mono example
make -C pipelines/4ch_aec_bf_nr_res test                                  # includes test_4ch_dfn
```

`mono_aec_dfn_res/Makefile` and `4ch_aec_bf_dfn_res/Makefile` are thin forwarders:
`make -C pipelines/dfn/mono_aec_dfn_res test` and `make -C pipelines/dfn/4ch_aec_bf_dfn_res test` build
and run that example's tests and the board skeleton with its argument checks. Malformed numbers on the
command line (`16000Hz`, `12dB`) exit with 2.

## Lookahead pairs

DeepFilterNet2 has two independent lookahead settings, `mask_lookahead` M (0..2) and `df_lookahead` D
(0..4, `DF_ORDER` - 1): 15 legal pairs, no ordering rule between M and D. The pair is part of the model
contract (a checkpoint belongs to one pair); the exported graph does not depend on it. The rules and the
export side are in `AINR/README.md`, "Lookahead pairs". Here the pair is a build setting of the C units
and a model property of the Python twins; the shipped pair is M = D = 1, the only one with a trained
checkpoint. The audio quality of the other pairs has not been measured.

| Item | Rule (L = M + D) |
|---|---|
| Stage output | `DFN_RES_STAGE_LOOKAHEAD_FRAMES` = `DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD` = L (cascade: the two add). The spectrum emitted on hop t belongs to source frame t - L; the first L hops emit an all-zero spectrum. `dfn_res_stage_begin()` returns 0 (no model frame) only on the first M hops; `dfn_res_stage_end()` returns 0 until the first emitted frame. |
| 48 kHz host | Algorithmic latency (L + 1) x 512 samples: the shipped pair 1536 samples (32 ms); the example prints it. |
| Rate bridge (8 / 16 kHz hosts) | Added delay = prefill + 16 + (L + 1) x 512 x sr / 48000 + 16 native samples, the product divided in integers (`dfn_rate_bridge_added_delay_samples()`); prefill is 128 (16 kHz, hop 128), 85 (16 kHz, hop 256), 42 (8 kHz, hop 128). Shipped pair: 672 / 629 / 330. The stage delay is a whole number of native samples only when L + 1 is a multiple of 3 (L = 2 or 5); otherwise it is floored, and the output trails the input by a further fraction of a sample (an impulse peak may land one sample after the added delay). Pair 0/1: 501 / 458 / 244; pair 2/4: 1354 / 1311 / 671. |
| Test gates | The pass-band error gates (5e-3 in the bridge test, 1e-3 in the two example tests) apply only when the delay is a whole number of native samples; for the other pairs the tests pin the impulse position (delay or delay + 1), the frame schedule, the fail-open identity and the counters. |
| Build | `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` make variables (default 1, one decimal in range, otherwise a Makefile error), forwarded as `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` to every unit when the pair is not 1/1, and part of the build signature (a pair gets its own `obj` / `bin` directories). `DFN3_MASK_LOOKAHEAD` / `DFN3_DF_LOOKAHEAD` exist in `pipelines/Makefile` the same way (the DFN3 sources are compiled there for the target-ABI test only). Objects of different pairs fail to link (symbol `dfn2_lookahead_mask<M>_df<D>`); an out-of-range pair is a compile error. |
| Python twins | `Dfn2Stage` takes M and D from the loaded model (`config.ini [model]`; the model constructor refuses an illegal pair and a checkpoint of another pair fails the checkpoint contract check); `Dfn2RateBridge.added_delay` uses the same formula as C. |

```sh
make -C pipelines WERROR=1 DFN2_MASK_LOOKAHEAD=0 DFN2_DF_LOOKAHEAD=1 libaudio_pipeline_dfn.a mono_aec_dfn_res
make -C pipelines/4ch_aec_bf_nr_res WERROR=1 DFN2_MASK_LOOKAHEAD=0 DFN2_DF_LOOKAHEAD=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res
make -C pipelines test-pairs                              # default DFN_TEST_PAIRS="0/0 0/1 2/0 2/4"
make -C pipelines test-pairs DFN_TEST_PAIRS="0/1 2/4"     # mask/df pairs from 0..2/0..4
```

`make test-pairs` rebuilds `libaudio_pipeline_dfn.a`, `test_dfn_res_stage`, `test_dfn_rate_bridge`,
`test_dfn_hop`, `test_mono_dfn`, `test_readme_flow` and `mono_aec_dfn_res` for each listed pair, checks that the
archive carries that pair's symbol, and runs them; `make test` is the shipped pair. The 4-channel programs take
the same two variables in their own Makefile and have no `test-pairs` target. The Python stage test runs the shipped
pair and (0,0), (0,1), (2,0), (2,4); the C-parity tests rebuild their shims for the shipped pair, (0,1) and (2,4)
(`tests/dfn2_pairs.py`). Only a subset of the 15 pairs has been built and run in the C programs; the others are not claimed as tested.

Board notes. Pass the same `-DDFN2_MASK_LOOKAHEAD` / `-DDFN2_DF_LOOKAHEAD` flags to every DFN2 unit
(`dfn2_process.c`, `dfn2_model_io.c`, `dfn2_prepost.c`, `dfn_res_stage.c`, `dfn_rate_bridge.c` and any unit that
includes their headers) when the pair is not 1/1, and install a model exported for that pair. A board build
that uses the shipped pair needs no flag. Board samples outside this repository that fix the pair
(`alg_ai3_dfn.c`, `uvc_ai_nr_stream.c`: MASK and DF are 1) refuse any other pair with a static assert. A board
build lists no `fix_gain.c` and no `third_party/gain` include path.

## Dependencies

- `common/`: `example_cli.h` (the strict command-line parsers and delay options) and, for the
  4-channel example, `4ch_aec_bf_nr_res/mcab_sample_post.h` (`mcab_sample_res_input()` and the
  comfort-noise constants). The mono example takes the grid and the lane config straight from lib/aec
  (`aec_resolve_signal_grid()`, `aec_config_from_preset()`).
- lib/aec (`Aec`, `AecRes`), audiolib `audio_common` (resamplers, STFT/ISTFT) and
  `AINR/DeepFilterNet2` (pre/post class, model I/O descriptor, ERB matrices).
- The 4-channel example also needs the mcab flow in `4ch_aec_bf_nr_res/` (`mcab.h`, archive
  `libaudio_pipeline_mcab.a` with `mcab.o` only) and its sample objects `mcab_sample_flow`,
  `mcab_sample_beam` and `mcab_sample_driver`, which are linked into the program and are not part of
  the archive.

Includes across directories are written from the `pipelines/` root (`"dfn/dfn_hop.h"`,
`"common/example_cli.h"`, `"4ch_aec_bf_nr_res/mcab.h"`); inside this folder they use the plain file
name. The Python twins are `dfn2_stage.py` and `dfn2_rate_adapter.py` here, `mono_aec_dfn_res/pipeline.py`,
`4ch_aec_bf_dfn_res/pipeline.py` and `common/cng_aec3.py`; their C parity tests are
`dfn/tests/test_dfn2_stage_c_parity.py` and `dfn/tests/test_dfn2_rate_bridge_c_parity.py`.
