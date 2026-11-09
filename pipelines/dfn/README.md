# pipelines/dfn

The DeepFilterNet2 (DFN2) post-RES stage for the AEC pipelines and the two example programs that
build on it. Two modules carry a layout version and a 32-byte descriptor: `dfn_res_stage` and
`dfn_rate_bridge`. Everything else here is example code (header-only or `main.c`, no ABI, copy and
adapt).

| File | What it is |
|---|---|
| `dfn_res_stage.{h,c}` | The post-RES DFN2 stage on the model grid its config carries (`DfnResStageConfig.grid`, built by the application with `dfn2_grid_make()`; 48 kHz / 1024 as shipped). No grid, pair or preset is compiled into the unit: the same code runs 16, 32 and 48 kHz grids, and `dfn_res_stage_default_config(const DFN2Grid *)` copies the grid in. The network estimates from E (the linear-AEC error) and is applied to P (E after the RES gain, plus comfort noise when enabled). The stage registers no recurrent state; `dfn_res_stage_state_generation()` tells a board when to reload quantized zeros. `DFN_RES_STAGE_LAYOUT_VERSION`. |
| `dfn_rate_bridge.{h,c}` | The rate bridge for hosts below 48 kHz: native WOLA, resampling to the stage's 48 kHz 1024/512 grid and back; adds a fixed delay. `dfn_rate_bridge_default_config(sample_rate, model_grid)`. Written for the shipped 48 kHz / 1024 model only: a stage config on any other grid makes it refuse (-1) at run time. `dfn_rate_bridge_state_generation()` forwards the stage's. `DFN_RATE_BRIDGE_LAYOUT_VERSION`. |
| `dfn_hop.h` | Example: one DFN2 hop (`DfnHop`) over the stage, with a caller-owned synthesis, or over the bridge; the phased begin / model / end calls. |
| `dfn_hop_pool.h` | Example: sizes and carves the DFN tail of an example's pool (the stage or the bridge, its synthesis and the output spectrum). |
| `dfn_example_args.h` | Example: the command line shared by the two board skeletons (`--sample-rate` the host rate, `--ai-sample-rate` / `--ai-fft-size` the DFN2 model grid, `--erb-fwd` / `--erb-inv` the two ERB matrix files, `--atten-lim`, `--cng`); the delay options come from `common/example_cli.h`. `dfn_example_load_matrices()` reads the two files (see "Build and test") or builds the unit partition. The (sample rate, FFT size) -> (n_erb, df_bins, lookahead pair) table is the example's own, two rows (16000/256 -> 24, 96, 0/0; 48000/1024 -> 32, 96, 1/1; no 32 kHz row, the AEC in front of the stage has none), default 48000/1024; an unknown pair is refused with the list (exit 2). |
| `dfn_unit_partition.h` | The unit ERB partition used when no ERB matrices are given (fail-open smoke runs only); takes the grid. |
| `dfn2_stage.py`, `dfn2_rate_adapter.py` | Python twins of the stage and the rate bridge (`pipelines.dfn.dfn2_stage`, `pipelines.dfn.dfn2_rate_adapter`). The stage takes its lookahead pair from the model it is built with. |
| `tests/` | C: `test_dfn_res_stage.c`, `test_dfn_rate_bridge.c`, `test_dfn_hop.c`, the shared `dfn_test_fixture.h` and `dfn_test_grid.h`, the only build-time grid selector left (`-DDFN2_GRID_KHZ=16\|32\|48`, `-DDFN2_SR` / `-DDFN2_N_FFT` / `-DDFN2_N_ERB` / `-DDFN2_DF_BINS`, `-DDFN2_MASK_LOOKAHEAD` / `-DDFN2_DF_LOOKAHEAD`): test programs size their arrays with its `DFN2_*` macros and hand `dfn_test_grid()` to the stage and the bridge; product code never includes it. `test_dfn_test_grid_pin.py` compares its model numbers with `AINR/DeepFilterNet2/config.ini`. `check_pair_report.sh` is the `make test-pairs` gate. Python: `test_dfn2_stage.py`, `test_dfn2_rate_adapter.py` and the C-parity tests `test_dfn2_stage_c_parity.py`, `test_dfn2_rate_bridge_c_parity.py`; `dfn2_pairs.py` lists the lookahead pairs they run. |
| `mono_aec_dfn_res/` | Mono example: lib/aec's `Aec` (linear AEC and its RES gain) -> DFN2. `pipeline.py` is its Python reference (`python -m pipelines.dfn.mono_aec_dfn_res.pipeline`); tests: `tests/test_mono_dfn.c`, `tests/test_dfn2_pipeline_contract.py`. |
| `4ch_aec_bf_dfn_res/` | 4-channel example: the mcab flow and the sample SRP-PHAT/BF helper, a post-beam `AecRes`, then DFN2. `pipeline.py` is its Python reference; tests: `tests/test_4ch_dfn.c`, `tests/test_dfn2_4ch_contract.py`. |

## How the examples are called

| Header | Sequence (app-side order) | Return values |
|---|---|---|
| `dfn_hop.h` | `dfn_hop_init_stage()` (host on the stage's model grid, 48 kHz as shipped; caller-owned `AcIstft`) or `dfn_hop_init_bridge()`; per hop `dfn_hop_begin(E, P)` -> `while (dfn_hop_nn_begin() == 1) dfn_hop_nn_end(rc)` -> `dfn_hop_end(out)`; `dfn_hop_reset()` on a refusal | `DFN_HOP_INVALID` (-1), `DFN_HOP_SEQUENCE_ERROR` (-2) |
| `dfn_hop_pool.h` | `dfn_hop_pool_plan()` -> `dfn_hop_pool_bytes()` -> `dfn_hop_pool_build()` | 0, or -1 for a configuration no stage accepts |

The host forms E and P itself before the hop begins; the two example mains show it (`rig_process()`
in `mono_aec_dfn_res/main.c`, `dfn_form()` in `4ch_aec_bf_dfn_res/main.c`). External parameters are
the arguments shown (`atten_lim_db`, the ERB matrices, the model callback and its I/O descriptor,
the host rate, the model grid in `DfnResStageConfig.grid`); the stage constants (the in and out scales, the
`dfn2_stft_scale()` / `dfn2_stft_unscale()` -- 2^-5 / 2^5 on the shipped 48 kHz grid -- and the bridge FIFO sizes) are internal.
The stage and the bridge take audio-scale spectra and apply those two scales themselves, so the caller's `AcStft` / `AcIstft`
keep audiolib's default `spec_scale` / `ispec_scale` of 1. They register no recurrent state: a float model callback owns it, a
board's accelerator holds it and reloads quantized zeros at start and whenever `dfn_res_stage_state_generation()` (through the
bridge, `dfn_rate_bridge_state_generation()`) changes. A reset, each warm-up frame and a refused commit (a non-finite head) advance it,
a nonzero model result does not. Every hop runs the same begin, model, end flow: the first M hops (`grid.mask_lookahead`) are warm-up
frames whose heads are discarded and whose state is zero again afterwards, so the audio is that of a model started at its first real frame.

## Build and test

The two programs are built by different Makefiles, because the 4-channel one links the mcab flow:

```sh
make -C pipelines WERROR=1 libaudio_pipeline_dfn.a mono_aec_dfn_res       # mono example
make -C pipelines/4ch_aec_bf_nr_res WERROR=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res   # 4-channel example
make -C pipelines test                                                    # stage, bridge, DfnHop, mono example
make -C pipelines/4ch_aec_bf_nr_res test                                  # includes test_4ch_dfn
make -C pipelines test-dfn-grids                                          # stage, bridge, DfnHop tests built for the 16 and 32 kHz test grids
```

The stage, the bridge and the two mains take their DFN2 grid at run time, so one build serves every grid
the library accepts (`dfn2_grid_make()`). `DFN2_TEST_GRID_KHZ=16|32` (default 48) and `DFN2_MASK_LOOKAHEAD` /
`DFN2_DF_LOOKAHEAD` reach the test programs only (`DFN_TEST_CFLAGS` in the Makefile), through
`tests/dfn_test_grid.h`. The two mains choose the model grid with `--ai-sample-rate` / `--ai-fft-size`
(host on the model's own rate: the stage directly; any other host rate: the bridge, 48 kHz / 1024 model only).
`make test-dfn-grids` runs the stage, bridge and hop tests at the 16 and 32 kHz test grids with the test header's default pair 1/1,
not with the 0/0 of those rows in the example's table: it shows that the same stage code runs another grid (finite output, latency
equal to the grid's lookahead, the bridge refusing it), not that a table row works.

The stage config takes the ERB matrices as two borrowed pointers (`erb_fwd`, `erb_inv`; the inverse a partition of unity) and reads no file; the application loads and owns them and keeps them alive while the stage or the bridge runs. The two mains
read them from two raw little-endian float32 files, `--erb-fwd FILE` and `--erb-inv FILE` (both or neither): the
`--runtime-bins` pair of `AINR/DeepFilterNet2/export_erb_matrix.py`, `erb_fwd.bin` bin-major `[n_bins][n_erb]` and `erb_inv.bin` band-major `[n_erb][n_bins]`, each exactly
`n_bins * n_erb * 4` bytes (65,664 B for the 48 kHz / 1024 / 32-band model). A file that cannot be opened, or that holds fewer or more values than the
model grid needs, is refused with a message (exit 1); a missing value, or one of the two options without the other, exits 2. Without the options a
unit partition serves the fail-open smoke run, and the main prints a notice. The standalone board samples live outside this repo and
read one file instead, `fb_<sr>_<n_erb>.bin` (the exporter's `erb_matrix.bin` renamed for its grid: int32 `n_bins`, int32 `n_bands`,
forward, inverse, both bin-major); they transpose the inverse to band-major themselves at init, see
`AINR/DeepFilterNet2/README.md`, "Choosing the grid".

`mono_aec_dfn_res/Makefile` and `4ch_aec_bf_dfn_res/Makefile` are thin forwarders:
`make -C pipelines/dfn/mono_aec_dfn_res test` and `make -C pipelines/dfn/4ch_aec_bf_dfn_res test` build
and run that example's tests and the board skeleton with its argument checks. Malformed numbers on the
command line (`16000Hz`, `12dB`) exit with 2.

## Lookahead pairs

DeepFilterNet2 has two independent lookahead settings, `mask_lookahead` M (0..2) and `df_lookahead` D
(0..4, `df_order` - 1): 15 legal pairs, no ordering rule between M and D. The pair is part of the model
contract (a checkpoint belongs to one pair); the exported graph does not depend on it. The rules and the
export side are in `AINR/README.md`, "Lookahead pairs". Here the pair is a field of the run-time `DFN2Grid` the application gives the C units (a row of the
example's table; the Makefile pair variables select only what the test programs run) and a model
property of the Python twins; the shipped pair is M = D = 1, the only one with a trained
checkpoint. The audio quality of the other pairs has not been measured.

| Item | Rule (L = M + D) |
|---|---|
| Stage output | `dfn_res_stage_lookahead_frames(grid)` = `grid.mask_lookahead + grid.df_lookahead` = L (cascade: the two add). The spectrum emitted on hop t belongs to source frame t - L; the first L hops emit an all-zero spectrum. `dfn_res_stage_begin()` returns 1 on every hop (the first M are warm-up frames: the model runs, its heads are discarded); `dfn_res_stage_end()` returns 0 until the first emitted frame. |
| 48 kHz host | Algorithmic latency (L + 1) x 512 samples: the shipped pair 1536 samples (32 ms); the example prints it. |
| Rate bridge (8 / 16 kHz hosts) | Added delay = prefill + 16 + (L + 1) x 512 x sr / 48000 + 16 native samples; prefill is 128 (16 kHz, hop 128), 85 (16 kHz, hop 256), 42 (8 kHz, hop 128). `dfn_rate_bridge_added_delay_samples()` returns it floored. The stage term is a whole number of native samples only when L + 1 is a multiple of 3 (L = 2 or 5); every other term is whole, so for the other pairs the exact delay is the floored one plus 1/3 or 2/3 of a sample. `dfn_rate_bridge_added_delay_exact()` returns it without the floor as `numerator / denominator` (denominator `DFN_RATE_BRIDGE_DELAY_DENOMINATOR` = 6, not reduced; numerator = (prefill + 16) x 6 + ((L + 1) x 512 + down-resampler latency, 48 or 96 frames at 48 kHz) x sr / 8000, an exact integer division); `Dfn2RateBridge.added_delay_exact` is the same pair. Shipped pair (floored = exact): 672 / 629 / 330. Pair 0/1: 501 1/3 / 458 1/3 / 244 2/3; pair 2/4: 1354 2/3 / 1311 2/3 / 671 1/3. |
| Test gates | The bridge test and the two example tests compare against the exact delay for every pair: the impulse response is a symmetric band-limited kernel, so its centroid must sit within 0.01 sample of `hop + exact delay` (bridge test; the example tests pin the peak sample to the nearest one), and the pass-band error gates (5e-3 in the bridge test against the tones evaluated at the fractional time, 1e-3 in the two example tests against the conventional output delayed by a windowed-sinc fractional delay) hold for the shipped and the non-integer pairs alike. |
| Build | `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` make variables (default 1, one decimal in range, otherwise a Makefile error). The pipeline units hold no pair, so the variables select what the test programs run: `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` reach the test units only (`DFN_TEST_CFLAGS`) when the pair is not 1/1, and the pair is part of the build signature (a pair gets its own `obj` / `bin` directories). The library has no pair symbol: a model exported for another pair is refused by its I/O descriptor and a pool of another pair by its hash, both at run time; `dfn2_grid_make()` refuses an out-of-range pair. |
| Frames per call | The DFN stage and the rate bridge are one-frame units: a grid with `frames_per_call` other than 1 is refused (-1) (the export option `--frames-per-call N` and the N-frame C host belong to `AINR/DeepFilterNet2`, see its README); `dfn2_stage.py` has no N-frame twin. |
| Python twins | `Dfn2Stage` takes M and D from the loaded model (`config.ini [model]`; the model constructor refuses an illegal pair and a checkpoint of another pair fails the checkpoint contract check); `Dfn2RateBridge.added_delay` and `added_delay_exact` use the same formulas as C. |

```sh
make -C pipelines WERROR=1 DFN2_MASK_LOOKAHEAD=0 DFN2_DF_LOOKAHEAD=1 test_dfn_res_stage test_mono_dfn
make -C pipelines/4ch_aec_bf_nr_res WERROR=1 DFN2_MASK_LOOKAHEAD=0 DFN2_DF_LOOKAHEAD=1 test_4ch_dfn
make -C pipelines test-pairs                              # default DFN_TEST_PAIRS="0/0 0/1 2/0 2/4"
make -C pipelines test-pairs DFN_TEST_PAIRS="0/1 2/4"     # mask/df pairs from 0..2/0..4
```

`make test-pairs` rebuilds `libaudio_pipeline_dfn.a`, `test_dfn_res_stage`, `test_dfn_rate_bridge`,
`test_dfn_hop`, `test_mono_dfn`, `test_readme_flow` and `mono_aec_dfn_res` for each listed pair and runs them; `make test` is the shipped pair.
Each pair test prints the pair it was built with (`<program>: lookahead pair M/D`) and, for the stage test and the mono
composition, the stage latency it measured (the first emitted hop of the stage; the shift against the hand-written chain);
`tests/check_pair_report.sh` compares them with the pair asked for and with `(M + D + 1) * hop`, so a pair flag that never
reached a test build (it would print 1/1) is a red gate. The 4-channel programs take
the same two variables in their own Makefile and have no `test-pairs` target. The Python stage test runs the shipped
pair and (0,0), (0,1), (2,0), (2,4); the C-parity tests rebuild their shims for the shipped pair, (0,1) and (2,4)
(`tests/dfn2_pairs.py`). Only a subset of the 15 pairs has been built and run in the C programs; the others are not claimed as tested.

Board notes. The pipeline units (`dfn_res_stage.c`, `dfn_rate_bridge.c`) need no grid or pair flag: the application
builds the `DFN2Grid` with `dfn2_grid_make()` (the pair is one of its fields; install a model exported for that pair) and puts it in
`DfnResStageConfig.grid`. A standalone
DFN2 board sample does not go through this folder: it picks the grid and the pair from its own (sample rate, FFT size) table, see
`AINR/DeepFilterNet2/README.md`, "Choosing the grid". A board build lists no `fix_gain.c` and no `third_party/gain` include path.

## Dependencies

- `common/`: `example_cli.h` (the strict command-line parsers and delay options) and, for the
  4-channel example, `4ch_aec_bf_nr_res/mcab_sample_post.h` (`mcab_sample_res_input()` and the
  comfort-noise constants). The mono example takes the grid and the lane config straight from lib/aec
  (`aec_resolve_signal_grid()`, `aec_config_from_level()`).
- lib/aec (`Aec`, `AecRes`), audiolib `audio_common` (resamplers, STFT/ISTFT) and
  `AINR/DeepFilterNet2` (grid, pre/post class, model I/O descriptor; the ERB matrices are the application's data).
- The 4-channel example also needs the mcab flow in `4ch_aec_bf_nr_res/` (`mcab.h`, archive
  `libaudio_pipeline_mcab.a` with `mcab.o` only) and its sample objects `mcab_sample_flow`,
  `mcab_sample_beam` and `mcab_sample_driver`, which are linked into the program and are not part of
  the archive.

Includes across directories are written from the `pipelines/` root (`"dfn/dfn_hop.h"`,
`"common/example_cli.h"`, `"4ch_aec_bf_nr_res/mcab.h"`); inside this folder they use the plain file
name. The Python twins are `dfn2_stage.py` and `dfn2_rate_adapter.py` here, `mono_aec_dfn_res/pipeline.py`,
`4ch_aec_bf_dfn_res/pipeline.py` and `common/cng_aec3.py`; their C parity tests are
`dfn/tests/test_dfn2_stage_c_parity.py` and `dfn/tests/test_dfn2_rate_bridge_c_parity.py`.
