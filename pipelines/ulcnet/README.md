# pipelines/ulcnet

The Align-ULCNet post-filter stage behind a linear AEC and the two example programs that compose it.
All of it is example code: no ABI, no layout version, no descriptor, in no archive. Copy a header or
a source file into your project and change it. The model's C pre/post code and the accelerator
adapter live in `AIAEC/Align_ULCNet/` and are linked into the programs.

| File | What it is |
|---|---|
| `ulcnet_nn_stage.{h,c}` | The post-filter stage: far-end STFT, model call policy (identity reprime after an alignment event, fail-open), WOLA. |
| `tests/test_ulcnet_nn_stage.c` | The stage on its own (framing, model policy, WOLA) on the compiled grid. |
| `mono_alignulcnet/` | Mono example: lib/aec's `Aec` (linear stage only) -> the stage. `tests/test_mono_ulcnet.c`. |
| `4ch_alignulcnet/` | 4-channel example: the mcab flow (`enable_post = 0`) and the sample SRP-PHAT/BF helper -> the stage. `tests/test_4ch_ulcnet.c`. |

## How the examples are called

| Header | Sequence (app-side order) | Return values |
|---|---|---|
| `ulcnet_nn_stage.h` | `ulcnet_nn_stage_mem_size()` -> `_init()` -> `_set_model()`; per hop `_begin()` (re/im arrays), `_begin_time()` or `_begin_complex()` -> `_nn_begin()` -> `_nn_end(rc)` -> `_end(out)`; `_reset()` on a refusal | `nn_begin` 1 = a frame to run, 0 = none (the reprime frame after an alignment event, or a frame already fetched) |

```c
size_t bytes = ulcnet_nn_stage_mem_size(0);          /* 1: also an STFT for begin_time() */
UlcnetNnStage* stage = ulcnet_nn_stage_init(pool, bytes, 0);
ulcnet_nn_stage_set_model(stage, &model);            /* UlcnetModel; identity when unset */

/* per hop: err = the linear-error spectrum (the helper's beam.spectrum, or the STFT of a mono
 * lane's formed hop), far = the same call's raw far-end samples */
UlcnetNnDelay delay = { info.delay.changed, info.delay.solid };
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay);
if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)
    ulcnet_nn_stage_nn_end(stage, run_network(&job));
ulcnet_nn_stage_end(stage, out);                      /* hop #0 is zeros */
```

A mono lane maps `AecLinearContext.delay_state` to `{ state == CHANGED, state != UNLOCKED }` and
passes `formed_hop` to `ulcnet_nn_stage_begin_time()` on a stage built with an error analysis. One
build serves one grid: `ULCNET_GRID=16k` (16 kHz / fft 512, default) or `48k` (48 kHz / fft 1024).
External parameters are the arguments shown (the delay profile flags of the mains, the grid, `D` of
the exported graph); `ULCNET_NN_STAGE_REPRIME_FRAMES` (1 frame) is a constant of the stage, not a knob.

## Build and test

The mono program is built by `pipelines/Makefile`, the 4-channel one by `4ch_aec_bf_nr_res/Makefile`
(it links the mcab flow):

```sh
make -C pipelines mono_alignulcnet test_mono_ulcnet test_ulcnet_nn_stage      # mono example, stage test
make -C pipelines/4ch_aec_bf_nr_res 4ch_alignulcnet test_4ch_ulcnet           # 4-channel example
make -C pipelines ULCNET_GRID=48k test                                        # the other grid
```

`mono_alignulcnet/Makefile` and `4ch_alignulcnet/Makefile` are thin forwarders:
`make -C pipelines/ulcnet/mono_alignulcnet test` and `make -C pipelines/ulcnet/4ch_alignulcnet test`
run that example's tests and the board skeleton at n=5 and n=1, and check that malformed numbers
(`--delay-num-filters 3x`, `--fixed-delay 12x`) exit with 2. Both parse numbers with
`common/example_cli.h` (`example_cli_parse_int`).

## Dependencies

- `common/`: `example_cli.h` (the strict command-line parsers and delay options). The mono
  example takes its lane config straight from lib/aec (`aec_config_from_preset()`).
- lib/aec (`Aec`), audiolib `audio_common` (STFT/ISTFT) and `AIAEC/Align_ULCNet`
  (`ulcnet_process`, `ulcnet_model_io`, `ulcnet_prepost`, `ulcnet_accelerator_adapter`).
- The 4-channel example also needs the mcab flow in `4ch_aec_bf_nr_res/` (`mcab.h`, archive
  `libaudio_pipeline_mcab.a` with `mcab.o` only) and its sample objects `mcab_sample_flow`,
  `mcab_sample_beam` and `mcab_sample_driver`, which are linked into the program and are not part of
  the archive.

Includes across directories are written from the `pipelines/` root (`"ulcnet/ulcnet_nn_stage.h"`,
`"common/example_cli.h"`, `"4ch_aec_bf_nr_res/mcab.h"`).
