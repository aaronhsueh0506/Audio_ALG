# pipelines/ulcnet

The Align-ULCNet post-filter stage behind a linear AEC and the two example programs that build on it.
All of it is example code: no ABI, no layout version, no descriptor, in no archive. Copy a header or
a source file into your project and change it. The model's C pre/post code and the accelerator
adapter live in `lib/aiaec/ulcnet/` and are linked into the programs. The K/V, logit and GRU state
belongs to the application: the examples register none (the accelerator holds it), and a board reloads
quantized zeros when `ulcnet_accelerator_adapter_state_generation()` changes.

| File | What it is |
|---|---|
| `ulcnet_nn_stage.{h,c}` | The post-filter stage: spectra in, one spectrum out; model call policy (identity reprime after an alignment event, fail-open). No STFT and no ISTFT: the application's `AcStft` / `AcIstft` do the framing. |
| `tests/test_ulcnet_nn_stage.c` | The stage on its own, wrapped in an application-owned `AcStft` / `AcIstft` (model policy, WOLA against independent references, grid and pool plan), on every grid of its table. |
| `mono/alignulcnet/` (sibling, not under this folder) | Mono example: lib/aec's `Aec` (linear stage only) -> the stage. `tests/test_mono_ulcnet.c`. |
| `4ch/alignulcnet/` (sibling, not under this folder) | 4-channel example: the four-microphone array (`residual = AEC_RESIDUAL_STATE`) and the sample SRP-PHAT/BF helper -> the stage. `tests/test_4ch_ulcnet.c`. |

## How the examples are called

| Header | Sequence (app-side order) | Return values |
|---|---|---|
| `ulcnet_nn_stage.h` | `ulcnet_grid_make()` -> `ulcnet_nn_stage_mem_size()` -> `_init()` -> `_set_model()`; per hop the application's `ac_stft_push()` of the raw far hop (and of the error hop when the error is a time-domain signal) -> `_begin_spec(err_spec, far_spec, delay)` -> `_nn_begin()` -> `_nn_end(rc)` -> `_end_spec(out_spec)` -> the application's `ac_istft_push()`; `_reset()` together with the application's `ac_stft_reset()` / `ac_istft_reset()` on a refusal | `nn_begin` 1 = a frame to run, 0 = none (the reprime frame after an alignment event, or a frame already fetched) |

```c
UlcnetGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = 8 };
UlcnetGrid grid;
ulcnet_grid_make(&grid, &spec);                      /* D = the export depth of the graph */
size_t bytes = ulcnet_nn_stage_mem_size(&grid);
UlcnetNnStage* stage = ulcnet_nn_stage_init(pool, bytes, &grid);
ulcnet_nn_stage_set_model(stage, &model);            /* UlcnetModel; identity when unset */

/* The application's framing: audiolib defaults on the grid, accumulate synthesis. */
AcStftConfig cfg;
ac_stft_config_default(&cfg, grid.n_fft);
cfg.ola = AC_OLA_ACCUMULATE;
AcStft*  far_stft = ac_stft_init(far_mem, ac_stft_get_mem_size(&cfg), &cfg);
AcIstft* istft    = ac_istft_init(syn_mem, ac_istft_get_mem_size(&cfg), &cfg);
/* + a second AcStft for a time-domain error hop (the mono lane's formed hop) */

/* per hop: err_spec = the linear-error spectrum (the helper's beam.spectrum, or the STFT of a mono
 * lane's formed hop), far = the same call's raw far-end samples */
UlcnetNnDelay delay = { info.delay.changed, info.delay.solid };
ac_stft_push(far_stft, far, far_spec);
ulcnet_nn_stage_begin_spec(stage, beam.spectrum, far_spec, delay);
if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)
    ulcnet_nn_stage_nn_end(stage, run_network(&job));
ulcnet_nn_stage_end_spec(stage, out_spec);            /* the accepted model spectrum, else err_spec */
ac_istft_push(istft, out_spec, out);                  /* a full hop every call, hop #0 included */
```

A mono lane maps `AecLinearView.delay_state` to `{ state == CHANGED, state != UNLOCKED }` and
analyses `formed_hop` with its own `AcStft` to get `err_spec`. The stage keeps no transform history:
on a refusal reset it together with the application's `AcStft` / `AcIstft` objects (and the AEC unit).
The total latency is one hop, as when the stage owned the transforms. The grid is a run-time
`UlcnetGrid`: the mains take `--sample-rate 16000` (fft 512, default) or `48000`
(fft 1024), and one binary serves both. External parameters are the arguments shown (the delay
profile flags and the sample rate of the mains, `D` of the exported graph);
`ULCNET_NN_STAGE_REPRIME_FRAMES` (1 frame) is a constant of the stage, not a knob.

## Build and test

The mono program is built by `pipelines/Makefile`, the 4-channel one by `4ch/Makefile`
(it links the array sample objects):

```sh
make -C pipelines mono_alignulcnet test_mono_ulcnet test_ulcnet_nn_stage      # mono example, stage test
make -C pipelines/4ch 4ch_alignulcnet test_4ch_ulcnet           # 4-channel example
```

`test_ulcnet_nn_stage`, `test_mono_ulcnet` and `test_4ch_ulcnet` each run every grid of their table
(16 kHz / fft 512 and 48 kHz / fft 1024) from one binary.

`mono/alignulcnet/Makefile` and `4ch/alignulcnet/Makefile` are thin forwarders:
`make -C pipelines/mono/alignulcnet test` and `make -C pipelines/4ch/alignulcnet test`
run that example's tests and the board skeleton at n=5 and n=1 on both grids (`--sample-rate`), and
check that malformed numbers (`--delay-num-filters 3x`, `--fixed-delay 12x`, `--sample-rate 16000x`)
and a rate that is not a shipped grid exit with 2. Both parse numbers with
`common/example_cli.h` (`example_cli_parse_int`).

## Dependencies

- `common/`: `example_cli.h` (the strict command-line parsers and delay options). The mono
  example takes its lane config straight from lib/aec (`aec_config_from_level()`).
- lib/aec (`Aec`), audiolib `audio_common` (`AcStft` / `AcIstft`, used by the callers, not by the stage) and `lib/aiaec/ulcnet`
  (`ulcnet_process`, `ulcnet_model_io`, `ulcnet_prepost`, `ulcnet_accelerator_adapter`).
- The 4-channel example also needs the array instance of lib/aec (`aec.h`, `AecConfig.n_mics = 4`, in `libaec.a`) and the sample objects
  `array_sample_beam` and `array_sample_driver` in `4ch/`, which are linked into the program and are not part of
  any archive.

Includes across directories are written from the `pipelines/` root (`"ulcnet/ulcnet_nn_stage.h"`,
`"common/example_cli.h"`, `"4ch/array_sample_beam.h"`).
