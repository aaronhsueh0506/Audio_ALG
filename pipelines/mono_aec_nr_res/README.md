# Mono AEC + NR/RES application

There is no mono pipeline library and no mono wrapper module: the example composes the libraries itself.
`lib/aec`'s `Aec` runs the linear stage (`aec_process_context()`) and carries the RES gain itself
(`ctx.res_gain`, read with `aec_get_res_context()` into an `AecResContext`), and the example writes the rest of the
per-hop chain (the denoiser's gain in `mono_nr_gain()`, then `min`, apply, comfort noise, synthesis in
`mono_form()`). Each library has its own `mem_size` and `init`; the example sums the
sizes it needs (`aec_get_mem_size()`, `anr_get_mem_size()` with `anr_config_for_level_grid()` and
the library-default `L`, `ac_istft_get_mem_size()`, plus the gain and spectrum buffers). The grid and the
`AecConfig` come from `derive_lane()` (`aec_resolve_signal_grid()`, `aec_config_from_level()`, then the
seam fields `enable_res = 0` and `return_res_context = 1`).

The WAV runner is `main.c`; the same source builds two binaries. `aec_nr_pipeline_static` places
every stage in one aligned caller pool and also answers `--print-mem-size`; `aec_nr_pipeline`
(`-DAEC_NR_POOL_SPLIT=1`) allocates the AEC in one block and the post stages (denoiser, synthesis,
gain and spectrum buffers) in a second one. `example_board_adapter.c` simulates the board's memory
manager around the same sequence. This directory builds no archive: the examples link the
producer libraries (`libaec.a`, `libanr.a`, `libaudio_common.a`) directly.

Per hop (the code is `process_hop()`, `mono_nr_gain()` and `mono_form()` in `main.c`):

```c
aec_process_context(aec, mic, ref);              /* --aec-only: aec_process(aec, mic, ref, out) */
aec_get_res_context(aec, &ctx);                  /* ctx.res_gain: the AEC's own RES gain */
/* nr: with use_r2 and ctx.usable_linear, extra = ctx.r2 / 32768^2 (sk_scale_f32), else NULL */
anr_process_gain(anr, ctx.error_spec, extra, NULL);   /* != 0: the denoiser refused the hop */
nr = anr_get_gain(anr, NULL);
sk_min_f32(total_gain, nr, ctx.res_gain, n_freqs);    /* unity vectors stand in for a stage that is off */
sk_capply_gain_f32(spec, ctx.error_spec, total_gain, n_freqs);
aec_cng_fill(&rng, spec, ctx.comfort_noise, ctx.res_gain, nr, CNG_NR_GAIN_FLOOR, n_freqs);
ac_istft_push(istft, spec, out);
/* on any refusal: aec_reset(aec); anr_reset(anr); ac_istft_reset(istft); rng = CNG_RNG_SEED; */
```

`CNG_NR_GAIN_FLOOR`, `CNG_RNG_SEED` and `PSD_SCALE_INV` are `#define`s at the top of `main.c` (and of
`example_board_adapter.c`), not library constants.

Strength: `aec_set_level(aec, level, ramp_ms)` for the AEC's RES gain,
`anr_set_level(anr, level)` for the denoiser.

## Calling sequence

```
App                      Aec                 AnrDenoiser        AcIstft
 |-- aec_resolve_signal_grid + aec_config_from_level + seam fields   (derive_lane())
 |-- aec_get_mem_size / aec_init ---->|                          (rig_build())
 |-- anr_config_for_level_grid; anr_get_mem_size / anr_init ----->|
 |-- ac_stft_config_default; ac_istft_get_mem_size / ac_istft_init ---------->|
 |   per hop: aec_process_context --> aec_get_res_context --> anr_process_gain / anr_get_gain --> sk_min_f32
 |            --> sk_capply_gain_f32 --> aec_cng_fill --> ac_istft_push --> out   (process_hop())
 |   refused hop: aec_reset + anr_reset + ac_istft_reset + rng = CNG_RNG_SEED  (rig_reset())
 |-- aec_set_level(...) / anr_set_level(...) between two hops
 |-- anr_destroy; aec_destroy; release the block (AcIstft has no destroy)
```

One hop is `grid.hop_size` samples (128 at 8 kHz and at the 16 kHz default, 256 at 16 kHz/512, 512 at
48 kHz); the output trails the input by that one hop (the synthesis), `--aec-only` by none.
Per-flow detail with return values: section 0 of
[`../../docs/integration_mono_zh_TW.md`](../../docs/integration_mono_zh_TW_01_overview_link_quickstart.md#0-速覽流程時序圖公開內部函式外部內部參數).

Command line of the two binaries built from `main.c` (`example_board_adapter.c` takes none):

| Argument | Meaning | Default |
|---|---|---|
| fourth positional (`mild`, `balanced`, `aggressive`, `no_floor`, or `0`..`3`), or `--aec-level <level>` | AEC level (`AecLevel`); an unknown word falls back to `balanced`; the positional and the option are the same setting and one AEC level may be given: a second one (positional or `--aec-level`, even an unknown word) is a command-line error, exit 2 | `balanced` |
| `--nr-level mild\|moderate\|balanced\|aggressive` (or `0`..`3`; `--nr-preset` is the same option) | `AnrLevel`; an unknown word falls back to `balanced` | `balanced` |
| `--aec-only` | only the `Aec` (`aec_process()`), no NR, RES gain, comfort noise or synthesis | off |
| `--print-mem-size [aec-level \| --aec-level <level>] [--nr-level ...] [--aec-only] [--no-nr] [--no-res]` | `aec_nr_pipeline_static` only: print the pool budget, run a real init of every stage, exit without WAV I/O | - |
| `--no-nr`, `--no-res`, `--no-cng` | leave out the denoiser, the RES gain, the comfort noise | off |
| `--legacy-amin` | the denoiser's noise floor without the residual-echo powers (`use_r2 = 0`) | off |
| `--fft-size 256\|512\|1024` | grid; 0 = the rate's default (also accepted by `--print-mem-size`) | 0 |
| `--debug`, `--timing` | once-per-second status line; per-stage cost at exit (`make PROFILE=1` for the AEC rows) | off |
| `--sample-rate <hz>` (`aec_nr_pipeline_static --print-mem-size` only) | the rate of the pool query | 16000 |

Public functions an integrator copies from this main: the ones in the sequence above and `aec_debug_status()`,
`anr_debug_status()`. Internal (not for application code): the AEC and NR headers other than
`aec.h`, `aec_res.h`, `anr.h`. External parameters: the `AecConfig` fields `sample_rate`, `fft_size`, `filter_length`, `delay_mode`,
`delay_num_filters`, `fixed_delay_samples` and the AEC level passed to `aec_config_from_level()`, `AnrLevel`, the three switches above.
Internal: the remaining `AecConfig` and `AnrConfig` fields, including the denoiser's `L` and
`num_init_frames` (pool-sizing); classification with header lines in section 0.3 of the manual above.

Python twin: `pipeline.py` (from the `Audio_ALG` root: `python -m pipelines.mono_aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav`,
presets `mild|balanced|aggressive` only; C level == Python preset). Tests: `tests/test_mono_chain.c` (`make test`), `tests/test_no_stdio_stack.c`
(`make audit-no-stdio-stack`) and `tests/test_signal_grids.py` (`python -m pytest pipelines/mono_aec_nr_res/tests` from the `Audio_ALG` root).

The top-level `pipelines/Makefile` owns the actual build so all applications
share one backend/SIMD configuration contract (this directory's `Makefile` forwards `all`, `test` and `clean` to it):

```sh
make SIMD=0 test
make SIMD=1 test
```

The configuration selects sample rate and FFT independently. Supported
production grids are 16 kHz/256/128, 16 kHz/512/256 and 48 kHz/1024/512.
`filter_length`, `delay_mode`, `delay_num_filters` and
`fixed_delay_samples` are independent initialization controls.
