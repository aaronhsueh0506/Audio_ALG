# Mono AEC + NR/RES application

There is no mono pipeline library and no mono unit: the example composes the libraries itself.
`lib/aec`'s `Aec` runs the linear stage (`aec_process_context()`) and carries the RES gain itself
(`ctx.res_gain`), `se_frame_from_aec()` (`common/se_stage.h`) turns its context into a
`SeSpecFrame` whose `g_res_ctx` is that gain, and the example writes the rest of the per-hop
chain (the denoiser's gain through `samples/se_anr_stage.h`, `min`, apply, comfort noise,
synthesis) in `mono_form()`. Each library has its own `mem_size` and `init`; the example sums the
sizes it needs (`aec_get_mem_size()`, `anr_get_mem_size()` with `anr_config_for_mode_grid()` and
the library-default `L`, `ac_istft_get_mem_size()`, plus the gain and spectrum buffers).

The heap and caller-pool WAV runners are `main.c` and `static_main.c`;
`example_board_adapter.c` simulates the board's memory manager around the same sequence.
This directory builds no archive: the examples link `common/se_config.o` and the producer
libraries directly.

Per hop (the code is `process_hop()` and `mono_form()` in `main.c`):

```c
aec_process_context(aec, mic, ref);              /* --aec-only: aec_process(aec, mic, ref, out) */
se_frame_from_aec(aec, &frame);                  /* frame.g_res_ctx: the AEC's own RES gain */
nr = se_anr_stage_gain(anr, &frame, scratch, use_r2);       /* NULL: the denoiser refused the hop */
sk_min_f32(total_gain, nr, frame.g_res_ctx, n_freqs);      /* unity vectors stand in for a stage that is off */
sk_capply_gain_f32(spec, frame.e, total_gain, n_freqs);
aec_cng_fill(&rng, spec, frame.cn, frame.g_res_ctx, nr, SE_CNG_NR_GAIN_FLOOR, n_freqs);
ac_istft_push(istft, spec, out);
/* on any refusal: aec_reset(aec); anr_reset(anr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED; */
```

Strength: `aec_set_preset(aec, preset, ramp_ms)` for the AEC's RES gain,
`anr_set_mode(anr, mode)` for the denoiser.

## Calling sequence

```
App                      Aec                 AnrDenoiser        AcIstft
 |-- se_derive(&req,&rules,&grid,&aec_cfg)                       (derive_lane())
 |-- aec_get_mem_size / aec_init ---->|                          (rig_build())
 |-- anr_config_for_mode_grid; anr_get_mem_size / anr_init ----->|
 |-- ac_stft_config_default; ac_istft_get_mem_size / ac_istft_init ---------->|
 |   per hop: aec_process_context --> se_frame_from_aec --> se_anr_stage_gain --> sk_min_f32
 |            --> sk_capply_gain_f32 --> aec_cng_fill --> ac_istft_push --> out   (process_hop())
 |   refused hop or non-finite out: aec_reset + anr_reset + ac_istft_reset + rng = SE_CNG_RNG_SEED  (rig_reset())
 |-- aec_set_preset(...) / anr_set_mode(...) between two hops
 |-- anr_destroy; aec_destroy; release the block (AcIstft has no destroy)
```

One hop is `grid.hop_size` samples (128 at 8 kHz and at the 16 kHz default, 256 at 16 kHz/512, 512 at
48 kHz); the output trails the input by that one hop (the synthesis), `--aec-only` by none.
Per-flow detail with return values: section 0 of
[`../../docs/integration_mono_zh_TW.md`](../../docs/integration_mono_zh_TW_01_overview_link_quickstart.md#0-速覽流程時序圖公開內部函式外部內部參數).

Command line of the three programs (`main.c`, `static_main.c`; `example_board_adapter.c` takes none):

| Argument | Meaning | Default |
|---|---|---|
| fourth positional (`mild`, `balanced`, `aggressive`, `no_floor`, or `0`..`3`) | AEC preset (`AecPreset`); an unknown word falls back to `balanced` | `balanced` |
| `--nr-preset mild\|moderate\|balanced\|aggressive` | `AnrMode` | `balanced` |
| `--aec-only` | only the `Aec` (`aec_process()`), no NR, RES gain, comfort noise or synthesis | off |
| `--no-nr`, `--no-res`, `--no-cng` | leave out the denoiser, the RES gain, the comfort noise | off |
| `--legacy-amin` | the denoiser's noise floor without the residual-echo powers (`use_r2 = 0`) | off |
| `--fft-size 256\|512\|1024` | grid; 0 = the rate's default | 0 |
| `--debug`, `--timing` | once-per-second status line; per-stage cost at exit (`make PROFILE=1` for the AEC rows) | off |
| `--sample-rate <hz>` (`static_main --print-mem-size` only) | the rate of the pool query | 16000 |

Public functions an integrator copies from these mains: the ones in the sequence above and `aec_debug_status()`,
`anr_debug_status()`. Internal (not for application code): the AEC and NR headers other than
`aec.h`, `aec_res.h`, `anr.h`. External parameters: `SeRequest` (rate, FFT size, `filter_length`, delay
mode, `delay_num_filters`, `fixed_delay_samples`, AEC preset), `AnrMode`, the three switches above.
Internal: the remaining `AecConfig` and `AnrConfig` fields, including the denoiser's `L` and
`num_init_frames` (pool-sizing); classification with header lines in section 0.3 of the manual above.

The top-level `pipelines/Makefile` owns the actual build so all applications
share one backend/SIMD configuration contract:

```sh
make SIMD=0 test
make SIMD=1 test
```

The configuration selects sample rate and FFT independently. Supported
production grids are 16 kHz/256/128, 16 kHz/512/256 and 48 kHz/1024/512.
`filter_length`, `delay_mode`, `delay_num_filters` and
`fixed_delay_samples` are independent initialization controls.
