# Mono AEC + NR/RES application

There is no mono pipeline library and no mono wrapper module: the example composes the libraries itself.
`lib/aec`'s `Aec` runs the linear stage (`aec_process_linear()`) and the residual stage (`aec_process_residual()`) and
carries the RES gain itself (`v.res_gain`, read with `aec_get_residual()` into an `AecResidualView`); the example
writes the rest of the per-hop chain (the denoiser's gain in `mono_nr_gain()`, then `aec_apply_gain()`,
`aec_apply_cng()` and the synthesis in `mono_form()`). The spectrum the chain renders is the AEC's *rendering base*
(`v.base_spec`), not `v.error_spec` (see "Two spectra" below). Each library has its own `mem_size` and `init`; the example sums the
sizes it needs (`aec_get_mem_size()`, `anr_get_mem_size()` with `anr_config_for_level_grid()` and
the library-default `L`, `ac_istft_get_mem_size()`, plus the gain and spectrum buffers). The grid and the
`AecConfig` come from `derive_lane()` (`aec_resolve_signal_grid()`, `aec_config_from_level()`, then
`residual = AEC_RESIDUAL_FULL`).

The WAV runner is `main.c`; the same source builds two binaries. `aec_nr_pipeline_static` places
every stage in one aligned caller pool and also answers `--print-mem-size`; `aec_nr_pipeline`
(`-DAEC_NR_POOL_SPLIT=1`) allocates the AEC in one block and the post stages (denoiser, synthesis,
gain and spectrum buffers) in a second one. `example_board_adapter.c` simulates the board's memory
manager around the same sequence. This directory builds no archive: the examples link the
producer libraries (`libaec.a`, `libanr.a`, `libaudio_common.a`) directly.

Per hop (the code is `process_hop()`, `mono_nr_gain()` and `mono_form()` in `main.c`):

```c
aec_process_linear(aec, mic, ref);               /* != AEC_OK: refused hop */
aec_process_residual(aec, NULL);
/* --aec-only (AEC_RESIDUAL_STATE): out = aec_get_linear().linear_hop, then return; no NR, gain, comfort noise or synthesis */
aec_get_residual(aec, &v);                       /* v.base_spec: the rendering base B; v.res_gain, v.r2, v.usable_linear */
if (aec_post_reset_serial(aec) != handled) {     /* a hard post-chain reset happened inside this hop */
    handled = aec_post_reset_serial(aec);
    ac_istft_reset(istft);                       /* before the push below; the library reseeds its own comfort noise */
}
/* nr: with use_r2 and v.usable_linear, residual_psd_audio = v.r2 / 32768^2 (sk_scale_f32), else NULL */
nr = anr_process_gain_with_residual_psd(anr, v.base_spec, residual_psd_audio);   /* NULL: the denoiser refused the hop */
aec_apply_gain(aec, nr, spec);                   /* spec = B * min(nr, G_res); nr NULL: B * G_res */
aec_apply_cng(aec, nr, spec);                    /* comfort noise on the bins the gain cut; the same nr */
ac_istft_push(istft, spec, out);
/* on any refusal: aec_reset(aec); anr_reset(anr); ac_istft_reset(istft); handled = aec_post_reset_serial(aec); */
```

`--no-res` skips the two apply calls and renders `aec_apply_gain_spec(spec, v.base_spec, unity, nr, n_freqs)`; the
comfort noise stays off. The residual-echo scale is `AEC_PSD_SCALE_INV_F` (1/32768²), a constant of `aec.h`.
The comfort-noise generator belongs to the `Aec` (seed `AEC_CNG_SEED`, 42, restored by every
hard post-chain reset and by `aec_reset()`); the floor of the denoiser-gain factor is `AEC_CNG_NR_GAIN_FLOOR`
(-10 dB), applied inside `aec_apply_cng()`.

### Two spectra

`v.error_spec` and `v.formed_hop` are the *analysis* seam: the selected linear residual, never altered by the
output guard, paired with the previous formed hop through the STFT. The *rendering base* `B` is what
`aec_process()` itself multiplies by its gain: `E`, or the capture spectrum `E + echo` on a hop where the linear
estimate is unusable and `E` is louder than the capture (the output guard, `v.guard_active`). `B` is `E` bit for bit on
every hop the guard leaves alone and does not pair with `formed_hop` on a guard hop. This chain renders `B`:

* the denoiser's estimator reads `B` and its gain is applied to `B` (it judges the very spectrum it filters; the same
  spectrum the full-mode chain applied its gain to). C (`mono_nr_gain()`) and the Python reference
  (`run_nr_spectrum()`) agree on this;
* with `--no-nr` and the RES gain on, the output equals the library's own `aec_process()` (second `Aec` with
  `AEC_RESIDUAL_FULL`) sample for sample, comfort noise on or off: the synthesis uses the AEC's window table
  (`aec_synthesis_window()` of the initialised `Aec`), not audiolib's default window, which differs in the last float bit.

The four-channel chain hands its denoiser the fused `base_spec` as well (`array_sample_anr_gain()`); there it differs
from `error_spec` (the beamformer output E) only on a hop where a contributing lane's output guard fired. The board
samples outside this repository (`alg_audio_aec_nr.c`, `uvc_audio_live2.c`) feed `base_spec` to the denoiser too.

### The hard post-chain reset event

A hard reset of the AEC post chain (a confirmed delay change that restarts the filter, the first acquisition without
a warm transfer) happens inside the `aec_process_linear()` call, before that hop's synthesis, and moves
`aec_post_reset_serial()`; so does a separate `aec_reset()` call, which the serial already reflects when it returns. `follow_post_reset()` keeps the last handled serial; when it
differs, the synthesis drops its overlap-add tail before its `ac_istft_push()`; the library has already reseeded
its comfort-noise generator to `AEC_CNG_SEED` before the hop's first draw. The program never resets the AEC in response. The denoiser is
not touched by the event (the library does not touch NR state and neither does this program: NR keeps running through
the reset; following the event in the denoiser would be an application policy and this one is "no"). Warm-transfer and
soft (double-talk-aware) realignments, the bad-relock soft revert, `aec_apply_external_realign()` and a change of the alignment generation
(`AecLinearView.delay_state == CHANGED`) alone do not move the serial.

Strength: `aec_set_level(aec, level)` for the AEC's RES gain,
`anr_set_level(anr, level)` for the denoiser.

## Calling sequence

```
App                      Aec                 AnrDenoiser        AcIstft
 |-- aec_resolve_signal_grid + aec_config_from_level + residual = FULL   (derive_lane())
 |-- aec_get_mem_size / aec_init ---->|                          (rig_build())
 |-- anr_config_for_level_grid; anr_get_mem_size / anr_init ----->|
 |-- ac_stft_config_default; ac_istft_get_mem_size / ac_istft_init ---------->|
 |   per hop: aec_process_linear --> aec_process_residual --> aec_get_residual --> follow_post_reset()
 |            --> anr_process_gain_with_residual_psd --> aec_apply_gain
 |            --> aec_apply_cng --> ac_istft_push --> out   (process_hop(), mono_form())
 |   refused hop: aec_reset + anr_reset + ac_istft_reset  (rig_reset())
 |-- aec_set_level(...) / anr_set_level(...) between two hops
 |-- anr_destroy; aec_destroy; release the block (AcIstft has no destroy)
```

One hop is `grid.hop_size` samples (128 at 8 kHz and at the 16 kHz default, 256 at 16 kHz/512, 512 at
48 kHz); the output trails the input by that one hop (the synthesis), `--aec-only` by none.
Per-flow detail with return values: section 0 of
[mono 整合](../../../docs/reference/mono/integration.md#0-速覽流程時序圖公開內部函式外部內部參數).

Command line of the two binaries built from `main.c` (`example_board_adapter.c` takes none):

An unknown option (`--bogus`, a lone `--`) or a positional that is not an AEC level (`-1`, `-x`, a lone `-`) prints
`unknown option or malformed argument '<token>'` and the usage to stderr and exits 2, in both binaries and in the
`--print-mem-size` path; nothing is skipped silently.

| Argument | Meaning | Default |
|---|---|---|
| fourth positional (one digit: `0` mild, `1` balanced, `2` aggressive, `3` no_floor), or `--aec-level <level>` | AEC level (`AecLevel`); any other text exits 2; the positional and the option are the same setting and one AEC level may be given: a second one (positional or `--aec-level`) is a command-line error, exit 2 | `1` |
| `--nr-level <level>` (one digit: `0` mild, `1` moderate, `2` balanced, `3` aggressive; `--nr-preset` is the same option) | `AnrLevel`; any other text exits 2 | `2` |
| `--aec-only` | only the `Aec`, built with `AEC_RESIDUAL_STATE` (both AEC stages run, as in the chain); the hop is `aec_get_linear().linear_hop` after `aec_process_linear()` and `aec_process_residual()`, bit for bit the linear hop of the full chain's lane (equals `aec_process()` of the same lane), no NR, RES gain, comfort noise or synthesis, no reset event to follow | off |
| `--print-mem-size [aec-level \| --aec-level <level>] [--nr-level <level>] [--aec-only] [--no-nr] [--no-res] [--no-cng]` | `aec_nr_pipeline_static` only: print the pool budget, run a real init of every stage, exit without WAV I/O | - |
| `--no-nr`, `--no-res`, `--no-cng` | leave out the denoiser, the RES gain, the comfort noise; `--no-nr` alone (RES on, with or without `--no-cng`) is the AEC's full output, equal to `aec_process()` | off |
| `--legacy-amin` | the denoiser's noise floor without the residual-echo powers (`use_r2 = 0`) | off |
| `--fft-size 256\|512\|1024` | grid; 0 = the rate's default (also accepted by `--print-mem-size`); a value that is not a whole decimal integer exits 2 | 0 |
| `--debug`, `--timing` | once-per-second status line; per-stage cost at exit (the AEC rows read 0 under `make PROFILE=0`) | off |
| `--sample-rate <hz>` (`aec_nr_pipeline_static --print-mem-size` only) | the rate of the pool query; not a whole decimal integer: exit 2 | 16000 |

Public functions an integrator copies from this main: the ones in the sequence above and `aec_debug_status()`,
`anr_debug_status()`. Internal (not for application code): the AEC and NR headers other than
`aec.h`, `anr.h`. External parameters: the `AecConfig` fields `sample_rate`, `fft_size`, `delay_mode`,
`delay_num_filters`, `fixed_delay_samples` and the AEC level passed to `aec_config_from_level()` (stored in `AecConfig.level`), `AnrLevel`, the three switches above.
Internal: the AEC library's own tuning (`AecTuning`, not part of `AecConfig`), the `AnrConfig` fields, including the denoiser's `L` and
`num_init_frames` (pool-sizing); classification with header lines in section 0.3 of the manual above.

Python twin: `pipeline.py` (from the `Audio_ALG` root: `python -m pipelines.mono.aec_nr_res.pipeline --mic mic.wav --ref ref.wav --output out.wav`,
presets `mild|balanced|aggressive` only; C level == Python preset, the C program takes the digit). Tests: `tests/test_mono_chain.c` (`make test`), `tests/test_no_stdio_stack.c`
(`make audit-no-stdio-stack`) and `tests/test_signal_grids.py` (`python -m pytest pipelines/mono/aec_nr_res/tests` from the `Audio_ALG` root).

The top-level `pipelines/Makefile` owns the actual build so all applications
share one backend/SIMD configuration contract (this directory's `Makefile` forwards `all`, `test` and `clean` to it):

```sh
make SIMD=0 test
make SIMD=1 test
```

The configuration selects sample rate and FFT independently. Supported
production grids are 16 kHz/256/128, 16 kHz/512/256 and 48 kHz/1024/512.
`delay_mode`, `delay_num_filters` and
`fixed_delay_samples` are independent initialization controls (the filter length follows the sample rate).
