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
