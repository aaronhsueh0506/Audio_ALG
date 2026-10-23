# Mono AEC + NR/RES application

There is no mono pipeline library and no mono unit: the example composes the libraries itself.
`lib/aec`'s `Aec` runs the linear stage (`aec_process_context()`), `se_frame_from_aec()`
(`common/se_stage.h`) turns its context into a `SeSpecFrame`, and
`samples/se_sample_chain.h` runs the gain sequence (the AEC's own RES gain, the denoiser,
`min`, apply, comfort noise) and the synthesis. Each library has its own `mem_size` and `init`;
the example sums the two sizes it needs (`aec_get_mem_size()`, `se_sample_chain_mem_size()`).

The heap and caller-pool WAV runners are `main.c` and `static_main.c`;
`example_board_adapter.c` simulates the board's memory manager around the same sequence.
This directory builds no archive: the examples link `common/se_config.o` and the producer
libraries directly.

Per hop (the code is `process_hop()` in `main.c`):

```c
aec_process_context(aec, mic, ref);              /* --aec-only: aec_process(aec, mic, ref, out) */
se_frame_from_aec(aec, &frame);
P = se_sample_chain_form(chain, &frame);         /* NULL: a stage refused the hop */
se_sample_chain_synthesize(chain, P, out);
/* on any refusal: aec_reset(aec); se_sample_chain_reset(chain); */
```

Strength: `aec_set_preset(aec, preset, ramp_ms)` for the AEC's RES gain,
`anr_set_mode(se_sample_chain_nr(chain), mode)` for the denoiser.

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
