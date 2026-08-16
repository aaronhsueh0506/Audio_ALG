# Mono AEC + NR/RES application

The application uses the existing AEC, NR and audio_common libraries through
`audio_pipeline.h`. The heap and caller-pool WAV runners are `main.c` and
`static_main.c`; `example_board_adapter.c` demonstrates board integration.
This directory is the stable application entry point and does not introduce
an additional application-specific archive beyond the reusable
`libaudio_pipeline.a` integration library.

The top-level `pipelines/Makefile` owns the actual build so all four
applications share one backend/SIMD configuration contract:

```sh
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```

The configuration selects sample rate and FFT independently. Supported
production grids are 16 kHz/256/128, 16 kHz/512/256 and 48 kHz/1024/512.
`filter_length`, `delay_mode`, `delay_num_filters` and
`fixed_delay_samples` are independent initialization controls.
