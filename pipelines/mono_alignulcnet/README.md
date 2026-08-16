# Mono PBFDKF + Align-ULCNet application

This directory is an application integration example, not another DSP
library. The executable directly links the pipeline wrapper object, the
Align-ULCNet C pre/post-processing object, and the existing AEC/audio_common
libraries.

`main.c` shows the board boundary. CPU memory owns the K/V ring, logit
history and GRU hidden states through `ulcnet_accelerator_adapter`; only the
stateless tensor invocation in `run_accelerator()` remains a board TODO.
Returning an error is intentionally fail-open and emits the linear error.

The exported checkpoint fixes `D`. The example uses `D=8`; change it only
together with the ONNX descriptor. `far_input_mode` must also equal the
checkpoint contract (`RAW` for existing checkpoints, `ALIGNED` only for a
checkpoint trained that way).

```sh
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```
