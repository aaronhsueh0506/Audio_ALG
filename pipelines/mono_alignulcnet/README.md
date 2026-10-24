# Mono PBFDKF + Align-ULCNet application

This directory is an application integration example, not another DSP
library, and there is no mono unit: `main.c` composes lib/aec's `Aec` (the linear stage) with the
example stage `samples/ulcnet_nn_stage.{h,c}` (far-end STFT, model call policy, WOLA). The
executable links the stage object, the Align-ULCNet C pre/post-processing object, and the existing
AEC/audiolib/common libraries. The AEC and the stage are initialised separately
(`aec_get_mem_size()`, `ulcnet_nn_stage_mem_size()`); the example prints both pool sizes.

`main.c` shows the board boundary. CPU memory owns the K/V windows, logit
window and GRU hidden state through `ulcnet_accelerator_adapter`; only the
stateless tensor invocation in `run_accelerator()` remains a board TODO. The
runtime binds each `*_out` to the address of the state input of the same name
(the graph shifts the windows itself, so nothing is copied) and reads every state
input before it writes any state output.
Returning an error is intentionally fail-open and emits the linear error.

The exported graph fixes `D`. The example constructs a `D=8` descriptor;
production board code loads the descriptor exported with its graph. Changing
D means exporting another graph/descriptor and rebuilding its state pool, but
does not require retraining the weights.

The production far branch is fixed: it always consumes the caller's raw
reference, matching training and the paper. PBFDKF uses its aligned copy only
to form the linear error; the model's TA block aligns raw far to that error.
The model state is reset when the AEC delay regime changes, including FIXED
mode's first usable error. The C-side STFT keeps running across it, so the
frames still straddling the error change emit identity WITHOUT stepping the model:
`ULCNET_NN_STAGE_REPRIME_FRAMES` = 1 frame (both branches are framed
from the current hop, as in the 4ch example). The constant is derived and
asserted by the straddle-derivation test (`tests/test_mono_ulcnet.c`), not estimated.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata requires the checkpoint and deployment contracts
to both be raw far.

## One hop in phases

`ulcnet_nn_stage_process()` is the sequence below with the model callback installed
on the stage. A board that runs the network itself (the NPU between `nn_begin` and
`nn_end`) calls the phases directly and needs no `infer` callback:

```c
UlcnetNnJob job;
UlcnetNnDelay delay;                                     /* from AecLinearContext.delay_state */
aec_process_context(aec, mic, ref);                      /* linear stage */
aec_get_res_context(aec, &rctx);  aec_get_linear_context(aec, &lctx);
delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
delay.solid   = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
ulcnet_nn_stage_begin_time(stage, rctx.formed_hop, ref, delay);   /* far analysis, reprime / model reset */
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {         /* 1 = a frame to run */
    int rc = run_network(&job);                          /* write every bin of job.out_re/out_im */
    ulcnet_nn_stage_nn_end(stage, rc);                   /* accepted only if rc == 0 and finite */
}
ulcnet_nn_stage_end(stage, out);                         /* WOLA, one hop */
/* on any negative return: aec_reset(aec); ulcnet_nn_stage_reset(stage); */
```

A frame left open or never fetched is the identity (fail-open); the reprime
frame after an alignment boundary has no model frame (`nn_begin` returns 0). The stage is built
with `ulcnet_nn_stage_mem_size(1)` here because the time-domain error entry needs its own STFT.

## Calling sequence and parameters

```
App                 Aec           UlcnetNnStage + adapter          accelerator (board)
 |-- se_derive; aec_get_mem_size / aec_init -->|
 |-- ulcnet_nn_stage_mem_size(1) / ulcnet_nn_stage_init; ulcnet_accelerator_adapter_* (K/V, logit, GRU state)
 |   per hop: aec_process_context --> aec_get_res_context / aec_get_linear_context --> delay {changed, solid}
 |            --> ulcnet_nn_stage_begin_time(formed_hop, ref, delay)
 |            --> ulcnet_nn_stage_nn_begin == 1 --> run_accelerator(job) --> ulcnet_nn_stage_nn_end(rc)
 |            --> ulcnet_nn_stage_end(out)                (hop 0 is zeros; fail-open = the error spectrum)
 |   refused call: aec_reset + ulcnet_nn_stage_reset   (an alignment change resets the model state itself)
 |-- aec_destroy; release the blocks
```

External parameters: the delay profile below (`--delay-mode`, `--delay-num-filters` 1..5, `--fixed-delay`), the
grid of the build (`ULCNET_GRID=16k|48k`) and `D` of the exported graph (the example builds a `D=8`
descriptor; production loads the descriptor exported with its graph). There is no strength setter on
purpose: the strength is a property of the model. Internal: `ULCNET_NN_STAGE_REPRIME_FRAMES` (derived by the
test, not a knob), the AEC lane constants, the model's pre/post constants.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up. `n` is an init parameter, not
a runtime setter — changing it means re-querying the pool and re-initializing.

```sh
./mono_alignulcnet                              # matched, n=5 (default)
./mono_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./mono_alignulcnet --delay-mode fixed --fixed-delay 1600
./mono_alignulcnet --delay-mode external        # caller pre-aligns the far
```

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5; each filter costs 5,728 bytes of pool. `0` is not "off" — use
`--delay-mode fixed` (delay known at bring-up) or `external` (upstream
guarantees alignment) instead. A bulk delay beyond the ceiling does not
merely fail to lock: with any in-range early reflection present the estimator
can lock onto that instead, at full confidence, and nothing in the AEC seam
distinguishes it from a correct lock — see the known-delay tests in
`tests/`.

```sh
make SIMD=0 test
make SIMD=1 test
```
