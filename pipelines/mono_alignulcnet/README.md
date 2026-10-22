# Mono PBFDKF + Align-ULCNet application

This directory is an application integration example, not another DSP
library. The executable directly links the pipeline wrapper object, the
Align-ULCNet C pre/post-processing object, and the existing AEC/audiolib/common
libraries.

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
`AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES` = 1 frame here (both branches are
pushed from the current hop), the same as the 4ch wrapper, which also frames both branches from the current hop. The constant is
derived and asserted by the straddle-derivation test, not estimated; option B
(keep stepping through those frames) is deferred pending an audio A/B.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata requires the checkpoint and deployment contracts
to both be raw far.

## One hop in phases

`audio_pipeline_ulcnet_process()` is the sequence below with the config's model
callback. A board that runs the network itself (the NPU between `nn_begin` and
`nn_end`) calls the phases directly and needs no `infer` callback:

```c
UlcnetNnJob job;
audio_pipeline_ulcnet_process_begin(p, mic, ref);        /* AEC, both analyses */
if (audio_pipeline_ulcnet_nn_begin(p, &job) > 0) {       /* 1 = a frame to run */
    int rc = run_network(&job);                          /* write every bin of job.out_re/out_im */
    audio_pipeline_ulcnet_nn_end(p, rc);                 /* accepted only if rc == 0 and finite */
}
audio_pipeline_ulcnet_process_end(p, out);               /* WOLA, one hop */
```

A frame left open or never fetched is the identity (fail-open); the reprime
frame after an alignment boundary has no model frame (`nn_begin` returns 0).

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
