# Mono PBFDKF + Align-ULCNet application

This directory is an application integration example, not another DSP
library. The executable directly links the pipeline wrapper object, the
Align-ULCNet C pre/post-processing object, and the existing AEC/audio_common
libraries.

`main.c` shows the board boundary. CPU memory owns the K/V ring, logit
history and GRU hidden states through `ulcnet_accelerator_adapter`; only the
stateless tensor invocation in `run_accelerator()` remains a board TODO.
Returning an error is intentionally fail-open and emits the linear error.

The exported graph fixes `D`. The example constructs a `D=8` descriptor;
production board code loads the descriptor exported with its graph. Changing
D means exporting another graph/descriptor and rebuilding its state pool, but
does not require retraining the weights.

The production far branch is fixed: it always consumes
`AecLinearContext.aligned_far_hop`, the same hop PBFDKF consumed. Before the
matched delay is acquired, this seam contains raw far and the model still
runs; D handles the remaining offset. After acquisition it contains aligned
far. The model state is reset at that boundary, including FIXED mode's first
usable ring output. The C-side STFT keeps running across it, so the frames
still straddling the switch emit the identity WITHOUT stepping the model:
`AUDIO_PIPELINE_ULCNET_REPRIME_FRAMES` = 1 frame here (both branches are
pushed from the current hop), the same as the 4ch wrapper, which since
2026-09-03 also frames both branches from the current hop. The constant is
derived and asserted by the straddle-derivation test, not estimated; option B
(keep stepping through those frames) is deferred pending an audio A/B.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata records the checkpoint's raw-far training
provenance separately from the aligned-far deployment contract.

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
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```
