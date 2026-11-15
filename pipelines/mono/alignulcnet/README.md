# Mono PBFDKF + Align-ULCNet application

This directory is an application integration example, not another DSP
library, and there is no mono unit: `main.c` composes lib/aec's `Aec` (the linear stage) with the
spectrum entries of the Align-ULCNet model stream, `stream/aiaec/ulcnet/ulcnet_stream.{h,c}`
(`ulcnet_stream_pre_process_freq()` / `ulcnet_stream_post_process_freq()`: model call policy;
spectra in, one spectrum out). The
framing is `main.c`'s own: an `AcStft` for the formed hop, one for the raw reference and an `AcIstft`
for the output spectrum. The
executable links the stream object, the Align-ULCNet C pre/post-processing objects, and the existing
AEC/audiolib/common libraries. The AEC, the stream and the framing objects are initialised separately
(`aec_get_mem_size()`, `ulcnet_stream_get_mem_size()`, `ac_stft_get_mem_size()` / `ac_istft_get_mem_size()`); the example prints the
total pool size. The stream's pool is 103,312 B at 16 kHz and 202,640 B at 48 kHz: it also holds the
analysis and synthesis objects of the stream's PCM entries, which this example does not call.

`main.c` shows the board boundary. The K/V windows, logit window and GRU hidden state belong
to the application; on a board they live in the accelerator's own buffers (the stream registers
none, so the state views in `UlcnetHopIO` are NULL) and only the stateless
tensor invocation in `run_accelerator()` remains a board TODO. The runtime binds each `*_out`
to the buffer of the state input of the same name (the graph shifts the windows itself, so
nothing is copied) and reads every state input before it writes any state output. A fixed-point
accelerator loads quantized zeros at start and whenever `UlcnetHopIO.state_generation`
changes (a reset, an alignment boundary, or a frame whose output was refused).
Returning an error is intentionally fail-open and emits the linear error.

The exported graph fixes `D`. The example constructs a `D=8` descriptor and checks it against the
grid (`ulcnet_model_io_descriptor_validate()`, one frame per call);
production board code loads the descriptor exported with its graph. Changing
D means exporting another graph and descriptor, but does not require retraining
the weights.

The grid is a run-time `UlcnetGrid`: `--sample-rate 16000` (fft 512, default) or `48000`
(fft 1024) builds the one that sizes the lane and the stream. A model descriptor of
another grid, of the aligned-far contract or of a packet graph is refused by that check before any hop runs.

The production far branch is fixed: it always consumes the caller's raw
reference, matching training and the paper. PBFDKF uses its aligned copy only
to form the linear error; the model's TA block aligns raw far to that error.
The model state is reset when the AEC delay regime changes, including FIXED
mode's first usable error. The application's STFT keeps running across it, so the
frames still straddling the error change emit identity WITHOUT stepping the model:
`ULCNET_STREAM_REPRIME_FRAMES` = 1 frame (both branches are framed
from the current hop, as in the 4ch example). The constant is derived and
asserted by the straddle-derivation test (`tests/test_mono_ulcnet.c`), not estimated.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata requires the checkpoint and deployment contracts
to both be raw far.

## One hop in phases

The network runs between the stream's two calls; there is no callback to install:

```c
UlcnetHopIO io;
UlcnetStreamDelay delay;                                 /* from AecLinearView.delay_state */
int ai_rc = -1;
aec_process_linear(aec, mic, ref);                       /* linear stage; AEC_RESIDUAL_STATE: */
aec_process_residual(aec, NULL);                         /*   the selected residual, no gain */
aec_get_residual(aec, &rctx);  aec_get_linear(aec, &lctx);
delay.changed = lctx.delay_state == AEC_LINEAR_DELAY_CHANGED;
delay.solid   = lctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
ac_stft_push_planar(err_stft, rctx.formed_hop, err_re, err_im);   /* the application's analyses */
ac_stft_push_planar(far_stft, ref, far_re, far_im);
pre = ulcnet_stream_pre_process_freq(stream, err_re, err_im, far_re, far_im, delay, &io);  /* reprime / model restart */
if (pre == 1)                                            /* 1 = a frame to run, 0 = the reprime hop */
    ai_rc = run_accelerator(NULL, &io.inputs, &io.outputs);   /* write every value of io.outputs.output */
ulcnet_stream_post_process_freq(stream, ai_rc, out_re, out_im);   /* accepted only if ai_rc == 0 and finite, else the error */
ac_istft_push_planar(istft, out_re, out_im, out);        /* the application's WOLA, one hop */
/* on any negative return: aec_reset(aec); ulcnet_stream_reset(stream);
 *                         ac_stft_reset(err_stft); ac_stft_reset(far_stft); ac_istft_reset(istft); */
```

The accelerator's tensors are the graph's: `io.inputs` holds the four feature tensors, `io.outputs.output`
the complex mask (real plane, then imaginary, NaN-prefilled), and the stream applies the mask.

**Data domain.** The stream is fed the AEC's analysis seam, `rctx.formed_hop` (the selected linear residual, the same
signal in training and in C inference; the AIAEC frontend contract). It does not read the rendering base
(`rctx.base_spec`): that spectrum exists for chains that render `base * gain` and does not pair with
`formed_hop` on a hop where the output guard fires. Consequently the output guard (capture instead of residual when the
estimate is unusable and the residual is louder) is not part of this chain's audio: on such hops the network sees the
real linear residual, which may exceed the capture (a characteristic to evaluate with the retrained model, not a
property fixed here). This program has no comfort noise and no synthesis of its own besides the application's WOLA, so the
AEC's hard post-chain reset event (`aec_post_reset_serial()`) has nothing to reseed or clear here; the stream's own
policy for a changed alignment is the `delay.changed` flag above, and following a reset event in the network state is
application policy (this program: none beyond that flag, and it never resets the AEC in response).

A hop closed with a non-zero `ai_rc`, a mask left unwritten or a non-finite mask or output is the identity
(fail-open; the last three also restart the state generation); the reprime
hop after an alignment boundary has no model frame (`pre_process_freq` returns 0). The error is a time-domain
hop here, so `main.c` owns a second `AcStft` for it; a caller whose error already is a spectrum (the 4-channel
beam spectrum) passes its planes straight to `pre_process_freq`.

## Calling sequence and parameters

```
App                 Aec           UlcnetStream (spectrum entries)   accelerator (board)
 |-- ulcnet_grid_make(--sample-rate); aec_config_from_level; aec_get_mem_size / aec_init -->|
 |-- ulcnet_stream_get_mem_size(&cfg, &req) / ulcnet_stream_init_ex; ac_stft_init x2 / ac_istft_init   (state: the accelerator's)
 |   per hop: aec_process_linear --> aec_process_residual --> aec_get_residual / aec_get_linear --> delay {changed, solid}
 |            --> ac_stft_push_planar(formed_hop), ac_stft_push_planar(ref) --> ulcnet_stream_pre_process_freq(err, far, delay, &io)
 |            --> returns 1 --> run_accelerator(io.inputs, io.outputs) --> ai_rc
 |            --> ulcnet_stream_post_process_freq(ai_rc, out) --> ac_istft_push_planar(out)   (overlap-add: hop 0 is real output; fail-open = the error spectrum)
 |   refused call: aec_reset + ulcnet_stream_reset + ac_stft_reset x2 + ac_istft_reset   (an alignment change restarts the model state itself)
 |-- aec_destroy; release the blocks
```

External parameters: the delay profile below (`--delay-mode`, `--delay-num-filters` 1..5, `--fixed-delay`), the
grid (`--sample-rate 16000|48000`) and `D` of the exported graph (the example builds a `D=8`
descriptor; production loads the descriptor exported with its graph). There is no strength setter on
purpose: the strength is a property of the model. Internal: `ULCNET_STREAM_REPRIME_FRAMES` (a constant
asserted by the test, not a knob), the AEC lane constants, the model's pre/post constants.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up. `n` is an init parameter, not
a runtime setter — changing it means re-querying the pool and re-initializing.

```sh
./mono_alignulcnet                              # 16 kHz, matched, n=5 (default)
./mono_alignulcnet --sample-rate 48000          # the 48 kHz / fft 1024 grid
./mono_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./mono_alignulcnet --delay-mode fixed --fixed-delay 1600
./mono_alignulcnet --delay-mode external        # caller pre-aligns the far
```

Numbers are parsed by `example_cli.h` (`example_cli_parse_int`): `--delay-num-filters 3x`,
`--fixed-delay 12x`, `--sample-rate 16000x`, an empty string, a leading space or a value outside
`int` exit with 2, as does a `--sample-rate` that is not 16000 or 48000; plain decimals and a
leading `+` are accepted.

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5; each filter costs 11,264 bytes of pool (23,552 at 48 kHz). `0` is not "off" — use
`--delay-mode fixed` (delay known at bring-up) or `external` (upstream
guarantees alignment) instead. A bulk delay beyond the ceiling does not
merely fail to lock: with any in-range early reflection present the estimator
can lock onto that instead, at full confidence, and nothing in the AEC seam
distinguishes it from a correct lock — see the known-delay tests in
`tests/`.

The binary is built by `pipelines/Makefile` (target `mono_alignulcnet`; the `Makefile` in this
directory forwards to it). From this directory:

```sh
make SIMD=0 test
make SIMD=1 test
```

`test` runs `test_mono_ulcnet` (both grids from one binary),
the skeleton at n=5 and n=1 on both grids, and the malformed-number checks above. The stream's spectrum entries on their own
(`stream/aiaec/ulcnet/tests/test_ulcnet_stream_freq.c`) run in `make -C stream/aiaec/ulcnet test`.
