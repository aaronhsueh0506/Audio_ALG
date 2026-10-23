# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It composes the 4-channel linear AEC +
BF unit (`AudioPipeline4Ch`, `audio_pipeline_4ch.h`) with the example stage
`samples/ulcnet_nn_stage.{h,c}` and the Align-ULCNet C pre/post code, over their existing
component libraries. The unit is initialized with `core.enable_post = 0`
(`ulcnet_example_unit_config()` in `main.c`: 16 kHz or 48 kHz default config, `fft_size` =
`ULCNET_N_FFT`), so the fuse buffers of the conventional post path are not allocated and
`audio_pipeline_4ch_fuse()` is unavailable. The unit and the stage are two libraries with two
pools: `main.c` queries `audio_pipeline_4ch_get_mem_requirements()` and
`ulcnet_nn_stage_mem_size(0)`, cuts both blocks (the stage's block 16-byte aligned) and prints the
two sizes.

The model accelerator is intentionally left as the TODO in
`run_accelerator()`. CPU memory owns every K/V, logit and GRU state tensor.
The runtime binds each `*_out` to the address of the state input of the same name; the graph shifts the windows itself, so the CPU neither copies nor shifts them.
The default callback failure demonstrates the production fail-open path.

The production far branch is fixed to the caller's raw reference, matching
training and the paper. The shared AEC ring supplies aligned far only to the
four PBFDKF lanes that form the linear-error spectra; the model's TA block
aligns the raw far to the beamformed error.
At an AEC delay-regime boundary — including FIXED mode's first usable error — the
stage does two things and nothing else: it calls `model->reset` (so the
runtime flushes its far attention ring and logit history) and it arms the
identity reprime. It reads the boundary from the unit's `front.delay`
(`changed`, `solid`). No buffer is cleared; every C-side framing state keeps
running across the switch. The frame whose analysis window still covers a pre-switch hop therefore
emits the identity WITHOUT stepping the model:
`ULCNET_NN_STAGE_REPRIME_FRAMES` = 1 frame, the same count as the
mono example, because both branches are framed from the CURRENT input hop and
each 50%-overlap frame spans two hops. The constant is derived and asserted
by the straddle-derivation test (`tests/test_4ch_ulcnet.c`), not estimated.

## Signal path and timing (direct path)

The BF's beamformed linear-error SPECTRUM is handed to the model directly:
it is already the sqrt-Hann, 50%-overlap, one-frame-per-hop analysis frame
the Align-ULCNet chain would otherwise compute, so nothing is reconstructed
or re-analysed in between. The far branch pushes the SAME hop's raw far samples (the
caller's reference, never the unit's aligned copy) through a rolling `AcStft`
(`ulcnet_framing_config`, `ulcnet_process.h`; `center` 0, exactly one frame per push from the first push). Both branches
therefore carry input hop t at pipeline hop t with no delay buffer on either
side, and every hop from hop #0 is exactly one inference.

Total added algorithmic latency is **1 hop** — 256 samples / 16 ms at 16 kHz,
512 samples / 10.67 ms at 48 kHz — the Align-ULCNet synthesis WOLA and
nothing else. `ulcnet_nn_stage_end()` emits zeros for hop #0 only; the output of hop p
(p >= 1) is the beamformed error of input hop p-1.

The beam hop is not reconstructed by WOLA and re-analysed (that would cost one
IFFT plus one RFFT and one extra hop of latency per hop), so the pipeline has no
reconstructed beam hop to hand out. The unit's layout version
(`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`) is **18**; a persisted
descriptor from any earlier version is refused by
`audio_pipeline_4ch_init_ex()`. The stage is example code: it has no
descriptor and no layout version.

Spectrogram-consistency note: `bf_spectrum` is per-bin weighted, i.e. an
inconsistent spectrogram, and a WOLA round trip would project it onto a
consistent one (on real 4-channel recordings that projection moves the
spectrum by about -30 dB complex). The direct path feeds the model the
un-projected spectrum.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata requires the checkpoint and deployment contracts
to both be raw far.

## One hop in phases

`ulcnet_nn_stage_process()` is the sequence below with the model callback installed on the
stage. A board that runs the network itself (the NPU between `nn_begin` and
`nn_end`) calls the phases directly and needs no `infer` callback (`ulc_hop()` in `main.c`
calls `ulcnet_nn_stage_process()` instead, which drives the same phases with the model
callback):

```c
UlcnetNnJob job;
AudioPipeline4ChBeam beam;  AudioPipeline4ChFrameInfo info;
audio_pipeline_4ch_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);  /* lanes, DOA, BF */
UlcnetNnDelay delay = { info.delay.changed, info.delay.solid };
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay);   /* far analysis, reprime / model reset */
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {         /* 1 = a frame to run */
    int rc = run_network(&job);                          /* write every bin of job.out_re/out_im */
    ulcnet_nn_stage_nn_end(stage, rc);                   /* accepted only if rc == 0 and finite */
}
ulcnet_nn_stage_end(stage, out);                         /* WOLA, one hop */
```

A frame left open or never fetched is the identity (fail-open); the reprime
frame after an alignment boundary has no model frame (`nn_begin` returns 0).

Recovery is the integrator's: after any refused call (a negative return from the unit or the
stage) call `audio_pipeline_4ch_reset(unit)` and `ulcnet_nn_stage_reset(stage)` together;
`test_4ch_ulcnet.c` checks that the pair then equals a fresh pair byte for byte.

### Mask and VAD derived from the lane spectra

A caller that owns a per-bin mask and a VAD computed from the lane spectra of
the same hop (a masker over lane 0, for example) cannot hand them to
`process_with_activity()`: they exist only after the lanes ran. It splits
the hop into two calls; `process_with_activity(unit, mics, far, v, v, NULL, ...)` is
`process_front()` followed by `process_beam(unit, &front, v, v, NULL, ...)`:

```c
FourAecFront front;
audio_pipeline_4ch_process_front(unit, mics, far, &front);   /* shared aligner and four lanes */
/* mask (n_freqs ints), vad_raw, vad_out from front.linear_spectra[0] */
audio_pipeline_4ch_process_beam(unit, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT, BF */
/* info: doa_raw/smooth/used in radians (NaN = none), bf_adaptive, delay */
/* then ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {front.delay.changed, front.delay.solid}),
   nn_begin / nn_end / end as above */
```

`mask[f]` nonzero: SRP uses bin `f`, BF freezes its adaptation on bin `f`;
NULL means all bins used and none frozen. `vad_raw` gates SRP, `vad_out` feeds the
DOA smoother and freezes BF adaptation. `front.linear_spectra` are read-only views
valid until the next `process_front()`, `process_with_activity()`, `reset()` or
`destroy()`. `process_beam()` without a pending front returns `SEQUENCE_ERROR`. The split adds
no state to the instance: the pool size and the layout version are unchanged.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up — at the default matched/n=5
profile, that banner reads a 3,117,520 B unit pool plus a 36,400 B stage pool at 16 kHz and a
6,550,384 B unit pool plus a 71,728 B stage pool at 48 kHz (`ULCNET_GRID=48k`). Read it from your
own build rather than copying it.
`n` is an init parameter, not a runtime setter —
changing it means re-querying the pool and re-initializing.

```sh
./4ch_alignulcnet                              # matched, n=5 (default)
./4ch_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./4ch_alignulcnet --delay-mode fixed --fixed-delay 1600
./4ch_alignulcnet --delay-mode external        # caller pre-aligns the far
```

There is exactly ONE matched bank here: the shared estimator in the core. The
four lane AECs run `EXTERNAL_ALIGNED` off its single aligned reference, so
each filter costs 5,728 bytes ONCE — not four times — and the lane pools do
not move with `n` at all (asserted in `tests/test_4aec_nr_res.c`).

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5. `0` is not "off" — use `--delay-mode fixed` (delay known at
bring-up) or `external` (upstream guarantees alignment) instead. A bulk delay
beyond the ceiling does not merely fail to lock: with any in-range early
reflection present the estimator can lock onto that instead, at full
confidence, and nothing in the delay seam distinguishes it from a correct
lock — see the known-delay tests.

```sh
make SIMD=0 test
make SIMD=1 test
```
