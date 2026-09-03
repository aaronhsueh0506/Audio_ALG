# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It directly composes the
shared 4-lane linear-AEC core, SRP-PHAT/GSC, Align-ULCNet C pre/post code and
their existing component libraries. The core is initialized in pre-only
mode, so the unused conventional NR/RES post path is not allocated.

The model accelerator is intentionally left as the TODO in
`run_accelerator()`. CPU memory owns every K/V, logit and GRU state tensor.
The default callback failure demonstrates the production fail-open path.

The production far branch is fixed to the shared AEC seam's
`pre.aligned_ref`, the same reference consumed by all four PBFDKF lanes.
On every hop the shared ring cannot yet serve the applied offset — before
acquisition under MATCHED, and for the whole ring-fill window under FIXED —
this seam carries the raw far hop, so the model still runs on real reference
audio; D handles the remaining offset. From the first hop the ring can serve
it, the seam carries the aligned far. The switch is whole-hop and coincides
with `pre.delay.solid`.
At that boundary — including FIXED mode's first usable ring output — the
wrapper does two things and nothing else: it calls `model->reset` (so the
runtime flushes its far attention ring and logit history) and it arms the
identity reprime. No buffer is cleared; every C-side framing state keeps
running across the switch. (The layout-16 wrapper wiped its beam-WOLA
accumulator at every boundary and emitted one half-window-tapered hop there;
with no reconstruction stage left there is nothing to wipe, so the boundary
hop now comes out whole — see the header's MODEL CALLBACK POLICY note.)
The frame whose analysis window still covers a pre-switch hop therefore
emits the identity WITHOUT stepping the model:
`AUDIO_PIPELINE_4CH_ULCNET_REPRIME_FRAMES` = 1 frame, the same count as the
mono wrapper, because both branches are framed from the CURRENT input hop and
each 50%-overlap frame spans two hops. The constant is derived and asserted
by the straddle-derivation test, not estimated; option B (keep stepping
through those frames) is deferred pending an audio A/B.

## Signal path and timing (direct path, 2026-09-03)

The GSC's beamformed linear-error SPECTRUM is handed to the model directly:
it is already the sqrt-Hann, 50%-overlap, one-frame-per-hop analysis frame
the Align-ULCNet chain would otherwise compute, so nothing is reconstructed
or re-analysed in between. The far branch pushes the SAME hop's
`pre.aligned_ref` through `ulcnet_analysis_push_frame()` (`ulcnet_process.h`;
center=False, exactly one frame per push from the first push). Both branches
therefore carry input hop t at pipeline hop t with no delay buffer on either
side, and every hop from hop #0 is exactly one inference.

Total added algorithmic latency is **1 hop** — 256 samples / 16 ms at 16 kHz,
512 samples / 10.67 ms at 48 kHz — the Align-ULCNet synthesis WOLA and
nothing else. `process()` emits zeros for hop #0 only; the output of hop p
(p >= 1) is the beamformed error of input hop p-1. Measured with a unit
impulse: far at sample 2085 comes out at 2341, i.e. impulse + 256 exactly.

This replaced a path that reconstructed a beam hop by WOLA and re-analysed
it, which cost one IFFT plus one RFFT and one extra hop of latency per hop.
Removed with it: the beam WOLA carves (`ifft`, `ola`, `synth_win`,
`beam_hop`), the error-branch `UlcnetAnalysis`, the two-frame staging, the
one-hop far compensation buffer, and the C API entry point
`audio_pipeline_4ch_ulcnet_last_beamformed_error()` — there is no
reconstructed beam hop to hand out any more. The layout version
(`AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION`) is **17**; a persisted
version-16 descriptor is refused by
`audio_pipeline_4ch_ulcnet_init_ex()`.

Spectrogram-consistency note: `gsc_spectrum` is per-bin weighted, i.e. an
inconsistent spectrogram, and the removed round trip projected it onto a
consistent one. Measured earlier on real 4-channel recordings, that
projection moved about -30 dB complex (-33 dB magnitude), -36..-43 dB over
the 750-4000 Hz speech band and -22 dB at DC in the worst case. The direct
path feeds the model the un-projected spectrum; no listening A/B has been
done.

Raw/aligned selection exists only in `sweep_delay_depth.py`, not in this
runtime API. Export metadata keeps the checkpoint's raw-far training
provenance separate from the aligned-far deployment contract.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up — at the default matched/n=5
profile, `BACKEND=ne10`, that banner reads 3,109,248 B at 16 kHz and
6,468,768 B at 48 kHz. Read it from your own build rather than copying it:
the FFT backend moves the total (the same 16 kHz profile costs 3,116,128 B
under `BACKEND=kiss`). `n` is an init parameter, not a runtime setter —
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
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```
