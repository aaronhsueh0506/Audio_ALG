# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It composes the 4-channel linear AEC flow
`mcab` (`4ch_aec_bf_nr_res/mcab.h`) and the sample SRP-PHAT/BF helper (`mcab_sample_beam.h`) with
the example stage `ulcnet/ulcnet_nn_stage.{h,c}` and the Align-ULCNet C pre/post code, over their
existing component libraries. mcab is a flow layer: the application initialises the four libaec
`Aec` lanes and the shared `DelayAec` and passes the pointers (`main.c` does it through
`mcab_sample_flow.h`), and mcab owns only its flow state. The flow is initialized with
`enable_post = 0` (`ulcnet_example_unit_config()` in `main.c`: `mcab_default_config(ULCNET_SR)` with
`fft_size` = `ULCNET_N_FFT`), so the fuse buffers of the conventional post path are not allocated,
there is no RES (mcab holds none; this application creates no `AecRes`) and `mcab_fuse()` /
`mcab_sample_beam_fuse()` are unavailable (`MCAB_INVALID_ARGUMENT`). The flow plus helper and the
stage are two pools: `main.c` queries `mcab_sample_flow_get_mem_size()`,
`mcab_sample_beam_get_mem_size()` and `ulcnet_nn_stage_mem_size(0)`, cuts both blocks (the stage's
block 16-byte aligned) and prints the two sizes.

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
identity reprime. It reads the boundary from the helper's `info.delay` (the `McabDelayState` of the
front, fields `changed`, `solid`). No buffer is cleared; every C-side framing state keeps
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
caller's reference, never the flow's aligned copy) through a rolling `AcStft`
(`ulcnet_framing_config`, `ulcnet_process.h`; `center` 0, exactly one frame per push from the first push). Both branches
therefore carry input hop t at pipeline hop t with no delay buffer on either
side, and every hop from hop #0 is exactly one inference.

Total added algorithmic latency is **1 hop** — 256 samples / 16 ms at 16 kHz,
512 samples / 10.67 ms at 48 kHz — the Align-ULCNet synthesis WOLA and
nothing else. `ulcnet_nn_stage_end()` emits zeros for hop #0 only; the output of hop p
(p >= 1) is the beamformed error of input hop p-1.

The beam hop is not reconstructed by WOLA and re-analysed (that would cost one
IFFT plus one RFFT and one extra hop of latency per hop), so the pipeline has no
reconstructed beam hop to hand out. mcab's pool layout version
(`MCAB_LAYOUT_VERSION`, 25) describes mcab's own pool only; a descriptor
whose layout version differs from the current build is refused by
`mcab_init_ex()`. The stage is example code: it has no
descriptor and no layout version.

Spectrogram-consistency note: `bf_spectrum` is per-bin weighted, i.e. an
inconsistent spectrogram, and a WOLA round trip would project it onto a
consistent one (how far that projection moves the spectrum on real
4-channel recordings is to be measured). The direct path feeds the model the
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
McabSampleBeamResult beam;  McabSampleBeamInfo info;
mcab_sample_beam_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);  /* lanes, DOA, BF */
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

Recovery is the integrator's: after any refused call (a negative return from the helper or the
stage) call `mcab_sample_beam_reset(unit)` and `ulcnet_nn_stage_reset(stage)` together;
`test_4ch_ulcnet.c` checks that the pair then equals a fresh pair byte for byte.

### Mask and VAD derived from the lane spectra

A caller that owns a per-bin mask and a VAD computed from the lane spectra of
the same hop (a masker over lane 0, for example) cannot hand them to
`process_with_activity()`: they exist only after the lanes ran. It splits
the hop into two calls; `mcab_sample_beam_process_with_activity(unit, mics, far, v, v, NULL, ...)` is
`mcab_sample_beam_process_front()` followed by
`mcab_sample_beam_process_beam(unit, &front, v, v, NULL, ...)`:

```c
McabFront front;
mcab_sample_beam_process_front(unit, mics, far, &front);   /* shared aligner and four lanes */
/* mask (n_freqs ints), vad_raw, vad_out from front.linear_spectra[0] */
mcab_sample_beam_process_beam(unit, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT, BF */
/* info: doa_raw/smooth/used in radians (NaN = none), bf_adaptive, delay */
/* then ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {front.delay.changed, front.delay.solid}),
   nn_begin / nn_end / end as above */
```

`mask[f]` nonzero: SRP uses bin `f`, BF freezes its adaptation on bin `f`;
NULL means all bins used and none frozen. `vad_raw` gates SRP, `vad_out` feeds the
DOA smoother and freezes BF adaptation. `front.linear_spectra` are read-only views
valid until the next `process_front()`, `process_with_activity()`, `reset()` or
`destroy()`. `process_beam()` without a pending front returns `MCAB_SEQUENCE_ERROR`. The split adds
no state to the helper: the pool sizes and the layout version are unchanged.

## Calling sequence and parameters

```
App       McabSampleFlow + Mcab (enable_post = 0)   McabSampleBeam      UlcnetNnStage + adapter       accelerator (board)
 |-- mcab_default_config (ulcnet_example_unit_config()); mcab_sample_flow_get_mem_size / _init -->|
 |-- mcab_sample_beam_default_params; mcab_sample_beam_get_mem_size / _init(flow) -->|
 |-- ulcnet_nn_stage_mem_size(0) / ulcnet_nn_stage_init; adapter state pools
 |   per hop: mcab_sample_beam_process_with_activity(mics, far, vad, vad, NULL, &beam, &info)   [or _process_front + mask/VAD + _process_beam]
 |            --> ulcnet_nn_stage_begin_complex(beam.spectrum, far, {info.delay.changed, info.delay.solid})
 |            --> nn_begin == 1 --> run_accelerator(job) --> nn_end(rc) --> ulcnet_nn_stage_end(out)
 |   no mcab_sample_beam_fuse() here (INVALID_ARGUMENT); no RES, no NR
 |   refused call: mcab_sample_beam_reset + ulcnet_nn_stage_reset
 |-- mcab_sample_beam_destroy; mcab_sample_flow_destroy; release the blocks
```

External parameters: the delay profile below, the grid of the build (`ULCNET_GRID`), `D` of the exported
graph, and the helper's array geometry and `num_angles` (the example uses the default 35 mm UCA; a product
sets the measured one through `McabSampleBeamParams`). Internal: the helper's BF / DOA / VAD constants (section 0.3 of
[`../../../docs/integration_4ch_spatial_zh_TW.md`](../../../docs/integration_4ch_spatial_zh_TW.md)),
`ULCNET_NN_STAGE_REPRIME_FRAMES`, the lane constants.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up — at the default matched/n=5
profile, that banner reads a 2,982,080 B unit pool (the flow objects plus the helper) plus a
36,400 B stage pool at 16 kHz and a 6,279,776 B unit pool plus a 71,728 B stage pool at 48 kHz
(`ULCNET_GRID=48k`). Read it from your
own build rather than copying it.
`n` is an init parameter, not a runtime setter —
changing it means re-querying the pool and re-initializing.

```sh
./4ch_alignulcnet                              # matched, n=5 (default)
./4ch_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./4ch_alignulcnet --delay-mode fixed --fixed-delay 1600
./4ch_alignulcnet --delay-mode external        # caller pre-aligns the far
```

Numbers are parsed by `common/example_cli.h` (`example_cli_parse_int`): `--delay-num-filters 3x`,
`--fixed-delay 12x`, an empty string, a leading space or a value outside `int` exit with 2; plain
decimals and a leading `+` are accepted.

There is exactly ONE matched bank here: the shared estimator that the application hands to mcab.
The four lane AECs run `EXTERNAL_ALIGNED` off its single aligned reference, so
each filter costs 5,728 bytes ONCE — not four times — and the lane pools and mcab's own pool do
not move with `n` at all (asserted in `4ch_aec_bf_nr_res/tests/test_mcab.c`).

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5. `0` is not "off" — use `--delay-mode fixed` (delay known at
bring-up) or `external` (upstream guarantees alignment) instead. A bulk delay
beyond the ceiling does not merely fail to lock: with any in-range early
reflection present the estimator can lock onto that instead, at full
confidence, and nothing in the delay seam distinguishes it from a correct
lock — see the known-delay tests.

The rules live in `pipelines/4ch_aec_bf_nr_res/Makefile` (targets `4ch_alignulcnet`,
`test_4ch_ulcnet`); the `Makefile` in this directory forwards to it. From this directory:

```sh
make SIMD=0 test
make SIMD=1 test
make ULCNET_GRID=48k test   # the other compiled grid (16k is the default)
```

`test` runs `test_4ch_ulcnet`, the skeleton at n=5 and n=1 and the malformed-number checks above.
The archive `libaudio_pipeline_mcab.a` holds `mcab.o` only; the helper and flow objects, the stage
and the ULCNet pre/post objects are linked into the program.
