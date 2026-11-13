# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It composes the 4-channel linear AEC array
(an `Aec` instance with `AecConfig.n_mics = AEC_ARRAY_MICS`, `aec.h`) and the sample SRP-PHAT/BF helper (`array_sample_beam.h`) with
the example stage `ulcnet/ulcnet_nn_stage.{h,c}` and the Align-ULCNet C pre/post code, over their
existing component libraries. The array builds its own four lanes and the shared estimator in one pool (`main.c` places it with
`aec_get_mem_requirements()` + `aec_init_ex()`); the application supplies none of them. The array is initialized with
`residual = AEC_RESIDUAL_STATE` (`ulcnet_example_unit_config()` in `main.c`: `array_sample_default_config(&cfg, grid.sample_rate)`
with `fft_size` = `grid.n_fft`; the grid is built with `ulcnet_grid_make()` from the `--sample-rate 16000|48000` row of the example's
`UlcnetGridSpec` table), so the fusion buffers and the gain stage of the conventional post path are not allocated,
there is no RES (this application has no gain stage) and `aec_process_residual()` /
`array_sample_beam_fuse()` are unavailable (`AEC_ERR_MODE`). The array plus helper and the
stage are two pools: `main.c` queries `aec_get_mem_requirements()`,
`array_sample_beam_get_mem_size()` and `ulcnet_nn_stage_mem_size(&grid)`, cuts both blocks (the stage's
block 16-byte aligned) and prints the sizes. The stage does no STFT or ISTFT: `main.c` owns an `AcStft` for the raw
far hop and an `AcIstft` for the output spectrum, in a third block (the beam spectrum is already a spectrum, so there is
no error analysis).

The model accelerator is intentionally left as the TODO in
`run_accelerator()`. The K/V, logit and GRU state tensors belong to the application; on a board
they live in the accelerator's own buffers, so the example registers none
(`ulcnet_accelerator_adapter_init(..., NULL, ...)`).
The runtime binds each `*_out` to the buffer of the state input of the same name; the graph shifts the windows itself, so the CPU neither copies nor shifts them.
A fixed-point accelerator loads quantized zeros at start and whenever
`ulcnet_accelerator_adapter_state_generation()` changes.
The default callback failure demonstrates the production fail-open path.

The production far branch is fixed to the caller's raw reference, matching
training and the paper. The shared AEC ring supplies aligned far only to the
four PBFDKF lanes that form the linear-error spectra; the model's TA block
aligns the raw far to the beamformed error.
At an AEC delay-regime boundary — including FIXED mode's first usable error — the
stage does two things and nothing else: it calls `model->reset` (so the
runtime flushes its far attention ring and logit history) and it arms the
identity reprime. It reads the boundary from the helper's `info.delay` (the `AecDelayState` of the
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
caller's reference, never the array's aligned copy) through a rolling `AcStft`
(the framing contract of `ulcnet_process.h`; `center` 0, exactly one frame per push from the first push). Both branches
therefore carry input hop t at pipeline hop t with no delay buffer on either
side, and every hop from hop #0 is exactly one inference.

Total added algorithmic latency is **1 hop** — 256 samples / 16 ms at 16 kHz,
512 samples / 10.67 ms at 48 kHz — the Align-ULCNet synthesis WOLA and
nothing else. The synthesis is overlap-add (`AC_OLA_ACCUMULATE`): every hop, hop #0 included,
is one full hop, this frame's windowed first half plus the previous frame's saved second half.
Hop #0 is real output (its first half is added to a zero history); in the identity case the output
of hop p is the beamformed error of input hop p-1.

The beam hop is not reconstructed by WOLA and re-analysed (that would cost one
IFFT plus one RFFT and one extra hop of latency per hop), so the pipeline has no
reconstructed beam hop to hand out. The array's pool layout version
(`AEC_LAYOUT_VERSION`) describes the array pool only; a descriptor
whose layout version differs from the current build is refused by
`aec_init_ex()`. The stage is example code: it has no
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
ArraySampleBeamResult beam;  ArraySampleBeamInfo info;
array_sample_beam_process_with_activity(unit, mics, far, vad, vad, NULL, &beam, &info);  /* lanes, DOA, BF */
UlcnetNnDelay delay = { info.delay.changed, info.delay.solid };
ac_stft_push(far_stft, far, far_spec);                   /* the application's far analysis */
ulcnet_nn_stage_begin_spec(stage, beam.spectrum, far_spec, delay);   /* reprime / model reset */
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {         /* 1 = a frame to run */
    int rc = run_network(&job);                          /* write every bin of job.out_re/out_im */
    ulcnet_nn_stage_nn_end(stage, rc);                   /* accepted only if rc == 0 and finite */
}
ulcnet_nn_stage_end_spec(stage, out_spec);               /* accepted model spectrum, else beam.spectrum */
ac_istft_push(istft, out_spec, out);                     /* the application's WOLA, one hop */
```

A frame left open or never fetched is the identity (fail-open); the reprime
frame after an alignment boundary has no model frame (`nn_begin` returns 0).

The network input is the analysis beam (`beam.spectrum`, the beam of the lanes' error spectra), as a network on
one lane reads that lane's error spectrum; the lanes' output-guard rendering bases (`AecArrayView.lane_base_spec`)
are not fed to it. The synthesis is the application's `AcIstft`; the lanes' post-chain reset counter
(`AecArrayView.post_reset_serial`) moves only with an array reset here, which the recovery below already pairs with
`ulcnet_nn_stage_reset()`.

Recovery is the integrator's: after any refused call (a negative return from the helper or the
stage) call `array_sample_beam_reset(unit)`, `ulcnet_nn_stage_reset(stage)`, `ac_stft_reset(far_stft)` and
`ac_istft_reset(istft)` together (the stage keeps no transform history);
`test_4ch_ulcnet.c` checks that the pair then equals a fresh pair byte for byte.

### Mask and VAD derived from the lane spectra

A caller that owns a per-bin mask and a VAD computed from the lane spectra of
the same hop (a masker over lane 0, for example) cannot hand them to
`process_with_activity()`: they exist only after the lanes ran. It splits
the hop into two calls; `array_sample_beam_process_with_activity(unit, mics, far, v, v, NULL, ...)` is
`array_sample_beam_process_front()` followed by
`array_sample_beam_process_beam(unit, &front, v, v, NULL, ...)`:

```c
AecArrayView front;
array_sample_beam_process_front(unit, mics, far, &front);   /* shared aligner and four lanes */
/* mask (n_freqs ints), vad_raw, vad_out from front.lane_error_spec[0] */
array_sample_beam_process_beam(unit, &front, vad_raw, vad_out, mask, &beam, &info);   /* SRP-PHAT, BF */
/* info: doa_raw/smooth/used in radians (NaN = none), bf_adaptive, delay */
/* then ac_stft_push(far_stft, far, far_spec) and
   ulcnet_nn_stage_begin_spec(stage, beam.spectrum, far_spec, {front.delay.changed, front.delay.solid}),
   nn_begin / nn_end / end_spec / ac_istft_push as above */
```

`mask[f]` nonzero: SRP uses bin `f`, BF freezes its adaptation on bin `f`;
NULL means all bins used and none frozen. `vad_raw` gates SRP, `vad_out` feeds the
DOA smoother and freezes BF adaptation. `front.lane_error_spec` are read-only views
valid until the next `process_front()`, `process_with_activity()`, `reset()` or
`destroy()`. `process_beam()` without a pending front returns `AEC_ERR_SEQUENCE`. The split adds
no state to the helper: the pool sizes and the layout version are unchanged.

## Calling sequence and parameters

```
App       Aec (array, residual = STATE)                   ArraySampleBeam      UlcnetNnStage + adapter       accelerator (board)
 |-- array_sample_default_config (ulcnet_example_unit_config()); aec_get_mem_requirements / aec_init_ex -->|
 |-- array_sample_beam_default_params; array_sample_beam_get_mem_size / _init(aec) -->|
 |-- ulcnet_grid_make(--sample-rate); ulcnet_nn_stage_mem_size(&grid) / ulcnet_nn_stage_init; ac_stft_init / ac_istft_init; adapter pool
 |   per hop: array_sample_beam_process_with_activity(mics, far, vad, vad, NULL, &beam, &info)   [or _process_front + mask/VAD + _process_beam]
 |            --> ac_stft_push(far) --> ulcnet_nn_stage_begin_spec(beam.spectrum, far_spec, {info.delay.changed, info.delay.solid})
 |            --> ulcnet_nn_stage_process(out_spec) (nn_begin == 1 --> run_accelerator(job) --> nn_end) --> ac_istft_push(out)
 |   no array_sample_beam_fuse() here (AEC_ERR_MODE); no RES, no NR
 |   refused call: array_sample_beam_reset + ulcnet_nn_stage_reset + ac_stft_reset + ac_istft_reset
 |-- array_sample_beam_destroy; aec_destroy; release the blocks
```

External parameters: the delay profile below, the grid (`--sample-rate 16000|48000`), `D` of the exported
graph, and the helper's array geometry and `num_angles` (the example uses the default 35 mm UCA; a product
sets the measured one through `ArraySampleBeamParams`). Internal: the helper's BF / DOA / VAD constants (section 0.3 of
[SRP／BF helper](../../../docs/reference/array/beam.md)),
`ULCNET_NN_STAGE_REPRIME_FRAMES`, the lane constants (derived inside the array).

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up — at the default matched/n=5
profile, that banner reads a 2,490,416 B unit pool (the array plus the helper; the 16 kHz row of the example's table is fft 512) plus a
6,320 B stage pool at 16 kHz and a 5,188,368 B unit pool plus a 12,464 B stage pool at 48 kHz
(`--sample-rate 48000`); the application's framing line follows (47,808 B at 16 kHz, 93,888 B at 48 kHz: the far `AcStft`, the output
`AcIstft` and two spectra); the adapter pool follows on its own line (14,864 B at 16 kHz,
29,200 B at 48 kHz; it follows the grid, not `D`). Read it from your
own build rather than copying it.
`n` is an init parameter, not a runtime setter —
changing it means re-querying the pool and re-initializing.

```sh
./4ch_alignulcnet                              # 16 kHz, matched, n=5 (default)
./4ch_alignulcnet --sample-rate 48000          # the 48 kHz / fft 1024 grid
./4ch_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./4ch_alignulcnet --delay-mode fixed --fixed-delay 1600
./4ch_alignulcnet --delay-mode external        # caller pre-aligns the far
```

Numbers are parsed by `common/example_cli.h` (`example_cli_parse_int`): `--delay-num-filters 3x`,
`--fixed-delay 12x`, `--sample-rate 16000x`, an empty string, a leading space or a value outside
`int` exit with 2, as does a `--sample-rate` that is not 16000 or 48000; plain decimals and a
leading `+` are accepted.

There is exactly ONE matched bank here: the array's shared estimator.
The four lane AECs run `EXTERNAL_ALIGNED` off its single aligned reference, so
each filter costs 5,728 bytes ONCE — not four times — and the lane pools inside the array
do not move with `n` at all (`aec_get_mem_breakdown()`: `lane_bytes` is the same for every `n`, `estimator_bytes` drops by 5,728 per filter).

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5. `0` is not "off" — use `--delay-mode fixed` (delay known at
bring-up) or `external` (upstream guarantees alignment) instead. A bulk delay
beyond the ceiling does not merely fail to lock: with any in-range early
reflection present the estimator can lock onto that instead, at full
confidence, and nothing in the delay seam distinguishes it from a correct
lock — see the known-delay tests.

The rules live in `pipelines/4ch/Makefile` (targets `4ch_alignulcnet`,
`test_4ch_ulcnet`); the `Makefile` in this directory forwards to it. From this directory:

```sh
make SIMD=0 test
make SIMD=1 test
```

`test` runs `test_4ch_ulcnet` (both grids from one binary), the skeleton at n=5 and n=1 on both
grids and the malformed-number checks above.
The array is part of `libaec.a`; the helper object, the stage
and the ULCNet pre/post objects are linked into the program.
