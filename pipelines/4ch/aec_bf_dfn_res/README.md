# 4-channel AEC + beamformer (bf) + RES + DeepFilterNet2

This example replaces the post-beam MMSE-LSA denoiser of the conventional four-channel
application (`4ch/`) with DeepFilterNet2 while keeping the shared delay estimator,
the four linear AEC lanes, the SRP-PHAT and beamformer (bf) glue, a post-beam AEC3 residual
suppressor and the comfort-noise recipe. The four-microphone flow is one `Aec` instance of the AEC library
(`aec.h`, `AecConfig.n_mics = AEC_ARRAY_MICS`): the four lanes, the shared `DelayAec`, the fusion buffers and
the residual gain stage are carved from one pool by `aec_init_ex()`; the application supplies none of them.
There is no DFN wrapper library either. `main.c` builds the array in one block with
`aec_get_mem_requirements()` + `aec_init_ex()` on `array_sample_default_config()` (residual `AEC_RESIDUAL_FULL`), adds the sample SRP-PHAT/BF helper
(`array_sample_beam.h`, in no archive), carves the P spectrum of the RES path (no denoiser)
and the DFN2 model stream (`stream/ainr/dfn2`, memory mode; at 16 kHz with the sample-rate bridge `stream/ainr/bridge` around it; sized and carved by `dfn/dfn_hop_pool.h`) separately,
forms P in `dfn_form()` (`aec_apply_gain(aec, NULL, P)`, optional `aec_apply_cng(aec, NULL, P)`), and `dfn/dfn_hop.h` runs
the network between the post-RES spectrum and the synthesis:

```text
mics/ref -> array_sample_beam_process(unit, mics, far, &beam, NULL)
            aec_process_linear() + aec_get_array(): shared delay, four lanes; helper: SRP-PHAT, beamformer (bf)
         -> array_sample_beam_fuse(unit, &v)         = aec_process_residual(aec, &{weights, E}) + aec_get_residual(aec, &v)
              E = beamformed error spectrum (v.error_spec)   -> DFN2 feature input
         -> G_res = the array's gain stage, applied by aec_apply_gain(aec, NULL, P)
         -> P = B * G_res (B = v.base_spec)         P = B * G_res (+ comfort noise) -> DFN2 applied here
                                                   (dfn_form(): aec_apply_gain + aec_apply_cng)
         -> dfn_hop_begin(&hop, v.error_spec, P)
         -> while (dfn_hop_nn_begin(&hop, &job) > 0) { run the model on the job; dfn_hop_nn_end(&hop, rc) }
         -> dfn_hop_end(&hop, out)                  -> enhanced P, frame t-L (L = M + D, 2 for the shipped pair),
                                                       the host's iFFT / OLA
```

The skeleton uses the helper's own SRP-PHAT and BF, so `array_sample_beam_fuse()` hands the array the
BF's spectrum and weights. An application with its own beamformer calls `aec_process_linear()` and
`aec_process_residual(aec, &{W, E})` instead. `dfn_hop_process(&hop, v.error_spec, P, out)` is the sequence of
the `dfn_hop_*` calls with the model callback (`dfn_hop_infer()` here) as the runner. The RES
strength is `aec_set_level(aec, level)` on the array. The array owns the RES gain stage and the
comfort-noise generator; the example owns the P buffer and the ISTFT in the `DfnHop`, and a refused call resets
the helper (and with it the array, gain stage and comfort-noise generator) and the `DfnHop` together (`dfn_reset()` in `main.c`). The
network estimates from the beamformed error and is applied to the post-RES
spectrum. The comfort-noise fill is kept as a switch but is OFF by
default in this variant: with it on, the additive fill perturbs P's phase in
the RES-cut bins, which breaks the estimate/apply phase consistency the
complex deep filter relies on (when on it travels through the network with the
rest of P and `atten_lim_db` bounds the attenuation of every bin). With no model callback
(or a failing one) the output is the
conventional chain with the denoiser off delayed: bit for bit by L = M + D hops at
48 kHz (two for the shipped lookahead pair; the acceptance gate in `tests/test_4ch_dfn.c` asserts it against the hand-written chain of
`4ch/tests/conv_chain_4ch.h`, with comfort noise off and on, across a real
shared-estimator delay re-lock, on a stimulus where a run without the array's gain stage provably
differs, and checks that the model stream is not reset on the shared aligner's delay change), and through the resampler round trip by the bridge delay at
16 kHz (the same test pins the delay with a near-end impulse, the frame
schedule, and, where the delay is a whole number of native samples, the pass-band error
against the conventional 16 kHz output, observed at about 1e-5 against a gate of 1e-3 for the shipped pair).

The flow runs on the product's own grid: 16 kHz (fft 256, hop 128) by
default, 16 kHz/512 (covered by `test_4ch_dfn`; the skeleton has no fft flag), or 48 kHz (the array's
rates, no 8 kHz). Only the DFN2 model runs
at 48 kHz: directly on the flow's spectra at 48 kHz (algorithmic latency
L + 1 hops; three hops, 32 ms, for the shipped pair), or through the rate bridge (`stream/ainr/bridge/ainr_bridge.h`)
at 16 kHz, which synthesises E and P on the native grid, resamples them up,
hands the hop the 48 kHz 1024/512 frames for the model stream and resamples the result down.
The four lanes, the beamformer (bf) seam, the RES and the comfort noise never leave
the product rate. The bridge adds `prefill + 32 + (L + 1) x 512 x sr / 48000` native samples (integer
division) to the host's own output: for the shipped pair 672 samples (42 ms) at 16 kHz with hop
128 and 629 (39 ms) with hop 256; total algorithmic
latency is that plus the host's WOLA hop (16 kHz, hop 128: 800 samples,
50 ms). A pair whose L + 1 is not a multiple of 3 has a fractional model delay that is floored
(pair 0/1: 501 / 458 samples). The model schedule per native hop is 0,1,1,1 frames at hop 128 and
1,2 at hop 256.

DFN2 grid and lookahead pair. The model grid is a run-time choice of `main.c`: `--ai-sample-rate <Hz>` / `--ai-fft-size <n>`
(default 48000 / 1024, a row of the two-row table in `dfn/dfn_example_args.h`) built with `dfn2_grid_make()` and handed to
`DfnHopPoolPlan.stream_cfg.grid` (`dfn_hop_pool_plan_init()`); the mask and deep-filter lookaheads M (0..2) and D (0..4) are fields of that grid and must equal the
pair the model was trained for. A host on the model's own rate (`--ai-sample-rate 16000 --ai-fft-size 256` with the 16 kHz core) runs
the model stream directly; any other host rate needs the bridge, which accepts the 48 kHz / 1024 model only (otherwise "pool query rejected",
exit 1). The `DFN2_MASK_LOOKAHEAD` / `DFN2_DF_LOOKAHEAD` make variables (default 1/1; this folder's Makefile forwards to
`4ch/Makefile`, which takes both, folds them into its build signature and has no `test-pairs` target) reach the test
program `test_4ch_dfn` only. The rules, the pipelines make
variables and `make -C pipelines test-pairs` are in [`../README.md`](../README.md). Only the shipped 1/1 pair has a
trained checkpoint; the numbers in this file are for it unless a pair is named.

Build and test from `pipelines/4ch/` (the Makefile that owns the rules):

```sh
make WERROR=1 libdfn_res_stage.a 4ch_aec_bf_dfn_res
make WERROR=1 test        # includes test_4ch_dfn and the skeleton
```

The `Makefile` in this directory is a thin forwarder: `make -C pipelines/4ch/aec_bf_dfn_res test`
builds and runs `test_4ch_dfn`, then the skeleton (default, `--delay-num-filters 1 --cng --atten-lim 12`,
`--sample-rate 48000`, `--ai-sample-rate 16000 --ai-fft-size 256`) and checks that `--sample-rate 16000Hz`, `--sample-rate 8000` and
`--atten-lim 12dB` exit with 2, and that a 32 kHz model grid (not in the table: the AEC has no 32 kHz grid) is a rejected argument (exit 2).

`libdfn_res_stage.a` holds the DFN2 model stream (`dfn2_stream.o`), the sample-rate bridge (`ainr_bridge.o`), the DFN2 host
pre/post objects, and the older stage and rate bridge adapter (`dfn_res_stage.o`, `dfn_rate_bridge.o`), which this example no longer calls; link it ahead of the producer archives (the SRP/BF archives of audiolib, `libaec` (which holds the array),
`libanr` and `libaudio_common`). The
sample objects `array_sample_beam.o` and `array_sample_driver.o` are linked into
the program, not archived. The board supplies the stateless model
callback (`DFN2Model`; its recurrent state is the accelerator's, reloaded with quantized zeros when
`job.state_generation` (`dfn_hop_state_generation()`) changes: a reset, a
refused commit or a run that started and failed; a callback that ran nothing returns `STREAM_RUN_NOT_STARTED` and moves
nothing) and the ERB matrices (`erb_fwd.bin` / `erb_inv.bin`
from `AINR/DeepFilterNet2/export_erb_matrix.py`; layout in [`../README.md`](../README.md), "Build and test"). `main.c` is the board
skeleton (delay-profile flags, `--sample-rate` (16000 default or 48000), `--ai-sample-rate` / `--ai-fft-size`,
`--erb-fwd` / `--erb-inv`, `--atten-lim`, `--cng`; the helper's default array geometry stands in for
the product's measured one; numbers are parsed strictly by `example_cli.h`, so values such as
`16000Hz` exit with 2; it
publishes the model I/O descriptor of the model grid, which a product replaces with
the one from its model's exported metadata).

The Python evaluator compares the original NR+RES tail, RES-only and the DFN
feature-source experiments on the same fused contexts. The C example uses
the beamformed-error source only; a raw-capture or single-lane estimate is
not phase-consistent with the post-beam spectrum, so those arms are
meaningful with a mask-only model (no deep filter) and stay experiments:

```sh
python pipelines/tools/eval_dfn2_4ch_inputs.py \
  --checkpoint /path/to/a-v6-checkpoint.pth \
  --output-dir dfn2-input-ab
```

`--identity-model` is a pipeline/resampler smoke gate only. It requires every
DFN arm to equal RES-only exactly and makes no speech-quality claim. The
contract gate refuses a checkpoint whose model/feature versions differ from the
current model-v6/feature-v5.

## Calling sequence and parameters

```
App          Aec (array)                   ArraySampleBeam   DfnHop + DFN2Stream (+ AinrBridge)   AcIstft (DFN2 grid rate; 48 kHz as shipped)
 |-- dfn2_grid_make (row of --ai-sample-rate / --ai-fft-size); load erb_fwd.bin / erb_inv.bin (--erb-fwd / --erb-inv)
 |-- array_sample_default_config; aec_get_mem_requirements; aec_init_ex -->|
 |-- array_sample_beam_default_params; array_sample_beam_get_mem_size / _init(aec) -->|
 |-- dfn_hop_pool_plan_init / dfn_hop_pool_plan / dfn_hop_pool_build: dfn2_stream_* (memory mode, freq_only) and the conversion buffers,
 |     then ac_istft_init + dfn_hop_init_direct (48 kHz) or ainr_bridge_* + dfn_hop_init_bridge (16 kHz)
 |   per hop: array_sample_beam_process --> array_sample_beam_fuse (the array's gain stage runs on E)
 |            --> v.post_reset_serial moved: ainr_bridge_reset_synthesis (16 kHz) | ac_istft_reset of the hop's synthesis dfn_hop_lookahead_frames() hops later (48 kHz) (the array reseeds its comfort noise) --> P = B * G_res (dfn_form())
 |            --> dfn_hop_begin(E, P) --> while (dfn_hop_nn_begin == 1) { model; dfn_hop_nn_end } --> dfn_hop_end --> out
 |   refused call: array_sample_beam_reset (helper + array) + dfn_hop_reset   (dfn_reset())
 |   never reset the hop on the shared aligner's delay change
 |-- array_sample_beam_destroy; aec_destroy; release the pool (the DFN tail owns nothing else)
```

The model estimates on the analysis beam `E` (`v.error_spec`) and is applied on `P`, formed over the rendering base
`B` (`v.base_spec`, see [the 4ch README](../../4ch/README.md), "Analysis and rendering spectra"). `B` is
`E` except on hops where a contributing lane's output guard fired; there `P` carries that lane's capture while the
estimate stays on `E`. A lane's post-chain reset reseeds the comfort noise and clears the synthesis. At 16 kHz that is
`ainr_bridge_reset_synthesis()` on the hop of the event (the bridge synthesises the host-grid spectra at the top of
its `begin()`; DFN2, the resamplers and the FIFOs keep running, no delay line). At 48 kHz the synthesis follows the model
by its lookahead: the frame built on the hop of the event is pushed `dfn_hop_lookahead_frames()` hops later, so the example
clears only its own synthesis (`ac_istft_reset()` on `hop.istft`) just before that push, through the per-hop shift register
of `dfn_reset_follow()` (`dfn/dfn_hop.h`; it starts at 0, and `main.c` and `tests/test_4ch_dfn.c` run the same function; the same
rule as `mono_aec_dfn_res`: the synthesis that receives the first post-reset frame is cleared just before it enters). DFN2 keeps
running and the hop is not reset. In this flow the serial moves only with an array reset (the lanes take an aligned far), which the
refused-call path pairs with a hop reset and a cleared shift register (`dfn_reset_follow_clear()`).

External parameters: the same board-skeleton flags as the mono variant (`--sample-rate` 16000 default or
48000, `--ai-sample-rate` / `--ai-fft-size`, delay profile, `--erb-fwd` / `--erb-inv`, `--atten-lim`, `--cng`), plus the array's `AecConfig`
(delay profile, `capture_proxy_channel`, `residual`) and the helper's `ArraySampleBeamParams`
(array geometry, `num_angles`; the skeleton uses the default 35 mm UCA, which a
product replaces with the measured one) and `aec_set_level(aec, ...)` on the array.
Internal: the helper's BF / DOA / VAD constants (section 0.3 of
[SRP／BF helper](../../../docs/reference/array/beam.md)), the model
grid and scale constants, the bridge FIR and schedule.

## Compute admission

`pipelines/tools/profile_dfn2_ops.py` counts the current streaming graph. The
v6 architecture has 2,134,971 trainable parameters and 3,307,136 major MACs
per 512-sample invocation, or 310.044 MMAC/s at 48 kHz. This excludes
elementwise operations, host pre/post DSP and memory traffic, so it is not a
CPU cycle claim. Weight storage alone is about 2.04 MiB int8, 4.07 MiB fp16,
or 8.14 MiB fp32.

For a 16-kHz source, the house FIR rate boundary adds approximately 1.584
MMAC/s per input channel and 1.552 MMAC/s on the mono output: 4.720 MMAC/s for
mono mic+render+output, or 9.472 MMAC/s for four mics+render+output. The DFN
model dominates this arithmetic budget; target-CPU-only admission still
requires the final int8/fp runtime and board cycle measurement.

Pool printed by the skeleton (`4ch_aec_bf_dfn_res`, matched delay, five-filter bank, NE10; the
array (four lanes, shared estimator, ring, fusion buffers and gain stage in one pool), the SRP/BF helper,
the P buffer, and the DFN tail of `dfn_hop_pool_bytes()`, each part 16-byte
aligned; shipped pair): 1,683,088 bytes at the 16 kHz default (array plus helper 1,459,184 bytes, the P buffer, and a DFN tail of
222,864 bytes: the model stream 92,688, the six conversion buffers 12,312 and the bridge 117,856) and
5,591,248 bytes at 48 kHz (array plus helper 5,435,184 bytes; DFN tail 151,952: the stream and the buffers, the synthesis 42,832
and the output spectrum 4,104). The stream is a `freq_only` one (`DFN2StreamConfig.freq_only`, set by `dfn_hop_pool_plan()`):
its pool holds the `DFN2Prepost` and no analysis, synthesis or PCM buffer. The array's own pool is 666,992 bytes at
16 kHz/256 (the lanes 570,880 of it; `residual = STATE` 561,456 bytes).
Model weights, accelerator activation/workspace memory and the recurrent state are external;
adding int8 weights alone brings the pool plus weights to about 3.7 MiB at
16 kHz.
