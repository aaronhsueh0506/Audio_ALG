# 4-ch shared-delay AEC + post-beam NR/RES

This is a separate zero-padding-free integration grid, not the conventional
mono pipeline's 20 ms / 10 ms frame grid and not an AIAEC neural-model
deployment path. The same boundary is available as a Python reference and as
a synchronous C library.

The production boundary is intentionally narrow:

```text
raw far reference ──> one shared matched-filter/delay estimator
                              |
                              v (one aligned reference)
mic 0 ──> linear AEC 0 ─┐                              ┌─> mono NR
mic 1 ──> linear AEC 1 ─┤  PRE HANDOFF   POST RESUME  ├─> gain fusion/OLA
mic 2 ──> linear AEC 2 ─┼─>[external SRP-PHAT/GSC]───┴─> mono RES
mic 3 ──> linear AEC 3 ─┘
```

Hard invariants:

- There is exactly **one** matched-filter instance. It uses a configurable
  capture proxy channel and the raw far reference.
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and reset together
  if that shared alignment changes.
- No NR, RES, neural model, or delay estimator is replicated per microphone.
- SRP-PHAT/GSC is externally owned and is not implemented here. The production
  Python constructor leaves `beamformer=None`; the C API likewise requires
  explicit pre/post calls.
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. The Python reference also accepts the external mono hop. The C
library instead synthesizes the mono hop internally from the same weighted AEC
spectra, so the output samples and the context cannot come from different
beamformer states. The far-end spectrum remains the one shared digital render
reference—it is verified equal across lanes and is not spatially weighted.
The result passes through one mono NR+RES path without a fifth AEC.

## C API

The deployable C seam is:

- [`4aec_nr_res.h`](4aec_nr_res.h): public config, lifecycle, pre/post
  calls, frame token, and structural audit accessors;
- [`4aec_nr_res.c`](4aec_nr_res.c): one shared `DelayAec3`, four linear
  `Aec` instances, coherent context projection, one `SuppressionGain`, one
  MMSE-LSA instance, and one final iFFT/OLA;
- [`test_4aec_nr_res.c`](test_4aec_nr_res.c): 16/48 kHz grid, lifecycle,
  ordering, token invalidation, invalid-config, weight, and finite-output
  acceptance tests.

```c
FourAecNrResConfig cfg;
FourAecNrResPreFrame pre;
FourAecNrRes *pipeline;

four_aec_nr_res_config_defaults(&cfg, 16000);
pipeline = four_aec_nr_res_create(&cfg);

/* microphones is interleaved [hop][4], reference is [hop]. */
four_aec_nr_res_process_pre(pipeline, microphones, reference, &pre);

/* External SRP-PHAT/GSC updates channel-major Complex[4][n_freqs].
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_gsc_update(pre.linear_interleaved, weights);

four_aec_nr_res_process_post(pipeline, &pre.token, weights, mono_output);
four_aec_nr_res_destroy(pipeline);
```

Only one pre frame may be in flight. `process_post()` must receive the exact
token returned by `process_pre()`; replay, cross-instance use, and a token
invalidated by `reset()` are rejected. Invalid weights leave the frame pending
so the caller may correct and retry them. Construction may allocate from the
heap, while both process calls perform no allocation.

Build the archive or run the complete C acceptance gate from
`Audio_ALG/pipelines`:

```bash
make lib4aec_nr_res.a
make test_4aec_nr_res      # build only the standalone 4-channel test binary
make test                 # build and run it plus mono/board-adapter tests
```

The archive contains this repository's wrapper object. A consumer links it
with `libaec.a`, `libmmse_lsa.a`, `libaudio_common.a`, and `-lm`, as the
Makefile test target demonstrates.

## Python reference seam

```python
from pipelines.aec_4ch import BeamformerFrame, FourChannelAecPipeline

pipeline = FourChannelAecPipeline()      # no beamformer is created

# 1. Our code: one matcher + four linear AEC lanes.
pre = pipeline.process_pre_beamformer(microphones_hop, far_hop)

# 2. Other owner's code: SRP-PHAT + GSC. This repository does not implement it.
mono_hop, effective_weights = external_srp_gsc.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights and calculate
#    one mono residual-suppression gain. The existing mono NR+RES seam consumes
#    post.beamformed and post.context.
post = pipeline.process_post_beamformer(
    pre,
    BeamformerFrame(samples=mono_hop, weights=effective_weights),
)
```

Python shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because post-BF RES has
  temporal state;
- a queued Python `PreBeamformerFrame` owns snapshots of all four contexts, so
  later linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight.

If the external GSC is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently beamforms error/echo spectra, combines R2
with the same spatial weights, then runs exactly one stateful
`PostBeamResidualSuppressor` to calculate the mono gain used after NR.

One parity limitation remains explicit: `AecResContext` does not yet export
unbounded R2 or the complete stationarity/AecState surface. The post-beam RES
therefore uses bounded R2 for both gain inputs and omits the stationary mask.
This is structurally correct but **not bit-exact** to a future fully extracted
AEC3 post-beam RES API; it must be cohort-tuned before production sign-off.

The C wrapper also does not yet expose a caller-owned-pool constructor, async
multi-frame queue, or a real SRP-PHAT/GSC implementation. Its automated tests
prove lifecycle, topology, supported grids, sequencing, and finite DSP output;
they do not replace real-array recordings, objective speech metrics, or
subjective quality sign-off with the external beamformer.

Default no-padding grids are `512/256 @ 16 kHz` and `1024/512 @ 48 kHz`, where
the frame length is the FFT length and hop is half the frame.

Reproduce the two checked-in recording tests from the `Audio_ALG` directory:

```bash
../.venv/bin/python -m pipelines.aec_4ch.evaluate_recordings
```

The command is an acceptance test, not only a report generator. It exits
nonzero unless both cases remain finite, preserve the `1 matcher / 4 linear /
1 post-beam RES` resource boundary, acquire a solid nonzero shared delay, and
finish within half a hop of the independent file-level delay measurement.
Use `--no-contract-check` only when inspecting a deliberately changed fixture.
The evaluator leaves the pipeline beamformer unconfigured and explicitly calls
pre → `EqualWeightBeamformer` → post for every frame. It tests our two sides of
the seam and does not claim to test the external SRP-PHAT/GSC.

`woman(ref).wav` and `man.wav` are two stems on one source timeline, while that
timeline and `unprocessed_4ch.wav` have different recording starts. The
evaluator estimates the one fixture offset from the near stem and applies it to
both. It deliberately does **not** align the far stem directly to its echo in
the mic, because doing so would erase the acoustic/system delay that the one
live shared matched filter is supposed to estimate.
