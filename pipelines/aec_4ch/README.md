# 4-ch shared-delay linear AEC shell around external SRP-PHAT/GSC

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
  constructor leaves `beamformer=None` and uses the explicit pre/post API.
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must return both its mono hop and complex weights with
shape `[4, n_freqs]`. With those weights, the post-beam error, mic and echo
spectra are coherent sums. The far-end spectrum remains the one shared digital
render reference—it is verified equal across lanes and is not spatially
weighted. The result can pass to the mono NR+RES seam without a fifth AEC.

## Explicit external-algorithm seam

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

Shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because post-BF RES has
  temporal state;
- a queued `PreBeamformerFrame` owns snapshots of all four contexts, so later
  linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight.

If the external GSC is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. An equally valid future
adapter is for the owner to apply the current coefficients (without adapting a
second copy) to the four echo-estimate streams and return the already projected
context. Returning only a mono waveform is insufficient for the traditional
RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently beamforms error/echo spectra, combines R2
with the same spatial weights, then runs exactly one stateful
`PostBeamResidualSuppressor` to calculate the mono gain used after NR.

One parity limitation remains explicit: `AecResContext` does not yet export
unbounded R2 or the complete stationarity/AecState surface. The post-beam RES
therefore uses bounded R2 for both gain inputs and omits the stationary mask.
This is structurally correct but **not bit-exact** to a future fully extracted
AEC3 post-beam RES API; it must be cohort-tuned before production sign-off.

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
