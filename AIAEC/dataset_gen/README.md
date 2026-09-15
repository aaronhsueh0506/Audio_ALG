# AEC dataset generation

Renders acoustic-echo scenarios as five separated WAV stems, then projects the
four signals required for training into `.pt` shards. This is the only AIAEC
dataset package. It reuses shared DSP from the separate `AINR/dataset_gen/` NR
generator rather than forking that DSP.

## The five stems

Every generated WAV is a `(5, T)` tensor in `STEM_ORDER`:

| # | stem | what it is |
|---|---|---|
| 0 | `far_render` | **X** — the far-end signal as the device rendered it, i.e. the AEC reference. Digital and clean: the loudspeaker's distortion happens *downstream* of this tap. |
| 1 | `near_speech` | **S** — the near talker through the full room RIR; retained in WAVs for mixing/audit, not copied into packed shards. |
| 2 | `near_target` | **S_early** — the same near talker and gain through the early/late-suppressed RIR; the common training target. |
| 3 | `mic_postclip` | **Y** — what a model actually receives, after capture clipping/AGC. |
| 4 | `linear_error` | **E** — frozen PBFDKF output, `Y - D_hat`. This is not oracle residual echo, and it is not unconditionally the filter's residual: the frontend runs the AEC3 post chain on the context seam, so a hop whose linear estimate the quality analyzer has not declared usable and whose residual carries more energy than the capture publishes the capture instead (`E == Y` there, crossfaded over 30 samples). |

**N** (`local_noise`, ambient noise at the mic) is no longer a persisted
stem: no current model task targets echo cancellation without denoising, so
it is audit-only now, like `echo` and `mic_preclip` always were (see below).

The signal model the corpus exists to serve:

```
Y = S + N + D          microphone
X                      far-end reference
D_hat = Y - E          frozen PBFDKF echo estimate, derived when needed
E     = Y - D_hat      stored linear error
R     = D - D_hat      residual echo — emerges, never a target
```

The default config renders each complete 20–30 second parent sequence first,
runs one stateful Python PBFDKF instance over `mic_postclip + far_render`, and
only then cuts all five WAV stems into 10-second chunks. The PBFDKF resets
between parent sequences and never at a chunk boundary.

The packer writes `(4,T)` tensors in `PACKED_STEM_ORDER`:
`far_render`, `mic_postclip`, `linear_error`, `near_target`. RES+NR candidates
read `E + X`; end-to-end candidates read `Y + X`; all target `S_early`
(denoised, dereverberated and echo-free). `D_hat` is derived as
`mic_postclip - linear_error` when required and is never stored separately.

**⚠ `echo` (D), `local_noise` (N) and `mic_preclip` (S+N+D, pre-clip/AGC) are
NOT stored.** No model task targets echo cancellation without denoising any
more, and no candidate sees oracle residual `R`. All three are still
**computed on every render** — `aec_dataset.AecSequenceRenderer.render()`
returns them under `RenderedSequence.audit` — so the corpus's central
invariants (`mic_preclip == S+N+D`, "echo really is a delayed copy of X") stay
verified at generation time; see `tests/test_aec_dataset.py`, which checks them
directly against the renderer rather than a packed shard. If you need `echo`/
`local_noise`/`mic_preclip` for a one-off analysis, call
`AecSequenceRenderer.render()` yourself — do not add them back to
`STEM_ORDER` for that.

Old six-channel WAVs/shards (with a separate `local_noise` stem) are rejected.
To upgrade an existing four-channel render without repeating speech/noise/RIR
mixing, run `rematerialize_linear_aec.py`; it reconstructs complete sequences
in `(sequence_id, chunk_index)` order from the filenames and rewrites the last
channel (`linear_error`) in place.

Existing five-channel WAV corpora need no regeneration for the four-channel
training contract: run `pack_aec_dataset.py` again. Old five-channel `.pt`
shards are intentionally rejected because their target semantics differ;
repacking reads the existing WAVs and drops only `near_speech`.

**⚠ That covers the channel layout, not the fifth channel's contents.** A corpus
rendered before the frozen frontend moved onto the context seam carries a
`linear_error` produced without the over-output capture guard, and the packer
stamps whatever contract `--config` builds today onto whatever audio it finds.
Rematerialize that channel first — see "Recomputing `linear_error` after a
frontend change".

`AecStems` gives these names; nothing indexes the channel axis by number.

```python
from AIAEC.dataset_gen import PackedAecDataset, build_model_view
ds = PackedAecDataset('data_aec/packed/train')
v = ds.stems_of(0)
view = build_model_view(v, 'DeepVQE_S', sample_rate=ds.sr)
mic, far, target = view.inputs['microphone'], view.inputs['far_end'], view.target
```

`build_spectral_model_view(view, grid)` converts that waveform contract into
the exact `[B,T,F]` keyword tensors accepted by each retained model.

## Metadata

**The rendered corpus is WAV and nothing else.** `<split>/seqs/SSSSSS_CCC.wav`
is the entire on-disk contract: no run `meta.json`, no per-sequence sidecar. A
packed clip therefore carries only where it came from:

| field | meaning |
|---|---|
| `sequence_id`, `chunk_index` | which parent sequence, and where in it — both read back out of the filename |

Everything else the shard needs is either fixed (`stems`, the channel order),
read from the audio (`sr`, chunk length) or rebuilt from `config.ini` at pack
time (`linear_aec`, `linear_aec_contract_hash` — the frozen PBFDKF contract,
which cannot be recovered from a WAV and which inference must reconstruct
exactly).

⚠ **What that costs.** The renderer still computes a full per-chunk
description — `speaker_id`, `far_speaker_id`, `noise_id`, `rir_id`, `room_id`,
`room_rir_count`, `device_id`, `ser_db`, `snr_db`, `erl_db`,
`bulk_delay_samples`, `delay_jitter`, `delay_step_samples`, `delay_step_at`, `sro_ppm`,
`nonlinear`, `clipped`, `agc`, `talk_mode`, `echo_mode`, `impairments`,
`acoustic_tails`, `path_motion`, `echo_path_moving`, `echo_path_waypoints`,
`echo_path_keyframes`, `echo_path_event_at`, `echo_path_gain_walk_db`,
`echo_path_mixture_depth`, `echo_path_position_correlation`,
`near_path_motion`, `near_path_mixture_depth`,
`near_path_position_correlation`, `path_position_redraws`,
`path_position_fallbacks`, `near_path_shared_positions`, `far_active`,
`near_active`, `scenario`,
`sequence_scenario`, `sequence_seed`,
`split` — but it is now visible only
in-process, on the `RenderedSequence` a worker hands back
(`tests/test_aec_dataset.py` reads it there). None of it reaches disk, so:

- a curriculum keyed on `scenario` has to measure the chunk instead, which the
  separated stems make possible;
- source-disjointness is audited at the renderer, not on the packed corpus;
- the packer can no longer tell WHICH config, seed or manifest
  produced a chunk. They check shape — chunk count, rate, length, channels,
  encoding — and nothing else. Resume into a directory only with the run that
  started it.

The clip-level facts below still hold of the audio itself, they are simply no
longer written down:

`clipped` and `agc` are separate distortions — one memoryless and
instantaneous, one a slow gain with memory — and a model that confuses them
fixes the wrong one. Together they are exact: `mic_postclip` differs from
`mic_preclip` **if and only if** one of the two is set (`mic_preclip` itself is
audit-only, see above — `tests/test_aec_dataset.py` checks this identity
against the renderer directly).

**⚠ `ser_db` / `snr_db` / `erl_db` are sequence-level.** They describe how the
whole configured parent sequence was set up. A single chunk departs from them
by a few dB (ERL) or by anything at all (SER/SNR), because a chunk in which the
near talker happens to be silent has no signal to define a ratio against.

**⚠ `±inf` is deliberate.** It marks a ratio that is undefined because one of
its two signals is absent, rather than a fabricated number that would silently
pass a threshold filter.

### Layered scenes

New corpora no longer choose one mutually-exclusive scenario. They draw four
orthogonal layers:

- `[talk_modes]`: `far_only`, `near_only`, `double_talk`, or
  `duplex_random` (relative categorical weights);
- `[echo_modes]`: normal echo, `ref_dropout`, or `far_active_no_echo`
  (conditional probabilities for far-capable sequences);
- `[impairments]`: independent path/capture events such as
  `nonlinear_spk`, `clipping_agc`, `delay_jitter`, `delay_step`, `sro`
  and `codec_mismatch`, plus the mutually exclusive **echo-path motion** draw
  (`slow_drift`, `movement`, `echo_path_change`) described below.
- `[acoustic_tails]`: independent low-probability operating points for
  120--300 ms bulk delay, −40 to −30 dBFS quiet references (with the ERL draw
  capped so the echo stays above the noise floor), and −10 to 0 dB
  strong-echo ERL. Ordinary sequences retain the original ranges.

This separation is intentional. A single categorical scenario cannot express
`double_talk` **and** nonlinear **and** clipped at once, so the difficult
intersection never existed. `[complex_cases] p_dt_stress_combo` guarantees a
small measurable tail containing DT + nonlinear loudspeaker + clipping/AGC —
path motion is deliberately not in that bundle, see below. Within that tail,
`[activity] dt_force_edge_overlap` pins DT to the leading edge of the first far
burst and trailing edge of the last, covering both a cold PBFDKF state and a
mature PBFDKF state late in the parent sequence without making ordinary DT
scripted. It does **not** create a 20--30 s carried neural state: current
trainers reset GRU state for each shuffled 10-second chunk.
`p_dt_acoustic_combo` independently guarantees a smaller DT tail containing
long delay + quiet far + strong echo. The three acoustic tails are also drawn
independently, so the corpus still contains short-delay strong-echo and
long-delay ordinary-level examples rather than learning one artificial bundle.

`[activity] p_far_then_near` covers a different failure: an ordinary echo path
adapts for several seconds before the near talker first appears. When selected,
chunk zero contains 0.5--4 s of far-only context, followed by near speech
through the end of that same training chunk. This spans both a relatively cold
and a mature linear-AEC state without treating one long script as the target.
Three equally sampled variants keep far continuously active, stop it 50--300 ms
before near onset, or restart it 0.5--1.5 s after near onset. The stopped modes
retain the complete 0.5--4 s active far pre-roll before that random turn-taking
gap, so they do not trade away AEC adaptation time. The continuous mode is
rendered as one schedule across near onset, avoiding an artificial speech-file
fade at the event. This curriculum is independent of
nonlinear/clipping stress and is recorded as `far_then_near_mode`,
`far_stop_sample`, `near_onset_sample`/`near_onset_sec`, and
`far_restart_sample`. It does not
overwrite a forced DT edge or a first-chunk path-motion event, whose own
guarantees would otherwise be lost.

`far_active_no_echo` is a hard negative: its far scheduler covers the complete
parent sequence — as back-to-back utterances, each drawing its own file, since
one whole-sequence run would be zero-padded to length and go silent after the
first utterance — while acoustic echo is exactly zero. The per-chunk label is
then measured rather than asserted, so no emitted chunk can silently degenerate
into the already-covered silent-reference case and keep the label anyway.

`scenario` remains a compatibility, **per-chunk** summary and cannot describe
all simultaneous conditions. A `ref_dropout` parent is mostly *not* in
dropout, and a path-changing parent has exactly one transition chunk. New
diagnostics must use `talk_mode`, `echo_mode`, `impairments` and
`acoustic_tails` in-process (or
measure the persisted separated stems). `sequence_scenario` remains only a
legacy summary.

**`ref_dropout` is load-bearing.** During a dropout the far end is genuinely
silent — `X == 0` **and** `D == 0` — so no model may hallucinate echo removal.
Every current candidate is a joint AEC+NR route and may still suppress `N`, so
`ref == 0 -> output == mic` is **not** a universal gate (that expectation
belonged to the now-retired AEC-only Align-CRUSE route, which targeted
`S+N`). `[dropout] ref_dropout_echo_continues_p` can make
the loudspeaker keep playing while the reference is lost, but that asks the
model to predict an echo from nothing, so it is **0 by default**: raising it
trains hallucination.

### The echo path moves

**The path is never frozen, and that is the point.** Measured on paired
static/movement far-end clips from the AEC-challenge blind set — 64 ms
frequency-domain Wiener path estimates over 300–4000 Hz in 2 s windows, after
delay compensation — the path's correlation with itself 1/2/4/8 s earlier, its
per-second relative change and its per-second level step are
`path_drift_metrics.CALIBRATION_TARGETS`. **That dict is the only copy of those
numbers**: it is what the calibration test asserts against, and restating it
here (or in a config comment) is how a re-calibration leaves five stale tables
behind. It holds two curves, `movement` for a device being moved and
`static_device` for one nobody touched; `CALIBRATION_TARGET_OF_MODE` says which
rendered mode is calibrated against which. The same module measures a rendered
pair, so the corpus and the captures are compared through one implementation —
including the delay search, which has to be wide enough for the whole delay a
pair can carry: outside its true delay the phase-transform peak is noise, and a
mislocked clip reads as violent movement rather than failing.

A device nobody touches still drifts — its correlation with its own path 8 s
earlier is well below 1.0 — and a moving one has largely forgotten its path by
then. A path that is exactly constant except at one instant is a path a filter
has to find once and then never again.

Motion is one **mutually exclusive** draw per far-capable sequence —
`p_slow_drift` (0.5), `p_movement` (0.15), `p_echo_path_change` (0.02), the
remainder still. Why the three cannot co-occur is on
`aec_dataset.PATH_MOTION_MODES`.

`slow_drift` and `movement` render a continuous trajectory over K = 3–4 RIRs
of the **same room**:

    echo = sum_k w_k(t) · conv(played, RIR_k),      sum_k w_k(t) = 1

with raised-cosine transitions between per-position mixtures
(`*_position_correlation` says how far a corner pulls the mixture toward one
position — see below; `*_segment_sec_*` how long a transition takes;
`*_dwell_*` how long it may hold still). Level is a **separate** bounded
dB-domain walk, because a real path's gain and its shape do not move together.
`echo_path_change` keeps the one-shot crossfade as a rare hard transition. All
of it is in `[path_motion]`; the drawn keyframes, waypoint RIR ids, level-walk
spread, solved mixture depth and reached path correlation are recorded per
chunk.

**⚠ The calibrated knob is a path CORRELATION, not a mixture depth.**
`[path_motion] *_position_correlation` states the correlation the path must
keep between the trajectory's anchor mixture and a corner, in the same
300–4000 Hz band **and through the same 64 ms analysis frame** the estimator
reads, and the renderer solves the depth that reaches it from the drawn room's
responses — per sequence. Why a depth is not a quantity a corpus can be
calibrated in is on `aec_dataset.solve_mixture_depth`. `movement`'s 0.87 solves
to a median depth of 0.50 on a 60 ms pool, 0.38 on a 200 ms one, 0.40 at
350 ms, 0.39 at 600 ms and 0.36 at 1 s (`slow_drift`'s 0.965 to
0.24/0.18/0.19/0.19/0.18); all of them render the same axis to within the
residual `RT60_POOL_TOLERANCE` records. Both numbers land in the chunk
metadata (`echo_path_mixture_depth`, `echo_path_position_correlation`, and
`near_path_*` for the near talker's own trajectory).

**⚠ A room whose positions cannot reach the target does not host the mode.**
Eligibility is decided before the room is drawn rather than discovered during a
render, and `AecSequenceRenderer.can_host` owns what it asks of a room. Rooms
that cannot host are excluded, printed as such in the preflight, and counted
against the two-room rule below; the renderer then asserts what the draw
guarantees, so a corpus whose movement axis is shallower than it asked for is
impossible rather than merely visible in the metadata.

**⚠ Two things bound how far a synthetic corpus can decorrelate**, and both are
worth knowing before reading a measurement:

- **the RIR pool.** Positions in one room share their direct path and early
  reflections, so a pool whose rooms are near-anechoic has a correlation floor
  no weight schedule can go below. The solve above is what keeps that a floor
  rather than a silent re-calibration — it reports the correlation each room
  actually reached — but no depth can pass it. Measure your own pool rather
  than assuming: `path_drift_metrics.py` takes `(far, echo)` and returns the
  table above, and `position_gram` takes the room's RIRs and returns the matrix
  the depth is solved from.
- **position-dependent propagation delay is not inferred from RIR file
  offsets.** `prepare_rir` peak-aligns every waypoint, so this trajectory
  changes path shape and level but not time of flight. The RIR files do not
  share a guaranteed recording time origin; preserving their raw peak offsets
  would silently turn file trimming into physical movement. Timing variation
  is instead carried by the independent, countable `delay_jitter`,
  `delay_step`, and `sro` axes, while `bulk_delay_samples` keeps describing the
  audio.

Measured back out of the rendered 30 s corpus the calibration test builds,
through the **shipped loudspeaker population** and a RIR pool drawn from the
shipped `[rir]` RT60 range — the models, drives and rooms the corpus actually
generates with, because that is what the trajectory has to reproduce the
measured curve through. What THIS build measures, per mode and per lag, the
calibration tests print themselves — the `_report` rows of
`tests/test_aec_dataset.py` under `pytest -s`, over `MOTION_SEQUENCES`
sequences, a count the suite sizes from the bootstrapped spread of its own
medians. A table here would be a copy that goes stale at the next
re-calibration.

Capture clipping and the AGC are *not* part of that: the estimator reads the
echo before the microphone stage, so `p_clipping` and `p_agc` cannot move any
of those numbers. The band each lag is held to is in the test
(`CALIBRATION_BAND`), sized from the bootstrapped spread of the median at those
sequence counts — one sequence reads ~0.998 at every lag with a linear
loudspeaker and 0.4–0.9 with a hard-clipping one, and the pool's rooms differ
in RT60 on top of that, so the median needs a hundred-odd sequences before it
is stable to the band. The 2 s and 8 s lags are banded wider than the rest
because their spread stops shrinking with the count: the short lags are the
loudspeaker's rather than the trajectory's and inherit the whole device draw.

**⚠ What the band can and cannot separate on that population.** The one-shot
crossfade is NOT rejected by it: with a distorting loudspeaker it lands inside
the band at every lag, because the device supplies decorrelation of its own. A
drift mode whose trajectory has stopped is not rejected either — the level walk
through the same distortion supplies most of the curve, and the mode ORDERING
(asserted at 4 s and 8 s) puts a frozen drift mode exactly where a moving one
belongs. So the statements about the trajectory are made with the loudspeaker
axis removed (`nonlinear_models = linear`): still 0.991/0.980/0.979/0.979,
one-shot 0.989/0.975/0.972/0.954, `slow_drift` 0.988/0.970/0.950/0.937,
`movement` 0.979/0.937/0.871/0.822. There a frozen path fails the band at two
lags against both curves and the crossfade at three against the `movement`
curve, and the two drift modes have to clear a long-lag ceiling
(`ISOLATED_TRAJECTORY_MAX`) that both mutations miss at both lags: a frozen
trajectory, and a path correlation target halfway to 1.0 — a corpus whose
positions barely differ. That is what makes "the trajectory moved" a tested
claim rather than an inference from a curve the loudspeaker could have produced
on its own. Every ceiling is held above the bootstrapped spread of its own
green median by an assertion, not by a comment.

**⚠ The still-path shortfall is the loudspeaker population, and the device
axis is calibrated separately from this one.** Read the still-path row those
tests print against `static_device` in
`path_drift_metrics.CALIBRATION_TARGETS`, which is the curve
`slow_drift` is held to: through the shipped drives a path that never moves
already reads far under the estimator's own floor on the calibration pool
itself, and at the short lags down to that curve, before the trajectory does
anything — which is where `slow_drift`'s own remaining deviation is. With
`nonlinear_models = linear` the same pool puts the still path back on the
estimator floor, so the gap is the drive population and not the rooms, the
estimator or the trajectory — and no `*_position_correlation` can close it at
any RT60, because a trajectory can only move the path further from itself,
never back toward the curve. The suite renders the still path both ways and
reports the two readings side by side, the mixture one with its bootstrap
interval (`test_a_still_path_stays_correlated_with_itself`): that population is
bimodal — a near-linear device reads on the floor and a hard-clipping one far
under it — so its median swings with the count and is reported rather than
gated. Re-fitting the drive population against the untouched-device curve is a
separate calibration and is not what the trajectory knobs are for.

**⚠ The pool the axis is fitted on stops short of `[rir] rt60_max`,** because
that floor deepens with RT60 — a tail several times the estimator's 64 ms
analysis frame is not one frequency response — while the room-to-room spread
out there is as large as the RT60 trend itself. Fitting there would calibrate
the trajectory against the device floor and the room draw instead of against
the path. A separate check (`RT60_POOLS`, 60 ms to 1 s) holds the *same* config
to the same axis across the whole range, isolated, to within the residual
`RT60_POOL_TOLERANCE` records,
and reports the shipped population on one pool above the span so what it does
there stays visible.

**⚠ The loudspeaker population is stratified, not drawn per device.** A seeded
permutation of `[devices] nonlinear_models` fills `device_ids` before the
remainder is drawn, so every configured model reaches every corpus seed
whenever there are at least as many ids as models. Drawing each id's model
independently leaves a third of the seeds with no linear device at all and the
rarer models missing from most of them, and since this axis is calibrated
*through* that population, which models it contains is part of what the corpus
is. What still moves between seeds is each id's **drive**: at the seeds the
test checks, `movement`'s 4 s median spans 0.024–0.139 from its target, which
is what the seed check's own allowance covers. The **split** is model-aware for
the same reason: `device_split = disjoint` renders one split's ids, so a
held-out id drawn without regard to its model takes that model out of training
whenever it is the only id carrying it (148 of 200 seeds at the shipped 8 ids
over 7 models, for a draw made on its own; through `build_manifest` the device
split follows three source splits on the same generator and the count moves
with them). Train is therefore filled with one id per model first and val
drawn from the rest; a split too small to hold every model falls back to a
plain permutation and is stated in the preflight instead.

**⚠ What `device_split = disjoint` holds out is therefore a device IDENTITY,
not a loudspeaker MODEL.** While train keeps every model, val's id carries its
own EQ, drive and level draw of a nonlinearity family train also has: at the
shipped 8 ids over 7 models the val id's model is one train holds at every
seed. Validation still answers "an unseen loudspeaker", but not "an unseen
distortion family" — say which one you mean when you report the score. Changing
which id is held out changes the corpus, so `MANIFEST_VERSION` moves with it
and an existing `manifest.json` is refused rather than reused; rebuild it with
`--rebuild-manifest`.

**⚠ The near-plateau between 2 s and 4 s in the reference curve is not
reachable from a trajectory, and trying is a trap.** It is the signature of an
incoherent component that fully decorrelates within 2 s and then stops
contributing. Anything faster than the estimator's own 2 s window is *averaged*,
not resolved, so adding a fast weight component RAISES this correlation instead
of lowering it. That part of the real number is the device, not the path —
which is why the calibration is run through the shipped device population
rather than around it.

**⚠ The measured per-second level step is not a pure level measurement**, so
it is not what the level walk is tuned against — `path_drift_metrics`'s own
`gain_step_db` says what the statistic is and what tuning against it would
cost. `[path_motion]`'s `*_gain_sigma_db_per_sec` and `*_gain_clamp_db` are set
by the measured level **spread** for a moving device instead, and the walk is
mean-reverting (`gain_recentre_sec`; `aec_dataset.gain_walk_db` says why a
merely-clipped walk is not enough). As rendered it spreads what
`LEVEL_SPREAD_DB` records per mode, and both the step and that spread are
banded there.
The spread is banded separately because the step cannot stand in for it: with
`slow_drift`'s walk turned off entirely the step barely moves against its
`static_device` target in `path_drift_metrics.CALIBRATION_TARGETS` — it stays
inside its own sampling spread — while the walk's p5–p95 goes to zero. (On
`movement` the step does notice.)
The estimator's `relative_change` is reported but deliberately not banded: no
level or trajectory setting the axis admits moves it more than 0.13 from the
measured 0.29, so a band loose enough to pass a green generator rejects only a
completely frozen path, which the correlation curve already rejects on more
evidence.

**DT movement.** A sequence with a near talker moves that talker's path too,
with the same trajectory shape at `near_slowdown` × the timescale and
`near_gain_scale` × the level swing: a person shifts in a chair, they do not
walk the way a hand-held loudspeaker does. `p_dt_stress_combo` forces
`nonlinear_spk` + `clipping_agc` only — motion is its own axis on 67% of
compatible echo paths at the shipped defaults, so bundling it there would make
inseparable from "DT while the loudspeaker distorts", and the CLI census counts
`dt+movement` and `dt+slow_drift` separately.

The CLI also prints the path correlation the rendered mixtures reached against
the one the config asks for, per mode: they can only differ by the solver's own
tolerance, since a trajectory only ever renders positions that reach the
target, so that row is the audit of the eligibility rule. Like the `Path
motion` row below it, it covers **the sequences this run rendered** — the reach
of a sequence an earlier run wrote is not on disk — and a fully resumed run
prints a line saying the audit is unavailable rather than nothing. A run that
had to draw a position set more than once to find one that reaches says so on
its own line; frequent re-draws mean the eligible rooms hold mostly
near-duplicate positions.

A second line counts the trajectories that **ran out of draws** and rendered
their room's certified set, and the two are separate because they cost
different things. A re-draw costs a Gram lookup and no audio. A fallback costs
variety: a room has one certified set per waypoint count, so every sequence
counted there renders one of a handful of trajectories however many sequences
the corpus holds — a pool whose eligible rooms are mostly near-duplicate
positions can therefore be correct on every reach and still repeat itself. The
near talker keeps its preference through that fallback (the certificate is
taken over the room minus the loudspeaker's positions), so a fallback does not
put the two paths on the same room response unless nothing else in the room
reaches — and when that happens a third line counts the sequences whose near
talker shares a position with the loudspeaker. All three position rows cover
the sequences this run rendered, like the reach row: a fully resumed run
prints that the position draws were not audited rather than nothing.

`Loudspeakers` is the population row: how many device ids this split holds, how
many of `[devices] nonlinear_models` they realise, and how many ids carry each
model. It is printed because the movement axis is calibrated *through* that
population and a source-disjoint split renders a subset of the id list, so
which models the split realises is a property of the split rather than of
`[devices]`. A split holding fewer ids than there are models cannot carry the
whole population however it is drawn, and the row is followed by a warning
naming the missing models.

**⚠ The `Path motion` census counts the mode each sequence was RENDERED with —
for the sequences this run rendered.** A sequence whose drawn motion did not
fit renders still, so counting the plan would report an axis the audio does not
carry; but nothing about a finished sequence is persisted beyond its chunk WAVs
(that is the on-disk contract, see below), so a `--resume` run cannot recover
the rendered mode of a sequence an earlier run wrote and falls back to the
planned one for those. The printed row says which part is which, and the
downgrade counter therefore covers this run's slice only. On the shipped values
nothing downgrades at all (the one-shot switch always finds room for its 1 s
window in a ≥ 20 s sequence), so the gap is zero there; a config with a large
`far_active_after_event_sec` is where it would matter, and a single-pass run
counts it exactly.

**Delay steps are NOT movement.** Delay jumps > 20 ms were measured in 53 % of
moving *and* 63 % of static clips: they are device/buffer events. `delay_step`
is therefore an independent impairment — one signed re-timing of the whole
echo at a random instant, `[echo_path] delay_step_ms_*` — so nothing in the
corpus lets "the path moved" be inferred from "the delay jumped". Its range
stacks on the long-delay tail and the jitter, and the sum has to stay inside
the frozen n=5 matched-filter reach (`linear_aec.MATCHED_REACH_MS`);
`plan_sequences()` refuses a config that breaks it. How the sign and the
magnitude are drawn, and what floors the rendered delay end to end through
`delay_jitter`, is on `aec_dataset._draw_delay_step`. The one thing that moves
the rendered delay past that floor is `sro`, which pulls the two clocks apart
by ppm × elapsed time (1.8 ms at 60 ppm over 30 s): two clocks running apart
genuinely do move the echo.
The instant is uniform except for a 0.5 s margin at each end
(`aec_dataset.DELAY_STEP_EDGE_MARGIN_SEC`), so both sides of the step can be
measured.

**`echo_path_moving` is measured, not asserted.** A chunk carries the label
when the largest within-chunk swing of any single weight reaches
`moving_label_weight_delta` **and** the chunk's own reference is measurably
active; `aec_dataset.moving_chunks` owns what that excursion is and why the
level walk is deliberately not part of it.

**⚠ How much that label carries depends on `chunk_sec`.** A dwell lasts at most
`*_dwell_sec_max` (1.5 s for `movement`, 4 s for `slow_drift`), so at the
shipped 10 s chunk the label says exactly what `path_motion` and `far_active`
already say, and it separates travelling chunks from dwelling ones only below
that. The label test asserts both regimes so neither claim drifts, and
`echo_path_change` still marks the one switch chunk.

**⚠ An event label displaces the activity label.** A drifting sequence's event
covers most of its chunks, so `scenario` alone would no longer show that a
moving chunk is double talk. Every chunk therefore also records measured
`far_active` / `near_active` booleans; filter a DT curriculum on those, not on
the string.

**⚠ A trajectory needs a room with at least `waypoints_min` RIR files, and
that is an inherent cue.** Every waypoint stays in the same room (a cross-room
waypoint would leave the near talker behind in the old room — the acoustic
"this is echo" leak the same-room invariant exists to prevent), so a moving
sequence can only ever come from a room that holds enough positions **whose
responses differ enough to reach `*_position_correlation`**, and a single-RIR
room only ever renders still. No draw rule removes that: drawing
every sequence from the whole pool and downgrading the ones that land in a
sparse room leaves `P(moving | sparse room) = 0` exactly as before, and pays
for it by losing most of the motion axis. So a sequence that draws a room too
sparse for its trajectory **draws again among the rooms that can hold it**, and
the rendered motion share is the planned one. What is done about the cue is to
make it countable: every chunk records `room_rir_count` beside `room_id`,
`path_motion` is the **rendered** mode, and `gen_aec_dataset.py` prints the
eligible-room share per mode before any worker starts — with the rooms it
excluded for too few positions and for positions too alike counted
separately, the second measured on the set the renderer would be handed if its
draws ran out (so it is the room's renderability, not a proof that no reaching
set exists in it) — and **refuses** a split where fewer than two rooms can host
a mode it *can* draw — with one, the trajectory would identify the room. A
split holding a single room altogether is exempt: the room is then constant
and identifies nothing. The refusal reads
every motion probability in the config rather than only the modes a plan
happened to draw, so a short `--hours` trial cannot pass a manifest the full
run refuses. What the re-draw costs is the other side of the cue: on a sparse
pool (3 of 40 rooms eligible) every moving sequence comes from those three
while still ones stay uniform over all 40, i.e. `P(moving | eligible room)` is
~93 %. That is the trade this design accepts — a strong forward cue on rich
pools in exchange for the rendered motion share equalling the planned one.

**⚠ The renderer** — not the planner — guarantees `far_active_after_event_sec`
of far-end activity **inside the window** that follows the event, and both
event kinds are placed where that window fits (`_switch_point`,
`_ensure_far_activity_after`). What they do when nothing fits differs, and that
is the trade: a one-shot switch that cannot be placed renders **still** and
records no event, rather than labelling an event the audio cannot support
(`gen_aec_dataset.py` counts those), while a trajectory reports its last
transition anyway and the window shrinks to what is left — the sequence keeps
its motion, which is real, and the guarantee is the part that gives. With the
shipped values that case cannot arise (a drift sequence's first transition
starts at most `*_dwell_sec_max` into a ≥ 20 s sequence, against a 1 s window);
a `far_active_after_event_sec` near the sequence length reaches it. A reference
dropout is steered away from the same window and shrinks to fit beside it
(`_dropout_placement`), with the chunk's measured `far_active` recording what
happened either way.

Note that `p_ref_dropout` is conditional per far-capable *sequence*; the
per-*chunk* share is smaller. If the idle term needs more, lengthen the dropouts
(`ref_dropout_chunks_max`) rather than adding sequences that are mostly not
dropouts. `near_only` supplies idle chunks too, but only `ref_dropout`
contains the *transition* into and out of idle, which is the hard part.

## Train/validation split

The selected training protocol generates one unified pool (`--split all`) and
uses `training_common.split_dataset_by_sample` after packing. A dedicated
seeded generator randomly assigns individual 10-second chunk indices, so the
split is reproducible, disjoint, and covers the whole corpus. Different chunks
from the same sequence, speaker, RIR, or device may intentionally straddle
train and validation. The train loader reshuffles every epoch; validation does
not shuffle.

This validation score is useful for optimization progress, not a
source-generalisation claim. Use a separately generated source-disjoint corpus
(`--split train`/`val`) or held-out real recordings for that measurement.
`manifest.py` retains the optional source-disjoint generator for this purpose.
The split is built deterministically in memory by default, so generation emits
only WAVs. Pass an explicit `--manifest PATH` only when the source directories
may change between the separate train/val runs and the split must be frozen on
disk.
Every checkpoint stores the dataset fingerprint, split seed/fraction, complete
train/validation indices, and PBFDKF contract so resume cannot silently change
the comparison.

## Why sequences are long

Parent sequences are 20–30 s by default, cut into consecutive fixed-length chunks that
share a `sequence_id` and carry an increasing `chunk_index`.

Long sequences are still required because PBFDKF adaptation from cold,
echo-path changes, and drift must happen before the last channel is cut. The packer
keeps `(sequence_id, chunk_index)` order for deterministic reconstruction and
streaming evaluation. Training itself treats chunks as independent shuffled
samples; `SequenceChunkSampler` remains only as an evaluation utility and is
not used by any trainer.

### Migrating an existing corpus

Changing only `linear_error` cannot create the newly composed acoustic scenes:
the four source stems already encode talk activity, path movement,
nonlinearity, capture clipping, bulk delay, far level and ERL. In particular,
`rematerialize_linear_aec.py` cannot add the `[acoustic_tails]` mixture or the
`[path_motion]` trajectory. Generate
a new WAV corpus in a new output directory, pack it, and train a new checkpoint.
Do **not** use `--resume` into an old corpus: WAV filenames carry no config
fingerprint, so shape-compatible old chunks would be accepted even though they
were rendered by the previous planner.

## Files

| File | Role |
|---|---|
| `aec_dataset.py` | the scenario simulator: nonlinearity, echo path, delay/jitter, SRO, dropout, AGC |
| `manifest.py` | unified/source-disjoint source manifest |
| `seq_layout.py` | the on-disk naming rules (`SSSSSS_CCC.wav`, temp names, scans) that the generator, packer and re-materializer all agree on |
| `gen_aec_dataset.py` | CLI — renders complete sequences to 5-channel WAV chunks |
| `linear_aec.py` | frozen PBFDKF contract and full-sequence materializer |
| `rematerialize_linear_aec.py` | rebuilds the last channel from existing four/five-channel WAVs |
| `materialize_pair.py` | **diagnostic**: one loose mic/far pair to a `linear_error` WAV, contract taken from the runtime |
| `pack_aec_dataset.py` | projects the five-channel WAVs into four-channel `.pt` shards (`--dtype float16` halves shard size and is safe to train on: every trainer widens to float32 at the device move, so only the stored dtype changes) |
| `packed_aec_dataset.py` | `PackedAecDataset`, returning `(stems, meta)` |
| `path_drift_metrics.py` | per-second Wiener path estimate and the drift statistics the movement axis is calibrated against |
| `aec_features.py` | **the shared module the model projects import** |
| `config.example.ini` | every knob, documented (16 kHz) |
| `config.example.48k.ini` | the same file with the 48 kHz recipe already applied; the tests derive it from the recipe above and compare key by key, so the two cannot drift apart |
| `tests/` | the invariants a consumer cannot detect being broken |

`aec_features.py` owns `AecGrid`, `stft`/`istft`, `alpha_from_tau`, `AecStems`,
`STEM_ORDER`, `PACKED_STEM_ORDER` and `SequenceChunkSampler`. **⚠ A model project that re-declares
any of these is opting out of the comparison** — the same failure that
`AINR/tests/test_bakeoff_protocol.py` already guards for the NR split, where a
5%-vs-10% divergence meant two models were compared on different corpora.

### The signal grid

16 kHz first target: `n_fft = 512`, `win_len = 512`, `hop_len = 256`,
`n_freqs = 257`, sqrt-Hann (periodic), 50 % overlap (COLA), 62.5 fps. The
48 kHz grid is `sr = 48000, n_fft = 1024, win_len = 1024, hop_len = 512` —
`AecGrid` derives `n_freqs` and `frame_rate`, and rejects a hop that is not
`win_len/2` or a `win_len` different from `n_fft`.

**⚠ `[signal]` is not the whole rate change.** The linear-AEC frontend's
`(frame, hop)` is frozen per rate in `linear_aec.py`'s
`FROZEN_FRAME_HOP_BY_SR`, so `[sequence] chunk_sec` must satisfy
`round(chunk_sec * sr) % hop == 0` — the shipped `10.0` is exact at 16 kHz and
not at 48 kHz, where whole seconds must be multiples of 4 (`8.0` is exact on
both grids and keeps the same 2–3 chunks per sequence). `[codec]
source_sr_values` is rate-dependent as well. The complete 48 kHz recipe
is at the top of `config.example.ini`, and `config.example.48k.ini` is that
recipe already applied — copy it instead of re-deriving the values by hand.
Generation refuses an inexact `chunk_sec` in its config preflight, before the
source/RIR scan.

**⚠ No frame counts, EMA coefficients or window lengths are hardcoded
anywhere.** Time constants are given in seconds and converted with
`alpha_from_tau(tau_sec, hop_len, sr) = exp(-hop_len / (sr · tau_sec))`. A
literal `0.92` in a config would be 191 ms on one grid and 128 ms on the other,
so the 48 kHz variant would quietly become a different algorithm. Use
`grid.n_frames(n_samples)` instead of writing a frame count.

## Usage

**Selected training protocol — unified pool and random chunk split:**

```bash
cp AIAEC/dataset_gen/config.example.ini AIAEC/dataset_gen/config.ini
# edit [paths] speech_dir / noise_dir / rir_dir, and [devices] device_ids
# optionally also [paths] far_speech_dir, for an independent far-end
# reference corpus that never shares a file/speaker with speech_dir

python3 -m AIAEC.dataset_gen.gen_aec_dataset \
    --config AIAEC/dataset_gen/config.ini --output data_aec \
    --hours 100 --split all --workers 4 --seed 42

python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all
```

⚠ The packer needs `--config` because the frozen linear-AEC contract that
produced the `linear_error` stem cannot be recovered from a WAV, and inference
has to construct the same one. Pass the config the corpus was generated with;
nothing cross-checks that claim any more.

⚠ `--output` must not already contain `shard_*.pt`. Loading a packed directory
takes every `shard_*.pt` in it and there is no index file naming this pack's own
shards, so a leftover from an earlier pack would silently join the corpus. Use
`--overwrite` to replace them deliberately. New shards are staged under
`.pt.tmp` names and published only after every WAV passes validation; a
validation/serialization failure keeps the previous pack intact.

⚠ Every current trainer runs at 16 kHz/512. A trainer on a different grid
would need a SEPARATE `config.ini` carrying the whole recipe for that rate
(`[signal]` alone is not enough — start from `config.example.48k.ini`, which
carries the whole 48 kHz recipe) and its OWN
`--output` (e.g. `data_aec_16k` / `data_aec_48k`, matching that trainer's
`packed_dir`): generating a second rate into the same `--output` as the first
is refused once chunk WAVs exist, since the directory is not namespaced by
sample rate. The rate is a config property with no CLI override, precisely so
that the config the packer is later handed cannot disagree with the audio.

Layout:

```
data_aec/
  all/seqs/000000_000.wav       5-channel chunk, channels = STEM_ORDER
  all/seqs/000000_001.wav
  packed/all/shard_00000.pt     4-channel tensor, channels = PACKED_STEM_ORDER
```

No JSON is created in the default flow. An explicit `--manifest PATH` is the
only opt-in exception, used to freeze a source-disjoint split; packing and
training never read it.

The four trainers read `packed/all` and create the deterministic random chunk
split from `[data] val_fraction` and the training seed.

The contract records the AEC's identity three ways, and they are used for
different things:

| Field | Scope | Used by |
|---|---|---|
| `aec_commit`, `aec_source_hash` | raw-text **provenance** | `fingerprint()` → `--resume`, packing, integrity |
| `aec_behavior_hash` | normalized-AST **behaviour** | `require_linear_aec_contract` → materialization + inference |
| `behavior_hash_schema` | which canonicalizer produced the hash | compared alongside it, so a serializer change reports itself by name |

`aec_behavior_hash` hashes the parsed AST with docstrings stripped, so a comment
reflow, docstring reword or reindent under `lib/aec/python` does **not**
invalidate existing shards or checkpoints, while any change to an expression,
constant or control-flow path does — and fails closed.

**It must not depend on the interpreter, and getting that right needed a custom
serializer.** The first implementation used `ast.dump()`. Python 3.13 changed
`ast.dump` to omit fields equal to their default, so the same 48 files digest to
`89b866cd` under 3.9 and `402acc1a` under 3.14 with no code difference at all —
a dataset generated under one interpreter would be refused by training under
another, and a checkpoint would become unloadable on a Python upgrade alone.
`aec_behavior_hash.py` therefore canonicalizes the tree itself (`_canon_ast`),
applying that same "drop empty fields" rule uniformly on every version. The rule
also absorbs fields that simply do not exist on older versions — `type_params`
(PEP 695) is absent on 3.9 and `[]` on 3.14, so both emit nothing — while a
field that is genuinely *used* is non-empty, is emitted, and does change the
hash. Nothing is dropped silently.

That module is deliberately free of third-party imports, so the parity test can
run it under every CPython on the machine.
`tests/test_linear_aec_behavior_hash.py` asserts they all agree, and runs a
control that reproduces the old `ast.dump` path and asserts it *disagrees* — the
stability claim would otherwise pass vacuously on a single-interpreter machine.

It is a hash rather than a hand-maintained `behavior_version` on purpose: every
other compared field is either a `__post_init__` literal or echoed out of the
recorded contract by both call sites, so a version integer would be the same
constant on both sides and could never differ. That tautology shipped once
(2026-08-06) and `test_contract_comparison_is_not_vacuous` now guards it.

Scope: `aec_behavior_hash` covers `aec.py` plus everything under `modules/` —
the code `LinearAecProcessor` can actually reach. `diag/`, `tests/` and the
bench/eval tooling are excluded, and the module-level `__version__` assignment
is stripped before hashing, so editing a test, a golden generator or the release
version cannot strand a checkpoint. `aec_source_hash` still covers every
Python file under `lib/aec/python`; unlike the 48-file signal-path scope, that
provenance count intentionally grows when tests or diagnostics are added.

**Recomputing `linear_error` after a frontend change.** The source WAVs do not
need re-rendering — only the fifth channel does, then the packed shards:

```bash
python3 -m AIAEC.dataset_gen.rematerialize_linear_aec \
    --input data_aec/all --config AIAEC/dataset_gen/config.ini --jobs 8

python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all --overwrite
```

Far, mic, near-target and the sequence boundaries are preserved; only
`linear_error` is recomputed. Any checkpoint trained on the old distribution
must be retrained.

**Before you retrain, you can measure what the change cost.** A checkpoint
carries the contract it was trained under, so every path that loads one refuses
an engine whose `aec_behavior_hash` has moved — that refusal is the guard doing
its job, not a defect. To see how the existing checkpoint behaves on the new
frontend's signal, materialize one pair and hand it straight to the neural
post-filter:

```bash
python3 -m AIAEC.dataset_gen.materialize_pair mic.wav far.wav error.wav

python3 -m AIAEC.Align_ULCNet.inference ckpt.pt error.wav far.wav out.wav \
    --input-is-linear-error
```

`materialize_pair.py` builds its contract from the installed library rather
than from a checkpoint, so it has no recorded hash to disagree with;
`--input-is-linear-error` bypasses the frontend entirely, so nothing on that
side carries a contract either. Pass the **raw** far to inference — the
training contract pairs `linear_error` with raw far and the model's alignment
consumes it.

⚠ Both halves are evaluation-only. `materialize_pair.py` pads one waveform to a
hop boundary and trims the result back, so its tail is not what the corpus path
produces; its output must never be written into a corpus, and no contract
comparison can tell the two apart. Run the same pair through the old library
too if you want the comparison to have a baseline.

**`--jobs` is the only lever that matters.** Measured on a 16 kHz corpus, the
Python PBFDKF is **99.8%** of the run and file I/O is 0.1%, so nothing about
the WAV handling is worth optimizing. One process sustains roughly 3.3x
realtime, which puts a 200-hour corpus near 62 hours; sequences are
independent, so `--jobs N` scales approximately with N until CPU or memory
bandwidth saturates. Do not expect exactly N×: process startup, memory
bandwidth and storage all take a cut, and the useful ceiling is usually below
the logical-core count.

`--jobs` cannot change the corpus. Each sequence gets a fresh PBFDKF, writes
only its own chunks, and has no random source, so the only thing N changes is
the order sequences finish in. Pinned by
`tests/test_rematerialize_linear_aec.py`, which compares every sample of
`--jobs 1` against `--jobs 3`.

> ⚠ Compare AUDIO, not file bytes, if you check this yourself. libsndfile
> stamps a `PEAK` chunk with the wall-clock time when it writes a float WAV,
> so any two runs seconds apart differ in one byte at offset 61 regardless of
> `--jobs`.

**`--resume` is safe for this.** It skips a sequence only when THIS contract
already wrote it AND that sequence's chunks still match on disk — the ledger
records what was written, not what survived, so every claim is re-checked
before it is honoured. A ledger written by a different contract is discarded
whole rather than partially trusted. The ledger records a sequence only after
all of its chunks are on disk, so an interrupted run redoes at most the ones
that were in flight (up to `--jobs` of them). A run WITHOUT `--resume` starts
by emptying the ledger, so it can never inherit a claim it did not make.

**Between the two commands.** The packer is the only thing that checks the
corpus is whole, so the order matters and there is no step to skip:

1. `rematerialize_linear_aec` must exit **0**. A non-zero exit means some
   sequences still carry the old fifth channel; the ledger will say so and the
   packer will refuse, but do not go looking for a way around that.
2. Do not run a second rematerializer against the same split concurrently.
   Both would write the same ledger and each would overwrite the other's
   claims.
3. Then pack. If it refuses, finish the rematerialization -- re-run it, this
   time WITH `--resume`, which now re-checks each claim against the chunks on
   disk before honouring it.

If the run was interrupted, re-running with `--resume` is the cheap path and
is safe: it redoes only what is not already recorded and verified. Re-running
without `--resume` is also correct, just slower -- it empties the ledger and
starts over.

**The packer refuses a half-rematerialized corpus.** If
`linear_error.done.json` exists but does not name every sequence, the split is
part new frontend and part old, and `pack_aec_dataset.py` stops instead of
labelling the shards with one contract while they hold two — the one failure
here with no downstream detector, since the fifth channel is just samples once
packed. A corpus with no ledger at all is fine: a one-pass generation never
had one, and the guard is against a PARTIAL claim, not an absent one.

**v2 → v3 has no automatic migration, deliberately.** A v2 contract records only
a raw-text source hash, so once `lib/aec` has moved on there is no way to
recover what the producing build's *behaviour* hash was — stamping the current
one would assert a compatibility nobody verified. Re-stamp a dataset by
re-running `rematerialize_linear_aec.py` (it re-runs PBFDKF and rewrites the
`linear_error` channel, which is the honest thing to do). A v2 **checkpoint**
cannot be repaired and must be retrained against a v3 corpus.

`behavior_hash_schema` is folded into the digest as well as compared, so the two
can never disagree. Bump it (`canon-ast-1` → `canon-ast-2`) whenever `_canon_ast`
changes what it emits for unchanged input; a mismatch then names the serializer
instead of looking like an AEC code change. No stamped artifact has ever carried
a v3 contract without this field, so there is no migration path for one.

The check is conservative: a pure refactor (renaming a local, reordering
independent statements) also changes the behaviour hash. Refusing to load is the
safe direction — rematerialize rather than loosening the check. Refresh the
channel with the command below, which avoids repeating acoustic mixing:

#### The one exception: verified frontend-equivalent migrations

> **⚠ The table is empty (2026-09-04).** The pair that carried the 200-hour
> corpus forward (the fresh-instance `aec_reset()` composed with the 48-kHz
> FilterAnalyzer correction, 16 kHz only) was retired when lib/aec retimed
> the shadow-copy error-baseline retention per grid (0.995 → 0.992 on this
> corpus's 512/256 grid) and made the C hard restart clear coarse/leakage
> evidence: both **move `linear_error`**, so no byte-identity evidence can
> describe a frontend after them. Every identity a corpus on disk may carry
> is now in `RETIRED_BEHAVIOR_HASHES` and is refused with an instruction to
> rematerialize (`rematerialize_linear_aec.py`, WITHOUT `--resume`, then
> repack, then retrain), not with a bare hash mismatch. A corpus that also
> needs the `[acoustic_tails]` mixture must be generated afresh instead —
> see "Migrating an existing corpus" above.
>
> Retired identities are **never retargeted**: pointing one at a live hash
> would declare an old waveform compatible with a build that does not produce
> it. `behavior_hash_schema` stays `canon-ast-1`: these are behaviour changes,
> not canonicalizer changes. Pinned by
> `tests/test_linear_aec_behavior_migration.py`.

`ACCEPTED_BEHAVIOR_HASH_MIGRATIONS` in `linear_aec.py` is an explicit table of
`recorded → current` behaviour-hash pairs that are known to produce the *same*
`linear_error`. `require_linear_aec_contract` accepts exactly those pairs, with
a `RuntimeWarning` naming the migration; the existing shards and the trained
checkpoint stay valid, so **no regeneration, no re-stamping, no retraining**.
`behavior_hash_schema` is *not* bumped — the canonicalizer is unchanged, only
the sources it is applied to.

This is not a loosening of the guard, and specifically is not a way to accept
"old hashes" in general:

- an entry is a **single explicit pair**, never a wildcard or a version floor;
- it applies **only when `aec_behavior_hash` is the sole differing field**, so a
  real frontend change cannot ride along with an accepted pair;
- it is **one-way**. A checkpoint recorded under the newer hash run against the
  older build is a downgrade and stays refused;
- it is **single-hop**: the table is not applied transitively, so two stacked
  migrations need the composed pair, re-verified end to end;
- an unlisted hash is refused exactly as before.

Admission requires *measured* evidence, not an argument that the change looks
inert: render the frozen frontend (`LinearAecProcessor`, formed_output seam)
before and after over a scene that actually reaches the changed code, and show
the bytes are identical — plus a control proving the same harness *can* fail
(render with the new mechanism enabled and confirm the bytes move). The
byte-equality of a dead harness is worth nothing. The rationale and the numbers
for each admitted entry live in the comment on the entry itself; the evidence
for retired entries stays in the history of the revision that admitted them.

#### Released-checkpoint inference exception

The 16-kHz Align-ULCNet checkpoint released immediately before the PBFDKF
causal-half TD-constraint correction remains loadable through
`require_inference_linear_aec_contract`. This is deliberately a different
allowlist from `ACCEPTED_BEHAVIOR_HASH_MIGRATIONS`: the correction changes
`linear_error`, so inference emits a warning that model output may differ.
Only that exact recorded/current hash pair, at 16 kHz and in the live
`LinearAecEngine`, is accepted. Dataset packing, rematerialization, training,
48-kHz use, reverse use and every unlisted hash still fail closed; regenerate
the corpus before training the next checkpoint.

Anything that retunes a *live* mechanism does not qualify for the
frontend-equivalent table — rematerialize instead:

```bash
python3 -m AIAEC.dataset_gen.rematerialize_linear_aec \
    --input data_aec/all \
    --config AIAEC/dataset_gen/config.ini
python3 -m AIAEC.dataset_gen.pack_aec_dataset \
    --config AIAEC/dataset_gen/config.ini \
    --input data_aec/all --output data_aec/packed/all --overwrite
```

`rematerialize_linear_aec.py --resume` skips a sequence only when the CURRENT
contract already wrote it, per the `linear_error.done.json` ledger beside the
corpus; a ledger from another contract is ignored whole. Re-running after a
`[linear_aec]` config edit therefore redoes every sequence on its own, with or
without `--resume`. That holds across an accepted migration too: the ledger is
keyed on `fingerprint()`, which folds in the raw-text `aec_source_hash` and
`aec_commit`, so an INTERRUPTED rematerialization restarts even where the
migration applies. A COMPLETED one does not — the packer honours that ledger
with a `RuntimeWarning`, because refusing there would strand a finished corpus
behind advice that cannot be followed: the config is identical, and what moved
is `lib/aec`.

**The ledger records the frontend that wrote it,** not only the fingerprint.
The fingerprint is one-way, so a ledger carrying nothing else can say a corpus
was written by a different contract but never *which* — which leaves an
operator comparing two opaque hashes. `linear_error.done.json` therefore also
carries the producing contract under `linear_aec`, and the packer bridges on
that: an exact, config-independent comparison through
`require_linear_aec_contract`, which reports the field that disagrees and
names both builds. A ledger written before that field existed (only `contract`
and `sequences`) still reads normally everywhere; for those the packer
reconstructs the migrated-from build's fingerprint from its recorded
provenance instead (`MIGRATED_SOURCE_PROVENANCE`, one candidate per `lib/aec`
revision that carried the migrated-from behaviour hash — a behaviour hash is
comment-insensitive, so a RANGE of revisions carries it and each one wrote a
different fingerprint). If that refuses, the message says the ledger records
no identity and asks for `git -C lib/aec rev-parse HEAD` plus the `[signal]`
and `[linear_aec]` config sections from the machine that ran the
rematerialization — the two things needed to identify the build. Add `--jobs N` to spread independent sequences across
cores — it does not change a single sample.

Generation is deterministic given `--seed`: each sequence is seeded from
`(seed, split, sequence_id)`, so it renders identically regardless of worker
count or ordering, and `--resume` continues exactly. `--hours` resolves to a
fixed sequence list up front, so extending a corpus keeps every sequence it
already had.

**⚠ `--wav-encoding` defaults to `float32`** because the corpus's central
invariant, `mic_preclip == near_speech + local_noise + echo`, is checked at
generation time against the renderer's un-quantised audit tensors (`echo`,
`local_noise` and `mic_preclip` are not among the generated WAV stems — see
"The five stems" above). Quantising the WAV stems to `int16` would still
degrade any downstream arithmetic that combines them (e.g.
`D_hat = mic_postclip - linear_error`) by ~1e-4. `int16` halves the disk cost
and is fine for listening, not for arithmetic.

**⚠ `--workers > 0` on macOS uses spawn.** The shipped CLI has the
`if __name__ == '__main__'` guard it needs; a script that calls
`gen_aec_dataset()` at module level will fail without one.

## Tests

```bash
python3 -m pytest AIAEC/dataset_gen/tests/ -q      # from Audio_ALG/
```

They render a small synthetic corpus through the real pipeline and check the
things a consumer cannot notice being wrong: stem channel order, the stem-sum
identity, that `ref_dropout` chunks really have a silent reference, that the
echo really is a delayed copy of the reference at the recorded delay, that the
split is disjoint in the *generated data* and not only in the manifest, that a
sequence's chunks are contiguous and ordered, and that the STFT round-trips on
both the 16 and 48 kHz grids.

The movement axis is checked by measuring it back out of rendered audio with
`path_drift_metrics.py` and comparing against the blind-set targets, so the
test cannot pass by reading the schedule the renderer wrote down. Its RIRs are
deliberately short enough to fit the estimator's 64 ms frame (a longer path
cannot be written as one frequency response and would put an estimator floor of
0.98 under every number) and deliberately reverberant enough that two positions
in a room are not near-identical. The `echo_path_moving` label test freezes the
weight schedule and requires the label to disappear.

## Reuse and approximations

`prepare_rir`, `estimate_rt60`, `fftconvolve`, `delay_signal`, `active_rms`,
`apply_clipping`, `prevent_clipping`, `rand_biquad_filter`, `sample_snr`,
`parse_snr_values` and `simulate_upsampled_source` are imported from
`AINR/dataset_gen/dataset.py`. Local noise uses the **same discrete SNR set** as the
NR generator, drawn with the same helpers, so the two corpora are comparable on
the noise axis instead of being two definitions of "10 dB SNR".

Two deliberate approximations, both flagged in the code:

- **SRO** is Catmull-Rom fractional interpolation, not a bandlimited
  resampler. A few ppm is a slowly accumulating fractional delay and no
  integer-rate resampler can express it — `resample(16000, 16001)` is 62.5 ppm,
  an order of magnitude too coarse.
- **`codec_mismatch`** is band limiting plus µ-law requantisation, not a real
  codec. It produces the property that matters (a nonlinear, non-invertible
  difference between the reference and what was played) without adding a
  dependency. **⚠ No result may be reported as "robust to \<codec\>".**
