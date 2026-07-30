# PostFilter — joint residual-echo + noise suppressor

Stage 3 of the AEC chain.  It **owns the suppression decision**: downstream there
is no classical residual-echo suppressor and no `min(g_nr, g_res)` fusion, so
anything this block declines to suppress ships.

## What it consumes and what it emits

```
in : (E, D_hat [, X])          E = Y - D_hat  (AEC output)
                               D_hat          (echo estimate from stage 1)
                               X              (far-end reference, optional)
out: G(f) in [0,1]             per ERB band, expanded to bins
     or  G * e^{j theta}       |mask| still in [0,1]   ([model] output_type)
target: S                      near-end speech, its room reverberation INCLUDED
```

`R = D - D_hat` (the residual echo) emerges from the front-end; it is never a
target.  The mask is applied to **E**, not to the microphone.

⚠ This is not a noise suppressor.  It has to handle residual echo *and* noise at
once, which is why the features carry echo-specific cues (the `|D_hat|²/|E|²`
ratio and the `E`/`D_hat` coherence) that a pure NR model has no use for.

## Front-end agnosticism — the design point

`(E, D_hat)` come from a **frozen** upstream stage that may be the classical
linear canceller or the learned AECNet.  Their residuals differ substantially:
convergence behaviour, musical noise, surviving nonlinear echo, phase error,
near-end leakage, path-change transients.

Two things follow, and both are enforced rather than documented:

1. **Every feature is scale-invariant.**  `E`, `D_hat` and `Y` are divided by
   `sqrt(mean(|E|² + |D_hat|²))` of the same frame before anything else happens,
   and every channel is then a ratio or a log-domain difference.  Multiply the
   front-end's whole output by a constant and the predicted gain does not move
   (measured: 6e-8 over six decades of scaling; the test asserts < 1e-4).  An
   upstream gain retune is therefore not a retrain.
   The single opt-out, `[feature] include_absolute_level`, is off by default and
   there is a test proving it really does break the property.
2. **The front-end's identity is in the checkpoint.**  `frontend_id` is recorded
   and gated; a resume against a different front-end is refused unless
   `--allow-frontend-change` is passed, and doing so writes a permanent
   `frontend_history` into every descendant checkpoint.  ⚠ Attaching a
   checkpoint to a front-end it was not trained behind is a valid OOD experiment
   and **not** a valid result; the code makes the two impossible to confuse.

## Architecture

ERB band-gain regression is the low-complexity reference point — 257 bins → 32
bands → one gain per band → expanded back through an ERB partition of unity
(so `gain = 1` leaves `E` untouched bit for bit).

```
features (7 ch x 32 bands)
  -> causal conv2d (3x3, lookahead knob)          left context carried as state
  -> 2 x separable freq conv, stride 2            32 -> 16 -> 8
  -> GRU 2 x 352                                  hidden state carried as state
  -> Linear 128 -> ReLU -> Linear 32 -> sigmoid
```

At the default config, **1,582,624 parameters / 111.9 M MACs/s** at 62.5 fps
(strictly causal, `lookahead_frames = 0`).  `[model] mask_resolution = full`
switches features *and* output to per-bin: 2,881,489 parameters / 262.4 M
MACs/s.  `output_type = complex` adds a unit-norm phase rotation and requires
`full` — one rotation per ERB band would be an average of decorrelated angles,
so the constructor refuses the combination instead of underperforming quietly.
A test pins both variants inside the 100–300 M MACs/s budget.

## Feature channels

| # | channel | why |
|---|---------|-----|
| 0 | `log10 band\|E\|²` (reference-normalised) | spectral shape |
| 1 | `log10(band\|D_hat\|² / band\|E\|²)` | the primary residual-echo cue |
| 2 | `log10(band\|E\|² / band\|Y\|²)` | how much the linear stage already removed |
| 3 | banded coherence `E` vs `D_hat` | residual echo is the part of E still correlated with the estimate |
| 4 | channel 0 minus its own causal EMA | stationary noise vs speech, without an absolute level |
| 5 | `log10 band\|X\|²`, X normalised by its **own** frame energy | optional |
| 6 | banded coherence `E` vs `X` | optional — a *linear* front-end's `D_hat` cannot contain the loudspeaker's nonlinear products |

All EMA time constants are specified in **seconds** in `config.ini` and converted
with `alpha_from_tau`; nothing derives a coefficient from a frame count, which is
what lets the 48 kHz variant be a config change only (verified end to end).

## Caller-side bounds

The network emits its decision and nothing else.  The preset floor, the
attenuation cap and the gain smoothing live in `postproc.py` and are read from
`[inference]`; `denoise.py` is the caller and applies them.  Folding them into
`forward` would make every preset a retrain and would let the model learn to
fight its own floor.  ⚠ dB on a gain converts with `/20`.

## Training

```
python train.py --config config.ini --packed-dir data/aec_packed --gpu 0
python train.py --config config.ini --packed-dir data/aec_packed \
                --resume output/postfilter_best.pth
```

Consumes the packed AEC corpus from `AIAEC/dataset_gen_aec/pack_aec_dataset.py` and
derives `Y = mic_postclip`, `X = far_render`, `D = echo`, `S = near_speech`.

Three things in the loop are load-bearing:

* **State carries along a lane.**  `SequenceChunkSampler` puts consecutive
  chunks of one sequence in lane *k* of consecutive batches; the GRU state, the
  feature EMAs, the causal conv's left context, the front-end's adaptive filter
  and the STFT overlap tail all persist along that lane and reset only where
  `chunk_index == 0`.  A per-chunk reset still converges — it just cannot show
  convergence, echo-path-change recovery or long-term drift, which is what the
  20–60 s sequences exist for.  A test asserts chunked processing is
  bit-equal to whole-sequence processing.
* **`center=False` plus an explicit overlap tail.**  `center=True` pads half a
  window of zeros at both ends of every chunk, i.e. a fabricated 32 ms gap every
  four seconds that the adaptive front-end re-converges across.
* **The split is over sequences, never chunks.**  A source-disjoint split
  recorded in the corpus is preferred when present; otherwise whole sequences
  are held out using the shared `locality_preserving_random_split`.  A
  chunk-level split would put the same speaker, room and echo path on both sides.

The loss is compressed magnitude + complex MSE against `S`, plus SI-SDR taken
between the ISTFT of the *predicted spectrum* and the ISTFT of the *target
spectrum* (never the raw waveform — that puts a WOLA reconstruction floor on the
term).  `[loss] idle_weight` upweights `ref_dropout` / `near_only` chunks, where
any suppression at all is pure regression and which are otherwise too rare to
generate gradient.  `[loss] echo_leak_weight` is an optional penalty on
`|mask * R|`; ⚠ it moves the operating point along the echo/near-end Pareto
curve, it does not improve the curve, and its value must be reported alongside
any echo number.

## Inference

```
# run the configured front-end too
python denoise.py --config config.ini --model output/postfilter_best.pth \
                  --mic mic.wav --ref far.wav --output enhanced.wav

# (E, D_hat) already exist -- e.g. the shipping C canceller produced them
python denoise.py --config config.ini --model output/postfilter_best.pth \
                  --aec-out e.wav --echo-est d_hat.wav [--ref far.wav] \
                  --output enhanced.wav
```

⚠ The second form cannot verify that the supplied `(E, D_hat)` came from the
front-end the checkpoint was trained behind — nothing in a wav records that.  It
prints the expected `frontend_id` so the operator can.  The first form enforces
it.

## Files

| file | role |
|---|---|
| `config.ini` | signal grid, front-end, features, model, loss, training, `[inference]` preset |
| `model.py` | ERB filterbank, `PostFilterFeatures`, `PostFilterNet`, MACs estimator |
| `frontends.py` | the frozen stage-1 adapters and `frontend_id` (`none` / `oracle` / `stft_nlms` / `plugin`) |
| `postproc.py` | caller-side floor, cap and smoothing — deliberately outside the network |
| `train.py` | contract gate, sequence-aware loop, checkpointing |
| `denoise.py` | inference, both input forms |
| `tests/` | 71 tests: gain range, scale invariance, echo-free operation, causality, state carry, contract and front-end gates, an end-to-end trainer run |

Run them with `python3 -m pytest tests/ -q` from this directory.

## Front-ends

`[frontend] kind`:

* `none` — `D_hat = 0`, `E = Y`.  The pure-NR ablation, and the only setting in
  which this project answers the same question as the NR bake-off models.
* `oracle` — `D_hat = D`, so `R = 0`.  ⚠ A diagnostic upper bound; any number
  from it must be reported as "oracle AEC".
* `stft_nlms` — a per-bin multi-tap NLMS canceller on this grid.  It converges
  over seconds, mis-converges on double talk and re-converges after a path
  change, which is the point.  ⚠ Per-bin filtering ignores cross-band leakage,
  so it cancels a little less than a time-domain filter of the same length; do
  not quote its ERLE as "the classical AEC's ERLE".  `taps` is in FRAMES and
  must cover the corpus's 120 ms bulk delay plus the room RIR — this is the one
  setting a 48 kHz variant has to change (16 → 24).
* `plugin` — `module:factory` plus a checkpoint, for the AECNet sibling project.
  The checkpoint's content hash is folded into `frontend_id`, so the same
  architecture retrained counts as a different front-end.

## Open questions for a reviewer

* **`stft_nlms` is the default front-end but it is an approximation**, not a port
  of the shipping canceller.  If the residual it produces is materially unlike
  the real one, the model is being trained on the wrong distribution.  Pointing
  `kind = plugin` at the real canceller is the fix and is already supported.
* **The front-end runs inside the training loop**, frame by frame, and is the
  step's bottleneck.  Pre-computing `(E, D_hat)` into the shards would be much
  faster; it was not done because that freezes the front-end into the corpus and
  makes swapping it a regeneration rather than a config change.
* **`idle_weight = 3.0` is a guess**, chosen to bring ~15% of chunks up to
  roughly a third of the gradient.  It is in the contract, so changing it forces
  a fresh run — deliberately, but it does make sweeping it expensive.
