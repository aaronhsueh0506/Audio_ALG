# AECNet — standalone AI acoustic echo canceller (stage 1)

```
in : Y (microphone) + X (far-end reference), complex spectra
out: D_hat — the ESTIMATED ECHO, complex, n_freqs bins
then E = Y - D_hat, by SUBTRACTION, outside the model
```

**⚠ The network's target is the echo itself.** It does not emit enhanced
speech, and its output is **not a mask on the microphone**. A mask on Y would
let one network perform cancellation, residual suppression and noise reduction
at once, and no measurement afterwards could attribute a failure to any of the
three. `tests/test_aecnet_model.py` pins this behaviourally: with `Y == 0` and
an active reference the model must still be able to emit an echo estimate,
which no mask-on-Y architecture can do.

The signal model, from `dataset_gen/aec`:

```
Y = S + N + D        microphone   (S near speech, N local noise, D echo)
X                    far-end reference
D_hat = AEC(Y, X)    stage-1 output — an ECHO ESTIMATE
E     = Y - D_hat    AEC output — a SUBTRACTION, not a mask on Y
R     = D - D_hat    residual echo — emerges naturally, never a target
```

`E` still contains `R` and all of `N`. A residual suppressor and a noise
reducer are separate downstream stages, deliberately.

## What it consumes

Packed AEC shards from `dataset_gen/aec/pack_aec_dataset.py` — a directory of
`.pt` files holding six separated stems per chunk. This project imports
`AecGrid`, `stft`/`istft`, `AecStems`, `PackedAecDataset`, `SequenceChunkSampler`
and `lane_reset_mask` from `dataset_gen.aec`, and the train/val split, the
seeder and the DataLoader worker kwargs from `dataset_gen`. **⚠ It re-declares
none of them** — a drift guard in `tests/test_aecnet_contract.py` asserts that,
for the same reason `ainr/tests/test_bakeoff_protocol.py` does on the NR side.

## What it emits

`(B, 2, T, F)` = `[Re(D_hat), Im(D_hat)]` on the configured grid.
`denoise.py` writes `E = Y - D_hat` to `--output` and, with `--echo-out`, the
estimate `D_hat` itself — which is the thing to listen to when deciding whether
the model is *modelling* the echo or merely gating the microphone.

## Architecture

FCRN / CRUSE-DAEC family, ~1.94 M parameters at the shipped config:

| stage | shape (16 kHz) |
|---|---|
| input `[Re(Y), Im(Y), Re(X), Im(X)]` | `(B, 4, T, 257)` |
| strided conv encoder, 4 stages `16→32→48→64` | `257 → 129 → 65 → 33 → 17` |
| grouped GRU bottleneck, 2 layers × 8 groups of 136 | `(B, T, 1088)` |
| mirrored transposed-conv decoder, skip-concatenated | `17 → 33 → 65 → 129 → 257` |
| output `[Re(D_hat), Im(D_hat)]` | `(B, 2, T, 257)` |

The bottleneck is ~92 % of the parameters, so `channels[-1]` and `gru_groups`
are the two knobs that matter for size. A channel shuffle sits between the GRU
layers: without it the groups are independent recurrent networks that can never
exchange information, which is fatal here because the cue that a bin contains
echo lives in the reference's activity across the whole spectrum.

Three choices a reviewer should look at:

- **Per-frame normalisation, not BatchNorm2d.** Every published CRUSE
  implementation uses `BatchNorm2d`, which normalises across the *time* axis at
  training time — frame *t*'s statistics include frames *t+1…T−1*. That is a
  normaliser that peeks at the future, and it makes the causality test pass in
  `eval()` while failing in `train()`. `FreqChannelNorm` normalises over
  (channel, frequency) per frame instead, so train and eval are the same
  function and causality is asserted in both modes.
- **Compressed domain in, compressed domain out.** The model reads *and*
  predicts `|Z|^0.3` with the phase intact, so `L_echo` is a plain MSE on the
  network's own output. Compressing only inside the loss would put a cube root
  between the parameters and the objective and reintroduce the 80 dB dynamic
  range the compression exists to remove. One exponent, shared by features,
  output and loss; it is in the checkpoint contract.
- **Lookahead is a label, not a padding, and that is forced.** Any model that
  can be fed in chunks satisfies `out[i] = f(x[0..i])` — when index *i* must be
  emitted, frame *i+1* does not exist yet. A right-padded "non-causal" conv does
  not escape this; its streaming implementation buffers, and the buffering *is*
  the delay. So `lookahead = L` declares that `out[i]` is the estimate for
  **input frame `i − L`**, which has therefore seen *L* frames of future. It
  costs *L* frames of latency (16 ms each at 16 kHz), it genuinely changes what
  the model learns because the target is shifted — and **⚠ the loss must be
  told the same value**, or it scores a target shifted by *L* frames, which
  presents as a model that simply will not converge.

## Loss

```
L = L_echo(D_hat, D)
  + lambda_out  * L_output(Y - D_hat, S + N)
  + lambda_near * near_end_preservation
  + lambda_idle * ||D_hat||   on frames where the reference is inactive
```

All spectral terms are complex MSE + magnitude MSE in the `|Z|^0.3` domain.

- **`L_echo`** — the primary objective. The magnitude term is not optional: a
  complex MSE alone is minimised by shrinking `|D_hat|` toward zero wherever the
  phase is uncertain, which is exactly the shape of "the canceller does nothing".
- **`L_output`** — the same form on `E = Y - D_hat` against `S + N`. The only
  place the subtraction appears; the network still never has `S + N` as its own
  target.
- **`near_end_preservation`** — **asymmetric on purpose.** It penalises only the
  deficit `relu(|S+N|^c - |E|^c)` on frames where the near talker is active:
  energy *removed* from the near path, never energy left behind. Residual echo
  left in `E` is still removable by a downstream residual suppressor; near speech
  this stage cancelled is gone and nothing downstream restores it. A symmetric
  term would price an irreversible loss like a recoverable one.
- **`idle`** — what makes *"reference silent ⇒ produce no echo estimate"*
  trainable, supervised by the `ref_dropout` and `near_only` scenarios where
  `X == 0` and `D == 0` exactly. **⚠ It is guarded:** a frame counts as idle only
  after the reference has been *continuously* silent for `idle_guard_sec`
  (1.5 s, above the corpus's 1.2 s `rt60_max`). Without the guard every brief
  pause in far-end speech is labelled idle and the model is punished for the
  echo *tail* that is still physically arriving — i.e. trained to truncate
  reverberant echo. The guard carries across chunk boundaries; without that, the
  first frames of every 4 s chunk look idle no matter how loud the previous
  chunk ended.

## The zero-reference hard gate

With `X == 0` the model must emit essentially no echo estimate, so `E = Y - D_hat`
degenerates to `Y` and the canceller is provably transparent when there is
nothing to cancel. `train.py` measures and prints it every epoch:

```
zero-ref leak -47.3 dB [PASS]
```

`model.assert_zero_reference_gate(model, y_spec, max_leak_db=-40.0)` is the
assertion helper a trained-model evaluation calls. **⚠ It is meaningless on an
untrained model** — random weights fail it by construction — so the unit tests
check shape and finiteness there, and exercise the helper itself against stubs.

## Training

```bash
# preferred: the generator's SOURCE-disjoint split, used as generated
python3 train.py --config config.ini \
    --packed-dir     data_aec/packed/train \
    --val-packed-dir data_aec/packed/val --gpu 0

# resume
python3 train.py --config config.ini --packed-dir data_aec/packed/train \
    --val-packed-dir data_aec/packed/val --resume output/aecnet_best.pth
```

**⚠ Give `--val-packed-dir`.** `dataset_gen/aec/manifest.py` decides a
speaker/noise/room/loudspeaker-disjoint split *before* generation and writes the
two corpora separately. Without it the trainer falls back to splitting one
corpus by **whole sequences** — never by clip, both because chunks of one
sequence are near-duplicates and because `SequenceChunkSampler` requires a
complete contiguous `chunk_index` run per sequence. The fallback still shares
talkers, rooms and devices across the fence, so its validation number partly
measures memorisation.

**`batch_size` is the number of lanes.** Lane *k* of batch *b+1* carries the
next chunk of the sequence lane *k* carried in batch *b*, so the recurrent state
is valid across the boundary and a whole 20–60 s sequence is seen unbroken.
State is reset per lane on `chunk_index == 0` and detached every chunk
(truncated BPTT). **⚠ A trainer that reset state every chunk would look
identical on the loss curve and would be unable to demonstrate convergence from
cold, recovery after an echo-path change, or long-term drift** — which is most
of what this corpus was built to expose.

The startup banner prints the resolved grid, the parameter count, all three
version strings, the loss weights, the guard in both seconds and frames, the
split mode, the lane count and the scenario histogram — and warns if
`ref_dropout` is too rare for the idle term to mean anything.

## Inference

```bash
python3 denoise.py --config config.ini --model output/aecnet_best.pth \
    --mic mic.wav --ref far_end.wav --output aec_out.wav --echo-out d_hat.wav

# or one interleaved file: ch0 = microphone, ch1 = reference
python3 denoise.py --config config.ini --model output/aecnet_best.pth \
    --input mic_and_ref.wav --output aec_out.wav --block-frames 32
```

`--block-frames` runs the streaming path, carrying state between blocks. It
produces the same answer as one call — `test_chunked_state_matches_one_shot`
asserts that — so it exists to exercise the streaming path on real audio, not to
change the result.

## Checkpoint contract

`MODEL_VERSION`, `FEATURE_VERSION` and `LOSS_VERSION` plus the grid, every
`[model]` key and **every loss weight**. Resuming a `lambda_idle = 1.0` run
under `lambda_idle = 0.0` is a different objective on the same weights, and
neither the shapes nor the version strings would notice.

There is no `allow_missing` escape hatch (unlike GTCRN): this project vendors no
foreign checkpoints, so a checkpoint without a contract is one from a code state
nobody can identify.

## Tests

```bash
python3 -m pytest AECNet/tests -q      # from ainr/
```

- `test_aecnet_model.py` — shape/finiteness on an untrained model; **output is
  D_hat, not a mask on Y**; strict causality of the core at every lookahead, in
  `train()` as well as `eval()`; lookahead equals exactly the declared output
  delay; chunked-with-state == one shot; state reset touches only the flagged
  lanes; compression round-trips and survives an exact zero (⚠ complex `abs()`
  has a NaN gradient at the origin, which a reference dropout produces); the
  zero-reference gate helper; the 48 kHz grid builds from config alone.
- `test_aecnet_contract.py` — `[model]` and `MODEL_KEYS` agree in both
  directions; unknown keys rejected; the contract gate rejects every version and
  every semantic field including the loss weights; `_VERSION_FIELDS` is derived
  from the version dict, not restated; no shared primitive is re-declared;
  `--seed` defaults to 42; the guard is in seconds (and is a *different* number
  of frames on the two grids, which is the point); the guard carries across
  chunks; near-end preservation is one-sided; the loss realigns for lookahead.
