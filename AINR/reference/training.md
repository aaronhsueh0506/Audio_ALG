# AINR training

## Dataset

`dataset_gen/` is the single noisy/clean augmentation implementation:

- run at 16 kHz for RNNoise-ERB and GTCRN;
- run at 48 kHz for DeepFilterNet3 and for the shipped DeepFilterNet2 grid (a DFN2 model trained for another grid takes that grid's sample rate);
- keep separate output directories for separate sample-rate runs.

The generator applies the same waveform transformation to noisy and target
where required and can pack WAV pairs for training. The persisted corpus is
WAV-only; optional sample metadata exists only in the in-process API. See
[`dataset_gen/README.md`](../dataset_gen/README.md).

## Typical commands

Run from the selected model directory:

```bash
python3 train.py --config config.ini --packed-dir /path/to/packed_data
python3 inference.py --config config.ini --model /path/to/checkpoint.pt \
    --input input.wav --output output.wav
```

Use `--resume` only with a checkpoint accepted by that directory's serialized
feature/model contract.

## Shared training safety and AINR schedule

The generic safety/checkpoint machinery lives in `AINR/training_common.py` and
is reused, through `AIAEC/training_common.py`, by the AEC candidates. It covers
non-finite handling, finite gradient clipping, evidence dumps, resume helpers
and the per-tensor vanished-weight guard. The numerical training recipe is not
universal: the AIAEC models deliberately use model-specific optimizers, losses
and schedules documented in `AIAEC/README.md`.

**AINR learning rate** — the AINR trainers use a per-*optimizer-step* linear
warmup from `lr_warmup` to `lr` over `warmup_epochs`, then cosine annealing to
`min_lr`. It decays unconditionally; nothing about it depends on validation
loss. Do not infer an AIAEC model's schedule from this section.

**Resume for reconstructible schedules** — warmup/cosine scheduler state is
deliberately not restored from the checkpoint: it would bring back the
`T_max` of the run that wrote it, so a longer resumed run would never reach
`min_lr`. `global_step` is stored instead, and `fast_forward_scheduler()`
rebuilds for the current run and indexes the fresh schedule by step. Stateful schedules,
such as Align-ULCNet's validation plateau scheduler, have their own checkpoint
contract described in `AIAEC/README.md`.

**Non-finite values** — the loss and the gradient are checked separately, because a
non-finite loss is a forward-side fault and a finite loss with an exploding
gradient is a backward-side one, and those have opposite fixes. Both route to
`halt_on_non_finite()`, which halts on the FIRST hit and writes
`output/nan_halt/e<epoch>_b<batch>_s<step>/`: the offending batch as `.pt` and
wavs, a per-lane table naming the suspect lane, and `pre_step.pth` — weights and
optimizer moments as they were BEFORE the step, so the run can be resumed from
the last uncontaminated state.

⚠ `clip_grad_norm_(..., error_if_nonfinite=True)` is not optional. Without the
flag, `clip_coef = max_norm / (inf + 1e-6)` is `0.0` and `inf * 0.0` is `NaN`, so
the clip itself writes NaN into every gradient and `optimizer.step()` carries it
into the weights and into Adam's moments, where no later clean batch recovers it.
With the flag the raise lands before any scaling. Both branches are pinned by
`AIAEC/tests/test_training_common.py::test_clipping_a_nonfinite_norm_without_the_flag_creates_the_nan`.

**Gradient trace** — every step's pre-clip norm goes to `output/grad_norm.csv`.
Once clipping is in effect the loss curve cannot distinguish a few enormous
isolated spikes (a pathological batch) from bounded gradients pointing the wrong
way (an optimizer-state problem); the trace can.

**Vanished weights** — a PER-TENSOR guard (`WeightScaleGuard`) runs once per
epoch, immediately before the checkpoint write, and halts when a parameter
tensor's `max|w|` has fallen below both 1e-06 of its own running peak and an
absolute 1e-12. Nothing else above can see that failure: the weights stay
FINITE, so the loss curve, `scan_non_finite()`, an ONNX `--verify` parity check
and the GLOBAL norm in `grad_norm.csv` all look healthy while one branch dies.

The baseline is the running maximum of each tensor's own `max|w|`, not its
initial value, so a bias that initializes at exactly zero is skipped until it
has grown and is still caught if it later collapses; under `--resume` the
baseline starts from the loaded weights.
`WeightScaleGuard`'s docstring carries the threshold arithmetic.

## DFN lookahead

The checkpoint fixes both mask lookahead M and deep-filter lookahead D. The supplied configuration uses (1, 1). With a three-frame feature window, the heads for frame t use frames t−(2−M) through t+M. Earlier features are zero at startup. With one frame per call, each of the first M hops still runs the accelerator. The host discards its warm-up heads and restarts the recurrent state.

DFN2 first applies the ERB mask and then deep filtering. Its full single-frame PCM latency is 1+M+D hops. DFN3 processes low and high bands in parallel, so its PCM latency is 1+max(M,D). Both figures include one hop of STFT/ISTFT framing. The grid, ONNX metadata and calibration must match the checkpoint. DFN2 batching changes when a job becomes ready. [Multi-frame timing](../DeepFilterNet2/reference/multiframe.md)

[返回本主題](../README.md)
