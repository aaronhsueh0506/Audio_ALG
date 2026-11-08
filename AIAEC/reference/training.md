# AIAEC training

## Dataset

dataset_gen/ renders a complete stateful AEC scenario, runs the frozen Python
PBFDKF over the complete sequence, then cuts it into 10-second WAV chunks. WAV
generation stores five stems:

1. far_render
2. near_speech
3. near_target
4. mic_postclip
5. linear_error

Packing stores only the four tensors used by training: far_render,
mic_postclip, linear_error, and near_target. Existing generated WAV data does
not need to be regenerated when repacking; to recompute `linear_error` after a
frontend change see `dataset_gen/README.md`, "Recomputing `linear_error` after a
frontend change".

model_views.build_model_view() is the only waveform-to-model mapping:
Align-ULCNet receives linear_error + far_render; the other three receive
mic_postclip + far_render. Every model targets near_target.

See dataset_gen/README.md for generation, packing, contract migration and
resume rules.

## Train and inference

Every retained model directory owns these user-facing files:

- train.py
- config.ini
- inference.py
- export_onnx.py

Shared training/checkpoint and signal utilities remain under AIAEC/; modules
whose filename starts with an underscore are implementation details, not
command-line entry points.

Example:

    cd AIAEC/Align_ULCNet
    python3 train.py --config config.ini \
      --packed-dir /path/to/packed/all --gpu 0 --mmap
    python3 inference.py checkpoint.pth mic.wav far.wav enhanced.wav

Inference is streaming by default: it consumes one audio hop, advances the
STFT, calls ``forward_stream()`` for one frame with persistent state, and
advances WOLA.  ``--verify`` additionally runs the whole-utterance reference
and reports the numerical difference.  Both mono inputs are resampled to the
checkpoint sample rate first; output remains at that rate. Inputs must
describe the same capture timeline; a tail-length mismatch is cropped or
zero-padded, but a start-time offset is not estimated here.

Align-ULCNet also accepts --input-is-linear-error for evaluating a precomputed
linear-AEC residual. Normal inference advances the checkpoint-matched PBFDKF
one hop at a time and feeds its formed error plus the original raw far to the
model; PBFDKF is not precomputed over the whole WAV.

## Training contract

Generic safety machinery is shared through `AIAEC/training_common.py`:
non-finite evidence dumps, finite gradient clipping, the per-tensor
vanished-weight guard, data loading, and checkpoint contracts. Optimizer and
the optimizer remain **model-specific**; Align-ULCNet, Align-CRUSE and DeepVQE-S
share the PLCPA objective and CAGCRN keeps its own. The shipped schedule is
deliberately common: three epochs of linear warmup followed by per-step cosine decay.

| model | optimizer / LR | schedule | objective | shipped batch / epochs |
|---|---|---|---|---|
| Align-ULCNet | Adam / 4e-3 | 3-epoch warmup + cosine to 1e-6 | STFT-consistent PLCPA, c=0.3, beta=0.7 on complex term (the papers state no formula) | 16 / 50 |
| Align-CRUSE | Adam / 1.5e-4, decay 5e-6 | 3-epoch warmup + cosine to 1e-6 | STFT-consistent PLCPA, c=0.3, beta=0.7 on complex term | 16 / 50 |
| DeepVQE-S | AdamW / 1.2e-3, decay 5e-7 | 3-epoch warmup + cosine to 1e-6 | inherited Align-CRUSE PLCPA; the DeepVQE paper does not state its loss | 8 / 50 |
| CAGCRN | AdamW / 1.2e-3, decay 5e-7 | 3-epoch warmup + cosine to 1e-6 | MSE + SI-SNR + normalized parameter L1 | 32 / 50 |

### What a four-way comparison can and cannot conclude

The four candidates do not share an optimizer, and CAGCRN has its own loss, so a
ranking of their final numbers is not a ranking of their architectures. Two
independent reasons, both measurable from this directory:

**The losses are different functions.** PLCPA (Align-ULCNet, Align-CRUSE,
DeepVQE-S) and MSE+SI-SNR+L1 (CAGCRN) do not share a scale or a zero. Loss
values across that boundary are not comparable at all -- only a common
external metric is.

**The optimisation budgets differ.** Batch sizes, epoch definitions, peak
learning rates and published training durations differ. Sharing the shape of
the schedule does not equalise update count or audio exposure, so a win by one
model is not attributable to architecture alone.

This is a deliberate trade: per-paper fidelity is chosen over a shared
protocol, and the two cannot both hold (Align-ULCNet's loss is the exception:
its paper states none, so it uses the shared PLCPA objective). Whichever is
wanted, it should be an explicit decision rather than an artefact:

- **To rank architectures**, put all four on one optimiser, one schedule and
  one loss, and accept that none is then paper-faithful.
- **To reproduce each paper**, keep these recipes and judge each model against
  its own published numbers, not against its siblings here.

The epoch budget is a project decision. Align-ULCNet batch 16 and DeepVQE-S batch 8 are GPU-memory overrides. The
Align-ULCNet paper uses 64; DeepVQE-S and Align-CRUSE both publish batch 400
(for 250 and 150 epochs respectively). CAGCRN explicitly publishes batch 32.
Early stopping is disabled for all four shipped 50-epoch runs; validation still
selects the best checkpoint. The common cosine schedule is a project decision,
not a claim that all four papers published that schedule.
Changing a loss recipe changes `loss_version`, so a checkpoint trained under a different
recipe is rejected instead of silently resuming under a new objective.
The checkpoint contract records the resolved training recipe -- optimizer
name, learning rate, weight decay, amsgrad, schedule and its scalars, batch
size, epoch budget, warmup and LR floor -- read once and used both to build the
optimizer and to write the contract, so `require_checkpoint_contract` rejects a
changed trajectory instead of silently resuming it. All four shipped configs use
`warmup_cosine`, rebuilt from the global optimizer-step count on resume instead
of restoring a stale scheduler horizon.

The stored project examples are 10-second chunks and one epoch is one complete
DataLoader pass. Align-ULCNet's paper used random 3-second examples and 20,000
steps per epoch.

[返回本主題](../README.md)
