# AIAEC models

AIAEC keeps four candidates. The older GTCRN- and DeepFilterNet2-derived AENR
variants were removed because Align-ULCNet now covers the linear-AEC + RES/NR
route.

| Route | Model | Model inputs | Training target | Status |
|---|---|---|---|---|
| linear AEC -> RES+NR | Align_ULCNet | linear error + far | early, clean near speech | selected |
| end-to-end AEC+RES+NR | Align_CRUSE | microphone + far | early, clean near speech | selected |
| end-to-end AEC+RES+NR | DeepVQE_S | microphone + far | early, clean near speech | primary |
| end-to-end AEC+RES+NR | CAGCRN | microphone + far | early, clean near speech | backup |

The model grid is zero-padding-free, sqrt-Hann and 50% overlap. Current model
configs use FFT/window/hop = 512/512/256 at 16 kHz. Checkpoint contracts store
the complete grid and inference rejects incompatible settings.

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
not need to be regenerated when repacking, but a corpus rendered before the
frozen frontend moved onto the context seam carries a `linear_error` produced
without the over-output capture guard; rematerialize that channel before
repacking (`dataset_gen/README.md`, "Recomputing `linear_error` after a
frontend change").

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
one hop at a time and feeds its formed error plus consumed aligned far to the
model; PBFDKF is not precomputed over the whole WAV.

## Training contract

Generic safety machinery is shared through `AIAEC/training_common.py`:
non-finite evidence dumps, finite gradient clipping, the per-tensor
vanished-weight guard, data loading, and checkpoint contracts.  The numerical
training recipe is intentionally **model-specific**; forcing all papers through
one optimizer/loss/scheduler was not a valid comparison protocol.

| model | optimizer / LR | schedule | objective | shipped batch / epochs |
|---|---|---|---|---|
| Align-ULCNet | Adam / 4e-3 | validation plateau, x0.1, patience 1 | component-compressed frequency MSE, c=0.3 | 16 / 50 |
| Align-CRUSE | Adam / 1.5e-4, decay 5e-6 | constant | STFT-consistent PLCPA, c=0.3, beta=0.7 on complex term | 16 / 50 |
| DeepVQE-S | AdamW / 1.2e-3, decay 5e-7 | warmup -> cosine (project choice) | inherited Align-CRUSE PLCPA; the DeepVQE paper does not state its loss | 8 / 50 |
| CAGCRN | AdamW / 1e-3, decay 5e-7 | constant | MSE + SI-SNR + normalized parameter L1 | 32 / 50 |

The epoch budget is the project decision requested for this training campaign.
Align-ULCNet batch 16 and DeepVQE-S batch 8 are GPU-memory overrides; the
papers use 64 and 400 respectively. CAGCRN explicitly publishes batch 32.
Early stopping cannot end these runs before epoch 50; validation still selects
the best checkpoint, and Align-ULCNet still applies its published LR reduction.
Changing a loss recipe changes `loss_version`, so a checkpoint from the old
generic recipe is rejected instead of silently resuming under a new objective.
The checkpoint contract also records the resolved training recipe -- optimizer
name, learning rate, weight decay, amsgrad, schedule and the schedule's own
scalars -- read once and used both to build the optimizer and to write the
contract. Without it two runs of one model that differ only in learning rate
produce identical contracts and resume into each other in silence, which is
exactly the drift `require_checkpoint_contract` exists to catch and which
became reachable the moment the four candidates stopped sharing one recipe.
Align-ULCNet is the only model with a STATEFUL scheduler: plateau state cannot
be reconstructed from a step count, so it is stored and restored with the
optimizer. DeepVQE-S also runs a schedule, but a reconstructible one -- the
shared per-step warmup/cosine, rebuilt from the epoch budget and fast-forwarded
on resume rather than checkpointed. Its paper publishes an optimizer but no
schedule, so that trajectory is a project choice and is labelled as one in its
config and README; a constant LR was the alternative and leaves the late epochs
taking early-epoch-sized steps. Align-CRUSE and CAGCRN report no schedule
either and do keep a constant LR.

The stored project examples remain 10-second chunks and one epoch remains one
complete DataLoader pass. Align-ULCNet's paper used random 3-second examples
and 20,000 steps per epoch; this campaign intentionally does not rewrite the
already generated dataset or silently redefine an epoch.

## ONNX export and calibration

The accelerator graphs are stateless: convolution history, attention rings
and recurrent state are explicit graph inputs and outputs. Complex values use
real/imaginary pairs.

One calibration run produces both deployment artifacts: it exports and
parity-checks the ONNX graph the tensors bind to (default `<output>.onnx`,
override with `--onnx`) from the same wrapper in the same process, then
records the calibration frames against it. Run it from the model directory:

    python3 inference.py calib \
      --checkpoint checkpoint.pth \
      --primary-dir /path/to/primary_wavs \
      --far-dir /path/to/far_wavs \
      --frames 8192 --format bin \
      --output calib/model

`export_onnx.py` remains available for a graph-only export:

    python3 export_onnx.py \
      --checkpoint checkpoint.pth \
      --output output/model.onnx --verify

For Align-ULCNet, --primary-dir contains materialized linear-error WAVs. For
the end-to-end models it contains microphone WAVs. Relative WAV paths in the
primary and far trees must match; differently-tokened names
(mic_001.wav vs lpb_001.wav) pair via an explicit --pair-replace mic:lpb
rule.

Calibration formats:

- --format bin: tensor/tensor_0000.bin, tensor/tensor_0001.bin, and so on, plus
  manifest.json. Each file is one complete ONNX input tensor, including batch
  dimension. This is the board-facing format.
- --format npz: one NumPy archive plus a sibling JSON report.

If --format is omitted, a .npz suffix selects NPZ; otherwise BIN is used.
Generated calib/ directories are ignored by Git.

Align-ULCNet supports --max-delay-frames D in both export and calibration.
The same D must be used for the ONNX graph, calibration artifact and CPU state
allocation. D changes state shapes, not learned weight shapes.

Align-ULCNet calibration uses the available training-domain
linear_error + raw_far WAV pair. Its report separately records that board
deployment supplies aligned_far.

## C pre/post-processing

    make -C AIAEC

This builds AIAEC/build/libaiaec_prepost.a.

| Model | Accelerator output | Host composition |
|---|---|---|
| Align-ULCNet | enhanced RI spectrum + delta state | WOLA + ulcnet_model_io_commit() |
| Align-CRUSE | real mask | aiaec_apply_real_mask() |
| DeepVQE-S | 3x3 complex CCM taps | deepvqe_ccm_process() |
| CAGCRN | complex mask | aiaec_apply_complex_mask() |

Align-ULCNet CPU state/ring ownership is documented in
Align_ULCNet/README.md and implemented by ulcnet_model_io.c/.h.

## Tests

From Audio_ALG/:

    python3 -m pytest AIAEC
    make -C AIAEC test

AIAEC/wav_testset/ is a local, gitignored Align-ULCNet listening-demo clone;
it is not part of the release.
