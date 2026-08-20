# AINR

Standalone neural speech-enhancement models. AINR datasets contain noisy/clean
pairs and do not include the AIAEC echo-stem contract.

## Models

| Directory | Rate / grid | Role | Checkpoint contract |
|---|---|---|---|
| `RNNoise-ERB` | 16 kHz, 512/512/256 | compact ERB recurrent suppressor | model-specific |
| `GTCRN` | 16 kHz, 512/512/256 | GTCRN baseline | model-specific |
| `DeepFilterNet2` | 48 kHz, 1024/1024/512 | DFN2 cascade + learned alpha | `MODEL_VERSION` v6 |

`DeepFilterNet2`'s final composition is a cascade:

```text
DFN2: full-band ERB mask → masked low-band DF → alpha blend
```

Do not force-load a checkpoint from another composition into DFN2. The
contract gate should reject it; bypassing the gate creates an invalid
comparison even when many parameter names happen to match.

## Dataset

`dataset_gen/` is the single noisy/clean augmentation implementation:

- run at 16 kHz for RNNoise-ERB and GTCRN;
- run at 48 kHz for DeepFilterNet2;
- keep separate output directories for separate sample-rate runs.

The generator applies the same waveform transformation to noisy and target
where required and can pack WAV pairs for training. The persisted corpus is
WAV-only; optional sample metadata exists only in the in-process API. See
[`dataset_gen/README.md`](dataset_gen/README.md).

## Current DFN interpretation

`DeepFilterNet2` is the branch to use for the restored DFN2 cascade/alpha
experiment. If it suppresses steady low-frequency speech, compare the output
composition, target/loss behavior, normalization state, and low-frequency
training coverage before attributing the result to the shared dataset.

Detailed contracts:

- [`DeepFilterNet2/README.md`](DeepFilterNet2/README.md)
- [`DeepFilterNet2/UPSTREAM_ALIGNMENT.md`](DeepFilterNet2/UPSTREAM_ALIGNMENT.md)

## Typical commands

Run from the selected model directory:

```bash
python3 train.py --config config.ini --packed-dir /path/to/packed_data
python3 inference.py --config config.ini --model /path/to/checkpoint.pt \
    --input input.wav --output output.wav
```

Use `--resume` only with a checkpoint accepted by that directory's serialized
feature/model contract.

## Training contract (shared by every trainer)

One definition, in `AINR/training_common.py`, imported by every trainer in
this directory and, through `AIAEC/training_common.py`, by all four AEC
candidates. This is the single write-up; `AIAEC/README.md` points here. The
schedule and the non-finite handling are part of the comparison protocol: two
models trained over "the same 100 epochs" are not comparable if one of them
annealed and the other sat at its initial learning rate.

**Learning rate** — per *optimizer step*, linear warmup from `lr_warmup` to `lr`
over `warmup_epochs`, then cosine annealing to `min_lr`. It decays
unconditionally; nothing about it depends on the validation loss.

**Resume** — the scheduler is deliberately NOT stored in the checkpoint. Restoring
it brings back the `T_max` of the run that wrote it: measured, a checkpoint from a
100-epoch run resumed at 120 epochs ends on lr 1.02e-04 against a `min_lr` of
1e-06, 102x. `global_step` is stored instead, and `fast_forward_scheduler()`
rebuilds for the current run and indexes the fresh schedule by step, which stays
correct across a change of epochs, batch size or corpus.

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

## C pre/post-processing

RNNoise-ERB, GTCRN, and DeepFilterNet2 provide streaming C analysis/synthesis
code; the neural inference between those two boundaries is left to the target
accelerator. The production default enables AArch64 NEON.
Use the same switch as the conventional stack to build the scalar reference:

```bash
make test-simd          # NEON and forced-scalar digest must match
make SIMD=0 test        # scalar-only build
make -C RNNoise-ERB SIMD=0 test
```

The DFN C API also performs ERB/complex feature normalization, mask expansion,
deep-filter composition, optional post-filter/attenuation limiting, and WOLA.
`dfn2_compose()` is the aligned/offline arithmetic reference. Hardware
integrations must use `dfn2_compose_stream()`: it explicitly pairs a delayed
network head with the spectrum frame it describes and returns the output frame
index. DFN2's cascade costs two hops because the future DF source first needs
its own lookahead mask. The API returns zero during warm-up; callers must not
send an invalid warm-up frame to synthesis. GTCRN uses the configured
unnormalised complex `[F,2]` network boundary. All three C implementations
use zero-padded streaming warm-up (`center=False` semantics), while the offline
PyTorch `inference.py` entry points (the whole-utterance path) use centered
STFT framing at clip boundaries.

DeepFilterNet2 and GTCRN also provide `export_onnx.py`,
`inference.py calib`, and `export_erb_matrix.py`. Install the optional
packages in `requirements-export.txt`; the model-specific READMEs define
whether recurrent state is explicit at the ONNX boundary and whether ERB is
inside or outside the graph.

RNNoise-ERB, DeepFilterNet2 and GTCRN calibration can emit `bin` or `npz`.
Every calibration run also exports and parity-checks the ONNX graph the
tensors bind to (default `<output>.onnx`, override with `--onnx`), from the
same model instance in the same process, so the two deployment artifacts
cannot drift apart; `export_onnx.py` remains available for a graph-only
export. The deployment BIN layout is identical for all three: one subdirectory per
ONNX input, one zero-padded `<tensor>_0000.bin`, `<tensor>_0001.bin`, ... per invocation, and a
`manifest.json`. Each file contains one complete graph input tensor, including
its batch dimension; only the filename index is the calibration-frame axis.
Keep generated data under each model's `calib/`; those
directories are ignored by Git and remain separate from `output/` weights and
ONNX graphs.

The active deployment exports are stateless accelerator graphs:

| Model | New signal context per call | Caller-owned model state |
|---|---:|---|
| RNNoise-ERB | 3 feature frames -> 1 gain frame | 3 GRU hidden tensors |
| DeepFilterNet2 | 3 feature frames -> 1 head frame | 3 GRU tensors + 4-frame DF pathway cache |
| GTCRN | 1 complex STFT frame | conv/TRA/inter-GRU caches |

“Stateless” describes the accelerator, not the algorithm. The host must return
every exported `*_out` state tensor as the matching input on the next call.
