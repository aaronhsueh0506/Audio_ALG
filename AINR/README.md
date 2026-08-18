# AINR

Standalone neural speech-enhancement models. AINR datasets contain noisy/clean
pairs and do not include the AIAEC echo-stem contract.

## Models

| Directory | Rate / grid | Role | Checkpoint contract |
|---|---|---|---|
| `RNNoise-ERB` | 16 kHz, 512/512/256 | compact ERB recurrent suppressor | model-specific |
| `GTCRN` | 16 kHz, 512/512/256 | GTCRN baseline | model-specific |
| `DeepFilterNet2` | 48 kHz, 1024/1024/512 | DFN2 cascade + learned alpha | `MODEL_VERSION` v6 |
| `DeepFilterNet3` | 48 kHz, 1024/1024/512 | preserved DFN3-style band split | `MODEL_VERSION` v5 |

The two DeepFilterNet directories intentionally share the feature contract and
most hyperparameters. Their final composition is different:

```text
DFN2: full-band ERB mask → masked low-band DF → alpha blend
DFN3: low-band DF || high-band ERB mask → band split
```

Do not load a DFN3/band-split checkpoint into DFN2. The contract gate should
reject it; bypassing the gate creates an invalid comparison even when many
parameter names happen to match.

## Dataset

`dataset_gen/` is the single noisy/clean augmentation implementation:

- run at 16 kHz for RNNoise-ERB and GTCRN;
- run at 48 kHz for both DeepFilterNet variants;
- keep separate output directories for separate sample-rate runs.

The generator applies the same waveform transformation to noisy and target
where required and can pack WAV pairs for training. The persisted corpus is
WAV-only; optional sample metadata exists only in the in-process API. See
[`dataset_gen/README.md`](dataset_gen/README.md).

## Current DFN interpretation

`DeepFilterNet2` is the branch to use for the restored DFN2 cascade/alpha
experiment. `DeepFilterNet3` is retained as a controlled comparison, not as a
replacement dataset pipeline. If one branch suppresses steady low-frequency
speech while the other does not, compare the output composition, target/loss
behavior, normalization state, and low-frequency training coverage before
attributing the result to the shared dataset.

Detailed contracts:

- [`DeepFilterNet2/README.md`](DeepFilterNet2/README.md)
- [`DeepFilterNet3/README.md`](DeepFilterNet3/README.md)
- [`DeepFilterNet2/UPSTREAM_ALIGNMENT.md`](DeepFilterNet2/UPSTREAM_ALIGNMENT.md)
- [`DeepFilterNet3/UPSTREAM_ALIGNMENT.md`](DeepFilterNet3/UPSTREAM_ALIGNMENT.md)

## Typical commands

Run from the selected model directory:

```bash
python3 train.py --config config.ini --packed-dir /path/to/packed_data
python3 inference.py --config config.ini --model /path/to/checkpoint.pt \
    --input input.wav --output output.wav
```

Use `--resume` only with a checkpoint accepted by that directory's serialized
feature/model contract.

## C pre/post-processing

RNNoise-ERB, GTCRN, DeepFilterNet2, and DeepFilterNet3 provide streaming C
analysis/synthesis code; the neural inference between those two boundaries is
left to the target accelerator. The production default enables AArch64 NEON.
Use the same switch as the conventional stack to build the scalar reference:

```bash
make test-simd          # NEON and forced-scalar digest must match
make SIMD=0 test        # scalar-only build
make -C RNNoise-ERB SIMD=0 test
```

The DFN C API also performs ERB/complex feature normalization, mask expansion,
deep-filter composition, optional post-filter/attenuation limiting, and WOLA.
`dfn*_compose()` is the aligned/offline arithmetic reference. Hardware
integrations must use `dfn*_compose_stream()`: it explicitly pairs a delayed
network head with the spectrum frame it describes and returns the output frame
index. DFN3's parallel 1/1 branches cost one hop; DFN2's cascade costs two hops
because the future DF source first needs its own lookahead mask. Both APIs
return zero during warm-up; callers must not send an invalid warm-up frame to
synthesis. GTCRN uses the configured unnormalised
complex `[F,2]` network boundary. All three C implementations use
zero-padded streaming warm-up (`center=False` semantics), while the offline
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
ONNX input, one `<tensor>_<1-based-frame>.bin` per invocation, and a
`manifest.json`. Each file contains one complete graph input tensor, including
its batch dimension; only the filename index is the calibration-frame axis.
Keep generated data under each model's `calib/`; those
directories are ignored by Git and remain separate from `output/` weights and
ONNX graphs. DeepFilterNet3 is a controlled comparison and does not advertise
a stateless ONNX exporter in this release.

The active deployment exports are stateless accelerator graphs:

| Model | New signal context per call | Caller-owned model state |
|---|---:|---|
| RNNoise-ERB | 3 feature frames -> 1 gain frame | 3 GRU hidden tensors |
| DeepFilterNet2 | 3 feature frames -> 1 head frame | 3 GRU tensors + 4-frame DF pathway cache |
| GTCRN | 1 complex STFT frame | conv/TRA/inter-GRU caches |

“Stateless” describes the accelerator, not the algorithm. The host must return
every exported `*_out` state tensor as the matching input on the next call.
