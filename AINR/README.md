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
where required, records run metadata, and can pack pairs for training. See
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
python3 denoise.py --config config.ini --model /path/to/checkpoint.pt \
    --input input.wav --output output.wav
```

Use `--resume` only with a checkpoint accepted by that directory's serialized
feature/model contract.
