# Audio_ALG

Audio processing workspace for conventional AEC/NR integration, standalone
AI noise reduction, and neural AEC research. These are three related but
independent product surfaces; do not mix their signal grids or checkpoint
contracts.

## Current scope

| Area | Purpose | Current entry point |
|---|---|---|
| `pipelines/` | Production-oriented linear AEC + echo-aware NR/RES | [`pipelines/README.md`](pipelines/README.md) |
| `AINR/` | Standalone speech enhancement | [`AINR/README.md`](AINR/README.md) |
| `AIAEC/` | Six neural AEC/AENR candidate architectures | [`AIAEC/README.md`](AIAEC/README.md) |
| `pipelines/4ch_pipelines/` | Python reference and C API for four linear AEC lanes around an externally owned SRP-PHAT/GSC | [`pipelines/4ch_pipelines/README.md`](pipelines/4ch_pipelines/README.md) |
| `lib/aec/`, `lib/nr/` | Conventional algorithm libraries | Git submodules |

The conventional pipeline is the deployable reference path. `AINR/` and
`AIAEC/` are model-training workspaces and are not silently inserted into that
pipeline.

## Model layout

### Standalone AINR

| Model | Rate / FFT / window / hop | Output composition |
|---|---|---|
| `RNNoise-ERB` | 16 kHz / 512 / 512 / 256 | ERB-band recurrent mask |
| `GTCRN` | 16 kHz / 512 / 512 / 256 | GTCRN complex enhancement |
| `DeepFilterNet2` | 48 kHz / 1024 / 1024 / 512 | full-band ERB mask → low-band DF → learned-alpha residual mix |
| `DeepFilterNet3` | 48 kHz / 1024 / 1024 / 512 | low-band DF and high-band ERB mask in a parallel band split |

DFN2 and DFN3 intentionally share most features and training code but have
different output graphs and different `MODEL_VERSION` values. Their
checkpoints are not interchangeable.

### Neural AEC candidates

`AIAEC/` contains:

- linear-AEC-conditioned RES+NR: `Align_ULCNet`, `GTCRN_AENR`,
  `DeepFilterNet_AENR`;
- end-to-end AEC+RES+NR: `Align_CRUSE`, `DeepVQE_S`, `CAGCRN`.

(`Align_CRUSE` previously ran its own AEC-only, noise-preserving route; that
route was retired and folded into the end-to-end task above — see
`AIAEC/README.md`.)

These candidates use clip-level PyTorch APIs. The 4-channel conventional AEC
shell is a separate integration boundary and does not replicate any neural
model per microphone.

## Signal-grid boundary

| Surface | Supported grid |
|---|---|
| Conventional mono pipeline, 8 kHz | frame/FFT 256, hop 128 |
| Conventional mono pipeline, 16 kHz | frame/FFT 256, hop 128 (default); frame/FFT 512, hop 256 (also supported) |
| Conventional mono pipeline, 48 kHz | frame/FFT 1024, hop 512 |
| Four-channel AEC shell, 16 / 48 kHz | 16k: 256/128 or 512/256; 48k: 1024/512, no padding |
| AIAEC, 16 kHz | frame/window/FFT 512, hop 256 |
| AIAEC, 48 kHz | frame/window/FFT 1024, hop 512 |

The conventional mono path, the 4-channel shell, and AIAEC all use explicit,
zero-padding-free 50%-overlap grids (frame == FFT, hop == FFT/2) — they
remain separate integration surfaces, but none of them derive a 20 ms
frame/10 ms hop any more (that legacy derivation predates the 16 kHz default
grid change; see AEC's CHANGELOG "16 kHz default signal grid" entry).
Standalone AINR follows each model's own config; see the table above.

## Clone and build

```bash
git clone --recursive https://github.com/aaronhsueh0506/Audio_ALG.git
cd Audio_ALG

# If the repository was cloned without submodules:
git submodule update --init --recursive

# Build the conventional mono binaries and libaudio_pipeline.a.
make -C pipelines

# Build the independent four-channel pipeline and its reusable dependencies.
make -C pipelines/4ch_pipelines
make -C pipelines/4ch_pipelines 4aec_nr_res_static

# The same switch is forwarded through every conventional/AI pre-post layer.
# SIMD=1 is the default; SIMD=0 forces scalar fallback for A/B tests.
make -C pipelines SIMD=0 test
make -C pipelines/4ch_pipelines SIMD=0 test
make -C AINR SIMD=0 test
make -C AINR/RNNoise-ERB SIMD=0 test

# Run the malloc reference executable.
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced \
    --nr-preset balanced
```

For the caller-owned-pool API, backend selection, memory contract, presets,
tests, and Python reference path, use the
[pipeline guide](pipelines/README.md). For model training, start from the
README inside the corresponding model directory.

## Documentation

- [Documentation index](docs/README.md)
- [C integration manual](docs/c_user_manual_zh_TW.md)
- [Frequency-domain pipeline design](docs/freq_domain_pipeline_design.md)
- [AIAEC current candidate matrix](docs/ai_aec_candidate_matrix.md)
- [AIAEC / 4-channel / signal-grid audit](docs/aiaec_4ch_signal_grid_review_2026_07_30.md)
- [Development and submodule workflow](docs/development.md)

## Repository structure

```text
Audio_ALG/
├── AIAEC/               # neural AEC/AENR candidates and AEC dataset
├── AINR/                # standalone neural noise reduction
├── docs/                # current docs plus archived design records
├── lib/
│   ├── aec/             # AEC submodule
│   └── nr/              # NR submodule
├── pipelines/           # conventional mono and 4-channel integration
├── scripts/             # repository helpers
└── shared/              # shared utilities
```

Submodule working trees are independent repositories. Commit their changes
inside `lib/aec` or `lib/nr` first, then update the gitlink in this repository
only when that submodule revision is intentionally part of an integration
release.
