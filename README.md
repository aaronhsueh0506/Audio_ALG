# Audio_ALG

Audio processing workspace for conventional AEC/NR integration, standalone
AI noise reduction, and neural AEC research. These are three related but
independent product surfaces; do not mix their signal grids or checkpoint
contracts.

## Current scope

| Area | Purpose | Current entry point |
|---|---|---|
| `pipelines/` | Production-oriented linear AEC + echo-aware NR/RES (mono), plus the DeepFilterNet2 (`dfn/`) and Align-ULCNet (`ulcnet/`) application examples | [`pipelines/README.md`](pipelines/README.md) |
| `AINR/` | Standalone speech enhancement | [`AINR/README.md`](AINR/README.md) |
| `AIAEC/` | Four neural AEC candidate architectures | [`AIAEC/README.md`](AIAEC/README.md) |
| `pipelines/4ch_aec_bf_nr_res/` | `mcab`, the four-microphone linear-AEC flow layer (`mcab.h`): the application creates four libaec `Aec` lanes and, in MATCHED delay mode, one shared `DelayAec`, and passes them in `McabLibs`; `mcab` keeps only flow state (delay admission, alignment ring, fuse buffers) and its per-hop output is `mcab_process_front()` then `mcab_fuse()` with the caller's beamformer weights. It holds no beamformer and no RES. Beside it: sample helpers outside the archive (`mcab_sample_flow`, `mcab_sample_beam` for SRP-PHAT/BF, `mcab_sample_driver`), the programs `mcab_raw`, `mcab_static` and `mcab_external_static`, and the Python twin `McabPipeline` | [`pipelines/4ch_aec_bf_nr_res/README.md`](pipelines/4ch_aec_bf_nr_res/README.md) |
| `lib/aec/`, `lib/nr/` | Conventional algorithm libraries (AEC, NR) | Git submodules (`.gitmodules`) |
| `../audiolib/common/` | Shared FFT / fast-math / SIMD kernel layer used by AEC, NR and the pipelines | Sibling checkout, expected revision in [`pipelines/audiolib.pin`](pipelines/audiolib.pin) (`make -C pipelines check-pins`) |

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
| `DeepFilterNet3` | 48 kHz / 1024 / 1024 / 512 | raw low-band DF + high-band ERB mask → concatenate |

Each model carries its own feature contract and `MODEL_VERSION`; checkpoints
are not interchangeable between them. `DeepFilterNet2` and `DeepFilterNet3`
additionally have a mask/deep-filter lookahead pair (15 legal pairs, shipped
1/1) that is part of the checkpoint contract and is chosen at build time for the C hosts; see
[`AINR/README.md`](AINR/README.md).

### Neural AEC candidates

`AIAEC/` contains:

- linear-AEC-conditioned RES+NR: `Align_ULCNet`;
- end-to-end AEC+RES+NR: `Align_CRUSE`, `DeepVQE_S`, `CAGCRN`.

Every AIAEC candidate is a joint AEC+RES+NR route; there is no AEC-only
candidate (see `AIAEC/README.md`).

These candidates use clip-level PyTorch APIs. The 4-channel conventional AEC
flow (`mcab`) is a separate integration boundary and does not replicate any
neural model per microphone.

## Signal-grid boundary

| Surface | Supported grid |
|---|---|
| Conventional mono pipeline, 8 kHz | frame/FFT 256, hop 128 |
| Conventional mono pipeline, 16 kHz | frame/FFT 256, hop 128 (default); frame/FFT 512, hop 256 (also supported) |
| Conventional mono pipeline, 48 kHz | frame/FFT 1024, hop 512 |
| Four-channel `mcab` flow, 16 / 48 kHz | 16k: 256/128 or 512/256; 48k: 1024/512, no padding |
| AIAEC, 16 kHz | frame/window/FFT 512, hop 256 |
| AIAEC, 48 kHz | frame/window/FFT 1024, hop 512 |

The conventional mono path, the 4-channel flow, and AIAEC all use explicit,
zero-padding-free 50%-overlap grids (frame == FFT, hop == FFT/2) — they
remain separate integration surfaces, but none of them derives a 20 ms
frame/10 ms hop.
Standalone AINR follows each model's own config; see the table above.

## Clone and build

```bash
git clone --recursive https://github.com/aaronhsueh0506/Audio_ALG.git
cd Audio_ALG

# If the repository was cloned without submodules:
git submodule update --init --recursive

# audiolib is a sibling checkout, not a submodule: clone it next to this
# repository at the revision recorded in pipelines/audiolib.pin, then
# verify all three producer revisions.
git clone https://github.com/aaronhsueh0506/audio_common.git ../audiolib
git -C ../audiolib checkout "$(sed -n 's/^AUDIOLIB_COMMIT=//p' pipelines/audiolib.pin)"
make -C pipelines check-pins

# Build the conventional mono example programs, the mono DFN2 and Align-ULCNet
# examples and the DFN2 stage archive.
make -C pipelines

# Build the four-channel flow archive (libaudio_pipeline_mcab.a), the programs
# mcab_raw, mcab_static and mcab_external_static, and the 4-channel DFN2 example.
make -C pipelines/4ch_aec_bf_nr_res

# The same switch is forwarded through every conventional/AI pre-post layer.
# SIMD=1 is the default; SIMD=0 forces scalar fallback for A/B tests.
make -C pipelines SIMD=0 test
make -C pipelines/4ch_aec_bf_nr_res SIMD=0 test
make -C AINR SIMD=0 test
make -C AINR/RNNoise-ERB SIMD=0 test

# Run the malloc reference executable (binaries land in a config-keyed directory).
BIN="$(make -s -C pipelines print-bin-dir)"
"$BIN"/aec_nr_pipeline mic.wav ref.wav out.wav balanced \
    --nr-level balanced
```

For the caller-owned-pool API, backend selection, memory contract, levels,
tests, and Python reference path, use the
[pipeline guide](pipelines/README.md). For model training, start from the
README inside the corresponding model directory.

## Documentation

- [Documentation index](docs/README.md)
- [C integration manual](docs/c_user_manual_zh_TW.md)
- [AEC / NR / audiolib C API quick reference](docs/audio_libraries_c_api_zh_TW.md) (index; chapters in `docs/audio_libraries_c_api_zh_TW_01_*.md` to `_04_*.md`, linked from the index)
- [Frequency-domain pipeline design](docs/freq_domain_pipeline_design.md)
- [AIAEC current candidate matrix](docs/ai_aec_candidate_matrix.md)
- [Development and submodule workflow](docs/development.md)

## Repository structure

```text
Audio_ALG/
├── AIAEC/               # neural AEC candidates and AEC dataset
├── AINR/                # standalone neural noise reduction
├── docs/                # current docs plus archived design records
├── PTQ/                 # post-export graph cleanup and calibration-set analysis
├── lib/
│   ├── aec/             # AEC submodule
│   └── nr/              # NR submodule
├── pipelines/           # conventional integration; tests/ and offline tools/ are separated
├── scripts/             # repository helpers
└── shared/              # shared utilities
```

Submodule working trees are independent repositories. Commit their changes
inside `lib/aec` or `lib/nr` first, then update the gitlink in this repository
only when that submodule revision is intentionally part of an integration
release. `audiolib` follows the same rule without a gitlink: commit and
push it first, then update `AUDIOLIB_COMMIT` in `pipelines/audiolib.pin`
in the same integration commit that bumps the gitlinks.
`make -C pipelines publish` enforces the same pin check inside the publish
lock before it builds any release artifact; `check-pins` remains the quick
standalone preflight command.
