# GTCRN

Compact 16 kHz noise-reduction model on the project `512/512/256`
FFT/window/hop grid. Training and offline inference use a periodic root-Hann
window and unnormalised complex STFT, matching the model's `[B,F,T,2]`
real/imag boundary.

`gtcrn_process.h/.c` implements the deployment-side streaming analysis and
WOLA synthesis. The model inference itself is intentionally outside this C
module and may run on the target accelerator. The streaming API consumes and
emits one 256-sample hop per call; its initial boundary is zero-padded rather
than using PyTorch's offline centered reflect padding.

Run the shared C parity and round-trip gate from `AINR/`:

```bash
make test-simd
make SIMD=0 test
```

Architecture/paper notes are retained in `docs/architecture_notes.md`.

## ONNX, ERB tables and calibration

Install `../requirements-export.txt` in addition to the normal model
dependencies. Unlike the offline training graph, the exported ONNX consumes
one STFT frame and carries every temporal cache explicitly, so it can run
continuously without resetting model history:

```bash
python3 export_onnx.py --model output/gtcrn_best.pth \
  --output output/gtcrn_stream.onnx --verify
python3 inference.py calib --model output/gtcrn_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 8192 --format bin \
  --output calib/gtcrn
python3 export_erb_matrix.py --model output/gtcrn_best.pth \
  --output-dir output/erb --format all
```

The ONNX I/O is `mix` plus `conv_cache`, `tra_cache`, and `inter_cache`, with
the enhanced spectrum and updated caches returned each call. Calibration
captures real pre-frame cache values rather than repeating zero state.
Use `--format npz --output calib/gtcrn.npz` when a NumPy archive is needed.
BIN output contains one folder per ONNX input and one numbered file per frame.
GTCRN's ERB transform is already inside the ONNX graph; the exported ERB
tables are for port verification only and must not be applied a second time.
`gtcrn_process.c/.h` remains the host STFT/WOLA boundary and defines
`GTCRNModelState` for caller-owned cache handoff. The model consumes one new
STFT frame per invocation; GTCRN must not be padded to an artificial
three-frame input because its full temporal context already lives in the
explicit cache tensors.

`gtcrn_model_state_commit()` is transactional and returns `int`. It validates
every element of all three caches before writing any of them, so a single NaN
or Inf anywhere refuses the whole commit with `-1` and leaves the previous
state byte-identical — the caller keeps replaying its last good state instead
of continuing from a half-updated one, which the next invocation could not
distinguish from a healthy state. Callers should check the return value; the
safe fallback is to reuse the previous state or reset. The exported metadata
carries `state_layout_version`, kept numerically equal to
`GTCRN_MODEL_LAYOUT_VERSION` in `gtcrn_process.h`, so an integrator can refuse
a graph whose cache layout no longer matches the struct it allocated.
