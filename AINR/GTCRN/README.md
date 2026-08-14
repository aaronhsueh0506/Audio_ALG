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
python3 export_calibration.py --model output/gtcrn_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 256 \
  --output output/gtcrn_calibration.npz
python3 export_erb_matrix.py --model output/gtcrn_best.pth \
  --output-dir output/erb --format all
```

The ONNX I/O is `mix` plus `conv_cache`, `tra_cache`, and `inter_cache`, with
the enhanced spectrum and updated caches returned each call. Calibration
captures real pre-frame cache values rather than repeating zero state.
GTCRN's ERB transform is already inside the ONNX graph; the exported ERB
tables are for port verification only and must not be applied a second time.
`gtcrn_process.c/.h` remains the host STFT/WOLA boundary.
