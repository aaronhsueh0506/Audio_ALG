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

Architecture/paper notes are retained in `GTCRN_Analysis.md`.
