# CAGCRN

CAGCRN estimates a complex mask from microphone and raw far-end spectra. Train it on the early, denoised near-end target. Use the [C mask helper and graph contract](../../docs/html/models/cagcrn/index.html) when the application provides its own framing, inference and recurrent state.

The 16 kHz configuration uses FFT/window/hop 512/512/256. It retains linear frequency bins through 2 kHz and applies a differentiable delay-window gate over the far-end history. Checkpoints must match this implementation’s encoder, delay gate and bounded complex mask.

## Training

The supplied config uses AdamW, peak LR `1.2e-3`, weight decay `5e-7`, batch 32 and 50 epochs. After three warm-up epochs, the per-step cosine schedule ends at `1e-6`. The loss combines MSE, SI-SNR and a mean absolute parameter penalty. [Training and dataset](../reference/training.md)

## ONNX and calibration

Run these commands from the model directory.

```bash
python3 export_onnx.py --checkpoint checkpoint.pth \
  --output output/cagcrn.onnx --verify
python3 inference.py calib --checkpoint checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 8192 --format bin --output calib/cagcrn
```

Use `--format npz --output calib/cagcrn.npz` for a NumPy archive. BIN output
contains one directory per graph input and one numbered file per invocation.
