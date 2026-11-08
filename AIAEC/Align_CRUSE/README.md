# Align-CRUSE

Align-CRUSE takes unaligned microphone and far-end spectra and predicts a real mask for the microphone, preserving its phase. Use the [C mask helper and graph contract](../../docs/html/models/align-cruse/index.html) when the application supplies the framing, accelerator and recurrent state.

The project grids use FFT/window/hop 512/512/256 at 16 kHz or 1024/1024/512 at 48 kHz. The hop duration determines the frame count for the one-second delay span. Use `alignment_mode="causal_running"` for streaming. The optional `paper_global` mode needs the complete utterance.

## Training

The default uses Adam with peak LR `1.5e-4` and coupled L2 `5e-6`. The STFT-consistent PLCPA loss uses power `0.3` and phase-aware weight `0.7`. The config specifies batch 16, 50 epochs and a three-epoch warm-up. [Training and dataset](../reference/training.md)

## ONNX and calibration

Run these commands from the model directory.

```bash
python3 export_onnx.py --checkpoint checkpoint.pth \
  --output output/align_cruse.onnx --verify
python3 inference.py calib --checkpoint checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 8192 --format bin --output calib/align_cruse
```

Use `--format npz --output calib/align_cruse.npz` for a NumPy archive.
Align-CRUSE requires one uninterrupted source long enough to satisfy
`--frames`; its cumulative score and frame counter remain outside integer PTQ
as recorded in `manifest.json`.
