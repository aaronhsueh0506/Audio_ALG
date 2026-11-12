# DeepVQE-S

DeepVQE-S consumes microphone and raw far-end audio and predicts complex CCM taps. The host applies the taps to the microphone spectrum. The single-model [PCM stream SDK](../../docs/html/models/deepvqe-s/index.html) handles the two analysis STFTs and output synthesis. For an application that already has spectra, use the [frequency core and state layout](../../docs/html/models/deepvqe-s/reference/core.html).

After a failed job the current spectral contribution is zero. Output PCM can still contain the preceding frame’s overlap. Refill device state if a failed accelerator run may have partly updated it.

## Training

The supplied config uses AdamW, peak LR `1.2e-3`, weight decay `5e-7`, batch 8 and 50 epochs. It uses the STFT-consistent PLCPA objective. The three-epoch warm-up and per-step cosine schedule end at `1e-6`. [Training and dataset](../reference/training.md)

## ONNX and calibration

Run these commands from the model directory.

```bash
python3 export_onnx.py --checkpoint checkpoint.pth \
  --output output/deepvqe_s.onnx --verify
python3 inference.py calib --checkpoint checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 8192 --format bin --output calib/deepvqe_s
```

Use `--format npz --output calib/deepvqe_s.npz` for a NumPy archive. BIN
output contains one directory per graph input and one numbered file per
streaming invocation. The `mic`/`far` tensors it records are the host-compressed
planar spectra the graph binds (layout version 4).

## Graph port order

The exported graph lists inputs as the signal inputs (`mic`, `far`) followed by the 16 state inputs, and outputs as the head
(`output`, the CCM taps) followed by the 16 state outputs named `<input>_out`; state `k` is input `2 + k` and output `1 + k`
with equal shapes (18 inputs, 17 outputs). `stream/aiaec` keeps no port list: `AiaecWork` carries the block's mic/far views and the CCM
taps, and the 16 state shapes are the board sample's model configuration (`stream/aiaec/ai3_model_io_deepvqes.h`, shared helper `stream/common/ai3_model_io.h`), which binds by position, never by
name. `AIAEC/tests/test_aiaec_bind_info_pin.py` compares the `AiaecWork` views of real pre calls with a fresh export and asserts this
name order. Fixed-point state ports need `zero_point` 0 at both ends (the sample's `stream_check_ports()` refuses a graph that does not).
