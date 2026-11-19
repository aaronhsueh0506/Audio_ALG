# DeepVQE-S

DeepVQE-S consumes microphone and raw far-end audio and predicts complex CCM taps. The host applies the taps to the microphone spectrum. The single-model [PCM stream SDK](../../docs/html/models/deepvqe-s/index.html) handles the two analysis STFTs and output synthesis. For an application that already has spectra, use the [frequency core and state layout](../../docs/html/models/deepvqe-s/reference/core.html).

After a failed job the current spectral contribution is zero. Output PCM can still contain the preceding frame’s overlap. The library moves the state generation after a run that started and failed, a refused result and a reset (and restarts its CCM ring); the sample zeroes the device state when it changes. Report `STREAM_RUN_NOT_STARTED` for a run that never reached the accelerator.

## Training

The supplied config uses AdamW, peak LR `1.2e-3`, weight decay `5e-7`, batch 8 and 50 epochs. It uses the STFT-consistent PLCPA objective. The three-epoch warm-up and per-step cosine schedule end at `1e-6`. [Training and dataset](../reference/training.md)

The model ships one self-contained config per board grid: `config_16k_256.ini`, `config_16k_512.ini` and `config_48k.ini`. `--config` is required, so a grid is always chosen; each file carries its own `[signal]`, `packed_dir` and `output_dir` ([per-grid configs and grid options](../reference/training.md)):

```bash
python3 train.py --config config_16k_512.ini --packed-dir /path/to/packed/all --gpu 0 --mmap
```

The 16 kHz / FFT 256 board row (`stream/aiaec/graph_io_deepvqes.h`, N = 1 only) takes the `config_16k_256.ini` model (`similarity_channels` 8, `score_kernel` 9,5, `gru_hidden` 256: `align_key_ring` `[1,8,63,33]`, `align_score_history` `[1,8,9,63]`, `h_gru` `[1,1,1,256]`); the 16 kHz / FFT 512 and 48 kHz rows take the constructor defaults of `config_16k_512.ini` and `config_48k.ini` (D = 32 and 47). A FFT 256 graph trained with the constructor defaults does not match its row and is refused by the sample's port check.

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

For a graph that runs N frames per call, add `--frames-per-call N` (board
blocks of 1024 samples: N = 4 at 16 kHz / FFT 512, N = 2 at 48 kHz / FFT 1024;
16 kHz / FFT 256 is registered at N = 1 only). The graph and every recorded
tensor then carry N frames, and `--frames` counts calls, so 2048 calls of 4
frames record 8192 frames, as many as above:

```bash
python3 inference.py calib --checkpoint checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 2048 --frames-per-call 4 --format bin --output calib/deepvqe_s_n4
```

## Graph port order

The exported graph lists inputs as the signal inputs (`mic`, `far`) followed by the 16 state inputs, and outputs as the head
(`output`, the CCM taps) followed by the 16 state outputs named `<input>_out`; state `k` is input `2 + k` and output `1 + k`
with equal shapes (18 inputs, 17 outputs). `stream/aiaec` keeps no port list: `StreamWork` carries the block's mic/far views and the CCM
taps, and the 16 state shapes are the board sample's model configuration (`stream/aiaec/graph_io_deepvqes.h`, shared helper `stream/common/graph_io.h`), which binds by position, never by
name. `AIAEC/tests/test_aiaec_bind_info_pin.py` compares the `StreamWork` views of real pre calls with a fresh export and asserts this
name order. Fixed-point state ports need `zero_point` 0 at both ends (the sample's `stream_check_ports()` refuses a graph that does not).
