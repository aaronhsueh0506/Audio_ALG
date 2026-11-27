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

To let the capture choose the patterns from every recording instead of taking
the first `--frames` calls, give `--select COUNT` (and `--jobs N` to scan N
recordings at a time; the result is the same): it writes the COUNT calls a
selection keeps and an analysis page, as `PTQ/calib_analyze.py analyze` would
choose them from a capture of every call (see `AIAEC/reference/export.md`).

### Cross-layer equalization (optional, before PTQ)

`equalize.py` rescales channels between consecutive layers so that the
channels of each weight tensor span similar ranges (cross-layer equalization,
Nagel et al. 2019), for a converter that quantizes weights to 8 bits. The FP32
output does not change; the tool checks that before it writes anything. Run
`export_onnx.py` and `inference.py calib` on the written checkpoint as usual:

```bash
python3 equalize.py checkpoint.pth -o checkpoint_cle.pth
python3 inference.py calib --checkpoint checkpoint_cle.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 8192 --format bin --output calib/deepvqe_s_cle
```

- Only `activation = relu`, `leaky_relu` or `prelu` checkpoints are accepted;
  ELU does not commute with a channel scale and is refused.
- The state inputs carry the rescaled channels, so the graph and its
  calibration data must both come from the equalized checkpoint. A
  calibration set captured from the original checkpoint does not fit the
  equalized graph.
- The tool prints the output SNR with 8-bit weights before and after, with one
  scale per tensor and with one scale per channel, on a random input.
  - CLE helps a converter with one weight scale per tensor when some channels
    are far larger than the rest.
  - Otherwise the change can go either way. On random test weights, one scale
    per channel moved by −6 to +4 dB.
  - Keep the equalized checkpoint only if the row that matches the
    converter's weight scaling improves, and turn the converter's own CLE off
    when you compare.
- A channel that is scaled up raises its activation, and the state that
  holds it, by the same factor, bias included.
  - The largest case is a channel with a near-zero gain but a real bias,
    which would grow about a thousandfold.
  - Every channel's total scale is therefore held to 1/16..16 (`MAX_SCALE`).
    The tool reports how many channels stopped at that bound.
  - It also prints the three internal tensors whose peak grew most on its
    random input.
  - With 16-bit activations a factor of 16 at most costs little.
  - To see the real ranges, compare both graphs with
    `PTQ/calib_analyze.py analyze` (it replays the graph beside the
    artifact).
- The written file holds the contract and the weights, without optimizer
  state. It is meant for export and calibration, not for further training.
  - Its `cross_layer_equalization` entry records the source checkpoint's
    SHA-256.
  - If the source file changes while the tool reads it (a training run
    saving over it), nothing is written.
- The GRU's recurrent weights (`W_hh`) and the attention score conv are
  outside every group and do not change.

## Graph port order

The exported graph lists inputs as the signal inputs (`mic`, `far`) followed by the 16 state inputs, and outputs as the head
(`output`, the CCM taps) followed by the 16 state outputs named `<input>_out`; state `k` is input `2 + k` and output `1 + k`
with equal shapes (18 inputs, 17 outputs). `stream/aiaec` keeps no port list: `StreamWork` carries the block's mic/far views and the CCM
taps, and the 16 state shapes are the board sample's model configuration (`stream/aiaec/graph_io_deepvqes.h`, shared helper `stream/common/graph_io.h`), which binds by position, never by
name. `AIAEC/tests/test_aiaec_bind_info_pin.py` compares the `StreamWork` views of real pre calls with a fresh export and asserts this
name order. Fixed-point state ports need `zero_point` 0 at both ends (the sample's `stream_check_ports()` refuses a graph that does not).
