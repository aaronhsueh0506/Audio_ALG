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

Architecture/paper notes are in `docs/architecture_notes.md`.

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

The C runtime loads the files written by `export_erb_matrix.py --runtime-bins DIR`;
the `--output-dir` formats are for toolchain validation.

The graph takes eleven inputs: `mag`, `real`, `imag`, six `conv_*` windows,
`h_tra` and `h_dpgrnn`. It returns
the ERB-domain complex mask and every updated state tensor on each call.
Calibration captures real pre-frame state values rather than repeating zero
state.
Use `--format npz --output calib/gtcrn.npz` when a NumPy archive is needed.
BIN output contains one folder per ONNX input and one numbered file per frame.
The deployment ONNX does not contain the fixed ERB front/back end:
`gtcrn_model_input()` applies `erb_fwd.bin` before inference and
`gtcrn_model_output()` applies `erb_inv.bin` plus the complex mask afterward.
`gtcrn_process.c/.h` also owns STFT/WOLA and defines `GTCRNModelState` for
caller-owned state. The model consumes one new STFT frame per
invocation; GTCRN must not be padded to an artificial three-frame input
because its full temporal context already lives in the explicit state
tensors.

`GTCRNModelState` is the state buffer itself. Bind each state input and its
`*_out` output to the same field (`&state->conv_enc0` ... `&state->conv_dec2`,
`&state->h_tra`, `&state->h_dpgrnn`), so the updated state is the next call's
input and the CPU copies nothing between invocations. The runtime must read
every state input before it writes any state output at the same address.
After each invocation `gtcrn_model_state_validate()` finite-checks the whole
state in place and returns `int`: `0` when every element is finite, `-1` on a
null argument or on any NaN/Inf, in which case the whole state is zeroed (it
restarts from zero) because the previous value no longer exists. A partial or mixed write that
stays finite is not detectable.

The library zeroes the state on `gtcrn_model_state_reset()` and on a refused
`gtcrn_model_state_validate()`, but the accelerator keeps its own fixed-point
copy that neither reaches. `gtcrn_model_state_generation()` returns 1 after
`gtcrn_model_state_init()` and advances by one on every reset and every refused
validation (modulo 2^32): a runtime that sees a value different from the one at
the previous frame refills its state tensors with quantized zeros.

## State boundary

State layout version 8. The exported metadata carries `state_layout_version`,
kept numerically equal to `GTCRN_MODEL_LAYOUT_VERSION` in `gtcrn_process.h`, so
an integrator can refuse a graph whose state layout no longer matches the
struct it allocated. Versions 3 to 7 are retired
(`RETIRED_LAYOUT_VERSIONS` in `export_onnx.py`). The metadata also records
`state_binding = in_place_same_address`, `state_handoff` and `state_slicing`.

Every state tensor is rank 4 and has the same shape at its input and at its
`*_out` output:

| tensor | shape | meaning |
| --- | --- | --- |
| `conv_enc0`, `conv_enc1`, `conv_enc2` | `(1, 16, 3 / 5 / 11, 33)` | full causal window of each encoder depthwise convolution |
| `conv_dec0`, `conv_dec1`, `conv_dec2` | `(1, 16, 11 / 5 / 3, 33)` | the same for the decoder, in reverse order |
| `h_tra` | `(1, 6, 1, 16)` | the six TRA attention GRUs, encoder 0-2 then decoder 0-2 |
| `h_dpgrnn` | `(1, 4, 33, 8)` | the four grouped inter GRUs, `h_dpgrnn1_0`, `h_dpgrnn1_1`, `h_dpgrnn2_0`, `h_dpgrnn2_1`; the hidden batch is the frequency axis |

A window input has exactly one consumer, a Slice of axis 2 with an explicit
integer range `[1:K]` and step 1. It drops the oldest frame; the graph appends
the new frame and returns the whole concatenation, so no state output is
produced by a Slice. The stacked GRU tensors are cut with Slices on dim 1 and
joined by one Concat on dim 1. `onnx_streaming_contract.py`
(`validate_inplace_state_graph`) checks all of this at export time;
`check_state_quantization.py` repeats the check on an exported graph and, given
a quantization report, aborts when a state input and its output differ in
dtype, scale, zero point or shape.

Total state is 84,864 bytes; `sizeof(GTCRNModelState)` is 84,868 with the
generation counter. A window row is 33 elements, so on an int8 target the
axis-2 Slice starts at a multiple of 33 bytes; how the toolchain aligns it is
not verified on a board.
