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

The default `split` state layout takes nineteen inputs: `mag`, `real`, `imag`,
six block-local `conv_*` histories, six `h_tra_*` tensors and four per-GRU
`h_dpgrnn*` tensors. The graph returns
the ERB-domain complex mask and every updated state tensor on each call.
Calibration captures real pre-frame state values rather than repeating zero
state.
Use `--format npz --output calib/gtcrn.npz` when a NumPy archive is needed.
BIN output contains one folder per ONNX input and one numbered file per frame.
The deployment ONNX does not contain the fixed ERB front/back end:
`gtcrn_model_input()` applies `erb_fwd.bin` before inference and
`gtcrn_model_output()` applies `erb_inv.bin` plus the complex mask afterward.
`gtcrn_process.c/.h` also owns STFT/WOLA and defines `GTCRNModelState` for
caller-owned state handoff. The model consumes one new STFT frame per
invocation; GTCRN must not be padded to an artificial three-frame input
because its full temporal context already lives in the explicit state
tensors.

`gtcrn_model_state_commit()` is transactional and returns `int`. It validates
every element of every state output before writing any of them, so a single
NaN or Inf anywhere refuses the whole commit with `-1` and leaves the previous
state byte-identical — the caller keeps replaying its last good state instead
of continuing from a half-updated one, which the next invocation could not
distinguish from a healthy state. Callers should check the return value; the
safe fallback is to reuse the previous state or reset. State layout v6 — the
`split` layout — keeps the same 72,192 state bytes as v5 but removes the
graph-side slicing and packing that a shared cache tensor needed. The exported metadata
carries `state_layout_version`, kept numerically equal to
`GTCRN_MODEL_LAYOUT_VERSION` in `gtcrn_process.h`, so an integrator can refuse
a graph whose cache layout no longer matches the struct it allocated.

## State layouts

`--state-layout` selects how the sixteen state tensors are cut at the graph
boundary, on `export_onnx.py` and on `inference.py calib` alike. Both layouts
run the same compute — `CombinedStateGTCRN` wraps `StreamGTCRN` rather than
reimplementing it — and hold the same 72,192 state bytes; only the boundary
and the published `state_layout_version` differ.

| layout | version | inputs | status |
| --- | --- | --- | --- |
| `split` (default) | 6 | 19: three features + sixteen per-slot state tensors | shipped; `GTCRNModelState` binds it |
| `combined` | 7 | 6: three features + `conv_cache`, `h_tra`, `h_dpgrnn` | experimental; no C runtime binds it |

Version 7 is reserved in `gtcrn_process.h` so a later C-side bump cannot take
the same number; the next real bump goes to 8.

The combined groups are the ones whose members already share a shape, and each
is concatenated along an axis it already has, so nothing gains a dimension and
the four-dimensional ceiling holds. On the shipped grid that measures
`conv_cache (2, 16, 16, 33)`, `h_tra (6, 1, 16)` and `h_dpgrnn (4, 33, 8)`:
the six convolution histories are `(1, C, pad, F)` with identical C and F and
differ only in depth, so the encoder's 2+4+10 = 16 and the decoder's
10+4+2 = 16 join along the depth axis, and the two rows then join along the
size-1 batch axis.

```bash
python3 export_onnx.py --model output/gtcrn_best.pth \
  --output output/gtcrn_combined.onnx --state-layout combined --verify
```

Traced node counts: 595 for `split` (29 `Slice`, 24 `Concat`) against 618 for
`combined` (4 `Split`, 31 `Slice`, 29 `Concat`); after the exporter's own
onnxoptimizer and constant-folding passes both land well inside the contract
test's operator budget.
