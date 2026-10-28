# DeepVQE-S

Primary end-to-end `AEC + RES + NR (+ dereverberation)` candidate. Microphone
and unaligned far-end spectra go directly to one causal network; there is no
matched filter or linear AEC in front.

Implemented paper details:

- power-law compressed complex input features;
- S encoder channels: mic `16,40,56,24`, far `8,24`;
- per-frame convolutional cross-attention over a causal delay buffer (the shipped
  checkpoint searches 0.5 s, D=32 at 16 kHz; the constructor default is 1.0 s);
- GRU plus linear-projection bottleneck;
- sub-pixel frequency decoder `40,32,32,27`;
- residual blocks only in the two middle decoder blocks;
- causal complex convolving mask with `3` past/current time positions and
  `3` frequency positions, using the paper's three 120-degree vectors. The
  exported graph folds the three-vector-to-complex map into the last
  convolution's weights, so it ends in that conv plus a rank-4 reshape and
  emits the `[1, 1, bins, 18]` taps directly.

The paper used 24 kHz, `480/480/240`, and `dmax=100`. This project adaptation
uses selectable 16/48 kHz power-of-two grids and the constructor default
preserves the one-second physical delay span (the shipped model was trained
with `max_delay_seconds = 0.5`). The paper did not publish code, loss details, GRU width,
similarity-head count, or a checkpoint. `gru_hidden=192` is inferred from the
published 0.59 M small-model class (the 16 kHz power-of-two adaptation is about
0.63 M); four similarity channels and the factorization `27=3*3*3` remain
explicit reconstruction choices. It is architecture-faithful, not checkpoint
or bit exact.

## Training recipe

The shipped `config.ini` uses the paper's AdamW settings (`lr=1.2e-3`,
`weight_decay=5e-7`). The shipped config adds the shared three-epoch warmup from
`1e-4`, followed by per-step cosine decay to `1e-6`. This is an explicit project
choice; the paper does not publish its LR schedule.

The paper trains batch 400 for 250 epochs. The shipped config uses batch 8
(GPU limit) for 50 epochs; those are substantial reproduction gaps. Gradient accumulation is not
presented as an exact substitute because the BatchNorm layers would still see
micro-batches of eight. The paper does not publish its training-example
duration, dataset hours, or steps per epoch, so there is no defensible
batch/time-normalised LR or epoch conversion. The published `1.2e-3` remains
the peak LR rather than inventing a linear or square-root batch scaling rule.

The other reconstruction choice is the loss. The paper does not publish one,
so this implementation
  explicitly inherits Align-CRUSE's STFT-consistent PLCPA objective (`c=0.3`,
  `beta=0.7` on the phase-aware complex term).

Early stopping is disabled; validation continues to select `*_best.pth`.

## C deployment boundary

`deepvqe_prepost.h/.c` is the integrator's entry point: one opaque
`DeepVqePrepost` composing the shared `aiaec_process.c` STFT and the
`deepvqe_process.c` CCM kernel; in `DEEPVQE_IO_TIME` the synthesis is the application's
audiolib `AcIstft` (`cfg.istft`, built from `deepvqe_framing_config()`). The lifecycle and per-hop sequence shared by every
class are described once, in `../README.md` ("C pre/post-processing").
Specific to this one: the accelerator boundary is the exporter's exactly --
power-law compressed `mic`/`far` in (planar `[1,2,1,F]`, real plane then
imaginary plane), packed CCM taps `[1,1,F,18]` out, and the sixteen explicit
rank-4 state tensors in `DeepVqeStateId` order (conv windows of 4 frames, the
alignment score window of 5, attention rings of D, `h_gru` `[1,1,1,192]`). The compression
(`x * |x|^(0.3-1)`, `|x| = sqrt(re^2 + im^2 + 1e-12)`) runs on the host in
`deepvqe_prepost_frame_inputs()`, so the graph starts at its first convolution;
the class keeps the raw microphone spectrum separately for the CCM ring. The
exponent is compiled into the host (`DEEPVQE_COMPRESSION_EXP`) and the
exporter refuses a checkpoint whose `compression_exponent` differs. The taps'
last axis preserves `[time][frequency][RI]` row-major order, so C consumes it
without a transpose or copy. The pool holds one host buffer per state tensor.
Every state comes back from the graph as a full window with its input's shape
(a Slice on the input window drops one frame and the new frame is
concatenated, so no state output is produced by a Slice), and the runtime
binds each `*_out` output to the pointer `frame_inputs` returned for its state
input; the host copies and swaps nothing, and the runtime must finish reading
every state input before it writes any state output at that address.
`frame_commit` validates (every tap, `h_gru`, and the newest frame of every
window: slot 0 of an attention ring, the last frame of a conv or score window)
and advances the CCM ring; on a non-finite value it zeroes all sixteen state
tensors, since the write cannot be rolled back, and
`deepvqe_prepost_state_generation()` advances (it also advances on
`deepvqe_prepost_reset`) so a board with fixed-point accelerator state can
refill it with quantized zeros. `frame_skip` leaves the state as the
accelerator wrote it and FAILS CLOSED (mutes the frame): stream 0 is the raw
microphone, so the pass-through identity a post-filter can take would emit the
uncancelled echo here.

The exporter writes `state_layout_version` (`DEEPVQE_PREPOST_LAYOUT_VERSION`,
currently 4; the exporter's `DEEPVQE_C_LAYOUT_VERSION` is the same number) and
a `c_descriptor` block measured from the built graph into the ONNX
metadata and the sidecar JSON; a board feeds that block to
`deepvqe_prepost_descriptor_validate()` before binding. The class gate is
`../tests/test_deepvqe_prepost_c.py`, which also pins the header, the exporter
table and the graph's state shapes to each other.

## ONNX and calibration

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
