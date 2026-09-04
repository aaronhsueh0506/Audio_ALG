# DeepVQE-S

Primary end-to-end `AEC + RES + NR (+ dereverberation)` candidate. Microphone
and unaligned far-end spectra go directly to one causal network; there is no
matched filter or linear AEC in front.

Implemented paper details:

- power-law compressed complex input features;
- S encoder channels: mic `16,40,56,24`, far `8,24`;
- per-frame convolutional cross-attention over a one-second causal delay buffer;
- GRU plus linear-projection bottleneck;
- sub-pixel frequency decoder `40,32,32,27`;
- residual blocks only in the two middle decoder blocks;
- causal complex convolving mask with `3` past/current time positions and
  `3` frequency positions, using the paper's three 120-degree vectors.

The paper used 24 kHz, `480/480/240`, and `dmax=100`. This project adaptation
uses selectable 16/48 kHz power-of-two grids and preserves the one-second
physical delay span. The paper did not publish code, loss details, GRU width,
similarity-head count, or a checkpoint. `gru_hidden=192` is inferred from the
published 0.59 M small-model class (the 16 kHz power-of-two adaptation is about
0.63 M); four similarity channels and the factorization `27=3*3*3` remain
explicit reconstruction choices. It is architecture-faithful, not checkpoint
or bit exact.

## Training recipe

The shipped `config.ini` uses the paper's AdamW settings (`lr=1.2e-3`,
`weight_decay=5e-7`). The local campaign adds the shared three-epoch warmup from
`1e-4`, followed by per-step cosine decay to `1e-6`. This is an explicit project
choice; the paper does not publish its LR schedule.

The paper trains batch 400 for 250 epochs. This campaign deliberately uses
batch 8 (GPU limit) for 50 epochs (requested budget); those are substantial
reproduction gaps and are recorded as such. Gradient accumulation is not
presented as an exact substitute because the BatchNorm layers would still see
micro-batches of eight. The paper does not publish its training-example
duration, dataset hours, or steps per epoch, so there is no defensible
batch/time-normalised LR or epoch conversion. The published `1.2e-3` remains
the peak LR rather than inventing a linear or square-root batch scaling rule.

The other reconstruction choice is the loss. The paper does not publish one,
so this implementation
  explicitly inherits Align-CRUSE's STFT-consistent PLCPA objective (`c=0.3`,
  `beta=0.7` on the phase-aware complex term).

Early stopping is disabled for the 50-epoch campaign; validation continues to
select `*_best.pth`.

## C deployment boundary

`deepvqe_prepost.h/.c` is the integrator's entry point: one opaque
`DeepVqePrepost` composing the shared `aiaec_process.c` STFT/WOLA and the
`deepvqe_process.c` CCM kernel (neither changes, so their standalone parity
tests keep linking). The lifecycle and per-hop sequence shared by every
class are described once, in `../README.md` ("C pre/post-processing").
Specific to this one: the accelerator boundary is the exporter's exactly --
raw RI `mic`/`far` in, packed CCM taps `[1,1,F,18]` out, and the sixteen
explicit state tensors in `DeepVqeStateId` order. The last axis preserves
`[time][frequency][RI]` row-major order, so C consumes it without a transpose
or copy. State is held in two host-side banks that `frame_commit`
swaps only after every tap and state element is finite -- and `frame_skip`
FAILS CLOSED (mutes the frame): stream 0 is the raw microphone, so the
pass-through identity a post-filter can take would emit the uncancelled
echo here.

The exporter writes `state_layout_version` (`DEEPVQE_PREPOST_LAYOUT_VERSION`)
and a `c_descriptor` block measured from the built graph into the ONNX
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
streaming invocation.
