# Align-CRUSE

Direct neural AEC/RES candidate. There is no matched filter or linear AEC in
front of this network. Inputs are unaligned microphone and far-end spectra; the
output is a real magnitude mask applied to the microphone spectrum, preserving
microphone phase. This project runs it as an **end-to-end AEC+RES+NR**
candidate: the training target is `near_target` (denoised, dereverberated,
echo-cancelled near speech), the same joint task and target DeepVQE-S uses.
This candidate's earlier standalone AEC-only route (target `near_speech +
local_noise`, noise deliberately preserved) was retired -- there is no more
AEC-only candidate in this project.

Paper-aligned details:

- log-power features from microphone and far end;
- mic encoder channels `16,40,72,32`, far encoder `8,24`;
- causal `4x3`, frequency-stride-2 convolutions;
- causal-running soft delay distribution and weighted far-feature alignment;
- GRU bottleneck, trainable skip projections, decoder `32,48,48`;
- sigmoid magnitude mask with learned output gain.

The paper used 16 kHz, `320/320/160` and `dmax=100`. This repository preserves
the one-second delay span while using the project grid (`512/512/256` at 16 kHz
or `1024/1024/512` at 48 kHz), so the number of delay frames is derived from the
hop rather than copied as 100. The deployment default
`alignment_mode="causal_running"` emits `D[B,T,dmax]` using past evidence
only. Select `paper_global` explicitly to reproduce the paper's one
`D[B,dmax]` vector; that mode reads the complete utterance and is not causal.

All four frequency restorations use `1x3`, stride-2 transposed convolutions;
the fourth is the shape-restoring convolution in the mask block. The default
192-unit GRU keeps the 16 kHz project-grid graph in the paper's approximately
0.75 M parameter class. No author code/checkpoint was released, so exact
padding and projection width remain documented reconstruction choices.

## Training recipe

The shipped `config.ini` uses the paper's Adam settings (`lr=1.5e-4`, coupled
L2 `weight_decay=5e-6`) and STFT-consistent PLCPA (`c=0.3`, `beta=0.7` on the
phase-aware complex term). LR is constant because the paper reports no
scheduler. Batch 16 is the project memory-fit setting and this campaign runs
50 epochs instead of the paper's batch 400 / 150 epochs. The paper does not
publish example duration or steps per epoch, so the LR is not rescaled from an
unknown exposure ratio. Early stopping is disabled; validation still selects
the best checkpoint.

## ONNX and calibration

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
