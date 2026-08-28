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
`weight_decay=5e-7`). Two things in this recipe are **not** from the paper and
are marked as such in the config:

- **The LR schedule.** The paper publishes an optimizer but no schedule. Rather
  than hold LR constant -- which leaves the late epochs taking
  early-epoch-sized steps -- this trainer uses the same per-step linear warmup
  into cosine annealing that `AINR/DeepFilterNet2/train.py` runs
  (`lr_warmup=1e-4` climbing to `lr` over `warmup_epochs=3`, then cosine to
  `min_lr=1e-6`). The schedule is rebuilt from the epoch budget and
  fast-forwarded on resume, never restored from the checkpoint. Weight decay
  stays at the published constant and is not given a schedule of its own.
- **The loss.** The paper does not publish one, so this implementation
  explicitly inherits Align-CRUSE's STFT-consistent PLCPA objective (`c=0.3`,
  `beta=0.7` on the phase-aware complex term).

This campaign uses batch 8 to fit the available GPU and runs the full 50-epoch
budget.

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
