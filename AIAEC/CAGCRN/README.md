# CAGCRN

Backup end-to-end `AEC + RES + NR` candidate, reconstructed from INTERSPEECH
2025 paper 608. It consumes unaligned mic/far spectra directly and uses the
project's common early/dereverberated target; that target extends the paper's
non-dereverberating task.

Implemented claims: ERB band merging/splitting (linear bins retained through
2 kHz); four residual causal encoder blocks per branch with channels
`12,12,12,24` (mic) and `12,24,24,24` (far); CATA between reference encoder
blocks 1 and 2; a separate 24-channel TF-GRU on each branch; 12-hidden-channel
TFAG; a four-block mirrored decoder using both skip streams; and a complex
mask. Including fixed ERB matrices, the 16 kHz state has about 0.068 M values,
matching the paper's 0.07 M class.

The paper used 16 kHz and `512/512/256`, exactly matching the project's 16 kHz
grid. It did **not** publish source, ERB-band count, delay maximum/initial value,
decoder channels, or Mask-block details. More importantly, its proposed
learnable integer `floor(D)` window cannot receive useful ordinary autograd
through a tensor-shape operation. This implementation uses a differentiable
soft delay-window gate over a configurable one-second buffer and a bounded CRM.
Those choices make the architecture trainable but checkpoint-incompatible with
any unpublished author implementation. CAGCRN remains a backup until these
ambiguities are resolved empirically.

## Training recipe

The shipped `config.ini` uses the paper's AdamW settings (`lr=1.2e-3`,
`weight_decay=5e-7`), batch 32, constant LR, and the published MSE + SI-SNR +
L1 objective. Because the paper does not define the L1 normalization, this
implementation uses the mean absolute parameter value and records that choice
in the checkpoint loss version. Both use 10-second examples and batch 32; this
campaign runs 50 epochs rather than the paper's 1000 and therefore has a much
smaller optimization budget. Early stopping is disabled rather than adding an
unpublished stopping rule.

## ONNX and calibration

```bash
python3 export_onnx.py --checkpoint checkpoint.pth \
  --output output/cagcrn.onnx --verify
python3 inference.py calib --checkpoint checkpoint.pth \
  --primary-dir /path/to/microphone --far-dir /path/to/far \
  --frames 8192 --format bin --output calib/cagcrn
```

Use `--format npz --output calib/cagcrn.npz` for a NumPy archive. BIN output
contains one directory per graph input and one numbered file per invocation.
