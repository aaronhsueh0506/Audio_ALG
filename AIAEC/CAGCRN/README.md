# CAGCRN

Backup end-to-end `AEC + RES + NR` candidate, reconstructed from INTERSPEECH
2025 paper 608. It consumes unaligned mic/far spectra directly.

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
