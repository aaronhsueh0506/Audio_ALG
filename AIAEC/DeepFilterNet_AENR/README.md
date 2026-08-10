# DeepFilterNet-AENR

Project variant for `linear AEC -> joint RES + NR`. The production linear error
is enhanced; far-end features condition both DFN branches. Its target is the
common denoised, echo-free, early/dereverberated near-end speech.

Everything after the feature boundary follows the current local DeepFilterNet2
cascade/alpha graph: a full-band ERB mask, a low-band five-tap complex deep
filter applied to the masked spectrum, and a learned sigmoid-alpha residual
blend between the deep-filtered and masked low bands. Mask and DF lookahead are
one frame; the optional fixed post-filter is disabled by default.

Two 1x1 conditioners fuse `[error, far]` ERB features and
`[error.re,error.im,far.re,far.im]` DF features. They initialize to an exact
error-only pass-through.

The public forward expects the normalized feature tensors explicitly; generate
each input's features with the same `extract_dfn2_features` configuration and
independent EMA states. Sharing an EMA between error and far is invalid.

Defaults: 48 kHz uses `1024/1024/512`, `df_bins=96`; 16 kHz uses
`512/512/256`, `df_bins=64`. This is not an upstream DeepFilterNet AEC model and
must be reported as a **DeepFilterNet-AENR project variant**.

Checkpoint initialization must match the standalone DFN2 v6 model/feature
contract and tensor shapes. A DFN3 v5 band-split checkpoint belongs in
`AINR/DeepFilterNet3` and cannot initialize this composition safely.

See [`../../AINR/DeepFilterNet2/README.md`](../../AINR/DeepFilterNet2/README.md)
for the baseline graph and checkpoint rules.
