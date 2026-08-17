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

The production ONNX boundary is stateless from the accelerator's perspective.
Each call receives four `[t-1,t,t+1]` feature windows (error/far ERB and
complex), the three GRU states, and the four-frame DF-pathway cache. It returns
one set of heads for frame `t` plus every next-state tensor. The host-side
windows and state handoff are implemented by `DfnAenrModelIOState` in
`dfn_aenr_process.c/.h`; pushing one zero feature pair flushes the final real
frame. State commit validates the complete accelerator result first: a null
or non-finite tensor returns `-1` and preserves the last good state. The
accelerator must not retain hidden state between calls.

That struct keeps its own field layout (four feature windows instead of one),
but every extent comes from the `DFN2_MODEL_*` macros and the window-slide and
state-commit work is done by the exported `dfn2_model_io.c/.h` helpers rather
than a second copy of them, so `dfn2_model_io.o` links into
`libaiaec_prepost.a` alongside `dfn2_process.o`. `DFN_AENR_MODEL_IO_LAYOUT_VERSION`
tracks `DFN2_MODEL_IO_LAYOUT_VERSION`; the C parity test asserts both the
version and every inherited field size.

Defaults: 48 kHz uses `1024/1024/512`, `df_bins=96`; 16 kHz uses
`512/512/256`, `df_bins=64`. This is not an upstream DeepFilterNet AEC model and
must be reported as a **DeepFilterNet-AENR project variant**.

Checkpoint initialization must match the standalone DFN2 v6 model/feature
contract and tensor shapes. A DFN3 v5 band-split checkpoint belongs in
`AINR/DeepFilterNet3` and cannot initialize this composition safely.

See [`../../AINR/DeepFilterNet2/README.md`](../../AINR/DeepFilterNet2/README.md)
for the baseline graph and checkpoint rules.
