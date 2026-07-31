# DeepFilterNet-AENR

Project variant for `linear AEC -> joint RES + NR`. The production linear error
is enhanced; far-end features condition both DFN branches.

Everything after the feature boundary is the audited local DeepFilterNet3-style
implementation: ERB mask head, low-band five-tap complex deep-filter head,
one-frame mask/DF lookahead, parallel low/high band composition, and optional
fixed post-filter. Two 1x1 conditioners fuse `[error, far]` ERB features and
`[error.re,error.im,far.re,far.im]` DF features. They initialize to an exact
error-only pass-through, so a standalone DFN checkpoint can be loaded before
fine-tuning without changing its initial function.

The public forward expects the normalized feature tensors explicitly; generate
each input's features with the same `extract_dfn2_features` configuration and
independent EMA states. Sharing an EMA between error and far is invalid.

Defaults: 48 kHz uses `1024/1024/512`, `df_bins=96`; 16 kHz uses
`512/512/256`, `df_bins=64`. This is not an upstream DeepFilterNet AEC model and
must be reported as a **DeepFilterNet-AENR project variant**.
