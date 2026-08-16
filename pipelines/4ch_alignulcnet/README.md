# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It directly composes the
shared 4-lane linear-AEC core, SRP-PHAT/GSC, Align-ULCNet C pre/post code and
their existing component libraries. The core is initialized in pre-only
mode, so the unused conventional NR/RES post path is not allocated.

The model accelerator is intentionally left as the TODO in
`run_accelerator()`. CPU memory owns every K/V, logit and GRU state tensor.
The default callback failure demonstrates the production fail-open path.

Existing checkpoints require `ULCNET_FAR_RAW`. `ULCNET_FAR_ALIGNED` is a
separate training/deployment contract, not a runtime quality switch.

```sh
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```
