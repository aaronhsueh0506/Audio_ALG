# 4-channel PBFDKF + BF + Align-ULCNet application

This is an application skeleton, not a new library. It directly composes the
shared 4-lane linear-AEC core, SRP-PHAT/GSC, Align-ULCNet C pre/post code and
their existing component libraries. The core is initialized in pre-only
mode, so the unused conventional NR/RES post path is not allocated.

The model accelerator is intentionally left as the TODO in
`run_accelerator()`. CPU memory owns every K/V, logit and GRU state tensor.
The default callback failure demonstrates the production fail-open path.

Existing checkpoints were trained with `ULCNET_FAR_RAW`. This project has
accepted reuse of those weights with `ULCNET_FAR_ALIGNED` after its deployment
sweep. The choice is still an init/export profile rather than a per-hop quality
switch: model descriptor and pipeline wiring must name the same mode.

## Delay profile

The matched-filter bank size `n` is a product deployment decision, so it is a
command-line argument rather than a literal in `main.c`; the resolved profile
and the pool it costs are printed at start-up. `n` is an init parameter, not
a runtime setter — changing it means re-querying the pool and re-initializing.

```sh
./4ch_alignulcnet                              # matched, n=5 (default)
./4ch_alignulcnet --delay-num-filters 3        # smaller bank, smaller pool
./4ch_alignulcnet --delay-mode fixed --fixed-delay 1600
./4ch_alignulcnet --delay-mode external        # caller pre-aligns the far
```

There is exactly ONE matched bank here: the shared estimator in the core. The
four lane AECs run `EXTERNAL_ALIGNED` off its single aligned reference, so
each filter costs 5,728 bytes ONCE — not four times — and the lane pools do
not move with `n` at all (asserted in `tests/test_4aec_nr_res.c`).

Choose `n` from the SKU's measured bulk far-to-mic delay distribution. The
reliable search ceiling per bank is ~125 / 221 / 317 / 413 / 509 ms for
n = 1..5. `0` is not "off" — use `--delay-mode fixed` (delay known at
bring-up) or `external` (upstream guarantees alignment) instead. A bulk delay
beyond the ceiling does not merely fail to lock: with any in-range early
reflection present the estimator can lock onto that instead, at full
confidence, and nothing in the delay seam distinguishes it from a correct
lock — see the known-delay tests.

```sh
make BACKEND=kiss SIMD=0 test
make BACKEND=ne10 SIMD=1 test
```
