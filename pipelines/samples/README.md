# pipelines/samples

Reference examples only. Nothing in this directory is linked into, or a
dependency of, any production pipeline target (`4ch_aec_bf_nr_res`,
`4ch_aec_bf_dfn_res`, `4ch_alignulcnet`). It is a sibling directory, built
and tested on its own (`make -C samples test`, also wired into the
top-level `pipelines/Makefile`'s own `make test`).

## mcab_demo (`mcab_*` = Multi-Channel AEC-BF)

A small, standalone demo wiring N independent AEC lanes directly to the
SRP-PHAT DOA estimator and the beamformer, calling only the public AEC
(`AEC/c_impl/include/aec.h`), DOA (`audiolib/doa/srp.h`) and beamformer
(`audiolib/bf/bf.h`) APIs -- no production scheduling in between.

**Production code should use `4ch_aec_bf_nr_res` (the real multi-channel
AEC+BF+NR pipeline), not this file.** This demo deliberately omits:

1. duty-cycle scheduling,
2. quarantine, and
3. shared-delay coordination,

all three of which `4ch_aec_bf_nr_res/4aec_nr_res.h`'s production module
implements and this demo intentionally does not re-implement a second time.

See `mcab_demo.c`'s own top-of-file comment for the full design rationale
(API choice, signal model, and why the output is self-evidently non-trivial).

```
make                 # build mcab_demo (SIMD=1 default)
make SIMD=0          # scalar spatial/SIMD kernels
make test            # build + run; mcab_demo itself exits nonzero on any
                      # NaN/zero/degenerate result
make clean
```
