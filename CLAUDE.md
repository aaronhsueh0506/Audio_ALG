# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Three independent product surfaces that share build conventions but must never share signal
grids or checkpoint contracts:

- **`pipelines/`** — the deployable conventional AEC + echo-aware NR/RES path (mono and
  4-channel), plus the DeepFilterNet2 (`pipelines/dfn/`) and Align-ULCNet
  (`pipelines/ulcnet/`) application examples that sit on top of it. This is the reference
  integration path; `AINR/` and `AIAEC/` models are not silently wired into it.
- **`AINR/`** — standalone AI speech-enhancement models (RNNoise-ERB, GTCRN, DeepFilterNet2,
  DeepFilterNet3): Python training/export plus a hand-written C runtime per model.
- **`AIAEC/`** — four neural AEC candidate architectures (Align_ULCNet, Align_CRUSE,
  DeepVQE_S, CAGCRN), every one a joint AEC+RES+NR route (no AEC-only candidate).

## External dependencies: mid-transition, read this before touching build paths

AEC and NR used to be pure git submodules (`lib/aec`, `lib/nr`); audiolib (`common/` = shared
FFT/fast-math/SIMD) used to be resolved by a single hardcoded pin file. As of a recent
"dependency-path convergence" pass, the **default** build source for all three is sibling dev
checkouts next to this repo (`../AEC`, `../NR`, `../audiolib`), resolved by `lib/deps.mk` against
`lib/aec.pin` / `lib/nr.pin` / `lib/audiolib.pin`. The `lib/aec` and `lib/nr` git submodules still
exist, still carry a gitlink, and still build standalone — they are a second, not-yet-retired
record of the same two dependencies (`AEC_DIR=lib/aec/c_impl NR_DIR=lib/nr/c_impl` to force the
submodule checkout). Migrating the two gitlinks onto the `.pin` files and retiring the submodule
is a deliberately separate, not-yet-done step. Any Makefile or script that needs AEC/NR/audiolib
paths should `include`/call `lib/deps.mk` rather than hardcoding a relative path a second time —
that duplication is exactly what silently drifted out of sync before.

```bash
make -f lib/deps.mk bootstrap-deps   # clone whichever of ../AEC, ../NR, ../audiolib is missing, pinned
make -C pipelines check-pins         # verify AEC_DIR/NR_DIR/AUDIOLIB_DIR match their .pin (+ gitlink drift note)
```

## Common commands

```bash
# Build conventional mono examples + the DFN2 stage archive
make -C pipelines
# Build the 4-channel mcab flow archive + examples
make -C pipelines/4ch_aec_bf_nr_res

# Tests (every C library/pipeline target takes SIMD=1 (default) / SIMD=0 (forced scalar, for
# NEON-vs-scalar A/B parity) and WERROR=1 (promote warnings to errors, the closest thing to lint)):
make -C pipelines test
make -C pipelines SIMD=0 test
make -C pipelines/4ch_aec_bf_nr_res test
make -C AINR test               # also: test-simd (NEON+scalar in one run), test-pairs (all 15 DFN lookahead pairs)
make -C AINR/RNNoise-ERB test
make -C AIAEC test

# A single test binary, once built, runs standalone (binaries land in a config-keyed directory):
BIN="$(make -s -C AINR print-bin-dir)"; "$BIN"/test_c_prepost

# Python side (training/export models, and the C<->Python parity suites) — run from repo root:
python3 -m pytest -q AINR/tests/test_dfn2_prepost_c.py           # a single file
python3 -m pytest -q AINR/tests/test_dfn2_prepost_c.py -k warmup # a single test by name
python3 -m pytest -q pipelines/tests/test_doc_layout_versions.py pipelines/tests/test_c_python_map.py
```

`make -C pipelines publish` re-checks pins before building any release artifact;
`check-pins` is the quick standalone preflight. `print-bin-dir` / `print-obj-dir` / `print-lib-path`
exist on every Makefile for locating the config-keyed build output without guessing the hash.

## Architecture that spans multiple files

**The three-file model pattern.** Every AINR/AIAEC model (DFN2, DFN3, GTCRN, RNNoise-ERB,
ULCNet, DeepVQE-S, ...) splits into `*_process.{c,h}` (feature normalization, mask/deep-filter
math — pure, stateless-ish numerics), `*_model_io.{c,h}` (tensor windowing/batching and
recurrent-state registration) and `*_prepost.{c,h}` (the frequency-domain per-hop lifecycle:
decides which frame's heads are ready, owns `frame_open`/`frame_finish`/`frame_bypass`). A new
`*_stream.c` layer (in progress, DFN2/Align-ULCNet/DeepVQE-S only so far) wraps PCM framing/OLA
around `*_prepost` so a board only needs the stream header. Reading just `*_prepost.h` is usually
enough to integrate a model; `*_process.c` is only needed to change the model's math itself.

**Per-hop contract (same shape across every model).** `pre_process_freq()` → `frame_open()` →
accelerator fills the tensors → `frame_finish()` (or `frame_bypass()` on failure/refusal) →
`post_process_freq()` → push to `AcIstft`. The first `mask_lookahead` hops after init/reset are
warm-up frames: they run this exact same flow (not a shortcut path) but their heads are discarded,
registered state is zeroed, and the state generation counter advances by exactly one, so the first
real frame always starts from zero state and the audio is bit-identical to skipping them. Missing
a `frame_finish`/`frame_bypass` before the next `pre_process`/`post_process` call is a caller bug,
not a library bug — the next call returns -1.

**State ownership and the generation counter.** Recurrent state is either registered by the host
(pointer+count pairs, all-or-none) and owned in caller memory, or left to a hardware accelerator
that tracks its own quantized copy. A `*_state_generation()` counter advances on every reset, every
warm-up frame, and every refused commit; any accelerator-side state must be reloaded with
quantized zero whenever it sees the generation change (not just at init/reset) or it will carry
stale state into the next real frame.

**Caller-owned memory.** `*_grid_make(spec)` → `*_get_mem_size(cfg)` → caller allocates one block
→ `*_init`/`*_init_ex`. No internal `malloc`. A `CARVE_VERSION`/layout-version constant and a
descriptor hash must bump together whenever a struct's carved layout changes size — both are
checked by `pipelines/tests/test_doc_layout_versions.py` and guard against a stale caller pool
being handed to a newer library build.

**Frequency domain only inside models.** Framing, windowing, FFT/IFFT belong to the application
via audiolib's `AcStft`/`AcIstft`; model libraries only ever see/produce spectra. A pipeline stage
that itself performs STFT/ISTFT internally (`dfn_res_stage`, `dfn_rate_bridge`,
`ulcnet_nn_stage`) is the exception that bridges a model into a different grid/host, not the norm.

**`mcab` (4-channel flow, `pipelines/4ch_aec_bf_nr_res/mcab.h`) owns no library instances.** The
application creates four `libaec` `Aec` lanes (plus one shared `DelayAec` in MATCHED delay mode)
and injects them via `McabLibs`; `mcab` itself holds only flow state (delay admission, alignment
ring, fuse buffers) and exposes no beamformer. This is deliberate — `mcab` is pipeline/flow
machinery, not a `lib/`-shaped algorithm core, and library tuning parameters stay outside it by
design, not by oversight.

**Python <-> C parity is enforced, not aspirational.** Changes to model math are expected
Python-first; the C mirror is then checked bit-exact (or against a frozen, previously-measured
tolerance) by dedicated parity suites (`pipelines/tests/pyc_parity/`,
`AINR/tests/test_python_c_prepost_parity.py`, and friends) that hash the sources they judged into
their result JSON. `docs/c_python_function_map.md` plus
`pipelines/tests/test_c_python_map.py` keep the C/Python symbol mapping and function counts in
sync — update the map in the same change as adding/removing a C function with a Python twin.

**Documentation has a single index and an offline HTML site.** `docs/README.md` is the current
source-of-truth index (source/tests win over docs if they disagree); `docs/html/` is a per-module
offline doc site (overview, API, block diagram, signal swimlane, I/O and state tables, latency)
kept in sync by the `algdoc` skill after an algorithm change. `docs/development.md` covers the
submodule/dependency workflow in more depth than the section above.

## Working across this repo's tree

Because AEC/NR/audiolib are developed in sibling directories (`../AEC`, `../NR`, `../audiolib`)
rather than inside this repo, a change that spans a model library and its conventional-algorithm
counterpart touches two separate git repositories. Commit and push inside the dependency's own
repo first, then bump its pin (`lib/aec.pin`'s `AEC_COMMIT`, etc.) in the same Audio_ALG commit —
never land a pin bump without the commit it points to already existing upstream.
