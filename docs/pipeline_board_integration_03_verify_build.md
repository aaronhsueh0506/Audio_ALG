> [Index](pipeline_board_integration.md) | [Previous: Warnings, descriptor and teardown contract](pipeline_board_integration_02_contract.md) <!-- split-nav -->
## Board-side verification checklist

Before trusting a board integration in production, verify
each of the following on-target. Most of these are properties of the
**caller's** allocator/integration code, not of the libraries themselves — a library only checks what it
can observe from inside a single call
(alignment of the pointer it was handed, the `bytes` count it was told,
the `expected` descriptor if one was passed); it has no visibility into the
platform allocator, DMA engine, or power state behind that pointer.

- **16-byte alignment**: every pool pointer is aligned exactly as
  the size query reports (16 today). `mcab_init_ex()` and `aec_init()` reject a
  misaligned `mem` argument, but only if one is actually passed in
  misaligned — the platform allocator itself must honor the alignment it
  was asked for.
- **Exact-bytes accounting**: the block handed to each init
  really is at least the queried bytes USABLE — verify the allocator's own
  bookkeeping (e.g. an internal header carved out of the block it hands
  back) doesn't silently shrink the usable region below what was requested.
- **Region non-overlap**: the pools are not aliased with any other
  allocation, DMA buffer, or another instance's pool — every
  sub-module pointer inside a pool is a raw pointer into it, not a copy,
  so an overlapping region is silent memory corruption, not a caught error.
  The NR/synthesis block, the flow's pool, each lane and the estimator (or the `Aec` block) must not overlap each other.
- **Exclusive lifetime**: nothing else reads or writes a pool for the
  entire lifetime of its handle, from `init` through
  the matching `destroy` — no other task, ISR, or DMA
  transfer touches it concurrently.
- **Cache coherence / DMA ownership**: if `mic`/`ref`/`out` (or the pool
  itself, on a non-cache-coherent platform) cross a DMA boundary, the
  board's own cache-maintenance operations (clean/invalidate) run at the
  correct points. These libraries have no notion of cache lines or DMA and
  perform none of their own — the per-hop calls assume `mic`/`ref`
  are already CPU-visible on entry and that `out` is fully written before
  any DMA-out on it begins.
- **Allocator-failure / partial-init rollback**: confirm the caller's own
  rollback path is correct for a `platform_alloc()` failure BEFORE any
  init is ever called, and for an init that fails after another succeeded (the AEC built, the denoiser
  refused: destroy the AEC and free the pool) — each init never partially
  commits (every rejection path returns `NULL` before writing into `mem`),
  so there is nothing on ITS side to roll back, but the caller's own
  pool-acquisition failure handling is not something a library can verify.
- **Reset / reconfigure / double-destroy / power-resume**: exercise
  the paired reset (`aec_reset()`/`mcab_reset()` with `aec_res_reset()` (4ch), `anr_reset()`, `ac_istft_reset()` and any
  neural stage reset — echo-path change), a full destroy-then-
  re-`init` on the SAME pool (config change), calling
  `destroy` twice in a row (must be a safe no-op the
  second time — see ["Teardown order"](pipeline_board_integration_02_contract.md#teardown-order) above), and a power-suspend/resume
  cycle if the target platform has one (these libraries have no notion of
  power state themselves; confirm the pool's contents survive whatever the
  platform does across suspend, or re-`init` from scratch on resume
  if they don't).
- **Allocator-hook trace of init→destroy (zero-heap)**:
  with a runtime allocator-hook/interposer (the style `lib/aec`'s and
  `audiolib`'s own zero-heap tests use), confirm NO `malloc`/`calloc`/
  `realloc`/`free` call happens between the last `init` returning and the
  first `destroy` returning, ON YOUR
  ACTUAL TARGET BUILD — NE10 is zero-heap end-to-end
  (vendored patch P0001; see "Two Versions" in
  [`../pipelines/README_02_build_debug_board.md`](../pipelines/README_02_build_debug_board.md)), but that is a
  property of THIS repo's reference builds, not a substitute for verifying
  your own board's build.
  **Include the logging path in this trace**: on the default (stdio-
  enabled) producer builds, a rejected init only
  ever touches `stderr` (no heap symbol involved), so it does not itself
  break a zero-heap trace — but if the target forbids the stdio symbol set
  outright (not just heap use), build with `NO_STDIO=1` ([above](pipeline_board_integration_02_contract.md#no_stdio1--building-without-libc-stdio)) so even
  that `stderr` reference is compiled out, and re-run the same audit
  against that build (`make BACKEND=<b> audit-no-stdio` is the static,
  `nm`-based version of this same check — see [above](pipeline_board_integration_02_contract.md#no_stdio1--building-without-libc-stdio)).

## Cross-compile / board build

The board deliverable is built with `make publish BACKEND=ne10` and consumed
from the stable `dist/ne10/current/` path — immutable, content-addressed
release dirs where `MANIFEST.txt` is the deterministic build-config record
(flags, tool identities, per-file SHA-256; byte-verified on every
republish, and it deliberately carries no commit/date so the same release
id always has the same MANIFEST bytes) and the append-only `ATTEST/`
directory records provenance.

`ATTEST/` is **one-event-one-file**: exactly one new
`attest-<utc>-<commit>[-dirty]-<seq>.txt` per publish event, including
idempotent republishes of an already-published release. The file is
installed atomically with a write-temp + `link(2)` no-clobber step (the
kernel-level equivalent of `O_CREAT|O_EXCL`) — a same-second name collision
(same UTC second, same commit) regenerates the event id with the next
`<seq>` and retries, so an existing attestation is **never** overwritten.
Each attestation's `event_id` is embedded inside the file, and it records a
**full 40-hex git commit OID** plus a dirty flag for **this repo AND all
three producers** (audiolib, lib/aec, lib/nr) — "which exact checkout
of the whole stack published this" lives here, not in MANIFEST.

`publish` **refuses by default**, checking tracked and untracked state as
two independent dimensions, when this repo or any of the three
producers (a) has uncommitted **tracked** changes, (b) contains **any
untracked, non-ignored file** (gitignored build output is excluded by
design and is never part of this provenance), or (c) isn't a git checkout
at all (commit unknown) — `make publish` FATALs and names exactly the
offending repo(s) *and* dimension. Case (c) is refused
**unconditionally**: no combination of `ALLOW_DIRTY_PUBLISH`/
`ALLOW_UNTRACKED_PUBLISH` admits an identity-less checkout, for this repo
or any producer — without a commit OID neither the tracked diff nor the
untracked enumeration can be named for it, so there is no escape hatch to
offer. `ALLOW_DIRTY_PUBLISH=1` admits tracked changes on an
otherwise-identified checkout — pass it to publish anyway; the deviation
is then recorded in the attestation (`allow_dirty_publish=1`) together
with a `dirty_diff_sha256` per dirty repo (sha256 of that repo's `git diff
--binary HEAD`). `ALLOW_UNTRACKED_PUBLISH=1` separately admits untracked
files, recording `allow_untracked_publish=1` plus an
`untracked_tree_sha256` per repo that had one — sha256 over sorted,
FIXED-FIELD per-file records (the `hash_untracked()` helper in
`../pipelines/Makefile`) in which every variable-length value — path, symlink target,
file content — is itself hashed before being placed in the record, so two
records can never be confused by naive concatenation (a raw "path target"
join is collision-prone: `"a b"->"c"` vs `"a"->"b c"` would otherwise
encode identically). Any `stat`/`readlink`/`shasum` failure, or a path
that is neither a regular file nor a symlink (an embedded git checkout, a
fifo, ...), downgrades that entry to an unhashable record, which is
**always** a hard FATAL naming the path rather than an empty or
best-effort hash (fail-closed) — a publish never proceeds as though it
could account for a source it can't. The two flags are orthogonal —
either, both, or neither may be needed for a given publish.

This repo's own untracked check is **whitelisted by prefix**: any path
starting `AINR/GTCRN/` or `AINR/gtcrn_github/` never counts against a
publish here (neither the refusal nor the provenance hash). The three producers get **no such
whitelist**: an untracked file there is always a violation.

**Attestations are UNSIGNED**: they provide traceability, not authenticity
— anyone with filesystem access could forge one, so do not treat `ATTEST/`
as tamper-proof under an attacker model. `MANIFEST.txt` byte-verification
(performed automatically on every republish) is the actual integrity
check; `ATTEST/` is the provenance log.

`make -n|-q|-t ... publish` (dry-run / question / touch mode) is fully
zero-write — each flag takes a
different path, but none of the three ever create `dist/`, take the
publish lock, stage/attest anything, or change an artifact's mtime. This
also holds for **combined** flags (e.g. `-nt`, `-tq`, `-nqt`): `-t` is
checked **first**, so any invocation that includes it takes the touch
no-op path regardless of what else is set — recursing for `-n` first
would hand a child `make` both flags together, and GNU make then really
applies touch semantics to the child's own targets, which is exactly the
real write this ordering exists to prevent. `-n` **prints** what a real
publish would run (recurses into a print-only child, same as any other
target, when `-t` is not also set); `-q` **answers via its exit status
alone** (question mode's documented "needs updating" reply — `publish` is
phony, so it always would run — with no output and no recursion); `-t`
(alone or combined) is an **explicit no-op**: one note printed to stdout
and exit 0, with no recursion. `OBJ_ROOT=`/`BIN_ROOT=` relocate this
repo's own keyed obj/bin build trees (default: `obj/`/`bin/` under
`../pipelines/`) — like `DIST_ROOT=`, these
are isolation-test knobs for running scratch-directory builds, not part of
the build's config identity (CFG_SIG).

The `current` symlink is swapped via a rename(2)-atomic helper
(`audiolib/common/tools/atomic_symlink_swap.c`, built at publish time with the
host compiler `HOSTCC`), and `publish` takes its per-backend lock BEFORE
building anything, so concurrent publishes serialize fully. When
cross-compiling, pass
the **full toolchain, not just CC** — BACKEND=ne10 compiles one C++ TU
(NE10's generic-radix kernel), so a partial override would mix host-built
C++ objects into a cross build and then try to link ARM objects with the
host driver:

```bash
make publish BACKEND=ne10 \
     CC=aarch64-linux-gnu-gcc \
     CXX=aarch64-linux-gnu-g++ \
     AR=aarch64-linux-gnu-ar \
     RANLIB=aarch64-linux-gnu-ranlib \
     EXTRA_CFLAGS='-mcpu=cortex-a53'      # or -mcpu=cortex-a73
```

Guard rails (all four repos):

- **CC/CXX target-coherence check**: every BACKEND=ne10 build compares
  `$(CC) -dumpmachine` against `$(CXX) -dumpmachine` and hard-fails on
  mismatch (exactly the partial-toolchain mistake above). `TOOLCHAIN_CHECK=0`
  skips it (participates in the config signature).
- **`CFLAGS=`/`CXXFLAGS=`/`LDFLAGS=`/`CPPFLAGS=`/`FP_POLICY=` cannot be
  overridden on the make command line** — doing so would silently drop the
  repo-pinned flags (`-ffp-contract=off`, `-fno-math-errno`, backend defines, `NO_STDIO`); the
  build errors out and points at `EXTRA_CFLAGS`/`EXTRA_LDFLAGS`, the two
  supported hooks. `EXTRA_CFLAGS` containing `-Ofast`/`-ffast-math`/
  `-ffp-contract=` is likewise rejected by the FP-policy conflict gate.
- **C++ runtime comes from the C++ driver** (`libstdc++` on GNU/Linux gcc,
  `libc++` on macOS/clang) — there is no hardcoded `-lc++`, so GNU
  toolchains link cleanly.

**Board sample build notes.** The board samples live outside this repository and are built from the same
library sources. Their build lists drop `fix_gain.c` and the `third_party/gain` include path (the
samples apply their input gain with a small `InputGain` helper over `sk_scale_f32()` and `sk_clip_f32()`).
A DeepFilterNet2 lookahead pair other than 1/1 is passed as `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` to
**every** unit that includes the DFN2 headers (the `AINR/DeepFilterNet2` sources, `dfn_res_stage.c`,
`dfn_rate_bridge.c` and the sample's own units); units built for different pairs fail to link, and the
checkpoint must be the one trained for that pair. `alg_ai3_dfn.c` and `uvc_ai_nr_stream.c` hard-code mask and
deep-filter lookahead 1 with a static assert, so they build only with the default pair.

## Compile flags (`EXTRA_CFLAGS`)

`../pipelines/Makefile` passes `EXTRA_CFLAGS` through to the `lib/aec` and
`lib/nr` sub-builds *and* this pipeline's own compile, so one invocation
reaches every `.o`. There are no optional performance flags.

No manual `make clean-libs` is needed when switching `EXTRA_CFLAGS` (or
`BACKEND`/`WERROR`): objects are flag-keyed — each distinct combination
of `BACKEND`/`CFLAGS`/`EXTRA_CFLAGS`/`WERROR` lands in its own hash-named
object directory (`obj/<backend>-<config-hash>/`, both under
`../pipelines/` and independently in `lib/aec`'s and `lib/nr`'s own
Makefiles), so a flag change
always compiles fresh objects into a fresh directory instead of reusing a
stale `.o`. Two builds with different flags/backends can even coexist or run
concurrently in the same worktree without stomping each other's objects.

## Unified FP-contraction policy

`-ffp-contract=off -fno-math-errno` is a **unified policy spanning all four repos**
(`audiolib`, `lib/nr`, `lib/aec`, the `../pipelines/` Makefile): every TU
each Makefile compiles — own sources and vendored NE10 alike — builds
with both flags. The second flag guarantees that guarded AArch64 `sqrtf`
lowers to `FSQRT` without a cold errno fallback; production code does not
read math `errno`.
They are appended LAST in the `CFLAGS` assembly (after
`-DAUDIO_PIPELINE_BACKEND_STR`, `EXTRA_CFLAGS`, `WERROR`) so nothing can
override it. `EXTRA_CFLAGS` (or a
`CFLAGS=` override) containing `-Ofast`/`-ffast-math`/`-ffp-contract=<any>`
is rejected at parse time:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:387: *** FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS.  Stop.
```

`../../audiolib/common/scripts/audit_fp_contract.sh` is the disassembly-level
proof the flag actually bites (audiolib's and NR's TUs —
`../pipelines/`'s own TUs, apart from the ones `pipelines/scripts/audit_fp_contract.sh` lists, are CLI/glue code with no
per-sample math loops of their own). See
`audiolib/common/README.md`'s "FP-contraction policy" section for the full
cross-repo writeup.
> [Index](pipeline_board_integration.md) | [Previous: Warnings, descriptor and teardown contract](pipeline_board_integration_02_contract.md) <!-- split-nav -->
