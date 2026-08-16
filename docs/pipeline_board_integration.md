# Board integration: `libaudio_pipeline.a`

## Scope and prerequisites

This document is the firmware-side contract for `libaudio_pipeline.a`: pool
sizing, the query -> init -> process -> destroy sequence, the memory
descriptor, the board build, and the on-target verification checklist.

It assumes [`../pipelines/README.md`](../pipelines/README.md) has been read
first — that file owns the algorithm description, the parameter/grid
alignment, the measured memory budget ("Memory Budget"), the byte-parity
statement ("Verification"), the heap-vs-pool overview ("Two Versions"), and
the tunable-parameter tables. References below of the form
`see "X" in ../pipelines/README.md` point back there.

Every relative path in this file is written from `Audio_ALG/docs/`; every
`make` invocation shown is run from `Audio_ALG/pipelines/`.

## Composite static APIs and pool sizing

The pipeline uses exactly THREE composite static APIs — there are no
per-submodule `_get_mem_size()` entry points; each library slices its own
internals (HPF, PBFDKF ×2, delay estimator, RES/post context, internal FFTs
for AEC; MCRA + SPP for NR) inside its single pool segment:

```c
size_t aec_sz = aec_get_mem_size(&aec_cfg);          /* lib/aec           */
size_t nr_sz  = mmse_lsa_get_mem_size(&nr_cfg);      /* lib/nr            */
size_t fft_sz = fft_get_mem_size(fft_size);          /* audio_common (OLA) */
/* + pipeline buffers; every segment 16-byte aligned (ALIGN16)            */

Aec*             aec = aec_init(mem_aec, aec_sz, &aec_cfg);
MmseLsaDenoiser* nr  = mmse_lsa_init(mem_nr, nr_sz, &nr_cfg);
FftHandle*       fft = fft_init(mem_fft, fft_sz, fft_size);
/* destroy() on static instances frees no pool memory (runtime is_static)
 * and, since P0001, releases no backend-owned handle OUTSIDE the pool
 * either: on BOTH backends every one of aec_destroy/mmse_lsa_destroy/
 * fft_destroy above is a genuine no-op for a pool-resident instance --
 * nothing lives outside mem_aec/mem_nr/mem_fft to leak or double-free, so
 * each is safe to call any number of times (including never, if the caller
 * is about to free/reuse the whole pool anyway). */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the reported total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. The pool base MUST be 16-byte
aligned (both libraries assert this).

## Sequence

```
1. query    AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
            cfg.aec_preset/nr_mode/aec_only/enable_cng/legacy_amin = ...;
            AudioPipelineMemReq req;
            audio_pipeline_get_mem_requirements(&cfg, &req);   // -> req.bytes/alignment/...
            // Query THIS SAME `req`, fresh, immediately before every
            // init_ex call below — never cache it (or just its `bytes`)
            // across a build/backend/config change and replay it later.
            // See "Warnings" below.

2. allocate void* pool = platform_alloc(req.bytes, req.alignment);
            // req.alignment is always 16 today; pool need NOT be zeroed —
            // see "Dirty-pool contract" below. Pool must stay STABLE and
            // EXCLUSIVE (nothing else reads/writes it, not shared with any
            // other instance) for the entire lifetime of the handle below.

3. init     AudioPipeline* p = audio_pipeline_init_ex(pool, req.bytes, &cfg, &req);
            // Passing `req` back in as `expected` is what makes this call
            // reject a STALE pool/descriptor instead of silently carving
            // into one sized/shaped for a different build — see
            // "Descriptor semantics" below for the seven-condition check
            // this performs. NULL on: any expected-descriptor mismatch
            // (descriptor_version / layout_version / backend_id /
            // build_flags_hash / alignment / bytes), a misaligned/undersized
            // pool, an invalid cfg, or a sub-module init/grid-agreement
            // failure (stderr has detail, UNLESS this library was built with
            // NO_STDIO=1 — see "NO_STDIO=1" below). `audio_pipeline_init(pool,
            // req.bytes, &cfg)` remains available and is EXACTLY
            // `audio_pipeline_init_ex(pool, req.bytes, &cfg, NULL)` — it
            // skips the descriptor check entirely (nothing to compare
            // against); prefer `_ex` whenever a descriptor was already
            // queried, which the board flow above always has at hand.

4. process  float mic[hop], ref[hop], out[hop];   // hop = audio_pipeline_hop_size(p)
            while (have_audio()) {
                read_hop(mic, ref, hop);
                audio_pipeline_process(p, mic, ref, out);
                write_hop(out, hop);
            }

5. reset?   audio_pipeline_reset(p);   // optional: echo-path change, stream switch
            // re-zeros pipeline/AEC/NR state in place; no re-validation, no
            // pool re-touch beyond that.

6. destroy  audio_pipeline_destroy(p);
            // NR -> pipeline FFT -> AEC, reverse of the init carve order.
            // NULL-safe; idempotent for THIS pool-resident instance (every
            // sub-destroy is already a genuine no-op on the pool path, on
            // BOTH backends — see "Two Versions" in
            // ../pipelines/README.md). Call it exactly once if `p` came
            // from audio_pipeline_create() instead
            // (ordinary free() semantics — see "Warnings" below).

7. release  platform_free(pool);   // only after step 6 — the pool is dead once
            //                        audio_pipeline_init_ex/destroy have run on it.
```

## Warnings

- **`audio_pipeline_create()` ALWAYS uses the heap** (`posix_memalign`
  under the hood) — it is the desktop/prototyping convenience path (see
  "Two Versions" in [`../pipelines/README.md`](../pipelines/README.md)),
  never the board path. **A board build must never call it.** The pool
  sequence above
  (`audio_pipeline_get_mem_requirements` + `audio_pipeline_init_ex` +
  `audio_pipeline_destroy`) is the only sequence that touches zero heap.
- **There is no `USE_EXT_MEM` macro.** Both the heap path and the pool
  path are always compiled into `libaudio_pipeline.a` — the *entry point
  you call* is the only switch, decided at runtime by which functions your
  code calls, not by any compile-time flag. See "`USE_EXT_MEM` — not a
  thing here" below; do not look for one to set, there is nothing to
  accidentally leave unset either.
- **Re-query the descriptor after ANY build, backend, or config change** —
  a firmware rebuild, a `BACKEND=kiss` ↔ `BACKEND=ne10` switch, or a change
  to any `AudioPipelineConfig` field that affects sizing. Never cache a
  bare byte count (or a whole `AudioPipelineMemReq`) across one of those
  changes and replay it into a later build's `audio_pipeline_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds even though V2's
  fixed 32-byte layout (below) makes the struct meaningfully
  serializable now — "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `audio_pipeline_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `audio_pipeline_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to.
- **Mixing `_create()` and pool APIs per submodule is not supported.**
  Don't hand-carve one of AEC/NR/the FFT backend into a pool segment
  yourself while letting `audio_pipeline_create()` heap-allocate another,
  and don't call `aec_create()`/`mmse_lsa_create()`/`fft_create()` directly
  alongside this library's own carve. `audio_pipeline_init()`/`_init_ex()`
  own the ENTIRE pool layout (control block + AEC + FFT + NR + the 8
  pipeline scratch buffers) as one unit — there is no supported way to substitute
  a heap-obtained sub-module handle into a pool-resident `AudioPipeline`,
  or vice versa.

## Descriptor semantics (`AudioPipelineMemReq`)

Descriptor V2: every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable, `_Static_assert`-enforced **32-byte** layout — see
"Serializing the descriptor" below for what that does (and does not) buy
you. This is a **BREAKING** change from the original descriptor shape
(`{size_t bytes; size_t alignment; uint32_t layout_version; const char*
backend; uint32_t build_flags_hash;}`); every caller in this repo has been
updated, there is no compatibility shim.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `AUDIO_PIPELINE_DESCRIPTOR_VERSION` (currently `2`). Bumped only when `AudioPipelineMemReq`'s field set/order/width changes, independent of `layout_version` below (which tracks THIS FILE's carve layout, not the descriptor struct's own shape). `audio_pipeline_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | Bumped whenever `../pipelines/mono_aec_nr_res/audio_pipeline.c`'s OWN carve order/buffer set/sizing formula changes — i.e. whenever a `bytes` figure computed by an older build would misdescribe a newer build's actual carve, or vice versa. Starts at 1. Does **not** need bumping for a change purely inside AEC's/NR's/an FFT backend's own internal `_get_mem_size` layout (each is consumed as one opaque composite blob here — a stale cached `bytes` from an old submodule build is still caught by the undersized-pool rejection at init). |
| `backend_id` | Compile-time FFT backend identity this `audio_pipeline.o` was built with, as a small integer — `AUDIO_PIPELINE_BACKEND_KISS` (1) or `AUDIO_PIPELINE_BACKEND_NE10` (2) (matches `../pipelines/Makefile`'s `BACKEND=`). A process-local rodata pointer cannot be serialized safely, so `backend_id` is compared with a plain integer `==`. The two backends are not byte-identical to each other; a descriptor from one is never valid for the other even at matching `bytes`. `0` is reserved for "unknown backend" and is never present in a descriptor this library actually returns — `audio_pipeline_get_mem_requirements()` rejects an unrecognized backend string outright. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the pipeline's own carve STRUCTURE: the backend identity above, a literal token list naming the 8 scratch buffers in carve order, and the alignment granularity — see `audio_pipeline_build_flags_hash()` in `../pipelines/mono_aec_nr_res/audio_pipeline.c`. **Covers:** a change to this file's own carve order/buffer set/alignment. **Does NOT cover:** `AudioPipelineConfig` preset/tunable VALUES (`aec_preset`, `nr_mode`, `sample_rate`, `aec_only`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; AEC's/NR's/an FFT backend's internal struct layouts (opaque blobs, as above); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, both FFT backends, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0. Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Total pool size to allocate (includes the opaque `AudioPipeline` control block itself, carved at the front — a few hundred bytes — plus AEC + FFT(OLA) + NR + the 8 pipeline scratch buffers, same carve `aec_nr_pipeline_static.c`'s old file-local `pipeline_pool_size()` produced). `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`audio_pipeline_init*()` on a target where the two widths differ. |

`audio_pipeline_init_ex(mem, bytes, cfg, expected)` (see "Sequence" above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment`/`bytes` and reject (`NULL`, with a diagnostic naming the
mismatched field) unless every one still agrees with `expected` — a board
integrator no longer has to hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `audio_pipeline_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

## Serializing the descriptor

`AudioPipelineMemReq` is a fixed-width, 32-byte POD (`_Static_assert`-pinned
in `../pipelines/mono_aec_nr_res/audio_pipeline.h`, sizeof and every field's offset) — it
can be copied byte-for-byte (`memcpy`) to a file, a flash region, or a wire
message, and read back later, even by a different process, even after a
restart. This is deliberately scoped narrower than "fully portable
serialization":

- **Same-endianness only.** The board firmware and this library build run
  on the SAME CPU, so this struct provides no byte-swap helpers and makes
  no attempt to support a big-endian producer read by a little-endian
  consumer (or vice versa). If you ever need cross-endian interchange, add
  your own swap routine at the serialization boundary — it does not exist
  here today.
- **Not a substitute for re-querying.** A persisted/transmitted descriptor
  is still exactly the "stale cache" risk "Warnings" above describes — the
  fixed layout only means the BYTES survive the round trip intact, not that
  they still describe the CURRENT build. Always pass a freshly-queried
  descriptor as `expected` to `audio_pipeline_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

## Dirty-pool contract

`audio_pipeline_init()` does **not** require a zero-filled pool. Every
pipeline-owned scratch buffer (the OLA accumulator, per-bin gain/spectrum
scratch, the aec_out hop copy) is explicitly zeroed at carve time,
and AEC/NR/the FFT backend each zero their own sub-region during their own
`_init()` — so a pool filled with poison bytes inits and processes
identically to a freshly-zeroed one.
`../pipelines/mono_aec_nr_res/tests/test_audio_pipeline.c`'s create-vs-init parity case
exercises exactly this: a `memset(pool, 0xA5,
bytes)`-poisoned pool run through `audio_pipeline_init()` produces
byte-for-byte the same 1000-hop output as `audio_pipeline_create()`'s
(unpoisoned) heap path. There is no need for a caller-side blanket
`memset(pool, 0, bytes)` before `audio_pipeline_init()`.

## `USE_EXT_MEM` — not a thing here

Both the heap path (`audio_pipeline_create`/`audio_pipeline_destroy`) and the
pool path (`audio_pipeline_get_mem_requirements`/`audio_pipeline_init`/
`audio_pipeline_destroy`) are always compiled into `libaudio_pipeline.a` —
which one you use is selected purely by which entry point you call, at
runtime, same as `lib/aec`'s and `lib/nr`'s own `_create` vs. `_get_mem_size`/
`_init` pairs. There is no `-DUSE_EXT_MEM`-style compile-time switch (that
pattern existed historically in `lib/nr` and was removed — see
`lib/nr/c_impl/CHANGELOG.md` `[v1.11.0]`/later entries); do not look for one,
and do not add one.

## `NO_STDIO=1` — building without libc stdio

`../pipelines/mono_aec_nr_res/audio_pipeline.c`'s own diagnostics (init/build-time reject
reasons only — nothing on the per-hop `audio_pipeline_process()` path ever
logs anything) are advisory: every failure this file can hit is ALSO signalled through its
return value (`NULL`/`-1`), so a board image that cannot or will not link
libc's stdio (no console, or a policy that forbids the stdio symbol set
outright) still gets a fully-functional library — it only loses the
human-readable "why" that would otherwise go to `stderr`.

```bash
make BACKEND=ne10 NO_STDIO=1 libaudio_pipeline.a
```

compiles `audio_pipeline.o` with `-DAUDIO_PIPELINE_NO_STDIO`, which turns
every diagnostic in that file into a no-op and drops its `<stdio.h>`
include entirely — the resulting `libaudio_pipeline.a` references none of
`fprintf`/`printf`/`puts`/`fputs`/`stderr`/`__stderrp`.

```bash
make BACKEND=ne10 audit-no-stdio   # PASS/FAIL, non-zero exit on FAIL
```

builds exactly that and asserts it with `nm` over `audio_pipeline.o` itself
(pattern/style follows `audio_common/scripts/audit_alloc_symbols.sh`) — run
it after touching any diagnostic in `../pipelines/mono_aec_nr_res/audio_pipeline.c` to
confirm the gate still holds.

`NO_STDIO` only ever changes `audio_pipeline.o`'s own compile flags (see the
`LIB_CFLAGS` comment in `../pipelines/Makefile`, and its `CFG_SIG`/`OBJ_DIR`
participation so a `NO_STDIO=1` build never shares an object directory
with a default
build). Both CLIs (`aec_nr_pipeline`/`aec_nr_pipeline_static` — host tools
that always do their own WAV-I/O stdio) and `test_audio_pipeline` (prints
its own PASS/FAIL via stdio) keep stdio regardless — neither references
`AUDIO_PIPELINE_NO_STDIO`, so there is nothing to gain (and real CLI/test
diagnostics to lose) by gating them too. `make BACKEND=<b> NO_STDIO=1`
(without a specific target) still builds both CLIs normally, linked against
a `NO_STDIO=1` `libaudio_pipeline.a` — this is a supported combination, it
simply means the library half of that binary stays silent on rejection
while the CLI half keeps its own WAV-path error messages.

A board build linking a `NO_STDIO=1` `libaudio_pipeline.a` must rely
entirely on return values (`NULL`/`-1`, including the field-by-field
`audio_pipeline_init_ex()` rejection reasons documented under "Sequence"
above) — there is no error-callback or status-code-detail mechanism beyond
what each function already returns. The diagnostic strings exist for a
dev/desktop build's console only; do not build board firmware assuming
they will be there.

## Teardown order

`audio_pipeline_destroy()` tears down NR → pipeline FFT (the OLA irfft
instance) → AEC — the reverse of `audio_pipeline_init()`'s carve order (AEC →
FFT → NR → scratch). Every one of those three calls is a genuine no-op for a
pool-resident instance today, on **both** backends (matches the "Two
Versions" section in [`../pipelines/README.md`](../pipelines/README.md) —
NE10's twiddle configs moved fully into the pool under vendored patch
P0001, closing the one case that used to need a real
release); the order is kept anyway as forward-compat insurance — a future
backend/module MAY hold something outside the pool that a destroy call
needs to release, and it is exactly what the heap-convenience path needs
for real (`free()` on the pool `audio_pipeline_create()` allocated).

## Board-side verification checklist

Before trusting a board integration of this library in production, verify
each of the following on-target. Most of these are properties of the
**caller's** allocator/integration code, not of
`../pipelines/mono_aec_nr_res/audio_pipeline.c` itself — this library only checks what it
can observe from inside a single call
(alignment of the pointer it was handed, the `bytes` count it was told,
the `expected` descriptor if one was passed); it has no visibility into the
platform allocator, DMA engine, or power state behind that pointer.

- **16-byte alignment**: the `pool` pointer is aligned exactly as
  `req.alignment` reports (16 today). `audio_pipeline_init_ex()` rejects a
  misaligned `mem` argument, but only if one is actually passed in
  misaligned — the platform allocator itself must honor the alignment it
  was asked for.
- **Exact-bytes accounting**: the block handed to `audio_pipeline_init_ex()`
  really is at least `req.bytes` USABLE bytes — verify the allocator's own
  bookkeeping (e.g. an internal header carved out of the block it hands
  back) doesn't silently shrink the usable region below what was requested.
- **Region non-overlap**: the pool is not aliased with any other
  allocation, DMA buffer, or another `AudioPipeline`'s pool — every
  sub-module pointer inside the pool is a raw pointer into it, not a copy,
  so an overlapping region is silent memory corruption, not a caught error.
- **Exclusive lifetime**: nothing else reads or writes the pool for the
  entire lifetime of the handle, from `audio_pipeline_init_ex()` through
  the matching `audio_pipeline_destroy()` — no other task, ISR, or DMA
  transfer touches it concurrently.
- **Cache coherence / DMA ownership**: if `mic`/`ref`/`out` (or the pool
  itself, on a non-cache-coherent platform) cross a DMA boundary, the
  board's own cache-maintenance operations (clean/invalidate) run at the
  correct points. This library has no notion of cache lines or DMA and
  performs none of its own — `audio_pipeline_process()` assumes `mic`/`ref`
  are already CPU-visible on entry and that `out` is fully written before
  any DMA-out on it begins.
- **Allocator-failure / partial-init rollback**: confirm the caller's own
  rollback path is correct for a `platform_alloc()` failure BEFORE
  `audio_pipeline_init_ex()` is ever called — this library never partially
  commits (every rejection path returns `NULL` before writing into `mem`),
  so there is nothing on ITS side to roll back, but the caller's own
  pool-acquisition failure handling is not something this library can
  verify.
- **Reset / reconfigure / double-destroy / power-resume**: exercise
  `audio_pipeline_reset()` (echo-path change), a full destroy-then-
  re-`init_ex()` on the SAME pool (config change), calling
  `audio_pipeline_destroy()` twice in a row (must be a safe no-op the
  second time — see "Teardown order" above), and a power-suspend/resume
  cycle if the target platform has one (this library has no notion of
  power state itself; confirm the pool's contents survive whatever the
  platform does across suspend, or re-`init_ex()` from scratch on resume
  if they don't).
- **Allocator-hook trace of init→destroy (zero-heap, both backends)**:
  with a runtime allocator-hook/interposer (the style `lib/aec`'s and
  `audio_common`'s own zero-heap tests use), confirm NO `malloc`/`calloc`/
  `realloc`/`free` call happens between `audio_pipeline_init_ex()`
  returning and the matching `audio_pipeline_destroy()` returning, ON YOUR
  ACTUAL TARGET BUILD — KISS and NE10 are both zero-heap end-to-end since
  P0001 (see "Two Versions" in
  [`../pipelines/README.md`](../pipelines/README.md)), but that is a
  property of THIS repo's reference builds, not a substitute for verifying
  your own board's build.
  **Include the logging path in this trace**: on the default (stdio-
  enabled) library build, a rejected `audio_pipeline_init_ex()` call only
  ever touches `stderr` (no heap symbol involved), so it does not itself
  break a zero-heap trace — but if the target forbids the stdio symbol set
  outright (not just heap use), build with `NO_STDIO=1` (above) so even
  that `stderr` reference is compiled out, and re-run the same audit
  against that build (`make BACKEND=<b> audit-no-stdio` is the static,
  `nm`-based version of this same check — see above).

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
three producers** (audio_common, lib/aec, lib/nr) — "which exact checkout
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
publish here (neither the refusal nor the provenance hash) — this working
tree permanently carries those two directories as user-owned content that
is never staged for a release. The three producers get **no such
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
`../pipelines/`, byte-identical to the previous hardcoded paths) — like `DIST_ROOT=`, these
are isolation-test knobs for running scratch-directory builds, not part of
the build's config identity (CFG_SIG).

The `current` symlink is swapped via a rename(2)-atomic helper
(`audio_common/tools/atomic_symlink_swap.c`, built at publish time with the
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
  repo-pinned flags (`-ffp-contract=off`, backend defines, `NO_STDIO`); the
  build errors out and points at `EXTRA_CFLAGS`/`EXTRA_LDFLAGS`, the two
  supported hooks. `EXTRA_CFLAGS` containing `-Ofast`/`-ffast-math`/
  `-ffp-contract=` is likewise rejected by the FP-policy conflict gate.
- **C++ runtime comes from the C++ driver** (`libstdc++` on GNU/Linux gcc,
  `libc++` on macOS/clang) — there is no hardcoded `-lc++` anywhere any
  more, so GNU toolchains link cleanly.

## Compile flags (`EXTRA_CFLAGS`)

`../pipelines/Makefile` passes `EXTRA_CFLAGS` through to the `lib/aec` and
`lib/nr` sub-builds *and* this pipeline's own compile, so one invocation
reaches every `.o`. The fast matched-filter arithmetic and delay-estimator
duty-cycling are built in unconditionally — there are no optional
performance flags at present.

No manual `make clean-libs` is needed when switching `EXTRA_CFLAGS` (or
`BACKEND`/`WERROR`): objects are now flag-keyed — each distinct combination
of `BACKEND`/`CFLAGS`/`EXTRA_CFLAGS`/`WERROR` lands in its own hash-named
object directory (`obj/<backend>-<config-hash>/`, both under
`../pipelines/` and independently in `lib/aec`'s and `lib/nr`'s own
Makefiles), so a flag change
always compiles fresh objects into a fresh directory instead of reusing a
stale `.o`. Two builds with different flags/backends can even coexist or run
concurrently in the same worktree without stomping each other's objects.

## Unified FP-contraction policy

`-ffp-contract=off` is a **unified policy spanning all four repos**
(`audio_common`, `lib/nr`, `lib/aec`, the `../pipelines/` Makefile): every TU
each Makefile compiles — own sources and vendored KISS/NE10 alike — builds
with the flag, appended LAST in the `CFLAGS`/`LIB_CFLAGS` assembly (after
`-DAUDIO_PIPELINE_BACKEND_STR`, `EXTRA_CFLAGS`, `WERROR`) so nothing can
override it. `EXTRA_CFLAGS` (or a
`CFLAGS=` override) containing `-Ofast`/`-ffast-math`/`-ffp-contract=<any>`
is rejected at parse time:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:217: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

`../../audio_common/scripts/audit_fp_contract.sh` is the disassembly-level
proof the flag actually bites (audio_common's and NR's TUs —
`../pipelines/`'s own three TUs, being pure CLI/glue code with no
per-sample math loops of their own, are outside that script's audit list). See
`audio_common/README.md`'s "FP-contraction policy" section for the full
cross-repo writeup.
