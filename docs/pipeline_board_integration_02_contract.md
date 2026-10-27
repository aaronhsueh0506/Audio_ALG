> [Index](pipeline_board_integration.md) | [Previous: Sequence and pool sizing](pipeline_board_integration_01_sequence.md) | [Next: Verification checklist and board build](pipeline_board_integration_03_verify_build.md) <!-- split-nav -->
## Warnings

- **`mcab_create()` ALWAYS uses the heap** (`posix_memalign` under the hood, for mcab's own pool
  only) — it is the desktop/prototyping convenience path (see
  "Two Versions" in [`../pipelines/README_02_build_debug_board.md`](../pipelines/README_02_build_debug_board.md)),
  never the board path. **A board build must never call it** (nor `aec_create()`/`anr_create()`).
  The pool sequence ([above](pipeline_board_integration_01_sequence.md#static-apis-and-pool-sizing), `*_get_mem_requirements`/`*_mem_size` + `*_init_ex`/`*_init` +
  `*_destroy`) is the only sequence that touches zero heap.
- **Both the heap path and the pool path are always compiled in.** The entry point you call is the
  only switch, decided at runtime; there is no compile-time flag to set.
- **Re-query every size and descriptor after ANY build, backend, or config change** —
  a firmware rebuild, or a change to any config field that affects sizing. Never cache a
  bare byte count (or a whole `McabMemReq`) across one of those
  changes and replay it into a later build's `mcab_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds with the fixed
  32-byte layout (below): "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `mcab_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `mcab_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to. The mono objects and the example programs have no
  descriptor, so re-querying the size on every build is the whole rule there.
- **The flow does not own its lanes.** `Mcab` drives the four `Aec` lanes and the shared `DelayAec`
  the application created; the lanes' `AecConfig` must carry the flow's contract
  (`mcab_apply_lane_contract()` / `mcab_lane_config()`: `EXTERNAL_ALIGNED`, `enable_res = 0`,
  `return_res_context = 1`, the context level that matches `enable_post`) or `mcab_init_ex()`
  returns `NULL`. Hand `mcab_init_ex()` only freshly initialised or reset objects: it cannot see that
  a lane or the estimator has already run, and a used estimator would give the new flow a stale
  delay. Every object must outlive the `Mcab` handle. The RES (`AecRes`), NR, synthesis and the NN
  stages are likewise separate objects with their own pools.
- **Failure recovery is yours.** When the flow, the denoiser or a stage refuses a call, or the emitted
  hop is not finite, reset the flow (`mcab_reset()` resets the four lanes and the estimator through
  their own reset functions) or `Aec`, the 4-channel `AecRes`, the denoiser, the synthesis (and the comfort-noise
  generator) and any neural stage together
  (`mcab_reset()` / `aec_reset()`, `aec_res_reset()`, `anr_reset()`, `ac_istft_reset()`, `ulcnet_nn_stage_reset()`).
  A `MCAB_DSP_ERROR` has already reset the flow itself. The DFN2 stage is not reset by an AEC delay change.

## Descriptor semantics (`McabMemReq`)

Every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable, `_Static_assert`-enforced **32-byte** layout — see
"Serializing the descriptor" below for what that does (and does not) buy
you. The DFN stage and the rate bridge use the same shape with their
own version counters.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `MCAB_DESCRIPTOR_VERSION` (currently `1`). Bumped only when `McabMemReq`'s field set/order/width changes, independent of `layout_version` below. `mcab_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | `MCAB_LAYOUT_VERSION` (currently `25`). Bumped whenever the pool size, carve order or descriptor content of `../pipelines/4ch_aec_bf_nr_res/mcab.c` changes. A cached `bytes` alone is only a lower bound, so this counter is what refuses a descriptor persisted by another build. Versions 21 to 24 belonged to the predecessor layers and are never reused. |
| `backend_id` | Compile-time FFT backend identity this build was made with, as a small integer — `MCAB_BACKEND_NE10` (2, `mcab.h`), the only FFT backend (matches the `BACKEND=` of `../pipelines/4ch_aec_bf_nr_res/Makefile`). A process-local rodata pointer cannot be serialized safely, so `backend_id` is compared with a plain integer `==`. A descriptor carrying any other value is never valid, even at matching `bytes`. `1` is never returned. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the carve STRUCTURE of mcab's pool: the backend identity above, a literal carve-order token list, the alignment granularity, and the pointer-width token — see `mcab_build_flags_hash()` in `../pipelines/4ch_aec_bf_nr_res/mcab.c`. **Covers:** a change to mcab's own carve order/buffer set/alignment. **Does NOT cover:** config VALUES (`sample_rate`, `enable_post`, `delay_mode`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `mcab_get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; the lanes', the estimator's and the beamformer helper's internal struct layouts (separate objects); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, the NE10 FFT backend, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0 and validated (`mcab_init_ex()` refuses a nonzero value). Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Size of mcab's OWN pool to allocate (the `Mcab` control block carved at the front, then the mic lane, the aligned far, the fuse buffers when `enable_post` is set, and the alignment ring in MATCHED/FIXED; 73,504 B for 16 kHz/256 MATCHED with `enable_post = 1`). It does NOT include the four lanes, the shared estimator, the beamformer helper, the RES, NR, synthesis or any neural stage. `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`mcab_init*()` on a target where the two widths differ. |

`mcab_init_ex(mem, bytes, cfg, libs, expected)` (see ["Sequence"](pipeline_board_integration_01_sequence.md#sequence-4-channel-the-mcab-flow-plus-the-libraries-it-feeds) above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment` and reject (`NULL`) unless every one still agrees with `expected`, `reserved` is 0 and `expected->bytes` is not below the current `bytes`, so a board
integrator does not hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `mcab_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

## Serializing the descriptor

`McabMemReq` is a fixed-width, 32-byte POD (`_Static_assert`-pinned
in `../pipelines/4ch_aec_bf_nr_res/mcab.h`, sizeof and every field's offset) — it
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
  descriptor as `expected` to `mcab_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

## Dirty-pool contract

None of the init functions requires a zero-filled pool. `mcab_init_ex()` clears the whole block it is
given, and AEC/NR/the FFT backend each zero
their own sub-region during their own `_init()` — so a pool filled with poison bytes inits and
processes identically to a freshly-zeroed one.
`../pipelines/4ch_aec_bf_nr_res/tests/test_mcab.c` (`mcab`, `0xA5`-poisoned caller pool and heap construction) and
`../pipelines/mono_aec_nr_res/tests/test_mono_chain.c` (mono, pool carved from a `0xA5`-poisoned
block against the heap flavor) exercise exactly this: byte-for-byte the same output. There is no
need for a caller-side blanket `memset(pool, 0, bytes)` before `init`.

## `NO_STDIO=1` — building without libc stdio

`mcab` prints nothing at all on any path (no stdio symbol is linked), and the example
header `mcab_sample_post.h` uses none. lib/aec, lib/nr and audiolib compile their init-time diagnostics out under
`NO_STDIO=1`; the DFN2 stage and rate bridge archive is audited for stdio symbols. Every failure is
signalled through the return value (`NULL`/`-1`/`0` size), so a board image that cannot or will not
link libc's stdio (no console, or a policy that forbids the stdio symbol set
outright) loses only the human-readable "why" the producers would otherwise write to `stderr`.

```bash
make BACKEND=ne10 NO_STDIO=1 libaudio_pipeline_dfn.a
make BACKEND=ne10 audit-no-stdio   # PASS/FAIL, non-zero exit on FAIL
make BACKEND=ne10 audit-no-stdio-stack
```

`audit-no-stdio` (in `pipelines/`) builds the DFN2 stage archive with `NO_STDIO=1` and asserts with
`nm` that it holds the stage, bridge and DFN2 pre/post members and none of
`fprintf`/`printf`/`puts`/`fputs`/`stderr`/`__stderrp` (pattern/style follows
`audiolib/common/scripts/audit_alloc_symbols.sh`); the same target in
`4ch_aec_bf_nr_res/` (run with `NO_STDIO=1`) scans the 4-channel build: `libaudio_pipeline_mcab.a`, the DFN archive, the producers (including the spatial archives) and the sample objects `mcab_sample_flow.o` and `mcab_sample_beam.o`. `audit-no-stdio-stack` also builds
AEC, NR and audiolib with `NO_STDIO=1`, scans each archive and links
`mono_aec_nr_res/tests/test_no_stdio_stack.c` (query -> init -> one hop -> reset -> destroy over
`Aec`, the denoiser and the synthesis, chained as the examples do) to prove the linked image has no stdio symbol. Run them after touching
any diagnostic.

`NO_STDIO` participates in `../pipelines/Makefile`'s `CFG_SIG`/`OBJ_DIR` so a `NO_STDIO=1` build
never shares an object directory with a default build. The example CLIs
(`aec_nr_pipeline`/`aec_nr_pipeline_static` — host tools that always do their own WAV-I/O stdio)
and the tests keep stdio regardless. `make BACKEND=<b> NO_STDIO=1` (without a specific target) still
builds them normally, linked against `NO_STDIO=1` producers — a supported combination: the library
half stays silent on rejection while the CLI half keeps its own WAV-path error messages.

A board build linking `NO_STDIO=1` producers must rely entirely on return values
(`NULL`/`-1`, `0` for a size query) — there is no error-callback or status-code-detail mechanism
beyond what each function already returns.

## Teardown order

For the 4-channel flow destroy in reverse order of construction: the sample helper
(`mcab_sample_beam_destroy()`: BF -> SRP-PHAT, BF borrows the steering table so it goes first), then
`mcab_destroy()`, then the lanes (`aec_destroy()` x4; the estimator needs none). `mcab_destroy()` touches no library object; on a caller pool it is idempotent, on a `mcab_create()` handle it frees the block and is called once. For the mono flow destroy the denoiser and then the
`Aec`. None of these calls frees a caller-owned pool, and they are repeatable (matches the "Two
Versions" section in [`../pipelines/README_02_build_debug_board.md`](../pipelines/README_02_build_debug_board.md) — NE10's twiddle configs live
in the pool under vendored patch P0001); the order is kept anyway — a future backend/module may hold
something outside the pool that a destroy call needs to release, and it is exactly what the
heap-convenience path needs for real (`free()` on the pool `mcab_create()`
allocated).
> [Index](pipeline_board_integration.md) | [Previous: Sequence and pool sizing](pipeline_board_integration_01_sequence.md) | [Next: Verification checklist and board build](pipeline_board_integration_03_verify_build.md) <!-- split-nav -->
