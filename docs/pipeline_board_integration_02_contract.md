> [Index](pipeline_board_integration.md) | [Previous: Sequence and pool sizing](pipeline_board_integration_01_sequence.md) | [Next: Verification checklist and board build](pipeline_board_integration_03_verify_build.md) <!-- split-nav -->
## Warnings

- **`audio_pipeline_4ch_create()` ALWAYS uses the heap** (`posix_memalign`
  under the hood) — it is the desktop/prototyping convenience path (see
  "Two Versions" in [`../pipelines/README_02_build_debug_board.md`](../pipelines/README_02_build_debug_board.md)),
  never the board path. **A board build must never call it** (nor `aec_create()`/`anr_create()`).
  The pool sequence ([above](pipeline_board_integration_01_sequence.md#static-apis-and-pool-sizing), `*_get_mem_requirements`/`*_mem_size` + `*_init_ex`/`*_init` +
  `*_destroy`) is the only sequence that touches zero heap.
- **Both the heap path and the pool path are always compiled in.** The entry point you call is the
  only switch, decided at runtime; there is no compile-time flag to set.
- **Re-query every size and descriptor after ANY build, backend, or config change** —
  a firmware rebuild, or a change to any config field that affects sizing. Never cache a
  bare byte count (or a whole `AudioPipeline4ChMemReq`) across one of those
  changes and replay it into a later build's `audio_pipeline_4ch_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds with the fixed
  32-byte layout (below): "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `audio_pipeline_4ch_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `audio_pipeline_4ch_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to. The mono objects and the example programs have no
  descriptor, so re-querying the size on every build is the whole rule there.
- **Mixing `_create()` and pool APIs per submodule is not supported.**
  Don't hand-carve one of AEC/SRP/BF into a pool segment yourself while
  letting `audio_pipeline_4ch_create()` heap-allocate another. `audio_pipeline_4ch_init()`/`_init_ex()`
  own the ENTIRE unit pool layout (control block + core + SRP + BF + scratch) as one unit — there is
  no supported way to substitute a heap-obtained sub-module handle into a pool-resident
  `AudioPipeline4Ch`, or vice versa. The RES (`AecRes`), NR, synthesis and the NN stages are separate objects
  with their own pools.
- **Failure recovery is yours.** When the unit, the denoiser or a stage refuses a call, or the emitted
  hop is not finite, reset the unit (or `Aec`), the 4-channel `AecRes`, the denoiser, the synthesis (and the comfort-noise
  generator) and any neural stage together
  (`audio_pipeline_4ch_reset()` / `aec_reset()`, `aec_res_reset()`, `anr_reset()`, `ac_istft_reset()`, `ulcnet_nn_stage_reset()`).
  The DFN2 stage is not reset by an AEC delay change.

## Descriptor semantics (`AudioPipeline4ChMemReq`)

Every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable, `_Static_assert`-enforced **32-byte** layout — see
"Serializing the descriptor" below for what that does (and does not) buy
you. The core `FourAecNrResMemReq`, the DFN stage and the rate bridge use the same shape with their
own version counters.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `AUDIO_PIPELINE_4CH_DESCRIPTOR_VERSION` (currently `1`). Bumped only when `AudioPipeline4ChMemReq`'s field set/order/width changes, independent of `layout_version` below. `audio_pipeline_4ch_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` (currently `21`). Bumped whenever the pool size, carve order or descriptor content of `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` changes — including a control-block-only change of the composed core. A cached `bytes` alone is only a lower bound, so this counter is what refuses a descriptor persisted by another build. |
| `backend_id` | Compile-time FFT backend identity this build was made with, as a small integer — the core's `backend_id`, `FOUR_AEC_NR_RES_BACKEND_NE10` (2, `4aec_nr_res.h`), the only FFT backend (matches the `BACKEND=` of `../pipelines/4ch_aec_bf_nr_res/Makefile`). A process-local rodata pointer cannot be serialized safely, so `backend_id` is compared with a plain integer `==`. A descriptor carrying any other value is never valid, even at matching `bytes`. `0` is reserved for "unknown backend" and is never present in a descriptor this library actually returns. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the unit's carve STRUCTURE: the backend identity above, a literal carve-order token list, the alignment granularity, and the core's own hash — see `audio_pipeline_4ch_build_flags_hash()` in `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c`. **Covers:** a change to the unit's own carve order/buffer set/alignment, and (through the core's hash) a core-layer layout bump. **Does NOT cover:** config VALUES (`sample_rate`, `num_angles`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; AEC's/SRP's/BF's/an FFT backend's internal struct layouts (opaque blobs); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, the NE10 FFT backend, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0. Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Total pool size to allocate (includes the opaque `AudioPipeline4Ch` control block itself, carved at the front, plus the core, SRP-PHAT, BF, the unit's scratch buffers; it does NOT include the RES, NR, synthesis or any neural stage). `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`audio_pipeline_4ch_init*()` on a target where the two widths differ. |

`audio_pipeline_4ch_init_ex(mem, bytes, cfg, expected)` (see ["Sequence"](pipeline_board_integration_01_sequence.md#sequence-4-channel-the-unit-plus-the-libraries-it-feeds) above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment`/`bytes` and reject (`NULL`) unless every one still agrees with `expected`, so a board
integrator does not hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `audio_pipeline_4ch_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

## Serializing the descriptor

`AudioPipeline4ChMemReq` is a fixed-width, 32-byte POD (`_Static_assert`-pinned
in `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h`, sizeof and every field's offset) — it
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
  descriptor as `expected` to `audio_pipeline_4ch_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

## Dirty-pool contract

None of the init functions requires a zero-filled pool. The unit clears the whole block it is
given, and AEC/NR/the FFT backend each zero
their own sub-region during their own `_init()` — so a pool filled with poison bytes inits and
processes identically to a freshly-zeroed one.
`../pipelines/4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` (unit, `0xA5`-poisoned pool against heap) and
`../pipelines/mono_aec_nr_res/tests/test_mono_chain.c` (mono, pool carved from a `0xA5`-poisoned
block against the heap flavor) exercise exactly this: byte-for-byte the same output. There is no
need for a caller-side blanket `memset(pool, 0, bytes)` before `init`.

## `NO_STDIO=1` — building without libc stdio

The unit and the core print nothing at all on any path (no stdio symbol is linked), and the example
header `se_anr_stage.h` uses none. lib/aec, lib/nr and audiolib compile their init-time diagnostics out under
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
`4ch_aec_bf_nr_res/` (run with `NO_STDIO=1`) scans the 4-channel archive, which includes the unit, the DFN archive and the producers. `audit-no-stdio-stack` also builds
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

`audio_pipeline_4ch_destroy()` tears down BF -> SRP-PHAT -> core — BF borrows the
steering table, so it goes first. For the mono flow destroy the denoiser and then the
`Aec`. None of these calls frees a caller-owned pool, and they are repeatable (matches the "Two
Versions" section in [`../pipelines/README_02_build_debug_board.md`](../pipelines/README_02_build_debug_board.md) — NE10's twiddle configs live
in the pool under vendored patch P0001); the order is kept anyway — a future backend/module may hold
something outside the pool that a destroy call needs to release, and it is exactly what the
heap-convenience path needs for real (`free()` on the pool `audio_pipeline_4ch_create()`
allocated).
> [Index](pipeline_board_integration.md) | [Previous: Sequence and pool sizing](pipeline_board_integration_01_sequence.md) | [Next: Verification checklist and board build](pipeline_board_integration_03_verify_build.md) <!-- split-nav -->
